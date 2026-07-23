import torch
import torch.nn as nn
import torch.nn.functional as F

from ...datasets.data_utils.constants import IMAGE_TOKEN_INDEX
from ...utils.adalava_assigner import budget_quantizing


def _masked_max_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"features must have shape [B, T, C], got {tuple(features.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [B, T], got {tuple(mask.shape)}")
    if features.shape[:2] != mask.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match feature prefix shape {tuple(features.shape[:2])}"
        )
    masked_features = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    pooled = masked_features.max(dim=1).values
    empty_mask = ~mask.any(dim=1)
    if empty_mask.any():
        pooled = pooled.clone()
        pooled[empty_mask] = 0.0
    return pooled


def compute_stage1_bal_loss(
    path_mask_st: torch.Tensor,
    budget_values: torch.Tensor,
    num_prefix_layers: int,
    num_hidden_layers: int,
    bal_epsilon: float,
):
    """计算 stage1 的 budget-path balance loss。"""
    num_suffix_layers = int(num_hidden_layers) - int(num_prefix_layers)
    if num_suffix_layers <= 0:
        raise ValueError(
            f"num_suffix_layers must be > 0, got num_hidden_layers={num_hidden_layers}, "
            f"num_prefix_layers={num_prefix_layers}"
        )
    budget_units, _ = budget_quantizing(
        budget=budget_values,
        num_prefix_layers=num_prefix_layers,
        num_hidden_layers=num_hidden_layers,
    )
    sub_budget = (budget_units / float(num_suffix_layers)).clamp(0.0, 1.0)
    target_sub_budget = sub_budget.mean()
    mean_path_activation = path_mask_st.float().mean(dim=0)
    bal_loss = F.relu((mean_path_activation - target_sub_budget).abs() - float(bal_epsilon)).pow(2).mean()
    return {
        "bal_loss": bal_loss,
        "target_sub_budget": target_sub_budget,
    }


class OrionStage1AuxLossComputer(nn.Module):
    """对齐 simlingo stage1 的 div / bal 两个辅助 loss。"""

    def __init__(self, div_queue_size: int = 32):
        super().__init__()
        self.div_queue_size = int(div_queue_size)
        if self.div_queue_size <= 0:
            raise ValueError(f"div_queue_size must be > 0, got {self.div_queue_size}")
        self.register_buffer("div_scene_queue", None, persistent=False)
        self.register_buffer("div_path_queue", None, persistent=False)
        self.register_buffer("div_queue_ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("div_queue_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def reset_div_memory(self):
        """清空 cross-batch div 的历史引用，恢复到冷启动状态。"""
        self.div_scene_queue = None
        self.div_path_queue = None
        self.div_queue_ptr.zero_()
        self.div_queue_count.zero_()

    def _ensure_div_memory(self, scene_norm: torch.Tensor, path_norm: torch.Tensor):
        scene_dim = int(scene_norm.size(-1))
        path_dim = int(path_norm.size(-1))
        if (
            self.div_scene_queue is not None
            and self.div_path_queue is not None
            and self.div_scene_queue.device == scene_norm.device
            and self.div_path_queue.device == path_norm.device
            and self.div_scene_queue.shape == (self.div_queue_size, scene_dim)
            and self.div_path_queue.shape == (self.div_queue_size, path_dim)
        ):
            return
        # 关键调用点：queue 只缓存 detached summary，避免把历史 token 级特征和梯度图留在显存里。
        self.div_scene_queue = torch.zeros(
            (self.div_queue_size, scene_dim),
            dtype=torch.float32,
            device=scene_norm.device,
        )
        self.div_path_queue = torch.zeros(
            (self.div_queue_size, path_dim),
            dtype=torch.float32,
            device=path_norm.device,
        )
        self.div_queue_ptr.zero_()
        self.div_queue_count.zero_()

    def _build_stage1_div_summary(
        self,
        language_inputs: torch.Tensor,
        language_ids: torch.Tensor,
        language_inputs_mask: torch.Tensor,
        budget_token_features: torch.Tensor,
        path_mask_st: torch.Tensor,
    ):
        language_inputs_mask = language_inputs_mask.to(dtype=torch.bool)
        visual_mask = language_inputs_mask & language_ids.eq(IMAGE_TOKEN_INDEX)
        prompt_mask = language_inputs_mask & language_ids.ge(0)

        prompt_pool = _masked_max_pool(language_inputs, prompt_mask)
        visual_pool = _masked_max_pool(language_inputs, visual_mask)
        budget_pool = budget_token_features.float()
        scene_features = torch.maximum(torch.maximum(prompt_pool.float(), visual_pool.float()), budget_pool)
        scene_norm = F.normalize(scene_features, p=2, dim=-1, eps=1e-6)
        path_norm = F.normalize(path_mask_st.float(), p=2, dim=-1, eps=1e-6)
        return scene_norm, path_norm

    def _compute_cross_batch_div_loss(
        self,
        scene_norm: torch.Tensor,
        path_norm: torch.Tensor,
        div_margin: float,
    ):
        self._ensure_div_memory(scene_norm, path_norm)
        ref_count = int(self.div_queue_count.item())
        ref_tensor = scene_norm.sum() * 0.0
        if ref_count == 0:
            return {
                "div_loss": ref_tensor,
                "scene_similarity_mean": ref_tensor,
                "path_similarity_mean": ref_tensor,
                "div_ref_count": ref_tensor.new_tensor(0.0),
                "div_queue_fill_ratio": ref_tensor,
            }
        # 关键调用点：当前 step 会在 forward 末尾继续写 queue，这里先 clone 一份历史 refs，
        # 避免 matmul 反向时读到被本次 enqueue 原地改写过的 buffer。
        ref_scene = self.div_scene_queue[:ref_count].clone()
        ref_path = self.div_path_queue[:ref_count].clone()
        scene_cross = scene_norm.float() @ ref_scene.transpose(0, 1)
        path_cross = path_norm.float() @ ref_path.transpose(0, 1)
        div_loss = F.relu(path_cross - scene_cross - float(div_margin)).pow(2).mean()
        return {
            "div_loss": div_loss,
            "scene_similarity_mean": scene_cross.mean(),
            "path_similarity_mean": path_cross.mean(),
            "div_ref_count": div_loss.new_tensor(float(ref_count)),
            "div_queue_fill_ratio": div_loss.new_tensor(float(ref_count) / float(self.div_queue_size)),
        }

    def _enqueue_div_summary(self, scene_norm: torch.Tensor, path_norm: torch.Tensor):
        self._ensure_div_memory(scene_norm, path_norm)
        scene_norm = scene_norm.detach().float()
        path_norm = path_norm.detach().float()
        batch_size = int(scene_norm.size(0))
        if batch_size <= 0:
            return
        if batch_size >= self.div_queue_size:
            self.div_scene_queue.copy_(scene_norm[-self.div_queue_size:])
            self.div_path_queue.copy_(path_norm[-self.div_queue_size:])
            self.div_queue_ptr.zero_()
            self.div_queue_count.fill_(self.div_queue_size)
            return
        ptr = int(self.div_queue_ptr.item())
        next_ptr = ptr + batch_size
        if next_ptr <= self.div_queue_size:
            self.div_scene_queue[ptr:next_ptr] = scene_norm
            self.div_path_queue[ptr:next_ptr] = path_norm
        else:
            first_len = self.div_queue_size - ptr
            second_len = batch_size - first_len
            self.div_scene_queue[ptr:] = scene_norm[:first_len]
            self.div_path_queue[ptr:] = path_norm[:first_len]
            self.div_scene_queue[:second_len] = scene_norm[first_len:]
            self.div_path_queue[:second_len] = path_norm[first_len:]
        self.div_queue_ptr.fill_(next_ptr % self.div_queue_size)
        self.div_queue_count.fill_(min(self.div_queue_size, int(self.div_queue_count.item()) + batch_size))

    def compute(
        self,
        language_inputs: torch.Tensor,
        language_ids: torch.Tensor,
        language_inputs_mask: torch.Tensor,
        budget_token_features: torch.Tensor,
        path_mask_st: torch.Tensor,
        path_mask_hard: torch.Tensor,
        budget_values: torch.Tensor,
        num_prefix_layers: int,
        num_hidden_layers: int,
        div_margin: float,
        bal_epsilon: float,
    ):
        scene_norm, path_norm = self._build_stage1_div_summary(
            language_inputs=language_inputs,
            language_ids=language_ids,
            language_inputs_mask=language_inputs_mask,
            budget_token_features=budget_token_features,
            path_mask_st=path_mask_st,
        )
        div_outputs = self._compute_cross_batch_div_loss(
            scene_norm=scene_norm,
            path_norm=path_norm,
            div_margin=div_margin,
        )
        if self.training:
            # 关键调用点：先用旧 queue 算 loss，再把当前 batch 入队，确保不会和自己形成伪对比信号。
            self._enqueue_div_summary(scene_norm=scene_norm, path_norm=path_norm)

        return {
            "div_loss": div_outputs["div_loss"],
            "scene_similarity_mean": div_outputs["scene_similarity_mean"],
            "path_similarity_mean": div_outputs["path_similarity_mean"],
            "div_ref_count": div_outputs["div_ref_count"],
            "div_queue_fill_ratio": div_outputs["div_queue_fill_ratio"],
        }
