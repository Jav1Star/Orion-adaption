import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


ADALAVA_BUDGET_TOKEN_INDEX = -300
ADALAVA_PATH_TOKEN_INDEX = -301
ADALAVA_SCENEAWARE_VISUAL_TOKEN_INDEX = -302
ADALAVA_SCENEAWARE_DET_TOKEN_INDEX = -303
ADALAVA_SCENEAWARE_MAP_TOKEN_INDEX = -304
ADALAVA_SCENEAWARE_TRAJ_TOKEN_INDEX = -305
ADALAVA_BUDGET_NUM_ACTIONS = 10


def masked_gumbel_softmax_topk(
    logits: torch.Tensor,
    k: int,
    tau: float = 1.0,
    hard: bool = True,
    dim: int = -1,
    training: bool = True,
) -> torch.Tensor:
    if training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        logits = (logits + gumbels) / float(tau)

    y_soft = logits.softmax(dim=dim)
    dim_size = int(logits.size(dim))
    k_eff = int(max(0, min(int(k), dim_size)))
    if k_eff == 0:
        return torch.zeros_like(logits)
    topk_indices = y_soft.topk(dim=dim, k=k_eff, largest=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, topk_indices, 1.0)

    if not hard:
        return y_soft
    if training:
        return y_hard - y_soft.detach() + y_soft
    return y_hard


class DeviationTokenEncoder(nn.Module):
    """将偏差序列编码成 1 个 scene-aware token。"""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, deviation_seq: torch.Tensor) -> torch.Tensor:
        if deviation_seq.ndim == 2:
            deviation_seq = deviation_seq.unsqueeze(0)
        if deviation_seq.ndim != 3:
            raise ValueError(f"deviation_seq must have shape [B, L, C] or [L, C], got {tuple(deviation_seq.shape)}")
        if deviation_seq.size(1) <= 0:
            raise ValueError("deviation_seq length must be > 0")
        conv_in = deviation_seq.transpose(1, 2)
        conv_feat = self.conv(conv_in).mean(dim=-1)
        return self.mlp(conv_feat).unsqueeze(1)


def resolve_budget_min(num_prefix_layers: int, num_hidden_layers: int) -> float:
    num_prefix = int(num_prefix_layers)
    num_hidden = int(num_hidden_layers)
    if not (0 < num_prefix < num_hidden):
        raise ValueError(
            f"budget min requires 0 < num_prefix_layers < num_hidden_layers, got {num_prefix} and {num_hidden}"
        )
    return float(max(0.2, float(num_prefix) / float(num_hidden)))


def build_budget_action_table(
    budget_min: float,
    device: torch.device,
    dtype: torch.dtype,
    num_actions: int = ADALAVA_BUDGET_NUM_ACTIONS,
) -> torch.Tensor:
    if budget_min < 0.0 or budget_min > 1.0:
        raise ValueError(f"budget_min must be in [0, 1], got {budget_min}")
    if int(num_actions) < 2:
        raise ValueError(f"num_actions must be >= 2, got {num_actions}")
    return torch.linspace(float(budget_min), 1.0, steps=int(num_actions), device=device, dtype=dtype)


def budget_quantizing(
    budget: torch.Tensor,
    num_prefix_layers: int,
    num_hidden_layers: int,
):
    budget = budget.float()
    if torch.any(budget < 0.0) or torch.any(budget > 1.0):
        raise ValueError(
            f"budget must be in [0, 1], got min={float(budget.min().item()):.6f}, max={float(budget.max().item()):.6f}"
        )
    budget_units = torch.floor(float(num_hidden_layers) * budget) - float(num_prefix_layers)
    budget_units = torch.relu(budget_units)
    budget_units_norm = budget_units / float(int(num_hidden_layers) - int(num_prefix_layers) + 1)
    return budget_units, budget_units_norm


class OrionBudgetAssigner(nn.Module):
    """管理 Orion 的预算决策、scene-aware 输入和 suffix layer 选择。"""

    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_prefix_layers,
        train_stage=None,
        budget_curriculum_start_min=1.0,
        budget_curriculum_warmup_steps=0,
        sample_interval=5,
        path_gumbel_tau=1.0,
        path_gumbel_hard=True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_prefix_layers = int(num_prefix_layers)
        self.num_suffix_layers = self.num_hidden_layers - self.num_prefix_layers
        self.budget_split_layer = self.num_prefix_layers // 2

        if self.num_prefix_layers <= 0 or self.num_prefix_layers >= self.num_hidden_layers:
            raise ValueError(
                f"num_prefix_layers must be in (0, {self.num_hidden_layers}), got {self.num_prefix_layers}"
            )
        self.train_stage = train_stage
        self.stage1_enabled = str(train_stage).strip().lower() == "stage1"
        self.budget_min = resolve_budget_min(
            num_prefix_layers=self.num_prefix_layers,
            num_hidden_layers=self.num_hidden_layers,
        )
        self.budget_num_actions = ADALAVA_BUDGET_NUM_ACTIONS
        self.stage1_budget_curriculum_start_min = float(budget_curriculum_start_min)
        self.stage1_budget_curriculum_warmup_steps = int(budget_curriculum_warmup_steps)
        self.path_gumbel_tau = float(path_gumbel_tau)
        self.path_gumbel_hard = bool(path_gumbel_hard)
        if self.stage1_budget_curriculum_start_min < self.budget_min or self.stage1_budget_curriculum_start_min > 1.0:
            raise ValueError(
                "budget_curriculum_start_min must be in "
                f"[budget_min={self.budget_min:.6f}, 1.0], got {self.stage1_budget_curriculum_start_min}"
            )
        if self.stage1_budget_curriculum_warmup_steps < 0:
            raise ValueError(
                f"budget_curriculum_warmup_steps must be >= 0, got {self.stage1_budget_curriculum_warmup_steps}"
            )

        self.budget_query_embed = nn.Parameter(torch.empty(1, self.hidden_size))
        self.path_query_embed = nn.Parameter(torch.empty(1, self.hidden_size))
        self.budget_scheduler = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.budget_num_actions),
        )
        self.budget_encoder = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.path_scheduler = nn.Sequential(
            nn.Linear(self.hidden_size * 6, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.num_suffix_layers),
        )

        self.sceneaware_enabled = False
        self.sceneaware_token_ids: List[int] = []
        self.det_num_classes = 0
        self.map_num_classes = 0
        self.visual_query_dim = 0
        self.sceneaware_image_prefix_len = 0
        self.sceneaware_state_by_route: Dict[str, Dict[str, Any]] = {}
        self.sceneaware_state_by_slot: Dict[int, Dict[str, Any]] = {}
        self._sceneaware_modules_initialized = False

        self.sceneaware_det_topk = 32
        self.sceneaware_map_topk = 32
        self.sceneaware_traj_t_lap = 0.5
        self.sceneaware_sample_interval = int(sample_interval)
        if self.sceneaware_sample_interval <= 0:
            raise ValueError(f"sample_interval must be > 0, got {self.sceneaware_sample_interval}")
        self.sceneaware_history_max_len = 2 * self.sceneaware_sample_interval + 1
        self.stage1_budget_total_iters = 0
        self.stage1_budget_runtime_iter = 0

        self._reset_parameters()
        self.reset_runtime_state()
        self.reset_sceneaware_history()

    def _reset_parameters(self):
        nn.init.normal_(self.budget_query_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.path_query_embed, mean=0.0, std=0.02)
        for module in [self.budget_scheduler, self.budget_encoder, self.path_scheduler]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.normal_(submodule.weight, mean=0.0, std=0.02)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def reset_runtime_state(self):
        self.budget_query_positions = None
        self.path_query_positions = None
        self.sceneaware_token_positions = None
        self.selected_budget_values = None
        self.runtime_budget_features = None
        self.execution_plan = None
        self.execution_plan_hard = None
        self.path_logits = None
        self.runtime_sceneaware_tokens = None
        self.runtime_language_inputs = None
        self.runtime_language_ids = None
        self.runtime_language_inputs_mask = None
        self.path_mask_st = None
        self.path_mask_hard = None

    def set_stage1_budget_total_iters(self, total_iters: int):
        self.stage1_budget_total_iters = int(total_iters)

    def advance_stage1_budget_runtime_iter(self):
        self.stage1_budget_runtime_iter += 1

    def _compute_stage1_curriculum_min(self) -> float:
        if self.stage1_budget_curriculum_warmup_steps <= 0:
            return float(self.budget_min)
        progress = float(
            max(min(self.stage1_budget_runtime_iter, self.stage1_budget_curriculum_warmup_steps), 0)
        ) / float(self.stage1_budget_curriculum_warmup_steps)
        start = float(self.stage1_budget_curriculum_start_min)
        end = float(self.budget_min)
        # 关键调用点：stage1 从高预算逐步过渡到低预算，先让 path 分支在容易样本上学会工作。
        return float(start + 0.5 * (1.0 - math.cos(math.pi * progress)) * (end - start))

    def init_sceneaware_modules(
        self,
        sceneaware_cfg,
        det_num_classes,
        map_num_classes,
        visual_query_dim,
        image_prefix_token_count,
    ):
        """初始化 scene-aware 编码器，历史缓存由 assigner 独占维护。"""
        self.sceneaware_enabled = bool(sceneaware_cfg.get("sceneaware_enabled", False))
        if not self.sceneaware_enabled:
            self.sceneaware_token_ids = []
            self._sceneaware_modules_initialized = False
            self.reset_sceneaware_history()
            return

        self.det_num_classes = int(det_num_classes)
        self.map_num_classes = int(map_num_classes)
        self.visual_query_dim = int(visual_query_dim)
        self.sceneaware_image_prefix_len = int(image_prefix_token_count)
        self.sceneaware_token_ids = [
            ADALAVA_SCENEAWARE_VISUAL_TOKEN_INDEX,
            ADALAVA_SCENEAWARE_DET_TOKEN_INDEX,
            ADALAVA_SCENEAWARE_MAP_TOKEN_INDEX,
            ADALAVA_SCENEAWARE_TRAJ_TOKEN_INDEX,
        ]
        class_embed_dim = min(64, self.hidden_size // 8)
        self.sceneaware_class_embed_dim = class_embed_dim
        self.det_class_embed = nn.Embedding(self.det_num_classes, class_embed_dim)
        self.map_class_embed = nn.Embedding(self.map_num_classes, class_embed_dim)
        self.visual_encoder = DeviationTokenEncoder(self.visual_query_dim, self.hidden_size)
        self.det_encoder = DeviationTokenEncoder(9 + class_embed_dim, self.hidden_size)
        self.map_encoder = DeviationTokenEncoder(33 + class_embed_dim, self.hidden_size)
        self.traj_encoder = DeviationTokenEncoder(2, self.hidden_size)
        nn.init.normal_(self.det_class_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.map_class_embed.weight, mean=0.0, std=0.02)
        self._sceneaware_modules_initialized = True
        self.reset_sceneaware_history()

    def reset_sceneaware_history(
        self,
        route_keys: Optional[List[str]] = None,
        sample_idxs: Optional[List[int]] = None,
    ):
        """重置 scene-aware 历史；训练态按 slot，推理态按 route。"""
        if route_keys is None and sample_idxs is None:
            self.sceneaware_state_by_route = {}
            self.sceneaware_state_by_slot = {}
            return
        if route_keys is not None:
            for route_key in route_keys:
                self.sceneaware_state_by_route.pop(str(route_key), None)
        if sample_idxs is not None:
            for sample_idx in sample_idxs:
                self.sceneaware_state_by_slot.pop(int(sample_idx), None)

    def _build_empty_sceneaware_state(self, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "last_route_key": None,
            "last_frame_idx": None,
            "history": [],
            "cached_traj_dev": torch.zeros((6, 2), device=device, dtype=dtype),
        }

    def _get_or_init_sceneaware_state(
        self,
        route_key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        key = str(route_key)
        if key not in self.sceneaware_state_by_route:
            self.sceneaware_state_by_route[key] = self._build_empty_sceneaware_state(device=device, dtype=dtype)
        return self.sceneaware_state_by_route[key]

    def _get_or_init_sceneaware_slot_state(
        self,
        sample_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        sample_idx = int(sample_idx)
        if sample_idx not in self.sceneaware_state_by_slot:
            self.sceneaware_state_by_slot[sample_idx] = self._build_empty_sceneaware_state(device=device, dtype=dtype)
        return self.sceneaware_state_by_slot[sample_idx]

    @staticmethod
    def _ensure_batch_size(route_keys: Optional[List[str]], batch_size: int):
        if route_keys is None:
            raise ValueError("route_keys is required when using route-level scene-aware cache")
        if len(route_keys) != batch_size:
            raise ValueError(f"route_keys length ({len(route_keys)}) must match batch size ({batch_size})")

    @staticmethod
    def _ensure_frame_batch_size(frame_idxs: Optional[List[int]], batch_size: int):
        if frame_idxs is None:
            raise ValueError("frame_idxs is required when using scene-aware cache")
        if len(frame_idxs) != batch_size:
            raise ValueError(f"frame_idxs length ({len(frame_idxs)}) must match batch size ({batch_size})")

    @staticmethod
    def _ensure_sample_batch_size(sample_idxs: Optional[List[int]], batch_size: int):
        if sample_idxs is None:
            raise ValueError("sample_idxs is required in training when using slot-level scene-aware cache")
        if len(sample_idxs) != batch_size:
            raise ValueError(f"sample_idxs length ({len(sample_idxs)}) must match batch size ({batch_size})")

    @staticmethod
    def _to_cpu_float_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.detach().to(dtype=torch.float32).cpu()

    @staticmethod
    def _to_cpu_long_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return value.detach().to(dtype=torch.long).cpu()

    def _get_sceneaware_reference_entry(self, state: Dict[str, Any], frame_idx: int) -> Optional[Dict[str, Any]]:
        target_frame_idx = int(frame_idx) - self.sceneaware_sample_interval
        for history_entry in reversed(state["history"]):
            if int(history_entry["frame_idx"]) == target_frame_idx:
                return history_entry
        return None

    def _append_sceneaware_history_entry(self, state: Dict[str, Any], history_entry: Dict[str, Any]):
        state["history"].append(history_entry)
        if len(state["history"]) > self.sceneaware_history_max_len:
            state["history"] = state["history"][-self.sceneaware_history_max_len:]

    def _encode_visual_token(self, current_visual_queries, reference_entry, device, dtype):
        prev_visual_queries = None if reference_entry is None else reference_entry["visual_queries"]
        current_visual_queries = current_visual_queries.to(device=device, dtype=dtype)
        if prev_visual_queries is None:
            visual_dev = torch.zeros_like(current_visual_queries)
        else:
            prev_visual_queries = prev_visual_queries.to(device=device, dtype=dtype)
            if current_visual_queries.shape != prev_visual_queries.shape:
                raise ValueError(
                    "scene-aware visual query shape mismatch: "
                    f"current={tuple(current_visual_queries.shape)} prev={tuple(prev_visual_queries.shape)}"
                )
            visual_dev = current_visual_queries - prev_visual_queries
        return self.visual_encoder(visual_dev)

    def _match_indices_by_class(self, current_centers, current_labels, prev_centers, prev_labels):
        match_rows: List[int] = []
        match_cols: List[int] = []
        if current_centers.size(0) == 0 or prev_centers.size(0) == 0:
            return match_rows, match_cols
        shared_labels = torch.unique(current_labels)
        for label in shared_labels.tolist():
            curr_idx = torch.nonzero(current_labels == label, as_tuple=False).flatten()
            prev_idx = torch.nonzero(prev_labels == label, as_tuple=False).flatten()
            if curr_idx.numel() == 0 or prev_idx.numel() == 0:
                continue
            cost = torch.cdist(current_centers[curr_idx], prev_centers[prev_idx]).detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost)
            for row, col in zip(row_idx.tolist(), col_idx.tolist()):
                match_rows.append(int(curr_idx[row].item()))
                match_cols.append(int(prev_idx[col].item()))
        return match_rows, match_cols

    def _build_det_dev_inputs(self, current_result, reference_entry, device, dtype):
        prev_boxes = None if reference_entry is None else reference_entry["det_boxes"]
        prev_labels = None if reference_entry is None else reference_entry["det_labels"]
        zero_dev = torch.zeros((1, 9), device=device, dtype=dtype)
        zero_label = torch.zeros((1,), device=device, dtype=torch.long)
        if current_result is None or prev_boxes is None or prev_labels is None:
            return zero_dev, zero_label

        current_boxes = current_result["boxes_3d"].tensor.detach().to(device=device, dtype=dtype)
        current_scores = current_result["scores_3d"].detach().to(device=device, dtype=dtype)
        current_labels = current_result["labels_3d"].detach().to(device=device, dtype=torch.long)
        prev_boxes = prev_boxes.to(device=device, dtype=dtype)
        prev_labels = prev_labels.to(device=device, dtype=torch.long)
        if current_boxes.numel() == 0 or prev_boxes.numel() == 0:
            return zero_dev, zero_label

        match_rows, match_cols = self._match_indices_by_class(
            current_centers=current_boxes[:, :2],
            current_labels=current_labels,
            prev_centers=prev_boxes[:, :2],
            prev_labels=prev_labels,
        )
        if len(match_rows) == 0:
            return zero_dev, zero_label

        matched_scores = current_scores[match_rows]
        sorted_idx = torch.argsort(matched_scores, descending=True)[: self.sceneaware_det_topk]
        current_idx = torch.as_tensor(match_rows, device=device, dtype=torch.long)[sorted_idx]
        prev_idx = torch.as_tensor(match_cols, device=device, dtype=torch.long)[sorted_idx]
        det_dev = current_boxes[current_idx] - prev_boxes[prev_idx]
        det_labels = current_labels[current_idx]
        return det_dev, det_labels

    def _encode_det_token(self, current_result, reference_entry, device, dtype):
        det_dev, det_labels = self._build_det_dev_inputs(
            current_result=current_result,
            reference_entry=reference_entry,
            device=device,
            dtype=dtype,
        )
        det_class_embed = self.det_class_embed(det_labels).to(dtype=dtype)
        det_inputs = torch.cat([det_dev, det_class_embed], dim=-1)
        return self.det_encoder(det_inputs.unsqueeze(0))

    def _build_map_dev_inputs(self, current_result, reference_entry, device, dtype):
        prev_map_pts = None if reference_entry is None else reference_entry["map_pts"]
        prev_map_labels = None if reference_entry is None else reference_entry["map_labels"]
        zero_dev = torch.zeros((1, 33), device=device, dtype=dtype)
        zero_label = torch.zeros((1,), device=device, dtype=torch.long)
        if current_result is None or prev_map_pts is None or prev_map_labels is None:
            return zero_dev, zero_label

        current_map_pts = current_result["map_pts_3d"].detach().to(device=device, dtype=dtype)
        current_map_scores = current_result["map_scores_3d"].detach().to(device=device, dtype=dtype)
        current_map_labels = current_result["map_labels_3d"].detach().to(device=device, dtype=torch.long)
        prev_map_pts = prev_map_pts.to(device=device, dtype=dtype)
        prev_map_labels = prev_map_labels.to(device=device, dtype=torch.long)
        if current_map_pts.numel() == 0 or prev_map_pts.numel() == 0:
            return zero_dev, zero_label

        match_rows, match_cols = self._match_indices_by_class(
            current_centers=current_map_pts[..., :2].mean(dim=1),
            current_labels=current_map_labels,
            prev_centers=prev_map_pts[..., :2].mean(dim=1),
            prev_labels=prev_map_labels,
        )
        if len(match_rows) == 0:
            return zero_dev, zero_label

        matched_scores = current_map_scores[match_rows]
        sorted_idx = torch.argsort(matched_scores, descending=True)[: self.sceneaware_map_topk]
        current_idx = torch.as_tensor(match_rows, device=device, dtype=torch.long)[sorted_idx]
        prev_idx = torch.as_tensor(match_cols, device=device, dtype=torch.long)[sorted_idx]
        map_dev = (current_map_pts[current_idx] - prev_map_pts[prev_idx]).flatten(1)
        map_labels = current_map_labels[current_idx]
        return map_dev, map_labels

    def _encode_map_token(self, current_result, reference_entry, device, dtype):
        map_dev, map_labels = self._build_map_dev_inputs(
            current_result=current_result,
            reference_entry=reference_entry,
            device=device,
            dtype=dtype,
        )
        map_class_embed = self.map_class_embed(map_labels).to(dtype=dtype)
        map_inputs = torch.cat([map_dev, map_class_embed], dim=-1)
        return self.map_encoder(map_inputs.unsqueeze(0))

    def _encode_traj_token(self, traj_dev, device, dtype):
        if traj_dev is None:
            traj_dev = torch.zeros((6, 2), device=device, dtype=dtype)
        else:
            traj_dev = traj_dev.to(device=device, dtype=dtype)
        return self.traj_encoder(traj_dev.unsqueeze(0))

    def _prepare_sceneaware_state_before_step(
        self,
        route_key: str,
        frame_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        sample_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.training:
            if sample_idx is None:
                raise ValueError("sample_idx is required when training scene-aware with slot memory")
            state = self._get_or_init_sceneaware_slot_state(sample_idx=sample_idx, device=device, dtype=dtype)
        else:
            state = self._get_or_init_sceneaware_state(route_key=route_key, device=device, dtype=dtype)

        last_route_key = state.get("last_route_key", None)
        last_frame_idx = state.get("last_frame_idx", None)
        if (
            last_route_key is not None
            and (str(last_route_key) != str(route_key) or int(frame_idx) != int(last_frame_idx) + 1)
        ):
            # 关键调用点：slot/route 不连续时立即清历史，避免 scene-aware 串帧。
            state = self._build_empty_sceneaware_state(device=device, dtype=dtype)
            if self.training:
                self.sceneaware_state_by_slot[int(sample_idx)] = state
            else:
                self.sceneaware_state_by_route[str(route_key)] = state
        return state

    def build_sceneaware_inputs_before_llm(
        self,
        current_visual_queries: torch.Tensor,
        current_det_results: List[Optional[Dict[str, Any]]],
        current_map_results: List[Optional[Dict[str, Any]]],
        route_keys: List[str],
        frame_idxs: List[int],
        sample_idxs: Optional[List[int]] = None,
    ):
        """在 LLM 前构造 scene-aware token。"""
        if not self.sceneaware_enabled:
            return
        if not self._sceneaware_modules_initialized:
            raise ValueError("scene-aware modules are not initialized")
        if current_visual_queries.ndim != 3:
            raise ValueError(
                "current_visual_queries must have shape [B, L, C], "
                f"got {tuple(current_visual_queries.shape)}"
            )
        if current_visual_queries.size(-1) != self.visual_query_dim:
            raise ValueError(
                f"current_visual_queries last dim must be {self.visual_query_dim}, "
                f"got {int(current_visual_queries.size(-1))}"
            )
        batch_size = int(current_visual_queries.size(0))
        if len(current_det_results) != batch_size or len(current_map_results) != batch_size:
            raise ValueError("current_det_results/current_map_results batch size mismatch")
        self._ensure_batch_size(route_keys, batch_size)
        self._ensure_frame_batch_size(frame_idxs, batch_size)
        if self.training:
            self._ensure_sample_batch_size(sample_idxs, batch_size)

        self.to(device=current_visual_queries.device, dtype=current_visual_queries.dtype)
        device = current_visual_queries.device
        dtype = current_visual_queries.dtype

        sceneaware_tokens = []
        for batch_idx in range(batch_size):
            state = self._prepare_sceneaware_state_before_step(
                route_key=route_keys[batch_idx],
                frame_idx=frame_idxs[batch_idx],
                device=device,
                dtype=dtype,
                sample_idx=None if sample_idxs is None else sample_idxs[batch_idx],
            )
            reference_entry = self._get_sceneaware_reference_entry(state, frame_idxs[batch_idx])
            visual_token = self._encode_visual_token(
                current_visual_queries=current_visual_queries[batch_idx],
                reference_entry=reference_entry,
                device=device,
                dtype=dtype,
            )
            det_token = self._encode_det_token(
                current_result=current_det_results[batch_idx],
                reference_entry=reference_entry,
                device=device,
                dtype=dtype,
            )
            map_token = self._encode_map_token(
                current_result=current_map_results[batch_idx],
                reference_entry=reference_entry,
                device=device,
                dtype=dtype,
            )
            # 关键调用点：当前轨迹要等 LLM 后才生成，这里只消费最近一次已缓存的 interval 偏差；
            # 如果当前帧对应的 interval 历史还不存在，则明确退化为 0。
            traj_token = self._encode_traj_token(
                traj_dev=None if reference_entry is None else state.get("cached_traj_dev", None),
                device=device,
                dtype=dtype,
            )
            sceneaware_tokens.append(torch.cat([visual_token, det_token, map_token, traj_token], dim=1))
        self.runtime_sceneaware_tokens = torch.cat(sceneaware_tokens, dim=0)

    def get_sceneaware_embeddings(self, sample_idx=None):
        if not self.sceneaware_enabled:
            raise ValueError("scene-aware is disabled")
        if not hasattr(self, "runtime_sceneaware_tokens") or self.runtime_sceneaware_tokens is None:
            raise ValueError("scene-aware runtime tokens are not prepared before LLM")
        if sample_idx is None:
            raise ValueError("sample_idx is required for scene-aware token lookup")
        return self.runtime_sceneaware_tokens[sample_idx:sample_idx + 1]

    def _transform_prev_waypoints_to_curr_frame(self, prev_waypoints, delta_xy_prev_frame, delta_yaw):
        centered = prev_waypoints - delta_xy_prev_frame[:, None, :]
        cos_yaw = torch.cos(delta_yaw)[:, None]
        sin_yaw = torch.sin(delta_yaw)[:, None]
        x_prev = centered[..., 0]
        y_prev = centered[..., 1]
        x_curr = cos_yaw * x_prev + sin_yaw * y_prev
        y_curr = -sin_yaw * x_prev + cos_yaw * y_prev
        return torch.stack([x_curr, y_curr], dim=-1)

    def _time_align_prev_waypoints(self, prev_waypoints_curr_frame, delta_tau):
        batch_size, waypoint_count, _ = prev_waypoints_curr_frame.shape
        base_idx = torch.arange(
            waypoint_count,
            device=prev_waypoints_curr_frame.device,
            dtype=prev_waypoints_curr_frame.dtype,
        )
        j_star = base_idx[None, :] + (delta_tau[:, None] / self.sceneaware_traj_t_lap)
        k = torch.floor(j_star).long()
        alpha = j_star - k.to(j_star.dtype)
        k_next = k + 1
        valid_mask = (k >= 0) & (k_next < waypoint_count)

        k = k.clamp(0, waypoint_count - 1)
        k_next = k_next.clamp(0, waypoint_count - 1)
        k_idx = k.unsqueeze(-1).expand(batch_size, waypoint_count, 2)
        k_next_idx = k_next.unsqueeze(-1).expand(batch_size, waypoint_count, 2)
        wp_k = torch.gather(prev_waypoints_curr_frame, dim=1, index=k_idx)
        wp_k_next = torch.gather(prev_waypoints_curr_frame, dim=1, index=k_next_idx)
        aligned = (1.0 - alpha.unsqueeze(-1)) * wp_k + alpha.unsqueeze(-1) * wp_k_next
        return aligned, valid_mask

    @staticmethod
    def _pose_xy_yaw_from_matrix(ego_pose: torch.Tensor):
        ego_xy = ego_pose[:2, 3]
        ego_yaw = torch.atan2(ego_pose[1, 0], ego_pose[0, 0])
        return ego_xy, ego_yaw

    def _build_traj_deviation(
        self,
        current_future_traj: torch.Tensor,
        current_ego_pose: torch.Tensor,
        current_timestamp: torch.Tensor,
        prev_future_traj: Optional[torch.Tensor],
        prev_ego_pose: Optional[torch.Tensor],
        prev_timestamp: Optional[torch.Tensor],
    ) -> torch.Tensor:
        traj_dev = torch.zeros_like(current_future_traj)
        if prev_future_traj is None or prev_ego_pose is None or prev_timestamp is None:
            return traj_dev

        current_ego_xy, current_ego_yaw = self._pose_xy_yaw_from_matrix(current_ego_pose)
        prev_ego_xy, prev_ego_yaw = self._pose_xy_yaw_from_matrix(prev_ego_pose)
        dx_global = current_ego_xy[0] - prev_ego_xy[0]
        dy_global = current_ego_xy[1] - prev_ego_xy[1]
        cos_prev = torch.cos(prev_ego_yaw)
        sin_prev = torch.sin(prev_ego_yaw)
        delta_x_prev = cos_prev * dx_global + sin_prev * dy_global
        delta_y_prev = -sin_prev * dx_global + cos_prev * dy_global
        delta_xy_prev_frame = torch.stack([delta_x_prev, delta_y_prev], dim=-1).unsqueeze(0)
        delta_yaw = (current_ego_yaw - prev_ego_yaw).unsqueeze(0)
        delta_tau = (current_timestamp - prev_timestamp).reshape(1)

        prev_traj_curr_frame = self._transform_prev_waypoints_to_curr_frame(
            prev_future_traj.unsqueeze(0),
            delta_xy_prev_frame,
            delta_yaw,
        )
        aligned_prev_traj, valid_mask = self._time_align_prev_waypoints(prev_traj_curr_frame, delta_tau)
        aligned_prev_traj = aligned_prev_traj.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
        traj_delta = current_future_traj - aligned_prev_traj
        traj_dev = torch.where(valid_mask.unsqueeze(-1), traj_delta, torch.zeros_like(traj_delta))
        return traj_dev

    def _extract_det_history_fields(self, bbox_result: Optional[Dict[str, Any]]):
        if bbox_result is None:
            return None, None
        return (
            self._to_cpu_float_tensor(bbox_result["boxes_3d"].tensor),
            self._to_cpu_long_tensor(bbox_result["labels_3d"]),
        )

    def _extract_map_history_fields(self, lane_result: Optional[Dict[str, Any]]):
        if lane_result is None:
            return None, None
        return (
            self._to_cpu_float_tensor(lane_result["map_pts_3d"]),
            self._to_cpu_long_tensor(lane_result["map_labels_3d"]),
        )

    def update_sceneaware_history_after_llm(
        self,
        route_keys: List[str],
        frame_idxs: List[int],
        current_visual_queries: torch.Tensor,
        current_det_results: List[Optional[Dict[str, Any]]],
        current_map_results: List[Optional[Dict[str, Any]]],
        current_future_trajs: List[torch.Tensor],
        current_ego_poses: torch.Tensor,
        current_timestamps: torch.Tensor,
        sample_idxs: Optional[List[int]] = None,
    ):
        """在 LLM 后更新缓存，下一帧 prefix 决策只读取已缓存状态。"""
        if not self.sceneaware_enabled:
            return
        if current_visual_queries.ndim != 3:
            raise ValueError(
                "current_visual_queries must have shape [B, L, C], "
                f"got {tuple(current_visual_queries.shape)}"
            )
        batch_size = int(current_visual_queries.size(0))
        self._ensure_batch_size(route_keys, batch_size)
        self._ensure_frame_batch_size(frame_idxs, batch_size)
        if self.training:
            self._ensure_sample_batch_size(sample_idxs, batch_size)
        if len(current_det_results) != batch_size or len(current_map_results) != batch_size:
            raise ValueError("current_det_results/current_map_results batch size mismatch in history update")
        if len(current_future_trajs) != batch_size:
            raise ValueError("current_future_trajs batch size mismatch in history update")

        for batch_idx in range(batch_size):
            state = self._prepare_sceneaware_state_before_step(
                route_key=route_keys[batch_idx],
                frame_idx=frame_idxs[batch_idx],
                device=current_visual_queries.device,
                dtype=current_visual_queries.dtype,
                sample_idx=None if sample_idxs is None else sample_idxs[batch_idx],
            )
            reference_entry = self._get_sceneaware_reference_entry(state, frame_idxs[batch_idx])
            current_traj = current_future_trajs[batch_idx].detach().to(dtype=torch.float32)
            current_pose = current_ego_poses[batch_idx].detach().to(dtype=torch.float32)
            current_timestamp = current_timestamps[batch_idx].detach().reshape(()).to(dtype=torch.float32)

            prev_future_traj = None if reference_entry is None else reference_entry["future_traj"]
            prev_ego_pose = None if reference_entry is None else reference_entry["ego_pose"]
            prev_timestamp = None if reference_entry is None else reference_entry["timestamp"]
            if prev_future_traj is not None:
                prev_future_traj = prev_future_traj.to(device=current_traj.device, dtype=torch.float32)
            if prev_ego_pose is not None:
                prev_ego_pose = prev_ego_pose.to(device=current_pose.device, dtype=torch.float32)
            if prev_timestamp is not None:
                prev_timestamp = prev_timestamp.to(device=current_timestamp.device, dtype=torch.float32)
            cached_traj_dev = self._build_traj_deviation(
                current_future_traj=current_traj,
                current_ego_pose=current_pose,
                current_timestamp=current_timestamp,
                prev_future_traj=prev_future_traj,
                prev_ego_pose=prev_ego_pose,
                prev_timestamp=prev_timestamp,
            )

            det_boxes, det_labels = self._extract_det_history_fields(current_det_results[batch_idx])
            map_pts, map_labels = self._extract_map_history_fields(current_map_results[batch_idx])
            state["cached_traj_dev"] = self._to_cpu_float_tensor(cached_traj_dev)
            self._append_sceneaware_history_entry(
                state=state,
                history_entry={
                    "frame_idx": int(frame_idxs[batch_idx]),
                    "visual_queries": self._to_cpu_float_tensor(current_visual_queries[batch_idx]),
                    "det_boxes": det_boxes,
                    "det_labels": det_labels,
                    "map_pts": map_pts,
                    "map_labels": map_labels,
                    "future_traj": self._to_cpu_float_tensor(current_traj),
                    "ego_pose": self._to_cpu_float_tensor(current_pose),
                    "timestamp": self._to_cpu_float_tensor(current_timestamp),
                },
            )
            state["last_route_key"] = str(route_keys[batch_idx])
            state["last_frame_idx"] = int(frame_idxs[batch_idx])

    def get_query_embeddings(self, device, dtype):
        self.to(device=device, dtype=dtype)
        budget_query = self.budget_query_embed.to(device=device, dtype=dtype)
        path_query = self.path_query_embed.to(device=device, dtype=dtype)
        return budget_query, path_query

    def encode_budget_token(self, budget_values: torch.Tensor) -> torch.Tensor:
        budget_values = budget_values.float().view(-1)
        if torch.any(budget_values < self.budget_min) or torch.any(budget_values > 1.0):
            raise ValueError(
                f"budget must be in [{self.budget_min:.6f}, 1.0], got "
                f"min={float(budget_values.min().item()):.6f}, max={float(budget_values.max().item()):.6f}"
            )
        _, quantized_budget = budget_quantizing(
            budget=budget_values,
            num_prefix_layers=self.num_prefix_layers,
            num_hidden_layers=self.num_hidden_layers,
        )
        scaled_values = quantized_budget * (2.0 * math.pi)
        frequencies = 1.0 / (
            10000
            ** (torch.arange(128, device=quantized_budget.device, dtype=quantized_budget.dtype) / 128)
        )
        sin_values = torch.sin(scaled_values[:, None] * frequencies[None, :])
        cos_values = torch.cos(scaled_values[:, None] * frequencies[None, :])
        budget_emb = torch.cat((sin_values, cos_values), dim=-1)
        target_dtype = self.budget_encoder[1].weight.dtype
        return self.budget_encoder(budget_emb.to(dtype=target_dtype))

    def _collect_single_positions(self, input_ids, token_id, token_name):
        positions = []
        for batch_idx, sample_ids in enumerate(input_ids):
            match_positions = torch.nonzero(sample_ids == token_id, as_tuple=False).flatten()
            if match_positions.numel() != 1:
                raise ValueError(
                    f"{token_name} token must appear exactly once per sample, "
                    f"got {match_positions.numel()} on batch index {batch_idx}"
                )
            positions.append(match_positions[0])
        return torch.stack(positions, dim=0)

    def _cache_language_runtime(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if inputs_embeds is not None:
            self.runtime_language_inputs = inputs_embeds
        self.runtime_language_ids = input_ids
        if attention_mask is None:
            self.runtime_language_inputs_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            self.runtime_language_inputs_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    def prepare_llm_runtime(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 关键调用点：每次进入 LLM 前都显式重置预算/path runtime，scene-aware token 已在外部写回 inputs_embeds。
        self.reset_runtime_state()
        if input_ids is None:
            raise ValueError("input_ids must not be None when adaption is enabled")
        self._cache_language_runtime(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        self.budget_query_positions = self._collect_single_positions(
            input_ids,
            ADALAVA_BUDGET_TOKEN_INDEX,
            "budget query",
        )
        self.path_query_positions = self._collect_single_positions(
            input_ids,
            ADALAVA_PATH_TOKEN_INDEX,
            "path query",
        )
        sceneaware_positions = []
        for token_id, token_name in [
            (ADALAVA_SCENEAWARE_VISUAL_TOKEN_INDEX, "scene-aware visual"),
            (ADALAVA_SCENEAWARE_DET_TOKEN_INDEX, "scene-aware det"),
            (ADALAVA_SCENEAWARE_MAP_TOKEN_INDEX, "scene-aware map"),
            (ADALAVA_SCENEAWARE_TRAJ_TOKEN_INDEX, "scene-aware traj"),
        ]:
            sceneaware_positions.append(self._collect_single_positions(input_ids, token_id, token_name))
        self.sceneaware_token_positions = torch.stack(sceneaware_positions, dim=1)

    def _gather_query_hidden(self, hidden_states, positions):
        positions = positions.to(hidden_states.device)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    def process_hidden_states_before_layer(self, hidden_states, layer_idx):
        if layer_idx != self.budget_split_layer:
            return hidden_states
        if self.budget_query_positions is None:
            raise ValueError("prepare_llm_runtime must be called before adaptive forward")
        if self.selected_budget_values is not None:
            return hidden_states

        budget_hidden = self._gather_query_hidden(hidden_states, self.budget_query_positions)
        if self.training and self.stage1_enabled:
            curriculum_min = self._compute_stage1_curriculum_min()
            budget_values = curriculum_min + (1.0 - curriculum_min) * torch.rand(
                (hidden_states.shape[0],),
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            budget_logits = self.budget_scheduler(budget_hidden)
            budget_probs = F.softmax(budget_logits, dim=-1)
            selected_budget_indices = budget_logits.argmax(dim=-1)
            hard_budget = F.one_hot(selected_budget_indices, num_classes=self.budget_num_actions).to(
                dtype=budget_probs.dtype
            )
            if self.training:
                budget_code = hard_budget - budget_probs.detach() + budget_probs
            else:
                budget_code = hard_budget
            budget_values = (budget_code * self._resolve_budget_action_values(hidden_states.device, budget_code.dtype).unsqueeze(0)).sum(dim=-1)

        encoded_budget = self.encode_budget_token(budget_values).to(dtype=hidden_states.dtype)
        updated_hidden_states = hidden_states.clone()
        budget_positions = self.budget_query_positions.to(hidden_states.device)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        updated_hidden_states[batch_indices, budget_positions] = budget_hidden + encoded_budget

        self.selected_budget_values = budget_values.to(device=hidden_states.device, dtype=torch.float32)
        self.runtime_budget_features = encoded_budget
        return updated_hidden_states

    def _resolve_budget_action_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return build_budget_action_table(
            budget_min=self.budget_min,
            device=device,
            dtype=dtype,
            num_actions=self.budget_num_actions,
        )

    def build_execution_plan_on_prefix_end(self, hidden_states, layer_idx):
        if layer_idx != self.num_prefix_layers:
            return
        if self.execution_plan is not None:
            return
        if self.selected_budget_values is None:
            raise ValueError("budget must be decided before building execution plan")
        if self.runtime_budget_features is None:
            raise ValueError("encoded budget feature must be prepared before building execution plan")
        if self.sceneaware_token_positions is None:
            raise ValueError("scene-aware token positions must be prepared before building execution plan")

        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        sceneaware_positions = self.sceneaware_token_positions.to(hidden_states.device)
        sceneaware_hidden = hidden_states[batch_indices[:, None], sceneaware_positions].reshape(hidden_states.shape[0], -1)
        path_hidden = self._gather_query_hidden(hidden_states, self.path_query_positions)
        scheduler_inputs = torch.cat([sceneaware_hidden, path_hidden, self.runtime_budget_features], dim=-1)
        path_logits = self.path_scheduler(scheduler_inputs)
        self.path_logits = path_logits
        execution_plan_hard = torch.zeros_like(path_logits, dtype=torch.bool)
        execution_plan = torch.zeros_like(path_logits, dtype=path_logits.dtype)
        budget_units, _ = budget_quantizing(
            budget=self.selected_budget_values,
            num_prefix_layers=self.num_prefix_layers,
            num_hidden_layers=self.num_hidden_layers,
        )
        for batch_idx in range(path_logits.shape[0]):
            topk = int(budget_units[batch_idx].item())
            if topk <= 0:
                continue
            sampled_mask = masked_gumbel_softmax_topk(
                logits=path_logits[batch_idx],
                k=topk,
                tau=self.path_gumbel_tau,
                hard=self.path_gumbel_hard,
                training=self.training,
            )
            execution_plan[batch_idx] = sampled_mask
            execution_plan_hard[batch_idx] = sampled_mask.detach().round().to(dtype=torch.bool)
        self.execution_plan = execution_plan
        self.execution_plan_hard = execution_plan_hard
        self.path_mask_st = execution_plan
        self.path_mask_hard = execution_plan_hard

    def get_layer_active_mask(self, layer_idx):
        if layer_idx < self.num_prefix_layers:
            raise ValueError(f"layer_idx={layer_idx} is not a suffix layer")
        if self.execution_plan is None or self.execution_plan_hard is None:
            raise ValueError("execution plan has not been built yet")
        suffix_idx = layer_idx - self.num_prefix_layers
        if self.training:
            return self.execution_plan[:, suffix_idx]
        return self.execution_plan_hard[:, suffix_idx]

    def get_stage1_aux_training_signals(self) -> Dict[str, Any]:
        """导出 stage1 aux loss 所需的 runtime 信号。"""
        required_fields = {
            "language_inputs": self.runtime_language_inputs,
            "language_ids": self.runtime_language_ids,
            "language_inputs_mask": self.runtime_language_inputs_mask,
            "budget_token_features": self.runtime_budget_features,
            "path_mask_st": self.path_mask_st,
            "path_mask_hard": self.path_mask_hard,
            "budget_values": self.selected_budget_values,
            "path_logits": self.path_logits,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ValueError(f"stage1 aux runtime signals are incomplete: missing {missing}")
        return {
            "language_inputs": self.runtime_language_inputs,
            "language_ids": self.runtime_language_ids,
            "language_inputs_mask": self.runtime_language_inputs_mask,
            "budget_token_features": self.runtime_budget_features,
            "path_mask_st": self.path_mask_st,
            "path_mask_hard": self.path_mask_hard,
            "budget_values": self.selected_budget_values,
            "path_logits": self.path_logits,
            "num_prefix_layers": self.num_prefix_layers,
            "num_hidden_layers": self.num_hidden_layers,
            "budget_min": self.budget_min,
        }
