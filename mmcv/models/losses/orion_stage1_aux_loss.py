import torch
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


class OrionStage1AuxLossComputer:
    """对齐 simlingo stage1 的 div / bal 两个辅助 loss。"""

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
        language_inputs_mask = language_inputs_mask.to(dtype=torch.bool)
        visual_mask = language_inputs_mask & language_ids.eq(IMAGE_TOKEN_INDEX)
        prompt_mask = language_inputs_mask & language_ids.ge(0)

        prompt_pool = _masked_max_pool(language_inputs, prompt_mask)
        visual_pool = _masked_max_pool(language_inputs, visual_mask)
        budget_pool = budget_token_features.float()
        scene_features = torch.maximum(torch.maximum(prompt_pool.float(), visual_pool.float()), budget_pool)

        scene_norm = F.normalize(scene_features, p=2, dim=-1, eps=1e-6)
        path_norm = F.normalize(path_mask_st.float(), p=2, dim=-1, eps=1e-6)
        scene_similarity = scene_norm @ scene_norm.transpose(0, 1)
        path_similarity = path_norm @ path_norm.transpose(0, 1)
        div_loss = F.relu(path_similarity - scene_similarity - float(div_margin)).pow(2).mean()

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
        sub_budget = (budget_units / float(num_suffix_layers)).clamp_(0.0, 1.0)
        target_sub_budget = sub_budget.mean()
        mean_path_activation = path_mask_st.float().mean(dim=0)
        bal_loss = F.relu((mean_path_activation - target_sub_budget).abs() - float(bal_epsilon)).pow(2).mean()

        return {
            "div_loss": div_loss,
            "bal_loss": bal_loss,
            "scene_similarity_mean": scene_similarity.mean(),
            "path_similarity_mean": path_similarity.mean(),
            "target_sub_budget": target_sub_budget,
            "path_mask_hard_mean": path_mask_hard.float().mean(),
            "budget_mean": budget_values.float().mean(),
        }
