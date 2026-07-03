import torch
import torch.nn as nn
import torch.nn.functional as F


ADALAVA_BUDGET_TOKEN_INDEX = -300
ADALAVA_PATH_TOKEN_INDEX = -301


class OrionBudgetAssigner(nn.Module):
    """管理 Orion 的预算决策和 suffix layer 选择。"""

    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_prefix_layers,
        budget_candidates,
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
        if not budget_candidates:
            raise ValueError("budget_candidates must not be empty when adaption is enabled")
        budget_candidates = [int(v) for v in budget_candidates]
        if any(v <= 0 for v in budget_candidates):
            raise ValueError(f"budget_candidates must be positive, got {budget_candidates}")
        if any(v > self.num_suffix_layers for v in budget_candidates):
            raise ValueError(
                f"budget_candidates must be <= num_suffix_layers={self.num_suffix_layers}, got {budget_candidates}"
            )
        self.budget_candidates = budget_candidates

        self.budget_query_embed = nn.Parameter(torch.empty(1, self.hidden_size))
        self.path_query_embed = nn.Parameter(torch.empty(1, self.hidden_size))
        self.budget_scheduler = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, len(self.budget_candidates)),
        )
        self.budget_encoder = nn.Sequential(
            nn.Linear(len(self.budget_candidates), self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.path_scheduler = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.num_suffix_layers),
        )
        self.register_buffer(
            "budget_candidate_tensor",
            torch.tensor(self.budget_candidates, dtype=torch.long),
            persistent=False,
        )
        self._reset_parameters()
        self.reset_runtime_state()

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
        self.selected_budget_indices = None
        self.selected_budget_values = None
        self.execution_plan = None

    def get_query_embeddings(self, device, dtype):
        self.to(device=device, dtype=dtype)
        budget_query = self.budget_query_embed.to(device=device, dtype=dtype)
        path_query = self.path_query_embed.to(device=device, dtype=dtype)
        return budget_query, path_query

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

    def prepare_llm_runtime(self, input_ids):
        # 关键调用点：每次进入 LLM 前都显式重置 runtime，避免跨样本串状态。
        self.reset_runtime_state()
        if input_ids is None:
            raise ValueError("input_ids must not be None when adaption is enabled")
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

    def _gather_query_hidden(self, hidden_states, positions):
        positions = positions.to(hidden_states.device)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    def process_hidden_states_before_layer(self, hidden_states, layer_idx):
        if layer_idx != self.budget_split_layer:
            return hidden_states
        if self.budget_query_positions is None:
            raise ValueError("prepare_llm_runtime must be called before adaptive forward")
        if self.selected_budget_indices is not None:
            return hidden_states

        budget_hidden = self._gather_query_hidden(hidden_states, self.budget_query_positions)
        budget_logits = self.budget_scheduler(budget_hidden)
        budget_probs = F.softmax(budget_logits, dim=-1)
        selected_budget_indices = budget_logits.argmax(dim=-1)
        hard_budget = F.one_hot(selected_budget_indices, num_classes=len(self.budget_candidates)).to(
            dtype=budget_probs.dtype
        )
        if self.training:
            budget_code = hard_budget - budget_probs.detach() + budget_probs
        else:
            budget_code = hard_budget

        encoded_budget = self.budget_encoder(budget_code.to(dtype=hidden_states.dtype))
        updated_hidden_states = hidden_states.clone()
        budget_positions = self.budget_query_positions.to(hidden_states.device)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        updated_hidden_states[batch_indices, budget_positions] = budget_hidden + encoded_budget

        self.selected_budget_indices = selected_budget_indices
        self.selected_budget_values = self.budget_candidate_tensor[selected_budget_indices].to(hidden_states.device)
        return updated_hidden_states

    def build_execution_plan_on_prefix_end(self, hidden_states, layer_idx):
        if layer_idx != self.num_prefix_layers:
            return
        if self.execution_plan is not None:
            return
        if self.selected_budget_values is None:
            raise ValueError("budget must be decided before building execution plan")

        budget_hidden = self._gather_query_hidden(hidden_states, self.budget_query_positions)
        path_hidden = self._gather_query_hidden(hidden_states, self.path_query_positions)
        path_logits = self.path_scheduler(torch.cat([path_hidden, budget_hidden], dim=-1))
        execution_plan = torch.zeros_like(path_logits, dtype=torch.bool)
        for batch_idx in range(path_logits.shape[0]):
            topk = int(self.selected_budget_values[batch_idx].item())
            topk_indices = torch.topk(path_logits[batch_idx], k=topk, dim=-1).indices
            execution_plan[batch_idx, topk_indices] = True
        self.execution_plan = execution_plan

    def get_layer_active_mask(self, layer_idx):
        if layer_idx < self.num_prefix_layers:
            raise ValueError(f"layer_idx={layer_idx} is not a suffix layer")
        if self.execution_plan is None:
            raise ValueError("execution plan has not been built yet")
        suffix_idx = layer_idx - self.num_prefix_layers
        return self.execution_plan[:, suffix_idx]
