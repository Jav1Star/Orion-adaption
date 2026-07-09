import torch
import torch.nn as nn
import torch.nn.functional as F


ADALAVA_BUDGET_TOKEN_INDEX = -300
ADALAVA_PATH_TOKEN_INDEX = -301
ADALAVA_SCENEAWARE_VISUAL_TOKEN_INDEX = -302
ADALAVA_SCENEAWARE_DET_TOKEN_INDEX = -303
ADALAVA_SCENEAWARE_MAP_TOKEN_INDEX = -304
ADALAVA_SCENEAWARE_TRAJ_TOKEN_INDEX = -305


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
        self.sceneaware_enabled = False
        self.sceneaware_num_tokens = 0
        self.sceneaware_token_ids = []
        self.det_num_classes = 0
        self.map_num_classes = 0
        self._sceneaware_modules_initialized = False
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
        self.selected_budget_indices = None
        self.selected_budget_values = None
        self.execution_plan = None
        self.sceneaware_batch_states = None

    def init_sceneaware_modules(self, sceneaware_cfg, det_num_classes, map_num_classes):
        """初始化 scene-aware 编码模块，历史缓存仍由 assigner 独占维护。"""
        self.sceneaware_enabled = bool(sceneaware_cfg.get("sceneaware_enabled", False))
        self.sceneaware_num_tokens = int(sceneaware_cfg.get("sceneaware_num_tokens", 0))
        if not self.sceneaware_enabled:
            self.sceneaware_num_tokens = 0
            self.sceneaware_token_ids = []
            self._sceneaware_modules_initialized = False
            self.reset_sceneaware_history()
            return
        if self.sceneaware_num_tokens != 4:
            raise ValueError(f"sceneaware_num_tokens must be 4, got {self.sceneaware_num_tokens}")

        self.det_num_classes = int(det_num_classes)
        self.map_num_classes = int(map_num_classes)
        class_embed_dim = min(64, self.hidden_size // 8)
        self.sceneaware_token_ids = [
            ADALAVA_SCENEAWARE_VISUAL_TOKEN_INDEX,
            ADALAVA_SCENEAWARE_DET_TOKEN_INDEX,
            ADALAVA_SCENEAWARE_MAP_TOKEN_INDEX,
            ADALAVA_SCENEAWARE_TRAJ_TOKEN_INDEX,
        ]
        self.visual_pair_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.det_class_embed = nn.Embedding(self.det_num_classes, class_embed_dim)
        self.map_class_embed = nn.Embedding(self.map_num_classes, class_embed_dim)
        self.det_item_mlp = nn.Sequential(
            nn.Linear(9 + 1 + class_embed_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.map_item_mlp = nn.Sequential(
            nn.Linear(33 + 1 + class_embed_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.traj_mlp = nn.Sequential(
            nn.Linear(12, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        nn.init.normal_(self.det_class_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.map_class_embed.weight, mean=0.0, std=0.02)
        for module in [self.visual_pair_mlp, self.det_item_mlp, self.map_item_mlp, self.traj_mlp]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.normal_(submodule.weight, mean=0.0, std=0.02)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
        self._sceneaware_modules_initialized = True
        self.reset_sceneaware_history()

    def reset_sceneaware_history(self):
        """重置 route 级 scene-aware 历史，只在新 route 开始时调用。"""
        self.prev_visual_summary = None
        self.prev_det_boxes = None
        self.prev_det_scores = None
        self.prev_det_labels = None
        self.prev_map_pts = None
        self.prev_map_scores = None
        self.prev_map_labels = None
        self.prev_ego_fut_preds = None

    def prepare_sceneaware_batch(self, sceneaware_batch_states):
        """关键调用点：训练态每个 batch 显式写入上一帧上下文，避免复用 route 级缓存。"""
        self.sceneaware_batch_states = sceneaware_batch_states

    def _get_sceneaware_state(self, sample_idx):
        if self.sceneaware_batch_states is not None:
            if sample_idx is None:
                raise ValueError("sample_idx is required when sceneaware_batch_states is set")
            return self.sceneaware_batch_states[sample_idx]
        return dict(
            prev_visual_summary=self.prev_visual_summary,
            prev_det_boxes=self.prev_det_boxes,
            prev_det_scores=self.prev_det_scores,
            prev_det_labels=self.prev_det_labels,
            prev_map_pts=self.prev_map_pts,
            prev_map_scores=self.prev_map_scores,
            prev_map_labels=self.prev_map_labels,
            prev_ego_fut_preds=self.prev_ego_fut_preds,
        )

    def _encode_visual_token(self, current_vision_tokens, state, device, dtype):
        prev_visual_summary = state.get("prev_visual_summary", None)
        if prev_visual_summary is None:
            return torch.zeros((1, 1, self.hidden_size), device=device, dtype=dtype)
        current_visual_summary = current_vision_tokens.mean(dim=1, keepdim=True)
        prev_visual_summary = prev_visual_summary.to(device=device, dtype=dtype)
        visual_pair = torch.cat(
            [current_visual_summary, prev_visual_summary, current_visual_summary - prev_visual_summary],
            dim=-1,
        )
        return self.visual_pair_mlp(visual_pair.squeeze(1)).unsqueeze(1)

    def _encode_det_token(self, state, device, dtype):
        det_boxes = state.get("prev_det_boxes", None)
        if det_boxes is None:
            return torch.zeros((1, 1, self.hidden_size), device=device, dtype=dtype)
        det_scores = state["prev_det_scores"].to(device=device, dtype=dtype)
        det_labels = state["prev_det_labels"].to(device=device, dtype=torch.long)
        det_boxes = det_boxes.to(device=device, dtype=dtype)
        if det_boxes.ndim != 2 or det_boxes.size(-1) != 9:
            raise ValueError(f"Expected det boxes with shape [N, 9], got {tuple(det_boxes.shape)}")
        det_class_embed = self.det_class_embed(det_labels).to(dtype=dtype)
        det_inputs = torch.cat([det_boxes, det_scores.unsqueeze(-1), det_class_embed], dim=-1)
        det_features = self.det_item_mlp(det_inputs)
        det_weight = det_scores.clamp_min(0).unsqueeze(-1)
        det_denom = det_weight.sum().clamp_min(1e-6)
        det_token = (det_features * det_weight).sum(dim=0, keepdim=True) / det_denom
        return det_token.unsqueeze(0)

    def _encode_map_token(self, state, device, dtype):
        map_pts = state.get("prev_map_pts", None)
        if map_pts is None:
            return torch.zeros((1, 1, self.hidden_size), device=device, dtype=dtype)
        map_scores = state["prev_map_scores"].to(device=device, dtype=dtype)
        map_labels = state["prev_map_labels"].to(device=device, dtype=torch.long)
        map_pts = map_pts.to(device=device, dtype=dtype)
        if map_pts.ndim != 3 or tuple(map_pts.shape[1:]) != (11, 3):
            raise ValueError(f"Expected map points with shape [N, 11, 3], got {tuple(map_pts.shape)}")
        map_class_embed = self.map_class_embed(map_labels).to(dtype=dtype)
        map_inputs = torch.cat([map_pts.flatten(1), map_scores.unsqueeze(-1), map_class_embed], dim=-1)
        map_features = self.map_item_mlp(map_inputs)
        map_weight = map_scores.clamp_min(0).unsqueeze(-1)
        map_denom = map_weight.sum().clamp_min(1e-6)
        map_token = (map_features * map_weight).sum(dim=0, keepdim=True) / map_denom
        return map_token.unsqueeze(0)

    def _encode_traj_token(self, state, device, dtype):
        ego_fut_preds = state.get("prev_ego_fut_preds", None)
        if ego_fut_preds is None:
            return torch.zeros((1, 1, self.hidden_size), device=device, dtype=dtype)
        ego_fut_preds = ego_fut_preds.to(device=device, dtype=dtype)
        if tuple(ego_fut_preds.shape) != (6, 2):
            raise ValueError(f"Expected ego trajectory with shape [6, 2], got {tuple(ego_fut_preds.shape)}")
        traj_token = self.traj_mlp(ego_fut_preds.reshape(1, -1))
        return traj_token.unsqueeze(0)

    def get_sceneaware_embeddings(self, current_vision_tokens, sample_idx=None):
        """构造 scene-aware 4 token：当前视觉差分 + 上一帧 det/map/traj。"""
        if not self.sceneaware_enabled:
            raise ValueError("scene-aware is disabled")
        if not self._sceneaware_modules_initialized:
            raise ValueError("scene-aware modules are not initialized")
        if current_vision_tokens.ndim != 3 or current_vision_tokens.size(0) != 1:
            raise ValueError(
                f"scene-aware inference expects current_vision_tokens with shape [1, N, C], got {tuple(current_vision_tokens.shape)}"
            )
        self.to(device=current_vision_tokens.device, dtype=current_vision_tokens.dtype)
        device = current_vision_tokens.device
        dtype = current_vision_tokens.dtype
        state = self._get_sceneaware_state(sample_idx=sample_idx)
        visual_token = self._encode_visual_token(
            current_vision_tokens=current_vision_tokens,
            state=state,
            device=device,
            dtype=dtype,
        )
        det_token = self._encode_det_token(state=state, device=device, dtype=dtype)
        map_token = self._encode_map_token(state=state, device=device, dtype=dtype)
        traj_token = self._encode_traj_token(state=state, device=device, dtype=dtype)
        return torch.cat([visual_token, det_token, map_token, traj_token], dim=1)

    def update_sceneaware_history(self, bbox_result, lane_result, ego_fut_pred, current_vision_tokens=None):
        """缓存上一帧最终解码预测，供下一帧 scene-aware token 直接编码。"""
        if not self.sceneaware_enabled:
            return
        if current_vision_tokens is not None:
            if current_vision_tokens.ndim != 3 or current_vision_tokens.size(0) != 1:
                raise ValueError(
                    "current_vision_tokens must have shape [1, N, C] when updating scene-aware history"
                )
            self.prev_visual_summary = current_vision_tokens.mean(dim=1, keepdim=True).detach().to(
                dtype=torch.float32
            ).cpu()
        boxes_3d = bbox_result["boxes_3d"].tensor.detach().to(dtype=torch.float32).cpu()
        scores_3d = bbox_result["scores_3d"].detach().to(dtype=torch.float32).cpu()
        labels_3d = bbox_result["labels_3d"].detach().to(dtype=torch.long).cpu()
        map_pts_3d = lane_result["map_pts_3d"].detach().to(dtype=torch.float32).cpu()
        map_scores_3d = lane_result["map_scores_3d"].detach().to(dtype=torch.float32).cpu()
        map_labels_3d = lane_result["map_labels_3d"].detach().to(dtype=torch.long).cpu()
        ego_fut_pred = ego_fut_pred.detach().to(dtype=torch.float32).cpu()

        self.prev_det_boxes = boxes_3d
        self.prev_det_scores = scores_3d
        self.prev_det_labels = labels_3d
        self.prev_map_pts = map_pts_3d
        self.prev_map_scores = map_scores_3d
        self.prev_map_labels = map_labels_3d
        self.prev_ego_fut_preds = ego_fut_pred

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
