_base_ = ["./orion_stage1_adaption_train.py"]

from datetime import datetime

run_name = f"orion_stage2_{datetime.now().strftime('%m%d_%H%M')}_budget_only_grpo"
work_dir = f"work_dirs/{run_name}"
del datetime

stage1_adaption_checkpoint = "work_dirs/orion_stage1_0720_1751/iter_468000.pth"
full_budget_losses_jsonl = "data/infos/full_budget_losses/orion_stage1_0720_1751_full_budget_all.jsonl"
orion_adaption_cfg = dict(
    enabled={{_base_.orion_adaption_cfg.enabled}},
    num_prefix_layers={{_base_.orion_adaption_cfg.num_prefix_layers}},
    budget_curriculum_start_min={{_base_.orion_adaption_cfg.budget_curriculum_start_min}},
    budget_curriculum_warmup_steps={{_base_.orion_adaption_cfg.budget_curriculum_warmup_steps}},
    path_gumbel_tau={{_base_.orion_adaption_cfg.path_gumbel_tau}},
    path_gumbel_hard={{_base_.orion_adaption_cfg.path_gumbel_hard}},
    stage1_aux_enabled=False,
    stage1_aux_div_enabled=False,
    stage1_aux_div_margin={{_base_.orion_adaption_cfg.stage1_aux_div_margin}},
    stage1_aux_bal_epsilon={{_base_.orion_adaption_cfg.stage1_aux_bal_epsilon}},
    stage1_aux_div_ratio_start={{_base_.orion_adaption_cfg.stage1_aux_div_ratio_start}},
    stage1_aux_div_ratio_end={{_base_.orion_adaption_cfg.stage1_aux_div_ratio_end}},
    stage1_aux_div_warmup_steps={{_base_.orion_adaption_cfg.stage1_aux_div_warmup_steps}},
    stage1_aux_bal_ratio={{_base_.orion_adaption_cfg.stage1_aux_bal_ratio}},
    stage1_aux_div_queue_size={{_base_.orion_adaption_cfg.stage1_aux_div_queue_size}},
    sceneaware_enabled=True,
    sample_interval={{_base_.sample_interval}},
    train_stage="stage2",
    stage2_grpo=dict(
        grpo_group_size=5,
        grpo_update_epochs=2,
        grpo_clip_eps=0.2,
        grpo_kl_beta=0.005,
        grpo_entropy_beta=0.01,
        grpo_entropy_beta_start=0.0,
        grpo_entropy_warmup_steps=1000,
        grpo_entropy_beta_final=0.003,
        grpo_entropy_decay_steps=50000,
        grpo_entropy_normalize=True,
        grpo_rollout_sampling="bucket",
        grpo_gap_abs_eps=0.02,
        grpo_gap_rel_eps=0.1,
        grpo_gate_temp=0.02,
        grpo_reward_hit=0.4,
        grpo_reward_save=0.10,
        grpo_reward_save_margin=0.03,
        grpo_reward_save_temp=0.02,
        grpo_reward_save_gamma=1.0,
        grpo_reward_margin=0.4,
        grpo_reward_margin_scale=0.08,
        grpo_reward_violate=2.0,
        grpo_reward_violate_scale=0.05,
        grpo_abs_reward_weight=0.2,
        grpo_abs_reward_target=0.0,
        grpo_abs_reward_scale=1.0,
        grpo_adv_eps=1e-6,
        grpo_budget_explore_enable=True,
        grpo_budget_explore_temp_start=1.5,
        grpo_budget_explore_temp_end=1.0,
        grpo_budget_explore_eps_start=0.20,
        grpo_budget_explore_eps_end=0.03,
        grpo_budget_explore_anneal_ratio=0.15,
        grpo_unfreeze_prefix=False,
        unfreeze_scene_aware=False,
        grpo_prefix_kl_beta=0.0,
        grpo_scene_kl_beta=0.0,
    ),
)

model = dict(adaption_cfg=orion_adaption_cfg)
samples_per_gpu = 1
data = dict(
    samples_per_gpu=samples_per_gpu,
    train=dict(full_budget_losses_jsonl=full_budget_losses_jsonl),
)

# 关键调用点：stage2 参考 SimLingo v2 的 1e-4 policy LR；避免继承 stage1 head_decay_rate=4 后实际变成 4e-4。
optimizer = dict(
    lr=1e-4,
    paramwise_cfg=dict(
        decay_rate=1.0,
        head_decay_rate=1.0,
        lm_head_decay_rate=1.0,
    ),
)
load_from = stage1_adaption_checkpoint
resume_from = None

# 关键调用点：stage2 policy-only 训练使用约 0.5% warmup 和非零 floor，对齐 SimLingo v3 的调度形状。
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=2500,
    warmup_ratio=0.1,
    min_lr_ratio=0.1,
)

# 关键调用点：stage2 同一 rollout 需要多次更新，不能使用跨 batch 的累积梯度 hook。
optimizer_config = dict(
    _delete_=True,
    type="OrionStage2GRPOOptimizerHook",
    grad_clip=dict(max_norm=35, norm_type=2),
)

find_unused_parameters = True
