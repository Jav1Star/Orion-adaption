llm_path = '/raid/yyj/Orion-adaption/Orion/pretrain_qformer/'

orion_adaption_cfg = dict(
    enabled=True,
    num_prefix_layers=16,
    budget_candidates=[4, 8, 12, 16],
    sceneaware_enabled=False,
    sceneaware_num_tokens=4,
    stage1_enable_prev_frame=False,
    stage1_use_sceneaware=False,
    stage1_loss_mode=None,
)
