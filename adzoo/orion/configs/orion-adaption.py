# 关键路径保留为用户目录形式，具体展开下沉到真正加载文件的代码路径。
orion_asset_root = '~/Orion-adaption/Orion'
llm_path = '~/Orion-adaption/Orion/pretrain_qformer'
eva02_petr_proj_path = '~/Orion-adaption/Orion/eva02_petr_proj.pth'
orion_checkpoint_path = '~/Orion-adaption/Orion/Orion.pth'


def _detect_flash_attn_support():
    try:
        torch_mod = __import__('torch')
    except Exception as exc:
        print(f"[orion config] Failed to import torch, disable flash_attn: {exc}")
        return False

    try:
        __import__('flash_attn')
    except Exception as exc:
        print(f"[orion config] Failed to import flash_attn, disable flash_attn: {exc}")
        return False

    if not torch_mod.cuda.is_available():
        print("[orion config] CUDA is unavailable, disable flash_attn.")
        return False

    try:
        device_index = torch_mod.cuda.current_device()
        device_name = torch_mod.cuda.get_device_name(device_index)
        major, minor = torch_mod.cuda.get_device_capability(device_index)
    except Exception as exc:
        print(f"[orion config] Failed to inspect CUDA device, disable flash_attn: {exc}")
        return False

    use_flash_attn = major >= 8
    status = "enable" if use_flash_attn else "disable"
    print(
        f"[orion config] Detected GPU {device_name} "
        f"(compute capability {major}.{minor}), {status} flash_attn."
    )
    return use_flash_attn


use_flash_attn = _detect_flash_attn_support()
del _detect_flash_attn_support

sample_interval = 5

orion_adaption_cfg = dict(
    enabled=True,
    num_prefix_layers=6,
    budget_curriculum_start_min=0.8,
    budget_curriculum_warmup_steps=20000,
    path_gumbel_tau=1.0,
    path_gumbel_hard=True,
    stage1_aux_enabled=False,
    stage1_aux_div_margin=0.05,
    stage1_aux_bal_epsilon=0.05,
    stage1_aux_div_ratio_start=0.01,
    stage1_aux_div_ratio_end=0.05,
    stage1_aux_div_warmup_steps=1000,
    stage1_aux_bal_ratio=0.01,
    sceneaware_enabled=True,
    sample_interval=sample_interval,
    train_stage=None,
)
