import os as _os

_base_ = ["./orion_stage3_agent.py"]

stage1_inference_budget = float(_os.environ.get("ORION_STAGE1_INFERENCE_BUDGET", "1.0"))

model = dict(
    adaption_cfg=dict(
        # 关键调用点：fp32 对照实验只改变推理精度，stage1 budget 仍由启动脚本统一注入。
        inference_budget=stage1_inference_budget,
        train_stage="stage1",
    ),
)

del _os
