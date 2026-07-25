import os as _os

_base_ = ["./orion_stage3_agent_fp16.py"]

stage1_inference_budget = float(_os.environ.get("ORION_STAGE1_INFERENCE_BUDGET", "1.0"))

model = dict(
    adaption_cfg=dict(
        inference_budget=stage1_inference_budget,
        train_stage="stage1",
    ),
)

del _os
