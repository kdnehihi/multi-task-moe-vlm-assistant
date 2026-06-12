# Notebooks

Use `train_qwen2vl_lora_baseline.ipynb` for the current Qwen2-VL sanity check.
It runs the shared-LoRA flow in one pass: data prep, label decode checks,
shared LoRA training, shared LoRA evaluation, and a final shared-LoRA report.
Zero-shot evaluation is skipped because the baseline metrics are already known.

The MoE/router notebooks were removed until the shared LoRA baseline is stable.
