# Experiment Tracking

This file will track experiments once implementation begins. The table is a placeholder and should be updated with config paths, dataset versions, checkpoints, and key observations.

| ID | Date | Stage | Model | Data | Routing | Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | TODO | Scaffold | None | None | None | None | Repository structure only. |
| EXP-001 | 2026-06-02 | Zero-shot baseline | BLIP VQA pretrained | Multi-task validation subset | None | TODO | Run with `python scripts/evaluate.py --model blip --limit <N>`. |
| EXP-002 | 2026-06-02 | LoRA baseline | BLIP VQA + PEFT LoRA | Multi-task validation subset | None | TODO | Run with `python scripts/evaluate.py --model blip_lora --adapter-path <checkpoint> --limit <N>`. |

## Baseline Evaluation Commands

Evaluate pretrained BLIP without fine-tuning:

```bash
python scripts/evaluate.py \
  --model blip \
  --metadata-path data/processed/multitask/validation.jsonl \
  --limit 100 \
  --predictions-path outputs/predictions/blip_pretrained.jsonl
```

Evaluate the BLIP LoRA adapter checkpoint:

```bash
python scripts/evaluate.py \
  --model blip_lora \
  --adapter-path outputs/checkpoints/blip_lora_baseline_sample \
  --metadata-path data/processed/multitask/validation.jsonl \
  --limit 100 \
  --predictions-path outputs/predictions/blip_lora.jsonl
```

## Planned Fields

- Config file
- Dataset versions
- Checkpoint path
- Weights & Biases run URL
- Exact Match
- ANLS
- Task-level accuracy
- Routing accuracy
- Latency
- Memory usage
- Qualitative failure cases
