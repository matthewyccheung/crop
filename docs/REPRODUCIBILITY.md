# Reproducibility Guide

Run commands from the repository root after installing the package.

```bash
python -m pip install -e ".[dev]"
```

## Validate The Code

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

## Prepare Feature Files

Target annotation/text features:

```bash
python -m crop.scripts.export_cheap_baseline_data \
  --crop_root data/crop_hf \
  --output_dir data/cheap_baselines
```

External process-supervision datasets:

```bash
python -m crop.scripts.import_external_process \
  --output_dir outputs/strengthened/final/external_process
```

Qwen PRM scores:

```bash
PYTHON_BIN=python bash scripts/launch_external_qwen_prm_full.sh
```

## Prefix Utility

Target-domain repeated splits:

```bash
python -m crop.experiments.exp09_process_repeated \
  --step_text_features data/cheap_baselines/crop_target_text_steps.npz \
  --step_coe_features data/cheap_baselines/crop_target_coe_steps.npz \
  --step_combined_features data/strengthened/crop_target_combined_steps.npz \
  --output_dir outputs/budget_cpcc
```

External datasets and the fixed-risk utility table:

```bash
bash scripts/run_external_fixed_score_60_20_20.sh
python scripts/build_fixed_score_table.py
```

Main AUROC-vs-prefix-utility figure:

```bash
python scripts/make_compact_auroc_prefix_figure.py
```

## Boundary Quality

```bash
python scripts/build_boundary_deviation_table.py
```

## Downstream Repair

Generate repeated-split repair outputs:

```bash
python scripts/run_repeated_split_repair.py \
  --seeds 2806:2826 \
  --domains arithmetic,gsm8k \
  --models gemma,qwen,deepseek,llama \
  --run_name full_repeated_split_table3_60_20_20
```

Build the compact repair table:

```bash
python scripts/build_repeated_split_repair_table.py
```

## Appendix Tables

```bash
python scripts/make_strengthening_artifacts.py
```
