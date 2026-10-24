# Reproducing the paper results

Run all commands from the repository root.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[demo,dev]"
```

## Use the released caches

Set `CROP_ARTIFACT_URL` to the companion-cache URL listed on the release page,
then download and verify the files:

```bash
python scripts/download_artifacts.py --base-url "$CROP_ARTIFACT_URL"
python scripts/download_artifacts.py --verify-only
```

Rebuild the main tables and figure:

```bash
bash scripts/run_external_fixed_score_60_20_20.sh
python scripts/build_fixed_score_table.py
python scripts/make_compact_auroc_prefix_figure.py
python scripts/build_boundary_deviation_table.py
```

## Regenerate model-derived inputs

This section is optional when using the released caches.

Export the target annotations and text features:

```bash
python -m crop.scripts.export_cheap_baseline_data \
  --crop_root data/crop_hf \
  --output_dir data/cheap_baselines
```

Import the external process-supervision datasets:

```bash
python -m crop.scripts.import_external_process \
  --output_dir outputs/strengthened/final/external_process
```

Generate Qwen PRM scores:

```bash
RUN_EXPENSIVE=1 PYTHON_BIN=python bash scripts/launch_external_qwen_prm_full.sh
```

## Prefix certification experiments

Run the target-domain repeated splits:

```bash
python -m crop.experiments.exp09_process_repeated \
  --step_text_features data/cheap_baselines/crop_target_text_steps.npz \
  --step_coe_features data/cheap_baselines/crop_target_coe_steps.npz \
  --step_combined_features data/strengthened/crop_target_combined_steps.npz \
  --output_dir outputs/budget_cpcc
```

Run the external datasets and rebuild the summary outputs:

```bash
bash scripts/run_external_fixed_score_60_20_20.sh
python scripts/build_fixed_score_table.py
python scripts/make_compact_auroc_prefix_figure.py
python scripts/build_boundary_deviation_table.py
```

## Downstream repair

Start a local Ollama server with the four repair models installed, then run:

```bash
python scripts/run_repeated_split_repair.py \
  --seeds 2806:2826 \
  --domains arithmetic,gsm8k \
  --models gemma,qwen,deepseek,llama \
  --run_name full_repeated_split_table3_60_20_20

python scripts/build_repeated_split_repair_table.py
```

Expected input paths and file formats are listed in
[INPUTS.md](INPUTS.md).
