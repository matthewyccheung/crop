# Expected Inputs

The release is source-only. Create or provide these files before running the
paper-result scripts:

- Target annotations: `data/crop_hf/`
- Target text features: `data/cheap_baselines/crop_target_text_steps.npz`
- Target trace features: `data/strengthened/crop_target_combined_steps.npz`
- Target PRM scores:
  `outputs/strengthened/final/process_repeated_qwen_prm/qwen_prm_scores.csv`
- External process features:
  `outputs/strengthened/final/external_process/*/*_combined_steps.npz`
- External PRM scores:
  `outputs/strengthened/final/external_process/*_qwen_prm/qwen_prm_scores.csv`

All `.npz` feature files should contain:

- `features`: numeric feature matrix
- `metadata`: per-step or per-trace metadata dictionaries
- `feature_names`: feature column names when available
