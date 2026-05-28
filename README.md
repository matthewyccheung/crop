# CROP Reproducibility Release

This source-only release contains the CROP package, tests, and the scripts
needed to regenerate the reported paper results after the required datasets and
model-output files are prepared.

It does not bundle datasets, feature files, generated figures, generated
tables, cached model outputs, or paper source.

## Setup

```bash
cd ~/crop_anon_release
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Validate

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

## Reproduce Results

See `docs/INPUTS.md` for required input files and
`docs/REPRODUCIBILITY.md` for the paper-result command map.
