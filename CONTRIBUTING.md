# Contributing

Issues and pull requests are welcome. For bugs, include the smallest runnable
example, Python/package versions, the score orientation, calibration-unit
definition, and whether splitting occurred by complete trace.

Before opening a pull request:

```bash
python -m pip install -e ".[demo,dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m py_compile crop/api.py scripts/*.py
```

New methods should include unit tests and should report both risk and utility.
Experiment contributions should state the dataset license, trace-level split,
random seeds, alpha selection policy, and whether results came from cached or
new model outputs. Do not commit private data, credentials, model checkpoints,
or generated result directories.

Changes to the risk-control guarantee or loss definition require corresponding
documentation and a focused regression test.
