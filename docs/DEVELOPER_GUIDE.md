# Developer Guide

## Column Namespace Enforcement

All DataFrames produced by the training pipeline must use the standardized
namespace prefixes:

- `feat__` for feature columns
- `label__` for label score columns
- `target__` for supervised targets
- `meta__` for metadata

Legacy single-underscore prefixes (e.g. `target_small`) are no longer allowed.
When creating or renaming columns use the helpers in
`src/training/steps/pre_training/column_naming.py` such as
`ensure_namespace` and `ensure_dataframe_namespace`.

A CI check (`code_quality/check_column_prefixes.py`) scans committed code for
illegal prefixes and will fail if new violations are introduced. Run it locally
before committing:

```bash
python code_quality/check_column_prefixes.py
```
