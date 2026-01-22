# Cross-Asset Panel Contract (Layer2)

This document defines the immutable panel schema required by the **LabelBasedLayer2** cross-asset pipeline.

## 1) Index
- **MultiIndex**: `(timestamp, ticker)`
- `timestamp` **must** be `DatetimeIndex`
- No duplicate index rows

## 2) Namespace prefixes
All columns must use one of the approved prefixes:

| Prefix | Meaning |
| --- | --- |
| `raw__` | raw market inputs or derived raw features |
| `y__` | labels / targets |
| `sa__` | single-asset features |
| `cs__` | cross-sectional features |
| `ca__` | cross-asset features |
| `ms__` | market-state features |
| `gate__` | gating outputs |

**Contract**: unprefixed columns are invalid. The panel builder will enforce these prefixes and raise if invalid columns remain.

## 3) Required columns
At minimum, the panel must contain:

- `raw__px`
- `y__ret_1`
- `raw__vol`
- `raw__dvol`

## 4) Default inputs used to build the panel
Each asset input dataframe should contain:

- `open`, `high`, `low`, `close`, `volume` (lowercase)
- If `close` is unavailable, the processor will attempt to use `px`, `price`, `last`, or `settle`.

## 5) Leakage sentinels
The panel processor produces:

- **Correlation sentinel**: compare feature-to-label correlation vs shifted label.
- **Timestamp perturbation**: shuffle time order and compare correlation collapse.

Artifacts are persisted to:

- `artifacts/cross_asset_layer2/leakage_report_<dataset_tag>_<run_id>.json`

## 6) Validation battery artifacts
When enabled, cross-asset validation produces:

- `validation_summary_<dataset_tag>_<run_id>.json`
- `validation_<split>_by_asset_<dataset_tag>_<run_id>.csv`
- `validation_<split>_by_sector_<dataset_tag>_<run_id>.csv`

Splits: `LOAO`, `LOSO`, `SYNTHETIC`.

## 7) Invariance artifacts
When enabled, meta-model invariance produces:

- `invariance_report_<dataset_tag>_<run_id>.json`

Includes dispersion metrics and removed features.
