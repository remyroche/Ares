# Stage-D action feature infrastructure audit

## Disposition

`PASS_FEATURE_INFRASTRUCTURE_WITH_DECLARED_CONDITIONAL_GROUP_REJECTIONS`

Canonical artifact: `data_perp/artifacts/stage_d_action_features_20260731_v3/`.
The independently regenerated v4 directory is byte-identical evidence only.

## Population and timestamps

- 108,139 unique feature rows; candidate set SHA256 `2088db5b78152be60a5b1b6a5f69500d6010e3e1a01b174dff0df5186d7b78b5`, exactly matching D0 v2.
- The completed path ends at the clear bar open and its close is exactly `action_decision_ts`; every path feature therefore stops at the action decision.
- Hourly market bars are open-stamped. A5 uses only a bar satisfying `bar_open + 1h <= action_decision_ts`; the source open and availability timestamp are persisted.
- The synchronized A5 universe contains every frozen-universe symbol present at that completed cutoff, not merely symbols with clear actions. Per-cutoff membership count and SHA256 are persisted.
- No counterfactual outcome, future MFE/MAE, entry-policy mutation, model, threshold, or portfolio field is present.

## Compute contract

- JSON payload parsing is the only per-payload operation.
- OHLC numerical transforms are NumPy-vectorized over bounded batches; fixed time-axis loops update all rows simultaneously.
- Path batches are streamed to temporary Parquet rather than accumulated as Python records.
- Observed maximum batch: 183 rows × 718 minutes. Conservative numeric working-set bound: 22,804,992 bytes.
- Scalar-versus-vectorized parity passed on synthetic edge cases and 30 immutable real paths; maximum observed relative difference was `3.11e-13`.

## Conditional dispositions

- A3: `REJECTED_SOURCE_UNAVAILABLE`. The sealed exact one-minute paths contain timestamp/OHLC but no volume, and no fully aligned immutable one-minute volume source was proven. Volume-dependent A9 composites are `NOT_RUN_BLOCKED_COMPONENT`.
- A6: `REJECTED_LINEAGE`.
- A7: `REJECTED_LINEAGE`.
- A8: `REJECTED_OOF_LINEAGE` because no strict action-level OOF/prequential sidecar was proven.
- A0 transitively rejects direct OI/funding/order-book fields and hidden composites `mkt_flush_exhaustion_score`, `mkt_leverage_rebuild_score`, `unwind`, `unwind_score`, and `xasset_mkt_spread_bps`. The complete per-side dependency disposition is persisted.

## Reproducibility and tests

- v3 and independent v4 manifests have the identical SHA256 `b6655bff5b9b73a8bc0c2b82df1c61b5a8beae7214ccc149136b9849d53f7bf8`.
- Every output SHA256, input SHA256, code SHA256, and compute-contract field matches across the two runs.
- `tests/test_stage_d_action_features.py`: 14 passed.
- `tests/test_materialize_stage_d_action_counterfactuals.py`: 19 passed; combined focused suite: 33 passed.

The earlier v1/v2 feature directories are superseded and must not be consumed by Stage D1/D2.
