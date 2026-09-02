# Stage-II enriched-ledger source map

`scripts/materialize_stage_ii_enriched_ledger.py` accepts one explicit JSON
map with schema `stage_ii_enriched_path_source_map_v1`. It must name exactly
these contiguous signal-close intervals:

```json
{
  "schema": "stage_ii_enriched_path_source_map_v1",
  "sources": [
    {
      "source_id": "historical_2024",
      "kind": "canonical_parquet",
      "start_utc": "2024-01-01T00:00:00Z",
      "end_utc": "2025-01-01T00:00:00Z",
      "paths": ["data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v2/joined_multitask_labels.parquet"],
      "columns": {"symbol": "__symbol__", "signal_close_ts": "__ts__", "decision_ts": "__decision_ts__", "label_available_ts": "__label_resolution_ts__"},
      "descriptor_mapping": {"__log1p_peak_mfe_atr_12h__": "__log1p_peak_mfe_atr_12h__"}
    },
    {
      "source_id": "native_january_2025",
      "kind": "native_path_descriptors",
      "start_utc": "2025-01-01T00:00:00Z",
      "end_utc": "2025-02-01T00:00:00Z",
      "paths": ["data_perp/artifacts/<frozen_january_descriptor_parquet>.parquet"],
      "descriptor_mapping": {"__log1p_peak_mfe_atr_12h__": "__log1p_peak_mfe_atr_12h__"}
    },
    {
      "source_id": "path_archetype_2025_plus",
      "kind": "canonical_parquet",
      "start_utc": "2025-02-01T00:00:00Z",
      "end_utc": "2026-07-11T00:00:00Z",
      "paths": ["data_perp/artifacts/20260722_path_archetype_labels_v8_base_top40_costaware_dense12h/path_archetype_labels.parquet"],
      "columns": {"symbol": "__symbol__", "signal_close_ts": "__ts__", "decision_ts": "__decision_ts__", "label_available_ts": "__label_end_ts__"},
      "descriptor_mapping": {"path_arch_peak_mfe_atr": "path_arch_peak_mfe_atr"}
    }
  ]
}
```

For January, `native_path_descriptors` may point directly at the authoritative
`*_paths.parquet` shards. In that case the source map must explicitly declare
`columns.native_path`, `columns.entry_price`, and
`columns.atr_fraction`; the bridge verifies an exact contiguous 720×1-minute
path starting at `decision_ts`, then materialises only the whitelisted 12-hour
auxiliary descriptors. It never chooses an ATR/barrier proxy. Alternatively it
can consume an already materialised canonical descriptor parquet with the same
explicit mapping. This avoids silently changing target semantics between the
three date ranges.

All descriptor names must be in the code whitelist and every candidate-spec
descriptor must be mapped by every routed source. `label_available_ts` must be
an explicit canonical field or an explicitly declared `+12h` derivation.
The bridge checks signal close → decision +1h → label availability +12h,
exact one-to-one identity coverage, finite fields, input file hashes and the
completed Stage-I artifact before writing its restart-safe checkpoints.
