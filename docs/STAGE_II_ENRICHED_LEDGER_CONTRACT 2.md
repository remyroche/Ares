# Stage-II enriched path/context ledger contract

This is the only accepted substrate for the Stage-II meta-archetype funnel and
its locked OOS replay. It is materialised after Stage I from existing canonical
candidate, path-label, and causal-context artifacts. It does not calculate a
path label from a Stage-I score, rank, mapped EV, or later feature join.

The parquet ledger has one immutable row per executable candidate identity:

```text
candidate_id, symbol, side_name, signal_close_ts, decision_ts, label_available_ts
```

`decision_ts` is exactly signal close +1 hour and `label_available_ts` is
exactly signal close +13 hours. The identity must match the completed Stage-I
full-history OOF ledger one-to-one. The canonical path-label source supplies
the realised descriptors; the canonical feature/context source supplies all
causal fields at `decision_ts`.

For every bounded Stage-II candidate configuration, the ledger must include:

- every selected ordinary meta field;
- every causal recogniser field;
- every declared realised `path_descriptor_col`.

All these fields are finite on the row population passed to Stage II. Path
descriptors are used only while fitting discovery/recogniser states on
prior-resolved training rows. They are explicitly dropped before the causal
recogniser transforms later rows and are never meta-model inputs.

The adjacent manifest is an immutable JSON object:

```json
{
  "schema": "stage_ii_enriched_path_context_ledger_v1",
  "ledger_sha256": "sha256 of this parquet file",
  "identity_columns": [
    "candidate_id", "symbol", "side_name", "signal_close_ts",
    "decision_ts", "label_available_ts"
  ],
  "causal_columns": ["all declared causal and ordinary meta fields"],
  "path_descriptor_columns": ["all declared realised path fields"],
  "label_lineage": {
    "artifact_path": "canonical path-label artifact path",
    "artifact_sha256": "canonical source content hash",
    "identity_sha256": "canonical source identity hash"
  },
  "context_lineage": {
    "artifact_path": "canonical decision-time context artifact path",
    "artifact_sha256": "canonical source content hash",
    "identity_sha256": "canonical source identity hash"
  }
}
```

The executors validate the parquet hash, canonical identity declaration,
required causal/path membership, and both lineage records. Missing rows,
duplicate identities, missing required fields, non-finite inputs, or a
manifest that does not bind the supplied parquet all fail closed.

The development funnel and locked OOS CLI must both receive this adjacent
`manifest.json`; passing only the parquet is intentionally not supported.
For the required three-source routing and January native-path rules, see
[`STAGE_II_ENRICHED_LEDGER_SOURCE_MAP.md`](STAGE_II_ENRICHED_LEDGER_SOURCE_MAP.md).
