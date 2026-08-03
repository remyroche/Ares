# Stage-C F4/F5 archived OI/funding lineage blocker

Status: `BLOCKED_NO_NATIVE_OBSERVATION_OR_AVAILABILITY_CLOCK`.

The archived Kraken hourly OI and funding sidecars were audited as read-only
sources.  They contain a value and only `ts` or a pandas index (funding may
also contain price columns).  These are nominal hourly labels, not native
source-observation or source-publication timestamps.

No F4/F5 availability adapter has been created.  Treating the index as a
publication time, borrowing filesystem modification time, or applying an
assumed delay would invent point-in-time lineage.  This is especially unsafe
because the current store uses unbounded forward fill for OI/funding at
`data_store.py:3042-3043`, `3496-3507`, and `4029-4033`.

The required row-level fields missing from both sources are:

- `provider`, `exchange`, `market_id`, `product_kind`
- `source_event_ts`, `source_observed_ts`, `available_ts`, `ingested_ts`
- `source_revision`, `raw_payload_sha256`

F4 additionally lacks `oi_unit` and `unit_conversion_price_ts`. F5
additionally lacks `funding_value_kind` and `settlement_ts`, so it cannot be
shown to be the last published/settled rate rather than an estimate or later
revision.

The sealed source inventory and hashes are in
`data_perp/artifacts/stage_c_oi_funding_lineage_blocker_20260801_v1/`.
That audit intentionally fails closed if native observation or availability
clocks are added in the future; the next step would then be a separately
reviewed as-of adapter using `available_ts <= feature_cutoff_ts`, documented
finite source-specific staleness, stale-row rejection, and PF-linear-USD
product/unit parity.
