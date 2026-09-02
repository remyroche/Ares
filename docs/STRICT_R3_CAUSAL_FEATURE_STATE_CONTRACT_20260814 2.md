# Strict-R3 causal feature-state contract

## Purpose

The canonical feature producer must not reconstruct years of history for each
live hour or training fold. Its causal operator state is part of the model
contract. Online inference and chronological training therefore restore and
advance the same immutable, hash-bound state representation.

## Persisted state layers

The current challenger persists:

1. 121 primitive rolling operators covering sum, mean, standard deviation,
   minimum, maximum, percentile rank, robust z-score and recursive EWMA.
2. The 720-row causal feature-transform state.
3. Exact bounded histories for the parents consumed by residual/beta/surprise
   features.
4. Exact bounded histories for the transformed parents consumed by
   cross-sectional regime/eigen composites.
5. The frozen market-spectral source contract and its exact rolling source
   history.
6. The 180-day OI robust-normalization source needed by the frozen OI geometry.
7. Four fixed-FFD states: recursive EWMA input buffers, exact convolution
   buffers, and 768 rows of transformed output history for `close` and
   `close_d04/d05/d06`.

State is append-only by timestamp and symbol. A gap, feature-contract change,
symbol-order change, or stage mismatch fails closed.

## Publication and failure semantics

An hourly/chunk advance is transactional. The producer copies the last
canonical cache into a private working directory, advances every state there,
writes and audits the feature output, and only then replaces the canonical
cache. A failed process cannot publish a mixed-hour cache.

Immutable bundles contain independent file copies, not hardlinks to a mutable
cache. Every state file, the source-panel checkpoint, the feature contract and
the relevant implementation files are SHA-256 bound in the manifest.

Schema v2 embeds the source-panel checkpoint in the bundle instead of merely
recording an external path.  The embedded panel is causally truncated; longer
memory remains in the serialized operators. The current audited bound is 1,536
hours because the OI/price block nests a 15-day realized-volatility window in a
30-day robust normalizer.  A restore
therefore has everything required to resume, while avoiding a full-history
panel load.  The original source-panel hash and the hash of the bundled tail
are both retained.

## Online use

The live producer restores the latest state bundle once, appends only the new
point-in-time source row, computes the current 170-symbol feature matrix, and
publishes a new immutable bundle after the checkpoint passes. The current
feature row is never computed from future or revised historical values.

Missing contemporaneous order-book inputs remain missing through feature
generation. The explicitly authorised trade-size proxy is filled only from
the same-timestamp complete-universe median after source missingness has been
preserved.

## Training use

The same producer supports `--emit-all-candidate-timestamps`. A chronological
training chunk restores the preceding state bundle, advances all new rows in
one vectorized pass, emits every complete-universe timestamp in that chunk,
then snapshots the ending state. This avoids hourly recomputation while
retaining the exact online recurrence and feature semantics.

Training chunks must remain chronological. They may not restore a state whose
last timestamp is later than the first row in the chunk, and every emitted
timestamp must contain the complete frozen universe.

This makes offline materialisation the same operation at a different batch
size: restore one state bundle, append a chronological block, emit all rows in
that block, and snapshot the ending state for the next block.  No feature is
recomputed from inception at a fold boundary.

## Current immutable receipt

- Bundle: `data_perp/artifacts/strict_r3_causal_feature_state_bundle_20260814T130000Z_v7_all_state_sourcebacked`
- State timestamp: 2026-08-14 12:00 UTC signal row
- State files: 131
- State size: 100,440,770 bytes
- Frozen feature contract SHA-256:
  `12672f92789107fab4c9ab76a20c0c6504e8adce215b4a7f3fc83171dc5705c4`
- Sealed source-panel SHA-256:
  `e04bde4d5452dae88fde92893172977195638679ad68f89ed71405ff27005bcd`
- State inventory SHA-256:
  `59d45fd88ad7fde0ba5f52de7c32f7ac1194099b2e619c3798e7ba4b55cc30e8`

The successor self-contained bundle is:

- Bundle: `data_perp/artifacts/strict_r3_causal_feature_state_bundle_20260814T130000Z_v8_selfcontained_tail768`
- Schema: `strict_r3_causal_feature_state_bundle_v2`
- Embedded source tail in this first v2 receipt: 768 hours, 7.2 MB (versus 42 MB
  for the source checkpoint). This bundle is superseded for promotion by the
  1,536-hour successor because the exact audit exposed the nested OI/price
  lookback requirement.
- Embedded source-tail SHA-256:
  `63f2226bb1b947884e62d7367d43b72d6a0a92444c989431c15788bf445b6975`
- Original source-panel SHA-256:
  `e04bde4d5452dae88fde92893172977195638679ad68f89ed71405ff27005bcd`

The promotion challenger is:

- Bundle: `data_perp/artifacts/strict_r3_causal_feature_state_bundle_20260814T130000Z_v9_selfcontained_tail1536_maskfix`
- Embedded source tail: 1,536 hours, 15 MB
- Embedded source-tail SHA-256:
  `c4cd2f094897a2b590bd4f8c0e7d1d8794a82b4cd4bd4223b7759971f0fdfda3`
- State inventory SHA-256:
  `225980ec2a6b92b9c9241c8afb541612c05b0373542a6eedef584ebcefb2c77b`
- The order-book carry at the beginning of the bounded tail is the exact last
  causally observed pre-tail value, matching the canonical feature adapter's
  forward-fill contract. Current availability is checked at the shifted source
  timestamp actually consumed by the feature.

## Validation status

Twenty-one focused state, fixed-FFD, restore, transaction, OI-lineage and
multi-timestamp extraction tests pass. The fixed-FFD append agrees with its
full-history formula within `2e-6` and refuses timestamp gaps. A real
source-backed one-hour advance over the 170-symbol universe completed in
approximately 36 seconds with a 768-hour bounded work tail; the incumbent
full-history feature stage took approximately 179 seconds. All 170 rows met
the frozen 120-field >=90% coverage gate and 95.9% were fully complete.

The following historical hour intentionally had no locally materialised
source primitives. It produced missing features and no actionable rows rather
than carrying the prior hour forward. That missing-source run is retained as a
fail-closed audit and is excluded from the source-backed state chain.

The bundle is not yet bound into the live inference configuration. Promotion
still requires an exact partitioned full-formula comparison (the monolithic
comparator exceeded available memory), followed by model-input/output,
admission and policy parity. The existing live bundle remains unchanged until
that receipt exists.

The dependency-matched 120-field audit is available at
`data_perp/artifacts/strict_r3_partitioned_feature_state_parity_20260814T130000Z_v2_dependency_matched`.
It passes 111 fields and rejects nine. The remaining failures are concentrated
in order-book depth/trade-size reductions, three spectral eigen summaries and
one sparse-symbol Donchian availability edge. This is a failed promotion
receipt, not a tolerance waiver.

A clean historical 11:00 -> 12:00 UTC append was then used to distinguish a
same-timestamp reconstruction from a real state advance. The append completed
in approximately 46 seconds versus approximately 179 seconds for the prior
full-history feature stage. It also proved that a compact state must persist
every nested derived boundary, not only the raw rolling operators: a clean
bootstrap followed by one append still changed 32 of 120 final fields beyond
the 0.01% gate. The emitted newest row now uses the canonical bounded batch
kernel for fixed-window sparse-prefix operators while advancing the compact
state transactionally. This repair passes the focused suite but is not yet a
promotion artifact; additional derived-state boundaries must be migrated and
the 120-field audit rerun.

### Nested-state schema-v2 challenger

A dependency-scoped nested-history store was added for the requested feature
closure. It uses SQLite rows containing compressed float32 symbol vectors and
commits inside the same state transaction. Hourly inference and chronological
training chunks append only newly computed timestamps.

The legacy feature graph must retain an exact built-in `dict`; replacing it
with a mapping subclass caused native-kernel failures. Restoration is therefore
performed at named graph checkpoints. Early seed/technical restoration is
excluded because those legacy kernels require their original construction
path. Safe restoration currently occurs after base, gated, change-point,
composite, position, regime and final stages.

Historical validation used a clean full-history bootstrap through 11:00 UTC,
then a state-only append of 12:00 UTC. The safe late-stage append completed and
the core feature graph took approximately 76 seconds. The post-feature source
repair remained serial and dominated the remaining runtime. The immutable
comparison receipt is
`data_perp/artifacts/strict_r3_partitioned_feature_state_parity_20260814T130000Z_v3_late_stage_state`.
It still rejects 32 of 120 fields (six missingness mismatches and 3,519 finite
rows beyond the 0.01% relative gate). This challenger is not promoted.

This falsifies history overlay as a complete repair. The frozen dependency
closure needs an explicit operator DAG whose nodes expose
`bootstrap/update/snapshot`, so each current-row formula advances from its own
sufficient state. The incumbent live bundle remains unchanged.

### Exact self-contained golden comparator

The post-materialisation repair now consumes the already loaded immutable
point-in-time source panel. It no longer reopens the 170 mutable source
archives after the feature graph, removing duplicate I/O and preventing the
same run from mixing two source vintages.

Until each remaining operator family has a native incremental implementation,
the hybrid comparator computes the declared long-memory closure on the sealed
complete panel and the remainder on the append-state tail. The resulting
candidate row is exact against a reference whose post-feature repairs were
regenerated from that same sealed panel:

- output: `data_perp/artifacts/strict_r3_incremental_features_20260814T130000Z_v44_hybrid_exact_complete_selfcontained`
- reference: `data_perp/artifacts/strict_r3_selfcontained_repair_reference_20260814T130000Z_v2_allrepair/partitioned_exact_features.parquet`
- audit: `data_perp/artifacts/strict_r3_hybrid_exact_selfcontained_parity_20260814T130000Z_v2_allrepair`
- frozen fields: 120
- candidate rows: 170
- missingness mismatches: 0
- rows beyond the 0.01% relative gate: 0
- result: pass

This establishes an exact, causal golden fallback. It is not the promoted fast
path: its approximately 174-second runtime is intentionally retained only as
a comparator while the allow-list is replaced family by family with native
`bootstrap/update/snapshot` operators. Every migrated family must match this
comparator exactly before its fallback is removed.

## Canonical state-DAG direction

Persisting only final features is insufficient. The schema-v3 state contract
will persist every causal boundary needed to advance the frozen closure:

1. source availability and last causally observed primitive values;
2. fixed-window buffers and sufficient statistics;
3. expanding/recursive estimators, including EWMA and online moments;
4. derived-parent histories used by residual, surprise and barrier features;
5. contemporaneous complete-universe reductions and spectral state;
6. final feature rows and their coverage masks;
7. model preprocessing, score-reference, Geometry/K9, calibration, admission,
   portfolio and open-position/exit state.

Each node has a stable identity, dependency hashes, symbol order, timestamp,
lookback/warm-up declaration and `bootstrap/update/snapshot` implementation.
Publication is atomic across the complete DAG. Training restores the same
snapshot and advances a chronological block; live inference advances one row.
No separate training-only feature implementation is permitted.
