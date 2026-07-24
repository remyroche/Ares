# R0 Missing Absolute-Path Triage

Date: 2026-07-24
Source: `r0/migration_inventory.json`

The P0 audit found all 14 configured artifact roots and successfully inspected
58 representative files. It also found 76 absolute paths that no longer exist.
They are not treated as a successful R0 gate.

## Active Pack-B lineage paths — blocking

Six missing references occur in the Pack-B final bundle:

- the source Pack-B report directory;
- the source AE/GMM state and manifest;
- the source frozen AE/GMM outputs.

These missing references agree with the independent R3 audit: the existing
bundle cannot prove its complete own-side OOF and frozen-state lineage. Do not
rewrite the historical manifest or substitute similarly named artifacts. Either
recover the exact hashed sources or regenerate Pack-B and residual OOF under the
new stage-manifest contract.

## Historical temporary feature-block paths — non-canonical

The other 70 findings are `/tmp/ares_july16_replay/...` paths embedded in
historical static-feature-block manifests. Their containing feature store is
present and hashed, but the original temporary paths are gone. Preserve these
manifests as historical evidence; do not use their stale absolute paths as
canonical upstream provenance.

## Gate decision

R0 remains `IN PROGRESS`:

- all local P0 roots exist;
- 14 deterministic tree checksums were emitted;
- 58 bounded read-only smoke checks passed;
- no cross-machine checksum baseline was supplied;
- active Pack-B lineage references are missing;
- the process audit was separately run with elevated read-only permission and
  found no active Python/LightGBM/CatBoost/Optuna training process.

The missing Pack-B sources trigger regeneration/recovery, not silent path
repair.
