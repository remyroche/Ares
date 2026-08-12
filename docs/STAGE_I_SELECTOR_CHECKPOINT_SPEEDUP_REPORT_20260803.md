# Stage-I selector checkpoint speed repair — 2026-08-03

## Scope

This change applies to future selector materialisations and resumes.  It does
not alter, terminate, restart, or rewrite the selector process/artifacts that
were already running when the repair was implemented.

The downstream MDA selector now also reduces future-round width gradually:
70% survival above 200 fields, then 80% survival down to a coarse floor of 120.
This reduces later-round model/permutation work without changing checkpoint
materialisation or creating fold-local contracts. The minimum-five rule is
evaluated once per complete aggregated round; evidence/protection shortfalls
stop unchanged and are persisted. Details and schema/resume compatibility are
recorded in `STAGE_I_MDA_HPO_SPEEDUP_REPORT_20260803.md`.

## Authoritative bottleneck

The former 96-column checkpoint loop revisited the full symbol surface for
every feature block.  It also split every symbol into requested calendar
months.  The current static store normally has one large Parquet row group per
symbol, so each monthly predicate decoded the same physical row group again.
Consequently the dominant cost was repeated physical store decoding and exact
row routing, not the 80K-row coverage arithmetic.

## New default path

`materialize_stage_i_selector_sample.py` now defaults to `symbol-first`:

1. Freeze unique `(__symbol__, __ts__)` points and one exact candidate-to-point
   integer indexer.  Long/short rows sharing a point are read only once.
2. Plan stable symbol×row tiles and canonical 32-column tiles. Stream only the
   currently scheduled column group over each symbol's requested timestamp
   span; a rows×full-union frame is never constructed. Exact timestamp
   reindexing remains mandatory; there is no as-of, fill, or nearest-time
   lookup.
3. Scatter float32 values into a disk-backed mmap.
4. Flush a tile, compute its identity-bound SHA-256 from the mmap, and then
   atomically commit its record. Every committed tile is rehashed on resume;
   corruption fails closed, while unrecorded crash residue is safely reread.
5. Emit the unchanged immutable feature-block Parquets, coverage evidence, and
   final selector matrix from synchronized bounded Parquet row batches. The
   final writer never constructs rows×the full retained union in memory.
   Public values and hashes retain the existing contract.

The compatibility flag `--feature-load-order block-first` retains the former
execution order for diagnosis.

## Memory and progress safety

- Physical projection width is based on the largest in-scope Parquet row
  group, current process RSS, and the explicit selector memory budget.
- The selector cap is 512 columns.  The generic loader default remains 64.
- The current 222-symbol selector surface has a largest physical row group of
  36,027 rows. A representative full-import scan at 420.0 MiB RSS selects 96
  physical columns/read under the 512 MiB ceiling.
- RSS is checked before projection, immediately after physical decode, and
  before mmap commit. Arrow/Python allocation failure halves the grouped
  projection and resumes from verified tile records, down to 32 columns.
- The mmap is disk-backed; only the current bounded point batch and final
  96-column checkpoint expansion need resident matrix memory.
- Flushed mmap pages receive `MADV_DONTNEED` where supported so completed pages
  do not accumulate in process RSS.
- `symbol_first_cache/progress.json` is atomically updated after each physical
  projection with completed/total tiles and truthful reader/checksum counters.
- The large month-by-side diagnostic aggregate is written once at finalisation
  instead of being rewritten cumulatively after every feature checkpoint.
- Finalisation opens bounded cursors over every public checkpoint, assembles at
  most one RSS-sized row batch in exact contract order, streams it through a
  Parquet writer, and then performs a second bounded readback. Identity plus
  canonical float32 values must have the same streaming SHA-256 before the
  final Parquet is atomically published.
- An exclusive output writer lock rejects concurrent resumes and recovers only
  a provably dead same-host PID. Checkpoint half-bundles are discarded and
  rebuilt; the final manifest is an atomic replace.

## Exactness and restart evidence

The integrated focused suite has 48 passing tests.  It covers:

- legacy month-bounded default behaviour;
- selector-only symbol-span read count and exact restored values;
- duplicate long/short point expansion;
- per-tile checksum verification and tamper failure;
- flush-before-record crash resume;
- cache-contract mismatch/size validation;
- memory-pressure projection fallback and no-headroom rejection;
- exclusive concurrent-writer rejection and stale-PID recovery;
- orphan mmap/checkpoint half-bundles and partial-manifest failure;
- existing block integrity hashes;
- per-symbol chronological warm-up prefixes followed by aggregate gates;
- wide-surface streaming finalisation column/value/hash parity and observed
  RSS below its declared ceiling;
- fallback telemetry using the actual attempted width rather than its wider
  configured ceiling;
- vectorised month/side diagnostic parity.

## Representative current-store scan

The frozen v5 selector/contract was inspected read-only; no v5 artifact or
active experiment was modified:

| Surface | Value |
|---|---:|
| Selector rows | 75,621 |
| Unique exact points | 71,690 |
| Symbols | 222 |
| Stage-I union fields | 1,183 |
| Symbol-month spans | 3,183 |
| Largest physical row group | 36,027 |
| Disk-backed mmap | 0.316 GiB |
| Canonical checksummed tiles | 8,214 |
| Estimated legacy block×month projections | 41,379 |
| Estimated tiled symbol-span projections | 2,886 |

The physical projection count falls by an estimated **14.34x** on the actual
surface, before accounting for the removal of repeated routing and growing
diagnostic rewrites.

A real-store read of the most represented symbol (`CHZ/USD:USD`) over its full
2022-08-31→2026-07-09 requested span projected 96 fields once: 5,731 exact
points, 33,785 physical source rows, 0.165 seconds, and a 107.2 MiB observed
RSS increase in the measurement process. The exact float32 output hash was
`4adb406574c7c82e9bba8a04a51808792996e116252333651e73b8ac9427066e`.

## Synthetic physical-I/O benchmark

A one-row-group store with 36,000 hourly rows, 256 float32 fields, and 36
requested timestamps produced identical output SHA-256
`5207caf33ebcb1bc3c0226b2ed16c84bd9458ff946dd0456b8e71babd99865d0`:

| Read order | Seconds |
|---|---:|
| 36 monthly predicates | 0.833 |
| one symbol-span predicate + exact reindex | 0.039 |

Observed I/O speedup: **21.58x**.  Peak process RSS in the generation + read
benchmark was 457.5 MiB.  Production sizing is more conservative because it
uses current RSS and the actual maximum row-group surface before choosing the
projection width.

The end-to-end gain depends on symbol histories and feature count, but the
current store geometry makes symbol-span traversal the material improvement;
small pandas bookkeeping optimisations alone cannot deliver a comparable gain.
