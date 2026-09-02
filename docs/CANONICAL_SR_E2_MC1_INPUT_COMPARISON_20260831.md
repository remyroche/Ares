# Causal S/R and E2 Inputs to the Retained Dual-MC1 Map

## Decision

This replaces the earlier narrow paired-residual S/R/E2 diagnostic as the
valid comparison.  That earlier diagnostic is retained as historical context
only: it was not the retained absolute-EV, dual-MC1 admission contract and
must not be used to assess the live/current model.

No S/R or E2 arm is promoted.  The S/R-only arm improves total contribution,
worst week and drawdown relative to the matched refit control, but gives up
substantial EV/trade and has only two already-inspected evaluation months.
The combined arm is more balanced than S/R-only but still has lower EV/trade
than the matched control.  Both require a later, frozen test before any
canonical change.

## Fixed contract

The study keeps these invariants fixed for every arm:

- Score families: the retained August-17 BCF and current-v5 MC1 score
  surfaces, separately mapped.
- MC1 target/model: family-specific absolute rich `policy_net_bps`, p02/p98
  target clipping, retained HGB (`depth=2`, `80` iterations, `0.04` learning
  rate, `L2=20`, `min_leaf=100`, seed `1729`), plus the retained 21-day,
  prior-resolved, 10%-trimmed score-band residual shift.
- Admission: `BCF MC1 >= +50 bps AND current-v5 MC1 >= +50 bps`.
- Auction: BCF mapped EV priority; existing controlled global `7x`, 10%
  margin-slot, two-new-entry, eight-concurrent, 80%-wallet portfolio replay.
- Outcomes: source-aligned rich-policy outcomes; invalid/unresolved paths are
  excluded before capacity and never create pseudo-trades.
- Scope: June--July 2026.  E2 receives February--May causal history and is
  fit monthly from prior resolved labels only.  This is a walk-forward research
  evaluation, not a new untouched promotion period.

The pre-February BCF warm-up score ledger used by the frozen reference was
pruned.  Therefore the frozen retained map is reported for context only.  All
input-arm deltas use `C0_refit_core_postfeb`, which is the common post-February
refit substrate shared by C1--C3.

## Inputs

| Arm | Additional mapper inputs | Authority |
|---|---|---|
| C0 | None | Matched core-only refit control |
| C1 | Eleven causal S/R outputs plus availability | Refit both absolute MC1 maps |
| C2 | Causal direct 15-minute E2 expected policy EV plus availability | Refit both absolute MC1 maps |
| C3 | Both C1 and C2 fields | Refit both absolute MC1 maps |

S/R is causal OOF state.  It is not a gate: missing rows remain candidates and
are represented by missing inputs plus the availability flag.  Its availability
in the held current-v5 score surface is 5.0% in June and 11.3% in July.

E2 is a separate causal LightGBM L1 proxy (`depth=4`, `15` leaves, 350 trees,
`lr=0.03`, `L2=4`, seed `1729`) trained monthly on the preceding four months'
resolved rich-policy labels.  It consumes the full 70-field target-free
15-minute contract.  E2 availability is 68.1% in June and 90.3% in July;
unavailable values stay missing with an explicit availability flag.  The raw
15-minute cache contains 260,884 `ok` rows and 12,815 locally unreadable-source
rows; no values were imputed.

## Constrained portfolio result

All figures below are realised rich-policy net bps after the fixed portfolio
constraints.

| Arm | Accepted trades | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Frozen retained C0 (context) | 480 | +158.71 | +76,183 | +155.48 | +0.37 | -42.94% |
| C0 refit core (delta baseline) | 183 | +264.05 | +48,322 | +227.92 | +72.96 | -37.27% |
| C1 core + causal S/R | 438 | +188.09 | +82,385 | +152.87 | +105.49 | -31.69% |
| C2 core + 15m E2 | 298 | +186.97 | +55,718 | +152.86 | +120.92 | -6.32% |
| C3 core + S/R + 15m E2 | 383 | +201.23 | +77,072 | +161.28 | +111.81 | -33.30% |

Relative to the matched C0 refit, C1 adds 255 accepted trades and +34,064 net
bps, improves worst-week EV by +32.53 bps/trade and reduces drawdown by 5.58
percentage points, but gives up 75.96 bps/trade.  C3 adds 200 accepted trades
and +28,750 net bps, improves worst-week EV by +38.85 bps/trade and reduces
drawdown by 3.97 points, while giving up 62.82 bps/trade.  C2 is the strongest
drawdown result, but its +7,396 total-bps increment is modest and it also loses
77.08 bps/trade.

| Arm | June trades / EV | July trades / EV |
|---|---:|---:|
| Frozen retained C0 | 275 / +161.13 | 205 / +155.48 |
| C0 refit core | 135 / +276.90 | 48 / +227.92 |
| C1 core + causal S/R | 166 / +245.81 | 272 / +152.87 |
| C2 core + 15m E2 | 143 / +223.95 | 155 / +152.86 |
| C3 core + S/R + 15m E2 | 158 / +258.12 | 225 / +161.28 |

## Causality and reproducibility

- The direct E2 feature cache was regenerated from the full target-free union
  of 273,699 February--July candidate identities before any outcomes were
  joined.  It has no future-path qualification.
- For E2, each held-month fit requires both decision time and label-availability
  time to precede the held month.  February/March have insufficient prior
  support and are explicitly unavailable; April--July are prequential.
- Each MC1 monthly fit uses only post-February rows with labels available before
  that held month.  The 21-day shift also uses only labels resolved before its
  decision-day bucket.
- Target-free admission panels contain no policy/outcome columns; a final audit
  verified exact candidate uniqueness, finite dual-map outputs, and exact
  dual-50 admission logic for every arm.
- The run has no exchange-writing or network calls.

## Artifacts

- Runner: `scripts/run_canonical_sr_e2_mc1_input_ablation.py`
- Contract test: `tests/test_canonical_sr_e2_mc1_input_contract.py`
- Immutable result: `data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v5`
- Primary outputs: `portfolio_summary.csv`, `monthly_metrics.parquet`,
  `mc1_fold_audit.parquet`, `e2_prequential_audit.parquet`, and
  `run_manifest.json` in that result directory.
