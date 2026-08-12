# TP6/SL4 rolling archetype + GAM month-ahead handover

## Decision

The frozen-development replay was replaced with an inference-compatible
month-ahead replay. For every target month, the archetype catalogue, co-firing
clusters, context fields, and GAM coefficients are fitted using only the
immediately preceding 1, 2, or 3 available model months. The target month is
then scored once and ranked globally.

This is a valid transport protocol, but it does not yet produce a broadly
portable production improvement. The one-month window is the only candidate
for further extreme-tail research; its advantage is concentrated at the top
0.5--1% and disappears or reverses at top 2--10%. The two- and three-month
windows do not improve the normal top-5/top-10 operating tails.

## Exact protocol

For target month `t`:

1. Load only the preceding `window_months` available model months. December
   2024 is absent from the source, so windows use preceding *available*
   months rather than assuming a contiguous calendar month.
2. Discover local structural archetypes from those training months. A
   recurrence contract is used when it yields at least two archetypes; for
   short windows with insufficient recurrence support, the strongest local
   structural paths are used and marked `local_top_paths`.
3. Match training and target catalogues to the same local archetype IDs with
   soft top-3 matching and an unmatched-mass threshold.
4. Select co-firing clusters from training rows only. The selected contract is
   gated by balance, per-period support, mapping quality, and transport score.
5. Select context fields by weighted conditional-MI on training rows only.
6. Fit the zero-at-exposure varying-coefficient GAM on the training residual
   (`realised net bps - base expected bps`). An intercept GAM is retained as a
   diagnostic arm.
7. Score the target month, apply the GAM only when its local cluster contract
   passes the transport gate, and otherwise fall back to the base expected-bps
   score. Ranking is global after score generation.

The GAM score is therefore a score conditioner, not a replacement for the
base model. All reported economics are TP6/SL4 net bps/trade under the saved
exit-policy labels.

## Coverage and transport

| preceding months | target months | valid cluster contracts | valid rate | average archetypes | average target matched mass | average unmatched mass |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19 | 11 | 57.9% | approximately 64 (capped) | approximately 0.65 | approximately 0.35 |
| 2 | 18 | 18 | 100.0% | approximately 7 | approximately 0.49 | approximately 0.51 |
| 3 | 17 | 13 | 76.5% | approximately 42 (higher in fallback months) | approximately 0.55 | approximately 0.45 |

The one-month window has more local paths but less genuine cross-month
recurrence. The three-month window has more stable recurrence when available,
but several target months still have too little transport support. Invalid
contracts are not silently applied: the production-style score falls back to
the base score for those months.

## Production-style gated economics

The table below is pooled over the eligible target months for each window. The
base arm is the unchanged global base ranking. `GAM-gated` means that the GAM
is used only for a transport-valid local contract; otherwise the base score is
used. Values are net bps/trade.

### One preceding month

| global tail | base | gated GAM, gamma .25 | gated GAM, gamma .50 | gated GAM, gamma 1.00 | positive months (base → .25) |
|---:|---:|---:|---:|---:|---:|
| 0.5% | -19.31 | **+54.13** | **+54.25** | **+50.36** | 12 → 13 |
| 1% | -5.31 | **+33.08** | **+37.70** | **+35.19** | 13 → 12 |
| 2% | -7.11 | -3.05 | -6.35 | -11.44 | 8 → 9 |
| 5% | -6.87 | -13.94 | -13.60 | -19.03 | 9 → 6 |
| 10% | -10.21 | -5.71 | -9.86 | -18.27 | 9 → 7 |

### Two preceding months

| global tail | base | gated GAM, gamma .25 | gated GAM, gamma .50 | gated GAM, gamma 1.00 |
|---:|---:|---:|---:|---:|
| 0.5% | -18.34 | **+0.54** | -4.46 | -8.03 |
| 1% | -6.28 | -40.88 | -40.68 | -57.85 |
| 2% | -5.16 | -48.00 | -53.09 | -69.22 |
| 5% | -7.82 | -34.24 | -39.55 | -44.34 |
| 10% | -13.43 | -11.68 | -15.99 | -20.43 |

### Three preceding months

| global tail | base | gated GAM, gamma .25 | gated GAM, gamma .50 | gated GAM, gamma 1.00 |
|---:|---:|---:|---:|---:|
| 0.5% | -26.24 | -34.66 | -33.13 | -73.51 |
| 1% | -5.63 | -12.72 | -12.19 | -32.07 |
| 2% | -5.71 | **+4.88** | -1.36 | -1.98 |
| 5% | -6.11 | -18.62 | -13.61 | -17.85 |
| 10% | -12.32 | -16.96 | -17.07 | -16.51 |

These are not claims of a positive production strategy. The apparent
one-month extreme-tail uplift is unstable: it does not survive the wider
operating tails, and the worst target month remains materially negative.

## Diagnostic raw-GAM result

Applying the raw GAM even when its local contract fails the transport gate is
an intentionally non-production diagnostic. It makes the failure mode clear:

- one-month raw zero-at-exposure GAM: +82.14 bps at top 0.5% and +29.03 bps at
  top 1%, but -19.18 bps at top 5% and -11.42 bps at top 10%;
- two-month raw zero-at-exposure GAM: +0.55 bps at top 0.5%, then -40.88,
  -48.00, -34.24, and -11.68 bps at top 1%, 2%, 5%, and 10%;
- three-month raw zero-at-exposure GAM: -68.48, -42.65, -0.15, -19.60,
  and -16.38 bps at top 0.5%, 1%, 2%, 5%, and 10%.

The raw result must not be promoted because it applies locally untransported
contracts. The gated result is the relevant inference analogue.

## Matched-month comparison

The 17 target months for which all three windows are available are
2024-07--2024-11 and 2025-01--2025-12. On this identical target-month set,
the base is -26.24 / -5.63 / -5.71 / -6.11 / -12.32 bps at the 0.5 / 1 / 2 /
5 / 10% tails. The gated zero-at-exposure GAM at gamma .25 is:

| preceding months | top 0.5% | top 1% | top 2% | top 5% | top 10% |
|---:|---:|---:|---:|---:|---:|
| 1 | **+60.84** | **+34.74** | -3.58 | -15.04 | -8.29 |
| 2 | -5.26 | -45.62 | -43.77 | -30.44 | -10.73 |
| 3 | -34.66 | -12.72 | **+4.88** | -18.62 | -16.96 |

Thus the one-month result is not merely caused by having a different set of
target months; it still shows a narrow extreme-tail uplift on the matched
set. It also still fails the normal top-5/top-10 requirement.

## Correctness audit

The saved correctness report confirms:

- target-month-only scoring: true;
- training window lengths: 1, 2, and 3 months;
- future target rows used in fit: false;
- future target paths used in target archetype fit: false;
- context selection training-only: true;
- global ranking after score generation: true;
- principal GAM is zero at zero archetype exposure: true;
- candidate IDs are unique and scores finite: true;
- failed transport months fall back to the base score: true.

## Artifacts

Authoritative run:

`data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/`

Key files:

- `rolling_oof_predictions.parquet` — target-month predictions and scores;
- `rolling_fit_audit.parquet` — training windows, archetype mode, coverage,
  cluster validity, and selected context counts;
- `rolling_archetype_audit.parquet` — local matching and unmatched mass;
- `rolling_cluster_audit.parquet` — cluster support and context selections;
- `rolling_metrics.parquet` — raw diagnostic tails;
- `rolling_gated_metrics.parquet` and `rolling_gated_summary.parquet` —
  inference-style gated tails;
- `correctness_test_report.json` — leakage and scoring-contract checks;
- `run_manifest.json` — source lineage and artifact inventory.

## Recommendation

Do not replace the current base ranking with this GAM globally. If the
extreme-tail use case is strategically important, carry forward only the
one-month window as a narrowly scoped research arm, with an explicit minimum
transport/support gate and a separate top-0.5--1% policy. For the normal top
5--10% operating range, the next work should improve the local structural
contract (matched archetype coverage and stable cross-month views) before
increasing GAM complexity or widening the fitting window.
