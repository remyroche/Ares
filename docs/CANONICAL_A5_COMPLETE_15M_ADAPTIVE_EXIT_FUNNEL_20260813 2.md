# Canonical A5 complete-15-minute adaptive-exit funnel — 2026-08-13

> **Superseded as an economic comparison.** This first replay imported the
> simulator before disabling its default stop-gap/spread proxy and therefore
> understated the canonical baseline by about 19 bps on fine-path rows.  It
> also replaced hourly-proxy outcomes with downloaded 15-minute paths.  Use
> `docs/AUTHORITATIVE_A5_ADAPTIVE_EXIT_RECONCILIATION_20260813.md` for the
> corrected source-aligned comparison.  The feature/controller diagnostics in
> this file remain development evidence only.

## Decision

The experiment is complete and remains **unpromoted**.  The 2025 development
winner is `F4_disagreement_abstain_p80`; it remains positive on the untouched
2026 block, but its incremental benefit over the simpler continuous F1
controller is small.  Stop and giveback remain frozen.

## Reproducible command

```bash
python3 scripts/run_canonical_a5_15m_adaptive_exit_funnel.py \
  --out-dir data_perp/artifacts/canonical_a5_15m_adaptive_exit_funnel_20260813_v2 \
  --max-train-states 40000
```

The missing Kraken 15-minute histories were first materialised with
`scripts/download_kraken_15m_hf.py`.  All 53,282 canonical A5 admitted long
candidate IDs are retained.  Future-path availability is used only for outcome
evaluation, not candidate selection or admission.

## Frozen policy and decision contract

- Entry: first 15-minute open one hour after the signal close.
- Controller clock: after completed hourly blocks (15-minute bars 3, 7, ..., 47).
- A new activation is effective on the next 15-minute bar.
- Stop: 4.152000643 ATR, frozen.
- Baseline trailing activation: 2.326224920 ATR.
- Giveback: 0.102371990 ATR, frozen.
- Timeout: H12.
- Cost: 100 bps exactly once.
- ATR: frozen canonical decision-time `policy_atr`; the subsequent path source
  cannot replace it.  A regression test protects this lineage.

The rebuilt downloaded-15-minute policy outcome differs from the stored mixed
exact/15-minute ledger because the price-source resolution and vintage differ.
Across all 53,282 IDs the difference has -20.97 bps bias, 23.20 bps MAE,
15.98 bps median absolute error and 33.59 bps p95 absolute error.  On rows whose
stored source is exact/15-minute, MAE is 18.68 bps; on historical hourly-proxy
rows it is 31.47 bps.  All challenger comparisons below are internally paired
on the newly rebuilt path and policy contract.

## Models and validation

The opportunity head keeps the 28-field causal path contract and predicts the
65th quantile of remaining favourable excursion from entry.  The F4 context
head adds entry trust/context and score-evolution fields, but remains a small
depth-3, seven-leaf model.  It is used in two ways:

1. an inner-OOF failure classifier, trained only after a chronological 65/35
   split and a 12-hour purge inside each outer training block;
2. an F1-versus-F4 forecast-disagreement veto, whose p80 threshold is fitted on
   the inner-OOF block.

Outer evaluation uses chronological three-month blocks, at most the prior nine
months, a 12-hour purge, and deterministic equal-month subsampling capped at
40,000 states.  The selection period is 2025; 2026 is untouched confirmation.

Stable archetype roles are learned from train-only bins of causal path state:
PnL, giveback, trade age, time since MFE, velocity and trailing state.  The
outputs are expected uplift, downside probability, effective support,
uncertainty and confidence with 200-row hierarchical shrinkage.  No raw cluster
ID or unstable K9 coordinate is exposed.

## Candidate-level OOF results

All figures are net bps per trade on the same candidate IDs and rebuilt
15-minute outcomes.

| Arm | 2025 uplift | 2026 uplift | Comment |
|---|---:|---:|---|
| F4 disagreement abstention p80 | +28.33 | +14.55 | 2025-selected winner |
| Continuous 0.75, tighten-only | +28.33 | +14.42 | Almost all useful edge without F4 |
| Continuous 0.75, early-50/late-25 | +28.29 | +14.30 | Asymmetric-bound challenger |
| F4 failure-probability continuous shrink | +28.25 | +14.05 | No material gain |
| Stable archetype hierarchical shrink | +8.37 | +4.48 | Very stable but over-shrunk |

The winner improves 64 of 68 evaluated weeks, has three negative-uplift weeks
and one zero week.  Median weekly uplift is +17.32 bps; the worst is -8.58 bps.
The archetype shrinker improves 63/68 weeks and limits the worst week to -1.15
bps, but sacrifices too much mean uplift.

The failure classifier is not a useful selector under the present definition:
OOF failure probability has median 0.0013, p90 0.0233 and maximum 0.1112, so the
0.5/0.6/0.7 abstention arms never bind materially.  The disagreement veto is
the only F4 mechanism that adds value, and its increment over the continuous
tighten-only arm is only +0.0005 bps in 2025 and +0.13 bps in 2026.

### Winner by month

| Month | Baseline net | Adaptive net | Uplift |
|---|---:|---:|---:|
| 2025-04 | 200.58 | 219.39 | +18.81 |
| 2025-05 | 44.32 | 85.84 | +41.52 |
| 2025-06 | -6.37 | 29.82 | +36.19 |
| 2025-07 | 86.21 | 118.30 | +32.08 |
| 2025-08 | 74.63 | 104.81 | +30.17 |
| 2025-09 | 54.86 | 70.79 | +15.93 |
| 2025-10 | 171.34 | 190.56 | +19.22 |
| 2025-11 | 120.43 | 171.61 | +51.18 |
| 2025-12 | 91.27 | 106.53 | +15.26 |
| 2026-01 | 207.63 | 220.65 | +13.02 |
| 2026-02 | 280.29 | 314.60 | +34.32 |
| 2026-03 | 84.80 | 99.68 | +14.89 |
| 2026-04 | 136.27 | 153.76 | +17.50 |
| 2026-05 | 82.20 | 91.16 | +8.96 |
| 2026-06 | 12.46 | 14.01 | +1.55 |
| 2026-07 | 133.08 | 130.25 | -2.82 |

## Capacity-aware portfolio replay

The auction is rerun from scratch using each arm's realised adaptive exit
timestamp.  Released capacity therefore changes later accept/reject decisions;
this is not a fixed-trade-ID portfolio comparison.

| Metric | Frozen baseline | F4 p80 | Change |
|---|---:|---:|---:|
| Accepted trades | 8,495 | 8,710 | +215 |
| Trades/day | 14.79 | 15.17 | +0.37 |
| Net bps/trade | 138.67 | 158.23 | +19.56 |
| Positive-trade rate | 62.66% | 66.76% | +4.10 pp |
| Sortino | 0.384 | 0.475 | +0.091 |
| Worst week, wallet return | -46.12% | -25.59% | +20.53 pp |
| Max drawdown | -79.69% | -79.69% | effectively unchanged |
| Mean open positions | 6.54 | 6.48 | -0.06 |
| Full-stop rate | 8.10% | 3.31% | -4.79 pp |
| Timeout rate | 42.20% | 36.06% | -6.14 pp |

The wallet compounds at seven-times leverage and therefore reaches meaningless
astronomical nominal values over the long replay.  Net EV/trade, acceptance,
drawdown and normalized risk statistics are the decision-useful measures; final
wallet is not.

The adaptive portfolio is better in every month from April 2025 through June
2026, but July 2026 is -4.11 bps/trade worse than baseline after the auction.
January-March 2025 use fail-closed baseline exits because they precede the first
outer OOF adaptive fold.

## Interpretation and next decision

Continuous activation-ATR authority is the robust improvement.  The best
simple arm is 75% shrink with asymmetric tighten-only bounds: the controller may
lower activation to 0.5 times baseline but may not raise it above baseline.
F4 disagreement abstention is directionally portable, but too incremental to
justify promotion yet.  The failure label should next be changed from the rare
strict-negative intervention event to a predeclared minimum-benefit hurdle or
an ordinal harmful/neutral/helpful target.  The stable archetype outputs should
be used as a low-authority risk veto or blended with continuous authority rather
than directly scaling the whole intervention.

Promotion requires a later frozen forward period.  This run does not alter the
canonical inference bundle.

## Artifacts

- Runner: `scripts/run_canonical_a5_15m_adaptive_exit_funnel.py`
- Continuous replay: `extreme_price_movements/path_based_exit_optimisation.py`
- Results: `data_perp/artifacts/canonical_a5_15m_adaptive_exit_funnel_20260813_v2`
- OOF replay: `oof_replay.parquet`
- F4/archetype diagnostics: `oof_context_and_archetype_roles.parquet`
- Capacity-aware decisions: `portfolio_BASELINE/` and
  `portfolio_F4_disagreement_abstain_p80/`
- Manifest: `run_manifest.json`
