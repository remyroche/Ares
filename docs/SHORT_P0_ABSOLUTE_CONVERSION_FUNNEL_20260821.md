# Short P0/F90 absolute-conversion funnel

**Date:** 2026-08-21  
**Scope:** short only. The long base, consensus, MC1/BCF, Geometry/K9 and live
paths were not modified.  
**Status:** completed research funnel; **no short model promoted**.

## Decision

The short P0/F90 base is an adequate *within-hour relative ranker*, but the
evidence does not support a short LambdaRank consensus at this stage.  The
tested first meta therefore predicts absolute policy opportunity for P0's
hourly winner, without reranking it.

The ordinal absolute model **M4** is a credible 2025--26 research challenger:

| Strict-OOS block | M4 causal train-top-20 | Trades | Net bps/trade | Worst month | Positive months |
|---|---:|---:|---:|---:|---:|
| 2025 Jan--Jun | yes | 1,394 | +90.59 | +30.52 | 6/6 |
| 2026 Jan--Jul | yes | 455 | +92.60 | +6.07 | 7/7 |

It does **not** advance to canonical short admission because the same family
failed the earlier 2024 May--December block (M4: -13.07 bps/trade, 4/8 positive
months).  This is an era-transport failure, not evidence that later metrics
are non-causal.  The required next research question is why the causal
absolute relation changes across the 2024 versus 2025--26 score/market
environment; it is not another broad short consensus search.

## Causal contract

1. Start from every target-free, feature-eligible short P0/F90 candidate.
2. Select P0 rank 1 within each decision hour before any outcome is joined.
3. Build winner score geometry, market state and recent conversion state at
   that decision timestamp.
4. For each held month, train only from rows with
   `policy_label_available_at < held_month` and `decision_ts < held_month`.
5. Fit the score-to-policy-bps map only on chronological OOF training
   predictions. Held outcomes never fit a model, calibration map, threshold,
   or feature.
6. Evaluate only valid, fully resolved policy paths after selection. Invalid
   paths are excluded from supervised fitting and outcome metrics; they are
   never silently encoded as economic failures.

The target is the canonical short parent-policy net outcome,
`p0_canonical_net_bps`, clipped to [-500, +500] bps for supervised fitting.
The policy itself is unchanged by this research.

The old +262-bps oracle cannot be used as a strict P0/F90 benchmark: its
candidate identities only overlap the new ledger partially because its
selection was built from an older score/decision convention.  On the current
strict ledger the ex-post top-20% P0-winner headroom remains large, but it is
not reported as deployable performance.

## Feature contract

All inputs are existing, target-free short P0/F90 fields.

| Block | Fields | Purpose |
|---|---:|---|
| Base/winner | 41 | P0 score/rank/anchor plus winner-specific price, OI, funding, liquidity, volatility, support/resistance and recovery context |
| Score geometry | 21 | top-1 gaps, score spread/MAD/IQR/entropy, tail slope, near-top fractions and rank-tail counts |
| Market state | 29 | returns, breadth, volatility, OI/funding, liquidity and structural-state context |
| Recent conversion | 34 | 7/14/28/56-day strictly resolved EV/hit/risk proxies plus shrunk session/direction/volatility conditional states |

Recent state is deliberately strict: an outcome contributes only when
`policy_label_available_at < decision_ts`.  The initial short funnel found no
incremental value from adding score geometry, market state, or recent
conversion state to the best base-only absolute model.

## M0--M8 model funnel

All non-anchor arms use LightGBM with 160 trees, learning rate 0.035, depth 3,
15 leaves, minimum child 35, 0.85 row/feature fractions, L1 0.10, L2 4.0, and
chronological OOF isotonic calibration to policy net bps.  Classification and
ordinal arms use balanced class weights.

| Arm | Target / role |
|---|---|
| M0 | Existing causal P0 policy anchor |
| M1 | Direct Huber policy net |
| M2 | `P(policy_net > 0)` mapped to bps |
| M3 | `P(policy_net > +100)` mapped to bps |
| M4 | Six-class ordinal net margin, mapped to bps |
| M5 | Causal base anchor + Huber residual |
| M6 | M5 + score geometry |
| M7 | M6 + market state |
| M8 | M7 + recent conversion state |

M4 classes are formed with frozen edges
`[-300, -100, +50, +150, +300]` bps and representative values
`[-400, -200, -25, +100, +225, +400]` bps.

### Causal train-top-20 results

| Arm | 2024 May--Dec | 2025 Jan--Jun | 2026 Jan--Jul |
|---|---:|---:|---:|
| M0 anchor | -10.22 | +43.25 | +22.20 |
| M1 direct Huber | -22.91 | +35.07 | -13.60 |
| M2 P(net > 0) | -21.96 | +73.20 | +14.59 |
| M3 P(net > 100) | **-8.35** | **+94.34** | +52.41 |
| M4 ordinal | -13.07 | +90.59 | **+92.60** |
| M5 anchor + residual | -19.74 | +40.36 | +16.50 |
| M6 + geometry | -26.24 | +40.36 | +16.07 |
| M7 + market | -24.90 | +40.98 | +16.07 |
| M8 + recent state | -19.70 | +40.68 | +13.73 |

Values are net bps/trade.  M3 is strongest in 2025, but its 2026 worst month
is -45.03 bps.  M4 is the only candidate with positive months throughout both
later blocks and positive worst months in both (+30.52 and +6.07 bps), but the
2024 failure prevents promotion.

For reference, M4's 2025/2026 average causal target quality is:

| Block | Spearman with realised policy net | AUC net > 0 | AUC net > +100 |
|---|---:|---:|---:|
| 2025 Jan--Jun | 0.1380 | 0.5669 | 0.5960 |
| 2026 Jan--Jul | 0.1169 | 0.5708 | 0.6038 |

## Top-K continuation

Only M4 passed the later-era hourly gate, so the narrowly predeclared top-K
continuation was run.  It uses the **exact persisted M4 OOF score and
calibration source** for the hourly gate.  The source hash is recorded in each
run manifest.  Isotonic p80 plateau ties use a recorded 1e-4-bps serialization
tolerance; it is solely a representation tie rule, not an economic threshold
relaxation.

| Arm | Selection rule |
|---|---|
| A | Frozen M4 hourly gate, then P0 rank 1 |
| B | P0 top 4, candidate absolute residual filter, retain P0 order |
| C | Frozen M4 hourly gate, P0 top 4, candidate residual filter, retain P0 order |

The candidate residual model is Huber on the 41 base/winner, 21 geometry and
29 market fields.  It never reranks survivors.  Filters tested `>= 0` and
`>= +50` expected bps.

| Block | Arm | Trades | Net bps/trade | Total net bps | Worst month | Positive months |
|---|---|---:|---:|---:|---:|---:|
| 2025 Jan--Jun | A exact M4 | 1,394 | +90.59 | +126,281 | +30.52 | 6/6 |
|  | B >= 0 | 2,839 | +39.58 | +112,369 | +21.94 | 4/6 |
|  | B >= 50 | 2,084 | +47.37 | +98,726 | +32.47 | 3/6 |
|  | C >= 0 | 1,115 | +105.27 | +117,380 | +76.82 | 4/6 |
|  | C >= 50 | 877 | +108.35 | +95,024 | +76.82 | 3/6 |
| 2026 Jan--Jul | A exact M4 | 455 | +92.60 | +42,134 | +6.07 | 7/7 |
|  | B >= 0 | 1,402 | +7.24 | +10,152 | -30.08 | 1/7 |
|  | B >= 50 | 392 | +76.25 | +29,890 | +76.25 | 1/7 |
|  | C >= 0 | 228 | +101.01 | +23,031 | +6.07 | 4/7 |
|  | C >= 50 | 45 | +205.51 | +9,248 | +205.51 | 1/7 |

Neither B nor C advances.  The C variants mostly reject P0 winners instead of
identifying an improved rank-2--4 candidate (`mean_p0_rank = 1.0`).  Their
apparently higher bps/trade comes from a sharp and non-portable participation
collapse.  B expands participation but destroys economic quality.

## Required follow-up

1. Do not add short consensus, BCF, or candidate residual routing yet.
2. Keep M4 as a **research challenger only** and use it to diagnose the
   2024-to-2025 absolute-conversion shift.
3. Before another target sweep, test whether stable causal regime/score-domain
   markers explain M4's 2024 failure while holding the M4 target and threshold
   fixed.  Any successor must retain 2025--26 later-era performance and make
   the 2024 result non-negative without reducing support to isolated months.
4. If that fails, conclude that the P0 ex-post headroom is not recoverable
   from the current causal state, rather than copying the long consensus.

## Reproducibility

Code and focused tests:

- `scripts/run_strict_r3_short_p0_absolute_conversion_funnel.py`
- `scripts/run_strict_r3_short_p0_topk_hierarchical_funnel.py`
- `extreme_price_movements/tests/test_short_absolute_conversion_funnel_contract.py`
- `extreme_price_movements/tests/test_short_topk_hierarchical_contract.py`

Completed immutable artifacts:

- `data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2024_maydec_20260821_v1`
- `data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2025_h1_20260821_v1`
- `data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2026_janjul_20260821_v1`
- `data_perp/artifacts/strict_r3_short_p0_topk_hierarchical_funnel_2025_h1_20260821_v8`
- `data_perp/artifacts/strict_r3_short_p0_topk_hierarchical_funnel_2026_janjul_20260821_v2`

Earlier top-K v1--v7 artifacts are retained as failed/provisional diagnostics;
only v8 (2025) and v2 (2026) use the exact frozen M4 source, correct candidate
identity merge, direct A control, and explicit p80 plateau-tie rule.
