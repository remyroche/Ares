# Full-universe T2/T4 sequential funnel — final audit

## Contract verified

- Universe: 1,850,552 eligible hourly long/short rows; no `selected_for_monitor` filter.
- Entry: decision close plus one hour, at the next one-minute open.
- Exit: exact first TP/SL touch, otherwise H12 timeout; never a synthetic `t+12` mark.
- Short paths: signed returns, not reciprocal pricing.
- Cost: 100 bps declared round-trip cost, reported separately from gross.
- Selection: one pooled, global long/short book after common-bps mapping; never a per-timestamp or side-quota book.
- Base/meta source fields: distinct config pools.  Base selection used 36 diverse training-only fields; meta used 36 separate context/regime/trust fields plus the same-side raw base probabilities.

## Stage 1: target screen

Development T2/T4 2x2 geometry screen selected the soft T2 TP3/SL2 arm.  The
user-selected working contract was therefore H12, TP +3 ATR / SL -2 ATR,
tau=.25.  (The original TP2/SL1 reference remains materialised in the panel.)

The OOS Aug--Nov T2 base result was gross +32.74 / net -67.26 bps at global
top-10.  The corresponding 200-tree 36-feature baseline is the best valid
base arm tested in this run.

## Target learnability and oracle

The target is economically meaningful:

| OOS score source | Global top-10 net bps |
|---|---:|
| Perfect realised-net oracle | +400.18 |
| Training-mapped soft-T2 oracle | +92.20 |
| Best causal base | -67.26 |

The causal base rank IC is 0.106 to its soft target and 0.073 to realised net.
The problem is consequently causal extraction, not a target without economic
headroom.

## Base ablations

| Variant | OOS global top-10 net bps | Decision |
|---|---:|---|
| LightGBM, 36 base fields | -67.26 | retain reference |
| 40 diverse base fields | -73.40 | reject |
| chronological MDA subset | -70.61 | reject |
| CatBoost, 40 iterations | -80.29 | reject |
| CatBoost, 100 iterations | -77.96 | reject |
| mild certainty weights | -72.05 | reject |
| strong certainty weights | -71.65 | reject |
| GAM logit residual base | -102.44 | reject |
| GAM probability augmentation | -75.65 | reject |
| D1 OOF future-teacher distillation | -72.95 | reject |

The certainty ledger covers all 1,850,552 rows, is future-derived and
training-only, and has median certainty .909.  It never enters inference.

## Meta/residual ablations

The meta learner receives only the candidate's same-side `p_upper`,
`p_lower`, and `p_timeout`, plus meta-only regime/context/trust fields.  It is
per-row, not per-period.  Independent side calibration is invalid for the
global book: it creates side-scale domination.  A shared common-unit model was
therefore also tested.

| Variant | OOS global top-10 net bps | Decision |
|---|---:|---|
| side-local direct raw-probability meta, 120d correct trailing window | -143.05 | reject |
| shared per-row common-unit meta, 120d | -84.51 | reject |
| shared per-row common-unit meta, 30d | -81.37 | reject |
| side-local 7d/30d calibration | degenerate non-positive OOF slope | reject |

## Sequential decision

The target-family and base-plus-meta economic gates are not met.  Do not
advance this stack to ranking, portfolio policy, execution timing, or action
optimization.  The valid final decision for this run is:

```text
ROBUST_TARGET_IDENTIFIED_MODEL_INSUFFICIENT
NO_TARGET_FAMILY_ADVANCES
```

The next research cycle should focus on new causal feature representation or a
new candidate-generation process, then repeat this frozen T2/T4 funnel; it
must not rescue this negative base/meta score through downstream policy tuning.
