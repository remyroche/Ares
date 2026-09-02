# Soft Label And Sample Weight Ablation Plan

## Objective

Improve the active policy-rollout labeling path using soft labels only, and A/B test sample-weight designs separately from the soft-label target design.

The goal is to determine:

- whether the soft target improves the learned signal;
- whether the sample weights improve where the model pays attention;
- whether the combination improves live top-k ranking.

## Stage 1: Soft-Label Arms

Run soft-label arms first with a fixed neutral weighting baseline:

```text
W0 = existing base sample weights
```

Test:

```text
S0 current soft label
S2 cost-aware net-return soft label
S3 path-quality soft label
S6 asymmetric downside soft label
S7 horizon-blended soft label
S8 rank-aware soft label
```

Select the top 2-3 label definitions by timestamp-balanced HR@30, NDCG@30, and weekly lower-tail HR.

## Stage 2: Sample-Weight A/B Tests

For each winning soft-label arm, run weight ablations:

| Weight Arm | Definition | Purpose |
| --- | --- | --- |
| W0 | Current/base weights | Baseline |
| W1 | Soft-confidence weights: `abs(soft_y - 0.5)^gamma` | Emphasize clearer labels |
| W2 | Boundary weights: rows near top-30 rank cutoff get higher weight | Improve live selection boundary |
| W3 | Asymmetric downside weights: high MAE / SL-first rows get boosted | Learn fragile failures |
| W4 | Opportunity-miss weights: high MFE but low score rows get boosted | Improve missed winners |
| W5 | Difficult-period weights: recent/bad-period rows get gradual boost | Improve weak weeks |
| W6 | Path-quality weights: clean fast TP and fast SL both boosted | Learn decisive examples |
| W7 | Timestamp-balanced weights: equal total mass per timestamp | Prevent dense periods dominating |
| W8 | Combined conservative: W7 x capped W1 x capped W3 | Practical production candidate |

## Weight HPO Search

For weight arms, tune:

```text
confidence_gamma: 1.0 to 4.0
max_weight: 2.0 to 6.0
min_weight: 0.10 to 0.75
boundary_rank_low: 0.50 to 0.70
boundary_rank_high: 0.70 to 0.90
downside_weight_power: 0.5 to 3.0
difficult_period_boost: 1.0 to 4.0
timestamp_balance_strength: 0.0 to 1.0
```

Keep all weights normalized so mean training weight stays near `1.0`, and enforce a minimum effective sample size.

## Avoid Combinatorial Explosion

Do not test every soft label against every weight arm initially.

Use:

```text
Round A:
S arms x W0

Round B:
top 2-3 S arms x W1/W2/W3/W7/W8

Round C:
best S/W candidates x full HPO

Round D:
final candidate vs current baseline on OOF/OOS replay
```

## Selection Objective

Use a ranking-aware objective:

```text
1.00 * timestamp_balanced_HR@30
+0.50 * NDCG@30
+0.35 * HR@20
+0.20 * HR@10
+0.25 * Q25_week_HR@30
+0.15 * Q10_week_HR@30
- downside / full-SL penalties
```

## Reporting

For every evaluated arm, report:

- timestamp-balanced HR@10/20/30;
- NDCG@30;
- weekly Q5/Q10/Q25/Q50/Q75 HR@30;
- full-SL rate;
- timeout rate;
- accepted trade count;
- top-30 mean return;
- top-30 q05 return;
- gross PnL, costs, and net PnL where replay is available;
- effective sample size and weight distribution diagnostics.

## Guardrails

- Use purged CV with purge/embargo at least equal to the maximum label holding horizon.
- Keep the candidate universe fixed across ablations.
- Keep rank references and portfolio policy fixed across comparable arms.
- Treat label-HPO metrics as development metrics, not clean OOS.
- Evaluate final candidates on OOF/OOS replay before promotion.
