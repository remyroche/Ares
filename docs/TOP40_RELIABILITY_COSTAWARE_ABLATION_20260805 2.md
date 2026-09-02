# Top-40% reliability residual and cost-aware specialist ablation

## Scope

This replay implements the requested architecture on the three existing
transport folds:

- admission: raw broad-base score top 40% within each 4-hour × side query;
- residual model: trained and applied only to admitted rows;
- target: three reliability classes derived from
  `net_bps - causal_base_map_bps`:
  - class 0: base overconfident, residual < −50 bps;
  - class 1: approximately correct, residual in [−50, +50] bps;
  - class 2: base underconfident, residual > +50 bps;
- objective: native LambdaRank with 4-hour × side groups;
- correction: raw ranking score transformed only through a bounded `tanh`,
  with OOF-selected lambda/cap/threshold; no isotonic calibration;
- specialist targets: explicit cost-aware robust-clear labels
  `gross_H12 − 100 bps > {25, 50, 75} bps`.

Artifact:

`data_perp/artifacts/top40_reliability_costaware_20260805_v2/`

Focused tests: 14 passed.

## Admission and query coverage

The admission mask is stable at 40.19% of candidates globally (40.17–40.20%
by fold/month). The LambdaRank query audit shows:

| Fold | Queries per side | Median admitted rows/query | Minimum |
|---|---:|---:|---:|
| Jul–Aug | 106 | 92.5 | 7 |
| Sep–Oct | 112 | 84.0 | 64 |
| Nov partial | 112 | 82.0 | 37 |

The query construction is therefore genuinely 4-hour × side, not timestamp or
per-period ranking.

## Global OOS net bps/trade

The no-op is the same side-local prior-mapped base score for every target arm.

| Specialist margin | System | Top 1% | Top 5% | Top 10% |
|---:|---|---:|---:|---:|
| — | No-op mapped base | **−22.33** | **+11.52** | **−43.33** |
| 25 | Reliability correction | −63.28 | −70.35 | −67.19 |
| 50 | Reliability correction | **−13.35** | −47.01 | −64.80 |
| 75 | Reliability correction | −56.49 | −84.57 | −81.31 |

The +50-bps cost-clear specialist target is the best of the three reliability
arms, but it still loses the no-op at top-5 (−58.53 bps) and top-10 (−21.47
bps). It only improves top-1 by 8.98 bps, while remaining negative.

## Side results for the best (+50 bps) target

| Side/system | Top 1% | Top 5% | Top 10% |
|---|---:|---:|---:|
| Long, no-op | +48.16 | +1.15 | +11.52 |
| Long, reliability | −32.02 | −15.51 | −40.23 |
| Short, no-op | −148.84 | −178.86 | −159.11 |
| Short, reliability | −103.14 | −93.27 | −93.81 |

The residual improves the short side relative to its extremely weak no-op, but
it damages the long side, which is where the no-op had its positive tails.
The pooled result is therefore not a genuine conversion improvement; it is
mostly a trade-off between two side-specific failures.

## Fold/month stability for the +50-bps target

| Period | Reliability top-1 | top-5 | top-10 | No-op top-1 | top-5 | top-10 |
|---|---:|---:|---:|---:|---:|---:|
| Jul–Aug | −51.56 | −75.33 | −76.20 | −64.05 | −134.19 | −104.61 |
| Sep–Oct | −69.85 | −86.91 | −85.57 | −44.39 | −108.26 | −107.90 |
| Nov partial | +113.02 | +74.31 | +24.87 | +131.00 | −27.89 | +16.31 |

The apparent positive improvement is concentrated in November. The earlier
periods remain negative, and the residual does not consistently transport.

## Reliability learnability

On admitted OOS rows, reliability-score rank IC against the realised residual
was:

- long: approximately +0.02 to +0.07;
- short: approximately −0.01 to +0.05;
- three-class accuracy: roughly 0.32–0.38, close to the 1/3 class baseline.

Thus the ranking signal is weak and the class prediction is only marginally
better than prevalence. The LambdaRank objective is being used correctly, but
there is not enough stable conversion information for the correction to be
trusted.

## Cost-aware specialist target stability

The +50-bps target rate among admitted test rows was approximately:

- long: 25.2% (Jul–Aug), 23.1% (Sep–Oct), 38.8% (November);
- short: 27.9% (Jul–Aug), 21.8% (Sep–Oct), 24.2% (November).

The +25 and +75 targets move the rate as expected, but do not repair the
transport shift. The target is economically explicit, yet its prevalence and
payoff distribution remain regime-dependent.

## Decision

The requested pipeline is implemented and verified, but this architecture does
not advance:

1. Top-40% admission is correctly enforced, but it does not make the residual
   reliably learnable.
2. Three-class reliability LambdaRank has weak cross-side signal.
3. The +50-bps cost-aware specialist target is the best tested target, but the
   correction still loses the no-op at the broader tails.

The production candidate remains the side-local mapped base score. The next
credible residual experiment should be smaller and more constrained: train a
side-local reliability model only on the base top-20%, use a penalty-only or
shrinkage correction, and select against the full global validation population
with a worst-month gate.

