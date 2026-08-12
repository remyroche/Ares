# Frozen stack: trained conversion/reliability learner ablation

Date: 2026-08-10  
Input: frozen ATR2 specialists → q4h×side ordinal residual LambdaRank  
Transport: July–November 2024, 388,494 rows

## Question

Can a dedicated conversion learner predict when the frozen residual score is
underconfident, approximately correct, or overconfident, using only causal
score/base/regime/context fields?

## Strict training contract

Separate long and short learners were fit at the start of every transport month
using only earlier rows with `label_available_ts < month_start`. July therefore
uses the identity correction because there is no earlier scored month in this
artifact; August uses July, September uses July–August, and so on.

The 25-field input contract contains:

- frozen `score` and `prequential_base_expected_net_bps`;
- `p_clear`, `p_adverse`, `p_weak`;
- four causal soft regime probabilities, entropy, transition onset, and state age;
- causal market return, liquidity, volatility, OI, funding, correlation,
  deleveraging, liquidation, breadth, resilience, and short-covering fields.

Targets:

- regression: `net_bps − frozen_score`, Huber loss;
- ordinal reliability: residual <= −75 bps = overconfident, |residual| < 75
  bps = approximately correct, residual >= +75 bps = underconfident.

The predicted correction is added to the frozen score at fixed strengths 0.25,
0.50, and 1.00. No test-month labels influence model fitting or strength.

Implementation: `scripts/run_frozen_conversion_reliability_learner_ablation.py`  
Artifacts: `data_perp/artifacts/frozen_conversion_reliability_learner_ablation_20260810_v1/`

## Pooled global results

Net bps/trade:

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% |
|---|---:|---:|---:|---:|---:|
| Frozen raw control | −23.86 | −7.30 | **+16.01** | **+8.89** | −37.63 |
| Regression α=.25 | −25.83 | −7.41 | +15.62 | −11.02 | −48.70 |
| Regression α=.50 | −25.92 | −7.27 | +2.05 | −40.07 | −65.01 |
| Regression α=1.00 | −43.92 | −72.56 | −89.99 | −106.08 | −110.15 |
| Ordinal α=.25 | −68.75 | −52.60 | −16.69 | −27.85 | −44.51 |
| Ordinal α=.50 | **+27.68** | −27.33 | −50.20 | −45.21 | −59.86 |
| Ordinal α=1.00 | **+54.09** | −33.36 | −42.14 | −43.87 | −68.95 |

The ordinal learner creates a positive very narrow top-0.5% point estimate, but
its top-5% and top-10% ranking is substantially worse. It is not a valid global
conversion repair.

## Monthly global top-5 net

| Arm | Jul | Aug | Sep | Oct | Nov | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw control | −51.55 | −171.07 | −58.25 | −81.07 | **+11.00** | −70.19 | −171.07 |
| Regression α=.25 | −51.55 | −186.75 | −76.46 | −81.32 | +9.97 | −77.22 | −186.75 |
| Regression α=.50 | −51.55 | −192.97 | −95.09 | −82.84 | +9.73 | −82.54 | −192.97 |
| Regression α=1.00 | −51.55 | −195.73 | −103.84 | −82.89 | +8.63 | −85.88 | −195.73 |
| Ordinal α=.25 | −51.55 | −92.91 | −63.64 | −89.20 | **+26.53** | −54.15 | −92.91 |
| Ordinal α=.50 | −51.55 | −45.69 | −48.86 | −110.79 | −20.10 | −55.40 | −110.79 |
| Ordinal α=1.00 | −51.55 | −25.03 | −37.00 | −129.45 | −103.34 | −69.27 | −129.45 |

Ordinal α=.25 improves the worst month from −171.07 to −92.91, but remains
negative in every month except November and loses 36.74 bps at pooled top-5.
The stronger ordinal variants show the same regime-instability pattern: they
help August/September and fail badly in October/November.

## Per-side top-5 net

| Arm | Long | Short |
|---|---:|---:|
| Raw control | **+17.25** | −195.71 |
| Regression α=.25 | +17.25 | −192.54 |
| Regression α=.50 | +13.30 | −187.76 |
| Regression α=1.00 | +4.19 | −164.37 |
| Ordinal α=.25 | −6.07 | −108.58 |
| Ordinal α=.50 | −28.80 | −62.01 |
| Ordinal α=1.00 | −36.34 | −49.87 |

The ordinal learner reduces the short-side loss by transferring the pooled tail
toward shorts, but it destroys the profitable long-side contribution. This is a
side-comparability trade, not a true economic repair.

## Decision

`NO_TRAINED_CONVERSION_ADVANCE`.

The dedicated reliability target is learnable enough to change the ranking, but
not stable enough to improve the required global top-5 economics. The result
confirms that the missing component is not simply a better residual target or a
larger context feature set. The conversion mapping must explicitly model
side/regime transport and select against equal-month/worst-month gates; a model
that merely predicts residual direction will trade one bad regime for another.
