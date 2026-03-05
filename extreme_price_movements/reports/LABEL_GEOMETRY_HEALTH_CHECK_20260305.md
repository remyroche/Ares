# Label Geometry Health Check (2026-03-05)

## Input bucket distribution reviewed

| Bucket | Median TP | Median SL | Median Timeout |
|---|---:|---:|---:|
| MR_long | 17.0% | 71.9% | 12.6% |
| MR_short | 17.4% | 70.3% | 12.3% |
| TF_long | 14.0% | 58.4% | 27.6% |
| TF_short | 20.3% | 65.6% | 13.0% |

## Quick economics check

Assume timeout contributes ~0 expectancy (flat/near-flat), and normalize SL loss to `-1R`.
Then the minimum TP reward multiple needed for break-even (before fees) is:

\[
R_{min} = \frac{p(SL)}{p(TP)}
\]

Using the medians above:

| Bucket | Required TP multiple for breakeven (no fees) |
|---|---:|
| MR_long | 4.23R |
| MR_short | 4.04R |
| TF_long | 4.17R |
| TF_short | 3.23R |

Interpretation:
- If your realized TP:SL payoff ratio is below these thresholds, expectancy is negative even **before fees/slippage**.
- With realistic crypto execution costs, required multiples are higher.

## Fee sanity check

For taker-style round-trip costs (roughly 8–16 bps total including slippage), these distributions are usually **not** fee-resilient unless:
1. TP distance is very large versus SL (roughly >3.5R to >4.5R depending on bucket), and
2. timeouts are close to flat and not leaking via drift/spread.

Given the observed TP frequencies (14–20%), this is a fragile regime and likely below fee-adjusted break-even in most practical settings.

## Learnability perspective

From a modeling perspective:
- Class imbalance is severe (TP minority), so raw accuracy can look good while trading expectancy remains poor.
- TF_long is additionally stressed by high timeout mass (27.6%), which dilutes signal and increases horizon/noise sensitivity.
- These distributions can still be learnable for ranking, but **not necessarily tradable** unless calibration + sizing + selective gating materially improve effective win/loss economics.

## Verdict

- **Healthy TP:SL?** Not by default; only potentially healthy if realized payoff ratio is exceptionally high (roughly 3.2R–4.2R+ depending on bucket).
- **Above fees?** Likely no for most conventional fee/slippage assumptions unless exits are highly favorable.
- **Learnable?** Potentially learnable as a probability-ranking task, but currently weak as a direct trading label regime without stronger filtering/calibration.

## Recommended next checks

1. Compute realized `E[R] = p(TP)*R_tp - p(SL)*1 - cost_R` per bucket using actual selected geometry distances and measured execution costs.
2. Re-run bucket report with stratification by volatility regime and spread regime to see where TP frequency improves.
3. Enforce minimum economic constraints in geometry search (e.g., reject configs with required break-even > 3.5R by default unless clearly justified by historical realized R).
   This earlier-stage gate should only remove the most obviously uneconomic candidates.
4. Consider reducing timeout proportion in TF_long via horizon/trigger refinement or explicit timeout utility shaping.
