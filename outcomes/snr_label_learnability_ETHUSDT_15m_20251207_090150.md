# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 21177
- Positive label rate: 10.2%

## Learnability
- Mean CV AUC: 0.6224
- Learnability score (AUC - 0.5 * std): 0.6165

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.4316

## Interpretation Hints
- Learnability (mean AUC=0.6224): Mean CV AUC 0.60–0.70 → moderate learnability.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.4316): Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

## Overall Learnability Score
- Score (0-1): 0.432
- Rating: Pass
- Summary: Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.