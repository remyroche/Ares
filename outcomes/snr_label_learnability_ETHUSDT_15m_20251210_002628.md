# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 21177
- Positive label rate: 20.0%

## Learnability
- Mean CV AUC: 0.5073
- Learnability score (AUC - 0.5 * std): 0.4907

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.3435

## Interpretation Hints
- Learnability (mean AUC=0.5073): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.3435): Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.

## Overall Learnability Score
- Score (0-1): 0.343
- Rating: Bad
- Summary: Combined score < 0.4 → overall label quality is weak; consider revisiting thresholds.