# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 4187
- Positive label rate: 25.1%

## Learnability
- Mean CV AUC: 0.5862
- Learnability score (AUC - 0.5 * std): 0.5785

## Entropy / Balance
- Balance score: 0.0000

## Combined Label-Quality Objective
- Combined score: 0.4049

## Interpretation Hints
- Learnability (mean AUC=0.5862): Mean CV AUC 0.55–0.60 → weak but potentially usable signal.
- Balance (entropy score=0.0000): Entropy score < 0.5 → labels are highly imbalanced or dominated by one class.
- Combined score (0.4049): Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

## Overall Learnability Score
- Score (0-1): 0.405
- Rating: Pass
- Summary: Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.