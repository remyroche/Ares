# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 1634
- Positive label rate: 26.6%

## Learnability
- Mean CV AUC: 0.5640
- Learnability score (AUC - 0.5 * std): 0.5503

## Entropy / Balance
- Balance score: 0.8351

## Combined Label-Quality Objective
- Combined score: 0.6357

## Interpretation Hints
- Learnability (mean AUC=0.5640): Mean CV AUC 0.55–0.60 → weak but potentially usable signal.
- Balance (entropy score=0.8351): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.6357): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.636
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.
