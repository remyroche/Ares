# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 2445
- Positive label rate: 27.4%

## Learnability
- Mean CV AUC: 0.6512
- Learnability score (AUC - 0.5 * std): 0.6484

## Entropy / Balance
- Balance score: 0.8472

## Combined Label-Quality Objective
- Combined score: 0.7081

## Interpretation Hints
- Learnability (mean AUC=0.6512): Mean CV AUC 0.60–0.70 → moderate learnability.
- Balance (entropy score=0.8472): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.7081): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.708
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.