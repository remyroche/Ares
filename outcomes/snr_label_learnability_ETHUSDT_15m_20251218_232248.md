# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 536
- Positive label rate: 38.2%

## Learnability
- Mean CV AUC: 0.5116
- Learnability score (AUC - 0.5 * std): 0.4537

## Entropy / Balance
- Balance score: 0.9598

## Combined Label-Quality Objective
- Combined score: 0.6055

## Interpretation Hints
- Learnability (mean AUC=0.5116): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.9598): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.6055): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.606
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.
