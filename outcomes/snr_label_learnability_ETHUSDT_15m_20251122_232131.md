# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 31972
- Positive label rate: 50.0%

## Learnability
- Mean CV AUC: 0.7763
- Learnability score (AUC - 0.5 * std): 0.7385

## Entropy / Balance
- Balance score: 1.0000

## Combined Label-Quality Objective
- Combined score: 0.8170

## Interpretation Hints
- Learnability (mean AUC=0.7763): Mean CV AUC ≥ 0.70 → strong learnability; labels are easy to learn.
- Balance (entropy score=1.0000): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.8170): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.817
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.