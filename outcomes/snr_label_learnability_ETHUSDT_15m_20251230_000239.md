# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 7421
- Positive label rate: 47.8%

## Learnability
- Mean CV AUC: 0.5089
- Learnability score (AUC - 0.5 * std): 0.5026

## Entropy / Balance
- Balance score: 0.9986

## Combined Label-Quality Objective
- Combined score: 0.6514

## Interpretation Hints
- Learnability (mean AUC=0.5089): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.9986): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.6514): Combined score ≥ 0.6 → good overall label quality.

## Overall Learnability Score
- Score (0-1): 0.651
- Rating: Great
- Summary: Combined score ≥ 0.6 → good overall label quality.
