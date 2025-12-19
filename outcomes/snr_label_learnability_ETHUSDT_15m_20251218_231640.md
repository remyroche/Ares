# Label-Learnability Diagnostics

**Symbol**: ETHUSDT
**Exchange**: binance
**Timeframe**: 15m

## Summary
- Valid labeled samples: 536
- Positive label rate: 38.8%

## Learnability
- Mean CV AUC: 0.4726
- Learnability score (AUC - 0.5 * std): 0.4063

## Entropy / Balance
- Balance score: 0.9635

## Combined Label-Quality Objective
- Combined score: 0.5735

## Interpretation Hints
- Learnability (mean AUC=0.4726): Mean CV AUC < 0.55 → very weak learnability; labels are close to random.
- Balance (entropy score=0.9635): Entropy score ≥ 0.8 → labels are well balanced.
- Combined score (0.5735): Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.

## Overall Learnability Score
- Score (0-1): 0.573
- Rating: Pass
- Summary: Combined score 0.4–0.6 → mixed quality; may be adequate for robust models.
