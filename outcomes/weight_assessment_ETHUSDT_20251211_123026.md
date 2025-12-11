# Label Weight Assessment Report

**Date:** 20251211_123026
**Symbol:** ETHUSDT
**Timeframe:** 15m

## 1. Weight Statistics
- Mean: 1.0008
- Std: 1.2208
- Min: 0.0100
- Max: 9.4979

## 2. Comprehensive ML Assessment (Probe Model)
*Note: Probabilities are calibrated via Isotonic Regression on an inner temporal split before computing Brier/ECE.*

| Metric | Baseline (Uniform) | Weighted | Delta |
|---|---|---|---|
| **AUC** | 0.9129 | 0.9241 | **+0.0112** |
| **Log Loss** | 1.2008 | 0.9073 | -0.2936 |
| **Brier Score** | 0.1236 | 0.1159 | -0.0078 |
| **ECE** | 0.0951 | 0.0821 | -0.0130 |
| **Sharpe Ratio** | 0.5326 | 0.5447 | **+0.0121** |
| **Info. Coeff.** | 0.8171 | 0.8394 | +0.0222 |
| Precision | 0.8162 | 0.8202 | +0.0040 |
| Recall | 0.8169 | 0.8498 | +0.0329 |
| F1-Score | 0.8040 | 0.8272 | +0.0233 |

## 3. Interpretation
✅ **Strong Positive Impact**: Weights improved both Learnability (AUC) and Profitability (Sharpe).
The weighting scheme is effectively highlighting high-quality signal events.
