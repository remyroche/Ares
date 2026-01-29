# Causal Gating (RegimeTree) Report
- **Timestamp**: 2026-01-29 00:13:01
- **Leaves**: 6
- **Experts Used**: fractal_efficiency_specialist, gap_specialist, trend_specialist, volume_specialist, volatility_specialist

## Leaf Assignments
### Leaf 0 (n=23849)
- **Best Expert**: `fractal_efficiency_specialist` (Score: 0.0195)
- **Expert Weights**:
  - `fractal_efficiency_specialist`: 51.1%
  - `trend_specialist`: 48.9%
- **Stability Metrics (Per Fold)**:
  - `fractal_efficiency_specialist`: 0.0241 avg across 8 folds
  - `gap_specialist`: -0.0175 avg across 8 folds
  - `trend_specialist`: 0.0243 avg across 8 folds
  - `volume_specialist`: -0.0034 avg across 8 folds
  - `volatility_specialist`: -0.0011 avg across 8 folds

### Leaf 1 (n=19078)
- **Best Expert**: `trend_specialist` (Score: 0.0202)
- **Expert Weights**:
  - `trend_specialist`: 51.3%
  - `fractal_efficiency_specialist`: 48.7%
- **Stability Metrics (Per Fold)**:
  - `fractal_efficiency_specialist`: 0.0216 avg across 8 folds
  - `gap_specialist`: -0.0023 avg across 8 folds
  - `trend_specialist`: 0.0224 avg across 8 folds
  - `volume_specialist`: -0.0026 avg across 8 folds
  - `volatility_specialist`: 0.0082 avg across 8 folds

### Leaf 2 (n=76308)
- **Best Expert**: `gap_specialist` (Score: 0.0024)
- **Expert Weights**:
  - `gap_specialist`: 51.3%
  - `volatility_specialist`: 48.7%
- **Stability Metrics (Per Fold)**:
  - `fractal_efficiency_specialist`: -0.0036 avg across 8 folds
  - `gap_specialist`: 0.0062 avg across 8 folds
  - `trend_specialist`: -0.0042 avg across 8 folds
  - `volume_specialist`: -0.0040 avg across 8 folds
  - `volatility_specialist`: -0.0041 avg across 8 folds

### Leaf 3 (n=13249)
- **Best Expert**: `volume_specialist` (Score: 0.0245)
- **Expert Weights**:
  - `volume_specialist`: 52.2%
  - `volatility_specialist`: 47.8%
- **Stability Metrics (Per Fold)**:
  - `fractal_efficiency_specialist`: 0.0042 avg across 8 folds
  - `gap_specialist`: 0.0124 avg across 8 folds
  - `trend_specialist`: -0.0044 avg across 8 folds
  - `volume_specialist`: 0.0415 avg across 8 folds
  - `volatility_specialist`: 0.0336 avg across 8 folds

### Leaf 4 (n=119235)
- **Best Expert**: `fractal_efficiency_specialist` (Score: 0.0001)
- **Expert Weights**:
  - `fractal_efficiency_specialist`: 100.0%

### Leaf 5 (n=132484)
- **Best Expert**: `gap_specialist` (Score: -0.0002)
- **Expert Weights**:
  - `gap_specialist`: 100.0%

## Feature Importance
- **trend_strength**: 0.0248
- **fam_score_entropy_other**: 0.0223
- **ca__beta_long_w96**: 0.0197
