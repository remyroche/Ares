# Feature Generation Report

**Generated:** 2025-10-25 14:37:45
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light

## Summary

✅ **Successfully generated 4 features** from 1,920 rows of data.

## Feature Statistics

- **Total Features:** 4
- **Data Samples:** 1,920
- **Memory Usage:** 0.07 MB
- **Missing Values:** 0
- **Missing Value %:** 0.00%

## Comprehensive Feature Analysis

### Feature Quality Metrics

| Metric | Value |
|--------|-------|
| High Quality Features (>0.7 score) | 0 |
| Medium Quality Features (0.4-0.7) | 0 |
| Low Quality Features (<0.4) | 3 |
| Constant Features | 0 |
| Highly Correlated Pairs | 0 |
| Average Correlation | 0.215 |
| Feature Stability Score | 0.689 |

### Top 10 Performing Features

| Rank | Feature | Quality Score | Correlation | Stability | Information |
|------|---------|---------------|-------------|-----------|-------------|
| 1 | `price_ma_5` | 0.206 | 0.034 | 0.948 | 0.004 |
| 2 | `price_std_5` | 0.189 | 0.093 | 0.561 | 0.215 |
| 3 | `volume_ma_5` | 0.162 | 0.062 | 0.559 | 0.121 |

### Feature Distribution Analysis

| Statistic | Value |
|-----------|-------|
| Mean Quality Score | 0.186 |
| Median Quality Score | 0.189 |
| Std Quality Score | 0.018 |
| Min Quality Score | 0.162 |
| Max Quality Score | 0.206 |

### Feature Redundancy Analysis

| Metric | Value |
|--------|-------|
| Redundant Feature Pairs | 0 |
| Redundancy Rate | 0.0% |
| Unique Features | 4 |
| Redundancy Score | 1.000 |

### Feature Stability Analysis

| Metric | Value |
|--------|-------|
| Stable Features (>0.8) | 1 |
| Moderately Stable (0.5-0.8) | 2 |
| Unstable Features (<0.5) | 0 |
| Average Stability | 0.689 |

### Feature Information Content

| Metric | Value |
|--------|-------|
| High Information (>0.7) | 0 |
| Medium Information (0.4-0.7) | 0 |
| Low Information (<0.4) | 3 |
| Average Information | 0.113 |

### Feature Recommendations

#### Features to Keep (High Quality)

#### Features to Consider Removing (Low Quality)
- `price_ma_5`
- `price_std_5`
- `volume_ma_5`

#### Features to Investigate (Medium Quality)

## Feature Categories

### Returns (1 features)

- `returns`

### Volume (1 features)

- `volume_ma_5`

### Volatility (1 features)

- `price_std_5`

### Trend (2 features)

- `price_ma_5`
- `volume_ma_5`

## Data Quality

| Metric | Value |
|--------|-------|
| Total Columns | 4 |
| Total Rows | 1,920 |
| Non-Null Values | 7,680 |
| Null Values | 0 |
| Memory Usage (MB) | 0.07 |

## Artifacts

### generated_features

**Path:** `artifacts/pre_training/long/Analyst/feature_generation_feature_generation_step/feature_generation_feature_generation_step_generated_features_long_Analyst_20251025_143706.parquet`
**Size:** 75.16 KB

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

