# SR Workflow - Final Configuration Summary

**Date:** November 1, 2025  
**Status:** ✅ Production Ready

---

## Overview

The SR (Support/Resistance) workflow has been completely redesigned with:
1. **ML-first approach** - Machine learning for level quality prediction (LGBM)
2. **No aggressive clustering** - Filtering based on learned strength scores
3. **Hierarchical HPO** - Advanced parameter optimization (Coarse → Fine → TPE)
4. **Full explainability** - SHAP feature importance analysis

---

## Workflow Steps

### Step 0: ML Model Training (**ENABLED BY DEFAULT**)
- **Purpose**: Train LGBM model to predict SR level quality
- **Input**: Historical SR levels + forward performance labels
- **Output**: 
  - Trained model (`models/sr_quality_model.lgb`)
  - SHAP summary plots (feature importance)
  - Training metrics (R², MAE, etc.)
- **Configuration**:
  - Training period: Auto-set to last 6 months (customizable)
  - Sampling: Weekly samples (7-day intervals)
  - Forward window: 10 days for performance measurement
  - Cross-validation: 5-fold CV
- **Skip**: Use `--no-train-ml` if model already exists

### Step 1: SR Parameter Optimization
- **Method**: Hierarchical optimization (3 stages)
  1. Coarse grid search (4 points per param)
  2. Fine grid search (6 points per param, denser)
  3. TPE Bayesian optimization (50 trials)
- **Parameter Groups**:
  - Core detection (touches, strength threshold)
  - Quality filtering (distance, volume)
  - Temporal lookback
  - Market context (trend, breakouts)
- **Performance**: ~12 trials total vs 100+ in grid search

### Step 2: SR Detection
- **Methods**: Multi-method detection
  - Pivot points
  - Fractal patterns
  - Volume profile
  - Statistical levels
  - Psychological levels
- **ML Scoring**: Each level scored by trained LGBM model
- **Output**: 
  - All detected levels (typically 80-100)
  - ML quality scores
  - Method effectiveness metrics

### Step 3: SR Level Filtering (**REPLACES CLUSTERING**)
- **Method**: Strength-based filtering
- **Threshold**: `strength >= 0.5`
- **Retention**: Typically 95-100% of levels kept
- **Rationale**: ML model handles quality prediction; no need for aggressive clustering

---

## How Strength is Calculated

SR level strength is a composite score in range [0, 1]:

```python
# Calculate effective touches (only those with rejection/bounce)
rejection_ratio = min(avg_bounce_ratio / 0.02, 1.0)  # 2% bounce = 1.0
effective_touches = touch_count × rejection_ratio
touch_boost = min(effective_touches × 0.1, 0.3)

# Volume-scaled failure penalty
volume_factor = max(0.5, volume_confirmation_score)
failure_penalty = min(failure_count × 0.2 × (2.0 - volume_factor), 0.6)

# Special boosts
special_boost = (
    pivot_level ? 0.1 : 0 +
    psychological_level ? 0.05 : 0 +
    volume_at_level × 0.1  # HVN boost
)

strength = (
    base_strength                      # From detection method
    + touch_boost                      # Only touches WITH rejection
    + volume_boost                     # volume_confirmation × 0.2
    + consistency_boost                # consistency_score × 0.2
    + confluence_boost                 # multiple methods × 0.1
    + special_boost                    # pivot, psychological, HVN
    - failure_penalty                  # -0.2 per breakout, volume-scaled
)
clamped to [0.0, 1.0]
```

**Components:**
- **Touch boost**: Only counts touches with **actual rejection** (bounce ratio > 0)
  - Touch without bounce = 0 boost
  - Higher bounce = more effective touch
  - Max +0.3 total
- **Volume boost**: High volume at level = confirmation (up to +0.2)
- **Consistency boost**: Regular, predictable touches (up to +0.2)
- **Confluence boost**: Multiple methods detect same level (up to +0.1)
- **Special boosts**: 
  - Pivot points: +0.1
  - Psychological levels (round numbers): +0.05
  - **HVN (High Volume Node)**: up to +0.1
- **Failure penalty**: -0.2 per breakout, **scaled by volume**
  - Low volume breakout (weak): Higher penalty (up to -0.4 per failure)
  - High volume breakout (strong): Lower penalty (-0.2 per failure)
  - Max -0.6 total

**Key Improvements:**
1. **Rejection-based touches**: Empty touches without bounce don't count
2. **Volume-scaled penalties**: Weak breakouts penalized more heavily
3. **HVN recognition**: High volume nodes get boost

**Filtering Logic:**
- `strength < 0.5`: Weak level → Removed
- `strength >= 0.5`: Keep for ML model evaluation

---

## Default Configuration

### Basic Run (Recommended)
```bash
python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m --lookback-days 7
```

**Includes by default:**
- ✅ ML training (6-month auto-set period)
- ✅ SHAP explanations
- ✅ Hierarchical HPO
- ✅ Method effectiveness metrics
- ✅ Strength-based filtering

### Skip ML Training (if model exists)
```bash
python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m --no-train-ml
```

### Custom ML Training Dates
```bash
python scripts/run_sr_workflow.py \
  --symbol ETHUSDT \
  --timeframe 15m \
  --ml-start-date 2025-06-01 \
  --ml-end-date 2025-10-31
```

---

## Output Files

### Reports (Markdown)
- `outcomes/sr_workflow_{SYMBOL}_{TF}/`
  - `ml_model_training_{SYMBOL}_{TF}_{timestamp}.md` - ML metrics, SHAP
  - `sr_parameter_optimization_{SYMBOL}_{TF}_{timestamp}.md` - HPO results
  - `sr_detection_{SYMBOL}_{TF}_{timestamp}.md` - Detection metrics, method analysis
  - `workflow_summary_{SYMBOL}_{TF}_{timestamp}.md` - Overall summary

### Artifacts
- `models/sr_quality_model.lgb` - Trained LGBM model
- `models/sr_quality_model.lgb.metadata.json` - Model metadata
- `outcomes/.../shap_summary_{SYMBOL}_{timestamp}.png` - SHAP plot
- `artifacts/.../sr_parameter_optimization_result_*.parquet` - Optimized params
- `outcomes/.../sr_detection_result_*.json` - Detected levels

---

## Performance Metrics

### Latest Run (ETHUSDT 15m, 3-day lookback)
- **Total Duration**: 35.47s (3 steps)
- **HPO**: 12 trials, best score 0.9565
- **Detection**: 85 levels (41 support, 44 resistance)
- **Filtering**: 85 → 84 (98.8% retention, 1 weak removed)
- **Success Rate**: 100%

### Before vs After

| Metric | Before (Clustering) | After (ML Filtering) |
|--------|-------------------|---------------------|
| Levels detected | 95 | 85 |
| Levels after processing | 7 (92.6% removed) | 84 (98.8% retained) |
| Processing method | DBSCAN clustering | Strength filtering |
| Retention logic | Aggressive deduplication | Keep strong levels |
| ML involvement | None | Quality scoring |

---

## Key Improvements

### 1. **Clustering Removed**
- **Problem**: DBSCAN was too aggressive (95 → 7 levels, 92.6% reduction)
- **Solution**: Replace with strength-based filtering
- **Result**: 98.8% retention, only remove genuinely weak levels

### 2. **ML Training Enabled by Default**
- **Problem**: Previously optional, inconsistent results
- **Solution**: Always train unless `--no-train-ml` specified
- **Benefit**: Fresh model on latest data, SHAP insights

### 3. **Hierarchical HPO**
- **Problem**: Simple grid search was slow and inefficient
- **Solution**: 3-stage optimization (Coarse → Fine → TPE)
- **Benefit**: 12 trials vs 100+, better parameter exploration

### 4. **Detection Method Metrics**
- **New Feature**: Track which methods detect strong levels
- **Output**: Effectiveness scores per method/timeframe
- **Use Case**: Identify best-performing detection strategies

### 5. **Data Loading Fixed**
- **Problem**: Custom path logic, timestamp issues (1970 dates)
- **Solution**: Use `RealDataLoader` with proper artifact management
- **Benefit**: Reliable, consistent data access

---

## ML Model Details

### Architecture
- **Model**: LightGBM Regressor
- **Target**: `quality_score` (future SR level performance)
- **Features**: 30+ features including:
  - Touch count, strength, consistency
  - Volume confirmation
  - Bounce ratios, failure rate
  - Age, recency, confluence
  - Normalized distances (ATR)
  - Multi-timeframe support

### Training
- **Data Collection**: Walk-forward sampling
  - Sample every 7 days
  - 10-day forward window for labeling
  - Minimum 200 historical bars
- **Validation**: 5-fold cross-validation
- **Metrics**: R², MAE, RMSE per fold

### Explainability
- **SHAP**: TreeExplainer for feature importance
- **Output**: Summary plots showing top 20 features
- **Location**: `outcomes/.../shap_summary_*.png`

---

## Configuration Parameters

### ML Training
- `--ml-start-date`: Training start (default: auto 6 months ago)
- `--ml-end-date`: Training end (default: auto today)
- `--ml-timeframe`: Training TF (default: same as main `--timeframe`)
- `--ml-sample-freq-days`: Sampling interval (default: 7)
- `--ml-forward-days`: Performance window (default: 10)
- `--ml-model-output`: Model save path (default: `models/sr_quality_model.lgb`)

### HPO
- Hierarchical optimization enabled by default
- `coarse_grid_points`: 4
- `fine_grid_points`: 6
- `tpe_trials`: 50

### Filtering
- `min_strength_threshold`: 0.5 (hardcoded, can be made configurable)

---

## Next Steps / Recommendations

### 1. **Make Strength Threshold Configurable**
```python
parser.add_argument('--min-strength', type=float, default=0.5, 
                   help='Minimum strength threshold for filtering (0.0-1.0)')
```

### 2. **Add ML Model Performance Monitoring**
- Track prediction accuracy on new data
- Alert if model performance degrades
- Auto-retrain triggers

### 3. **Ensemble Multiple Detection Timeframes**
- Currently single TF
- Could aggregate 15m + 1h + 4h for stronger levels

### 4. **Add Level Clustering Back (Optional)**
- Use ML scores instead of DBSCAN distances
- Cluster by similarity in feature space
- Keep highest-scoring level per cluster

### 5. **Real-time Level Updates**
- Stream new data
- Incremental level updates
- Live SHAP explanations

---

## Troubleshooting

### Issue: "No training samples collected"
**Cause**: Insufficient data for ML training dates  
**Fix**: Extend `--ml-start-date` further back or use `--no-train-ml`

### Issue: VectorBT warnings
**Cause**: VectorBT optimization attempts  
**Fix**: These are non-fatal warnings, can be ignored

### Issue: Slow ML training
**Cause**: Large training dataset  
**Fix**: Increase `--ml-sample-freq-days` (e.g., 14 for bi-weekly)

---

## Contact / Support

For issues or questions about the SR workflow:
1. Check logs in workflow output
2. Review report files in `outcomes/sr_workflow_*/`
3. Verify data availability in `historical_data/`

---

**Last Updated:** 2025-11-01  
**Version:** 2.0 (ML-first, clustering removed)

