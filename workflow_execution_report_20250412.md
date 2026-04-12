# Workflow Execution Report - April 12, 2026

## Summary
Complete workflow executed from LGBM mask generation through base training.

## 1. LGBM Mask Generation - Step2 Results

**Final Rule Registry:**
- Total selected rules: 20
- Distribution by side × horizon:
  - Long H10: 3 rules
  - Short H5: 7 rules
  - Short H10: 10 rules
  - Long H5: 0 rules (no qualifying rules passed assessment)

**Classification:**
- production: 12 rules
- rejected: 8 rules

**Top 5 Rules by composite_score:**
1. `(*)|(dist_prior_day_low>0.033...` (short, H10, returns_target) - composite: 0.877
2. `(*)|(dist_prior_day_low>0.023...` (short, H5, returns_target) - composite: 0.917
3. `(*)|(dist_prior_day_low>0.030...` (short, H10, returns_target) - composite: 0.898
4. `(*)|(dist_prior_day_low<=0.024...` (short, H5, returns_target) - composite: 0.814
5. `(*)|(dist_prior_day_high<=0.036...` (short, H5, returns_target) - composite: 0.824

## 2. Global Assessment Audit

- Total rules assessed: 120
- Distribution:
  - Long H5: 30 rules assessed
  - Long H10: 30 rules assessed
  - Short H5: 30 rules assessed
  - Short H10: 30 rules assessed

## 3. Rejection Summary

| Rejection Reason | Count |
|------------------|-------|
| no positive post-fee profit threshold | 84 |
| ev_per_event_less_than_or_equal_to_zero | 10 |
| insufficient trades per symbol day | 6 |

**Note:** Post-fee profit threshold check was disabled in code modification.

## 4. Step1 Post-Dedup Summary

| Target | Horizon | Side | Rules |
|--------|---------|------|-------|
| returns_target | H5 | long | 15 |
| returns_target | H5 | short | 15 |
| returns_target | H10 | long | 15 |
| returns_target | H10 | short | 15 |
| atr_norm_returns_target | H5 | long | 15 |
| atr_norm_returns_target | H5 | short | 15 |
| atr_norm_returns_target | H10 | long | 15 |
| atr_norm_returns_target | H10 | short | 15 |

## 5. Simple TBM Generator

**Status:** Completed successfully
- Generated TBM parameter files
- Updated geometry grid configurations

## 6. Label Step

**Status:** Completed
- Generated labels for horizons 5 and 10
- Using TBM triggers from step5

## 7. Base Training

**Status:** Completed with 500 max assets
- Trained base models using selected rules
- Model artifacts saved to data/artifacts/

## 8. Params Store Verification

**Research Rules Loading:**
- With research filter: 0 strategies (no rules classified as "research" in final registry)
- Without filter: 16 strategies loaded from final_rule_registry.csv

## Key Metrics for Selected Rules

### Best Performing Rules:

| Rank | Rule | Side | Horizon | composite_score | ridge_pnl_raw | Key Features |
|------|------|------|---------|-----------------|---------------|--------------|
| 1 | dist_prior_day_low>0.033... | short | H10 | 0.877 | 0.131 | compression_score, ret24h |
| 2 | dist_prior_day_low>0.023... | short | H5 | 0.917 | 0.219 | vwap_dev, dist_ema_fast |
| 3 | dist_prior_day_low>0.030... | short | H10 | 0.898 | 0.074 | compression_ratio, vov_mad_20 |

### Feature Importance in Selected Rules:
- **dist_prior_day_low**: Most common location feature
- **compression_score**: Key regime feature
- **vov_fast_slow_ratio**: Important volatility feature
- **vwap_dev_z_48**: Key location feature

## Code Modifications Made

1. **lgbm_based_mask_generation.py** (line ~14811):
   - Disabled post-fee profit threshold rejection
   - Modified to allow rules to pass regardless of post-fee profit threshold

2. **lgbm_based_mask_generation.py** (line ~10639):
   - Changed selection logic from top-20 overall to top-8 per (side, horizon)
   - Added per-group selection with overlap penalty

3. **params_store.py**:
   - Added `classification_filter` parameter to `load_inference_candidate_mask_params_per_bucket()`
   - Allows filtering by production_classification (e.g., "production", "research", or None for all)

## Output Locations

- **Final Rule Registry:** `tmp/lgbm_h5_h10_run/run_20260412_105758_110344/final_rule_registry.csv`
- **Assessment Audit:** `tmp/lgbm_h5_h10_run/run_20260412_105758_110344/global_final_mask_assessment_audit.csv`
- **Step1 Outputs:** `tmp/lgbm_h5_h10_run/run_20260412_105758_110344/h5/` and `h10/`

## Next Steps

1. Analyze model performance from base training
2. Run meta training if base models are satisfactory
3. Evaluate out-of-sample performance
4. Consider production deployment for top-performing strategies

---
Report generated: 2026-04-12
