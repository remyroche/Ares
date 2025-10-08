# Pre-Training Pipeline - Implementation Checklist
## ML Best Practices & Data Science Fixes

**Based on**: DATA_SCIENCE_CODE_REVIEW.md  
**Date**: 2025-10-08  
**Status**: 0/49 Complete

---

## 🔴 P0: CRITICAL (Must Fix Before Production)

### 1. Target Alignment & Label Semantics
- [ ] **1.1** Add `enforce_feature_temporal_alignment()` to `validation/schemas.py`
- [ ] **1.2** Create unit test `test_no_contemporaneous_feature_access()`
- [ ] **1.3** Add `feature_metadata` tracking with `lag` field to all features
- [ ] **1.4** Audit all feature families for `shift >= 1` compliance
- [ ] **1.5** Add `target_shift=h` field validation to label configs
- [ ] **1.6** Document minimum lag requirements in feature engineering guide

**Priority**: P0  
**Estimated Time**: 2 days  
**Risk if Skipped**: Lookahead bias in features → Invalid results

---

### 2. Ex-Post Scaling Leakage
- [ ] **2.1** Add `closed='left'` to all `rolling()` operations in volatility modeling
- [ ] **2.2** Implement `_enforce_past_only_windows()` helper function
- [ ] **2.3** Add `_validate_no_future_leakage()` to volatility module
- [ ] **2.4** Add test: `test_volatility_no_contemporaneous_correlation()`
- [ ] **2.5** Document embargo zones around label horizons
- [ ] **2.6** Verify no volatility window overlaps with label window

**Priority**: P0  
**Estimated Time**: 1.5 days  
**Risk if Skipped**: Volatility estimates leak future → Overfitting

---

### 3. Multiple Comparisons Correction
- [ ] **3.1** Implement `HypothesisTracker` class
- [ ] **3.2** Add `apply_multiple_testing_correction()` function
- [ ] **3.3** Add `report_hypothesis_count()` to all selection pipelines
- [ ] **3.4** Apply Benjamini-Hochberg FDR correction by default
- [ ] **3.5** Log `n_hypotheses` in all selection artifacts
- [ ] **3.6** Add warning if total hypotheses > 100
- [ ] **3.7** Update all p-value reporting to show adjusted values

**Priority**: P0  
**Estimated Time**: 1 day  
**Risk if Skipped**: P-hacking → False discoveries

---

### 4. Time-Series CV Implementation
- [ ] **4.1** Implement `purged_walk_forward_cv()` function
- [ ] **4.2** Add `validate_cv_no_leakage()` validation
- [ ] **4.3** Replace all `KFold` with purged walk-forward
- [ ] **4.4** Add test: `test_purged_cv_no_overlap()`
- [ ] **4.5** Add test: `test_embargo_enforced()`
- [ ] **4.6** Verify embargo >= label horizon in all CV
- [ ] **4.7** Document CV strategy in pipeline README

**Priority**: P0  
**Estimated Time**: 2 days  
**Risk if Skipped**: CV leakage → Overfit performance estimates

---

### 5. Normalization Leakage Prevention
- [ ] **5.1** Implement `SplitAwareScaler` wrapper class
- [ ] **5.2** Add `test_scaler_sees_only_train()` unit test
- [ ] **5.3** Replace all direct `StandardScaler().fit_transform()` calls
- [ ] **5.4** Audit all PCA/normalization for split awareness
- [ ] **5.5** Add validation that transform never refits
- [ ] **5.6** Document normalization strategy

**Priority**: P0  
**Estimated Time**: 1 day  
**Risk if Skipped**: Test set leakage via normalization statistics

---

## 🟡 P1: HIGH PRIORITY (Short-Term)

### 6. Trading-Aware Metrics
- [ ] **6.1** Implement `calculate_information_coefficient()` (IC)
- [ ] **6.2** Implement `calculate_cost_adjusted_sharpe()`
- [ ] **6.3** Implement `calculate_turnover_penalized_metric()`
- [ ] **6.4** Replace all RMSE/accuracy with IC + Sharpe
- [ ] **6.5** Add turnover to all strategy evaluations
- [ ] **6.6** Document metric choice rationale

**Priority**: P1  
**Estimated Time**: 1.5 days  
**Risk if Skipped**: Optimizing wrong objective → Impractical strategies

---

### 7. Temporal Leakage Linting
- [ ] **7.1** Implement `lint_for_temporal_leakage()` function
- [ ] **7.2** Add checks for `center=True` in rolling operations
- [ ] **7.3** Add checks for negative shifts outside label code
- [ ] **7.4** Add checks for missing `closed=` parameter
- [ ] **7.5** Create pre-commit hook script
- [ ] **7.6** Run linter on all existing feature files
- [ ] **7.7** Add to CI/CD pipeline

**Priority**: P1  
**Estimated Time**: 1 day  
**Risk if Skipped**: Leakage can slip in during development

---

### 8. Turnover & Capacity Modeling
- [ ] **8.1** Implement `calculate_turnover_metrics()`
- [ ] **8.2** Implement `apply_market_impact_model()`
- [ ] **8.3** Implement `reject_high_turnover_configs()`
- [ ] **8.4** Add turnover constraints to backtests
- [ ] **8.5** Add capacity limits to strategy evaluation
- [ ] **8.6** Report turnover in all artifacts
- [ ] **8.7** Add test: `test_turnover_within_limits()`

**Priority**: P1  
**Estimated Time**: 1.5 days  
**Risk if Skipped**: May select impractical/untradeable strategies

---

### 9. HAC-Robust Statistics
- [ ] **9.1** Implement `calculate_hac_robust_statistics()`
- [ ] **9.2** Use Newey-West for all IC calculations
- [ ] **9.3** Use Newey-West for all PnL statistics
- [ ] **9.4** Update all t-stat calculations
- [ ] **9.5** Update all p-value calculations
- [ ] **9.6** Add max_lags parameter (default: 12)
- [ ] **9.7** Document HAC adjustment in reports

**Priority**: P1  
**Estimated Time**: 1 day  
**Risk if Skipped**: Naive t-stats overstate significance

---

### 10. Nested CV for Hyperparameters
- [ ] **10.1** Implement `nested_cv_hyperparameter_selection()`
- [ ] **10.2** Add `validate_hyperparameter_stability()` checks
- [ ] **10.3** Add `penalize_extreme_lags()` in search
- [ ] **10.4** Replace all single-level CV with nested CV
- [ ] **10.5** Add MAD/median < 15% stability requirement
- [ ] **10.6** Use log-spaced grids for lag search
- [ ] **10.7** Document nested CV strategy

**Priority**: P1  
**Estimated Time**: 2 days  
**Risk if Skipped**: Overfit hyperparameters → Poor generalization

---

## 🟠 P2: MEDIUM PRIORITY

### 11. Block Permutation Importance
- [ ] **11.1** Implement `block_permutation_importance()`
- [ ] **11.2** Replace standard permutation importance
- [ ] **11.3** Use block size >= label horizon
- [ ] **11.4** Compute only on validation fold
- [ ] **11.5** Add test: `test_block_permutation_preserves_structure()`

**Priority**: P2  
**Estimated Time**: 0.5 days

---

### 12. Block Bootstrap CIs
- [ ] **12.1** Implement `block_bootstrap_confidence_intervals()`
- [ ] **12.2** Replace naive CIs with block bootstrap
- [ ] **12.3** Use appropriate block size (20-50 bars)
- [ ] **12.4** Report CIs for all key metrics
- [ ] **12.5** Add test: `test_block_bootstrap_coverage()`

**Priority**: P2  
**Estimated Time**: 0.5 days

---

### 13. Feature Metadata System
- [ ] **13.1** Create `FeatureMetadata` dataclass
- [ ] **13.2** Add `lag`, `window`, `closed` fields
- [ ] **13.3** Track metadata for all features
- [ ] **13.4** Add metadata validation functions
- [ ] **13.5** Serialize metadata with features

**Priority**: P2  
**Estimated Time**: 1 day

---

### 14. Stationarity Testing
- [ ] **14.1** Add ADF test for feature stationarity
- [ ] **14.2** Add KPSS test for feature stationarity
- [ ] **14.3** Record % stationary features in metadata
- [ ] **14.4** Warn if non-stationary features detected
- [ ] **14.5** Add test: `test_returns_are_stationary()`

**Priority**: P2  
**Estimated Time**: 0.5 days

---

### 15. VIF & Multicollinearity
- [ ] **15.1** Add VIF calculation to feature selection
- [ ] **15.2** Implement correlation clustering
- [ ] **15.3** Add VIF cap (e.g., VIF < 10)
- [ ] **15.4** Log dropped features due to multicollinearity
- [ ] **15.5** Report final VIF statistics

**Priority**: P2  
**Estimated Time**: 0.5 days

---

### 16. Uncertainty Quantification
- [ ] **16.1** Add reliability diagrams for classifiers
- [ ] **16.2** Calculate Brier score for probabilities
- [ ] **16.3** Implement conformal prediction for regressors
- [ ] **16.4** Add uncertainty metrics to reports
- [ ] **16.5** Visualize calibration curves

**Priority**: P2  
**Estimated Time**: 1 day

---

### 17. Documentation & Testing
- [ ] **17.1** Add docstrings to all new functions
- [ ] **17.2** Create usage examples for each fix
- [ ] **17.3** Update pipeline README with new requirements
- [ ] **17.4** Add integration test: `test_full_pipeline_no_leakage()`
- [ ] **17.5** Add to CI/CD: Run all temporal checks
- [ ] **17.6** Create troubleshooting guide

**Priority**: P2  
**Estimated Time**: 1 day

---

## Progress Tracking

### By Priority
- [ ] P0 Complete (0/28 items)
- [ ] P1 Complete (0/34 items)
- [ ] P2 Complete (0/27 items)

### By Category
- [ ] Target Alignment (0/6)
- [ ] Volatility Leakage (0/6)
- [ ] Multiple Testing (0/7)
- [ ] CV Implementation (0/7)
- [ ] Normalization (0/6)
- [ ] Trading Metrics (0/6)
- [ ] Temporal Linting (0/7)
- [ ] Turnover Modeling (0/7)
- [ ] Statistical Rigor (0/7)
- [ ] Nested CV (0/7)
- [ ] Other Improvements (0/23)

### Overall Progress
**0 / 89 items complete (0%)**

---

## Implementation Timeline

### Week 1 (P0 - Critical)
**Goal**: Fix all P0 issues
- Days 1-2: Target alignment (#1.1-1.6)
- Days 3-4: CV implementation (#4.1-4.7) + Volatility leakage (#2.1-2.6)
- Day 5: Multiple testing (#3.1-3.7) + Normalization (#5.1-5.6)

### Week 2 (P1 - High Priority)
**Goal**: Add trading-aware infrastructure
- Days 1-2: Trading metrics (#6.1-6.6) + Temporal linting (#7.1-7.7)
- Days 3-4: Statistical rigor (#9.1-9.7) + Nested CV (#10.1-10.7)
- Day 5: Turnover modeling (#8.1-8.7)

### Week 3 (P2 - Polish)
**Goal**: Complete remaining improvements
- Days 1-2: Block bootstrap + permutation importance (#11, #12)
- Days 3-4: Feature metadata + stationarity (#13, #14)
- Day 5: Documentation + testing (#17)

---

## Validation Checklist

Before marking as complete, verify:

### Code Quality
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] All functions have unit tests
- [ ] Code passes linting
- [ ] No TODO comments remain

### Testing
- [ ] Unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Edge cases tested
- [ ] Performance acceptable

### Documentation
- [ ] Usage examples provided
- [ ] API documented
- [ ] README updated
- [ ] Migration guide written (if breaking changes)

### Review
- [ ] Code reviewed by peer
- [ ] QA tested
- [ ] Performance profiled
- [ ] Security reviewed (if applicable)

---

## Success Criteria

### P0 Complete When:
1. All features have `lag >= 1` enforced
2. All CV uses purged walk-forward
3. All p-values are FDR-adjusted
4. All volatility uses `closed='left'`
5. All scalers are split-aware

### P1 Complete When:
6. All metrics are trading-aware (IC, Sharpe, turnover)
7. Temporal linting runs on every commit
8. Turnover constraints enforced in backtests
9. All statistics use HAC adjustment
10. All hyperparameter search uses nested CV

### P2 Complete When:
11. Block permutation importance used everywhere
12. All CIs use block bootstrap
13. Feature metadata tracked systematically
14. Stationarity tests integrated
15. Documentation complete

---

## Risk Mitigation

### If Timeline Slips:
1. **Must complete P0** before any deployment
2. P1 can be phased (start with #6, #9)
3. P2 can be deferred to maintenance cycle

### If Resource Constrained:
- Prioritize items #1.1-1.4 (temporal alignment)
- Then #4.1-4.4 (purged CV)
- Then #3.1-3.5 (FDR correction)

### If Blockers Arise:
- Document blocker in this file
- Implement workaround if safe
- Escalate to team lead

---

## Notes & Updates

### 2025-10-08
- Initial checklist created from code review
- 89 total items identified
- Estimated 3 weeks to complete all items

### (Add updates here as work progresses)

---

**Status**: 🔴 NOT STARTED  
**Target Completion**: (3 weeks from start date)  
**Owner**: (Assign team/person)  
**Reviewer**: (Assign reviewer)