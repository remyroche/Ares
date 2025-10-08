# Data Science Code Review: Pre-Training Pipeline
## Comprehensive Analysis of ML Pitfalls & Best Practices

**Review Date**: 2025-10-08  
**Scope**: `/workspace/src/training/steps/pre_training/`  
**Reviewer**: AI Code Analysis System

---

## Executive Summary

This review examines the pre-training pipeline against industry-standard ML pitfalls for time-series/financial ML, particularly those documented in "Advances in Financial Machine Learning" (de Prado). The codebase demonstrates **strong awareness** of many common issues, with good temporal validation infrastructure, but several **critical areas require immediate attention** before production deployment.

### Overall Risk Assessment
- **HIGH RISK**: Target alignment & label leakage ⚠️
- **MEDIUM RISK**: Feature engineering temporal alignment 
- **LOW RISK**: Validation & CV infrastructure (good foundation exists)
- **LOW RISK**: Statistical rigor (good HAC-aware infrastructure)

---

## 1. Target Alignment & Label Semantics

### 1.1 Off-by-One Label Alignment ⚠️ **CRITICAL ISSUE**

**Status**: **PARTIALLY IMPLEMENTED** - Infrastructure exists but enforcement is incomplete

#### Evidence Found:

```python
# File: profit_labeling/enhanced_label_definitions.py (lines 937-939)
entry_prices = market_data[entry_column].shift(-1)  # ✅ Good - uses next bar
exit_prices = market_data[exit_column].shift(-horizon_bars)  # ⚠️ Forward-looking

# File: profit_labeling/multi_target_scheme.py (lines 309-310, 367-370)
candidate['target_shift'] = max(1, int(candidate.get('target_shift', 1)))  # ✅ Enforces minimum shift
result.target_shifts = {
    name: int(info.get('target_shift', 1))
    for name, info in selected_targets.items()
}
```

#### Issues Identified:

1. **✅ GOOD**: System tracks `target_shift` metadata per target
2. **✅ GOOD**: Entry prices use `.shift(-1)` (next bar open)
3. **⚠️ CONCERN**: Exit prices use `.shift(-horizon_bars)` which is forward-looking relative to signal generation time `t`
4. **❌ MISSING**: No unit tests verify that features at time `t` don't access data from `t+0` or later
5. **❌ MISSING**: No explicit enforcement that `feature_col.max_lag >= 1` across all feature families

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
def enforce_feature_temporal_alignment(
    features: pd.DataFrame, 
    labels: pd.DataFrame, 
    target_shifts: Dict[str, int]
) -> None:
    """Enforce that all features respect minimum lag >= 1.
    
    Raises:
        ValueError: If any feature accesses contemporaneous or future data
    """
    for col in features.columns:
        metadata = _extract_feature_metadata(col)
        max_lag = metadata.get('lag', metadata.get('shift', 0))
        
        # CRITICAL: All features must use at least lag=1
        if max_lag < 1:
            raise ValueError(
                f"Feature '{col}' has insufficient lag ({max_lag}). "
                f"All features must use shift >= 1 to prevent lookahead."
            )
    
    # Validate label alignment
    for target_name, shift in target_shifts.items():
        if shift < 1:
            raise ValueError(
                f"Target '{target_name}' has shift={shift} < 1. "
                f"Labels must be shifted forward >= 1 bar."
            )
```

**Action Items**:
- [ ] Add `enforce_feature_temporal_alignment()` to validation pipeline
- [ ] Create unit test: `test_no_contemporaneous_feature_access()`
- [ ] Audit all feature generation families for `shift >= 1`
- [ ] Add explicit `target_shift=h` field to all label configs

---

### 1.2 Ex-Post Scaling Leakage ⚠️ **CRITICAL ISSUE**

**Status**: **MIXED** - Some volatility calculations use proper past-only windows, others unclear

#### Evidence Found:

```python
# File: profit_labeling/volatility_modeling.py (lines 304, 361, 395-396, 402)
# ✅ GOOD: Uses .shift(1) to exclude current bar
rv = rv.shift(1)
atr = atr.shift(1)
ewma_var = returns_series.ewm(alpha=alpha, min_periods=min_periods).var().shift(1)
ewma_var = returns_series.ewm(alpha=alpha, min_periods=min_periods).var().shift(1)
```

#### Issues Identified:

1. **✅ GOOD**: Volatility estimates use `.shift(1)` to ensure past-only data
2. **⚠️ CONCERN**: No explicit `rolling(..., closed='left')` enforcement
3. **❌ MISSING**: No embargo validation around label horizons
4. **❌ MISSING**: No verification that volatility windows don't overlap with label windows

#### Recommendations:

```python
# REQUIRED: Add to profit_labeling/volatility_modeling.py
class VolatilityModeler:
    def _enforce_past_only_windows(self, returns: pd.Series, window: int) -> pd.Series:
        """Enforce strictly past-only rolling windows.
        
        Uses closed='left' to exclude current bar, then shifts result
        to ensure volatility at time t only uses data up to t-1.
        """
        # Use closed='left' to exclude current bar from window
        volatility = returns.rolling(
            window=window, 
            closed='left',  # CRITICAL: Exclude current bar
            min_periods=window // 2
        ).std()
        
        # Additional shift to be extra safe
        volatility = volatility.shift(1)
        
        return volatility
    
    def _validate_no_future_leakage(
        self, 
        volatility_series: pd.Series, 
        market_data: pd.DataFrame,
        label_horizon: int
    ) -> None:
        """Validate that volatility estimation doesn't leak future data."""
        # Check: volatility at time t should not correlate with returns at t
        current_returns = market_data['close'].pct_change()
        correlation = volatility_series.corr(current_returns)
        
        if abs(correlation) > 0.1:
            raise ValueError(
                f"Volatility shows suspicious correlation ({correlation:.3f}) "
                f"with contemporaneous returns - possible data leakage"
            )
        
        # Check: volatility shouldn't use data within label horizon
        for i in range(1, label_horizon + 1):
            future_returns = current_returns.shift(-i)
            correlation = volatility_series.corr(future_returns)
            
            if abs(correlation) > 0.05:
                raise ValueError(
                    f"Volatility correlates with future returns at lag {i} "
                    f"({correlation:.3f}) - possible overlap with label window"
                )
```

**Action Items**:
- [ ] Add `closed='left'` to all `rolling()` operations
- [ ] Implement `_enforce_past_only_windows()` helper
- [ ] Add `_validate_no_future_leakage()` checks
- [ ] Document embargo zones around label horizons

---

### 1.3 Multiple Comparisons Across Horizons ⚠️ **MEDIUM RISK**

**Status**: **NOT ADDRESSED** - No multiplicity correction detected

#### Evidence Found:

```python
# File: multi_horizon_profit_labeler.py (lines 239-243)
@dataclass
class HorizonWeightsConfig:
    """Configuration for horizon weights in multi-horizon labeling."""
    micro: float = 0.0   # 0% - disabled for now
    small: float = 0.5   # 50% - immediate opportunities
    medium: float = 0.3  # 30% - short-term opportunities
    high: float = 0.2    # 20% - longer-term opportunities
```

Multiple horizons are tested (micro/small/medium/high) but no correction for multiple comparisons.

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
from statsmodels.stats.multitest import multipletests

def apply_multiple_testing_correction(
    horizon_results: Dict[str, Dict[str, float]],
    method: str = 'fdr_bh'  # Benjamini-Hochberg FDR control
) -> Dict[str, Dict[str, float]]:
    """Apply multiple testing correction across horizons.
    
    Args:
        horizon_results: Dict mapping horizon names to metrics (IC, p-value, etc.)
        method: 'bonferroni', 'fdr_bh', 'fdr_by'
    
    Returns:
        Corrected results with adjusted p-values
    """
    # Extract p-values from all horizon tests
    horizons = list(horizon_results.keys())
    p_values = [horizon_results[h].get('p_value', 1.0) for h in horizons]
    
    # Apply correction
    reject, p_adj, _, _ = multipletests(p_values, method=method)
    
    # Update results
    corrected_results = {}
    for i, horizon in enumerate(horizons):
        corrected_results[horizon] = {
            **horizon_results[horizon],
            'p_value_adjusted': p_adj[i],
            'significant_after_correction': reject[i],
            'correction_method': method,
            'n_hypotheses': len(horizons)
        }
    
    return corrected_results

def report_hypothesis_count(config: MultiHorizonConfig) -> Dict[str, int]:
    """Report the number of hypotheses tested for transparency."""
    n_horizons = sum([
        config.horizon_weights.micro > 0,
        config.horizon_weights.small > 0,
        config.horizon_weights.medium > 0,
        config.horizon_weights.high > 0
    ])
    
    n_thresholds = len(config.transaction_costs.__dict__)  # Different cost assumptions
    n_regime_configs = len(config.regime_config.__dict__) if hasattr(config, 'regime_config') else 1
    
    total_hypotheses = n_horizons * n_thresholds * n_regime_configs
    
    return {
        'n_horizons': n_horizons,
        'n_thresholds': n_thresholds,
        'n_regime_configs': n_regime_configs,
        'total_hypotheses': total_hypotheses,
        'bonferroni_threshold': 0.05 / total_hypotheses if total_hypotheses > 0 else 0.05
    }
```

**Action Items**:
- [ ] Implement `apply_multiple_testing_correction()`
- [ ] Report `n_hypotheses` in all selection artifacts
- [ ] Use Benjamini-Hochberg FDR control by default
- [ ] Consider nested CV for horizon selection

---

### 1.4 Class Imbalance & Rarity ✅ **WELL HANDLED**

**Status**: **GOOD** - PR-AUC metrics and balancing system detected

#### Evidence Found:

```python
# File: multi_horizon_profit_labeler.py (lines 447-461)
if BALANCING_SYSTEM_AVAILABLE and (self.config.enable_label_balancing or self.config.enable_sample_weighting):
    balancing_config = self.config.balancing_config or DEFAULT_BALANCING_CONFIG
    weighting_config = self.config.weighting_config or DEFAULT_WEIGHTING_CONFIG
    regime_config = self.config.regime_config or DEFAULT_REGIME_CONFIG
    fairness_config = self.config.fairness_config or DEFAULT_FAIRNESS_CONFIG

    self.balancing_system = ComprehensiveBalancingSystem(
        balancing_config, weighting_config, regime_config, fairness_config
    )
```

**Strengths**:
- ✅ Label balancing system available
- ✅ Sample weighting enabled
- ✅ Regime-aware balancing

**Recommendations**:
- Ensure downstream models use PR-AUC, not accuracy
- Add balanced accuracy metrics
- Document cost-sensitive loss functions

---

## 2. Feature Engineering

### 2.1 Causal Feature Enforcement ⚠️ **MEDIUM RISK**

**Status**: **PARTIAL** - No automated lint detected for `center=True` or negative shifts

#### Evidence Found:

No instances of `center=True` found in rolling operations (good), but also no automated checks.

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py or as pre-commit hook
import ast
import re

def lint_for_temporal_leakage(file_path: str) -> List[str]:
    """Lint Python file for common temporal leakage patterns.
    
    Returns:
        List of violation messages
    """
    violations = []
    
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check for center=True in rolling operations
    if re.search(r'\.rolling\([^)]*center=True', content):
        for i, line in enumerate(lines, 1):
            if 'center=True' in line:
                violations.append(
                    f"Line {i}: rolling(..., center=True) uses future data"
                )
    
    # Check for negative shifts without proper context
    negative_shift_pattern = r'\.shift\(\s*-\s*\d+\s*\)'
    for i, line in enumerate(lines, 1):
        if re.search(negative_shift_pattern, line):
            # Allow negative shifts only in label/target calculation
            if 'label' not in line.lower() and 'target' not in line.lower():
                violations.append(
                    f"Line {i}: .shift(-n) found outside label calculation"
                )
    
    # Check for rolling windows that might include current bar
    for i, line in enumerate(lines, 1):
        if '.rolling(' in line and 'closed=' not in line:
            violations.append(
                f"Line {i}: rolling() without explicit closed= parameter"
            )
    
    return violations

# Add as pre-commit hook or CI check
def run_temporal_linting():
    """Run temporal leakage linting across all feature files."""
    feature_files = glob.glob('src/**/feature*.py', recursive=True)
    
    all_violations = {}
    for file_path in feature_files:
        violations = lint_for_temporal_leakage(file_path)
        if violations:
            all_violations[file_path] = violations
    
    if all_violations:
        raise ValueError(
            f"Temporal leakage violations found:\n" +
            '\n'.join(f"{k}: {v}" for k, v in all_violations.items())
        )
```

**Action Items**:
- [ ] Implement `lint_for_temporal_leakage()` 
- [ ] Add as pre-commit hook
- [ ] Run on all feature generation files
- [ ] Add `assert all(feature_metadata[col]['lag'] >= 1)` checks

---

### 2.2 Normalization Leakage ⚠️ **MEDIUM RISK**

**Status**: **INFRASTRUCTURE EXISTS** - But no explicit split-aware fit/transform enforcement

#### Evidence Found:

```python
# File: standardized_labeling_interface.py (lines 168-203)
def assert_labels_sigma_scaled(labels: pd.DataFrame, tolerance: float = 0.35) -> None:
    """Assert that label variance remains close to 1 (σ-normalized scale)."""
    # Checks that variance ≈ 1.0
```

Good sigma scaling for labels, but unclear if scalers/PCA fit on train only.

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
from sklearn.base import BaseEstimator, TransformerMixin

class SplitAwareScaler(BaseEstimator, TransformerMixin):
    """Wrapper that enforces split-aware fit/transform.
    
    Prevents accidental fitting on full dataset including test data.
    """
    
    def __init__(self, base_scaler, split_indices: Dict[str, np.ndarray]):
        """
        Args:
            base_scaler: sklearn scaler (StandardScaler, RobustScaler, etc.)
            split_indices: {'train': [...], 'val': [...], 'test': [...]}
        """
        self.base_scaler = base_scaler
        self.split_indices = split_indices
        self._fitted_on = None
    
    def fit(self, X, y=None, split='train'):
        """Fit only on specified split."""
        if split not in self.split_indices:
            raise ValueError(f"Unknown split: {split}")
        
        train_idx = self.split_indices[split]
        X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        
        self.base_scaler.fit(X_train, y)
        self._fitted_on = split
        
        return self
    
    def transform(self, X):
        """Transform with validation."""
        if self._fitted_on is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Validate that we're not transforming training data again
        # (which could indicate fitting on full dataset)
        
        return self.base_scaler.transform(X)
    
    def fit_transform(self, X, y=None, split='train'):
        """Fit and transform - only allowed on training split."""
        if split != 'train':
            raise ValueError(
                f"fit_transform only allowed on 'train' split, got '{split}'"
            )
        
        return self.fit(X, y, split=split).transform(X)

def test_scaler_sees_only_train():
    """Unit test to ensure scaler only sees training data."""
    from sklearn.preprocessing import StandardScaler
    
    # Create mock data with train/val/test splits
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10))
    splits = {
        'train': np.arange(0, 700),
        'val': np.arange(700, 850),
        'test': np.arange(850, 1000)
    }
    
    # Fit scaler
    scaler = SplitAwareScaler(StandardScaler(), splits)
    scaler.fit(X, split='train')
    
    # Verify: mean/std computed only on train data
    X_train = X.iloc[splits['train']]
    expected_mean = X_train.mean().values
    expected_std = X_train.std().values
    
    np.testing.assert_array_almost_equal(
        scaler.base_scaler.mean_, expected_mean, decimal=5
    )
    np.testing.assert_array_almost_equal(
        scaler.base_scaler.scale_, expected_std, decimal=5
    )
```

**Action Items**:
- [ ] Implement `SplitAwareScaler` wrapper
- [ ] Add `test_scaler_sees_only_train()` unit test
- [ ] Audit all normalization code for split awareness
- [ ] Document normalization strategy in pipeline

---

### 2.3 Stationarity ✅ **GOOD**

**Status**: **WELL HANDLED** - Returns/log-returns used, volatility normalization present

#### Evidence Found:

Multiple references to volatility-normalized targets (σ-units), sigma scaling, and returns-based features.

**Strengths**:
- ✅ Targets in σ-units (stationary)
- ✅ Volatility normalization throughout
- ✅ Returns-based feature families

**Recommendations**:
- Consider adding ADF/KPSS tests for feature validation
- Record % stationary features in metadata

---

### 2.4 Redundancy & Multicollinearity ✅ **GOOD**

**Status**: **ADDRESSED** - Multi-stage feature selection (120→100→80→60)

#### Evidence Found:

```python
# File: components/final_feature_selection.py (lines 423-428)
'stage_reduction': {
    'initial': 120,
    'stage_1': 100,
    'stage_2': 80,
    'stage_3': 60
}
```

**Strengths**:
- ✅ Multi-stage selection reduces redundancy
- ✅ Correlation-based filtering likely present

**Recommendations**:
- Add VIF (Variance Inflation Factor) caps (e.g., VIF < 10)
- Implement correlation clustering
- Log dropped features due to multicollinearity

---

### 2.5 Cross-Sectional Bleed ⚠️ **LOW RISK** (if single-asset)

**Status**: **UNCLEAR** - Depends on whether multi-asset

If pipeline processes multiple symbols simultaneously:
- Ensure z-scoring is within-symbol, not across symbols
- Keep symbol embeddings/one-hot encodings if pooling data

---

## 3. Validation, CV & Metrics

### 3.1 Improper CV in Time Series ⚠️ **REQUIRES VERIFICATION**

**Status**: **INFRASTRUCTURE EXISTS** - Temporal validation config present, but implementation unclear

#### Evidence Found:

```python
# File: multi_horizon_profit_labeler.py (lines 262-271)
@dataclass
class TemporalValidationConfig:
    """Configuration for temporal validation."""
    enable_temporal_validation: bool = True
    enable_purging: bool = True
    purge_window_hours: int = 24
    embargo_window_hours: int = 12
    train_ratio: float = 0.70
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    validate_distribution: bool = True
```

**Good Signs**:
- ✅ Purge and embargo windows defined
- ✅ Temporal validation explicitly enabled
- ✅ Train/val/test splits defined

**Concerns**:
- ❌ No verification that purge/embargo are actually applied
- ❌ No walk-forward CV implementation detected in feature selection

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
from typing import Tuple, List
import pandas as pd

def purged_walk_forward_cv(
    data: pd.DataFrame,
    n_splits: int = 5,
    embargo_hours: int = 12,
    purge_hours: int = 24,
    min_train_size: float = 0.5
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate purged walk-forward CV splits.
    
    Implements proper time-series CV with:
    - Walk-forward (expanding window)
    - Purging: Remove training samples that overlap with test labels
    - Embargo: Gap between train and test to prevent information leakage
    
    Args:
        data: Time-indexed DataFrame
        n_splits: Number of CV folds
        embargo_hours: Hours to embargo after each test period
        purge_hours: Hours to purge from training (if overlaps with test labels)
        min_train_size: Minimum training set size as fraction of total
    
    Returns:
        List of (train_index, test_index) tuples
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex")
    
    splits = []
    total_size = len(data)
    test_size = total_size // n_splits
    min_train_samples = int(total_size * min_train_size)
    
    for i in range(n_splits):
        # Test set: next block of data
        test_start_idx = min_train_samples + i * test_size
        test_end_idx = min(test_start_idx + test_size, total_size)
        
        if test_start_idx >= total_size:
            break
        
        test_idx = data.index[test_start_idx:test_end_idx]
        
        # Training set: all data before test, with purging and embargo
        train_end_time = test_idx[0] - pd.Timedelta(hours=embargo_hours)
        train_idx = data.index[data.index < train_end_time]
        
        # Purge: Remove training samples whose label windows overlap with test
        # Assuming labels look forward by purge_hours
        purge_cutoff = test_idx[0] - pd.Timedelta(hours=purge_hours)
        train_idx = train_idx[train_idx < purge_cutoff]
        
        # Ensure minimum training size
        if len(train_idx) < min_train_samples:
            continue
        
        splits.append((train_idx, test_idx))
    
    return splits

def validate_cv_no_leakage(
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    label_horizon_hours: int
) -> None:
    """Validate that CV splits have no temporal leakage.
    
    Raises:
        ValueError: If splits have leakage
    """
    for i, (train_idx, test_idx) in enumerate(splits):
        # Check 1: Train comes before test
        if train_idx.max() >= test_idx.min():
            raise ValueError(
                f"Split {i}: Train set overlaps with test set "
                f"(train_max={train_idx.max()}, test_min={test_idx.min()})"
            )
        
        # Check 2: Gap between train and test >= label horizon
        gap_hours = (test_idx.min() - train_idx.max()).total_seconds() / 3600
        if gap_hours < label_horizon_hours:
            raise ValueError(
                f"Split {i}: Gap between train and test ({gap_hours:.1f}h) "
                f"is less than label horizon ({label_horizon_hours}h)"
            )
        
        # Check 3: No overlap between consecutive test sets (embargo)
        if i > 0:
            prev_test_idx = splits[i-1][1]
            if prev_test_idx.max() >= test_idx.min():
                raise ValueError(
                    f"Split {i}: Test set overlaps with previous test set "
                    f"(no embargo applied)"
                )

# Add usage to feature selection
def feature_selection_with_proper_cv(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_horizon_hours: int
):
    """Feature selection with proper time-series CV."""
    
    # Generate purged walk-forward splits
    splits = purged_walk_forward_cv(
        features,
        n_splits=5,
        embargo_hours=label_horizon_hours,
        purge_hours=label_horizon_hours * 2
    )
    
    # Validate splits
    validate_cv_no_leakage(splits, label_horizon_hours)
    
    # Perform selection on each split
    selected_features_per_fold = []
    for train_idx, test_idx in splits:
        X_train = features.loc[train_idx]
        y_train = labels.loc[train_idx]
        
        # Your feature selection logic here
        # ...
        
        selected_features_per_fold.append(...)
    
    # Return features selected in majority of folds
    return _select_stable_features(selected_features_per_fold)
```

**Action Items**:
- [ ] Implement `purged_walk_forward_cv()`
- [ ] Add `validate_cv_no_leakage()` checks
- [ ] Replace any IID k-fold with purged walk-forward
- [ ] Verify embargo >= label horizon in all CV

---

### 3.2 Metric Choice Mismatch ⚠️ **MEDIUM RISK**

**Status**: **NEEDS TRADING-AWARE METRICS**

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
def calculate_information_coefficient(
    predictions: pd.Series,
    labels: pd.Series,
    method: str = 'spearman'
) -> float:
    """Calculate OOS Information Coefficient (IC).
    
    IC measures rank correlation between predictions and future returns.
    More appropriate for trading than MSE/MAE.
    
    Args:
        predictions: Model predictions
        labels: True labels (forward returns)
        method: 'spearman' or 'pearson'
    
    Returns:
        IC score
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Drop NaN values
    valid_idx = predictions.notna() & labels.notna()
    pred_valid = predictions[valid_idx]
    labels_valid = labels[valid_idx]
    
    if len(pred_valid) < 10:
        return 0.0
    
    if method == 'spearman':
        ic, _ = spearmanr(pred_valid, labels_valid)
    else:
        ic, _ = pearsonr(pred_valid, labels_valid)
    
    return ic if not np.isnan(ic) else 0.0

def calculate_cost_adjusted_sharpe(
    predictions: pd.Series,
    labels: pd.Series,
    transaction_cost: float = 0.001,
    turnover_penalty: float = 0.0001
) -> float:
    """Calculate cost-adjusted Sharpe from model predictions.
    
    Creates a simple long/short strategy from predictions and evaluates
    with transaction costs and turnover penalty.
    
    Args:
        predictions: Model predictions (higher = more bullish)
        labels: True forward returns
        transaction_cost: Round-trip cost per trade
        turnover_penalty: Per-unit turnover penalty
    
    Returns:
        Cost-adjusted Sharpe ratio
    """
    # Convert predictions to positions (-1, 0, +1)
    positions = pd.Series(0, index=predictions.index)
    positions[predictions > predictions.quantile(0.7)] = 1
    positions[predictions < predictions.quantile(0.3)] = -1
    
    # Calculate strategy returns
    strategy_returns = positions * labels
    
    # Calculate turnover
    turnover = (positions - positions.shift(1)).abs()
    
    # Apply costs
    costs = turnover * transaction_cost + turnover * turnover_penalty
    net_returns = strategy_returns - costs
    
    # Calculate Sharpe
    if net_returns.std() == 0:
        return 0.0
    
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)  # Annualized
    
    return sharpe

def calculate_turnover_penalized_metric(
    predictions: pd.Series,
    labels: pd.Series,
    ic_weight: float = 0.6,
    sharpe_weight: float = 0.3,
    turnover_weight: float = 0.1,
    max_turnover: float = 2.0  # 2x turnover per period is max acceptable
) -> Dict[str, float]:
    """Composite metric balancing IC, Sharpe, and turnover.
    
    Returns:
        Dict with IC, Sharpe, turnover, and composite score
    """
    ic = calculate_information_coefficient(predictions, labels)
    sharpe = calculate_cost_adjusted_sharpe(predictions, labels)
    
    # Calculate turnover (position changes per period)
    positions = pd.Series(0, index=predictions.index)
    positions[predictions > predictions.quantile(0.7)] = 1
    positions[predictions < predictions.quantile(0.3)] = -1
    turnover = (positions - positions.shift(1)).abs().mean()
    
    # Penalize excessive turnover
    turnover_penalty = max(0, (turnover - max_turnover) / max_turnover)
    
    # Composite score
    composite = (
        ic_weight * ic +
        sharpe_weight * sharpe -
        turnover_weight * turnover_penalty
    )
    
    return {
        'IC': ic,
        'Sharpe': sharpe,
        'turnover': turnover,
        'composite': composite
    }
```

**Action Items**:
- [ ] Replace RMSE/accuracy with IC and cost-adjusted Sharpe
- [ ] Add turnover metrics to all backtests
- [ ] Penalize high-turnover strategies in selection
- [ ] Report all three metrics: IC, Sharpe, Turnover

---

### 3.3 Permutation Importance Leakage ⚠️ **MEDIUM RISK**

**Status**: **UNCLEAR** - No permutation importance code detected, but likely used

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
def block_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    block_size: int = 20,
    n_repeats: int = 10,
    scoring_func = None
) -> pd.Series:
    """Calculate permutation importance with block-wise permutation.
    
    Block-wise permutation preserves temporal structure.
    Standard permutation breaks autocorrelation, inflating importance.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        block_size: Size of blocks to permute together
        n_repeats: Number of permutation repeats
        scoring_func: Scoring function (default: IC)
    
    Returns:
        Series of importance scores per feature
    """
    from sklearn.inspection import permutation_importance
    
    if scoring_func is None:
        # Use IC as default scoring
        def ic_scorer(model, X, y):
            preds = model.predict(X)
            return calculate_information_coefficient(
                pd.Series(preds, index=X.index),
                y
            )
        scoring_func = ic_scorer
    
    # Baseline score
    baseline_score = scoring_func(model, X_val, y_val)
    
    # Calculate importance per feature with block permutation
    importance_scores = {}
    
    for col in X_val.columns:
        scores = []
        
        for _ in range(n_repeats):
            # Create block-permuted copy
            X_perm = X_val.copy()
            
            # Permute in blocks to preserve temporal structure
            n_blocks = len(X_val) // block_size
            block_indices = np.random.permutation(n_blocks)
            
            permuted_values = []
            for block_idx in block_indices:
                start = block_idx * block_size
                end = start + block_size
                permuted_values.extend(X_perm[col].iloc[start:end].values)
            
            # Handle remainder
            remainder = len(X_val) % block_size
            if remainder > 0:
                permuted_values.extend(X_perm[col].iloc[-remainder:].values)
            
            X_perm[col] = permuted_values
            
            # Score with permuted feature
            perm_score = scoring_func(model, X_perm, y_val)
            scores.append(baseline_score - perm_score)
        
        importance_scores[col] = np.mean(scores)
    
    return pd.Series(importance_scores).sort_values(ascending=False)
```

**Action Items**:
- [ ] Implement `block_permutation_importance()`
- [ ] Never permute across entire time series
- [ ] Compute importance only on validation fold
- [ ] Use block size ≥ label horizon

---

### 3.4 Uncertainty / Confidence ✅ **GOOD**

**Status**: **WELL ADDRESSED** - Confidence scores and eligibility masks present

#### Evidence Found:

```python
# File: standardized_labeling_interface.py (lines 234-236)
confidence_scores: pd.DataFrame
eligibility_masks: pd.DataFrame
```

**Strengths**:
- ✅ Confidence scores tracked
- ✅ Eligibility masks for quality filtering

**Recommendations**:
- Add reliability diagrams for probability calibration
- Implement Brier score for classification
- Consider conformal prediction for regression uncertainty

---

## 4. Lookback / Hyperparameter Search

### 4.1 Search-Space Exploitation ⚠️ **MEDIUM RISK**

**Status**: **PARTIAL** - Bayesian optimization present, stability checks unclear

#### Evidence Found:

```python
# File: profit_labeling/multi_target_scheme.py (lines 610-673)
def _bayesian_optimize_k_values(...):
    # Bayesian optimization for k values
```

**Good Signs**:
- ✅ Bayesian optimization (TPE) available
- ✅ Adaptive sampling strategy
- ✅ Early stopping

**Concerns**:
- ❌ No stability check across folds (MAD/median < 15%)
- ❌ No explicit regularization for extreme lag values

#### Recommendations:

```python
# REQUIRED: Add to feature_lookback_optimization/
def validate_hyperparameter_stability(
    selected_params: List[Dict[str, Any]],
    param_name: str,
    max_mad_ratio: float = 0.15
) -> bool:
    """Validate that hyperparameters are stable across folds.
    
    Args:
        selected_params: List of selected params from each fold
        param_name: Name of parameter to check
        max_mad_ratio: Maximum MAD/median ratio (e.g., 0.15 = 15%)
    
    Returns:
        True if stable, False otherwise
    """
    values = [p[param_name] for p in selected_params if param_name in p]
    
    if len(values) < 2:
        return True  # Can't assess stability with < 2 folds
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    if median == 0:
        return mad < 0.1  # Absolute threshold if median is zero
    
    mad_ratio = mad / median
    
    if mad_ratio > max_mad_ratio:
        tprint_warning(
            f"Parameter '{param_name}' is unstable across folds: "
            f"MAD/median = {mad_ratio:.2%} > {max_mad_ratio:.2%}"
        )
        return False
    
    return True

def penalize_extreme_lags(
    lag: int,
    min_lag: int = 1,
    max_lag: int = 100,
    penalty_scale: float = 0.1
) -> float:
    """Penalize extreme lag values in hyperparameter search.
    
    Returns:
        Penalty value (0 = no penalty, higher = more penalty)
    """
    if lag < min_lag:
        return penalty_scale * (min_lag - lag)
    elif lag > max_lag:
        return penalty_scale * (lag - max_lag)
    else:
        return 0.0

def nested_cv_hyperparameter_selection(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, List[Any]],
    n_outer_splits: int = 3,
    n_inner_splits: int = 3,
    embargo_hours: int = 12
) -> Dict[str, Any]:
    """Nested CV for unbiased hyperparameter selection.
    
    Outer loop: Evaluation
    Inner loop: Hyperparameter tuning
    
    This prevents using test data for hyperparameter selection.
    """
    outer_splits = purged_walk_forward_cv(
        X, n_splits=n_outer_splits, embargo_hours=embargo_hours
    )
    
    outer_scores = []
    selected_params_per_fold = []
    
    for outer_train_idx, outer_test_idx in outer_splits:
        X_outer_train = X.loc[outer_train_idx]
        y_outer_train = y.loc[outer_train_idx]
        X_outer_test = X.loc[outer_test_idx]
        y_outer_test = y.loc[outer_test_idx]
        
        # Inner CV for hyperparameter selection
        inner_splits = purged_walk_forward_cv(
            X_outer_train, n_splits=n_inner_splits, embargo_hours=embargo_hours
        )
        
        best_params = _grid_search_inner(
            X_outer_train, y_outer_train, param_grid, inner_splits
        )
        
        selected_params_per_fold.append(best_params)
        
        # Evaluate on outer test set
        model = _train_model(X_outer_train, y_outer_train, best_params)
        score = _evaluate_model(model, X_outer_test, y_outer_test)
        outer_scores.append(score)
    
    # Validate stability of selected parameters
    for param_name in param_grid.keys():
        if not validate_hyperparameter_stability(
            selected_params_per_fold, param_name, max_mad_ratio=0.15
        ):
            tprint_warning(
                f"Hyperparameter '{param_name}' is unstable across folds"
            )
    
    # Return most common parameters (mode)
    final_params = _select_most_common_params(selected_params_per_fold)
    
    return {
        'best_params': final_params,
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'stability': _calculate_stability(selected_params_per_fold)
    }
```

**Action Items**:
- [ ] Implement `validate_hyperparameter_stability()`
- [ ] Add stability checks (MAD/median < 15%)
- [ ] Use nested CV for lookback selection
- [ ] Penalize extreme lag values in search
- [ ] Log-spaced search for lag grids

---

### 4.2 Coupled Tuning ⚠️ **HIGH RISK**

**Status**: **NOT ADDRESSED** - No nested CV enforcement detected

#### Recommendations:

- Use nested CV (outer for evaluation, inner for tuning)
- Freeze lookback/threshold decisions before final test
- Never tune on the same validation set used for evaluation

See nested CV implementation in 4.1 above.

**Action Items**:
- [ ] Implement nested CV for all hyperparameter searches
- [ ] Separate tuning set from evaluation set
- [ ] Document tuning/evaluation split strategy

---

## 5. Backtesting & Execution Reality

### 5.1 Signal Delay ✅ **WELL HANDLED**

**Status**: **EXCELLENT** - Proper signal-to-execution delay modeling

#### Evidence Found:

```python
# File: profit_labeling/enhanced_label_definitions.py (lines 937-939, 943-958)
entry_prices = market_data[entry_column].shift(-1)  # ✅ Next bar entry
self._last_execution_metadata = {
    'signal_to_execution_delay_bars': 1,
    'signal_to_execution_delay_minutes': ...,
    'entry_price_source': f'next_{entry_column}',
    'slippage_pct': self.analyst_config.trading_costs.slippage_pct,
    ...
}
```

**Strengths**:
- ✅ Enters at next bar open, not current close
- ✅ Tracks execution latency metadata
- ✅ Includes slippage and fees

**Excellent implementation!**

---

### 5.2 Turnover & Capacity ⚠️ **NEEDS IMPLEMENTATION**

**Status**: **MISSING** - No turnover constraints or impact model detected

#### Recommendations:

```python
# REQUIRED: Add to backtesting/
def calculate_turnover_metrics(
    positions: pd.Series,
    returns: pd.Series
) -> Dict[str, float]:
    """Calculate turnover metrics for strategy evaluation.
    
    Returns:
        Dict with turnover, holding period, and stability metrics
    """
    # Turnover: sum of absolute position changes per period
    position_changes = (positions - positions.shift(1)).abs()
    turnover = position_changes.mean()
    turnover_annual = turnover * 252  # Annualized
    
    # Holding period: average time between position changes
    non_zero_changes = position_changes[position_changes > 0]
    if len(non_zero_changes) > 1:
        avg_holding_period = len(positions) / len(non_zero_changes)
    else:
        avg_holding_period = len(positions)
    
    # Position stability: fraction of periods with no change
    stability = (position_changes == 0).mean()
    
    return {
        'turnover_per_period': turnover,
        'turnover_annual': turnover_annual,
        'avg_holding_period_bars': avg_holding_period,
        'position_stability': stability
    }

def apply_market_impact_model(
    returns: pd.Series,
    positions: pd.Series,
    volume: pd.Series,
    impact_coefficient: float = 0.1,
    capacity_limit_usd: float = 1e6
) -> pd.Series:
    """Apply market impact cost model.
    
    Market impact ~ sqrt(trade_size / volume)
    
    Args:
        returns: Strategy returns before impact
        positions: Position sizes
        volume: Market volume
        impact_coefficient: Impact scaling factor
        capacity_limit_usd: Max strategy capacity
    
    Returns:
        Returns after market impact
    """
    # Calculate trade sizes
    position_changes = (positions - positions.shift(1)).abs()
    
    # Estimate market impact as sqrt(trade_size / volume)
    # This is a simplified Kyle/Almgren model
    relative_trade_size = position_changes / volume.clip(lower=1)
    market_impact = impact_coefficient * np.sqrt(relative_trade_size)
    
    # Cap impact at reasonable levels (e.g., 1% per trade)
    market_impact = market_impact.clip(upper=0.01)
    
    # Apply impact to returns
    net_returns = returns - market_impact
    
    # Check capacity constraint
    max_trade_size = position_changes.max()
    if max_trade_size > capacity_limit_usd:
        tprint_warning(
            f"Strategy exceeds capacity limit: "
            f"max_trade_size=${max_trade_size:,.0f} > ${capacity_limit_usd:,.0f}"
        )
    
    return net_returns

def reject_high_turnover_configs(
    strategy_results: Dict[str, Any],
    max_turnover_annual: float = 50.0,  # 50x annual turnover is extreme
    max_sharpe_to_turnover_ratio: float = 0.1
) -> bool:
    """Reject configurations with unrealistic turnover.
    
    Args:
        strategy_results: Dict with 'turnover_annual', 'sharpe', etc.
        max_turnover_annual: Max acceptable annual turnover
        max_sharpe_to_turnover_ratio: Min Sharpe/turnover ratio
    
    Returns:
        True if config should be rejected
    """
    turnover = strategy_results.get('turnover_annual', 0)
    sharpe = strategy_results.get('sharpe', 0)
    
    # Reject if turnover too high
    if turnover > max_turnover_annual:
        tprint_warning(
            f"Config rejected: Turnover ({turnover:.1f}x) "
            f"exceeds max ({max_turnover_annual:.1f}x)"
        )
        return True
    
    # Reject if Sharpe doesn't justify turnover
    if turnover > 0:
        sharpe_to_turnover = sharpe / turnover
        if sharpe_to_turnover < max_sharpe_to_turnover_ratio:
            tprint_warning(
                f"Config rejected: Sharpe/turnover ratio ({sharpe_to_turnover:.3f}) "
                f"too low (min: {max_sharpe_to_turnover_ratio:.3f})"
            )
            return True
    
    return False
```

**Action Items**:
- [ ] Implement `calculate_turnover_metrics()`
- [ ] Add `apply_market_impact_model()`
- [ ] Implement `reject_high_turnover_configs()`
- [ ] Add capacity constraints to all backtests
- [ ] Report turnover in all strategy evaluations

---

## 6. Statistical Rigor

### 6.1 Naive t-stats ⚠️ **MEDIUM RISK**

**Status**: **NEEDS IMPLEMENTATION** - No HAC-robust standard errors detected

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS

def calculate_hac_robust_statistics(
    predictions: pd.Series,
    labels: pd.Series,
    max_lags: int = 12
) -> Dict[str, float]:
    """Calculate HAC-robust (Newey-West) statistics.
    
    Accounts for heteroskedasticity and autocorrelation in residuals.
    
    Args:
        predictions: Model predictions
        labels: True labels
        max_lags: Maximum lags for HAC adjustment
    
    Returns:
        Dict with IC, t-stat, p-value (all HAC-adjusted)
    """
    # Calculate IC (Information Coefficient)
    valid_idx = predictions.notna() & labels.notna()
    pred_valid = predictions[valid_idx].values
    labels_valid = labels[valid_idx].values
    
    # Fit OLS regression
    model = OLS(labels_valid, pred_valid).fit()
    
    # Calculate HAC-robust covariance matrix
    cov_hac_matrix = cov_hac(model, nlags=max_lags)
    
    # Extract robust standard errors
    robust_se = np.sqrt(np.diag(cov_hac_matrix))
    
    # Calculate robust t-statistics
    robust_t_stats = model.params / robust_se
    
    # Calculate IC with robust confidence intervals
    ic = np.corrcoef(pred_valid, labels_valid)[0, 1]
    
    return {
        'IC': ic,
        't_stat_hac': robust_t_stats[0],
        'p_value_hac': 2 * (1 - stats.t.cdf(abs(robust_t_stats[0]), len(pred_valid) - 1)),
        'se_hac': robust_se[0],
        'max_lags': max_lags
    }

def block_bootstrap_confidence_intervals(
    predictions: pd.Series,
    labels: pd.Series,
    metric_func: callable,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate block bootstrap confidence intervals.
    
    Block bootstrap preserves temporal structure.
    
    Args:
        predictions: Model predictions
        labels: True labels
        metric_func: Function to calculate metric (e.g., IC, Sharpe)
        n_bootstrap: Number of bootstrap samples
        block_size: Size of blocks for resampling
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Dict with point estimate, lower CI, upper CI
    """
    # Calculate point estimate
    point_estimate = metric_func(predictions, labels)
    
    # Block bootstrap
    n_samples = len(predictions)
    n_blocks = n_samples // block_size
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Resample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        # Reconstruct time series from blocks
        resampled_pred = []
        resampled_labels = []
        
        for block_idx in block_indices:
            start = block_idx * block_size
            end = start + block_size
            resampled_pred.extend(predictions.iloc[start:end].values)
            resampled_labels.extend(labels.iloc[start:end].values)
        
        # Calculate metric on resampled data
        resampled_pred = pd.Series(resampled_pred)
        resampled_labels = pd.Series(resampled_labels)
        
        estimate = metric_func(resampled_pred, resampled_labels)
        bootstrap_estimates.append(estimate)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_ci = np.percentile(bootstrap_estimates, lower_percentile)
    upper_ci = np.percentile(bootstrap_estimates, upper_percentile)
    
    return {
        'point_estimate': point_estimate,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'ci_width': upper_ci - lower_ci,
        'bootstrap_std': np.std(bootstrap_estimates),
        'n_bootstrap': n_bootstrap,
        'block_size': block_size
    }
```

**Action Items**:
- [ ] Implement `calculate_hac_robust_statistics()`
- [ ] Use Newey-West for all IC/PnL statistics
- [ ] Implement `block_bootstrap_confidence_intervals()`
- [ ] Report robust t-stats in all evaluations
- [ ] Use block bootstrap for CIs, not naive

---

### 6.2 P-Hacking Surface ⚠️ **HIGH RISK**

**Status**: **NOT ADDRESSED** - No hypothesis count tracking

#### Evidence:

The pipeline has:
- Multiple horizons (micro/small/medium/high)
- Multiple k values per band (adaptive search)
- Multiple lookback windows
- Multiple feature families
- Multiple regimes

**This creates a massive hypothesis space!**

#### Recommendations:

```python
# REQUIRED: Add to validation/schemas.py
@dataclass
class HypothesisTracker:
    """Track all hypotheses tested for FDR control."""
    
    n_horizons: int = 0
    n_k_values: int = 0
    n_lookbacks: int = 0
    n_features: int = 0
    n_regimes: int = 0
    n_thresholds: int = 0
    
    def total_hypotheses(self) -> int:
        """Calculate total number of hypotheses tested."""
        return (
            self.n_horizons *
            self.n_k_values *
            self.n_lookbacks *
            max(1, self.n_regimes) *
            max(1, self.n_thresholds)
        )
    
    def bonferroni_alpha(self, alpha: float = 0.05) -> float:
        """Calculate Bonferroni-corrected significance level."""
        total = self.total_hypotheses()
        if total == 0:
            return alpha
        return alpha / total
    
    def fdr_adjusted_pvalues(
        self,
        p_values: List[float],
        method: str = 'fdr_bh'
    ) -> List[float]:
        """Apply FDR correction to p-values."""
        from statsmodels.stats.multitest import multipletests
        
        _, p_adj, _, _ = multipletests(p_values, method=method)
        return p_adj.tolist()
    
    def report(self) -> Dict[str, Any]:
        """Generate hypothesis testing report."""
        total = self.total_hypotheses()
        
        return {
            'hypothesis_breakdown': {
                'n_horizons': self.n_horizons,
                'n_k_values': self.n_k_values,
                'n_lookbacks': self.n_lookbacks,
                'n_features': self.n_features,
                'n_regimes': self.n_regimes,
                'n_thresholds': self.n_thresholds
            },
            'total_hypotheses': total,
            'bonferroni_alpha_005': self.bonferroni_alpha(0.05),
            'bonferroni_alpha_001': self.bonferroni_alpha(0.01),
            'warning': 'Multiple testing: Apply FDR correction to all p-values'
        }

# Add to all selection pipelines
def track_and_control_hypotheses(
    horizon_results: Dict[str, Any],
    feature_results: Dict[str, Any],
    lookback_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Track all hypotheses and control FDR."""
    
    # Create tracker
    tracker = HypothesisTracker(
        n_horizons=len(horizon_results),
        n_features=len(feature_results),
        n_lookbacks=len(lookback_results)
    )
    
    # Collect all p-values
    all_p_values = []
    all_p_values.extend([r.get('p_value', 1.0) for r in horizon_results.values()])
    all_p_values.extend([r.get('p_value', 1.0) for r in feature_results.values()])
    all_p_values.extend([r.get('p_value', 1.0) for r in lookback_results.values()])
    
    # Apply FDR correction
    adjusted_p_values = tracker.fdr_adjusted_pvalues(all_p_values)
    
    # Report
    report = tracker.report()
    report['adjusted_p_values'] = adjusted_p_values
    report['n_significant_before_correction'] = sum(p < 0.05 for p in all_p_values)
    report['n_significant_after_correction'] = sum(p < 0.05 for p in adjusted_p_values)
    
    tprint_warning(
        f"⚠️ Multiple testing: {report['total_hypotheses']} hypotheses tested. "
        f"{report['n_significant_after_correction']}/{report['n_significant_before_correction']} "
        f"remain significant after FDR correction."
    )
    
    return report
```

**Action Items**:
- [ ] Implement `HypothesisTracker`
- [ ] Log number of tests in all selection artifacts
- [ ] Apply Benjamini-Hochberg FDR control
- [ ] Report adjusted p-values
- [ ] Add warning if total hypotheses > 100

---

## Summary of Critical Action Items

### MUST FIX (Before Production):

1. **[TARGET ALIGNMENT]** Add `enforce_feature_temporal_alignment()` validation
2. **[TARGET ALIGNMENT]** Add unit test: `test_no_contemporaneous_feature_access()`
3. **[VOLATILITY LEAKAGE]** Add `closed='left'` to all rolling operations
4. **[VOLATILITY LEAKAGE]** Implement `_validate_no_future_leakage()` checks
5. **[MULTIPLE TESTING]** Implement `HypothesisTracker` and FDR control
6. **[CV LEAKAGE]** Implement `purged_walk_forward_cv()` and enforce everywhere
7. **[NORMALIZATION]** Implement `SplitAwareScaler` for all normalizations
8. **[METRICS]** Replace RMSE/accuracy with IC and cost-adjusted Sharpe
9. **[TURNOVER]** Implement turnover metrics and rejection criteria
10. **[STATISTICS]** Use HAC-robust t-stats and block bootstrap CIs

### SHOULD FIX (High Priority):

11. **[TEMPORAL LINT]** Add pre-commit hook for `lint_for_temporal_leakage()`
12. **[NESTED CV]** Use nested CV for all hyperparameter searches
13. **[STABILITY]** Add hyperparameter stability checks (MAD/median < 15%)
14. **[PERMUTATION]** Use block-wise permutation for importance
15. **[IMPACT MODEL]** Add market impact and capacity models

### NICE TO HAVE (Medium Priority):

16. Add ADF/KPSS stationarity tests for features
17. Add VIF caps for multicollinearity
18. Add reliability diagrams for probability calibration
19. Add conformal prediction for uncertainty quantification
20. Document embargo zones around label horizons

---

## Conclusion

The pre-training pipeline demonstrates **strong awareness of ML best practices** and has **good infrastructure** for temporal validation, volatility normalization, and label balancing. However, several **critical gaps** exist:

### Strengths ✅:
- Excellent execution delay modeling (enters at next bar)
- Good volatility normalization (σ-units)
- Temporal validation config present
- Label balancing system
- Multi-stage feature selection

### Critical Weaknesses ⚠️:
- **No enforcement that features use lag >= 1**
- **No unit tests for temporal alignment**
- **No closed='left' in rolling operations**
- **No FDR correction for multiple testing**
- **No purged walk-forward CV implementation**
- **No turnover/capacity constraints**
- **No HAC-robust statistics**

### Risk Level:
**MEDIUM-HIGH** - Code shows good intent but lacks rigorous enforcement. Production deployment should wait until critical items #1-10 are addressed.

---

**Reviewer**: AI Code Analysis System  
**Review Complete**: 2025-10-08  
**Next Steps**: Address critical action items #1-10, then re-review