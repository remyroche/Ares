# Dampened Kelly Sizing - Implementation Status

## Overview
Production-hardened dampened, posterior-aware Kelly sizing system for unified position sizing AND leverage calculation with regime conditioning, adaptive binning, and comprehensive risk management.

**Implementation Date**: October 30, 2025  
**Status**: Phase 1 Complete, Phases 2-4 Pending Integration

---

## ✅ COMPLETED: Phase 1 - Core Infrastructure (100%)

### 1.1 Dampened Kelly Engine ✓
**File**: `src/trading/sizing/dampened_kelly_engine.py`

**Implemented Features**:
- ✅ Unified position sizing AND leverage calculation (same algorithm)
- ✅ Bayesian posterior estimation (Beta distribution)
- ✅ Modular lambda_eff computation (ESS, entropy, variance components)
- ✅ Kelly fraction clipping (max_kelly_fraction parameter)
- ✅ Drawdown dampening (scales with current DD)
- ✅ Exploration floors (f_floor, lev_floor)
- ✅ Thread-safe versioned config for hot-swapping
- ✅ Comprehensive metadata and reason codes
- ✅ Regime-aware parameters with global fallback

**Key Classes**:
- `DampenedKellyEngine`: Main engine with all calculations
- `KellyResult`: Complete result with metadata
- `KellyConfigVersion`: Versioned config for hot-swapping
- `ReasonCode`: Enum for sizing decision reasons

**Key Methods**:
- `calculate_position_and_leverage()`: Main entry point
- `compute_posterior_mean_var()`: Beta posterior
- `compute_f_kelly()`, `compute_leverage_kelly()`: Kelly formulas
- `compute_lambda_eff()`: Modular dampening
- `compute_f_final()`, `compute_leverage_final()`: Dampened outputs
- `apply_drawdown_dampening()`: DD-based scaling
- `calculate_r_conservative()`: Realized R distribution

### 1.2 Kelly History Tracker ✓
**File**: `src/trading/sizing/kelly_history_tracker.py`

**Implemented Features**:
- ✅ 3D binning: (model_score, volatility, regime)
- ✅ Adaptive bin merging with 3-level hierarchical fallback:
  1. Exact bin
  2. Regime-agnostic merge
  3. Coarse bins (adjacent buckets)
  4. Global prior
- ✅ Realized R tracking per bin (percentile-based)
- ✅ R instability detection (std/mean ratio)
- ✅ Temporal integrity: purging + embargo periods
- ✅ Regime-adaptive decay rates (based on stability)
- ✅ Stale bin detection (>90 days)
- ✅ Bin coverage statistics
- ✅ Artifact persistence (save/load)

**Key Classes**:
- `KellyHistoryTracker`: Main tracker
- `BinData`: Individual bin with win/loss/R_realized

**Key Methods**:
- `update_bin()`: Add trade outcome
- `lookup_bin()`: Adaptive bin lookup with fallback
- `purge_overlapping_trades()`: Temporal leakage prevention
- `get_embargo_period()`: Calculate embargo
- `check_staleness_all_bins()`: Stale detection
- `save_to_file()`, `load_from_file()`: Persistence

### 1.3 Portfolio Correlation Handler ✓
**File**: `src/trading/sizing/portfolio_correlation_handler.py`

**Implemented Features**:
- ✅ Rolling correlation matrix (30-day window)
- ✅ Portfolio-level high-leverage limit adjustment
  - High correlation (>0.7): -30% penalty
  - Moderate correlation (0.4-0.7): -15% penalty
- ✅ Per-trade correlation checks vs existing high-lev positions
- ✅ Correlation-adjusted position sizing
- ✅ Price history tracking for correlation calculation
- ✅ Position tracking (size, leverage, entry_time)

**Key Methods**:
- `calculate_correlation_matrix()`: Rolling returns correlation
- `get_adjusted_portfolio_limit()`: Correlation-adjusted limit
- `check_new_position_correlation()`: Per-trade check
- `calculate_correlation_adjusted_size()`: Reduce size if high corr
- `get_portfolio_stats()`: Current portfolio state

### 1.4 Hot-Swap Configuration ✓
**File**: `src/launcher/trading_launcher.py` (extended)

**Implemented Features**:
- ✅ Versioned config management (atomic updates)
- ✅ Config version history tracking
- ✅ Thread-safe parameter updates
- ✅ Per-trade config version logging

**New Hot-Swap Methods**:
- `hot_swap_max_leverage(leverage)`: Max leverage limit
- `hot_swap_max_per_trade_pct(pct)`: Max position size per trade
- `hot_swap_max_exposure_per_asset(pct)`: Max exposure per asset
- `hot_swap_max_kelly_fraction(fraction)`: Max Kelly fraction
- `hot_swap_max_acceptable_drawdown(dd)`: Max acceptable drawdown
- `get_kelly_config_version_history()`: View update history

**Usage Examples**:
```python
# From trade_launcher CLI
python trade_launcher.py --hot-swap max_leverage=2.5
python trade_launcher.py --hot-swap max_per_trade_pct=0.10

# From code
from src.launcher.trading_launcher import get_parameter_manager
manager = get_parameter_manager()
result = manager.hot_swap_max_leverage(2.5)
# Returns: {'success': True, 'config_version': 2, 'old_value': 3.0, 'new_value': 2.5, ...}
```

### 1.5 Configuration File ✓
**File**: `src/config/kelly_sizing_config.yaml`

**Comprehensive configuration including**:
- ✅ Regime-aware parameters (per-regime tuning)
- ✅ Global fallback for unknown regimes
- ✅ Lambda_eff modular components
- ✅ Binning configuration (score/vol bin edges)
- ✅ Realized R tracking settings
- ✅ Temporal integrity settings (embargo, purging)
- ✅ Hot-swappable safety limits
- ✅ Correlation handling settings
- ✅ Calibration tracking settings
- ✅ Monte Carlo sampling (optional)
- ✅ Persistence & artifact settings
- ✅ Logging & monitoring settings
- ✅ Feature flags for A/B testing

---

## 🚧 PENDING: Phase 2 - Integration (0%)

### 2.1 Position Sizer Replacement
**File**: `src/trading/sizing/position_sizer.py` (needs replacement)

**Required Changes**:
1. Import dampened Kelly engine and tracker
2. Initialize engine and tracker in `__init__`
3. Replace `_calculate_kelly_position_size()` with:
   - Extract inputs: score, volatility, regime_id, ESS, entropy
   - Lookup bin with adaptive fallback
   - Get realized R (25th percentile)
   - Call `engine.calculate_position_and_leverage()`
   - Apply correlation adjustment
   - Return enhanced `PositionSizeResult`
4. Add input validation (price > 0, balance > 0)
5. Integrate with correlation handler
6. Log config version with each trade

**Integration Points**:
- Load Kelly config from `kelly_sizing_config.yaml`
- Initialize `DampenedKellyEngine(config['dampened_kelly'])`
- Initialize `KellyHistoryTracker(config['dampened_kelly'])`
- Initialize `PortfolioCorrelationHandler(config['dampened_kelly'])`
- Load bins from artifact if available
- Extract volatility from market data (ATR)
- Get regime_id from `StandardizedRegimeExtractor`
- Get ESS/entropy from ensemble predictions

### 2.2 Leverage Manager Replacement
**File**: `src/trading/sizing/leverage_manager.py` (needs replacement)

**Required Changes**:
1. Share same dampened Kelly engine instance with position sizer
2. Replace `calculate_leverage()` with:
   - Use same bin lookup as position sizer
   - Get `leverage_final` from Kelly result
   - Apply portfolio correlation checks
   - Return enhanced `LeverageResult`
3. Unified logic (same algorithm as position sizing)

**Key**: Should reuse the `KellyResult` from position sizer to avoid duplicate calculations

---

## 🚧 PENDING: Phase 3 - Backtesting Integration (0%)

### 3.1 Paper Trading Engine Integration
**File**: `src/training/steps/backtesting/abc_testing/paper_trading_engine.py`

**Required Changes**:
1. Initialize `KellyHistoryTracker` at backtest start
2. For each trade decision:
   - Call dampened Kelly position sizer
   - Log metadata (config_version, regime_id, bin_merge_level, all adjustments)
3. After trade closes:
   - Calculate realized R: `(exit_pnl / entry_risk)`
   - Update bin: `tracker.update_bin(score, vol, regime, is_win, R_realized, timestamp)`
   - Apply purging if needed
4. At backtest end:
   - Save bins to artifact: `kelly_bins_{symbol}_{timeframe}.pkl`
   - Generate calibration report: posterior_mean vs actual_win_rate per bin
   - Save metadata: reason code distribution, bin coverage

### 3.2 Walk-Forward Validation
**File**: `src/training/steps/backtesting/walk_forward_kelly_validation.py` (new)

**Required Implementation**:
1. Rolling windows with embargo: 24m train | 5% embargo | 6m test
2. Purge overlapping trades between folds
3. Six variants to compare:
   - Baseline: Current simple Kelly
   - Dampened Kelly (no ESS/entropy)
   - + ESS scaling
   - + Entropy veto
   - + Adaptive bins + realized R
   - Full system
4. Enhanced metrics per fold:
   - Performance: Sharpe, geometric return, max DD, Sortino
   - Calibration: |actual_win_rate - posterior_mean| per bin
   - Regime stability: % trades with mid-trade regime switches
   - Bin coverage: % trades in bins with ≥ n_min samples
   - Parameter sensitivity: metrics with ±20% parameter perturbation
5. Generate comprehensive report: `kelly_validation_report_{date}.json`

### 3.3 Safety Gate Validation
**Criteria** (must pass all):
- Full system: +10% geometric mean OR -20% max DD (≥90% baseline growth)
- Worst fold DD < 15%
- Calibration: Mean |actual - predicted| < 10% across bins with ≥20 samples
- Bin coverage: ≥70% trades with sufficient samples or successful fallback
- Regime stability: <10% mid-trade regime switches
- High-leverage trades: win rate > 50%, tail loss < 5%
- No NaN/Inf in any fold

---

## 🚧 PENDING: Phase 4 - Optimization Extensions (0%)

### 4.1 Nested Optimization
**File**: `src/training/steps/backtesting/final_parameters_optimization.py`

**Required Extension**:
1. Add new parameter category: `kelly_sizing_params`
2. Implement nested optimization:
   - Global optimization (150 trials): Optimize global fallback
   - Per-regime refinement (50 trials each): Starting from global, tune per-regime
   - Hierarchical regularization: `loss += L2_penalty * ||params_regime - params_global||^2`
3. Parameters to optimize (per-regime):
   - lambda_base, beta_position, beta_leverage
   - prior_alpha, ess_threshold, entropy_threshold
   - n_min, f_floor, lev_floor, decay_theta
4. Global parameters:
   - ess_sigmoid_kappa, entropy_scale, variance_penalty
   - max_kelly_fraction
5. Six objectives (multi-objective):
   - geometric_mean (maximize)
   - sharpe_ratio (maximize)
   - max_drawdown (minimize)
   - high_leverage_frequency (minimize)
   - calibration_error (minimize)
   - bin_coverage (maximize)
6. Meta-learning: Share parameters across similar regimes

### 4.2 Optimization Configuration
**File**: `src/config/step17_enhanced_optimization_config.yaml`

**Required Addition**:
```yaml
kelly_sizing_optimization:
  enabled: true
  regime_aware: true
  nested_optimization:
    enabled: true
    global_trials: 150
    per_regime_trials: 50
  hierarchical_regularization:
    l2_penalty: 0.1
    min_regime_samples: 50
    enable_meta_learning: true
  objectives:
    - {name: geometric_mean, weight: 1.0, maximize: true}
    - {name: sharpe_ratio, weight: 0.8, maximize: true}
    - {name: max_drawdown, weight: 1.2, maximize: false}
    - {name: high_leverage_frequency, weight: 0.5, maximize: false}
    - {name: calibration_error, weight: 0.7, maximize: false}
    - {name: bin_coverage, weight: 0.6, maximize: true}
  cv_config:
    method: "purged_walk_forward"
    n_folds: 5
    train_window_months: 24
    test_window_months: 6
    embargo_pct: 0.05
  optimization:
    n_trials: 150
    timeout_hours: 16
    parallel_jobs: 4
  parameter_sensitivity:
    enabled: true
    perturbation_pct: 0.20
```

### 4.3 Pareto Frontier Generation
**Output**: `kelly_params_optimized_{symbol}_{timeframe}.json`

**Required**:
1. Generate three configurations from Pareto frontier:
   - Conservative: Lower leverage, higher dampening, stricter thresholds
   - Balanced: Middle ground
   - Aggressive: Higher leverage when conditions warrant
2. For each config, include:
   - All parameters (per-regime + global)
   - Robustness metrics:
     - Avg leverage, %time high-lev, high-lev win rate
     - Calibration error, bin coverage
     - Parameter sensitivity: max metric degradation with ±20% perturbation
     - Regime stability: % mid-trade switches
     - Tail stats: 95th/99th percentile loss
   - Deployment recommendations

---

## 📊 Testing Requirements (Pending)

### Unit Tests
**File**: `tests/trading/sizing/test_dampened_kelly_engine.py` (to create)

**Coverage needed**:
1. Core edge cases:
   - Zero wins → f_final ≈ f_floor
   - All wins → approaches lambda_eff with Kelly clip
   - Low ESS → dampening applied
   - High entropy → veto triggered
   - Extreme R values (R ≤ 0.01, R > 100)
   - Posterior boundaries (p ≈ 0, p ≈ 1)

2. Bin sparsity/merging:
   - Insufficient samples → bin merging triggered
   - Hierarchical fallback: regime → regime-agnostic → coarse → prior
   - merge_level metadata correct

3. Realized R instability:
   - High std → prior weight increased
   - Empty R_realized[] → falls back to default

4. Temporal leakage:
   - Purging removes overlapping trades
   - Embargo period enforced

5. Regime switching:
   - Mid-trade regime switch → logged correctly
   - Parameters update when regime changes

6. Hot-swap thread safety:
   - Concurrent hot-swaps don't race
   - Config versioning works correctly

7. Drawdown dampening:
   - DD factor applied correctly
   - Never below minimum (0.3)

8. Calibration:
   - actual_win_rate ≈ posterior_mean ± 2σ

---

## 🔧 Integration Checklist

### Before Live Trading:
- [ ] Complete Phase 2: Position sizer & leverage manager integration
- [ ] Complete Phase 3: Backtesting integration & walk-forward validation
- [ ] Pass all safety gates
- [ ] Complete Phase 4: Parameter optimization with Pareto configs
- [ ] Run comprehensive unit tests (all passing)
- [ ] Run stress tests (all scenarios within limits)
- [ ] Generate calibration reports (error < 10%)
- [ ] Shadow mode testing (2-4 weeks, distribution matches backtest)
- [ ] Canary deployment stages (1% → 5% → 20% → full)

### Configuration Setup:
1. Load Kelly config:
   ```python
   import yaml
   with open('src/config/kelly_sizing_config.yaml') as f:
       kelly_config = yaml.safe_load(f)['dampened_kelly']
   ```

2. Initialize components:
   ```python
   from src.trading.sizing.dampened_kelly_engine import DampenedKellyEngine
   from src.trading.sizing.kelly_history_tracker import KellyHistoryTracker
   from src.trading.sizing.portfolio_correlation_handler import PortfolioCorrelationHandler
   
   engine = DampenedKellyEngine(kelly_config)
   tracker = KellyHistoryTracker(kelly_config)
   corr_handler = PortfolioCorrelationHandler(kelly_config)
   
   # Load bins if available
   from pathlib import Path
   bin_file = Path(f"checkpoints/kelly_sizing/kelly_bins_{symbol}_{timeframe}.pkl")
   if bin_file.exists():
       tracker = KellyHistoryTracker.load_from_file(bin_file)
   ```

3. Set up hot-swap manager:
   ```python
   from src.launcher.trading_launcher import get_parameter_manager
   
   manager = get_parameter_manager()
   manager.set_dampened_kelly_engine(engine)
   ```

### Runtime Usage:
```python
# During trade decision
kelly_result = engine.calculate_position_and_leverage(
    wins=bin_data.wins,
    losses=bin_data.losses,
    regime_id=current_regime,
    ess=ensemble_ess,
    entropy=ensemble_entropy,
    r_realized=bin_data.r_realized,
    current_dd=portfolio_drawdown,
    bin_merge_level=bin_data.merge_level,
    bin_last_updated=bin_data.last_updated,
    is_bin_stale=bin_data.is_stale
)

# Use results
position_size = kelly_result.f_final
leverage = kelly_result.leverage_final
config_version = kelly_result.config_version  # Log this with trade

# After trade closes
tracker.update_bin(
    score=model_score,
    volatility=atr_normalized,
    regime_id=regime_at_entry,
    is_win=(pnl > 0),
    r_realized=abs(pnl) / risk_taken,
    timestamp=exit_time
)

# Periodically save bins
tracker.save_to_file(bin_file)
```

---

## 📈 Expected Performance Improvements

### Backtesting Validation Targets:
- **Geometric Mean Return**: +15-30% vs baseline
- **Max Drawdown**: ≤ baseline OR -20% reduction
- **Sharpe Ratio**: ≥ baseline
- **Calibration Error**: < 10%
- **Bin Coverage**: ≥ 70%

### Risk Management:
- 95% of position sizes < max_per_trade_pct
- 99% of leverage < max_leverage
- Zero NaN/Inf calculations
- Cooldown prevents leverage clustering
- Correlation adjustment reduces portfolio blow-up risk
- Drawdown dampening maintains operation during adversity

---

## 📝 Notes

### Critical Implementation Details:
1. **Temporal Leakage Prevention**: Always use purged walk-forward CV with embargo periods
2. **Bin Sparsity**: Adaptive merging prevents failures when data is sparse
3. **R Instability**: System automatically increases prior weight when R is unstable
4. **Regime Stability**: Decay rates adapt to regime turnover
5. **Thread Safety**: All hot-swap operations are atomic with version tracking
6. **Calibration**: Must be monitored continuously; miscalibration indicates model drift

### Production Considerations:
1. **Bin Updates**: Enable incremental updates after 100 live trades
2. **Stale Bins**: Alert if bins not updated in >90 days
3. **Config Versioning**: Every trade logs config version for auditability
4. **Correlation Tracking**: Update price history continuously for correlation matrix
5. **Parameter Sensitivity**: Test ±20% perturbations before deploying new params

### Future Enhancements:
1. Multi-asset correlation matrix (beyond pairwise)
2. Regime prediction lookahead (predict regime switches)
3. Adaptive exploration floors (reduce as confidence grows)
4. Cross-symbol meta-learning (share parameters across similar assets)
5. Online learning (update parameters incrementally during live trading)

---

## 🎯 Next Steps

**Immediate (Phase 2)**:
1. Integrate dampened Kelly into position_sizer.py
2. Integrate dampened Kelly into leverage_manager.py
3. Test integration with existing trading pipeline

**Short-term (Phase 3)**:
1. Integrate tracker into backtesting engine
2. Implement walk-forward validation script
3. Run validation and verify safety gates pass

**Medium-term (Phase 4)**:
1. Extend step17 optimizer with nested Kelly optimization
2. Run optimization to generate Pareto configs
3. Select deployment configuration (conservative/balanced/aggressive)

**Pre-Live**:
1. Comprehensive unit test suite
2. Stress testing all scenarios
3. Shadow mode validation (2-4 weeks)
4. Staged canary deployment

---

## 📞 Support & Documentation

- **Implementation Plan**: `/dampened-kelly-sizing.plan.md`
- **Configuration**: `src/config/kelly_sizing_config.yaml`
- **Code Review**: Ensure all error handling, logging, and validation in place
- **Performance Monitoring**: Track calibration, bin coverage, reason codes

**Questions/Issues**: Refer to plan document for detailed specifications of each component.

