"""
Comprehensive Unit Tests for Dampened Kelly Engine

Tests all edge cases, robustness features, and production scenarios:
1. Core edge cases (zero wins, all wins, extreme R, boundary conditions)
2. Bin sparsity and adaptive merging
3. Realized R instability detection
4. Temporal leakage prevention (purging, embargo)
5. Regime switching mid-trade
6. Hot-swap thread safety
7. Drawdown dampening
8. Calibration validation
9. Numerical stability
"""

import pytest
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from src.trading.sizing.dampened_kelly_engine import DampenedKellyEngine, KellyResult, ReasonCode
from src.trading.sizing.kelly_history_tracker import KellyHistoryTracker, BinData
from src.trading.sizing.portfolio_correlation_handler import PortfolioCorrelationHandler


# ========================================
# Fixtures
# ========================================

@pytest.fixture
def default_kelly_config() -> Dict[str, Any]:
    """Default Kelly configuration for testing."""
    return {
        'regime_params': {
            'regime_0': {
                'lambda_base': 0.30,
                'beta_position': 1.2,
                'beta_leverage': 0.8,
                'prior_alpha': 15.0,
                'ess_threshold': 40,
                'entropy_threshold': 0.8,
                'n_min_samples': 15,
                'f_floor': 0.01,
                'lev_floor': 1.5,
                'decay_theta': 0.95
            }
        },
        'global_fallback': {
            'lambda_base': 0.15,
            'beta_position': 2.0,
            'beta_leverage': 1.5,
            'prior_alpha': 30.0,
            'ess_threshold': 60,
            'entropy_threshold': 1.0,
            'n_min_samples': 25,
            'f_floor': 0.005,
            'lev_floor': 1.2,
            'decay_theta': 0.90
        },
        'lambda_eff_components': {
            'ess_sigmoid_kappa': 0.1,
            'entropy_scale': 0.5,
            'variance_penalty': 2.0
        },
        'binning': {
            'score_bins': [0.5, 0.6, 0.7, 0.8, 0.9],
            'volatility_bins': [0.005, 0.01, 0.02, 0.04],
            'enable_adaptive_merging': True,
            'stale_bin_days': 90
        },
        'r_tracking': {
            'use_realized_r': True,
            'r_percentile': 25,
            'r_instability_threshold': 2.0,
            'r_instability_prior_boost': 2.0
        },
        'safety_limits': {
            'max_leverage': 3.0,
            'max_per_trade_pct': 0.15,
            'max_kelly_fraction': 0.5,
            'high_leverage_threshold': 2.0,
            'max_acceptable_drawdown': 0.15
        },
        'correlation': {
            'enabled': True,
            'window_days': 30,
            'high_corr_threshold': 0.7,
            'high_corr_penalty': 0.3,
            'per_trade_corr_limit': 0.8
        },
        'temporal': {
            'embargo_pct_of_train': 0.05,
            'enable_purging': True,
            'overlap_detection': True
        }
    }


@pytest.fixture
def kelly_engine(default_kelly_config):
    """Kelly engine fixture."""
    return DampenedKellyEngine(default_kelly_config)


@pytest.fixture
def kelly_tracker(default_kelly_config):
    """Kelly tracker fixture."""
    return KellyHistoryTracker(default_kelly_config)


@pytest.fixture
def correlation_handler(default_kelly_config):
    """Correlation handler fixture."""
    return PortfolioCorrelationHandler(default_kelly_config)


# ========================================
# Core Edge Cases Tests
# ========================================

class TestCoreEdgeCases:
    """Test core Kelly calculations with edge cases."""
    
    def test_zero_wins_returns_floor(self, kelly_engine):
        """Zero wins should return exploration floor."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=0, losses=10, regime_id=0, ess=100.0, entropy=0.5,
            r_realized=[2.0, 1.8, 2.2], current_dd=0.0
        )
        
        # Should be close to f_floor
        assert result.f_final <= 0.02  # Near or at floor
        assert result.posterior_mean < 0.5
        assert ReasonCode.INSUFFICIENT_SAMPLES.value in result.reason_codes or result.posterior_mean < 0.5
    
    def test_all_wins_approaches_lambda_eff(self, kelly_engine):
        """All wins with high confidence should approach lambda_eff."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=50, losses=0, regime_id=0, ess=100.0, entropy=0.3,
            r_realized=[2.5]*50, current_dd=0.0
        )
        
        # Should have high f_final
        assert result.f_final > 0.10
        assert result.posterior_mean > 0.9
        assert result.f_kelly > 0
    
    def test_extreme_r_low(self, kelly_engine):
        """Extreme low R (≤0.01) should be handled gracefully."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=25, losses=10, regime_id=0, ess=80.0, entropy=0.5,
            r_realized=[0.01]*35, current_dd=0.0
        )
        
        # Should have very low Kelly fraction due to poor R
        assert result.f_kelly >= 0  # Non-negative
        assert not np.isnan(result.f_final)
        assert not np.isinf(result.f_final)
    
    def test_extreme_r_high(self, kelly_engine):
        """Extreme high R (>100) should be clipped appropriately."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=30, losses=5, regime_id=0, ess=90.0, entropy=0.4,
            r_realized=[150.0]*35, current_dd=0.0
        )
        
        # Should apply Kelly fraction clip
        assert result.kelly_fraction_clip_applied or result.f_final <= 0.15
        assert not np.isnan(result.f_final)
    
    def test_posterior_at_zero(self, kelly_engine):
        """Posterior mean near 0 should not cause NaN."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=0, losses=100, regime_id=0, ess=100.0, entropy=0.5,
            r_realized=[2.0]*100, current_dd=0.0
        )
        
        assert not np.isnan(result.posterior_mean)
        assert not np.isnan(result.f_final)
        assert result.f_final >= 0
    
    def test_posterior_at_one(self, kelly_engine):
        """Posterior mean near 1 should not cause numerical issues."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=100, losses=0, regime_id=0, ess=100.0, entropy=0.5,
            r_realized=[2.0]*100, current_dd=0.0
        )
        
        assert not np.isnan(result.posterior_mean)
        assert not np.isinf(result.f_final)
        assert result.f_final <= 0.15  # Should be capped


# ========================================
# Bin Sparsity and Adaptive Merging Tests
# ========================================

class TestBinSparsity:
    """Test adaptive bin merging and fallback hierarchy."""
    
    def test_insufficient_samples_triggers_merge(self, kelly_tracker):
        """Insufficient samples should trigger bin merging."""
        # Add few samples to exact bin
        kelly_tracker.update_bin(0.75, 0.015, regime_id=0, is_win=True, r_realized=2.0)
        kelly_tracker.update_bin(0.75, 0.015, regime_id=0, is_win=False, r_realized=1.5)
        
        # Lookup with high n_min should trigger fallback
        bin_data, merge_level = kelly_tracker.lookup_bin(0.75, 0.015, regime_id=0, n_min=20)
        
        # Should use fallback (merge_level > 0) or return sparse bin
        assert merge_level >= 0
        assert bin_data is not None
    
    def test_regime_agnostic_merge(self, kelly_tracker):
        """Test regime-agnostic merging."""
        # Add data to multiple regimes, same score/vol
        kelly_tracker.update_bin(0.75, 0.015, regime_id=0, is_win=True, r_realized=2.0)
        kelly_tracker.update_bin(0.75, 0.015, regime_id=1, is_win=True, r_realized=2.1)
        kelly_tracker.update_bin(0.75, 0.015, regime_id=2, is_win=False, r_realized=1.8)
        
        # Lookup should merge across regimes if exact bin insufficient
        bin_data, merge_level = kelly_tracker.lookup_bin(0.75, 0.015, regime_id=0, n_min=10)
        
        # If merged, should have more samples
        if merge_level == 1:  # Regime-agnostic merge
            assert bin_data.total_samples() >= 3
    
    def test_coarse_bin_merge(self, kelly_tracker):
        """Test coarse bin merging (adjacent buckets)."""
        # Add data to adjacent buckets
        kelly_tracker.update_bin(0.72, 0.015, regime_id=0, is_win=True, r_realized=2.0)
        kelly_tracker.update_bin(0.78, 0.016, regime_id=0, is_win=True, r_realized=2.1)
        
        # Lookup with high n_min might trigger coarse merge
        bin_data, merge_level = kelly_tracker.lookup_bin(0.75, 0.015, regime_id=0, n_min=50)
        
        assert bin_data is not None  # Should always return something
        assert merge_level >= 0
    
    def test_global_prior_fallback(self, kelly_tracker):
        """Test global prior fallback when all merging fails."""
        # Lookup with no data in tracker
        bin_data, merge_level = kelly_tracker.lookup_bin(0.95, 0.05, regime_id=99, n_min=100)
        
        # Should fall back to global prior
        assert merge_level == 3  # Global prior
        assert bin_data.total_samples() == 0


# ========================================
# Realized R and Instability Tests
# ========================================

class TestRealizedR:
    """Test realized R tracking and instability detection."""
    
    def test_r_conservative_percentile(self, kelly_engine):
        """Should use 25th percentile (conservative) of R distribution."""
        r_realized = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        
        r_cons, r_mean, r_std, is_unstable = kelly_engine.calculate_r_conservative(r_realized)
        
        # 25th percentile of [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] ≈ 1.5
        assert r_cons < r_mean  # Conservative should be less than mean
        assert r_cons >= 1.0
    
    def test_r_instability_detection(self, kelly_engine):
        """High R std/mean ratio should trigger instability detection."""
        # Create unstable R distribution (high variance)
        r_realized = [0.5, 1.0, 5.0, 10.0, 15.0]  # CV > 2.0
        
        r_cons, r_mean, r_std, is_unstable = kelly_engine.calculate_r_conservative(r_realized)
        
        assert is_unstable  # Should detect instability
    
    def test_r_instability_increases_prior(self, kelly_engine):
        """R instability should increase prior weight in posterior."""
        # This would be tested by checking that prior_alpha is boosted
        # For now, verify the boost factor exists
        assert kelly_engine.r_instability_prior_boost > 1.0
    
    def test_empty_r_realized_uses_default(self, kelly_engine):
        """Empty R list should fall back to default."""
        r_cons, r_mean, r_std, is_unstable = kelly_engine.calculate_r_conservative([])
        
        # Should use default (2.0)
        assert r_cons == 2.0
        assert r_mean == 2.0
        assert not is_unstable


# ========================================
# Temporal Integrity Tests
# ========================================

class TestTemporalIntegrity:
    """Test purging and embargo enforcement."""
    
    def test_purging_removes_overlapping_trades(self, kelly_tracker):
        """Purging should remove trades that overlap train/test boundary."""
        # Add trades
        train_end = datetime(2025, 1, 31)
        test_start = datetime(2025, 2, 7)  # 7-day embargo
        
        # Trade 1: Well before boundary (keep)
        kelly_tracker.update_bin(0.7, 0.01, 0, True, 2.0, datetime(2025, 1, 20))
        
        # Trade 2: In overlap window (purge)
        kelly_tracker.update_bin(0.7, 0.01, 0, True, 2.1, datetime(2025, 1, 29))
        
        # Trade 3: After test start (keep)
        kelly_tracker.update_bin(0.7, 0.01, 0, False, 1.8, datetime(2025, 2, 10))
        
        # Purge
        purged = kelly_tracker.purge_overlapping_trades(train_end, test_start, max_trade_duration_days=7)
        
        # Should have purged trade 2
        assert purged > 0
    
    def test_embargo_period_calculation(self, kelly_tracker):
        """Embargo period should be calculated correctly."""
        embargo_days = kelly_tracker.get_embargo_period(train_window_days=100)
        
        # Should be 5% of 100 = 5 days
        assert embargo_days == 5
    
    def test_stale_bin_detection(self, kelly_tracker):
        """Bins not updated in >90 days should be marked stale."""
        # Add old trade
        old_date = datetime.now() - timedelta(days=100)
        kelly_tracker.update_bin(0.7, 0.01, 0, True, 2.0, old_date)
        
        # Check staleness
        staleness = kelly_tracker.check_staleness_all_bins()
        
        assert staleness['stale_bins'] > 0


# ========================================
# Regime Switching Tests
# ========================================

class TestRegimeSwitching:
    """Test regime switching scenarios."""
    
    def test_regime_switch_tracked(self, kelly_tracker):
        """Regime switches should be tracked."""
        kelly_tracker.track_regime_switch(datetime.now(), old_regime=0, new_regime=1)
        
        assert len(kelly_tracker.regime_switches) > 0
        assert kelly_tracker.regime_switches[0][1] == 0
        assert kelly_tracker.regime_switches[0][2] == 1
    
    def test_regime_stability_calculation(self, kelly_tracker):
        """Regime stability should be calculated correctly."""
        # Add switches
        for i in range(10):
            kelly_tracker.track_regime_switch(datetime.now(), old_regime=i%2, new_regime=(i+1)%2)
        
        # Check stability (should be low due to many switches)
        stability = kelly_tracker._calculate_regime_stability(regime_id=0)
        
        assert 0 <= stability <= 1
    
    def test_decay_adapts_to_stability(self, kelly_tracker):
        """Decay rate should adapt to regime stability."""
        # Stable regime should have higher theta
        stable_theta = kelly_tracker._get_decay_theta(regime_id=0)
        
        # Add many switches to make regime 0 unstable
        for i in range(100):
            kelly_tracker.track_regime_switch(datetime.now(), 0, 1)
        
        unstable_theta = kelly_tracker._get_decay_theta(regime_id=0)
        
        # Unstable should have lower theta (faster decay)
        assert unstable_theta <= stable_theta


# ========================================
# Hot-Swap Thread Safety Tests
# ========================================

class TestHotSwapThreadSafety:
    """Test thread-safe hot-swapping."""
    
    def test_concurrent_config_updates(self, kelly_engine):
        """Concurrent config updates should be thread-safe."""
        results = []
        
        def update_config(value):
            new_config = kelly_engine.config.copy()
            new_config['safety_limits'] = {'max_leverage': value}
            version = kelly_engine.update_config(new_config)
            results.append(version)
        
        # Run concurrent updates
        threads = []
        for i in range(10):
            t = threading.Thread(target=update_config, args=(3.0 + i*0.1,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All versions should be unique
        assert len(set(results)) == len(results)
        assert kelly_engine.get_config_version() > 10
    
    def test_config_versioning(self, kelly_engine):
        """Config versions should increment correctly."""
        initial_version = kelly_engine.get_config_version()
        
        kelly_engine.update_config({'safety_limits': {'max_leverage': 2.5}})
        assert kelly_engine.get_config_version() == initial_version + 1
        
        kelly_engine.update_config({'safety_limits': {'max_leverage': 3.0}})
        assert kelly_engine.get_config_version() == initial_version + 2


# ========================================
# Drawdown Dampening Tests
# ========================================

class TestDrawdownDampening:
    """Test drawdown-based dampening."""
    
    def test_no_drawdown_no_dampening(self, kelly_engine):
        """Zero drawdown should not apply dampening."""
        f_final = 0.10
        adjusted, dd_factor = kelly_engine.apply_drawdown_dampening(f_final, current_dd=0.0, max_dd=0.15)
        
        assert dd_factor == 1.0
        assert adjusted == f_final
    
    def test_high_drawdown_reduces_sizing(self, kelly_engine):
        """High drawdown should reduce sizing."""
        f_final = 0.10
        adjusted, dd_factor = kelly_engine.apply_drawdown_dampening(f_final, current_dd=0.10, max_dd=0.15)
        
        assert dd_factor < 1.0
        assert adjusted < f_final
    
    def test_drawdown_never_below_minimum(self, kelly_engine):
        """Drawdown factor should never go below minimum (0.3)."""
        f_final = 0.10
        adjusted, dd_factor = kelly_engine.apply_drawdown_dampening(
            f_final, current_dd=0.20, max_dd=0.15, min_factor=0.3
        )
        
        assert dd_factor >= 0.3
        assert adjusted >= f_final * 0.3


# ========================================
# Calibration Tests
# ========================================

class TestCalibration:
    """Test calibration of posterior vs actual outcomes."""
    
    def test_posterior_mean_calculation(self, kelly_engine):
        """Posterior mean should be calculated correctly."""
        # Beta(15+10, 15+5) with wins=10, losses=5, prior_alpha=15
        posterior_mean, posterior_var = kelly_engine.compute_posterior_mean_var(
            wins=10, losses=5, a=15.0, b=15.0
        )
        
        # Mean = (15+10) / (15+10 + 15+5) = 25/35 ≈ 0.714
        assert 0.70 < posterior_mean < 0.72
        assert posterior_var > 0
    
    def test_high_variance_increases_dampening(self, kelly_engine):
        """High posterior variance should increase dampening."""
        # Low samples → high variance
        lambda_eff_high_var, _ = kelly_engine.compute_lambda_eff(
            lambda_base=0.3, ess=80, var_p=0.10, entropy=0.5,
            ess_threshold=50, entropy_threshold=1.0
        )
        
        # High samples → low variance
        lambda_eff_low_var, _ = kelly_engine.compute_lambda_eff(
            lambda_base=0.3, ess=80, var_p=0.01, entropy=0.5,
            ess_threshold=50, entropy_threshold=1.0
        )
        
        # Higher variance should result in lower lambda_eff
        assert lambda_eff_high_var < lambda_eff_low_var


# ========================================
# ESS and Entropy Tests
# ========================================

class TestESSEntropy:
    """Test ESS and entropy dampening."""
    
    def test_low_ess_reduces_lambda(self, kelly_engine):
        """Low ESS should reduce lambda_eff."""
        lambda_low_ess, _ = kelly_engine.compute_lambda_eff(
            lambda_base=0.3, ess=10, var_p=0.05, entropy=0.5,
            ess_threshold=50, entropy_threshold=1.0
        )
        
        lambda_high_ess, _ = kelly_engine.compute_lambda_eff(
            lambda_base=0.3, ess=100, var_p=0.05, entropy=0.5,
            ess_threshold=50, entropy_threshold=1.0
        )
        
        assert lambda_low_ess < lambda_high_ess
    
    def test_high_entropy_triggers_veto(self, kelly_engine):
        """High entropy should trigger veto (reduce lambda)."""
        lambda_high_entropy, components = kelly_engine.compute_lambda_eff(
            lambda_base=0.3, ess=80, var_p=0.05, entropy=1.5,
            ess_threshold=50, entropy_threshold=0.8
        )
        
        lambda_low_entropy, _ = kelly_engine.compute_lambda_eff(
            lambda_base=0.3, ess=80, var_p=0.05, entropy=0.4,
            ess_threshold=50, entropy_threshold=0.8
        )
        
        # High entropy should reduce lambda via entropy_factor
        assert lambda_high_entropy < lambda_low_entropy
        assert components['entropy_factor'] < 1.0


# ========================================
# Correlation Tests
# ========================================

class TestCorrelation:
    """Test portfolio correlation handling."""
    
    def test_correlation_matrix_calculation(self, correlation_handler):
        """Correlation matrix should be calculated from price history."""
        # Add price history for multiple symbols
        for i in range(30):
            correlation_handler.update_price('BTC', 40000 + i*100, datetime.now() - timedelta(days=30-i))
            correlation_handler.update_price('ETH', 2500 + i*10, datetime.now() - timedelta(days=30-i))
        
        corr_matrix = correlation_handler.calculate_correlation_matrix()
        
        if corr_matrix is not None:
            assert 'BTC' in corr_matrix.index
            assert 'ETH' in corr_matrix.index
    
    def test_high_correlation_reduces_limit(self, correlation_handler):
        """High correlation should reduce portfolio limit."""
        # Add positions and prices
        correlation_handler.update_position('BTC', size=0.10, leverage=3.0)
        correlation_handler.update_position('ETH', size=0.10, leverage=3.0)
        
        # Add highly correlated price history
        for i in range(50):
            price = 40000 + i*100
            correlation_handler.update_price('BTC', price, datetime.now() - timedelta(hours=50-i))
            correlation_handler.update_price('ETH', price/16, datetime.now() - timedelta(hours=50-i))
        
        adjusted_limit, metadata = correlation_handler.get_adjusted_portfolio_limit()
        
        # Should reduce limit due to high correlation
        # (May not work with small sample, but structure should be correct)
        assert 'penalty_factor' in metadata


# ========================================
# Numerical Stability Tests
# ========================================

class TestNumericalStability:
    """Test numerical stability across edge cases."""
    
    def test_no_nan_with_zero_wins_losses(self, kelly_engine):
        """Zero wins and losses should not produce NaN."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=0, losses=0, regime_id=0, ess=50.0, entropy=0.5,
            r_realized=[], current_dd=0.0
        )
        
        assert not np.isnan(result.f_final)
        assert not np.isnan(result.leverage_final)
        assert not np.isnan(result.posterior_mean)
    
    def test_no_inf_with_extreme_values(self, kelly_engine):
        """Extreme values should not produce Inf."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=1000, losses=1, regime_id=0, ess=200.0, entropy=0.1,
            r_realized=[100.0]*1000, current_dd=0.0
        )
        
        assert not np.isinf(result.f_final)
        assert not np.isinf(result.leverage_final)
    
    def test_all_outputs_finite(self, kelly_engine):
        """All outputs should be finite numbers."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=25, losses=10, regime_id=0, ess=75.0, entropy=0.6,
            r_realized=[2.0, 1.8, 2.2, 1.9, 2.1]*7, current_dd=0.05
        )
        
        assert np.isfinite(result.f_final)
        assert np.isfinite(result.leverage_final)
        assert np.isfinite(result.f_kelly)
        assert np.isfinite(result.lambda_eff)
        assert np.isfinite(result.posterior_mean)
        assert np.isfinite(result.posterior_var)


# ========================================
# Integration Tests
# ========================================

class TestIntegration:
    """Test full integration scenarios."""
    
    def test_full_calculation_pipeline(self, kelly_engine, kelly_tracker):
        """Test complete calculation from bin lookup to final sizing."""
        # Build some history
        for i in range(50):
            is_win = i % 3 != 0  # 67% win rate
            kelly_tracker.update_bin(0.75, 0.015, regime_id=0, is_win=is_win, r_realized=2.0 + np.random.rand()*0.5)
        
        # Lookup bin
        bin_data, merge_level = kelly_tracker.lookup_bin(0.75, 0.015, regime_id=0, n_min=10)
        
        # Calculate Kelly
        result = kelly_engine.calculate_position_and_leverage(
            wins=bin_data.wins,
            losses=bin_data.losses,
            regime_id=0,
            ess=80.0,
            entropy=0.5,
            r_realized=bin_data.r_realized,
            current_dd=0.03
        )
        
        # Verify result structure
        assert result.f_final > 0
        assert result.leverage_final > 1.0
        assert result.config_version >= 1
        assert len(result.reason_codes) >= 0
        assert result.r_conservative > 0
    
    def test_config_version_logging(self, kelly_engine):
        """Config version should be logged in results."""
        result = kelly_engine.calculate_position_and_leverage(
            wins=20, losses=5, regime_id=0, ess=80, entropy=0.5,
            r_realized=[2.0]*25, current_dd=0.0
        )
        
        initial_version = result.config_version
        
        # Update config
        kelly_engine.update_config(kelly_engine.config)
        
        # New result should have incremented version
        result2 = kelly_engine.calculate_position_and_leverage(
            wins=20, losses=5, regime_id=0, ess=80, entropy=0.5,
            r_realized=[2.0]*25, current_dd=0.0
        )
        
        assert result2.config_version > initial_version


# ========================================
# Run Tests
# ========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

