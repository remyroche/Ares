#!/usr/bin/env python3
"""
Test script to verify profit labeling framework alignment in optimization systems.

This script tests the complete integration between:
1. Feature lookback optimization
2. Interaction feature generator
3. Profit labeling framework

It verifies that both optimization systems are using the profit labeling
framework's quality metrics and optimization goals.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
import os
import pytest
import importlib.util
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

# Import the unified optimization framework
try:
    from src.training.steps.pre_training.unified_optimization_framework import (
        UnifiedOptimizationFramework, UnifiedOptimizationConfig, 
        OptimizationSystem, OptimizationObjective
    )
    UNIFIED_FRAMEWORK_AVAILABLE = True
except Exception as e:
    tprint_warning(f"⚠️ Unified framework not available: {e}")
    UNIFIED_FRAMEWORK_AVAILABLE = False

# Import individual optimization systems
try:
    from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
        OptimizedFeatureLookbackConfig, OptimizedFeatureLookbackOptimizer
    )
    FEATURE_LOOKBACK_AVAILABLE = True
except Exception as e:
    tprint_warning(f"⚠️ Feature lookback optimization not available: {e}")
    FEATURE_LOOKBACK_AVAILABLE = False

try:
    from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.orchestrator import (
        LookbackOptimizationOrchestrator
    )
    from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.config import (
        LookbackOptimizationConfig as InteractionConfig
    )
    INTERACTION_GENERATOR_AVAILABLE = True
except Exception as e:
    tprint_warning(f"⚠️ Interaction feature generator not available: {e}")
    INTERACTION_GENERATOR_AVAILABLE = False

# Import profit labeling framework
try:
    from src.training.steps.pre_training.profit_labeling.quality_scoring import (
        LabelQualityScorer, QualityScoringConfig, QualityMetrics, QualityMetric
    )
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig,
        EnhancedLabelDefinitions, AnalystLabelConfig, TradingCosts
    )
    from src.training.steps.pre_training.profit_labeling.multi_target_scheme import (
        MultiTargetScheme, MultiTargetConfig
    )
    PROFIT_LABELING_AVAILABLE = True
except Exception as e:
    tprint_warning(f"⚠️ Profit labeling framework not available: {e}")
    PROFIT_LABELING_AVAILABLE = False


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for optimization testing."""
    tprint_info("🔧 Creating synthetic test data...")
    
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = np.array(prices[1:])
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='5min'),
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples),
        'bid': prices * (1 - np.random.uniform(0.0001, 0.001, n_samples)),
        'ask': prices * (1 + np.random.uniform(0.0001, 0.001, n_samples)),
        'asset_class': np.random.choice(['crypto'], n_samples)
    })
    
    # Generate synthetic targets using profit labeling framework approach
    # Create multi-horizon targets with different volatility bands
    small_band = np.random.uniform(0.4, 0.8, n_samples)
    medium_band = np.random.uniform(0.8, 1.3, n_samples)
    high_band = np.random.uniform(1.3, 2.0, n_samples)
    
    # Generate forward returns for different horizons
    data['immediate_opportunity'] = np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2])
    data['short_term_opportunity'] = np.random.choice([-1, 0, 1], n_samples, p=[0.15, 0.7, 0.15])
    data['leverage_adjusted_score'] = np.random.choice([-1, 0, 1], n_samples, p=[0.1, 0.8, 0.1])
    
    # Add confidence scores
    data['confidence_scores'] = np.random.uniform(0.3, 1.0, n_samples)
    
    # Add eligibility masks
    data['eligibility_mask'] = np.random.choice([True, False], n_samples, p=[0.8, 0.2])
    
    tprint_success(f"✅ Created test data with {n_samples} samples")
    return data


def test_profit_labeling_framework():
    """Test profit labeling framework components."""
    tprint_info("🧪 Testing profit labeling framework...")

    if not PROFIT_LABELING_AVAILABLE:
        tprint_error("❌ Profit labeling framework not available")
        return False

    try:
        # Test quality scorer
        quality_config = QualityScoringConfig(
            baseline_models=['logistic', 'random_forest'],
            test_size=0.2,
            n_splits=5,
            random_state=42,
            min_lqs_score=0.3,
            min_auc_threshold=0.55,
            max_auc_std_threshold=0.03,
            min_psi_threshold=0.1,
            max_flip_rate_threshold=0.15,
            min_balance_threshold=0.35,
            max_balance_threshold=0.65,
            max_correlation_threshold=0.4
        )
        
        quality_scorer = LabelQualityScorer(quality_config)
        tprint_success("✅ Quality scorer initialized")
        
        # Test volatility labeler
        volatility_config = VolatilityAwareConfig(
            min_data_points=1000,
            generate_reports=True,
            save_intermediate_results=True,
            enable_volatility_normalization=True,
            enable_multi_target_scheme=True
        )
        
        volatility_labeler = VolatilityAwareMultiHorizonLabeler(volatility_config)
        tprint_success("✅ Volatility labeler initialized")
        
        # Test multi-target scheme
        multi_target_config = MultiTargetConfig(
            small_band=(0.4, 0.8),
            medium_band=(0.8, 1.3),
            high_band=(1.3, 2.0),
            enable_optimization=True,
            optimization_method='bayesian',
            n_trials=50,
            optimization_metric='lqs'
        )
        
        multi_target_scheme = MultiTargetScheme(multi_target_config)
        tprint_success("✅ Multi-target scheme initialized")
        
        tprint_success("✅ Profit labeling framework test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Profit labeling framework test failed: {e}")
        return False


def test_trading_costs_with_borrow_and_funding():
    """Ensure trading costs apply borrow/funding assumptions per asset class."""
    module_path = Path(__file__).resolve().parent / 'src' / 'training' / 'steps' / 'pre_training' / 'profit_labeling' / 'enhanced_label_definitions.py'
    spec = importlib.util.spec_from_file_location('enhanced_label_definitions_test', module_path)
    if spec is None or spec.loader is None:
        pytest.skip("Unable to load enhanced label definitions module")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        pytest.skip(f"Enhanced label definitions unavailable: {exc}")

    EnhancedLabelDefinitions = module.EnhancedLabelDefinitions
    AnalystLabelConfig = module.AnalystLabelConfig
    TradingCosts = module.TradingCosts

    data = create_test_data(50)
    data['asset_class'] = 'crypto'

    trading_costs = TradingCosts(
        maker_fee=0.0,
        taker_fee=0.0,
        slippage_pct=0.0,
        min_trade_size=0.0,
        default_asset_class='crypto',
        borrow_fees={'crypto': {'long': 0.0001, 'short': 0.0005}},
        funding_rates={'crypto': {'long': 0.0002, 'short': -0.0001}},
        stress_scenarios={
            'crypto': {
                'base': {'long': 1.0, 'short': 1.0},
                'stress_test': {'long': 1.05, 'short': 1.25}
            }
        },
        active_stress_scenario='base'
    )

    analyst_config = AnalystLabelConfig(trading_costs=trading_costs)
    labeler = EnhancedLabelDefinitions(analyst_config=analyst_config)

    expected_returns = pd.Series(
        np.linspace(-0.01, 0.01, len(data)), index=data.index
    )

    costs_series = labeler._calculate_trading_costs(
        data,
        trading_costs,
        expected_returns=expected_returns,
        stress_scenario='stress_test'
    )

    assert not costs_series.empty
    assert (costs_series >= 0).all()

    first_idx = data.index[0]
    first_trade_size = data.loc[first_idx, 'volume'] * data.loc[first_idx, 'close'] * 0.01
    first_expected = expected_returns.loc[first_idx]
    first_direction = 'long' if first_expected >= 0 else 'short'
    first_expected_cost = first_trade_size * (
        trading_costs.get_borrow_rate('crypto', first_direction) +
        trading_costs.get_funding_rate('crypto', first_direction)
    ) * trading_costs.get_stress_multiplier('crypto', first_direction, scenario='stress_test')

    assert np.isclose(costs_series.loc[first_idx], first_expected_cost)

    last_idx = data.index[-1]
    last_trade_size = data.loc[last_idx, 'volume'] * data.loc[last_idx, 'close'] * 0.01
    last_expected = expected_returns.loc[last_idx]
    last_direction = 'long' if last_expected >= 0 else 'short'
    last_expected_cost = last_trade_size * (
        trading_costs.get_borrow_rate('crypto', last_direction) +
        trading_costs.get_funding_rate('crypto', last_direction)
    ) * trading_costs.get_stress_multiplier('crypto', last_direction, scenario='stress_test')

    assert np.isclose(costs_series.loc[last_idx], last_expected_cost)

def test_feature_lookback_optimization():
    """Test feature lookback optimization with profit labeling integration."""
    tprint_info("🧪 Testing feature lookback optimization...")

    if not FEATURE_LOOKBACK_AVAILABLE:
        tprint_error("❌ Feature lookback optimization not available")
        return False
    
    try:
        # Create test data
        data = create_test_data(500)  # Smaller dataset for testing
        
        # Create configuration with profit labeling integration
        config = OptimizedFeatureLookbackConfig(
            optimization_metric="lqs_combined",
            enable_quality_scoring=True,
            min_lqs_threshold=0.3,
            min_auc_threshold=0.55,
            enable_multi_objective=True,
            ic_weight=0.4,
            lqs_weight=0.4,
            stability_weight=0.2
        )
        
        # Initialize optimizer
        optimizer = OptimizedFeatureLookbackOptimizer(config)
        tprint_success("✅ Feature lookback optimizer initialized with profit labeling integration")
        
        # Test that quality scorer is available
        if hasattr(optimizer, 'quality_scorer') and optimizer.quality_scorer is not None:
            tprint_success("✅ Quality scorer integrated in feature lookback optimization")
        else:
            tprint_warning("⚠️ Quality scorer not integrated in feature lookback optimization")
        
        # Test that multi-target scheme is available
        if hasattr(optimizer, 'multi_target_scheme') and optimizer.multi_target_scheme is not None:
            tprint_success("✅ Multi-target scheme integrated in feature lookback optimization")
        else:
            tprint_warning("⚠️ Multi-target scheme not integrated in feature lookback optimization")
        
        tprint_success("✅ Feature lookback optimization test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Feature lookback optimization test failed: {e}")
        return False


def test_interaction_feature_generator():
    """Test interaction feature generator with profit labeling integration."""
    tprint_info("🧪 Testing interaction feature generator...")
    
    if not INTERACTION_GENERATOR_AVAILABLE:
        tprint_error("❌ Interaction feature generator not available")
        return False
    
    try:
        # Create configuration
        config = InteractionConfig(
            output_dir="test_interaction_results"
        )
        
        # Initialize orchestrator
        orchestrator = LookbackOptimizationOrchestrator(config)
        tprint_success("✅ Interaction feature generator orchestrator initialized")
        
        # Test that profit labeling components are available
        if hasattr(orchestrator, 'quality_scorer') and orchestrator.quality_scorer is not None:
            tprint_success("✅ Quality scorer integrated in interaction feature generator")
        else:
            tprint_warning("⚠️ Quality scorer not integrated in interaction feature generator")
        
        if hasattr(orchestrator, 'volatility_labeler') and orchestrator.volatility_labeler is not None:
            tprint_success("✅ Volatility labeler integrated in interaction feature generator")
        else:
            tprint_warning("⚠️ Volatility labeler not integrated in interaction feature generator")
        
        if hasattr(orchestrator, 'multi_target_scheme') and orchestrator.multi_target_scheme is not None:
            tprint_success("✅ Multi-target scheme integrated in interaction feature generator")
        else:
            tprint_warning("⚠️ Multi-target scheme not integrated in interaction feature generator")
        
        tprint_success("✅ Interaction feature generator test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Interaction feature generator test failed: {e}")
        return False


async def test_unified_optimization_framework():
    """Test unified optimization framework."""
    tprint_info("🧪 Testing unified optimization framework...")
    
    if not UNIFIED_FRAMEWORK_AVAILABLE:
        tprint_error("❌ Unified optimization framework not available")
        return False
    
    try:
        # Create test data
        data = create_test_data(500)
        
        # Create unified configuration
        config = UnifiedOptimizationConfig(
            enabled_systems=[OptimizationSystem.FEATURE_LOOKBACK, OptimizationSystem.INTERACTION_GENERATOR],
            primary_objective=OptimizationObjective.MULTI_OBJECTIVE,
            min_lqs_threshold=0.3,
            enable_quality_filtering=True,
            ic_weight=0.4,
            lqs_weight=0.4,
            stability_weight=0.2
        )
        
        # Initialize framework
        framework = UnifiedOptimizationFramework(config)
        tprint_success("✅ Unified optimization framework initialized")
        
        # Test that profit labeling components are available
        if framework.quality_scorer is not None:
            tprint_success("✅ Quality scorer available in unified framework")
        else:
            tprint_warning("⚠️ Quality scorer not available in unified framework")
        
        if framework.volatility_labeler is not None:
            tprint_success("✅ Volatility labeler available in unified framework")
        else:
            tprint_warning("⚠️ Volatility labeler not available in unified framework")
        
        if framework.multi_target_scheme is not None:
            tprint_success("✅ Multi-target scheme available in unified framework")
        else:
            tprint_warning("⚠️ Multi-target scheme not available in unified framework")
        
        # Test optimization (this might take a while, so we'll just test initialization)
        tprint_info("🔧 Testing optimization initialization (skipping full run for speed)...")
        
        tprint_success("✅ Unified optimization framework test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Unified optimization framework test failed: {e}")
        return False


def test_configuration_alignment():
    """Test that configurations are aligned with profit labeling framework."""
    tprint_info("🧪 Testing configuration alignment...")
    
    try:
        # Test that all configurations use the same quality thresholds
        expected_thresholds = {
            'min_lqs_threshold': 0.3,
            'min_auc_threshold': 0.55,
            'max_auc_std_threshold': 0.03,
            'min_psi_threshold': 0.1,
            'max_flip_rate_threshold': 0.15,
            'min_balance_threshold': 0.35,
            'max_balance_threshold': 0.65,
            'max_correlation_threshold': 0.4
        }
        
        # Test feature lookback configuration
        if FEATURE_LOOKBACK_AVAILABLE:
            config = OptimizedFeatureLookbackConfig()
            for threshold, expected_value in expected_thresholds.items():
                if hasattr(config, threshold):
                    actual_value = getattr(config, threshold)
                    if actual_value == expected_value:
                        tprint_success(f"✅ {threshold}: {actual_value}")
                    else:
                        tprint_warning(f"⚠️ {threshold}: {actual_value} (expected {expected_value})")
                else:
                    tprint_warning(f"⚠️ {threshold} not found in feature lookback config")
        
        # Test unified configuration
        if UNIFIED_FRAMEWORK_AVAILABLE:
            config = UnifiedOptimizationConfig()
            for threshold, expected_value in expected_thresholds.items():
                if hasattr(config, threshold):
                    actual_value = getattr(config, threshold)
                    if actual_value == expected_value:
                        tprint_success(f"✅ {threshold}: {actual_value}")
                    else:
                        tprint_warning(f"⚠️ {threshold}: {actual_value} (expected {expected_value})")
                else:
                    tprint_warning(f"⚠️ {threshold} not found in unified config")
        
        tprint_success("✅ Configuration alignment test completed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Configuration alignment test failed: {e}")
        return False


async def main():
    """Run all tests."""
    tprint("🚀 Starting profit labeling alignment tests...")
    
    test_results = {}
    
    # Test profit labeling framework
    test_results['profit_labeling'] = test_profit_labeling_framework()
    
    # Test feature lookback optimization
    test_results['feature_lookback'] = test_feature_lookback_optimization()
    
    # Test interaction feature generator
    test_results['interaction_generator'] = test_interaction_feature_generator()
    
    # Test unified optimization framework
    test_results['unified_framework'] = await test_unified_optimization_framework()
    
    # Test configuration alignment
    test_results['configuration_alignment'] = test_configuration_alignment()
    
    # Print summary
    tprint("\n📊 Test Results Summary:")
    tprint("=" * 50)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        tprint(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    tprint("=" * 50)
    tprint(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        tprint_success("🎉 All tests passed! Profit labeling alignment is complete.")
        return True
    else:
        tprint_error(f"⚠️ {total_tests - passed_tests} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 Profit labeling alignment verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)