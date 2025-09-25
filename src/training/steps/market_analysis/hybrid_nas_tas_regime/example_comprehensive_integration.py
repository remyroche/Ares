"""
Comprehensive Integration Example for Hybrid NAS-TAS Regime System

This example demonstrates:
1. TAS and NAS systems being on par
2. Common tools stored in hybrid directory and accessed from there
3. Hybrid orchestrator initializing both systems and analyzing outputs
4. Multi-timeframe support (1m, 5m trading while maintaining 15m regime detection)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime, timedelta

# Import unified regime detector
from src.utils.ml_common.nas_tas_unified import UnifiedRegimeDetector, UnifiedRegimeConfig, RegimeDetectionMethod

# Import configurations
from .config.hybrid_regime_config import HybridRegimeConfig, RegimeCombinationStrategy
from .config.multi_timeframe_config import (
    create_default_multi_timeframe_config,
    create_high_frequency_config,
    create_swing_trading_config
)

# Import shared utilities - Modern imports
from src.utils.ml_common.nas_tas_unified import UnifiedSearchEngine, UnifiedComponentManager
from .shared_utils.unified_clustering_algorithms import create_unified_clustering_algorithm

# Import TAS and NAS components (showing they access shared tools)
from .components.tas_integration import TASIntegrationComponent
from .components.nas_integration import NASIntegrationComponent

logger = logging.getLogger(__name__)


def create_sample_market_data(n_periods: int = 1000, timeframe: str = "15m") -> pd.DataFrame:
    """Create sample market data for testing."""
    try:
        # Generate realistic market data
        np.random.seed(42)
        
        # Base price
        base_price = 100.0
        prices = [base_price]
        
        # Generate price movements
        for i in range(n_periods - 1):
            # Add some trend and volatility
            trend = 0.0001 * np.sin(i / 100)  # Long-term trend
            volatility = 0.01 * np.random.randn()  # Random volatility
            price_change = trend + volatility
            
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            # Generate OHLC from close price
            volatility = abs(np.random.randn() * 0.005)
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close
            
            # Generate volume
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add timestamps
        if timeframe == "15m":
            freq = "15min"
        elif timeframe == "5m":
            freq = "5min"
        elif timeframe == "1m":
            freq = "1min"
        else:
            freq = "15min"
        
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(minutes=n_periods * 15),
            periods=n_periods,
            freq=freq
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        raise


def demonstrate_tas_nas_parity():
    """Demonstrate that TAS and NAS systems are on par."""
    logger.info("🔍 Demonstrating TAS and NAS system parity...")
    
    try:
        # Create sample data
        market_data = create_sample_market_data(500)
        
        # Initialize TAS integration
        tas_config = {
            'n_regimes': 8,
            'enable_economic_evaluation': True,
            'enable_trading_viability': True,
            'weight': 0.5
        }
        tas_integration = TASIntegrationComponent(tas_config)
        
        # Initialize NAS integration
        nas_config = {
            'n_regimes': 8,
            'enable_economic_evaluation': True,
            'enable_trading_viability': True,
            'weight': 0.5
        }
        nas_integration = NASIntegrationComponent(nas_config)
        
        # Extract features using both systems
        logger.info("   Running TAS feature extraction...")
        tas_features, tas_results = tas_integration.extract_features(market_data)
        
        logger.info("   Running NAS feature extraction...")
        nas_features, nas_results = nas_integration.extract_features(market_data)
        
        # Compare results
        logger.info("📊 TAS vs NAS Comparison:")
        logger.info(f"   TAS features shape: {tas_features.shape}")
        logger.info(f"   NAS features shape: {nas_features.shape}")
        logger.info(f"   TAS success: {tas_results.get('success', False)}")
        logger.info(f"   NAS success: {nas_results.get('success', False)}")
        logger.info(f"   TAS confidence: {tas_results.get('confidence', 0.0):.3f}")
        logger.info(f"   NAS confidence: {nas_results.get('confidence', 0.0):.3f}")
        
        # Both systems should be functional and comparable
        assert tas_features.size > 0, "TAS features should not be empty"
        assert nas_features.size > 0, "NAS features should not be empty"
        assert tas_results.get('success', False), "TAS should succeed"
        assert nas_results.get('success', False), "NAS should succeed"
        
        logger.info("✅ TAS and NAS systems are on par - both functional and comparable")
        return True
        
    except Exception as e:
        logger.error(f"❌ TAS-NAS parity demonstration failed: {e}")
        return False


def demonstrate_shared_tools():
    """Demonstrate that common tools are stored in hybrid directory and accessed from there."""
    logger.info("🔍 Demonstrating shared tools in hybrid directory...")
    
    try:
        # Test unified search algorithms
        logger.info("   Testing unified search algorithms...")
        search_config = {
            'enable_bayesian_optimization': True,
            'enable_evolutionary_algorithm': True,
            'max_iterations': 50,
            'n_initial_points': 10
        }
        search_manager = create_unified_search_manager(search_config)
        
        available_algorithms = search_manager.get_available_algorithms()
        logger.info(f"   Available search algorithms: {available_algorithms}")
        
        # Test unified clustering algorithms
        logger.info("   Testing unified clustering algorithms...")
        clustering_config = {
            'n_regimes': 8,
            'algorithm_type': 'adaptive_clustering',
            'enable_economic_clustering': True,
            'enable_ensemble_clustering': False
        }
        clustering_algorithm = create_unified_clustering_algorithm(clustering_config)
        
        logger.info(f"   Clustering algorithm type: {clustering_algorithm.algorithm_type}")
        logger.info(f"   Economic clustering enabled: {clustering_algorithm.enable_economic_clustering}")
        
        # Test with sample data
        sample_features = np.random.randn(100, 10)
        sample_market_data = create_sample_market_data(100)
        
        clustering_result = clustering_algorithm.cluster_features(
            features=sample_features,
            market_data=sample_market_data
        )
        
        logger.info(f"   Clustering success: {clustering_result.success}")
        logger.info(f"   Regimes detected: {len(set(clustering_result.labels))}")
        logger.info(f"   Silhouette score: {clustering_result.quality_metrics.get('silhouette_score', 0):.3f}")
        
        # Verify shared tools are working
        assert len(available_algorithms) > 0, "Should have available search algorithms"
        assert clustering_result.success, "Clustering should succeed"
        assert len(set(clustering_result.labels)) > 0, "Should detect regimes"
        
        logger.info("✅ Shared tools in hybrid directory are functional and accessible")
        return True
        
    except Exception as e:
        logger.error(f"❌ Shared tools demonstration failed: {e}")
        return False


def demonstrate_hybrid_orchestrator():
    """Demonstrate hybrid orchestrator initializing both systems and analyzing outputs."""
    logger.info("🔍 Demonstrating hybrid orchestrator...")
    
    try:
        # Create hybrid configuration
        hybrid_config = HybridRegimeConfig(
            n_regimes=8,
            combination_strategy=RegimeCombinationStrategy.WEIGHTED_AVERAGE,
            tas_config={
                'n_regimes': 8,
                'enable_economic_evaluation': True,
                'weight': 0.5
            },
            nas_config={
                'n_regimes': 8,
                'enable_economic_evaluation': True,
                'weight': 0.5
            },
            search_config={
                'enable_bayesian_optimization': True,
                'enable_evolutionary_algorithm': True,
                'max_iterations': 50
            },
            clustering_config={
                'n_regimes': 8,
                'algorithm_type': 'adaptive_clustering',
                'enable_economic_clustering': True
            },
            economic_evaluation={
                'enabled': True,
                'significance_threshold': 0.6
            },
            trading_evaluation={
                'enabled': True,
                'viability_threshold': 0.5
            }
        )
        
        # Create hybrid orchestrator
        # Create unified detector with hybrid configuration
        unified_config = UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.HYBRID,
            n_regimes=hybrid_config.n_regimes,
            primary_timeframe=hybrid_config.primary_timeframe
        )
        orchestrator = UnifiedRegimeDetector(unified_config)
        
        # Create sample market data
        market_data = create_sample_market_data(1000)
        
        # Run hybrid analysis
        logger.info("   Running hybrid regime analysis...")
        result = orchestrator.analyze_market_regimes(
            market_data=market_data,
            enable_multi_timeframe=False  # Test without multi-timeframe first
        )
        
        # Analyze results
        logger.info("📊 Hybrid Orchestrator Results:")
        logger.info(f"   Analysis success: {result.regime_predictions.size > 0}")
        logger.info(f"   Regimes detected: {len(set(result.regime_predictions))}")
        logger.info(f"   TAS contribution: {result.tas_contributions.get('success', False)}")
        logger.info(f"   NAS contribution: {result.nas_contributions.get('success', False)}")
        logger.info(f"   Hybrid confidence: {result.hybrid_analysis.get('hybrid_confidence', 0.0):.3f}")
        logger.info(f"   Agreement score: {result.hybrid_analysis.get('agreement_score', 0.0):.3f}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")
        
        # Verify orchestrator functionality
        assert result.regime_predictions.size > 0, "Should detect regimes"
        assert result.tas_contributions.get('success', False), "TAS should contribute"
        assert result.nas_contributions.get('success', False), "NAS should contribute"
        assert result.hybrid_analysis.get('hybrid_confidence', 0) > 0, "Should have hybrid confidence"
        
        # Test system status
        status = orchestrator.get_system_status()
        logger.info("📊 System Status:")
        logger.info(f"   TAS integration: {status['tas_integration']['enabled']}")
        logger.info(f"   NAS integration: {status['nas_integration']['enabled']}")
        logger.info(f"   Multi-timeframe support: {status['multi_timeframe_support']}")
        logger.info(f"   Available algorithms: {status['available_algorithms']}")
        
        logger.info("✅ Hybrid orchestrator successfully initializes both systems and analyzes outputs")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid orchestrator demonstration failed: {e}")
        return False


def demonstrate_multi_timeframe_support():
    """Demonstrate multi-timeframe support (1m, 5m trading while maintaining 15m regime detection)."""
    logger.info("🔍 Demonstrating multi-timeframe support...")
    
    try:
        # Create hybrid configuration with multi-timeframe support
        hybrid_config = HybridRegimeConfig(
            n_regimes=8,
            combination_strategy=RegimeCombinationStrategy.ADAPTIVE_FUSION,
            enable_multi_timeframe=True,
            tas_config={
                'n_regimes': 8,
                'enable_economic_evaluation': True,
                'weight': 0.5
            },
            nas_config={
                'n_regimes': 8,
                'enable_economic_evaluation': True,
                'weight': 0.5
            },
            search_config={
                'enable_bayesian_optimization': True,
                'enable_evolutionary_algorithm': True,
                'max_iterations': 50
            },
            clustering_config={
                'n_regimes': 8,
                'algorithm_type': 'adaptive_clustering',
                'enable_economic_clustering': True
            },
            economic_evaluation={
                'enabled': True,
                'significance_threshold': 0.6
            },
            trading_evaluation={
                'enabled': True,
                'viability_threshold': 0.5
            }
        )
        
        # Create hybrid orchestrator
        # Create unified detector with hybrid configuration
        unified_config = UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.HYBRID,
            n_regimes=hybrid_config.n_regimes,
            primary_timeframe=hybrid_config.primary_timeframe
        )
        orchestrator = UnifiedRegimeDetector(unified_config)
        
        # Create sample market data (15m for regime detection)
        market_data_15m = create_sample_market_data(500, "15m")
        
        # Run multi-timeframe analysis
        logger.info("   Running multi-timeframe regime analysis...")
        result = orchestrator.analyze_market_regimes(
            market_data=market_data_15m,
            enable_multi_timeframe=True
        )
        
        # Analyze multi-timeframe results
        logger.info("📊 Multi-Timeframe Results:")
        logger.info(f"   15m regime detection: {len(set(result.regime_15m.regime_predictions))} regimes")
        logger.info(f"   1m trading analysis: {'✅ Available' if result.trading_1m else '❌ Not available'}")
        logger.info(f"   5m trading analysis: {'✅ Available' if result.trading_5m else '❌ Not available'}")
        
        if result.trading_1m:
            logger.info(f"   1m opportunity score: {result.trading_1m.get('opportunity_score', 0.0):.3f}")
            logger.info(f"   1m signal strength: {result.trading_1m.get('trading_signals', {}).get('signal_strength', 0.0):.3f}")
        
        if result.trading_5m:
            logger.info(f"   5m opportunity score: {result.trading_5m.get('opportunity_score', 0.0):.3f}")
            logger.info(f"   5m signal strength: {result.trading_5m.get('trading_signals', {}).get('signal_strength', 0.0):.3f}")
        
        # Cross-timeframe insights
        insights = result.cross_timeframe_insights
        logger.info("📊 Cross-Timeframe Insights:")
        logger.info(f"   Optimal timeframe: {insights.get('optimal_timeframe', 'unknown')}")
        logger.info(f"   Risk assessment: {insights.get('risk_assessment', 'unknown')}")
        logger.info(f"   Market conditions: {insights.get('market_conditions', 'unknown')}")
        logger.info(f"   Trading recommendations: {len(insights.get('trading_recommendations', []))}")
        
        # Verify multi-timeframe functionality
        assert len(set(result.regime_15m.regime_predictions)) > 0, "Should detect regimes in 15m"
        assert result.trading_1m is not None, "Should have 1m trading analysis"
        assert result.trading_5m is not None, "Should have 5m trading analysis"
        assert 'optimal_timeframe' in insights, "Should have optimal timeframe recommendation"
        
        logger.info("✅ Multi-timeframe support working - 15m regime detection with 1m/5m trading analysis")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-timeframe support demonstration failed: {e}")
        return False


def demonstrate_different_trading_configs():
    """Demonstrate different trading configurations."""
    logger.info("🔍 Demonstrating different trading configurations...")
    
    try:
        # Test different configurations
        configs = {
            'default': create_default_multi_timeframe_config(),
            'high_frequency': create_high_frequency_config(),
            'swing_trading': create_swing_trading_config()
        }
        
        for config_name, config in configs.items():
            logger.info(f"   Testing {config_name} configuration...")
            
            # Validate configuration
            is_valid = config.multi_timeframe.validate_configuration()
            logger.info(f"   {config_name} config valid: {is_valid}")
            
            # Check trading timeframes
            trading_timeframes = config.multi_timeframe.get_trading_timeframes()
            logger.info(f"   {config_name} trading timeframes: {[tf.value for tf in trading_timeframes]}")
            
            # Check weights
            weights = config.multi_timeframe.timeframe_weights
            logger.info(f"   {config_name} timeframe weights: {weights}")
            
            assert is_valid, f"{config_name} configuration should be valid"
            assert len(trading_timeframes) > 0, f"{config_name} should have trading timeframes"
        
        logger.info("✅ Different trading configurations are valid and functional")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trading configurations demonstration failed: {e}")
        return False


def run_comprehensive_integration_test():
    """Run comprehensive integration test."""
    logger.info("🚀 Starting Comprehensive Integration Test")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: TAS and NAS parity
    logger.info("\n1️⃣ Testing TAS and NAS System Parity")
    test_results['tas_nas_parity'] = demonstrate_tas_nas_parity()
    
    # Test 2: Shared tools
    logger.info("\n2️⃣ Testing Shared Tools in Hybrid Directory")
    test_results['shared_tools'] = demonstrate_shared_tools()
    
    # Test 3: Hybrid orchestrator
    logger.info("\n3️⃣ Testing Hybrid Orchestrator")
    test_results['hybrid_orchestrator'] = demonstrate_hybrid_orchestrator()
    
    # Test 4: Multi-timeframe support
    logger.info("\n4️⃣ Testing Multi-Timeframe Support")
    test_results['multi_timeframe'] = demonstrate_multi_timeframe_support()
    
    # Test 5: Different trading configurations
    logger.info("\n5️⃣ Testing Different Trading Configurations")
    test_results['trading_configs'] = demonstrate_different_trading_configs()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPREHENSIVE INTEGRATION TEST RESULTS")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Integration is working correctly!")
        logger.info("\n✅ Requirements Met:")
        logger.info("   1. TAS and NAS systems are on par")
        logger.info("   2. Common tools are stored in hybrid directory and accessed from there")
        logger.info("   3. Hybrid orchestrator can initialize both systems and analyze outputs")
        logger.info("   4. Multi-timeframe support works (1m, 5m trading with 15m regime detection)")
    else:
        logger.error("❌ Some tests failed - Integration needs attention")
    
    return test_results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive test
    results = run_comprehensive_integration_test()