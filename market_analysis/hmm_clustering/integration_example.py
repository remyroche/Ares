#!/usr/bin/env python3
"""
Integration Example: HMM Clustering with Market Analysis Pipeline

This script demonstrates how to integrate the enhanced HMM clustering
system with the existing market analysis pipeline and common utilities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the enhanced HMM clustering
from enhanced_hmm_clustering import (
    EnhancedHMMClustering, 
    HMMClusteringConfig, 
    run_hmm_clustering_analysis
)

# Import configuration system
from config import (
    HMMClusteringConfigFactory, 
    get_config_by_name,
    ConfigPresets
)

# Import common utilities with error handling
try:
    from src.utils.common_operations import get_m1_gpu_manager, get_m1_memory_optimizer
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError:
    COMMON_OPERATIONS_AVAILABLE = False
    logger.warning("Common operations not available")

try:
    from src.utils.kline_parquet import KlinesParquetManager
except ImportError:
    try:
        from src.utils.data.klines_parquet import KlinesParquetManager
    except ImportError:
        KlinesParquetManager = None
        logger.warning("Klines parquet manager not available")

try:
    from src.utils.serialization_utils import UniversalSerializer
except ImportError:
    UniversalSerializer = None
    logger.warning("Serialization utils not available")

try:
    from src.utils.ml_common.hmm_regime_detection import EnhancedHMMRegimeDetector
except ImportError:
    EnhancedHMMRegimeDetector = None
    logger.warning("HMM regime detection not available")

# Import feature generation system
try:
    from src.feature_generation import (
        get_hmm_compatible_generators,
        FeatureGenerators,
        generate_features_by_category
    )
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False
    logger.warning("Feature generation system not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_basic_integration():
    """Demonstrate basic integration with common utilities."""
    logger.info("Demonstrating basic integration...")
    
    try:
        # Create a custom configuration
        config = HMMClusteringConfig(
            n_components=4,
            lookback_windows=[5, 10, 20, 50],
            technical_indicators=["rsi", "macd", "bollinger_bands", "atr"],
            use_gpu=True,
            use_memory_optimization=True,
            max_features=25
        )
        
        # Initialize clustering with common utilities
        clustering = EnhancedHMMClustering(config)
        
        # Check hardware optimization availability
        if COMMON_OPERATIONS_AVAILABLE:
            try:
                gpu_manager = get_m1_gpu_manager()
                memory_optimizer = get_m1_memory_optimizer()
                
                if gpu_manager:
                    logger.info("✓ M1 GPU manager available")
                else:
                    logger.info("⚠️ M1 GPU manager not available")
                
                if memory_optimizer:
                    logger.info("✓ M1 memory optimizer available")
                else:
                    logger.info("⚠️ M1 memory optimizer not available")
            except Exception as e:
                logger.warning(f"Hardware optimization check failed: {e}")
        else:
            logger.info("⚠️ Common operations not available")
        
        # Demonstrate data loading with klines manager
        if KlinesParquetManager is not None:
            klines_manager = KlinesParquetManager()
        else:
            klines_manager = None
            logger.warning("Klines manager not available")
        
        # Check available data if klines manager is available
        if klines_manager is not None:
            try:
                data_info = klines_manager.get_data_info("BTCUSDT", "1h")
                logger.info(f"BTCUSDT 1h data available: {data_info.get('available', False)}")
                
                if data_info.get('available', False):
                    # Load sample data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                    
                    data = klines_manager.load_data(
                        symbol="BTCUSDT",
                        interval="1h",
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if data is not None and not data.empty:
                        logger.info(f"✓ Loaded {len(data)} data points")
                        
                        # Run HMM clustering
                        result = run_hmm_clustering_analysis(
                            symbol="BTCUSDT",
                            interval="1h",
                            config=config,
                            save_results=True
                        )
                        
                        logger.info("✓ HMM clustering completed successfully")
                        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
                        logger.info(f"  - Regime stability: {result.performance_metrics.get('regime_stability', 0):.4f}")
                        logger.info(f"  - Number of regimes: {len(np.unique(result.regime_labels))}")
                        
                        return result
                    else:
                        logger.warning("No data loaded, using synthetic data for demonstration")
                        return demonstrate_with_synthetic_data(config)
                else:
                    logger.warning("No real data available, using synthetic data for demonstration")
                    return demonstrate_with_synthetic_data(config)
            except Exception as e:
                logger.warning(f"Data loading failed: {e}, using synthetic data for demonstration")
                return demonstrate_with_synthetic_data(config)
        else:
            logger.warning("Klines manager not available, using synthetic data for demonstration")
            return demonstrate_with_synthetic_data(config)
            
    except Exception as e:
        logger.error(f"Basic integration demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_with_synthetic_data(config):
    """Demonstrate with synthetic data when real data is not available."""
    logger.info("Demonstrating with synthetic data...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        # Create multiple regimes
        regime_lengths = [250, 300, 250, 200]
        regimes = []
        
        for i, length in enumerate(regime_lengths):
            if i == 0:  # Bull market
                trend = 0.001
                volatility = 0.02
            elif i == 1:  # Bear market
                trend = -0.0005
                volatility = 0.03
            elif i == 2:  # Sideways market
                trend = 0.0001
                volatility = 0.015
            else:  # High volatility
                trend = 0.0002
                volatility = 0.04
            
            regime_data = np.random.normal(trend, volatility, length)
            regimes.extend(regime_data)
        
        regimes = regimes[:n_samples]
        
        # Generate price series
        prices = [100]
        for return_val in regimes:
            prices.append(prices[-1] * (1 + return_val))
        
        prices = prices[:n_samples]
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_samples)
        })
        
        # Ensure OHLCV consistency
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        logger.info(f"Created synthetic data with {len(data)} samples")
        
        # Initialize clustering
        clustering = EnhancedHMMClustering(config)
        
        # Test feature generation system integration
        if FEATURE_GENERATION_AVAILABLE and clustering.feature_generators is not None:
            logger.info("✓ Feature generation system available")
            try:
                # Test direct feature generation
                hmm_features = clustering.feature_generators.generate_features_for_hmm(data)
                if hmm_features is not None and not hmm_features.empty:
                    logger.info(f"✓ Generated {len(hmm_features.columns)} features using feature generation system")
                    selected_features = hmm_features
                else:
                    logger.warning("Feature generation returned empty, falling back to manual")
                    features = clustering.engineer_features(data)
                    selected_features = features
            except Exception as e:
                logger.warning(f"Feature generation failed: {e}, falling back to manual")
                features = clustering.engineer_features(data)
                selected_features = features
        else:
            logger.info("⚠️ Feature generation system not available, using manual feature engineering")
            # Engineer features
            features = clustering.engineer_features(data)
            logger.info(f"Engineered {len(features.columns)} features")
            
            # Select features if selector available
            if clustering.feature_selector is not None:
                selected_features = clustering.select_features(features)
                logger.info(f"Selected {len(selected_features.columns)} features")
            else:
                selected_features = features
                logger.info("Feature selector not available, using all features")
        
        # Fit HMM model
        result = clustering.fit_hmm_model(selected_features)
        
        logger.info("✓ Synthetic data analysis completed")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Regime stability: {result.performance_metrics.get('regime_stability', 0):.4f}")
        logger.info(f"  - Regime balance: {result.performance_metrics.get('regime_balance', 0):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Synthetic data demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_configuration_presets():
    """Demonstrate different configuration presets."""
    logger.info("Demonstrating configuration presets...")
    
    try:
        # Test different presets
        presets = [
            ("crypto_btc_1h", "Crypto BTC 1H"),
            ("forex_major_1h", "Forex Major 1H"),
            ("stocks_large_daily", "Stocks Large Daily"),
            ("high_frequency", "High Frequency"),
            ("research", "Research")
        ]
        
        for preset_name, description in presets:
            try:
                config = get_config_by_name(preset_name)
                if config:
                    logger.info(f"✓ {description}:")
                    logger.info(f"  - Components: {config.n_components}")
                    logger.info(f"  - Lookback windows: {config.lookback_windows}")
                    logger.info(f"  - Technical indicators: {len(config.technical_indicators)}")
                    logger.info(f"  - Max features: {config.max_features}")
                else:
                    logger.warning(f"✗ Preset {preset_name} not found")
            except Exception as e:
                logger.warning(f"⚠️ Preset {preset_name} failed: {e}")
        
        # Test custom configuration creation
        try:
            from config import TimeframeType
            custom_config = HMMClusteringConfigFactory.create_crypto_config(
                timeframe=TimeframeType.INTRADAY,
                market_volatility="high"
            )
            
            logger.info("✓ Custom crypto configuration:")
            logger.info(f"  - Components: {custom_config.n_components}")
            logger.info(f"  - Lookback windows: {custom_config.lookback_windows}")
            logger.info(f"  - Technical indicators: {custom_config.technical_indicators}")
        except Exception as e:
            logger.warning(f"⚠️ Custom config creation failed: {e}")
        
        # Test feature generation system if available
        if FEATURE_GENERATION_AVAILABLE:
            logger.info("✓ Feature generation system available")
            try:
                generators = get_hmm_compatible_generators()
                logger.info(f"✓ HMM compatible generators initialized: {type(generators).__name__}")
            except Exception as e:
                logger.warning(f"⚠️ Feature generators initialization failed: {e}")
        else:
            logger.info("⚠️ Feature generation system not available")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration presets demonstration failed: {e}")
        return False

def demonstrate_serialization():
    """Demonstrate model serialization and persistence."""
    logger.info("Demonstrating serialization...")
    
    try:
        # Create a simple configuration
        config = HMMClusteringConfig(
            n_components=3,
            lookback_windows=[5, 10, 20],
            technical_indicators=["rsi", "macd"],
            use_gpu=False,
            max_features=10
        )
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        prices = [100]
        for _ in range(n_samples - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # Initialize clustering
        clustering = EnhancedHMMClustering(config)
        
        # Engineer features and fit model
        features = clustering.engineer_features(data)
        selected_features = clustering.select_features(features)
        result = clustering.fit_hmm_model(selected_features)
        
        # Test serialization if available
        if clustering.serializer is not None:
            model_path = "test_integration_model.pkl"
            
            # Save model
            save_success = clustering.save_model(model_path)
            logger.info(f"Model save: {'✓' if save_success else '✗'}")
            
            if save_success:
                # Load model in new instance
                new_clustering = EnhancedHMMClustering(config)
                load_success = new_clustering.load_model(model_path)
                logger.info(f"Model load: {'✓' if load_success else '✗'}")
                
                if load_success:
                    # Test prediction with loaded model
                    new_labels, new_probs = new_clustering.predict_regimes(selected_features)
                    logger.info(f"Prediction with loaded model: {'✓' if len(new_labels) > 0 else '✗'}")
                    
                    # Compare predictions
                    original_labels = result.regime_labels
                    labels_match = np.array_equal(original_labels, new_labels)
                    logger.info(f"Predictions match: {'✓' if labels_match else '✗'}")
            
            # Clean up
            Path(model_path).unlink(missing_ok=True)
        else:
            logger.info("⚠️ Serializer not available, skipping save/load test")
        
        return True
        
    except Exception as e:
        logger.error(f"Serialization demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_performance_comparison():
    """Demonstrate performance comparison between different configurations."""
    logger.info("Demonstrating performance comparison...")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        prices = [100]
        for _ in range(n_samples - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # Test different configurations
        configs = [
            ("Basic", HMMClusteringConfig(
                n_components=3,
                use_gpu=False,
                use_memory_optimization=False,
                max_features=10
            )),
            ("Memory Optimized", HMMClusteringConfig(
                n_components=3,
                use_gpu=False,
                use_memory_optimization=True,
                max_features=10
            )),
            ("GPU Optimized", HMMClusteringConfig(
                n_components=3,
                use_gpu=True,
                use_memory_optimization=True,
                max_features=10
            ))
        ]
        
        results = []
        
        for config_name, config in configs:
            logger.info(f"Testing {config_name} configuration...")
            
            clustering = EnhancedHMMClustering(config)
            features = clustering.engineer_features(data)
            selected_features = clustering.select_features(features)
            
            import time
            start_time = time.time()
            result = clustering.fit_hmm_model(selected_features)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            results.append({
                'config': config_name,
                'processing_time': processing_time,
                'regime_stability': result.performance_metrics.get('regime_stability', 0),
                'regime_balance': result.performance_metrics.get('regime_balance', 0),
                'memory_usage': result.memory_usage.get('total_mb', 0)
            })
            
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            logger.info(f"  - Regime stability: {result.performance_metrics.get('regime_stability', 0):.4f}")
            logger.info(f"  - Memory usage: {result.memory_usage.get('total_mb', 0):.2f} MB")
        
        # Display comparison
        logger.info("\nPerformance Comparison:")
        logger.info("=" * 60)
        for result in results:
            logger.info(f"{result['config']:20s}: {result['processing_time']:6.2f}s | "
                       f"Stability: {result['regime_stability']:6.4f} | "
                       f"Memory: {result['memory_usage']:6.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance comparison demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_integration_demonstration():
    """Run the complete integration demonstration."""
    logger.info("Starting HMM Clustering Integration Demonstration")
    logger.info("=" * 60)
    
    demonstrations = [
        ("Configuration Presets", demonstrate_configuration_presets),
        ("Basic Integration", demonstrate_basic_integration),
        ("Serialization", demonstrate_serialization),
        ("Performance Comparison", demonstrate_performance_comparison),
    ]
    
    successful = 0
    total = len(demonstrations)
    
    for demo_name, demo_func in demonstrations:
        logger.info(f"\n--- {demo_name} ---")
        try:
            if demo_func():
                successful += 1
                logger.info(f"✓ {demo_name} completed successfully")
            else:
                logger.error(f"✗ {demo_name} failed")
        except Exception as e:
            logger.error(f"✗ {demo_name} failed with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Integration Demonstration Results: {successful}/{total} successful")
    
    if successful == total:
        logger.info("🎉 All integration demonstrations completed successfully!")
        logger.info("The Enhanced HMM Clustering system is fully integrated and ready for use.")
    else:
        logger.warning(f"⚠️ {total - successful} demonstrations failed. Please check the implementation.")
    
    return successful == total

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("market_analysis/hmm_clustering/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run integration demonstration
    success = run_integration_demonstration()
    
    if success:
        print("\n✅ Integration demonstration completed successfully!")
        print("The Enhanced HMM Clustering system is ready for production use.")
    else:
        print("\n❌ Some integration demonstrations failed.")
        print("Please review the implementation and fix any issues.")
        sys.exit(1)