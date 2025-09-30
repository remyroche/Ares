"""
Enhanced TAS Regime Detection Demo

This script demonstrates the enhanced TAS regime detection system with:
- Memory-efficient processing for large datasets
- Parallel processing across timeframes
- Intelligent caching
- Cross-validation and out-of-sample testing
- Regime persistence analysis
- Bayesian TPE optimization
- M1 hardware optimizations
- Advanced matrix operations
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional
import warnings

# Import enhanced TAS components
from ..core.enhanced_tas_regime_detector import EnhancedTASRegimeDetector, EnhancedTASRegimeResult
from ..core.tas_regime_config import TASRegimeConfig

# Import optimization tools
try:
    from ...utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from ...utils.ml_common.optimization.grid_utils import GridSearchOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    warnings.warn("Optimization tools not available")

# Import matrix operations
try:
    from ...utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    warnings.warn("Matrix operations not available")

# Import M1 hardware optimizations
try:
    from ...utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from ...utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from ...utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    warnings.warn("Hardware optimizations not available")

logger = logging.getLogger(__name__)

def generate_sample_market_data(n_samples: int = 10000, n_features: int = 10) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    logger.info(f"📊 Generating sample market data: {n_samples} samples, {n_features} features")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate time series data with regime-like patterns
    timestamps = pd.date_range('2020-01-01', periods=n_samples, freq='15T')
    
    # Create regime-like patterns
    regime_length = n_samples // 8  # 8 regimes
    regimes = np.repeat(np.arange(8), regime_length)[:n_samples]
    
    # Generate OHLCV data with regime-specific characteristics
    data = []
    for i in range(n_samples):
        regime = regimes[i]
        
        # Regime-specific parameters
        if regime == 0:  # Bull market
            trend = 0.001
            volatility = 0.02
        elif regime == 1:  # Bear market
            trend = -0.001
            volatility = 0.03
        elif regime == 2:  # Sideways market
            trend = 0.0001
            volatility = 0.015
        elif regime == 3:  # High volatility
            trend = 0.0005
            volatility = 0.04
        elif regime == 4:  # Low volatility
            trend = 0.0002
            volatility = 0.01
        elif regime == 5:  # Trending up
            trend = 0.002
            volatility = 0.025
        elif regime == 6:  # Trending down
            trend = -0.002
            volatility = 0.025
        else:  # Random walk
            trend = 0.0001
            volatility = 0.02
        
        # Generate price data
        if i == 0:
            price = 100.0
        else:
            price = data[-1]['close'] * (1 + trend + np.random.normal(0, volatility))
        
        # Generate OHLCV
        high = price * (1 + abs(np.random.normal(0, volatility/2)))
        low = price * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = data[-1]['close'] if i > 0 else price
        close = price
        volume = np.random.exponential(1000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add technical indicators
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    logger.info(f"✅ Generated market data: {df.shape}")
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def demonstrate_enhanced_tas_regime_detection():
    """Demonstrate enhanced TAS regime detection system."""
    logger.info("🚀 Starting Enhanced TAS Regime Detection Demo")
    
    try:
        # Step 1: Generate sample data
        logger.info("📊 Step 1: Generating sample market data...")
        market_data = generate_sample_market_data(n_samples=5000, n_features=10)
        
        # Step 2: Create enhanced TAS configuration
        logger.info("⚙️ Step 2: Creating enhanced TAS configuration...")
        config = TASRegimeConfig.create_research_config()
        config.n_regimes = 8
        config.tree_depth = 8
        config.n_estimators = 500
        config.enable_hardware_optimization = True
        config.enable_matrix_optimization = True
        config.enable_memory_optimization = True
        
        # Step 3: Initialize enhanced TAS detector
        logger.info("🔧 Step 3: Initializing enhanced TAS detector...")
        detector = EnhancedTASRegimeDetector(config)
        
        # Step 4: Perform enhanced regime detection
        logger.info("🎯 Step 4: Performing enhanced regime detection...")
        start_time = time.time()
        
        result = detector.detect_regimes_enhanced(
            market_data=market_data,
            timestamps=market_data['timestamp'].values,
            enable_bayesian_optimization=True,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            enable_cross_validation=True,
            enable_out_of_sample_validation=True,
            enable_regime_persistence_analysis=True
        )
        
        detection_time = time.time() - start_time
        
        # Step 5: Display results
        logger.info("📊 Step 5: Displaying results...")
        display_enhanced_results(result, detection_time)
        
        # Step 6: Demonstrate individual components
        logger.info("🔍 Step 6: Demonstrating individual components...")
        demonstrate_individual_components(detector, market_data)
        
        logger.info("✅ Enhanced TAS Regime Detection Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced TAS demo failed: {e}")
        raise

def display_enhanced_results(result: EnhancedTASRegimeResult, detection_time: float):
    """Display enhanced TAS results."""
    logger.info("📊 Enhanced TAS Regime Detection Results:")
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Execution time: {detection_time:.2f}s")
    logger.info(f"   Regimes detected: {result.regime_count}")
    logger.info(f"   Data points: {len(result.regime_predictions)}")
    
    if result.success:
        logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
        logger.info(f"   Trading viability: {np.mean(result.trading_viability_scores):.3f}")
        logger.info(f"   Regime stability: {np.mean(result.regime_stability_scores):.3f}")
        
        # Display cross-validation results
        if result.cv_scores:
            logger.info("   Cross-validation results:")
            for metric, value in result.cv_scores.items():
                if isinstance(value, float):
                    logger.info(f"     {metric}: {value:.3f}")
        
        # Display out-of-sample results
        if result.oos_metrics:
            logger.info("   Out-of-sample results:")
            for metric, value in result.oos_metrics.items():
                if isinstance(value, float):
                    logger.info(f"     {metric}: {value:.3f}")
        
        # Display persistence analysis
        if result.persistence_analysis:
            logger.info("   Regime persistence analysis:")
            if 'regime_durations' in result.persistence_analysis:
                durations = result.persistence_analysis['regime_durations']
                logger.info(f"     Mean regime duration: {np.mean(durations):.1f} periods")
                logger.info(f"     Max regime duration: {np.max(durations)} periods")
                logger.info(f"     Min regime duration: {np.min(durations)} periods")
        
        # Display optimization results
        if result.optimization_results:
            logger.info("   Optimization results:")
            for metric, value in result.optimization_results.items():
                if isinstance(value, float):
                    logger.info(f"     {metric}: {value:.3f}s")
        
        # Display matrix operations stats
        if result.matrix_operations_stats:
            logger.info("   Matrix operations stats:")
            for metric, value in result.matrix_operations_stats.items():
                if isinstance(value, float):
                    logger.info(f"     {metric}: {value:.3f}")
        
        # Display hardware optimization stats
        if result.hardware_optimization_stats:
            logger.info("   Hardware optimization stats:")
            for component, stats in result.hardware_optimization_stats.items():
                if isinstance(stats, dict):
                    logger.info(f"     {component}: {stats}")

def demonstrate_individual_components(detector: EnhancedTASRegimeDetector, market_data: pd.DataFrame):
    """Demonstrate individual enhanced components."""
    logger.info("🔍 Demonstrating individual enhanced components...")
    
    # 1. Bayesian optimization demonstration
    if hasattr(detector, 'bayesian_optimizer') and detector.bayesian_optimizer:
        logger.info("🔬 Demonstrating Bayesian TPE optimization...")
        try:
            # Define simple search space
            search_space = {
                'n_regimes': {'type': 'int', 'low': 3, 'high': 8},
                'tree_depth': {'type': 'int', 'low': 4, 'high': 8}
            }
            
            # Simple objective function
            def simple_objective(params):
                return np.random.uniform(0.5, 0.9)  # Mock objective
            
            # Run optimization
            best_params = detector.bayesian_optimizer.optimize(simple_objective, search_space)
            logger.info(f"   Best parameters: {best_params}")
            
        except Exception as e:
            logger.warning(f"⚠️ Bayesian optimization demo failed: {e}")
    
    # 2. Matrix operations demonstration
    if hasattr(detector, 'enhanced_matrix_ops') and detector.enhanced_matrix_ops:
        logger.info("🔢 Demonstrating enhanced matrix operations...")
        try:
            # Create sample matrices
            A = np.random.randn(100, 50)
            B = np.random.randn(50, 100)
            
            # Matrix multiplication
            result = detector.enhanced_matrix_ops.matrix_multiply(A, B)
            logger.info(f"   Matrix multiplication result shape: {result.shape}")
            
            # Correlation matrix
            corr_matrix = detector.enhanced_matrix_ops.safe_correlation_matrix(A)
            logger.info(f"   Correlation matrix shape: {corr_matrix.shape}")
            
        except Exception as e:
            logger.warning(f"⚠️ Matrix operations demo failed: {e}")
    
    # 3. Hardware optimization demonstration
    if hasattr(detector, 'm1_gpu_manager') and detector.m1_gpu_manager:
        logger.info("⚡ Demonstrating M1 hardware optimizations...")
        try:
            # Get GPU info
            gpu_info = detector.m1_gpu_manager.get_gpu_info()
            logger.info(f"   GPU info: {gpu_info}")
            
        except Exception as e:
            logger.warning(f"⚠️ GPU optimization demo failed: {e}")
    
    if hasattr(detector, 'm1_memory_optimizer') and detector.m1_memory_optimizer:
        try:
            # Get memory stats
            memory_stats = detector.m1_memory_optimizer.get_memory_stats()
            logger.info(f"   Memory stats: {memory_stats}")
            
        except Exception as e:
            logger.warning(f"⚠️ Memory optimization demo failed: {e}")
    
    if hasattr(detector, 'm1_cpu_optimizer') and detector.m1_cpu_optimizer:
        try:
            # Get CPU info
            cpu_info = detector.m1_cpu_optimizer.get_cpu_info()
            logger.info(f"   CPU info: {cpu_info}")
            
        except Exception as e:
            logger.warning(f"⚠️ CPU optimization demo failed: {e}")

def demonstrate_performance_comparison():
    """Demonstrate performance comparison between standard and enhanced TAS."""
    logger.info("⚡ Demonstrating performance comparison...")
    
    try:
        # Generate test data
        test_data = generate_sample_market_data(n_samples=2000, n_features=5)
        
        # Standard TAS configuration
        standard_config = TASRegimeConfig.create_production_config()
        
        # Enhanced TAS configuration
        enhanced_config = TASRegimeConfig.create_research_config()
        
        # Test standard TAS
        logger.info("📊 Testing standard TAS...")
        from ..core.tas_regime_detector import TASRegimeDetector
        
        standard_detector = TASRegimeDetector(standard_config)
        start_time = time.time()
        standard_result = standard_detector.detect_regimes(test_data)
        standard_time = time.time() - start_time
        
        # Test enhanced TAS
        logger.info("🚀 Testing enhanced TAS...")
        enhanced_detector = EnhancedTASRegimeDetector(enhanced_config)
        start_time = time.time()
        enhanced_result = enhanced_detector.detect_regimes_enhanced(
            test_data,
            enable_bayesian_optimization=False,  # Skip for speed
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            enable_cross_validation=False,  # Skip for speed
            enable_out_of_sample_validation=False,  # Skip for speed
            enable_regime_persistence_analysis=False  # Skip for speed
        )
        enhanced_time = time.time() - start_time
        
        # Compare results
        logger.info("📊 Performance Comparison:")
        logger.info(f"   Standard TAS time: {standard_time:.2f}s")
        logger.info(f"   Enhanced TAS time: {enhanced_time:.2f}s")
        logger.info(f"   Speed improvement: {standard_time/enhanced_time:.2f}x")
        
        if standard_result.success and enhanced_result.success:
            logger.info(f"   Standard regimes: {len(np.unique(standard_result.regime_predictions))}")
            logger.info(f"   Enhanced regimes: {len(np.unique(enhanced_result.regime_predictions))}")
        
    except Exception as e:
        logger.warning(f"⚠️ Performance comparison failed: {e}")

def main():
    """Main demonstration function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🎯 Enhanced TAS Regime Detection System Demo")
    logger.info("=" * 60)
    
    try:
        # Run main demonstration
        demonstrate_enhanced_tas_regime_detection()
        
        # Run performance comparison
        demonstrate_performance_comparison()
        
        logger.info("🎉 Enhanced TAS Regime Detection Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()