"""
Example Usage of Data-Driven Lookback Optimization System

This script demonstrates how to use the three-stage Bayesian optimization
system to replace hardcoded lookback ceilings with data-driven inference.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import the optimization system
from .orchestrator import LookbackOptimizationOrchestrator
from .config import create_development_config, create_production_config, FamilyType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_symbols: int = 3, n_days: int = 1000) -> tuple:
    """Generate sample market data for demonstration."""
    logger.info(f"Generating sample data for {n_symbols} symbols over {n_days} days...")
    
    data = {}
    targets = {}
    
    for i in range(n_symbols):
        symbol = f"SYMBOL_{i+1}"
        
        # Generate price data
        np.random.seed(42 + i)  # Reproducible randomness
        
        # Random walk with trend
        returns = np.random.normal(0.0001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        high_low_noise = np.random.uniform(0.001, 0.005, n_days)
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_days)),
            'high': prices * (1 + high_low_noise),
            'low': prices * (1 - high_low_noise),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, n_days)
        })
        
        # Add some technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = calculate_rsi(df['close'], 14)
        
        data[symbol] = df
        
        # Generate target (future returns)
        future_returns = df['close'].pct_change(5).shift(-5)  # 5-day forward returns
        targets[symbol] = future_returns.fillna(0).values
    
    logger.info(f"Generated data for symbols: {list(data.keys())}")
    return data, targets


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = self._vectorbt_rolling_operation(gain, "mean", window)
    avg_loss = self._vectorbt_rolling_operation(loss, "mean", window)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)


def run_development_example():
    """Run a development example with relaxed constraints."""
    logger.info("🚀 Running development example...")
    
    # Generate sample data
    data, targets = generate_sample_data(n_symbols=2, n_days=500)
    
    # Create development configuration
    config = create_development_config()
    
    # Initialize orchestrator
    orchestrator = LookbackOptimizationOrchestrator(config)
    
    # Define feature names
    feature_names = {
        FamilyType.MOMENTUM: "momentum_feature",
        FamilyType.VOLATILITY: "volatility_feature", 
        FamilyType.GK: "gk_volatility_feature",
        FamilyType.VWAP_ROLL: "vwap_roll_feature",
        FamilyType.RSI: "rsi_feature",
        FamilyType.AUTOCORR: "autocorr_feature"
    }
    
    # Run optimization
    result = orchestrator.optimize_lookbacks(data, targets, feature_names)
    
    if result.success:
        logger.info("✅ Optimization completed successfully!")
        
        # Generate comprehensive report
        report = orchestrator.generate_comprehensive_report(result)
        
        # Print key results
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        print(f"Execution time: {result.execution_time:.3f}s")
        print(f"Symbols processed: {len(result.ic_surface_results)}")
        print(f"Success: {result.success}")
        
        # Print decision summary
        decision_counts = {'discrete': 0, 'blend': 0, 'default': 0, 'inactive': 0}
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                decision_type = decision.lookback_spec.decision_type.value
                decision_counts[decision_type] += 1
        
        print(f"\nDecision types:")
        for decision_type, count in decision_counts.items():
            print(f"  {decision_type}: {count}")
        
        # Print family performance
        print(f"\nFamily performance:")
        for family in FamilyType:
            family_ics = []
            for symbol_results in result.ic_surface_results.values():
                if family in symbol_results:
                    family_ics.append(symbol_results[family].optimal_ic)
            
            if family_ics:
                avg_ic = np.mean(family_ics)
                print(f"  {family.value}: {avg_ic:.4f} (avg IC)")
        
        # Print feature quality
        if result.feature_results:
            all_quality_scores = []
            for symbol_results in result.feature_results.values():
                for feature_result in symbol_results.values():
                    all_quality_scores.append(feature_result.quality_score)
            
            if all_quality_scores:
                avg_quality = np.mean(all_quality_scores)
                print(f"\nAverage feature quality: {avg_quality:.3f}")
        
        # Print recommendations
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("="*60)
        
    else:
        logger.error(f"❌ Optimization failed: {result.error_message}")
    
    return result


def run_production_example():
    """Run a production example with strict constraints."""
    logger.info("🏭 Running production example...")
    
    # Generate more realistic data
    data, targets = generate_sample_data(n_symbols=5, n_days=2000)
    
    # Create production configuration
    config = create_production_config()
    
    # Initialize orchestrator
    orchestrator = LookbackOptimizationOrchestrator(config)
    
    # Define feature names
    feature_names = {
        FamilyType.MOMENTUM: "momentum_feature",
        FamilyType.VOLATILITY: "volatility_feature",
        FamilyType.GK: "gk_volatility_feature", 
        FamilyType.VWAP_ROLL: "vwap_roll_feature",
        FamilyType.RSI: "rsi_feature",
        FamilyType.AUTOCORR: "autocorr_feature"
    }
    
    # Run optimization
    result = orchestrator.optimize_lookbacks(data, targets, feature_names)
    
    if result.success:
        logger.info("✅ Production optimization completed successfully!")
        
        # Generate report
        report = orchestrator.generate_comprehensive_report(result)
        
        # Print production-specific metrics
        print("\n" + "="*60)
        print("PRODUCTION OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"Execution time: {result.execution_time:.3f}s")
        print(f"Symbols processed: {len(result.ic_surface_results)}")
        
        # Check production constraints
        total_features = 0
        for symbol_results in result.feature_results.values():
            total_features += len(symbol_results)
        
        print(f"Total features generated: {total_features}")
        print(f"Feature budget (120): {'✅' if total_features <= 120 else '❌'}")
        
        # Check decision stability
        stable_decisions = 0
        total_decisions = 0
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                total_decisions += 1
                if decision.lookback_spec.decision_type.value in ['discrete', 'blend']:
                    stable_decisions += 1
        
        stability_rate = stable_decisions / total_decisions if total_decisions > 0 else 0
        print(f"Decision stability rate: {stability_rate:.3f}")
        print(f"Stability threshold (0.8): {'✅' if stability_rate >= 0.8 else '❌'}")
        
        print("="*60)
        
    else:
        logger.error(f"❌ Production optimization failed: {result.error_message}")
    
    return result


def demonstrate_individual_stages():
    """Demonstrate individual stage usage for advanced users."""
    logger.info("🔧 Demonstrating individual stage usage...")
    
    # Generate sample data
    data, targets = generate_sample_data(n_symbols=1, n_days=1000)
    symbol = list(data.keys())[0]
    symbol_data = data[symbol]
    symbol_target = targets[symbol]
    
    # Stage 1: IC Surface Estimation
    from .ic_surface import ICSurfaceEstimator
    from .config import create_development_config
    
    config = create_development_config()
    ic_estimator = ICSurfaceEstimator(config)
    
    logger.info("Running Stage 1: IC Surface Estimation...")
    ic_result = ic_estimator.estimate_surface(
        symbol_data, symbol_target, FamilyType.MOMENTUM, "momentum_feature"
    )
    
    print(f"Optimal lookback: {ic_result.optimal_lookback:.1f}")
    print(f"Optimal IC: {ic_result.optimal_ic:.4f}")
    print(f"R-squared: {ic_result.r_squared:.3f}")
    
    # Stage 2: Stability Testing
    from .wf_stability import StabilityTester
    
    stability_tester = StabilityTester(config)
    
    logger.info("Running Stage 2: Stability Testing...")
    stability_result = stability_tester.test_stability(
        symbol_data, symbol_target, ic_result, "momentum_feature"
    )
    
    print(f"Match rate: {stability_result.match_rate:.3f}")
    print(f"Stability score: {stability_result.stability_score:.3f}")
    print(f"Recommendation: {stability_result.recommendation}")
    
    # Stage 3: Hierarchical Shrinkage (if PyMC available)
    try:
        from .hierarchical import HierarchicalBayesianShrinkage, SymbolFamilyData

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
        
        hierarchical_shrinkage = HierarchicalBayesianShrinkage(config)
        
        # Create sample symbol-family data
        symbol_family_data = [
            SymbolFamilyData(
                symbol=symbol,
                family=FamilyType.MOMENTUM,
                estimated_lookback=ic_result.optimal_lookback,
                lookback_std=ic_result.optimal_ic_error,
                ic_value=ic_result.optimal_ic,
                ic_std=ic_result.optimal_ic_error,
                n_observations=len(ic_result.lookbacks),
                stability_score=stability_result.stability_score
            )
        ]
        
        logger.info("Running Stage 3: Hierarchical Shrinkage...")
        hierarchical_result = hierarchical_shrinkage.apply_shrinkage(symbol_family_data)
        
        print(f"Family mean: {hierarchical_result.family_means[FamilyType.MOMENTUM]:.3f}")
        print(f"Family std: {hierarchical_result.family_std[FamilyType.MOMENTUM]:.3f}")
        
    except ImportError:
        logger.warning("PyMC not available - skipping hierarchical shrinkage demo")
    
    logger.info("✅ Individual stage demonstration completed!")


if __name__ == "__main__":
    print("Data-Driven Lookback Optimization System - Example Usage")
    print("=" * 60)
    
    # Run development example
    print("\n1. Development Example (relaxed constraints)")
    dev_result = run_development_example()
    
    # Run production example
    print("\n2. Production Example (strict constraints)")
    prod_result = run_production_example()
    
    # Demonstrate individual stages
    print("\n3. Individual Stage Demonstration")
    demonstrate_individual_stages()
    
    print("\n" + "="*60)
    print("Example usage completed!")
    print("="*60)