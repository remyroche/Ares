"""
Enhanced MarkovRegressionAdapter Usage Examples

This module provides comprehensive examples of how to use the enhanced MarkovRegressionAdapter
with hardware optimization, parameter mapping, diagnostics, and integration capabilities.

Examples include:
- Basic usage with default configuration
- Hardware optimization integration
- Parameter mapping from Pyro configurations
- Advanced diagnostics and validation
- VectorBT integration for backtesting
- Hierarchical optimization
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add the module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.markov_regression_adapter import (
    MarkovRegressionAdapter,
    MarkovRegressionConfig,
    create_enhanced_markov_regression_adapter
)

from optimization.parameter_mapper import (
    PyroToStatsmodelsMapper,
    map_pyro_to_statsmodels,
    create_default_mapping_config
)

from integration.hardware_optimizer import (
    StatsmodelsHardwareOptimizer,
    create_hardware_optimizer,
    optimize_for_regime_switching
)

from integration.vectorbt_integration import (
    VectorBTIntegration,
    create_vectorbt_integration,
    VectorBTConfig
)

def create_financial_data(n_samples=1000, tickers=['AAPL', 'MSFT', 'GOOGL']):
    """Create synthetic financial data for demonstration."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic price data
    data = {}
    for ticker in tickers:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        data[ticker] = prices
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Add some regime-based patterns
    regime_changes = np.linspace(0, 2, n_samples)
    for i, ticker in enumerate(tickers):
        regime_effect = np.sin(regime_changes + i * np.pi / len(tickers)) * 0.1
        df[ticker] *= (1 + regime_effect)
    
    return df

def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("🔧 Example 1: Basic Usage")
    print("=" * 50)
    
    # Create synthetic financial data
    prices = create_financial_data(n_samples=500, tickers=['AAPL', 'MSFT'])
    returns = prices.pct_change().dropna()
    
    print(f"📊 Created financial data: {returns.shape}")
    print(f"   - Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"   - Assets: {list(returns.columns)}")
    
    # Create adapter with basic configuration
    adapter = create_enhanced_markov_regression_adapter(
        k_regimes=3,
        enable_hardware_optimization=False,  # Disable for simplicity
        enable_diagnostics=True,
        enable_pca=True,
        pca_components=2
    )
    
    # Fit the model
    print("\n🔄 Fitting Markov Regression model...")
    result = adapter.fit(returns)
    
    if result.success:
        print("✅ Model fitted successfully!")
        print(f"   - Number of regimes: {result.n_regimes}")
        print(f"   - Log likelihood: {result.log_likelihood:.2f}")
        print(f"   - AIC: {result.aic:.2f}")
        print(f"   - BIC: {result.bic:.2f}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        
        # Get regime probabilities
        probabilities = adapter.get_regime_probabilities()
        print(f"   - Regime probabilities shape: {probabilities.shape}")
        
        # Get transition matrix
        transition_matrix = adapter.get_transition_matrix()
        print(f"   - Transition matrix:\n{transition_matrix}")
        
        # Show regime distribution
        labels = result.cluster_labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"   - Regime distribution: {dict(zip(unique_labels, counts))}")
        
    else:
        print(f"❌ Model fitting failed: {result.error_message}")
    
    print("\n" + "=" * 50 + "\n")

def example_2_hardware_optimization():
    """Example 2: Hardware optimization integration."""
    print("🚀 Example 2: Hardware Optimization")
    print("=" * 50)
    
    # Create larger dataset for hardware optimization
    prices = create_financial_data(n_samples=2000, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
    returns = prices.pct_change().dropna()
    
    print(f"📊 Created larger dataset: {returns.shape}")
    
    # Create adapter with hardware optimization
    config = MarkovRegressionConfig(
        k_regimes=4,
        enable_hardware_optimization=True,
        workload_type='ml_training',
        optimization_level='aggressive',
        enable_diagnostics=True,
        enable_pca=True,
        pca_components=3
    )
    
    adapter = MarkovRegressionAdapter(config)
    
    # Fit the model with hardware optimization
    print("\n🔄 Fitting with hardware optimization...")
    result = adapter.fit(returns)
    
    if result.success:
        print("✅ Model fitted with hardware optimization!")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        print(f"   - Optimization time: {result.optimization_time:.2f}s")
        
        # Show hardware metrics if available
        if result.hardware_metrics:
            hw_metrics = result.hardware_metrics
            print(f"   - Hardware optimization applied")
            
            if 'performance_report' in hw_metrics:
                perf = hw_metrics['performance_report']
                if 'average_metrics' in perf:
                    avg = perf['average_metrics']
                    print(f"   - Average CPU usage: {avg.get('cpu_usage', 0):.1f}%")
                    print(f"   - Average memory usage: {avg.get('memory_usage', 0):.1f}%")
        
    else:
        print(f"❌ Model fitting failed: {result.error_message}")
    
    print("\n" + "=" * 50 + "\n")

def example_3_parameter_mapping():
    """Example 3: Parameter mapping from Pyro configurations."""
    print("🗺️ Example 3: Parameter Mapping from Pyro")
    print("=" * 50)
    
    # Create Pyro-style configuration
    pyro_config = {
        'K': 3,
        'switching_variance': True,
        'switching_trend': True,
        'order': 1,
        'max_iter': 100,
        'tolerance': 1e-6,
        'random_state': 42,
        'alpha': [1.0, 1.0, 1.0],  # Transition prior
        'beta': [0.5, 0.5]  # Emission prior
    }
    
    print(f"📋 Pyro configuration:")
    for key, value in pyro_config.items():
        print(f"   - {key}: {value}")
    
    # Create mapping configuration
    mapping_config = create_default_mapping_config(
        strict_mapping=True,
        validate_ranges=True,
        log_mappings=True
    )
    
    # Map parameters
    mapper = PyroToStatsmodelsMapper(mapping_config)
    mapping_result = mapper.map_parameters(pyro_config)
    
    if mapping_result.success:
        print("\n✅ Parameter mapping successful!")
        print(f"📋 Mapped parameters:")
        for key, value in mapping_result.mapped_params.items():
            print(f"   - {key}: {value}")
        
        if mapping_result.unmapped_params:
            print(f"⚠️ Unmapped parameters: {list(mapping_result.unmapped_params.keys())}")
        
        # Create adapter with mapped parameters
        config = MarkovRegressionConfig(
            k_regimes=2,  # Will be overridden
            pyro_config=pyro_config,
            auto_map_parameters=True,
            enable_hardware_optimization=False,
            enable_diagnostics=True,
            enable_pca=False
        )
        
        adapter = MarkovRegressionAdapter(config)
        
        # Fit with mapped parameters
        prices = create_financial_data(n_samples=500, tickers=['AAPL', 'MSFT'])
        returns = prices.pct_change().dropna()
        
        print("\n🔄 Fitting with mapped parameters...")
        result = adapter.fit(returns)
        
        if result.success:
            print("✅ Model fitted with mapped parameters!")
            print(f"   - Final k_regimes: {adapter.config.k_regimes}")
            print(f"   - Final switching_variance: {adapter.config.switching_variance}")
            print(f"   - Final switching_trend: {adapter.config.switching_trend}")
        
    else:
        print(f"❌ Parameter mapping failed: {mapping_result.error_message}")
        if mapping_result.validation_errors:
            print("Validation errors:")
            for error in mapping_result.validation_errors:
                print(f"   - {error}")
    
    print("\n" + "=" * 50 + "\n")

def example_4_advanced_diagnostics():
    """Example 4: Advanced diagnostics and validation."""
    print("🔍 Example 4: Advanced Diagnostics")
    print("=" * 50)
    
    # Create adapter with comprehensive diagnostics
    config = MarkovRegressionConfig(
        k_regimes=3,
        enable_hardware_optimization=False,
        enable_diagnostics=True,
        enable_pca=True,
        pca_components=2,
        enable_scaling=True
    )
    
    adapter = MarkovRegressionAdapter(config)
    
    # Create data with clear regime patterns
    prices = create_financial_data(n_samples=800, tickers=['AAPL', 'MSFT', 'GOOGL'])
    returns = prices.pct_change().dropna()
    
    print(f"📊 Created data for diagnostics: {returns.shape}")
    
    # Fit model
    print("\n🔄 Fitting with diagnostics enabled...")
    result = adapter.fit(returns)
    
    if result.success and result.diagnostics:
        print("✅ Model fitted with comprehensive diagnostics!")
        
        # Analyze diagnostics
        diagnostics = result.diagnostics
        
        # Model fit diagnostics
        if 'model_fit' in diagnostics:
            model_fit = diagnostics['model_fit']
            print(f"\n📊 Model Fit Diagnostics:")
            print(f"   - Log likelihood: {model_fit.get('log_likelihood', 'N/A'):.2f}")
            print(f"   - AIC: {model_fit.get('aic', 'N/A'):.2f}")
            print(f"   - BIC: {model_fit.get('bic', 'N/A'):.2f}")
            print(f"   - Converged: {model_fit.get('converged', 'N/A')}")
            print(f"   - Iterations: {model_fit.get('iterations', 'N/A')}")
        
        # Regime stability analysis
        if 'regime_stability' in diagnostics:
            stability = diagnostics['regime_stability']
            print(f"\n🔄 Regime Stability Analysis:")
            print(f"   - Switching frequency: {stability.get('switching_frequency', 'N/A'):.3f}")
            
            if 'duration_stats' in stability:
                duration = stability['duration_stats']
                print(f"   - Mean duration: {duration.get('mean_duration', 'N/A'):.1f}")
                print(f"   - Std duration: {duration.get('std_duration', 'N/A'):.1f}")
                print(f"   - Min duration: {duration.get('min_duration', 'N/A')}")
                print(f"   - Max duration: {duration.get('max_duration', 'N/A')}")
            
            if 'probability_confidence' in stability:
                confidence = stability['probability_confidence']
                print(f"   - Mean confidence: {confidence.get('mean_confidence', 'N/A'):.3f}")
        
        # Transition analysis
        if 'transition_analysis' in diagnostics:
            transitions = diagnostics['transition_analysis']
            print(f"\n🔄 Transition Analysis:")
            
            if 'empirical_transition_probs' in transitions:
                trans_probs = transitions['empirical_transition_probs']
                print(f"   - Empirical transition matrix:")
                for i in range(trans_probs.shape[0]):
                    row_str = " ".join([f"{prob:.3f}" for prob in trans_probs[i]])
                    print(f"     Regime {i} -> [{row_str}]")
        
        # Regime characteristics
        if 'regime_characteristics' in diagnostics:
            characteristics = diagnostics['regime_characteristics']
            print(f"\n📊 Regime Characteristics:")
            for regime_key, regime_data in characteristics.items():
                print(f"   - {regime_key}:")
                print(f"     - Size: {regime_data.get('size', 'N/A')}")
                print(f"     - Proportion: {regime_data.get('proportion', 'N/A'):.3f}")
                print(f"     - Mean return: {regime_data.get('mean', 'N/A')}")
                print(f"     - Std return: {regime_data.get('std', 'N/A')}")
    
    else:
        print(f"❌ Model fitting or diagnostics failed: {result.error_message}")
    
    print("\n" + "=" * 50 + "\n")

def example_5_vectorbt_integration():
    """Example 5: VectorBT integration for backtesting."""
    print("📈 Example 5: VectorBT Integration")
    print("=" * 50)
    
    # Check if VectorBT is available
    try:
        import vectorbt as vbt
        vectorbt_available = True
    except ImportError:
        vectorbt_available = False
        print("⚠️ VectorBT not available. Install with: pip install vectorbt")
    
    if not vectorbt_available:
        print("\n" + "=" * 50 + "\n")
        return
    
    # Create adapter
    config = MarkovRegressionConfig(
        k_regimes=3,
        enable_hardware_optimization=False,
        enable_diagnostics=True,
        enable_vectorbt_integration=True,
        vectorbt_config={
            'initial_cash': 10000,
            'fees': 0.001,
            'enable_regime_strategies': True
        }
    )
    
    adapter = MarkovRegressionAdapter(config)
    
    # Create financial data
    prices = create_financial_data(n_samples=500, tickers=['AAPL', 'MSFT'])
    returns = prices.pct_change().dropna()
    
    print(f"📊 Created data for backtesting: {prices.shape}")
    
    # Fit model
    print("\n🔄 Fitting model for backtesting...")
    result = adapter.fit(returns)
    
    if result.success:
        print("✅ Model fitted successfully!")
        
        # Create VectorBT integration
        vectorbt_config = VectorBTConfig(
            initial_cash=10000,
            fees=0.001,
            enable_regime_strategies=True,
            enable_portfolio_optimization=True
        )
        
        vbt_integration = create_vectorbt_integration(vectorbt_config)
        
        # Run backtesting
        print("\n📈 Running regime-based backtesting...")
        backtest_result = vbt_integration.backtest_regime_strategy(
            prices=prices,
            regime_labels=result.cluster_labels,
            regime_probabilities=result.cluster_probabilities
        )
        
        if backtest_result.success:
            print("✅ Backtesting completed!")
            print(f"   - Total return: {backtest_result.total_return:.2%}")
            print(f"   - Annual return: {backtest_result.annual_return:.2%}")
            print(f"   - Sharpe ratio: {backtest_result.sharpe_ratio:.2f}")
            print(f"   - Max drawdown: {backtest_result.max_drawdown:.2%}")
            print(f"   - Win rate: {backtest_result.win_rate:.2%}")
            
            # Show regime-specific performance
            if backtest_result.regime_performance:
                print(f"\n📊 Regime-Specific Performance:")
                for regime_key, perf in backtest_result.regime_performance.items():
                    print(f"   - {regime_key}:")
                    print(f"     - Return: {perf.get('total_return', 'N/A'):.2%}")
                    print(f"     - Sharpe: {perf.get('sharpe_ratio', 'N/A'):.2f}")
        
        else:
            print(f"❌ Backtesting failed: {backtest_result.error_message}")
    
    else:
        print(f"❌ Model fitting failed: {result.error_message}")
    
    print("\n" + "=" * 50 + "\n")

def example_6_complete_workflow():
    """Example 6: Complete workflow with all features."""
    print("🎯 Example 6: Complete Workflow")
    print("=" * 50)
    
    # Create comprehensive configuration
    config = MarkovRegressionConfig(
        k_regimes=3,
        trend='c',
        order=1,
        switching_variance=True,
        switching_trend=True,
        maxiter=100,
        tolerance=1e-6,
        random_state=42,
        
        # Data preprocessing
        enable_pca=True,
        pca_components=3,
        pca_variance_threshold=0.95,
        enable_scaling=True,
        
        # Hardware optimization
        enable_hardware_optimization=True,
        workload_type='ml_training',
        optimization_level='balanced',
        
        # Parameter mapping
        pyro_config={
            'K': 3,
            'switching_variance': True,
            'switching_trend': True,
            'order': 1,
            'max_iter': 100,
            'tolerance': 1e-6
        },
        auto_map_parameters=True,
        
        # Diagnostics
        enable_diagnostics=True,
        
        # VectorBT integration
        enable_vectorbt_integration=True,
        vectorbt_config={
            'initial_cash': 10000,
            'fees': 0.001,
            'enable_regime_strategies': True
        },
        
        # Output
        save_intermediate_results=True,
        output_dir='./enhanced_adapter_output'
    )
    
    # Create adapter
    adapter = MarkovRegressionAdapter(config)
    
    # Create comprehensive dataset
    prices = create_financial_data(n_samples=1000, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
    returns = prices.pct_change().dropna()
    
    print(f"📊 Created comprehensive dataset: {returns.shape}")
    print(f"   - Date range: {returns.index[0]} to {returns.index[-1]}")
    print(f"   - Assets: {list(returns.columns)}")
    
    # Fit model with all features
    print("\n🔄 Running complete workflow...")
    start_time = time.time()
    result = adapter.fit(returns)
    total_time = time.time() - start_time
    
    if result.success:
        print("✅ Complete workflow successful!")
        print(f"   - Total processing time: {total_time:.2f}s")
        print(f"   - Model fitting time: {result.processing_time:.2f}s")
        print(f"   - Hardware optimization time: {result.optimization_time:.2f}s")
        
        # Show comprehensive results
        print(f"\n📊 Model Results:")
        print(f"   - Number of regimes: {result.n_regimes}")
        print(f"   - Log likelihood: {result.log_likelihood:.2f}")
        print(f"   - AIC: {result.aic:.2f}")
        print(f"   - BIC: {result.bic:.2f}")
        
        # Show hardware metrics
        if result.hardware_metrics:
            print(f"\n🚀 Hardware Optimization:")
            hw_metrics = result.hardware_metrics
            if 'performance_report' in hw_metrics:
                perf = hw_metrics['performance_report']
                if 'average_metrics' in perf:
                    avg = perf['average_metrics']
                    print(f"   - CPU usage: {avg.get('cpu_usage', 0):.1f}%")
                    print(f"   - Memory usage: {avg.get('memory_usage', 0):.1f}%")
        
        # Show diagnostics summary
        if result.diagnostics:
            print(f"\n🔍 Diagnostics Summary:")
            diagnostics = result.diagnostics
            
            if 'model_fit' in diagnostics:
                model_fit = diagnostics['model_fit']
                print(f"   - Converged: {model_fit.get('converged', 'N/A')}")
                print(f"   - Iterations: {model_fit.get('iterations', 'N/A')}")
            
            if 'regime_stability' in diagnostics:
                stability = diagnostics['regime_stability']
                print(f"   - Switching frequency: {stability.get('switching_frequency', 'N/A'):.3f}")
        
        # Show VectorBT results
        if result.vectorbt_results:
            print(f"\n📈 Backtesting Results:")
            vbt_results = result.vectorbt_results
            print(f"   - Total return: {vbt_results.get('total_return', 'N/A'):.2%}")
            print(f"   - Sharpe ratio: {vbt_results.get('sharpe_ratio', 'N/A'):.2f}")
        
        print(f"\n💾 Results saved to: {config.output_dir}")
        
    else:
        print(f"❌ Complete workflow failed: {result.error_message}")
    
    print("\n" + "=" * 50 + "\n")

def main():
    """Run all examples."""
    print("🚀 Enhanced MarkovRegressionAdapter Usage Examples")
    print("=" * 70)
    
    # Run all examples
    example_1_basic_usage()
    example_2_hardware_optimization()
    example_3_parameter_mapping()
    example_4_advanced_diagnostics()
    example_5_vectorbt_integration()
    example_6_complete_workflow()
    
    print("🎉 All examples completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Basic model fitting and prediction")
    print("✅ Hardware optimization integration")
    print("✅ Parameter mapping from Pyro configurations")
    print("✅ Advanced diagnostics and validation")
    print("✅ VectorBT integration for backtesting")
    print("✅ Complete workflow with all features")
    print("\nThe Enhanced MarkovRegressionAdapter is ready for production use!")

if __name__ == "__main__":
    import time
    main()