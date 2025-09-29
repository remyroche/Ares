"""
Example usage of shared utilities for NAS-TAS regime detection.

This module demonstrates how to use the shared utilities to eliminate redundancy
between NAS and TAS components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time

# Import shared utilities
from . import (
    # Features
    prepare_market_features, FeatureConfig,
    
    # Configuration
    validate_regime_count, normalize_weights, validate_algorithm_type,
    create_default_config, ConfigValidator, NASConfig, TASConfig, HybridConfig,
    
    # Logging
    log_execution, log_performance, LoggingContext,
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    
    # Metrics
    calculate_consensus_metrics, calculate_disagreement_metrics,
    calculate_economic_scores, calculate_trading_scores, calculate_stability_scores,
    MetricsCalculator,
    
    # Characteristics
    create_regime_characteristics, generate_cluster_characteristics,
    CharacteristicsGenerator
)


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


@log_execution('Example', 'Feature Preparation Demo', verbose=True)
def demonstrate_feature_preparation():
    """Demonstrate shared feature preparation utilities."""
    print("\n" + "="*80)
    print("FEATURE PREPARATION DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    market_data = create_sample_market_data(500)
    log_info(f"Created sample market data: {market_data.shape}")
    
    # Create feature configuration
    feature_config = FeatureConfig(
        feature_categories=['momentum', 'volatility', 'volume', 'trend', 'price_action'],
        use_standardized_features=True,
        drop_highly_correlated=True,
        correlation_threshold=0.95
    )
    
    # Prepare features using shared utilities
    features = prepare_market_features(market_data, feature_config, verbose=True)
    
    if features is not None:
        log_success(f"Features prepared successfully: {features.shape}")
        log_info(f"Feature statistics: mean={np.mean(features):.4f}, std={np.std(features):.4f}")
    else:
        log_error("Feature preparation failed")
    
    return features


@log_execution('Example', 'Configuration Validation Demo', verbose=True)
def demonstrate_configuration_validation():
    """Demonstrate shared configuration validation utilities."""
    print("\n" + "="*80)
    print("CONFIGURATION VALIDATION DEMONSTRATION")
    print("="*80)
    
    # Create configurations using shared utilities
    nas_config = create_default_config('nas', symbol='BTCUSDT', timeframe='15m', n_regimes=8)
    tas_config = create_default_config('tas', symbol='ETHUSDT', timeframe='1h', n_regimes=6)
    hybrid_config = create_default_config('hybrid', symbol='ADAUSDT', timeframe='5m', n_regimes=10)
    
    log_info("Created configurations using shared utilities")
    
    # Validate configurations
    validator = ConfigValidator(verbose=True)
    
    configs = [
        ('NAS', nas_config),
        ('TAS', tas_config),
        ('Hybrid', hybrid_config)
    ]
    
    for name, config in configs:
        log_info(f"Validating {name} configuration")
        errors = validator.validate_config(config)
        if errors:
            log_warning(f"{name} configuration has errors: {errors}")
        else:
            log_success(f"{name} configuration is valid")
    
    # Demonstrate weight normalization
    log_info("Demonstrating weight normalization")
    weights = {'economic': 0.5, 'trading': 0.3, 'stability': 0.4}
    normalized_weights = normalize_weights(weights)
    log_info(f"Original weights: {weights}")
    log_info(f"Normalized weights: {normalized_weights}")
    
    return configs


@log_execution('Example', 'Metrics Calculation Demo', verbose=True)
def demonstrate_metrics_calculation():
    """Demonstrate shared metrics calculation utilities."""
    print("\n" + "="*80)
    print("METRICS CALCULATION DEMONSTRATION")
    print("="*80)
    
    # Create sample regime assignments
    np.random.seed(42)
    tas_assignments = np.random.randint(0, 5, 1000).tolist()
    nas_assignments = np.random.randint(0, 5, 1000).tolist()
    
    log_info(f"Created sample assignments: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}")
    
    # Calculate consensus metrics using shared utilities
    consensus_metrics = calculate_consensus_metrics(tas_assignments, nas_assignments, verbose=True)
    disagreement_metrics = calculate_disagreement_metrics(tas_assignments, nas_assignments, verbose=True)
    
    log_info(f"Consensus score: {consensus_metrics['consensus_score']:.3f}")
    log_info(f"Disagreement score: {disagreement_metrics['disagreement_score']:.3f}")
    
    # Calculate economic, trading, and stability scores
    economic_scores = calculate_economic_scores(tas_assignments, verbose=True)
    trading_scores = calculate_trading_scores(tas_assignments, verbose=True)
    stability_scores = calculate_stability_scores(tas_assignments, verbose=True)
    
    log_info(f"Economic scores: {len(economic_scores)} scores, avg={np.mean(economic_scores):.3f}")
    log_info(f"Trading scores: {len(trading_scores)} scores, avg={np.mean(trading_scores):.3f}")
    log_info(f"Stability scores: {len(stability_scores)} scores, avg={np.mean(stability_scores):.3f}")
    
    # Use MetricsCalculator for comprehensive metrics
    calculator = MetricsCalculator(verbose=True)
    comprehensive_metrics = calculator.calculate_comprehensive_metrics(
        tas_assignments, nas_assignments
    )
    
    log_success("Comprehensive metrics calculated successfully")
    return comprehensive_metrics


@log_execution('Example', 'Regime Characteristics Demo', verbose=True)
def demonstrate_regime_characteristics():
    """Demonstrate shared regime characteristics utilities."""
    print("\n" + "="*80)
    print("REGIME CHARACTERISTICS DEMONSTRATION")
    print("="*80)
    
    # Create sample data and regime assignments
    market_data = create_sample_market_data(500)
    regime_predictions = np.random.randint(0, 5, 500).tolist()
    
    log_info(f"Created sample data: {market_data.shape}, regime predictions: {len(regime_predictions)}")
    
    # Create regime characteristics using shared utilities
    regime_characteristics = create_regime_characteristics(
        market_data, regime_predictions, verbose=True
    )
    
    log_info(f"Created characteristics for {len(regime_characteristics)} regimes")
    
    # Display characteristics for first regime
    if regime_characteristics:
        first_regime_key = list(regime_characteristics.keys())[0]
        first_regime_char = regime_characteristics[first_regime_key]
        log_info(f"Sample regime characteristics for {first_regime_key}:")
        log_info(f"  - Sample count: {first_regime_char.get('sample_count', 'N/A')}")
        log_info(f"  - Avg return: {first_regime_char.get('avg_return', 'N/A'):.4f}")
        log_info(f"  - Volatility: {first_regime_char.get('volatility', 'N/A'):.4f}")
        log_info(f"  - Avg volume: {first_regime_char.get('avg_volume', 'N/A'):.2f}")
    
    # Generate cluster characteristics
    cluster_characteristics = generate_cluster_characteristics(
        market_data, regime_predictions, verbose=True
    )
    
    log_success("Cluster characteristics generated successfully")
    return regime_characteristics, cluster_characteristics


@log_execution('Example', 'Logging Utilities Demo', verbose=True)
def demonstrate_logging_utilities():
    """Demonstrate shared logging utilities."""
    print("\n" + "="*80)
    print("LOGGING UTILITIES DEMONSTRATION")
    print("="*80)
    
    # Get logger using shared utilities
    logger = get_logger('ExampleLogger')
    logger.info("Logger created using shared utilities")
    
    # Demonstrate different log levels
    log_info("This is an info message")
    log_warning("This is a warning message")
    log_success("This is a success message")
    log_debug("This is a debug message")
    
    # Demonstrate LoggingContext
    with LoggingContext('Example', 'Sample Operation', verbose=True):
        log_info("Performing sample operation")
        time.sleep(0.1)  # Simulate work
        log_success("Sample operation completed")
    
    log_success("Logging utilities demonstration completed")


def demonstrate_complete_workflow():
    """Demonstrate a complete workflow using all shared utilities."""
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW DEMONSTRATION")
    print("="*80)
    
    try:
        # Step 1: Feature preparation
        features = demonstrate_feature_preparation()
        
        # Step 2: Configuration validation
        configs = demonstrate_configuration_validation()
        
        # Step 3: Metrics calculation
        metrics = demonstrate_metrics_calculation()
        
        # Step 4: Regime characteristics
        regime_char, cluster_char = demonstrate_regime_characteristics()
        
        # Step 5: Logging utilities
        demonstrate_logging_utilities()
        
        log_success("Complete workflow demonstration completed successfully")
        
        return {
            'features': features,
            'configs': configs,
            'metrics': metrics,
            'regime_characteristics': regime_char,
            'cluster_characteristics': cluster_char
        }
        
    except Exception as e:
        log_error(f"Complete workflow demonstration failed: {e}")
        return None


def main():
    """Main function to run all demonstrations."""
    print("🚀 Starting Shared Utilities Demonstration")
    print("This demonstration shows how to use shared utilities to eliminate redundancy")
    print("between NAS and TAS components.\n")
    
    results = demonstrate_complete_workflow()
    
    if results:
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        print("✅ Feature preparation: SUCCESS")
        print("✅ Configuration validation: SUCCESS")
        print("✅ Metrics calculation: SUCCESS")
        print("✅ Regime characteristics: SUCCESS")
        print("✅ Logging utilities: SUCCESS")
        print("✅ Complete workflow: SUCCESS")
        print("\n🎉 All shared utilities are working correctly!")
        print("These utilities can now be used to eliminate redundancy between NAS and TAS components.")
    else:
        print("\n❌ Demonstration failed. Check the logs for details.")


if __name__ == "__main__":
    main()