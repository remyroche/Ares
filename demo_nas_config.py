#!/usr/bin/env python3
"""
Demonstration script for NAS Configuration Module

This script demonstrates the key features and functionality of the NAS clustering
configuration system.
"""

import nas_config
from nas_config import NASArchitectureType, create_default_config, create_trading_config, create_production_config

def main():
    """Main demonstration function."""
    print("🚀 NAS Configuration Module Demonstration")
    print("=" * 50)
    
    # 1. Architecture Type Demonstration
    print("\n1. Architecture Type Features:")
    print("-" * 30)
    
    # Show different architecture types
    arch_types = [
        NASArchitectureType.RANDOM_FOREST,
        NASArchitectureType.LSTM_NN,
        NASArchitectureType.TRADING_ENSEMBLE,
        NASArchitectureType.REGIME_AWARE_TREE
    ]
    
    for arch_type in arch_types:
        print(f"  • {arch_type.name}")
        print(f"    - Tree-based: {arch_type.is_tree_based()}")
        print(f"    - Neural network: {arch_type.is_neural_network()}")
        print(f"    - Ensemble: {arch_type.is_ensemble()}")
        print(f"    - Trading-specific: {arch_type.is_trading_specific()}")
        print(f"    - Complexity factor: {arch_type.get_complexity_factor()}")
        print()
    
    # 2. Configuration Creation
    print("2. Configuration Creation:")
    print("-" * 30)
    
    # Default configuration
    default_config = create_default_config()
    print(f"  • Default config: {default_config.architecture_type.name}")
    print(f"    - Clustering: {default_config.clustering_config.algorithm}")
    print(f"    - Search strategy: {default_config.search_config.search_strategy}")
    print(f"    - Max generations: {default_config.search_config.max_generations}")
    
    # Trading configuration
    trading_config = create_trading_config()
    print(f"  • Trading config: {trading_config.architecture_type.name}")
    print(f"    - Clustering: {trading_config.clustering_config.algorithm}")
    print(f"    - Time series CV: {trading_config.validation_config['time_series_split']}")
    print(f"    - Purged CV: {trading_config.validation_config['purged_cv']}")
    
    # Production configuration
    production_config = create_production_config()
    print(f"  • Production config: {production_config.architecture_type.name}")
    print(f"    - Max generations: {production_config.search_config.max_generations}")
    print(f"    - Population size: {production_config.search_config.population_size}")
    print(f"    - Memory limit: {production_config.hardware_config.memory_limit_gb} GB")
    
    # 3. Hardware Optimization
    print("\n3. Hardware Optimization:")
    print("-" * 30)
    
    optimization_results = trading_config.optimize_for_hardware()
    print(f"  • M1 optimization: {optimization_results['m1_optimization']}")
    print(f"  • MPS acceleration: {optimization_results['mps_acceleration']}")
    print(f"  • Memory optimization: {optimization_results['memory_optimization']}")
    print(f"  • CPU optimization: {optimization_results['cpu_optimization']}")
    
    if optimization_results['recommendations']:
        print("  • Recommendations:")
        for rec in optimization_results['recommendations']:
            print(f"    - {rec}")
    
    # 4. Data Validation
    print("\n4. Data Validation:")
    print("-" * 30)
    
    # Test with sample data
    sample_data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0]
    ]
    
    validation_result = trading_config.validate_data(sample_data)
    print(f"  • Data valid: {validation_result['is_valid']}")
    print(f"  • Data type: {validation_result['data_type']}")
    print(f"  • Data shape: {validation_result['data_shape']}")
    print(f"  • Quality metrics: {validation_result['quality_metrics']}")
    
    if validation_result['issues']:
        print("  • Issues:")
        for issue in validation_result['issues']:
            print(f"    - {issue}")
    
    if validation_result['recommendations']:
        print("  • Recommendations:")
        for rec in validation_result['recommendations']:
            print(f"    - {rec}")
    
    # 5. Architecture Complexity Analysis
    print("\n5. Architecture Complexity Analysis:")
    print("-" * 30)
    
    complexity = trading_config.get_architecture_complexity()
    print(f"  • Architecture: {complexity['architecture_type']}")
    print(f"  • Complexity factor: {complexity['complexity_factor']}")
    print(f"  • Estimated training time: {complexity['estimated_training_time']:.1f}s")
    print(f"  • Estimated memory usage: {complexity['estimated_memory_usage']:.1f} MB")
    print(f"  • Recommended workers: {complexity['recommended_workers']}")
    
    # 6. Serialization
    print("\n6. Configuration Serialization:")
    print("-" * 30)
    
    # Save configuration
    config_file = "/tmp/demo_nas_config.json"
    success = trading_config.save_to_file(config_file)
    print(f"  • Configuration saved: {success}")
    
    # Load configuration
    loaded_config = nas_config.NASClusteringConfig.load_from_file(config_file)
    print(f"  • Configuration loaded: {loaded_config.architecture_type.name}")
    print(f"  • Configurations match: {trading_config.architecture_type == loaded_config.architecture_type}")
    
    # 7. Configuration Summary
    print("\n7. Configuration Summary:")
    print("-" * 30)
    
    config_dict = trading_config.to_dict()
    print(f"  • Total configuration keys: {len(config_dict)}")
    print(f"  • Architecture type: {config_dict['architecture_type']['name']}")
    print(f"  • Version: {config_dict['version']}")
    print(f"  • Timestamp: {config_dict['timestamp']}")
    
    print("\n✅ Demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  • Multiple architecture types with complexity analysis")
    print("  • Configuration presets for different use cases")
    print("  • Hardware optimization for M1 systems")
    print("  • Data validation with quality metrics")
    print("  • Serialization to/from JSON files")
    print("  • Integration with utility modules")

if __name__ == "__main__":
    main()