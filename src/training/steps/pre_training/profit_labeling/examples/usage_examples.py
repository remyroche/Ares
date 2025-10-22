"""
Usage Examples for Enhanced Profit Labeling System

This module provides comprehensive examples of how to use the enhanced profit labeling
system with different configurations and use cases.

Author: AI Assistant
Date: 2025-01-10
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system import (
    EnhancedProfitLabelingSystem, ProfitLabelingConfig
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    tprint_info("🔧 Example 1: Basic Usage")
    
    # Create basic configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        max_features=100,
        enable_bayesian_optimization=False
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Run pipeline
    results = system.run_full_pipeline()
    
    tprint_success("✅ Basic usage completed")
    return results


def example_advanced_usage():
    """Example 2: Advanced usage with optimization and multiple symbols."""
    tprint_info("🔧 Example 2: Advanced Usage")
    
    # Create advanced configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        timeframes=["1h", "4h", "1d"],
        feature_categories=["volatility", "momentum", "volume", "trend"],
        max_features=500,
        feature_selection_method="ensemble",
        enable_bayesian_optimization=True,
        n_trials=100,
        n_jobs=4,
        enable_gpu=False,
        enable_parallel=True,
        memory_efficient=True,
        volatility_threshold=0.025,
        target_thresholds={
            "small": 0.01,
            "medium": 0.025,
            "high": 0.05
        },
        min_quality_score=0.75,
        enable_noise_gating=True,
        enable_leakage_detection=True
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Run pipeline
    results = system.run_full_pipeline()
    
    tprint_success("✅ Advanced usage completed")
    return results


def example_custom_data_loading():
    """Example 3: Custom data loading with specific date ranges."""
    tprint_info("🔧 Example 3: Custom Data Loading")
    
    # Create configuration with custom date range
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        max_features=200
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Load data manually
    data = system.load_data()
    tprint_info(f"Loaded {len(data)} datasets")
    
    # Generate features manually
    features = system.generate_features(data)
    tprint_info(f"Generated features for {len(features)} datasets")
    
    # Generate labels manually
    labels = system.generate_labels(data)
    tprint_info(f"Generated labels for {len(labels)} datasets")
    
    tprint_success("✅ Custom data loading completed")
    return {"data": data, "features": features, "labels": labels}


def example_feature_selection_comparison():
    """Example 4: Compare different feature selection methods."""
    tprint_info("🔧 Example 4: Feature Selection Comparison")
    
    # Create configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        max_features=50
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Load data
    data = system.load_data()
    features = system.generate_features(data)
    labels = system.generate_labels(data)
    
    # Compare different feature selection methods
    methods = ["mrmr", "lasso", "rfe", "ensemble"]
    results = {}
    
    for method in methods:
        tprint_info(f"Testing {method} feature selection...")
        
        # Update configuration
        config.feature_selection_method = method
        
        # Reinitialize system with new method
        system = EnhancedProfitLabelingSystem(config)
        
        # Select features
        selected_features = system.select_features(features, labels)
        
        results[method] = selected_features
        tprint_info(f"{method}: {len(selected_features.get('BTCUSDT_1h', []))} features selected")
    
    tprint_success("✅ Feature selection comparison completed")
    return results


def example_hyperparameter_optimization():
    """Example 5: Hyperparameter optimization with custom search space."""
    tprint_info("🔧 Example 5: Hyperparameter Optimization")
    
    # Create configuration with optimization enabled
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        enable_bayesian_optimization=True,
        n_trials=50,
        max_features=100
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Load data
    data = system.load_data()
    features = system.generate_features(data)
    labels = system.generate_labels(data)
    
    # Run optimization
    optimization_results = system.optimize_hyperparameters(features, labels)
    
    tprint_info(f"Optimization results: {optimization_results}")
    tprint_success("✅ Hyperparameter optimization completed")
    return optimization_results


def example_quality_evaluation():
    """Example 6: Label quality evaluation and analysis."""
    tprint_info("🔧 Example 6: Quality Evaluation")
    
    # Create configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        max_features=200
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Run pipeline
    results = system.run_full_pipeline()
    
    # Analyze evaluation results
    evaluation = results.get('evaluation', {})
    
    tprint_info("📊 Quality Evaluation Results:")
    for dataset, metrics in evaluation.items():
        tprint_info(f"  {dataset}:")
        for metric, value in metrics.items():
            tprint_info(f"    {metric}: {value:.4f}")
    
    tprint_success("✅ Quality evaluation completed")
    return evaluation


def example_batch_processing():
    """Example 7: Batch processing multiple configurations."""
    tprint_info("🔧 Example 7: Batch Processing")
    
    # Define multiple configurations
    configurations = [
        {
            "name": "Conservative",
            "config": ProfitLabelingConfig(
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                volatility_threshold=0.01,
                target_thresholds={"small": 0.005, "medium": 0.01, "high": 0.02},
                max_features=100
            )
        },
        {
            "name": "Moderate",
            "config": ProfitLabelingConfig(
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                volatility_threshold=0.02,
                target_thresholds={"small": 0.01, "medium": 0.02, "high": 0.04},
                max_features=200
            )
        },
        {
            "name": "Aggressive",
            "config": ProfitLabelingConfig(
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                volatility_threshold=0.03,
                target_thresholds={"small": 0.02, "medium": 0.04, "high": 0.08},
                max_features=300
            )
        }
    ]
    
    batch_results = {}
    
    for config_info in configurations:
        name = config_info["name"]
        config = config_info["config"]
        
        tprint_info(f"Processing {name} configuration...")
        
        # Initialize system
        system = EnhancedProfitLabelingSystem(config)
        
        # Run pipeline
        results = system.run_full_pipeline()
        
        batch_results[name] = results
        tprint_info(f"Completed {name}: {len(results['data'])} datasets")
    
    tprint_success("✅ Batch processing completed")
    return batch_results


def example_custom_feature_categories():
    """Example 8: Custom feature categories and generation."""
    tprint_info("🔧 Example 8: Custom Feature Categories")
    
    # Create configuration with custom feature categories
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        feature_categories=["volatility", "momentum", "volume"],
        max_features=150
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Load data
    data = system.load_data()
    
    # Generate features with custom categories
    features = system.generate_features(data)
    
    # Analyze feature categories
    for dataset, feature_df in features.items():
        tprint_info(f"Features for {dataset}:")
        tprint_info(f"  Total features: {len(feature_df.columns)}")
        tprint_info(f"  Feature types: {feature_df.dtypes.value_counts().to_dict()}")
    
    tprint_success("✅ Custom feature categories completed")
    return features


def example_performance_monitoring():
    """Example 9: Performance monitoring and optimization."""
    tprint_info("🔧 Example 9: Performance Monitoring")
    
    # Create configuration with performance monitoring
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        enable_parallel=True,
        memory_efficient=True,
        max_features=300
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    # Monitor performance during pipeline execution
    import time
    start_time = time.time()
    
    # Run pipeline
    results = system.run_full_pipeline()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    tprint_info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
    tprint_info(f"📊 Datasets processed: {len(results['data'])}")
    tprint_info(f"🔧 Features generated: {sum(results['features'].values())}")
    tprint_info(f"🏷️ Labels generated: {sum(results['labels'].values())}")
    
    tprint_success("✅ Performance monitoring completed")
    return results


def main():
    """Run all examples."""
    tprint_info("🚀 Running Enhanced Profit Labeling System Examples")
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Advanced Usage", example_advanced_usage),
        ("Custom Data Loading", example_custom_data_loading),
        ("Feature Selection Comparison", example_feature_selection_comparison),
        ("Hyperparameter Optimization", example_hyperparameter_optimization),
        ("Quality Evaluation", example_quality_evaluation),
        ("Batch Processing", example_batch_processing),
        ("Custom Feature Categories", example_custom_feature_categories),
        ("Performance Monitoring", example_performance_monitoring)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            tprint_info(f"\n{'='*50}")
            tprint_info(f"Running: {name}")
            tprint_info(f"{'='*50}")
            
            result = example_func()
            results[name] = result
            
            tprint_success(f"✅ {name} completed successfully")
            
        except Exception as e:
            tprint_warning(f"⚠️ {name} failed: {e}")
            results[name] = {"error": str(e)}
    
    tprint_info(f"\n{'='*50}")
    tprint_info("📋 All Examples Summary")
    tprint_info(f"{'='*50}")
    
    for name, result in results.items():
        if "error" in result:
            tprint_warning(f"❌ {name}: Failed")
        else:
            tprint_success(f"✅ {name}: Completed")
    
    return results


if __name__ == "__main__":
    main()