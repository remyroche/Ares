"""
Gate Feature Integration Example

This example demonstrates how to use the gate feature integration system
in the pre-training pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager, GateFeatureConfig, create_gate_manager,
    enable_gate_protection, disable_gate_protection, get_gate_manager
)
from src.training.steps.pre_training.gate_feature_pipeline_integration import (
    GateFeaturePipelineIntegration, create_gate_feature_integration,
    integrate_gate_features_with_pipeline
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success, tprint_error


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create good quality features
    good_features = pd.DataFrame({
        'price_momentum': np.random.randn(1000),
        'volume_ratio': np.random.randn(1000),
        'volatility': np.random.randn(1000),
        'rsi': np.random.randn(1000),
        'macd': np.random.randn(1000)
    })
    
    # Create targets
    targets = pd.Series(np.random.randint(0, 2, 1000))
    
    return good_features, targets


def create_problematic_data():
    """Create problematic data for demonstration."""
    np.random.seed(42)
    
    # Create features with various issues
    problematic_features = pd.DataFrame({
        'good_feature': np.random.randn(1000),
        'high_nan_feature': [np.nan if i % 3 == 0 else np.random.randn() for i in range(1000)],
        'low_variance_feature': np.ones(1000) + np.random.normal(0, 0.001, 1000),
        'correlated_feature_1': np.random.randn(1000),
        'correlated_feature_2': np.random.randn(1000) + 0.99 * np.random.randn(1000)  # Highly correlated
    })
    
    # Create targets with low variance
    targets = pd.Series(np.ones(1000))  # All same value - no variance
    
    return problematic_features, targets


def demonstrate_basic_gate_features():
    """Demonstrate basic gate feature functionality."""
    tprint_info("🔍 Demonstrating Basic Gate Feature Functionality")
    
    # Create sample data
    features, targets = create_sample_data()
    
    # Create gate manager
    manager = create_gate_manager()
    
    # Enable gate protection
    manager.enable_gate_protection()
    tprint_success("✅ Gate protection enabled")
    
    # Evaluate gate features
    tprint_info("Evaluating gate features...")
    gate_results = manager.evaluate_gate_features(features, targets)
    
    # Display results
    tprint_info(f"Found {len(gate_results)} gate results:")
    for result in gate_results:
        status_emoji = {
            'passed': '✅',
            'failed': '❌',
            'warning': '⚠️',
            'skipped': '⏭️'
        }.get(result.status.value, '❓')
        
        tprint_info(f"  {status_emoji} {result.feature_name}: {result.message}")
    
    # Select gate features
    selected_features = manager.select_gate_features(features, targets)
    tprint_info(f"Selected gate features: {selected_features}")
    
    # Get gate status
    status = manager.get_gate_status()
    tprint_info(f"Gate status: {status}")


def demonstrate_problematic_data_handling():
    """Demonstrate handling of problematic data."""
    tprint_info("🚨 Demonstrating Problematic Data Handling")
    
    # Create problematic data
    features, targets = create_problematic_data()
    
    # Create gate manager
    manager = create_gate_manager()
    
    # Evaluate gate features
    tprint_info("Evaluating problematic data...")
    gate_results = manager.evaluate_gate_features(features, targets)
    
    # Display results
    tprint_info(f"Found {len(gate_results)} gate results:")
    for result in gate_results:
        status_emoji = {
            'passed': '✅',
            'failed': '❌',
            'warning': '⚠️',
            'skipped': '⏭️'
        }.get(result.status.value, '❓')
        
        tprint_info(f"  {status_emoji} {result.feature_name}: {result.message}")
    
    # Show which gates failed
    failed_gates = [r for r in gate_results if r.status.value == 'failed']
    if failed_gates:
        tprint_warning(f"⚠️ {len(failed_gates)} gates failed - corrective measures needed")


def demonstrate_pipeline_integration():
    """Demonstrate pipeline integration."""
    tprint_info("🔧 Demonstrating Pipeline Integration")
    
    # Create sample data
    features, targets = create_sample_data()
    
    # Create pipeline data
    pipeline_data = {
        'features': features,
        'targets': targets,
        'metadata': {
            'symbol': 'ETHUSDT',
            'timeframe': '15m',
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Create integration component
    integration = create_gate_feature_integration()
    
    # Process through integration
    tprint_info("Processing data through gate feature integration...")
    result = integration.process(pipeline_data)
    
    if result.success:
        tprint_success("✅ Pipeline integration successful")
        
        # Display gate features
        if 'gate_features' in result.data:
            gate_features = result.data['gate_features']
            tprint_info(f"Selected gate features: {gate_features['selected_features']}")
            tprint_info(f"Gate evaluation count: {gate_features['evaluation_count']}")
        
        # Display gate status
        if 'gate_status' in result.data:
            gate_status = result.data['gate_status']
            tprint_info(f"Gate status: {gate_status}")
    else:
        tprint_error(f"❌ Pipeline integration failed: {result.error_message}")


def demonstrate_corrective_measures():
    """Demonstrate corrective measures for problematic data."""
    tprint_info("🔧 Demonstrating Corrective Measures")
    
    # Create problematic data
    features, targets = create_problematic_data()
    
    # Create pipeline data
    pipeline_data = {
        'features': features,
        'targets': targets
    }
    
    # Create integration with corrective measures enabled
    integration = create_gate_feature_integration()
    
    # Process through integration
    tprint_info("Processing problematic data with corrective measures...")
    result = integration.process(pipeline_data)
    
    if result.success:
        tprint_success("✅ Corrective measures applied successfully")
        
        # Show statistics
        stats = integration.get_gate_statistics()
        tprint_info(f"Gate statistics: {stats}")
    else:
        tprint_error(f"❌ Corrective measures failed: {result.error_message}")


def demonstrate_convenience_function():
    """Demonstrate the convenience integration function."""
    tprint_info("🎯 Demonstrating Convenience Function")
    
    # Create sample data
    features, targets = create_sample_data()
    
    # Create pipeline data
    pipeline_data = {
        'features': features,
        'targets': targets
    }
    
    # Use convenience function
    tprint_info("Using convenience function for integration...")
    result = integrate_gate_features_with_pipeline(pipeline_data)
    
    if 'gate_features' in result:
        tprint_success("✅ Convenience function successful")
        tprint_info(f"Gate features integrated: {len(result['gate_features']['selected_features'])} features")
    else:
        tprint_warning("⚠️ No gate features found in result")


def demonstrate_configuration():
    """Demonstrate configuration options."""
    tprint_info("⚙️ Demonstrating Configuration Options")
    
    # Create custom configuration
    custom_config = {
        'enable_gate_protection': True,
        'max_gate_features_per_base': 5,
        'min_gate_ic_improvement': 0.01,
        'min_gate_stability': 0.5,
        'max_nan_ratio': 0.2,
        'min_variance_threshold': 1e-6,
        'enable_gate_monitoring': True,
        'enable_gate_reporting': True
    }
    
    # Create manager with custom config
    manager = create_gate_manager(custom_config)
    
    tprint_info(f"Custom configuration applied:")
    tprint_info(f"  Max gate features per base: {manager.config.max_gate_features_per_base}")
    tprint_info(f"  Min IC improvement: {manager.config.min_gate_ic_improvement}")
    tprint_info(f"  Min gate stability: {manager.config.min_gate_stability}")
    tprint_info(f"  Max NaN ratio: {manager.config.max_nan_ratio}")
    
    # Test with sample data
    features, targets = create_sample_data()
    gate_results = manager.evaluate_gate_features(features, targets)
    
    tprint_info(f"Gate evaluation completed with {len(gate_results)} results")


def demonstrate_monitoring_and_reporting():
    """Demonstrate monitoring and reporting capabilities."""
    tprint_info("📊 Demonstrating Monitoring and Reporting")
    
    # Create manager with monitoring enabled
    config = {
        'enable_gate_monitoring': True,
        'enable_gate_reporting': True,
        'gate_monitoring_frequency': 1
    }
    
    manager = create_gate_manager(config)
    
    # Process multiple datasets
    for i in range(3):
        features, targets = create_sample_data()
        gate_results = manager.evaluate_gate_features(features, targets)
        
        tprint_info(f"Dataset {i+1}: {len(gate_results)} gate results")
    
    # Get final status
    status = manager.get_gate_status()
    tprint_info(f"Final gate status: {status}")


def main():
    """Main demonstration function."""
    tprint("🚀 Gate Feature Integration Demonstration", color="cyan", bold=True)
    tprint("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_basic_gate_features()
        print()
        
        demonstrate_problematic_data_handling()
        print()
        
        demonstrate_pipeline_integration()
        print()
        
        demonstrate_corrective_measures()
        print()
        
        demonstrate_convenience_function()
        print()
        
        demonstrate_configuration()
        print()
        
        demonstrate_monitoring_and_reporting()
        print()
        
        tprint_success("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
