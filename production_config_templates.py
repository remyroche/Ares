#!/usr/bin/env python3
"""
Production Configuration Templates

This script creates production-ready configuration templates for different environments
and use cases of the comprehensive training pipeline.
"""

import json
import os
from pathlib import Path
from datetime import datetime

def create_production_config_templates():
    """Create production-ready configuration templates."""
    
    # Create configs directory
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    # Production configuration template
    production_config = {
        "environment": "production",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "description": "Production configuration for comprehensive training pipeline",
        
        # Basic settings
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "data_dir": "/data/market_data",
        "output_dir": "/output/training_results",
        "model_dir": "/models/trained_models",
        "log_dir": "/logs/training_logs",
        
        # Performance settings
        "enable_gpu": True,
        "enable_parallel": True,
        "max_workers": 16,
        "memory_limit": 0.9,
        "timeout_seconds": 7200,
        "random_state": 42,
        
        # Debug and logging
        "debug_mode": False,
        "verbose_logging": False,
        "log_level": "INFO",
        
        # Model training configuration
        "model_training_config": {
            "enable_confidence_metrics": True,
            "enable_calibration_assessment": True,
            "enable_feature_importance": True,
            "enable_cross_validation": True,
            "enable_model_explanations": True,
            "enable_post_training_hpo": True,
            "cv_folds": 10,
            "test_size": 0.2,
            "validation_size": 0.1,
            "early_stopping_patience": 50,
            "max_epochs": 1000,
            "batch_size": 128,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "mse",
            "metrics": ["mae", "mse", "r2", "mape"]
        },
        
        # Evaluation configuration
        "evaluation_config": {
            "enable_cross_validation": True,
            "enable_time_series_validation": True,
            "enable_confidence_intervals": True,
            "enable_model_comparison": True,
            "enable_feature_importance_analysis": True,
            "enable_prediction_analysis": True,
            "cv_folds": 10,
            "confidence_level": 0.95,
            "bootstrap_samples": 2000,
            "validation_metrics": ["sharpe_ratio", "max_drawdown", "total_return", "win_rate"],
            "backtesting_period": "2y",
            "walk_forward_analysis": True
        },
        
        # Feature engineering configuration
        "feature_engineering_config": {
            "enable_technical_indicators": True,
            "enable_statistical_features": True,
            "enable_lag_features": True,
            "enable_rolling_features": True,
            "enable_interaction_features": True,
            "max_lag": 50,
            "rolling_windows": [5, 10, 20, 50, 100, 200],
            "feature_selection_method": "recursive_feature_elimination",
            "max_features": 100,
            "feature_importance_threshold": 0.01,
            "correlation_threshold": 0.95,
            "vif_threshold": 10.0
        },
        
        # Data quality configuration
        "data_quality_config": {
            "enable_outlier_detection": True,
            "enable_missing_value_handling": True,
            "enable_data_validation": True,
            "outlier_threshold": 3.0,
            "missing_value_threshold": 0.05,
            "data_quality_threshold": 0.95,
            "enable_concept_drift_detection": True,
            "drift_detection_window": 1000,
            "drift_threshold": 0.1
        },
        
        # Performance configuration
        "performance_config": {
            "enable_memory_optimization": True,
            "enable_parallel_processing": True,
            "enable_caching": True,
            "cache_size": 50000,
            "chunk_size": 100000,
            "batch_size": 256,
            "enable_gpu_acceleration": True,
            "gpu_memory_fraction": 0.8,
            "enable_mixed_precision": True,
            "enable_gradient_checkpointing": True
        },
        
        # Monitoring and alerting
        "monitoring_config": {
            "enable_performance_monitoring": True,
            "enable_model_monitoring": True,
            "enable_data_drift_monitoring": True,
            "monitoring_interval": 300,  # 5 minutes
            "alert_thresholds": {
                "accuracy_drop": 0.05,
                "latency_increase": 2.0,
                "memory_usage": 0.9,
                "error_rate": 0.01
            },
            "notification_channels": ["email", "slack", "webhook"]
        },
        
        # Security configuration
        "security_config": {
            "enable_encryption": True,
            "enable_audit_logging": True,
            "enable_access_control": True,
            "data_retention_days": 365,
            "model_versioning": True,
            "enable_backup": True,
            "backup_frequency": "daily"
        }
    }
    
    # Development configuration template
    development_config = {
        "environment": "development",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "description": "Development configuration for comprehensive training pipeline",
        
        # Basic settings
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "data_dir": "./data",
        "output_dir": "./output",
        "model_dir": "./models",
        "log_dir": "./logs",
        
        # Performance settings
        "enable_gpu": False,
        "enable_parallel": True,
        "max_workers": 4,
        "memory_limit": 0.6,
        "timeout_seconds": 1800,
        "random_state": 42,
        
        # Debug and logging
        "debug_mode": True,
        "verbose_logging": True,
        "log_level": "DEBUG",
        
        # Model training configuration
        "model_training_config": {
            "enable_confidence_metrics": True,
            "enable_calibration_assessment": True,
            "enable_feature_importance": True,
            "enable_cross_validation": True,
            "enable_model_explanations": True,
            "enable_post_training_hpo": False,
            "cv_folds": 3,
            "test_size": 0.2,
            "validation_size": 0.1,
            "early_stopping_patience": 10,
            "max_epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "mse",
            "metrics": ["mae", "mse", "r2"]
        },
        
        # Evaluation configuration
        "evaluation_config": {
            "enable_cross_validation": True,
            "enable_time_series_validation": True,
            "enable_confidence_intervals": True,
            "enable_model_comparison": True,
            "enable_feature_importance_analysis": True,
            "enable_prediction_analysis": True,
            "cv_folds": 3,
            "confidence_level": 0.95,
            "bootstrap_samples": 100,
            "validation_metrics": ["sharpe_ratio", "max_drawdown", "total_return"],
            "backtesting_period": "1m",
            "walk_forward_analysis": False
        },
        
        # Feature engineering configuration
        "feature_engineering_config": {
            "enable_technical_indicators": True,
            "enable_statistical_features": True,
            "enable_lag_features": True,
            "enable_rolling_features": True,
            "enable_interaction_features": False,
            "max_lag": 10,
            "rolling_windows": [5, 10, 20],
            "feature_selection_method": "mutual_info",
            "max_features": 50,
            "feature_importance_threshold": 0.01,
            "correlation_threshold": 0.95,
            "vif_threshold": 10.0
        },
        
        # Data quality configuration
        "data_quality_config": {
            "enable_outlier_detection": True,
            "enable_missing_value_handling": True,
            "enable_data_validation": True,
            "outlier_threshold": 3.0,
            "missing_value_threshold": 0.1,
            "data_quality_threshold": 0.8,
            "enable_concept_drift_detection": False,
            "drift_detection_window": 100,
            "drift_threshold": 0.1
        },
        
        # Performance configuration
        "performance_config": {
            "enable_memory_optimization": True,
            "enable_parallel_processing": True,
            "enable_caching": True,
            "cache_size": 1000,
            "chunk_size": 10000,
            "batch_size": 32,
            "enable_gpu_acceleration": False,
            "gpu_memory_fraction": 0.5,
            "enable_mixed_precision": False,
            "enable_gradient_checkpointing": False
        },
        
        # Monitoring and alerting
        "monitoring_config": {
            "enable_performance_monitoring": True,
            "enable_model_monitoring": False,
            "enable_data_drift_monitoring": False,
            "monitoring_interval": 3600,  # 1 hour
            "alert_thresholds": {
                "accuracy_drop": 0.1,
                "latency_increase": 5.0,
                "memory_usage": 0.8,
                "error_rate": 0.05
            },
            "notification_channels": ["console"]
        },
        
        # Security configuration
        "security_config": {
            "enable_encryption": False,
            "enable_audit_logging": True,
            "enable_access_control": False,
            "data_retention_days": 30,
            "model_versioning": True,
            "enable_backup": False,
            "backup_frequency": "weekly"
        }
    }
    
    # Testing configuration template
    testing_config = {
        "environment": "testing",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "description": "Testing configuration for comprehensive training pipeline",
        
        # Basic settings
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "data_dir": "./test_data",
        "output_dir": "./test_output",
        "model_dir": "./test_models",
        "log_dir": "./test_logs",
        
        # Performance settings
        "enable_gpu": False,
        "enable_parallel": False,
        "max_workers": 1,
        "memory_limit": 0.3,
        "timeout_seconds": 300,
        "random_state": 42,
        
        # Debug and logging
        "debug_mode": True,
        "verbose_logging": True,
        "log_level": "DEBUG",
        
        # Model training configuration
        "model_training_config": {
            "enable_confidence_metrics": False,
            "enable_calibration_assessment": False,
            "enable_feature_importance": False,
            "enable_cross_validation": False,
            "enable_model_explanations": False,
            "enable_post_training_hpo": False,
            "cv_folds": 2,
            "test_size": 0.2,
            "validation_size": 0.1,
            "early_stopping_patience": 3,
            "max_epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "loss_function": "mse",
            "metrics": ["mae"]
        },
        
        # Evaluation configuration
        "evaluation_config": {
            "enable_cross_validation": False,
            "enable_time_series_validation": False,
            "enable_confidence_intervals": False,
            "enable_model_comparison": False,
            "enable_feature_importance_analysis": False,
            "enable_prediction_analysis": False,
            "cv_folds": 2,
            "confidence_level": 0.95,
            "bootstrap_samples": 10,
            "validation_metrics": ["mae"],
            "backtesting_period": "1d",
            "walk_forward_analysis": False
        },
        
        # Feature engineering configuration
        "feature_engineering_config": {
            "enable_technical_indicators": False,
            "enable_statistical_features": False,
            "enable_lag_features": False,
            "enable_rolling_features": False,
            "enable_interaction_features": False,
            "max_lag": 2,
            "rolling_windows": [5],
            "feature_selection_method": "none",
            "max_features": 10,
            "feature_importance_threshold": 0.1,
            "correlation_threshold": 0.9,
            "vif_threshold": 5.0
        },
        
        # Data quality configuration
        "data_quality_config": {
            "enable_outlier_detection": False,
            "enable_missing_value_handling": True,
            "enable_data_validation": False,
            "outlier_threshold": 5.0,
            "missing_value_threshold": 0.5,
            "data_quality_threshold": 0.5,
            "enable_concept_drift_detection": False,
            "drift_detection_window": 10,
            "drift_threshold": 0.5
        },
        
        # Performance configuration
        "performance_config": {
            "enable_memory_optimization": False,
            "enable_parallel_processing": False,
            "enable_caching": False,
            "cache_size": 100,
            "chunk_size": 1000,
            "batch_size": 16,
            "enable_gpu_acceleration": False,
            "gpu_memory_fraction": 0.1,
            "enable_mixed_precision": False,
            "enable_gradient_checkpointing": False
        },
        
        # Monitoring and alerting
        "monitoring_config": {
            "enable_performance_monitoring": False,
            "enable_model_monitoring": False,
            "enable_data_drift_monitoring": False,
            "monitoring_interval": 86400,  # 24 hours
            "alert_thresholds": {
                "accuracy_drop": 0.5,
                "latency_increase": 10.0,
                "memory_usage": 0.9,
                "error_rate": 0.1
            },
            "notification_channels": ["console"]
        },
        
        # Security configuration
        "security_config": {
            "enable_encryption": False,
            "enable_audit_logging": False,
            "enable_access_control": False,
            "data_retention_days": 1,
            "model_versioning": False,
            "enable_backup": False,
            "backup_frequency": "never"
        }
    }
    
    # Save configuration templates
    configs = {
        'production': production_config,
        'development': development_config,
        'testing': testing_config
    }
    
    for env_name, config in configs.items():
        config_file = configs_dir / f'{env_name}_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"✅ Created {env_name} configuration: {config_file}")
    
    # Create configuration loader
    config_loader_content = '''#!/usr/bin/env python3
"""
Configuration Loader

This module provides utilities for loading and managing configuration templates.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(environment: str = 'development') -> Dict[str, Any]:
    """Load configuration for specified environment."""
    config_file = Path(f'configs/{environment}_config.json')
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def get_available_configs() -> list:
    """Get list of available configuration environments."""
    configs_dir = Path('configs')
    if not configs_dir.exists():
        return []
    
    config_files = list(configs_dir.glob('*_config.json'))
    return [f.stem.replace('_config', '') for f in config_files]

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_keys = [
        'environment', 'symbol', 'exchange', 'timeframe',
        'model_training_config', 'evaluation_config',
        'feature_engineering_config', 'data_quality_config',
        'performance_config'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required configuration key: {key}")
            return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    print("Available configurations:", get_available_configs())
    
    for env in ['development', 'testing', 'production']:
        try:
            config = load_config(env)
            print(f"\\n{env.upper()} Configuration:")
            print(f"  Environment: {config['environment']}")
            print(f"  Symbol: {config['symbol']}")
            print(f"  Max Workers: {config['max_workers']}")
            print(f"  Memory Limit: {config['memory_limit']}")
            validate_config(config)
        except FileNotFoundError:
            print(f"❌ Configuration not found for environment: {env}")
'''
    
    config_loader_file = Path('config_loader.py')
    with open(config_loader_file, 'w', encoding='utf-8') as f:
        config_loader_file.write_text(config_loader_content)
    print(f"✅ Created configuration loader: {config_loader_file}")
    
    # Create README for configurations
    readme_content = '''# Production Configuration Templates

This directory contains production-ready configuration templates for the comprehensive training pipeline.

## Available Configurations

### Production (`production_config.json`)
- **Purpose**: Production environment with full features and optimized performance
- **Resources**: High (16 workers, 90% memory, GPU enabled)
- **Features**: All features enabled, comprehensive monitoring, security
- **Use Case**: Live trading, production deployment

### Development (`development_config.json`)
- **Purpose**: Development environment with debugging enabled
- **Resources**: Medium (4 workers, 60% memory, no GPU)
- **Features**: Most features enabled, debugging tools, reduced resource usage
- **Use Case**: Development, testing, experimentation

### Testing (`testing_config.json`)
- **Purpose**: Testing environment with minimal resource usage
- **Resources**: Low (1 worker, 30% memory, no GPU)
- **Features**: Basic features only, fast execution
- **Use Case**: Unit tests, integration tests, CI/CD

## Usage

```python
from config_loader import load_config, validate_config

# Load configuration
config = load_config('production')

# Validate configuration
validate_config(config)

# Use with pipeline
from src.training.steps.comprehensive_training_pipeline import ComprehensiveTrainingPipeline
pipeline = ComprehensiveTrainingPipeline(config)
```

## Configuration Structure

Each configuration includes:

- **Basic Settings**: Symbol, exchange, timeframes, directories
- **Performance Settings**: Workers, memory, GPU, timeouts
- **Model Training**: Training parameters, validation, optimization
- **Evaluation**: Metrics, backtesting, validation methods
- **Feature Engineering**: Feature types, selection methods, thresholds
- **Data Quality**: Validation, cleaning, drift detection
- **Performance**: Optimization, caching, parallel processing
- **Monitoring**: Performance monitoring, alerting, notifications
- **Security**: Encryption, audit logging, access control

## Customization

To create a custom configuration:

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Validate the configuration
4. Use with the pipeline

## Best Practices

- **Production**: Use production config for live trading
- **Development**: Use development config for experimentation
- **Testing**: Use testing config for automated tests
- **Validation**: Always validate configurations before use
- **Monitoring**: Enable monitoring in production environments
- **Security**: Enable security features in production
'''
    
    readme_file = configs_dir / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        readme_file.write_text(readme_content)
    print(f"✅ Created configuration README: {readme_file}")
    
    print(f"\\n🎉 Production configuration templates created successfully!")
    print(f"📁 Configuration files saved in: {configs_dir}")
    print(f"📋 Available environments: {list(configs.keys())}")

if __name__ == "__main__":
    create_production_config_templates()