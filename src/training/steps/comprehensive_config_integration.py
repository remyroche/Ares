"""
Comprehensive Configuration Integration

This module provides complete configuration integration for the comprehensive training pipeline,
ensuring all configuration parameters are properly validated, integrated, and used throughout the pipeline.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Import configuration validation
from .standardized_config_validation import (
    validate_config,
    validate_and_fix_config,
    get_default_config
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class ComprehensiveConfigIntegration:
    """
    Comprehensive Configuration Integration for the training pipeline.
    
    This class provides complete configuration management including:
    - Configuration validation and fixing
    - Default configuration generation
    - Configuration merging and inheritance
    - Environment-specific configurations
    - Configuration documentation and examples
    """
    
    def __init__(self):
        """Initialize comprehensive configuration integration."""
        self.logger = logger.getChild('ComprehensiveConfigIntegration')
        self.config_templates = self._load_config_templates()
        self.logger.info("🔧 Comprehensive Configuration Integration initialized")
    
    def _load_config_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration templates for different scenarios."""
        return {
            'development': self._get_development_config(),
            'testing': self._get_testing_config(),
            'production': self._get_production_config(),
            'minimal': self._get_minimal_config(),
            'comprehensive': self._get_comprehensive_config()
        }
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Get development configuration template."""
        return {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'data_dir': 'data',
            'output_dir': 'output',
            'model_dir': 'models',
            'log_dir': 'logs',
            'enable_gpu': False,
            'enable_parallel': True,
            'max_workers': 2,
            'memory_limit': 0.6,
            'timeout_seconds': 1800,
            'random_state': 42,
            'debug_mode': True,
            'verbose_logging': True,
            
            # Model training configuration
            'model_training_config': {
                'enable_confidence_metrics': True,
                'enable_calibration_assessment': True,
                'enable_feature_importance': True,
                'enable_cross_validation': True,
                'enable_model_explanations': True,
                'enable_post_training_hpo': False,
                'cv_folds': 3,
                'test_size': 0.2,
                'validation_size': 0.1,
                'early_stopping_patience': 10,
                'max_epochs': 100
            },
            
            # Evaluation configuration
            'evaluation_config': {
                'enable_cross_validation': True,
                'enable_time_series_validation': True,
                'enable_confidence_intervals': True,
                'enable_model_comparison': True,
                'enable_feature_importance_analysis': True,
                'enable_prediction_analysis': True,
                'cv_folds': 3,
                'confidence_level': 0.95,
                'bootstrap_samples': 100
            },
            
            # Feature engineering configuration
            'feature_engineering_config': {
                'enable_technical_indicators': True,
                'enable_statistical_features': True,
                'enable_lag_features': True,
                'enable_rolling_features': True,
                'enable_interaction_features': False,
                'max_lag': 10,
                'rolling_windows': [5, 10, 20],
                'feature_selection_method': 'mutual_info'
            },
            
            # Data quality configuration
            'data_quality_config': {
                'enable_outlier_detection': True,
                'enable_missing_value_handling': True,
                'enable_data_validation': True,
                'outlier_threshold': 3.0,
                'missing_value_threshold': 0.1,
                'data_quality_threshold': 0.8
            },
            
            # Performance configuration
            'performance_config': {
                'enable_memory_optimization': True,
                'enable_parallel_processing': True,
                'enable_caching': True,
                'cache_size': 1000,
                'chunk_size': 10000,
                'batch_size': 32
            }
        }
    
    def _get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration template."""
        config = self._get_development_config()
        config.update({
            'debug_mode': True,
            'verbose_logging': True,
            'timeout_seconds': 300,
            'max_workers': 1,
            'memory_limit': 0.3,
            'model_training_config': {
                **config['model_training_config'],
                'cv_folds': 2,
                'max_epochs': 10,
                'early_stopping_patience': 3
            },
            'evaluation_config': {
                **config['evaluation_config'],
                'cv_folds': 2,
                'bootstrap_samples': 10
            }
        })
        return config
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Get production configuration template."""
        config = self._get_development_config()
        config.update({
            'debug_mode': False,
            'verbose_logging': False,
            'timeout_seconds': 7200,
            'max_workers': 8,
            'memory_limit': 0.9,
            'model_training_config': {
                **config['model_training_config'],
                'cv_folds': 5,
                'max_epochs': 500,
                'early_stopping_patience': 20,
                'enable_post_training_hpo': True
            },
            'evaluation_config': {
                **config['evaluation_config'],
                'cv_folds': 5,
                'bootstrap_samples': 1000
            },
            'performance_config': {
                **config['performance_config'],
                'enable_memory_optimization': True,
                'enable_parallel_processing': True,
                'cache_size': 10000,
                'chunk_size': 50000,
                'batch_size': 128
            }
        })
        return config
    
    def _get_minimal_config(self) -> Dict[str, Any]:
        """Get minimal configuration template."""
        return {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'data_dir': 'data',
            'output_dir': 'output',
            'model_dir': 'models',
            'log_dir': 'logs',
            'enable_gpu': False,
            'enable_parallel': False,
            'max_workers': 1,
            'memory_limit': 0.5,
            'timeout_seconds': 600,
            'random_state': 42,
            'debug_mode': False,
            'verbose_logging': False,
            
            'model_training_config': {
                'enable_confidence_metrics': False,
                'enable_calibration_assessment': False,
                'enable_feature_importance': False,
                'enable_cross_validation': False,
                'enable_model_explanations': False,
                'enable_post_training_hpo': False,
                'cv_folds': 2,
                'test_size': 0.2,
                'validation_size': 0.1,
                'early_stopping_patience': 5,
                'max_epochs': 50
            },
            
            'evaluation_config': {
                'enable_cross_validation': False,
                'enable_time_series_validation': False,
                'enable_confidence_intervals': False,
                'enable_model_comparison': False,
                'enable_feature_importance_analysis': False,
                'enable_prediction_analysis': False,
                'cv_folds': 2,
                'confidence_level': 0.95,
                'bootstrap_samples': 10
            }
        }
    
    def _get_comprehensive_config(self) -> Dict[str, Any]:
        """Get comprehensive configuration template."""
        config = self._get_production_config()
        config.update({
            'model_training_config': {
                **config['model_training_config'],
                'enable_confidence_metrics': True,
                'enable_calibration_assessment': True,
                'enable_feature_importance': True,
                'enable_cross_validation': True,
                'enable_model_explanations': True,
                'enable_post_training_hpo': True,
                'cv_folds': 10,
                'max_epochs': 1000,
                'early_stopping_patience': 50
            },
            
            'evaluation_config': {
                **config['evaluation_config'],
                'enable_cross_validation': True,
                'enable_time_series_validation': True,
                'enable_confidence_intervals': True,
                'enable_model_comparison': True,
                'enable_feature_importance_analysis': True,
                'enable_prediction_analysis': True,
                'cv_folds': 10,
                'bootstrap_samples': 2000
            },
            
            'feature_engineering_config': {
                'enable_technical_indicators': True,
                'enable_statistical_features': True,
                'enable_lag_features': True,
                'enable_rolling_features': True,
                'enable_interaction_features': True,
                'max_lag': 20,
                'rolling_windows': [5, 10, 20, 50, 100],
                'feature_selection_method': 'recursive_feature_elimination'
            }
        })
        return config
    
    def get_config_template(self, template_name: str) -> Dict[str, Any]:
        """Get a specific configuration template."""
        if template_name not in self.config_templates:
            available_templates = list(self.config_templates.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available templates: {available_templates}")
        
        return self.config_templates[template_name].copy()
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations with override taking precedence."""
        merged_config = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key] = self.merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
        
        return merged_config
    
    def validate_and_integrate_config(self, config: Dict[str, Any], template_name: str = 'development') -> Dict[str, Any]:
        """Validate and integrate configuration with template."""
        try:
            self.logger.info(f"🔧 Validating and integrating configuration with template '{template_name}'")
            
            # Get base template
            base_config = self.get_config_template(template_name)
            
            # Merge with provided config
            merged_config = self.merge_configs(base_config, config)
            
            # Validate the merged configuration
            validated_config = validate_and_fix_config(merged_config, 'comprehensive_training')
            
            # Add integration metadata
            validated_config['_integration_metadata'] = {
                'template_used': template_name,
                'validation_timestamp': datetime.now().isoformat(),
                'validation_status': 'passed',
                'config_version': '1.0.0'
            }
            
            self.logger.info("✅ Configuration validated and integrated successfully")
            return validated_config
            
        except Exception as e:
            self.logger.exception(f"Configuration validation and integration failed: {e}")
            raise
    
    def create_environment_config(self, environment: str, custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create environment-specific configuration."""
        try:
            self.logger.info(f"🔧 Creating configuration for environment: {environment}")
            
            # Get base template for environment
            if environment in self.config_templates:
                base_config = self.get_config_template(environment)
            else:
                self.logger.warning(f"Unknown environment '{environment}', using development template")
                base_config = self.get_config_template('development')
            
            # Apply custom overrides if provided
            if custom_overrides:
                base_config = self.merge_configs(base_config, custom_overrides)
            
            # Validate configuration
            validated_config = validate_and_fix_config(base_config, 'comprehensive_training')
            
            # Add environment metadata
            validated_config['_environment_metadata'] = {
                'environment': environment,
                'creation_timestamp': datetime.now().isoformat(),
                'custom_overrides_applied': custom_overrides is not None
            }
            
            self.logger.info(f"✅ Environment configuration created for: {environment}")
            return validated_config
            
        except Exception as e:
            self.logger.exception(f"Environment configuration creation failed: {e}")
            raise
    
    def get_config_documentation(self) -> Dict[str, Any]:
        """Get comprehensive configuration documentation."""
        return {
            'overview': 'Comprehensive Training Pipeline Configuration',
            'version': '1.0.0',
            'templates_available': list(self.config_templates.keys()),
            'template_descriptions': {
                'development': 'Development environment with debugging enabled and reduced resource usage',
                'testing': 'Testing environment with minimal resource usage and fast execution',
                'production': 'Production environment with full features and optimized performance',
                'minimal': 'Minimal configuration with basic features only',
                'comprehensive': 'Comprehensive configuration with all features enabled'
            },
            'configuration_sections': {
                'model_training_config': 'Model training parameters and options',
                'evaluation_config': 'Model evaluation parameters and metrics',
                'feature_engineering_config': 'Feature engineering parameters and options',
                'data_quality_config': 'Data quality validation and cleaning parameters',
                'performance_config': 'Performance optimization parameters'
            },
            'usage_examples': {
                'development': 'config = config_integration.create_environment_config("development")',
                'production': 'config = config_integration.create_environment_config("production", {"max_workers": 16})',
                'custom': 'config = config_integration.validate_and_integrate_config(custom_config, "development")'
            }
        }
    
    def save_config_template(self, template_name: str, config: Dict[str, Any], file_path: str):
        """Save configuration template to file."""
        try:
            import json
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info(f"✅ Configuration template '{template_name}' saved to {file_path}")
            
        except Exception as e:
            self.logger.exception(f"Failed to save configuration template: {e}")
            raise
    
    def load_config_template(self, file_path: str) -> Dict[str, Any]:
        """Load configuration template from file."""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"✅ Configuration template loaded from {file_path}")
            return config
            
        except Exception as e:
            self.logger.exception(f"Failed to load configuration template: {e}")
            raise


# Global instance for easy access
config_integration = ComprehensiveConfigIntegration()


# Convenience functions
def get_development_config() -> Dict[str, Any]:
    """Get development configuration."""
    return config_integration.get_config_template('development')


def get_production_config() -> Dict[str, Any]:
    """Get production configuration."""
    return config_integration.get_config_template('production')


def get_testing_config() -> Dict[str, Any]:
    """Get testing configuration."""
    return config_integration.get_config_template('testing')


def create_custom_config(template_name: str = 'development', overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create custom configuration with overrides."""
    return config_integration.create_environment_config(template_name, overrides)


def validate_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate pipeline configuration."""
    return config_integration.validate_and_integrate_config(config)


# Example usage
if __name__ == "__main__":
    # Example: Create different configurations
    print("🔧 Configuration Integration Examples")
    print("=" * 50)
    
    # Development config
    dev_config = get_development_config()
    print(f"Development config: {dev_config['symbol']} on {dev_config['exchange']}")
    
    # Production config
    prod_config = get_production_config()
    print(f"Production config: {prod_config['max_workers']} workers, {prod_config['memory_limit']} memory limit")
    
    # Custom config with overrides
    custom_config = create_custom_config('development', {'max_workers': 4, 'symbol': 'ETHUSDT'})
    print(f"Custom config: {custom_config['symbol']} with {custom_config['max_workers']} workers")
    
    # Get documentation
    docs = config_integration.get_config_documentation()
    print(f"Available templates: {docs['templates_available']}")
    
    print("✅ Configuration integration examples completed")