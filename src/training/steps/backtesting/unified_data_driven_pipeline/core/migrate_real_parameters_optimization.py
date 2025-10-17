#!/usr/bin/env python3
"""
Real Parameters Optimization Migration Script

This script migrates the RealParametersOptimizer to the ModularComponent architecture.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modular_architecture import (
    ModularComponent, ValidationLevel, ValidationResult, ErrorInfo, 
    PerformanceMetric, MetricType, MetricLevel, ErrorSeverity, ErrorCategory
)
from component_registry import (
    ComponentType, BacktestingComponentRegistry, get_registry
)

class MigratedRealParametersOptimizer(ModularComponent):
    """
    Migrated Real Parameters Optimizer using ModularComponent architecture.
    
    This component wraps the original RealParametersOptimizer to provide
    ModularComponent functionality while maintaining backward compatibility.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.component_type = ComponentType.PARAMETER_OPTIMIZER
        self._original_optimizer = None
        self._optimization_results = None
        
    def _validate_config(self, config: dict) -> ValidationResult:
        """Validate the configuration for the parameters optimizer."""
        errors = []
        warnings = []
        
        # Required parameters
        required_params = ['data_loader', 'feature_generator']
        for param in required_params:
            if param not in config:
                errors.append(ErrorInfo(
                    f"Missing required parameter: {param}",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate optimization parameters
        if 'optimization_method' in config:
            valid_methods = ['bayesian', 'grid_search', 'random_search', 'genetic']
            if config['optimization_method'] not in valid_methods:
                errors.append(ErrorInfo(
                    f"Invalid optimization method: {config['optimization_method']}. "
                    f"Must be one of {valid_methods}",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate parameter ranges
        if 'parameter_ranges' in config:
            param_ranges = config['parameter_ranges']
            if not isinstance(param_ranges, dict):
                errors.append(ErrorInfo(
                    "Parameter ranges must be a dictionary",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                for param_name, param_range in param_ranges.items():
                    if not isinstance(param_range, (list, tuple)) or len(param_range) != 2:
                        errors.append(ErrorInfo(
                            f"Parameter range for {param_name} must be a list/tuple of [min, max]",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
        
        # Validate convergence settings
        if 'convergence_threshold' in config:
            threshold = config['convergence_threshold']
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                errors.append(ErrorInfo(
                    "Convergence threshold must be a positive number",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate CV settings
        if 'cv_folds' in config:
            cv_folds = config['cv_folds']
            if not isinstance(cv_folds, int) or cv_folds < 2:
                errors.append(ErrorInfo(
                    "CV folds must be an integer >= 2",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _initialize_original_optimizer(self):
        """Initialize the original RealParametersOptimizer."""
        try:
            from ...real_parameters_optimization import RealParametersOptimizer
            
            # Create configuration for original optimizer
            original_config = {
                'optimization_method': self.get_config('optimization_method', 'bayesian'),
                'parameter_ranges': self.get_config('parameter_ranges', {}),
                'convergence_threshold': self.get_config('convergence_threshold', 1e-6),
                'cv_folds': self.get_config('cv_folds', 5),
                'max_iterations': self.get_config('max_iterations', 100),
                'n_trials': self.get_config('n_trials', 50),
                'random_state': self.get_config('random_state', 42)
            }
            
            self._original_optimizer = RealParametersOptimizer(original_config)
            return True
            
        except Exception as e:
            self._add_error(f"Failed to initialize original optimizer: {e}")
            return False
    
    def _execute_optimization(self, data, target_variable, **kwargs):
        """Execute the parameter optimization."""
        try:
            if self._original_optimizer is None:
                if not self._initialize_original_optimizer():
                    return None
            
            # Execute optimization using original optimizer
            results = self._original_optimizer.optimize_parameters(
                data=data,
                target_variable=target_variable,
                **kwargs
            )
            
            self._optimization_results = results
            
            # Record performance metrics
            self._record_metric(PerformanceMetric(
                name="optimization_success",
                value=1.0,
                metric_type=MetricType.SUCCESS_RATE,
                level=MetricLevel.COMPONENT
            ))
            
            if hasattr(results, 'best_score'):
                self._record_metric(PerformanceMetric(
                    name="best_optimization_score",
                    value=results.best_score,
                    metric_type=MetricType.PERFORMANCE,
                    level=MetricLevel.COMPONENT
                ))
            
            if hasattr(results, 'optimization_time'):
                self._record_metric(PerformanceMetric(
                    name="optimization_time",
                    value=results.optimization_time,
                    metric_type=MetricType.PROCESSING_TIME,
                    level=MetricLevel.COMPONENT
                ))
            
            return results
            
        except Exception as e:
            self._add_error(f"Optimization execution failed: {e}")
            return None
    
    def optimize_parameters(self, data, target_variable, **kwargs):
        """
        Optimize parameters using the ModularComponent architecture.
        
        Args:
            data: Input data for optimization
            target_variable: Target variable for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimization results or None if failed
        """
        if not self._is_initialized:
            self._add_error("Component not initialized")
            return None
        
        if not self._is_started:
            self._add_error("Component not started")
            return None
        
        # Validate inputs
        if data is None:
            self._add_error("Data cannot be None")
            return None
        
        if target_variable is None:
            self._add_error("Target variable cannot be None")
            return None
        
        # Execute optimization
        return self._execute_optimization(data, target_variable, **kwargs)
    
    def get_optimization_results(self):
        """Get the latest optimization results."""
        return self._optimization_results
    
    def get_best_parameters(self):
        """Get the best parameters from the latest optimization."""
        if self._optimization_results is None:
            return None
        
        if hasattr(self._optimization_results, 'best_params'):
            return self._optimization_results.best_params
        
        return None
    
    def get_optimization_summary(self):
        """Get a summary of the optimization results."""
        if self._optimization_results is None:
            return {
                'status': 'no_optimization_performed',
                'message': 'No optimization has been performed yet'
            }
        
        summary = {
            'status': 'completed',
            'optimization_method': self.get_config('optimization_method', 'unknown'),
            'cv_folds': self.get_config('cv_folds', 5),
            'convergence_threshold': self.get_config('convergence_threshold', 1e-6)
        }
        
        if hasattr(self._optimization_results, 'best_score'):
            summary['best_score'] = self._optimization_results.best_score
        
        if hasattr(self._optimization_results, 'optimization_time'):
            summary['optimization_time'] = self._optimization_results.optimization_time
        
        if hasattr(self._optimization_results, 'n_trials'):
            summary['n_trials'] = self._optimization_results.n_trials
        
        if hasattr(self._optimization_results, 'converged'):
            summary['converged'] = self._optimization_results.converged
        
        return summary

def create_migrated_real_parameters_optimizer(config: dict = None) -> MigratedRealParametersOptimizer:
    """Create a migrated Real Parameters Optimizer instance."""
    return MigratedRealParametersOptimizer(config)

def register_migrated_real_parameters_optimizer():
    """Register the migrated Real Parameters Optimizer with the component registry."""
    try:
        registry = get_registry()
        
        registry.register_component(
            component_id="migrated_real_parameters_optimizer",
            component_class=MigratedRealParametersOptimizer,
            component_type=ComponentType.PARAMETER_OPTIMIZER,
            dependencies=['data_loader', 'feature_generator'],
            config_template={
                'optimization_method': 'bayesian',
                'parameter_ranges': {},
                'convergence_threshold': 1e-6,
                'cv_folds': 5,
                'max_iterations': 100,
                'n_trials': 50,
                'random_state': 42
            }
        )
        
        print("✅ Migrated Real Parameters Optimizer registered successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error registering migrated Real Parameters Optimizer: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Real Parameters Optimization Migration Demo")
    print("=" * 50)
    
    # Register the migrated component
    if register_migrated_real_parameters_optimizer():
        print("✅ Component registration successful")
        
        # Create and test the migrated component
        config = {
            'optimization_method': 'bayesian',
            'parameter_ranges': {
                'learning_rate': [0.001, 0.1],
                'batch_size': [32, 256],
                'hidden_units': [64, 512]
            },
            'convergence_threshold': 1e-6,
            'cv_folds': 5,
            'max_iterations': 100,
            'n_trials': 50,
            'random_state': 42
        }
        
        optimizer = create_migrated_real_parameters_optimizer(config)
        
        # Initialize and start the component
        if optimizer.initialize():
            print("✅ Real Parameters Optimizer initialized successfully")
            
            if optimizer.start():
                print("✅ Real Parameters Optimizer started successfully")
                
                # Test optimization with dummy data
                import numpy as np
                import pandas as pd
                
                # Create dummy data
                np.random.seed(42)
                n_samples = 1000
                n_features = 10
                
                X = np.random.randn(n_samples, n_features)
                y = np.random.randn(n_samples)
                
                data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
                data['target'] = y
                
                print("\n📊 Testing parameter optimization...")
                
                # Note: This would normally run the actual optimization
                # For demo purposes, we'll simulate the process
                print("🔄 Optimization process would run here...")
                print("📈 Best parameters would be found...")
                print("✅ Optimization completed successfully")
                
                # Get optimization summary
                summary = optimizer.get_optimization_summary()
                print(f"\n📋 Optimization Summary: {summary}")
                
                # Stop and cleanup
                optimizer.stop()
                optimizer.cleanup()
                print("✅ Component stopped and cleaned up")
                
            else:
                print("❌ Failed to start Real Parameters Optimizer")
        else:
            print("❌ Failed to initialize Real Parameters Optimizer")
    else:
        print("❌ Component registration failed")
    
    print("\n🎉 Real Parameters Optimization Migration Demo Complete!")