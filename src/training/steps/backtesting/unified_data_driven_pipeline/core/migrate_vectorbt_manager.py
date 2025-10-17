"""
Migration Example: VectorBT Unified Manager to ModularComponent

This script demonstrates how to migrate the existing VectorBTUnifiedManager
to use the ModularComponent architecture.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.backtesting.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent,
    create_backtesting_component,
    ValidationLevel,
    ErrorInfo,
    ErrorSeverity,
    ErrorCategory
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.component_registry import (
    get_registry,
    register_component,
    ComponentType
)

class MigratedVectorBTManager(ModularComponent):
    """
    Migrated VectorBT Unified Manager using ModularComponent architecture.
    
    This is a migrated version of the VectorBTUnifiedManager that inherits
    from ModularComponent and provides all the backtesting-specific features.
    """
    
    def __init__(self, config: dict = None, logger=None):
        super().__init__(
            name="vectorbt_manager",
            config=config or {},
            logger=logger
        )
        
        # VectorBT specific configuration
        self._vectorbt_config = self.config.get('vectorbt', {})
        self._optimization_config = self.config.get('optimization', {})
        self._performance_config = self.config.get('performance', {})
        
        # VectorBT operation tracking
        self._operation_history = []
        self._performance_metrics = {}
        self._optimization_results = {}
    
    def _initialize_resources(self) -> bool:
        """Initialize VectorBT manager resources."""
        try:
            # Initialize VectorBT configuration
            self._vectorbt_config = {
                'enable_gpu': self.config.get('vectorbt', {}).get('enable_gpu', False),
                'enable_parallel': self.config.get('vectorbt', {}).get('enable_parallel', True),
                'memory_limit': self.config.get('vectorbt', {}).get('memory_limit', '2GB'),
                'chunk_size': self.config.get('vectorbt', {}).get('chunk_size', 1000)
            }
            
            # Initialize optimization configuration
            self._optimization_config = {
                'enable_optimization': self.config.get('optimization', {}).get('enable_optimization', True),
                'max_workers': self.config.get('optimization', {}).get('max_workers', 4),
                'optimization_method': self.config.get('optimization', {}).get('method', 'grid_search')
            }
            
            # Initialize performance tracking
            self._performance_metrics = {
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'avg_processing_time': 0.0,
                'memory_usage': 0.0,
                'gpu_utilization': 0.0
            }
            
            self.logger.info("VectorBT Manager resources initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorBT Manager resources: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup VectorBT manager resources."""
        try:
            # Clear operation history
            self._operation_history = []
            self._performance_metrics = {}
            self._optimization_results = {}
            
            # Clear any cached data
            if hasattr(self, '_cached_results'):
                delattr(self, '_cached_results')
            
            self.logger.info("VectorBT Manager resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during VectorBT Manager cleanup: {e}")
    
    def _process_data(self, data: any, **kwargs) -> any:
        """Process data using VectorBT operations."""
        try:
            # Validate input data
            if not self._validate_vectorbt_data(data):
                raise ValueError("Invalid VectorBT data provided")
            
            # Extract operation type and parameters
            operation_type = data.get('operation_type', 'rolling_statistics')
            operation_params = data.get('operation_params', {})
            market_data = data.get('market_data', {})
            
            # Execute VectorBT operation
            result = self._execute_vectorbt_operation(
                operation_type, 
                operation_params, 
                market_data
            )
            
            # Update performance metrics
            self._update_performance_metrics(operation_type, result)
            
            # Return results
            return {
                'operation_type': operation_type,
                'result': result,
                'performance_metrics': self._performance_metrics.copy(),
                'operation_history': self._operation_history[-10:]  # Last 10 operations
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data in VectorBT Manager: {e}")
            raise
    
    def _validate_vectorbt_data(self, data: any) -> bool:
        """Validate data for VectorBT operations."""
        if not isinstance(data, dict):
            return False
        
        # Check for required fields
        if 'operation_type' not in data:
            return False
        
        # Validate operation type
        valid_operations = [
            'rolling_statistics', 'correlation_analysis', 'risk_metrics',
            'performance_analysis', 'regime_analysis', 'bootstrap_sampling',
            'feature_selection', 'parameter_optimization'
        ]
        
        if data['operation_type'] not in valid_operations:
            return False
        
        # Validate market data if provided
        if 'market_data' in data:
            market_data = data['market_data']
            if not isinstance(market_data, dict):
                return False
        
        return True
    
    def _execute_vectorbt_operation(self, operation_type: str, params: dict, market_data: dict) -> dict:
        """Execute a VectorBT operation."""
        import time
        import numpy as np
        
        start_time = time.time()
        
        try:
            if operation_type == 'rolling_statistics':
                result = self._execute_rolling_statistics(params, market_data)
            elif operation_type == 'correlation_analysis':
                result = self._execute_correlation_analysis(params, market_data)
            elif operation_type == 'risk_metrics':
                result = self._execute_risk_metrics(params, market_data)
            elif operation_type == 'performance_analysis':
                result = self._execute_performance_analysis(params, market_data)
            elif operation_type == 'regime_analysis':
                result = self._execute_regime_analysis(params, market_data)
            elif operation_type == 'bootstrap_sampling':
                result = self._execute_bootstrap_sampling(params, market_data)
            elif operation_type == 'feature_selection':
                result = self._execute_feature_selection(params, market_data)
            elif operation_type == 'parameter_optimization':
                result = self._execute_parameter_optimization(params, market_data)
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            # Record operation
            operation_record = {
                'operation_type': operation_type,
                'timestamp': time.time(),
                'processing_time': time.time() - start_time,
                'success': True,
                'params': params
            }
            
            self._operation_history.append(operation_record)
            
            return result
            
        except Exception as e:
            # Record failed operation
            operation_record = {
                'operation_type': operation_type,
                'timestamp': time.time(),
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'params': params
            }
            
            self._operation_history.append(operation_record)
            raise
    
    def _execute_rolling_statistics(self, params: dict, market_data: dict) -> dict:
        """Execute rolling statistics operation."""
        # This is a simplified implementation
        # In a real implementation, you would use actual VectorBT operations
        
        window_size = params.get('window_size', 20)
        statistics = params.get('statistics', ['mean', 'std', 'min', 'max'])
        
        # Simulate rolling statistics calculation
        result = {
            'window_size': window_size,
            'statistics': statistics,
            'calculated': True,
            'data_points': len(market_data.get('prices', [])) if 'prices' in market_data else 0
        }
        
        return result
    
    def _execute_correlation_analysis(self, params: dict, market_data: dict) -> dict:
        """Execute correlation analysis operation."""
        # Simulate correlation analysis
        result = {
            'correlation_matrix': {},
            'analysis_type': 'pearson',
            'calculated': True
        }
        
        return result
    
    def _execute_risk_metrics(self, params: dict, market_data: dict) -> dict:
        """Execute risk metrics calculation."""
        # Simulate risk metrics calculation
        result = {
            'var_95': 0.05,
            'var_99': 0.01,
            'expected_shortfall': 0.03,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15
        }
        
        return result
    
    def _execute_performance_analysis(self, params: dict, market_data: dict) -> dict:
        """Execute performance analysis."""
        # Simulate performance analysis
        result = {
            'total_return': 0.15,
            'annualized_return': 0.12,
            'volatility': 0.18,
            'sharpe_ratio': 0.67,
            'sortino_ratio': 0.89
        }
        
        return result
    
    def _execute_regime_analysis(self, params: dict, market_data: dict) -> dict:
        """Execute regime analysis."""
        # Simulate regime analysis
        result = {
            'regimes_detected': 3,
            'regime_probabilities': [0.4, 0.35, 0.25],
            'regime_characteristics': {}
        }
        
        return result
    
    def _execute_bootstrap_sampling(self, params: dict, market_data: dict) -> dict:
        """Execute bootstrap sampling."""
        # Simulate bootstrap sampling
        n_samples = params.get('n_samples', 1000)
        result = {
            'n_samples': n_samples,
            'bootstrap_results': [],
            'confidence_intervals': {}
        }
        
        return result
    
    def _execute_feature_selection(self, params: dict, market_data: dict) -> dict:
        """Execute feature selection."""
        # Simulate feature selection
        result = {
            'selected_features': [],
            'feature_importance': {},
            'selection_method': params.get('method', 'correlation')
        }
        
        return result
    
    def _execute_parameter_optimization(self, params: dict, market_data: dict) -> dict:
        """Execute parameter optimization."""
        # Simulate parameter optimization
        result = {
            'optimized_parameters': {},
            'optimization_score': 0.85,
            'optimization_method': params.get('method', 'grid_search')
        }
        
        return result
    
    def _update_performance_metrics(self, operation_type: str, result: dict) -> None:
        """Update performance metrics."""
        self._performance_metrics['total_operations'] += 1
        
        if result.get('calculated', False):
            self._performance_metrics['successful_operations'] += 1
        else:
            self._performance_metrics['failed_operations'] += 1
        
        # Calculate success rate
        total = self._performance_metrics['total_operations']
        successful = self._performance_metrics['successful_operations']
        self._performance_metrics['success_rate'] = successful / total if total > 0 else 0
        
        # Update average processing time
        if self._operation_history:
            recent_operations = self._operation_history[-10:]  # Last 10 operations
            avg_time = sum(op['processing_time'] for op in recent_operations) / len(recent_operations)
            self._performance_metrics['avg_processing_time'] = avg_time
    
    def _get_validation_rules(self) -> dict:
        """Get validation rules for VectorBT operations."""
        return {
            'operation_data': {
                'required_fields': ['operation_type'],
                'data_types': {
                    'operation_type': str,
                    'operation_params': dict,
                    'market_data': dict
                }
            },
            'vectorbt_config': {
                'enable_gpu': {
                    'type': bool
                },
                'enable_parallel': {
                    'type': bool
                },
                'memory_limit': {
                    'type': str
                }
            }
        }
    
    def _validate_component_specific(self, data: any) -> bool:
        """Validate component-specific data."""
        if not isinstance(data, dict):
            return False
        
        # Check for operation type
        if 'operation_type' not in data:
            return False
        
        # Validate operation type
        valid_operations = [
            'rolling_statistics', 'correlation_analysis', 'risk_metrics',
            'performance_analysis', 'regime_analysis', 'bootstrap_sampling',
            'feature_selection', 'parameter_optimization'
        ]
        
        if data['operation_type'] not in valid_operations:
            return False
        
        return True
    
    def get_operation_history(self, limit: int = 10) -> list:
        """Get operation history."""
        return self._operation_history[-limit:] if limit else self._operation_history
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        return self._performance_metrics.copy()
    
    def get_vectorbt_config(self) -> dict:
        """Get current VectorBT configuration."""
        return self._vectorbt_config.copy()
    
    def update_vectorbt_config(self, config: dict) -> bool:
        """Update VectorBT configuration."""
        try:
            # Validate new configuration
            if 'enable_gpu' in config and not isinstance(config['enable_gpu'], bool):
                return False
            
            # Update configuration
            self._vectorbt_config.update(config)
            self.config['vectorbt'].update(config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating VectorBT config: {e}")
            return False


def create_migrated_vectorbt_manager(config: dict = None) -> MigratedVectorBTManager:
    """Create a migrated VectorBT manager instance."""
    return MigratedVectorBTManager(config)


def register_migrated_vectorbt_manager():
    """Register the migrated VectorBT manager in the component registry."""
    registry = get_registry()
    
    # Register the component
    registry.register_component(
        name='vectorbt_manager',
        component_type=ComponentType.VECTORBT_MANAGER,
        component_class=MigratedVectorBTManager,
        dependencies=['data_loader'],
        metadata={
            'migrated': True,
            'original_file': 'src/training/steps/backtesting/vectorbt_unified_manager.py',
            'migration_strategy': 'direct',
            'migration_timestamp': time.time()
        }
    )
    
    print("Migrated VectorBT Manager registered successfully")


if __name__ == '__main__':
    # Example usage
    import time
    
    # Create configuration
    config = {
        'vectorbt': {
            'enable_gpu': False,
            'enable_parallel': True,
            'memory_limit': '2GB',
            'chunk_size': 1000
        },
        'optimization': {
            'enable_optimization': True,
            'max_workers': 4,
            'method': 'grid_search'
        }
    }
    
    # Create migrated component
    manager = create_migrated_vectorbt_manager(config)
    
    # Initialize
    if manager.initialize():
        print("VectorBT Manager initialized successfully")
        
        # Example data
        sample_data = {
            'operation_type': 'rolling_statistics',
            'operation_params': {
                'window_size': 20,
                'statistics': ['mean', 'std', 'min', 'max']
            },
            'market_data': {
                'prices': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
            }
        }
        
        # Process data
        result = manager.process(sample_data)
        print(f"Operation completed: {result['operation_type']}")
        print(f"Performance metrics: {result['performance_metrics']}")
        
        # Cleanup
        manager.cleanup()
        print("VectorBT Manager cleaned up")
    
    # Register in registry
    register_migrated_vectorbt_manager()