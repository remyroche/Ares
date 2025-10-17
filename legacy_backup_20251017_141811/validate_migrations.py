"""
Validation Script for Migrated Models Training Components

This script validates that all migrated components work correctly with the
ModularComponent architecture. It performs comprehensive testing including:
- Component initialization and cleanup
- Data processing and validation
- Performance monitoring
- Health status checking
- Integration testing

Usage:
    python validate_migrations.py --components all
    python validate_migrations.py --components analyst_models,analyst_ensemble
    python validate_migrations.py --mode quick
    python validate_migrations.py --mode comprehensive
"""

import argparse
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import migrated components
from .components.analyst_models_training_modular import (
    AnalystModelsTrainingModular, create_analyst_models_training
)
from .components.analyst_ensemble_training_modular import (
    AnalystEnsembleTrainingModular, create_analyst_ensemble_training
)
from .components.ml_entry_timing_labeler_modular import (
    MLEntryTimingLabelerModular, create_ml_entry_timing_labeler
)
from .unified_training_pipeline_modular import (
    UnifiedTrainingPipelineModular, create_unified_training_pipeline
)


class MigrationValidator:
    """Validator for migrated components."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results = {}
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> Dict[str, Any]:
        """Create comprehensive test data for validation."""
        # Create synthetic training data
        n_samples = 1000
        n_features = 10
        
        # Analyst data
        analyst_data = {
            'X_train': pd.DataFrame({
                f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
            }),
            'y_train': np.random.randint(0, 2, n_samples),
            'X_val': pd.DataFrame({
                f'feature_{i}': np.random.randn(200) for i in range(n_features)
            }),
            'y_val': np.random.randint(0, 2, 200),
            'regime_data': np.random.randint(0, 3, n_samples)
        }
        
        # Tactician data
        tactician_data = {
            'X_train': pd.DataFrame({
                f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
            }),
            'y_train': np.random.randint(0, 2, n_samples),
            'X_val': pd.DataFrame({
                f'feature_{i}': np.random.randn(200) for i in range(n_features)
            }),
            'y_val': np.random.randint(0, 2, 200)
        }
        
        # Labeling data
        labeling_data = {
            'features': np.random.randn(n_samples, n_features),
            'market_data': np.random.randn(n_samples, 5)
        }
        
        # Base model outputs for ensemble
        base_model_outputs = {
            'tcn': np.random.randn(n_samples),
            'lightgbm': np.random.randn(n_samples),
            'ridge': np.random.randn(n_samples),
            'elastic_net': np.random.randn(n_samples),
            'random_forest': np.random.randn(n_samples)
        }
        
        return {
            'analyst_data': analyst_data,
            'tactician_data': tactician_data,
            'labeling_data': labeling_data,
            'base_model_outputs': base_model_outputs
        }
    
    def validate_component(self, component_name: str, component_class: Any, 
                          test_data: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a single component."""
        self.logger.info(f"🔍 Validating component: {component_name}")
        
        validation_result = {
            'component_name': component_name,
            'success': False,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': {},
            'errors': [],
            'warnings': [],
            'performance_metrics': {},
            'health_status': None
        }
        
        try:
            # Test 1: Component Creation
            self.logger.info(f"  Test 1: Creating {component_name} component")
            component = component_class(
                name=f"test_{component_name}",
                config=config or {},
                logger=self.logger
            )
            validation_result['test_results']['creation'] = True
            validation_result['tests_passed'] += 1
            
            # Test 2: Initialization
            self.logger.info(f"  Test 2: Initializing {component_name} component")
            init_success = component.initialize()
            validation_result['test_results']['initialization'] = init_success
            if init_success:
                validation_result['tests_passed'] += 1
            else:
                validation_result['tests_failed'] += 1
                validation_result['errors'].append("Initialization failed")
            
            if init_success:
                # Test 3: Configuration Management
                self.logger.info(f"  Test 3: Testing configuration management")
                config_test = self._test_configuration_management(component)
                validation_result['test_results']['configuration'] = config_test['success']
                if config_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(config_test['errors'])
                
                # Test 4: State Management
                self.logger.info(f"  Test 4: Testing state management")
                state_test = self._test_state_management(component)
                validation_result['test_results']['state_management'] = state_test['success']
                if state_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(state_test['errors'])
                
                # Test 5: Data Processing
                self.logger.info(f"  Test 5: Testing data processing")
                processing_test = self._test_data_processing(component, test_data)
                validation_result['test_results']['data_processing'] = processing_test['success']
                if processing_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(processing_test['errors'])
                
                # Test 6: Performance Monitoring
                self.logger.info(f"  Test 6: Testing performance monitoring")
                performance_test = self._test_performance_monitoring(component)
                validation_result['test_results']['performance_monitoring'] = performance_test['success']
                if performance_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(performance_test['errors'])
                
                validation_result['performance_metrics'] = performance_test.get('metrics', {})
                
                # Test 7: Health Status
                self.logger.info(f"  Test 7: Testing health status")
                health_test = self._test_health_status(component)
                validation_result['test_results']['health_status'] = health_test['success']
                if health_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(health_test['errors'])
                
                validation_result['health_status'] = health_test.get('health', {})
                
                # Test 8: Serialization
                self.logger.info(f"  Test 8: Testing serialization")
                serialization_test = self._test_serialization(component)
                validation_result['test_results']['serialization'] = serialization_test['success']
                if serialization_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(serialization_test['errors'])
                
                # Test 9: Cleanup
                self.logger.info(f"  Test 9: Testing cleanup")
                cleanup_test = self._test_cleanup(component)
                validation_result['test_results']['cleanup'] = cleanup_test['success']
                if cleanup_test['success']:
                    validation_result['tests_passed'] += 1
                else:
                    validation_result['tests_failed'] += 1
                    validation_result['errors'].extend(cleanup_test['errors'])
            
            # Determine overall success
            validation_result['success'] = (
                validation_result['tests_passed'] > 0 and 
                validation_result['tests_failed'] == 0
            )
            
            if validation_result['success']:
                self.logger.info(f"✅ {component_name} validation passed ({validation_result['tests_passed']} tests)")
            else:
                self.logger.error(f"❌ {component_name} validation failed ({validation_result['tests_failed']} tests failed)")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ {component_name} validation failed with exception: {e}")
            validation_result['errors'].append(str(e))
            validation_result['tests_failed'] += 1
            return validation_result
    
    def _test_configuration_management(self, component: Any) -> Dict[str, Any]:
        """Test configuration management."""
        try:
            # Test get_config
            config = component.get_config()
            if not isinstance(config, dict):
                return {'success': False, 'errors': ['get_config() should return dict']}
            
            # Test update_config
            component.update_config({'test_param': 'test_value'})
            if component.get_config('test_param') != 'test_value':
                return {'success': False, 'errors': ['update_config() not working']}
            
            # Test validate_config
            if hasattr(component, 'validate_config'):
                is_valid = component.validate_config()
                if not isinstance(is_valid, bool):
                    return {'success': False, 'errors': ['validate_config() should return bool']}
            
            return {'success': True, 'errors': []}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def _test_state_management(self, component: Any) -> Dict[str, Any]:
        """Test state management."""
        try:
            # Test set_state and get_state
            component.set_state('test_key', 'test_value')
            if component.get_state('test_key') != 'test_value':
                return {'success': False, 'errors': ['set_state/get_state not working']}
            
            # Test has_state
            if not component.has_state('test_key'):
                return {'success': False, 'errors': ['has_state not working']}
            
            # Test get_all_state
            all_state = component.get_all_state()
            if not isinstance(all_state, dict):
                return {'success': False, 'errors': ['get_all_state should return dict']}
            
            # Test ML state if available
            if hasattr(component, 'set_ml_state'):
                component.set_ml_state('test_ml_key', 'test_ml_value')
                if component.get_ml_state('test_ml_key') != 'test_ml_value':
                    return {'success': False, 'errors': ['ML state management not working']}
            
            return {'success': True, 'errors': []}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def _test_data_processing(self, component: Any, test_data: Any) -> Dict[str, Any]:
        """Test data processing."""
        try:
            # Test input validation
            validation_result = component.validate_input(test_data)
            if not isinstance(validation_result, object):  # ValidationResult object
                return {'success': False, 'errors': ['validate_input should return ValidationResult']}
            
            # Test can_process
            if hasattr(component, 'can_process'):
                can_process = component.can_process(test_data)
                if not isinstance(can_process, bool):
                    return {'success': False, 'errors': ['can_process should return bool']}
            
            # Test process (if validation passes)
            if validation_result.is_valid:
                result = component.process(test_data)
                if result is None:
                    return {'success': False, 'errors': ['process should return result']}
            
            return {'success': True, 'errors': []}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def _test_performance_monitoring(self, component: Any) -> Dict[str, Any]:
        """Test performance monitoring."""
        try:
            # Test get_performance_stats
            stats = component.get_performance_stats()
            if not isinstance(stats, dict):
                return {'success': False, 'errors': ['get_performance_stats should return dict']}
            
            # Test get_performance_summary
            if hasattr(component, 'get_performance_summary'):
                summary = component.get_performance_summary()
                if not isinstance(summary, dict):
                    return {'success': False, 'errors': ['get_performance_summary should return dict']}
            
            # Test reset_stats
            if hasattr(component, 'reset_stats'):
                component.reset_stats()
                stats_after_reset = component.get_performance_stats()
                if stats_after_reset.get('total_operations', 0) != 0:
                    return {'success': False, 'errors': ['reset_stats not working']}
            
            return {'success': True, 'errors': [], 'metrics': stats}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def _test_health_status(self, component: Any) -> Dict[str, Any]:
        """Test health status."""
        try:
            # Test get_status
            status = component.get_status()
            if not isinstance(status, dict):
                return {'success': False, 'errors': ['get_status should return dict']}
            
            # Test get_health_report
            if hasattr(component, 'get_health_report'):
                health = component.get_health_report()
                if not isinstance(health, dict):
                    return {'success': False, 'errors': ['get_health_report should return dict']}
                
                return {'success': True, 'errors': [], 'health': health}
            
            return {'success': True, 'errors': [], 'health': status}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def _test_serialization(self, component: Any) -> Dict[str, Any]:
        """Test serialization."""
        try:
            # Test serialize
            serialized = component.serialize()
            if not isinstance(serialized, dict):
                return {'success': False, 'errors': ['serialize should return dict']}
            
            # Test deserialize
            new_component = component.__class__(f"test_{component.name}_deserialized")
            new_component.deserialize(serialized)
            
            # Verify deserialization
            if new_component.name != component.name:
                return {'success': False, 'errors': ['deserialization not working correctly']}
            
            return {'success': True, 'errors': []}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def _test_cleanup(self, component: Any) -> Dict[str, Any]:
        """Test cleanup."""
        try:
            # Test cleanup
            component.cleanup()
            
            # Verify cleanup
            if component.is_initialized():
                return {'success': False, 'errors': ['cleanup should set initialized to False']}
            
            return {'success': True, 'errors': []}
            
        except Exception as e:
            return {'success': False, 'errors': [str(e)]}
    
    def validate_all_components(self, mode: str = 'comprehensive') -> Dict[str, Any]:
        """Validate all migrated components."""
        self.logger.info(f"🚀 Starting validation in {mode} mode")
        
        # Define components to validate
        components_to_validate = {
            'analyst_models': {
                'class': AnalystModelsTrainingModular,
                'factory': create_analyst_models_training,
                'test_data': self.test_data['analyst_data'],
                'config': {
                    'model': {'model_types': ['tcn', 'lightgbm', 'ridge']},
                    'training': {'epochs': 5}
                }
            },
            'analyst_ensemble': {
                'class': AnalystEnsembleTrainingModular,
                'factory': create_analyst_ensemble_training,
                'test_data': {
                    **self.test_data['analyst_data'],
                    'base_model_outputs': self.test_data['base_model_outputs']
                },
                'config': {
                    'model': {'base_models': ['tcn', 'lightgbm', 'ridge']},
                    'training': {'epochs': 5}
                }
            },
            'ml_labeling': {
                'class': MLEntryTimingLabelerModular,
                'factory': create_ml_entry_timing_labeler,
                'test_data': self.test_data['labeling_data'],
                'config': {
                    'model': {'ml_model_type': 'random_forest'},
                    'training': {'epochs': 5}
                }
            },
            'unified_pipeline': {
                'class': UnifiedTrainingPipelineModular,
                'factory': create_unified_training_pipeline,
                'test_data': self.test_data,
                'config': {
                    'pipeline': {'phases': ['analyst_models', 'ml_labeling']},
                    'analyst_models': {'model_types': ['tcn', 'lightgbm']},
                    'ml_labeling': {'ml_model_type': 'random_forest'}
                }
            }
        }
        
        validation_results = {}
        start_time = time.time()
        
        for comp_name, comp_info in components_to_validate.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Validating {comp_name}")
            self.logger.info(f"{'='*60}")
            
            # Validate component
            result = self.validate_component(
                comp_name,
                comp_info['class'],
                comp_info['test_data'],
                comp_info['config']
            )
            
            validation_results[comp_name] = result
            
            # Log summary
            if result['success']:
                self.logger.info(f"✅ {comp_name} validation PASSED")
            else:
                self.logger.error(f"❌ {comp_name} validation FAILED")
                for error in result['errors']:
                    self.logger.error(f"  Error: {error}")
        
        # Calculate overall results
        total_components = len(validation_results)
        successful_components = sum(1 for r in validation_results.values() if r['success'])
        failed_components = total_components - successful_components
        
        total_tests = sum(r['tests_passed'] + r['tests_failed'] for r in validation_results.values())
        total_passed = sum(r['tests_passed'] for r in validation_results.values())
        total_failed = sum(r['tests_failed'] for r in validation_results.values())
        
        overall_results = {
            'validation_time': time.time() - start_time,
            'total_components': total_components,
            'successful_components': successful_components,
            'failed_components': failed_components,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0,
            'component_results': validation_results
        }
        
        # Log overall summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Components: {total_components}")
        self.logger.info(f"Successful: {successful_components}")
        self.logger.info(f"Failed: {failed_components}")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {total_passed}")
        self.logger.info(f"Failed: {total_failed}")
        self.logger.info(f"Success Rate: {overall_results['success_rate']:.2%}")
        self.logger.info(f"Validation Time: {overall_results['validation_time']:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        self.logger.info(f"\n💾 Results saved to: {results_file}")
        
        return overall_results


def main():
    """Main validation script entry point."""
    parser = argparse.ArgumentParser(description='Validate migrated models training components')
    parser.add_argument('--components', nargs='+', default=['all'],
                       help='Components to validate (default: all)')
    parser.add_argument('--mode', choices=['quick', 'comprehensive'], default='comprehensive',
                       help='Validation mode (default: comprehensive)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create validator
    validator = MigrationValidator(logger)
    
    try:
        logger.info("🚀 Starting Migration Validation")
        
        # Validate components
        results = validator.validate_all_components(args.mode)
        
        # Determine exit code
        if results['failed_components'] == 0:
            logger.info("✅ All validations passed!")
            return 0
        else:
            logger.error(f"❌ {results['failed_components']} components failed validation")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())