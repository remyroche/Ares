"""
Comprehensive BaseStep Utility Usage Examples

This module demonstrates how to properly use the comprehensive BaseStep utilities
in training components. It shows best practices for:

1. Initialization with comprehensive utilities
2. Data extraction and validation
3. Data quality analysis
4. Hardware optimization
5. Performance monitoring
6. Model persistence and caching
7. Error handling and logging
8. Configuration management

Usage Examples:
==============

# Basic usage
from src.training.steps.models_training.components.analyst_base_training import AnalystBaseTraining

# Create component with comprehensive utilities
component = AnalystBaseTraining(
    name="example_analyst_training",
    config={
        'model_types': ['LIGHTGBM', 'CATBOOST'],
        'timeframe': '15m',
        'symbol': 'ETHUSDT'
    }
)

# Initialize with utility integration
await component.initialize()

# Run training with comprehensive utilities
result = await component.run(training_data)

# Advanced usage with custom utilities
component = AnalystBaseTraining(
    name="advanced_analyst_training",
    config={
        'model_types': ['LIGHTGBM', 'CATBOOST'],
        'timeframe': '15m',
        'symbol': 'ETHUSDT',
        'enable_hardware_optimization': True,
        'enable_data_quality_analysis': True,
        'enable_performance_monitoring': True
    }
)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.training.steps.models_training.components.analyst_base_training import AnalystBaseTraining
from src.training.steps.models_training.components.tactician_base_training import TacticianBaseTraining
from src.training.steps.models_training.components.analyst_ensemble_training import AnalystEnsembleTraining
from src.training.steps.models_training.components.tactician_ensemble_training import TacticianEnsembleTraining
from src.training.steps.models_training.components.ml_entry_timing_labeler_modular import MLEntryTimingLabelerModular


class ComprehensiveUtilityUsageExample:
    """
    Example class demonstrating comprehensive BaseStep utility usage.
    
    This class shows how to properly use all the comprehensive utilities
    available in BaseStep for training components.
    """
    
    def __init__(self):
        """Initialize the example with comprehensive utilities."""
        self.logger = logging.getLogger(__name__)
        self.examples = {}
    
    async def demonstrate_analyst_base_training(self) -> Dict[str, Any]:
        """
        Demonstrate AnalystBaseTraining with comprehensive utilities.
        
        Returns:
            Training result with comprehensive utility usage
        """
        try:
            # Create component with comprehensive utilities
            component = AnalystBaseTraining(
                name="example_analyst_training",
                config={
                    'model_types': ['LIGHTGBM', 'CATBOOST'],
                    'timeframe': '15m',
                    'symbol': 'ETHUSDT',
                    'auto_save': True,
                    'enable_patchtst_features': True,
                    'enable_regime_features': True,
                    'enable_multi_timeframe': True
                }
            )
            
            # Initialize with utility integration
            self.logger.info("Initializing AnalystBaseTraining with comprehensive utilities...")
            if not await component.initialize():
                return {'success': False, 'error': 'Initialization failed'}
            
            # Create sample training data
            training_data = self._create_sample_training_data()
            
            # Run training with comprehensive utilities
            self.logger.info("Running AnalystBaseTraining with comprehensive utilities...")
            result = await component.run(training_data)
            
            # Demonstrate utility usage
            self._demonstrate_utility_usage(component)
            
            return result
            
        except Exception as e:
            self.logger.error(f"AnalystBaseTraining example failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def demonstrate_tactician_base_training(self) -> Dict[str, Any]:
        """
        Demonstrate TacticianBaseTraining with comprehensive utilities.
        
        Returns:
            Training result with comprehensive utility usage
        """
        try:
            # Create component with comprehensive utilities
            component = TacticianBaseTraining(
                name="example_tactician_training",
                config={
                    'model_types': ['LIGHTGBM', 'CATBOOST'],
                    'timeframe': '15m',
                    'symbol': 'ETHUSDT',
                    'auto_save': True,
                    'enable_entry_timing': True,
                    'enable_exit_timing': True,
                    'enable_position_sizing': True
                }
            )
            
            # Initialize with utility integration
            self.logger.info("Initializing TacticianBaseTraining with comprehensive utilities...")
            if not await component.initialize():
                return {'success': False, 'error': 'Initialization failed'}
            
            # Create sample training data
            training_data = self._create_sample_training_data()
            
            # Run training with comprehensive utilities
            self.logger.info("Running TacticianBaseTraining with comprehensive utilities...")
            result = await component.run(training_data)
            
            # Demonstrate utility usage
            self._demonstrate_utility_usage(component)
            
            return result
            
        except Exception as e:
            self.logger.error(f"TacticianBaseTraining example failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def demonstrate_ensemble_training(self) -> Dict[str, Any]:
        """
        Demonstrate ensemble training with comprehensive utilities.
        
        Returns:
            Training result with comprehensive utility usage
        """
        try:
            # Create Analyst ensemble component
            analyst_ensemble = AnalystEnsembleTraining(
                name="example_analyst_ensemble",
                config={
                    'base_models': ['XGBOOST', 'CATBOOST', 'LIGHTGBM'],
                    'ensemble_method': 'VOTING',
                    'timeframe': '15m',
                    'symbol': 'ETHUSDT',
                    'save_models': True,
                    'enable_evaluation': True
                }
            )
            
            # Create Tactician ensemble component
            tactician_ensemble = TacticianEnsembleTraining(
                name="example_tactician_ensemble",
                config={
                    'base_models': ['LIGHTGBM', 'CATBOOST', 'NEURAL_NETWORK'],
                    'ensemble_method': 'STACKING',
                    'timeframe': '15m',
                    'symbol': 'ETHUSDT',
                    'auto_save': True
                }
            )
            
            # Initialize both components
            self.logger.info("Initializing ensemble components with comprehensive utilities...")
            if not await analyst_ensemble.initialize():
                return {'success': False, 'error': 'Analyst ensemble initialization failed'}
            if not await tactician_ensemble.initialize():
                return {'success': False, 'error': 'Tactician ensemble initialization failed'}
            
            # Create sample training data
            training_data = self._create_sample_training_data()
            
            # Run both ensemble trainings
            self.logger.info("Running ensemble trainings with comprehensive utilities...")
            analyst_result = await analyst_ensemble.run(training_data)
            tactician_result = await tactician_ensemble.run(training_data)
            
            # Demonstrate utility usage
            self._demonstrate_utility_usage(analyst_ensemble)
            self._demonstrate_utility_usage(tactician_ensemble)
            
            return {
                'analyst_ensemble': analyst_result,
                'tactician_ensemble': tactician_result
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble training example failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def demonstrate_ml_entry_timing_labeler(self) -> Dict[str, Any]:
        """
        Demonstrate MLEntryTimingLabelerModular with comprehensive utilities.
        
        Returns:
            Training result with comprehensive utility usage
        """
        try:
            # Create component with comprehensive utilities
            component = MLEntryTimingLabelerModular(
                name="example_ml_entry_timing_labeler",
                config={
                    'model': {
                        'type': 'ml_labeler',
                        'labeling_method': 'iterative',
                        'ml_model_type': 'random_forest',
                        'model_params': {}
                    },
                    'training': {
                        'max_iterations': 5,
                        'convergence_threshold': 0.01,
                        'validation_split': 0.2
                    },
                    'data': {
                        'min_samples_per_class': 100,
                        'max_samples_per_class': 10000
                    }
                }
            )
            
            # Initialize with utility integration
            self.logger.info("Initializing MLEntryTimingLabelerModular with comprehensive utilities...")
            if not await component.initialize():
                return {'success': False, 'error': 'Initialization failed'}
            
            # Create sample training data
            training_data = self._create_sample_training_data()
            
            # Run training with comprehensive utilities
            self.logger.info("Running MLEntryTimingLabelerModular with comprehensive utilities...")
            result = await component.run(training_data)
            
            # Demonstrate utility usage
            self._demonstrate_utility_usage(component)
            
            return result
            
        except Exception as e:
            self.logger.error(f"MLEntryTimingLabelerModular example failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_sample_training_data(self) -> Dict[str, Any]:
        """Create sample training data for examples."""
        try:
            # Create sample features
            np.random.seed(42)
            n_samples = 1000
            n_features = 20
            
            X_train = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            # Create sample targets
            y_train = pd.Series(
                np.random.randint(0, 2, n_samples),
                name='target'
            )
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'features': X_train,  # For ensemble components
                'targets': y_train    # For ensemble components
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create sample training data: {e}")
            return {}
    
    def _demonstrate_utility_usage(self, component) -> None:
        """Demonstrate comprehensive utility usage."""
        try:
            # Demonstrate utility availability checking
            availability = component._get_availability_status()
            component.tprint_info(f"📊 Utility availability: {availability}")
            
            # Demonstrate configuration preview
            component.tprint_config_preview(component.config.__dict__, "Component Configuration")
            
            # Demonstrate performance summary
            component.tprint_performance_summary({
                'component_name': component.name,
                'utilities_available': sum(availability.values()),
                'total_utilities': len(availability)
            })
            
            # Demonstrate hardware stats (if available)
            if component.hardware_utils:
                hw_stats = component._get_hardware_stats()
                component.tprint_hardware_stats(hw_stats)
            
            # Demonstrate memory usage (if available)
            if component.hardware_utils and 'get_memory_usage' in component.hardware_utils:
                memory_usage = component.hardware_utils['get_memory_usage']()
                component.tprint_memory_usage(memory_usage)
            
            # Demonstrate safe operations
            safe_result = component._safe_divide(10, 2, default=0)
            component.tprint_info(f"📊 Safe division result: {safe_result}")
            
            # Demonstrate validation
            validation_result = component._validate_finite(3.14, default=0)
            component.tprint_info(f"📊 Validation result: {validation_result}")
            
            # Demonstrate directory operations
            test_dir = "/tmp/test_directory"
            if component._ensure_directory(test_dir):
                component.tprint_success("✅ Directory created successfully")
            
            # Demonstrate JSON operations
            test_data = {'test': 'data', 'number': 42}
            if component._safe_json_save(test_data, "/tmp/test_data.json"):
                component.tprint_success("✅ JSON saved successfully")
                loaded_data = component._safe_json_load("/tmp/test_data.json")
                component.tprint_info(f"📊 Loaded JSON data: {loaded_data}")
            
        except Exception as e:
            self.logger.error(f"Utility demonstration failed: {e}")
    
    async def run_all_examples(self) -> Dict[str, Any]:
        """Run all comprehensive utility usage examples."""
        try:
            self.logger.info("🚀 Starting comprehensive utility usage examples...")
            
            results = {}
            
            # Run AnalystBaseTraining example
            self.logger.info("📊 Running AnalystBaseTraining example...")
            results['analyst_base_training'] = await self.demonstrate_analyst_base_training()
            
            # Run TacticianBaseTraining example
            self.logger.info("📊 Running TacticianBaseTraining example...")
            results['tactician_base_training'] = await self.demonstrate_tactician_base_training()
            
            # Run ensemble training examples
            self.logger.info("📊 Running ensemble training examples...")
            results['ensemble_training'] = await self.demonstrate_ensemble_training()
            
            # Run ML Entry Timing Labeler example
            self.logger.info("📊 Running MLEntryTimingLabelerModular example...")
            results['ml_entry_timing_labeler'] = await self.demonstrate_ml_entry_timing_labeler()
            
            self.logger.info("✅ All comprehensive utility usage examples completed!")
            
            return {
                'success': True,
                'results': results,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive utility usage examples failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }


# Factory function for easy usage
def create_comprehensive_utility_example() -> ComprehensiveUtilityUsageExample:
    """Create a comprehensive utility usage example instance."""
    return ComprehensiveUtilityUsageExample()


# Main execution function
async def main():
    """Main execution function for comprehensive utility usage examples."""
    try:
        # Create example instance
        example = create_comprehensive_utility_example()
        
        # Run all examples
        results = await example.run_all_examples()
        
        # Print results
        print("Comprehensive Utility Usage Examples Results:")
        print("=" * 50)
        for component, result in results['results'].items():
            print(f"\n{component}:")
            print(f"  Success: {result.get('success', False)}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            if 'training_time' in result:
                print(f"  Training Time: {result['training_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive utility usage examples failed: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())