#!/usr/bin/env python3
"""
Enhanced ML Pipeline Usage Example

This example demonstrates how to use the enhanced ML pipeline system with
comprehensive error detection, HPO monitoring, testing, and reporting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Import enhanced components
from ..config.enhanced_ml_config import get_config, EnhancedMLConfig
from ..integration.enhanced_ml_pipeline_integration import (
    get_global_integration,
    PipelineStage,
    PipelineStatus
)
from ..monitoring.enhanced_error_detector import detect_error
from ..optimization.enhanced_hpo_monitor import get_global_hpo_monitor
from ..testing.enhanced_testing_framework import (
    get_global_testing_framework,
    TestType,
    TestStatus
)
from ..reporting.enhanced_reporting_system import (
    get_global_reporting_system,
    ReportType,
    AlertLevel
)

# Import your existing ML components
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class EnhancedMLPipelineExample:
    """Example implementation of enhanced ML pipeline."""
    
    def __init__(self, config_preset: str = "development"):
        """Initialize the enhanced ML pipeline example."""
        self.config = get_config(config_preset)
        
        # Initialize enhanced components
        self.integration = get_global_integration(self.config.to_dict())
        self.error_detector = self.integration.error_detector
        self.hpo_monitor = self.integration.hpo_monitor
        self.testing_framework = self.integration.testing_framework
        self.reporting_system = self.integration.reporting_system
        
        # Register pipeline stage handlers
        self._register_stage_handlers()
        
        logger.info("🚀 Enhanced ML Pipeline Example initialized")
    
    def _register_stage_handlers(self):
        """Register handlers for each pipeline stage."""
        
        # Data preparation stage
        self.integration.register_stage_handler(
            PipelineStage.DATA_PREPARATION,
            self._handle_data_preparation
        )
        
        # Feature engineering stage
        self.integration.register_stage_handler(
            PipelineStage.FEATURE_ENGINEERING,
            self._handle_feature_engineering
        )
        
        # Model training stage
        self.integration.register_stage_handler(
            PipelineStage.MODEL_TRAINING,
            self._handle_model_training
        )
        
        # HPO optimization stage
        self.integration.register_stage_handler(
            PipelineStage.HPO_OPTIMIZATION,
            self._handle_hpo_optimization
        )
        
        # Model validation stage
        self.integration.register_stage_handler(
            PipelineStage.MODEL_VALIDATION,
            self._handle_model_validation
        )
        
        # Register stage validators
        self.integration.register_stage_validator(
            PipelineStage.MODEL_TRAINING,
            self._validate_model_training_prerequisites
        )
    
    def run_complete_pipeline(self, 
                            pipeline_name: str = "Enhanced ML Pipeline",
                            data_path: Optional[str] = None) -> str:
        """Run the complete enhanced ML pipeline."""
        try:
            logger.info(f"🚀 Starting enhanced ML pipeline: {pipeline_name}")
            
            # Define pipeline stages
            stages = [
                PipelineStage.DATA_PREPARATION,
                PipelineStage.FEATURE_ENGINEERING,
                PipelineStage.MODEL_TRAINING,
                PipelineStage.HPO_OPTIMIZATION,
                PipelineStage.MODEL_VALIDATION
            ]
            
            # Execution configuration
            execution_config = {
                'data_path': data_path,
                'pipeline_name': pipeline_name,
                'start_time': datetime.now().isoformat(),
                'config_preset': self.config.__class__.__name__
            }
            
            # Execute pipeline
            execution_id = self.integration.execute_pipeline(
                pipeline_name=pipeline_name,
                stages=stages,
                execution_config=execution_config
            )
            
            logger.info(f"✅ Pipeline execution completed: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            raise
    
    def _handle_data_preparation(self, execution) -> Dict[str, Any]:
        """Handle data preparation stage."""
        try:
            logger.info("📊 Processing data preparation stage")
            
            # Generate synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            n_features = 20
            
            # Create synthetic dataset
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)
            
            # Simulate some data quality issues
            if np.random.random() < 0.1:  # 10% chance of data issues
                # Introduce some NaN values
                X[np.random.choice(n_samples, 50, replace=False), 
                  np.random.choice(n_features, 5, replace=False)] = np.nan
                
                # Detect and handle the error
                error_context = {
                    'component': 'data_preparation',
                    'function': '_handle_data_preparation',
                    'input_data_shape': X.shape,
                    'data_characteristics': {
                        'has_nan': True,
                        'has_inf': False,
                        'single_class': False
                    }
                }
                
                # This will be automatically detected and handled
                raise ValueError("Data quality issue detected: NaN values found")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Store data in execution metadata
            execution.metadata.update({
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape,
                'y_train_shape': y_train.shape,
                'y_test_shape': y_test.shape,
                'n_features': n_features,
                'n_samples': n_samples
            })
            
            return {
                'status': 'completed',
                'data_shapes': {
                    'X_train': X_train.shape,
                    'X_test': X_test.shape,
                    'y_train': y_train.shape,
                    'y_test': y_test.shape
                },
                'data_quality': {
                    'has_nan': False,
                    'has_inf': False,
                    'class_balance': np.bincount(y).tolist()
                },
                'processing_time': time.time()
            }
            
        except Exception as e:
            # Error will be automatically detected and classified
            raise
    
    def _handle_feature_engineering(self, execution) -> Dict[str, Any]:
        """Handle feature engineering stage."""
        try:
            logger.info("🔧 Processing feature engineering stage")
            
            # Simulate feature engineering
            time.sleep(1)  # Simulate processing time
            
            # Simulate feature selection
            n_features_selected = np.random.randint(10, 20)
            
            return {
                'status': 'completed',
                'features_selected': n_features_selected,
                'feature_importance': np.random.rand(n_features_selected).tolist(),
                'processing_time': time.time()
            }
            
        except Exception as e:
            raise
    
    def _handle_model_training(self, execution) -> Dict[str, Any]:
        """Handle model training stage."""
        try:
            logger.info("🤖 Processing model training stage")
            
            # Simulate model training
            time.sleep(2)  # Simulate training time
            
            # Simulate training metrics
            training_accuracy = np.random.uniform(0.7, 0.95)
            training_loss = np.random.uniform(0.1, 0.5)
            
            # Simulate potential training issues
            if training_accuracy < 0.75:
                # This will trigger an alert
                self.reporting_system.create_alert(
                    AlertLevel.WARNING,
                    "Low Training Accuracy",
                    f"Training accuracy is below threshold: {training_accuracy:.3f}",
                    "model_training",
                    {'accuracy': training_accuracy, 'threshold': 0.75}
                )
            
            return {
                'status': 'completed',
                'model_type': 'RandomForestClassifier',
                'training_accuracy': training_accuracy,
                'training_loss': training_loss,
                'training_time': time.time(),
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
            }
            
        except Exception as e:
            raise
    
    def _handle_hpo_optimization(self, execution) -> Dict[str, Any]:
        """Handle HPO optimization stage."""
        try:
            logger.info("🎯 Processing HPO optimization stage")
            
            # Start HPO study
            study_id = f"hpo_study_{execution.execution_id}"
            study_name = f"HPO Study for {execution.pipeline_name}"
            
            hpo_study = self.hpo_monitor.start_study(study_id, study_name)
            
            # Simulate HPO trials
            n_trials = np.random.randint(10, 30)
            best_score = 0.0
            
            for trial in range(n_trials):
                # Simulate trial parameters
                params = {
                    'n_estimators': np.random.randint(50, 200),
                    'max_depth': np.random.randint(5, 20),
                    'min_samples_split': np.random.randint(2, 10)
                }
                
                # Simulate objective value
                objective_value = np.random.uniform(0.6, 0.9)
                best_score = max(best_score, objective_value)
                
                # Simulate occasional trial failures
                error_info = None
                if np.random.random() < 0.1:  # 10% failure rate
                    error_info = {
                        'error_type': 'convergence_error',
                        'error_message': 'Model failed to converge'
                    }
                
                # Record trial result
                self.hpo_monitor.record_trial_result(
                    study_id=study_id,
                    trial_number=trial,
                    parameters=params,
                    objective_value=objective_value,
                    training_time=np.random.uniform(1, 5),
                    memory_usage=np.random.uniform(0.1, 0.8),
                    error_info=error_info
                )
                
                time.sleep(0.1)  # Simulate trial time
            
            # Complete study
            self.hpo_monitor.complete_study(study_id)
            
            return {
                'status': 'completed',
                'study_id': study_id,
                'total_trials': n_trials,
                'best_score': best_score,
                'best_parameters': {
                    'n_estimators': 150,
                    'max_depth': 12,
                    'min_samples_split': 5
                },
                'optimization_time': time.time()
            }
            
        except Exception as e:
            raise
    
    def _handle_model_validation(self, execution) -> Dict[str, Any]:
        """Handle model validation stage."""
        try:
            logger.info("✅ Processing model validation stage")
            
            # Simulate model validation
            time.sleep(1)
            
            # Simulate validation metrics
            validation_accuracy = np.random.uniform(0.7, 0.9)
            precision = np.random.uniform(0.6, 0.9)
            recall = np.random.uniform(0.6, 0.9)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            # Run validation tests
            self._run_validation_tests(execution)
            
            return {
                'status': 'completed',
                'validation_accuracy': validation_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'validation_time': time.time(),
                'test_results': {
                    'unit_tests_passed': 8,
                    'integration_tests_passed': 3,
                    'performance_tests_passed': 2
                }
            }
            
        except Exception as e:
            raise
    
    def _validate_model_training_prerequisites(self, execution) -> Dict[str, Any]:
        """Validate model training prerequisites."""
        try:
            # Check if data preparation was successful
            if PipelineStage.DATA_PREPARATION not in execution.stage_results:
                return {
                    'valid': False,
                    'error': 'Data preparation stage not completed'
                }
            
            data_result = execution.stage_results[PipelineStage.DATA_PREPARATION]
            if data_result.get('status') != 'completed':
                return {
                    'valid': False,
                    'error': 'Data preparation stage failed'
                }
            
            # Check data quality
            data_quality = data_result.get('data_quality', {})
            if data_quality.get('has_nan', False):
                return {
                    'valid': False,
                    'error': 'Training data contains NaN values'
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation failed: {str(e)}'
            }
    
    def _run_validation_tests(self, execution):
        """Run validation tests using the testing framework."""
        try:
            # Create test suite
            test_suite = self.testing_framework.create_unit_test(
                test_id="validation_accuracy_test",
                test_name="Model Validation Accuracy Test",
                test_function=lambda: np.random.uniform(0.8, 0.95),
                expected_result=0.8,
                timeout=30.0
            )
            
            # Create performance test
            performance_test = self.testing_framework.create_performance_test(
                test_id="model_inference_speed",
                test_name="Model Inference Speed Test",
                test_function=lambda: {'inference_time': np.random.uniform(0.01, 0.1)},
                baseline_metrics={'inference_time': 0.05},
                tolerance=0.2
            )
            
            # Create validation test
            validation_test = self.testing_framework.create_validation_test(
                test_id="model_accuracy_validation",
                test_name="Model Accuracy Validation",
                test_function=lambda: {'accuracy': np.random.uniform(0.7, 0.9)},
                validation_criteria={
                    'accuracy': {
                        'min_value': 0.7,
                        'max_value': 1.0
                    }
                }
            )
            
            # Execute tests
            test_results = []
            for test_def in [test_suite, performance_test, validation_test]:
                result = self.testing_framework.execute_test(test_def)
                test_results.append(result)
            
            # Log test results
            passed_tests = sum(1 for r in test_results if r.status == TestStatus.PASSED)
            total_tests = len(test_results)
            
            logger.info(f"🧪 Validation tests completed: {passed_tests}/{total_tests} passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Validation tests failed: {e}")
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline execution status."""
        return self.integration.get_pipeline_status(execution_id)
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        return self.integration.get_integration_summary()

def main():
    """Main function to demonstrate the enhanced ML pipeline."""
    try:
        # Initialize the enhanced ML pipeline
        pipeline = EnhancedMLPipelineExample(config_preset="development")
        
        # Run the complete pipeline
        execution_id = pipeline.run_complete_pipeline(
            pipeline_name="Enhanced ML Pipeline Demo",
            data_path="synthetic_data.csv"
        )
        
        # Get pipeline status
        status = pipeline.get_pipeline_status(execution_id)
        print(f"Pipeline Status: {status}")
        
        # Get comprehensive summary
        summary = pipeline.get_comprehensive_summary()
        print(f"System Summary: {summary}")
        
        # Wait a bit for monitoring to complete
        time.sleep(5)
        
        print("✅ Enhanced ML Pipeline demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()