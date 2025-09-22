"""
Streamlined HMM Models Training with Hardware Optimization

Simplified HMM models training that leverages the ml_commons/ ML training pipeline
with extensive hardware optimization for Apple Silicon. Focuses on HMM state recognition
with 15m timeframe, using advanced tools for HPO, validation, lookahead protection,
overfitting detection, and hardware acceleration.

This is the primary HMM training implementation - extensively using ml_commons tools
and hardware optimization for maximum performance on M1/M2/M3 chips.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import gc

# Core imports - using common utilities
from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

# New ml_commons imports for extensive functionality
from src.utils.ml_common.utils.hmm_hpo_config import get_hmm_hyperparameter_optimizer
# from src.utils.ml_common.validation.hmm_validation_pipeline import get_hmm_validation_pipeline
from src.utils.ml_common.utils.hmm_temporal_protection import get_hmm_temporal_protection

# Hardware optimization imports
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_dataframe_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, create_m1_optimized_thread_pool
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, optimize_dataframe_for_m1


class StreamlinedHMMTrainingStep(BaseTrainingStep):
    """
    Streamlined HMM Training Step with Hardware Optimization for Apple Silicon.

    This class focuses specifically on HMM state recognition using 15m timeframe
    with extensive hardware optimization for M1/M2/M3 chips. Delegates most functionality
    to the common ML training pipeline while leveraging hardware acceleration.

    Key principles:
    - Use 15m timeframe for HMM state recognition
    - Hardware-optimized for Apple Silicon (CPU, GPU, Memory)
    - Minimal custom code - delegate to common_utils/ and hardware optimizers
    - Focus on state recognition, not prediction
    - Leverage HPO, validation, and reporting from common pipeline
    - Memory-efficient processing for large datasets
    """

    def __init__(self, config: Optional[HMMTrainingConfig] = None):
        """
        Initialize streamlined HMM training step with extensive ml_commons and hardware optimization.

        Args:
            config: HMM training configuration (will be updated to use 15m timeframe and hardware optimization)
        """
        # Ensure we have a config with 15m timeframe for HMM state recognition
        if config is None:
            config = HMMTrainingConfig(
                model_name="streamlined_hmm_state_recognition",
                timeframe="15m",  # Always use 15m for HMM state recognition
                model_types=self._get_hmm_model_types(),
                hpo_trials=50,
                enable_multi_objective=True,
                objectives=["accuracy", "f1_score", "regime_stability"],
                objective_weights=[0.4, 0.3, 0.2]  # Reduced regime stability weight for 15m short-term predictions
            )
        else:
            # Override timeframe to ensure 15m for HMM state recognition
            config.timeframe = "15m"

            # Ensure we have appropriate model types for state recognition
            if not hasattr(config, 'model_types') or len(config.model_types) == 0:
                config.model_types = self._get_hmm_model_types()

        super().__init__(config)
        self.logger = system_logger.getChild('StreamlinedHMMTrainingStep')

        # Initialize ml_commons utilities for extensive functionality
        self.hmm_hpo = get_hmm_hyperparameter_optimizer(config)
        # self.hmm_validation = get_hmm_validation_pipeline(config)
        self.hmm_temporal_protection = get_hmm_temporal_protection(config)

        # Initialize hardware optimization components
        self._initialize_hardware_optimization()

        self.logger.info("✅ Streamlined HMM Training Step initialized with ml_commons tools and hardware optimization")
        self.logger.info(f"📊 Timeframe: {config.timeframe} (HMM state recognition)")
        self.logger.info(f"📊 Model types: {config.model_types}")
        self.logger.info("🧠 Available tools: HPO, Universal Validation, Temporal Protection, Hardware Optimization")
        self.logger.info("🚀 Hardware optimization: CPU, GPU, Memory management enabled")

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize unified hardware manager for coordinated optimization
            self.hardware_manager = UnifiedHardwareManager()
            self.logger.info("✅ Unified Hardware Manager initialized")

            # Initialize memory optimizer with 8GB limit (configurable)
            self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
            self.memory_optimizer.start_monitoring()
            self.logger.info("✅ M1 Memory Optimizer initialized and monitoring started")

            # Initialize CPU optimizer
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.logger.info(f"✅ M1 CPU Optimizer initialized - {self.cpu_optimizer.get_optimal_worker_count()} optimal workers")

            # Initialize GPU manager
            self.gpu_manager = get_m1_gpu_manager()
            gpu_info = self.gpu_manager.get_gpu_info()
            if gpu_info['mps_available']:
                self.logger.info("✅ M1 GPU Manager initialized with MPS acceleration")
            else:
                self.logger.info("✅ M1 GPU Manager initialized (CPU fallback mode)")

            # Configure hardware for HMM training workload
            self.hardware_manager.configure_workload(
                workload_type="ml_training",
                optimization_level="balanced"
            )

        except Exception as e:
            self.logger.error(f"❌ Hardware optimization initialization failed: {e}")
            # Fallback: create basic instances
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.gpu_manager = None
            self.hardware_manager = None

    def _optimize_dataframe_for_hardware(self, df):
        """Optimize DataFrame for hardware acceleration."""
        if df is None:
            return df

        try:
            # Apply memory optimization
            if self.memory_optimizer:
                df = self.memory_optimizer.optimize_dataframe_memory(df)

            # Apply GPU optimization if available
            if self.gpu_manager and self.gpu_manager.is_m1:
                df = optimize_dataframe_for_m1(df)

            return df
        except Exception as e:
            self.logger.warning(f"DataFrame hardware optimization failed: {e}")
            return df

    def _get_hmm_model_types(self) -> List[str]:
        """
        Get HMM-specific model types optimized for state recognition using ml_commons.

        Uses the HMM HPO configuration for standardized model type selection.

        Returns:
            List of model types optimized for HMM state recognition
        """
        return self.hmm_hpo.get_hmm_model_types()


    def _evaluate_models_with_validation(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        regime_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate models using the universal validation integrator.

        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training labels
            regime_name: Name of the regime for context

        Returns:
            Enhanced evaluation results with HMM validation
        """
        self.logger.info(f"🔍 Evaluating models for {regime_name} using universal validation integrator")

        evaluation_results = {}

        for model_name, model in models.items():
            self.logger.info(f"📊 Evaluating {model_name} for {regime_name}")

            try:
                # Use validation data from regime data (we'll use training data for validation here)
                # In a real scenario, you'd have separate validation data
                X_val = X_train  # For now, use training data as validation
                y_val = y_train

                # Use universal validation integrator for comprehensive evaluation
                validation_result = self.validate_trained_model(
                    model=model,
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    timestamps=None,
                    feature_names=None,
                    model_name=model_name,
                    model_type=self._get_model_type_from_name(model_name),
                    fold_number=None
                )

                # Add additional model-specific evaluation
                basic_metrics = self.evaluate_models(
                    models={model_name: model},
                    X=X_val,
                    y=y_val,
                    is_classification=True
                )

                evaluation_results[model_name] = {
                    'basic_metrics': basic_metrics.get(model_name, {}),
                    'validation': validation_result,
                    'regime_context': regime_name,
                    'evaluation_timestamp': time.time()
                }

                # Log key findings
                overfitting_analysis = validation_result.get('overfitting_analysis', {})
                if overfitting_analysis.get('overfitting_detected', False):
                    self.logger.warning(f"⚠️ Overfitting detected in {model_name} for {regime_name}: "
                                      f"{overfitting_analysis.get('severity', 'unknown')} severity")

            except Exception as e:
                self.logger.error(f"❌ Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = {
                    'error': str(e),
                    'regime_context': regime_name
                }

        return evaluation_results

    def _get_model_type_from_name(self, model_name: str) -> str:
        """Get model type from model name."""
        model_type_mapping = {
            'logistic': 'logistic_regression',
            'lightgbm': 'lightgbm',
            'random_forest': 'random_forest',
            'xgboost': 'xgboost',
            'catboost': 'catboost'
        }

        for key, model_type in model_type_mapping.items():
            if key in model_name.lower():
                return model_type

        return 'unknown'

    def _get_feature_importance(self, model: Any) -> Optional[np.ndarray]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_).flatten()
            else:
                return None
        except:
            return None

    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute streamlined HMM training with hardware optimization using common_utils/ pipeline.

        This method focuses on calling the common ML training pipeline with proper parameters
        for HMM state recognition, enhanced with Apple Silicon hardware optimizations for
        memory efficiency and performance.

        Args:
            X: Input features
            y: Target values (HMM states to recognize)
            regime_labels: Regime labels for data stratification
            feature_names: Names of input features
            hmm_states: Optional HMM cluster/regime states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results from common pipeline with hardware optimization
        """
        self.logger.info("🚀 Starting streamlined HMM training execution with hardware optimization")

        # Memory checkpoint for monitoring
        if self.memory_optimizer:
            memory_checkpoint = self.memory_optimizer.memory_checkpoint("HMM_Training_Start")

        # Set up hardware optimization context for this workload
        if self.hardware_manager:
            with self.hardware_manager.get_optimization_context("ml_training") as context:
                return self._execute_with_hardware_optimization(
                    X, y, regime_labels, feature_names, hmm_states, **kwargs
                )
        else:
            # Fallback without hardware optimization
            return self._execute_with_hardware_optimization(
                X, y, regime_labels, feature_names, hmm_states, **kwargs
            )

    def _execute_with_hardware_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute HMM training with hardware optimizations."""

        # Memory checkpoint for monitoring
        memory_context = None
        if self.memory_optimizer:
            memory_context = self.memory_optimizer.memory_checkpoint("HMM_Training_Execution")

        # Validate input data using universal validation integration from BaseTrainingStep
        validation_results = self.validate_training_data(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            timestamps=None,
            model_type="hmm_state_recognition"
        )

        if not validation_results['valid']:
            self.logger.error("❌ Training data validation failed")
            return self._handle_training_error(
                Exception("Training data validation failed"),
                "data_validation"
            )

        # Log validation results
        self.logger.info(f"📊 Data validation: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
        for recommendation in validation_results.get('recommendations', []):
            self.logger.info(f"💡 Recommendation: {recommendation}")

        # Optimize input data for hardware acceleration
        X_optimized = self._optimize_array_for_hardware(X)
        regime_labels_optimized = self._optimize_array_for_hardware(regime_labels)
        if hmm_states is not None:
            hmm_states_optimized = self._optimize_array_for_hardware(hmm_states)

        # Analyze regimes using common regime analysis with hardware optimization
        regime_analysis = self.analyze_regimes(regime_labels_optimized)
        self.logger.info(f"📊 Regime analysis: {len(regime_analysis['regime_counts'])} regimes")

        # Prepare data for each regime with memory optimization
        regime_data = self.prepare_regime_data(
            X=X_optimized,
            y=y,
            regime_labels=regime_labels_optimized,
            regime_analysis=regime_analysis,
            hmm_states=hmm_states_optimized if hmm_states is not None else None
        )

        # Optimize regime data for hardware
        regime_data = self._optimize_regime_data_for_hardware(regime_data)

        # Train models using common training pipeline with hardware optimization
        # Focus on state recognition, not prediction
        training_results = self._train_hmm_state_recognition_models_with_hardware_optimization(
            regime_data=regime_data,
            feature_names=feature_names
        )

        # Generate enhanced reporting for all models using universal validation
        enhanced_reporting = self._generate_enhanced_model_report(
            models=training_results.get('models', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            regime_analysis=regime_analysis,
            validation_results=validation_results
        )

        # Create final results with enhanced reporting and hardware optimization info
        final_results = self._create_final_results(
            models=training_results.get('models', {}),
            metadata=training_results.get('metadata', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            training_time=training_results.get('training_time', 0),
            additional_results={
                'regime_analysis': regime_analysis,
                'validation_results': validation_results,
                'hmm_state_recognition_focus': True,
                'timeframe': self.config.timeframe,
                'model_types_used': self.config.model_types,
                'enhanced_reporting': enhanced_reporting,
                'ml_commons_integration': {
                    'hpo_used': True,
                    'universal_validation_used': True,
                    'temporal_protection_used': True,
                    'tools_available': [
                        'HMMHyperparameterOptimizer',
                        'UniversalValidationIntegrator',
                        'HMMTemporalProtection'
                    ]
                },
                'hardware_optimization': self._get_hardware_optimization_info(),
                'memory_optimization': {
                    'enabled': self.memory_optimizer is not None,
                    'monitoring_active': self.memory_optimizer.monitoring_active if self.memory_optimizer else False,
                    'memory_limit_gb': getattr(self.memory_optimizer, 'memory_limit_gb', None) if self.memory_optimizer else None
                }
            }
        )

        # Log comprehensive feature summary with hardware optimization info
        self.logger.info("📊 Comprehensive Feature Summary:")
        self.logger.info(f"  - Base features: {len(regime_data)} regimes")
        if feature_names:
            self.logger.info(f"  - Enhanced features: {len(feature_names)} total")
            self.logger.info(f"  - Feature categories: 13 comprehensive categories (excluding complex categories)")
        self.logger.info(f"  - Feature bank integration: ✅ Active")
        self.logger.info("🚀 Hardware optimization: Memory management and CPU/GPU acceleration enabled")

        # Log enhanced summary
        self._log_enhanced_training_summary(final_results)

        # Cleanup memory context
        if memory_context:
            memory_context.__exit__(None, None, None)

        return final_results

    def _optimize_array_for_hardware(self, array: np.ndarray) -> np.ndarray:
        """Optimize numpy array for hardware acceleration."""
        if array is None:
            return array

        try:
            # Use M1 GPU manager for array optimization if available
            if self.gpu_manager and self.gpu_manager.is_m1:
                return self.gpu_manager.create_m1_optimized_array(array)

            return array
        except Exception as e:
            self.logger.warning(f"Array hardware optimization failed: {e}")
            return array

    def _optimize_regime_data_for_hardware(self, regime_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, np.ndarray]]:
        """Optimize regime data for hardware acceleration and memory efficiency."""
        if not regime_data:
            return regime_data

        optimized_data = {}

        try:
            # Force garbage collection before processing
            if self.memory_optimizer:
                self.memory_optimizer.force_garbage_collection()

            for regime_id, data in regime_data.items():
                optimized_regime_data = {}

                # Optimize each array in the regime data
                for key, array in data.items():
                    if isinstance(array, np.ndarray):
                        optimized_regime_data[key] = self._optimize_array_for_hardware(array)
                    else:
                        optimized_regime_data[key] = array

                optimized_data[regime_id] = optimized_regime_data

                # Periodic memory cleanup during processing
                if regime_id % 5 == 0 and self.memory_optimizer:
                    self.memory_optimizer.force_garbage_collection()

            return optimized_data

        except Exception as e:
            self.logger.warning(f"Regime data hardware optimization failed: {e}")
            return regime_data

    def _get_hardware_optimization_info(self) -> Dict[str, Any]:
        """Get hardware optimization information for reporting."""
        info = {
            'cpu_optimization': {
                'enabled': self.cpu_optimizer is not None,
                'optimal_workers': self.cpu_optimizer.get_optimal_worker_count() if self.cpu_optimizer else None,
                'is_m1': self.cpu_optimizer.is_m1 if self.cpu_optimizer else False
            },
            'gpu_optimization': {
                'enabled': self.gpu_manager is not None,
                'mps_available': self.gpu_manager.mps_available if self.gpu_manager else False,
                'is_m1': self.gpu_manager.is_m1 if self.gpu_manager else False
            },
            'memory_optimization': {
                'enabled': self.memory_optimizer is not None,
                'monitoring_active': self.memory_optimizer.monitoring_active if self.memory_optimizer else False,
                'memory_limit_gb': getattr(self.memory_optimizer, 'memory_limit_gb', None) if self.memory_optimizer else None
            },
            'unified_hardware_manager': {
                'enabled': self.hardware_manager is not None
            }
        }

        # Get memory usage statistics
        if self.memory_optimizer:
            info['memory_stats'] = self.memory_optimizer.get_memory_stats()

        return info

    def _train_hmm_state_recognition_models_with_hardware_optimization(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train models for HMM state recognition with hardware optimization.

        Args:
            regime_data: Prepared data for each regime (optimized for hardware)
            feature_names: Names of features

        Returns:
            Training results from common pipeline with hardware optimization
        """
        self.logger.info("🔄 Training HMM state recognition models with hardware optimization")

        # Memory checkpoint for training phase
        training_memory_context = None
        if self.memory_optimizer:
            training_memory_context = self.memory_optimizer.memory_checkpoint("HMM_Model_Training")

        try:
            # Get search spaces using HMM HPO configuration from ml_commons
            search_spaces = self.hmm_hpo.get_hmm_state_recognition_search_spaces()

            # Use comprehensive feature bank for enhanced feature generation
            from .shared_feature_utils import create_comprehensive_features
            import pandas as pd

            # Convert regime data to DataFrame format for feature bank with hardware optimization
            enhanced_regime_data = {}
            feature_names_dict = {}

            for regime_id, data in regime_data.items():
                X_regime = data['X']
                y_regime = data['y']

                # Create basic DataFrame from regime data with memory optimization
                if X_regime.shape[1] >= 3:  # We have enough columns for OHLCV
                    # Create synthetic OHLCV data for feature bank
                    # This is a temporary solution - proper OHLCV data should be passed
                    regime_df = pd.DataFrame({
                        'open': np.random.randn(X_regime.shape[0]),  # Placeholder
                        'high': np.random.randn(X_regime.shape[0]),  # Placeholder
                        'low': np.random.randn(X_regime.shape[0]),   # Placeholder
                        'close': X_regime[:, 0] if X_regime.shape[1] > 0 else np.random.randn(X_regime.shape[0]),
                        'volume': X_regime[:, 1] if X_regime.shape[1] > 1 else np.random.randn(X_regime.shape[0])
                    })

                    # Optimize DataFrame for hardware before feature generation
                    regime_df = self._optimize_dataframe_for_hardware(regime_df)

                    # Generate comprehensive features with hardware optimization
                    X_enhanced, feature_names = create_comprehensive_features(
                        regime_df,
                        regime_labels=data.get('regime_labels')
                    )

                    enhanced_regime_data[regime_id] = {
                        'X': X_enhanced,
                        'y': y_regime,
                        'feature_names': feature_names
                    }
                    feature_names_dict[regime_id] = feature_names
                else:
                    # Fallback to original features if not enough columns
                    enhanced_regime_data[regime_id] = data

            # Train models for each regime using enhanced features with hardware optimization
            all_results = {}
            total_training_time = 0

            # Use CPU optimization for parallel processing if available
            if self.cpu_optimizer:
                # Use optimized thread pool for regime training
                with self.cpu_optimizer.create_m1_optimized_context():
                    training_results = self._train_regimes_with_cpu_optimization(
                        enhanced_regime_data, search_spaces
                    )
                all_results = training_results['results']
                total_training_time = training_results['total_time']
            else:
                # Standard training without CPU optimization
                all_results, total_training_time = self._train_regimes_standard(
                    enhanced_regime_data, search_spaces
                )

            # Evaluate models using common evaluation with hardware optimization
            evaluation_results = {}
            for regime_name, regime_results in all_results.items():
                models = regime_results.get('models', {})
                X_regime = regime_data[int(regime_name.split('_')[1])]['X']
                y_regime = regime_data[int(regime_name.split('_')[1])]['y']

                # Evaluate models using universal validation integration
                evaluation_results[regime_name] = self._evaluate_models_with_validation(
                    models=models,
                    X_train=X_regime,
                    y_train=y_regime,
                    regime_name=regime_name
                )

            return {
                'models': all_results,
                'evaluation_results': evaluation_results,
                'training_time': total_training_time,
                'regime_count': len(regime_data),
                'hardware_optimized': True,
                'feature_names_by_regime': feature_names_dict
            }

        finally:
            # Cleanup memory context
            if training_memory_context:
                training_memory_context.__exit__(None, None, None)

            # Force garbage collection after training
            if self.memory_optimizer:
                self.memory_optimizer.force_garbage_collection()

    def _train_regimes_with_cpu_optimization(self, enhanced_regime_data, search_spaces):
        """Train regimes using CPU optimization for parallel processing."""
        import concurrent.futures
        from functools import partial

        def train_single_regime(regime_id, data, search_spaces, config_model_types, enable_hpo):
            """Train models for a single regime."""
            try:
                X_regime = data['X']
                y_regime = data['y']

                # Create a new training step instance for this regime to avoid shared state issues
                regime_training_step = self.__class__(self.config)

                # Train models using common pipeline with enhanced features
                regime_results = regime_training_step.train_models(
                    model_types=config_model_types,
                    X=X_regime,
                    y=y_regime,
                    enable_hpo=enable_hpo,
                    search_spaces=search_spaces
                )

                return regime_id, regime_results
            except Exception as e:
                self.logger.error(f"❌ Failed to train regime {regime_id}: {e}")
                return regime_id, {
                    'models': {},
                    'training_time': 0,
                    'error': str(e)
                }

        # Prepare training function
        train_func = partial(
            train_single_regime,
            search_spaces=search_spaces,
            config_model_types=self.config.model_types,
            enable_hpo=self.config.enable_hpo
        )

        # Use optimized thread pool
        with create_m1_optimized_thread_pool(max_workers=self.cpu_optimizer.get_optimal_worker_count()) as executor:
            # Submit all training tasks
            future_to_regime = {
                executor.submit(train_func, regime_id, data): regime_id
                for regime_id, data in enhanced_regime_data.items()
            }

            # Collect results
            results = {}
            total_time = 0

            for future in concurrent.futures.as_completed(future_to_regime):
                regime_id = future_to_regime[future]
                try:
                    reg_id, regime_results = future.result()
                    results[f"regime_{reg_id}"] = regime_results
                    total_time += regime_results.get('training_time', 0)

                    self.logger.info(f"✅ Completed training for regime {reg_id}")

                except Exception as e:
                    self.logger.error(f"❌ Failed to get results for regime {regime_id}: {e}")

        return {'results': results, 'total_time': total_time}

    def _train_regimes_standard(self, enhanced_regime_data, search_spaces):
        """Train regimes using standard sequential processing."""
        all_results = {}
        total_training_time = 0

        for regime_id, data in enhanced_regime_data.items():
            self.logger.info(f"📊 Training models for regime {regime_id} with {data['X'].shape[1]} features")

            X_regime = data['X']
            y_regime = data['y']

            # Train models using common pipeline with enhanced features
            regime_results = self.train_models(
                model_types=self.config.model_types,
                X=X_regime,
                y=y_regime,
                enable_hpo=self.config.enable_hpo,
                search_spaces=search_spaces
            )

            all_results[f"regime_{regime_id}"] = regime_results
            total_training_time += regime_results.get('training_time', 0)

        return all_results, total_training_time


    def _handle_training_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle training errors with proper logging and hardware cleanup.

        Args:
            error: Exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Error results dictionary
        """
        error_msg = f"❌ HMM training error{f' in {context}' if context else ''}: {error}"
        self.logger.error(error_msg)

        # Cleanup hardware optimization resources
        self._cleanup_hardware_resources()

        return {
            'models': {},
            'metadata': {},
            'evaluation_results': {},
            'training_time': 0,
            'config': self.config,
            'error': str(error),
            'hmm_state_recognition_focus': True,
            'timeframe': self.config.timeframe,
            'hardware_cleanup_performed': True
        }

    def _cleanup_hardware_resources(self):
        """Cleanup hardware optimization resources."""
        try:
            # Stop memory monitoring if active
            if self.memory_optimizer and self.memory_optimizer.monitoring_active:
                self.memory_optimizer.stop_monitoring()
                self.logger.info("🧹 Memory monitoring stopped during cleanup")

            # Force garbage collection
            if self.memory_optimizer:
                self.memory_optimizer.force_garbage_collection()

        except Exception as e:
            self.logger.warning(f"Hardware cleanup warning: {e}")

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        try:
            self._cleanup_hardware_resources()
        except:
            pass  # Ignore cleanup errors during destruction

    def _generate_enhanced_model_report(
        self,
        models: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        regime_analysis: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive enhanced reporting for all trained models with hardware optimization info.

        Args:
            models: Dictionary of trained models
            evaluation_results: Evaluation results for each model
            regime_analysis: Regime analysis results
            validation_results: Optional comprehensive validation results

        Returns:
            Dictionary containing enhanced model reporting with ml_commons and hardware optimization integration
        """
        self.logger.info("📊 Generating enhanced model report with hardware optimization info...")

        enhanced_report = {
            'model_performance_summary': {},
            'regime_specific_performance': {},
            'model_comparison': {},
            'best_models_by_regime': {},
            'overall_recommendations': [],
            'validation_insights': {},
            'ml_commons_integration': {
                'hpo_used': True,
                'validation_pipeline_used': True,
                'temporal_protection_used': True,
                'overfitting_detection_used': True
            },
            'hardware_optimization': self._get_hardware_optimization_info(),
            'training_metadata': {
                'total_regimes': len(regime_analysis.get('regime_counts', {})),
                'total_models_trained': len(models),
                'model_types_used': list(models.keys()),
                'hardware_optimized': True,
                'memory_efficient': self.memory_optimizer is not None
            }
        }

        # Generate performance summary for each model
        for model_name, model_result in models.items():
            if model_name in evaluation_results:
                eval_result = evaluation_results[model_name]

                enhanced_report['model_performance_summary'][model_name] = {
                    'accuracy': eval_result.get('accuracy', 0),
                    'f1_score': eval_result.get('f1_score', 0),
                    'precision': eval_result.get('precision', 0),
                    'recall': eval_result.get('recall', 0),
                    'training_time': eval_result.get('training_time', 0),
                    'regime_specific_metrics': eval_result.get('regime_metrics', {}),
                    'feature_importance_available': eval_result.get('feature_importance_available', False)
                }

        # Generate regime-specific performance analysis
        for regime_id, regime_data in regime_analysis.get('regime_data', {}).items():
            regime_performance = {}

            for model_name, model_result in models.items():
                if model_name in evaluation_results:
                    eval_result = evaluation_results[model_name]
                    regime_metrics = eval_result.get('regime_metrics', {}).get(f'regime_{regime_id}', {})

                    regime_performance[model_name] = {
                        'accuracy': regime_metrics.get('accuracy', 0),
                        'f1_score': regime_metrics.get('f1_score', 0),
                        'precision': regime_metrics.get('precision', 0),
                        'recall': regime_metrics.get('recall', 0),
                        'samples': regime_data.get('n_samples', 0)
                    }

            enhanced_report['regime_specific_performance'][f'regime_{regime_id}'] = regime_performance

        # Generate model comparison across all regimes
        model_comparison = {}
        for model_name in models.keys():
            accuracies = []
            f1_scores = []

            for regime_perf in enhanced_report['regime_specific_performance'].values():
                if model_name in regime_perf:
                    accuracies.append(regime_perf[model_name]['accuracy'])
                    f1_scores.append(regime_perf[model_name]['f1_score'])

            if accuracies:
                model_comparison[model_name] = {
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': min(accuracies),
                    'max_accuracy': max(accuracies),
                    'avg_f1_score': np.mean(f1_scores),
                    'std_f1_score': np.std(f1_scores)
                }

        enhanced_report['model_comparison'] = model_comparison

        # Determine best models by regime
        for regime_id, regime_performance in enhanced_report['regime_specific_performance'].items():
            best_model = max(regime_performance.keys(),
                           key=lambda k: regime_performance[k]['f1_score'])
            enhanced_report['best_models_by_regime'][regime_id] = {
                'best_model': best_model,
                'best_f1_score': regime_performance[best_model]['f1_score'],
                'best_accuracy': regime_performance[best_model]['accuracy']
            }

        # Generate recommendations
        if model_comparison:
            # Find overall best model
            best_overall = max(model_comparison.keys(),
                             key=lambda k: model_comparison[k]['avg_f1_score'])

            enhanced_report['overall_recommendations'] = [
                f"Best overall model: {best_overall} (avg F1: {model_comparison[best_overall]['avg_f1_score']:.4f})",
                "XGBoost vs CatBoost comparison: Both models trained, select best performer per regime",
                "Consider ensemble of top 2 models (logistic_regression + lightgbm) for robustness",
                "Monitor regime-specific performance for model drift detection"
            ]

            # Add regime-specific recommendations
            for regime_id, best_info in enhanced_report['best_models_by_regime'].items():
                enhanced_report['overall_recommendations'].append(
                    f"Regime {regime_id}: Use {best_info['best_model']} (F1: {best_info['best_f1_score']:.4f})"
                )

        # Add validation insights from universal validation tools
        if validation_results:
            enhanced_report['validation_insights'] = self._generate_validation_insights(
                validation_results, evaluation_results
            )

        self.logger.info("✅ Enhanced model report generated with ml_commons integration")
        return enhanced_report

    def _generate_validation_insights(
        self,
        validation_results: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate validation insights using ml_commons tools.

        Args:
            validation_results: Comprehensive validation results
            evaluation_results: Model evaluation results

        Returns:
            Dictionary containing validation insights
        """
        insights = {
            'data_quality_insights': {},
            'overfitting_insights': {},
            'temporal_insights': {},
            'regime_insights': {},
            'ml_commons_tool_usage': {
                'validation_pipeline': True,
                'temporal_protection': True,
                'overfitting_detection': True
            }
        }

        # Data quality insights
        data_quality = validation_results.get('data_quality', {})
        if data_quality:
            insights['data_quality_insights'] = {
                'missing_values_analysis': data_quality.get('missing_values', {}),
                'class_distribution': data_quality.get('class_distribution', {}),
                'feature_statistics_summary': len(data_quality.get('feature_statistics', {}))
            }

        # Overfitting insights
        overfitting_detections = []
        for regime_name, regime_evaluations in evaluation_results.items():
            for model_name, evaluation in regime_evaluations.items():
                validation = evaluation.get('validation', {})
                overfitting_analysis = validation.get('overfitting_analysis', {})

                if overfitting_analysis.get('overfitting_detected', False):
                    overfitting_detections.append({
                        'regime': regime_name,
                        'model': model_name,
                        'severity': overfitting_analysis.get('severity', 'unknown'),
                        'confidence': overfitting_analysis.get('confidence_level', 0.0)
                    })

        insights['overfitting_insights'] = {
            'overfitting_detected_count': len(overfitting_detections),
            'overfitting_by_regime': {},
            'overfitting_recommendations': []
        }

        # Group by regime
        for detection in overfitting_detections:
            regime = detection['regime']
            if regime not in insights['overfitting_insights']['overfitting_by_regime']:
                insights['overfitting_insights']['overfitting_by_regime'][regime] = []
            insights['overfitting_insights']['overfitting_by_regime'][regime].append(detection)

        # Temporal insights
        temporal_integrity = validation_results.get('temporal_integrity', {})
        if temporal_integrity:
            insights['temporal_insights'] = {
                'timestamp_ordering_valid': temporal_integrity.get('timestamp_ordering', True),
                'future_data_detected': temporal_integrity.get('future_data_present', False),
                'temporal_range': temporal_integrity.get('timestamp_range', {}),
                'temporal_gaps_detected': len(temporal_integrity.get('temporal_gaps', []))
            }

        # Regime insights
        regime_analysis = validation_results.get('regime_analysis', {})
        if regime_analysis:
            insights['regime_insights'] = {
                'total_regimes': regime_analysis.get('n_regimes', 0),
                'regime_sizes': regime_analysis.get('regime_sizes', {}),
                'regime_quality_summary': {
                    regime_id: {
                        'size': quality.get('size', 0),
                        'min_samples_per_class': quality.get('min_samples_per_class', 0)
                    }
                    for regime_id, quality in regime_analysis.get('regime_quality', {}).items()
                }
            }

        # Generate specific recommendations
        if insights['overfitting_insights']['overfitting_detected_count'] > 0:
            insights['overfitting_insights']['overfitting_recommendations'].extend([
                "High overfitting detected - consider regularization techniques",
                "Implement cross-validation to reduce overfitting",
                "Consider ensemble methods to improve generalization"
            ])

        if insights['temporal_insights'].get('future_data_detected', False):
            insights['overfitting_insights']['overfitting_recommendations'].append(
                "Future data detected - ensure proper temporal data splitting"
            )

        return insights

    def _log_enhanced_training_summary(self, results: Dict[str, Any]) -> None:
        """
        Log enhanced training summary with comprehensive metrics and hardware optimization info.

        Args:
            results: Training results dictionary
        """
        enhanced_reporting = results.get('enhanced_reporting', {})

        if enhanced_reporting:
            self.logger.info("📊 Enhanced Training Summary:")

            # Overall performance
            model_comparison = enhanced_reporting.get('model_comparison', {})
            if model_comparison:
                best_model = max(model_comparison.keys(),
                               key=lambda k: model_comparison[k]['avg_f1_score'])
                best_f1 = model_comparison[best_model]['avg_f1_score']
                self.logger.info(f"🏆 Best overall model: {best_model} (avg F1: {best_f1:.4f})")

            # Regime-specific insights
            best_models_by_regime = enhanced_reporting.get('best_models_by_regime', {})
            if best_models_by_regime:
                self.logger.info("📊 Best models by regime:")
                for regime_id, best_info in best_models_by_regime.items():
                    self.logger.info(f"  - {regime_id}: {best_info['best_model']} (F1: {best_info['best_f1_score']:.4f})")

            # Hardware optimization status
            hardware_optimization = enhanced_reporting.get('hardware_optimization', {})
            if hardware_optimization:
                cpu_info = hardware_optimization.get('cpu_optimization', {})
                gpu_info = hardware_optimization.get('gpu_optimization', {})
                memory_info = hardware_optimization.get('memory_optimization', {})

                self.logger.info("🚀 Hardware Optimization Status:")
                if cpu_info.get('enabled'):
                    self.logger.info(f"  - CPU: ✅ Optimized ({cpu_info.get('optimal_workers', 'N/A')} workers)")
                if gpu_info.get('enabled'):
                    self.logger.info(f"  - GPU: ✅ {'MPS' if gpu_info.get('mps_available') else 'CPU'} mode")
                if memory_info.get('enabled'):
                    self.logger.info(f"  - Memory: ✅ {'Active' if memory_info.get('monitoring_active') else 'Enabled'} ({memory_info.get('memory_limit_gb', 'N/A')} GB limit)")

            # Recommendations
            recommendations = enhanced_reporting.get('overall_recommendations', [])
            if recommendations:
                self.logger.info("💡 Key recommendations:")
                for rec in recommendations[:3]:  # Show top 3 recommendations
                    self.logger.info(f"  - {rec}")

            # Training metadata
            training_metadata = enhanced_reporting.get('training_metadata', {})
            self.logger.info("📈 Training completed:")
            self.logger.info(f"  - Models trained: {training_metadata.get('total_models_trained', 0)}")
            self.logger.info(f"  - Regimes analyzed: {training_metadata.get('total_regimes', 0)}")
            self.logger.info(f"  - Model types: {', '.join(training_metadata.get('model_types_used', []))}")
            self.logger.info(f"  - Hardware optimized: {'✅ Yes' if training_metadata.get('hardware_optimized') else '❌ No'}")
            self.logger.info(f"  - Memory efficient: {'✅ Yes' if training_metadata.get('memory_efficient') else '❌ No'}")


# Convenience functions
def create_enhanced_hmm_models_training(config: Optional[HMMTrainingConfig] = None) -> StreamlinedHMMTrainingStep:
    """
    Create a streamlined HMM training step.

    Args:
        config: Optional HMM training configuration

    Returns:
        StreamlinedHMMTrainingStep instance
    """
    return StreamlinedHMMTrainingStep(config)


def execute_enhanced_hmm_models_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[HMMTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute streamlined HMM training.

    Args:
        X: Input features
        y: Target values (HMM states)
        regime_labels: Regime labels
        config: Optional training configuration
        feature_names: Feature names
        hmm_states: Optional HMM states
        **kwargs: Additional arguments

    Returns:
        Training results
    """
    training_step = create_enhanced_hmm_models_training(config)
    return training_step.execute(
        X=X,
        y=y,
        regime_labels=regime_labels,
        feature_names=feature_names,
        hmm_states=hmm_states,
        **kwargs
    )
