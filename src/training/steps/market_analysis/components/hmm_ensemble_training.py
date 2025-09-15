"""
HMM Ensemble Training Component.

This component trains HMM ensemble (meta-model) with hyperparameter optimization.
Enhanced with comprehensive validation, progress tracking, and robust error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Handle optional dependencies gracefully with detailed error tracking
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    NUMPY_AVAILABLE = False
    np = None
    NUMPY_ERROR = str(e)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    PANDAS_AVAILABLE = False
    pd = None
    PANDAS_ERROR = str(e)

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


# Custom exception classes for better error handling
class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class DependencyError(Exception):
    """Raised when required dependencies are missing."""
    pass


class TrainingError(Exception):
    """Raised when training process fails."""
    pass


class TrainingProgressTracker:
    """Real-time progress tracking for training steps."""
    
    def __init__(self, total_steps: int, logger: logging.Logger):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.logger = logger
        self.step_times = []
    
    def __enter__(self):
        self.logger.info(f"🚀 Starting training with {self.total_steps} steps")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = time.time() - self.start_time
        self.logger.info(f"✅ Training completed in {total_time:.2f}s")
    
    def update_progress(self, step_name: str, metrics: Optional[Dict[str, float]] = None):
        """Update progress with step information and metrics."""
        step_start = time.time()
        self.current_step += 1
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        # Calculate estimated time remaining
        if self.current_step > 1:
            avg_step_time = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time
            eta_str = f", ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        # Format metrics
        metrics_str = ""
        if metrics:
            metrics_str = " - " + ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        
        self.logger.info(
            f"🔄 [{progress_pct:.1f}%] {step_name}{metrics_str} "
            f"(Elapsed: {elapsed:.1f}s{eta_str})"
        )
        
        # Store step timing
        step_time = time.time() - step_start
        self.step_times.append(step_time)


class HMMEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    HMM Ensemble Training Component.
    
    Trains HMM ensemble (meta-model) with hyperparameter optimization.
    Enhanced with comprehensive validation, progress tracking, and robust error handling.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM ensemble training component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMEnsembleTraining')
        
        # Validate dependencies
        self._validate_dependencies()
        
        # Configuration validation thresholds
        self.min_accuracy_threshold = 0.6
        self.max_overfitting_ratio = 0.1
        
        self.logger.info("✅ HMM Ensemble Training Component initialized with enhanced validation")
    
    def _validate_dependencies(self):
        """Validate that all required dependencies are available."""
        missing_deps = []
        
        if not NUMPY_AVAILABLE:
            missing_deps.append(f"numpy: {NUMPY_ERROR}")
        if not PANDAS_AVAILABLE:
            missing_deps.append(f"pandas: {PANDAS_ERROR}")
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
            self.logger.error(f"❌ {error_msg}")
            raise DependencyError(error_msg)
    
    def validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Comprehensive input validation with detailed error reporting."""
        errors = []
        warnings = []
        
        # Data validation
        if data is None:
            errors.append("Input data is None")
        elif hasattr(data, 'empty') and data.empty:
            errors.append("Input data is empty")
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            if len(data) < 100:
                warnings.append(f"Dataset is small ({len(data)} rows), may affect training quality")
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            if missing_pct > 5:
                warnings.append(f"High missing value percentage: {missing_pct:.1f}%")
        
        # Pipeline state validation
        hmm_models_training = pipeline_state.get('hmm_models_training_result', {})
        if not hmm_models_training:
            errors.append("No HMM base models available in pipeline state")
        elif not isinstance(hmm_models_training, dict):
            errors.append("HMM models training result is not a dictionary")
        
        return {"errors": errors, "warnings": warnings}
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM ensemble training with enhanced validation and progress tracking.
        
        Args:
            data: Market data for training
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with ensemble training results
        """
        self.logger.info('🎭 Starting Enhanced HMM Ensemble Training')
        
        try:
            # Step 1: Validate inputs
            validation = self.validate_inputs(data, pipeline_state)
            if validation["errors"]:
                error_msg = f"Input validation failed: {'; '.join(validation['errors'])}"
                self.logger.error(f"❌ {error_msg}")
                raise ValidationError(error_msg)
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    self.logger.warning(f"⚠️ {warning}")
            
            # Step 2: Execute training with progress tracking
            with TrainingProgressTracker(total_steps=6, logger=self.logger) as tracker:
                result = await self._execute_with_tracking(data, pipeline_state, tracker)
            
            return result
            
        except ValidationError as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Validation error: {e}"
            )
        except DependencyError as e:
            self.logger.error(f"❌ Dependency error: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Dependency error: {e}"
            )
        except TrainingError as e:
            self.logger.error(f"❌ Training failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Training error: {e}"
            )
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            import traceback
            self.logger.error(f"❌ Error details: {traceback.format_exc()}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Unexpected error: {e}"
            )
    
    async def _execute_with_tracking(
        self, 
        data: Any, 
        pipeline_state: Dict[str, Any], 
        tracker: TrainingProgressTracker
    ) -> ComponentResult:
        """Execute training with progress tracking."""
        
        # Step 1: Load and prepare data
        tracker.update_progress("Loading and preparing data")
        market_data = await self._load_market_data(data)
        if market_data is None:
            raise TrainingError("Failed to load market data")
        
        # Step 2: Get base models
        tracker.update_progress("Retrieving base models")
        hmm_models_training = pipeline_state.get('hmm_models_training_result', {})
        if not hmm_models_training:
            raise TrainingError("No HMM base models available")
        
        # Step 3: Initialize HMM manager
        tracker.update_progress("Initializing HMM composite manager")
        try:
            from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
            hmm_manager = EnhancedHMMCompositeManager()
        except ImportError as e:
            raise TrainingError(f"Failed to import HMM composite manager: {e}")
        
        # Step 4: Prepare ensemble configuration
        tracker.update_progress("Configuring ensemble training")
        ensemble_config = self._prepare_ensemble_config()
        
        # Step 5: Perform ensemble training
        tracker.update_progress("Training ensemble models")
        ensemble_result = await self._perform_ensemble_training(
            hmm_manager, market_data, hmm_models_training, ensemble_config
        )
        
        # Step 6: Validate and format results
        tracker.update_progress("Validating and formatting results")
        artifacts = self._create_enhanced_artifacts(ensemble_result, market_data)
        
        # Validate artifacts
        if not self.validate_artifacts(artifacts):
            raise TrainingError("Generated artifacts failed validation")
        
        self.logger.info(f'✅ HMM Ensemble Training completed successfully')
        return ComponentResult(
            success=True,
            artifacts=artifacts,
            metadata={
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'ensemble_models_trained': len(artifacts.get('hmm_ensemble_training_result', {}).get('hmm_ensemble_models', [])),
                'execution_time': time.time() - tracker.start_time
            }
        )
    
    def _prepare_ensemble_config(self) -> Dict[str, Any]:
        """Prepare ensemble configuration with enhanced settings."""
        return {
            'ensemble_methods': ['voting', 'stacking', 'bagging'],
            'meta_models': ['random_forest', 'gradient_boosting', 'neural_network'],
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            
            # Hyperparameter optimization
            'enable_hpo': True,
            'hpo_method': 'bayesian_optimization',
            'n_trials': 30,
            'optimization_metric': 'accuracy',
            
            # Hardware optimization
            'enable_parallel_processing': True,
            'enable_gpu_acceleration': True,
            'memory_limit_gb': 8.0,
            
            # Enhanced validation
            'min_accuracy_threshold': self.min_accuracy_threshold,
            'max_overfitting_ratio': self.max_overfitting_ratio
        }
    
    def _create_enhanced_artifacts(self, ensemble_result: Dict[str, Any], market_data: Any) -> Dict[str, Any]:
        """Create enhanced artifacts with comprehensive validation and reporting."""
        hmm_ensemble_models = ensemble_result.get('hmm_ensemble_models', [])
        ensemble_metrics = ensemble_result.get('ensemble_metrics', {})
        hpo_results = ensemble_result.get('hpo_results', {})
        
        # Validate that we have ensemble results
        if not hmm_ensemble_models:
            raise TrainingError("HMM ensemble training completed but no ensemble models were trained")
        
        # Create comprehensive artifact
        artifacts = {
            'hmm_ensemble_training_result': {
                'hmm_ensemble_models': hmm_ensemble_models,
                'ensemble_metrics': ensemble_metrics,
                'hpo_results': hpo_results,
                'ensemble_summary': {
                    'total_ensemble_models': len(hmm_ensemble_models),
                    'best_ensemble_method': ensemble_metrics.get('best_ensemble_method', 'unknown'),
                    'best_accuracy': ensemble_metrics.get('best_accuracy', 0.0),
                    'ensemble_training_time': ensemble_result.get('ensemble_training_time', 0.0),
                    'hpo_trials': hpo_results.get('n_trials', 0),
                    'performance_validation': {
                        'meets_accuracy_threshold': ensemble_metrics.get('best_accuracy', 0.0) >= self.min_accuracy_threshold,
                        'overfitting_detected': self._detect_overfitting(ensemble_metrics),
                        'model_stability': self._assess_model_stability(ensemble_metrics)
                    }
                },
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data) if market_data is not None else 0,
                    'execution_timestamp': datetime.now().isoformat(),
                    'validation_summary': {
                        'dependencies_validated': True,
                        'input_validation_passed': True,
                        'performance_thresholds_met': ensemble_metrics.get('best_accuracy', 0.0) >= self.min_accuracy_threshold
                    }
                }
            }
        }
        
        return artifacts
    
    def _detect_overfitting(self, ensemble_metrics: Dict[str, Any]) -> bool:
        """Detect potential overfitting based on metrics."""
        train_accuracy = ensemble_metrics.get('train_accuracy', 0.0)
        test_accuracy = ensemble_metrics.get('test_accuracy', 0.0)
        
        if train_accuracy > 0 and test_accuracy > 0:
            overfitting_ratio = (train_accuracy - test_accuracy) / train_accuracy
            return overfitting_ratio > self.max_overfitting_ratio
        
        return False
    
    def _assess_model_stability(self, ensemble_metrics: Dict[str, Any]) -> float:
        """Assess model stability based on cross-validation scores."""
        cv_scores = ensemble_metrics.get('cv_scores', [])
        if cv_scores and len(cv_scores) > 1 and NUMPY_AVAILABLE:
            return 1.0 - (np.std(cv_scores) / np.mean(cv_scores))
        return 0.0
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for ensemble training with enhanced validation."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            # Create a copy to avoid modifying original data
            data_copy = data.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data_copy.columns:
                    if col == 'volume':
                        data_copy[col] = 1000  # Default volume
                        self.logger.warning(f"Added default volume column")
                    else:
                        data_copy[col] = data_copy.get('close', 100.0)
                        self.logger.warning(f"Added {col} column using close price")
            
            return data_copy
        
        # Handle other data types if needed
        return data
    
    async def _perform_ensemble_training(
        self, 
        hmm_manager: Any, 
        market_data: Any, 
        hmm_models_training: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual HMM ensemble training process with enhanced error handling."""
        try:
            # Prepare data for ensemble training
            prepared_data = self._prepare_data_for_ensemble_training(market_data, hmm_models_training)
            
            # Perform HMM ensemble training with HPO
            ensemble_result = await hmm_manager.train_hmm_ensemble(prepared_data, config)
            
            # Validate ensemble result
            if not ensemble_result or 'hmm_ensemble_models' not in ensemble_result:
                raise TrainingError("HMM ensemble training returned invalid result")
            
            ensemble_models = ensemble_result.get('hmm_ensemble_models', [])
            if not ensemble_models:
                raise TrainingError("No ensemble models were trained")
            
            # Check performance thresholds
            ensemble_metrics = ensemble_result.get('ensemble_metrics', {})
            best_accuracy = ensemble_metrics.get('best_accuracy', 0.0)
            if best_accuracy < self.min_accuracy_threshold:
                self.logger.warning(
                    f"⚠️ Best ensemble accuracy ({best_accuracy:.3f}) is below threshold "
                    f"({self.min_accuracy_threshold})"
                )
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"HMM ensemble training process failed: {e}")
            raise TrainingError(f"Ensemble training failed: {e}")
    
    def _prepare_data_for_ensemble_training(self, data: Any, hmm_models_training: Dict[str, Any]) -> Any:
        """Prepare market data and base models for ensemble training with enhanced validation."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'hmm_models_training': hmm_models_training
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for ensemble training: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        # Validate data quality
        if PANDAS_AVAILABLE:
            data_quality = {
                'total_rows': len(data),
                'missing_values': data.isnull().sum().sum(),
                'duplicate_rows': data.duplicated().sum(),
                'data_completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            }
            
            if data_quality['data_completeness'] < 95:
                self.logger.warning(f"Data completeness is {data_quality['data_completeness']:.1f}%, may affect training quality")
            
            if data_quality['duplicate_rows'] > 0:
                self.logger.warning(f"Found {data_quality['duplicate_rows']} duplicate rows")
        
        return {
            'market_data': data,
            'hmm_models_training': hmm_models_training,
            'data_quality': data_quality if PANDAS_AVAILABLE else {}
        }