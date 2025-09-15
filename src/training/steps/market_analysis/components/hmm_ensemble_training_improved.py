"""
Improved HMM Ensemble Training Component.

This is a streamlined, robust version that addresses the issues identified in the original implementation:
- Prevents silent failures with comprehensive validation
- Enhanced reporting with real-time progress tracking
- Simplified configuration management
- Robust error handling with specific exception types
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import traceback

# Core dependencies with proper error handling
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


@dataclass
class ValidationResult:
    """Result of input validation."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {"errors": self.errors, "warnings": self.warnings}


@dataclass
class HMMEnsembleConfig:
    """Streamlined configuration for HMM ensemble training."""
    # Core ensemble settings
    ensemble_methods: List[str] = field(default_factory=lambda: ['stacking'])
    meta_model: str = 'XGBClassifier'
    base_models: List[str] = field(default_factory=lambda: ['wavenet', 'logistic_regression', 'hist_gradient_boosting'])
    
    # Training parameters
    hpo_trials: int = 30
    validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Performance settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Validation thresholds
    min_accuracy_threshold: float = 0.6
    max_overfitting_ratio: float = 0.1
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.hpo_trials < 1:
            raise ValueError("hpo_trials must be at least 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.min_accuracy_threshold < 1:
            raise ValueError("min_accuracy_threshold must be between 0 and 1")


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


class HMMEnsembleTrainingImproved(BaseMarketAnalysisComponent):
    """
    Improved HMM Ensemble Training Component.
    
    Features:
    - Comprehensive input validation
    - Real-time progress tracking
    - Robust error handling
    - Enhanced reporting
    - Fail-fast error propagation
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None, ensemble_config: Optional[HMMEnsembleConfig] = None):
        """Initialize the improved HMM ensemble training component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMEnsembleTrainingImproved')
        self.ensemble_config = ensemble_config or HMMEnsembleConfig()
        
        # Validate dependencies
        self._validate_dependencies()
        
        self.logger.info("✅ Improved HMM Ensemble Training initialized")
    
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
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']
    
    def validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> ValidationResult:
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
        
        # Configuration validation
        if not self.ensemble_config.ensemble_methods:
            errors.append("No ensemble methods specified in configuration")
        
        return ValidationResult(errors=errors, warnings=warnings)
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM ensemble training with comprehensive error handling.
        
        Args:
            data: Market data for training
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with ensemble training results
        """
        self.logger.info('🎭 Starting Improved HMM Ensemble Training')
        
        try:
            # Step 1: Validate inputs
            validation = self.validate_inputs(data, pipeline_state)
            if validation.has_errors():
                error_msg = f"Input validation failed: {'; '.join(validation.errors)}"
                self.logger.error(f"❌ {error_msg}")
                raise ValidationError(error_msg)
            
            if validation.has_warnings():
                for warning in validation.warnings:
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
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
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
        artifacts = self._create_artifacts(ensemble_result, market_data)
        
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
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for ensemble training."""
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
        
        return data
    
    def _prepare_ensemble_config(self) -> Dict[str, Any]:
        """Prepare ensemble configuration from component config."""
        return {
            'ensemble_methods': self.ensemble_config.ensemble_methods,
            'meta_models': [self.ensemble_config.meta_model],
            'cross_validation_folds': self.ensemble_config.validation_folds,
            'test_size': self.ensemble_config.test_size,
            'random_state': self.ensemble_config.random_state,
            
            # Hyperparameter optimization
            'enable_hpo': True,
            'hpo_method': 'bayesian_optimization',
            'n_trials': self.ensemble_config.hpo_trials,
            'optimization_metric': 'accuracy',
            
            # Hardware optimization
            'enable_parallel_processing': self.ensemble_config.enable_parallel_processing,
            'enable_gpu_acceleration': self.ensemble_config.enable_gpu_acceleration,
            'memory_limit_gb': self.ensemble_config.memory_limit_gb
        }
    
    async def _perform_ensemble_training(
        self, 
        hmm_manager: Any, 
        market_data: Any, 
        hmm_models_training: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual HMM ensemble training process."""
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
            if best_accuracy < self.ensemble_config.min_accuracy_threshold:
                self.logger.warning(
                    f"⚠️ Best ensemble accuracy ({best_accuracy:.3f}) is below threshold "
                    f"({self.ensemble_config.min_accuracy_threshold})"
                )
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"HMM ensemble training process failed: {e}")
            raise TrainingError(f"Ensemble training failed: {e}")
    
    def _prepare_data_for_ensemble_training(self, data: Any, hmm_models_training: Dict[str, Any]) -> Any:
        """Prepare market data and base models for ensemble training."""
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
        
        return {
            'market_data': data,
            'hmm_models_training': hmm_models_training
        }
    
    def _create_artifacts(self, ensemble_result: Dict[str, Any], market_data: Any) -> Dict[str, Any]:
        """Create standardized artifacts from ensemble training results."""
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
                        'meets_accuracy_threshold': ensemble_metrics.get('best_accuracy', 0.0) >= self.ensemble_config.min_accuracy_threshold,
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
                    'configuration': {
                        'ensemble_methods': self.ensemble_config.ensemble_methods,
                        'meta_model': self.ensemble_config.meta_model,
                        'hpo_trials': self.ensemble_config.hpo_trials,
                        'validation_folds': self.ensemble_config.validation_folds
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
            return overfitting_ratio > self.ensemble_config.max_overfitting_ratio
        
        return False
    
    def _assess_model_stability(self, ensemble_metrics: Dict[str, Any]) -> float:
        """Assess model stability based on cross-validation scores."""
        cv_scores = ensemble_metrics.get('cv_scores', [])
        if cv_scores and len(cv_scores) > 1:
            return 1.0 - (np.std(cv_scores) / np.mean(cv_scores))
        return 0.0


# Convenience functions for backward compatibility
def create_improved_hmm_ensemble_training(
    config: Optional[ComponentConfig] = None,
    ensemble_config: Optional[HMMEnsembleConfig] = None
) -> HMMEnsembleTrainingImproved:
    """Create improved HMM ensemble training component."""
    return HMMEnsembleTrainingImproved(config, ensemble_config)


# Example usage
if __name__ == "__main__":
    print("Improved HMM Ensemble Training Component")
    print("=" * 50)
    
    # Create configuration
    ensemble_config = HMMEnsembleConfig(
        ensemble_methods=['stacking'],
        meta_model='XGBClassifier',
        hpo_trials=50,
        validation_folds=5,
        min_accuracy_threshold=0.65
    )
    
    # Create component
    component = create_improved_hmm_ensemble_training(ensemble_config=ensemble_config)
    
    print(f"✅ Created improved component with {len(ensemble_config.ensemble_methods)} ensemble methods")
    print(f"📊 Meta-learner: {ensemble_config.meta_model}")
    print(f"📊 HPO trials: {ensemble_config.hpo_trials}")
    print(f"📊 Min accuracy threshold: {ensemble_config.min_accuracy_threshold}")
    
    print("\n🎯 Improvements over original version:")
    print("- ✅ Comprehensive input validation with detailed error reporting")
    print("- ✅ Real-time progress tracking with ETA estimation")
    print("- ✅ Robust error handling with specific exception types")
    print("- ✅ Fail-fast error propagation (no silent failures)")
    print("- ✅ Enhanced reporting with performance validation")
    print("- ✅ Streamlined configuration management")
    print("- ✅ Dependency validation with clear error messages")
    print("- ✅ Overfitting detection and model stability assessment")