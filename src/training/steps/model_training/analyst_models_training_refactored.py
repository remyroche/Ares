"""
Analyst Models Training Step - Cleaned and Optimized

This step handles per-regime training of individual Analyst models using common dependencies.
Features:
- Fast-fail error handling
- Comprehensive datetime-stamped reports
- Clean, maintainable code structure
- Proper resource management
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep

logger = system_logger.getChild('AnalystModelsTrainingRefactored')


class AnalystModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Analyst Models Training Step with per-regime training, HPO, saving, and metrics.
    
    This is a refactored version that uses common dependencies to reduce code duplication.
    """
    
    def __init__(self, config: Optional[PerRegimeTrainingConfig] = None):
        """
        Initialize Analyst models training step with enhanced error handling.
        
        Args:
            config: Per-regime training configuration
        """
        # Set default configuration for analyst models
        if config is None:
            config = PerRegimeTrainingConfig(
                model_name="analyst_models",
                timeframe="5m",
                model_types=["TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING", "EXTRA_TREES"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/analyst_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
            )
        
        super().__init__(config)
        self.logger = logger.getChild('AnalystModelsTrainingStepRefactored')
        
        # Validate configuration
        if not self._validate_config(config):
            raise ValueError("Invalid configuration provided for Analyst models training")
        
        self.logger.info("✅ Analyst Models Training Step (Cleaned) initialized")
    
    def _validate_config(self, config: PerRegimeTrainingConfig) -> bool:
        """Validate configuration parameters."""
        try:
            if not config.model_name or not config.timeframe:
                self.logger.error("❌ Missing required config fields: model_name, timeframe")
                return False
            
            if not config.model_types:
                self.logger.error("❌ No model types specified")
                return False
            
            if config.hpo_n_trials <= 0:
                self.logger.error("❌ Invalid HPO trials: must be > 0")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Config validation failed: {e}")
            return False
    
    def _generate_datetime_stamp(self) -> str:
        """Generate a consistent datetime stamp for artifacts."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_training_report(
        self, 
        results: Dict[str, Any], 
        execution_time: float,
        status: str = "SUCCESS"
    ) -> str:
        """Create a comprehensive training report with datetime stamp."""
        timestamp = self._generate_datetime_stamp()
        report_filename = f"analyst_models_training_report_{timestamp}.json"
        report_path = f"{self.config.model_save_path}/reports/{report_filename}"
        
        # Ensure reports directory exists
        Path(f"{self.config.model_save_path}/reports").mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        report_data = {
            "metadata": {
                "model_name": self.config.model_name,
                "timeframe": self.config.timeframe,
                "timestamp": timestamp,
                "execution_time_seconds": execution_time,
                "status": status,
                "config": {
                    "model_types": self.config.model_types,
                    "hpo_n_trials": self.config.hpo_n_trials,
                    "hpo_timeout_seconds": self.config.hpo_timeout_seconds,
                    "min_samples_per_regime": self.config.min_samples_per_regime,
                    "enable_data_augmentation": self.config.enable_data_augmentation,
                    "augmentation_method": self.config.augmentation_method,
                    "evaluation_metrics": self.config.evaluation_metrics
                }
            },
            "results": results,
            "summary": {
                "models_trained": len(results.get('models', [])),
                "regimes_processed": len(results.get('regime_analysis', {}).get('unique_regimes', [])),
                "best_performing_model": results.get('best_models_per_regime', {}),
                "training_successful": status == "SUCCESS"
            }
        }
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"📋 Training report saved: {report_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save training report: {e}")
            report_path = None
        
        return report_path
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst models training step with enhanced error handling and reporting.
        
        Args:
            X: Input features
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            
        Returns:
            Dictionary containing training results and metadata
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting Analyst models training step (cleaned)")
        
        try:
            # Fast-fail: Validate input data
            if not self._validate_input_data(X, y, regime_labels):
                raise ValueError("Invalid input data provided")
            
            # VECTORIZED: Use ultra-fast vectorized training by default
            self.logger.info("🚀 Using VECTORIZED analyst models training")
            try:
                results = super().execute_vectorized(
                    X=X,
                    y=y,
                    regime_labels=regime_labels,
                    feature_names=feature_names,
                    hmm_states=hmm_states,
                    is_classification=False,  # Analyst models are typically regression
                    symbol=None,  # Can be passed as kwargs
                    exchange=None,
                    timeframe=self.config.timeframe
                )
                if results.get('vectorized', False):
                    self.logger.info("✅ VECTORIZED analyst training completed successfully")
                else:
                    self.logger.warning("⚠️ VECTORIZED analyst training failed, falling back to standard method")
                    results = super().execute(
                        X=X,
                        y=y,
                        regime_labels=regime_labels,
                        feature_names=feature_names,
                        hmm_states=hmm_states,
                        is_classification=False,  # Analyst models are typically regression
                        symbol=None,  # Can be passed as kwargs
                        exchange=None,
                        timeframe=self.config.timeframe
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ VECTORIZED analyst training failed: {e}, falling back to standard method")
                results = super().execute(
                    X=X,
                    y=y,
                    regime_labels=regime_labels,
                    feature_names=feature_names,
                    hmm_states=hmm_states,
                    is_classification=False,  # Analyst models are typically regression
                    symbol=None,  # Can be passed as kwargs
                    exchange=None,
                    timeframe=self.config.timeframe
                )
            
            # Add analyst-specific post-processing if needed
            if 'error' not in results:
                results = self._add_analyst_specific_metadata(results)
            
            # Create comprehensive training report
            execution_time = (datetime.now() - start_time).total_seconds()
            report_path = self._create_training_report(results, execution_time, "SUCCESS")
            if report_path:
                results['training_report'] = report_path
            
            self.logger.info(f"✅ Analyst models training completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Analyst models training failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            
            # Create failure report
            failure_results = {'error': error_msg, 'execution_time': execution_time}
            self._create_training_report(failure_results, execution_time, "FAILED")
            
            # Fast-fail: Re-raise the exception
            raise RuntimeError(error_msg) from e
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> bool:
        """Validate input data for training."""
        try:
            if X is None or y is None or regime_labels is None:
                self.logger.error("❌ Input data cannot be None")
                return False
            
            if len(X) != len(y) or len(X) != len(regime_labels):
                self.logger.error("❌ Input data length mismatch")
                return False
            
            if len(X) == 0:
                self.logger.error("❌ Input data is empty")
                return False
            
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                self.logger.error("❌ Input features contain NaN or infinite values")
                return False
            
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                self.logger.error("❌ Target values contain NaN or infinite values")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Input data validation failed: {e}")
            return False
    
    def _add_analyst_specific_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add analyst-specific metadata to results.
        
        Args:
            results: Training results
            
        Returns:
            Enhanced results with analyst-specific metadata
        """
        # Add analyst-specific analysis
        if 'regime_analysis' in results:
            regime_analysis = results['regime_analysis']
            
            # Calculate analyst-specific metrics
            analyst_metrics = {
                'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                'regime_balance': regime_analysis.get('regime_balance_train', 0.0)
            }
            
            results['analyst_metrics'] = analyst_metrics
        
        # Add model performance summary
        if 'evaluation_results' in results:
            evaluation_results = results['evaluation_results']
            
            # Calculate best performing model per regime
            best_models = {}
            for regime, regime_metrics in evaluation_results.items():
                if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                    best_model = None
                    best_r2 = -np.inf
                    
                    for model_name, metrics in regime_metrics.items():
                        if isinstance(metrics, dict) and 'r2' in metrics:
                            if metrics['r2'] > best_r2:
                                best_r2 = metrics['r2']
                                best_model = model_name
                    
                    if best_model:
                        best_models[regime] = {
                            'model': best_model,
                            'r2_score': best_r2
                        }
            
            results['best_models_per_regime'] = best_models
        
        return results


# Convenience functions
def create_analyst_models_training_step_refactored(
    config: Optional[PerRegimeTrainingConfig] = None
) -> AnalystModelsTrainingStepRefactored:
    """Create Analyst models training step (cleaned)."""
    return AnalystModelsTrainingStepRefactored(config)


def execute_analyst_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute Analyst models training step (cleaned) with fast-fail."""
    step = create_analyst_models_training_step_refactored(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states)