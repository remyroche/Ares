"""Step 16: Confidence Calibration - Updated to use BaseStep pattern."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict
from src.core.decorators import handles_errors, log_execution_time
from .base_validation_step import BaseValidationStep
from copy import copy
from typing import Dict, List, Optional, Union, Any, Tuple

class ConfidenceCalibrationStep(BaseValidationStep):
    """Step 16: Confidence Calibration for model predictions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Confidence Calibration step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '16', 'confidence_calibration')

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.calibration_config = {'method': self.config.get('calibration_method', 'isotonic'), 'cv_folds': self.config.get('calibration_cv_folds', 3), 'ensemble_calibration': self.config.get('ensemble_calibration', True), 'regime_specific_calibration': self.config.get('regime_specific_calibration', True), 'min_samples_for_calibration': self.config.get('min_samples_for_calibration', 100)}
        self.calibrated_models: Dict[str, Any] = {}
        self.calibration_metrics: Dict[str, Dict[str, float]] = {}

    def _validate_step_specific_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> List[str]:
        """Validate step-specific inputs."""
        errors = []
        models = self._get_models_for_validation(pipeline_state)
        if not any((hasattr(model, 'predict_proba') for model in models.values())):
            errors.append('No models with probability prediction capability found')
        return errors

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='confidence calibration logic')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the confidence calibration logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with calibrated models
        """
        self.logger.info('🎯 Starting confidence calibration...')
        X_val, y_val = self._extract_validation_data(pipeline_state)
        if X_val.empty or len(y_val) == 0:
            self.logger.warning('No validation data available for calibration')
            return pipeline_state
        models = self._get_models_for_validation(pipeline_state)
        
        # Calibrate each model using time-aware CV (TimeSeriesSplit)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=max(2, int(self.calibration_config["cv_folds"])) )
        for model_name, model in models.items():
            if not hasattr(model, 'predict_proba'):
                self.logger.info(f'Skipping {model_name} - no probability prediction')
                continue
            
            self.logger.info(f"Calibrating {model_name}...")
            
            # Apply calibration
            calibrated_model, metrics = await self._calibrate_model(
                model, X_val, y_val, model_name, tscv
            )
            
            if calibrated_model is not None:
                self.calibrated_models[model_name] = calibrated_model
                self.calibration_metrics[model_name] = metrics
        if self.calibration_config['ensemble_calibration']:
            ensemble_calibration = await self._apply_ensemble_calibration(self.calibrated_models, X_val, y_val)
            if ensemble_calibration:
                self.calibrated_models.update(ensemble_calibration)
        result = pipeline_state.copy()
        result['calibrated_models'] = self.calibrated_models
        result[f'{self.full_step_name}_results'] = {'calibration_metrics': self.calibration_metrics, 'models_calibrated': len(self.calibrated_models), 'calibration_method': self.calibration_config['method']}
        result[f'{self.full_step_name}_summary'] = self._create_validation_summary({'model_results': self.calibration_metrics, 'overall_metrics': self._calculate_overall_calibration_metrics()})
        return result
    
    async def _calibrate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        tscv=None
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """Calibrate a single model.
        
        Args:
            model: Model to calibrate
            X: Features
            y: Labels
            model_name: Name of the model
            
        Returns:
            Tuple of (calibrated model, metrics)
        """
        metrics = {}
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
            metrics["pre_calibration_brier"] = brier_score_loss(y, y_pred_proba)
            metrics["pre_calibration_log_loss"] = log_loss(y, y_pred_proba)
            
            # Apply calibration using time-aware CV
            calibrated = CalibratedClassifierCV(
                model,
                method=self.calibration_config["method"],
                cv=tscv if tscv is not None else self.calibration_config["cv_folds"]
            )
            
            calibrated.fit(X, y)
            
            # Calculate post-calibration metrics
            # Evaluate on a holdout tail slice to avoid train reuse
            holdout_frac = 0.2
            n = len(X)
            split_idx = int(n * (1.0 - holdout_frac))
            X_holdout = X.iloc[split_idx:]
            y_holdout = y.iloc[split_idx:]
            y_cal_proba = calibrated.predict_proba(X_holdout)[:, 1]
            metrics["post_calibration_brier"] = brier_score_loss(y_holdout, y_cal_proba)
            metrics["post_calibration_log_loss"] = log_loss(y_holdout, y_cal_proba)
            
            # Calculate improvement
            metrics["brier_improvement"] = (
                metrics["pre_calibration_brier"] - metrics["post_calibration_brier"]
            ) / metrics["pre_calibration_brier"]
            
            metrics["log_loss_improvement"] = (
                metrics["pre_calibration_log_loss"] - metrics["post_calibration_log_loss"]
            ) / metrics["pre_calibration_log_loss"]
            
            self.logger.info(
                f"  Calibrated {model_name}: "
                f"Brier improvement: {metrics['brier_improvement']:.2%}, "
                f"Log loss improvement: {metrics['log_loss_improvement']:.2%}"
            )
            
            return calibrated, metrics
            
        except Exception as e:
            self.logger.error(f'Failed to calibrate {model_name}: {str(e)}')
            return (None, {'error': str(e)})

    async def _apply_ensemble_calibration(self, calibrated_models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Apply ensemble calibration techniques.
        
        Args:
            calibrated_models: Already calibrated models
            X: Features
            y: Labels
            
        Returns:
            Dictionary of ensemble calibrated models
        """
        ensemble_calibrated = {}
        if len(calibrated_models) > 1:
            try:
                predictions = []
                for model in calibrated_models.values():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                        predictions.append(pred)
                if predictions:
                    ensemble_pred = np.mean(predictions, axis=0)
                    temperature = self._find_optimal_temperature(ensemble_pred, y)
                    ensemble_calibrated['ensemble_temperature_scaled'] = {'models': calibrated_models, 'temperature': temperature, 'method': 'temperature_scaling'}
                    self.logger.info(f'Created temperature-scaled ensemble with T={temperature:.3f}')
            except Exception as e:
                self.logger.error(f'Failed to create ensemble calibration: {str(e)}')
        return ensemble_calibrated

    def _find_optimal_temperature(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal temperature for probability scaling.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Optimal temperature value
        """
        from scipy.optimize import minimize
from src.core.decorators.errors import handles_errors

        def temperature_loss(t: Any) -> None:
            scaled_probs = predictions / t
            scaled_probs = np.clip(scaled_probs, 1e-07, 1 - 1e-07)
            return log_loss(labels, scaled_probs)
        result = minimize(temperature_loss, x0=1.0, bounds=[(0.1, 10.0)])
        return result.x[0] if result.success else 1.0

    def _calculate_overall_calibration_metrics(self) -> Dict[str, float]:
        """Calculate overall calibration metrics."""
        metrics = {'avg_brier_improvement': [], 'avg_log_loss_improvement': [], 'n_calibrated': len(self.calibrated_models)}
        for model_metrics in self.calibration_metrics.values():
            if 'brier_improvement' in model_metrics:
                metrics['avg_brier_improvement'].append(model_metrics['brier_improvement'])
            if 'log_loss_improvement' in model_metrics:
                metrics['avg_log_loss_improvement'].append(model_metrics['log_loss_improvement'])
        if metrics['avg_brier_improvement']:
            metrics['avg_brier_improvement'] = np.mean(metrics['avg_brier_improvement'])
        else:
            metrics['avg_brier_improvement'] = 0.0
        if metrics['avg_log_loss_improvement']:
            metrics['avg_log_loss_improvement'] = np.mean(metrics['avg_log_loss_improvement'])
        else:
            metrics['avg_log_loss_improvement'] = 0.0
        return metrics

    def _validate_step_specific_outputs(self, pipeline_state: Dict[str, Any]) -> List[str]:
        """Validate step-specific outputs."""
        errors = []
        if 'calibrated_models' not in pipeline_state:
            errors.append('No calibrated models found in output')
        elif len(pipeline_state['calibrated_models']) == 0:
            errors.append('No models were successfully calibrated')
        return errors

    def _add_step_specific_summary(self, summary: Dict[str, Any], validation_results: Dict[str, Any]) -> None:
        """Add step-specific items to summary."""
        overall = validation_results.get('overall_metrics', {})
        if overall.get('avg_brier_improvement', 0) > 0:
            summary['key_findings'].append(f"Average Brier score improvement: {overall['avg_brier_improvement']:.2%}")
        if overall.get('avg_log_loss_improvement', 0) > 0:
            summary['key_findings'].append(f"Average log loss improvement: {overall['avg_log_loss_improvement']:.2%}")
        if overall.get('n_calibrated', 0) == 0:
            summary['warnings'].append('No models were successfully calibrated')
        if overall.get('avg_brier_improvement', 0) < 0:
            summary['recommendations'].append('Calibration worsened performance - consider using uncalibrated models')

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['tactician_specialist_models', 'features', 'step15_tactician_specialist_training_completed']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return ['calibrated_models', f'{self.full_step_name}_results', f'{self.full_step_name}_summary']

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['step15_tactician_specialist_training']