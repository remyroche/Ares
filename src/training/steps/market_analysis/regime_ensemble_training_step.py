"""
Regime Ensemble Training Step

BaseStep-based implementation for training regime detection ensemble models.
Migrated from the component pattern to use the new BaseStep architecture.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import warnings

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.logger import system_logger

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import ML libraries with error handling
ML_LIBRARIES_AVAILABLE = False
ML_LIBRARY_VERSIONS = {}
ML_IMPORT_ERRORS = []

try:
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    import sklearn
    ML_LIBRARY_VERSIONS['sklearn'] = sklearn.__version__
    tprint("✅ scikit-learn imported successfully", "SUCCESS")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"scikit-learn: {e}")
    tprint(f"❌ Failed to import scikit-learn: {e}", "ERROR")

try:
    import lightgbm as lgb
    ML_LIBRARY_VERSIONS['lightgbm'] = lgb.__version__
    tprint("✅ LightGBM imported successfully", "SUCCESS")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"LightGBM: {e}")
    tprint(f"❌ Failed to import LightGBM: {e}", "ERROR")

if not ML_IMPORT_ERRORS:
    ML_LIBRARIES_AVAILABLE = True
    tprint("🎉 All ML libraries imported successfully", "SUCCESS")
else:
    tprint(f"⚠️ Import errors: {ML_IMPORT_ERRORS}", "WARNING")


class RegimeEnsembleTrainingStep(BaseStep):
    """
    Regime Ensemble Training Step using BaseStep pattern.
    
    Trains ensemble models for regime detection:
    - Stacking ensemble with base models
    - LightGBM meta-learner with probability calibration
    """
    
    def __init__(self, step_name: str = "regime_ensemble_training"):
        """Initialize the regime ensemble training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('RegimeEnsembleTrainingStep')
        
        if not ML_LIBRARIES_AVAILABLE:
            tprint("⚠️ Some ML libraries not available - functionality may be limited", "WARNING")
        
        tprint("✅ RegimeEnsembleTrainingStep initialized", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime ensemble training step.
        
        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - train_data: Training data (optional)
                - regime_labels: Regime labels (optional)
                - base_models: Base models to use (optional)
                - ensemble_params: Ensemble parameters (optional)
                
        Returns:
            Dictionary with ensemble training results and model artifacts
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting regime ensemble training for {config.get('symbol', 'UNKNOWN')}", "INFO")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol', 'ETHUSDT'),
                exchange=config.get('exchange', 'binance'),
                direction=config.get('direction', 'both'),
                model=config.get('model', 'default')
            )
            
            # Load training data
            train_data = self._load_training_data(config)
            if train_data is None:
                raise ValueError("No training data found")
            
            tprint(f"✅ Loaded training data: {train_data.shape[0]} rows, {train_data.shape[1]} columns", "SUCCESS")
            
            # Load regime labels
            regime_labels = self._load_regime_labels(config)
            if regime_labels is None:
                raise ValueError("No regime labels found")
            
            tprint(f"✅ Loaded regime labels: {len(regime_labels)} labels", "SUCCESS")
            
            # Load base models
            base_models = self._load_base_models(config)
            if not base_models:
                raise ValueError("No base models found")
            
            tprint(f"✅ Loaded {len(base_models)} base models", "SUCCESS")
            
            # Prepare features and targets
            X, y = self._prepare_training_data(train_data, regime_labels)
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X, y, config)
            
            # Train ensemble
            ensemble_model = self._train_ensemble(base_models, X_train, y_train, config)
            
            # Evaluate ensemble
            evaluation_results = self._evaluate_ensemble(ensemble_model, X_test, y_test)
            
            # Save ensemble and results
            self._save_ensemble_and_results(ensemble_model, evaluation_results, config)
            
            # Calculate metrics
            metrics = self._calculate_ensemble_metrics(ensemble_model, evaluation_results, start_time, config)
            
            # Create outcome report
            outcome_report = self._create_outcome_report(ensemble_model, evaluation_results, metrics, config)
            
            tprint(f"✅ Regime ensemble training completed", "SUCCESS")
            
            return {
                'success': True,
                'ensemble_model': ensemble_model,
                'evaluation_results': evaluation_results,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            error_msg = f"Regime ensemble training failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'ensemble_model': None,
                'evaluation_results': {},
                'metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _load_training_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load training data from artifacts or config."""
        try:
            # Try to load from artifacts first
            train_data = self._load_dataframe('train_data')
            if train_data is not None:
                return train_data
            
            # Try alternative artifact names
            train_data = self._load_dataframe('training_data') or self._load_dataframe('market_data')
            if train_data is not None:
                return train_data
            
            # Try to load from config
            if 'train_data' in config:
                return pd.DataFrame(config['train_data'])
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load training data: {e}", "WARNING")
            return None
    
    def _load_regime_labels(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load regime labels from artifacts or config."""
        try:
            # Try to load from artifacts first
            regime_data = self._get_artifact('regime_labels')
            if regime_data is not None:
                if isinstance(regime_data, dict) and 'labels' in regime_data:
                    return np.array(regime_data['labels'])
                elif isinstance(regime_data, (list, np.ndarray)):
                    return np.array(regime_data)
            
            # Try to load from config
            if 'regime_labels' in config:
                return np.array(config['regime_labels'])
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load regime labels: {e}", "WARNING")
            return None
    
    def _load_base_models(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load base models from artifacts or config."""
        try:
            base_models = {}
            
            # Try to load from artifacts
            model_artifacts = ['catboost_model', 'extratrees_model', 'rulelist_model', 'lightgbm_model']
            for artifact_name in model_artifacts:
                try:
                    model = self._load_model(artifact_name)
                    if model is not None:
                        model_name = artifact_name.replace('_model', '')
                        base_models[model_name] = model
                except Exception as e:
                    tprint(f"⚠️ Could not load {artifact_name}: {e}", "WARNING")
            
            # Try to load from config
            if 'base_models' in config:
                for name, model in config['base_models'].items():
                    base_models[name] = model
            
            return base_models
            
        except Exception as e:
            tprint(f"⚠️ Failed to load base models: {e}", "WARNING")
            return {}
    
    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare features and targets for training."""
        try:
            # Remove non-numeric columns and handle missing values
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data = numeric_data.fillna(numeric_data.median())
            
            # Ensure we have the same number of samples
            min_length = min(len(numeric_data), len(regime_labels))
            X = numeric_data.iloc[:min_length]
            y = regime_labels[:min_length]
            
            # Remove any remaining NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            tprint(f"✅ Prepared training data: {X.shape[0]} samples, {X.shape[1]} features", "SUCCESS")
            
            return X, y
            
        except Exception as e:
            tprint(f"❌ Failed to prepare training data: {e}", "ERROR")
            raise
    
    def _split_data(self, X: pd.DataFrame, y: np.ndarray, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        try:
            from sklearn.model_selection import train_test_split
            
            test_size = config.get('test_size', 0.2)
            random_state = config.get('random_state', 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            tprint(f"✅ Data split: {len(X_train)} train, {len(X_test)} test", "SUCCESS")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            tprint(f"❌ Failed to split data: {e}", "ERROR")
            raise
    
    def _train_ensemble(self, base_models: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray, config: Dict[str, Any]) -> Any:
        """Train ensemble model."""
        try:
            if not ML_LIBRARIES_AVAILABLE:
                raise ValueError("Required ML libraries not available")
            
            # Get ensemble parameters
            ensemble_params = config.get('ensemble_params', {})
            
            # Create base estimators list for stacking
            base_estimators = []
            for name, model in base_models.items():
                base_estimators.append((name, model))
            
            if not base_estimators:
                raise ValueError("No base models available for ensemble")
            
            # Create meta-learner (LightGBM)
            meta_learner = lgb.LGBMClassifier(
                n_estimators=ensemble_params.get('n_estimators', 100),
                learning_rate=ensemble_params.get('learning_rate', 0.1),
                max_depth=ensemble_params.get('max_depth', 6),
                random_state=42,
                verbose=-1
            )
            
            # Create stacking ensemble
            tprint("🔗 Creating stacking ensemble...", "INFO")
            ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=ensemble_params.get('cv', 5),
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            # Train ensemble
            tprint("🚀 Training ensemble model...", "INFO")
            ensemble.fit(X_train, y_train)
            
            # Apply probability calibration if requested
            if ensemble_params.get('calibrate', True):
                tprint("🎯 Applying probability calibration...", "INFO")
                calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
                calibrated_ensemble.fit(X_train, y_train)
                ensemble = calibrated_ensemble
            
            tprint("✅ Ensemble training completed", "SUCCESS")
            
            return ensemble
            
        except Exception as e:
            tprint(f"❌ Failed to train ensemble: {e}", "ERROR")
            raise
    
    def _evaluate_ensemble(self, ensemble: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble model."""
        try:
            tprint("📊 Evaluating ensemble model...", "INFO")
            
            # Make predictions
            y_pred = ensemble.predict(X_test)
            y_pred_proba = ensemble.predict_proba(X_test) if hasattr(ensemble, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation score
            cv_scores = cross_val_score(ensemble, X_test, y_test, cv=3, scoring='accuracy')
            
            evaluation_results = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'classification_report': class_report,
                'n_test_samples': len(y_test)
            }
            
            tprint(f"✅ Ensemble evaluation completed (accuracy: {accuracy:.3f})", "SUCCESS")
            
            return evaluation_results
            
        except Exception as e:
            tprint(f"❌ Failed to evaluate ensemble: {e}", "ERROR")
            raise
    
    def _save_ensemble_and_results(self, ensemble: Any, evaluation_results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save ensemble model and evaluation results."""
        try:
            # Save ensemble model
            self._save_model('ensemble_model', ensemble)
            
            # Save evaluation results
            self._save_artifact('ensemble_evaluation_results', evaluation_results)
            
            # Save ensemble metadata
            ensemble_metadata = {
                'model_type': 'stacking_ensemble',
                'base_models': list(config.get('base_models', {}).keys()),
                'calibrated': config.get('ensemble_params', {}).get('calibrate', True),
                'training_timestamp': datetime.now().isoformat(),
                'config': config
            }
            self._save_metadata(ensemble_metadata)
            
            tprint("✅ Ensemble and results saved to artifacts", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save ensemble and results: {e}", "WARNING")
    
    def _calculate_ensemble_metrics(self, ensemble: Any, evaluation_results: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble training metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'accuracy': evaluation_results.get('accuracy', 0.0),
                'cv_mean': evaluation_results.get('cv_mean', 0.0),
                'cv_std': evaluation_results.get('cv_std', 0.0),
                'n_test_samples': evaluation_results.get('n_test_samples', 0),
                'model_type': 'stacking_ensemble',
                'calibrated': config.get('ensemble_params', {}).get('calibrate', True),
                'success': True
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    def _create_outcome_report(self, ensemble: Any, evaluation_results: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# Regime Ensemble Training Outcome Report

## Execution Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}

## Ensemble Training Results
- **Model Type**: {metrics.get('model_type', 'stacking_ensemble')}
- **Calibrated**: {'✅ Yes' if metrics.get('calibrated', False) else '❌ No'}
- **Accuracy**: {metrics.get('accuracy', 0):.3f}
- **CV Mean**: {metrics.get('cv_mean', 0):.3f} ± {metrics.get('cv_std', 0):.3f}
- **Test Samples**: {metrics.get('n_test_samples', 0):,}

## Base Models Used
"""
            
            base_models = config.get('base_models', {})
            for model_name in base_models.keys():
                report += f"- **{model_name}**: ✅ Available\n"
            
            report += f"""
## Performance Metrics
- **Accuracy**: {evaluation_results.get('accuracy', 0):.3f}
- **Cross-Validation Mean**: {evaluation_results.get('cv_mean', 0):.3f}
- **Cross-Validation Std**: {evaluation_results.get('cv_std', 0):.3f}

## Generated Artifacts
- Ensemble model (pickle file)
- Evaluation results
- Ensemble metadata

---
*Generated by Regime Ensemble Training Step at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# Regime Ensemble Training Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_regime_ensemble_training_step():
    """Register the regime ensemble training step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
    tprint("✅ Regime ensemble training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_ensemble_training_step()