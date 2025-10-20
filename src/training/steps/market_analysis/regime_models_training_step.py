"""
Regime Models Training Step

BaseStep-based implementation for training regime detection models.
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
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import sklearn
    ML_LIBRARY_VERSIONS['sklearn'] = sklearn.__version__
    tprint("✅ scikit-learn imported successfully", "SUCCESS")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"scikit-learn: {e}")
    tprint(f"❌ Failed to import scikit-learn: {e}", "ERROR")

try:
    import catboost as cb
    ML_LIBRARY_VERSIONS['catboost'] = cb.__version__
    tprint("✅ CatBoost imported successfully", "SUCCESS")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"CatBoost: {e}")
    tprint(f"❌ Failed to import CatBoost: {e}", "ERROR")

try:
    import lightgbm as lgb
    ML_LIBRARY_VERSIONS['lightgbm'] = lgb.__version__
    tprint("✅ LightGBM imported successfully", "SUCCESS")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"LightGBM: {e}")
    tprint(f"❌ Failed to import LightGBM: {e}", "ERROR")

try:
    from imodels import GreedyRuleListClassifier
    ML_LIBRARY_VERSIONS['imodels'] = "1.0.0"
    tprint("✅ imodels (Greedy Rule Lists) imported successfully", "SUCCESS")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"imodels: {e}")
    tprint(f"❌ Failed to import imodels: {e}", "ERROR")

if not ML_IMPORT_ERRORS:
    ML_LIBRARIES_AVAILABLE = True
    tprint("🎉 All ML libraries imported successfully", "SUCCESS")
else:
    tprint(f"⚠️ Import errors: {ML_IMPORT_ERRORS}", "WARNING")


class RegimeModelsTrainingStep(BaseStep):
    """
    Regime Models Training Step using BaseStep pattern.
    
    Trains regime detection models:
    - CatBoost (base model)
    - Greedy Rule Lists (base model)
    - ExtraTrees (base model)
    - LightGBM (meta-learner)
    """
    
    def __init__(self, step_name: str = "regime_models_training"):
        """Initialize the regime models training step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('RegimeModelsTrainingStep')
        
        if not ML_LIBRARIES_AVAILABLE:
            tprint("⚠️ Some ML libraries not available - functionality may be limited", "WARNING")
        
        tprint("✅ RegimeModelsTrainingStep initialized", "SUCCESS")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime models training step.
        
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
                - model_params: Model parameters (optional)
                
        Returns:
            Dictionary with training results and model artifacts
        """
        start_time = datetime.now()
        
        try:
            tprint(f"🔍 Starting regime models training for {config.get('symbol', 'UNKNOWN')}", "INFO")
            
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
            
            # Prepare features and targets
            X, y = self._prepare_training_data(train_data, regime_labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train models
            trained_models = self._train_models(X_train, y_train, config)
            
            # Evaluate models
            evaluation_results = self._evaluate_models(trained_models, X_test, y_test)
            
            # Save models and results
            self._save_models_and_results(trained_models, evaluation_results, config)
            
            # Calculate metrics
            metrics = self._calculate_training_metrics(trained_models, evaluation_results, start_time, config)
            
            # Create outcome report
            outcome_report = self._create_outcome_report(trained_models, evaluation_results, metrics, config)
            
            tprint(f"✅ Regime models training completed: {len(trained_models)} models trained", "SUCCESS")
            
            return {
                'success': True,
                'trained_models': trained_models,
                'evaluation_results': evaluation_results,
                'metrics': metrics,
                'outcome_report': outcome_report,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            error_msg = f"Regime models training failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'trained_models': {},
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
    
    def _train_models(self, X_train: pd.DataFrame, y_train: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train all regime detection models."""
        trained_models = {}
        
        try:
            # Get model parameters from config
            model_params = config.get('model_params', {})
            
            # Train CatBoost
            if 'catboost' in ML_LIBRARY_VERSIONS:
                tprint("🌳 Training CatBoost model...", "INFO")
                catboost_params = model_params.get('catboost', {
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'random_seed': 42,
                    'verbose': False
                })
                
                catboost_model = cb.CatBoostClassifier(**catboost_params)
                catboost_model.fit(X_train, y_train)
                trained_models['catboost'] = catboost_model
                tprint("✅ CatBoost trained successfully", "SUCCESS")
            
            # Train ExtraTrees
            if 'sklearn' in ML_LIBRARY_VERSIONS:
                tprint("🌲 Training ExtraTrees model...", "INFO")
                extratrees_params = model_params.get('extratrees', {
                    'n_estimators': 100,
                    'random_state': 42,
                    'n_jobs': -1
                })
                
                extratrees_model = ExtraTreesClassifier(**extratrees_params)
                extratrees_model.fit(X_train, y_train)
                trained_models['extratrees'] = extratrees_model
                tprint("✅ ExtraTrees trained successfully", "SUCCESS")
            
            # Train Greedy Rule Lists
            if 'imodels' in ML_LIBRARY_VERSIONS:
                tprint("📋 Training Greedy Rule Lists model...", "INFO")
                rulelist_params = model_params.get('rulelist', {
                    'max_depth': 5,
                    'random_state': 42
                })
                
                rulelist_model = GreedyRuleListClassifier(**rulelist_params)
                rulelist_model.fit(X_train, y_train)
                trained_models['rulelist'] = rulelist_model
                tprint("✅ Greedy Rule Lists trained successfully", "SUCCESS")
            
            # Train LightGBM
            if 'lightgbm' in ML_LIBRARY_VERSIONS:
                tprint("💡 Training LightGBM model...", "INFO")
                lightgbm_params = model_params.get('lightgbm', {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42,
                    'verbose': -1
                })
                
                lightgbm_model = lgb.LGBMClassifier(**lightgbm_params)
                lightgbm_model.fit(X_train, y_train)
                trained_models['lightgbm'] = lightgbm_model
                tprint("✅ LightGBM trained successfully", "SUCCESS")
            
            return trained_models
            
        except Exception as e:
            tprint(f"❌ Failed to train models: {e}", "ERROR")
            raise
    
    def _evaluate_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained models."""
        evaluation_results = {}
        
        try:
            for model_name, model in models.items():
                tprint(f"📊 Evaluating {model_name}...", "INFO")
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                    'classification_report': class_report,
                    'n_test_samples': len(y_test)
                }
                
                tprint(f"✅ {model_name} evaluation completed (accuracy: {accuracy:.3f})", "SUCCESS")
            
            return evaluation_results
            
        except Exception as e:
            tprint(f"❌ Failed to evaluate models: {e}", "ERROR")
            raise
    
    def _save_models_and_results(self, models: Dict[str, Any], evaluation_results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save trained models and evaluation results."""
        try:
            # Save models using artifact manager
            for model_name, model in models.items():
                self._save_model(f'{model_name}_model', model)
            
            # Save evaluation results
            self._save_artifact('evaluation_results', evaluation_results)
            
            # Save model metadata
            model_metadata = {
                'model_names': list(models.keys()),
                'n_models': len(models),
                'training_timestamp': datetime.now().isoformat(),
                'config': config
            }
            self._save_metadata(model_metadata)
            
            tprint("✅ Models and results saved to artifacts", "SUCCESS")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save models and results: {e}", "WARNING")
    
    def _calculate_training_metrics(self, models: Dict[str, Any], evaluation_results: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate training metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate average accuracy
            accuracies = [result['accuracy'] for result in evaluation_results.values()]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            # Calculate best model
            best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])[0] if evaluation_results else None
            
            metrics = {
                'processing_time_seconds': processing_time,
                'n_models_trained': len(models),
                'n_models_evaluated': len(evaluation_results),
                'average_accuracy': avg_accuracy,
                'best_model': best_model,
                'best_accuracy': evaluation_results[best_model]['accuracy'] if best_model else 0.0,
                'model_accuracies': {name: result['accuracy'] for name, result in evaluation_results.items()},
                'success': True
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}
    
    def _create_outcome_report(self, models: Dict[str, Any], evaluation_results: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# Regime Models Training Outcome Report

## Execution Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}

## Training Results
- **Models Trained**: {metrics.get('n_models_trained', 0)}
- **Models Evaluated**: {metrics.get('n_models_evaluated', 0)}
- **Average Accuracy**: {metrics.get('average_accuracy', 0):.3f}
- **Best Model**: {metrics.get('best_model', 'N/A')}
- **Best Accuracy**: {metrics.get('best_accuracy', 0):.3f}

## Model Performance
"""
            
            for model_name, result in evaluation_results.items():
                report += f"- **{model_name}**: {result['accuracy']:.3f} accuracy\n"
            
            report += f"""
## Model Details
- **CatBoost**: {'✅ Trained' if 'catboost' in models else '❌ Not available'}
- **ExtraTrees**: {'✅ Trained' if 'extratrees' in models else '❌ Not available'}
- **Greedy Rule Lists**: {'✅ Trained' if 'rulelist' in models else '❌ Not available'}
- **LightGBM**: {'✅ Trained' if 'lightgbm' in models else '❌ Not available'}

## Generated Artifacts
- Trained models (pickle files)
- Evaluation results
- Model metadata

---
*Generated by Regime Models Training Step at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# Regime Models Training Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_regime_models_training_step():
    """Register the regime models training step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_models_training", RegimeModelsTrainingStep)
    tprint("✅ Regime models training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_models_training_step()