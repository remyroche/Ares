"""
Universal ML Profit Integration System.

This module ensures that all ML models in the system deliver multi-output predictions
including direction, profit, confidence, and high-value trade factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb
import xgboost as xgb

from src.utils.logging import get_logger
from src.utils.decorators import handle_errors, with_tracing_span
from src.utils.training_pipeline_decorators import (
    validate_data_quality,
    validate_step_output,
    memory_efficient,
    quality_gate,
    prevent_data_leakage
)


@dataclass
class UniversalMLConfig:
    """Configuration for universal ML profit integration."""
    
    # Model types and parameters
    direction_model_types: List[str] = None  # ["RandomForest", "LightGBM", "XGBoost", "SVM", "NeuralNetwork"]
    profit_model_types: List[str] = None     # ["RandomForest", "LightGBM", "XGBoost", "Linear", "NeuralNetwork"]
    
    # Model parameters
    model_params: Dict[str, Dict[str, Any]] = None
    
    # Training configuration
    n_splits: int = 5  # Time series cross-validation splits
    test_size: float = 0.2
    random_state: int = 42
    
    # Profit prediction settings
    enable_profit_prediction: bool = True
    enable_ensemble_prediction: bool = True
    ensemble_method: str = "weighted_average"  # "weighted_average", "voting", "stacking"
    
    # Feature engineering
    enable_feature_selection: bool = True
    max_features: int = 100
    
    # High-value trade thresholds
    high_profit_threshold: float = 0.02  # 2% profit threshold
    high_loss_threshold: float = -0.01   # -1% loss threshold
    
    # Model persistence
    save_models: bool = True
    model_save_path: str = "models/universal_ml_profit"
    
    def __post_init__(self):
        if self.direction_model_types is None:
            self.direction_model_types = ["RandomForest", "LightGBM", "XGBoost"]
        if self.profit_model_types is None:
            self.profit_model_types = ["RandomForest", "LightGBM", "Linear"]
        if self.model_params is None:
            self.model_params = {
                "RandomForest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                "LightGBM": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "SVM": {"C": 1.0, "kernel": "rbf", "random_state": 42},
                "NeuralNetwork": {"hidden_layer_sizes": (100, 50), "random_state": 42},
                "Linear": {"random_state": 42}
            }


class UniversalMLProfitIntegrator:
    """
    Universal ML profit integration system that ensures all ML models deliver multi-output predictions.
    
    This system:
    1. Trains multiple model types for both direction and profit prediction
    2. Ensures all models deliver consistent multi-output format
    3. Provides ensemble predictions for improved accuracy
    4. Maintains compatibility with existing model interfaces
    """
    
    def __init__(self, config: Optional[UniversalMLConfig] = None):
        """Initialize the universal ML profit integrator."""
        self.config = config or UniversalMLConfig()
        self.logger = get_logger("UniversalMLProfitIntegrator")
        
        # Model storage
        self.direction_models = {}
        self.profit_models = {}
        self.feature_scalers = {}
        self.model_weights = {}
        
        # Training state
        self.is_trained = False
        self.feature_names = None
        
        # Performance metrics
        self.training_metrics = {}
        
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="universal_ml_profit_integration.train"
    )
    @with_tracing_span("UniversalMLProfitIntegrator.train", log_args=False)
    @validate_data_quality(
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        validation_score_requirements={"feature_quality": 0.7}
    )
    @prevent_data_leakage
    @memory_efficient
    @quality_gate
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train universal ML profit prediction system.
        
        Args:
            data: DataFrame with features, labels, and potential_profit_pct
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting universal ML profit prediction training with {len(data)} samples")
        
        # Prepare data
        X, y_direction, y_profit = self._prepare_data(data)
        
        if len(X) == 0:
            self.logger.error("No valid training data after preparation")
            return {}
        
        # Train direction models
        direction_results = self._train_direction_models(X, y_direction)
        
        # Train profit models
        profit_results = self._train_profit_models(X, y_profit)
        
        # Calculate ensemble weights
        self._calculate_ensemble_weights(direction_results, profit_results)
        
        # Save models if configured
        if self.config.save_models:
            self.save_models()
        
        self.is_trained = True
        self.training_metrics = {
            "direction_models": direction_results,
            "profit_models": profit_results,
            "ensemble_weights": self.model_weights
        }
        
        self.logger.info("Universal ML profit prediction training completed successfully")
        return self.training_metrics
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training."""
        # Filter out HOLD samples (label == 0) for binary classification
        if 'label' in data.columns:
            data = data[data['label'] != 0].copy()
        
        # Prepare features (exclude label and profit columns)
        feature_columns = [col for col in data.columns 
                          if col not in ['label', 'potential_profit_pct', 'timestamp']]
        
        X = data[feature_columns].copy()
        self.feature_names = feature_columns
        
        # Prepare labels
        y_direction = data['label'].copy()
        y_profit = data['potential_profit_pct'].copy()
        
        # Convert direction to binary (1 for BUY, 0 for SELL)
        y_direction = (y_direction == 1).astype(int)
        
        return X, y_direction, y_profit
    
    def _train_direction_models(self, X: pd.DataFrame, y_direction: pd.Series) -> Dict[str, Any]:
        """Train multiple direction prediction models."""
        self.logger.info("Training direction prediction models...")
        
        results = {}
        
        for model_type in self.config.direction_model_types:
            try:
                model = self._create_direction_model(model_type)
                scaler = StandardScaler()
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
                    
                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    scores.append(accuracy_score(y_val, pred))
                
                # Final training on full dataset
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y_direction)
                
                # Store model and scaler
                self.direction_models[model_type] = model
                self.feature_scalers[f"direction_{model_type}"] = scaler
                
                results[model_type] = {
                    "cv_accuracy": np.mean(scores),
                    "cv_std": np.std(scores),
                    "final_accuracy": accuracy_score(y_direction, model.predict(X_scaled))
                }
                
                self.logger.info(f"  {model_type}: CV accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_type} direction model: {e}")
                continue
        
        return results
    
    def _train_profit_models(self, X: pd.DataFrame, y_profit: pd.Series) -> Dict[str, Any]:
        """Train multiple profit prediction models."""
        self.logger.info("Training profit prediction models...")
        
        results = {}
        
        for model_type in self.config.profit_model_types:
            try:
                model = self._create_profit_model(model_type)
                scaler = StandardScaler()
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y_profit.iloc[train_idx], y_profit.iloc[val_idx]
                    
                    # Scale features
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    scores.append(r2_score(y_val, pred))
                
                # Final training on full dataset
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y_profit)
                
                # Store model and scaler
                self.profit_models[model_type] = model
                self.feature_scalers[f"profit_{model_type}"] = scaler
                
                results[model_type] = {
                    "cv_r2": np.mean(scores),
                    "cv_std": np.std(scores),
                    "final_r2": r2_score(y_profit, model.predict(X_scaled))
                }
                
                self.logger.info(f"  {model_type}: CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_type} profit model: {e}")
                continue
        
        return results
    
    def _create_direction_model(self, model_type: str) -> Any:
        """Create direction prediction model."""
        params = self.config.model_params.get(model_type, {})
        
        if model_type == "RandomForest":
            return RandomForestClassifier(**params)
        elif model_type == "LightGBM":
            return lgb.LGBMClassifier(**params)
        elif model_type == "XGBoost":
            return xgb.XGBClassifier(**params)
        elif model_type == "SVM":
            return SVC(probability=True, **params)
        elif model_type == "NeuralNetwork":
            return MLPClassifier(**params)
        elif model_type == "LogisticRegression":
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported direction model type: {model_type}")
    
    def _create_profit_model(self, model_type: str) -> Any:
        """Create profit prediction model."""
        params = self.config.model_params.get(model_type, {})
        
        if model_type == "RandomForest":
            return RandomForestRegressor(**params)
        elif model_type == "LightGBM":
            return lgb.LGBMRegressor(**params)
        elif model_type == "XGBoost":
            return xgb.XGBRegressor(**params)
        elif model_type == "Linear":
            return LinearRegression(**params)
        elif model_type == "NeuralNetwork":
            return MLPRegressor(**params)
        elif model_type == "Ridge":
            return Ridge(**params)
        elif model_type == "Lasso":
            return Lasso(**params)
        else:
            raise ValueError(f"Unsupported profit model type: {model_type}")
    
    def _calculate_ensemble_weights(self, direction_results: Dict[str, Any], profit_results: Dict[str, Any]):
        """Calculate ensemble weights based on model performance."""
        self.model_weights = {}
        
        # Direction model weights
        total_direction_score = sum(result.get("cv_accuracy", 0) for result in direction_results.values())
        for model_type, result in direction_results.items():
            if total_direction_score > 0:
                self.model_weights[f"direction_{model_type}"] = result.get("cv_accuracy", 0) / total_direction_score
            else:
                self.model_weights[f"direction_{model_type}"] = 1.0 / len(direction_results)
        
        # Profit model weights
        total_profit_score = sum(result.get("cv_r2", 0) for result in profit_results.values())
        for model_type, result in profit_results.items():
            if total_profit_score > 0:
                self.model_weights[f"profit_{model_type}"] = result.get("cv_r2", 0) / total_profit_score
            else:
                self.model_weights[f"profit_{model_type}"] = 1.0 / len(profit_results)
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="universal_ml_profit_integration.predict"
    )
    @with_tracing_span("UniversalMLProfitIntegrator.predict", log_args=False)
    @validate_step_output
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make ensemble predictions using all trained models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Ensure we have the right features
        if self.feature_names is not None:
            X = X[self.feature_names].copy()
        
        # Make direction predictions
        direction_predictions = {}
        direction_probabilities = {}
        
        for model_type, model in self.direction_models.items():
            scaler = self.feature_scalers[f"direction_{model_type}"]
            X_scaled = scaler.transform(X)
            
            direction_predictions[model_type] = model.predict(X_scaled)
            if hasattr(model, 'predict_proba'):
                direction_probabilities[model_type] = model.predict_proba(X_scaled)
        
        # Make profit predictions
        profit_predictions = {}
        
        for model_type, model in self.profit_models.items():
            scaler = self.feature_scalers[f"profit_{model_type}"]
            X_scaled = scaler.transform(X)
            
            profit_predictions[model_type] = model.predict(X_scaled)
        
        # Calculate ensemble predictions
        ensemble_direction = self._ensemble_direction_predictions(direction_predictions, direction_probabilities)
        ensemble_profit = self._ensemble_profit_predictions(profit_predictions)
        ensemble_confidence = self._calculate_ensemble_confidence(direction_probabilities)
        ensemble_high_value = self._calculate_high_value_factors(ensemble_direction, ensemble_profit)
        
        return {
            "direction": ensemble_direction,
            "profit": ensemble_profit,
            "confidence": ensemble_confidence,
            "high_value_trades": ensemble_high_value,
            "individual_predictions": {
                "direction": direction_predictions,
                "profit": profit_predictions
            }
        }
    
    def _ensemble_direction_predictions(self, direction_predictions: Dict[str, np.ndarray], 
                                      direction_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate ensemble direction predictions."""
        if self.config.ensemble_method == "weighted_average":
            # Weighted average of probabilities
            weighted_probs = np.zeros((len(next(iter(direction_predictions.values()))), 2))
            
            for model_type, probs in direction_probabilities.items():
                weight = self.model_weights.get(f"direction_{model_type}", 1.0)
                weighted_probs += weight * probs
            
            return np.argmax(weighted_probs, axis=1)
        else:
            # Simple majority voting
            all_predictions = np.array(list(direction_predictions.values()))
            return np.round(np.mean(all_predictions, axis=0)).astype(int)
    
    def _ensemble_profit_predictions(self, profit_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate ensemble profit predictions."""
        if self.config.ensemble_method == "weighted_average":
            # Weighted average of predictions
            weighted_preds = np.zeros(len(next(iter(profit_predictions.values()))))
            
            for model_type, preds in profit_predictions.items():
                weight = self.model_weights.get(f"profit_{model_type}", 1.0)
                weighted_preds += weight * preds
            
            return weighted_preds
        else:
            # Simple average
            all_predictions = np.array(list(profit_predictions.values()))
            return np.mean(all_predictions, axis=0)
    
    def _calculate_ensemble_confidence(self, direction_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate ensemble confidence scores."""
        if not direction_probabilities:
            return np.ones(len(next(iter(direction_probabilities.values())))) * 0.5
        
        # Weighted average of max probabilities
        weighted_confidence = np.zeros(len(next(iter(direction_probabilities.values()))))
        total_weight = 0
        
        for model_type, probs in direction_probabilities.items():
            weight = self.model_weights.get(f"direction_{model_type}", 1.0)
            max_probs = np.max(probs, axis=1)
            weighted_confidence += weight * max_probs
            total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return np.ones(len(weighted_confidence)) * 0.5
    
    def _calculate_high_value_factors(self, direction_pred: np.ndarray, profit_pred: np.ndarray) -> np.ndarray:
        """Calculate high-value trade factors."""
        high_value_factors = np.zeros(len(direction_pred))
        
        for i in range(len(direction_pred)):
            if direction_pred[i] == 1:  # BUY signal
                if profit_pred[i] > self.config.high_profit_threshold:
                    factor = min(1.0, profit_pred[i] / 0.05)
                    high_value_factors[i] = factor
                elif profit_pred[i] > 0:
                    factor = profit_pred[i] / self.config.high_profit_threshold
                    high_value_factors[i] = factor * 0.5
                else:
                    factor = max(-1.0, profit_pred[i] / self.config.high_loss_threshold)
                    high_value_factors[i] = factor * 0.5
            else:  # SELL signal
                if profit_pred[i] < self.config.high_loss_threshold:
                    factor = max(-1.0, profit_pred[i] / -0.03)
                    high_value_factors[i] = factor
                elif profit_pred[i] < 0:
                    factor = profit_pred[i] / self.config.high_loss_threshold
                    high_value_factors[i] = factor * 0.5
                else:
                    factor = min(1.0, profit_pred[i] / self.config.high_profit_threshold)
                    high_value_factors[i] = -factor * 0.5
        
        return high_value_factors
    
    def save_models(self):
        """Save trained models."""
        try:
            save_path = Path(self.config.model_save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save direction models
            for model_type, model in self.direction_models.items():
                joblib.dump(model, save_path / f"direction_{model_type}.joblib")
                joblib.dump(self.feature_scalers[f"direction_{model_type}"], 
                           save_path / f"direction_{model_type}_scaler.joblib")
            
            # Save profit models
            for model_type, model in self.profit_models.items():
                joblib.dump(model, save_path / f"profit_{model_type}.joblib")
                joblib.dump(self.feature_scalers[f"profit_{model_type}"], 
                           save_path / f"profit_{model_type}_scaler.joblib")
            
            # Save configuration and weights
            joblib.dump(self.config, save_path / "config.joblib")
            joblib.dump(self.model_weights, save_path / "model_weights.joblib")
            joblib.dump(self.feature_names, save_path / "feature_names.joblib")
            
            self.logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self, model_path: str):
        """Load trained models."""
        try:
            load_path = Path(model_path)
            
            # Load configuration and weights
            self.config = joblib.load(load_path / "config.joblib")
            self.model_weights = joblib.load(load_path / "model_weights.joblib")
            self.feature_names = joblib.load(load_path / "feature_names.joblib")
            
            # Load direction models
            for model_type in self.config.direction_model_types:
                model_file = load_path / f"direction_{model_type}.joblib"
                scaler_file = load_path / f"direction_{model_type}_scaler.joblib"
                
                if model_file.exists() and scaler_file.exists():
                    self.direction_models[model_type] = joblib.load(model_file)
                    self.feature_scalers[f"direction_{model_type}"] = joblib.load(scaler_file)
            
            # Load profit models
            for model_type in self.config.profit_model_types:
                model_file = load_path / f"profit_{model_type}.joblib"
                scaler_file = load_path / f"profit_{model_type}_scaler.joblib"
                
                if model_file.exists() and scaler_file.exists():
                    self.profit_models[model_type] = joblib.load(model_file)
                    self.feature_scalers[f"profit_{model_type}"] = joblib.load(scaler_file)
            
            self.is_trained = True
            self.logger.info(f"Models loaded from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=pd.DataFrame(),
    context="universal_ml_profit_integration.integrate"
)
@with_tracing_span("UniversalMLProfitIntegration.integrate", log_args=False)
@validate_data_quality(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7}
)
@memory_efficient
@quality_gate
def integrate_universal_ml_profit_prediction(data: pd.DataFrame, config: Optional[UniversalMLConfig] = None) -> pd.DataFrame:
    """
    Integrate universal ML profit prediction into the pipeline.
    
    This function ensures that all ML models deliver multi-output predictions
    including direction, profit, confidence, and high-value trade factors.
    
    Args:
        data: DataFrame with features and profit information
        config: Configuration for universal ML integration
        
    Returns:
        DataFrame with universal ML profit predictions added
    """
    if 'potential_profit_pct' not in data.columns:
        logging.warning("No 'potential_profit_pct' column found. Skipping universal ML profit integration.")
        return data
    
    integrator = UniversalMLProfitIntegrator(config)
    
    # Train the universal ML system
    training_results = integrator.train(data)
    
    if not training_results:
        logging.error("Universal ML profit prediction training failed")
        return data
    
    # Make predictions on the data
    predictions = integrator.predict(data)
    
    # Add predictions to the data
    data['universal_direction'] = predictions['direction']
    data['universal_profit'] = predictions['profit']
    data['universal_confidence'] = predictions['confidence']
    data['universal_high_value'] = predictions['high_value_trades']
    
    logging.info("Universal ML profit prediction integration completed successfully")
    return data