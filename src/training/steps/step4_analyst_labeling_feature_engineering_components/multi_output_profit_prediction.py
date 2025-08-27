"""
Multi-Output Profit Prediction System.

This module implements intelligent multi-output prediction for both trade direction
and profit magnitude, with automatic fallback to profit-weighted classification
when direct profit prediction is not feasible.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

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
class MultiOutputConfig:
    """Configuration for multi-output prediction system."""
    
    # Model types
    direction_model_type: str = "RandomForest"  # "RandomForest" or "LogisticRegression"
    profit_model_type: str = "RandomForest"     # "RandomForest" or "LinearRegression"
    
    # Model parameters
    direction_model_params: Dict[str, Any] = None
    profit_model_params: Dict[str, Any] = None
    
    # Training configuration
    n_splits: int = 5  # Time series cross-validation splits
    test_size: float = 0.2
    random_state: int = 42
    
    # Profit prediction feasibility thresholds
    min_samples_for_profit_prediction: int = 100
    min_profit_variance: float = 0.0001
    min_profit_range: float = 0.01
    
    # Fallback configuration
    enable_profit_weighted_fallback: bool = True
    profit_weight_multiplier: float = 20.0  # Multiplier for profit-based sample weights
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "RandomForest"  # "RandomForest" or "correlation"
    max_features: int = 100
    
    # High-value trade thresholds
    high_profit_threshold: float = 0.02  # 2% profit threshold
    high_loss_threshold: float = -0.01   # -1% loss threshold
    
    # Model persistence
    save_models: bool = True
    model_save_path: str = "models/multi_output_profit"


class MultiOutputProfitPredictor:
    """
    Multi-output prediction system for trade direction and profit magnitude.
    
    Implements intelligent method selection:
    1. Direct profit prediction when feasible
    2. Profit-weighted classification fallback when not feasible
    """
    
    def __init__(self, config: Optional[MultiOutputConfig] = None):
        """Initialize the multi-output predictor."""
        self.config = config or MultiOutputConfig()
        self.logger = get_logger("MultiOutputProfitPredictor")
        
        # Set default model parameters
        if self.config.direction_model_params is None:
            self.config.direction_model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": self.config.random_state
            }
        
        if self.config.profit_model_params is None:
            self.config.profit_model_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": self.config.random_state
            }
        
        # Model storage
        self.direction_model = None
        self.profit_model = None
        self.feature_scaler = None
        self.feature_selector = None
        
        # Training state
        self.is_trained = False
        self.can_predict_profit = False
        self.feature_names = None
        
        # Performance metrics
        self.training_metrics = {}
        
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training/prediction.
        
        Args:
            data: DataFrame with features and labels
            
        Returns:
            Tuple of (features, direction_labels, profit_labels)
        """
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
    
    def _select_features(self, X: pd.DataFrame, y_direction: pd.Series, y_profit: pd.Series) -> pd.DataFrame:
        """
        Select most important features for training.
        
        Args:
            X: Feature DataFrame
            y_direction: Direction labels
            y_profit: Profit labels
            
        Returns:
            DataFrame with selected features
        """
        if not self.config.enable_feature_selection:
            return X
        
        try:
            if self.config.feature_selection_method == "RandomForest":
                # Use RandomForest feature importance
                rf = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.config.random_state
                )
                rf.fit(X, y_direction)
                
                # Get feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Select top features
                top_features = feature_importance.head(self.config.max_features)['feature'].tolist()
                
            elif self.config.feature_selection_method == "correlation":
                # Use correlation with profit
                correlations = X.corrwith(y_profit).abs().sort_values(ascending=False)
                top_features = correlations.head(self.config.max_features).index.tolist()
            
            else:
                # Use sklearn's SelectKBest
                selector = SelectKBest(score_func=f_classif, k=self.config.max_features)
                X_selected = selector.fit_transform(X, y_direction)
                top_features = X.columns[selector.get_support()].tolist()
            
            self.logger.info(f"Selected {len(top_features)} features out of {len(X.columns)}")
            return X[top_features]
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}. Using all features.")
            return X
    
    def _can_predict_profit(self, y_profit: pd.Series) -> bool:
        """
        Check if direct profit prediction is feasible.
        
        Args:
            y_profit: Profit labels
            
        Returns:
            True if profit prediction is feasible
        """
        try:
            # Check sample count
            if len(y_profit) < self.config.min_samples_for_profit_prediction:
                self.logger.info(f"Insufficient samples for profit prediction: {len(y_profit)} < {self.config.min_samples_for_profit_prediction}")
                return False
            
            # Check variance
            profit_variance = y_profit.var()
            if profit_variance < self.config.min_profit_variance:
                self.logger.info(f"Insufficient profit variance: {profit_variance:.6f} < {self.config.min_profit_variance}")
                return False
            
            # Check range
            profit_range = y_profit.max() - y_profit.min()
            if profit_range < self.config.min_profit_range:
                self.logger.info(f"Insufficient profit range: {profit_range:.6f} < {self.config.min_profit_range}")
                return False
            
            # Check for non-zero profits
            non_zero_profits = (y_profit != 0).sum()
            if non_zero_profits < len(y_profit) * 0.1:  # At least 10% non-zero
                self.logger.info(f"Too few non-zero profits: {non_zero_profits}/{len(y_profit)}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking profit prediction feasibility: {e}")
            return False
    
    def _train_direct_profit_models(self, X: pd.DataFrame, y_direction: pd.Series, y_profit: pd.Series) -> Dict[str, Any]:
        """
        Train separate models for direction and profit prediction.
        
        Args:
            X: Feature DataFrame
            y_direction: Direction labels
            y_profit: Profit labels
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training direct profit prediction models...")
        
        # Initialize models
        if self.config.direction_model_type == "RandomForest":
            self.direction_model = RandomForestClassifier(**self.config.direction_model_params)
        elif self.config.direction_model_type == "LogisticRegression":
            self.direction_model = LogisticRegression(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported direction model type: {self.config.direction_model_type}")
        
        if self.config.profit_model_type == "RandomForest":
            self.profit_model = RandomForestRegressor(**self.config.profit_model_params)
        elif self.config.profit_model_type == "LinearRegression":
            self.profit_model = LinearRegression()
        else:
            raise ValueError(f"Unsupported profit model type: {self.config.profit_model_type}")
        
        # Initialize scaler
        self.feature_scaler = StandardScaler()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        direction_scores = []
        profit_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_dir_train, y_dir_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
            y_prof_train, y_prof_val = y_profit.iloc[train_idx], y_profit.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train direction model
            self.direction_model.fit(X_train_scaled, y_dir_train)
            dir_pred = self.direction_model.predict(X_val_scaled)
            direction_scores.append(accuracy_score(y_dir_val, dir_pred))
            
            # Train profit model
            self.profit_model.fit(X_train_scaled, y_prof_train)
            prof_pred = self.profit_model.predict(X_val_scaled)
            profit_scores.append(r2_score(y_prof_val, prof_pred))
        
        # Final training on full dataset
        X_scaled = self.feature_scaler.fit_transform(X)
        self.direction_model.fit(X_scaled, y_direction)
        self.profit_model.fit(X_scaled, y_profit)
        
        # Calculate final metrics
        dir_pred_final = self.direction_model.predict(X_scaled)
        prof_pred_final = self.profit_model.predict(X_scaled)
        
        results = {
            "direction_accuracy": accuracy_score(y_direction, dir_pred_final),
            "direction_cv_accuracy": np.mean(direction_scores),
            "profit_r2": r2_score(y_profit, prof_pred_final),
            "profit_cv_r2": np.mean(profit_scores),
            "profit_rmse": np.sqrt(mean_squared_error(y_profit, prof_pred_final)),
            "method": "direct_profit_prediction"
        }
        
        self.logger.info(f"Direct profit prediction training completed:")
        self.logger.info(f"  Direction accuracy: {results['direction_accuracy']:.4f}")
        self.logger.info(f"  Profit R²: {results['profit_r2']:.4f}")
        self.logger.info(f"  Profit RMSE: {results['profit_rmse']:.6f}")
        
        return results
    
    def _train_profit_weighted_fallback(self, X: pd.DataFrame, y_direction: pd.Series, y_profit: pd.Series) -> Dict[str, Any]:
        """
        Train profit-weighted classification model as fallback.
        
        Args:
            X: Feature DataFrame
            y_direction: Direction labels
            y_profit: Profit labels
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training profit-weighted classification fallback...")
        
        # Initialize direction model
        if self.config.direction_model_type == "RandomForest":
            self.direction_model = RandomForestClassifier(**self.config.direction_model_params)
        elif self.config.direction_model_type == "LogisticRegression":
            self.direction_model = LogisticRegression(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported direction model type: {self.config.direction_model_type}")
        
        # Initialize scaler
        self.feature_scaler = StandardScaler()
        
        # Calculate sample weights based on profit magnitude
        sample_weights = np.abs(y_profit) + 0.001  # Add small constant to avoid zero weights
        sample_weights = sample_weights * self.config.profit_weight_multiplier
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        direction_scores = []
        high_value_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_dir_train, y_dir_val = y_direction.iloc[train_idx], y_direction.iloc[val_idx]
            y_prof_train, y_prof_val = y_profit.iloc[train_idx], y_profit.iloc[val_idx]
            weights_train = sample_weights.iloc[train_idx]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train direction model with sample weights
            self.direction_model.fit(X_train_scaled, y_dir_train, sample_weight=weights_train)
            dir_pred = self.direction_model.predict(X_val_scaled)
            
            # Calculate metrics
            direction_scores.append(accuracy_score(y_dir_val, dir_pred))
            
            # Calculate high-value trade accuracy
            high_value_mask = (y_prof_val > self.config.high_profit_threshold) | (y_prof_val < self.config.high_loss_threshold)
            if high_value_mask.sum() > 0:
                high_value_accuracy = accuracy_score(y_dir_val[high_value_mask], dir_pred[high_value_mask])
                high_value_scores.append(high_value_accuracy)
        
        # Final training on full dataset
        X_scaled = self.feature_scaler.fit_transform(X)
        self.direction_model.fit(X_scaled, y_direction, sample_weight=sample_weights)
        
        # Calculate final metrics
        dir_pred_final = self.direction_model.predict(X_scaled)
        
        results = {
            "direction_accuracy": accuracy_score(y_direction, dir_pred_final),
            "direction_cv_accuracy": np.mean(direction_scores),
            "high_value_accuracy": np.mean(high_value_scores) if high_value_scores else 0.0,
            "method": "profit_weighted_fallback"
        }
        
        self.logger.info(f"Profit-weighted fallback training completed:")
        self.logger.info(f"  Direction accuracy: {results['direction_accuracy']:.4f}")
        self.logger.info(f"  High-value trade accuracy: {results['high_value_accuracy']:.4f}")
        
        return results
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="multi_output_profit_prediction.train"
    )
    @with_tracing_span("MultiOutputProfitPredictor.train", log_args=False)
    @validate_data_quality(
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        validation_score_requirements={"feature_quality": 0.7}
    )
    @prevent_data_leakage
    @memory_efficient
    @quality_gate
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the multi-output prediction system.
        
        Args:
            data: DataFrame with features, labels, and potential_profit_pct
            
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting multi-output profit prediction training with {len(data)} samples")
        
        # Prepare data
        X, y_direction, y_profit = self._prepare_data(data)
        
        if len(X) == 0:
            self.logger.error("No valid training data after preparation")
            return {}
        
        # Select features
        X_selected = self._select_features(X, y_direction, y_profit)
        
        # Check if profit prediction is feasible
        self.can_predict_profit = self._can_predict_profit(y_profit)
        
        if self.can_predict_profit:
            # Train direct profit prediction models
            results = self._train_direct_profit_models(X_selected, y_direction, y_profit)
        elif self.config.enable_profit_weighted_fallback:
            # Train profit-weighted fallback
            results = self._train_profit_weighted_fallback(X_selected, y_direction, y_profit)
        else:
            self.logger.error("Profit prediction not feasible and fallback disabled")
            return {}
        
        # Save models if configured
        if self.config.save_models:
            self.save_models()
        
        self.is_trained = True
        self.training_metrics = results
        
        self.logger.info(f"Multi-output training completed using method: {results['method']}")
        return results
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="multi_output_profit_prediction.predict"
    )
    @with_tracing_span("MultiOutputProfitPredictor.predict", log_args=False)
    @validate_step_output
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using trained models.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Select features if feature selection was used
        if self.feature_names is not None:
            X = X[self.feature_names].copy()
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Make predictions
        direction_pred = self.direction_model.predict(X_scaled)
        direction_proba = self.direction_model.predict_proba(X_scaled) if hasattr(self.direction_model, 'predict_proba') else None
        
        if self.can_predict_profit and self.profit_model is not None:
            profit_pred = self.profit_model.predict(X_scaled)
        else:
            profit_pred = np.zeros(len(X))
        
        # Calculate confidence scores
        if direction_proba is not None:
            confidence = np.max(direction_proba, axis=1)
        else:
            confidence = np.ones(len(X)) * 0.5
        
        # Calculate high-value trade factors as continuous values between -1 and 1
        high_value_factors = np.zeros(len(X))
        
        for i in range(len(X)):
            if direction_pred[i] == 1:  # BUY signal
                if profit_pred[i] > self.config.high_profit_threshold:
                    # High profit buy: scale from threshold to max expected profit (e.g., 0.05)
                    factor = min(1.0, profit_pred[i] / 0.05)  # Normalize to [0, 1]
                    high_value_factors[i] = factor
                elif profit_pred[i] > 0:
                    # Low profit buy: scale from 0 to threshold
                    factor = profit_pred[i] / self.config.high_profit_threshold
                    high_value_factors[i] = factor * 0.5  # Scale to [0, 0.5]
                else:
                    # Negative profit buy: scale from negative to 0
                    factor = max(-1.0, profit_pred[i] / self.config.high_loss_threshold)
                    high_value_factors[i] = factor * 0.5  # Scale to [-0.5, 0]
            else:  # SELL signal
                if profit_pred[i] < self.config.high_loss_threshold:
                    # High profit sell: scale from threshold to max expected loss (e.g., -0.03)
                    factor = max(-1.0, profit_pred[i] / -0.03)  # Normalize to [-1, 0]
                    high_value_factors[i] = factor
                elif profit_pred[i] < 0:
                    # Low loss sell: scale from 0 to threshold
                    factor = profit_pred[i] / self.config.high_loss_threshold
                    high_value_factors[i] = factor * 0.5  # Scale to [-0.5, 0]
                else:
                    # Positive profit sell: scale from positive to 0
                    factor = min(1.0, profit_pred[i] / self.config.high_profit_threshold)
                    high_value_factors[i] = -factor * 0.5  # Scale to [0, -0.5]
        
        return {
            "direction": direction_pred,
            "direction_proba": direction_proba,
            "profit": profit_pred,
            "confidence": confidence,
            "high_value_trades": high_value_factors,  # Now continuous values between -1 and 1
            "method": "direct_profit" if self.can_predict_profit else "profit_weighted"
        }
    
    def save_models(self, save_path: Optional[str] = None) -> None:
        """Save trained models to disk."""
        if not self.is_trained:
            self.logger.warning("No trained models to save")
            return
        
        save_path = save_path or self.config.model_save_path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save models
            if self.direction_model is not None:
                joblib.dump(self.direction_model, f"{save_path}/direction_model.pkl")
            
            if self.profit_model is not None:
                joblib.dump(self.profit_model, f"{save_path}/profit_model.pkl")
            
            # Save scaler and feature names
            if self.feature_scaler is not None:
                joblib.dump(self.feature_scaler, f"{save_path}/feature_scaler.pkl")
            
            if self.feature_names is not None:
                joblib.dump(self.feature_names, f"{save_path}/feature_names.pkl")
            
            # Save configuration and state
            state = {
                "config": self.config,
                "is_trained": self.is_trained,
                "can_predict_profit": self.can_predict_profit,
                "training_metrics": self.training_metrics
            }
            joblib.dump(state, f"{save_path}/model_state.pkl")
            
            self.logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self, load_path: str) -> bool:
        """Load trained models from disk."""
        try:
            # Load models
            direction_model_path = f"{load_path}/direction_model.pkl"
            if Path(direction_model_path).exists():
                self.direction_model = joblib.load(direction_model_path)
            
            profit_model_path = f"{load_path}/profit_model.pkl"
            if Path(profit_model_path).exists():
                self.profit_model = joblib.load(profit_model_path)
            
            # Load scaler and feature names
            scaler_path = f"{load_path}/feature_scaler.pkl"
            if Path(scaler_path).exists():
                self.feature_scaler = joblib.load(scaler_path)
            
            feature_names_path = f"{load_path}/feature_names.pkl"
            if Path(feature_names_path).exists():
                self.feature_names = joblib.load(feature_names_path)
            
            # Load state
            state_path = f"{load_path}/model_state.pkl"
            if Path(state_path).exists():
                state = joblib.load(state_path)
                self.config = state.get("config", self.config)
                self.is_trained = state.get("is_trained", False)
                self.can_predict_profit = state.get("can_predict_profit", False)
                self.training_metrics = state.get("training_metrics", {})
            
            self.logger.info(f"Models loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False


@handle_errors(
    exceptions=(Exception,),
    default_return={},
    context="multi_output_profit_prediction.integrate"
)
@with_tracing_span("MultiOutputProfitPrediction.integrate", log_args=False)
@validate_data_quality(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7}
)
@prevent_data_leakage
@memory_efficient
@quality_gate
def integrate_multi_output_prediction(data: pd.DataFrame, config: Optional[MultiOutputConfig] = None) -> Dict[str, Any]:
    """
    Integrate multi-output prediction into the existing pipeline.
    
    Args:
        data: DataFrame with features, labels, and potential_profit_pct
        config: Configuration for multi-output prediction
        
    Returns:
        Dictionary with training results and predictions
    """
    if 'potential_profit_pct' not in data.columns:
        logging.warning("No 'potential_profit_pct' column found. Skipping multi-output prediction.")
        return {}
    
    predictor = MultiOutputProfitPredictor(config)
    results = predictor.train(data)
    
    if results:
        # Make predictions on training data
        predictions = predictor.predict(data)
        results["predictions"] = predictions
    
    return results