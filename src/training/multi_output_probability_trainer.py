"""
Multi-Output Probability Trainer

This module implements multi-output training for probability outputs, replacing
the post-training calculation approach with direct training on probability targets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from datetime import datetime
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.centralized_decorators import (
    handle_errors,
    comprehensive_validation,
    performance_monitor,
    PerformanceLevel,
    ValidationLevel
)
from src.utils.logger import system_logger

logger = system_logger


class ProbabilityTargetGenerator:
    """
    Generates probability targets for multi-output training.
    
    This class creates training targets for each of the 4 probability types:
    1. Triple barrier probability
    2. Direction probability
    3. Magnitude probability
    4. Barrier avoidance probability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Default parameters
        self.profit_target = self.config.get('profit_target', 0.02)
        self.stop_loss = self.config.get('stop_loss', 0.01)
        self.look_ahead_periods = self.config.get('look_ahead_periods', 20)
        self.magnitude_threshold_factor = self.config.get('magnitude_threshold_factor', 0.8)
        self.adverse_threshold = self.config.get('adverse_threshold', 0.01)
        self.avoidance_look_ahead = self.config.get('avoidance_look_ahead', 10)
    
    @handle_errors(default_return=np.array([]), context="generate_triple_barrier_targets")
    @comprehensive_validation(level=ValidationLevel.STRICT)
    def generate_triple_barrier_targets(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market_data: pd.DataFrame,
        profit_target: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate triple barrier probability targets.
        
        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information
            profit_target: Profit target percentage
            stop_loss: Stop loss percentage
            
        Returns:
            Array of triple barrier probability targets
        """
        profit_target = profit_target or self.profit_target
        stop_loss = stop_loss or self.stop_loss
        
        targets = []
        
        for i in range(len(X)):
            if i >= len(market_data) - self.look_ahead_periods:
                # Not enough future data, use neutral target
                target = 0.5
            else:
                # Calculate actual triple barrier outcome
                entry_price = market_data['close'].iloc[i]
                future_prices = market_data['close'].iloc[i+1:i+self.look_ahead_periods+1]
                
                # Check if profit target or stop loss hit first
                profit_hit = any(future_prices >= entry_price * (1 + profit_target))
                stop_hit = any(future_prices <= entry_price * (1 - stop_loss))
                
                if profit_hit and not stop_hit:
                    target = 1.0  # Success
                elif stop_hit and not profit_hit:
                    target = 0.0  # Failure
                else:
                    # Partial success or no clear outcome
                    target = 0.5
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return=np.array([]), context="generate_direction_targets")
    @comprehensive_validation(level=ValidationLevel.STRICT)
    def generate_direction_targets(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate direction probability targets.
        
        Args:
            X: Feature matrix
            y: Target values (price changes)
            
        Returns:
            Array of direction probability targets
        """
        targets = []
        
        for i in range(len(X)):
            # Calculate actual direction accuracy
            predicted_direction = np.sign(y[i])
            actual_direction = np.sign(y[i])  # Assuming y contains actual price changes
            
            if predicted_direction == actual_direction:
                target = 1.0  # Correct direction
            else:
                target = 0.0  # Wrong direction
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return=np.array([]), context="generate_magnitude_targets")
    @comprehensive_validation(level=ValidationLevel.STRICT)
    def generate_magnitude_targets(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market_data: pd.DataFrame,
        threshold_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate magnitude probability targets.
        
        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information
            threshold_factor: Threshold factor for magnitude comparison
            
        Returns:
            Array of magnitude probability targets
        """
        threshold_factor = threshold_factor or self.magnitude_threshold_factor
        targets = []
        
        for i in range(len(X)):
            if i >= len(market_data) - 1:
                # Not enough future data
                target = 0.5
            else:
                # Calculate actual magnitude outcome
                predicted_magnitude = abs(y[i])
                actual_magnitude = abs(market_data['close'].pct_change().iloc[i])
                
                if predicted_magnitude >= actual_magnitude * threshold_factor:
                    target = 1.0  # Magnitude prediction successful
                else:
                    target = 0.0  # Magnitude prediction failed
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return=np.array([]), context="generate_barrier_avoidance_targets")
    @comprehensive_validation(level=ValidationLevel.STRICT)
    def generate_barrier_avoidance_targets(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market_data: pd.DataFrame,
        adverse_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate barrier avoidance probability targets.
        
        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information
            adverse_threshold: Threshold for adverse movements
            
        Returns:
            Array of barrier avoidance probability targets
        """
        adverse_threshold = adverse_threshold or self.adverse_threshold
        targets = []
        
        for i in range(len(X)):
            if i >= len(market_data) - self.avoidance_look_ahead:
                # Not enough future data
                target = 0.5
            else:
                # Calculate actual avoidance outcome
                future_returns = market_data['close'].pct_change().iloc[i+1:i+self.avoidance_look_ahead+1]
                adverse_movements = abs(future_returns) > adverse_threshold
                
                if not any(adverse_movements):
                    target = 1.0  # Successfully avoided adverse movements
                else:
                    target = 0.0  # Hit adverse movement
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return={}, context="generate_all_targets")
    @comprehensive_validation(level=ValidationLevel.STRICT)
    def generate_all_targets(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Generate all 4 probability targets.
        
        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information
            
        Returns:
            Dictionary containing all 4 probability targets
        """
        self.logger.info("Generating all probability targets for multi-output training")
        
        targets = {
            "triple_barrier": self.generate_triple_barrier_targets(X, y, market_data),
            "direction": self.generate_direction_targets(X, y),
            "magnitude": self.generate_magnitude_targets(X, y, market_data),
            "barrier_avoidance": self.generate_barrier_avoidance_targets(X, y, market_data)
        }
        
        # Validate targets
        for target_name, target_values in targets.items():
            if len(target_values) != len(X):
                raise ValueError(f"Target length mismatch for {target_name}")
            if not np.all((target_values >= 0) & (target_values <= 1)):
                self.logger.warning(f"Target values outside [0,1] range for {target_name}")
        
        self.logger.info(f"Generated targets for {len(X)} samples")
        return targets


class MultiOutputModel:
    """
    Multi-output model architecture for training 4 probability outputs.
    
    This class manages individual models for each probability type and
    provides ensemble capabilities with optimized weights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Model configuration
        self.use_lightgbm = self.config.get('use_lightgbm', True)
        self.n_estimators = self.config.get('n_estimators', 1000)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.max_depth = self.config.get('max_depth', 8)
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize models
        self.models = {}
        self.calibrators = {}
        self.ensemble_weights = None
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual models for each probability type."""
        for output_type in ['triple_barrier', 'direction', 'magnitude', 'avoidance']:
            self.models[output_type] = self._create_model(output_type)
    
    def _create_model(self, output_type: str):
        """Create model for specific output type."""
        if self.use_lightgbm:
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
                objective='binary'
            )
        else:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
    
    @handle_errors(default_return={}, context="fit_multi_output_models")
    @performance_monitor(level=PerformanceLevel.DETAILED)
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train_multi: Dict[str, np.ndarray], 
        X_val: np.ndarray, 
        y_val_multi: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Train all 4 probability models.
        
        Args:
            X_train: Training features
            y_train_multi: Training targets for all 4 probability types
            X_val: Validation features
            y_val_multi: Validation targets for all 4 probability types
            
        Returns:
            Dictionary containing trained models and metadata
        """
        self.logger.info("Starting multi-output model training")
        trained_models = {}
        
        for output_type in ['triple_barrier', 'direction', 'magnitude', 'avoidance']:
            self.logger.info(f"Training {output_type} model...")
            
            # Get model and targets
            model = self.models[output_type]
            y_train_target = y_train_multi[output_type]
            y_val_target = y_val_multi[output_type]
            
            # Handle class imbalance for certain targets
            sample_weights = None
            if output_type in ['triple_barrier', 'avoidance']:
                # These targets are often imbalanced
                try:
                    class_weights = compute_class_weight(
                        'balanced', 
                        classes=np.unique(y_train_target), 
                        y=y_train_target
                    )
                    sample_weights = class_weights[y_train_target.astype(int)]
                except Exception as e:
                    self.logger.warning(f"Could not compute class weights for {output_type}: {e}")
            
            # Train model
            if hasattr(model, 'fit'):
                if sample_weights is not None:
                    model.fit(X_train, y_train_target, sample_weight=sample_weights)
                else:
                    model.fit(X_train, y_train_target)
            
            # Calibrate probabilities
            try:
                calibrator = CalibratedClassifierCV(model, cv=5, method='isotonic')
                calibrator.fit(X_val, y_val_target)
                self.calibrators[output_type] = calibrator
                trained_models[output_type] = calibrator
            except Exception as e:
                self.logger.warning(f"Calibration failed for {output_type}, using original model: {e}")
                trained_models[output_type] = model
        
        # Optimize ensemble weights
        self.ensemble_weights = self._optimize_ensemble_weights(
            trained_models, X_val, y_val_multi
        )
        
        self.logger.info("Multi-output model training completed")
        return trained_models
    
    @handle_errors(default_return=None, context="optimize_ensemble_weights")
    def _optimize_ensemble_weights(
        self, 
        models: Dict[str, Any], 
        X_val: np.ndarray, 
        y_val_multi: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights for better probability accuracy.
        
        Args:
            models: Trained models
            X_val: Validation features
            y_val_multi: Validation targets
            
        Returns:
            Dictionary of optimized weights
        """
        def objective(weights):
            """Objective function to minimize."""
            total_loss = 0
            
            for i, output_type in enumerate(['triple_barrier', 'direction', 'magnitude', 'avoidance']):
                model = models[output_type]
                y_true = y_val_multi[output_type]
                
                try:
                    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class
                    # Calculate Brier score (lower is better)
                    brier_score = np.mean((y_pred_proba - y_true) ** 2)
                    total_loss += brier_score * weights[i]
                except Exception as e:
                    self.logger.warning(f"Error calculating loss for {output_type}: {e}")
                    total_loss += 1.0 * weights[i]  # Penalty for failed prediction
            
            return total_loss
        
        # Initial weights (equal)
        initial_weights = [0.25, 0.25, 0.25, 0.25]
        
        try:
            # Optimize weights
            result = minimize(
                objective,
                initial_weights,
                method='L-BFGS-B',
                bounds=[(0.1, 0.4) for _ in range(4)]  # Constrain weights
            )
            
            optimized_weights = dict(zip(
                ['triple_barrier', 'direction', 'magnitude', 'avoidance'], 
                result.x
            ))
            
            self.logger.info(f"Optimized ensemble weights: {optimized_weights}")
            return optimized_weights
            
        except Exception as e:
            self.logger.warning(f"Ensemble weight optimization failed: {e}")
            return dict(zip(
                ['triple_barrier', 'direction', 'magnitude', 'avoidance'], 
                initial_weights
            ))
    
    @handle_errors(default_return={}, context="predict_probabilities")
    def predict_probabilities(
        self, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Generate all 4 probability outputs.
        
        Args:
            X_test: Test features
            market_data: Market data (for compatibility)
            
        Returns:
            Dictionary containing all 4 probability outputs
        """
        probabilities = {}
        
        for output_type in ['triple_barrier', 'direction', 'magnitude', 'avoidance']:
            model = self.calibrators.get(output_type, self.models[output_type])
            
            try:
                # Get probability predictions
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] > 1:
                        # Binary classification, get positive class probability
                        prob_value = proba[:, 1].mean()
                    else:
                        # Single class, use the probability
                        prob_value = proba[:, 0].mean()
                else:
                    # Fallback to prediction
                    pred = model.predict(X_test)
                    prob_value = pred.mean()
                
                # Ensure probability is in [0, 1] range
                prob_value = np.clip(prob_value, 0.0, 1.0)
                
                probabilities[f"{output_type}_probability"] = float(prob_value)
                
            except Exception as e:
                self.logger.error(f"Error predicting {output_type} probability: {e}")
                probabilities[f"{output_type}_probability"] = 0.5
        
        # Add metadata
        probabilities["generation_timestamp"] = datetime.now().isoformat()
        probabilities["model_type"] = "multi_output"
        
        return probabilities


class MultiOutputProbabilityTrainer:
    """
    Main class for multi-output probability training.
    
    This class coordinates the entire multi-output training process,
    from target generation to model training and prediction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.target_generator = ProbabilityTargetGenerator(config)
        self.multi_output_model = MultiOutputModel(config)
        
        # Training state
        self.is_trained = False
        self.trained_models = None
    
    @handle_errors(default_return={}, context="prepare_multi_output_targets")
    @comprehensive_validation(level=ValidationLevel.STRICT)
    def prepare_multi_output_targets(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        market_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Generate 4 probability targets for training.
        
        Args:
            X: Feature matrix
            y: Target values
            market_data: Market data with OHLCV information
            
        Returns:
            Dictionary containing all 4 probability targets
        """
        self.logger.info("Preparing multi-output targets for training")
        return self.target_generator.generate_all_targets(X, y, market_data)
    
    @handle_errors(default_return={}, context="train_multi_output_model")
    @performance_monitor(level=PerformanceLevel.DETAILED)
    def train_multi_output_model(
        self, 
        X_train: np.ndarray, 
        y_train_multi: Dict[str, np.ndarray], 
        X_val: np.ndarray, 
        y_val_multi: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Train model on all 4 probability targets.
        
        Args:
            X_train: Training features
            y_train_multi: Training targets for all 4 probability types
            X_val: Validation features
            y_val_multi: Validation targets for all 4 probability types
            
        Returns:
            Dictionary containing trained models and metadata
        """
        self.logger.info("Starting multi-output model training")
        
        # Train the multi-output model
        self.trained_models = self.multi_output_model.fit(
            X_train, y_train_multi, X_val, y_val_multi
        )
        
        self.is_trained = True
        self.logger.info("Multi-output model training completed")
        
        return self.trained_models
    
    @handle_errors(default_return={}, context="predict_probabilities")
    def predict_probabilities(
        self, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Generate all 4 probability outputs.
        
        Args:
            X_test: Test features
            market_data: Market data (for compatibility)
            
        Returns:
            Dictionary containing all 4 probability outputs
        """
        if not self.is_trained:
            self.logger.error("Model not trained. Call train_multi_output_model first.")
            return self._get_default_probabilities()
        
        self.logger.info("Generating probability predictions")
        return self.multi_output_model.predict_probabilities(X_test, market_data)
    
    def _get_default_probabilities(self) -> Dict[str, float]:
        """Get default probabilities when training fails."""
        return {
            "triple_barrier_probability": 0.5,
            "direction_probability": 0.5,
            "magnitude_probability": 0.5,
            "barrier_avoidance_probability": 0.5,
            "generation_timestamp": datetime.now().isoformat(),
            "model_type": "multi_output"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "ensemble_weights": self.multi_output_model.ensemble_weights,
            "model_types": {name: type(model).__name__ for name, model in self.trained_models.items()},
            "calibrators": list(self.multi_output_model.calibrators.keys())
        }