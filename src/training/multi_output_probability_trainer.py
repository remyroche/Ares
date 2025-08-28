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

# Import advanced neural network models
from .advanced_neural_models import (
    create_neural_model, 
    NEURAL_MODEL_CONFIGS,
    NeuralNetworkWrapper
)

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
    @comprehensive_validation()
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
                    target = 1  # Success
                elif stop_hit and not profit_hit:
                    target = 0  # Failure
                else:
                    # Partial success or no clear outcome - use deterministic approach
                    # If profit target is closer than stop loss, consider it success
                    max_profit = (future_prices.max() - entry_price) / entry_price
                    max_loss = (entry_price - future_prices.min()) / entry_price
                    target = 1 if max_profit > max_loss else 0
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return=np.array([]), context="generate_direction_targets")
    @comprehensive_validation()
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
                target = 1  # Correct direction
            else:
                target = 0  # Wrong direction
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return=np.array([]), context="generate_magnitude_targets")
    @comprehensive_validation()
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
                target = 0
            else:
                # Calculate actual magnitude outcome
                predicted_magnitude = abs(y[i])
                actual_magnitude = abs(market_data['close'].pct_change().iloc[i])
                
                if predicted_magnitude >= actual_magnitude * threshold_factor:
                    target = 1  # Magnitude prediction successful
                else:
                    target = 0  # Magnitude prediction failed
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return=np.array([]), context="generate_barrier_avoidance_targets")
    @comprehensive_validation()
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
                target = 0
            else:
                # Calculate actual avoidance outcome
                future_returns = market_data['close'].pct_change().iloc[i+1:i+self.avoidance_look_ahead+1]
                adverse_movements = abs(future_returns) > adverse_threshold
                
                if not any(adverse_movements):
                    target = 1  # Successfully avoided adverse movements
                else:
                    target = 0  # Hit adverse movement
            
            targets.append(target)
        
        return np.array(targets)
    
    @handle_errors(default_return={}, context="generate_all_targets")
    @comprehensive_validation()
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
            # Convert any non-binary values to binary
            target_values = np.array(target_values)
            target_values = np.where(target_values > 0.5, 1, 0)
            targets[target_name] = target_values
            self.logger.info(f"Target {target_name} validated and converted to binary")
        
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
        
        # Advanced model configuration
        self.model_architectures = self.config.get('model_architectures', {
            "1m": "cnn",      # CNN for 1-minute data
            "5m": "tcn",      # TCN for 5-minute data
            "15m": "transformer", # Transformer for 15-minute data
            "30m": "lightgbm",    # LightGBM for 30-minute data
            "1h": "lstm",     # LSTM for 1-hour data
            "4h": "gru",      # GRU for 4-hour data
            "1d": "randomforest"  # RandomForest for daily data
        })
        
        # Neural network configuration
        self.neural_config = self.config.get('neural_config', {})
        
        # Initialize models
        self.models = {}
        self.calibrators = {}
        self.ensemble_weights = None
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual models for each probability type."""
        for output_type in ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']:
            self.models[output_type] = self._create_model(output_type)
    
    def _create_model(self, output_type: str):
        """Create model for specific output type with advanced model selection."""
        
        # Get model type based on output type or use default
        model_type = self.config.get(f'{output_type}_model_type', 'lightgbm')
        
        # Determine input size (will be set during training)
        input_size = self.config.get('input_size', 50)  # Default, will be updated
        
        if model_type.lower() in ['lightgbm', 'lgb']:
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
                objective='binary'
            )
        elif model_type.lower() in ['randomforest', 'rf']:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        elif model_type.lower() in ['xgboost', 'xgb']:
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif model_type.lower() in ['catboost', 'cat']:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.max_depth,
                random_state=self.random_state,
                verbose=False
            )
        elif model_type.lower() in ['tcn', 'cnn', 'transformer', 'lstm', 'gru']:
            # Create neural network model
            neural_config = self.neural_config.get(model_type.lower(), {})
            return create_neural_model(
                model_type=model_type.lower(),
                input_size=input_size,
                num_classes=2,
                **neural_config
            )
        else:
            # Default to LightGBM
            self.logger.warning(f"Unknown model type '{model_type}', defaulting to LightGBM")
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbose=-1,
                objective='binary'
            )
    
    @performance_monitor()
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
        
        for output_type in ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']:
            self.logger.info(f"Training {output_type} model...")
            
            # Get model and targets
            model = self.models[output_type]
            y_train_target = y_train_multi[output_type]
            y_val_target = y_val_multi[output_type]
            
            # Update input size for neural networks if needed
            if hasattr(model, 'model_class') and hasattr(model, 'model_params'):
                # This is a neural network wrapper
                model.model_params['input_size'] = X_train.shape[1]
            
            # Handle class imbalance for certain targets
            sample_weights = None
            if output_type in ['triple_barrier', 'barrier_avoidance']:
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
            try:
                if hasattr(model, 'fit'):
                    # Check if it's a neural network (NeuralNetworkWrapper)
                    if isinstance(model, NeuralNetworkWrapper):
                        # Neural networks handle their own training
                        model.fit(X_train, y_train_target)
                        trained_models[output_type] = model
                    else:
                        # Traditional ML models
                        if sample_weights is not None:
                            model.fit(X_train, y_train_target, sample_weight=sample_weights)
                        else:
                            model.fit(X_train, y_train_target)
                        
                        # Calibrate probabilities for non-neural models
                        try:
                            calibrator = CalibratedClassifierCV(model, cv=5, method='isotonic')
                            calibrator.fit(X_val, y_val_target)
                            self.calibrators[output_type] = calibrator
                            trained_models[output_type] = calibrator
                        except Exception as e:
                            self.logger.warning(f"Calibration failed for {output_type}, using original model: {e}")
                            trained_models[output_type] = model
                else:
                    self.logger.error(f"Model {output_type} does not have fit method")
                    raise ValueError(f"Model {output_type} does not have fit method")
            except Exception as e:
                self.logger.error(f"Training failed for {output_type}: {e}")
                # Continue with other models instead of failing completely
                self.logger.warning(f"Skipping {output_type} model due to training failure")
                continue
        
        # Optimize ensemble weights
        self.ensemble_weights = self._optimize_ensemble_weights(
            trained_models, X_val, y_val_multi
        )
        
        self.logger.info("Multi-output model training completed")
        self.logger.info(f"Successfully trained {len(trained_models)} out of 4 models")
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
            
            for i, output_type in enumerate(['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']):
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
                ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance'], 
                result.x
            ))
            
            self.logger.info(f"Optimized ensemble weights: {optimized_weights}")
            return optimized_weights
            
        except Exception as e:
            self.logger.warning(f"Ensemble weight optimization failed: {e}")
            return dict(zip(
                ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance'], 
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
        
        for output_type in ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']:
            # Check if model exists
            if output_type not in self.models or self.models[output_type] is None:
                self.logger.warning(f"Model for {output_type} not available, using default probability")
                probabilities[f"{output_type}_probability"] = 0.5
                continue
                
            model = self.calibrators.get(output_type, self.models[output_type])
            
            try:
                # Get probability predictions
                if hasattr(model, 'predict_proba'):
                    # Handle both traditional ML models and neural networks
                    if isinstance(model, NeuralNetworkWrapper):
                        # Neural networks return probabilities directly
                        proba = model.predict_proba(X_test)
                    else:
                        # Traditional ML models
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
                probabilities[f"{output_type}_probability"] = 0.5  # Default fallback
        
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
        self.ensemble_weights = None
        self.calibrators = None
        
        # Advanced model configuration
        self.model_architectures = self.config.get('model_architectures', {
            "1m": "cnn",      # CNN for 1-minute data
            "5m": "tcn",      # TCN for 5-minute data
            "15m": "transformer", # Transformer for 15-minute data
            "30m": "lightgbm",    # LightGBM for 30-minute data
            "1h": "lstm",     # LSTM for 1-hour data
            "4h": "gru",      # GRU for 4-hour data
            "1d": "randomforest"  # RandomForest for daily data
        })
        
        # Neural network configuration
        self.neural_config = self.config.get('neural_config', {})
        
        # Configure models based on timeframe if provided
        self.timeframe = self.config.get('timeframe', '30m')
        self._configure_models_for_timeframe()
    
    def _configure_models_for_timeframe(self):
        """Configure models based on the specified timeframe."""
        if self.timeframe in self.model_architectures:
            model_type = self.model_architectures[self.timeframe]
            self.logger.info(f"Configuring models for {self.timeframe} timeframe using {model_type}")
            
            # Update config for each output type
            for output_type in ['triple_barrier', 'direction', 'magnitude', 'barrier_avoidance']:
                self.config[f'{output_type}_model_type'] = model_type
                
            # Update neural config if it's a neural network
            if model_type in ['tcn', 'cnn', 'transformer', 'lstm', 'gru']:
                self.neural_config[model_type] = NEURAL_MODEL_CONFIGS.get(model_type, {})
        else:
            self.logger.warning(f"No specific model configuration for timeframe {self.timeframe}, using defaults")
    
    @handle_errors(default_return={}, context="prepare_multi_output_targets")
    @comprehensive_validation()
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
    
    @performance_monitor()
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
        
        # Get ensemble weights and calibrators from the multi-output model
        self.ensemble_weights = self.multi_output_model.ensemble_weights
        self.calibrators = self.multi_output_model.calibrators
        
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
        if not self.is_trained or self.trained_models is None:
            self.logger.error("Model not trained. Call train_multi_output_model first.")
            return self._get_default_probabilities()
        
        self.logger.info("Generating probability predictions")
        
        # Use the multi-output model's prediction method
        try:
            return self.multi_output_model.predict_probabilities(X_test, market_data)
        except Exception as e:
            self.logger.error(f"Error in multi-output model prediction: {e}")
            return self._get_default_probabilities()
    
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