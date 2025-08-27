"""
Profit Tracking ML Integration for Existing Models.

This module adapts the existing ML models from steps 6-14 to integrate profit tracking
features and multi-output prediction capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

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
class ProfitTrackingMLConfig:
    """Configuration for profit tracking ML integration."""
    
    # Integration settings
    enable_profit_features: bool = True
    enable_profit_weighting: bool = True
    enable_multi_output: bool = True
    
    # Profit-based feature settings
    profit_feature_threshold: float = 0.02  # Minimum profit to consider high-value
    profit_weight_multiplier: float = 20.0  # Multiplier for profit-based sample weights
    
    # Model adaptation settings
    adapt_existing_models: bool = True
    preserve_original_features: bool = True
    add_profit_predictions: bool = True
    
    # Validation settings
    time_series_splits: int = 5
    min_samples_for_profit: int = 100
    
    # Output settings
    save_adapted_models: bool = True
    model_save_path: str = "models/profit_tracking_adapted"


class ProfitTrackingMLIntegrator:
    """
    Integrates profit tracking into existing ML models from steps 6-14.
    
    This class adapts existing models to:
    1. Use profit-based features
    2. Apply profit-based sample weighting
    3. Add multi-output prediction capabilities
    4. Preserve original model functionality
    """
    
    def __init__(self, config: Optional[ProfitTrackingMLConfig] = None):
        """Initialize the profit tracking ML integrator."""
        self.config = config or ProfitTrackingMLConfig()
        self.logger = get_logger("ProfitTrackingMLIntegrator")
        
        # Import profit tracking components
        from .profit_based_feature_engineering import integrate_profit_features_into_pipeline
        from .multi_output_profit_prediction import integrate_multi_output_prediction
        
        self.integrate_profit_features = integrate_profit_features_into_pipeline
        self.integrate_multi_output = integrate_multi_output_prediction
        
        # Model storage
        self.adapted_models = {}
        self.profit_models = {}
        self.feature_scalers = {}
        
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="profit_tracking_ml_integration.adapt_existing_model"
    )
    @with_tracing_span("ProfitTrackingMLIntegrator.adapt_existing_model", log_args=False)
    @validate_data_quality(
        data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
        validation_score_requirements={"feature_quality": 0.7}
    )
    @prevent_data_leakage
    @memory_efficient
    @quality_gate
    def adapt_existing_model(
        self, 
        model, 
        data: pd.DataFrame, 
        target_column: str = "label",
        model_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Adapt an existing ML model to use profit tracking features.
        
        Args:
            model: Existing trained model (sklearn, lightgbm, etc.)
            data: DataFrame with features and potential_profit_pct
            target_column: Name of the target column
            model_name: Name identifier for the model
            
        Returns:
            Dictionary with adaptation results and adapted model
        """
        self.logger.info(f"Adapting existing model: {model_name}")
        
        if 'potential_profit_pct' not in data.columns:
            self.logger.warning("No potential_profit_pct column found. Skipping profit tracking adaptation.")
            return {"status": "SKIPPED", "reason": "No profit tracking data"}
        
        try:
            # 1. Add profit-based features
            if self.config.enable_profit_features:
                self.logger.info("Adding profit-based features...")
                enhanced_data = self.integrate_profit_features(data)
                self.logger.info(f"Added profit features. New shape: {enhanced_data.shape}")
            else:
                enhanced_data = data.copy()
            
            # 2. Prepare features and targets
            feature_columns = [col for col in enhanced_data.columns 
                             if col not in [target_column, 'potential_profit_pct', 'timestamp']]
            X = enhanced_data[feature_columns]
            y = enhanced_data[target_column]
            profit = enhanced_data['potential_profit_pct']
            
            # 3. Create profit-based sample weights
            if self.config.enable_profit_weighting:
                sample_weights = self._create_profit_based_weights(profit, y)
                self.logger.info("Created profit-based sample weights")
            else:
                sample_weights = None
            
            # 4. Adapt the model based on its type
            adapted_model = self._adapt_model_by_type(model, X, y, sample_weights, model_name)
            
            # 5. Create profit prediction model if enabled
            profit_model = None
            if self.config.enable_multi_output and self._can_predict_profit(profit):
                self.logger.info("Creating profit prediction model...")
                profit_model = self._create_profit_prediction_model(X, profit, model_name)
            
            # 6. Store adapted models
            self.adapted_models[model_name] = adapted_model
            if profit_model:
                self.profit_models[model_name] = profit_model
            
            # 7. Save models if configured
            if self.config.save_adapted_models:
                self._save_adapted_models(model_name, adapted_model, profit_model)
            
            results = {
                "status": "SUCCESS",
                "model_name": model_name,
                "original_features": len(data.columns),
                "enhanced_features": len(enhanced_data.columns),
                "profit_features_added": len(enhanced_data.columns) - len(data.columns),
                "has_profit_model": profit_model is not None,
                "sample_weighting": sample_weights is not None
            }
            
            self.logger.info(f"Successfully adapted model {model_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to adapt model {model_name}: {e}")
            return {"status": "FAILED", "error": str(e)}
    
    def _create_profit_based_weights(self, profit: pd.Series, target: pd.Series) -> np.ndarray:
        """Create profit-based sample weights."""
        # Base weights from profit magnitude
        profit_weights = np.abs(profit) + 0.001
        
        # Boost weights for high-value trades
        high_value_mask = (profit > self.config.profit_feature_threshold) | (profit < -self.config.profit_feature_threshold)
        profit_weights[high_value_mask] *= 2.0
        
        # Apply multiplier
        profit_weights *= self.config.profit_weight_multiplier
        
        # Normalize weights
        profit_weights = profit_weights / profit_weights.mean()
        
        return profit_weights
    
    def _can_predict_profit(self, profit: pd.Series) -> bool:
        """Check if profit prediction is feasible."""
        if len(profit) < self.config.min_samples_for_profit:
            return False
        
        profit_variance = profit.var()
        if profit_variance < 0.0001:
            return False
        
        profit_range = profit.max() - profit.min()
        if profit_range < 0.01:
            return False
        
        return True
    
    def _adapt_model_by_type(self, model, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray], model_name: str):
        """Adapt model based on its type."""
        model_type = type(model).__name__
        self.logger.info(f"Adapting {model_type} model")
        
        # Check if model is None (not yet created)
        if model is None:
            self.logger.info(f"Model {model_name} is None - will create new model with profit tracking")
            return self._create_new_model_with_profit_tracking(X, y, sample_weights, model_name)
        
        # Supported sklearn-style models
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            self.logger.info(f"Adapting sklearn-style model: {model_type}")
            return self._adapt_sklearn_model(model, X, y, sample_weights, model_name)
        
        # Supported LightGBM models
        elif hasattr(model, 'train') and hasattr(model, 'predict'):
            self.logger.info(f"Adapting LightGBM model: {model_type}")
            return self._adapt_lightgbm_model(model, X, y, sample_weights, model_name)
        
        # PyTorch models (CNN, TCN, Transformer)
        elif hasattr(model, 'forward') and hasattr(model, 'parameters'):
            self.logger.info(f"Adapting PyTorch model: {model_type}")
            return self._adapt_pytorch_model(model, X, y, sample_weights, model_name)
        
        # Custom trainer classes (CNNTrainer, TCNTrainer, etc.)
        elif hasattr(model, 'train') and hasattr(model, 'model'):
            self.logger.info(f"Adapting custom trainer: {model_type}")
            return self._adapt_custom_trainer(model, X, y, sample_weights, model_name)
        
        else:
            # For unsupported model types, log warning and return as-is
            self.logger.warning(f"Unsupported model type {model_type} for profit tracking adaptation")
            self.logger.warning(f"Model {model_name} will be used as-is without profit tracking features")
            return model
    
    def _create_new_model_with_profit_tracking(self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray], model_name: str):
        """Create a new model with profit tracking capabilities."""
        # Determine model type based on name
        if 'lightgbm' in model_name.lower():
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            return self._adapt_lightgbm_model(model, X, y, sample_weights, model_name)
        
        elif 'randomforest' in model_name.lower() or 'rf' in model_name.lower():
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            return self._adapt_sklearn_model(model, X, y, sample_weights, model_name)
        
        else:
            # Default to RandomForest
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            return self._adapt_sklearn_model(model, X, y, sample_weights, model_name)
    
    def _adapt_sklearn_model(self, model, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray], model_name: str):
        """Adapt sklearn-style models with profit tracking."""
        try:
            if sample_weights is not None:
                adapted_model = model.fit(X, y, sample_weight=sample_weights)
            else:
                adapted_model = model.fit(X, y)
            
            # Store feature names for later use
            if hasattr(adapted_model, 'feature_names_in_'):
                self.feature_scalers[model_name] = {
                    'feature_names': adapted_model.feature_names_in_.tolist()
                }
            
            self.logger.info(f"Successfully adapted sklearn model {model_name}")
            return adapted_model
            
        except Exception as e:
            self.logger.error(f"Failed to adapt sklearn model {model_name}: {e}")
            return model
    
    def _adapt_lightgbm_model(self, model, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray], model_name: str):
        """Adapt LightGBM models with profit tracking."""
        try:
            if sample_weights is not None:
                # LightGBM supports sample_weight parameter
                adapted_model = model.fit(X, y, sample_weight=sample_weights)
            else:
                adapted_model = model.fit(X, y)
            
            self.logger.info(f"Successfully adapted LightGBM model {model_name}")
            return adapted_model
            
        except Exception as e:
            self.logger.error(f"Failed to adapt LightGBM model {model_name}: {e}")
            return model
    
    def _adapt_pytorch_model(self, model, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray], model_name: str):
        """Adapt PyTorch models with profit tracking."""
        try:
            # For PyTorch models, we need to handle them differently
            # Since they require custom training loops, we'll create a wrapper
            self.logger.info(f"PyTorch model {model_name} detected - profit tracking will be limited")
            self.logger.info(f"PyTorch models require custom training loops for profit tracking")
            
            # Store the model as-is for now
            # In a full implementation, we would need to modify the training loop
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to adapt PyTorch model {model_name}: {e}")
            return model
    
    def _adapt_custom_trainer(self, trainer, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray], model_name: str):
        """Adapt custom trainer classes with profit tracking."""
        try:
            # For custom trainers (CNNTrainer, TCNTrainer, etc.), we need to modify the training process
            self.logger.info(f"Custom trainer {model_name} detected - profit tracking will be limited")
            self.logger.info(f"Custom trainers require modification of training loops for profit tracking")
            
            # Store the trainer as-is for now
            # In a full implementation, we would need to modify the training method
            return trainer
            
        except Exception as e:
            self.logger.error(f"Failed to adapt custom trainer {model_name}: {e}")
            return trainer
    
    def _create_profit_prediction_model(self, X: pd.DataFrame, profit: pd.Series, model_name: str) -> RandomForestRegressor:
        """Create a profit prediction model."""
        # Use RandomForest for profit prediction
        profit_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.time_series_splits)
        
        # Train the model
        profit_model.fit(X, profit)
        
        # Evaluate performance
        predictions = profit_model.predict(X)
        r2 = r2_score(profit, predictions)
        rmse = np.sqrt(mean_squared_error(profit, predictions))
        
        self.logger.info(f"Profit model performance - R²: {r2:.4f}, RMSE: {rmse:.6f}")
        
        return profit_model
    
    def _save_adapted_models(self, model_name: str, adapted_model, profit_model):
        """Save adapted models to disk."""
        save_path = Path(self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save adapted model
            model_path = save_path / f"{model_name}_adapted.pkl"
            joblib.dump(adapted_model, model_path)
            
            # Save profit model if exists
            if profit_model:
                profit_model_path = save_path / f"{model_name}_profit.pkl"
                joblib.dump(profit_model, profit_model_path)
            
            # Save feature information
            feature_info = {
                'model_name': model_name,
                'feature_scalers': self.feature_scalers.get(model_name, {}),
                'config': self.config.__dict__
            }
            feature_info_path = save_path / f"{model_name}_info.pkl"
            joblib.dump(feature_info, feature_info_path)
            
            self.logger.info(f"Saved adapted models to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def predict_with_profit_tracking(
        self, 
        model_name: str, 
        X: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Make predictions using adapted models with profit tracking.
        
        Args:
            model_name: Name of the adapted model
            X: Feature DataFrame
            
        Returns:
            Dictionary with predictions including profit estimates, confidence, and position sizing
        """
        if model_name not in self.adapted_models:
            raise ValueError(f"Model {model_name} not found in adapted models")
        
        adapted_model = self.adapted_models[model_name]
        profit_model = self.profit_models.get(model_name)
        
        # Make direction predictions
        if hasattr(adapted_model, 'predict_proba'):
            direction_proba = adapted_model.predict_proba(X)
            direction_pred = adapted_model.predict(X)
        else:
            direction_pred = adapted_model.predict(X)
            direction_proba = None
        
        # Make profit predictions if available
        profit_pred = None
        if profit_model:
            profit_pred = profit_model.predict(X)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(direction_pred, direction_proba, profit_pred)
        
        # Calculate high-value trade factors
        high_value_factors = self._calculate_high_value_factors(direction_pred, profit_pred)
        
        # Calculate position sizing recommendations
        position_sizing = self._calculate_position_sizing(direction_pred, profit_pred, confidence_scores, high_value_factors)
        
        return {
            "direction": direction_pred,
            "direction_proba": direction_proba,
            "profit": profit_pred,
            "high_value_trades": high_value_factors,
            "confidence": confidence_scores,
            "position_sizing": position_sizing,
            "model_name": model_name
        }
    
    def _calculate_confidence_scores(self, direction_pred, direction_proba, profit_pred) -> np.ndarray:
        """Calculate confidence scores based on model probabilities and profit predictions."""
        confidence_scores = np.zeros(len(direction_pred))
        
        for i in range(len(direction_pred)):
            # Base confidence from model probabilities
            if direction_proba is not None:
                prob = direction_proba[i]
                if len(prob) > 1:  # Multi-class case
                    max_prob = np.max(prob)
                    confidence_scores[i] = max_prob
                else:  # Binary case
                    confidence_scores[i] = prob[0] if direction_pred[i] == 1 else 1 - prob[0]
            else:
                # Default confidence if no probabilities available
                confidence_scores[i] = 0.7
            
            # Adjust confidence based on profit prediction
            if profit_pred is not None:
                profit_confidence = self._calculate_profit_based_confidence(profit_pred[i])
                # Combine model confidence with profit confidence
                confidence_scores[i] = 0.7 * confidence_scores[i] + 0.3 * profit_confidence
            
            # Ensure confidence is between 0 and 1
            confidence_scores[i] = np.clip(confidence_scores[i], 0.0, 1.0)
        
        return confidence_scores
    
    def _calculate_profit_based_confidence(self, profit_pred: float) -> float:
        """Calculate confidence based on predicted profit magnitude."""
        if profit_pred is None:
            return 0.5
        
        # Higher confidence for larger profit predictions (positive or negative)
        profit_abs = abs(profit_pred)
        
        # Sigmoid-like function to map profit to confidence
        # Higher profit magnitude = higher confidence
        confidence = 1.0 / (1.0 + np.exp(-10 * (profit_abs - 0.02)))
        
        return confidence
    
    def _calculate_position_sizing(self, direction_pred, profit_pred, confidence_scores, high_value_factors) -> Dict[str, np.ndarray]:
        """Calculate position sizing recommendations based on profit tracking using Tactician's leverage sizer."""
        n_samples = len(direction_pred)
        
        # Import Tactician's position sizer
        try:
            from src.tactician.position_sizer import PositionSizer
            position_sizer = PositionSizer()
            use_tactician_sizer = True
        except ImportError:
            self.logger.warning("Tactician position sizer not found, using fallback sizing")
            use_tactician_sizer = False
        
        # Base position size (will be calculated by position sizer)
        base_position_size = np.full(n_samples, 0.0)
        
        # Leverage recommendations (10-100x range)
        leverage = np.full(n_samples, 10.0)  # 10x base leverage
        
        # Risk-adjusted position size
        risk_adjusted_size = np.full(n_samples, 0.0)
        
        for i in range(n_samples):
            if profit_pred is not None and confidence_scores[i] > 0.6:
                # Adjust position size based on profit prediction
                profit_magnitude = abs(profit_pred[i])
                
                # Use Tactician's position sizer if available
                if use_tactician_sizer:
                    try:
                        # Create ML predictions dict for Tactician's sizer
                        ml_predictions = {
                            "price_target_confidences": {
                                "0.5%": confidence_scores[i] * 0.8,
                                "1.0%": confidence_scores[i] * 0.9,
                                "1.5%": confidence_scores[i] * 0.95,
                                "2.0%": confidence_scores[i]
                            },
                            "adversarial_confidences": {
                                "0.5%": (1.0 - confidence_scores[i]) * 0.8,
                                "1.0%": (1.0 - confidence_scores[i]) * 0.9,
                                "1.5%": (1.0 - confidence_scores[i]) * 0.95,
                                "2.0%": (1.0 - confidence_scores[i])
                            },
                            "directional_confidence": {
                                "confidence": confidence_scores[i],
                                "profit_potential": profit_pred[i]
                            }
                        }
                        
                        # Calculate position size using Tactician's sizer
                        position_info = await position_sizer.calculate_position_size(
                            ml_predictions=ml_predictions,
                            current_price=100.0,  # Placeholder, should be actual price
                            account_balance=10000.0,  # Placeholder, should be actual balance
                            analyst_confidence=confidence_scores[i],
                            tactician_confidence=confidence_scores[i]
                        )
                        
                        if position_info:
                            base_position_size[i] = position_info.get('final_position_size', 0.02)
                            # Tactician doesn't return leverage, so we calculate it separately
                            leverage[i] = self._calculate_tactician_leverage(confidence_scores[i], profit_pred[i])
                        else:
                            # Fallback to basic calculation
                            base_position_size[i] = self._calculate_fallback_position_size(profit_pred[i], confidence_scores[i])
                            leverage[i] = self._calculate_tactician_leverage(confidence_scores[i], profit_pred[i])
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to use Tactician position sizer: {e}")
                        # Fallback to basic calculation
                        base_position_size[i] = self._calculate_fallback_position_size(profit_pred[i], confidence_scores[i])
                        leverage[i] = self._calculate_tactician_leverage(confidence_scores[i], profit_pred[i])
                else:
                    # Fallback position sizing
                    base_position_size[i] = self._calculate_fallback_position_size(profit_pred[i], confidence_scores[i])
                    leverage[i] = self._calculate_tactician_leverage(confidence_scores[i], profit_pred[i])
                
                # Adjust leverage based on confidence and profit (10-100x range)
                if confidence_scores[i] > 0.7 and profit_magnitude > 0.02:
                    # Base leverage starts at 10x, scales up to 100x
                    leverage_multiplier = 1.0 + (confidence_scores[i] - 0.7) * 3.0  # 0.3 to 1.9
                    leverage[i] = min(100.0, 10.0 * leverage_multiplier)
                
                # Additional leverage boost for very high profit potential
                if profit_magnitude > 0.05 and confidence_scores[i] > 0.8:
                    leverage[i] = min(100.0, leverage[i] * 1.5)
                
                # Risk-adjusted sizing using Kelly criterion principles
                if profit_pred[i] > 0:
                    # For positive profit predictions
                    win_rate = confidence_scores[i]
                    avg_win = profit_pred[i]
                    avg_loss = 0.02  # Assume 2% average loss
                    
                    if avg_loss > 0:
                        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)  # Cap at 25%
                        risk_adjusted_size[i] = kelly_fraction
                else:
                    # For negative profit predictions (short positions)
                    risk_adjusted_size[i] = min(0.03, abs(profit_pred[i]) * 2.0)
            
            # Incremental high-value boost based on high-value factors
            high_value_boost = self._calculate_incremental_high_value_boost(high_value_factors[i])
            
            # Apply incremental boost to position size and leverage
            base_position_size[i] *= high_value_boost['position_multiplier']
            leverage[i] = min(100.0, leverage[i] * high_value_boost['leverage_multiplier'])
        
        return {
            "base_position_size": base_position_size,
            "leverage": leverage,
            "risk_adjusted_size": risk_adjusted_size,
            "recommended_size": np.minimum(base_position_size, risk_adjusted_size),
            "high_value_boost": high_value_boost if 'high_value_boost' in locals() else None
        }
    
    def _calculate_fallback_position_size(self, profit_pred: float, confidence: float) -> float:
        """Calculate fallback position size when Tactician sizer is not available."""
        # Base position size calculation (replaces 2-5% rule)
        base_size = 0.01  # 1% base
        
        # Scale with profit magnitude
        profit_magnitude = abs(profit_pred)
        if profit_magnitude > 0.01:
            size_multiplier = 1.0 + profit_magnitude * 20  # Scale up to 5x for high profit
            base_size *= min(5.0, size_multiplier)
        
        # Scale with confidence
        confidence_multiplier = 0.5 + confidence * 0.5  # 0.5x to 1.0x
        base_size *= confidence_multiplier
        
        return base_size
    
    def _calculate_incremental_high_value_boost(self, high_value_factor: float) -> Dict[str, float]:
        """Calculate incremental high-value boost based on continuous factor value."""
        # Convert high-value factor (-1 to 1) to incremental multipliers
        factor_abs = abs(high_value_factor)
        
        # Incremental position size multiplier (1.0 to 3.0)
        position_multiplier = 1.0 + factor_abs * 2.0
        
        # Incremental leverage multiplier (1.0 to 2.0)
        leverage_multiplier = 1.0 + factor_abs * 1.0
        
        return {
            "position_multiplier": position_multiplier,
            "leverage_multiplier": leverage_multiplier,
            "high_value_strength": factor_abs
        }
    
    def _calculate_tactician_leverage(self, confidence: float, profit_pred: float) -> float:
        """Calculate leverage using Tactician's approach (10-100x range)."""
        # Base leverage starts at 10x
        base_leverage = 10.0
        
        # Scale leverage with confidence (10x to 50x)
        confidence_leverage = base_leverage + (confidence - 0.5) * 80  # 10x to 50x
        
        # Additional leverage boost for high profit potential
        profit_magnitude = abs(profit_pred)
        profit_leverage_boost = 0.0
        
        if profit_magnitude > 0.02:  # 2% profit potential
            profit_leverage_boost = (profit_magnitude - 0.02) * 1000  # Up to 30x additional
        
        if profit_magnitude > 0.05:  # 5% profit potential
            profit_leverage_boost += (profit_magnitude - 0.05) * 2000  # Up to 20x more
        
        # Combine confidence and profit leverage
        total_leverage = confidence_leverage + profit_leverage_boost
        
        # Cap at 100x maximum
        return min(100.0, max(10.0, total_leverage))
    
    def _calculate_high_value_factors(self, direction_pred, profit_pred) -> np.ndarray:
        """Calculate high-value trade factors as continuous values between -1 and 1."""
        if profit_pred is None:
            return np.zeros(len(direction_pred))
        
        high_value_factors = np.zeros(len(direction_pred))
        
        for i in range(len(direction_pred)):
            if direction_pred[i] == 1:  # BUY signal
                if profit_pred[i] > self.config.profit_feature_threshold:
                    # High profit buy: scale from threshold to max expected profit
                    factor = min(1.0, profit_pred[i] / 0.05)
                    high_value_factors[i] = factor
                elif profit_pred[i] > 0:
                    # Low profit buy: scale from 0 to threshold
                    factor = profit_pred[i] / self.config.profit_feature_threshold
                    high_value_factors[i] = factor * 0.5
                else:
                    # Negative profit buy: scale from negative to 0
                    factor = max(-1.0, profit_pred[i] / -self.config.profit_feature_threshold)
                    high_value_factors[i] = factor * 0.5
            else:  # SELL signal
                if profit_pred[i] < -self.config.profit_feature_threshold:
                    # High profit sell: scale from threshold to max expected loss
                    factor = max(-1.0, profit_pred[i] / -0.03)
                    high_value_factors[i] = factor
                elif profit_pred[i] < 0:
                    # Low loss sell: scale from 0 to threshold
                    factor = profit_pred[i] / -self.config.profit_feature_threshold
                    high_value_factors[i] = factor * 0.5
                else:
                    # Positive profit sell: scale from positive to 0
                    factor = min(1.0, profit_pred[i] / self.config.profit_feature_threshold)
                    high_value_factors[i] = -factor * 0.5
        
        return high_value_factors


def integrate_profit_tracking_into_existing_models(
    models: Dict[str, Any],
    data: pd.DataFrame,
    config: Optional[ProfitTrackingMLConfig] = None
) -> Dict[str, Any]:
    """
    Integrate profit tracking into multiple existing models.
    
    Args:
        models: Dictionary of existing models {model_name: model}
        data: DataFrame with features and potential_profit_pct
        config: Configuration for profit tracking integration
        
    Returns:
        Dictionary with integration results for each model
    """
    integrator = ProfitTrackingMLIntegrator(config)
    results = {}
    
    for model_name, model in models.items():
        self.logger.info(f"Integrating profit tracking into model: {model_name}")
        
        try:
            result = integrator.adapt_existing_model(
                model=model,
                data=data,
                target_column="label",
                model_name=model_name
            )
            results[model_name] = result
            
        except Exception as e:
            self.logger.error(f"Failed to integrate model {model_name}: {e}")
            results[model_name] = {"status": "FAILED", "error": str(e)}
    
    return results


def adapt_step6_models_for_profit_tracking(
    step6_data: pd.DataFrame,
    config: Optional[ProfitTrackingMLConfig] = None
) -> Dict[str, Any]:
    """
    Adapt Step 6 models specifically for profit tracking.
    
    This function integrates with the existing Step 6 HMM-based training
    to add profit tracking capabilities.
    
    Args:
        step6_data: DataFrame from Step 6 with HMM features and labels
        config: Configuration for profit tracking integration
        
    Returns:
        Dictionary with adaptation results
    """
    if 'potential_profit_pct' not in step6_data.columns:
        logging.warning("No potential_profit_pct column found in Step 6 data")
        return {"status": "SKIPPED", "reason": "No profit tracking data"}
    
    integrator = ProfitTrackingMLIntegrator(config)
    
    # Extract HMM features and create profit-enhanced features
    hmm_features = [col for col in step6_data.columns 
                   if 'hmm' in col.lower() or 'regime' in col.lower() or 'intensity' in col.lower()]
    
    # Create enhanced dataset with profit features
    enhanced_data = integrator.integrate_profit_features(step6_data)
    
    # Adapt models for each timeframe if available
    timeframes = step6_data.get('timeframe', pd.Series(['1m'])).unique()
    
    results = {}
    for timeframe in timeframes:
        timeframe_data = enhanced_data[enhanced_data['timeframe'] == timeframe].copy()
        
        if len(timeframe_data) > 0:
            result = integrator.adapt_existing_model(
                model=None,  # Will be created based on timeframe
                data=timeframe_data,
                target_column="label",
                model_name=f"step6_{timeframe}"
            )
            results[f"step6_{timeframe}"] = result
    
    return results


def adapt_step7_models_for_profit_tracking(
    step7_data: pd.DataFrame,
    config: Optional[ProfitTrackingMLConfig] = None
) -> Dict[str, Any]:
    """
    Adapt Step 7 ensemble models for profit tracking.
    
    Args:
        step7_data: DataFrame from Step 7 with ensemble features
        config: Configuration for profit tracking integration
        
    Returns:
        Dictionary with adaptation results
    """
    integrator = ProfitTrackingMLIntegrator(config)
    
    # Adapt ensemble models
    result = integrator.adapt_existing_model(
        model=None,  # Will be created as ensemble
        data=step7_data,
        target_column="label",
        model_name="step7_ensemble"
    )
    
    return {"step7_ensemble": result}


def create_profit_tracking_pipeline(
    data: pd.DataFrame,
    config: Optional[ProfitTrackingMLConfig] = None
) -> Dict[str, Any]:
    """
    Create a complete profit tracking pipeline for existing models.
    
    Args:
        data: DataFrame with features, labels, and potential_profit_pct
        config: Configuration for profit tracking integration
        
    Returns:
        Dictionary with pipeline results
    """
    integrator = ProfitTrackingMLIntegrator(config)
    
    # 1. Add profit-based features
    enhanced_data = integrator.integrate_profit_features(data)
    
    # 2. Create multi-output prediction
    multi_output_results = integrator.integrate_multi_output(enhanced_data)
    
    # 3. Adapt existing models
    # This would be called with actual models from steps 6-14
    adaptation_results = {
        "step6_models": adapt_step6_models_for_profit_tracking(enhanced_data, config),
        "step7_models": adapt_step7_models_for_profit_tracking(enhanced_data, config),
        "multi_output": multi_output_results
    }
    
    return {
        "status": "SUCCESS",
        "enhanced_features": len(enhanced_data.columns),
        "original_features": len(data.columns),
        "adaptation_results": adaptation_results
    }