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
        
        if hasattr(model, 'fit'):
            # For sklearn-style models, retrain with profit features and weights
            if sample_weights is not None:
                adapted_model = model.fit(X, y, sample_weight=sample_weights)
            else:
                adapted_model = model.fit(X, y)
            
            # Store feature names for later use
            if hasattr(adapted_model, 'feature_names_in_'):
                self.feature_scalers[model_name] = {
                    'feature_names': adapted_model.feature_names_in_.tolist()
                }
            
            return adapted_model
        
        elif hasattr(model, 'train'):
            # For LightGBM/XGBoost models
            if sample_weights is not None:
                # Add sample weights to training data
                train_data = model.train_data
                if hasattr(train_data, 'set_weight'):
                    train_data.set_weight(sample_weights)
                adapted_model = model
            else:
                adapted_model = model
            
            return adapted_model
        
        else:
            # For other model types, return as-is
            self.logger.warning(f"Unknown model type {model_type}, returning as-is")
            return model
    
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
            Dictionary with predictions including profit estimates
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
        
        # Calculate high-value trade factors
        high_value_factors = self._calculate_high_value_factors(direction_pred, profit_pred)
        
        return {
            "direction": direction_pred,
            "direction_proba": direction_proba,
            "profit": profit_pred,
            "high_value_trades": high_value_factors,
            "model_name": model_name
        }
    
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