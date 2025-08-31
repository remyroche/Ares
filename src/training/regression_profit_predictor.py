# src/training/regression_profit_predictor.py

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


class RegressionProfitPredictor:
    """
    Regression-based profit predictor for enhanced trading decisions.
    
    This system predicts actual percentage returns instead of discrete categories,
    enabling more sophisticated position sizing and risk management.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the regression profit predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RegressionProfitPredictor")
        
        # Model state
        self.model: Any = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False
        self.feature_names: list[str] = []
        
        # Configuration
        self.model_type: str = config.get("model_type", "LightGBM")
        self.min_profit_threshold: float = config.get("min_profit_threshold", 0.005)  # 0.5%
        self.max_profit_threshold: float = config.get("max_profit_threshold", 0.05)   # 5%
        self.position_sizing_enabled: bool = config.get("position_sizing_enabled", True)
        
        # Performance tracking
        self.training_history: list[Dict[str, float]] = []
        self.prediction_history: list[Dict[str, Any]] = []
        
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="model training"
    )
    async def train_model(
        self, 
        features: pd.DataFrame, 
        profit_targets: pd.Series,
        validation_split: float = 0.2
    ) -> bool:
        """Train the regression model to predict profit percentages.
        
        Args:
            features: Feature DataFrame
            profit_targets: Actual profit/loss percentages from historical data
            validation_split: Fraction of data to use for validation
            
        Returns:
            bool: True if training successful
        """
        try:
            self.logger.info(f"🚀 Training regression profit predictor with {len(features)} samples")
            
            # Validate inputs
            if features.empty or profit_targets.empty:
                self.logger.error("Empty features or profit targets provided")
                return False
                
            if len(features) != len(profit_targets):
                self.logger.error("Feature and target lengths don't match")
                return False
            
            # Store feature names
            self.feature_names = list(features.columns)
            
            # Prepare data
            X = features.values
            y = profit_targets.values
            
            # Time series split for validation
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize model based on type
            self.model = self._initialize_model()
            
            # Cross-validation training
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                self.logger.info(f"🔄 Training fold {fold + 1}/{n_splits}")
                
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                if self.model_type == "LightGBM":
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                cv_scores.append({
                    'fold': fold + 1,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                })
                
                self.logger.info(f"   Fold {fold + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
            
            # Final training on full dataset
            self.model.fit(X_scaled, y)
            
            # Calculate overall metrics
            y_pred_full = self.model.predict(X_scaled)
            overall_mse = mean_squared_error(y, y_pred_full)
            overall_mae = mean_absolute_error(y, y_pred_full)
            overall_r2 = r2_score(y, y_pred_full)
            
            # Store training results
            self.training_history.append({
                'timestamp': pd.Timestamp.now(),
                'n_samples': len(features),
                'n_features': len(self.feature_names),
                'overall_mse': overall_mse,
                'overall_mae': overall_mae,
                'overall_r2': overall_r2,
                'cv_scores': cv_scores
            })
            
            self.is_trained = True
            
            self.logger.info(f"✅ Training completed - R²: {overall_r2:.4f}, MAE: {overall_mae:.6f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            return False
    
    def _initialize_model(self) -> Any:
        """Initialize the regression model based on configuration."""
        if self.model_type == "LightGBM":
            return lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == "XGBoost":
            return xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
        elif self.model_type == "RandomForest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "GradientBoosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="profit prediction"
    )
    async def predict_profit(
        self, 
        features: pd.DataFrame,
        current_price: float,
        include_confidence: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Predict expected profit percentage for a potential trade.
        
        Args:
            features: Feature DataFrame for prediction
            current_price: Current market price
            include_confidence: Whether to include confidence metrics
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("Model not trained, cannot make predictions")
                return None
            
            if features.empty:
                self.logger.error("Empty features provided for prediction")
                return None
            
            # Prepare features
            X = features.values
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            predicted_profit_pct = self.model.predict(X_scaled)[0]
            
            # Calculate confidence metrics if requested
            confidence_metrics = {}
            if include_confidence and hasattr(self.model, 'predict_proba'):
                # For models that support probability estimation
                try:
                    # Use prediction variance as confidence proxy
                    if hasattr(self.model, 'estimators_'):
                        predictions = []
                        for estimator in self.model.estimators_:
                            predictions.append(estimator.predict(X_scaled)[0])
                        confidence_metrics['prediction_std'] = np.std(predictions)
                        confidence_metrics['prediction_confidence'] = 1.0 / (1.0 + confidence_metrics['prediction_std'])
                except Exception as e:
                    self.logger.warning(f"Could not calculate confidence metrics: {e}")
            
            # Build result
            result = {
                'predicted_profit_pct': predicted_profit_pct,
                'predicted_profit_abs': predicted_profit_pct * current_price,
                'current_price': current_price,
                'timestamp': pd.Timestamp.now(),
                'model_type': self.model_type,
                'confidence_metrics': confidence_metrics
            }
            
            # Add position sizing recommendations
            if self.position_sizing_enabled:
                result.update(self._calculate_position_sizing(predicted_profit_pct))
            
            # Store prediction history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {str(e)}")
            return None
    
    def _calculate_position_sizing(self, predicted_profit_pct: float) -> Dict[str, Any]:
        """Calculate position sizing recommendations based on predicted profit.
        
        Args:
            predicted_profit_pct: Predicted profit percentage
            
        Returns:
            Dictionary with position sizing recommendations
        """
        # Base position size (1.0 = full position)
        base_position_size = 1.0
        
        # Adjust based on predicted profit
        if predicted_profit_pct > self.max_profit_threshold:
            # High confidence - full position
            position_size = base_position_size
            confidence_level = "high"
        elif predicted_profit_pct > self.min_profit_threshold:
            # Medium confidence - scaled position
            profit_ratio = (predicted_profit_pct - self.min_profit_threshold) / (self.max_profit_threshold - self.min_profit_threshold)
            position_size = base_position_size * (0.5 + 0.5 * profit_ratio)
            confidence_level = "medium"
        else:
            # Low confidence - minimal or no position
            position_size = 0.0
            confidence_level = "low"
        
        return {
            'recommended_position_size': position_size,
            'confidence_level': confidence_level,
            'trade_recommendation': 'enter' if position_size > 0 else 'skip',
            'profit_threshold_met': predicted_profit_pct > self.min_profit_threshold
        }
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="model evaluation"
    )
    async def evaluate_model(self, test_features: pd.DataFrame, test_targets: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            test_features: Test feature DataFrame
            test_targets: Test profit targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_trained:
                self.logger.error("Model not trained, cannot evaluate")
                return {}
            
            X_test = test_features.values
            X_test_scaled = self.scaler.transform(X_test)
            y_test = test_targets.values
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Calculate profit-specific metrics
            profitable_predictions = y_pred > self.min_profit_threshold
            actual_profitable = y_test > self.min_profit_threshold
            
            if len(profitable_predictions) > 0:
                metrics['profit_accuracy'] = np.mean(profitable_predictions == actual_profitable)
                metrics['profit_precision'] = np.mean(y_test[profitable_predictions] > self.min_profit_threshold) if np.any(profitable_predictions) else 0
                metrics['profit_recall'] = np.mean(profitable_predictions[actual_profitable]) if np.any(actual_profitable) else 0
            
            self.logger.info(f"📊 Model Evaluation - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.6f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {str(e)}")
            return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                self.logger.warning("Model does not support feature importance")
                return {}
        except Exception as e:
            self.logger.error(f"Could not get feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if save successful
        """
        try:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'training_history': self.training_history
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            bool: True if load successful
        """
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data.get('training_history', [])
            self.is_trained = True
            
            self.logger.info(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {str(e)}")
            return False