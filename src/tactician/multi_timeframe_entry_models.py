# src/tactician/multi_timeframe_entry_models.py

"""
Multi-timeframe Entry Models for Tactician.
ML models on 1m, 5m, 15m timeframes for optimal entry timing with high accuracy
required for 10-100x leverage trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import joblib
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.tactician.exit_strategy_feature_engineering import ExitStrategyFeatureEngineering


class MultiTimeframeEntryModels:
    """
    Multi-timeframe ML models for entry timing optimization.
    Provides accurate entry signals for high-leverage trading (10-100x).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize multi-timeframe entry models.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MultiTimeframeEntryModels")
        
        # Load configuration
        self.entry_config = config.get("entry_models", {})
        self.timeframes = self.entry_config.get("timeframes", ["1m", "5m"])
        self.models_dir = self.entry_config.get("models_dir", "models/entry_models")
        
        # Model configurations
        self.model_configs = {
            "1m": {
                "lookback_periods": [5, 10, 20, 50],
                "prediction_horizon": 1,  # 1 minute ahead
                "min_accuracy": 0.75,
                "ensemble_size": 5
            },
            "5m": {
                "lookback_periods": [10, 20, 50, 100],
                "prediction_horizon": 5,  # 5 minutes ahead
                "min_accuracy": 0.78,
                "ensemble_size": 7
            },

        }
        
        # Initialize models storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Initialize feature engineering
        self.feature_engineering = ExitStrategyFeatureEngineering(config)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="multi-timeframe entry models initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the multi-timeframe entry models system.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🔧 Multi-timeframe entry models system initialized")
            
            # Initialize feature engineering
            if not await self.feature_engineering.initialize():
                self.logger.error("❌ Feature engineering initialization failed")
                return False
            
            # Create models directory
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid multi-timeframe entry models configuration")
                return False
                
            self.logger.info("✅ Multi-timeframe entry models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Multi-timeframe entry models initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if not self.timeframes:
                self.logger.error("No timeframes specified")
                return False
                
            for tf in self.timeframes:
                if tf not in self.model_configs:
                    self.logger.error(f"Missing configuration for timeframe: {tf}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @guard_dataframe_nulls
    @with_tracing_span("train_multi_timeframe_models")
    async def train_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Train multi-timeframe entry models.

        Args:
            data_dict: Dictionary of dataframes for each timeframe

        Returns:
            Dict containing training results
        """
        try:
            self.logger.info("🚀 Training multi-timeframe entry models")
            
            training_results = {}
            
            for timeframe in self.timeframes:
                if timeframe not in data_dict:
                    self.logger.warning(f"⚠️ No data provided for timeframe: {timeframe}")
                    continue
                    
                self.logger.info(f"   Training {timeframe} model...")
                
                # Train model for this timeframe
                result = await self._train_timeframe_model(timeframe, data_dict[timeframe])
                training_results[timeframe] = result
                
                if result["success"]:
                    self.logger.info(f"   ✅ {timeframe} model trained successfully")
                    self.logger.info(f"      - Accuracy: {result['accuracy']:.3f}")
                    self.logger.info(f"      - Precision: {result['precision']:.3f}")
                    self.logger.info(f"      - Recall: {result['recall']:.3f}")
                else:
                    self.logger.error(f"   ❌ {timeframe} model training failed")
            
            self.logger.info("✅ Multi-timeframe model training completed")
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Multi-timeframe model training failed: {e}")
            return {}

    async def _train_timeframe_model(self, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model for specific timeframe.

        Args:
            timeframe: Timeframe string (1m, 5m, 15m)
            df: Input dataframe

        Returns:
            Dict containing training results
        """
        try:
            config = self.model_configs[timeframe]
            
            # Prepare features and labels
            features_df, labels = await self._prepare_training_data(timeframe, df, config)
            
            if features_df.empty or labels is None:
                return {"success": False, "error": "Failed to prepare training data"}
            
            # Create ensemble of models
            ensemble_models = await self._create_ensemble_models(timeframe, config)
            
            # Train ensemble
            trained_models = []
            scalers = []
            feature_importance_list = []
            
            for i, (model, scaler) in enumerate(ensemble_models):
                self.logger.info(f"      Training ensemble model {i+1}/{len(ensemble_models)}")
                
                # Scale features
                X_scaled = scaler.fit_transform(features_df)
                
                # Train model
                model.fit(X_scaled, labels)
                
                # Evaluate model
                cv_scores = cross_val_score(model, X_scaled, labels, cv=5, scoring='accuracy')
                accuracy = cv_scores.mean()
                
                # Store model and scaler
                trained_models.append(model)
                scalers.append(scaler)
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance_list.append(model.feature_importances_)
                elif hasattr(model, 'coef_'):
                    feature_importance_list.append(np.abs(model.coef_[0]))
                else:
                    feature_importance_list.append(np.ones(features_df.shape[1]))
            
            # Calculate ensemble performance
            ensemble_accuracy = np.mean([cv_scores.mean() for cv_scores in 
                                       [cross_val_score(model, scaler.transform(features_df), labels, cv=5) 
                                        for model, scaler in zip(trained_models, scalers)]])
            
            # Check if accuracy meets minimum requirement
            if ensemble_accuracy < config["min_accuracy"]:
                self.logger.warning(f"⚠️ {timeframe} model accuracy ({ensemble_accuracy:.3f}) below minimum ({config['min_accuracy']})")
            
            # Store models and metadata
            self.models[timeframe] = trained_models
            self.scalers[timeframe] = scalers
            self.feature_importance[timeframe] = np.mean(feature_importance_list, axis=0)
            
            # Calculate final metrics
            final_metrics = await self._calculate_final_metrics(timeframe, features_df, labels)
            
            # Save models
            await self._save_models(timeframe)
            
            return {
                "success": True,
                "accuracy": ensemble_accuracy,
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "f1_score": final_metrics["f1_score"],
                "roc_auc": final_metrics["roc_auc"],
                "ensemble_size": len(trained_models),
                "feature_count": features_df.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Timeframe {timeframe} training failed: {e}")
            return {"success": False, "error": str(e)}

    async def _prepare_training_data(self, timeframe: str, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Prepare features and labels for training.

        Args:
            timeframe: Timeframe string
            df: Input dataframe
            config: Model configuration

        Returns:
            Tuple of (features_df, labels)
        """
        try:
            # Apply exit strategy feature engineering
            features_df = await self.feature_engineering.apply_all(df)
            
            # Create entry timing labels
            labels = await self._create_entry_labels(df, config["prediction_horizon"])
            
            if labels is None:
                return pd.DataFrame(), None
            
            # Remove rows with NaN values
            valid_mask = ~(features_df.isna().any(axis=1) | pd.isna(labels))
            features_df = features_df[valid_mask]
            labels = labels[valid_mask]
            
            # Ensure we have enough data
            if len(features_df) < 1000:
                self.logger.warning(f"⚠️ Insufficient data for {timeframe}: {len(features_df)} samples")
                return pd.DataFrame(), None
            
            return features_df, labels
            
        except Exception as e:
            self.logger.error(f"Data preparation failed for {timeframe}: {e}")
            return pd.DataFrame(), None

    async def _create_entry_labels(self, df: pd.DataFrame, prediction_horizon: int) -> Optional[np.ndarray]:
        """
        Create entry timing labels based on future price movement.

        Args:
            df: Input dataframe
            prediction_horizon: Number of periods to look ahead

        Returns:
            Array of labels (1 for good entry, 0 for bad entry)
        """
        try:
            # Calculate future returns
            future_returns = df['close'].shift(-prediction_horizon) / df['close'] - 1
            
            # Define entry criteria for high-leverage trading
            # We want entries that lead to significant positive returns
            threshold = 0.005  # 0.5% minimum return for good entry
            
            # Create labels based on future returns
            labels = np.where(future_returns > threshold, 1, 0)
            
            # Remove NaN values
            labels = labels[:-prediction_horizon]  # Remove last few rows where we can't calculate future returns
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Label creation failed: {e}")
            return None

    async def _create_ensemble_models(self, timeframe: str, config: Dict[str, Any]) -> List[Tuple[Any, Any]]:
        """
        Create ensemble of models for the timeframe.

        Args:
            timeframe: Timeframe string
            config: Model configuration

        Returns:
            List of (model, scaler) tuples
        """
        try:
            ensemble_size = config["ensemble_size"]
            models = []
            
            # Create different model types for diversity
            model_types = [
                (RandomForestClassifier, {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
                (GradientBoostingClassifier, {"n_estimators": 100, "max_depth": 6, "random_state": 42}),
                (xgb.XGBClassifier, {"n_estimators": 100, "max_depth": 6, "random_state": 42}),
                (lgb.LGBMClassifier, {"n_estimators": 100, "max_depth": 6, "random_state": 42}),
                (LogisticRegression, {"random_state": 42, "max_iter": 1000}),
                (SVC, {"kernel": "rbf", "random_state": 42, "probability": True}),
                (MLPClassifier, {"hidden_layer_sizes": (100, 50), "random_state": 42, "max_iter": 1000})
            ]
            
            for i in range(ensemble_size):
                model_type, params = model_types[i % len(model_types)]
                
                # Add some randomization to parameters for diversity
                if "n_estimators" in params:
                    params["n_estimators"] = params["n_estimators"] + np.random.randint(-20, 20)
                if "max_depth" in params:
                    params["max_depth"] = max(3, params["max_depth"] + np.random.randint(-2, 2))
                
                model = model_type(**params)
                scaler = RobustScaler()  # Use robust scaler for financial data
                
                models.append((model, scaler))
            
            return models
            
        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            return []

    async def _calculate_final_metrics(self, timeframe: str, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate final performance metrics.

        Args:
            timeframe: Timeframe string
            features_df: Feature dataframe
            labels: True labels

        Returns:
            Dict of performance metrics
        """
        try:
            # Use ensemble for predictions
            predictions = await self.predict_ensemble(timeframe, features_df)
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            # Calculate ROC AUC (use probability predictions)
            proba_predictions = await self.predict_proba_ensemble(timeframe, features_df)
            roc_auc = roc_auc_score(labels, proba_predictions[:, 1])
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0
            }

    async def predict_ensemble(self, timeframe: str, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble of models.

        Args:
            timeframe: Timeframe string
            features_df: Feature dataframe

        Returns:
            Array of predictions
        """
        try:
            if timeframe not in self.models:
                raise ValueError(f"No models found for timeframe: {timeframe}")
            
            predictions = []
            
            for model, scaler in zip(self.models[timeframe], self.scalers[timeframe]):
                X_scaled = scaler.transform(features_df)
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Ensemble prediction (majority vote)
            ensemble_pred = np.mean(predictions, axis=0) >= 0.5
            return ensemble_pred.astype(int)
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return np.zeros(len(features_df))

    async def predict_proba_ensemble(self, timeframe: str, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using ensemble of models.

        Args:
            timeframe: Timeframe string
            features_df: Feature dataframe

        Returns:
            Array of probability predictions
        """
        try:
            if timeframe not in self.models:
                raise ValueError(f"No models found for timeframe: {timeframe}")
            
            probabilities = []
            
            for model, scaler in zip(self.models[timeframe], self.scalers[timeframe]):
                X_scaled = scaler.transform(features_df)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)
                else:
                    # For models without predict_proba, use decision_function
                    decision = model.decision_function(X_scaled)
                    proba = np.column_stack([1 - decision, decision])
                
                probabilities.append(proba)
            
            # Average probabilities across ensemble
            ensemble_proba = np.mean(probabilities, axis=0)
            return ensemble_proba
            
        except Exception as e:
            self.logger.error(f"Ensemble probability prediction failed: {e}")
            return np.column_stack([np.ones(len(features_df)), np.zeros(len(features_df))])

    async def get_entry_signal(self, timeframe: str, current_data: pd.DataFrame, 
                              position_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get entry signal for specific timeframe.

        Args:
            timeframe: Timeframe string
            current_data: Current market data
            position_context: Optional position context

        Returns:
            Dict containing entry signal and confidence
        """
        try:
            if timeframe not in self.models:
                return {
                    "signal": 0,
                    "confidence": 0.0,
                    "probability": 0.0,
                    "error": f"No models available for {timeframe}"
                }
            
            # Apply feature engineering
            features_df = await self.feature_engineering.apply_all(current_data, position_context)
            
            if features_df.empty:
                return {
                    "signal": 0,
                    "confidence": 0.0,
                    "probability": 0.0,
                    "error": "Feature engineering failed"
                }
            
            # Get latest features
            latest_features = features_df.iloc[-1:].fillna(0)
            
            # Make prediction
            prediction = await self.predict_ensemble(timeframe, latest_features)
            probability = await self.predict_proba_ensemble(timeframe, latest_features)
            
            # Calculate confidence based on ensemble agreement
            individual_predictions = []
            for model, scaler in zip(self.models[timeframe], self.scalers[timeframe]):
                X_scaled = scaler.transform(latest_features)
                pred = model.predict(X_scaled)
                individual_predictions.append(pred[0])
            
            # Confidence is the proportion of models agreeing with ensemble prediction
            ensemble_pred = prediction[0]
            agreement_ratio = np.mean([p == ensemble_pred for p in individual_predictions])
            
            return {
                "signal": ensemble_pred,
                "confidence": agreement_ratio,
                "probability": probability[0, 1],  # Probability of good entry
                "timeframe": timeframe,
                "timestamp": current_data.index[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Entry signal generation failed for {timeframe}: {e}")
            return {
                "signal": 0,
                "confidence": 0.0,
                "probability": 0.0,
                "error": str(e)
            }

    async def get_multi_timeframe_entry_signal(self, data_dict: Dict[str, pd.DataFrame], 
                                              position_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get combined entry signal from all timeframes.

        Args:
            data_dict: Dictionary of dataframes for each timeframe
            position_context: Optional position context

        Returns:
            Dict containing combined entry signal
        """
        try:
            self.logger.info("🔍 Generating multi-timeframe entry signal")
            
            timeframe_signals = {}
            total_confidence = 0.0
            total_probability = 0.0
            signal_count = 0
            
            for timeframe in self.timeframes:
                if timeframe not in data_dict:
                    continue
                
                signal = await self.get_entry_signal(timeframe, data_dict[timeframe], position_context)
                timeframe_signals[timeframe] = signal
                
                if signal["signal"] == 1:  # Good entry signal
                    total_confidence += signal["confidence"]
                    total_probability += signal["probability"]
                    signal_count += 1
            
            # Calculate combined signal
            if signal_count > 0:
                avg_confidence = total_confidence / signal_count
                avg_probability = total_probability / signal_count
                
                # Combined signal requires majority of timeframes to agree
                majority_threshold = len(self.timeframes) / 2
                combined_signal = 1 if signal_count >= majority_threshold else 0
            else:
                combined_signal = 0
                avg_confidence = 0.0
                avg_probability = 0.0
            
            result = {
                "combined_signal": combined_signal,
                "combined_confidence": avg_confidence,
                "combined_probability": avg_probability,
                "timeframe_signals": timeframe_signals,
                "signal_count": signal_count,
                "total_timeframes": len(self.timeframes)
            }
            
            self.logger.info(f"✅ Multi-timeframe entry signal generated")
            self.logger.info(f"   - Combined signal: {combined_signal}")
            self.logger.info(f"   - Confidence: {avg_confidence:.3f}")
            self.logger.info(f"   - Probability: {avg_probability:.3f}")
            self.logger.info(f"   - Timeframes with signals: {signal_count}/{len(self.timeframes)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe entry signal generation failed: {e}")
            return {
                "combined_signal": 0,
                "combined_confidence": 0.0,
                "combined_probability": 0.0,
                "error": str(e)
            }

    async def _save_models(self, timeframe: str) -> bool:
        """
        Save trained models to disk.

        Args:
            timeframe: Timeframe string

        Returns:
            bool: True if save successful
        """
        try:
            if timeframe not in self.models:
                return False
            
            # Create timeframe directory
            timeframe_dir = Path(self.models_dir) / timeframe
            timeframe_dir.mkdir(parents=True, exist_ok=True)
            
            # Save models
            for i, (model, scaler) in enumerate(zip(self.models[timeframe], self.scalers[timeframe])):
                model_path = timeframe_dir / f"model_{i}.joblib"
                scaler_path = timeframe_dir / f"scaler_{i}.joblib"
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata = {
                "timeframe": timeframe,
                "feature_importance": self.feature_importance[timeframe].tolist(),
                "model_count": len(self.models[timeframe]),
                "training_timestamp": datetime.now().isoformat()
            }
            
            metadata_path = timeframe_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Models saved for {timeframe}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model save failed for {timeframe}: {e}")
            return False

    async def load_models(self, timeframe: str) -> bool:
        """
        Load trained models from disk.

        Args:
            timeframe: Timeframe string

        Returns:
            bool: True if load successful
        """
        try:
            timeframe_dir = Path(self.models_dir) / timeframe
            
            if not timeframe_dir.exists():
                self.logger.warning(f"⚠️ No models found for {timeframe}")
                return False
            
            # Load models
            models = []
            scalers = []
            
            i = 0
            while True:
                model_path = timeframe_dir / f"model_{i}.joblib"
                scaler_path = timeframe_dir / f"scaler_{i}.joblib"
                
                if not model_path.exists() or not scaler_path.exists():
                    break
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                models.append(model)
                scalers.append(scaler)
                i += 1
            
            if not models:
                self.logger.warning(f"⚠️ No models loaded for {timeframe}")
                return False
            
            # Load metadata
            metadata_path = timeframe_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    import json
                    metadata = json.load(f)
                    self.feature_importance[timeframe] = np.array(metadata["feature_importance"])
            
            self.models[timeframe] = models
            self.scalers[timeframe] = scalers
            
            self.logger.info(f"✅ Loaded {len(models)} models for {timeframe}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model load failed for {timeframe}: {e}")
            return False

    async def get_model_performance(self, timeframe: str) -> Dict[str, Any]:
        """
        Get performance metrics for a timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            Dict containing performance metrics
        """
        try:
            if timeframe not in self.models:
                return {"error": f"No models found for {timeframe}"}
            
            # Load performance metrics if available
            performance_path = Path(self.models_dir) / timeframe / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    import json
                    return json.load(f)
            else:
                return {"error": f"No performance data found for {timeframe}"}
                
        except Exception as e:
            self.logger.error(f"Performance retrieval failed for {timeframe}: {e}")
            return {"error": str(e)}

    async def get_feature_importance(self, timeframe: str) -> Dict[str, Any]:
        """
        Get feature importance for a timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            Dict containing feature importance
        """
        try:
            if timeframe not in self.feature_importance:
                return {"error": f"No feature importance found for {timeframe}"}
            
            importance = self.feature_importance[timeframe]
            
            # Get feature names from feature engineering
            feature_names = self.feature_engineering.feature_categories
            
            # Create feature importance dict
            importance_dict = {}
            for i, name in enumerate(feature_names):
                if i < len(importance):
                    importance_dict[name] = float(importance[i])
            
            return {
                "timeframe": timeframe,
                "feature_importance": importance_dict,
                "top_features": sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance retrieval failed for {timeframe}: {e}")
            return {"error": str(e)}