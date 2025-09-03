# src/training/steps/sr_outcome_model_trainer.py

"""S/R Outcome Model Trainer."

Trains ML models to predict S/R outcomes (breakout/rebounce/consolidation)
using LightGBM + XGBoost ensemble with comprehensive feature engineering and time-series validation.
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.centralized_decorators import (
import asyncio

    validate_feature_engineering_with_lookahead_bias_detection,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
import copy
import os

warnings.filterwarnings("ignore")


class SROutcomeModelTrainer:
    """Trainer for S/R outcome prediction models using LightGBM + XGBoost ensemble."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("SROutcomeModelTrainer")

        # Model configuration
        self.model_config = config.get("sr_outcome_model", {})
        self.model_type = self.model_config.get(
            "model_type", "ensemble",
        )  # ensemble, lightgbm, xgboost
        self.feature_config = self.model_config.get("features", {})

        # Training configuration
        self.training_config = self.model_config.get("training", {})
        self.validation_months = self.training_config.get("validation_months", 1)
        self.training_months = self.training_config.get("training_months", 3)
        self.min_samples_per_class = self.training_config.get(
            "min_samples_per_class", 1000,
        )

        # Ensemble configuration
        self.ensemble_config = self.model_config.get("ensemble", {})
        self.use_ensemble = self.ensemble_config.get("use_ensemble", True)
        self.ensemble_weights = self.ensemble_config.get(
            "weights", [0.6, 0.4],
        )  # LightGBM, XGBoost
        self.voting_method = self.ensemble_config.get("voting", "soft")  # soft, hard

        # Feature engineering configuration
        self.use_temporal_features = self.feature_config.get(
            "use_temporal_features", True,
        )
        self.use_volatility_regime = self.feature_config.get(
            "use_volatility_regime", True,
        )
        self.use_order_flow = self.feature_config.get("use_order_flow", False)

        # Model artifacts
        self.artifacts_dir = self.model_config.get("artifacts_dir", "models/sr_outcome")
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Initialize components
        # Initialize SRBreakoutPredictor with optimized parameters
        sr_config = config.copy()
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        self.sr_predictor = SRBreakoutPredictor(sr_config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}  # Store individual models
        self.ensemble_model = None
        self.feature_names = []

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="S/R outcome model initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the S/R outcome model trainer."""
        try:
            self.logger.info("Initializing S/R Outcome Model Trainer...")

            # Initialize SR predictor
            sr_init_success = await self.sr_predictor.initialize()
            if not sr_init_success:
                self.logger.warning("Failed to initialize SRBreakoutPredictor")

            # Initialize label encoder
            self.label_encoder.fit(["breakout", "rebounce", "consolidation"])

            self.logger.info("✅ S/R Outcome Model Trainer initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to initialize S/R Outcome Model Trainer: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="S/R outcome model training",
    )
    async def train_model(self, training_data: dict[str, pd.DataFrame]) -> bool:
        """Train the S/R outcome prediction model ensemble."""
        try:
            self.logger.info("🔄 Starting S/R outcome model training...")

            # Prepare training data
            prepared_data = await self._prepare_training_data(training_data)
            if prepared_data is None:
                self.logger.error("Failed to prepare training data")
                return False

            # Feature engineering
            X, y = await self._engineer_features(prepared_data)
            if X is None or y is None:
                self.logger.error("Failed to engineer features")
                return False

            # Train models based on configuration
            if self.use_ensemble:
                training_result = await self._train_ensemble_models(X, y)
            # Train single model based on model_type
            elif self.model_type == "lightgbm":
                training_result = await self._train_lightgbm_model(X, y)
            elif self.model_type == "xgboost":
                training_result = await self._train_xgboost_model(X, y)
            elif self.model_type == "logistic":
                training_result = await self._train_logistic_model(X, y)
            else:
                self.logger.error(f"Unknown model_type: {self.model_type}")
                return False

            return bool(training_result)
        except Exception as e:
            self.logger.exception(f"Error during model training: {e}")
            return False

    async def _prepare_training_data(
        self, training_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame | None:
        """Prepare training data with S/R context and outcome labeling."""
        try:
            self.logger.info("🔄 Preparing training data...")

            # Combine data from different timeframes
            combined_data = pd.DataFrame()

            for timeframe, data in training_data.items():
                if data.empty:
                    continue

                self.logger.info(f"Processing {timeframe} data: {len(data)} samples")

                # Add timeframe identifier
                data_copy = data.copy()
                data_copy["timeframe"] = timeframe

                # Add S/R context and outcome labels
                labeled_data = await self._label_sr_outcomes(data_copy, timeframe)
                if labeled_data is not None:
                    combined_data = pd.concat([combined_data, labeled_data], ignore_index=True)

            if combined_data.empty:
                self.logger.error("No valid training data found")
                return None

            self.logger.info(f"✅ Prepared training data: {len(combined_data)} samples")
            return combined_data
        except Exception as e:
            self.logger.exception(f"Error preparing training data: {e}")
            return None

    async def _label_sr_outcomes(
        self, data: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame | None:
        """Label S/R outcomes for training data."""
        try:
            if data.empty:
                return None

            # Sample data for efficiency (process every 10th row for large datasets)
            sample_interval = max(1, len(data) // 5000)  # Sample up to 5000 points per timeframe
            sample_data = data.iloc[::sample_interval].copy()

            labeled_samples: list[dict[str, Any]] = []

            for idx, row in sample_data.iterrows():
                try:
                    # Get current price and market context
                    current_price = row["close"]

                    # Create market data slice for S/R analysis
                    market_slice = data.loc[:idx].tail(100)
                    if len(market_slice) < 20:
                        continue

                    # Get S/R context and outcome prediction using centralized logic
                    sr_context = await self.sr_predictor.get_sr_context(
                        market_data=market_slice, current_price=current_price
                    )
                    sr_outcome = await self.sr_predictor.predict_sr_outcome(
                        market_data=market_slice, current_price=current_price, sr_context=sr_context
                    )

                    # Check if near S/R level
                    is_near_sr = sr_outcome.get("is_near_sr_level", False)

                    if is_near_sr:
                        # Create labeled sample
                        sample = {
                            "timestamp": row.get("timestamp", idx),
                            "timeframe": timeframe,
                            "price": current_price,
                            "outcome": sr_outcome.get("outcome", "consolidation"),
                            "confidence": sr_outcome.get("confidence", 0.5),
                            "sr_context": sr_context,
                            "market_data": market_slice.tail(20).to_dict(
                                "records",
                            ),  # Last 20 bars
                            "features": await self._extract_features(
                                market_data=market_slice, current_price=current_price, sr_context=sr_context
                            ),
                        }
                        labeled_samples.append(sample)
                except Exception as e:
                    self.logger.debug(f"Error labeling sample {idx}: {e}")
                    continue

            if not labeled_samples:
                return None

            # Convert to DataFrame
            labeled_df = pd.DataFrame(labeled_samples)

            # Balance classes
            balanced_df = self._balance_classes(labeled_df)

            self.logger.info(f"✅ Labeled {len(balanced_df)} samples for {timeframe}")
            return balanced_df
        except Exception as e:
            self.logger.exception(f"Error labeling S/R outcomes: {e}")
            return None

    async def _extract_features(
        self, market_data: pd.DataFrame, current_price: float, sr_context: dict
    ) -> dict[str, float]:
        """Extract comprehensive features for S/R outcome prediction."""
        try:
            features: dict[str, float] = {}

            # Price-based features
            features["price_change_1m"] = (
                market_data["close"].pct_change().iloc[-1]
                if len(market_data) > 1
                else 0
            )
            features["price_change_5m"] = (
                market_data["close"].pct_change(5).iloc[-1]
                if len(market_data) > 5
                else 0
            )
            features["price_change_15m"] = (
                market_data["close"].pct_change(15).iloc[-1]
                if len(market_data) > 15
                else 0
            )
            features["price_volatility"] = (
                market_data["close"].rolling(20).std().iloc[-1]
                if len(market_data) >= 20
                else 0
            )

            # Volume-based features
            features["volume_ratio"] = (
                (
                    market_data["volume"].iloc[-1]
                    / market_data["volume"].rolling(20).mean().iloc[-1]
                )
                if len(market_data) >= 20
                else 1.0
            )
            features["volume_momentum"] = (
                market_data["volume"].pct_change().iloc[-1]
                if len(market_data) > 1
                else 0
            )
            features["volume_volatility"] = (
                market_data["volume"].rolling(10).std().iloc[-1]
                if len(market_data) >= 10
                else 0
            )

            # Technical indicators
            features["rsi"] = (
                self._calculate_rsi(market_data["close"]).iloc[-1]
                if len(market_data) >= 14
                else 50
            )
            features["macd"] = (
                self._calculate_macd(market_data["close"]).iloc[-1]
                if len(market_data) >= 26
                else 0
            )
            features["bb_position"] = (
                self._calculate_bb_position(market_data["close"]).iloc[-1]
                if len(market_data) >= 20
                else 0.5
            )

            # S/R-specific features
            if sr_context:
                nearest_support = sr_context.get("nearest_support", current_price)
                nearest_resistance = sr_context.get("nearest_resistance", current_price)

                features["distance_to_support"] = (
                    current_price - nearest_support
                ) / current_price
                features["distance_to_resistance"] = (
                    nearest_resistance - current_price
                ) / current_price
                features["support_strength"] = sr_context.get("support_strength", 0.5)
                features["resistance_strength"] = sr_context.get(
                    "resistance_strength", 0.5,
                )

                # Pivot level features
                pivot_levels = sr_context.get("pivot_levels", {})
                if pivot_levels:
                    features["nearest_pivot_strength"] = pivot_levels.get(
                        "nearest_strength", 0.5,
                    )
                    features["pivot_touches"] = pivot_levels.get("nearest_touches", 0)
                else:
                    features["nearest_pivot_strength"] = 0.5
                    features["pivot_touches"] = 0

            # Market context features
            features["market_trend"] = self._calculate_market_trend(market_data)
            features["momentum_strength"] = self._calculate_momentum_strength(
                market_data,
            )

            # Temporal features
            if self.use_temporal_features:
                features["time_since_sr_touch"] = self._calculate_time_since_sr_touch(
                    market_data=market_data, sr_context=sr_context
                )
                features["sr_touch_frequency"] = self._calculate_sr_touch_frequency(
                    market_data=market_data, sr_context=sr_context
                )

            # Volatility regime features
            if self.use_volatility_regime:
                features["volatility_regime"] = self._classify_volatility_regime(
                    market_data,
                )
                features["atr_ratio"] = self._calculate_atr_ratio(market_data)

            return features
        except Exception as e:
            self.logger.exception(f"Error extracting features: {e}")
            return {}

    def _balance_classes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Balance classes to handle imbalanced S/R outcomes."""
        try:
            # Count samples per class
            class_counts = data["outcome"].value_counts()
            min_count = min(class_counts.min(), self.min_samples_per_class)

            balanced_samples = []

            for outcome in ["breakout", "rebounce", "consolidation"]:
                outcome_data = data[data["outcome"] == outcome]

                if len(outcome_data) > min_count:
                    # Sample down to min_count
                    balanced_samples.append(
                        outcome_data.sample(n=min_count, random_state=42)
                    )
                else:
                    # Keep all samples if below min_count
                    balanced_samples.append(outcome_data)

            balanced_df = pd.concat(balanced_samples, ignore_index=True)

            self.logger.info(
                f"Balanced classes: {balanced_df['outcome'].value_counts().to_dict()}",
            )
            return balanced_df
        except Exception as e:
            self.logger.exception(f"Error balancing classes: {e}")
            return data

    @validate_feature_engineering_with_lookahead_bias_detection
    async def _engineer_features(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Engineer features for model training."""
        try:
            self.logger.info("🔄 Engineering features...")

            # Extract features from all samples
            feature_vectors: list[list[float]] = []
            labels: list[str] = []

            for _, row in data.iterrows():
                features = row.get("features", {})
                if features:
                    # Create feature vector
                    feature_vector = self._create_feature_vector(features)
                    if feature_vector is not None:
                        feature_vectors.append(feature_vector)
                        labels.append(row["outcome"])

            if not feature_vectors:
                self.logger.error("No valid feature vectors found")
                return None, None

            # Convert to numpy arrays
            X = np.array(feature_vectors)
            y = np.array(labels)

            # Encode labels
            y_encoded = self.label_encoder.transform(y)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Store feature names
            self.feature_names = self._get_feature_names()

            self.logger.info(f"✅ Engineered features: {X_scaled.shape}")
            return X_scaled, y_encoded
        except Exception as e:
            self.logger.exception(f"Error engineering features: {e}")
            return None, None

    def _create_feature_vector(self, features: dict) -> list[float] | None:
        """Create feature vector from features dictionary."""
        try:
            feature_names = self._get_feature_names()
            feature_vector = []

            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0.0))

            return feature_vector
        except Exception as e:
            self.logger.exception(f"Error creating feature vector: {e}")
            return None

    def _get_feature_names(self) -> list[str]:
        """Get list of feature names in order."""
        base_features = [
            "price_change_1m",
            "price_change_5m",
            "price_change_15m",
            "price_volatility",
            "volume_ratio",
            "volume_momentum",
            "volume_volatility",
            "rsi",
            "macd",
            "bb_position",
            "distance_to_support",
            "distance_to_resistance",
            "support_strength",
            "resistance_strength",
            "nearest_pivot_strength",
            "pivot_touches",
            "market_trend",
            "momentum_strength",
        ]

        if self.use_temporal_features:
            base_features.extend(["time_since_sr_touch", "sr_touch_frequency"])

        if self.use_volatility_regime:
            base_features.extend(["volatility_regime", "atr_ratio"])

        return base_features

    async def _train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train LightGBM model with hyperparameter optimization."""
        try:
            self.logger.info("🔄 Training LightGBM model...")

            # Calculate class weights
            class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
            weight_dict = dict(zip(np.unique(y), class_weights))

            # Create sample weights
            sample_weights = np.array([weight_dict[label] for label in y])

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Hyperparameter optimization for LightGBM
            best_params = await self._optimize_lightgbm_hyperparameters(
                X, y, sample_weights, tscv
            )

            # Train final model with best parameters
            final_model = lgb.LGBMClassifier(**best_params, random_state=42)
            final_model.fit(X, y, sample_weight=sample_weights)

            # Store model
            self.models["lgb"] = final_model

            # If not using ensemble, set as primary model
            if not self.use_ensemble:
                self.ensemble_model = final_model

            # Evaluate model
            await self._evaluate_model(X, y, model_name="LightGBM")

            self.logger.info("✅ LightGBM model training completed")
            return True
        except Exception as e:
            self.logger.exception(f"Error training LightGBM model: {e}")
            return False

    async def _train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train XGBoost model with hyperparameter optimization."""
        try:
            self.logger.info("🔄 Training XGBoost model...")

            # Calculate class weights
            class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
            weight_dict = dict(zip(np.unique(y), class_weights))

            # Create sample weights
            sample_weights = np.array([weight_dict[label] for label in y])

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Hyperparameter optimization for XGBoost
            best_params = await self._optimize_xgboost_hyperparameters(
                X, y, sample_weights, tscv
            )

            # Train final model with best parameters
            final_model = xgb.XGBClassifier(**best_params, random_state=42)
            final_model.fit(X, y, sample_weight=sample_weights)

            # Store model
            self.models["xgb"] = final_model

            # If not using ensemble, set as primary model
            if not self.use_ensemble:
                self.ensemble_model = final_model

            # Evaluate model
            await self._evaluate_model(X, y, model_name="XGBoost")

            self.logger.info("✅ XGBoost model training completed")
            return True
        except Exception as e:
            self.logger.exception(f"Error training XGBoost model: {e}")
            return False

    async def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train LightGBM and XGBoost models and create an ensemble."""
        try:
            self.logger.info("🔄 Training LightGBM and XGBoost ensemble...")

            # Train LightGBM
            lgb_model_success = await self._train_lightgbm_model(X, y)
            if not lgb_model_success:
                self.logger.error("Failed to train LightGBM model for ensemble")
                return False

            # Train XGBoost
            xgb_model_success = await self._train_xgboost_model(X, y)
            if not xgb_model_success:
                self.logger.error("Failed to train XGBoost model for ensemble")
                return False

            # Create ensemble model
            self.ensemble_model = VotingClassifier(
                estimators=[("lgb", self.models["lgb"]), ("xgb", self.models["xgb"])],
                voting=self.voting_method,
                weights=self.ensemble_weights,
            )

            # Fit ensemble
            self.ensemble_model.fit(X, y)

            # Evaluate ensemble
            await self._evaluate_model(X, y, model_name="Ensemble")

            self.logger.info("✅ Ensemble training completed")
            return True
        except Exception as e:
            self.logger.exception(f"Error training ensemble models: {e}")
            return False

    async def _optimize_lightgbm_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, tscv: TimeSeriesSplit,
    ) -> dict:
        """Optimize LightGBM hyperparameters using Optuna."""
        try:

            def objective(trial):
                params = {
                    "objective": "multiclass",
                    "num_class": 3,
                    "boosting_type": "gbdt",
                    "metric": "multi_logloss",
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.1, log=True
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "max_depth": trial.suggest_int("max_depth", 4, 12),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
                    "feature_fraction": trial.suggest_float(
                        "feature_fraction", 0.6, 0.9,
                    ),
                    "bagging_fraction": trial.suggest_float(
                        "bagging_fraction", 0.6, 0.9,
                    ),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.3, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 0.01, 0.3, log=True
                    ),
                    "random_state": 42,
                }

                # Cross-validation
                scores: list[float] = []
                for train_idx, val_idx in tscv.split(X):
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    w_train = sample_weights[train_idx]
                    X_val = X[val_idx]
                    y_val = y[val_idx]
                    w_val = sample_weights[val_idx]

                    model = lgb.LGBMClassifier(**params, random_state=42)
                    model.fit(X_train, y_train, sample_weight=w_train)

                    y_pred_proba = model.predict_proba(X_val)
                    score = roc_auc_score(y_val, y_pred_proba, multi_class="ovr")
                    scores.append(score)

                return float(np.mean(scores))

            # Get trials from training input or use default
            sr_lightgbm_trials = getattr(self, 'training_input', {}).get("sr_lightgbm_trials", 30)
            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=sr_lightgbm_trials)

            best_params = study.best_params
            best_params.update(
                {
                    "objective": "multiclass",
                    "num_class": 3,
                    "boosting_type": "gbdt",
                    "metric": "multi_logloss",
                    "random_state": 42,
                },
            )

            self.logger.info(f"Best LightGBM hyperparameters: {best_params}")
            return best_params
        except Exception as e:
            self.logger.exception(f"Error optimizing LightGBM hyperparameters: {e}")
            # Return default parameters
            return {
                "objective": "multiclass",
                "num_class": 3,
                "boosting_type": "gbdt",
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 8,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
            }

    async def _optimize_xgboost_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, tscv: TimeSeriesSplit,
    ) -> dict:
        """Optimize XGBoost hyperparameters using Optuna."""
        try:

            def objective(trial):
                params = {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.1, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 0.9,
                    ),
                    "gamma": trial.suggest_float("gamma", 0, 0.5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.3, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 0.01, 0.3, log=True
                    ),
                    "random_state": 42,
                }

                # Cross-validation
                scores: list[float] = []
                for train_idx, val_idx in tscv.split(X):
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    w_train = sample_weights[train_idx]
                    X_val = X[val_idx]
                    y_val = y[val_idx]
                    w_val = sample_weights[val_idx]

                    model = xgb.XGBClassifier(**params, random_state=42)
                    model.fit(X_train, y_train, sample_weight=w_train)

                    y_pred_proba = model.predict_proba(X_val)
                    score = roc_auc_score(y_val, y_pred_proba, multi_class="ovr")
                    scores.append(score)

                return float(np.mean(scores))

            # Get trials from training input or use default
            sr_xgboost_trials = getattr(self, 'training_input', {}).get("sr_xgboost_trials", 30)
            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=sr_xgboost_trials)

            best_params = study.best_params
            best_params.update(
                {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "random_state": 42,
                },
            )

            self.logger.info(f"Best XGBoost hyperparameters: {best_params}")
            return best_params
        except Exception as e:
            self.logger.exception(f"Error optimizing XGBoost hyperparameters: {e}")
            # Return default parameters
            return {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
            }

    async def _evaluate_model(
        self, X: np.ndarray, y: np.ndarray, model_name: str = "Model"
    ) -> None:
        """Evaluate the trained model."""
        try:
            # Use appropriate model for evaluation
            if model_name == "Ensemble" and self.ensemble_model is not None:
                model_to_evaluate = self.ensemble_model
            elif model_name == "LightGBM" and "lgb" in self.models:
                model_to_evaluate = self.models["lgb"]
            elif model_name == "XGBoost" and "xgb" in self.models:
                model_to_evaluate = self.models["xgb"]
            else:
                model_to_evaluate = self.ensemble_model

            if model_to_evaluate is None:
                self.logger.warning(f"No model available for evaluation: {model_name}")
                return

            # Predictions
            y_pred = model_to_evaluate.predict(X)
            y_pred_proba = model_to_evaluate.predict_proba(X)

            # Metrics
            report = classification_report(
                y, y_pred, target_names=self.label_encoder.classes_
            )
            conf_matrix = confusion_matrix(y, y_pred)
            auc_score = roc_auc_score(y, y_pred_proba, multi_class="ovr")

            # Feature importance (for individual models)
            feature_importance = None
            if hasattr(model_to_evaluate, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {
                        "feature": self.feature_names,
                        "importance": model_to_evaluate.feature_importances_,
                    },
                ).sort_values("importance", ascending=False)
            elif model_name == "Ensemble":
                # For ensemble, combine feature importance from both models
                lgb_importance = (
                    self.models["lgb"].feature_importances_
                    if "lgb" in self.models
                    else None
                )
                xgb_importance = (
                    self.models["xgb"].feature_importances_
                    if "xgb" in self.models
                    else None
                )

                if lgb_importance is not None and xgb_importance is not None:
                    # Weighted average of feature importance
                    weighted_importance = (
                        lgb_importance * self.ensemble_weights[0]
                        + xgb_importance * self.ensemble_weights[1]
                    )
                    feature_importance = pd.DataFrame(
                        {
                            "feature": self.feature_names,
                            "importance": weighted_importance,
                        },
                    ).sort_values("importance", ascending=False)

            # Log results
            self.logger.info(f"Model Evaluation Results for {model_name}:")
            self.logger.info(f"AUC Score: {auc_score:.4f}")
            self.logger.info(f"Classification Report:\n{report}")
            if feature_importance is not None:
                self.logger.info(f"Top 10 Features:\n{feature_importance.head(10)}")

            # Save evaluation results
            evaluation_results = {
                "model_name": model_name,
                "auc_score": float(auc_score),
                "classification_report": report,
                "confusion_matrix": conf_matrix.tolist(),
                "feature_importance": feature_importance.to_dict("records")
                if feature_importance is not None
                else None,
                "timestamp": datetime.now().isoformat(),
            }

            with open(
                os.path.join(self.artifacts_dir, f"{model_name.lower()}_evaluation_results.json"),
                "w",
            ) as f:
                json.dump(evaluation_results, f, indent=2)
        except Exception as e:
            self.logger.exception(f"Error evaluating model: {e}")

    async def _save_model_artifacts(self) -> None:
        """Save model artifacts and metadata."""
        try:
            # Save individual models
            if "lgb" in self.models:
                lgb_path = os.path.join(self.artifacts_dir, "lightgbm_model.pkl")
                with open(lgb_path, "wb") as f:
                    pickle.dump(self.models["lgb"], f)

            if "xgb" in self.models:
                xgb_path = os.path.join(self.artifacts_dir, "xgboost_model.pkl")
                with open(xgb_path, "wb") as f:
                    pickle.dump(self.models["xgb"], f)

            # Save ensemble model
            if self.ensemble_model is not None:
                ensemble_path = os.path.join(self.artifacts_dir, "ensemble_model.pkl")
                with open(ensemble_path, "wb") as f:
                    pickle.dump(self.ensemble_model, f)

            # Save scaler
            scaler_path = os.path.join(self.artifacts_dir, "sr_outcome_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            # Save label encoder
            encoder_path = os.path.join(self.artifacts_dir, "sr_outcome_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)

            # Save feature names
            feature_names_path = os.path.join(self.artifacts_dir, "feature_names.json")
            with open(feature_names_path, "w") as f:
                json.dump(self.feature_names, f)

            # Save configuration
            config_save = {
                "model_config": self.model_config,
                "ensemble_config": self.ensemble_config,
                "feature_names": self.feature_names,
                "training_timestamp": datetime.now().isoformat(),
                "model_type": self.model_type,
                "use_ensemble": self.use_ensemble,
                "ensemble_weights": self.ensemble_weights,
                "voting_method": self.voting_method,
            }

            config_path = os.path.join(self.artifacts_dir, "model_config.json")
            with open(config_path, "w") as f:
                json.dump(config_save, f, indent=2)

            self.logger.info(f"✅ Model artifacts saved to {self.artifacts_dir}")
        except Exception as e:
            self.logger.exception(f"Error saving model artifacts: {e}")

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Make prediction using the trained ensemble or individual model."""
        try:
            if self.ensemble_model is None:
                return {
                    "probabilities": {
                        "breakout": 0.33,
                        "rebounce": 0.33,
                        "consolidation": 0.34,
                    },
                    "confidence": 0.5,
                    "outcome": "consolidation",
                    "model_type": "none",
                }

            # Create feature vector
            feature_vector = self._create_feature_vector(features)
            if feature_vector is None:
                return {
                    "probabilities": {
                        "breakout": 0.33,
                        "rebounce": 0.33,
                        "consolidation": 0.34,
                    },
                    "confidence": 0.5,
                    "outcome": "consolidation",
                    "model_type": "error",
                }

            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])

            # Make prediction
            if self.use_ensemble and self.ensemble_model is not None:
                # Use ensemble prediction
                y_pred_proba = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
                y_pred = self.ensemble_model.predict(feature_vector_scaled)[0]
                model_type = "ensemble"
            else:
                # Use individual model prediction
                y_pred_proba = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
                y_pred = self.ensemble_model.predict(feature_vector_scaled)[0]
                model_type = self.model_type

            # Map prediction to outcome
            outcome_mapping = {0: "breakout", 1: "rebounce", 2: "consolidation"}
            outcome = outcome_mapping.get(int(y_pred), "consolidation")

            # Create probability dict
            prob_dict = {
                "breakout": float(y_pred_proba[0]),
                "rebounce": float(y_pred_proba[1]),
                "consolidation": float(y_pred_proba[2]),
            }

            # Calculate confidence
            confidence = float(max(y_pred_proba))

            return {
                "probabilities": prob_dict,
                "confidence": confidence,
                "outcome": outcome,
                "model_type": model_type,
            }
        except Exception as e:
            self.logger.exception(f"Error making prediction: {e}")
            return {
                "probabilities": {
                    "breakout": 0.33,
                    "rebounce": 0.33,
                    "consolidation": 0.34,
                },
                "confidence": 0.5,
                "outcome": "consolidation",
                "model_type": "error",
            }

    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26
    ) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def _calculate_bb_position(
        self, prices: pd.Series, period: int = 20, std: int = 2
    ) -> pd.Series:
        """Calculate Bollinger Band position."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        # Position within bands (0, at lower band, 1, at upper band)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position.clip(0, 1)

    def _calculate_market_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate market trend strength."""
        try:
            if len(market_data) < 20:
                return 0.0

            prices = market_data["close"].values
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]

            avg_price = np.mean(prices)
            normalized_slope = slope / avg_price if avg_price > 0 else 0

            return float(np.clip(normalized_slope * 100, -1, 1))
        except Exception as e:
            self.logger.exception(f"Error calculating market trend: {e}")
            return 0.0

    def _calculate_momentum_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum strength."""
        try:
            if len(market_data) < 10:
                return 0.0

            short_momentum = (
                market_data["close"].pct_change(5).iloc[-1]
                if len(market_data) > 5
                else 0
            )
            long_momentum = (
                market_data["close"].pct_change(20).iloc[-1]
                if len(market_data) > 20
                else 0
            )

            momentum = short_momentum * 0.7 + long_momentum * 0.3

            return float(np.clip(momentum * 100, -1, 1))
        except Exception as e:
            self.logger.exception(f"Error calculating momentum strength: {e}")
            return 0.0

    def _calculate_time_since_sr_touch(
        self, market_data: pd.DataFrame, sr_context: dict
    ) -> float:
        """Calculate time since last S/R level touch."""
        # Placeholder implementation
        return 0.5

    def _calculate_sr_touch_frequency(
        self, market_data: pd.DataFrame, sr_context: dict
    ) -> float:
        """Calculate S/R level touch frequency."""
        # Placeholder implementation
        return 0.5

    def _classify_volatility_regime(self, market_data: pd.DataFrame) -> float:
        """Classify volatility regime."""
        try:
            if len(market_data) < 20:
                return 0.5

            # Calculate ATR-based volatility
            high_low = market_data["high"] - market_data["low"]
            high_close = np.abs(market_data["high"] - market_data["close"].shift())
            low_close = np.abs(market_data["low"] - market_data["close"].shift())

            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]

            # Normalize ATR by price
            avg_price = market_data["close"].mean()
            normalized_atr = atr / avg_price if avg_price > 0 else 0

            # Classify regime (0, low volatility, 1, high volatility)
            return float(min(1.0, normalized_atr * 100))
        except Exception as e:
            self.logger.exception(f"Error classifying volatility regime: {e}")
            return 0.5

    def _calculate_atr_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ATR ratio for volatility analysis."""
        try:
            if len(market_data) < 20:
                return 1.0

            # Calculate current ATR vs historical ATR
            high_low = market_data["high"] - market_data["low"]
            high_close = np.abs(market_data["high"] - market_data["close"].shift())
            low_close = np.abs(market_data["low"] - market_data["close"].shift())

            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            current_atr = true_range.rolling(14).mean().iloc[-1]
            historical_atr = true_range.rolling(50).mean().iloc[-1]

            return float(current_atr / historical_atr) if historical_atr > 0 else 1.0
        except Exception as e:
            self.logger.exception(f"Error calculating ATR ratio: {e}")
            return 1.0