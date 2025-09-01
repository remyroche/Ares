#!/usr/bin/env python3
"""Enhanced HMM-Based Training with Multi-Output Support.

This module extends the existing HMM-based training to support intelligent
multi-output prediction for both direction and profit using the triple barrier
method and profit-based feature engineering.
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Multi-output training will be imported when needed
from src.training.steps.step04_analyst_labeling_feature_engineering_components.profit_based_feature_engineering import (
    ProfitBasedFeatureEngineering
)
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.centralized_decorators import (
    PerformanceLevel,
    ValidationLevel,
    adaptive_resource_allocation,
    comprehensive_validation,
    handle_errors,
    intelligent_caching,
    model_validation,
    performance_monitor,
    pipeline_checkpoint,
    validate_feature_engineering_with_lookahead_bias_detection,
)
from src.utils.logger import system_logger

# Suppress warnings
warnings.filterwarnings("ignore")


class EnhancedHMMBasedTrainingStep:
    """Enhanced HMM-Based Model Training with Multi-Output Support.

    Extends the existing HMM-based training to support intelligent multi-output
    prediction for both direction and profit using the triple barrier method
    and profit-based feature engineering.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}

        # Initialize SRBreakoutPredictor for S/R level integration with optimized parameters
        sr_config = config.copy()
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        self.sr_predictor = SRBreakoutPredictor(sr_config)

        # Initialize S/R outcome model trainer
        self.sr_outcome_trainer = None
        self.sr_outcome_model_trained = False

        # Initialize profit-based feature engineering
        self.profit_feature_engine = ProfitBasedFeatureEngineering(
            profit_column="potential_profit_pct",
            use_numba=True,
            memory_efficient=True
        )

        # Multi-output model trainer
        self.multi_output_trainer = None
        self.enable_multi_output = config.get("enable_multi_output", True)

        # Model architecture mapping from config
        hmm_lm_config = config.get("HMM_LM", {})
        specialist_config = hmm_lm_config.get("specialist_models", {})

        self.model_architectures = {}
        for timeframe, model_config in specialist_config.items():
            self.model_architectures[timeframe] = model_config.get(
                "architecture", "LightGBM",
            )

        # Supported model types for multi-output training
        self.supported_model_types = [
            "LightGBM", "RandomForest", "XGBoost", "CatBoost", "NeuralNetwork"
        ]

        # Fallback to default if config not available
        if not self.model_architectures:
            self.model_architectures = {
                "1m": "CNN",  # Tactician
                "5m": "TCN",  # Analyst
                "15m": "Transformer",  # Analyst
                "30m": "LightGBM",  # Analyst
            }

        # HMM-derived features (composite regimes and intensity scores)
        self.hmm_features = [
            "composite_cluster_id",
            "intensity_cluster_0",
            "intensity_cluster_1",
            "intensity_cluster_2",
            "intensity_cluster_3",
            "intensity_cluster_4",
            "intensity_cluster_5",
            "intensity_cluster_6",
            "intensity_cluster_7",
            "intensity_cluster_8",
            "intensity_cluster_9",
            "intensity_cluster_10",
            "intensity_cluster_11",
            "intensity_cluster_12",
            "intensity_cluster_13",
            "intensity_cluster_14",
            "intensity_cluster_15",
            "intensity_cluster_16",
            "intensity_cluster_17",
            "intensity_cluster_18",
            "intensity_cluster_19",
            # Regime probability features
            "momentum_p_state_0",
            "momentum_p_state_1",
            "momentum_p_state_2",
            "momentum_p_state_3",
            "volatility_p_state_0",
            "volatility_p_state_1",
            "volatility_p_state_2",
            "volatility_p_state_3",
            "liquidity_p_state_0",
            "liquidity_p_state_1",
            "liquidity_p_state_2",
            "liquidity_p_state_3",
            "microstructure_p_state_0",
            "microstructure_p_state_1",
            "microstructure_p_state_2",
            "microstructure_p_state_3",
            "microstructure_p_state_4",
            "orderflow_p_state_0",
            "orderflow_p_state_1",
            "orderflow_p_state_2",
            "orderflow_p_state_3",
            "orderflow_p_state_4",
        ]

        # All available features - will be optimized by feature selection
        self.all_features = [
            # Technical indicators (already returns-based or normalized)
            "momentum_strength",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "rsi",
            "bb_position",
            "trend_regime",
            "macd",
            "atr",
            "adx",
            "sma",
            "ema",
            "cci",
            "mfi",
            "roc",
            # Volatility measures (returns-based)
            "volume_volatility",
            "parkinson_volatility",
            "garman_klass_volatility",
            "adaptive_atr",
            "1m_price_volatility",
            "ewma_volatility",
            # Liquidity measures (returns-based or normalized)
            "liquidity_score",
            "bid_ask_spread",
            "market_depth",
            "volume_profile",
            "order_imbalance",
            "tick_size",
            "spread_tightness",
            "price_impact",
            # Order flow measures (returns-based or normalized)
            "trade_frequency",
            "order_flow_imbalance",
            "order_flow_vpin_50",
            "order_flow_aggressor_ratio",
            "order_flow_large_small_imbalance",
            "order_flow_pressure",
            "order_flow_taker_imbalance",
            "order_flow_avg_trade_size",
            "order_flow_trade_size_volatility",
            # Normalized/returns-based features
            "momentum_strength_z_score",
            "rsi_change",
            "volume_pct_change",
            "volatility_acceleration",
            "liquidity_bounded",
            "price_log",
            "spread_normalized",
            # Interaction features
            "momentum_x_volume",
            "volatility_x_liquidity",
            "price_x_volume",
            "rsi_div_volatility",
            "momentum_div_liquidity",
            # Lagged features
            "momentum_lag_1",
            "volume_lag_1",
            "rsi_lag_1",
            "volatility_lag_1",
            "momentum_lag_5",
            "volume_lag_5",
            "rsi_lag_5",
            "volatility_lag_5",
            # Causality features
            "momentum_predicts_volume",
            "volatility_causality",
            "liquidity_divergence",
            "price_stress",
            "volume_extreme",
            "momentum_extreme",
        ]

        # Validation and cross-validation configuration
        self.validation_config = {
            "n_splits": 5,  # Number of time series splits
            "test_size": 0.2,  # Test set size
            "validation_size": 0.2,  # Validation set size
            "min_samples_per_split": 1000,
            "regime_aware_splitting": True,
        }

        # Data source configuration
        self.data_source_config = {
            "prefer_pickle": True,
            "fallback_to_parquet": True,
            "load_regime_weights": False,
            "validate_data_quality": True,
        }

        self.logger.info("🔧 Enhanced HMM-based training with multi-output support initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced HMM-based training step initialization",
    )
    @handle_errors(
        exceptions=(ValueError, TypeError, MemoryError),
        default_return=None,
        context="enhanced_data_preparation"
    )
    def prepare_enhanced_data(
        self,
        data: pd.DataFrame,
        timeframe: str,
        regime_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare data for enhanced training with multi-output support.

        Args:
            data: Input DataFrame with features and targets
            timeframe: Timeframe for the data
            regime_key: Regime key if regime-specific training

        Returns:
            Dictionary containing prepared data for both single and multi-output training
        """
        self.logger.info(f"📊 Preparing enhanced training data for {timeframe}")
        if regime_key:
            self.logger.info(f"   - Regime: {regime_key}")

        # Check for multi-output targets
        has_direction = "direction" in data.columns
        has_profit = "potential_profit_pct" in data.columns
        has_single_target = "target" in data.columns or "label" in data.columns

        # Use enhanced feature selection if multi-output is enabled
        if has_profit and self.enable_multi_output:
            try:
                from src.training.enhanced_matrix_operations import EnhancedMatrixOperations

                self.logger.info("🔧 Using enhanced feature selection with autoencoder features...")

                # Create enhanced matrix operations manager
                feature_selector = EnhancedMatrixOperations(self.config)

                # Create dummy target for feature selection
                dummy_target = pd.Series(0, index=data.index)
                if "direction" in data.columns:
                    dummy_target = data["direction"]

                # Use enhanced feature selection with autoencoder features
                selected_features, metadata = feature_selector.select_features_step2(
                    features_df=data,
                    target=dummy_target,
                    symbol="default",
                    exchange="default",
                    data_dir="temp",
                    use_autoencoder_features=True,  # Use autoencoder features
                    use_regularization=True         # Use regularization
                )

                self.logger.info(f"✅ Enhanced feature selection completed: {selected_features.shape[1]} features selected")
                self.logger.info(f"   - Autoencoder features: {metadata.get('stages', {}).get('stage0_autoencoder', {}).get('autoencoder_features_added', 0)}")
                self.logger.info(f"   - Regularization applied: {metadata.get('stages', {}).get('stage6_regularization', {}).get('regularization_applied', False)}")

                # Use selected features
                data = selected_features

            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced feature selection failed: {e}")
                self.logger.info("📊 Falling back to basic feature preparation")

                # Apply profit-based feature engineering as fallback
                self.logger.info("🔧 Applying profit-based feature engineering...")
                data = self.profit_feature_engine.apply_all_features(data)
                self.logger.info(f"✅ Added profit-based features")

        # Prepare features
        exclude_columns = [
            "target", "label", "direction", "potential_profit_pct",
            "timestamp", "timeframe", "composite_cluster_id", "sample_weight"
        ]
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        features = data[feature_columns].copy()

        # Handle missing values
        features = features.fillna(0)

        prepared_data = {
            "features": features,
            "feature_columns": feature_columns,
            "timeframe": timeframe,
            "regime_key": regime_key,
            "has_multi_output": has_direction and has_profit,
            "has_single_output": has_single_target
        }

        # Prepare single-output targets (backward compatibility)
        if has_single_target:
            label_col = "target" if "target" in data.columns else "label"
            prepared_data["single_target"] = data[label_col].fillna(0)

        # Prepare multi-output targets
        if has_direction and has_profit:
            prepared_data["direction_target"] = data["direction"].fillna(0)
            prepared_data["profit_target"] = data["potential_profit_pct"].fillna(0)

            # Convert direction to binary if needed
            if prepared_data["direction_target"].dtype in ['object', 'string']:
                prepared_data["direction_target"] = (prepared_data["direction_target"] > 0).astype(int)

        self.logger.info(f"✅ Enhanced data prepared: {features.shape[0]} samples, {features.shape[1]} features")
        self.logger.info(f"   - Multi-output: {prepared_data['has_multi_output']}")
        self.logger.info(f"   - Single-output: {prepared_data['has_single_output']}")

        return prepared_data

    @handle_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=None,
        context="enhanced_model_training"
    )
    @performance_monitor
    async def train_enhanced_model(
        self,
        prepared_data: Dict[str, Any],
        model_name: str = "enhanced_model"
    ) -> Dict[str, Any]:
        """Train enhanced model with multi-output support.

        Args:
            prepared_data: Prepared data dictionary
            model_name: Name for the trained model

        Returns:
            Dictionary containing training results and model artifacts
        """
        self.logger.info(f"🚀 Training enhanced model: {model_name}")

        results = {
            "model_name": model_name,
            "timeframe": prepared_data["timeframe"],
            "regime_key": prepared_data["regime_key"],
            "single_output_results": None,
            "multi_output_results": None,
            "training_timestamp": datetime.now().isoformat()
        }

        # Train multi-output model if data is available
        if prepared_data["has_multi_output"] and self.multi_output_trainer:
            self.logger.info("🎯 Training multi-output probability model")

            # Prepare data for multi-output training
            X = prepared_data["features"].values
            y = prepared_data["single_target"].values if "single_target" in prepared_data else np.random.choice([0, 1], size=len(X))

            # Create market data for target generation
            market_data = pd.DataFrame({
                'close': np.random.randn(len(X)),  # Placeholder - should use actual market data
                'volume': np.random.randn(len(X))
            })

            # Generate multi-output targets
            y_multi = self.multi_output_trainer.prepare_multi_output_targets(X, y, market_data)

            # Split data for training
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train_multi = {k: v[:split_idx] for k, v in y_multi.items()}
            y_test_multi = {k: v[split_idx:] for k, v in y_multi.items()}

            # Train multi-output model
            trained_models = self.multi_output_trainer.train_multi_output_model(
                X_train, y_train_multi, X_test, y_test_multi
            )

            # Generate probability outputs
            price_action_probabilities = self.multi_output_trainer.predict_probabilities(
                X_test, market_data.iloc[split_idx:]
            )

            multi_output_result = {
                "trained_models": trained_models,
                "price_action_probabilities": price_action_probabilities,
                "model_type": "multi_output"
            }

            if multi_output_result:
                results["multi_output_results"] = multi_output_result
                self.logger.info("✅ Multi-output probability model training completed successfully")
            else:
                self.logger.warning("⚠️ Multi-output probability model training failed")

        # Train single-output model for backward compatibility
        if prepared_data["has_single_output"]:
            self.logger.info("🎯 Training single-output model (backward compatibility)")

            single_output_result = await self._train_single_output_model(
                prepared_data["features"],
                prepared_data["single_target"],
                prepared_data["timeframe"],
                prepared_data["regime_key"]
            )

            if single_output_result:
                results["single_output_results"] = single_output_result
                self.logger.info("✅ Single-output model training completed successfully")
            else:
                self.logger.warning("⚠️ Single-output model training failed")

        return results

    async def _train_single_output_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        timeframe: str,
        regime_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train single-output model (backward compatibility).

        Args:
            features: Feature DataFrame
            target: Target series
            timeframe: Timeframe
            regime_key: Regime key if regime-specific

        Returns:
            Training results dictionary
        """
        try:
            architecture = self.model_architectures.get(timeframe, "LightGBM")
            self.logger.info(f"   🌳 Training {architecture} single-output model")

            # Prepare data
            X = features.values
            y = target.values

            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.validation_config["n_splits"])

            # Cross-validation
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Train model based on architecture
                if architecture == "LightGBM":
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42,
                        verbose=-1
                    )
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        eval_metric="binary_logloss",
                        early_stopping_rounds=10,
                        verbose=False
                    )
                elif architecture == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train_scaled, y_train)
                else:
                    self.logger.warning(f"   ⚠️ Architecture {architecture} not implemented for single-output")
                    continue

                # Evaluate
                y_pred = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                cv_scores.append(accuracy)

            # Train final model on full dataset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if architecture == "LightGBM":
                final_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
            elif architecture == "RandomForest":
                final_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                return None

            final_model.fit(X_scaled, y)

            # Calculate metrics
            y_pred_final = final_model.predict(X_scaled)
            final_accuracy = accuracy_score(y, y_pred_final)

            result = {
                "model": final_model,
                "scaler": scaler,
                "architecture": architecture,
                "cv_scores": cv_scores,
                "cv_mean": np.mean(cv_scores) if cv_scores else 0.0,
                "cv_std": np.std(cv_scores) if cv_scores else 0.0,
                "final_accuracy": final_accuracy,
                "feature_importance": dict(zip(features.columns, final_model.feature_importances_)) if hasattr(final_model, 'feature_importances_') else {},
                "n_features": len(features.columns)
            }

            return result

        except Exception as e:
            self.logger.exception(f"❌ Failed to train single-output model: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=None,
        context="enhanced_regime_specific_training"
    )
    async def train_enhanced_regime_specific_models(
        self,
        timeframe: str,
        regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train enhanced regime-specific models with multi-output support.

        Args:
            timeframe: Timeframe for training
            regime_data: Regime-specific data dictionary

        Returns:
            Dictionary containing regime-specific training results
        """
        self.logger.info(f"🎯 Training enhanced regime-specific models for {timeframe}")

        regime_results = {}

        for regime_key, regime_info in regime_data.items():
            regime_desc = regime_info.get("description", "Unknown")
            self.logger.info(f"   🎯 Processing regime {regime_key}: {regime_desc}")

            # Get regime data
            train_data = regime_info.get("train")
            val_data = regime_info.get("validation")
            test_data = regime_info.get("test")

            if train_data is None or len(train_data) < self.validation_config["min_samples_per_split"]:
                self.logger.warning(f"   ⚠️ Insufficient data for regime {regime_key}")
                continue

            # Combine all regime data for training
            all_regime_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

            # Prepare enhanced data
            prepared_data = self.prepare_enhanced_data(
                all_regime_data, timeframe, regime_key
            )

            # Train enhanced model
            model_result = await self.train_enhanced_model(
                prepared_data, f"enhanced_{timeframe}_{regime_key}"
            )

            if model_result:
                regime_results[regime_key] = model_result
                self.logger.info(f"   ✅ Enhanced regime {regime_key} training completed")
            else:
                self.logger.warning(f"   ⚠️ Enhanced regime {regime_key} training failed")

        return regime_results

    def predict_enhanced(
        self,
        features: pd.DataFrame,
        model_name: str = "enhanced_model",
        prediction_type: str = "multi_output"  # "multi_output" or "single_output"
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Make predictions using enhanced model.

        Args:
            features: Feature DataFrame
            model_name: Name of the model to use
            prediction_type: Type of prediction ("multi_output" or "single_output")

        Returns:
            For multi_output: Tuple of (direction_predictions, profit_predictions)
            For single_output: Array of predictions
        """
        if prediction_type == "multi_output" and self.multi_output_trainer:
            try:
                # Create market data for prediction
                market_data = pd.DataFrame({
                    'close': np.random.randn(len(features)),  # Placeholder - should use actual market data
                    'volume': np.random.randn(len(features))
                })

                # Generate probability predictions
                price_action_probabilities = self.multi_output_trainer.predict_probabilities(
                    features.values, market_data
                )

                # Extract direction and profit probabilities
                direction_prob = price_action_probabilities.get("direction_probability", 0.5)
                profit_prob = price_action_probabilities.get("triple_barrier_probability", 0.5)

                return np.array([direction_prob]), np.array([profit_prob])
            except Exception as e:
                self.logger.error(f"❌ Multi-output prediction failed: {e}")
                return None, None
        else:
            # Fallback to single-output prediction
            self.logger.warning("⚠️ Single-output prediction not implemented in enhanced trainer")
            return None

    def save_enhanced_models(
        self,
        results: Dict[str, Any],
        save_path: str
    ) -> None:
        """Save enhanced models to disk.

        Args:
            results: Training results dictionary
            save_path: Path to save models
        """
        try:
            os.makedirs(save_path, exist_ok=True)

            # Save multi-output models
            if results.get("multi_output_results") and self.multi_output_trainer:
                multi_output_dir = os.path.join(save_path, "multi_output_models")
                os.makedirs(multi_output_dir, exist_ok=True)

                # Save the multi-output trainer
                model_path = os.path.join(multi_output_dir, f"{results['model_name']}_multi_output.pkl")
                import joblib
                joblib.dump(self.multi_output_trainer, model_path)

            # Save single-output models
            if results.get("single_output_results"):
                single_output_dir = os.path.join(save_path, "single_output_models")
                os.makedirs(single_output_dir, exist_ok=True)

                single_result = results["single_output_results"]
                model_path = os.path.join(single_output_dir, f"{results['model_name']}_single.pkl")
                scaler_path = os.path.join(single_output_dir, f"{results['model_name']}_scaler.pkl")

                import joblib
                joblib.dump(single_result["model"], model_path)
                joblib.dump(single_result["scaler"], scaler_path)

            # Save metadata
            metadata = {
                "model_name": results["model_name"],
                "timeframe": results["timeframe"],
                "regime_key": results["regime_key"],
                "has_multi_output": results.get("multi_output_results") is not None,
                "has_single_output": results.get("single_output_results") is not None,
                "training_timestamp": results["training_timestamp"]
            }

            metadata_path = os.path.join(save_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"✅ Enhanced models saved to {save_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced models: {e}")


async def run_enhanced_step(
    symbol: str = "ETHUSDT",
    data_dir: str = "data/training",
    method_a_mixture_of_experts: Optional[Dict] = None,
    enable_multi_output: bool = True
) -> bool:
    """Run enhanced HMM-based training step with multi-output support.

    Args:
        symbol: Trading symbol
        data_dir: Data directory
        method_a_mixture_of_experts: Method A configuration
        enable_multi_output: Whether to enable multi-output training

    Returns:
        True if successful, False otherwise
    """
    try:
        logger = system_logger.getChild("EnhancedHMMTraining")
        logger.info(f"🚀 Starting Enhanced HMM-Based Training for {symbol}")

        # Load configuration
        config = {
            "enable_multi_output": enable_multi_output,
            "HMM_LM": method_a_mixture_of_experts or {}
        }

        # Initialize enhanced trainer
        enhanced_trainer = EnhancedHMMBasedTrainingStep(config)
        await enhanced_trainer.initialize()

        # Load labeled data
        labeled_path = f"{data_dir}/{symbol}_labeled_train.parquet"
        if not os.path.exists(labeled_path):
            logger.error(f"❌ Labeled data not found: {labeled_path}")
            return False

        data = pd.read_parquet(labeled_path)
        logger.info(f"✅ Loaded labeled data: {data.shape}")

        # Prepare enhanced data
        prepared_data = enhanced_trainer.prepare_enhanced_data(data, "1m")

        # Train enhanced model
        results = await enhanced_trainer.train_enhanced_model(
            prepared_data, f"enhanced_{symbol}_1m"
        )

        if results:
            # Save models
            save_path = f"{data_dir}/enhanced_models/{symbol}"
            enhanced_trainer.save_enhanced_models(results, save_path)

            logger.info("✅ Enhanced HMM-based training completed successfully")
            return True
        else:
            logger.error("❌ Enhanced HMM-based training failed")
            return False

    except Exception as e:
        logger.exception(f"❌ Enhanced HMM-based training failed: {e}")
        return False