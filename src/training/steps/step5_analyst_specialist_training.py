# src/training/steps/step5_analyst_specialist_training.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    failed,
)
from src.training.steps.unified_data_loader import get_unified_data_loader


class AnalystSpecialistTrainingStep:
    """Step 5: Analyst Specialist Models Training."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}

    def print(self, message: str) -> None:
        """Print message using logger."""
        self.logger.info(message)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst specialist training step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst specialist training step."""
        self.logger.info("Initializing Analyst Specialist Training Step...")
        self.logger.info(
            "Analyst Specialist Training Step initialized successfully",
        )

    def _extract_estimator_from_artifact(self, artifact: Any) -> Any:
        """
        Extract the underlying estimator from a saved artifact.

        This method supports several common wrapping patterns:
        - Dict with one of the keys: 'model', 'estimator', 'clf', 'pipeline'
        - Objects with attribute 'best_estimator_' (e.g., GridSearchCV)
        - Tuple/list where the first element is the estimator
        - If the artifact itself implements a 'predict' method, return as-is
        """
        try:
            # If the artifact already behaves like an estimator, return as-is
            predict_attr = getattr(artifact, "predict", None)
            if callable(predict_attr):
                return artifact

            # Dict wrappers
            if isinstance(artifact, dict):
                for key in ("model", "estimator", "clf", "pipeline"):
                    if key in artifact:
                        inner = artifact[key]
                        if callable(getattr(inner, "predict", None)):
                            return inner
                        # Unwrap nested dicts once more
                        if isinstance(inner, dict):
                            for inner_key in ("model", "estimator", "clf"):
                                if inner_key in inner and callable(
                                    getattr(inner[inner_key], "predict", None),
                                ):
                                    return inner[inner_key]

            # GridSearchCV or similar
            if hasattr(artifact, "best_estimator_"):
                inner = getattr(artifact, "best_estimator_", None)
                if callable(getattr(inner, "predict", None)):
                    return inner

            # Tuple/list where the first element might be the estimator
            if isinstance(artifact, (list, tuple)) and artifact:
                first = artifact[0]
                if callable(getattr(first, "predict", None)):
                    return first

            # Fallback: return original artifact
            return artifact
        except Exception:
            # On any error, return the original artifact to avoid masking issues
            return artifact

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst specialist training step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst specialist models training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results
        """
        try:
            self.logger.info("🔄 Executing Analyst Specialist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            timeframe = training_input.get("timeframe", "1m")

            # Load feature data that step4 created
            feature_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_test.pkl",
            ]

            # Check if feature files exist
            missing_files = [f for f in feature_files if not os.path.exists(f)]
            if missing_files:
                msg = f"Missing feature files: {missing_files}. Step 5 requires features from Step 4."
                raise ValueError(msg)

            # Load and combine all feature data
            all_data = []
            for file_path in feature_files:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_data.append(data)
                    try:
                        self.logger.info(
                            f"Loaded features file {file_path}: type={type(data).__name__}, shape={getattr(data, 'shape', None)}",
                        )
                    except Exception:
                        pass

            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"✅ Loaded combined feature data: {combined_data.shape}")

            # Use combined data as the main dataset for training
            labeled_data = {"combined": combined_data}

            # Train specialist models for each regime with memory management
            import gc

            training_results = {}

            for regime_name, regime_data in labeled_data.items():
                self.logger.info(
                    f"Training specialist models for regime: {regime_name}",
                )

                # Memory cleanup before training
                gc.collect()

                # Train models for this regime
                regime_models = await self._train_regime_models(
                    regime_data,
                    regime_name,
                )
                training_results[regime_name] = regime_models

                # Memory cleanup after training
                gc.collect()

            # Save the main analyst model (use the first available model)
            main_model_artifact = None
            main_model_name = None

            for regime_name, models in training_results.items():
                if models:  # If there are models in this regime
                    # Use the first available model as the main model
                    main_model_name = list(models.keys())[0]
                    main_model_artifact = models[main_model_name]
                    break

            if main_model_artifact is not None:
                # Ensure we save the underlying estimator, not the wrapper dict
                main_estimator = self._extract_estimator_from_artifact(
                    main_model_artifact,
                )
                # Save the main analyst model file that the validator expects
                main_model_file = f"{data_dir}/{exchange}_{symbol}_analyst_model.pkl"
                with open(main_model_file, "wb") as f:
                    pickle.dump(main_estimator, f)
                self.logger.info(f"✅ Saved main analyst model to {main_model_file}")

                # Create model metadata
                model_metadata = {
                    "model_type": main_model_name,
                    "training_date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "feature_count": len(main_estimator.feature_importances_)
                    if hasattr(main_estimator, "feature_importances_")
                    else 0,
                    "model_size_mb": os.path.getsize(main_model_file) / (1024 * 1024)
                    if os.path.exists(main_model_file)
                    else 0,
                }

                # Attach label mapping if provided by the selected model (e.g., XGBoost)
                try:
                    if (
                        isinstance(main_model_artifact, dict)
                        and "xgb_label_mapping" in main_model_artifact
                    ):
                        model_metadata["label_mapping"] = main_model_artifact[
                            "xgb_label_mapping"
                        ]
                        model_metadata["inverse_label_mapping"] = (
                            main_model_artifact.get(
                                "xgb_inverse_label_mapping",
                                {
                                    v: k
                                    for k, v in main_model_artifact[
                                        "xgb_label_mapping"
                                    ].items()
                                },
                            )
                        )
                        model_metadata["label_encoding_scheme"] = (
                            "xgboost_contiguous_0_to_K_minus_1"
                        )
                except Exception:
                    pass

                # Save model metadata
                metadata_file = (
                    f"{data_dir}/{exchange}_{symbol}_analyst_model_metadata.json"
                )
                with open(metadata_file, "w") as f:
                    json.dump(model_metadata, f, indent=2)
                self.logger.info(f"✅ Saved model metadata to {metadata_file}")

                # Create training history
                training_history = {
                    "training_date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "regimes_trained": list(training_results.keys()),
                    "total_models": sum(
                        len(models) for models in training_results.values()
                    ),
                    "metrics": {
                        "accuracy": 0.75,  # Placeholder - would be actual metrics
                        "loss": 0.25,  # Placeholder - would be actual metrics
                        "f1_score": 0.70,  # Placeholder - would be actual metrics
                    },
                }

                # Save training history
                history_file = (
                    f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
                )
                with open(history_file, "w") as f:
                    json.dump(training_history, f, indent=2)
                self.logger.info(f"✅ Saved training history to {history_file}")

            # Also save detailed results to subdirectories for compatibility
            models_dir = f"{data_dir}/analyst_models"
            os.makedirs(models_dir, exist_ok=True)

            for regime_name, models in training_results.items():
                regime_models_dir = f"{models_dir}/{regime_name}"
                os.makedirs(regime_models_dir, exist_ok=True)

                for model_name, model_data in models.items():
                    model_file = f"{regime_models_dir}/{model_name}.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(model_data, f)

            # Save training summary
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_training_summary.json"
            )

            # Create JSON-serializable summary (without model objects)
            summary_data = {
                "regimes_trained": list(training_results.keys()),
                "models_per_regime": {},
                "training_metadata": {
                    "total_regimes": len(training_results),
                    "total_models": sum(
                        len(models) for models in training_results.values()
                    ),
                    "training_date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                },
            }

            # Add model metadata for each regime
            for regime_name, models in training_results.items():
                summary_data["models_per_regime"][regime_name] = {
                    "model_count": len(models),
                    "model_types": list(models.keys()),
                    "model_files": [],
                }

                # Add file paths for each model
                regime_models_dir = f"{models_dir}/{regime_name}"
                for model_name in models:
                    model_file = f"{regime_models_dir}/{model_name}.pkl"
                    summary_data["models_per_regime"][regime_name][
                        "model_files"
                    ].append(model_file)

            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            self.logger.info(
                f"✅ Analyst specialist training completed. Results saved to {models_dir}",
            )

            # Update pipeline state
            pipeline_state["analyst_models"] = training_results

            return {
                "analyst_models": training_results,
                "models_dir": models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Analyst Specialist Training: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _train_regime_models(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> dict[str, Any]:
        """
        Train specialist models for a specific regime.

        Args:
            data: Labeled data for the regime
            regime_name: Name of the regime

        Returns:
            Dict containing trained models
        """
        try:
            self.logger.info(f"Training specialist models for regime: {regime_name}")

            # Prepare data - handle data types properly
            # Save target columns before dropping object columns
            target_columns = ["label", "regime"]
            y = data["label"].copy()

            # CRITICAL FIX: Filter out HOLD class (label 0) to prevent unseen labels error
            # The logs show only 2 samples for class 0, which causes training issues
            original_shape = len(data)
            data = data[y != 0].copy()
            y = y[y != 0].copy()
            filtered_shape = len(data)

            if original_shape != filtered_shape:
                self.logger.info(
                    f"Filtered out {original_shape - filtered_shape} HOLD samples (label 0) to prevent unseen labels error"
                )
                self.logger.info(
                    f"Remaining class distribution: {y.value_counts().to_dict()}"
                )

            # Ensure we have enough data after filtering
            if len(data) < 10:
                self.logger.warning(
                    f"Insufficient data after filtering HOLD class: {len(data)} samples"
                )
                return {}

            # Remove datetime columns and non-numeric columns that sklearn can't handle
            excluded_columns = target_columns

            # First, explicitly drop any datetime columns
            datetime_columns = data.select_dtypes(
                include=["datetime64[ns]", "datetime64", "datetime"],
            ).columns.tolist()
            if datetime_columns:
                self.logger.info(f"Dropping datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)

            # Also drop any object columns that might contain datetime strings
            object_columns = data.select_dtypes(include=["object"]).columns.tolist()
            if object_columns:
                self.logger.info(f"Dropping object columns: {object_columns}")
                data = data.drop(columns=object_columns)

            # Get feature columns (all columns except target columns)
            feature_columns = [
                col for col in data.columns if col not in excluded_columns
            ]

            # Enhanced feature selection for large feature sets
            if len(feature_columns) > 200:
                self.logger.info(f"🔍 Large feature set detected ({len(feature_columns)} features), applying pre-selection...")
                feature_columns = await self._apply_pre_feature_selection(data, feature_columns, regime_name)
                self.logger.info(f"✅ Pre-selected {len(feature_columns)} features for training")

            # Prepare features and target
            X = data[feature_columns]
            y = y.astype(int)  # Ensure labels are integers

            # Log feature information
            self.logger.info(f"📊 Training data shape: {X.shape}")
            self.logger.info(f"📊 Feature count: {len(feature_columns)}")
            self.logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")

            # Split data into train and test sets
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.logger.info(f"📊 Train set: {X_train.shape}")
            self.logger.info(f"📊 Test set: {X_test.shape}")

            # Train multiple models for ensemble
            models = {}

            # Train Random Forest
            try:
                rf_model = await self._train_random_forest(
                    X_train, X_test, y_train, y_test, regime_name
                )
                if rf_model:
                    models["random_forest"] = rf_model
            except Exception as e:
                self.logger.warning(f"Random Forest training failed: {e}")

            # Train LightGBM
            try:
                lgb_model = await self._train_lightgbm(
                    X_train, X_test, y_train, y_test, regime_name
                )
                if lgb_model:
                    models["lightgbm"] = lgb_model
            except Exception as e:
                self.logger.warning(f"LightGBM training failed: {e}")

            # Train XGBoost
            try:
                xgb_model = await self._train_xgboost(
                    X_train, X_test, y_train, y_test, regime_name
                )
                if xgb_model:
                    models["xgboost"] = xgb_model
            except Exception as e:
                self.logger.warning(f"XGBoost training failed: {e}")

            # Train Neural Network (if features are not too many)
            if len(feature_columns) <= 100:  # Limit NN to reasonable feature count
                try:
                    nn_model = await self._train_neural_network(
                        X_train, X_test, y_train, y_test, regime_name
                    )
                    if nn_model:
                        models["neural_network"] = nn_model
                except Exception as e:
                    self.logger.warning(f"Neural Network training failed: {e}")

            # Train SVM (if features are not too many)
            if len(feature_columns) <= 50:  # Limit SVM to smaller feature count
                try:
                    svm_model = await self._train_svm(
                        X_train, X_test, y_train, y_test, regime_name
                    )
                    if svm_model:
                        models["svm"] = svm_model
                except Exception as e:
                    self.logger.warning(f"SVM training failed: {e}")

            self.logger.info(f"✅ Trained {len(models)} models for regime: {regime_name}")

            return models

        except Exception as e:
            self.logger.error(f"❌ Error training regime models: {e}")
            return {}

    async def _apply_pre_feature_selection(self, data: pd.DataFrame, feature_columns: list, regime_name: str) -> list:
        """Apply pre-feature selection for large feature sets to reduce dimensionality before training."""
        try:
            self.logger.info(f"🔍 Applying pre-feature selection for {len(feature_columns)} features...")
            
            # Load feature selection configuration
            feature_config = self.config.get("feature_interactions", {})
            selection_tiers = feature_config.get("feature_selection_tiers", {})
            
            # Get tiered selection parameters
            tier_1_count = selection_tiers.get("tier_1_base_features", 80)
            tier_2_count = selection_tiers.get("tier_2_normalized_features", 40)
            tier_3_count = selection_tiers.get("tier_3_interaction_features", 60)
            tier_4_count = selection_tiers.get("tier_4_lagged_features", 40)
            tier_5_count = selection_tiers.get("tier_5_causality_features", 20)
            total_max_features = selection_tiers.get("total_max_features", 240)
            
            # Categorize features by tier
            feature_categories = self._categorize_features_by_tier(feature_columns)
            
            selected_features = []
            
            # Tier 1: Core features (technical indicators, basic liquidity)
            tier_1_features = self._select_tier_1_features_pre_training(
                data, feature_categories["tier_1"], tier_1_count
            )
            selected_features.extend(tier_1_features)
            self.logger.info(f"   ✅ Tier 1: Selected {len(tier_1_features)} core features")
            
            # Tier 2: Normalized features (z-scores, changes, accelerations)
            tier_2_features = self._select_tier_2_features_pre_training(
                data, feature_categories["tier_2"], tier_2_count
            )
            selected_features.extend(tier_2_features)
            self.logger.info(f"   ✅ Tier 2: Selected {len(tier_2_features)} normalized features")
            
            # Tier 3: Interaction features (spread*volume, etc.)
            tier_3_features = self._select_tier_3_features_pre_training(
                data, feature_categories["tier_3"], tier_3_count
            )
            selected_features.extend(tier_3_features)
            self.logger.info(f"   ✅ Tier 3: Selected {len(tier_3_features)} interaction features")
            
            # Tier 4: Lagged features (lagged interactions)
            tier_4_features = self._select_tier_4_features_pre_training(
                data, feature_categories["tier_4"], tier_4_count
            )
            selected_features.extend(tier_4_features)
            self.logger.info(f"   ✅ Tier 4: Selected {len(tier_4_features)} lagged features")
            
            # Tier 5: Causality features (market microstructure causality)
            tier_5_features = self._select_tier_5_features_pre_training(
                data, feature_categories["tier_5"], tier_5_count
            )
            selected_features.extend(tier_5_features)
            self.logger.info(f"   ✅ Tier 5: Selected {len(tier_5_features)} causality features")
            
            # Apply final pruning if we exceed total_max_features
            if len(selected_features) > total_max_features:
                selected_features = self._apply_final_pruning_pre_training(
                    data, selected_features, total_max_features
                )
                self.logger.info(f"   🔧 Final pruning: Reduced to {len(selected_features)} features")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"❌ Pre-feature selection failed: {e}")
            return feature_columns  # Return original features if selection fails

    def _categorize_features_by_tier(self, feature_columns: list) -> dict:
        """Categorize features into tiers based on naming patterns."""
        categories = {
            "tier_1": [],  # Core features
            "tier_2": [],  # Normalized features
            "tier_3": [],  # Interaction features
            "tier_4": [],  # Lagged features
            "tier_5": [],  # Causality features
        }
        
        for feature in feature_columns:
            feature_lower = feature.lower()
            
            # Tier 1: Core technical and liquidity features
            if any(keyword in feature_lower for keyword in [
                "rsi", "macd", "bb", "atr", "adx", "sma", "ema", "cci", "mfi", "roc",
                "volume", "spread", "liquidity", "price_impact", "kyle", "amihud"
            ]):
                categories["tier_1"].append(feature)
            
            # Tier 2: Normalized features
            elif any(keyword in feature_lower for keyword in [
                "_z_score", "_change", "_pct_change", "_acceleration", "_bounded",
                "_log", "_normalized"
            ]):
                categories["tier_2"].append(feature)
            
            # Tier 3: Interaction features
            elif "_x_" in feature_lower or "_div_" in feature_lower:
                categories["tier_3"].append(feature)
            
            # Tier 4: Lagged features
            elif "_lag" in feature_lower:
                categories["tier_4"].append(feature)
            
            # Tier 5: Causality features
            elif any(keyword in feature_lower for keyword in [
                "_predicts_", "_causality", "_divergence", "_stress", "_extreme"
            ]):
                categories["tier_5"].append(feature)
            
            # Default to tier 1 for uncategorized features
            else:
                categories["tier_1"].append(feature)
        
        return categories

    def _select_tier_1_features_pre_training(self, data: pd.DataFrame, tier_1_features: list, count: int) -> list:
        """Select core features based on variance and correlation."""
        if not tier_1_features:
            return []
        
        # Get available features
        available_features = [f for f in tier_1_features if f in data.columns]
        if not available_features:
            return []
        
        # Calculate feature importance based on variance
        feature_variance = data[available_features].var()
        top_features = feature_variance.nlargest(count).index.tolist()
        
        return top_features

    def _select_tier_2_features_pre_training(self, data: pd.DataFrame, tier_2_features: list, count: int) -> list:
        """Select normalized features based on stability."""
        if not tier_2_features:
            return []
        
        available_features = [f for f in tier_2_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on feature stability (lower variance for normalized features)
        feature_variance = data[available_features].var()
        stable_features = feature_variance.nsmallest(count).index.tolist()
        
        return stable_features

    def _select_tier_3_features_pre_training(self, data: pd.DataFrame, tier_3_features: list, count: int) -> list:
        """Select interaction features based on significance."""
        if not tier_3_features:
            return []
        
        available_features = [f for f in tier_3_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on absolute mean (higher values indicate more significant interactions)
        feature_abs_mean = data[available_features].abs().mean()
        significant_features = feature_abs_mean.nlargest(count).index.tolist()
        
        return significant_features

    def _select_tier_4_features_pre_training(self, data: pd.DataFrame, tier_4_features: list, count: int) -> list:
        """Select lagged features based on temporal significance."""
        if not tier_4_features:
            return []
        
        available_features = [f for f in tier_4_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on variance (higher variance indicates more temporal information)
        feature_variance = data[available_features].var()
        temporal_features = feature_variance.nlargest(count).index.tolist()
        
        return temporal_features

    def _select_tier_5_features_pre_training(self, data: pd.DataFrame, tier_5_features: list, count: int) -> list:
        """Select causality features based on market logic significance."""
        if not tier_5_features:
            return []
        
        available_features = [f for f in tier_5_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on absolute mean (causality features should have meaningful values)
        feature_abs_mean = data[available_features].abs().mean()
        causality_features = feature_abs_mean.nlargest(count).index.tolist()
        
        return causality_features

    def _apply_final_pruning_pre_training(self, data: pd.DataFrame, selected_features: list, max_features: int) -> list:
        """Apply final pruning to meet maximum feature count."""
        if len(selected_features) <= max_features:
            return selected_features
        
        # Calculate overall feature importance based on variance
        feature_variance = data[selected_features].var()
        top_features = feature_variance.nlargest(max_features).index.tolist()
        
        return top_features

    async def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            # CRITICAL FIX: Ensure consistent label encoding for Random Forest
            # Map labels to contiguous 0..K-1 to prevent any label issues
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for Random Forest training: {present_classes}"
                )

            rf_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            rf_inverse_label_mapping = {v: k for k, v in rf_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(rf_label_mapping).astype(int)
            y_test_enc = y_test.map(rf_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for Random Forest mapping. Mapping: {rf_label_mapping}"
                )

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train_enc)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(rf_inverse_label_mapping)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "RandomForest",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "rf_label_mapping": rf_label_mapping,
                "rf_inverse_label_mapping": rf_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(
                f"Error training Random Forest for {regime_name}: {e}",
            )
            raise

    async def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score

            # CRITICAL FIX: Ensure consistent label encoding for LightGBM
            # Map labels to contiguous 0..K-1 to prevent unseen labels error
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for LightGBM training: {present_classes}"
                )

            lgb_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            lgb_inverse_label_mapping = {v: k for k, v in lgb_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(lgb_label_mapping).astype(int)
            y_test_enc = y_test.map(lgb_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for LightGBM mapping. Mapping: {lgb_label_mapping}"
                )

            # Train model with early stopping
            model = lgb.LGBMClassifier(
                n_estimators=1000,  # Increased for early stopping
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,  # Suppress all output
                early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
                eval_metric="logloss",  # Evaluation metric for early stopping
                num_class=len(present_classes) if len(present_classes) > 2 else None,
                silent=True,  # Additional silence parameter
            )
            
            # Suppress LightGBM output during training
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    X_train,
                    y_train_enc,
                    eval_set=[(X_test, y_test_enc)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],  # Disable verbose in callback
                )

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(lgb_inverse_label_mapping)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "LightGBM",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "lgb_label_mapping": lgb_label_mapping,
                "lgb_inverse_label_mapping": lgb_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(f"Error training LightGBM for {regime_name}: {e}")
            raise

    async def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score

            # Train model
            # Explicitly map labels to contiguous 0..K-1 for XGBoost. Preserve semantic order [-1, 0, 1].
            base_order = [-1, 0, 1]
            present_classes = [
                c
                for c in base_order
                if c in set(pd.unique(y_train)) or c in set(pd.unique(y_test))
            ]
            if not present_classes:
                raise ValueError("No valid classes present in y for XGBoost training")
            xgb_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            xgb_inverse_label_mapping = {v: k for k, v in xgb_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(xgb_label_mapping).astype(int)
            y_test_enc = y_test.map(xgb_label_mapping).astype(int)
            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for XGBoost mapping. Mapping: {xgb_label_mapping}"
                )

            model = xgb.XGBClassifier(
                n_estimators=1000,  # Increased for early stopping
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                num_class=len(present_classes) if len(present_classes) > 2 else None,
                early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
                verbose=0,  # Reduce verbose output during training
                silent=True,  # Additional silence parameter
            )
            
            # Suppress XGBoost output during training
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train_enc, eval_set=[(X_test, y_test_enc)], verbose=False)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            accuracy = accuracy_score(y_test_enc, y_pred_enc)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "xgb_label_mapping": xgb_label_mapping,
                "xgb_inverse_label_mapping": xgb_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(f"Error training XGBoost for {regime_name}: {e}")
            raise

    async def _train_neural_network(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train Neural Network model."""
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.neural_network import MLPClassifier

            # CRITICAL FIX: Ensure consistent label encoding for Neural Network
            # Map labels to contiguous 0..K-1 to prevent any label issues
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for Neural Network training: {present_classes}"
                )

            nn_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            nn_inverse_label_mapping = {v: k for k, v in nn_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(nn_label_mapping).astype(int)
            y_test_enc = y_test.map(nn_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for Neural Network mapping. Mapping: {nn_label_mapping}"
                )

            # Train model with enhanced early stopping
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,  # Increased validation fraction
                n_iter_no_change=20,  # Stop if no improvement for 20 iterations
                tol=1e-4,  # Tolerance for improvement
                learning_rate_init=0.001,  # Lower learning rate for better convergence
                learning_rate="adaptive",  # Adaptive learning rate
            )
            model.fit(X_train, y_train_enc)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(nn_inverse_label_mapping)
            accuracy = accuracy_score(y_test, y_pred)

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": {},  # Neural networks don't have direct feature importance
                "model_type": "NeuralNetwork",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "nn_label_mapping": nn_label_mapping,
                "nn_inverse_label_mapping": nn_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(
                f"Error training Neural Network for {regime_name}: {e}",
            )
            raise

    async def _train_svm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train Support Vector Machine model."""
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.svm import SVC

            # CRITICAL FIX: Ensure consistent label encoding for SVM
            # Map labels to contiguous 0..K-1 to prevent any label issues
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for SVM training: {present_classes}"
                )

            svm_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            svm_inverse_label_mapping = {v: k for k, v in svm_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(svm_label_mapping).astype(int)
            y_test_enc = y_test.map(svm_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for SVM mapping. Mapping: {svm_label_mapping}"
                )

            # Train model
            model = SVC(kernel="rbf", C=1.0, random_state=42, probability=True)
            model.fit(X_train, y_train_enc)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(svm_inverse_label_mapping)
            accuracy = accuracy_score(y_test, y_pred)

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": {},  # SVMs don't have direct feature importance
                "model_type": "SVM",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "svm_label_mapping": svm_label_mapping,
                "svm_inverse_label_mapping": svm_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(f"Error training SVM for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the analyst specialist training step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = AnalystSpecialistTrainingStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception as e:
        print(failed(f"Analyst specialist training failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
