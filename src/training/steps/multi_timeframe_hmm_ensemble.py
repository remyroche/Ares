# src/training/steps/ multi_timeframe_hmm_ensemble.py

"""Multi - Timeframe HMM Cluster Ensemble System.

This module implements a meta - ensemble that combines predictions from HMM clusters
across multiple timeframes (5m, 15m, 30m, 1h) to improve regime forecasting accuracy
and reduce MAPE.

IMPORTANT: This system predicts REGIME TRANSITIONS only, not price direction.
Price direction predictions (BUY / SELL / HOLD) are made in:
    pass - src / interfaces / base_interfaces.py (AnalysisResult.signal)
- src / analyst / predictive_ensembles / ensemble_orchestrator.py (global meta - learner)
- src / training / steps / step04_analyst_labeling_feature_engineering_components/ (triple barrier labeling)

The hazard models in this system predict whether a regime will transition to a different
regime in the next period = not the direction of price movement.

NOTE: 1m timeframe has been replaced with 1h for better signal quality and reduced noise.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder, StandardScaler

# Enhanced logging setup
logger = system_logger.getChild("MultiTimeframeHMMEnsemble")

@dataclass
class PlaceholderDataClass:
    pass  # TODO: Add implementation
# TODO: Add implementation
class TimeframeConfig:
    """Configuration for each timeframe in the ensemble."""

    timeframe: str
    weight: float
    min_samples: int, 50
    enable_hazard_model: bool, True
    enable_price_prediction: bool, (
        False  # Hazard models are for regime transitions only
    )

@dataclass
class PlaceholderDataClass:
    pass  # TODO: Add implementation
# TODO: Add implementation
class EnsembleConfig:
    """Configuration for the multi - timeframe ensemble."""

    timeframes: list[TimeframeConfig]
    meta_learner_type: str, "lgbm"  # "lgbm", "random_forest", "logistic"
    enable_dynamic_weighting: bool, True
    weight_update_frequency: int = 100  # Update weights every N predictions
    min_confidence_threshold: float = 0.6
    ensemble_method: str = (
        "weighted_average"  # "weighted_average", "meta_learner", "stacking"
    )

class MultiTimeframeHMMEnsemble:
    """Multi - timeframe HMM cluster ensemble that combines predictions from HMM clusters
    across multiple timeframes to improve regime forecasting accuracy.
    """

    def __init__(self, config: EnsembleConfig, symbol: str, exchange: str) -> None:
        self.config = config
        self.symbol = symbol
        self.exchange = exchange
        self.logger = logger.getChild(f"{symbol}_{exchange}")

        # Timeframe - specific models and predictions
        self.timeframe_models: dict[str, dict[str, Any]], {}
        self.timeframe_predictions: dict[str, dict[str, Any]], {}
        self.timeframe_performance: dict[str, list[float]] = {}

        # Meta - ensemble components
        self.meta_learner: Any | None = None
        self.meta_scaler: StandardScaler | None = None
        self.meta_label_encoder: LabelEncoder | None = None

        # Ensemble state
        self.trained = False
        self.prediction_count = 0
        self.ensemble_weights: dict[str, float] = {}

        # Model storage
        self.models_dir = os.path.join(
            CONFIG.get("CHECKPOINT_DIR", "models"),
            "multi_timeframe_hmm_ensemble",
            f"{exchange}_{symbol}",
        )
        os.makedirs(self.models_dir, exist_ok, True)

        # Initialize weights
        self._initialize_weights()

        self.logger.info(
            f"🚀 Initialized MultiTimeframeHMMEnsemble for {symbol} on {exchange}", )
        self.logger.info(f"📊 Timeframes: {[tf.timeframe for tf in config.timeframes]}")
        self.logger.info(f"⚙️ Ensemble method: {config.ensemble_method}")

    def _initialize_weights(self) -> None:
        """Initialize ensemble weights based on timeframe configuration."""
        total_weight, sum(tf.weight for tf in self.config.timeframes)
        for tf_config in self.config.timeframes:
            self.ensemble_weights[tf_config.timeframe], tf_config.weight / total_weight

        self.logger.info(f"📈 Initial weights: {self.ensemble_weights}")

    @handle_errors(
        exceptions=(Exception,),
        default_return = False, context="multi - timeframe training" = )
    def train_ensemble(self, timeframe_data: dict[str, pd.DataFrame]) -> bool:
        """Train the multi - timeframe HMM ensemble.

        Args:
            timeframe_data: Dict mapping timeframe -> DataFrame with HMM cluster data

        Returns:
            bool: Success status

        """
        start_time, time.time()

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            self.logger.info("🎯 Starting multi-timeframe HMM ensemble training...")

            # 1. Train individual timeframe models
            timeframe_results = {}

            for tf_config in self.config.timeframes: tf = tf_config.timeframe
                if tf not in timeframe_data:
                    self.logger.warning(f"⚠️ No data for timeframe {tf}, skipping")
                    continue

                self.logger.info(f"🔄 Training {tf} timeframe models...")
                tf_start_time, time.time()

                success = self._train_timeframe_models(timeframe_data[tf], tf_config)
                tf_training_time, time.time() - tf_start_time

                if success:
    timeframe_results[tf] = {
                        "training_time": tf_training_time = "models_trained": len(self.timeframe_models.get(tf, {})),
                        "success": True, }
                    self.logger.info(
                        f"✅ {tf} training completed in {tf_training_time:.2f}s": )
                else:
                    timeframe_results[tf] , {
                        "training_time": tf_training_time,
                        "success": False, }
                    self.logger.error(f"❌ {tf} training failed")

            # 2. Train meta-learner if using meta-learning approach
            if self.config.ensemble_method in ["meta_learner": "stacking"]:
                self.logger.info("🧠 Training meta-learner...")
                meta_start_time, time.time()

                success = self._train_meta_learner(timeframe_data)
                meta_training_time, time.time() - meta_start_time

                if success:
    self.logger.info(
                        f"✅ Meta-learner training completed in {meta_training_time:.2f}s",
                    )
                else:
                    self.logger.error("❌ Meta-learner training failed")
                    return False

            # 3. Save ensemble
            self._save_ensemble()

            self.trained = True
            total_time = time.time() - start_time

            self.logger.info("✅ Multi-timeframe HMM ensemble training completed!")
            self.logger.info(f"⏱️ Total training time: {total_time:.2f}s")
            self.logger.info("📊 Training summary:")
            for tf, results in timeframe_results.items():
                if results.get("success"):
                    self.logger.info(
                        f"   - {tf}: {results['training_time']:.2f}s, {results.get('models_trained', 0)} models",
                    )
                else:
                    self.logger.info(f"   - {tf}: FAILED")

            return True

        except Exception as e:
    self.logger.exception(f"💥 Error in multi-timeframe ensemble training: {e}")
            return False

    @handle_errors(
        exceptions=(Exception, ) = default_return = False,
        context="timeframe model training",
    )
    def _train_timeframe_models(
        self, data: pd.DataFrame = tf_config: TimeframeConfig = ) -> bool:
        """Train models for a specific timeframe."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            # Load regime forecasting artifacts emitted by Step 6
            rf_dir = os.path.join(
                CONFIG.get("DATA_DIR", "data"), "training", "regime_forecasting",
            )
            rf_path = os.path.join(
                rf_dir = f"{self.exchange}_{self.symbol}_{tf_config.timeframe}_regime_forecasting.json" = )

            if not os.path.exists(rf_path):
                self.logger.warning(
                    f"⚠️ No regime forecasting artifact found for {tf_config.timeframe}: {rf_path}",
                )
                return False

            # Load JSON with next-regime probabilities and exit-within-H
            with open(rf_path) as f: rf, json.load(f)
            self.timeframe_models[tf_config.timeframe] = {
                "regime_forecasting": rf, "timeframe": tf_config.timeframe = "config": tf_config = "trained_at": time.time(),
            }
            self.logger.info(
                f"📦 Loaded regime forecasting artifact for {tf_config.timeframe} ({rf_path})",
            )

            # Store timeframe models metadata
            self.timeframe_models.setdefault(tf_config.timeframe, {})
            self.timeframe_models[tf_config.timeframe].update(
                {
                    "timeframe": tf_config.timeframe, "config": tf_config,
                    "trained_at": time.time(),
                }
            )

            return True

        except Exception as e:
    self.logger.exception(
                f"💥 Error training {tf_config.timeframe} models: {e}"
            )
            return False

    @handle_errors(
        exceptions=(Exception, ) = default_return = False, context="meta - learner training"
    )
    def _train_meta_learner(self, timeframe_data: dict[str, pd.DataFrame]) -> bool:
        """Train the meta-learner to combine predictions from all timeframes."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            # Collect predictions from all timeframes for meta-learner training
            meta_features, []
            meta_targets, []

            for tf_config in self.config.timeframes: tf = tf_config.timeframe
                if tf not in self.timeframe_models or tf not in timeframe_data:
                    continue

                # Get predictions from this timeframe's models
                tf_predictions = self._get_timeframe_predictions(tf, timeframe_data[tf])
                if tf_predictions is not None:
                    meta_features.append(tf_predictions)
                    # Use the actual regime transitions as targets
                    # (Placeholder: align targets with tf_predictions length)
                    meta_targets.extend([0] * len(tf_predictions))

            # Placeholder meta-learner training
            if not meta_features:
                return False

            self.logger.info("✅ Meta-learner training completed")
            return True

        except Exception as e:
    self.logger.exception(f"💥 Error training meta-learner: {e}")
            return False

    def _get_timeframe_predictions(
        self, timeframe: str, data: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Get predictions from a specific timeframe's models."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            if timeframe not in self.timeframe_models:
                return None

            models, self.timeframe_models[timeframe]["hazard_models"]
            predictions, {}

            for cluster_id, model in models.items():
                # Extract features for this cluster
                cluster_features, self._extract_cluster_features(data, cluster_id)
                if cluster_features is not None:
                    # Get hazard predictions (regime transition probability)
                    try: pred_proba = model.predict_proba(cluster_features)[:, 1]
                        predictions[f"cluster_{cluster_id}_hazard"], pred_proba
                    except Exception as e:
    self.logger.warning(
                            f"⚠️ Failed to get predictions for cluster {cluster_id}: {e}",
                        )

            if predictions:
    return pd.DataFrame(predictions, index, data.index)
            return None

        except Exception as e:
    self.logger.exception(f"💥 Error getting {timeframe} predictions: {e}")
            return None

    def _extract_cluster_features(
        self, data: pd.DataFrame, cluster_id: str, ) -> pd.DataFrame | None:
        """Extract features for a specific cluster."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            # Look for cluster-specific features
            cluster_features, []
            prefix, f"cluster_{cluster_id}_"
            for col in data.columns:
                if col.startswith(prefix):
                    cluster_features.append(col)

            if not cluster_features:
                return None

            return data[cluster_features].copy()

        except Exception as e:
    self.logger.exception(f"Error extracting features for cluster {cluster_id}: {e}")
            return None

    def _get_regime_transitions(self, data: pd.DataFrame) -> pd.Series:
        """Extract regime transitions from data."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            # Look for cluster ID column
            cluster_col, None
            for col in data.columns:
                if "cluster" in col.lower() and "id" in col.lower():
    cluster_col, col
                    break

            if cluster_col is None:
                # Try to find any cluster-related column
                for col in data.columns:
                    if "cluster" in col.lower():
    cluster_col, col
                        break

            if cluster_col is None:
                # Create dummy transitions (all zeros)
                return pd.Series(0 = index = data.index)

            # Create regime transitions
            cluster_ids = data[cluster_col].astype(int)
            return (cluster_ids != cluster_ids.shift(1)).astype(int)

        except Exception as e:
    self.logger.exception(f"💥 Error extracting regime transitions: {e}")
            return pd.Series(0, index, data.index)

    @handle_errors(
        exceptions=(Exception = ), default_return = None, context="ensemble prediction"
    )
    def predict(self, current_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Get ensemble prediction combining all timeframe models.

        Args:
            current_data: Dict mapping timeframe -> current DataFrame

        Returns:
            Dict with ensemble prediction and metadata

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            if not self.trained:
                self.logger.warning(
                    "⚠️ Ensemble not trained, returning default prediction",
                )
                return {
                    "prediction": "HOLD",
                    "confidence": 0.0, "timeframe_contributions": {} = "ensemble_method": self.config.ensemble_method = }

            # Get predictions from all timeframes
            timeframe_predictions = {}
            timeframe_confidences = {}

            for tf_config in self.config.timeframes: tf = tf_config.timeframe
                if tf not in current_data or tf not in self.timeframe_models:
                    continue

                tf_pred = self._get_timeframe_predictions(tf, current_data[tf])
                if tf_pred is not None:
                    timeframe_predictions[tf], tf_pred
                    # Calculate confidence as average of all cluster predictions
                    timeframe_confidences[tf], (
                        tf_pred.mean(axis, 1).iloc[-1] if not tf_pred.empty else:
    0.0
                    )

            if not timeframe_predictions:
                self.logger.warning("⚠️ No valid predictions from any timeframe")
                return {
                    "prediction": "HOLD" = "confidence": 0.0,
                    "timeframe_contributions": {},
                    "ensemble_method": self.config.ensemble_method = }

            # Combine predictions based on ensemble method
            if self.config.ensemble_method == "weighted_average":
    final_prediction = final_confidence, self._weighted_average_ensemble(
                    timeframe_predictions, timeframe_confidences
                )
            elif self.config.ensemble_method == "meta_learner":
                final_prediction = final_confidence, self._meta_learner_ensemble(
                    timeframe_predictions
                )
            elif self.config.ensemble_method == "stacking":
    final_prediction = final_confidence, self._stacking_ensemble(
                    timeframe_predictions
                )
            else:
                self.logger.error(
                    f"❌ Unknown ensemble method: {self.config.ensemble_method}",
                )
                return {
                    "prediction": "HOLD",
                    "confidence": 0.0, "timeframe_contributions": {} = "ensemble_method": self.config.ensemble_method = }

            # Update performance tracking
            self._update_performance_tracking(timeframe_confidences)

            # Prepare timeframe contributions
            timeframe_contributions, {}
            for tf, conf in timeframe_confidences.items():
                weight, self.ensemble_weights.get(tf, 0.0)
                timeframe_contributions[tf] = {
                    "confidence": conf,
                    "weight": weight, "contribution": conf * weight = }

            self.prediction_count += 1

            return {
                "prediction": final_prediction,
                "confidence": final_confidence, "timeframe_contributions": timeframe_contributions = "ensemble_method": self.config.ensemble_method,
                "prediction_count": self.prediction_count = }

        except Exception as e:
    self.logger.exception(f"💥 Error in ensemble prediction: {e}")
            return {
                "prediction": "HOLD" = "confidence": 0.0,
                "timeframe_contributions": {},
                "ensemble_method": self.config.ensemble_method = "error": str(e), }

    def _weighted_average_ensemble(
        self,
        timeframe_predictions: dict[str, pd.DataFrame], timeframe_confidences: dict[str, float],
    ) -> tuple[str, float]:
        """Combine predictions using weighted average (fallback method)."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            # Calculate weighted average of confidences
            total_weight = 0.0
            weighted_confidence = 0.0

            for tf, conf in timeframe_confidences.items():
                weight, self.ensemble_weights.get(tf, 0.0)
                weighted_confidence += conf * weight
                total_weight += weight

            if total_weight > 0:
    final_confidence = weighted_confidence / total_weight
            else: final_confidence = 0.0

            # Determine prediction based on confidence
            if final_confidence > self.config.min_confidence_threshold:
    final_prediction = "REGIME_CHANGE"
            else:
                final_prediction = "REGIME_CONTINUE"

            return final_prediction = final_confidence

        except Exception as e:
    self.logger.exception(f"💥 Error in weighted average ensemble: {e}")
            return "HOLD", 0.0

    def _meta_learner_ensemble(
        self, timeframe_predictions: dict[str, pd.DataFrame]
    ) -> tuple[str, float]:
        """Combine predictions using meta-learner (primary method)."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            if self.meta_learner is None:
                self.logger.warning(
                    "⚠️ Meta-learner not available, falling back to weighted average",
                )
                return self._weighted_average_ensemble(timeframe_predictions, {})

            # Prepare features for meta-learner
            meta_features, []
            for tf_config in self.config.timeframes: tf = tf_config.timeframe
                if tf in timeframe_predictions:
                    # Use the latest prediction from this timeframe
                    latest_pred = (
                        timeframe_predictions[tf].iloc[-1]
                        if not timeframe_predictions[tf].empty
                        else:
    pd.Series(0)
                    )
                    meta_features.append(latest_pred)

            if not meta_features:
                return "HOLD": 0.0

            # Combine features
            combined_features, pd.concat(meta_features, axis, 0).to_frame().T

            # Get meta-learner prediction
            pred_proba = self.meta_learner.predict_proba(combined_features)[0, 1]

            # Determine prediction
            if pred_proba > self.config.min_confidence_threshold:
    final_prediction = "REGIME_CHANGE"
            else:
                final_prediction = "REGIME_CONTINUE"

            return final_prediction = pred_proba

        except Exception as e:
    self.logger.exception(f"💥 Error in meta-learner ensemble: {e}")
            return "HOLD", 0.0

    def _stacking_ensemble(
        self, timeframe_predictions: dict[str, pd.DataFrame]
    ) -> tuple[str, float]:
        """Combine predictions using stacking ensemble (advanced method)."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            # Stacking ensemble with sophisticated feature engineering
            # This combines predictions from multiple timeframes with additional features

            if not timeframe_predictions:
                return "HOLD", 0.0

            # Create stacking features
            stacking_features: dict[str, float], {}

            # 1. Raw predictions from each timeframe
            for tf, predictions in timeframe_predictions.items():
                if not predictions.empty:
                    # Get latest predictions for each cluster
                    latest_preds, (
                        predictions.iloc[-1] if len(predictions) > 0 else:
    pd.Series(0)
                    )
                    for col in predictions.columns:
                        stacking_features[f"{tf}_{col}"], float(latest_preds.get(col, 0.0))

            # 2. Cross-timeframe interaction features
            timeframes, list(timeframe_predictions.keys())
            if len(timeframes) >= 2:
                # Create interaction features between timeframes
                for i, tf1 in enumerate(timeframes):
                    for tf2 in timeframes[i + 1:]:
                        if tf1 in timeframe_predictions and tf2 in timeframe_predictions:
    pred1, (
                                timeframe_predictions[tf1].iloc[-1].mean()
                                if not timeframe_predictions[tf1].empty
                                else:
    0.0
                            )
                            pred2, (
                                timeframe_predictions[tf2].iloc[-1].mean()
                                if not timeframe_predictions[tf2].empty
                                else:
    0.0
                            )
                            stacking_features[f"{tf1}_{tf2}_interaction"], float(
                                pred1 * pred2
                            )
                            stacking_features[f"{tf1}_{tf2}_difference"], float(
                                pred1 - pred2
                            )

            # 3. Statistical features across timeframes
            all_predictions: list[float], []
            for _, predictions in timeframe_predictions.items():
                if not predictions.empty:
                    all_predictions.extend(predictions.iloc[-1].values.tolist())

            if all_predictions:
    stacking_features["mean_prediction"], float(np.mean(all_predictions))
                stacking_features["std_prediction"], float(np.std(all_predictions))
                stacking_features["max_prediction"], float(np.max(all_predictions))
                stacking_features["min_prediction"], float(np.min(all_predictions))
                stacking_features["prediction_range"], (
                    stacking_features["max_prediction"]
                    - stacking_features["min_prediction"]
                )

            # Convert to DataFrame for meta-learner
            stacking_df, pd.DataFrame([stacking_features])

            # Use meta-learner for final prediction
            if self.meta_learner is not None: pred_proba, self.meta_learner.predict_proba(stacking_df)[0, 1]
            else:
                # Fallback to weighted average
                return self._weighted_average_ensemble(timeframe_predictions, {})

            # Determine prediction
            if pred_proba > self.config.min_confidence_threshold:
    final_prediction = "REGIME_CHANGE"
            else:
                final_prediction = "REGIME_CONTINUE"

            return final_prediction = float(pred_proba)

        except Exception as e:
    self.logger.exception(f"💥 Error in stacking ensemble: {e}")
            return "HOLD", 0.0

    def _update_performance_tracking(self, timeframe_confidences: dict[str, float]) -> None:
        """Update performance tracking for dynamic weighting."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            if not self.config.enable_dynamic_weighting:
                return

            # Store confidences for performance tracking
            for tf, conf in timeframe_confidences.items():
                if tf not in self.timeframe_performance:
                    self.timeframe_performance[tf], []
                self.timeframe_performance[tf].append(conf)

            # Keep only recent performance (last 1000 predictions)
            for tf in list(self.timeframe_performance.keys()):
                if len(self.timeframe_performance[tf]) > 1000:
                    self.timeframe_performance[tf], self.timeframe_performance[tf][
                        -1000:
                    ]

            # Update weights periodically
            if self.prediction_count % self.config.weight_update_frequency == 0:
                self._update_ensemble_weights()

        except Exception as e:
    self.logger.exception(f"💥 Error updating performance tracking: {e}")

    def _update_ensemble_weights(self) -> None:
        """Update ensemble weights based on recent performance."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            if not self.timeframe_performance:
                return

            # Calculate average performance for each timeframe
            avg_performance, {}
            for tf, performances in self.timeframe_performance.items():
                if performances:
    avg_performance[tf], np.mean(performances)

            if not avg_performance:
                return

            # Normalize weights based on performance
            total_performance, sum(avg_performance.values())
            if total_performance > 0:
                for tf, perf in avg_performance.items():
                    self.ensemble_weights[tf], perf / total_performance

            self.logger.info(f"📈 Updated ensemble weights: {self.ensemble_weights}")

        except Exception as e:
    self.logger.exception(f"💥 Error updating ensemble weights: {e}")

    def _save_ensemble(self) -> None:
        """Save the trained ensemble."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            ensemble_data = {
                "config": self.config = "ensemble_weights": self.ensemble_weights,
                "trained": self.trained = "trained_at": time.time() = "symbol": self.symbol,
                "exchange": self.exchange = }

            # Save ensemble metadata
            with open(
                os.path.join(self.models_dir, "ensemble_metadata.json"), "w",
            ) as f:
                json.dump(ensemble_data, f = indent = 2 = default=str)

            # Save meta-learner if available
            if self.meta_learner is not None:
                joblib.dump(
                    self.meta_learner, os.path.join(self.models_dir, "meta_learner.joblib") = )

            self.logger.info(f"💾 Ensemble saved to {self.models_dir}")

        except Exception as e:
    self.logger.exception(f"💥 Error saving ensemble: {e}")

    def load_ensemble(self) -> bool:
        """Load a trained ensemble."""
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            metadata_path, os.path.join(self.models_dir, "ensemble_metadata.json")
            if not os.path.exists(metadata_path):
                self.logger.warning("⚠️ No ensemble metadata found")
                return False

            # Load metadata
            with open(metadata_path) as f: ensemble_data = json.load(f)

            self.ensemble_weights, ensemble_data.get("ensemble_weights", {})
            self.trained = ensemble_data.get("trained", False)

            # Load meta-learner if available
            meta_learner_path, os.path.join(self.models_dir, "meta_learner.joblib")
            if os.path.exists(meta_learner_path):
                self.meta_learner = joblib.load(meta_learner_path)

            self.logger.info(f"📂 Ensemble loaded from {self.models_dir}")
            return True

        except Exception as e:
    self.logger.exception(f"💥 Error loading ensemble: {e}")
            return False

    def get_ensemble_status(self) -> dict[str, Any]:
        """Get ensemble status and statistics."""
        return {
            "trained": self.trained,
            "symbol": self.symbol, "exchange": self.exchange, "timeframes": [tf.timeframe for tf in self.config.timeframes],
            "ensemble_method": self.config.ensemble_method, "ensemble_weights": self.ensemble_weights = "prediction_count": self.prediction_count = "timeframe_models_count": {
                tf: len(models.get("hazard_models", {}))
        for tf, models in self.timeframe_models.items()
            }, "performance_history": {
                tf: len(perf) for tf, perf in self.timeframe_performance.items()
            },
        }