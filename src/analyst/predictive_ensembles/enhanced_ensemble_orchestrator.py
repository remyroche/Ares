# src/analyst/predictive_ensembles/enhanced_ensemble_orchestrator.py

"""
Enhanced Ensemble Orchestrator

This integrates multi-timeframe training into the existing ensemble system,
making each individual model (XGBoost, LSTM, etc.) a multi-timeframe ensemble.
"""

import os
import time
from typing import Any

import pandas as pd
import numpy as np

from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.analyst.predictive_ensembles.multi_timeframe_ensemble import (
    MultiTimeframeEnsemble,
)
from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    warning,
)


class EnhancedRegimePredictiveEnsembles(RegimePredictiveEnsembles):
    """
    Enhanced ensemble orchestrator that integrates multi-timeframe training.

    Each individual model (XGBoost, LSTM, etc.) becomes a multi-timeframe ensemble.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.logger = system_logger.getChild("EnhancedRegimePredictiveEnsembles")

        # Multi-timeframe configuration
        self.timeframes = CONFIG.get("TIMEFRAMES", {})
        self.timeframe_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "intraday")
        self.active_timeframes = CONFIG.get("TIMEFRAME_SETS", {}).get(
            self.timeframe_set,
            [],
        )

        # Model types to train
        self.model_types = ["xgboost", "lstm", "random_forest"]

        # Enhanced regime ensembles with multi-timeframe models
        self.enhanced_regime_ensembles: dict[
            str,
            dict[str, MultiTimeframeEnsemble],
        ] = {}

        # Log initialization
        self.logger.info("🚀 Initializing EnhancedRegimePredictiveEnsembles")
        self.logger.info(f"📊 Active timeframes: {self.active_timeframes}")
        self.logger.info(f"🔧 Model types: {self.model_types}")
        self.logger.info(f"⚙️ Timeframe set: {self.timeframe_set}")

    def train_all_models(
        self,
        asset: str,
        prepared_data: dict[str, pd.DataFrame],  # Now accepts multi-timeframe data
        model_path_prefix: str | None = None,
    ):
        """
        Train all enhanced multi-timeframe ensemble models.

        Args:
            asset: Asset symbol
            prepared_data: Dict with timeframe -> DataFrame mapping
            model_path_prefix: Optional path prefix for model storage
        """
        start_time = time.time()

        self.logger.info(
            f"🎯 Starting enhanced multi-timeframe ensemble training for {asset}",
        )
        self.logger.info(f"📊 Available timeframes: {list(prepared_data.keys())}")
        self.logger.info(
            f"📈 Data shapes: {[(tf, df.shape) for tf, df in prepared_data.items()]}",
        )

        # Initialize enhanced regime ensembles
        self._initialize_enhanced_ensembles()

        # Training statistics
        training_stats = {
            "total_ensembles": 0,
            "successful_ensembles": 0,
            "failed_ensembles": 0,
            "regime_stats": {},
        }

        # Train each regime ensemble with multi-timeframe models
        for regime_idx, regime_key in enumerate(self.regime_ensembles.keys(), 1):
            self.logger.info(
                f"🔄 [{regime_idx}/{len(self.regime_ensembles)}] Training enhanced ensemble for regime: {regime_key}",
            )

            regime_start_time = time.time()
            regime_stats = {
                "model_types": 0,
                "successful_models": 0,
                "failed_models": 0,
                "training_time": 0.0,
            }

            # Train each model type for this regime
            for model_idx, model_type in enumerate(self.model_types, 1):
                ensemble_key = f"{regime_key}_{model_type}"
                training_stats["total_ensembles"] += 1
                regime_stats["model_types"] += 1

                self.logger.info(
                    f"🔧 [{model_idx}/{len(self.model_types)}] Training {model_type} for {regime_key}",
                )

                if ensemble_key not in self.enhanced_regime_ensembles[regime_key]:
                    self.enhanced_regime_ensembles[regime_key][ensemble_key] = (
                        MultiTimeframeEnsemble(
                            model_name=model_type,
                            regime=regime_key,
                            config=self.config,
                        )
                    )

                # Train multi-timeframe ensemble
                model_start_time = time.time()
                success = self.enhanced_regime_ensembles[regime_key][
                    ensemble_key
                ].train_multi_timeframe_ensemble(prepared_data, model_type)
                model_training_time = time.time() - model_start_time

                if success:
                    self.logger.info(
                        f"✅ {model_type} for {regime_key} trained successfully in {model_training_time:.2f}s",
                    )
                    training_stats["successful_ensembles"] += 1
                    regime_stats["successful_models"] += 1

                    # Save model
                    if model_path_prefix:
                        model_path = os.path.join(
                            model_path_prefix,
                            f"{asset}_{ensemble_key}",
                        )
                        save_success = self.enhanced_regime_ensembles[regime_key][
                            ensemble_key
                        ].save_model(model_path)
                        if save_success:
                            self.logger.info(f"💾 Saved {ensemble_key} to {model_path}")
                        else:
                            self.print(failed("⚠️ Failed to save {ensemble_key}"))
                else:
                    self.logger.error(
                        f"❌ {model_type} for {regime_key} training failed",
                    )
                    training_stats["failed_ensembles"] += 1
                    regime_stats["failed_models"] += 1

                regime_stats["training_time"] += model_training_time

            regime_total_time = time.time() - regime_start_time
            regime_stats["training_time"] = regime_total_time
            training_stats["regime_stats"][regime_key] = regime_stats

            self.logger.info(
                f"📊 {regime_key} summary: {regime_stats['successful_models']}/{regime_stats['model_types']} models successful, "
                f"time: {regime_total_time:.2f}s",
            )

        # Train global meta-learner with enhanced predictions
        self.logger.info("🧠 Training enhanced global meta-learner...")
        meta_start_time = time.time()

        self._train_enhanced_global_meta_learner(asset, prepared_data)

        meta_training_time = time.time() - meta_start_time
        total_time = time.time() - start_time

        # Final training summary
        self.logger.info("✅ Enhanced multi-timeframe ensemble training completed!")
        self.logger.info(f"⏱️ Total training time: {total_time:.2f}s")
        self.logger.info("📊 Training summary:")
        self.logger.info(f"   - Asset: {asset}")
        self.logger.info(f"   - Regimes: {len(self.regime_ensembles)}")
        self.logger.info(f"   - Model types: {len(self.model_types)}")
        self.logger.info(f"   - Total ensembles: {training_stats['total_ensembles']}")
        self.logger.info(f"   - Successful: {training_stats['successful_ensembles']}")
        self.logger.info(f"   - Failed: {training_stats['failed_ensembles']}")
        self.logger.info(
            f"   - Success rate: {training_stats['successful_ensembles']/training_stats['total_ensembles']*100:.1f}%",
        )
        self.logger.info(f"   - Meta-learner training time: {meta_training_time:.2f}s")

        # Detailed regime statistics
        for regime_key, stats in training_stats["regime_stats"].items():
            success_rate = (
                stats["successful_models"] / stats["model_types"] * 100
                if stats["model_types"] > 0
                else 0
            )
            self.logger.info(
                f"   - {regime_key}: {stats['successful_models']}/{stats['model_types']} models "
                f"({success_rate:.1f}%), time: {stats['training_time']:.2f}s",
            )

    def _initialize_enhanced_ensembles(self):
        """Initialize enhanced regime ensembles."""
        self.logger.info("🔧 Initializing enhanced regime ensembles...")

        for regime_key in self.regime_ensembles:
            self.enhanced_regime_ensembles[regime_key] = {}
            self.logger.debug(f"📊 Initialized {regime_key} ensemble")

    def _train_enhanced_global_meta_learner(
        self,
        asset: str,
        prepared_data: dict[str, pd.DataFrame],
    ):
        """Train global meta-learner with enhanced multi-timeframe predictions."""
        self.logger.info("🧠 Training enhanced global meta-learner...")

        # Collect enhanced predictions for meta-learner training
        meta_learner_data = []
        ensemble_count = 0

        for regime_key, enhanced_ensembles in self.enhanced_regime_ensembles.items():
            self.logger.info(f"📊 Collecting predictions from {regime_key} regime...")

            for ensemble_key, ensemble in enhanced_ensembles.items():
                if ensemble.trained:
                    ensemble_count += 1
                    self.logger.debug(f"📊 Collecting from {ensemble_key}...")

                    # Get predictions from multi-timeframe ensemble
                    predictions = self._get_enhanced_predictions_for_meta_learner(
                        ensemble,
                        prepared_data,
                    )

                    if predictions:
                        meta_learner_data.extend(predictions)
                        self.logger.debug(
                            f"📊 Added {len(predictions)} predictions from {ensemble_key}",
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ No predictions collected from {ensemble_key}",
                        )
                else:
                    self.print(warning("⚠️ {ensemble_key} not trained, skipping"))

        self.logger.info(f"📊 Collected data from {ensemble_count} trained ensembles")

        # Train global meta-learner
        if meta_learner_data:
            self.logger.info(
                f"📊 Training global meta-learner with {len(meta_learner_data)} data points...",
            )
            self._train_global_meta_learner(meta_learner_data)
        else:
            self.logger.warning(
                "⚠️ No enhanced data collected for global meta-learner training",
            )

    def _get_enhanced_predictions_for_meta_learner(
        self,
        ensemble: MultiTimeframeEnsemble,
        prepared_data: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """Get enhanced predictions for meta-learner training."""
        predictions = []

        try:
            # Use 1h data as base for meta-learner training
            base_timeframe = "1h"
            if base_timeframe in prepared_data:
                data = prepared_data[base_timeframe]
                self.logger.debug(
                    f"📊 Using {base_timeframe} data for meta-learner training",
                )

                # Get predictions from multi-timeframe ensemble
                for idx, row in data.iterrows():
                    features = pd.DataFrame([row])

                    prediction_output = ensemble.get_prediction(features)

                    if prediction_output:
                        predictions.append(
                            {
                                "timestamp": idx,
                                "regime": ensemble.regime,
                                "model_name": ensemble.model_name,
                                "prediction": prediction_output.get(
                                    "prediction",
                                    "HOLD",
                                ),
                                "confidence": prediction_output.get("confidence", 0.0),
                                "timeframe_predictions": prediction_output.get(
                                    "timeframe_predictions",
                                    {},
                                ),
                                "timeframe_confidences": prediction_output.get(
                                    "timeframe_confidences",
                                    {},
                                ),
                                "true_target": "HOLD",  # In practice, use actual target
                            },
                        )
            else:
                self.logger.warning(
                    f"⚠️ No {base_timeframe} data available for meta-learner training",
                )

        except Exception:
            self.print(error("💥 Error getting enhanced predictions: {e}"))

        return predictions

    def get_all_predictions(
        self,
        asset: str,
        current_features: pd.DataFrame,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Get enhanced predictions from all multi-timeframe ensembles.

        Args:
            asset: Asset symbol
            current_features: Current market features
            **kwargs: Additional arguments

        Returns:
            Enhanced prediction with multi-timeframe details
        """
        self.logger.info(f"🔮 Getting enhanced predictions for {asset}")

        primary_regime = self.get_current_regime(current_features)
        self.logger.info(f"📊 Primary regime: {primary_regime}")

        # Collect predictions from all enhanced ensembles
        enhanced_predictions = {}
        enhanced_confidences = {}
        combined_base_predictions = {}
        timeframe_details = {}

        ensemble_count = 0
        successful_ensembles = 0

        for regime_key, enhanced_ensembles in self.enhanced_regime_ensembles.items():
            self.logger.debug(f"📊 Processing {regime_key} regime...")

            for ensemble_key, ensemble in enhanced_ensembles.items():
                ensemble_count += 1

                if ensemble.trained:
                    self.logger.debug(f"📊 Getting prediction from {ensemble_key}...")

                    prediction_output = ensemble.get_prediction(
                        current_features,
                        **kwargs,
                    )

                    if prediction_output:
                        successful_ensembles += 1

                        # Store enhanced predictions
                        enhanced_predictions[ensemble_key] = prediction_output.get(
                            "prediction",
                            "HOLD",
                        )
                        enhanced_confidences[ensemble_key] = prediction_output.get(
                            "confidence",
                            0.0,
                        )

                        # Store timeframe details
                        timeframe_details[ensemble_key] = {
                            "timeframe_predictions": prediction_output.get(
                                "timeframe_predictions",
                                {},
                            ),
                            "timeframe_confidences": prediction_output.get(
                                "timeframe_confidences",
                                {},
                            ),
                            "model_name": prediction_output.get("model_name", ""),
                            "regime": prediction_output.get("regime", ""),
                        }

                        # Store base predictions
                        combined_base_predictions[ensemble_key] = prediction_output.get(
                            "prediction",
                            "HOLD",
                        )

                        self.logger.debug(
                            f"📊 {ensemble_key}: {prediction_output.get('prediction', 'HOLD')} "
                            f"(confidence: {prediction_output.get('confidence', 0.0):.3f})",
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ No prediction output from {ensemble_key}",
                        )
                else:
                    self.print(warning("⚠️ {ensemble_key} not trained"))

        self.logger.info(
            f"📊 Processed {ensemble_count} ensembles, {successful_ensembles} successful",
        )

        # Use global meta-learner for final prediction
        if enhanced_predictions and enhanced_confidences:
            self.logger.info("🧠 Combining predictions with global meta-learner...")
            final_prediction, final_confidence = (
                self._predict_with_enhanced_global_meta_learner(
                    primary_regime,
                    enhanced_predictions,
                    enhanced_confidences,
                    current_features,
                )
            )
        else:
            self.print(warning("⚠️ No enhanced predictions available, using fallback"))
            final_prediction, final_confidence = "HOLD", 0.0

        # Get current ensemble weights
        current_ensemble_weights_snapshot = {
            regime: ens.get_current_weights()
            if hasattr(ens, "get_current_weights")
            else {}
            for regime, ens in self.enhanced_regime_ensembles.items()
        }

        self.logger.info(
            f"🎯 Final prediction: {final_prediction} (confidence: {final_confidence:.3f})",
        )

        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "regime": primary_regime,
            "base_predictions": combined_base_predictions,
            "ensemble_weights": current_ensemble_weights_snapshot,
            "enhanced_predictions": enhanced_predictions,
            "enhanced_confidences": enhanced_confidences,
            "timeframe_details": timeframe_details,
            "multi_timeframe_enabled": True,
        }

    def _predict_with_enhanced_global_meta_learner(
        self,
        primary_regime: str,
        enhanced_predictions: dict[str, str],
        enhanced_confidences: dict[str, float],
        current_features: pd.DataFrame,
    ) -> tuple[str, float]:
        """Make final prediction using enhanced global meta-learner."""
        try:
            if (
                not hasattr(self, "global_meta_learner")
                or self.global_meta_learner is None
            ):
                self.logger.warning(
                    "⚠️ No global meta-learner available, using simple combination",
                )
                # Fallback to simple averaging
                return self._simple_enhanced_combine_predictions(
                    enhanced_predictions,
                    enhanced_confidences,
                )

            self.logger.debug("🧠 Using global meta-learner for final prediction...")

            # Prepare enhanced meta-features
            meta_features = self._prepare_enhanced_meta_features(
                enhanced_predictions,
                enhanced_confidences,
                current_features,
            )

            # Scale features
            meta_features_scaled = self.global_meta_scaler.transform([meta_features])

            # Get prediction
            prediction_encoded = self.global_meta_learner.predict(meta_features_scaled)[
                0
            ]
            prediction = self.global_meta_label_encoder.inverse_transform(
                [prediction_encoded],
            )[0]

            # Get confidence
            if hasattr(self.global_meta_learner, "predict_proba"):
                probas = self.global_meta_learner.predict_proba(meta_features_scaled)
                confidence = np.max(probas[0])
            else:
                confidence = 0.5

            self.logger.debug(
                f"🎯 Global meta-learner prediction: {prediction} (confidence: {confidence:.3f})",
            )
            return prediction, confidence

        except Exception as e:
            self.logger.exception(
                f"💥 Error in enhanced global meta-learner prediction: {e}",
            )
            return "HOLD", 0.0

    def _prepare_enhanced_meta_features(
        self,
        enhanced_predictions: dict[str, str],
        enhanced_confidences: dict[str, float],
        current_features: pd.DataFrame,
    ) -> list[float]:
        """Prepare enhanced meta-features for global meta-learner."""
        meta_features = []

        self.logger.debug("🔧 Preparing enhanced meta-features...")

        # Add predictions and confidences from each enhanced ensemble
        for ensemble_key, prediction in enhanced_predictions.items():
            confidence = enhanced_confidences.get(ensemble_key, 0.0)

            # One-hot encode prediction
            pred_encoded = [
                1.0 if prediction == "BUY" else 0.0,
                1.0 if prediction == "SELL" else 0.0,
                1.0 if prediction == "HOLD" else 0.0,
            ]
            meta_features.extend(pred_encoded)
            meta_features.append(confidence)

            self.logger.debug(
                f"📊 {ensemble_key}: {prediction} (confidence: {confidence:.3f})",
            )

        self.logger.debug(f"📊 Meta-features prepared: {len(meta_features)} features")
        return meta_features

    def _simple_enhanced_combine_predictions(
        self,
        enhanced_predictions: dict[str, str],
        enhanced_confidences: dict[str, float],
    ) -> tuple[str, float]:
        """Simple combination of enhanced predictions (fallback)."""
        try:
            if not enhanced_predictions:
                self.print(warning("⚠️ No enhanced predictions available"))
                return "HOLD", 0.0

            self.logger.debug("📊 Using simple prediction combination...")

            # Count predictions
            pred_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            total_confidence = 0.0

            for pred, conf in zip(
                enhanced_predictions.values(),
                enhanced_confidences.values(),
                strict=False,
            ):
                pred_counts[pred] += 1
                total_confidence += conf

            # Get most common prediction
            final_prediction = max(pred_counts, key=pred_counts.get)

            # Average confidence
            final_confidence = (
                total_confidence / len(enhanced_confidences)
                if enhanced_confidences
                else 0.0
            )

            self.logger.debug(
                f"📊 Simple combination: {pred_counts}, final: {final_prediction} (confidence: {final_confidence:.3f})",
            )
            return final_prediction, final_confidence

        except Exception as e:
            self.logger.exception(
                f"💥 Error in simple enhanced prediction combination: {e}",
            )
            return "HOLD", 0.0

    def save_enhanced_models(self, asset: str, model_path_prefix: str):
        """Save all enhanced multi-timeframe models."""
        self.logger.info(f"💾 Saving all enhanced multi-timeframe models for {asset}")

        saved_count = 0
        total_count = 0

        for enhanced_ensembles in self.enhanced_regime_ensembles.values():
            for ensemble_key, ensemble in enhanced_ensembles.items():
                total_count += 1

                if ensemble.trained:
                    model_path = os.path.join(
                        model_path_prefix,
                        f"{asset}_{ensemble_key}",
                    )
                    success = ensemble.save_model(model_path)

                    if success:
                        saved_count += 1
                        self.logger.debug(f"💾 Saved {ensemble_key}")
                    else:
                        self.print(failed("⚠️ Failed to save {ensemble_key}"))
                else:
                    self.print(warning("⚠️ {ensemble_key} not trained, skipping save"))

        self.logger.info(f"💾 Saved {saved_count}/{total_count} enhanced models")

    def load_enhanced_models(self, asset: str, model_path_prefix: str):
        """Load all enhanced multi-timeframe models."""
        self.logger.info(f"📂 Loading all enhanced multi-timeframe models for {asset}")

        loaded_count = 0
        total_count = 0

        for regime_key in self.regime_ensembles:
            for model_type in self.model_types:
                ensemble_key = f"{regime_key}_{model_type}"
                total_count += 1

                model_path = os.path.join(model_path_prefix, f"{asset}_{ensemble_key}")

                if ensemble_key not in self.enhanced_regime_ensembles[regime_key]:
                    self.enhanced_regime_ensembles[regime_key][ensemble_key] = (
                        MultiTimeframeEnsemble(
                            model_name=model_type,
                            regime=regime_key,
                            config=self.config,
                        )
                    )

                success = self.enhanced_regime_ensembles[regime_key][
                    ensemble_key
                ].load_model(model_path)

                if success:
                    loaded_count += 1
                    self.logger.debug(f"📂 Loaded {ensemble_key}")
                else:
                    self.print(failed("⚠️ Failed to load {ensemble_key}"))

        self.logger.info(f"📂 Loaded {loaded_count}/{total_count} enhanced models")

    def get_enhanced_ensemble_info(self) -> dict[str, Any]:
        """Get information about enhanced ensembles."""
        self.logger.info("📊 Getting enhanced ensemble information...")

        info = {
            "active_timeframes": self.active_timeframes,
            "timeframe_set": self.timeframe_set,
            "model_types": self.model_types,
            "enhanced_ensembles": {},
        }

        total_ensembles = 0
        trained_ensembles = 0

        for regime_key, enhanced_ensembles in self.enhanced_regime_ensembles.items():
            info["enhanced_ensembles"][regime_key] = {}

            for ensemble_key, ensemble in enhanced_ensembles.items():
                total_ensembles += 1

                info["enhanced_ensembles"][regime_key][ensemble_key] = {
                    "trained": ensemble.trained,
                    "model_name": ensemble.model_name,
                    "regime": ensemble.regime,
                    "timeframe_models": list(ensemble.timeframe_models.keys()),
                }

                if ensemble.trained:
                    trained_ensembles += 1

        self.logger.info(
            f"📊 Enhanced ensemble summary: {trained_ensembles}/{total_ensembles} ensembles trained",
        )

        return info
