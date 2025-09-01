# src/analyst/predictive_ensembles/enhanced_ensemble_orchestrator.py

"""
Enhanced Ensemble Orchestrator

This integrates multi-timeframe training into the existing ensemble system, making each individual model (XGBoost, LSTM, etc.) a multi-timeframe ensemble.
"""

import os
import time
from typing import Any

import pandas as pd

from src.analyst.predictive_ensembles.ensemble_orchestrator import (
import RegimePredictiveEnsembles,
    RegimePredictiveEnsembles,
)
from src.analyst.predictive_ensembles.multi_timeframe_ensemble import (
import MultiTimeframeEnsemble,
    MultiTimeframeEnsemble,
)
from src.config import CONFIG
from src.utils.logger import system_logger


import class EnhancedRegimePredictiveEnsembles
class EnhancedRegimePredictiveEnsembles(RegimePredictiveEnsembles):
    """
    Enhanced ensemble orchestrator that integrates multi-timeframe training.

    Each individual model (XGBoost, LSTM, etc.) becomes a multi-timeframe ensemble.
    """

    def __init__(self, config: dict[str, Any]):
    pass
    pass
    pass
        super().__init__(config)
        self.logger = system_logger.getChild("EnhancedRegimePredictiveEnsembles")

        # Multi-timeframe configuration
        self.timeframes = CONFIG.get("TIMEFRAMES", {})
        self.timeframe_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "intraday")
        self.active_timeframes = CONFIG.get("TIMEFRAME_SETS", {}).get(
            self.timeframe_set, [],
        )

        # Model types to train
        self.model_types = ["xgboost", "lstm", "random_forest"]

        # Enhanced regime ensembles with multi-timeframe models
        self.enhanced_regime_ensembles: dict[
            str, dict[str, MultiTimeframeEnsemble],
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
    pass
    pass
    pass
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
    pass
    pass
    pass
                self.logger.info(
                    f"🔧 [{regime_idx}.{model_idx}] Training {model_type} for regime: {regime_key}",
                )

                try:
                    # Create multi-timeframe ensemble for this model type
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                    ensemble = MultiTimeframeEnsemble(
                        model_type=model_type,
                        timeframes=self.active_timeframes,
                        config=self.config,
                    )

                    # Train the ensemble
                    ensemble.train(prepared_data)

                    # Store the trained ensemble
                    if regime_key not in self.enhanced_regime_ensembles:
    pass
    pass
    pass
                        self.enhanced_regime_ensembles[regime_key] = {}
                    self.enhanced_regime_ensembles[regime_key][model_type] = ensemble

                    regime_stats["successful_models"] += 1
                    self.logger.info(
                        f"✅ [{regime_idx}.{model_idx}] {model_type} trained successfully for regime: {regime_key}",
                    )

                except Exception as e:
                    regime_stats["failed_models"] += 1
                    self.logger.error(
                        f"❌ [{regime_idx}.{model_idx}] Failed to train {model_type} for regime {regime_key}: {e}",
                    )

                regime_stats["model_types"] += 1

            # Calculate regime training time
            regime_stats["training_time"] = time.time() - regime_start_time
            training_stats["regime_stats"][regime_key] = regime_stats

            self.logger.info(
                f"📊 Regime {regime_key} completed: {regime_stats['successful_models']}/{regime_stats['model_types']} models successful",
            )

        # Calculate overall statistics
        training_stats["total_ensembles"] = len(self.regime_ensembles) * len(self.model_types)
        training_stats["successful_ensembles"] = sum(
            stats["successful_models"] for stats in training_stats["regime_stats"].values()
        )
        training_stats["failed_ensembles"] = sum(
            stats["failed_models"] for stats in training_stats["regime_stats"].values()
        )

        total_time = time.time() - start_time

        # Log final statistics
        self.logger.info("🎉 Enhanced ensemble training completed!")
        self.logger.info(f"📊 Total ensembles: {training_stats['total_ensembles']}")
        self.logger.info(f"✅ Successful: {training_stats['successful_ensembles']}")
        self.logger.info(f"❌ Failed: {training_stats['failed_ensembles']}")
        self.logger.info(f"⏱️ Total time: {total_time:.2f} seconds")

        return training_stats

    def _initialize_enhanced_ensembles(self):
    pass
    pass
    pass
        """Initialize enhanced regime ensembles."""
        self.logger.info("🔧 Initializing enhanced regime ensembles...")
        self.enhanced_regime_ensembles = {}

    def predict(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pass
    pass
    pass
        """
        Make predictions using all enhanced ensembles.

        Args:
            data: Multi-timeframe data for prediction

        Returns:
            Dictionary with regime -> predictions mapping
        """
        predictions = {}

        for regime_key, regime_ensembles in self.enhanced_regime_ensembles.items():
    pass
    pass
    pass
            regime_predictions = {}

            for model_type, ensemble in regime_ensembles.items():
    pass
    pass
    pass
                try:
                    prediction = ensemble.predict(data)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                    regime_predictions[model_type] = prediction
                except Exception as e:
                    self.logger.error(f"❌ Prediction failed for {model_type} in regime {regime_key}: {e}")
                    regime_predictions[model_type] = None

            predictions[regime_key] = regime_predictions

        return predictions

    def save_models(self, base_path: str):
    pass
    pass
    pass
        """Save all trained models."""
        self.logger.info(f"💾 Saving enhanced ensemble models to {base_path}")

        for regime_key, regime_ensembles in self.enhanced_regime_ensembles.items():
    pass
    pass
    pass
            regime_path = os.path.join(base_path, f"regime_{regime_key}")
            os.makedirs(regime_path, exist_ok=True)

            for model_type, ensemble in regime_ensembles.items():
    pass
    pass
    pass
                try:
                    model_path = os.path.join(regime_path, f"{model_type}_ensemble.pkl")
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                    ensemble.save(model_path)
                    self.logger.info(f"✅ Saved {model_type} ensemble for regime {regime_key}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to save {model_type} ensemble for regime {regime_key}: {e}")

    def load_models(self, base_path: str):
    pass
    pass
    pass
        """Load all trained models."""
        self.logger.info(f"📂 Loading enhanced ensemble models from {base_path}")

        for regime_key in self.regime_ensembles.keys():
    pass
    pass
    pass
            regime_path = os.path.join(base_path, f"regime_{regime_key}")

            if regime_key not in self.enhanced_regime_ensembles:
    pass
    pass
    pass
                self.enhanced_regime_ensembles[regime_key] = {}

            for model_type in self.model_types:
    pass
    pass
    pass
                try:
                    model_path = os.path.join(regime_path, f"{model_type}_ensemble.pkl")
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                    if os.path.exists(model_path):
    pass
    pass
    pass
                        ensemble = MultiTimeframeEnsemble.load(model_path)
                        self.enhanced_regime_ensembles[regime_key][model_type] = ensemble
                        self.logger.info(f"✅ Loaded {model_type} ensemble for regime {regime_key}")
                    else:
                        self.logger.warning(f"⚠️ Model file not found: {model_path}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {model_type} ensemble for regime {regime_key}: {e}")

    def get_ensemble_summary(self) -> dict[str, Any]:
    pass
    pass
    pass
        """Get a summary of all ensembles."""
        summary = {
            "total_regimes": len(self.enhanced_regime_ensembles),
            "total_models": 0,
            "regime_details": {},
        }

        for regime_key, regime_ensembles in self.enhanced_regime_ensembles.items():
    pass
    pass
    pass
            regime_summary = {
                "model_count": len(regime_ensembles),
                "model_types": list(regime_ensembles.keys()),
                "is_trained": all(ensemble.is_trained for ensemble in regime_ensembles.values()),
            }
            summary["regime_details"][regime_key] = regime_summary
            summary["total_models"] += regime_summary["model_count"]

        return summary
