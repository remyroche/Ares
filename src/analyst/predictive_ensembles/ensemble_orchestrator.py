import logging
import os
import pandas as pd
from joblib import dump, load

from src.utils.logger import system_logger
from .regime_ensembles.bull_trend_ensemble import BullTrendEnsemble
from .regime_ensembles.bear_trend_ensemble import BearTrendEnsemble
from .regime_ensembles.sideways_range_ensemble import SidewaysRangeEnsemble
from .regime_ensembles.sr_zone_action_ensemble import SRZoneActionEnsemble
from .regime_ensembles.high_impact_candle_ensemble import HighImpactCandleEnsemble

class RegimePredictiveEnsembles:
    """
    ## CHANGE: Fully updated the Ensemble Orchestrator.
    ## This class is now a clean, high-level orchestrator that correctly manages
    ## the training and prediction workflows for all specialized ensembles. It no longer
    ## simulates its own data and properly delegates tasks, aligning with the
    ## advanced, modular architecture of the system.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {})
        self.logger = system_logger.getChild('PredictiveEnsembles.Orchestrator')
        
        # Initialize all possible ensemble instances
        self.regime_ensembles = {
            "BULL_TREND": BullTrendEnsemble(config, "BullTrendEnsemble"),
            "BEAR_TREND": BearTrendEnsemble(config, "BearTrendEnsemble"),
            "SIDEWAYS_RANGE": SidewaysRangeEnsemble(config, "SidewaysRangeEnsemble"),
            "SR_ZONE_ACTION": SRZoneActionEnsemble(config, "SRZoneActionEnsemble"),
            "HIGH_IMPACT_CANDLE": HighImpactCandleEnsemble(config, "HighImpactCandleEnsemble")
        }
        self.model_storage_path = self.config.get("model_storage_path", "models/analyst/") + "ensembles/"
        os.makedirs(self.model_storage_path, exist_ok=True)

    def train_all_models(self, asset: str, prepared_data: pd.DataFrame):
        """
        Orchestrates the training of all regime-specific ensembles.
        It splits the prepared data by regime and passes the relevant slice to each ensemble.
        """
        self.logger.info(f"Orchestrator: Starting training for all ensembles for asset {asset}...")
        
        if 'regime' not in prepared_data.columns or 'target' not in prepared_data.columns:
            self.logger.error("Prepared data is missing 'regime' or 'target' column. Halting training.")
            return

        for regime, ensemble in self.regime_ensembles.items():
            self.logger.info(f"--- Processing ensemble for {regime} ---")
            
            regime_data = prepared_data[prepared_data['regime'] == regime]
            
            if regime_data.empty or len(regime_data['target'].unique()) < 2:
                self.logger.warning(f"Insufficient or single-class data for {regime}. Skipping training.")
                continue

            historical_features = regime_data.drop(columns=['target'])
            historical_targets = regime_data['target']

            ensemble.train_ensemble(historical_features, historical_targets)
            
            if ensemble.trained:
                model_path = os.path.join(self.model_storage_path, f"{asset}_{regime.lower()}_ensemble.joblib")
                self.save_model(ensemble, model_path)

    def get_all_predictions(self, asset: str, current_features: pd.DataFrame, **kwargs) -> dict:
        """
        Gets a prediction by identifying the current regime and delegating to the
        appropriate trained ensemble.
        """
        regime = self.get_current_regime(current_features)
        ensemble = self.regime_ensembles.get(regime)

        if not ensemble:
            self.logger.warning(f"No ensemble defined for regime: {regime}.")
            return {"prediction": "HOLD", "confidence": 0.0, "regime": regime, "base_predictions": {}}

        if not ensemble.trained:
            self.logger.warning(f"Ensemble for {regime} is not trained. Attempting to load model...")
            model_path = os.path.join(self.model_storage_path, f"{asset}_{regime.lower()}_ensemble.joblib")
            if not self.load_model(ensemble, model_path):
                 self.logger.error(f"Failed to load model for {regime}. Cannot make prediction.")
                 return {"prediction": "HOLD", "confidence": 0.0, "regime": regime, "base_predictions": {}}

        final_prediction_output = ensemble.get_prediction(current_features, **kwargs)
        
        # Get base model predictions for the dynamic weighter
        base_predictions = {}
        if hasattr(ensemble, '_get_meta_features'):
             # We get the meta-features, which are the outputs of the base models
             base_predictions = ensemble._get_meta_features(current_features, is_live=True, **kwargs)

        return {
            "prediction": final_prediction_output.get("prediction", "HOLD"),
            "confidence": final_prediction_output.get("confidence", 0.0),
            "regime": regime,
            "base_predictions": base_predictions,
        }

    def get_current_regime(self, current_features: pd.DataFrame) -> str:
        """Determines the most recent market regime from the feature data."""
        if not current_features.empty and 'regime' in current_features.columns:
            return current_features['regime'].iloc[-1]
        return "UNKNOWN"

    def save_model(self, ensemble_instance, path: str):
        """Saves a trained ensemble instance to a file."""
        try:
            dump(ensemble_instance, path)
            self.logger.info(f"Successfully saved trained ensemble to {path}")
        except Exception as e:
            self.logger.error(f"Error saving ensemble model to {path}: {e}", exc_info=True)

    def load_model(self, ensemble_instance, path: str) -> bool:
        """Loads a trained ensemble instance from a file."""
        if not os.path.exists(path):
            return False
        try:
            loaded_ensemble = load(path)
            # Update the existing instance's state instead of replacing it
            ensemble_instance.__dict__.update(loaded_ensemble.__dict__)
            self.logger.info(f"Successfully loaded pre-trained ensemble from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ensemble model from {path}: {e}", exc_info=True)
            return False

    def load_weights(self, weights: dict):
        """Loads updated weights into the ensembles for dynamic weighting."""
        for regime, ensemble_weights in weights.items():
            if regime in self.regime_ensembles:
                self.regime_ensembles[regime].ensemble_weights = ensemble_weights

    def get_current_weights(self) -> dict:
        """Returns the current weights of all ensembles."""
        return {regime: ens.ensemble_weights for regime, ens in self.regime_ensembles.items()}
