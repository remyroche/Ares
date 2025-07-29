import logging
import os
import pandas as pd
from joblib import dump, load
from typing import Any, Optional # Import Any and Optional

from src.utils.logger import system_logger
from src.config import CONFIG # Import CONFIG to get checkpoint paths
from .regime_ensembles.bull_trend_ensemble import BullTrendEnsemble
from .regime_ensembles.bear_trend_ensemble import BearTrendEnsemble
from .regime_ensembles.sideways_range_ensemble import SidewaysRangeEnsemble
from .regime_ensembles.sr_zone_action_ensemble import SRZoneActionEnsemble
from .regime_ensembles.high_impact_candle_ensemble import HighImpactCandleEnsemble

class RegimePredictiveEnsembles:
    """
    Orchestrates the training and prediction workflows for all specialized ensembles.
    Now includes checkpointing for ensemble models.
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
        # Model storage path is now dynamic based on checkpoint config
        self.model_storage_dir = os.path.join(CONFIG['CHECKPOINT_DIR'], "analyst_models", "ensembles")
        os.makedirs(self.model_storage_dir, exist_ok=True)

    def train_all_models(self, asset: str, prepared_data: pd.DataFrame, model_path_prefix: Optional[str] = None):
        """
        Orchestrates the training of all regime-specific ensembles.
        It splits the prepared data by regime and passes the relevant slice to each ensemble.
        
        Args:
            asset (str): The trading asset (e.g., "BTCUSDT").
            prepared_data (pd.DataFrame): The full prepared historical data with 'regime' and 'target' columns.
            model_path_prefix (str, optional): A prefix for saving models (e.g., includes fold_id).
        """
        self.logger.info(f"Orchestrator: Starting training for all ensembles for asset {asset} (prefix: {model_path_prefix})...")
        
        if 'Market_Regime_Label' not in prepared_data.columns or 'target' not in prepared_data.columns:
            self.logger.error("Prepared data is missing 'Market_Regime_Label' or 'target' column. Halting training.")
            return

        for regime_key, ensemble_instance in self.regime_ensembles.items():
            self.logger.info(f"--- Processing ensemble for {regime_key} ---")
            
            # Filter data for the current regime
            regime_data = prepared_data[prepared_data['Market_Regime_Label'] == regime_key]
            
            if regime_data.empty or len(regime_data['target'].unique()) < 2:
                self.logger.warning(f"Insufficient or single-class data for {regime_key}. Skipping training.")
                continue

            historical_features = regime_data.drop(columns=['target', 'Market_Regime_Label'], errors='ignore')
            historical_targets = regime_data['target']

            # Construct the specific model path for this ensemble and fold
            model_file_name = f"{regime_key.lower()}_ensemble.joblib"
            if model_path_prefix:
                # model_path_prefix will be like "checkpoints/analyst_models/ensemble_fold_X_"
                full_model_path = f"{model_path_prefix}{model_file_name}"
            else:
                # Default path if no prefix (e.g., for live mode, load final model)
                full_model_path = os.path.join(self.model_storage_dir, f"final_{model_file_name}")

            # Try to load the model first
            if os.path.exists(full_model_path):
                self.logger.info(f"Attempting to load {regime_key} ensemble from {full_model_path}...")
                if ensemble_instance.load_model(full_model_path):
                    self.logger.info(f"Successfully loaded {regime_key} ensemble.")
                    continue # Skip training if loaded successfully
                else:
                    self.logger.warning(f"Failed to load {regime_key} ensemble from {full_model_path}. Retraining.")
            
            # If not loaded, train the model
            ensemble_instance.train_ensemble(historical_features, historical_targets)
            
            # Save the trained model
            if ensemble_instance.trained:
                ensemble_instance.save_model(full_model_path)


    def get_all_predictions(self, asset: str, current_features: pd.DataFrame, **kwargs) -> dict:
        """
        Gets a prediction by identifying the current regime and delegating to the
        appropriate trained ensemble.
        """
        regime = self.get_current_regime(current_features)
        ensemble = self.regime_ensembles.get(regime)

        if not ensemble:
            self.logger.warning(f"No ensemble defined for regime: {regime}.")
            return {"prediction": "HOLD", "confidence": 0.0, "regime": regime, "base_predictions": {}, "ensemble_weights": {}}

        # For live prediction, we need to load the *final* trained model if not already loaded
        if not ensemble.trained:
            self.logger.warning(f"Ensemble for {regime} is not trained. Attempting to load final model...")
            final_model_file_name = os.path.join(self.model_storage_dir, f"final_{regime.lower()}_ensemble.joblib")
            if not ensemble.load_model(final_model_file_name):
                 self.logger.error(f"Failed to load final model for {regime}. Cannot make prediction.")
                 return {"prediction": "HOLD", "confidence": 0.0, "regime": regime, "base_predictions": {}, "ensemble_weights": {}}

        final_prediction_output = ensemble.get_prediction(current_features, **kwargs)
        
        base_predictions = {}
        if hasattr(ensemble, '_get_meta_features'):
             base_predictions = ensemble._get_meta_features(current_features, is_live=True, **kwargs)

        # Include the ensemble's current weights in the output
        current_ensemble_weights = ensemble.ensemble_weights if hasattr(ensemble, 'ensemble_weights') else {}

        return {
            "prediction": final_prediction_output.get("prediction", "HOLD"),
            "confidence": final_prediction_output.get("confidence", 0.0),
            "regime": regime,
            "base_predictions": base_predictions,
            "ensemble_weights": current_ensemble_weights, # Added ensemble weights here
        }

    def get_current_regime(self, current_features: pd.DataFrame) -> str:
        """Determines the most recent market regime from the feature data."""
        if not current_features.empty and 'Market_Regime_Label' in current_features.columns:
            return current_features['Market_Regime_Label'].iloc[-1]
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
            ensemble_instance.trained = True # Ensure 'trained' flag is set
            self.logger.info(f"Successfully loaded pre-trained ensemble from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ensemble model from {path}: {e}", exc_info=True)
            return False

    def load_weights(self, weights: dict):
        """Loads updated weights into the ensembles for dynamic weighting."""
        for regime, ensemble_weights in weights.items():
            if regime in self.regime_ensembles:
                # Assuming BaseEnsemble has an attribute 'ensemble_weights'
                self.regime_ensembles[regime].ensemble_weights = ensemble_weights

    def get_current_weights(self) -> dict:
        """Returns the current weights of all ensembles."""
        return {regime: ens.ensemble_weights for regime, ens in self.regime_ensembles.items()}
