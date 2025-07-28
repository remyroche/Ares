# src/analyst/predictive_ensembles/ensemble_orchestrator.py
import pandas as pd
import numpy as np
import os # For path manipulation
from src.utils.logger import system_logger

# Import individual ensemble classes
from .regime_ensembles.bull_trend_ensemble import BullTrendEnsemble
from .regime_ensembles.bear_trend_ensemble import BearTrendEnsemble
from .regime_ensembles.sideways_range_ensemble import SidewaysRangeEnsemble
from .regime_ensembles.sr_zone_action_ensemble import SRZoneActionEnsemble
from .regime_ensembles.high_impact_candle_ensemble import HighImpactCandleEnsemble

class RegimePredictiveEnsembles:
    """
    Orchestrates regime-specific predictive ensembles, loading and utilizing the
    appropriate ensemble based on the detected market regime.
    """
    def __init__(self, config):
        self.config = config.get("analyst", {}).get("regime_predictive_ensembles", {})
        self.logger = system_logger.getChild('PredictiveEnsembles.Orchestrator')
        
        # Initialize all possible ensemble instances
        self.ensembles = {
            "BULL_TREND": BullTrendEnsemble(config, "BULL_TREND"), # Pass full config for sub-ensembles
            "BEAR_TREND": BearTrendEnsemble(config, "BEAR_TREND"),
            "SIDEWAYS_RANGE": SidewaysRangeEnsemble(config, "SIDEWAYS_RANGE"),
            "SR_ZONE_ACTION": SRZoneActionEnsemble(config, "SR_ZONE_ACTION"),
            "HIGH_IMPACT_CANDLE": HighImpactCandleEnsemble(config, "HIGH_IMPACT_CANDLE")
        }
        self.model_storage_path = config['analyst'].get("model_storage_path", "models/analyst/") + "ensembles/"
        os.makedirs(self.model_storage_path, exist_ok=True) # Ensure ensemble model storage path exists


    def train_all_ensembles(self, historical_features: pd.DataFrame, historical_targets: dict):
        """
        Trains all regime-specific ensembles.
        `historical_targets` would be a dictionary mapping regime names to their respective target series.
        """
        self.logger.info("Orchestrator: Training all Regime-Specific Predictive Ensembles...")
        
        # Ensure historical_features has all necessary columns for ensemble training
        # This includes the new S/R interaction type features
        required_sr_features = ['Is_SR_Support_Interacting', 'Is_SR_Resistance_Interacting']
        for col in required_sr_features:
            if col not in historical_features.columns:
                historical_features[col] = 0 # Add missing columns with default 0

        for regime, ensemble_instance in self.ensembles.items():
            self.logger.info(f"--- Orchestrator: Processing ensemble for {regime} regime ---")
            
            # Attempt to load the ensemble model first
            ensemble_model_path = os.path.join(self.model_storage_path, f"{regime.lower()}_ensemble.joblib")
            if ensemble_instance.load_model(ensemble_model_path):
                self.logger.info(f"Pre-trained {regime} ensemble model loaded successfully.")
            else:
                self.logger.warning(f"Pre-trained {regime} ensemble model not found or failed to load. Training a new model.")
                
                # Simulate a target for each regime for training purposes
                simulated_target = pd.Series(0, index=historical_features.index) # Default target
                if regime == "BULL_TREND":
                    simulated_target = (historical_features['close'].pct_change(5).shift(-5) > 0.005).astype(int)
                elif regime == "BEAR_TREND":
                    simulated_target = (historical_features['close'].pct_change(5).shift(-5) < -0.005).astype(int)
                elif regime == "SIDEWAYS_RANGE":
                    simulated_target = ((historical_features['close'].pct_change(5).shift(-5).abs() < 0.002) & 
                                        (historical_features['close'].pct_change(5).shift(-5).abs() > 0.0001)).astype(int)
                elif regime == "SR_ZONE_ACTION":
                    target_lookahead_sr = 2
                    price_change_sr = historical_features['close'].pct_change(target_lookahead_sr).shift(-target_lookahead_sr)
                    up_threshold_sr = 0.002
                    down_threshold_sr = -0.002
                    
                    simulated_target_sr = pd.Series('HOLD_SR', index=price_change_sr.index)
                    simulated_target_sr[price_change_sr > up_threshold_sr] = 'BREAKTHROUGH_UP_GENERIC'
                    simulated_target_sr[price_change_sr < down_threshold_sr] = 'BREAKTHROUGH_DOWN_GENERIC'
                    simulated_target = simulated_target_sr
                elif regime == "HIGH_IMPACT_CANDLE":
                    simulated_target = (historical_features['close'].pct_change(1).shift(-1).abs() > 0.005).astype(int)

                # Drop NaNs from target and align with features
                aligned_features = historical_features.loc[simulated_target.dropna().index]
                aligned_target = simulated_target.dropna()

                if not aligned_features.empty and not aligned_target.empty:
                    ensemble_instance.train_ensemble(aligned_features, aligned_target)
                    if ensemble_instance.trained: # Only save if training was successful
                        ensemble_instance.save_model(ensemble_model_path)
                else:
                    self.logger.warning(f"Orchestrator: Insufficient aligned data for training {regime} ensemble. Skipping.")

        self.logger.info("Orchestrator: Regime-Specific Predictive Ensembles training/loading complete.")

    def get_ensemble_prediction(self, regime: str, current_features: pd.DataFrame, 
                                klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, 
                                order_book_data: dict, current_price: float) -> dict:
        """
        Gets a combined prediction and confidence score for the given regime
        by delegating to the appropriate ensemble.
        """
        if regime not in self.ensembles:
            self.logger.warning(f"Orchestrator: No ensemble defined for regime: {regime}. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        ensemble_instance = self.ensembles[regime]
        
        # Ensure the ensemble is trained before attempting to get a prediction
        if not ensemble_instance.trained:
            self.logger.warning(f"Orchestrator: Ensemble for {regime} is not trained. Attempting to load for prediction.")
            ensemble_model_path = os.path.join(self.model_storage_path, f"{regime.lower()}_ensemble.joblib")
            if not ensemble_instance.load_model(ensemble_model_path):
                self.logger.error(f"Ensemble for {regime} is not trained and failed to load. Returning HOLD.")
                return {"prediction": "HOLD", "confidence": 0.0}


        self.logger.info(f"Orchestrator: Getting prediction from {regime} ensemble.")
        return ensemble_instance.get_prediction(current_features, klines_df, agg_trades_df, order_book_data, current_price)
