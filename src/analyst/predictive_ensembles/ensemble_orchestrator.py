# src/analyst/predictive_ensembles/ensemble_orchestrator.py
import pandas as pd
import numpy as np
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
            "BULL_TREND": BullTrendEnsemble(self.config, "BULL_TREND"),
            "BEAR_TREND": BearTrendEnsemble(self.config, "BEAR_TREND"),
            "SIDEWAYS_RANGE": SidewaysRangeEnsemble(self.config, "SIDEWAYS_RANGE"),
            "SR_ZONE_ACTION": SRZoneActionEnsemble(self.config, "SR_ZONE_ACTION"),
            "HIGH_IMPACT_CANDLE": HighImpactCandleEnsemble(self.config, "HIGH_IMPACT_CANDLE")
        }

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
            self.logger.info(f"--- Orchestrator: Training ensemble for {regime} regime ---")
            # In a real scenario, you'd filter historical_features and historical_targets
            # based on the regime for training each specific ensemble.
            # For this placeholder, we'll pass the full historical_features and a dummy target.
            
            # Simulate a target for each regime for training purposes
            # This would be derived from actual historical price movements relative to the regime
            simulated_target = pd.Series(0, index=historical_features.index) # Default target
            if regime == "BULL_TREND":
                # Example: target is 1 if price increased by > X% in next Y periods, 0 otherwise
                simulated_target = (historical_features['close'].pct_change(5).shift(-5) > 0.005).astype(int)
            elif regime == "BEAR_TREND":
                simulated_target = (historical_features['close'].pct_change(5).shift(-5) < -0.005).astype(int)
            elif regime == "SIDEWAYS_RANGE":
                simulated_target = ((historical_features['close'].pct_change(5).shift(-5).abs() < 0.002) & 
                                    (historical_features['close'].pct_change(5).shift(-5).abs() > 0.0001)).astype(int) # Small, non-zero change
            elif regime == "SR_ZONE_ACTION":
                # For SR_ZONE_ACTION, use the more nuanced targets defined in the ensemble itself
                # This requires the ensemble to generate its own targets or for them to be passed in historical_targets
                # For now, we'll use a generic target to allow training to proceed, and rely on the ensemble's internal target logic.
                # A more robust system would have historical_targets pre-labeled for each regime.
                target_lookahead_sr = 2
                price_change_sr = historical_features['close'].pct_change(target_lookahead_sr).shift(-target_lookahead_sr)
                up_threshold_sr = 0.002
                down_threshold_sr = -0.002
                
                simulated_target_sr = pd.Series('HOLD_SR', index=price_change_sr.index)
                simulated_target_sr[price_change_sr > up_threshold_sr] = 'BREAKTHROUGH_UP_GENERIC'
                simulated_target_sr[price_change_sr < down_threshold_sr] = 'BREAKTHROUGH_DOWN_GENERIC'
                simulated_target = simulated_target_sr # Use this for SR_ZONE_ACTION
            elif regime == "HIGH_IMPACT_CANDLE":
                simulated_target = (historical_features['close'].pct_change(1).shift(-1).abs() > 0.005).astype(int) # Follow through or reversal

            # Drop NaNs from target and align with features
            aligned_features = historical_features.loc[simulated_target.dropna().index]
            aligned_target = simulated_target.dropna()

            if not aligned_features.empty and not aligned_target.empty:
                ensemble_instance.train_ensemble(aligned_features, aligned_target)
            else:
                self.logger.warning(f"Orchestrator: Insufficient aligned data for training {regime} ensemble. Skipping.")

        self.logger.info("Orchestrator: Regime-Specific Predictive Ensembles training complete (Placeholders).")

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
            self.logger.warning(f"Orchestrator: Ensemble for {regime} is not trained. Cannot get prediction. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}


        self.logger.info(f"Orchestrator: Getting prediction from {regime} ensemble.")
        return ensemble_instance.get_prediction(current_features, klines_df, agg_trades_df, order_book_data, current_price)
