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
        for regime, ensemble_instance in self.ensembles.items():
            self.logger.info(f"--- Orchestrator: Training ensemble for {regime} regime ---")
            # In a real scenario, you'd filter historical_features and historical_targets
            # based on the regime for training each specific ensemble.
            # For this placeholder, we'll pass the full historical_features and a dummy target.
            
            # Simulate a target for each regime for training purposes
            # This would be derived from actual historical price movements relative to the regime
            if regime == "BULL_TREND":
                # Example: target is 1 if price increased by > X% in next Y periods, 0 otherwise
                simulated_target = (historical_features['close'].pct_change(5).shift(-5) > 0.005).astype(int)
            elif regime == "BEAR_TREND":
                simulated_target = (historical_features['close'].pct_change(5).shift(-5) < -0.005).astype(int)
            elif regime == "SIDEWAYS_RANGE":
                simulated_target = ((historical_features['close'].pct_change(5).shift(-5).abs() < 0.002) & 
                                    (historical_features['close'].pct_change(5).shift(-5).abs() > 0.0001)).astype(int) # Small, non-zero change
            elif regime == "SR_ZONE_ACTION":
                simulated_target = (historical_features['close'].pct_change(2).shift(-2).abs() > 0.001).astype(int) # Breakout or rejection
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
            self.logger.warning(f"Orchestrator: Ensemble for {regime} is not trained. Attempting to train (one-off for demo)...")
            # In a real system, training would happen during setup.
            # This is a fallback for demo purposes if training was skipped.
            # We'll need some dummy historical data for this one-off train.
            # For a proper system, this would indicate a setup issue.
            
            # For now, we'll just return HOLD if not trained.
            self.logger.error(f"Ensemble for {regime} is not trained. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}


        self.logger.info(f"Orchestrator: Getting prediction from {regime} ensemble.")
        return ensemble_instance.get_prediction(current_features, klines_df, agg_trades_df, order_book_data, current_price)

