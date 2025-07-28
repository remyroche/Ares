# src/analyst/analyst.py
import pandas as pd
import numpy as np
import os
import sys
import datetime
import json # For serializing dicts
import asyncio # For async operations

# Assume these are available in the same package or through sys.path
from config import CONFIG, KLINES_FILENAME, AGG_TRADES_FILENAME, FUTURES_FILENAME
from sr_analyzer import SRLevelAnalyzer # Assuming sr_analyzer.py is at the root or accessible
from .feature_engineering import FeatureEngineeringEngine
from .regime_classifier import MarketRegimeClassifier
from .predictive_ensembles import RegimePredictiveEnsembles
from .liquidation_risk_model import ProbabilisticLiquidationRiskModel
from .market_health_analyzer import GeneralMarketAnalystModule
from .specialized_models import SpecializedModels
from .data_utils import load_klines_data, load_agg_trades_data, load_futures_data, simulate_order_book_data

from database.firestore_manager import FirestoreManager # New import
from utils.logger import system_logger # Centralized logger

class Analyst:
    """
    The core Analyst module, responsible for real-time intelligence and predictions.
    It orchestrates various sub-modules to provide comprehensive market insights.
    """
    def __init__(self, config=CONFIG, firestore_manager: FirestoreManager = None):
        self.config = config
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('Analyst') # Child logger for Analyst

        self.sr_analyzer = SRLevelAnalyzer(config["sr_analyzer"])
        self.feature_engine = FeatureEngineeringEngine(config)
        self.regime_classifier = MarketRegimeClassifier(config, self.sr_analyzer)
        self.predictive_ensembles = RegimePredictiveEnsembles(config)
        self.liquidation_risk_model = ProbabilisticLiquidationRiskModel(config)
        self.market_health_analyzer = GeneralMarketAnalystModule(config)
        self.specialized_models = SpecializedModels(config)
        
        # Internal state for historical data (for training/feature generation)
        self._historical_klines = None
        self._historical_agg_trades = None
        self._historical_futures = None
        self._sr_levels = [] # Store identified S/R levels

        self.model_storage_path = self.config['analyst'].get("model_storage_path", "models/analyst/")
        os.makedirs(self.model_storage_path, exist_ok=True) # Ensure model storage path exists

    async def load_and_prepare_historical_data(self):
        """
        Loads historical data and performs initial preparation for Analyst's internal use.
        This should be called once at startup or periodically for retraining.
        """
        self.logger.info("Analyst: Loading and preparing historical data...")
        self._historical_klines = load_klines_data(KLINES_FILENAME)
        self._historical_agg_trades = load_agg_trades_data(AGG_TRADES_FILENAME)
        self._historical_futures = load_futures_data(FUTURES_FILENAME)

        if self._historical_klines.empty:
            self.logger.error("Analyst: Failed to load historical k-lines data. Cannot proceed with training.")
            return False

        # Identify S/R levels from historical data (e.g., daily aggregation)
        daily_df = self._historical_klines.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if not daily_df.empty:
            daily_df.rename(columns={'open': 'Open', 'high': 'low', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True) # Fix column names for SR Analyzer
            self._sr_levels = self.sr_analyzer.analyze(daily_df)
            self.logger.info(f"Analyst: Identified {len(self._sr_levels)} S/R levels from historical data.")
        else:
            self.logger.warning("Analyst: No daily data for S/R analysis.")
            self._sr_levels = []
        
        # Train autoencoder and classifier on historical data
        historical_features_for_training = self.feature_engine.generate_all_features(
            self._historical_klines, self._historical_agg_trades, self._historical_futures, self._sr_levels
        )
        if not historical_features_for_training.empty:
            # Train and save/load models
            # Autoencoder training uses a subset of features
            ae_train_df = historical_features_for_training[['close', 'volume']].dropna()
            if not ae_train_df.empty:
                self.feature_engine.train_autoencoder(ae_train_df)
                await self.save_model_metadata("Analyst_FeatureEngineer_Autoencoder", "v1.0", {"loss": 0.01}, f"{self.model_storage_path}autoencoder_model.h5")
            else:
                self.logger.warning("Analyst: Not enough data for autoencoder training.")

            # --- Demonstrate GBM Feature Selection ---
            # Create a simulated target for demonstration purposes:
            # E.g., 1 if price increased by more than 0.5% in the next 5 periods, 0 otherwise.
            target_lookahead = 5
            price_change = historical_features_for_training['close'].pct_change(target_lookahead).shift(-target_lookahead)
            # Binary target: 1 for significant up move, 0 otherwise (can be refined)
            simulated_target = (price_change > 0.005).astype(int)
            simulated_target.name = 'price_up_target'
            
            # Align features and target, drop NaNs
            features_and_target = historical_features_for_training.join(simulated_target).dropna()
            
            if not features_and_target.empty and 'price_up_target' in features_and_target.columns:
                X_for_gbm = features_and_target.drop(columns=['price_up_target'])
                y_for_gbm = features_and_target['price_up_target']
                
                self.logger.info("Analyst: Running GBM Feature Selection demo...")
                selected_features = self.feature_engine.select_features_with_gbm(X_for_gbm, y_for_gbm)
                self.logger.info(f"Analyst: GBM Selected Features: {selected_features}")
                # In a real system, you'd now use these selected_features for subsequent model training.
            else:
                self.logger.warning("Analyst: Not enough data for GBM feature selection demo after target creation.")
            # --- End GBM Feature Selection Demo ---

            self.regime_classifier.train_classifier(historical_features_for_training, self._historical_klines)
            await self.save_model_metadata("Analyst_RegimeClassifier_LGBM", "v1.0", {"accuracy": 0.9}, f"{self.model_storage_path}regime_classifier.pkl")
            
            self.predictive_ensembles.train_all_ensembles(historical_features_for_training, {}) # {} for historical_targets
            # Add more save_model_metadata calls for other ensemble models here
        else:
            self.logger.warning("Analyst: Not enough historical features for model training.")

        self.logger.info("Analyst: Historical data preparation complete. (Note: Full model training is placeholder).")
        return True

    def load_and_prepare_historical_data_for_sim(self):
        """
        Helper for simulation mode to get historical klines for feature generation.
        """
        if self._historical_klines is None or self._historical_klines.empty:
            # Fallback to loading from file if not already loaded
            self.logger.info("Analyst: Loading historical klines for simulation from file.")
            return load_klines_data(KLINES_FILENAME)
        return self._historical_klines

    async def save_model_metadata(self, model_name: str, version: str, performance_metrics: dict, file_path_reference: str):
        """
        Saves model metadata to Firestore and CSV.
        """
        metadata = {
            "model_name": model_name,
            "version": version,
            "training_date": datetime.datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "file_path_reference": file_path_reference,
            "config_snapshot": json.dumps(self.config.get("analyst", {})) # Snapshot relevant config
        }
        
        # Save to Firestore
        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            await self.firestore_manager.set_document(
                self.config['firestore']['model_metadata_collection'],
                doc_id=f"{model_name}_{version}",
                data=metadata,
                is_public=True # Model metadata can be public
            )
            self.logger.info(f"Model metadata for {model_name} saved to Firestore.")

        # Export to CSV
        try:
            with open(self.config['supervisor']['model_metadata_csv'], 'a') as f:
                # Ensure the order matches the CSV header
                f.write(f"{metadata['model_name']},{metadata['version']},{metadata['training_date']},"
                        f"\"{json.dumps(metadata['performance_metrics'])}\"," # Quote JSON to handle commas
                        f"{metadata['file_path_reference']},"
                        f"\"{metadata['config_snapshot']}\"\n") # Quote JSON
            self.logger.info(f"Model metadata for {model_name} exported to CSV.")
        except Exception as e:
            self.logger.error(f"Error exporting model metadata for {model_name} to CSV: {e}")

    def get_intelligence(self, current_klines_data: pd.DataFrame, current_agg_trades_data: pd.DataFrame, 
                         current_futures_data: pd.DataFrame, current_order_book_data: dict,
                         current_position_notional: float = 0.0, current_liquidation_price: float = 0.0):
        """
        Provides real-time intelligence by orchestrating calls to all sub-modules.
        :param current_klines_data: DataFrame of recent k-line data (e.g., last N candles).
        :param current_agg_trades_data: DataFrame of recent aggregated trades.
        :param current_futures_data: DataFrame of recent futures data (funding rates, open interest).
        :param current_order_book_data: Dictionary of current order book (bids, asks).
        :param current_position_notional: Current open position notional value (USD).
        :param current_liquidation_price: Current liquidation price for the open position.
        :return: A dictionary containing all aggregated intelligence.
        """
        self.logger.info("\n--- Analyst: Generating Real-Time Intelligence ---")

        intelligence = {}

        # 1. Feature Engineering
        # Combine current and historical data for feature generation
        # Ensure _historical_klines, _historical_agg_trades, _historical_futures are not None
        hist_klines = self._historical_klines.tail(2000) if self._historical_klines is not None else pd.DataFrame()
        hist_agg_trades = self._historical_agg_trades.tail(5000) if self._historical_agg_trades is not None else pd.DataFrame()
        hist_futures = self._historical_futures.tail(500) if self._historical_futures is not None else pd.DataFrame()

        combined_klines = pd.concat([hist_klines, current_klines_data]).drop_duplicates().sort_index().last('2D')
        combined_agg_trades = pd.concat([hist_agg_trades, current_agg_trades_data]).drop_duplicates().sort_index().last('2D')
        combined_futures = pd.concat([hist_futures, current_futures_data]).drop_duplicates().sort_index().last('2D')

        # Generate features using the feature engineering engine
        current_features = self.feature_engine.generate_all_features(
            combined_klines, combined_agg_trades, combined_futures, self._sr_levels
        )
        if not current_features.empty:
            intelligence['features'] = current_features.tail(1).to_dict('records')[0]
        else:
            self.logger.warning("Analyst: No features generated. Returning empty intelligence.")
            return {} # Return empty if no features

        # 2. Market Regime Classification
        regime, trend_strength, adx_value = self.regime_classifier.predict_regime(
            current_features, combined_klines, self._sr_levels
        )
        intelligence['market_regime'] = regime
        intelligence['trend_strength_score'] = trend_strength
        intelligence['adx_value'] = adx_value

        # 3. Regime-Specific Predictive Ensembles
        ensemble_output = self.predictive_ensembles.get_ensemble_prediction(
            regime, current_features, combined_klines, combined_agg_trades, current_order_book_data, 
            current_klines_data['close'].iloc[-1]
        )
        intelligence['directional_prediction'] = ensemble_output['prediction']
        intelligence['directional_confidence_score'] = ensemble_output['confidence']

        # 4. Probabilistic Liquidation Risk Model (LSS)
        lss, lss_reasons = self.liquidation_risk_model.calculate_lss(
            current_klines_data['close'].iloc[-1], 
            current_position_notional, 
            current_liquidation_price, 
            combined_klines, 
            current_order_book_data
        )
        intelligence['liquidation_safety_score'] = lss
        intelligence['lss_reasons'] = lss_reasons

        # 5. General Market Analyst Module (Market Health Score)
        market_health = self.market_health_analyzer.get_market_health_score(combined_klines)
        intelligence['market_health_score'] = market_health

        # 6. Specialized Models
        sr_interaction_signal = self.specialized_models.get_sr_interaction_signal(
            current_klines_data['close'].iloc[-1], 
            self._sr_levels, 
            intelligence['features'].get('ATR', 0) # Use ATR from generated features
        )
        intelligence['sr_interaction_signal'] = sr_interaction_signal

        high_impact_candle_signal = self.specialized_models.get_high_impact_candle_signal(combined_klines)
        intelligence['high_impact_candle_signal'] = high_impact_candle_signal

        self.logger.info("--- Analyst: Intelligence Generation Complete ---")
        return intelligence
