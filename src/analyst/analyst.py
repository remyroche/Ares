# src/analyst/analyst.py
import pandas as pd
import numpy as np
import os
import sys

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

class Analyst:
    """
    The core Analyst module, responsible for real-time intelligence and predictions.
    It orchestrates various sub-modules to provide comprehensive market insights.
    """
    def __init__(self, config=CONFIG):
        self.config = config
        self.sr_analyzer = SRLevelAnalyzer(config["sr_analyzer"]) # Initialize S/R Analyzer
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

    def load_and_prepare_historical_data(self):
        """
        Loads historical data and performs initial preparation for Analyst's internal use.
        This should be called once at startup or periodically for retraining.
        """
        print("Analyst: Loading and preparing historical data...")
        self._historical_klines = load_klines_data(KLINES_FILENAME)
        self._historical_agg_trades = load_agg_trades_data(AGG_TRADES_FILENAME)
        self._historical_futures = load_futures_data(FUTURES_FILENAME)

        if self._historical_klines.empty:
            print("Analyst: Failed to load historical k-lines data. Cannot proceed with training.")
            return False

        # Identify S/R levels from historical data (e.g., daily aggregation)
        daily_df = self._historical_klines.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if not daily_df.empty:
            # Ensure column names match what SRLevelAnalyzer expects
            daily_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            self._sr_levels = self.sr_analyzer.analyze(daily_df)
            print(f"Analyst: Identified {len(self._sr_levels)} S/R levels from historical data.")
        else:
            print("Analyst: No daily data for S/R analysis.")
            self._sr_levels = []
        
        # Train autoencoder and classifier on historical data
        # Feature Engineering: Train autoencoder
        # The autoencoder needs to be trained on a representative dataset.
        # For simplicity, we'll train it on a subset of klines features.
        # In a real scenario, you'd prepare a dedicated dataset for AE training.
        if not self._historical_klines.empty:
            # Create a simplified feature set for autoencoder training
            ae_train_df = self._historical_klines[['close', 'volume']].copy().dropna()
            if not ae_train_df.empty:
                self.feature_engine.train_autoencoder(ae_train_df)
            else:
                print("Analyst: Not enough data for autoencoder training.")

        # Market Regime Classifier: Train classifier
        # This requires historical features and labels. For now, we'll use a placeholder.
        # In a real scenario, `historical_features` would be generated from `_historical_klines`, etc.
        # and `historical_targets` would be derived from these features (e.g., via clustering).
        historical_features_for_training = self.feature_engine.generate_all_features(
            self._historical_klines, self._historical_agg_trades, self._historical_futures, self._sr_levels
        )
        if not historical_features_for_training.empty:
            self.regime_classifier.train_classifier(historical_features_for_training, self._historical_klines)
        else:
            print("Analyst: Not enough historical features for regime classifier training.")

        # Regime Predictive Ensembles: Train all ensembles
        # This also requires historical features and targets for each regime.
        # self.predictive_ensembles.train_all_ensembles(historical_features_for_training, historical_targets_for_ensembles)
        print("Analyst: Historical data preparation complete. (Note: Full model training is placeholder).")
        return True

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
        print("\n--- Analyst: Generating Real-Time Intelligence ---")

        intelligence = {}

        # 1. Feature Engineering
        # Combine current and historical data for feature generation
        # For real-time, current_klines_data should be recent history sufficient for indicators
        # and current_agg_trades_data/current_futures_data should be the latest snapshots.
        
        # Ensure sufficient data for feature generation by combining with historical if needed
        # (This is a simplified approach; a live system would manage a rolling window of data)
        combined_klines = pd.concat([self._historical_klines, current_klines_data]).drop_duplicates().sort_index().last('2D') # Last 2 days for features
        combined_agg_trades = pd.concat([self._historical_agg_trades, current_agg_trades_data]).drop_duplicates().sort_index().last('2D')
        combined_futures = pd.concat([self._historical_futures, current_futures_data]).drop_duplicates().sort_index().last('2D')

        # Generate features using the feature engineering engine
        current_features = self.feature_engine.generate_all_features(
            combined_klines, combined_agg_trades, combined_futures, self._sr_levels
        )
        intelligence['features'] = current_features.tail(1).to_dict('records')[0] # Get latest features

        # 2. Market Regime Classification
        regime, trend_strength, adx_value = self.regime_classifier.predict_regime(
            current_features, combined_klines, self._sr_levels
        )
        intelligence['market_regime'] = regime
        intelligence['trend_strength_score'] = trend_strength
        intelligence['adx_value'] = adx_value

        # 3. Regime-Specific Predictive Ensembles
        # Call the appropriate ensemble based on the predicted regime
        ensemble_output = self.predictive_ensembles.get_ensemble_prediction(
            regime, current_features, combined_klines, combined_agg_trades, current_order_book_data, 
            current_klines_data['close'].iloc[-1] # Pass current price
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
        # S/R Interaction Signal
        sr_interaction_signal = self.specialized_models.get_sr_interaction_signal(
            current_klines_data['close'].iloc[-1], 
            self._sr_levels, 
            current_features['ATR'].iloc[-1] if 'ATR' in current_features.columns else 0
        )
        intelligence['sr_interaction_signal'] = sr_interaction_signal

        # High-Impact Candle Signal
        high_impact_candle_signal = self.specialized_models.get_high_impact_candle_signal(combined_klines)
        intelligence['high_impact_candle_signal'] = high_impact_candle_signal

        print("--- Analyst: Intelligence Generation Complete ---")
        return intelligence

# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    print("Running Analyst Module Demonstration...")

    # Ensure data files exist for demonstration
    # You might need to run ares_data_downloader.py and ares_data_preparer.py first
    # For this demo, we'll try to load them, but they might be empty if not run.
    
    # Simulate historical data for training if files don't exist or are empty
    def create_dummy_data(filename, data_type='klines'):
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            print(f"Creating dummy {data_type} data for {filename}...")
            dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')
            if data_type == 'klines':
                dummy_df = pd.DataFrame({
                    'open': np.random.rand(500) * 100 + 2000,
                    'high': np.random.rand(500) * 100 + 2050,
                    'low': np.random.rand(500) * 100 + 1950,
                    'close': np.random.rand(500) * 100 + 2000,
                    'volume': np.random.rand(500) * 1000 + 100,
                    'close_time': dates.astype(np.int64) // 10**6,
                    'quote_asset_volume': np.random.rand(500) * 100000 + 10000,
                    'number_of_trades': np.random.randint(100, 1000, 500),
                    'taker_buy_base_asset_volume': np.random.rand(500) * 500 + 50,
                    'taker_buy_quote_asset_volume': np.random.rand(500) * 50000 + 5000,
                    'ignore': 0
                }, index=dates)
                dummy_df.index.name = 'open_time'
            elif data_type == 'agg_trades':
                dummy_df = pd.DataFrame({
                    'price': np.random.rand(500) * 100 + 2000,
                    'quantity': np.random.rand(500) * 10 + 1,
                    'is_buyer_maker': np.random.choice([True, False], 500),
                    'a': range(500), # Aggregate trade ID
                    'T': dates.astype(np.int64) // 10**6
                }, index=dates)
                dummy_df.index.name = 'timestamp'
            elif data_type == 'futures':
                dummy_df = pd.DataFrame({
                    'fundingRate': np.random.rand(500) * 0.001 - 0.0005, # -0.05% to +0.05%
                    'openInterest': np.random.rand(500) * 1000000 + 100000,
                    'fundingTime': dates.astype(np.int64) // 10**6,
                    'timestamp': dates.astype(np.int64) // 10**6
                }, index=dates)
                dummy_df.index.name = 'timestamp'
            
            dummy_df.to_csv(filename)
            print(f"Dummy data saved to {filename}")
            return True
        return False

    create_dummy_data(KLINES_FILENAME, 'klines')
    create_dummy_data(AGG_TRADES_FILENAME, 'agg_trades')
    create_dummy_data(FUTURES_FILENAME, 'futures')

    analyst = Analyst()

    # Step 1: Load and prepare historical data (including training models)
    if not analyst.load_and_prepare_historical_data():
        print("Analyst: Initial setup failed. Exiting demo.")
        sys.exit(1)

    # Step 2: Simulate real-time data for intelligence generation
    print("\nSimulating real-time data for intelligence generation...")
    current_klines_snapshot = load_klines_data(KLINES_FILENAME).tail(10) # Last 10 candles for current view
    current_agg_trades_snapshot = load_agg_trades_data(AGG_TRADES_FILENAME).tail(50) # Last 50 trades
    current_futures_snapshot = load_futures_data(FUTURES_FILENAME).tail(1) # Latest futures data

    if current_klines_snapshot.empty:
        print("Error: No k-lines data available for real-time simulation. Exiting.")
        sys.exit(1)

    current_price = current_klines_snapshot['close'].iloc[-1]
    current_order_book = simulate_order_book_data(current_price)
    
    # Simulate an open position for LSS calculation
    simulated_position_notional = 5000 # Example: $5000 long position
    simulated_liquidation_price = current_price * 0.95 # Example: 5% below current price

    # Step 3: Get intelligence
    intelligence_report = analyst.get_intelligence(
        current_klines_snapshot,
        current_agg_trades_snapshot,
        current_futures_snapshot,
        current_order_book,
        simulated_position_notional,
        simulated_liquidation_price
    )

    print("\n--- Final Analyst Intelligence Report ---")
    for key, value in intelligence_report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    print("\nAnalyst Module Demonstration Complete.")
