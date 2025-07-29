import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time
import os # Import os for path manipulation

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings, CONFIG
from src.utils.state_manager import StateManager
from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.analyst.regime_classifier import MarketRegimeClassifier
from src.analyst.sr_analyzer import SRLevelAnalyzer
from src.analyst.technical_analyzer import TechnicalAnalyzer
from src.analyst.market_health_analyzer import GeneralMarketAnalystModule
from src.analyst.liquidation_risk_model import ProbabilisticLiquidationRiskModel
from src.analyst.predictive_ensembles.ensemble_orchestrator import RegimePredictiveEnsembles


class Analyst:
    """
    The Analyst processes real-time and historical market data to generate actionable intelligence.
    It operates in an event-driven manner, triggering its analysis pipeline upon the
    closure of a new candlestick.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager):
        """
        Initializes the Analyst.

        Args:
            exchange_client: An instance of the BinanceExchange client to access data.
            state_manager: An instance of the StateManager to save analysis results.
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Analyst')
        self.trade_symbol = settings.trade_symbol
        self.timeframe = settings.timeframe
        self.last_kline_open_time = None

        # Internal storage for historical data, used during training pipeline
        self._historical_klines = pd.DataFrame()
        self._historical_agg_trades = pd.DataFrame()
        self._historical_futures = pd.DataFrame()

        # Checkpoint directory for models
        self.model_checkpoint_dir = os.path.join(CONFIG['CHECKPOINT_DIR'], "analyst_models")
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        # Instantiate all sub-analyzer components
        self.feature_engineering = FeatureEngineeringEngine(settings.CONFIG)
        self.sr_analyzer = SRLevelAnalyzer(settings.CONFIG["analyst"]["sr_analyzer"])
        self.regime_classifier = MarketRegimeClassifier(settings.CONFIG, self.sr_analyzer)
        self.technical_analyzer = TechnicalAnalyzer(settings.CONFIG["analyst"]["technical_analyzer"])
        self.market_health_analyzer = GeneralMarketAnalystModule(settings.CONFIG)
        self.liquidation_risk_model = ProbabilisticLiquidationRiskModel(settings.CONFIG)
        self.predictive_ensembles = RegimePredictiveEnsembles(settings.CONFIG)

        self.logger.info("Analyst and all sub-analyzers initialized.")

    async def start(self):
        """
        Starts the main analysis loop.
        The loop waits for a new kline to close before running the full analysis pipeline.
        """
        self.logger.info("Analyst started. Waiting for new kline events...")
        while True:
            try:
                # Wait for the next kline to be available from the WebSocket stream
                latest_kline = self.exchange.kline_data
                if latest_kline and latest_kline.get('is_closed'):
                    # Check if this is a new kline we haven't processed yet
                    if latest_kline['open_time'] != self.last_kline_open_time:
                        self.last_kline_open_time = latest_kline['open_time']
                        self.logger.info(f"New kline closed at {pd.to_datetime(self.last_kline_open_time, unit='ms')}. Triggering analysis.")
                        
                        # Run the analysis pipeline
                        await self.run_analysis_pipeline()
                
                # Sleep for a short duration to prevent a tight loop
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Analyst task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Analyst loop: {e}", exc_info=True)
                # Wait before retrying to prevent rapid failure loops
                await asyncio.sleep(10)

    async def load_and_prepare_historical_data(
        self, 
        historical_klines: Optional[pd.DataFrame] = None,
        historical_agg_trades: Optional[pd.DataFrame] = None,
        historical_futures: Optional[pd.DataFrame] = None,
        fold_id: Optional[Any] = None # Added fold_id for checkpointing
    ) -> bool:
        """
        Loads historical data from the exchange or uses provided dataframes (for backtesting/training).
        Then, it prepares this data by generating features and training/loading models.
        Includes robust error handling for data loading and processing.
        """
        self.logger.info(f"Loading and preparing historical data for Analyst components (Fold ID: {fold_id})...")

        if historical_klines is not None and historical_agg_trades is not None and historical_futures is not None:
            # Use provided dataframes (e.g., from TrainingPipeline)
            self._historical_klines = historical_klines
            self._historical_agg_trades = historical_agg_trades
            self._historical_futures = historical_futures
            self.logger.info("Using historical data provided externally.")
        else:
            # Fetch data from exchange (for live/paper trading startup)
            self.logger.info("Fetching historical data from exchange for live/paper trading.")
            try:
                lookback_days = settings.CONFIG.get("analyst_historical_data_lookback_days", 30)
                end_time_ms = self.exchange._get_timestamp()
                start_time_ms = end_time_ms - int(timedelta(days=lookback_days).total_seconds() * 1000)

                klines_raw = await self.exchange.get_klines(self.trade_symbol, self.timeframe, limit=5000)
                if klines_raw:
                    self._historical_klines = pd.DataFrame(klines_raw)
                    self._historical_klines['open_time'] = pd.to_datetime(self._historical_klines['open_time'], unit='ms')
                    self._historical_klines.set_index('open_time', inplace=True)
                    klines_column_names = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                    self._historical_klines.columns = klines_column_names[:len(self._historical_klines.columns)] # Ensure column count matches data
                    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                        if col in self._historical_klines.columns:
                            self._historical_klines[col] = pd.to_numeric(self._historical_klines[col], errors='coerce')
                else:
                    self.logger.warning("Failed to fetch historical klines from exchange.")
                    self._historical_klines = pd.DataFrame()

                agg_trades_raw = await self.exchange.get_historical_agg_trades(self.trade_symbol, start_time_ms, end_time_ms)
                if agg_trades_raw:
                    self._historical_agg_trades = pd.DataFrame(agg_trades_raw)
                    self._historical_agg_trades['timestamp'] = pd.to_datetime(self._historical_agg_trades['T'], unit='ms')
                    self._historical_agg_trades.set_index('timestamp', inplace=True)
                    self._historical_agg_trades.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
                    for col in ['price', 'quantity']:
                        agg_trades_df[col] = pd.to_numeric(agg_trades_df[col], errors='coerce') # Fix: use agg_trades_df here
                        self._historical_agg_trades[col] = pd.to_numeric(self._historical_agg_trades[col], errors='coerce')
                else:
                    self.logger.warning("Failed to fetch historical aggregated trades from exchange.")
                    self._historical_agg_trades = pd.DataFrame()

                futures_raw = await self.exchange.get_historical_futures_data(self.trade_symbol, start_time_ms, end_time_ms)
                if futures_raw:
                    funding_df = pd.DataFrame(futures_raw.get('funding_rates', []))
                    if not funding_df.empty:
                        funding_df['timestamp'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
                        funding_df.set_index('timestamp', inplace=True)
                        funding_df['fundingRate'] = pd.to_numeric(funding_df['fundingRate'], errors='coerce')
                        funding_df = funding_df[['fundingRate']]
                    
                    oi_df = pd.DataFrame(futures_raw.get('open_interest', []))
                    if not oi_df.empty:
                        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                        oi_df.set_index('timestamp', inplace=True)
                        oi_df['openInterest'] = pd.to_numeric(oi_df['sumOpenInterest'], errors='coerce')
                        oi_df = oi_df[['openInterest']]

                    self._historical_futures = pd.concat([funding_df, oi_df], axis=1).ffill().bfill()
                    self._historical_futures.dropna(inplace=True)
                else:
                    self.logger.warning("Failed to fetch historical futures data from exchange.")
                    self._historical_futures = pd.DataFrame()

            except Exception as e:
                self.logger.error(f"Failed to fetch historical data from exchange: {e}", exc_info=True)
                return False

        if self._historical_klines.empty:
            self.logger.error("No historical klines data available for preparation.")
            return False

        daily_df_for_sr = self._historical_klines.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        daily_df_for_sr.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        sr_levels = self.sr_analyzer.analyze(daily_df_for_sr)
        self.state_manager.set_state("sr_levels", sr_levels)

        historical_features_df = self.feature_engineering.generate_all_features(
            self._historical_klines, self._historical_agg_trades, self._historical_futures, sr_levels
        )
        if historical_features_df.empty:
            self.logger.error("Failed to generate historical features.")
            return False

        # --- Train/Load Market Regime Classifier ---
        regime_classifier_model_path = os.path.join(self.model_checkpoint_dir, f"{CONFIG['REGIME_CLASSIFIER_MODEL_PREFIX']}{fold_id}.joblib") if fold_id is not None else None

        if regime_classifier_model_path and os.path.exists(regime_classifier_model_path):
            self.logger.info(f"Loading Market Regime Classifier for fold {fold_id} from {regime_classifier_model_path}...")
            if not self.regime_classifier.load_model(model_path=regime_classifier_model_path):
                self.logger.warning(f"Failed to load classifier for fold {fold_id}. Retraining.")
                self.regime_classifier.train_classifier(historical_features_df.copy(), self._historical_klines.copy(), model_path=regime_classifier_model_path)
        else:
            self.logger.info(f"Training Market Regime Classifier for fold {fold_id}...")
            self.regime_classifier.train_classifier(historical_features_df.copy(), self._historical_klines.copy(), model_path=regime_classifier_model_path)


        # --- Train/Load Predictive Ensembles ---
        ensemble_model_path_prefix = os.path.join(self.model_checkpoint_dir, f"{CONFIG['ENSEMBLE_MODEL_PREFIX']}{fold_id}_") if fold_id is not None else None
        
        self.logger.info(f"Training/Loading Predictive Ensembles for fold {fold_id}...")
        self.predictive_ensembles.train_all_models(
            asset=self.trade_symbol, 
            prepared_data=historical_features_df.copy(),
            model_path_prefix=ensemble_model_path_prefix
        )

        self.logger.info(f"Historical data preparation and model training/loading complete for fold {fold_id}.")
        return True

    async def run_analysis_pipeline(self):
        """
        Executes the full sequence of analysis tasks.
        """
        self.logger.info("--- Starting Analysis Pipeline ---")
        # Initialize analyst_intelligence with default values to ensure all keys exist even if parts fail
        analyst_intelligence = {
            "timestamp": int(time.time() * 1000),
            "market_regime": "UNKNOWN",
            "trend_strength": 0.0,
            "adx": 0.0,
            "support_resistance": [],
            "technical_signals": {}, # Full technical signals dict
            "market_health_score": 50.0,
            "liquidation_risk_score": 100.0,
            "liquidation_risk_reasons": "N/A",
            "ensemble_prediction": "HOLD",
            "ensemble_confidence": 0.0,
            "directional_confidence_score": 0.0,
            "base_model_predictions": {},
            "ensemble_weights": {},
            "volume_delta": 0.0,
            "current_price": 0.0, # Kept
            "market_health_volatility_component": 50.0, # Kept
            # Other raw features from df_features (initialized to None/0.0)
            'bb_bandwidth': None, 'stoch_k': None, 'CMF': None, 
            'autoencoder_reconstruction_error': None, 'oi_roc': None, 
            'fundingRate': None, 'log_returns': None, 'volatility_20': None, 
            'skewness_20': None, 'kurtosis_20': None, 'avg_body_size_20': None, 
            'avg_range_size_20': None, 'volume_change': None, 'OBV': None, 
            'price_vs_vwap': None, 'ATR': None,
            'current_open_interest': None, 'current_funding_rate': None,
            'total_bid_liquidity': 0.0, 'total_ask_liquidity': 0.0, 'order_book_imbalance': 0.0
        }

        try:
            klines_raw_data = await self.exchange.get_klines(self.trade_symbol, self.timeframe, limit=500)
            if not klines_raw_data:
                self.logger.error("Could not fetch latest klines. Aborting analysis cycle.")
                self.state_manager.set_state("analyst_intelligence", analyst_intelligence) # Save default intel
                return

            klines = pd.DataFrame(klines_raw_data)
            klines['open_time'] = pd.to_datetime(klines['open_time'], unit='ms')
            klines.set_index('open_time', inplace=True)
            klines_column_names = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            klines.columns = klines_column_names[:len(klines.columns)]
            for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                if col in klines.columns:
                    klines[col] = pd.to_numeric(klines[col], errors='coerce')
            
            if klines.empty:
                self.logger.error("Formatted klines DataFrame is empty. Aborting analysis cycle.")
                self.state_manager.set_state("analyst_intelligence", analyst_intelligence)
                return

            # Initialize bids and asks here, before the try-except for order_book processing
            bids = {}
            asks = {}
            order_book = self.exchange.order_book # Get the raw order_book from exchange
            if order_book:
                bids = order_book.get('bids', {})
                asks = order_book.get('asks', {})


            recent_trades = self.exchange.recent_trades
            
            agg_trades_df = pd.DataFrame(recent_trades)
            if not agg_trades_df.empty:
                agg_trades_df['timestamp'] = pd.to_datetime(agg_trades_df['T'], unit='ms')
                agg_trades_df.set_index('timestamp', inplace=True)
                agg_trades_df.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
                for col in ['price', 'quantity']:
                    agg_trades_df[col] = pd.to_numeric(agg_trades_df[col], errors='coerce')
            else:
                agg_trades_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])

            futures_df = self._historical_futures.tail(100) # Use a recent window of historical futures data
            if futures_df.empty:
                self.logger.warning("Historical futures data is empty. Some features may be missing.")


            # Ensure ATR is calculated on klines before passing to feature_engineering
            klines.ta.atr(length=settings.CONFIG['BEST_PARAMS']['atr_period'], append=True, col_names=('ATR'))
            if klines['ATR'].iloc[-1] is None or pd.isna(klines['ATR'].iloc[-1]):
                self.logger.warning("ATR calculation resulted in NaN for the latest kline. This might affect downstream models.")
                # Fallback or handle appropriately, e.g., fill with average or default

            df_features = self.feature_engineering.generate_all_features(klines.copy(), agg_trades_df.copy(), futures_df.copy(), self.state_manager.get_state("sr_levels", []))
            
            if df_features.empty:
                self.logger.error("Feature generation resulted in empty DataFrame. Aborting analysis cycle.")
                self.state_manager.set_state("analyst_intelligence", analyst_intelligence)
                return

            # Ensure ATR from klines is also in df_features if not already there
            if 'ATR' not in df_features.columns and 'ATR' in klines.columns:
                df_features['ATR'] = klines['ATR']

            current_price = klines['close'].iloc[-1]
            analyst_intelligence['current_price'] = float(current_price) # Ensure float
            sr_levels_from_state = self.state_manager.get_state("sr_levels", [])

            # --- Market Regime Classification ---
            try:
                market_regime, trend_strength, adx = self.regime_classifier.predict_regime(df_features.copy(), klines.copy(), sr_levels_from_state)
                analyst_intelligence['market_regime'] = market_regime
                analyst_intelligence['trend_strength'] = trend_strength
                analyst_intelligence['adx'] = adx
                analyst_intelligence['support_resistance'] = sr_levels_from_state # Use pre-calculated SR levels
            except Exception as e:
                self.logger.error(f"Error during market regime classification: {e}. Using default values.", exc_info=True)

            # --- Technical Analysis ---
            try:
                technical_signals = self.technical_analyzer.analyze(klines.copy())
                analyst_intelligence['technical_signals'] = technical_signals
            except Exception as e:
                self.logger.error(f"Error during technical analysis: {e}. Using empty technical signals.", exc_info=True)

            # --- Market Health Analysis ---
            try:
                market_health_score = self.market_health_analyzer.get_market_health_score(klines.copy())
                analyst_intelligence['market_health_score'] = market_health_score
                analyst_intelligence['market_health_volatility_component'] = market_health_score # As requested
            except Exception as e:
                self.logger.error(f"Error during market health analysis: {e}. Using default score.", exc_info=True)

            # --- Liquidation Risk Model ---
            try:
                current_position_risk = await self.exchange.get_position_risk(self.trade_symbol)
                current_position_notional = 0.0
                current_liquidation_price = 0.0
                if current_position_risk and len(current_position_risk) > 0:
                    pos = current_position_risk[0]
                    current_position_notional = float(pos.get('positionAmt', 0)) * current_price
                    current_liquidation_price = float(pos.get('liquidationPrice', 0))

                liquidation_risk_score, lss_reasons = self.liquidation_risk_model.calculate_lss(
                    current_price, current_position_notional, current_liquidation_price, klines.copy(), order_book
                )
                analyst_intelligence['liquidation_risk_score'] = liquidation_risk_score
                analyst_intelligence['liquidation_risk_reasons'] = lss_reasons
            except Exception as e:
                self.logger.error(f"Error during liquidation risk calculation: {e}. Using default scores.", exc_info=True)

            # --- Predictive Ensembles ---
            try:
                ensemble_prediction_result = self.predictive_ensembles.get_all_predictions(
                    asset=self.trade_symbol, 
                    current_features=df_features.copy(),
                    klines_df=klines.copy(),
                    agg_trades_df=agg_trades_df.copy(),
                    order_book_data=order_book,
                    current_price=current_price
                )
                analyst_intelligence['ensemble_prediction'] = ensemble_prediction_result["prediction"]
                analyst_intelligence['ensemble_confidence'] = ensemble_prediction_result["confidence"]
                analyst_intelligence['directional_confidence_score'] = ensemble_prediction_result["confidence"]
                analyst_intelligence['base_model_predictions'] = ensemble_prediction_result.get("base_predictions", {})
                analyst_intelligence['ensemble_weights'] = ensemble_prediction_result.get("ensemble_weights", {})
            except Exception as e:
                self.logger.error(f"Error during predictive ensembles prediction: {e}. Using default values.", exc_info=True)

            # --- Add volume_delta from df_features directly to analyst_intelligence ---
            if 'volume_delta' in df_features.columns and not df_features.empty:
                analyst_intelligence['volume_delta'] = float(df_features['volume_delta'].iloc[-1])
            else:
                analyst_intelligence['volume_delta'] = 0.0 # Default if not found

            # --- Add current open interest and funding rate values from futures_df ---
            if 'openInterest' in futures_df.columns and not futures_df.empty:
                analyst_intelligence['current_open_interest'] = float(futures_df['openInterest'].iloc[-1])
            else:
                analyst_intelligence['current_open_interest'] = None

            if 'fundingRate' in futures_df.columns and not futures_df.empty:
                analyst_intelligence['current_funding_rate'] = float(futures_df['fundingRate'].iloc[-1])
            else:
                analyst_intelligence['current_funding_rate'] = None

            # --- Add current order book liquidity/imbalance metrics ---
            try:
                total_bid_qty = sum(bids.values()) if bids else 0.0
                total_ask_qty = sum(asks.values()) if asks else 0.0
                
                analyst_intelligence['total_bid_liquidity'] = total_bid_qty
                analyst_intelligence['total_ask_liquidity'] = total_ask_qty
                if total_bid_qty + total_ask_qty > 0:
                    analyst_intelligence['order_book_imbalance'] = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
                else:
                    analyst_intelligence['order_book_imbalance'] = 0.0
            except Exception as e:
                self.logger.error(f"Error calculating order book metrics: {e}. Using default values.", exc_info=True)


            # Finally, save the consolidated intelligence
            self.state_manager.set_state("analyst_intelligence", analyst_intelligence)
            self.logger.info(f"Analysis complete. Intelligence package updated. Regime: {analyst_intelligence['market_regime']}, Ensemble Pred: {analyst_intelligence['ensemble_prediction']}, Conf: {analyst_intelligence['ensemble_confidence']:.2f}")
            self.logger.debug(f"Analyst Intelligence: {analyst_intelligence}")

        except Exception as e:
            self.logger.error(f"A critical error occurred during analysis pipeline: {e}", exc_info=True)
            # Ensure a default/empty intelligence is saved even on critical failure
            self.state_manager.set_state("analyst_intelligence", analyst_intelligence) 

    def _calculate_confidence(self, signals: Dict, regime: str) -> float:
        return signals.get('ensemble_confidence', 0.5)
