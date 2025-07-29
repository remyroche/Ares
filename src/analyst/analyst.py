import asyncio
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager
from src.analyst.feature_engineering import FeatureEngineeringEngine # Corrected import name
from src.analyst.regime_classifier import MarketRegimeClassifier
from src.analyst.sr_analyzer import SRLevelAnalyzer
from src.analyst.technical_analyzer import TechnicalAnalyzer
from src.analyst.market_health_analyzer import GeneralMarketAnalystModule
from src.analyst.liquidation_risk_model import ProbabilisticLiquidationRiskModel # Corrected import name
from src.analyst.predictive_ensembles.ensemble_orchestrator import RegimePredictiveEnsembles # Corrected import name


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

        # Instantiate all sub-analyzer components
        self.feature_engineering = FeatureEngineeringEngine(settings.CONFIG) # Pass CONFIG for sub-component init
        self.sr_analyzer = SRLevelAnalyzer(settings.CONFIG["analyst"]["sr_analyzer"]) # Pass relevant config
        self.regime_classifier = MarketRegimeClassifier(settings.CONFIG, self.sr_analyzer) # Pass CONFIG and sr_analyzer
        self.technical_analyzer = TechnicalAnalyzer(settings.CONFIG["analyst"]["technical_analyzer"]) # Pass relevant config
        self.market_health_analyzer = GeneralMarketAnalystModule(settings.CONFIG) # Pass CONFIG
        self.liquidation_risk_model = ProbabilisticLiquidationRiskModel(settings.CONFIG) # Pass CONFIG
        self.predictive_ensembles = RegimePredictiveEnsembles(settings.CONFIG) # Pass CONFIG

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
        historical_futures: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Loads historical data from the exchange or uses provided dataframes (for backtesting/training).
        Then, it prepares this data by generating features and training/loading models.
        """
        self.logger.info("Loading and preparing historical data for Analyst components...")

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
                # Define lookback period for historical data (e.g., 30 days for live analysis startup)
                lookback_days = settings.CONFIG.get("analyst_historical_data_lookback_days", 30)
                end_time_ms = self.exchange._get_timestamp()
                start_time_ms = end_time_ms - int(timedelta(days=lookback_days).total_seconds() * 1000)

                # Fetch klines
                klines_raw = await self.exchange.get_klines(self.trade_symbol, self.timeframe, limit=5000) # Max limit for klines
                if klines_raw:
                    self._historical_klines = pd.DataFrame(klines_raw)
                    self._historical_klines['open_time'] = pd.to_datetime(self._historical_klines['open_time'], unit='ms')
                    self._historical_klines.set_index('open_time', inplace=True)
                    self._historical_klines.columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                        if col in self._historical_klines.columns:
                            self._historical_klines[col] = pd.to_numeric(self._historical_klines[col], errors='coerce')
                else:
                    self.logger.warning("Failed to fetch historical klines from exchange.")
                    self._historical_klines = pd.DataFrame()

                # Fetch historical aggregated trades
                agg_trades_raw = await self.exchange.get_historical_agg_trades(self.trade_symbol, start_time_ms, end_time_ms)
                if agg_trades_raw:
                    self._historical_agg_trades = pd.DataFrame(agg_trades_raw)
                    self._historical_agg_trades['timestamp'] = pd.to_datetime(self._historical_agg_trades['T'], unit='ms')
                    self._historical_agg_trades.set_index('timestamp', inplace=True)
                    self._historical_agg_trades.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
                    for col in ['price', 'quantity']:
                        self._historical_agg_trades[col] = pd.to_numeric(self._historical_agg_trades[col], errors='coerce')
                else:
                    self.logger.warning("Failed to fetch historical aggregated trades from exchange.")
                    self._historical_agg_trades = pd.DataFrame()

                # Fetch historical futures data (funding rates, open interest)
                futures_raw = await self.exchange.get_historical_futures_data(self.trade_symbol, start_time_ms, end_time_ms)
                if futures_raw:
                    # Combine funding rates and open interest into a single DataFrame
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

                    # Merge funding rates and open interest, forward-filling to align
                    self._historical_futures = pd.concat([funding_df, oi_df], axis=1).ffill().bfill()
                    self._historical_futures.dropna(inplace=True) # Drop rows that are still all NaN
                else:
                    self.logger.warning("Failed to fetch historical futures data from exchange.")
                    self._historical_futures = pd.DataFrame()

            except Exception as e:
                self.logger.error(f"Failed to fetch historical data from exchange: {e}", exc_info=True)
                return False

        if self._historical_klines.empty:
            self.logger.error("No historical klines data available for preparation.")
            return False

        # Prepare data for S/R analysis (daily aggregated)
        daily_df_for_sr = self._historical_klines.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        daily_df_for_sr.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        sr_levels = self.sr_analyzer.analyze(daily_df_for_sr)
        self.state_manager.set_state("sr_levels", sr_levels) # Save SR levels to state manager

        # Generate all features for historical data
        historical_features_df = self.feature_engineering.generate_all_features(
            self._historical_klines, self._historical_agg_trades, self._historical_futures, sr_levels
        )
        if historical_features_df.empty:
            self.logger.error("Failed to generate historical features.")
            return False

        # Train/Load Market Regime Classifier
        self.regime_classifier.train_classifier(historical_features_df.copy(), self._historical_klines.copy())

        # Train/Load Predictive Ensembles
        # The ensembles expect historical_features and historical_targets (which are pseudo-labeled)
        # For training, we need to generate pseudo-targets for each regime.
        # This is simplified here; in a full training pipeline, targets would be derived.
        self.logger.info("Training/Loading Predictive Ensembles...")
        # For simplicity, passing empty dict for historical_targets for now, as actual targets are complex.
        # The ensemble orchestrator has its own simulated target generation for training.
        self.predictive_ensembles.train_all_ensembles(historical_features_df.copy(), {}) 

        self.logger.info("Historical data preparation and model training/loading complete.")
        return True

    async def run_analysis_pipeline(self):
        """
        Executes the full sequence of analysis tasks.
        """
        self.logger.info("--- Starting Analysis Pipeline ---")
        try:
            # 1. Fetch necessary historical data (for features that need lookback)
            # Use exchange's real-time data attributes for current context
            # get_latest_klines is a method that fetches historical klines, not real-time stream data.
            # The real-time kline data is in self.exchange.kline_data (a dict).
            # For analysis pipeline, we need a recent window of klines as a DataFrame.
            klines = self.exchange.get_latest_klines(num_klines=500) # This needs to be implemented in BinanceExchange to return a DataFrame

            # Corrected: If get_latest_klines is not implemented to return DataFrame,
            # we need to build it from raw kline data or fetch it here.
            # Assuming get_latest_klines is a helper that fetches and formats.
            # If not, this line would need to be:
            # klines_raw = await self.exchange.get_klines(self.trade_symbol, self.timeframe, limit=500)
            # klines = pd.DataFrame(klines_raw)
            # ... and then proper DataFrame formatting as done in load_and_prepare_historical_data

            if klines.empty:
                self.logger.error("Could not fetch latest klines. Aborting analysis cycle.")
                return

            # Ensure klines DataFrame has correct columns and types (assuming get_latest_klines returns raw data)
            # This block might be redundant if get_latest_klines already returns a formatted DataFrame.
            # It's safer to ensure consistency.
            if not isinstance(klines.index, pd.DatetimeIndex):
                klines['open_time'] = pd.to_datetime(klines['open_time'], unit='ms')
                klines.set_index('open_time', inplace=True)
                klines.columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
                for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                     if col in klines.columns: # Check if column exists before converting
                        klines[col] = pd.to_numeric(klines[col], errors='coerce')


            # 2. Get real-time data from the exchange client's state
            order_book = self.exchange.order_book
            recent_trades = self.exchange.recent_trades # This is a list of dicts
            
            # Convert recent_trades to DataFrame for feature engineering if needed
            agg_trades_df = pd.DataFrame(recent_trades)
            if not agg_trades_df.empty:
                agg_trades_df['timestamp'] = pd.to_datetime(agg_trades_df['T'], unit='ms') # 'T' is trade time in agg trades
                agg_trades_df.set_index('timestamp', inplace=True)
                agg_trades_df.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
                for col in ['price', 'quantity']:
                    agg_trades_df[col] = pd.to_numeric(agg_trades_df[col], errors='coerce')
            else:
                agg_trades_df = pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'is_buyer_maker'])

            # Futures data is not typically a real-time stream like klines/trades/orderbook.
            # It's usually fetched periodically or from historical data.
            # For live analysis, we might use the last available historical futures data or fetch a small amount.
            # For now, use the last known historical futures data or an empty DF.
            futures_df = self._historical_futures.tail(100) # Use a recent window of historical futures data

            # 3. Run all analysis components in sequence
            # Need to ensure ATR is calculated on klines before passing to feature_engineering
            klines.ta.atr(length=settings.CONFIG['BEST_PARAMS']['atr_period'], append=True, col_names=('ATR'))
            
            df_features = self.feature_engineering.generate_all_features(klines.copy(), agg_trades_df.copy(), futures_df.copy(), self.state_manager.get_state("sr_levels", []))
            
            if df_features.empty:
                self.logger.error("Feature generation resulted in empty DataFrame. Aborting analysis cycle.")
                return

            # Ensure 'ATR' is in df_features before passing to other modules if they rely on it
            if 'ATR' not in df_features.columns and 'ATR' in klines.columns:
                df_features['ATR'] = klines['ATR'] # Copy ATR from klines to features

            # Get latest price and order book for liquidation risk and technical analysis
            current_price = klines['close'].iloc[-1]
            
            # The SRLevelAnalyzer needs a daily DF for its analysis, not real-time klines.
            # It should ideally be run less frequently or its output cached.
            # For real-time analysis, we use the pre-calculated sr_levels from state_manager.
            sr_levels_from_state = self.state_manager.get_state("sr_levels", [])

            market_regime, trend_strength, adx = self.regime_classifier.predict_regime(df_features.copy(), klines.copy(), sr_levels_from_state)
            technical_signals = self.technical_analyzer.analyze(klines.copy()) # Technical analyzer uses klines
            market_health_score = self.market_health_analyzer.get_market_health_score(klines.copy()) # Market health uses klines

            # Liquidation risk needs current position info, which comes from exchange/tactician
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

            # Get ensemble prediction for the current regime
            ensemble_prediction_result = self.predictive_ensembles.get_ensemble_prediction(
                regime=market_regime,
                current_features=df_features.copy(), # Pass current features for ensemble
                klines_df=klines.copy(),
                agg_trades_df=agg_trades_df.copy(),
                order_book_data=order_book,
                current_price=current_price
            )

            # 4. Consolidate intelligence
            analyst_intelligence = {
                "timestamp": int(time.time() * 1000),
                "market_regime": market_regime,
                "trend_strength": trend_strength,
                "adx": adx,
                "support_resistance": sr_levels_from_state, # Use pre-calculated SR levels
                "technical_signals": technical_signals,
                "market_health_score": market_health_score,
                "liquidation_risk_score": liquidation_risk_score,
                "liquidation_risk_reasons": lss_reasons,
                "ensemble_prediction": ensemble_prediction_result["prediction"],
                "ensemble_confidence": ensemble_prediction_result["confidence"],
                "directional_confidence_score": ensemble_prediction_result["confidence"] # Use ensemble confidence as overall directional confidence
            }

            # 5. Save the final intelligence packet to the StateManager
            self.state_manager.set_state("analyst_intelligence", analyst_intelligence)
            self.logger.info(f"Analysis complete. Intelligence package updated. Regime: {market_regime}, Ensemble Pred: {ensemble_prediction_result['prediction']}, Conf: {ensemble_prediction_result['confidence']:.2f}")
            self.logger.debug(f"Analyst Intelligence: {analyst_intelligence}")

        except Exception as e:
            self.logger.error(f"Error during analysis pipeline: {e}", exc_info=True)

    def _calculate_confidence(self, signals: Dict, regime: str) -> float:
        """
        This method is now largely superseded by the ensemble's confidence.
        Keeping for legacy or minor adjustments if needed.
        """
        # The ensemble_prediction_result already provides a confidence score.
        # This method might be deprecated or repurposed.
        return signals.get('ensemble_confidence', 0.5) # Fallback to 0.5 if not found
