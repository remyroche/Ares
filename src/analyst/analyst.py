import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional, Tuple, List  # Ensure Optional is imported
from datetime import timedelta
import time
import os  # Import os for path manipulation

from src.utils.state_manager import StateManager
from exchange.binance import BinanceExchange
from src.utils.logger import system_logger
from src.config import settings, CONFIG
from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.analyst.regime_classifier import MarketRegimeClassifier
from src.analyst.sr_analyzer import SRLevelAnalyzer
from src.analyst.technical_analyzer import TechnicalAnalyzer
from src.analyst.market_health_analyzer import GeneralMarketAnalystModule
from src.analyst.liquidation_risk_model import ProbabilisticLiquidationRiskModel
from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.utils.error_handler import (
    handle_errors,
)


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
        self.logger = system_logger.getChild("Analyst")
        self.trade_symbol = settings.trade_symbol
        self.timeframe = settings.timeframe
        self.last_kline_open_time = None

        # self.config = config
        # self.event_bus = event_bus
        # ModelManager removed to avoid circular import
        self.state_manager = StateManager()

        # Internal storage for historical data, used during training pipeline
        self._historical_klines = pd.DataFrame()
        self._historical_agg_trades = pd.DataFrame()
        self._historical_futures = pd.DataFrame()

        # Checkpoint directory for models
        self.model_checkpoint_dir = os.path.join(
            CONFIG["CHECKPOINT_DIR"], "analyst_models"
        )
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        # Instantiate all sub-analyzer components
        self.feature_engineering = FeatureEngineeringEngine(CONFIG)
        self.sr_analyzer = SRLevelAnalyzer(CONFIG["analyst"]["sr_analyzer"])
        self.regime_classifier = MarketRegimeClassifier(
            CONFIG, self.sr_analyzer
        )
        self.technical_analyzer = TechnicalAnalyzer(
            CONFIG["analyst"]["technical_analyzer"]
        )
        self.market_health_analyzer = GeneralMarketAnalystModule(CONFIG)
        self.liquidation_risk_model = ProbabilisticLiquidationRiskModel(CONFIG)
        self.predictive_ensembles = RegimePredictiveEnsembles(CONFIG)

        self.logger.info("Analyst and all sub-analyzers initialized.")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="analyst_start"
    )
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
                if latest_kline and latest_kline.get("is_closed"):
                    # Check if this is a new kline we haven't processed yet
                    if latest_kline["open_time"] != self.last_kline_open_time:
                        self.last_kline_open_time = latest_kline["open_time"]
                        self.logger.info(
                            f"New kline closed at {pd.to_datetime(self.last_kline_open_time, unit='ms')}. Triggering analysis."
                        )

                        # Run the analysis pipeline
                        await self.run_analysis_pipeline()

                # Sleep for a short duration to prevent a tight loop
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                self.logger.info("Analyst task cancelled.")
                break
            except Exception as e:
                self.logger.error(
                    f"An error occurred in the Analyst loop: {e}", exc_info=True
                )
                # Wait before retrying to prevent rapid failure loops
                await asyncio.sleep(10)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="load_and_prepare_historical_data",
    )
    async def load_and_prepare_historical_data(
        self,
        historical_klines: Optional[pd.DataFrame] = None,
        historical_agg_trades: Optional[pd.DataFrame] = None,
        historical_futures: Optional[pd.DataFrame] = None,
        fold_id: Optional[Any] = None,  # Added fold_id for checkpointing
    ) -> bool:
        """
        Loads historical data from the exchange or uses provided dataframes (for backtesting/training).
        Then, it prepares this data by generating features and training/loading models.
        Includes robust error handling for data loading and processing.
        """
        self.logger.info(
            f"Loading and preparing historical data for Analyst components (Fold ID: {fold_id})..."
        )

        if (
            historical_klines is not None
            and historical_agg_trades is not None
            and historical_futures is not None
        ):
            # Use provided dataframes (e.g., from TrainingPipeline)
            self._historical_klines = historical_klines
            self._historical_agg_trades = historical_agg_trades
            self._historical_futures = historical_futures
            self.logger.info("Using historical data provided externally.")
        else:
            # Fetch data from exchange (for live/paper trading startup)
            self.logger.info(
                "Fetching historical data from exchange for live/paper trading."
            )
            try:
                lookback_days = CONFIG.get(
                    "analyst_historical_data_lookback_days", 30
                )
                end_time_ms = self.exchange._get_timestamp()
                start_time_ms = end_time_ms - int(
                    timedelta(days=lookback_days).total_seconds() * 1000
                )

                klines_raw = await self.exchange.get_klines(
                    self.trade_symbol, self.timeframe, limit=5000
                )
                if klines_raw:
                    self._historical_klines = pd.DataFrame(klines_raw)
                    self._historical_klines["open_time"] = pd.to_datetime(
                        self._historical_klines["open_time"], unit="ms"
                    )
                    self._historical_klines.set_index("open_time", inplace=True)
                    klines_column_names = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ]
                    self._historical_klines.columns = klines_column_names[
                        : len(self._historical_klines.columns)
                    ]  # Ensure column count matches data
                    for col in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                    ]:
                        if col in self._historical_klines.columns:
                            self._historical_klines[col] = pd.to_numeric(
                                self._historical_klines[col], errors="coerce"
                            )
                else:
                    self.logger.warning(
                        "Failed to fetch historical klines from exchange."
                    )
                    self._historical_klines = pd.DataFrame()

                # Initialize agg_trades_df before the if block
                agg_trades_df = pd.DataFrame()
                agg_trades_raw = await self.exchange.get_historical_agg_trades(
                    self.trade_symbol, start_time_ms, end_time_ms
                )
                if agg_trades_raw:
                    agg_trades_df = pd.DataFrame(
                        agg_trades_raw
                    )  # Assign to agg_trades_df
                    agg_trades_df["timestamp"] = pd.to_datetime(
                        agg_trades_df["T"], unit="ms"
                    )
                    agg_trades_df.set_index("timestamp", inplace=True)
                    agg_trades_df.rename(
                        columns={"p": "price", "q": "quantity", "m": "is_buyer_maker"},
                        inplace=True,
                    )
                    for col in ["price", "quantity"]:
                        agg_trades_df[col] = pd.to_numeric(
                            agg_trades_df[col], errors="coerce"
                        )
                    self._historical_agg_trades = (
                        agg_trades_df  # Assign to _historical_agg_trades
                    )
                else:
                    self.logger.warning(
                        "Failed to fetch historical aggregated trades from exchange."
                    )
                    self._historical_agg_trades = pd.DataFrame()

                futures_data = await self.exchange.get_historical_futures_data(
                    self.trade_symbol, start_time_ms, end_time_ms
                )
                if futures_data:
                    funding_df = pd.DataFrame(futures_data.get("funding_rates", []))
                    if not funding_df.empty:
                        funding_df["timestamp"] = pd.to_datetime(
                            funding_df["fundingTime"], unit="ms"
                        )
                        funding_df.set_index("timestamp", inplace=True)
                        funding_df["fundingRate"] = pd.to_numeric(
                            funding_df["fundingRate"], errors="coerce"
                        )
                        self._historical_futures = funding_df[["fundingRate"]]
                        self._historical_futures.dropna(inplace=True)
                    else:
                        self._historical_futures = pd.DataFrame()
                else:
                    self.logger.warning(
                        "Failed to fetch historical futures data from exchange."
                    )
                    self._historical_futures = pd.DataFrame()

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch historical data from exchange: {e}", exc_info=True
                )
                return False

        if self._historical_klines.empty:
            self.logger.error("No historical klines data available for preparation.")
            return False

        try:
            daily_df_for_sr = (
                self._historical_klines.resample("D")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            daily_df_for_sr.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            sr_levels = self.sr_analyzer.analyze(daily_df_for_sr)
            self.state_manager.set_state("sr_levels", sr_levels)
        except Exception as e:
            self.logger.error(
                f"Error during S/R level analysis: {e}. Using empty SR levels.",
                exc_info=True,
            )
            sr_levels = []
            self.state_manager.set_state("sr_levels", [])

        try:
            historical_features_df = self.feature_engineering.generate_all_features(
                self._historical_klines,
                self._historical_agg_trades,
                self._historical_futures,
                sr_levels,
            )
            if historical_features_df.empty:
                self.logger.error("Failed to generate historical features.")
                return False
        except Exception as e:
            self.logger.error(
                f"Error during historical feature generation: {e}. Aborting data preparation.",
                exc_info=True,
            )
            return False

        # --- Train/Load Market Regime Classifier ---
        regime_classifier_model_path = (
            os.path.join(
                self.model_checkpoint_dir,
                f"{CONFIG['REGIME_CLASSIFIER_MODEL_PREFIX']}{fold_id}.joblib",
            )
            if fold_id is not None
            else None
        )

        try:
            if regime_classifier_model_path and os.path.exists(
                regime_classifier_model_path
            ):
                self.logger.info(
                    f"Loading Market Regime Classifier for fold {fold_id} from {regime_classifier_model_path}..."
                )
                if not self.regime_classifier.load_model(
                    model_path=regime_classifier_model_path
                ):
                    self.logger.warning(
                        f"Failed to load classifier for fold {fold_id}. Retraining."
                    )
                    self.regime_classifier.train_classifier(
                        historical_features_df.copy(),
                        self._historical_klines.copy(),
                        model_path=regime_classifier_model_path,
                    )
            else:
                self.logger.info(
                    f"Training Market Regime Classifier for fold {fold_id}..."
                )
                self.regime_classifier.train_classifier(
                    historical_features_df.copy(),
                    self._historical_klines.copy(),
                    model_path=regime_classifier_model_path,
                )
        except Exception as e:
            self.logger.error(
                f"Error during Market Regime Classifier training/loading: {e}. Aborting data preparation.",
                exc_info=True,
            )
            return False

        # --- Train/Load Predictive Ensembles ---
        ensemble_model_path_prefix = (
            os.path.join(
                self.model_checkpoint_dir,
                f"{CONFIG['ENSEMBLE_MODEL_PREFIX']}{fold_id}_",
            )
            if fold_id is not None
            else None
        )

        try:
            self.logger.info(
                f"Training/Loading Predictive Ensembles for fold {fold_id}..."
            )
            self.predictive_ensembles.train_all_models(
                asset=self.trade_symbol,
                prepared_data=historical_features_df.copy(),
                model_path_prefix=ensemble_model_path_prefix,
            )
        except Exception as e:
            self.logger.error(
                f"Error during Predictive Ensembles training/loading: {e}. Aborting data preparation.",
                exc_info=True,
            )
            return False

        self.logger.info(
            f"Historical data preparation and model training/loading complete for fold {fold_id}."
        )
        return True

    async def run_analysis_pipeline(self):
        """
        Executes the full sequence of analysis tasks.
        """
        self.logger.info("--- Starting Analysis Pipeline ---")

        # Initialize analyst_intelligence with default values
        analyst_intelligence = self._initialize_analyst_intelligence()

        try:
            # Prepare data
            data_package = await self._prepare_analysis_data()
            if not data_package:
                self.logger.error(
                    "Could not prepare analysis data. Aborting analysis cycle."
                )
                self.state_manager.set_state(
                    "analyst_intelligence", analyst_intelligence
                )
                return

            klines, agg_trades_df, futures_df, order_book, current_price = data_package

            # Generate features
            df_features = self._generate_features(klines, agg_trades_df, futures_df)
            if df_features.empty:
                self.logger.error(
                    "Feature generation resulted in empty DataFrame. Aborting analysis cycle."
                )
                self.state_manager.set_state(
                    "analyst_intelligence", analyst_intelligence
                )
                return

            # Run analysis components
            analyst_intelligence = await self._run_analysis_components(
                analyst_intelligence,
                klines,
                df_features,
                agg_trades_df,
                futures_df,
                order_book,
                current_price,
            )

            # Save the consolidated intelligence
            self.state_manager.set_state("analyst_intelligence", analyst_intelligence)
            self.logger.info(
                f"Analysis complete. Intelligence package updated. Regime: {analyst_intelligence['market_regime']}, Ensemble Pred: {analyst_intelligence['ensemble_prediction']}, Conf: {analyst_intelligence['ensemble_confidence']:.2f}"
            )
            self.logger.debug(f"Analyst Intelligence: {analyst_intelligence}")

        except Exception as e:
            self.logger.error(
                f"A critical error occurred during analysis pipeline: {e}",
                exc_info=True,
            )
            # Ensure a default/empty intelligence is saved even on critical failure
            self.state_manager.set_state("analyst_intelligence", analyst_intelligence)

    def _initialize_analyst_intelligence(self) -> Dict[str, Any]:
        """Initialize analyst_intelligence with default values."""
        return {
            "timestamp": int(time.time() * 1000),
            "market_regime": "UNKNOWN",
            "trend_strength": 0.0,
            "adx": 0.0,
            "support_resistance": [],
            "technical_signals": {},
            "market_health_score": 50.0,
            "liquidation_risk_score": 100.0,
            "liquidation_risk_reasons": "N/A",
            "ensemble_prediction": "HOLD",
            "ensemble_confidence": 0.0,
            "directional_confidence_score": 0.0,
            "base_model_predictions": {},
            "ensemble_weights": {},
            "volume_delta": 0.0,
            "current_price": 0.0,
            "market_health_volatility_component": 50.0,
            # Other raw features from df_features (initialized to None/0.0)
            "bb_bandwidth": None,
            "stoch_k": None,
            "CMF": None,
            "fundingRate": None,
            "log_returns": None,
            "volatility_20": None,
            "skewness_20": None,
            "kurtosis_20": None,
            "avg_body_size_20": None,
            "avg_range_size_20": None,
            "volume_change": None,
            "OBV": None,
            "price_vs_vwap": None,
            "ATR": None,
            "current_funding_rate": None,
            "total_bid_liquidity": 0.0,
            "total_ask_liquidity": 0.0,
            "order_book_imbalance": 0.0,
        }

    async def _prepare_analysis_data(
        self,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, float]]:
        """Prepare all data needed for analysis."""
        # Fetch klines data
        klines_raw_data = await self.exchange.get_klines(
            self.trade_symbol, self.timeframe, limit=500
        )
        if not klines_raw_data:
            self.logger.error("Could not fetch latest klines. Aborting analysis cycle.")
            return None

        klines = self._format_klines_data(klines_raw_data)
        if klines.empty:
            self.logger.error(
                "Formatted klines DataFrame is empty. Aborting analysis cycle."
            )
            return None

        # Get order book data
        order_book = self.exchange.order_book
        # bids = order_book.get("bids", {}) if order_book else {}
        # asks = order_book.get("asks", {}) if order_book else {}

        # Prepare aggregated trades data
        agg_trades_df = self._prepare_agg_trades_data()

        # Prepare futures data
        futures_df = self._historical_futures.tail(100)
        if futures_df.empty:
            self.logger.warning(
                "Historical futures data is empty. Some features may be missing."
            )

        current_price = klines["close"].iloc[-1]

        return klines, agg_trades_df, futures_df, order_book, current_price

    def _format_klines_data(self, klines_raw_data: List) -> pd.DataFrame:
        """Format raw klines data into DataFrame."""
        klines = pd.DataFrame(klines_raw_data)
        klines["open_time"] = pd.to_datetime(klines["open_time"], unit="ms")
        klines.set_index("open_time", inplace=True)

        klines_column_names = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        klines.columns = klines_column_names[: len(klines.columns)]

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]:
            if col in klines.columns:
                klines[col] = pd.to_numeric(klines[col], errors="coerce")

        return klines

    def _prepare_agg_trades_data(self) -> pd.DataFrame:
        """Prepare aggregated trades data."""
        recent_trades = self.exchange.recent_trades
        agg_trades_df = pd.DataFrame(recent_trades)

        if not agg_trades_df.empty:
            agg_trades_df["timestamp"] = pd.to_datetime(agg_trades_df["T"], unit="ms")
            agg_trades_df.set_index("timestamp", inplace=True)
            agg_trades_df.rename(
                columns={"p": "price", "q": "quantity", "m": "is_buyer_maker"},
                inplace=True,
            )
            for col in ["price", "quantity"]:
                agg_trades_df[col] = pd.to_numeric(agg_trades_df[col], errors="coerce")
        else:
            agg_trades_df = pd.DataFrame(
                columns=["timestamp", "price", "quantity", "is_buyer_maker"]
            )

        return agg_trades_df

    def _generate_features(
        self,
        klines: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
        futures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate features for analysis."""
        # Ensure ATR is calculated on klines before passing to feature_engineering
        klines.ta.atr(
            length=CONFIG["best_params"]["atr_period"],
            append=True,
            col_names=("ATR"),
        )
        if klines["ATR"].iloc[-1] is None or pd.isna(klines["ATR"].iloc[-1]):
            self.logger.warning(
                "ATR calculation resulted in NaN for the latest kline. This might affect downstream models."
            )

        df_features = self.feature_engineering.generate_all_features(
            klines.copy(),
            agg_trades_df.copy(),
            futures_df.copy(),
            self.state_manager.get_state("sr_levels", []),
        )

        # Ensure ATR from klines is also in df_features if not already there
        if "ATR" not in df_features.columns and "ATR" in klines.columns:
            df_features["ATR"] = klines["ATR"]

        return df_features

    async def _run_analysis_components(
        self,
        analyst_intelligence: Dict[str, Any],
        klines: pd.DataFrame,
        df_features: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        order_book: Dict,
        current_price: float,
    ) -> Dict[str, Any]:
        """Run all analysis components and update analyst_intelligence."""
        # Market Regime Classification
        analyst_intelligence = await self._run_market_regime_analysis(
            analyst_intelligence, klines, df_features
        )

        # Support/Resistance Analysis
        analyst_intelligence = await self._run_sr_analysis(analyst_intelligence, klines)

        # Technical Analysis
        analyst_intelligence = await self._run_technical_analysis(
            analyst_intelligence, klines
        )

        # Market Health Analysis
        analyst_intelligence = await self._run_market_health_analysis(
            analyst_intelligence, klines
        )

        # Liquidation Risk Analysis
        analyst_intelligence = await self._run_liquidation_risk_analysis(
            analyst_intelligence, klines, order_book, current_price
        )

        # Predictive Ensembles
        analyst_intelligence = await self._run_predictive_ensembles(
            analyst_intelligence,
            df_features,
            klines,
            agg_trades_df,
            order_book,
            current_price,
        )

        # Additional Features
        analyst_intelligence = self._add_additional_features(
            analyst_intelligence, df_features, futures_df, order_book
        )

        return analyst_intelligence

    async def _run_market_regime_analysis(
        self,
        analyst_intelligence: Dict[str, Any],
        klines: pd.DataFrame,
        df_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run market regime classification analysis."""
        try:
            predicted_regime, trend_strength, current_adx = (
                self.regime_classifier.predict_regime(df_features.copy())
            )
            analyst_intelligence["market_regime"] = predicted_regime
            analyst_intelligence["trend_strength"] = trend_strength
            analyst_intelligence["adx"] = current_adx
        except Exception as e:
            self.logger.error(
                f"Error during market regime classification: {e}. Using default values.",
                exc_info=True,
            )

        return analyst_intelligence

    async def _run_sr_analysis(
        self, analyst_intelligence: Dict[str, Any], klines: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run support/resistance analysis."""
        try:
            sr_levels = self.sr_analyzer.analyze_support_resistance(klines.copy())
            analyst_intelligence["support_resistance"] = sr_levels
        except Exception as e:
            self.logger.error(
                f"Error during support/resistance analysis: {e}. Using empty list.",
                exc_info=True,
            )

        return analyst_intelligence

    async def _run_technical_analysis(
        self, analyst_intelligence: Dict[str, Any], klines: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run technical analysis."""
        try:
            technical_signals = self.technical_analyzer.analyze(klines.copy())
            analyst_intelligence["technical_signals"] = technical_signals
        except Exception as e:
            self.logger.error(
                f"Error during technical analysis: {e}. Using empty technical signals.",
                exc_info=True,
            )

        return analyst_intelligence

    async def _run_market_health_analysis(
        self, analyst_intelligence: Dict[str, Any], klines: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run market health analysis."""
        try:
            market_health_score = self.market_health_analyzer.get_market_health_score(
                klines.copy()
            )
            analyst_intelligence["market_health_score"] = market_health_score
            analyst_intelligence["market_health_volatility_component"] = (
                market_health_score
            )
        except Exception as e:
            self.logger.error(
                f"Error during market health analysis: {e}. Using default score.",
                exc_info=True,
            )

        return analyst_intelligence

    async def _run_liquidation_risk_analysis(
        self,
        analyst_intelligence: Dict[str, Any],
        klines: pd.DataFrame,
        order_book: Dict,
        current_price: float,
    ) -> Dict[str, Any]:
        """Run liquidation risk analysis."""
        try:
            current_position_risk = await self.exchange.get_position_risk(
                self.trade_symbol
            )
            current_position_notional = 0.0
            current_liquidation_price = 0.0

            if current_position_risk and len(current_position_risk) > 0:
                pos = current_position_risk[0]
                current_position_notional = (
                    float(pos.get("positionAmt", 0)) * current_price
                )
                current_liquidation_price = float(pos.get("liquidationPrice", 0))

            liquidation_risk_score, lss_reasons = (
                self.liquidation_risk_model.calculate_lss(
                    current_price,
                    current_position_notional,
                    current_liquidation_price,
                    klines.copy(),
                    order_book,
                )
            )
            analyst_intelligence["liquidation_risk_score"] = liquidation_risk_score
            analyst_intelligence["liquidation_risk_reasons"] = lss_reasons
        except Exception as e:
            self.logger.error(
                f"Error during liquidation risk calculation: {e}. Using default scores.",
                exc_info=True,
            )

        return analyst_intelligence

    async def _run_predictive_ensembles(
        self,
        analyst_intelligence: Dict[str, Any],
        df_features: pd.DataFrame,
        klines: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
        order_book: Dict,
        current_price: float,
    ) -> Dict[str, Any]:
        """Run predictive ensembles analysis."""
        try:
            ensemble_prediction_result = self.predictive_ensembles.get_all_predictions(
                asset=self.trade_symbol,
                current_features=df_features.copy(),
                klines_df=klines.copy(),
                agg_trades_df=agg_trades_df.copy(),
                order_book_data=order_book,
                current_price=current_price,
            )
            analyst_intelligence["ensemble_prediction"] = ensemble_prediction_result[
                "prediction"
            ]
            analyst_intelligence["ensemble_confidence"] = ensemble_prediction_result[
                "confidence"
            ]
            analyst_intelligence["directional_confidence_score"] = (
                ensemble_prediction_result["confidence"]
            )
            analyst_intelligence["base_model_predictions"] = (
                ensemble_prediction_result.get("base_predictions", {})
            )
            analyst_intelligence["ensemble_weights"] = ensemble_prediction_result.get(
                "ensemble_weights", {}
            )
        except Exception as e:
            self.logger.error(
                f"Error during predictive ensembles prediction: {e}. Using default values.",
                exc_info=True,
            )

        return analyst_intelligence

    def _add_additional_features(
        self,
        analyst_intelligence: Dict[str, Any],
        df_features: pd.DataFrame,
        futures_df: pd.DataFrame,
        order_book: Dict,
    ) -> Dict[str, Any]:
        """Add additional features to analyst_intelligence."""
        # Add volume_delta from df_features
        if "volume_delta" in df_features.columns and not df_features.empty:
            analyst_intelligence["volume_delta"] = float(
                df_features["volume_delta"].iloc[-1]
            )
        else:
            analyst_intelligence["volume_delta"] = 0.0

        if "fundingRate" in futures_df.columns and not futures_df.empty:
            analyst_intelligence["current_funding_rate"] = float(
                futures_df["fundingRate"].iloc[-1]
            )
        else:
            analyst_intelligence["current_funding_rate"] = None

        # Add current order book liquidity/imbalance metrics
        try:
            bids = order_book.get("bids", {}) if order_book else {}
            asks = order_book.get("asks", {}) if order_book else {}

            total_bid_qty = sum(bids.values()) if bids else 0.0
            total_ask_qty = sum(asks.values()) if asks else 0.0

            analyst_intelligence["total_bid_liquidity"] = total_bid_qty
            analyst_intelligence["total_ask_liquidity"] = total_ask_qty
            if total_bid_qty + total_ask_qty > 0:
                analyst_intelligence["order_book_imbalance"] = (
                    total_bid_qty - total_ask_qty
                ) / (total_bid_qty + total_ask_qty)
            else:
                analyst_intelligence["order_book_imbalance"] = 0.0
        except Exception as e:
            self.logger.error(
                f"Error calculating order book metrics: {e}. Using default values.",
                exc_info=True,
            )

        return analyst_intelligence

    @handle_errors(
        exceptions=(KeyError, TypeError, AttributeError),
        default_return=0.5,
        context="calculate_confidence",
    )
    def _calculate_confidence(self, signals: Dict, regime: str) -> float:
        return signals.get("ensemble_confidence", 0.5)
