# src/components/modular_analyst.py

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import pandas as pd

from src.utils.model_manager import ModelManager
from backtesting.ares_data_preparer import get_sr_levels, calculate_and_label_regimes
from src.interfaces import IAnalyst, MarketData, AnalysisResult, EventType
from src.interfaces.base_interfaces import IExchangeClient, IStateManager, IEventBus
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
    handle_data_processing_errors,
)


class ModularAnalyst(IAnalyst):
    """
    Modular implementation of the Analyst that implements the IAnalyst interface.
    Uses dependency injection and event-driven communication.
    """

    def __init__(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        event_bus: Optional[IEventBus] = None,
    ):
        """
        Initialize the modular analyst.

        Args:
            exchange_client: Exchange client for data access
            state_manager: State manager for persistence
            event_bus: Optional event bus for communication
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.logger = system_logger.getChild("ModularAnalyst")

        self.model_manager = ModelManager()

        # Configuration
        self.trade_symbol = settings.trade_symbol
        self.timeframe = settings.timeframe
        self.last_kline_open_time = None
        self.running = False

        # Internal storage for historical data
        self._historical_klines = pd.DataFrame()
        self._historical_agg_trades = pd.DataFrame()
        self._historical_futures = pd.DataFrame()

        # Model checkpoint directory
        self.model_checkpoint_dir = os.path.join(
            CONFIG["CHECKPOINT_DIR"], "analyst_models"
        )
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        # Initialize sub-analyzers
        self.feature_engineering = FeatureEngineeringEngine(settings.CONFIG)
        self.sr_analyzer = SRLevelAnalyzer(settings.CONFIG["analyst"]["sr_analyzer"])
        self.regime_classifier = MarketRegimeClassifier(
            settings.CONFIG, self.sr_analyzer
        )
        self.technical_analyzer = TechnicalAnalyzer(
            settings.CONFIG["analyst"]["technical_analyzer"]
        )
        self.market_health_analyzer = GeneralMarketAnalystModule(settings.CONFIG)
        self.liquidation_risk_model = ProbabilisticLiquidationRiskModel(settings.CONFIG)
        self.predictive_ensembles = RegimePredictiveEnsembles(settings.CONFIG)

        self.logger.info("ModularAnalyst initialized")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="modular_analyst_start"
    )
    async def start(self) -> None:
        """Start the modular analyst"""
        self.logger.info("Starting ModularAnalyst")
        self.running = True

        # Load initial historical data before subscribing to live updates
        await self._load_historical_data(self.trade_symbol, None, None)

        # Subscribe to market data events if event bus is available
        if self.event_bus:
            await self.event_bus.subscribe(
                EventType.MARKET_DATA_RECEIVED, self._handle_market_data
            )

        self.logger.info("ModularAnalyst started")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="modular_analyst_stop"
    )
    async def stop(self) -> None:
        """Stop the modular analyst"""
        self.logger.info("Stopping ModularAnalyst")
        self.running = False

        # Unsubscribe from events if event bus is available
        if self.event_bus:
            await self.event_bus.unsubscribe(
                EventType.MARKET_DATA_RECEIVED, self._handle_market_data
            )

        self.logger.info("ModularAnalyst stopped")

    @handle_errors(
        exceptions=(Exception,), default_return=None, context="analyze_market_data"
    )
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult:
        """
        Analyze market data and return analysis result.

        Args:
            market_data: Market data to analyze

        Returns:
            Analysis result
        """
        if not self.running:
            self.logger.warning("Analyst not running, skipping analysis")
            return None

        self.logger.debug(f"Analyzing market data for {market_data.symbol}")

        # Convert MarketData to DataFrame format
        df = pd.DataFrame(
            [
                {
                    "open_time": market_data.timestamp,
                    "open": market_data.open,
                    "high": market_data.high,
                    "low": market_data.low,
                    "close": market_data.close,
                    "volume": market_data.volume,
                    "close_time": market_data.timestamp + timedelta(minutes=1),
                }
            ]
        )

        # Run analysis pipeline
        analysis_result = await self._run_analysis_pipeline(df)

        # Publish analysis completed event
        if self.event_bus and analysis_result:
            await self.event_bus.publish(
                EventType.ANALYSIS_COMPLETED, analysis_result, "ModularAnalyst"
            )

        return analysis_result

    @handle_errors(
        exceptions=(Exception,), default_return=[], context="get_historical_analysis"
    )
    async def get_historical_analysis(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[AnalysisResult]:
        """
        Get historical analysis results.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of analysis results
        """
        self.logger.info(
            f"Getting historical analysis for {symbol} from {start_date} to {end_date}"
        )

        # Load historical data
        historical_data = await self._load_historical_data(symbol, start_date, end_date)

        if historical_data.empty:
            self.logger.warning("No historical data found")
            return []

        # Run analysis on historical data
        analysis_results = []
        for _, row in historical_data.iterrows():
            market_data = MarketData(
                symbol=symbol,
                timestamp=row["open_time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                interval=self.timeframe,
            )

            result = await self.analyze_market_data(market_data)
            if result:
                analysis_results.append(result)

        return analysis_results

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="train_models"
    )
    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """
        Train analysis models.

        Args:
            training_data: Training data

        Returns:
            True if training successful
        """
        self.logger.info("Training analysis models")

        try:
            # Train feature engineering
            await self.feature_engineering.generate_all_features(training_data)

            # Train regime classifier
            await self.regime_classifier.train(training_data)

            # Train predictive ensembles
            await self.predictive_ensembles.train_models(training_data)

            # Save models
            await self._save_models()

            self.logger.info("Model training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model training failed: {e}", exc_info=True)
            return False

    @handle_errors(exceptions=(Exception,), default_return=False, context="load_models")
    async def load_models(self, model_path: str) -> bool:
        """
        Load trained models.

        Args:
            model_path: Path to model files

        Returns:
            True if loading successful
        """
        self.logger.info(f"Loading models from {model_path}")

        try:
            # Load regime classifier
            await self.regime_classifier.load_models(model_path)

            # Load predictive ensembles
            await self.predictive_ensembles.load_models(model_path)

            # Load feature engineering models
            await self.feature_engineering.load_autoencoder(model_path)

            self.logger.info("Models loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}", exc_info=True)
            return False

    async def _handle_market_data(self, event) -> None:
        """
        Handles new market data events, updates historical data,
        and triggers a new analysis run.
        """
        market_data: MarketData = event.data

        # Convert new data to a DataFrame row
        new_kline_row = pd.DataFrame(
            [
                {
                    "open": market_data.open,
                    "high": market_data.high,
                    "low": market_data.low,
                    "close": market_data.close,
                    "volume": market_data.volume,
                }
            ],
            index=[pd.to_datetime(market_data.timestamp, unit="ms")],
        )

        # Append new data and keep the last N rows (e.g., 1000) for analysis
        self._historical_klines = pd.concat([self._historical_klines, new_kline_row])
        self._historical_klines = self._historical_klines.iloc[-1000:]

        # Now, run the full analysis on the updated historical data
        await self.run_full_analysis()

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=None,
        context="run_analysis_pipeline",
    )
    async def _run_analysis_pipeline(
        self, df: pd.DataFrame
    ) -> Optional[AnalysisResult]:
        """Run the complete analysis pipeline"""
        try:
            # Initialize analysis data
            analysis_data = await self._initialize_analyst_intelligence(df)

            # Prepare analysis data
            prepared_data = await self._prepare_analysis_data(analysis_data)

            # Run analysis components
            analysis_results = await self._run_analysis_components(prepared_data)

            # Build analysis result
            result = AnalysisResult(
                timestamp=datetime.now(),
                symbol=self.trade_symbol,
                confidence=analysis_results.get("confidence", 0.0),
                signal=analysis_results.get("signal", "HOLD"),
                features=analysis_results.get("features", {}),
                technical_indicators=analysis_results.get("technical_indicators", {}),
                market_regime=analysis_results.get("market_regime", "UNKNOWN"),
                support_resistance=analysis_results.get("support_resistance", {}),
                risk_metrics=analysis_results.get("risk_metrics", {}),
            )

            return result

        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {e}", exc_info=True)
            return None

    async def _initialize_analyst_intelligence(
        self, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Initialize analyst intelligence with data"""
        return {
            "klines_data": df,
            "agg_trades_data": pd.DataFrame(),
            "futures_data": pd.DataFrame(),
        }

    async def _prepare_analysis_data(
        self, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for analysis"""
        klines_df = analysis_data["klines_data"]

        # Format klines data
        formatted_klines = await self._format_klines_data(klines_df)

        # Prepare aggregated trades data
        agg_trades_data = await self._prepare_agg_trades_data(
            analysis_data.get("agg_trades_data", pd.DataFrame())
        )

        return {
            "formatted_klines": formatted_klines,
            "agg_trades_data": agg_trades_data,
            "futures_data": analysis_data.get("futures_data", pd.DataFrame()),
        }

    async def _format_klines_data(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """Format klines data for analysis"""
        if klines_df.empty:
            return pd.DataFrame()

        # Ensure required columns exist
        required_columns = ["open_time", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in klines_df.columns:
                self.logger.warning(f"Missing column: {col}")
                return pd.DataFrame()

        return klines_df.copy()

    async def _prepare_agg_trades_data(
        self, agg_trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare aggregated trades data"""
        if agg_trades_df.empty:
            return pd.DataFrame()

        return agg_trades_df.copy()

    async def _run_analysis_components(
        self, prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all analysis components"""
        results = {}

        try:
            # Market regime analysis
            regime_result = await self._run_market_regime_analysis(prepared_data)
            results.update(regime_result)

            # Support/Resistance analysis
            sr_result = await self._run_sr_analysis(prepared_data)
            results.update(sr_result)

            # Technical analysis
            technical_result = await self._run_technical_analysis(prepared_data)
            results.update(technical_result)

            # Market health analysis
            health_result = await self._run_market_health_analysis(prepared_data)
            results.update(health_result)

            # Liquidation risk analysis
            risk_result = await self._run_liquidation_risk_analysis(prepared_data)
            results.update(risk_result)

            # Predictive ensembles
            ensemble_result = await self._run_predictive_ensembles(prepared_data)
            results.update(ensemble_result)

        except Exception as e:
            self.logger.error(f"Error running analysis components: {e}", exc_info=True)

        return results

    async def _run_market_regime_analysis(
        self, prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run market regime analysis"""
        try:
            klines_df = prepared_data["formatted_klines"]
            if klines_df.empty:
                return {"market_regime": "UNKNOWN"}

            regime = await self.regime_classifier.predict_regime(klines_df)
            return {"market_regime": regime}

        except Exception as e:
            self.logger.error(f"Market regime analysis failed: {e}")
            return {"market_regime": "UNKNOWN"}

    async def _run_sr_analysis(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run support/resistance analysis"""
        try:
            klines_df = prepared_data["formatted_klines"]
            if klines_df.empty:
                return {"support_resistance": {}}

            sr_levels = await self.sr_analyzer.analyze(klines_df)
            return {"support_resistance": sr_levels}

        except Exception as e:
            self.logger.error(f"SR analysis failed: {e}")
            return {"support_resistance": {}}

    async def _run_technical_analysis(
        self, prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run technical analysis"""
        try:
            klines_df = prepared_data["formatted_klines"]
            if klines_df.empty:
                return {"technical_indicators": {}}

            technical_indicators = await self.technical_analyzer.analyze(klines_df)
            return {"technical_indicators": technical_indicators}

        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {"technical_indicators": {}}

    async def _run_market_health_analysis(
        self, prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run market health analysis"""
        try:
            klines_df = prepared_data["formatted_klines"]
            if klines_df.empty:
                return {"market_health": 0.0}

            health_score = await self.market_health_analyzer.get_market_health_score(
                klines_df
            )
            return {"market_health": health_score}

        except Exception as e:
            self.logger.error(f"Market health analysis failed: {e}")
            return {"market_health": 0.0}

    async def _run_liquidation_risk_analysis(
        self, prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run liquidation risk analysis"""
        try:
            klines_df = prepared_data["formatted_klines"]
            if klines_df.empty:
                return {"risk_metrics": {}}

            risk_metrics = await self.liquidation_risk_model.get_risk_metrics(klines_df)
            return {"risk_metrics": risk_metrics}

        except Exception as e:
            self.logger.error(f"Liquidation risk analysis failed: {e}")
            return {"risk_metrics": {}}

    async def run_full_analysis(self):
        """
        Runs the entire analysis pipeline on the current historical data.
        This is the new central point for triggering analysis.
        """
        if not self.running or self._historical_klines.empty:
            return

        self.logger.debug(
            "Running full analysis pipeline on updated historical data..."
        )

        # The existing pipeline structure can now be called with the full dataframes
        analysis_result = await self._run_analysis_pipeline(self._historical_klines)

        if self.event_bus and analysis_result:
            await self.event_bus.publish(
                EventType.ANALYSIS_COMPLETED, analysis_result, "ModularAnalyst"
            )

    async def _run_predictive_ensembles(
        self, prepared_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs prediction using the production model loaded from MLflow.
        This method is now the core of the predictive logic.
        """
        self.logger.debug("Running prediction with production MLflow model.")
        try:
            # 1. Get the production model run ID from the state manager
            run_id = self.state_manager.get_state("production_model_run_id")
            if not run_id:
                self.logger.warning(
                    "No production model run ID found in state. Cannot make prediction."
                )
                return {"confidence": 0.5, "signal": "HOLD", "features": {}}

            # 2. Load the model using the ModelManager
            # The artifact path must match what was used during training
            model = self.model_manager.load_model_from_mlflow(run_id, "model_live")
            if not model:
                self.logger.error(f"Failed to load model from MLflow run ID: {run_id}")
                return {"confidence": 0.5, "signal": "HOLD", "features": {}}

            # 3. Use the functions from ares_data_preparer to generate features
            # This ensures 100% consistency between training and live prediction.
            klines_df = prepared_data["formatted_klines"]
            agg_trades_df = self._historical_agg_trades  # Use the full historical data
            futures_df = self._historical_futures

            daily_df = (
                klines_df.resample("D")
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
            sr_levels = get_sr_levels(daily_df)
            params = CONFIG.get("BEST_PARAMS", {})

            featured_data = calculate_and_label_regimes(
                klines_df, agg_trades_df, futures_df, params, sr_levels
            )

            latest_data_point = featured_data.tail(1)
            if latest_data_point.empty:
                return {"confidence": 0.5, "signal": "HOLD", "features": {}}

            # 4. Predict using the loaded model
            training_features = model.feature_name_
            # Ensure all required columns are present, fill with 0 if not
            for col in training_features:
                if col not in latest_data_point.columns:
                    latest_data_point[col] = 0

            prediction_proba = model.predict_proba(latest_data_point[training_features])
            confidence = prediction_proba[0][1]  # Probability of the "1" class (long)
            signal = (
                "BUY"
                if confidence > 0.55
                else ("SELL" if confidence < 0.45 else "HOLD")
            )

            return {
                "confidence": confidence,
                "signal": signal,
                "features": latest_data_point.to_dict("records")[0],
            }

        except Exception as e:
            self.logger.error(f"Predictive ensembles failed: {e}", exc_info=True)
            return {"confidence": 0.5, "signal": "HOLD", "features": {}}

    async def _load_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Loads historical data from the cached CSV files into memory.
        """
        from src.analyst.data_utils import (
            load_klines_data,
            load_agg_trades_data,
            load_futures_data,
        )

        self.logger.info("Loading historical data from cache files...")
        try:
            klines_filename = CONFIG["KLINES_FILENAME"]
            agg_trades_filename = CONFIG["AGG_TRADES_FILENAME"]
            futures_filename = CONFIG["FUTURES_FILENAME"]

            if os.path.exists(klines_filename):
                self._historical_klines = load_klines_data(klines_filename)
                self.logger.info(
                    f"Loaded {len(self._historical_klines)} historical klines."
                )
            else:
                self.logger.warning(f"Kline cache file not found: {klines_filename}")

            if os.path.exists(agg_trades_filename):
                self._historical_agg_trades = load_agg_trades_data(agg_trades_filename)
            if os.path.exists(futures_filename):
                self._historical_futures = load_futures_data(futures_filename)

        except Exception as e:
            self.logger.error(
                f"Failed to load historical data from files: {e}", exc_info=True
            )

    async def _save_models(self) -> None:
        """Save trained models"""
        try:
            # Save to checkpoint directory
            checkpoint_path = os.path.join(
                self.model_checkpoint_dir, f"{self.trade_symbol}_models"
            )
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save individual models
            await self.regime_classifier.save_models(checkpoint_path)
            await self.predictive_ensembles.save_models(checkpoint_path)
            await self.feature_engineering.save_autoencoder(checkpoint_path)

            self.logger.info(f"Models saved to {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
