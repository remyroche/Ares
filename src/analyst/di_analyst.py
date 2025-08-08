# src/analyst/di_analyst.py

"""
Dependency injection-aware Analyst implementation.

This module provides an Analyst implementation that properly supports
dependency injection patterns and modern architectural practices.
"""

from typing import Any
from datetime import datetime

import pandas as pd

from src.core.injectable_base import AnalystBase
from src.interfaces.base_interfaces import (
    AnalysisResult,
    IAnalyst,
    IEventBus,
    IExchangeClient,
    IStateManager,
    MarketData,
)
from src.analyst.dual_model_system import DualModelSystem
from src.analyst.market_health_analyzer import MarketHealthAnalyzer
from src.analyst.liquidation_risk_model import LiquidationRiskModel
from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from src.utils.error_handler import handle_errors


class DIAnalyst(AnalystBase, IAnalyst):
    """
    Dependency injection-aware Analyst implementation.
    
    This analyst implementation properly supports dependency injection,
    configuration management, and modern architectural patterns.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        exchange_client: IExchangeClient | None = None,
        state_manager: IStateManager | None = None,
        event_bus: IEventBus | None = None,
    ):
        super().__init__(config, exchange_client, state_manager, event_bus)
        
        # Analyst state
        self.is_analyzing = False
        self.analysis_results: dict[str, Any] = {}
        self.analysis_history: list[dict[str, Any]] = []
        
        # Configuration
        self.analyst_config = self.config.get("analyst", {})
        self.analysis_interval = self.analyst_config.get("analysis_interval", 3600)
        self.max_analysis_history = self.analyst_config.get("max_analysis_history", 100)
        self.enable_technical_analysis = self.analyst_config.get("enable_technical_analysis", True)
        
        # Analysis components (will be initialized later)
        self.dual_model_system: DualModelSystem | None = None
        self.market_health_analyzer: MarketHealthAnalyzer | None = None
        self.liquidation_risk_model: LiquidationRiskModel | None = None
        self.feature_engineering_orchestrator: FeatureEngineeringOrchestrator | None = None

    async def initialize(self) -> bool:
        """Initialize the analyst with all dependencies."""
        if not await super().initialize():
            return False

        try:
            # Initialize analysis components
            await self._initialize_analysis_components()
            
            # Set up event subscriptions if event bus is available
            if self.event_bus:
                await self._setup_event_subscriptions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analyst: {e}")
            return False

    async def _initialize_analysis_components(self) -> None:
        """Initialize analysis components with proper configuration."""
        # Dual Model System
        if self.analyst_config.get("enable_dual_model_system", True):
            self.dual_model_system = DualModelSystem(
                self.analyst_config.get("dual_model_system", {})
            )
            await self.dual_model_system.initialize()

        # Market Health Analyzer
        if self.analyst_config.get("enable_market_health_analysis", True):
            self.market_health_analyzer = MarketHealthAnalyzer(
                self.analyst_config.get("market_health_analyzer", {})
            )
            await self.market_health_analyzer.initialize()

        # Liquidation Risk Model
        if self.analyst_config.get("enable_liquidation_risk_analysis", True):
            self.liquidation_risk_model = LiquidationRiskModel(
                self.analyst_config.get("liquidation_risk_model", {})
            )
            await self.liquidation_risk_model.initialize()

        # Feature Engineering Orchestrator
        if self.analyst_config.get("enable_feature_engineering", True):
            self.feature_engineering_orchestrator = FeatureEngineeringOrchestrator(
                self.analyst_config.get("feature_engineering_orchestrator", {})
            )
            await self.feature_engineering_orchestrator.initialize()

        self.logger.info("Analysis components initialized")

    async def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for market data."""
        from src.interfaces.event_bus import EventType
        
        # Subscribe uses string event types in EventBus implementation
        self.event_bus.subscribe(
            EventType.MARKET_DATA_RECEIVED.value,
            self.analyze_market_data
        )
        self.logger.debug("Event subscriptions set up")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="market data analysis"
    )
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult | None:
        """Analyze market data and return analysis result."""
        if not self.is_initialized or not self._validate_dependencies():
            self.logger.error("Analyst not properly initialized")
            return None

        try:
            self.is_analyzing = True
            self.logger.debug(f"Analyzing market data for {market_data.symbol}")

            # Perform comprehensive analysis
            analysis_result = await self._perform_comprehensive_analysis(market_data)
            
            # Store analysis result
            if analysis_result:
                await self._store_analysis_result(analysis_result)
                
                # Publish analysis completed event (uses string event type)
                if self.event_bus:
                    from src.interfaces.event_bus import EventType
                    await self.event_bus.publish(
                        EventType.ANALYSIS_COMPLETED.value,
                        analysis_result
                    )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return None
        finally:
            self.is_analyzing = False

    async def _perform_comprehensive_analysis(self, market_data: MarketData) -> AnalysisResult | None:
        """Perform comprehensive market analysis using all available components."""
        try:
            # Initialize analysis components
            features = {}
            technical_indicators = {}
            risk_metrics = {}
            support_resistance = {}
            market_regime = "UNKNOWN"
            signal = "HOLD"
            confidence = 0.0

            # Dual model system analysis
            if self.dual_model_system:
                dual_result = await self.dual_model_system.analyze(market_data)
                if dual_result:
                    signal = dual_result.get("signal", "HOLD")
                    confidence = dual_result.get("confidence", 0.0)
                    features.update(dual_result.get("features", {}))

            # Market health analysis
            if self.market_health_analyzer:
                health_result = await self.market_health_analyzer.analyze(market_data)
                if health_result:
                    risk_metrics.update(health_result.get("risk_metrics", {}))
                    market_regime = health_result.get("market_regime", "UNKNOWN")

            # Liquidation risk analysis
            if self.liquidation_risk_model:
                liquidation_result = await self.liquidation_risk_model.analyze(market_data)
                if liquidation_result:
                    risk_metrics.update(liquidation_result.get("risk_metrics", {}))

            # Feature engineering
            if self.feature_engineering_orchestrator:
                feature_result = await self.feature_engineering_orchestrator.analyze(market_data)
                if feature_result:
                    features.update(feature_result.get("features", {}))

            # Build analysis result
            analysis_result = AnalysisResult(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                confidence=confidence,
                signal=signal,
                features=features,
                technical_indicators=technical_indicators,
                market_regime=market_regime,
                support_resistance=support_resistance,
                risk_metrics=risk_metrics,
            )

            return analysis_result
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return None

    async def _store_analysis_result(self, analysis_result: AnalysisResult) -> None:
        """Store analysis result in history."""
        try:
            record = {
                "timestamp": analysis_result.timestamp,
                "symbol": analysis_result.symbol,
                "confidence": analysis_result.confidence,
                "signal": analysis_result.signal,
            }
            self.analysis_history.append(record)
            if len(self.analysis_history) > self.max_analysis_history:
                self.analysis_history = self.analysis_history[-self.max_analysis_history :]
        except Exception as e:
            self.logger.error(f"Failed to store analysis result: {e}")

    async def get_historical_analysis(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[AnalysisResult]:
        """Get historical analysis results."""
        try:
            # Filter history by symbol and date range
            filtered_results = []
            
            for result in self.analysis_history:
                result_time = datetime.fromisoformat(result["timestamp"])
                if (result.get("symbol") == symbol and 
                    start_date <= result_time <= end_date):
                    
                    # Convert back to AnalysisResult object
                    analysis_result = AnalysisResult(
                        timestamp=result_time,
                        symbol=result["symbol"],
                        confidence=result["confidence"],
                        signal=result["signal"],
                        features={},  # Historical features not stored in summary
                        technical_indicators={},
                        market_regime=result["market_regime"],
                        support_resistance={},
                        risk_metrics={},
                    )
                    filtered_results.append(analysis_result)

            return filtered_results

        except Exception as e:
            self.logger.error(f"Failed to get historical analysis: {e}")
            return []

    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """Train analysis models."""
        try:
            self.logger.info("Training analysis models")
            
            success = True
            
            # Train dual model system
            if self.dual_model_system:
                if not await self.dual_model_system.train(training_data):
                    success = False

            # Train other components that support training
            if self.liquidation_risk_model:
                if not await self.liquidation_risk_model.train(training_data):
                    success = False

            self.logger.info(f"Model training {'completed' if success else 'failed'}")
            return success

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False

    async def load_models(self, model_path: str) -> bool:
        """Load trained models."""
        try:
            self.logger.info(f"Loading models from {model_path}")
            
            success = True
            
            # Load dual model system
            if self.dual_model_system:
                if not await self.dual_model_system.load_models(model_path):
                    success = False

            # Load other components that support model loading
            if self.liquidation_risk_model:
                if not await self.liquidation_risk_model.load_models(model_path):
                    success = False

            self.logger.info(f"Model loading {'completed' if success else 'failed'}")
            return success

        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False

    async def _start_component(self) -> None:
        """Start analyst-specific operations."""
        self.logger.info("Analyst component started")

    async def _stop_component(self) -> None:
        """Stop analyst-specific operations."""
        self.is_analyzing = False
        self.logger.info("Analyst component stopped")