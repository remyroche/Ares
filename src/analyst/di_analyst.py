from src.utils.tprint import tprint

from src.core.error_classes import execution_error, initialization_error
from .core.decorators import handles_errors
'\nDependency injection-aware Analyst implementation.\n\nThis module provides an Analyst implementation that properly supports\ndependency injection patterns and modern architectural practices.\n'
import logging
from datetime import datetime
from typing import Any
from .analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from .analyst.liquidation_risk_model import LiquidationRiskModel
from .analyst.market_health_analyzer import MarketHealthAnalyzer
from .core.injectable_base import AnalystBase
from src.interfaces.base_interfaces import AnalysisResult, IAnalyst, IEventBus, IExchangeClient, IStateManager, MarketData
# Note: dual_model_system has been refactored into training steps
# Using training steps components instead
try:
    from .training.steps.model_training import GeneralModelTrainer, AnalystModelTrainer
    TRAINING_STEPS_AVAILABLE = True
except ImportError:
    TRAINING_STEPS_AVAILABLE = False
    GeneralModelTrainer = None
    AnalystModelTrainer = None
from src.utils.warning_symbols import failed, initialization_error
import pandas as pd
import time

class DIAnalyst(AnalystBase, IAnalyst):
    """
    Dependency injection-aware Analyst implementation.

    This analyst implementation properly supports dependency injection,
    configuration management, and modern architectural patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None, exchange_client: IExchangeClient | None = None, state_manager: IStateManager | None = None, event_bus: IEventBus | None = None) -> None:
        super().__init__(config, exchange_client, state_manager, event_bus)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_analyzing = False
        self.analysis_results: dict[str, Any] = {}
        self.analysis_history: list[dict[str, Any]] = []
        self.analyst_config = self.config.get('analyst', {})
        self.analysis_interval = self.analyst_config.get('analysis_interval', 3600)
        self.max_analysis_history = self.analyst_config.get('max_analysis_history', 100)
        self.enable_technical_analysis = self.analyst_config.get('enable_technical_analysis', True)
        self.dual_model_system: GeneralModelTrainer | None = None
        self.market_health_analyzer: MarketHealthAnalyzer | None = None
        self.liquidation_risk_model: LiquidationRiskModel | None = None
        self.feature_engineering_orchestrator: FeatureEngineeringOrchestrator | None = None

    async def initialize(self) -> bool:
        """Initialize the analyst with all dependencies."""
        if not await super().initialize():
            return False
        try:
            await self._initialize_analysis_components()
            if self.event_bus:
                await self._setup_event_subscriptions()
            return True
        except Exception:
            self.tprint(failed('Failed to initialize analyst: {e}'))
            return False

    async def _initialize_analysis_components(self) -> None:
        """Initialize analysis components with proper configuration."""
        if self.analyst_config.get('enable_dual_model_system', True):
            self.dual_model_system = DualModelSystem(self.analyst_config.get('dual_model_system', {}))
            await self.dual_model_system.initialize()
        if self.analyst_config.get('enable_market_health_analysis', True):
            self.market_health_analyzer = MarketHealthAnalyzer(self.analyst_config.get('market_health_analyzer', {}))
            await self.market_health_analyzer.initialize()
        if self.analyst_config.get('enable_liquidation_risk_analysis', True):
            self.liquidation_risk_model = LiquidationRiskModel(self.analyst_config.get('liquidation_risk_model', {}))
            await self.liquidation_risk_model.initialize()
        if self.analyst_config.get('enable_feature_engineering', True):
            self.feature_engineering_orchestrator = FeatureEngineeringOrchestrator(self.analyst_config.get('feature_engineering_orchestrator', {}))
            await self.feature_engineering_orchestrator.initialize()
        self.logger.info('Analysis components initialized')

    async def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for market data."""
        from .interfaces.event_bus import EventType
        self.event_bus.subscribe(EventType.MARKET_DATA_RECEIVED.value, self.analyze_market_data)
        self.logger.debug('Event subscriptions set up')

    @handles_errors(exceptions=(Exception,), default_return = None, context='market data analysis')
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult | None:
        """Analyze market data and return analysis result."""
        if not self.is_initialized or not self._validate_dependencies():
            self.tprint(initialization_error('Analyst not properly initialized'))
            return None
        try:
            self.is_analyzing = True
            self.logger.debug(f'Analyzing market data for {market_data.symbol}')
            analysis_result = await self._perform_comprehensive_analysis(market_data)
            if analysis_result:
                await self._store_analysis_result(analysis_result)
                if self.event_bus:
                    await self.event_bus.publish(EventType.ANALYSIS_COMPLETED.value, analysis_result)
            return analysis_result
        except Exception:
            self.tprint(failed('Analysis failed: {e}'))
            return None
        finally:
            self.is_analyzing = False

    async def _perform_comprehensive_analysis(self, market_data: MarketData) -> AnalysisResult | None:
        """Perform comprehensive market analysis using all available components."""
        try:
            features = {}
            technical_indicators = {}
            risk_metrics = {}
            support_resistance = {}
            market_regime = 'UNKNOWN'
            signal = 'HOLD'
            confidence = 0.0
            if self.dual_model_system:
                dual_result = await self.dual_model_system.analyze(market_data)
                if dual_result:
                    signal = dual_result.get('signal', 'HOLD')
                    confidence = dual_result.get('confidence', 0.0)
                    features.update(dual_result.get('features', {}))
            if self.market_health_analyzer:
                health_result = await self.market_health_analyzer.analyze(market_data)
                if health_result:
                    risk_metrics.update(health_result.get('risk_metrics', {}))
                    market_regime = health_result.get('market_regime', 'UNKNOWN')
            if self.liquidation_risk_model:
                liquidation_result = await self.liquidation_risk_model.analyze(market_data)
                if liquidation_result:
                    risk_metrics.update(liquidation_result.get('risk_metrics', {}))
            if self.feature_engineering_orchestrator:
                feature_result = await self.feature_engineering_orchestrator.analyze(market_data)
                if feature_result:
                    features.update(feature_result.get('features', {}))
                    technical_indicators.update(feature_result.get('technical_indicators', {}))
                    support_resistance.update(feature_result.get('support_resistance', {}))
            return AnalysisResult(timestamp = market_data.timestamp, symbol = market_data.symbol, confidence = confidence, signal = signal, features = features, technical_indicators = technical_indicators, market_regime = market_regime, support_resistance = support_resistance, risk_metrics = risk_metrics)
        except Exception:
            self.tprint(failed('Comprehensive analysis failed: {e}'))
            return None

    async def _store_analysis_result(self, analysis_result: AnalysisResult) -> None:
        """Store analysis result in history."""
        try:
            record = {'timestamp': analysis_result.timestamp.isoformat(), 'symbol': analysis_result.symbol, 'confidence': analysis_result.confidence, 'signal': analysis_result.signal, 'market_regime': analysis_result.market_regime}
            self.analysis_history.append(record)
            if len(self.analysis_history) > self.max_analysis_history:
                self.analysis_history = self.analysis_history[-self.max_analysis_history:]
        except Exception:
            self.tprint(failed('Failed to store analysis result: {e}'))

    async def get_historical_analysis(self, symbol: str, start_date: datetime, end_date: datetime) -> list[AnalysisResult]:
        """Get historical analysis results."""
        try:
            filtered_results = []
            for result in self.analysis_history:
                result_time = datetime.fromisoformat(result['timestamp'])
                if result.get('symbol') == symbol and start_date <= result_time <= end_date:
                    analysis_result = AnalysisResult(timestamp = result_time, symbol = result['symbol'], confidence = result['confidence'], signal = result['signal'], features={}, technical_indicators={}, market_regime = result['market_regime'], support_resistance={}, risk_metrics={})
                    filtered_results.append(analysis_result)
            return filtered_results
        except Exception:
            self.tprint(failed('Failed to get historical analysis: {e}'))
            return []

    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """Train analysis models."""
        try:
            self.logger.info('Training analysis models')
            success = True
            if self.dual_model_system:
                if not await self.dual_model_system.train(training_data):
                    success = False
            if self.liquidation_risk_model:
                if not await self.liquidation_risk_model.train(training_data):
                    success = False
            self.logger.info(f"Model training {('completed' if success else 'failed')}")
            return success
        except Exception:
            self.tprint(failed('Model training failed: {e}'))
            return False

    async def load_models(self, model_path: str) -> bool:
        """Load trained models."""
        try:
            self.logger.info(f'Loading models from {model_path}')
            success = True
            if self.dual_model_system:
                if not await self.dual_model_system.load_models(model_path):
                    success = False
            if self.liquidation_risk_model:
                if not await self.liquidation_risk_model.load_models(model_path):
                    success = False
            self.logger.info(f"Model loading {('completed' if success else 'failed')}")
            return success
        except Exception:
            self.tprint(failed('Model loading failed: {e}'))
            return False

    async def _start_component(self) -> None:
        """Start analyst-specific operations."""
        self.logger.info('Analyst component started')

    async def _stop_component(self) -> None:
        """Stop analyst-specific operations."""
        self.is_analyzing = False
        self.logger.info('Analyst component stopped')
