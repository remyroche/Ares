# src/analyst/analyst.py

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from src.analyst.feature_engineering_orchestrator import \
    FeatureEngineeringOrchestrator
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

if TYPE_CHECKING:
    from src.analyst.liquidation_risk_model import LiquidationRiskModel
    from src.analyst.market_health_analyzer import MarketHealthAnalyzer
    from src.training.dual_model_system import DualModelSystem


class Analyst:
    """
    Minimal, syntactically-correct Analyst facade to unblock imports and flows.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("Analyst")

        analyst_cfg = config.get("analyst", {})
        self.analysis_interval: int = int(analyst_cfg.get("analysis_interval", 3600))
        self.max_analysis_history: int = int(
            analyst_cfg.get("max_analysis_history", 100)
        )
        self.enable_technical_analysis: bool = bool(
            analyst_cfg.get("enable_technical_analysis", True)
        )
        self.enable_dual_model_system: bool = bool(
            analyst_cfg.get("enable_dual_model_system", True)
        )
        self.enable_market_health_analysis: bool = bool(
            analyst_cfg.get("enable_market_health_analysis", True)
        )
        self.enable_liquidation_risk_analysis: bool = bool(
            analyst_cfg.get("enable_liquidation_risk_analysis", True)
        )
        self.enable_feature_engineering: bool = bool(
            analyst_cfg.get("enable_feature_engineering", True)
        )
        self.enable_ml_predictions: bool = bool(
            analyst_cfg.get("enable_ml_predictions", True)
        )
        self.enable_regime_classification: bool = bool(
            analyst_cfg.get("enable_regime_classification", True)
        )

        self.is_analyzing: bool = False
        self.analysis_results: dict[str, Any] = {}
        self.analysis_history: list[dict[str, Any]] = []

        self.dual_model_system: Optional[DualModelSystem] = None
        self.market_health_analyzer: Optional[MarketHealthAnalyzer] = None
        self.liquidation_risk_model: Optional[LiquidationRiskModel] = None
        self.feature_engineering_orchestrator: Optional[
            FeatureEngineeringOrchestrator
        ] = None
        self.regime_classifier: Optional[UnifiedRegimeClassifier] = None
        self.ml_confidence_predictor: Any = None

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="analyst initialization"
    )
    async def initialize(self) -> bool:
        self.logger.info("Initializing Analyst...")
        if not self._validate_configuration():
            self.logger.error("Invalid configuration for analyst")
            return False
        await self._initialize_modules()
        self.logger.info("Analyst initialization completed successfully")
        return True

    def _validate_configuration(self) -> bool:
        try:
            if self.analysis_interval <= 0:
                self.logger.error("analysis_interval must be positive")
                return False
            return True
        except Exception as e:
            self.logger.exception(f"Configuration validation failed: {e}")
            return False

    async def _initialize_modules(self) -> None:
        if self.enable_dual_model_system:
            try:
                from src.training.dual_model_system import \
                    setup_dual_model_system

                self.dual_model_system = await setup_dual_model_system(self.config)
                if self.dual_model_system:
                    self.logger.info("Dual Model System initialized successfully")
            except Exception as e:
                self.logger.exception(f"Error initializing Dual Model System: {e}")
        if self.enable_market_health_analysis:
            try:
                from src.analyst.market_health_analyzer import \
                    setup_market_health_analyzer

                self.market_health_analyzer = await setup_market_health_analyzer(
                    self.config
                )
                if self.market_health_analyzer:
                    self.logger.info("Market Health Analyzer initialized successfully")
            except Exception as e:
                self.logger.exception(f"Error initializing Market Health Analyzer: {e}")
        if self.enable_liquidation_risk_analysis:
            try:
                from src.analyst.liquidation_risk_model import \
                    setup_liquidation_risk_model

                self.liquidation_risk_model = await setup_liquidation_risk_model(
                    self.config
                )
                if self.liquidation_risk_model:
                    self.logger.info("Liquidation Risk Model initialized successfully")
            except Exception as e:
                self.logger.exception(f"Error initializing Liquidation Risk Model: {e}")
        if self.enable_feature_engineering:
            try:
                self.feature_engineering_orchestrator = FeatureEngineeringOrchestrator()
                await self.feature_engineering_orchestrator.initialize()
            except Exception as e:
                self.logger.exception(
                    f"Error initializing Feature Engineering Orchestrator: {e}"
                )
        if self.enable_regime_classification:
            try:
                self.regime_classifier = UnifiedRegimeClassifier(self.config)
                await self.regime_classifier.initialize()
            except Exception as e:
                self.logger.exception(
                    f"Error initializing Unified Regime Classifier: {e}"
                )

    @handle_errors(exceptions=(Exception,), default_return=None, context="analyst run")
    async def run_analysis(self, klines_df: pd.DataFrame) -> Optional[dict[str, Any]]:
        if klines_df is None or klines_df.empty:
            return None
        self.is_analyzing = True
        try:
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "rows": int(len(klines_df)),
            }
            self.analysis_history.append(result)
            self.analysis_history = self.analysis_history[-self.max_analysis_history :]
            self.analysis_results = result
            return result
        finally:
            self.is_analyzing = False
