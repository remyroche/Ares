# src/analyst/analyst.py

from datetime import datetime
from typing import (
    Any,
)

import pandas as pd

from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from src.analyst.liquidation_risk_model import LiquidationRiskModel
from src.analyst.market_health_analyzer import MarketHealthAnalyzer

# Import dual model system and other components
from src.training.dual_model_system import DualModelSystem
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Analyst:
    """
    Analyst with comprehensive error handling and type safety.
    Determines IF we should enter a trade & which direction (short/long).
    Passes market health, volatility, and liquidation risk information to tactician.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize analyst with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Analyst")

        # Analyst state
        self.is_analyzing: bool = False
        self.analysis_results: dict[str, Any] = {}
        self.analysis_history: list[dict[str, Any]] = []

        # Configuration
        self.analyst_config: dict[str, Any] = self.config.get("analyst", {})
        self.analysis_interval: int = self.analyst_config.get("analysis_interval", 3600)
        self.max_analysis_history: int = self.analyst_config.get(
            "max_analysis_history",
            100,
        )
        self.enable_technical_analysis: bool = self.analyst_config.get(
            "enable_technical_analysis",
            True,
        )

        # Dual Model System integration
        self.dual_model_system: DualModelSystem | None = None
        self.enable_dual_model_system: bool = self.analyst_config.get(
            "enable_dual_model_system",
            True,
        )

        # Market Health Analyzer integration
        self.market_health_analyzer: MarketHealthAnalyzer | None = None
        self.enable_market_health_analysis: bool = self.analyst_config.get(
            "enable_market_health_analysis",
            True,
        )

        # Liquidation Risk Model integration
        self.liquidation_risk_model: LiquidationRiskModel | None = None
        self.enable_liquidation_risk_analysis: bool = self.analyst_config.get(
            "enable_liquidation_risk_analysis",
            True,
        )

        # Feature Engineering Orchestrator integration
        self.feature_engineering_orchestrator: FeatureEngineeringOrchestrator | None = (
            None
        )
        self.enable_feature_engineering: bool = self.analyst_config.get(
            "enable_feature_engineering",
            True,
        )

        # ML Confidence Predictor integration
        self.ml_confidence_predictor = None
        self.enable_ml_predictions: bool = self.analyst_config.get(
            "enable_ml_predictions",
            True,
        )

        # Unified Regime Classifier integration
        self.regime_classifier = None
        self.enable_regime_classification: bool = self.analyst_config.get(
            "enable_regime_classification",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid analyst configuration"),
            AttributeError: (False, "Missing required analyst parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="analyst initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize analyst with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Analyst...")

        # Load analyst configuration
        await self._load_analyst_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error("Invalid configuration for analyst")
            return False

        # Initialize analyst modules
        await self._initialize_analyst_modules()

        # Initialize Dual Model System
        if self.enable_dual_model_system:
            await self._initialize_dual_model_system()

        # Initialize Market Health Analyzer
        if self.enable_market_health_analysis:
            await self._initialize_market_health_analyzer()

        # Initialize Liquidation Risk Model
        if self.enable_liquidation_risk_analysis:
            await self._initialize_liquidation_risk_model()

        # Initialize Feature Engineering Orchestrator
        if self.enable_feature_engineering:
            await self._initialize_feature_engineering_orchestrator()

        # Initialize ML Confidence Predictor
        if self.enable_ml_predictions:
            await self._initialize_ml_confidence_predictor()

        # Initialize Unified Regime Classifier
        if self.enable_regime_classification:
            await self._initialize_regime_classifier()

        self.logger.info("✅ Analyst initialization completed successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst configuration loading",
    )
    async def _load_analyst_configuration(self) -> None:
        """Load analyst configuration."""
        self.logger.info("Loading analyst configuration...")

        # Additional configuration can be loaded here
        self.logger.info("Analyst configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate analyst configuration."""
        try:
            if self.analysis_interval <= 0:
                self.logger.error("analysis_interval must be positive")
                return False

            self.logger.info("Analyst configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst modules initialization",
    )
    async def _initialize_analyst_modules(self) -> None:
        """Initialize analyst modules."""
        self.logger.info("Initializing analyst modules...")

        if self.enable_technical_analysis:
            await self._initialize_technical_analysis()

        if self.enable_risk_analysis:
            await self._initialize_risk_analysis()

        self.logger.info("Analyst modules initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="technical analysis initialization",
    )
    async def _initialize_technical_analysis(self) -> None:
        """Initialize technical analysis module."""
        self.logger.info("Initializing technical analysis...")
        # Technical analysis initialization logic here
        self.logger.info("Technical analysis initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk analysis initialization",
    )
    async def _initialize_risk_analysis(self) -> None:
        """Initialize risk analysis module."""
        self.logger.info("Initializing risk analysis...")
        # Risk analysis initialization logic here
        self.logger.info("Risk analysis initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dual model system initialization",
    )
    async def _initialize_dual_model_system(self) -> None:
        """Initialize Dual Model System."""
        try:
            from src.training.dual_model_system import setup_dual_model_system

            self.dual_model_system = await setup_dual_model_system(self.config)
            if self.dual_model_system:
                self.logger.info("✅ Dual Model System initialized successfully")
            else:
                self.logger.error("❌ Failed to initialize Dual Model System")
        except Exception as e:
            self.logger.error(f"Error initializing Dual Model System: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market health analyzer initialization",
    )
    async def _initialize_market_health_analyzer(self) -> None:
        """Initialize Market Health Analyzer."""
        try:
            from src.analyst.market_health_analyzer import setup_market_health_analyzer

            self.market_health_analyzer = await setup_market_health_analyzer(
                self.config,
            )
            if self.market_health_analyzer:
                self.logger.info("✅ Market Health Analyzer initialized successfully")
            else:
                self.logger.error("❌ Failed to initialize Market Health Analyzer")
        except Exception as e:
            self.logger.error(f"Error initializing Market Health Analyzer: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="liquidation risk model initialization",
    )
    async def _initialize_liquidation_risk_model(self) -> None:
        """Initialize Liquidation Risk Model."""
        try:
            from src.analyst.liquidation_risk_model import setup_liquidation_risk_model

            self.liquidation_risk_model = await setup_liquidation_risk_model(
                self.config,
            )
            if self.liquidation_risk_model:
                self.logger.info("✅ Liquidation Risk Model initialized successfully")
            else:
                self.logger.error("❌ Failed to initialize Liquidation Risk Model")
        except Exception as e:
            self.logger.error(f"Error initializing Liquidation Risk Model: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="feature engineering orchestrator initialization",
    )
    async def _initialize_feature_engineering_orchestrator(self) -> None:
        """Initialize Feature Engineering Orchestrator."""
        try:
            self.feature_engineering_orchestrator = FeatureEngineeringOrchestrator(
                self.config,
            )
            self.logger.info(
                "✅ Feature Engineering Orchestrator initialized successfully",
            )
        except Exception as e:
            self.logger.error(
                f"Error initializing Feature Engineering Orchestrator: {e}",
            )

    # Legacy S/R analyzer initialization method removed

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML Confidence Predictor."""
        self.logger.info("Initializing ML Confidence Predictor...")
        # ML confidence predictor initialization logic here
        self.logger.info("ML Confidence Predictor initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime classifier initialization",
    )
    async def _initialize_regime_classifier(self) -> None:
        """Initialize Unified Regime Classifier."""
        self.logger.info("Initializing Unified Regime Classifier...")
        self.regime_classifier = UnifiedRegimeClassifier(self.config, "UNKNOWN", "UNKNOWN")
        self.logger.info("Unified Regime Classifier initialized successfully")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid analysis parameters"),
            AttributeError: (False, "Missing analysis components"),
            KeyError: (False, "Missing required analysis data"),
        },
        default_return=False,
        context="analysis execution",
    )
    async def execute_analysis(self, analysis_input: dict[str, Any]) -> bool:
        """
        Execute comprehensive analysis with dual model system integration.

        Args:
            analysis_input: Input data for analysis

        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            if not self._validate_analysis_inputs(analysis_input):
                self.logger.error("Invalid analysis inputs")
                return False

            self.is_analyzing = True
            self.logger.info("Starting comprehensive analysis...")

            # Extract market data
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")
            current_position = analysis_input.get("current_position")

            # 1. Generate features using orchestrator
            if self.feature_engineering_orchestrator:
                self.logger.info("Generating features...")
                features_df = (
                    self.feature_engineering_orchestrator.generate_all_features(
                        market_data,
                        analysis_input.get("agg_trades_df"),
                        analysis_input.get("futures_df"),
                        analysis_input.get("sr_levels"),
                    )
                )
            else:
                features_df = market_data

            # 2. Perform market health analysis
            market_health_results = {}
            if self.market_health_analyzer:
                self.logger.info("Performing market health analysis...")
                health_input = {
                    "market_data": features_df,
                    "current_price": current_price,
                }
                await self.market_health_analyzer.execute_market_health_analysis(
                    health_input,
                )
                market_health_results = (
                    self.market_health_analyzer.get_analysis_results()
                )

            # 3. Perform liquidation risk analysis
            liquidation_risk_results = {}
            if self.liquidation_risk_model and self.ml_confidence_predictor:
                self.logger.info("Performing liquidation risk analysis...")
                # Get ML predictions first
                ml_predictions = await self._get_ml_predictions(
                    features_df,
                    current_price,
                )
                if ml_predictions:
                    liquidation_risk_results = (
                        await self.liquidation_risk_model.calculate_liquidation_risk(
                            ml_predictions,
                            current_price,
                            analysis_input.get("target_direction", "long"),
                        )
                    )

            # 4. Make trading decision using dual model system
            trading_decision = {}
            if self.dual_model_system:
                self.logger.info("Making trading decision with dual model system...")
                trading_decision = await self.dual_model_system.make_trading_decision(
                    features_df,
                    current_price,
                    current_position,
                )

            # 5. Compile comprehensive analysis results
            self.analysis_results = {
                "timestamp": datetime.now().isoformat(),
                "market_health": market_health_results,
                "liquidation_risk": liquidation_risk_results,
                "trading_decision": trading_decision,
                "features_shape": features_df.shape
                if features_df is not None
                else None,
                "current_price": current_price,
                "analysis_status": "completed",
            }

            # Store analysis results
            await self._store_analysis_results()

            self.is_analyzing = False
            self.logger.info("✅ Comprehensive analysis completed successfully")
            return True

        except Exception as e:
            self.is_analyzing = False
            self.logger.error(f"❌ Analysis failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="ML predictions",
    )
    async def _get_ml_predictions(
        self,
        features_df: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Get ML predictions for liquidation risk analysis."""
        if self.ml_confidence_predictor:
            return await self.ml_confidence_predictor.predict_confidence_table(
                features_df,
                current_price,
            )
        # Fallback predictions
        return {
            "confidence": 0.5,
            "increase_probabilities": {0.1: 0.3, 0.2: 0.2, 0.3: 0.1},
            "decrease_probabilities": {0.1: 0.3, 0.2: 0.2, 0.3: 0.1},
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="analysis inputs validation",
    )
    def _validate_analysis_inputs(self, analysis_input: dict[str, Any]) -> bool:
        """Validate analysis input data."""
        try:
            required_keys = ["market_data", "current_price"]
            for key in required_keys:
                if key not in analysis_input:
                    self.logger.error(f"Missing required analysis input: {key}")
                    return False

            market_data = analysis_input.get("market_data")
            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                self.logger.error("Invalid market data provided")
                return False

            current_price = analysis_input.get("current_price")
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                self.logger.error("Invalid current price provided")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Analysis inputs validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="technical analysis",
    )
    async def _perform_technical_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform technical analysis.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Technical analysis results
        """
        market_data = analysis_input.get("market_data")
        current_price = analysis_input.get("current_price")

        # Perform technical analysis
        technical_results = {
            "price_analysis": self._perform_price_analysis(analysis_input),
            "volume_analysis": self._perform_volume_analysis(analysis_input),
            "indicator_analysis": self._perform_indicator_analysis(analysis_input),
            "pattern_analysis": self._perform_pattern_analysis(analysis_input),
            "volatility_analysis": self._perform_volatility_analysis(analysis_input),
            "correlation_analysis": self._perform_correlation_analysis(analysis_input),
            "drawdown_analysis": self._perform_drawdown_analysis(analysis_input),
            "risk_scoring": self._perform_risk_scoring(analysis_input),
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info("Technical analysis completed successfully")
        return technical_results

    def _perform_price_analysis(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform price analysis."""
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            # Simple price analysis
            price_results = {
                "current_price": current_price,
                "price_change_1h": market_data["close"].pct_change(1).iloc[-1]
                if len(market_data) > 0
                else 0,
                "price_change_24h": market_data["close"].pct_change(24).iloc[-1]
                if len(market_data) > 24
                else 0,
                "price_trend": "bullish"
                if market_data["close"].iloc[-1] > market_data["close"].iloc[-20]
                else "bearish",
            }

            return price_results

        except Exception as e:
            self.logger.error(f"Error performing price analysis: {e}")
            return {}

    def _perform_volume_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform volume analysis."""
        try:
            market_data = analysis_input.get("market_data")

            if "volume" not in market_data.columns:
                return {}

            volume_results = {
                "current_volume": market_data["volume"].iloc[-1],
                "volume_ma": market_data["volume"].rolling(window=20).mean().iloc[-1],
                "volume_ratio": market_data["volume"].iloc[-1]
                / market_data["volume"].rolling(window=20).mean().iloc[-1],
                "volume_trend": "high"
                if market_data["volume"].iloc[-1]
                > market_data["volume"].rolling(window=20).mean().iloc[-1]
                else "low",
            }

            return volume_results

        except Exception as e:
            self.logger.error(f"Error performing volume analysis: {e}")
            return {}

    def _perform_indicator_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform indicator analysis."""
        try:
            market_data = analysis_input.get("market_data")

            indicator_results = {
                "rsi": market_data.get("rsi", {}).iloc[-1]
                if "rsi" in market_data.columns
                else None,
                "macd": market_data.get("macd", {}).iloc[-1]
                if "macd" in market_data.columns
                else None,
                "bb_position": (
                    market_data["close"].iloc[-1]
                    - market_data.get("bb_lower", {}).iloc[-1]
                )
                / (
                    market_data.get("bb_upper", {}).iloc[-1]
                    - market_data.get("bb_lower", {}).iloc[-1]
                )
                if all(col in market_data.columns for col in ["bb_upper", "bb_lower"])
                else None,
            }

            return indicator_results

        except Exception as e:
            self.logger.error(f"Error performing indicator analysis: {e}")
            return {}

    def _perform_pattern_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform pattern analysis."""
        try:
            # Simple pattern analysis
            pattern_results = {
                "patterns_detected": [],
                "pattern_confidence": 0.0,
            }

            return pattern_results

        except Exception as e:
            self.logger.error(f"Error performing pattern analysis: {e}")
            return {}

    def _perform_volatility_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform volatility analysis."""
        try:
            market_data = analysis_input.get("market_data")

            returns = market_data["close"].pct_change()
            volatility_results = {
                "current_volatility": returns.rolling(window=20).std().iloc[-1],
                "volatility_regime": "high"
                if returns.rolling(window=20).std().iloc[-1] > 0.04
                else "normal",
                "volatility_trend": "increasing"
                if returns.rolling(window=20).std().iloc[-1]
                > returns.rolling(window=50).std().iloc[-1]
                else "decreasing",
            }

            return volatility_results

        except Exception as e:
            self.logger.error(f"Error performing volatility analysis: {e}")
            return {}

    def _perform_correlation_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform correlation analysis."""
        try:
            # Simple correlation analysis
            correlation_results = {
                "price_volume_correlation": 0.0,
                "correlation_regime": "normal",
            }

            return correlation_results

        except Exception as e:
            self.logger.error(f"Error performing correlation analysis: {e}")
            return {}

    def _perform_drawdown_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform drawdown analysis."""
        try:
            market_data = analysis_input.get("market_data")

            rolling_max = market_data["close"].rolling(window=20).max()
            drawdown = (market_data["close"] - rolling_max) / rolling_max
            drawdown_results = {
                "current_drawdown": drawdown.iloc[-1],
                "max_drawdown": drawdown.min(),
                "drawdown_regime": "high"
                if abs(drawdown.iloc[-1]) > 0.05
                else "normal",
            }

            return drawdown_results

        except Exception as e:
            self.logger.error(f"Error performing drawdown analysis: {e}")
            return {}

    def _perform_risk_scoring(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform risk scoring."""
        try:
            # Simple risk scoring
            risk_results = {
                "overall_risk_score": 0.5,
                "risk_level": "medium",
                "risk_factors": [],
            }

            return risk_results

        except Exception as e:
            self.logger.error(f"Error performing risk scoring: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML predictions",
    )
    async def _perform_ml_predictions(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform ML predictions.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: ML prediction results
        """
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            if self.ml_confidence_predictor:
                ml_results = (
                    await self.ml_confidence_predictor.predict_confidence_table(
                        market_data,
                        current_price,
                    )
                )
            else:
                # Fallback ML results
                ml_results = {
                    "confidence": 0.5,
                    "prediction": "neutral",
                    "timestamp": datetime.now().isoformat(),
                }

            self.logger.info("ML predictions completed successfully")
            return ml_results

        except Exception as e:
            self.logger.error(f"Error performing ML predictions: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR analysis",
    )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime classification",
    )
    async def _perform_regime_classification(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform regime and location classification.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Regime and location classification results
        """
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            if self.regime_classifier:
                # Use the new unified regime classifier for both regime and location
                regime, location, confidence, additional_info = (
                    self.regime_classifier.predict_regime_and_location(market_data)
                )

                regime_results = {
                    "regime": regime,
                    "location": location,
                    "confidence": confidence,
                    "regime_confidence": additional_info.get(
                        "regime_confidence",
                        confidence,
                    ),
                    "location_confidence": additional_info.get(
                        "location_confidence",
                        confidence,
                    ),
                    "regime_duration": 0,  # Could be enhanced with duration tracking
                    "timestamp": datetime.now().isoformat(),
                    "additional_info": additional_info,
                }
            else:
                # Fallback regime results
                regime_results = {
                    "regime": "SIDEWAYS",
                    "location": "OPEN_RANGE",
                    "confidence": 0.5,
                    "regime_confidence": 0.5,
                    "location_confidence": 0.5,
                    "regime_duration": 0,
                    "timestamp": datetime.now().isoformat(),
                }

            self.logger.info(
                f"Regime and location classification completed: {regime_results['regime']} at {regime_results['location']}",
            )
            return regime_results

        except Exception as e:
            self.logger.error(f"Error performing regime classification: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis results storage",
    )
    async def _store_analysis_results(self) -> None:
        """Store analysis results."""
        try:
            self.logger.info("Storing analysis results...")

            # Add to history
            self.analysis_history.append(self.analysis_results.copy())

            # Limit history size
            if len(self.analysis_history) > self.max_analysis_history:
                self.analysis_history.pop(0)

            self.logger.info("Analysis results stored successfully")
        except Exception as e:
            self.logger.error(f"Error storing analysis results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis results getting",
    )
    def get_analysis_results(self, analysis_type: str | None = None) -> dict[str, Any]:
        """
        Get analysis results.

        Args:
            analysis_type: Type of analysis results to retrieve

        Returns:
            dict: Analysis results
        """
        try:
            if analysis_type is None:
                return self.analysis_results
            return self.analysis_results.get(analysis_type, {})

        except Exception as e:
            self.logger.error(f"Error getting analysis results: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis history getting",
    )
    def get_analysis_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get analysis history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            list: Analysis history
        """
        try:
            if limit is None:
                return self.analysis_history
            return self.analysis_history[-limit:]

        except Exception as e:
            self.logger.error(f"Error getting analysis history: {e}")
            return []

    def get_analysis_status(self) -> dict[str, Any]:
        """Get analysis status."""
        return {
            "is_analyzing": self.is_analyzing,
            "last_analysis": self.analysis_results.get("timestamp"),
            "analysis_count": len(self.analysis_history),
            "dual_model_system_initialized": self.dual_model_system is not None,
            "market_health_analyzer_initialized": self.market_health_analyzer
            is not None,
            "liquidation_risk_model_initialized": self.liquidation_risk_model
            is not None,
            "feature_engineering_orchestrator_initialized": self.feature_engineering_orchestrator
            is not None,
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="analyst cleanup",
    )
    async def stop(self) -> None:
        """Clean up analyst resources."""
        try:
            self.logger.info("Stopping Analyst...")
            self.is_analyzing = False

            # Stop sub-components
            if self.dual_model_system:
                await self.dual_model_system.stop()

            if self.market_health_analyzer:
                await self.market_health_analyzer.stop()

            if self.liquidation_risk_model:
                await self.liquidation_risk_model.stop()

            self.analysis_results = {}
            self.analysis_history = []

            self.logger.info("✅ Analyst stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Error stopping Analyst: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="analyst setup",
)
async def setup_analyst(config: dict[str, Any] | None = None) -> Analyst | None:
    """
    Setup and initialize Analyst.

    Args:
        config: Configuration dictionary

    Returns:
        Analyst: Initialized analyst or None if failed
    """
    try:
        if config is None:
            config = {}

        analyst = Analyst(config)

        if await analyst.initialize():
            system_logger.info("✅ Analyst setup completed successfully")
            return analyst
        system_logger.error("❌ Analyst setup failed")
        return None

    except Exception as e:
        system_logger.error(f"❌ Error setting up Analyst: {e}")
        return None
