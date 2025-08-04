# src/analyst/analyst.py 

from typing import (
    Any,
)

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Analyst:
    """
    Analyst with comprehensive error handling and type safety.from datetime import datetime
from typing import (
    Any,
)

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Analyst:
    """
    Analyst with comprehensive error handling and type safety.
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

        
        # SR Analyzer integration
        self.sr_analyzer = None
        self.enable_sr_analysis: bool = self.analyst_config.get("enable_sr_analysis", True)
        
        # ML Confidence Predictor integration
        self.ml_confidence_predictor = None
        self.enable_ml_predictions: bool = self.analyst_config.get("enable_ml_predictions", True)

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
        
        # Initialize SR analyzer
        if self.enable_sr_analysis:
            await self._initialize_sr_analyzer()
            
        # Initialize ML Confidence Predictor
        if self.enable_ml_predictions:
            await self._initialize_ml_confidence_predictor()

        self.logger.info("✅ Analyst initialization completed successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst configuration loading",
    )
    async def _load_analyst_configuration(self) -> None:
        """Load analyst configuration."""
        # Set default analyst parameters
        self.analyst_config.setdefault("analysis_interval", 3600)
        self.analyst_config.setdefault("max_analysis_history", 100)
        self.analyst_config.setdefault("enable_technical_analysis", True)
        self.analyst_config.setdefault("enable_risk_analysis", True)
        self.analysis_interval = self.analyst_config["analysis_interval"]
        self.max_analysis_history = self.analyst_config["max_analysis_history"]
        self.enable_technical_analysis = self.analyst_config["enable_technical_analysis"]


        self.logger.info("Analyst configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate analyst configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate analysis interval
        if self.analysis_interval <= 0:
            self.logger.error("Invalid analysis interval")
            return False

        # Validate max analysis history
        if self.max_analysis_history <= 0:
            self.logger.error("Invalid max analysis history")
            return False

        # Validate that at least one analysis type is enabled
        if not any(
            [
                self.enable_technical_analysis,
                self.analyst_config.get("enable_risk_analysis", True),
            ],
        ):
            self.logger.error("At least one analysis type must be enabled")
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst modules initialization",
    )
    async def _initialize_analyst_modules(self) -> None:
        """Initialize analyst modules."""
        try:
            # Initialize technical analysis module
            if self.enable_technical_analysis:
                await self._initialize_technical_analysis()

            # Initialize risk analysis module
            if self.analyst_config.get("enable_risk_analysis", True):
                await self._initialize_risk_analysis()

            self.logger.info("Analyst modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing analyst modules: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="technical analysis initialization",
    )
    async def _initialize_technical_analysis(self) -> None:
        """Initialize technical analysis module."""
        try:
            # Initialize technical analysis components
            self.technical_analysis_components = {
                "price_analysis": True,
                "volume_analysis": True,
                "indicator_analysis": True,
                "pattern_analysis": True,
            }

            self.logger.info("Technical analysis module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing technical analysis: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk analysis initialization",
    )
    async def _initialize_risk_analysis(self) -> None:
        """Initialize risk analysis module."""
        try:
            # Initialize risk analysis components
            self.risk_analysis_components = {
                "volatility_analysis": True,
                "correlation_analysis": True,
                "drawdown_analysis": True,
                "risk_scoring": True,
            }

            self.logger.info("Risk analysis module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing risk analysis: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR analyzer initialization",
    )
    async def _initialize_sr_analyzer(self) -> None:
        """Initialize SR analyzer module."""
        try:
            from src.analyst.sr_analyzer import SRLevelAnalyzer
            
            self.sr_analyzer = SRLevelAnalyzer(self.config)
            await self.sr_analyzer.initialize()
            self.logger.info("✅ SR analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SR analyzer: {e}")
            self.sr_analyzer = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML confidence predictor module."""
        try:
            from src.analyst.ml_confidence_predictor import setup_ml_confidence_predictor
            
            self.ml_confidence_predictor = await setup_ml_confidence_predictor(self.config)
            if self.ml_confidence_predictor:
                await self.ml_confidence_predictor.initialize()
                self.logger.info("✅ ML confidence predictor initialized successfully")
            else:
                self.logger.error("❌ Failed to initialize ML confidence predictor")
                
        except Exception as e:
            self.logger.error(f"Error initializing ML confidence predictor: {e}")
            self.ml_confidence_predictor = None

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
        Execute analysis operations.

        Args:
            analysis_input: Analysis input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_analysis_inputs(analysis_input):
                return False

            self.is_analyzing = True
            self.logger.info("🔄 Starting analysis execution...")

            # Perform technical analysis
            if self.enable_technical_analysis:
                technical_results = await self._perform_technical_analysis(
                    analysis_input,
                )
                self.analysis_results["technical_analysis"] = technical_results


            # Perform risk analysis
            if self.analyst_config.get("enable_risk_analysis", True):
                risk_results = await self._perform_risk_analysis(analysis_input)
                self.analysis_results["risk_analysis"] = risk_results

            # Perform SR analysis
            if self.enable_sr_analysis and self.sr_analyzer:
                sr_results = await self._perform_sr_analysis(analysis_input)
                self.analysis_results["sr_analysis"] = sr_results

            # Perform ML confidence predictions
            if self.enable_ml_predictions and self.ml_confidence_predictor:
                ml_results = await self._perform_ml_predictions(analysis_input)
                self.analysis_results["ml_predictions"] = ml_results

            # Store analysis results
            await self._store_analysis_results()

            self.is_analyzing = False
            self.logger.info("✅ Analysis execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing analysis: {e}")
            self.is_analyzing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="analysis inputs validation",
    )
    def _validate_analysis_inputs(self, analysis_input: dict[str, Any]) -> bool:
        """
        Validate analysis inputs.

        Args:
            analysis_input: Analysis input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required analysis input fields
            required_fields = ["analysis_type", "symbol", "timestamp"]
            for field in required_fields:
                if field not in analysis_input:
                    self.logger.error(f"Missing required analysis input field: {field}")
                    return False

            # Validate data types
            if not isinstance(analysis_input["analysis_type"], str):
                self.logger.error("Invalid analysis type")
                return False

            if not isinstance(analysis_input["symbol"], str):
                self.logger.error("Invalid symbol")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating analysis inputs: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="technical analysis",
    )
    async def _perform_technical_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform technical analysis.

        Args:
            analysis_input: Analysis input dictionary

        Returns:
            dict[str, Any]: Technical analysis results
        """
        try:
            results = {}

            # Perform price analysis
            if self.technical_analysis_components.get("price_analysis", False):
                results["price_analysis"] = self._perform_price_analysis(analysis_input)

            # Perform volume analysis
            if self.technical_analysis_components.get("volume_analysis", False):
                results["volume_analysis"] = self._perform_volume_analysis(
                    analysis_input,
                )

            # Perform indicator analysis
            if self.technical_analysis_components.get("indicator_analysis", False):
                results["indicator_analysis"] = self._perform_indicator_analysis(
                    analysis_input,
                )

            # Perform pattern analysis
            if self.technical_analysis_components.get("pattern_analysis", False):
                results["pattern_analysis"] = self._perform_pattern_analysis(
                    analysis_input,
                )

            self.logger.info("Technical analysis completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing technical analysis: {e}")
            return {}



    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk analysis",
    )
    async def _perform_risk_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform risk analysis.

        Args:
            analysis_input: Analysis input dictionary

        Returns:
            dict[str, Any]: Risk analysis results
        """
        try:
            results = {}

            # Perform volatility analysis
            if self.risk_analysis_components.get("volatility_analysis", False):
                results["volatility_analysis"] = self._perform_volatility_analysis(
                    analysis_input,
                )

            # Perform correlation analysis
            if self.risk_analysis_components.get("correlation_analysis", False):
                results["correlation_analysis"] = self._perform_correlation_analysis(
                    analysis_input,
                )

            # Perform drawdown analysis
            if self.risk_analysis_components.get("drawdown_analysis", False):
                results["drawdown_analysis"] = self._perform_drawdown_analysis(
                    analysis_input,
                )

            # Perform risk scoring
            if self.risk_analysis_components.get("risk_scoring", False):
                results["risk_scoring"] = self._perform_risk_scoring(analysis_input)

            self.logger.info("Risk analysis completed")
            return results

        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return {}

    # Technical analysis methods
    def _perform_price_analysis(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform price analysis."""
        try:
            # Simulate price analysis
            return {
                "price_analysis_completed": True,
                "price_trend": "bullish",
                "support_levels": [100, 95, 90],
                "resistance_levels": [110, 115, 120],
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing price analysis: {e}")
            return {}

    def _perform_volume_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform volume analysis."""
        try:
            # Simulate volume analysis
            return {
                "volume_analysis_completed": True,
                "volume_trend": "increasing",
                "volume_ma": 1500000,
                "volume_ratio": 1.2,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing volume analysis: {e}")
            return {}

    def _perform_indicator_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform indicator analysis."""
        try:
            # Simulate indicator analysis
            return {
                "indicator_analysis_completed": True,
                "rsi": 65.5,
                "macd": "bullish",
                "bollinger_position": "upper",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing indicator analysis: {e}")
            return {}

    def _perform_pattern_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform pattern analysis."""
        try:
            # Simulate pattern analysis
            return {
                "pattern_analysis_completed": True,
                "patterns_found": ["double_top", "support_bounce"],
                "pattern_confidence": 0.85,
                "pattern_direction": "bearish",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing pattern analysis: {e}")
            return {}


    # Risk analysis methods
    def _perform_volatility_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform volatility analysis."""
        try:
            # Simulate volatility analysis
            return {
                "volatility_analysis_completed": True,
                "historical_volatility": 0.25,
                "implied_volatility": 0.28,
                "volatility_regime": "normal",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing volatility analysis: {e}")
            return {}

    def _perform_correlation_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform correlation analysis."""
        try:
            # Simulate correlation analysis
            return {
                "correlation_analysis_completed": True,
                "market_correlation": 0.65,
                "sector_correlation": 0.45,
                "correlation_trend": "decreasing",
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing correlation analysis: {e}")
            return {}

    def _perform_drawdown_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform drawdown analysis."""
        try:
            # Simulate drawdown analysis
            return {
                "drawdown_analysis_completed": True,
                "max_drawdown": 0.15,
                "current_drawdown": 0.05,
                "drawdown_duration": 30,
                "training_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error performing drawdown analysis: {e}")
            return {}

    def _perform_risk_scoring(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform risk scoring."""
        try:
            # Simulate risk scoring
            return {
                "risk_scoring_completed": True,
                "overall_risk_score": 0.35,
                "risk_category": "moderate",
                "risk_factors": ["volatility", "correlation"],
                "training_time": datetime.now().isoformat(),
            }
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
        Perform ML confidence predictions.

        Args:
            analysis_input: Analysis input data

        Returns:
            dict[str, Any]: ML prediction results
        """
        try:
            if not self.ml_confidence_predictor:
                return {"error": "ML confidence predictor not initialized"}

            # Extract market data and current price
            market_data = analysis_input.get("market_data", {})
            current_price = analysis_input.get("current_price", 0.0)

            if not market_data or current_price <= 0:
                return {"error": "Invalid market data or current price"}

            # Generate ML predictions
            predictions = await self.ml_confidence_predictor.predict_confidence_table(
                market_data, current_price
            )

            if predictions:
                return {
                    "status": "success",
                    "predictions": predictions,
                    "timestamp": datetime.now(),
                }
            else:
                return {"error": "Failed to generate ML predictions"}

        except Exception as e:
            self.logger.error(f"Error performing ML predictions: {e}")
            return {"error": f"ML prediction error: {e}"}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR analysis",
    )
    async def _perform_sr_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform SR analysis using SR analyzer."""
        try:
            if not self.sr_analyzer:
                self.logger.warning("SR analyzer not initialized")
                return {}

            # Get market data from analysis input
            market_data = analysis_input.get("market_data", {})
            if not market_data:
                self.logger.warning("No market data provided for SR analysis")
                return {}

            # Convert market data to DataFrame if needed
            if isinstance(market_data, dict):
                import pandas as pd
                df = pd.DataFrame([market_data])
            else:
                df = market_data

            # Perform SR analysis
            sr_results = await self.sr_analyzer.analyze(df)
            
            if sr_results:
                return {
                    "support_levels": sr_results.get("support_levels", []),
                    "resistance_levels": sr_results.get("resistance_levels", []),
                    "support_confidence": sr_results.get("support_confidence", 0.0),
                    "resistance_confidence": sr_results.get("resistance_confidence", 0.0),
                    "analysis_time": sr_results.get("analysis_time"),
                    "data_points_analyzed": sr_results.get("data_points_analyzed", 0),
                    "lookback_period": sr_results.get("lookback_period", 0),
                }
            else:
                return {}

        except Exception as e:
            self.logger.error(f"Error performing SR analysis: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis results storage",
    )
    async def _store_analysis_results(self) -> None:
        """Store analysis results."""
        try:
            # Add timestamp
            self.analysis_results["timestamp"] = datetime.now().isoformat()

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
            analysis_type: Optional analysis type filter

        Returns:
            dict[str, Any]: Analysis results
        """
        try:
            if analysis_type:
                return self.analysis_results.get(analysis_type, {})
            return self.analysis_results.copy()

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
            limit: Optional limit on number of records

        Returns:
            list[dict[str, Any]]: Analysis history
        """
        try:
            history = self.analysis_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.error(f"Error getting analysis history: {e}")
            return []

    def get_analysis_status(self) -> dict[str, Any]:
        """
        Get analysis status information.

        Returns:
            dict[str, Any]: Analysis status
        """
        return {
            "is_analyzing": self.is_analyzing,
            "analysis_interval": self.analysis_interval,
            "max_analysis_history": self.max_analysis_history,
            "enable_technical_analysis": self.enable_technical_analysis,
            "enable_risk_analysis": self.analyst_config.get(
                "enable_risk_analysis",
                True,
            ),
            "analysis_history_count": len(self.analysis_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="analyst cleanup",
    )
    async def stop(self) -> None:
        """Stop the analyst."""
        self.logger.info("🛑 Stopping Analyst...")

        try:
            # Stop analyzing
            self.is_analyzing = False

            # Clear results
            self.analysis_results.clear()

            # Clear history
            self.analysis_history.clear()

            self.logger.info("✅ Analyst stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping analyst: {e}")


# Global analyst instance
analyst: Analyst | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="analyst setup",
)
async def setup_analyst(config: dict[str, Any] | None = None) -> Analyst | None:
    """
    Setup global analyst.

    Args:
        config: Optional configuration dictionary

    Returns:
        Analyst | None: Global analyst instance
    """
    try:
        global analyst

        if config is None:
            config = {
                "analyst": {
                    "analysis_interval": 3600,
                    "max_analysis_history": 100,
                    "enable_technical_analysis": True,
                    "enable_risk_analysis": True,
                },
            }

        # Create analyst
        analyst = Analyst(config)

        # Initialize analyst
        success = await analyst.initialize()
        if success:
            return analyst
        return None

    except Exception as e:
        print(f"Error setting up analyst: {e}")
        return None

