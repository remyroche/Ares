from src.utils.tprint import tprint
import warnings

# VectorBT imports for native optimization
from src.vectorbt import (
    vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
    rolling_sum, rolling_apply, rolling_corr, rolling_cov, scale, rank, 
    zscore, winsorize, clip, quantile, VECTORBT_AVAILABLE
)
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# src/analyst/analyst.py

from datetime import datetime
import logging
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from src.analyst.liquidation_risk_model import LiquidationRiskModel
from src.analyst.liquidation_risk_model import setup_liquidation_risk_model
from src.analyst.market_health_analyzer import MarketHealthAnalyzer
from src.analyst.market_health_analyzer import setup_market_health_analyzer
from src.analyst.ml_confidence_predictor import MLConfidencePredictor
from src.core.decorators import handles_errors
from src.core.error_classes import execution_error
from src.core.error_classes import initialization_error
# Note: dual_model_system has been refactored into training steps
# Using training steps components instead
try:
    from src.training.steps.model_training import GeneralModelTrainer, AnalystModelTrainer
    TRAINING_STEPS_AVAILABLE = True
except ImportError:
    TRAINING_STEPS_AVAILABLE = False
    GeneralModelTrainer = None
    AnalystModelTrainer = None
# Note: compat module has been refactored, using enhanced_error_handler instead
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.compat import handle_specific_errors
from src.utils.logger import system_logger
from src.utils.lookahead_bias_detector import LookaheadBiasError
from src.utils.lookahead_bias_detector import get_global_detector
from src.utils.lookahead_bias_detector import validate_no_future_data
from src.utils.warning_symbols import failed
from src.utils.warning_symbols import initialization_error
# Live trading utilities
from src.utils.model_manager import ModelManager
from src.utils.performance_utils import PerformanceMonitor, global_monitor
from src.utils.unified_cache import cached
# Live trading validation
from src.utils.trading_decorators import validate_trading_inputs

# Import dual model system and other components

# Placeholder implementations for missing decorators and classes
def validate_data_quality(validation_level: str = "WARNING"):
    """Placeholder decorator for data quality validation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def traced(operation_name: str):
    """Placeholder decorator for tracing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

class UnifiedRegimeClassifierFractal:
    """Placeholder class for Unified Regime Classifier Fractal."""
    def __init__(self, config: dict, exchange: str):
        self.config = config
        self.exchange = exchange

if TYPE_CHECKING:
    pass

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
        self.logger = logging.getLogger(self.__class__.__name__)
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
        self.enable_risk_analysis: bool = self.analyst_config.get(
            "enable_risk_analysis",
            False,
        )

        # Triple Barrier Configuration
        self.triple_barrier_config: dict[str, Any] = self.analyst_config.get("triple_barrier", {})
        self.profit_take_multiplier: float = self.triple_barrier_config.get("profit_take_multiplier", 0.002)  # 0.2%
        self.stop_loss_multiplier: float = self.triple_barrier_config.get("stop_loss_multiplier", 0.001)  # 0.1%
        self.confidence_threshold: float = self.triple_barrier_config.get("confidence_threshold", 0.6)  # 60% threshold for green light

        # Dual Model System integration
        self.dual_model_system: GeneralModelTrainer | None = None
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

        # Live trading utilities
        self.model_manager: ModelManager | None = None
        self.selected_model: str | None = None
        self.model_cache: dict[str, Any] = {}

        # Performance monitoring for live trading
        self.performance_monitor: PerformanceMonitor | None = None
        self.global_monitor = global_monitor
        self.prediction_cache: dict[str, Any] = {}

        # ML Confidence Predictor integration
        self.ml_confidence_predictor = None
        self.enable_ml_predictions: bool = self.analyst_config.get(
            "enable_ml_predictions",
            True,
        )

        # Enhanced predictions from supervisor
        self.enable_enhanced_predictions: bool = self.analyst_config.get(
            "enable_enhanced_predictions",
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

        # Enhanced predictions are now handled by the supervisor
        # No local initialization needed

        # Initialize Unified Regime Classifier
        if self.enable_regime_classification:
            await self._initialize_regime_classifier()

        # Initialize live trading utilities
        await self._initialize_live_trading_utilities()

        # Initialize performance monitoring
        await self._initialize_performance_monitoring()

        self.logger.info("✅ Analyst initialization completed successfully")
        return True

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst configuration loading",
    )
    async def _load_analyst_configuration(self) -> None:
        """Load analyst configuration."""
        self.logger.info("Loading analyst configuration...")

        # Additional configuration can be loaded here
        self.logger.info("Analyst configuration loaded successfully")

    @handles_errors(
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

        except Exception:
            self.logger.exception("Configuration validation failed")
            return False

    @handles_errors(
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

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="technical analysis initialization",
    )
    async def _initialize_technical_analysis(self) -> None:
        """Initialize technical analysis module."""
        self.logger.info("Initializing technical analysis...")
        # Technical analysis initialization logic here
        self.logger.info("Technical analysis initialized successfully")

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk analysis initialization",
    )
    async def _initialize_risk_analysis(self) -> None:
        """Initialize risk analysis module."""
        self.logger.info("Initializing risk analysis...")
        # Risk analysis initialization logic here
        self.logger.info("Risk analysis initialized successfully")

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="dual model system initialization",
    )
    async def _initialize_dual_model_system(self) -> None:
        """Initialize Training Steps System."""
        try:
            if TRAINING_STEPS_AVAILABLE:
                # Initialize the new training steps components
                self.dual_model_system = GeneralModelTrainer(self.config)
                self.logger.info("✅ Training Steps System initialized successfully")
                self.logger.info("   📊 Using new training steps architecture")
            else:
                self.logger.error(failed("❌ Failed to initialize Training Steps System"))

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing Dual Model System: {e}"),
            )

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market health analyzer initialization",
    )
    async def _initialize_market_health_analyzer(self) -> None:
        """Initialize Market Health Analyzer."""
        try:

            self.market_health_analyzer = await setup_market_health_analyzer(
                self.config,
            )
            if self.market_health_analyzer:
                self.logger.info("✅ Market Health Analyzer initialized successfully")
            else:
                self.logger.error(failed("❌ Failed to initialize Market Health Analyzer"))

        except Exception as e:
            self.logger.error(
                initialization_error(
                    f"Error initializing Market Health Analyzer: {e}",
                ),
            )

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="liquidation risk model initialization",
    )
    async def _initialize_liquidation_risk_model(self) -> None:
        """Initialize Liquidation Risk Model."""
        try:

            self.liquidation_risk_model = await setup_liquidation_risk_model(
                self.config,
            )
            if self.liquidation_risk_model:
                self.logger.info("✅ Liquidation Risk Model initialized successfully")
            else:
                self.logger.error(failed("❌ Failed to initialize Liquidation Risk Model"))

        except ImportError as e:
            self.logger.error(
                failed(f"❌ Failed to import liquidation risk model: {e}")

            )
            self.liquidation_risk_model = None
        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing Liquidation Risk Model: {e}"),
            )

    @handles_errors(
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
            self.logger.exception(
                f"Error initializing Feature Engineering Orchestrator: {e}",
            )

    # Legacy S/R analyzer initialization method removed

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML Confidence Predictor."""
        self.logger.info("Initializing ML Confidence Predictor...")
        try:
            if MLConfidencePredictor is not None:
                self.ml_confidence_predictor = MLConfidencePredictor(self.config)
                self.logger.info("ML Confidence Predictor initialized successfully")
            else:
                self.logger.warning("MLConfidencePredictor not available; using fallback probabilities")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MLConfidencePredictor: {e}")

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime classifier initialization",
    )
    async def _initialize_regime_classifier(self) -> None:
        """Initialize Unified Regime Classifier (Fractal Location-based)."""
        self.logger.info("Initializing Fractal Location Classifier...")
        self.regime_classifier = UnifiedRegimeClassifierFractal(
            self.config,
            self.analyst_config.get("exchange", "UNKNOWN"),
            self.analyst_config.get("symbol", "UNKNOWN"),
        )
        # Initialize the classifier
        if await self.regime_classifier.initialize():
            self.logger.info("✅ Fractal Location Classifier initialized successfully")
        else:
            self.logger.error("❌ Failed to initialize Fractal Location Classifier")

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

            # 2b. Compute probabilities to hit price targets (upside and downside)
            price_target_probabilities = {}
            if self.enable_ml_predictions and features_df is not None:
                try:
                    price_target_probabilities = await self._get_price_target_probabilities(
                        features_df,
                        current_price,
                    )
                except Exception:
                    price_target_probabilities = {}

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

            # 5. Get enhanced predictions from supervisor if available
            enhanced_predictions = {}
            if self.enable_enhanced_predictions and hasattr(self, 'supervisor'):
                # Extract required parameters from analysis_input
                symbol = analysis_input.get("symbol", "UNKNOWN")
                exchange = analysis_input.get("exchange", "UNKNOWN")
                timeframe = analysis_input.get("timeframe", "1h")

                # Get location info from fractal classifier if available
                regime_info = {}
                if self.regime_classifier and features_df is not None:
                    try:
                        regime_info = await self.analyze_regime(features_df)
                    except Exception as e:
                        self.logger.warning(f"Failed to get location info: {e}")
                        regime_info = {"regime": "UNKNOWN", "confidence": 0.0}

                enhanced_predictions = await self.supervisor.get_analyst_predictions(
                    features_df, regime_info, symbol, exchange, timeframe
                )

            # 6. Compile comprehensive analysis results
            self.analysis_results = {
                "timestamp": datetime.now().isoformat(),
                "market_health": market_health_results,
                "liquidation_risk": liquidation_risk_results,
                "trading_decision": trading_decision,
                "enhanced_predictions": enhanced_predictions,
                "price_target_probabilities": price_target_probabilities,
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

        except Exception:
            self.is_analyzing = False
            self.logger.error(failed("❌ Analysis failed: {e}"))

            return False

    @handles_errors(
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

    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="price target probabilities",
    )
    async def _get_price_target_probabilities(
        self,
        features_df: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """Compute probabilities to hit specific price targets up and down.

        Returns structure:
            {
                "upside": {"0.1%": p, ...},
                "downside": {"0.1%": p, ...},
                "best_targets": {
                    "upside": {"target": "x%", "probability": p},
                    "downside": {"target": "y%", "probability": q}
                }
            }
        """
        # Prefer ML predictor if available
        if self.ml_confidence_predictor:
            table = await self.ml_confidence_predictor.predict_confidence_table(
                features_df,
                current_price,
            )
            if table:
                # Use the new extraction method for consistent formatting
                extracted = self._extract_price_target_probabilities(table)
                return extracted

        # Fallback: derive naive probabilities from volatility
        if "close" in features_df.columns and len(features_df) > 50:
            returns = features_df["close"].pct_change().dropna()
            vol = float(self._vectorbt_rolling_operation(returns, "std", 20).iloc[-1] or 0.0)
        else:
            vol = 0.01

        # Define default target ladder (percent values as strings)
        targets = [f"{x/10:.1f}%" for x in range(1, 21)]  # 0.1% .. 2.0%

        # Simple mapping: higher vol => higher chance to hit further targets, but cap at 1
        def prob_for(level_str: str) -> float:
            level = float(level_str.replace("%", "")) / 100.0
            base = min(1.0, max(0.05, (vol * 5) / max(level, 1e-6)))
            return float(np.clip(base, 0.0, 1.0))

        # Create fallback predictions in the same format as ML predictor
        price_target_confidences = {t: prob_for(t) for t in targets}
        adversarial_confidences = {t: prob_for(t) for t in targets}

        # Create a mock ML prediction result for extraction
        mock_ml_predictions = {
            "price_target_confidences": price_target_confidences,
            "adversarial_confidences": adversarial_confidences,
            "directional_analysis": {
                "bullish": 0.5,
                "bearish": 0.5,
                "neutral": 0.0,
                "primary_direction": "neutral",
                "confidence": 0.5
            },
            "model_status": "fallback",
            "timestamp": datetime.now().isoformat()
        }

        # Use the extraction method for consistent formatting
        return self._extract_price_target_probabilities(mock_ml_predictions)

    def _extract_price_target_probabilities(
        self,
        ml_predictions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract and consolidate price target probabilities from ML predictions.
        Implements triple barrier logic and ensures all probabilities sum to 1.

        Args:
            ml_predictions: ML prediction results from confidence predictor

        Returns:
            dict: Consolidated probability outputs with triple barrier analysis
        """
        try:
            if not ml_predictions:
                return {
                    "price_target_probabilities": {},
                    "adversarial_risk_probabilities": {},
                    "directional_analysis": {},
                    "triple_barrier_analysis": {},
                    "summary": {"status": "no_predictions"}
                }

            # Extract price target confidences
            price_target_confidences = ml_predictions.get("price_target_confidences", {})
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})
            directional_analysis = ml_predictions.get("directional_analysis", {})

            # Normalize probabilities to ensure they sum to 1
            normalized_probabilities = self._normalize_probabilities(
                price_target_confidences,
                adversarial_confidences
            )

            # Convert to probability format expected by the system
            price_target_probabilities = {}
            for target, confidence in normalized_probabilities["upside"].items():
                price_target_probabilities[target] = {
                    "probability": float(confidence),
                    "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
                }

            # Convert adversarial confidences to risk probabilities
            adversarial_risk_probabilities = {}
            for target, confidence in normalized_probabilities["downside"].items():
                adversarial_risk_probabilities[target] = {
                    "risk_probability": float(confidence),
                    "risk_level": "high" if confidence > 0.6 else "medium" if confidence > 0.3 else "low"
                }

            # Calculate triple barrier analysis
            triple_barrier_analysis = self._calculate_triple_barrier_analysis(
                normalized_probabilities["upside"],
                normalized_probabilities["downside"]
            )

            # Extract directional analysis with normalized probabilities
            directional_summary = {
                "bullish_probability": directional_analysis.get("bullish", 0.0),
                "bearish_probability": directional_analysis.get("bearish", 0.0),
                "neutral_probability": directional_analysis.get("neutral", 0.0),
                "primary_direction": directional_analysis.get("primary_direction", "neutral"),
                "confidence": directional_analysis.get("confidence", 0.0)
            }

            # Find best targets
            best_upside = max(normalized_probabilities["upside"].items(), key=lambda x: x[1]) if normalized_probabilities["upside"] else (None, 0.0)
            best_downside = max(normalized_probabilities["downside"].items(), key=lambda x: x[1]) if normalized_probabilities["downside"] else (None, 0.0)

            summary = {
                "status": "success",
                "model_status": ml_predictions.get("model_status", "unknown"),
                "total_targets": len(normalized_probabilities["upside"]),
                "total_risk_levels": len(normalized_probabilities["downside"]),
                "probability_sum": sum(normalized_probabilities["upside"].values()) + sum(normalized_probabilities["downside"].values()),
                "best_upside_target": {
                    "target": best_upside[0],
                    "probability": float(best_upside[1])
                } if best_upside[0] else None,
                "best_downside_risk": {
                    "target": best_downside[0],
                    "probability": float(best_downside[1])
                } if best_downside[0] else None,
                "green_light_decision": triple_barrier_analysis.get("green_light", False),
                "confidence_threshold_met": triple_barrier_analysis.get("threshold_met", False),
                "timestamp": ml_predictions.get("timestamp", datetime.now().isoformat())
            }

            return {
                "price_target_probabilities": price_target_probabilities,
                "adversarial_risk_probabilities": adversarial_risk_probabilities,
                "directional_analysis": directional_summary,
                "triple_barrier_analysis": triple_barrier_analysis,
                "summary": summary
            }

        except Exception as e:
            self.logger.error(f"Error extracting price target probabilities: {e}")
            return {
                "price_target_probabilities": {},
                "adversarial_risk_probabilities": {},
                "directional_analysis": {},
                "triple_barrier_analysis": {},
                "summary": {"status": "error", "error": str(e)}
            }

    def _normalize_probabilities(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """
        Normalize probabilities to ensure they sum to 1 across both directions.

        Args:
            price_target_confidences: Upside price target confidences
            adversarial_confidences: Downside risk confidences

        Returns:
            dict: Normalized probabilities for upside and downside
        """
        try:
            # Combine all probabilities
            all_probabilities = {}
            all_probabilities.update(price_target_confidences)
            all_probabilities.update(adversarial_confidences)

            if not all_probabilities:
                return {"upside": {}, "downside": {}}

            # Calculate total probability
            total_prob = sum(all_probabilities.values())

            if total_prob <= 0:
                # If no probabilities, distribute equally
                n_targets = len(price_target_confidences)
                n_risks = len(adversarial_confidences)
                total_items = n_targets + n_risks

                if total_items > 0:
                    equal_prob = 1.0 / total_items
                    normalized_upside = {k: equal_prob for k in price_target_confidences.keys()}
                    normalized_downside = {k: equal_prob for k in adversarial_confidences.keys()}
                else:
                    normalized_upside = {}
                    normalized_downside = {}
            else:
                # Normalize to sum to 1
                normalized_upside = {k: v / total_prob for k, v in price_target_confidences.items()}
                normalized_downside = {k: v / total_prob for k, v in adversarial_confidences.items()}

            return {
                "upside": normalized_upside,
                "downside": normalized_downside
            }

        except Exception as e:
            self.logger.error(f"Error normalizing probabilities: {e}")
            return {"upside": {}, "downside": {}}

    def _calculate_triple_barrier_analysis(
        self,
        upside_probabilities: dict[str, float],
        downside_probabilities: dict[str, float]
    ) -> dict[str, Any]:
        """
        Calculate triple barrier analysis for green light decision.

        Args:
            upside_probabilities: Normalized upside probabilities
            downside_probabilities: Normalized downside probabilities

        Returns:
            dict: Triple barrier analysis results
        """
        try:
            # Convert upper barrier to percentage string for comparison
            upper_barrier_pct = f"{self.profit_take_multiplier * 100:.1f}%"
            lower_barrier_pct = f"{self.stop_loss_multiplier * 100:.1f}%"

            # Calculate cumulative confidence for upper barrier and above
            cumulative_upper_confidence = 0.0
            upper_barrier_targets = []

            for target, prob in upside_probabilities.items():
                # Convert target string to float for comparison
                target_value = float(target.replace("%", ""))
                upper_barrier_value = self.profit_take_multiplier * 100

                if target_value >= upper_barrier_value:
                    cumulative_upper_confidence += prob
                    upper_barrier_targets.append({
                        "target": target,
                        "probability": prob,
                        "contribution": prob
                    })

            # Calculate cumulative confidence for lower barrier and below (adversarial)
            cumulative_lower_confidence = 0.0
            lower_barrier_targets = []

            for target, prob in downside_probabilities.items():
                # Convert target string to float for comparison
                target_value = float(target.replace("%", ""))
                lower_barrier_value = self.stop_loss_multiplier * 100

                if target_value >= lower_barrier_value:
                    cumulative_lower_confidence += prob
                    lower_barrier_targets.append({
                        "target": target,
                        "probability": prob,
                        "contribution": prob
                    })

            # Determine if confidence threshold is met
            threshold_met = cumulative_upper_confidence >= self.confidence_threshold

            # Directional decision logic
            # Compare upside vs downside confidence to determine direction
            upside_advantage = cumulative_upper_confidence - cumulative_lower_confidence
            directional_threshold = 0.15  # Minimum advantage for directional signal

            # Calculate risk-reward ratio
            risk_reward_ratio = (
                cumulative_upper_confidence / cumulative_lower_confidence
                if cumulative_lower_confidence > 0 else float('inf')
            )

            # Determine directional signal
            if threshold_met and upside_advantage > directional_threshold:
                # Strong long signal
                directional_signal = 1  # Long
                signal_strength = cumulative_upper_confidence
                direction = "LONG"
                decision_reasoning = self._get_directional_decision_reasoning(
                    cumulative_upper_confidence,
                    cumulative_lower_confidence,
                    threshold_met,
                    upside_advantage,
                    direction,
                    signal_strength
                )
            elif threshold_met and upside_advantage < -directional_threshold:
                # Strong short signal
                directional_signal = -1  # Short
                signal_strength = cumulative_lower_confidence
                direction = "SHORT"
                decision_reasoning = self._get_directional_decision_reasoning(
                    cumulative_upper_confidence,
                    cumulative_lower_confidence,
                    threshold_met,
                    upside_advantage,
                    direction,
                    signal_strength
                )
            else:
                # No clear directional signal
                directional_signal = 0  # Neutral/No signal
                signal_strength = 0.0
                direction = "NEUTRAL"
                decision_reasoning = self._get_directional_decision_reasoning(
                    cumulative_upper_confidence,
                    cumulative_lower_confidence,
                    threshold_met,
                    upside_advantage,
                    direction,
                    signal_strength
                )

            return {
                "upper_barrier_threshold": upper_barrier_pct,
                "lower_barrier_threshold": lower_barrier_pct,
                "confidence_threshold": self.confidence_threshold,
                "cumulative_upper_confidence": float(cumulative_upper_confidence),
                "cumulative_lower_confidence": float(cumulative_lower_confidence),
                "threshold_met": threshold_met,
                "upside_advantage": float(upside_advantage),
                "directional_threshold": directional_threshold,
                "directional_signal": directional_signal,
                "signal_strength": float(signal_strength),
                "direction": direction,
                "risk_reward_ratio": float(risk_reward_ratio),
                "upper_barrier_targets": upper_barrier_targets,
                "lower_barrier_targets": lower_barrier_targets,
                "decision_reasoning": decision_reasoning
            }

        except Exception as e:
            self.logger.error(f"Error calculating triple barrier analysis: {e}")
            return {
                "upper_barrier_threshold": f"{self.profit_take_multiplier * 100:.1f}%",
                "lower_barrier_threshold": f"{self.stop_loss_multiplier * 100:.1f}%",
                "confidence_threshold": self.confidence_threshold,
                "cumulative_upper_confidence": 0.0,
                "cumulative_lower_confidence": 0.0,
                "threshold_met": False,
                "upside_advantage": 0.0,
                "directional_threshold": 0.15,
                "directional_signal": 0,
                "signal_strength": 0.0,
                "direction": "NEUTRAL",
                "risk_reward_ratio": 0.0,
                "upper_barrier_targets": [],
                "lower_barrier_targets": [],
                "decision_reasoning": f"Error in calculation: {str(e)}"
            }

    def _get_directional_decision_reasoning(
        self,
        cumulative_upper_confidence: float,
        cumulative_lower_confidence: float,
        threshold_met: bool,
        upside_advantage: float,
        direction: str,
        signal_strength: float
    ) -> str:
        """
        Generate human-readable directional decision reasoning.

        Args:
            cumulative_upper_confidence: Cumulative confidence for upper barrier
            cumulative_lower_confidence: Cumulative confidence for lower barrier
            threshold_met: Whether confidence threshold is met
            upside_advantage: Difference between upside and downside confidence
            direction: The directional signal (LONG, SHORT, NEUTRAL)
            signal_strength: Strength of the directional signal

        Returns:
            str: Decision reasoning
        """
        if direction == "LONG":
            return (
                f"LONG SIGNAL: Strong upside confidence ({cumulative_upper_confidence:.1%}) "
                f"exceeds threshold ({self.confidence_threshold:.1%}) with significant advantage "
                f"({upside_advantage:.1%}) over downside risk ({cumulative_lower_confidence:.1%})"
            )
        elif direction == "SHORT":
            return (
                f"SHORT SIGNAL: Strong downside confidence ({cumulative_lower_confidence:.1%}) "
                f"exceeds threshold ({self.confidence_threshold:.1%}) with significant advantage "
                f"({abs(upside_advantage):.1%}) over upside potential ({cumulative_upper_confidence:.1%})"
            )
        else:
            if threshold_met:
                return (
                    f"NEUTRAL SIGNAL: Threshold met but insufficient directional advantage "
                    f"({upside_advantage:.1%}, need {0.15:.1%}). Upside: {cumulative_upper_confidence:.1%}, "
                    f"Downside: {cumulative_lower_confidence:.1%}"
                )
            else:
                return (
                    f"NEUTRAL SIGNAL: Insufficient confidence. Upside: {cumulative_upper_confidence:.1%}, "
                    f"Downside: {cumulative_lower_confidence:.1%}, need {self.confidence_threshold:.1%}"
                )

    def _get_decision_reasoning(
        self,
        cumulative_upper_confidence: float,
        cumulative_lower_confidence: float,
        threshold_met: bool,
        green_light: bool
    ) -> str:
        """
        Legacy method for backward compatibility - redirects to directional reasoning.
        """
        # Convert old binary green light to directional format
        if green_light:
            direction = "LONG"
            upside_advantage = cumulative_upper_confidence - cumulative_lower_confidence
        else:
            direction = "NEUTRAL"
            upside_advantage = 0.0

        return self._get_directional_decision_reasoning(
            cumulative_upper_confidence,
            cumulative_lower_confidence,
            threshold_met,
            upside_advantage,
            direction,
            cumulative_upper_confidence if green_light else 0.0
        )

    @handles_errors(
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
                    self.logger.error("Missing required analysis input: %s", key)
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

        except Exception:
            self.logger.exception("Analysis inputs validation failed")
            return False

    @handles_errors(
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
        analysis_input.get("market_data")
        analysis_input.get("current_price")

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

    @validate_data_quality(validation_level="WARNING")
    @traced("price_analysis")
    def _perform_price_analysis(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform price analysis."""
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            # Simple price analysis
            return {
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

        except Exception:
            self.logger.error("Error performing price analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("volume_analysis")
    def _perform_volume_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform volume analysis."""
        try:
            market_data = analysis_input.get("market_data")

            if "volume" not in market_data.columns:
                return {}

            volume_ma = market_data["volume"].rolling(window=20).mean().iloc[-1]
            return {
                "current_volume": market_data["volume"].iloc[-1],
                "volume_ma": volume_ma,
                "volume_ratio": market_data["volume"].iloc[-1] / volume_ma if volume_ma > 0 else 1.0,
                "volume_trend": "high" if market_data["volume"].iloc[-1] > volume_ma else "low",
            }

        except Exception:
            self.logger.error("Error performing volume analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("indicator_analysis")
    def _perform_indicator_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform indicator analysis."""
        try:
            market_data = analysis_input.get("market_data")

            return {
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

        except Exception:
            self.logger.error("Error performing indicator analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("pattern_analysis")
    def _perform_pattern_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform pattern analysis."""
        try:
            # Simple pattern analysis
            return {
                "patterns_detected": [],
                "pattern_confidence": 0.0,
            }

        except Exception:
            self.logger.error("Error performing pattern analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("volatility_analysis")
    def _perform_volatility_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform volatility analysis."""
        try:
            market_data = analysis_input.get("market_data")

            returns = market_data["close"].pct_change()
            return {
                "current_volatility": self._vectorbt_rolling_operation(returns, "std", 20).iloc[-1],
                "volatility_regime": "high"
                if self._vectorbt_rolling_operation(returns, "std", 20).iloc[-1] > 0.04
                else "normal",
                "volatility_trend": "increasing"
                if self._vectorbt_rolling_operation(returns, "std", 20).iloc[-1]
                > self._vectorbt_rolling_operation(returns, "std", 50).iloc[-1]
                else "decreasing",
            }

        except Exception:
            self.logger.error("Error performing volatility analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("correlation_analysis")
    def _perform_correlation_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform correlation analysis."""
        try:
            # Simple correlation analysis
            return {
                "price_volume_correlation": 0.0,
                "correlation_regime": "normal",
            }

        except Exception:
            self.logger.error("Error performing correlation analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("drawdown_analysis")
    def _perform_drawdown_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform drawdown analysis."""
        try:
            market_data = analysis_input.get("market_data")

            rolling_max = market_data["close"].rolling(window=20).max()
            drawdown = (market_data["close"] - rolling_max) / rolling_max
            return {
                "current_drawdown": drawdown.iloc[-1],
                "max_drawdown": drawdown.min(),
                "drawdown_regime": "high"
                if abs(drawdown.iloc[-1]) > 0.05
                else "normal",
            }

        except Exception:
            self.logger.error("Error performing drawdown analysis: {e}")

            return {}

    @validate_data_quality(validation_level="WARNING")
    @traced("risk_scoring")
    def _perform_risk_scoring(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform risk scoring."""
        try:
            # Simple risk scoring
            return {
                "overall_risk_score": 0.5,
                "risk_level": "medium",
                "risk_factors": [],
            }

        except Exception:
            self.logger.error("Error performing risk scoring: {e}")

            return {}

    @handles_errors(
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

        except Exception:
            self.logger.error("Error performing ML predictions: {e}")

            return {}

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR analysis",
    )
    @cached(ttl=60, key_func=lambda self, features_df: f"regime_analysis_{hash(str(features_df.values.tolist()))}")
    @global_monitor.track_function
    async def analyze_regime(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze location using fractal classification.
        This method is called by supervisor for regime info.
        """
        if not self.regime_classifier:
            self.logger.warning("Regime classifier not available")
            tprint("⚠️ Regime classifier not available")
            return {"regime": "UNKNOWN", "confidence": 0.0}

        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_timer("regime_analysis")

        self.logger.info("Starting regime analysis...")
        tprint("Starting regime analysis...")

        try:
            # Get fractal location classification
            location_result = await self.regime_classifier.classify_location(features_df)

            # Convert to regime info format expected by supervisor
            # Note: Actual regime is determined by HMM in training pipeline
            regime_info = {
                "regime": "LOCATION_BASED",  # Placeholder - actual regime from HMM
                "location": location_result.get("primary_location", "OPEN_RANGE"),
                "confidence": location_result.get("location_strength", 0.5),
                "action_bias": location_result.get("action_bias", "NEUTRAL"),
                "location_details": location_result.get("location_details", {}),
                "nearby_levels": location_result.get("nearby_levels", []),
                "fractal_analysis": location_result.get("fractal_analysis", {})
            }

            # End performance monitoring
            if self.performance_monitor:
                execution_time = self.performance_monitor.end_timer("regime_analysis")
                self.logger.info(f"Regime analysis completed in {execution_time:.3f}s")
                tprint(f"Regime analysis completed in {execution_time:.3f}s")

            self.logger.info(f"✅ Regime analysis completed: {regime_info.get('regime', 'UNKNOWN')}")
            tprint(f"✅ Regime analysis completed: {regime_info.get('regime', 'UNKNOWN')}")
            return regime_info

        except Exception as e:
            error_msg = f"Error in fractal location analysis: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")

            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("regime_analysis")

            return {"regime": "UNKNOWN", "confidence": 0.0}

    @handle_errors_with_tracking(
        context="model loading for live trading",
        log_level="INFO",
        print_errors=True
    )
    async def load_analyst_model(self) -> bool:
        """
        Load the single analyst model trained on various market conditions.

        Returns:
            bool: True if model loading successful
        """
        if not self.model_manager:
            error_msg = "Model Manager not available"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return False

        try:
            # Use the single analyst model trained on various market conditions
            model_name = "analyst_market_analysis_model"

            self.logger.info(f"Loading analyst model for live trading: {model_name}")
            tprint(f"Loading analyst model for live trading: {model_name}")

            # Check if model is available
            available_models = await self.model_manager.list_available_models()
            if model_name not in available_models:
                error_msg = f"Analyst model {model_name} not available for live trading"
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return False

            # Load and cache the model
            model = await self.model_manager.load_model(model_name)
            if model:
                self.selected_model = model_name
                self.model_cache[model_name] = model
                self.logger.info(f"✅ Analyst model loaded and cached: {model_name}")
                tprint(f"✅ Analyst model loaded and cached: {model_name}")
                return True
            else:
                error_msg = f"Failed to load analyst model: {model_name}"
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return False

        except Exception as e:
            error_msg = f"Error loading analyst model: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return False

    @cached(ttl=60, key_func=lambda self, data, model_name: f"prediction_{model_name}_{hash(str(data.tail(5).values.tolist()))}")
    @handle_errors_with_tracking(
        context="live trading prediction",
        log_level="INFO",
        print_errors=True
    )
    async def get_model_prediction(self, data: pd.DataFrame, model_name: str = None) -> dict[str, Any]:
        """
        Get prediction from selected pre-trained model for live trading.

        Args:
            data: Input data for prediction
            model_name: Specific model to use (defaults to selected model)

        Returns:
            dict: Model prediction results
        """
        if not self.model_manager:
            error_msg = "Model Manager not available"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

        model_name = model_name or self.selected_model
        if not model_name:
            error_msg = "No model selected for prediction"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

        try:
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_timer("model_prediction")

            self.logger.info(f"Getting prediction from model: {model_name}")
            tprint(f"Getting prediction from model: {model_name}")

            # Get model from cache or load it
            model = self.model_cache.get(model_name)
            if not model:
                model = await self.model_manager.load_model(model_name)
                if model:
                    self.model_cache[model_name] = model
                else:
                    error_msg = f"Failed to load model: {model_name}"
                    self.logger.error(error_msg)
                    tprint(f"❌ {error_msg}")
                    return {"error": error_msg}

            # Get prediction
            prediction = await self.model_manager.get_prediction(model, data)

            # End performance monitoring
            if self.performance_monitor:
                execution_time = self.performance_monitor.end_timer("model_prediction")
                self.logger.info(f"Model prediction completed in {execution_time:.3f}s")
                tprint(f"Model prediction completed in {execution_time:.3f}s")

            self.logger.info(f"✅ Prediction obtained from model: {model_name}")
            tprint(f"✅ Prediction obtained from model: {model_name}")
            return prediction

        except Exception as e:
            error_msg = f"Error getting prediction from model {model_name}: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")

            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("model_prediction")

            return {"error": error_msg}

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime classification",
    )
    async def _perform_regime_classification(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform fractal location classification.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Location classification results
        """
        try:
            market_data = analysis_input.get("market_data")

            if self.regime_classifier and market_data is not None:
                # Use fractal location classifier
                location_result = await self.regime_classifier.classify_location(market_data)

                # Format results for compatibility
                regime_results = {
                    "regime": "LOCATION_BASED",  # Actual regime comes from HMM
                    "location": location_result.get("primary_location", "OPEN_RANGE"),
                    "confidence": location_result.get("location_strength", 0.5),
                    "regime_confidence": 0.5,  # Regime confidence from HMM
                    "location_confidence": location_result.get("location_strength", 0.5),
                    "action_bias": location_result.get("action_bias", "NEUTRAL"),
                    "regime_duration": 0,
                    "timestamp": datetime.now().isoformat(),
                    "additional_info": {
                        "location_details": location_result.get("location_details", {}),
                        "nearby_levels": location_result.get("nearby_levels", []),
                        "fractal_locations": location_result.get("fractal_locations", {})
                    }
                }

                # Add location features for ML models
                if hasattr(self.regime_classifier, 'get_location_features'):
                    location_features = self.regime_classifier.get_location_features(location_result)
                    regime_results["location_features"] = location_features.to_dict()
            else:
                # Fallback results
                regime_results = {
                    "regime": "UNKNOWN",
                    "location": "OPEN_RANGE",
                    "confidence": 0.5,
                    "regime_confidence": 0.5,
                    "location_confidence": 0.5,
                    "action_bias": "NEUTRAL",
                    "regime_duration": 0,
                    "timestamp": datetime.now().isoformat(),
                }

            self.logger.info(
                f"Fractal location classification completed: {regime_results['location']} with confidence {regime_results['location_confidence']:.2f}",
            )
            return regime_results

        except Exception:
            self.logger.error("Error performing regime classification: {e}")

            return {}

    @handles_errors(
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
        except Exception:
            self.logger.error("Error storing analysis results: {e}")

    @handles_errors(
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

        except Exception:
            self.logger.error("Error getting analysis results: {e}")

            return {}

    @handles_errors(
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

        except Exception:
            self.logger.error("Error getting analysis history: {e}")

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

    # Enhanced predictions are now handled by the supervisor
    # No local methods needed

    @handle_errors_with_tracking(
        context="live trading utilities initialization",
        log_level="INFO",
        print_errors=True
    )
    async def _initialize_live_trading_utilities(self) -> bool:
        """Initialize live trading utilities."""
        try:
            self.logger.info("Initializing live trading utilities...")
            tprint("Initializing live trading utilities...")

            # Initialize Model Manager for model selection and loading
            self.model_manager = ModelManager()
            self.logger.info("✅ Model Manager initialized")
            tprint("✅ Model Manager initialized")

            # Load the single analyst model
            success = await self.load_analyst_model()
            if not success:
                self.logger.warning("⚠️ Failed to load analyst model during initialization")
                tprint("⚠️ Failed to load analyst model during initialization")

            # Initialize model cache
            self.model_cache = {}
            self.prediction_cache = {}
            self.logger.info("✅ Model and prediction caches initialized")
            tprint("✅ Model and prediction caches initialized")

            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing live trading utilities: {e}")
            tprint(f"❌ Error initializing live trading utilities: {e}")
            return False

    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance monitoring initialization",
    )
    async def _initialize_performance_monitoring(self) -> bool:
        """Initialize performance monitoring."""
        try:
            self.logger.info("Initializing performance monitoring...")

            # Initialize Performance Monitor
            self.performance_monitor = PerformanceMonitor()
            self.logger.info("✅ Performance Monitor initialized")

            # Enable global monitoring
            self.global_monitor.enable()
            self.logger.info("✅ Global monitoring enabled")

            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing performance monitoring: {e}")
            return False

    @validate_trading_inputs(required_columns=["timestamp", "price", "volume"])
    @handle_errors_with_tracking(
        context="live trading data validation",
        log_level="INFO",
        print_errors=True
    )
    async def validate_trading_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Validate live trading data for real-time analysis.

        Args:
            data: Live trading data to validate

        Returns:
            dict: Validation results
        """
        try:
            self.logger.info("Validating live trading data...")
            tprint("Validating live trading data...")

            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }

            # Check for required columns
            required_columns = ["timestamp", "price", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Missing required columns: {missing_columns}")

            # Check for recent data (within last 5 minutes)
            if "timestamp" in data.columns and not data.empty:
                latest_timestamp = data["timestamp"].max()
                current_time = pd.Timestamp.now()
                time_diff = (current_time - latest_timestamp).total_seconds()
                if time_diff > 300:  # 5 minutes
                    validation_results["warnings"].append(f"Data is {time_diff:.0f} seconds old")

            # Check for valid price data
            if "price" in data.columns:
                if data["price"].isna().any():
                    validation_results["is_valid"] = False
                    validation_results["errors"].append("Price data contains NaN values")
                if (data["price"] <= 0).any():
                    validation_results["is_valid"] = False
                    validation_results["errors"].append("Price data contains non-positive values")

            self.logger.info(f"✅ Live trading data validation completed: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
            tprint(f"✅ Live trading data validation completed: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
            return validation_results

        except Exception as e:
            error_msg = f"Error validating live trading data: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

    @handle_errors_with_tracking(
        context="HMM regime-based model coordination",
        log_level="INFO",
        print_errors=True
    )
    async def coordinate_with_hmm_regime(self, hmm_regime: str, regime_confidence: float) -> dict[str, Any]:
        """
        Coordinate model usage based on HMM regime detection.

        Args:
            hmm_regime: Detected HMM regime (e.g., "bull_market", "bear_market", "sideways")
            regime_confidence: Confidence in the regime detection

        Returns:
            dict: Coordination results and regime-specific parameters
        """
        if not self.model_manager or not self.selected_model:
            error_msg = "Model Manager or selected model not available"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

        try:
            self.logger.info(f"Coordinating with HMM regime: {hmm_regime} (confidence: {regime_confidence:.3f})")
            tprint(f"Coordinating with HMM regime: {hmm_regime} (confidence: {regime_confidence:.3f})")

            # Get the single model (trained on various market conditions)
            model = self.model_cache.get(self.selected_model)
            if not model:
                error_msg = f"Model {self.selected_model} not loaded in cache"
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return {"error": error_msg}

            # Configure regime-specific parameters for the same model
            regime_config = {
                "hmm_regime": hmm_regime,
                "regime_confidence": regime_confidence,
                "model_name": self.selected_model,
                "regime_parameters": {}
            }

            # Set regime-specific parameters based on HMM regime (15-25 regimes)
            # Parameters are optimized during training in final_parameters_optimization.py
            regime_config["regime_parameters"] = self._get_optimized_regime_parameters(hmm_regime, regime_confidence)

            self.logger.info(f"✅ HMM regime coordination completed: {hmm_regime}")
            tprint(f"✅ HMM regime coordination completed: {hmm_regime}")
            return regime_config

        except Exception as e:
            error_msg = f"Error coordinating with HMM regime: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

    def _get_optimized_regime_parameters(self, hmm_regime: str, regime_confidence: float) -> dict[str, Any]:
        """
        Get optimized regime-specific parameters from training optimization.

        Args:
            hmm_regime: Detected HMM regime (15-25 possible regimes)
            regime_confidence: Confidence in regime detection

        Returns:
            dict: Optimized parameters for the regime
        """
        try:
            # Load optimized parameters from training (final_parameters_optimization.py)
            # These parameters are optimized during training and stored in the model artifacts
            optimized_params = self._load_optimized_parameters_for_regime(hmm_regime)

            if optimized_params:
                # Apply confidence-based adjustments
                confidence_adjustment = 0.8 + (regime_confidence * 0.4)  # 0.8 to 1.2 range

                adjusted_params = {}
                for param_name, param_value in optimized_params.items():
                    if param_name in ["confidence_threshold", "analyst_confidence_threshold"]:
                        # Higher confidence = lower threshold (more aggressive)
                        adjusted_params[param_name] = param_value * (2.0 - confidence_adjustment)
                    elif param_name in ["lookback_period", "volatility_adjustment"]:
                        # Higher confidence = more stable parameters
                        adjusted_params[param_name] = param_value * confidence_adjustment
                    else:
                        adjusted_params[param_name] = param_value

                return adjusted_params
            else:
                # Fallback to default parameters if optimization not available
                return self._get_default_regime_parameters(hmm_regime, regime_confidence)

        except Exception as e:
            self.logger.error(f"Error getting optimized regime parameters: {e}")
            return self._get_default_regime_parameters(hmm_regime, regime_confidence)

    def _load_optimized_parameters_for_regime(self, hmm_regime: str) -> dict[str, Any] | None:
        """
        Load optimized parameters for a specific regime from training artifacts.

        Args:
            hmm_regime: HMM regime identifier

        Returns:
            dict: Optimized parameters or None if not found
        """
        try:
            # This would load from the optimized parameters saved during training
            # The parameters are optimized in final_parameters_optimization.py
            # and stored in model artifacts

            # For now, return None to use fallback parameters
            # In production, this would load from:
            # - Model artifacts
            # - Optimization results from final_parameters_optimization.py
            # - Regime-specific parameter files

            return None

        except Exception as e:
            self.logger.error(f"Error loading optimized parameters for regime {hmm_regime}: {e}")
            return None

    def _get_default_regime_parameters(self, hmm_regime: str, regime_confidence: float) -> dict[str, Any]:
        """
        Get default regime parameters as fallback.

        Args:
            hmm_regime: HMM regime identifier
            regime_confidence: Confidence in regime detection

        Returns:
            dict: Default parameters for the regime
        """
        # Base parameters that work across all regimes
        base_params = {
            "confidence_threshold": 0.6,
            "lookback_period": 20,
            "volatility_adjustment": 1.0,
            "analyst_confidence_threshold": 0.7
        }

        # Apply confidence-based adjustments
        confidence_adjustment = 0.8 + (regime_confidence * 0.4)

        adjusted_params = {}
        for param_name, param_value in base_params.items():
            if param_name in ["confidence_threshold", "analyst_confidence_threshold"]:
                adjusted_params[param_name] = param_value * (2.0 - confidence_adjustment)
            elif param_name in ["lookback_period", "volatility_adjustment"]:
                adjusted_params[param_name] = param_value * confidence_adjustment
            else:
                adjusted_params[param_name] = param_value

        return adjusted_params

    @handle_errors_with_tracking(
        context="analyst cleanup",
        log_level="INFO",
        print_errors=True
    )
    async def stop(self) -> None:
        """Clean up analyst resources."""
        try:
            self.logger.info("Stopping Analyst...")
            self.is_analyzing = False

            # Stop sub-components with enhanced error handling
            if self.dual_model_system:
                try:
                    await self.dual_model_system.stop()
                    self.logger.info("✅ Dual model system stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping dual model system: {e}")
                    tprint(f"❌ Error stopping dual model system: {e}")

            if self.market_health_analyzer:
                try:
                    await self.market_health_analyzer.stop()
                    self.logger.info("✅ Market health analyzer stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping market health analyzer: {e}")
                    tprint(f"❌ Error stopping market health analyzer: {e}")

            if self.liquidation_risk_model:
                try:
                    await self.liquidation_risk_model.stop()
                    self.logger.info("✅ Liquidation risk model stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping liquidation risk model: {e}")
                    tprint(f"❌ Error stopping liquidation risk model: {e}")

            # Clean up live trading utilities
            if self.model_manager:
                try:
                    # Clear model cache
                    self.model_cache.clear()
                    self.prediction_cache.clear()
                    self.logger.info("✅ Model and prediction caches cleared")
                    tprint("✅ Model and prediction caches cleared")
                except Exception as e:
                    self.logger.error(f"❌ Error cleaning up model caches: {e}")
                    tprint(f"❌ Error cleaning up model caches: {e}")

            if self.performance_monitor:
                try:
                    self.performance_monitor.stop()
                    self.logger.info("✅ Performance monitor stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping performance monitor: {e}")
                    tprint(f"❌ Error stopping performance monitor: {e}")

            self.analysis_results = {}
            self.analysis_history = []

            self.logger.info("✅ Analyst stopped successfully")
            tprint("✅ Analyst stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Error stopping Analyst: {e}")
            tprint(f"❌ Error stopping Analyst: {e}")
            raise
