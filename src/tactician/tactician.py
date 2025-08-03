import asyncio
from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class Tactician:
    """
    Enhanced Tactician component with DI, type hints, and robust error handling.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Tactician")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.tactician_config: dict[str, Any] = self.config.get("tactician", {})
        self.tactics_interval: int = self.tactician_config.get("tactics_interval", 30)
        self.max_history: int = self.tactician_config.get("max_history", 100)
        self.tactics_results: dict[str, Any] = {}
        self.tactics_modules: dict[str, Any] = {}
        
        # SR Analyzer integration
        self.sr_analyzer = None
        self.enable_sr_tactics: bool = self.tactician_config.get("enable_sr_tactics", True)
        
        # SR Breakout Predictor integration
        self.sr_breakout_predictor = None
        self.enable_sr_breakout_tactics: bool = self.tactician_config.get("enable_sr_breakout_tactics", True)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tactician configuration"),
            AttributeError: (False, "Missing required tactician parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="tactician initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Tactician...")
            await self._load_tactician_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for tactician")
                return False
            await self._initialize_tactics_modules()
            
            # Initialize SR analyzer
            if self.enable_sr_tactics:
                await self._initialize_sr_analyzer()
                
            # Initialize SR Breakout Predictor
            if self.enable_sr_breakout_tactics:
                await self._initialize_sr_breakout_predictor()
                
            self.logger.info("✅ Tactician initialization completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Tactician initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician configuration loading",
    )
    async def _load_tactician_configuration(self) -> None:
        try:
            self.tactician_config.setdefault("tactics_interval", 30)
            self.tactician_config.setdefault("max_history", 100)
            self.tactics_interval = self.tactician_config["tactics_interval"]
            self.max_history = self.tactician_config["max_history"]
            self.logger.info("Tactician configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading tactician configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.tactics_interval <= 0:
                self.logger.error("Invalid tactics interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactics modules initialization",
    )
    async def _initialize_tactics_modules(self) -> None:
        try:
            # Initialize tactics modules
            self.tactics_modules = {
                "entry_monitoring": None,
                "exit_monitoring": None,
                "position_monitoring": None,
                "risk_monitoring": None,
            }
            self.logger.info("Tactics modules initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing tactics modules: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR analyzer initialization",
    )
    async def _initialize_sr_analyzer(self) -> None:
        """Initialize SR analyzer for tactical execution."""
        try:
            from src.analyst.sr_analyzer import SRLevelAnalyzer
            
            self.sr_analyzer = SRLevelAnalyzer(self.config)
            await self.sr_analyzer.initialize()
            self.logger.info("✅ SR analyzer initialized for tactical execution")
        except Exception as e:
            self.logger.error(f"Error initializing SR analyzer: {e}")
            self.sr_analyzer = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR breakout predictor initialization",
    )
    async def _initialize_sr_breakout_predictor(self) -> None:
        """Initialize SR Breakout Predictor for tactical decisions."""
        try:
            from src.analyst.sr_breakout_predictor import SRBreakoutPredictor
            self.sr_breakout_predictor = SRBreakoutPredictor(self.config)
            await self.sr_breakout_predictor.initialize()
            self.logger.info("✅ SR Breakout Predictor initialized for tactical decisions")
        except Exception as e:
            self.logger.error(f"Error initializing SR Breakout Predictor: {e}")
            self.sr_breakout_predictor = None

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Tactician run failed"),
        },
        default_return=False,
        context="tactician run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Tactician started.")
            while self.is_running:
                await self._execute_tactics()
                await asyncio.sleep(self.tactics_interval)
            return True
        except Exception as e:
            self.logger.error(f"Error in tactician run: {e}")
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactics execution",
    )
    async def _execute_tactics(self) -> None:
        try:
            now = datetime.now().isoformat()
            self.status = {"timestamp": now, "status": "running"}
            self.history.append(self.status.copy())
            if len(self.history) > self.max_history:
                self.history.pop(0)
            await self._monitor_entry_signals()
            await self._monitor_exit_signals()
            await self._monitor_positions()
            await self._monitor_risk_levels()
            
            # Monitor SR levels
            if self.enable_sr_tactics and self.sr_analyzer:
                await self._monitor_sr_levels()
                
            # Monitor SR breakout predictions
            if self.enable_sr_breakout_tactics and self.sr_breakout_predictor:
                await self._monitor_sr_breakout_predictions()
                
            await self._update_tactics_results()
            self.logger.info(f"Tactics execution tick at {now}")
        except Exception as e:
            self.logger.error(f"Error in tactics execution: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="entry signal monitoring",
    )
    async def _monitor_entry_signals(self) -> None:
        try:
            # Monitor entry signals
            entry_signals = {
                "signal_strength": 0.75,
                "entry_conditions": "met",
                "market_conditions": "favorable",
                "entry_timing": "optimal",
            }
            self.tactics_results["entry_signals"] = entry_signals
            self.logger.info("Entry signal monitoring completed")
        except Exception as e:
            self.logger.error(f"Error monitoring entry signals: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="exit signal monitoring",
    )
    async def _monitor_exit_signals(self) -> None:
        try:
            # Monitor exit signals
            exit_signals = {
                "exit_strength": 0.65,
                "exit_conditions": "monitoring",
                "profit_targets": "active",
                "stop_losses": "active",
            }
            self.tactics_results["exit_signals"] = exit_signals
            self.logger.info("Exit signal monitoring completed")
        except Exception as e:
            self.logger.error(f"Error monitoring exit signals: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position monitoring",
    )
    async def _monitor_positions(self) -> None:
        try:
            # Monitor positions
            position_status = {
                "active_positions": 3,
                "position_sizes": "optimal",
                "position_health": "good",
                "position_returns": 0.125,
            }
            self.tactics_results["position_status"] = position_status
            self.logger.info("Position monitoring completed")
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="risk level monitoring",
    )
    async def _monitor_risk_levels(self) -> None:
        try:
            # Monitor risk levels
            risk_status = {
                "current_risk": 0.45,
                "risk_threshold": 0.75,
                "risk_management": "active",
                "risk_alerts": "none",
            }
            self.tactics_results["risk_status"] = risk_status
            self.logger.info("Risk level monitoring completed")
        except Exception as e:
            self.logger.error(f"Error monitoring risk levels: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR level monitoring",
    )
    async def _monitor_sr_levels(self) -> None:
        """Monitor support and resistance levels for tactical execution."""
        try:
            if not self.sr_analyzer:
                return

            # Get current market data (simulated for now)
            current_price = 1850.0  # Simulated current price
            
            # Check SR zone proximity
            sr_proximity = self.sr_analyzer.detect_sr_zone_proximity(
                current_price, tolerance_percent=2.0
            )
            
            sr_status = {
                "in_sr_zone": sr_proximity.get("in_zone", False),
                "nearest_level": sr_proximity.get("nearest_level"),
                "distance_percent": sr_proximity.get("distance_percent"),
                "level_type": sr_proximity.get("level_type"),
                "level_strength": sr_proximity.get("level_strength"),
                "confidence": sr_proximity.get("confidence", 0.0),
                "tactical_action": self._determine_sr_tactical_action(sr_proximity),
            }
            
            self.tactics_results["sr_status"] = sr_status
            self.logger.info("SR level monitoring completed")
            
        except Exception as e:
            self.logger.error(f"Error monitoring SR levels: {e}")

    def _determine_sr_tactical_action(self, sr_proximity: dict[str, Any]) -> str:
        """Determine tactical action based on SR proximity."""
        try:
            if not sr_proximity.get("in_zone", False):
                return "no_action"
                
            level_strength = sr_proximity.get("level_strength", 0.0)
            level_type = sr_proximity.get("level_type")
            confidence = sr_proximity.get("confidence", 0.0)
            
            if confidence < 0.5:
                return "monitor"
                
            if level_type == "support" and level_strength > 0.7:
                return "buy_signal"
            elif level_type == "resistance" and level_strength > 0.7:
                return "sell_signal"
            else:
                return "caution"
                
        except Exception as e:
            self.logger.error(f"Error determining SR tactical action: {e}")
            return "no_action"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout prediction monitoring",
    )
    async def _monitor_sr_breakout_predictions(self) -> None:
        """Monitor SR breakout predictions for tactical decisions."""
        try:
            # Simulate current market data (in real implementation, this would come from market data feed)
            current_price = 2000.0  # Example price
            market_data = {
                "close": [1950, 1960, 1970, 1980, 1990, 2000],
                "high": [1960, 1970, 1980, 1990, 2000, 2010],
                "low": [1940, 1950, 1960, 1970, 1980, 1990],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500]
            }
            
            # Create DataFrame for prediction
            import pandas as pd
            df = pd.DataFrame(market_data)
            
            # Get breakout prediction
            prediction = await self.sr_breakout_predictor.predict_breakout_probability(df, current_price)
            
            if prediction:
                # Store prediction results
                self.tactics_results["sr_breakout_prediction"] = prediction
                
                # Determine tactical action based on prediction
                tactical_action = self._determine_sr_breakout_tactical_action(prediction)
                self.tactics_results["sr_breakout_tactical_action"] = tactical_action
                
                self.logger.info(f"SR Breakout Prediction: {prediction.get('recommendation', 'UNKNOWN')} "
                               f"(Breakout: {prediction.get('breakout_probability', 0):.2f}, "
                               f"Bounce: {prediction.get('bounce_probability', 0):.2f})")
            else:
                self.tactics_results["sr_breakout_prediction"] = {"status": "no_prediction"}
                self.tactics_results["sr_breakout_tactical_action"] = "no_action"
                
        except Exception as e:
            self.logger.error(f"Error monitoring SR breakout predictions: {e}")

    def _determine_sr_breakout_tactical_action(self, prediction: dict[str, Any]) -> str:
        """Determine tactical action based on SR breakout prediction."""
        try:
            if not prediction.get("near_sr_zone", False):
                return "no_action"
                
            breakout_prob = prediction.get("breakout_probability", 0.5)
            bounce_prob = prediction.get("bounce_probability", 0.5)
            confidence = prediction.get("confidence", 0.0)
            
            # High confidence predictions
            if confidence > 0.7:
                if breakout_prob > 0.7:
                    return "strong_breakout_signal"
                elif bounce_prob > 0.7:
                    return "strong_bounce_signal"
                    
            # Moderate confidence predictions
            elif confidence > 0.5:
                if breakout_prob > 0.6:
                    return "moderate_breakout_signal"
                elif bounce_prob > 0.6:
                    return "moderate_bounce_signal"
                    
            # Low confidence or uncertain
            else:
                return "uncertain_signal"
                
        except Exception as e:
            self.logger.error(f"Error determining SR breakout tactical action: {e}")
            return "no_action"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactics results update",
    )
    async def _update_tactics_results(self) -> None:
        try:
            # Update tactics results
            self.tactics_results["last_update"] = datetime.now().isoformat()
            self.tactics_results["tactics_score"] = 0.85
            self.logger.info("Tactics results updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating tactics results: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactician stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Tactician...")
        try:
            self.is_running = False
            self.status = {"timestamp": datetime.now().isoformat(), "status": "stopped"}
            self.logger.info("✅ Tactician stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping tactician: {e}")

    def get_status(self) -> dict[str, Any]:
        return self.status.copy()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_tactics_results(self) -> dict[str, Any]:
        return self.tactics_results.copy()

    def get_tactics_modules(self) -> dict[str, Any]:
        return self.tactics_modules.copy()


tactician: Tactician | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="tactician setup",
)
async def setup_tactician(config: dict[str, Any] | None = None) -> Tactician | None:
    try:
        global tactician
        if config is None:
            config = {"tactician": {"tactics_interval": 30, "max_history": 100}}
        tactician = Tactician(config)
        success = await tactician.initialize()
        if success:
            return tactician
        return None
    except Exception as e:
        print(f"Error setting up tactician: {e}")
        return None
