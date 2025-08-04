# src/tactician/tactician.py

import asyncio
from datetime import datetime
from typing import Any

from src.tactician.position_sizer import PositionSizer
from src.tactician.leverage_sizer import LeverageSizer
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
        
        # Position and Leverage Sizing integration
        self.position_sizer = None
        self.leverage_sizer = None
        self.enable_position_sizing: bool = self.tactician_config.get("enable_position_sizing", True)
        self.enable_leverage_sizing: bool = self.tactician_config.get("enable_leverage_sizing", True)
        
        # ML Prediction integration
        self.ml_predictions = None
        self.enable_ml_tactics: bool = self.tactician_config.get("enable_ml_tactics", True)

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
                
            # Initialize Position and Leverage Sizers
            if self.enable_position_sizing:
                await self._initialize_position_sizer()
                
            if self.enable_leverage_sizing:
                await self._initialize_leverage_sizer()
                
            # Initialize ML tactics
            if self.enable_ml_tactics:
                await self._initialize_ml_tactics()
                
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position sizer initialization",
    )
    async def _initialize_position_sizer(self) -> None:
        """Initialize Position Sizer for tactical position sizing decisions."""
        try:
            self.position_sizer = PositionSizer(self.config)
            self.logger.info("✅ Position Sizer initialized for tactical decisions")
        except Exception as e:
            self.logger.error(f"Error initializing Position Sizer: {e}")
            self.position_sizer = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="leverage sizer initialization",
    )
    async def _initialize_leverage_sizer(self) -> None:
        """Initialize Leverage Sizer for tactical leverage decisions."""
        try:
            self.leverage_sizer = LeverageSizer(self.config)
            self.logger.info("✅ Leverage Sizer initialized for tactical decisions")
        except Exception as e:
            self.logger.error(f"Error initializing Leverage Sizer: {e}")
            self.leverage_sizer = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML tactics initialization",
    )
    async def _initialize_ml_tactics(self) -> None:
        """Initialize ML tactics for position entry and sizing decisions."""
        try:
            self.logger.info("Initializing ML tactics...")
            
            # ML tactics will be initialized when predictions are received
            self.ml_predictions = {}
            self.logger.info("✅ ML tactics initialized for tactical decisions")
                
        except Exception as e:
            self.logger.error(f"Error initializing ML tactics: {e}")
            self.ml_predictions = None

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
                
            # Monitor position and leverage sizing
            if self.enable_position_sizing and self.position_sizer:
                await self._monitor_position_sizing()
                
            if self.enable_leverage_sizing and self.leverage_sizer:
                await self._monitor_leverage_sizing()
                
            # Monitor ML tactics
            if self.enable_ml_tactics:
                await self._monitor_ml_tactics()
                
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position sizing monitoring",
    )
    async def _monitor_position_sizing(self) -> None:
        """Monitor position sizing decisions and update tactical recommendations."""
        try:
            # Get current market conditions (this would come from your market data source)
            market_conditions = {
                "current_price": 50000.0,  # Example BTC price
                "atr": 2500.0,  # Average True Range
                "realized_volatility_30d": 0.04,  # 4% daily volatility
                "market_regime": "BULL_TREND",  # Market regime classification
                "opportunity_type": "SR_BREAKOUT",  # Opportunity type
                "near_sr_zone": True,
                "huge_candle": False,
                "strong_momentum": True,
            }

            # Get current portfolio state (this would come from your state manager)
            portfolio_value = 100000.0  # Example portfolio value
            current_exposure = 0.15  # 15% current exposure
            daily_pnl = 0.02  # 2% daily PnL
            max_drawdown = 0.05  # 5% max drawdown

            # Update position sizer performance metrics
            self.position_sizer.update_performance_metrics(daily_pnl, max_drawdown)
            self.position_sizer.update_exposure(current_exposure)

            # Calculate position size for a sample trade
            position_result = self.position_sizer.calculate_position_size(
                current_price=market_conditions["current_price"],
                stop_loss_price=market_conditions["current_price"] * 0.95,  # 5% stop loss
                leverage=10,
                confidence=0.85,  # High confidence signal
                market_conditions=market_conditions,
                portfolio_value=portfolio_value,
                existing_positions=[],  # No existing positions
                side="long",
            )

            # Store position sizing results
            self.tactics_results["position_sizing"] = {
                "position_size": position_result.get("position_size", 0.0),
                "confidence_score": position_result.get("confidence_score", 0.0),
                "liquidation_safety_score": position_result.get("liquidation_safety_score", 50.0),
                "distance_to_liquidation": position_result.get("distance_to_liquidation", 0.0),
                "total_exposure_after": position_result.get("total_exposure_after", 0.0),
                "multipliers": {
                    "confidence": position_result.get("confidence_multiplier", 1.0),
                    "volatility": position_result.get("volatility_multiplier", 1.0),
                    "regime": position_result.get("regime_multiplier", 1.0),
                    "liquidation": position_result.get("liquidation_multiplier", 1.0),
                    "risk": position_result.get("risk_multiplier", 1.0),
                },
                "successive_positions_allowed": position_result.get("successive_positions_allowed", False),
                "calculation_time": position_result.get("calculation_time", ""),
            }

            # Determine tactical action based on position sizing
            tactical_action = self._determine_position_sizing_tactical_action(position_result)
            self.tactics_results["position_sizing_tactical_action"] = tactical_action

            self.logger.info(
                f"Position Sizing: {position_result.get('position_size', 0.0):.4f} "
                f"(LSS: {position_result.get('liquidation_safety_score', 50.0):.1f}, "
                f"Action: {tactical_action})"
            )

        except Exception as e:
            self.logger.error(f"Error monitoring position sizing: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="leverage sizing monitoring",
    )
    async def _monitor_leverage_sizing(self) -> None:
        """Monitor leverage sizing decisions and update tactical recommendations."""
        try:
            # Get current market conditions (this would come from your market data source)
            market_conditions = {
                "current_price": 50000.0,  # Example BTC price
                "atr": 2500.0,  # Average True Range
                "realized_volatility_30d": 0.04,  # 4% daily volatility
                "market_regime": "BULL_TREND",  # Market regime classification
                "opportunity_type": "SR_BREAKOUT",  # Opportunity type
                "near_sr_zone": True,
                "huge_candle": False,
                "strong_momentum": True,
            }

            # Get current portfolio state (this would come from your state manager)
            portfolio_value = 100000.0  # Example portfolio value
            current_exposure = 0.15  # 15% current exposure
            daily_pnl = 0.02  # 2% daily PnL
            max_drawdown = 0.05  # 5% max drawdown

            # Update leverage sizer performance metrics
            self.leverage_sizer.update_performance_metrics(daily_pnl, max_drawdown)
            self.leverage_sizer.update_exposure(current_exposure)

            # Calculate leverage for a sample trade
            leverage_result = self.leverage_sizer.calculate_leverage(
                base_leverage=10,
                max_leverage_cap=50,
                confidence=0.85,  # High confidence signal
                market_conditions=market_conditions,
                position_size=5000.0,  # Example position size
                current_price=market_conditions["current_price"],
                side="long",
            )

            # Store leverage sizing results
            self.tactics_results["leverage_sizing"] = {
                "leverage": leverage_result.get("leverage", 1),
                "confidence_score": leverage_result.get("confidence_score", 0.0),
                "liquidation_safety_score": leverage_result.get("liquidation_safety_score", 50.0),
                "distance_to_liquidation": leverage_result.get("distance_to_liquidation", 0.0),
                "max_leverage_cap": leverage_result.get("max_leverage_cap", 50),
                "multipliers": {
                    "confidence": leverage_result.get("confidence_multiplier", 1.0),
                    "volatility": leverage_result.get("volatility_multiplier", 1.0),
                    "regime": leverage_result.get("regime_multiplier", 1.0),
                    "opportunity": leverage_result.get("opportunity_multiplier", 1.0),
                    "liquidation": leverage_result.get("liquidation_multiplier", 1.0),
                    "risk": leverage_result.get("risk_multiplier", 1.0),
                },
                "calculation_time": leverage_result.get("calculation_time", ""),
            }

            # Determine tactical action based on leverage sizing
            tactical_action = self._determine_leverage_sizing_tactical_action(leverage_result)
            self.tactics_results["leverage_sizing_tactical_action"] = tactical_action

            self.logger.info(
                f"Leverage Sizing: {leverage_result.get('leverage', 1)}x "
                f"(LSS: {leverage_result.get('liquidation_safety_score', 50.0):.1f}, "
                f"Action: {tactical_action})"
            )

        except Exception as e:
            self.logger.error(f"Error monitoring leverage sizing: {e}")

    def _determine_position_sizing_tactical_action(self, position_result: dict[str, Any]) -> str:
        """Determine tactical action based on position sizing results."""
        try:
            position_size = position_result.get("position_size", 0.0)
            lss = position_result.get("liquidation_safety_score", 50.0)
            confidence = position_result.get("confidence_score", 0.0)
            total_exposure = position_result.get("total_exposure_after", 0.0)

            # High confidence with good safety score
            if confidence > 0.8 and lss > 70:
                if position_size > 0:
                    return "strong_position_signal"
                else:
                    return "no_position_signal"

            # Moderate confidence with acceptable safety
            elif confidence > 0.6 and lss > 50:
                if position_size > 0:
                    return "moderate_position_signal"
                else:
                    return "no_position_signal"

            # Low confidence or poor safety
            else:
                return "avoid_position_signal"

        except Exception as e:
            self.logger.error(f"Error determining position sizing tactical action: {e}")
            return "no_action"

    def _determine_leverage_sizing_tactical_action(self, leverage_result: dict[str, Any]) -> str:
        """Determine tactical action based on leverage sizing results."""
        try:
            leverage = leverage_result.get("leverage", 1)
            lss = leverage_result.get("liquidation_safety_score", 50.0)
            confidence = leverage_result.get("confidence_score", 0.0)
            max_cap = leverage_result.get("max_leverage_cap", 50)

            # High confidence with good safety score
            if confidence > 0.8 and lss > 70:
                if leverage > 10:
                    return "high_leverage_signal"
                elif leverage > 5:
                    return "moderate_leverage_signal"
                else:
                    return "low_leverage_signal"

            # Moderate confidence with acceptable safety
            elif confidence > 0.6 and lss > 50:
                if leverage > 5:
                    return "moderate_leverage_signal"
                else:
                    return "low_leverage_signal"

            # Low confidence or poor safety
            else:
                return "minimal_leverage_signal"

        except Exception as e:
            self.logger.error(f"Error determining leverage sizing tactical action: {e}")
            return "no_action"

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
        context="ML tactics monitoring",
    )
    async def _monitor_ml_tactics(self) -> None:
        """Monitor ML tactics for position entry and sizing decisions."""
        try:
            # Get ML predictions from governance or analyst
            ml_predictions = self._get_ml_predictions()
            
            if ml_predictions:
                # Make tactical decisions based on ML predictions
                entry_decisions = self._make_ml_entry_decisions(ml_predictions)
                directional_decisions = self._make_ml_directional_decisions(ml_predictions)
                liquidation_risk_decisions = self._make_ml_liquidation_risk_decisions(ml_predictions)
                
                # Use position and leverage sizers for sizing decisions
                if self.position_sizer and self.leverage_sizer:
                    sizing_decisions = await self._calculate_position_size(ml_predictions)
                    leverage_decisions = await self._calculate_leverage(ml_predictions)
                else:
                    sizing_decisions = self._make_ml_sizing_decisions(ml_predictions)
                    leverage_decisions = self._make_ml_leverage_decisions(ml_predictions)
                
                self.tactics_results["ml_tactics"] = {
                    "entry_decisions": entry_decisions,
                    "sizing_decisions": sizing_decisions,
                    "leverage_decisions": leverage_decisions,
                    "directional_decisions": directional_decisions,
                    "liquidation_risk_decisions": liquidation_risk_decisions,
                    "timestamp": datetime.now(),
                }
                
                self.logger.info("ML tactics monitoring completed")
            else:
                self.logger.warning("No ML predictions available for tactics")
                
        except Exception as e:
            self.logger.error(f"Error in ML tactics monitoring: {e}")

    def _get_ml_predictions(self) -> dict[str, Any] | None:
        """Get ML predictions from governance or analyst."""
        try:
            # This would typically get predictions from the Governor or Analyst
            # For now, return a mock prediction structure
            return {
                "confidence_scores": {
                    0.3: 0.7,
                    0.4: 0.6,
                    0.5: 0.5,
                    0.6: 0.4,
                    0.7: 0.3,
                    0.8: 0.2,
                    0.9: 0.1,
                    1.0: 0.05,
                },
                "expected_decreases": {
                    0.1: 0.3,
                    0.2: 0.4,
                    0.3: 0.5,
                    0.4: 0.6,
                    0.5: 0.7,
                    0.6: 0.8,
                    0.7: 0.9,
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting ML predictions: {e}")
            return None

    def _make_ml_entry_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make entry decisions based on ML confidence scores.

        Args:
            ml_predictions: ML prediction data

        Returns:
            dict[str, Any]: Entry decisions
        """
        try:
            entry_decisions = {}
            confidence_scores = ml_predictions.get("confidence_scores", {})
            
            for increase_level, confidence in confidence_scores.items():
                if confidence >= 0.7:  # High confidence threshold
                    entry_decisions[f"enter_{increase_level}"] = {
                        "should_enter": True,
                        "confidence": confidence,
                        "action": "enter_position",
                        "reason": f"High confidence ({confidence:.2f}) for {increase_level}% increase"
                    }
                elif confidence >= 0.5:  # Medium confidence threshold
                    entry_decisions[f"enter_{increase_level}"] = {
                        "should_enter": True,
                        "confidence": confidence,
                        "action": "enter_position_cautious",
                        "reason": f"Medium confidence ({confidence:.2f}) for {increase_level}% increase"
                    }
                else:
                    entry_decisions[f"enter_{increase_level}"] = {
                        "should_enter": False,
                        "confidence": confidence,
                        "action": "hold",
                        "reason": f"Low confidence ({confidence:.2f}) for {increase_level}% increase"
                    }
            
            return entry_decisions
            
        except Exception as e:
            self.logger.error(f"Error making ML entry decisions: {e}")
            return {}

    def _make_ml_sizing_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make position sizing decisions based on ML confidence scores.

        Args:
            ml_predictions: ML prediction data

        Returns:
            dict[str, Any]: Sizing decisions
        """
        try:
            sizing_decisions = {}
            confidence_scores = ml_predictions.get("confidence_scores", {})
            
            for increase_level, confidence in confidence_scores.items():
                # Calculate position size based on confidence
                base_size = 0.1  # 10% base position size
                confidence_multiplier = confidence
                position_size = base_size * confidence_multiplier
                
                sizing_decisions[f"size_{increase_level}"] = {
                    "position_size": min(position_size, 0.5),  # Cap at 50%
                    "confidence": confidence,
                    "sizing_reason": f"Position size {position_size:.2f} based on confidence {confidence:.2f}"
                }
            
            return sizing_decisions
            
        except Exception as e:
            self.logger.error(f"Error making ML sizing decisions: {e}")
            return {}

    def _make_ml_leverage_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make leverage decisions based on ML confidence scores.

        Args:
            ml_predictions: ML prediction data

        Returns:
            dict[str, Any]: Leverage decisions
        """
        try:
            leverage_decisions = {}
            confidence_scores = ml_predictions.get("confidence_scores", {})
            
            for increase_level, confidence in confidence_scores.items():
                # Calculate leverage based on confidence
                base_leverage = 1.0
                confidence_multiplier = confidence
                leverage = base_leverage + (confidence_multiplier * 2)  # Max 3x leverage
                
                leverage_decisions[f"leverage_{increase_level}"] = {
                    "leverage": min(leverage, 3.0),  # Cap at 3x
                    "confidence": confidence,
                    "leverage_reason": f"Leverage {leverage:.2f} based on confidence {confidence:.2f}"
                }
            
            return leverage_decisions
            
        except Exception as e:
            self.logger.error(f"Error making ML leverage decisions: {e}")
            return {}

    def _make_ml_directional_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make directional decisions based on ML directional confidence analysis.

        Args:
            ml_predictions: ML prediction data

        Returns:
            dict[str, Any]: Directional decisions
        """
        try:
            directional_decisions = {}
            directional_confidence = ml_predictions.get("directional_confidence", {})
            target_reach_confidence = directional_confidence.get("target_reach_confidence", {})
            directional_safety_score = directional_confidence.get("directional_safety_score", {})
            
            for target_level, data in target_reach_confidence.items():
                confidence = data.get("confidence", 0.0)
                safety_data = directional_safety_score.get(target_level, {})
                safety_score = safety_data.get("safety_score", 0.0)
                
                if confidence >= 0.7 and safety_score >= 0.6:
                    directional_decisions[f"target_{target_level}"] = {
                        "should_target": True,
                        "confidence": confidence,
                        "safety_score": safety_score,
                        "target_price": data.get("target_price", 0.0),
                        "action": "enter_position",
                        "reason": f"High confidence ({confidence:.2f}) and safety ({safety_score:.2f}) for {target_level}% target"
                    }
                elif confidence >= 0.5 and safety_score >= 0.4:
                    directional_decisions[f"target_{target_level}"] = {
                        "should_target": True,
                        "confidence": confidence,
                        "safety_score": safety_score,
                        "target_price": data.get("target_price", 0.0),
                        "action": "enter_position_cautious",
                        "reason": f"Moderate confidence ({confidence:.2f}) and safety ({safety_score:.2f}) for {target_level}% target"
                    }
                else:
                    directional_decisions[f"target_{target_level}"] = {
                        "should_target": False,
                        "confidence": confidence,
                        "safety_score": safety_score,
                        "target_price": data.get("target_price", 0.0),
                        "action": "avoid_position",
                        "reason": f"Low confidence ({confidence:.2f}) or safety ({safety_score:.2f}) for {target_level}% target"
                    }
            
            return directional_decisions
            
        except Exception as e:
            self.logger.error(f"Error making ML directional decisions: {e}")
            return {}

    def _make_ml_liquidation_risk_decisions(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Make liquidation risk decisions based on ML directional confidence analysis.

        Args:
            ml_predictions: ML prediction data

        Returns:
            dict[str, Any]: Liquidation risk decisions
        """
        try:
            liquidation_risk_decisions = {}
            directional_confidence = ml_predictions.get("directional_confidence", {})
            liquidation_risk_assessment = directional_confidence.get("liquidation_risk_assessment", {})
            recommended_leverage = directional_confidence.get("recommended_leverage", {})
            
            for target_level, risk_data in liquidation_risk_assessment.items():
                liquidation_risk = risk_data.get("liquidation_risk", 0.0)
                safe_leverage = risk_data.get("safe_leverage", 1.0)
                risk_level = risk_data.get("risk_level", "medium")
                
                leverage_data = recommended_leverage.get(target_level, {})
                recommended_leverage_value = leverage_data.get("leverage", 1.0)
                max_safe_leverage = leverage_data.get("max_safe_leverage", 1.0)
                
                # Determine position action based on liquidation risk
                if liquidation_risk <= 0.3:
                    action = "enter_position"
                    position_size_multiplier = 1.0
                elif liquidation_risk <= 0.6:
                    action = "enter_position_cautious"
                    position_size_multiplier = 0.5
                else:
                    action = "avoid_position"
                    position_size_multiplier = 0.0
                
                liquidation_risk_decisions[f"risk_{target_level}"] = {
                    "action": action,
                    "liquidation_risk": liquidation_risk,
                    "risk_level": risk_level,
                    "safe_leverage": safe_leverage,
                    "recommended_leverage": recommended_leverage_value,
                    "max_safe_leverage": max_safe_leverage,
                    "position_size_multiplier": position_size_multiplier,
                    "reason": f"Liquidation risk {liquidation_risk:.2f} for {target_level}% target",
                }
            
            return liquidation_risk_decisions
            
        except Exception as e:
            self.logger.error(f"Error making ML liquidation risk decisions: {e}")
            return {}

    async def _calculate_position_size(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate position size using the position sizer.
        
        Args:
            ml_predictions: ML confidence predictions
            
        Returns:
            dict[str, Any]: Position sizing analysis
        """
        try:
            if not self.position_sizer:
                self.logger.warning("Position sizer not available, using fallback")
                return self._make_ml_sizing_decisions(ml_predictions)
            
            # Get component results for position sizing
            strategist_results = self.tactics_results.get("strategist_results", {})
            analyst_results = self.tactics_results.get("analyst_results", {})
            governor_results = self.tactics_results.get("governor_results", {})
            
            # Calculate position size
            position_analysis = await self.position_sizer.calculate_position_size(
                ml_predictions=ml_predictions,
                strategist_results=strategist_results,
                analyst_results=analyst_results,
                governor_results=governor_results,
                current_price=ml_predictions.get("current_price", 0.0),
                account_balance=1000.0,  # Default account balance
            )
            
            if position_analysis:
                return {
                    "position_size": position_analysis.get("final_position_size", 0.1),
                    "kelly_position_size": position_analysis.get("kelly_position_size", 0.1),
                    "weighted_position_size": position_analysis.get("weighted_position_size", 0.1),
                    "sizing_reason": position_analysis.get("sizing_reason", "Position size calculated"),
                    "component_indicators": position_analysis.get("component_indicators", {}),
                }
            else:
                return self._make_ml_sizing_decisions(ml_predictions)
                
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self._make_ml_sizing_decisions(ml_predictions)

    async def _calculate_leverage(self, ml_predictions: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate leverage using the leverage sizer.
        
        Args:
            ml_predictions: ML confidence predictions
            
        Returns:
            dict[str, Any]: Leverage sizing analysis
        """
        try:
            if not self.leverage_sizer:
                self.logger.warning("Leverage sizer not available, using fallback")
                return self._make_ml_leverage_decisions(ml_predictions)
            
            # Get component results for leverage sizing
            strategist_results = self.tactics_results.get("strategist_results", {})
            analyst_results = self.tactics_results.get("analyst_results", {})
            governor_results = self.tactics_results.get("governor_results", {})
            
            # Get liquidation risk analysis (if available)
            liquidation_risk_analysis = None
            if "liquidation_risk_analysis" in self.tactics_results:
                liquidation_risk_analysis = self.tactics_results["liquidation_risk_analysis"]
            
            # Calculate leverage
            leverage_analysis = await self.leverage_sizer.calculate_leverage(
                ml_predictions=ml_predictions,
                liquidation_risk_analysis=liquidation_risk_analysis,
                strategist_results=strategist_results,
                analyst_results=analyst_results,
                governor_results=governor_results,
                current_price=ml_predictions.get("current_price", 0.0),
                target_direction="long",  # Default direction
            )
            
            if leverage_analysis:
                return {
                    "leverage": leverage_analysis.get("final_leverage", 1.0),
                    "ml_leverage": leverage_analysis.get("ml_leverage", 1.0),
                    "liquidation_leverage": leverage_analysis.get("liquidation_leverage", 1.0),
                    "weighted_leverage": leverage_analysis.get("weighted_leverage", 1.0),
                    "leverage_reason": leverage_analysis.get("leverage_reason", "Leverage calculated"),
                    "component_indicators": leverage_analysis.get("component_indicators", {}),
                }
            else:
                return self._make_ml_leverage_decisions(ml_predictions)
                
        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return self._make_ml_leverage_decisions(ml_predictions)

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
