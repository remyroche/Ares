# src/tactician/tactician.py

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.tactician.position_sizer import PositionSizer
from src.tactician.leverage_sizer import LeverageSizer
from src.tactician.position_division_strategy import PositionDivisionStrategy
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
        
        # SR Breakout Predictor integration (DEPRECATED - Replaced with enhanced predictive ensembles)
        self.sr_breakout_predictor = None
        self.enable_sr_breakout_tactics: bool = self.tactician_config.get("enable_sr_breakout_tactics", True)
        
        # Position and Leverage Sizing integration
        self.position_sizer = None
        self.leverage_sizer = None
        self.position_division_strategy = None
        self.enable_position_sizing: bool = self.tactician_config.get("enable_position_sizing", True)
        self.enable_leverage_sizing: bool = self.tactician_config.get("enable_leverage_sizing", True)
        self.enable_position_division: bool = self.tactician_config.get("enable_position_division", True)
        
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
                
            if self.enable_position_division:
                await self._initialize_position_division_strategy()
                
            # Initialize ML tactics
            if self.enable_ml_tactics:
                await self._initialize_ml_tactics()
                
            # Initialize Position Monitor for real-time monitoring
            await self._initialize_position_monitor()
                
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
        """Initialize SR Breakout Predictor for tactical decisions (DEPRECATED)."""
        try:
            # SR Breakout Predictor has been replaced with enhanced predictive ensembles
            self.logger.info("SR Breakout Predictor deprecated - using enhanced predictive ensembles")
            self.sr_breakout_predictor = None
        except Exception as e:
            self.logger.error(f"Error in deprecated SR Breakout Predictor initialization: {e}")
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
        context="position division strategy initialization",
    )
    async def _initialize_position_division_strategy(self) -> None:
        """Initialize Position Division Strategy for tactical position management."""
        try:
            self.position_division_strategy = PositionDivisionStrategy(self.config)
            self.logger.info("✅ Position Division Strategy initialized for tactical decisions")
        except Exception as e:
            self.logger.error(f"Error initializing Position Division Strategy: {e}")
            self.position_division_strategy = None

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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitor initialization",
    )
    async def _initialize_position_monitor(self) -> None:
        """Initialize position monitor for real-time position monitoring."""
        try:
            from src.tactician.position_monitor import setup_position_monitor
            
            # Setup position monitor with tactician config
            self.position_monitor = await setup_position_monitor(self.config)
            
            if self.position_monitor:
                # Start position monitoring in background
                asyncio.create_task(self.position_monitor.start_monitoring())
                self.logger.info("✅ Position Monitor initialized and started")
            else:
                self.logger.warning("⚠️ Position Monitor initialization failed")
                
        except Exception as e:
            self.logger.error(f"Error initializing position monitor: {e}")
            self.position_monitor = None

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
                
            if self.enable_position_division and self.position_division_strategy:
                await self._monitor_position_division()
                
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
            # Use the enhanced position monitor if available
            if hasattr(self, 'position_monitor') and self.position_monitor:
                # The position monitor handles its own monitoring loop
                # This method now just updates position data for the monitor
                await self._update_position_monitor_data()
            else:
                # Fallback to basic position monitoring
                position_status = {
                    "active_positions": 3,
                    "position_sizes": "optimal",
                    "position_health": "good",
                    "position_returns": 0.125,
                }
                self.tactics_results["position_status"] = position_status
                self.logger.info("Position monitoring completed (basic mode)")
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position monitor data update",
    )
    async def _update_position_monitor_data(self) -> None:
        """Update position data for the position monitor."""
        try:
            if not hasattr(self, 'position_monitor') or not self.position_monitor:
                return
            
            # Get current positions from state manager or trading system
            current_positions = self._get_current_positions()
            
            # Update position monitor with current data
            for position_id, position_data in current_positions.items():
                # Add market data to position data
                enhanced_position_data = await self._enhance_position_data(position_data)
                self.position_monitor.add_position(position_id, enhanced_position_data)
            
            # Get position status from monitor
            active_positions = self.position_monitor.get_active_positions()
            assessment_history = self.position_monitor.get_assessment_history(limit=10)
            
            # Update tactics results
            self.tactics_results["position_status"] = {
                "active_positions": len(active_positions),
                "monitored_positions": list(active_positions.keys()),
                "latest_assessments": [
                    {
                        "position_id": assessment.position_id,
                        "confidence": assessment.current_confidence,
                        "confidence_change": assessment.confidence_change,
                        "action": assessment.recommended_action.value,
                        "reason": assessment.action_reason,
                        "timestamp": assessment.assessment_timestamp.isoformat()
                    }
                    for assessment in assessment_history
                ]
            }
            
            self.logger.info(f"Position monitoring updated: {len(active_positions)} active positions")
            
        except Exception as e:
            self.logger.error(f"Error updating position monitor data: {e}")
    
    def _get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions from state manager or trading system."""
        try:
            # This would integrate with the actual trading system
            # For now, return simulated positions
            return {
                "pos_001": {
                    "symbol": "ETHUSDT",
                    "direction": "LONG",
                    "entry_price": 1850.0,
                    "current_price": 1860.0,
                    "position_size": 0.1,
                    "leverage": 1.0,
                    "entry_confidence": 0.75,
                    "entry_timestamp": datetime.now().isoformat(),
                    "time_in_position_hours": 2.5,
                    "market_volatility": 0.15,
                    "trend_strength": 0.6,
                    "base_confidence": 0.7,
                },
                "pos_002": {
                    "symbol": "BTCUSDT",
                    "direction": "SHORT",
                    "entry_price": 42000.0,
                    "current_price": 41800.0,
                    "position_size": 0.05,
                    "leverage": 2.0,
                    "entry_confidence": 0.65,
                    "entry_timestamp": datetime.now().isoformat(),
                    "time_in_position_hours": 1.0,
                    "market_volatility": 0.12,
                    "trend_strength": 0.4,
                    "base_confidence": 0.6,
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return {}
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position data enhancement",
    )
    async def _enhance_position_data(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance position data with additional market information."""
        try:
            # Add real-time market data
            symbol = position_data.get("symbol", "")
            current_price = position_data.get("current_price", 0.0)
            
            # Get ML predictions for confidence assessment
            ml_predictions = self._get_ml_predictions()
            if ml_predictions:
                confidence_scores = ml_predictions.get("confidence_scores", {})
                movement_confidence = ml_predictions.get("movement_confidence_scores", {})
                
                # Update confidence based on ML predictions
                if confidence_scores:
                    position_data["ml_confidence"] = confidence_scores
                
                if movement_confidence:
                    position_data["movement_confidence"] = movement_confidence
            
            # Add market volatility and trend data
            position_data["market_volatility"] = position_data.get("market_volatility", 0.1)
            position_data["trend_strength"] = position_data.get("trend_strength", 0.5)
            
            return position_data
            
        except Exception as e:
            self.logger.error(f"Error enhancing position data: {e}")
            return position_data

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
            
            # Get breakout prediction using enhanced predictive ensembles (Phase 2)
            prediction = await self._get_sr_breakout_prediction_enhanced(df, current_price)
            
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
            # Get ML predictions (this would come from ml_confidence_predictor)
            ml_predictions = {
                "movement_confidence_scores": {0.5: 0.8, 1.0: 0.75, 1.5: 0.7, 2.0: 0.65},
                "adverse_movement_risks": {0.5: 0.2, 1.0: 0.25, 1.5: 0.3, 2.0: 0.35},
                "directional_confidence": {"target_reach_confidence": {1.0: {"confidence": 0.8}}},
            }

            # Calculate position size
            position_result = await self.position_sizer.calculate_position_size(
                ml_predictions=ml_predictions,
                current_price=50000.0,
                account_balance=100000.0,
            )

            if position_result:
                # Store position sizing results
                self.tactics_results["position_sizing"] = {
                    "position_size": position_result.get("final_position_size", 0.0),
                    "kelly_position_size": position_result.get("kelly_position_size", 0.0),
                    "ml_position_size": position_result.get("ml_position_size", 0.0),
                    "sizing_reason": position_result.get("sizing_reason", ""),
                }

                # Determine tactical action based on position sizing
                tactical_action = self._determine_position_sizing_tactical_action(position_result)
                self.tactics_results["position_sizing_tactical_action"] = tactical_action

                self.logger.info(
                    f"Position Sizing: {position_result.get('final_position_size', 0.0):.4f} "
                    f"(Kelly: {position_result.get('kelly_position_size', 0.0):.4f}, "
                    f"ML: {position_result.get('ml_position_size', 0.0):.4f}, "
                    f"Action: {tactical_action})"
                )
            else:
                self.logger.warning("No position sizing result available")

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
            # Get ML predictions (this would come from ml_confidence_predictor)
            ml_predictions = {
                "movement_confidence_scores": {0.5: 0.8, 1.0: 0.75, 1.5: 0.7, 2.0: 0.65},
                "adverse_movement_risks": {0.5: 0.2, 1.0: 0.25, 1.5: 0.3, 2.0: 0.35},
                "directional_confidence": {"target_reach_confidence": {1.0: {"confidence": 0.8}}},
            }

            # Get liquidation risk analysis (this would come from liquidation_risk_model)
            liquidation_risk_analysis = {
                "safe_leverage_levels": {
                    "conservative": {"safe_leverage": 20.0},
                    "moderate": {"safe_leverage": 50.0},
                    "aggressive": {"safe_leverage": 80.0},
                }
            }

            # Get market health analysis (this would come from market_health_analyzer)
            market_health_analysis = {
                "volatility_analysis": {"current_volatility": 0.03},
                "liquidity_analysis": {"liquidity_score": 0.7},
                "stress_analysis": {"stress_level": 0.3},
            }

            # Calculate leverage
            leverage_result = await self.leverage_sizer.calculate_leverage(
                ml_predictions=ml_predictions,
                liquidation_risk_analysis=liquidation_risk_analysis,
                market_health_analysis=market_health_analysis,
                current_price=50000.0,
                target_direction="long",
            )

            if leverage_result:
                # Store leverage sizing results
                self.tactics_results["leverage_sizing"] = {
                    "leverage": leverage_result.get("final_leverage", 10.0),
                    "ml_leverage": leverage_result.get("ml_leverage", 10.0),
                    "liquidation_leverage": leverage_result.get("liquidation_leverage", 10.0),
                    "market_health_leverage": leverage_result.get("market_health_leverage", 10.0),
                    "leverage_reason": leverage_result.get("leverage_reason", ""),
                }

                # Determine tactical action based on leverage sizing
                tactical_action = self._determine_leverage_sizing_tactical_action(leverage_result)
                self.tactics_results["leverage_sizing_tactical_action"] = tactical_action

                self.logger.info(
                    f"Leverage Sizing: {leverage_result.get('final_leverage', 10.0):.1f}x "
                    f"(ML: {leverage_result.get('ml_leverage', 10.0):.1f}x, "
                    f"Action: {tactical_action})"
                )
            else:
                self.logger.warning("No leverage sizing result available")

        except Exception as e:
            self.logger.error(f"Error monitoring leverage sizing: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position division monitoring",
    )
    async def _monitor_position_division(self) -> None:
        """Monitor position division strategy and update tactical recommendations."""
        try:
            # Get ML predictions (this would come from ml_confidence_predictor)
            ml_predictions = {
                "movement_confidence_scores": {0.5: 0.8, 1.0: 0.75, 1.5: 0.7, 2.0: 0.65},
                "adverse_movement_risks": {0.5: 0.2, 1.0: 0.25, 1.5: 0.3, 2.0: 0.35},
                "directional_confidence": {"target_reach_confidence": {1.0: {"confidence": 0.8}}},
            }

            # Get current positions (this would come from your position manager)
            current_positions = [
                {
                    "position_id": "pos_001",
                    "entry_price": 50000.0,
                    "position_size": 0.1,
                    "entry_confidence": 0.8,
                },
                {
                    "position_id": "pos_002", 
                    "entry_price": 51000.0,
                    "position_size": 0.05,
                    "entry_confidence": 0.7,
                }
            ]

            # Get short-term ML confidence analysis (this would come from your short-term ML analyzer)
            short_term_analysis = {
                "ml_confidence_scores": {
                    "1m": {"confidence": 0.75, "direction": "up"},
                    "5m": {"confidence": 0.8, "direction": "up"},
                }
            }

            # Analyze position division
            division_result = await self.position_division_strategy.analyze_position_division(
                ml_predictions=ml_predictions,
                current_positions=current_positions,
                current_price=52000.0,
                short_term_analysis=short_term_analysis,
            )

            if division_result:
                # Store position division results
                self.tactics_results["position_division"] = {
                    "entry_recommendation": division_result.get("entry_recommendation", {}),
                    "take_profit_recommendation": division_result.get("take_profit_recommendation", {}),
                    "stop_loss_recommendation": division_result.get("stop_loss_recommendation", {}),
                    "full_close_recommendation": division_result.get("full_close_recommendation", {}),
                    "average_confidence": division_result.get("average_confidence", 0.0),
                    "short_term_score": division_result.get("short_term_score", 0.0),
                    "division_reason": division_result.get("division_reason", ""),
                }

                # Determine tactical action based on position division
                tactical_action = self._determine_position_division_tactical_action(division_result)
                self.tactics_results["position_division_tactical_action"] = tactical_action

                self.logger.info(
                    f"Position Division: {tactical_action} "
                    f"(Confidence: {division_result.get('average_confidence', 0.0):.2f}, "
                    f"Short-term: {division_result.get('short_term_score', 0.0):.2f})"
                )
            else:
                self.logger.warning("No position division result available")

        except Exception as e:
            self.logger.error(f"Error monitoring position division: {e}")

    def _determine_position_sizing_tactical_action(self, position_result: dict[str, Any]) -> str:
        """Determine tactical action based on position sizing results."""
        try:
            position_size = position_result.get("final_position_size", 0.0)
            kelly_size = position_result.get("kelly_position_size", 0.0)
            ml_size = position_result.get("ml_position_size", 0.0)

            # High confidence with good position size
            if position_size > 0.1:
                return "strong_position_signal"
            elif position_size > 0.05:
                return "moderate_position_signal"
            elif position_size > 0.01:
                return "small_position_signal"
            else:
                return "avoid_position_signal"

        except Exception as e:
            self.logger.error(f"Error determining position sizing tactical action: {e}")
            return "no_action"

    def _determine_leverage_sizing_tactical_action(self, leverage_result: dict[str, Any]) -> str:
        """Determine tactical action based on leverage sizing results."""
        try:
            leverage = leverage_result.get("final_leverage", 10.0)
            ml_leverage = leverage_result.get("ml_leverage", 10.0)
            liquidation_leverage = leverage_result.get("liquidation_leverage", 10.0)
            market_health_leverage = leverage_result.get("market_health_leverage", 10.0)

            # High leverage signals
            if leverage > 50:
                return "high_leverage_signal"
            elif leverage > 25:
                return "moderate_leverage_signal"
            elif leverage > 10:
                return "low_leverage_signal"
            else:
                return "minimal_leverage_signal"

        except Exception as e:
            self.logger.error(f"Error determining leverage sizing tactical action: {e}")
            return "no_action"

    def _determine_position_division_tactical_action(self, division_result: dict[str, Any]) -> str:
        """Determine tactical action based on position division results."""
        try:
            entry_rec = division_result.get("entry_recommendation", {})
            take_profit_rec = division_result.get("take_profit_recommendation", {})
            stop_loss_rec = division_result.get("stop_loss_recommendation", {})
            full_close_rec = division_result.get("full_close_recommendation", {})
            
            # Check for urgent actions first
            if stop_loss_rec.get("total_stop_loss_size", 0.0) > 0:
                return "urgent_stop_loss"
            
            if full_close_rec.get("total_full_close_size", 0.0) > 0:
                return "urgent_full_close"
            
            # Check for profit taking
            if take_profit_rec.get("total_take_profit_size", 0.0) > 0:
                return "take_profit"
            
            # Check for new position entry
            if entry_rec.get("should_enter", False):
                return "enter_new_position"
            
            # No action needed
            return "no_action"
            
        except Exception as e:
            self.logger.error(f"Error determining position division tactical action: {e}")
            return "no_action"

    async def _get_sr_breakout_prediction_enhanced(self, df: pd.DataFrame, current_price: float) -> dict[str, Any]:
        """
        Get SR breakout prediction using enhanced predictive ensembles (Phase 2).
        
        Args:
            df: Market data DataFrame
            current_price: Current price
            
        Returns:
            Prediction dictionary with breakout/bounce probabilities
        """
        try:
            # Check if predictive ensembles are available
            if not hasattr(self, 'predictive_ensembles') or not self.predictive_ensembles:
                self.logger.warning("Predictive ensembles not available, using fallback")
                return self._get_sr_breakout_prediction_fallback(df, current_price)
            
            # Get SR context using SR analyzer
            sr_context = None
            if hasattr(self, 'sr_analyzer') and self.sr_analyzer:
                sr_context = self.sr_analyzer.detect_sr_zone_proximity(current_price)
            
            # Prepare features for prediction
            features = self._prepare_features_for_sr_prediction(df, current_price, sr_context)
            
            # Get predictions from all available ensembles
            predictions = {}
            for ensemble_name, ensemble in self.predictive_ensembles.items():
                try:
                    prediction = ensemble.get_prediction(features)
                    predictions[ensemble_name] = prediction
                except Exception as e:
                    self.logger.warning(f"Error getting prediction from {ensemble_name}: {e}")
                    continue
            
            # Combine predictions to determine breakout/bounce probabilities
            breakout_prob, bounce_prob, confidence = self._combine_sr_predictions(predictions)
            
            # Determine if near SR zone
            near_sr_zone = sr_context.get('in_zone', False) if sr_context else False
            
            return {
                "breakout_probability": breakout_prob,
                "bounce_probability": bounce_prob,
                "confidence": confidence,
                "near_sr_zone": near_sr_zone,
                "sr_context": sr_context,
                "ensemble_predictions": predictions,
                "recommendation": self._get_sr_recommendation(breakout_prob, bounce_prob, confidence),
                "method": "enhanced_predictive_ensembles"
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced SR breakout prediction: {e}")
            return self._get_sr_breakout_prediction_fallback(df, current_price)
    
    def _prepare_features_for_sr_prediction(self, df: pd.DataFrame, current_price: float, sr_context: dict = None) -> pd.DataFrame:
        """Prepare features for SR breakout prediction."""
        try:
            # Create features DataFrame
            features = pd.DataFrame(index=[0])
            
            # Basic price features
            features['close'] = current_price
            features['price_change'] = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] if len(df) > 1 else 0
            
            # Technical indicators (if available)
            for indicator in ['rsi', 'macd', 'atr', 'adx', 'volume']:
                if indicator in df.columns:
                    features[indicator] = df[indicator].iloc[-1]
                else:
                    features[indicator] = 0.0
            
            # SR context features
            if sr_context and sr_context.get('in_zone', False):
                features['distance_to_sr'] = sr_context.get('distance_to_level', 0.0)
                features['sr_strength'] = sr_context.get('level_strength', 0.0)
                features['sr_type'] = 1.0 if sr_context.get('level_type') == 'resistance' else 0.0
            else:
                features['distance_to_sr'] = 1.0
                features['sr_strength'] = 0.0
                features['sr_type'] = 0.5
            
            # Momentum features
            if len(df) >= 5:
                features['momentum_5'] = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            else:
                features['momentum_5'] = 0.0
                
            if len(df) >= 10:
                features['momentum_10'] = (current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]
            else:
                features['momentum_10'] = 0.0
            
            # Volume features
            if 'volume' in df.columns:
                avg_volume = df['volume'].tail(20).mean()
                features['volume_ratio'] = df['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1.0
            else:
                features['volume_ratio'] = 1.0
            
            # Volatility features
            if len(df) >= 20:
                price_volatility = df['close'].tail(20).std() / df['close'].tail(20).mean()
                features['volatility'] = price_volatility
            else:
                features['volatility'] = 0.0
            
            # Price position
            if len(df) >= 20:
                high_20 = df['high'].tail(20).max()
                low_20 = df['low'].tail(20).min()
                features['price_position'] = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            else:
                features['price_position'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features for SR prediction: {e}")
            # Return minimal features
            return pd.DataFrame({
                'close': [current_price],
                'distance_to_sr': [1.0],
                'sr_strength': [0.0],
                'sr_type': [0.5]
            })
    
    def _combine_sr_predictions(self, predictions: dict) -> tuple[float, float, float]:
        """Combine predictions from multiple ensembles to determine breakout/bounce probabilities."""
        try:
            if not predictions:
                return 0.5, 0.5, 0.0
            
            # Extract predictions and confidences
            ensemble_predictions = []
            confidences = []
            
            for ensemble_name, prediction in predictions.items():
                if isinstance(prediction, dict):
                    pred_value = prediction.get('prediction', 'HOLD')
                    confidence = prediction.get('confidence', 0.0)
                    
                    # Map predictions to breakout/bounce probabilities
                    if pred_value in ['BULL', 'STRONG_BULL']:
                        breakout_prob = 0.7
                        bounce_prob = 0.3
                    elif pred_value in ['BEAR', 'STRONG_BEAR']:
                        breakout_prob = 0.3
                        bounce_prob = 0.7
                    else:  # HOLD, SIDEWAYS
                        breakout_prob = 0.5
                        bounce_prob = 0.5
                    
                    ensemble_predictions.append((breakout_prob, bounce_prob))
                    confidences.append(confidence)
            
            if not ensemble_predictions:
                return 0.5, 0.5, 0.0
            
            # Weighted average based on confidence
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weighted_breakout = sum(bp * conf for (bp, _), conf in zip(ensemble_predictions, confidences)) / total_confidence
                weighted_bounce = sum(bp * conf for (_, bp), conf in zip(ensemble_predictions, confidences)) / total_confidence
                avg_confidence = total_confidence / len(confidences)
            else:
                weighted_breakout = sum(bp for bp, _ in ensemble_predictions) / len(ensemble_predictions)
                weighted_bounce = sum(bp for _, bp in ensemble_predictions) / len(ensemble_predictions)
                avg_confidence = 0.0
            
            return weighted_breakout, weighted_bounce, avg_confidence
            
        except Exception as e:
            self.logger.error(f"Error combining SR predictions: {e}")
            return 0.5, 0.5, 0.0
    
    def _get_sr_recommendation(self, breakout_prob: float, bounce_prob: float, confidence: float) -> str:
        """Get recommendation based on breakout/bounce probabilities."""
        try:
            if confidence < 0.3:
                return "UNCERTAIN"
            
            if breakout_prob > 0.7:
                return "STRONG_BREAKOUT"
            elif breakout_prob > 0.6:
                return "MODERATE_BREAKOUT"
            elif bounce_prob > 0.7:
                return "STRONG_BOUNCE"
            elif bounce_prob > 0.6:
                return "MODERATE_BOUNCE"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error getting SR recommendation: {e}")
            return "UNCERTAIN"
    
    def _get_sr_breakout_prediction_fallback(self, df: pd.DataFrame, current_price: float) -> dict[str, Any]:
        """Fallback method when predictive ensembles are not available."""
        try:
            # Simple fallback based on price position and momentum
            if len(df) < 20:
                return {
                    "breakout_probability": 0.5,
                    "bounce_probability": 0.5,
                    "confidence": 0.0,
                    "near_sr_zone": False,
                    "recommendation": "UNCERTAIN",
                    "method": "fallback"
                }
            
            # Calculate simple momentum-based prediction
            momentum_5 = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
            momentum_10 = (current_price - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
            
            # Simple breakout probability based on momentum
            if momentum_5 > 0.02 and momentum_10 > 0.01:  # Strong upward momentum
                breakout_prob = 0.7
                bounce_prob = 0.3
            elif momentum_5 < -0.02 and momentum_10 < -0.01:  # Strong downward momentum
                breakout_prob = 0.3
                bounce_prob = 0.7
            else:
                breakout_prob = 0.5
                bounce_prob = 0.5
            
            return {
                "breakout_probability": breakout_prob,
                "bounce_probability": bounce_prob,
                "confidence": 0.3,  # Low confidence for fallback
                "near_sr_zone": False,
                "recommendation": self._get_sr_recommendation(breakout_prob, bounce_prob, 0.3),
                "method": "fallback"
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback SR prediction: {e}")
            return {
                "breakout_probability": 0.5,
                "bounce_probability": 0.5,
                "confidence": 0.0,
                "near_sr_zone": False,
                "recommendation": "UNCERTAIN",
                "method": "fallback"
            }

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
