# src/components/modular_tactician.py

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import traceback

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, initialization_error, invalid, missing


class ModularTactician:
    """
    Enhanced modular tactician with comprehensive error handling and type safety.
    
    This class provides tactical execution capabilities including entry/exit monitoring,
    position monitoring, risk monitoring, and trade execution.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ModularTactician with configuration.
        
        Args:
            config: Configuration dictionary containing tactician settings
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("ModularTactician")
        
        # Tactician state
        self.is_tactician_active: bool = False
        self.tactician_results: Dict[str, Any] = {}
        self.tactician_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.tactician_config: Dict[str, Any] = self.config.get("modular_tactician", {})
        self.tactician_interval: int = self.tactician_config.get("tactician_interval", 5)
        self.max_tactician_history: int = self.tactician_config.get("max_tactician_history", 100)
        self.enable_entry_monitoring: bool = self.tactician_config.get("enable_entry_monitoring", True)
        self.enable_exit_monitoring: bool = self.tactician_config.get("enable_exit_monitoring", True)
        self.enable_position_monitoring: bool = self.tactician_config.get("enable_position_monitoring", False)
        self.enable_risk_monitoring: bool = self.tactician_config.get("enable_risk_monitoring", True)
        
        # Tactical modules
        self.entry_monitor = None
        self.exit_monitor = None
        self.position_monitor = None
        self.risk_monitor = None
        
        self.logger.info("ModularTactician initialized with configuration")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular tactician configuration"),
            AttributeError: (False, "Missing required tactician parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular tactician initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the tactician and all its modules.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Modular Tactician...")
            
            # Load tactician configuration
            await self._load_tactician_configuration()
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for modular tactician"))
                return False
            
            # Initialize tactician modules
            await self._initialize_tactician_modules()
            
            self.logger.info("✅ Modular Tactician initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Modular Tactician initialization failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician configuration loading",
    )
    async def _load_tactician_configuration(self) -> None:
        """
        Load and validate tactician configuration.
        """
        try:
            # Set default tactician parameters
            self.tactician_config.setdefault("tactician_interval", 5)
            self.tactician_config.setdefault("max_tactician_history", 100)
            self.tactician_config.setdefault("enable_entry_monitoring", True)
            self.tactician_config.setdefault("enable_exit_monitoring", True)
            self.tactician_config.setdefault("enable_position_monitoring", False)
            self.tactician_config.setdefault("enable_risk_monitoring", True)
            
            # Update configuration
            self.tactician_interval = self.tactician_config["tactician_interval"]
            self.max_tactician_history = self.tactician_config["max_tactician_history"]
            self.enable_entry_monitoring = self.tactician_config["enable_entry_monitoring"]
            self.enable_exit_monitoring = self.tactician_config["enable_exit_monitoring"]
            self.enable_position_monitoring = self.tactician_config["enable_position_monitoring"]
            self.enable_risk_monitoring = self.tactician_config["enable_risk_monitoring"]
            
            self.logger.info("Tactician configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading tactician configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """
        Validate the tactician configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = ["tactician_interval", "max_tactician_history"]
            for key in required_keys:
                if key not in self.tactician_config:
                    self.logger.error(missing(f"Missing required configuration key: {key}"))
                    return False
            
            if self.tactician_interval <= 0:
                self.logger.error(invalid("Tactician interval must be positive"))
                return False
                
            if self.max_tactician_history <= 0:
                self.logger.error(invalid("Max tactician history must be positive"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    async def _initialize_tactician_modules(self) -> None:
        """
        Initialize all tactician modules based on configuration.
        """
        try:
            if self.enable_entry_monitoring:
                self.entry_monitor = EntryMonitor(self.tactician_config)
                self.logger.info("Entry monitor initialized")
            
            if self.enable_exit_monitoring:
                self.exit_monitor = ExitMonitor(self.tactician_config)
                self.logger.info("Exit monitor initialized")
            
            if self.enable_position_monitoring:
                self.position_monitor = PositionMonitor(self.tactician_config)
                self.logger.info("Position monitor initialized")
            
            if self.enable_risk_monitoring:
                self.risk_monitor = RiskMonitor(self.tactician_config)
                self.logger.info("Risk monitor initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing tactician modules: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError, RuntimeError),
        default_return=None,
        context="tactical execution",
    )
    async def execute_tactics(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any], strategy_signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute tactical decisions based on market data, portfolio state, and strategy signals.
        
        Args:
            market_data: Current market data
            portfolio_state: Current portfolio state
            strategy_signals: Strategy signals from analyst/strategist
            
        Returns:
            Dict containing tactical execution results or None if execution fails
        """
        try:
            if self.is_tactician_active:
                self.logger.warning("Tactical execution already in progress")
                return None
            
            self.is_tactician_active = True
            self.logger.info("Starting tactical execution...")
            
            tactical_result = {
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "portfolio_state": portfolio_state,
                "strategy_signals": strategy_signals,
                "entry_signals": None,
                "exit_signals": None,
                "position_adjustments": None,
                "risk_alerts": None,
                "executed_trades": [],
                "overall_tactical_score": 0.0,
                "action_items": []
            }
            
            # Monitor entry opportunities
            if self.entry_monitor and self.enable_entry_monitoring:
                try:
                    tactical_result["entry_signals"] = await self.entry_monitor.analyze_entry_opportunities(
                        market_data, portfolio_state, strategy_signals
                    )
                except Exception as e:
                    self.logger.error(f"Entry monitoring failed: {e}")
            
            # Monitor exit opportunities
            if self.exit_monitor and self.enable_exit_monitoring:
                try:
                    tactical_result["exit_signals"] = await self.exit_monitor.analyze_exit_opportunities(
                        market_data, portfolio_state, strategy_signals
                    )
                except Exception as e:
                    self.logger.error(f"Exit monitoring failed: {e}")
            
            # Monitor positions
            if self.position_monitor and self.enable_position_monitoring:
                try:
                    tactical_result["position_adjustments"] = await self.position_monitor.analyze_positions(
                        portfolio_state, market_data
                    )
                except Exception as e:
                    self.logger.error(f"Position monitoring failed: {e}")
            
            # Monitor risk
            if self.risk_monitor and self.enable_risk_monitoring:
                try:
                    tactical_result["risk_alerts"] = await self.risk_monitor.analyze_risk(
                        portfolio_state, market_data
                    )
                except Exception as e:
                    self.logger.error(f"Risk monitoring failed: {e}")
            
            # Execute trades based on signals
            tactical_result["executed_trades"] = await self._execute_trades(tactical_result)
            
            # Calculate overall tactical score
            tactical_result["overall_tactical_score"] = self._calculate_tactical_score(tactical_result)
            
            # Generate action items
            tactical_result["action_items"] = self._generate_action_items(tactical_result)
            
            # Store results
            self.tactician_results = tactical_result
            self._add_to_history(tactical_result)
            
            self.logger.info(f"Tactical execution completed. Overall score: {tactical_result['overall_tactical_score']:.2f}")
            return tactical_result
            
        except Exception as e:
            self.logger.error(f"Tactical execution failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            self.is_tactician_active = False

    async def _execute_trades(self, tactical_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute trades based on tactical signals.
        
        Args:
            tactical_result: Tactical execution results
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        try:
            # Execute entry trades
            if tactical_result.get("entry_signals"):
                entry_signals = tactical_result["entry_signals"]
                if entry_signals.get("should_enter", False):
                    trade = {
                        "type": "entry",
                        "symbol": entry_signals.get("symbol", "UNKNOWN"),
                        "quantity": entry_signals.get("quantity", 0),
                        "price": entry_signals.get("price", 0.0),
                        "timestamp": datetime.now().isoformat(),
                        "confidence": entry_signals.get("confidence", 0.0)
                    }
                    executed_trades.append(trade)
                    self.logger.info(f"Entry trade executed: {trade}")
            
            # Execute exit trades
            if tactical_result.get("exit_signals"):
                exit_signals = tactical_result["exit_signals"]
                if exit_signals.get("should_exit", False):
                    trade = {
                        "type": "exit",
                        "symbol": exit_signals.get("symbol", "UNKNOWN"),
                        "quantity": exit_signals.get("quantity", 0),
                        "price": exit_signals.get("price", 0.0),
                        "timestamp": datetime.now().isoformat(),
                        "confidence": exit_signals.get("confidence", 0.0)
                    }
                    executed_trades.append(trade)
                    self.logger.info(f"Exit trade executed: {trade}")
            
            # Execute position adjustments
            if tactical_result.get("position_adjustments"):
                position_adjustments = tactical_result["position_adjustments"]
                if position_adjustments.get("adjustments_needed", False):
                    for adjustment in position_adjustments.get("adjustments", []):
                        trade = {
                            "type": "adjustment",
                            "symbol": adjustment.get("symbol", "UNKNOWN"),
                            "quantity": adjustment.get("quantity", 0),
                            "price": adjustment.get("price", 0.0),
                            "timestamp": datetime.now().isoformat(),
                            "confidence": adjustment.get("confidence", 0.0)
                        }
                        executed_trades.append(trade)
                        self.logger.info(f"Position adjustment executed: {trade}")
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
        
        return executed_trades

    def _calculate_tactical_score(self, tactical_result: Dict[str, Any]) -> float:
        """
        Calculate overall tactical score based on execution results.
        
        Args:
            tactical_result: Tactical execution results dictionary
            
        Returns:
            float: Overall score between 0.0 and 1.0
        """
        try:
            scores = []
            weights = []
            
            # Entry signals score
            if tactical_result["entry_signals"]:
                entry_score = tactical_result["entry_signals"].get("score", 0.0)
                scores.append(entry_score)
                weights.append(0.3)
            
            # Exit signals score
            if tactical_result["exit_signals"]:
                exit_score = tactical_result["exit_signals"].get("score", 0.0)
                scores.append(exit_score)
                weights.append(0.3)
            
            # Position adjustments score
            if tactical_result["position_adjustments"]:
                pos_score = tactical_result["position_adjustments"].get("score", 0.0)
                scores.append(pos_score)
                weights.append(0.2)
            
            # Risk monitoring score (inverted - lower risk = higher score)
            if tactical_result["risk_alerts"]:
                risk_score = 1.0 - tactical_result["risk_alerts"].get("risk_level", 0.5)
                scores.append(risk_score)
                weights.append(0.2)
            
            if not scores:
                return 0.0
            
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating tactical score: {e}")
            return 0.0

    def _generate_action_items(self, tactical_result: Dict[str, Any]) -> List[str]:
        """
        Generate actionable items based on tactical results.
        
        Args:
            tactical_result: Tactical execution results dictionary
            
        Returns:
            List of action item strings
        """
        action_items = []
        
        try:
            overall_score = tactical_result.get("overall_tactical_score", 0.0)
            
            if overall_score >= 0.8:
                action_items.append("Execute all tactical signals immediately - high confidence")
            elif overall_score >= 0.6:
                action_items.append("Proceed with tactical signals - moderate confidence")
            elif overall_score >= 0.4:
                action_items.append("Review tactical signals before execution - low confidence")
            else:
                action_items.append("Hold current positions - tactical signals unclear")
            
            # Add specific action items based on individual components
            if tactical_result.get("entry_signals"):
                entry_signals = tactical_result["entry_signals"]
                if entry_signals.get("should_enter", False):
                    action_items.append(f"Enter position in {entry_signals.get('symbol', 'UNKNOWN')} at {entry_signals.get('price', 0.0)}")
            
            if tactical_result.get("exit_signals"):
                exit_signals = tactical_result["exit_signals"]
                if exit_signals.get("should_exit", False):
                    action_items.append(f"Exit position in {exit_signals.get('symbol', 'UNKNOWN')} at {exit_signals.get('price', 0.0)}")
            
            if tactical_result.get("risk_alerts"):
                risk_level = tactical_result["risk_alerts"].get("risk_level", 0.5)
                if risk_level > 0.7:
                    action_items.append("High risk detected - implement immediate risk mitigation")
                elif risk_level > 0.5:
                    action_items.append("Elevated risk - monitor positions closely")
            
            # Add trade execution summary
            if tactical_result.get("executed_trades"):
                trade_count = len(tactical_result["executed_trades"])
                action_items.append(f"Executed {trade_count} trades based on tactical signals")
            
        except Exception as e:
            self.logger.error(f"Error generating action items: {e}")
            action_items.append("Unable to generate specific action items due to tactical errors")
        
        return action_items

    def _add_to_history(self, tactical_result: Dict[str, Any]) -> None:
        """
        Add tactical result to history, maintaining maximum history size.
        
        Args:
            tactical_result: Tactical result to add
        """
        try:
            self.tactician_history.append(tactical_result)
            
            # Maintain maximum history size
            if len(self.tactician_history) > self.max_tactician_history:
                self.tactician_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")

    def get_tactical_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get tactical execution history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of tactical execution results
        """
        try:
            if limit is None:
                return self.tactician_history.copy()
            else:
                return self.tactician_history[-limit:].copy()
        except Exception as e:
            self.logger.error(f"Error retrieving tactical history: {e}")
            return []

    def get_latest_tactical_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent tactical execution result.
        
        Returns:
            Latest tactical result or None if no execution performed
        """
        try:
            if self.tactician_history:
                return self.tactician_history[-1].copy()
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving latest tactical result: {e}")
            return None

    def clear_history(self) -> None:
        """Clear tactical execution history."""
        try:
            self.tactician_history.clear()
            self.logger.info("Tactical history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing history: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current tactician status.
        
        Returns:
            Dictionary containing current status information
        """
        try:
            return {
                "is_tactician_active": self.is_tactician_active,
                "tactician_interval": self.tactician_interval,
                "history_size": len(self.tactician_history),
                "max_history_size": self.max_tactician_history,
                "enabled_modules": {
                    "entry_monitoring": self.enable_entry_monitoring,
                    "exit_monitoring": self.enable_exit_monitoring,
                    "position_monitoring": self.enable_position_monitoring,
                    "risk_monitoring": self.enable_risk_monitoring
                },
                "last_execution": self.tactician_history[-1]["timestamp"] if self.tactician_history else None
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}


# Placeholder classes for tactical modules
class EntryMonitor:
    """Placeholder for entry monitoring module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze_entry_opportunities(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any], strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder entry opportunity analysis."""
        return {
            "score": 0.7,
            "should_enter": False,
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.0,
            "confidence": 0.75
        }


class ExitMonitor:
    """Placeholder for exit monitoring module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze_exit_opportunities(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any], strategy_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder exit opportunity analysis."""
        return {
            "score": 0.6,
            "should_exit": False,
            "symbol": "AAPL",
            "quantity": 100,
            "price": 155.0,
            "confidence": 0.65
        }


class PositionMonitor:
    """Placeholder for position monitoring module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze_positions(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder position analysis."""
        return {
            "score": 0.8,
            "adjustments_needed": False,
            "adjustments": [],
            "confidence": 0.85
        }


class RiskMonitor:
    """Placeholder for risk monitoring module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze_risk(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder risk analysis."""
        return {
            "risk_level": 0.3,
            "risk_factors": [],
            "mitigation_strategies": [],
            "confidence": 0.8
        }
