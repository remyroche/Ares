# src/components/modular_strategist.py

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import traceback

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, initialization_error, invalid, missing


class ModularStrategist:
    """
    Enhanced modular strategist with comprehensive error handling and type safety.
    
    This class provides trading strategy capabilities including position sizing,
    risk management, portfolio optimization, and dynamic rebalancing.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ModularStrategist with configuration.
        
        Args:
            config: Configuration dictionary containing strategist settings
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("ModularStrategist")
        
        # Strategy state
        self.is_strategizing: bool = False
        self.strategy_results: Dict[str, Any] = {}
        self.strategy_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.strategist_config: Dict[str, Any] = self.config.get("modular_strategist", {})
        self.strategy_interval: int = self.strategist_config.get("strategy_interval", 60)
        self.max_strategy_history: int = self.strategist_config.get("max_strategy_history", 100)
        self.enable_position_sizing: bool = self.strategist_config.get("enable_position_sizing", True)
        self.enable_risk_management: bool = self.strategist_config.get("enable_risk_management", True)
        self.enable_portfolio_optimization: bool = self.strategist_config.get("enable_portfolio_optimization", False)
        self.enable_dynamic_rebalancing: bool = self.strategist_config.get("enable_dynamic_rebalancing", True)
        
        # Strategy modules
        self.position_sizer = None
        self.risk_manager = None
        self.portfolio_optimizer = None
        self.rebalancer = None
        
        self.logger.info("ModularStrategist initialized with configuration")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular strategist configuration"),
            AttributeError: (False, "Missing required strategist parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular strategist initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the strategist and all its modules.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Modular Strategist...")
            
            # Load strategist configuration
            await self._load_strategist_configuration()
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for modular strategist"))
                return False
            
            # Initialize strategy modules
            await self._initialize_strategy_modules()
            
            self.logger.info("✅ Modular Strategist initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Modular Strategist initialization failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="strategist configuration loading",
    )
    async def _load_strategist_configuration(self) -> None:
        """
        Load and validate strategist configuration.
        """
        try:
            # Set default strategist parameters
            self.strategist_config.setdefault("strategy_interval", 60)
            self.strategist_config.setdefault("max_strategy_history", 100)
            self.strategist_config.setdefault("enable_position_sizing", True)
            self.strategist_config.setdefault("enable_risk_management", True)
            self.strategist_config.setdefault("enable_portfolio_optimization", False)
            self.strategist_config.setdefault("enable_dynamic_rebalancing", True)
            
            # Update configuration
            self.strategy_interval = self.strategist_config["strategy_interval"]
            self.max_strategy_history = self.strategist_config["max_strategy_history"]
            self.enable_position_sizing = self.strategist_config["enable_position_sizing"]
            self.enable_risk_management = self.strategist_config["enable_risk_management"]
            self.enable_portfolio_optimization = self.strategist_config["enable_portfolio_optimization"]
            self.enable_dynamic_rebalancing = self.strategist_config["enable_dynamic_rebalancing"]
            
            self.logger.info("Strategist configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading strategist configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """
        Validate the strategist configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = ["strategy_interval", "max_strategy_history"]
            for key in required_keys:
                if key not in self.strategist_config:
                    self.logger.error(missing(f"Missing required configuration key: {key}"))
                    return False
            
            if self.strategy_interval <= 0:
                self.logger.error(invalid("Strategy interval must be positive"))
                return False
                
            if self.max_strategy_history <= 0:
                self.logger.error(invalid("Max strategy history must be positive"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    async def _initialize_strategy_modules(self) -> None:
        """
        Initialize all strategy modules based on configuration.
        """
        try:
            if self.enable_position_sizing:
                self.position_sizer = PositionSizer(self.strategist_config)
                self.logger.info("Position sizer initialized")
            
            if self.enable_risk_management:
                self.risk_manager = RiskManager(self.strategist_config)
                self.logger.info("Risk manager initialized")
            
            if self.enable_portfolio_optimization:
                self.portfolio_optimizer = PortfolioOptimizer(self.strategist_config)
                self.logger.info("Portfolio optimizer initialized")
            
            if self.enable_dynamic_rebalancing:
                self.rebalancer = PortfolioRebalancer(self.strategist_config)
                self.logger.info("Portfolio rebalancer initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing strategy modules: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError, RuntimeError),
        default_return=None,
        context="trading strategy generation",
    )
    async def generate_strategy(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive trading strategy.
        
        Args:
            market_data: Current market data
            portfolio_state: Current portfolio state
            
        Returns:
            Dict containing strategy recommendations or None if strategy generation fails
        """
        try:
            if self.is_strategizing:
                self.logger.warning("Strategy generation already in progress")
                return None
            
            self.is_strategizing = True
            self.logger.info("Starting strategy generation...")
            
            strategy_result = {
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "portfolio_state": portfolio_state,
                "position_sizing": None,
                "risk_assessment": None,
                "portfolio_optimization": None,
                "rebalancing_recommendations": None,
                "overall_strategy_score": 0.0,
                "action_items": []
            }
            
            # Generate position sizing recommendations
            if self.position_sizer and self.enable_position_sizing:
                try:
                    strategy_result["position_sizing"] = await self.position_sizer.calculate_positions(
                        market_data, portfolio_state
                    )
                except Exception as e:
                    self.logger.error(f"Position sizing failed: {e}")
            
            # Assess risk
            if self.risk_manager and self.enable_risk_management:
                try:
                    strategy_result["risk_assessment"] = await self.risk_manager.assess_portfolio_risk(
                        portfolio_state, market_data
                    )
                except Exception as e:
                    self.logger.error(f"Risk assessment failed: {e}")
            
            # Optimize portfolio
            if self.portfolio_optimizer and self.enable_portfolio_optimization:
                try:
                    strategy_result["portfolio_optimization"] = await self.portfolio_optimizer.optimize_portfolio(
                        portfolio_state, market_data
                    )
                except Exception as e:
                    self.logger.error(f"Portfolio optimization failed: {e}")
            
            # Generate rebalancing recommendations
            if self.rebalancer and self.enable_dynamic_rebalancing:
                try:
                    strategy_result["rebalancing_recommendations"] = await self.rebalancer.generate_rebalancing_plan(
                        portfolio_state, market_data
                    )
                except Exception as e:
                    self.logger.error(f"Rebalancing plan generation failed: {e}")
            
            # Calculate overall strategy score
            strategy_result["overall_strategy_score"] = self._calculate_strategy_score(strategy_result)
            
            # Generate action items
            strategy_result["action_items"] = self._generate_action_items(strategy_result)
            
            # Store results
            self.strategy_results = strategy_result
            self._add_to_history(strategy_result)
            
            self.logger.info(f"Strategy generation completed. Overall score: {strategy_result['overall_strategy_score']:.2f}")
            return strategy_result
            
        except Exception as e:
            self.logger.error(f"Strategy generation failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            self.is_strategizing = False

    def _calculate_strategy_score(self, strategy_result: Dict[str, Any]) -> float:
        """
        Calculate overall strategy score based on individual components.
        
        Args:
            strategy_result: Strategy results dictionary
            
        Returns:
            float: Overall score between 0.0 and 1.0
        """
        try:
            scores = []
            weights = []
            
            # Position sizing score
            if strategy_result["position_sizing"]:
                pos_score = strategy_result["position_sizing"].get("score", 0.0)
                scores.append(pos_score)
                weights.append(0.25)
            
            # Risk assessment score (inverted - lower risk = higher score)
            if strategy_result["risk_assessment"]:
                risk_score = 1.0 - strategy_result["risk_assessment"].get("risk_level", 0.5)
                scores.append(risk_score)
                weights.append(0.3)
            
            # Portfolio optimization score
            if strategy_result["portfolio_optimization"]:
                opt_score = strategy_result["portfolio_optimization"].get("optimization_score", 0.0)
                scores.append(opt_score)
                weights.append(0.25)
            
            # Rebalancing score
            if strategy_result["rebalancing_recommendations"]:
                reb_score = strategy_result["rebalancing_recommendations"].get("rebalancing_score", 0.0)
                scores.append(reb_score)
                weights.append(0.2)
            
            if not scores:
                return 0.0
            
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy score: {e}")
            return 0.0

    def _generate_action_items(self, strategy_result: Dict[str, Any]) -> List[str]:
        """
        Generate actionable items based on strategy results.
        
        Args:
            strategy_result: Strategy results dictionary
            
        Returns:
            List of action item strings
        """
        action_items = []
        
        try:
            overall_score = strategy_result.get("overall_strategy_score", 0.0)
            
            if overall_score >= 0.8:
                action_items.append("Execute strategy immediately - high confidence signals")
            elif overall_score >= 0.6:
                action_items.append("Proceed with strategy - moderate confidence signals")
            elif overall_score >= 0.4:
                action_items.append("Review and adjust strategy - low confidence signals")
            else:
                action_items.append("Hold current positions - strategy signals unclear")
            
            # Add specific action items based on individual components
            if strategy_result.get("position_sizing"):
                pos_sizing = strategy_result["position_sizing"]
                if pos_sizing.get("recommendation", "").lower() == "increase":
                    action_items.append("Increase position sizes based on favorable market conditions")
                elif pos_sizing.get("recommendation", "").lower() == "decrease":
                    action_items.append("Decrease position sizes due to increased market risk")
            
            if strategy_result.get("rebalancing_recommendations"):
                rebalancing = strategy_result["rebalancing_recommendations"]
                if rebalancing.get("rebalancing_needed", False):
                    action_items.append("Execute portfolio rebalancing to maintain target allocations")
            
            if strategy_result.get("risk_assessment"):
                risk_level = strategy_result["risk_assessment"].get("risk_level", 0.5)
                if risk_level > 0.7:
                    action_items.append("Implement additional risk management measures")
                elif risk_level < 0.3:
                    action_items.append("Consider increasing risk exposure within acceptable limits")
            
        except Exception as e:
            self.logger.error(f"Error generating action items: {e}")
            action_items.append("Unable to generate specific action items due to strategy errors")
        
        return action_items

    def _add_to_history(self, strategy_result: Dict[str, Any]) -> None:
        """
        Add strategy result to history, maintaining maximum history size.
        
        Args:
            strategy_result: Strategy result to add
        """
        try:
            self.strategy_history.append(strategy_result)
            
            # Maintain maximum history size
            if len(self.strategy_history) > self.max_strategy_history:
                self.strategy_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")

    def get_strategy_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get strategy history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of strategy results
        """
        try:
            if limit is None:
                return self.strategy_history.copy()
            else:
                return self.strategy_history[-limit:].copy()
        except Exception as e:
            self.logger.error(f"Error retrieving strategy history: {e}")
            return []

    def get_latest_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent strategy result.
        
        Returns:
            Latest strategy result or None if no strategy generated
        """
        try:
            if self.strategy_history:
                return self.strategy_history[-1].copy()
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving latest strategy: {e}")
            return None

    def clear_history(self) -> None:
        """Clear strategy history."""
        try:
            self.strategy_history.clear()
            self.logger.info("Strategy history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing history: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current strategist status.
        
        Returns:
            Dictionary containing current status information
        """
        try:
            return {
                "is_strategizing": self.is_strategizing,
                "strategy_interval": self.strategy_interval,
                "history_size": len(self.strategy_history),
                "max_history_size": self.max_strategy_history,
                "enabled_modules": {
                    "position_sizing": self.enable_position_sizing,
                    "risk_management": self.enable_risk_management,
                    "portfolio_optimization": self.enable_portfolio_optimization,
                    "dynamic_rebalancing": self.enable_dynamic_rebalancing
                },
                "last_strategy": self.strategy_history[-1]["timestamp"] if self.strategy_history else None
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}


# Placeholder classes for strategy modules
class PositionSizer:
    """Placeholder for position sizing module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def calculate_positions(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder position sizing calculation."""
        return {
            "score": 0.75,
            "recommendation": "hold",
            "position_sizes": {},
            "confidence": 0.8
        }


class RiskManager:
    """Placeholder for risk management module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def assess_portfolio_risk(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder risk assessment."""
        return {
            "risk_level": 0.35,
            "risk_factors": [],
            "mitigation_strategies": [],
            "confidence": 0.85
        }


class PortfolioOptimizer:
    """Placeholder for portfolio optimization module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def optimize_portfolio(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder portfolio optimization."""
        return {
            "optimization_score": 0.7,
            "target_allocations": {},
            "optimization_recommendations": [],
            "confidence": 0.75
        }


class PortfolioRebalancer:
    """Placeholder for portfolio rebalancing module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def generate_rebalancing_plan(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder rebalancing plan generation."""
        return {
            "rebalancing_score": 0.65,
            "rebalancing_needed": False,
            "rebalancing_trades": [],
            "confidence": 0.7
        }
