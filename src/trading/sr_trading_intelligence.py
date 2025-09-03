#!/usr/bin/env python3
"""
SR Trading Intelligence - Comprehensive Access to SR Levels for Trading Decisions

This module provides:
1. Real-time access to SR levels with all metadata
2. Trading decision support based on SR analysis
3. Integration with live trading systems
4. Performance tracking and optimization
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.tactician.sr_levels_manager import SRLevelsManager, SRLevel
from src.tactician.sr_levels_manager import SRLevel, SRLevelsManager
from src.utils.error_handler import handle_errors, handle_specific_errors

logger = system_logger.getChild("SRTradingIntelligence")

class SRTradingIntelligence:
    """Trading Intelligence system that provides comprehensive access to SR levels.

    Features:
    - Real-time SR level access
    - Trading decision support
    - Performance tracking
    - Risk management integration
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SR Trading Intelligence."""
        self.config = config
        self.logger = system_logger.getChild("SRTradingIntelligence")

        # Configuration
        self.intelligence_config = config.get("sr_trading_intelligence", {})
        self.enable_real_time_updates = self.intelligence_config.get(
            "enable_real_time_updates", True
        )
        self.update_interval_seconds = self.intelligence_config.get(
            "update_interval_seconds", 60
        )
        self.max_position_size = self.intelligence_config.get("max_position_size", 0.1)
        self.risk_tolerance = self.intelligence_config.get("risk_tolerance", 0.02)

        # SR Levels Manager
        self.sr_manager: Optional[SRLevelsManager] = None

        # Trading state
        self.current_position: Dict[str, Any] = {}
        self.trading_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Real-time update task
        self._update_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def initialize(self) -> bool:
        """Initialize the SR Trading Intelligence system."""
        try:
            self.logger.info("🔧 Initializing SR Trading Intelligence...")

            # Initialize SR Levels Manager
            self.sr_manager = await self._create_sr_manager()
            if not self.sr_manager:
                self.logger.error("❌ Failed to initialize SR Levels Manager")
                return False

            # Load trading history
            await self._load_trading_history()

            # Start real-time updates if enabled
            if self.enable_real_time_updates:
                await self._start_real_time_updates()

            self.logger.info("✅ SR Trading Intelligence initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SR Trading Intelligence: {e}")
            return False

    async def _create_sr_manager(self) -> Optional[SRLevelsManager]:
        """Create and initialize SR Levels Manager."""
        try:
            from src.tactician.sr_levels_manager import create_sr_levels_manager

            return await create_sr_levels_manager(self.config)
        except Exception as e:
            self.logger.error(f"❌ Error creating SR Levels Manager: {e}")
            return None

    async def _start_real_time_updates(self):
        """Start real-time SR level updates."""
        if self._update_task and not self._update_task.done():
            return

        self._is_running = True
        self._update_task = asyncio.create_task(self._real_time_update_loop())
        self.logger.info("🚀 Started real-time SR level updates")

    async def _real_time_update_loop(self):
        """Real-time update loop for SR levels."""
        try:
            while self._is_running:
                await asyncio.sleep(self.update_interval_seconds)

                # Get current market data (this would come from your exchange integration)
                current_data = await self._get_current_market_data()
                if current_data:
                    await self._update_sr_levels_with_market_data(current_data)

        except asyncio.CancelledError:
            self.logger.info("🛑 Real-time updates cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in real-time update loop: {e}")

    async def _get_current_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data from exchange."""
        # This would integrate with your exchange data feed
        # For now, return None to indicate no data available
        return None

    async def _update_sr_levels_with_market_data(self, market_data: Dict[str, Any]):
        """Update SR levels with current market data."""
        try:
            if not self.sr_manager:
                return

            current_price = market_data.get("price", 0)
            current_volume = market_data.get("volume", 0)
            current_time = market_data.get("timestamp", datetime.now())

            if current_price > 0:
                await self.sr_manager.update_levels_with_live_data(
                    current_price, current_volume, current_time
                )

        except Exception as e:
            self.logger.error(f"❌ Error updating SR levels with market data: {e}")

    def get_sr_levels_for_trading(
        self, current_price: float, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get SR levels optimized for trading decisions.

        Args:
            current_price: Current market price
            include_metadata: Whether to include detailed metadata

        Returns:
            Trading-optimized SR levels with decision support
        """
        try:
            if not self.sr_manager:
                return {"error": "SR Manager not initialized"}

            # Get basic SR levels
            sr_levels = self.sr_manager.get_sr_levels_for_trading(
                current_price, include_metadata
            )

            # Add trading intelligence
            trading_intelligence = self._generate_trading_intelligence(
                current_price, sr_levels
            )

            # Combine results
            result = {
                **sr_levels,
                "trading_intelligence": trading_intelligence,
                "risk_assessment": self._assess_risk(current_price, sr_levels),
                "position_recommendations": self._generate_position_recommendations(
                    current_price, sr_levels
                ),
            }

            return result

        except Exception as e:
            self.logger.error(f"❌ Error getting SR levels for trading: {e}")
            return {"error": str(e)}

    def _generate_trading_intelligence(
        self, current_price: float, sr_levels: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading intelligence based on SR levels."""
        try:
            intelligence = {
                "market_position": "neutral",
                "trend_direction": "sideways",
                "volatility_assessment": "normal",
                "entry_opportunities": [],
                "exit_signals": [],
                "risk_level": "medium",
            }

            # Analyze nearest levels
            nearest_support = sr_levels.get("nearest_support")
            nearest_resistance = sr_levels.get("nearest_resistance")

            if nearest_support and nearest_resistance:
                # Determine market position
                support_distance = (
                    abs(current_price - nearest_support["price"]) / current_price
                )
                resistance_distance = (
                    abs(current_price - nearest_resistance["price"]) / current_price
                )

                if support_distance < 0.01:  # Within 1% of support
                    intelligence["market_position"] = "near_support"
                    intelligence["entry_opportunities"].append(
                        {
                            "type": "long_entry",
                            "price": nearest_support["price"],
                            "confidence": nearest_support["quality_score"],
                            "reason": "Price near strong support level",
                        }
                    )
                elif resistance_distance < 0.01:  # Within 1% of resistance
                    intelligence["market_position"] = "near_resistance"
                    intelligence["exit_signals"].append(
                        {
                            "type": "long_exit",
                            "price": nearest_resistance["price"],
                            "confidence": nearest_resistance["quality_score"],
                            "reason": "Price near strong resistance level",
                        }
                    )

                # Determine trend direction
                if nearest_support["price"] > nearest_resistance["price"]:
                    intelligence["trend_direction"] = "downtrend"
                elif nearest_resistance["price"] > nearest_support["price"]:
                    intelligence["trend_direction"] = "uptrend"

                # Assess volatility
                level_distance = (
                    abs(nearest_resistance["price"] - nearest_support["price"])
                    / current_price
                )
                if level_distance < 0.02:
                    intelligence["volatility_assessment"] = "low"
                elif level_distance > 0.05:
                    intelligence["volatility_assessment"] = "high"

                # Assess risk level
                support_strength = nearest_support.get("strength", 0.5)
                resistance_strength = nearest_resistance.get("strength", 0.5)
                avg_strength = (support_strength + resistance_strength) / 2

                if avg_strength < 0.4:
                    intelligence["risk_level"] = "high"
                elif avg_strength > 0.7:
                    intelligence["risk_level"] = "low"

            return intelligence

        except Exception as e:
            self.logger.error(f"❌ Error generating trading intelligence: {e}")
            return {"error": str(e)}

    def _assess_risk(
        self, current_price: float, sr_levels: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risk based on SR levels and current position."""
        try:
            risk_assessment = {
                "overall_risk": "medium",
                "position_risk": "low",
                "market_risk": "medium",
                "risk_factors": [],
                "risk_score": 0.5,
            }

            # Calculate risk score based on various factors
            risk_score = 0.5

            # Factor 1: Distance to nearest levels
            nearest_support = sr_levels.get("nearest_support")
            nearest_resistance = sr_levels.get("nearest_resistance")

            if nearest_support and nearest_resistance:
                support_distance = (
                    abs(current_price - nearest_support["price"]) / current_price
                )
                resistance_distance = (
                    abs(current_price - nearest_resistance["price"]) / current_price
                )

                # Closer to levels = higher risk
                if min(support_distance, resistance_distance) < 0.005:  # Within 0.5%
                    risk_score += 0.2
                    risk_assessment["risk_factors"].append(
                        "Price very close to SR level"
                    )

                # Factor 2: Level strength
                support_strength = nearest_support.get("strength", 0.5)
                resistance_strength = nearest_resistance.get("strength", 0.5)
                avg_strength = (support_strength + resistance_strength) / 2

                if avg_strength < 0.4:
                    risk_score += 0.2
                    risk_assessment["risk_factors"].append("Weak SR levels")
                elif avg_strength > 0.7:
                    risk_score -= 0.1
                    risk_assessment["risk_factors"].append("Strong SR levels")

                # Factor 3: Level age
                support_age = nearest_support.get("age_hours", 0)
                resistance_age = nearest_resistance.get("age_hours", 0)
                avg_age = (support_age + resistance_age) / 2

                if avg_age > 168:  # Older than 1 week
                    risk_score += 0.1
                    risk_assessment["risk_factors"].append("SR levels are old")

                # Factor 4: Current position
                if self.current_position:
                    position_size = abs(self.current_position.get("size", 0))
                    if position_size > self.max_position_size * 0.8:
                        risk_score += 0.2
                        risk_assessment["risk_factors"].append("Large position size")

            # Normalize risk score
            risk_score = max(0.0, min(1.0, risk_score))
            risk_assessment["risk_score"] = risk_score

            # Categorize overall risk
            if risk_score < 0.3:
                risk_assessment["overall_risk"] = "low"
            elif risk_score > 0.7:
                risk_assessment["overall_risk"] = "high"

            return risk_assessment

        except Exception as e:
            self.logger.error(f"❌ Error assessing risk: {e}")
            return {"error": str(e)}

    def _generate_position_recommendations(
        self, current_price: float, sr_levels: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate position recommendations based on SR analysis."""
        try:
            recommendations = []

            nearest_support = sr_levels.get("nearest_support")
            nearest_resistance = sr_levels.get("nearest_resistance")

            if not nearest_support or not nearest_resistance:
                return recommendations

            # Long entry recommendation
            if (
                nearest_support["quality_score"] > 0.7
                and nearest_support["proximity"] < 0.02
            ):
                recommendations.append(
                    {
                        "action": "long_entry",
                        "entry_price": nearest_support["price"],
                        "stop_loss": nearest_support["price"]
                        * 0.99,  # 1% below support
                        "take_profit": nearest_resistance["price"]
                        * 1.02,  # 2% above resistance
                        "confidence": nearest_support["quality_score"],
                        "reason": f"Strong support at {nearest_support['price']:.4f} with quality {nearest_support['quality_score']:.2f}",
                        "risk_reward_ratio": 2.0,
                    }
                )

            # Short entry recommendation
            if (
                nearest_resistance["quality_score"] > 0.7
                and nearest_resistance["proximity"] < 0.02
            ):
                recommendations.append(
                    {
                        "action": "short_entry",
                        "entry_price": nearest_resistance["price"],
                        "stop_loss": nearest_resistance["price"]
                        * 1.01,  # 1% above resistance
                        "take_profit": nearest_support["price"]
                        * 0.98,  # 2% below support
                        "confidence": nearest_resistance["quality_score"],
                        "reason": f"Strong resistance at {nearest_resistance['price']:.4f} with quality {nearest_resistance['quality_score']:.2f}",
                        "risk_reward_ratio": 2.0,
                    }
                )

            # Exit recommendations for existing positions
            if self.current_position:
                position_type = self.current_position.get("type", "long")
                entry_price = self.current_position.get("entry_price", 0)

                if position_type == "long" and nearest_resistance["proximity"] < 0.01:
                    recommendations.append(
                        {
                            "action": "exit_long",
                            "exit_price": nearest_resistance["price"],
                            "confidence": nearest_resistance["quality_score"],
                            "reason": f"Price near strong resistance, consider taking profits",
                            "urgency": (
                                "high"
                                if nearest_resistance["proximity"] < 0.005
                                else "medium"
                            ),
                        }
                    )

                elif position_type == "short" and nearest_support["proximity"] < 0.01:
                    recommendations.append(
                        {
                            "action": "exit_short",
                            "exit_price": nearest_support["price"],
                            "confidence": nearest_support["quality_score"],
                            "reason": f"Price near strong support, consider covering short",
                            "urgency": (
                                "high"
                                if nearest_support["proximity"] < 0.005
                                else "medium"
                            ),
                        }
                    )

            return recommendations

        except Exception as e:
            self.logger.error(f"❌ Error generating position recommendations: {e}")
            return []

    async def update_position(
        self, position_type: str, entry_price: float, size: float, timestamp: datetime
    ):
        """Update current position information."""
        try:
            self.current_position = {
                "type": position_type,
                "entry_price": entry_price,
                "size": size,
                "entry_time": timestamp,
                "last_update": timestamp,
            }

            # Add to trading history
            self.trading_history.append(
                {**self.current_position, "action": "position_update"}
            )

            # Update performance metrics
            await self._update_performance_metrics()

            self.logger.info(
                f"✅ Updated position: {position_type} {size} @ {entry_price}"
            )

        except Exception as e:
            self.logger.error(f"❌ Error updating position: {e}")

    async def close_position(self, exit_price: float, timestamp: datetime):
        """Close current position and record trade."""
        try:
            if not self.current_position:
                self.logger.warning("⚠️ No position to close")
                return

            # Calculate P&L
            entry_price = self.current_position["entry_price"]
            size = self.current_position["size"]
            position_type = self.current_position["type"]

            if position_type == "long":
                pnl = (exit_price - entry_price) * size
            else:  # short
                pnl = (entry_price - exit_price) * size

            # Record trade
            trade_record = {
                "entry_time": self.current_position["entry_time"],
                "exit_time": timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_type": position_type,
                "size": size,
                "pnl": pnl,
                "pnl_percentage": (pnl / (entry_price * size)) * 100,
            }

            self.trading_history.append({**trade_record, "action": "position_close"})

            # Clear current position
            self.current_position = {}

            # Update performance metrics
            await self._update_performance_metrics()

            self.logger.info(
                f"✅ Closed position: P&L {pnl:.2f} ({trade_record['pnl_percentage']:.2f}%)"
            )

        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics based on trading history."""
        try:
            if not self.trading_history:
                return

            # Calculate basic metrics
            total_trades = len(
                [t for t in self.trading_history if t.get("action") == "position_close"]
            )
            winning_trades = len(
                [t for t in self.trading_history if t.get("pnl", 0) > 0]
            )

            if total_trades > 0:
                win_rate = winning_trades / total_trades
            else:
                win_rate = 0.0

            # Calculate P&L metrics
            pnl_values = [
                t.get("pnl", 0)
                for t in self.trading_history
                if t.get("action") == "position_close"
            ]

            if pnl_values:
                total_pnl = sum(pnl_values)
                avg_pnl = np.mean(pnl_values)
                max_profit = max(pnl_values) if pnl_values else 0
                max_loss = min(pnl_values) if pnl_values else 0
            else:
                total_pnl = avg_pnl = max_profit = max_loss = 0

            self.performance_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "last_update": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")

    async def _load_trading_history(self):
        """Load trading history from storage."""
        try:
            history_file = Path("data/trading_history.json")
            if history_file.exists():
                with open(history_file, "r") as f:
                    data = json.load(f)

                self.trading_history = data.get("trades", [])
                self.performance_metrics = data.get("performance", {})

                self.logger.info(
                    f"✅ Loaded {len(self.trading_history)} trading records"
                )
            else:
                self.logger.info("No trading history found, starting fresh")

        except Exception as e:
            self.logger.error(f"❌ Error loading trading history: {e}")

    async def save_trading_history(self):
        """Save trading history to storage."""
        try:
            data = {
                "trades": self.trading_history,
                "performance": self.performance_metrics,
                "last_save": datetime.now().isoformat(),
            }

            history_file = Path("data/trading_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)

            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"❌ Error saving trading history: {e}")

    async def shutdown(self):
        """Shutdown the trading intelligence system."""
        try:
            self._is_running = False

            if self._update_task and not self._update_task.done():
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass

            # Save trading history
            await self.save_trading_history()

            self.logger.info("✅ SR Trading Intelligence shutdown complete")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


async def create_sr_trading_intelligence(
    config: Dict[str, Any]
) -> SRTradingIntelligence:
    """Factory function to create and initialize SR Trading Intelligence."""
    intelligence = SRTradingIntelligence(config)
    if await intelligence.initialize():
        return intelligence
    return None
