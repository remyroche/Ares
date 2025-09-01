# src/tactician/position_division_strategy.py

"""
Position Division Strategy for tactical position management.
Defines strategies for multiple positions, take profit, stop loss, and position closure.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.error_handler import (
    handle_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    warning,
)

class PositionDivisionStrategy:
    """
    Position Division Strategy for managing multiple positions and their lifecycle.

    Features:
    - Multiple position management
    - Take profit and stop loss strategies
    - Position closure logic
    - Risk management rules
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the position division strategy.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PositionDivisionStrategy")

        # Configuration
        self.strategy_config = config.get("position_division_strategy", {})
        self.max_positions = self.strategy_config.get("max_positions", 5)
        self.position_size_limit = self.strategy_config.get("position_size_limit", 0.2)  # 20% per position
        self.take_profit_pct = self.strategy_config.get("take_profit_pct", 0.02)  # 2%
        self.stop_loss_pct = self.strategy_config.get("stop_loss_pct", 0.01)  # 1%

        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, Any] = {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position division strategy initialization"
    )
    def _validate_configuration(self) -> bool:
        """
        Validate position division strategy configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.max_positions <= 0:
                self.logger.error(invalid("Max positions must be positive"))
                return False

            if not 0 < self.position_size_limit <= 1:
                self.logger.error(invalid("Position size limit must be between 0 and 1"))
                return False

            if self.take_profit_pct <= 0:
                self.logger.error(invalid("Take profit percentage must be positive"))
                return False

            if self.stop_loss_pct <= 0:
                self.logger.error(invalid("Stop loss percentage must be positive"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position division calculation"
    )
    def _calculate_num_positions(self, confidence_score: float) -> int:
        """
        Calculate number of positions based on confidence score.

        Args:
            confidence_score: Confidence score (0-1)

        Returns:
            int: Number of positions
        """
        try:
            # Higher confidence = fewer positions (more concentrated)
            # Lower confidence = more positions (more diversified)

            if confidence_score >= 0.8:
                return 1  # High confidence = single position
            elif confidence_score >= 0.6:
                return 2  # Medium-high confidence = 2 positions
            elif confidence_score >= 0.4:
                return 3  # Medium confidence = 3 positions
            elif confidence_score >= 0.2:
                return 4  # Medium-low confidence = 4 positions
            else:
                return min(5, self.max_positions)  # Low confidence = max positions

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating number of positions: {e}"))
            return 1

    def _calculate_position_sizes(
        self,
        total_capital: float,
        num_positions: int,
        confidence_score: float
    ) -> List[float]:
        """
        Calculate position sizes for each position.

        Args:
            total_capital: Total available capital
            num_positions: Number of positions
            confidence_score: Confidence score

        Returns:
            List[float]: Position sizes
        """
        try:
            position_sizes = []

            if num_positions == 1:
                # Single position - use full allocation
                position_sizes.append(total_capital * self.position_size_limit)
            else:
                # Multiple positions - distribute capital
                base_size = total_capital * self.position_size_limit / num_positions

                # Adjust based on confidence (higher confidence = larger first position)
                confidence_multiplier = 1 + (confidence_score - 0.5) * 0.5  # 0.75 to 1.25

                for i in range(num_positions):
                    if i == 0:
                        # First position gets confidence-adjusted size
                        size = base_size * confidence_multiplier
                    else:
                        # Remaining positions get equal size
                        size = base_size

                    position_sizes.append(size)

            return position_sizes

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating position sizes: {e}"))
            return [total_capital * 0.1]  # Fallback to 10%

    def _calculate_tp_sl_levels(self, market_conditions: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Calculate take profit and stop loss levels.

        Args:
            market_conditions: Current market conditions

        Returns:
            Dict: Take profit and stop loss levels
        """
        try:
            # Get market volatility
            volatility = market_conditions.get("volatility", 0.02)  # Default 2%

            # Adjust TP/SL based on volatility
            tp_adjustment = min(volatility * 2, 0.05)  # Max 5%
            sl_adjustment = min(volatility, 0.03)  # Max 3%

            take_profit_levels = []
            stop_loss_levels = []

            # Calculate levels for each position
            for i in range(self.max_positions):
                # Progressive TP/SL levels
                tp_level = self.take_profit_pct + (i * 0.005) + tp_adjustment  # Increase by 0.5% per position
                sl_level = self.stop_loss_pct + (i * 0.002) + sl_adjustment  # Increase by 0.2% per position

                take_profit_levels.append(tp_level)
                stop_loss_levels.append(sl_level)

            return {
                "take_profit": take_profit_levels,
                "stop_loss": stop_loss_levels
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating TP/SL levels: {e}"))
            return {
                "take_profit": [self.take_profit_pct] * self.max_positions,
                "stop_loss": [self.stop_loss_pct] * self.max_positions
            }

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position management"
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closure"
    )
    def _update_performance_metrics(self, closure_record: Dict[str, Any]) -> None:
        """
        Update performance metrics based on closed position.

        Args:
            closure_record: Position closure record
        """
        try:
            # Update basic metrics
            total_positions = len(self.position_history)
            total_pnl = sum(pos.get("pnl", 0) for pos in self.position_history)
            winning_positions = sum(1 for pos in self.position_history if pos.get("pnl", 0) > 0)

            self.strategy_performance.update({
                "total_positions": total_positions,
                "total_pnl": total_pnl,
                "winning_positions": winning_positions,
                "losing_positions": total_positions - winning_positions,
                "win_rate": winning_positions / total_positions if total_positions > 0 else 0.0,
                "average_pnl": total_pnl / total_positions if total_positions > 0 else 0.0,
                "last_updated": datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating performance metrics: {e}"))
