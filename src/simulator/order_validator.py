"""
Order Validator

Validates orders before execution, checking balance, position limits,
price reasonableness, and direction constraints.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.tprint import tprint
from .config import SimulatorConfig


class ValidationResult(Enum):
    """Order validation results"""
    VALID = "valid"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    EXCEEDS_POSITION_LIMIT = "exceeds_position_limit"
    PRICE_DEVIATION_TOO_LARGE = "price_deviation_too_large"
    DIRECTION_CONSTRAINT_VIOLATION = "direction_constraint_violation"
    INVALID_QUANTITY = "invalid_quantity"
    INVALID_SYMBOL = "invalid_symbol"


@dataclass
class ValidationResponse:
    """Response from order validation"""
    is_valid: bool
    result: ValidationResult
    message: str
    suggested_quantity: Optional[float] = None


class OrderValidator:
    """
    Validate orders before execution.
    
    Checks:
    - Sufficient balance
    - Position size limits
    - Price reasonableness
    - Quantity precision
    - Direction constraints
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize order validator.

        Args:
            config: Simulator configuration
        """
        tprint(f"[ORDER_VAL] __init__: Initializing order validator")
        self.config = config
        self.logger = logging.getLogger(__name__)
        tprint(f"[ORDER_VAL] __init__ -> initialized with max_position_size={config.max_position_size_usd}, max_exposure={config.max_total_exposure_usd}")
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_balance: float,
        current_positions: Dict[str, Dict[str, Any]],
        direction_constraint: Optional[str] = None,
        current_price: Optional[float] = None
    ) -> ValidationResponse:
        """
        Validate an order before execution.

        Args:
            symbol: Trading symbol
            side: Order side ("buy", "sell", "long", "short")
            quantity: Order quantity
            price: Order price
            current_balance: Current account balance
            current_positions: Dict of current positions {symbol: position_data}
            direction_constraint: Direction constraint ("long", "short", "both", or None)
            current_price: Current market price for price validation

        Returns:
            ValidationResponse indicating if order is valid
        """
        tprint(f"[ORDER_VAL] validate_order: symbol={symbol}, side={side}, qty={quantity}, price={price}, balance={current_balance:.2f}, positions={len(current_positions)}")

        # Check quantity
        if quantity <= 0:
            tprint(f"[ORDER_VAL] validate_order -> INVALID_QUANTITY: {quantity}", color="red")
            return ValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_QUANTITY,
                message=f"Invalid quantity: {quantity}"
            )

        # Check symbol
        if not symbol or len(symbol) == 0:
            tprint(f"[ORDER_VAL] validate_order -> INVALID_SYMBOL", color="red")
            return ValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_SYMBOL,
                message="Invalid symbol"
            )
        
        # Check balance for buy orders
        is_buy = side.lower() in ["buy", "long"]
        if is_buy:
            notional_value = quantity * price
            tprint(f"[ORDER_VAL] validate_order: Checking balance for buy order, notional={notional_value:.2f}, balance={current_balance:.2f}")
            if notional_value > current_balance:
                max_qty = current_balance / price if price > 0 else 0
                tprint(f"[ORDER_VAL] validate_order -> INSUFFICIENT_BALANCE: need {notional_value:.2f}, have {current_balance:.2f}", color="red")
                return ValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INSUFFICIENT_BALANCE,
                    message=f"Insufficient balance. Need {notional_value:.2f}, have {current_balance:.2f}",
                    suggested_quantity=max_qty
                )
        
        # Check position limits
        position_size = self._calculate_position_impact(
            symbol, side, quantity, price, current_positions
        )
        tprint(f"[ORDER_VAL] validate_order: Position impact={position_size:.2f} USD (limit={self.config.max_position_size_usd:.2f})")

        if abs(position_size) > self.config.max_position_size_usd:
            tprint(f"[ORDER_VAL] validate_order -> EXCEEDS_POSITION_LIMIT: {position_size:.2f} > {self.config.max_position_size_usd:.2f}", color="red")
            return ValidationResponse(
                is_valid=False,
                result=ValidationResult.EXCEEDS_POSITION_LIMIT,
                message=f"Position size {position_size:.2f} USD exceeds limit {self.config.max_position_size_usd:.2f} USD"
            )

        # Check total exposure
        total_exposure = self._calculate_total_exposure(current_positions, position_size)
        tprint(f"[ORDER_VAL] validate_order: Total exposure={total_exposure:.2f} USD (limit={self.config.max_total_exposure_usd:.2f})")
        if total_exposure > self.config.max_total_exposure_usd:
            tprint(f"[ORDER_VAL] validate_order -> EXCEEDS_TOTAL_EXPOSURE: {total_exposure:.2f} > {self.config.max_total_exposure_usd:.2f}", color="red")
            return ValidationResponse(
                is_valid=False,
                result=ValidationResult.EXCEEDS_POSITION_LIMIT,
                message=f"Total exposure {total_exposure:.2f} USD exceeds limit {self.config.max_total_exposure_usd:.2f} USD"
            )
        
        # Check price deviation
        if current_price and current_price > 0:
            deviation = abs(price - current_price) / current_price
            tprint(f"[ORDER_VAL] validate_order: Price deviation={deviation:.2%} (threshold={self.config.price_deviation_threshold_pct:.2%})")
            if deviation > self.config.price_deviation_threshold_pct:
                tprint(f"[ORDER_VAL] validate_order -> PRICE_DEVIATION_TOO_LARGE: {deviation:.2%} > {self.config.price_deviation_threshold_pct:.2%}", color="red")
                return ValidationResponse(
                    is_valid=False,
                    result=ValidationResult.PRICE_DEVIATION_TOO_LARGE,
                    message=f"Price deviation {deviation:.2%} exceeds threshold {self.config.price_deviation_threshold_pct:.2%}"
                )

        # Check direction constraint
        if direction_constraint:
            constraint_result = self._check_direction_constraint(
                side, direction_constraint
            )
            if not constraint_result[0]:
                tprint(f"[ORDER_VAL] validate_order -> DIRECTION_CONSTRAINT_VIOLATION: {constraint_result[1]}", color="red")
                return ValidationResponse(
                    is_valid=False,
                    result=ValidationResult.DIRECTION_CONSTRAINT_VIOLATION,
                    message=constraint_result[1]
                )

        # All checks passed
        tprint(f"[ORDER_VAL] validate_order -> VALID: All checks passed", color="green")
        self.logger.debug(
            f"Order validated: {symbol} {side} qty={quantity} price={price}"
        )

        return ValidationResponse(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Order is valid"
        )
    
    def _calculate_position_impact(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_positions: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate total position value after this order."""
        tprint(f"[ORDER_VAL] _calculate_position_impact: symbol={symbol}, side={side}, qty={quantity}, price={price}")

        # Get current position for symbol
        current_pos = current_positions.get(symbol, {
            "quantity": 0.0,
            "direction": None,
            "avg_entry_price": 0.0
        })

        current_qty = current_pos.get("quantity", 0.0)
        current_direction = current_pos.get("direction", None)

        # Determine if this order increases or decreases position
        is_buy = side.lower() in ["buy", "long"]
        is_closing = (is_buy and current_direction == "short") or (not is_buy and current_direction == "long")

        if is_closing:
            # Closing position, calculate net exposure after close
            closed_qty = min(abs(current_qty), quantity)
            remaining_qty = abs(current_qty) - closed_qty
            new_exposure = remaining_qty * price
            tprint(f"[ORDER_VAL] _calculate_position_impact -> closing: remaining_qty={remaining_qty}, new_exposure={new_exposure:.2f}")
        else:
            # Opening or adding to position
            new_qty = current_qty + (quantity if is_buy else -quantity)
            new_exposure = abs(new_qty) * price
            tprint(f"[ORDER_VAL] _calculate_position_impact -> opening/adding: new_qty={new_qty}, new_exposure={new_exposure:.2f}")

        return new_exposure
    
    def _calculate_total_exposure(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        new_position_impact: float
    ) -> float:
        """Calculate total exposure across all positions."""
        tprint(f"[ORDER_VAL] _calculate_total_exposure: num_positions={len(current_positions)}, new_impact={new_position_impact:.2f}")

        total = 0.0

        # Sum existing positions
        for pos_data in current_positions.values():
            qty = pos_data.get("quantity", 0.0)
            price = pos_data.get("avg_entry_price", 0.0)
            if price > 0:
                total += abs(qty) * price

        # Add new position impact
        total += new_position_impact

        tprint(f"[ORDER_VAL] _calculate_total_exposure -> total={total:.2f}")
        return total
    
    def _check_direction_constraint(
        self,
        side: str,
        direction_constraint: str
    ) -> Tuple[bool, str]:
        """
        Check if order side violates direction constraint.

        Returns:
            Tuple of (is_valid, error_message)
        """
        tprint(f"[ORDER_VAL] _check_direction_constraint: side={side}, constraint={direction_constraint}")

        is_buy = side.lower() in ["buy", "long"]
        constraint_lower = direction_constraint.lower()

        if constraint_lower == "long":
            if not is_buy:
                tprint(f"[ORDER_VAL] _check_direction_constraint -> FAILED: long constraint violated by {side}", color="red")
                return False, f"Direction constraint 'long' only allows buy orders, got {side}"

        elif constraint_lower == "short":
            if is_buy:
                tprint(f"[ORDER_VAL] _check_direction_constraint -> FAILED: short constraint violated by {side}", color="red")
                return False, f"Direction constraint 'short' only allows sell orders, got {side}"

        elif constraint_lower == "both":
            # Both directions allowed, no constraint
            pass

        else:
            tprint(f"[ORDER_VAL] _check_direction_constraint -> FAILED: unknown constraint {direction_constraint}", color="red")
            return False, f"Unknown direction constraint: {direction_constraint}"

        tprint(f"[ORDER_VAL] _check_direction_constraint -> VALID", color="green")
        return True, ""
