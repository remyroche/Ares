# src/validation/critical_path_validators.py

"""
Critical path type validators for trading system safety.
"""

from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

from src.custom_types.validation import (
import RuntimeTypeError,
    RuntimeTypeError,
    TypeValidator,
    validate_market_data,
    validate_model_input,
)
from src.custom_types import (
import OrderRequest,
    OrderRequest,
    PositionInfo,
    TradeDecision,
    TradingSignal,
)
from src.utils.structured_logging import get_correlation_id
from src.utils.warning_symbols import error, failed

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CriticalPathValidator:
    """Validator for critical trading system paths."""

    @staticmethod
    def validate_trading_signal(signal: Any) -> TradingSignal:
    pass
    pass
        """Validate trading signal with comprehensive checks."""
        try:
            validated_signal = TypeValidator.validate_type(
                signal, TradingSignal, "trading_signal"
    except Exception as e:
        pass
    except Exception as e:
        pass
            )

            # Additional business logic validation
            if validated_signal["strength"] < 0.0 or validated_signal["strength"] > 1.0:
    pass
    pass
                raise RuntimeTypeError(
                    TradingSignal,
                    signal,
                    "signal strength must be between 0.0 and 1.0",
                )

            if (
                validated_signal["confidence"] < 0.0
                or validated_signal["confidence"] > 1.0
            ):
                raise RuntimeTypeError(
                    TradingSignal,
                    signal,
                    "confidence must be between 0.0 and 1.0",
                )

            return validated_signal

        except Exception as e:
            logger.exception(f"Trading signal validation failed: {e}")
            raise

    @staticmethod
    def validate_trade_decision(decision: Any) -> TradeDecision:
    pass
    pass
        """Validate trade decision with risk checks."""
        try:
            validated_decision = TypeValidator.validate_type(
                decision, TradeDecision, "trade_decision"
    except Exception as e:
        pass
    except Exception as e:
        pass
            )

            # Risk validation
            if validated_decision["quantity"] <= 0:
    pass
    pass
                raise RuntimeTypeError(
                    TradeDecision,
                    decision,
                    "quantity must be positive",
                )

            if (
                validated_decision["risk_score"] < 0.0
                or validated_decision["risk_score"] > 1.0
            ):
                raise RuntimeTypeError(
                    TradeDecision,
                    decision,
                    "risk score must be between 0.0 and 1.0",
                )

            # Validate stop loss and take profit relationships
            if "stop_loss" in validated_decision and "price" in validated_decision:
    pass
    pass
                if (
                    validated_decision["action"] in ["open_long"]
                    and validated_decision["stop_loss"]
                ):
                    if validated_decision["stop_loss"] >= validated_decision["price"]:
    pass
    pass
                        raise RuntimeTypeError(
                            TradeDecision,
                            decision,
                            "stop loss must be below entry price for long positions",
                        )

                elif (
                    validated_decision["action"] in ["open_short"]
                    and validated_decision["stop_loss"]
                    and validated_decision["stop_loss"] <= validated_decision["price"]
                ):
                    raise RuntimeTypeError(
                        TradeDecision,
                        decision,
                        "stop loss must be above entry price for short positions",
                    )

            return validated_decision

        except Exception as e:
            logger.exception(f"Trade decision validation failed: {e}")
            raise

    @staticmethod
    def validate_order_request(order: Any) -> OrderRequest:
    pass
    pass
        """Validate order request for execution safety."""
        try:
            validated_order = TypeValidator.validate_type(
                order, OrderRequest, "order_request"
    except Exception as e:
        pass
    except Exception as e:
        pass
            )

            # Order validation
            if validated_order["quantity"] <= 0:
    pass
    pass
                raise RuntimeTypeError(
                    OrderRequest,
                    order,
                    "order quantity must be positive",
                )

            if validated_order["type"] == "limit" and "price" not in validated_order:
    pass
    pass
                raise RuntimeTypeError(
                    OrderRequest,
                    order,
                    "limit orders must have a price",
                )

            if (
                validated_order["type"] in ["stop", "stop_limit"]
                and "stop_price" not in validated_order
            ):
                raise RuntimeTypeError(
                    OrderRequest,
                    order,
                    "stop orders must have a stop price",
                )

            return validated_order

        except Exception as e:
            logger.exception(f"Order request validation failed: {e}")
            raise

    @staticmethod
    def validate_position_info(position: Any) -> PositionInfo:
    pass
    pass
        """Validate position information."""
        try:
            validated_position = TypeValidator.validate_type(
                position, PositionInfo, "position_info"
    except Exception as e:
        pass
    except Exception as e:
        pass
            )

            # Position validation
            if validated_position["size"] < 0:
    pass
    pass
                raise RuntimeTypeError(
                    PositionInfo,
                    position,
                    "position size cannot be negative",
                )

            if validated_position["leverage"] <= 0:
    pass
    pass
                raise RuntimeTypeError(
                    PositionInfo,
                    position,
                    "leverage must be positive",
                )

            return validated_position

        except Exception as e:
            logger.exception(f"Position info validation failed: {e}")
            raise


def validate_trading_signal_critical(func: Callable) -> Callable:
    pass
    pass
    """Decorator for critical trading signal validation."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    pass
    pass
        result = func(*args, **kwargs)
        return CriticalPathValidator.validate_trading_signal(result)

    return wrapper


def validate_trade_decision_critical(func: Callable) -> Callable:
    pass
    pass
    """Decorator for critical trade decision validation."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    pass
    pass
        result = func(*args, **kwargs)
        if result is not None:
    pass
    pass
            return CriticalPathValidator.validate_trade_decision(result)
        return result

    return wrapper


def validate_order_execution_critical(func: Callable) -> Callable:
    pass
    pass
    """Decorator for critical order execution validation."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    pass
    pass
        # Validate input order if present
        if args and hasattr(args[0], "__dict__"):
    pass
    pass
            for arg in args:
    pass
    pass
                if isinstance(arg, dict) and "symbol" in arg and "side" in arg:
    pass
    pass
                    CriticalPathValidator.validate_order_request(arg)

        return func(*args, **kwargs)

    return wrapper


def validate_market_data_critical(func: Callable) -> Callable:
    pass
    pass
    """Decorator for critical market data validation."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    pass
    pass
        result = func(*args, **kwargs)
        if isinstance(result, dict):
    pass
    pass
            return validate_market_data(result)
        return result

    return wrapper


def validate_ml_input_critical(func: Callable) -> Callable:
    pass
    pass
    """Decorator for critical ML input validation."""

    @wraps(func)
    def wrapper(*args, **kwargs):
    pass
    pass
        result = func(*args, **kwargs)
        if isinstance(result, dict) and "features" in result:
    pass
    pass
            return validate_model_input(result)
        return result

    return wrapper


class TypeSafetyMonitor:
    """Monitor type safety violations in production."""

    def __init__(self):
    pass
    pass
        self.violations: list = []
        self.violation_counts: dict = {}

    def record_violation(self, violation: RuntimeTypeError) -> None:
    pass
    pass
        """Record a type safety violation."""
        self.violations.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "expected_type": str(violation.expected_type),
                "actual_type": str(type(violation.actual_value)),
                "context": violation.context,
                "message": str(violation),
                "correlation_id": get_correlation_id(),
            }
        )

        # Count violations by type
        violation_key = f"{violation.expected_type}_{violation.context}"
        self.violation_counts[violation_key] = (
            self.violation_counts.get(violation_key, 0) + 1
        )

        # Log critical violations (correlation_id is included by filter)
        logger.warning(f"Type safety violation: {violation}")

    def get_violation_summary(self) -> dict:
    pass
    pass
        """Get summary of type safety violations."""
        return {
            "total_violations": len(self.violations),
            "violation_counts": self.violation_counts.copy(),
            "recent_violations": self.violations[-10:] if self.violations else [],
        }

    def reset_violations(self) -> None:
    pass
    pass
        """Reset violation tracking."""
        self.violations.clear()
        self.violation_counts.clear()


# Global type safety monitor
_type_safety_monitor = TypeSafetyMonitor()


def get_type_safety_monitor() -> TypeSafetyMonitor:
    pass
    pass
    """Get the global type safety monitor."""
    return _type_safety_monitor


def safe_execute_with_validation(func: Callable[..., T], *args, **kwargs) -> T | None:
    pass
    pass
    """
    Execute function with comprehensive type validation and error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or None if validation fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except RuntimeTypeError as e:
        _type_safety_monitor.record_violation(e)
        print(failed(f"Type validation failed in {func.__name__}: {e}"))
        return None
    except Exception as e:
        print(error(f"Unexpected error in {func.__name__}: {e}"))
        return None
