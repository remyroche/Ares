# src/validation/critical_path_validators.py

"""
Critical path type validators for trading system safety.
"""

from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar, Dict, List, Optional, Union

from src.custom_types.validation import (
    RuntimeTypeError,
    TypeValidator,
    validate_market_data,
    validate_model_input,
)
from src.custom_types import (
    OrderRequest,
    PositionInfo,
    TradeDecision,
    TradingSignal,
)
from src.utils.structured_logging import get_correlation_id
from src.utils.warning_symbols import error, failed
from src.utils.error_handler import handle_errors

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CriticalPathValidator:
    """Validator for critical trading system paths."""
    
    def __init__(self):
        """Initialize CriticalPathValidator."""
        self.logger = logger
        self.is_initialized = False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="criticalpathvalidator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize CriticalPathValidator."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    @staticmethod
    def validate_trading_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trading signal data structure and business logic."""
        try:
            validated_signal = TypeValidator.validate_type(
                signal, TradingSignal, "trading_signal"
            )

            # Additional business logic validation
            if validated_signal["strength"] < 0.0 or validated_signal["strength"] > 1.0:
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
    def validate_trade_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade decision data structure and business logic."""
        try:
            validated_decision = TypeValidator.validate_type(
                decision, TradeDecision, "trade_decision"
            )

            # Risk validation
            if validated_decision["quantity"] <= 0:
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
                if (
                    validated_decision["action"] in ["open_long"]
                    and validated_decision["stop_loss"]
                ):
                    if validated_decision["stop_loss"] >= validated_decision["price"]:
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
    def validate_order_request(order: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order request data structure and business logic."""
        try:
            validated_order = TypeValidator.validate_type(
                order, OrderRequest, "order_request"
            )

            # Order validation
            if validated_order["quantity"] <= 0:
                raise RuntimeTypeError(
                    OrderRequest,
                    order,
                    "order quantity must be positive",
                )

            if validated_order["type"] == "limit" and "price" not in validated_order:
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
    def validate_position_info(position: Dict[str, Any]) -> Dict[str, Any]:
        """Validate position info data structure and business logic."""
        try:
            validated_position = TypeValidator.validate_type(
                position, PositionInfo, "position_info"
            )

            # Position validation
            if validated_position["size"] < 0:
                raise RuntimeTypeError(
                    PositionInfo,
                    position,
                    "position size cannot be negative",
                )

            if validated_position["leverage"] <= 0:
                raise RuntimeTypeError(
                    PositionInfo,
                    position,
                    "leverage must be positive",
                )

            return validated_position

        except Exception as e:
            logger.exception(f"Position info validation failed: {e}")
            raise


def validate_trading_signal_critical(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate trading signal output."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        result = func(*args, **kwargs)
        return CriticalPathValidator.validate_trading_signal(result)
    return wrapper


def validate_trade_decision_critical(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate trade decision output."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        result = func(*args, **kwargs)
        if result is not None:
            return CriticalPathValidator.validate_trade_decision(result)
        return result
    return wrapper


def validate_order_execution_critical(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate order execution input and output."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Validate input order if present
        if args and hasattr(args[0], "__dict__"):
            for arg in args:
                if isinstance(arg, dict) and "symbol" in arg and "side" in arg:
                    CriticalPathValidator.validate_order_request(arg)

        return func(*args, **kwargs)
    return wrapper


def validate_market_data_critical(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate market data output."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            return validate_market_data(result)
        return result
    return wrapper


def validate_ml_input_critical(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to validate ML model input."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        result = func(*args, **kwargs)
        if isinstance(result, dict) and "features" in result:
            return validate_model_input(result)
        return result
    return wrapper


class TypeSafetyMonitor:
    """Monitor type safety violations in production."""
    
    def __init__(self):
        """Initialize TypeSafetyMonitor."""
        self.violations: List[Dict[str, Any]] = []
        self.violation_counts: Dict[str, int] = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="typesafetymonitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TypeSafetyMonitor."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    def record_violation(self, violation: RuntimeTypeError) -> None:
        """Record a type safety violation."""
        self.violations.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "expected_type": str(violation.expected_type),
            "actual_type": str(type(violation.actual_value)),
            "context": violation.context,
            "message": str(violation),
            "correlation_id": get_correlation_id(),
        })

        # Count violations by type
        violation_key = f"{violation.expected_type}_{violation.context}"
        self.violation_counts[violation_key] = (
            self.violation_counts.get(violation_key, 0) + 1
        )

        # Log critical violations (correlation_id is included by filter)
        logger.warning(f"Type safety violation: {violation}")

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of type safety violations."""
        return {
            "total_violations": len(self.violations),
            "violation_counts": self.violation_counts.copy(),
            "recent_violations": self.violations[-10:] if self.violations else [],
        }

    def reset_violations(self) -> None:
        """Reset violation tracking."""
        self.violations.clear()
        self.violation_counts.clear()


# Global type safety monitor
_type_safety_monitor = TypeSafetyMonitor()


def get_type_safety_monitor() -> TypeSafetyMonitor:
    """Get the global type safety monitor instance."""
    return _type_safety_monitor


def safe_execute_with_validation(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Decorator to safely execute functions with type validation."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except RuntimeTypeError as e:
            _type_safety_monitor.record_violation(e)
            print(failed(f"Type validation failed in {func.__name__}: {e}"))
            return None
        except Exception as e:
            print(error(f"Unexpected error in {func.__name__}: {e}"))
            return None
    return wrapper
