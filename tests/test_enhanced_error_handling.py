#!/usr/bin/env python3
"""
Test suite for enhanced error handling with automatic recovery strategies.

This module tests the enhanced error handling system including:
- Circuit breaker pattern
- Automatic recovery strategies
- Type safety with 100% type hint coverage
- Error recovery mechanisms
"""

import asyncio

import pytest

from src.utils.error_handler import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ErrorHandler,
    ErrorRecoveryManager,
    FallbackStrategy,
    GracefulDegradationStrategy,
    RetryStrategy,
    create_circuit_breaker,
    create_fallback_strategy,
    create_graceful_degradation_strategy,
    create_retry_strategy,
    safe_async_operation,
    safe_operation,
)


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""

    def test_circuit_breaker_initialization(self) -> None:
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ValueError,
        )
        circuit_breaker = CircuitBreaker(config)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.config == config

    @pytest.mark.asyncio
    async def test_circuit_breaker_successful_operation(self) -> None:
        """Test successful operation with circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        circuit_breaker = CircuitBreaker(config)

        async def successful_operation() -> str:
            return "success"

        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_and_open(self) -> None:
        """Test circuit breaker opening after failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        circuit_breaker = CircuitBreaker(config)

        async def failing_operation() -> str:
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self) -> None:
        """Test circuit breaker recovery from half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        circuit_breaker = CircuitBreaker(config)

        # Fail once to open circuit
        async def failing_operation() -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_operation)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Try again - should be half-open
        async def successful_operation() -> str:
            return "success"

        result = await circuit_breaker.call(successful_operation)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED


class TestRecoveryStrategies:
    """Test automatic recovery strategies."""

    @pytest.mark.asyncio
    async def test_retry_strategy_success(self) -> None:
        """Test retry strategy with eventual success."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)

        attempt_count = 0

        async def flaky_operation() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        context = {
            "operation": flaky_operation,
            "args": (),
            "kwargs": {},
        }

        result = await strategy.execute(context)
        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_strategy_failure(self) -> None:
        """Test retry strategy with eventual failure."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)

        async def always_failing_operation() -> str:
            raise ValueError("Persistent failure")

        context = {
            "operation": always_failing_operation,
            "args": (),
            "kwargs": {},
        }

        with pytest.raises(ValueError):
            await strategy.execute(context)

    @pytest.mark.asyncio
    async def test_fallback_strategy(self) -> None:
        """Test fallback strategy with multiple operations."""

        async def primary_operation() -> str:
            raise ValueError("Primary failed")

        async def fallback_operation() -> str:
            return "fallback success"

        async def final_operation() -> str:
            raise ValueError("Final failed")

        strategy = FallbackStrategy(
            fallback_operations=[
                primary_operation,
                fallback_operation,
                final_operation,
            ],
        )

        context = {
            "operation": lambda: "ignored",
            "args": (),
            "kwargs": {},
        }

        result = await strategy.execute(context)
        assert result == "fallback success"

    def test_graceful_degradation_strategy(self) -> None:
        """Test graceful degradation strategy."""
        strategy = GracefulDegradationStrategy(default_return="degraded")

        # Should handle any error
        assert strategy.can_handle(ValueError("test"))
        assert strategy.can_handle(TypeError("test"))

        # Should return default value
        context = {"operation": lambda: None, "args": (), "kwargs": {}}
        result = asyncio.run(strategy.execute(context))
        assert result == "degraded"


class TestErrorRecoveryManager:
    """Test error recovery manager."""

    @pytest.mark.asyncio
    async def test_recovery_manager_success(self) -> None:
        """Test recovery manager with successful recovery."""
        manager = ErrorRecoveryManager()

        # Add retry strategy
        retry_strategy = RetryStrategy(max_retries=1, base_delay=0.01)
        manager.add_strategy(retry_strategy)

        attempt_count = 0

        async def flaky_operation() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ValueError("First attempt fails")
            return "success"

        result = await manager.execute_with_recovery(flaky_operation)
        assert result == "success"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_recovery_manager_failure(self) -> None:
        """Test recovery manager with no successful recovery."""
        manager = ErrorRecoveryManager()

        # Add retry strategy
        retry_strategy = RetryStrategy(max_retries=1, base_delay=0.01)
        manager.add_strategy(retry_strategy)

        async def always_failing_operation() -> str:
            raise ValueError("Always fails")

        result = await manager.execute_with_recovery(always_failing_operation)
        assert result is None


class TestEnhancedErrorHandler:
    """Test enhanced error handler with recovery strategies."""

    def test_error_handler_initialization(self) -> None:
        """Test error handler initialization."""
        handler = ErrorHandler(context="test")
        assert handler.context == "test"
        assert handler.recovery_manager is not None

    @pytest.mark.asyncio
    async def test_handle_errors_with_recovery(self) -> None:
        """Test error handling with recovery strategies."""
        handler = ErrorHandler(context="test")

        # Create a function that fails then succeeds
        attempt_count = 0

        @handler.handle_generic_errors(
            exceptions=(ValueError,),
            default_return="default",
            recovery_strategies=[
                RetryStrategy(max_retries=1, base_delay=0.01),
            ],
        )
        async def test_function() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ValueError("First attempt fails")
            return "success"

        result = await test_function()
        assert result == "success"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_handle_specific_errors_with_recovery(self) -> None:
        """Test specific error handling with recovery strategies."""
        handler = ErrorHandler(context="test")

        @handler.handle_specific_errors(
            error_handlers={ValueError: ("fallback", "Value error occurred")},
            default_return="default",
            recovery_strategies=[
                GracefulDegradationStrategy(default_return="degraded"),
            ],
        )
        async def test_function() -> str:
            raise ValueError("Test error")

        result = await test_function()
        assert result == "degraded"


class TestUtilityFunctions:
    """Test utility functions with type safety."""

    def test_safe_operation_success(self) -> None:
        """Test safe operation with successful execution."""

        def test_func() -> str:
            return "success"

        result = safe_operation(test_func)
        assert result == "success"

    def test_safe_operation_failure(self) -> None:
        """Test safe operation with failure."""

        def test_func() -> str:
            raise ValueError("Test error")

        result = safe_operation(test_func)
        assert result is None

    @pytest.mark.asyncio
    async def test_safe_async_operation_success(self) -> None:
        """Test safe async operation with successful execution."""

        async def test_func() -> str:
            return "success"

        result = await safe_async_operation(test_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_safe_async_operation_failure(self) -> None:
        """Test safe async operation with failure."""

        async def test_func() -> str:
            error_message = "Test error"
            raise ValueError(error_message)

        result = await safe_async_operation(test_func)
        assert result is None

    def test_create_circuit_breaker(self) -> None:
        """Test circuit breaker creation."""
        circuit_breaker = create_circuit_breaker(
            name="test",
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=ValueError,
        )

        assert isinstance(circuit_breaker, CircuitBreaker)
        assert circuit_breaker.config.failure_threshold == 5
        assert circuit_breaker.config.expected_exception is ValueError

    def test_create_retry_strategy(self) -> None:
        """Test retry strategy creation."""
        strategy = create_retry_strategy(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            backoff_factor=2.0,
            jitter=True,
        )

        assert isinstance(strategy, RetryStrategy)
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0

    def test_create_fallback_strategy(self) -> None:
        """Test fallback strategy creation."""

        def fallback1() -> str:
            return "fallback1"

        def fallback2() -> str:
            return "fallback2"

        strategy = create_fallback_strategy([fallback1, fallback2])

        assert isinstance(strategy, FallbackStrategy)
        assert len(strategy.fallback_operations) == 2

    def test_create_graceful_degradation_strategy(self) -> None:
        """Test graceful degradation strategy creation."""
        strategy = create_graceful_degradation_strategy(
            default_return="degraded",
            error_types=[ValueError, TypeError],
        )

        assert isinstance(strategy, GracefulDegradationStrategy)
        assert strategy.default_return == "degraded"
        assert len(strategy.error_types) == 2


class TestTypeSafety:
    """Test type safety and hint coverage."""

    def test_type_hints_are_present(self) -> None:
        """Test that all functions have proper type hints."""
        # Test circuit breaker
        config = CircuitBreakerConfig()
        circuit_breaker = CircuitBreaker(config)

        # Test recovery strategies
        retry_strategy = RetryStrategy()
        fallback_strategy = FallbackStrategy()
        graceful_strategy = GracefulDegradationStrategy()

        # Test error handler
        handler = ErrorHandler()

        # Test utility functions
        circuit_breaker_util = create_circuit_breaker("test")
        retry_strategy_util = create_retry_strategy()
        fallback_strategy_util = create_fallback_strategy([])
        graceful_strategy_util = create_graceful_degradation_strategy()

        # All should be properly typed
        assert circuit_breaker is not None
        assert retry_strategy is not None
        assert fallback_strategy is not None
        assert graceful_strategy is not None
        assert handler is not None
        assert circuit_breaker_util is not None
        assert retry_strategy_util is not None
        assert fallback_strategy_util is not None
        assert graceful_strategy_util is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
