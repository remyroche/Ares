from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Callable, Awaitable, Union
import asyncio

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel

@dataclass
class PendingTrade:
    """Represents a pending trade waiting in the gate queue."""
    symbol: str
    created_at: datetime
    decision: Any
    future: asyncio.Future[bool] = field(default_factory=asyncio.Future)

class GlobalTradeGate:
    """
    Global gate to serialize trade executions across all symbols.

    - Uses a semaphore (size 1) to ensure only one trade executes at a time
    - Maintains a simple FIFO queue for optional queuing behavior
    - Provides a hook for cross-asset risk validation before granting access
    """

    def __init__(self, enable_queue: bool = True) -> None:
        """Initialize the global trade gate."""
        tprint_info(f"Initializing GlobalTradeGate (queue_enabled={enable_queue})")
        self._sem: asyncio.Semaphore = asyncio.Semaphore(1)
        self._queue: Optional[asyncio.Queue[PendingTrade]] = asyncio.Queue() if enable_queue else None
        self._active_trade_id: Optional[str] = None
        self._owner_symbol: Optional[str] = None
        self._last_acquired_at: Optional[datetime] = None
        self._enable_queue: bool = enable_queue
        self._lock: asyncio.Lock = asyncio.Lock()
        self._risk_validator: Optional[Callable[[str, Any], Union[bool, Awaitable[bool]]]] = None  # callable: (symbol, decision) -> bool
        tprint_success("GlobalTradeGate initialized")

    async def try_acquire(self, symbol: str, decision: Any) -> bool:
        """Attempt to acquire the gate for a trade execution."""
        tprint_info(f"Attempting to acquire gate for {symbol} (non-blocking)")
        if not await self._passes_risk_checks(symbol, decision):
            tprint_warning(f"Risk checks failed for {symbol} - gate acquisition denied")
            return False

        # Non-blocking path: if permit available, take it; else, optionally enqueue
        if self._sem.locked():
            if not self._enable_queue:
                tprint_info(f"Gate locked for {symbol}, queue disabled - acquisition denied")
                return False
            tprint_info(f"Gate locked for {symbol}, enqueueing trade")
            await self._enqueue(symbol, decision)
            return False

        acquired: bool = await self._sem.acquire()
        if acquired:
            async with self._lock:
                self._owner_symbol = symbol
                self._last_acquired_at = datetime.utcnow()
            tprint_success(f"Gate acquired for {symbol}")
            return True
        tprint_warning(f"Gate acquisition failed for {symbol}")
        return False

    async def acquire(self, symbol: str, decision: Any, timeout: Optional[float] = None) -> bool:
        """
        Acquire the gate, optionally waiting in a FIFO queue until available.
        Returns True when the caller owns the gate, False if risk check fails or timeout occurs.
        """
        tprint_info(f"Acquiring gate for {symbol} (timeout={timeout})")
        if not await self._passes_risk_checks(symbol, decision):
            tprint_warning(f"Risk checks failed for {symbol} - gate acquisition denied")
            return False

        # Fast path
        if not self._sem.locked():
            await self._sem.acquire()
            async with self._lock:
                self._owner_symbol = symbol
                self._last_acquired_at = datetime.utcnow()
            tprint_success(f"Gate acquired immediately for {symbol}")
            return True

        if not self._enable_queue:
            tprint_info(f"Gate locked for {symbol}, queue disabled - acquisition denied")
            return False

        tprint_info(f"Gate locked for {symbol}, queuing trade")
        pending: PendingTrade = PendingTrade(symbol=symbol, created_at=datetime.utcnow(), decision=decision)
        await self._queue.put(pending)
        try:
            await asyncio.wait_for(pending.future, timeout=timeout)
            tprint_success(f"Gate acquired for {symbol} after queue wait")
            return True
        except asyncio.TimeoutError:
            tprint_warning(f"Gate acquisition timeout for {symbol} after {timeout}s")
            # Remove pending if still in queue (best-effort)
            # Note: asyncio.Queue has no direct remove; leave it to be skipped when processed
            return False

    async def release(self, trade_id: Optional[str] = None) -> None:
        """Release the gate and process next queued trade if any (soft handoff)."""
        async with self._lock:
            old_symbol: Optional[str] = self._owner_symbol
            self._active_trade_id = None
            self._owner_symbol = None
            self._last_acquired_at = None

        try:
            self._sem.release()
            if old_symbol:
                tprint_info(f"Gate released by {old_symbol}")
        except ValueError:
            # Release called without acquire; ignore
            tprint_warning("Gate release called without prior acquisition")
            return

        # Hand off to next queued waiter, if any
        if self._enable_queue and self._queue is not None and not self._queue.empty():
            try:
                next_pending: PendingTrade = self._queue.get_nowait()
                tprint_info(f"Processing queued trade for {next_pending.symbol}")
            except asyncio.QueueEmpty:
                return
            # Acquire immediately for the next waiter
            await self._sem.acquire()
            async with self._lock:
                self._owner_symbol = next_pending.symbol
                self._last_acquired_at = datetime.utcnow()
            if not next_pending.future.done():
                next_pending.future.set_result(True)
                tprint_success(f"Gate handed off to queued trade for {next_pending.symbol}")

    def set_active_trade_id(self, trade_id: str) -> None:
        """Set the active trade ID."""
        self._active_trade_id = trade_id
        tprint_info(f"Active trade ID set: {trade_id}")

    async def _enqueue(self, symbol: str, decision: Any) -> None:
        """Enqueue a pending trade."""
        if self._queue is not None:
            pending: PendingTrade = PendingTrade(symbol=symbol, created_at=datetime.utcnow(), decision=decision)
            await self._queue.put(pending)
            tprint_info(f"Trade enqueued for {symbol} (queue size: {self._queue.qsize()})")

    async def _passes_risk_checks(self, symbol: str, decision: Any) -> bool:
        """Cross-asset risk validation hook. Extend with real checks as needed."""
        # Placeholder for portfolio-level constraints (exposure, VaR, correlations, etc.)
        if callable(self._risk_validator):
            try:
                result: Union[bool, Awaitable[bool]] = self._risk_validator(symbol, decision)
                if asyncio.iscoroutine(result):
                    result = await result
                risk_passed: bool = bool(result)
                if not risk_passed:
                    tprint_warning(f"Risk validator rejected trade for {symbol}")
                return risk_passed
            except Exception as e:
                tprint_error(f"Risk validator error for {symbol}: {e}")
                return False
        return True

    def set_risk_validator(self, validator: Optional[Callable[[str, Any], Union[bool, Awaitable[bool]]]]) -> None:
        """Set a custom risk validator callable: (symbol, decision) -> bool|awaitable."""
        self._risk_validator = validator
        if validator:
            tprint_info("Custom risk validator set")
        else:
            tprint_info("Risk validator cleared")

    def state(self) -> Dict[str, Any]:
        """Get the current state of the gate."""
        state_data: Dict[str, Any] = {
            "locked": self._sem.locked(),
            "owner_symbol": self._owner_symbol,
            "active_trade_id": self._active_trade_id,
            "queued": (self._queue.qsize() if self._queue is not None else 0),
            "last_acquired_at": self._last_acquired_at.isoformat() if self._last_acquired_at else None,
        }
        return state_data
