import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class PendingTrade:
    symbol: str
    created_at: datetime
    decision: Any
    future: asyncio.Future = field(default_factory=asyncio.Future)


class GlobalTradeGate:
    """
    Global gate to serialize trade executions across all symbols.

    - Uses a semaphore (size 1) to ensure only one trade executes at a time
    - Maintains a simple FIFO queue for optional queuing behavior
    - Provides a hook for cross-asset risk validation before granting access
    """

    def __init__(self, enable_queue: bool = True) -> None:
        self._sem = asyncio.Semaphore(1)
        self._queue: asyncio.Queue[PendingTrade] = asyncio.Queue() if enable_queue else None
        self._active_trade_id: Optional[str] = None
        self._owner_symbol: Optional[str] = None
        self._last_acquired_at: Optional[datetime] = None
        self._enable_queue = enable_queue
        self._lock = asyncio.Lock()
        self._risk_validator = None  # callable: (symbol, decision) -> bool

    async def try_acquire(self, symbol: str, decision: Any) -> bool:
        """Attempt to acquire the gate for a trade execution."""
        if not await self._passes_risk_checks(symbol, decision):
            return False

        # Non-blocking path: if permit available, take it; else, optionally enqueue
        if self._sem.locked():
            if not self._enable_queue:
                return False
            await self._enqueue(symbol, decision)
            return False

        acquired = await self._sem.acquire()
        if acquired:
            async with self._lock:
                self._owner_symbol = symbol
                self._last_acquired_at = datetime.utcnow()
            return True
        return False

    async def acquire(self, symbol: str, decision: Any, timeout: Optional[float] = None) -> bool:
        """
        Acquire the gate, optionally waiting in a FIFO queue until available.
        Returns True when the caller owns the gate, False if risk check fails or timeout occurs.
        """
        if not await self._passes_risk_checks(symbol, decision):
            return False

        # Fast path
        if not self._sem.locked():
            await self._sem.acquire()
            async with self._lock:
                self._owner_symbol = symbol
                self._last_acquired_at = datetime.utcnow()
            return True

        if not self._enable_queue:
            return False

        pending = PendingTrade(symbol=symbol, created_at=datetime.utcnow(), decision=decision)
        await self._queue.put(pending)
        try:
            await asyncio.wait_for(pending.future, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            # Remove pending if still in queue (best-effort)
            # Note: asyncio.Queue has no direct remove; leave it to be skipped when processed
            return False

    async def release(self, trade_id: Optional[str] = None) -> None:
        """Release the gate and process next queued trade if any (soft handoff)."""
        async with self._lock:
            self._active_trade_id = None
            self._owner_symbol = None
            self._last_acquired_at = None

        try:
            self._sem.release()
        except ValueError:
            # Release called without acquire; ignore
            return

        # Hand off to next queued waiter, if any
        if self._enable_queue and self._queue is not None and not self._queue.empty():
            try:
                next_pending: PendingTrade = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            # Acquire immediately for the next waiter
            await self._sem.acquire()
            async with self._lock:
                self._owner_symbol = next_pending.symbol
                self._last_acquired_at = datetime.utcnow()
            if not next_pending.future.done():
                next_pending.future.set_result(True)

    def set_active_trade_id(self, trade_id: str) -> None:
        self._active_trade_id = trade_id

    async def _enqueue(self, symbol: str, decision: Any) -> None:
        if self._queue is not None:
            await self._queue.put(PendingTrade(symbol=symbol, created_at=datetime.utcnow(), decision=decision))

    async def _passes_risk_checks(self, symbol: str, decision: Any) -> bool:
        """Cross-asset risk validation hook. Extend with real checks as needed."""
        # Placeholder for portfolio-level constraints (exposure, VaR, correlations, etc.)
        if callable(self._risk_validator):
            try:
                result = self._risk_validator(symbol, decision)
                if asyncio.iscoroutine(result):
                    result = await result
                return bool(result)
            except Exception:
                return False
        return True

    def set_risk_validator(self, validator) -> None:
        """Set a custom risk validator callable: (symbol, decision) -> bool|awaitable."""
        self._risk_validator = validator

    def state(self) -> Dict[str, Any]:
        return {
            "locked": self._sem.locked(),
            "owner_symbol": self._owner_symbol,
            "active_trade_id": self._active_trade_id,
            "queued": (self._queue.qsize() if self._queue is not None else 0),
            "last_acquired_at": self._last_acquired_at,
        }

