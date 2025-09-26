"""Utility for tracking optimisation runs without silent failures."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("OptimizationMonitor")


@dataclass
class OptimizationPerformance:
    optimization_id: str
    model_type: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    objective_score: float = 0.0
    validation_score: float = 0.0
    status: str = "running"  # running/completed/failed
    error_message: Optional[str] = None

    def finish(self, status: str, objective: float, validation: float, error: Optional[str] = None) -> None:
        self.end_time = datetime.utcnow()
        self.status = status
        self.objective_score = float(objective)
        self.validation_score = float(validation)
        self.error_message = error

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.utcnow()
        return float((end - self.start_time).total_seconds())


class OptimizationMonitor:
    """Track optimisation progress in a background thread."""

    def __init__(self, interval_seconds: float = 60.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._optimizations: Dict[str, OptimizationPerformance] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._loop,
            name="OptimizationMonitor",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        logger.debug("Optimization monitor thread started")

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=self.interval_seconds)
            if thread.is_alive():
                logger.warning("Optimization monitor thread did not stop within timeout")
        self._thread = None
        logger.debug("Optimization monitor thread stopped")

    def register(self, optimization_id: str, model_type: str) -> None:
        with self._lock:
            self._optimizations[optimization_id] = OptimizationPerformance(
                optimization_id=optimization_id, model_type=model_type
            )

    def complete(self, optimization_id: str, objective: float, validation: float) -> None:
        with self._lock:
            perf = self._ensure_registered(optimization_id)
            perf.finish("completed", objective, validation)

    def fail(self, optimization_id: str, error_message: str) -> None:
        with self._lock:
            perf = self._ensure_registered(optimization_id)
            perf.finish("failed", perf.objective_score, perf.validation_score, error_message)

    def summary(self) -> Dict[str, float]:
        with self._lock:
            completed = [p for p in self._optimizations.values() if p.status == "completed"]
        if not completed:
            return {"completed_runs": 0}
        objective = np.mean([p.objective_score for p in completed])
        validation = np.mean([p.validation_score for p in completed])
        duration = np.mean([p.duration_seconds for p in completed])
        return {
            "completed_runs": len(completed),
            "avg_objective": float(objective),
            "avg_validation": float(validation),
            "avg_duration": float(duration),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _loop(self) -> None:  # pragma: no cover - relies on timing
        while not self._stop_event.wait(self.interval_seconds):
            try:
                snapshot = self.summary()
                logger.debug("Optimization monitor heartbeat: %s", snapshot)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to produce optimisation summary")
                self._stop_event.set()

    def _ensure_registered(self, optimization_id: str) -> OptimizationPerformance:
        try:
            return self._optimizations[optimization_id]
        except KeyError as exc:  # pragma: no cover - guard clause
            raise KeyError(f"Unknown optimisation id '{optimization_id}'") from exc

    def __enter__(self) -> "OptimizationMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = ["OptimizationMonitor", "OptimizationPerformance"]
