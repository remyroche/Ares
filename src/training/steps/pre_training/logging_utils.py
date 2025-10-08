"""Structured logging utilities for the pre-training pipeline.

This module configures a JSON-formatted logging pipeline that is both machine
parsable and readable from the console.  It exposes helper dataclasses and
emission utilities that attach common metadata required for observability of the
pre-training workflow.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

# Import tprint for enhanced logging capabilities
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success


class _JSONFormatter(logging.Formatter):
    """JSON formatter that keeps console output readable and machine friendly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
        }

        message = record.getMessage()
        if message:
            payload["message"] = message

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)

        return json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ": "),
        )


_CONFIGURED = False
_LOGGER_NAME = "ares.pre_training"


def configure_pre_training_logging() -> logging.Logger:
    """Configure and return the structured pre-training logger."""
    global _CONFIGURED

    tprint_debug(f"🔧 Configuring pre-training logging system (logger: {_LOGGER_NAME})")

    logger = logging.getLogger(_LOGGER_NAME)

    if not _CONFIGURED:
        tprint_debug("📋 Setting up JSON formatter and handlers for structured logging")
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        handler.setLevel(logging.INFO)
        handler.addFilter(_DuplicateFilter())
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True

        tprint_info(f"✅ Pre-training logging system configured successfully")

    return logger


class _DuplicateFilter(logging.Filter):
    """Filter to avoid duplicate console lines when multiple handlers exist."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - defensive
        return True


@dataclass
class StepLogContext:
    """Context shared across begin/end events for a single step."""

    run_id: str
    step: str
    symbol: str
    timeframe: str
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreTrainingEventLogger:
    """Emit structured begin/end events for the pre-training sub-pipeline."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or configure_pre_training_logging()

    def pipeline_begin(self, *, run_id: str, symbol: str, timeframe: str, mode: str, metadata: Dict[str, Any]) -> None:
        tprint(f"🚀 Pre-training pipeline started: {symbol}_{timeframe} (mode: {mode})")
        tprint_debug(f"📋 Run ID: {run_id}")

        self._emit(
            event="pipeline_begin",
            message="Pre-training pipeline started",
            payload={
                "run_id": run_id,
                "step": "pipeline",
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": mode,
                "rows_in": None,
                "rows_out": None,
                "duration_ms": None,
                "metadata": metadata,
            },
        )

    def pipeline_end(
        self,
        *,
        run_id: str,
        symbol: str,
        timeframe: str,
        mode: str,
        success: bool,
        duration_ms: Optional[float],
        completed_steps: int,
        total_steps: int,
        metadata: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        # Enhanced pipeline end logging
        if success:
            tprint_success(f"🎉 Pre-training pipeline completed successfully: {symbol}_{timeframe}")
            tprint(f"📊 Steps: {completed_steps}/{total_steps} completed")
            if duration_ms:
                tprint(f"⏱️ Duration: {duration_ms/1000:.2f} seconds")
        else:
            tprint_error(f"❌ Pre-training pipeline failed: {symbol}_{timeframe}")
            tprint(f"📊 Steps: {completed_steps}/{total_steps} completed")
            if duration_ms:
                tprint_warning(f"⏱️ Duration before failure: {duration_ms/1000:.2f} seconds")
            if error:
                tprint_error(f"💥 Error: {error}")

        payload = {
            "run_id": run_id,
            "step": "pipeline",
            "symbol": symbol,
            "timeframe": timeframe,
            "mode": mode,
            "rows_in": None,
            "rows_out": None,
            "duration_ms": duration_ms,
            "success": success,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "metadata": metadata,
        }
        if error:
            payload["error"] = error
        self._emit(
            event="pipeline_end",
            message="Pre-training pipeline finished",
            payload=payload,
            level=logging.INFO if success else logging.ERROR,
        )

    def step_begin(self, context: StepLogContext) -> None:
        payload = asdict(context)
        payload.update({"duration_ms": None, "phase": "begin"})
        self._emit(
            event="step_begin",
            message=f"Begin step: {context.step}",
            payload=payload,
        )

    def step_end(
        self,
        context: StepLogContext,
        *,
        duration_ms: Optional[float],
        success: bool,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = asdict(context)
        payload.update(
            {
                "duration_ms": duration_ms,
                "success": success,
                "phase": "end",
            }
        )
        if error:
            payload["error"] = error
        if extra:
            payload.update(extra)
        self._emit(
            event="step_end",
            message=f"End step: {context.step}",
            payload=payload,
            level=logging.INFO if success else logging.ERROR,
        )

    def info(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(context or {})
        self._emit("info", message=message, payload=payload)

    def warning(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(context or {})
        self._emit("warning", message=message, payload=payload, level=logging.WARNING)

    def error(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(context or {})
        self._emit("error", message=message, payload=payload, level=logging.ERROR)

    def _emit(
        self,
        event: str,
        *,
        message: str,
        payload: Dict[str, Any],
        level: int = logging.INFO,
    ) -> None:
        body = dict(payload)
        body.setdefault("event", event)
        body.setdefault("run_id", None)
        body.setdefault("step", None)
        body.setdefault("symbol", None)
        body.setdefault("timeframe", None)
        body.setdefault("rows_in", None)
        body.setdefault("rows_out", None)
        body.setdefault("duration_ms", None)
        self._logger.log(level, message, extra={"extra_fields": body})
