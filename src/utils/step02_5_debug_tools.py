#!/usr/bin/env python3
"""
Step02_5 Debug Tools

Purpose:
- Provide lightweight, opt-in instrumentation to thoroughly debug Step 2.5 (SR Optimization)
- Capture timings, function call hierarchy, inputs/outputs summaries, memory/CPU snapshots
- Save comprehensive JSON debug reports and concise console summaries

Usage patterns:
- From code: create a DebugTracker and instrument an SROptimizationStep instance via instrument_sr_step
- From CLI: use the debug_step02_5.py runner that leverages these tools

Design goals:
- No hard dependency on optional third-party libraries; degrade gracefully
- Safe in both sync and async contexts
- Zero behavior changes to existing step code when not used
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # Graceful degradation if psutil unavailable

try:
    import tracemalloc
    _TRACEMALLOC_AVAILABLE = True
except Exception:
    _TRACEMALLOC_AVAILABLE = False

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - avoid hard dependency
    pd = None  # type: ignore

try:
    from .logger import system_logger, log_dataframe_overview
except Exception:  # pragma: no cover
    import logging
    system_logger = logging.getLogger(__name__)  # type: ignore

    def log_dataframe_overview(logger: Any, df: Any, *, name: str | None = None, sample_rows: int = 3) -> None:  # type: ignore
        try:
            if hasattr(df, 'shape'):
                rows, cols = getattr(df, 'shape', (None, None))
                logger.info(f"DF overview{name and f' ({name})' or ''}: rows={rows} cols={cols}")
        except Exception:
            pass


class DebugConfig:
    """Configuration for Step02_5 debug instrumentation."""

    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_cpu_tracking: bool = True,
        capture_data_overview: bool = True,
        capture_args_preview: bool = True,
        capture_kwargs_preview: bool = True,
        max_string_preview: int = 200,
        max_call_history: int = 2000,
        report_dir: str | os.PathLike = 'src/training/reports/step02_5_debug',
        enable_sections_timing: bool = True,
        enable_function_wrapping: bool = True,
    ) -> None:
        self.enable_memory_tracking = bool(enable_memory_tracking)
        self.enable_cpu_tracking = bool(enable_cpu_tracking)
        self.capture_data_overview = bool(capture_data_overview)
        self.capture_args_preview = bool(capture_args_preview)
        self.capture_kwargs_preview = bool(capture_kwargs_preview)
        self.max_string_preview = int(max_string_preview)
        self.max_call_history = int(max_call_history)
        self.report_dir = Path(report_dir)
        self.enable_sections_timing = bool(enable_sections_timing)
        self.enable_function_wrapping = bool(enable_function_wrapping)

    @classmethod
    def from_env(cls) -> 'DebugConfig':
        """Build configuration from environment variables prefixed with STEP025_."""
        def _get_bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

        def _get_int(name: str, default: int) -> int:
            try:
                value = os.getenv(name)
                return int(value) if value is not None else default
            except Exception:
                return default

        report_dir = os.getenv('STEP025_REPORT_DIR', 'src/training/reports/step02_5_debug')
        return cls(
            enable_memory_tracking=_get_bool('STEP025_MEMORY', True),
            enable_cpu_tracking=_get_bool('STEP025_CPU', True),
            capture_data_overview=_get_bool('STEP025_DATA_OVERVIEW', True),
            capture_args_preview=_get_bool('STEP025_ARGS_PREVIEW', True),
            capture_kwargs_preview=_get_bool('STEP025_KWARGS_PREVIEW', True),
            max_string_preview=_get_int('STEP025_STR_PREVIEW', 200),
            max_call_history=_get_int('STEP025_MAX_CALLS', 2000),
            report_dir=report_dir,
            enable_sections_timing=_get_bool('STEP025_SECTIONS', True),
            enable_function_wrapping=_get_bool('STEP025_WRAP_FUNCS', True),
        )


class DebugTracker:
    """Collects debug events, resource snapshots, and prepares a report."""

    def __init__(self, config: DebugConfig | None = None, logger: Any | None = None) -> None:
        self.config = config or DebugConfig.from_env()
        self.logger = logger or system_logger
        self.session_started_at: float | None = None
        self.session_context: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []
        self.exceptions: List[Dict[str, Any]] = []
        self.memory_samples: List[Dict[str, Any]] = []
        self._tracemalloc_started: bool = False

    # ---- session ----
    def start_session(self, context: Dict[str, Any] | None = None) -> None:
        self.session_started_at = time.time()
        if context:
            self.session_context.update(context)
        self._maybe_start_tracemalloc()
        self._snapshot_resources(label='session_start')

    def end_session(self, label: str = 'session_end') -> None:
        self._snapshot_resources(label=label)
        self._maybe_stop_tracemalloc()

    # ---- sections ----
    @contextmanager
    def section(self, name: str, **extra: Any) -> Iterable[None]:
        if not self.config.enable_sections_timing:
            yield
            return
        start = time.perf_counter()
        section_entry = {
            'name': name,
            'status': 'running',
            'started_at': datetime.now().isoformat(timespec='seconds'),
            'extra': extra or {},
        }
        self.sections.append(section_entry)
        try:
            yield
            duration = time.perf_counter() - start
            section_entry['status'] = 'success'
            section_entry['duration_seconds'] = round(duration, 6)
        except Exception as e:
            duration = time.perf_counter() - start
            section_entry['status'] = 'error'
            section_entry['duration_seconds'] = round(duration, 6)
            self.record_exception(e, context={'section': name, **(extra or {})})
            raise

    # ---- function wrapping ----
    def wrap_function(self, func: Callable[..., Any], name: Optional[str] = None) -> Callable[..., Any]:
        """Wrap a sync or async function to record calls, timing and exceptions."""
        if not self.config.enable_function_wrapping:
            return func

        func_name = name or getattr(func, '__qualname__', getattr(func, '__name__', 'anonymous'))

        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call_id = len(self.events) + 1
            start = time.perf_counter()
            event: Dict[str, Any] = {
                'id': call_id,
                'type': 'function_call',
                'function': func_name,
                'kind': 'async',
                'status': 'running',
                'args_preview': self._preview_args(args),
                'kwargs_preview': self._preview_kwargs(kwargs),
                'start_time': datetime.now().isoformat(timespec='seconds'),
            }
            self.events.append(event)
            try:
                result = await func(*args, **kwargs)
                event['status'] = 'success'
                event['duration_seconds'] = round(time.perf_counter() - start, 6)
                event['result_type'] = type(result).__name__
                event['result_preview'] = self._preview_value(result)
                return result
            except Exception as e:  # pragma: no cover - runtime capture
                event['status'] = 'error'
                event['duration_seconds'] = round(time.perf_counter() - start, 6)
                self.record_exception(e, context={'function': func_name})
                raise

        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            call_id = len(self.events) + 1
            start = time.perf_counter()
            event: Dict[str, Any] = {
                'id': call_id,
                'type': 'function_call',
                'function': func_name,
                'kind': 'sync',
                'status': 'running',
                'args_preview': self._preview_args(args),
                'kwargs_preview': self._preview_kwargs(kwargs),
                'start_time': datetime.now().isoformat(timespec='seconds'),
            }
            self.events.append(event)
            try:
                result = func(*args, **kwargs)
                event['status'] = 'success'
                event['duration_seconds'] = round(time.perf_counter() - start, 6)
                event['result_type'] = type(result).__name__
                event['result_preview'] = self._preview_value(result)
                return result
            except Exception as e:  # pragma: no cover - runtime capture
                event['status'] = 'error'
                event['duration_seconds'] = round(time.perf_counter() - start, 6)
                self.record_exception(e, context={'function': func_name})
                raise

        if asyncio.iscoroutinefunction(func):
            return _async_wrapper
        return _sync_wrapper

    # ---- utilities ----
    def record_exception(self, error: BaseException, context: Optional[Dict[str, Any]] = None) -> None:
        try:
            exc_info = {
                'time': datetime.now().isoformat(timespec='seconds'),
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {},
            }
            self.exceptions.append(exc_info)
            if self.logger is not None:
                try:
                    self.logger.error(f"🔴 Exception in Step02_5: {exc_info['type']}: {exc_info['message']}")
                except Exception:
                    pass
        except Exception:
            pass

    def _preview_args(self, args: Tuple[Any, ...]) -> List[str]:
        if not self.config.capture_args_preview:
            return []
        return [self._preview_value(a) for a in args]

    def _preview_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, str]:
        if not self.config.capture_kwargs_preview:
            return {}
        return {k: self._preview_value(v) for k, v in kwargs.items()}

    def _preview_value(self, value: Any) -> str:
        try:
            if pd is not None and hasattr(pd, 'DataFrame') and isinstance(value, getattr(pd, 'DataFrame')):
                rows, cols = getattr(value, 'shape', (None, None))
                return f'DataFrame[{rows}x{cols}]'
            text = str(value)
            if len(text) > self.config.max_string_preview:
                return text[: self.config.max_string_preview] + '…'
            return text
        except Exception:
            return type(value).__name__

    def _maybe_start_tracemalloc(self) -> None:
        try:
            if self.config.enable_memory_tracking and _TRACEMALLOC_AVAILABLE and not tracemalloc.is_tracing():
                tracemalloc.start()
                self._tracemalloc_started = True
        except Exception:
            self._tracemalloc_started = False

    def _maybe_stop_tracemalloc(self) -> None:
        try:
            if self._tracemalloc_started and _TRACEMALLOC_AVAILABLE and tracemalloc.is_tracing():
                tracemalloc.stop()
        except Exception:
            pass

    def _snapshot_resources(self, label: str) -> None:
        snapshot: Dict[str, Any] = {'label': label, 'time': datetime.now().isoformat(timespec='seconds')}
        if self.config.enable_cpu_tracking and psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                snapshot['rss_mb'] = float(process.memory_info().rss) / 1024.0 / 1024.0
                snapshot['cpu_percent'] = float(process.cpu_percent(interval=0.0))
            except Exception:
                pass
        if self.config.enable_memory_tracking and _TRACEMALLOC_AVAILABLE and tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                snapshot['tracemalloc_current_mb'] = float(current) / 1024.0 / 1024.0
                snapshot['tracemalloc_peak_mb'] = float(peak) / 1024.0 / 1024.0
            except Exception:
                pass
        self.memory_samples.append(snapshot)

    # ---- reporting ----
    def to_report(self) -> Dict[str, Any]:
        started_at = self.session_started_at or time.time()
        return {
            'meta': {
                'generated_at': datetime.now().isoformat(timespec='seconds'),
                'duration_seconds': round(time.time() - started_at, 6),
                'python': sys.version.split()[0],
                'platform': platform.platform(),
                'pid': os.getpid(),
                'psutil_available': psutil is not None,
                'tracemalloc_available': _TRACEMALLOC_AVAILABLE,
            },
            'session_context': self.session_context,
            'events': self.events[-self.config.max_call_history :],
            'sections': self.sections,
            'exceptions': self.exceptions,
            'resource_samples': self.memory_samples,
        }

    def save_report(self, filename_prefix: str = 'step02_5_debug') -> Path:
        try:
            self.config.report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = self.config.report_dir / f'{filename_prefix}_{timestamp}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_report(), f, indent=2, default=str)
            if self.logger is not None:
                try:
                    self.logger.info(f'📝 Step02_5 debug report saved: {json_path}')
                except Exception:
                    pass
            return json_path
        except Exception as e:  # pragma: no cover
            try:
                if self.logger is not None:
                    self.logger.error(f'Failed to save Step02_5 debug report: {e}')
            except Exception:
                pass
            raise


def instrument_sr_step(step_instance: Any, tracker: DebugTracker) -> Any:
    """
    Monkey-patch key methods of SROptimizationStep for debug instrumentation.

    This avoids modifying the original implementation while enabling detailed
    diagnostics for debugging sessions.
    """
    if step_instance is None:
        return None

    def _wrap_if_present(attr_name: str) -> None:
        try:
            if hasattr(step_instance, attr_name):
                original = getattr(step_instance, attr_name)
                if callable(original):
                    wrapped = tracker.wrap_function(original, name=f'{step_instance.__class__.__name__}.{attr_name}')
                    setattr(step_instance, attr_name, wrapped)
        except Exception:
            pass

    # Wrap primary lifecycle and heavy methods if present
    for method_name in (
        'initialize',
        'execute',
        'execute_logic',
        'execute_main_logic',
        '_prepare_features',
        '_calculate_features_locally',
        '_detect_sr_levels',
        '_optimize_parameters',
        '_evaluate_models',
        '_save_optimization_results',
    ):
        _wrap_if_present(method_name)

    # Track data overview from pipeline_state when available
    def _augment_execute(original_execute: Callable[..., Any]) -> Callable[..., Any]:
        async def _wrapped(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if tracker.config.capture_data_overview and pipeline_state is not None:
                    df = pipeline_state.get('dataframe') if isinstance(pipeline_state, dict) else None
                    if df is not None and pd is not None:
                        try:
                            log_dataframe_overview(tracker.logger, df, name='step02_5_input')
                        except Exception:
                            pass
            except Exception:
                pass
            return await original_execute(training_input, pipeline_state)

        return _wrapped

    try:
        if hasattr(step_instance, 'execute'):
            original_execute_fn = getattr(step_instance, 'execute')
            if asyncio.iscoroutinefunction(original_execute_fn):
                setattr(step_instance, 'execute', _augment_execute(original_execute_fn))
    except Exception:
        pass

    return step_instance


def summarize_result_for_console(result: Dict[str, Any]) -> str:
    """Create a concise, single-line summary for console output."""
    try:
        success = bool(result.get('success', False))
        sr = result.get('sr_levels', {}) or {}
        n_support = len(sr.get('support_levels', []) or [])
        n_resistance = len(sr.get('resistance_levels', []) or [])
        exec_time = result.get('execution_time') or result.get('duration') or 0.0
        return (
            f"success={success} sr_levels={{support:{n_support},resistance:{n_resistance}}} "
            f"time={exec_time:.2f}s"
        )
    except Exception:
        return 'unavailable'


__all__ = [
    'DebugConfig',
    'DebugTracker',
    'instrument_sr_step',
    'summarize_result_for_console',
]

