"""
Standardized trial/result logging and summaries for HPO and CV runs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from datetime import datetime

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.LoggingUtils")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.LoggingUtils")


@dataclass
class TrialLog:
    params: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: str
    end_time: str
    duration_s: float
    notes: Optional[str] = None


def log_trial(trial: TrialLog) -> None:
    try:
        _LOGGER.info(f"HPO Trial | params={trial.params} | metrics={trial.metrics} | duration={trial.duration_s:.3f}s")
    except Exception:
        pass


def summarize_trials(trials: List[TrialLog], key_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    if not trials:
        return {'n_trials': 0}
    if key_metrics is None:
        # pick first trial's metrics keys
        key_metrics = list(trials[0].metrics.keys())
    summary: Dict[str, Any] = {'n_trials': len(trials), 'metrics': {}}
    for m in key_metrics:
        vals = [t.metrics.get(m) for t in trials if m in t.metrics]
        if vals:
            import numpy as np
            arr = np.array(vals, dtype=float)
            summary['metrics'][m] = {
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
            }
    return summary


def start_trial_log(params: Dict[str, Any]) -> Dict[str, Any]:
    return {'params': params, 'start': datetime.now()}


def end_trial_log(state: Dict[str, Any], metrics: Dict[str, float], notes: Optional[str] = None) -> TrialLog:
    end = datetime.now()
    start = state.get('start', end)
    return TrialLog(
        params=state.get('params', {}),
        metrics=metrics,
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        duration_s=(end - start).total_seconds(),
        notes=notes,
    )


__all__ = [
    'TrialLog',
    'log_trial',
    'summarize_trials',
    'start_trial_log',
    'end_trial_log',
]

