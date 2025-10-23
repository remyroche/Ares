from __future__ import annotations

from src.utils.tprint import tprint

"""
Standardized trial/result logging and summaries for HPO and CV runs.

Enhanced with comprehensive type hints, error handling, and extensive logging.
"""

import logging
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.LoggingUtils")
    tprint("✅ Custom logger available for MLCommon.LoggingUtils")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.LoggingUtils")
    _LOGGER.setLevel(logging.INFO)

@dataclass
class TrialLog:
    """Container for a single trial's complete information."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: str
    end_time: str
    duration_s: float
    notes: Optional[str] = None
    trial_id: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert TrialLog to dictionary."""
        try:
            return asdict(self)
        except Exception as e:
            _LOGGER.error(f"❌ Failed to convert TrialLog to dict: {e}")
            return {
                'params': self.params,
                'metrics': self.metrics,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration_s': self.duration_s,
                'notes': self.notes,
                'trial_id': self.trial_id,
                'error_message': self.error_message
            }

def log_trial(trial: TrialLog, log_level: str = 'info') -> None:
    """Log a trial with comprehensive error handling.

    Args:
        trial: TrialLog object to log
        log_level: Logging level ('info', 'debug', 'warning', 'error')
    """
    try:
        if not isinstance(trial, TrialLog):
            _LOGGER.error(f"❌ Invalid trial type: {type(trial)}")
            return

        # Format metrics for logging
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in trial.metrics.items()])

        # Create log message
        log_msg = (
            f"HPO Trial | ID={trial.trial_id or 'unknown'} | "
            f"params={trial.params} | metrics={metrics_str} | "
            f"duration={trial.duration_s:.3f}s"
        )

        if trial.notes:
            log_msg += f" | notes={trial.notes}"

        if trial.error_message:
            log_msg += f" | ERROR={trial.error_message}"

        # Log at appropriate level
        if log_level.lower() == 'debug':
            _LOGGER.debug(log_msg)
        elif log_level.lower() == 'warning':
            _LOGGER.warning(log_msg)
        elif log_level.lower() == 'error':
            _LOGGER.error(log_msg)
        else:
            _LOGGER.info(log_msg)

    except Exception as e:
        _LOGGER.error(f"❌ Failed to log trial: {e}")
        _LOGGER.error(f"Trial data: {trial}")

def summarize_trials(trials: List[TrialLog], key_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Summarize multiple trials with comprehensive statistics.

    Args:
        trials: List of TrialLog objects
        key_metrics: List of metrics to summarize (if None, uses all available)

    Returns:
        Dictionary with comprehensive trial summary
    """
    try:
        if not trials:
            _LOGGER.warning("⚠️ No trials provided for summary")
            return {'n_trials': 0, 'error': 'No trials provided'}

        if not isinstance(trials, list):
            _LOGGER.error(f"❌ Invalid trials type: {type(trials)}")
            return {'error': f'Invalid trials type: {type(trials)}'}

        # Validate trials
        valid_trials = [t for t in trials if isinstance(t, TrialLog)]
        if len(valid_trials) != len(trials):
            _LOGGER.warning(f"⚠️ {len(trials) - len(valid_trials)} invalid trials filtered out")

        if not valid_trials:
            _LOGGER.error("❌ No valid trials found")
            return {'error': 'No valid trials found'}

        # Determine key metrics
        if key_metrics is None:
            # Collect all unique metrics from all trials
            all_metrics = set()
            for trial in valid_trials:
                if hasattr(trial, 'metrics') and isinstance(trial.metrics, dict):
                    all_metrics.update(trial.metrics.keys())
            key_metrics = list(all_metrics)

        if not key_metrics:
            _LOGGER.warning("⚠️ No metrics found in trials")
            return {'n_trials': len(valid_trials), 'metrics': {}}

        summary: Dict[str, Any] = {
            'n_trials': len(valid_trials),
            'metrics': {},
            'duration_stats': {},
            'error_count': 0
        }

        # Calculate metric statistics
        for metric in key_metrics:
            try:
                vals = []
                for trial in valid_trials:
                    if hasattr(trial, 'metrics') and isinstance(trial.metrics, dict):
                        val = trial.metrics.get(metric)
                        if val is not None and isinstance(val, (int, float)):
                            vals.append(float(val))

                if vals:
                    import numpy as np
                    arr = np.array(vals, dtype=float)
                    summary['metrics'][metric] = {
                        'mean': float(np.nanmean(arr)),
                        'std': float(np.nanstd(arr)),
                        'min': float(np.nanmin(arr)),
                        'max': float(np.nanmax(arr)),
                        'median': float(np.nanmedian(arr)),
                        'count': len(vals)
                    }
                else:
                    _LOGGER.warning(f"⚠️ No valid values found for metric: {metric}")

            except Exception as e:
                _LOGGER.error(f"❌ Failed to calculate stats for metric {metric}: {e}")
                summary['metrics'][metric] = {'error': str(e)}

        # Calculate duration statistics
        try:
            durations = [trial.duration_s for trial in valid_trials if hasattr(trial, 'duration_s')]
            if durations:
                arr = np.array(durations, dtype=float)
                summary['duration_stats'] = {
                    'mean': float(np.nanmean(arr)),
                    'std': float(np.nanstd(arr)),
                    'min': float(np.nanmin(arr)),
                    'max': float(np.nanmax(arr)),
                    'total': float(np.nansum(arr))
                }
        except Exception as e:
            _LOGGER.error(f"❌ Failed to calculate duration stats: {e}")
            summary['duration_stats'] = {'error': str(e)}

        # Count errors
        error_count = sum(1 for trial in valid_trials if hasattr(trial, 'error_message') and trial.error_message)
        summary['error_count'] = error_count

        _LOGGER.info(f"✅ Summarized {len(valid_trials)} trials with {len(key_metrics)} metrics")
        return summary

    except Exception as e:
        _LOGGER.error(f"❌ Trial summarization failed: {e}")
        _LOGGER.error(f"Traceback: {traceback.format_exc()}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

def start_trial_log(params: Dict[str, Any], trial_id: Optional[str] = None) -> Dict[str, Any]:
    """Start logging a trial with comprehensive error handling.

    Args:
        params: Dictionary of trial parameters
        trial_id: Optional trial identifier

    Returns:
        Dictionary with trial state information
    """
    try:
        if not isinstance(params, dict):
            _LOGGER.error(f"❌ Invalid params type: {type(params)}")
            params = {}

        state = {
            'params': params,
            'start': datetime.now(),
            'trial_id': trial_id or f"trial_{datetime.now().timestamp()}"
        }

        _LOGGER.debug(f"🔄 Started trial {state['trial_id']} with params: {params}")
        return state

    except Exception as e:
        _LOGGER.error(f"❌ Failed to start trial log: {e}")
        return {
            'params': {},
            'start': datetime.now(),
            'trial_id': f"error_trial_{datetime.now().timestamp()}",
            'error': str(e)
        }

def end_trial_log(
    state: Dict[str, Any],
    metrics: Dict[str, float],
    notes: Optional[str] = None,
    error_message: Optional[str] = None
) -> TrialLog:
    """End trial logging with comprehensive error handling.

    Args:
        state: Trial state from start_trial_log
        metrics: Dictionary of trial metrics
        notes: Optional notes about the trial
        error_message: Optional error message if trial failed

    Returns:
        TrialLog object with complete trial information
    """
    try:
        if not isinstance(state, dict):
            _LOGGER.error(f"❌ Invalid state type: {type(state)}")
            state = {'start': datetime.now()}

        if not isinstance(metrics, dict):
            _LOGGER.error(f"❌ Invalid metrics type: {type(metrics)}")
            metrics = {}

        end = datetime.now()
        start = state.get('start', end)

        trial_log = TrialLog(
            params=state.get('params', {}),
            metrics=metrics,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            notes=notes,
            trial_id=state.get('trial_id'),
            error_message=error_message
        )

        _LOGGER.debug(f"✅ Completed trial {trial_log.trial_id} in {trial_log.duration_s:.3f}s")
        return trial_log

    except Exception as e:
        _LOGGER.error(f"❌ Failed to end trial log: {e}")
        _LOGGER.error(f"Traceback: {traceback.format_exc()}")

        # Return minimal trial log
        return TrialLog(
            params={},
            metrics={},
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_s=0.0,
            notes=notes,
            trial_id="error_trial",
            error_message=f"Logging error: {e}"
        )

__all__ = [
    'TrialLog',
    'log_trial',
    'summarize_trials',
    'start_trial_log',
    'end_trial_log',
]
