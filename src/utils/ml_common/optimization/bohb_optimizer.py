"""
BOHB-Style (Bayesian Optimization + Hyperband) Optimizer

This module implements a BOHB-like pipeline using Optuna's TPE sampler
combined with Hyperband/ASHA-style pruning for multi-fidelity optimization.

It is designed as a drop-in sibling to the provided staged TPE optimizer,
but replaces the explicit coarse→fine grids with bandit-style resource
allocation. You still get:
  • Hardware acceleration hooks
  • VectorBT-based batching hooks
  • Adaptive/early stopping utilities

Notes
-----
- This implementation uses Optuna's TPESampler + HyperbandPruner (or ASHA) to
  approximate BOHB behavior without requiring HpBandSter.
- The objective is expected to support multi-fidelity. See `objective` contract
  in `optimize()` for details.
"""
from __future__ import annotations

import time
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# --- Optional deps mirroring the staged optimizer's environment ---
try:
    import vectorbt as vbt  # type: ignore
    VECTORBT_AVAILABLE = True
except Exception as e:
    logging.warning(f"VectorBT not available for BOHB optimizer: {e}")
    VECTORBT_AVAILABLE = False
    vbt = None  # type: ignore

try:
    from src.utils.hardware.optimization_decorators import performance_tracked  # type: ignore
except Exception:
    def performance_tracked(*args, **kwargs):  # no-op fallback
        def deco(f):
            return f
        return deco

# Dummy M1 decorator to match your environment
try:
    from ...hardware import UnifiedHardwareManager, HardwareOptimizedMatrixProcessor, BatchMatrixProcessor  # type: ignore
except Exception:
    UnifiedHardwareManager = None
    HardwareOptimizedMatrixProcessor = None
    BatchMatrixProcessor = None

try:
    from ..logger import get_logger  # type: ignore
except Exception:
    def get_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
            handler.setFormatter(fmt)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

@dataclass
class BOHBConfig:
    """Configuration for BOHB-style optimization."""
    # Core budget
    n_trials: int = 100
    timeout: Optional[float] = None

    # Multi-fidelity axis
    resource_name: str = "epoch"  # e.g., 'epoch', 'steps', 'n_estimators', 'data_frac'
    min_resource: int = 1          # r_min in Hyperband
    max_resource: int = 81         # R (full budget)
    reduction_factor: int = 3      # eta

    # Sampler (TPE)
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = True
    group: bool = True
    gamma: Callable[[int], int] = lambda t: min(int(np.ceil(0.15 * t)), 100)
    seed: Optional[int] = None

    # Direction/metric
    direction: str = "maximize"  # 'maximize' or 'minimize'
    metric_name: str = "objective"

    # Pruner / scheduler
    pruner_type: str = "hyperband"  # 'hyperband' | 'asha' | 'median'
    pruner_params: Optional[Dict[str, Any]] = None

    # Early stopping (study-level)
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # Hardware / batching hooks
    enable_hardware_optimization: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 32
    memory_limit_gb: float = 8.0

    # VectorBT hooks
    enable_vectorbt_optimization: bool = True
    vectorbt_chunk_size: int = 512
    vectorbt_enable_parallel: bool = True

    # Caps to avoid giant memory footprints
    max_trial_history: int = 200

    def validate(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.min_resource <= 0 or self.max_resource <= 0:
            raise ValueError("min/max resource must be positive")
        if self.min_resource > self.max_resource:
            raise ValueError("min_resource must be <= max_resource")
        if self.reduction_factor < 2:
            raise ValueError("reduction_factor (eta) must be >= 2")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")

class BOHBOptimizer:
    """
    BOHB-style optimizer: TPE sampler + Hyperband/ASHA pruning across a fidelity axis.

    Objective contract
    ------------------
    The optimizer expects an objective callable. It will:
      1) Ask Optuna for hyperparameters.
      2) Run a rung-based loop: for resource in {r_min, r_min*eta, ..., R}
         - Call your evaluation function with the current resource.
         - Report intermediate metric via `trial.report(metric, resource)`.
         - Let the pruner decide whether to continue.

    To make this work, your `objective` should accept either of the following signatures:
      • objective(params, resource) -> float (metric at that resource)
      • objective(params, **kwargs) where kwargs may include {resource_name: resource}

    The final returned value should be the metric at the *maximum* reached resource
    for that trial (Optuna requires a single scalar return).
    """

    def __init__(self, config: Optional[BOHBConfig] = None, **kwargs):
        self.config = config or BOHBConfig()
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.validate()

        self.logger = get_logger("BOHBOptimizer")
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for BOHBOptimizer. Install with: pip install optuna>=3.0.0")

        # Hardware hooks
        self.hardware_manager = None
        self.batch_processor = None
        self._init_hardware()

        # VectorBT hook (optional)
        self.vectorbt_enabled = VECTORBT_AVAILABLE and self.config.enable_vectorbt_optimization

        # State
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.performance_metrics: List[Dict[str, Any]] = []

        self.logger.info("✅ BOHB-style optimizer initialized")

    # -------------------- Initialization --------------------
    def _init_hardware(self) -> None:
        try:
            if self.config.enable_hardware_optimization and UnifiedHardwareManager:
                self.hardware_manager = UnifiedHardwareManager()
                self.hardware_manager.initialize()
                if self.config.memory_limit_gb:
                    self.hardware_manager.set_memory_limit_gb(self.config.memory_limit_gb)
                if BatchMatrixProcessor:
                    self.batch_processor = BatchMatrixProcessor(
                        chunk_size_mb=int(self.config.memory_limit_gb * 128),
                        enable_gpu=True,
                        enable_parallel=True,
                        max_workers=4,
                    )
                self.logger.info("   → Hardware optimization: Enabled")
            else:
                self.logger.info("   → Hardware optimization: Disabled or unavailable")
        except Exception as e:
            self.logger.warning(f"Hardware init failed: {e}")
            self.hardware_manager = None
            self.batch_processor = None

    # -------------------- Public API --------------------
    @performance_tracked(log_performance=True, track_memory=True)
    def optimize(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        sampler = TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            n_ei_candidates=self.config.n_ei_candidates,
            gamma=self.config.gamma,
            seed=self.config.seed,
            multivariate=self.config.multivariate,
            group=self.config.group,
        )

        pruner = self._make_pruner()

        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"bohb_{int(time.time())}",
        )

        self.logger.info(
            f"🚀 Starting BOHB optimization: trials={self.config.n_trials}, resource=[{self.config.min_resource}..{self.config.max_resource}] η={self.config.reduction_factor}"
        )

        self.study.optimize(
            self._make_optuna_objective(objective, search_space),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        result = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "optimization_time": time.time() - start,
            "history": self._compact_trials(self.study.trials),
            "resource_axis": {
                "name": self.config.resource_name,
                "min": self.config.min_resource,
                "max": self.config.max_resource,
                "eta": self.config.reduction_factor,
            },
        }
        self.logger.info(f"✅ BOHB optimization finished. Best value: {self.best_value:.6f}")
        return result

    # -------------------- Internals --------------------
    def _make_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        p = (self.config.pruner_type or "hyperband").lower()
        params = self.config.pruner_params or {}
        try:
            if p in ("hyperband", "hb"):
                # Align with Optuna's HyperbandPruner args
                return optuna.pruners.HyperbandPruner(
                    min_resource=self.config.min_resource,
                    max_resource=self.config.max_resource,
                    reduction_factor=self.config.reduction_factor,
                    **params,
                )
            if p in ("asha", "successive_halving", "sha"):
                return optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=self.config.min_resource,
                    reduction_factor=self.config.reduction_factor,
                    **params,
                )
            if p == "median":
                return optuna.pruners.MedianPruner(**params)
        except Exception as e:
            self.logger.warning(f"Pruner init failed, disabling pruning: {e}")
        return None

    def _make_optuna_objective(self, user_objective: Callable, search_space: Dict[str, Any]) -> Callable[[optuna.Trial], float]:
        def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
            params: Dict[str, Any] = {}
            for name, cfg in search_space.items():
                if isinstance(cfg, tuple) and len(cfg) == 2:
                    lo, hi = cfg
                    if isinstance(lo, int) and isinstance(hi, int):
                        params[name] = trial.suggest_int(name, lo, hi)
                    else:
                        params[name] = trial.suggest_float(name, lo, hi)
                elif isinstance(cfg, list):
                    params[name] = trial.suggest_categorical(name, cfg)
                elif isinstance(cfg, dict):
                    t = cfg.get("type", "float")
                    if t == "int":
                        params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
                    elif t == "float":
                        params[name] = trial.suggest_float(
                            name, cfg["low"], cfg["high"], log=cfg.get("log", False)
                        )
                    elif t == "categorical":
                        params[name] = trial.suggest_categorical(name, cfg["choices"]) 
                    else:
                        raise ValueError(f"Unknown param type for {name}: {t}")
                else:
                    raise ValueError(f"Unsupported search space entry for {name}: {cfg}")
            return params

        def call_objective(obj: Callable, params: Dict[str, Any], resource: int) -> float:
            # Try (params, resource)
            try:
                return obj(params, resource)
            except TypeError:
                # Try keyword with resource_name
                try:
                    return obj(params, **{self.config.resource_name: resource})
                except TypeError:
                    # Last resort: assume single-fidelity objective (use resource only to report)
                    return obj(params)

        def optuna_objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)

            # Determine rung sequence for this trial (same as Hyperband levels)
            r = self.config.min_resource
            levels: List[int] = [r]
            while r < self.config.max_resource:
                r = int(max(r * self.config.reduction_factor, r + 1))
                if r > self.config.max_resource:
                    r = self.config.max_resource
                if r not in levels:
                    levels.append(r)
                if r == self.config.max_resource:
                    break

            best_seen: Optional[float] = None
            for resource in levels:
                value = call_objective(user_objective, params, resource)
                # Report intermediate metric at current resource (step = resource)
                trial.report(value, step=resource)

                # Keep best seen (for return)
                if best_seen is None:
                    best_seen = value
                else:
                    if self.config.direction == "maximize":
                        best_seen = max(best_seen, value)
                    else:
                        best_seen = min(best_seen, value)

                # Ask pruner whether to stop this trial early
                if trial.should_prune():
                    raise optuna.TrialPruned()

            assert best_seen is not None
            return float(best_seen)

        return optuna_objective

    # -------------------- Utilities --------------------
    def _compact_trials(self, trials: List[optuna.trial.FrozenTrial]) -> List[Dict[str, Any]]:
        # Keep memory under control; store a compact view
        if self.config.max_trial_history and len(trials) > self.config.max_trial_history:
            trials = trials[-self.config.max_trial_history:]
        hist: List[Dict[str, Any]] = []
        for t in trials:
            hist.append(
                {
                    "trial": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": str(t.state),
                    "duration": t.duration.total_seconds() if t.duration else None,
                    # Optional: last reported step/value if available
                    "last_step": t.last_step,
                }
            )
        return hist

    # Convenience accessors
    def get_parameter_importance(self) -> Dict[str, float]:
        if not self.study:
            return {}
        try:
            imp = optuna.importance.get_param_importances(self.study)
            return dict(imp)
        except Exception as e:
            self.logger.warning(f"Parameter importance failed: {e}")
            return {}

    def save_study(self, filepath: str) -> None:
        if not self.study:
            raise ValueError("No optimization has been run yet")
        try:
            import joblib
            joblib.dump(self.study, filepath)
            self.logger.info(f"💾 Study saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save study: {e}")
            raise

    def load_study(self, filepath: str) -> None:
        try:
            import joblib
            self.study = joblib.load(filepath)
            if self.study and self.study.trials:
                self.best_params = self.study.best_params
                self.best_value = self.study.best_value
            self.logger.info(f"📂 Study loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load study: {e}")
            raise