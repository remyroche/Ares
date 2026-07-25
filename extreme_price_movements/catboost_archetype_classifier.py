"""Leakage-safe future-path archetypes and a pre-entry CatBoost classifier.

Future path measurements in this module are *training targets only*.  Discovery
is fitted on an authorised train period and converted to frozen cluster rules;
the only supported live assignment route is :class:`PathArchetypeClassifier`,
which accepts explicitly validated pre-entry features.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .path_archetype_labels import ATR_REALIZATION_THRESHOLDS
from .path_archetype_support import MERGED_PATH_ARCHETYPE_CLASSES

PATH_HORIZONS_HOURS: tuple[int, ...] = (1, 2, 4, 8, 12, 24)
PATH_SUMMARY_PREFIX = "path_arch_"
_GIB = 1024**3
_CATBOOST_RAM_PER_THREAD_BYTES = 6 * _GIB
_FORBIDDEN_PREENTRY_TOKENS = (
    "mfe",
    "mae",
    "future",
    "forward",
    "outcome",
    "time_to_",
    "reversal",
    "final_peak",
    "target",
    "label",
    "realized",
    "realised",
    "exit_",
)
_FORBIDDEN_PREENTRY_PREFIXES = (
    "path_arch_",
    "path_archetype",
    "path_shape_archetype",
    "path_realization_strength",
    "discovery_cluster_id",
)
_BASE_META_UNIVERSE_GROUPS = (
    "base_shared_feature_keys",
    "base_long_feature_keys",
    "base_short_feature_keys",
    "meta_shared_feature_keys",
)
_BASE_PERFORMANCE_GROUPS = (
    "META_BASE_PERFORMANCE_FEATURE_KEYS",
    "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS",
)
STAGED_PERMUTATION_ACCELERATION_VERSION = "cached_float32_batched_screened_v2"
PATH_ARCHETYPE_ADVERSE_CLASSES: tuple[str, ...] = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "dead_timeout",
)
PATH_ARCHETYPE_FAVORABLE_CLASSES: tuple[str, ...] = (
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
)
PATH_ARCHETYPE_NEUTRAL_CLASSES: tuple[str, ...] = ("noisy_timeout_usable_mfe",)
CATBOOST_CLASS_BALANCE_ARM_UNIFORM = "uniform"
CATBOOST_CLASS_BALANCE_ARM_EXPONENTS: tuple[float, ...] = (0.25, 0.50, 0.75)
CATBOOST_CLASS_BALANCE_MAX_WEIGHT_RATIO = 4.0
CATBOOST_CLASS_BALANCE_MIN_CLASS_SUPPORT = 2
CATBOOST_CLASS_BALANCE_MIN_PREDICTED_SHARE = 0.002
CATBOOST_CLASS_BALANCE_MIN_NORMALIZED_ENTROPY = 0.05
CATBOOST_CLASS_BALANCE_SEARCH_SCHEMA = "catboost_path_archetype_class_balance_search_v2"
CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA = (
    "catboost_path_archetype_class_balance_arm_selection_v1"
)
CATBOOST_CLASS_BALANCE_MINI_SWEEP_SCHEMA = (
    "catboost_path_archetype_class_balance_mini_sweep_v1"
)
CATBOOST_STRUCTURAL_HPO_SCHEMA = "catboost_path_archetype_structural_hpo_v1"
_CLASS_BALANCE_SELECTION_LINK_FIELDS: tuple[str, ...] = (
    "selection_status",
    "promotion_reason",
    "economic_oof_schema",
    "economic_oof_report_sha256",
    "economic_selector_config_sha256",
    "economic_oof_config_sha256",
    "economic_config_sha256",
    "mini_sweep_report_sha256",
    "class_balance_mini_sweep_report_sha256",
    "class_balance_mini_sweep_sha256",
    "structural_fingerprint",
    "feature_fingerprint",
    "geometry_fingerprint",
)
_CLASS_BALANCE_SELECTION_BULKY_REPORT_FIELDS: frozenset[str] = frozenset(
    {
        "per_arm",
        "folds",
        "months",
        "train_only_priors",
        "probabilities",
        "raw_oof",
        "economic_oof_report",
        "economic_report",
        "mini_sweep_report",
    }
)
LOGGER = logging.getLogger(__name__)


class CatBoostUnavailableError(ImportError):
    """Raised only when a CatBoost-dependent operation is requested."""


class CatBoostClassBalanceError(ValueError):
    """Raised when a predeclared balance arm has unsafe support or OOF output."""


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except Exception:
        return False
    return True


def _require_catboost() -> Any:
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover - depends on the environment.
        raise CatBoostUnavailableError(
            "CatBoost is required to fit or score the path-archetype classifier. "
            "Install `catboost`, or use discovery/fast feature screening only."
        ) from exc
    return CatBoostClassifier


@dataclass(frozen=True)
class PathArchetypeConfig:
    timestamp_col: str = "__ts__"
    label_end_col: str = "__label_end_ts__"
    availability_threshold: float = 0.95
    max_feature_candidates: int = 500
    corr_threshold: float = 0.90
    random_state: int = 20260722
    embargo: pd.Timedelta = pd.Timedelta(hours=24)
    oof_folds: int = 5
    catboost_thread_count: int = 4
    catboost_os_reserve_gib: float = 4.0
    unsafe_allow_catboost_threads: bool = False
    selector_sample_rows: int = 2_500
    relief_sample_rows: int = 2_500
    selector_parallel_jobs: int = 4
    relief_neighbors: int = 10
    permutation_stages: tuple[int, ...] = (150, 125, 100, 75)
    # Stage-one MDA can reuse the pre-permutation OOF models when both calls
    # share this cache.  The cache deliberately stores only float32 numeric
    # inputs and fold-local medians, never target-derived state.
    permutation_batch_max_bytes: int = 256 * 1024**2
    permutation_validation_cache_max_bytes: int = 512 * 1024**2
    permutation_screening_enabled: bool = True
    permutation_screen_margin: int = 32
    # The runner supplies the seven-class future-training contract.  Keeping
    # this optional preserves the generic low-level helpers used in diagnostics.
    class_order: tuple[str, ...] | None = None
    # Balance is deliberately a compact, predeclared HPO dimension rather than
    # an open class-weight surface.  These limits apply to every arm, including
    # the final refit payload reconstructed from a selected OOF trial.
    class_balance_max_weight_ratio: float = CATBOOST_CLASS_BALANCE_MAX_WEIGHT_RATIO
    class_balance_min_class_support: int = CATBOOST_CLASS_BALANCE_MIN_CLASS_SUPPORT
    class_balance_min_predicted_share: float = (
        CATBOOST_CLASS_BALANCE_MIN_PREDICTED_SHARE
    )
    class_balance_min_normalized_entropy: float = (
        CATBOOST_CLASS_BALANCE_MIN_NORMALIZED_ENTROPY
    )
    legacy_allow_class_weights: bool = False


def staged_permutation_acceleration_contract(
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, Any]:
    """Versioned execution contract for cached staged permutation MDA."""
    return {
        "algorithm_version": STAGED_PERMUTATION_ACCELERATION_VERSION,
        "numeric_cache_dtype": "float32",
        "fold_imputation": (
            "per-fold train/validation median with non-finite fallback to zero"
        ),
        "batch_max_bytes": int(config.permutation_batch_max_bytes),
        "validation_cache_max_bytes": int(
            config.permutation_validation_cache_max_bytes
        ),
        "screening_enabled": bool(config.permutation_screening_enabled),
        "screen_margin": int(config.permutation_screen_margin),
        "screening_contract": (
            "deterministic first-and-last-fold permutation; retain by maximum "
            "endpoint loss, then run full MDA for mandatory features and the "
            "stage keep count plus the borderline buffer"
        ),
    }


def _physical_ram_bytes() -> int:
    """Return installed physical memory without depending on one OS command."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and physical_pages > 0:
            return page_size * physical_pages
    except (AttributeError, OSError, ValueError):
        pass
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows.
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    raise RuntimeError("Unable to determine physical RAM for CatBoost resource limits")


def catboost_resource_contract(
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, Any]:
    """Derive a conservative host-memory cap for this multiclass workload."""
    physical_ram_bytes = _physical_ram_bytes()
    requested_threads = max(1, int(config.catboost_thread_count))
    requested_reserve_bytes = max(0, int(float(config.catboost_os_reserve_gib) * _GIB))
    # Keep at least three quarters of small hosts usable for the process while
    # reserving the configured four GiB on ordinary research machines.
    os_reserve_bytes = min(requested_reserve_bytes, physical_ram_bytes // 4)
    used_ram_limit_bytes = max(_GIB, physical_ram_bytes - os_reserve_bytes)
    safe_thread_cap = max(1, used_ram_limit_bytes // _CATBOOST_RAM_PER_THREAD_BYTES)
    unsafe_override = bool(config.unsafe_allow_catboost_threads)
    effective_threads = (
        requested_threads
        if unsafe_override
        else min(requested_threads, safe_thread_cap)
    )
    requested_selector_parallel_jobs = max(1, int(config.selector_parallel_jobs))
    effective_selector_parallel_jobs = min(
        requested_selector_parallel_jobs, int(effective_threads)
    )
    used_ram_limit_mib = max(1, used_ram_limit_bytes // (1024**2))
    return {
        "policy": "physical_ram_minus_os_reserve_with_6_gib_per_safe_catboost_thread",
        "requested_thread_count": requested_threads,
        "effective_thread_count": int(effective_threads),
        "physical_ram_bytes": int(physical_ram_bytes),
        "os_reserve_bytes": int(os_reserve_bytes),
        "used_ram_limit_bytes": int(used_ram_limit_bytes),
        "used_ram_limit": f"{used_ram_limit_mib}MB",
        "unsafe_allow_catboost_threads": unsafe_override,
        "requested_selector_parallel_jobs": requested_selector_parallel_jobs,
        "effective_selector_parallel_jobs": effective_selector_parallel_jobs,
    }


def _catboost_params_with_resource_limits(
    params: Mapping[str, Any] | None,
    *,
    config: PathArchetypeConfig = PathArchetypeConfig(),
    used_ram_limit_bytes: int | None = None,
) -> dict[str, Any]:
    """Apply the non-bypassable thread and RAM limits to CatBoost parameters."""
    resource = catboost_resource_contract(config)
    limit_bytes = int(used_ram_limit_bytes or resource["used_ram_limit_bytes"])
    limited = dict(params or {})
    limited["thread_count"] = int(resource["effective_thread_count"])
    limited["used_ram_limit"] = f"{max(1, limit_bytes // (1024**2))}MB"
    return limited


def capped_catboost_params(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cap geometry-refit params without reopening arbitrary class weighting."""
    requested_threads = int(
        params.get("thread_count", PathArchetypeConfig().catboost_thread_count)
    )
    config = PathArchetypeConfig(catboost_thread_count=max(1, requested_threads))
    validated = _catboost_params(config, params)
    # Geometry callers supply their own explicit CatBoost defaults.  Retain
    # that historical surface while using the stricter validator above to
    # remove internal balance metadata and reject undeclared raw weights.
    limited = dict(params)
    for key in (
        "class_balance_arm",
        "class_balance_final_weights",
        "class_balance_provenance",
        "class_balance_selection_provenance",
    ):
        limited.pop(key, None)
    if "class_weights" in validated:
        limited["class_weights"] = validated["class_weights"]
    return (
        _catboost_params_with_resource_limits(limited, config=config),
        catboost_resource_contract(config),
    )


def _first_time(values: np.ndarray, predicate: np.ndarray, bar_hours: float) -> float:
    hit = np.flatnonzero(predicate)
    # Path element zero is the first completed future bar after entry, so its
    # realised passage time is one bar rather than zero elapsed hours.
    return float((hit[0] + 1) * bar_hours) if hit.size else float("nan")


def summarize_future_path(
    future_r: Iterable[float],
    *,
    bar_hours: float = 1.0,
    take_profit_r: float = 1.0,
    trailing_trigger_r: float | None = None,
    stop_r: float = 1.0,
    meaningful_mfe_r: float | None = None,
    atr_r: float = 1.0,
    cost_r: float = 0.03,
    activation_r: float = 1.0,
    horizons_hours: Sequence[int] = PATH_HORIZONS_HOURS,
    prefix: str = PATH_SUMMARY_PREFIX,
) -> dict[str, float]:
    """Summarise a post-entry path expressed in favourable-direction R units.

    The first element is the first observation *after* entry.  A positive path
    is favourable for both sides; callers must direction-normalise short paths
    before calling this function.
    """
    path = np.asarray(list(future_r), dtype=float)
    path = path[np.isfinite(path)]
    out: dict[str, float] = {}
    if path.size == 0 or bar_hours <= 0:
        for horizon in horizons_hours:
            out[f"{prefix}mfe_{horizon}h_r"] = float("nan")
            out[f"{prefix}mae_{horizon}h_r"] = float("nan")
        for name in (
            "time_to_025r_h",
            "time_to_05r_h",
            "time_to_1r_h",
            "time_to_tp_h",
            "time_to_trailing_h",
            "time_to_stop_h",
            "mfe_before_mae",
            "mae_before_mfe",
            "time_to_first_meaningful_mfe_h",
            "time_to_90pct_peak_mfe_h",
            "peak_mfe_r",
            "peak_mfe_atr",
            "mfe_to_cost",
            "mfe_to_activation_distance",
            "early_late_ratio",
            "efficiency",
            "reversal_count",
            "final_return_r",
            "final_to_peak",
        ):
            out[f"{prefix}{name}"] = float("nan")
        for threshold in ATR_REALIZATION_THRESHOLDS:
            token = f"{int(round(threshold * 100)):03d}atr"
            out[f"{prefix}reached_{token}"] = float("nan")
            out[f"{prefix}time_to_{token}_h"] = float("nan")
        return out

    for horizon in horizons_hours:
        stop = min(path.size, max(1, int(math.floor(horizon / bar_hours))))
        window = path[:stop]
        out[f"{prefix}mfe_{horizon}h_r"] = float(np.max(window))
        out[f"{prefix}mae_{horizon}h_r"] = float(np.min(window))

    meaningful_mfe_r = max(
        1.5 * float(atr_r),
        float(meaningful_mfe_r) if meaningful_mfe_r is not None else 0.0,
    )
    peak_i = int(np.argmax(path))
    trough_i = int(np.argmin(path))
    trailing_trigger_r = (
        take_profit_r if trailing_trigger_r is None else trailing_trigger_r
    )
    out.update(
        {
            f"{prefix}time_to_025r_h": _first_time(path, path >= 0.25, bar_hours),
            f"{prefix}time_to_05r_h": _first_time(path, path >= 0.50, bar_hours),
            f"{prefix}time_to_1r_h": _first_time(path, path >= 1.0, bar_hours),
            f"{prefix}time_to_tp_h": _first_time(
                path, path >= take_profit_r, bar_hours
            ),
            f"{prefix}time_to_trailing_h": _first_time(
                path, path >= trailing_trigger_r, bar_hours
            ),
            f"{prefix}time_to_stop_h": _first_time(
                path, path <= -abs(stop_r), bar_hours
            ),
            f"{prefix}mfe_before_mae": float(peak_i < trough_i),
            f"{prefix}mae_before_mfe": float(trough_i < peak_i),
            f"{prefix}time_to_first_meaningful_mfe_h": _first_time(
                path, path >= meaningful_mfe_r, bar_hours
            ),
            f"{prefix}time_to_90pct_peak_mfe_h": _first_time(
                path, path >= 0.90 * float(path[peak_i]), bar_hours
            ),
            f"{prefix}peak_mfe_r": float(path[peak_i]),
            f"{prefix}peak_mfe_atr": float(
                np.clip(path[peak_i] / max(float(atr_r), 1e-9), 0.0, 10.0)
            ),
            f"{prefix}mfe_to_cost": float(
                max(path[peak_i], 0.0) / max(float(cost_r), 1e-9)
            ),
            f"{prefix}mfe_to_activation_distance": float(
                max(path[peak_i], 0.0) / max(float(activation_r), 1e-9)
            ),
        }
    )
    for threshold in ATR_REALIZATION_THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}atr"
        hits = path >= threshold * float(atr_r)
        out[f"{prefix}reached_{token}"] = float(bool(hits.any()))
        out[f"{prefix}time_to_{token}_h"] = _first_time(path, hits, bar_hours)
    split = max(1, path.size // 2)
    early_peak = float(np.max(path[:split]))
    late_peak = float(np.max(path[split:])) if split < path.size else early_peak
    out[f"{prefix}early_late_ratio"] = early_peak / max(abs(late_peak), 1e-6)
    variation = float(np.abs(np.diff(np.r_[0.0, path])).sum())
    out[f"{prefix}efficiency"] = float(path[-1] / variation) if variation else 0.0
    moves = np.sign(np.diff(path))
    moves = moves[moves != 0]
    out[f"{prefix}reversal_count"] = float(np.sum(moves[1:] != moves[:-1]))
    out[f"{prefix}final_return_r"] = float(path[-1])
    out[f"{prefix}final_to_peak"] = float(path[-1] / max(abs(path[peak_i]), 1e-6))
    return out


def summarize_future_paths(
    frame: pd.DataFrame,
    path_column: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Vectorised convenience wrapper; it deliberately never mutates ``frame``."""
    return pd.DataFrame(
        [summarize_future_path(path, **kwargs) for path in frame[path_column]],
        index=frame.index,
    )


def path_summary_columns(prefix: str = PATH_SUMMARY_PREFIX) -> tuple[str, ...]:
    sample = summarize_future_path([0.0], prefix=prefix)
    return tuple(sample)


def validate_preentry_features(columns: Iterable[str]) -> tuple[str, ...]:
    """Reject target/path/outcome fields before any supervised classifier fit."""
    accepted = tuple(columns)
    violations = [
        column
        for column in accepted
        if any(token in str(column).lower() for token in _FORBIDDEN_PREENTRY_TOKENS)
        or str(column).lower().startswith(_FORBIDDEN_PREENTRY_PREFIXES)
    ]
    if violations:
        raise ValueError(
            "Path archetype classifier received non-pre-entry features: "
            + ", ".join(violations[:8])
        )
    return accepted


def configured_base_meta_preselection_universe(
    available_columns: Iterable[str],
    *,
    config_mapping: Mapping[str, Any] | None = None,
    frozen_representation_features: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return causal configured inputs plus an explicit frozen representation.

    The model must not discover its own input universe from a broad training
    frame: ``config.py`` is the authority for the base and meta feature sets.
    Nested config feature groups are expanded recursively and only columns
    present in ``available_columns`` are returned.
    """
    if config_mapping is None:
        from .config import CFG

        config_mapping = CFG
    from .config import (
        MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS,
    )
    from .features_gmm_ae import AE_GMM_FEATURE_COLUMNS
    from .training_utils import get_base_feature_keys, get_meta_feature_keys

    available = tuple(str(column) for column in available_columns)
    frozen = tuple(dict.fromkeys(map(str, frozen_representation_features)))
    expected_frozen = tuple(map(str, AE_GMM_FEATURE_COLUMNS))
    if frozen and frozen != expected_frozen:
        raise ValueError(
            "frozen representation must be exactly AE_GMM_FEATURE_COLUMNS in canonical order"
        )

    def expand(key: str, seen: set[str]) -> list[str]:
        if key in seen:
            return []
        value = config_mapping.get(key)
        if not isinstance(value, (list, tuple, set)):
            return [key]
        next_seen = {*seen, key}
        result: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                continue
            result.extend(expand(item, next_seen) if item in config_mapping else [item])
        return result

    resolved_config = dict(config_mapping)
    declared = set(get_base_feature_keys("long", resolved_config))
    declared.update(get_base_feature_keys("short", resolved_config))
    for head in ("reg", "clf", "mfe", "mae", "asym"):
        declared.update(get_meta_feature_keys(head, resolved_config))
    excluded: set[str] = set()
    for group in _BASE_PERFORMANCE_GROUPS:
        excluded.update(expand(group, set()))
    # This exported set is intentionally not a CFG node, so expanding its name
    # through config_mapping would otherwise retain base/meta OOF performance
    # fields. Permit only the separately supplied frozen AE/GMM representation.
    excluded.update(map(str, MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS))
    configured_model_derived = config_mapping.get(
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS", ()
    )
    if isinstance(configured_model_derived, (list, tuple, set, frozenset)):
        excluded.update(map(str, configured_model_derived))
    allowed = set(frozen)
    # Outcome-derived fields are forbidden independently of config membership.
    configured = [
        column
        for column in available
        if column in declared
        and (column not in excluded or column in allowed)
        and not any(token in column.lower() for token in _FORBIDDEN_PREENTRY_TOKENS)
    ]
    configured.extend(column for column in frozen if column in available)
    return tuple(dict.fromkeys(configured))


def _rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values).rank(method="average", na_option="keep").to_numpy(dtype=float)
    )


def _finite_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    """Materialize numeric CatBoost inputs with deterministic median imputation."""
    values = (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=True)
    )
    with np.errstate(all="ignore"):
        medians = np.nanmedian(values, axis=0)
    medians = np.asarray(medians, dtype=np.float32)
    medians[~np.isfinite(medians)] = 0.0
    np.copyto(values, medians[None, :], where=~np.isfinite(values))
    return values


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    if len(x) < 8 or len(np.unique(y)) < 2 or np.nanstd(x) == 0:
        return 0.0
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    xb = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    table = pd.crosstab(xb, y).to_numpy(dtype=float)
    joint = table / table.sum()
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(joint > 0, joint * np.log(joint / (px * py)), 0.0)
    return float(np.nansum(term))


def _relief_multisurf_proxy(
    values: np.ndarray, y: np.ndarray, neighbors: int
) -> np.ndarray:
    """Small deterministic nearest-neighbour Relief/MultiSURF-style rescue."""
    n, p = values.shape
    if n < 4 or len(np.unique(y)) < 2:
        return np.zeros(p)
    scaled = (values - np.nanmedian(values, axis=0)) / np.maximum(
        np.nanstd(values, axis=0), 1e-6
    )
    scaled = np.nan_to_num(scaled)
    # Compute squared distances without materialising an n x n x p cube.
    # The selector sample is deliberately bounded, but the previous broadcast
    # still became needlessly large for a 500-column preselection universe.
    norm = np.einsum("ij,ij->i", scaled, scaled)
    distance = np.maximum(norm[:, None] + norm[None, :] - 2.0 * scaled @ scaled.T, 0.0)
    np.fill_diagonal(distance, np.inf)
    scores = np.zeros(p)
    k = min(max(1, neighbors), n - 1)
    for row in range(n):
        near = np.argsort(distance[row])[: max(k * 3, k)]
        hit = near[y[near] == y[row]][:k]
        miss = near[y[near] != y[row]][:k]
        if hit.size and miss.size:
            scores += np.mean(np.abs(scaled[row] - scaled[miss]), axis=0)
            scores -= np.mean(np.abs(scaled[row] - scaled[hit]), axis=0)
    return scores / n


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    finite = np.array(
        [value for value in values.values() if np.isfinite(value)], dtype=float
    )
    if not finite.size or np.ptp(finite) <= 1e-12:
        return {key: 0.0 for key in values}
    low, span = float(finite.min()), float(np.ptp(finite))
    return {
        key: (float(value) - low) / span if np.isfinite(value) else 0.0
        for key, value in values.items()
    }


def _univariate_catboost_oos_logloss_gain(
    sample: np.ndarray,
    target: np.ndarray,
    columns: Sequence[str],
    *,
    random_state: int,
    parallel_jobs: int = 1,
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, float]:
    """Score each feature on two expanding chronological holdouts."""

    CatBoostClassifier = _require_catboost()
    resource = catboost_resource_contract(config)
    effective_parallel_jobs = min(
        max(1, int(parallel_jobs)), int(resource["effective_thread_count"])
    )
    per_fit_ram_limit = max(
        _GIB, int(resource["used_ram_limit_bytes"]) // effective_parallel_jobs
    )
    n = len(target)
    boundaries = (int(0.50 * n), int(0.75 * n), n)
    classes = np.arange(int(np.max(target)) + 1, dtype=int)
    fold_specs: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for fold, (train_end, valid_end) in enumerate(
        ((boundaries[0], boundaries[1]), (boundaries[1], boundaries[2]))
    ):
        train = np.arange(train_end, dtype=int)
        valid = np.arange(train_end, valid_end, dtype=int)
        if len(train) < 20 or len(valid) < 8 or len(np.unique(target[train])) < 2:
            continue
        counts = np.bincount(target[train], minlength=len(classes)).astype(float) + 1.0
        prior = counts / counts.sum()
        baseline = multiclass_log_loss(
            target[valid],
            np.broadcast_to(prior, (len(valid), len(prior))),
            classes,
        )
        fold_specs.append((fold, train, valid, baseline))

    def score_column(item: tuple[int, str]) -> tuple[str, float]:
        column_index, column = item
        column_values = sample[:, column_index]
        finite_values = column_values[np.isfinite(column_values)]
        if not finite_values.size or np.ptp(finite_values) <= 1e-12:
            return str(column), 0.0
        gains: list[float] = []
        for fold, train, valid, baseline in fold_specs:
            train_values = sample[train, column_index]
            finite_train = train_values[np.isfinite(train_values)]
            if not finite_train.size or np.ptp(finite_train) <= 1e-12:
                # A feature can vary only late in history and still be constant
                # in an early expanding fold. It has no trainable univariate
                # evidence in that fold and must not abort the selector.
                continue
            model = CatBoostClassifier(
                **_catboost_params_with_resource_limits(
                    {
                        "loss_function": "MultiClass",
                        "iterations": 40,
                        "depth": 5,
                        "learning_rate": 0.05,
                        "l2_leaf_reg": 20.0,
                        "random_seed": int(random_state + fold),
                        "verbose": False,
                        "allow_writing_files": False,
                        "bootstrap_type": "Bayesian",
                        "grow_policy": "SymmetricTree",
                    },
                    config=PathArchetypeConfig(
                        catboost_thread_count=1,
                        catboost_os_reserve_gib=config.catboost_os_reserve_gib,
                    ),
                    used_ram_limit_bytes=per_fit_ram_limit,
                )
            )
            model.fit(sample[train][:, [column_index]], target[train])
            local = model.predict_proba(sample[valid][:, [column_index]])
            aligned = np.full((len(valid), len(classes)), 1e-12, dtype=float)
            aligned[:, np.asarray(model.classes_, dtype=int)] = local
            gains.append(
                baseline - multiclass_log_loss(target[valid], aligned, classes)
            )
        return str(column), float(np.mean(gains)) if gains else 0.0

    items = list(enumerate(columns))
    progress_every = max(25, min(100, max(1, len(items) // 10)))

    def collect(scored: Iterable[tuple[str, float]]) -> dict[str, float]:
        results: dict[str, float] = {}
        started = time.perf_counter()
        for position, (column, gain) in enumerate(scored, start=1):
            results[column] = gain
            if position % progress_every == 0 or position == len(items):
                LOGGER.info(
                    "CatBoost univariate screen: %s/%s features (%.1f%%) in %.1fs",
                    position,
                    len(items),
                    100.0 * position / max(len(items), 1),
                    time.perf_counter() - started,
                )
        return results

    if effective_parallel_jobs <= 1:
        return collect(map(score_column, items))
    with ThreadPoolExecutor(max_workers=effective_parallel_jobs) as executor:
        return collect(executor.map(score_column, items))


@dataclass(frozen=True)
class FastSelectorResult:
    selected_features: tuple[str, ...]
    candidate_features: tuple[str, ...]
    mandatory_features: tuple[str, ...]
    availability: Mapping[str, float]
    scores: pd.DataFrame
    correlation_clusters: tuple[tuple[str, ...], ...]
    proxy_backend: str


def fast_select_preentry_features(
    features: pd.DataFrame,
    target: Sequence[Any],
    *,
    mandatory_features: Sequence[str] = (),
    warmup_mask: Sequence[bool] | None = None,
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> FastSelectorResult:
    """MI + univariate CatBoost + Relief union, then Spearman pruning.

    Availability is measured after caller-supplied warmup rows are excluded.
    The scored union is capped at ``max_feature_candidates`` *in addition to*
    mandatory features. A binned fallback remains available for minimal test
    environments, but production preselection uses independent CatBoost fits.
    """
    columns = validate_preentry_features(features.columns)
    mandatory = validate_preentry_features(mandatory_features)
    missing = set(mandatory).difference(columns)
    if missing:
        raise KeyError(f"Mandatory features missing from frame: {sorted(missing)}")
    y = _categorical_target(target, features.index, config=config).cat.codes.to_numpy()
    eligible_rows = (
        np.ones(len(features), dtype=bool)
        if warmup_mask is None
        else ~np.asarray(warmup_mask, dtype=bool)
    )
    if eligible_rows.sum() < 8:
        raise ValueError("Need at least eight non-warmup rows for feature selection")
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(features[c])]
    availability = {
        c: float(features.loc[eligible_rows, c].notna().mean()) for c in numeric
    }
    available = [c for c in numeric if availability[c] > config.availability_threshold]
    for col in mandatory:
        if availability.get(col, 0.0) <= config.availability_threshold:
            raise ValueError(
                f"Mandatory feature {col!r} fails strict >{config.availability_threshold:.0%} availability"
            )
    if not available:
        raise ValueError("No numeric pre-entry features meet availability threshold")
    idx = np.flatnonzero(eligible_rows)
    if len(idx) > config.selector_sample_rows:
        idx = np.linspace(0, len(idx) - 1, config.selector_sample_rows, dtype=int)
        idx = np.flatnonzero(eligible_rows)[idx]
    # Every eligible configured input participates in all three independent
    # ranking methods before the union cap is applied.
    raw_sample = _finite_matrix(features.iloc[idx], available)
    sample_y = y[idx]
    raw_mi = {
        col: _mutual_information(raw_sample[:, j], sample_y)
        for j, col in enumerate(available)
    }
    screened = list(dict.fromkeys([*mandatory, *available]))
    # Mandatory columns have already passed the same availability criterion,
    # so ``screened`` is only a deterministic reordering/subset of ``available``.
    # Reuse the finite float matrix instead of converting the full BME sample a
    # second time before the expensive univariate CatBoost and Relief passes.
    if tuple(screened) == tuple(available):
        sample = raw_sample
    else:
        available_index = {
            column: position for position, column in enumerate(available)
        }
        screened_indices = np.fromiter(
            (available_index[column] for column in screened), dtype=np.intp
        )
        sample = np.take(raw_sample, screened_indices, axis=1)
    mi = {col: raw_mi[col] for col in screened}
    proxy = dict(mi)
    backend = "binned_multiclass_proxy"
    if catboost_available() and len(np.unique(sample_y)) > 1:
        backend = "catboost_univariate_chronological_oos_logloss_gain"
        proxy = _univariate_catboost_oos_logloss_gain(
            sample,
            sample_y,
            screened,
            random_state=config.random_state,
            parallel_jobs=config.selector_parallel_jobs,
            config=config,
        )
    relief_count = min(len(sample), max(8, int(config.relief_sample_rows)))
    relief_idx = np.linspace(0, len(sample) - 1, relief_count, dtype=int)
    relief_values = _relief_multisurf_proxy(
        sample[relief_idx], sample_y[relief_idx], config.relief_neighbors
    )
    relief = {col: float(relief_values[j]) for j, col in enumerate(screened)}
    cap = max(int(config.max_feature_candidates), 1)
    per_method = 200
    ranked_union = set(mandatory)
    for metric in (mi, proxy, relief):
        ranked_union.update(sorted(metric, key=metric.get, reverse=True)[:per_method])
    combined = {
        col: sum(_normalise(metric).get(col, 0.0) for metric in (mi, proxy, relief))
        for col in ranked_union
    }
    candidates = list(dict.fromkeys(mandatory)) + [
        col
        for col in sorted(
            ranked_union.difference(mandatory), key=combined.get, reverse=True
        )
    ]
    candidates = candidates[: len(mandatory) + cap]
    selected, clusters = prune_spearman_clusters(
        features.iloc[idx], candidates, combined, mandatory, config.corr_threshold
    )
    score_frame = pd.DataFrame(
        {
            "mi": pd.Series(mi),
            "catboost_proxy": pd.Series(proxy),
            "relief": pd.Series(relief),
        }
    )
    score_frame["combined"] = pd.Series(combined)
    return FastSelectorResult(
        tuple(selected),
        tuple(candidates),
        tuple(mandatory),
        availability,
        score_frame,
        tuple(clusters),
        backend,
    )


def prune_spearman_clusters(
    features: pd.DataFrame,
    candidates: Sequence[str],
    scores: Mapping[str, float],
    mandatory_features: Sequence[str] = (),
    threshold: float = 0.90,
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Keep one best feature per abs-Spearman cluster, preserving mandatory fields."""
    candidates = list(dict.fromkeys(candidates))
    mandatory = set(mandatory_features)
    if not candidates:
        return [], []
    values = _finite_matrix(features, candidates)
    ranks = np.column_stack([_rank(values[:, i]) for i in range(values.shape[1])])
    corr = np.corrcoef(ranks, rowvar=False)
    parent = list(range(len(candidates)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for i in range(len(candidates)):
        for j in range(i):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= threshold:
                union(i, j)
    groups: dict[int, list[str]] = {}
    for index, col in enumerate(candidates):
        groups.setdefault(find(index), []).append(col)
    selected: list[str] = []
    clusters: list[tuple[str, ...]] = []
    for group in groups.values():
        clusters.append(tuple(group))
        forced = [col for col in group if col in mandatory]
        selected.extend(
            forced or [max(group, key=lambda col: scores.get(col, float("-inf")))]
        )
    return selected, clusters


@dataclass(frozen=True)
class PurgedFold:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    fold_id: int


def purged_chronological_folds(
    timestamps: Sequence[Any],
    *,
    label_end: Sequence[Any] | None = None,
    n_splits: int = 5,
    embargo: pd.Timedelta = pd.Timedelta(hours=24),
) -> list[PurgedFold]:
    """Expanding chronological folds, purged for open labels and embargoed."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    if not ts.is_monotonic_increasing:
        raise ValueError(
            "timestamps must already be chronological; sorting silently breaks row alignment"
        )
    end = (
        ts
        if label_end is None
        else pd.to_datetime(pd.Series(label_end), utc=True, errors="raise")
    )
    if len(ts) < n_splits + 2:
        raise ValueError("Too few rows for requested chronological OOF folds")
    chunks = np.array_split(np.arange(len(ts)), n_splits + 1)
    folds: list[PurgedFold] = []
    for fold_id, valid in enumerate(chunks[1:]):
        if not len(valid):
            continue
        start = ts.iloc[int(valid[0])]
        prior = np.arange(int(valid[0]))
        keep = (end.iloc[prior] < start).to_numpy() & (
            ts.iloc[prior] < start - embargo
        ).to_numpy()
        train = prior[keep]
        if len(train):
            folds.append(PurgedFold(train, valid, fold_id))
    if not folds:
        raise ValueError("Purge/embargo leaves no train rows")
    return folds


def default_catboost_hpo_space() -> dict[str, Any]:
    """Exact, fixed-contract CatBoost HPO space for this classifier."""
    return {
        "depth": (5, 7),
        "iterations": 3000,
        "od_wait": 150,
        "learning_rate": (0.015, 0.06),
        "l2_leaf_reg": (8.0, 80.0),
        "random_strength": (0.1, 3.0),
        "bagging_temperature": (0.2, 2.5),
        "rsm": (0.65, 1.0),
        "border_count": (64, 128),
        "auto_class_weights": None,
        "bootstrap_type": "Bayesian",
        "grow_policy": "SymmetricTree",
        "class_balance_arms": tuple(
            arm["name"] for arm in predeclared_catboost_class_balance_arms()
        ),
    }


def multiclass_log_loss(
    y: Sequence[int], probabilities: np.ndarray, classes: Sequence[int] | None = None
) -> float:
    y = np.asarray(y, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    class_order = (
        np.arange(proba.shape[1]) if classes is None else np.asarray(classes, dtype=int)
    )
    if proba.ndim != 2 or proba.shape[0] != len(y):
        raise ValueError("probabilities must have one row per target")
    if proba.shape[1] != len(class_order):
        raise ValueError("probabilities and classes have incompatible widths")
    # Preserve the historical dict lookup semantics, including a missing label
    # defaulting to column zero, while selecting all rows in compiled NumPy.
    # MDA invokes this once per feature/fold, so the former Python row loop was
    # material on large validation blocks.
    reversed_labels = class_order[::-1]
    labels, reverse_positions = np.unique(reversed_labels, return_index=True)
    last_positions = len(class_order) - 1 - reverse_positions
    insertion = np.searchsorted(labels, y)
    selected_positions = np.zeros(len(y), dtype=np.intp)
    matched = insertion < len(labels)
    matched[matched] = labels[insertion[matched]] == y[matched]
    selected_positions[matched] = last_positions[insertion[matched]]
    chosen = proba[np.arange(len(y)), selected_positions]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))


def catboost_hpo_objective_components(
    y: Sequence[int], oof_probabilities: np.ndarray, fold_ids: Sequence[int]
) -> dict[str, float]:
    """Return the independently auditable components of the minimised objective."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(oof_probabilities, dtype=float)
    folds = np.asarray(fold_ids)
    loss = multiclass_log_loss(y, p)
    fold_losses = [
        multiclass_log_loss(y[folds == f], p[folds == f])
        for f in np.unique(folds)
        if np.any(folds == f)
    ]
    stability_penalty = float(np.std(fold_losses)) if fold_losses else 0.0
    classes = np.arange(p.shape[1], dtype=int)
    one_hot = np.eye(len(classes), dtype=float)[y]
    macro_brier = float(np.mean(np.mean((p - one_hot) ** 2, axis=0)))
    classwise_ece = float(
        np.mean([_binary_ece(one_hot[:, index], p[:, index]) for index in classes])
    )
    return {
        "mean_logloss": loss,
        "macro_brier": macro_brier,
        "classwise_ece": classwise_ece,
        "fold_logloss_std": stability_penalty,
        "objective": (
            loss + 0.25 * macro_brier + 0.15 * classwise_ece + 0.20 * stability_penalty
        ),
    }


def catboost_hpo_objective(
    y: Sequence[int], oof_probabilities: np.ndarray, fold_ids: Sequence[int]
) -> float:
    """Minimise OOF log loss plus calibration and fold-instability penalties."""
    return catboost_hpo_objective_components(y, oof_probabilities, fold_ids)[
        "objective"
    ]


def _json_ready(value: Any) -> Any:
    """Convert numpy/pandas scalars used in reports to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON progress artifact without partial readers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        _json_ready(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _binary_ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    if not len(target):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for value in range(bins):
        mask = bucket == value
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return float(ece)


def multiclass_classification_diagnostics(
    y: Sequence[int],
    probabilities: np.ndarray,
    *,
    fold_ids: Sequence[int] | None = None,
    class_names: Sequence[Any] | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute persistable OOF classification, calibration, and stability metrics."""
    labels = np.asarray(y, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2 or len(labels) != len(proba):
        raise ValueError("labels and multiclass probabilities must have aligned rows")
    valid = np.isfinite(proba).all(axis=1)
    labels, proba = labels[valid], proba[valid]
    if not len(labels):
        raise ValueError("No finite OOF probabilities are available for diagnostics")
    classes = np.arange(proba.shape[1], dtype=int)
    names = tuple(map(str, class_names if class_names is not None else classes))
    if len(names) != len(classes):
        raise ValueError("class_names must align with probability columns")
    predicted = np.argmax(proba, axis=1)
    support = np.bincount(labels, minlength=len(classes)).astype(int)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    precision = np.divide(
        np.diag(confusion),
        confusion.sum(axis=0),
        out=np.zeros(len(classes)),
        where=confusion.sum(axis=0) > 0,
    )
    recall = np.divide(
        np.diag(confusion), support, out=np.zeros(len(classes)), where=support > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=(precision + recall) > 0,
    )
    one_hot = np.eye(len(classes), dtype=float)[labels]
    brier_by_class = np.mean((proba - one_hot) ** 2, axis=0)
    classwise = {
        names[index]: {
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "brier": float(brier_by_class[index]),
            "ece": _binary_ece(one_hot[:, index], proba[:, index], ece_bins),
        }
        for index in range(len(classes))
    }
    weights = support / max(int(support.sum()), 1)
    report: dict[str, Any] = {
        "rows": int(len(labels)),
        "logloss": multiclass_log_loss(labels, proba, classes),
        "brier_macro": float(np.mean(brier_by_class)),
        "brier_weighted": float(np.sum(weights * brier_by_class)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": float(np.sum(weights * f1)),
        "recall_macro": float(np.mean(recall)),
        "recall_weighted": float(np.sum(weights * recall)),
        "confusion_matrix": confusion.tolist(),
        "class_names": list(names),
        "classwise": classwise,
    }
    if fold_ids is not None:
        all_folds = np.asarray(fold_ids)[valid]
        fold_rows: list[dict[str, Any]] = []
        fold_means: list[np.ndarray] = []
        for fold_id in sorted(
            int(value) for value in np.unique(all_folds) if int(value) >= 0
        ):
            mask = all_folds == fold_id
            if not np.any(mask):
                continue
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "rows": int(mask.sum()),
                    "logloss": multiclass_log_loss(labels[mask], proba[mask], classes),
                    "f1_macro": float("nan"),
                }
            )
            # Per-fold F1 is intentionally derived from that fold's rows rather
            # than reusing aggregate confusion statistics.
            fold_pred = predicted[mask]
            fold_truth = labels[mask]
            fold_conf = np.zeros_like(confusion)
            np.add.at(fold_conf, (fold_truth, fold_pred), 1)
            fold_support = fold_conf.sum(axis=1)
            fold_precision = np.divide(
                np.diag(fold_conf),
                fold_conf.sum(axis=0),
                out=np.zeros(len(classes)),
                where=fold_conf.sum(axis=0) > 0,
            )
            fold_recall = np.divide(
                np.diag(fold_conf),
                fold_support,
                out=np.zeros(len(classes)),
                where=fold_support > 0,
            )
            fold_f1 = np.divide(
                2.0 * fold_precision * fold_recall,
                fold_precision + fold_recall,
                out=np.zeros(len(classes)),
                where=(fold_precision + fold_recall) > 0,
            )
            fold_rows[-1]["f1_macro"] = float(np.mean(fold_f1))
            fold_means.append(proba[mask].mean(axis=0))
        losses = [float(row["logloss"]) for row in fold_rows]
        report["temporal_stability"] = {
            "folds": fold_rows,
            "logloss_mean": float(np.mean(losses)) if losses else float("nan"),
            "logloss_std": float(np.std(losses)) if losses else float("nan"),
            "logloss_worst": float(np.max(losses)) if losses else float("nan"),
            "probability_drift_first_to_last": float(
                np.mean(np.abs(fold_means[-1] - fold_means[0]))
            )
            if len(fold_means) > 1
            else 0.0,
        }
    return _json_ready(report)


def class_balance_oof_guard(
    target_codes: Sequence[int],
    probabilities: np.ndarray,
    *,
    classes: Sequence[Any] | None,
    fold_ids: Sequence[int] | None = None,
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, Any]:
    """Reject degenerate class-balance candidates using frozen OOF rows only.

    The guard intentionally uses neither a final model nor in-sample scores.
    ``predicted_probability_share`` is the mean probability mass assigned to a
    class, rather than its argmax frequency: rare but calibrated classes need
    not win an argmax to remain usable.
    """
    target = np.asarray(target_codes, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2 or len(proba) != len(target) or not len(target):
        raise CatBoostClassBalanceError(
            "class-balance OOF guard requires aligned non-empty target and "
            "probabilities"
        )
    if (
        not np.isfinite(proba).all()
        or (proba < 0.0).any()
        or not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise CatBoostClassBalanceError(
            "class-balance OOF guard rejected invalid probability output"
        )
    if (target < 0).any() or np.max(target) >= proba.shape[1]:
        raise CatBoostClassBalanceError(
            "class-balance OOF guard found target codes outside probability columns"
        )
    names = _class_names_for_codes(classes, proba.shape[1])
    min_support = int(config.class_balance_min_class_support)
    min_share = float(config.class_balance_min_predicted_share)
    if not np.isfinite(min_share) or min_share < 0.0:
        raise ValueError("class_balance_min_predicted_share must be finite and >= 0")
    min_entropy = float(config.class_balance_min_normalized_entropy)
    if not np.isfinite(min_entropy) or not 0.0 <= min_entropy <= 1.0:
        raise ValueError(
            "class_balance_min_normalized_entropy must be finite and within [0, 1]"
        )

    def check_scope(
        scope_target: np.ndarray,
        scope_proba: np.ndarray,
        *,
        scope_name: str,
    ) -> dict[str, Any]:
        """Validate one aggregate or single-fold frozen OOF population."""

        support = np.bincount(scope_target, minlength=proba.shape[1]).astype(int)
        unsupported = np.flatnonzero(support < min_support)
        if len(unsupported):
            raise CatBoostClassBalanceError(
                "class-balance OOF guard rejected insufficient validation support"
                f" ({scope_name}) for "
                + ", ".join(names[index] for index in unsupported)
            )
        predicted_share = scope_proba.mean(axis=0)
        collapsed_share = np.flatnonzero(predicted_share < min_share)
        if len(collapsed_share):
            raise CatBoostClassBalanceError(
                "class-balance OOF guard rejected predicted-share collapse"
                f" ({scope_name}) for "
                + ", ".join(names[index] for index in collapsed_share)
            )
        normalized_entropy = -np.sum(
            scope_proba * np.log(np.clip(scope_proba, 1e-12, 1.0)), axis=1
        ) / math.log(proba.shape[1])
        mean_entropy = float(np.mean(normalized_entropy))
        if mean_entropy < min_entropy:
            raise CatBoostClassBalanceError(
                "class-balance OOF guard rejected entropy collapse"
                f" ({scope_name}): {mean_entropy:.6f} < {min_entropy:.6f}"
            )
        argmax_share = np.bincount(
            np.argmax(scope_proba, axis=1), minlength=proba.shape[1]
        ) / float(len(scope_proba))
        return {
            "scope": scope_name,
            "rows": int(len(scope_target)),
            "validation_support": support.tolist(),
            "predicted_probability_share": predicted_share.tolist(),
            "predicted_argmax_share": argmax_share.tolist(),
            "mean_normalized_entropy": mean_entropy,
        }

    aggregate = check_scope(target, proba, scope_name="aggregate")
    per_fold: list[dict[str, Any]] = []
    if fold_ids is not None:
        folds = np.asarray(fold_ids)
        if folds.ndim != 1 or len(folds) != len(target):
            raise CatBoostClassBalanceError(
                "class-balance OOF guard requires one aligned fold id per OOF row"
            )
        for fold_id in np.unique(folds):
            mask = folds == fold_id
            if not np.any(mask):  # Defensive: unique() normally makes this unreachable.
                continue
            report = check_scope(
                target[mask], proba[mask], scope_name=f"fold={fold_id}"
            )
            report["fold_id"] = int(fold_id)
            per_fold.append(report)
    return _json_ready(
        {
            "passed": True,
            "evaluation_scope": "frozen_purged_oof_validation_rows_only_aggregate_and_per_fold"
            if fold_ids is not None
            else "frozen_purged_oof_validation_rows_only_aggregate",
            "final_refit_used_for_selection": False,
            "class_order": list(names),
            **aggregate,
            "aggregate": aggregate,
            "per_fold": per_fold,
            "min_predicted_share_required": min_share,
            "min_normalized_entropy_required": min_entropy,
        }
    )


@dataclass
class OOFPathArchetypeResult:
    probabilities: np.ndarray
    fold_ids: np.ndarray
    folds: list[PurgedFold]
    models: list[Any]
    classes: np.ndarray
    feature_columns: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] | None = None
    staged_matrix_cache: "StagedPermutationMatrixCache | None" = None


@dataclass
class StagedPermutationMatrixCache:
    """Bounded float32 numeric cache for one staged-MDA feature universe.

    The old selector rebuilt DataFrame-to-array conversions and fold-local
    imputation for every stage.  This cache preserves that fold-local median
    rule while retaining the converted raw matrix once.  Materialized
    train/validation matrices are intentionally short lived so the cache does
    not retain one copy per fold and stage.
    """

    feature_columns: tuple[str, ...]
    values: np.ndarray
    column_index: Mapping[str, int]
    train_medians: Mapping[int, np.ndarray]
    validation_medians: Mapping[int, np.ndarray]
    validation_matrices: Mapping[int, np.ndarray]

    @classmethod
    def from_frame(
        cls,
        features: pd.DataFrame,
        folds: Sequence[PurgedFold],
        *,
        validation_cache_max_bytes: int,
    ) -> "StagedPermutationMatrixCache":
        columns = tuple(validate_preentry_features(features.columns))
        # CatBoost's numeric path is float32.  Storing it that way avoids an
        # additional 2x cache while bounding the representation error to one
        # float32 rounding at conversion time.
        values = (
            features.loc[:, columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32, copy=True)
        )
        train_medians: dict[int, np.ndarray] = {}
        validation_medians: dict[int, np.ndarray] = {}
        for fold in folds:
            train_medians[fold.fold_id] = cls._medians(values[fold.train_indices])
            validation_medians[fold.fold_id] = cls._medians(
                values[fold.validation_indices]
            )
        validation_bytes = sum(
            len(fold.validation_indices) * len(columns) * values.dtype.itemsize
            for fold in folds
        )
        validation_matrices: dict[int, np.ndarray] = {}
        if validation_bytes <= max(0, int(validation_cache_max_bytes)):
            for fold in folds:
                matrix = values[fold.validation_indices].copy()
                np.copyto(
                    matrix,
                    validation_medians[fold.fold_id][None, :],
                    where=~np.isfinite(matrix),
                )
                validation_matrices[fold.fold_id] = matrix
        return cls(
            feature_columns=columns,
            values=values,
            column_index={column: index for index, column in enumerate(columns)},
            train_medians=train_medians,
            validation_medians=validation_medians,
            validation_matrices=validation_matrices,
        )

    @staticmethod
    def _medians(values: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        medians = np.asarray(medians, dtype=np.float32)
        medians[~np.isfinite(medians)] = 0.0
        return medians

    def supports(self, columns: Sequence[str]) -> bool:
        return all(column in self.column_index for column in columns)

    def matrix(
        self, fold: PurgedFold, columns: Sequence[str], *, training: bool
    ) -> np.ndarray:
        if not self.supports(columns):
            missing = sorted(set(columns).difference(self.column_index))
            raise KeyError(f"Staged MDA cache is missing features: {missing[:8]}")
        column_indices = np.fromiter(
            (self.column_index[column] for column in columns), dtype=np.intp
        )
        rows = fold.train_indices if training else fold.validation_indices
        cached_validation = (
            None if training else self.validation_matrices.get(fold.fold_id)
        )
        if cached_validation is not None:
            if tuple(columns) == self.feature_columns:
                return cached_validation
            return np.take(cached_validation, column_indices, axis=1)
        # Advanced indexing materializes only the active row/column rectangle,
        # rather than an all-column row copy followed by a second column copy.
        # It is released after the fold instead of caching stage-sized matrices.
        matrix = self.values[np.ix_(rows, column_indices)]
        medians = (
            self.train_medians[fold.fold_id]
            if training
            else self.validation_medians[fold.fold_id]
        )[column_indices]
        np.copyto(matrix, medians[None, :], where=~np.isfinite(matrix))
        return matrix

    @property
    def cache_bytes(self) -> int:
        return int(
            self.values.nbytes
            + sum(matrix.nbytes for matrix in self.validation_matrices.values())
        )


def staged_permutation_feature_order(
    columns: Sequence[str], mandatory_features: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return the historical mandatory-first MDA feature ordering."""
    return tuple(dict.fromkeys([*mandatory_features, *columns]))


def build_staged_permutation_matrix_cache(
    features: pd.DataFrame,
    timestamps: Sequence[Any],
    *,
    label_end: Sequence[Any] | None = None,
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> StagedPermutationMatrixCache:
    """Build the cache shared by pre-MDA OOF fitting and staged MDA."""
    folds = purged_chronological_folds(
        timestamps,
        label_end=label_end,
        n_splits=config.oof_folds,
        embargo=config.embargo,
    )
    return StagedPermutationMatrixCache.from_frame(
        features,
        folds,
        validation_cache_max_bytes=config.permutation_validation_cache_max_bytes,
    )


def predeclared_catboost_class_balance_arms(
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> tuple[dict[str, Any], ...]:
    """Return the only class-balance arms eligible for this classifier HPO.

    The ratio cap is deliberately part of the declared contract.  No caller
    may inject an arbitrary CatBoost ``class_weights`` vector as an additional
    arm: non-uniform weights must be derived from one of these exponents and
    measured only on the corresponding fold's training labels.
    """
    ratio_cap = float(config.class_balance_max_weight_ratio)
    if not np.isfinite(ratio_cap) or ratio_cap < 1.0:
        raise ValueError("class_balance_max_weight_ratio must be finite and >= 1")
    arms: list[dict[str, Any]] = [
        {
            "name": CATBOOST_CLASS_BALANCE_ARM_UNIFORM,
            "frequency_exponent": 0.0,
            "max_weight_ratio": ratio_cap,
            "control": True,
        }
    ]
    for exponent in CATBOOST_CLASS_BALANCE_ARM_EXPONENTS:
        arms.append(
            {
                "name": f"frequency_power_{exponent:.2f}",
                "frequency_exponent": float(exponent),
                "max_weight_ratio": ratio_cap,
                "control": False,
            }
        )
    return tuple(arms)


def _matched_initial_class_balance_hpo_params() -> dict[str, Any]:
    """Return the fixed non-arm coordinates for mandatory arm comparison.

    These values sit inside ``suggest_catboost_hpo_params``' declared search
    bounds.  Every arm receives this exact first purged-OOF evaluation before
    TPE is allowed to vary any hyperparameter, making the control comparison
    fair and durable across an interrupted study.
    """

    return {
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 30.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "rsm": 0.8,
        "border_count": 64,
    }


def catboost_class_balance_search_contract(
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, Any]:
    """Return the versioned balance-search payload for manifests/fingerprints.

    The payload deliberately contains every outcome-sensitive balance setting,
    mandatory baseline schedule, and final-refit weight rule.  Callers may
    embed it unchanged in a higher-level HPO fingerprint; changing any balance
    behaviour then invalidates stale reusable HPO artifacts.
    """

    arms = predeclared_catboost_class_balance_arms(config)
    arm_names = [str(arm["name"]) for arm in arms]
    matched_params = _matched_initial_class_balance_hpo_params()
    return _json_ready(
        {
            "schema": CATBOOST_CLASS_BALANCE_SEARCH_SCHEMA,
            "search_mode": "mandatory_matched_arm_oof_then_free_hpo",
            "declared_arms": [dict(arm) for arm in arms],
            "candidate_arm_count": int(len(arms)),
            "mandatory_initial_evaluation": {
                "required": True,
                "scheduled_arm_order": arm_names,
                "matched_non_arm_hpo_params": matched_params,
                "required_terminal_outcome": (
                    "completed_oof_guard_passed_or_explicit_oof_guard_rejection"
                ),
            },
            "coverage_requirement": {
                "production_minimum_total_trials": int(len(arms)),
                "fail_closed_when_incomplete": True,
                "incomplete_override": (
                    "explicit_non_promotable_smoke_or_diagnostic_only"
                ),
            },
            "oof_guard": {
                "evaluation_scope": "aggregate_and_each_purged_oof_fold",
                "min_class_support": int(config.class_balance_min_class_support),
                "min_predicted_share": float(config.class_balance_min_predicted_share),
                "min_normalized_entropy": float(
                    config.class_balance_min_normalized_entropy
                ),
            },
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "final_weight_materialisation": (
                "derive_again_from_actual_final_refit_train_labels_after_oof_arm_selection"
            ),
        }
    )


def catboost_class_balance_mini_sweep_contract(
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, Any]:
    """Describe the post-structural-HPO fixed-parameter balance evaluation.

    This is the preferred production sequence.  Structural CatBoost
    hyperparameters are selected first from purged OOF evidence without using a
    balance-arm result.  This mini-sweep then evaluates every declared arm on
    identical purged folds and frozen structural parameters.  A caller can join
    the returned OOF probabilities to economic outcomes and select an arm using
    predeclared ML *and* economic OOF criteria.  No final refit is involved in
    that choice.
    """

    arms = predeclared_catboost_class_balance_arms(config)
    return _json_ready(
        {
            "schema": CATBOOST_CLASS_BALANCE_MINI_SWEEP_SCHEMA,
            "evaluation_mode": "fixed_structural_params_per_arm_purged_oof",
            "declared_arms": [dict(arm) for arm in arms],
            "scheduled_arm_order": [str(arm["name"]) for arm in arms],
            "intended_sequence": [
                "freeze_structural_model_hpo_from_purged_oof_without_balance_arm_selection",
                "run_fixed_parameter_balance_mini_sweep_on_identical_purged_oof_folds",
                "choose_arm_from_predeclared_ml_and_economic_oof_evidence",
                "rematerialize_weights_from_actual_final_train_labels_after_arm_selection",
                "final_refit",
            ],
            "fold_weight_contract": "each_arm_each_fold_uses_train_labels_only",
            "oof_guard": {
                "evaluation_scope": "aggregate_and_each_purged_oof_fold",
                "min_class_support": int(config.class_balance_min_class_support),
                "min_predicted_share": float(config.class_balance_min_predicted_share),
                "min_normalized_entropy": float(
                    config.class_balance_min_normalized_entropy
                ),
            },
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "returns_all_arms_without_selecting_one": True,
        }
    )


def catboost_structural_hpo_contract(
    config: PathArchetypeConfig = PathArchetypeConfig(),
) -> dict[str, Any]:
    """Return the fingerprint-ready production HPO contract without arm search.

    Structural CatBoost HPO must not compare architecture parameters while a
    class-balance arm changes at the same time.  It trains every candidate with
    uniform balance, then delegates the four-arm comparison to
    :func:`sweep_purged_catboost_class_balance_arms` after structural parameters
    are frozen.
    """

    return _json_ready(
        {
            "schema": CATBOOST_STRUCTURAL_HPO_SCHEMA,
            "hpo_mode": "structural_only_uniform_balance",
            "fixed_class_balance_arm": CATBOOST_CLASS_BALANCE_ARM_UNIFORM,
            "class_balance_arm_is_hpo_dimension": False,
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "required_next_stage": "fixed_parameter_class_balance_mini_sweep",
            "post_hpo_balance_mini_sweep_contract": (
                catboost_class_balance_mini_sweep_contract(config)
            ),
        }
    )


def _predeclared_class_balance_arm(
    arm_name: Any,
    *,
    config: PathArchetypeConfig,
) -> dict[str, Any]:
    requested = (
        CATBOOST_CLASS_BALANCE_ARM_UNIFORM if arm_name is None else str(arm_name)
    )
    arms = {
        str(arm["name"]): arm for arm in predeclared_catboost_class_balance_arms(config)
    }
    if requested not in arms:
        raise ValueError(
            "class_balance_arm must be one of the predeclared arms: " + ", ".join(arms)
        )
    return dict(arms[requested])


def _class_names_for_codes(
    classes: Sequence[Any] | None,
    class_count: int,
) -> tuple[str, ...]:
    if classes is None:
        return tuple(str(index) for index in range(int(class_count)))
    values = tuple(map(str, classes))
    if len(values) != int(class_count) or len(set(values)) != len(values):
        raise ValueError("class names must be unique and align with class codes")
    return values


def _predeclared_class_balance_weights(
    target_codes: Sequence[int],
    *,
    classes: Sequence[Any] | None,
    arm_name: Any,
    config: PathArchetypeConfig,
    provenance_scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive one bounded, ordered weight vector from authorised train labels."""
    target = np.asarray(target_codes, dtype=int)
    if target.ndim != 1 or not len(target) or (target < 0).any():
        raise CatBoostClassBalanceError(
            "class-balance fitting requires one non-missing non-negative class "
            "code per row"
        )
    class_count = len(classes) if classes is not None else int(np.max(target)) + 1
    names = _class_names_for_codes(classes, class_count)
    if np.max(target) >= class_count:
        raise CatBoostClassBalanceError("class code exceeds the frozen class order")
    min_support = int(config.class_balance_min_class_support)
    if min_support < 1:
        raise ValueError("class_balance_min_class_support must be positive")
    support = np.bincount(target, minlength=class_count).astype(int)
    collapsed = np.flatnonzero(support < min_support)
    if len(collapsed):
        labels = ", ".join(names[index] for index in collapsed)
        raise CatBoostClassBalanceError(
            "class-balance arm rejected: train class support below "
            f"{min_support} for {labels}"
        )
    arm = _predeclared_class_balance_arm(arm_name, config=config)
    exponent = float(arm["frequency_exponent"])
    # Relative to the majority class, then cap the minority/majority ratio
    # before mean-normalising.  The ratio is invariant to normalization.
    raw = np.power(float(np.max(support)) / support.astype(float), exponent)
    raw = np.clip(raw, 1.0, float(arm["max_weight_ratio"]))
    weights = raw / np.average(raw, weights=support.astype(float))
    ratio = float(np.max(weights) / np.min(weights))
    if (
        not np.isfinite(weights).all()
        or (weights <= 0.0).any()
        or ratio > float(arm["max_weight_ratio"]) + 1e-12
    ):
        raise CatBoostClassBalanceError("class-balance arm produced unsafe weights")
    provenance = {
        "schema": "catboost_path_archetype_class_balance_v1",
        "arm": str(arm["name"]),
        "frequency_exponent": exponent,
        "max_weight_ratio": float(arm["max_weight_ratio"]),
        "realized_weight_ratio": ratio,
        "class_order": list(names),
        "class_support": support.tolist(),
        "weight_estimation_scope": str(provenance_scope),
        "selection_evidence": "purged_chronological_oof_validation_only",
        "final_refit_used_for_selection": False,
    }
    return weights.astype(float), provenance


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable digest for compact OOF-selection provenance.

    The digest is deliberately over the complete selection object, not only
    the compact fields copied into final-fit provenance.  This keeps the
    final model auditable against the exact OOF decision artifact without
    embedding fold-level probabilities or economic reports in model params.
    """

    try:
        encoded = json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "class-balance selection provenance must be canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _validated_sha256_link(value: Any, *, field: str) -> str:
    """Validate an immutable artifact digest before linking it to final fit."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"class-balance selection {field} must be a SHA-256 hex digest"
        )
    return value


def _selection_provenance_final_links(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract only compact immutable links for final weight provenance."""

    links: dict[str, Any] = {
        "selected_arm_selection_provenance_sha256": _canonical_json_sha256(provenance)
    }
    for field in _CLASS_BALANCE_SELECTION_LINK_FIELDS:
        if field not in provenance:
            continue
        value = provenance[field]
        if field.endswith("_sha256"):
            value = _validated_sha256_link(value, field=field)
        elif not isinstance(value, str) or not value:
            raise ValueError(
                "class-balance selection "
                f"{field} must be a non-empty string when present"
            )
        links[f"selected_arm_{field}"] = value
    return links


def _validated_oof_selected_class_balance_arm(
    params: Mapping[str, Any],
    *,
    config: PathArchetypeConfig,
    require_complete_coverage: bool = True,
) -> dict[str, Any]:
    """Validate that an arm was selected from frozen purged-OOF evidence only."""

    arm = _predeclared_class_balance_arm(params.get("class_balance_arm"), config=config)
    raw = params.get("class_balance_selection_provenance")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "class-balance final refit requires OOF-selected arm provenance; "
            "HPO-sample weight vectors cannot be reused"
        )
    provenance = dict(raw)
    bulky_fields = sorted(
        _CLASS_BALANCE_SELECTION_BULKY_REPORT_FIELDS.intersection(provenance)
    )
    if bulky_fields:
        raise ValueError(
            "class-balance selection provenance must link to, not embed, "
            "raw OOF reports: " + ", ".join(bulky_fields)
        )
    if (
        provenance.get("schema") != CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA
        or provenance.get("arm") != arm["name"]
        or provenance.get("selection_evidence")
        != "purged_chronological_oof_validation_only"
        or bool(provenance.get("final_refit_used_for_selection", True))
    ):
        raise ValueError(
            "class-balance arm lacks the required frozen purged-OOF selection provenance"
        )
    class_order = tuple(map(str, provenance.get("class_order", ())))
    if not class_order:
        raise ValueError("class-balance arm selection provenance lacks class order")
    if config.class_order is not None and class_order != tuple(
        map(str, config.class_order)
    ):
        raise ValueError(
            "class-balance arm selection provenance does not preserve the frozen class order"
        )
    if require_complete_coverage and (
        not bool(provenance.get("mandatory_initial_coverage_complete", False))
        or not bool(provenance.get("promotion_eligible", False))
    ):
        raise ValueError(
            "class-balance arm selection is coverage-incomplete and non-promotable"
        )
    return provenance


def rematerialize_final_class_balance_params(
    params: Mapping[str, Any],
    target: Sequence[Any],
    *,
    config: PathArchetypeConfig = PathArchetypeConfig(),
    allow_nonpromotable_selection: bool = False,
) -> dict[str, Any]:
    """Derive final-fit class weights from actual final-train labels.

    HPO determines only the balance *arm* from purged OOF data.  The final
    refit has a different labelled support than the HPO proxy, so it must
    materialise a fresh bounded vector here.  This helper is intentionally
    explicit: callers cannot accidentally reuse an HPO-sample vector.
    """

    materialized = dict(params)
    arm = _predeclared_class_balance_arm(
        materialized.get("class_balance_arm"), config=config
    )
    selection_provenance = _validated_oof_selected_class_balance_arm(
        materialized,
        config=config,
        require_complete_coverage=not allow_nonpromotable_selection,
    )
    encoded = _categorical_target(target, pd.RangeIndex(len(target)), config=config)
    final_class_order = tuple(map(str, encoded.cat.categories))
    if final_class_order != tuple(map(str, selection_provenance["class_order"])):
        raise ValueError(
            "actual final refit labels do not preserve the OOF-selected class order"
        )
    weights, final_provenance = _predeclared_class_balance_weights(
        encoded.cat.codes.to_numpy(),
        classes=encoded.cat.categories,
        arm_name=arm["name"],
        config=config,
        provenance_scope="final_refit_train_labels_after_oof_arm_selection",
    )
    final_provenance.update(
        {
            "selected_arm_provenance_schema": selection_provenance["schema"],
            "selected_arm_mandatory_initial_coverage_complete": bool(
                selection_provenance.get("mandatory_initial_coverage_complete", False)
            ),
            "selected_arm_promotion_eligible": bool(
                selection_provenance.get("promotion_eligible", False)
            ),
        }
    )
    # Preserve a compact, immutable connection to the full OOF selection
    # artifact.  The complete economic/OOF report remains an external
    # artifact: final model parameters contain only its digest and explicit
    # contract links, never probability arrays or fold-level reports.
    final_provenance.update(_selection_provenance_final_links(selection_provenance))
    materialized["class_balance_final_weights"] = weights.tolist()
    materialized["class_balance_provenance"] = final_provenance
    return materialized


def _validated_persisted_class_balance_weights(
    params: Mapping[str, Any],
    *,
    config: PathArchetypeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate a selected OOF arm payload before it reaches CatBoost final fit."""
    if (
        "class_balance_final_weights" not in params
        or "class_balance_provenance" not in params
    ):
        raise ValueError(
            "a non-uniform class_balance_arm requires OOF-selected final weights "
            "and provenance; arbitrary class_weights are not permitted"
        )
    arm = _predeclared_class_balance_arm(params.get("class_balance_arm"), config=config)
    provenance_raw = params.get("class_balance_provenance")
    if not isinstance(provenance_raw, Mapping):
        raise ValueError("class_balance_provenance must be a mapping")
    provenance = dict(provenance_raw)
    if (
        provenance.get("schema") != "catboost_path_archetype_class_balance_v1"
        or provenance.get("arm") != arm["name"]
        or provenance.get("selection_evidence")
        != "purged_chronological_oof_validation_only"
        or bool(provenance.get("final_refit_used_for_selection", True))
        or provenance.get("weight_estimation_scope")
        != "final_refit_train_labels_after_oof_arm_selection"
    ):
        raise ValueError(
            "class-balance weights lack required OOF-selection and final-refit provenance"
        )
    weights = np.asarray(params.get("class_balance_final_weights"), dtype=float)
    class_order = tuple(map(str, provenance.get("class_order", ())))
    if weights.ndim != 1 or not len(weights) or len(weights) != len(class_order):
        raise ValueError(
            "class-balance final weights must align with the recorded class order"
        )
    if config.class_order is not None and class_order != tuple(
        map(str, config.class_order)
    ):
        raise ValueError(
            "class-balance final weights do not preserve the frozen class order"
        )
    ratio_cap = float(arm["max_weight_ratio"])
    ratio = float(np.max(weights) / np.min(weights)) if len(weights) else float("inf")
    if (
        not np.isfinite(weights).all()
        or (weights <= 0.0).any()
        or ratio > ratio_cap + 1e-12
    ):
        raise ValueError(
            "class-balance final weights are unsafe or exceed the declared cap"
        )
    if not np.isclose(
        float(provenance.get("frequency_exponent", float("nan"))),
        float(arm["frequency_exponent"]),
    ):
        raise ValueError("class-balance provenance exponent does not match its arm")
    selection_digest = provenance.get("selected_arm_selection_provenance_sha256")
    if selection_digest is not None:
        _validated_sha256_link(
            selection_digest,
            field="selected_arm_selection_provenance_sha256",
        )
    return weights, provenance


def _catboost_fold_params(
    config: PathArchetypeConfig,
    params: Mapping[str, Any] | None,
    *,
    target_codes: Sequence[int],
    classes: Sequence[Any],
    provenance_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialise an eligible balance arm using *only* a fold's train rows."""
    requested = dict(params or {})
    arm_name = requested.get("class_balance_arm", CATBOOST_CLASS_BALANCE_ARM_UNIFORM)
    arm = _predeclared_class_balance_arm(arm_name, config=config)
    weights, provenance = _predeclared_class_balance_weights(
        target_codes,
        classes=classes,
        arm_name=arm["name"],
        config=config,
        provenance_scope=provenance_scope,
    )
    # This private hand-off is generated immediately above.  It is distinct
    # from user-provided CatBoost ``class_weights``, which remain rejected by
    # the public parameter validator below.
    requested["_predeclared_class_balance_weights"] = weights.tolist()
    requested["_predeclared_class_balance_provenance"] = provenance
    return _catboost_params(config, requested), provenance


def _catboost_params(
    config: PathArchetypeConfig, params: Mapping[str, Any] | None
) -> dict[str, Any]:
    requested = dict(params or {})
    generated_weights = requested.pop("_predeclared_class_balance_weights", None)
    generated_provenance = requested.pop("_predeclared_class_balance_provenance", None)
    arm_name = requested.pop("class_balance_arm", None)
    final_weights_present = "class_balance_final_weights" in requested
    final_provenance_present = "class_balance_provenance" in requested
    # These are pipeline metadata, never CatBoost parameters.
    persisted_payload = {
        key: requested.pop(key)
        for key in ("class_balance_final_weights", "class_balance_provenance")
        if key in requested
    }
    # OOF arm-selection evidence is retained in artifacts and consumed by the
    # explicit final-weight materialiser, never forwarded as a CatBoost option.
    selection_payload = requested.pop("class_balance_selection_provenance", None)
    base = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 3000,
        "od_wait": 150,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 30.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "rsm": 0.8,
        "border_count": 64,
        "auto_class_weights": None,
        "bootstrap_type": "Bayesian",
        "grow_policy": "SymmetricTree",
        "random_seed": config.random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    base.update(requested)
    if not config.legacy_allow_class_weights:
        if base.get("auto_class_weights") is not None:
            raise ValueError(
                "future-training CatBoost requires uniform sample weights; "
                "auto_class_weights must be None; only uniform weights are allowed"
            )
        if base.get("class_weights") is not None:
            raise ValueError(
                "future-training CatBoost requires uniform sample weights; "
                "class_weights are not permitted; only uniform weights are allowed"
            )
        if base.get("scale_pos_weight") is not None:
            raise ValueError(
                "future-training CatBoost requires uniform sample weights; "
                "scale_pos_weight is not permitted; only uniform weights are allowed"
            )
        if generated_weights is not None or generated_provenance is not None:
            if generated_weights is None or not isinstance(
                generated_provenance, Mapping
            ):
                raise ValueError("generated class-balance weights require provenance")
            generated_arm = _predeclared_class_balance_arm(
                generated_provenance.get("arm"), config=config
            )
            if arm_name not in (None, generated_arm["name"]):
                raise ValueError(
                    "generated class-balance arm does not match requested arm"
                )
            weights = np.asarray(generated_weights, dtype=float)
            ratio = (
                float(np.max(weights) / np.min(weights))
                if len(weights)
                else float("inf")
            )
            if (
                weights.ndim != 1
                or not len(weights)
                or not np.isfinite(weights).all()
                or (weights <= 0.0).any()
                or ratio > float(generated_arm["max_weight_ratio"]) + 1e-12
            ):
                raise ValueError("generated class-balance weights are unsafe")
            base["class_weights"] = weights.tolist()
        elif arm_name is not None and (
            arm_name != CATBOOST_CLASS_BALANCE_ARM_UNIFORM
            or final_weights_present
            or final_provenance_present
        ):
            # Final refits invoked outside the fold helper may only consume a
            # frozen OOF-selected payload.  The payload is auditable and does
            # not let a final fit choose its own class-weight arm.
            weights, final_provenance = _validated_persisted_class_balance_weights(
                {
                    "class_balance_arm": arm_name,
                    **persisted_payload,
                },
                config=config,
            )
            selection_provenance = _validated_oof_selected_class_balance_arm(
                {
                    "class_balance_arm": arm_name,
                    "class_balance_selection_provenance": selection_payload,
                },
                config=config,
            )
            recorded_selection_digest = final_provenance.get(
                "selected_arm_selection_provenance_sha256"
            )
            if (
                recorded_selection_digest is not None
                and recorded_selection_digest
                != _canonical_json_sha256(selection_provenance)
            ):
                raise ValueError(
                    "class-balance final weights do not match the frozen "
                    "OOF selection provenance digest"
                )
            base["class_weights"] = weights.tolist()
        elif final_weights_present or final_provenance_present:
            raise ValueError(
                "class-balance final weights are valid only with a selected "
                "non-uniform arm"
            )
    return _catboost_params_with_resource_limits(base, config=config)


def _categorical_target(
    target: Sequence[Any],
    index: pd.Index,
    *,
    config: PathArchetypeConfig,
) -> pd.Series:
    """Encode a target against an optional frozen ordered taxonomy."""

    values = pd.Series(target, index=index).astype("string").str.strip()
    if config.class_order is None:
        return values.astype("category")
    classes = tuple(map(str, config.class_order))
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("class_order must be a non-empty, unique ordered taxonomy")
    unexpected = sorted(set(values.dropna().astype(str)).difference(classes))
    if unexpected:
        raise ValueError(
            "target contains labels outside the frozen path taxonomy: "
            + ", ".join(unexpected[:8])
        )
    return pd.Series(
        pd.Categorical(values, categories=list(classes), ordered=True), index=index
    )


def path_archetype_probability_contract(
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Return the fixed seven-class raw-probability scoring contract."""

    classes = tuple(map(str, class_names))
    if classes != MERGED_PATH_ARCHETYPE_CLASSES:
        raise ValueError(
            "future path-archetype scoring requires the ordered seven-class taxonomy"
        )
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(classes):
        raise ValueError("probabilities must have one column for each frozen class")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("raw probabilities must sum to one")
    output = pd.DataFrame(values, index=index, columns=classes)
    ordered = np.sort(values, axis=1)
    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
    output["max_probability"] = values.max(axis=1)
    output["probability_entropy"] = entropy
    output["normalized_entropy"] = entropy / math.log(len(classes))
    output["top2_probability_margin"] = ordered[:, -1] - ordered[:, -2]
    output["adverse_probability_mass"] = output.loc[
        :, PATH_ARCHETYPE_ADVERSE_CLASSES
    ].sum(axis=1)
    output["favorable_probability_mass"] = output.loc[
        :, PATH_ARCHETYPE_FAVORABLE_CLASSES
    ].sum(axis=1)
    return output


def fit_purged_chronological_oof_catboost(
    features: pd.DataFrame,
    target: Sequence[Any],
    timestamps: Sequence[Any],
    *,
    label_end: Sequence[Any] | None = None,
    config: PathArchetypeConfig = PathArchetypeConfig(),
    params: Mapping[str, Any] | None = None,
    fold_callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    staged_matrix_cache: StagedPermutationMatrixCache | None = None,
    force_classes_count: bool = True,
) -> OOFPathArchetypeResult:
    """Fit one CatBoost model per expanding, purged validation fold."""
    feature_columns = validate_preentry_features(features.columns)
    CatBoostClassifier = _require_catboost()
    y_series = _categorical_target(target, features.index, config=config)
    y, classes = y_series.cat.codes.to_numpy(), y_series.cat.categories.to_numpy()
    folds = purged_chronological_folds(
        timestamps,
        label_end=label_end,
        n_splits=config.oof_folds,
        embargo=config.embargo,
    )
    if staged_matrix_cache is not None and not staged_matrix_cache.supports(
        feature_columns
    ):
        missing = sorted(
            set(feature_columns).difference(staged_matrix_cache.column_index)
        )
        raise KeyError(f"Staged MDA cache is missing OOF features: {missing[:8]}")
    x = (
        None
        if staged_matrix_cache is not None
        else _finite_matrix(features, feature_columns)
    )
    probabilities = np.full((len(features), len(classes)), np.nan)
    fold_ids = np.full(len(features), -1, dtype=int)
    models: list[Any] = []
    fitted_folds: list[PurgedFold] = []
    fold_fit_reports: list[dict[str, Any]] = []
    for fold in folds:
        if len(np.unique(y[fold.train_indices])) < 2:
            continue
        train_matrix = (
            staged_matrix_cache.matrix(fold, feature_columns, training=True)
            if staged_matrix_cache is not None
            else x[fold.train_indices]
        )
        validation_matrix = (
            staged_matrix_cache.matrix(fold, feature_columns, training=False)
            if staged_matrix_cache is not None
            else x[fold.validation_indices]
        )
        fold_params, balance_provenance = _catboost_fold_params(
            config,
            params,
            target_codes=y[fold.train_indices],
            classes=classes,
            provenance_scope=("purged_chronological_oof_fold_train_only"),
        )
        # Ordinary OOF fitting keeps the globally frozen archetype universe
        # explicit.  Staged MDA can opt out to reproduce its historical fold
        # estimator exactly while reusing those models in its first stage.
        if force_classes_count:
            fold_params["classes_count"] = int(len(classes))
        fold_params["use_best_model"] = True
        model = CatBoostClassifier(**fold_params)
        model.fit(
            train_matrix,
            y[fold.train_indices],
            eval_set=(validation_matrix, y[fold.validation_indices]),
            early_stopping_rounds=int(fold_params["od_wait"]),
            verbose=False,
        )
        local = model.predict_proba(validation_matrix)
        aligned = np.zeros((len(fold.validation_indices), len(classes)))
        aligned[:, np.asarray(model.classes_, dtype=int)] = local
        probabilities[fold.validation_indices] = aligned
        fold_ids[fold.validation_indices] = fold.fold_id
        models.append(model)
        fitted_folds.append(fold)
        best_iteration = None
        if hasattr(model, "get_best_iteration"):
            value = model.get_best_iteration()
            best_iteration = int(value) if value is not None else None
        tree_count = getattr(model, "tree_count_", None)
        fold_fit_reports.append(
            {
                "fold_id": int(fold.fold_id),
                "train_rows": int(len(fold.train_indices)),
                "validation_rows": int(len(fold.validation_indices)),
                "use_best_model": bool(fold_params["use_best_model"]),
                "eval_set_used": True,
                "early_stopping_rounds": int(fold_params["od_wait"]),
                "best_iteration": best_iteration,
                "tree_count": int(tree_count) if tree_count is not None else None,
                "class_balance": balance_provenance,
            }
        )
        LOGGER.info(
            "CatBoost purged OOF fold %s complete: train=%s validation=%s "
            "best_iteration=%s trees=%s ceiling=%s od_wait=%s",
            fold.fold_id,
            len(fold.train_indices),
            len(fold.validation_indices),
            best_iteration,
            int(tree_count) if tree_count is not None else None,
            int(fold_params["iterations"]),
            int(fold_params["od_wait"]),
        )
        if fold_callback is not None:
            fold_callback(len(fitted_folds) - 1, probabilities, fold_ids)
    diagnostics = None
    valid = fold_ids >= 0
    if np.any(valid):
        diagnostics = multiclass_classification_diagnostics(
            y[valid],
            probabilities[valid],
            fold_ids=fold_ids[valid],
            class_names=classes,
        )
        diagnostics["fold_fit_reports"] = fold_fit_reports
    return OOFPathArchetypeResult(
        probabilities,
        fold_ids,
        fitted_folds,
        models,
        classes,
        tuple(feature_columns),
        diagnostics,
        staged_matrix_cache,
    )


@dataclass(frozen=True)
class CatBoostClassBalanceMiniSweepArmResult:
    """One fixed-parameter arm's purged-OOF evidence; never a final refit."""

    arm: str
    params: Mapping[str, Any]
    status: str
    oof: OOFPathArchetypeResult | None
    objective_components: Mapping[str, Any] | None
    guard: Mapping[str, Any]
    fold_signature: Mapping[str, Any] | None
    fold_balance_provenance: tuple[Mapping[str, Any], ...] | None
    rejection_reason: str | None = None

    def report(self) -> dict[str, Any]:
        """Return compact evidence while retaining raw OOF arrays on ``oof``."""

        return _json_ready(
            {
                "arm": self.arm,
                "status": self.status,
                "params": dict(self.params),
                "objective_components": self.objective_components,
                "oof_guard": self.guard,
                "fold_signature": self.fold_signature,
                "fold_balance_provenance": self.fold_balance_provenance,
                "rejection_reason": self.rejection_reason,
                "oof_probability_arrays_exposed": self.oof is not None,
                "final_refit_used_for_selection": False,
            }
        )


@dataclass(frozen=True)
class CatBoostClassBalanceMiniSweepResult:
    """All fixed-parameter class-balance OOF arms, deliberately unranked."""

    structural_params: Mapping[str, Any]
    arms: tuple[CatBoostClassBalanceMiniSweepArmResult, ...]
    contract: Mapping[str, Any]

    @property
    def eligible_arms(self) -> tuple[CatBoostClassBalanceMiniSweepArmResult, ...]:
        return tuple(arm for arm in self.arms if arm.status == "eligible")

    @property
    def oof_by_arm(self) -> Mapping[str, OOFPathArchetypeResult]:
        """Expose raw probability arrays for an external OOF economic scorer."""

        return {arm.arm: arm.oof for arm in self.arms if arm.oof is not None}

    def report(self) -> dict[str, Any]:
        """Compact manifest evidence; intentionally does not select a winner."""

        return _json_ready(
            {
                "contract": dict(self.contract),
                "structural_params": dict(self.structural_params),
                "arms": [arm.report() for arm in self.arms],
                "eligible_arms": [arm.arm for arm in self.eligible_arms],
                "winner_selected": False,
                "final_refit_used_for_selection": False,
            }
        )


def _fixed_structural_params_for_class_balance_mini_sweep(
    params: Mapping[str, Any],
    *,
    config: PathArchetypeConfig,
) -> tuple[dict[str, Any], list[str]]:
    """Remove only OOF balance metadata; reject any final-refit payload."""

    structural = dict(params)
    final_refit_keys = {
        "class_balance_final_weights",
        "class_balance_provenance",
    }
    present_final_refit_keys = sorted(final_refit_keys.intersection(structural))
    if present_final_refit_keys:
        raise ValueError(
            "class-balance mini-sweep accepts structural HPO parameters only; "
            "final-refit balance payloads are forbidden: "
            + ", ".join(present_final_refit_keys)
        )
    for key in ("class_weights", "scale_pos_weight"):
        if structural.get(key) is not None:
            raise ValueError(
                "class-balance mini-sweep derives each arm's fold-local weights; "
                f"{key} must not be supplied"
            )
    if structural.get("auto_class_weights") is not None:
        raise ValueError("class-balance mini-sweep requires auto_class_weights=None")
    removed: list[str] = []
    arm = structural.pop("class_balance_arm", None)
    if arm is not None:
        _predeclared_class_balance_arm(arm, config=config)
        removed.append("class_balance_arm")
    if "class_balance_selection_provenance" in structural:
        structural.pop("class_balance_selection_provenance")
        removed.append("class_balance_selection_provenance")
    return structural, removed


def _class_balance_mini_sweep_fold_signature(
    oof: OOFPathArchetypeResult,
) -> dict[str, Any]:
    """Summarise the exact OOF fold layout without embedding row identities."""

    valid = np.asarray(oof.fold_ids) >= 0
    fold_ids = np.asarray(oof.fold_ids)
    return {
        "oof_row_count": int(np.sum(valid)),
        "fold_validation_rows": {
            str(fold_id): int(np.sum(fold_ids == fold_id))
            for fold_id in sorted(map(int, np.unique(fold_ids[valid])))
        },
        "fitted_fold_ids": [int(fold.fold_id) for fold in oof.folds],
    }


def _class_balance_mini_sweep_fold_balance_provenance(
    oof: OOFPathArchetypeResult,
) -> tuple[Mapping[str, Any], ...]:
    """Expose only the per-fold train-derived balance evidence in compact form."""

    diagnostics = oof.diagnostics if isinstance(oof.diagnostics, Mapping) else {}
    reports = diagnostics.get("fold_fit_reports", [])
    if not isinstance(reports, list):
        return ()
    return tuple(
        {
            "fold_id": report.get("fold_id"),
            "class_balance": report.get("class_balance"),
        }
        for report in reports
        if isinstance(report, Mapping)
    )


def sweep_purged_catboost_class_balance_arms(
    features: pd.DataFrame,
    target: Sequence[Any],
    timestamps: Sequence[Any],
    *,
    structural_params: Mapping[str, Any],
    label_end: Sequence[Any] | None = None,
    config: PathArchetypeConfig = PathArchetypeConfig(),
    arm_callback: Callable[[CatBoostClassBalanceMiniSweepArmResult], None]
    | None = None,
    arm_fold_callback: Callable[[str, int, np.ndarray, np.ndarray], None] | None = None,
) -> CatBoostClassBalanceMiniSweepResult:
    """Evaluate all declared balance arms after structural HPO, without selecting.

    ``structural_params`` must be frozen before this call.  Every arm is fitted
    through the same purged chronological OOF routine, which derives weights
    solely from that fold's training labels.  The returned raw OOF probabilities
    are intentionally available to a downstream economic scorer; this function
    only supplies ML diagnostics and OOF safety guards, and never refits a final
    model or picks a winning arm.  ``arm_fold_callback`` receives the declared
    arm name plus the underlying OOF ``fold_callback`` progress tuple, so a
    caller can checkpoint resources after every heavy fold.  ``arm_callback``
    remains the once-per-terminal-arm notification.
    """

    base_params, removed_metadata = (
        _fixed_structural_params_for_class_balance_mini_sweep(
            structural_params, config=config
        )
    )
    contract = {
        **catboost_class_balance_mini_sweep_contract(config),
        "structural_params": _json_ready(base_params),
        "removed_input_balance_metadata": removed_metadata,
    }
    encoded_target = _categorical_target(target, features.index, config=config)
    target_codes = encoded_target.cat.codes.to_numpy()
    results: list[CatBoostClassBalanceMiniSweepArmResult] = []
    reference_fold_ids: np.ndarray | None = None
    reference_classes: tuple[str, ...] | None = None
    for declared_arm in predeclared_catboost_class_balance_arms(config):
        arm_name = str(declared_arm["name"])
        params = {**base_params, "class_balance_arm": arm_name}
        oof: OOFPathArchetypeResult | None = None
        try:
            oof = fit_purged_chronological_oof_catboost(
                features,
                target,
                timestamps,
                label_end=label_end,
                config=config,
                params=params,
                fold_callback=(
                    (
                        lambda fold_index,
                        partial_probabilities,
                        partial_fold_ids,
                        _arm_name=arm_name: arm_fold_callback(
                            _arm_name,
                            fold_index,
                            partial_probabilities,
                            partial_fold_ids,
                        )
                    )
                    if arm_fold_callback is not None
                    else None
                ),
            )
            valid = np.asarray(oof.fold_ids) >= 0
            if not np.any(valid):
                raise CatBoostClassBalanceError(
                    "class-balance mini-sweep found no purged OOF validation predictions"
                )
            current_fold_ids = np.asarray(oof.fold_ids)
            current_classes = tuple(map(str, oof.classes))
            if reference_fold_ids is None:
                reference_fold_ids = current_fold_ids.copy()
                reference_classes = current_classes
            elif (
                not np.array_equal(reference_fold_ids, current_fold_ids)
                or reference_classes != current_classes
            ):
                raise RuntimeError(
                    "class-balance mini-sweep arms must use identical purged OOF folds "
                    "and class order"
                )
            components = catboost_hpo_objective_components(
                target_codes[valid], oof.probabilities[valid], oof.fold_ids[valid]
            )
            guard = class_balance_oof_guard(
                target_codes[valid],
                oof.probabilities[valid],
                classes=oof.classes,
                fold_ids=oof.fold_ids[valid],
                config=config,
            )
            result = CatBoostClassBalanceMiniSweepArmResult(
                arm=arm_name,
                params=params,
                status="eligible",
                oof=oof,
                objective_components=components,
                guard=guard,
                fold_signature=_class_balance_mini_sweep_fold_signature(oof),
                fold_balance_provenance=(
                    _class_balance_mini_sweep_fold_balance_provenance(oof)
                ),
            )
        except CatBoostClassBalanceError as exc:
            result = CatBoostClassBalanceMiniSweepArmResult(
                arm=arm_name,
                params=params,
                status="rejected_by_oof_guard",
                oof=oof,
                objective_components=None,
                guard={
                    "passed": False,
                    "evaluation_scope": (
                        "frozen_purged_oof_validation_rows_only_aggregate_and_per_fold"
                    ),
                    "final_refit_used_for_selection": False,
                    "rejection_reason": str(exc),
                },
                fold_signature=(
                    _class_balance_mini_sweep_fold_signature(oof)
                    if oof is not None
                    else None
                ),
                fold_balance_provenance=(
                    _class_balance_mini_sweep_fold_balance_provenance(oof)
                    if oof is not None
                    else None
                ),
                rejection_reason=str(exc),
            )
        results.append(result)
        if arm_callback is not None:
            arm_callback(result)
    return CatBoostClassBalanceMiniSweepResult(
        structural_params=base_params,
        arms=tuple(results),
        contract=contract,
    )


def suggest_catboost_hpo_params(
    trial: Any,
    *,
    config: PathArchetypeConfig = PathArchetypeConfig(),
    structural_only_hpo: bool = False,
) -> dict[str, Any]:
    """Sample structural CatBoost parameters, optionally fixing uniform balance."""
    balance_arms = predeclared_catboost_class_balance_arms(config)
    params = {
        "depth": trial.suggest_int("depth", 5, 7),
        "iterations": 3000,
        "od_wait": 150,
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 8.0, 80.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 3.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.2, 2.5),
        "rsm": trial.suggest_float("rsm", 0.65, 1.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128]),
        "auto_class_weights": None,
        "bootstrap_type": "Bayesian",
        "grow_policy": "SymmetricTree",
    }
    params["class_balance_arm"] = (
        CATBOOST_CLASS_BALANCE_ARM_UNIFORM
        if structural_only_hpo
        else trial.suggest_categorical(
            "class_balance_arm", [str(arm["name"]) for arm in balance_arms]
        )
    )
    return params


@dataclass(frozen=True)
class CatBoostHPOResult:
    """JSON-persistable Optuna outcome plus the selected trial's OOF evidence."""

    best_params: Mapping[str, Any]
    best_objective: float
    oof: OOFPathArchetypeResult
    trials: tuple[Mapping[str, Any], ...]
    study_name: str | None
    best_oof_reused_from_current_process: bool = False
    class_balance_search: Mapping[str, Any] | None = None

    def report(self) -> dict[str, Any]:
        return _json_ready(
            {
                "objective_name": "purged_oof_logloss_plus_temporal_penalties",
                "direction": "minimize",
                "best_objective": self.best_objective,
                "best_params": dict(self.best_params),
                "study_name": self.study_name,
                "trials": list(self.trials),
                "oof_diagnostics": self.oof.diagnostics,
                "best_oof_reused_from_current_process": (
                    self.best_oof_reused_from_current_process
                ),
                "class_balance_search": self.class_balance_search,
            }
        )


def _hpo_trial_record(
    trial: Any,
    *,
    search_iterations: int,
    search_od_wait: int,
    no_improvement_trials: int,
) -> dict[str, Any]:
    return _json_ready(
        {
            "number": int(trial.number),
            "state": str(trial.state.name),
            "value": trial.value,
            "params": trial.params,
            "effective_search_iterations": int(search_iterations),
            "effective_search_od_wait": int(search_od_wait),
            "pruner": (
                "MedianPruner(startup_trials=3,warmup_steps=0,"
                "interval_steps=1,min_trials=2)"
            ),
            "study_no_improvement_patience_trials": int(no_improvement_trials),
            "objective_components": trial.user_attrs.get("objective_components"),
            "oof_diagnostics": trial.user_attrs.get("oof_diagnostics"),
            "class_balance_arm": trial.params.get(
                "class_balance_arm",
                trial.system_attrs.get("fixed_params", {}).get(
                    "class_balance_arm",
                    trial.user_attrs.get("evaluated_class_balance_arm"),
                ),
            ),
            "class_balance_search_phase": trial.user_attrs.get(
                "class_balance_search_phase"
            ),
            "class_balance_schedule_index": trial.user_attrs.get(
                "class_balance_schedule_index"
            ),
            "class_balance_guard": trial.user_attrs.get("class_balance_guard"),
        }
    )


def _hpo_state_counts(trials: Sequence[Any]) -> dict[str, int]:
    counts = {"completed": 0, "pruned": 0, "failed": 0, "running": 0}
    for trial in trials:
        state = str(trial.state.name)
        if state == "COMPLETE":
            counts["completed"] += 1
        elif state == "PRUNED":
            counts["pruned"] += 1
        elif state == "FAIL":
            counts["failed"] += 1
        elif state == "RUNNING":
            counts["running"] += 1
    return counts


def optimize_purged_catboost_hpo(
    features: pd.DataFrame,
    target: Sequence[Any],
    timestamps: Sequence[Any],
    *,
    label_end: Sequence[Any] | None = None,
    config: PathArchetypeConfig = PathArchetypeConfig(),
    n_trials: int = 30,
    study_name: str | None = None,
    storage: str | None = None,
    search_iterations: int = 1500,
    search_od_wait: int = 100,
    no_improvement_trials: int = 30,
    progress_path: Path | None = None,
    allow_incomplete_class_balance_coverage: bool = False,
    structural_only_hpo: bool = False,
) -> CatBoostHPOResult:
    """Run minimising Optuna HPO using only purged chronological OOF loss.

    ``n_trials`` is a total study target when durable storage is supplied, not
    a count of additional trials on resume.  ``progress_path`` is atomically
    replaced after every terminal trial for external monitoring.

    The backwards-compatible default jointly searches structural parameters
    and the predeclared class-balance arms.  Production callers should pass
    ``structural_only_hpo=True``: every structural candidate is then fitted
    with uniform balance, and the required four-arm OOF comparison happens
    afterwards through :func:`sweep_purged_catboost_class_balance_arms`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if no_improvement_trials < 1:
        raise ValueError("no_improvement_trials must be positive")
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise ImportError("Optuna is required for CatBoost HPO") from exc
    encoded_target = _categorical_target(target, features.index, config=config)
    y = encoded_target.cat.codes.to_numpy()
    balance_contract = (
        catboost_structural_hpo_contract(config)
        if structural_only_hpo
        else catboost_class_balance_search_contract(config)
    )
    declared_arms = predeclared_catboost_class_balance_arms(config)
    arm_names = [str(arm["name"]) for arm in declared_arms]
    if (
        not structural_only_hpo
        and int(n_trials) < len(arm_names)
        and not allow_incomplete_class_balance_coverage
    ):
        raise ValueError(
            "production class-balance HPO requires at least one mandatory OOF "
            f"evaluation for every declared arm ({len(arm_names)} trials); "
            "set allow_incomplete_class_balance_coverage=True only for an "
            "explicit non-promotable smoke/diagnostic run"
        )
    balance_contract = {
        **balance_contract,
        "class_order": list(map(str, encoded_target.cat.categories)),
        "requested_total_trials": int(n_trials),
        "allow_incomplete_class_balance_coverage": bool(
            allow_incomplete_class_balance_coverage
        ),
        "structural_only_hpo": bool(structural_only_hpo),
    }
    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=0,
        interval_steps=1,
        n_min_trials=2,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(study_name and storage),
    )

    matched_params = (
        dict(
            balance_contract["mandatory_initial_evaluation"][
                "matched_non_arm_hpo_params"
            ]
        )
        if not structural_only_hpo
        else {}
    )
    mandatory_schedule = (
        [
            {
                "schedule_index": index,
                "arm": arm,
                "params": {**matched_params, "class_balance_arm": arm},
            }
            for index, arm in enumerate(arm_names)
        ]
        if not structural_only_hpo
        else []
    )

    def trial_params(trial: Any) -> dict[str, Any]:
        """Read fixed enqueued values before Optuna promotes a waiting trial."""

        fixed = trial.system_attrs.get("fixed_params", {})
        values = dict(fixed) if isinstance(fixed, Mapping) else {}
        values.update(dict(trial.params))
        return values

    def trial_phase(trial: Any) -> str | None:
        value = trial.user_attrs.get("class_balance_search_phase")
        return str(value) if value is not None else None

    def matching_scheduled_trials(spec: Mapping[str, Any]) -> list[Any]:
        return [
            trial
            for trial in study.trials
            if trial_phase(trial) == "mandatory_matched_baseline"
            and trial.user_attrs.get("class_balance_schedule_index")
            == spec["schedule_index"]
            and trial.user_attrs.get("class_balance_scheduled_arm") == spec["arm"]
        ]

    existing_trials = list(study.trials)
    allowed_phases = (
        {"structural_only_uniform_hpo"}
        if structural_only_hpo
        else {"mandatory_matched_baseline", "free_hpo"}
    )
    for trial in existing_trials:
        phase = trial_phase(trial)
        if phase not in allowed_phases:
            raise ValueError(
                "cannot safely resume this HPO mode without matching durable "
                "class-balance phase metadata; start a new versioned study"
            )
    for spec in mandatory_schedule:
        matches = matching_scheduled_trials(spec)
        if len(matches) > 1:
            raise ValueError("class-balance mandatory schedule contains duplicate arms")
        if not matches:
            continue
        actual = trial_params(matches[0])
        if any(actual.get(key) != value for key, value in spec["params"].items()):
            raise ValueError(
                "class-balance mandatory schedule does not retain matched "
                "hyperparameters"
            )
    free_trial_numbers = [
        int(trial.number)
        for trial in existing_trials
        if trial_phase(trial) == "free_hpo"
    ]
    missing_before_enqueue = [
        spec for spec in mandatory_schedule if not matching_scheduled_trials(spec)
    ]
    if free_trial_numbers and missing_before_enqueue:
        raise ValueError(
            "cannot safely resume free HPO before every mandatory class-balance "
            "arm was scheduled"
        )
    for spec in missing_before_enqueue:
        study.enqueue_trial(
            dict(spec["params"]),
            user_attrs={
                "class_balance_search_phase": "mandatory_matched_baseline",
                "class_balance_schedule_index": int(spec["schedule_index"]),
                "class_balance_scheduled_arm": str(spec["arm"]),
            },
        )
    all_scheduled_numbers = [
        int(matches[0].number)
        for spec in mandatory_schedule
        for matches in [matching_scheduled_trials(spec)]
        if matches
    ]
    if free_trial_numbers and (
        len(all_scheduled_numbers) != len(mandatory_schedule)
        or max(all_scheduled_numbers) >= min(free_trial_numbers)
    ):
        raise ValueError(
            "free HPO is not strictly sequenced after mandatory class-balance coverage"
        )

    def mandatory_coverage() -> dict[str, Any]:
        if structural_only_hpo:
            return {
                "coverage_gate_required": False,
                "scheduled_arm_order": [],
                "scheduled_records": [],
                "mandatory_initial_coverage_complete": False,
                "balance_arm_selection_performed": False,
                "unresolved_arms": [],
                "unrecoverable_arms": [],
            }
        records: list[dict[str, Any]] = []
        for spec in mandatory_schedule:
            matches = matching_scheduled_trials(spec)
            trial = matches[0] if matches else None
            state = str(trial.state.name) if trial is not None else "NOT_ENQUEUED"
            guard = (
                trial.user_attrs.get("class_balance_guard")
                if trial is not None
                else None
            )
            completed = (
                state == "COMPLETE"
                and isinstance(guard, Mapping)
                and bool(guard.get("passed", False))
            )
            explicit_rejection = (
                state == "PRUNED"
                and isinstance(guard, Mapping)
                and not bool(guard.get("passed", True))
            )
            records.append(
                {
                    "schedule_index": int(spec["schedule_index"]),
                    "arm": str(spec["arm"]),
                    "trial_number": int(trial.number) if trial is not None else None,
                    "state": state,
                    "matched_hyperparameters": dict(matched_params),
                    "guard": guard,
                    "completed_oof_guard_passed": completed,
                    "explicit_oof_guard_rejection": explicit_rejection,
                    "resolved": bool(completed or explicit_rejection),
                }
            )
        unresolved = [record for record in records if not record["resolved"]]
        unrecoverable = [
            record
            for record in unresolved
            if record["state"] in {"FAIL", "COMPLETE", "PRUNED"}
        ]
        return {
            "coverage_gate_required": True,
            "scheduled_arm_order": list(arm_names),
            "scheduled_records": records,
            "mandatory_initial_coverage_complete": not unresolved,
            "balance_arm_selection_performed": True,
            "unresolved_arms": [record["arm"] for record in unresolved],
            "unrecoverable_arms": [record["arm"] for record in unrecoverable],
        }

    started = time.perf_counter()
    prior_elapsed_seconds = 0.0
    if progress_path is not None and Path(progress_path).is_file():
        try:
            prior_progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
            prior_elapsed_seconds = float(prior_progress.get("elapsed_seconds", 0.0))
        except (OSError, ValueError, TypeError):
            prior_elapsed_seconds = 0.0
    best_oof_by_trial: dict[int, OOFPathArchetypeResult] = {}
    known_trials = [
        trial for trial in study.trials if str(trial.state.name) != "WAITING"
    ]
    state_counts = _hpo_state_counts(known_trials)
    best_number: int | None = None
    best_value: float | None = None
    best_params: dict[str, Any] | None = None
    last_improvement_position = -1
    for position, prior_trial in enumerate(known_trials):
        if (
            str(prior_trial.state.name) == "COMPLETE"
            and prior_trial.value is not None
            and (best_value is None or float(prior_trial.value) < best_value)
        ):
            best_number = int(prior_trial.number)
            best_value = float(prior_trial.value)
            best_params = dict(prior_trial.params)
            last_improvement_position = position

    def write_progress(*, current_trial: Any | None, status: str) -> None:
        if progress_path is None:
            return
        elapsed_seconds = prior_elapsed_seconds + (time.perf_counter() - started)
        latest_trial = current_trial or (known_trials[-1] if known_trials else None)
        _atomic_json_write(
            Path(progress_path),
            {
                "schema": "catboost_path_archetype_hpo_progress_v1",
                "status": status,
                "study_name": study.study_name,
                "storage": storage,
                "target_trials": int(n_trials),
                "elapsed_seconds": float(elapsed_seconds),
                "completed_trial_count": state_counts["completed"],
                "pruned_trial_count": state_counts["pruned"],
                "failed_trial_count": state_counts["failed"],
                "running_trial_count": state_counts["running"],
                "total_recorded_trial_count": int(len(study.trials)),
                "waiting_trial_count": int(
                    sum(str(trial.state.name) == "WAITING" for trial in study.trials)
                ),
                "current_trial": (
                    _hpo_trial_record(
                        latest_trial,
                        search_iterations=search_iterations,
                        search_od_wait=search_od_wait,
                        no_improvement_trials=no_improvement_trials,
                    )
                    if latest_trial is not None
                    else None
                ),
                "current_params": (
                    _json_ready(dict(latest_trial.params))
                    if latest_trial is not None
                    else None
                ),
                "best_trial_number": best_number,
                "best_objective": best_value,
                "best_params": _json_ready(best_params),
                "no_wall_clock_timeout": True,
                "class_balance_search_contract": {
                    **balance_contract,
                    "mandatory_initial_coverage": mandatory_coverage(),
                },
            },
        )

    write_progress(current_trial=None, status="running")

    def objective(trial: Any) -> float:
        phase = trial_phase(trial)
        if phase is None:
            phase = "structural_only_uniform_hpo" if structural_only_hpo else "free_hpo"
            trial.set_user_attr("class_balance_search_phase", phase)
        params = suggest_catboost_hpo_params(
            trial,
            config=config,
            structural_only_hpo=structural_only_hpo,
        )
        trial.set_user_attr("evaluated_class_balance_arm", params["class_balance_arm"])
        params["iterations"] = int(search_iterations)
        params["od_wait"] = int(search_od_wait)
        if phase == "mandatory_matched_baseline":
            expected_arm = trial.user_attrs.get("class_balance_scheduled_arm")
            expected_index = trial.user_attrs.get("class_balance_schedule_index")
            if (
                expected_arm not in arm_names
                or expected_index not in range(len(mandatory_schedule))
                or params["class_balance_arm"] != expected_arm
                or any(
                    params.get(key) != value for key, value in matched_params.items()
                )
            ):
                raise RuntimeError(
                    "mandatory class-balance baseline trial lost its fixed schedule"
                )

        def report_fold(
            step: int, partial_probabilities: np.ndarray, partial_fold_ids: np.ndarray
        ) -> None:
            partial_valid = partial_fold_ids >= 0
            if not np.any(partial_valid):
                return
            partial = catboost_hpo_objective_components(
                y[partial_valid],
                partial_probabilities[partial_valid],
                partial_fold_ids[partial_valid],
            )
            trial.report(float(partial["objective"]), step=int(step))
            # Every mandatory arm must reach the aggregate/per-fold guard.  A
            # median prune here would leave coverage unresolved and would make
            # the control comparison dependent on execution order.
            if phase != "mandatory_matched_baseline" and trial.should_prune():
                raise optuna.TrialPruned(f"median-pruned after purged fold {step + 1}")

        try:
            oof = fit_purged_chronological_oof_catboost(
                features,
                target,
                timestamps,
                label_end=label_end,
                config=config,
                params=params,
                fold_callback=report_fold,
            )
            valid = oof.fold_ids >= 0
            if not np.any(valid):
                raise CatBoostClassBalanceError(
                    "class-balance OOF guard found no validation predictions"
                )
            components = catboost_hpo_objective_components(
                y[valid], oof.probabilities[valid], oof.fold_ids[valid]
            )
            balance_guard = class_balance_oof_guard(
                y[valid],
                oof.probabilities[valid],
                classes=oof.classes,
                fold_ids=oof.fold_ids[valid],
                config=config,
            )
        except CatBoostClassBalanceError as exc:
            # An unsafe arm is deliberately not assigned a finite objective.
            # This makes the search fail closed without allowing one bad arm to
            # terminate evaluation of the remaining declared controls.
            trial.set_user_attr(
                "class_balance_guard",
                {
                    "passed": False,
                    "evaluation_scope": (
                        "frozen_purged_oof_validation_rows_only_aggregate_and_per_fold"
                    ),
                    "final_refit_used_for_selection": False,
                    "rejection_reason": str(exc),
                },
            )
            raise optuna.TrialPruned(str(exc)) from exc
        trial.set_user_attr("oof_diagnostics", _json_ready(oof.diagnostics))
        trial.set_user_attr("objective_components", _json_ready(components))
        trial.set_user_attr("class_balance_guard", balance_guard)
        best_oof_by_trial[int(trial.number)] = oof
        return float(components["objective"])

    completed_before = int(
        sum(
            str(trial.state.name) in {"COMPLETE", "PRUNED", "FAIL"}
            for trial in study.trials
        )
    )

    def stop_after_stagnation(study: Any, trial: Any) -> None:
        """Stop after a bounded number of trials without a new best value."""
        nonlocal best_number, best_value, best_params, last_improvement_position
        known_trials.append(trial)
        trial_state = str(trial.state.name)
        if trial_state == "COMPLETE":
            state_counts["completed"] += 1
        elif trial_state == "PRUNED":
            state_counts["pruned"] += 1
        elif trial_state == "FAIL":
            state_counts["failed"] += 1
        elif trial_state == "RUNNING":
            state_counts["running"] += 1
        if (
            trial_state == "COMPLETE"
            and trial.value is not None
            and (best_value is None or float(trial.value) < best_value)
        ):
            best_number = int(trial.number)
            best_value = float(trial.value)
            best_params = dict(trial.params)
            last_improvement_position = len(known_trials) - 1
        elapsed_seconds = prior_elapsed_seconds + (time.perf_counter() - started)
        write_progress(current_trial=trial, status="running")
        LOGGER.info(
            "CatBoost HPO trial %s %s value=%s; best=%s; complete=%s pruned=%s failed=%s elapsed=%.1fs",
            trial.number,
            trial.state.name,
            trial.value,
            best_value,
            state_counts["completed"],
            state_counts["pruned"],
            state_counts["failed"],
            elapsed_seconds,
        )
        if not structural_only_hpo:
            coverage = mandatory_coverage()
            if coverage["unrecoverable_arms"]:
                study.stop()
                return
            # Never halt by value-stagnation before every baseline arm reaches
            # its full OOF guard (or is explicitly ruled out by that guard).
            if not coverage["mandatory_initial_coverage_complete"]:
                return
        if last_improvement_position >= 0 and len(
            known_trials
        ) - 1 - last_improvement_position >= int(no_improvement_trials):
            study.stop()

    remaining_trials = max(0, int(n_trials) - completed_before)
    study.optimize(
        objective,
        n_trials=remaining_trials,
        callbacks=[stop_after_stagnation],
    )
    coverage = mandatory_coverage()
    if not structural_only_hpo and not coverage["mandatory_initial_coverage_complete"]:
        write_progress(current_trial=None, status="coverage_incomplete")
        if not allow_incomplete_class_balance_coverage:
            raise RuntimeError(
                "CatBoost HPO completed without mandatory class-balance arm "
                "coverage: " + ", ".join(coverage["unresolved_arms"])
            )
    else:
        write_progress(current_trial=None, status="complete")
    if not any(str(trial.state.name) == "COMPLETE" for trial in study.trials):
        raise RuntimeError("CatBoost HPO completed without a successful trial")
    best_params = suggest_catboost_hpo_params(
        _FixedTrial(study.best_trial.params),
        config=config,
        structural_only_hpo=structural_only_hpo,
    )
    best_params["iterations"] = int(search_iterations)
    best_params["od_wait"] = int(search_od_wait)
    coverage_complete = (
        bool(coverage["mandatory_initial_coverage_complete"])
        if not structural_only_hpo
        else False
    )
    if not structural_only_hpo:
        best_params["class_balance_selection_provenance"] = {
            "schema": CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA,
            "arm": best_params["class_balance_arm"],
            "class_order": list(map(str, encoded_target.cat.categories)),
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "mandatory_initial_coverage_complete": coverage_complete,
            "promotion_eligible": coverage_complete,
            "selection_status": (
                "coverage_complete"
                if coverage_complete
                else "coverage_incomplete_non_promotable"
            ),
        }
    best_oof = best_oof_by_trial.get(int(study.best_trial.number))
    reused_best_oof = best_oof is not None
    if best_oof is None:
        best_oof = fit_purged_chronological_oof_catboost(
            features,
            target,
            timestamps,
            label_end=label_end,
            config=config,
            params=best_params,
        )
    trials = tuple(
        _hpo_trial_record(
            trial,
            search_iterations=search_iterations,
            search_od_wait=search_od_wait,
            no_improvement_trials=no_improvement_trials,
        )
        for trial in study.trials
    )

    def evaluated_balance_arm(trial: Any) -> Any:
        """Recover the fixed uniform arm omitted from structural trial params."""

        return trial_params(trial).get(
            "class_balance_arm",
            (CATBOOST_CLASS_BALANCE_ARM_UNIFORM if structural_only_hpo else None),
        )

    arm_trial_counts = {
        arm: int(
            sum(1 for trial in study.trials if evaluated_balance_arm(trial) == arm)
        )
        for arm in arm_names
    }
    arm_completed_counts = {
        arm: int(
            sum(
                1
                for trial in study.trials
                if evaluated_balance_arm(trial) == arm
                and str(trial.state.name) == "COMPLETE"
            )
        )
        for arm in arm_names
    }
    arm_guard_rejections = {
        arm: int(
            sum(
                1
                for trial in study.trials
                if evaluated_balance_arm(trial) == arm
                and isinstance(trial.user_attrs.get("class_balance_guard"), Mapping)
                and not bool(
                    trial.user_attrs["class_balance_guard"].get("passed", False)
                )
            )
        )
        for arm in arm_names
    }
    class_balance_search = _json_ready(
        {
            **balance_contract,
            "arm_trial_counts": arm_trial_counts,
            "arm_completed_counts": arm_completed_counts,
            "arm_guard_rejections": arm_guard_rejections,
            "total_hpo_trial_count": int(len(study.trials)),
            "mandatory_initial_coverage": coverage,
            "mandatory_initial_coverage_complete": coverage_complete,
            "balance_arm_selection_complete": (
                not structural_only_hpo and coverage_complete
            ),
            "promotion_eligible": (not structural_only_hpo and coverage_complete),
            "selected_arm": (
                best_params["class_balance_arm"]
                if not structural_only_hpo and coverage_complete
                else None
            ),
            "provisional_arm": (
                None
                if structural_only_hpo or coverage_complete
                else best_params["class_balance_arm"]
            ),
            "fixed_training_arm": (
                CATBOOST_CLASS_BALANCE_ARM_UNIFORM if structural_only_hpo else None
            ),
            "post_hpo_mini_sweep_required": bool(structural_only_hpo),
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "class_order": list(map(str, encoded_target.cat.categories)),
        }
    )
    return CatBoostHPOResult(
        best_params,
        float(study.best_value),
        best_oof,
        trials,
        study.study_name,
        reused_best_oof,
        class_balance_search,
    )


class _FixedTrial:
    """Minimal Optuna trial adapter used to reconstruct fixed best parameters."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = params

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(self.params[name])

    def suggest_float(
        self, name: str, low: float, high: float, **_kwargs: Any
    ) -> float:
        return float(self.params[name])

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        return self.params.get(name, choices[0])


def _permutation_orders(
    columns: Sequence[str], rows: int, *, random_state: int, fold_id: int
) -> dict[str, np.ndarray]:
    """Generate the historical one-per-column RNG sequence once per fold."""
    rng = np.random.default_rng(random_state + fold_id)
    return {column: rng.permutation(rows) for column in columns}


def _batched_permutation_losses(
    model: Any,
    matrix: np.ndarray,
    target: np.ndarray,
    columns: Sequence[str],
    candidate_columns: Sequence[str],
    orders: Mapping[str, np.ndarray],
    baseline: float,
    *,
    max_batch_bytes: int,
) -> tuple[dict[str, float], int, int]:
    """Score one-feature permutations in bounded prediction batches.

    Every batch contains the same matrices the prior loop predicted one at a
    time.  Only the CatBoost prediction call is batched; RNG ordering and each
    per-feature replacement remain unchanged.
    """
    candidates = list(candidate_columns)
    if not candidates:
        return {}, 0, 0
    index = {column: position for position, column in enumerate(columns)}
    per_matrix_bytes = max(1, int(matrix.nbytes))
    batch_size = min(len(candidates), max(1, int(max_batch_bytes) // per_matrix_bytes))
    losses: dict[str, float] = {}
    predict_calls = 0
    for start in range(0, len(candidates), batch_size):
        batch_columns = candidates[start : start + batch_size]
        # This is the only intentionally materialized batch.  Its allocation
        # is bounded by permutation_batch_max_bytes before CatBoost receives it.
        batch = np.repeat(matrix[None, :, :], len(batch_columns), axis=0)
        for offset, column in enumerate(batch_columns):
            column_index = index[column]
            batch[offset, :, column_index] = matrix[orders[column], column_index]
        probabilities = model.predict_proba(batch.reshape(-1, matrix.shape[1]))
        predict_calls += 1
        probabilities = np.asarray(probabilities).reshape(
            len(batch_columns), len(matrix), -1
        )
        classes = getattr(model, "classes_", None)
        for offset, column in enumerate(batch_columns):
            loss = multiclass_log_loss(target, probabilities[offset], classes)
            losses[column] = float(loss - baseline)
    return losses, predict_calls, batch_size


def _screened_mda_candidates(
    columns: Sequence[str],
    mandatory: Sequence[str],
    screen_losses: Mapping[str, float],
    *,
    keep: int,
    config: PathArchetypeConfig,
) -> tuple[list[str], bool, float | None]:
    """Select a deterministic full-MDA shortlist around the stage cutoff."""
    nonmandatory = [column for column in columns if column not in set(mandatory)]
    minimum = max(0, keep - len(mandatory))
    candidate_limit = min(
        len(nonmandatory), minimum + max(0, int(config.permutation_screen_margin))
    )
    if not config.permutation_screening_enabled or candidate_limit >= len(nonmandatory):
        return list(columns), False, None
    index = {column: position for position, column in enumerate(columns)}
    ranked = sorted(
        nonmandatory,
        key=lambda column: (-float(screen_losses[column]), index[column]),
    )
    candidates = list(dict.fromkeys([*mandatory, *ranked[:candidate_limit]]))
    cutoff = (
        float(screen_losses[ranked[candidate_limit - 1]]) if candidate_limit else None
    )
    return candidates, True, cutoff


def staged_permutation_selection(
    features: pd.DataFrame,
    target: Sequence[Any],
    oof: OOFPathArchetypeResult,
    *,
    mandatory_features: Sequence[str] = (),
    stages: Sequence[int] = (150, 125, 100, 75),
    random_state: int = 20260722,
    config: PathArchetypeConfig | None = None,
    params: Mapping[str, Any] | None = None,
    completed_stages: Sequence[Mapping[str, Any]] = (),
    stage_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Run staged, purged permutation MDA with audited bounded acceleration.

    Every fully evaluated candidate retains the original multi-fold MDA score,
    stability, and drift contract.  When screening is enabled, deterministic
    first-and-last-fold permutations retain features with evidence at either
    endpoint before full multi-fold evaluation.  The report makes this
    approximation explicit per stage; mandatory features always bypass it.
    """
    columns = list(validate_preentry_features(features.columns))
    config = config or PathArchetypeConfig(random_state=random_state)
    y = _categorical_target(target, features.index, config=config).cat.codes.to_numpy()
    mandatory = list(validate_preentry_features(mandatory_features))
    missing = set(mandatory).difference(columns)
    if missing:
        raise KeyError(f"Mandatory features missing from frame: {sorted(missing)}")
    current = list(staged_permutation_feature_order(columns, mandatory))
    rows: list[dict[str, Any]] = []
    if not oof.folds:
        raise ValueError("No fitted OOF models; cannot run permutation selection")
    cache = oof.staged_matrix_cache
    if cache is None:
        cache = StagedPermutationMatrixCache.from_frame(
            features,
            oof.folds,
            validation_cache_max_bytes=config.permutation_validation_cache_max_bytes,
        )
    if not cache.supports(columns):
        missing_cache_columns = sorted(set(columns).difference(cache.column_index))
        raise KeyError(
            f"Staged MDA cache is missing features: {missing_cache_columns[:8]}"
        )
    CatBoostClassifier = _require_catboost()
    reused_models = {fold.fold_id: model for fold, model in zip(oof.folds, oof.models)}

    completed = list(completed_stages)
    if len(completed) > len(stages):
        raise ValueError("MDA checkpoint has more completed stages than requested")
    for stage_index, stage in enumerate(stages):
        if stage_index < len(completed):
            checkpoint = completed[stage_index]
            if int(checkpoint.get("stage_index", -1)) != stage_index or int(
                checkpoint.get("stage", -1)
            ) != int(stage):
                raise ValueError(
                    "MDA checkpoint stages do not match the current contract"
                )
            input_features = list(map(str, checkpoint.get("input_features", ())))
            selected_features = list(map(str, checkpoint.get("selected_features", ())))
            records = checkpoint.get("records")
            if (
                input_features != current
                or not selected_features
                or not isinstance(records, list)
            ):
                raise ValueError("MDA checkpoint has invalid stage state")
            current = selected_features
            rows.extend(records)
            LOGGER.info(
                "CatBoost MDA resume: stage %s/%s (%s) already complete; %s features retained",
                stage_index + 1,
                len(stages),
                stage,
                len(current),
            )
            continue
        stage_started = time.perf_counter()
        stage_input_columns = list(current)
        keep = max(len(mandatory), min(int(stage), len(current)))
        can_reuse = (
            stage_index == 0
            and tuple(current) == oof.feature_columns
            and oof.staged_matrix_cache is cache
            and len(reused_models) == len(oof.folds)
        )
        stage_models: list[tuple[PurgedFold, Any, np.ndarray, float]] = []
        fit_calls = 0
        baseline_predict_calls = 0
        fit_seconds = 0.0
        baseline_seconds = 0.0
        stage_fit_reports: list[dict[str, Any]] = []
        for fold in oof.folds:
            valid = fold.validation_indices
            if not len(valid):
                continue
            matrix = cache.matrix(fold, current, training=False)
            if can_reuse:
                model = reused_models[fold.fold_id]
                baseline = multiclass_log_loss(
                    y[valid], oof.probabilities[valid], np.arange(len(oof.classes))
                )
                prior_reports = (oof.diagnostics or {}).get("fold_fit_reports", [])
                prior_report = next(
                    (
                        dict(report)
                        for report in prior_reports
                        if int(report.get("fold_id", -1)) == int(fold.fold_id)
                    ),
                    None,
                )
                if prior_report is not None:
                    stage_fit_reports.append(prior_report)
            else:
                train_matrix = cache.matrix(fold, current, training=True)
                fold_params = _catboost_params(config, params)
                fold_params["use_best_model"] = True
                model = CatBoostClassifier(**fold_params)
                fit_started = time.perf_counter()
                model.fit(
                    train_matrix,
                    y[fold.train_indices],
                    eval_set=(matrix, y[valid]),
                    early_stopping_rounds=int(fold_params["od_wait"]),
                    verbose=False,
                )
                fit_seconds += time.perf_counter() - fit_started
                fit_calls += 1
                baseline_started = time.perf_counter()
                baseline = multiclass_log_loss(
                    y[valid],
                    model.predict_proba(matrix),
                    getattr(model, "classes_", None),
                )
                baseline_seconds += time.perf_counter() - baseline_started
                baseline_predict_calls += 1
                best_iteration = None
                if hasattr(model, "get_best_iteration"):
                    value = model.get_best_iteration()
                    best_iteration = int(value) if value is not None else None
                tree_count = getattr(model, "tree_count_", None)
                stage_fit_reports.append(
                    {
                        "fold_id": int(fold.fold_id),
                        "train_rows": int(len(fold.train_indices)),
                        "validation_rows": int(len(fold.validation_indices)),
                        "use_best_model": True,
                        "eval_set_used": True,
                        "early_stopping_rounds": int(fold_params["od_wait"]),
                        "best_iteration": best_iteration,
                        "tree_count": int(tree_count)
                        if tree_count is not None
                        else None,
                    }
                )
            stage_models.append((fold, model, matrix, baseline))
        if not stage_models:
            raise ValueError("No usable OOF folds; cannot run permutation selection")

        orders = {
            fold.fold_id: _permutation_orders(
                current, len(matrix), random_state=random_state, fold_id=fold.fold_id
            )
            for fold, _model, matrix, _baseline in stage_models
        }
        screen_seconds = 0.0
        screen_predict_calls = 0
        screen_batch_size = 0
        screen_losses: dict[str, float] = {}
        screen_losses_by_fold: dict[int, dict[str, float]] = {}
        screen_models = [stage_models[0]]
        if stage_models[-1][0].fold_id != stage_models[0][0].fold_id:
            screen_models.append(stage_models[-1])
        screen_fold_ids = [int(fold.fold_id) for fold, *_rest in screen_models]
        screen_aggregation = (
            "max_first_last_loss_conservative_retention"
            if len(screen_models) > 1
            else "single_fold_loss"
        )
        # The selector skips screening only when it cannot eliminate a feature;
        # that path is exact full MDA and carries no approximation claim.
        potential_screen = config.permutation_screening_enabled and len(
            current
        ) > keep + max(0, int(config.permutation_screen_margin))
        if potential_screen:
            screen_started = time.perf_counter()
            for fold, model, matrix, baseline in screen_models:
                fold_losses, calls, batch_size = _batched_permutation_losses(
                    model,
                    matrix,
                    y[fold.validation_indices],
                    current,
                    current,
                    orders[fold.fold_id],
                    baseline,
                    max_batch_bytes=config.permutation_batch_max_bytes,
                )
                screen_losses_by_fold[fold.fold_id] = fold_losses
                screen_predict_calls += calls
                screen_batch_size = max(screen_batch_size, batch_size)
            # Candidate retention is intentionally conservative: a feature is
            # screened out only if it is weak on *both* temporal endpoints.
            screen_losses = {
                column: max(
                    fold_losses[column]
                    for fold_losses in screen_losses_by_fold.values()
                )
                for column in current
            }
            screen_seconds = time.perf_counter() - screen_started
        full_candidates, screened, screen_cutoff = _screened_mda_candidates(
            current,
            mandatory,
            screen_losses,
            keep=keep,
            config=config,
        )
        scores: dict[str, list[float]] = {column: [] for column in full_candidates}
        permutation_predict_calls = screen_predict_calls
        permutation_seconds = 0.0
        max_batch_size = screen_batch_size
        for fold, model, matrix, baseline in stage_models:
            if screened and fold.fold_id in screen_losses_by_fold:
                fold_losses = {
                    column: screen_losses_by_fold[fold.fold_id][column]
                    for column in full_candidates
                }
            else:
                permutation_started = time.perf_counter()
                fold_losses, calls, batch_size = _batched_permutation_losses(
                    model,
                    matrix,
                    y[fold.validation_indices],
                    current,
                    full_candidates,
                    orders[fold.fold_id],
                    baseline,
                    max_batch_bytes=config.permutation_batch_max_bytes,
                )
                permutation_seconds += time.perf_counter() - permutation_started
                permutation_predict_calls += calls
                max_batch_size = max(max_batch_size, batch_size)
            for column, loss in fold_losses.items():
                scores[column].append(loss)

        mean_loss = {
            column: float(np.mean(values)) if values else float("nan")
            for column, values in scores.items()
        }
        stability = {
            column: (
                1.0
                / (1.0 + float(np.std(values)) / (abs(float(np.mean(values))) + 1e-6))
                if values
                else float("nan")
            )
            for column, values in scores.items()
        }
        drift = {
            column: abs(values[-1] - values[0]) if len(values) > 1 else 0.0
            for column, values in scores.items()
        }
        a, b, c = _normalise(mean_loss), _normalise(stability), _normalise(drift)
        combined = {
            column: 0.5 * a[column] + 0.3 * b[column] - 0.2 * c[column]
            for column in full_candidates
        }
        current = list(
            dict.fromkeys(
                mandatory
                + sorted(
                    (column for column in full_candidates if column not in mandatory),
                    key=combined.get,
                    reverse=True,
                )[: max(0, keep - len(mandatory))]
            )
        )
        stage_total_seconds = time.perf_counter() - stage_started
        stage_fields = {
            "stage_acceleration_algorithm_version": STAGED_PERMUTATION_ACCELERATION_VERSION,
            "stage_input_feature_count": int(len(stage_input_columns)),
            "stage_keep_count": int(keep),
            "stage_full_mda_candidate_count": int(len(full_candidates)),
            "stage_screened_out_count": int(
                len(stage_input_columns) - len(full_candidates)
            ),
            "stage_screening_used": bool(screened),
            "stage_screen_fold_ids": screen_fold_ids if screened else [],
            "stage_screen_fold_count": int(len(screen_models) if screened else 0),
            "stage_screen_aggregation": screen_aggregation if screened else None,
            "stage_screen_cutoff_loss": screen_cutoff,
            "stage_selection_semantics": (
                "deterministic_screened_mda_approximation"
                if screened
                else "exact_full_mda"
            ),
            "stage_reused_oof_models": bool(can_reuse),
            "stage_fit_calls": int(fit_calls),
            "stage_baseline_predict_calls": int(baseline_predict_calls),
            "stage_permutation_predict_calls": int(permutation_predict_calls),
            "stage_max_permutation_batch_size": int(max_batch_size),
            "stage_matrix_cache_bytes": int(cache.cache_bytes),
            "stage_validation_matrix_cache_bytes": int(
                sum(matrix.nbytes for matrix in cache.validation_matrices.values())
            ),
            "stage_validation_matrix_cache_used": bool(cache.validation_matrices),
            "stage_matrix_dtype": str(cache.values.dtype),
            "stage_fit_seconds": float(fit_seconds),
            "stage_fold_fit_reports": stage_fit_reports,
            "stage_baseline_predict_seconds": float(baseline_seconds),
            "stage_screen_seconds": float(screen_seconds),
            "stage_permutation_predict_seconds": float(permutation_seconds),
            "stage_total_seconds": float(stage_total_seconds),
        }
        evaluated = set(full_candidates)
        stage_records = [
            {
                "stage": int(stage),
                "feature": column,
                "loss_increase": mean_loss.get(column, float("nan")),
                "stability": stability.get(column, float("nan")),
                "drift_instability": drift.get(column, float("nan")),
                "score": combined.get(column, float("nan")),
                "screen_loss_increase": screen_losses.get(column, float("nan")),
                "full_mda_evaluated": column in evaluated,
                "selected": column in current,
                **stage_fields,
            }
            for column in stage_input_columns
        ]
        rows.extend(stage_records)
        checkpoint = {
            "stage_index": int(stage_index),
            "stage": int(stage),
            "input_features": list(stage_input_columns),
            "selected_features": list(current),
            "records": _json_ready(stage_records),
            "stage_total_seconds": float(stage_total_seconds),
        }
        LOGGER.info(
            "CatBoost MDA stage %s/%s (%s) complete: %s -> %s features in %.1fs",
            stage_index + 1,
            len(stages),
            stage,
            len(stage_input_columns),
            len(current),
            stage_total_seconds,
        )
        if stage_callback is not None:
            stage_callback(checkpoint)
    return current, pd.DataFrame(rows)


@dataclass
class PathArchetypeClassifier:
    """Frozen multiclass CatBoost classifier whose inputs are pre-entry only."""

    feature_columns: tuple[str, ...]
    class_names: tuple[str, ...]
    model: Any
    selector: FastSelectorResult | None = None
    config: PathArchetypeConfig = field(default_factory=PathArchetypeConfig)
    training_report: Mapping[str, Any] | None = None

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        path_archetypes: Sequence[Any],
        timestamps: Sequence[Any],
        *,
        label_end: Sequence[Any] | None = None,
        mandatory_features: Sequence[str] = (),
        config: PathArchetypeConfig = PathArchetypeConfig(),
        params: Mapping[str, Any] | None = None,
        run_permutation_selection: bool = False,
        warmup_mask: Sequence[bool] | None = None,
        config_mapping: Mapping[str, Any] | None = None,
        run_hpo_trials: int = 0,
        hpo_study_name: str | None = None,
        hpo_storage: str | None = None,
    ) -> "PathArchetypeClassifier":
        validate_preentry_features(features.columns)
        universe = configured_base_meta_preselection_universe(
            features.columns,
            config_mapping=config_mapping,
        )
        if not universe:
            raise ValueError(
                "No configured base/meta preselection features are present in the training frame"
            )
        mandatory = tuple(
            feature for feature in mandatory_features if feature in universe
        )
        missing_mandatory = set(mandatory_features).difference(universe)
        if missing_mandatory:
            raise ValueError(
                "Mandatory path-archetype features must belong to the configured base/meta universe: "
                + ", ".join(sorted(missing_mandatory))
            )
        selector = fast_select_preentry_features(
            features.loc[:, universe],
            path_archetypes,
            mandatory_features=mandatory,
            warmup_mask=warmup_mask,
            config=config,
        )
        selected = list(selector.selected_features)
        hpo_result: CatBoostHPOResult | None = None
        fitted_params = dict(params or {})
        mda_columns = staged_permutation_feature_order(selected, mandatory)
        mda_features = features.loc[:, mda_columns]
        mda_cache = (
            build_staged_permutation_matrix_cache(
                mda_features,
                timestamps,
                label_end=label_end,
                config=config,
            )
            if run_permutation_selection
            else None
        )
        selection_oof_features = (
            mda_features if run_permutation_selection else features.loc[:, selected]
        )
        selection_oof = fit_purged_chronological_oof_catboost(
            selection_oof_features,
            path_archetypes,
            timestamps,
            label_end=label_end,
            config=config,
            params=fitted_params,
            staged_matrix_cache=mda_cache,
            force_classes_count=not run_permutation_selection,
        )
        permutation_report: list[dict[str, Any]] | None = None
        if run_permutation_selection:
            selected, permutation = staged_permutation_selection(
                mda_features,
                path_archetypes,
                selection_oof,
                mandatory_features=mandatory,
                stages=config.permutation_stages,
                random_state=config.random_state,
                config=config,
                params=fitted_params,
            )
            permutation_report = _json_ready(permutation.to_dict(orient="records"))
        if run_hpo_trials:
            hpo_result = optimize_purged_catboost_hpo(
                features.loc[:, selected],
                path_archetypes,
                timestamps,
                label_end=label_end,
                config=config,
                n_trials=run_hpo_trials,
                study_name=hpo_study_name,
                storage=hpo_storage,
            )
            fitted_params.update(hpo_result.best_params)
        oof = fit_purged_chronological_oof_catboost(
            features.loc[:, selected],
            path_archetypes,
            timestamps,
            label_end=label_end,
            config=config,
            params=fitted_params,
        )
        CatBoostClassifier = _require_catboost()
        y_series = _categorical_target(path_archetypes, features.index, config=config)
        if "class_balance_selection_provenance" in fitted_params:
            fitted_params = rematerialize_final_class_balance_params(
                fitted_params, path_archetypes, config=config
            )
        final_params = _catboost_params(config, fitted_params)
        final_params["use_best_model"] = False
        model = CatBoostClassifier(**final_params)
        model.fit(_finite_matrix(features, selected), y_series.cat.codes.to_numpy())
        training_phase_order = ["fast_feature_selection"]
        if run_permutation_selection:
            training_phase_order.append("permutation_feature_selection")
        if run_hpo_trials:
            training_phase_order.append("hpo_on_frozen_selected_features")
        training_phase_order.append("final_oof_and_refit")
        report = _json_ready(
            {
                "configured_universe": list(universe),
                "selected_features": selected,
                "training_phase_order": training_phase_order,
                "hpo_feature_count": int(len(selected)) if run_hpo_trials else None,
                "selector_backend": selector.proxy_backend,
                "permutation_acceleration_contract": staged_permutation_acceleration_contract(
                    config
                ),
                "selection_oof_diagnostics": selection_oof.diagnostics,
                "oof_diagnostics": oof.diagnostics,
                "hpo": hpo_result.report() if hpo_result is not None else None,
                "permutation_stages": permutation_report,
            }
        )
        return cls(
            tuple(selected),
            tuple(map(str, y_series.cat.categories)),
            model,
            selector,
            config,
            report,
        )

    def predict_proba(self, preentry_features: pd.DataFrame) -> pd.DataFrame:
        validate_preentry_features(preentry_features.columns)
        missing = set(self.feature_columns).difference(preentry_features.columns)
        if missing:
            raise KeyError(f"Missing frozen classifier features: {sorted(missing)}")
        values = np.asarray(
            self.model.predict_proba(
                _finite_matrix(preentry_features, self.feature_columns)
            ),
            dtype=float,
        )
        if tuple(self.class_names) == MERGED_PATH_ARCHETYPE_CLASSES:
            model_classes = np.asarray(
                getattr(self.model, "classes_", np.arange(values.shape[1])),
                dtype=object,
            )
            aligned = np.zeros((len(values), len(self.class_names)), dtype=float)
            numeric_classes = (
                pd.to_numeric(pd.Series(model_classes), errors="coerce").notna().all()
            )
            if numeric_classes:
                aligned[:, np.asarray(model_classes, dtype=int)] = values
            else:
                positions = {
                    name: position for position, name in enumerate(self.class_names)
                }
                for source, name in enumerate(model_classes.astype(str)):
                    aligned[:, positions[name]] = values[:, source]
            return path_archetype_probability_contract(
                aligned, self.class_names, index=preentry_features.index
            )
        return pd.DataFrame(
            values, index=preentry_features.index, columns=self.class_names
        )

    def predict(self, preentry_features: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(preentry_features)
        return (
            proba.loc[:, list(self.class_names)].idxmax(axis=1).rename("path_archetype")
        )
