#!/usr/bin/env python3
"""
CLI entry point for extreme_price_movements pipeline.

Usage:
    python3 extreme_price_movements/run_pipeline.py labels
"""
import os
import sys
import warnings
import json
import pickle
import re
from pathlib import Path

# Avoid expensive/warning-prone Matplotlib cache initialization under read-only HOME.
_mpl_cfg = os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_epm")
os.environ.setdefault("MPLBACKEND", "Agg")
#
# Keep native thread pools conservative on Apple Silicon. The older meta-training
# stack mixes Arrow, OpenBLAS, OpenMP/joblib and Python worker pools; unrestricted
# defaults have repeatedly produced EXC_BAD_ACCESS / SIGSEGV crashes under load.
# Native LGBM training is different: capping OMP_THREAD_LIMIT at 1 silently forces
# LightGBM's n_jobs>1 fits back to one OpenMP thread. Users can still override any
# of these via the environment.
_is_lgbm_pipeline_run = (
    str(os.environ.get("EPM_MODEL_BACKEND", "")).strip().lower() == "lgbm_pipeline"
    or "lgbm_pipeline" in {str(arg).strip().lower() for arg in sys.argv}
    or "recency_hpo" in {str(arg).strip().lower() for arg in sys.argv}
    or "final_model_fit" in {str(arg).strip().lower() for arg in sys.argv}
)
try:
    _lgbm_threads = str(max(1, int(os.environ.get("EPM_LGBM_N_JOBS", "3") or "3")))
except Exception:
    _lgbm_threads = "3"
_default_omp_threads = _lgbm_threads if _is_lgbm_pipeline_run else "1"
for _thread_env, _default in (
    ("OMP_NUM_THREADS", _default_omp_threads),
    ("OMP_THREAD_LIMIT", _default_omp_threads),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("ARROW_NUM_THREADS", "1"),
    ("POLARS_MAX_THREADS", "1"),
):
    os.environ.setdefault(_thread_env, _default)
_loky_cpu = str(os.environ.get("LOKY_MAX_CPU_COUNT", "")).strip()
if not _loky_cpu.isdigit():
    os.environ["LOKY_MAX_CPU_COUNT"] = str(min(os.cpu_count() or 1, 2))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores for the following reason:",
    category=UserWarning,
)
try:
    os.makedirs(_mpl_cfg, exist_ok=True)
except Exception:
    pass

_is_strategy_selection_run = "strategies_selection" in {
    str(arg).strip().lower() for arg in sys.argv
}


def _set_strategy_selection_lgbm_defaults() -> None:
    """Set faster LGBM feature-selection defaults before lgbm_pipeline imports."""
    if not _is_strategy_selection_run:
        return
    defaults = {
        "EPM_LGBM_CV_SPLITS": "2",
        "EPM_LGBM_RACE_MAX_ROWS": "30000",
        "EPM_LGBM_UNIVARIATE_MAX_ROWS": "6000",
        "EPM_LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC": "0.35",
        "EPM_LGBM_RELIEF_REPEATS": "1",
        "EPM_LGBM_RELIEF_RESCUE_MAX": "30",
        "EPM_LGBM_RELIEF_RESCUE_MIN": "8",
        "EPM_LGBM_RELIEF_ANCHOR_MAX_ROWS": "256",
        "EPM_LGBM_RELIEF_NEIGHBOR_CANDIDATES": "768",
        "EPM_LGBM_DIRECTION_MAX_ROWS": "1500",
        "EPM_LGBM_MAX_ROUNDS": "6",
        "EPM_LGBM_STABILITY_CONFIGS": "2",
        "EPM_LGBM_MIN_FEATURES": "24",
        "EPM_LGBM_SELECTED_FEATURES_MIN": "40",
        "EPM_LGBM_SELECTED_FEATURES_MAX": "120",
        "EPM_LGBM_PERMUTATION_TOP_CONFIGS": "1",
        "EPM_LGBM_PERMUTATION_MAX_FEATURES": "20",
        "EPM_LGBM_PERMUTATION_MAX_ROWS": "5000",
        "EPM_LGBM_PERMUTATION_REPEATS": "1",
        "EPM_LGBM_HPO_TRIALS": "0",
        "EPM_LGBM_HPO_MAX_ROWS": "3000",
        "EPM_LGBM_FINAL_MODEL_COUNT": "1",
        "EPM_LGBM_OOF_DISTILLATION_PASSES": "0",
        "EPM_LGBM_MIN_OOF_DISTILLATION_PASSES": "2",
        "EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES": "2",
    }
    for key, value in defaults.items():
        override_key = f"EPM_STRATEGY_SELECTION_{key.removeprefix('EPM_')}"
        os.environ.setdefault(key, os.environ.get(override_key, value))


_set_strategy_selection_lgbm_defaults()

# Add parent directory to Python path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import extreme_price_movements.mask_optimiser as mask_opt
from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.data_store import (
    _compute_missing_funding_ranges,
    _compute_missing_hourly_ranges,
    _fetch_ccxt_history_paged,
    _has_sparse_perp_auxiliary_data,
    _resolve_perp_symbol,
    build_hourly_orderbook_proxy_from_ohlcv,
    fetch_hourly_orderbook_proxy,
    exchange_data_component,
    make_perp_exchange,
    make_ohlcv_store,
    make_spot_exchange,
    normalize_orderbook_proxy_frame,
    scoped_data_root,
)
from extreme_price_movements.offline_optimisers.params_store import (
    apply_offline_optimizer_best_params,
    load_inference_candidate_mask_params_per_bucket,
)
from extreme_price_movements.optimise import (
    Policy,
    run_optimise_from_ridge_oof,
    run_optimise_step,
)
from extreme_price_movements.path_utils import mode_file_candidates, resolve_mode_file
from extreme_price_movements.pipeline_steps import (
    run_backtest_step,
    run_base_hpo_step,
    run_feature_generation_step,
    run_label_generation_step_v2,
    run_risk_optimization_step,
    run_training_step,
)
from extreme_price_movements.policy_optimiser import run_policy_optimisation
from extreme_price_movements.simple_position_sizer import (
    run_simple_position_sizer_from_artifacts,
    write_holdout_multi_metrics,
)
from extreme_price_movements.slice_plan_store import (
    apply_stage_usage_limits,
    load_or_build_slice_plan,
)
from extreme_price_movements.strategy_registry import (
    get_strategies,
    strategy_runtime_horizons,
)
from extreme_price_movements.universe import (
    apply_hardcoded_universe_exclusions,
    build_fetch_universe,
    deduplicate_symbols_by_base,
    get_available_perp_spot_symbols,
    refresh_margin_universe_daily,
)
from extreme_price_movements.utils import tprint

# SINGLE SOURCE OF TRUTH FOR FEES - All fee configuration comes from these constants
# Spot trading fees (default)
BASE_ROUND_TRIP_FEE_PCT = 0.3  # 0.3% round-trip = 0.15% per side (15 bps)
# Perpetual trading fees (when --perps flag used)
PERP_ROUND_TRIP_FEE_PCT = 0.1  # 0.1% round-trip = 0.05% per side (5 bps)

# Market order fee per side (used when not using limit orders)
MARKET_ORDER_FEE_BPS = 25.0  # 0.25% per side
# Limit order fee per side (used when limit order fills)
LIMIT_ORDER_FEE_BPS = 10.0  # 0.10% per side


def _apply_fee_model(cfg: dict, round_trip_fee_pct: float) -> None:
    """Normalize fee keys used across training, sizing, and optimisation steps."""
    rt = float(round_trip_fee_pct)
    side_bps = rt * 100.0 / 2.0
    fee_dec = rt / 100.0
    cfg["label_round_trip_fee_pct"] = rt
    cfg["sample_weight_fee_rt"] = fee_dec
    cfg["fee_bps"] = side_bps
    cfg["optimiser_fee_pct"] = fee_dec
    cfg["ridge_cost_pct"] = fee_dec
    cfg["limit_fill_fee_bps"] = side_bps

    # New fee structure for limit orders
    cfg["fee_bps_market"] = MARKET_ORDER_FEE_BPS
    cfg["fee_bps_limit_entry"] = LIMIT_ORDER_FEE_BPS
    cfg["fee_bps_limit_exit"] = LIMIT_ORDER_FEE_BPS
    cfg["fee_bps_market_exit"] = MARKET_ORDER_FEE_BPS

    # Enable MAE/MFE-based limit offset estimation
    cfg["use_mae_mfe_limit_offset"] = True
    cfg["use_exit_limit_orders"] = True


MARKET_MODE_SUFFIXES = {"spot": "_spot", "perps": "_perp"}
LEGACY_MARKET_SUFFIXES = ("_spot", "_perps", "_perp")


def _append_suffix(path: str, suffix: str) -> str:
    norm = path.rstrip("/\\")
    if norm.endswith(suffix):
        return norm
    return f"{norm}{suffix}"

def _market_mode_from_cfg(cfg: dict) -> str:
    mode = str(cfg.get("market_mode") or "").strip().lower()
    if mode in {"perp", "perps", "futures"}:
        return "perps"
    if mode == "spot":
        return "spot"
    return "perps" if bool(cfg.get("use_perps", False)) else "spot"


def _with_market_suffix(path: str, market_mode: str) -> str:
    norm = str(path or "").rstrip("/\\")
    suffix = MARKET_MODE_SUFFIXES["perps" if market_mode == "perps" else "spot"]
    for existing in LEGACY_MARKET_SUFFIXES:
        if norm.endswith(existing):
            norm = norm[: -len(existing)]
            break
    return f"{norm}{suffix}"


def _apply_market_mode_paths(cfg: dict, market_mode: str) -> None:
    mode = "perps" if market_mode == "perps" else "spot"
    cfg["market_mode"] = mode
    cfg["use_perps"] = mode == "perps"
    for key, default in (
        ("data_root", "data"),
        ("reports_root", "reports"),
        ("hf_data_dir", "15m_ohlcv"),
    ):
        cfg[key] = _with_market_suffix(str(cfg.get(key, default)), mode)
    os.environ["EPM_MARKET_MODE"] = mode
    os.environ["EPM_HF_DATA_DIR"] = str(cfg["hf_data_dir"])


def _resolve_path(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _normalize_cfg_paths(cfg: dict) -> None:
    """
    Normalize relative config paths to stable absolute paths independent of cwd.
    """
    # Resolve paths relative to the project root (parent of this script's directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg["data_root"] = _resolve_path(project_root, str(cfg.get("data_root", "data")))
    cfg["reports_root"] = _resolve_path(
        project_root, str(cfg.get("reports_root", "reports"))
    )
    cfg["hf_data_dir"] = _resolve_path(
        project_root, str(cfg.get("hf_data_dir", "15m_ohlcv"))
    )


def _configure_report_roots(cfg: dict) -> None:
    report_root = cfg.get("reports_root")
    if report_root:
        os.environ["EPM_REPORTS_DIR"] = str(report_root)


def _load_mask_params_by_mode(cfg: dict) -> dict:
    """Refresh cfg with persisted offline optimizer params (including mask params by mode).
    Also populates cfg['strategies'] from final_rule_registry.csv via load_inference_candidate_mask_params_per_bucket().
    """
    merged = apply_offline_optimizer_best_params(dict(cfg))
    cfg.update(merged)
    if str(os.environ.get("EPM_SKIP_MASK_STRATEGY_PARAMS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        tprint(
            "Mask strategy params load skipped by EPM_SKIP_MASK_STRATEGY_PARAMS=1; "
            "using configured/static strategy registry."
        )
        return dict(cfg.get("candidate_mask_params_by_mode", {}) or {})

    # Populate strategies from final_rule_registry.csv
    _strategy_top_n = int(
        os.environ.get("EPM_MASK_STRATEGY_TOP_N", cfg.get("mask_strategy_top_n", 2))
        or 2
    )
    _strategy_ranking_metric = str(
        os.environ.get(
            "EPM_MASK_STRATEGY_RANKING_METRIC",
            cfg.get("mask_strategy_ranking_metric", "score_for_best_params"),
        )
    )
    _strategy_class_filter = str(
        os.environ.get(
            "EPM_MASK_STRATEGY_CLASSIFICATION_FILTER",
            cfg.get("mask_strategy_classification_filter", ""),
        )
        or ""
    ).strip()
    strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=max(1, _strategy_top_n),
        ranking_metric=_strategy_ranking_metric,
        classification_filter=_strategy_class_filter or None,
        market_mode=_market_mode_from_cfg(cfg),
    )
    if not strategies:
        from extreme_price_movements.offline_optimisers.params_store import (
            REPORTS_DIR as _OPT_REPORTS_DIR,
        )

        _bucket_csv = resolve_mode_file(
            _OPT_REPORTS_DIR / "inference_candidate_mask_best_params_per_bucket.csv",
            _market_mode_from_cfg(cfg),
        )
        if _bucket_csv.exists():
            import json as _json

            _bdf = pd.read_csv(_bucket_csv)
            if not _bdf.empty and "strategy_id" in _bdf.columns:
                strategies = []
                for _, _row in _bdf.iterrows():
                    _s = {
                        "strategy_id": str(_row["strategy_id"]),
                        "trade_side": str(_row.get("trade_side", "long")).lower(),
                        "base_event_trigger": str(_row.get("base_event_trigger", "")),
                        "source_horizon": int(_row.get("source_horizon", 5)),
                    }
                    _mpj = _row.get("mask_params_json")
                    if isinstance(_mpj, str) and _mpj.strip().startswith("{"):
                        try:
                            _s["mask_params"] = _json.loads(_mpj)
                        except Exception:
                            pass
                    strategies.append(_s)
                tprint(
                    f"Fallback: loaded {len(strategies)} strategies from "
                    f"inference_candidate_mask_best_params_per_bucket.csv"
                )
    if strategies:
        valid_strategies = [
            s for s in strategies if s.get("source_horizon") is not None
        ]
        dropped = len(strategies) - len(valid_strategies)
        if dropped:
            tprint(
                f"Dropping {dropped} mask strategies without source_horizon; "
                "primary training requires one horizon per strategy."
            )
        cfg["strategies"] = valid_strategies
        from extreme_price_movements.slice_plan_store import (
            apply_stage_usage_limits,
            load_or_build_slice_plan,
        )

        horizons = sorted(
            {
                int(s["source_horizon"])
                for s in valid_strategies
                if s.get("source_horizon") is not None
            }
        )
        tprint(
            f"Loaded {len(valid_strategies)} strategies "
            f"with source_horizons={horizons}"
        )
    else:
        tprint("WARNING: No strategies loaded — will fall back to legacy strategies.")

    return dict(cfg.get("candidate_mask_params_by_mode", {}) or {})


def _truthy_env(name: str, default: str = "") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _maybe_extend_training_stage_to_latest(stage_view: dict, *, stage_name: str) -> dict:
    """Opt-in retraining override: keep stage start/symbols, remove stale OOS cutoff."""
    if not _truthy_env("EPM_TRAIN_EXTEND_TO_LATEST"):
        return stage_view
    view = dict(stage_view or {})
    original_end = view.get("allowed_end_ts") or view.get("fit_end")
    original_periods = view.get("allowed_periods")
    view.pop("allowed_end_ts", None)
    view.pop("fit_end", None)
    view.pop("allowed_periods", None)
    if _truthy_env("EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER"):
        view["disable_exact_plan_row_filter"] = True
    else:
        view.pop("disable_exact_plan_row_filter", None)
        view["preserve_exact_plan_row_filter"] = True
    view["extended_to_latest"] = True
    view["stage_name"] = view.get("stage_name") or stage_name
    recent_days_raw = str(os.environ.get("EPM_TRAIN_RECENT_DAYS", "")).strip()
    if recent_days_raw:
        try:
            recent_days = max(1.0, float(recent_days_raw))
            recent_start = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
            existing_start = pd.to_datetime(
                view.get("allowed_start_ts") or view.get("fit_start"),
                utc=True,
                errors="coerce",
            )
            if pd.isna(existing_start) or recent_start > existing_start:
                view["allowed_start_ts"] = recent_start.isoformat()
                view["fit_start"] = recent_start.isoformat()
            tprint(
                f"{stage_name}: EPM_TRAIN_RECENT_DAYS={recent_days:g}; "
                f"clamped start={view.get('allowed_start_ts') or view.get('fit_start')}."
            )
        except Exception as exc:
            tprint(
                f"{stage_name}: ignored invalid EPM_TRAIN_RECENT_DAYS={recent_days_raw!r}: {exc}"
            )
    tprint(
        f"{stage_name}: EPM_TRAIN_EXTEND_TO_LATEST=1; preserving "
        f"start={view.get('allowed_start_ts') or view.get('fit_start') or 'artifact-start'} "
        f"and symbols={len(view.get('symbols') or []) or 'ALL'}, removing "
        f"end={original_end or 'none'} and "
        f"allowed_periods={'yes' if original_periods else 'no'}; "
        f"exact_plan_row_filter="
        f"{'disabled' if view.get('disable_exact_plan_row_filter') else 'preserved'}."
    )
    return view


def _load_training_slice_plan(cfg: dict, source_ts_sig: pd.Timestamp) -> dict:
    """Load the training slice plan, optionally pinned for reproducible comparisons."""
    override_raw = str(os.environ.get("EPM_TRAIN_SLICE_PLAN_PATH", "") or "").strip()
    if override_raw:
        override_path = Path(override_raw).expanduser()
        with override_path.open("r", encoding="utf-8") as f:
            slice_plan = json.load(f)
        cfg["_training_slice_plan_path"] = str(override_path)
        event_run_id = str(
            os.environ.get("EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID", "") or ""
        ).strip()
        if event_run_id:
            cfg["_training_slice_plan_run_id"] = event_run_id
        else:
            try:
                cfg["_training_slice_plan_run_id"] = str(
                    override_path.parents[1].name
                )
            except Exception:
                cfg["_training_slice_plan_run_id"] = (
                    source_ts_sig.strftime("%Y%m%d_%H%M%S")
                )
        try:
            slice_plan["_event_source_run_id"] = str(
                cfg.get("_training_slice_plan_run_id") or ""
            )
        except Exception:
            pass
        tprint(f"Training slice plan override loaded from {override_path}")
        return slice_plan
    run_id = str(
        os.environ.get("EPM_TRAIN_SLICE_PLAN_RUN_ID", "")
        or cfg.get("_training_slice_plan_run_id")
        or cfg.get("label_source_run_id")
        or cfg.get("artifact_source_run_id")
        or cfg.get("output_run_id")
        or cfg.get("run_id")
        or source_ts_sig.strftime("%Y%m%d_%H%M%S")
    ).strip()
    if not run_id:
        run_id = source_ts_sig.strftime("%Y%m%d_%H%M%S")
    cfg["_training_slice_plan_run_id"] = run_id
    cfg["_training_slice_plan_path"] = str(
        Path(cfg["data_root"]) / "artifacts" / run_id / "slices" / "slice_plan.json"
    )
    return load_or_build_slice_plan(
        cfg, source_ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
    )


def _strategy_source_run_id(cfg: dict) -> str:
    return str(
        os.environ.get("EPM_STRATEGY_SOURCE_RUN_ID", "")
        or os.environ.get("EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID", "")
        or cfg.get("strategy_source_run_id")
        or cfg.get("lgbm_native_preset_source_run_id")
        or cfg.get("artifact_source_run_id")
        or cfg.get("run_id")
        or ""
    ).strip()


def _strategy_aliases(strategy_id: str, trade_side: str | None = None) -> set[str]:
    raw = str(strategy_id or "").strip()
    aliases = {raw} if raw else set()
    for prefix in ("long_", "short_"):
        if raw.startswith(prefix):
            aliases.add(raw[len(prefix) :])
    side = str(trade_side or "").strip().lower()
    base_ids = list(aliases)
    if side in {"long", "short"}:
        aliases.update({f"{side}_{sid}" for sid in base_ids if sid})
    return {sid for sid in aliases if sid}


def _load_policy_strategy_overrides(cfg: dict, source_run_id: str) -> dict[str, dict]:
    policy_path = (
        Path(str(cfg.get("data_root", "data")))
        / "artifacts"
        / str(source_run_id)
        / "policy_params"
        / "strategy_for_inference_perps.json"
    )
    if not policy_path.exists():
        return {}
    try:
        payload = json.loads(policy_path.read_text())
    except Exception as exc:
        tprint(f"WARNING: failed to read policy strategy overrides {policy_path}: {exc}")
        return {}
    out: dict[str, dict] = {}
    for row in payload.get("strategies", []) or []:
        if not isinstance(row, dict):
            continue
        mask = dict(row.get("lgbm_regime_mask", {}) or {})
        side = str(row.get("side") or mask.get("trade_side") or "").strip().lower()
        raw_sid = str(
            row.get("canonical_strategy_id")
            or row.get("strategy_id")
            or mask.get("strategy_id")
            or ""
        ).strip()
        if side in {"long", "short"} and raw_sid.startswith(f"{side}_"):
            sid = raw_sid[len(side) + 1 :]
        else:
            sid = raw_sid
        canonical_key = str(
            mask.get("canonical_key")
            or mask.get("base_event_trigger")
            or row.get("base_event_trigger")
            or ""
        ).strip()
        if not sid or not canonical_key:
            continue
        strategy = {
            "strategy_id": sid,
            "trade_side": "short" if side == "short" else "long",
            "base_event_trigger": canonical_key,
            "regime_filters": [],
            "source_horizon": int(mask.get("source_horizon") or 5),
            "mask_params": dict(mask.get("mask_params", {}) or {"canonical_key": canonical_key}),
        }
        if mask.get("source_target"):
            strategy["source_target"] = mask.get("source_target")
        for alias in _strategy_aliases(sid, strategy["trade_side"]):
            out[alias] = dict(strategy)
    return out


def _load_contract_strategies(cfg: dict, source_run_id: str) -> dict[str, dict]:
    contract_path = (
        Path(str(cfg.get("data_root", "data")))
        / "artifacts"
        / str(source_run_id)
        / "base_meta_contract.json"
    )
    if not contract_path.exists():
        return {}
    try:
        payload = json.loads(contract_path.read_text())
    except Exception as exc:
        tprint(f"WARNING: failed to read strategy contract {contract_path}: {exc}")
        return {}
    policy_overrides = _load_policy_strategy_overrides(cfg, source_run_id)
    allow_contract_only = (
        _truthy_env("EPM_ALLOW_CONTRACT_STRATEGY_WITHOUT_POLICY_MASK")
        or str(cfg.get("allow_contract_strategy_without_policy_mask", ""))
        .strip()
        .lower()
        in {"1", "true", "yes", "y", "on"}
    )
    out: dict[str, dict] = {}
    for row in payload.get("strategies", []) or []:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("strategy_id") or "").strip()
        side = str(row.get("trade_side") or "").strip().lower()
        if not sid or side not in {"long", "short"}:
            continue
        override = None
        for alias in _strategy_aliases(sid, side):
            if alias in policy_overrides:
                override = policy_overrides[alias]
                break
        if override is None:
            # The base/meta contract alone tells us the artifact strategy id and
            # side, but not the original canonical mask expression used to
            # generate labels. Do not invent a mask for ordinary retraining.
            # Opt-in contract-only hydration is allowed for artifact-reuse
            # train_meta runs where labels/OOF are loaded from existing files and
            # the mask is not recomputed.
            if not allow_contract_only:
                continue
            horizons = row.get("horizons") or [5]
            strategy = {
                "strategy_id": sid,
                "trade_side": "short" if side == "short" else "long",
                "base_event_trigger": str(row.get("base_event_trigger") or sid),
                "regime_filters": [],
                "source_horizon": int(horizons[0] if horizons else 5),
                "mask_params": {
                    "canonical_key": str(row.get("base_event_trigger") or sid)
                },
                "contract_only_artifact_reuse": True,
            }
        else:
            horizons = row.get("horizons") or [override.get("source_horizon", 5)]
            strategy = dict(override)
        strategy["strategy_id"] = sid
        strategy["trade_side"] = "short" if side == "short" else "long"
        strategy["source_horizon"] = int(horizons[0] if horizons else 5)
        for alias in _strategy_aliases(sid, side):
            out[alias] = dict(strategy)
    return out


def _select_explicit_strategies(
    cfg: dict,
    requested_ids: Sequence[str],
    *,
    env_label: str,
) -> tuple[list[dict], list[str]]:
    from extreme_price_movements.strategy_registry import get_strategies

    requested_ids = [str(s).strip() for s in requested_ids if str(s).strip()]
    requested_set = set(requested_ids)
    selected: list[dict] = []
    selected_aliases: set[str] = set()
    source_run_id = _strategy_source_run_id(cfg)
    hydrated = _load_contract_strategies(cfg, source_run_id) if source_run_id else {}
    if hydrated:
        for sid in requested_ids:
            strategy = hydrated.get(sid)
            if strategy is None:
                continue
            aliases = _strategy_aliases(
                str(strategy.get("strategy_id", "")).strip(),
                str(strategy.get("trade_side", "")).strip(),
            )
            if aliases & selected_aliases:
                selected_aliases.update(aliases)
                continue
            selected.append(strategy)
            selected_aliases.update(aliases)
        if selected_aliases & requested_set:
            tprint(
                f"{env_label}: hydrated {len(selected_aliases & requested_set)}/"
                f"{len(requested_ids)} requested strategies from source contract "
                f"run_id={source_run_id}"
            )
    for strategy in get_strategies(cfg):
        aliases = _strategy_aliases(
            str(strategy.get("strategy_id", "")).strip(),
            str(strategy.get("trade_side", "")).strip(),
        )
        if aliases & requested_set:
            selected.append(strategy)
            selected_aliases.update(aliases)

    missing = [sid for sid in requested_ids if sid not in selected_aliases]
    if missing:
        for sid in list(missing):
            strategy = hydrated.get(sid)
            if strategy is None:
                continue
            aliases = _strategy_aliases(
                str(strategy.get("strategy_id", "")).strip(),
                str(strategy.get("trade_side", "")).strip(),
            )
            if aliases & selected_aliases:
                selected_aliases.update(aliases)
                continue
            selected.append(strategy)
            selected_aliases.update(aliases)
        missing = [sid for sid in requested_ids if sid not in selected_aliases]
        if hydrated:
            tprint(
                f"{env_label}: hydrated {len(selected_aliases & requested_set)}/"
                f"{len(requested_ids)} requested strategies from source contract "
                f"run_id={source_run_id}"
            )

    order: dict[str, int] = {}
    for i, sid in enumerate(requested_ids):
        order[sid] = i
        side = "short" if sid.startswith("short_") else "long" if sid.startswith("long_") else ""
        for alias in _strategy_aliases(sid, side):
            order.setdefault(alias, i)
    selected.sort(
        key=lambda s: min(
            order.get(alias, 10**9)
            for alias in _strategy_aliases(
                str(s.get("strategy_id", "")).strip(),
                str(s.get("trade_side", "")).strip(),
            )
        )
    )
    return selected, missing


def _maybe_set_strategy_selection_mask_source(cfg: dict) -> None:
    """Prefer the latest generated LGBM mask registry for strategy-selection runs."""
    if str(os.environ.get("EPM_MASK_STRATEGY_SOURCE_CSV", "")).strip():
        return
    if str(os.environ.get("EPM_STRATEGY_SELECTION_AUTO_MASK_SOURCE", "1")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    try:
        min_candidates = int(
            os.environ.get("EPM_STRATEGY_SELECTION_MIN_MASK_CANDIDATES", "10") or "10"
        )
    except Exception:
        min_candidates = 10
    artifacts_root = Path(str(cfg.get("data_root", "data"))) / "artifacts"
    candidates = [
        p
        for p in artifacts_root.glob(
            "*/lgbm_based_mask_generation*/run_*/final_rule_registry*.csv"
        )
        if p.is_file()
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            df = pd.read_csv(path, usecols=["canonical_key"])
        except Exception:
            continue
        n_rows = int(len(df))
        n_parseable = int(df["canonical_key"].astype(str).str.contains("|", regex=False).sum())
        if n_rows < min_candidates:
            continue
        if n_parseable != n_rows:
            tprint(
                "STRATEGY SELECTION: skipping generated LGBM mask registry with "
                f"non-parseable keys {path} ({n_parseable}/{n_rows} parseable)."
            )
            continue
        os.environ["EPM_MASK_STRATEGY_SOURCE_CSV"] = str(path)
        tprint(
            "STRATEGY SELECTION: using generated LGBM mask registry "
            f"{path} ({n_rows} candidates)."
        )
        return


def _downcast_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce optimiser memory footprint."""
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_float_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _resolve_ts_sig(cfg: dict, ts_override=None) -> pd.Timestamp:
    if ts_override:
        try:
            _raw_ts = str(ts_override)
            _ts_str = _raw_ts.split("_v")[0] if "_v" in _raw_ts else _raw_ts
            _match = re.match(r"^(\d{8}_\d{6})", _ts_str)
            if _match:
                _ts_str = _match.group(1)
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg.get("data_root", "data"))
    return ts_sig


def _choose_policy_stage_view(materialized: dict) -> Optional[dict]:
    """Select the policy-optimiser training/tuning slice, then the eval fallback."""
    preferred_keys = (
        "sizer_train",
        "utility_policy_optimisation",
        "holdout_strategy_eval",
    )
    for key in preferred_keys:
        stage_view = materialized.get(key)
        if not isinstance(stage_view, dict):
            continue
        periods = stage_view.get("allowed_periods") or []
        symbols = stage_view.get("symbols") or []
        if periods and symbols:
            msg = (
                f"Policy optimiser using slice stage '{key}' "
                f"(periods={len(periods)}, symbols={len(symbols)})"
            )
            if key == "sizer_train":
                msg += " — repurposing the former sizer slice as the primary policy-optimisation train/tuning split."
            elif key == "holdout_strategy_eval":
                msg += " — using evaluation holdout fallback only."
            tprint(msg)
            return stage_view
    return None


def _strict_policy_optimiser_stage_view(slice_plan: dict) -> Optional[dict]:
    """Return the exact final policy-optimiser prediction slice."""
    consumers = slice_plan.get("consumer_plans", {})
    if not isinstance(consumers, dict):
        return None
    plans = consumers.get("policy_optimiser", [])
    if not isinstance(plans, list) or not plans:
        return None

    allowed_periods: list[dict[str, str]] = []
    symbols: set[str] = set()
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        meta = plan.get("metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        start = meta.get("predict_actual_start") or meta.get("predict_start")
        end = meta.get("predict_actual_end") or meta.get("predict_end")
        if start and end:
            allowed_periods.append({"start_ts": str(start), "end_ts": str(end)})
        for symbol in plan.get("symbols_predict", []) or []:
            symbols.add(str(symbol))
    if not allowed_periods:
        return None

    starts = [
        pd.to_datetime(p["start_ts"], utc=True, errors="coerce")
        for p in allowed_periods
    ]
    ends = [
        pd.to_datetime(p["end_ts"], utc=True, errors="coerce")
        for p in allowed_periods
    ]
    starts = [ts for ts in starts if not pd.isna(ts)]
    ends = [ts for ts in ends if not pd.isna(ts)]
    return {
        "stage_name": "policy_optimiser",
        "source_roles": ["policy_optimiser"],
        "symbols": sorted(symbols),
        "allowed_symbols": sorted(symbols),
        "allowed_periods": allowed_periods,
        "allowed_start_ts": min(starts).isoformat() if starts else None,
        "allowed_end_ts": max(ends).isoformat() if ends else None,
        "n_plans": int(len(plans)),
        "policy_source": "consumer_predict_plans",
    }


def _attach_feature_availability_policy_view(cfg: dict, slice_plan: dict, ts_sig: pd.Timestamp) -> None:
    """Use the simple_policy_optimiser slice for feature coverage pruning."""
    policy_view = _strict_policy_optimiser_stage_view(slice_plan)
    if not policy_view:
        tprint("Feature availability reference: strict policy_optimiser slice unavailable; using training rows.")
        return
    feature_source_run_id = (
        str(os.environ.get("EPM_POLICY_FEATURE_SOURCE_RUN_ID") or "").strip()
        or str(os.environ.get("EPM_FEATURE_SOURCE_RUN_ID") or "").strip()
        or ts_sig.strftime("%Y%m%d_%H%M%S")
    )
    cfg["_feature_availability_stage_view"] = policy_view
    cfg["_feature_availability_run_id"] = feature_source_run_id
    periods = policy_view.get("allowed_periods") or []
    symbols = policy_view.get("symbols") or []
    tprint(
        "Feature availability reference: using strict simple_policy_optimiser "
        f"slice (periods={len(periods)}, symbols={len(symbols)}, "
        f"start={policy_view.get('allowed_start_ts')}, end={policy_view.get('allowed_end_ts')}, "
        f"feature_run_id={feature_source_run_id})."
    )


def _find_latest_feature_ts(data_root):
    """Find the latest feature timestamp directory."""
    import glob
    import os

    feat_dir = os.path.join(data_root, "features")
    if not os.path.exists(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    if not dirs:
        return None
    latest = os.path.basename(dirs[-1])
    return pd.to_datetime(latest, format="%Y%m%d_%H%M%S").tz_localize("UTC")


def run_download(cfg):
    """Download OHLCV data from Binance for the full training universe."""
    cfg.setdefault("allow_15m_download", False)
    import time as _time

    from extreme_price_movements.hf_data_loader import sync_15m_ohlcv_range

    tprint("STEP: DOWNLOAD START")
    store = make_ohlcv_store(cfg)
    use_perps = bool(cfg.get("use_perps", False))
    _check_complete = str(
        os.environ.get(
            "EPM_DOWNLOAD_CHECK_COMPLETE", str(cfg.get("download_check_complete", True))
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    _missing_lt_days = float(
        os.environ.get(
            "EPM_DOWNLOAD_SKIP_LT_DAYS",
            cfg.get("download_skip_if_missing_lt_days", 3.0),
        )
        or 0.0
    )
    _download_15m_enabled = str(
        os.environ.get(
            "EPM_DOWNLOAD_15M_ENABLED",
            str(cfg.get("download_15m_enabled", True)),
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    _download_microdata_enabled = str(
        os.environ.get(
            "EPM_DOWNLOAD_MICRODATA_ENABLED",
            str(cfg.get("download_microdata_enabled", True)),
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}

    # --- Freshness check: skip download if data is < 6 days old ---
    import glob as _glob
    import json as _json

    _FRESHNESS_DAYS = 6
    _force_download = str(
        os.environ.get("EPM_DOWNLOAD_FORCE", str(cfg.get("download_force", False)))
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    _meta_dir = store.ohlcv_dir
    _meta_files = _glob.glob(os.path.join(_meta_dir, "*.meta.json"))
    if _meta_files and (not _force_download) and (not _check_complete):
        _latest_ms = 0
        for mf in _meta_files[:20]:  # sample up to 20 symbols
            try:
                with open(mf) as _fp:
                    _m = _json.load(_fp)
                _latest_ms = max(_latest_ms, _m.get("last_ts_ms", 0))
            except Exception:
                pass
        if _latest_ms > 0:
            _latest_ts = pd.to_datetime(_latest_ms, unit="ms", utc=True)
            _age = pd.Timestamp.utcnow() - _latest_ts
            tprint(f"Data freshness: latest={_latest_ts}, age={_age}")
            if _age < pd.Timedelta(days=_FRESHNESS_DAYS):
                tprint(
                    f"Data is {_age.total_seconds()/3600:.1f}h old (< {_FRESHNESS_DAYS}d). Skipping download."
                )
                tprint("STEP: DOWNLOAD COMPLETE (fresh)")
                return
    elif _force_download:
        tprint("Freshness gate bypassed (download_force=True)")

    ex = make_perp_exchange() if use_perps else make_spot_exchange()
    spot_aux_ex = None
    if use_perps:
        try:
            spot_aux_ex = make_spot_exchange()
        except Exception as exc:
            tprint(f"Spot auxiliary exchange unavailable; spot/perp features degraded: {exc}")
    funding_ex = ex
    if not use_perps:
        try:
            funding_ex = make_perp_exchange()
        except Exception as exc:
            funding_ex = None
            tprint(f"Funding exchange unavailable; skipping funding refresh: {exc}")

    if use_perps:
        exchange_id = str(cfg.get("exchange_id") or cfg.get("exchange") or "").lower()
        if exchange_id == "binance":
            raw_perp_symbols = sorted(get_available_perp_spot_symbols(force_refresh=True))
            mu = refresh_margin_universe_daily(None, quotes=("USDC",))
            margin_usdc_bases = {
                sym.split("/", 1)[0]
                for sym in apply_hardcoded_universe_exclusions(mu.symbols)
                if "/" in sym
            }
            sanitized_perp_symbols = apply_hardcoded_universe_exclusions(raw_perp_symbols)
            perp_symbols = [
                sym
                for sym in sanitized_perp_symbols
                if "/" in sym and sym.split("/", 1)[0] in margin_usdc_bases
            ]
            universe_source = (
                "active USDC/USDT perpetual markets intersected with USDC "
                "margin spot bases"
            )
            universe_counts = (
                f"{len(raw_perp_symbols)} raw perps, "
                f"{len(sanitized_perp_symbols)} sanitized, "
                f"{len(margin_usdc_bases)} margin bases"
            )
        else:
            raw_perp_symbols = []
            for market in getattr(ex, "markets", {}).values():
                if not isinstance(market, dict):
                    continue
                if not bool(market.get("swap") or market.get("future")):
                    continue
                if market.get("active") is False:
                    continue
                symbol = str(market.get("symbol") or "").strip()
                if not symbol or "/" not in symbol:
                    continue
                quote = str(market.get("quote") or symbol.split("/", 1)[1]).split(":", 1)[0]
                if quote.upper() not in {"USD", "USDC", "USDT"}:
                    continue
                raw_perp_symbols.append(symbol)
            sanitized_perp_symbols = apply_hardcoded_universe_exclusions(
                sorted(set(raw_perp_symbols))
            )
            perp_symbols = deduplicate_symbols_by_base(sanitized_perp_symbols)
            universe_source = f"active {exchange_id} perpetual markets"
            universe_counts = (
                f"{len(raw_perp_symbols)} raw exchange perps, "
                f"{len(sanitized_perp_symbols)} sanitized"
            )
        _perps_m = int(cfg.get("fetch_symbols_M", 0) or 0)
        fetch_syms = (
            perp_symbols[:_perps_m]
            if _perps_m > 0 and _perps_m < len(perp_symbols)
            else perp_symbols
        )
        tprint(
            f"Perps download universe source: {universe_source} "
            f"({universe_counts}, {len(fetch_syms)} before runtime slicing)"
        )
    else:
        mu = refresh_margin_universe_daily(None, quotes=("USDC",))
        fetch_syms = build_fetch_universe(
            mu.symbols, cfg["market_basket"], cfg["fetch_symbols_M"]
        )
    _base_n = len(fetch_syms)
    _symbol_allowlist_file = str(
        os.environ.get("EPM_DOWNLOAD_SYMBOLS_FILE", "")
    ).strip()
    _symbol_allowlist_raw = str(os.environ.get("EPM_DOWNLOAD_SYMBOLS", "")).strip()
    if _symbol_allowlist_file:
        try:
            with open(_symbol_allowlist_file, "r", encoding="utf-8") as fp:
                _file_symbols = [line.strip() for line in fp if line.strip()]
            _symbol_allowlist_raw = ",".join(
                [_symbol_allowlist_raw, *_file_symbols]
                if _symbol_allowlist_raw
                else _file_symbols
            )
        except Exception as exc:
            tprint(
                f"WARNING: could not read EPM_DOWNLOAD_SYMBOLS_FILE="
                f"{_symbol_allowlist_file}: {exc}"
            )
    if _symbol_allowlist_raw:
        _symbol_allowlist = [
            s.strip() for s in _symbol_allowlist_raw.split(",") if s.strip()
        ]
        _symbol_allowset = set(_symbol_allowlist)
        _before_allowlist = len(fetch_syms)
        _replace_with_allowlist = str(
            os.environ.get("EPM_DOWNLOAD_SYMBOLS_REPLACE", "0")
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if _replace_with_allowlist:
            fetch_syms = list(dict.fromkeys(_symbol_allowlist))
            _allow_mode = "replace"
        else:
            fetch_syms = [s for s in fetch_syms if s in _symbol_allowset]
            _allow_mode = "filter"
        tprint(
            f"Download symbol allowlist applied ({_allow_mode}): {_before_allowlist} -> "
            f"{len(fetch_syms)} symbols"
        )

    # Runtime overrides for parallel download orchestrations.
    _order = (
        str(
            os.environ.get(
                "EPM_DOWNLOAD_SYMBOL_ORDER", cfg.get("download_symbol_order", "volume")
            )
        )
        .strip()
        .lower()
    )
    _stride = max(
        1,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_SYMBOL_STRIDE", cfg.get("download_symbol_stride", 1)
            )
        ),
    )
    _offset = max(
        0,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_SYMBOL_OFFSET", cfg.get("download_symbol_offset", 0)
            )
        ),
    )
    _max_symbols = int(
        os.environ.get("EPM_DOWNLOAD_MAX_SYMBOLS", cfg.get("download_max_symbols", 0))
        or 0
    )
    _part_count = max(
        1,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_PARTITION_COUNT", cfg.get("download_partition_count", 1)
            )
        ),
    )
    _part_id = max(
        0,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_PARTITION_ID", cfg.get("download_partition_id", 0)
            )
        ),
    )
    if _part_id >= _part_count:
        _part_id = _part_count - 1

    # Disjoint partitioning is based on canonical alpha order to avoid overlap
    # between multiple concurrent downloaders.
    _alpha = sorted(fetch_syms)
    if _part_count > 1:
        _selected = [s for i, s in enumerate(_alpha) if (i % _part_count) == _part_id]
    else:
        _selected = _alpha

    if _order in {"alpha_desc", "reverse_alpha", "reverse_alphabetical"}:
        fetch_syms = sorted(_selected, reverse=True)
    elif _order in {"alpha_asc", "alphabetical"}:
        fetch_syms = sorted(_selected)
    else:
        _sel = set(_selected)
        fetch_syms = [s for s in fetch_syms if s in _sel]

    if _stride > 1 or _offset > 0:
        fetch_syms = fetch_syms[_offset::_stride]

    if _max_symbols > 0:
        fetch_syms = fetch_syms[:_max_symbols]

    tprint(
        f"Download universe: {len(fetch_syms)} symbols "
        f"(base={_base_n}, order={_order}, stride={_stride}, offset={_offset}, "
        f"partition={_part_id}/{_part_count}, max={_max_symbols if _max_symbols > 0 else 'all'})"
    )

    fetch_years = float(
        os.environ.get("EPM_DOWNLOAD_FETCH_YEARS", cfg.get("fetch_years", 3)) or 3
    )
    since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(fetch_years * 365))
    since_ms = int(since.value // 10**6)
    now_utc = pd.Timestamp.now(tz="UTC")
    since_1h = since.floor("1h")
    now_1h = now_utc.floor("1h")
    since_15m = since.floor("15min")
    now_15m = now_utc.floor("15min")
    micro_since = since_1h

    def _panel_complete(
        df: Optional[pd.DataFrame],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        freq: str,
    ) -> bool:
        if df is None or df.empty:
            return False
        idx = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.to_datetime(df.index)
        )
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        effective_start = max(start_ts, idx.min())
        window = idx[(idx >= effective_start) & (idx <= end_ts)]
        if len(window) == 0:
            return False
        expected = len(
            pd.date_range(start=effective_start, end=end_ts, freq=freq, tz="UTC")
        )
        return (
            (len(window) == expected)
            and (window.min() <= effective_start)
            and (window.max() >= end_ts)
        )

    def _panel_missing_days(
        df: Optional[pd.DataFrame],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        freq: str,
    ) -> float:
        expected = len(pd.date_range(start=start_ts, end=end_ts, freq=freq, tz="UTC"))
        if expected <= 0:
            return 0.0
        if df is None or df.empty:
            return float((end_ts - start_ts) / pd.Timedelta(days=1))
        idx = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.to_datetime(df.index)
        )
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        effective_start = max(start_ts, idx.min())
        window = idx[(idx >= effective_start) & (idx <= end_ts)]
        observed = len(pd.DatetimeIndex(window).unique())
        expected = len(
            pd.date_range(start=effective_start, end=end_ts, freq=freq, tz="UTC")
        )
        missing_bars = max(0, expected - observed)
        step = pd.to_timedelta(freq)
        return float((missing_bars * step) / pd.Timedelta(days=1))

    def _symbol_status_1h(sym: str) -> Tuple[bool, float]:
        try:
            df_local = store.load(sym)
            if use_perps and _has_sparse_perp_auxiliary_data(
                df_local,
                since_ms=int(pd.Timestamp(since_1h).value // 10**6),
                timeframe="1h",
            ):
                return False, 1e9
            return (
                _panel_complete(df_local, since_1h, now_1h, "1h"),
                _panel_missing_days(df_local, since_1h, now_1h, "1h"),
            )
        except Exception:
            return False, 1e9

    def _symbol_status_15m(sym: str) -> Tuple[bool, float]:
        try:
            from extreme_price_movements.hf_data_loader import _load_existing_data

            df_local = _load_existing_data(sym)
            return (
                _panel_complete(df_local, since_15m, now_15m, "15min"),
                _panel_missing_days(df_local, since_15m, now_15m, "15min"),
            )
        except Exception:
            return False, 1e9

    market_data_root = Path(scoped_data_root(cfg))
    ob_dir = market_data_root / "orderbook_hourly"
    fr_dir = market_data_root / "funding_hourly"
    ob_dir.mkdir(parents=True, exist_ok=True)
    fr_dir.mkdir(parents=True, exist_ok=True)

    def _safe_symbol_path(sym: str) -> str:
        return sym.replace("/", "_").replace(":", "_")

    microdata_min_ranges = int(
        os.environ.get("EPM_DOWNLOAD_MIN_MICRODATA_RANGES", "10")
    )
    microdata_min_ranges = max(10, int(microdata_min_ranges))
    funding_history_days = float(
        os.environ.get("EPM_FUNDING_HISTORY_DAYS", cfg.get("funding_history_days", 30.0))
        or 30.0
    )
    funding_history_floor = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(funding_history_days, 0.0))
    ).floor("1h")

    def _update_symbol_microdata(
        sym: str,
    ) -> tuple[bool, bool, int, int, str, str]:
        now_h = pd.Timestamp.now(tz="UTC").floor("1h")
        sym_key = _safe_symbol_path(sym)
        ob_path = ob_dir / f"{sym_key}.parquet"
        fr_path = fr_dir / f"{sym_key}.parquet"

        ob_ok, fr_ok = False, False
        missing_ob_ranges = 0
        missing_fr_ranges = 0
        ob_io = "skip:no_missing"
        fr_io = "skip:no_missing"
        try:
            existing_ob = pd.read_parquet(ob_path) if ob_path.exists() else None
            existing_idx = (
                pd.to_datetime(existing_ob.index, utc=True, errors="coerce")
                if existing_ob is not None and not existing_ob.empty
                else None
            )
            missing_ranges = list(
                _compute_missing_hourly_ranges(
                    existing_idx,
                    micro_since,
                    now_h,
                )
            )
            missing_ob_hours = int(
                sum(
                    max(
                        0,
                        int(
                            (pd.Timestamp(range_end) - pd.Timestamp(range_start))
                            / pd.Timedelta(hours=1)
                        ),
                    )
                    for range_start, range_end in missing_ranges
                )
            )
            skip_reason = ""
            if missing_ob_hours < microdata_min_ranges:
                skip_reason = f"skip:insufficient_ranges<{microdata_min_ranges}"
                missing_ranges = []
                missing_ob_ranges = 0
            else:
                missing_ob_ranges = missing_ob_hours
            if not missing_ranges:
                ob_ok = existing_ob is not None and not existing_ob.empty
                ob_io = skip_reason or "skip:no_missing"
            else:
                tprint(
                    f"  orderbook microdata {sym}: fetching "
                    f"{len(missing_ranges)} missing hourly range(s)"
                )
                ob_io = "write:attempted"
            ob_frames = []
            if missing_ranges:
                ob_symbol = sym
                if use_perps:
                    ob_symbol = _resolve_perp_symbol(ex, sym) or sym
                for range_start, range_end in missing_ranges:
                    proxy_df = fetch_hourly_orderbook_proxy(
                        ex,
                        ob_symbol,
                        int(range_start.value // 10**6),
                        int(range_end.value // 10**6),
                    )
                    if proxy_df is not None and not proxy_df.empty:
                        ob_frames.append(proxy_df)
                if not ob_frames:
                    local_ob = store.load(sym, start_ts=micro_since, end_ts=now_h)
                    proxy_df = build_hourly_orderbook_proxy_from_ohlcv(local_ob)
                    if proxy_df is not None and not proxy_df.empty:
                        ob_frames.append(proxy_df)
                        ob_io = "write:attempted_local_ohlcv"
            if existing_ob is not None and not existing_ob.empty:
                ob_frames.insert(0, existing_ob)
            if missing_ranges and ob_frames:
                rec = pd.concat(ob_frames).sort_index().groupby(level=0).last()
                rec = normalize_orderbook_proxy_frame(rec)
                rec.to_parquet(ob_path)
                ob_ok = True
                ob_io = "write:ok"
            elif missing_ranges:
                ob_ok = False
                ob_io = "skip:insufficient_ranges<write_failed"
        except Exception as exc:
            tprint(f"  WARN orderbook microdata {sym}: {exc}")
            ob_ok = False
            ob_io = "write:fail"

        try:
            fr_df = None
            funding_symbol = _resolve_perp_symbol(funding_ex, sym) if use_perps else sym
            funding_client = funding_ex
            existing_funding = None
            if not use_perps:
                funding_symbol = (
                    _resolve_perp_symbol(funding_ex, sym)
                    if funding_ex is not None
                    else None
                )
                if funding_symbol is None:
                    raise ValueError(f"No perp funding symbol found for {sym}")
            elif funding_symbol is None:
                raise ValueError(f"No perp funding symbol found for {sym}")
            if funding_client is not None and hasattr(
                funding_client, "fetch_funding_rate_history"
            ):
                if fr_path.exists():
                    existing_funding = pd.read_parquet(fr_path)
                until_ms_local = int((now_h + pd.Timedelta(hours=1)).value // 10**6)
                existing_idx = (
                    pd.to_datetime(existing_funding.index, utc=True, errors="coerce")
                    if existing_funding is not None and not existing_funding.empty
                    else None
                )
                funding_missing_ranges = list(
                    _compute_missing_funding_ranges(
                        existing_idx,
                        max(micro_since, funding_history_floor),
                        now_h,
                    )
                )
                missing_funding_points = int(
                    sum(
                        max(
                            0,
                            int(
                                (pd.Timestamp(range_end) - pd.Timestamp(range_start))
                                / pd.Timedelta(hours=8)
                            )
                            + 1,
                        )
                        for range_start, range_end in funding_missing_ranges
                    )
                )
                if missing_funding_points < microdata_min_ranges:
                    funding_missing_ranges = []
                    missing_fr_ranges = 0
                else:
                    missing_fr_ranges = missing_funding_points
                if not funding_missing_ranges:
                    fr_ok = True
                    fr_io = "skip:no_missing"
                elif existing_funding is None:
                    fr_io = f"skip:insufficient_ranges<{microdata_min_ranges}"
                else:
                    tprint(
                        f"  funding microdata {sym}: fetching "
                        f"{len(funding_missing_ranges)} missing hourly range(s)"
                    )
                    fr_io = "write:attempted"
                funding_frames = []
                if funding_missing_ranges:
                    for range_start, range_end in funding_missing_ranges:
                        hist = _fetch_ccxt_history_paged(
                            funding_client.fetch_funding_rate_history,
                            funding_symbol,
                            int(range_start.value // 10**6),
                            int(range_end.value // 10**6),
                            value_keys=["fundingRate", "funding_rate", "rate"],
                            exchange=funding_client,
                            limit=1000,
                        )
                        if len(hist) > 0:
                            funding_frames.append(hist.to_frame(name="funding_rate"))
                if funding_frames:
                    fr_df = pd.concat(funding_frames).sort_index()
            if (
                fr_df is None
                and funding_client is not None
                and hasattr(funding_client, "fetch_funding_rate")
                and (
                    existing_funding is None or getattr(existing_funding, "empty", True)
                )
            ):
                fr = funding_client.fetch_funding_rate(funding_symbol)
                fr_df = pd.DataFrame([fr])
            if fr_df is not None and not fr_df.empty:
                if "funding_rate" in fr_df.columns and isinstance(
                    fr_df.index, pd.DatetimeIndex
                ):
                    fr_df = fr_df.copy()
                    fr_df.index = pd.to_datetime(fr_df.index, utc=True).floor("1h")
                    fr_df = fr_df[["funding_rate"]]
                else:
                    ts_col = (
                        "timestamp"
                        if "timestamp" in fr_df.columns
                        else "fundingTimestamp"
                    )
                    rate_col = (
                        "fundingRate"
                        if "fundingRate" in fr_df.columns
                        else "funding_rate"
                    )
                    fr_df["ts"] = pd.to_datetime(
                        fr_df[ts_col], unit="ms", utc=True
                    ).dt.floor("1h")
                    fr_df = (
                        fr_df[["ts", rate_col]]
                        .rename(columns={rate_col: "funding_rate"})
                        .set_index("ts")
                    )
                    fr_df["funding_rate"] = pd.to_numeric(
                fr_df["funding_rate"], errors="coerce"
            ).astype(np.float32)
                if existing_funding is None and fr_path.exists():
                    existing_funding = pd.read_parquet(fr_path)
                if existing_funding is not None and not existing_funding.empty:
                    fr_df = (
                        pd.concat([existing_funding, fr_df])
                        .sort_index()
                        .groupby(level=0)
                        .last()
                    )
                fr_df.to_parquet(fr_path)
                fr_ok = True
                fr_io = "write:ok"
        except Exception as exc:
            tprint(f"  WARN funding microdata {sym}: {exc}")
            fr_ok = False
            fr_io = "write:fail"
        return ob_ok, fr_ok, missing_ob_ranges, missing_fr_ranges, ob_io, fr_io

    success_1h, fail_1h = 0, 0
    success_15m, fail_15m = 0, 0
    skip_1h, skip_15m = 0, 0
    skip_small_1h, skip_small_15m = 0, 0
    for i, sym in enumerate(fetch_syms):
        if _check_complete:
            complete_1h, missing_1h_days = _symbol_status_1h(sym)
            if _download_15m_enabled:
                complete_15m, missing_15m_days = _symbol_status_15m(sym)
            else:
                complete_15m, missing_15m_days = True, 0.0
        else:
            complete_1h, complete_15m = False, False
            missing_1h_days, missing_15m_days = 1e9, 1e9

        symbol_1h_status = "skip:complete" if complete_1h else "pending"
        symbol_15m_status = "skip:complete" if complete_15m else "pending"
        if complete_1h:
            skip_1h += 1
        elif missing_1h_days < _missing_lt_days:
            skip_1h += 1
            skip_small_1h += 1
            symbol_1h_status = f"skip:recent_missing<{_missing_lt_days:g}d"
        else:
            try:
                if use_perps:
                    store.update_symbol_perp(ex, sym, since_ms, spot_exchange=spot_aux_ex)
                else:
                    store.update_symbol(ex, sym, since_ms)
                success_1h += 1
                symbol_1h_status = "write:ok"
            except Exception as e:
                fail_1h += 1
                symbol_1h_status = f"fail:{e.__class__.__name__}"
                tprint(f"  FAIL 1h {sym}: {e}")

        if not _download_15m_enabled:
            skip_15m += 1
            symbol_15m_status = "skip:disabled"
        elif complete_15m:
            skip_15m += 1
        elif missing_15m_days < _missing_lt_days:
            skip_15m += 1
            skip_small_15m += 1
            symbol_15m_status = f"skip:recent_missing<{_missing_lt_days:g}d"
        else:
            try:
                dl_15m_symbol = _resolve_perp_symbol(ex, sym) if use_perps else sym
                if dl_15m_symbol is None:
                    raise ValueError(f"No perp 15m OHLCV symbol found for {sym}")
                df_15m = sync_15m_ohlcv_range(
                    ex,
                    dl_15m_symbol,
                    since,
                    pd.Timestamp.now(tz="UTC"),
                    full_backfill=bool(cfg.get("download_15m_full_backfill", False)),
                )
                if df_15m is None or df_15m.empty:
                    fail_15m += 1
                    symbol_15m_status = "fail:empty"
                    tprint(f"  FAIL 15m {sym}: empty range")
                else:
                    success_15m += 1
                    symbol_15m_status = "write:ok"
            except Exception as e:
                fail_15m += 1
                symbol_15m_status = f"fail:{e.__class__.__name__}"
                tprint(f"  FAIL 15m {sym}: {e}")

        if _download_microdata_enabled:
            ob_ok, fr_ok, missing_ob_ranges, missing_fr_ranges, ob_io, fr_io = (
                _update_symbol_microdata(sym)
            )
        else:
            ob_ok, fr_ok = False, False
            missing_ob_ranges, missing_fr_ranges = 0, 0
            ob_io, fr_io = "skip:disabled", "skip:disabled"
        tprint(
            f"  [{i+1:04d}/{len(fetch_syms):04d}] {sym} "
            f"1h={symbol_1h_status} "
            f"15m={symbol_15m_status} "
            f"ob={ob_io}({missing_ob_ranges}) "
            f"fr={fr_io}({missing_fr_ranges})"
        )
        try:
            tprint(
                f"  Download progress: {i+1}/{len(fetch_syms)} "
                f"(1h ok={success_1h}, 1h skip={skip_1h} [<{_missing_lt_days:g}d={skip_small_1h}], 1h fail={fail_1h}, "
                f"15m ok={success_15m}, 15m skip={skip_15m} [<{_missing_lt_days:g}d={skip_small_15m}], 15m fail={fail_15m})"
            )
        except Exception:
            pass
        _time.sleep(0.1)  # gentle rate limit

    tprint(
        f"STEP: DOWNLOAD COMPLETE — symbols={len(fetch_syms)} "
        f"(1h ok={success_1h}, 1h skip={skip_1h} [<{_missing_lt_days:g}d={skip_small_1h}], 1h fail={fail_1h}; "
        f"15m ok={success_15m}, 15m skip={skip_15m} [<{_missing_lt_days:g}d={skip_small_15m}], 15m fail={fail_15m})"
    )


def _label_artifacts_ready(cfg, ts_sig):
    """Check whether core label artifacts exist for this run timestamp."""
    import os

    run_id = str(
        cfg.get("label_source_run_id")
        or cfg.get("artifact_source_run_id")
        or cfg.get("_label_artifact_run_id")
        or ts_sig.strftime("%Y%m%d_%H%M%S")
    ).strip()
    from extreme_price_movements.data_store import load_artifact_manifest

    labels_manifest = load_artifact_manifest(cfg["data_root"], run_id, "labels") or {}
    manifest_datasets = labels_manifest.get("datasets") or {}

    if not manifest_datasets:
        tprint(f"Label artifacts incomplete for run_id={run_id}: empty manifest.")
        return False

    n_total = len(manifest_datasets)
    n_valid = 0
    for name in manifest_datasets:
        fpath = os.path.join(
            cfg["data_root"], "artifacts", run_id, "labels", f"{name}.parquet"
        )
        if not os.path.exists(fpath):
            continue
        try:
            if os.path.getsize(fpath) <= 0:
                continue
        except OSError:
            continue
        n_valid += 1

    if n_valid == 0:
        tprint(
            f"Label artifacts incomplete for run_id={run_id}: {n_total} manifest entries but 0 valid parquet files."
        )
        return False

    tprint(
        f"Label artifacts ready for run_id={run_id}: {n_valid}/{n_total} datasets present."
    )
    return True


def _gc_checkpoint(tag: str) -> int:
    """Trigger GC and emit a short checkpoint log."""
    import gc

    collected = gc.collect()
    tprint(f"GC[{tag}]: collected={collected}")
    return collected


def _clear_runtime_cache_dir(cdir: str) -> None:
    """Clear pipeline scratch cache without deleting live inference feature state."""
    import shutil

    preserve_live_cache = os.environ.get("EPM_CLEAR_LIVE_INFERENCE_CACHE", "0") not in {
        "1",
        "true",
        "TRUE",
        "yes",
    }
    if not preserve_live_cache or os.path.basename(cdir) != "cache":
        shutil.rmtree(cdir)
        return

    for name in os.listdir(cdir):
        path = os.path.join(cdir, name)
        if name == "inference_live_features":
            tprint(
                "CACHE: preserving live inference feature cache "
                f"{path}; set EPM_CLEAR_LIVE_INFERENCE_CACHE=1 to wipe it"
            )
            continue
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def _cache_checkpoint(tag: str) -> None:
    """Clear known runtime cache directories only if memory is running low."""
    import psutil

    mem = psutil.virtual_memory()
    # Only blast cache if available memory is under 25% or we have less than 4GB free
    if mem.percent < 75.0 and mem.available > 4 * 1024 * 1024 * 1024:
        tprint(
            f"CACHE[{tag}]: skipped cache wipe (mem_avail={mem.available/1e9:.1f}GB)"
        )
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dirs = [
        os.path.join(project_root, "cache"),
        os.path.join(project_root, "data_cache"),
    ]
    for cdir in cache_dirs:
        if os.path.exists(cdir):
            try:
                _clear_runtime_cache_dir(cdir)
                tprint(f"CACHE[{tag}]: cleared {cdir}")
            except Exception as e:
                tprint(f"CACHE[{tag}]: failed {cdir}: {e}")


def _maintenance_checkpoint(tag: str) -> None:
    """Run cache cleanup + GC checkpoint."""
    _cache_checkpoint(tag)
    _gc_checkpoint(tag)


def run_labels(cfg, horizons=None, ts_override=None, store=None):
    _maintenance_checkpoint("labels:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found. Run feature_generation first.")
        return

    label_artifact_run_id = str(
        cfg.get("_label_artifact_run_id")
        or cfg.get("output_run_id")
        or os.environ.get("EPM_LABEL_ARTIFACT_RUN_ID")
        or ""
    ).strip()
    if label_artifact_run_id:
        cfg["_label_artifact_run_id"] = label_artifact_run_id
        tprint(f"Labels artifact output run_id={label_artifact_run_id}")

    tprint(f"Labels mode. ts_sig={ts_sig} horizons={horizons}")
    _rollout_env = os.getenv("EPM_POLICY_ROLLOUT_LABELING_ENABLE", "").strip().lower()
    if _rollout_env:
        cfg["policy_rollout_labeling_enable"] = _rollout_env not in {
            "0",
            "false",
            "no",
            "off",
        }
        tprint(
            "Labels override: "
            f"policy_rollout_labeling_enable={bool(cfg['policy_rollout_labeling_enable'])} "
            "from EPM_POLICY_ROLLOUT_LABELING_ENABLE"
        )
    cfg.setdefault("use_noise_filter", False)
    _noise_filter_env = os.getenv("EPM_LABEL_USE_NOISE_FILTER", "").strip().lower()
    if _noise_filter_env:
        cfg["use_noise_filter"] = _noise_filter_env not in {
            "0",
            "false",
            "no",
            "off",
        }
        tprint(
            "Labels override: "
            f"use_noise_filter={bool(cfg['use_noise_filter'])} "
            "from EPM_LABEL_USE_NOISE_FILTER"
        )
    _noise_wick_env = os.getenv("EPM_LABEL_NOISE_FILTER_WICK_THR", "").strip()
    if _noise_wick_env:
        try:
            cfg["noise_filter_wick_thr"] = float(_noise_wick_env)
            tprint(
                "Labels override: "
                f"noise_filter_wick_thr={float(cfg['noise_filter_wick_thr']):.6f} "
                "from EPM_LABEL_NOISE_FILTER_WICK_THR"
            )
        except ValueError:
            tprint(
                "WARNING: ignoring invalid EPM_LABEL_NOISE_FILTER_WICK_THR="
                f"{_noise_wick_env!r}"
            )
    cfg["label_skip_slice_planner"] = True
    cfg["label_persist_incremental"] = True
    _label_persist_incremental_env = os.getenv(
        "EPM_LABEL_PERSIST_INCREMENTAL", ""
    ).strip()
    if _label_persist_incremental_env:
        cfg["label_persist_incremental"] = _label_persist_incremental_env.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        tprint(
            "Labels override: "
            f"label_persist_incremental={bool(cfg['label_persist_incremental'])} "
            "from EPM_LABEL_PERSIST_INCREMENTAL"
        )
    cfg["label_parallel_enable"] = False
    cfg["label_tb_cache_parallel"] = False
    cfg["label_tb_cache_workers"] = 1
    cfg["label_geom_keep_all_if_lte_per_cell"] = 6
    cfg.setdefault("label_geom_keep_topn_per_cell", 6)
    cfg.setdefault("label_geom_heartbeat_every", 4)
    cfg.setdefault("label_geom_heartbeat_secs", 60.0)
    cfg.setdefault("label_raw_tb_payload_cache_mb", 2048.0)
    _label_incremental_only_env = os.getenv(
        "EPM_LABEL_INCREMENTAL_ONLY_MISSING", ""
    ).strip()
    if _label_incremental_only_env:
        cfg["label_incremental_only_missing"] = _label_incremental_only_env.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        tprint(
            "Labels override: "
            f"label_incremental_only_missing={bool(cfg['label_incremental_only_missing'])} "
            "from EPM_LABEL_INCREMENTAL_ONLY_MISSING"
        )
    _tb_worker_target = 2
    _tb_worker_mode = "auto"
    _tb_worker_fallback_avail_mb = 6144.0
    try:
        import psutil

        _vmem = psutil.virtual_memory()
        if float(_vmem.available / (1024**2)) < _tb_worker_fallback_avail_mb:
            _tb_worker_target = 1
            _tb_worker_mode = "fixed"
    except Exception:
        pass
    os.environ["EPM_LABEL_TB_WORKERS"] = str(_tb_worker_target)
    os.environ["EPM_LABEL_TB_WORKER_MODE"] = _tb_worker_mode
    os.environ["EPM_LABEL_TB_WORKER_FALLBACK_AVAIL_MB"] = str(
        int(_tb_worker_fallback_avail_mb)
    )
    tprint(
        "Labels optimization mode: "
        f"incremental_persist={bool(cfg['label_persist_incremental'])} "
        f"keep_all_if_lte={int(cfg['label_geom_keep_all_if_lte_per_cell'])} "
        f"keep_top_n={int(cfg['label_geom_keep_topn_per_cell'])} "
        f"geom_hb_every={int(cfg['label_geom_heartbeat_every'])} "
        f"geom_hb_secs={float(cfg['label_geom_heartbeat_secs']):.0f} "
        f"raw_tb_payload_cache_mb={float(cfg['label_raw_tb_payload_cache_mb']):.0f} "
        f"tb_workers={_tb_worker_target} worker_mode={_tb_worker_mode} "
        "geom_payload=compact(tp_vals/sl_vals)"
    )
    _load_mask_params_by_mode(cfg)
    _label_strategy_ids_env = (
        os.getenv("EPM_LABEL_STRATEGY_IDS", "").strip()
        or os.getenv("EPM_BASE_STRATEGY_IDS", "").strip()
        or os.getenv("EPM_META_STRATEGY_IDS", "").strip()
    )
    if _label_strategy_ids_env:
        requested_ids = [
            s.strip() for s in _label_strategy_ids_env.split(",") if s.strip()
        ]
        selected_strategies, missing_ids = _select_explicit_strategies(
            cfg,
            requested_ids,
            env_label="EPM_LABEL_STRATEGY_IDS",
        )
        if missing_ids:
            msg = (
                "EPM_LABEL_STRATEGY_IDS requested strategies not found after "
                f"mask-param/source-contract load: {missing_ids}"
            )
            if _truthy_env("EPM_REQUIRE_STRATEGY_ALLOWLIST"):
                raise RuntimeError(msg)
            tprint(f"WARNING: {msg}")
        if selected_strategies:
            cfg["strategies"] = selected_strategies
            tprint(
                "Labels override: explicit strategy allowlist active after "
                f"mask-param/source-contract load; selected {len(selected_strategies)}/"
                f"{len(requested_ids)} strategies"
            )
        else:
            msg = (
                "EPM_LABEL_STRATEGY_IDS matched no configured/source-contract "
                "strategies; falling back to configured strategy list"
            )
            if _truthy_env("EPM_REQUIRE_STRATEGY_ALLOWLIST"):
                raise RuntimeError(msg)
            tprint(f"WARNING: {msg}")

    if store is None:
        store = make_ohlcv_store(cfg)

    # No exchange needed — data already in store, features already on disk
    run_label_generation_step_v2(ts_sig, None, cfg, store, None, horizons=horizons)

    tprint("LABELS PIPELINE COMPLETE")
    _maintenance_checkpoint("labels:end")


def run_features(
    cfg,
    ts_override=None,
    force_recompute: bool = False,
    store=None,
    max_assets: int | None = None,
    feature_symbols: list[str] | None = None,
):
    _maintenance_checkpoint("features:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        ts_sig = pd.Timestamp.utcnow().floor("h")
    tprint(f"Features mode. Target ts_sig={ts_sig}")
    _load_mask_params_by_mode(cfg)

    if max_assets is not None:
        cfg["fetch_symbols_M"] = int(max_assets)
        cfg["skip_feature_snapshot_validation"] = True
        cfg["skip_feature_postsave_checks"] = True
        tprint(f"Features mode: limiting universe to {int(max_assets)} assets")

    feature_symbols = [
        str(s).strip() for s in (feature_symbols or []) if str(s).strip()
    ]
    if feature_symbols:
        cfg["skip_feature_snapshot_validation"] = True
        cfg["skip_feature_postsave_checks"] = True
        tprint("Features mode: explicit feature symbols=" + ", ".join(feature_symbols))

    if store is None:
        store = make_ohlcv_store(cfg)

    # Pass None for margin_symbols to trigger auto-refresh in universe logic.
    # Explicit symbols use subset mode, which still loads basket/BTC/ETH context.
    run_feature_generation_step(
        ts_sig,
        feature_symbols or None,
        cfg,
        store,
        force_full_recompute=bool(force_recompute),
    )

    tprint("FEATURES PIPELINE COMPLETE")
    _maintenance_checkpoint("features:end")


def run_backtest(cfg, ts_override=None, store=None):
    _maintenance_checkpoint("backtest:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    # Slice Plan Injection
    try:
        slice_plan = load_or_build_slice_plan(
            cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
        )
        if "holdout_strategy_eval" in slice_plan.get("materialized_views", {}):
            stage_view = (
                slice_plan["materialized_views"]["holdout_strategy_eval"]
                .get("sub_views", {})
                .get("backtest_eval")
            )
            if stage_view:
                stage_view = apply_stage_usage_limits(
                    stage_view,
                    max_assets=cfg.get("planned_max_assets"),
                    max_months=cfg.get("planned_max_months"),
                )
                cfg["_active_stage_view"] = stage_view
        else:
            tprint(
                f"Warning: stage holdout_strategy_eval not found in materialized_views"
            )
    except Exception as e:
        tprint(f"Slice plan loading failed: {e}")

    feature_run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    run_id = str(
        os.environ.get("EPM_OUTPUT_RUN_ID", "")
        or cfg.get("output_run_id")
        or feature_run_id
    ).strip()
    if not run_id:
        run_id = feature_run_id
    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    if not os.path.exists(state_file):
        tprint(
            f"ERROR: Trained state not found at {state_file}. Run 'train' mode first."
        )
        return

    tprint(f"Backtest mode. ts_sig={ts_sig}")
    if store is None:
        store = make_ohlcv_store(cfg)
    run_backtest_step(ts_sig, None, cfg, store, state_file)
    tprint("BACKTEST PIPELINE COMPLETE")
    _maintenance_checkpoint("backtest:end")


def run_inference_backtest(cfg, ts_override=None, store=None):
    """Run inference-aligned walk-forward backtest on unseen holdout periods."""
    _maintenance_checkpoint("inference_backtest:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    # Slice Plan Injection
    try:
        slice_plan = load_or_build_slice_plan(
            cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
        )
        if "holdout_strategy_eval" in slice_plan.get("materialized_views", {}):
            stage_view = (
                slice_plan["materialized_views"]["holdout_strategy_eval"]
                .get("sub_views", {})
                .get("backtest_eval")
            )
            if stage_view:
                stage_view = apply_stage_usage_limits(
                    stage_view,
                    max_assets=cfg.get("planned_max_assets"),
                    max_months=cfg.get("planned_max_months"),
                )
                cfg["_active_stage_view"] = stage_view
        else:
            tprint(
                f"Warning: stage holdout_strategy_eval not found in materialized_views"
            )
    except Exception as e:
        tprint(f"Slice plan loading failed: {e}")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    import os

    # Check for trained state
    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    if not os.path.exists(state_file):
        tprint(
            f"ERROR: Trained state not found at {state_file}. Run 'train' mode first."
        )
        return

    tprint(f"Inference backtest mode. ts_sig={ts_sig}")

    # Load trained state
    import pickle

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    # Extract necessary components
    if store is None:
        store = make_ohlcv_store(cfg)

    # Load panel data
    tprint("Loading panel data...")
    panel, symbols = store.load_panel(
        symbols=state.get("symbols", None),
        start_ts=None,
        end_ts=None,
    )
    if panel is None:
        tprint("ERROR: Failed to load panel data.")
        return

    # Load features
    tprint("Loading features...")
    from extreme_price_movements.data_store import load_features_selected
    from extreme_price_movements.slice_plan_store import load_features_for_stage_or_all

    feats = load_features_for_stage_or_all(
        cfg,
        ts_sig,
        root_dir=cfg["data_root"],
        symbols=symbols,
    )

    # Load mask params by mode
    tprint("Loading mask params by mode...")
    from extreme_price_movements.offline_optimisers import (
        apply_offline_optimizer_best_params,
    )

    mask_params_by_mode = dict(cfg.get("candidate_mask_params_by_mode", {}) or {})
    if not mask_params_by_mode:
        # Try to load from offline optimizer results
        mask_params = apply_offline_optimizer_best_params(cfg)
        mask_params_by_mode = dict(
            mask_params.get("candidate_mask_params_by_mode", {}) or {}
        )

    # Load strategy exit params
    strategy_exit_params = dict(
        cfg.get("strategy_exit_params", cfg.get("bucket_exit_params", {})) or {}
    )

    # Load trades from state or backtest results
    tprint("Loading trade candidates...")
    trades = state.get("trades")
    if trades is None:
        # Try to load from backtest results
        backtest_file = os.path.join(
            cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
        )
        if os.path.exists(backtest_file):
            import pandas as pd

            trades = pd.read_csv(backtest_file)
        else:
            tprint("ERROR: No trades found in state or backtest_results.csv")
            return

    # Run inference backtest
    tprint("Running inference backtest...")
    from extreme_price_movements.inference_backtest import (
        InferenceBacktestConfig,
        run_inference_backtest,
    )
    from extreme_price_movements.periods_symbols_management import SlicePlannerConfig

    # Configure inference backtest
    ib_config = InferenceBacktestConfig(
        fee_round_trip_pct=cfg.get("round_trip_fee_pct", 0.3),
        top_fracs=tuple(
            cfg.get("inference_backtest_top_fracs", (0.10, 0.20, 0.30, 0.40))
        ),
        annual_days=365,
        sizing_mode=cfg.get("inference_backtest_sizing_mode", "linear"),
        base_position_size=cfg.get("inference_backtest_base_position_size", 1.0),
        default_limit_offset_bps=cfg.get(
            "inference_backtest_default_limit_offset_bps", 0.0
        ),
    )

    # Use SlicePlanner for unseen holdout periods
    planner_cfg = SlicePlannerConfig.fast_defaults()

    results = run_inference_backtest(
        trades=trades,
        panel=panel,
        feats=feats,
        mask_params_by_mode=mask_params_by_mode,
        strategy_exit_params=strategy_exit_params,
        config=ib_config,
        planner_cfg=planner_cfg,
    )

    # Save results
    tprint("Saving inference backtest results...")
    reports_root = cfg.get("reports_root", "reports")
    os.makedirs(reports_root, exist_ok=True)
    output_file = os.path.join(reports_root, f"inference_backtest_{run_id}.json")

    import json

    # Convert numpy types to serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    tprint(f"Inference backtest results saved to {output_file}")
    tprint(f"Results: {json.dumps(serializable_results, indent=2)}")
    tprint("INFERENCE BACKTEST PIPELINE COMPLETE")
    _maintenance_checkpoint("inference_backtest:end")


def run_train(cfg, ts_override=None, base_only=False, meta_only=False, store=None):
    _maintenance_checkpoint("train:start")
    _backend = str(
        cfg.get("model_backend")
        or os.getenv("EPM_MODEL_BACKEND", "")
        or os.getenv("EPM_TRAINING_MODEL_BACKEND", "")
        or "lgbm_pipeline"
    ).strip().lower()
    _backend_aliases = {
        "ebm": "ebm_on_lgbm",
        "ebm_only": "ebm_on_lgbm",
        "ebm_on_lgbm_only": "ebm_on_lgbm",
        "lgbm": "lgbm_pipeline",
        "lgbm_stability": "lgbm_pipeline",
        "lgbm_stability_pipeline": "lgbm_pipeline",
    }
    _backend = _backend_aliases.get(_backend, _backend)
    cfg["model_backend"] = _backend
    cfg["base_model_backend"] = _backend
    tprint(f"Train backend: {_backend}")
    _base_hpo_trials_env = os.getenv("EPM_BASE_HPO_TRIALS")
    if _base_hpo_trials_env:
        try:
            cfg["base_hpo_n_trials"] = int(_base_hpo_trials_env)
            tprint(f"Base override: base_hpo_n_trials={cfg['base_hpo_n_trials']}")
        except Exception as e:
            tprint(
                f"WARNING: invalid EPM_BASE_HPO_TRIALS={_base_hpo_trials_env!r}: {e}"
            )
    _base_max_strats_env = os.getenv("EPM_BASE_MAX_STRATEGY_IDS")
    if _base_max_strats_env:
        try:
            cfg["base_max_strategy_ids"] = int(_base_max_strats_env)
            tprint(
                f"Base override: base_max_strategy_ids={cfg['base_max_strategy_ids']}"
            )
        except Exception as e:
            tprint(
                f"WARNING: invalid EPM_BASE_MAX_STRATEGY_IDS={_base_max_strats_env!r}: {e}"
            )
    _base_strategy_ids_env = os.getenv("EPM_BASE_STRATEGY_IDS", "")
    _merge_existing_base_env = str(os.getenv("EPM_MERGE_EXISTING_BASE_MODELS", "")).strip().lower()
    if _merge_existing_base_env in {"1", "true", "yes", "on"}:
        cfg["merge_existing_base_models"] = True
        tprint("Base override: merge_existing_base_models=True")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found. Run feature_generation first.")
        return
    source_ts_sig = (
        _resolve_ts_sig(cfg, cfg.get("artifact_source_run_id"))
        if cfg.get("artifact_source_run_id")
        else ts_sig
    )
    if source_ts_sig is None:
        source_ts_sig = ts_sig

    # Slice Plan Injection
    try:
        slice_plan = _load_training_slice_plan(cfg, source_ts_sig)
        _attach_feature_availability_policy_view(cfg, slice_plan, source_ts_sig)
        if "train_base" in slice_plan.get("materialized_views", {}):
            stage_view = slice_plan["materialized_views"]["train_base"]
            stage_view = apply_stage_usage_limits(
                stage_view,
                max_assets=cfg.get("planned_max_assets"),
                max_months=cfg.get("planned_max_months"),
            )
            stage_view = _maybe_extend_training_stage_to_latest(
                stage_view,
                stage_name="train_base",
            )
            cfg["_active_stage_view"] = stage_view
        else:
            tprint(f"Warning: stage train_base not found in materialized_views")
    except Exception as e:
        tprint(f"Slice plan loading failed: {e}")

    tprint(f"Train mode. ts_sig={ts_sig} base_only={base_only} meta_only={meta_only}")
    _load_mask_params_by_mode(cfg)
    if _base_strategy_ids_env.strip():
        requested_ids = [
            s.strip() for s in _base_strategy_ids_env.split(",") if s.strip()
        ]
        selected_strategies, missing_ids = _select_explicit_strategies(
            cfg,
            requested_ids,
            env_label="EPM_BASE_STRATEGY_IDS",
        )
        if missing_ids:
            msg = (
                "EPM_BASE_STRATEGY_IDS requested strategies not found after "
                f"mask-param/source-contract load: {missing_ids}"
            )
            if _truthy_env("EPM_REQUIRE_STRATEGY_ALLOWLIST"):
                raise RuntimeError(msg)
            tprint(f"WARNING: {msg}")
        if selected_strategies:
            cfg["strategies"] = selected_strategies
            tprint(
                "Base override: explicit strategy allowlist active after "
                f"mask-param/source-contract load; selected {len(selected_strategies)}/"
                f"{len(requested_ids)} strategies"
            )
        else:
            msg = (
                "EPM_BASE_STRATEGY_IDS matched no configured/source-contract "
                "strategies; falling back to configured strategy list"
            )
            if _truthy_env("EPM_REQUIRE_STRATEGY_ALLOWLIST"):
                raise RuntimeError(msg)
            tprint(f"WARNING: {msg}")

    # TP/SL optimisation happens during label generation (see training.generate_label_datasets).
    # Check if labels already exist before refreshing to avoid unnecessary recomputation.
    if store is None:
        store = make_ohlcv_store(cfg)
    if not _label_artifacts_ready(cfg, source_ts_sig):
        tprint(
            "ERROR: Label artifacts are missing. Run 'labels' mode first to generate them."
        )
        return

    if not meta_only:
        if bool(cfg.get("base_model_race_extratrees_enabled", False)):
            tprint("STEP: BASE HPO")
            try:
                run_base_hpo_step(ts_sig, cfg)
            except Exception as e:
                tprint(f"WARNING: Base HPO failed: {e}")
        else:
            tprint(
                "STEP: BASE HPO skipped "
                "(ExtraTrees base path disabled; RidgeOnLGBM/EBMOnLGBM use all features)."
            )

    state = run_training_step(
        ts_sig,
        cfg,
        store=store,
        margin_symbols=None,
        base_only=base_only,
        meta_only=meta_only,
    )
    if state:
        tprint("TRAINING PIPELINE COMPLETE")

        # Run breakdown diagnostics after base training
        try:
            run_breakdown_diagnostics_integration(cfg, ts_sig)
        except Exception as e:
            tprint(f"WARNING: breakdown diagnostics failed: {e}")
    else:
        tprint("TRAINING PIPELINE FAILED")
    _maintenance_checkpoint("train:end")


def run_risk_opt(
    cfg, ts_override=None, parsed_ts_sig=None, skip_maintenance=False, store=None
):
    if not skip_maintenance:
        _maintenance_checkpoint("risk_opt:start")

    if parsed_ts_sig:
        ts_sig = parsed_ts_sig
    elif ts_override:
        try:
            _ts_str = (
                str(ts_override).split("_v")[0]
                if "_v" in str(ts_override)
                else str(ts_override)
            )
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found.")
            return

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )

    tprint(f"Risk Optimization mode. ts_sig={ts_sig}")
    if store is None:
        store = make_ohlcv_store(cfg)
    run_risk_optimization_step(ts_sig, None, cfg, store, state_file)
    tprint("RISK OPTIMIZATION COMPLETE")

    if not skip_maintenance:
        _maintenance_checkpoint("risk_opt:end")


def run_sizer(cfg, ts_override=None, store=None):
    """Run the simple artifact-backed sizer on meta model OOF predictions."""
    _maintenance_checkpoint("sizer:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    # Slice Plan Injection
    try:
        slice_plan = load_or_build_slice_plan(
            cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
        )
        if "sizer_train" in slice_plan.get("materialized_views", {}):
            stage_view = slice_plan["materialized_views"]["sizer_train"]
            stage_view = apply_stage_usage_limits(
                stage_view,
                max_assets=cfg.get("planned_max_assets"),
                max_months=cfg.get("planned_max_months"),
            )
            stage_view = _maybe_extend_training_stage_to_latest(
                stage_view,
                stage_name="train_meta",
            )
            cfg["_active_stage_view"] = stage_view
        else:
            tprint(f"Warning: stage sizer_train not found in materialized_views")
    except Exception as e:
        tprint(f"Slice plan loading failed: {e}")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    tprint(f"Sizer mode (simple_position_sizer). ts_sig={ts_sig}")
    _load_mask_params_by_mode(cfg)
    result = run_simple_position_sizer_from_artifacts(
        data_root=cfg["data_root"],
        run_id=run_id,
        top_n_strategies=int(cfg.get("simple_sizer_top_n_strategies", 4)),
        use_ridge_head_sizer=True,
        use_et_head_sizer=True,
    )
    if result:
        tprint("SIZER COMPLETE — simple_position_sizer")

        # Generate OOS backtest metrics immediately after sizer training.
        if bool(cfg.get("sizer_run_oos_backtest", True)):
            try:
                tprint("SIZER: running OOS backtest with updated sizer bundle...")
                if store is None:
                    store = make_ohlcv_store(cfg)
                bt_cfg = dict(cfg)
                bt_cfg["sizer_oos_mode"] = True

                # We need to downcast the trades DataFrame to float32 before generating the backtest
                # This is a memory optimization
                trades_path = os.path.join(
                    cfg["data_root"],
                    "artifacts",
                    run_id,
                    "backtest_results.csv",
                )
                if os.path.exists(trades_path):
                    trades = pd.read_csv(trades_path, low_memory=False)
                    trades = _downcast_numeric_frame(trades)
                    trades.to_csv(trades_path, index=False)

                state_file = os.path.join(
                    cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
                )
                run_backtest_step(ts_sig, None, bt_cfg, store, state_file)
                tprint("SIZER: OOS backtest complete.")
            except Exception as e:
                tprint(f"WARNING: sizer OOS backtest failed: {e}")

        # Run breakdown diagnostics after ridge sizer
        try:
            run_breakdown_diagnostics_integration(cfg, ts_sig)
        except Exception as e:
            tprint(f"WARNING: breakdown diagnostics failed: {e}")

        _maintenance_checkpoint("sizer:end")
        return True
    else:
        tprint("SIZER: No results (possibly no meta OOF predictions found)")
        _maintenance_checkpoint("sizer:end")
        return False


def run_trigger_discovery(cfg, ts_override=None):
    """Run Trigger Discovery (Phase 2.75) via mask_optimiser."""
    _maintenance_checkpoint("trigger_discovery:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found for Trigger Discovery.")
        return False

    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    tprint(f"Trigger Discovery (Phase 2.75) mode. ts_sig={ts_str}")

    # Construct args for mask_optimiser
    args = argparse.Namespace()
    args.data_root = cfg["data_root"]
    args.ts = ts_str
    args.features = None  # Automatic discovery in mask_optimiser
    args.perps = bool(cfg.get("use_perps", False))
    args.max_symbols = cfg.get("mask_opt_max_symbols")
    args.lookback_years = float(cfg.get("mask_opt_lookback_years", 1.5))
    args.horizons = ",".join(map(str, cfg.get("horizons", [1, 2, 4, 8])))
    args.modes = "long,short"  # Refactored side-based modes
    args.diverse_count = int(cfg.get("mask_opt_diverse_count", 4))

    try:
        mask_opt.run_mask_optimization_4modes(args)
        tprint("TRIGGER DISCOVERY COMPLETE")
        return True
    except Exception as e:
        tprint(f"ERROR: Trigger Discovery failed: {e}")
        return False
    finally:
        _maintenance_checkpoint("trigger_discovery:end")


def _run_offset_generation_stage(cfg, ts_sig):
    """Run simple offset generation before policy optimisation."""
    try:
        from extreme_price_movements.policy_optimiser import (
            _load_best_strategy,
            load_meta_oof_predictions,
            load_trade_outcomes,
            resolve_optimised_selection_frac,
        )
        from extreme_price_movements.simple_offset_generator import (
            run_simple_offset_generator_from_sizer,
        )

        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        selected = _load_best_strategy(cfg.get("data_root", "data"), run_id)
        sid = selected.get("strategy_id", "")
        if not sid:
            tprint("OFFSET GENERATION: no selected strategy found; skipping")
            return None
        meta = load_meta_oof_predictions(cfg.get("data_root", "data"), run_id)
        key = next((k for k in meta.keys() if sid in k), None)
        if key is None:
            scored_keys = []
            for mk, mdf in meta.items():
                if hasattr(mdf, "columns") and "oof_u_hat" in mdf.columns:
                    scored_keys.append(
                        (
                            float(
                                np.nanmean(
                                    np.asarray(mdf["oof_u_hat"], dtype=np.float32)
                                )
                            ),
                            mk,
                        )
                    )
            if scored_keys:
                scored_keys.sort(reverse=True)
                key = scored_keys[0][1]
                tprint(
                    "OFFSET GENERATION: selected strategy missing in meta OOF; "
                    f"falling back to best available meta bucket {key}"
                )
            else:
                key = next(iter(meta.keys()), None)
                if key is None:
                    tprint("OFFSET GENERATION: no meta OOF buckets available; skipping")
                    return None
        outcomes = load_trade_outcomes(cfg.get("data_root", "data"), run_id, meta[key])
        conf = np.asarray(
            outcomes.get("oof_u_hat", pd.Series(np.zeros(len(outcomes)))).values,
            dtype=np.float32,
        )
        frac = resolve_optimised_selection_frac(
            data_root=cfg.get("data_root", "data"),
            run_id=run_id,
            selected=selected,
        )
        k = max(1, int(len(conf) * frac))
        idx = np.argpartition(conf, -k)[-k:]
        sizer_stub = {
            "best_simple_score_": conf,
            "best_simple_score_name_": "Ridge_Head_Sizer",
            "ridge_profit_proxy_table_": pd.DataFrame(
                [
                    {
                        "selection_frac": frac,
                        "wallet_min": 0.05,
                        "wallet_max": 0.15,
                        "sizing_mode": "linear",
                        "is_optimal": True,
                    }
                ]
            ),
            "opt_rets_": np.asarray(
                outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
                dtype=np.float32,
            )[idx],
            "opt_ts_": np.asarray(
                outcomes.get("timestamp", pd.Series(np.arange(len(outcomes)))).values
            )[idx],
        }
        return run_simple_offset_generator_from_sizer(
            sizer_results=sizer_stub,
            trade_outcomes=outcomes,
            cost_pct=float(cfg.get("ridge_cost_pct", 0.003)),
        )
    except Exception as exc:
        tprint(f"WARNING: offset generation stage failed: {exc}")
        return None


def run_all(cfg, ts_override=None):
    """Run download -> features -> train (includes labels) -> optimise (learn entry) -> sizer -> optimise (sizing) in order.

    Note: 'train' already refreshes labels internally.
    Note: 'optimise' triggers backtest internally if backtest_results.csv is missing,
          then runs the tpsl_optimiser pipeline (TP/SL calibration, loss limiter,
          profit exit, position sizing, holdout evaluation).
    """
    _maintenance_checkpoint("run_all:start")
    # run_download(cfg)  <- User requested only download if explicitly in download mode
    _maintenance_checkpoint("run_all:after_download")

    # Instantiate store once for use across steps
    store = make_ohlcv_store(cfg)

    run_features(cfg, ts_override=ts_override, store=store)
    _maintenance_checkpoint("run_all:after_features")

    if bool(cfg.get("enable_trigger_discovery_stage", False)):
        success = run_trigger_discovery(cfg, ts_override=ts_override)
        if not success:
            tprint("ERROR: Trigger Discovery stage failed. Aborting pipeline.")
            return
        # RE-LOAD MASK PARAMS: ensure cfg["strategies"] is populated from new winners!
        _load_mask_params_by_mode(cfg)
        _maintenance_checkpoint("run_all:after_trigger_discovery")

    run_train(cfg, ts_override=ts_override, store=store)
    _maintenance_checkpoint("run_all:after_train")

    # 1. Optimise: learn entry policy (fill model + delta) using default sizing/risk
    #    This ensures ridge_sizer sees the correct trade filter.
    tprint("STEP: OPTIMISE (Phase 1 - Entry Policy)")
    success = run_optimise(cfg, ts_override=ts_override, store=store)
    if not success:
        tprint("ERROR: Phase 1 Optimise failed. Aborting pipeline.")
        return
    _maintenance_checkpoint("run_all:after_optimise_phase1")

    # 2. Sizer: learn meta-model weights using the optimized entry policy
    tprint("STEP: SIZER")
    success = run_sizer(cfg, ts_override=ts_override, store=store)
    if not success:
        tprint("ERROR: Sizer step failed. Aborting pipeline.")
        return
    _maintenance_checkpoint("run_all:after_sizer")

    # 3. Optimise: re-run to allow scalar position sizing (Step 40) to use fresh ridge weights
    tprint("STEP: OPTIMISE (Phase 2 - Sizing with Ridge Weights)")
    success = run_optimise(cfg, ts_override=ts_override, store=store)
    if not success:
        tprint("ERROR: Phase 2 Optimise failed. Aborting pipeline.")
        return
    _maintenance_checkpoint("run_all:after_optimise_phase2")

    # 4. Offset generation is intentionally skipped by default.
    # Enable it explicitly via cfg["run_limit_offset_optimiser"] when needed.
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig and bool(cfg.get("run_limit_offset_optimiser", False)):
        tprint("STEP: OFFSET GENERATION")
        _run_offset_generation_stage(cfg, ts_sig)
    elif ts_sig:
        tprint("STEP: OFFSET GENERATION (skipped by config)")

    # 5. Policy optimiser
    if ts_sig:
        tprint("STEP: POLICY OPTIMISER")
        try:
            try:
                slice_plan = load_or_build_slice_plan(
                    cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
                )
                materialized = slice_plan.get("materialized_views", {})
                stage_view = _choose_policy_stage_view(materialized)
                if stage_view is None and "holdout_strategy_eval" in materialized:
                    stage_view = materialized["holdout_strategy_eval"]
                if stage_view is not None:
                    cfg["_active_stage_view"] = apply_stage_usage_limits(
                        stage_view,
                        max_assets=cfg.get("planned_max_assets"),
                        max_months=cfg.get("planned_max_months"),
                    )
            except Exception as _slice_exc:
                tprint(f"Policy optimiser slice plan loading failed: {_slice_exc}")
            run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
            run_policy_optimisation(
                data_root=cfg["data_root"],
                run_id=run_id,
                holdout_frac=float(cfg.get("policy_optimiser_holdout_frac", 0.30)),
                cost_pct=float(
                    cfg.get("ridge_cost_pct", cfg.get("fee_bps", 50.0) / 10000.0)
                ),
                use_offset_optimiser=bool(cfg.get("run_limit_offset_optimiser", False)),
                stage_view=cfg.get("_active_stage_view"),
            )
        except Exception as e:
            tprint(f"WARNING: Policy optimiser failed: {e}")
    _maintenance_checkpoint("run_all:after_policy_optimiser")

    # Holdout multi-metrics: filter strategies by quality gates
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig:
        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        try:
            write_holdout_multi_metrics(cfg["data_root"], run_id)
        except Exception as e:
            tprint(f"WARNING: holdout_multi_metrics write failed: {e}")

    # Final Summary
    ts_sig = _resolve_ts_sig(cfg, ts_override)

    if ts_sig:
        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        import os

        res_path = os.path.join(
            cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
        )
        if os.path.exists(res_path):
            tprint("\n=== FINAL PIPELINE SUMMARY ===")
            try:
                df = pd.read_csv(res_path)
                count = len(df)

                # Gross vs net summary with explicit distinction.
                gross_total = (
                    float(df["gross_ret"].sum())
                    if "gross_ret" in df.columns
                    else float("nan")
                )
                if "net_ret_equity" in df.columns:
                    net_total = float(df["net_ret_equity"].sum())
                elif "pnl" in df.columns:
                    # Legacy backtest output stores net return under `pnl`.
                    net_total = float(df["pnl"].sum())
                else:
                    net_total = float("nan")

                positive_net_share = (
                    float((df["pnl"] > 0).mean())
                    if (count > 0 and "pnl" in df.columns)
                    else float("nan")
                )
                avg_net_per_trade = (
                    (net_total / count)
                    if count > 0 and pd.notna(net_total)
                    else float("nan")
                )

                if pd.notna(gross_total):
                    tprint(f"Total Gross Return: {gross_total:.4f}")
                tprint(
                    f"Total Net Return: {net_total:.4f}"
                    if pd.notna(net_total)
                    else "Total Net Return: n/a"
                )
                tprint(f"Total Trades: {count}")
                if pd.notna(positive_net_share):
                    tprint(f"Positive-Net Share: {positive_net_share:.2%}")
                if pd.notna(avg_net_per_trade):
                    tprint(f"Avg Net Return per Trade: {avg_net_per_trade:.4f}")
                tprint("==============================\n")
            except Exception as e:
                tprint(f"Could not read results for summary: {e}")
    _maintenance_checkpoint("run_all:end")


def run_train_meta(cfg, ts_override=None, store=None):
    """Re-run only meta model training, reusing existing base models."""
    _maintenance_checkpoint("train_meta:start")
    _backend = str(
        cfg.get("model_backend")
        or os.getenv("EPM_MODEL_BACKEND", "")
        or os.getenv("EPM_TRAINING_MODEL_BACKEND", "")
        or "lgbm_pipeline"
    ).strip().lower()
    _backend_aliases = {
        "ebm": "ebm_on_lgbm",
        "ebm_only": "ebm_on_lgbm",
        "ebm_on_lgbm_only": "ebm_on_lgbm",
        "lgbm": "lgbm_pipeline",
        "lgbm_stability": "lgbm_pipeline",
        "lgbm_stability_pipeline": "lgbm_pipeline",
    }
    _backend = _backend_aliases.get(_backend, _backend)
    cfg["model_backend"] = _backend
    cfg["meta_train_regression_bucket_model"] = False
    cfg.setdefault("meta_train_q20_regression", False)
    cfg["meta_train_q20_regression"] = False
    cfg["meta_train_calibration_reg"] = False
    cfg["meta_train_aligned_mae_heads"] = False
    cfg["meta_train_aligned_mfe_heads"] = False
    cfg["meta_train_aux_heads"] = False
    cfg["meta_clf_enabled"] = True
    _meta_top_frac = float(os.getenv("EPM_META_TOP_FRAC", "0.40"))
    cfg["meta_clf_top_frac"] = _meta_top_frac
    cfg["meta_move_top_frac"] = _meta_top_frac
    cfg["meta_trade_topx_values"] = [40]
    _base_target_env = (
        str(os.getenv("EPM_META_TRAIN_BASE_TARGET_CLF_HEAD", "0")).strip().lower()
    )
    cfg["meta_train_base_target_clf_head"] = _base_target_env not in {
        "0",
        "false",
        "no",
        "off",
    }
    _tbm_env = str(os.getenv("EPM_META_TRAIN_TBM_CLF_HEAD", "1")).strip().lower()
    cfg["meta_train_tbm_clf_head"] = _tbm_env not in {
        "0",
        "false",
        "no",
        "off",
    }
    cfg["meta_train_early_invalidation_head"] = False
    _correctness_env = (
        str(os.getenv("EPM_META_TRAIN_CORRECTNESS_CLF_HEAD", "0")).strip().lower()
    )
    cfg["meta_train_correctness_clf_head"] = _correctness_env not in {
        "0",
        "false",
        "no",
        "off",
    }
    cfg["meta_model_backend"] = (
        "lgbm_pipeline" if _backend == "lgbm_pipeline" else "ebm_on_lgbm_only"
    )
    cfg["meta_training_pipeline_version"] = (
        "lgbm_pipeline" if _backend == "lgbm_pipeline" else "legacy"
    )
    cfg["meta_run_pre_risk_optimisation"] = False
    _enabled_heads = []
    if cfg["meta_train_base_target_clf_head"]:
        _enabled_heads.append("base-target")
    if cfg["meta_train_correctness_clf_head"]:
        _enabled_heads.append("base-correctness")
    if cfg["meta_train_tbm_clf_head"]:
        _enabled_heads.append("TBM")
    _head_msg = (
        ", ".join(_enabled_heads) + " classifier head(s)"
        if _enabled_heads
        else "no classifier heads"
    )
    tprint(
        f"Meta training backend={cfg['meta_model_backend']}: "
        "regression/XGB/Ridge heads disabled; "
        f"{_head_msg} enabled; top fraction={_meta_top_frac:.0%}."
    )
    _meta_hpo_trials_env = os.getenv("EPM_META_HPO_TRIALS")
    if _meta_hpo_trials_env:
        try:
            cfg["meta_hpo_trials"] = int(_meta_hpo_trials_env)
            tprint(f"Meta override: meta_hpo_trials={cfg['meta_hpo_trials']}")
        except Exception as e:
            tprint(
                f"WARNING: invalid EPM_META_HPO_TRIALS={_meta_hpo_trials_env!r}: {e}"
            )
    _meta_strategy_ids_env = os.getenv("EPM_META_STRATEGY_IDS", "")
    _meta_max_strats_env = os.getenv("EPM_META_MAX_STRATEGY_IDS")
    if _meta_max_strats_env:
        try:
            cfg["meta_max_strategy_ids"] = int(_meta_max_strats_env)
            tprint(
                f"Meta override: meta_max_strategy_ids={cfg['meta_max_strategy_ids']}"
            )
        except Exception as e:
            tprint(
                f"WARNING: invalid EPM_META_MAX_STRATEGY_IDS={_meta_max_strats_env!r}: {e}"
            )
    _meta_clf_enabled_env = os.getenv("EPM_META_CLF_ENABLED")
    if _meta_clf_enabled_env is not None:
        _v = str(_meta_clf_enabled_env).strip().lower()
        if _v in {"1", "true", "yes", "on"}:
            cfg["meta_clf_enabled"] = True
        elif _v in {"0", "false", "no", "off"}:
            cfg["meta_clf_enabled"] = False
        else:
            tprint(
                f"WARNING: invalid EPM_META_CLF_ENABLED={_meta_clf_enabled_env!r}; "
                "expected one of 1/0/true/false/yes/no/on/off"
            )
        tprint(f"Meta override: meta_clf_enabled={bool(cfg.get('meta_clf_enabled'))}")
    _meta_q20_reg_env = os.getenv("EPM_META_TRAIN_Q20_REGRESSION")
    if _meta_q20_reg_env is not None:
        _v = str(_meta_q20_reg_env).strip().lower()
        if _v in {"1", "true", "yes", "on"}:
            cfg["meta_train_q20_regression"] = True
        elif _v in {"0", "false", "no", "off"}:
            cfg["meta_train_q20_regression"] = False
        else:
            tprint(
                f"WARNING: invalid EPM_META_TRAIN_Q20_REGRESSION={_meta_q20_reg_env!r}; "
                "expected one of 1/0/true/false/yes/no/on/off"
            )
        tprint(
            "Meta override: "
            f"meta_train_q20_regression={bool(cfg.get('meta_train_q20_regression'))}"
        )
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return
    source_ts_sig = (
        _resolve_ts_sig(cfg, cfg.get("artifact_source_run_id"))
        if cfg.get("artifact_source_run_id")
        else ts_sig
    )
    if source_ts_sig is None:
        source_ts_sig = ts_sig

    # Slice Plan Injection
    try:
        slice_plan = _load_training_slice_plan(cfg, source_ts_sig)
        _attach_feature_availability_policy_view(cfg, slice_plan, source_ts_sig)
        if "train_meta" in slice_plan.get("materialized_views", {}):
            stage_view = slice_plan["materialized_views"]["train_meta"]
            stage_view = apply_stage_usage_limits(
                stage_view,
                max_assets=cfg.get("planned_max_assets"),
                max_months=cfg.get("planned_max_months"),
            )
            cfg["_active_stage_view"] = stage_view
        else:
            tprint(f"Warning: stage train_meta not found in materialized_views")
    except Exception as e:
        tprint(f"Slice plan loading failed: {e}")

    _load_mask_params_by_mode(cfg)
    if _meta_strategy_ids_env.strip():
        requested_ids = [
            s.strip() for s in _meta_strategy_ids_env.split(",") if s.strip()
        ]
        selected_strategies, missing_ids = _select_explicit_strategies(
            cfg,
            requested_ids,
            env_label="EPM_META_STRATEGY_IDS",
        )
        if missing_ids:
            msg = (
                "EPM_META_STRATEGY_IDS requested strategies not found after "
                f"mask-param/source-contract load: {missing_ids}"
            )
            if _truthy_env("EPM_REQUIRE_STRATEGY_ALLOWLIST"):
                raise RuntimeError(msg)
            tprint(f"WARNING: {msg}")
        if selected_strategies:
            cfg["strategies"] = selected_strategies
            tprint(
                "Meta override: explicit strategy allowlist active after "
                f"mask-param/source-contract load; selected {len(selected_strategies)}/"
                f"{len(requested_ids)} strategies"
            )
        else:
            msg = (
                "EPM_META_STRATEGY_IDS matched no configured/source-contract "
                "strategies; falling back to configured strategy list"
            )
            if _truthy_env("EPM_REQUIRE_STRATEGY_ALLOWLIST"):
                raise RuntimeError(msg)
            tprint(f"WARNING: {msg}")
    _meta_max_strategy_ids = int(cfg.get("meta_max_strategy_ids", 0) or 0)
    if _meta_max_strategy_ids > 0:
        from extreme_price_movements.strategy_registry import get_strategies

        _strategies = get_strategies(cfg)
        if len(_strategies) > _meta_max_strategy_ids:
            cfg["strategies"] = _strategies[:_meta_max_strategy_ids]
            tprint(
                "Meta training: limiting all train_meta stages, including risk "
                f"optimisation, to first {_meta_max_strategy_ids} strategies"
            )
    from extreme_price_movements.main import train_daily_meta

    if store is None:
        store = make_ohlcv_store(cfg)

    if bool(cfg.get("meta_run_pre_risk_optimisation", False)):
        tprint("Optimising TP:SL before meta-training...")
        try:
            run_risk_opt(cfg, parsed_ts_sig=ts_sig, skip_maintenance=True, store=store)
        except Exception as _e_risk:
            tprint(
                f"WARNING: risk optimisation failed ({_e_risk}); proceeding with existing barrier params."
            )
    else:
        tprint("Meta pre-risk optimisation disabled for EBM-only train_meta.")

    # Meta training reuses local artifacts and does not require a live exchange.
    result = train_daily_meta(ts_sig, None, cfg, store, None)
    if result:
        import gc

        import joblib

        run_id = str(cfg.get("output_run_id") or ts_sig.strftime("%Y%m%d_%H%M%S")).strip()
        models_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "models")
        os.makedirs(models_dir, exist_ok=True)
        meta_state_path = os.path.join(models_dir, "model_state_meta.pkl")
        tmp_meta_state_path = f"{meta_state_path}.tmp"

        joblib.dump(result, tmp_meta_state_path)
        # Verify before replacing the last usable state. An interrupted joblib dump
        # leaves a truncated pickle, which is worse than keeping the prior model.
        joblib.load(tmp_meta_state_path)
        os.replace(tmp_meta_state_path, meta_state_path)
        tprint(f"Meta model state saved atomically to {meta_state_path} using joblib")
        _write_meta_artifact_policy_oos_provenance(
            cfg,
            run_id=run_id,
            meta_state_path=meta_state_path,
        )

        # Free memory before moving on
        del result
        gc.collect()

        # NOTE: Breakdown diagnostics removed here as they require the ridge sizer
        # to represent the final trading policy correctly.
        # Meta-layer metrics (AUC, Lift, IC) are logged naturally during train_daily_meta.

        tprint("TRAIN_META PIPELINE COMPLETE")
    else:
        tprint("TRAIN_META PIPELINE FAILED")
    _maintenance_checkpoint("train_meta:end")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _stage_fit_bounds_from_cfg(cfg: dict):
    stage_view = cfg.get("_active_stage_view") or {}
    return (
        stage_view.get("allowed_start_ts") or stage_view.get("fit_start"),
        stage_view.get("allowed_end_ts") or stage_view.get("fit_end"),
    )


def _active_stage_preserves_exact_plan_filter(cfg: dict) -> bool:
    stage_view = cfg.get("_active_stage_view") or {}
    return not bool(stage_view.get("disable_exact_plan_row_filter"))


def _write_meta_artifact_policy_oos_provenance(
    cfg: dict,
    *,
    run_id: str,
    meta_state_path: str,
) -> None:
    try:
        from extreme_price_movements.policy_oos_provenance import (
            parquet_timestamp_bounds,
            sha256_file,
            slice_plan_fit_bounds,
            write_source_artifact_provenance_manifest,
        )

        run_root = Path(str(cfg["data_root"])) / "artifacts" / str(run_id)
        slice_plan_path = run_root / "slices" / "slice_plan.json"
        meta_feature_contract_path = run_root / "meta_oof" / "meta_feature_contract.json"
        fit_start, fit_end = parquet_timestamp_bounds(
            sorted((run_root / "meta_oof").glob("meta_oof_*.parquet"))
        )
        if (
            (fit_start is None or fit_end is None)
            and _active_stage_preserves_exact_plan_filter(cfg)
        ):
            plan_fit_start, plan_fit_end = slice_plan_fit_bounds(
                slice_plan_path,
                ("meta_model_fit",),
            )
            fit_start = fit_start or plan_fit_start
            fit_end = fit_end or plan_fit_end
        if fit_start is None or fit_end is None:
            fit_start, fit_end = _stage_fit_bounds_from_cfg(cfg)
        feature_contract_hash = (
            sha256_file(meta_feature_contract_path)
            if meta_feature_contract_path.exists()
            else ""
        )
        manifest_path = write_source_artifact_provenance_manifest(
            artifact_path=Path(meta_state_path),
            run_root=run_root,
            slice_plan_path=slice_plan_path,
            source_slice_role="meta_model_fit",
            source_model_fit_start=fit_start,
            source_model_fit_end=fit_end,
            feature_contract_hash=feature_contract_hash,
            generated_from_final_fit_bundle=bool(
                cfg.get("train_full_inference_models", False)
            ),
            extra={
                "generated_by": "train_meta",
                "stage_name": (cfg.get("_active_stage_view") or {}).get(
                    "stage_name", "train_meta"
                ),
                "model_backend": cfg.get("model_backend"),
                "base_models_intermediate_path": str(
                    Path(str(cfg["data_root"]))
                    / "artifacts"
                    / str(run_id)
                    / "base_models_intermediate.pkl"
                ),
            },
        )
        tprint(f"Meta artifact policy-OOS provenance written to {manifest_path}")
    except Exception as exc:
        tprint(f"WARNING: failed to write meta artifact policy-OOS provenance: {exc}")


@contextmanager
def _temporary_env(overrides: Dict[str, Any]):
    old = {}
    try:
        for key, val in overrides.items():
            old[key] = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(val)
        yield
    finally:
        for key, val in old.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _strategy_selection_dir(cfg: dict, run_id: str) -> Path:
    path = Path(cfg["data_root"]) / "artifacts" / run_id / "strategy_selection"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _strategy_id_list(strategies: Iterable[dict]) -> list[str]:
    return [
        str(s.get("strategy_id", "")).strip()
        for s in strategies
        if isinstance(s, dict) and str(s.get("strategy_id", "")).strip()
    ]


def _strategy_side_from_id(strategy_id: str) -> str:
    sid = str(strategy_id).lower()
    if sid.startswith("short"):
        return "short"
    if sid.startswith("long"):
        return "long"
    return "unknown"


def _first_finite(mapping: dict, *keys: str, default: float = np.nan) -> float:
    for key in keys:
        if key not in mapping:
            continue
        try:
            val = float(mapping.get(key))
        except Exception:
            continue
        if np.isfinite(val):
            return val
    return float(default)


def _ndcg_binary_at_k(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    y = np.asarray(y[valid], dtype=float)
    score = np.asarray(score[valid], dtype=float)
    if len(y) < 2:
        return float("nan")
    k = max(1, int(np.ceil(float(frac) * len(y))))
    order = np.argsort(score)[-k:][::-1]
    ideal = np.argsort(y)[-k:][::-1]
    denom = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((2.0**y[order] - 1.0) / denom))
    idcg = float(np.sum((2.0**y[ideal] - 1.0) / denom))
    return float(dcg / max(idcg, 1e-12))


def _synthetic_tp_sl_hit_metrics(
    score: np.ndarray,
    mfe: np.ndarray | None,
    mae: np.ndarray | None,
) -> dict[str, float]:
    if mfe is None or mae is None:
        return {}
    score = np.asarray(score, dtype=float)
    mfe = np.asarray(mfe, dtype=float)
    mae = np.abs(np.asarray(mae, dtype=float))
    n = min(len(score), len(mfe), len(mae))
    if n < 20:
        return {}
    score = score[:n]
    mfe = mfe[:n]
    mae = mae[:n]
    valid = np.isfinite(score) & np.isfinite(mfe) & np.isfinite(mae)
    if int(valid.sum()) < 20:
        return {}
    score_v = score[valid]
    mfe_v = mfe[valid]
    mae_v = mae[valid]
    order = np.argsort(score_v)
    out: dict[str, float] = {}
    for name, tp, sl in (("tp2_sl1", 0.02, 0.01), ("tp3_sl15", 0.03, 0.015)):
        hit = (mfe_v >= tp) & (mae_v < sl)
        for frac in (0.10, 0.30):
            tag = str(int(round(frac * 100)))
            k = max(1, int(np.ceil(frac * len(score_v))))
            idx = order[-k:]
            out[f"hit_rate_{name}_top{tag}"] = (
                float(np.mean(hit[idx])) if len(idx) else float("nan")
            )
    return out


def _score_oof_frame(df: pd.DataFrame) -> dict[str, Any]:
    score_col = next(
        (
            c
            for c in (
                "clf",
                "oof_meta_clf",
                "cv_meta_clf",
                "oof_pred",
                "oof_prob",
                "oof_p_tp",
                "calibrated_score",
            )
            if c in df.columns
        ),
        None,
    )
    y_col = next(
        (c for c in ("y_bin", "target", "label", "outcome") if c in df.columns), None
    )
    if score_col is None or y_col is None:
        return {}
    score = pd.to_numeric(df[score_col], errors="coerce").to_numpy(float)
    y = (pd.to_numeric(df[y_col], errors="coerce").to_numpy(float) >= 0.5).astype(float)
    valid = np.isfinite(score) & np.isfinite(y)
    if int(valid.sum()) < 20:
        return {}
    score_v = score[valid]
    y_v = y[valid]
    order = np.argsort(score_v)
    base_rate = float(np.mean(y_v))
    out: dict[str, Any] = {
        "n_samples": int(len(y_v)),
        "base_rate": base_rate,
        "score_col": score_col,
        "target_col": y_col,
    }
    for frac in (0.10, 0.20, 0.30):
        tag = str(int(round(frac * 100)))
        k = max(1, int(np.ceil(frac * len(y_v))))
        idx = order[-k:]
        hit = float(np.mean(y_v[idx])) if len(idx) else float("nan")
        out[f"hit_rate{tag}"] = hit
        out[f"lift{tag}"] = float(hit / max(base_rate, 1e-12))
        out[f"ndcg{tag}"] = _ndcg_binary_at_k(y_v, score_v, frac)
        ret_col = "y_ret" if "y_ret" in df.columns else ("return" if "return" in df.columns else None)
        if ret_col is not None:
            y_ret = pd.to_numeric(df.loc[valid, ret_col], errors="coerce").to_numpy(float)
            ret_vals = y_ret[idx] if len(idx) else np.asarray([], dtype=float)
            ret_vals = ret_vals[np.isfinite(ret_vals)]
            out[f"mean_ret{tag}"] = float(np.mean(ret_vals)) if len(ret_vals) else float("nan")
    mfe_col = next((c for c in ("mfe_ret", "mfe", "__mfe_ret__", "__mfe__") if c in df.columns), None)
    mae_col = next((c for c in ("mae_ret", "mae", "__mae_ret__", "__mae__") if c in df.columns), None)
    if mfe_col is not None and mae_col is not None:
        out.update(
            _synthetic_tp_sl_hit_metrics(
                score_v,
                pd.to_numeric(df.loc[valid, mfe_col], errors="coerce").to_numpy(float),
                pd.to_numeric(df.loc[valid, mae_col], errors="coerce").to_numpy(float),
            )
        )
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df.loc[valid, "timestamp"], errors="coerce")
        for frac in (0.10, 0.20, 0.30):
            tag = str(int(round(frac * 100)))
            period_hits = {"daily": [], "weekly": [], "monthly": []}
            for freq_name, period_freq in (
                ("daily", "D"),
                ("weekly", "W"),
                ("monthly", "M"),
            ):
                periods = ts.dt.to_period(period_freq).astype(str).to_numpy()
                for _, idx_ser in pd.Series(np.arange(len(y_v))).groupby(periods).groups.items():
                    idx_arr = np.asarray(list(idx_ser), dtype=int)
                    if len(idx_arr) < 20:
                        continue
                    k = max(1, int(np.ceil(frac * len(idx_arr))))
                    top = idx_arr[np.argsort(score_v[idx_arr])[-k:]]
                    period_hits[freq_name].append(float(np.mean(y_v[top])))
                vals = period_hits[freq_name]
                if vals:
                    out[f"hit_rate{tag}_{freq_name}_std"] = (
                        float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    )
        for freq_name, period_freq in (("daily", "D"), ("weekly", "W"), ("monthly", "M")):
            vals = []
            periods = ts.dt.to_period(period_freq).astype(str).to_numpy()
            for _, idx_ser in pd.Series(np.arange(len(y_v))).groupby(periods).groups.items():
                idx_arr = np.asarray(list(idx_ser), dtype=int)
                if len(idx_arr) < 20:
                    continue
                k = max(1, int(np.ceil(0.30 * len(idx_arr))))
                top = idx_arr[np.argsort(score_v[idx_arr])[-k:]]
                vals.append(float(np.mean(y_v[top])))
            if vals:
                out[f"hit_rate30_{freq_name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _load_base_diag_metrics(cfg: dict, run_id: str) -> list[dict[str, Any]]:
    path = Path(cfg["data_root"]) / "artifacts" / run_id / "base_models_intermediate.pkl"
    if not path.exists():
        return []
    try:
        with open(path, "rb") as fp:
            bundle = pickle.load(fp)
    except Exception as exc:
        tprint(f"Strategy selection: could not load base diagnostics: {exc}")
        return []
    rows: list[dict[str, Any]] = []
    for key, diag in dict(bundle.get("alpha_fit_diagnostics", {}) or {}).items():
        if not isinstance(diag, dict):
            continue
        m = re.match(r"^(long|short)_(.+)_H(\d+)$", str(key))
        if not m:
            continue
        side, strategy_id, horizon = m.group(1), m.group(2), int(m.group(3))
        row = {
            "strategy_id": strategy_id,
            "side": side,
            "horizon": horizon,
            "source": "base_models_intermediate.alpha_fit_diagnostics",
            "hit_rate30": _first_finite(diag, "hit_rate30", "precision30", "prec30", "en_hit_rate30"),
            "lift30": _first_finite(diag, "lift30", "en_lift30", "Lift@30"),
            "mean_ret30": _first_finite(
                diag,
                "mean_ret30",
                "mean_return30_gross",
                "mean_return30",
                "top30_mean_ret",
            ),
            "hit_rate_tp2_sl1_top10": _first_finite(
                diag, "hit_rate_tp2_sl1_top10"
            ),
            "hit_rate_tp2_sl1_top30": _first_finite(
                diag, "hit_rate_tp2_sl1_top30"
            ),
            "hit_rate_tp3_sl15_top10": _first_finite(
                diag, "hit_rate_tp3_sl15_top10"
            ),
            "hit_rate_tp3_sl15_top30": _first_finite(
                diag, "hit_rate_tp3_sl15_top30"
            ),
            "lift20": _first_finite(diag, "lift20", "Lift@20"),
            "hit_rate30_std": _first_finite(diag, "hit_rate30_weekly_std", "hit_rate30_daily_std", default=0.0),
            "ic": _first_finite(diag, "ic", "rank_ic", "ic_cs", "mean_ic"),
            "ic_stability": _first_finite(diag, "ic_stability", "ir_weekly", default=0.0),
            "degenerate": bool(diag.get("degenerate", False)),
            "fit_status": diag.get("fit_status"),
        }
        row["selection_score"] = _strategy_metric_score(row)
        rows.append(row)
    return rows


def _strategy_horizon_map(strategies: Sequence[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for strat in strategies:
        sid = str(strat.get("strategy_id", "")).strip()
        if not sid:
            continue
        try:
            horizons = strategy_runtime_horizons(strat, {})
            if horizons:
                out[sid] = int(horizons[0])
                continue
        except Exception:
            pass
        try:
            out[sid] = int(strat.get("source_horizon", 5))
        except Exception:
            out[sid] = 5
    return out


def _load_oof_strategy_metrics(
    cfg: dict,
    run_id: str,
    strategies: Sequence[dict],
    *,
    layer: str,
) -> list[dict[str, Any]]:
    artifacts = Path(cfg["data_root"]) / "artifacts" / run_id
    horizon_by_id = _strategy_horizon_map(strategies)
    rows: list[dict[str, Any]] = []
    for sid in _strategy_id_list(strategies):
        paths: list[Path] = []
        if layer == "base":
            h = int(horizon_by_id.get(sid, 5))
            paths = [artifacts / "oof" / f"oof_{sid}_H{h}.parquet"]
        else:
            meta_dir = artifacts / "meta_oof"
            paths = sorted(meta_dir.glob(f"meta_oof_{sid}*_clf.parquet"))
        for path in paths:
            if not path.exists():
                continue
            try:
                metrics = _score_oof_frame(pd.read_parquet(path))
            except Exception as exc:
                tprint(f"Strategy selection: could not score {path.name}: {exc}")
                metrics = {}
            if not metrics:
                continue
            metrics.update(
                {
                    "strategy_id": sid,
                    "side": _strategy_side_from_id(sid),
                    "horizon": int(horizon_by_id.get(sid, 0) or 0),
                    "source": str(path),
                    "layer": layer,
                }
            )
            metrics["hit_rate30_std"] = _first_finite(
                metrics,
                "hit_rate30_weekly_std",
                "hit_rate30_daily_std",
                "hit_rate30_monthly_std",
                default=0.0,
            )
            metrics["selection_score"] = _strategy_metric_score(metrics)
            rows.append(metrics)
            break
    if layer == "base":
        seen = {r["strategy_id"] for r in rows}
        allowed = set(_strategy_id_list(strategies))
        for row in _load_base_diag_metrics(cfg, run_id):
            if row["strategy_id"] in allowed and row["strategy_id"] not in seen:
                rows.append(row)
    return rows


def _strategy_metric_score(row: dict[str, Any]) -> float:
    hit30 = _first_finite(row, "hit_rate30", "precision30", "prec30", default=0.0)
    mean_ret30 = _first_finite(
        row,
        "mean_ret30",
        "mean_return30_gross",
        "mean_return30",
        "top30_mean_ret",
        default=0.0,
    )
    ndcg30 = _first_finite(row, "ndcg30", default=0.0)
    std30 = _first_finite(row, "hit_rate30_std", default=0.0)
    score = 0.20 * hit30 + 0.20 * mean_ret30 + 0.50 * ndcg30 - 0.25 * std30
    if bool(row.get("degenerate", False)):
        score -= 1e6
    return float(score if np.isfinite(score) else -1e12)


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _select_top_fraction(rows: list[dict[str, Any]], frac: float) -> list[str]:
    ranked = sorted(rows, key=lambda r: float(r.get("selection_score", -1e12)), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for side in ("long", "short"):
        side_rows = [r for r in ranked if str(r.get("side", "")) == side]
        n_side = max(1, int(np.ceil(float(frac) * len(side_rows)))) if side_rows else 0
        for row in side_rows[:n_side]:
            sid = str(row["strategy_id"])
            if sid not in seen:
                selected.append(row)
                seen.add(sid)
    for row in ranked:
        sid = str(row["strategy_id"])
        if sid not in seen and str(row.get("side", "")) not in {"long", "short"}:
            selected.append(row)
            seen.add(sid)
    selected.sort(key=lambda r: float(r.get("selection_score", -1e12)), reverse=True)
    return [str(r["strategy_id"]) for r in selected]


def _strategy_period_return_series(row: dict[str, Any]) -> dict[str, pd.Series]:
    source = str(row.get("source", "") or "")
    path = Path(source)
    if not path.exists() or path.suffix.lower() != ".parquet":
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if "timestamp" not in df.columns or "y_ret" not in df.columns:
        return {}
    score_col = "oof_prob" if "oof_prob" in df.columns else None
    if score_col is None:
        score_candidates = [
            c
            for c in df.columns
            if c.startswith("meta_") or c.startswith("oof_") or c.endswith("_prob")
        ]
        score_col = score_candidates[0] if score_candidates else None
    if score_col is None:
        return {}
    tmp = df[["timestamp", "y_ret", score_col]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
    tmp["y_ret"] = pd.to_numeric(tmp["y_ret"], errors="coerce")
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp = tmp.dropna(subset=["timestamp", "y_ret", score_col])
    if len(tmp) < 20:
        return {}
    out: dict[str, pd.Series] = {}
    for name, freq in (("daily", "D"), ("weekly", "W"), ("monthly", "M")):
        values: dict[str, float] = {}
        for period, part in tmp.groupby(tmp["timestamp"].dt.to_period(freq).astype(str)):
            if len(part) < 5:
                continue
            k = max(1, int(np.ceil(0.30 * len(part))))
            top = part.nlargest(k, score_col)
            values[str(period)] = float(np.mean(top["y_ret"]))
        if len(values) >= 2:
            out[name] = pd.Series(values, dtype=np.float64)
    return out


def _max_same_side_period_corr(
    row: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    period_cache: dict[str, dict[str, pd.Series]],
) -> float:
    sid = str(row.get("strategy_id", ""))
    side = str(row.get("side", ""))
    row_series = period_cache.setdefault(sid, _strategy_period_return_series(row))
    if not row_series:
        return 0.0
    max_corr = 0.0
    for other in selected:
        if str(other.get("side", "")) != side:
            continue
        other_sid = str(other.get("strategy_id", ""))
        other_series = period_cache.setdefault(
            other_sid, _strategy_period_return_series(other)
        )
        for freq in ("daily", "weekly", "monthly"):
            a = row_series.get(freq)
            b = other_series.get(freq)
            if a is None or b is None:
                continue
            aligned = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(aligned) < 2:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if np.isfinite(corr):
                max_corr = max(max_corr, abs(corr))
    return float(max_corr)


def _select_portfolio_pool(
    rows: list[dict[str, Any]],
    *,
    pool_n: int,
    min_side_share: float,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: float(r.get("selection_score", -1e12)), reverse=True)
    by_side = {
        "long": [r for r in ranked if str(r.get("side")) == "long"],
        "short": [r for r in ranked if str(r.get("side")) == "short"],
    }
    min_per_side = int(np.floor(float(pool_n) * float(min_side_share)))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    period_cache: dict[str, dict[str, pd.Series]] = {}
    corr_penalty = float(os.environ.get("EPM_STRATEGY_SELECTION_SAME_SIDE_CORR_PENALTY", "0.35"))

    def _with_portfolio_score(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        max_corr = _max_same_side_period_corr(out, selected, period_cache)
        out["same_side_period_corr_penalty"] = float(max_corr)
        out["portfolio_selection_score"] = float(
            row.get("selection_score", -1e12)
        ) - corr_penalty * float(max_corr)
        return out

    for side in ("long", "short"):
        for row in by_side[side][:min_per_side]:
            sid = str(row["strategy_id"])
            if sid not in used:
                selected.append(_with_portfolio_score(row))
                used.add(sid)
    while len(selected) < int(pool_n):
        available = [r for r in ranked if str(r["strategy_id"]) not in used]
        if not available:
            break
        scored = [_with_portfolio_score(r) for r in available]
        row = max(scored, key=lambda r: float(r.get("portfolio_selection_score", -1e12)))
        if len(selected) >= int(pool_n):
            break
        sid = str(row["strategy_id"])
        if sid not in used:
            selected.append(row)
            used.add(sid)
    selected.sort(key=lambda r: float(r.get("portfolio_selection_score", r.get("selection_score", -1e12))), reverse=True)
    return selected[: int(pool_n)]


def _load_policy_winners(cfg: dict, run_id: str) -> list[str]:
    candidates = [
        Path(cfg["data_root"]) / "artifacts" / run_id / "strategy_for_inference.json",
        Path(cfg["data_root"]) / "artifacts" / run_id / "policy_params" / "strategy_for_inference.json",
        Path(cfg["data_root"]) / "artifacts" / run_id / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
    ]
    candidates = [
        candidate
        for path in candidates
        for candidate in mode_file_candidates(path, _market_mode_from_cfg(cfg))
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out = [
            str(row.get("strategy_id", "")).strip()
            for row in payload.get("strategies", [])
            if isinstance(row, dict) and bool(row.get("selected", False))
        ]
        if out:
            return out
    return []


def _selection_profile_env(*, final: bool = False) -> dict[str, Any]:
    lgbm_fast_selection_env = {
        "EPM_LGBM_CV_SPLITS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_CV_SPLITS", "2"),
        "EPM_LGBM_RACE_MAX_ROWS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_RACE_MAX_ROWS", "30000"),
        "EPM_LGBM_UNIVARIATE_MAX_ROWS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_UNIVARIATE_MAX_ROWS", "6000"),
        "EPM_LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_UNIVARIATE_ROW_SUBSAMPLE_FRAC", "0.35"),
        "EPM_LGBM_RELIEF_REPEATS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_RELIEF_REPEATS", "1"),
        "EPM_LGBM_RELIEF_RESCUE_MAX": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_RELIEF_RESCUE_MAX", "30"),
        "EPM_LGBM_RELIEF_RESCUE_MIN": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_RELIEF_RESCUE_MIN", "8"),
        "EPM_LGBM_RELIEF_ANCHOR_MAX_ROWS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_RELIEF_ANCHOR_MAX_ROWS", "256"),
        "EPM_LGBM_RELIEF_NEIGHBOR_CANDIDATES": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_RELIEF_NEIGHBOR_CANDIDATES", "768"),
        "EPM_LGBM_DIRECTION_MAX_ROWS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_DIRECTION_MAX_ROWS", "1500"),
        "EPM_LGBM_MAX_ROUNDS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_MAX_ROUNDS", "6"),
        "EPM_LGBM_MIN_FEATURES": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_MIN_FEATURES", "24"),
        "EPM_LGBM_SELECTED_FEATURES_MIN": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_SELECTED_FEATURES_MIN", "40"),
        "EPM_LGBM_SELECTED_FEATURES_MAX": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_SELECTED_FEATURES_MAX", "120"),
        "EPM_LGBM_PERMUTATION_TOP_CONFIGS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_PERMUTATION_TOP_CONFIGS", "1"),
        "EPM_LGBM_PERMUTATION_MAX_FEATURES": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_PERMUTATION_MAX_FEATURES", "20"),
        "EPM_LGBM_PERMUTATION_MAX_ROWS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_PERMUTATION_MAX_ROWS", "5000"),
        "EPM_LGBM_PERMUTATION_REPEATS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_PERMUTATION_REPEATS", "1"),
        "EPM_LGBM_HPO_TRIALS": "0",
        "EPM_LGBM_HPO_MAX_ROWS": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_HPO_MAX_ROWS", "3000"),
        "EPM_LGBM_FINAL_MODEL_COUNT": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_FINAL_MODEL_COUNT", "1"),
        "EPM_LGBM_OOF_DISTILLATION_PASSES": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_OOF_DISTILLATION_PASSES", "0"),
        "EPM_LGBM_MIN_OOF_DISTILLATION_PASSES": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_MIN_OOF_DISTILLATION_PASSES", "1"),
        "EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES": os.environ.get("EPM_STRATEGY_SELECTION_LGBM_META_MIN_OOF_DISTILLATION_PASSES", "1"),
    }
    if final:
        return {
            "EPM_MASK_STRATEGY_TOP_N": os.environ.get("EPM_MASK_STRATEGY_TOP_N", "999"),
            "EPM_EBM_FINAL_FIT_MAX_ROWS": "0",
            "EPM_EBM_FINAL_FIT_METRIC_MAX_ROWS": "0",
            "EPM_EBM_FINAL_UNCERTAINTY_MAX_ROWS": "0",
            "EPM_EBM_OOF_DISTILLATION_PASSES": os.environ.get("EPM_STRATEGY_SELECTION_SELF_DISTILL_ROUNDS", "1"),
            "EPM_BASE_HPO_TRIALS": "0",
            "EPM_META_HPO_TRIALS": "0",
            "EPM_EBM_HPO_TRIALS": "0",
            "EPM_EBM_MAX_ROUNDS": os.environ.get("EPM_EBM_MAX_ROUNDS", "1"),
            "EPM_EBM_PRUNE_MODEL_COUNT": "1",
            "EPM_EBM_ROW_SUBSAMPLE_FRAC": "0.50",
            "EPM_EBM_FINAL_MODEL_COUNT": "1",
            "EPM_EBM_HONEST_EVAL_MIN_MODELS": "1",
            **lgbm_fast_selection_env,
        }
    return {
        "EPM_MASK_STRATEGY_TOP_N": os.environ.get("EPM_MASK_STRATEGY_TOP_N", "999"),
        "EPM_BASE_HPO_TRIALS": "0",
        "EPM_META_HPO_TRIALS": "0",
        "EPM_META_TOP_FRAC": os.environ.get("EPM_STRATEGY_SELECTION_BASE_TOP_FRAC", "0.66"),
        "EPM_EBM_HPO_TRIALS": "0",
        "EPM_EBM_RACE_MAX_ROWS": os.environ.get("EPM_EBM_RACE_MAX_ROWS", "60000"),
        "EPM_EBM_TREE_LGBM_MAX_FIT_ROWS": os.environ.get("EPM_EBM_TREE_LGBM_MAX_FIT_ROWS", "30000"),
        "EPM_EBM_HPO_MAX_ROWS": os.environ.get("EPM_EBM_HPO_MAX_ROWS", "10000"),
        "EPM_EBM_MAX_ROUNDS": os.environ.get("EPM_EBM_MAX_ROUNDS", "1"),
        "EPM_EBM_FINAL_FIT_MAX_ROWS": "0",
        "EPM_EBM_OOF_DISTILLATION_PASSES": os.environ.get("EPM_STRATEGY_SELECTION_SELF_DISTILL_ROUNDS", "1"),
        "EPM_EBM_PRUNE_MODEL_COUNT": "1",
        "EPM_EBM_ROW_SUBSAMPLE_FRAC": "0.50",
        "EPM_EBM_FINAL_MODEL_COUNT": "1",
        "EPM_EBM_HONEST_EVAL_MIN_MODELS": "1",
        **lgbm_fast_selection_env,
    }


def _apply_strategy_selection_no_penalty(cfg: dict) -> None:
    """Disable selector-side penalty terms for strategy-selection audit runs."""
    if os.environ.get("EPM_STRATEGY_SELECTION_NO_PENALTY", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return
    for key in (
        "base_selector_cfg",
        "meta_selector_cfg",
        "aux_mae_selector_cfg",
        "aux_mfe_selector_cfg",
    ):
        sel_cfg = cfg.get(key)
        if not isinstance(sel_cfg, dict):
            continue
        sel_cfg["selector_interaction_corr_penalty"] = False
        sel_cfg["selector_family_penalty"] = False
        sel_cfg["interaction"] = 0.0
    tprint(
        "STRATEGY SELECTION: no-penalty selector mode enabled "
        "(interaction/family selector penalties disabled)."
    )


def _apply_training_no_penalty(cfg: dict) -> None:
    """Disable selector-side penalty terms for explicit training audit runs."""
    env_value = os.environ.get("EPM_TRAINING_NO_PENALTY", "1").strip().lower()
    if env_value in {"0", "false", "no", "n", "off"}:
        return
    for key in (
        "base_selector_cfg",
        "meta_selector_cfg",
        "aux_mae_selector_cfg",
        "aux_mfe_selector_cfg",
    ):
        sel_cfg = cfg.get(key)
        if not isinstance(sel_cfg, dict):
            continue
        sel_cfg["selector_interaction_corr_penalty"] = False
        sel_cfg["selector_family_penalty"] = False
        sel_cfg["interaction"] = 0.0
    tprint(
        "TRAINING: no-penalty selector mode enabled "
        "(interaction/family selector penalties disabled)."
    )


def _run_final_retraining_layers(
    cfg: dict,
    ts_sig: pd.Timestamp,
    strategy_ids: Sequence[str],
    *,
    store=None,
) -> dict[str, Any]:
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    sel_dir = _strategy_selection_dir(cfg, run_id)
    rounds = max(1, int(os.environ.get("EPM_FINAL_RETRAIN_REWEIGHT_ROUNDS", "3")))
    ids_csv = ",".join(str(s) for s in strategy_ids if str(s).strip())
    if not ids_csv:
        return {"status": "skipped", "reason": "no_policy_winners"}
    final_cfg = dict(cfg)
    final_cfg["planned_max_assets"] = None
    final_cfg["planned_max_months"] = None
    final_cfg["sample_weight_opt_enable"] = str(
        os.environ.get("EPM_FINAL_RETRAIN_SAMPLE_WEIGHT_OPT_ENABLE", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    final_cfg["train_full_inference_models"] = True
    final_cfg["strategy_selection_final_retrain"] = True
    final_cfg["final_retrain_folds"] = int(os.environ.get("EPM_FINAL_RETRAIN_FOLDS", "4"))
    final_cfg["final_retrain_sample_multiplier"] = float(
        os.environ.get("EPM_FINAL_RETRAIN_SAMPLE_MULTIPLIER", "2")
    )
    report: dict[str, Any] = {"strategy_ids": list(strategy_ids), "base_rounds": [], "meta_rounds": []}
    env = _selection_profile_env(final=True)
    env.update(
        {
            "EPM_BASE_STRATEGY_IDS": ids_csv,
            "EPM_META_STRATEGY_IDS": ids_csv,
            "EPM_POLICY_STRATEGY_IDS": ids_csv,
        }
    )
    for round_i in range(1, rounds + 1):
        tprint(f"FINAL RETRAIN base round {round_i}/{rounds}")
        round_cfg = dict(final_cfg)
        round_cfg["final_retrain_round"] = round_i
        with _temporary_env(env):
            run_train(round_cfg, ts_override=ts_sig.strftime("%Y%m%d_%H%M%S"), base_only=True, meta_only=False, store=store)
        metrics = _load_oof_strategy_metrics(round_cfg, run_id, round_cfg.get("strategies", []), layer="base")
        _write_metrics_csv(sel_dir / f"final_base_round_{round_i}_metrics.csv", metrics)
        report["base_rounds"].append({"round": round_i, "metrics": metrics})
    for round_i in range(1, rounds + 1):
        tprint(f"FINAL RETRAIN meta round {round_i}/{rounds}")
        round_cfg = dict(final_cfg)
        round_cfg["final_retrain_round"] = round_i
        with _temporary_env(env):
            run_train_meta(round_cfg, ts_override=ts_sig.strftime("%Y%m%d_%H%M%S"), store=store)
        metrics = _load_oof_strategy_metrics(round_cfg, run_id, round_cfg.get("strategies", []), layer="meta")
        _write_metrics_csv(sel_dir / f"final_meta_round_{round_i}_metrics.csv", metrics)
        report["meta_rounds"].append({"round": round_i, "metrics": metrics})
    _write_json(sel_dir / "final_retrain_report.json", report)
    return report


def _run_final_model_fit(
    cfg: dict,
    *,
    ts_override: str | None = None,
) -> None:
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return
    feature_run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    run_id = str(
        os.environ.get("EPM_OUTPUT_RUN_ID", "")
        or cfg.get("output_run_id")
        or feature_run_id
    ).strip()
    if not run_id:
        run_id = feature_run_id
    policy_run_id = str(
        os.environ.get("EPM_POLICY_ARTIFACT_RUN_ID", "")
        or cfg.get("policy_artifact_run_id")
        or run_id
    ).strip()
    explicit_ids = [
        s.strip()
        for s in str(os.environ.get("EPM_FINAL_MODEL_STRATEGY_IDS", "")).split(",")
        if s.strip()
    ]
    strategy_ids = explicit_ids or _load_policy_winners(cfg, policy_run_id)
    if not strategy_ids:
        tprint(
            "ERROR: final_model_fit found no selected policy strategies. "
            "Run simple_policy_optimiser first or set EPM_FINAL_MODEL_STRATEGY_IDS."
        )
        return

    ids_csv = ",".join(strategy_ids)
    fit_cfg = dict(cfg)
    fit_cfg["run_id"] = run_id
    fit_cfg["output_run_id"] = run_id
    label_artifact_run_id = str(
        os.environ.get("EPM_LABEL_ARTIFACT_RUN_ID", "") or policy_run_id
    ).strip()
    fit_cfg["_label_artifact_run_id"] = label_artifact_run_id
    fit_cfg["label_source_run_id"] = label_artifact_run_id
    fit_cfg["feature_source_run_id"] = feature_run_id
    fit_cfg["train_full_inference_models"] = True
    fit_cfg["lgbm_use_native_preset"] = True
    fit_cfg["lgbm_require_native_preset"] = True
    fit_cfg["lgbm_native_preset_source_run_id"] = policy_run_id
    fit_cfg["recency_hpo_use_winner"] = True
    fit_cfg["model_backend"] = "lgbm_pipeline"
    fit_cfg["base_model_backend"] = "lgbm_pipeline"
    fit_cfg["meta_model_backend"] = "lgbm_pipeline"
    fit_cfg["planned_max_assets"] = None
    fit_cfg["planned_max_months"] = None

    env = {
        "EPM_MODEL_BACKEND": "lgbm_pipeline",
        "EPM_LGBM_USE_NATIVE_PRESET": "1",
        "EPM_LGBM_REQUIRE_NATIVE_PRESET": "1",
        "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": policy_run_id,
        "EPM_FEATURE_SOURCE_RUN_ID": feature_run_id,
        "EPM_RECENCY_HPO_USE_WINNER": "1",
        "EPM_LGBM_BASE_LABEL_WEIGHT_HPO": "0",
        "EPM_TRAIN_EXTEND_TO_LATEST": "1",
        "EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER": "1",
        "EPM_BASE_STRATEGY_IDS": ids_csv,
        "EPM_META_STRATEGY_IDS": ids_csv,
        "EPM_LABEL_STRATEGY_IDS": ids_csv,
        "EPM_POLICY_STRATEGY_IDS": ids_csv,
        "EPM_FINAL_MODEL_STRATEGY_IDS": ids_csv,
    }
    registry_dir = (
        Path(fit_cfg["data_root"]) / "artifacts" / policy_run_id / "strategy_registry"
    )
    registry_candidates = sorted(registry_dir.glob("*.csv")) if registry_dir.exists() else []
    if registry_candidates and not os.environ.get("EPM_MASK_STRATEGY_SOURCE_CSV"):
        env["EPM_MASK_STRATEGY_SOURCE_CSV"] = str(registry_candidates[0])
        env["EPM_MASK_STRATEGY_TOP_N"] = str(max(10, len(strategy_ids)))
        env["EPM_MASK_STRATEGY_RANKING_METRIC"] = "stage_e_rank_score"
        env["EPM_REQUIRE_STRATEGY_ALLOWLIST"] = "1"

    out_dir = Path(fit_cfg["data_root"]) / "artifacts" / run_id

    def _require_final_fit_artifacts(stage: str, rel_paths: list[str]) -> None:
        missing = [rel for rel in rel_paths if not (out_dir / rel).exists()]
        if missing:
            raise RuntimeError(
                "final_model_fit failed after "
                f"{stage}: missing required artifacts for run_id={run_id}: "
                + ", ".join(missing)
            )

    tprint(
        "FINAL MODEL FIT START: "
        f"run_id={run_id} feature_run_id={feature_run_id} policy_run_id={policy_run_id} "
        f"label_artifact_run_id={label_artifact_run_id} "
        f"feature_source_run_id={fit_cfg.get('feature_source_run_id')} "
        f"strategies={len(strategy_ids)} native_preset=yes recency_hpo_winner=yes "
        "fit_window=all_available_rows"
    )
    with _temporary_env(env):
        run_train(
            fit_cfg,
            ts_override=feature_run_id,
            base_only=True,
            meta_only=False,
        )
        _require_final_fit_artifacts(
            "train_base",
            ["base_models_intermediate.pkl", "models/trained_state.pkl"],
        )
        try:
            from extreme_price_movements.base_error_archetype_backfill import (
                backfill_artifact_base_error_archetypes,
            )

            _base_error_manifest = backfill_artifact_base_error_archetypes(
                out_dir,
                random_state=int(fit_cfg.get("seed", 42) or 42),
                force=False,
            )
            tprint(
                "FINAL MODEL FIT: base-error archetype OOF backfill complete "
                f"(states={int((_base_error_manifest or {}).get('state_count', 0) or 0)})."
            )
        except Exception as exc:
            tprint(
                "WARNING: final_model_fit base-error archetype backfill failed before "
                f"train_meta: {exc}"
            )
        run_train_meta(fit_cfg, ts_override=feature_run_id)
        _require_final_fit_artifacts(
            "train_meta",
            ["models/model_state_meta.pkl"],
        )

    manifest = {
        "mode": "final_model_fit",
        "run_id": run_id,
        "model_artifact_run_id": run_id,
        "feature_run_id": feature_run_id,
        "policy_artifact_run_id": policy_run_id,
        "label_artifact_run_id": label_artifact_run_id,
        "strategy_ids": strategy_ids,
        "strategy_source": (
            "EPM_FINAL_MODEL_STRATEGY_IDS" if explicit_ids else "policy_winners"
        ),
        "native_preset_source_run_id": policy_run_id,
        "native_preset_required": True,
        "recency_hpo_use_winner": True,
        "train_extend_to_latest": True,
        "disable_exact_plan_row_filter": True,
        "validation_note": (
            "Deployment final fit only. Metrics from these models are not clean OOS "
            "validation because all available rows may be used for fitting."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "final_model_fit_manifest.json", manifest)
    if policy_run_id != run_id:
        policy_dir = Path(fit_cfg["data_root"]) / "artifacts" / policy_run_id
        policy_dir.mkdir(parents=True, exist_ok=True)
        _write_json(policy_dir / "final_model_fit_manifest.json", manifest)
    tprint(
        "FINAL MODEL FIT COMPLETE: "
        f"manifest={out_dir / 'final_model_fit_manifest.json'}"
    )


def run_strategies_selection(cfg, ts_override=None, store=None):
    _maintenance_checkpoint("strategies_selection:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    sel_dir = _strategy_selection_dir(cfg, run_id)
    assets = int(os.environ.get("EPM_STRATEGY_SELECTION_ASSETS", "100"))
    base_top_frac = float(os.environ.get("EPM_STRATEGY_SELECTION_BASE_TOP_FRAC", "0.66"))
    final_pool_n = int(os.environ.get("EPM_STRATEGY_SELECTION_FINAL_POOL_N", "15"))
    inference_max_n = int(os.environ.get("EPM_STRATEGY_SELECTION_INFERENCE_MAX_N", "12"))
    max_per_side = int(os.environ.get("EPM_STRATEGY_SELECTION_MAX_PER_SIDE", "8"))
    min_side_share = float(os.environ.get("EPM_STRATEGY_SELECTION_MIN_SIDE_SHARE", "0.40"))
    policy_trials = int(os.environ.get("EPM_STRATEGY_SELECTION_POLICY_TRIALS", "50"))

    if store is None:
        store = make_ohlcv_store(cfg)

    selection_cfg = dict(cfg)
    selection_cfg["planned_max_assets"] = assets
    selection_cfg["train_full_inference_models"] = False
    selection_cfg["strategy_selection_mode"] = True
    selection_cfg["meta_filter_saved_oof_to_active_symbols"] = True
    selection_cfg["mask_strategy_top_n"] = int(os.environ.get("EPM_MASK_STRATEGY_TOP_N", "999"))
    _apply_strategy_selection_no_penalty(selection_cfg)
    _maybe_set_strategy_selection_mask_source(selection_cfg)
    if assets <= 10:
        selection_cfg["min_train_samples"] = int(
            os.environ.get("EPM_STRATEGY_SELECTION_MIN_TRAIN_SAMPLES", "100")
        )
        selection_cfg["base_min_samples_hard_floor"] = int(
            os.environ.get("EPM_STRATEGY_SELECTION_BASE_MIN_SAMPLES", "200")
        )
        selection_cfg["sample_weight_opt_min_samples"] = int(
            os.environ.get("EPM_STRATEGY_SELECTION_SAMPLE_WEIGHT_MIN_SAMPLES", "50")
        )
    _load_mask_params_by_mode(selection_cfg)
    all_strategies = list(get_strategies(selection_cfg))
    _write_json(sel_dir / "candidate_strategies.json", all_strategies)

    with _temporary_env(_selection_profile_env(final=False)):
        tprint("STRATEGY SELECTION: lightweight train_base")
        run_train(selection_cfg, ts_override=ts_sig.strftime("%Y%m%d_%H%M%S"), base_only=True, meta_only=False, store=store)

    base_metric_strategies = all_strategies
    _base_ids_env = os.environ.get("EPM_BASE_STRATEGY_IDS", "")
    if _base_ids_env.strip():
        _base_ids = {s.strip() for s in _base_ids_env.split(",") if s.strip()}
        base_metric_strategies = [
            s
            for s in all_strategies
            if str(s.get("strategy_id", "")).strip() in _base_ids
        ]
    base_metrics = _load_oof_strategy_metrics(selection_cfg, run_id, base_metric_strategies, layer="base")
    _write_metrics_csv(sel_dir / "base_metrics.csv", base_metrics)
    base_selected_ids = _select_top_fraction(base_metrics, base_top_frac)
    _write_json(sel_dir / "base_selected_strategy_ids.json", base_selected_ids)
    if not base_selected_ids:
        tprint("STRATEGY SELECTION: no base strategies selected; stopping.")
        return

    meta_cfg = dict(selection_cfg)
    with _temporary_env(
        {
            **_selection_profile_env(final=False),
            "EPM_META_STRATEGY_IDS": ",".join(base_selected_ids),
        }
    ):
        tprint(
            "STRATEGY SELECTION: lightweight train_meta "
            f"on {len(base_selected_ids)} base-selected strategies"
        )
        run_train_meta(meta_cfg, ts_override=ts_sig.strftime("%Y%m%d_%H%M%S"), store=store)

    meta_strategies = [s for s in all_strategies if str(s.get("strategy_id")) in set(base_selected_ids)]
    meta_metrics = _load_oof_strategy_metrics(meta_cfg, run_id, meta_strategies, layer="meta")
    if not meta_metrics:
        meta_metrics = [r for r in base_metrics if str(r.get("strategy_id")) in set(base_selected_ids)]
    _write_metrics_csv(sel_dir / "meta_metrics.csv", meta_metrics)

    top15_pool = _select_portfolio_pool(
        meta_metrics,
        pool_n=final_pool_n,
        min_side_share=min_side_share,
    )
    top15_ids = [str(r["strategy_id"]) for r in top15_pool]
    _write_json(sel_dir / "top15_strategy_pool.json", top15_pool)

    from extreme_price_movements.simple_policy_optimiser import (
        run_simple_policy_optimisation,
    )

    with _temporary_env(
        {
            "EPM_POLICY_STRATEGY_IDS": ",".join(top15_ids),
            "EPM_POLICY_MAX_DEPLOYMENT_STRATEGIES": inference_max_n,
            "EPM_POLICY_MAX_STRATEGIES_PER_SIDE": max_per_side,
            "EPM_POLICY_DEPLOYMENT_SELECTION_METRIC": "top_30",
        }
    ):
        tprint(f"STRATEGY SELECTION: simple_policy_optimiser on top-{len(top15_ids)} pool")
        run_simple_policy_optimisation(
            data_root=selection_cfg["data_root"],
            run_id=run_id,
            cost_pct=float(selection_cfg.get("ridge_cost_pct", selection_cfg.get("fee_bps", 50.0) / 10000.0)),
            max_strategies=len(top15_ids),
            n_trials=policy_trials,
            strategy_ids=top15_ids,
            market_mode=_market_mode_from_cfg(selection_cfg),
        )

    policy_winners = [
        sid for sid in _load_policy_winners(selection_cfg, run_id) if sid in set(top15_ids)
    ]
    _write_json(sel_dir / "policy_winners.json", policy_winners)
    if not policy_winners:
        tprint("STRATEGY SELECTION: no policy winners selected; skipping final retrain.")
        _write_json(
            sel_dir / "strategy_selection_report.json",
            {
                "run_id": run_id,
                "base_selected_count": len(base_selected_ids),
                "top15_count": len(top15_ids),
                "policy_winner_count": 0,
                "status": "no_policy_winners",
            },
        )
        _maintenance_checkpoint("strategies_selection:end")
        return

    if os.environ.get("EPM_STRATEGY_SELECTION_SKIP_FINAL_RETRAIN", "").strip() in {"1", "true", "True", "yes"}:
        tprint("STRATEGY SELECTION: skipping final retrain because EPM_STRATEGY_SELECTION_SKIP_FINAL_RETRAIN is set.")
        _write_json(
            sel_dir / "strategy_selection_report.json",
            {
                "run_id": run_id,
                "candidate_count": len(all_strategies),
                "base_selected_count": len(base_selected_ids),
                "top15_count": len(top15_ids),
                "policy_winner_count": len(policy_winners),
                "policy_winners": policy_winners,
                "final_retrain_status": "skipped_by_env",
            },
        )
        _maintenance_checkpoint("strategies_selection:end")
        return

    final_report = _run_final_retraining_layers(
        selection_cfg,
        ts_sig,
        policy_winners,
        store=store,
    )
    _write_json(
        sel_dir / "strategy_selection_report.json",
        {
            "run_id": run_id,
            "candidate_count": len(all_strategies),
            "base_selected_count": len(base_selected_ids),
            "top15_count": len(top15_ids),
            "policy_winner_count": len(policy_winners),
            "policy_winners": policy_winners,
            "final_retrain_report_path": str(sel_dir / "final_retrain_report.json"),
            "final_retrain_status": final_report.get("status", "complete") if isinstance(final_report, dict) else "complete",
        },
    )
    tprint("STRATEGY SELECTION COMPLETE")
    _maintenance_checkpoint("strategies_selection:end")


def run_optimise(cfg, ts_override=None, store=None):
    _maintenance_checkpoint("optimise:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    _load_mask_params_by_mode(cfg)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return False

    # Slice Plan Injection
    try:
        slice_plan = load_or_build_slice_plan(
            cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
        )
        if "utility_policy_optimisation" in slice_plan.get("materialized_views", {}):
            stage_view = slice_plan["materialized_views"]["utility_policy_optimisation"]
            stage_view = apply_stage_usage_limits(
                stage_view,
                max_assets=cfg.get("planned_max_assets"),
                max_months=cfg.get("planned_max_months"),
            )
            cfg["_active_stage_view"] = stage_view
        else:
            tprint(
                f"Warning: stage utility_policy_optimisation not found in materialized_views"
            )
    except Exception as e:
        tprint(f"Slice plan loading failed: {e}")

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    if bool(cfg.get("optimise_use_ridge_oof", False)):
        try:
            run_optimise_from_ridge_oof(
                run_id=run_id,
                data_root=cfg["data_root"],
                fee_roundtrip=float(cfg.get("optimiser_fee_pct", 0.003)),
                cooldown_hours=float(cfg.get("optimise_ridge_oof_cooldown_hours", 0.0)),
            )
        except Exception as exc:
            tprint(f"ERROR: Ridge OOF optimise failed: {exc}")
            return False
        _maintenance_checkpoint("optimise:end")
        return True

    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    backtest_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
    )
    if not os.path.exists(backtest_file):
        tprint(
            "Backtest results not found. Running backtest to generate trade data for optimiser..."
        )
        bt_cfg = dict(cfg)
        bt_cfg["offline_backtest_skip_universe_refresh"] = True
        run_backtest(bt_cfg, ts_override=ts_override, store=store)
        if not os.path.exists(backtest_file):
            tprint(
                f"ERROR: Backtest still not found at {backtest_file}. Aborting optimise."
            )
            return False
    trades = pd.read_csv(backtest_file, low_memory=False)
    trades = _downcast_numeric_frame(trades)
    trades.attrs["threaded_exit_stream"] = True  # Inject attribute stripped by CSV save
    if "optimiser_fee_pct" in cfg:
        try:
            trades.attrs["fee_pct"] = float(cfg["optimiser_fee_pct"])
        except Exception:
            pass
    if "atr_pct_15m" in trades.columns:
        atr_15m = trades["atr_pct_15m"]
    elif "atr" in trades.columns:
        atr_15m = trades["atr"]
    else:
        atr_15m = pd.Series(0.01, index=trades.index)

    params_path = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "strategy_params.json"
    )
    run_optimise_step(
        trades=trades,
        atr_15m=atr_15m,
        output_path=params_path,
        policy=Policy(mode="train_baseline", params_path=params_path),
        state_path=state_file if os.path.exists(state_file) else None,
        store_base_dir=cfg.get("data_root"),
        run_id=run_id,
        data_root=cfg["data_root"],
        ohlcv_store=store,
    )
    tprint(f"OPTIMISE COMPLETE: {params_path}")

    # Run breakdown diagnostics after optimization
    try:
        run_breakdown_diagnostics_integration(cfg, ts_sig)
    except Exception as e:
        tprint(f"WARNING: breakdown diagnostics failed: {e}")

    try:
        from extreme_price_movements.reports.bucket_report import report_optimise

        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        rp = report_optimise(run_id, cfg["data_root"], base_dir=cfg.get("reports_root"))
        tprint(f"Optimise strategy report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: optimise strategy report failed: {_re}")
    _maintenance_checkpoint("optimise:end")
    return True


def clear_caches():
    """Force garbage collection and clear the on-disk caches before a run."""
    import gc
    import os

    # 1. Force Python garbage collection
    collected = gc.collect()
    tprint(f"GC: Collected {collected} objects on startup.")

    # 2. Clear known temporary cache directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dirs = [
        os.path.join(project_root, "cache"),
        os.path.join(project_root, "data_cache"),
    ]

    for cdir in cache_dirs:
        if os.path.exists(cdir):
            try:
                _clear_runtime_cache_dir(cdir)
                tprint(f"CACHE: Cleared directory {cdir}")
            except Exception as e:
                tprint(f"CACHE: Failed to clear {cdir}: {e}")


def run_breakdown_diagnostics_integration(cfg: dict, ts_sig: pd.Timestamp) -> None:
    """Run breakdown diagnostics integrated into pipeline steps."""
    from extreme_price_movements.breakdown_diagnostics import run_breakdown_diagnostics

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["data_root"], "artifacts", run_id)

    # Check if OHLC data exists for diagnostics
    ohlc_path = os.path.join(run_dir, "ohlc.parquet")
    if not os.path.exists(ohlc_path):
        # Try to create OHLC from store if missing
        try:
            store = make_ohlcv_store(cfg)
            # Get a representative symbol for OHLC extraction (store has no list_symbols API).
            symbols = []
            ohlcv_dir = getattr(store, "ohlcv_dir", None)
            if ohlcv_dir and os.path.isdir(ohlcv_dir):
                import glob

                for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
                    base = os.path.basename(path)
                    if not base.startswith("symbol="):
                        continue
                    raw = base.replace("symbol=", "")
                    symbols.append(raw.replace("_", "/", 1))
            symbols = sorted(set(symbols))

            if symbols:
                symbol = symbols[0]  # Use first available symbol
                ohlc_data = store.load(symbol)
                if ohlc_data is not None and len(ohlc_data) > 0:
                    ohlc_data.to_parquet(ohlc_path)
                    tprint(f"Created OHLC data for diagnostics from {symbol}")
                else:
                    tprint("WARNING: No OHLC data available for breakdown diagnostics")
                    return
            else:
                tprint("WARNING: No symbols found in store for breakdown diagnostics")
                return
        except Exception as e:
            tprint(f"WARNING: Could not create OHLC data for diagnostics: {e}")
            return

    # Configure breakdown diagnostics
    diag_cfg = {
        "ohlc_path": ohlc_path,
        "lookback_h": cfg.get("breakdown_lookback_h", 12),
        "baseline_trigger": cfg.get("breakdown_trigger", 0.08),
        "trigger_sweep": cfg.get(
            "breakdown_trigger_sweep", [0.06, 0.07, 0.08, 0.09, 0.10]
        ),
        "decluster_h": cfg.get("breakdown_decluster_h", 6),
        "max_event_h": cfg.get("breakdown_max_event_h", 72),
        "entry_offsets_h": cfg.get(
            "breakdown_entry_offsets", [-12, -6, -4, -2, -1, 0, 1, 2, 4, 6, 12]
        ),
        "directions": cfg.get("breakdown_directions", ["follow", "fade"]),
        "cost_stress_multipliers": cfg.get(
            "breakdown_cost_stress", [1.0, 1.25, 1.5, 2.0]
        ),
        "optimise_run_dir": run_dir,
    }

    try:
        tprint("Running breakdown diagnostics...")
        result = run_breakdown_diagnostics(diag_cfg, run_dir)

        # Log key verdicts
        verdict = result.get("verdict", {})
        tprint("BREAKDOWN DIAGNOSTICS VERDICT:")
        for key, value in verdict.items():
            if key == "recommendations":
                continue
            tprint(f"  {key}: {value}")

        recommendations = verdict.get("recommendations", {})
        if recommendations:
            tprint("RECOMMENDATIONS:")
            for key, rec in recommendations.items():
                if verdict.get(
                    key, False
                ):  # Only show recommendations for failed checks
                    tprint(f"  {key}: {rec}")

        tprint(f"Breakdown diagnostics saved to: {run_dir}/breakdown_diagnostics/")

    except Exception as e:
        tprint(f"ERROR: breakdown diagnostics failed: {e}")
        raise


def run_breakdown_diagnostics_standalone(cfg: dict, ts_override: str = None) -> None:
    """Standalone breakdown diagnostics mode."""
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    tprint(f"Breakdown Diagnostics mode. ts_sig={ts_sig}")
    run_breakdown_diagnostics_integration(cfg, ts_sig)
    tprint("BREAKDOWN DIAGNOSTICS COMPLETE")


def main():
    clear_caches()
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument(
        "mode",
        choices=[
            "download",
            "labels",
            "features",
            "train",
            "base_training",
            "train_base",
            "strategies_selection",
            "meta_training",
            "train_meta",
            "base_hpo",
            "recency_hpo",
            "final_model_fit",
            "sizer",
            "policy_optimiser",
            "optimise",
            "backtest",
            "inference_backtest",
            "run",
            "breakdown_diagnostics",
            "oos_eval",
        ],
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--market-mode",
        choices=["spot", "perps"],
        default="spot",
        help="Market mode for data/features/artifacts (default: spot).",
    )
    parser.add_argument(
        "--exchange",
        default=None,
        help="Exchange id for scoped data/model artifacts (default: EPM_EXCHANGE or binance).",
    )
    parser.add_argument(
        "--model-backend",
        choices=["ebm_on_lgbm", "lgbm_pipeline"],
        default=None,
        help=(
            "Training specialist backend for train_base/train_meta "
            "(default: EPM_MODEL_BACKEND or ebm_on_lgbm)."
        ),
    )
    parser.add_argument(
        "-perps",
        "--perps",
        action="store_true",
        help="Alias for --market-mode perps (isolated *_perps roots)",
    )
    parser.add_argument(
        "--force-feature-recompute",
        action="store_true",
        help="Force full recompute in features mode",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Optional horizon override. If omitted, use per-strategy runtime horizons.",
    )
    parser.add_argument(
        "--ts", dest="ts_override", help="Timestamp override (YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--run-id",
        dest="run_id_override",
        help="Alias for --ts when run id equals artifact timestamp.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only train base models (alpha, spike, exh)",
    )
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Only train meta models (runs train_meta)",
    )
    parser.add_argument(
        "--robust-mode",
        action="store_true",
        help="Use robust planner mode (enables full inference retrain)",
    )
    parser.add_argument(
        "--enable-trigger-discovery-stage",
        action="store_true",
        help="Enable the trigger discovery stage in the pipeline",
    )
    parser.add_argument(
        "--optimise-use-ridge-oof",
        action="store_true",
        help="Run optimise in cheap Ridge/limit-offset OOF mode instead of using backtest_results.csv",
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=400,
        help="Number of assets to sample for oos_eval basket (default: 400)",
    )
    parser.add_argument(
        "--feature-assets",
        type=int,
        default=None,
        help="Limit features mode to the first N assets in the training universe",
    )
    parser.add_argument(
        "--feature-symbols",
        type=str,
        default=None,
        help="Comma-separated explicit symbols for features mode, e.g. BTC/USDT,ETH/USDT",
    )
    parser.add_argument(
        "--skip-feature-postsave-checks",
        action="store_true",
        help="Skip post-save feature completeness and health checks",
    )

    parser.add_argument(
        "--planned-max-assets",
        type=int,
        default=None,
        help="Optional limit on number of symbols to run inside the plan stage",
    )
    parser.add_argument(
        "--planned-max-months",
        type=int,
        default=None,
        help="Optional limit on the number of months to run inside the plan stage",
    )
    parser.add_argument(
        "--refresh-slice-plan", action="store_true", help="Force rebuild the slice plan"
    )
    args = parser.parse_args()
    if args.run_id_override and not args.ts_override:
        _run_id_for_ts = str(args.run_id_override).strip()
        if re.match(r"^\d{8}_\d{6}(?:_|$)", _run_id_for_ts):
            args.ts_override = args.run_id_override

    cfg = CFG.copy()
    _apply_fee_model(cfg, BASE_ROUND_TRIP_FEE_PCT)
    _normalize_cfg_paths(cfg)
    _epm_data_root = str(os.environ.get("EPM_DATA_ROOT", "")).strip()
    if _epm_data_root:
        cfg["data_root"] = os.path.abspath(_epm_data_root)
        cfg["reports_root"] = os.path.join(os.path.abspath(_epm_data_root), "reports")
        tprint(f"EPM_DATA_ROOT override: data_root={cfg['data_root']}")
    _source_run_id = str(
        os.environ.get("EPM_ARTIFACT_SOURCE_RUN_ID")
        or os.environ.get("EPM_SOURCE_RUN_ID")
        or ""
    ).strip()
    if _source_run_id:
        cfg["artifact_source_run_id"] = _source_run_id
        cfg["label_source_run_id"] = _source_run_id
        cfg["feature_source_run_id"] = _source_run_id
        tprint(
            "Artifact source override: "
            f"labels/features/slices will read from run_id={_source_run_id}"
        )
    _label_source_run_id = str(os.environ.get("EPM_LABEL_SOURCE_RUN_ID") or "").strip()
    if _label_source_run_id:
        cfg["label_source_run_id"] = _label_source_run_id
        tprint(f"Label source override: labels will read from run_id={_label_source_run_id}")
    _feature_source_run_id = str(os.environ.get("EPM_FEATURE_SOURCE_RUN_ID") or "").strip()
    if _feature_source_run_id:
        cfg["feature_source_run_id"] = _feature_source_run_id
        tprint(
            "Feature source override: features/slices will read from "
            f"run_id={_feature_source_run_id}"
        )
    if args.run_id_override:
        cfg["output_run_id"] = str(args.run_id_override).strip()
    _label_artifact_run_id = str(
        os.environ.get("EPM_LABEL_ARTIFACT_RUN_ID")
        or cfg.get("output_run_id")
        or ""
    ).strip()
    if _label_artifact_run_id:
        cfg["_label_artifact_run_id"] = _label_artifact_run_id
    _mr_tf_env_map = {
        "EPM_MR_TF_MASKS_ENABLED": ("enabled", "bool"),
        "EPM_MR_TF_OPTUNA_ENABLED": ("optuna_enabled", "bool"),
        "EPM_MR_TF_OPTUNA_TRIALS": ("optuna_trials", "int"),
        "EPM_MR_TF_OPTUNA_PATIENCE": ("optuna_patience", "int"),
        "EPM_MR_TF_MIN_TRAIN_SAMPLES": ("min_train_samples", "int"),
        "EPM_MR_TF_PROMOTION_MARGIN": ("promotion_margin", "float"),
        "EPM_MR_TF_PROMOTION_TOP_FRAC": ("promotion_top_frac", "float"),
        "EPM_MR_TF_OPTUNA_USE_NUMBA": ("optuna_use_numba", "bool"),
        "EPM_MR_TF_OPTUNA_NUMBA_MIN_ROWS": ("optuna_numba_min_rows", "int"),
        "EPM_MR_TF_SUPPORT_LOSS_HURDLE": ("support_loss_hurdle", "float"),
        "EPM_MR_TF_SUPPORT_LOSS_HURDLE_RATIO": ("support_loss_hurdle_ratio", "float"),
        "EPM_MR_TF_SUPPORT_LOSS_HURDLE_FLOOR": ("support_loss_hurdle_floor", "float"),
        "EPM_MR_TF_SUPPORT_LOSS_QUADRATIC_MULTIPLIER": (
            "support_loss_quadratic_multiplier",
            "float",
        ),
        "EPM_MR_TF_SUPPORT_LOSS_HARD_VETO": ("support_loss_hard_veto", "bool"),
        "EPM_MR_TF_SUPPORT_VALUE_POWER": ("support_value_power", "float"),
        "EPM_MR_TF_MIN_COVERAGE": ("min_coverage", "float"),
        "EPM_MR_TF_MIN_EARNED_QUALITY_UPLIFT": (
            "min_earned_quality_uplift",
            "float",
        ),
    }
    _mr_tf_section = dict(cfg.get("mr_tf_masks") or {})
    _mr_tf_changed = False
    for _env_name, (_cfg_key, _kind) in _mr_tf_env_map.items():
        _raw = str(os.environ.get(_env_name, "")).strip()
        if _raw == "":
            continue
        try:
            if _kind == "bool":
                _value = _raw.lower() in {"1", "true", "yes", "on"}
            elif _kind == "int":
                _value = int(float(_raw))
            else:
                _value = float(_raw)
        except Exception as exc:
            tprint(f"WARNING: ignoring invalid {_env_name}={_raw!r}: {exc}")
            continue
        _mr_tf_section[_cfg_key] = _value
        _mr_tf_changed = True
    if _mr_tf_changed:
        cfg["mr_tf_masks"] = _mr_tf_section
        tprint(f"MR/TF mask override: {json.dumps(_mr_tf_section, sort_keys=True)}")
    market_mode = "perps" if args.perps else args.market_mode
    exchange_id = str(args.exchange or os.environ.get("EPM_EXCHANGE") or "binance").strip().lower()
    if exchange_id in {"krakenfutures", "kraken_futures"}:
        exchange_id = "kraken"
    cfg["exchange_id"] = exchange_id
    cfg["exchange"] = exchange_id
    os.environ["EPM_EXCHANGE"] = exchange_id
    _apply_market_mode_paths(cfg, market_mode)
    cfg["exchange_data_component"] = exchange_data_component(exchange_id, market_mode)
    tprint(
        f"Market mode: {cfg['market_mode']} "
        f"exchange={exchange_id} scope={cfg['exchange_data_component']} "
        f"(data_root={cfg['data_root']}, reports_root={cfg['reports_root']})"
    )
    if market_mode == "perps":
        cfg = enable_perp_feature_keys(cfg)
        # Perp-mode fee model: 0.10% round-trip (5 bps/side).
        _apply_fee_model(cfg, PERP_ROUND_TRIP_FEE_PCT)
        cfg["feature_save_workers"] = max(
            1,
            int(
                os.environ.get(
                    "EPM_FEATURE_SAVE_WORKERS",
                    cfg.get("feature_save_workers", 4),
                )
            ),
        )
    elif os.environ.get("EPM_FEATURE_SAVE_WORKERS"):
        cfg["feature_save_workers"] = max(
            1, int(os.environ["EPM_FEATURE_SAVE_WORKERS"])
        )
    if os.environ.get("EPM_FEATURE_BACKFILL_SYMBOL_CHUNK_SIZE"):
        cfg["feature_backfill_symbol_chunk_size"] = max(
            1, int(os.environ["EPM_FEATURE_BACKFILL_SYMBOL_CHUNK_SIZE"])
        )
    if os.environ.get("EPM_FEATURE_BACKFILL_KEY_BATCH_SIZE"):
        # 0 means "do not split requested keys into smaller batches"; this is
        # the default and avoids recomputing shared feature dependencies once
        # per key during targeted historical backfills.
        cfg["feature_backfill_key_batch_size"] = max(
            0, int(os.environ["EPM_FEATURE_BACKFILL_KEY_BATCH_SIZE"])
        )
    if os.environ.get("EPM_FEATURE_BACKFILL_COMPUTE_WORKERS"):
        cfg["feature_backfill_compute_workers"] = min(
            2, max(1, int(os.environ["EPM_FEATURE_BACKFILL_COMPUTE_WORKERS"]))
        )
    if os.environ.get("EPM_FEATURE_PORTABILITY_MODE"):
        cfg["feature_portability_mode"] = str(
            os.environ["EPM_FEATURE_PORTABILITY_MODE"]
        ).strip()
    if os.environ.get("EPM_FEATURE_PORTABILITY_STRICT"):
        cfg["feature_portability_strict"] = (
            os.environ["EPM_FEATURE_PORTABILITY_STRICT"].strip().lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_FEATURE_PORTABILITY_ALLOW_VOLUME_SOURCE_DEPENDENT"):
        cfg["feature_portability_allow_volume_source_dependent"] = (
            os.environ["EPM_FEATURE_PORTABILITY_ALLOW_VOLUME_SOURCE_DEPENDENT"]
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_FEATURE_TAIL_COMPUTE_WARMUP_HOURS"):
        cfg["feature_tail_compute_warmup_hours"] = max(
            1, int(os.environ["EPM_FEATURE_TAIL_COMPUTE_WARMUP_HOURS"])
        )
    if os.environ.get("EPM_FEATURE_CAUSAL_STATE_PATH"):
        cfg["feature_causal_transform_state_path"] = os.environ[
            "EPM_FEATURE_CAUSAL_STATE_PATH"
        ]
    if os.environ.get("EPM_FEATURE_CAUSAL_STATE"):
        cfg["feature_causal_transform_state_enabled"] = (
            os.environ["EPM_FEATURE_CAUSAL_STATE"].strip().lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_FEATURE_CAUSAL_STATE_IGNORE_STALE_MIN_REQUIRED"):
        cfg["feature_causal_transform_state_ignore_stale_min_required"] = (
            os.environ["EPM_FEATURE_CAUSAL_STATE_IGNORE_STALE_MIN_REQUIRED"]
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_FEATURE_RAW_ROLLING_STATE_PATH"):
        cfg["feature_raw_rolling_state_path"] = os.environ[
            "EPM_FEATURE_RAW_ROLLING_STATE_PATH"
        ]
    if os.environ.get("EPM_FEATURE_RAW_ROLLING_STATE"):
        cfg["feature_raw_rolling_state_enabled"] = (
            os.environ["EPM_FEATURE_RAW_ROLLING_STATE"].strip().lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_FEATURE_RAW_ROLLING_STATE_MIN_WINDOW"):
        cfg["feature_raw_rolling_state_min_window"] = max(
            1, int(os.environ["EPM_FEATURE_RAW_ROLLING_STATE_MIN_WINDOW"])
        )
    if os.environ.get("EPM_FEATURE_RAW_ROLLING_STATE_SPARSE_PREFIX"):
        cfg["feature_raw_rolling_state_sparse_prefix_enabled"] = (
            os.environ["EPM_FEATURE_RAW_ROLLING_STATE_SPARSE_PREFIX"]
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_MIN_TRAIN_SAMPLES"):
        cfg["min_train_samples"] = max(1, int(os.environ["EPM_MIN_TRAIN_SAMPLES"]))
    if os.environ.get("EPM_BASE_MIN_SAMPLES_HARD_FLOOR"):
        cfg["base_min_samples_hard_floor"] = max(
            1, int(os.environ["EPM_BASE_MIN_SAMPLES_HARD_FLOOR"])
        )
    if os.environ.get("EPM_SAMPLE_WEIGHT_OPT_MIN_SAMPLES"):
        cfg["sample_weight_opt_min_samples"] = max(
            1, int(os.environ["EPM_SAMPLE_WEIGHT_OPT_MIN_SAMPLES"])
        )
    if os.environ.get("EPM_BASE_REQUIRE_POSITIVE_OOF_EXPECTANCY"):
        cfg["base_require_positive_oof_expectancy"] = (
            os.environ["EPM_BASE_REQUIRE_POSITIVE_OOF_EXPECTANCY"].strip().lower()
            not in {"0", "false", "no", "off"}
        )
    if os.environ.get("EPM_BASE_OOF_EXPECTANCY_TOP_FRAC"):
        cfg["base_oof_expectancy_top_frac"] = float(
            os.environ["EPM_BASE_OOF_EXPECTANCY_TOP_FRAC"]
        )
    if os.environ.get("EPM_META_REQUIRE_DISTILLED_BASE_OOF"):
        cfg["meta_require_distilled_base_oof"] = (
            os.environ["EPM_META_REQUIRE_DISTILLED_BASE_OOF"].strip().lower()
            not in {"0", "false", "no", "off"}
        )
        tprint(
            "Meta base-OOF provenance override: "
            f"meta_require_distilled_base_oof={bool(cfg['meta_require_distilled_base_oof'])}"
        )
    if os.environ.get("EPM_META_MIN_BASE_OOF_DISTILLATION_PASSES"):
        cfg["meta_min_base_oof_distillation_passes"] = max(
            0, int(float(os.environ["EPM_META_MIN_BASE_OOF_DISTILLATION_PASSES"]))
        )
        tprint(
            "Meta base-OOF provenance override: "
            f"meta_min_base_oof_distillation_passes={int(cfg['meta_min_base_oof_distillation_passes'])}"
        )
    if os.environ.get("EPM_LGBM_MIN_OOF_DISTILLATION_PASSES"):
        cfg["lgbm_min_oof_distillation_passes"] = max(
            0, int(float(os.environ["EPM_LGBM_MIN_OOF_DISTILLATION_PASSES"]))
        )

    _configure_report_roots(cfg)
    if args.model_backend:
        cfg["model_backend"] = args.model_backend
        cfg["base_model_backend"] = args.model_backend
        cfg["meta_model_backend"] = (
            "lgbm_pipeline"
            if args.model_backend == "lgbm_pipeline"
            else "ebm_on_lgbm_only"
        )
        tprint(f"CLI model backend override: {args.model_backend}")
    cfg["optimise_use_ridge_oof"] = bool(args.optimise_use_ridge_oof)
    cfg["slice_planner_preset"] = "robust" if bool(args.robust_mode) else "fast"
    cfg["train_full_inference_models"] = bool(args.robust_mode)
    cfg["skip_feature_postsave_checks"] = bool(args.skip_feature_postsave_checks)

    tprint(
        f"Planner preset: {cfg['slice_planner_preset']} (full_inference_retrain={cfg['train_full_inference_models']})"
    )
    cfg["enable_trigger_discovery_stage"] = bool(args.enable_trigger_discovery_stage)

    cfg["planned_max_assets"] = args.planned_max_assets
    cfg["planned_max_months"] = args.planned_max_months
    cfg["refresh_slice_plan"] = bool(args.refresh_slice_plan)
    if args.mode in {"train", "base_training", "train_base", "meta_training", "train_meta", "recency_hpo", "final_model_fit"}:
        _apply_training_no_penalty(cfg)
    if str(os.environ.get("EPM_DISABLE_REGIME_ADAPTORS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        cfg["regime_adaptor.enabled"] = False
        tprint("Regime adaptors disabled by EPM_DISABLE_REGIME_ADAPTORS.")
    if args.mode == "download":
        run_download(cfg)
    elif args.mode == "labels":
        run_labels(cfg, horizons=args.horizons, ts_override=args.ts_override)
    elif args.mode == "features":
        _feature_symbols_file_arg = str(
            os.environ.get("EPM_FEATURE_SYMBOLS_FILE", "")
        ).strip()
        _feature_symbols_from_file = ""
        if _feature_symbols_file_arg:
            try:
                _feature_symbols_from_file = Path(_feature_symbols_file_arg).read_text(
                    encoding="utf-8"
                )
                tprint(
                    "Feature symbol allowlist file loaded: "
                    f"{_feature_symbols_file_arg}"
                )
            except Exception as exc:
                tprint(
                    "WARNING: could not read EPM_FEATURE_SYMBOLS_FILE="
                    f"{_feature_symbols_file_arg!r}: {exc}"
                )
        _feature_symbols_arg = args.feature_symbols or os.environ.get(
            "EPM_FEATURE_SYMBOLS", ""
        ) or _feature_symbols_from_file
        _feature_symbols = [
            s.strip()
            for part in str(_feature_symbols_arg).splitlines()
            for s in part.split(",")
            if s.strip()
        ]
        run_features(
            cfg,
            ts_override=args.ts_override,
            force_recompute=bool(args.force_feature_recompute),
            max_assets=args.feature_assets,
            feature_symbols=_feature_symbols,
        )
    elif args.mode == "train":
        run_train(
            cfg,
            ts_override=args.ts_override,
            base_only=args.base_only,
            meta_only=args.meta_only,
        )
    elif args.mode == "base_training":
        run_train(cfg, ts_override=args.ts_override, base_only=True, meta_only=False)
    elif args.mode == "train_base":
        run_train(cfg, ts_override=args.ts_override, base_only=True, meta_only=False)
    elif args.mode == "strategies_selection":
        run_strategies_selection(cfg, ts_override=args.ts_override)
    elif args.mode == "meta_training":
        run_train_meta(cfg, ts_override=args.ts_override)
    elif args.mode == "train_meta":
        run_train_meta(cfg, ts_override=args.ts_override)
    elif args.mode == "base_hpo":
        ts_sig = _resolve_ts_sig(cfg, args.ts_override)
        if ts_sig:
            run_base_hpo_step(ts_sig, cfg)
        else:
            tprint("ERROR: No feature directories found.")
    elif args.mode == "recency_hpo":
        scope_filter = str(os.environ.get("EPM_RECENCY_HPO_SCOPE", "base")).strip().lower()
        scope_tokens = {p.strip() for p in scope_filter.split(",") if p.strip()}
        key_filter = str(
            os.environ.get("EPM_RECENCY_HPO_SCOPE_KEY", "")
            or os.environ.get("EPM_RECENCY_HPO_STRATEGY_ID", "")
        ).strip()
        strategy_filter = str(os.environ.get("EPM_RECENCY_HPO_STRATEGY_ID", "")).strip()
        if not key_filter:
            tprint(
                "ERROR: recency_hpo requires EPM_RECENCY_HPO_SCOPE_KEY or "
                "EPM_RECENCY_HPO_STRATEGY_ID so it runs on one intended "
                "strategy/head instead of every native LGBM scope."
            )
            sys.exit(2)
        cfg["recency_hpo_enabled"] = True
        cfg["recency_hpo_only"] = True
        cfg["lgbm_use_native_preset"] = True
        cfg["model_backend"] = "lgbm_pipeline"
        cfg["base_model_backend"] = "lgbm_pipeline"
        cfg["meta_model_backend"] = "lgbm_pipeline"
        os.environ.setdefault("EPM_MODEL_BACKEND", "lgbm_pipeline")
        os.environ.setdefault("EPM_LGBM_USE_NATIVE_PRESET", "1")
        os.environ.setdefault("EPM_RECENCY_HPO_ENABLED", "1")
        os.environ.setdefault("EPM_RECENCY_HPO_ONLY", "1")
        os.environ.setdefault("EPM_TRAIN_EXTEND_TO_LATEST", "1")
        os.environ.setdefault("EPM_TRAIN_EXTEND_DISABLE_EXACT_PLAN_FILTER", "1")
        if strategy_filter:
            if scope_tokens == {"meta"}:
                os.environ.setdefault("EPM_META_STRATEGY_IDS", strategy_filter)
            elif "both" in scope_tokens or {"base", "meta"}.issubset(scope_tokens):
                os.environ.setdefault("EPM_BASE_STRATEGY_IDS", strategy_filter)
                os.environ.setdefault("EPM_META_STRATEGY_IDS", strategy_filter)
            else:
                os.environ.setdefault("EPM_BASE_STRATEGY_IDS", strategy_filter)
        tprint(
            "STEP: RECENCY HPO START "
            f"(scope={scope_filter}, key_filter={key_filter}, "
            "native_preset_required=yes)"
        )
        from extreme_price_movements.lgbm_recency_hpo import RecencyHPOComplete

        def _run_recency_hpo_stage(stage_name, fn):
            try:
                fn()
            except RecencyHPOComplete as exc:
                winner = exc.payload.get("winner", {}) if isinstance(exc.payload, dict) else {}
                tprint(
                    "RECENCY HPO COMPLETE "
                    f"(stage={stage_name}, scope={exc.scope}, "
                    f"scope_key={exc.scope_key}, winner={winner})"
                )
                return True
            return False

        if scope_tokens == {"meta"}:
            completed = _run_recency_hpo_stage(
                "meta",
                lambda: run_train_meta(cfg, ts_override=args.ts_override),
            )
            if not completed:
                tprint("WARNING: recency_hpo meta completed without matching a target scope.")
        elif "both" in scope_tokens or {"base", "meta"}.issubset(scope_tokens):
            base_completed = _run_recency_hpo_stage(
                "base",
                lambda: run_train(
                    cfg,
                    ts_override=args.ts_override,
                    base_only=True,
                    meta_only=False,
                ),
            )
            meta_completed = _run_recency_hpo_stage(
                "meta",
                lambda: run_train_meta(cfg, ts_override=args.ts_override),
            )
            if not base_completed:
                tprint("WARNING: recency_hpo base completed without matching a target scope.")
            if not meta_completed:
                tprint("WARNING: recency_hpo meta completed without matching a target scope.")
        else:
            completed = _run_recency_hpo_stage(
                "base",
                lambda: run_train(
                    cfg,
                    ts_override=args.ts_override,
                    base_only=True,
                    meta_only=False,
                ),
            )
            if not completed:
                tprint("WARNING: recency_hpo base completed without matching a target scope.")
    elif args.mode == "final_model_fit":
        _run_final_model_fit(cfg, ts_override=args.ts_override)
    elif args.mode == "sizer":
        run_sizer(cfg, ts_override=args.ts_override)
    elif args.mode == "policy_optimiser":
        ts_sig = _resolve_ts_sig(cfg, args.ts_override)
        if ts_sig:
            run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
            try:
                slice_plan = load_or_build_slice_plan(
                    cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
                )
                materialized = slice_plan.get("materialized_views", {})
                stage_view = _choose_policy_stage_view(materialized)
                if stage_view is None and "holdout_strategy_eval" in materialized:
                    stage_view = materialized["holdout_strategy_eval"]
                if stage_view is not None:
                    cfg["_active_stage_view"] = apply_stage_usage_limits(
                        stage_view,
                        max_assets=cfg.get("planned_max_assets"),
                        max_months=cfg.get("planned_max_months"),
                    )
            except Exception as _slice_exc:
                tprint(f"Policy optimiser slice plan loading failed: {_slice_exc}")
            tprint("STEP: POLICY OPTIMISER START (policy_optimiser.py)")
            run_policy_optimisation(
                data_root=cfg["data_root"],
                run_id=run_id,
                holdout_frac=float(cfg.get("policy_optimiser_holdout_frac", 0.30)),
                cost_pct=float(
                    cfg.get("ridge_cost_pct", cfg.get("fee_bps", 50.0) / 10000.0)
                ),
                use_offset_optimiser=bool(cfg.get("run_limit_offset_optimiser", False)),
                stage_view=cfg.get("_active_stage_view"),
            )
        else:
            tprint("ERROR: No feature directories found.")
    elif args.mode == "optimise":
        run_optimise(cfg, ts_override=args.ts_override)
    elif args.mode == "backtest":
        run_backtest(cfg, ts_override=args.ts_override)
    elif args.mode == "inference_backtest":
        run_inference_backtest(cfg, ts_override=args.ts_override)
    elif args.mode == "breakdown_diagnostics":
        run_breakdown_diagnostics_standalone(cfg, ts_override=args.ts_override)
    elif args.mode == "oos_eval":
        from extreme_price_movements.oos_pipeline import run_oos_eval_pipeline

        ts_sig = _resolve_ts_sig(cfg, args.ts_override)
        if ts_sig:
            try:
                slice_plan = load_or_build_slice_plan(
                    cfg, ts_sig, force_refresh=cfg.get("refresh_slice_plan", False)
                )
                if "holdout_strategy_eval" in slice_plan.get("materialized_views", {}):
                    stage_view = (
                        slice_plan["materialized_views"]["holdout_strategy_eval"]
                        .get("sub_views", {})
                        .get("backtest_eval")
                    )
                    if stage_view:
                        cfg["_active_stage_view"] = apply_stage_usage_limits(
                            stage_view,
                            max_assets=cfg.get("planned_max_assets"),
                            max_months=cfg.get("planned_max_months"),
                        )
            except Exception as e:
                tprint(f"Slice plan loading failed: {e}")
        run_oos_eval_pipeline(cfg, n_assets=args.n_assets, ts_override=args.ts_override)
    elif args.mode == "run":
        run_all(cfg, ts_override=args.ts_override)


if __name__ == "__main__":
    main()
