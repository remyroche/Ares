from __future__ import annotations

import ast
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Import canonical constants from central config
from extreme_price_movements.config import (
    CANON_BUCKETS,
    CANON_HORIZONS,
    CANON_SIDES,
    CANON_SIDE_HORIZON_CELLS,
)
from extreme_price_movements.path_utils import resolve_mode_file
from extreme_price_movements.strategy_registry import normalize_strategy_horizon


OFFLINE_OPTIMISERS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = OFFLINE_OPTIMISERS_DIR / "reports"

CANDIDATE_BEST_PARAMS_CSV = REPORTS_DIR / "candidate_thresholds_best_params.csv"
INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV = (
    REPORTS_DIR / "inference_candidate_mask_best_params.csv"
)
INFERENCE_CANDIDATE_MASK_BEST_PARAMS_PER_BUCKET_CSV = (
    REPORTS_DIR / "inference_candidate_mask_best_params_per_bucket.csv"
)
TBM_BEST_PARAMS_CSV = REPORTS_DIR / "tbm_best_params.csv"
TBM_BEST_PARAMS_PER_BUCKET_CSV = REPORTS_DIR / "tbm_best_params_per_bucket.csv"
TBM_BEST_PARAMS_PER_CELL_CSV = REPORTS_DIR / "tbm_best_params_per_cell.csv"
TBM_BEST_PARAMS_PER_SIDE_HORIZON_CSV = (
    REPORTS_DIR / "tbm_best_params_per_side_horizon.csv"
)
TBM_GEOMETRY_GRID_CSV = REPORTS_DIR / "tbm_geometry_grid.csv"
SAMPLE_WEIGHT_BEST_PARAMS_CSV = REPORTS_DIR / "sample_weight_best_params.csv"


def normalize_market_mode(market_mode: str | None = None) -> str:
    """Return the canonical market data mode used for persisted optimizer files."""
    import os

    mode = str(market_mode or os.environ.get("EPM_MARKET_MODE", "spot")).strip().lower()
    if mode in {"perp", "perps", "futures", "future"}:
        return "perps"
    return "spot"


def market_suffix(market_mode: str | None = None) -> str:
    return f"_{normalize_market_mode(market_mode)}"


def market_report_path(path: Path, market_mode: str | None = None) -> Path:
    """Return the spot/perps-specific variant of a report path.

    Existing market suffixes are replaced instead of appended, so callers can
    safely pass either unsuffixed or already suffixed paths.
    """
    mode = normalize_market_mode(market_mode)
    stem = path.stem
    for suffix in ("_spot", "_perps", "_perp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return path.with_name(f"{stem}_{mode}{path.suffix}")


def _safe_strategy_id_from_rule(rule_key: str) -> str:
    import re

    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(rule_key))
    safe_id = re.sub(r"_+", "_", safe_id)
    return safe_id.strip("_")


def _allow_legacy_market_fallback() -> bool:
    import os

    return str(os.environ.get("EPM_ALLOW_LEGACY_MARKET_FALLBACK", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _market_preferred_paths(path: Path, market_mode: str | None = None) -> list[Path]:
    mode = normalize_market_mode(market_mode)
    mode_path = market_report_path(path, market_mode)
    # Most historical spot optimiser reports were written without a market suffix.
    # Keep perps strict, but allow spot to read the unsuffixed spot report when a
    # newer *_spot file has not been emitted yet.
    if mode == "spot" and mode_path != path:
        return [mode_path, path]
    if _allow_legacy_market_fallback() and mode_path != path:
        return [mode_path, path]
    return [mode_path]

# Bucket to strategy_id mapping (new convention)
BUCKET_TO_STRATEGY_ID = {
    "MR_long": "long_mr",
    "TF_long": "long_tf",
    "MR_short": "short_mr",
    "TF_short": "short_tf",
}
STRATEGY_ID_TO_BUCKET = {v: k for k, v in BUCKET_TO_STRATEGY_ID.items()}

# Canonical constants - import from central config
TBM_BUCKET_NAMES = CANON_BUCKETS  # Now uses strategy_ids: ["long_tf", "long_mr", "short_tf", "short_mr"]
TBM_HORIZONS = CANON_HORIZONS  # [5, 10]
TBM_SIDES = CANON_SIDES
TBM_SIDE_HORIZON_CELLS = CANON_SIDE_HORIZON_CELLS


def _side_horizon_alias_from_cell(cell_key: str) -> str | None:
    parts = str(cell_key or "").split("_")
    if len(parts) < 3:
        return None
    side = parts[1]
    horizon = parts[2]
    if side not in {"long", "short"} or not horizon.startswith("H"):
        return None
    return f"{side}_{horizon}"


def _to_scalar(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        import numpy as _np

        if isinstance(v, (_np.generic,)):
            return v.item()
    except Exception:
        pass
    return str(v)


def _coerce_numeric_if_possible(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return v
        low = s.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        try:
            if "." in s or "e" in low:
                return float(s)
            return int(s)
        except Exception:
            return v
    return v


def _parse_numeric_series(v: Any) -> np.ndarray:
    if v is None:
        return np.array([], dtype=np.float32)
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False)
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)
    if not isinstance(v, str):
        return np.array([], dtype=np.float32)

    s = v.strip()
    if not s:
        return np.array([], dtype=np.float32)

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return np.asarray(parsed, dtype=np.float32)
    except Exception:
        pass

    cleaned = s.strip("[]()")
    arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    if arr.size == 0 and "," in cleaned:
        arr = np.fromstring(cleaned.replace(",", " "), sep=" ", dtype=np.float32)
    return arr


def _read_best_params_csv(path: Path) -> Dict[str, Any]:
    import pandas as pd

    path = resolve_mode_file(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k == "saved_at":
            continue
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    out[k] = json.loads(s)
                    continue
                except Exception:
                    pass
        out[k] = _coerce_numeric_if_possible(_to_scalar(v))
    return out


def load_inference_candidate_mask_params_by_mode(
    market_mode: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    mode_to_path = {
        "price_up_tf": resolve_mode_file(
            REPORTS_DIR / "inference_candidate_mask_best_params_price_up_tf.csv",
        ),
        "price_up_mr": resolve_mode_file(
            REPORTS_DIR / "inference_candidate_mask_best_params_price_up_mr.csv",
        ),
        "price_down_tf": resolve_mode_file(
            REPORTS_DIR / "inference_candidate_mask_best_params_price_down_tf.csv",
        ),
        "price_down_mr": resolve_mode_file(
            REPORTS_DIR / "inference_candidate_mask_best_params_price_down_mr.csv",
        ),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for mode, path in mode_to_path.items():
        row = _read_best_params_csv(market_report_path(path, market_mode))
        if not row:
            continue
        row.setdefault("mode", mode)
        out[mode] = row
    return out


def load_inference_candidate_mask_params_per_bucket(
    top_n: int = 2,
    ranking_metric: str = "score_for_best_params",
    classification_filter: str | None = None,
    market_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Load top-N dynamically generated strategy parameters from the mask-optimiser.

    By default this returns the top-2 rules per (side, horizon) group, ensuring
    diversity across long/short and different horizons (e.g., H3, H10).

    Args:
        top_n: Number of top rules to load per (side, horizon) group (default: 2)
        ranking_metric: Metric to rank rules by (default: "score_for_best_params")
                      Options: "score_for_best_params", "composite_score",
                            "learnability_step_c_score", "stage2_score", "mask_oof_corr"
        classification_filter: Filter by production_classification (e.g., "production",
                            "research", or None for all). Default None allows all.
        market_mode: Market mode to filter/select market-specific strategy registries.
                     Set EPM_MASK_STRATEGY_SOURCE_CSV to force an explicit registry.

    Returns:
        List of strategy dicts, each with keys:
            - strategy_id: Safe identifier for the rule
            - trade_side: "long" or "short"
            - base_event_trigger: The canonical_key
            - mask_params: Dict with canonical_key
    """
    import glob
    import os
    import pandas as pd
    import logging as _logging

    _log = _logging.getLogger("params_store")

    path = None
    forced_source_csv = str(os.environ.get("EPM_MASK_STRATEGY_SOURCE_CSV", "")).strip()
    if forced_source_csv:
        forced_path = Path(forced_source_csv).expanduser()
        if forced_path.exists() and forced_path.stat().st_size > 100:
            path = forced_path
        else:
            _log.warning(
                "[params_store] EPM_MASK_STRATEGY_SOURCE_CSV=%s is missing or empty",
                forced_source_csv,
            )
    preferred_paths = []
    if path is None:
        preferred_paths = [
            p
            for base_path in (
                INFERENCE_CANDIDATE_MASK_BEST_PARAMS_PER_BUCKET_CSV,
                INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
            )
            for p in _market_preferred_paths(base_path, market_mode)
        ]
        if str(os.environ.get("EPM_MASK_STRATEGY_SKIP_REPORT_INPUTS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            preferred_paths = []
    for candidate_path in preferred_paths:
        try:
            if candidate_path.exists() and candidate_path.stat().st_size > 100:
                path = candidate_path
                break
        except OSError:
            continue

    mode = normalize_market_mode(market_mode)
    candidate_paths: list[Path] = []
    if path is None:
        output_roots = [Path(f"production_lgbm_outputs_{mode}")]
        if _allow_legacy_market_fallback():
            output_roots.append(Path("production_lgbm_outputs"))
        for output_root in output_roots:
            candidate_paths.extend(
                Path(p)
                for p in glob.glob(
                    str(output_root / "run_*" / "final_rule_registry.csv")
                )
            )
        candidate_paths.extend(
            Path(p)
            for p in glob.glob(
                str(
                    Path("tmp")
                    / f"lgbm_*{mode}*_run*"
                    / "run_*"
                    / "final_rule_registry.csv"
                )
            )
        )
        if _allow_legacy_market_fallback():
            candidate_paths.extend(
                Path(p)
                for p in glob.glob(
                    str(
                        Path("tmp")
                        / "lgbm_*_run*"
                        / "run_*"
                        / "final_rule_registry.csv"
                    )
                )
            )
        opposite = "spot" if mode == "perps" else "perps"
        candidate_paths = [
            p
            for p in candidate_paths
            if f"_{opposite}" not in str(p)
            and (opposite != "perps" or "_perp" not in str(p))
        ]

    if path is None:
        if candidate_paths:
            candidate_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for _cp in candidate_paths:
                try:
                    if _cp.stat().st_size > 100:
                        path = _cp
                        break
                except OSError:
                    continue
        else:
            path = (
                Path(f"production_lgbm_outputs_{mode}")
                / "combined_accepted_rule_registry.csv"
            )
            if not path.exists():
                path = market_report_path(
                    REPORTS_DIR / "lgbm_accepted_rule_registry.csv", mode
                )
            if _allow_legacy_market_fallback() and not path.exists():
                legacy_path = (
                    Path("production_lgbm_outputs") / "combined_accepted_rule_registry.csv"
                )
                path = (
                    legacy_path
                    if legacy_path.exists()
                    else REPORTS_DIR / "lgbm_accepted_rule_registry.csv"
                )

    if not path or not path.exists():
        return []

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []
    if "market_mode" in df.columns:
        df = df[df["market_mode"].map(normalize_market_mode) == mode].copy()
        if df.empty:
            _log.warning("[params_store] No %s rows found in %s", mode, path)
            return []

    # Ensure required columns exist
    if "side" not in df.columns:
        if "trade_side" in df.columns:
            df["side"] = df["trade_side"].astype(str).str.lower()
        else:
            df["side"] = "long"
    if "canonical_key" not in df.columns and "base_event_trigger" in df.columns:
        df["canonical_key"] = df["base_event_trigger"].astype(str)
    if "source_horizon" not in df.columns:
        df["source_horizon"] = 100

    # Apply classification filter if specified
    if classification_filter is not None and "production_classification" in df.columns:
        df = df[
            df["production_classification"].astype(str).str.lower()
            == classification_filter.lower()
        ]
        _log.info(
            f"[params_store] Filtered to {len(df)} rules with classification='{classification_filter}'"
        )

    if df.empty:
        _log.warning(
            f"[params_store] No rules match classification filter '{classification_filter}'"
        )
        return []

    strategies = []

    df_sorted = df
    if ranking_metric not in df.columns and "ranking_score" in df.columns:
        ranking_metric = "ranking_score"
        _log.info("[params_store] ranking_metric fallback to 'ranking_score'")

    registry_ranking_fallbacks = (
        "stage_e_rank_score",
        "final_selection_frontier_score",
        "final_candidate_rank_score",
        "standalone_quality_score",
        "composite_score",
        "directional_mean_ret",
        "mean_net_ret",
        "mean_uplift",
    )
    if ranking_metric not in df.columns:
        for _fallback in registry_ranking_fallbacks:
            if _fallback in df.columns:
                ranking_metric = _fallback
                _log.info(
                    f"[params_store] ranking_metric fallback to '{ranking_metric}'"
                )
                break
        else:
            _log.warning(
                f"[params_store] ranking_metric '{ranking_metric}' not found in {path}, "
                f"using default ranking"
            )
    if ranking_metric in df.columns:
        metric_values = pd.to_numeric(df[ranking_metric], errors="coerce")
        if not bool(np.isfinite(metric_values.to_numpy()).any()):
            _log.warning(
                f"[params_store] ranking_metric '{ranking_metric}' in {path} "
                "has no finite values; searching finite fallback metrics"
            )
            ranking_metric = "__missing__"
        else:
            df_sorted = df.assign(_ranking_metric_value=metric_values).sort_values(
                by="_ranking_metric_value",
                ascending=False,
            )
            df_sorted = df_sorted.drop(columns=["_ranking_metric_value"])

    if ranking_metric not in df.columns:
        for _fallback in registry_ranking_fallbacks:
            if _fallback not in df.columns:
                continue
            metric_values = pd.to_numeric(df[_fallback], errors="coerce")
            if not bool(np.isfinite(metric_values.to_numpy()).any()):
                continue
            ranking_metric = _fallback
            _log.info(f"[params_store] ranking_metric fallback to '{ranking_metric}'")
            df_sorted = df.assign(_ranking_metric_value=metric_values).sort_values(
                by="_ranking_metric_value",
                ascending=False,
            )
            df_sorted = df_sorted.drop(columns=["_ranking_metric_value"])
            break

    # Group by (side, source_horizon) and take top_n from each group.
    # Within a group, apply a weak diversity penalty using IC-series overlap.
    grouped = df_sorted.groupby(["side", "source_horizon"], sort=False)

    overlap_weight = 0.15
    if "ic_series" in df_sorted.columns:
        ic_series_by_idx = {
            idx: _parse_numeric_series(val)
            for idx, val in df_sorted["ic_series"].items()
        }
    else:
        ic_series_by_idx = {}

    def _ic_overlap(idx_a: int, idx_b: int) -> float:
        ic_a = ic_series_by_idx.get(idx_a, np.array([], dtype=np.float32))
        ic_b = ic_series_by_idx.get(idx_b, np.array([], dtype=np.float32))
        if ic_a.size < 2 or ic_b.size < 2:
            return 0.0
        min_len = min(ic_a.size, ic_b.size)
        if min_len < 2:
            return 0.0
        a = ic_a[:min_len]
        b = ic_b[:min_len]
        valid = np.isfinite(a) & np.isfinite(b)
        if int(valid.sum()) < 2:
            return 0.0
        a = a[valid]
        b = b[valid]
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            return 0.0
        corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return max(corr, 0.0)

    for (side, horizon), group in grouped:
        if ranking_metric in group.columns:
            raw_series = pd.to_numeric(group[ranking_metric], errors="coerce")
        else:
            raw_series = pd.Series(np.nan, index=group.index, dtype="float64")
        finite_raw = raw_series[np.isfinite(raw_series.to_numpy())]
        if finite_raw.empty:
            raw_min = 0.0
            raw_max = 0.0
        else:
            raw_min = float(finite_raw.min())
            raw_max = float(finite_raw.max())
        raw_span = max(raw_max - raw_min, 1e-12)

        remaining = group.copy()
        selected_indices: list[int] = []

        while len(selected_indices) < top_n and not remaining.empty:
            best_idx = None
            best_adjusted_score = -np.inf
            best_base_score = -np.inf

            for idx, row in remaining.iterrows():
                base_score = pd.to_numeric(
                    row.get(ranking_metric, np.nan), errors="coerce"
                )
                if not np.isfinite(base_score):
                    continue
                normalized_base = float((float(base_score) - raw_min) / raw_span)
                normalized_base = float(np.clip(normalized_base, 0.0, 1.0))
                overlap = 0.0
                for selected_idx in selected_indices:
                    overlap = max(overlap, _ic_overlap(idx, selected_idx))
                weak_factor = max(0.0, 1.0 - overlap_weight * overlap)
                adjusted_score = normalized_base * weak_factor
                if adjusted_score > best_adjusted_score or (
                    np.isclose(adjusted_score, best_adjusted_score)
                    and float(base_score) > best_base_score
                ):
                    best_idx = idx
                    best_adjusted_score = adjusted_score
                    best_base_score = float(base_score)

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            row = remaining.loc[best_idx]
            key = str(row.get("base_event_trigger", "") or row.get("canonical_key", ""))
            side_val = str(row.get("side", "long")).lower()
            if side_val == "mixed":
                side_val = "long"
            if not key:
                remaining = remaining.drop(index=best_idx)
                continue

            explicit_strategy_id = str(row.get("strategy_id", "") or "").strip()
            safe_id = explicit_strategy_id or _safe_strategy_id_from_rule(key)

            move_bucket = ""
            explicit_move_bucket = str(row.get("move_bucket", "")).strip().lower()
            if explicit_move_bucket in {"up", "down"}:
                move_bucket = explicit_move_bucket
            else:
                trigger = str(row.get("trigger", "")).strip().lower()
                if "price_up" in trigger:
                    move_bucket = "up"
                elif "price_down" in trigger:
                    move_bucket = "down"

            strategy = {
                "strategy_id": safe_id,
                "trade_side": side_val,
                "base_event_trigger": key,
                "canonical_key": key,
                "mask_params": {"canonical_key": key},
                "source_target": str(row.get("source_target", "")).strip(),
                "source_horizon": normalize_strategy_horizon(
                    row.get("source_horizon", horizon)
                ),
                "ranking_metric": ranking_metric,
                "ranking_score": best_base_score,
                "ranking_score_norm": float(
                    np.clip((best_base_score - raw_min) / raw_span, 0.0, 1.0)
                ),
                "adjusted_ranking_score": best_adjusted_score,
            }
            if move_bucket:
                strategy["move_bucket"] = move_bucket
                strategy["candidate_bucket"] = (
                    "best" if move_bucket == "up" else "worst"
                )

            strategies.append(strategy)
            remaining = remaining.drop(index=best_idx)

    _log.info(
        f"[params_store] Loaded {len(strategies)} strategies from {path} "
        f"({len(grouped)} groups x top_{top_n}, ranked by {ranking_metric})"
    )

    return strategies


def load_tbm_geometry_grid() -> Dict[str, Any]:
    """Load the geometry grid saved by compare_tbm_parameters.py.

    Returns a dict with keys:
        per_cell    : dict[cell_key -> {
                          "k_tp_grid"    : sorted unique k_tp values for this cell,
                          "sl_base_grid" : sorted unique sl_as_tp_pct values for this cell,
                          "validated_pairs": list of (k_tp, sl_as_tp_pct) tuples that were
                                            explicitly validated by the optimizer — callers
                                            should sweep only these pairs, not the full
                                            Cartesian product of k_tp_grid × sl_base_grid,
                          "atr_windows"  : sorted unique base_atr_window values for this cell
                                           (replaces single "atr_window" — callers should
                                            iterate over all windows),
                          "atr_window"   : first atr_window (backward-compat alias),
                          "tp_abs_lo_pct": TP floor (min across cell rows),
                          "sl_abs_lo_pct": SL floor (min across cell rows),
                      }]
                      cell_key format: "MR_long_H4", "TF_short_H2", etc.
        k_tp_grid   : global fallback — sorted unique k_tp across all cells
        sl_base_grid: global fallback — sorted unique sl_as_tp_pct across all cells
        atr_window  : global fallback — base_atr_window from first row

    All keys fall back to None if the file is absent or malformed.
    Callers should use per_cell[cell_key] when available, else fall back to
    k_tp_grid / sl_base_grid / atr_window.
    """
    import pandas as pd

    _empty = {
        "per_cell": {},
        "k_tp_grid": None,
        "sl_base_grid": None,
        "atr_window": None,
    }
    grid_path = resolve_mode_file(TBM_GEOMETRY_GRID_CSV)
    if not grid_path.exists():
        return _empty
    try:
        df = pd.read_csv(grid_path)
        if df.empty:
            return _empty

        # Global fallbacks
        k_tp_grid = (
            sorted(df["k_tp"].dropna().unique().tolist())
            if "k_tp" in df.columns
            else None
        )
        sl_base_grid = (
            sorted(df["sl_as_tp_pct"].dropna().unique().tolist())
            if "sl_as_tp_pct" in df.columns
            else None
        )
        atr_window = (
            int(df["base_atr_window"].iloc[0])
            if "base_atr_window" in df.columns
            else None
        )

        # Per-cell grids (new format has "cell_key" column)
        per_cell: Dict[str, Any] = {}
        if "cell_key" in df.columns:
            for cell_key, grp in df.groupby("cell_key"):
                _tp_lo_vals = (
                    grp["tp_abs_lo_pct"].dropna().unique().tolist()
                    if "tp_abs_lo_pct" in grp.columns
                    else []
                )
                _sl_lo_vals = (
                    grp["sl_abs_lo_pct"].dropna().unique().tolist()
                    if "sl_abs_lo_pct" in grp.columns
                    else []
                )
                # Validated triplets: exact (k_tp, sl_as_tp_pct, atr_window) per optimizer row.
                # The window is part of each validated config — callers iterate these triplets
                # directly, pre-computing one barrier base per unique window and reusing it.
                _triplets: list = []
                _has_win = "base_atr_window" in grp.columns
                _cols = ["k_tp", "sl_as_tp_pct"] + (
                    ["base_atr_window"] if _has_win else []
                )
                for _, row in grp[_cols].dropna().iterrows():
                    win = (
                        int(row["base_atr_window"]) if _has_win else (atr_window or 720)
                    )
                    triplet = (
                        round(float(row["k_tp"]), 6),
                        round(float(row["sl_as_tp_pct"]), 6),
                        win,
                    )
                    if triplet not in _triplets:
                        _triplets.append(triplet)
                # Unique windows needed to pre-compute barrier bases (one per window, reused).
                _win_vals: list = sorted(set(t[2] for t in _triplets))
                _first_win = _win_vals[0] if _win_vals else atr_window
                per_cell[str(cell_key)] = {
                    "k_tp_grid": sorted(grp["k_tp"].dropna().unique().tolist()),
                    "sl_base_grid": sorted(
                        grp["sl_as_tp_pct"].dropna().unique().tolist()
                    ),
                    "validated_triplets": _triplets,
                    "validated_pairs": [(t[0], t[1]) for t in _triplets],
                    "atr_windows": _win_vals,
                    "atr_window": _first_win,
                    "tp_abs_lo_pct": float(min(_tp_lo_vals)) if _tp_lo_vals else None,
                    "sl_abs_lo_pct": float(min(_sl_lo_vals)) if _sl_lo_vals else None,
                }

        # Side-only aliases: dynamic strategies no longer distinguish MR/TF.
        side_groups: Dict[str, list[dict[str, Any]]] = {}
        for cell_key, payload in list(per_cell.items()):
            alias = _side_horizon_alias_from_cell(cell_key)
            if alias is None:
                continue
            side_groups.setdefault(alias, []).append(payload)
        for alias, cells in side_groups.items():
            if alias in per_cell or not cells:
                continue
            triplets: list[tuple[float, float, int]] = []
            for cell in cells:
                for triplet in cell.get("validated_triplets") or []:
                    if triplet not in triplets:
                        triplets.append(triplet)
            tp_abs_vals = [
                float(cell.get("tp_abs_lo_pct"))
                for cell in cells
                if cell.get("tp_abs_lo_pct") is not None
            ]
            sl_abs_vals = [
                float(cell.get("sl_abs_lo_pct"))
                for cell in cells
                if cell.get("sl_abs_lo_pct") is not None
            ]
            per_cell[alias] = {
                "k_tp_grid": sorted({float(t[0]) for t in triplets}),
                "sl_base_grid": sorted({float(t[1]) for t in triplets}),
                "validated_triplets": triplets,
                "validated_pairs": [(t[0], t[1]) for t in triplets],
                "atr_windows": sorted({int(t[2]) for t in triplets}),
                "atr_window": int(triplets[0][2]) if triplets else atr_window,
                "tp_abs_lo_pct": float(min(tp_abs_vals)) if tp_abs_vals else None,
                "sl_abs_lo_pct": float(min(sl_abs_vals)) if sl_abs_vals else None,
            }

        return {
            "per_cell": per_cell,
            "k_tp_grid": k_tp_grid,
            "sl_base_grid": sl_base_grid,
            "atr_window": atr_window,
        }
    except Exception:
        return _empty


def save_best_params_csv(
    path: Path, best_params: Dict[str, Any], metadata: Dict[str, Any] | None = None
) -> Path:
    import pandas as pd

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    payload.update({k: _to_scalar(v) for k, v in (metadata or {}).items()})
    payload.update(
        {
            k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else _to_scalar(v)
            for k, v in best_params.items()
        }
    )
    payload["saved_at"] = pd.Timestamp.utcnow().isoformat()
    pd.DataFrame([payload]).to_csv(path, index=False)
    return path


def load_tbm_best_params_per_bucket() -> Dict[str, Dict[str, Any]]:
    """Load per-strategy best TBM params from tbm_best_params_per_bucket.csv.

    Returns a dict keyed by strategy_id (e.g. 'long_tf', 'long_mr',
    'short_tf', 'short_mr'). Each value is a dict of barrier params
    ready for injection into cfg. Falls back to empty dict if the file
    does not exist.
    """
    import pandas as pd

    path = resolve_mode_file(TBM_BEST_PARAMS_PER_BUCKET_CSV)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            # Support both old 'bucket' column and new 'strategy_id' column
            strategy_id = str(row.get("strategy_id", row.get("bucket", "")))
            if not strategy_id:
                continue
            result[strategy_id] = {
                k: _coerce_numeric_if_possible(_to_scalar(v)) for k, v in row.items()
            }
        return result
    except Exception:
        return {}


def load_tbm_best_params_per_cell() -> Dict[str, Dict[str, Any]]:
    """Load per-cell best TBM params from tbm_best_params_per_cell.csv.

    Returns a dict keyed by cell name (e.g. 'long_tf_H3', 'long_mr_H8').
    Each value is a dict of barrier params ready for injection into cfg.
    Falls back to per-strategy params if the cell-specific file is missing.
    """
    import pandas as pd

    path = resolve_mode_file(TBM_BEST_PARAMS_PER_CELL_CSV)
    if not path.exists():
        return load_tbm_best_params_per_bucket()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return load_tbm_best_params_per_bucket()
        if "rank_in_cell" in df.columns:
            df = df.sort_values(["cell_key", "rank_in_cell"], ascending=[True, True])
        result: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            cell = str(row.get("cell_key", ""))
            if not cell:
                # Fallback to strategy_id if cell_key is missing
                cell = str(row.get("strategy_id", row.get("bucket", "")))
            if not cell:
                continue
            if cell in result:
                continue
            result[cell] = {
                k: _coerce_numeric_if_possible(_to_scalar(v)) for k, v in row.items()
            }
        for cell, payload in list(result.items()):
            alias = _side_horizon_alias_from_cell(cell)
            if alias is None or alias in result:
                continue
            result[alias] = dict(payload)
        return result
    except Exception:
        return load_tbm_best_params_per_bucket()


def load_tbm_all_params_per_cell() -> Dict[str, list[Dict[str, Any]]]:
    """Load the full ranked per-cell TBM params set from tbm_best_params_per_cell.csv."""
    import pandas as pd

    path = resolve_mode_file(TBM_BEST_PARAMS_PER_CELL_CSV)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        if "rank_in_cell" in df.columns:
            df = df.sort_values(["cell_key", "rank_in_cell"], ascending=[True, True])
        result: Dict[str, list[Dict[str, Any]]] = {}
        for _, row in df.iterrows():
            cell = str(row.get("cell_key", "")) or str(
                row.get("strategy_id", row.get("bucket", ""))
            )
            if not cell:
                continue
            result.setdefault(cell, []).append(
                {k: _coerce_numeric_if_possible(_to_scalar(v)) for k, v in row.items()}
            )
        for cell, rows in list(result.items()):
            alias = _side_horizon_alias_from_cell(cell)
            if alias is None:
                continue
            alias_rows = result.setdefault(alias, [])
            for row in rows:
                if row not in alias_rows:
                    alias_rows.append(dict(row))
        return result
    except Exception:
        return {}


# =============================================================================
# Side-Horizon TBM Params (agnostic to MR/TF distinction)
# =============================================================================

TBM_BEST_PARAMS_PER_SIDE_HORIZON_CSV = (
    REPORTS_DIR / "tbm_best_params_per_side_horizon.csv"
)


def load_tbm_best_params_per_side_horizon() -> Dict[str, Dict[str, Any]]:
    """Load per-(side, horizon) TBM params from tbm_best_params_per_side_horizon.csv.

    Returns a dict keyed by cell name (e.g. 'long_H2', 'short_H4').
    Each value is a dict of barrier params ready for injection into cfg.
    Falls back to per-bucket params if the side-horizon file is missing.
    """
    import pandas as pd

    path = resolve_mode_file(TBM_BEST_PARAMS_PER_SIDE_HORIZON_CSV)
    if not path.exists():
        return load_tbm_best_params_per_bucket()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return load_tbm_best_params_per_bucket()
        if "rank_in_cell" in df.columns:
            df = df.sort_values(["cell_key", "rank_in_cell"], ascending=[True, True])
        result: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            cell = str(row.get("cell_key", ""))
            if not cell:
                continue
            result[cell] = {
                k: _coerce_numeric_if_possible(_to_scalar(v)) for k, v in row.items()
            }
        return result
    except Exception:
        return load_tbm_best_params_per_bucket()


def load_tbm_all_params_per_side_horizon() -> Dict[str, list[Dict[str, Any]]]:
    """Load the full ranked per-(side, horizon) TBM params set from tbm_best_params_per_side_horizon.csv.

    Returns dict keyed by cell name (e.g. 'long_H2', 'short_H4').
    Each value is a list of param dicts (ranked by rank_in_cell).
    """
    import pandas as pd

    path = resolve_mode_file(TBM_BEST_PARAMS_PER_SIDE_HORIZON_CSV)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        if "rank_in_cell" in df.columns:
            df = df.sort_values(["cell_key", "rank_in_cell"], ascending=[True, True])
        result: Dict[str, list[Dict[str, Any]]] = {}
        for _, row in df.iterrows():
            cell = str(row.get("cell_key", ""))
            if not cell:
                continue
            result.setdefault(cell, []).append(
                {k: _coerce_numeric_if_possible(_to_scalar(v)) for k, v in row.items()}
            )
        return result
    except Exception:
        return {}


_TBM_BUCKET_KEY_MAP = {
    "k_tp": "barrier_k_tp",
    "sl_as_tp_pct": "barrier_sl_base_mult",
    "tp_abs_lo_pct": "barrier_tp_lo",
    "tp_abs_hi_pct": "barrier_tp_hi",
    "sl_abs_lo_pct": "barrier_sl_lo",
    "sl_abs_hi_pct": "barrier_sl_hi",
    "tp_base_pct": "barrier_tp_base_pct",
    "tp_method": "barrier_tp_method",
    "sl_method": "barrier_sl_method",
    "base_atr_window": "barrier_atr_window",
    "horizon_base": "label_horizon_base",
    "horizon_scaling": "label_horizon_scaling",
    "mode": "barrier_mode",
}


def apply_per_bucket_tbm_params_to_cfg(
    strategy_id: str,
    cfg: Dict[str, Any],
    *,
    per_strategy_params: Dict[str, Dict[str, Any]] | None = None,
    fallback_to_global: bool = True,
) -> Dict[str, Any]:
    """Inject the winning TBM barrier geometry for ``strategy_id`` into ``cfg``.

    Parameters
    ----------
    strategy_id:
        One of TBM_BUCKET_NAMES ('long_tf', 'long_mr', 'short_tf', 'short_mr').
    cfg:
        The run config dict to update in-place (a deepcopy is returned).
    per_strategy_params:
        Pre-loaded output of load_tbm_best_params_per_bucket() — avoids re-reading
        the CSV on every call.  If None, the file is read on demand.
    fallback_to_global:
        If True and no per-strategy entry is found for ``strategy_id``, falls back to the
        global TBM best params (apply_offline_optimizer_best_params).

    Returns
    -------
    A shallow copy of ``cfg`` with barrier keys injected.
    """
    import logging as _logging

    _log = _logging.getLogger("params_store")

    if per_strategy_params is None:
        per_strategy_params = load_tbm_best_params_per_bucket()

    bkt_params = per_strategy_params.get(strategy_id)
    if not bkt_params:
        if fallback_to_global:
            _log.warning(
                "[params_store] No per-strategy params for strategy_id=%s — falling back to global best",
                strategy_id,
            )
            return apply_offline_optimizer_best_params(cfg)
        return cfg

    from copy import deepcopy

    merged = deepcopy(cfg)
    injected: Dict[str, Any] = {}
    for src, dst in _TBM_BUCKET_KEY_MAP.items():
        if src in bkt_params and bkt_params[src] is not None:
            merged[dst] = bkt_params[src]
            injected[dst] = bkt_params[src]
    _log.info(
        "[params_store] Per-strategy TBM params for %s: %s",
        strategy_id,
        "  ".join(f"{k}={v}" for k, v in sorted(injected.items())),
    )
    return merged


# Backward compatibility alias
apply_per_strategy_tbm_params_to_cfg = apply_per_bucket_tbm_params_to_cfg


def apply_offline_optimizer_best_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import logging as _logging

    _log = _logging.getLogger("params_store")

    def _tprint(msg: str) -> None:
        import datetime as _dt

        ts = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC] {msg}", flush=True)
        _log.info(msg)

    merged = deepcopy(cfg)
    market_mode = merged.get("market_mode")

    cand_path = market_report_path(CANDIDATE_BEST_PARAMS_CSV, market_mode)
    cand = _read_best_params_csv(cand_path)
    if cand:
        for key in (
            "train_extreme_pct_hourly",
            "train_min_move_12h_pct",
            "train_min_range_pct",
            "train_min_vol_zscore",
            "train_candidate_metric",
        ):
            if key in cand and cand[key] is not None:
                merged[key] = cand[key]

    # Incorporate mask_optimiser output from the active market-specific files.
    mask_opt_path = market_report_path(
        INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV, market_mode
    )
    mask_opt = _read_best_params_csv(mask_opt_path)
    if mask_opt:
        for key in ("family", "param", "z_hours", "conditioner_mode", "duration_hours"):
            if key in mask_opt and mask_opt[key] is not None:
                merged[key] = mask_opt[key]

    mode_mask_opt = load_inference_candidate_mask_params_by_mode(
        market_mode=market_mode
    )
    if mode_mask_opt:
        merged["candidate_mask_params_by_mode"] = mode_mask_opt

    dyn_top_n = int(os.environ.get("EPM_MASK_STRATEGY_TOP_N", "2") or 2)
    dyn_ranking_metric = str(
        os.environ.get("EPM_MASK_STRATEGY_RANKING_METRIC", "score_for_best_params")
        or "score_for_best_params"
    )
    dyn_class_filter = str(
        os.environ.get("EPM_MASK_STRATEGY_CLASSIFICATION_FILTER", "") or ""
    ).strip()
    dyn_strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=max(1, dyn_top_n),
        ranking_metric=dyn_ranking_metric,
        classification_filter=dyn_class_filter or None,
        market_mode=market_mode,
    )
    if dyn_strategies:
        _tprint(
            f"[params_store] Loaded {len(dyn_strategies)} dynamic strategies from mask_optimiser"
        )
        merged["strategies"] = dyn_strategies
        # Maintain backward compatibility for single-mode logic if needed
        if "family" not in merged and len(dyn_strategies) > 0:
            merged.update(dyn_strategies[0].get("mask_params", {}))

    tbm_path = market_report_path(TBM_BEST_PARAMS_CSV, market_mode)
    tbm = _read_best_params_csv(tbm_path)
    if tbm:
        _tprint(
            f"[params_store] TBM best params loaded from {tbm_path}: "
            f"config_id={tbm.get('config_id','?')}  "
            f"base_atr_window={tbm.get('base_atr_window','?')}  "
            f"k_tp={tbm.get('k_tp','?')}  "
            f"sl_as_tp_pct={tbm.get('sl_as_tp_pct','?')}  "
            f"tp_base_pct={tbm.get('tp_base_pct','?')}  "
            f"mode={tbm.get('mode','?')}  "
            f"horizon_scaling={tbm.get('horizon_scaling','?')}"
        )
        key_map = {
            # Barrier geometry
            "k_tp": "barrier_k_tp",
            "sl_as_tp_pct": "barrier_sl_base_mult",
            "tp_abs_lo_pct": "barrier_tp_lo",
            "tp_abs_hi_pct": "barrier_tp_hi",
            "sl_abs_lo_pct": "barrier_sl_lo",
            "sl_abs_hi_pct": "barrier_sl_hi",
            "tp_base_pct": "barrier_tp_base_pct",
            "tp_abs_pct": "barrier_tp_abs_pct",
            # ATR method + window — mapped to barrier_atr_window (read by training.py)
            "tp_method": "barrier_tp_method",
            "sl_method": "barrier_sl_method",
            "base_atr_window": "barrier_atr_window",
            # Horizon
            "horizon_base": "label_horizon_base",
            "horizon_scaling": "label_horizon_scaling",
            # Mode tag (canonical, stripped of internal suffixes by compare_tbm_parameters)
            "mode": "barrier_mode",
        }
        injected = {}
        for src, dst in key_map.items():
            if src in tbm and tbm[src] is not None:
                merged[dst] = tbm[src]
                injected[dst] = tbm[src]
        _tprint(
            f"[params_store] Injected into cfg: "
            + "  ".join(f"{k}={v}" for k, v in sorted(injected.items()))
        )
    else:
        _tprint(
            f"[params_store] WARNING: TBM best params CSV not found or empty at {tbm_path} — using cfg defaults"
        )

    sw = _read_best_params_csv(
        market_report_path(SAMPLE_WEIGHT_BEST_PARAMS_CSV, market_mode)
    )
    if sw:
        if "component_alphas" in sw and isinstance(sw["component_alphas"], dict):
            merged["sample_weight_component_alphas"] = sw["component_alphas"]
        if "component_alphas_base" in sw and isinstance(
            sw["component_alphas_base"], dict
        ):
            merged["sample_weight_component_alphas_base"] = sw["component_alphas_base"]
        if "component_alphas_meta" in sw and isinstance(
            sw["component_alphas_meta"], dict
        ):
            merged["sample_weight_component_alphas_meta"] = sw["component_alphas_meta"]
        for key in (
            "sample_weight_vol_power",
            "sample_weight_distance_k",
            "sample_weight_distance_min_dist",
            "sample_weight_recency_half_life_bars",
        ):
            if key in sw and sw[key] is not None:
                merged[key] = sw[key]

    return merged
