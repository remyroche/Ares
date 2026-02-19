from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


OFFLINE_OPTIMISERS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = OFFLINE_OPTIMISERS_DIR / "reports"

CANDIDATE_BEST_PARAMS_CSV = REPORTS_DIR / "candidate_thresholds_best_params.csv"
TBM_BEST_PARAMS_CSV = REPORTS_DIR / "tbm_best_params.csv"
TBM_GEOMETRY_GRID_CSV = REPORTS_DIR / "tbm_geometry_grid.csv"
SAMPLE_WEIGHT_BEST_PARAMS_CSV = REPORTS_DIR / "sample_weight_best_params.csv"

TBM_BUCKET_NAMES = ["TF_long", "TF_short", "MR_long", "MR_short"]


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


def _read_best_params_csv(path: Path) -> Dict[str, Any]:
    import pandas as pd

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

    _empty = {"per_cell": {}, "k_tp_grid": None, "sl_base_grid": None, "atr_window": None}
    if not TBM_GEOMETRY_GRID_CSV.exists():
        return _empty
    try:
        df = pd.read_csv(TBM_GEOMETRY_GRID_CSV)
        if df.empty:
            return _empty

        # Global fallbacks
        k_tp_grid = sorted(df["k_tp"].dropna().unique().tolist()) if "k_tp" in df.columns else None
        sl_base_grid = sorted(df["sl_as_tp_pct"].dropna().unique().tolist()) if "sl_as_tp_pct" in df.columns else None
        atr_window = int(df["base_atr_window"].iloc[0]) if "base_atr_window" in df.columns else None

        # Per-cell grids (new format has "cell_key" column)
        per_cell: Dict[str, Any] = {}
        if "cell_key" in df.columns:
            for cell_key, grp in df.groupby("cell_key"):
                _tp_lo_vals = grp["tp_abs_lo_pct"].dropna().unique().tolist() if "tp_abs_lo_pct" in grp.columns else []
                _sl_lo_vals = grp["sl_abs_lo_pct"].dropna().unique().tolist() if "sl_abs_lo_pct" in grp.columns else []
                # Validated triplets: exact (k_tp, sl_as_tp_pct, atr_window) per optimizer row.
                # The window is part of each validated config — callers iterate these triplets
                # directly, pre-computing one barrier base per unique window and reusing it.
                _triplets: list = []
                _has_win = "base_atr_window" in grp.columns
                _cols = ["k_tp", "sl_as_tp_pct"] + (["base_atr_window"] if _has_win else [])
                for _, row in grp[_cols].dropna().iterrows():
                    win = int(row["base_atr_window"]) if _has_win else (atr_window or 720)
                    triplet = (round(float(row["k_tp"]), 6), round(float(row["sl_as_tp_pct"]), 6), win)
                    if triplet not in _triplets:
                        _triplets.append(triplet)
                # Unique windows needed to pre-compute barrier bases (one per window, reused).
                _win_vals: list = sorted(set(t[2] for t in _triplets))
                _first_win = _win_vals[0] if _win_vals else atr_window
                per_cell[str(cell_key)] = {
                    "k_tp_grid": sorted(grp["k_tp"].dropna().unique().tolist()),
                    "sl_base_grid": sorted(grp["sl_as_tp_pct"].dropna().unique().tolist()),
                    "validated_triplets": _triplets,
                    "validated_pairs": [(t[0], t[1]) for t in _triplets],
                    "atr_windows": _win_vals,
                    "atr_window": _first_win,
                    "tp_abs_lo_pct": float(min(_tp_lo_vals)) if _tp_lo_vals else None,
                    "sl_abs_lo_pct": float(min(_sl_lo_vals)) if _sl_lo_vals else None,
                }

        return {"per_cell": per_cell, "k_tp_grid": k_tp_grid, "sl_base_grid": sl_base_grid, "atr_window": atr_window}
    except Exception:
        return _empty



def save_best_params_csv(path: Path, best_params: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> Path:
    import pandas as pd

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    payload.update({k: _to_scalar(v) for k, v in (metadata or {}).items()})
    payload.update({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else _to_scalar(v) for k, v in best_params.items()})
    payload["saved_at"] = pd.Timestamp.utcnow().isoformat()
    pd.DataFrame([payload]).to_csv(path, index=False)
    return path


def apply_offline_optimizer_best_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import logging as _logging
    _log = _logging.getLogger("params_store")

    def _tprint(msg: str) -> None:
        import datetime as _dt
        ts = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC] {msg}", flush=True)
        _log.info(msg)

    merged = deepcopy(cfg)

    cand = _read_best_params_csv(CANDIDATE_BEST_PARAMS_CSV)
    if cand:
        for key in ("train_extreme_pct_hourly", "train_min_range_pct", "train_min_vol_zscore", "min_feat_sign_consistency"):
            if key in cand and cand[key] is not None:
                merged[key] = cand[key]

    tbm = _read_best_params_csv(TBM_BEST_PARAMS_CSV)
    if tbm:
        _tprint(
            f"[params_store] TBM best params loaded from {TBM_BEST_PARAMS_CSV}: "
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
        _tprint(f"[params_store] WARNING: TBM best params CSV not found or empty at {TBM_BEST_PARAMS_CSV} — using cfg defaults")

    sw = _read_best_params_csv(SAMPLE_WEIGHT_BEST_PARAMS_CSV)
    if sw:
        if "component_alphas" in sw and isinstance(sw["component_alphas"], dict):
            merged["sample_weight_component_alphas"] = sw["component_alphas"]
        if "component_alphas_base" in sw and isinstance(sw["component_alphas_base"], dict):
            merged["sample_weight_component_alphas_base"] = sw["component_alphas_base"]
        if "component_alphas_meta" in sw and isinstance(sw["component_alphas_meta"], dict):
            merged["sample_weight_component_alphas_meta"] = sw["component_alphas_meta"]
        for key in ("sample_weight_vol_power", "sample_weight_distance_k", "sample_weight_distance_min_dist", "sample_weight_recency_half_life_bars"):
            if key in sw and sw[key] is not None:
                merged[key] = sw[key]

    return merged
