from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


OFFLINE_OPTIMISERS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = OFFLINE_OPTIMISERS_DIR / "reports"

CANDIDATE_BEST_PARAMS_CSV = REPORTS_DIR / "candidate_thresholds_best_params.csv"
TBM_BEST_PARAMS_CSV = REPORTS_DIR / "tbm_best_params.csv"
SAMPLE_WEIGHT_BEST_PARAMS_CSV = REPORTS_DIR / "sample_weight_best_params.csv"


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
    merged = deepcopy(cfg)

    cand = _read_best_params_csv(CANDIDATE_BEST_PARAMS_CSV)
    if cand:
        for key in ("train_extreme_pct_hourly", "train_min_range_pct", "train_min_vol_zscore", "min_feat_sign_consistency"):
            if key in cand and cand[key] is not None:
                merged[key] = cand[key]

    tbm = _read_best_params_csv(TBM_BEST_PARAMS_CSV)
    if tbm:
        key_map = {
            "k_tp": "barrier_k_tp",
            "sl_as_tp_pct": "barrier_sl_base_mult",
            "tp_abs_lo_pct": "barrier_tp_lo",
            "tp_abs_hi_pct": "barrier_tp_hi",
            "horizon_base": "label_horizon_base",
        }
        for src, dst in key_map.items():
            if src in tbm and tbm[src] is not None:
                merged[dst] = tbm[src]

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
