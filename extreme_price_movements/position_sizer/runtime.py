import hashlib
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.tprint import tprint
from .sizer import PositionSizerConfig, size_positions_ranked

SUPPORTED_BUNDLE_VERSION = 1


@dataclass
class EVDecompositionBundle:
    feature_cols: list[str]
    pwin_model: object
    win_model: dict
    loss_model: dict
    tp_sl_defaults: dict | None
    config: dict
    version: str = "v1"
    bundle_version: int = SUPPORTED_BUNDLE_VERSION
    created_at: str = ""
    git_sha: str = ""
    schema_hash: str = ""



def compute_schema_hash(feature_cols: list[str], extra: dict | None = None) -> str:
    tprint("Computing EV decomposition schema hash")
    payload = {"feature_cols": list(feature_cols), "extra": extra or {}}
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def load_ev_decomposition_bundle(bundle_path, allow_unknown_version: bool = False) -> EVDecompositionBundle:
    tprint(f"Loading EVDecompositionBundle from {bundle_path}")
    with open(bundle_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, EVDecompositionBundle):
        b = obj
    else:
        b = EVDecompositionBundle(**obj)

    if not int(getattr(b, "bundle_version", 1)) == SUPPORTED_BUNDLE_VERSION and not allow_unknown_version:
        tprint(f"Unsupported EVDecompositionBundle version={getattr(b, 'bundle_version', 'NA')} (supported={SUPPORTED_BUNDLE_VERSION})")
        raise RuntimeError(
            f"Unsupported EVDecompositionBundle version={getattr(b, 'bundle_version', 'NA')} "
            f"(supported={SUPPORTED_BUNDLE_VERSION})"
        )
    tprint(f"Successfully loaded EVDecompositionBundle version {getattr(b, 'bundle_version', 'NA')}")
    return b



def _ensure_X(X_batch, feature_cols):
    tprint(f"Ensuring X_batch has required features: {len(feature_cols)} columns")
    if isinstance(X_batch, pd.DataFrame):
        missing = [c for c in feature_cols if c not in X_batch.columns]
        if missing:
            tprint(f"Missing required EV decomposition features: {missing[:10]}")
            raise ValueError(f"Missing required EV decomposition features: {missing[:10]}")
        X_df = X_batch.loc[:, feature_cols].copy()
    else:
        arr = np.asarray(X_batch, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != len(feature_cols):
            tprint(f"Array feature width mismatch: got {arr.shape[1]}, expected {len(feature_cols)}")
            raise ValueError(
                f"Array feature width mismatch: got {arr.shape[1]}, expected {len(feature_cols)}"
            )
        X_df = pd.DataFrame(arr, columns=feature_cols)

    for c in feature_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    if X_df[feature_cols].isna().any().any():
        bad = X_df[feature_cols].isna().sum()
        tprint(f"NaNs found in EV decomposition features: {bad[bad > 0].to_dict()}")
        raise ValueError(f"NaNs found in EV decomposition features: {bad[bad > 0].to_dict()}")
    tprint("X_batch validation successful")
    return X_df


def predict_ev_components(bundle: EVDecompositionBundle, X_batch, regime_labels=None, row_ids=None) -> dict:
    tprint("Predicting with EV decomposition bundle models")
    X_df = _ensure_X(X_batch, bundle.feature_cols)
    pwin = bundle.pwin_model.predict_proba(X_df.values, regime_labels=regime_labels, row_ids=row_ids)[:, 1]
    qwin50 = np.maximum(bundle.win_model["q50"].predict(X_df.values), 0.0)
    qwin80 = np.maximum(bundle.win_model["q80"].predict(X_df.values), qwin50)
    qloss50 = np.maximum(bundle.loss_model["q50"].predict(X_df.values), 0.0)
    qloss90 = np.maximum(bundle.loss_model["q90"].predict(X_df.values), qloss50)
    tprint("EV decomposition predictions completed successfully")
    return {
        "pwin": pwin,
        "qwin50": qwin50,
        "qwin80": qwin80,
        "qloss50": qloss50,
        "qloss90": qloss90,
    }



def compute_ev_risk(preds: dict, costs: float, cfg: PositionSizerConfig):
    tprint("Computing EV and Risk")
    p = np.clip(np.asarray(preds["pwin"], dtype=float), cfg.p_min, 1.0 - cfg.p_min)
    W = np.asarray(preds["qwin80" if cfg.exp_win_quantile >= 0.8 else "qwin50"], dtype=float)
    L = np.asarray(preds["qloss90" if cfg.risk_loss_quantile >= 0.9 else "qloss50"], dtype=float)
    W = np.maximum(W, 0.0)
    L = np.maximum(L, 0.0)

    if cfg.costs_mode == "included_in_labels" and abs(float(costs)) > 1e-12:
        tprint("Double-cost risk: costs_mode='included_in_labels' requires costs=0 in EV computation")
        raise ValueError("Double-cost risk: costs_mode='included_in_labels' requires costs=0 in EV computation")

    ev = p * W - (1.0 - p) * L - float(costs)
    risk = (1.0 - p) * L
    tprint(f"Computed EV (mean: {np.mean(ev):.4f}) and Risk (mean: {np.mean(risk):.4f})")
    return ev, risk


def gate_and_size(EV, Risk, cfg: PositionSizerConfig, optional_rank_context=None, alpha_score=None):
    tprint("Gating and sizing positions")
    ev = np.asarray(EV, dtype=float)
    risk = np.asarray(Risk, dtype=float)
    if alpha_score is None:
        alpha_score = np.ones_like(ev)
    out = size_positions_ranked(
        ev_hat=ev,
        risk_hat=risk,
        alpha_score=np.asarray(alpha_score, dtype=float),
        cfg=cfg,
        group_ids=optional_rank_context,
    )
    tprint(f"Position sizing complete. Allowed {np.sum(out['trade_allowed'])}/{len(ev)} trades")
    return out["trade_allowed"], out["size"]


def make_bundle_metadata(git_sha: str = "") -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
    }
