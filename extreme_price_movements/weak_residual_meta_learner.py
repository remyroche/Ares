from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .utils import tprint
from .ebm_on_lgbm import (
    _metric_pack as _ebm_metric_pack,
    fit_ebm_on_lgbm_full_model,
    train_ebm_on_lgbm_candidate,
)

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


WR_META_CV_SPLITS = 3
WR_META_ALPHA_GRID = (0.1, 0.5, 1.0)
WR_META_L1_RATIO_GRID = (0.5, 0.8)
WR_META_SIGN_CONSISTENCY = (0.6, 0.7, 0.75, 0.8, 0.85)
WR_META_RESID_EPS = 1e-8
WR_META_EN_N_JOBS = 2
WR_META_EN_SUBSAMPLE_ROWS = 5000
WR_META_EN_TOL = 3e-4
WR_META_LOGIT_MAX_ITER = 2000
WR_META_EN_MAX_ITER = 4000


def _ranknorm(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if len(v) == 0:
        return v.astype(np.float32)
    order = np.argsort(v, kind="mergesort")
    out = np.zeros(len(v), dtype=np.float64)
    out[order] = np.linspace(0.0, 1.0, len(v), endpoint=True)
    return out.astype(np.float32)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 8:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)


def _vectorized_spearman_scores_2d(
    X: np.ndarray, y: np.ndarray, *, signed: bool = True
) -> np.ndarray:
    x = np.asarray(X, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    yv = np.asarray(y, dtype=np.float64)
    rows = np.isfinite(yv)
    if int(np.sum(rows)) < 8:
        return np.zeros(x.shape[1], dtype=np.float32)
    xr = pd.DataFrame(x[rows]).rank(pct=True).to_numpy(dtype=np.float64)
    yr = pd.Series(yv[rows]).rank(pct=True).to_numpy(dtype=np.float64)
    xr = xr - np.nanmean(xr, axis=0)
    yr = yr - np.nanmean(yr)
    x_std = np.sqrt(np.nanmean(xr * xr, axis=0))
    y_std = float(np.sqrt(np.nanmean(yr * yr)))
    denom = np.maximum(x_std * max(y_std, 1e-12), 1e-12)
    corr = np.nanmean(xr * yr[:, None], axis=0) / denom
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if not signed:
        corr = np.abs(corr)
    return corr.astype(np.float32)


def _ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    m = np.isfinite(y) & np.isfinite(p)
    if np.sum(m) < 10:
        return 0.0
    y = y[m]
    p = p[m]
    q = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
    q[0] -= 1e-12
    q[-1] += 1e-12
    ece = 0.0
    for i in range(bins):
        mm = (p >= q[i]) & (p < q[i + 1] if i < bins - 1 else p <= q[i + 1])
        if not np.any(mm):
            continue
        ece += abs(float(np.mean(y[mm])) - float(np.mean(p[mm]))) * (
            np.sum(mm) / len(p)
        )
    return float(ece)


def _norm_07_13(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return np.ones(len(x), dtype=np.float32)
    z = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (0.7 + 0.6 * z).astype(np.float32)


def _topk_mask(score: np.ndarray, frac: float = 0.30) -> np.ndarray:
    n = len(score)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(np.asarray(score, dtype=np.float64))[-k:]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def _stratified_subsample_idx(
    strata: np.ndarray, max_n: int, seed: int = 42
) -> np.ndarray:
    n = len(strata)
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(strata):
        ids = np.where(strata == s)[0]
        if len(ids) == 0:
            continue
        take = max(1, int(round(max_n * (len(ids) / n))))
        take = min(take, len(ids))
        out.append(rng.choice(ids, size=take, replace=False))
    if not out:
        return np.arange(max_n, dtype=np.int32)
    idx = np.sort(np.concatenate(out).astype(np.int32))
    if len(idx) > max_n:
        idx = np.sort(rng.choice(idx, size=max_n, replace=False).astype(np.int32))
    return idx


def _metric_pack(y: np.ndarray, pred: np.ndarray, classifier: bool) -> Dict[str, float]:
    # Internal proxy objective for feature filtering only (not canonical business metrics).
    m30 = _topk_mask(pred, 0.30)
    if classifier:
        yb = np.asarray(y > 0.5, dtype=np.float64)
        br = float(np.mean(yb))
        p30 = float(np.mean(yb[m30])) if np.any(m30) else 0.0
        lift = p30 / max(br, 1e-6)
        ic30 = _safe_spearman(pred[m30], yb[m30]) if np.any(m30) else 0.0
        # split top30 into 5 slices, use negative std as stability proxy
        s_top = np.asarray(pred[m30], dtype=np.float64)
        y_top = yb[m30]
        if len(s_top) >= 10:
            q = np.quantile(s_top, np.linspace(0.0, 1.0, 6))
            vals = []
            for i in range(5):
                mm = (s_top >= q[i]) & (
                    s_top < q[i + 1] if i < 4 else s_top <= q[i + 1]
                )
                if np.any(mm):
                    vals.append(float(np.mean(y_top[mm])))
            stab = float(1.0 / (1.0 + np.std(vals))) if vals else 0.0
        else:
            stab = 0.0
        brier = float(np.mean((np.clip(pred, 1e-4, 1 - 1e-4) - yb) ** 2))
        ece = _ece(yb, np.clip(pred, 1e-4, 1 - 1e-4), bins=10)
    else:
        yt = np.asarray(y, dtype=np.float64)
        # Custom regression lift proxy: can behave differently on centered/heavy-tailed targets.
        lift = (
            float(np.mean(yt[m30]) / (np.mean(np.abs(yt)) + 1e-6))
            if np.any(m30)
            else 0.0
        )
        ic30 = _safe_spearman(pred[m30], yt[m30]) if np.any(m30) else 0.0
        if np.any(m30):
            s_top = np.asarray(pred[m30], dtype=np.float64)
            y_top = yt[m30]
            q = np.quantile(s_top, np.linspace(0.0, 1.0, 6))
            vals = []
            for i in range(5):
                mm = (s_top >= q[i]) & (
                    s_top < q[i + 1] if i < 4 else s_top <= q[i + 1]
                )
                if np.any(mm):
                    vals.append(float(np.mean(y_top[mm])))
            stab = float(1.0 / (1.0 + np.std(vals))) if vals else 0.0
        else:
            stab = 0.0
        brier = float(np.mean((pred - yt) ** 2))
        ece = 0.0
    return {
        "lift30": lift,
        "ic30": ic30,
        "stability30": stab,
        "brier": brier,
        "ece": ece,
    }


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s < 1e-9:
        return np.zeros(len(x), dtype=np.float64)
    return (x - np.nanmean(x)) / s


def _fit_wr_en_combo(
    alpha: float,
    l1: float,
    fold_cache: list[dict[str, Any]],
    yv: np.ndarray,
    classifier: bool,
) -> Optional[dict[str, Any]]:
    oof = np.zeros(len(yv), dtype=np.float32)
    fold_stats = []
    coef = np.zeros(fold_cache[0]["n_features"], dtype=np.float64)
    convergence_hits = 0
    for fold in fold_cache:
        sub = fold["sub_idx"]
        try:
            if classifier:
                c_val = 1.0 / max(alpha, 1e-6)
                mdl = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=c_val,
                    max_iter=WR_META_LOGIT_MAX_ITER,
                    tol=WR_META_EN_TOL,
                    random_state=42,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    mdl.fit(fold["Xtr"][sub], fold["y_split_tr"][sub])
                pred = mdl.predict_proba(fold["Xva"])[:, 1].astype(np.float32)
                coef += np.abs(np.ravel(mdl.coef_))
            else:
                mdl = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1,
                    max_iter=WR_META_EN_MAX_ITER,
                    tol=WR_META_EN_TOL,
                    random_state=42,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    mdl.fit(fold["Xtr"][sub], fold["yv_tr"][sub])
                pred = mdl.predict(fold["Xva"]).astype(np.float32)
                coef += np.abs(mdl.coef_)
            convergence_hits += int(
                any(issubclass(w.category, ConvergenceWarning) for w in caught)
            )
            if hasattr(mdl, "n_iter_") and int(np.max(mdl.n_iter_)) >= (
                WR_META_LOGIT_MAX_ITER if classifier else WR_META_EN_MAX_ITER
            ):
                convergence_hits += 1
        except Exception:
            return None
        va = fold["va"]
        oof[va] = pred
        fold_stats.append(_metric_pack(yv[va], oof[va], classifier=classifier))
    if not fold_stats:
        return None
    metrics = {
        k: float(np.mean([r[k] for r in fold_stats])) for k in fold_stats[0].keys()
    }
    return {
        "metrics": metrics,
        "coef": coef / max(1, len(fold_cache)),
        "convergence_hits": int(convergence_hits),
    }


def _preridge_elasticnet_select(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    classifier: bool,
    max_keep: int = 120,
) -> list[str]:
    cols = list(X.columns)
    if len(cols) <= max_keep:
        return cols
    Xv = X.to_numpy(dtype=np.float32)
    yv = np.asarray(y, dtype=np.float32)
    grid = [(a, l) for a in WR_META_ALPHA_GRID for l in WR_META_L1_RATIO_GRID]
    cv = (
        StratifiedKFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
        if classifier
        else KFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
    )
    y_split = yv.astype(np.int8) if classifier else yv
    fold_cache: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(cv.split(Xv, y_split), start=1):
        sc = RobustScaler()
        Xtr = sc.fit_transform(Xv[tr])
        Xva = sc.transform(Xv[va])
        if classifier:
            sub_strata = y_split[tr]
        else:
            sub_rank = pd.Series(yv[tr]).rank(pct=True).to_numpy(dtype=np.float32)
            sub_strata = np.clip((sub_rank * 5).astype(np.int32), 0, 4)
        sub_idx = _stratified_subsample_idx(
            sub_strata,
            max_n=WR_META_EN_SUBSAMPLE_ROWS,
            seed=42 + fold_id,
        )
        fold_cache.append(
            {
                "tr": tr,
                "va": va,
                "Xtr": Xtr,
                "Xva": Xva,
                "yv_tr": yv[tr],
                "y_split_tr": y_split[tr],
                "sub_idx": sub_idx,
                "n_features": Xv.shape[1],
            }
        )
    tprint(
        f"WRMeta EN selector: {len(grid)} hyperparameter combos across "
        f"{len(fold_cache)} cached folds (subsample={WR_META_EN_SUBSAMPLE_ROWS}, "
        f"n_jobs={WR_META_EN_N_JOBS}, tol={WR_META_EN_TOL:g})"
    )
    results = Parallel(n_jobs=WR_META_EN_N_JOBS, prefer="threads")(
        delayed(_fit_wr_en_combo)(
            alpha,
            l1,
            fold_cache,
            yv,
            classifier,
        )
        for alpha, l1 in grid
    )
    valid = [r for r in results if r is not None]
    if not valid:
        return cols[:max_keep]
    conv_hits = int(sum(r.get("convergence_hits", 0) for r in valid))
    if conv_hits > 0:
        tprint(
            f"WRMeta EN selector convergence monitor: {conv_hits} fold fits hit "
            f"warning/max_iter (tol={WR_META_EN_TOL:g}, n_jobs={WR_META_EN_N_JOBS})"
        )
    rec = [r["metrics"] for r in valid]
    coef_maps = [r["coef"] for r in valid]
    lift_z = _z(np.array([r["lift30"] for r in rec]))
    ic_z = _z(np.array([r["ic30"] for r in rec]))
    stab_z = _z(np.array([r["stability30"] for r in rec]))
    brier_z = _z(np.array([r["brier"] for r in rec]))
    ece_z = _z(np.array([r["ece"] for r in rec]))
    obj = 0.35 * lift_z + 0.35 * ic_z + 0.20 * stab_z - 0.05 * brier_z - 0.05 * ece_z
    best = int(np.nanargmax(obj))
    coef = coef_maps[best]
    order = np.argsort(coef)[::-1][:max_keep]
    return [cols[i] for i in order]


def _cluster_redundant_features(
    X: pd.DataFrame, y: np.ndarray, thr: float = 0.98
) -> list[str]:
    cols = list(X.columns)
    if len(cols) <= 2:
        return cols
    xs = X.to_numpy(dtype=np.float32)
    sub_n = min(len(xs), 5000)
    rng = np.random.default_rng(42)
    idx = (
        rng.choice(len(xs), size=sub_n, replace=False)
        if len(xs) > sub_n
        else np.arange(len(xs))
    )
    sub = pd.DataFrame(xs[idx], columns=cols).rank(pct=True)
    corr = np.abs(np.corrcoef(sub.to_numpy(dtype=np.float64), rowvar=False))
    yv = np.asarray(y, dtype=np.float64)
    rel = np.abs(_vectorized_spearman_scores_2d(xs, yv, signed=True)).astype(np.float64)
    keep = []
    dropped = np.zeros(len(cols), dtype=bool)
    for i in np.argsort(rel)[::-1]:
        if dropped[i]:
            continue
        keep.append(cols[i])
        drop_i = corr[i] > thr
        dropped |= drop_i
        dropped[i] = False
    return keep


def _univariate_screen(
    X: pd.DataFrame,
    y: np.ndarray,
    ridge_pred: np.ndarray,
    signed_residual: np.ndarray,
    *,
    top_k: int = 150,
) -> list[str]:
    cols = list(X.columns)
    if len(cols) <= top_k:
        return cols
    rp = _ranknorm(ridge_pred)
    buckets = np.clip((rp * 5).astype(np.int32), 0, 4)
    idx = _stratified_subsample_idx(buckets, max_n=25000, seed=42)
    Xs = X.iloc[idx]
    xs_mat = Xs.to_numpy(dtype=np.float32)
    y_s = np.asarray(y, dtype=np.float64)[idx]
    r_s = np.asarray(ridge_pred, dtype=np.float64)[idx]
    sr_s = np.asarray(signed_residual, dtype=np.float64)[idx]
    b_s = buckets[idx]
    c1_rows: list[np.ndarray] = []
    c2_rows: list[np.ndarray] = []
    for b in range(5):
        m = b_s == b
        if np.sum(m) < 8:
            continue
        c1_rows.append(_vectorized_spearman_scores_2d(xs_mat[m], sr_s[m], signed=True))
        c2_rows.append(
            _vectorized_spearman_scores_2d(
                xs_mat[m] * r_s[m, None].astype(np.float32), y_s[m], signed=True
            )
        )
    if not c1_rows:
        return cols[:top_k]
    c1 = np.vstack(c1_rows).astype(np.float64, copy=False)
    c2 = np.vstack(c2_rows).astype(np.float64, copy=False)
    scores = (
        np.abs(np.nanmean(c1, axis=0))
        - 0.5 * np.nanstd(c1, axis=0)
        + np.abs(np.nanmean(c2, axis=0))
        - 0.5 * np.nanstd(c2, axis=0)
    )
    scores = np.nan_to_num(scores, nan=-1e9, posinf=-1e9, neginf=-1e9)
    ord_idx = np.argsort(scores)[::-1][:top_k]
    return [cols[i] for i in ord_idx]


def _iterative_fold_presence_prune(
    X: pd.DataFrame,
    target: np.ndarray,
    *,
    classifier: bool,
    min_features: int = 40,
) -> list[str]:
    cols = list(X.columns)
    if len(cols) <= min_features or lgb is None:
        return cols
    presence = {c: 0 for c in cols}
    gain = {c: 0.0 for c in cols}
    y_split = target.astype(np.int32) if classifier else target
    if classifier:
        _classes, _counts = np.unique(y_split, return_counts=True)
        if len(_classes) >= 2 and int(np.min(_counts)) >= 5:
            cv = StratifiedKFold(
                n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42
            )
        else:
            cv = KFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
    Xv = X.to_numpy(dtype=np.float32)
    yv = np.asarray(target)
    for tr, va in cv.split(Xv, y_split):
        if classifier:
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=500,
                max_depth=2,
                min_data_in_leaf=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=2,
            )
        else:
            model = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=500,
                max_depth=2,
                min_data_in_leaf=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=2,
            )
        model.fit(
            Xv[tr],
            yv[tr],
            eval_set=[(Xv[va], yv[va])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        imp = model.booster_.feature_importance(importance_type="gain")
        nz = imp > 0
        for i, c in enumerate(cols):
            if nz[i]:
                presence[c] += 1
                gain[c] += float(imp[i])
    curr = set(cols)
    for ratio in WR_META_SIGN_CONSISTENCY:
        need = max(1, int(np.ceil(WR_META_CV_SPLITS * ratio)))
        curr = {c for c in curr if presence.get(c, 0) >= need}
        if len(curr) <= min_features:
            break
    if len(curr) < min_features:
        rank = sorted(
            cols, key=lambda c: (presence.get(c, 0), gain.get(c, 0.0)), reverse=True
        )
        curr = set(rank[:min_features])
    return [c for c in cols if c in curr]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _resolve_atr_scale(
    X: pd.DataFrame,
    n: int,
    kwargs: dict[str, Any],
    target_horizon: Optional[int],
) -> np.ndarray:
    for key in (
        "atr",
        "atr_scale",
        "atr_target",
        "atr_4h",
        "atr_pct",
        "residual_atr",
    ):
        val = kwargs.get(key)
        if val is not None:
            arr = np.asarray(val, dtype=np.float32)
            if len(arr) == n:
                return np.maximum(np.abs(arr), WR_META_RESID_EPS).astype(np.float32)

    if target_horizon is not None and target_horizon > 0:
        for col in (
            f"atr_{4 * int(target_horizon)}",
            f"atr_pct_{4 * int(target_horizon)}",
            f"atr_h{4 * int(target_horizon)}",
        ):
            if col in X.columns:
                arr = X[col].to_numpy(dtype=np.float32)
                return np.maximum(np.abs(arr), WR_META_RESID_EPS).astype(np.float32)

    for col in ("atr", "ATR", "atr_pct", "atr_ratio"):
        if col in X.columns:
            arr = X[col].to_numpy(dtype=np.float32)
            return np.maximum(np.abs(arr), WR_META_RESID_EPS).astype(np.float32)

    return np.ones(n, dtype=np.float32)


def _mad_scale(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if not np.isfinite(mad) or mad < WR_META_RESID_EPS:
        mad = float(np.nanstd(v))
    return max(mad, WR_META_RESID_EPS)


def _build_residual_target(
    *,
    y_base_target: np.ndarray,
    base_oof_pred: np.ndarray,
    atr_scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    raw_resid = (
        np.asarray(y_base_target, dtype=np.float32)
        - np.asarray(base_oof_pred, dtype=np.float32)
    ).astype(np.float32)
    resid_atr = raw_resid / (
        np.asarray(atr_scale, dtype=np.float32) + WR_META_RESID_EPS
    )
    mad = _mad_scale(resid_atr)
    resid_z = np.clip(resid_atr / (mad + WR_META_RESID_EPS), -5.0, 5.0)
    target = (0.7 * np.arcsinh(resid_z) + 0.3 * resid_z).astype(np.float32)
    diag = {
        "resid_mad": float(mad),
        "resid_target_std": float(np.nanstd(target)),
        "raw_resid_std": float(np.nanstd(raw_resid)),
        "atr_scale_median": float(np.nanmedian(atr_scale)),
    }
    return target, diag


def _lgbm_leaf_ridge_fit(
    X: pd.DataFrame,
    target: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    n = len(X)
    empty = {
        "oof_pred": np.zeros(n, dtype=np.float32),
        "model": None,
        "scaler": None,
        "encoder": None,
        "ridge": None,
        "features": list(X.columns),
        "score": 0.0,
    }
    if lgb is None or X.shape[1] == 0 or n < WR_META_CV_SPLITS * 8:
        return empty

    Xv = X.to_numpy(dtype=np.float32)
    yv = np.asarray(target, dtype=np.float32)
    sw = (
        np.ones(n, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )
    cv = KFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=random_state)
    oof = np.zeros(n, dtype=np.float32)
    for fold_id, (tr, va) in enumerate(cv.split(Xv)):
        tree = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=250,
            max_depth=3,
            min_data_in_leaf=80,
            learning_rate=0.04,
            random_state=random_state + fold_id,
            n_jobs=2,
            verbosity=-1,
        )
        tree.fit(Xv[tr], yv[tr], sample_weight=sw[tr])
        leaves_tr = np.asarray(tree.predict(Xv[tr], pred_leaf=True), dtype=np.int32)
        leaves_va = np.asarray(tree.predict(Xv[va], pred_leaf=True), dtype=np.int32)
        if leaves_tr.ndim == 1:
            leaves_tr = leaves_tr.reshape(-1, 1)
            leaves_va = leaves_va.reshape(-1, 1)
        enc = _one_hot_encoder()
        Ltr = enc.fit_transform(leaves_tr)
        Lva = enc.transform(leaves_va)
        sc = RobustScaler()
        Xtr = sparse.hstack(
            [sparse.csr_matrix(sc.fit_transform(Xv[tr])), Ltr], format="csr"
        )
        Xva = sparse.hstack(
            [sparse.csr_matrix(sc.transform(Xv[va])), Lva], format="csr"
        )
        ridge = Ridge(alpha=1.0, random_state=random_state)
        ridge.fit(Xtr, yv[tr], sample_weight=sw[tr])
        oof[va] = ridge.predict(Xva).astype(np.float32)

    full_tree = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=250,
        max_depth=3,
        min_data_in_leaf=80,
        learning_rate=0.04,
        random_state=random_state,
        n_jobs=2,
        verbosity=-1,
    )
    full_tree.fit(Xv, yv, sample_weight=sw)
    leaves = np.asarray(full_tree.predict(Xv, pred_leaf=True), dtype=np.int32)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    encoder = _one_hot_encoder()
    L = encoder.fit_transform(leaves)
    scaler = RobustScaler()
    design = sparse.hstack(
        [sparse.csr_matrix(scaler.fit_transform(Xv)), L], format="csr"
    )
    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(design, yv, sample_weight=sw)
    empty.update(
        {
            "oof_pred": oof.astype(np.float32),
            "model": full_tree,
            "scaler": scaler,
            "encoder": encoder,
            "ridge": ridge,
            "score": _safe_spearman(oof, yv),
        }
    )
    return empty


def _lgbm_leaf_ridge_predict(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    encoder = bundle.get("encoder")
    ridge = bundle.get("ridge")
    if model is None or scaler is None or encoder is None or ridge is None:
        return np.zeros(len(X), dtype=np.float32)
    Xv = X.to_numpy(dtype=np.float32)
    leaves = np.asarray(model.predict(Xv, pred_leaf=True), dtype=np.int32)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    design = sparse.hstack(
        [sparse.csr_matrix(scaler.transform(Xv)), encoder.transform(leaves)],
        format="csr",
    )
    return ridge.predict(design).astype(np.float32)


class WeakResidualMetaRegressor:
    def __init__(
        self,
        strategy_name: Optional[str] = None,
        reports_dir: Optional[str] = None,
        lambda_reg: float = 0.3,
    ):
        self.strategy_name = strategy_name
        self.reports_dir = reports_dir
        self.lambda_reg = lambda_reg
        self.selected_features: list[str] = []
        self.ridge_model: Optional[Pipeline] = None
        self.lgbm_model: Optional[Any] = None
        self.lgbm_leaf_ridge_bundle: Optional[dict[str, Any]] = None
        self.ebm_model: Optional[Any] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.model: Optional[Any] = None
        self._diag: Dict[str, np.ndarray] = {}
        self._leaf_var_maps: list[dict[int, float]] = []
        self._leaf_cnt_maps: list[dict[int, int]] = []
        self._reg_unc_a: float = 1.0
        self._reg_unc_C: float = 1.0
        self._reg_unc_lo: float = 0.0
        self._reg_unc_hi: float = 1.0

    def __repr__(self) -> str:
        n_feat = len(self.selected_features)
        has_lgbm = self.lgbm_model is not None
        return (
            f"WeakResidualMetaRegressor("
            f"strategy={self.strategy_name!r}, "
            f"features={n_feat}, "
            f"lambda_reg={self.lambda_reg}, "
            f"lgbm={has_lgbm}, "
            f"ebm={self.ebm_model is not None})"
        )

    def clear_diagnostics(self) -> None:
        self._diag = {}
        self._leaf_var_maps = []
        self._leaf_cnt_maps = []

    def _compute_reg_lgbm_uncertainty(
        self, X_lgb_np: np.ndarray
    ) -> dict[str, np.ndarray]:
        n = len(X_lgb_np)
        if (
            len(self._leaf_var_maps) == 0 or len(self._leaf_cnt_maps) == 0
        ) and isinstance(self._diag, dict):
            _v = self._diag.get("leaf_var_maps")
            _c = self._diag.get("leaf_cnt_maps")
            if _v is not None and _c is not None:
                try:
                    self._leaf_var_maps = list(_v)
                    self._leaf_cnt_maps = list(_c)
                except Exception:
                    self._leaf_var_maps = []
                    self._leaf_cnt_maps = []
        if (
            self.lgbm_model is None
            or not hasattr(self.lgbm_model, "booster_")
            or len(self._leaf_var_maps) == 0
            or len(self._leaf_cnt_maps) == 0
        ):
            ones = np.ones(n, dtype=np.float32)
            return {
                "leaf_var": np.zeros(n, dtype=np.float32),
                "leaf_count": ones.copy(),
                "support_factor": ones.copy(),
                "uncertainty": ones.copy(),
                "leaf_count_q25": ones.copy(),
            }
        leaf = self.lgbm_model.booster_.predict(X_lgb_np, pred_leaf=True)
        leaf = np.asarray(leaf, dtype=np.int64)
        if leaf.ndim == 1:
            leaf = leaf.reshape(-1, 1)
        n_trees = leaf.shape[1]
        lv = np.zeros(len(leaf), dtype=np.float64)
        lc = np.zeros(len(leaf), dtype=np.float64)
        lc_trees = np.zeros((len(leaf), n_trees), dtype=np.float64)
        for t in range(n_trees):
            vmap = self._leaf_var_maps[t] if t < len(self._leaf_var_maps) else {}
            cmap = self._leaf_cnt_maps[t] if t < len(self._leaf_cnt_maps) else {}
            ids = leaf[:, t]
            if vmap:
                max_leaf = max(vmap.keys()) + 1
                lookup_v = np.zeros(max_leaf, dtype=np.float64)
                for k, v in vmap.items():
                    lookup_v[k] = v
                safe_ids = np.clip(ids, 0, max_leaf - 1)
                lv += lookup_v[safe_ids]
            if cmap:
                max_leaf_c = max(cmap.keys()) + 1
                lookup_c = np.zeros(max_leaf_c, dtype=np.float64)
                lookup_c.fill(1.0)
                for k, v in cmap.items():
                    lookup_c[k] = v
                safe_ids_c = np.clip(ids, 0, max_leaf_c - 1)
                cnt_vals = lookup_c[safe_ids_c]
                lc += cnt_vals
                lc_trees[:, t] = cnt_vals
        mean_leaf_var = (lv / max(n_trees, 1)).astype(np.float32)
        mean_leaf_count = (lc / max(n_trees, 1)).astype(np.float32)
        support_factor = np.log1p(mean_leaf_count) / max(
            np.log1p(float(self._reg_unc_C)), 1e-6
        )
        unc_raw = support_factor / (1.0 + float(self._reg_unc_a) * mean_leaf_var)
        lo = float(self._reg_unc_lo)
        hi = float(self._reg_unc_hi)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            unc = 0.7 + 0.6 * np.clip((unc_raw - lo) / (hi - lo), 0.0, 1.0)
        else:
            unc = np.ones(len(unc_raw), dtype=np.float32)
        return {
            "leaf_var": mean_leaf_var.astype(np.float32),
            "leaf_count": mean_leaf_count.astype(np.float32),
            "support_factor": np.asarray(support_factor, dtype=np.float32),
            "uncertainty": np.asarray(unc, dtype=np.float32),
            "leaf_count_q25": np.nanpercentile(lc_trees, 25, axis=1).astype(np.float32),
        }

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        groups=None,
        y_per_horizon=None,
        y_binary=None,
        **kwargs,
    ):
        from extreme_price_movements.utils import tprint as _tprint

        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_t = np.asarray(y, dtype=np.float32)
        _tprint(f"WRMetaRegressor.fit: n={len(y_t)}, p={X_df.shape[1]}")

        ridge_feats = _preridge_elasticnet_select(
            X_df, y_t, classifier=False, max_keep=120
        )
        X_ridge = X_df[ridge_feats]
        X_ridge_np = X_ridge.to_numpy(dtype=np.float32)
        sw = (
            np.ones(len(X_ridge_np), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )

        cv = KFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
        ridge_oof_pred = np.zeros(len(X_ridge_np), dtype=np.float32)
        for tr, va in cv.split(X_ridge_np):
            fold = Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("ridge", Ridge(alpha=0.5, random_state=42)),
                ]
            )
            fold.fit(X_ridge_np[tr], y_t[tr], ridge__sample_weight=sw[tr])
            ridge_oof_pred[va] = fold.predict(X_ridge_np[va]).astype(np.float32)

        ridge = Pipeline(
            [("scaler", RobustScaler()), ("ridge", Ridge(alpha=0.5, random_state=42))]
        )
        ridge.fit(X_ridge_np, y_t, ridge__sample_weight=sw)
        _tprint(
            f"  Ridge CV done: {len(ridge_feats)} features, residual std={np.std(y_t - ridge_oof_pred):.4f}"
        )

        target_horizon = kwargs.get("target_horizon")
        if target_horizon is None and y_per_horizon is not None:
            target_horizon = kwargs.get("horizon")
        target_horizon_int = int(target_horizon) if target_horizon is not None else None
        y_base_target = np.asarray(kwargs.get("y_base_target", y_t), dtype=np.float32)
        base_pred_for_resid = np.asarray(
            kwargs.get("base_oof_pred", ridge_oof_pred), dtype=np.float32
        )
        atr_scale = _resolve_atr_scale(
            X_df, len(y_t), kwargs, target_horizon_int
        ).astype(np.float32)
        signed_residual, resid_diag = _build_residual_target(
            y_base_target=y_base_target,
            base_oof_pred=base_pred_for_resid,
            atr_scale=atr_scale,
        )
        ridge_target = y_base_target
        lgb_feats = _cluster_redundant_features(
            X_df[ridge_feats], signed_residual, thr=0.98
        )
        X_lgb0 = X_df[lgb_feats]
        lgb_feats = _univariate_screen(
            X_lgb0, ridge_target, ridge_oof_pred, signed_residual, top_k=150
        )
        lgb_feats = _cluster_redundant_features(
            X_lgb0[lgb_feats], signed_residual, thr=0.95
        )
        X_lgb1 = X_lgb0[lgb_feats]
        lgb_feats = _iterative_fold_presence_prune(
            X_lgb1, signed_residual, classifier=False, min_features=40
        )
        X_lgb = X_lgb1[lgb_feats]
        _tprint(
            f"  Feature selection: ridge={len(ridge_feats)} -> lgb={len(lgb_feats)}"
        )

        if lgb is not None and X_lgb.shape[1] > 0:
            X_lgb_np = X_lgb.to_numpy(dtype=np.float32)
            lgb_oof_pred = np.zeros(len(X_lgb_np), dtype=np.float32)
            cv_lgb = KFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
            for tr, va in cv_lgb.split(X_lgb_np):
                fold_lgb = lgb.LGBMRegressor(
                    objective="huber",
                    n_estimators=500,
                    max_depth=2,
                    min_data_in_leaf=100,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                )
                fold_lgb.fit(X_lgb_np[tr], signed_residual[tr])
                lgb_oof_pred[va] = fold_lgb.predict(X_lgb_np[va]).astype(np.float32)

            lgbm = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=500,
                max_depth=2,
                min_data_in_leaf=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            )
            lgbm.fit(X_lgb_np, signed_residual)
            lgb_pred = lgb_oof_pred
            _tprint(
                f"  LightGBM CV done: OOF residual correlation={_safe_spearman(lgb_oof_pred, signed_residual):.4f}"
            )

            train_leaf = np.asarray(
                lgbm.booster_.predict(X_lgb_np, pred_leaf=True), dtype=np.int64
            )
            if train_leaf.ndim == 1:
                train_leaf = train_leaf.reshape(-1, 1)
            self._leaf_var_maps = []
            self._leaf_cnt_maps = []
            for t in range(train_leaf.shape[1]):
                ids = train_leaf[:, t]
                vmap: dict[int, float] = {}
                cmap: dict[int, int] = {}
                uniq = np.unique(ids)
                for lid in uniq:
                    m = ids == lid
                    vals = signed_residual[m]
                    vmap[int(lid)] = float(np.var(vals)) if len(vals) > 0 else 0.0
                    cmap[int(lid)] = int(np.sum(m))
                self._leaf_var_maps.append(vmap)
                self._leaf_cnt_maps.append(cmap)
            train_unc = self._compute_reg_lgbm_uncertainty(X_lgb_np)
            leaf_var = train_unc["leaf_var"]
            leaf_cnt = train_unc["leaf_count"]
        else:
            lgbm = None
            lgb_pred = np.zeros(len(X_df), dtype=np.float32)
            leaf_var = np.full(len(X_df), np.nanvar(signed_residual), dtype=np.float32)
            leaf_cnt = np.ones(len(X_df), dtype=np.float32)
            self._leaf_var_maps = []
            self._leaf_cnt_maps = []

        leaf_ridge_bundle = _lgbm_leaf_ridge_fit(
            X_lgb,
            signed_residual,
            sample_weight=sw,
            random_state=47,
        )
        leaf_ridge_pred = np.asarray(
            leaf_ridge_bundle.get("oof_pred", np.zeros(len(X_df), dtype=np.float32)),
            dtype=np.float32,
        )
        leaf_ridge_score = float(leaf_ridge_bundle.get("score", 0.0))
        lgb_score = _safe_spearman(lgb_pred, signed_residual)

        ebm_res = train_ebm_on_lgbm_candidate(
            X_lgb,
            signed_residual,
            sample_weight=sw,
            random_state=42,
            mode="regressor",
        )
        ebm_score = 0.0
        ebm_oof = np.zeros(len(X_df), dtype=np.float32)
        if ebm_res and "oof_probs" in ebm_res:
            ebm_oof = np.asarray(ebm_res["oof_probs"], dtype=np.float32)
            ebm_score = _safe_spearman(ebm_oof, signed_residual)
            _tprint(f"  EBM CV done: OOF residual correlation={ebm_score:.4f}")

        scores = {
            "lgbm": lgb_score,
            "leaf_ridge": leaf_ridge_score,
            "ebm": ebm_score,
        }
        residual_model_name = max(scores, key=scores.get)
        ebm_residual_weighted_metrics: dict[str, Any] | None = None
        ebm_residual_uniform_metrics: dict[str, Any] | None = None

        if residual_model_name == "ebm" and isinstance(ebm_res, dict):
            residual_pred = ebm_oof
            ebm_features = ebm_res.get(
                "selected_features_from_cv", np.array([], dtype=np.int32)
            )

            def _residual_binary_metrics_for_ebm(pred: np.ndarray) -> dict[str, float]:
                y_pos = (signed_residual >= 0).astype(np.float64)
                p = np.clip(np.asarray(pred, dtype=np.float64), -20.0, 20.0)
                p_prob = 1.0 / (1.0 + np.exp(-p))
                return _ebm_metric_pack(
                    y_pos,
                    p_prob,
                    classifier=True,
                )

            self.ebm_model = fit_ebm_on_lgbm_full_model(
                X_lgb,
                signed_residual,
                sample_weight=sw,
                selected_features_from_cv=ebm_features,
                random_state=42,
                mode="regressor",
            )
            weighted_metrics = None
            if self.ebm_model is not None:
                try:
                    weighted_pred = np.asarray(
                        self.ebm_model.predict(X_lgb), dtype=np.float32
                    )
                    weighted_metrics = _residual_binary_metrics_for_ebm(weighted_pred)
                except Exception as exc:
                    _tprint(f"  EBM residual weighted metrics skipped ({exc})")
            uniform_metrics = None
            try:
                uniform_model = fit_ebm_on_lgbm_full_model(
                    X_lgb,
                    signed_residual,
                    sample_weight=np.ones(len(sw), dtype=np.float32),
                    selected_features_from_cv=ebm_features,
                    random_state=42,
                    mode="regressor",
                )
                if uniform_model is not None:
                    uniform_pred = np.asarray(
                        uniform_model.predict(X_lgb), dtype=np.float32
                    )
                    uniform_metrics = _residual_binary_metrics_for_ebm(uniform_pred)
            except Exception as exc:
                _tprint(f"  EBM residual uniform-weight fit skipped ({exc})")

            if (
                weighted_metrics is not None
                and uniform_metrics is not None
                and "ndcg_at_20" in weighted_metrics
                and "ndcg_at_20" in uniform_metrics
            ):
                _weighted_lift20 = float(weighted_metrics.get("lift20", np.nan))
                _uniform_lift20 = float(uniform_metrics.get("lift20", np.nan))
                _weighted_ndcg20 = float(
                    weighted_metrics.get(
                        "ndcg_at_20", weighted_metrics.get("ndcg@20", np.nan)
                    )
                )
                _uniform_ndcg20 = float(
                    uniform_metrics.get(
                        "ndcg_at_20", uniform_metrics.get("ndcg@20", np.nan)
                    )
                )
                _weighted_ece20 = float(
                    weighted_metrics.get(
                        "ece_top20", weighted_metrics.get("ece_at_20", np.nan)
                    )
                )
                _uniform_ece20 = float(
                    uniform_metrics.get(
                        "ece_top20", uniform_metrics.get("ece_at_20", np.nan)
                    )
                )
                _tprint(
                    "  EBM residual weighted-vs-uniform fit comparison: "
                    f"weighted(lift20={_weighted_lift20:.4f}, ndcg20={_weighted_ndcg20:.4f}, "
                    f"ece20={_weighted_ece20:.4f}) "
                    f"uniform(lift20={_uniform_lift20:.4f}, ndcg20={_uniform_ndcg20:.4f}, "
                    f"ece20={_uniform_ece20:.4f}) "
                    f"delta_lift20={(_uniform_lift20 - _weighted_lift20):.4f} "
                    f"delta_ndcg20={(_uniform_ndcg20 - _weighted_ndcg20):.4f} "
                    f"delta_ece20={(_uniform_ece20 - _weighted_ece20):.4f}"
                )
            if weighted_metrics is not None:
                ebm_residual_weighted_metrics = dict(weighted_metrics)
            if uniform_metrics is not None:
                ebm_residual_uniform_metrics = dict(uniform_metrics)
        elif residual_model_name == "ebm":
            residual_model_name = "lgbm"
            residual_pred = lgb_pred
            self.ebm_model = None
        elif residual_model_name == "leaf_ridge":
            residual_pred = leaf_ridge_pred
            self.ebm_model = None
        else:
            residual_pred = lgb_pred
            self.ebm_model = None

        self._reg_unc_a = 1.0
        self._reg_unc_C = float(np.percentile(leaf_cnt, 95))
        support_factor = np.log1p(leaf_cnt) / max(np.log1p(self._reg_unc_C), 1e-6)
        unc_raw = support_factor / (1.0 + self._reg_unc_a * leaf_var)
        self._reg_unc_lo = float(np.percentile(unc_raw, 5))
        self._reg_unc_hi = float(np.percentile(unc_raw, 95))
        if self._reg_unc_hi > self._reg_unc_lo:
            unc = 0.7 + 0.6 * np.clip(
                (unc_raw - self._reg_unc_lo) / (self._reg_unc_hi - self._reg_unc_lo),
                0.0,
                1.0,
            )
        else:
            unc = np.ones(len(unc_raw), dtype=np.float32)
        _tprint(f"  Uncertainty: lo={self._reg_unc_lo:.4f} hi={self._reg_unc_hi:.4f}")
        final = ridge_oof_pred + self.lambda_reg * residual_pred * unc

        self.selected_features = list(dict.fromkeys(ridge_feats + lgb_feats))
        self.ridge_model = ridge
        self.lgbm_model = lgbm
        self.lgbm_leaf_ridge_bundle = leaf_ridge_bundle
        self.model = ridge
        self.oof_probs = final.astype(np.float32)
        self._diag = {
            "ridge_features": list(ridge_feats),
            "lgbm_features": list(lgb_feats),
            "ridge_pred": ridge_oof_pred.astype(np.float32),
            "lgbm_pred": lgb_pred.astype(np.float32),
            "lgbm_leaf_ridge_pred": leaf_ridge_pred.astype(np.float32),
            "ebm_pred": ebm_oof.astype(np.float32),
            "residual_model_name": residual_model_name,
            "residual_pred": residual_pred.astype(np.float32),
            "meta_reg_leaf_var": leaf_var,
            "meta_reg_leaf_count": leaf_cnt,
            "meta_reg_support_factor": support_factor.astype(np.float32),
            "meta_reg_uncertainty": unc.astype(np.float32),
            "final": final.astype(np.float32),
            "ebm_residual_weighted_metrics": ebm_residual_weighted_metrics,
            "ebm_residual_uniform_metrics": ebm_residual_uniform_metrics,
            "unc_lo": float(self._reg_unc_lo),
            "unc_hi": float(self._reg_unc_hi),
            "unc_a": float(self._reg_unc_a),
            "leaf_count_cap_C": float(self._reg_unc_C),
            "leaf_var_maps": self._leaf_var_maps,
            "leaf_cnt_maps": self._leaf_cnt_maps,
            "lambda_reg": float(self.lambda_reg),
            "resid_target": signed_residual.astype(np.float32),
            "resid_target_uses_raw_base": kwargs.get("base_oof_pred") is not None,
            **resid_diag,
        }
        _tprint(f"WRMetaRegressor.fit: complete, final std={np.std(final):.4f}")
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        rf = [c for c in self.selected_features if c in X_df.columns]
        lf = [c for c in self._diag.get("lgbm_features", []) if c in X_df.columns]
        X_ridge = X_df.reindex(columns=rf, fill_value=0.0).to_numpy(dtype=np.float32)
        r = self.ridge_model.predict(X_ridge).astype(np.float32)
        X_lgb_np = X_df.reindex(columns=lf, fill_value=0.0).to_numpy(dtype=np.float32)
        l = (
            self.lgbm_model.predict(X_lgb_np).astype(np.float32)
            if self.lgbm_model is not None and len(lf) > 0
            else np.zeros(len(X_df), dtype=np.float32)
        )
        lr = (
            _lgbm_leaf_ridge_predict(
                self.lgbm_leaf_ridge_bundle,
                X_df.reindex(columns=lf, fill_value=0.0),
            )
            if self.lgbm_leaf_ridge_bundle is not None and len(lf) > 0
            else np.zeros(len(X_df), dtype=np.float32)
        )
        residual_model_name = self._diag.get("residual_model_name", "lgbm")
        if residual_model_name == "ebm" and self.ebm_model is not None:
            residual_pred = self.ebm_model.predict(X_df).astype(np.float32)
        elif residual_model_name == "leaf_ridge":
            residual_pred = lr
        else:
            residual_pred = l
        u_dict = self._compute_reg_lgbm_uncertainty(X_lgb_np)
        u = u_dict["uncertainty"]
        return (r + self.lambda_reg * residual_pred * u).astype(np.float32)

    def predict_uncertainty_features(self, X):
        n = len(X)
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        lf = [c for c in self._diag.get("lgbm_features", []) if c in X_df.columns]
        X_lgb_np = X_df.reindex(columns=lf, fill_value=0.0).to_numpy(dtype=np.float32)
        uu = self._compute_reg_lgbm_uncertainty(X_lgb_np)
        out = {}
        out["leaf_var"] = (
            uu["leaf_var"]
            if len(uu["leaf_var"]) == n
            else np.full(n, np.nan, dtype=np.float32)
        )
        out["leaf_count"] = (
            uu["leaf_count"]
            if len(uu["leaf_count"]) == n
            else np.full(n, np.nan, dtype=np.float32)
        )
        out["support_factor"] = (
            uu["support_factor"]
            if len(uu["support_factor"]) == n
            else np.full(n, np.nan, dtype=np.float32)
        )
        out["uncertainty"] = (
            uu["uncertainty"]
            if len(uu["uncertainty"]) == n
            else np.full(n, np.nan, dtype=np.float32)
        )
        out["prefix_std"] = np.full(
            n, np.nanstd(self._diag.get("final", np.zeros(n))), dtype=np.float32
        )
        out["leaf_support_q25"] = (
            uu["leaf_count_q25"]
            if len(uu.get("leaf_count_q25", [])) == n
            else np.full(n, np.nan, dtype=np.float32)
        )
        out["leaf_target_iqr_mean"] = out["leaf_var"]
        return out


class WeakResidualMetaClassifier:
    def __init__(
        self, strategy_name: Optional[str] = None, reports_dir: Optional[str] = None
    ):
        self.strategy_name = strategy_name
        self.reports_dir = reports_dir
        self.selected_features: list[str] = []
        self.ridge_model: Optional[Pipeline] = None
        self.lgbm_model: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.model: Optional[Any] = None
        self.oof_probs: Optional[np.ndarray] = None
        self._diag: Dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        n_feat = len(self.selected_features)
        has_lgbm = self.lgbm_model is not None
        has_cal = self.calibrator is not None
        return (
            f"WeakResidualMetaClassifier("
            f"strategy={self.strategy_name!r}, "
            f"features={n_feat}, "
            f"lgbm={has_lgbm}, "
            f"calibrator={has_cal})"
        )

    def clear_diagnostics(self) -> None:
        self._diag = {}

    def fit(self, X, y, sample_weight=None, groups=None, **kwargs):
        from extreme_price_movements.utils import tprint as _tprint

        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        base_oof_pred = kwargs.get("base_oof_pred")
        y_true_clf = kwargs.get("y_true_clf")
        base_threshold = float(kwargs.get("base_threshold", 0.5))
        if base_oof_pred is None:
            base_oof_pred = kwargs.get("y_move_override", y)
        if y_true_clf is None:
            y_true_clf = kwargs.get("y_class_override", y)

        base_prob = np.asarray(base_oof_pred, dtype=np.float32)
        y_true = (np.asarray(y_true_clf, dtype=np.float32) > 0.5).astype(np.int8)
        base_pred_class = (base_prob >= base_threshold).astype(np.int8)
        base_clf_correct = (base_pred_class == y_true).astype(np.int8)
        _tprint(
            f"WRMetaClassifier.fit: n={len(y_true)}, p={X_df.shape[1]}, base_acc={float(np.mean(base_clf_correct)):.3f}"
        )

        ridge_feats = _preridge_elasticnet_select(
            X_df, base_clf_correct.astype(np.float32), classifier=True, max_keep=120
        )
        X_ridge = X_df[ridge_feats]
        X_ridge_np = X_ridge.to_numpy(dtype=np.float32)
        sw = (
            np.ones(len(X_ridge_np), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )

        cv = StratifiedKFold(n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=42)
        ridge_oof_prob = np.zeros(len(X_ridge_np), dtype=np.float32)
        for tr, va in cv.split(X_ridge_np, base_clf_correct):
            fold = Pipeline(
                [
                    ("scaler", RobustScaler()),
                    ("ridge", RidgeClassifier(alpha=0.5, random_state=42)),
                ]
            )
            fold.fit(
                X_ridge_np[tr],
                base_clf_correct[tr],
                ridge__sample_weight=sw[tr],
            )
            score_va = fold.decision_function(X_ridge_np[va]).astype(np.float32)
            ridge_oof_prob[va] = 1.0 / (1.0 + np.exp(-score_va))

        ridge = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("ridge", RidgeClassifier(alpha=0.5, random_state=42)),
            ]
        )
        ridge.fit(X_ridge_np, base_clf_correct, ridge__sample_weight=sw)
        _tprint(
            f"  Ridge CV done: OOF IC={_safe_spearman(ridge_oof_prob, base_clf_correct.astype(np.float32)):.4f}"
        )

        clf_residual = base_clf_correct.astype(np.float32) - ridge_oof_prob
        y3 = np.ones(len(clf_residual), dtype=np.int32)
        for tr, va in cv.split(X_ridge_np, base_clf_correct):
            q1, q2 = np.quantile(clf_residual[tr], [1 / 3, 2 / 3])
            y3[va] = 1
            y3[va][clf_residual[va] < q1] = 0
            y3[va][clf_residual[va] >= q2] = 2
        _tprint(
            f"  y3 class distribution: {dict(zip(*np.unique(y3, return_counts=True)))}"
        )

        lgb_feats = _cluster_redundant_features(
            X_df[ridge_feats], clf_residual, thr=0.98
        )
        X_lgb0 = X_df[lgb_feats]
        lgb_feats = _univariate_screen(
            X_lgb0,
            base_clf_correct.astype(np.float32),
            ridge_oof_prob,
            clf_residual,
            top_k=150,
        )
        lgb_feats = _cluster_redundant_features(
            X_lgb0[lgb_feats], clf_residual, thr=0.95
        )
        X_lgb1 = X_lgb0[lgb_feats]
        lgb_feats = _iterative_fold_presence_prune(
            X_lgb1, y3, classifier=True, min_features=40
        )
        X_lgb = X_lgb1[lgb_feats]
        _tprint(
            f"  Feature selection: ridge={len(ridge_feats)} -> lgb={len(lgb_feats)}"
        )

        y3_classes, y3_counts = np.unique(y3, return_counts=True)
        can_fit_lgbm_clf = (
            lgb is not None
            and X_lgb.shape[1] > 0
            and len(y3_classes) == 3
            and int(np.min(y3_counts)) >= WR_META_CV_SPLITS
        )

        if can_fit_lgbm_clf:
            X_lgb_np = X_lgb.to_numpy(dtype=np.float32)
            p3 = np.zeros((len(X_lgb_np), 3), dtype=np.float32)
            cv_lgb = StratifiedKFold(
                n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=44
            )
            for tr, va in cv_lgb.split(X_lgb_np, y3):
                clf_fold = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=3,
                    n_estimators=500,
                    max_depth=2,
                    min_data_in_leaf=100,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                )
                clf_fold.fit(X_lgb_np[tr], y3[tr])
                p3[va] = clf_fold.predict_proba(X_lgb_np[va]).astype(np.float32)
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=500,
                max_depth=2,
                min_data_in_leaf=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            )
            clf.fit(X_lgb_np, y3)
            _tprint(f"  LightGBM 3-class CV done")
        else:
            clf = None
            p3 = np.full((len(X_lgb), 3), 1.0 / 3.0, dtype=np.float32)

        eps = 1e-9
        ent = -np.sum(p3 * np.log(np.clip(p3, eps, 1.0)), axis=1) / np.log(3.0)
        extreme = p3[:, 0] + p3[:, 2]
        top2 = np.partition(p3, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        unc_raw = 0.5 * (1.0 - ent) + 0.25 * extreme + 0.25 * margin
        unc = _norm_07_13(unc_raw)
        lambda_clf = float(kwargs.get("lambda_clf", 0.3))
        lgb_signed = p3[:, 2] - p3[:, 0]  # p_under - p_over
        final_raw = np.clip(
            ridge_oof_prob + lambda_clf * lgb_signed * unc, 1e-4, 1 - 1e-4
        )

        final_cal_oof = np.zeros(len(final_raw), dtype=np.float32)
        cal_cv = StratifiedKFold(
            n_splits=WR_META_CV_SPLITS, shuffle=True, random_state=43
        )
        for tr, va in cal_cv.split(final_raw.reshape(-1, 1), base_clf_correct):
            cal_fold = IsotonicRegression(out_of_bounds="clip")
            cal_fold.fit(final_raw[tr], base_clf_correct[tr])
            final_cal_oof[va] = np.clip(cal_fold.predict(final_raw[va]), 1e-4, 1 - 1e-4)
        final = np.clip(final_cal_oof, 1e-4, 1 - 1e-4)
        # Deployment calibrator is full-fit on all training rows.
        # _diag["final"] remains cross-fit calibrated OOF.
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(final_raw, base_clf_correct)

        self.selected_features = list(dict.fromkeys(ridge_feats + lgb_feats))
        self.ridge_model = ridge
        self.lgbm_model = clf
        self.calibrator = cal
        self.model = ridge
        self.oof_probs = final.astype(np.float32)
        self._diag = {
            "ridge_features": list(ridge_feats),
            "lgbm_features": list(lgb_feats),
            "base_clf_correct": base_clf_correct.astype(np.float32),
            "ridge_prob_correct": ridge_oof_prob.astype(np.float32),
            "meta_clf_lgbm_adj": lgb_signed.astype(np.float32),
            "lgbm_pred": lgb_signed.astype(np.float32),  # backward compat
            "meta_clf_entropy": ent.astype(np.float32),
            "meta_clf_extreme_mass": extreme.astype(np.float32),
            "meta_clf_margin": margin.astype(np.float32),
            "meta_clf_uncertainty": unc.astype(np.float32),
            "meta_clf_final_raw": final_raw.astype(np.float32),
            "final": final.astype(np.float32),
            "unc_lo": float(np.nanpercentile(unc_raw, 5)),
            "unc_hi": float(np.nanpercentile(unc_raw, 95)),
            "lambda_clf": lambda_clf,
        }
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        rf = [c for c in self._diag.get("ridge_features", []) if c in X_df.columns]
        lf = [c for c in self._diag.get("lgbm_features", []) if c in X_df.columns]
        X_ridge = X_df.reindex(columns=rf, fill_value=0.0).to_numpy(dtype=np.float32)
        score = self.ridge_model.decision_function(X_ridge).astype(np.float32)
        p_r = 1.0 / (1.0 + np.exp(-score))
        if self.lgbm_model is not None and len(lf) > 0:
            p3 = self.lgbm_model.predict_proba(
                X_df.reindex(columns=lf, fill_value=0.0)
            ).astype(np.float32)
            lgb_signed = p3[:, 2] - p3[:, 0]
            eps = 1e-9
            ent = -np.sum(p3 * np.log(np.clip(p3, eps, 1.0)), axis=1) / np.log(3.0)
            extreme = p3[:, 0] + p3[:, 2]
            top2 = np.partition(p3, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            unc_raw = 0.5 * (1.0 - ent) + 0.25 * extreme + 0.25 * margin
            lo = float(self._diag.get("unc_lo", np.nanpercentile(unc_raw, 5)))
            hi = float(self._diag.get("unc_hi", np.nanpercentile(unc_raw, 95)))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                unc = np.clip((unc_raw - lo) / (hi - lo), 0.0, 1.0)
                u = (0.7 + 0.6 * unc).astype(np.float32)
            else:
                u = np.ones(len(X_df), dtype=np.float32)
        else:
            lgb_signed = np.zeros(len(X_df), dtype=np.float32)
            u = np.ones(len(X_df), dtype=np.float32)
        lam = float(self._diag.get("lambda_clf", 0.3))
        f = np.clip(p_r + lam * lgb_signed * u, 1e-4, 1 - 1e-4)
        if self.calibrator is not None:
            f = np.clip(self.calibrator.predict(f), 1e-4, 1 - 1e-4)
        return np.column_stack([1.0 - f, f]).astype(np.float32)

    def predict(self, X):
        return self.predict_proba(X)[:, 1]

    def predict_uncertainty_features(self, X):
        n = len(X)
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        lf = [c for c in self._diag.get("lgbm_features", []) if c in X_df.columns]
        out = {}
        if self.lgbm_model is not None and len(lf) > 0:
            p3 = self.lgbm_model.predict_proba(
                X_df.reindex(columns=lf, fill_value=0.0)
            ).astype(np.float32)
            eps = 1e-9
            ent = -np.sum(p3 * np.log(np.clip(p3, eps, 1.0)), axis=1) / np.log(3.0)
            extreme = p3[:, 0] + p3[:, 2]
            top2 = np.partition(p3, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            unc_raw = 0.5 * (1.0 - ent) + 0.25 * extreme + 0.25 * margin
            lo = float(self._diag.get("unc_lo", np.nanpercentile(unc_raw, 5)))
            hi = float(self._diag.get("unc_hi", np.nanpercentile(unc_raw, 95)))
            unc = (
                (0.7 + 0.6 * np.clip((unc_raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0))
                if hi > lo
                else np.ones(n, dtype=np.float32)
            )
            out["entropy"] = ent.astype(np.float32)
            out["extreme_mass"] = extreme.astype(np.float32)
            out["margin"] = margin.astype(np.float32)
            out["uncertainty"] = np.asarray(unc, dtype=np.float32)
        else:
            out["entropy"] = np.full(n, np.nan, dtype=np.float32)
            out["extreme_mass"] = np.full(n, np.nan, dtype=np.float32)
            out["margin"] = np.full(n, np.nan, dtype=np.float32)
            out["uncertainty"] = np.ones(n, dtype=np.float32)
        out["prefix_std"] = np.full(
            n, np.nanstd(self._diag.get("final", np.zeros(n))), dtype=np.float32
        )
        out["leaf_support_q25"] = np.full(n, np.nan, dtype=np.float32)
        out["leaf_target_iqr_mean"] = np.full(n, np.nan, dtype=np.float32)
        return out


def save_weak_meta_outputs(
    *,
    out_dir: str,
    base_clf_pred: np.ndarray,
    base_reg_pred: np.ndarray,
    clf_model: WeakResidualMetaClassifier,
    reg_model: WeakResidualMetaRegressor,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    clf_final = np.asarray(clf_model._diag.get("final"), dtype=np.float32)
    reg_final = np.asarray(reg_model._diag.get("final"), dtype=np.float32)
    df = pd.DataFrame(
        {
            "base_clf_pred": np.asarray(base_clf_pred, dtype=np.float32),
            "base_reg_pred": np.asarray(base_reg_pred, dtype=np.float32),
            "meta_clf_ridge_pred": np.asarray(
                clf_model._diag.get("ridge_prob_correct"), dtype=np.float32
            ),
            "meta_clf_lgbm_pred": np.asarray(
                clf_model._diag.get("meta_clf_lgbm_adj"), dtype=np.float32
            ),
            "meta_clf_uncertainty": np.asarray(
                clf_model._diag.get("meta_clf_uncertainty"), dtype=np.float32
            ),
            "meta_clf_final_raw": np.asarray(
                clf_model._diag.get("meta_clf_final_raw"), dtype=np.float32
            ),
            "meta_clf_final": clf_final,
            "meta_reg_ridge_pred": np.asarray(
                reg_model._diag.get("ridge_pred"), dtype=np.float32
            ),
            "meta_reg_lgbm_pred": np.asarray(
                reg_model._diag.get("lgbm_pred"), dtype=np.float32
            ),
            "meta_reg_uncertainty": np.asarray(
                reg_model._diag.get("meta_reg_uncertainty"), dtype=np.float32
            ),
            "meta_reg_final": reg_final,
        }
    )
    df["score_base_x_meta_clf"] = _ranknorm(
        df["base_clf_pred"].values * df["meta_clf_final"].values
    )
    # Assumes meta_reg_final is a correction on the same return scale as base_reg_pred.
    df["score_base_plus_meta_reg"] = _ranknorm(
        df["base_reg_pred"].values + df["meta_reg_final"].values
    )
    df["score_combo_add"] = _ranknorm(
        0.5 * df["score_base_x_meta_clf"] + 0.5 * df["score_base_plus_meta_reg"]
    )
    df["score_combo_mult"] = _ranknorm(
        df["score_base_x_meta_clf"] * df["score_base_plus_meta_reg"]
    )
    df.to_parquet(
        os.path.join(out_dir, "weak_residual_meta_outputs.parquet"), index=False
    )

    diag = pd.DataFrame(
        {
            "meta_clf_entropy": clf_model._diag.get("meta_clf_entropy"),
            "meta_clf_extreme_mass": clf_model._diag.get("meta_clf_extreme_mass"),
            "meta_clf_margin": clf_model._diag.get("meta_clf_margin"),
            "meta_reg_leaf_var": reg_model._diag.get("meta_reg_leaf_var"),
            "meta_reg_leaf_count": reg_model._diag.get("meta_reg_leaf_count"),
            "meta_reg_support_factor": reg_model._diag.get("meta_reg_support_factor"),
        }
    )
    diag.to_parquet(
        os.path.join(out_dir, "weak_residual_meta_diagnostics.parquet"), index=False
    )
