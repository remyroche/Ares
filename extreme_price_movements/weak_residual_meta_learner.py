from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, Ridge, RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


def _ranknorm(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if len(v) == 0:
        return v.astype(np.float32)
    order = np.argsort(v, kind="mergesort")
    out = np.zeros(len(v), dtype=np.float64)
    out[order] = np.linspace(0.0, 1.0, len(v), endpoint=True)
    return out.astype(np.float32)


def _signed_rank_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float32)
    pos_mask = y > 0
    neg_mask = y < 0
    if np.any(pos_mask):
        import pandas as pd
        out[pos_mask] = pd.Series(y[pos_mask]).rank(pct=True).to_numpy()
    if np.any(neg_mask):
        import pandas as pd
        out[neg_mask] = -pd.Series(np.abs(y[neg_mask])).rank(pct=True).to_numpy()
    return out

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 8:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)


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
    grid = [(a, l) for a in (0.01, 0.05, 0.1, 0.5, 1.0) for l in (0.2, 0.5, 0.8)]
    cv = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if classifier
        else KFold(n_splits=5, shuffle=True, random_state=42)
    )
    y_split = yv.astype(np.int8) if classifier else yv
    rec = []
    coef_maps = []
    for alpha, l1 in grid:
        oof = np.zeros(len(yv), dtype=np.float32)
        fold_stats = []
        coef = np.zeros(Xv.shape[1], dtype=np.float64)
        for tr, va in cv.split(Xv, y_split):
            sc = RobustScaler()
            Xtr = sc.fit_transform(Xv[tr])
            Xva = sc.transform(Xv[va])
            if classifier:
                # Deliberate classifier-aligned sparse linear selector (L1 logistic),
                # used only as a feature ranking/filter stage before RidgeClassifier.
                c_val = 1.0 / max(alpha, 1e-6)
                mdl = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=c_val,
                    max_iter=2000,
                    random_state=42,
                )
                mdl.fit(Xtr, y_split[tr])
                oof[va] = mdl.predict_proba(Xva)[:, 1].astype(np.float32)
                coef += np.abs(np.ravel(mdl.coef_))
            else:
                mdl = ElasticNet(
                    alpha=alpha, l1_ratio=l1, max_iter=4000, random_state=42
                )
                mdl.fit(Xtr, yv[tr])
                oof[va] = mdl.predict(Xva).astype(np.float32)
                coef += np.abs(mdl.coef_)
            fold_stats.append(_metric_pack(yv[va], oof[va], classifier=classifier))
        coef_maps.append(coef / max(1, cv.get_n_splits()))
        metrics = {
            k: float(np.mean([r[k] for r in fold_stats])) for k in fold_stats[0].keys()
        }
        rec.append(metrics)
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
    X: pd.DataFrame, y: np.ndarray, thr: float = 0.97
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
    rel = np.array(
        [abs(_safe_spearman(X[c].values, yv)) for c in cols], dtype=np.float64
    )
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
    y_s = np.asarray(y, dtype=np.float64)[idx]
    r_s = np.asarray(ridge_pred, dtype=np.float64)[idx]
    sr_s = np.asarray(signed_residual, dtype=np.float64)[idx]
    b_s = buckets[idx]
    scores = []
    for c in cols:
        z = np.asarray(Xs[c].values, dtype=np.float64)
        c1 = []
        c2 = []
        for b in range(5):
            m = b_s == b
            if np.sum(m) < 8:
                continue
            c1.append(_safe_spearman(z[m], sr_s[m]))
            c2.append(_safe_spearman((z[m] * r_s[m]), y_s[m]))
        if not c1:
            scores.append(-1e9)
            continue
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        sc = abs(np.mean(c1)) - 0.5 * np.std(c1) + abs(np.mean(c2)) - 0.5 * np.std(c2)
        scores.append(float(sc))
    ord_idx = np.argsort(np.asarray(scores))[::-1][:top_k]
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
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
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
    for ratio in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        need = max(1, int(np.ceil(5 * ratio)))
        curr = {c for c in curr if presence.get(c, 0) >= need}
        if len(curr) <= min_features:
            break
    if len(curr) < min_features:
        rank = sorted(
            cols, key=lambda c: (presence.get(c, 0), gain.get(c, 0.0)), reverse=True
        )
        curr = set(rank[:min_features])
    return [c for c in cols if c in curr]


class WeakResidualMetaRegressor:
    def __init__(
        self, strategy_name: Optional[str] = None, reports_dir: Optional[str] = None
    ):
        self.strategy_name = strategy_name
        self.reports_dir = reports_dir
        self.selected_features: list[str] = []
        self.ridge_model: Optional[Pipeline] = None
        self.lgbm_model: Optional[Any] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.model: Optional[Any] = None
        self._diag: Dict[str, np.ndarray] = {}
        self._leaf_var_maps: list[dict[int, float]] = []
        self._leaf_cnt_maps: list[dict[int, int]] = []
        self._reg_unc_a: float = 1.0
        self._reg_unc_C: float = 1.0
        self._reg_unc_lo: float = 0.0
        self._reg_unc_hi: float = 1.0

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
            lv += np.array(
                [float(vmap.get(int(i), 0.0)) for i in ids], dtype=np.float64
            )
            lc += np.array([float(cmap.get(int(i), 1)) for i in ids], dtype=np.float64)
            lc_trees[:, t] = np.array(
                [float(cmap.get(int(i), 1)) for i in ids], dtype=np.float64
            )
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
        self, X, y, sample_weight=None, groups=None, y_per_horizon=None, y_binary=None
    ):
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_t = np.asarray(y, dtype=np.float32)

        ridge_feats = _preridge_elasticnet_select(
            X_df, y_t, classifier=False, max_keep=120
        )
        X_ridge = X_df[ridge_feats]
        ridge = Pipeline(
            [("scaler", RobustScaler()), ("ridge", Ridge(alpha=0.5, random_state=42))]
        )
        y_t_fit = (0.65 * np.arcsinh(y_t) + 0.35 * y_t).astype(np.float32)
        if sample_weight is None:
            ridge.fit(X_ridge, y_t_fit)
        else:
            ridge.fit(
                X_ridge,
                y_t_fit,
                ridge__sample_weight=np.asarray(sample_weight, dtype=np.float32),
            )
        ridge_pred = ridge.predict(X_ridge).astype(np.float32)

        ridge_target = y_t
        signed_residual = (ridge_target - ridge_pred).astype(np.float32)
        lgb_feats = _cluster_redundant_features(
            X_df[ridge_feats], signed_residual, thr=0.97
        )
        X_lgb0 = X_df[lgb_feats]
        lgb_feats = _univariate_screen(
            X_lgb0, ridge_target, ridge_pred, signed_residual, top_k=150
        )
        X_lgb1 = X_lgb0[lgb_feats]
        lgb_feats = _iterative_fold_presence_prune(
            X_lgb1, signed_residual, classifier=False, min_features=40
        )
        X_lgb = X_lgb1[lgb_feats]

        if lgb is not None and X_lgb.shape[1] > 0:
            lgbm = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=500,
                max_depth=2,
                min_data_in_leaf=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=2,
            )
            X_lgb_np = X_lgb.to_numpy(dtype=np.float32)
            lgbm.fit(X_lgb_np, signed_residual)
            lgb_pred = lgbm.predict(X_lgb_np).astype(np.float32)
            train_leaf = np.asarray(
                lgbm.booster_.predict(X_lgb_np, pred_leaf=True), dtype=np.int64
            )
            if train_leaf.ndim == 1:
                train_leaf = train_leaf.reshape(-1, 1)
            self._leaf_var_maps = []
            self._leaf_cnt_maps = []
            # Note: these maps are fit on the same rows used to train this v1 model.
            # Diagnostics on train rows are therefore optimistic by construction.
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

        en_res = _elasticnet_lgbm_pipeline(X_df, signed_residual, base_score=ridge_pred, classifier=False, random_state=42)
        en_ridge_pred = en_res['oof_predictions'].astype(np.float32)

        final = ridge_pred + 0.3 * lgb_pred * unc + 0.3 * en_ridge_pred * unc

        self.en_pipeline = en_res
        self.selected_features = list(dict.fromkeys(ridge_feats + lgb_feats))
        self.ridge_model = ridge
        self.lgbm_model = lgbm
        self.model = ridge
        self.oof_probs = final.astype(np.float32)
        self._diag = {
            "ridge_features": np.array(ridge_feats, dtype=object),
            "lgbm_features": np.array(lgb_feats, dtype=object),
            "ridge_pred": ridge_pred,
            "lgbm_pred": lgb_pred,
            "meta_reg_en_ridge_pred": en_ridge_pred,
            "meta_reg_leaf_var": leaf_var,
            "meta_reg_leaf_count": leaf_cnt,
            "meta_reg_support_factor": support_factor.astype(np.float32),
            "meta_reg_uncertainty": unc.astype(np.float32),
            "final": final.astype(np.float32),
            "unc_lo": np.float32(self._reg_unc_lo),
            "unc_hi": np.float32(self._reg_unc_hi),
            "unc_a": np.float32(self._reg_unc_a),
            "leaf_count_cap_C": np.float32(self._reg_unc_C),
            # Persist compact leaf maps in diagnostic payload for artifact tracing.
            "leaf_var_maps": np.array(self._leaf_var_maps, dtype=object),
            "leaf_cnt_maps": np.array(self._leaf_cnt_maps, dtype=object),
        }
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        rf = [c for c in self._diag.get("ridge_features", []) if c in X_df.columns]
        lf = [c for c in self._diag.get("lgbm_features", []) if c in X_df.columns]
        r = self.ridge_model.predict(X_df.reindex(columns=rf, fill_value=0.0)).astype(
            np.float32
        )
        X_lgb_np = X_df.reindex(columns=lf, fill_value=0.0).to_numpy(dtype=np.float32)
        l = (
            self.lgbm_model.predict(X_lgb_np).astype(np.float32)
            if self.lgbm_model is not None and len(lf) > 0
            else np.zeros(len(X_df), dtype=np.float32)
        )
        u_dict = self._compute_reg_lgbm_uncertainty(X_lgb_np)
        u = u_dict["uncertainty"]

        en_ridge_pred = 0.0
        if hasattr(self, 'en_pipeline') and self.en_pipeline is not None:
            en_ridge_pred = _en_pipeline_predict(X_df, self.en_pipeline, classifier=False)

        return (r + 0.3 * l * u + 0.3 * en_ridge_pred * u).astype(np.float32)

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

    def fit(self, X, y, sample_weight=None, groups=None, **kwargs):
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

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

        clf_residual = base_clf_correct.astype(np.float32) - ridge_oof_prob
        y3 = np.ones(len(clf_residual), dtype=np.int32)
        for tr, va in cv.split(X_ridge_np, base_clf_correct):
            q1, q2 = np.quantile(clf_residual[tr], [1 / 3, 2 / 3])
            y3[va] = 1
            y3[va][clf_residual[va] < q1] = 0
            y3[va][clf_residual[va] >= q2] = 2

        lgb_feats = _cluster_redundant_features(
            X_df[ridge_feats], clf_residual, thr=0.97
        )
        X_lgb0 = X_df[lgb_feats]
        lgb_feats = _univariate_screen(
            X_lgb0,
            base_clf_correct.astype(np.float32),
            ridge_oof_prob,
            clf_residual,
            top_k=150,
        )
        X_lgb1 = X_lgb0[lgb_feats]
        lgb_feats = _iterative_fold_presence_prune(
            X_lgb1, y3, classifier=True, min_features=40
        )
        X_lgb = X_lgb1[lgb_feats]

        if lgb is not None and X_lgb.shape[1] > 0:
            X_lgb_np = X_lgb.to_numpy(dtype=np.float32)
            p3 = np.zeros((len(X_lgb_np), 3), dtype=np.float32)
            cv_lgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
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
            )
            clf.fit(X_lgb_np, y3)
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

        en_res = _elasticnet_lgbm_pipeline(X_df, y3, base_score=ridge_oof_prob, classifier=True, random_state=42)
        en_ridge_oof_prob = en_res['oof_predictions'].astype(np.float32)

        final_raw = np.clip(
            ridge_oof_prob + lambda_clf * lgb_signed * unc + 0.3 * en_ridge_oof_prob, 1e-4, 1 - 1e-4
        )

        final_cal_oof = np.zeros(len(final_raw), dtype=np.float32)
        cal_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
        for tr, va in cal_cv.split(final_raw.reshape(-1, 1), base_clf_correct):
            cal_fold = IsotonicRegression(out_of_bounds="clip")
            cal_fold.fit(final_raw[tr], base_clf_correct[tr])
            final_cal_oof[va] = np.clip(cal_fold.predict(final_raw[va]), 1e-4, 1 - 1e-4)
        final = np.clip(final_cal_oof, 1e-4, 1 - 1e-4)
        # Deployment calibrator is full-fit on all training rows.
        # _diag["final"] remains cross-fit calibrated OOF.
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(final_raw, base_clf_correct)

        self.en_pipeline = en_res
        self.selected_features = list(dict.fromkeys(ridge_feats + lgb_feats))
        self.ridge_model = ridge
        self.lgbm_model = clf
        self.calibrator = cal
        self.model = ridge
        self.oof_probs = final.astype(np.float32)
        self._diag = {
            "ridge_features": np.array(ridge_feats, dtype=object),
            "lgbm_features": np.array(lgb_feats, dtype=object),
            "base_clf_correct": base_clf_correct.astype(np.float32),
            "ridge_prob_correct": ridge_oof_prob.astype(np.float32),
            "meta_clf_lgbm_adj": lgb_signed.astype(np.float32),
            "lgbm_pred": lgb_signed.astype(np.float32),  # backward compat
            "meta_clf_en_ridge_pred": en_ridge_oof_prob.astype(np.float32),
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
        score = self.ridge_model.decision_function(
            X_df.reindex(columns=rf, fill_value=0.0)
        ).astype(np.float32)
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

        en_ridge_pred = 0.5
        if hasattr(self, 'en_pipeline') and self.en_pipeline is not None:
            en_ridge_pred = _en_pipeline_predict(X_df, self.en_pipeline, classifier=True)

        f = np.clip(p_r + lam * lgb_signed * u + 0.3 * en_ridge_pred, 1e-4, 1 - 1e-4)
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
            "meta_clf_en_ridge_pred": np.asarray(
                clf_model._diag.get("meta_clf_en_ridge_pred"), dtype=np.float32
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
            "meta_reg_en_ridge_pred": np.asarray(
                reg_model._diag.get("meta_reg_en_ridge_pred"), dtype=np.float32
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
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score
import warnings

try:
    import lightgbm as lgb
except Exception:
    lgb = None

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.sum(a_b & b_b)
    union = np.sum(a_b | b_b)
    if union == 0:
        return 0.0
    return float(inter / union)

def _signed_rank_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float32)
    pos_mask = y > 0
    neg_mask = y < 0
    if np.any(pos_mask):
        import pandas as pd
        out[pos_mask] = pd.Series(y[pos_mask]).rank(pct=True).to_numpy()
    if np.any(neg_mask):
        import pandas as pd
        out[neg_mask] = -pd.Series(np.abs(y[neg_mask])).rank(pct=True).to_numpy()
    return out

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 8:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)

def _safe_pointbiserial(x: np.ndarray, y: np.ndarray) -> float:
    # y is binary (0/1)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 8:
        return 0.0
    r = pointbiserialr(y[m], x[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)

def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10 or len(np.unique(y[m])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[m], x[m]))
    except:
        return 0.5

def _get_top_k_mask(pred: np.ndarray, pct: float) -> np.ndarray:
    n = len(pred)
    k = max(1, int(np.ceil(pct * n)))
    idx = np.argsort(pred)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def _fast_corr_matrix(Z, is_binary):
    n_features = Z.shape[1]
    corr = np.zeros((n_features, n_features), dtype=np.float32)
    Z_rank = pd.DataFrame(Z).rank(pct=True).to_numpy()

    dense_mask = ~np.array(is_binary)
    if np.any(dense_mask):
        Z_dense = Z_rank[:, dense_mask]
        corr_dense = np.abs(np.corrcoef(Z_dense.T))
    else:
        corr_dense = None

    bin_idx = np.where(is_binary)[0]
    dense_idx = np.where(~np.array(is_binary))[0]

    if np.any(dense_mask) and corr_dense is not None:
        for i, d1 in enumerate(dense_idx):
            for j, d2 in enumerate(dense_idx):
                corr[d1, d2] = corr_dense[i, j]

    for i, b1 in enumerate(bin_idx):
        for j, b2 in enumerate(bin_idx):
            if i == j:
                corr[b1, b2] = 1.0
            elif i < j:
                val = _jaccard(Z[:, b1], Z[:, b2])
                corr[b1, b2] = val
                corr[b2, b1] = val

    for d in dense_idx:
        for b in bin_idx:
            val = abs(np.corrcoef(Z_rank[:, d], Z[:, b])[0, 1])
            if not np.isfinite(val):
                val = 0.0
            corr[d, b] = val
            corr[b, d] = val

    return corr

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pointbiserialr
import warnings

try:
    import lightgbm as lgb
except Exception:
    lgb = None

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.sum(a_b & b_b)
    union = np.sum(a_b | b_b)
    if union == 0:
        return 0.0
    return float(inter / union)

def _signed_rank_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float32)
    pos_mask = y > 0
    neg_mask = y < 0
    if np.any(pos_mask):
        import pandas as pd
        out[pos_mask] = pd.Series(y[pos_mask]).rank(pct=True).to_numpy()
    if np.any(neg_mask):
        import pandas as pd
        out[neg_mask] = -pd.Series(np.abs(y[neg_mask])).rank(pct=True).to_numpy()
    return out

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 8:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)

def _safe_pointbiserial(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 8:
        return 0.0
    r = pointbiserialr(y[m], x[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)

def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10 or len(np.unique(y[m])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[m], x[m]))
    except:
        return 0.5

def _get_top_k_mask(pred: np.ndarray, pct: float) -> np.ndarray:
    n = len(pred)
    k = max(1, int(np.ceil(pct * n)))
    idx = np.argsort(pred)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def _fast_corr_matrix(Z, is_binary):
    n_features = Z.shape[1]
    corr = np.zeros((n_features, n_features), dtype=np.float32)
    Z_rank = pd.DataFrame(Z).rank(pct=True).to_numpy()

    dense_mask = ~np.array(is_binary)
    if np.any(dense_mask):
        Z_dense = Z_rank[:, dense_mask]
        corr_dense = np.abs(np.corrcoef(Z_dense.T))
    else:
        corr_dense = None

    bin_idx = np.where(is_binary)[0]
    dense_idx = np.where(~np.array(is_binary))[0]

    if np.any(dense_mask) and corr_dense is not None:
        for i, d1 in enumerate(dense_idx):
            for j, d2 in enumerate(dense_idx):
                corr[d1, d2] = corr_dense[i, j]

    for i, b1 in enumerate(bin_idx):
        for j, b2 in enumerate(bin_idx):
            if i == j:
                corr[b1, b2] = 1.0
            elif i < j:
                val = _jaccard(Z[:, b1], Z[:, b2])
                corr[b1, b2] = val
                corr[b2, b1] = val

    for d in dense_idx:
        for b in bin_idx:
            val = abs(np.corrcoef(Z_rank[:, d], Z[:, b])[0, 1])
            if not np.isfinite(val):
                val = 0.0
            corr[d, b] = val
            corr[b, d] = val

    return corr

def _get_lgbm_base_params(d, leaf_pct, n_samples, classifier, random_state, is_linear=False):
    min_data = max(10, int(n_samples * leaf_pct))
    params = {
        'n_estimators': 400,
        'learning_rate': 0.05,
        'max_depth': d,
        'num_leaves': 2**d if d else 31,
        'min_data_in_leaf': min_data,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l2': 5.0,
        'min_gain_to_split': 0.001,
        'max_bin': 127,
        'random_state': random_state,
        'n_jobs': 2,
    }
    if is_linear:
        params['linear_tree'] = True
    if classifier:
        params['lambda_l1'] = 0.0
        params['min_sum_hessian_in_leaf'] = 1e-3
    return params

def _train_lgbm_models_and_extract(X_train, y_train, X_val, n_samples, classifier, random_state, return_models=False):
    # Ensure y_train is already transformed correctly if regression, before being passed to this function.

    leaf_matrices = []
    lgb_models = []
    lin_raw_features = []
    lin_models = []

    def fit_lgbm(params, is_linear=False):
        if classifier:
            _classes = np.unique(y_train)
            if len(_classes) > 2:
                model = lgb.LGBMClassifier(objective='multiclass', num_class=len(_classes), **params)
            else:
                model = lgb.LGBMClassifier(objective='binary', **params)
        else:
            model = lgb.LGBMRegressor(objective='huber', alpha=0.9, **params)

        early_stop = 15 if is_linear else 25
        # internal eval for early stopping
        rng = np.random.default_rng(random_state)
        eval_idx = rng.choice(len(X_train), int(0.2*len(X_train)), replace=False)
        tr_idx = np.setdiff1d(np.arange(len(X_train)), eval_idx)

        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[eval_idx], y_train[eval_idx])],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]
        )
        return model

    # Depth 3 models
    for pct in [0.02, 0.04, 0.06]:
        params = _get_lgbm_base_params(3, pct, n_samples, classifier, random_state)
        model = fit_lgbm(params)
        lgb_models.append(model)

    # Depth 4 models
    for pct in [0.04, 0.06, 0.08]:
        params = _get_lgbm_base_params(4, pct, n_samples, classifier, random_state)
        params['num_leaves'] = 16
        model = fit_lgbm(params)
        lgb_models.append(model)

    # Extract leaves on val
    for model in lgb_models:
        leaves = model.booster_.predict(X_val, pred_leaf=True)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        leaf_matrices.append(leaves)

    # Linear Tree LGBM models
    for pct in [0.05, 0.10, 0.15]:
        params = _get_lgbm_base_params(3, pct, n_samples, classifier, random_state, is_linear=True)
        model = fit_lgbm(params, is_linear=True)
        lin_models.append(model)

        # model level raw score on val
        raw_score = model.predict(X_val, raw_score=True)
        if raw_score.ndim > 1 and raw_score.shape[1] == 1:
            raw_score = raw_score.ravel()
        elif raw_score.ndim > 1: # multiclass
            for c in range(raw_score.shape[1]):
                lin_raw_features.append(raw_score[:, c].astype(np.float32))
            continue

        lin_raw_features.append(raw_score.astype(np.float32))

        # per-tree raw scores
        num_trees = model.booster_.num_trees()
        prev = np.zeros(len(X_val))
        for k in range(1, num_trees + 1):
            cum = model.booster_.predict(X_val, raw_score=True, num_iteration=k)
            if cum.ndim > 1 and cum.shape[1] > 1:
                continue
            if cum.ndim > 1:
                cum = cum.ravel()
            tree_k_score = cum - prev
            prev = cum
            lin_raw_features.append(tree_k_score.astype(np.float32))

    if return_models:
        return np.hstack(leaf_matrices) if leaf_matrices else np.empty((len(X_val), 0)), \
               np.column_stack(lin_raw_features) if lin_raw_features else np.empty((len(X_val), 0)), \
               lgb_models, lin_models

    return np.hstack(leaf_matrices) if leaf_matrices else np.empty((len(X_val), 0)), \
           np.column_stack(lin_raw_features) if lin_raw_features else np.empty((len(X_val), 0))

def _elasticnet_lgbm_pipeline(X: pd.DataFrame, y: np.ndarray, base_score: np.ndarray = None, classifier: bool = False, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    X_np = X.to_numpy(dtype=np.float32)
    y_target = y.astype(np.int32) if classifier else y.astype(np.float32)

    if not classifier:
        y_fit = (0.65 * np.arcsinh(y_target) + 0.35 * y_target).astype(np.float32)
    else:
        y_fit = y_target.copy()
    if base_score is None:
        base_score = np.abs(y_target)

    raw_feature_names = list(X.columns)

    # 1. Generate Cross-Fitted (OOF) LGBM Features to prevent leakage during EN Selection
    cv_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if classifier else KFold(n_splits=5, shuffle=True, random_state=random_state)

    oof_all_leaves = None
    oof_lin_features = None

    for tr, va in cv_gen.split(X_np, y_target):
        val_leaves, val_lin = _train_lgbm_models_and_extract(
            X_np[tr], y_fit[tr], X_np[va], n_samples, classifier, random_state
        )
        if oof_all_leaves is None:
            oof_all_leaves = np.zeros((n_samples, val_leaves.shape[1]), dtype=np.float32)
            oof_lin_features = np.zeros((n_samples, val_lin.shape[1]), dtype=np.float32)
        oof_all_leaves[va] = val_leaves
        oof_lin_features[va] = val_lin

    # To build OHE features safely across folds, we find globals or just use the full-fit later for the final pipeline.
    # For selection, we use OOF.
    unique_leaf_vals = []
    ohe_leaves_oof = []

    for c in range(oof_all_leaves.shape[1]):
        col = oof_all_leaves[:, c]
        uniques = np.unique(col)
        unique_leaf_vals.append(uniques)
        for val in uniques:
            ohe_leaves_oof.append((col == val).astype(np.float32))

    L_features_oof = np.column_stack(ohe_leaves_oof) if ohe_leaves_oof else np.empty((n_samples, 0), dtype=np.float32)

    # Scale dense (Raw + Lin_features_oof)
    scaler = RobustScaler()
    Dense_oof = np.hstack([X_np, oof_lin_features])
    Dense_scaled_oof = scaler.fit_transform(Dense_oof)

    Z_oof = np.hstack([Dense_scaled_oof, L_features_oof])
    is_binary = [False] * Dense_scaled_oof.shape[1] + [True] * L_features_oof.shape[1]

    # 2. Fast Pruning (using Z_oof)
    sub_5k_idx = rng.choice(n_samples, min(5000, n_samples), replace=False)
    Z_5k = Z_oof[sub_5k_idx]
    y_5k = y_target[sub_5k_idx]

    ic_scores = []
    for j in range(Z_oof.shape[1]):
        if classifier:
            if is_binary[j]:
                val = max(abs(_safe_pointbiserial(Z_5k[:, j], y_5k)), abs(_safe_auc(Z_5k[:, j], y_5k) - 0.5) * 2)
            else:
                val = abs(_safe_auc(Z_5k[:, j], y_5k) - 0.5) * 2
        else:
            val = abs(_safe_spearman(Z_5k[:, j], y_5k))
        ic_scores.append(val)
    ic_scores = np.array(ic_scores)

    keep_count_1 = int(200 + 0.33 * Z_oof.shape[1])
    keep_count_1 = min(keep_count_1, Z_oof.shape[1])
    idx_top_ic = np.argsort(ic_scores)[-keep_count_1:]

    sub_3k_idx = rng.choice(n_samples, min(3000, n_samples), replace=False)
    Z_3k = Z_oof[sub_3k_idx][:, idx_top_ic]
    ic_scores_subset = ic_scores[idx_top_ic]

    order = np.argsort(ic_scores_subset)[::-1]
    keep_idx_2_rel = []
    dropped_2 = np.zeros(len(order), dtype=bool)

    is_binary_subset = [is_binary[i] for i in idx_top_ic]
    corr_3k = _fast_corr_matrix(Z_3k, is_binary_subset)

    for i in range(len(order)):
        curr_i = order[i]
        if dropped_2[curr_i]:
            continue
        keep_idx_2_rel.append(curr_i)
        dropped_2 |= (corr_3k[curr_i] > 0.98)
        dropped_2[curr_i] = False

    idx_stage_2 = idx_top_ic[keep_idx_2_rel]

    stability_scores = []
    for j in range(len(idx_stage_2)):
        orig_j = idx_stage_2[j]
        ic_vals = []
        for _ in range(5):
            sub_idx = rng.choice(n_samples, min(3000, n_samples), replace=False)
            if classifier:
                val = _safe_pointbiserial(Z_oof[sub_idx, orig_j], y_target[sub_idx])
            else:
                val = _safe_spearman(Z_oof[sub_idx, orig_j], y_target[sub_idx])
            ic_vals.append(val)
        ic_vals = np.array(ic_vals)
        pos_frac = np.mean(ic_vals > 0)
        neg_frac = np.mean(ic_vals < 0)
        sign_consistency = max(pos_frac, neg_frac)
        stability = np.median(np.abs(ic_vals)) * sign_consistency
        stability_scores.append(stability)

    stability_scores = np.array(stability_scores)
    keep_count_3 = int(100 + 0.25 * len(idx_stage_2))
    keep_count_3 = min(keep_count_3, len(idx_stage_2))
    idx_top_stab_rel = np.argsort(stability_scores)[-keep_count_3:]
    idx_stage_3 = idx_stage_2[idx_top_stab_rel]

    sub_5k_idx_2 = rng.choice(n_samples, min(5000, n_samples), replace=False)
    Z_5k_2 = Z_oof[sub_5k_idx_2][:, idx_stage_3]
    stab_subset = stability_scores[idx_top_stab_rel]

    order_3 = np.argsort(stab_subset)[::-1]
    keep_idx_4_rel = []
    dropped_4 = np.zeros(len(order_3), dtype=bool)

    is_binary_subset_2 = [is_binary[i] for i in idx_stage_3]
    corr_5k = _fast_corr_matrix(Z_5k_2, is_binary_subset_2)

    for i in range(len(order_3)):
        curr_i = order_3[i]
        if dropped_4[curr_i]:
            continue
        keep_idx_4_rel.append(curr_i)
        dropped_4 |= (corr_5k[curr_i] > 0.96)
        dropped_4[curr_i] = False

    active_features = list(idx_stage_3[keep_idx_4_rel])

    # 3. Iterative ElasticNet Pruning
    sub_20k_idx = rng.choice(n_samples, min(20000, n_samples), replace=False)
    Z_20k = Z_oof[sub_20k_idx]
    y_20k = y_target[sub_20k_idx]
    bs_20k = base_score[sub_20k_idx] if base_score is not None else np.abs(y_20k)

    alpha_grid = [0.05, 0.1, 0.3, 0.5]
    l1_ratio_grid = [0.4, 0.6, 0.8]

    best_historical_J = -np.inf
    iteration = 1
    models_history = []
    prev_preds = bs_20k.copy()

    while True:
        folds = int(2 + 0.5 * iteration)
        threshold = 0.10 * iteration
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state) if classifier else KFold(n_splits=folds, shuffle=True, random_state=random_state)

        feature_metrics = {f_idx: {'coeffs': [], 'presence': 0, 'top30_contrib': []} for f_idx in active_features}
        total_fits = folds * len(alpha_grid) * len(l1_ratio_grid)

        iter_models = []
        Z_iter = Z_20k[:, active_features]

        # Base top30 weights using prev_preds
        top30_mask = _get_top_k_mask(prev_preds, 0.30)
        sample_weights = np.ones(len(y_20k), dtype=np.float32)
        top30_weight = min(2.5, 1.0 + 0.25 * iteration)
        sample_weights[top30_mask] = top30_weight

        for alpha in alpha_grid:
            for l1 in l1_ratio_grid:
                oof_preds = np.zeros(len(y_20k))
                for tr, va in cv.split(Z_iter, y_20k):
                    if classifier:
                        C_val = 1.0 / max(alpha, 1e-6)
                        mdl = LogisticRegression(penalty='l1', solver='liblinear', C=C_val, random_state=random_state, max_iter=1000)
                    else:
                        mdl = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=random_state, max_iter=2000)

                    y_fit_20k = y_fit[sub_20k_idx]

                    rank_focus_tr = 0.7 + 0.6 * np.sqrt(np.clip((pd.Series(prev_preds[tr]).rank(pct=True).to_numpy()), 0, 1))
                    if classifier:
                        y_bin_tr = (y_20k[tr] > 0.5).astype(int)
                        confidence_tr = np.clip(prev_preds[tr], 0, 1)
                        pred_class_tr = (prev_preds[tr] > 0.5).astype(int)
                        match_tr = (y_bin_tr == pred_class_tr)
                        error_weight_tr = np.where(match_tr,
                            np.minimum(1.6, 1.0 + 0.2 * confidence_tr),
                            np.minimum(1.6, 1.0 + 0.5 * confidence_tr))
                    else:
                        resid_tr = np.abs(y_20k[tr] - prev_preds[tr])
                        resid_rank_tr = pd.Series(resid_tr).rank(pct=True).to_numpy()
                        error_weight_tr = np.minimum(1.6, 1.0 + 0.3 * resid_rank_tr)

                    weight_distillation_tr = sample_weights[tr] * error_weight_tr * np.sqrt(rank_focus_tr)

                    mdl.fit(Z_iter[tr], y_fit_20k[tr], sample_weight=weight_distillation_tr)

                    if classifier:
                        if hasattr(mdl, "decision_function"):
                            score = mdl.decision_function(Z_iter[va])
                            if score.ndim > 1:
                                # Multiclass LogisticRegression
                                exps = np.exp(score - np.max(score, axis=1, keepdims=True))
                                p = exps / np.sum(exps, axis=1, keepdims=True)
                                pred_va = p[:, 2] - p[:, 0]
                            else:
                                pred_va = 1.0 / (1.0 + np.exp(-score))
                        else:
                            pred_va = mdl.predict_proba(Z_iter[va])[:, 1]
                    else:
                        pred_va = mdl.predict(Z_iter[va])

                    oof_preds[va] = pred_va

                    if classifier and len(mdl.classes_) > 2:
                        coefs = np.mean(np.abs(mdl.coef_), axis=0) # avg magnitude across classes
                    else:
                        coefs = np.ravel(mdl.coef_)

                    va_top30 = _get_top_k_mask(bs_20k[va], 0.30)
                    for i_act, f_idx in enumerate(active_features):
                        val = coefs[i_act]
                        if abs(val) > 1e-6:
                            feature_metrics[f_idx]['presence'] += 1
                            feature_metrics[f_idx]['coeffs'].append(val)
                            contrib = Z_iter[va][va_top30, i_act] * val
                            feature_metrics[f_idx]['top30_contrib'].append(np.mean(contrib) if len(contrib) > 0 else 0)

                m30_mask = _get_top_k_mask(oof_preds, 0.30)
                if classifier:
                    y_bin = (y_20k > 0.5).astype(float)
                    s_top = oof_preds[m30_mask]
                    y_top = y_bin[m30_mask]
                    hr30 = np.mean(y_top) if np.any(m30_mask) else 0.0
                    if len(s_top) >= 10:
                        q = np.quantile(s_top, np.linspace(0.0, 1.0, 6))
                        vals = [float(np.mean(y_top[(s_top >= q[i]) & (s_top <= q[i + 1])])) for i in range(5) if np.any((s_top >= q[i]) & (s_top <= q[i + 1]))]
                        std30 = np.std(vals) if vals else 0.0
                    else:
                        std30 = 0.0
                    J = hr30 - 0.5 * std30
                else:
                    ic30 = _safe_spearman(oof_preds[m30_mask], y_20k[m30_mask]) if np.any(m30_mask) else 0.0
                    s_top = oof_preds[m30_mask]
                    y_top = y_20k[m30_mask]
                    if len(s_top) >= 10:
                        q = np.quantile(s_top, np.linspace(0.0, 1.0, 6))
                        vals = [float(np.mean(y_top[(s_top >= q[i]) & (s_top <= q[i + 1])])) for i in range(5) if np.any((s_top >= q[i]) & (s_top <= q[i + 1]))]
                        std30 = np.std(vals) if vals else 0.0
                    else:
                        std30 = 0.0
                    J = ic30 - 0.5 * std30

                iter_models.append({'alpha': alpha, 'l1_ratio': l1, 'J': J, 'features': active_features.copy(), 'oof_preds': oof_preds})

        iter_best_J = max(m['J'] for m in iter_models)
        best_iter_model = max(iter_models, key=lambda m: m['J'])
        prev_preds = best_iter_model['oof_preds']
        models_history.extend(iter_models)

        SE = 0.02
        if iter_best_J < best_historical_J - SE:
            break
        if iter_best_J > best_historical_J:
            best_historical_J = iter_best_J

        new_active = []
        for f_idx in active_features:
            mets = feature_metrics[f_idx]
            presence_pct = mets['presence'] / total_fits
            if presence_pct == 0:
                continue
            coeffs = np.array(mets['coeffs'])
            pos_frac = np.mean(coeffs > 0)
            neg_frac = np.mean(coeffs < 0)
            sign_consistency = max(pos_frac, neg_frac)
            median_mag = np.median(np.abs(coeffs))

            top30 = np.array(mets['top30_contrib'])
            top30_mean = np.mean(top30) if len(top30)>0 else 0
            top30_std = np.std(top30) if len(top30)>0 else 1
            top30_stability = top30_mean / max(top30_std, 1e-6)

            score = (sign_consistency ** 1.5) * median_mag * abs(top30_stability)
            new_active.append((f_idx, score, presence_pct))

        if not new_active:
            break

        scores = np.array([x[1] for x in new_active])
        if len(scores) > 1 and np.max(scores) > np.min(scores):
            scores_scaled = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            scores_scaled = np.ones(len(scores))

        final_active = []
        for i, (f_idx, score, pres) in enumerate(new_active):
            val = scores_scaled[i] * pres
            if val >= threshold:
                final_active.append(f_idx)

        if len(final_active) == len(active_features) or len(final_active) < 5:
            break

        active_features = final_active
        iteration += 1

    # 4. Final Full-fit
    models_history.sort(key=lambda x: x['J'], reverse=True)
    best_pareto = max(models_history[:5], key=lambda m: m['J'] - 0.001 * len(m['features']))
    final_features = best_pareto['features']

    # Train full-fit LGBM generators
    full_all_leaves, full_lin_features, lgb_models, lin_models = _train_lgbm_models_and_extract(
        X_np, y_fit, X_np, n_samples, classifier, random_state, return_models=True
    )

    ohe_leaves_full = []
    for c in range(full_all_leaves.shape[1]):
        col = full_all_leaves[:, c]
        uniques = unique_leaf_vals[c]
        for val in uniques:
            ohe_leaves_full.append((col == val).astype(np.float32))
    L_features_full = np.column_stack(ohe_leaves_full) if ohe_leaves_full else np.empty((n_samples, 0), dtype=np.float32)

    final_scaler = RobustScaler()
    Dense_full = np.hstack([X_np, full_lin_features])
    Dense_scaled_full = final_scaler.fit_transform(Dense_full)
    Z_full = np.hstack([Dense_scaled_full, L_features_full])

    Z_final = Z_full[:, final_features]

    # Train final Ridge/Logistic
    top30_mask = _get_top_k_mask(base_score, 0.30)
    sample_weights = np.ones(n_samples, dtype=np.float32)
    sample_weights[top30_mask] = 1.0 + 0.5 * iteration

    if classifier:
        final_model = RidgeClassifier(alpha=1.0, random_state=random_state)
    else:
        final_model = Ridge(alpha=1.0, random_state=random_state)

    # We must generate cross-validated OOF predictions for the final model so it's safe to use for ensembling
    cv_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if classifier else KFold(n_splits=5, shuffle=True, random_state=random_state)
    final_oof_preds = np.zeros(n_samples, dtype=np.float32)

    for tr, va in cv_final.split(Z_final, y_target):
        if classifier:
            m = LogisticRegression(penalty='l2', solver='liblinear', random_state=random_state)
        else:
            m = Ridge(alpha=1.0, random_state=random_state)
        m.fit(Z_final[tr], y_target[tr], sample_weight=sample_weights[tr])
        if classifier:
            if hasattr(m, "decision_function"):
                score = m.decision_function(Z_final[va])
                if score.ndim > 1:
                    exps = np.exp(score - np.max(score, axis=1, keepdims=True))
                    p = exps / np.sum(exps, axis=1, keepdims=True)
                    pred_va = p[:, 2] - p[:, 0]
                else:
                    pred_va = 1.0 / (1.0 + np.exp(-score))
            else:
                pred_va = m.predict_proba(Z_final[va])[:, 1]
        else:
            pred_va = m.predict(Z_final[va])
        final_oof_preds[va] = pred_va

    # Final fit on all data
    final_model.fit(Z_final, y_target, sample_weight=sample_weights)

    return {
        'model': final_model,
        'features': final_features,
        'oof_predictions': final_oof_preds,
        'scaler': final_scaler,
        'lgb_models': lgb_models,
        'lin_models': lin_models,
        'unique_leaf_vals': unique_leaf_vals,
        'raw_feature_names': raw_feature_names,
        'n_raw_features': X.shape[1]
    }

def _en_pipeline_predict(X: pd.DataFrame, en_res: dict, classifier: bool = False):
    X_reindexed = X.reindex(columns=en_res['raw_feature_names'], fill_value=0.0)
    X_np = X_reindexed.to_numpy(dtype=np.float32)
    n_samples = len(X)

    leaf_matrices = []
    for model in en_res['lgb_models']:
        leaves = model.booster_.predict(X_np, pred_leaf=True)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        leaf_matrices.append(leaves)

    all_leaves = np.hstack(leaf_matrices) if leaf_matrices else np.empty((n_samples, 0))

    ohe_leaves = []
    for c in range(all_leaves.shape[1]):
        col = all_leaves[:, c]
        uniques = en_res['unique_leaf_vals'][c]
        for val in uniques:
            ohe_leaves.append((col == val).astype(np.float32))

    L_features = np.column_stack(ohe_leaves) if ohe_leaves else np.empty((n_samples, 0), dtype=np.float32)

    lin_raw_features = []
    for model in en_res['lin_models']:
        raw_score = model.predict(X_np, raw_score=True)
        if raw_score.ndim > 1 and raw_score.shape[1] == 1:
            raw_score = raw_score.ravel()
        elif raw_score.ndim > 1:
            for c in range(raw_score.shape[1]):
                lin_raw_features.append(raw_score[:, c].astype(np.float32))
            continue

        lin_raw_features.append(raw_score.astype(np.float32))

        num_trees = model.booster_.num_trees()
        prev = np.zeros(n_samples)
        for k in range(1, num_trees + 1):
            cum = model.booster_.predict(X_np, raw_score=True, num_iteration=k)
            if cum.ndim > 1 and cum.shape[1] > 1:
                continue
            if cum.ndim > 1:
                cum = cum.ravel()
            tree_k_score = cum - prev
            prev = cum
            lin_raw_features.append(tree_k_score.astype(np.float32))

    Lin_features = np.column_stack(lin_raw_features) if lin_raw_features else np.empty((n_samples, 0), dtype=np.float32)

    Dense = np.hstack([X_np, Lin_features])
    Dense_scaled = en_res['scaler'].transform(Dense)
    Z = np.hstack([Dense_scaled, L_features])

    Z_final = Z[:, en_res['features']]

    if classifier:
        score = en_res['model'].decision_function(Z_final)
        if score.ndim > 1:
            exps = np.exp(score - np.max(score, axis=1, keepdims=True))
            p = exps / np.sum(exps, axis=1, keepdims=True)
            preds = p[:, 2] - p[:, 0]
        else:
            preds = 1.0 / (1.0 + np.exp(-score))
    else:
        preds = en_res['model'].predict(Z_final)

    return preds
class WeakResidualMetaRegressor:
    def __init__(
        self, strategy_name: Optional[str] = None, reports_dir: Optional[str] = None
    ):
        self.strategy_name = strategy_name
        self.reports_dir = reports_dir
        self.selected_features: list[str] = []
        self.ridge_model: Optional[Pipeline] = None
        self.lgbm_model: Optional[Any] = None
        self.oof_probs: Optional[np.ndarray] = None
        self.model: Optional[Any] = None
        self._diag: Dict[str, np.ndarray] = {}
        self._leaf_var_maps: list[dict[int, float]] = []
        self._leaf_cnt_maps: list[dict[int, int]] = []
        self._reg_unc_a: float = 1.0
        self._reg_unc_C: float = 1.0
        self._reg_unc_lo: float = 0.0
        self._reg_unc_hi: float = 1.0

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
            lv += np.array(
                [float(vmap.get(int(i), 0.0)) for i in ids], dtype=np.float64
            )
            lc += np.array([float(cmap.get(int(i), 1)) for i in ids], dtype=np.float64)
            lc_trees[:, t] = np.array(
                [float(cmap.get(int(i), 1)) for i in ids], dtype=np.float64
            )
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
        self, X, y, sample_weight=None, groups=None, y_per_horizon=None, y_binary=None
    ):
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_t = np.asarray(y, dtype=np.float32)

        ridge_feats = _preridge_elasticnet_select(
            X_df, y_t, classifier=False, max_keep=120
        )
        X_ridge = X_df[ridge_feats]
        ridge = Pipeline(
            [("scaler", RobustScaler()), ("ridge", Ridge(alpha=0.5, random_state=42))]
        )
        y_t_fit = (0.65 * np.arcsinh(y_t) + 0.35 * y_t).astype(np.float32)
        if sample_weight is None:
            ridge.fit(X_ridge, y_t_fit)
        else:
            ridge.fit(
                X_ridge,
                y_t_fit,
                ridge__sample_weight=np.asarray(sample_weight, dtype=np.float32),
            )
        ridge_pred = ridge.predict(X_ridge).astype(np.float32)

        ridge_target = y_t
        signed_residual = (ridge_target - ridge_pred).astype(np.float32)
        lgb_feats = _cluster_redundant_features(
            X_df[ridge_feats], signed_residual, thr=0.97
        )
        X_lgb0 = X_df[lgb_feats]
        lgb_feats = _univariate_screen(
            X_lgb0, ridge_target, ridge_pred, signed_residual, top_k=150
        )
        X_lgb1 = X_lgb0[lgb_feats]
        lgb_feats = _iterative_fold_presence_prune(
            X_lgb1, signed_residual, classifier=False, min_features=40
        )
        X_lgb = X_lgb1[lgb_feats]

        if lgb is not None and X_lgb.shape[1] > 0:
            lgbm = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=500,
                max_depth=2,
                min_data_in_leaf=100,
                learning_rate=0.05,
                random_state=42,
                n_jobs=2,
            )
            X_lgb_np = X_lgb.to_numpy(dtype=np.float32)
            lgbm.fit(X_lgb_np, signed_residual)
            lgb_pred = lgbm.predict(X_lgb_np).astype(np.float32)
            train_leaf = np.asarray(
                lgbm.booster_.predict(X_lgb_np, pred_leaf=True), dtype=np.int64
            )
            if train_leaf.ndim == 1:
                train_leaf = train_leaf.reshape(-1, 1)
            self._leaf_var_maps = []
            self._leaf_cnt_maps = []
            # Note: these maps are fit on the same rows used to train this v1 model.
            # Diagnostics on train rows are therefore optimistic by construction.
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

        en_res = _elasticnet_lgbm_pipeline(X_df, signed_residual, base_score=ridge_pred, classifier=False, random_state=42)
        en_ridge_pred = en_res['oof_predictions'].astype(np.float32)

        final = ridge_pred + 0.3 * lgb_pred * unc + 0.3 * en_ridge_pred * unc

        self.en_pipeline = en_res
        self.selected_features = list(dict.fromkeys(ridge_feats + lgb_feats))
        self.ridge_model = ridge
        self.lgbm_model = lgbm
        self.model = ridge
        self.oof_probs = final.astype(np.float32)
        self._diag = {
            "ridge_features": np.array(ridge_feats, dtype=object),
            "lgbm_features": np.array(lgb_feats, dtype=object),
            "ridge_pred": ridge_pred,
            "lgbm_pred": lgb_pred,
            "meta_reg_en_ridge_pred": en_ridge_pred,
            "meta_reg_leaf_var": leaf_var,
            "meta_reg_leaf_count": leaf_cnt,
            "meta_reg_support_factor": support_factor.astype(np.float32),
            "meta_reg_uncertainty": unc.astype(np.float32),
            "final": final.astype(np.float32),
            "unc_lo": np.float32(self._reg_unc_lo),
            "unc_hi": np.float32(self._reg_unc_hi),
            "unc_a": np.float32(self._reg_unc_a),
            "leaf_count_cap_C": np.float32(self._reg_unc_C),
            # Persist compact leaf maps in diagnostic payload for artifact tracing.
            "leaf_var_maps": np.array(self._leaf_var_maps, dtype=object),
            "leaf_cnt_maps": np.array(self._leaf_cnt_maps, dtype=object),
        }
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        rf = [c for c in self._diag.get("ridge_features", []) if c in X_df.columns]
        lf = [c for c in self._diag.get("lgbm_features", []) if c in X_df.columns]
        r = self.ridge_model.predict(X_df.reindex(columns=rf, fill_value=0.0)).astype(
            np.float32
        )
        X_lgb_np = X_df.reindex(columns=lf, fill_value=0.0).to_numpy(dtype=np.float32)
        l = (
            self.lgbm_model.predict(X_lgb_np).astype(np.float32)
            if self.lgbm_model is not None and len(lf) > 0
            else np.zeros(len(X_df), dtype=np.float32)
        )
        u_dict = self._compute_reg_lgbm_uncertainty(X_lgb_np)
        u = u_dict["uncertainty"]

        en_ridge_pred = 0.0
        if hasattr(self, 'en_pipeline') and self.en_pipeline is not None:
            en_ridge_pred = _en_pipeline_predict(X_df, self.en_pipeline, classifier=False)

        return (r + 0.3 * l * u + 0.3 * en_ridge_pred * u).astype(np.float32)

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

    def fit(self, X, y, sample_weight=None, groups=None, **kwargs):
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

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

        clf_residual = base_clf_correct.astype(np.float32) - ridge_oof_prob
        y3 = np.ones(len(clf_residual), dtype=np.int32)
        for tr, va in cv.split(X_ridge_np, base_clf_correct):
            q1, q2 = np.quantile(clf_residual[tr], [1 / 3, 2 / 3])
            y3[va] = 1
            y3[va][clf_residual[va] < q1] = 0
            y3[va][clf_residual[va] >= q2] = 2

        lgb_feats = _cluster_redundant_features(
            X_df[ridge_feats], clf_residual, thr=0.97
        )
        X_lgb0 = X_df[lgb_feats]
        lgb_feats = _univariate_screen(
            X_lgb0,
            base_clf_correct.astype(np.float32),
            ridge_oof_prob,
            clf_residual,
            top_k=150,
        )
        X_lgb1 = X_lgb0[lgb_feats]
        lgb_feats = _iterative_fold_presence_prune(
            X_lgb1, y3, classifier=True, min_features=40
        )
        X_lgb = X_lgb1[lgb_feats]

        if lgb is not None and X_lgb.shape[1] > 0:
            X_lgb_np = X_lgb.to_numpy(dtype=np.float32)
            p3 = np.zeros((len(X_lgb_np), 3), dtype=np.float32)
            cv_lgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
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
            )
            clf.fit(X_lgb_np, y3)
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

        en_res = _elasticnet_lgbm_pipeline(X_df, y3, base_score=ridge_oof_prob, classifier=True, random_state=42)
        en_ridge_oof_prob = en_res['oof_predictions'].astype(np.float32)

        final_raw = np.clip(
            ridge_oof_prob + lambda_clf * lgb_signed * unc + 0.3 * en_ridge_oof_prob, 1e-4, 1 - 1e-4
        )

        final_cal_oof = np.zeros(len(final_raw), dtype=np.float32)
        cal_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
        for tr, va in cal_cv.split(final_raw.reshape(-1, 1), base_clf_correct):
            cal_fold = IsotonicRegression(out_of_bounds="clip")
            cal_fold.fit(final_raw[tr], base_clf_correct[tr])
            final_cal_oof[va] = np.clip(cal_fold.predict(final_raw[va]), 1e-4, 1 - 1e-4)
        final = np.clip(final_cal_oof, 1e-4, 1 - 1e-4)
        # Deployment calibrator is full-fit on all training rows.
        # _diag["final"] remains cross-fit calibrated OOF.
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(final_raw, base_clf_correct)

        self.en_pipeline = en_res
        self.selected_features = list(dict.fromkeys(ridge_feats + lgb_feats))
        self.ridge_model = ridge
        self.lgbm_model = clf
        self.calibrator = cal
        self.model = ridge
        self.oof_probs = final.astype(np.float32)
        self._diag = {
            "ridge_features": np.array(ridge_feats, dtype=object),
            "lgbm_features": np.array(lgb_feats, dtype=object),
            "base_clf_correct": base_clf_correct.astype(np.float32),
            "ridge_prob_correct": ridge_oof_prob.astype(np.float32),
            "meta_clf_lgbm_adj": lgb_signed.astype(np.float32),
            "lgbm_pred": lgb_signed.astype(np.float32),  # backward compat
            "meta_clf_en_ridge_pred": en_ridge_oof_prob.astype(np.float32),
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
        score = self.ridge_model.decision_function(
            X_df.reindex(columns=rf, fill_value=0.0)
        ).astype(np.float32)
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

        en_ridge_pred = 0.5
        if hasattr(self, 'en_pipeline') and self.en_pipeline is not None:
            en_ridge_pred = _en_pipeline_predict(X_df, self.en_pipeline, classifier=True)

        f = np.clip(p_r + lam * lgb_signed * u + 0.3 * en_ridge_pred, 1e-4, 1 - 1e-4)
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
            "meta_clf_en_ridge_pred": np.asarray(
                clf_model._diag.get("meta_clf_en_ridge_pred"), dtype=np.float32
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
            "meta_reg_en_ridge_pred": np.asarray(
                reg_model._diag.get("meta_reg_en_ridge_pred"), dtype=np.float32
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
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import roc_auc_score
import warnings

try:
    import lightgbm as lgb
except Exception:
    lgb = None

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.sum(a_b & b_b)
    union = np.sum(a_b | b_b)
    if union == 0:
        return 0.0
    return float(inter / union)

def _signed_rank_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float32)
    pos_mask = y > 0
    neg_mask = y < 0
    if np.any(pos_mask):
        import pandas as pd
        out[pos_mask] = pd.Series(y[pos_mask]).rank(pct=True).to_numpy()
    if np.any(neg_mask):
        import pandas as pd
        out[neg_mask] = -pd.Series(np.abs(y[neg_mask])).rank(pct=True).to_numpy()
    return out

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 8:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)

def _safe_pointbiserial(x: np.ndarray, y: np.ndarray) -> float:
    # y is binary (0/1)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 8:
        return 0.0
    r = pointbiserialr(y[m], x[m]).correlation
    return float(0.0 if not np.isfinite(r) else r)

def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10 or len(np.unique(y[m])) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[m], x[m]))
    except:
        return 0.5

def _get_top_k_mask(pred: np.ndarray, pct: float) -> np.ndarray:
    n = len(pred)
    k = max(1, int(np.ceil(pct * n)))
    idx = np.argsort(pred)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def _fast_corr_matrix(Z, is_binary):
    n_features = Z.shape[1]
    corr = np.zeros((n_features, n_features), dtype=np.float32)
    Z_rank = pd.DataFrame(Z).rank(pct=True).to_numpy()

    dense_mask = ~np.array(is_binary)
    if np.any(dense_mask):
        Z_dense = Z_rank[:, dense_mask]
        corr_dense = np.abs(np.corrcoef(Z_dense.T))
    else:
        corr_dense = None

    bin_idx = np.where(is_binary)[0]
    dense_idx = np.where(~np.array(is_binary))[0]

    if np.any(dense_mask) and corr_dense is not None:
        for i, d1 in enumerate(dense_idx):
            for j, d2 in enumerate(dense_idx):
                corr[d1, d2] = corr_dense[i, j]

    for i, b1 in enumerate(bin_idx):
        for j, b2 in enumerate(bin_idx):
            if i == j:
                corr[b1, b2] = 1.0
            elif i < j:
                val = _jaccard(Z[:, b1], Z[:, b2])
                corr[b1, b2] = val
                corr[b2, b1] = val

    for d in dense_idx:
        for b in bin_idx:
            val = abs(np.corrcoef(Z_rank[:, d], Z[:, b])[0, 1])
            if not np.isfinite(val):
                val = 0.0
            corr[d, b] = val
            corr[b, d] = val

    return corr

def _elasticnet_lgbm_pipeline(X: pd.DataFrame, y: np.ndarray, classifier: bool = False, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n_samples = len(X)

    if lgb is None:
        raise RuntimeError("lightgbm is not available.")

    X_np = X.to_numpy(dtype=np.float32)
    y_target = y.astype(np.int32) if classifier else y.astype(np.float32)

    leaf_matrices = []
    lgb_models = []
    unique_leaf_vals = []
    leaf_feature_names = []
    ohe_leaves = []

    def get_lgbm_base_params(d, leaf_pct, is_linear=False):
        min_data = max(10, int(n_samples * leaf_pct))
        params = {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'max_depth': d,
            'num_leaves': 2**d if d else 31,
            'min_data_in_leaf': min_data,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l2': 5.0,
            'min_gain_to_split': 0.001,
            'max_bin': 127,
            'random_state': random_state,
            'n_jobs': 2,
        }
        if is_linear:
            params['linear_tree'] = True
        if classifier:
            params['lambda_l1'] = 0.0
            params['min_sum_hessian_in_leaf'] = 1e-3
        return params

    def fit_lgbm(params, is_linear=False):
        if classifier:
            _classes = np.unique(y_target)
            if len(_classes) > 2:
                model = lgb.LGBMClassifier(objective='multiclass', num_class=len(_classes), **params)
            else:
                model = lgb.LGBMClassifier(objective='binary', **params)
        else:
            model = lgb.LGBMRegressor(objective='huber', alpha=0.9, **params)

        early_stop = 15 if is_linear else 25
        # we need eval set for early stopping
        eval_idx = rng.choice(n_samples, int(0.2*n_samples), replace=False)
        tr_idx = np.setdiff1d(np.arange(n_samples), eval_idx)

        model.fit(
            X_np[tr_idx], y_target[tr_idx],
            eval_set=[(X_np[eval_idx], y_target[eval_idx])],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]
        )
        return model

    # Depth 3 models
    for pct in [0.02, 0.04, 0.06]:
        params = get_lgbm_base_params(3, pct)
        model = fit_lgbm(params)
        lgb_models.append(('tree_d3_p' + str(int(pct*100)), model))

    # Depth 4 models
    for pct in [0.04, 0.06, 0.08]:
        params = get_lgbm_base_params(4, pct)
        params['num_leaves'] = 16
        model = fit_lgbm(params)
        lgb_models.append(('tree_d4_p' + str(int(pct*100)), model))

    # Extract leaves
    for name, model in lgb_models:
        leaves = model.booster_.predict(X_np, pred_leaf=True)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        for c in range(leaves.shape[1]):
            col = leaves[:, c]
            uniques = np.unique(col)
            unique_leaf_vals.append(uniques)
            for val in uniques:
                ohe_leaves.append((col == val).astype(np.float32))
                leaf_feature_names.append(f"LGBM_{name}_tree{c}_val{val}")

    # Linear Tree LGBM models
    lin_raw_features = []
    lin_feature_names = []
    lin_models = []

    for pct in [0.05, 0.10, 0.15]:
        params = get_lgbm_base_params(3, pct, is_linear=True)
        model = fit_lgbm(params, is_linear=True)
        name = f"linear_lgbm_p{int(pct*100)}"
        lin_models.append((name, model))

        # model level raw score
        raw_score = model.predict(X_np, raw_score=True)
        if raw_score.ndim > 1 and raw_score.shape[1] == 1:
            raw_score = raw_score.ravel()
        elif raw_score.ndim > 1: # multiclass
            for c in range(raw_score.shape[1]):
                lin_raw_features.append(raw_score[:, c].astype(np.float32))
                lin_feature_names.append(f"{name}_c{c}_raw_score")
            continue

        lin_raw_features.append(raw_score.astype(np.float32))
        lin_feature_names.append(f"{name}_raw_score")

        # per-tree raw scores
        num_trees = model.booster_.num_trees()
        prev = np.zeros(n_samples)
        for k in range(1, num_trees + 1):
            cum = model.booster_.predict(X_np, raw_score=True, num_iteration=k)
            if cum.ndim > 1 and cum.shape[1] > 1:
                # ignore per-tree multiclass raw scores to save complexity
                continue
            if cum.ndim > 1:
                cum = cum.ravel()
            tree_k_score = cum - prev
            prev = cum
            lin_raw_features.append(tree_k_score.astype(np.float32))
            lin_feature_names.append(f"{name}_tree{k:03d}_raw_score")

    if ohe_leaves:
        L_features = np.column_stack(ohe_leaves)
    else:
        L_features = np.empty((n_samples, 0), dtype=np.float32)

    if lin_raw_features:
        Lin_features = np.column_stack(lin_raw_features)
    else:
        Lin_features = np.empty((n_samples, 0), dtype=np.float32)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_np)

    Z = np.hstack([X_scaled, Lin_features, L_features])
    all_feature_names = list(X.columns) + lin_feature_names + leaf_feature_names
    is_binary = [False] * X_scaled.shape[1] + [False] * Lin_features.shape[1] + [True] * L_features.shape[1]

    # --- Step 2: IC-based Fast Pruning ---
    sub_5k_idx = rng.choice(n_samples, min(5000, n_samples), replace=False)
    Z_5k = Z[sub_5k_idx]
    y_5k = y_target[sub_5k_idx]

    ic_scores = []
    for j in range(Z.shape[1]):
        if classifier:
            if is_binary[j]:
                # point-biserial / AUC isn't strictly IC but requested
                val = max(abs(_safe_pointbiserial(Z_5k[:, j], y_5k)), abs(_safe_auc(Z_5k[:, j], y_5k) - 0.5) * 2)
            else:
                val = abs(_safe_auc(Z_5k[:, j], y_5k) - 0.5) * 2
        else:
            val = abs(_safe_spearman(Z_5k[:, j], y_5k))
        ic_scores.append(val)
    ic_scores = np.array(ic_scores)

    keep_count_1 = int(200 + 0.33 * Z.shape[1])
    keep_count_1 = min(keep_count_1, Z.shape[1])
    idx_top_ic = np.argsort(ic_scores)[-keep_count_1:]

    sub_3k_idx = rng.choice(n_samples, min(3000, n_samples), replace=False)
    Z_3k = Z[sub_3k_idx][:, idx_top_ic]
    ic_scores_subset = ic_scores[idx_top_ic]

    order = np.argsort(ic_scores_subset)[::-1]
    keep_idx_2_rel = []
    dropped_2 = np.zeros(len(order), dtype=bool)

    is_binary_subset = [is_binary[i] for i in idx_top_ic]
    corr_3k = _fast_corr_matrix(Z_3k, is_binary_subset)

    for i in range(len(order)):
        curr_i = order[i]
        if dropped_2[curr_i]:
            continue
        keep_idx_2_rel.append(curr_i)
        dropped_2 |= (corr_3k[curr_i] > 0.98)
        dropped_2[curr_i] = False

    idx_stage_2 = idx_top_ic[keep_idx_2_rel]

    stability_scores = []
    for j in range(len(idx_stage_2)):
        orig_j = idx_stage_2[j]
        ic_vals = []
        for _ in range(5):
            sub_idx = rng.choice(n_samples, min(3000, n_samples), replace=False)
            if classifier:
                val = _safe_pointbiserial(Z[sub_idx, orig_j], y_target[sub_idx])
            else:
                val = _safe_spearman(Z[sub_idx, orig_j], y_target[sub_idx])
            ic_vals.append(val)
        ic_vals = np.array(ic_vals)
        pos_frac = np.mean(ic_vals > 0)
        neg_frac = np.mean(ic_vals < 0)
        sign_consistency = max(pos_frac, neg_frac)
        stability = np.median(np.abs(ic_vals)) * sign_consistency
        stability_scores.append(stability)

    stability_scores = np.array(stability_scores)
    keep_count_3 = int(100 + 0.25 * len(idx_stage_2))
    keep_count_3 = min(keep_count_3, len(idx_stage_2))
    idx_top_stab_rel = np.argsort(stability_scores)[-keep_count_3:]
    idx_stage_3 = idx_stage_2[idx_top_stab_rel]

    sub_5k_idx_2 = rng.choice(n_samples, min(5000, n_samples), replace=False)
    Z_5k_2 = Z[sub_5k_idx_2][:, idx_stage_3]
    stab_subset = stability_scores[idx_top_stab_rel]

    order_3 = np.argsort(stab_subset)[::-1]
    keep_idx_4_rel = []
    dropped_4 = np.zeros(len(order_3), dtype=bool)

    is_binary_subset_2 = [is_binary[i] for i in idx_stage_3]
    corr_5k = _fast_corr_matrix(Z_5k_2, is_binary_subset_2)

    for i in range(len(order_3)):
        curr_i = order_3[i]
        if dropped_4[curr_i]:
            continue
        keep_idx_4_rel.append(curr_i)
        dropped_4 |= (corr_5k[curr_i] > 0.96)
        dropped_4[curr_i] = False

    final_fast_pruned_idx = idx_stage_3[keep_idx_4_rel]
    active_features = list(final_fast_pruned_idx)

    # --- Step 3: Iterative ElasticNet Pruning ---
    sub_20k_idx = rng.choice(n_samples, min(20000, n_samples), replace=False)
    Z_20k = Z[sub_20k_idx]
    y_20k = y_target[sub_20k_idx]

    alpha_grid = [0.05, 0.1, 0.3, 0.5]
    l1_ratio_grid = [0.4, 0.6, 0.8]

    best_historical_J = -np.inf
    iteration = 1
    models_history = []
    prev_preds = bs_20k.copy()

    while True:
        folds = int(2 + 0.5 * iteration)
        threshold = 0.10 * iteration

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state) if classifier else KFold(n_splits=folds, shuffle=True, random_state=random_state)

        feature_metrics = {f_idx: {'coeffs': [], 'presence': 0, 'top30_contrib': []} for f_idx in active_features}
        total_fits = folds * len(alpha_grid) * len(l1_ratio_grid)

        iter_models = []
        Z_iter = Z_20k[:, active_features]

        top30_mask = _get_top_k_mask(bs_20k, 0.30)
        sample_weights = np.ones(len(y_20k), dtype=np.float32)
        sample_weights[top30_mask] = 1.0 + 0.5 * iteration

        for alpha in alpha_grid:
            for l1 in l1_ratio_grid:
                oof_preds = np.zeros(len(y_20k))

                for tr, va in cv.split(Z_iter, y_20k):
                    if classifier:
                        C_val = 1.0 / max(alpha, 1e-6)
                        mdl = LogisticRegression(penalty='l1', solver='liblinear', C=C_val, random_state=random_state, max_iter=1000)
                    else:
                        mdl = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=random_state, max_iter=2000)

                    y_fit_20k = y_fit[sub_20k_idx]

                    rank_focus_tr = 0.7 + 0.6 * np.sqrt(np.clip((pd.Series(prev_preds[tr]).rank(pct=True).to_numpy()), 0, 1))
                    if classifier:
                        y_bin_tr = (y_20k[tr] > 0.5).astype(int)
                        confidence_tr = np.clip(prev_preds[tr], 0, 1)
                        pred_class_tr = (prev_preds[tr] > 0.5).astype(int)
                        match_tr = (y_bin_tr == pred_class_tr)
                        error_weight_tr = np.where(match_tr,
                            np.minimum(1.6, 1.0 + 0.2 * confidence_tr),
                            np.minimum(1.6, 1.0 + 0.5 * confidence_tr))
                    else:
                        resid_tr = np.abs(y_20k[tr] - prev_preds[tr])
                        resid_rank_tr = pd.Series(resid_tr).rank(pct=True).to_numpy()
                        error_weight_tr = np.minimum(1.6, 1.0 + 0.3 * resid_rank_tr)

                    weight_distillation_tr = sample_weights[tr] * error_weight_tr * np.sqrt(rank_focus_tr)

                    mdl.fit(Z_iter[tr], y_fit_20k[tr], sample_weight=weight_distillation_tr)

                    if classifier:
                        pred_va = mdl.predict_proba(Z_iter[va])[:, 1]
                    else:
                        pred_va = mdl.predict(Z_iter[va])

                    oof_preds[va] = pred_va

                    coefs = np.ravel(mdl.coef_)

                    # Compute feature contribution on va top30
                    va_top30 = _get_top_k_mask(pred_va, 0.30)

                    for i_act, f_idx in enumerate(active_features):
                        val = coefs[i_act]
                        if abs(val) > 1e-6:
                            feature_metrics[f_idx]['presence'] += 1
                            feature_metrics[f_idx]['coeffs'].append(val)

                            contrib = Z_iter[va][va_top30, i_act] * val
                            feature_metrics[f_idx]['top30_contrib'].append(np.mean(contrib) if len(contrib) > 0 else 0)

                m30_mask = _get_top_k_mask(oof_preds, 0.30)
                m15_mask = _get_top_k_mask(oof_preds, 0.15)

                if classifier:
                    y_bin = (y_20k > 0.5).astype(float)
                    hr30 = np.mean(y_bin[m30_mask]) if np.any(m30_mask) else 0.0
                    hr15 = np.mean(y_bin[m15_mask]) if np.any(m15_mask) else 0.0

                    s_top = oof_preds[m30_mask]
                    y_top = y_bin[m30_mask]
                    if len(s_top) >= 10:
                        q = np.quantile(s_top, np.linspace(0.0, 1.0, 6))
                        vals = []
                        for i in range(5):
                            mm = (s_top >= q[i]) & (s_top < q[i + 1] if i < 4 else s_top <= q[i + 1])
                            if np.any(mm):
                                vals.append(float(np.mean(y_top[mm])))
                        std30 = np.std(vals) if vals else 0.0
                    else:
                        std30 = 0.0
                    J = 0.4 * hr30 + 0.3 * hr15 - 0.3 * std30
                else:
                    ic30 = _safe_spearman(oof_preds[m30_mask], y_20k[m30_mask]) if np.any(m30_mask) else 0.0
                    ic15 = _safe_spearman(oof_preds[m15_mask], y_20k[m15_mask]) if np.any(m15_mask) else 0.0

                    s_top = oof_preds[m30_mask]
                    y_top = y_20k[m30_mask]
                    if len(s_top) >= 10:
                        q = np.quantile(s_top, np.linspace(0.0, 1.0, 6))
                        vals = []
                        for i in range(5):
                            mm = (s_top >= q[i]) & (s_top < q[i + 1] if i < 4 else s_top <= q[i + 1])
                            if np.any(mm):
                                vals.append(float(np.mean(y_top[mm])))
                        std30 = np.std(vals) if vals else 0.0
                    else:
                        std30 = 0.0
                    J = 0.4 * ic30 + 0.3 * ic15 - 0.3 * std30

                iter_models.append({'alpha': alpha, 'l1_ratio': l1, 'J': J, 'features': active_features.copy(), 'oof_preds': oof_preds})

        iter_best_J = max(m['J'] for m in iter_models)
        best_iter_model = max(iter_models, key=lambda m: m['J'])
        prev_preds = best_iter_model['oof_preds']
        models_history.extend(iter_models)

        SE = 0.02
        if iter_best_J < best_historical_J - SE:
            break

        if iter_best_J > best_historical_J:
            best_historical_J = iter_best_J

        new_active = []
        for f_idx in active_features:
            mets = feature_metrics[f_idx]
            presence_pct = mets['presence'] / total_fits
            if presence_pct == 0:
                continue
            coeffs = np.array(mets['coeffs'])
            pos_frac = np.mean(coeffs > 0)
            neg_frac = np.mean(coeffs < 0)
            sign_consistency = max(pos_frac, neg_frac)
            median_mag = np.median(np.abs(coeffs))

            top30 = np.array(mets['top30_contrib'])
            top30_mean = np.mean(top30) if len(top30)>0 else 0
            top30_std = np.std(top30) if len(top30)>0 else 1
            top30_stability = top30_mean / max(top30_std, 1e-6)

            score = (sign_consistency ** 1.5) * median_mag * abs(top30_stability)
            new_active.append((f_idx, score, presence_pct))

        if not new_active:
            break

        scores = np.array([x[1] for x in new_active])
        if len(scores) > 1 and np.max(scores) > np.min(scores):
            scores_scaled = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            scores_scaled = np.ones(len(scores))

        final_active = []
        for i, (f_idx, score, pres) in enumerate(new_active):
            val = scores_scaled[i] * pres
            if val >= threshold:
                final_active.append(f_idx)

        if len(final_active) == len(active_features) or len(final_active) < 5:
            break

        active_features = final_active
        iteration += 1

    # --- Step 4: Pareto Selection & Final Model ---
    models_history.sort(key=lambda x: x['J'], reverse=True)
    top_5 = models_history[:5]

    best_pareto = None
    best_pareto_score = -np.inf

    for m in top_5:
        score = m['J'] - 0.001 * len(m['features'])
        if score > best_pareto_score:
            best_pareto_score = score
            best_pareto = m

    final_features = best_pareto['features']

    Z_final = Z[:, final_features]

    top30_mask = _get_top_k_mask(base_score, 0.30)
    sample_weights = np.ones(len(y_target), dtype=np.float32)
    sample_weights[top30_mask] = 1.0 + 0.5 * iteration

    if classifier:
        final_model = RidgeClassifier(alpha=1.0, random_state=random_state)
    else:
        final_model = Ridge(alpha=1.0, random_state=random_state)

    final_model.fit(Z_final, y_target, sample_weight=sample_weights)

    if classifier:
        final_preds = final_model.decision_function(Z_final)
        final_preds = 1.0 / (1.0 + np.exp(-final_preds))
    else:
        final_preds = final_model.predict(Z_final)

    return {
        'model': final_model,
        'features': final_features,
        'predictions': final_preds,
        'scaler': scaler,
        'lgb_models': lgb_models,
        'lin_models': lin_models,
        'unique_leaf_vals': unique_leaf_vals,
        'n_raw_features': X.shape[1]
    }
