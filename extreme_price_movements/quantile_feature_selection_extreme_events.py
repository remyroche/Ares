"""Quantile-aware feature selection for extreme-event meta models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


@dataclass
class MDISelectionResult:
    metrics_table: pd.DataFrame
    selected_features: List[str]
    kept_after_dedupe: List[str]


def _tail_label(y: np.ndarray, tail_q: float) -> np.ndarray:
    return (y >= np.quantile(y, tail_q)).astype(np.int8)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _univariate_tail_screen(X: pd.DataFrame, y: np.ndarray, target_count: int, tail_q: float) -> pd.Series:
    z = _tail_label(y, tail_q)
    scores = {}
    for col in X.columns:
        xv = np.nan_to_num(X[col].to_numpy(dtype=float), nan=0.0)
        corr = abs(_safe_corr(xv, z))
        try:
            auc = roc_auc_score(z, xv)
            auc = max(auc, 1.0 - auc)
        except ValueError:
            auc = 0.5
        scores[col] = 0.6 * corr + 0.4 * max(0.0, auc - 0.5)
    keep = min(300, max(target_count * 10, target_count))
    return pd.Series(scores).sort_values(ascending=False).head(keep)


def _dedupe_corr_tail(X: pd.DataFrame, y: np.ndarray, cols: Sequence[str], threshold: float, tail_q: float) -> List[str]:
    if len(cols) <= 1:
        return list(cols)
    z = _tail_label(y, tail_q)
    work = list(cols)
    tail_rel = {c: abs(_safe_corr(np.nan_to_num(X[c].to_numpy(dtype=float), nan=0.0), z)) for c in work}
    corr = X[work].corr().abs().fillna(0.0)
    removed = set()
    for i, c1 in enumerate(work):
        if c1 in removed:
            continue
        for c2 in work[i + 1 :]:
            if c2 in removed:
                continue
            if corr.loc[c1, c2] > threshold:
                if tail_rel[c1] >= tail_rel[c2]:
                    removed.add(c2)
                else:
                    removed.add(c1)
                    break
    return [c for c in work if c not in removed]


def _fit_quantile_lgbm(X_tr, y_tr, X_va, y_va, alpha: float, seed: int) -> lgb.LGBMRegressor:
    n_train = max(1, int(len(X_tr)))
    min_data_in_leaf = max(2, int(np.ceil(0.012 * n_train)))
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=800,
        num_leaves=63,
        max_depth=5,
        min_data_in_leaf=min_data_in_leaf,
        min_sum_hessian_in_leaf=3e-2,
        min_gain_to_split=0.05,
        lambda_l1=1.0,
        lambda_l2=10.0,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        min_data_in_bin=127,
        max_bin=127,
        random_state=seed,
        n_jobs=3,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _dual_alpha_ranks(X: pd.DataFrame, y: np.ndarray, cols: List[str], alphas: Sequence[float], n_splits: int, tail_q: float, random_state: int) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    tables = []
    for alpha in alphas:
        fold_ranks = []
        for fold_id, (tr, va) in enumerate(kf.split(X)):
            z_tr = _tail_label(y[tr], tail_q)
            if z_tr.mean() < 0.15:
                continue
            model = _fit_quantile_lgbm(
                X.iloc[tr][cols].to_numpy(dtype=float),
                y[tr],
                X.iloc[va][cols].to_numpy(dtype=float),
                y[va],
                alpha=alpha,
                seed=random_state + fold_id,
            )
            imp = np.asarray(model.feature_importances_, dtype=float)
            order = np.argsort(-imp)
            rank = np.empty_like(order)
            rank[order] = np.arange(1, len(cols) + 1)
            fold_ranks.append(rank)
        if not fold_ranks:
            continue
        fold_ranks = np.vstack(fold_ranks)
        median_rank = np.median(fold_ranks, axis=0)
        top_m = max(5, int(0.3 * len(cols)))
        hit_rate = np.mean(fold_ranks <= top_m, axis=0)
        stability = hit_rate / np.maximum(median_rank, 1.0)
        mean_imp_proxy = 1.0 / np.maximum(median_rank, 1.0)
        std_imp_proxy = np.std(fold_ranks, axis=0)
        tables.append(
            pd.DataFrame(
                {
                    f"median_rank_{alpha}": median_rank,
                    f"hit_rate_{alpha}": hit_rate,
                    f"stability_{alpha}": stability,
                    f"mean_imp_{alpha}": mean_imp_proxy,
                    f"std_imp_{alpha}": std_imp_proxy,
                    f"rank_{alpha}": median_rank,
                },
                index=cols,
            )
        )
    if not tables:
        return pd.DataFrame(index=cols)
    out = pd.concat(tables, axis=1)
    if "rank_0.85" in out.columns and "rank_0.7" in out.columns:
        out["rank_fused"] = 0.7 * out["rank_0.85"] + 0.3 * out["rank_0.7"]
    else:
        rank_cols = [c for c in out.columns if c.startswith("rank_")]
        out["rank_fused"] = out[rank_cols].mean(axis=1)
    return out.sort_values("rank_fused")


def mdi_feature_selection_v3(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    random_state: int = 42,
    end_features: Optional[int] = None,
    min_features: int = 15,
    max_features: int = 50,
    alpha: float = 0.85,
    dual_alpha: Sequence[float] = (0.7, 0.85),
    tail_q: float = 0.80,
    **_: dict,
) -> MDISelectionResult:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    y_np = np.asarray(y, dtype=float)
    target = end_features if end_features is not None else max(min_features, min(max_features, int(np.sqrt(max(1, X.shape[1])) * 3)))

    uni = _univariate_tail_screen(X, y_np, target_count=target, tail_q=tail_q)
    pre_cols = uni.index.tolist()
    pre_cols = _dedupe_corr_tail(X, y_np, pre_cols, threshold=0.95, tail_q=tail_q)

    rank_df = _dual_alpha_ranks(X, y_np, pre_cols, alphas=dual_alpha, n_splits=n_splits, tail_q=tail_q, random_state=random_state)
    if rank_df.empty:
        selected = pre_cols[: max(min_features, min(max_features, len(pre_cols)))]
        return MDISelectionResult(metrics_table=pd.DataFrame(index=pre_cols), selected_features=selected, kept_after_dedupe=pre_cols)

    mean_cols = [c for c in rank_df.columns if c.startswith("mean_imp_")]
    std_cols = [c for c in rank_df.columns if c.startswith("std_imp_")]
    rank_df["Imp_eff"] = rank_df[mean_cols].mean(axis=1) - 0.5 * rank_df[std_cols].mean(axis=1)
    rank_df["Imp_eff"] = np.maximum(rank_df["Imp_eff"], 0.0)
    imp_floor = np.median(rank_df["Imp_eff"]) / 12.0 if len(rank_df) else 0.0
    rank_df = rank_df[rank_df["Imp_eff"] >= imp_floor]
    if rank_df.empty:
        rank_df = _dual_alpha_ranks(X, y_np, pre_cols, alphas=dual_alpha, n_splits=n_splits, tail_q=tail_q, random_state=random_state)

    rank_df = rank_df.sort_values("rank_fused")
    w = rank_df["Imp_eff"].to_numpy(dtype=float)
    if np.sum(w) > 0:
        cs = np.cumsum(w / np.sum(w))
        n_cap = int(np.searchsorted(cs, 0.98) + 1)
    else:
        n_cap = len(rank_df)

    n_final = int(np.clip(min(n_cap, target), min_features, min(max_features, len(rank_df))))
    selected = rank_df.head(n_final).index.tolist()
    return MDISelectionResult(metrics_table=rank_df, selected_features=selected, kept_after_dedupe=pre_cols)


mdi_feature_selection_leakage_safe = mdi_feature_selection_v3
