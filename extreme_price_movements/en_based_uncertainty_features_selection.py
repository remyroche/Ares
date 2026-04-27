from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from extreme_price_movements.src_utils_tprint import tprint
except Exception:
    tprint = print

ALPHA_GRID = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
L1_RATIO_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass
class FoldTransform:
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class CandidateResult:
    alpha: float
    l1_ratio: float
    mean_j: float
    std_j: float
    mean_auc: float
    mean_mono: float
    mean_stab: float
    fold_metrics: list[dict[str, float]]


def _alpha_to_c(alpha: float) -> float:
    """Map alpha->C for sklearn LogisticRegression.

    Consistent mapping used everywhere in this module:
    C = 1 / alpha
    """
    return 1.0 / max(float(alpha), 1e-12)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1 - 1e-4)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _assemble_raw_features(
    df: pd.DataFrame, mode: str = "clf"
) -> tuple[pd.DataFrame, str, str]:
    base_col = _resolve_col(
        df,
        [
            "oof_prob",
            "oof_pred",
            "pred",
            "base",
            "base_H2",
            "base_H4",
            "p_final_oof",
        ],
    )
    if base_col is None:
        raise ValueError("No base prediction column found.")
    if str(mode).lower() == "reg":
        y_col = _resolve_col(df, ["y_ret", "__y_ret__", "return", "target_reg"])
    else:
        y_col = _resolve_col(df, ["y_bin", "label", "target", "y_true"])
    if y_col is None:
        if str(mode).lower() == "reg":
            raise ValueError(
                "No regression target found (expected y_ret/__y_ret__/return/target_reg)."
            )
        raise ValueError(
            "No binary label column found (expected y_bin/label/target/y_true)."
        )

    raw = pd.DataFrame(index=df.index)
    base_prob = np.asarray(df[base_col].values, dtype=np.float32)
    raw["base_pred"] = base_prob
    raw["base_logit"] = _logit(base_prob)
    raw["abs_base_logit"] = np.abs(raw["base_logit"].values).astype(np.float32)

    col_map = {
        "pred_std_robust": ["oof_sigma_robust", "robust_sigma", "pred_std_robust"],
        "vote_entropy": ["oof_tree_vote_entropy", "vote_entropy"],
        "leaf_support_q25": ["oof_tree_leaf_support_q25", "leaf_support_q25"],
        "leaf_target_std_mean": [
            "oof_tree_leaf_target_std_mean",
            "leaf_target_std_mean",
            "oof_tree_leaf_target_iqr_mean",
            "leaf_target_iqr_mean",
        ],
        "leaf_centroid_dist_mean": [
            "oof_tree_leaf_centroid_dist_mean",
            "leaf_centroid_dist_mean",
        ],
        "leaf_centroid_dist_cv": [
            "oof_tree_leaf_centroid_dist_cv",
            "leaf_centroid_dist_cv",
        ],
        "vote_margin": ["oof_tree_vote_margin", "vote_margin"],
        "vote_top_gap": ["oof_tree_vote_top_gap", "vote_top_gap"],
    }
    for k, cand in col_map.items():
        c = _resolve_col(df, cand)
        if c is not None:
            raw[k] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # uncertainty direction unification
    for maybe_flip in ["vote_margin", "vote_top_gap", "leaf_support_q25"]:
        if maybe_flip in raw.columns:
            raw[maybe_flip] = -raw[maybe_flip].values

    if "pred_std_robust" not in raw.columns:
        raw["pred_std_robust"] = np.abs(raw["base_logit"].values) * 0.0
    if "vote_entropy" not in raw.columns:
        raw["vote_entropy"] = np.abs(raw["base_logit"].values) * 0.0
    if "leaf_support_q25" not in raw.columns:
        raw["leaf_support_q25"] = np.abs(raw["base_logit"].values) * 0.0
    if "leaf_target_std_mean" not in raw.columns:
        raw["leaf_target_std_mean"] = np.abs(raw["base_logit"].values) * 0.0
    if "leaf_centroid_dist_mean" not in raw.columns:
        raw["leaf_centroid_dist_mean"] = np.abs(raw["base_logit"].values) * 0.0
    if "leaf_centroid_dist_cv" not in raw.columns:
        raw["leaf_centroid_dist_cv"] = np.abs(raw["base_logit"].values) * 0.0

    raw["distance_x_low_support"] = (
        raw["leaf_centroid_dist_mean"] * raw["leaf_support_q25"]
    )
    raw["distance_x_leaf_target_std"] = (
        raw["leaf_centroid_dist_mean"] * raw["leaf_target_std_mean"]
    )
    raw["disagreement_x_low_support"] = raw["vote_entropy"] * raw["leaf_support_q25"]
    raw["disagreement_x_distance"] = (
        raw["vote_entropy"] * raw["leaf_centroid_dist_mean"]
    )
    raw["abs_base_logit_x_disagreement"] = raw["abs_base_logit"] * raw["vote_entropy"]
    raw["abs_base_logit_x_low_support"] = (
        raw["abs_base_logit"] * raw["leaf_support_q25"]
    )
    raw["abs_base_logit_x_distance"] = (
        raw["abs_base_logit"] * raw["leaf_centroid_dist_mean"]
    )

    return raw, base_col, y_col


def _fit_transform_fold(
    X_tr: np.ndarray, X_va: np.ndarray
) -> tuple[np.ndarray, np.ndarray, FoldTransform]:
    lo = np.nanpercentile(X_tr, 1.0, axis=0)
    hi = np.nanpercentile(X_tr, 99.0, axis=0)
    X_tr_w = np.clip(X_tr, lo, hi)
    X_va_w = np.clip(X_va, lo, hi)
    mu = np.nanmean(X_tr_w, axis=0)
    sd = np.nanstd(X_tr_w, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X_tr_z = (X_tr_w - mu) / sd
    X_va_z = (X_va_w - mu) / sd
    X_tr_z = np.nan_to_num(X_tr_z, nan=0.0, posinf=0.0, neginf=0.0)
    X_va_z = np.nan_to_num(X_va_z, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        X_tr_z.astype(np.float32),
        X_va_z.astype(np.float32),
        FoldTransform(lo, hi, mu, sd),
    )


def _mono_top3(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 20:
        return 0.0
    order = np.argsort(-p)
    y_ord = y[order]
    bins = np.array_split(y_ord, 10)
    r = [float(np.mean(b >= 0.5)) if len(b) else 0.0 for b in bins[:3]]
    r1, r2, r3 = r
    return float((r1 - r3) - max(0.0, r2 - r1) - max(0.0, r3 - r2))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (p >= bins[i]) & (p < bins[i + 1])
        else:
            m = (p >= bins[i]) & (p <= bins[i + 1])
        if not np.any(m):
            continue
        ece += float(np.mean(m)) * abs(float(np.mean(p[m]) - np.mean(y[m])))
    return float(ece)


def _rank_ic(y: np.ndarray, p: np.ndarray) -> float:
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p, dtype=float)
    if len(yv) < 5:
        return 0.0
    ry = pd.Series(yv).rank(pct=True).values
    rp = pd.Series(pv).rank(pct=True).values
    return (
        float(np.corrcoef(ry, rp)[0, 1])
        if np.isfinite(np.corrcoef(ry, rp)[0, 1])
        else 0.0
    )


def _top_slice_metrics(
    y: np.ndarray, p: np.ndarray, frac: float = 0.30
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-p)[:k]
    y_top = y[idx]
    y_all = y
    hit_top = float(np.mean(y_top >= 0.5))
    hit_all = float(np.mean(y_all >= 0.5))
    lift = hit_top / max(hit_all, 1e-6)
    ic = _rank_ic(y_top, p[idx])
    # chunked IC std as stability proxy
    chunks = np.array_split(np.arange(len(idx)), min(5, len(idx)))
    chunk_ics = []
    for c in chunks:
        if len(c) < 3:
            continue
        chunk_ics.append(_rank_ic(y_top[c], p[idx][c]))
    ic_std = float(np.std(chunk_ics)) if chunk_ics else 0.0
    stability = float(ic - 0.5 * ic_std)
    # top decile hit rate std
    order = np.argsort(-p)
    decs = np.array_split(order, 10)
    d_hits = [float(np.mean(y[d] >= 0.5)) if len(d) else 0.0 for d in decs]
    mono = _mono_top3(y, p)
    td_std = float(np.std(d_hits[:1] if len(d_hits) == 1 else d_hits[:2]))
    ece_top30 = _ece(y_top, np.clip(p[idx], 1e-6, 1 - 1e-6))
    return {
        "lift_top30": lift,
        "ic_top30": ic,
        "ic_top30_std": ic_std,
        "stability_top30": stability,
        "decile_monotonicity": mono,
        "top_decile_hit_rate_std": td_std,
        "ece_top30": ece_top30,
    }


def _global_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    brier = _brier(y, p)
    top = _top_slice_metrics(y, p, frac=0.30)
    return {
        "auc": auc,
        "auc_over_random": auc / 0.5,
        "brier": brier,
        **top,
    }


def _global_metrics_reg(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(yv, pv))) if len(yv) else 0.0
    mae = float(mean_absolute_error(yv, pv)) if len(yv) else 0.0
    ic = _rank_ic(yv, pv)
    top = _top_slice_metrics((yv > np.nanmedian(yv)).astype(float), pv, frac=0.30)
    return {
        "rmse": rmse,
        "mae": mae,
        "ic_global": ic,
        **top,
    }


def _evaluate_candidate(
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    alpha: float,
    l1_ratio: float,
    mode: str = "clf",
) -> CandidateResult:
    fold_vals: list[dict[str, float]] = []
    js: list[float] = []
    monos: list[float] = []
    aucs: list[float] = []
    lifts: list[float] = []
    for i, (tr, va) in enumerate(folds):
        X_tr, X_va = fold_cache[i]
        y_tr = y[tr]
        y_va = y[va]
        if str(mode).lower() == "reg":
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                fit_intercept=True,
                max_iter=8000,
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            p = model.predict(X_va)
            mono = _mono_top3((y_va > np.nanmedian(y_va)).astype(float), p)
            auc = _rank_ic(y_va, p)
        else:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                fit_intercept=True,
                max_iter=8000,
                C=_alpha_to_c(alpha),
                l1_ratio=float(l1_ratio),
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_va)[:, 1]
            mono = _mono_top3(y_va, p)
            auc = float(roc_auc_score(y_va, p)) if len(np.unique(y_va)) > 1 else 0.5
        k = max(1, int(np.ceil(0.30 * len(y_va))))
        top_idx = np.argsort(-p)[:k]
        hit_top = float(np.mean(y_va[top_idx] >= 0.5)) if mode != "reg" else 0.0
        hit_all = float(np.mean(y_va >= 0.5)) if mode != "reg" else 0.0
        lift = hit_top / max(hit_all, 1e-6) if mode != "reg" else 0.0
        monos.append(mono)
        aucs.append(auc)
        lifts.append(lift)
        fold_vals.append({"mono_top3": mono, "auc": auc, "lift_top30": lift})
    mean_mono = float(np.mean(monos))
    std_mono = float(np.std(monos))
    m_stab = float(mean_mono - 0.5 * std_mono)
    mean_auc = float(np.mean(aucs))
    mean_lift = float(np.mean(lifts))
    j = float(0.40 * mean_mono + 0.40 * m_stab + 0.20 * mean_lift)
    js = [j]
    return CandidateResult(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        mean_j=float(np.mean(js)),
        std_j=0.0,
        mean_auc=mean_auc,
        mean_mono=mean_mono,
        mean_stab=m_stab,
        fold_metrics=fold_vals,
    )


def _stage_a_prune(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    coef_active_threshold: float = 0.01,
    mode: str = "clf",
) -> tuple[list[str], list[dict[str, Any]]]:
    hist: list[dict[str, Any]] = []
    current = list(feature_names)
    n0 = len(current)
    floor_keep = int(math.ceil(0.70 * n0))
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    folds = list(skf.split(X_raw.values, y))

    round_idx = 0
    while True:
        round_idx += 1
        # cache transformed matrices by fold
        fold_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, (tr, va) in enumerate(folds):
            X_tr = X_raw.iloc[tr][current].to_numpy(dtype=np.float32)
            X_va = X_raw.iloc[va][current].to_numpy(dtype=np.float32)
            X_tr_z, X_va_z, _ = _fit_transform_fold(X_tr, X_va)
            fold_cache[i] = (X_tr_z, X_va_z)

        cand_results: list[CandidateResult] = []
        for alpha in ALPHA_GRID:
            for l1 in L1_RATIO_GRID:
                cand_results.append(
                    _evaluate_candidate(y, folds, fold_cache, alpha, l1, mode=mode)
                )

        scores = np.array([c.mean_j for c in cand_results], dtype=float)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        se = float(0.75 * np.std(scores) / max(math.sqrt(len(scores)), 1.0))
        contenders = [c for c in cand_results if c.mean_j >= best_score - se]

        feat_stats = {
            f: {"active": 0, "total": 0, "pos": 0, "neg": 0, "coef_abs": []}
            for f in current
        }
        for c in contenders:
            for tr, va in folds:
                X_tr = X_raw.iloc[tr][current].to_numpy(dtype=np.float32)
                X_va = X_raw.iloc[va][current].to_numpy(dtype=np.float32)
                X_tr_z, _, _ = _fit_transform_fold(X_tr, X_va)
                if str(mode).lower() == "reg":
                    model = ElasticNet(
                        alpha=float(c.alpha),
                        l1_ratio=float(c.l1_ratio),
                        fit_intercept=True,
                        max_iter=8000,
                        random_state=42,
                    )
                else:
                    model = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        fit_intercept=True,
                        max_iter=8000,
                        C=_alpha_to_c(c.alpha),
                        l1_ratio=float(c.l1_ratio),
                        random_state=42,
                    )
                model.fit(X_tr_z, y[tr])
                coef = model.coef_.reshape(-1)
                for j, f in enumerate(current):
                    v = float(coef[j])
                    abs_v = abs(v)
                    feat_stats[f]["total"] += 1
                    if abs_v > coef_active_threshold:
                        feat_stats[f]["active"] += 1
                        feat_stats[f]["coef_abs"].append(abs_v)
                        if v >= 0:
                            feat_stats[f]["pos"] += 1
                        else:
                            feat_stats[f]["neg"] += 1

        keep = []
        dropped = []
        for f in current:
            st = feat_stats[f]
            tot = max(int(st["total"]), 1)
            freq = float(st["active"]) / tot
            sign_cons = float(max(st["pos"], st["neg"])) / max(int(st["active"]), 1)
            med_abs = float(np.median(st["coef_abs"])) if st["coef_abs"] else 0.0
            if freq >= 0.70 and sign_cons >= 0.80:
                keep.append(f)
            else:
                dropped.append((f, freq, sign_cons, med_abs))

        if len(keep) < floor_keep:
            need = floor_keep - len(keep)
            dropped_sorted = sorted(
                dropped,
                key=lambda x: x[1] * x[2] * x[3],
                reverse=True,
            )
            keep.extend([f for f, *_ in dropped_sorted[:need]])

        keep = list(dict.fromkeys(keep))
        removed = [f for f in current if f not in keep]
        hist.append(
            {
                "round": round_idx,
                "n_features": len(current),
                "n_keep": len(keep),
                "n_removed": len(removed),
                "best_mean_j": best_score,
                "se": se,
            }
        )
        if len(removed) <= 1:
            break
        current = keep

    return current, hist


def _fit_5fold_oof(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    alpha: float,
    l1_ratio: float,
    mode: str = "clf",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_s = np.full(len(X_raw), np.nan, dtype=np.float32)
    oof_p = np.full(len(X_raw), np.nan, dtype=np.float32)
    fold_transforms: list[dict[str, Any]] = []
    for fold_i, (tr, va) in enumerate(skf.split(X_raw.values, y)):
        X_tr = X_raw.iloc[tr][features].to_numpy(dtype=np.float32)
        X_va = X_raw.iloc[va][features].to_numpy(dtype=np.float32)
        X_tr_z, X_va_z, tf = _fit_transform_fold(X_tr, X_va)
        if str(mode).lower() == "reg":
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                fit_intercept=True,
                max_iter=10000,
                random_state=42,
            )
        else:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                fit_intercept=True,
                max_iter=10000,
                C=_alpha_to_c(alpha),
                l1_ratio=float(l1_ratio),
                random_state=42,
            )
        model.fit(X_tr_z, y[tr])
        if str(mode).lower() == "reg":
            pred = model.predict(X_va_z).astype(np.float32)
            oof_s[va] = pred
            oof_p[va] = pred
        else:
            oof_s[va] = model.decision_function(X_va_z).astype(np.float32)
            oof_p[va] = model.predict_proba(X_va_z)[:, 1].astype(np.float32)
        fold_transforms.append(
            {
                "fold": int(fold_i),
                "lower": tf.lower.tolist(),
                "upper": tf.upper.tolist(),
                "mean": tf.mean.tolist(),
                "std": tf.std.tolist(),
            }
        )

    X_full = X_raw[features].to_numpy(dtype=np.float32)
    Xf_w = np.clip(
        X_full,
        np.nanpercentile(X_full, 1, axis=0),
        np.nanpercentile(X_full, 99, axis=0),
    )
    mu = np.nanmean(Xf_w, axis=0)
    sd = np.where(np.nanstd(Xf_w, axis=0) < 1e-6, 1.0, np.nanstd(Xf_w, axis=0))
    Xf_z = np.nan_to_num((Xf_w - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    if str(mode).lower() == "reg":
        final_model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=12000,
            random_state=42,
        )
    else:
        final_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            fit_intercept=True,
            max_iter=12000,
            C=_alpha_to_c(alpha),
            l1_ratio=float(l1_ratio),
            random_state=42,
        )
    final_model.fit(Xf_z, y)
    final_bundle = {
        "features": features,
        "lower": np.nanpercentile(X_full, 1, axis=0).tolist(),
        "upper": np.nanpercentile(X_full, 99, axis=0).tolist(),
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "model": final_model,
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "C": float(_alpha_to_c(alpha)),
        "mode": str(mode).lower(),
    }
    return oof_s, oof_p, fold_transforms, final_bundle


def _stratified_subsample(
    y: np.ndarray, max_rows: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    if len(y) <= max_rows:
        return np.arange(len(y))
    if rng is None:
        rng = np.random.default_rng(42)
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, max_rows // len(classes))
    idx_parts: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > per_class:
            chosen = rng.choice(cls_idx, size=per_class, replace=False)
        else:
            chosen = cls_idx
        idx_parts.append(chosen)
    all_idx = np.concatenate(idx_parts)
    if len(all_idx) > max_rows:
        all_idx = rng.choice(all_idx, size=max_rows, replace=False)
    return np.sort(all_idx)


def run_en_uncertainty_combiner(
    df: pd.DataFrame,
    *,
    coef_active_threshold: float = 0.01,
    min_improvement: float = 0.0,
    mode: str = "clf",
    max_rows: int = 0,
) -> dict[str, Any]:
    tprint(
        "[en_uncertainty] start run_en_uncertainty_combiner "
        f"rows={len(df)} coef_active_threshold={coef_active_threshold:.4f} "
        f"min_improvement={min_improvement:.4f} max_rows={max_rows}"
    )
    mode = str(mode).lower()
    raw_full, base_col, y_col = _assemble_raw_features(df, mode=mode)
    y_full_raw = np.asarray(df[y_col].values, dtype=np.float32)
    valid = np.isfinite(y_full_raw)
    y_full_raw = y_full_raw[valid]
    raw_full = raw_full.loc[valid].reset_index(drop=True)
    if mode == "clf":
        y_full = (y_full_raw >= 0.5).astype(np.int8)
    else:
        y_full = y_full_raw

    if max_rows > 0 and len(y_full) > max_rows:
        ss_idx = _stratified_subsample(y_full if mode == "clf" else (y_full > np.nanmedian(y_full)).astype(np.int8), max_rows)
        y_search = y_full[ss_idx]
        raw_search = raw_full.iloc[ss_idx].reset_index(drop=True)
        tprint(f"[en_uncertainty] subsampled {len(y_search)}/{len(y_full)} for search (max_rows={max_rows})")
    else:
        y_search = y_full
        raw_search = raw_full

    tprint(
        f"[en_uncertainty] full rows={len(y_full)} search rows={len(y_search)} "
        f"base_col={base_col} y_col={y_col} raw_features={len(raw_full.columns)}"
    )

    base_pred_full = np.asarray(raw_full["base_pred"].values, dtype=float)
    if mode == "clf":
        base_auc = float(roc_auc_score(y_full, base_pred_full)) if len(np.unique(y_full)) > 1 else 0.5
        base_mono = _mono_top3(y_full, base_pred_full)
        base_stab = base_mono
        base_global = _global_metrics(y_full, np.clip(base_pred_full, 1e-6, 1 - 1e-6))
        base_lift = base_global["lift_top30"]
        base_j = float(0.40 * base_mono + 0.40 * base_stab + 0.20 * base_lift)
    else:
        base_auc = _rank_ic(y_full, base_pred_full)
        base_mono = _mono_top3((y_full > np.nanmedian(y_full)).astype(float), base_pred_full)
        base_stab = float(base_mono)
        base_global = _global_metrics_reg(y_full, base_pred_full)
        base_lift = 0.0
        base_j = float(0.40 * base_mono + 0.40 * base_stab + 0.20 * base_lift)
    tprint(
        "[en_uncertainty] BEFORE "
        f"mode={mode} "
        f"AUC={base_global.get('auc', base_global.get('ic_global', 0.0)):.4f} "
        f"AUC/Random={base_global.get('auc_over_random', 1.0):.3f} "
        f"Brier={base_global.get('brier', np.nan):.5f} Lift@30={base_global['lift_top30']:.4f} "
        f"IC@30={base_global['ic_top30']:.4f} ICstd@30={base_global['ic_top30_std']:.4f} "
        f"Stab@30={base_global['stability_top30']:.4f} Mono={base_global['decile_monotonicity']:.4f} "
        f"TopDecHitStd={base_global['top_decile_hit_rate_std']:.4f} ECE_top30={base_global['ece_top30']:.4f}"
    )

    feature_names = [c for c in raw_search.columns if c not in {"base_pred"}]
    reduced, pruning_hist = _stage_a_prune(
        raw_search, y_search, feature_names, coef_active_threshold=coef_active_threshold, mode=mode
    )
    tprint(
        f"[en_uncertainty] stage-a done: features {len(feature_names)} -> {len(reduced)} rounds={len(pruning_hist)}"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(raw_search.values, y_search))
    fold_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, (tr, va) in enumerate(folds):
        X_tr = raw_search.iloc[tr][reduced].to_numpy(dtype=np.float32)
        X_va = raw_search.iloc[va][reduced].to_numpy(dtype=np.float32)
        fold_cache[i] = _fit_transform_fold(X_tr, X_va)[:2]

    results: list[CandidateResult] = []
    for alpha in ALPHA_GRID:
        for l1 in L1_RATIO_GRID:
            results.append(
                _evaluate_candidate(y_search, folds, fold_cache, alpha, l1, mode=mode)
            )

    scores = np.array([r.mean_j for r in results], dtype=float)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    se = float(np.std(scores) / max(math.sqrt(len(scores)), 1.0))
    contenders = [r for r in results if r.mean_j >= best_score - se]

    def _model_size(cols: list[str]) -> int:
        inter = sum(1 for c in cols if "_x_" in c)
        main = len(cols) - inter
        return int(main + 2 * inter)

    safe_contenders = [
        r for r in contenders
        if r.mean_mono >= base_mono and r.mean_stab >= base_stab
    ]
    if safe_contenders:
        safe_contenders = sorted(safe_contenders, key=lambda r: (-r.mean_j, _model_size(reduced)))
        chosen = safe_contenders[0]
    else:
        contenders = sorted(
            contenders,
            key=lambda r: (
                _model_size(reduced),
                -r.mean_j,
                -r.mean_mono,
                -r.mean_auc,
            ),
        )
        chosen = contenders[0] if contenders else results[best_idx]
    tprint(
        "[en_uncertainty] stage-b chosen "
        f"alpha={chosen.alpha:.4f} l1_ratio={chosen.l1_ratio:.4f} "
        f"mean_J={chosen.mean_j:.5f} mono={chosen.mean_mono:.5f} auc={chosen.mean_auc:.5f} "
        f"from_safe_pool={bool(safe_contenders)}"
    )

    guard_monotonic = chosen.mean_mono >= base_mono
    guard_stability = chosen.mean_stab >= base_stab
    guard_j = chosen.mean_j >= base_j + float(min_improvement)
    improved = bool(guard_j and guard_monotonic and guard_stability)
    tprint(
        "[en_uncertainty] guardrails "
        f"J={guard_j} mono={guard_monotonic} stability={guard_stability} "
        f"-> improved={improved}"
    )
    if improved:
        tprint(f"[en_uncertainty] refitting 5-fold OOF on full data ({len(y_full)} rows)")
        s_oof, p_oof, fold_tf, final_bundle = _fit_5fold_oof(
            raw_full, y_full, reduced, chosen.alpha, chosen.l1_ratio, mode=mode
        )
    else:
        if mode == "clf":
            s_oof = _logit(raw_full["base_pred"].values)
            p_oof = np.asarray(raw_full["base_pred"].values, dtype=np.float32)
        else:
            s_oof = np.asarray(raw_full["base_pred"].values, dtype=np.float32)
            p_oof = np.asarray(raw_full["base_pred"].values, dtype=np.float32)
        fold_tf = []
        final_bundle = {}

    out_df = pd.DataFrame(index=np.flatnonzero(valid))
    out_df["s_final_oof"] = s_oof
    if mode == "clf":
        out_df["p_final_oof"] = p_oof
        final_global = _global_metrics(y_full, np.clip(p_oof, 1e-6, 1 - 1e-6))
    else:
        out_df["yhat_final_oof"] = p_oof
        final_global = _global_metrics_reg(y_full, p_oof)
    rank_corr = _rank_ic(base_pred_full, p_oof)
    base_top_dec = np.zeros(len(y_full), dtype=bool)
    final_top_dec = np.zeros(len(y_full), dtype=bool)
    k10 = max(1, int(np.ceil(0.10 * len(y_full))))
    base_top_dec[np.argsort(-base_pred_full)[:k10]] = True
    final_top_dec[np.argsort(-p_oof)[:k10]] = True
    entering = float(np.mean((~base_top_dec) & final_top_dec))
    leaving = float(np.mean(base_top_dec & (~final_top_dec)))
    tprint(
        "[en_uncertainty] AFTER "
        f"mode={mode} "
        f"AUC={final_global.get('auc', final_global.get('ic_global', 0.0)):.4f} "
        f"AUC/Random={final_global.get('auc_over_random', 1.0):.3f} "
        f"Brier={final_global.get('brier', np.nan):.5f} Lift@30={final_global['lift_top30']:.4f} "
        f"IC@30={final_global['ic_top30']:.4f} ICstd@30={final_global['ic_top30_std']:.4f} "
        f"Stab@30={final_global['stability_top30']:.4f} Mono={final_global['decile_monotonicity']:.4f} "
        f"TopDecHitStd={final_global['top_decile_hit_rate_std']:.4f} ECE_top30={final_global['ece_top30']:.4f} "
        f"Spearman(base,final)={rank_corr:.4f} EnterTop10={entering:.4f} LeaveTop10={leaving:.4f}"
    )

    return {
        "base_col": base_col,
        "label_col": y_col,
        "raw_feature_table": raw_full,
        "selected_features": reduced,
        "selected_hyperparams": {
            "alpha": float(chosen.alpha),
            "l1_ratio": float(chosen.l1_ratio),
            "C": float(_alpha_to_c(chosen.alpha)),
            "mode": mode,
        },
        "pruning_history": pruning_hist,
        "candidate_metrics": [r.__dict__ for r in results],
        "comparison": {
            "base_j": base_j,
            "base_mono": base_mono,
            "base_auc": base_auc,
            "final_j": float(chosen.mean_j),
            "final_mono": float(chosen.mean_mono),
            "final_auc": float(chosen.mean_auc),
            "improved": improved,
            "guardrails": {
                "j": bool(guard_j),
                "monotonicity": bool(guard_monotonic),
                "stability": bool(guard_stability),
            },
            "before_global": base_global,
            "after_global": final_global,
            "spearman_base_final": float(rank_corr),
            "boundary_migration": {
                "pct_entering_top_decile": entering,
                "pct_leaving_top_decile": leaving,
            },
        },
        "oof_outputs": out_df,
        "fold_transforms": fold_tf,
        "final_model_bundle": final_bundle,
    }


def run_on_artifacts(data_root: str, run_id: str, max_rows: int = 0) -> dict[str, Any]:
    oof_dir = Path(data_root) / "artifacts" / run_id / "oof"
    labels_dir = Path(data_root) / "artifacts" / run_id / "labels"
    out_dir = Path(data_root) / "artifacts" / run_id / "en_uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"processed": [], "skipped": []}
    tprint(
        f"[en_uncertainty] run_on_artifacts start: run_id={run_id} oof_dir={oof_dir}"
    )

    def _load_label_frame_for_oof(stem: str) -> pd.DataFrame | None:
        # oof_{strategy}_H{h} -> train_{strategy}_{h}
        m = re.match(r"^oof_(.+)_H(\d+)$", stem)
        if not m:
            return None
        strategy_id = m.group(1)
        horizon = int(m.group(2))
        label_path = labels_dir / f"train_{strategy_id}_{horizon}.parquet"
        if not label_path.exists():
            return None
        try:
            return pd.read_parquet(label_path)
        except Exception:
            return None

    def _attach_target_if_missing(
        df: pd.DataFrame, stem: str, mode: str
    ) -> pd.DataFrame:
        if mode == "reg":
            if any(
                c in df.columns for c in ["y_ret", "__y_ret__", "return", "target_reg"]
            ):
                return df
        elif "y_bin" in df.columns:
            return df
        label_df = _load_label_frame_for_oof(stem)
        if label_df is None:
            return df
        out = df.copy()
        if mode == "reg":
            if "__y_ret__" in label_df.columns:
                yv = np.asarray(label_df["__y_ret__"].values, dtype=np.float32)
                t_col = "y_ret"
            elif "y_ret" in label_df.columns:
                yv = np.asarray(label_df["y_ret"].values, dtype=np.float32)
                t_col = "y_ret"
            elif "return" in label_df.columns:
                yv = np.asarray(label_df["return"].values, dtype=np.float32)
                t_col = "y_ret"
            else:
                return df
        else:
            if "__y_bin__" in label_df.columns:
                yv = np.asarray(label_df["__y_bin__"].values, dtype=np.float32)
            elif "y_bin" in label_df.columns:
                yv = np.asarray(label_df["y_bin"].values, dtype=np.float32)
            else:
                return df
            t_col = "y_bin"
        n = min(len(out), len(yv))
        out[t_col] = np.nan
        if n > 0:
            out.loc[: n - 1, t_col] = yv[:n]
        return out

    for p in sorted(oof_dir.glob("oof_*.parquet")):
        try:
            df = pd.read_parquet(p)
            _mode = (
                "clf"
                if "oof_prob" in df.columns
                else ("reg" if "oof_pred" in df.columns else "skip")
            )
            if _mode == "skip":
                summary["skipped"].append(
                    {"file": p.name, "reason": "not_supported_oof"}
                )
                continue
            df = _attach_target_if_missing(df, p.stem, _mode)
            if _mode == "clf" and "y_bin" not in df.columns:
                summary["skipped"].append(
                    {"file": p.name, "reason": "missing_y_bin", "mode": _mode}
                )
                continue
            if _mode == "reg" and not any(
                c in df.columns for c in ["y_ret", "__y_ret__", "return", "target_reg"]
            ):
                summary["skipped"].append(
                    {"file": p.name, "reason": "missing_y_ret", "mode": _mode}
                )
                continue
            res = run_en_uncertainty_combiner(df, mode=_mode, max_rows=max_rows)
            oof_cols = res["oof_outputs"]
            df2 = df.copy()
            df2.loc[oof_cols.index, "s_final_oof"] = oof_cols["s_final_oof"].values
            if "p_final_oof" in oof_cols.columns:
                df2.loc[oof_cols.index, "p_final_oof"] = oof_cols["p_final_oof"].values
            if "yhat_final_oof" in oof_cols.columns:
                df2.loc[oof_cols.index, "yhat_final_oof"] = oof_cols[
                    "yhat_final_oof"
                ].values
            for c in ["base_logit", "abs_base_logit"]:
                if c in res["raw_feature_table"].columns:
                    df2.loc[oof_cols.index, c] = res["raw_feature_table"][c].values
            df2.to_parquet(p, index=False)

            stem = p.stem.replace("oof_", "")
            metrics_path = out_dir / f"{stem}_metrics.json"
            model_path = out_dir / f"{stem}_model.pkl"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "selected_features": res["selected_features"],
                        "selected_hyperparams": res["selected_hyperparams"],
                        "pruning_history": res["pruning_history"],
                        "candidate_metrics": res["candidate_metrics"],
                        "comparison": res["comparison"],
                    },
                    f,
                    indent=2,
                )
            if res.get("final_model_bundle"):
                with open(model_path, "wb") as f:
                    pickle.dump(res["final_model_bundle"], f)
            processing_contract_path = out_dir / f"{stem}_processing_contract.json"
            with open(processing_contract_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "alpha_to_c_mapping": (
                            "C = 1 / alpha" if _mode == "clf" else "not_applicable"
                        ),
                        "mode": _mode,
                        "winsorization": "1-99 percentile per fold on train split only",
                        "standardization": "z-score per fold on train split only",
                        "uncertainty_direction_rule": "higher = more uncertain; vote_margin/vote_top_gap/leaf_support flipped",
                        "required_outputs": (
                            ["s_final_oof", "p_final_oof"]
                            if _mode == "clf"
                            else ["s_final_oof", "yhat_final_oof"]
                        ),
                        "live_inference_preprocess_keys": [
                            "features",
                            "lower",
                            "upper",
                            "mean",
                            "std",
                            "alpha",
                            "l1_ratio",
                            "C",
                        ],
                    },
                    f,
                    indent=2,
                )
            summary["processed"].append(
                {
                    "file": p.name,
                    "metrics": str(metrics_path),
                    "model": str(model_path),
                    "processing_contract": str(processing_contract_path),
                }
            )
            tprint(f"EN uncertainty combiner: processed {p.name}")
        except Exception as exc:
            summary["skipped"].append({"file": p.name, "reason": str(exc)})
            tprint(f"EN uncertainty combiner: skipped {p.name} ({exc})")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ElasticNet-based uncertainty OOF combiner"
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=15000)
    args = parser.parse_args()
    summary = run_on_artifacts(args.data_root, args.run_id, max_rows=args.max_rows)
    tprint(
        f"EN uncertainty combiner done: processed={len(summary['processed'])} skipped={len(summary['skipped'])}"
    )


if __name__ == "__main__":
    main()
