from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from .utils import tprint

EPS = 1e-6
BLEND_GRID = np.asarray(
    [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50], dtype=np.float32
)


@dataclass
class EBMUncertaintyState:
    feature_names: list[str]
    bin_edges: dict[str, np.ndarray]
    bin_counts: dict[str, np.ndarray]
    feature_medians: dict[str, float]


@dataclass
class ENUncertaintyAdjuster:
    feature_names: list[str]
    alpha: float
    l1_ratio: float
    blend: float
    scaler: StandardScaler
    model: ElasticNet
    metrics: dict[str, float]

    def residual(self, features: pd.DataFrame) -> np.ndarray:
        X = _prepare_feature_frame(features, self.feature_names)
        Xs = self.scaler.transform(X).astype(np.float32, copy=False)
        out = np.asarray(self.model.predict(Xs), dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def predict(self, base_pred: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        residual = self.residual(features)
        return apply_en_prediction(base_pred, residual, self.blend)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(np.asarray(x, dtype=np.float64)).astype(np.float32)


def safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)).astype(
        np.float32
    )


def rank_norm(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.zeros(len(arr), dtype=np.float32)
    mask = np.isfinite(arr)
    if int(np.sum(mask)) <= 1:
        return out
    out[mask] = (rankdata(arr[mask], method="average") - 1.0) / max(
        int(np.sum(mask)) - 1, 1
    )
    return out


def initial_rank_weights(base_pred: np.ndarray) -> np.ndarray:
    rn = rank_norm(base_pred)
    w = np.clip(5.0 * np.sqrt(0.01 + rn), 0.5, 5.0)
    return (w / max(float(np.mean(w)), EPS)).clip(0.5, 5.0).astype(np.float32)


def update_rank_weights(
    previous_weight: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    prev = np.asarray(previous_weight, dtype=np.float32).reshape(-1)
    new = prev * np.clip(5.0 * np.sqrt(0.01 + rank_norm(prediction)), 0.5, 5.0)
    new = new / max(float(np.mean(new)), EPS)
    return np.clip(new, 0.5, 5.0).astype(np.float32)


def fit_uncertainty_state(
    X: pd.DataFrame,
    feature_names: list[str],
    max_bins: int = 64,
) -> EBMUncertaintyState:
    X_df = _prepare_feature_frame(X, feature_names)
    bin_edges: dict[str, np.ndarray] = {}
    bin_counts: dict[str, np.ndarray] = {}
    feature_medians: dict[str, float] = {}
    qs = np.linspace(0.0, 1.0, int(max_bins) + 1)
    for name in feature_names:
        vals = np.asarray(X_df[name], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            edges = np.asarray([-1.0, 1.0], dtype=np.float32)
            counts = np.asarray([len(X_df)], dtype=np.float32)
            median = 0.0
        else:
            median = float(np.nanmedian(vals))
            edges = np.unique(np.nanquantile(vals, qs)).astype(np.float32)
            if edges.size < 2:
                center = float(edges[0]) if edges.size else median
                edges = np.asarray([center - 0.5, center + 0.5], dtype=np.float32)
            counts, _ = np.histogram(
                np.asarray(X_df[name], dtype=np.float64), bins=edges
            )
            counts = np.maximum(counts.astype(np.float32), 1.0)
        bin_edges[name] = edges.astype(np.float32, copy=False)
        bin_counts[name] = counts.astype(np.float32, copy=False)
        feature_medians[name] = median
    return EBMUncertaintyState(
        feature_names=list(feature_names),
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        feature_medians=feature_medians,
    )


def compute_uncertainty_features(
    X: pd.DataFrame,
    models: list[Any],
    mode: str,
    raw_predict_fn: Callable[[Any, pd.DataFrame, str], np.ndarray],
    state: EBMUncertaintyState | None = None,
) -> pd.DataFrame:
    base_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_df = _prepare_feature_frame(base_df, list(base_df.columns))
    n = len(X_df)
    if not models:
        return _empty_features(n)

    contribs: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for model in models:
        terms_fn = getattr(model, "eval_terms", None)
        if terms_fn is not None:
            try:
                terms = np.asarray(
                    terms_fn(_coerce_ebm_feature_types(X_df)), dtype=np.float32
                )
                if terms.ndim == 2 and terms.shape[0] == n:
                    contribs.append(
                        np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
                    )
            except Exception:
                pass
        try:
            raw = np.asarray(
                raw_predict_fn(model, X_df, mode), dtype=np.float32
            ).reshape(-1)
            if raw.shape[0] == n:
                if mode == "classifier":
                    raw = safe_logit(raw)
                logits.append(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception:
            pass

    if not logits:
        return _empty_features(n)
    L = _align_stack(logits)
    if not contribs:
        C = np.zeros((L.shape[0], n, 1), dtype=np.float32)
        term_features: list[tuple[int, ...]] = [(0,)]
    else:
        C = _align_term_stack(contribs)
        term_features = _term_features(models[0], C.shape[2])

    C_mean = np.mean(C, axis=0).astype(np.float32)
    abs_c = np.abs(C_mean)
    mas = np.sum(abs_c, axis=1).astype(np.float32)
    total_mean = np.sum(C_mean, axis=1).astype(np.float32)
    term_var = np.std(C, axis=0).astype(np.float32)
    p_bag = sigmoid(L) if mode == "classifier" else sigmoid(L)
    entropy = _binary_entropy(p_bag)
    main_mask = np.asarray(
        [len(t) == 1 for t in term_features[: C.shape[2]]], dtype=bool
    )
    if main_mask.size != C.shape[2]:
        main_mask = np.ones(C.shape[2], dtype=bool)

    main_sum = np.sum(C_mean[:, main_mask], axis=1) if np.any(main_mask) else total_mean
    inter_sum = (
        np.sum(C_mean[:, ~main_mask], axis=1) if np.any(~main_mask) else np.zeros(n)
    ).astype(np.float32)

    support_mean, support_min, proximity_mean, proximity_min = _support_proximity(
        X_df, term_features, C.shape[2], state
    )
    grad_mean, grad_max = _local_gradients(X_df, models, term_features, C_mean, 0.80)
    top50_sign_ratio = _sign_ratio_top_mas(C_mean, abs_c, 0.50)

    logodds_std = np.std(L, axis=0).astype(np.float32)
    pi_width = (np.percentile(L, 90, axis=0) - np.percentile(L, 10, axis=0)).astype(
        np.float32
    )
    gap50rel = _gap50_relative(L)
    conflict = (np.sum(abs_c, axis=1) - np.abs(total_mean)).astype(np.float32)
    conflict_norm = conflict / (mas + EPS)
    concentration = (np.max(abs_c, axis=1) / (mas + EPS)).astype(np.float32)
    sign_ratio = (
        np.abs(np.sum(np.sign(C_mean), axis=1)) / max(float(C_mean.shape[1]), 1.0)
    ).astype(np.float32)
    interaction_share = (
        np.abs(inter_sum) / (np.abs(main_sum) + np.abs(inter_sum) + EPS)
    ).astype(np.float32)
    consensus_edge = (np.abs(np.median(L, axis=0)) / (np.abs(gap50rel) + EPS)).astype(
        np.float32
    )
    uncertainty_weight = (
        1.0 / (1.0 + logodds_std) / (1.0 + pi_width) / (1.0 + np.maximum(gap50rel, 0.0))
    ).astype(np.float32)
    friction_weight = (1.0 - np.clip(conflict_norm, 0.0, 1.0)).astype(np.float32)
    grad_damped = (np.mean(L, axis=0) / (1.0 + np.abs(grad_mean))).astype(np.float32)
    convexity_boosted = (
        np.mean(L, axis=0) * friction_weight * uncertainty_weight
    ).astype(np.float32)
    edge_case_vol = (logodds_std / (proximity_min + EPS)).astype(np.float32)
    snr = (support_mean * top50_sign_ratio).astype(np.float32)
    support_adj_var = (logodds_std / np.sqrt(np.maximum(support_mean, 1.0))).astype(
        np.float32
    )
    support_adj_unc = (
        logodds_std / (1.0 + np.log1p(np.maximum(support_mean, 0.0)))
    ).astype(np.float32)

    data = {
        "ebm_unc_term_var_mean": np.mean(term_var, axis=1),
        "ebm_unc_term_var_max": np.max(term_var, axis=1),
        "ebm_unc_logodds_var": logodds_std,
        "ebm_unc_support_mean": support_mean,
        "ebm_unc_support_min": support_min,
        "ebm_unc_conflict": conflict,
        "ebm_unc_conflict_norm": conflict_norm,
        "ebm_unc_proximity_mean": proximity_mean,
        "ebm_unc_proximity_min": proximity_min,
        "ebm_unc_concentration": concentration,
        "ebm_unc_sign_ratio": sign_ratio,
        "ebm_unc_sign_ratio_top50_mas": top50_sign_ratio,
        "ebm_unc_gradient_mean_top80_mas": grad_mean,
        "ebm_unc_gradient_max_top80_mas": grad_max,
        "ebm_unc_entropy_mean": np.mean(entropy, axis=0),
        "ebm_unc_entropy_std": np.std(entropy, axis=0),
        "ebm_unc_interaction_share": interaction_share,
        "ebm_unc_gap50rel": gap50rel,
        "ebm_unc_consensus_edge": consensus_edge,
        "ebm_unc_pi_width": pi_width,
        "ebm_unc_gradient_damped_logit": grad_damped,
        "ebm_unc_convexity_boosted_logit": convexity_boosted,
        "ebm_unc_chaos_factor": conflict * np.mean(entropy, axis=0),
        "ebm_unc_edge_case_volatility": edge_case_vol,
        "ebm_unc_signal_to_noise": snr,
        "ebm_unc_support_adjusted_variance": support_adj_var,
        "ebm_unc_support_adjusted_uncertainty": support_adj_unc,
        "ebm_unc_uncertainty_weight": uncertainty_weight,
        "ebm_unc_friction_weight": friction_weight,
    }
    return pd.DataFrame(
        {
            k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for k, v in data.items()
        },
        index=X_df.index,
    )


def fit_en_uncertainty_adjuster(
    base_pred: np.ndarray,
    y: np.ndarray,
    features: pd.DataFrame,
    returns: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    random_state: int = 42,
    n_trials: int = 30,
) -> ENUncertaintyAdjuster | None:
    p = np.asarray(base_pred, dtype=np.float32).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    n = min(len(p), len(y_arr), len(features))
    if n < 50:
        return None
    p = np.clip(p[:n], EPS, 1.0 - EPS).astype(np.float32)
    y_arr = y_arr[:n].astype(np.float32)
    X = _prepare_feature_frame(features.iloc[:n], list(features.columns))
    feature_names = list(X.columns)
    fit_idx, blend_idx = _fit_blend_split(y_arr, random_state)
    if len(fit_idx) < 20 or len(blend_idx) < 10:
        return None

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X.iloc[fit_idx]).astype(np.float32, copy=False)
    X_blend = scaler.transform(X.iloc[blend_idx]).astype(np.float32, copy=False)
    target = (y_arr - p).astype(np.float32)
    current_weights = initial_rank_weights(p[fit_idx])
    best_model: ElasticNet | None = None
    best_score = -np.inf
    best_alpha = 1.0
    best_l1 = 0.7
    best_blend = 0.0
    best_metrics: dict[str, float] = {}
    best_residual_std = 0.0
    best_coef_nnz = 0
    base_blend_metric = _selection_metrics(
        y_arr[blend_idx],
        p[blend_idx],
        returns=None if returns is None else np.asarray(returns)[:n][blend_idx],
        groups=None if groups is None else np.asarray(groups)[:n][blend_idx],
    )

    def _run_trial(alpha: float, l1_ratio: float) -> float:
        nonlocal current_weights
        nonlocal best_alpha, best_blend, best_l1, best_metrics, best_model, best_score
        nonlocal best_coef_nnz, best_residual_std
        mdl = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            max_iter=3000,
            selection="cyclic",
            random_state=random_state,
        )
        with warnings.catch_warnings():
            if float(alpha) <= 0.0:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message=".*alpha=0.*")
            mdl.fit(X_fit, target[fit_idx], sample_weight=current_weights)
        train_resid = np.asarray(mdl.predict(X_fit), dtype=np.float32)
        current_weights = update_rank_weights(
            current_weights, apply_en_prediction(p[fit_idx], train_resid, 1.0)
        )
        blend_resid = np.asarray(mdl.predict(X_blend), dtype=np.float32)
        trial_best = -np.inf
        for blend in BLEND_GRID:
            pred = apply_en_prediction(p[blend_idx], blend_resid, float(blend))
            metric = _selection_metrics(
                y_arr[blend_idx],
                pred,
                returns=None if returns is None else np.asarray(returns)[:n][blend_idx],
                groups=None if groups is None else np.asarray(groups)[:n][blend_idx],
            )
            lift_delta = float(metric["precision30"] - base_blend_metric["precision30"])
            stability_delta = float(
                metric["stability30"] - base_blend_metric["stability30"]
            )
            return10_delta = float(
                metric["mean_gross_return10"] - base_blend_metric["mean_gross_return10"]
            )
            score = 0.4 * lift_delta + 0.4 * stability_delta + 0.2 * return10_delta
            gates_ok = (
                metric["precision30"] + 1e-6 >= 0.95 * base_blend_metric["precision30"]
                and metric["stability30"] + 1e-6
                >= 0.95 * base_blend_metric["stability30"]
            )
            if (
                gates_ok
                and _guardrails_ok(y_arr[blend_idx], p[blend_idx], pred)
                and score > best_score
            ):
                best_score = float(score)
                best_model = mdl
                best_alpha = float(alpha)
                best_l1 = float(l1_ratio)
                best_blend = float(blend)
                best_metrics = dict(metric)
                best_metrics["objective"] = float(score)
                best_metrics["lift30_delta"] = float(lift_delta)
                best_metrics["stability30_delta"] = float(stability_delta)
                best_metrics["mean_gross_return10_delta"] = float(return10_delta)
                best_metrics["base_precision30"] = float(
                    base_blend_metric["precision30"]
                )
                best_metrics["base_stability30"] = float(
                    base_blend_metric["stability30"]
                )
                best_metrics["base_mean_gross_return10"] = float(
                    base_blend_metric["mean_gross_return10"]
                )
                best_residual_std = float(np.nanstd(blend_resid))
                best_coef_nnz = int(np.sum(np.abs(mdl.coef_) > 1e-12))
            trial_best = max(trial_best, float(score))
        return float(trial_best if np.isfinite(trial_best) else -1e9)

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        def objective(trial: Any) -> float:
            alpha = float(
                np.round(trial.suggest_float("alpha", 0.0, 1.0) * 10.0) / 10.0
            )
            l1_ratio = float(
                np.round(trial.suggest_float("l1_ratio", 0.4, 0.9) * 10.0) / 10.0
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
            l1_ratio = float(np.clip(l1_ratio, 0.4, 0.9))
            trial.set_user_attr("alpha_rounded", alpha)
            trial.set_user_attr("l1_ratio_rounded", l1_ratio)
            return _run_trial(alpha, l1_ratio)

        for alpha, l1_ratio in (
            (1.0, 0.7),
            (0.0, 0.4),
            (0.0, 0.7),
            (0.0, 0.9),
            (0.1, 0.4),
            (0.1, 0.7),
            (0.1, 0.9),
            (0.2, 0.4),
        ):
            study.enqueue_trial({"alpha": alpha, "l1_ratio": l1_ratio})
        study.optimize(objective, n_trials=max(int(n_trials), 1))
    except Exception:
        for alpha, l1_ratio in _trial_grid(random_state, n_trials):
            _run_trial(alpha, l1_ratio)
    if best_model is None:
        return None
    full_base_metric = _selection_metrics(
        y_arr,
        p,
        returns=None if returns is None else np.asarray(returns)[:n],
        groups=None if groups is None else np.asarray(groups)[:n],
    )
    full_resid = np.asarray(
        best_model.predict(scaler.transform(X).astype(np.float32, copy=False)),
        dtype=np.float32,
    )
    full_best_blend = 0.0
    full_best_score = -np.inf
    full_best_metrics: dict[str, float] = {}
    for blend in BLEND_GRID:
        full_pred = apply_en_prediction(p, full_resid, float(blend))
        metric = _selection_metrics(
            y_arr,
            full_pred,
            returns=None if returns is None else np.asarray(returns)[:n],
            groups=None if groups is None else np.asarray(groups)[:n],
        )
        lift_delta = float(metric["precision30"] - full_base_metric["precision30"])
        stability_delta = float(
            metric["stability30"] - full_base_metric["stability30"]
        )
        return10_delta = float(
            metric["mean_gross_return10"]
            - full_base_metric["mean_gross_return10"]
        )
        score = 0.4 * lift_delta + 0.4 * stability_delta + 0.2 * return10_delta
        gates_ok = (
            metric["precision30"] + 1e-6 >= 0.95 * full_base_metric["precision30"]
            and metric["stability30"] + 1e-6
            >= 0.95 * full_base_metric["stability30"]
            and _guardrails_ok(y_arr, p, full_pred)
        )
        if gates_ok and score > full_best_score:
            full_best_score = float(score)
            full_best_blend = float(blend)
            full_best_metrics = dict(metric)
            full_best_metrics["objective"] = float(score)
            full_best_metrics["lift30_delta"] = float(lift_delta)
            full_best_metrics["stability30_delta"] = float(stability_delta)
            full_best_metrics["mean_gross_return10_delta"] = float(return10_delta)
    if not full_best_metrics:
        full_best_metrics = dict(full_base_metric)
        full_best_metrics["objective"] = 0.0
        full_best_metrics["lift30_delta"] = 0.0
        full_best_metrics["stability30_delta"] = 0.0
        full_best_metrics["mean_gross_return10_delta"] = 0.0
    full_best_metrics["base_precision30"] = float(full_base_metric["precision30"])
    full_best_metrics["base_stability30"] = float(full_base_metric["stability30"])
    full_best_metrics["base_mean_gross_return10"] = float(
        full_base_metric["mean_gross_return10"]
    )
    if abs(full_best_blend - best_blend) > 1e-9:
        tprint(
            "EBM uncertainty EN: full-OOF hard gate changed blend "
            f"{best_blend:.2f} -> {full_best_blend:.2f} "
            f"(lift30 {full_base_metric['precision30']:.4f} -> "
            f"{full_best_metrics['precision30']:.4f}, "
            f"stability30 {full_base_metric['stability30']:.4f} -> "
            f"{full_best_metrics['stability30']:.4f})."
        )
    best_blend = float(full_best_blend)
    best_score = float(full_best_metrics["objective"])
    best_metrics.update(full_best_metrics)
    tprint(
        "EBM uncertainty EN: "
        f"alpha={best_alpha:.2f}, l1_ratio={best_l1:.2f}, blend={best_blend:.2f}, "
        f"objective={best_score:.4f}, residual_std={best_residual_std:.6f}, "
        f"coef_nnz={best_coef_nnz}."
    )
    best_metrics["residual_std"] = float(best_residual_std)
    best_metrics["coef_nnz"] = float(best_coef_nnz)
    return ENUncertaintyAdjuster(
        feature_names=feature_names,
        alpha=best_alpha,
        l1_ratio=best_l1,
        blend=best_blend,
        scaler=scaler,
        model=best_model,
        metrics=best_metrics,
    )


def apply_en_prediction(
    base_pred: np.ndarray, en_pred: np.ndarray, blend: float
) -> np.ndarray:
    z = safe_logit(base_pred) + float(blend) * np.asarray(en_pred, dtype=np.float32)
    return np.clip(sigmoid(z), 1e-4, 1.0 - 1e-4).astype(np.float32)


def uncertainty_weighted_prediction(
    base_pred: np.ndarray, features: pd.DataFrame, en_pred: np.ndarray | None = None
) -> np.ndarray:
    p = np.asarray(en_pred if en_pred is not None else base_pred, dtype=np.float32)
    if "ebm_unc_uncertainty_weight" in features:
        w = np.asarray(features["ebm_unc_uncertainty_weight"], dtype=np.float32)
    else:
        w = np.ones(len(p), dtype=np.float32)
    centered = (p - 0.5) * np.clip(w, 0.0, 1.0)
    return np.clip(0.5 + centered, 1e-4, 1.0 - 1e-4).astype(np.float32)


def _prepare_feature_frame(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X_df.columns = [str(c) for c in X_df.columns]
    for col in columns:
        if col not in X_df:
            X_df[col] = 0.0
    out = X_df.reindex(columns=columns, fill_value=0.0)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def _coerce_ebm_feature_types(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )
    return out


def _align_stack(items: list[np.ndarray]) -> np.ndarray:
    min_n = min(len(x) for x in items)
    return np.vstack([np.asarray(x[:min_n], dtype=np.float32) for x in items])


def _align_term_stack(items: list[np.ndarray]) -> np.ndarray:
    min_n = min(x.shape[0] for x in items)
    min_t = min(x.shape[1] for x in items)
    return np.stack(
        [x[:min_n, :min_t].astype(np.float32, copy=False) for x in items], axis=0
    )


def _term_features(model: Any, n_terms: int) -> list[tuple[int, ...]]:
    tf = getattr(model, "term_features_", None)
    if tf is None:
        return [(i,) for i in range(n_terms)]
    out: list[tuple[int, ...]] = []
    for term in list(tf)[:n_terms]:
        try:
            out.append(tuple(int(i) for i in term))
        except Exception:
            out.append((len(out),))
    while len(out) < n_terms:
        out.append((len(out),))
    return out


def _empty_features(n: int) -> pd.DataFrame:
    zeros = np.zeros(n, dtype=np.float32)
    return pd.DataFrame(
        {
            "ebm_unc_logodds_var": zeros,
            "ebm_unc_entropy_mean": zeros,
            "ebm_unc_uncertainty_weight": np.ones(n, dtype=np.float32),
            "ebm_unc_conflict": zeros,
            "ebm_unc_support_mean": np.ones(n, dtype=np.float32),
        }
    )


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    pp = np.clip(np.asarray(p, dtype=np.float32), EPS, 1.0 - EPS)
    return (-(pp * np.log(pp) + (1.0 - pp) * np.log(1.0 - pp))).astype(np.float32)


def _support_proximity(
    X: pd.DataFrame,
    term_features: list[tuple[int, ...]],
    n_terms: int,
    state: EBMUncertaintyState | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    support_cols: list[np.ndarray] = []
    prox_cols: list[np.ndarray] = []
    if state is None:
        return (
            np.ones(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        )
    columns = list(X.columns)
    for term in term_features[:n_terms]:
        if len(term) != 1:
            continue
        idx = int(term[0])
        if idx < 0 or idx >= len(columns):
            continue
        name = columns[idx]
        edges = state.bin_edges.get(name)
        counts = state.bin_counts.get(name)
        if edges is None or counts is None or len(edges) < 2:
            continue
        vals = np.asarray(X[name], dtype=np.float32)
        bin_id = np.searchsorted(edges, vals, side="right") - 1
        bin_id = np.clip(bin_id, 0, len(edges) - 2)
        support_cols.append(counts[np.clip(bin_id, 0, len(counts) - 1)])
        left = edges[bin_id]
        right = edges[bin_id + 1]
        width = np.maximum(right - left, EPS)
        prox = np.minimum(np.abs(vals - left), np.abs(right - vals)) / width
        prox_cols.append(np.clip(prox, 0.0, 1.0).astype(np.float32))
    if not support_cols:
        ones = np.ones(n, dtype=np.float32)
        return ones, ones, ones, ones
    support = np.column_stack(support_cols).astype(np.float32)
    prox = np.column_stack(prox_cols).astype(np.float32)
    return (
        np.mean(support, axis=1).astype(np.float32),
        np.min(support, axis=1).astype(np.float32),
        np.mean(prox, axis=1).astype(np.float32),
        np.min(prox, axis=1).astype(np.float32),
    )


def _local_gradients(
    X: pd.DataFrame,
    models: list[Any],
    term_features: list[tuple[int, ...]],
    C_mean: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not models:
        z = np.zeros(len(X), dtype=np.float32)
        return z, z
    scores = getattr(models[0], "term_scores_", None)
    if scores is None:
        z = np.zeros(len(X), dtype=np.float32)
        return z, z
    columns = list(X.columns)
    grads: list[np.ndarray] = []
    grad_term_ids: list[int] = []
    for ti, term in enumerate(term_features[: C_mean.shape[1]]):
        if len(term) != 1 or ti >= len(scores):
            continue
        idx = int(term[0])
        if idx < 0 or idx >= len(columns):
            continue
        arr = np.asarray(scores[ti], dtype=np.float32).reshape(-1)
        if arr.size < 3:
            continue
        vals = np.asarray(X[columns[idx]], dtype=np.float32)
        ranks = rank_norm(vals)
        bid = np.clip((ranks * (arr.size - 1)).astype(np.int32), 0, arr.size - 1)
        left = arr[np.clip(bid - 1, 0, arr.size - 1)]
        right = arr[np.clip(bid + 1, 0, arr.size - 1)]
        grads.append((right - left).astype(np.float32))
        grad_term_ids.append(ti)
    if not grads:
        z = np.zeros(len(X), dtype=np.float32)
        return z, z
    G = np.column_stack(grads).astype(np.float32)
    abs_c = np.abs(C_mean[:, grad_term_ids]).astype(np.float32)
    mask = _top_mas_mask(abs_c, threshold)
    denom = np.maximum(np.sum(mask, axis=1), 1.0)
    abs_g = np.abs(G)
    return (
        (np.sum(abs_g * mask, axis=1) / denom).astype(np.float32),
        np.max(np.where(mask > 0, abs_g, 0.0), axis=1).astype(np.float32),
    )


def _top_mas_mask(abs_c: np.ndarray, threshold: float) -> np.ndarray:
    n, t = abs_c.shape
    order = np.argsort(-abs_c, axis=1)
    sorted_abs = np.take_along_axis(abs_c, order, axis=1)
    denom = np.maximum(np.sum(sorted_abs, axis=1, keepdims=True), EPS)
    cum = np.cumsum(sorted_abs, axis=1) / denom
    sorted_mask = (cum <= float(threshold)).astype(np.float32)
    sorted_mask[:, 0] = 1.0
    mask = np.zeros((n, t), dtype=np.float32)
    np.put_along_axis(mask, order, sorted_mask, axis=1)
    return mask


def _sign_ratio_top_mas(
    C_mean: np.ndarray, abs_c: np.ndarray, threshold: float
) -> np.ndarray:
    mask = _top_mas_mask(abs_c, threshold)
    denom = np.maximum(np.sum(mask, axis=1), 1.0)
    return (np.abs(np.sum(np.sign(C_mean) * mask, axis=1)) / denom).astype(np.float32)


def _gap50_relative(L: np.ndarray) -> np.ndarray:
    k = L.shape[0]
    if k < 2:
        return np.zeros(L.shape[1], dtype=np.float32)
    ordered = np.sort(np.asarray(L, dtype=np.float32), axis=0)
    half = max(1, k // 2)
    bot_med = np.median(ordered[:half, :], axis=0).astype(np.float32)
    top_med = np.median(ordered[-half:, :], axis=0).astype(np.float32)
    return ((top_med - bot_med) / (np.abs(top_med) + np.abs(bot_med) + EPS)).astype(
        np.float32
    )


def _fit_blend_split(y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state + 7103)
    n = len(y)
    idx = np.arange(n, dtype=np.int32)
    blend_n = max(10, int(round(0.10 * n)))
    try:
        cls = (np.asarray(y) >= np.nanmedian(y)).astype(np.int8)
        blend_parts = []
        for val in np.unique(cls):
            members = idx[cls == val]
            take = max(1, int(round(blend_n * len(members) / max(n, 1))))
            blend_parts.append(
                rng.choice(members, min(take, len(members)), replace=False)
            )
        blend_idx = np.unique(np.concatenate(blend_parts)).astype(np.int32)
    except Exception:
        blend_idx = rng.choice(idx, min(blend_n, n), replace=False).astype(np.int32)
    fit_idx = np.setdiff1d(idx, blend_idx, assume_unique=False).astype(np.int32)
    return fit_idx, blend_idx


def _trial_grid(random_state: int, n_trials: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(random_state + 911)
    params: list[tuple[float, float]] = [
        (1.0, 0.7),
        (0.0, 0.4),
        (0.0, 0.7),
        (0.0, 0.9),
        (0.5, 0.8),
        (0.2, 0.6),
    ]
    while len(params) < max(int(n_trials), 1):
        alpha = float(np.round(rng.uniform(0.0, 1.0) * 10.0) / 10.0)
        l1 = float(np.round(rng.uniform(0.4, 0.9) * 10.0) / 10.0)
        params.append((float(np.clip(alpha, 0.0, 1.0)), float(np.clip(l1, 0.4, 0.9))))
    return params[: max(int(n_trials), 1)]


def _selection_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    returns: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    ret = np.asarray(returns, dtype=np.float32) if returns is not None else y_arr
    precision30 = _top_mean(y_arr, p, 0.30)
    mean_ret10 = _top_mean(ret, p, 0.10)
    stability30 = _stability_by_group(y_arr, p, groups)
    return {
        "precision30": float(precision30),
        "stability30": float(stability30),
        "mean_gross_return10": float(mean_ret10),
    }


def _top_mean(values: np.ndarray, pred: np.ndarray, frac: float) -> float:
    n = len(pred)
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(float(frac) * n)))
    idx = np.argsort(pred)[-k:]
    vals = np.asarray(values, dtype=np.float32)[idx]
    return float(np.nanmean(vals)) if vals.size else 0.0


def _stability_by_group(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray | None
) -> float:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    if len(yy) < 8:
        return 0.0
    if groups is None or len(groups) != len(pred):
        return _top30_bucket_stability(yy, pp)
    vals: list[float] = []
    g = np.asarray(groups, dtype=object)[m]
    for key in pd.unique(pd.Series(g)):
        mask = g == key
        if int(np.sum(mask)) < 20:
            continue
        y_group = yy[mask]
        p_group = pp[mask]
        vals.append(_top_mean(y_group, p_group, 0.30))
    if len(vals) < 3:
        return _top30_bucket_stability(yy, pp)
    arr = np.asarray(vals, dtype=np.float32)
    mean_v = float(np.nanmean(arr))
    std_v = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.clip(1.0 / (1.0 + std_v / (abs(mean_v) + 1e-6)), 0.0, 1.0))


def _top30_bucket_stability(y: np.ndarray, pred: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    if len(pp) < 20:
        return 0.0
    top_n = max(1, int(np.ceil(0.30 * len(pp))))
    top_idx = np.argsort(pp)[-top_n:]
    score_top = pp[top_idx]
    y_top = yy[top_idx]
    qs = np.quantile(score_top, np.linspace(0.0, 1.0, 6))
    vals: list[float] = []
    for i in range(5):
        right_mask = score_top <= qs[i + 1] if i == 4 else score_top < qs[i + 1]
        mask = (score_top >= qs[i]) & right_mask
        if np.any(mask):
            vals.append(float(np.mean(y_top[mask])))
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(1.0 / (1.0 + np.std(arr)))


def _guardrails_ok(y: np.ndarray, base_pred: np.ndarray, pred: np.ndarray) -> bool:
    base = np.asarray(base_pred, dtype=np.float32)
    adj = np.asarray(pred, dtype=np.float32)
    if float(np.nanstd(adj)) < 1e-6:
        return False
    base_prec = _top_mean(y, base, 0.30)
    adj_prec = _top_mean(y, adj, 0.30)
    return bool(adj_prec + 1e-6 >= 0.90 * base_prec)
