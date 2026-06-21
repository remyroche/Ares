"""Bounded LGBM add-on filter for newly generated regime features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None


def _lightgbm_module() -> Any:
    if not _LIGHTGBM_AVAILABLE:
        return None
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception:
        return None


@dataclass(frozen=True)
class RegimeFeatureLGBMFilterConfig:
    n_folds: int = 7
    fold_sample_fraction: float = 0.50
    random_state: int = 42
    objective_mode: str = "train_base"
    max_rows: int = 0
    max_trees: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 5
    min_child_samples: int = 50
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    use_shadow_gain: bool = True
    structural_min_trees_using: int = 10
    structural_min_tree_fraction: float = 0.01
    structural_min_leaf_path_share: float = 0.001
    structural_min_sample_path_exposure: float = 0.005
    structural_min_gain_ratio: float = 1.25
    shadow_gain_quantile: float = 0.85
    exposure_quantile: float = 0.67
    lift_positive_fold_fraction: float = 0.60
    stability_positive_fold_fraction: float = 0.60
    score_positive_fold_fraction: float = 0.60
    base_context_filter_enabled: bool = True
    interaction_path_share_min: float = 0.80
    signal_dense_interaction_share_min: float = 0.66
    signal_gain_share_min: float = 0.25
    signal_exposure_share_min: float = 0.50
    coarse_gate_context_enabled: bool = True
    coarse_gate_min_fold_fraction: float = 0.50
    coarse_gate_min_tree_fraction_of_structural: float = 0.25
    conditional_signal_uplift_enabled: bool = True
    signal_uplift_max_signal_features: int = 24
    signal_uplift_low_quantile: float = 0.33
    signal_uplift_high_quantile: float = 0.67
    signal_uplift_min_abs_mean: float = 0.05
    signal_uplift_min_abs_max: float = 0.10
    signal_uplift_min_fold_fraction: float = 0.70
    signal_uplift_top_quantile: float = 0.75
    risk_gate_min_hr_avoidance: float = 0.0
    risk_gate_min_stability_avoidance: float = 0.0
    risk_gate_min_oof_failure_lift: float = 0.0
    risk_gate_min_oof_residual_abs_lift: float = 0.0
    risk_gate_min_oof_wrong_rate_lift: float = 0.0
    risk_gate_min_negative_fold_fraction: float = 0.70
    risk_gate_min_effective_structural_pass_rate: float = 0.50
    risk_gate_min_oof_failure_positive_rate: float = 0.50
    risk_gate_require_oof_alignment: bool = True
    risk_budget_scaler_enabled: bool = True
    risk_budget_scaler_min_fold_fraction: float = 0.60
    risk_budget_scaler_min_scaled_hr_lift: float = 0.0
    risk_budget_scaler_min_high_low_hr_lift: float = 0.0
    risk_budget_scaler_min_failure_avoidance: float = 0.0
    risk_budget_scaler_quantile_bins: int = 5
    require_model_helpfulness_for_selection: bool = True
    oof_failure_top_quantile: float = 0.67
    redundancy_spearman_threshold: float = 0.95
    source_cumulative_usefulness_keep: float = 0.80
    top_n: int = 30
    min_group_rows: int = 10
    route_max_rows: int = 5000
    stratified_period_bins: int = 24


@dataclass(frozen=True)
class RegimeFeatureLGBMFilterResult:
    selected_features: list[str]
    feature_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    source_metrics: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _is_train_base(objective_mode: str | None) -> bool:
    return str(objective_mode or "").strip().lower() in {"", "train_base", "base"}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def extract_lgbm_reuse_contract(
    artifact: Any,
    *,
    stage: str = "train_base",
) -> dict[str, Any]:
    """Best-effort extraction of frozen features and params from a saved model artifact."""

    if artifact is None:
        return {"selected_features": [], "params": {}, "stage": str(stage), "source": "none"}

    def iter_candidates(value: Any, depth: int = 0) -> list[Any]:
        if value is None or depth > 6:
            return []
        out = [value]
        if isinstance(value, Mapping):
            preferred = [
                value.get(stage),
                value.get(f"{stage}_model"),
                value.get("model"),
                value.get("lgbm_model"),
            ]
            preferred_ids = {id(item) for item in preferred if item is not None}
            for item in preferred:
                if item is not None:
                    out.extend(iter_candidates(item, depth + 1))
            for item in value.values():
                if id(item) not in preferred_ids:
                    out.extend(iter_candidates(item, depth + 1))
        elif isinstance(value, (list, tuple)):
            for item in value:
                out.extend(iter_candidates(item, depth + 1))
        return out

    candidates: list[Any] = []
    if isinstance(artifact, Mapping):
        candidates.extend(
            [
                artifact.get(stage),
                artifact.get(f"{stage}_model"),
                artifact.get("model"),
                artifact.get("lgbm_model"),
                artifact,
            ]
        )
    else:
        candidates.extend(
            [
                getattr(artifact, stage, None),
                getattr(artifact, f"{stage}_model", None),
                artifact,
            ]
        )
    candidates.extend(iter_candidates(artifact))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(candidate, Mapping):
            selected = (
                candidate.get("selected_features")
                or candidate.get("selected_feature_names")
                or candidate.get("feature_names")
                or candidate.get("input_feature_names")
                or []
            )
            params = (
                candidate.get("best_params")
                or candidate.get("params")
                or candidate.get("model_params")
                or {}
            )
        else:
            selected = (
                getattr(candidate, "selected_features", None)
                or getattr(candidate, "selected_feature_names", None)
                or getattr(candidate, "feature_names", None)
                or getattr(candidate, "input_feature_names", None)
                or []
            )
            params = (
                getattr(candidate, "best_params", None)
                or getattr(candidate, "params", None)
                or getattr(candidate, "model_params", None)
                or {}
            )
        selected_list = [str(c) for c in selected if str(c).strip()]
        params_dict = dict(params or {}) if isinstance(params, Mapping) else {}
        if selected_list or params_dict:
            return {
                "selected_features": selected_list,
                "params": params_dict,
                "stage": str(stage),
                "source": type(candidate).__name__,
            }
    return {"selected_features": [], "params": {}, "stage": str(stage), "source": "unresolved"}


def _prepare_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [str(c) for c in dict.fromkeys(columns) if str(c) in frame.columns]
    if not cols:
        return pd.DataFrame(index=frame.index)
    out = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    med = out.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(med)
    return out.astype(np.float32, copy=False)


def _binary_target(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    values = pd.to_numeric(pd.Series(arr), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    unique = np.unique(values[np.isfinite(values)])
    if unique.size <= 2 and set(np.round(unique, 6).tolist()).issubset({0.0, 1.0}):
        return (values > 0.5).astype(np.int8)
    return (values > np.nanmedian(values)).astype(np.int8)


def _coerce_oof_prediction(values: Any, n: int) -> np.ndarray:
    if values is None:
        return np.full(int(n), np.nan, dtype=np.float32)
    if isinstance(values, pd.DataFrame):
        if len(values) != int(n):
            return np.full(int(n), np.nan, dtype=np.float32)
        cols = [
            str(col)
            for col in values.columns
            if str(col).lower() not in {"timestamp", "ts", "symbol", "asset"}
            and "sigma" not in str(col).lower()
        ]
        if not cols:
            return np.full(int(n), np.nan, dtype=np.float32)
        frame = values.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
        arr = frame.to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(arr)
        count = finite.sum(axis=1)
        total = np.where(finite, arr, 0.0).sum(axis=1)
        out = np.full(int(n), np.nan, dtype=np.float32)
        mask = count > 0
        out[mask] = (total[mask] / count[mask]).astype(np.float32)
        return out
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    else:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != int(n):
        return np.full(int(n), np.nan, dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _binary_auc_fast(y: np.ndarray, score: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.int8).reshape(-1)
    ss = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.isfinite(ss)
    yy = yy[mask]
    ss = ss[mask]
    if yy.size < 3:
        return 0.5
    pos = yy > 0
    n_pos = int(pos.sum())
    n_neg = int(yy.size - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return 0.5
    order = np.argsort(ss, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, yy.size + 1, dtype=np.float64)
    auc = (float(ranks[pos].sum()) - float(n_pos * (n_pos + 1)) / 2.0) / max(float(n_pos * n_neg), 1.0)
    return float(np.clip(auc, 0.0, 1.0))


def _select_signal_orientations(
    x: pd.DataFrame,
    base_features: Sequence[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    max_features: int,
) -> dict[str, float]:
    rows: list[tuple[float, str, float]] = []
    for feature in base_features:
        if feature not in x.columns:
            continue
        values = x[feature].to_numpy(dtype=np.float32, copy=False)
        train_values = values[train_idx]
        finite = np.isfinite(train_values)
        if int(finite.sum()) < 10:
            continue
        auc = _binary_auc_fast(y[train_idx], train_values)
        var = float(np.nanvar(train_values[finite])) if finite.any() else 0.0
        strength = abs(float(auc) - 0.5) + min(max(var, 0.0), 1.0) * 1e-6
        direction = 1.0 if float(auc) >= 0.5 else -1.0
        rows.append((float(strength), str(feature), float(direction)))
    rows.sort(key=lambda item: item[0], reverse=True)
    cap = max(1, int(max_features or 1))
    return {feature: direction for _strength, feature, direction in rows[:cap]}


def _conditional_signal_uplift(
    x: pd.DataFrame,
    y_val: np.ndarray,
    val_idx: np.ndarray,
    regime_feature: str,
    signal_orientations: Mapping[str, float],
    *,
    low_quantile: float,
    high_quantile: float,
    min_rows: int,
) -> dict[str, Any]:
    if regime_feature not in x.columns or not signal_orientations:
        return {
            "signal_uplift_mean_abs": 0.0,
            "signal_uplift_max_abs": 0.0,
            "signal_uplift_positive_rate": 0.0,
            "signal_uplift_negative_rate": 0.0,
            "signal_uplift_best_signal": "",
            "signal_uplift_best_lift": 0.0,
            "signal_uplift_pair_count": 0,
        }
    regime = x[regime_feature].to_numpy(dtype=np.float32, copy=False)[val_idx]
    finite = np.isfinite(regime)
    if int(finite.sum()) < max(int(min_rows) * 2, 10):
        return {
            "signal_uplift_mean_abs": 0.0,
            "signal_uplift_max_abs": 0.0,
            "signal_uplift_positive_rate": 0.0,
            "signal_uplift_negative_rate": 0.0,
            "signal_uplift_best_signal": "",
            "signal_uplift_best_lift": 0.0,
            "signal_uplift_pair_count": 0,
        }
    lo_q = float(np.clip(low_quantile, 0.01, 0.49))
    hi_q = float(np.clip(high_quantile, 0.51, 0.99))
    low_cut, high_cut = np.nanquantile(regime[finite], [lo_q, hi_q])
    high = finite & (regime >= float(high_cut))
    low = finite & (regime <= float(low_cut))
    lifts: list[float] = []
    best_signal = ""
    best_lift = 0.0
    for signal, direction in signal_orientations.items():
        if signal not in x.columns:
            continue
        pred = x[signal].to_numpy(dtype=np.float32, copy=False)[val_idx] * float(direction)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        hr_high = _hr_star(y_val, pred, high, int(min_rows))
        hr_low = _hr_star(y_val, pred, low, int(min_rows))
        lift = float(hr_high - hr_low)
        lifts.append(lift)
        if abs(lift) > abs(best_lift):
            best_lift = lift
            best_signal = str(signal)
    if not lifts:
        return {
            "signal_uplift_mean_abs": 0.0,
            "signal_uplift_max_abs": 0.0,
            "signal_uplift_positive_rate": 0.0,
            "signal_uplift_negative_rate": 0.0,
            "signal_uplift_best_signal": "",
            "signal_uplift_best_lift": 0.0,
            "signal_uplift_pair_count": 0,
        }
    arr = np.asarray(lifts, dtype=np.float64)
    return {
        "signal_uplift_mean_abs": float(np.nanmean(np.abs(arr))),
        "signal_uplift_max_abs": float(np.nanmax(np.abs(arr))),
        "signal_uplift_positive_rate": float(np.mean(arr > 0.0)),
        "signal_uplift_negative_rate": float(np.mean(arr < 0.0)),
        "signal_uplift_best_signal": best_signal,
        "signal_uplift_best_lift": float(best_lift),
        "signal_uplift_pair_count": int(arr.size),
    }


def _oof_failure_mode_metrics(
    y_val: np.ndarray,
    oof_val: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    min_rows: int,
) -> dict[str, float]:
    pred = np.asarray(oof_val, dtype=np.float32).reshape(-1)
    valid = np.isfinite(pred)
    if int(valid.sum()) < max(int(min_rows) * 2, 10):
        return {
            "oof_available": 0.0,
            "oof_failure_high": 0.0,
            "oof_failure_low": 0.0,
            "oof_failure_lift": 0.0,
            "oof_wrong_rate_lift": 0.0,
            "oof_residual_abs_lift": 0.0,
        }
    pred = np.clip(pred, 0.0, 1.0)
    yy = np.asarray(y_val, dtype=np.float32).reshape(-1)
    high_mask = np.asarray(high, dtype=bool) & valid
    low_mask = np.asarray(low, dtype=bool) & valid
    if int(high_mask.sum()) < int(min_rows) or int(low_mask.sum()) < int(min_rows):
        return {
            "oof_available": float(np.mean(valid)),
            "oof_failure_high": 0.0,
            "oof_failure_low": 0.0,
            "oof_failure_lift": 0.0,
            "oof_wrong_rate_lift": 0.0,
            "oof_residual_abs_lift": 0.0,
        }
    residual = yy - pred
    failure = np.abs(residual)
    wrong = ((pred >= 0.5).astype(np.int8) != (yy > 0.5).astype(np.int8)).astype(np.float32)
    failure_high = float(np.nanmean(failure[high_mask]))
    failure_low = float(np.nanmean(failure[low_mask]))
    wrong_high = float(np.nanmean(wrong[high_mask]))
    wrong_low = float(np.nanmean(wrong[low_mask]))
    resid_abs_high = float(np.nanmean(np.abs(residual[high_mask])))
    resid_abs_low = float(np.nanmean(np.abs(residual[low_mask])))
    return {
        "oof_available": float(np.mean(valid)),
        "oof_failure_high": failure_high,
        "oof_failure_low": failure_low,
        "oof_failure_lift": float(failure_high - failure_low),
        "oof_wrong_rate_lift": float(wrong_high - wrong_low),
        "oof_residual_abs_lift": float(resid_abs_high - resid_abs_low),
    }


def _time_order_and_period_codes(
    n: int,
    timestamps: Any,
    *,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    if timestamps is not None and len(np.asarray(timestamps)) == n:
        ts = pd.to_datetime(pd.Series(np.asarray(timestamps)), utc=True, errors="coerce")
        order = np.argsort(
            ts.fillna(pd.Timestamp("1970-01-01", tz="UTC")).astype("int64").to_numpy(),
            kind="mergesort",
        ).astype(np.int64)
    else:
        order = np.arange(n, dtype=np.int64)
    period_codes = np.zeros(n, dtype=np.int16)
    bins = max(1, int(n_periods or 1))
    for period_i, block in enumerate(np.array_split(order, bins)):
        if block.size:
            period_codes[block.astype(np.int64)] = int(period_i)
    return order, period_codes


def _stratified_sample_positions(
    candidates: np.ndarray,
    y: np.ndarray,
    period_codes: np.ndarray,
    *,
    size: int,
    random_state: int,
) -> np.ndarray:
    cand = np.asarray(candidates, dtype=np.int64)
    target = int(size or 0)
    if target <= 0 or cand.size <= target:
        return cand
    rng = np.random.default_rng(int(random_state))
    groups: dict[tuple[int, int], np.ndarray] = {}
    for period in np.unique(period_codes[cand]):
        period_mask = period_codes[cand] == int(period)
        period_idx = cand[period_mask]
        for cls in np.unique(y[period_idx]):
            cls_idx = period_idx[y[period_idx] == int(cls)]
            if cls_idx.size:
                groups[(int(period), int(cls))] = cls_idx.astype(np.int64, copy=False)
    if not groups:
        return np.sort(rng.choice(cand, size=target, replace=False)).astype(np.int64)
    keys = list(groups)
    sizes = np.asarray([groups[key].size for key in keys], dtype=np.float64)
    ideal = sizes / max(float(sizes.sum()), 1.0) * float(target)
    allocation = np.floor(ideal).astype(np.int64)
    allocation = np.minimum(allocation, sizes.astype(np.int64))
    remaining = int(target - allocation.sum())
    if remaining > 0:
        order = np.argsort(ideal - allocation, kind="mergesort")[::-1]
        for pos in order:
            if remaining <= 0:
                break
            room = int(sizes[int(pos)] - allocation[int(pos)])
            if room <= 0:
                continue
            add = min(room, remaining)
            allocation[int(pos)] += add
            remaining -= add
    selected: list[np.ndarray] = []
    for key, take in zip(keys, allocation):
        take_i = int(take)
        if take_i <= 0:
            continue
        values = groups[key]
        chosen = values if values.size <= take_i else rng.choice(values, size=take_i, replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))
    if selected:
        out = np.concatenate(selected).astype(np.int64)
    else:
        out = np.zeros(0, dtype=np.int64)
    if out.size < target:
        missing = np.setdiff1d(cand, out, assume_unique=False)
        if missing.size:
            fill = rng.choice(missing, size=min(target - out.size, missing.size), replace=False)
            out = np.concatenate([out, np.asarray(fill, dtype=np.int64)])
    order_rank = {int(idx): rank for rank, idx in enumerate(cand)}
    return np.asarray(
        sorted((int(idx) for idx in out[:target]), key=lambda idx: order_rank.get(idx, len(order_rank))),
        dtype=np.int64,
    )


def _period_folds(
    n: int,
    y: np.ndarray,
    timestamps: Any,
    *,
    n_folds: int,
    sample_fraction: float,
    max_rows: int,
    random_state: int,
    stratified_period_bins: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    order, period_codes = _time_order_and_period_codes(
        n,
        timestamps,
        n_periods=max(int(stratified_period_bins or 1), int(n_folds or 1)),
    )
    if int(max_rows or 0) > 0 and n > int(max_rows):
        sampled = _stratified_sample_positions(
            order,
            y,
            period_codes,
            size=int(max_rows),
            random_state=int(random_state) + 10,
        )
        sampled_set = set(int(idx) for idx in sampled)
        order = np.asarray([int(idx) for idx in order if int(idx) in sampled_set], dtype=np.int64)
    splits = [block.astype(np.int64, copy=False) for block in np.array_split(order, max(2, int(n_folds))) if block.size]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_pos = np.asarray(order, dtype=np.int64)
    frac = float(np.clip(sample_fraction, 0.05, 1.0))
    for fold_i, val_idx in enumerate(splits):
        train_pool = np.setdiff1d(all_pos, val_idx, assume_unique=False).astype(np.int64)
        if train_pool.size < 10 or val_idx.size < 5:
            continue
        train_size = max(10, int(np.ceil(frac * train_pool.size)))
        train_size = min(train_size, train_pool.size)
        train_idx = _stratified_sample_positions(
            train_pool,
            y,
            period_codes,
            size=train_size,
            random_state=int(random_state) + 1000 + fold_i,
        )
        if np.unique(y[train_idx]).size < 2 or np.unique(y[val_idx]).size < 2:
            continue
        folds.append((train_idx, np.sort(val_idx).astype(np.int64)))
    return folds


def _params(config: RegimeFeatureLGBMFilterConfig, reused: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(reused or {})
    aliases = {
        "reg_lambda": "lambda_l2",
        "reg_alpha": "lambda_l1",
        "colsample_bytree": "feature_fraction",
        "subsample": "bagging_fraction",
    }
    for old, new in aliases.items():
        if old in params and new not in params:
            params[new] = params.pop(old)
    out = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": int(min(max(10, int(params.get("n_estimators", config.max_trees))), int(config.max_trees))),
        "learning_rate": float(params.get("learning_rate", config.learning_rate)),
        "num_leaves": int(params.get("num_leaves", config.num_leaves)),
        "max_depth": int(params.get("max_depth", config.max_depth)),
        "min_child_samples": int(params.get("min_child_samples", config.min_child_samples)),
        "subsample": float(params.get("bagging_fraction", params.get("subsample", config.subsample))),
        "colsample_bytree": float(params.get("feature_fraction", params.get("colsample_bytree", config.colsample_bytree))),
        "random_state": int(config.random_state),
        "n_jobs": 1,
        "verbosity": -1,
    }
    return out


def _shadow_name(feature: str) -> str:
    return f"__shadow__{feature}"


def _add_shadow_features(
    x: pd.DataFrame,
    regime_features: Sequence[str],
    *,
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(int(random_state))
    out = x.copy()
    shadow_cols: list[str] = []
    for feature in regime_features:
        if feature not in out.columns:
            continue
        name = _shadow_name(feature)
        values = out[feature].to_numpy(dtype=np.float32, copy=True)
        out[name] = values[rng.permutation(len(values))].astype(np.float32)
        shadow_cols.append(name)
    return out, shadow_cols


def _tree_paths(tree: Mapping[str, Any], feature_names: Sequence[str]) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []

    def walk(node: Mapping[str, Any], features: list[str], gains: list[float]) -> None:
        if "leaf_index" in node:
            paths.append({"features": list(features), "gains": list(gains)})
            return
        idx = node.get("split_feature")
        name = str(feature_names[int(idx)]) if idx is not None and int(idx) < len(feature_names) else ""
        gain = float(node.get("split_gain", 0.0) or 0.0)
        for child_name in ("left_child", "right_child"):
            child = node.get(child_name)
            if isinstance(child, Mapping):
                walk(child, features + [name], gains + [gain])

    root = tree.get("tree_structure", tree)
    if isinstance(root, Mapping):
        walk(root, [], [])
    return paths


def _collect_tree_structures(model: Any) -> tuple[list[Mapping[str, Any]], list[str]]:
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    feature_names = [str(f) for f in dump.get("feature_names", [])]
    trees = [tree for tree in dump.get("tree_info", []) if isinstance(tree, Mapping)]
    return trees, feature_names


def _gain_split_maps(trees: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    gain: dict[str, float] = {}
    split: dict[str, int] = {}
    tree_use: dict[str, int] = {}
    for tree in trees:
        used: set[str] = set()

        def walk(node: Mapping[str, Any]) -> None:
            if "leaf_index" in node:
                return
            idx = node.get("split_feature")
            if idx is not None and int(idx) < len(feature_names):
                name = str(feature_names[int(idx)])
                gain[name] = gain.get(name, 0.0) + float(node.get("split_gain", 0.0) or 0.0)
                split[name] = split.get(name, 0) + 1
                used.add(name)
            for child_name in ("left_child", "right_child"):
                child = node.get(child_name)
                if isinstance(child, Mapping):
                    walk(child)

        root = tree.get("tree_structure", tree)
        if isinstance(root, Mapping):
            walk(root)
        for name in used:
            tree_use[name] = tree_use.get(name, 0) + 1
    return gain, split, tree_use


def _node_go_left(node: Mapping[str, Any], value: float) -> bool:
    if not np.isfinite(value):
        return bool(node.get("default_left", True))
    threshold = float(node.get("threshold", 0.0) or 0.0)
    decision = str(node.get("decision_type", "<="))
    if decision in {"<=", "<"}:
        return value <= threshold
    if decision in {">", ">="}:
        return value > threshold
    return value <= threshold


def _route_exposures(
    model: Any,
    x_val: pd.DataFrame,
    regime_features: Sequence[str],
    signal_features: Sequence[str],
    *,
    max_rows: int,
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    trees, feature_names = _collect_tree_structures(model)
    regime_set = set(str(f) for f in regime_features)
    signal_set = set(str(f) for f in signal_features)
    n = len(x_val)
    if n == 0 or not trees:
        return {}, {}
    route_idx = np.arange(n, dtype=np.int64)
    cap = int(max_rows or 0)
    if cap > 0 and n > cap:
        rng = np.random.default_rng(int(random_state) + 3100)
        route_idx = np.sort(rng.choice(route_idx, size=cap, replace=False)).astype(np.int64)
    arr = x_val.reindex(columns=feature_names, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)
    exposures = {feature: np.zeros(n, dtype=np.float32) for feature in regime_set}
    route_counts = {feature: 0 for feature in regime_set}
    signal_route_counts = {feature: 0 for feature in regime_set}
    dense_signal_counts = {feature: 0 for feature in regime_set}
    signal_gain = {feature: 0.0 for feature in regime_set}
    total_gain = {feature: 0.0 for feature in regime_set}
    for tree in trees:
        root = tree.get("tree_structure", tree)
        if not isinstance(root, Mapping):
            continue
        for row_i in route_idx:
            node = root
            path_features: list[str] = []
            path_gains: list[float] = []
            while isinstance(node, Mapping) and "leaf_index" not in node:
                idx = int(node.get("split_feature", -1))
                name = str(feature_names[idx]) if 0 <= idx < len(feature_names) else ""
                path_features.append(name)
                path_gains.append(float(node.get("split_gain", 0.0) or 0.0))
                go_left = _node_go_left(node, float(arr[int(row_i), idx]) if 0 <= idx < arr.shape[1] else np.nan)
                child = node.get("left_child" if go_left else "right_child")
                if not isinstance(child, Mapping):
                    break
                node = child
            path_set = set(path_features)
            signal_in_path = path_set.intersection(signal_set)
            signal_density = len([f for f in path_features if f in signal_set]) / max(len(path_features), 1)
            gains = np.asarray(path_gains, dtype=np.float64)
            for feature in path_set.intersection(regime_set):
                exposures[feature][int(row_i)] += 1.0
                route_counts[feature] += 1
                if signal_in_path:
                    signal_route_counts[feature] += 1
                if signal_density >= 0.75:
                    dense_signal_counts[feature] += 1
                if gains.size:
                    total_gain[feature] += float(np.sum(gains))
                    signal_gain[feature] += float(
                        np.sum([g for f, g in zip(path_features, path_gains) if f in signal_set])
                    )
    denom = max(float(len(trees)), 1.0)
    for values in exposures.values():
        values /= denom
    context: dict[str, dict[str, float]] = {}
    for feature in regime_set:
        routes = max(route_counts.get(feature, 0), 1)
        context[feature] = {
            "sample_path_exposure": float(route_counts.get(feature, 0) / max(len(route_idx) * len(trees), 1)),
            "interaction_path_share": float(signal_route_counts.get(feature, 0) / routes),
            "signal_dense_interaction_share": float(dense_signal_counts.get(feature, 0) / routes),
            "signal_gain_share": float(signal_gain.get(feature, 0.0) / max(total_gain.get(feature, 0.0), 1e-12)),
            "signal_exposure_share": float(signal_route_counts.get(feature, 0) / routes),
        }
    return exposures, context


def _hr_star(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, min_rows: int) -> float:
    idx = np.flatnonzero(mask)
    if idx.size < int(min_rows):
        return 0.0
    scores: list[float] = []
    weights = [1.0, 0.5, 0.33]
    for k in (30, 20, 10):
        kk = min(int(k), idx.size)
        if kk <= 0:
            scores.append(0.0)
            continue
        order = idx[np.argsort(pred[idx], kind="mergesort")[::-1][:kk]]
        scores.append(float(np.mean(y[order] > 0)))
    return float(np.dot(scores, weights) / sum(weights))


def _weekly_stability(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, timestamps: Any, min_rows: int) -> float:
    idx = np.flatnonzero(mask)
    if idx.size < int(min_rows):
        return 0.0
    if timestamps is None or len(np.asarray(timestamps)) != len(y):
        return _hr_star(y, pred, mask, min_rows)
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)[idx]), utc=True, errors="coerce")
    try:
        ts_for_period = ts.dt.tz_convert(None)
    except Exception:
        ts_for_period = ts
    weeks = ts_for_period.dt.to_period("W").astype(str)
    vals: list[float] = []
    for week in pd.unique(weeks):
        local = idx[weeks.to_numpy() == week]
        if local.size >= max(3, min_rows // 3):
            local_mask = np.zeros(len(y), dtype=bool)
            local_mask[local] = True
            vals.append(_hr_star(y, pred, local_mask, max(3, min_rows // 3)))
    if len(vals) < 2:
        return _hr_star(y, pred, mask, min_rows)
    arr = np.asarray(vals, dtype=np.float64)
    q5, q10, q50 = np.nanpercentile(arr, [5.0, 10.0, 50.0])
    if not np.isfinite(q50) or q50 <= 1e-12:
        return 0.0
    return float(np.clip((q5 + q10) / (2.0 * q50), 0.0, 2.0))


def _is_context_portfolio_feature(feature: str) -> bool:
    return "ctx_portfolio_" in str(feature).lower()


def _risk_budget_direction(feature: str) -> float:
    low = str(feature).lower()
    if "risk_cut" in low or "defensive" in low or "no_trade" in low:
        return -1.0
    return 1.0


def _percent_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if int(finite.sum()) <= 1:
        return out
    local = arr[finite]
    order = np.argsort(local, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = (np.arange(order.size, dtype=np.float64) + 0.5) / max(float(order.size), 1.0)
    out[finite] = ranks
    return out


def _risk_budget_scaler_metrics(
    x: pd.DataFrame,
    y_val: np.ndarray,
    val_idx: np.ndarray,
    model_pred: np.ndarray,
    oof_val: np.ndarray,
    feature: str,
    *,
    timestamps: Any,
    min_rows: int,
    quantile_bins: int,
) -> dict[str, float]:
    """Evaluate a context portfolio as a monotonic risk-budget scaler.

    This deliberately treats the feature as a multiplier on an existing ranking
    instead of requiring it to be structurally used as a LightGBM split.
    """

    defaults = {
        "risk_budget_scaler_available": 0.0,
        "risk_budget_scaled_hr_lift": 0.0,
        "risk_budget_high_low_hr_lift": 0.0,
        "risk_budget_failure_avoidance": 0.0,
        "risk_budget_wrong_rate_avoidance": 0.0,
        "risk_budget_monotonicity": 0.0,
        "risk_budget_monotonic_consistency": 0.0,
        "risk_budget_stability_lift": 0.0,
    }
    if not _is_context_portfolio_feature(feature) or feature not in x.columns:
        return defaults
    values = x[feature].to_numpy(dtype=np.float32, copy=False)[val_idx]
    finite_values = np.isfinite(values)
    if int(finite_values.sum()) < max(int(min_rows) * 2, 10):
        defaults["risk_budget_scaler_available"] = float(np.mean(finite_values)) if values.size else 0.0
        return defaults
    oof_finite = np.isfinite(oof_val)
    pred_ref = np.asarray(oof_val if int(oof_finite.sum()) >= max(int(min_rows) * 2, 10) else model_pred, dtype=np.float32).reshape(-1)
    valid = finite_values & np.isfinite(pred_ref)
    if int(valid.sum()) < max(int(min_rows) * 2, 10):
        defaults["risk_budget_scaler_available"] = float(np.mean(finite_values)) if values.size else 0.0
        return defaults

    yy = np.asarray(y_val, dtype=np.int8).reshape(-1)[valid]
    raw = np.clip(np.asarray(pred_ref, dtype=np.float32).reshape(-1)[valid], 0.0, 1.0)
    budget = _percent_rank(values[valid])
    if float(_risk_budget_direction(feature)) < 0.0:
        budget = 1.0 - budget
    budget = np.nan_to_num(budget, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
    all_mask = np.ones(yy.size, dtype=bool)
    scaled = np.clip(raw * (0.20 + 0.80 * budget), 0.0, 1.0)
    raw_hr = _hr_star(yy, raw, all_mask, int(min_rows))
    scaled_hr = _hr_star(yy, scaled, all_mask, int(min_rows))
    lo, hi = np.nanquantile(budget, [0.33, 0.67])
    low_budget = budget <= float(lo)
    high_budget = budget >= float(hi)
    high_hr = _hr_star(yy, raw, high_budget, int(min_rows))
    low_hr = _hr_star(yy, raw, low_budget, int(min_rows))
    stab_high = _weekly_stability(yy, raw, high_budget, np.asarray(timestamps)[valid] if timestamps is not None else None, int(min_rows))
    stab_low = _weekly_stability(yy, raw, low_budget, np.asarray(timestamps)[valid] if timestamps is not None else None, int(min_rows))
    residual = yy.astype(np.float32) - raw
    failure = np.abs(residual)
    wrong = ((raw >= 0.5).astype(np.int8) != (yy > 0).astype(np.int8)).astype(np.float32)
    failure_avoidance = (
        float(np.nanmean(failure[low_budget]) - np.nanmean(failure[high_budget]))
        if int(low_budget.sum()) >= int(min_rows) and int(high_budget.sum()) >= int(min_rows)
        else 0.0
    )
    wrong_avoidance = (
        float(np.nanmean(wrong[low_budget]) - np.nanmean(wrong[high_budget]))
        if int(low_budget.sum()) >= int(min_rows) and int(high_budget.sum()) >= int(min_rows)
        else 0.0
    )

    bin_scores: list[float] = []
    bins = max(3, int(quantile_bins or 5))
    cuts = np.unique(np.nanquantile(budget, np.linspace(0.0, 1.0, bins + 1)))
    if cuts.size >= 3:
        for i in range(cuts.size - 1):
            left = float(cuts[i])
            right = float(cuts[i + 1])
            mask = budget >= left
            mask &= budget <= right if i == cuts.size - 2 else budget < right
            if int(mask.sum()) >= int(min_rows):
                bin_scores.append(_hr_star(yy, raw, mask, int(min_rows)))
    if len(bin_scores) >= 2:
        arr = np.asarray(bin_scores, dtype=np.float64)
        idx = np.arange(arr.size, dtype=np.float64)
        if float(np.nanstd(arr)) > 1e-12:
            monotonicity = float(np.corrcoef(idx, arr)[0, 1])
        else:
            monotonicity = 0.0
        diffs = np.diff(arr)
        consistency = float(np.mean(diffs >= -1e-12)) if diffs.size else 0.0
    else:
        monotonicity = 0.0
        consistency = 0.0
    return {
        "risk_budget_scaler_available": float(np.mean(valid)),
        "risk_budget_scaled_hr_lift": float(scaled_hr - raw_hr),
        "risk_budget_high_low_hr_lift": float(high_hr - low_hr),
        "risk_budget_failure_avoidance": failure_avoidance,
        "risk_budget_wrong_rate_avoidance": wrong_avoidance,
        "risk_budget_monotonicity": float(np.nan_to_num(monotonicity, nan=0.0)),
        "risk_budget_monotonic_consistency": float(np.nan_to_num(consistency, nan=0.0)),
        "risk_budget_stability_lift": float(stab_high - stab_low),
    }


def _trimmed_mean(values: Sequence[float], trim: float = 0.10) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if arr.size < 4:
        return float(np.mean(arr))
    lo = int(np.floor(trim * arr.size))
    hi = int(np.ceil((1.0 - trim) * arr.size))
    arr = np.sort(arr)[lo:max(lo + 1, hi)]
    return float(np.mean(arr)) if arr.size else 0.0


def _directional_lift_pass(
    median_value: float,
    trimmed_value: float,
    positive_rate: float,
    negative_rate: float,
    *,
    min_fold_fraction: float,
) -> tuple[bool, bool, str]:
    opportunity = bool(
        float(median_value) > 0.0
        and float(trimmed_value) > 0.0
        and float(positive_rate) >= float(min_fold_fraction)
    )
    risk_gate = bool(
        float(median_value) < 0.0
        and float(trimmed_value) < 0.0
        and float(negative_rate) >= float(min_fold_fraction)
    )
    if opportunity:
        role = "opportunity_gate"
    elif risk_gate:
        role = "risk_gate"
    else:
        role = "mixed_or_weak_context"
    return opportunity, risk_gate, role


def _source_for_feature(feature: str, source_map: Mapping[str, str] | None = None) -> str:
    if source_map and feature in source_map:
        return str(source_map[feature])
    low = str(feature).lower()
    if "url_sigreg__" in low or "__x__url_" in low:
        return "signal_regime_interaction"
    if "ctx_portfolio_" in low:
        return "context_portfolio"
    if "latent_" in low:
        return "latent_context"
    if "residual_structure" in low:
        return "residual_structure"
    if "family_crowding" in low:
        return "crowding_regime"
    if "family_leverage_unwind" in low:
        return "leverage_unwind_regime"
    if "family_low_participation_rebound" in low:
        return "low_participation_rebound_regime"
    if "family_cross_asset_decoupling" in low:
        return "cross_asset_decoupling_regime"
    if "family_session_microstructure" in low:
        return "session_microstructure_regime"
    if "family_liquidity_risk" in low:
        return "liquidity_risk_regime"
    if "family_market_structure" in low:
        return "market_structure_regime"
    if "family_path_opportunity" in low:
        return "path_opportunity_regime"
    if "mfa" in low:
        return "mfa"
    if "sparse_ae" in low or "contrastive_ae" in low:
        return "autoencoder"
    if "leaf" in low:
        return "leaf"
    if "regime_prob" in low:
        return "probability"
    if low.endswith("_regime") or "smoothed_regime" in low:
        return "label"
    if "transition" in low or "hazard" in low or "duration" in low:
        return "transition"
    if "xs_z" in low:
        return "cross_sectional_residual"
    if "market" in low:
        return "market_aggregate"
    return str(feature).split("__", 1)[0]


def _absolute_contribution_exposure(
    model: Any,
    x_val: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return absolute model-contribution exposure per feature when available.

    The filter compares high- vs low-exposure samples for a candidate regime
    feature. Exposure is directional-agnostic: a feature can matter because it
    pushes predictions up or down, so using only positive contributions would
    undercount useful context filters with negative SHAP/contribution signs.
    """

    try:
        booster = model.booster_ if hasattr(model, "booster_") else model
        dump = booster.dump_model()
        feature_names = [str(f) for f in dump.get("feature_names", [])]
        contrib = booster.predict(
            x_val.reindex(columns=feature_names, fill_value=0.0),
            pred_contrib=True,
        )
        contrib_arr = np.asarray(contrib, dtype=np.float32)
        if contrib_arr.ndim != 2 or contrib_arr.shape[1] < len(feature_names):
            return {}
        out: dict[str, np.ndarray] = {}
        pos = {name: i for i, name in enumerate(feature_names)}
        for feature in features:
            idx = pos.get(str(feature))
            if idx is None:
                continue
            out[str(feature)] = np.abs(contrib_arr[:, int(idx)]).astype(np.float32)
        return out
    except Exception:
        return {}


def _aggregate_feature_metrics(
    fold_rows: pd.DataFrame,
    features: Sequence[str],
    config: RegimeFeatureLGBMFilterConfig,
    source_map: Mapping[str, str] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        cur = fold_rows.loc[fold_rows["feature"].astype(str).eq(str(feature))]
        if cur.empty:
            continue
        def col_values(name: str) -> np.ndarray:
            if name not in cur.columns:
                return np.zeros(len(cur), dtype=np.float64)
            return pd.to_numeric(cur[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        hr = pd.to_numeric(cur["HR_lift"], errors="coerce").to_numpy(dtype=np.float64)
        stab = pd.to_numeric(cur["stability_lift"], errors="coerce").to_numpy(dtype=np.float64)
        score = pd.to_numeric(cur["score"], errors="coerce").to_numpy(dtype=np.float64)
        signal_abs = col_values("signal_uplift_mean_abs")
        signal_max = col_values("signal_uplift_max_abs")
        oof_available = col_values("oof_available")
        oof_failure_lift = col_values("oof_failure_lift")
        oof_wrong_lift = col_values("oof_wrong_rate_lift")
        oof_resid_lift = col_values("oof_residual_abs_lift")
        budget_available = col_values("risk_budget_scaler_available")
        budget_scaled_hr = col_values("risk_budget_scaled_hr_lift")
        budget_high_low_hr = col_values("risk_budget_high_low_hr_lift")
        budget_failure_avoid = col_values("risk_budget_failure_avoidance")
        budget_wrong_avoid = col_values("risk_budget_wrong_rate_avoidance")
        budget_monotonicity = col_values("risk_budget_monotonicity")
        budget_consistency = col_values("risk_budget_monotonic_consistency")
        budget_stability = col_values("risk_budget_stability_lift")
        budget_hit = (
            (budget_scaled_hr > float(config.risk_budget_scaler_min_scaled_hr_lift))
            & (budget_high_low_hr > float(config.risk_budget_scaler_min_high_low_hr_lift))
            & (budget_failure_avoid >= float(config.risk_budget_scaler_min_failure_avoidance))
            & (budget_monotonicity >= 0.0)
            & (budget_consistency >= 0.50)
            & (budget_available > 0.0)
        )
        structural = cur["structural_pass"].astype(bool).to_numpy()
        context_pass = cur["context_pass"].astype(bool).to_numpy()
        score_hit_rate = float(np.mean(score > 0.0)) if score.size else 0.0
        structural_pass_rate = float(np.mean(structural)) if structural.size else 0.0
        row = {
            "feature": str(feature),
            "source": _source_for_feature(str(feature), source_map),
            "fold_count": int(len(cur)),
            "n_trees_using_mean": float(pd.to_numeric(cur["n_trees_using"], errors="coerce").mean()),
            "leaf_path_share_mean": float(pd.to_numeric(cur["leaf_path_share"], errors="coerce").mean()),
            "sample_path_exposure_mean": float(pd.to_numeric(cur["sample_path_exposure"], errors="coerce").mean()),
            "gain_ratio_mean": float(pd.to_numeric(cur["gain_ratio"], errors="coerce").mean()),
            "gain_mean": float(pd.to_numeric(cur["gain"], errors="coerce").mean()),
            "shadow_gain_p85_mean": float(pd.to_numeric(cur["shadow_gain_p85"], errors="coerce").mean()),
            "structural_pass_rate": structural_pass_rate,
            "median_HR_lift": float(np.nanmedian(hr)) if hr.size else 0.0,
            "trimmed_mean_HR_lift": _trimmed_mean(hr),
            "HR_lift_positive_rate": float(np.mean(hr > 0.0)) if hr.size else 0.0,
            "HR_lift_negative_rate": float(np.mean(hr < 0.0)) if hr.size else 0.0,
            "median_stability_lift": float(np.nanmedian(stab)) if stab.size else 0.0,
            "trimmed_mean_stability_lift": _trimmed_mean(stab),
            "stability_lift_positive_rate": float(np.mean(stab > 0.0)) if stab.size else 0.0,
            "stability_lift_negative_rate": float(np.mean(stab < 0.0)) if stab.size else 0.0,
            "median_score": float(np.nanmedian(score)) if score.size else 0.0,
            "trimmed_mean_score": _trimmed_mean(score),
            "score_hit_rate": score_hit_rate,
            "interaction_path_share_mean": float(pd.to_numeric(cur["interaction_path_share"], errors="coerce").mean()),
            "signal_dense_interaction_share_mean": float(pd.to_numeric(cur["signal_dense_interaction_share"], errors="coerce").mean()),
            "signal_gain_share_mean": float(pd.to_numeric(cur["signal_gain_share"], errors="coerce").mean()),
            "signal_exposure_share_mean": float(pd.to_numeric(cur["signal_exposure_share"], errors="coerce").mean()),
            "context_interaction_pass_rate": float(np.mean(cur["context_interaction_pass"].astype(bool)))
            if "context_interaction_pass" in cur.columns
            else 0.0,
            "coarse_gate_context_pass_rate": float(np.mean(cur["coarse_gate_context_pass"].astype(bool)))
            if "coarse_gate_context_pass" in cur.columns
            else 0.0,
            "context_pass_rate": float(np.mean(context_pass)) if context_pass.size else 0.0,
            "signal_uplift_mean_abs": float(np.nanmean(signal_abs)) if signal_abs.size else 0.0,
            "signal_uplift_max_abs": float(np.nanmax(signal_max)) if signal_max.size else 0.0,
            "signal_uplift_abs_hit_rate": float(
                np.mean(
                    (signal_abs >= float(config.signal_uplift_min_abs_mean))
                    | (signal_max >= float(config.signal_uplift_min_abs_max))
                )
            )
            if signal_abs.size
            else 0.0,
            "signal_uplift_context_pass_rate": float(
                np.mean(
                    (signal_abs >= float(config.signal_uplift_min_abs_mean))
                    | (signal_max >= float(config.signal_uplift_min_abs_max))
                )
            )
            if signal_abs.size
            else 0.0,
            "oof_available_rate": float(np.nanmean(oof_available)) if oof_available.size else 0.0,
            "median_oof_failure_lift": float(np.nanmedian(oof_failure_lift)) if oof_failure_lift.size else 0.0,
            "mean_oof_failure_lift": float(np.nanmean(oof_failure_lift)) if oof_failure_lift.size else 0.0,
            "oof_failure_lift_positive_rate": float(np.mean(oof_failure_lift > 0.0)) if oof_failure_lift.size else 0.0,
            "median_oof_wrong_rate_lift": float(np.nanmedian(oof_wrong_lift)) if oof_wrong_lift.size else 0.0,
            "median_oof_residual_abs_lift": float(np.nanmedian(oof_resid_lift)) if oof_resid_lift.size else 0.0,
            "risk_budget_scaler_available_rate": float(np.nanmean(budget_available)) if budget_available.size else 0.0,
            "median_risk_budget_scaled_hr_lift": float(np.nanmedian(budget_scaled_hr)) if budget_scaled_hr.size else 0.0,
            "trimmed_mean_risk_budget_scaled_hr_lift": _trimmed_mean(budget_scaled_hr),
            "median_risk_budget_high_low_hr_lift": float(np.nanmedian(budget_high_low_hr)) if budget_high_low_hr.size else 0.0,
            "median_risk_budget_failure_avoidance": float(np.nanmedian(budget_failure_avoid)) if budget_failure_avoid.size else 0.0,
            "median_risk_budget_wrong_rate_avoidance": float(np.nanmedian(budget_wrong_avoid)) if budget_wrong_avoid.size else 0.0,
            "median_risk_budget_stability_lift": float(np.nanmedian(budget_stability)) if budget_stability.size else 0.0,
            "risk_budget_monotonicity_mean": float(np.nanmean(budget_monotonicity)) if budget_monotonicity.size else 0.0,
            "risk_budget_monotonic_consistency_mean": float(np.nanmean(budget_consistency)) if budget_consistency.size else 0.0,
            "risk_budget_scaler_hit_rate": float(np.mean(budget_hit)) if budget_hit.size else 0.0,
            "rank_score": 0.0,
        }
        is_budget_portfolio = _is_context_portfolio_feature(str(feature))
        row["risk_budget_scaler_pass"] = bool(
            bool(config.risk_budget_scaler_enabled)
            and is_budget_portfolio
            and row["risk_budget_scaler_available_rate"] > 0.0
            and row["risk_budget_scaler_hit_rate"] >= float(config.risk_budget_scaler_min_fold_fraction)
            and row["median_risk_budget_scaled_hr_lift"] > float(config.risk_budget_scaler_min_scaled_hr_lift)
            and row["median_risk_budget_high_low_hr_lift"] > float(config.risk_budget_scaler_min_high_low_hr_lift)
            and row["median_risk_budget_failure_avoidance"] >= float(config.risk_budget_scaler_min_failure_avoidance)
            and row["risk_budget_monotonic_consistency_mean"] >= 0.50
        )
        row["risk_budget_scaler_score"] = float(
            max(float(row["median_risk_budget_scaled_hr_lift"]), 0.0)
            * (1.0 + max(float(row["median_risk_budget_high_low_hr_lift"]), 0.0))
            * (1.0 + max(float(row["median_risk_budget_failure_avoidance"]), 0.0))
            * max(float(row["risk_budget_scaler_hit_rate"]), 0.0)
            * max(float(row["risk_budget_scaler_available_rate"]), 0.0)
            * max(0.0, 0.50 + 0.50 * float(row["risk_budget_monotonicity_mean"]))
        )
        row["structural_pass"] = bool(
            row["n_trees_using_mean"] >= max(
                float(config.structural_min_trees_using),
                float(config.structural_min_tree_fraction) * float(cur["tree_count"].max()),
            )
            and row["leaf_path_share_mean"] >= float(config.structural_min_leaf_path_share)
            and row["sample_path_exposure_mean"] >= float(config.structural_min_sample_path_exposure)
            and row["gain_ratio_mean"] >= float(config.structural_min_gain_ratio)
            and (
                (not bool(config.use_shadow_gain))
                or row["gain_mean"] > row["shadow_gain_p85_mean"]
            )
        )
        hr_opp, hr_risk, hr_role = _directional_lift_pass(
            float(row["median_HR_lift"]),
            float(row["trimmed_mean_HR_lift"]),
            float(row["HR_lift_positive_rate"]),
            float(row["HR_lift_negative_rate"]),
            min_fold_fraction=float(config.lift_positive_fold_fraction),
        )
        stab_opp, stab_risk, stab_role = _directional_lift_pass(
            float(row["median_stability_lift"]),
            float(row["trimmed_mean_stability_lift"]),
            float(row["stability_lift_positive_rate"]),
            float(row["stability_lift_negative_rate"]),
            min_fold_fraction=float(config.stability_positive_fold_fraction),
        )
        row["HR_opportunity_pass"] = bool(hr_opp)
        row["HR_risk_gate_pass"] = bool(hr_risk)
        row["stability_opportunity_pass"] = bool(stab_opp)
        row["stability_risk_gate_pass"] = bool(stab_risk)
        row["context_role"] = (
            "opportunity_gate"
            if hr_opp and stab_opp
            else "risk_gate"
            if hr_risk and stab_risk
            else hr_role
            if hr_role == stab_role
            else "mixed_or_weak_context"
        )
        row["exposure_precision_pass"] = bool(hr_opp or hr_risk)
        row["stability_pass"] = bool(stab_opp or stab_risk)
        row["combined_score_pass"] = bool(
            row["median_score"] > 0.0
            and row["trimmed_mean_score"] > 0.0
            and row["score_hit_rate"] >= float(config.score_positive_fold_fraction)
        )
        row["signal_uplift_context_pass"] = bool(
            row["signal_uplift_context_pass_rate"] >= float(config.signal_uplift_min_fold_fraction)
        )
        row["oof_failure_alignment_pass"] = bool(
            row["oof_available_rate"] > 0.0
            and row["median_oof_failure_lift"] >= float(config.risk_gate_min_oof_failure_lift)
            and row["oof_failure_lift_positive_rate"] >= float(config.coarse_gate_min_fold_fraction)
        )
        coarse_context_pass = bool(
            bool(config.coarse_gate_context_enabled)
            and row["coarse_gate_context_pass_rate"] >= float(config.coarse_gate_min_fold_fraction)
        )
        interaction_context_pass = bool(
            row["interaction_path_share_mean"] >= float(config.interaction_path_share_min)
            and row["signal_dense_interaction_share_mean"] >= float(config.signal_dense_interaction_share_min)
            and row["signal_gain_share_mean"] >= float(config.signal_gain_share_min)
            and row["signal_exposure_share_mean"] >= float(config.signal_exposure_share_min)
        )
        row["coarse_gate_context_pass"] = bool(coarse_context_pass)
        row["interaction_context_pass"] = bool(interaction_context_pass)
        row["context_pass"] = bool(
            (not _is_train_base(config.objective_mode))
            or (not bool(config.base_context_filter_enabled))
            or interaction_context_pass
            or coarse_context_pass
            or row["risk_budget_scaler_pass"]
        )
        effective_structural_pass_rate = max(
            float(row["structural_pass_rate"]),
            float(row["coarse_gate_context_pass_rate"]),
            float(row["risk_budget_scaler_hit_rate"]) if row["risk_budget_scaler_pass"] else 0.0,
        )
        row["effective_structural_pass_rate"] = effective_structural_pass_rate
        row["effective_structural_pass"] = bool(
            row["structural_pass"]
            or row["coarse_gate_context_pass"]
            or row["risk_budget_scaler_pass"]
        )
        row["risk_gate_hr_avoidance"] = float(max(-float(row["median_HR_lift"]), 0.0))
        row["risk_gate_stability_avoidance"] = float(max(-float(row["median_stability_lift"]), 0.0))
        row["risk_gate_oof_failure_lift"] = float(max(float(row["median_oof_failure_lift"]), 0.0))
        row["risk_gate_oof_wrong_rate_lift"] = float(max(float(row["median_oof_wrong_rate_lift"]), 0.0))
        row["risk_gate_oof_residual_abs_lift"] = float(max(float(row["median_oof_residual_abs_lift"]), 0.0))
        row["risk_gate_signal_uplift_abs"] = float(row["signal_uplift_mean_abs"])
        row["risk_gate_acceptance_score"] = float(
            row["risk_gate_hr_avoidance"]
            * max(row["risk_gate_stability_avoidance"], 1e-6)
            * max(float(row["score_hit_rate"]), 0.0)
            * max(float(effective_structural_pass_rate), 0.0)
            * (1.0 + row["risk_gate_oof_failure_lift"])
            * (1.0 + row["risk_gate_oof_wrong_rate_lift"])
            * (1.0 + row["risk_gate_oof_residual_abs_lift"])
            * (1.0 + row["risk_gate_signal_uplift_abs"])
        )
        row["risk_gate_defensive_score"] = float(
            row["risk_gate_acceptance_score"]
            * max(float(row["oof_available_rate"]), 0.0)
        )
        row["risk_gate_defensive_pass"] = bool(
            row["oof_failure_alignment_pass"]
            and row["risk_gate_oof_failure_lift"] >= float(config.risk_gate_min_oof_failure_lift)
            and row["risk_gate_oof_wrong_rate_lift"] >= float(config.risk_gate_min_oof_wrong_rate_lift)
            and row["risk_gate_oof_residual_abs_lift"] >= float(config.risk_gate_min_oof_residual_abs_lift)
        )
        row["risk_gate_acceptance_pass"] = bool(
            row["context_role"] == "risk_gate"
            and row["risk_gate_hr_avoidance"] >= float(config.risk_gate_min_hr_avoidance)
            and row["risk_gate_stability_avoidance"] >= float(config.risk_gate_min_stability_avoidance)
            and row["HR_risk_gate_pass"]
            and row["stability_risk_gate_pass"]
            and row["combined_score_pass"]
            and row["risk_gate_defensive_pass"]
            and (
                row["oof_available_rate"] <= 0.0
                or row["oof_failure_alignment_pass"]
                or row["signal_uplift_context_pass"]
            )
        )
        row["opportunity_helpfulness_score"] = float(
            max(float(row["median_score"]), 0.0)
            * float(row["score_hit_rate"])
            * float(effective_structural_pass_rate)
            * (1.0 + float(row["signal_uplift_mean_abs"]))
        )
        row["model_helpfulness_score"] = float(
            max(
                row["opportunity_helpfulness_score"]
                if row["context_role"] == "opportunity_gate"
                else 0.0,
                row["risk_gate_defensive_score"]
                if row["context_role"] == "risk_gate"
                else 0.0,
                float(row["signal_uplift_context_pass_rate"]) * float(row["signal_uplift_mean_abs"]),
                float(row["risk_budget_scaler_score"]),
            )
        )
        row["model_helpfulness_pass"] = bool(
            (
                row["context_role"] == "opportunity_gate"
                and row["HR_opportunity_pass"]
                and row["stability_opportunity_pass"]
                and row["signal_uplift_context_pass"]
            )
            or row["risk_gate_acceptance_pass"]
            or row["risk_budget_scaler_pass"]
        )
        row["rank_score"] = float(
            row["model_helpfulness_score"]
            * max(float(effective_structural_pass_rate), 1e-6)
        )
        row["pre_redundancy_keep"] = bool(
            (
                row["effective_structural_pass"]
                and row["exposure_precision_pass"]
                and row["stability_pass"]
                and row["combined_score_pass"]
                and row["context_pass"]
                and (
                    (not bool(config.require_model_helpfulness_for_selection))
                    or row["model_helpfulness_pass"]
                )
            )
            or row["risk_budget_scaler_pass"]
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = _finalize_acceptance_metrics(out, config)
    return out.sort_values("rank_score", ascending=False, kind="mergesort")


def _finalize_acceptance_metrics(
    metrics: pd.DataFrame,
    config: RegimeFeatureLGBMFilterConfig,
) -> pd.DataFrame:
    """Apply cross-feature acceptance thresholds after per-feature aggregation."""

    if metrics.empty:
        return metrics
    out = metrics.copy()

    def bool_col(name: str, default: bool = False) -> pd.Series:
        if name in out.columns:
            return out[name].astype(bool)
        return pd.Series(bool(default), index=out.index)

    def str_col(name: str, default: str = "") -> pd.Series:
        if name in out.columns:
            return out[name].astype(str)
        return pd.Series(str(default), index=out.index)

    signal_values = pd.to_numeric(out.get("signal_uplift_mean_abs", 0.0), errors="coerce").fillna(0.0)
    budget_score = pd.to_numeric(out.get("risk_budget_scaler_score", 0.0), errors="coerce").fillna(0.0)
    budget_hit_rate = pd.to_numeric(out.get("risk_budget_scaler_hit_rate", 0.0), errors="coerce").fillna(0.0)
    budget_available = pd.to_numeric(out.get("risk_budget_scaler_available_rate", 0.0), errors="coerce").fillna(0.0)
    is_budget_portfolio = str_col("source").eq("context_portfolio") | str_col("feature").str.contains(
        "ctx_portfolio_",
        case=False,
        regex=False,
    )
    finite_signal = signal_values.to_numpy(dtype=np.float64)
    finite_signal = finite_signal[np.isfinite(finite_signal)]
    if finite_signal.size:
        quantile_threshold = float(
            np.nanquantile(
                finite_signal,
                float(np.clip(config.signal_uplift_top_quantile, 0.0, 1.0)),
            )
        )
    else:
        quantile_threshold = 0.0
    signal_threshold = max(float(config.signal_uplift_min_abs_mean), quantile_threshold)
    out["signal_uplift_rank_threshold"] = float(signal_threshold)
    out["signal_uplift_top_quantile_pass"] = signal_values >= float(signal_threshold)
    out["signal_uplift_context_pass"] = (
        out["signal_uplift_top_quantile_pass"].astype(bool)
        & pd.to_numeric(out.get("signal_uplift_context_pass_rate", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.signal_uplift_min_fold_fraction))
    )
    out["risk_budget_scaler_pass"] = (
        bool_col("risk_budget_scaler_pass")
        | (
            bool(config.risk_budget_scaler_enabled)
            & is_budget_portfolio
            & budget_available.gt(0.0)
            & budget_hit_rate.ge(float(config.risk_budget_scaler_min_fold_fraction))
            & pd.to_numeric(out.get("median_risk_budget_scaled_hr_lift", 0.0), errors="coerce")
            .fillna(0.0)
            .gt(float(config.risk_budget_scaler_min_scaled_hr_lift))
            & pd.to_numeric(out.get("median_risk_budget_high_low_hr_lift", 0.0), errors="coerce")
            .fillna(0.0)
            .gt(float(config.risk_budget_scaler_min_high_low_hr_lift))
            & pd.to_numeric(out.get("median_risk_budget_failure_avoidance", 0.0), errors="coerce")
            .fillna(0.0)
            .ge(float(config.risk_budget_scaler_min_failure_avoidance))
            & pd.to_numeric(out.get("risk_budget_monotonic_consistency_mean", 0.0), errors="coerce")
            .fillna(0.0)
            .ge(0.50)
        )
    )

    out["oof_failure_alignment_pass"] = (
        pd.to_numeric(out.get("oof_available_rate", 0.0), errors="coerce").fillna(0.0).gt(0.0)
        & pd.to_numeric(out.get("median_oof_failure_lift", 0.0), errors="coerce")
        .fillna(0.0)
        .gt(float(config.risk_gate_min_oof_failure_lift))
        & pd.to_numeric(out.get("median_oof_wrong_rate_lift", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.risk_gate_min_oof_wrong_rate_lift))
        & pd.to_numeric(out.get("median_oof_residual_abs_lift", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.risk_gate_min_oof_residual_abs_lift))
        & pd.to_numeric(out.get("oof_failure_lift_positive_rate", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.risk_gate_min_oof_failure_positive_rate))
    )
    negative_rate_pass = (
        pd.to_numeric(out.get("HR_lift_negative_rate", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.risk_gate_min_negative_fold_fraction))
        & pd.to_numeric(out.get("stability_lift_negative_rate", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.risk_gate_min_negative_fold_fraction))
    )
    structural_support_pass = (
        pd.to_numeric(out.get("effective_structural_pass_rate", 0.0), errors="coerce")
        .fillna(0.0)
        .ge(float(config.risk_gate_min_effective_structural_pass_rate))
    )
    if bool(config.risk_gate_require_oof_alignment):
        oof_condition = out["oof_failure_alignment_pass"].astype(bool)
    else:
        oof_condition = (
            pd.to_numeric(out.get("oof_available_rate", 0.0), errors="coerce").fillna(0.0).le(0.0)
            | out["oof_failure_alignment_pass"].astype(bool)
            | out["signal_uplift_context_pass"].astype(bool)
        )
    out["risk_gate_oof_failure_lift"] = pd.to_numeric(
        out.get("risk_gate_oof_failure_lift", out.get("median_oof_failure_lift", 0.0)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    out["risk_gate_oof_wrong_rate_lift"] = pd.to_numeric(
        out.get("risk_gate_oof_wrong_rate_lift", out.get("median_oof_wrong_rate_lift", 0.0)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    out["risk_gate_oof_residual_abs_lift"] = pd.to_numeric(
        out.get("risk_gate_oof_residual_abs_lift", out.get("median_oof_residual_abs_lift", 0.0)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    out["risk_gate_defensive_score"] = pd.to_numeric(
        out.get("risk_gate_acceptance_score", 0.0),
        errors="coerce",
    ).fillna(0.0) * pd.to_numeric(out.get("oof_available_rate", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["risk_gate_defensive_pass"] = (
        out["oof_failure_alignment_pass"].astype(bool)
        & out["risk_gate_oof_failure_lift"].ge(float(config.risk_gate_min_oof_failure_lift))
        & out["risk_gate_oof_wrong_rate_lift"].ge(float(config.risk_gate_min_oof_wrong_rate_lift))
        & out["risk_gate_oof_residual_abs_lift"].ge(float(config.risk_gate_min_oof_residual_abs_lift))
    )
    out["risk_gate_acceptance_pass"] = (
        str_col("context_role").eq("risk_gate")
        & pd.to_numeric(out.get("risk_gate_hr_avoidance", 0.0), errors="coerce")
        .fillna(0.0)
        .gt(float(config.risk_gate_min_hr_avoidance))
        & pd.to_numeric(out.get("risk_gate_stability_avoidance", 0.0), errors="coerce")
        .fillna(0.0)
        .gt(float(config.risk_gate_min_stability_avoidance))
        & bool_col("HR_risk_gate_pass")
        & bool_col("stability_risk_gate_pass")
        & bool_col("combined_score_pass")
        & out["risk_gate_defensive_pass"].astype(bool)
        & negative_rate_pass
        & structural_support_pass
        & oof_condition
    )
    out["risk_gate_acceptance_reason"] = np.where(
        out["risk_gate_acceptance_pass"].astype(bool),
        "accepted_oof_aligned_risk_gate",
        "rejected_by_strict_risk_gate_acceptance",
    )
    out["opportunity_context_pass"] = (
        str_col("context_role").eq("opportunity_gate")
        & bool_col("HR_opportunity_pass")
        & bool_col("stability_opportunity_pass")
        & out["signal_uplift_context_pass"].astype(bool)
        & (
            bool_col("effective_structural_pass")
            | bool_col("context_pass")
            | out["oof_failure_alignment_pass"].astype(bool)
        )
    )
    out["context_helper_candidate_pass"] = (
        (
            out["signal_uplift_context_pass"].astype(bool)
            | out["risk_budget_scaler_pass"].astype(bool)
        )
        & ~out["risk_gate_acceptance_pass"].astype(bool)
        & (
            out["opportunity_context_pass"].astype(bool)
            | bool_col("effective_structural_pass")
            | out["oof_failure_alignment_pass"].astype(bool)
            | out["risk_budget_scaler_pass"].astype(bool)
            | str_col("source").eq("signal_regime_interaction")
        )
    )
    out["context_helper_reason"] = np.select(
        [
            out["risk_budget_scaler_pass"].astype(bool),
            out["opportunity_context_pass"].astype(bool),
            out["oof_failure_alignment_pass"].astype(bool),
            bool_col("effective_structural_pass"),
            str_col("source").eq("signal_regime_interaction"),
        ],
        [
            "monotonic_risk_budget_scaler",
            "opportunity_gate_with_conditional_signal_uplift",
            "oof_failure_aligned_context_helper",
            "structurally_used_context_helper",
            "signal_regime_interaction_context_helper",
        ],
        default="not_context_helper",
    )
    out["opportunity_helpfulness_score"] = (
        pd.to_numeric(out.get("median_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(out.get("score_hit_rate", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(out.get("effective_structural_pass_rate", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        * (1.0 + signal_values)
    )
    out["model_helpfulness_pass"] = (
        (
            out["opportunity_context_pass"].astype(bool)
        )
        | out["risk_gate_acceptance_pass"].astype(bool)
        | out["risk_budget_scaler_pass"].astype(bool)
    )
    out["model_helpfulness_score"] = np.where(
        str_col("context_role").eq("risk_gate"),
        out["risk_gate_defensive_score"].to_numpy(dtype=np.float64, copy=False),
        out["opportunity_helpfulness_score"].to_numpy(dtype=np.float64, copy=False),
    )
    fallback_helpfulness = (
        pd.to_numeric(out.get("signal_uplift_context_pass_rate", 0.0), errors="coerce").fillna(0.0)
        * signal_values
    )
    out["model_helpfulness_score"] = np.maximum(
        pd.to_numeric(out["model_helpfulness_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        fallback_helpfulness.to_numpy(dtype=np.float64, copy=False),
    )
    out["model_helpfulness_score"] = np.maximum(
        pd.to_numeric(out["model_helpfulness_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        budget_score.to_numpy(dtype=np.float64, copy=False),
    )
    rank_multiplier = np.maximum(
        pd.to_numeric(out.get("effective_structural_pass_rate", 0.0), errors="coerce").fillna(0.0).clip(lower=1e-6).to_numpy(dtype=np.float64),
        np.where(out["risk_budget_scaler_pass"].astype(bool), budget_hit_rate.to_numpy(dtype=np.float64, copy=False), 0.0),
    )
    out["rank_score"] = (
        pd.to_numeric(out["model_helpfulness_score"], errors="coerce").fillna(0.0)
        * rank_multiplier
    )
    if bool(config.require_model_helpfulness_for_selection):
        out["pre_redundancy_keep"] = (
            bool_col("pre_redundancy_keep") & out["model_helpfulness_pass"].astype(bool)
        ) | out["risk_budget_scaler_pass"].astype(bool)
    return out


def _redundancy_prune(
    x: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    threshold: float,
) -> set[str]:
    kept: list[str] = []
    for feature in metrics.sort_values("rank_score", ascending=False, kind="mergesort")["feature"].astype(str):
        if feature not in x.columns:
            continue
        drop = False
        for existing in kept:
            corr = x[[feature, existing]].corr(method="spearman").iloc[0, 1]
            if np.isfinite(corr) and abs(float(corr)) > float(threshold):
                drop = True
                break
        if not drop:
            kept.append(feature)
    return set(kept)


def _source_prune(metrics: pd.DataFrame, config: RegimeFeatureLGBMFilterConfig) -> tuple[set[str], pd.DataFrame]:
    if metrics.empty:
        return set(), pd.DataFrame()
    source = (
        metrics.groupby("source", dropna=False)["rank_score"]
        .sum()
        .reset_index(name="source_usefulness")
        .sort_values("source_usefulness", ascending=False, kind="mergesort")
    )
    total = float(source["source_usefulness"].sum())
    if total <= 0.0:
        source["source_share"] = 0.0
        source["cumulative_source_share"] = 0.0
        source["source_keep"] = True
        return set(source["source"].astype(str)), source
    source["source_share"] = source["source_usefulness"] / total
    source["cumulative_source_share"] = source["source_share"].cumsum()
    cutoff = float(np.clip(config.source_cumulative_usefulness_keep, 0.01, 1.0))
    keep = []
    cumulative_before = 0.0
    for row in source.itertuples(index=False):
        keep.append(str(row.source))
        cumulative_before = float(row.cumulative_source_share)
        if cumulative_before >= cutoff:
            break
    source["source_keep"] = source["source"].astype(str).isin(set(keep))
    return set(keep), source


def select_regime_lgbm_addon_features(
    frame: pd.DataFrame,
    y: Any,
    *,
    base_features: Sequence[str],
    regime_features: Sequence[str],
    timestamps: Any = None,
    sample_weight: Any = None,
    base_oof_pred: Any = None,
    reused_model_params: Mapping[str, Any] | None = None,
    source_map: Mapping[str, str] | None = None,
    config: RegimeFeatureLGBMFilterConfig = RegimeFeatureLGBMFilterConfig(),
) -> RegimeFeatureLGBMFilterResult:
    """Evaluate new regime features without rerunning feature selection/HPO."""

    lgb = _lightgbm_module()
    if lgb is None:
        return RegimeFeatureLGBMFilterResult(
            selected_features=[],
            feature_metrics=pd.DataFrame(),
            fold_metrics=pd.DataFrame(),
            source_metrics=pd.DataFrame(),
            diagnostics={"status": "lightgbm_unavailable", "config": _jsonable(asdict(config))},
        )
    base = [str(c) for c in dict.fromkeys(base_features) if str(c) in frame.columns]
    regime = [str(c) for c in dict.fromkeys(regime_features) if str(c) in frame.columns and str(c) not in set(base)]
    x = _prepare_matrix(frame, base + regime)
    y_bin = _binary_target(y)
    oof_pred = _coerce_oof_prediction(base_oof_pred, len(x))
    if len(x) != len(y_bin) or not base or not regime:
        return RegimeFeatureLGBMFilterResult(
            selected_features=[],
            feature_metrics=pd.DataFrame(),
            fold_metrics=pd.DataFrame(),
            source_metrics=pd.DataFrame(),
            diagnostics={
                "status": "invalid_input",
                "base_feature_count": int(len(base)),
                "regime_feature_count": int(len(regime)),
                "row_count": int(len(frame)),
            },
        )
    folds = _period_folds(
        len(x),
        y_bin,
        timestamps,
        n_folds=int(config.n_folds),
        sample_fraction=float(config.fold_sample_fraction),
        max_rows=int(config.max_rows),
        random_state=int(config.random_state),
        stratified_period_bins=int(config.stratified_period_bins),
    )
    if not folds:
        return RegimeFeatureLGBMFilterResult(
            selected_features=[],
            feature_metrics=pd.DataFrame(),
            fold_metrics=pd.DataFrame(),
            source_metrics=pd.DataFrame(),
            diagnostics={"status": "no_valid_folds", "config": _jsonable(asdict(config))},
        )
    fold_rows: list[dict[str, Any]] = []
    params = _params(config, reused_model_params or {})
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        x_fold = x.copy()
        shadow_cols: list[str] = []
        if bool(config.use_shadow_gain):
            x_fold, shadow_cols = _add_shadow_features(
                x_fold,
                regime,
                random_state=int(config.random_state) + fold_i * 100,
            )
        train_cols = base + regime + shadow_cols
        model = lgb.LGBMClassifier(**{**params, "random_state": int(config.random_state) + fold_i})
        model.fit(
            x_fold.iloc[train_idx][train_cols],
            y_bin[train_idx],
            sample_weight=(sw[train_idx] if sw is not None and len(sw) == len(x) else None),
        )
        pred = model.predict_proba(x_fold.iloc[val_idx][train_cols])[:, 1].astype(np.float32)
        trees, feature_names = _collect_tree_structures(model)
        gain, split, tree_use = _gain_split_maps(trees, feature_names)
        paths = []
        for tree in trees:
            paths.extend(_tree_paths(tree, feature_names))
        path_count = max(len(paths), 1)
        non_regime_gains = [
            float(v)
            for k, v in gain.items()
            if k not in set(regime) and not str(k).startswith("__shadow__") and float(v) > 0.0
        ]
        gain_denom = float(np.nanmedian(non_regime_gains)) if non_regime_gains else 1.0
        shadow_gains = [float(gain.get(col, 0.0)) for col in shadow_cols]
        shadow_p85 = float(np.nanpercentile(shadow_gains, 85.0)) if shadow_gains else 0.0
        exposures, context = _route_exposures(
            model,
            x_fold.iloc[val_idx][train_cols],
            regime,
            base,
            max_rows=int(config.route_max_rows),
            random_state=int(config.random_state) + fold_i * 17,
        )
        contribution_exposure = _absolute_contribution_exposure(
            model,
            x_fold.iloc[val_idx][train_cols],
            regime,
        )
        signal_orientations = (
            _select_signal_orientations(
                x,
                base,
                y_bin,
                train_idx,
                max_features=int(config.signal_uplift_max_signal_features),
            )
            if bool(config.conditional_signal_uplift_enabled)
            else {}
        )
        y_val = y_bin[val_idx]
        oof_val = oof_pred[val_idx] if len(oof_pred) == len(x) else np.full(len(val_idx), np.nan, dtype=np.float32)
        ts_val = np.asarray(timestamps)[val_idx] if timestamps is not None and len(np.asarray(timestamps)) == len(x) else None
        for feature in regime:
            feature_gain = float(gain.get(feature, 0.0))
            feature_paths = sum(1 for path in paths if feature in set(path.get("features", [])))
            route_exposure = exposures.get(feature, np.zeros(len(val_idx), dtype=np.float32))
            exposure = contribution_exposure.get(feature)
            if exposure is None or not np.isfinite(exposure).any() or float(np.nanmax(exposure)) <= 0.0:
                exposure = route_exposure
            threshold = (
                float(np.nanquantile(exposure, float(config.exposure_quantile)))
                if np.isfinite(exposure).any()
                else 0.0
            )
            high = exposure > max(threshold, 0.0)
            low = ~high
            signal_uplift = _conditional_signal_uplift(
                x,
                y_val,
                val_idx,
                feature,
                signal_orientations,
                low_quantile=float(config.signal_uplift_low_quantile),
                high_quantile=float(config.signal_uplift_high_quantile),
                min_rows=int(config.min_group_rows),
            )
            oof_failure = _oof_failure_mode_metrics(
                y_val,
                oof_val,
                high,
                low,
                min_rows=int(config.min_group_rows),
            )
            budget_scaler = (
                _risk_budget_scaler_metrics(
                    x,
                    y_val,
                    val_idx,
                    pred,
                    oof_val,
                    feature,
                    timestamps=ts_val,
                    min_rows=int(config.min_group_rows),
                    quantile_bins=int(config.risk_budget_scaler_quantile_bins),
                )
                if bool(config.risk_budget_scaler_enabled)
                else {}
            )
            hr_high = _hr_star(y_val, pred, high, int(config.min_group_rows))
            hr_low = _hr_star(y_val, pred, low, int(config.min_group_rows))
            stab_high = _weekly_stability(y_val, pred, high, ts_val, int(config.min_group_rows))
            stab_low = _weekly_stability(y_val, pred, low, ts_val, int(config.min_group_rows))
            hr_lift = float(hr_high - hr_low)
            stability_lift = float(stab_high - stab_low)
            same_direction = bool(
                (hr_lift > 0.0 and stability_lift > 0.0)
                or (hr_lift < 0.0 and stability_lift < 0.0)
            )
            score = (
                abs(hr_lift * stability_lift)
                if same_direction
                else -abs(hr_lift * stability_lift)
            )
            ctx = context.get(feature, {})
            min_tree_use = max(
                float(config.structural_min_trees_using),
                float(config.structural_min_tree_fraction) * float(len(trees)),
            )
            structural_pass = bool(
                float(tree_use.get(feature, 0)) >= min_tree_use
                and float(feature_paths / path_count) >= float(config.structural_min_leaf_path_share)
                and float(ctx.get("sample_path_exposure", 0.0)) >= float(config.structural_min_sample_path_exposure)
                and float(feature_gain / max(gain_denom, 1e-12)) >= float(config.structural_min_gain_ratio)
                and ((not bool(config.use_shadow_gain)) or feature_gain > shadow_p85)
            )
            context_pass = bool(
                (not _is_train_base(config.objective_mode))
                or (not bool(config.base_context_filter_enabled))
            )
            context_interaction_pass = bool(
                float(ctx.get("interaction_path_share", 0.0)) >= float(config.interaction_path_share_min)
                and float(ctx.get("signal_dense_interaction_share", 0.0)) >= float(config.signal_dense_interaction_share_min)
                and float(ctx.get("signal_gain_share", 0.0)) >= float(config.signal_gain_share_min)
                and float(ctx.get("signal_exposure_share", 0.0)) >= float(config.signal_exposure_share_min)
            )
            coarse_min_tree_use = max(
                1.0,
                float(config.coarse_gate_min_tree_fraction_of_structural) * float(min_tree_use),
            )
            coarse_gate_context_pass = bool(
                bool(config.coarse_gate_context_enabled)
                and float(tree_use.get(feature, 0)) >= coarse_min_tree_use
                and float(ctx.get("sample_path_exposure", 0.0)) >= float(config.structural_min_sample_path_exposure)
                and float(feature_gain / max(gain_denom, 1e-12)) >= float(config.structural_min_gain_ratio)
                and ((not bool(config.use_shadow_gain)) or feature_gain > shadow_p85)
            )
            context_pass = bool(
                context_pass
                or context_interaction_pass
                or coarse_gate_context_pass
            )
            fold_rows.append(
                {
                    "fold": int(fold_i),
                    "feature": str(feature),
                    "tree_count": int(len(trees)),
                    "n_trees_using": int(tree_use.get(feature, 0)),
                    "leaf_path_share": float(feature_paths / path_count),
                    "sample_path_exposure": float(ctx.get("sample_path_exposure", 0.0)),
                    "gain": feature_gain,
                    "split": int(split.get(feature, 0)),
                    "gain_ratio": float(feature_gain / max(gain_denom, 1e-12)),
                    "shadow_gain_p85": shadow_p85,
                    "HR_high": float(hr_high),
                    "HR_low": float(hr_low),
                    "HR_lift": hr_lift,
                    "stability_high": float(stab_high),
                    "stability_low": float(stab_low),
                    "stability_lift": stability_lift,
                    "score": score,
                    "context_role": (
                        "opportunity_gate"
                        if hr_lift > 0.0 and stability_lift > 0.0
                        else "risk_gate"
                        if hr_lift < 0.0 and stability_lift < 0.0
                        else "mixed_or_weak_context"
                    ),
                    "structural_pass": structural_pass,
                    "interaction_path_share": float(ctx.get("interaction_path_share", 0.0)),
                    "signal_dense_interaction_share": float(ctx.get("signal_dense_interaction_share", 0.0)),
                    "signal_gain_share": float(ctx.get("signal_gain_share", 0.0)),
                    "signal_exposure_share": float(ctx.get("signal_exposure_share", 0.0)),
                    "context_interaction_pass": context_interaction_pass,
                    "coarse_gate_context_pass": coarse_gate_context_pass,
                    "context_pass": context_pass,
                    **signal_uplift,
                    **oof_failure,
                    **budget_scaler,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                }
            )
    fold_metrics = pd.DataFrame(fold_rows)
    feature_metrics = _aggregate_feature_metrics(fold_metrics, regime, config, source_map)
    if feature_metrics.empty:
        return RegimeFeatureLGBMFilterResult(
            selected_features=[],
            feature_metrics=feature_metrics,
            fold_metrics=fold_metrics,
            source_metrics=pd.DataFrame(),
            diagnostics={"status": "no_feature_metrics", "fold_count": int(len(folds))},
        )
    keep = feature_metrics.loc[feature_metrics["pre_redundancy_keep"].astype(bool)].copy()
    redundancy_kept = _redundancy_prune(
        x,
        keep,
        threshold=float(config.redundancy_spearman_threshold),
    )
    feature_metrics["redundancy_keep"] = feature_metrics["feature"].astype(str).isin(redundancy_kept)
    keep = feature_metrics.loc[feature_metrics["pre_redundancy_keep"].astype(bool) & feature_metrics["redundancy_keep"].astype(bool)].copy()
    kept_sources, source_metrics = _source_prune(keep, config)
    feature_metrics["source_keep"] = feature_metrics["source"].astype(str).isin(kept_sources)
    final = feature_metrics.loc[
        feature_metrics["pre_redundancy_keep"].astype(bool)
        & feature_metrics["redundancy_keep"].astype(bool)
        & feature_metrics["source_keep"].astype(bool)
    ].sort_values("rank_score", ascending=False, kind="mergesort")
    selected = final["feature"].astype(str).head(int(config.top_n)).tolist()
    diagnostics = {
        "status": "completed",
        "config": _jsonable(asdict(config)),
        "base_feature_count": int(len(base)),
        "candidate_regime_feature_count": int(len(regime)),
        "fold_count": int(len(folds)),
        "pre_redundancy_keep_count": int(feature_metrics["pre_redundancy_keep"].sum()),
        "redundancy_keep_count": int(feature_metrics["redundancy_keep"].sum()),
        "source_keep_count": int(len(kept_sources)),
        "selected_feature_count": int(len(selected)),
        "selected_features": list(selected),
        "risk_gate_acceptance_pass_count": int(feature_metrics.get("risk_gate_acceptance_pass", pd.Series(False)).sum()),
        "risk_budget_scaler_pass_count": int(feature_metrics.get("risk_budget_scaler_pass", pd.Series(False)).sum()),
        "opportunity_context_pass_count": int(feature_metrics.get("opportunity_context_pass", pd.Series(False)).sum()),
        "context_helper_candidate_pass_count": int(
            feature_metrics.get("context_helper_candidate_pass", pd.Series(False)).sum()
        ),
        "reused_param_keys": sorted(str(k) for k in (reused_model_params or {}).keys()),
        "oof_prediction_coverage": float(np.mean(np.isfinite(oof_pred))) if len(oof_pred) == len(x) else 0.0,
        "conditional_signal_uplift_enabled": bool(config.conditional_signal_uplift_enabled),
    }
    return RegimeFeatureLGBMFilterResult(
        selected_features=selected,
        feature_metrics=feature_metrics.reset_index(drop=True),
        fold_metrics=fold_metrics.reset_index(drop=True),
        source_metrics=source_metrics.reset_index(drop=True) if isinstance(source_metrics, pd.DataFrame) else pd.DataFrame(),
        diagnostics=diagnostics,
    )
