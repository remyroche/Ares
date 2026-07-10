#!/usr/bin/env python3
"""Ablate regime x archetype calibration model families.

The base/meta predictions are fixed.  HPO is performed on the first evaluation
month only; the winning calibration family and hyperparameters are then replayed
month-by-month with walk-forward training data and exported as the default
``per_regime_archetype_calibration_v1`` artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except Exception:  # pragma: no cover
    ExplainableBoostingRegressor = None  # type: ignore[assignment]

from extreme_price_movements.regime_ev_calibration import CALIBRATION_POLICY_ID
from scripts.report_meta_oos_regime_calibration import (
    ARCH_COL,
    DEFAULT_HANDOFF,
    DEFAULT_META_RUN,
    DEFAULT_OUT,
    KEYS,
    OUTCOME_COLS,
    REGIME_SPECS,
    SCORE_COL,
    _available_feature_cols,
    _derive_joined_features,
    _derive_prediction_features,
    _load_feature_slice,
    _load_predictions,
    _safe_numeric,
    _schema_cols,
)


TOP10_CUT = 0.90
DEFAULT_ROLLING_CALIBRATION_OUT = DEFAULT_OUT.parent / "meta_oos_regime_calibration_rolling60d_oos15_20260708"

ARCHETYPE_PREFIX_ALIASES = [
    {"prefix": "long__long_mixed_gmm_", "alias": "long_mixed_wideslow_tentative"},
    {"prefix": "long__compression_release", "alias": "long_volcompression_wideslow_candidate"},
    {"prefix": "long__loud_breakout_impulse", "alias": "long_breakout_diagnostic_candidate"},
    {"prefix": "long__dirty_shock_avoid", "alias": "long_dirtyavoid_sparse_questionable"},
    {"prefix": "long__volatile_mean_reversion", "alias": "long_mixed_wideslow_tentative"},
    {"prefix": "long__retest_reversal", "alias": "long_mixed_wideslow_tentative"},
    {"prefix": "short__late_run_continuation", "alias": "short_mixed_clean_path"},
    {"prefix": "short__quiet_continuation", "alias": "short_mixed_clean_path"},
    {"prefix": "short__retest_reversal", "alias": "short_default_clean_path"},
    {"prefix": "short__run_entry", "alias": "short_breakout_precision"},
    {"prefix": "short__ambiguous_none", "alias": "short_mixed_clean_path"},
    {"prefix": "short__dirty_shock_avoid", "alias": "short_mixed_clean_path"},
    {"prefix": "short__volatile_mean_reversion", "alias": "short_default_clean_path"},
    {"prefix": "short__compression_release", "alias": "short_default_clean_path"},
    {"prefix": "short__loud_breakout_impulse", "alias": "short_breakout_precision"},
]


def _risk_target(frame: pd.DataFrame) -> pd.Series:
    ev = _safe_numeric(frame["ev_after_1pct"]).fillna(0.0)
    bad = _safe_numeric(frame["full_path_bad_mae_1r"]).fillna(0.0)
    timeout = _safe_numeric(frame["timeout"]).fillna(0.0)
    dirty = _safe_numeric(frame["dirty_positive"]).fillna(0.0)
    clean = _safe_numeric(frame["clean_exec"]).fillna(0.0)
    y = -ev + 0.006 * bad + 0.006 * timeout + 0.004 * dirty - 0.004 * clean
    return y.clip(-0.08, 0.08).astype("float32")


def _top10_objective(frame: pd.DataFrame, score_col: str) -> Dict[str, float]:
    rank = _safe_numeric(frame[score_col]).rank(pct=True, method="first")
    top = frame.loc[rank.ge(TOP10_CUT)].copy()
    ev = _safe_numeric(top["ev_after_1pct"])
    if top.empty:
        return {"top10_ev": float("nan"), "top10_q15_day_ev": float("nan"), "objective": -1e9}
    day = pd.to_datetime(top["__ts__"], utc=True, errors="coerce").dt.date.astype(str)
    day_ev = ev.groupby(day).mean()
    q15 = float(day_ev.quantile(0.15)) if len(day_ev) else float("nan")
    mean_ev = float(ev.mean())
    objective = 0.7 * mean_ev + 0.3 * (q15 if np.isfinite(q15) else mean_ev)
    return {
        "top10_rows": int(len(top)),
        "top10_ev": mean_ev,
        "top10_q15_day_ev": q15,
        "top10_clean": float(_safe_numeric(top["clean_exec"]).mean()),
        "top10_bad_mae": float(_safe_numeric(top["full_path_bad_mae_1r"]).mean()),
        "top10_timeout": float(_safe_numeric(top["timeout"]).mean()),
        "objective": float(objective),
    }


def _loss_sequence_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "ev_after_1pct" not in frame.columns:
        return {
            "hit_rate": float("nan"),
            "loss_rate": float("nan"),
            "loss_autocorr_lag1": float("nan"),
            "max_loss_streak": 0.0,
            "mean_loss_streak": float("nan"),
        }
    ordered = frame.sort_values(["__ts__", "__symbol__", "side_name"], kind="mergesort")
    ev = _safe_numeric(ordered["ev_after_1pct"]).fillna(0.0)
    loss = ev.le(0.0).to_numpy(dtype=np.int8, copy=False)
    hit = ev.gt(0.0).to_numpy(dtype=np.int8, copy=False)
    if loss.size >= 3 and np.unique(loss).size >= 2 and np.unique(loss[:-1]).size >= 2 and np.unique(loss[1:]).size >= 2:
        autocorr = float(np.corrcoef(loss[:-1].astype("float64"), loss[1:].astype("float64"))[0, 1])
    else:
        autocorr = float("nan")
    streaks: list[int] = []
    cur = 0
    for value in loss:
        if int(value):
            cur += 1
        elif cur:
            streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    return {
        "hit_rate": float(hit.mean()) if hit.size else float("nan"),
        "loss_rate": float(loss.mean()) if loss.size else float("nan"),
        "loss_autocorr_lag1": autocorr,
        "max_loss_streak": float(max(streaks) if streaks else 0),
        "mean_loss_streak": float(np.mean(streaks)) if streaks else 0.0,
    }


def _metric_rows(frame: pd.DataFrame, score_col: str, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, cut in {"top10": 0.90, "top20": 0.80, "top30": 0.70}.items():
        rank = _safe_numeric(frame[score_col]).groupby(frame["month"]).rank(pct=True, method="first")
        top = frame.loc[rank.ge(cut)]
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", top)]
        for col in ["month", "week_start", "side_name", ARCH_COL]:
            groups.extend((col, str(k), g) for k, g in top.groupby(col, dropna=False, observed=True))
        for group, group_value, g in groups:
            loss_metrics = _loss_sequence_metrics(g)
            rows.append(
                {
                    "arm": arm,
                    "top_scope": scope,
                    "group": group,
                    "group_value": group_value,
                    "rows": int(len(g)),
                    "mean_ev_after_1pct": float(_safe_numeric(g["ev_after_1pct"]).mean()) if len(g) else np.nan,
                    "sum_ev_after_1pct": float(_safe_numeric(g["ev_after_1pct"]).sum()) if len(g) else np.nan,
                    "clean_exec_rate": float(_safe_numeric(g["clean_exec"]).mean()) if len(g) else np.nan,
                    "dirty_positive_rate": float(_safe_numeric(g["dirty_positive"]).mean()) if len(g) else np.nan,
                    "full_path_bad_mae_rate": float(_safe_numeric(g["full_path_bad_mae_1r"]).mean()) if len(g) else np.nan,
                    "timeout_rate": float(_safe_numeric(g["timeout"]).mean()) if len(g) else np.nan,
                    **loss_metrics,
                }
            )
    return rows


def _feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _trade_relevant_focus_mask(frame: pd.DataFrame) -> pd.Series:
    if SCORE_COL not in frame.columns:
        return pd.Series(True, index=frame.index)
    rank = _safe_numeric(frame[SCORE_COL]).rank(pct=True, method="first")
    mask = rank.ge(0.80)
    return mask if int(mask.sum()) >= 50 else pd.Series(True, index=frame.index)


def _select_feature_cols(
    train: pd.DataFrame,
    feature_cols: list[str],
    max_features: int = 10,
    *,
    allow_binary: bool = False,
    focus_mask: pd.Series | None = None,
) -> list[str]:
    if int(max_features) <= 0:
        return []
    sample = train
    if focus_mask is not None:
        mask = focus_mask.reindex(train.index).fillna(False)
        if int(mask.sum()) >= 50:
            sample = train.loc[mask]
    y = _risk_target(sample)
    if y.nunique(dropna=True) < 2:
        return []
    scored: list[tuple[float, str]] = []
    for col in feature_cols:
        x = _safe_numeric(sample[col]) if col in sample.columns else pd.Series(np.nan, index=sample.index)
        if x.notna().mean() < 0.40:
            continue
        nunique = int(x.nunique(dropna=True))
        if allow_binary:
            if nunique < 2:
                continue
            counts = x.dropna().value_counts()
            if nunique <= 2 and (counts.empty or int(counts.min()) < 5):
                continue
        elif nunique < 5:
            continue
        corr = abs(float(x.corr(y, method="spearman")))
        if np.isfinite(corr):
            scored.append((corr, col))
    scored.sort(reverse=True)
    return [col for _, col in scored[:max_features]]


def _sample_weight(frame: pd.DataFrame) -> np.ndarray:
    score = _safe_numeric(frame[SCORE_COL]).fillna(0.5).clip(0.05, 1.0)
    return score.to_numpy(dtype="float32")


def _make_spline_model(params: dict[str, Any]) -> Pipeline:
    variant = str(params.get("variant") or "two_sided")
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if variant in {"u_shaped", "convex_risk"}:
        steps.append(("abs", FunctionTransformer(np.abs, validate=False)))
    steps.extend(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=int(params.get("n_splines", 6)),
                    degree=int(params.get("degree", 3)),
                    include_bias=False,
                    knots="quantile",
                ),
            ),
            (
                "model",
                ElasticNet(
                    alpha=float(params.get("alpha", 1.0)),
                    l1_ratio=float(params.get("l1_ratio", 0.5)),
                    max_iter=3000,
                    positive=variant in {"monotone", "convex_risk"},
                    random_state=42,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def _make_gam_model(params: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if int(params.get("interactions", 0)) > 0:
        steps.append(
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            )
        )
    steps.extend(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=int(params.get("n_splines", 6)),
                    degree=int(params.get("degree", 3)),
                    include_bias=False,
                    knots="quantile",
                ),
            ),
            ("model", Ridge(alpha=float(params.get("lambda", 30.0)), random_state=42)),
        ]
    )
    return Pipeline(steps)


def _make_ebm_model(params: dict[str, Any]) -> Any:
    if ExplainableBoostingRegressor is None:
        raise RuntimeError("interpret is unavailable; cannot fit EBM calibration")
    return ExplainableBoostingRegressor(
        max_bins=int(params.get("max_bins", 128)),
        interactions=int(params.get("interactions", 0)),
        max_rounds=500,
        early_stopping_rounds=20,
        min_samples_leaf=200,
        max_leaves=int(params.get("max_leaves", 3)),
        outer_bags=4,
        inner_bags=0,
        n_jobs=1,
        random_state=42,
    )


def _make_model(family: str, params: dict[str, Any]) -> Any:
    if family == "bucket":
        return _make_spline_model({"variant": "two_sided", "alpha": 10.0, "l1_ratio": 0.9, "n_splines": 5, "degree": 1})
    if family == "spline":
        return _make_spline_model(params)
    if family == "gam":
        return _make_gam_model(params)
    if family == "ebm":
        return _make_ebm_model(params)
    raise ValueError(f"unknown calibration family: {family}")


def _safe_name(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "").replace("+", "")


def _month_start(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month}-01", tz="UTC")


def _month_end(month: str) -> pd.Timestamp:
    return _month_start(month) + pd.offsets.MonthBegin(1)


def _rolling_windows(
    eval_months: list[str],
    *,
    train_days: int,
    oos_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    if not eval_months:
        return []
    start = min(_month_start(month) for month in eval_months)
    end = max(_month_end(month) for month in eval_months)
    cursor = start
    windows: list[dict[str, Any]] = []
    while cursor < end:
        valid_to = min(cursor + pd.Timedelta(days=int(oos_days)), end)
        window_id = f"{cursor.strftime('%Y%m%d')}_{valid_to.strftime('%Y%m%d')}"
        windows.append(
            {
                "window_id": window_id,
                "train_start": cursor - pd.Timedelta(days=int(train_days)),
                "train_end": cursor,
                "valid_from": cursor,
                "valid_to": valid_to,
            }
        )
        cursor += pd.Timedelta(days=int(step_days))
    return windows


def _fit_predict_window(
    frame: pd.DataFrame,
    window: dict[str, Any],
    feature_cols: list[str],
    family: str,
    params: dict[str, Any],
    *,
    candidate_feature_map: dict[str, list[str]] | None = None,
    candidate_feature_quota: int = 0,
    save_models_dir: Path | None = None,
    model_base_dir: Path | None = None,
    latest: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    train_all = frame.loc[
        ts.ge(window["train_start"]) & ts.lt(window["train_end"])
    ]
    eval_all = frame.loc[
        ts.ge(window["valid_from"]) & ts.lt(window["valid_to"])
    ].copy()
    eval_all["risk_score"] = 0.0
    eval_all["risk_effect_count"] = 0
    effects: list[dict[str, Any]] = []
    if train_all.empty or eval_all.empty:
        return eval_all, effects
    for (side, arch), eval_group in eval_all.groupby(["side_name", ARCH_COL], dropna=False, observed=True):
        train = train_all.loc[
            train_all["side_name"].eq(side) & train_all[ARCH_COL].eq(arch)
        ]
        if len(train) < 250 or len(eval_group) < 30:
            continue
        group_key = f"{side}|{arch}"
        candidate_cols = (candidate_feature_map or {}).get(group_key, [])
        max_features = int(params.get("max_features", 10))
        candidate_focus = _trade_relevant_focus_mask(train)
        if int(candidate_feature_quota) > 0 and candidate_cols:
            reserved = _select_feature_cols(
                train,
                candidate_cols,
                max_features=min(int(candidate_feature_quota), max_features),
                allow_binary=True,
                focus_mask=candidate_focus,
            )
            base_cols = _select_feature_cols(
                train,
                feature_cols,
                max_features=max(max_features - len(reserved), 0),
            )
            cols = list(dict.fromkeys([*reserved, *base_cols]))[:max_features]
        elif candidate_cols:
            cols = _select_feature_cols(
                train,
                list(dict.fromkeys([*feature_cols, *candidate_cols])),
                max_features=max_features,
                allow_binary=True,
                focus_mask=candidate_focus,
            )
        else:
            cols = _select_feature_cols(train, feature_cols, max_features=max_features)
        if not cols:
            continue
        x_train = _feature_frame(train, cols)
        y_train = _risk_target(train)
        x_eval = _feature_frame(eval_group, cols)
        model = _make_model(family, params)
        try:
            model.fit(x_train, y_train, **({"sample_weight": _sample_weight(train)} if family in {"gam", "ebm"} else {}))
            pred = np.asarray(model.predict(x_eval), dtype="float32")
        except Exception:
            continue
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        eval_all.loc[eval_group.index, "risk_score"] = pred
        eval_all.loc[eval_group.index, "risk_effect_count"] = 1
        if save_models_dir is not None:
            save_models_dir.mkdir(parents=True, exist_ok=True)
            safe_side = _safe_name(side)
            safe_arch = _safe_name(arch)
            model_path = save_models_dir / f"{family}_{safe_side}_{safe_arch}.joblib"
            joblib.dump(model, model_path)
            base_dir = model_base_dir or save_models_dir.parent
            effects.append(
                {
                    "side_name": str(side),
                    "archetype_policy_key": str(arch),
                    "shape": f"{family}_pickle",
                    "model_type": family,
                    "model_path": str(model_path.relative_to(base_dir)),
                    "feature_cols": cols,
                    "candidate_feature_quota": int(candidate_feature_quota),
                    "fill_values": {col: float(_safe_numeric(train[col]).median()) if train[col].notna().any() else 0.0 for col in cols},
                    "train_rows": int(len(train)),
                    "window_id": str(window["window_id"]),
                    "train_start": pd.Timestamp(window["train_start"]).isoformat(),
                    "train_end": pd.Timestamp(window["train_end"]).isoformat(),
                    "valid_from": pd.Timestamp(window["valid_from"]).isoformat(),
                    "valid_to": pd.Timestamp(window["valid_to"]).isoformat(),
                    "latest": bool(latest),
                }
            )
    cap_neg = float(params.get("risk_cap_negative", 0.08))
    cap_pos = float(params.get("risk_cap_positive", 0.02))
    eval_all["risk_score"] = _safe_numeric(eval_all["risk_score"]).clip(-cap_neg, cap_pos)
    eval_all["score_regime_calibrated"] = (
        _safe_numeric(eval_all[SCORE_COL]) - _safe_numeric(eval_all["risk_score"]).fillna(0.0)
    ).clip(0.0, 1.0).astype("float32")
    return eval_all, effects


def _train_latest_effects(
    frame: pd.DataFrame,
    feature_cols: list[str],
    family: str,
    params: dict[str, Any],
    *,
    candidate_feature_map: dict[str, list[str]] | None = None,
    candidate_feature_quota: int = 0,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_from: pd.Timestamp,
    save_models_dir: Path,
    model_base_dir: Path,
) -> list[dict[str, Any]]:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    train_all = frame.loc[ts.ge(train_start) & ts.lt(train_end)]
    effects: list[dict[str, Any]] = []
    if train_all.empty:
        return effects
    for (side, arch), group in train_all.groupby(["side_name", ARCH_COL], dropna=False, observed=True):
        if len(group) < 250:
            continue
        group_key = f"{side}|{arch}"
        candidate_cols = (candidate_feature_map or {}).get(group_key, [])
        max_features = int(params.get("max_features", 10))
        candidate_focus = _trade_relevant_focus_mask(group)
        if int(candidate_feature_quota) > 0 and candidate_cols:
            reserved = _select_feature_cols(
                group,
                candidate_cols,
                max_features=min(int(candidate_feature_quota), max_features),
                allow_binary=True,
                focus_mask=candidate_focus,
            )
            base_cols = _select_feature_cols(
                group,
                feature_cols,
                max_features=max(max_features - len(reserved), 0),
            )
            cols = list(dict.fromkeys([*reserved, *base_cols]))[:max_features]
        elif candidate_cols:
            cols = _select_feature_cols(
                group,
                list(dict.fromkeys([*feature_cols, *candidate_cols])),
                max_features=max_features,
                allow_binary=True,
                focus_mask=candidate_focus,
            )
        else:
            cols = _select_feature_cols(group, feature_cols, max_features=max_features)
        if not cols:
            continue
        x = _feature_frame(group, cols)
        y = _risk_target(group)
        model = _make_model(family, params)
        try:
            model.fit(x, y, **({"sample_weight": _sample_weight(group)} if family in {"gam", "ebm"} else {}))
        except Exception:
            continue
        save_models_dir.mkdir(parents=True, exist_ok=True)
        safe_side = _safe_name(side)
        safe_arch = _safe_name(arch)
        model_path = save_models_dir / f"{family}_{safe_side}_{safe_arch}.joblib"
        joblib.dump(model, model_path)
        effects.append(
            {
                "side_name": str(side),
                "archetype_policy_key": str(arch),
                "shape": f"{family}_pickle",
                "model_type": family,
                "model_path": str(model_path.relative_to(model_base_dir)),
                "feature_cols": cols,
                "candidate_feature_quota": int(candidate_feature_quota),
                "fill_values": {col: float(_safe_numeric(group[col]).median()) if group[col].notna().any() else 0.0 for col in cols},
                "train_rows": int(len(group)),
                "window_id": "latest_live_fallback",
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "valid_from": valid_from.isoformat(),
                "valid_to": None,
                "latest": True,
            }
        )
    return effects


def _evaluate_params(
    frame: pd.DataFrame,
    eval_months: list[str],
    feature_cols: list[str],
    family: str,
    params: dict[str, Any],
    *,
    candidate_feature_map: dict[str, list[str]] | None = None,
    candidate_feature_quota: int = 0,
    train_days: int,
    oos_days: int,
    step_days: int,
) -> pd.DataFrame:
    parts = []
    for window in _rolling_windows(
        eval_months,
        train_days=train_days,
        oos_days=oos_days,
        step_days=step_days,
    ):
        pred, _ = _fit_predict_window(
            frame,
            window,
            feature_cols,
            family,
            params,
            candidate_feature_map=candidate_feature_map,
            candidate_feature_quota=int(candidate_feature_quota),
        )
        parts.append(pred)
    return pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()


class NoImprovementStopper:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best: float | None = None
        self.best_trial = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_value is None:
            return
        if self.best is None or study.best_value > self.best + 1e-12:
            self.best = float(study.best_value)
            self.best_trial = int(trial.number)
        elif int(trial.number) - self.best_trial >= self.patience:
            study.stop()


def _suggest(trial: optuna.Trial, families: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    allowed = families or ["bucket", "spline", "gam", "ebm"]
    family = trial.suggest_categorical("family", allowed)
    params: dict[str, Any] = {
        "risk_cap_negative": trial.suggest_categorical("risk_cap_negative", [0.04, 0.06, 0.08, 0.10]),
        "risk_cap_positive": trial.suggest_categorical("risk_cap_positive", [0.02, 0.04, 0.06]),
        "max_features": trial.suggest_categorical("max_features", [6, 8, 10]),
    }
    if family == "spline":
        params.update(
            {
                "variant": trial.suggest_categorical("spline_variant", ["monotone", "convex_risk", "concave_confidence", "u_shaped", "two_sided"]),
                "alpha": trial.suggest_categorical("alpha", [1.0, 2.0, 5.0, 10.0]),
                "l1_ratio": trial.suggest_categorical("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9]),
                "n_splines": trial.suggest_categorical("spline_n_splines", [5, 6, 7, 8]),
                "degree": trial.suggest_categorical("spline_degree", [2, 3]),
            }
        )
    elif family == "gam":
        params.update(
            {
                "n_splines": trial.suggest_categorical("gam_n_splines", [5, 6, 7, 8]),
                "lambda": trial.suggest_categorical("gam_lambda", [10.0, 30.0, 60.0, 100.0]),
                "degree": trial.suggest_categorical("gam_degree", [2, 3]),
                "interactions": trial.suggest_categorical("gam_interactions", [0, 1, 2, 3]),
            }
        )
    elif family == "ebm":
        params.update(
            {
                "max_bins": trial.suggest_categorical("ebm_max_bins", [64, 128, 256]),
                "interactions": trial.suggest_categorical("ebm_interactions", [0, 1, 2, 3, 4, 5]),
                "max_leaves": trial.suggest_categorical("ebm_max_leaves", [2, 3, 4]),
            }
        )
    return family, params


def _load_merged(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    pred = _derive_prediction_features(_load_predictions(args.meta_run, args.all_months))
    handoff_cols = set(_schema_cols(args.handoff))
    pred_cols = set(pred.columns)
    chosen = _available_feature_cols(handoff_cols, pred_cols)
    derived_source_cols = {
        col
        for col in handoff_cols
        if col.startswith("gmm_cluster_posterior_") or col in {"gmm_posterior_max"}
    }
    candidate_cols = sorted(
        {
            col
            for col in chosen.values()
            if col and not col.startswith("__derived_") and col not in pred_cols
        }
        | derived_source_cols
    )
    features = _load_feature_slice(args.handoff, pred, candidate_cols) if candidate_cols else pd.DataFrame(columns=KEYS)
    merged = pred.merge(features, on=KEYS, how="left", validate="many_to_one") if len(features) else pred
    if getattr(args, "additional_feature_frame", None):
        extra_path = Path(args.additional_feature_frame)
        if extra_path.exists():
            extra = pd.read_parquet(extra_path)
            join_cols = [col for col in KEYS if col in extra.columns and col in merged.columns]
            if len(join_cols) == len(KEYS):
                extra["__ts__"] = pd.to_datetime(extra["__ts__"], utc=True, errors="coerce")
                extra["__symbol__"] = extra["__symbol__"].astype(str)
                extra["side_name"] = extra["side_name"].astype(str)
                extra_cols = [
                    col
                    for col in extra.columns
                    if col not in set(join_cols) and col not in set(merged.columns)
                ]
                if extra_cols:
                    merged = merged.merge(
                        extra[join_cols + extra_cols].drop_duplicates(join_cols),
                        on=join_cols,
                        how="left",
                        validate="many_to_one",
                    )
    merged = _derive_joined_features(merged)
    feature_map = {name: col for name, col in chosen.items() if col and col in merged.columns}
    feature_cols = sorted(set(feature_map.values()))
    candidate_feature_map: dict[str, list[str]] = {}
    manifest_arg = getattr(args, "candidate_feature_manifest", None)
    manifest_path = Path(manifest_arg) if manifest_arg else None
    if manifest_path is not None and manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for group_key, features_for_group in dict(payload.get("feature_map") or {}).items():
            if isinstance(features_for_group, list):
                cols = [str(col) for col in features_for_group if str(col) in merged.columns]
                if cols:
                    candidate_feature_map[str(group_key)] = list(dict.fromkeys(cols))
    for col in OUTCOME_COLS + [SCORE_COL]:
        if col not in merged.columns:
            raise RuntimeError(f"Required column missing from prediction shards: {col}")
    merged["__ts__"] = pd.to_datetime(merged["__ts__"], utc=True, errors="coerce")
    merged = merged.sort_values("__ts__").reset_index(drop=True)
    return merged, feature_cols, candidate_feature_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-run", type=Path, default=DEFAULT_META_RUN)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT.parent / "meta_oos_regime_calibration_model_ablation_rolling60d_oos15_20260708")
    parser.add_argument("--artifact-output-dir", type=Path, default=DEFAULT_ROLLING_CALIBRATION_OUT)
    parser.add_argument("--all-months", nargs="+", default=["2026-04", "2026-05", "2026-06"])
    parser.add_argument("--eval-months", nargs="+", default=["2026-05", "2026-06"])
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rolling-train-days", type=int, default=60)
    parser.add_argument("--rolling-oos-days", type=int, default=15)
    parser.add_argument("--rolling-step-days", type=int, default=15)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["bucket", "spline", "gam", "ebm"],
        choices=["bucket", "spline", "gam", "ebm"],
        help="Calibration model families to search. Use '--families ebm' for an EBM-only comparison.",
    )
    parser.add_argument(
        "--additional-feature-frame",
        type=Path,
        default=None,
        help="Optional parquet with KEYS plus materialized economic composite features to join before EBM/GAM fitting.",
    )
    parser.add_argument(
        "--candidate-feature-manifest",
        type=Path,
        default=None,
        help="Optional economic relevance manifest whose feature_map augments calibration feature_cols.",
    )
    parser.add_argument(
        "--candidate-feature-quota",
        type=int,
        default=0,
        help="Reserve up to this many local side x archetype manifest features in each EBM/GAM fit before filling normal features.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged, feature_cols, candidate_feature_map = _load_merged(args)
    families = list(dict.fromkeys(str(f) for f in args.families))
    hpo_month = args.eval_months[0]
    baseline = merged.loc[merged["month"].isin(args.eval_months)].copy()
    baseline["score_regime_calibrated"] = baseline[SCORE_COL]

    trials: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        family, params = _suggest(trial, families)
        adjusted = _evaluate_params(
            merged,
            [hpo_month],
            feature_cols,
            family,
            params,
            candidate_feature_map=candidate_feature_map,
            candidate_feature_quota=int(args.candidate_feature_quota),
            train_days=args.rolling_train_days,
            oos_days=args.rolling_oos_days,
            step_days=args.rolling_step_days,
        )
        if adjusted.empty:
            return -1e9
        metrics = _top10_objective(adjusted, "score_regime_calibrated")
        value = float(metrics["objective"])
        trial.set_user_attr("family", family)
        trial.set_user_attr("params_json", json.dumps(params, sort_keys=True))
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trials.append({"trial": trial.number, "family": family, **params, **metrics})
        return value if np.isfinite(value) else -1e9

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=12, n_warmup_steps=0),
    )
    for arm in families:
        study.enqueue_trial({"family": arm})
    study.optimize(objective, n_trials=args.n_trials, callbacks=[NoImprovementStopper(args.patience)])

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(args.output_dir / "hpo_trials.csv", index=False)
    best_family = str(study.best_trial.user_attrs.get("family"))
    best_params = json.loads(str(study.best_trial.user_attrs.get("params_json")))
    best_payload = {
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_family": best_family,
        "best_params": best_params,
        "hpo_month": hpo_month,
        "n_trials_completed": len(study.trials),
        "hpo_once": True,
        "families": families,
        "candidate_feature_quota": int(args.candidate_feature_quota),
        "rolling_train_days": int(args.rolling_train_days),
        "rolling_oos_days": int(args.rolling_oos_days),
        "rolling_step_days": int(args.rolling_step_days),
    }
    (args.output_dir / "best_hpo.json").write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")

    family_rows: list[dict[str, Any]] = []
    monthly_parts: list[pd.DataFrame] = []
    for family, fam_trials in trials_df.groupby("family", dropna=False):
        row = fam_trials.sort_values("objective", ascending=False).iloc[0].to_dict()
        params = {
            k: row[k]
            for k in [
                "risk_cap_negative",
                "risk_cap_positive",
                "max_features",
                "variant",
                "alpha",
                "l1_ratio",
                "n_splines",
                "degree",
                "lambda",
                "interactions",
                "max_bins",
                "max_leaves",
            ]
            if k in row and pd.notna(row[k])
        }
        adjusted = _evaluate_params(
            merged,
            args.eval_months,
            feature_cols,
            str(family),
            params,
            candidate_feature_map=candidate_feature_map,
            candidate_feature_quota=int(args.candidate_feature_quota),
            train_days=args.rolling_train_days,
            oos_days=args.rolling_oos_days,
            step_days=args.rolling_step_days,
        )
        monthly_parts.append(adjusted.assign(__arm__=str(family)))
        metric = _top10_objective(adjusted, "score_regime_calibrated")
        family_rows.append({"family": family, **params, **metric})
    pd.DataFrame(family_rows).sort_values("objective", ascending=False).to_csv(
        args.output_dir / "family_best_summary.csv",
        index=False,
    )

    winner_adjusted = _evaluate_params(
        merged,
        args.eval_months,
        feature_cols,
        best_family,
        best_params,
        candidate_feature_map=candidate_feature_map,
        candidate_feature_quota=int(args.candidate_feature_quota),
        train_days=args.rolling_train_days,
        oos_days=args.rolling_oos_days,
        step_days=args.rolling_step_days,
    )
    metrics = pd.DataFrame(
        [
            *_metric_rows(baseline, SCORE_COL, "baseline_meta"),
            *_metric_rows(winner_adjusted, "score_regime_calibrated", f"winner_{best_family}"),
        ]
    )
    base_overall = metrics.loc[(metrics["arm"].eq("baseline_meta")) & (metrics["group"].eq("overall"))]
    win_overall = metrics.loc[(metrics["arm"].eq(f"winner_{best_family}")) & (metrics["group"].eq("overall"))]
    delta = win_overall.merge(base_overall, on=["top_scope", "group", "group_value"], suffixes=("_winner", "_baseline"))
    delta_cols = [
        "mean_ev_after_1pct",
        "clean_exec_rate",
        "dirty_positive_rate",
        "full_path_bad_mae_rate",
        "timeout_rate",
        "hit_rate",
        "loss_rate",
        "loss_autocorr_lag1",
        "max_loss_streak",
        "mean_loss_streak",
    ]
    for col in delta_cols:
        delta[f"delta_{col}"] = delta[f"{col}_winner"] - delta[f"{col}_baseline"]
    metrics.to_csv(args.output_dir / "winner_monthly_metrics.csv", index=False)
    delta.to_csv(args.output_dir / "winner_delta_vs_baseline_overall.csv", index=False)
    month_base = metrics.loc[(metrics["arm"].eq("baseline_meta")) & (metrics["group"].eq("month"))]
    month_win = metrics.loc[(metrics["arm"].eq(f"winner_{best_family}")) & (metrics["group"].eq("month"))]
    month_delta = month_win.merge(month_base, on=["top_scope", "group", "group_value"], suffixes=("_winner", "_baseline"))
    for col in delta_cols:
        month_delta[f"delta_{col}"] = month_delta[f"{col}_winner"] - month_delta[f"{col}_baseline"]
    month_delta.to_csv(args.output_dir / "winner_month_delta_vs_baseline.csv", index=False)

    final_effects: list[dict[str, Any]] = []
    args.artifact_output_dir.mkdir(parents=True, exist_ok=True)
    rolling_windows = _rolling_windows(
        args.eval_months,
        train_days=args.rolling_train_days,
        oos_days=args.rolling_oos_days,
        step_days=args.rolling_step_days,
    )
    for window in rolling_windows:
        model_dir = args.artifact_output_dir / "models" / str(window["window_id"])
        _, effects = _fit_predict_window(
            merged,
            window,
            feature_cols,
            best_family,
            best_params,
            candidate_feature_map=candidate_feature_map,
            candidate_feature_quota=int(args.candidate_feature_quota),
            save_models_dir=model_dir,
            model_base_dir=args.artifact_output_dir,
        )
        final_effects.extend(effects)
    latest_valid_to = max((pd.Timestamp(w["valid_to"]) for w in rolling_windows), default=None)
    if latest_valid_to is not None:
        latest_effects = _train_latest_effects(
            merged,
            feature_cols,
            best_family,
            best_params,
            candidate_feature_map=candidate_feature_map,
            candidate_feature_quota=int(args.candidate_feature_quota),
            train_start=latest_valid_to - pd.Timedelta(days=int(args.rolling_train_days)),
            train_end=latest_valid_to,
            valid_from=latest_valid_to,
            save_models_dir=args.artifact_output_dir / "models" / "latest_live_fallback",
            model_base_dir=args.artifact_output_dir,
        )
        final_effects.extend(latest_effects)

    artifact = {
        "artifact_id": CALIBRATION_POLICY_ID,
        "policy_id": CALIBRATION_POLICY_ID,
        "source": "run_regime_calibration_model_ablation.py",
        "source_score_col": SCORE_COL,
        "adjusted_score_col": "score_regime_calibrated",
        "risk_score_col": "regime_ev_risk_score",
        "effect_count_col": "regime_ev_effect_count",
        "risk_cap": 0.06,
        "risk_cap_negative": float(best_params.get("risk_cap_negative", 0.08)),
        "risk_cap_positive": float(best_params.get("risk_cap_positive", 0.02)),
        "hpo": best_payload,
        "time_windowed_effects": True,
        "latest_valid_to": latest_valid_to.isoformat() if latest_valid_to is not None else None,
        "calibration_schedule": {
            "hpo_once": True,
            "hpo_month": hpo_month,
            "rolling_train_days": int(args.rolling_train_days),
            "rolling_oos_days": int(args.rolling_oos_days),
            "rolling_step_days": int(args.rolling_step_days),
            "bar_minutes": 15,
        },
        "archetype_prefix_aliases": ARCHETYPE_PREFIX_ALIASES,
        "feature_columns": feature_cols,
        "candidate_feature_map": candidate_feature_map,
        "candidate_feature_quota": int(args.candidate_feature_quota),
        "effects": final_effects,
        "notes": (
            "Calibration model family selected once by first-month HPO. "
            "Effects are frozen per side x archetype, retrained every 15 days "
            "on the previous 60 days, and applied only to the next 15-day OOS window. "
            "The latest effect set is a live fallback after latest_valid_to."
        ),
    }
    (args.artifact_output_dir / "regime_ev_calibration.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifact_path": str(args.artifact_output_dir / "regime_ev_calibration.json"), **best_payload}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(best_payload, indent=2, sort_keys=True))
    print(f"[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
