#!/usr/bin/env python3
"""Fast strict-label learnability smoke inside execution limits.

This is a pre-training diagnostic. It uses month-forward ExtraTrees proxies to
ask whether strict soft labels are learnable enough to select economically clean
rows before running the production base/meta pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.diagnose_label_matched_clean_dirty_feature_gap import _build_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _make_targets,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.run_soft_label_rounda_topk_proxy_diagnostics import (  # noqa: E402
    _strict_rounda_targets,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/strict_label_fast_learnability_smoke_v1")
DEFAULT_LABEL_ARMS = (
    "S121_s8_clean_rank_veto",
    "S122_clean_dirty_contrast_rank",
    "S126_clean_net_direct_rank",
)
EXPERIMENTAL_LABEL_ARMS = (
    "S128_econ_admissible_support",
    "S129_support_first_net_rank",
    "S130_support_dirty_contrast_rank",
    "S131_net_utility_rank",
    "S132_return_net_utility_rank",
)
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.0025, 0.005, 0.01)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _month_model_frame(
    frame: pd.DataFrame,
    *,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = frame.loc[train_mask, features].copy()
    x_valid = frame.loc[valid_mask, features].copy()
    med = x_train.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return x_train.astype(np.float32, copy=False), x_valid.astype(np.float32, copy=False)


def _fit_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int,
    min_samples_leaf: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=int(min_samples_leaf),
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    y = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    model.fit(x_train, y)
    return model.predict(x_valid).astype(np.float32)


def _sigmoid_series(values: Any, index: pd.Index) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0))), index=index).clip(0.0, 1.0)


def _rank_pct(values: Any) -> pd.Series:
    out = pd.to_numeric(pd.Series(values), errors="coerce").rank(method="average", pct=True)
    return out.fillna(0.5).clip(0.0, 1.0)


def _masked_timestamp_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    rank = raw.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    rank = rank.fillna(raw.rank(method="average", pct=True)).clip(0.0, 1.0)
    return (rank * raw.gt(0.0).astype(float)).clip(0.0, 1.0)


def _target_frame(soft: pd.Series, hard: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": pd.to_numeric(soft, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "target_hard": pd.Series(hard, index=soft.index).fillna(False).astype(float),
        },
        index=soft.index,
    )


def _experimental_two_stage_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx = metrics.index
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(-0.02)
    ret_net = pd.to_numeric(metrics["ret_net"], errors="coerce").fillna(-0.02)
    mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0)
    mfe = pd.to_numeric(metrics["mfe_norm"], errors="coerce").fillna(0.0)
    barrier = pd.to_numeric(metrics["barrier"], errors="coerce").fillna(1.0)
    timeout = pd.to_numeric(metrics["is_timeout"].astype(float), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    mfe_mae = (mfe / mae.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=10.0)

    net_floor = ((u - 0.0010) / 0.0120).clip(0.0, 1.0)
    net_margin_floor = ((u - 0.0030) / 0.0120).clip(0.0, 1.0)
    ret_net_floor = ((ret_net - 0.0010) / 0.0120).clip(0.0, 1.0)
    ret_net_margin_floor = ((ret_net - 0.0030) / 0.0120).clip(0.0, 1.0)
    mae_floor = ((0.95 - mae) / 0.95).clip(0.0, 1.0)
    strict_mae_floor = ((0.75 - mae) / 0.75).clip(0.0, 1.0)
    barrier_floor = ((0.027 - barrier) / 0.027).clip(0.0, 1.0)
    efficiency_floor = ((mfe_mae - 1.05) / 2.00).clip(0.0, 1.0)
    no_timeout = (1.0 - timeout).clip(0.0, 1.0)
    economic_core = (
        (u > 0.0010)
        & (mae <= 0.95)
        & (barrier <= 0.027)
        & (mfe_mae >= 1.05)
        & (timeout <= 0.0)
    )
    support_soft = (
        economic_core.astype(float)
        * (
            0.35 * net_floor
            + 0.25 * mae_floor
            + 0.20 * barrier_floor
            + 0.20 * efficiency_floor
        )
    ).clip(0.0, 1.0)
    strict_support_soft = (
        economic_core.astype(float)
        * (
            0.40 * net_margin_floor
            + 0.25 * strict_mae_floor
            + 0.15 * barrier_floor
            + 0.20 * efficiency_floor
        )
        * no_timeout
    ).clip(0.0, 1.0)
    support_rank = _masked_timestamp_rank(frame, support_soft)
    strict_support_rank = _masked_timestamp_rank(frame, strict_support_soft)
    net_utility_rank = _masked_timestamp_rank(frame, net_floor)
    ret_net_utility_rank = _masked_timestamp_rank(frame, ret_net_floor)
    dirty_penalty = (
        0.40 * _sigmoid_series((mae - 1.00) / 0.20, idx)
        + 0.25 * timeout
        + 0.20 * _sigmoid_series((barrier - 0.025) / 0.004, idx)
        + 0.15 * _sigmoid_series((1.20 - mfe_mae) / 0.25, idx)
    ).clip(0.0, 1.0)
    contrast = (support_soft * (1.0 - dirty_penalty)).clip(0.0, 1.0)
    contrast_rank = _masked_timestamp_rank(frame, contrast)
    return {
        "S128_econ_admissible_support": _target_frame(support_soft, economic_core),
        "S129_support_first_net_rank": _target_frame(
            (0.70 * support_soft + 0.30 * support_rank).clip(0.0, 1.0),
            economic_core & net_floor.gt(0.0),
        ),
        "S130_support_dirty_contrast_rank": _target_frame(
            (0.55 * contrast + 0.25 * contrast_rank + 0.20 * strict_support_rank).clip(0.0, 1.0),
            economic_core & strict_support_soft.gt(0.0),
        ),
        "S131_net_utility_rank": _target_frame(
            (0.55 * net_floor + 0.45 * net_utility_rank).clip(0.0, 1.0),
            u > 0.0030,
        ),
        "S132_return_net_utility_rank": _target_frame(
            (0.55 * ret_net_floor + 0.45 * ret_net_utility_rank).clip(0.0, 1.0),
            ret_net > 0.0030,
        ),
    }


def _constrained_top_indices(
    *,
    score: pd.Series,
    eligible: pd.Series,
    top_frac: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_s = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    eligible_s = eligible.reset_index(drop=True).fillna(False).astype(bool)
    finite_idx = np.flatnonzero(score_s.notna().to_numpy())
    target_rows = max(1, int(math.ceil(float(top_frac) * len(finite_idx)))) if len(finite_idx) else 0
    eligible_idx = np.flatnonzero((score_s.notna() & eligible_s).to_numpy())
    if target_rows <= 0 or not len(eligible_idx):
        return np.asarray([], dtype=np.int64), {
            "target_rows": int(target_rows),
            "eligible_rows": int(len(eligible_idx)),
            "selected_fill_rate": 0.0,
        }
    order = eligible_idx[
        np.argsort(-score_s.iloc[eligible_idx].to_numpy(dtype=np.float64), kind="mergesort")
    ]
    selected = order[: min(int(target_rows), len(order))].astype(np.int64, copy=False)
    return selected, {
        "target_rows": int(target_rows),
        "eligible_rows": int(len(eligible_idx)),
        "selected_fill_rate": float(len(selected) / max(int(target_rows), 1)),
    }


def _evaluate_selection(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    selected_idx: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=arm,
        selector=selector,
        period=period,
        top_frac=float(top_frac),
        selected_idx=selected_idx,
    )
    row.update(_decile_diagnostics(score, metrics["u_policy_net"]))
    row.update(
        {
            "score_ic_u": _spearman(score, metrics["u_policy_net"]),
            "score_ic_bad_mae": _spearman(score, (metrics["mae_norm"] >= 1.0).astype(float)),
            "score_ic_timeout": _spearman(score, metrics["is_timeout"].astype(float)),
            "score_ic_clean_path": _spearman(
                score,
                (
                    (metrics["u_policy_net"] > 0.0)
                    & (metrics["mae_norm"] <= 1.0)
                    & (metrics["barrier"] <= 0.025)
                    & (metrics["is_timeout"].astype(float) <= 0.0)
                ).astype(float),
            ),
        }
    )
    if extra:
        row.update(extra)
    return row


def _weekly_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    month: str,
    top_frac: float,
    eligible: pd.Series | None = None,
) -> list[dict[str, Any]]:
    weeks = frame["__ts__"].dt.to_period("W-SUN").astype(str)
    rows: list[dict[str, Any]] = []
    for week, ids in pd.Series(np.arange(len(frame)), index=frame.index).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        if len(pos) < 20:
            continue
        local_score = score.iloc[pos].reset_index(drop=True)
        selected_idx = None
        extra: dict[str, Any] = {}
        if eligible is not None:
            selected_idx, extra = _constrained_top_indices(
                score=local_score,
                eligible=eligible.iloc[pos].reset_index(drop=True),
                top_frac=float(top_frac),
            )
        row = _evaluate_selection(
            frame=frame.iloc[pos].reset_index(drop=True),
            metrics=metrics.iloc[pos].reset_index(drop=True),
            target=target.iloc[pos].reset_index(drop=True),
            score=local_score,
            arm=arm,
            selector=selector,
            period=str(week),
            top_frac=float(top_frac),
            selected_idx=selected_idx,
            extra=extra,
        )
        row["month"] = str(month)
        row["label_arm"] = str(arm).split("::", 1)[0]
        rows.append(row)
    return rows


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    groups = monthly.groupby(["label_arm", "selector", "top_frac"], dropna=False, observed=True)
    for (label_arm, selector, top_frac), group in groups:
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        bad = pd.to_numeric(group["bad_mae_1r_rate"], errors="coerce")
        p90_mae = pd.to_numeric(group["p90_mae_norm"], errors="coerce")
        wide = pd.to_numeric(group["wide_barrier_25bps_rate"], errors="coerce")
        timeout = pd.to_numeric(group["timeout_rate"], errors="coerce")
        score_ic_u = pd.to_numeric(group["score_ic_u"], errors="coerce")
        score_ic_bad = pd.to_numeric(group["score_ic_bad_mae"], errors="coerce")
        positive_months = int((mean_u > 0.0).sum())
        worst_month = float(mean_u.min()) if len(mean_u.dropna()) else float("nan")
        mean_u_value = _safe_mean(mean_u)
        row = {
            "label_arm": str(label_arm),
            "selector": str(selector),
            "top_frac": float(top_frac),
            "months": int(group["period"].nunique()),
            "positive_months": positive_months,
            "mean_u": mean_u_value,
            "worst_month_mean_u": worst_month,
            "hit_u": _safe_mean(group["hit_u"]),
            "mean_return_net": _safe_mean(group["mean_return_net"]),
            "q10_u": _safe_mean(group["q10_u"]),
            "score_ic_u": _safe_mean(score_ic_u),
            "score_ic_bad_mae": _safe_mean(score_ic_bad),
            "score_ic_timeout": _safe_mean(group["score_ic_timeout"]),
            "score_ic_clean_path": _safe_mean(group["score_ic_clean_path"]),
            "decile_spearman_u": _safe_mean(group["decile_spearman_u"]),
            "top_bottom_decile_spread_u": _safe_mean(group["top_bottom_decile_spread_u"]),
            "bad_mae_1r_rate": _safe_mean(bad),
            "p90_mae_norm": _safe_mean(p90_mae),
            "wide_barrier_25bps_rate": _safe_mean(wide),
            "timeout_rate": _safe_mean(timeout),
            "clean_row_rate": _safe_mean(group["clean_row_rate"]),
            "strict_clean_row_rate": _safe_mean(group["strict_clean_row_rate"]),
            "mean_selected_rows": _safe_mean(selected_rows),
            "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
            "top_symbol_share": _safe_mean(group["top_symbol_share"]),
        }
        row["trainworthy_pass"] = bool(
            positive_months >= 3
            and math.isfinite(mean_u_value)
            and mean_u_value > 0.0
            and math.isfinite(worst_month)
            and worst_month > 0.0
            and _safe_mean(score_ic_u) > 0.0
            and _safe_mean(score_ic_bad) < 0.0
            and _safe_mean(bad) <= 0.40
            and _safe_mean(p90_mae) <= 4.0
            and _safe_mean(wide) <= 0.05
            and _safe_mean(timeout) <= 0.10
            and int(selected_rows.min()) >= 8
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["trainworthy_pass", "mean_u", "worst_month_mean_u"],
        ascending=[False, False, False],
    )


def _write_markdown(output_dir: Path, manifest: dict[str, Any], aggregate: pd.DataFrame) -> Path:
    path = output_dir / "strict_label_fast_learnability_smoke.md"
    cols = [
        "trainworthy_pass",
        "label_arm",
        "selector",
        "top_frac",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "score_ic_u",
        "score_ic_bad_mae",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "clean_row_rate",
        "strict_clean_row_rate",
        "mean_selected_rows",
        "min_selected_rows",
    ]
    view = aggregate[[c for c in cols if c in aggregate.columns]].head(40)
    lines = [
        "# Strict Label Fast Learnability Smoke",
        "",
        "Scope: month-forward ExtraTrees proxies for strict labels plus simple learned execution-risk gates. This is not production model training.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Features: `{manifest['feature_count']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        "",
        "## Aggregate",
        "",
        view.to_markdown(index=False) if not view.empty else "No rows.",
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_smoke(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arms: list[str],
    months: list[str],
    top_fracs: list[float],
    train_lookback_months: int | None,
    seed: int,
    min_samples_leaf: int,
    include_causal_state_path_priors: bool = False,
    include_event_confirmation_features: bool = False,
    include_adverse_path_composites: bool = False,
    prior_windows_days: list[float] | None = None,
    prior_embargo_hours: float = 24.0,
    state_path_prior_features: list[str] | None = None,
    event_feature_store_features: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_windows_days = list(prior_windows_days or DEFAULT_PRIOR_WINDOWS_DAYS)
    state_path_prior_features = list(state_path_prior_features or DEFAULT_STATE_PATH_PRIOR_FEATURES)
    event_feature_store_features = list(event_feature_store_features or DEFAULT_EVENT_FEATURE_STORE_FEATURES)
    if (
        include_causal_state_path_priors
        or include_event_confirmation_features
        or include_adverse_path_composites
    ):
        frame, metrics, frame_reports = _build_frame(
            labels_path=labels_path,
            feature_dir=feature_dir,
            feature_list_csv=feature_list_csv,
            max_feature_store_features=max_feature_store_features,
            include_causal_outcome_priors=False,
            include_causal_state_path_priors=include_causal_state_path_priors,
            include_event_confirmation_features=include_event_confirmation_features,
            include_adverse_path_composites=include_adverse_path_composites,
            prior_windows_days=prior_windows_days,
            prior_embargo_hours=prior_embargo_hours,
            state_path_prior_features=state_path_prior_features,
            event_feature_store_features=event_feature_store_features,
        )
    else:
        frame = _load_labels(labels_path)
        selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
        feature_matrix, feature_store_report = _load_feature_store_columns(
            frame,
            feature_dir=feature_dir,
            selected_features=selected_features,
        )
        if not feature_matrix.empty:
            frame = pd.concat(
                [frame.reset_index(drop=True), feature_matrix.astype(np.float32, copy=False).reset_index(drop=True)],
                axis=1,
                copy=False,
            )
        metrics = _path_metrics(frame)
        frame_reports = {
            "feature_store": feature_store_report,
            "causal_state_path_priors": {"enabled": False},
            "event_confirmation_features": {"enabled": False},
            "adverse_path_composites": {"enabled": False},
        }
    base_targets = _make_targets(frame, metrics)
    targets = _label_targets(frame, metrics)
    targets.update(
        _strict_rounda_targets(
            frame=frame,
            metrics=metrics,
            base_targets={**base_targets, **targets},
        )
    )
    targets.update(_experimental_two_stage_targets(frame, metrics))
    missing = sorted(set(label_arms) - set(targets))
    if missing:
        raise ValueError(f"Unknown label arms: {missing}")
    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for month in months:
        train_mask = month_period < month
        if train_lookback_months is not None and int(train_lookback_months) > 0:
            prior_months = sorted(month_period[train_mask].dropna().unique())
            keep = set(prior_months[-int(train_lookback_months) :])
            train_mask = train_mask & month_period.isin(keep)
        valid_mask = month_period == month
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            diagnostics.append(
                {
                    "period": month,
                    "skipped": True,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                }
            )
            continue
        x_train, x_valid = _month_model_frame(
            frame,
            train_mask=train_mask,
            valid_mask=valid_mask,
            features=features,
        )
        valid_frame = frame.loc[valid_mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        train_metrics = metrics.loc[train_mask].copy().reset_index(drop=True)
        y_bad = (train_metrics["mae_norm"] >= 1.0).astype(float)
        y_timeout = train_metrics["is_timeout"].astype(float)
        y_clean_path = (
            (train_metrics["u_policy_net"] > 0.0)
            & (train_metrics["mae_norm"] <= 1.0)
            & (train_metrics["barrier"] <= 0.025)
            & (train_metrics["is_timeout"].astype(float) <= 0.0)
        ).astype(float)
        bad_pred = pd.Series(
            _fit_predict(
                x_train=x_train,
                y_train=y_bad,
                x_valid=x_valid,
                seed=seed + 10_000,
                min_samples_leaf=min_samples_leaf,
            ),
            index=valid_frame.index,
        ).clip(0.0, 1.0)
        timeout_pred = pd.Series(
            _fit_predict(
                x_train=x_train,
                y_train=y_timeout,
                x_valid=x_valid,
                seed=seed + 20_000,
                min_samples_leaf=min_samples_leaf,
            ),
            index=valid_frame.index,
        ).clip(0.0, 1.0)
        clean_pred = pd.Series(
            _fit_predict(
                x_train=x_train,
                y_train=y_clean_path,
                x_valid=x_valid,
                seed=seed + 30_000,
                min_samples_leaf=min_samples_leaf,
            ),
            index=valid_frame.index,
        ).clip(0.0, 1.0)
        econ_support_target = targets["S128_econ_admissible_support"]
        econ_support_train = econ_support_target.loc[train_mask].copy()
        econ_support_valid = econ_support_target.loc[valid_mask].copy().reset_index(drop=True)
        econ_support_pred = pd.Series(
            _fit_predict(
                x_train=x_train,
                y_train=econ_support_train["target_soft"],
                x_valid=x_valid,
                seed=seed + 50_000,
                min_samples_leaf=min_samples_leaf,
            ),
            index=valid_frame.index,
        ).clip(0.0, 1.0)
        econ_support_hard_pred = pd.Series(
            _fit_predict(
                x_train=x_train,
                y_train=econ_support_train["target_hard"],
                x_valid=x_valid,
                seed=seed + 60_000,
                min_samples_leaf=min_samples_leaf,
            ),
            index=valid_frame.index,
        ).clip(0.0, 1.0)
        diagnostics.append(
            {
                "period": month,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "bad_pred_ic": _spearman(bad_pred, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                "timeout_pred_ic": _spearman(timeout_pred, valid_metrics["is_timeout"].astype(float)),
                "clean_pred_ic": _spearman(
                    clean_pred,
                    (
                        (valid_metrics["u_policy_net"] > 0.0)
                        & (valid_metrics["mae_norm"] <= 1.0)
                        & (valid_metrics["barrier"] <= 0.025)
                        & (valid_metrics["is_timeout"].astype(float) <= 0.0)
                    ).astype(float),
                ),
                "bad_pred_mean": _safe_mean(bad_pred),
                "bad_pred_p10": _safe_quantile(bad_pred, 0.10),
                "bad_pred_p50": _safe_quantile(bad_pred, 0.50),
                "bad_pred_p90": _safe_quantile(bad_pred, 0.90),
                "timeout_pred_mean": _safe_mean(timeout_pred),
                "timeout_pred_p10": _safe_quantile(timeout_pred, 0.10),
                "timeout_pred_p50": _safe_quantile(timeout_pred, 0.50),
                "timeout_pred_p90": _safe_quantile(timeout_pred, 0.90),
                "clean_pred_mean": _safe_mean(clean_pred),
                "clean_pred_p10": _safe_quantile(clean_pred, 0.10),
                "clean_pred_p50": _safe_quantile(clean_pred, 0.50),
                "clean_pred_p90": _safe_quantile(clean_pred, 0.90),
                "econ_support_pred_ic": _spearman(econ_support_pred, econ_support_valid["target_soft"]),
                "econ_support_hard_pred_ic": _spearman(
                    econ_support_hard_pred,
                    econ_support_valid["target_hard"],
                ),
                "econ_support_pred_mean": _safe_mean(econ_support_pred),
                "econ_support_pred_p10": _safe_quantile(econ_support_pred, 0.10),
                "econ_support_pred_p50": _safe_quantile(econ_support_pred, 0.50),
                "econ_support_pred_p90": _safe_quantile(econ_support_pred, 0.90),
            }
        )
        for label_arm in label_arms:
            target = targets[label_arm]
            target_train = target.loc[train_mask].copy()
            target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
            target_pred = pd.Series(
                _fit_predict(
                    x_train=x_train,
                    y_train=target_train["target_soft"],
                    x_valid=x_valid,
                    seed=seed,
                    min_samples_leaf=min_samples_leaf,
                ),
                index=valid_frame.index,
            )
            hard_pred = pd.Series(
                _fit_predict(
                    x_train=x_train,
                    y_train=target_train["target_hard"],
                    x_valid=x_valid,
                    seed=seed + 40_000,
                    min_samples_leaf=min_samples_leaf,
                ),
                index=valid_frame.index,
            ).clip(0.0, 1.0)
            risk_penalty = (target_pred - 0.55 * bad_pred - 0.15 * timeout_pred).astype(np.float32)
            clean_blend = (
                0.65 * target_pred
                + 0.35 * clean_pred
                - 0.35 * bad_pred
                - 0.10 * timeout_pred
            ).astype(np.float32)
            target_rank = _rank_pct(target_pred)
            support_rank = _rank_pct(hard_pred)
            bad_rank = _rank_pct(bad_pred)
            timeout_rank = _rank_pct(timeout_pred)
            clean_rank = _rank_pct(clean_pred)
            econ_support_rank = (
                0.65 * _rank_pct(econ_support_pred) + 0.35 * _rank_pct(econ_support_hard_pred)
            ).clip(0.0, 1.0)
            low_adverse_rank = (
                0.70 * (1.0 - bad_rank) + 0.30 * (1.0 - timeout_rank)
            ).clip(0.0, 1.0)
            two_stage_support = (
                0.65 * support_rank
                + 0.35 * target_rank
                - 0.25 * bad_rank
                - 0.10 * timeout_rank
            ).astype(np.float32)
            two_stage_gate_score = (
                0.70 * target_rank
                + 0.30 * support_rank
                - 0.30 * bad_rank
                - 0.10 * timeout_rank
            ).astype(np.float32)
            econ_gate_blend = (
                0.65 * target_rank
                + 0.25 * econ_support_rank
                + 0.10 * low_adverse_rank
            ).astype(np.float32)
            econ_clean_gate_blend = (
                0.55 * target_rank
                + 0.25 * econ_support_rank
                + 0.10 * low_adverse_rank
                + 0.10 * clean_rank
            ).astype(np.float32)
            selector_specs: list[tuple[str, pd.Series, pd.Series | None, dict[str, Any]]] = [
                ("raw_target_model", target_pred, None, {}),
                ("risk_penalty_model", risk_penalty, None, {}),
                (
                    "hard_risk_gate_model",
                    risk_penalty,
                    (bad_pred <= 0.57) & (timeout_pred <= 0.12),
                    {"bad_pred_cap": 0.57, "timeout_pred_cap": 0.12},
                ),
                ("clean_path_blend_model", clean_blend, None, {}),
                ("two_stage_support_rank_model", two_stage_support, None, {}),
                (
                    "two_stage_hard_support_gate_model",
                    two_stage_gate_score,
                    support_rank >= 0.70,
                    {"support_rank_floor": 0.70},
                ),
                (
                    "econ_gate70_then_target_rank_model",
                    target_rank,
                    econ_support_rank >= 0.70,
                    {"econ_support_rank_floor": 0.70},
                ),
                (
                    "econ_gate80_then_target_rank_model",
                    target_rank,
                    econ_support_rank >= 0.80,
                    {"econ_support_rank_floor": 0.80},
                ),
                (
                    "econ_lowadverse_gate70_then_target_rank_model",
                    econ_gate_blend,
                    (econ_support_rank >= 0.70) & (low_adverse_rank >= 0.45),
                    {"econ_support_rank_floor": 0.70, "low_adverse_rank_floor": 0.45},
                ),
                (
                    "econ_lowadverse_gate80_then_target_rank_model",
                    econ_clean_gate_blend,
                    (econ_support_rank >= 0.80) & (low_adverse_rank >= 0.50),
                    {"econ_support_rank_floor": 0.80, "low_adverse_rank_floor": 0.50},
                ),
                (
                    "strict_clean_path_gate_model",
                    clean_blend,
                    (bad_pred <= 0.45) & (timeout_pred <= 0.08) & (clean_pred >= 0.55),
                    {"bad_pred_cap": 0.45, "timeout_pred_cap": 0.08, "clean_pred_floor": 0.55},
                ),
            ]
            base_extra = {
                "label_arm": label_arm,
                "target_train_mean": _safe_mean(target_train["target_soft"]),
                "target_train_hard_rate": _safe_mean(target_train["target_hard"]),
                "target_pred_ic_label": _spearman(target_pred, target_valid["target_soft"]),
                "hard_pred_ic_label": _spearman(hard_pred, target_valid["target_hard"]),
                "selected_bad_pred_mean": float("nan"),
                "selected_timeout_pred_mean": float("nan"),
                "selected_clean_pred_mean": float("nan"),
                "selected_hard_pred_mean": float("nan"),
                "selected_econ_support_pred_mean": float("nan"),
                "selected_econ_support_hard_pred_mean": float("nan"),
                "selected_econ_support_rank_mean": float("nan"),
                "selected_low_adverse_rank_mean": float("nan"),
            }
            for selector, score, eligible, selector_extra in selector_specs:
                for top_frac in top_fracs:
                    selected_idx = None
                    extra = {**base_extra, **selector_extra}
                    if eligible is not None:
                        selected_idx, constrained = _constrained_top_indices(
                            score=score,
                            eligible=eligible,
                            top_frac=float(top_frac),
                        )
                        extra.update(constrained)
                    else:
                        selected_idx = _rank_top_indices(score, float(top_frac))
                    if selected_idx is not None and len(selected_idx):
                        extra.update(
                            {
                                "selected_bad_pred_mean": _safe_mean(bad_pred.iloc[selected_idx]),
                                "selected_timeout_pred_mean": _safe_mean(timeout_pred.iloc[selected_idx]),
                                "selected_clean_pred_mean": _safe_mean(clean_pred.iloc[selected_idx]),
                                "selected_hard_pred_mean": _safe_mean(hard_pred.iloc[selected_idx]),
                                "selected_econ_support_pred_mean": _safe_mean(econ_support_pred.iloc[selected_idx]),
                                "selected_econ_support_hard_pred_mean": _safe_mean(
                                    econ_support_hard_pred.iloc[selected_idx],
                                ),
                                "selected_econ_support_rank_mean": _safe_mean(econ_support_rank.iloc[selected_idx]),
                                "selected_low_adverse_rank_mean": _safe_mean(low_adverse_rank.iloc[selected_idx]),
                            }
                        )
                    row = _evaluate_selection(
                        frame=valid_frame,
                        metrics=valid_metrics,
                        target=target_valid,
                        score=score,
                        arm=f"{label_arm}::{selector}",
                        selector=selector,
                        period=month,
                        top_frac=float(top_frac),
                        selected_idx=selected_idx,
                        extra=extra,
                    )
                    monthly_rows.append(row)
                    weekly_rows.extend(
                        _weekly_rows(
                            frame=valid_frame,
                            metrics=valid_metrics,
                            target=target_valid,
                            score=score,
                            arm=f"{label_arm}::{selector}",
                            selector=selector,
                            month=month,
                            top_frac=float(top_frac),
                            eligible=eligible,
                        )
                    )
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    aggregate = _aggregate(monthly)
    paths = {
        "monthly": output_dir / "strict_label_fast_learnability_monthly.csv",
        "weekly": output_dir / "strict_label_fast_learnability_weekly.csv",
        "aggregate": output_dir / "strict_label_fast_learnability_aggregate.csv",
        "diagnostics": output_dir / "strict_label_fast_learnability_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics_df.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "fast_strict_label_learnability_smoke_not_production_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        **frame_reports,
        "label_arms": list(label_arms),
        "months": list(months),
        "top_fracs": [float(v) for v in top_fracs],
        "train_lookback_months": int(train_lookback_months)
        if train_lookback_months is not None
        else None,
        "seed": int(seed),
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": 80,
            "max_depth": 8,
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": "sqrt",
        },
        "include_causal_outcome_priors": False,
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, manifest, aggregate)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=160)
    parser.add_argument("--label-arms", type=str, default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--train-lookback-months", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        type=str,
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=str,
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=str,
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        train_lookback_months=args.train_lookback_months,
        seed=int(args.seed),
        min_samples_leaf=int(args.min_samples_leaf),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=_parse_float_csv(args.prior_windows_days, DEFAULT_PRIOR_WINDOWS_DAYS),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv(
            args.state_path_prior_features,
            DEFAULT_STATE_PATH_PRIOR_FEATURES,
        ),
        event_feature_store_features=_parse_csv(
            args.event_feature_store_features,
            DEFAULT_EVENT_FEATURE_STORE_FEATURES,
        ),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
