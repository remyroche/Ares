#!/usr/bin/env python3
"""Two-stage utility plus path-risk veto label ablation.

This is a proxy-only diagnostic. It reuses the Stage 96 path-aware utility
target, then tests whether separately fitted causal timeout/tail-MAE vetoes can
repair adverse-path exposure before any base/meta model training.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _fit_holdout_summary,
    _score_period,
    _score_proxy,
    _slice_week_positions,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _spearman,
)
from scripts.run_path_aware_label_target_grid import (  # noqa: E402
    TargetSpec,
    _target_for_spec,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/path_aware_two_stage_veto_stage97_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_FRACS = (0.03, 0.05)
DEFAULT_RISK_KEEP_FRACS = (0.30, 0.50, 0.70)
DEFAULT_RISK_PENALTIES = (0.25, 0.50, 1.00)


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _sigmoid(values: Any) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0))))


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    ratio = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _utility_spec() -> TargetSpec:
    return TargetSpec(
        name="stage96_nearmiss_u0000_mae100_mm100_bar250_tonone_bars12",
        u_floor=0.0,
        u_temp=0.008,
        mae_cap=1.0,
        mae_temp=0.35,
        mfe_mae_min=1.0,
        mfe_mae_temp=0.35,
        barrier_cap=0.025,
        barrier_temp=0.006,
        timeout_mode="none",
        bars_cap=12.0,
        bars_temp=6.0,
    )


def _risk_targets(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0).reset_index(drop=True)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).reset_index(drop=True)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0).reset_index(drop=True)
    barrier = _safe_numeric(metrics["barrier"]).fillna(1.0).reset_index(drop=True)
    mfe_mae = _mfe_mae(metrics).fillna(0.0).reset_index(drop=True)

    bad_mae = _sigmoid((mae - 1.0) / 0.25).clip(0.0, 1.0)
    tail_mae_4r = _sigmoid((mae - 4.0) / 1.00).clip(0.0, 1.0)
    timeout_risk = timeout.clip(0.0, 1.0)
    slow_mfe = _sigmoid((bars - 12.0) / 5.0).clip(0.0, 1.0)
    wide_barrier = _sigmoid((barrier - 0.025) / 0.006).clip(0.0, 1.0)
    low_mfe_mae = _sigmoid((1.0 - mfe_mae) / 0.35).clip(0.0, 1.0)
    timeout_tail = pd.concat([timeout_risk, tail_mae_4r], axis=1).max(axis=1).clip(0.0, 1.0)
    adverse_path = pd.concat(
        [bad_mae, tail_mae_4r, timeout_risk, slow_mfe, wide_barrier, low_mfe_mae],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    return {
        "bad_mae": bad_mae,
        "tail_mae_4r": tail_mae_4r,
        "timeout": timeout_risk,
        "slow_mfe": slow_mfe,
        "wide_barrier": wide_barrier,
        "low_mfe_mae": low_mfe_mae,
        "timeout_tail": timeout_tail,
        "adverse_path": adverse_path,
    }


def _rank_pct(values: pd.Series) -> pd.Series:
    return _safe_numeric(values).rank(method="average", pct=True).clip(0.0, 1.0)


def _timestamp_rank_pct(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    score = _safe_numeric(values).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce").reset_index(drop=True)
    return score.groupby(timestamps, dropna=False).rank(method="average", pct=True).clip(0.0, 1.0)


def _build_selector_scores(
    *,
    valid: pd.DataFrame,
    utility_score: pd.Series,
    risk_scores: dict[str, pd.Series],
    risk_keep_fracs: list[float],
    risk_penalties: list[float],
) -> list[dict[str, Any]]:
    utility = _safe_numeric(utility_score).reset_index(drop=True)
    rows: list[dict[str, Any]] = [
        {
            "selector": "utility_proxy_oos",
            "economic_arm": "no_veto",
            "score": utility,
            "risk_arm": "",
            "risk_keep_frac": float("nan"),
            "risk_penalty": float("nan"),
            "mode": "base",
        }
    ]
    for risk_arm, raw_risk in risk_scores.items():
        risk = _safe_numeric(raw_risk).reset_index(drop=True)
        risk_rank_global = _rank_pct(risk).fillna(1.0)
        risk_rank_ts = _timestamp_rank_pct(valid, risk).fillna(1.0)
        for keep_frac in risk_keep_fracs:
            keep = float(keep_frac)
            rows.append(
                {
                    "selector": "utility_risk_veto_global_oos",
                    "economic_arm": f"{risk_arm}_keep{int(round(keep * 100)):02d}",
                    "score": utility.where(risk_rank_global <= keep),
                    "risk_arm": risk_arm,
                    "risk_keep_frac": keep,
                    "risk_penalty": float("nan"),
                    "mode": "global_veto",
                }
            )
            rows.append(
                {
                    "selector": "utility_risk_veto_timestamp_oos",
                    "economic_arm": f"{risk_arm}_tskeep{int(round(keep * 100)):02d}",
                    "score": utility.where(risk_rank_ts <= keep),
                    "risk_arm": risk_arm,
                    "risk_keep_frac": keep,
                    "risk_penalty": float("nan"),
                    "mode": "timestamp_veto",
                }
            )
        for penalty in risk_penalties:
            penalty_value = float(penalty)
            rows.append(
                {
                    "selector": "utility_minus_risk_oos",
                    "economic_arm": f"{risk_arm}_penalty{int(round(penalty_value * 100)):03d}",
                    "score": utility - penalty_value * risk_rank_global,
                    "risk_arm": risk_arm,
                    "risk_keep_frac": float("nan"),
                    "risk_penalty": penalty_value,
                    "mode": "penalty",
                }
            )
    return rows


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    utility_target: pd.DataFrame,
    risk_targets: dict[str, pd.Series],
    features: list[str],
    month: str,
    top_fracs: list[float],
    proxy_top_k: int,
    risk_keep_fracs: list[float],
    risk_penalties: list[float],
    min_train_rows: int,
    min_valid_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = period.lt(str(month))
    valid_mask = period.eq(str(month))
    if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
        return [], []

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    valid_target = utility_target.loc[valid_mask].copy().reset_index(drop=True)

    utility_score, utility_diag = _score_proxy(
        train=train,
        valid=frame.loc[valid_mask].copy(),
        features=features,
        y_train=utility_target.loc[train_mask, "target_soft"],
        proxy_top_k=int(proxy_top_k),
    )
    utility_score = utility_score.reset_index(drop=True)

    diagnostics: list[dict[str, Any]] = [
        {
            "month": str(month),
            "component": "utility",
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "proxy_ic_u": _spearman(utility_score, valid_metrics["u_policy_net"]),
            "proxy_ic_bad_mae": _spearman(utility_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
            "proxy_ic_tail_mae_4r": _spearman(utility_score, (valid_metrics["mae_norm"] >= 4.0).astype(float)),
            "proxy_ic_timeout": _spearman(utility_score, valid_metrics["is_timeout"].astype(float)),
            "proxy_ic_target": _spearman(utility_score, valid_target["target_soft"]),
            "proxy_features": ",".join(utility_diag.get("proxy_features", [])),
            "proxy_top_abs_ic": utility_diag.get("proxy_top_abs_ic"),
            "proxy_mean_top_abs_ic": utility_diag.get("proxy_mean_top_abs_ic"),
        }
    ]

    valid_risk_scores: dict[str, pd.Series] = {}
    risk_feature_names: dict[str, str] = {}
    for risk_arm, risk_target in risk_targets.items():
        score, diag = _score_proxy(
            train=train,
            valid=frame.loc[valid_mask].copy(),
            features=features,
            y_train=risk_target.loc[train_mask],
            proxy_top_k=int(proxy_top_k),
        )
        score = score.reset_index(drop=True)
        valid_risk_scores[risk_arm] = score
        risk_feature_names[risk_arm] = ",".join(diag.get("proxy_features", []))
        diagnostics.append(
            {
                "month": str(month),
                "component": risk_arm,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "proxy_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                "proxy_ic_bad_mae": _spearman(score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                "proxy_ic_tail_mae_4r": _spearman(score, (valid_metrics["mae_norm"] >= 4.0).astype(float)),
                "proxy_ic_timeout": _spearman(score, valid_metrics["is_timeout"].astype(float)),
                "proxy_ic_target": _spearman(score, risk_target.loc[valid_mask].reset_index(drop=True)),
                "proxy_features": risk_feature_names[risk_arm],
                "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
            }
        )

    selector_scores = _build_selector_scores(
        valid=valid,
        utility_score=utility_score,
        risk_scores=valid_risk_scores,
        risk_keep_fracs=risk_keep_fracs,
        risk_penalties=risk_penalties,
    )

    period_rows: list[dict[str, Any]] = []
    period_slices = [("month", str(month), np.arange(len(valid), dtype=np.int64))]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for selector_spec in selector_scores:
        score = _safe_numeric(selector_spec["score"]).reset_index(drop=True)
        risk_arm = str(selector_spec["risk_arm"])
        econ_features = risk_feature_names.get(risk_arm, "")
        for period_type, period_name, pos in period_slices:
            for frac in top_fracs:
                row = _score_period(
                    frame=valid.iloc[pos].reset_index(drop=True),
                    metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                    target=valid_target.iloc[pos].reset_index(drop=True),
                    score=score.iloc[pos].reset_index(drop=True),
                    period_type=period_type,
                    period=period_name,
                    month=str(month),
                    selector=str(selector_spec["selector"]),
                    label_arm="stage96_nearmiss_utility",
                    economic_arm=str(selector_spec["economic_arm"]),
                    top_frac=float(frac),
                    label_score=valid_target["target_soft"].iloc[pos].reset_index(drop=True),
                    economic_score=valid_risk_scores[risk_arm].iloc[pos].reset_index(drop=True)
                    if risk_arm in valid_risk_scores
                    else None,
                    economic_target=risk_targets[risk_arm].loc[valid_mask].reset_index(drop=True).iloc[pos].reset_index(drop=True)
                    if risk_arm in risk_targets
                    else None,
                    label_proxy_features=",".join(utility_diag.get("proxy_features", [])),
                    economic_proxy_features=econ_features,
                )
                row["risk_arm"] = risk_arm
                row["risk_keep_frac"] = selector_spec["risk_keep_frac"]
                row["risk_penalty"] = selector_spec["risk_penalty"]
                row["selector_mode"] = selector_spec["mode"]
                period_rows.append(row)
    return period_rows, diagnostics


def _write_markdown(
    *,
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    period_rows: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "path_aware_two_stage_veto_ablation.md"
    best = (
        fit_holdout.sort_values(
            [
                "trainworthy_pass",
                "holdout_economic_pass",
                "fit_economic_pass",
                "holdout_mean_return_net",
            ],
            ascending=[False, False, False, False],
        )
        if not fit_holdout.empty
        else fit_holdout
    )
    positive_dirty = (
        fit_holdout[
            fit_holdout["holdout_mean_return_net"].gt(0.0)
            & ~fit_holdout["holdout_economic_pass"].fillna(False).astype(bool)
        ].sort_values("holdout_mean_return_net", ascending=False)
        if not fit_holdout.empty
        else fit_holdout
    )
    month_cols = [
        "month",
        "selector",
        "economic_arm",
        "top_frac",
        "selected_rows",
        "mean_return_net",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "score_ic_u",
        "score_ic_economic",
    ]
    near_months = (
        period_rows[
            period_rows["period_type"].astype(str).eq("month")
            & period_rows["selector"].astype(str).eq("utility_proxy_oos")
            & pd.to_numeric(period_rows["top_frac"], errors="coerce").eq(0.03)
        ].sort_values("month")
        if not period_rows.empty
        else period_rows
    )
    lines = [
        "# Path-Aware Two-Stage Veto Ablation",
        "",
        "Scope: proxy-only label/execution diagnostic. No LightGBM, Optuna, policy optimisation, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Proxy top-k: `{manifest['proxy_top_k']}`",
        f"Risk arms: `{', '.join(manifest['risk_arms'])}`",
        "",
        "## Counts",
        "",
        f"- Fit/holdout rows: `{len(fit_holdout)}`",
        f"- Train-worthy rows: `{int(fit_holdout['trainworthy_pass'].sum()) if not fit_holdout.empty else 0}`",
        f"- Fit economic pass: `{int(fit_holdout['fit_economic_pass'].sum()) if not fit_holdout.empty else 0}`",
        f"- Holdout economic pass: `{int(fit_holdout['holdout_economic_pass'].sum()) if not fit_holdout.empty else 0}`",
        "",
        "## Best Rows",
        "",
        _table(
            best,
            [
                "trainworthy_pass",
                "fit_economic_pass",
                "holdout_economic_pass",
                "selector",
                "economic_arm",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ],
            limit=80,
        ),
        "",
        "## Positive But Rejected Rows",
        "",
        _table(
            positive_dirty,
            [
                "selector",
                "economic_arm",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ],
            limit=80,
        ),
        "",
        "## Base Utility Month Rows",
        "",
        _table(near_months, month_cols, limit=20),
        "",
        "## Component Proxy IC",
        "",
        _table(
            proxy_ic[proxy_ic["month"].astype(str).isin(manifest["fit_months"] + [manifest["holdout_month"]])]
            if not proxy_ic.empty
            else proxy_ic,
            [
                "month",
                "component",
                "proxy_ic_target",
                "proxy_ic_u",
                "proxy_ic_bad_mae",
                "proxy_ic_tail_mae_4r",
                "proxy_ic_timeout",
                "proxy_features",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    max_feature_columns: int | None,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    proxy_top_k: int,
    min_train_rows: int,
    min_valid_rows: int,
    max_timeout_rate: float | None,
    risk_keep_fracs: list[float],
    risk_penalties: list[float],
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    features = _feature_columns(frame)
    if max_feature_columns is not None and int(max_feature_columns) > 0:
        features = features[: int(max_feature_columns)]

    utility_target = _target_for_spec(metrics.reset_index(drop=True), _utility_spec(), frame["__ts__"].reset_index(drop=True))
    utility_target.index = frame.index
    risk_targets = _risk_targets(metrics.reset_index(drop=True))
    for target in risk_targets.values():
        target.index = frame.index

    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    for month in months:
        cur_rows, cur_diag = _run_month(
            frame=frame,
            metrics=metrics,
            utility_target=utility_target,
            risk_targets=risk_targets,
            features=features,
            month=str(month),
            top_fracs=top_fracs,
            proxy_top_k=int(proxy_top_k),
            risk_keep_fracs=risk_keep_fracs,
            risk_penalties=risk_penalties,
            min_train_rows=int(min_train_rows),
            min_valid_rows=int(min_valid_rows),
        )
        period_rows.extend(cur_rows)
        proxy_ic_rows.extend(cur_diag)
        print(json.dumps({"month": str(month), "period_rows": len(cur_rows), "proxy_ic_rows": len(cur_diag)}, sort_keys=True))

    period_frame = pd.DataFrame(period_rows)
    fit_holdout = _fit_holdout_summary(
        period_frame,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=10,
        max_timeout_rate=max_timeout_rate,
    )
    proxy_ic = pd.DataFrame(proxy_ic_rows)

    paths = {
        "fit_holdout": output_dir / "path_aware_two_stage_veto_fit_holdout.csv",
        "period_rows": output_dir / "path_aware_two_stage_veto_period_rows.csv",
        "proxy_ic": output_dir / "path_aware_two_stage_veto_proxy_ic.csv",
        "manifest": output_dir / "manifest.json",
    }
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    period_frame.to_csv(paths["period_rows"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)

    manifest = {
        "scope": "path_aware_two_stage_veto_ablation",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "months": [str(v) for v in months],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "proxy_top_k": int(proxy_top_k),
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "max_timeout_rate": max_timeout_rate,
        "risk_keep_fracs": [float(v) for v in risk_keep_fracs],
        "risk_penalties": [float(v) for v in risk_penalties],
        "risk_arms": list(risk_targets.keys()),
        "utility_spec": _utility_spec().to_dict(),
        "reports": reports,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        fit_holdout=fit_holdout,
        period_rows=period_frame,
        proxy_ic=proxy_ic,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "path_aware_two_stage_veto_ablation.md")}},
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=240)
    parser.add_argument("--max-feature-columns", type=int, default=240)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=list(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--max-timeout-rate", type=float, default=0.50)
    parser.add_argument("--risk-keep-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_RISK_KEEP_FRACS))
    parser.add_argument("--risk-penalties", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_RISK_PENALTIES))
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        default=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
        default=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        max_feature_columns=args.max_feature_columns,
        months=list(args.months),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        proxy_top_k=int(args.proxy_top_k),
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        max_timeout_rate=args.max_timeout_rate,
        risk_keep_fracs=list(args.risk_keep_fracs),
        risk_penalties=list(args.risk_penalties),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
