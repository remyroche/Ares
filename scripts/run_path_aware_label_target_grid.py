#!/usr/bin/env python3
"""Proxy-only grid over path-aware label target definitions.

The goal is to find label targets that are both learnable from causal features
and profitable after costs/path limits before training base or meta models.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
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
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    PROXY_OBJECTIVES,
    _proxy_score as _economic_proxy_score,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/path_aware_label_target_grid_stage96_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_FRACS = (0.03, 0.05)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    u_floor: float
    u_temp: float
    mae_cap: float
    mae_temp: float
    mfe_mae_min: float
    mfe_mae_temp: float
    barrier_cap: float
    barrier_temp: float
    timeout_mode: str
    bars_cap: float
    bars_temp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _filter_features(
    features: list[str],
    *,
    include_regex: str | None,
    exclude_regex: str | None,
) -> tuple[list[str], dict[str, Any]]:
    include_text = str(include_regex or "").strip()
    exclude_text = str(exclude_regex or "").strip()
    include_pattern = re.compile(include_text) if include_text else None
    exclude_pattern = re.compile(exclude_text) if exclude_text else None
    kept: list[str] = []
    for feature in features:
        if include_pattern is not None and include_pattern.search(feature) is None:
            continue
        if exclude_pattern is not None and exclude_pattern.search(feature) is not None:
            continue
        kept.append(feature)
    return kept, {
        "feature_include_regex": include_text,
        "feature_exclude_regex": exclude_text,
        "feature_count_before_filter": int(len(features)),
        "feature_count_after_filter": int(len(kept)),
    }


def _sigmoid(values: Any) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0))))


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    ratio = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _target_for_spec(
    metrics: pd.DataFrame,
    spec: TargetSpec,
    ts: pd.Series,
    *,
    target_mode: str = "soft_rank_blend",
    target_rank_weight: float = 0.30,
    target_soft_power: float = 1.0,
) -> pd.DataFrame:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02).reset_index(drop=True)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0).reset_index(drop=True)
    barrier = _safe_numeric(metrics["barrier"]).fillna(1.0).reset_index(drop=True)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).reset_index(drop=True)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0).reset_index(drop=True)
    mfe_mae = _mfe_mae(metrics).fillna(0.0).reset_index(drop=True)

    utility_soft = _sigmoid((u - float(spec.u_floor)) / max(float(spec.u_temp), 1e-8))
    mae_soft = _sigmoid((float(spec.mae_cap) - mae) / max(float(spec.mae_temp), 1e-8))
    ratio_soft = _sigmoid((mfe_mae - float(spec.mfe_mae_min)) / max(float(spec.mfe_mae_temp), 1e-8))
    barrier_soft = _sigmoid((float(spec.barrier_cap) - barrier) / max(float(spec.barrier_temp), 1e-8))
    if float(spec.bars_cap) > 0.0:
        bars_soft = _sigmoid((float(spec.bars_cap) - bars) / max(float(spec.bars_temp), 1e-8))
    else:
        bars_soft = pd.Series(1.0, index=u.index)

    timeout_mode = str(spec.timeout_mode)
    if timeout_mode == "hard":
        timeout_soft = 1.0 - timeout.clip(0.0, 1.0)
    elif timeout_mode == "soft":
        timeout_soft = 1.0 - 0.50 * timeout.clip(0.0, 1.0)
    elif timeout_mode == "none":
        timeout_soft = pd.Series(1.0, index=u.index)
    else:
        raise ValueError(f"Unknown timeout mode: {timeout_mode}")

    raw_soft = (
        utility_soft
        * mae_soft
        * ratio_soft
        * barrier_soft
        * bars_soft
        * timeout_soft
    ).clip(0.0, 1.0)
    hard = (
        (u > float(spec.u_floor))
        & (mae <= float(spec.mae_cap))
        & (mfe_mae >= float(spec.mfe_mae_min))
        & (barrier <= float(spec.barrier_cap))
    )
    if timeout_mode == "hard":
        hard = hard & timeout.le(0.0)
    if float(spec.bars_cap) > 0.0:
        hard = hard & bars.le(float(spec.bars_cap))
    hard_float = hard.fillna(False).astype(float)

    power = max(float(target_soft_power), 1e-8)
    raw_soft = raw_soft.pow(power).clip(0.0, 1.0)
    mode = str(target_mode)
    if mode == "soft_rank_blend":
        rank_weight = min(max(float(target_rank_weight), 0.0), 1.0)
        ts_reset = pd.to_datetime(ts, utc=True, errors="coerce").reset_index(drop=True)
        fallback_rank = raw_soft.rank(method="average", pct=True)
        ts_rank = raw_soft.groupby(ts_reset, dropna=False).rank(method="average", pct=True)
        target_soft = ((1.0 - rank_weight) * raw_soft + rank_weight * ts_rank.fillna(fallback_rank)).clip(0.0, 1.0)
    elif mode == "raw_soft":
        target_soft = raw_soft
    elif mode == "hard_binary":
        target_soft = hard_float
    elif mode == "hard_utility_soft":
        target_soft = (hard_float * utility_soft).clip(0.0, 1.0)
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return pd.DataFrame(
        {
            "target_soft": target_soft.astype(np.float32),
            "target_hard": hard_float.astype(np.float32),
        },
        index=metrics.index,
    )


def _build_specs(
    *,
    mae_caps: list[float],
    mfe_mae_mins: list[float],
    barrier_caps: list[float],
    timeout_modes: list[str],
    bars_caps: list[float],
    u_floor: float,
    u_temp: float,
) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for mae_cap in mae_caps:
        for ratio_min in mfe_mae_mins:
            for barrier_cap in barrier_caps:
                for timeout_mode in timeout_modes:
                    for bars_cap in bars_caps:
                        name = (
                            f"pathgrid_u{int(round(u_floor * 10000)):04d}"
                            f"_mae{int(round(mae_cap * 100)):03d}"
                            f"_mm{int(round(ratio_min * 100)):03d}"
                            f"_bar{int(round(barrier_cap * 10000)):03d}"
                            f"_to{timeout_mode}"
                            f"_bars{int(round(bars_cap)):02d}"
                        )
                        specs.append(
                            TargetSpec(
                                name=name,
                                u_floor=float(u_floor),
                                u_temp=float(u_temp),
                                mae_cap=float(mae_cap),
                                mae_temp=0.35,
                                mfe_mae_min=float(ratio_min),
                                mfe_mae_temp=0.35,
                                barrier_cap=float(barrier_cap),
                                barrier_temp=0.006,
                                timeout_mode=str(timeout_mode),
                                bars_cap=float(bars_cap),
                                bars_temp=6.0,
                            )
                        )
    return specs


def _prevalence_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    spec: TargetSpec,
    months: list[str],
) -> list[dict[str, Any]]:
    period = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for month in months:
        mask = period.eq(str(month))
        local_metrics = metrics.loc[mask]
        local_target = target.loc[mask]
        mfe_mae = _mfe_mae(local_metrics)
        rows.append(
            {
                "candidate": spec.name,
                "month": str(month),
                "rows": int(mask.sum()),
                "target_soft_mean": _safe_mean(local_target["target_soft"]),
                "target_soft_std": float(local_target["target_soft"].std()) if len(local_target) else float("nan"),
                "target_hard_rate": _safe_mean(local_target["target_hard"]),
                "mean_return_net": _safe_mean(local_metrics["ret_net"]),
                "mean_u": _safe_mean(local_metrics["u_policy_net"]),
                "hit_u": _safe_mean(local_metrics["u_policy_net"] > 0.0),
                "bad_mae_1r_rate": _safe_mean(local_metrics["mae_norm"] >= 1.0),
                "p90_mae_norm": _safe_quantile(local_metrics["mae_norm"], 0.90),
                "timeout_rate": _safe_mean(local_metrics["is_timeout"].astype(float)),
                "profit_low_mae_no_timeout_rate": _safe_mean(
                    (local_metrics["u_policy_net"] > 0.0)
                    & (local_metrics["mae_norm"] <= 1.0)
                    & (local_metrics["is_timeout"].astype(float) <= 0.0)
                ),
                "decisive_profit_low_mae_rate": _safe_mean(
                    (local_metrics["u_policy_net"] > 0.0)
                    & (local_metrics["mae_norm"] <= 1.0)
                    & (mfe_mae >= 1.25)
                    & (local_metrics["is_timeout"].astype(float) <= 0.0)
                ),
            }
        )
    return rows


def _run_candidate(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    spec: TargetSpec,
    months: list[str],
    top_fracs: list[float],
    proxy_top_k: int,
    proxy_objective: str,
    proxy_min_target_ic: float,
    proxy_min_utility_ic: float,
    proxy_max_bad_mae_ic: float,
    proxy_max_wide_ic: float,
    proxy_max_timeout_ic: float,
    proxy_utility_weight: float,
    proxy_bad_mae_weight: float,
    proxy_wide_weight: float,
    proxy_timeout_weight: float,
    min_train_rows: int,
    min_valid_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    period = frame["__ts__"].dt.to_period("M").astype(str)
    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    for month in months:
        train_mask = period.lt(str(month))
        valid_mask = period.eq(str(month))
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
        if str(proxy_objective) == "target_ic":
            proxy_score, diag = _score_proxy(
                train=train,
                valid=frame.loc[valid_mask].copy(),
                features=features,
                y_train=target.loc[train_mask, "target_soft"],
                proxy_top_k=int(proxy_top_k),
            )
            selector_name = "fit_ic_proxy_oos"
        else:
            proxy_score, diag = _economic_proxy_score(
                train=train,
                valid=frame.loc[valid_mask].copy(),
                features=features,
                target_train=target.loc[train_mask, "target_soft"],
                metrics_train=metrics.loc[train_mask].copy(),
                top_k=int(proxy_top_k),
                proxy_objective=str(proxy_objective),
                min_target_ic=float(proxy_min_target_ic),
                min_utility_ic=float(proxy_min_utility_ic),
                max_bad_mae_ic=float(proxy_max_bad_mae_ic),
                max_wide_ic=float(proxy_max_wide_ic),
                max_timeout_ic=float(proxy_max_timeout_ic),
                utility_weight=float(proxy_utility_weight),
                bad_mae_weight=float(proxy_bad_mae_weight),
                wide_weight=float(proxy_wide_weight),
                timeout_weight=float(proxy_timeout_weight),
            )
            selector_name = f"{proxy_objective}_proxy_oos"
        proxy_score = proxy_score.reset_index(drop=True)
        selector_specs = [
            ("oracle_target_sort", valid_target["target_soft"], "", 1.0),
            (selector_name, proxy_score, ",".join(diag.get("proxy_features", [])), 0.0),
        ]
        proxy_ic_rows.append(
            {
                "candidate": spec.name,
                "month": str(month),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "proxy_ic_target": _spearman(proxy_score, valid_target["target_soft"]),
                "proxy_ic_u": _spearman(proxy_score, valid_metrics["u_policy_net"]),
                "proxy_ic_bad_mae": _spearman(proxy_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                "proxy_ic_timeout": _spearman(proxy_score, valid_metrics["is_timeout"].astype(float)),
                "proxy_objective": diag.get("proxy_objective", proxy_objective),
                "proxy_candidate_count": diag.get("proxy_candidate_count"),
                "proxy_features": ",".join(diag.get("proxy_features", [])),
                "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
                "proxy_top_ranking_score": diag.get("proxy_top_ranking_score"),
                "proxy_mean_ranking_score": diag.get("proxy_mean_ranking_score"),
                "proxy_mean_train_target_ic": diag.get("proxy_mean_train_target_ic"),
                "proxy_mean_train_utility_ic": diag.get("proxy_mean_train_utility_ic"),
                "proxy_mean_train_bad_mae_ic": diag.get("proxy_mean_train_bad_mae_ic"),
                "proxy_mean_train_wide_ic": diag.get("proxy_mean_train_wide_ic"),
                "proxy_mean_train_timeout_ic": diag.get("proxy_mean_train_timeout_ic"),
            }
        )
        period_slices = [("month", str(month), np.arange(len(valid), dtype=np.int64))]
        period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
        for selector, score, proxy_features, oracle_flag in selector_specs:
            score = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
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
                        selector=selector,
                        label_arm=spec.name,
                        economic_arm="path_grid",
                        top_frac=float(frac),
                        label_score=valid_target["target_soft"].iloc[pos].reset_index(drop=True),
                        economic_score=None,
                        economic_target=None,
                        label_proxy_features=proxy_features,
                        economic_proxy_features="",
                    )
                    row["oracle_selector"] = bool(oracle_flag)
                    period_rows.append(row)
    return period_rows, proxy_ic_rows


def _candidate_summary(
    *,
    prevalence: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    specs: list[TargetSpec],
    holdout_month: str,
) -> pd.DataFrame:
    spec_map = {spec.name: spec.to_dict() for spec in specs}
    rows: list[dict[str, Any]] = []
    for candidate, params in spec_map.items():
        prev = prevalence[prevalence["candidate"].astype(str).eq(candidate)].copy()
        fit = fit_holdout[fit_holdout["label_arm"].astype(str).eq(candidate)].copy()
        non_oracle = fit[~fit["selector"].astype(str).str.startswith("oracle_")].copy()
        oracle = fit[fit["selector"].astype(str).str.startswith("oracle_")].copy()
        june_prev = prev[prev["month"].astype(str).eq(str(holdout_month))]
        june_ic = proxy_ic[
            proxy_ic["candidate"].astype(str).eq(candidate)
            & proxy_ic["month"].astype(str).eq(str(holdout_month))
        ]
        best_non = (
            non_oracle.sort_values(
                ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_mean_return_net"],
                ascending=[False, False, False, False],
            ).head(1)
            if not non_oracle.empty
            else pd.DataFrame()
        )
        best_oracle = (
            oracle.sort_values(
                ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_mean_return_net"],
                ascending=[False, False, False, False],
            ).head(1)
            if not oracle.empty
            else pd.DataFrame()
        )
        row: dict[str, Any] = {
            "candidate": candidate,
            **params,
            "june_target_hard_rate": float(june_prev["target_hard_rate"].iloc[0]) if len(june_prev) else float("nan"),
            "june_target_soft_std": float(june_prev["target_soft_std"].iloc[0]) if len(june_prev) else float("nan"),
            "june_proxy_ic_target": _safe_mean(june_ic["proxy_ic_target"]) if len(june_ic) else float("nan"),
            "june_proxy_ic_u": _safe_mean(june_ic["proxy_ic_u"]) if len(june_ic) else float("nan"),
            "non_oracle_trainworthy": int(non_oracle["trainworthy_pass"].sum()) if not non_oracle.empty else 0,
            "non_oracle_fit_economic": int(non_oracle["fit_economic_pass"].sum()) if not non_oracle.empty else 0,
            "non_oracle_holdout_economic": int(non_oracle["holdout_economic_pass"].sum()) if not non_oracle.empty else 0,
            "oracle_trainworthy": int(oracle["trainworthy_pass"].sum()) if not oracle.empty else 0,
        }
        if not best_non.empty:
            best = best_non.iloc[0]
            for col in [
                "selector",
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
            ]:
                row[f"best_non_oracle_{col}"] = best.get(col)
        if not best_oracle.empty:
            best = best_oracle.iloc[0]
            row["best_oracle_holdout_mean_return_net"] = best.get("holdout_mean_return_net")
            row["best_oracle_holdout_bad_mae_1r_rate"] = best.get("holdout_bad_mae_1r_rate")
            row["best_oracle_holdout_p90_mae_norm"] = best.get("holdout_p90_mae_norm")
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "non_oracle_trainworthy",
            "non_oracle_holdout_economic",
            "non_oracle_fit_economic",
            "best_non_oracle_holdout_mean_return_net",
            "june_proxy_ic_u",
        ],
        ascending=[False, False, False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    candidate_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    prevalence: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "path_aware_label_target_grid.md"
    non_oracle = fit_holdout[~fit_holdout["selector"].astype(str).str.startswith("oracle_")].copy()
    lines = [
        "# Path-Aware Label Target Grid",
        "",
        "Scope: proxy-only target-definition ablation. No model training, Optuna, or policy optimisation.",
        "",
        f"Months: `{', '.join(manifest['months'])}`. Fit: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Candidates: `{manifest['candidate_count']}`. Features: `{manifest['feature_count']}`. Proxy top-k: `{manifest['proxy_top_k']}`.",
        "",
        "## Candidate Summary",
        "",
        _table(
            candidate_summary,
            [
                "candidate",
                "non_oracle_trainworthy",
                "non_oracle_fit_economic",
                "non_oracle_holdout_economic",
                "oracle_trainworthy",
                "june_target_hard_rate",
                "june_proxy_ic_u",
                "best_non_oracle_selector",
                "best_non_oracle_top_frac",
                "best_non_oracle_fit_mean_return_net",
                "best_non_oracle_holdout_mean_return_net",
                "best_non_oracle_fit_bad_mae_1r_rate",
                "best_non_oracle_holdout_bad_mae_1r_rate",
                "best_non_oracle_fit_p90_mae_norm",
                "best_non_oracle_holdout_p90_mae_norm",
                "best_non_oracle_fit_timeout_rate",
                "best_non_oracle_holdout_timeout_rate",
            ],
            limit=80,
        ),
        "",
        "## Best Non-Oracle Fit/Holdout Rows",
        "",
        _table(
            non_oracle.sort_values(
                ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_mean_return_net"],
                ascending=[False, False, False, False],
            ),
            [
                "trainworthy_pass",
                "fit_economic_pass",
                "holdout_economic_pass",
                "fit_sign_pass",
                "holdout_sign_pass",
                "selector",
                "label_arm",
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
        "## June Target Prevalence",
        "",
        _table(
            prevalence[prevalence["month"].astype(str).eq(str(manifest["holdout_month"]))].sort_values(
                "target_hard_rate",
                ascending=False,
            ),
            [
                "candidate",
                "target_hard_rate",
                "target_soft_mean",
                "target_soft_std",
                "mean_return_net",
                "bad_mae_1r_rate",
                "p90_mae_norm",
                "timeout_rate",
                "profit_low_mae_no_timeout_rate",
                "decisive_profit_low_mae_rate",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Candidate summary: `{manifest['outputs']['candidate_summary']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Prevalence: `{manifest['outputs']['prevalence']}`",
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
    proxy_objective: str,
    proxy_min_target_ic: float,
    proxy_min_utility_ic: float,
    proxy_max_bad_mae_ic: float,
    proxy_max_wide_ic: float,
    proxy_max_timeout_ic: float,
    proxy_utility_weight: float,
    proxy_bad_mae_weight: float,
    proxy_wide_weight: float,
    proxy_timeout_weight: float,
    min_train_rows: int,
    min_valid_rows: int,
    max_timeout_rate: float | None,
    mae_caps: list[float],
    mfe_mae_mins: list[float],
    barrier_caps: list[float],
    timeout_modes: list[str],
    bars_caps: list[float],
    u_floor: float,
    u_temp: float,
    target_mode: str,
    target_rank_weight: float,
    target_soft_power: float,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    feature_include_regex: str | None,
    feature_exclude_regex: str | None,
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
    features, feature_filter_report = _filter_features(
        features,
        include_regex=feature_include_regex,
        exclude_regex=feature_exclude_regex,
    )
    if max_feature_columns is not None and int(max_feature_columns) > 0:
        features = features[: int(max_feature_columns)]
        feature_filter_report["feature_count_after_max_columns"] = int(len(features))
    else:
        feature_filter_report["feature_count_after_max_columns"] = int(len(features))
    specs = _build_specs(
        mae_caps=mae_caps,
        mfe_mae_mins=mfe_mae_mins,
        barrier_caps=barrier_caps,
        timeout_modes=timeout_modes,
        bars_caps=bars_caps,
        u_floor=u_floor,
        u_temp=u_temp,
    )

    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        target = _target_for_spec(
            metrics.reset_index(drop=True),
            spec,
            frame["__ts__"].reset_index(drop=True),
            target_mode=target_mode,
            target_rank_weight=target_rank_weight,
            target_soft_power=target_soft_power,
        )
        target.index = frame.index
        prevalence_rows.extend(
            _prevalence_rows(frame=frame, metrics=metrics, target=target, spec=spec, months=months)
        )
        cur_period, cur_proxy_ic = _run_candidate(
            frame=frame,
            metrics=metrics,
            target=target,
            features=features,
            spec=spec,
            months=months,
            top_fracs=top_fracs,
            proxy_top_k=proxy_top_k,
            proxy_objective=proxy_objective,
            proxy_min_target_ic=proxy_min_target_ic,
            proxy_min_utility_ic=proxy_min_utility_ic,
            proxy_max_bad_mae_ic=proxy_max_bad_mae_ic,
            proxy_max_wide_ic=proxy_max_wide_ic,
            proxy_max_timeout_ic=proxy_max_timeout_ic,
            proxy_utility_weight=proxy_utility_weight,
            proxy_bad_mae_weight=proxy_bad_mae_weight,
            proxy_wide_weight=proxy_wide_weight,
            proxy_timeout_weight=proxy_timeout_weight,
            min_train_rows=min_train_rows,
            min_valid_rows=min_valid_rows,
        )
        period_rows.extend(cur_period)
        proxy_ic_rows.extend(cur_proxy_ic)
        if idx == 1 or idx % 12 == 0 or idx == len(specs):
            print(json.dumps({"progress": f"{idx}/{len(specs)}", "candidate": spec.name}, sort_keys=True))

    period_frame = pd.DataFrame(period_rows)
    fit_holdout = _fit_holdout_summary(
        period_frame,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=10,
        max_timeout_rate=max_timeout_rate,
    )
    prevalence = pd.DataFrame(prevalence_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    summary = _candidate_summary(
        prevalence=prevalence,
        fit_holdout=fit_holdout,
        proxy_ic=proxy_ic,
        specs=specs,
        holdout_month=holdout_month,
    )

    paths = {
        "candidate_summary": output_dir / "path_aware_label_target_candidate_summary.csv",
        "fit_holdout": output_dir / "path_aware_label_target_fit_holdout.csv",
        "period_rows": output_dir / "path_aware_label_target_period_rows.csv",
        "prevalence": output_dir / "path_aware_label_target_prevalence.csv",
        "proxy_ic": output_dir / "path_aware_label_target_proxy_ic.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["candidate_summary"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    period_frame.to_csv(paths["period_rows"], index=False)
    prevalence.to_csv(paths["prevalence"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)

    manifest = {
        "scope": "path_aware_label_target_grid",
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
        "feature_filter": feature_filter_report,
        "candidate_count": int(len(specs)),
        "months": list(months),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "proxy_top_k": int(proxy_top_k),
        "proxy_objective": str(proxy_objective),
        "proxy_min_target_ic": float(proxy_min_target_ic),
        "proxy_min_utility_ic": float(proxy_min_utility_ic),
        "proxy_max_bad_mae_ic": float(proxy_max_bad_mae_ic),
        "proxy_max_wide_ic": float(proxy_max_wide_ic),
        "proxy_max_timeout_ic": float(proxy_max_timeout_ic),
        "proxy_utility_weight": float(proxy_utility_weight),
        "proxy_bad_mae_weight": float(proxy_bad_mae_weight),
        "proxy_wide_weight": float(proxy_wide_weight),
        "proxy_timeout_weight": float(proxy_timeout_weight),
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "max_timeout_rate": max_timeout_rate,
        "grid": {
            "mae_caps": [float(v) for v in mae_caps],
            "mfe_mae_mins": [float(v) for v in mfe_mae_mins],
            "barrier_caps": [float(v) for v in barrier_caps],
            "timeout_modes": list(timeout_modes),
            "bars_caps": [float(v) for v in bars_caps],
            "u_floor": float(u_floor),
            "u_temp": float(u_temp),
            "target_mode": str(target_mode),
            "target_rank_weight": float(target_rank_weight),
            "target_soft_power": float(target_soft_power),
        },
        "reports": reports,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        candidate_summary=summary,
        fit_holdout=fit_holdout,
        prevalence=prevalence,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "path_aware_label_target_grid.md")}},
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
    parser.add_argument("--feature-include-regex", default="")
    parser.add_argument("--feature-exclude-regex", default="")
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=list(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--proxy-objective", choices=PROXY_OBJECTIVES, default="target_ic")
    parser.add_argument("--proxy-min-target-ic", type=float, default=0.0)
    parser.add_argument("--proxy-min-utility-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-bad-mae-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-wide-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-timeout-ic", type=float, default=0.0)
    parser.add_argument("--proxy-utility-weight", type=float, default=1.0)
    parser.add_argument("--proxy-bad-mae-weight", type=float, default=1.0)
    parser.add_argument("--proxy-wide-weight", type=float, default=0.5)
    parser.add_argument("--proxy-timeout-weight", type=float, default=0.5)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--max-timeout-rate", type=float, default=0.50)
    parser.add_argument("--mae-caps", type=lambda value: _parse_float_csv(value), default=[1.0, 1.25, 1.5])
    parser.add_argument("--mfe-mae-mins", type=lambda value: _parse_float_csv(value), default=[1.0, 1.25])
    parser.add_argument("--barrier-caps", type=lambda value: _parse_float_csv(value), default=[0.025, 0.035])
    parser.add_argument("--timeout-modes", type=lambda value: _parse_csv(value, ("none", "soft", "hard")), default=["none", "soft", "hard"])
    parser.add_argument("--bars-caps", type=lambda value: _parse_float_csv(value), default=[0.0, 12.0])
    parser.add_argument("--u-floor", type=float, default=0.0)
    parser.add_argument("--u-temp", type=float, default=0.008)
    parser.add_argument(
        "--target-mode",
        choices=("soft_rank_blend", "raw_soft", "hard_binary", "hard_utility_soft"),
        default="soft_rank_blend",
    )
    parser.add_argument("--target-rank-weight", type=float, default=0.30)
    parser.add_argument("--target-soft-power", type=float, default=1.0)
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
        proxy_objective=str(args.proxy_objective),
        proxy_min_target_ic=float(args.proxy_min_target_ic),
        proxy_min_utility_ic=float(args.proxy_min_utility_ic),
        proxy_max_bad_mae_ic=float(args.proxy_max_bad_mae_ic),
        proxy_max_wide_ic=float(args.proxy_max_wide_ic),
        proxy_max_timeout_ic=float(args.proxy_max_timeout_ic),
        proxy_utility_weight=float(args.proxy_utility_weight),
        proxy_bad_mae_weight=float(args.proxy_bad_mae_weight),
        proxy_wide_weight=float(args.proxy_wide_weight),
        proxy_timeout_weight=float(args.proxy_timeout_weight),
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        max_timeout_rate=args.max_timeout_rate,
        mae_caps=list(args.mae_caps),
        mfe_mae_mins=list(args.mfe_mae_mins),
        barrier_caps=list(args.barrier_caps),
        timeout_modes=list(args.timeout_modes),
        bars_caps=list(args.bars_caps),
        u_floor=float(args.u_floor),
        u_temp=float(args.u_temp),
        target_mode=str(args.target_mode),
        target_rank_weight=float(args.target_rank_weight),
        target_soft_power=float(args.target_soft_power),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        feature_include_regex=str(args.feature_include_regex),
        feature_exclude_regex=str(args.feature_exclude_regex),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
