#!/usr/bin/env python3
"""Leakage-safe soft AE/GMM state-risk overlay for S52 ranker ledgers.

This is a follow-up to the hard single-feature state blocklist. It learns
side-specific bucket risk on fit months only, converts those bucket risks into a
continuous pre-entry state penalty, and re-ranks by:

    adjusted_score = score - alpha * state_path_risk

Selection budgets are preserved per month x side so the test separates ranking
quality from simply trading less.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_ranker_smoke import _state_feature_columns  # noqa: E402


DEFAULT_LEDGER = Path(
    "data_perp/reports/s52_ordered_clean_ev_ranker_smoke_tp100_sl050_20260705_v1/"
    "s52_ranker_smoke_scored_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_state_risk_penalty_overlay_v1")


def _parse_csv(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...] = ()) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_numeric(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    ser = _safe_numeric(values)
    ser = ser[np.isfinite(ser.to_numpy(dtype=np.float64))]
    return float(ser.mean()) if len(ser) else float("nan")


def _fit_bucket_spec(series: pd.Series, *, q: int = 5) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
    name = str(series.name or "").lower()
    unique = finite.drop_duplicates()
    discrete = len(unique) <= 12 or name.endswith("cluster_id") or "cluster_t" in name
    if discrete:
        return {"kind": "discrete"}
    if len(finite) < max(20, int(q) * 4):
        return {"kind": "all"}
    try:
        _, edges = pd.qcut(finite, q=int(q), duplicates="drop", retbins=True)
    except ValueError:
        return {"kind": "all"}
    edges = np.asarray(edges, dtype=np.float64)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(edges)
    if len(edges) < 3:
        return {"kind": "all"}
    edges[0] = -np.inf
    edges[-1] = np.inf
    return {"kind": "continuous", "edges": edges.tolist()}


def _apply_bucket_spec(series: pd.Series, spec: dict[str, Any]) -> pd.Series:
    kind = str(spec.get("kind", "all"))
    values = pd.to_numeric(series, errors="coerce")
    if kind == "discrete":
        return values.round(0).astype("Int64").astype(str).replace("<NA>", "missing")
    if kind == "continuous":
        edges = np.asarray(spec.get("edges", []), dtype=np.float64)
        if len(edges) >= 3:
            return pd.cut(values, bins=edges, include_lowest=True).astype(str).replace("nan", "missing")
    return pd.Series("all", index=series.index, dtype=object)


def _metrics(frame: pd.DataFrame, *, round_trip_cost: float) -> dict[str, float]:
    if frame.empty:
        return {
            "selected_rows": 0.0,
            "mean_u": float("nan"),
            "ev_weighted_first_touch_precision": float("nan"),
            "first_pass_good_rate": float("nan"),
            "first_pass_bad_rate": float("nan"),
            "first_touch_bad_mae_1r_rate": float("nan"),
            "mfe_before_mae_1r_rate": float("nan"),
            "mae_before_mfe_1r_rate": float("nan"),
            "mean_max_adverse_before_mfe_1r": float("nan"),
            "mean_underwater_bars_before_mfe": float("nan"),
            "mean_underwater_fraction_before_mfe": float("nan"),
            "timeout_rate": float("nan"),
        }
    gross = _safe_numeric(frame.get("first_touch_net", pd.Series(0.0, index=frame.index))).fillna(0.0)
    gross = gross + float(round_trip_cost)
    good = _safe_numeric(frame.get("first_pass_good", pd.Series(0.0, index=frame.index))).fillna(0.0).gt(0.5)
    denom = float(gross.abs().sum())
    evw = float(gross.where(good, 0.0).clip(lower=0.0).sum() / denom) if denom > 1e-12 else float("nan")
    return {
        "selected_rows": float(len(frame)),
        "mean_u": _safe_mean(frame.get("u_policy_net", [])),
        "ev_weighted_first_touch_precision": evw,
        "first_pass_good_rate": _safe_mean(good.astype(float)),
        "first_pass_bad_rate": _safe_mean(
            _safe_numeric(frame.get("first_pass_bad", pd.Series(dtype=float))).fillna(0.0)
        ),
        "first_touch_bad_mae_1r_rate": _safe_mean(
            _safe_numeric(frame.get("first_touch_mae_norm", pd.Series(dtype=float))).ge(1.0).astype(float)
        ),
        "mfe_before_mae_1r_rate": _safe_mean(frame.get("mfe_1r_before_mae_1r", [])),
        "mae_before_mfe_1r_rate": _safe_mean(frame.get("mae_1r_before_mfe_1r", [])),
        "mean_max_adverse_before_mfe_1r": _safe_mean(frame.get("max_adverse_before_mfe_1r", [])),
        "mean_underwater_bars_before_mfe": _safe_mean(frame.get("underwater_bars_before_mfe_1r", [])),
        "mean_underwater_fraction_before_mfe": _safe_mean(frame.get("underwater_fraction_before_mfe_1r", [])),
        "timeout_rate": _safe_mean(
            _safe_numeric(frame.get("is_timeout", pd.Series(dtype=float))).fillna(0.0).gt(0.5).astype(float)
        ),
    }


def _prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _bucket_risk(
    bucket_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    *,
    mae_weight: float,
    adverse_weight: float,
    underwater_weight: float,
    ev_weight: float,
    mean_u_weight: float,
) -> float:
    risk = 0.0
    mae = bucket_metrics.get("mae_before_mfe_1r_rate", float("nan"))
    base_mae = baseline_metrics.get("mae_before_mfe_1r_rate", float("nan"))
    if math.isfinite(mae) and math.isfinite(base_mae):
        risk += float(mae_weight) * max(0.0, mae - base_mae)
    adverse = bucket_metrics.get("mean_max_adverse_before_mfe_1r", float("nan"))
    base_adverse = baseline_metrics.get("mean_max_adverse_before_mfe_1r", float("nan"))
    if math.isfinite(adverse) and math.isfinite(base_adverse):
        risk += float(adverse_weight) * max(0.0, adverse - base_adverse) / 2.0
    underwater = bucket_metrics.get("mean_underwater_bars_before_mfe", float("nan"))
    base_underwater = baseline_metrics.get("mean_underwater_bars_before_mfe", float("nan"))
    if math.isfinite(underwater) and math.isfinite(base_underwater):
        risk += float(underwater_weight) * max(0.0, underwater - base_underwater) / 12.0
    evw = bucket_metrics.get("ev_weighted_first_touch_precision", float("nan"))
    base_evw = baseline_metrics.get("ev_weighted_first_touch_precision", float("nan"))
    if math.isfinite(evw) and math.isfinite(base_evw):
        risk += float(ev_weight) * max(0.0, base_evw - evw)
    mean_u = bucket_metrics.get("mean_u", float("nan"))
    base_u = baseline_metrics.get("mean_u", float("nan"))
    if math.isfinite(mean_u) and math.isfinite(base_u):
        risk += float(mean_u_weight) * max(0.0, base_u - mean_u)
    return float(max(risk, 0.0))


def _learn_state_risk_rules(
    ledger: pd.DataFrame,
    *,
    selected_col: str,
    fit_months: list[str],
    sides: set[str],
    min_fit_selected_rows: int,
    round_trip_cost: float,
    max_features: int,
    mae_weight: float,
    adverse_weight: float,
    underwater_weight: float,
    ev_weight: float,
    mean_u_weight: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    fit_mask = ledger["month"].astype(str).isin([str(m) for m in fit_months])
    selected_fit = ledger[fit_mask & ledger[selected_col].astype(bool)].copy()
    state_cols = _state_feature_columns(ledger.columns)
    if int(max_features) > 0:
        state_cols = state_cols[: int(max_features)]
    specs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    baseline_by_side = {
        str(side): _metrics(group, round_trip_cost=float(round_trip_cost))
        for side, group in selected_fit.groupby("side_name", observed=True, dropna=False)
    }
    for feature in state_cols:
        spec = _fit_bucket_spec(ledger.loc[fit_mask, feature])
        specs[feature] = spec
        bucketed = _apply_bucket_spec(ledger[feature], spec)
        local = selected_fit.assign(__bucket__=bucketed.loc[selected_fit.index].to_numpy())
        for (side, bucket), group in local.groupby(["side_name", "__bucket__"], observed=True, dropna=False):
            side_str = str(side)
            if sides and side_str not in sides:
                continue
            if len(group) < int(min_fit_selected_rows):
                continue
            metrics = _metrics(group, round_trip_cost=float(round_trip_cost))
            baseline = baseline_by_side.get(side_str, {})
            risk = _bucket_risk(
                metrics,
                baseline,
                mae_weight=float(mae_weight),
                adverse_weight=float(adverse_weight),
                underwater_weight=float(underwater_weight),
                ev_weight=float(ev_weight),
                mean_u_weight=float(mean_u_weight),
            )
            if risk <= 0.0:
                continue
            rows.append(
                {
                    "state_feature": str(feature),
                    "side": side_str,
                    "bucket": str(bucket),
                    "risk": float(risk),
                    **{f"fit_{key}": value for key, value in metrics.items()},
                    **{f"side_baseline_{key}": value for key, value in baseline.items()},
                    "bucket_spec": json.dumps(_json_safe(spec), sort_keys=True),
                }
            )
    rules = pd.DataFrame(rows)
    if not rules.empty:
        rules = rules.sort_values(["risk", "fit_selected_rows"], ascending=[False, False]).reset_index(drop=True)
    return rules, specs


def _score_state_risk(
    ledger: pd.DataFrame,
    rules: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    *,
    combine_mode: str,
    top_n: int,
) -> pd.Series:
    if rules.empty:
        return pd.Series(0.0, index=ledger.index, dtype=np.float32)
    risk_lists: list[list[float]] = [[] for _ in range(len(ledger))]
    for feature, feature_rules in rules.groupby("state_feature", observed=True, dropna=False):
        feature = str(feature)
        if feature not in ledger.columns:
            continue
        bucketed = _apply_bucket_spec(ledger[feature], specs.get(feature, {"kind": "all"}))
        risk_map = {
            (str(row.side), str(row.bucket)): float(row.risk)
            for row in feature_rules.itertuples(index=False)
        }
        for pos, (side, bucket) in enumerate(zip(ledger["side_name"].astype(str), bucketed.astype(str))):
            risk = risk_map.get((str(side), str(bucket)), 0.0)
            if risk > 0.0:
                risk_lists[pos].append(float(risk))
    mode = str(combine_mode or "mean_top3").strip().lower()
    values = np.zeros(len(ledger), dtype=np.float32)
    n = max(1, int(top_n))
    for i, risks in enumerate(risk_lists):
        if not risks:
            continue
        arr = np.sort(np.asarray(risks, dtype=np.float64))[::-1]
        if mode == "max":
            values[i] = float(arr[0])
        elif mode == "sum":
            values[i] = float(arr[:n].sum())
        else:
            values[i] = float(arr[:n].mean())
    return pd.Series(values, index=ledger.index, dtype=np.float32)


def _select_budgeted(
    ledger: pd.DataFrame,
    *,
    selected_col: str,
    adjusted_score_col: str,
    sides: set[str],
    group_cols: list[str],
) -> pd.Series:
    selected = pd.Series(False, index=ledger.index)
    for _, group in ledger.groupby(group_cols, observed=True, dropna=False, sort=False):
        side = str(group["side_name"].iloc[0]) if "side_name" in group.columns and len(group) else ""
        if sides and side not in sides:
            selected.loc[group.index] = group[selected_col].astype(bool)
            continue
        budget = int(group[selected_col].astype(bool).sum())
        if budget <= 0:
            continue
        ordered = group.sort_values(adjusted_score_col, ascending=False, kind="mergesort")
        selected.loc[ordered.head(budget).index] = True
    return selected.astype(bool)


def _filter_selected_by_risk(
    ledger: pd.DataFrame,
    *,
    selected_col: str,
    risk: pd.Series,
    threshold: float,
    sides: set[str],
) -> pd.Series:
    selected = ledger[selected_col].astype(bool).copy()
    target_side = ledger["side_name"].astype(str).isin(sides) if sides else pd.Series(True, index=ledger.index)
    selected.loc[target_side] = selected.loc[target_side] & risk.loc[target_side].le(float(threshold))
    return selected.astype(bool)


def _slice_metrics(
    ledger: pd.DataFrame,
    selected_mask: pd.Series,
    *,
    months: list[str],
    side: str | None,
    round_trip_cost: float,
) -> dict[str, float]:
    mask = ledger["month"].astype(str).isin([str(m) for m in months]) & selected_mask.astype(bool)
    if side:
        mask = mask & ledger["side_name"].astype(str).eq(str(side))
    return _metrics(ledger.loc[mask], round_trip_cost=float(round_trip_cost))


def run_overlay(
    *,
    ledger_path: Path,
    output_dir: Path,
    variant: str,
    selected_cols: list[str],
    fit_months: list[str],
    holdout_month: str,
    alphas: list[float],
    sides: list[str],
    combine_modes: list[str],
    group_cols: list[str],
    round_trip_cost: float,
    min_fit_selected_rows: int,
    max_features: int,
    top_n: int,
    filter_thresholds: list[float],
    mae_weight: float,
    adverse_weight: float,
    underwater_weight: float,
    ev_weight: float,
    mean_u_weight: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(ledger_path)
    if variant:
        ledger = ledger[ledger["variant"].astype(str).eq(str(variant))].copy()
    ledger = ledger.reset_index(drop=True)
    side_set = {str(side).strip() for side in sides if str(side).strip()}
    summaries: list[dict[str, Any]] = []
    all_rules: list[pd.DataFrame] = []
    for selected_col in selected_cols:
        if selected_col not in ledger.columns:
            raise KeyError(f"missing selected column {selected_col!r}")
        rules, specs = _learn_state_risk_rules(
            ledger,
            selected_col=str(selected_col),
            fit_months=fit_months,
            sides=side_set,
            min_fit_selected_rows=int(min_fit_selected_rows),
            round_trip_cost=float(round_trip_cost),
            max_features=int(max_features),
            mae_weight=float(mae_weight),
            adverse_weight=float(adverse_weight),
            underwater_weight=float(underwater_weight),
            ev_weight=float(ev_weight),
            mean_u_weight=float(mean_u_weight),
        )
        if not rules.empty:
            all_rules.append(rules.assign(selected_col=str(selected_col)))
        for combine_mode in combine_modes:
            risk = _score_state_risk(
                ledger,
                rules,
                specs,
                combine_mode=str(combine_mode),
                top_n=int(top_n),
            )
            for alpha in alphas:
                alpha = float(alpha)
                adjusted = ledger["score"].astype(float) - alpha * risk.astype(float)
                work = ledger.assign(__state_path_risk__=risk, __adjusted_score__=adjusted)
                adjusted_selected = _select_budgeted(
                    work,
                    selected_col=str(selected_col),
                    adjusted_score_col="__adjusted_score__",
                    sides=side_set,
                    group_cols=group_cols,
                )
                baseline_selected = ledger[selected_col].astype(bool)
                row: dict[str, Any] = {
                    "overlay_action": "rerank_budgeted",
                    "selected_col": str(selected_col),
                    "alpha": alpha,
                    "risk_threshold": float("nan"),
                    "combine_mode": str(combine_mode),
                    "rule_count": int(len(rules)),
                    "risk_nonzero_share": float(risk.gt(0.0).mean()) if len(risk) else float("nan"),
                    "risk_mean": float(risk.mean()) if len(risk) else float("nan"),
                    "risk_p95": float(risk.quantile(0.95)) if len(risk) else float("nan"),
                }
                for prefix, months in (
                    ("fit", fit_months),
                    ("holdout", [holdout_month]),
                    ("all", [*fit_months, holdout_month]),
                ):
                    for side in (None, "long", "short"):
                        side_prefix = "all_sides" if side is None else side
                        row.update(
                            _prefix(
                                f"{prefix}_{side_prefix}_baseline",
                                _slice_metrics(
                                    ledger,
                                    baseline_selected,
                                    months=months,
                                    side=side,
                                    round_trip_cost=float(round_trip_cost),
                                ),
                            )
                        )
                        row.update(
                            _prefix(
                                f"{prefix}_{side_prefix}_adjusted",
                                _slice_metrics(
                                    ledger,
                                    adjusted_selected,
                                    months=months,
                                    side=side,
                                    round_trip_cost=float(round_trip_cost),
                                ),
                            )
                        )
                holdout_long_evw = row.get("holdout_long_adjusted_ev_weighted_first_touch_precision", 0.0) or 0.0
                holdout_long_mae = row.get("holdout_long_adjusted_mae_before_mfe_1r_rate", 1.0) or 1.0
                holdout_all_evw = row.get("holdout_all_sides_adjusted_ev_weighted_first_touch_precision", 0.0) or 0.0
                holdout_all_u = row.get("holdout_all_sides_adjusted_mean_u", -0.02) or -0.02
                row["overlay_objective"] = float(
                    holdout_all_evw
                    + 0.40 * holdout_long_evw
                    + 2.0 * max(holdout_all_u, -0.02)
                    - 0.30 * max(0.0, holdout_long_mae - 0.25)
                )
                summaries.append(row)
            for threshold in filter_thresholds:
                threshold = float(threshold)
                baseline_selected = ledger[selected_col].astype(bool)
                adjusted_selected = _filter_selected_by_risk(
                    ledger,
                    selected_col=str(selected_col),
                    risk=risk,
                    threshold=threshold,
                    sides=side_set,
                )
                row = {
                    "overlay_action": "risk_filter",
                    "selected_col": str(selected_col),
                    "alpha": float("nan"),
                    "risk_threshold": threshold,
                    "combine_mode": str(combine_mode),
                    "rule_count": int(len(rules)),
                    "risk_nonzero_share": float(risk.gt(0.0).mean()) if len(risk) else float("nan"),
                    "risk_mean": float(risk.mean()) if len(risk) else float("nan"),
                    "risk_p95": float(risk.quantile(0.95)) if len(risk) else float("nan"),
                }
                for prefix, months in (
                    ("fit", fit_months),
                    ("holdout", [holdout_month]),
                    ("all", [*fit_months, holdout_month]),
                ):
                    for side in (None, "long", "short"):
                        side_prefix = "all_sides" if side is None else side
                        row.update(
                            _prefix(
                                f"{prefix}_{side_prefix}_baseline",
                                _slice_metrics(
                                    ledger,
                                    baseline_selected,
                                    months=months,
                                    side=side,
                                    round_trip_cost=float(round_trip_cost),
                                ),
                            )
                        )
                        row.update(
                            _prefix(
                                f"{prefix}_{side_prefix}_adjusted",
                                _slice_metrics(
                                    ledger,
                                    adjusted_selected,
                                    months=months,
                                    side=side,
                                    round_trip_cost=float(round_trip_cost),
                                ),
                            )
                        )
                holdout_long_evw = row.get("holdout_long_adjusted_ev_weighted_first_touch_precision", 0.0) or 0.0
                holdout_long_mae = row.get("holdout_long_adjusted_mae_before_mfe_1r_rate", 1.0) or 1.0
                holdout_all_evw = row.get("holdout_all_sides_adjusted_ev_weighted_first_touch_precision", 0.0) or 0.0
                holdout_all_u = row.get("holdout_all_sides_adjusted_mean_u", -0.02) or -0.02
                holdout_long_rows = row.get("holdout_long_adjusted_selected_rows", 0.0) or 0.0
                holdout_long_base_rows = row.get("holdout_long_baseline_selected_rows", 1.0) or 1.0
                retention = float(holdout_long_rows) / max(float(holdout_long_base_rows), 1.0)
                row["holdout_long_retention"] = retention
                row["overlay_objective"] = float(
                    holdout_all_evw
                    + 0.40 * holdout_long_evw
                    + 2.0 * max(holdout_all_u, -0.02)
                    - 0.30 * max(0.0, holdout_long_mae - 0.25)
                    - 0.08 * max(0.0, 0.60 - retention)
                )
                summaries.append(row)
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(
            [
                "overlay_objective",
                "holdout_all_sides_adjusted_ev_weighted_first_touch_precision",
                "holdout_long_adjusted_mae_before_mfe_1r_rate",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    rules_df = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()
    paths = {
        "summary": output_dir / "s52_state_risk_penalty_overlay_summary.csv",
        "rules": output_dir / "s52_state_risk_penalty_overlay_rules.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_state_risk_penalty_overlay.md",
    }
    summary.to_csv(paths["summary"], index=False)
    rules_df.to_csv(paths["rules"], index=False)
    manifest = {
        "scope": "s52_state_risk_penalty_overlay",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "variant": str(variant),
        "selected_cols": [str(v) for v in selected_cols],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "alphas": [float(v) for v in alphas],
        "sides": sorted(side_set),
        "combine_modes": [str(v) for v in combine_modes],
        "group_cols": [str(v) for v in group_cols],
        "round_trip_cost": float(round_trip_cost),
        "min_fit_selected_rows": int(min_fit_selected_rows),
        "max_features": int(max_features),
        "top_n": int(top_n),
        "filter_thresholds": [float(v) for v in filter_thresholds],
        "weights": {
            "mae": float(mae_weight),
            "adverse": float(adverse_weight),
            "underwater": float(underwater_weight),
            "ev": float(ev_weight),
            "mean_u": float(mean_u_weight),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report_cols = [
        "selected_col",
        "overlay_action",
        "alpha",
        "risk_threshold",
        "combine_mode",
        "overlay_objective",
        "rule_count",
        "risk_nonzero_share",
        "holdout_all_sides_baseline_ev_weighted_first_touch_precision",
        "holdout_all_sides_adjusted_ev_weighted_first_touch_precision",
        "holdout_all_sides_baseline_mean_u",
        "holdout_all_sides_adjusted_mean_u",
        "holdout_long_baseline_ev_weighted_first_touch_precision",
        "holdout_long_adjusted_ev_weighted_first_touch_precision",
        "holdout_long_baseline_mae_before_mfe_1r_rate",
        "holdout_long_adjusted_mae_before_mfe_1r_rate",
        "holdout_long_baseline_mean_underwater_bars_before_mfe",
        "holdout_long_adjusted_mean_underwater_bars_before_mfe",
    ]
    lines = [
        "# S52 State Risk Penalty Overlay",
        "",
        "Fit-only AE/GMM bucket risks applied as a continuous score penalty.",
        "",
        summary[[c for c in report_cols if c in summary.columns]].head(30).to_markdown(index=False)
        if not summary.empty
        else "No rows.",
        "",
        "## Outputs",
        "",
        f"- Summary: `{paths['summary']}`",
        f"- Rules: `{paths['rules']}`",
        f"- Manifest: `{paths['manifest']}`",
    ]
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "summary": str(paths["summary"]),
        "rules": str(paths["rules"]),
        "report": str(paths["markdown"]),
        "top": _json_safe(summary.head(10).to_dict(orient="records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="ranker_timestamp_cleangross")
    parser.add_argument("--selected-cols", default="selected_top10,selected_top20,selected_top30")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--alphas", default="0.0,0.25,0.50,0.75,1.0,1.5,2.0")
    parser.add_argument("--sides", default="long")
    parser.add_argument("--combine-modes", default="mean_top3,max")
    parser.add_argument("--group-cols", default="month,side_name")
    parser.add_argument("--round-trip-cost", type=float, default=0.0100)
    parser.add_argument("--min-fit-selected-rows", type=int, default=60)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--filter-thresholds", default="")
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--adverse-weight", type=float, default=0.75)
    parser.add_argument("--underwater-weight", type=float, default=0.50)
    parser.add_argument("--ev-weight", type=float, default=1.0)
    parser.add_argument("--mean-u-weight", type=float, default=8.0)
    args = parser.parse_args()
    result = run_overlay(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        variant=str(args.variant),
        selected_cols=_parse_csv(args.selected_cols, ()),
        fit_months=_parse_csv(args.fit_months, ()),
        holdout_month=str(args.holdout_month),
        alphas=_parse_float_csv(args.alphas, ()),
        sides=_parse_csv(args.sides, ()),
        combine_modes=_parse_csv(args.combine_modes, ()),
        group_cols=_parse_csv(args.group_cols, ("month", "side_name")),
        round_trip_cost=float(args.round_trip_cost),
        min_fit_selected_rows=int(args.min_fit_selected_rows),
        max_features=int(args.max_features),
        top_n=int(args.top_n),
        filter_thresholds=_parse_float_csv(args.filter_thresholds, ()),
        mae_weight=float(args.mae_weight),
        adverse_weight=float(args.adverse_weight),
        underwater_weight=float(args.underwater_weight),
        ev_weight=float(args.ev_weight),
        mean_u_weight=float(args.mean_u_weight),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
