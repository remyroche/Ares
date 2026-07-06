#!/usr/bin/env python3
"""Round-A label recoverability by source/regime.

This is a no-training diagnostic. It reconstructs the strict Round-A label
targets and the closest economic proxy selectors, then compares oracle-valid
rows, proxy-selected rows, missed oracle rows, and dirty false positives by
source tag and regime.
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
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _make_targets,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.run_soft_label_rounda_topk_proxy_diagnostics import (  # noqa: E402
    ROUND_TRIP_COST,
    _gated_selector_scores,
    _mfe_mae,
    _parse_csv,
    _parse_float_csv,
    _rounda_proxy_score,
    _safe_numeric,
    _strict_rounda_targets,
    _topk_positions_by_timestamp,
)


DEFAULT_SOURCE_TAGS_PATH = Path(
    "data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic/candidate_source_tags.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/soft_label_rounda_source_regime_recoverability_stage126_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_LABEL_ARMS = ("S124_s3_net_floor_veto", "S126_clean_net_direct_rank", "S127_fast_clean_net_rank")
DEFAULT_SELECTOR_SPECS = (
    "hard_gate70_blend_proxy_oos:S124_s3_net_floor_veto:5",
    "hard_gate70_blend_proxy_oos:S124_s3_net_floor_veto:3",
    "support_gate70_blend_proxy_oos:S126_clean_net_direct_rank:5",
    "support_gate70_blend_proxy_oos:S127_fast_clean_net_rank:5",
)
DEFAULT_GATE_MIN_SCORES = (0.70,)
DEFAULT_ORACLE_TOP_K = 1
DEFAULT_ORACLE_MIN_SCORE = 0.0
DEFAULT_PROXY_MIN_SCORE = -1.0


def _parse_int_csv(value: str | list[int] | tuple[int, ...], default: tuple[int, ...] = ()) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_selector_specs(values: str | list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    raw = _parse_csv(values, DEFAULT_SELECTOR_SPECS)
    specs: list[dict[str, Any]] = []
    for item in raw:
        parts = [part.strip() for part in str(item).split(":")]
        if len(parts) != 3:
            raise ValueError(f"Selector spec must be selector:label_arm:top_k, got {item!r}")
        selector, label_arm, top_k = parts
        specs.append({"selector": selector, "label_arm": label_arm, "top_k": int(top_k)})
    return specs


def _period_month(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["__ts__"], errors="coerce").dt.to_period("M").astype(str)


def _week_label(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("W-SUN").astype(str)


def _selected_positions(frame: pd.DataFrame, score: pd.Series, top_k: int, *, min_score: float) -> np.ndarray:
    selections = _topk_positions_by_timestamp(frame, score, int(top_k), min_score=float(min_score))
    if not selections:
        return np.array([], dtype=np.int64)
    return np.concatenate([idx for _, idx in selections]).astype(np.int64, copy=False)


def _load_source_tags(path: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = [
        "__ts__",
        "__symbol__",
        "G_VOL",
        "__regime_vol_12h__",
        "__regime_vol_48h__",
        "__regime_volume_12h__",
        "__regime_volume_48h__",
        "__regime_trend_12h__",
        "__regime_trend_48h__",
        "primary_source_tag",
        "source_tag_reason_codes",
        "quiet_continuation_score",
        "loud_breakout_impulse_score",
        "dirty_shock_avoid_score",
        "clean_execution_context_score",
        "calm_positive_source_score",
        "loud_clean_execution_score",
        "clean_run_entry_score",
        "compression_capture_candidate_score",
        "risk_adjusted_capture_candidate_score",
        "clean_economic_capture_candidate_score",
        "misleading_location_risk_score",
    ]
    all_tags = pd.read_parquet(path)
    read_cols = [col for col in columns if col in set(all_tags.columns)]
    tags = all_tags.loc[:, read_cols].copy()
    tags["__ts__"] = pd.to_datetime(tags["__ts__"], errors="coerce")
    tags["__symbol__"] = tags["__symbol__"].astype(str)
    tags = tags.drop_duplicates(["__ts__", "__symbol__"], keep="first")
    joined = frame[["__ts__", "__symbol__"]].merge(tags, on=["__ts__", "__symbol__"], how="left")
    report = {
        "source_tags_path": str(path),
        "source_tag_rows": int(len(tags)),
        "source_tag_read_columns": list(read_cols),
        "label_rows": int(len(frame)),
        "matched_rows": int(joined["primary_source_tag"].notna().sum())
        if "primary_source_tag" in joined.columns
        else 0,
    }
    report["match_rate"] = float(report["matched_rows"] / report["label_rows"]) if report["label_rows"] else 0.0
    return joined.drop(columns=["__ts__", "__symbol__"]), report


def _metric_dict(metrics: pd.DataFrame, target: pd.DataFrame, pos: np.ndarray) -> pd.DataFrame:
    selected = metrics.iloc[pos].reset_index(drop=True).copy() if len(pos) else metrics.iloc[:0].copy()
    selected_target = target.iloc[pos].reset_index(drop=True).copy() if len(pos) else target.iloc[:0].copy()
    out = selected.copy()
    out["target_soft"] = selected_target.get("target_soft", pd.Series(dtype=float))
    out["target_hard"] = selected_target.get("target_hard", pd.Series(dtype=float))
    mfe_mae = _mfe_mae(selected) if len(selected) else pd.Series(dtype=float)
    out["mfe_mae_ratio"] = mfe_mae.reset_index(drop=True)
    out["row_positive_net"] = _safe_numeric(out.get("u_policy_net")).gt(0.0)
    out["row_bad_mae_1r"] = _safe_numeric(out.get("mae_norm")).ge(1.0)
    out["row_wide25"] = _safe_numeric(out.get("barrier")).gt(0.025)
    out["row_timeout"] = _safe_numeric(out.get("is_timeout")).gt(0.0)
    out["row_target_clean"] = _safe_numeric(out.get("target_hard")).gt(0.0)
    out["row_dirty_path"] = out["row_bad_mae_1r"] | out["row_wide25"] | out["row_timeout"]
    out["row_relaxed_clean"] = (
        out["row_positive_net"]
        & _safe_numeric(out.get("mae_norm")).le(1.0)
        & _safe_numeric(out.get("barrier")).le(0.027)
        & _safe_numeric(out.get("is_timeout")).le(0.0)
        & _safe_numeric(out.get("mfe_mae_ratio")).ge(1.05)
    )
    return out


def _make_ledger_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    source_tags: pd.DataFrame,
    month: str,
    selector: str,
    label_arm: str,
    top_k: int,
    selection_side: str,
    positions: np.ndarray,
    score: pd.Series,
    oracle_positions: set[int],
    proxy_positions: set[int],
    proxy_features: str,
    proxy_diag: dict[str, Any],
) -> pd.DataFrame:
    if len(positions) == 0:
        return pd.DataFrame()
    base = frame.iloc[positions][["__ts__", "__symbol__"]].reset_index(drop=True).copy()
    base["valid_pos"] = positions.astype(np.int64, copy=False)
    base["month"] = str(month)
    base["week"] = _week_label(base["__ts__"])
    base["selector"] = str(selector)
    base["label_arm"] = str(label_arm)
    base["top_k"] = int(top_k)
    base["selection_side"] = str(selection_side)
    base["score"] = _safe_numeric(score).iloc[positions].to_numpy(dtype=np.float64, copy=False)
    metric_cols = _metric_dict(metrics, target, positions)
    tag_cols = source_tags.iloc[positions].reset_index(drop=True).copy()
    out = pd.concat([base, metric_cols, tag_cols], axis=1)
    out["side_name"] = np.where(_safe_numeric(out.get("side", pd.Series(1.0, index=out.index))) < 0.0, "short", "long")
    out["primary_source_tag"] = out.get("primary_source_tag", pd.Series("source_tag_missing", index=out.index)).fillna(
        "source_tag_missing",
    )
    out["in_oracle_top1"] = out["valid_pos"].isin(oracle_positions)
    out["in_proxy_selection"] = out["valid_pos"].isin(proxy_positions)
    out["proxy_features"] = str(proxy_features)
    out["proxy_candidate_count"] = int(proxy_diag.get("proxy_candidate_count", 0) or 0)
    out["proxy_mean_train_target_ic"] = float(proxy_diag.get("proxy_mean_train_target_ic", np.nan))
    out["proxy_mean_train_utility_ic"] = float(proxy_diag.get("proxy_mean_train_utility_ic", np.nan))
    out["proxy_mean_train_bad_mae_ic"] = float(proxy_diag.get("proxy_mean_train_bad_mae_ic", np.nan))
    out["proxy_mean_train_timeout_ic"] = float(proxy_diag.get("proxy_mean_train_timeout_ic", np.nan))
    if selection_side == "oracle_selected":
        out["classification"] = np.where(out["in_proxy_selection"], "oracle_captured_by_proxy", "missed_oracle")
    else:
        out["classification"] = np.select(
            [
                out["in_oracle_top1"],
                out["row_target_clean"],
                out["row_positive_net"] & out["row_dirty_path"],
                out["row_positive_net"],
            ],
            [
                "proxy_oracle_overlap",
                "proxy_clean_nonoracle",
                "proxy_positive_dirty",
                "proxy_positive_nonoracle",
            ],
            default="proxy_negative_false_positive",
        )
    return out


def _period_stats(group: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    ret_net = _safe_numeric(group.get("ret_net"))
    mae = _safe_numeric(group.get("mae_norm"))
    barrier = _safe_numeric(group.get("barrier"))
    timeout = _safe_numeric(group.get("is_timeout"))
    return {
        f"{prefix}_rows": int(len(group)),
        f"{prefix}_symbols": int(group["__symbol__"].nunique(dropna=True)) if "__symbol__" in group else 0,
        f"{prefix}_mean_return_net": _safe_mean(ret_net),
        f"{prefix}_net_pnl": float(ret_net.sum(skipna=True)) if len(ret_net) else 0.0,
        f"{prefix}_hit_u": _safe_mean(_safe_numeric(group.get("u_policy_net")).gt(0.0)),
        f"{prefix}_target_hard_rate": _safe_mean(group.get("row_target_clean")),
        f"{prefix}_relaxed_clean_rate": _safe_mean(group.get("row_relaxed_clean")),
        f"{prefix}_bad_mae_1r_rate": _safe_mean(mae.ge(1.0)),
        f"{prefix}_p90_mae_norm": _safe_quantile(mae, 0.90),
        f"{prefix}_wide25_rate": _safe_mean(barrier.gt(0.025)),
        f"{prefix}_timeout_rate": _safe_mean(timeout.gt(0.0)),
        f"{prefix}_oracle_overlap_rate": _safe_mean(group.get("in_oracle_top1")),
        f"{prefix}_top_source_share": (
            float(group["primary_source_tag"].astype(str).value_counts(normalize=True).iloc[0])
            if len(group) and "primary_source_tag" in group
            else 0.0
        ),
    }


def _spec_summary(ledger: pd.DataFrame, *, fit_months: list[str], holdout_month: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["selector", "label_arm", "top_k"]
    for key, group in ledger.groupby(keys, dropna=False, observed=True):
        selector, label_arm, top_k = key
        proxy = group[group["selection_side"].eq("proxy_selected")]
        oracle = group[group["selection_side"].eq("oracle_selected")]
        fit_proxy = proxy[proxy["month"].astype(str).isin(fit_months)]
        hold_proxy = proxy[proxy["month"].astype(str).eq(str(holdout_month))]
        fit_oracle = oracle[oracle["month"].astype(str).isin(fit_months)]
        hold_oracle = oracle[oracle["month"].astype(str).eq(str(holdout_month))]
        row: dict[str, Any] = {
            "selector": str(selector),
            "label_arm": str(label_arm),
            "top_k": int(top_k),
            "fit_oracle_rows": int(len(fit_oracle)),
            "holdout_oracle_rows": int(len(hold_oracle)),
            "fit_oracle_capture_rate": _safe_mean(fit_oracle["classification"].eq("oracle_captured_by_proxy")),
            "holdout_oracle_capture_rate": _safe_mean(hold_oracle["classification"].eq("oracle_captured_by_proxy")),
        }
        row.update(_period_stats(fit_proxy, prefix="fit_proxy"))
        row.update(_period_stats(hold_proxy, prefix="holdout_proxy"))
        row["fit_pass_relaxed"] = bool(
            row["fit_proxy_rows"] >= 30
            and row["fit_proxy_mean_return_net"] > 0.0
            and row["fit_proxy_bad_mae_1r_rate"] <= 0.40
            and row["fit_proxy_p90_mae_norm"] <= 4.0
            and row["fit_proxy_timeout_rate"] <= 0.50
        )
        row["holdout_pass_relaxed"] = bool(
            row["holdout_proxy_rows"] >= 10
            and row["holdout_proxy_mean_return_net"] > 0.0
            and row["holdout_proxy_bad_mae_1r_rate"] <= 0.40
            and row["holdout_proxy_p90_mae_norm"] <= 4.0
            and row["holdout_proxy_timeout_rate"] <= 0.50
        )
        row["proxy_trainworthy_relaxed"] = bool(row["fit_pass_relaxed"] and row["holdout_pass_relaxed"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["proxy_trainworthy_relaxed", "holdout_pass_relaxed", "fit_pass_relaxed", "holdout_proxy_mean_return_net"],
        ascending=[False, False, False, False],
    )


def _source_group_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    scopes: list[tuple[str, pd.Series]] = [
        ("primary_source_tag", ledger["primary_source_tag"].astype(str)),
        ("symbol", ledger["__symbol__"].astype(str)),
        ("week", ledger["week"].astype(str)),
        ("side", ledger["side_name"].astype(str)),
    ]
    for col in ("G_VOL", "__regime_vol_48h__", "__regime_trend_48h__", "__regime_volume_48h__"):
        if col in ledger.columns:
            scopes.append((col, col + "=" + ledger[col].astype(str)))
            scopes.append((f"source_x_{col}", ledger["primary_source_tag"].astype(str) + "|" + col + "=" + ledger[col].astype(str)))
    for scope, values in scopes:
        work = ledger.copy()
        work["group_scope"] = scope
        work["group_value"] = values.fillna("NA").astype(str)
        rows.append(work)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _monthly_group_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    grouped = _source_group_rows(ledger)
    if grouped.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [
        "month",
        "selector",
        "label_arm",
        "top_k",
        "selection_side",
        "classification",
        "group_scope",
        "group_value",
    ]
    for key, group in grouped.groupby(keys, dropna=False, observed=True):
        month, selector, label_arm, top_k, selection_side, classification, group_scope, group_value = key
        row = {
            "month": str(month),
            "selector": str(selector),
            "label_arm": str(label_arm),
            "top_k": int(top_k),
            "selection_side": str(selection_side),
            "classification": str(classification),
            "group_scope": str(group_scope),
            "group_value": str(group_value),
            "rows": int(len(group)),
            "symbols": int(group["__symbol__"].nunique(dropna=True)),
            "timestamps": int(pd.to_datetime(group["__ts__"], errors="coerce").nunique(dropna=True)),
            "mean_return_net": _safe_mean(group["ret_net"]),
            "net_pnl": float(_safe_numeric(group["ret_net"]).sum(skipna=True)),
            "hit_u": _safe_mean(group["row_positive_net"]),
            "target_hard_rate": _safe_mean(group["row_target_clean"]),
            "relaxed_clean_rate": _safe_mean(group["row_relaxed_clean"]),
            "bad_mae_1r_rate": _safe_mean(group["row_bad_mae_1r"]),
            "p90_mae_norm": _safe_quantile(group["mae_norm"], 0.90),
            "wide25_rate": _safe_mean(group["row_wide25"]),
            "timeout_rate": _safe_mean(group["row_timeout"]),
            "oracle_overlap_rate": _safe_mean(group["in_oracle_top1"]),
            "score_ic_u": _spearman(group["score"], group["u_policy_net"]),
            "score_ic_bad_mae": _spearman(group["score"], group["row_bad_mae_1r"].astype(float)),
            "top_symbols": ",".join(group["__symbol__"].astype(str).value_counts().head(5).index.tolist()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["month", "selector", "label_arm", "top_k", "group_scope", "rows"], ascending=[True, True, True, True, True, False])


def _fit_holdout_group_summary(monthly: pd.DataFrame, *, fit_months: list[str], holdout_month: str) -> pd.DataFrame:
    proxy = monthly[
        monthly["selection_side"].eq("proxy_selected")
        & monthly["classification"].isin(
            [
                "proxy_oracle_overlap",
                "proxy_clean_nonoracle",
                "proxy_positive_dirty",
                "proxy_positive_nonoracle",
                "proxy_negative_false_positive",
            ]
        )
        & monthly["group_scope"].isin(
            [
                "primary_source_tag",
                "G_VOL",
                "__regime_vol_48h__",
                "__regime_trend_48h__",
                "source_x_G_VOL",
                "source_x___regime_vol_48h__",
                "source_x___regime_trend_48h__",
            ]
        )
    ].copy()
    if proxy.empty:
        return pd.DataFrame()
    # Re-aggregate across classifications first so each group represents the full proxy selection.
    agg_rows: list[dict[str, Any]] = []
    agg_keys = ["month", "selector", "label_arm", "top_k", "group_scope", "group_value"]
    for key, group in proxy.groupby(agg_keys, dropna=False, observed=True):
        month, selector, label_arm, top_k, group_scope, group_value = key
        rows = int(group["rows"].sum())
        if rows <= 0:
            continue
        weights = _safe_numeric(group["rows"]).replace(0, np.nan)

        def wavg(col: str) -> float:
            values = _safe_numeric(group[col])
            mask = values.notna() & weights.notna()
            return float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if int(mask.sum()) else float("nan")

        agg_rows.append(
            {
                "month": str(month),
                "selector": str(selector),
                "label_arm": str(label_arm),
                "top_k": int(top_k),
                "group_scope": str(group_scope),
                "group_value": str(group_value),
                "rows": rows,
                "net_pnl": float(_safe_numeric(group["net_pnl"]).sum(skipna=True)),
                "mean_return_net": wavg("mean_return_net"),
                "hit_u": wavg("hit_u"),
                "target_hard_rate": wavg("target_hard_rate"),
                "relaxed_clean_rate": wavg("relaxed_clean_rate"),
                "bad_mae_1r_rate": wavg("bad_mae_1r_rate"),
                "p90_mae_norm": wavg("p90_mae_norm"),
                "wide25_rate": wavg("wide25_rate"),
                "timeout_rate": wavg("timeout_rate"),
                "oracle_overlap_rate": wavg("oracle_overlap_rate"),
            }
        )
    monthly_proxy = pd.DataFrame(agg_rows)
    rows: list[dict[str, Any]] = []
    keys = ["selector", "label_arm", "top_k", "group_scope", "group_value"]
    for key, group in monthly_proxy.groupby(keys, dropna=False, observed=True):
        selector, label_arm, top_k, group_scope, group_value = key
        fit = group[group["month"].astype(str).isin(fit_months)]
        hold = group[group["month"].astype(str).eq(str(holdout_month))]
        if fit.empty or hold.empty:
            continue

        def period(group_in: pd.DataFrame, prefix: str) -> dict[str, Any]:
            total_rows = int(group_in["rows"].sum())
            weights = _safe_numeric(group_in["rows"]).replace(0, np.nan)

            def wavg(col: str) -> float:
                values = _safe_numeric(group_in[col])
                mask = values.notna() & weights.notna()
                return float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if int(mask.sum()) else float("nan")

            return {
                f"{prefix}_months": int(group_in["month"].nunique(dropna=True)),
                f"{prefix}_rows": total_rows,
                f"{prefix}_net_pnl": float(_safe_numeric(group_in["net_pnl"]).sum(skipna=True)),
                f"{prefix}_mean_return_net": wavg("mean_return_net"),
                f"{prefix}_hit_u": wavg("hit_u"),
                f"{prefix}_target_hard_rate": wavg("target_hard_rate"),
                f"{prefix}_relaxed_clean_rate": wavg("relaxed_clean_rate"),
                f"{prefix}_bad_mae_1r_rate": wavg("bad_mae_1r_rate"),
                f"{prefix}_p90_mae_norm": wavg("p90_mae_norm"),
                f"{prefix}_wide25_rate": wavg("wide25_rate"),
                f"{prefix}_timeout_rate": wavg("timeout_rate"),
                f"{prefix}_oracle_overlap_rate": wavg("oracle_overlap_rate"),
            }

        row: dict[str, Any] = {
            "selector": str(selector),
            "label_arm": str(label_arm),
            "top_k": int(top_k),
            "group_scope": str(group_scope),
            "group_value": str(group_value),
        }
        row.update(period(fit, "fit"))
        row.update(period(hold, "holdout"))
        row["fit_group_pass_relaxed"] = bool(
            row["fit_rows"] >= 20
            and row["fit_mean_return_net"] > 0.0
            and row["fit_bad_mae_1r_rate"] <= 0.40
            and row["fit_p90_mae_norm"] <= 4.0
            and row["fit_timeout_rate"] <= 0.50
        )
        row["holdout_group_pass_relaxed"] = bool(
            row["holdout_rows"] >= 5
            and row["holdout_mean_return_net"] > 0.0
            and row["holdout_bad_mae_1r_rate"] <= 0.40
            and row["holdout_p90_mae_norm"] <= 4.0
            and row["holdout_timeout_rate"] <= 0.50
        )
        row["candidate_pocket_pass"] = bool(row["fit_group_pass_relaxed"] and row["holdout_group_pass_relaxed"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["candidate_pocket_pass", "holdout_group_pass_relaxed", "fit_group_pass_relaxed", "holdout_mean_return_net"],
        ascending=[False, False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    monthly_groups: pd.DataFrame,
    fit_holdout_groups: pd.DataFrame,
    proxy_features: pd.DataFrame,
) -> Path:
    path = output_dir / "rounda_source_regime_recoverability.md"
    summary_cols = [
        "proxy_trainworthy_relaxed",
        "fit_pass_relaxed",
        "holdout_pass_relaxed",
        "selector",
        "label_arm",
        "top_k",
        "fit_oracle_rows",
        "holdout_oracle_rows",
        "fit_oracle_capture_rate",
        "holdout_oracle_capture_rate",
        "fit_proxy_rows",
        "holdout_proxy_rows",
        "fit_proxy_mean_return_net",
        "holdout_proxy_mean_return_net",
        "fit_proxy_bad_mae_1r_rate",
        "holdout_proxy_bad_mae_1r_rate",
        "fit_proxy_p90_mae_norm",
        "holdout_proxy_p90_mae_norm",
        "fit_proxy_timeout_rate",
        "holdout_proxy_timeout_rate",
        "fit_proxy_oracle_overlap_rate",
        "holdout_proxy_oracle_overlap_rate",
    ]
    pocket_cols = [
        "candidate_pocket_pass",
        "fit_group_pass_relaxed",
        "holdout_group_pass_relaxed",
        "selector",
        "label_arm",
        "top_k",
        "group_scope",
        "group_value",
        "fit_rows",
        "holdout_rows",
        "fit_mean_return_net",
        "holdout_mean_return_net",
        "fit_bad_mae_1r_rate",
        "holdout_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "holdout_p90_mae_norm",
        "fit_timeout_rate",
        "holdout_timeout_rate",
        "fit_oracle_overlap_rate",
        "holdout_oracle_overlap_rate",
    ]
    month_cols = [
        "month",
        "selector",
        "label_arm",
        "top_k",
        "selection_side",
        "classification",
        "group_scope",
        "group_value",
        "rows",
        "mean_return_net",
        "hit_u",
        "target_hard_rate",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "oracle_overlap_rate",
    ]
    feature_cols = [
        "month",
        "label_arm",
        "selector",
        "proxy_role",
        "proxy_features",
        "proxy_candidate_count",
        "proxy_mean_train_target_ic",
        "proxy_mean_train_utility_ic",
        "proxy_mean_train_bad_mae_ic",
        "proxy_mean_train_timeout_ic",
    ]
    source_month = monthly_groups[
        monthly_groups["group_scope"].eq("primary_source_tag")
        & monthly_groups["selection_side"].isin(["proxy_selected", "oracle_selected"])
    ].copy()
    lines = [
        "# Round-A Source/Regime Recoverability",
        "",
        "Scope: no model training, no Optuna, no policy geometry optimisation. This reconstructs strict labels and economic proxy selectors only.",
        "",
        f"Months: `{', '.join(manifest['months'])}`. Fit months: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Source tags: `{manifest['source_tags_path']}`. Join match rate: `{manifest['source_tag_report']['match_rate']:.4f}`.",
        f"Features: `{manifest['feature_count']}`. Proxy top-k features: `{manifest['proxy_top_k']}`. Proxy objective: `{manifest['proxy_objective']}`.",
        "",
        "## Selector Summary",
        "",
        _table(summary, summary_cols, limit=40),
        "",
        "## Candidate Source/Regime Pockets",
        "",
        _table(fit_holdout_groups, pocket_cols, limit=80),
        "",
        "## Primary Source Month Detail",
        "",
        _table(source_month.sort_values(["month", "selector", "label_arm", "top_k", "selection_side", "rows"], ascending=[True, True, True, True, True, False]), month_cols, limit=160),
        "",
        "## Proxy Features",
        "",
        _table(proxy_features, feature_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Ledger: `{manifest['outputs']['ledger']}`",
        f"- Monthly groups: `{manifest['outputs']['monthly_groups']}`",
        f"- Fit/holdout groups: `{manifest['outputs']['fit_holdout_groups']}`",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Proxy features: `{manifest['outputs']['proxy_features']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    source_tags_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    label_arms: list[str],
    selector_specs: list[dict[str, Any]],
    oracle_top_k: int,
    oracle_min_score: float,
    proxy_min_score: float,
    gate_min_scores: list[float],
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
    frame, metrics, build_reports = _build_frame(
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
    source_tags, source_tag_report = _load_source_tags(source_tags_path, frame)
    features = _feature_columns(frame)
    base_targets = _make_targets(frame, metrics)
    strict_targets = _strict_rounda_targets(frame=frame, metrics=metrics, base_targets=base_targets)
    targets = {**base_targets, **strict_targets}
    missing = sorted(set(label_arms) - set(targets))
    if missing:
        raise ValueError(f"Unknown label arms: {missing}")
    spec_arms = sorted({str(spec["label_arm"]) for spec in selector_specs})
    missing_specs = sorted(set(spec_arms) - set(targets))
    if missing_specs:
        raise ValueError(f"Unknown selector spec label arms: {missing_specs}")

    period = _period_month(frame)
    ledger_parts: list[pd.DataFrame] = []
    proxy_feature_rows: list[dict[str, Any]] = []
    specs_by_arm: dict[str, list[dict[str, Any]]] = {}
    for spec in selector_specs:
        specs_by_arm.setdefault(str(spec["label_arm"]), []).append(spec)

    for month in months:
        train_mask = period.lt(str(month))
        valid_mask = period.eq(str(month))
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        train_metrics = metrics.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_tags = source_tags.loc[valid_mask].copy().reset_index(drop=True)

        for arm in sorted(set(label_arms) | set(specs_by_arm)):
            target = targets[arm]
            target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
            oracle_score = target_valid["target_soft"].reset_index(drop=True)
            oracle_pos = _selected_positions(valid, oracle_score, int(oracle_top_k), min_score=float(oracle_min_score))
            oracle_set = set(int(v) for v in oracle_pos.tolist())

            label_proxy, label_diag = _rounda_proxy_score(
                train=train,
                valid=frame.loc[valid_mask].copy(),
                features=features,
                y_train=target.loc[train_mask, "target_soft"],
                metrics_train=train_metrics,
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
            )
            hard_proxy, hard_diag = _rounda_proxy_score(
                train=train,
                valid=frame.loc[valid_mask].copy(),
                features=features,
                y_train=target.loc[train_mask, "target_hard"],
                metrics_train=train_metrics,
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
            )
            support_proxy, support_diag = _rounda_proxy_score(
                train=train,
                valid=frame.loc[valid_mask].copy(),
                features=features,
                y_train=target.loc[train_mask, "target_soft"].gt(0.0).astype(float),
                metrics_train=train_metrics,
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
            )
            gated_scores = _gated_selector_scores(
                label_proxy=label_proxy.reset_index(drop=True),
                hard_proxy=hard_proxy.reset_index(drop=True),
                support_proxy=support_proxy.reset_index(drop=True),
                label_features=",".join(label_diag.get("proxy_features", [])),
                hard_features=",".join(hard_diag.get("proxy_features", [])),
                support_features=",".join(support_diag.get("proxy_features", [])),
                gate_min_scores=gate_min_scores,
            )
            selector_scores = {str(name): score for name, score, _ in gated_scores}
            diag_by_selector: dict[str, dict[str, Any]] = {}
            feature_by_selector: dict[str, str] = {}
            for selector, _, feature_desc in gated_scores:
                diag_by_selector[str(selector)] = label_diag
                if str(selector).startswith("hard_gate"):
                    diag_by_selector[str(selector)] = hard_diag if str(selector).endswith("_gate_proxy_oos") else label_diag
                if str(selector).startswith("support_gate"):
                    diag_by_selector[str(selector)] = support_diag if str(selector).endswith("_gate_proxy_oos") else label_diag
                feature_by_selector[str(selector)] = str(feature_desc)
            for role, diag in (("label", label_diag), ("hard", hard_diag), ("support", support_diag)):
                proxy_feature_rows.append(
                    {
                        "month": str(month),
                        "label_arm": str(arm),
                        "selector": "",
                        "proxy_role": role,
                        "proxy_features": ",".join(diag.get("proxy_features", [])),
                        "proxy_candidate_count": int(diag.get("proxy_candidate_count", 0) or 0),
                        "proxy_mean_train_target_ic": float(diag.get("proxy_mean_train_target_ic", np.nan)),
                        "proxy_mean_train_utility_ic": float(diag.get("proxy_mean_train_utility_ic", np.nan)),
                        "proxy_mean_train_bad_mae_ic": float(diag.get("proxy_mean_train_bad_mae_ic", np.nan)),
                        "proxy_mean_train_timeout_ic": float(diag.get("proxy_mean_train_timeout_ic", np.nan)),
                    }
                )

            for spec in specs_by_arm.get(str(arm), []):
                selector = str(spec["selector"])
                top_k = int(spec["top_k"])
                if selector not in selector_scores:
                    raise ValueError(f"Selector {selector!r} was not constructed. Gate thresholds: {gate_min_scores}")
                proxy_score = _safe_numeric(selector_scores[selector]).reset_index(drop=True)
                proxy_pos = _selected_positions(valid, proxy_score, top_k, min_score=float(proxy_min_score))
                proxy_set = set(int(v) for v in proxy_pos.tolist())
                diag = diag_by_selector.get(selector, label_diag)
                feature_desc = feature_by_selector.get(selector, "")
                proxy_feature_rows.append(
                    {
                        "month": str(month),
                        "label_arm": str(arm),
                        "selector": selector,
                        "proxy_role": "selected",
                        "proxy_features": feature_desc,
                        "proxy_candidate_count": int(diag.get("proxy_candidate_count", 0) or 0),
                        "proxy_mean_train_target_ic": float(diag.get("proxy_mean_train_target_ic", np.nan)),
                        "proxy_mean_train_utility_ic": float(diag.get("proxy_mean_train_utility_ic", np.nan)),
                        "proxy_mean_train_bad_mae_ic": float(diag.get("proxy_mean_train_bad_mae_ic", np.nan)),
                        "proxy_mean_train_timeout_ic": float(diag.get("proxy_mean_train_timeout_ic", np.nan)),
                    }
                )
                ledger_parts.append(
                    _make_ledger_rows(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target_valid,
                        source_tags=valid_tags,
                        month=str(month),
                        selector=selector,
                        label_arm=str(arm),
                        top_k=top_k,
                        selection_side="proxy_selected",
                        positions=proxy_pos,
                        score=proxy_score,
                        oracle_positions=oracle_set,
                        proxy_positions=proxy_set,
                        proxy_features=feature_desc,
                        proxy_diag=diag,
                    )
                )
                ledger_parts.append(
                    _make_ledger_rows(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target_valid,
                        source_tags=valid_tags,
                        month=str(month),
                        selector=selector,
                        label_arm=str(arm),
                        top_k=top_k,
                        selection_side="oracle_selected",
                        positions=oracle_pos,
                        score=oracle_score,
                        oracle_positions=oracle_set,
                        proxy_positions=proxy_set,
                        proxy_features="oracle_target_soft",
                        proxy_diag=diag,
                    )
                )
        print(json.dumps({"month": str(month), "progress": "complete"}))

    ledger = pd.concat([part for part in ledger_parts if not part.empty], ignore_index=True) if ledger_parts else pd.DataFrame()
    summary = _spec_summary(ledger, fit_months=fit_months, holdout_month=holdout_month)
    monthly_groups = _monthly_group_summary(ledger)
    fit_holdout_groups = _fit_holdout_group_summary(
        monthly_groups,
        fit_months=fit_months,
        holdout_month=holdout_month,
    )
    proxy_features = pd.DataFrame(proxy_feature_rows).drop_duplicates().reset_index(drop=True)

    paths = {
        "ledger": output_dir / "rounda_source_regime_selected_ledger.csv",
        "monthly_groups": output_dir / "rounda_source_regime_monthly_groups.csv",
        "fit_holdout_groups": output_dir / "rounda_source_regime_fit_holdout_groups.csv",
        "summary": output_dir / "rounda_source_regime_summary.csv",
        "proxy_features": output_dir / "rounda_source_regime_proxy_features.csv",
        "manifest": output_dir / "manifest.json",
    }
    ledger.to_csv(paths["ledger"], index=False)
    monthly_groups.to_csv(paths["monthly_groups"], index=False)
    fit_holdout_groups.to_csv(paths["fit_holdout_groups"], index=False)
    summary.to_csv(paths["summary"], index=False)
    proxy_features.to_csv(paths["proxy_features"], index=False)

    manifest = {
        "scope": "rounda_source_regime_recoverability",
        "labels_path": str(labels_path),
        "source_tags_path": str(source_tags_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "months": list(months),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "label_arms": list(label_arms),
        "selector_specs": list(selector_specs),
        "oracle_top_k": int(oracle_top_k),
        "oracle_min_score": float(oracle_min_score),
        "proxy_min_score": float(proxy_min_score),
        "gate_min_scores": [float(v) for v in gate_min_scores],
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
        "round_trip_cost": float(ROUND_TRIP_COST),
        "source_tag_report": source_tag_report,
        "build_reports": build_reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "rounda_source_regime_recoverability.md")}},
        summary=summary,
        monthly_groups=monthly_groups,
        fit_holdout_groups=fit_holdout_groups,
        proxy_features=proxy_features,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--source-tags-path", type=Path, default=DEFAULT_SOURCE_TAGS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=498)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--label-arms", default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--selector-specs", default=",".join(DEFAULT_SELECTOR_SPECS))
    parser.add_argument("--oracle-top-k", type=int, default=DEFAULT_ORACLE_TOP_K)
    parser.add_argument("--oracle-min-score", type=float, default=DEFAULT_ORACLE_MIN_SCORE)
    parser.add_argument("--proxy-min-score", type=float, default=DEFAULT_PROXY_MIN_SCORE)
    parser.add_argument("--gate-min-scores", default=",".join(str(v) for v in DEFAULT_GATE_MIN_SCORES))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--proxy-objective", choices=("target_ic", "economic_ic", "economic_score"), default="economic_ic")
    parser.add_argument("--proxy-min-target-ic", type=float, default=0.0)
    parser.add_argument("--proxy-min-utility-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-bad-mae-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-wide-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-timeout-ic", type=float, default=0.0)
    parser.add_argument("--proxy-utility-weight", type=float, default=1.0)
    parser.add_argument("--proxy-bad-mae-weight", type=float, default=1.0)
    parser.add_argument("--proxy-wide-weight", type=float, default=0.5)
    parser.add_argument("--proxy-timeout-weight", type=float, default=0.5)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prior-windows-days", default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument("--state-path-prior-features", default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES))
    parser.add_argument("--event-feature-store-features", default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        source_tags_path=args.source_tags_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        fit_months=_parse_csv(args.fit_months, DEFAULT_FIT_MONTHS),
        holdout_month=str(args.holdout_month),
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        selector_specs=_parse_selector_specs(args.selector_specs),
        oracle_top_k=int(args.oracle_top_k),
        oracle_min_score=float(args.oracle_min_score),
        proxy_min_score=float(args.proxy_min_score),
        gate_min_scores=_parse_float_csv(args.gate_min_scores),
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
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=_parse_float_csv(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv(args.state_path_prior_features, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=_parse_csv(args.event_feature_store_features, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
