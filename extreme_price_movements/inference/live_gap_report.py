"""Machine-readable and markdown reports for live-vs-OOS gap diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.live_replay import summarize_gap_decomposition

REQUIRED_INTERPRETATIONS = [
    "If signal_forward_return is good but fill_forward_return is bad: execution/timing gap.",
    "If signal_forward_return is already bad: model/rank/live-feature drift.",
    "If signal_forward_return is good for rejected candidates but not traded candidates: selection/gating/portfolio constraints issue.",
    "If fill_forward_return is good but realized_trade_return is bad: exit/stop/slippage/cost issue.",
]


def _bool_series(values: pd.Series) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(False)
    return values.map(
        lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "traded", "accepted", "filled"}
        if pd.notna(x)
        else False
    )


def _numeric_report_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def classify_gap_rows(
    replay: pd.DataFrame,
    *,
    good_signal_bps: float = 0.0,
    good_fill_bps: float = 0.0,
    good_realized_bps: float = 0.0,
) -> pd.DataFrame:
    """Classify each replay row into a primary OOS/live gap bucket."""
    if replay is None or replay.empty:
        out = pd.DataFrame() if replay is None else replay.copy()
        out["gap_classification"] = []
        out["diagnostic_detail"] = []
        return out
    out = replay.copy()
    signal = _numeric_report_series(out, "signal_forward_net_bps")
    fill = _numeric_report_series(out, "fill_forward_net_bps")
    realized = _numeric_report_series(out, "realized_trade_net_bps")
    traded = _bool_series(out.get("was_traded", pd.Series(True, index=out.index)))

    classes = []
    details = []
    for idx in out.index:
        signal_good = bool(pd.notna(signal.loc[idx]) and signal.loc[idx] > good_signal_bps)
        fill_good = bool(pd.notna(fill.loc[idx]) and fill.loc[idx] > good_fill_bps)
        realized_good = bool(pd.notna(realized.loc[idx]) and realized.loc[idx] > good_realized_bps)
        was_traded = bool(traded.loc[idx])
        if pd.isna(signal.loc[idx]):
            cls = "missing_forward_outcome"
            detail = "Signal-forward outcome is missing; cannot classify prediction gap."
        elif signal_good and not was_traded:
            cls = "selection_or_gating_gap"
            detail = "Signal-forward return was good, but the candidate was rejected or not traded."
        elif signal_good and was_traded and not fill_good:
            cls = "execution_timing_gap"
            detail = "Signal-forward return was good, but fill-forward return was bad after delay/fill."
        elif not signal_good:
            cls = "prediction_or_live_feature_drift"
            detail = "Signal-forward return was already bad from the signal timestamp."
        elif fill_good and was_traded and pd.isna(realized.loc[idx]):
            cls = "unresolved_trade"
            detail = "Fill-forward return was good, but realized trade outcome is not resolved yet."
        elif fill_good and was_traded and not realized_good:
            cls = "exit_stop_slippage_cost_gap"
            detail = "Fill-forward return was good, but realized trade return was bad after exit/costs."
        else:
            cls = "no_major_gap"
            detail = "Signal, fill, and realized outcomes do not indicate a major gap."
        classes.append(cls)
        details.append(detail)
    out["gap_classification"] = classes
    out["diagnostic_detail"] = details
    return out


def _group_records(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    if df is None or df.empty or by not in df.columns:
        return []
    metric_cols = [
        c
        for c in [
            "oos_expected_net_bps",
            "signal_forward_net_bps",
            "fill_forward_net_bps",
            "realized_trade_net_bps",
            "gap_oos_vs_realized_bps",
        ]
        if c in df.columns
    ]
    grouped = df.groupby(by, dropna=False)
    rows = []
    for key, grp in grouped:
        row: dict[str, Any] = {by: key, "rows": int(len(grp))}
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"mean_{col}"] = float(vals.mean()) if vals.notna().any() else np.nan
        rows.append(row)
    return rows


def _non_null_count(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").notna().sum())


def _truthy_count(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    vals = df[col]
    if vals.dtype == bool:
        return int(vals.fillna(False).sum())
    return int(vals.map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"} if pd.notna(x) else False).sum())


def _unique_non_null(df: pd.DataFrame, col: str) -> list[Any]:
    if df is None or df.empty or col not in df.columns:
        return []
    vals = [v for v in df[col].dropna().unique().tolist() if str(v) != ""]
    return vals


def _spearman_ic(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    ok = xv.notna() & yv.notna()
    if int(ok.sum()) < 3:
        return np.nan
    xr = xv[ok].rank(method="average")
    yr = yv[ok].rank(method="average")
    if float(xr.std(ddof=0)) == 0.0 or float(yr.std(ddof=0)) == 0.0:
        return np.nan
    return float(xr.corr(yr))


def _prediction_score_series(df: pd.DataFrame) -> pd.Series:
    for col in (
        "rank_score",
        "normalized_rank_score",
        "adjusted_rank_score",
        "calibrated_score",
        "oos_expected_net_bps",
    ):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                return vals
    return pd.Series(np.nan, index=df.index, dtype=float)


def _ic_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    x = _prediction_score_series(df)
    y = pd.to_numeric(df.get("signal_forward_net_bps", pd.Series(np.nan, index=df.index)), errors="coerce")
    out: dict[str, Any] = {
        "overall_ic": _spearman_ic(x, y),
    }
    work = df.copy()
    work["_pred_score"] = x
    work["_outcome"] = y
    ts = pd.to_datetime(
        work.get("signal_bar_ts", work.get("timestamp", pd.Series(pd.NaT, index=work.index))),
        utc=True,
        errors="coerce",
    )
    ts_naive = ts.dt.tz_localize(None)
    work["_week"] = ts_naive.dt.to_period("W").astype(str)
    work["_month"] = ts_naive.dt.to_period("M").astype(str)
    groups = {
        "symbols": "symbol",
        "weeks": "_week",
        "months": "_month",
    }
    for label, col in groups.items():
        ics: list[float] = []
        if col in work.columns:
            for _, grp in work.dropna(subset=[col]).groupby(col, dropna=False):
                ic = _spearman_ic(grp["_pred_score"], grp["_outcome"])
                if np.isfinite(ic):
                    ics.append(float(ic))
        arr = np.asarray(ics, dtype=float)
        out[f"ic_mean_across_{label}"] = float(np.nanmean(arr)) if arr.size else np.nan
        out[f"ic_std_across_{label}"] = float(np.nanstd(arr, ddof=0)) if arr.size else np.nan
        out[f"ic_n_{label}"] = int(arr.size)
    return out


def _recommended_action(class_counts: dict[str, int], parity_summary: dict[str, Any]) -> str:
    if parity_summary.get("lookahead", 0) or parity_summary.get("mismatches", 0):
        return "Start with feature parity/timestamp leakage: mismatches or lookahead violations were detected."
    if not class_counts:
        return "Collect more replay rows with forward outcomes before drawing conclusions."
    top = max(class_counts.items(), key=lambda kv: kv[1])[0]
    mapping = {
        "execution_timing_gap": "Prioritize execution timing, entry routing, and fill-price controls.",
        "prediction_or_live_feature_drift": "Prioritize model/rank drift and live feature parity investigation.",
        "selection_or_gating_gap": "Prioritize portfolio gating, liquidity filters, and candidate selection constraints.",
        "exit_stop_slippage_cost_gap": "Prioritize exit/stop execution, slippage, and realized cost accounting.",
    }
    return mapping.get(top, "No single dominant gap bucket; inspect strategy/symbol breakdowns.")


def build_live_gap_report(
    replay: pd.DataFrame,
    *,
    parity_report: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Build a machine-readable live-vs-OOS gap report."""
    classified = classify_gap_rows(replay)
    n = int(len(classified)) if classified is not None else 0
    traded = _bool_series(classified.get("was_traded", pd.Series(dtype=bool))) if n else pd.Series(dtype=bool)
    class_counts = classified.get("gap_classification", pd.Series(dtype=object)).value_counts(dropna=False).to_dict() if n else {}

    parity_summary: dict[str, Any] = {}
    if parity_report is not None and not parity_report.empty:
        statuses = parity_report.get("parity_status", pd.Series(dtype=object)).astype(str)
        parity_summary = {
            "rows": int(len(parity_report)),
            "matches": int(statuses.isin({"match", "match_asof"}).sum()),
            "mismatches": int((statuses == "mismatch").sum()),
            "missing": int(statuses.str.startswith("missing").sum()),
            "lookahead": int(parity_report.get("lookahead_violation", pd.Series(False, index=parity_report.index)).astype(bool).sum()),
        }
    else:
        parity_summary = {"rows": 0, "matches": 0, "mismatches": 0, "missing": 0, "lookahead": 0}

    gap_decomp_df = summarize_gap_decomposition(classified)
    gap_decomp = {
        str(row["component"]): {
            "mean_bps": row["mean_bps"],
            "median_bps": row["median_bps"],
            "sum_bps": row["sum_bps"],
            "non_null": row["non_null"],
        }
        for row in gap_decomp_df.to_dict("records")
    }

    sig = pd.to_numeric(classified.get("signal_forward_net_bps"), errors="coerce") if n else pd.Series(dtype=float)
    selection_summary = {
        "mean_traded_signal_forward_bps": float(sig[traded].mean()) if n and sig[traded].notna().any() else np.nan,
        "mean_rejected_signal_forward_bps": float(sig[~traded].mean()) if n and sig[~traded].notna().any() else np.nan,
        "rejected_positive_signal_count": int(((~traded) & (sig > 0)).sum()) if n else 0,
        "rejected_positive_signal_sum_bps": float(sig[(~traded) & (sig > 0)].sum()) if n else 0.0,
    }
    if np.isfinite(selection_summary["mean_rejected_signal_forward_bps"]) and np.isfinite(
        selection_summary["mean_traded_signal_forward_bps"]
    ):
        selection_summary["selection_opportunity_cost_bps"] = (
            selection_summary["mean_rejected_signal_forward_bps"]
            - selection_summary["mean_traded_signal_forward_bps"]
        )
    else:
        selection_summary["selection_opportunity_cost_bps"] = np.nan

    diagnostic_coverage = {
        "replay_rows": n,
        "rows_with_oos_join": _non_null_count(classified, "oos_expected_net_bps"),
        "rows_with_signal_forward": _non_null_count(classified, "signal_forward_net_bps"),
        "rows_with_fill_forward": _non_null_count(classified, "fill_forward_net_bps"),
        "rows_with_realized_exits": _non_null_count(classified, "realized_exit_price"),
        "rows_with_realized_trade_net": _non_null_count(classified, "realized_trade_net_bps"),
        "diagnostic_complete_rows": _truthy_count(classified, "diagnostic_complete"),
        "missing_forward_outcome_rows": int(class_counts.get("missing_forward_outcome", 0)),
        "unresolved_trade_rows": int(class_counts.get("unresolved_trade", 0)),
        "ledger_decision_ts_non_null": int(pd.to_datetime(classified.get("decision_ts", pd.Series(dtype=object)), utc=True, errors="coerce").notna().sum()) if n else 0,
        "ledger_signal_bar_ts_non_null": int(pd.to_datetime(classified.get("signal_bar_ts", pd.Series(dtype=object)), utc=True, errors="coerce").notna().sum()) if n else 0,
        "ledger_feature_source_max_ts_non_null": int(pd.to_datetime(classified.get("feature_source_max_ts", pd.Series(dtype=object)), utc=True, errors="coerce").notna().sum()) if n else 0,
        "ledger_feature_available_ts_non_null": int(pd.to_datetime(classified.get("feature_available_ts", pd.Series(dtype=object)), utc=True, errors="coerce").notna().sum()) if n else 0,
    }
    metadata = {
        "primary_horizon_bars": _unique_non_null(classified, "primary_horizon_bars"),
        "bar_minutes": _unique_non_null(classified, "bar_minutes"),
    }
    unit_warnings = {}
    if n and "unit_warning" in classified.columns:
        warned = classified["unit_warning"].dropna().astype(str)
        warned = warned[warned != ""]
        unit_warnings = warned.value_counts().to_dict()
    ic_metrics = _ic_metrics(classified)

    summary = {
        "rows": n,
        "traded_rows": int(traded.sum()) if n else 0,
        "rejected_rows": int((~traded).sum()) if n else 0,
        "mean_oos_expected_net_bps": float(pd.to_numeric(classified.get("oos_expected_net_bps"), errors="coerce").mean()) if n else np.nan,
        "mean_signal_forward_net_bps": float(pd.to_numeric(classified.get("signal_forward_net_bps"), errors="coerce").mean()) if n else np.nan,
        "mean_realized_trade_net_bps": float(pd.to_numeric(classified.get("realized_trade_net_bps"), errors="coerce").mean()) if n else np.nan,
    }

    return {
        "summary": summary,
        "metadata": metadata,
        "diagnostic_coverage": diagnostic_coverage,
        "unit_warnings": unit_warnings,
        "feature_parity": parity_summary,
        "gap_decomposition": gap_decomp,
        "selection_summary": selection_summary,
        "ic_metrics": ic_metrics,
        "classification_counts": {str(k): int(v) for k, v in class_counts.items()},
        "by_strategy": _group_records(classified, "strategy_id"),
        "by_symbol": _group_records(classified, "symbol"),
        "by_exit_reason": _group_records(classified, "exit_reason"),
        "by_reject_reason": _group_records(classified, "portfolio_reject_reason"),
        "recommended_next_action": _recommended_action({str(k): int(v) for k, v in class_counts.items()}, parity_summary),
    }


def render_live_gap_report_markdown(report: Dict[str, Any]) -> str:
    """Render a markdown summary for a live gap report dict."""
    report = report or {}
    lines = ["# Live vs OOS Gap Report", "", "## Required interpretations"]
    for i, text in enumerate(REQUIRED_INTERPRETATIONS, 1):
        lines.append(f"{i}. {text}")
    lines.extend(["", "## Summary"])
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Metadata"])
    for key, value in (report.get("metadata") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Diagnostic coverage"])
    for key, value in (report.get("diagnostic_coverage") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Unit warnings"])
    for key, value in (report.get("unit_warnings") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Feature parity"])
    for key, value in (report.get("feature_parity") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Selection summary"])
    for key, value in (report.get("selection_summary") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## IC metrics"])
    for key, value in (report.get("ic_metrics") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Classification counts"])
    for key, value in (report.get("classification_counts") or {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Recommended next action", str(report.get("recommended_next_action", ""))])
    return "\n".join(lines) + "\n"
