#!/usr/bin/env python3
"""Clean-vs-dirty source diagnostics for broad GMM/base candidate ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LEDGER_PATH = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "gmm_train_meta_path_filter_smoke_s20_spread170_top05_exec_clean_v1/"
    "base_candidate_streams/label_feature_store_model_smoke_candidate_ledger.csv"
)
DEFAULT_SPREAD_PATH = Path(
    "data_perp/exchanges/krakenfutures/spread_model/per_asset_spread_baseline_latest.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "broad_source_clean_dirty_diagnostics_s20_spread170_v1"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _num(values: Any, default: float = np.nan) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if np.isfinite(default):
        series = series.fillna(float(default))
    return series


def _safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.mean()) if bool(series.notna().any()) else float("nan")


def _rate(values: Any) -> float:
    series = pd.Series(values)
    return float(series.astype(bool).mean()) if len(series) else float("nan")


def _bucket_quantiles(values: pd.Series, *, labels: tuple[str, ...]) -> pd.Series:
    out = pd.Series("missing", index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna()
    if int(finite.sum()) < len(labels):
        out.loc[finite] = "all_finite"
        return out
    try:
        out.loc[finite] = pd.qcut(
            numeric.loc[finite].rank(method="first"),
            q=len(labels),
            labels=labels,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        out.loc[finite] = "all_finite"
    return out


def _posterior_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    cols = []
    for col in frame.columns:
        name = str(col)
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            cols.append(name)
    return sorted(cols, key=lambda name: int(str(name)[len(prefix) :]))


def _posterior_argmax_bucket(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    missing_label: str,
) -> pd.Series:
    out = pd.Series(missing_label, index=frame.index, dtype=object)
    if not cols:
        return out
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    finite = values.notna().any(axis=1)
    if not bool(finite.any()):
        return out
    argmax = values.loc[finite].to_numpy(dtype=np.float64)
    out.loc[finite] = [f"cluster_{int(i)}" for i in np.nanargmax(argmax, axis=1)]
    return out


def _add_ae_gmm_regime_buckets(data: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Add global and side-specific AE/GMM buckets from posterior columns.

    Prefer posterior argmax buckets over low-cardinality helper flags. Those
    helper flags are useful as coverage checks but not as regime definitions.
    """
    global_cols = _posterior_columns(data, "ctx_gmm_cluster_posterior_")
    long_cols = _posterior_columns(data, "ctx_long_gmm_cluster_posterior_")
    short_cols = _posterior_columns(data, "ctx_short_gmm_cluster_posterior_")
    if global_cols:
        data["global_gmm_ae_regime_bucket"] = _posterior_argmax_bucket(
            data,
            global_cols,
            missing_label="global_missing",
        )
        global_source = "ctx_gmm_cluster_posterior_*"
    else:
        data["global_gmm_ae_regime_bucket"] = "global_missing"
        global_source = None
    long_bucket = _posterior_argmax_bucket(data, long_cols, missing_label="long_missing")
    short_bucket = _posterior_argmax_bucket(data, short_cols, missing_label="short_missing")
    side = _num(data.get("side"), default=1.0)
    data["side_gmm_ae_regime_bucket"] = np.where(side < 0.0, short_bucket, long_bucket)
    if long_cols or short_cols:
        side_source = "ctx_long/short_gmm_cluster_posterior_*"
    else:
        side_source = None
    if side_source is not None:
        data["gmm_ae_regime_bucket"] = data["side_gmm_ae_regime_bucket"].astype(str)
        source = side_source
    elif global_source is not None:
        data["gmm_ae_regime_bucket"] = data["global_gmm_ae_regime_bucket"].astype(str)
        source = global_source
    else:
        data["gmm_ae_regime_bucket"] = "missing"
        source = None
    return data, source


def _prepare_frame(
    ledger_path: Path,
    *,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    score_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)
    data = pd.read_csv(ledger_path)
    if data.empty:
        raise ValueError(f"candidate ledger is empty: {ledger_path}")
    required = {"timestamp", "symbol", "selector_variant", score_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"candidate ledger missing required columns: {missing}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["month"] = data["timestamp"].dt.to_period("M").astype(str)
    data["week"] = data["timestamp"].dt.to_period("W-SUN").astype(str)
    side = _num(data.get("side"), default=1.0)
    data["side_bucket"] = np.where(side < 0.0, "short", "long")
    data["score_for_gap"] = _num(data[score_column])
    data["u_policy_net"] = _num(data.get("u_policy_net"), default=0.0)
    data["bad_mae_1r"] = pd.Series(data.get("bad_mae_1r", False)).astype(bool)
    data["timeout_flag"] = _num(data.get("is_timeout"), default=0.0).gt(0.5)
    if "simple_policy_exit_reason" in data.columns:
        data["full_sl"] = data["simple_policy_exit_reason"].astype(str).eq("full_sl")
        full_sl_source = "simple_policy_exit_reason"
    elif "full_sl" in data.columns:
        data["full_sl"] = pd.Series(data["full_sl"]).astype(bool)
        full_sl_source = "full_sl"
    elif "full_stop_loss" in data.columns:
        data["full_sl"] = _num(data["full_stop_loss"], default=0.0).gt(0.5)
        full_sl_source = "full_stop_loss"
    else:
        data["full_sl"] = data["bad_mae_1r"]
        full_sl_source = "bad_mae_1r_proxy"
    data["clean_positive"] = (
        data["u_policy_net"].gt(0.0)
        & (~data["bad_mae_1r"])
        & (~data["timeout_flag"])
        & (~data["full_sl"])
    )
    data["dirty_positive"] = (
        data["u_policy_net"].gt(0.0)
        & (data["bad_mae_1r"] | data["timeout_flag"] | data["full_sl"])
    )
    data["oracle_top"] = pd.Series(data.get("oracle_top", False)).astype(bool)
    data["clean_oracle_top"] = pd.Series(data.get("clean_oracle_top", False)).astype(bool)

    spread_report: dict[str, Any] = {
        "enabled": False,
        "spread_baseline_path": str(spread_baseline_path) if spread_baseline_path else None,
        "spread_rank_column": spread_rank_column,
    }
    if spread_baseline_path is not None and spread_baseline_path.exists():
        spread = pd.read_csv(spread_baseline_path)
        if "symbol" in spread.columns and spread_rank_column in spread.columns:
            spread = spread[["symbol", spread_rank_column]].copy()
            spread["symbol"] = spread["symbol"].astype(str)
            spread[spread_rank_column] = _num(spread[spread_rank_column])
            data = data.merge(spread, on="symbol", how="left")
            data["spread_bucket"] = _bucket_quantiles(
                data[spread_rank_column],
                labels=("spread_q1_low", "spread_q2", "spread_q3", "spread_q4", "spread_q5_high"),
            )
            spread_report.update(
                {
                    "enabled": True,
                    "matched_rows": int(data[spread_rank_column].notna().sum()),
                    "missing_rows": int(data[spread_rank_column].isna().sum()),
                    "matched_symbols": int(
                        data.loc[data[spread_rank_column].notna(), "symbol"].nunique()
                    ),
                    "missing_symbols": int(
                        data.loc[data[spread_rank_column].isna(), "symbol"].nunique()
                    ),
                }
            )
        else:
            data["spread_bucket"] = "missing"
            spread_report["error"] = "missing_symbol_or_rank_column"
    else:
        data["spread_bucket"] = "missing"

    symbol_counts = data.groupby("symbol", dropna=False).size().rename("symbol_trade_count")
    symbol_clean_rate = data.groupby("symbol", dropna=False)["clean_positive"].mean().rename(
        "symbol_clean_positive_rate"
    )
    data = data.merge(symbol_counts, on="symbol", how="left")
    data = data.merge(symbol_clean_rate, on="symbol", how="left")
    data["symbol_liquidity_bucket"] = _bucket_quantiles(
        data["symbol_trade_count"],
        labels=("trade_q1_low", "trade_q2", "trade_q3", "trade_q4", "trade_q5_high"),
    )
    data["symbol_clean_rate_bucket"] = _bucket_quantiles(
        data["symbol_clean_positive_rate"],
        labels=("clean_q1_low", "clean_q2", "clean_q3", "clean_q4", "clean_q5_high"),
    )
    data, gmm_bucket_source = _add_ae_gmm_regime_buckets(data)
    if gmm_bucket_source is None:
        context_columns = [
            c
            for c in data.columns
            if str(c).startswith("ctx_")
            or any(
                keyword in str(c).lower()
                for keyword in (
                    "gmm",
                    "cluster",
                    "archetype",
                    "posterior",
                    "reconstruction",
                    "latent",
                    "state_spectral",
                    "bars_in_high_vol_state",
                )
            )
        ]
        numeric_context = [
            c
            for c in context_columns
            if bool(pd.to_numeric(data[c], errors="coerce").notna().any())
        ]
        if numeric_context:
            gmm_bucket_source = numeric_context[0]
            data["gmm_ae_regime_bucket"] = _bucket_quantiles(
                pd.to_numeric(data[gmm_bucket_source], errors="coerce"),
                labels=(
                    "context_q1_low",
                    "context_q2",
                    "context_q3",
                    "context_q4",
                    "context_q5_high",
                ),
            )
        else:
            data["gmm_ae_regime_bucket"] = "missing"
            gmm_bucket_source = None

    report = {
        "ledger_path": str(ledger_path),
        "rows": int(len(data)),
        "symbols": int(data["symbol"].nunique(dropna=True)),
        "selectors": sorted(data["selector_variant"].astype(str).unique().tolist()),
        "timestamp_min": data["timestamp"].min(),
        "timestamp_max": data["timestamp"].max(),
        "score_column": score_column,
        "full_sl_source": full_sl_source,
        "spread": spread_report,
        "gmm_ae_regime_bucket_source": gmm_bucket_source,
    }
    return data, report


def _summarize_group(group: pd.DataFrame, *, score_column: str) -> dict[str, Any]:
    score = pd.to_numeric(group[score_column], errors="coerce")
    clean = group["clean_positive"].astype(bool)
    dirty = group["dirty_positive"].astype(bool)
    oracle = group["oracle_top"].astype(bool)
    clean_oracle = group["clean_oracle_top"].astype(bool)
    mean_clean = _safe_mean(score.loc[clean])
    mean_dirty = _safe_mean(score.loc[dirty])
    return {
        "rows": int(len(group)),
        "symbols": int(group["symbol"].nunique(dropna=True)),
        "mean_u": _safe_mean(group["u_policy_net"]),
        "bad_mae": _rate(group["bad_mae_1r"]),
        "timeout": _rate(group["timeout_flag"]),
        "full_sl": _rate(group["full_sl"]),
        "clean_positive_rate": _rate(clean),
        "dirty_positive_rate": _rate(dirty),
        "mean_score_clean_positive": mean_clean,
        "mean_score_dirty_positive": mean_dirty,
        "score_gap_clean_minus_dirty": (
            mean_clean - mean_dirty
            if np.isfinite(mean_clean) and np.isfinite(mean_dirty)
            else float("nan")
        ),
        "oracle_rows": int(oracle.sum()),
        "oracle_recall": _rate(oracle),
        "clean_oracle_rows": int(clean_oracle.sum()),
        "clean_oracle_recall": _rate(clean_oracle),
        "mean_spread_bps": _safe_mean(group.get("p75_spread_bps", np.nan)),
        "mean_symbol_trade_count": _safe_mean(group["symbol_trade_count"]),
        "mean_symbol_clean_positive_rate": _safe_mean(group["symbol_clean_positive_rate"]),
    }


def _slice_summary(data: pd.DataFrame, *, keys: list[str], slice_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["selector_variant", *keys]
    for group_values, group in data.groupby(group_keys, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {"slice_name": slice_name}
        for key, value in zip(group_keys, group_values):
            row[key] = value
        row.update(_summarize_group(group, score_column="score_for_gap"))
        rows.append(row)
    return pd.DataFrame(rows)


def _first_numeric(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce")
    return float(values.iloc[0]) if len(values) else float("nan")


def _all_numeric(df: pd.DataFrame, column: str, op: str, threshold: float) -> bool:
    if df.empty or column not in df.columns:
        return False
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return False
    if op == ">":
        return bool((values > threshold).all())
    if op == "<=":
        return bool((values <= threshold).all())
    raise ValueError(f"unsupported comparison op: {op}")


def _selector_gate_summary(
    *,
    aggregate: pd.DataFrame,
    month_rows: pd.DataFrame,
    spread_rows: pd.DataFrame,
    month_spread_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selectors = sorted(aggregate["selector_variant"].astype(str).unique().tolist())
    for selector in selectors:
        agg = aggregate[aggregate["selector_variant"].astype(str).eq(selector)]
        june = month_rows[
            month_rows["selector_variant"].astype(str).eq(selector)
            & month_rows.get("month").astype(str).eq("2026-06")
        ]
        high_spread = spread_rows[
            spread_rows["selector_variant"].astype(str).eq(selector)
            & spread_rows.get("spread_bucket").astype(str).eq("spread_q5_high")
        ]
        high_spread_month = month_spread_rows[
            month_spread_rows["selector_variant"].astype(str).eq(selector)
            & month_spread_rows.get("spread_bucket").astype(str).eq("spread_q5_high")
        ]
        aggregate_score_gap = _first_numeric(agg, "score_gap_clean_minus_dirty")
        june_score_gap = _first_numeric(june, "score_gap_clean_minus_dirty")
        aggregate_gap_pass = bool(np.isfinite(aggregate_score_gap) and aggregate_score_gap > 0.0)
        june_gap_pass = bool(np.isfinite(june_score_gap) and june_score_gap > 0.0)
        high_spread_bucket_pass = bool(
            _all_numeric(high_spread, "mean_u", ">", 0.0)
            and _all_numeric(high_spread, "bad_mae", "<=", 0.65)
            and _all_numeric(high_spread, "timeout", "<=", 0.15)
        )
        high_spread_month_pass = bool(
            _all_numeric(high_spread_month, "mean_u", ">", 0.0)
            and _all_numeric(high_spread_month, "bad_mae", "<=", 0.65)
            and _all_numeric(high_spread_month, "timeout", "<=", 0.15)
        )
        rows.append(
            {
                "selector_variant": selector,
                "rows": int(_first_numeric(agg, "rows")) if np.isfinite(_first_numeric(agg, "rows")) else 0,
                "symbols": int(_first_numeric(agg, "symbols"))
                if np.isfinite(_first_numeric(agg, "symbols"))
                else 0,
                "mean_u": _first_numeric(agg, "mean_u"),
                "bad_mae": _first_numeric(agg, "bad_mae"),
                "timeout": _first_numeric(agg, "timeout"),
                "oracle_recall": _first_numeric(agg, "oracle_recall"),
                "clean_oracle_recall": _first_numeric(agg, "clean_oracle_recall"),
                "aggregate_score_gap": aggregate_score_gap,
                "june_score_gap": june_score_gap,
                "high_spread_mean_u": _first_numeric(high_spread, "mean_u"),
                "high_spread_bad_mae": _first_numeric(high_spread, "bad_mae"),
                "high_spread_timeout": _first_numeric(high_spread, "timeout"),
                "high_spread_min_month_u": (
                    float(pd.to_numeric(high_spread_month["mean_u"], errors="coerce").min())
                    if not high_spread_month.empty
                    else float("nan")
                ),
                "high_spread_max_month_bad_mae": (
                    float(pd.to_numeric(high_spread_month["bad_mae"], errors="coerce").max())
                    if not high_spread_month.empty
                    else float("nan")
                ),
                "high_spread_max_month_timeout": (
                    float(pd.to_numeric(high_spread_month["timeout"], errors="coerce").max())
                    if not high_spread_month.empty
                    else float("nan")
                ),
                "aggregate_gap_pass": aggregate_gap_pass,
                "june_gap_pass": june_gap_pass,
                "high_spread_bucket_pass": high_spread_bucket_pass,
                "high_spread_month_pass": high_spread_month_pass,
                "gate_pass": bool(
                    aggregate_gap_pass
                    and june_gap_pass
                    and high_spread_bucket_pass
                    and high_spread_month_pass
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(output_dir: Path, summary: dict[str, Any], aggregate: pd.DataFrame) -> Path:
    path = output_dir / "broad_source_clean_dirty_diagnostics.md"
    lines = [
        "# Broad Source Clean-vs-Dirty Diagnostics",
        "",
        f"- ledger: `{summary['ledger_path']}`",
        f"- rows: `{summary['rows']}`",
        f"- symbols: `{summary['symbols']}`",
        f"- timestamp range: `{summary['timestamp_min']}` to `{summary['timestamp_max']}`",
        f"- score column: `{summary['score_column']}`",
        f"- full-SL source: `{summary['full_sl_source']}`",
        f"- GMM/AE regime bucket source: `{summary['gmm_ae_regime_bucket_source']}`",
        "",
        "## Gate Read",
        "",
        f"- aggregate score gap pass: `{summary['aggregate_score_gap_pass']}`",
        f"- June score gap pass: `{summary['june_score_gap_pass']}`",
        f"- high-spread bucket pass: `{summary['high_spread_bucket_pass']}`",
        f"- status: `{summary['status']}`",
        "",
        "## Aggregate By Selector",
        "",
    ]
    if aggregate.empty:
        lines.append("_No aggregate rows._")
    else:
        cols = [
            "selector_variant",
            "rows",
            "symbols",
            "mean_u",
            "bad_mae",
            "timeout",
            "full_sl",
            "clean_positive_rate",
            "dirty_positive_rate",
            "score_gap_clean_minus_dirty",
            "oracle_recall",
            "clean_oracle_recall",
        ]
        lines.append(aggregate[cols].to_markdown(index=False))
    selector_gate_path = output_dir / "broad_source_clean_dirty_selector_gate.csv"
    if selector_gate_path.exists():
        selector_gate = pd.read_csv(selector_gate_path)
        if not selector_gate.empty:
            lines.extend(
                [
                    "",
                    "## Selector Gate",
                    "",
                ]
            )
            cols = [
                "selector_variant",
                "mean_u",
                "bad_mae",
                "timeout",
                "oracle_recall",
                "aggregate_score_gap",
                "june_score_gap",
                "high_spread_mean_u",
                "high_spread_bad_mae",
                "high_spread_timeout",
                "gate_pass",
            ]
            lines.append(selector_gate[cols].to_markdown(index=False))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_diagnostics(
    *,
    ledger_path: Path,
    output_dir: Path,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    score_column: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, summary = _prepare_frame(
        ledger_path,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        score_column=score_column,
    )
    slices = [
        _slice_summary(data, keys=[], slice_name="aggregate"),
        _slice_summary(data, keys=["month"], slice_name="month"),
        _slice_summary(data, keys=["side_bucket"], slice_name="side"),
        _slice_summary(data, keys=["spread_bucket"], slice_name="spread_bucket"),
        _slice_summary(
            data,
            keys=["spread_bucket", "side_bucket"],
            slice_name="spread_side",
        ),
        _slice_summary(data, keys=["symbol_liquidity_bucket"], slice_name="symbol_liquidity"),
        _slice_summary(
            data,
            keys=["symbol_clean_rate_bucket"],
            slice_name="symbol_clean_rate",
        ),
        _slice_summary(data, keys=["gmm_ae_regime_bucket"], slice_name="gmm_ae"),
        _slice_summary(data, keys=["month", "side_bucket"], slice_name="month_side"),
        _slice_summary(data, keys=["week", "side_bucket"], slice_name="week_side"),
        _slice_summary(data, keys=["month", "spread_bucket"], slice_name="month_spread"),
        _slice_summary(
            data,
            keys=["month", "spread_bucket", "side_bucket"],
            slice_name="month_spread_side",
        ),
        _slice_summary(data, keys=["month", "symbol_liquidity_bucket"], slice_name="month_liquidity"),
        _slice_summary(
            data,
            keys=["month", "symbol_clean_rate_bucket"],
            slice_name="month_symbol_clean_rate",
        ),
        _slice_summary(data, keys=["month", "gmm_ae_regime_bucket"], slice_name="month_gmm_ae"),
    ]
    diagnostics = pd.concat(slices, ignore_index=True, sort=False)
    aggregate = diagnostics[diagnostics["slice_name"].eq("aggregate")].copy()
    month_rows = diagnostics[diagnostics["slice_name"].eq("month")].copy()
    spread_rows = diagnostics[diagnostics["slice_name"].eq("spread_bucket")].copy()
    month_spread_rows = diagnostics[diagnostics["slice_name"].eq("month_spread")].copy()
    selector_gate = _selector_gate_summary(
        aggregate=aggregate,
        month_rows=month_rows,
        spread_rows=spread_rows,
        month_spread_rows=month_spread_rows,
    )
    aggregate_gap_pass = bool(selector_gate["aggregate_gap_pass"].any())
    june_gap_pass = bool(selector_gate["june_gap_pass"].any())
    high_spread_pass = bool(selector_gate["high_spread_bucket_pass"].any())
    selector_gate_pass = bool(selector_gate["gate_pass"].any())
    passing_selectors = (
        selector_gate.loc[selector_gate["gate_pass"], "selector_variant"].astype(str).tolist()
    )
    summary.update(
        {
            "aggregate_score_gap_pass": aggregate_gap_pass,
            "june_score_gap_pass": june_gap_pass,
            "high_spread_bucket_pass": high_spread_pass,
            "selector_gate_pass": selector_gate_pass,
            "passing_selectors": passing_selectors,
            "status": "pass" if selector_gate_pass else "fail",
        }
    )
    paths = {
        "diagnostics": output_dir / "broad_source_clean_dirty_diagnostics.csv",
        "aggregate": output_dir / "broad_source_clean_dirty_aggregate.csv",
        "selector_gate": output_dir / "broad_source_clean_dirty_selector_gate.csv",
        "prepared_ledger": output_dir / "broad_source_clean_dirty_prepared_ledger.parquet",
        "manifest": output_dir / "manifest.json",
    }
    diagnostics.to_csv(paths["diagnostics"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    selector_gate.to_csv(paths["selector_gate"], index=False)
    data.to_parquet(paths["prepared_ledger"], index=False)
    markdown = _write_markdown(output_dir, summary, aggregate)
    paths["markdown"] = markdown
    manifest = {**summary, "outputs": {k: str(v) for k, v in paths.items()}}
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spread-baseline-path", type=Path, default=DEFAULT_SPREAD_PATH)
    parser.add_argument("--spread-rank-column", type=str, default="p75_spread_bps")
    parser.add_argument("--score-column", type=str, default="selector_score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostics(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=str(args.spread_rank_column),
        score_column=str(args.score_column),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
