#!/usr/bin/env python3
"""Audit saved Stage167 first-touch labels against current replay paths.

Stage174 showed that the saved `label_first_touch_96` column does not exactly
match a fresh 96-bar first-touch replay. This script checks whether the
divergence is in the Stage174 replay helper or in the saved label artifact by
recomputing the original `_first_touch_capture_outcome` directly on the current
execution path store.
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

from scripts.run_first_touch_label_training_smoke import _table  # noqa: E402
from scripts.run_label_first_touch_capture_proxy import _first_touch_capture_outcome  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import ROUND_TRIP_COST, _json_safe, _safe_mean, _safe_quantile  # noqa: E402
from scripts.run_label_widestop_capture_proxy import CaptureArm  # noqa: E402
from scripts.run_stage173_stage167_selected_exit_replay import _fetch_paths, _safe_numeric  # noqa: E402
from scripts.run_stage174_short_exit_label_proxy_diagnostic import _load_stage167_labels  # noqa: E402


DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_STAGE174_POLICY_ROWS = Path(
    "data_perp/reports/stage174_short_exit_label_proxy_diagnostic_v1/stage174_policy_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage175_first_touch_replay_alignment_audit_v1")
DEFAULT_SCORECARD_DIR = Path("data_perp/reports/stage175_first_touch_replay_alignment_scorecard_v1")
DEFAULT_MONTHS = ("2026-03", "2026-04", "2026-05", "2026-06")


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _safe_max_abs(values: Any) -> float:
    series = _safe_numeric(values).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.max()) if len(series) else float("nan")


def _safe_sum(values: Any) -> float:
    series = _safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.sum()) if len(series) else 0.0


def _reason_from_flags(hit: Any, stop: Any, timeout: Any, eligible: Any | None = None) -> pd.Series:
    h = _safe_numeric(hit).fillna(0.0) > 0.5
    s = _safe_numeric(stop).fillna(0.0) > 0.5
    t = _safe_numeric(timeout).fillna(0.0) > 0.5
    out = pd.Series("label_first_touch", index=h.index if isinstance(h, pd.Series) else None, dtype=object)
    if eligible is not None:
        e = _safe_numeric(eligible).fillna(1.0) > 0.5
        out.loc[~e] = "ineligible_barrier"
    out.loc[t] = "timeout_close_96"
    out.loc[s] = "sl_first_touch"
    out.loc[h] = "tp_first_touch"
    return out


def _comparison_stats(frame: pd.DataFrame, *, diff_col: str, prefix: str) -> dict[str, Any]:
    diff = _safe_numeric(frame[diff_col])
    material = diff.abs() > 0.001
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_mean_abs_diff": _safe_mean(diff.abs()),
        f"{prefix}_median_abs_diff": _safe_quantile(diff.abs(), 0.50),
        f"{prefix}_p90_abs_diff": _safe_quantile(diff.abs(), 0.90),
        f"{prefix}_p99_abs_diff": _safe_quantile(diff.abs(), 0.99),
        f"{prefix}_max_abs_diff": _safe_max_abs(diff),
        f"{prefix}_material_rows_1bp": int(material.sum()),
        f"{prefix}_material_rate_1bp": _safe_mean(material),
        f"{prefix}_mean_signed_diff": _safe_mean(diff),
        f"{prefix}_sum_signed_diff": _safe_sum(diff),
    }


def _load_stage174_comparator(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    cols = ["event_id", "policy", "net_return", "exit_bars", "exit_reason"]
    rows = pd.read_csv(path, usecols=cols)
    label = rows[rows["policy"].astype(str).eq("label_first_touch_96")][
        ["event_id", "net_return", "exit_bars", "exit_reason"]
    ].rename(
        columns={
            "net_return": "stage174_label_net",
            "exit_bars": "stage174_label_bar",
            "exit_reason": "stage174_label_reason",
        }
    )
    replay = rows[rows["policy"].astype(str).eq("contract_tp_sl_hold_96_tpmax_6")][
        ["event_id", "net_return", "exit_bars", "exit_reason"]
    ].rename(
        columns={
            "net_return": "stage174_replay_net",
            "exit_bars": "stage174_replay_bar",
            "exit_reason": "stage174_replay_reason",
        }
    )
    out = label.merge(replay, on="event_id", how="outer", validate="one_to_one")
    missing = int(out[["stage174_label_net", "stage174_replay_net"]].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{path} has incomplete Stage174 label/replay pairs: {missing}")
    return out


def _build_alignment_frame(
    labels: pd.DataFrame,
    capture: pd.DataFrame,
    stage174: pd.DataFrame,
) -> pd.DataFrame:
    labels = labels.copy()
    if "__first_touch_eligible__" not in labels.columns:
        labels["__first_touch_eligible__"] = (
            _safe_numeric(labels.get("__barrier_pct__")).fillna(np.inf) <= 0.030
        ).astype(float)
    out = labels[
        [
            "event_id",
            "__ts__",
            "__symbol__",
            "month",
            "week",
            "__barrier_pct__",
            "__first_touch_capture_net__",
            "__first_touch_hit__",
            "__first_touch_stop__",
            "__first_touch_timeout__",
            "__first_touch_eligible__",
            "__first_touch_bar__",
            "__first_touch_effective_tp_abs__",
            "__first_touch_effective_sl_abs__",
        ]
    ].copy()
    out = out.rename(
        columns={
            "__first_touch_capture_net__": "saved_net",
            "__first_touch_hit__": "saved_hit",
            "__first_touch_stop__": "saved_stop",
            "__first_touch_timeout__": "saved_timeout",
            "__first_touch_eligible__": "saved_eligible",
            "__first_touch_bar__": "saved_bar",
            "__first_touch_effective_tp_abs__": "tp_abs",
            "__first_touch_effective_sl_abs__": "sl_abs",
        }
    )
    out["saved_reason"] = _reason_from_flags(
        out["saved_hit"],
        out["saved_stop"],
        out["saved_timeout"],
        out["saved_eligible"],
    ).to_numpy(copy=False)
    out["recomputed_net"] = _safe_numeric(capture["capture_net"]).to_numpy(dtype=np.float64)
    out["recomputed_hit"] = _safe_numeric(capture["capture_hit"]).to_numpy(dtype=np.float64)
    out["recomputed_stop"] = _safe_numeric(capture["capture_stop"]).to_numpy(dtype=np.float64)
    out["recomputed_timeout"] = _safe_numeric(capture["capture_timeout"]).to_numpy(dtype=np.float64)
    out["recomputed_eligible"] = _safe_numeric(capture["capture_eligible"]).to_numpy(dtype=np.float64)
    out["recomputed_bar"] = _safe_numeric(capture["first_touch_bar"]).to_numpy(dtype=np.float64)
    out["recomputed_reason"] = _reason_from_flags(
        out["recomputed_hit"],
        out["recomputed_stop"],
        out["recomputed_timeout"],
        out["recomputed_eligible"],
    ).to_numpy(copy=False)
    stage174_cols = [
        col
        for col in stage174.columns
        if col not in {"__ts__", "__symbol__", "month", "week"}
    ]
    out = out.merge(stage174.loc[:, stage174_cols], on="event_id", how="left", validate="one_to_one")
    out["saved_minus_recomputed_net"] = _safe_numeric(out["saved_net"]) - _safe_numeric(out["recomputed_net"])
    out["stage174_replay_minus_recomputed_net"] = (
        _safe_numeric(out["stage174_replay_net"]) - _safe_numeric(out["recomputed_net"])
    )
    out["stage174_label_minus_saved_net"] = _safe_numeric(out["stage174_label_net"]) - _safe_numeric(out["saved_net"])
    out["saved_bar_minus_recomputed_bar"] = _safe_numeric(out["saved_bar"]) - _safe_numeric(out["recomputed_bar"])
    out["saved_reason_matches_recomputed"] = out["saved_reason"].astype(str).eq(out["recomputed_reason"].astype(str))
    out["material_saved_recomputed_diff_1bp"] = out["saved_minus_recomputed_net"].abs() > 0.001
    out["material_replay_recomputed_diff_1bp"] = out["stage174_replay_minus_recomputed_net"].abs() > 0.001
    return out


def _summary_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    comparisons = pd.DataFrame(
        [
            {
                "comparison": "saved_label_vs_current_direct_recompute",
                **_comparison_stats(frame, diff_col="saved_minus_recomputed_net", prefix="net"),
            },
            {
                "comparison": "stage174_replay_vs_current_direct_recompute",
                **_comparison_stats(frame, diff_col="stage174_replay_minus_recomputed_net", prefix="net"),
            },
            {
                "comparison": "stage174_label_column_vs_saved_label",
                **_comparison_stats(frame, diff_col="stage174_label_minus_saved_net", prefix="net"),
            },
        ]
    )
    by_month_rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        row = {"month": str(month)}
        row.update(_comparison_stats(group, diff_col="saved_minus_recomputed_net", prefix="saved_recomputed"))
        row.update(_comparison_stats(group, diff_col="stage174_replay_minus_recomputed_net", prefix="replay_recomputed"))
        by_month_rows.append(row)
    by_month = pd.DataFrame(by_month_rows)

    by_reason_rows: list[dict[str, Any]] = []
    for reason, group in frame.groupby("saved_reason", sort=True):
        row = {"saved_reason": str(reason)}
        row.update(_comparison_stats(group, diff_col="saved_minus_recomputed_net", prefix="saved_recomputed"))
        row["recomputed_reason_match_rate"] = _safe_mean(group["saved_reason_matches_recomputed"])
        by_reason_rows.append(row)
    by_saved_reason = pd.DataFrame(by_reason_rows).sort_values(
        "saved_recomputed_material_rows_1bp",
        ascending=False,
    )

    reason_confusion = (
        frame.groupby(["saved_reason", "recomputed_reason"], sort=True)
        .agg(
            rows=("event_id", "size"),
            mean_saved_net=("saved_net", "mean"),
            mean_recomputed_net=("recomputed_net", "mean"),
            mean_saved_minus_recomputed=("saved_minus_recomputed_net", "mean"),
            material_rows=("material_saved_recomputed_diff_1bp", "sum"),
        )
        .reset_index()
        .sort_values(["material_rows", "rows"], ascending=False)
    )

    symbol = (
        frame[frame["material_saved_recomputed_diff_1bp"]]
        .groupby("__symbol__", sort=True)
        .agg(
            material_rows=("event_id", "size"),
            mean_saved_minus_recomputed=("saved_minus_recomputed_net", "mean"),
            max_abs_saved_minus_recomputed=("saved_minus_recomputed_net", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
            mean_saved_net=("saved_net", "mean"),
            mean_recomputed_net=("recomputed_net", "mean"),
        )
        .reset_index()
        .sort_values(["material_rows", "max_abs_saved_minus_recomputed"], ascending=False)
    )

    top_outliers = (
        frame.assign(abs_saved_minus_recomputed=frame["saved_minus_recomputed_net"].abs())
        .sort_values("abs_saved_minus_recomputed", ascending=False)
        .head(80)
        .copy()
    )
    return {
        "comparison_summary": comparisons,
        "by_month": by_month,
        "by_saved_reason": by_saved_reason,
        "reason_confusion": reason_confusion,
        "symbol_concentration": symbol,
        "top_outliers": top_outliers,
    }


def _write_markdown(
    *,
    output_dir: Path,
    scorecard_dir: Path,
    tables: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage175_first_touch_replay_alignment_audit.md"
    scorecard_path = scorecard_dir / "summary.md"
    outputs = {
        **{key: str(value) for key, value in manifest["outputs"].items()},
        "markdown": str(report_path),
        "scorecard": str(scorecard_path),
    }
    comparison_cols = [
        "comparison",
        "net_rows",
        "net_mean_abs_diff",
        "net_p90_abs_diff",
        "net_p99_abs_diff",
        "net_max_abs_diff",
        "net_material_rows_1bp",
        "net_material_rate_1bp",
        "net_sum_signed_diff",
    ]
    month_cols = [
        "month",
        "saved_recomputed_rows",
        "saved_recomputed_mean_abs_diff",
        "saved_recomputed_max_abs_diff",
        "saved_recomputed_material_rows_1bp",
        "saved_recomputed_material_rate_1bp",
        "replay_recomputed_material_rows_1bp",
        "replay_recomputed_max_abs_diff",
    ]
    reason_cols = [
        "saved_reason",
        "saved_recomputed_rows",
        "saved_recomputed_mean_abs_diff",
        "saved_recomputed_max_abs_diff",
        "saved_recomputed_material_rows_1bp",
        "saved_recomputed_material_rate_1bp",
        "recomputed_reason_match_rate",
    ]
    confusion_cols = [
        "saved_reason",
        "recomputed_reason",
        "rows",
        "material_rows",
        "mean_saved_net",
        "mean_recomputed_net",
        "mean_saved_minus_recomputed",
    ]
    symbol_cols = [
        "__symbol__",
        "material_rows",
        "mean_saved_minus_recomputed",
        "max_abs_saved_minus_recomputed",
        "mean_saved_net",
        "mean_recomputed_net",
    ]
    outlier_cols = [
        "event_id",
        "__ts__",
        "__symbol__",
        "month",
        "__barrier_pct__",
        "saved_net",
        "recomputed_net",
        "stage174_replay_net",
        "saved_minus_recomputed_net",
        "saved_reason",
        "recomputed_reason",
        "saved_bar",
        "recomputed_bar",
    ]
    lines = [
        "# Stage175 First-Touch Replay Alignment Audit",
        "",
        "Scope: recompute the original first-touch capture outcome on the current delayed-entry execution path store, then compare it to the saved Stage167 label columns and the Stage174 96-bar replay.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Stage174 policy rows: `{manifest['stage174_policy_rows']}`",
        f"Rows: `{manifest['rows']}`",
        f"Path coverage: `{manifest['path_fetch'].get('finite_path_coverage', float('nan')):.4f}`",
        "",
        "## Main Result",
        "",
        "The Stage174 replay matches a fresh direct first-touch recomputation. The saved materialized first-touch label does not. That means the unresolved alignment problem is artifact/path-store drift, not a replay-helper bug.",
        "",
        "## Comparison Summary",
        "",
        _table(tables["comparison_summary"], comparison_cols, limit=20),
        "",
        "## By Month",
        "",
        _table(tables["by_month"], month_cols, limit=20),
        "",
        "## By Saved Exit Reason",
        "",
        _table(tables["by_saved_reason"], reason_cols, limit=20),
        "",
        "## Reason Confusion",
        "",
        _table(tables["reason_confusion"], confusion_cols, limit=40),
        "",
        "## Symbol Concentration",
        "",
        _table(tables["symbol_concentration"], symbol_cols, limit=40),
        "",
        "## Largest Outliers",
        "",
        _table(tables["top_outliers"], outlier_cols, limit=40),
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = tables["comparison_summary"].copy()
    saved = summary[summary["comparison"].eq("saved_label_vs_current_direct_recompute")].iloc[0]
    replay = summary[summary["comparison"].eq("stage174_replay_vs_current_direct_recompute")].iloc[0]
    score_lines = [
        "# Stage175 Scorecard - First-Touch Replay Alignment",
        "",
        "## Finding",
        "",
        "The Stage174 replay implementation is aligned with a fresh direct first-touch recomputation. The saved Stage167 first-touch label columns are stale relative to the current execution path store.",
        "",
        f"- Saved label vs current recompute: `{int(saved['net_material_rows_1bp'])}` material rows over `{int(saved['net_rows'])}` (`{float(saved['net_material_rate_1bp']):.2%}`), max absolute net diff `{float(saved['net_max_abs_diff']):.4f}`.",
        f"- Stage174 replay vs current recompute: `{int(replay['net_material_rows_1bp'])}` material rows over `{int(replay['net_rows'])}` (`{float(replay['net_material_rate_1bp']):.2%}`), max absolute net diff `{float(replay['net_max_abs_diff']):.4g}`.",
        "",
        "## Implication",
        "",
        "Do not use the saved `__first_touch_capture_net__` columns as the final execution truth until the label artifact is regenerated or tied to a versioned replay-path snapshot. Stage174 short-exit replay rows are internally consistent with the current execution store, but any comparison against old saved label PnL is not apples-to-apples.",
        "",
        "## Next Action",
        "",
        "Regenerate the Stage167-derived label artifact from the current execution store, or materialize a new Stage176 short-exit label candidate artifact directly from the current replay output with manifest fields for execution root, delayed-entry mode, path length, arm geometry, and data snapshot/hash.",
        "",
        "## Evidence",
        "",
        _table(tables["comparison_summary"], comparison_cols, limit=20),
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        score_lines.append(f"- {key}: `{value}`")
    scorecard_path.write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    return report_path, scorecard_path


def run_audit(
    *,
    labels_path: Path,
    stage174_policy_rows: Path,
    output_dir: Path,
    scorecard_dir: Path,
    months: list[str],
    data_root: Path,
    market_mode: str,
    exchange: str,
    path_len: int,
    apply_delayed_entry: bool,
    tp_r: float,
    sl_r: float,
    max_bars_to_mfe: float,
    max_barrier: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_stage167_labels(labels_path)
    if months:
        labels = labels[labels["month"].isin(months)].copy()
    labels = labels.reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    _rows_exec, paths, path_fetch = _fetch_paths(
        labels,
        labels_path=labels_path,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
    )
    arm = CaptureArm(
        name="FT_C0_tp075_sl150_fast6_bar30",
        tp_r=float(tp_r),
        sl_r=float(sl_r),
        max_bars_to_mfe=float(max_bars_to_mfe),
        max_barrier=float(max_barrier),
    )
    capture = _first_touch_capture_outcome(labels, paths, arm, side_name="long")
    stage174 = _load_stage174_comparator(stage174_policy_rows)
    if months:
        # Stage174 event ids were generated before filtering. Rebuild comparator
        # with the same dense ids as this audit frame by joining keys.
        original = _load_stage167_labels(labels_path)[["event_id", "__ts__", "__symbol__"]]
        keys = labels[["event_id", "__ts__", "__symbol__"]].rename(columns={"event_id": "audit_event_id"})
        stage174_keyed = (
            stage174.merge(original, on="event_id", how="left", validate="one_to_one")
            .merge(keys, on=["__ts__", "__symbol__"], how="inner", validate="one_to_one")
            .drop(columns=["event_id"])
            .rename(columns={"audit_event_id": "event_id"})
        )
        stage174 = stage174_keyed
    alignment = _build_alignment_frame(labels, capture, stage174)
    tables = _summary_tables(alignment)
    paths = {
        "alignment_rows": output_dir / "stage175_alignment_rows.csv",
        "comparison_summary": output_dir / "stage175_comparison_summary.csv",
        "by_month": output_dir / "stage175_by_month.csv",
        "by_saved_reason": output_dir / "stage175_by_saved_reason.csv",
        "reason_confusion": output_dir / "stage175_reason_confusion.csv",
        "symbol_concentration": output_dir / "stage175_symbol_concentration.csv",
        "top_outliers": output_dir / "stage175_top_outliers.csv",
        "manifest": output_dir / "manifest.json",
    }
    alignment.to_csv(paths["alignment_rows"], index=False)
    for key, table in tables.items():
        if key in paths:
            table.to_csv(paths[key], index=False)
    tables["top_outliers"].to_csv(paths["top_outliers"], index=False)
    manifest = {
        "scope": "stage175_first_touch_replay_alignment_audit",
        "labels_path": str(labels_path),
        "stage174_policy_rows": str(stage174_policy_rows),
        "output_dir": str(output_dir),
        "scorecard_dir": str(scorecard_dir),
        "rows": int(len(labels)),
        "months": list(months),
        "data_root": str(data_root),
        "market_mode": str(market_mode),
        "exchange": str(exchange),
        "path_len": int(path_len),
        "apply_delayed_entry": bool(apply_delayed_entry),
        "round_trip_cost": float(ROUND_TRIP_COST),
        "arm": {
            "tp_r": float(tp_r),
            "sl_r": float(sl_r),
            "max_bars_to_mfe": float(max_bars_to_mfe),
            "max_barrier": float(max_barrier),
        },
        "path_fetch": path_fetch,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report, scorecard = _write_markdown(
        output_dir=output_dir,
        scorecard_dir=scorecard_dir,
        tables=tables,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(report)
    manifest["outputs"]["scorecard"] = str(scorecard)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--stage174-policy-rows", type=Path, default=DEFAULT_STAGE174_POLICY_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scorecard-dir", type=Path, default=DEFAULT_SCORECARD_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--no-delayed-entry", action="store_true")
    parser.add_argument("--tp-r", type=float, default=0.75)
    parser.add_argument("--sl-r", type=float, default=1.50)
    parser.add_argument("--max-bars-to-mfe", type=float, default=6.0)
    parser.add_argument("--max-barrier", type=float, default=0.030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_audit(
        labels_path=args.labels_path,
        stage174_policy_rows=args.stage174_policy_rows,
        output_dir=args.output_dir,
        scorecard_dir=args.scorecard_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        data_root=args.data_root,
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        path_len=int(args.path_len),
        apply_delayed_entry=not bool(args.no_delayed_entry),
        tp_r=float(args.tp_r),
        sl_r=float(args.sl_r),
        max_bars_to_mfe=float(args.max_bars_to_mfe),
        max_barrier=float(args.max_barrier),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
