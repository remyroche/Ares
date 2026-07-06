#!/usr/bin/env python3
"""Execution-side diagnostic for Stage167 selected first-touch trades.

This report starts from the Stage167 selected-row ledger and joins it back to
the Stage167 label artifact. It checks whether the label artifact contains a
separate longer-hold PnL field, then summarizes selected-row first-touch net,
timing, full-path adverse excursion, and path archetypes month by month and
week by week.
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
from scripts.run_label_quality_proxy_diagnostics import _json_safe  # noqa: E402


DEFAULT_LEDGER_CSV = Path(
    "data_perp/reports/stage167_full_path_tail_feature_gap_v1/"
    "stage167_full_path_tail_selected_ledger.csv"
)
DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage172_stage167_firsttouch_exit_diagnostic_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")


def _safe_numeric(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _safe_mean(values: Any) -> float:
    series = _safe_numeric(values)
    if series.dropna().empty:
        return float("nan")
    return float(series.mean())


def _safe_sum(values: Any) -> float:
    series = _safe_numeric(values)
    if series.dropna().empty:
        return 0.0
    return float(series.sum())


def _safe_quantile(values: Any, q: float) -> float:
    series = _safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return float("nan")
    return float(series.quantile(float(q)))


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "__ts__",
        "__symbol__",
        "period",
        "first_touch_net",
        "clean_first_touch_exec",
        "first_touch_timeout",
        "first_touch_mae_to_sl",
        "full_path_mae_to_sl",
        "full_path_mfe_to_tp",
        "score",
        "utility_pred",
        "support_pred",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["period"] = frame["period"].astype(str)
    if frame["__ts__"].isna().any():
        raise ValueError(f"{path} contains non-parseable __ts__ values")
    dupes = int(frame.duplicated(["__ts__", "__symbol__"]).sum())
    if dupes:
        raise ValueError(f"{path} contains duplicate __ts__/__symbol__ keys: {dupes}")
    return frame.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)


def _label_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No parquet label files found at {path}")


def _load_label_subset(path: Path) -> pd.DataFrame:
    requested = [
        "__ts__",
        "__symbol__",
        "__u_policy_net__",
        "__r_policy_net__",
        "__y_ret__",
        "__y_bin__",
        "__y_outcome__",
        "__is_timeout__",
        "__bars_policy__",
        "__bars_to_mfe__",
        "__first_touch_bar__",
        "__first_touch_hit__",
        "__first_touch_stop__",
        "__first_touch_timeout__",
        "__first_touch_capture_net__",
        "__first_touch_mae_to_sl__",
        "__first_touch_mfe_to_tp__",
        "__first_touch_full_path_mae_to_sl__",
        "__first_touch_full_path_mfe_to_tp__",
        "__barrier_pct__",
        "__mfe_ret__",
        "__mae_ret__",
    ]
    parts: list[pd.DataFrame] = []
    for file in _label_parquet_files(path):
        columns = pd.read_parquet(file, columns=None).columns
        keep = [col for col in requested if col in columns]
        missing_keys = sorted({"__ts__", "__symbol__"}.difference(keep))
        if missing_keys:
            raise ValueError(f"{file} is missing label key columns: {missing_keys}")
        parts.append(pd.read_parquet(file, columns=keep))
    frame = pd.concat(parts, ignore_index=True)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    dupes = int(frame.duplicated(["__ts__", "__symbol__"]).sum())
    if dupes:
        raise ValueError(f"{path} contains duplicate __ts__/__symbol__ keys: {dupes}")
    return frame.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)


def _join_ledger_labels(ledger: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    joined = ledger.merge(labels, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")
    missing = int(joined["__u_policy_net__"].isna().sum()) if "__u_policy_net__" in joined.columns else len(joined)
    if missing:
        raise ValueError(f"Missing joined label rows for selected ledger keys: {missing}")
    joined["month"] = joined["__ts__"].dt.to_period("M").astype(str)
    joined["week"] = joined["__ts__"].dt.to_period("W-SUN").astype(str)
    joined["policy_minus_first_touch_bars"] = (
        _safe_numeric(joined.get("__bars_policy__"), index=joined.index)
        - _safe_numeric(joined.get("__first_touch_bar__"), index=joined.index)
    )
    return joined


def _first_touch_clean(frame: pd.DataFrame, *, first_touch_clean_r: float) -> pd.Series:
    return (
        (_safe_numeric(frame.get("clean_first_touch_exec"), index=frame.index) >= 0.5)
        & (_safe_numeric(frame.get("first_touch_timeout"), index=frame.index) < 0.5)
        & (_safe_numeric(frame.get("first_touch_net"), index=frame.index) > 0.0)
        & (_safe_numeric(frame.get("first_touch_mae_to_sl"), index=frame.index) <= float(first_touch_clean_r))
    ).fillna(False)


def _add_archetypes(
    frame: pd.DataFrame,
    *,
    first_touch_clean_r: float,
    full_path_clean_r: float,
    full_path_dirty_r: float,
) -> pd.DataFrame:
    out = frame.copy()
    first_clean = _first_touch_clean(out, first_touch_clean_r=first_touch_clean_r)
    full_mae = _safe_numeric(out.get("full_path_mae_to_sl"), index=out.index)
    clean_continuation = first_clean & (full_mae <= float(full_path_clean_r))
    dirty_reversal = first_clean & (full_mae >= float(full_path_dirty_r))
    noisy_middle = first_clean & (~clean_continuation) & (~dirty_reversal)
    dirty_first_touch = ~first_clean
    out["stage172_first_touch_clean"] = first_clean.astype(float)
    out["stage172_clean_continuation"] = clean_continuation.astype(float)
    out["stage172_dirty_reversal"] = dirty_reversal.astype(float)
    out["stage172_noisy_middle"] = noisy_middle.astype(float)
    out["stage172_dirty_first_touch"] = dirty_first_touch.astype(float)
    out["stage172_archetype"] = np.select(
        [
            clean_continuation.to_numpy(dtype=bool),
            dirty_reversal.to_numpy(dtype=bool),
            noisy_middle.to_numpy(dtype=bool),
            dirty_first_touch.to_numpy(dtype=bool),
        ],
        [
            "clean_continuation",
            "dirty_reversal",
            "noisy_middle",
            "dirty_first_touch",
        ],
        default="unclassified",
    )
    return out


def _top_symbol_share(frame: pd.DataFrame) -> float:
    if frame.empty or "__symbol__" not in frame.columns:
        return float("nan")
    return float(frame["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])


def _summarize_group(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(key, observed=True, sort=True):
        ft = _safe_numeric(group.get("first_touch_net"), index=group.index)
        u_net = _safe_numeric(group.get("__u_policy_net__"), index=group.index)
        y_ret = _safe_numeric(group.get("__y_ret__"), index=group.index)
        r_net = _safe_numeric(group.get("__r_policy_net__"), index=group.index)
        clean = _safe_numeric(group.get("clean_first_touch_exec"), index=group.index)
        timeout = _safe_numeric(group.get("first_touch_timeout"), index=group.index)
        first_mae = _safe_numeric(group.get("first_touch_mae_to_sl"), index=group.index)
        full_mae = _safe_numeric(group.get("full_path_mae_to_sl"), index=group.index)
        full_mfe = _safe_numeric(group.get("full_path_mfe_to_tp"), index=group.index)
        hit = _safe_numeric(group.get("__first_touch_hit__"), index=group.index)
        stop = _safe_numeric(group.get("__first_touch_stop__"), index=group.index)
        ft_bar = _safe_numeric(group.get("__first_touch_bar__"), index=group.index)
        policy_bar = _safe_numeric(group.get("__bars_policy__"), index=group.index)
        extra_bars = _safe_numeric(group.get("policy_minus_first_touch_bars"), index=group.index)
        row = {
            key: str(value),
            "rows": int(len(group)),
            "first_touch_sum_net": _safe_sum(ft),
            "first_touch_mean_net": _safe_mean(ft),
            "first_touch_positive_rate": _safe_mean(ft > 0.0),
            "label_u_policy_sum_net": _safe_sum(u_net),
            "label_u_policy_mean_net": _safe_mean(u_net),
            "label_y_ret_sum": _safe_sum(y_ret),
            "label_r_policy_sum": _safe_sum(r_net),
            "delta_sum_ft_vs_u_policy": _safe_sum(ft) - _safe_sum(u_net),
            "max_abs_diff_ft_vs_u_policy": float((ft - u_net).abs().max()) if len(group) else float("nan"),
            "max_abs_diff_ft_vs_y_ret": float((ft - y_ret).abs().max()) if len(group) else float("nan"),
            "clean_first_touch_exec_rate": _safe_mean(clean),
            "first_touch_hit_rate": _safe_mean(hit),
            "first_touch_stop_rate": _safe_mean(stop),
            "first_touch_timeout_rate": _safe_mean(timeout >= 0.5),
            "bad_first_touch_mae_1r_rate": _safe_mean(first_mae >= 1.0),
            "first_touch_bar_p50": _safe_quantile(ft_bar, 0.50),
            "first_touch_bar_p90": _safe_quantile(ft_bar, 0.90),
            "policy_bar_p50": _safe_quantile(policy_bar, 0.50),
            "policy_bar_p90": _safe_quantile(policy_bar, 0.90),
            "extra_bars_after_first_touch_p50": _safe_quantile(extra_bars, 0.50),
            "extra_bars_after_first_touch_p90": _safe_quantile(extra_bars, 0.90),
            "bad_full_path_mae_3r_rate": _safe_mean(full_mae >= 3.0),
            "full_path_mae_to_sl_p50": _safe_quantile(full_mae, 0.50),
            "full_path_mae_to_sl_p90": _safe_quantile(full_mae, 0.90),
            "full_path_mfe_to_tp_p50": _safe_quantile(full_mfe, 0.50),
            "full_path_mfe_to_tp_p90": _safe_quantile(full_mfe, 0.90),
            "clean_continuation_rate": _safe_mean(group.get("stage172_clean_continuation")),
            "dirty_reversal_rate": _safe_mean(group.get("stage172_dirty_reversal")),
            "noisy_middle_rate": _safe_mean(group.get("stage172_noisy_middle")),
            "dirty_first_touch_rate": _safe_mean(group.get("stage172_dirty_first_touch")),
            "top_symbol_share": _top_symbol_share(group),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _archetype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (month, archetype), group in frame.groupby(["month", "stage172_archetype"], observed=True, sort=True):
        rows.append(
            {
                "month": str(month),
                "archetype": str(archetype),
                "rows": int(len(group)),
                "row_share": float(len(group) / max(1, int((frame["month"] == month).sum()))),
                "first_touch_sum_net": _safe_sum(group.get("first_touch_net")),
                "first_touch_mean_net": _safe_mean(group.get("first_touch_net")),
                "first_touch_bar_p50": _safe_quantile(group.get("__first_touch_bar__"), 0.50),
                "first_touch_bar_p90": _safe_quantile(group.get("__first_touch_bar__"), 0.90),
                "policy_bar_p50": _safe_quantile(group.get("__bars_policy__"), 0.50),
                "extra_bars_after_first_touch_p90": _safe_quantile(group.get("policy_minus_first_touch_bars"), 0.90),
                "bad_full_path_mae_3r_rate": _safe_mean(_safe_numeric(group.get("full_path_mae_to_sl")) >= 3.0),
                "full_path_mae_to_sl_p90": _safe_quantile(group.get("full_path_mae_to_sl"), 0.90),
                "top_symbol_share": _top_symbol_share(group),
            }
        )
    return pd.DataFrame(rows)


def _speed_bucket(bar: Any) -> str:
    if pd.isna(bar):
        return "missing"
    value = float(bar)
    if value <= 2.0:
        return "b00_<=2"
    if value <= 4.0:
        return "b01_3_4"
    if value <= 8.0:
        return "b02_5_8"
    return "b03_>8"


def _speed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    with_bucket = frame.copy()
    with_bucket["first_touch_speed_bucket"] = with_bucket["__first_touch_bar__"].map(_speed_bucket)
    rows: list[dict[str, Any]] = []
    for (month, bucket), group in with_bucket.groupby(["month", "first_touch_speed_bucket"], observed=True, sort=True):
        rows.append(
            {
                "month": str(month),
                "first_touch_speed_bucket": str(bucket),
                "rows": int(len(group)),
                "row_share": float(len(group) / max(1, int((with_bucket["month"] == month).sum()))),
                "first_touch_sum_net": _safe_sum(group.get("first_touch_net")),
                "first_touch_mean_net": _safe_mean(group.get("first_touch_net")),
                "dirty_reversal_rate": _safe_mean(group.get("stage172_dirty_reversal")),
                "bad_full_path_mae_3r_rate": _safe_mean(_safe_numeric(group.get("full_path_mae_to_sl")) >= 3.0),
                "full_path_mae_to_sl_p90": _safe_quantile(group.get("full_path_mae_to_sl"), 0.90),
                "policy_bar_p50": _safe_quantile(group.get("__bars_policy__"), 0.50),
                "extra_bars_after_first_touch_p90": _safe_quantile(group.get("policy_minus_first_touch_bars"), 0.90),
            }
        )
    return pd.DataFrame(rows)


def _symbol_summary(frame: pd.DataFrame, *, min_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (month, symbol), group in frame.groupby(["month", "__symbol__"], observed=True, sort=True):
        if len(group) < int(min_rows):
            continue
        rows.append(
            {
                "month": str(month),
                "__symbol__": str(symbol),
                "rows": int(len(group)),
                "first_touch_sum_net": _safe_sum(group.get("first_touch_net")),
                "first_touch_mean_net": _safe_mean(group.get("first_touch_net")),
                "dirty_reversal_rate": _safe_mean(group.get("stage172_dirty_reversal")),
                "bad_full_path_mae_3r_rate": _safe_mean(_safe_numeric(group.get("full_path_mae_to_sl")) >= 3.0),
                "full_path_mae_to_sl_p90": _safe_quantile(group.get("full_path_mae_to_sl"), 0.90),
            }
        )
    return pd.DataFrame(rows).sort_values(["month", "rows", "first_touch_sum_net"], ascending=[True, False, False])


def _write_markdown(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    archetypes: pd.DataFrame,
    speed: pd.DataFrame,
    symbols: pd.DataFrame,
) -> Path:
    path = output_dir / "stage172_stage167_firsttouch_exit_diagnostic.md"
    monthly_cols = [
        "month",
        "rows",
        "first_touch_sum_net",
        "first_touch_mean_net",
        "label_u_policy_sum_net",
        "delta_sum_ft_vs_u_policy",
        "clean_first_touch_exec_rate",
        "first_touch_bar_p50",
        "first_touch_bar_p90",
        "policy_bar_p50",
        "policy_bar_p90",
        "extra_bars_after_first_touch_p90",
        "bad_full_path_mae_3r_rate",
        "full_path_mae_to_sl_p90",
        "clean_continuation_rate",
        "dirty_reversal_rate",
    ]
    weekly_cols = [
        "week",
        "rows",
        "first_touch_sum_net",
        "first_touch_mean_net",
        "label_u_policy_sum_net",
        "delta_sum_ft_vs_u_policy",
        "first_touch_bar_p90",
        "policy_bar_p90",
        "bad_full_path_mae_3r_rate",
        "full_path_mae_to_sl_p90",
        "dirty_reversal_rate",
    ]
    archetype_cols = [
        "month",
        "archetype",
        "rows",
        "row_share",
        "first_touch_sum_net",
        "first_touch_mean_net",
        "first_touch_bar_p90",
        "policy_bar_p50",
        "bad_full_path_mae_3r_rate",
        "full_path_mae_to_sl_p90",
    ]
    speed_cols = [
        "month",
        "first_touch_speed_bucket",
        "rows",
        "row_share",
        "first_touch_sum_net",
        "first_touch_mean_net",
        "dirty_reversal_rate",
        "bad_full_path_mae_3r_rate",
        "full_path_mae_to_sl_p90",
        "extra_bars_after_first_touch_p90",
    ]
    symbol_cols = [
        "month",
        "__symbol__",
        "rows",
        "first_touch_sum_net",
        "dirty_reversal_rate",
        "bad_full_path_mae_3r_rate",
        "full_path_mae_to_sl_p90",
    ]
    lines = [
        "# Stage172 Stage167 First-Touch Exit Diagnostic",
        "",
        "Scope: report-only execution diagnostic on the Stage167 selected-row ledger. No model is trained here.",
        "",
        f"Selected ledger: `{manifest['ledger_csv']}`",
        f"Labels: `{manifest['labels_path']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Selected rows: `{manifest['rows']}`",
        f"Max abs diff first_touch_net vs __u_policy_net__: `{manifest['max_abs_diff_ft_vs_u_policy']:.12g}`",
        f"Max abs diff first_touch_net vs __r_policy_net__: `{manifest['max_abs_diff_ft_vs_r_policy']:.12g}`",
        f"Max abs diff first_touch_net vs __y_ret__: `{manifest['max_abs_diff_ft_vs_y_ret']:.12g}`",
        "",
        "Interpretation note: for this Stage167 label artifact, `__u_policy_net__`, `__r_policy_net__`, `__y_ret__`, and `__first_touch_capture_net__` collapse to first-touch net on the selected rows. The artifact therefore does not contain a separate longer-hold/trailing PnL comparison; longer-path risk is visible through full-path MAE/MFE and bar-duration fields.",
        "",
        "## Monthly Execution Summary",
        "",
        _table(monthly, monthly_cols, limit=80),
        "",
        "## Weekly Execution Summary",
        "",
        _table(weekly, weekly_cols, limit=160),
        "",
        "## Monthly Path Archetypes",
        "",
        _table(archetypes, archetype_cols, limit=120),
        "",
        "## First-Touch Speed Buckets",
        "",
        _table(speed, speed_cols, limit=120),
        "",
        "## Repeated Symbols",
        "",
        _table(symbols, symbol_cols, limit=120),
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    ledger_csv: Path,
    labels_path: Path,
    output_dir: Path,
    months: list[str],
    first_touch_clean_r: float,
    full_path_clean_r: float,
    full_path_dirty_r: float,
    min_symbol_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(ledger_csv)
    labels = _load_label_subset(labels_path)
    joined = _join_ledger_labels(ledger, labels)
    if months:
        joined = joined[joined["month"].isin(months)].copy()
    joined = _add_archetypes(
        joined,
        first_touch_clean_r=first_touch_clean_r,
        full_path_clean_r=full_path_clean_r,
        full_path_dirty_r=full_path_dirty_r,
    )

    monthly = _summarize_group(joined, "month")
    weekly = _summarize_group(joined, "week")
    archetypes = _archetype_summary(joined)
    speed = _speed_summary(joined)
    symbols = _symbol_summary(joined, min_rows=min_symbol_rows)

    paths = {
        "joined": output_dir / "stage172_joined_selected_ledger.csv",
        "monthly": output_dir / "stage172_monthly_execution_summary.csv",
        "weekly": output_dir / "stage172_weekly_execution_summary.csv",
        "archetypes": output_dir / "stage172_monthly_archetypes.csv",
        "speed": output_dir / "stage172_first_touch_speed_buckets.csv",
        "symbols": output_dir / "stage172_repeated_symbols.csv",
        "manifest": output_dir / "manifest.json",
    }
    joined.to_csv(paths["joined"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    archetypes.to_csv(paths["archetypes"], index=False)
    speed.to_csv(paths["speed"], index=False)
    symbols.to_csv(paths["symbols"], index=False)

    ft = _safe_numeric(joined.get("first_touch_net"), index=joined.index)
    u_net = _safe_numeric(joined.get("__u_policy_net__"), index=joined.index)
    r_net = _safe_numeric(joined.get("__r_policy_net__"), index=joined.index)
    y_ret = _safe_numeric(joined.get("__y_ret__"), index=joined.index)
    capture = _safe_numeric(joined.get("__first_touch_capture_net__"), index=joined.index)
    manifest = {
        "scope": "stage172_stage167_firsttouch_exit_diagnostic",
        "ledger_csv": str(ledger_csv),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "months": list(months),
        "rows": int(len(joined)),
        "first_touch_clean_r": float(first_touch_clean_r),
        "full_path_clean_r": float(full_path_clean_r),
        "full_path_dirty_r": float(full_path_dirty_r),
        "max_abs_diff_ft_vs_u_policy": float((ft - u_net).abs().max()) if len(joined) else float("nan"),
        "max_abs_diff_ft_vs_r_policy": float((ft - r_net).abs().max()) if len(joined) else float("nan"),
        "max_abs_diff_ft_vs_y_ret": float((ft - y_ret).abs().max()) if len(joined) else float("nan"),
        "max_abs_diff_ft_vs_first_touch_capture": float((ft - capture).abs().max()) if len(joined) else float("nan"),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        manifest=manifest,
        monthly=monthly,
        weekly=weekly,
        archetypes=archetypes,
        speed=speed,
        symbols=symbols,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--first-touch-clean-r", type=float, default=1.0)
    parser.add_argument("--full-path-clean-r", type=float, default=1.0)
    parser.add_argument("--full-path-dirty-r", type=float, default=3.0)
    parser.add_argument("--min-symbol-rows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_diagnostic(
        ledger_csv=args.ledger_csv,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        first_touch_clean_r=args.first_touch_clean_r,
        full_path_clean_r=args.full_path_clean_r,
        full_path_dirty_r=args.full_path_dirty_r,
        min_symbol_rows=args.min_symbol_rows,
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
