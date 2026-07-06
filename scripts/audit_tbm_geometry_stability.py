#!/usr/bin/env python3
"""Audit TBM geometry stability by side, spread, barrier, and holding time.

This is a label-design diagnostic. It does not train a selector. It asks which
parts of the current executable label geometry remain stable in June and which
parts should be redesigned or excluded before the next base/meta run.
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

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_LABELS_DIR,
    _load_labels,
    _path_metrics,
    _safe_mean,
)
from scripts.run_label_feature_store_model_smoke import _apply_spread_symbol_universe  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/tbm_geometry_stability_audit_v1")
DEFAULT_TRAIN_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_min(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.min()) if len(vals) else float("nan")


def _safe_max(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.max()) if len(vals) else float("nan")


def _safe_quantile(values: pd.Series, q: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.quantile(float(q))) if len(vals) else float("nan")


def _bucket_quantile(values: pd.Series, prefix: str, q: int = 5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(f"{prefix}_unknown", index=values.index, dtype=object)
    valid = numeric.notna()
    if int(valid.sum()) < q * 20:
        return out
    try:
        codes = pd.qcut(
            numeric.loc[valid].rank(method="first"),
            q=q,
            labels=False,
            duplicates="drop",
        )
        out.loc[valid] = [f"{prefix}_q{int(code)}" for code in codes.astype(int)]
    except ValueError:
        pass
    return out


def _bars_bucket(values: pd.Series) -> pd.Series:
    bars = pd.to_numeric(values, errors="coerce")
    out = pd.Series("bars_unknown", index=values.index, dtype=object)
    out.loc[bars.le(3.0)] = "bars_00_03"
    out.loc[bars.gt(3.0) & bars.le(8.0)] = "bars_04_08"
    out.loc[bars.gt(8.0) & bars.le(14.0)] = "bars_09_14"
    out.loc[bars.gt(14.0)] = "bars_15_24"
    return out


def _exec_margin_clean(metrics: pd.DataFrame) -> pd.Series:
    index = metrics.index
    u = pd.to_numeric(metrics["u_policy_net"], errors="coerce").fillna(0.0)
    mae_norm = pd.to_numeric(metrics["mae_norm"], errors="coerce").fillna(10.0)
    mfe_norm = pd.to_numeric(metrics["mfe_norm"], errors="coerce").fillna(0.0)
    bars_to_mfe = pd.to_numeric(metrics["bars_to_mfe"], errors="coerce").fillna(10_000.0)
    timeout = pd.to_numeric(metrics["is_timeout"], errors="coerce").fillna(1.0).gt(0.5)
    barrier = pd.to_numeric(metrics.get("barrier", pd.Series(0.02, index=index)), errors="coerce").fillna(0.02)
    mfe_mae = (mfe_norm / mae_norm.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    exec_margin = (
        u
        - 0.0040 * (mae_norm - 0.65).clip(lower=0.0)
        - 0.0050 * mae_norm.ge(1.0).astype(float)
        - 0.0060 * timeout.astype(float)
        - 0.0010 * np.log1p(bars_to_mfe.clip(lower=0.0))
        - 0.75 * (barrier - 0.020).clip(lower=0.0)
        + 0.0015 * (mfe_mae - 1.25).clip(lower=0.0, upper=2.0)
    )
    return (
        u.gt(0.0)
        & exec_margin.gt(0.0005)
        & mae_norm.le(0.85)
        & (~timeout)
        & mfe_norm.ge(1.0)
        & mfe_mae.ge(1.25)
        & bars_to_mfe.le(14.0)
    ).fillna(False)


def _attach_spread(frame: pd.DataFrame, symbol_universe: pd.DataFrame | None, spread_rank_column: str) -> pd.Series:
    if symbol_universe is None or symbol_universe.empty:
        return pd.Series(np.nan, index=frame.index, dtype=np.float32)
    symbol_cols = [c for c in ("symbol", "__symbol__", "asset") if c in symbol_universe.columns]
    if not symbol_cols or spread_rank_column not in symbol_universe.columns:
        return pd.Series(np.nan, index=frame.index, dtype=np.float32)
    mapping = (
        symbol_universe[[symbol_cols[0], spread_rank_column]]
        .dropna(subset=[symbol_cols[0]])
        .drop_duplicates(subset=[symbol_cols[0]], keep="first")
        .set_index(symbol_cols[0])[spread_rank_column]
    )
    return frame["__symbol__"].map(mapping).astype(np.float32)


def _summarise_group(group: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    positive = group["positive_u"].astype(bool)
    clean = group["exec_margin_clean"].astype(bool)
    return {
        f"{prefix}_rows": int(len(group)),
        f"{prefix}_positive_rows": int(positive.sum()),
        f"{prefix}_positive_rate": _safe_mean(positive.astype(float)),
        f"{prefix}_clean_rate_all": _safe_mean(clean.astype(float)),
        f"{prefix}_clean_rate_positive": _safe_mean(clean.loc[positive].astype(float)) if int(positive.sum()) else float("nan"),
        f"{prefix}_mean_u": _safe_mean(group["u_policy_net"]),
        f"{prefix}_positive_mean_u": _safe_mean(group.loc[positive, "u_policy_net"]) if int(positive.sum()) else float("nan"),
        f"{prefix}_bad_mae_rate": _safe_mean(group["bad_mae_1r"]),
        f"{prefix}_timeout_rate": _safe_mean(group["is_timeout_float"]),
        f"{prefix}_barrier_median": _safe_quantile(group["barrier"], 0.50),
        f"{prefix}_barrier_p75": _safe_quantile(group["barrier"], 0.75),
        f"{prefix}_bars_median": _safe_quantile(group["bars_policy"], 0.50),
        f"{prefix}_bars_p75": _safe_quantile(group["bars_policy"], 0.75),
    }


def _status(row: pd.Series, min_positive_rows: int) -> str:
    holdout_pos = float(row.get("holdout_positive_rows", 0.0) or 0.0)
    train_clean = float(row.get("train_clean_rate_positive", float("nan")))
    holdout_clean = float(row.get("holdout_clean_rate_positive", float("nan")))
    holdout_bad = float(row.get("holdout_bad_mae_rate", float("nan")))
    holdout_timeout = float(row.get("holdout_timeout_rate", float("nan")))
    if holdout_pos < float(min_positive_rows):
        return "too_few_holdout_positive_rows"
    if math.isfinite(holdout_clean) and math.isfinite(train_clean) and holdout_clean + 0.10 < train_clean:
        return "unstable_clean_rate_decay"
    if math.isfinite(holdout_clean) and holdout_clean < 0.30:
        return "low_holdout_clean_rate"
    if math.isfinite(holdout_bad) and holdout_bad > 0.70:
        return "excess_bad_mae"
    if math.isfinite(holdout_timeout) and holdout_timeout > 0.15:
        return "excess_timeout"
    return "candidate_stable"


def run_audit(
    *,
    labels_path: Path,
    output_dir: Path,
    train_months: list[str],
    holdout_month: str,
    spread_baseline_path: Path | None,
    spread_rank_column: str,
    target_symbol_count: int | None,
    max_spread_bps: float | None,
    min_positive_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    frame, symbol_universe_filter, symbol_universe = _apply_spread_symbol_universe(
        frame,
        spread_baseline_path=spread_baseline_path,
        spread_rank_column=spread_rank_column,
        target_symbol_count=target_symbol_count,
        max_spread_bps=max_spread_bps,
    )
    metrics = _path_metrics(frame).reset_index(drop=True)
    data = pd.concat(
        [frame[["__ts__", "__symbol__"]].reset_index(drop=True), metrics],
        axis=1,
        copy=False,
    )
    data["month"] = data["__ts__"].dt.to_period("M").astype(str)
    data["side_name"] = np.where(pd.to_numeric(data["side"], errors="coerce").fillna(1.0).ge(0.0), "long", "short")
    data["spread_bps"] = _attach_spread(frame.reset_index(drop=True), symbol_universe, spread_rank_column).reset_index(drop=True)
    data["spread_bucket"] = _bucket_quantile(data["spread_bps"], "spread")
    data["barrier_bucket"] = _bucket_quantile(data["barrier"], "barrier")
    data["bars_bucket"] = _bars_bucket(data["bars_policy"])
    data["positive_u"] = pd.to_numeric(data["u_policy_net"], errors="coerce").fillna(0.0).gt(0.0)
    data["bad_mae_1r"] = pd.to_numeric(data["mae_norm"], errors="coerce").fillna(10.0).ge(1.0).astype(float)
    data["is_timeout_float"] = pd.to_numeric(data["is_timeout"], errors="coerce").fillna(1.0).gt(0.5).astype(float)
    data["exec_margin_clean"] = _exec_margin_clean(metrics)

    monthly_rows = []
    for keys in (
        ["month", "side_name"],
        ["month", "side_name", "spread_bucket"],
        ["month", "side_name", "barrier_bucket"],
        ["month", "side_name", "bars_bucket"],
        ["month", "side_name", "spread_bucket", "barrier_bucket"],
    ):
        for key, group in data.groupby(keys, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(keys, key))
            row["slice_type"] = " x ".join(keys[1:])
            row.update(_summarise_group(group, prefix="month"))
            monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)

    stability_rows = []
    for keys in (
        ["side_name"],
        ["side_name", "spread_bucket"],
        ["side_name", "barrier_bucket"],
        ["side_name", "bars_bucket"],
        ["side_name", "spread_bucket", "barrier_bucket"],
    ):
        train = data[data["month"].isin(train_months)].copy()
        holdout = data[data["month"].eq(holdout_month)].copy()
        train_groups = {key: group for key, group in train.groupby(keys, dropna=False)}
        holdout_groups = {key: group for key, group in holdout.groupby(keys, dropna=False)}
        all_keys = sorted(set(train_groups) | set(holdout_groups), key=lambda x: str(x))
        for key in all_keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = dict(zip(keys, key_tuple))
            row["slice_type"] = " x ".join(keys)
            row.update(_summarise_group(train_groups.get(key, train.iloc[0:0]), prefix="train"))
            row.update(_summarise_group(holdout_groups.get(key, holdout.iloc[0:0]), prefix="holdout"))
            row["clean_rate_positive_delta"] = (
                float(row["holdout_clean_rate_positive"]) - float(row["train_clean_rate_positive"])
                if math.isfinite(float(row["holdout_clean_rate_positive"])) and math.isfinite(float(row["train_clean_rate_positive"]))
                else float("nan")
            )
            row["mean_u_delta"] = (
                float(row["holdout_mean_u"]) - float(row["train_mean_u"])
                if math.isfinite(float(row["holdout_mean_u"])) and math.isfinite(float(row["train_mean_u"]))
                else float("nan")
            )
            row["status"] = _status(pd.Series(row), min_positive_rows=min_positive_rows)
            stability_rows.append(row)
    stability = pd.DataFrame(stability_rows)

    paths = {
        "monthly": output_dir / "tbm_geometry_monthly.csv",
        "stability": output_dir / "tbm_geometry_stability.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "tbm_geometry_stability.md",
    }
    monthly.to_csv(paths["monthly"], index=False)
    stability.to_csv(paths["stability"], index=False)
    manifest = {
        "scope": "tbm_geometry_side_spread_barrier_stability",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(data)),
        "symbols": int(data["__symbol__"].nunique(dropna=True)),
        "timestamp_min": data["__ts__"].min(),
        "timestamp_max": data["__ts__"].max(),
        "train_months": train_months,
        "holdout_month": holdout_month,
        "min_positive_rows": int(min_positive_rows),
        "symbol_universe_filter": symbol_universe_filter,
        "status_counts": stability["status"].value_counts(dropna=False).to_dict() if not stability.empty else {},
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_markdown(paths["markdown"], manifest, stability)
    return manifest


def _write_markdown(path: Path, manifest: dict[str, Any], stability: pd.DataFrame) -> None:
    lines = [
        "# TBM Geometry Stability Audit",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Train months: `{', '.join(manifest['train_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        "",
        "## Status Counts",
        "",
        pd.Series(manifest["status_counts"]).rename("count").to_frame().to_markdown(),
        "",
        "## Worst Holdout Slices",
        "",
    ]
    if stability.empty:
        lines.append("No stability rows produced.")
    else:
        cols = [
            "slice_type",
            "side_name",
            "spread_bucket",
            "barrier_bucket",
            "bars_bucket",
            "train_positive_rows",
            "holdout_positive_rows",
            "train_clean_rate_positive",
            "holdout_clean_rate_positive",
            "clean_rate_positive_delta",
            "holdout_bad_mae_rate",
            "holdout_timeout_rate",
            "status",
        ]
        display = stability[[col for col in cols if col in stability.columns]].copy()
        display = display.sort_values(
            ["status", "clean_rate_positive_delta", "holdout_clean_rate_positive"],
            ascending=[True, True, True],
            kind="mergesort",
        ).head(80)
        lines.append(display.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-months", type=lambda value: _parse_csv(value, DEFAULT_TRAIN_MONTHS), default=",".join(DEFAULT_TRAIN_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--spread-baseline-path", type=Path, default=None)
    parser.add_argument("--spread-rank-column", default="p75_spread_bps")
    parser.add_argument("--target-symbol-count", type=int, default=None)
    parser.add_argument("--max-spread-bps", type=float, default=None)
    parser.add_argument("--min-positive-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_audit(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        train_months=list(args.train_months),
        holdout_month=str(args.holdout_month),
        spread_baseline_path=args.spread_baseline_path,
        spread_rank_column=str(args.spread_rank_column),
        target_symbol_count=args.target_symbol_count,
        max_spread_bps=args.max_spread_bps,
        min_positive_rows=int(args.min_positive_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
