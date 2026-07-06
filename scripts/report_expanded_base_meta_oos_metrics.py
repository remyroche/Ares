#!/usr/bin/env python3
"""Expanded OOS metrics for base/meta ledgers and overlap comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOP_FRACS = (0.05, 0.10, 0.20, 0.30)
DEFAULT_META_SELECTORS = (
    "score_meta_long_aware_clean_minus_risk",
    "score_meta_exec_margin_risk_blend",
    "score_meta_clean_minus_risk",
    "score_meta_exec_margin",
    "score_base",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(float(default), index=frame.index, dtype=np.float64)


def _mean(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(ser.mean()) if len(ser) else float("nan")


def _rate(values: Any) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(ser.clip(0.0, 1.0).mean()) if len(ser) else float("nan")


def _q(values: Any, quantile: float) -> float:
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(ser.quantile(float(quantile))) if len(ser) else float("nan")


def _add_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts_col = "__ts__" if "__ts__" in out.columns else "timestamp"
    ts = pd.to_datetime(out.get(ts_col), utc=True, errors="coerce")
    out["__ts_metric"] = ts
    if "calendar_month" not in out.columns:
        out["calendar_month"] = ts.dt.strftime("%Y-%m")
    out["week_start"] = ts.dt.tz_localize(None).dt.to_period("W-MON").dt.start_time.astype(str)
    return out


def _side(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        side = frame["side_name"].astype(str).str.lower()
        return side.where(side.isin(["long", "short"]), "unknown")
    if "__side__" in frame.columns:
        vals = _num(frame, "__side__", 0.0)
    else:
        vals = _num(frame, "side", 0.0)
    return pd.Series(np.where(vals.to_numpy(dtype=float) < 0.0, "short", "long"), index=frame.index)


def _archetype(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
        "source_semantic_family",
        "long_source_regime_split",
        "label_archetype",
        "archetype",
        "side_aegmm_cluster",
        "aegmm_cluster",
    ]
    for col in candidates:
        if col in frame.columns:
            ser = frame[col].astype(str).replace({"nan": "", "None": ""})
            if ser.str.len().gt(0).any():
                return ser.where(ser.str.len().gt(0), "missing")
    return pd.Series("missing", index=frame.index, dtype="object")


def _selected_by_score(frame: pd.DataFrame, score_col: str, frac: float) -> pd.Series:
    score = _num(frame, score_col)
    out = pd.Series(False, index=frame.index)
    valid = score.dropna()
    if valid.empty:
        return out
    keep = max(1, int(math.ceil(len(valid) * float(frac))))
    out.loc[valid.sort_values(ascending=False).head(keep).index] = True
    return out


def _selection_masks(frame: pd.DataFrame, layer: str, selectors: list[str]) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    if layer == "base":
        for frac in TOP_FRACS:
            col = f"selected_top{int(round(frac * 100)):02d}"
            alt = f"selected_top{int(round(frac * 100))}"
            if col in frame.columns:
                masks[f"base_{col}"] = frame[col].astype(bool)
            elif alt in frame.columns:
                masks[f"base_{alt}"] = frame[alt].astype(bool)
            elif "score" in frame.columns:
                masks[f"base_score_top{int(round(frac * 100)):02d}"] = _selected_by_score(frame, "score", frac)
    else:
        for selector in selectors:
            if selector not in frame.columns:
                continue
            for frac in TOP_FRACS:
                masks[f"{selector}_top{int(round(frac * 100)):02d}"] = _selected_by_score(frame, selector, frac)
    return masks


def _metric_row(frame: pd.DataFrame, mask: pd.Series, *, version: str, layer: str, selector: str) -> dict[str, Any]:
    selected = frame.loc[mask].copy()
    side = _side(selected) if len(selected) else pd.Series(dtype="object")
    return {
        "version": version,
        "layer": layer,
        "selector": selector,
        "rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selected_symbols": int(selected["__symbol__"].nunique(dropna=True)) if "__symbol__" in selected.columns else 0,
        "months": int(frame["calendar_month"].nunique(dropna=True)) if "calendar_month" in frame.columns else 0,
        "mean_ret_net": _mean(selected.get("ret_net", pd.Series(dtype=float))),
        "mean_u_policy_net": _mean(selected.get("u_policy_net", pd.Series(dtype=float))),
        "mean_ev_after_1pct": _mean(selected.get("ev_after_1pct", pd.Series(dtype=float))),
        "mean_exec_margin": _mean(selected.get("exec_margin", pd.Series(dtype=float))),
        "clean_exec_rate": _rate(selected.get("clean_exec", selected.get("target_hard", pd.Series(dtype=float)))),
        "dirty_positive_rate": _rate(selected.get("dirty_positive", pd.Series(dtype=float))),
        "first_touch_bad_mae_rate": _rate(selected.get("first_touch_bad_mae_1r", selected.get("first_touch_stop", pd.Series(dtype=float)))),
        "full_path_bad_mae_rate": _rate(selected.get("full_path_bad_mae_1r", selected.get("bad_mae_1r", pd.Series(dtype=float)))),
        "timeout_rate": _rate(selected.get("timeout", selected.get("first_touch_timeout", pd.Series(dtype=float)))),
        "stop_rate": _rate(selected.get("first_touch_stop", selected.get("full_stop_loss", pd.Series(dtype=float)))),
        "q10_ret_net": _q(selected.get("ret_net", pd.Series(dtype=float)), 0.10),
        "long_share": float(side.eq("long").mean()) if len(side) else float("nan"),
        "short_share": float(side.eq("short").mean()) if len(side) else float("nan"),
    }


def _summary(frame: pd.DataFrame, *, version: str, layer: str, selectors: list[str]) -> pd.DataFrame:
    frame = _add_time_columns(frame)
    masks = _selection_masks(frame, layer, selectors)
    return pd.DataFrame(
        [_metric_row(frame, mask, version=version, layer=layer, selector=name) for name, mask in masks.items()]
    )


def _group_metrics(frame: pd.DataFrame, *, version: str, layer: str, selectors: list[str], group_cols: list[str]) -> pd.DataFrame:
    frame = _add_time_columns(frame)
    frame["side_name_metric"] = _side(frame)
    frame["archetype_metric"] = _archetype(frame)
    masks = _selection_masks(frame, layer, selectors)
    rows: list[dict[str, Any]] = []
    for selector, mask in masks.items():
        selected = frame.loc[mask].copy()
        if selected.empty:
            continue
        for keys, group in selected.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: val for col, val in zip(group_cols, keys)}
            row.update(
                _metric_row(
                    group,
                    pd.Series(True, index=group.index),
                    version=version,
                    layer=layer,
                    selector=selector,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _load(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _overlap(new: pd.DataFrame, old: pd.DataFrame, *, new_name: str, old_name: str, layer: str, selectors: list[str]) -> pd.DataFrame:
    if new.empty or old.empty:
        return pd.DataFrame()
    new = _add_time_columns(new)
    old = _add_time_columns(old)
    months = sorted(set(new["calendar_month"].dropna().astype(str)) & set(old["calendar_month"].dropna().astype(str)))
    if not months:
        return pd.DataFrame()
    new_summary = _summary(new[new["calendar_month"].astype(str).isin(months)], version=new_name, layer=layer, selectors=selectors)
    old_summary = _summary(old[old["calendar_month"].astype(str).isin(months)], version=old_name, layer=layer, selectors=selectors)
    out = pd.concat([new_summary, old_summary], ignore_index=True)
    out["comparison_scope"] = "overlap_months"
    out["overlap_months"] = ",".join(months)
    return out


def _new_months(new: pd.DataFrame, old: pd.DataFrame, *, version: str, layer: str, selectors: list[str]) -> pd.DataFrame:
    if new.empty:
        return pd.DataFrame()
    new = _add_time_columns(new)
    old_months = set(_add_time_columns(old)["calendar_month"].dropna().astype(str)) if not old.empty else set()
    months = sorted(set(new["calendar_month"].dropna().astype(str)) - old_months)
    if not months:
        return pd.DataFrame()
    out = _summary(new[new["calendar_month"].astype(str).isin(months)], version=version, layer=layer, selectors=selectors)
    out["comparison_scope"] = "newly_available_months"
    out["new_months"] = ",".join(months)
    return out


def _write_markdown(out_dir: Path, manifest: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Expanded Base/Meta OOS Metrics", ""]
    lines.append("## Manifest")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    lines.append("```")
    for name, table in tables.items():
        lines.extend(["", f"## {name.replace('_', ' ').title()}", ""])
        if table.empty:
            lines.append("No rows.")
        else:
            lines.append(table.head(80).to_markdown(index=False))
    (out_dir / "expanded_base_meta_oos_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ledger", type=Path, required=True)
    parser.add_argument("--meta-predictions", type=Path, default=None)
    parser.add_argument("--old-base-ledger", type=Path, default=None)
    parser.add_argument("--old-meta-predictions", type=Path, default=None)
    parser.add_argument("--new-version", default="new")
    parser.add_argument("--old-version", default="old")
    parser.add_argument("--meta-selectors", default=",".join(DEFAULT_META_SELECTORS))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selectors = [part.strip() for part in str(args.meta_selectors).split(",") if part.strip()]
    base = _load(args.base_ledger)
    meta = _load(args.meta_predictions)
    old_base = _load(args.old_base_ledger)
    old_meta = _load(args.old_meta_predictions)
    tables = {
        "base_summary": _summary(base, version=args.new_version, layer="base", selectors=[]),
        "meta_summary": _summary(meta, version=args.new_version, layer="meta", selectors=selectors) if not meta.empty else pd.DataFrame(),
        "base_overlap_comparison": _overlap(base, old_base, new_name=args.new_version, old_name=args.old_version, layer="base", selectors=[]),
        "meta_overlap_comparison": _overlap(meta, old_meta, new_name=args.new_version, old_name=args.old_version, layer="meta", selectors=selectors) if not meta.empty else pd.DataFrame(),
        "base_new_months": _new_months(base, old_base, version=args.new_version, layer="base", selectors=[]),
        "meta_new_months": _new_months(meta, old_meta, version=args.new_version, layer="meta", selectors=selectors) if not meta.empty else pd.DataFrame(),
        "base_week_archetype": _group_metrics(
            base,
            version=args.new_version,
            layer="base",
            selectors=[],
            group_cols=["week_start", "side_name_metric", "archetype_metric"],
        ),
        "base_month_archetype": _group_metrics(
            base,
            version=args.new_version,
            layer="base",
            selectors=[],
            group_cols=["calendar_month", "side_name_metric", "archetype_metric"],
        ),
        "meta_week_archetype": _group_metrics(
            meta,
            version=args.new_version,
            layer="meta",
            selectors=selectors,
            group_cols=["week_start", "side_name_metric", "archetype_metric"],
        )
        if not meta.empty
        else pd.DataFrame(),
        "meta_month_archetype": _group_metrics(
            meta,
            version=args.new_version,
            layer="meta",
            selectors=selectors,
            group_cols=["calendar_month", "side_name_metric", "archetype_metric"],
        )
        if not meta.empty
        else pd.DataFrame(),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    manifest = {
        "base_ledger": str(args.base_ledger),
        "meta_predictions": str(args.meta_predictions) if args.meta_predictions else None,
        "old_base_ledger": str(args.old_base_ledger) if args.old_base_ledger else None,
        "old_meta_predictions": str(args.old_meta_predictions) if args.old_meta_predictions else None,
        "new_version": str(args.new_version),
        "old_version": str(args.old_version),
        "meta_selectors": selectors,
        "outputs": {name: str(out_dir / f"{name}.csv") for name in tables},
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(out_dir, manifest, tables)
    print(json.dumps(_json_safe({"event": "expanded_oos_metrics_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
