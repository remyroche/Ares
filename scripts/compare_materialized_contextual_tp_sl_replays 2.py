#!/usr/bin/env python3
"""Compare materialized contextual TP/SL combo replays.

Inputs are directories produced by `materialize_contextual_tp_sl_combo_replay.py`.
The output is a small audit package that compares replay metrics globally,
per head, per week, per month, and per head-week against a chosen baseline
label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_manifest(label: str, path: Path) -> Dict[str, Any]:
    manifest_path = path / "combo_replay_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for {label}: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["_label"] = label
    data["_path"] = str(path)
    return data


def _head_from_strategy(strategy_id: pd.Series) -> pd.Series:
    text = strategy_id.astype(str)
    return text.str.extract(r"^(short_bollinger|long_bars|long_dist|short_asset)", expand=False)


def _accepted_decisions(path: Path) -> pd.DataFrame:
    decisions_path = path / "combo_replay_decisions.parquet"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing replay decisions: {decisions_path}")
    df = pd.read_parquet(decisions_path)
    if df.empty or "accepted" not in df.columns:
        return pd.DataFrame()
    out = df.loc[df["accepted"].astype(bool)].copy()
    if out.empty:
        return out
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["week"] = ts.dt.to_period("W").astype(str)
    out["month"] = ts.dt.to_period("M").astype(str)
    out["day"] = ts.dt.date.astype(str)
    out["head"] = _head_from_strategy(out["strategy_id"])
    size = pd.to_numeric(out.get("position_size", 0.0), errors="coerce").fillna(0.0)
    net = pd.to_numeric(out.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(out.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    out["net_pnl_amount"] = size * net
    out["gross_pnl_amount"] = size * gross
    out["is_win"] = net > 0.0
    exit_reason = (
        out["position_exit_reason"]
        if "position_exit_reason" in out.columns
        else pd.Series("", index=out.index)
    )
    out["is_full_sl"] = exit_reason.astype(str).str.contains("full_sl", case=False, na=False)
    out["is_timeout"] = exit_reason.astype(str).str.contains("timeout", case=False, na=False)
    return out


def _global_rows(items: List[Tuple[str, Path, Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, path, manifest in items:
        metrics = manifest.get("metrics", {})
        rows.append(
            {
                "label": label,
                "path": str(path),
                "combo_id": manifest.get("combo_id"),
                "candidate_start": manifest.get("candidate_start"),
                "candidate_end": manifest.get("candidate_end"),
                "candidate_rows": manifest.get("candidate_rows"),
                "start_filter": manifest.get("start_filter"),
                "end_filter": manifest.get("end_filter"),
                "net_pnl": metrics.get("net_pnl"),
                "gross_pnl": metrics.get("gross_pnl"),
                "trade_count": metrics.get("trade_count"),
                "full_sl_rate": metrics.get("full_sl_rate"),
                "timeout_rate": metrics.get("timeout_rate"),
                "max_drawdown": metrics.get("max_drawdown"),
                "strategy_concentration": metrics.get("strategy_concentration"),
                "side_concentration": metrics.get("side_concentration"),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_accepted(label: str, accepted: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=["label", *group_cols])
    out = (
        accepted.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            net_pnl=("net_pnl_amount", "sum"),
            gross_pnl=("gross_pnl_amount", "sum"),
            trades=("accepted", "size"),
            hit_rate=("is_win", "mean"),
            full_sl_rate=("is_full_sl", "mean"),
            timeout_rate=("is_timeout", "mean"),
        )
        .sort_values(group_cols)
    )
    if "label" not in group_cols:
        out.insert(0, "label", label)
    return out


def _add_deltas(frame: pd.DataFrame, baseline_label: str, keys: List[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    metric_cols = [
        c
        for c in frame.columns
        if c not in {"label", "path", "combo_id", "candidate_start", "candidate_end", "start_filter", "end_filter", *keys}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    base = frame.loc[frame["label"].eq(baseline_label), [*keys, *metric_cols]].copy()
    if base.empty:
        return frame
    if keys:
        base = base.rename(columns={c: f"{c}_baseline" for c in metric_cols})
        out = frame.merge(base, on=keys, how="left")
    else:
        base_row = base.iloc[0]
        out = frame.copy()
        for c in metric_cols:
            out[f"{c}_baseline"] = base_row[c]
    for c in metric_cols:
        out[f"delta_{c}"] = out[c] - out[f"{c}_baseline"]
    return out


def _write_report(
    out_dir: Path,
    baseline_label: str,
    global_df: pd.DataFrame,
    head_df: pd.DataFrame,
    week_df: pd.DataFrame,
    month_df: pd.DataFrame,
    head_week_df: pd.DataFrame,
) -> None:
    lines = [
        "# Materialized Contextual TP/SL Replay Comparison",
        "",
        f"Baseline label: `{baseline_label}`",
        "Costs are included through the materialized portfolio replay outputs.",
        "",
        "## Global",
        "",
        global_df.to_markdown(index=False),
        "",
        "## Per Head",
        "",
        head_df.to_markdown(index=False),
        "",
        "## Per Week",
        "",
        week_df.to_markdown(index=False),
        "",
        "## Per Month",
        "",
        month_df.to_markdown(index=False),
        "",
        "## Per Head Week",
        "",
        head_week_df.to_markdown(index=False),
    ]
    (out_dir / "materialized_replay_comparison_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="static")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Label=path to a materialized combo replay directory. Repeatable.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items: List[Tuple[str, Path, Dict[str, Any]]] = []
    for raw in args.run:
        if "=" not in raw:
            raise ValueError(f"Invalid --run {raw!r}; expected label=path")
        label, path_s = raw.split("=", 1)
        path = Path(path_s)
        items.append((label, path, _load_manifest(label, path)))

    global_df = _global_rows(items)
    global_df = _add_deltas(global_df, str(args.baseline_label), keys=[])

    accepted_frames = []
    for label, path, _manifest in items:
        acc = _accepted_decisions(path)
        if not acc.empty:
            acc.insert(0, "label", label)
        accepted_frames.append(acc)
    accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    head_df = _aggregate_accepted("", accepted_all, ["label", "head"])
    week_df = _aggregate_accepted("", accepted_all, ["label", "week"])
    month_df = _aggregate_accepted("", accepted_all, ["label", "month"])
    head_week_df = _aggregate_accepted("", accepted_all, ["label", "head", "week"])

    head_df = _add_deltas(head_df, str(args.baseline_label), keys=["head"])
    week_df = _add_deltas(week_df, str(args.baseline_label), keys=["week"])
    month_df = _add_deltas(month_df, str(args.baseline_label), keys=["month"])
    head_week_df = _add_deltas(head_week_df, str(args.baseline_label), keys=["head", "week"])

    global_df.to_csv(args.out_dir / "materialized_replay_global_comparison.csv", index=False)
    head_df.to_csv(args.out_dir / "materialized_replay_head_comparison.csv", index=False)
    week_df.to_csv(args.out_dir / "materialized_replay_week_comparison.csv", index=False)
    month_df.to_csv(args.out_dir / "materialized_replay_month_comparison.csv", index=False)
    head_week_df.to_csv(args.out_dir / "materialized_replay_head_week_comparison.csv", index=False)
    payload = {
        "baseline_label": str(args.baseline_label),
        "runs": [{"label": label, "path": str(path)} for label, path, _ in items],
    }
    (args.out_dir / "materialized_replay_comparison_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    _write_report(args.out_dir, str(args.baseline_label), global_df, head_df, week_df, month_df, head_week_df)
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "runs": len(items),
                    "baseline_label": str(args.baseline_label),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
