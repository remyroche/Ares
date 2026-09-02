#!/usr/bin/env python3
"""Replay contextual TP/SL combos with head-specific portfolio overlays.

This is intentionally narrow: it starts from already-materialized candidate
tables and tests portfolio-level overlays that the replay engine already
understands through per-row columns. It does not create new TP/SL outcomes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    ARMS,
    _accepted_period_tables,
    _arm_combinations,
    _combo_id,
    _concat_nonempty,
    _head_name,
    _json_safe,
    _load_arm_tables,
    _load_requested_combo_ids,
    _period_metrics,
)


HEADS = ("long_bars", "long_dist", "short_asset", "short_bollinger")


def _parse_overlay_spec(text: str) -> Dict[str, Dict[str, Any]]:
    """Parse `head:key=value,key=value;head:key=value` overlay specs."""
    spec: Dict[str, Dict[str, Any]] = {}
    text = str(text or "").strip()
    if not text or text.lower() in {"none", "baseline"}:
        return spec
    for raw_head_spec in text.split(";"):
        raw_head_spec = raw_head_spec.strip()
        if not raw_head_spec:
            continue
        if ":" not in raw_head_spec:
            raise ValueError(f"Invalid overlay component `{raw_head_spec}`; expected head:key=value")
        head, raw_items = raw_head_spec.split(":", 1)
        head = head.strip()
        if head not in HEADS:
            raise ValueError(f"Unknown overlay head `{head}`")
        values: Dict[str, Any] = {}
        for item in raw_items.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid overlay item `{item}`; expected key=value")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "strategy_cap":
                values[key] = int(value)
            elif key in {"size", "priority", "rank"}:
                values[key] = float(value)
            else:
                raise ValueError(f"Unknown overlay key `{key}`")
        if values:
            spec[head] = values
    return spec


def _default_overlay_specs() -> Dict[str, str]:
    """Small default grid targeting heads that lost PnL in matched replay."""
    return {
        "none": "",
        "short_asset_size_50": "short_asset:size=0.50",
        "short_asset_size_25": "short_asset:size=0.25",
        "short_asset_priority_50": "short_asset:priority=0.50",
        "short_asset_rank_minus_002": "short_asset:rank=-0.02",
        "short_asset_rank_minus_005": "short_asset:rank=-0.05",
        "short_asset_strategy_cap_1": "short_asset:strategy_cap=1",
        "long_bars_size_50": "long_bars:size=0.50",
        "long_bars_priority_50": "long_bars:priority=0.50",
        "long_bars_rank_minus_002": "long_bars:rank=-0.02",
        "long_bars_strategy_cap_1": "long_bars:strategy_cap=1",
        "weak_heads_size_50": "short_asset:size=0.50;long_bars:size=0.50",
        "weak_heads_priority_50": "short_asset:priority=0.50;long_bars:priority=0.50",
        "weak_heads_rank_minus_002": "short_asset:rank=-0.02;long_bars:rank=-0.02",
        "weak_heads_strategy_cap_1": "short_asset:strategy_cap=1;long_bars:strategy_cap=1",
    }


def _load_overlay_specs(
    overlay: Sequence[str] | None,
    overlay_file: Path | None,
) -> Dict[str, str]:
    specs = _default_overlay_specs() if not overlay and overlay_file is None else {}
    for idx, item in enumerate(overlay or []):
        text = str(item).strip()
        if not text:
            continue
        starts_with_head = any(text.startswith(f"{head}:") for head in HEADS)
        if starts_with_head:
            specs[f"overlay_{idx + 1}"] = text
        elif "=" in text:
            label, spec = text.split("=", 1)
            specs[label.strip()] = spec.strip()
        else:
            raise ValueError(f"Invalid --overlay `{text}`; expected label=spec")
    if overlay_file is not None:
        if not overlay_file.exists():
            raise FileNotFoundError(f"Missing overlay file: {overlay_file}")
        if overlay_file.suffix.lower() == ".csv":
            frame = pd.read_csv(overlay_file)
            if not {"overlay_id", "overlay_spec"}.issubset(frame.columns):
                raise ValueError(f"{overlay_file} must contain overlay_id and overlay_spec columns")
            for _, row in frame.iterrows():
                specs[str(row["overlay_id"]).strip()] = str(row["overlay_spec"]).strip()
        else:
            for line in overlay_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                label, spec = line.split("=", 1)
                specs[label.strip()] = spec.strip()
    return {label: spec for label, spec in specs.items() if label}


def _apply_overlay(candidates: pd.DataFrame, overlay: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    if not overlay:
        return candidates
    out = candidates.copy()
    head = out["strategy_id"].astype(str).map(_head_name)
    for col, default in (
        ("portfolio_size_multiplier", 1.0),
        ("portfolio_priority_multiplier", 1.0),
        ("portfolio_rank_adjustment", 0.0),
    ):
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    if "portfolio_max_concurrent_per_strategy" not in out.columns:
        out["portfolio_max_concurrent_per_strategy"] = np.nan
    out["portfolio_max_concurrent_per_strategy"] = pd.to_numeric(
        out["portfolio_max_concurrent_per_strategy"],
        errors="coerce",
    )

    for overlay_head, values in overlay.items():
        mask = head.eq(overlay_head)
        if not bool(mask.any()):
            continue
        if "size" in values:
            out.loc[mask, "portfolio_size_multiplier"] = (
                out.loc[mask, "portfolio_size_multiplier"] * float(values["size"])
            )
        if "priority" in values:
            out.loc[mask, "portfolio_priority_multiplier"] = (
                out.loc[mask, "portfolio_priority_multiplier"] * float(values["priority"])
            )
        if "rank" in values:
            out.loc[mask, "portfolio_rank_adjustment"] = (
                out.loc[mask, "portfolio_rank_adjustment"] + float(values["rank"])
            )
        if "strategy_cap" in values:
            cap = int(values["strategy_cap"])
            current = out.loc[mask, "portfolio_max_concurrent_per_strategy"]
            out.loc[mask, "portfolio_max_concurrent_per_strategy"] = np.where(
                current.notna(),
                np.minimum(current.to_numpy(dtype=float), float(cap)),
                float(cap),
            )
    return out


def _head_delta_table(accepted: pd.DataFrame, baseline_combo: str, baseline_overlay: str) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame()
    work = accepted.copy()
    work["head"] = work["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(work["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(work["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(work["position_gross_return"], errors="coerce").fillna(0.0)
    work["net_pnl"] = size * net
    work["gross_pnl"] = size * gross
    work["hit"] = net.gt(0.0)
    work["full_sl"] = work["position_exit_reason"].astype(str).eq("full_sl")
    grouped = (
        work.groupby(["combo_id", "overlay_id", "head"], dropna=False)
        .agg(
            trades=("head", "size"),
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            hit_rate=("hit", "mean"),
            full_sl_rate=("full_sl", "mean"),
        )
        .reset_index()
    )
    baseline = grouped.loc[
        grouped["combo_id"].eq(baseline_combo) & grouped["overlay_id"].eq(baseline_overlay)
    ].set_index("head")
    rows: List[Dict[str, Any]] = []
    for (combo_id, overlay_id), group in grouped.groupby(["combo_id", "overlay_id"], sort=False):
        current = group.set_index("head")
        for head in sorted(set(baseline.index).union(current.index)):
            rec = {"combo_id": combo_id, "overlay_id": overlay_id, "head": head}
            for col in ("trades", "net_pnl", "gross_pnl", "hit_rate", "full_sl_rate"):
                b = float(baseline.loc[head, col]) if head in baseline.index else 0.0
                c = float(current.loc[head, col]) if head in current.index else 0.0
                rec[f"baseline_{col}"] = b
                rec[f"candidate_{col}"] = c
                rec[f"delta_{col}"] = c - b
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--combo-id", action="append", default=None)
    parser.add_argument("--combo-file", type=Path, default=None)
    parser.add_argument("--overlay", action="append", default=None, help="Overlay as label=head:key=value")
    parser.add_argument("--overlay-file", type=Path, default=None)
    parser.add_argument("--save-accepted-decisions", action="store_true")
    args = parser.parse_args()

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = _load_arm_tables(args.source_dir, arms)
    heads = sorted(tables[arms[0]]["head"].dropna().astype(str).unique())
    requested_combo_ids = _load_requested_combo_ids(args.combo_id, args.combo_file)
    overlay_specs = _load_overlay_specs(args.overlay, args.overlay_file)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)

    rows: List[Dict[str, Any]] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    accepted_frames: List[pd.DataFrame] = []

    for mapping in _arm_combinations(heads, arms):
        combo_id = _combo_id(mapping)
        if requested_combo_ids and combo_id not in requested_combo_ids:
            continue
        frames = []
        for head, arm in mapping.items():
            source = tables[arm]
            frames.append(source.loc[source["head"].eq(head)].copy())
        base_candidates = (
            pd.concat(frames, ignore_index=True)
            .drop(columns=["head"], errors="ignore")
            .sort_values(["timestamp", "strategy_id", "symbol"])
            .reset_index(drop=True)
        )
        for overlay_id, overlay_text in overlay_specs.items():
            overlay = _parse_overlay_spec(overlay_text)
            candidates = _apply_overlay(base_candidates, overlay)
            ev_curve = fit_hierarchical_ev_curves(candidates)
            decisions, _equity, metrics = replay_candidates(
                candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            daily, weekly = _accepted_period_tables(decisions)
            for frame in (daily, weekly):
                if not frame.empty:
                    frame.insert(0, "combo_id", combo_id)
                    frame.insert(1, "overlay_id", overlay_id)
                    frame.insert(2, "overlay_spec", overlay_text)
                    for head, arm in mapping.items():
                        frame[f"{head}_arm"] = arm
            daily_frames.append(daily)
            weekly_frames.append(weekly)
            if args.save_accepted_decisions and "accepted" in decisions.columns:
                accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
                if not accepted.empty:
                    accepted.insert(0, "combo_id", combo_id)
                    accepted.insert(1, "overlay_id", overlay_id)
                    accepted.insert(2, "overlay_spec", overlay_text)
                    for head, arm in mapping.items():
                        accepted[f"{head}_arm"] = arm
                    accepted_frames.append(accepted)
            rec = {
                "combo_id": combo_id,
                "overlay_id": overlay_id,
                "overlay_spec": overlay_text,
                **{f"{head}_arm": arm for head, arm in mapping.items()},
                "candidate_rows": int(len(candidates)),
                "candidate_start": str(pd.to_datetime(candidates["timestamp"], utc=True).min()),
                "candidate_end": str(pd.to_datetime(candidates["timestamp"], utc=True).max()),
                "objective": float(metrics.get("objective", 0.0)),
                "net_pnl": float(metrics.get("net_pnl", 0.0)),
                "gross_pnl": float(metrics.get("gross_pnl", 0.0)),
                "trade_count": int(metrics.get("trade_count", 0) or 0),
                "mean_net_return": float(metrics.get("mean_net_return_per_trade", 0.0)),
                "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
                "timeout_rate": float(metrics.get("timeout_rate", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "worst_week_return": float(metrics.get("worst_week", 0.0)),
                "strategy_concentration": float(metrics.get("strategy_concentration", 0.0)),
                "side_concentration": float(metrics.get("side_concentration", 0.0)),
            }
            rec.update(_period_metrics(daily.get("net_pnl", pd.Series(dtype=float)), "daily"))
            rec.update(_period_metrics(weekly.get("net_pnl", pd.Series(dtype=float)), "weekly"))
            rows.append(rec)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        weekly_count = pd.to_numeric(summary["weekly_count"], errors="coerce").replace(0.0, np.nan)
        summary["avg_week_pnl"] = pd.to_numeric(summary["net_pnl"], errors="coerce") / weekly_count
        summary["objective_avgweek_0p7dayq35_0p3dayq20"] = (
            summary["avg_week_pnl"].fillna(0.0)
            + 0.7 * pd.to_numeric(summary["daily_q35_pnl"], errors="coerce").fillna(0.0)
            + 0.3 * pd.to_numeric(summary["daily_q20_pnl"], errors="coerce").fillna(0.0)
        )
        summary = summary.sort_values(
            "objective_avgweek_0p7dayq35_0p3dayq20",
            ascending=False,
        ).reset_index(drop=True)
    daily_all = _concat_nonempty(daily_frames)
    weekly_all = _concat_nonempty(weekly_frames)
    accepted_all = _concat_nonempty(accepted_frames)
    summary.to_csv(args.out_dir / "head_overlay_summary.csv", index=False)
    daily_all.to_csv(args.out_dir / "head_overlay_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "head_overlay_weekly.csv", index=False)
    if args.save_accepted_decisions:
        accepted_all.to_parquet(args.out_dir / "head_overlay_accepted_decisions.parquet", index=False)
        head_delta = _head_delta_table(accepted_all, "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S", "none")
        head_delta.to_csv(args.out_dir / "head_overlay_per_head_delta_vs_static_none.csv", index=False)

    keep = [
        "combo_id",
        "overlay_id",
        "objective_avgweek_0p7dayq35_0p3dayq20",
        "avg_week_pnl",
        "net_pnl",
        "trade_count",
        "daily_q20_pnl",
        "daily_q35_pnl",
        "weekly_q05_pnl",
        "weekly_q10_pnl",
        "weekly_q20_pnl",
        "weekly_q35_pnl",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
        "overlay_spec",
    ]
    lines = [
        "# Contextual TP/SL Head Portfolio Overlay Ablation",
        "",
        f"Source: `{args.source_dir}`",
        f"Rows: `{len(summary)}`",
        "Period: full source candidate table period. Costs included.",
        "",
        "## Top Requested Objective",
        "",
        summary[[c for c in keep if c in summary.columns]].head(30).round(6).to_markdown(index=False)
        if not summary.empty
        else "_No rows._",
    ]
    (args.out_dir / "head_overlay_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "combo_ids": sorted(requested_combo_ids),
        "overlay_specs": overlay_specs,
        "rows": int(len(summary)),
        "top_requested_objective": summary.head(10).to_dict(orient="records"),
    }
    (args.out_dir / "head_overlay_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(summary)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
