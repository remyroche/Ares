#!/usr/bin/env python3
"""Replay frozen smooth-penalty candidate outputs on a common flat universe.

This is a delayed/prospective dual-scoring helper: references and candidate
adjustments are already frozen in the input parquet files.  The script fits the
portfolio EV curve only on rows before the evaluation start, then replays
baseline and candidate outputs on the same post-cutoff rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _head_name(strategy_id: Any) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"timestamp", "strategy_id", "symbol"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    if "portfolio_rank_adjustment" not in out.columns:
        out["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        out["portfolio_rank_adjustment"] = (
            pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
    return out.sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["timestamp", "strategy_id", "symbol"]].reset_index(drop=True)


def _assert_same_universe(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} row count mismatch: {len(candidate)} != {len(reference)}")
    ref_key = _key_frame(reference)
    cand_key = _key_frame(candidate)
    if not ref_key.equals(cand_key):
        mismatch = (ref_key != cand_key).any(axis=1)
        first = int(np.flatnonzero(mismatch.to_numpy())[0]) if mismatch.any() else -1
        raise ValueError(f"{label} candidate universe/key ordering mismatch at row {first}")


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame(), pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    reason = accepted.get("position_exit_reason", accepted.get("exit_reason", ""))
    if not isinstance(reason, pd.Series):
        reason = pd.Series("", index=accepted.index)
    accepted["is_full_sl"] = reason.astype(str).str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = reason.astype(str).str.contains("timeout", case=False, na=False)

    frames: list[pd.DataFrame] = []
    for cols in (["week"], ["week", "head"], ["day"], ["day", "head"]):
        cur = (
            accepted.groupby(cols, as_index=False)
            .agg(
                net_pnl=("net_pnl_amount", "sum"),
                gross_pnl=("gross_pnl_amount", "sum"),
                trades=("accepted", "size"),
                hit_rate=("is_win", "mean"),
                full_sl_rate=("is_full_sl", "mean"),
                timeout_rate=("is_timeout", "mean"),
            )
            .sort_values(cols)
        )
        cur.insert(0, "period_type", "_".join(cols))
        frames.append(cur)
    return pd.concat(frames[2:], ignore_index=True), pd.concat(frames[:2], ignore_index=True)


def _accepted_table(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    reason = accepted.get("position_exit_reason", accepted.get("exit_reason", ""))
    if not isinstance(reason, pd.Series):
        reason = pd.Series("", index=accepted.index)
    accepted["is_full_sl"] = reason.astype(str).str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = reason.astype(str).str.contains("timeout", case=False, na=False)
    return accepted


def _head_summary(label: str, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = _accepted_table(decisions)
    if accepted.empty:
        return pd.DataFrame()
    out = (
        accepted.groupby("head", as_index=False)
        .agg(
            net_pnl=("net_pnl_amount", "sum"),
            gross_pnl=("gross_pnl_amount", "sum"),
            trades=("accepted", "size"),
            hit_rate=("is_win", "mean"),
            full_sl_rate=("is_full_sl", "mean"),
            timeout_rate=("is_timeout", "mean"),
        )
        .sort_values("head")
    )
    out.insert(0, "variant", label)
    return out


def _decision_keys(decisions: pd.DataFrame) -> set[tuple[Any, str, str]]:
    if decisions.empty or "accepted" not in decisions.columns:
        return set()
    accepted = decisions.loc[decisions["accepted"].astype(bool)]
    return set(
        zip(
            pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce"),
            accepted["strategy_id"].astype(str),
            accepted["symbol"].astype(str),
        )
    )


def _summary_delta(base: dict[str, Any], cur: dict[str, Any]) -> dict[str, float]:
    keys = [
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
    ]
    return {f"delta_{key}": float(cur.get(key, 0.0) or 0.0) - float(base.get(key, 0.0) or 0.0) for key in keys}


def _adjustment_summary(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    adj = pd.to_numeric(frame["portfolio_rank_adjustment"], errors="coerce").fillna(0.0)
    nonzero = adj.ne(0.0)
    return {
        "variant": label,
        "rows": int(len(frame)),
        "adjusted_rows": int(nonzero.sum()),
        "adjusted_share": float(nonzero.mean()) if len(frame) else 0.0,
        "mean_adjustment_on_adjusted": float(adj[nonzero].mean()) if nonzero.any() else 0.0,
        "min_adjustment": float(adj[nonzero].min()) if nonzero.any() else 0.0,
        "max_adjustment": float(adj[nonzero].max()) if nonzero.any() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="label=path. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-start", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC") if args.eval_end else None
    baseline_all = _load_candidates(args.baseline)
    candidate_paths: dict[str, Path] = {}
    for raw in args.candidate:
        if "=" not in raw:
            raise ValueError("--candidate must be label=path")
        label, path = raw.split("=", 1)
        candidate_paths[label.strip()] = Path(path)
    if not candidate_paths:
        raise ValueError("At least one --candidate label=path is required")

    train = baseline_all[baseline_all["timestamp"].lt(eval_start)].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError(f"No baseline rows before eval-start {eval_start.isoformat()} for EV fit")
    eval_mask = baseline_all["timestamp"].ge(eval_start)
    if eval_end is not None:
        eval_mask &= baseline_all["timestamp"].le(eval_end)
    baseline_eval = baseline_all.loc[eval_mask].copy().reset_index(drop=True)
    if baseline_eval.empty:
        raise ValueError("No baseline rows in evaluation window")

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_curve = fit_hierarchical_ev_curves(train)
    variants: dict[str, pd.DataFrame] = {"baseline": baseline_eval}
    input_hashes = {"baseline": _sha256(args.baseline)}
    for label, path in candidate_paths.items():
        cand_all = _load_candidates(path)
        _assert_same_universe(baseline_all, cand_all, label)
        cand_eval = cand_all.loc[eval_mask].copy().reset_index(drop=True)
        variants[label] = cand_eval
        input_hashes[label] = _sha256(path)

    summaries: list[dict[str, Any]] = []
    head_rows: list[pd.DataFrame] = []
    weekly_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    adjustment_rows: list[dict[str, Any]] = []
    accepted_sets: dict[str, set[tuple[Any, str, str]]] = {}
    base_metrics: dict[str, Any] | None = None
    for label, candidates in variants.items():
        out_dir = args.output_dir / label
        out_dir.mkdir(exist_ok=True)
        decisions, equity, metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        decisions.to_parquet(out_dir / "decisions.parquet", index=False)
        equity.to_parquet(out_dir / "equity.parquet", index=False)
        daily, weekly = _period_tables(decisions)
        if not daily.empty:
            daily["variant"] = label
            daily_rows.append(daily)
        if not weekly.empty:
            weekly["variant"] = label
            weekly_rows.append(weekly)
        head = _head_summary(label, decisions)
        if not head.empty:
            head_rows.append(head)
        accepted_sets[label] = _decision_keys(decisions)
        if label == "baseline":
            base_metrics = dict(metrics)
        row = {"variant": label, **dict(metrics)}
        if base_metrics is not None and label != "baseline":
            row.update(_summary_delta(base_metrics, dict(metrics)))
        summaries.append(row)
        adjustment_rows.append(_adjustment_summary(label, candidates))

    summary = pd.DataFrame(summaries)
    adjustments = pd.DataFrame(adjustment_rows)
    head_out = pd.concat(head_rows, ignore_index=True) if head_rows else pd.DataFrame()
    weekly_out = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()
    daily_out = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()

    overlap_rows: list[dict[str, Any]] = []
    base_set = accepted_sets.get("baseline", set())
    for label, cur_set in accepted_sets.items():
        if label == "baseline":
            continue
        overlap_rows.append(
            {
                "variant": label,
                "baseline_accepted": int(len(base_set)),
                "candidate_accepted": int(len(cur_set)),
                "overlap": int(len(base_set & cur_set)),
                "entrants": int(len(cur_set - base_set)),
                "removed": int(len(base_set - cur_set)),
                "jaccard": float(len(base_set & cur_set) / max(len(base_set | cur_set), 1)),
            }
        )
    overlaps = pd.DataFrame(overlap_rows)

    summary.to_csv(args.output_dir / "dual_scoring_summary.csv", index=False)
    adjustments.to_csv(args.output_dir / "dual_scoring_adjustment_summary.csv", index=False)
    overlaps.to_csv(args.output_dir / "dual_scoring_accepted_overlap.csv", index=False)
    head_out.to_csv(args.output_dir / "dual_scoring_per_head.csv", index=False)
    weekly_out.to_csv(args.output_dir / "dual_scoring_weekly.csv", index=False)
    daily_out.to_csv(args.output_dir / "dual_scoring_daily.csv", index=False)

    manifest = {
        "generated_by": "replay_frozen_smooth_penalty_dual_scoring",
        "baseline": str(args.baseline),
        "candidates": {label: str(path) for label, path in candidate_paths.items()},
        "input_hashes": input_hashes,
        "eval_start": eval_start.isoformat(),
        "eval_end": eval_end.isoformat() if eval_end is not None else "",
        "train_rows_for_ev": int(len(train)),
        "eval_rows": int(len(baseline_eval)),
        "eval_timestamp_min": baseline_eval["timestamp"].min().isoformat(),
        "eval_timestamp_max": baseline_eval["timestamp"].max().isoformat(),
        "policy_params": asdict(params),
        "market_mode": args.market_mode,
    }
    (args.output_dir / "dual_scoring_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Frozen Smooth-Penalty Dual Scoring Replay",
        "",
        f"Evaluation: `{manifest['eval_timestamp_min']}` to `{manifest['eval_timestamp_max']}`",
        f"EV reference rows before evaluation: `{manifest['train_rows_for_ev']}`",
        f"Evaluation rows: `{manifest['eval_rows']}`",
        "Costs are included through `portfolio_policy_replay.replay_candidates`.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Adjustment Coverage",
        "",
        adjustments.to_markdown(index=False),
        "",
        "## Accepted-Trade Overlap",
        "",
        overlaps.to_markdown(index=False),
    ]
    (args.output_dir / "dual_scoring_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
