#!/usr/bin/env python3
"""Analyze why frozen smooth-penalty dual-scoring candidates do or do not bind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_manifest(dual_dir: Path) -> dict[str, Any]:
    return json.loads((dual_dir / "dual_scoring_manifest.json").read_text())


def _load_eval_candidates(path: Path, eval_start: pd.Timestamp, eval_end: pd.Timestamp | None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = frame["timestamp"].ge(eval_start)
    if eval_end is not None:
        mask &= frame["timestamp"].le(eval_end)
    out = frame.loc[mask].copy().reset_index(drop=True)
    if "portfolio_rank_adjustment" not in out.columns:
        out["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        out["portfolio_rank_adjustment"] = (
            pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
    return out


def _load_decisions(dual_dir: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_parquet(dual_dir / variant / "decisions.parquet")
    frame["candidate_index"] = pd.to_numeric(frame["candidate_index"], errors="coerce").astype("Int64")
    frame["accepted"] = frame["accepted"].astype(bool)
    frame["rank_margin"] = (
        pd.to_numeric(frame["effective_rank_score"], errors="coerce")
        - pd.to_numeric(frame["dynamic_threshold"], errors="coerce")
    )
    return frame


def _accepted_floor_by_timestamp(decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"]].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["timestamp", "accepted_min_priority", "accepted_min_score"])
    return (
        accepted.groupby("timestamp", as_index=False)
        .agg(
            accepted_min_priority=("portfolio_priority", "min"),
            accepted_min_score=("effective_rank_score", "min"),
            accepted_count=("accepted", "size"),
        )
    )


def _summary_stats(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"min": np.nan, "p10": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _variant_boundary(
    *,
    label: str,
    baseline_decisions: pd.DataFrame,
    candidate_decisions: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    adjusted = candidates.loc[pd.to_numeric(candidates["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).ne(0.0)].copy()
    base = baseline_decisions.set_index("candidate_index", drop=False)
    cand = candidate_decisions.set_index("candidate_index", drop=False)
    rows: list[dict[str, Any]] = []
    floor = _accepted_floor_by_timestamp(candidate_decisions)
    if not adjusted.empty:
        adjusted = adjusted.reset_index().rename(columns={"index": "candidate_index"})
        for _, row in adjusted.iterrows():
            idx = int(row["candidate_index"])
            if idx not in base.index or idx not in cand.index:
                continue
            b = base.loc[idx]
            c = cand.loc[idx]
            item = {
                "variant": label,
                "candidate_index": idx,
                "timestamp": c["timestamp"],
                "strategy_id": c["strategy_id"],
                "symbol": c["symbol"],
                "adjustment": float(row["portfolio_rank_adjustment"]),
                "baseline_accepted": bool(b["accepted"]),
                "candidate_accepted": bool(c["accepted"]),
                "baseline_rejection_reason": str(b.get("rejection_reason", "")),
                "candidate_rejection_reason": str(c.get("rejection_reason", "")),
                "baseline_effective_rank_score": float(b["effective_rank_score"]),
                "candidate_effective_rank_score": float(c["effective_rank_score"]),
                "candidate_dynamic_threshold": float(c["dynamic_threshold"]),
                "candidate_rank_margin": float(c["rank_margin"]),
                "candidate_portfolio_priority": float(c["portfolio_priority"]),
            }
            rows.append(item)
    detail = pd.DataFrame(rows)
    if not detail.empty and not floor.empty:
        detail = detail.merge(floor, on="timestamp", how="left")
        detail["priority_gap_to_accepted_floor"] = (
            pd.to_numeric(detail["candidate_portfolio_priority"], errors="coerce")
            - pd.to_numeric(detail["accepted_min_priority"], errors="coerce")
        )
        detail["score_gap_to_accepted_floor"] = (
            pd.to_numeric(detail["candidate_effective_rank_score"], errors="coerce")
            - pd.to_numeric(detail["accepted_min_score"], errors="coerce")
        )
    elif not detail.empty:
        detail["priority_gap_to_accepted_floor"] = np.nan
        detail["score_gap_to_accepted_floor"] = np.nan

    accepted_adjusted = detail["candidate_accepted"].sum() if not detail.empty else 0
    changed_acceptance = (
        (detail["baseline_accepted"] != detail["candidate_accepted"]).sum() if not detail.empty else 0
    )
    margin_stats = _summary_stats(detail["candidate_rank_margin"] if not detail.empty else pd.Series(dtype=float))
    accepted_margins = candidate_decisions.loc[candidate_decisions["accepted"], "rank_margin"]
    summary = {
        "variant": label,
        "eval_rows": int(len(candidates)),
        "adjusted_rows": int(len(detail)),
        "adjusted_share": float(len(detail) / max(len(candidates), 1)),
        "adjusted_candidate_accepted": int(accepted_adjusted),
        "adjusted_acceptance_changed": int(changed_acceptance),
        "adjusted_below_threshold": int((detail["candidate_rank_margin"] < 0.0).sum()) if not detail.empty else 0,
        "adjusted_within_0p005_of_threshold": int((detail["candidate_rank_margin"].abs() <= 0.005).sum()) if not detail.empty else 0,
        "adjusted_within_0p010_of_threshold": int((detail["candidate_rank_margin"].abs() <= 0.010).sum()) if not detail.empty else 0,
        "adjusted_within_0p020_of_threshold": int((detail["candidate_rank_margin"].abs() <= 0.020).sum()) if not detail.empty else 0,
        "adjusted_margin_min": margin_stats["min"],
        "adjusted_margin_p10": margin_stats["p10"],
        "adjusted_margin_median": margin_stats["median"],
        "adjusted_margin_p90": margin_stats["p90"],
        "adjusted_margin_max": margin_stats["max"],
        "accepted_margin_min": _summary_stats(accepted_margins)["min"],
        "accepted_margin_median": _summary_stats(accepted_margins)["median"],
        "capacity_rejected_adjusted": int(
            detail["candidate_rejection_reason"].str.contains("max_new_entries|max_concurrent", case=False, na=False).sum()
        )
        if not detail.empty
        else 0,
        "threshold_rejected_adjusted": int(
            detail["candidate_rejection_reason"].str.contains("below_dynamic_threshold", case=False, na=False).sum()
        )
        if not detail.empty
        else 0,
        "symbol_state_rejected_adjusted": int(
            detail["candidate_rejection_reason"].str.contains("symbol_", case=False, na=False).sum()
        )
        if not detail.empty
        else 0,
    }
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(args.dual_dir)
    eval_start = pd.Timestamp(manifest["eval_start"], tz="UTC")
    eval_end = pd.Timestamp(manifest["eval_end"], tz="UTC") if manifest.get("eval_end") else None
    baseline_decisions = _load_decisions(args.dual_dir, "baseline")
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for label, path in manifest["candidates"].items():
        candidates = _load_eval_candidates(Path(path), eval_start, eval_end)
        candidate_decisions = _load_decisions(args.dual_dir, label)
        summary, detail = _variant_boundary(
            label=label,
            baseline_decisions=baseline_decisions,
            candidate_decisions=candidate_decisions,
            candidates=candidates,
        )
        summaries.append(summary)
        if not detail.empty:
            details.append(detail)
    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_df.to_csv(args.output_dir / "boundary_summary.csv", index=False)
    detail_df.to_csv(args.output_dir / "boundary_adjusted_rows.csv", index=False)
    out_manifest = {
        "generated_by": "analyze_frozen_dual_scoring_boundary",
        "dual_dir": str(args.dual_dir),
        "eval_start": eval_start.isoformat(),
        "eval_end": eval_end.isoformat() if eval_end is not None else "",
        "variants": list(manifest["candidates"]),
    }
    (args.output_dir / "boundary_manifest.json").write_text(
        json.dumps(_json_safe(out_manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Frozen Dual-Scoring Boundary Analysis",
        "",
        f"Dual replay: `{args.dual_dir}`",
        f"Evaluation: `{out_manifest['eval_start']}` to `{out_manifest['eval_end'] or 'open'}`",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
    ]
    if not detail_df.empty:
        lines.extend(
            [
                "",
                "## Adjusted Rejection Reasons",
                "",
                detail_df.groupby(["variant", "candidate_rejection_reason"]).size().reset_index(name="rows").to_markdown(index=False),
            ]
        )
    (args.output_dir / "boundary_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
