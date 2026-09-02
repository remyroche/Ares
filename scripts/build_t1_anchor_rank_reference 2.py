#!/usr/bin/env python3
"""Build a causal global-over-time anchor-score rank reference for T1.

This persists pre-June anchor/meta score distributions for percentile lookup.
It is a score-normalization artifact only: it does not use June rows, labels,
returns, or policy performance to choose thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    persist_fullscope_score_distribution_reference,
)
from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import _file_sha256, _json_safe  # noqa: E402


DEFAULT_SOURCE_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_RUN_ID = "reliability_blend_anchor_rank_reference_20260625_prejune"


def _infer_head(strategy_id: Any) -> str:
    sid = str(strategy_id)
    for head in ("long_bars", "long_dist", "short_asset", "short_boll"):
        if sid.startswith(head):
            return head
    return "unknown"


def _parse_heads(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _frames_by_strategy(
    candidates: pd.DataFrame,
    *,
    score_col: str,
    heads: set[str],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    out: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []
    work = candidates.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if "head" not in work.columns:
        work["head"] = work["strategy_id"].map(_infer_head)
    if heads:
        work = work.loc[work["head"].astype(str).isin(heads)].copy()
    for strategy_id, group in work.groupby("strategy_id", sort=True):
        score = pd.to_numeric(group[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(group["timestamp"], utc=True, errors="coerce"),
                "symbol": group["symbol"].astype(str),
                "strategy_id": str(strategy_id),
                "head": group["head"].astype(str).to_numpy(),
                "calibrated_score": score,
            }
        ).dropna(subset=["timestamp", "calibrated_score"])
        if "side" in group.columns:
            frame["side"] = group["side"].to_numpy()
        frame = frame.drop_duplicates(["timestamp", "symbol", "strategy_id"], keep="last")
        if frame.empty:
            diagnostics.append(
                {
                    "strategy_id": str(strategy_id),
                    "head": str(group["head"].iloc[0]) if len(group) else "",
                    "status": "empty",
                    "rows": 0,
                }
            )
            continue
        out[str(strategy_id)] = frame
        diagnostics.append(
            {
                "strategy_id": str(strategy_id),
                "head": str(frame["head"].iloc[0]),
                "status": "ok",
                "rows": int(len(frame)),
                "timestamp_min": frame["timestamp"].min().isoformat(),
                "timestamp_max": frame["timestamp"].max().isoformat(),
                "score_min": float(frame["calibrated_score"].min()),
                "score_max": float(frame["calibrated_score"].max()),
            }
        )
    return out, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidates", type=Path, default=DEFAULT_SOURCE_CANDIDATES)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--score-col", default="calibrated_score")
    parser.add_argument("--heads", default="long_bars,long_dist,short_asset,short_boll")
    parser.add_argument("--cutoff", default="2026-06-15T04:00:00Z")
    args = parser.parse_args()

    candidates = pd.read_parquet(args.source_candidates)
    if args.score_col not in candidates.columns:
        raise RuntimeError(f"source candidates missing score column {args.score_col!r}")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    candidates = candidates.loc[candidates["timestamp"] < cutoff].copy()
    heads = _parse_heads(args.heads)
    frames, diagnostics = _frames_by_strategy(candidates, score_col=str(args.score_col), heads=heads)
    if not frames:
        raise RuntimeError("no finite anchor-score frames were available for rank-reference persistence")
    manifest_path = persist_fullscope_score_distribution_reference(
        frames,
        data_root=args.data_root,
        run_id=str(args.run_id),
        market_mode=str(args.market_mode),
        score_col="calibrated_score",
        provenance={
            "source": "t1_anchor_meta_score_prejune_candidate_distribution",
            "source_candidates": str(args.source_candidates),
            "source_candidates_sha256": _file_sha256(args.source_candidates),
            "source_score_col": str(args.score_col),
            "cutoff_exclusive": cutoff.isoformat(),
            "heads": sorted(heads),
            "leakage_contract": (
                "Rank reference is fitted only on pre-cutoff score distributions. "
                "It is for percentile mapping only and makes no performance or EV claim."
            ),
        },
    )
    freeze_manifest = {
        "generated_by": "build_t1_anchor_rank_reference",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(args.run_id),
        "market_mode": str(args.market_mode),
        "source_candidates": str(args.source_candidates),
        "source_candidates_sha256": _file_sha256(args.source_candidates),
        "score_col": str(args.score_col),
        "cutoff_exclusive": cutoff.isoformat(),
        "heads": sorted(heads),
        "rank_reference_manifest": str(manifest_path),
        "diagnostics": diagnostics,
    }
    out_path = Path(args.data_root) / "artifacts" / str(args.run_id) / "t1_anchor_rank_reference_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(freeze_manifest), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(freeze_manifest), indent=2)[:6000])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
