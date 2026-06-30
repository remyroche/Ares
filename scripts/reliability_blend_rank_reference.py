"""Shared rank-reference helpers for reliability-blend replay/materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
)


def _side_value(row: pd.Series) -> str:
    for col in ("side_name", "side"):
        if col not in row.index:
            continue
        value = row[col]
        text = str(value).strip().lower()
        if text in {"long", "short"}:
            return text
        if text in {"1", "1.0"}:
            return "long"
        if text in {"-1", "-1.0"}:
            return "short"
    strategy = str(row.get("strategy_id", "")).strip().lower()
    if strategy.startswith("short"):
        return "short"
    return "long"


def apply_frozen_policy_rank_reference(
    frame: pd.DataFrame,
    *,
    data_root: str | Path,
    run_id: str | None,
    score_col: str = "calibrated_score",
    allow_window_rank_debug: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach live-equivalent per-strategy and auction ranks.

    The production path uses :class:`PolicyRankReferenceStore`, not a percentile
    over the current replay window.  Window ranks remain available only as an
    explicit debug/audit fallback.
    """

    out = frame.copy()
    if not run_id:
        if not allow_window_rank_debug:
            raise RuntimeError(
                "Missing --rank-reference-run-id. Refusing to compute whole-window "
                "rank percentiles for deployable reliability-blend replay."
            )
        rank = pd.to_numeric(out[score_col], errors="coerce").rank(method="average", pct=True)
        out["policy_rank_pct"] = rank
        out["strategy_rank_pct"] = rank
        out["normalized_rank_score"] = rank
        out["auction_rank_score"] = rank
        out["threshold_rank_score_source"] = "window_rank_debug_not_deployable"
        return out, {
            "rank_reference_run_id": None,
            "rank_source": "window_rank_debug_not_deployable",
            "missing_rank_rows": 0,
            "missing_auction_rank_rows": 0,
            "ranked_rows": int(rank.notna().sum()),
            "auction_ranked_rows": int(rank.notna().sum()),
            "window_rank_debug_used": True,
        }

    store = PolicyRankReferenceStore(data_root=data_root, run_id=str(run_id))
    policy_rank = np.full(len(out), np.nan, dtype=np.float64)
    auction_rank = np.full(len(out), np.nan, dtype=np.float64)
    rank_ref_n = np.zeros(len(out), dtype=np.int64)
    auction_ref_n = np.zeros(len(out), dtype=np.int64)
    rank_sources: list[str] = [""] * len(out)
    auction_sources: list[str] = [""] * len(out)
    scores = pd.to_numeric(out[score_col], errors="coerce").to_numpy(dtype=np.float64)

    for i, (_, row) in enumerate(out.iterrows()):
        score = float(scores[i])
        if not np.isfinite(score):
            continue
        side = _side_value(row)
        strategy_id = str(row.get("strategy_id", "")).strip()
        lookup = store.lookup(
            strategy_id=strategy_id,
            side=side,
            calibrated_score=score,
        )
        auction = store.lookup_auction(calibrated_score=score)
        policy_rank[i] = float(lookup.policy_rank_pct)
        auction_rank[i] = float(auction.policy_rank_pct)
        rank_ref_n[i] = int(lookup.n_rows)
        auction_ref_n[i] = int(auction.n_rows)
        rank_sources[i] = str(lookup.source or lookup.strategy_id or "")
        auction_sources[i] = str(auction.source or auction.strategy_id or "")

    missing = int((~np.isfinite(policy_rank)).sum())
    missing_auction = int((~np.isfinite(auction_rank)).sum())
    if missing or missing_auction:
        if not allow_window_rank_debug:
            raise RuntimeError(
                f"Policy/auction rank reference missing for policy={missing}, "
                f"auction={missing_auction} rows. "
                "Pass --allow-window-rank-debug only for non-deployable audits."
            )
        fallback = pd.to_numeric(out[score_col], errors="coerce").rank(method="average", pct=True)
        mask = ~np.isfinite(policy_rank)
        policy_rank[mask] = fallback.to_numpy(dtype=np.float64)[mask]
        auction_rank[~np.isfinite(auction_rank)] = fallback.to_numpy(dtype=np.float64)[
            ~np.isfinite(auction_rank)
        ]
        rank_sources = [
            src if src else "window_rank_debug_fallback"
            for src in rank_sources
        ]
        auction_sources = [
            src if src else "window_rank_debug_fallback"
            for src in auction_sources
        ]

    out["policy_rank_pct"] = policy_rank
    out["strategy_rank_pct"] = policy_rank
    out["normalized_rank_score"] = policy_rank
    out["auction_rank_score"] = auction_rank
    out["policy_rank_reference_n"] = rank_ref_n
    out["auction_rank_reference_n"] = auction_ref_n
    out["policy_rank_reference_source"] = rank_sources
    out["auction_rank_reference_source"] = auction_sources
    out["threshold_rank_score_source"] = "policy_rank_reference_percentile"
    finite_ref_n = rank_ref_n[rank_ref_n > 0]
    finite_auction_ref_n = auction_ref_n[auction_ref_n > 0]
    return out, {
        "rank_reference_run_id": str(run_id),
        "rank_source": "policy_rank_reference_percentile",
        "missing_rank_rows": missing,
        "missing_auction_rank_rows": missing_auction,
        "ranked_rows": int(np.isfinite(policy_rank).sum()),
        "auction_ranked_rows": int(np.isfinite(auction_rank).sum()),
        "policy_rank_reference_n_min": int(finite_ref_n.min()) if finite_ref_n.size else 0,
        "auction_rank_reference_n_min": int(finite_auction_ref_n.min()) if finite_auction_ref_n.size else 0,
        "policy_rank_reference_source_counts": dict(pd.Series(rank_sources).value_counts(dropna=False).to_dict()),
        "auction_rank_reference_source_counts": dict(pd.Series(auction_sources).value_counts(dropna=False).to_dict()),
        "window_rank_debug_used": bool(
            any("window_rank_debug" in str(src) for src in rank_sources)
            or any("window_rank_debug" in str(src) for src in auction_sources)
        ),
    }
