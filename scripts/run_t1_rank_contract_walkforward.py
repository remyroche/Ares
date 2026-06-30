#!/usr/bin/env python3
"""Walk-forward comparison of T1 timestamp and global rank contracts.

This is the pre-June validation track for the rank-contract question. It keeps
the score column, active heads, static thresholds, EV mapping, cost model and
global auction policy fixed, then compares:

* ``short_boll_timestamp_rank``: the measured T1 contract;
* ``fold_causal_global_rank_reference``: percentile ranks fitted only on each
  fold's training timestamps.

The script does not use market-state outputs, q-fail, HeadHealth, controller
thresholds or current-period performance features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402
from scripts.materialize_t1_repaired_static_baseline import _json_safe, _sha256  # noqa: E402
from scripts.reliability_blend_rank_reference import _side_value  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/t1_rank_contract_walkforward_20260626")
TIMESTAMP_ARM = "timestamp_rank_t1"
GLOBAL_ARM = "fold_causal_global_rank_reference"


@dataclass(frozen=True)
class TimeFold:
    fold: int
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end_exclusive: pd.Timestamp


def _load_candidates(path: Path, *, disabled_heads: set[str]) -> pd.DataFrame:
    return mstc._disable_heads(mstc._load_candidates(path), disabled_heads)


def _make_time_folds(
    timestamps: pd.Series,
    *,
    train_min_days: int,
    valid_days: int,
    step_days: int,
    embargo_hours: int,
) -> list[TimeFold]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    if ts.empty:
        return []
    min_ts = ts.iloc[0]
    max_ts = ts.iloc[-1]
    valid_start = min_ts + pd.Timedelta(days=int(train_min_days))
    folds: list[TimeFold] = []
    fold_idx = 1
    while valid_start <= max_ts:
        valid_end = valid_start + pd.Timedelta(days=int(valid_days))
        train_end = valid_start - pd.Timedelta(hours=int(embargo_hours))
        train_mask = ts < train_end
        valid_mask = (ts >= valid_start) & (ts < valid_end)
        if train_mask.any() and valid_mask.any():
            folds.append(
                TimeFold(
                    fold=fold_idx,
                    train_start=ts.loc[train_mask].iloc[0],
                    train_end_exclusive=train_end,
                    valid_start=ts.loc[valid_mask].iloc[0],
                    valid_end_exclusive=valid_end,
                )
            )
            fold_idx += 1
        valid_start = valid_start + pd.Timedelta(days=int(step_days))
    return folds


def _score_distribution_map(
    train: pd.DataFrame,
    *,
    score_col: str = "calibrated_score",
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    work = train.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work["_side_key"] = work.apply(_side_value, axis=1)
    refs: dict[tuple[str, str], np.ndarray] = {}
    for (strategy_id, side), group in work.groupby(["strategy_id", "_side_key"], sort=True):
        values = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            values.sort()
            refs[(str(strategy_id), str(side))] = values
    auction = pd.to_numeric(work[score_col], errors="coerce").to_numpy(dtype=np.float64)
    auction = auction[np.isfinite(auction)]
    auction.sort()
    return refs, auction


def _percentile_from_sorted(sorted_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ref = np.asarray(sorted_scores, dtype=np.float64)
    out = np.full(len(scores), np.nan, dtype=np.float64)
    if ref.size == 0:
        return out
    score = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(score)
    if mask.any():
        out[mask] = np.searchsorted(ref, score[mask], side="right") / float(ref.size)
    return out


def _apply_fold_global_rank_reference(
    frame: pd.DataFrame,
    *,
    reference_train: pd.DataFrame,
    score_col: str = "calibrated_score",
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy().reset_index(drop=True)
    refs, auction_ref = _score_distribution_map(reference_train, score_col=score_col)
    out["_side_key"] = out.apply(_side_value, axis=1)
    scores = pd.to_numeric(out[score_col], errors="coerce").to_numpy(dtype=np.float64)
    policy_rank = np.full(len(out), np.nan, dtype=np.float64)
    reference_n = np.zeros(len(out), dtype=np.int64)

    for (strategy_id, side), index in out.groupby(["strategy_id", "_side_key"], sort=False).groups.items():
        ref = refs.get((str(strategy_id), str(side)))
        idx = np.asarray(list(index), dtype=int)
        if ref is None or ref.size == 0:
            continue
        policy_rank[idx] = _percentile_from_sorted(ref, scores[idx])
        reference_n[idx] = int(ref.size)

    auction_rank = _percentile_from_sorted(auction_ref, scores)
    missing_policy = int((~np.isfinite(policy_rank)).sum())
    missing_auction = int((~np.isfinite(auction_rank)).sum())
    if (missing_policy or missing_auction) and not allow_missing:
        raise RuntimeError(
            "fold global rank reference missing ranks: "
            f"policy={missing_policy}, auction={missing_auction}"
        )
    for col in ("normalized_rank_score", "strategy_rank_pct", "policy_rank_pct", "rank_pct"):
        if col in out.columns:
            out[col] = policy_rank
    out["auction_rank_score"] = auction_rank
    out["policy_rank_reference_n"] = reference_n
    out["auction_rank_reference_n"] = int(auction_ref.size)
    out["rank_contract_source"] = "fold_causal_global_score_distribution"
    out["threshold_rank_score_source"] = "fold_causal_policy_rank_reference_percentile"
    out = out.drop(columns=["_side_key"], errors="ignore")
    return mstc.normalise_candidate_table(out), {
        "policy_reference_groups": int(len(refs)),
        "auction_reference_rows": int(auction_ref.size),
        "missing_policy_rank_rows": missing_policy,
        "missing_auction_rank_rows": missing_auction,
        "ranked_rows": int(np.isfinite(policy_rank).sum()),
        "auction_ranked_rows": int(np.isfinite(auction_rank).sum()),
    }


def _deployable_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    rank_col = next(
        (col for col in ("policy_rank_pct", "normalized_rank_score", "strategy_rank_pct", "rank_pct") if col in candidates.columns),
        None,
    )
    if rank_col is None:
        return candidates.iloc[0:0].copy()
    threshold_col = "deployment_rank_threshold" if "deployment_rank_threshold" in candidates.columns else "base_strategy_threshold"
    rank = pd.to_numeric(candidates[rank_col], errors="coerce")
    threshold = pd.to_numeric(candidates[threshold_col], errors="coerce").fillna(np.inf)
    return candidates.loc[(rank >= threshold).fillna(False)].copy()


def _accepted_overlap(base: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, Any]:
    base_keys = mstc._accepted_key_set(base)
    challenger_keys = mstc._accepted_key_set(challenger)
    union = base_keys | challenger_keys
    inter = base_keys & challenger_keys
    return {
        "intersection": int(len(inter)),
        "union": int(len(union)),
        "jaccard": float(len(inter) / len(union)) if union else 1.0,
        "base_only": int(len(base_keys - challenger_keys)),
        "challenger_only": int(len(challenger_keys - base_keys)),
    }


def _swap_pnl(base: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, float]:
    if base.empty and challenger.empty:
        return {
            "removed_count": 0.0,
            "added_count": 0.0,
            "removed_net_pnl": 0.0,
            "added_net_pnl": 0.0,
            "removed_winner_pnl": 0.0,
            "removed_loser_loss": 0.0,
            "added_winner_pnl": 0.0,
            "added_loser_loss": 0.0,
        }
    base = base.copy()
    challenger = challenger.copy()
    base["_key"] = list(mstc._normalised_decision_keys(base).itertuples(index=False, name=None))
    challenger["_key"] = list(mstc._normalised_decision_keys(challenger).itertuples(index=False, name=None))
    challenger_keys = set(challenger["_key"])
    base_keys = set(base["_key"])
    removed = base.loc[~base["_key"].isin(challenger_keys)]
    added = challenger.loc[~challenger["_key"].isin(base_keys)]
    removed_pnl = pd.to_numeric(removed.get("net_pnl"), errors="coerce").fillna(0.0)
    added_pnl = pd.to_numeric(added.get("net_pnl"), errors="coerce").fillna(0.0)
    return {
        "removed_count": float(len(removed)),
        "added_count": float(len(added)),
        "removed_net_pnl": float(removed_pnl.sum()),
        "added_net_pnl": float(added_pnl.sum()),
        "removed_winner_pnl": float(removed_pnl.clip(lower=0.0).sum()),
        "removed_loser_loss": float((-removed_pnl.clip(upper=0.0)).sum()),
        "added_winner_pnl": float(added_pnl.clip(lower=0.0).sum()),
        "added_loser_loss": float((-added_pnl.clip(upper=0.0)).sum()),
    }


def _accepted_pnl(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    if "net_pnl" in frame.columns:
        return pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0.0)
    if "net_return" in frame.columns:
        return pd.to_numeric(frame["net_return"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _timestamp_accepted_summary(
    accepted: pd.DataFrame,
    timestamps: pd.Series,
    *,
    prefix: str,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    out = pd.DataFrame({"timestamp": ts})
    base_cols = {
        f"{prefix}_trade_count": 0.0,
        f"{prefix}_net_pnl": 0.0,
        f"{prefix}_short_asset_net_pnl": 0.0,
        f"{prefix}_short_boll_net_pnl": 0.0,
        f"{prefix}_short_asset_trades": 0.0,
        f"{prefix}_short_boll_trades": 0.0,
        f"{prefix}_full_sl_rate": 0.0,
        f"{prefix}_timeout_rate": 0.0,
    }
    for col, value in base_cols.items():
        out[col] = value
    if accepted.empty:
        return out

    work = accepted.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return out
    work["_net_pnl"] = _accepted_pnl(work)
    if "head" not in work.columns:
        work["head"] = work.get("strategy_id", pd.Series(dtype=object)).astype(str).map(mstc._infer_head)
    exit_reason = work.get("simple_policy_exit_reason", pd.Series("", index=work.index)).astype(str).str.lower()
    work["_full_sl"] = exit_reason.str.contains("sl", regex=False).astype(float)
    work["_timeout"] = exit_reason.str.contains("timeout", regex=False).astype(float)

    total = (
        work.groupby("timestamp", observed=True)
        .agg(
            **{
                f"{prefix}_trade_count": ("_net_pnl", "size"),
                f"{prefix}_net_pnl": ("_net_pnl", "sum"),
                f"{prefix}_full_sl_rate": ("_full_sl", "mean"),
                f"{prefix}_timeout_rate": ("_timeout", "mean"),
            }
        )
        .reset_index()
    )
    by_head = (
        work.groupby(["timestamp", "head"], observed=True)
        .agg(head_net_pnl=("_net_pnl", "sum"), head_trades=("_net_pnl", "size"))
        .reset_index()
    )
    for head in ("short_asset", "short_boll"):
        head_rows = by_head.loc[by_head["head"].astype(str).eq(head), ["timestamp", "head_net_pnl", "head_trades"]]
        head_rows = head_rows.rename(
            columns={
                "head_net_pnl": f"{prefix}_{head}_net_pnl",
                "head_trades": f"{prefix}_{head}_trades",
            }
        )
        total = total.merge(head_rows, on="timestamp", how="left")

    merged = out[["timestamp"]].merge(total, on="timestamp", how="left")
    for col, value in base_cols.items():
        if col not in merged.columns:
            merged[col] = value
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(value)
    return merged


def _replay_fold_arm(
    *,
    arm: str,
    valid_broad: pd.DataFrame,
    train_deployable: pd.DataFrame,
    params: Any,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    decisions, _equity, metrics = replay_candidates(
        valid_broad,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(market_mode),
    )
    accepted = mstc._accepted_trades(valid_broad, decisions)
    summary = pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule=None)])
    by_head = mstc._by_head(arm, accepted)
    return summary, by_head, accepted


def _fold_mask(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.Series:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if start is None:
        return ts < end
    return (ts >= start) & (ts < end)


def _render_report(
    *,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    fold_delta: pd.DataFrame,
    overlap: pd.DataFrame,
) -> str:
    lines = [
        "# T1 Rank-Contract Pre-June Walk-Forward",
        "",
        "This report compares the measured timestamp-rank T1 contract against a fold-causal global-over-time rank reference. It keeps the policy, thresholds, active heads, EV mapping method, score column and auction fixed.",
        "",
        "## Contract",
        "",
        f"- Source candidates: `{manifest['inputs']['source_broad_candidates']}`",
        f"- Train deployable candidates: `{manifest['inputs']['train_deployable_candidates']}`",
        f"- Policy variant: `{manifest['policy_variant']}`",
        f"- Disabled heads: `{', '.join(manifest['disabled_heads'])}`",
        f"- Folds: `{manifest['fold_count']}`",
        f"- Embargo hours: `{manifest['embargo_hours']}`",
        "",
        "## Aggregate",
        "",
        "| arm | folds | total_trades | total_net_pnl | median_fold_net_pnl | q25_fold_net_pnl | positive_fold_share | mean_full_sl_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['folds'])} | {int(row['total_trades'])} | "
            f"{float(row['total_net_pnl']):.6f} | {float(row['median_fold_net_pnl']):.6f} | "
            f"{float(row['q25_fold_net_pnl']):.6f} | {float(row['positive_fold_share']):.6f} | "
            f"{float(row['mean_full_sl_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fold Delta: Global Minus Timestamp",
            "",
            "| fold | valid_start | valid_end | delta_net_pnl | delta_trades | delta_full_sl_rate | accepted_jaccard |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in fold_delta.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['valid_start']} | {row['valid_end']} | "
            f"{float(row['delta_net_pnl']):.6f} | {float(row['delta_trade_count']):.0f} | "
            f"{float(row['delta_full_sl_rate']):.6f} | {float(row['accepted_jaccard']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Accepted-Set Swap Totals",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    if overlap.empty:
        lines.append("| folds | 0 |")
    else:
        totals = {
            "base_only": float(overlap["base_only"].sum()),
            "challenger_only": float(overlap["challenger_only"].sum()),
            "removed_net_pnl": float(overlap["removed_net_pnl"].sum()),
            "added_net_pnl": float(overlap["added_net_pnl"].sum()),
            "removed_winner_pnl": float(overlap["removed_winner_pnl"].sum()),
            "removed_loser_loss": float(overlap["removed_loser_loss"].sum()),
            "added_winner_pnl": float(overlap["added_winner_pnl"].sum()),
            "added_loser_loss": float(overlap["added_loser_loss"].sum()),
        }
        for key, value in totals.items():
            lines.append(f"| {key} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A global rank contract should only replace the provisional timestamp-rank T1 contract if it wins across these pre-June folds and a later untouched matured period. This artifact is a validation input, not a production switch.",
            "",
            "## Outputs",
            "",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
            f"- Fold summary: `{manifest['outputs']['fold_summary']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Fold delta: `{manifest['outputs']['fold_delta']}`",
            f"- By head: `{manifest['outputs']['by_head']}`",
            f"- Overlap: `{manifest['outputs']['accepted_overlap']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_walkforward(args: argparse.Namespace) -> dict[str, Path]:
    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    broad = _load_candidates(args.source_broad_candidates, disabled_heads=disabled_heads)
    train_deployable_all = _load_candidates(args.train_deployable_candidates, disabled_heads=disabled_heads)
    broad["timestamp"] = pd.to_datetime(broad["timestamp"], utc=True, errors="coerce")
    train_deployable_all["timestamp"] = pd.to_datetime(train_deployable_all["timestamp"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(args.cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    broad = broad.loc[broad["timestamp"] < cutoff].copy()
    train_deployable_all = train_deployable_all.loc[train_deployable_all["timestamp"] < cutoff].copy()
    folds = _make_time_folds(
        broad["timestamp"],
        train_min_days=int(args.train_min_days),
        valid_days=int(args.valid_days),
        step_days=int(args.step_days),
        embargo_hours=int(args.embargo_hours),
    )
    if not folds:
        raise RuntimeError("no walk-forward folds were generated")
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)

    summaries: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    overlap_rows: list[dict[str, Any]] = []
    rank_diag_rows: list[dict[str, Any]] = []
    fold_delta_rows: list[dict[str, Any]] = []
    timestamp_utility_frames: list[pd.DataFrame] = []
    accepted_ledger_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_broad = broad.loc[_fold_mask(broad, None, fold.train_end_exclusive)].copy()
        valid_broad = broad.loc[_fold_mask(broad, fold.valid_start, fold.valid_end_exclusive)].copy()
        train_deployable = train_deployable_all.loc[
            _fold_mask(train_deployable_all, None, fold.train_end_exclusive)
        ].copy()
        if train_broad.empty or valid_broad.empty or train_deployable.empty:
            continue

        ts_valid = mstc._apply_rank_contract(valid_broad, "short_boll_timestamp_rank")
        ts_train_dep = mstc._apply_rank_contract(train_deployable, "short_boll_timestamp_rank")
        global_valid, global_valid_diag = _apply_fold_global_rank_reference(
            valid_broad,
            reference_train=train_broad,
            allow_missing=bool(args.allow_missing_global_rank),
        )
        global_train_dep, global_train_diag = _apply_fold_global_rank_reference(
            train_deployable,
            reference_train=train_broad,
            allow_missing=bool(args.allow_missing_global_rank),
        )

        fold_meta = {
            "fold": fold.fold,
            "train_start": fold.train_start.isoformat(),
            "train_end_exclusive": fold.train_end_exclusive.isoformat(),
            "valid_start": fold.valid_start.isoformat(),
            "valid_end": fold.valid_end_exclusive.isoformat(),
            "train_broad_rows": int(len(train_broad)),
            "valid_broad_rows": int(len(valid_broad)),
            "train_deployable_rows": int(len(train_deployable)),
        }
        ts_accepted = pd.DataFrame()
        global_accepted = pd.DataFrame()
        for arm, candidates, train_dep in (
            (TIMESTAMP_ARM, ts_valid, ts_train_dep),
            (GLOBAL_ARM, global_valid, global_train_dep),
        ):
            summary, by_head, accepted = _replay_fold_arm(
                arm=arm,
                valid_broad=candidates,
                train_deployable=train_dep,
                params=params,
                market_mode=str(args.market_mode),
            )
            for key, value in fold_meta.items():
                summary[key] = value
                if not by_head.empty:
                    by_head[key] = value
            summaries.append(summary)
            if not by_head.empty:
                by_head_frames.append(by_head)
            if bool(args.persist_fold_ledgers) and not accepted.empty:
                accepted_out = accepted.copy()
                accepted_out["arm"] = arm
                for key, value in fold_meta.items():
                    accepted_out[key] = value
                accepted_ledger_frames.append(accepted_out)
            if arm == TIMESTAMP_ARM:
                ts_summary = summary.iloc[0]
                ts_accepted = accepted
            else:
                global_summary = summary.iloc[0]
                global_accepted = accepted

        valid_timestamps = pd.to_datetime(valid_broad["timestamp"], utc=True, errors="coerce")
        ts_util = _timestamp_accepted_summary(ts_accepted, valid_timestamps, prefix="timestamp_rank")
        global_util = _timestamp_accepted_summary(global_accepted, valid_timestamps, prefix="global_rank")
        timestamp_utility = ts_util.merge(global_util, on="timestamp", how="outer").sort_values("timestamp")
        for key, value in fold_meta.items():
            timestamp_utility[key] = value
        timestamp_utility["timestamp_minus_global_net_pnl"] = (
            pd.to_numeric(timestamp_utility["timestamp_rank_net_pnl"], errors="coerce").fillna(0.0)
            - pd.to_numeric(timestamp_utility["global_rank_net_pnl"], errors="coerce").fillna(0.0)
        )
        timestamp_utility["global_minus_timestamp_net_pnl"] = -timestamp_utility[
            "timestamp_minus_global_net_pnl"
        ]
        timestamp_utility["timestamp_minus_global_short_boll_net_pnl"] = (
            pd.to_numeric(timestamp_utility["timestamp_rank_short_boll_net_pnl"], errors="coerce").fillna(0.0)
            - pd.to_numeric(timestamp_utility["global_rank_short_boll_net_pnl"], errors="coerce").fillna(0.0)
        )
        timestamp_utility_frames.append(timestamp_utility)

        overlap = _accepted_overlap(ts_accepted, global_accepted)
        swap = _swap_pnl(ts_accepted, global_accepted)
        overlap_rows.append({**fold_meta, **overlap, **swap})
        rank_diag_rows.append(
            {
                **fold_meta,
                "global_valid_missing_policy_rank_rows": global_valid_diag["missing_policy_rank_rows"],
                "global_valid_missing_auction_rank_rows": global_valid_diag["missing_auction_rank_rows"],
                "global_valid_ranked_rows": global_valid_diag["ranked_rows"],
                "global_train_missing_policy_rank_rows": global_train_diag["missing_policy_rank_rows"],
                "global_train_missing_auction_rank_rows": global_train_diag["missing_auction_rank_rows"],
                "global_train_ranked_rows": global_train_diag["ranked_rows"],
            }
        )
        fold_delta_rows.append(
            {
                **fold_meta,
                "delta_trade_count": float(global_summary["trade_count"] - ts_summary["trade_count"]),
                "delta_net_pnl": float(global_summary["net_pnl"] - ts_summary["net_pnl"]),
                "delta_gross_pnl": float(global_summary["gross_pnl"] - ts_summary["gross_pnl"]),
                "delta_cost_pnl": float(global_summary["cost_pnl"] - ts_summary["cost_pnl"]),
                "delta_full_sl_rate": float(global_summary["full_sl_rate"] - ts_summary["full_sl_rate"]),
                "delta_timeout_rate": float(global_summary["timeout_rate"] - ts_summary["timeout_rate"]),
                "delta_worst_24h_net_pnl": float(global_summary["worst_24h_net_pnl"] - ts_summary["worst_24h_net_pnl"]),
                "accepted_jaccard": float(overlap["jaccard"]),
            }
        )

    if not summaries:
        raise RuntimeError("all generated folds were empty")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summary = pd.concat(summaries, ignore_index=True)
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    overlap = pd.DataFrame(overlap_rows)
    rank_diag = pd.DataFrame(rank_diag_rows)
    fold_delta = pd.DataFrame(fold_delta_rows)
    timestamp_utility = (
        pd.concat(timestamp_utility_frames, ignore_index=True)
        if timestamp_utility_frames
        else pd.DataFrame()
    )
    accepted_ledger = (
        pd.concat(accepted_ledger_frames, ignore_index=True)
        if accepted_ledger_frames
        else pd.DataFrame()
    )
    aggregate_rows = []
    for arm, group in fold_summary.groupby("arm", sort=False):
        net = pd.to_numeric(group["net_pnl"], errors="coerce")
        trades = pd.to_numeric(group["trade_count"], errors="coerce").fillna(0)
        aggregate_rows.append(
            {
                "arm": arm,
                "folds": int(group["fold"].nunique()),
                "total_trades": int(trades.sum()),
                "total_net_pnl": float(net.sum()),
                "median_fold_net_pnl": float(net.median()),
                "q25_fold_net_pnl": float(net.quantile(0.25)),
                "positive_fold_share": float((net > 0.0).mean()),
                "mean_full_sl_rate": float(pd.to_numeric(group["full_sl_rate"], errors="coerce").mean()),
                "mean_timeout_rate": float(pd.to_numeric(group["timeout_rate"], errors="coerce").mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)

    paths = {
        "manifest": output_dir / "t1_rank_contract_walkforward_manifest.json",
        "fold_summary": output_dir / "rank_contract_walkforward_fold_summary.csv",
        "aggregate": output_dir / "rank_contract_walkforward_aggregate.csv",
        "fold_delta": output_dir / "rank_contract_walkforward_fold_delta.csv",
        "by_head": output_dir / "rank_contract_walkforward_by_head.csv",
        "accepted_overlap": output_dir / "rank_contract_walkforward_accepted_overlap.csv",
        "rank_diagnostics": output_dir / "rank_contract_walkforward_rank_diagnostics.csv",
        "timestamp_utility": output_dir / "rank_contract_walkforward_timestamp_utility.csv",
        "report": output_dir / "t1_rank_contract_walkforward_report.md",
    }
    if bool(args.persist_fold_ledgers):
        paths["accepted_trades"] = output_dir / "rank_contract_walkforward_accepted_trades.parquet"
    manifest = {
        "generated_by": "run_t1_rank_contract_walkforward",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "pre_june_rank_contract_validation",
        "arms": {
            TIMESTAMP_ARM: {
                "rank_contract": "short_boll_timestamp_rank",
                "rank_scope": "within_timestamp",
            },
            GLOBAL_ARM: {
                "rank_contract": "fold_causal_global_rank_reference",
                "rank_scope": "global_over_time",
                "fit_scope": "training_timestamps_only_per_fold",
            },
        },
        "policy_variant": str(args.policy_variant),
        "disabled_heads": sorted(disabled_heads),
        "active_heads": ["short_asset", "short_boll"],
        "fixed_policy_contract": {
            "score_path": "anchor_meta_calibrated_score",
            "active_score_column": "calibrated_score",
            "static_base_thresholds": True,
            "policy_variant": str(args.policy_variant),
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": sorted(disabled_heads),
            "auction": "global_auction",
            "ev_mapping": "hierarchical_ev_curves_fitted_on_train_deployable",
            "market_state_threshold_controller_active": False,
            "qfail_active": False,
            "native_reliability_blend_active": False,
            "headhealth_active": False,
            "rank_contract_is_the_only_arm_difference": True,
        },
        "cutoff_exclusive": cutoff.isoformat(),
        "train_min_days": int(args.train_min_days),
        "valid_days": int(args.valid_days),
        "step_days": int(args.step_days),
        "embargo_hours": int(args.embargo_hours),
        "fold_count": int(fold_summary["fold"].nunique()),
        "inputs": {
            "source_broad_candidates": str(args.source_broad_candidates),
            "source_broad_candidates_sha256": _sha256(args.source_broad_candidates),
            "train_deployable_candidates": str(args.train_deployable_candidates),
            "train_deployable_candidates_sha256": _sha256(args.train_deployable_candidates),
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": _sha256(args.policy_manifest),
            "policy_manifest_run_id": policy_payload.get("run_id"),
        },
        "leakage_contract": {
            "split_by_complete_timestamps": True,
            "global_rank_reference_uses_validation_rows": False,
            "global_rank_reference_uses_future_rows": False,
            "market_state_controller_active": False,
            "qfail_active": False,
            "rank_contract_is_the_only_arm_difference": True,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    fold_summary.to_csv(paths["fold_summary"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    fold_delta.to_csv(paths["fold_delta"], index=False)
    by_head.to_csv(paths["by_head"], index=False)
    overlap.to_csv(paths["accepted_overlap"], index=False)
    rank_diag.to_csv(paths["rank_diagnostics"], index=False)
    timestamp_utility.to_csv(paths["timestamp_utility"], index=False)
    if bool(args.persist_fold_ledgers):
        accepted_ledger.to_parquet(paths["accepted_trades"], index=False)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(
        _render_report(
            manifest=manifest,
            aggregate=aggregate,
            fold_delta=fold_delta,
            overlap=overlap,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-broad-candidates", type=Path, default=mstc.DEFAULT_TRAIN_BROAD)
    parser.add_argument("--train-deployable-candidates", type=Path, default=mstc.DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=mstc.DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--disable-heads", default="long_bars,long_dist")
    parser.add_argument("--cutoff", default="2026-06-15T04:00:00Z")
    parser.add_argument("--train-min-days", type=int, default=21)
    parser.add_argument("--valid-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--embargo-hours", type=int, default=48)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--allow-missing-global-rank", action="store_true")
    parser.add_argument(
        "--persist-fold-ledgers",
        action="store_true",
        help="Persist accepted trades by fold/arm for downstream state-response diagnostics.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_walkforward(args)
    print(f"Wrote rank-contract walk-forward report: {paths['report']}")


if __name__ == "__main__":
    main()
