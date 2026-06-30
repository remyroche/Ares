#!/usr/bin/env python3
"""Generate no-op replay parity evidence for exact-state size-action runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.run_exact_state_counterfactual_oracle import (
    _capture_snapshots,
    _decision_signature,
    _slice_from_timestamp,
)
from scripts.run_exact_state_size_action_learning import (
    DEFAULT_POLICY_MANIFEST,
    DEFAULT_TRAIN_BROAD,
    DEFAULT_TRAIN_DEPLOYABLE,
)
from scripts.run_global_portfolio_period_multiplier import _load_candidates, _load_policy_params
from scripts.run_global_portfolio_period_multiplier_walkforward import _build_folds, _timestamp_mask


def _timestamps_from_panel(panel: pd.DataFrame, fold_id: int, split: str) -> pd.DatetimeIndex:
    if not {"fold_id", "split", "timestamp"}.issubset(panel.columns):
        return pd.DatetimeIndex([], tz="UTC")
    mask = pd.to_numeric(panel["fold_id"], errors="coerce").eq(int(fold_id)) & panel["split"].astype(str).eq(split)
    values = pd.to_datetime(panel.loc[mask, "timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(values)


def _parity_rows_for_split(
    *,
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    timestamps: pd.DatetimeIndex,
    horizon_hours: int,
    market_mode: str,
    fold_id: int,
    split: str,
) -> list[dict[str, Any]]:
    if candidates.empty or len(timestamps) == 0:
        return []
    baseline_decisions, _equity, snapshots = _capture_snapshots(candidates, params, ev_curve, market_mode)
    rows: list[dict[str, Any]] = []
    for timestamp in timestamps:
        ts = pd.Timestamp(timestamp)
        state = snapshots.get(ts)
        if state is None:
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "split": split,
                    "timestamp": ts,
                    "noop_decision_signature_equal": False,
                    "noop_reason": "missing_snapshot",
                }
            )
            continue
        suffix = _slice_from_timestamp(candidates, ts, int(horizon_hours))
        if suffix.empty:
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "split": split,
                    "timestamp": ts,
                    "noop_decision_signature_equal": False,
                    "noop_reason": "empty_suffix",
                }
            )
            continue
        clone_decisions, _clone_equity, _clone_metrics = replay_candidates(
            suffix,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            initial_state=state,
            market_mode=market_mode,
        )
        baseline_sig = _decision_signature(baseline_decisions, ts)
        clone_sig = _decision_signature(clone_decisions, ts)
        rows.append(
            {
                "fold_id": int(fold_id),
                "split": split,
                "timestamp": ts,
                "noop_decision_signature_equal": bool(baseline_sig.equals(clone_sig)),
                "noop_open_positions": int(len(state.open_positions)),
                "noop_wallet": float(state.wallet),
                "noop_baseline_decisions": int(len(baseline_sig)),
                "noop_clone_decisions": int(len(clone_sig)),
                "noop_reason": "ok" if baseline_sig.equals(clone_sig) else "signature_mismatch",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing size-action run directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output no-op parity CSV.")
    parser.add_argument("--broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--horizon-hours", type=int, default=72)
    parser.add_argument("--embargo-hours", type=int, default=96)
    parser.add_argument("--min-train-hours", type=int, default=336)
    parser.add_argument("--fold-hours", type=int, default=168)
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument(
        "--splits",
        default="train,eval",
        help="Comma-separated exact-panel splits to verify. Use eval for held-out parity evidence.",
    )
    args = parser.parse_args()

    panel_path = args.run_dir / "size_action_exact_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"missing exact panel: {panel_path}")

    panel = pd.read_csv(panel_path, usecols=["fold_id", "split", "timestamp"])
    params, _policy_payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    broad = normalise_candidate_table(_load_candidates(args.broad_candidates))
    deployable = normalise_candidate_table(_load_candidates(args.deployable_candidates))
    folds = _build_folds(
        broad["timestamp"],
        min_train_hours=int(args.min_train_hours),
        fold_hours=int(args.fold_hours),
        embargo_hours=int(args.embargo_hours),
        max_folds=int(args.max_folds),
    )

    rows: list[dict[str, Any]] = []
    splits = {part.strip() for part in str(args.splits).split(",") if part.strip()}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_end = pd.Timestamp(fold["train_end"])
        eval_start = pd.Timestamp(fold["eval_start"])
        eval_end = pd.Timestamp(fold["eval_end"]) + pd.Timedelta(nanoseconds=1)
        ev_curve = fit_hierarchical_ev_curves(
            deployable.loc[_timestamp_mask(deployable, end=train_end + pd.Timedelta(nanoseconds=1))].copy()
        )
        if "train" in splits:
            train_candidates = broad.loc[_timestamp_mask(broad, end=train_end + pd.Timedelta(nanoseconds=1))].copy()
            rows.extend(
                _parity_rows_for_split(
                    candidates=train_candidates,
                    params=params,
                    ev_curve=ev_curve,
                    timestamps=_timestamps_from_panel(panel, fold_id, "train"),
                    horizon_hours=int(args.horizon_hours),
                    market_mode=args.market_mode,
                    fold_id=fold_id,
                    split="train",
                )
            )
        if "eval" in splits:
            eval_candidates = broad.loc[_timestamp_mask(broad, start=eval_start, end=eval_end)].copy()
            rows.extend(
                _parity_rows_for_split(
                    candidates=eval_candidates,
                    params=params,
                    ev_curve=ev_curve,
                    timestamps=_timestamps_from_panel(panel, fold_id, "eval"),
                    horizon_hours=int(args.horizon_hours),
                    market_mode=args.market_mode,
                    fold_id=fold_id,
                    split="eval",
                )
            )
        if rows:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    equal = bool(out["noop_decision_signature_equal"].fillna(False).astype(bool).all()) if not out.empty else False
    print(
        {
            "rows": int(len(out)),
            "all_equal": equal,
            "failures": int((~out["noop_decision_signature_equal"].fillna(False).astype(bool)).sum()) if not out.empty else 0,
            "out": str(args.out),
        }
    )


if __name__ == "__main__":
    main()
