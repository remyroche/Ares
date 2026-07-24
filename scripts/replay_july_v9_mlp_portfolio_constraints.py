#!/usr/bin/env python3
"""Replay corrected July V9+MLP admissions through the frozen portfolio policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.simple_policy_optimiser import (
    _with_policy_spread_cost_columns,
)
from scripts.backfill_complete_july_meta_predictions import (
    _capture_for_policy_keys,
    _load_json,
)


DEFAULT_PREDICTIONS = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "july_complete_01_12_v9_mlp_strict_consistent_parity_20260713/"
    "july_08_10_complete_predictions.parquet"
)
DEFAULT_LABELS = Path(
    "data_perp/artifacts/20260711_s59_h5_july_trailing_cost100bps_labels/labels"
)
DEFAULT_POLICY = Path(
    "data_perp/artifacts/"
    "s59_s52_frozen_inference_bundle_v9_tail95_mlp_hierev_20260713/"
    "policy_params/optimized_portfolio_policy_config.json"
)
DEFAULT_EV_REFERENCE = DEFAULT_POLICY.parent / "threshold_basis_reference_candidates.parquet"
DEFAULT_OUTPUT = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "july_01_12_v9_mlp_portfolio_replay_20260713"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--ev-reference", type=Path, default=DEFAULT_EV_REFERENCE)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--rank-threshold", type=float, default=0.90)
    parser.add_argument(
        "--rank-column",
        default="expected_ev_rank_score",
        help="Prediction column used for admission, ranking, and portfolio priority.",
    )
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _attach_label_replay_context(rows: pd.DataFrame, labels_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    months = sorted(
        pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
        .dropna()
        .dt.to_period("M")
        .astype(str)
        .unique()
    )
    for side in ("long", "short"):
        for month in months:
            year, month_number = month.split("-")
            path = labels_dir / f"train_global_{side}_5_{year}_{month_number}.parquet"
            if not path.exists():
                continue
            part = pd.read_parquet(
                path,
                columns=[
                    "__ts__",
                    "__symbol__",
                    "__barrier_pct__",
                    "__first_touch_bar__",
                    "__first_touch_stop__",
                    "__first_touch_timeout__",
                ],
            )
            part["side_name"] = side
            part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
            parts.append(part)
    if not parts:
        raise FileNotFoundError(
            f"No monthly label files found under {labels_dir} for months={months}"
        )
    history = pd.concat(parts, ignore_index=True, copy=False)
    exact = history.drop(columns="__barrier_pct__")
    out = rows.merge(
        exact,
        on=["__ts__", "__symbol__", "side_name"],
        how="left",
        validate="one_to_one",
    )
    out_parts: list[pd.DataFrame] = []
    for side, group in out.groupby("side_name", sort=False):
        right = history.loc[
            history["side_name"].eq(side),
            ["__ts__", "__symbol__", "__barrier_pct__"],
        ]
        group = group.sort_values(["__ts__", "__symbol__"], kind="stable")
        right = right.sort_values(["__ts__", "__symbol__"], kind="stable")
        merged = pd.merge_asof(
            group,
            right,
            on="__ts__",
            by="__symbol__",
            direction="backward",
            allow_exact_matches=True,
        )
        out_parts.append(merged)
    out = pd.concat(out_parts, ignore_index=True, copy=False)
    out["__barrier_pct__"] = (
        pd.to_numeric(out["__barrier_pct__"], errors="coerce")
        .fillna(0.02)
        .astype(np.float32)
    )
    return out


def _historical_ev_curve(path: Path) -> dict:
    reference = pd.read_parquet(path)
    reference["timestamp"] = pd.to_datetime(reference["timestamp"], utc=True, errors="coerce")
    reference = reference.loc[reference["timestamp"] < pd.Timestamp("2026-07-01", tz="UTC")].copy()
    reference["normalized_rank_score"] = pd.to_numeric(reference["rank_pct"], errors="coerce")
    reference["net_return"] = pd.to_numeric(reference["ret_net_notional"], errors="coerce")
    reference["base_strategy_threshold"] = 0.90
    reference["entry_price"] = 1.0
    reference["exit_price"] = 1.0
    reference["exit_timestamp"] = reference["timestamp"] + pd.Timedelta(minutes=15)
    reference["gross_return"] = reference["net_return"] + 0.01
    reference["holding_bars"] = 1.0
    reference["simple_policy_exit_reason"] = "historical_reference"
    return fit_hierarchical_ev_curves(reference)


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    predictions["__ts__"] = pd.to_datetime(predictions["__ts__"], utc=True, errors="coerce")
    if args.rank_column not in predictions.columns:
        raise KeyError(
            f"Rank column {args.rank_column!r} is absent from {args.predictions}"
        )
    replay_rank = pd.to_numeric(predictions[args.rank_column], errors="coerce")
    admitted = predictions.loc[
        replay_rank >= float(args.rank_threshold)
    ].copy()
    admitted["__portfolio_replay_rank__"] = replay_rank.loc[admitted.index].astype(
        np.float32
    )
    admitted = _attach_label_replay_context(admitted, args.labels_dir)
    policy_manifest = _load_json(args.labels_dir / "side_archetype_label_manifest.json")

    candidate_parts: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}
    for side, rows in admitted.groupby("side_name", sort=True):
        rows = rows.reset_index(drop=True)
        needs_capture = pd.to_numeric(
            rows["__first_touch_bar__"], errors="coerce"
        ).isna()
        capture_rows = rows.loc[needs_capture].reset_index(drop=True)
        capture, stats = _capture_for_policy_keys(
            capture_rows,
            side=str(side),
            policy_keys=capture_rows["archetype_policy_key"],
            policy_manifest=policy_manifest,
            data_root=args.data_root,
            path_len=int(args.path_len),
            min_path_coverage=0.0,
            allow_partial_paths=True,
        )
        capture_valid = pd.to_numeric(
            capture.get("capture_valid_path"), errors="coerce"
        ).gt(0.5)
        capture_bar = pd.Series(np.nan, index=rows.index, dtype=np.float64)
        capture_stop = pd.Series(np.nan, index=rows.index, dtype=np.float64)
        capture_timeout = pd.Series(np.nan, index=rows.index, dtype=np.float64)
        capture_valid_full = pd.Series(False, index=rows.index, dtype=bool)
        capture_eligible_full = pd.Series(False, index=rows.index, dtype=bool)
        capture_resolved_full = pd.Series(False, index=rows.index, dtype=bool)
        capture_net = pd.Series(np.nan, index=rows.index, dtype=np.float64)
        capture_gross = pd.Series(np.nan, index=rows.index, dtype=np.float64)
        capture_positions = np.flatnonzero(needs_capture.to_numpy())
        if len(capture_positions):
            capture_bar.iloc[capture_positions] = pd.to_numeric(
                capture["first_touch_bar"], errors="coerce"
            ).to_numpy()
            capture_stop.iloc[capture_positions] = pd.to_numeric(
                capture["capture_stop"], errors="coerce"
            ).to_numpy()
            capture_timeout.iloc[capture_positions] = pd.to_numeric(
                capture["capture_timeout"], errors="coerce"
            ).to_numpy()
            capture_valid_full.iloc[capture_positions] = capture_valid.to_numpy()
            capture_eligible_full.iloc[capture_positions] = pd.to_numeric(
                capture["capture_eligible"], errors="coerce"
            ).gt(0.5).to_numpy()
            capture_resolved_full.iloc[capture_positions] = pd.to_numeric(
                capture["capture_outcome_resolved"], errors="coerce"
            ).gt(0.5).to_numpy()
            capture_net.iloc[capture_positions] = pd.to_numeric(
                capture["capture_net"], errors="coerce"
            ).to_numpy()
            capture_gross.iloc[capture_positions] = pd.to_numeric(
                capture["capture_gross"], errors="coerce"
            ).to_numpy()
        label_bar = pd.to_numeric(rows["__first_touch_bar__"], errors="coerce")
        geometry_eligible = capture_valid_full & capture_eligible_full
        executable = label_bar.notna() | (geometry_eligible & capture_resolved_full)
        excluded_non_executable = int((needs_capture & ~geometry_eligible).sum())
        excluded_unresolved = int(
            (needs_capture & geometry_eligible & ~capture_resolved_full).sum()
        )
        rows = rows.loc[executable].reset_index(drop=True)
        label_bar = label_bar.loc[executable].reset_index(drop=True)
        capture_bar = capture_bar.loc[executable].reset_index(drop=True)
        capture_stop = capture_stop.loc[executable].reset_index(drop=True)
        capture_timeout = capture_timeout.loc[executable].reset_index(drop=True)
        capture_net = capture_net.loc[executable].reset_index(drop=True)
        capture_gross = capture_gross.loc[executable].reset_index(drop=True)
        holding = label_bar.fillna(capture_bar)
        historical = rows.loc[label_bar.notna()].copy()
        historical["_holding"] = label_bar.loc[label_bar.notna()]
        archetype_median = historical.groupby("archetype_policy_key", observed=True)[
            "_holding"
        ].median()
        fallback = rows["archetype_policy_key"].map(archetype_median)
        side_median = float(label_bar.median()) if label_bar.notna().any() else 8.0
        holding = holding.fillna(fallback).fillna(side_median).clip(1, args.path_len)
        holding_source = np.where(
            label_bar.notna(),
            "materialized_label",
            np.where(capture_bar.notna(), "replayed_path", "archetype_median_imputed"),
        )
        net = pd.to_numeric(rows["ev_after_1pct"], errors="coerce")
        replayed = label_bar.isna()
        net.loc[replayed] = capture_net.loc[replayed]
        gross = net + 0.01
        gross.loc[replayed] = capture_gross.loc[replayed]
        outcome_source = np.where(
            replayed, "kraken_15m_recomputed", "materialized_label"
        )
        side_sign = 1.0 if side == "long" else -1.0
        stop = pd.to_numeric(rows["__first_touch_stop__"], errors="coerce").fillna(
            capture_stop
        )
        timeout = pd.to_numeric(
            rows["__first_touch_timeout__"], errors="coerce"
        ).fillna(capture_timeout)
        exit_reason = np.select(
            [
                stop.gt(0.5),
                timeout.gt(0.5),
            ],
            ["stop", "timeout"],
            default="trailing",
        )
        policy_archetype = str(side) + "__" + rows["archetype_policy_key"].astype(str)
        rank = pd.to_numeric(rows["__portfolio_replay_rank__"], errors="coerce")
        candidate_parts.append(
            pd.DataFrame(
                {
                    "timestamp": rows["__ts__"],
                    "symbol": rows["__symbol__"].astype(str),
                    "side": side_sign,
                    "side_name": str(side),
                    "strategy_id": f"{side}_s52_meta_threshold_handoff",
                    "policy_archetype": policy_archetype,
                    "local_side_archetype": policy_archetype,
                    "normalized_rank_score": rank,
                    "strategy_rank_pct": rank,
                    "base_strategy_threshold": float(args.rank_threshold),
                    "calibrated_score": rank,
                    "barrier_pct": rows["__barrier_pct__"],
                    "entry_price": 1.0,
                    "exit_timestamp": rows["__ts__"] + pd.to_timedelta(holding * 15.0, unit="m"),
                    "exit_price": 1.0 + side_sign * gross,
                    "net_return": net,
                    "gross_return": gross,
                    "holding_bars": holding,
                    "holding_source": holding_source,
                    "outcome_source": outcome_source,
                    "simple_policy_exit_reason": exit_reason,
                    "fees_bps": 100.0,
                    "slippage_bps": 0.0,
                    "expected_friction_bps": 100.0,
                    "price_gap_bps": 0.0,
                    "liquidity_capacity_weight": 1.0,
                }
            )
        )
        coverage[str(side)] = {
            **stats,
            "label_exact_holding_rows": int(label_bar.notna().sum()),
            "tail_capture_requested_rows": int(needs_capture.sum()),
            "tail_capture_valid_rows": int(capture_valid.sum()),
            "excluded_non_executable_rows": excluded_non_executable,
            "excluded_unresolved_rows": excluded_unresolved,
            "imputed_holding_rows": int((label_bar.isna() & capture_bar.isna()).sum()),
            "holding_imputation": "none_after_geometry_eligibility_filter",
        }

    raw_candidates = pd.concat(candidate_parts, ignore_index=True)
    raw_candidates = _with_policy_spread_cost_columns(
        raw_candidates,
        market_mode="perps",
    )
    spread_adjustment_bps = (
        pd.to_numeric(raw_candidates["spread_cost_bps"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        + pd.to_numeric(raw_candidates["exit_spread_cost_bps"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    raw_candidates["net_return_before_spread"] = pd.to_numeric(
        raw_candidates["net_return"], errors="coerce"
    )
    raw_candidates["spread_adjustment_bps"] = spread_adjustment_bps.astype(
        np.float32
    )
    raw_candidates["net_return"] = (
        raw_candidates["net_return_before_spread"]
        - spread_adjustment_bps / 10_000.0
    )
    raw_candidates["expected_friction_bps"] = (
        pd.to_numeric(raw_candidates["fees_bps"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        + spread_adjustment_bps
    ).astype(np.float32)
    candidates = normalise_candidate_table(raw_candidates)
    params = load_portfolio_policy_params(args.policy_config)
    ev_curve = _historical_ev_curve(args.ev_reference)
    decisions, equity, summary = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    accepted = decisions.loc[decisions["accepted"]].copy()
    accepted["day"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    first_day = admitted["__ts__"].min().normalize()
    last_day = admitted["__ts__"].max().normalize()
    daily = (
        accepted.groupby("day", observed=True)
        .agg(
            trades=("accepted", "size"),
            mean_net_ev_per_trade=("position_net_return", "mean"),
            sum_notional_net_ev=("position_net_return", "sum"),
            mean_position_size=("position_size", "mean"),
        )
        .reindex(pd.date_range(first_day, last_day, freq="D").strftime("%Y-%m-%d"))
        .fillna(0.0)
        .reset_index(names="day")
    )
    daily["bankroll_pnl"] = daily["day"].map(
        accepted.assign(
            bankroll_pnl=lambda x: x["position_size"] * x["position_net_return"]
        ).groupby("day", observed=True)["bankroll_pnl"].sum()
    ).fillna(0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output_dir / "portfolio_candidates.parquet", index=False)
    decisions.to_parquet(args.output_dir / "portfolio_decisions.parquet", index=False)
    equity.to_parquet(args.output_dir / "portfolio_equity.parquet", index=False)
    daily.to_csv(args.output_dir / "daily_metrics.csv", index=False)
    manifest = {
        "schema": "july_v9_mlp_frozen_portfolio_replay_v1",
        "predictions": str(args.predictions),
        "policy_config": str(args.policy_config),
        "ev_reference": str(args.ev_reference),
        "ev_reference_cutoff_exclusive": "2026-07-01T00:00:00Z",
        "rank_threshold": float(args.rank_threshold),
        "rank_column": str(args.rank_column),
        "round_trip_cost": 0.01,
        "spread_cost": "symbol_baseline_full_spread_entry_plus_exit",
        "mean_spread_adjustment_bps": float(spread_adjustment_bps.mean()),
        "mean_expected_friction_bps": float(
            pd.to_numeric(candidates["expected_friction_bps"], errors="coerce").mean()
        ),
        "admitted_rows": int(len(admitted)),
        "replay_candidate_rows": int(len(candidates)),
        "accepted_rows": int(len(accepted)),
        "path_coverage": coverage,
        "portfolio_summary": summary,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(daily.to_string(index=False))
    print(json.dumps({k: manifest[k] for k in ("admitted_rows", "replay_candidate_rows", "accepted_rows")}, indent=2))


if __name__ == "__main__":
    main()
