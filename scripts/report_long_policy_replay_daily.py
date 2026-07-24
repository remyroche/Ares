#!/usr/bin/env python3
"""Replay a frozen long-only side/archetype policy and report daily metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    run_portfolio_policy_replay,
)
from extreme_price_movements.simple_policy_optimiser import (
    _attach_policy_archetype_column,
    simulate_and_score,
)
from scripts.ablate_simple_policy_exit_geometry import _load_bundles, _prepare_rows
from scripts.run_s52_side_archetype_simple_policy_optimiser import (
    _geometry_params_from_archetype_row,
    _params_from_parent_summary_row,
)


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _policy_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.get("strategy_id", "")), str(row.get("policy_archetype", "")))
        if all(key):
            lookup[key] = row.to_dict()
    return lookup


def _replay_group(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: Mapping[str, Any],
    size_power: float,
    cost_pct_per_side: float,
    policy_source: str,
) -> pd.DataFrame:
    params = dict(params)
    for name in (
        "max_concurrent_trades",
        "max_concurrent_per_asset",
        "max_new_entries_per_bar",
    ):
        params.pop(name, None)
    metrics = simulate_and_score(
        rows,
        *paths,
        cost_pct=float(cost_pct_per_side),
        size_power=float(size_power),
        max_concurrent_trades=1_000_000,
        max_concurrent_per_asset=1_000_000,
        max_new_entries_per_bar=1_000_000,
        **params,
    )
    selected = np.asarray(metrics.get("selected_mask", []), dtype=bool)
    if len(selected) != len(rows):
        raise ValueError("simulator selected-mask length does not match policy rows")
    out = rows.iloc[np.flatnonzero(selected)].copy().reset_index(drop=True)
    arrays = {
        "net_return_notional": metrics.get("net_returns", []),
        "gross_return_notional": metrics.get("gross_returns", []),
        "fee_return": metrics.get("fee_returns", []),
        "position_size": metrics.get("sizes", []),
        "exit_bar": metrics.get("exit_bars", []),
        "exit_reason": metrics.get("exit_reason", []),
        "entry_price": metrics.get("entry_prices", []),
        "exit_price": metrics.get("exit_prices", []),
        "expected_spread_bps": metrics.get("expected_spread_bps", []),
        "entry_half_spread_bps": metrics.get("entry_half_spread_bps", []),
        "exit_half_spread_bps": metrics.get("exit_spread_cost_bps", []),
    }
    for name, values in arrays.items():
        values = np.asarray(values)
        if len(values) != len(out):
            raise ValueError(
                f"simulator output {name} length does not match selected rows"
            )
        out[name] = values
    out["policy_source"] = str(policy_source)
    out["bankroll_pnl"] = pd.to_numeric(
        out["position_size"], errors="coerce"
    ) * pd.to_numeric(out["net_return_notional"], errors="coerce")
    signal_ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    decision_ts = pd.to_datetime(
        out.get("decision_timestamp", signal_ts + pd.Timedelta(hours=1)),
        utc=True,
        errors="coerce",
    )
    out["decision_timestamp"] = decision_ts
    out["exit_timestamp"] = decision_ts + pd.to_timedelta(
        pd.to_numeric(out["exit_bar"], errors="coerce").clip(lower=1), unit="m"
    )
    return out


def replay_policy(
    *,
    candidates: Path,
    parent_summary: Path,
    archetype_summary: Path,
    data_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rank: float,
    round_trip_cost_pct: float,
    prediction_provenance: str,
    geometry_source: str = "saved_policy",
) -> pd.DataFrame:
    rows = _prepare_rows(
        candidates,
        min_rank=float(min_rank),
        rank_score_col="rank_pct",
        rank_scope="global",
        regime_ev_calibration_artifact=Path("/nonexistent/regime_calibration.json"),
        regime_ev_feature_handoff=Path("/nonexistent/regime_features.parquet"),
        apply_regime_ev_calibration_artifact=False,
    )
    timestamp = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.loc[timestamp.ge(start) & timestamp.lt(end)].reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame()
    if not rows["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("long-only replay received non-long rows")
    bundles = _load_bundles(
        rows,
        data_root=str(data_root),
        market_mode="perps",
        path_len=1440,
        min_rows_per_strategy=1,
    )
    if len(bundles) != 1:
        raise ValueError(f"expected one long strategy bundle, found {len(bundles)}")
    bundle = bundles[0]
    parent_frame = pd.read_csv(parent_summary)
    parent_row = parent_frame.loc[
        parent_frame["strategy_id"].astype(str).eq(str(bundle.strategy_id))
    ]
    if len(parent_row) != 1:
        raise ValueError("could not identify exactly one long parent policy")
    parent_params, parent_size = _params_from_parent_summary_row(
        parent_row.iloc[0].to_dict()
    )
    archetypes = pd.read_csv(archetype_summary)
    lookup = _policy_lookup(archetypes)
    work = _attach_policy_archetype_column(
        bundle.rows.copy(), strategy_id=str(bundle.strategy_id)
    ).reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    if geometry_source == "candidate_label":
        geometry_columns = ["archetype_tp_r", "archetype_sl_r", "archetype_trail_r"]
        missing = [name for name in geometry_columns if name not in work.columns]
        if missing:
            raise ValueError(f"candidate label geometry is missing columns: {missing}")
        grouping = work.groupby(geometry_columns, sort=True, dropna=False).groups
    elif geometry_source == "saved_policy":
        grouping = work.groupby("policy_archetype", sort=True).groups
    else:
        raise ValueError(f"unsupported geometry source: {geometry_source}")

    for group_key, indices in grouping.items():
        index = np.asarray(list(indices), dtype=np.int64)
        if geometry_source == "candidate_label":
            tp_r, sl_r, trail_r = (float(value) for value in group_key)
            params = {
                "sl_mult": sl_r,
                "trailing_activation_mult": tp_r,
                "fixed_trailing_gap_mult": trail_r,
                "capital_preservation_enabled": False,
                "adverse_exit_enabled": False,
            }
            size_power = parent_size
            source = "candidate_archetype_label_geometry"
        else:
            archetype = str(group_key)
            local_row = lookup.get((str(bundle.strategy_id), archetype))
            if local_row is None:
                params, size_power = parent_params, parent_size
                source = "side_parent_fallback"
            else:
                params, size_power = _geometry_params_from_archetype_row(
                    local_row,
                    parent_params=parent_params,
                    parent_size_power=parent_size,
                )
                source = "side_archetype_shrunk_geometry"
        parts.append(
            _replay_group(
                work.iloc[index].copy().reset_index(drop=True),
                tuple(path[index] for path in bundle.paths),
                params=params,
                size_power=size_power,
                cost_pct_per_side=float(round_trip_cost_pct) / 2.0,
                policy_source=source,
            )
        )
    out = pd.concat(parts, ignore_index=True, copy=False)
    out["prediction_provenance"] = str(prediction_provenance)
    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def summarize(replay: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if replay.empty:
        return pd.DataFrame()
    frame = replay.copy()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frequency == "day":
        frame["period"] = ts.dt.strftime("%Y-%m-%d")
    elif frequency == "week":
        frame["period"] = (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.strftime(
            "%Y-%m-%d"
        )
    else:
        raise ValueError(f"unsupported frequency: {frequency}")
    frame["positive"] = pd.to_numeric(frame["net_return_notional"], errors="coerce").gt(
        0
    )
    frame["full_stop"] = frame["exit_reason"].astype(str).eq("full_sl")
    frame["timeout"] = frame["exit_reason"].astype(str).eq("timeout")
    frame["trailing"] = frame["exit_reason"].astype(str).eq("trailing")
    grouped = frame.groupby(
        ["prediction_provenance", "period"], sort=True, observed=True
    )
    out = grouped.agg(
        trades=("net_return_notional", "count"),
        symbols=("symbol", "nunique"),
        avg_net_return_notional=("net_return_notional", "mean"),
        sum_net_return_notional=("net_return_notional", "sum"),
        avg_gross_return_notional=("gross_return_notional", "mean"),
        avg_fee_return=("fee_return", "mean"),
        positive_rate=("positive", "mean"),
        full_stop_rate=("full_stop", "mean"),
        timeout_rate=("timeout", "mean"),
        trailing_rate=("trailing", "mean"),
        bankroll_pnl=("bankroll_pnl", "sum"),
    ).reset_index()
    return out


def complete_daily_calendar(
    daily: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    prediction_provenance: str,
) -> pd.DataFrame:
    """Represent every UTC date in the requested window, including no-trade days."""
    calendar = pd.DataFrame(
        {
            "period": pd.date_range(
                start=start.normalize(),
                end=(end_exclusive - pd.Timedelta(nanoseconds=1)).normalize(),
                freq="D",
                tz="UTC",
            ).strftime("%Y-%m-%d")
        }
    )
    calendar["prediction_provenance"] = str(prediction_provenance)
    if daily.empty:
        out = calendar
    else:
        out = calendar.merge(
            daily,
            on=["prediction_provenance", "period"],
            how="left",
            validate="one_to_one",
        )
    for name in ("trades", "symbols"):
        if name not in out:
            out[name] = 0
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0).astype(int)
    for name in (
        "avg_net_return_notional",
        "sum_net_return_notional",
        "avg_gross_return_notional",
        "avg_fee_return",
        "positive_rate",
        "full_stop_rate",
        "timeout_rate",
        "trailing_rate",
        "bankroll_pnl",
    ):
        if name not in out:
            out[name] = 0.0
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
    return out


def to_portfolio_candidates(
    replay: pd.DataFrame,
    *,
    base_strategy_threshold: float,
) -> pd.DataFrame:
    """Map simulated long trades into the canonical portfolio candidate contract.

    ``net_return_notional`` already includes the simple-policy 1% round-trip
    cost.  The portfolio replay consumes that precomputed net return directly;
    it must not create a second fee or spread haircut.
    """
    required = {
        "timestamp",
        "symbol",
        "side_name",
        "strategy_id",
        "rank_pct",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "net_return_notional",
        "gross_return_notional",
        "exit_reason",
    }
    missing = sorted(required.difference(replay.columns))
    if missing:
        raise ValueError(f"simulated replay is missing portfolio fields: {missing}")
    work = replay.copy()
    if not work["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("post-portfolio candidate adapter accepts long rows only")
    decision_timestamp = pd.to_datetime(
        work.get("decision_timestamp", work["timestamp"]), utc=True, errors="coerce"
    )
    exit_timestamp = pd.to_datetime(work["exit_timestamp"], utc=True, errors="coerce")
    holding_bars = np.ceil(
        (exit_timestamp - decision_timestamp).dt.total_seconds() / (15.0 * 60.0)
    ).clip(lower=1)
    net_return = pd.to_numeric(work["net_return_notional"], errors="coerce")
    gross_return = pd.to_numeric(work["gross_return_notional"], errors="coerce")
    fee_return = pd.to_numeric(
        work.get("fee_return", gross_return - net_return), errors="coerce"
    )
    if not np.allclose(
        (gross_return - net_return).to_numpy(dtype=float),
        fee_return.to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError("simulated gross/net/fee returns do not reconcile")
    rank = pd.to_numeric(work["rank_pct"], errors="coerce")
    out = pd.DataFrame(
        {
            # The original signal timestamp is retained separately for audit;
            # the canonical replay orders candidates at the executable decision.
            "timestamp": decision_timestamp,
            "signal_timestamp": pd.to_datetime(
                work["timestamp"], utc=True, errors="coerce"
            ),
            "symbol": work["symbol"].astype(str),
            "side": "long",
            "strategy_id": work["strategy_id"].astype(str),
            "normalized_rank_score": rank,
            "strategy_rank_pct": rank,
            "base_strategy_threshold": float(base_strategy_threshold),
            "calibrated_score": rank,
            "entry_price": pd.to_numeric(work["entry_price"], errors="coerce"),
            "exit_timestamp": exit_timestamp,
            "exit_price": pd.to_numeric(work["exit_price"], errors="coerce"),
            # These are copied, never recomputed from prices or costs.
            "net_return": net_return,
            "gross_return": gross_return,
            "fees_bps": fee_return * 10_000.0,
            "slippage_bps": 0.0,
            "expected_friction_bps": fee_return * 10_000.0,
            "price_gap_bps": 0.0,
            "liquidity_capacity_weight": 1.0,
            "holding_bars": holding_bars.astype(int),
            "simple_policy_exit_reason": work["exit_reason"].astype(str),
            "policy_archetype": work.get("policy_archetype", "missing"),
            "market_mode": "perps",
            "source_net_return_notional": net_return,
            "source_gross_return_notional": gross_return,
            "source_fee_return": fee_return,
        }
    )
    if (
        out[["timestamp", "exit_timestamp", "net_return", "gross_return"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("simulated replay has invalid timestamps or returns")
    return out


def summarize_portfolio_decisions(
    decisions: pd.DataFrame, frequency: str
) -> pd.DataFrame:
    """Summarize all candidate decisions while attributing PnL only to fills."""
    if decisions.empty:
        return pd.DataFrame()
    frame = decisions.copy()
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frequency == "day":
        frame["period"] = ts.dt.strftime("%Y-%m-%d")
    elif frequency == "week":
        frame["period"] = (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.strftime(
            "%Y-%m-%d"
        )
    else:
        raise ValueError(f"unsupported frequency: {frequency}")
    frame["calendar_day"] = ts.dt.strftime("%Y-%m-%d")
    frame["accepted"] = frame["accepted"].astype(bool)
    frame["rejected"] = ~frame["accepted"]
    frame["net_return"] = pd.to_numeric(
        frame.get("position_net_return"), errors="coerce"
    )
    frame["position_size"] = pd.to_numeric(frame["position_size"], errors="coerce")
    frame["bankroll_pnl"] = np.where(
        frame["accepted"], frame["position_size"] * frame["net_return"], 0.0
    )
    frame["full_stop"] = frame["accepted"] & frame["position_exit_reason"].astype(
        str
    ).eq("full_sl")
    frame["timeout"] = frame["accepted"] & frame["position_exit_reason"].astype(str).eq(
        "timeout"
    )
    grouped = frame.groupby("period", sort=True, observed=True)
    out = grouped.agg(
        candidates=("accepted", "size"),
        accepted_count=("accepted", "sum"),
        rejected_count=("rejected", "sum"),
        calendar_days=("calendar_day", "nunique"),
        accepted_notional=(
            "position_size",
            lambda values: values[frame.loc[values.index, "accepted"]].sum(),
        ),
        bankroll_pnl=("bankroll_pnl", "sum"),
        full_stop_count=("full_stop", "sum"),
        timeout_count=("timeout", "sum"),
    ).reset_index()
    out["trades_per_day"] = out["accepted_count"] / out["calendar_days"].clip(lower=1)
    out["notional_net_return_per_trade"] = out["bankroll_pnl"] / out[
        "accepted_notional"
    ].replace(0.0, np.nan)
    out["full_stop_rate"] = out["full_stop_count"] / out["accepted_count"].replace(
        0, np.nan
    )
    out["timeout_rate"] = out["timeout_count"] / out["accepted_count"].replace(
        0, np.nan
    )
    return out


def run_post_portfolio_replay(
    replay: pd.DataFrame,
    *,
    output_dir: Path,
    base_strategy_threshold: float,
    fixed_policy_config: Path | None,
    optimize_end_exclusive: pd.Timestamp | None,
    ev_curve_end_exclusive: pd.Timestamp | None,
    evaluate_start: pd.Timestamp | None,
    max_evaluations: int | None,
) -> dict[str, Any]:
    """Run a frozen or temporally separated canonical portfolio replay."""
    candidates = to_portfolio_candidates(
        replay, base_strategy_threshold=float(base_strategy_threshold)
    )
    root = output_dir / "post_portfolio_replay"
    root.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(root / "portfolio_candidates.parquet", index=False)
    evaluation = candidates
    mode: str
    fit_report: dict[str, Any] | None = None
    if fixed_policy_config is not None:
        if optimize_end_exclusive is not None:
            raise ValueError(
                "fixed policy config cannot be combined with portfolio optimisation"
            )
        if ev_curve_end_exclusive is None:
            raise ValueError(
                "fixed policy replay requires --portfolio-ev-curve-end-exclusive "
                "to keep EV ordering fit separate from the evaluation window"
            )
        evaluation_start = evaluate_start or ev_curve_end_exclusive
        if evaluation_start < ev_curve_end_exclusive:
            raise ValueError("portfolio evaluation may not overlap the EV-curve fit")
        evaluation = candidates.loc[candidates["timestamp"].ge(evaluation_start)].copy()
        policy_config = Path(fixed_policy_config)
        mode = "fixed_policy_config"
        ev_curve_candidates = candidates.loc[
            candidates["timestamp"].lt(ev_curve_end_exclusive)
        ].copy()
        if ev_curve_candidates.empty:
            raise ValueError("portfolio EV-curve fit window contains no candidates")
    else:
        if optimize_end_exclusive is None:
            raise ValueError(
                "post-portfolio replay requires --portfolio-fixed-policy-config or "
                "--portfolio-optimize-end-exclusive"
            )
        evaluation_start = evaluate_start or optimize_end_exclusive
        if evaluation_start < optimize_end_exclusive:
            raise ValueError(
                "portfolio evaluation may not overlap the policy-fit window"
            )
        fit_candidates = candidates.loc[
            candidates["timestamp"].lt(optimize_end_exclusive)
        ].copy()
        evaluation = candidates.loc[candidates["timestamp"].ge(evaluation_start)].copy()
        if fit_candidates.empty or evaluation.empty:
            raise ValueError(
                "portfolio fit and evaluation windows must both contain candidates"
            )
        fit_path = root / "policy_fit_candidates.parquet"
        fit_candidates.to_parquet(fit_path, index=False)
        fit_report = run_portfolio_policy_replay(
            data_root=str(root),
            run_id="long_policy_post_portfolio",
            market_mode="perps",
            candidate_path=fit_path,
            output_dir=root / "policy_fit",
            max_evaluations=max_evaluations,
            persist_live_artifacts=False,
        )
        policy_config = root / "policy_fit" / "optimized_portfolio_policy_config.json"
        mode = "pre_cutoff_optimisation_then_holdout"
        ev_curve_candidates = fit_candidates
    if evaluation.empty:
        raise ValueError("portfolio evaluation window contains no candidates")
    evaluation_path = root / "portfolio_evaluation_candidates.parquet"
    ev_curve_path = root / "portfolio_ev_curve_candidates.parquet"
    evaluation.to_parquet(evaluation_path, index=False)
    ev_curve_candidates.to_parquet(ev_curve_path, index=False)
    evaluation_report = run_portfolio_policy_replay(
        data_root=str(root),
        run_id="long_policy_post_portfolio",
        market_mode="perps",
        candidate_path=evaluation_path,
        output_dir=root / "portfolio_replay",
        fixed_policy_config_path=policy_config,
        ev_curve_candidate_path=ev_curve_path,
        persist_live_artifacts=False,
    )
    decisions = pd.read_parquet(
        root / "portfolio_replay" / "per_candidate_replay_decisions.parquet"
    )
    summarize_portfolio_decisions(decisions, "day").to_csv(
        root / "portfolio_daily_metrics.csv", index=False
    )
    summarize_portfolio_decisions(decisions, "week").to_csv(
        root / "portfolio_weekly_metrics.csv", index=False
    )
    return {
        "mode": mode,
        "candidate_rows": int(len(candidates)),
        "evaluation_candidate_rows": int(len(evaluation)),
        "fixed_policy_config": str(policy_config),
        "fit_report": fit_report,
        "evaluation_report": evaluation_report,
        "cost_contract": "precomputed net_return is passed unchanged to canonical replay; no fee or spread is subtracted by this adapter",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--archetype-summary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--min-rank", type=float, default=0.90)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.003)
    parser.add_argument("--prediction-provenance", required=True)
    parser.add_argument(
        "--geometry-source",
        choices=("saved_policy", "candidate_label"),
        default="saved_policy",
    )
    parser.add_argument(
        "--post-portfolio-replay",
        action="store_true",
        help="Replay the already simulated rows through the canonical portfolio auction.",
    )
    parser.add_argument(
        "--portfolio-fixed-policy-config",
        type=Path,
        default=None,
        help="Frozen canonical portfolio config used for the evaluation replay.",
    )
    parser.add_argument(
        "--portfolio-optimize-end-exclusive",
        default=None,
        help="UTC cutoff: fit only rows before it, then replay the later holdout.",
    )
    parser.add_argument(
        "--portfolio-evaluate-start",
        default=None,
        help="Optional UTC start for the fixed-policy or post-fit evaluation window.",
    )
    parser.add_argument(
        "--portfolio-ev-curve-end-exclusive",
        default=None,
        help="UTC cutoff for the frozen-policy EV ordering fit; required with a fixed config.",
    )
    parser.add_argument(
        "--portfolio-base-strategy-threshold",
        type=float,
        default=None,
        help="Canonical candidate threshold; defaults to --min-rank.",
    )
    parser.add_argument(
        "--portfolio-max-evaluations",
        type=int,
        default=25,
        help="Bounded canonical optimisation evaluations for the pre-cutoff fit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replay = replay_policy(
        candidates=args.candidates,
        parent_summary=args.parent_summary,
        archetype_summary=args.archetype_summary,
        data_root=args.data_root,
        start=_utc(args.start),
        end=_utc(args.end_exclusive),
        min_rank=float(args.min_rank),
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        prediction_provenance=str(args.prediction_provenance),
        geometry_source=str(args.geometry_source),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay.to_parquet(args.output_dir / "per_trade_replay.parquet", index=False)
    complete_daily_calendar(
        summarize(replay, "day"),
        start=_utc(args.start),
        end_exclusive=_utc(args.end_exclusive),
        prediction_provenance=str(args.prediction_provenance),
    ).to_csv(args.output_dir / "daily_metrics.csv", index=False)
    summarize(replay, "week").to_csv(
        args.output_dir / "weekly_metrics.csv", index=False
    )
    portfolio_report: dict[str, Any] | None = None
    if args.post_portfolio_replay:
        portfolio_report = run_post_portfolio_replay(
            replay,
            output_dir=args.output_dir,
            base_strategy_threshold=(
                float(args.portfolio_base_strategy_threshold)
                if args.portfolio_base_strategy_threshold is not None
                else float(args.min_rank)
            ),
            fixed_policy_config=args.portfolio_fixed_policy_config,
            optimize_end_exclusive=(
                _utc(args.portfolio_optimize_end_exclusive)
                if args.portfolio_optimize_end_exclusive is not None
                else None
            ),
            ev_curve_end_exclusive=(
                _utc(args.portfolio_ev_curve_end_exclusive)
                if args.portfolio_ev_curve_end_exclusive is not None
                else None
            ),
            evaluate_start=(
                _utc(args.portfolio_evaluate_start)
                if args.portfolio_evaluate_start is not None
                else None
            ),
            max_evaluations=(
                int(args.portfolio_max_evaluations)
                if args.portfolio_max_evaluations is not None
                else None
            ),
        )
    manifest = {
        "schema": "long_policy_replay_daily_v2",
        "candidates": str(args.candidates),
        "parent_summary": str(args.parent_summary),
        "archetype_summary": str(args.archetype_summary),
        "prediction_provenance": str(args.prediction_provenance),
        "geometry_source": str(args.geometry_source),
        "start": _utc(args.start).isoformat(),
        "end_exclusive": _utc(args.end_exclusive).isoformat(),
        "min_rank": float(args.min_rank),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_contract": (
            f"{0.5 * float(args.round_trip_cost_pct):.6f} per side exactly once; "
            "per-symbol p90 full spread is split across executable entry/exit prices"
        ),
        "portfolio_constraints_applied": False,
        "post_portfolio_replay": portfolio_report,
        "rows": int(len(replay)),
    }
    if portfolio_report is not None:
        manifest["portfolio_constraints_applied"] = True
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
