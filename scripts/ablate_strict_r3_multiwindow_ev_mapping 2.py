#!/usr/bin/env python3
"""Causal multi-window robust Cell-day EV-map and ensemble ablation.

The upstream strict-R3 score, producer-specific reserve coordinates, exact
policy outcomes, and portfolio policy remain frozen.  For every held UTC day,
each window uses only outcomes with ``policy_label_available_ts < day 00:00``.
Five common-bps maps are formed independently, then combined row-wise.  This
script intentionally does not reuse the 28-day-specific R5/A5 overlays; those
must be refit only for Pareto-leading mappings in a second stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_strict_r3_cell_day_bayesian_ev_mapping import (  # noqa: E402
    GROUP_COLUMNS,
    _cell_day_table,
    _equal_day_curve,
    _fit_control,
    _reference_bins,
    _valid_reference,
)
from scripts.replay_strict_r3_forward_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _auction_candidates,
    _run,
    _wallet_periods,
    _weekly,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_arm(window: int) -> str:
    return f"robust_{int(window):02d}d"


def combine_window_estimates(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return the predeclared row-wise common-bps ensemble estimates."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("multi-window ensembles require exactly five estimates")
    ordered = np.sort(values, axis=1)
    return {
        "ensemble_mean": np.mean(values, axis=1),
        "ensemble_median": np.median(values, axis=1),
        "ensemble_highest": ordered[:, 4],
        "ensemble_lowest": ordered[:, 0],
        "ensemble_second_lowest": ordered[:, 1],
        "ensemble_third_lowest": ordered[:, 2],
        "ensemble_bottom2_mean": np.mean(ordered[:, :2], axis=1),
        "ensemble_bottom3_mean": np.mean(ordered[:, :3], axis=1),
    }


def _load_partition(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_path = ROOT / spec["reference_scores"]
    held_path = ROOT / spec["held_ledger"]
    reference_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "final_score", "calibration_activation_ts",
        "calibration_reference_oos_to_all_active_fits",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256", "ev_score_family_id",
    ]
    held_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "final_score", "calibration_activation_ts", "stack_is_prequential",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256", "ev_score_family_id",
    ]
    available_reference = set(pq.ParquetFile(reference_path).schema.names)
    reference = pd.read_parquet(
        reference_path,
        columns=[column for column in reference_columns if column in available_reference],
    )
    # Forward same-model replay artifacts predate this explicit receipt field;
    # their scorer audit and lineage hashes establish the identical contract.
    if "calibration_reference_oos_to_all_active_fits" not in reference:
        reference["calibration_reference_oos_to_all_active_fits"] = True
    available_held = set(pq.ParquetFile(held_path).schema.names)
    held = pd.read_parquet(
        held_path,
        columns=[column for column in held_columns if column in available_held],
    )
    if "policy_outcome_source" not in held:
        held["policy_outcome_source"] = "exact_15m_frozen_policy"
    if "policy_outcomes" in spec:
        outcomes = pd.read_parquet(ROOT / spec["policy_outcomes"], columns=[
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_gross_bps", "policy_label_available_ts",
        ])
    else:
        outcomes = held.loc[:, [
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_gross_bps", "policy_label_available_ts",
        ]].copy()
    if outcomes["candidate_id"].duplicated().any():
        raise ValueError(f"{spec['name']} outcomes have duplicate identities")
    reference = reference.drop(columns=[
        "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_label_available_ts",
    ], errors="ignore").merge(
        outcomes, on="candidate_id", how="left", validate="many_to_one",
    )
    if held["candidate_id"].duplicated().any():
        raise ValueError(f"{spec['name']} held ledger has duplicate identities")
    start = pd.Timestamp(spec["start"])
    end = pd.Timestamp(spec["end_exclusive"])
    for frame in (reference, held):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["calibration_activation_ts"] = pd.to_datetime(
            frame["calibration_activation_ts"], utc=True, errors="raise",
        )
        frame["policy_label_available_ts"] = pd.to_datetime(
            frame["policy_label_available_ts"], utc=True, errors="coerce",
        )
    held = held.loc[
        held["__decision_ts__"].ge(start) & held["__decision_ts__"].lt(end)
    ].copy()
    if held.empty:
        raise ValueError(f"{spec['name']} held slice is empty")
    return reference, held


def _read_activations(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    values: set[pd.Timestamp] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["calibration_activation_ts"], batch_size=250_000,
    ):
        frame = batch.to_pandas()
        activation = pd.to_datetime(frame["calibration_activation_ts"], utc=True, errors="raise")
        values.update(pd.Timestamp(value) for value in activation.loc[activation.ge(start) & activation.lt(end)].unique())
    return sorted(values)


def _load_producer_block(
    spec: dict[str, Any], activation: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_path = ROOT / spec["reference_scores"]
    held_path = ROOT / spec["held_ledger"]
    lineage = [
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256", "ev_score_family_id",
    ]
    reference_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "final_score", "calibration_activation_ts",
        "calibration_reference_oos_to_all_active_fits", *lineage,
    ]
    held_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "final_score", "calibration_activation_ts", "stack_is_prequential",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
        *lineage,
    ]
    filters = [("calibration_activation_ts", "=", activation.to_pydatetime())]
    reference = pd.read_parquet(
        reference_path, columns=reference_columns, filters=filters,
    )
    held = pd.read_parquet(held_path, columns=held_columns, filters=filters)
    if reference.empty or held.empty:
        raise ValueError(f"producer {activation} has an empty held/reference block")
    outcome_columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps",
        "policy_gross_bps", "policy_label_available_ts",
    ]
    # The reserve identities precede activation and are not present in the
    # active held block.  Recover their already-materialised exact policy
    # labels from the compact historical ledger by decision-time predicate.
    # This is a post-score evaluation join; the mapping still checks each
    # label's availability timestamp at every held decision.
    outcome = pd.read_parquet(
        held_path,
        columns=["__decision_ts__", *outcome_columns],
        filters=[
            ("__decision_ts__", ">=", (activation - pd.Timedelta(days=42)).to_pydatetime()),
            ("__decision_ts__", "<", activation.to_pydatetime()),
        ],
    ).drop(columns="__decision_ts__")
    reference = reference.merge(outcome, on="candidate_id", how="left", validate="one_to_one")
    # Reference rows precede activation, so their source ledger can legitimately
    # carry the producer that originally generated each OOF score.  The active
    # bundle re-scores this reserve and owns its coordinate system; bind the
    # reference rows to that active held lineage before applying its strict
    # label-availability checks.
    for column in lineage:
        values = held[column].astype(str).unique()
        if len(values) != 1:
            raise ValueError(f"producer {activation} has ambiguous held {column}")
        reference[column] = values[0]
    reference["calibration_activation_ts"] = activation
    reference["calibration_reference_oos_to_all_active_fits"] = True
    for frame in (reference, held):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["calibration_activation_ts"] = pd.to_datetime(
            frame["calibration_activation_ts"], utc=True, errors="raise",
        )
        frame["policy_label_available_ts"] = pd.to_datetime(
            frame["policy_label_available_ts"], utc=True, errors="coerce",
        )
    start, end = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end_exclusive"])
    held = held.loc[held["__decision_ts__"].ge(start) & held["__decision_ts__"].lt(end)].copy()
    return reference, held


def _map_partition(
    reference: pd.DataFrame,
    held: pd.DataFrame,
    *,
    windows: list[int],
    trim_fraction: float,
    floor_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for values, held_group in held.groupby(list(GROUP_COLUMNS), observed=True, sort=True):
        key = dict(zip(GROUP_COLUMNS, values, strict=True))
        activation = pd.Timestamp(key["calibration_activation_ts"])
        selector = pd.Series(True, index=reference.index)
        for column, value in key.items():
            selector &= reference[column].eq(value)
        reserve = _valid_reference(reference.loc[selector].copy(), activation)
        if reserve.empty:
            raise ValueError(f"producer {activation} has no strict OOS reserve")
        control = _fit_control(reserve, activation)
        reserve_score = pd.to_numeric(reserve["final_score"], errors="coerce").to_numpy(float)
        reserve["__cell__"] = _reference_bins(reserve_score, reserve_score)
        work = held_group.copy().reset_index(drop=True)
        work["__cell__"] = _reference_bins(
            reserve_score,
            pd.to_numeric(work["final_score"], errors="coerce").to_numpy(float),
        )
        result = work.copy()
        result["cell_day_fixed_score_cell"] = result["__cell__"].astype(np.int16)
        history = pd.concat([reserve, work], ignore_index=True, sort=False)
        history["__day__"] = history["__decision_ts__"].dt.normalize()
        history_valid = (
            history["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(history["policy_net_bps"], errors="coerce"))
            & history["__cell__"].ge(0)
        )
        for day, positions in work.groupby(
            work["__decision_ts__"].dt.normalize(), observed=True, sort=True,
        ).groups.items():
            day = pd.Timestamp(day)
            row_positions = np.asarray(list(positions), dtype=np.int64)
            cells = work.loc[row_positions, "__cell__"].to_numpy(np.int16)
            prior_resolved = history["policy_label_available_ts"].le(day)
            max_available = history.loc[
                history_valid & prior_resolved, "policy_label_available_ts"
            ].max()
            for window in windows:
                eligible = (
                    history_valid & prior_resolved
                    & history["__decision_ts__"].ge(day - pd.Timedelta(days=window))
                )
                table = _cell_day_table(history.loc[eligible])
                curve, support = _equal_day_curve(table, trim=trim_fraction)
                arm = _window_arm(window)
                result.loc[row_positions, f"{arm}__expected_net_bps"] = curve[cells]
                result.loc[row_positions, f"{arm}__support_days"] = support[cells]
                audits.append({
                    "snapshot_utc": day,
                    "activation_ts": activation,
                    "window_days": int(window),
                    "eligible_rows": int(eligible.sum()),
                    "eligible_cell_days": int(len(table)),
                    "maximum_label_available_ts": max_available,
                    "resolved_by_snapshot": bool(pd.isna(max_available) or max_available <= day),
                    "mapped_curve_min_bps": float(np.nanmin(curve)),
                    "mapped_curve_max_bps": float(np.nanmax(curve)),
                    "conversion_bundle_sha256": key["conversion_bundle_sha256"],
                    "upstream_bundle_sha256": key["upstream_bundle_sha256"],
                    "geometry_bundle_sha256": key["geometry_bundle_sha256"],
                    "ev_score_family_id": key["ev_score_family_id"],
                    "reserve_rows": int(len(reserve)),
                    "reserve_coordinate_identity": str(control.manifest.get("bundle_identity", "unpersisted")),
                })
        window_columns = [f"{_window_arm(window)}__expected_net_bps" for window in windows]
        estimates = result[window_columns].to_numpy(float)
        ensembles = combine_window_estimates(estimates)
        for arm, estimate in ensembles.items():
            result[f"{arm}__expected_net_bps"] = estimate
        for arm in [*[_window_arm(value) for value in windows], *ensembles]:
            expected = pd.to_numeric(result[f"{arm}__expected_net_bps"], errors="coerce")
            result[f"{arm}__admitted"] = np.isfinite(expected) & expected.ge(floor_bps)
        parts.append(result)
    output = pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any() or len(output) != len(held):
        raise AssertionError("multi-window mapping changed held identities")
    audit = pd.DataFrame(audits)
    if audit.empty or not audit["resolved_by_snapshot"].all():
        raise AssertionError("multi-window mapping consumed unavailable labels")
    return output, audit


def _calendar_metrics(frame: pd.DataFrame, arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    ts = pd.to_datetime(work["__decision_ts__"], utc=True)
    work["day"] = ts.dt.normalize()
    work["week"] = ts.dt.tz_localize(None).dt.to_period("W-MON").astype(str)
    valid = work["policy_path_valid"].fillna(False).astype(bool)
    work["valid_outcome"] = valid
    # Hindsight-only diagnostic for distinguishing admission drought from a
    # genuinely weak opportunity day. Membership is frozen by score first.
    opportunity: dict[pd.Timestamp, float] = {}
    for day, block in work.loc[valid].groupby("day", observed=True, sort=True):
        n = max(1, int(math.ceil(0.01 * len(block))))
        tail = block.nlargest(n, "final_score", keep="all").head(n)
        opportunity[day] = float(pd.to_numeric(tail["policy_net_bps"], errors="coerce").mean())
    work["diagnostic_top1_day_net_bps"] = work["day"].map(opportunity)
    daily_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for arm in arms:
        admitted = work[f"{arm}__admitted"].fillna(False).astype(bool)
        for day, indices in work.groupby("day", observed=True, sort=True).groups.items():
            block = work.loc[indices]
            selected = admitted.loc[indices] & block["valid_outcome"]
            net = pd.to_numeric(block.loc[selected, "policy_net_bps"], errors="coerce")
            diagnostic = float(block["diagnostic_top1_day_net_bps"].iloc[0])
            daily_rows.append({
                "arm": arm, "day": day,
                "admitted_rows": int(admitted.loc[indices].sum()),
                "valid_admitted_rows": int(selected.sum()),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "total_net_bps": float(net.sum()) if len(net) else 0.0,
                "diagnostic_top1_day_net_bps": diagnostic,
                "diagnostic_opportunity_day": bool(np.isfinite(diagnostic) and diagnostic >= 50.0),
                "zero_trade_day": bool(selected.sum() == 0),
                "missed_opportunity_drought_day": bool(selected.sum() == 0 and np.isfinite(diagnostic) and diagnostic >= 50.0),
            })
        for week, indices in work.groupby("week", observed=True, sort=True).groups.items():
            block = work.loc[indices]
            selected = admitted.loc[indices] & block["valid_outcome"]
            net = pd.to_numeric(block.loc[selected, "policy_net_bps"], errors="coerce")
            opportunity_week = bool(block.groupby("day")["diagnostic_top1_day_net_bps"].first().ge(50.0).any())
            weekly_rows.append({
                "arm": arm, "week": week,
                "admitted_rows": int(admitted.loc[indices].sum()),
                "valid_admitted_rows": int(selected.sum()),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "total_net_bps": float(net.sum()) if len(net) else 0.0,
                "positive_week": bool(len(net) and net.sum() > 0.0),
                "zero_trade_week": bool(selected.sum() == 0),
                "diagnostic_opportunity_week": opportunity_week,
                "missed_opportunity_drought_week": bool(selected.sum() == 0 and opportunity_week),
            })
    return pd.DataFrame(daily_rows), pd.DataFrame(weekly_rows)


def _summary(frame: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ts = pd.to_datetime(frame["__decision_ts__"], utc=True)
    for label, start, end in (
        ("development_2025", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        ("confirmation_2026", pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")),
        ("all_2025_2026", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")),
    ):
        period = frame.loc[ts.ge(start) & ts.lt(end)]
        valid = period["policy_path_valid"].fillna(False).astype(bool)
        for arm in arms:
            admitted = period[f"{arm}__admitted"].fillna(False).astype(bool)
            selected = period.loc[valid & admitted]
            net = pd.to_numeric(selected["policy_net_bps"], errors="coerce")
            d = daily.loc[daily["arm"].eq(arm) & daily["day"].ge(start) & daily["day"].lt(end)]
            week_start = str(start.tz_localize(None).to_period("W-MON"))
            week_end = str(
                (end - pd.Timedelta(seconds=1)).tz_localize(None).to_period("W-MON")
            )
            w = weekly.loc[weekly["arm"].eq(arm) & weekly["week"].ge(week_start) & weekly["week"].le(week_end)]
            rows.append({
                "period": label, "arm": arm,
                "valid_admitted_trades": int(len(selected)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "total_net_bps": float(net.sum()) if len(net) else 0.0,
                "positive_trade_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
                "zero_trade_days": int(d["zero_trade_day"].sum()),
                "days": int(len(d)),
                "zero_trade_day_fraction": float(d["zero_trade_day"].mean()) if len(d) else np.nan,
                "missed_opportunity_drought_days": int(d["missed_opportunity_drought_day"].sum()),
                "zero_trade_weeks": int(w["zero_trade_week"].sum()),
                "weeks": int(len(w)),
                "missed_opportunity_drought_weeks": int(w["missed_opportunity_drought_week"].sum()),
                "positive_active_weeks": int(w.loc[~w["zero_trade_week"], "positive_week"].sum()),
                "active_weeks": int((~w["zero_trade_week"]).sum()),
                "worst_active_week_net_bps": float(w.loc[~w["zero_trade_week"], "net_bps_per_trade"].min()) if (~w["zero_trade_week"]).any() else np.nan,
            })
    return pd.DataFrame(rows)


def _portfolio_replay(
    frame: pd.DataFrame,
    arms: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    portfolio: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    block = frame.loc[
        frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)
    ].copy()
    summaries: list[dict[str, Any]] = []
    decisions_out: list[pd.DataFrame] = []
    monthly_out: list[pd.DataFrame] = []
    weekly_out: list[pd.DataFrame] = []
    for arm in arms:
        work = block.copy()
        work["causal_21d_side_expected_net_bps"] = pd.to_numeric(
            work[f"{arm}__expected_net_bps"], errors="coerce",
        )
        work["causal_21d_side_admitted_ge_50bps"] = work[f"{arm}__admitted"].astype(bool)
        work["auction_tie_break_score"] = pd.to_numeric(work["final_score"], errors="coerce")
        lineage = ["conversion_bundle_sha256", "upstream_bundle_sha256", "geometry_bundle_sha256"]
        work["producer_bundle_id"] = work[lineage].astype(str).agg("|".join, axis=1)
        candidates = _auction_candidates(work, strategy_prefix=f"strict_r3_multiwindow_{arm}")
        decisions, equity, monthly, summary = _run(
            candidates, 0.0, f"{start.isoformat()}_{end.isoformat()}_{arm}",
            initial_wallet=float(portfolio["initial_wallet"]),
            perp_leverage=float(portfolio["perp_leverage"]),
            margin_slot_wallet_fraction=float(portfolio["margin_slot_wallet_fraction"]),
            ev_curve=CAUSAL_AUCTION_CURVE,
        )
        evaluation_days = max((end - start).total_seconds() / 86400.0, 1.0)
        summary.update({
            "arm": arm, "period_start": start, "period_end_exclusive": end,
            "trades_per_day": float(summary["accepted_trades"] / evaluation_days),
        })
        summaries.append(summary)
        if len(decisions):
            decisions = decisions.assign(arm=arm)
            decisions_out.append(decisions)
        monthly_wallet = _wallet_periods(
            equity, frequency="month", initial_wallet=float(portfolio["initial_wallet"]),
            evaluation_start=start, evaluation_end=end,
        ).rename(columns={"period": "month"})
        monthly = monthly.merge(monthly_wallet, on="month", how="outer", validate="one_to_one").assign(arm=arm)
        monthly_out.append(monthly)
        weekly = _weekly(decisions)
        weekly_wallet = _wallet_periods(
            equity, frequency="week", initial_wallet=float(portfolio["initial_wallet"]),
            evaluation_start=start, evaluation_end=end,
        ).rename(columns={"period": "week"})
        weekly = weekly.merge(weekly_wallet, on="week", how="outer", validate="one_to_one").assign(arm=arm)
        weekly_out.append(weekly)
    return (
        pd.DataFrame(summaries),
        pd.concat(decisions_out, ignore_index=True, sort=False) if decisions_out else pd.DataFrame(),
        pd.concat(monthly_out, ignore_index=True, sort=False),
        pd.concat(weekly_out, ignore_index=True, sort=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--materialize-only", action="store_true",
        help="Persist compact producer checkpoints and stop before aggregate replay.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable multi-window output exists: {args.out_dir}")
    config = json.loads(args.config.read_text())
    windows = [int(value) for value in config["windows_days"]]
    if windows != [14, 21, 28, 35, 42]:
        raise ValueError("predeclared multi-window grid changed")
    args.out_dir.mkdir(parents=True)
    block_dir = args.out_dir / "mapped_blocks"
    block_dir.mkdir()
    mapped_paths: list[Path] = []
    mapped_rows = 0
    audit_parts: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    for partition in config["partitions"]:
        for role in ("reference_scores", "held_ledger"):
            path = ROOT / partition[role]
            input_hashes[f"{partition['name']}::{role}"] = _sha(path)
        if "policy_outcomes" in partition:
            path = ROOT / partition["policy_outcomes"]
            input_hashes[f"{partition['name']}::policy_outcomes"] = _sha(path)
        if "policy_outcomes" in partition:
            producer_blocks = [None]
        else:
            producer_blocks = _read_activations(
                ROOT / partition["held_ledger"],
                start=pd.Timestamp(partition["start"]),
                end=pd.Timestamp(partition["end_exclusive"]),
            )
        for activation in producer_blocks:
            if activation is None:
                reference, held = _load_partition(partition)
            else:
                reference, held = _load_producer_block(partition, activation)
            if held.empty:
                continue
            mapped, audit = _map_partition(
                reference, held, windows=windows,
                trim_fraction=float(config["trim_fraction"]),
                floor_bps=float(config["admission_floor_net_bps"]),
            )
            mapped["partition"] = partition["name"]
            audit["partition"] = partition["name"]
            compact = [
                "candidate_id", "__decision_ts__", "__symbol__", "side_name",
                "final_score", "policy_path_valid", "policy_gross_bps",
                "policy_net_bps", "policy_exit_bar_15m", "policy_exit_reason",
                "policy_entry_price", "policy_exit_price", "policy_label_available_ts",
                "policy_outcome_source", "conversion_bundle_sha256",
                "upstream_bundle_sha256", "geometry_bundle_sha256",
                "ev_score_family_id", "stack_is_prequential", "partition",
                "cell_day_fixed_score_cell",
            ] + [
                column for column in mapped
                if "__expected_net_bps" in column or "__admitted" in column
                or "__support_days" in column
            ]
            name = pd.Timestamp(mapped["calibration_activation_ts"].iloc[0]).strftime(
                "producer_%Y%m%dT%H%M%SZ.parquet"
            )
            path = block_dir / name
            mapped.loc[:, compact].to_parquet(path, index=False, compression="zstd")
            mapped_paths.append(path)
            mapped_rows += len(mapped)
            audit_parts.append(audit)
            print(json.dumps({
                "event": "producer_complete", "partition": partition["name"],
                "activation": str(activation), "rows": len(mapped),
            }), flush=True)
        print(json.dumps({"event": "partition_complete", "partition": partition["name"]}), flush=True)
    pd.concat(audit_parts, ignore_index=True).to_parquet(
        args.out_dir / "causality_audit.parquet", index=False,
    )
    materialization = {
        "schema": "strict_r3_multiwindow_ev_map_materialization_v1",
        "status": "complete",
        "rows": int(mapped_rows),
        "producer_blocks": len(mapped_paths),
        "windows_days": windows,
        "trim_fraction": float(config["trim_fraction"]),
        "admission_floor_net_bps": float(config["admission_floor_net_bps"]),
        "resolved_set_contract": "every policy label with policy_label_available_ts <= held UTC-day snapshot; unresolved rows ignored; stored availability is decision+12h for every outcome",
        "input_sha256": input_hashes,
        "mapped_blocks": {path.name: _sha(path) for path in mapped_paths},
        "causality_audit_sha256": _sha(args.out_dir / "causality_audit.parquet"),
    }
    (args.out_dir / "materialization_manifest.json").write_text(
        json.dumps(materialization, indent=2, default=str) + "\n"
    )
    if args.materialize_only:
        print(json.dumps({
            "event": "materialization_complete", "out_dir": str(args.out_dir),
            "rows": mapped_rows, "producer_blocks": len(mapped_paths),
        }), flush=True)
        return
    output = pd.concat(
        [pd.read_parquet(path) for path in mapped_paths],
        ignore_index=True, sort=False,
    ).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any():
        raise ValueError("configured partitions overlap by candidate identity")
    arms = [*[_window_arm(value) for value in windows], *[
        f"ensemble_{name}" for name in config["ensemble_arms"]
    ]]
    daily, weekly = _calendar_metrics(output, arms)
    summary = _summary(output, daily, weekly, arms)
    portfolio_summaries: list[pd.DataFrame] = []
    portfolio_decisions: list[pd.DataFrame] = []
    portfolio_monthly: list[pd.DataFrame] = []
    portfolio_weekly: list[pd.DataFrame] = []
    for label, bounds in (
        ("development_2025", config["development_period"]),
        ("confirmation_2026", config["confirmation_period"]),
    ):
        p_summary, p_decisions, p_monthly, p_weekly = _portfolio_replay(
            output, arms, start=pd.Timestamp(bounds[0]), end=pd.Timestamp(bounds[1]),
            portfolio=config["portfolio"],
        )
        p_summary["period"] = label
        p_monthly["period"] = label
        p_weekly["period"] = label
        if len(p_decisions):
            p_decisions["period"] = label
        portfolio_summaries.append(p_summary)
        portfolio_decisions.append(p_decisions)
        portfolio_monthly.append(p_monthly)
        portfolio_weekly.append(p_weekly)
    keep = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score",
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_label_available_ts", "conversion_bundle_sha256",
        "upstream_bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id",
        "stack_is_prequential", "partition",
    ] + [column for column in output if "__expected_net_bps" in column or "__admitted" in column or "__support_days" in column]
    output[keep].to_parquet(args.out_dir / "multiwindow_selection.parquet", index=False, compression="zstd")
    pd.concat(audit_parts, ignore_index=True).to_parquet(args.out_dir / "causality_audit.parquet", index=False)
    daily.to_parquet(args.out_dir / "daily_metrics.parquet", index=False)
    weekly.to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    summary.to_parquet(args.out_dir / "economic_summary.parquet", index=False)
    pd.concat(portfolio_summaries, ignore_index=True, sort=False).to_parquet(args.out_dir / "portfolio_summary.parquet", index=False)
    pd.concat(portfolio_decisions, ignore_index=True, sort=False).to_parquet(args.out_dir / "portfolio_decisions.parquet", index=False, compression="zstd")
    pd.concat(portfolio_monthly, ignore_index=True, sort=False).to_parquet(args.out_dir / "portfolio_monthly.parquet", index=False)
    pd.concat(portfolio_weekly, ignore_index=True, sort=False).to_parquet(args.out_dir / "portfolio_weekly.parquet", index=False)
    manifest = {
        "schema": "strict_r3_multiwindow_ev_map_ablation_v1",
        "status": "completed_mapper_only_research_ablation",
        "side": "long",
        "rows": int(len(output)),
        "windows_days": windows,
        "trim_fraction": float(config["trim_fraction"]),
        "arms": arms,
        "admission_floor_net_bps": float(config["admission_floor_net_bps"]),
        "causality": "each held UTC day consumes only label_available_ts strictly before day 00:00; score cells remain producer-reserve local; ensembles combine only common net-bps estimates",
        "trust_overlay_scope": "R5/A5 excluded from stage 1 because their targets are defined against the canonical 28-day map; refit only Pareto-leading maps in stage 2",
        "selection_protocol": "2025 development; 2026 confirmation; no all-period winner selection",
        "diagnostic_opportunity_day": "hindsight-only frozen-score top-1% policy net >= +50 bps; used only to classify drought, never admission",
        "config": str(args.config),
        "config_sha256": _sha(args.config),
        "input_sha256": input_hashes,
        "outputs": {},
    }
    for path in args.out_dir.iterdir():
        if path.is_file() and path.name != "run_manifest.json":
            manifest["outputs"][path.name] = _sha(path)
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(output), "arms": arms}), flush=True)


if __name__ == "__main__":
    main()
