#!/usr/bin/env python3
"""Matched cell-day and empirical-Bayes EV-map ablations for strict-R3.

Each producer's excluded reserve fixes twenty score cells.  Calibration
evidence is then reduced to one mean policy-net observation per UTC day and
cell.  Held-day maps use only observations whose labels resolved before that
day, so high-volume shock days cannot dominate merely because they emitted
more candidates.

The canonical rolling evidence window is 28 calendar days.  ``--window-days``
exists only to reproduce explicitly labelled legacy research artifacts; the
chosen value is persisted in every output and causality receipt.

The Bayesian arms use a normal/normal empirical-Bayes approximation.  Cell EV
shrinks toward an equal blend of the producer-wide equal-day EV and the frozen
same-producer model-family prior.  Admission is based on ``P(mu_cell > 0)``;
the posterior mean in bps remains the auction ordering value.  This is a
research ablation, not a promoted live contract.
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
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    EVBridgeSpec,
    apply_strict_r3_ev_bridge,
    fit_strict_r3_ev_bridge,
)
from extreme_price_movements.strict_r3_cell_day_admission import (  # noqa: E402
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
)


GROUP_COLUMNS = (
    "ev_score_family_id", "geometry_bundle_sha256",
    "conversion_bundle_sha256", "upstream_bundle_sha256",
    "calibration_activation_ts",
)
BINS = 20
DEFAULT_WINDOW_DAYS = 28
TRIM_FRACTIONS = (0.10, 0.15, 0.20, 0.25)
RECENT_BOTTOM_PROTECTION_DAYS = 5
BAYES_GRID = (
    (3.0, 0.70), (7.0, 0.70), (14.0, 0.70),
    (7.0, 0.80), (7.0, 0.90),
)
COMBINED_GRID = ((0.10, 7.0, 0.70), (0.15, 7.0, 0.70), (0.15, 7.0, 0.80))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _trim_arm(value: float) -> str:
    return f"cell_day_trim_{int(round(100 * value)):02d}pct"


def _reactive_trim_arm(value: float) -> str:
    return f"cell_day_trim_{int(round(100 * value)):02d}pct_protect_recent5d_bottom"


def _bayes_arm(k0: float, probability: float) -> str:
    return f"bayes_k{int(k0):02d}_p{int(round(100 * probability)):02d}"


def _combined_arm(trim: float, k0: float, probability: float) -> str:
    return f"cell_day_trim_{int(round(100 * trim)):02d}_bayes_k{int(k0):02d}_p{int(round(100 * probability)):02d}"


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(value, dtype=float) / math.sqrt(2.0)))


def _reference_bins(reference_score: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_score, dtype=float), kind="stable")
    if len(reference) < BINS * 4 or np.unique(reference).size < 4:
        raise ValueError("producer reserve has insufficient score support for fixed cells")
    current = np.asarray(values, dtype=float)
    output = np.full(len(current), -1, dtype=np.int16)
    finite = np.isfinite(current)
    rank = np.searchsorted(reference, current[finite], side="right")
    output[finite] = np.minimum(rank * BINS // len(reference), BINS - 1).astype(np.int16)
    return output


def _trim_values(values: np.ndarray, fraction: float) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=float))
    trim = int(math.floor(len(ordered) * float(fraction)))
    return ordered[trim:len(ordered) - trim] if len(ordered) - 2 * trim else ordered


def _monotone(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    support = np.asarray(support, dtype=float)
    usable = np.isfinite(values) & (support > 0.0)
    if usable.sum() < 2:
        return values
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    x = (np.arange(BINS, dtype=float) + 0.5) / BINS
    fitted = values.copy()
    fitted[usable] = model.fit(x[usable], values[usable], sample_weight=support[usable]).predict(x[usable])
    if not usable.all():
        fitted[~usable] = np.interp(x[~usable], x[usable], fitted[usable])
    return fitted


def _cell_day_table(history: pd.DataFrame) -> pd.DataFrame:
    return history.groupby(["__day__", "__cell__"], observed=True, sort=True).agg(
        cell_day_ev_bps=("policy_net_bps", "mean"),
        cell_day_trades=("candidate_id", "size"),
    ).reset_index()


def _equal_day_curve(table: pd.DataFrame, *, trim: float) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(BINS, np.nan, dtype=float)
    days = np.zeros(BINS, dtype=np.int64)
    for cell in range(BINS):
        values = table.loc[table["__cell__"].eq(cell), "cell_day_ev_bps"].to_numpy(float)
        retained = _trim_values(values, trim)
        days[cell] = len(retained)
        if len(retained):
            means[cell] = float(np.mean(retained))
    return _monotone(means, days), days


def _reactive_equal_day_curve(
    table: pd.DataFrame, *, trim: float, snapshot: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim old bottom days and all top days while retaining recent bad news.

    A bottom-tail cell-day from ``[snapshot-5d, snapshot)`` can never be
    removed.  Therefore deterioration reaches the map immediately.  The
    symmetric top-tail removal remains defined over the complete causal
    window, preventing an unusually favorable recent day from creating the
    opposite optimistic distortion.
    """
    means = np.full(BINS, np.nan, dtype=float)
    days = np.zeros(BINS, dtype=np.int64)
    recent_start = pd.Timestamp(snapshot) - pd.Timedelta(days=RECENT_BOTTOM_PROTECTION_DAYS)
    for cell in range(BINS):
        block = table.loc[table["__cell__"].eq(cell), ["__day__", "cell_day_ev_bps"]].copy()
        if block.empty:
            continue
        trim_count = int(math.floor(len(block) * float(trim)))
        keep = pd.Series(True, index=block.index)
        if trim_count:
            old = block["__day__"].lt(recent_start)
            bottom = block.loc[old].sort_values(
                ["cell_day_ev_bps", "__day__"], kind="stable",
            ).index[:trim_count]
            keep.loc[bottom] = False
            # Top-tail trimming is unrestricted and is computed after the
            # protected-bottom decision so the two tails cannot remove the
            # same observation.
            top = block.loc[keep].sort_values(
                ["cell_day_ev_bps", "__day__"], ascending=[False, True], kind="stable",
            ).index[:trim_count]
            keep.loc[top] = False
        retained = block.loc[keep, "cell_day_ev_bps"].to_numpy(float)
        days[cell] = len(retained)
        if len(retained):
            means[cell] = float(np.mean(retained))
    return _monotone(means, days), days


def _bayesian_curve(
    table: pd.DataFrame,
    *,
    model_prior: np.ndarray,
    prior_days: float,
    trim: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_values = table["cell_day_ev_bps"].to_numpy(float)
    global_ev = float(np.mean(_trim_values(all_values, trim))) if len(all_values) else 0.0
    global_std = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else 200.0
    global_std = max(global_std, 25.0)
    posterior = np.full(BINS, np.nan, dtype=float)
    probability = np.full(BINS, np.nan, dtype=float)
    support = np.zeros(BINS, dtype=np.int64)
    for cell in range(BINS):
        values = table.loc[table["__cell__"].eq(cell), "cell_day_ev_bps"].to_numpy(float)
        values = _trim_values(values, trim)
        support[cell] = len(values)
        prior_mean = 0.5 * global_ev + 0.5 * float(model_prior[cell])
        if len(values):
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else global_std**2
            variance = max(variance, 25.0**2)
            posterior[cell] = (float(np.sum(values)) + prior_days * prior_mean) / (len(values) + prior_days)
            posterior_se = math.sqrt(variance / (len(values) + prior_days))
        else:
            posterior[cell] = prior_mean
            posterior_se = global_std / math.sqrt(prior_days)
        probability[cell] = float(_normal_cdf(np.array([posterior[cell] / max(posterior_se, 1e-9)]))[0])
    return _monotone(posterior, np.maximum(support, 1)), probability, support


def _valid_reference(frame: pd.DataFrame, activation: pd.Timestamp) -> pd.DataFrame:
    valid = (
        frame["calibration_reference_oos_to_all_active_fits"].fillna(False).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce").lt(activation)
        & np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    )
    output = frame.loc[valid].copy()
    output["stack_is_prequential"] = True
    return output


def _fit_control(reference: pd.DataFrame, activation: pd.Timestamp) -> Any:
    return fit_strict_r3_ev_bridge(
        reference,
        fit_cutoff=activation,
        spec=EVBridgeSpec(prior_bins=BINS, prior_trim_fraction=0.05),
        producer_lineage={
            "conversion_bundle_sha256": str(reference["conversion_bundle_sha256"].iloc[0]),
            "upstream_bundle_sha256": str(reference["upstream_bundle_sha256"].iloc[0]),
        },
    )


def _period_metrics(frame: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    periods: list[tuple[str, str, np.ndarray]] = [("all", "all", np.arange(len(frame)))]
    for frequency in ("M", "W-MON"):
        labels = timestamp.dt.tz_localize(None).dt.to_period(frequency).astype(str)
        for period, indices in labels.groupby(labels, observed=True, sort=True).groups.items():
            periods.append((frequency, str(period), np.asarray(list(indices), dtype=np.int64)))
    rows: list[dict[str, object]] = []
    for arm in arms:
        for frequency, period, indices in periods:
            admitted = frame[f"{arm}__admitted"].iloc[indices].fillna(False).astype(bool)
            selected = admitted & valid.iloc[indices]
            block = frame.iloc[indices].loc[selected]
            net = pd.to_numeric(block["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "frequency": frequency, "period": period,
                "scored_rows": int(len(indices)), "admitted_rows": int(admitted.sum()),
                "valid_admitted_rows": int(len(block)), "admission_rate": float(admitted.mean()),
                "net_bps_per_trade": float(net.mean()) if len(block) else np.nan,
                "gross_bps_per_trade": float(pd.to_numeric(block["policy_gross_bps"], errors="coerce").mean()) if len(block) else np.nan,
                "positive_net_rate": float(net.gt(0.0).mean()) if len(block) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--held-ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path)
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=(
            "Strictly prior resolved Cell-day history window. The canonical "
            "contract is 28 days; other values are legacy/research only."
        ),
    )
    args = parser.parse_args()
    if args.window_days <= 0:
        raise ValueError("--window-days must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable cell-day Bayesian EV-map output exists: {args.out_dir}")
    reference = pd.read_parquet(args.reference_scores)
    outcome = pd.read_parquet(args.policy_outcomes)
    held = pd.read_parquet(args.held_ledger)
    if outcome["candidate_id"].duplicated().any() or held["candidate_id"].duplicated().any():
        raise ValueError("policy outcomes and held ledger require unique candidate IDs")
    reference = reference.merge(
        outcome.loc[:, [
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_gross_bps", "policy_label_available_ts",
        ]], on="candidate_id", how="left", validate="many_to_one",
    )
    for frame in (reference, held):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["calibration_activation_ts"] = pd.to_datetime(
            frame["calibration_activation_ts"], utc=True, errors="raise",
        )
        frame["policy_label_available_ts"] = pd.to_datetime(
            frame["policy_label_available_ts"], utc=True, errors="coerce",
        )

    arms = ["exact_reserve_control", "cell_day_equal_weight"]
    arms.extend(_trim_arm(value) for value in TRIM_FRACTIONS)
    arms.extend(_reactive_trim_arm(value) for value in TRIM_FRACTIONS)
    arms.extend(_bayes_arm(k0, probability) for k0, probability in BAYES_GRID)
    arms.extend(_combined_arm(trim, k0, probability) for trim, k0, probability in COMBINED_GRID)
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    canonical_daily_audits: list[dict[str, object]] = []
    control_lookup = None
    if args.control_predictions is not None:
        control = pd.read_parquet(args.control_predictions)
        if set(control["candidate_id"]) != set(held["candidate_id"]):
            raise ValueError("control predictions do not cover held candidates exactly")
        control_lookup = control.set_index("candidate_id")["causal_21d_side_expected_net_bps"]

    for values, held_group in held.groupby(list(GROUP_COLUMNS), observed=True, sort=True):
        key = dict(zip(GROUP_COLUMNS, values, strict=True))
        activation = pd.Timestamp(key["calibration_activation_ts"])
        selector = pd.Series(True, index=reference.index)
        for column, value in key.items():
            selector &= reference[column].eq(value)
        reserve = _valid_reference(reference.loc[selector].copy(), activation)
        if reserve.empty:
            raise ValueError(f"producer {activation} has no exact OOS reserve")
        control_bundle = _fit_control(reserve, activation)
        mapped_control, _ = apply_strict_r3_ev_bridge(held_group.copy(), bundle=control_bundle)
        result = mapped_control.loc[:, [
            "candidate_id", "__decision_ts__", "policy_path_valid", "policy_gross_bps",
            "policy_net_bps", "policy_label_available_ts", "final_score",
            "conversion_bundle_sha256", "upstream_bundle_sha256", "geometry_bundle_sha256",
            "ev_score_family_id", "stack_is_prequential",
        ]].copy()
        result["calibration_activation_ts"] = activation
        result["exact_reserve_control__expected_net_bps"] = mapped_control[
            "causal_21d_side_expected_net_bps"
        ].to_numpy(float)
        result["exact_reserve_control__admitted"] = result[
            "exact_reserve_control__expected_net_bps"
        ].ge(50.0).fillna(False)
        parity = np.nan
        if control_lookup is not None:
            expected = result["candidate_id"].map(control_lookup).to_numpy(float)
            parity = float(np.nanmax(np.abs(
                expected - result["exact_reserve_control__expected_net_bps"].to_numpy(float)
            )))
            if parity > 1e-9:
                raise AssertionError(f"exact-reserve control parity changed by {parity} bps")

        reserve_score = pd.to_numeric(reserve["final_score"], errors="coerce").to_numpy(float)
        reserve["__cell__"] = _reference_bins(reserve_score, reserve_score)
        held_work = held_group.copy()
        held_work["__cell__"] = _reference_bins(
            reserve_score, pd.to_numeric(held_work["final_score"], errors="coerce").to_numpy(float),
        )
        model_prior = control_bundle.predict_prior(pd.DataFrame({
            **{column: [held_group[column].iloc[0]] * BINS for column in (
                "ev_score_family_id", "geometry_bundle_sha256",
                "conversion_bundle_sha256", "upstream_bundle_sha256",
            )},
            "candidate_id": [f"prior-{cell}" for cell in range(BINS)],
            "__decision_ts__": [activation] * BINS,
            "side_name": ["long"] * BINS,
            "final_score": [float(np.quantile(reserve_score, (cell + 0.5) / BINS)) for cell in range(BINS)],
            "stack_is_prequential": [True] * BINS,
        }))
        history = pd.concat([reserve, held_work], ignore_index=True, sort=False)
        history["__day__"] = history["__decision_ts__"].dt.normalize()
        history_valid = (
            history["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(history["policy_net_bps"], errors="coerce"))
            & history["__cell__"].ge(0)
        )
        for day, current_positions in held_work.groupby(
            held_work["__decision_ts__"].dt.normalize(), observed=True, sort=True,
        ).groups.items():
            day = pd.Timestamp(day)
            eligible = (
                history_valid
                & history["policy_label_available_ts"].lt(day)
                & history["__decision_ts__"].ge(
                    day - pd.Timedelta(days=args.window_days)
                )
            )
            eligible_history = history.loc[eligible]
            table = _cell_day_table(eligible_history)
            maximum_label_available = pd.to_datetime(
                eligible_history["policy_label_available_ts"], utc=True, errors="coerce",
            ).max()
            canonical_daily_audits.append({
                "snapshot_utc": day,
                "activation_ts": activation,
                "ev_score_family_id": key["ev_score_family_id"],
                "geometry_bundle_sha256": key["geometry_bundle_sha256"],
                "conversion_bundle_sha256": key["conversion_bundle_sha256"],
                "upstream_bundle_sha256": key["upstream_bundle_sha256"],
                "eligible_rows": int(len(eligible_history)),
                "eligible_cell_days": int(len(table)),
                "minimum_decision_ts": (
                    eligible_history["__decision_ts__"].min()
                    if len(eligible_history) else pd.NaT
                ),
                "maximum_label_available_ts": maximum_label_available,
                "strictly_prior_resolved": bool(
                    pd.isna(maximum_label_available) or maximum_label_available < day
                ),
                "window_days": int(args.window_days),
            })
            current_index = np.asarray(list(current_positions), dtype=np.int64)
            cells = held_work.loc[current_index, "__cell__"].to_numpy(np.int16)
            equal_curve, equal_support = _equal_day_curve(table, trim=0.0)
            result.loc[current_index, "cell_day_equal_weight__expected_net_bps"] = equal_curve[cells]
            result.loc[current_index, "cell_day_equal_weight__admitted"] = equal_curve[cells] >= 50.0
            for trim in TRIM_FRACTIONS:
                arm = _trim_arm(trim)
                curve, trim_support = _equal_day_curve(table, trim=trim)
                result.loc[current_index, f"{arm}__expected_net_bps"] = curve[cells]
                result.loc[current_index, f"{arm}__admitted"] = curve[cells] >= 50.0
                if np.isclose(trim, 0.15, atol=0.0, rtol=0.0):
                    result.loc[current_index, "cell_day_trim_15pct__fixed_score_cell"] = cells
                    result.loc[
                        current_index, "cell_day_trim_15pct__retained_day_support"
                    ] = trim_support[cells]
                reactive_arm = _reactive_trim_arm(trim)
                reactive_curve, _ = _reactive_equal_day_curve(
                    table, trim=trim, snapshot=day,
                )
                result.loc[current_index, f"{reactive_arm}__expected_net_bps"] = reactive_curve[cells]
                result.loc[current_index, f"{reactive_arm}__admitted"] = reactive_curve[cells] >= 50.0
            for k0, probability_floor in BAYES_GRID:
                arm = _bayes_arm(k0, probability_floor)
                curve, probability, _ = _bayesian_curve(
                    table, model_prior=model_prior, prior_days=k0, trim=0.0,
                )
                result.loc[current_index, f"{arm}__expected_net_bps"] = curve[cells]
                result.loc[current_index, f"{arm}__posterior_positive_probability"] = probability[cells]
                result.loc[current_index, f"{arm}__admitted"] = probability[cells] >= probability_floor
            for trim, k0, probability_floor in COMBINED_GRID:
                arm = _combined_arm(trim, k0, probability_floor)
                curve, probability, _ = _bayesian_curve(
                    table, model_prior=model_prior, prior_days=k0, trim=trim,
                )
                result.loc[current_index, f"{arm}__expected_net_bps"] = curve[cells]
                result.loc[current_index, f"{arm}__posterior_positive_probability"] = probability[cells]
                result.loc[current_index, f"{arm}__admitted"] = probability[cells] >= probability_floor
        for arm in arms:
            if result[f"{arm}__admitted"].isna().any():
                raise AssertionError(f"arm {arm} has indeterminate held admissions")
            result[f"{arm}__admitted"] = result[f"{arm}__admitted"].astype(bool)
        audits.append({
            "activation_ts": activation, "reserve_rows": int(len(reserve)),
            "reserve_days": int(reserve["__decision_ts__"].dt.normalize().nunique()),
            "held_rows": int(len(held_group)), "control_parity_max_abs_bps": parity,
            "cell_count": BINS, "window_days": int(args.window_days),
        })
        parts.append(result)

    output = pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any() or len(output) != len(held):
        raise AssertionError("cell-day mapping changed held identities")
    metrics = _period_metrics(output, arms)
    daily = output.assign(
        day=pd.to_datetime(output["__decision_ts__"], utc=True).dt.normalize(),
    ).groupby("day", observed=True, sort=True).agg(**{
        f"{arm}__admitted_rows": (f"{arm}__admitted", "sum") for arm in arms
    }).reset_index()
    selection = ["candidate_id", "__decision_ts__", "final_score"]
    selection.extend(column for column in output if "__expected_net_bps" in column or "__admitted" in column or "__posterior_positive_probability" in column)
    args.out_dir.mkdir(parents=True)
    output.loc[:, selection].to_parquet(args.out_dir / "cell_day_bayesian_selection.parquet", index=False, compression="zstd")
    canonical_columns = [
        "candidate_id", "__decision_ts__", "final_score",
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256", "ev_score_family_id",
    ]
    if "stack_is_prequential" in held:
        canonical_columns.append("stack_is_prequential")
    else:
        raise ValueError("canonical Cell-day provenance requires stack_is_prequential")
    canonical = output.loc[:, canonical_columns].copy()
    canonical["causal_21d_side_expected_net_bps"] = output[
        "cell_day_trim_15pct__expected_net_bps"
    ].to_numpy(float)
    canonical["causal_21d_side_admitted_ge_50bps"] = output[
        "cell_day_trim_15pct__admitted"
    ].to_numpy(bool)
    canonical["causal_21d_side_mapping_status"] = CELL_DAY_TRIM_15_CALIBRATION_MODE
    canonical["cell_day_fixed_score_cell"] = pd.to_numeric(
        output["cell_day_trim_15pct__fixed_score_cell"], errors="raise",
    ).astype(np.int16)
    canonical["cell_day_retained_day_support"] = pd.to_numeric(
        output["cell_day_trim_15pct__retained_day_support"], errors="raise",
    ).astype(np.int16)
    canonical.to_parquet(
        args.out_dir / "score_and_cell_day_admission_provenance.parquet",
        index=False, compression="zstd",
    )
    canonical_audit = pd.DataFrame(canonical_daily_audits)
    if canonical_audit.empty or not canonical_audit[
        "strictly_prior_resolved"
    ].fillna(False).astype(bool).all():
        raise AssertionError("canonical Cell-day daily audit is not strictly prior-resolved")
    canonical_audit.to_parquet(
        args.out_dir / "cell_day_admission_audit.parquet", index=False,
    )
    metrics.to_parquet(args.out_dir / "cell_day_bayesian_metrics.parquet", index=False)
    daily.to_parquet(args.out_dir / "daily_admission_continuity.parquet", index=False)
    pd.DataFrame(audits).to_parquet(args.out_dir / "producer_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_cell_day_bayesian_ev_map_ablation_v1",
        "promotion_status": "research_ablation_only",
        "rows": int(len(output)), "producer_groups": int(len(audits)),
        "score_cells": BINS, "rolling_window_days": int(args.window_days),
        "arms": arms,
        "reference_scores": str(args.reference_scores), "reference_scores_sha256": _sha(args.reference_scores),
        "policy_outcomes": str(args.policy_outcomes), "policy_outcomes_sha256": _sha(args.policy_outcomes),
        "held_ledger": str(args.held_ledger), "held_ledger_sha256": _sha(args.held_ledger),
        "control_predictions": str(args.control_predictions) if args.control_predictions else None,
        "period_weighting": "one observation per UTC day x fixed producer-reserve score cell",
        "reactive_trim_rule": (
            "bottom-tail cell-days inside the preceding five calendar days are never "
            "trimmed; top-tail trimming remains active over the full causal window"
        ),
        "bayesian_prior": "equal blend of producer-wide equal-day EV and frozen same-producer model-family prior",
        "admission": "control/trim arms expected net >= 50 bps; Bayesian arms posterior P(mu_cell > 0) >= declared probability",
        "causality": (
            "for held day t, label_available_ts < t 00:00 UTC and "
            f"decision_ts >= t-{int(args.window_days)}d"
        ),
        "mapping": CELL_DAY_TRIM_15_CALIBRATION_MODE,
        "canonical_provenance": str(
            args.out_dir / "score_and_cell_day_admission_provenance.parquet"
        ),
        "canonical_provenance_identity_rows": int(len(canonical)),
        "canonical_provenance_strictly_prior_resolved": True,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
