#!/usr/bin/env python3
"""Ablate shock-robust, exact-producer strict-R3 EV admission maps.

The upstream score, policy labels, model vintages, candidate population, and
portfolio contract remain frozen.  This experiment changes only the causal
score-to-policy-net conversion:

* symmetric robust-z day trimming of the 42-day same-producer reserve;
* ordinary standard-deviation day rejection on that reserve; and
* causal daily residual-state corrections using EV trend, slope, dispersion,
  and sign entropy.

Daily outliers are defined from calibration residuals (realised policy net
minus the provisional same-producer EV map), not from raw PnL.  Therefore a
large market day is retained when the score-to-EV map anticipated it.  Every
state feature for held day ``t`` uses only labels available before 00:00 UTC
on ``t``.  The output is an ablation ledger, never a promoted live map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_ev_bridge import (  # noqa: E402
    EVBridgeSpec,
    apply_strict_r3_ev_bridge,
    fit_strict_r3_ev_bridge,
)


GROUP_COLUMNS = (
    "ev_score_family_id",
    "geometry_bundle_sha256",
    "conversion_bundle_sha256",
    "upstream_bundle_sha256",
    "calibration_activation_ts",
)
TRIM_FRACTIONS = (0.10, 0.15, 0.20, 0.25)
STD_CUTOFFS = (1.0, 1.5, 2.0)
STATE_ARMS = (
    "state_ev_trend",
    "state_ev_slope",
    "state_ev_entropy",
    "state_ev_std",
    "state_ev_trend_slope_entropy_std",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _arm_trim(value: float) -> str:
    return f"robust_z_trim_days_{int(round(100 * value)):02d}pct"


def _arm_std(value: float) -> str:
    return f"std_day_filter_{str(value).replace('.', 'p')}sigma"


def _valid_policy_rows(frame: pd.DataFrame, *, cutoff: pd.Timestamp) -> pd.Series:
    return (
        frame["calibration_reference_oos_to_all_active_fits"].fillna(False).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce").lt(cutoff)
        & np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    )


def _fit_bundle(reference: pd.DataFrame, *, activation: pd.Timestamp) -> Any:
    lineage = {
        "conversion_bundle_sha256": str(reference["conversion_bundle_sha256"].iloc[0]),
        "upstream_bundle_sha256": str(reference["upstream_bundle_sha256"].iloc[0]),
    }
    return fit_strict_r3_ev_bridge(
        reference,
        fit_cutoff=activation,
        spec=EVBridgeSpec(
            prior_bins=20,
            prior_trim_fraction=0.05,
            required_prior_rows_per_side=100,
            minimum_residual_rows=20,
        ),
        producer_lineage=lineage,
    )


def _daily_calibration_residuals(
    reference: pd.DataFrame, *, provisional_bundle: Any,
) -> pd.DataFrame:
    work = reference.copy()
    prior = provisional_bundle.predict_prior(work)
    work["__calibration_residual_bps__"] = (
        pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float) - prior
    )
    work["__day__"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.normalize()
    daily = work.groupby("__day__", observed=True, sort=True).agg(
        daily_calibration_residual_bps=("__calibration_residual_bps__", "mean"),
        daily_policy_net_bps=("policy_net_bps", "mean"),
        rows=("candidate_id", "size"),
    ).reset_index()
    values = daily["daily_calibration_residual_bps"].to_numpy(float)
    centre = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - centre)))
    robust_scale = max(1.4826 * mad, 1e-9)
    daily["robust_z"] = (values - centre) / robust_scale
    ordinary_centre = float(np.nanmean(values))
    ordinary_scale = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    daily["ordinary_z"] = (
        (values - ordinary_centre) / ordinary_scale
        if ordinary_scale > 1e-9 else np.zeros(len(values), dtype=float)
    )
    return daily


def _keep_trimmed_days(daily: pd.DataFrame, fraction: float) -> set[pd.Timestamp]:
    ordered = daily.sort_values(
        ["robust_z", "__day__"], kind="stable",
    ).reset_index(drop=True)
    trim = int(math.floor(len(ordered) * float(fraction)))
    if len(ordered) - 2 * trim < 4:
        raise ValueError("robust day trim leaves fewer than four calibration days")
    return set(ordered.iloc[trim:len(ordered) - trim]["__day__"])


def _keep_std_days(daily: pd.DataFrame, cutoff: float) -> set[pd.Timestamp]:
    kept = daily.loc[daily["ordinary_z"].abs().le(float(cutoff)), "__day__"]
    if len(kept) < 4:
        raise ValueError("standard-deviation day filter leaves fewer than four calibration days")
    return set(kept)


def _entropy(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    positive = float(np.mean(finite >= 0.0))
    if positive <= 0.0 or positive >= 1.0:
        return 0.0
    return float(-(positive * math.log(positive) + (1.0 - positive) * math.log(1.0 - positive)) / math.log(2.0))


def _ewma(values: np.ndarray, ages_days: np.ndarray, *, half_life_days: float) -> float:
    weights = np.exp(-math.log(2.0) * np.asarray(ages_days, dtype=float) / half_life_days)
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def _daily_residual_state(mapped: pd.DataFrame) -> pd.DataFrame:
    """Materialise causal daily EV state once per exact producer block."""
    work = mapped.copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True, errors="coerce",
    )
    decision_day = work["__decision_ts__"].dt.normalize()
    residual = pd.to_numeric(work["ev_bridge_policy_residual_bps"], errors="coerce")
    valid = work["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(residual)
    output = pd.DataFrame({"__position__": work["__position__"]})
    for column in (
        "ev_state_level21_bps", "ev_state_ewma3_bps", "ev_state_ewma14_bps",
        "ev_state_trend_bps", "ev_state_slope_bps_per_day", "ev_state_std_bps",
        "ev_state_sign_entropy", "ev_state_reference_days",
    ):
        output[column] = np.nan

    for snapshot in sorted(decision_day.unique()):
        snapshot = pd.Timestamp(snapshot)
        current = decision_day.eq(snapshot)
        prior = (
            valid
            & work["policy_label_available_ts"].lt(snapshot)
            & work["__decision_ts__"].ge(snapshot - pd.Timedelta(days=21))
        )
        reference = work.loc[prior, ["__decision_ts__"]].copy()
        reference["residual"] = residual.loc[prior].to_numpy(float)
        reference["day"] = reference["__decision_ts__"].dt.normalize()
        daily = reference.groupby("day", observed=True, sort=True)["residual"].mean()
        if len(daily) < 3:
            continue
        values = daily.to_numpy(float)
        ages = (snapshot - pd.DatetimeIndex(daily.index)).total_seconds().to_numpy(float) / 86_400.0
        level = _ewma(values, ages, half_life_days=7.0)
        ewma3 = _ewma(values, ages, half_life_days=3.0)
        ewma14 = _ewma(values, ages, half_life_days=14.0)
        trend = ewma3 - ewma14
        x = -ages
        slope = float(np.polyfit(x, values, deg=1)[0]) if np.ptp(x) > 0.0 else 0.0
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        entropy = _entropy(values)
        indices = output.index[current]
        output.loc[indices, "ev_state_level21_bps"] = level
        output.loc[indices, "ev_state_ewma3_bps"] = ewma3
        output.loc[indices, "ev_state_ewma14_bps"] = ewma14
        output.loc[indices, "ev_state_trend_bps"] = trend
        output.loc[indices, "ev_state_slope_bps_per_day"] = slope
        output.loc[indices, "ev_state_std_bps"] = std
        output.loc[indices, "ev_state_sign_entropy"] = entropy
        output.loc[indices, "ev_state_reference_days"] = float(len(daily))
    return output.sort_values("__position__", kind="stable").reset_index(drop=True)


def _state_expected(prior: np.ndarray, state: pd.DataFrame, arm: str) -> np.ndarray:
    level = state["ev_state_level21_bps"].fillna(0.0).to_numpy(float)
    trend = state["ev_state_trend_bps"].fillna(0.0).to_numpy(float)
    slope = state["ev_state_slope_bps_per_day"].fillna(0.0).to_numpy(float)
    std = state["ev_state_std_bps"].fillna(0.0).to_numpy(float)
    entropy = state["ev_state_sign_entropy"].fillna(1.0).to_numpy(float)
    support = state["ev_state_reference_days"].fillna(0.0).to_numpy(float)
    supported = support >= 3.0
    clipped_trend = np.clip(trend, -std, std)
    clipped_slope = np.clip(3.0 * slope, -std, std)
    if arm == "state_ev_trend":
        correction = level + clipped_trend
    elif arm == "state_ev_slope":
        correction = level + clipped_slope
    elif arm == "state_ev_entropy":
        correction = level * (1.0 - 0.50 * entropy)
    elif arm == "state_ev_std":
        correction = level / (1.0 + std / 200.0)
    elif arm == "state_ev_trend_slope_entropy_std":
        raw = level + 0.5 * clipped_trend + 0.5 * clipped_slope
        correction = raw * (1.0 - 0.50 * entropy) / (1.0 + std / 200.0)
    else:  # pragma: no cover - protected by STATE_ARMS
        raise ValueError(f"unknown state arm: {arm}")
    correction = np.where(supported, correction, 0.0)
    return np.asarray(prior, dtype=float) + correction


def _metric_rows(frame: pd.DataFrame, *, arms: Iterable[str]) -> pd.DataFrame:
    work = frame.copy()
    timestamp = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    valid = work["policy_path_valid"].fillna(False).astype(bool)
    output: list[dict[str, object]] = []
    periods: list[tuple[str, str, np.ndarray]] = [("all", "all", np.arange(len(work)))]
    for frequency in ("M", "W-MON"):
        label = timestamp.dt.tz_localize(None).dt.to_period(frequency).astype(str)
        for period, indices in label.groupby(label, observed=True, sort=True).groups.items():
            periods.append((frequency, str(period), np.asarray(list(indices), dtype=np.int64)))
    for arm in arms:
        expected = pd.to_numeric(work[f"{arm}__expected_net_bps"], errors="coerce")
        admitted = work[f"{arm}__admitted"].fillna(False).astype(bool)
        for frequency, period, indices in periods:
            selected = admitted.iloc[indices] & valid.iloc[indices]
            block = work.iloc[indices].loc[selected]
            output.append({
                "arm": arm,
                "frequency": frequency,
                "period": period,
                "scored_rows": int(len(indices)),
                "mapped_rows": int(np.isfinite(expected.iloc[indices]).sum()),
                "admitted_rows": int(admitted.iloc[indices].sum()),
                "valid_admitted_rows": int(len(block)),
                "admission_rate": float(admitted.iloc[indices].mean()),
                "expected_net_bps_per_trade": float(expected.iloc[indices].loc[selected].mean()) if len(block) else np.nan,
                "net_bps_per_trade": float(pd.to_numeric(block["policy_net_bps"], errors="coerce").mean()) if len(block) else np.nan,
                "gross_bps_per_trade": float(pd.to_numeric(block["policy_gross_bps"], errors="coerce").mean()) if len(block) else np.nan,
                "positive_net_rate": float(pd.to_numeric(block["policy_net_bps"], errors="coerce").gt(0.0).mean()) if len(block) else np.nan,
            })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scores", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--held-ledger", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--net-floor-bps", type=float, default=50.0)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable robust EV-map ablation exists: {args.out_dir}")

    reference = pd.read_parquet(args.reference_scores)
    outcomes = pd.read_parquet(args.policy_outcomes)
    held = pd.read_parquet(args.held_ledger)
    control = pd.read_parquet(args.control_predictions)
    if any(frame["candidate_id"].duplicated().any() for frame in (outcomes, held, control)):
        raise ValueError("held, outcome, and control ledgers require unique candidate IDs")
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True, errors="raise")
    reference["calibration_activation_ts"] = pd.to_datetime(
        reference["calibration_activation_ts"], utc=True, errors="raise",
    )
    reference = reference.merge(
        outcomes.loc[:, [
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_gross_bps", "policy_label_available_ts",
        ]],
        on="candidate_id", how="left", validate="many_to_one",
    )
    reference["policy_label_available_ts"] = pd.to_datetime(
        reference["policy_label_available_ts"], utc=True, errors="coerce",
    )
    held["calibration_activation_ts"] = pd.to_datetime(
        held["calibration_activation_ts"], utc=True, errors="raise",
    )
    if set(held["candidate_id"]) != set(control["candidate_id"]):
        raise ValueError("control predictions do not cover the held ledger exactly")

    control_expected = control.set_index("candidate_id")["causal_21d_side_expected_net_bps"]
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    all_arms = ["exact_reserve_control"]
    all_arms.extend(_arm_trim(value) for value in TRIM_FRACTIONS)
    all_arms.extend(_arm_std(value) for value in STD_CUTOFFS)
    all_arms.extend(STATE_ARMS)

    for values, held_group in held.groupby(list(GROUP_COLUMNS), observed=True, sort=True):
        key = dict(zip(GROUP_COLUMNS, values, strict=True))
        activation = pd.Timestamp(key["calibration_activation_ts"])
        mask = pd.Series(True, index=reference.index)
        for column, value in key.items():
            mask &= reference[column].eq(value)
        reserve = reference.loc[mask].copy()
        reserve = reserve.loc[_valid_policy_rows(reserve, cutoff=activation)].copy()
        if reserve.empty:
            raise ValueError(f"producer {activation} has no valid exact OOS reserve")
        # This derived flag is added by the canonical immediate-calibration
        # builder only after the reserve-exclusion and label-resolution gates.
        # Mirror that contract here; never trust an input-side assertion in
        # place of these explicit checks.
        reserve["stack_is_prequential"] = True
        control_bundle = _fit_bundle(reserve, activation=activation)
        daily = _daily_calibration_residuals(reserve, provisional_bundle=control_bundle)
        mapped_control, _ = apply_strict_r3_ev_bridge(held_group.copy(), bundle=control_bundle)
        result = mapped_control.loc[:, [
            "candidate_id", "__decision_ts__", "policy_path_valid", "policy_gross_bps",
            "policy_net_bps", "policy_label_available_ts", "final_score",
            "conversion_bundle_sha256", "upstream_bundle_sha256", "geometry_bundle_sha256",
        ]].copy()
        result["calibration_activation_ts"] = activation
        result["exact_reserve_control__expected_net_bps"] = mapped_control[
            "causal_21d_side_expected_net_bps"
        ].to_numpy(float)
        result["exact_reserve_control__admitted"] = result[
            "exact_reserve_control__expected_net_bps"
        ].ge(args.net_floor_bps).fillna(False)

        expected_existing = result["candidate_id"].map(control_expected).to_numpy(float)
        parity_delta = float(np.nanmax(np.abs(
            expected_existing - result["exact_reserve_control__expected_net_bps"].to_numpy(float)
        )))
        if parity_delta > 1e-9:
            raise AssertionError(f"recomputed exact-reserve control changed by {parity_delta}")

        for fraction in TRIM_FRACTIONS:
            arm = _arm_trim(fraction)
            keep_days = _keep_trimmed_days(daily, fraction)
            filtered = reserve.loc[reserve["__decision_ts__"].dt.normalize().isin(keep_days)].copy()
            bundle = _fit_bundle(filtered, activation=activation)
            mapped, _ = apply_strict_r3_ev_bridge(held_group.copy(), bundle=bundle)
            result[f"{arm}__expected_net_bps"] = mapped[
                "causal_21d_side_expected_net_bps"
            ].to_numpy(float)
            result[f"{arm}__admitted"] = result[f"{arm}__expected_net_bps"].ge(
                args.net_floor_bps,
            ).fillna(False)
            diagnostics.append({
                "activation_ts": activation, "arm": arm, "reserve_days": int(len(daily)),
                "retained_days": int(len(keep_days)), "reserve_rows": int(len(reserve)),
                "retained_rows": int(len(filtered)), "control_parity_max_abs_bps": parity_delta,
            })

        for cutoff in STD_CUTOFFS:
            arm = _arm_std(cutoff)
            keep_days = _keep_std_days(daily, cutoff)
            filtered = reserve.loc[reserve["__decision_ts__"].dt.normalize().isin(keep_days)].copy()
            bundle = _fit_bundle(filtered, activation=activation)
            mapped, _ = apply_strict_r3_ev_bridge(held_group.copy(), bundle=bundle)
            result[f"{arm}__expected_net_bps"] = mapped[
                "causal_21d_side_expected_net_bps"
            ].to_numpy(float)
            result[f"{arm}__admitted"] = result[f"{arm}__expected_net_bps"].ge(
                args.net_floor_bps,
            ).fillna(False)
            diagnostics.append({
                "activation_ts": activation, "arm": arm, "reserve_days": int(len(daily)),
                "retained_days": int(len(keep_days)), "reserve_rows": int(len(reserve)),
                "retained_rows": int(len(filtered)), "control_parity_max_abs_bps": parity_delta,
            })

        # The exact-producer reserve is available, resolved, and excluded from
        # every supervised fit.  Seed the state history with its calibration
        # residuals so trend/dispersion/entropy are usable from the producer's
        # first held day rather than manufacturing a three-day live cold start.
        reserve_state = reserve.loc[:, [
            "candidate_id", "__decision_ts__", "policy_label_available_ts",
            "policy_path_valid",
        ]].copy()
        reserve_state["ev_bridge_policy_residual_bps"] = (
            pd.to_numeric(reserve["policy_net_bps"], errors="coerce").to_numpy(float)
            - control_bundle.predict_prior(reserve)
        )
        held_state = mapped_control.loc[:, [
            "candidate_id", "__decision_ts__", "policy_label_available_ts",
            "policy_path_valid", "ev_bridge_policy_residual_bps",
        ]].copy()
        state = _daily_residual_state(pd.concat(
            [reserve_state, held_state], ignore_index=True,
        )).iloc[len(reserve_state):].reset_index(drop=True)
        prior = mapped_control["ev_bridge_prior_expected_net_bps"].to_numpy(float)
        for column in state.columns:
            if column != "__position__":
                result[column] = state[column].to_numpy(float)
        for arm in STATE_ARMS:
            result[f"{arm}__expected_net_bps"] = _state_expected(prior, state, arm)
            result[f"{arm}__admitted"] = result[f"{arm}__expected_net_bps"].ge(
                args.net_floor_bps,
            ).fillna(False)
            diagnostics.append({
                "activation_ts": activation, "arm": arm, "reserve_days": int(len(daily)),
                "retained_days": int(len(daily)), "reserve_rows": int(len(reserve)),
                "retained_rows": int(len(reserve)), "control_parity_max_abs_bps": parity_delta,
            })
        parts.append(result)

    output = pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any() or len(output) != len(held):
        raise AssertionError("robust EV-map ablation changed held candidate identity")
    metrics = _metric_rows(output, arms=all_arms)
    daily_admission = output.assign(
        day=pd.to_datetime(output["__decision_ts__"], utc=True).dt.normalize(),
    ).groupby("day", observed=True, sort=True).agg(**{
        f"{arm}__admitted_rows": (f"{arm}__admitted", "sum") for arm in all_arms
    }).reset_index()

    args.out_dir.mkdir(parents=True)
    selection_columns = ["candidate_id", "__decision_ts__", "final_score"]
    for arm in all_arms:
        selection_columns.extend([f"{arm}__expected_net_bps", f"{arm}__admitted"])
    output.loc[:, selection_columns].to_parquet(
        args.out_dir / "robust_ev_map_selection.parquet", index=False, compression="zstd",
    )
    output.loc[:, [
        "candidate_id", "__decision_ts__", "calibration_activation_ts",
        *[column for column in output if column.startswith("ev_state_")],
    ]].to_parquet(args.out_dir / "causal_ev_state_features.parquet", index=False, compression="zstd")
    metrics.to_parquet(args.out_dir / "robust_ev_map_metrics.parquet", index=False)
    daily_admission.to_parquet(args.out_dir / "daily_admission_continuity.parquet", index=False)
    pd.DataFrame(diagnostics).to_parquet(args.out_dir / "producer_filter_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_robust_exact_reserve_ev_map_ablation_v1",
        "promotion_status": "research_ablation_only",
        "reference_scores": str(args.reference_scores),
        "reference_scores_sha256": _sha(args.reference_scores),
        "policy_outcomes": str(args.policy_outcomes),
        "policy_outcomes_sha256": _sha(args.policy_outcomes),
        "held_ledger": str(args.held_ledger),
        "held_ledger_sha256": _sha(args.held_ledger),
        "control_predictions": str(args.control_predictions),
        "control_predictions_sha256": _sha(args.control_predictions),
        "rows": int(len(output)),
        "producer_groups": int(output["calibration_activation_ts"].nunique()),
        "net_floor_bps": float(args.net_floor_bps),
        "arms": all_arms,
        "robust_day_rule": (
            "fit provisional same-producer reserve map; aggregate its exact policy-net "
            "calibration residual by UTC decision day; symmetrically remove declared "
            "robust-z ranks; refit on retained days"
        ),
        "std_day_rule": (
            "ordinary mean/std z score of same-producer daily calibration residual; "
            "retain days inside declared sigma cutoff"
        ),
        "state_rule": (
            "per held UTC day and exact producer, use only outcomes whose labels resolved "
            "before day start; 21-day daily residual level/trend/slope/std/sign entropy"
        ),
        "matched_contract": (
            "unchanged strict-R3 scores, exact producer lineage, frozen geometry/K9, "
            "SimplePolicyOptimiser policy outcomes, +50 bps floor, and candidate IDs"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
