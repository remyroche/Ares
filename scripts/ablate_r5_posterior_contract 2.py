#!/usr/bin/env python3
"""Matched causal audit of R5 posterior-domain and decision-rule repairs.

The script never selects or promotes an arm.  It compares predeclared arms on
identical monthly OOS candidate identities.  Calibration for month ``t`` uses
only earlier monthly OOS predictions whose policy labels resolved before the
month starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import HuberRegressor


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
MIN_CALIBRATION_ROWS = 2_000
CALIBRATION_CAP = 120_000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("prediction argument must be arm=/path/to/predictions.parquet")
    arm, path = value.split("=", 1)
    return arm, Path(path)


def _load(paths: Iterable[Path], arm: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{arm} contains duplicate candidate IDs")
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    return frame.reset_index(drop=True)


def _equal_month_cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    months = frame["__decision_ts__"].dt.strftime("%Y-%m")
    groups = list(frame.groupby(months, sort=True))
    per_month = max(1, cap // len(groups))
    parts = []
    for _month, group in groups:
        if len(group) <= per_month:
            parts.append(group)
        else:
            index = np.linspace(0, len(group) - 1, per_month).round().astype(int)
            parts.append(group.iloc[index])
    result = pd.concat(parts, ignore_index=True)
    return result.iloc[:cap].copy()


def prequential_calibration(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Calibrate posterior mean and scale using prior resolved OOS rows only."""

    calibrated = np.full(len(frame), np.nan, dtype=float)
    probability = np.full(len(frame), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for month, held in frame.groupby("month", sort=True):
        held_index = held.index.to_numpy()
        cutoff = held["__decision_ts__"].min().floor("D")
        prior = frame.loc[
            frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame["posterior_expected_bps"], errors="coerce"))
        ].copy()
        prior = _equal_month_cap(prior, CALIBRATION_CAP)
        slope, intercept, scale, status = 1.0, 0.0, 1.0, "identity_cold_start"
        if len(prior) >= MIN_CALIBRATION_ROWS:
            x = pd.to_numeric(prior["posterior_expected_bps"], errors="raise").to_numpy(float)
            y = pd.to_numeric(prior["policy_net_bps"], errors="raise").to_numpy(float)
            model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=250).fit(x[:, None], y)
            candidate_slope = float(model.coef_[0])
            if np.isfinite(candidate_slope) and candidate_slope >= 0.0:
                slope = candidate_slope
                intercept = float(model.intercept_)
                fitted = intercept + slope * x
                sd = np.maximum(
                    pd.to_numeric(prior["posterior_predictive_sd"], errors="raise").to_numpy(float),
                    1.0,
                )
                z80 = float(np.quantile(np.abs(y - fitted) / sd, 0.80, method="linear"))
                scale = float(np.clip(z80 / 1.2815515655446004, 0.25, 4.0))
                status = "prior_oos_huber_and_80pct_scale"
            else:
                status = "identity_nonmonotonic_fit_rejected"
        held_mean = pd.to_numeric(held["posterior_expected_bps"], errors="coerce").to_numpy(float)
        held_sd = np.maximum(
            pd.to_numeric(held["posterior_predictive_sd"], errors="coerce").to_numpy(float) * scale,
            1.0,
        )
        calibrated[held_index] = intercept + slope * held_mean
        probability[held_index] = ndtr(calibrated[held_index] / held_sd)
        rows.append({
            "month": month, "cutoff": cutoff, "prior_oos_rows": int(len(prior)),
            "slope": slope, "intercept": intercept, "predictive_sd_scale": scale,
            "status": status,
        })
    return calibrated, probability, pd.DataFrame(rows)


def _period_metrics(frame: pd.DataFrame, arm: str, expected: str, admitted: str) -> pd.DataFrame:
    work = frame.copy()
    work["week"] = work["__decision_ts__"].dt.strftime("%G-W%V")
    periods = [("all", "all", work)]
    periods.extend(("month", str(k), v) for k, v in work.groupby("month", sort=True))
    periods.extend(("week", str(k), v) for k, v in work.groupby("week", sort=True))
    output: list[dict[str, object]] = []
    for scope, period, block in periods:
        valid_mask = (
            block["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(block["policy_net_bps"], errors="coerce"))
        )
        admitted_pool = block.loc[block[admitted].fillna(False).astype(bool)].sort_values(
            [expected, "final_score", "candidate_id"], ascending=[False, False, True], kind="stable",
        )
        selections: list[tuple[str, pd.DataFrame]] = [("all_admitted", admitted_pool)]
        selections.extend((
            f"admitted_top_{tail:g}",
            admitted_pool.head(max(1, int(math.ceil(tail * len(admitted_pool)))))
            if len(admitted_pool) else admitted_pool,
        ) for tail in TAILS)
        population = block.sort_values(
            [expected, "final_score", "candidate_id"], ascending=[False, False, True], kind="stable",
        )
        selections.extend((
            f"population_top_{tail:g}",
            population.head(max(1, int(math.ceil(tail * len(population)))))
            if len(population) else population,
        ) for tail in TAILS)
        for kind, selected in selections:
            valid = selected.loc[valid_mask.reindex(selected.index, fill_value=False)]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            output.append({
                "arm": arm, "period_scope": scope, "period": period, "kind": kind,
                "score_rows": int(len(block)), "admitted_rows": int(len(admitted_pool)),
                "selected_rows": int(len(selected)), "valid_outcomes": int(len(valid)),
                "outcome_coverage": float(len(valid) / max(len(selected), 1)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "median_net_bps": float(net.median()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
                "net_standard_error": float(net.std(ddof=1) / math.sqrt(len(net))) if len(net) > 1 else np.nan,
            })
    return pd.DataFrame(output)


def _stability(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    monthly = metrics.loc[metrics["period_scope"].eq("month")]
    pooled = metrics.loc[metrics["period_scope"].eq("all") & metrics["period"].eq("all")]
    for (arm, kind), group in monthly.groupby(["arm", "kind"], sort=True):
        value = pd.to_numeric(group["net_bps_per_trade"], errors="coerce").dropna().to_numpy(float)
        overall = pooled.loc[pooled["arm"].eq(arm) & pooled["kind"].eq(kind), "net_bps_per_trade"]
        median = float(np.median(value)) if len(value) else np.nan
        rows.append({
            "arm": arm, "kind": kind,
            "pooled_net_bps_per_trade": float(overall.iloc[0]) if len(overall) else np.nan,
            "months": int(len(value)), "positive_months": int(np.sum(value > 0.0)),
            "worst_month_net_bps": float(np.min(value)) if len(value) else np.nan,
            "median_month_net_bps": median,
            "month_mad_bps": float(np.median(np.abs(value - median))) if len(value) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    parsed = [_parse(value) for value in args.prediction]
    grouped: dict[str, list[Path]] = {}
    for arm, path in parsed:
        grouped.setdefault(arm, []).append(path)
    required = {"A0_current", "A2_mixed_weighted", "A3_mixed_neutral", "A4_independent_local"}
    if set(grouped) != required:
        raise ValueError(f"expected exactly {sorted(required)}, got {sorted(grouped)}")
    loaded = {arm: _load(paths, arm) for arm, paths in grouped.items()}
    identity = loaded["A0_current"][["candidate_id", "__decision_ts__"]]
    for arm, frame in loaded.items():
        if not frame[["candidate_id", "__decision_ts__"]].reset_index(drop=True).equals(
            identity.reset_index(drop=True)
        ):
            raise ValueError(f"candidate identity mismatch for {arm}")
    base = loaded["A0_current"].copy()
    # Copy challenger outputs onto one immutable identity/outcome ledger.
    for arm, frame in loaded.items():
        token = arm.lower()
        for field in (
            "posterior_expected_bps", "posterior_predictive_sd", "p_ev_positive",
            "trust_effective_support", "timestamp_top30",
        ):
            base[f"{token}__{field}"] = frame[field].to_numpy()

    arm_contracts: dict[str, tuple[str, str]] = {}
    for arm in ("A0_current", "A2_mixed_weighted", "A3_mixed_neutral", "A4_independent_local"):
        token = arm.lower()
        expected = f"{token}__expected"
        admitted = f"{token}__admitted"
        base[expected] = pd.to_numeric(base[f"{token}__posterior_expected_bps"], errors="coerce")
        base[admitted] = base[expected].ge(50.0)
        arm_contracts[arm] = (expected, admitted)
    base["a1_domain_gated__expected"] = base["a0_current__expected"]
    base["a1_domain_gated__admitted"] = (
        base["a0_current__admitted"] & base["a0_current__timestamp_top30"].fillna(False).astype(bool)
    )
    arm_contracts["A1_current_domain_gated"] = (
        "a1_domain_gated__expected", "a1_domain_gated__admitted",
    )

    calibration_input = loaded["A4_independent_local"].copy()
    calibrated, calibrated_probability, calibration = prequential_calibration(calibration_input)
    base["a5_calibrated__expected"] = calibrated
    base["a5_calibrated__p_positive"] = calibrated_probability
    base["a5_calibrated__admitted"] = base["a5_calibrated__expected"].ge(50.0)
    arm_contracts["A5_prequential_calibrated"] = (
        "a5_calibrated__expected", "a5_calibrated__admitted",
    )
    base["a6_conservative__expected"] = base["a5_calibrated__expected"]
    base["a6_conservative__admitted"] = (
        base["a5_calibrated__admitted"] & base["a5_calibrated__p_positive"].ge(0.60)
    )
    arm_contracts["A6_calibrated_p60"] = (
        "a6_conservative__expected", "a6_conservative__admitted",
    )

    raw = pd.to_numeric(base["raw_expected_bps"], errors="coerce").to_numpy(float)
    delta = calibrated - raw
    base["a7_demotion_only__expected"] = raw + np.minimum(delta, 0.0)
    base["a7_demotion_only__admitted"] = base["a7_demotion_only__expected"].ge(50.0)
    arm_contracts["A7_calibrated_demotion_only"] = (
        "a7_demotion_only__expected", "a7_demotion_only__admitted",
    )
    base["a8_capped_promotion__expected"] = raw + np.where(delta < 0.0, delta, np.minimum(delta, 50.0))
    base["a8_capped_promotion__admitted"] = base["a8_capped_promotion__expected"].ge(50.0)
    arm_contracts["A8_full_demotion_capped50_promotion"] = (
        "a8_capped_promotion__expected", "a8_capped_promotion__admitted",
    )
    support = pd.to_numeric(base["a4_independent_local__trust_effective_support"], errors="coerce")
    promote = (delta > 0.0) & base["a5_calibrated__p_positive"].ge(0.70).to_numpy(bool) & support.ge(300.0).to_numpy(bool)
    asymmetric_delta = np.where(delta < 0.0, delta, np.where(promote, np.minimum(delta, 50.0), 0.0))
    base["a9_strict_promotion__expected"] = raw + asymmetric_delta
    base["a9_strict_promotion__admitted"] = base["a9_strict_promotion__expected"].ge(50.0)
    arm_contracts["A9_full_demotion_strict_promotion"] = (
        "a9_strict_promotion__expected", "a9_strict_promotion__admitted",
    )

    metrics = pd.concat([
        _period_metrics(base, arm, expected, admitted)
        for arm, (expected, admitted) in arm_contracts.items()
    ], ignore_index=True)
    stability = _stability(metrics)
    domain = base.assign(
        outside_top30=~base["a0_current__timestamp_top30"].fillna(False).astype(bool),
        current_admitted=base["a0_current__admitted"],
    ).groupby(["month", "outside_top30"], sort=True).agg(
        rows=("candidate_id", "size"), admitted=("current_admitted", "sum"),
        valid=("policy_path_valid", "sum"), net_bps=("policy_net_bps", "mean"),
    ).reset_index()
    interval = []
    for arm in ("A0_current", "A4_independent_local"):
        token = arm.lower()
        mean = pd.to_numeric(base[f"{token}__posterior_expected_bps"], errors="coerce")
        sd = pd.to_numeric(base[f"{token}__posterior_predictive_sd"], errors="coerce")
        actual = pd.to_numeric(base["policy_net_bps"], errors="coerce")
        valid = base["policy_path_valid"].fillna(False).astype(bool) & mean.notna() & sd.gt(0) & actual.notna()
        z = ((actual[valid] - mean[valid]) / sd[valid]).abs()
        interval.append({
            "arm": arm, "rows": int(valid.sum()), "within_50pct": float(z.le(0.67449).mean()),
            "within_80pct": float(z.le(1.28155).mean()), "within_90pct": float(z.le(1.64485).mean()),
            "median_predictive_sd": float(sd[valid].median()),
        })
    args.out_dir.mkdir(parents=True)
    base.to_parquet(args.out_dir / "selection_ledger.parquet", index=False)
    metrics.to_parquet(args.out_dir / "metrics.parquet", index=False)
    stability.to_parquet(args.out_dir / "monthly_stability.parquet", index=False)
    calibration.to_parquet(args.out_dir / "prequential_calibration.parquet", index=False)
    domain.to_parquet(args.out_dir / "domain_audit.parquet", index=False)
    pd.DataFrame(interval).to_parquet(args.out_dir / "interval_calibration.parquet", index=False)
    manifest = {
        "schema": "r5_posterior_contract_ablation_v1",
        "inputs": {arm: [{"path": str(p), "sha256": _sha(p)} for p in paths] for arm, paths in grouped.items()},
        "arms": {arm: {"expected": pair[0], "admitted": pair[1]} for arm, pair in arm_contracts.items()},
        "threshold_bps": 50.0, "tails": list(TAILS),
        "calibration": "month t uses only earlier OOS rows with label_available_ts < month t",
        "winner_promoted": False,
        "outcomes_used_only_after_causal_score_and_admission_construction": True,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(base), "arms": list(arm_contracts)}))


if __name__ == "__main__":
    main()
