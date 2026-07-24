#!/usr/bin/env python3
"""Fit frozen same-day failure-detector bundles for forward scoring.

The research detector writes chronological OOS predictions only.  This utility
fits the equivalent local ``negative_ev_day`` detector once at a chosen
boundary, preserving its train-only feature screen, scaling, Platt calibration
and 95% inner-validation threshold.  It deliberately does not score rows or
accept outcome columns; a separate scorer consumes observable daily state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    ProspectiveFailureDetectorConfig,
    add_causal_state_dynamics,
    attach_failure_mode_targets,
    fit_frozen_same_day_detector,
    is_batch_layout_dependent_ae_gmm_feature,
)
from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    validate_inference_feature_columns,
)


DEFAULT_TAXONOMY = Path("data_perp/reports/failure_episode_taxonomy_20260719_v17_three_year_taxonomy")
DEFAULT_OUTPUT = Path("data_perp/reports/prospective_failure_mode_detection_20260719_v7_three_year/final_forward_bundle")


def _observable(state: pd.DataFrame) -> list[str]:
    keys = {"day", "side_name", "archetype_policy_key"}
    columns = [
        name for name in state
        if name not in keys and not is_batch_layout_dependent_ae_gmm_feature(name)
    ]
    validate_inference_feature_columns(columns)
    return columns


def run(args: argparse.Namespace) -> dict[str, object]:
    root, output = Path(args.taxonomy), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    requested = {
        tuple(value.strip().split("::", 1))
        for value in str(args.only_keys or "").split(",")
        if "::" in value
    }
    state = pd.read_parquet(root / "daily_observable_state.parquet")
    # Dynamics are calculated within side x archetype groups. Filter before
    # the expensive market-geometry expansion when this is a targeted forward
    # replay; this is numerically identical for the retained local cells.
    if requested:
        state = state.loc[
            state.apply(
                lambda row: (str(row["side_name"]), str(row["archetype_policy_key"]))
                in requested,
                axis=1,
            )
        ].copy()
    state = add_causal_state_dynamics(
        state, lookback_days=int(args.inner_validation_days), add_market_geometry=True
    )
    observable = _observable(state)
    calendar = pd.read_parquet(root / "local_adverse_calendar.parquet")
    assignments = pd.read_parquet(root / str(args.assignments))
    labelled = attach_failure_mode_targets(
        state, calendar, assignments, lead_days=tuple(int(x) for x in args.lead_days.split(",") if x)
    )
    config = ProspectiveFailureDetectorConfig(
        min_train_days=int(args.min_train_days),
        eval_days=int(args.eval_days),
        inner_validation_days=int(args.inner_validation_days),
        min_positive_days=int(args.min_positive_days),
        max_features=int(args.max_features),
        alert_quantile=float(args.alert_quantile),
        probability_calibration=str(args.probability_calibration),
        lead_days=tuple(int(x) for x in args.lead_days.split(",") if x),
        embargo_days=int(args.embargo_days),
    )
    boundary = pd.Timestamp(args.boundary)
    if boundary.tzinfo is None:
        raise ValueError("--boundary must be timezone-aware, e.g. 2026-07-18T00:00:00Z")
    bundles = {}
    reports = []
    for side, archetype in labelled[["side_name", "archetype_policy_key"]].drop_duplicates().itertuples(index=False):
        if requested and (str(side), str(archetype)) not in requested:
            continue
        bundle = fit_frozen_same_day_detector(
            labelled,
            side_name=str(side),
            archetype_policy_key=str(archetype),
            boundary=boundary,
            config=config,
            feature_columns=observable,
        )
        key = f"{side}::{archetype}"
        if bundle is None:
            reports.append({"key": key, "status": "insufficient_label_available_support"})
            continue
        bundles[key] = bundle
        reports.append({
            "key": key, "status": "fit", "features": len(bundle.selected_features),
            "train_rows": bundle.train_rows, "positive_days": bundle.train_positive_days,
            "threshold": bundle.threshold, "calibration": bundle.calibration_method,
        })
    artifact = output / "local_same_day_negative_ev_forward_detectors.joblib"
    joblib.dump(bundles, artifact)
    pd.DataFrame(reports).to_csv(output / "fit_report.csv", index=False)
    manifest = {
        "schema": "frozen_forward_failure_detector_bundle_v1",
        "taxonomy": str(root.resolve()), "boundary": str(boundary),
        "target": "target__negative_ev_day", "state_feature_count": len(observable),
        "bundle_count": len(bundles), "bundle": str(artifact.resolve()),
        "requested_cells": sorted("::".join(key) for key in requested),
        "leakage_contract": "Each bundle is fit only on labels available before boundary; state inputs are observable at the day open. Missing forward state is rejected, never imputed as low risk.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assignments", default="local_frozen_failure_mode_assignments.parquet")
    parser.add_argument("--boundary", default="2026-07-18T00:00:00Z")
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--eval-days", type=int, default=45)
    parser.add_argument("--inner-validation-days", type=int, default=35)
    parser.add_argument("--min-positive-days", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--alert-quantile", type=float, default=0.95)
    parser.add_argument("--probability-calibration", default="platt")
    parser.add_argument("--lead-days", default="1,3")
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument(
        "--only-keys", default="",
        help="Comma-separated side::archetype cells. Empty fits every cell.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
