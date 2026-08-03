#!/usr/bin/env python3
"""Remove cross-era score-scale drift with complete-group rank coordinates."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_cross_era_wait10_transition_ablation as engine
from scripts.run_cross_era_wait10_transition_ablation_v2 import (
    EXPANDED_TRANSITIONS,
)

PANEL_ROOT = (
    ROOT
    / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v2"
)
OUT = (
    ROOT
    / "data_perp/artifacts/cross_era_wait10_rank_normalized_ablation_20260730_v1"
)
SCHEMA = "cross_era_wait10_rank_normalized_ablation_v1"
IDENTITY = engine.IDENTITY
WEIGHTS = engine.WEIGHTS
RAW_SCORES = engine.SCORE_COMMON
RANK_FEATURES = (
    "base_rank_pct_timestamp_side_cross_era",
    "base_score_z_timestamp_side_cross_era",
    "residual_rank_pct_timestamp_side_cross_era",
    "residual_score_z_timestamp_side_cross_era",
    "residual_minus_base_rank_pct_cross_era",
)
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "rank_common": RANK_FEATURES,
    "transition_expanded": EXPANDED_TRANSITIONS,
    "rank_plus_transition_expanded": (*RANK_FEATURES, *EXPANDED_TRANSITIONS),
}


class RankContractError(RuntimeError):
    pass


def add_complete_group_rank_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*IDENTITY, *RAW_SCORES}
    if not required.issubset(frame.columns):
        raise RankContractError("complete candidate panel lacks common score fields")
    result = frame.copy()
    keys = ["__ts__", "side_name"]
    groups = result.groupby(keys, sort=False)
    base = pd.to_numeric(result["score_base_alpha"], errors="raise")
    residual = pd.to_numeric(result["score_residual_expected_ev"], errors="raise")
    result["base_rank_pct_timestamp_side_cross_era"] = groups[
        "score_base_alpha"
    ].rank(method="average", ascending=False, pct=True)
    result["residual_rank_pct_timestamp_side_cross_era"] = groups[
        "score_residual_expected_ev"
    ].rank(method="average", ascending=False, pct=True)
    base_mean = groups["score_base_alpha"].transform("mean")
    base_std = groups["score_base_alpha"].transform("std").replace(0.0, np.nan)
    residual_mean = groups["score_residual_expected_ev"].transform("mean")
    residual_std = groups["score_residual_expected_ev"].transform("std").replace(
        0.0, np.nan
    )
    result["base_score_z_timestamp_side_cross_era"] = (
        (base - base_mean) / base_std
    ).fillna(0.0)
    result["residual_score_z_timestamp_side_cross_era"] = (
        (residual - residual_mean) / residual_std
    ).fillna(0.0)
    result["residual_minus_base_rank_pct_cross_era"] = (
        result["residual_rank_pct_timestamp_side_cross_era"]
        - result["base_rank_pct_timestamp_side_cross_era"]
    )
    values = result.loc[:, list(RANK_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values[:, 0] <= 0) | (values[:, 0] > 1)).any():
        raise RankContractError("invalid complete-group rank coordinates")
    return result


def verify_current_panel(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "canonical_execution_reliability_input_v2"
        or engine.sha256(root / "panel.parquet")
        != manifest.get("outputs_sha256", {}).get("panel.parquet")
    ):
        raise RankContractError("canonical current complete-group panel does not verify")
    return manifest


def _load_inputs(
    historical_root: Path,
    current_root: Path,
    handoff_root: Path,
    calendar_root: Path,
    panel_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    engine.TRANSITION_COMMON = EXPANDED_TRANSITIONS
    historical_manifest = engine.verify_artifact(
        historical_root,
        "historical_oof_2023apr_2024_current_policy_wait10_action_v1",
    )
    current_manifest = engine.verify_artifact(
        current_root, "febapr2025_current_policy_wait10_action_v1"
    )
    handoff_manifest = engine.verify_artifact(
        handoff_root, "frozen_entry_action_handoff_v2"
    )
    panel_manifest = verify_current_panel(panel_root)
    calendar, calendar_provenance = engine.load_calendar(calendar_root)

    labels = pd.read_parquet(historical_root / "action_labels.parquet")
    scores = pd.read_parquet(
        historical_root / "preentry_features.parquet",
        columns=[*IDENTITY, *RAW_SCORES],
    )
    historical = labels.merge(scores, on=list(IDENTITY), validate="one_to_one")
    historical = add_complete_group_rank_coordinates(historical)
    historical = engine.join_calendar(historical, calendar)
    historical["execution_decision_utc"] = pd.to_datetime(
        historical["execution_decision_utc"], utc=True
    )
    historical["execution_label_end_utc"] = pd.to_datetime(
        historical["execution_label_end_utc"], utc=True
    )

    panel = pd.read_parquet(
        panel_root / "panel.parquet", columns=[*IDENTITY, *RAW_SCORES]
    )
    panel = add_complete_group_rank_coordinates(panel)
    handoff = pd.read_parquet(
        handoff_root / "handoff.parquet",
        columns=[*IDENTITY, "candidate_month", *WEIGHTS],
    )
    ranked_handoff = handoff.merge(
        panel.loc[:, ["candidate_id", *RANK_FEATURES]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if ranked_handoff[list(RANK_FEATURES)].isna().any().any():
        raise RankContractError("frozen book lacks complete-group rank coordinates")
    current_labels = pd.read_parquet(current_root / "action_labels.parquet")
    evaluation = ranked_handoff.merge(
        current_labels.drop(columns=["candidate_month"]),
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if evaluation["wait_delta"].isna().any():
        raise RankContractError("frozen book lacks current exact action labels")
    evaluation = engine.join_calendar(evaluation, calendar)
    evaluation["execution_decision_utc"] = pd.to_datetime(
        evaluation["execution_decision_utc"], utc=True
    )
    evaluation["execution_label_end_utc"] = pd.to_datetime(
        evaluation["execution_label_end_utc"], utc=True
    )
    provenance = {
        "historical_manifest_sha256": engine.sha256(
            historical_root / "manifest.json"
        ),
        "historical_labels_sha256": historical_manifest["outputs_sha256"][
            "action_labels.parquet"
        ],
        "current_manifest_sha256": engine.sha256(current_root / "manifest.json"),
        "current_labels_sha256": current_manifest["outputs_sha256"][
            "action_labels.parquet"
        ],
        "handoff_manifest_sha256": engine.sha256(handoff_root / "manifest.json"),
        "handoff_sha256": handoff_manifest["outputs_sha256"]["handoff.parquet"],
        "complete_group_panel_manifest_sha256": engine.sha256(
            panel_root / "manifest.json"
        ),
        "complete_group_panel_sha256": panel_manifest["outputs_sha256"][
            "panel.parquet"
        ],
        **calendar_provenance,
    }
    return historical, evaluation, provenance


def run(
    historical_root: Path = engine.HISTORICAL_ROOT,
    current_root: Path = engine.CURRENT_ROOT,
    handoff_root: Path = engine.HANDOFF_ROOT,
    calendar_root: Path = engine.CALENDAR_ROOT,
    panel_root: Path = PANEL_ROOT,
    output: Path = OUT,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    historical, evaluation, provenance = _load_inputs(
        historical_root,
        current_root,
        handoff_root,
        calendar_root,
        panel_root,
    )
    predictions: list[pd.DataFrame] = []
    calibration_records: list[dict[str, Any]] = []
    shift_records: list[dict[str, Any]] = []
    for source_index, (source_name, source_rows) in enumerate(
        engine.historical_sources(historical).items()
    ):
        for feature_index, (feature_name, features) in enumerate(FEATURE_SETS.items()):
            for side_index, side in enumerate(("long", "short")):
                train = source_rows.loc[source_rows["side_name"].eq(side)].copy()
                core_indices, calibration_indices = engine.calibration_split(train)
                core = train.iloc[core_indices].copy()
                calibration = train.iloc[calibration_indices].copy()
                x_core, core_medians = engine.prepare_x(core, features)
                x_calibration, _ = engine.prepare_x(
                    calibration, features, core_medians
                )
                seed = (
                    20260730
                    + source_index * 1_000
                    + feature_index * 100
                    + side_index * 10
                )
                calibration_models = engine.fit_bundle(
                    x_core, core["wait_delta"].to_numpy(dtype=float), seed
                )
                calibration_prediction = engine.predict_bundle(
                    calibration_models, x_calibration
                )
                calibration_scored = pd.concat(
                    [
                        calibration.loc[
                            :, ["execution_decision_utc", "wait_delta"]
                        ].reset_index(drop=True),
                        calibration_prediction,
                    ],
                    axis=1,
                )
                threshold, audit = engine.choose_weighted_threshold(
                    calibration_scored
                )
                calibration_records.append(
                    {
                        "training_source": source_name,
                        "feature_set": feature_name,
                        "side_name": side,
                        "source_rows": len(train),
                        "core_rows": len(core),
                        "calibration_rows": len(calibration),
                        "core_label_end_max_utc": core[
                            "execution_label_end_utc"
                        ].max(),
                        "calibration_start_utc": calibration[
                            "execution_decision_utc"
                        ].min(),
                        "strict_resolution_before_calibration": bool(
                            core["execution_label_end_utc"].max()
                            < calibration["execution_decision_utc"].min()
                        ),
                        "selected_threshold": threshold,
                        "threshold_selection": audit["selection"],
                        "threshold_audit": json.dumps(
                            engine.safe(audit), sort_keys=True
                        ),
                    }
                )
                x_train, medians = engine.prepare_x(train, features)
                models = engine.fit_bundle(
                    x_train, train["wait_delta"].to_numpy(dtype=float), seed + 50
                )
                for month in ("2025-03", "2025-04"):
                    valid = evaluation.loc[
                        evaluation["candidate_month"].eq(month)
                        & evaluation["side_name"].eq(side)
                    ].copy()
                    x_valid, _ = engine.prepare_x(valid, features, medians)
                    prediction = engine.predict_bundle(models, x_valid)
                    scored = pd.concat(
                        [valid.reset_index(drop=True), prediction], axis=1
                    )
                    scored["training_source"] = source_name
                    scored["feature_set"] = feature_name
                    scored["evaluation"] = f"{month}_frozen_global_book"
                    scored["calibrated_threshold"] = threshold
                    predictions.append(scored)
                    shift_records.extend(
                        engine.feature_shift_records(
                            train,
                            valid,
                            features,
                            source=source_name,
                            feature_set=feature_name,
                            evaluation=f"{month}_frozen_global_book",
                            side=side,
                        )
                    )

    ledger = pd.concat(predictions, ignore_index=True)
    prediction_columns = [
        "pred_direct_delta",
        "pred_q25_delta",
        "pred_event_probability",
        "pred_weighted_event_score",
        "pred_positive_delta",
        "pred_negative_delta",
        "pred_soft_score",
        "pred_expected_delta",
    ]
    if not np.isfinite(ledger[prediction_columns].to_numpy(dtype=float)).all():
        raise RankContractError("rank-normalized predictions are non-finite")
    calibration_frame = pd.DataFrame(calibration_records)
    if not calibration_frame["strict_resolution_before_calibration"].all():
        raise RankContractError("rank-normalized calibration uses unresolved labels")
    outputs_frame = {
        "calibration_audit.csv": calibration_frame,
        "policy_metrics.csv": engine.economics(ledger),
        "head_metrics.csv": engine.head_metrics(ledger),
        "daily_bootstrap_ci_top10.csv": engine.bootstrap_top10(ledger),
        "feature_shift.csv": pd.DataFrame(shift_records),
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, frame in outputs_frame.items():
            frame.to_csv(temporary / name, index=False)
        retained = [
            *IDENTITY,
            "candidate_month",
            "execution_decision_utc",
            *WEIGHTS,
            "enter_now_gross",
            "enter_now_cost",
            "enter_now_net",
            "wait10_gross",
            "wait10_cost",
            "wait10_net",
            "wait_delta",
            "training_source",
            "feature_set",
            "evaluation",
            "calibrated_threshold",
            *prediction_columns,
        ]
        ledger.loc[:, retained].to_parquet(
            temporary / "action_predictions.parquet",
            index=False,
            compression="zstd",
        )
        output_names = [*outputs_frame, "action_predictions.parquet"]
        outputs_sha = {
            name: engine.sha256(temporary / name) for name in output_names
        }
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_CROSS_ERA_RANK_NORMALIZED_DIAGNOSTIC_NO_PROMOTION",
            "contract": {
                "training": "Apr-2023--Dec-2024 exact labels; all-period, 2024-only and Q4-2024 sources",
                "score_normalization": "descending percentile rank and z-score computed within each complete timestamp-side candidate group in each era; frozen selected rows are joined only after current complete-panel coordinates exist",
                "transitions": "expanded causal transition subtype fields from the same sealed hourly calendar in both eras",
                "evaluation": "unchanged March/April 2025 frozen pooled-global books and weights; no reranking or reconstruction",
                "selection": "no model, feature arm or threshold may be promoted from reused months",
            },
            "historical_rows": len(historical),
            "frozen_rows": len(evaluation),
            "prediction_rows": len(ledger),
            "frozen_identity_sha256": engine.identity_digest(evaluation),
            "rank_features": list(RANK_FEATURES),
            "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
            "training_sources": list(engine.historical_sources(historical)),
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "input_provenance": provenance,
            "outputs_sha256": outputs_sha,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": engine.sha256(Path(__file__).resolve()),
                "shared_engine_path": str(Path(engine.__file__).resolve()),
                "shared_engine_sha256": engine.sha256(Path(engine.__file__).resolve()),
            },
        }
        engine.write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(
            f"{engine.sha256(temporary / 'manifest.json')}  manifest.json\n"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--historical-root", type=Path, default=engine.HISTORICAL_ROOT)
    result.add_argument("--current-root", type=Path, default=engine.CURRENT_ROOT)
    result.add_argument("--handoff-root", type=Path, default=engine.HANDOFF_ROOT)
    result.add_argument("--calendar-root", type=Path, default=engine.CALENDAR_ROOT)
    result.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            engine.safe(
                run(
                    args.historical_root,
                    args.current_root,
                    args.handoff_root,
                    args.calendar_root,
                    args.panel_root,
                    args.output,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
