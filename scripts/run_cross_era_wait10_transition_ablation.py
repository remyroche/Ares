#!/usr/bin/env python3
"""Train Wait10 heads on 2023--2024 and score the frozen 2025 book.

The cross-era contract uses only score fields present with the same semantics
in both eras and raw causal regime-transition fields from one sealed hourly
calendar.  Historical OOF transition probabilities are evaluated separately
in their native era; their component geometry is not projected into 2025.
"""
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

from scripts.materialize_febapr_current_policy_wait10_action import (
    identity_digest,
    sha256,
)
from scripts.run_frozen_older_data_wait10_action_ablation import (
    POLICIES,
    bootstrap_top10,
    calibration_split,
    choose_weighted_threshold,
    economics,
    fit_bundle,
    head_metrics,
    predict_bundle,
    prepare_x,
)

HISTORICAL_ROOT = (
    ROOT
    / "data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1"
)
CURRENT_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_current_policy_wait10_action_20260730_v1"
)
HANDOFF_ROOT = ROOT / "data_perp/artifacts/frozen_entry_action_handoff_20260730_v2"
CALENDAR_ROOT = (
    ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
)
OUT = (
    ROOT
    / "data_perp/artifacts/cross_era_wait10_transition_ablation_20260730_v1"
)

SCHEMA = "cross_era_wait10_transition_ablation_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
WEIGHTS = ("weight_top_01", "weight_top_05", "weight_top_10", "weight_top_20")
SCORE_COMMON = (
    "score_base_alpha",
    "score_residual_expected_ev",
)
TRANSITION_COMMON = (
    "correlation_breakdown_dispersion",
    "correlation_heterogeneity_dispersion",
    "fragile_leverage_rebuild",
    "compressed_index_fragmented_assets",
    "deleveraging_without_followthrough",
    "fragmented_flush_recovery",
    "negative_breadth_pct",
    "peer_volatility_decoupling",
    "post_flush_leverage_rebuild",
    "thin_compression",
    "market_state_transition_entropy_5d",
    "market_state_persistence_5d",
    "recovery_failure_score_24h",
    "breakout_efficiency_4h",
    "breakout_participation_4h",
    "breakout_retention_4h",
    "breakout_disagreement_score",
    "transition_new__correlation_breakdown_dispersion__delta_3h",
    "transition_new__correlation_breakdown_dispersion__delta_12h",
    "transition_new__negative_breadth_pct__delta_3h",
    "transition_new__negative_breadth_pct__delta_12h",
    "transition_new__flush_recovery_state__delta_3h",
    "transition_new__flush_recovery_state__delta_12h",
    "transition_new__short_covering_score_market__delta_3h",
    "transition_new__short_covering_score_market__delta_12h",
)
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "score_common": SCORE_COMMON,
    "transition_common": TRANSITION_COMMON,
    "score_plus_transition_common": (*SCORE_COMMON, *TRANSITION_COMMON),
}


class CrossEraError(RuntimeError):
    pass


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_artifact(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise CrossEraError(f"sealed artifact required: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise CrossEraError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise CrossEraError(f"schema mismatch: {root}")
    for name, digest in manifest.get("outputs_sha256", {}).items():
        if sha256(root / name) != digest:
            raise CrossEraError(f"output hash mismatch: {root / name}")
    return manifest


def load_calendar(calendar_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    path = calendar_root / "hourly_state_calendar.parquet"
    manifest_path = calendar_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "regime_episode_calendar_v1"
        or sha256(path)
        != manifest.get("outputs_sha256", {}).get("hourly_state_calendar.parquet")
    ):
        raise CrossEraError("sealed hourly transition calendar does not verify")
    calendar = pd.read_parquet(path, columns=["source_utc", *TRANSITION_COMMON])
    calendar["source_utc"] = pd.to_datetime(calendar["source_utc"], utc=True)
    if calendar["source_utc"].duplicated().any():
        raise CrossEraError("hourly transition calendar source UTC is duplicated")
    return calendar, {
        "calendar_sha256": sha256(path),
        "calendar_manifest_sha256": sha256(manifest_path),
    }


def join_calendar(rows: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    result = rows.merge(
        calendar,
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    if result["source_utc"].isna().any():
        raise CrossEraError("common transition calendar coverage is incomplete")
    result = result.drop(columns=["source_utc"])
    if not np.isfinite(result[list(TRANSITION_COMMON)].to_numpy(dtype=float)).all():
        raise CrossEraError("joined common transition fields are non-finite")
    return result


def historical_sources(rows: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    return {
        "all_2023apr_2024": rows.copy(),
        "calendar_2024": rows.loc[
            rows["candidate_month"].astype(str).str.startswith("2024-")
        ].copy(),
        "q4_2024": rows.loc[
            rows["candidate_month"].isin(("2024-10", "2024-11", "2024-12"))
        ].copy(),
    }


def feature_shift_records(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    *,
    source: str,
    feature_set: str,
    evaluation: str,
    side: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in features:
        left = pd.to_numeric(train[feature], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(valid[feature], errors="raise").to_numpy(dtype=float)
        q01, q25, q50, q75, q99 = np.quantile(
            left, [0.01, 0.25, 0.50, 0.75, 0.99]
        )
        scale = max(float(q75 - q25), 1e-9)
        records.append(
            {
                "training_source": source,
                "feature_set": feature_set,
                "evaluation": evaluation,
                "side_name": side,
                "feature": feature,
                "train_rows": len(left),
                "evaluation_rows": len(right),
                "train_median": float(q50),
                "evaluation_median": float(np.median(right)),
                "median_shift_train_iqr": float((np.median(right) - q50) / scale),
                "evaluation_outside_train_1_99_rate": float(
                    np.mean((right < q01) | (right > q99))
                ),
            }
        )
    return records


def run(
    historical_root: Path = HISTORICAL_ROOT,
    current_root: Path = CURRENT_ROOT,
    handoff_root: Path = HANDOFF_ROOT,
    calendar_root: Path = CALENDAR_ROOT,
    output: Path = OUT,
) -> dict[str, Any]:
    historical_manifest = verify_artifact(
        historical_root,
        "historical_oof_2023apr_2024_current_policy_wait10_action_v1",
    )
    current_manifest = verify_artifact(
        current_root, "febapr2025_current_policy_wait10_action_v1"
    )
    handoff_manifest = verify_artifact(
        handoff_root, "frozen_entry_action_handoff_v2"
    )
    if output.exists():
        raise FileExistsError(output)
    calendar, calendar_provenance = load_calendar(calendar_root)

    historical_labels = pd.read_parquet(
        historical_root / "action_labels.parquet"
    )
    historical_features = pd.read_parquet(
        historical_root / "preentry_features.parquet",
        columns=[*IDENTITY, *SCORE_COMMON],
    )
    historical = historical_labels.merge(
        historical_features, on=list(IDENTITY), validate="one_to_one"
    )
    historical = join_calendar(historical, calendar)
    historical["execution_decision_utc"] = pd.to_datetime(
        historical["execution_decision_utc"], utc=True
    )
    historical["execution_label_end_utc"] = pd.to_datetime(
        historical["execution_label_end_utc"], utc=True
    )

    current_labels = pd.read_parquet(current_root / "action_labels.parquet")
    handoff_columns = [
        *IDENTITY,
        "candidate_month",
        *WEIGHTS,
        *SCORE_COMMON,
    ]
    handoff = pd.read_parquet(handoff_root / "handoff.parquet", columns=handoff_columns)
    evaluation = handoff.merge(
        current_labels.drop(columns=["candidate_month"]),
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if len(evaluation) != len(handoff) or evaluation["wait_delta"].isna().any():
        raise CrossEraError("current frozen book lacks exact action outcomes")
    evaluation = join_calendar(evaluation, calendar)
    evaluation["execution_decision_utc"] = pd.to_datetime(
        evaluation["execution_decision_utc"], utc=True
    )
    evaluation["execution_label_end_utc"] = pd.to_datetime(
        evaluation["execution_label_end_utc"], utc=True
    )

    predictions: list[pd.DataFrame] = []
    calibration_records: list[dict[str, Any]] = []
    shift_records: list[dict[str, Any]] = []
    for source_index, (source_name, source_rows) in enumerate(
        historical_sources(historical).items()
    ):
        for feature_index, (feature_name, features) in enumerate(FEATURE_SETS.items()):
            for side_index, side in enumerate(("long", "short")):
                train = source_rows.loc[source_rows["side_name"].eq(side)].copy()
                core_indices, calibration_indices = calibration_split(train)
                core = train.iloc[core_indices].copy()
                calibration = train.iloc[calibration_indices].copy()
                x_core, core_medians = prepare_x(core, features)
                x_calibration, _ = prepare_x(calibration, features, core_medians)
                seed = (
                    20260730
                    + source_index * 1_000
                    + feature_index * 100
                    + side_index * 10
                )
                calibrator = fit_bundle(
                    x_core, core["wait_delta"].to_numpy(dtype=float), seed
                )
                calibration_prediction = predict_bundle(
                    calibrator, x_calibration
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
                threshold, audit = choose_weighted_threshold(calibration_scored)
                calibration_records.append(
                    {
                        "training_source": source_name,
                        "feature_set": feature_name,
                        "side_name": side,
                        "source_rows": len(train),
                        "core_rows": len(core),
                        "calibration_rows": len(calibration),
                        "core_label_end_max_utc": core["execution_label_end_utc"].max(),
                        "calibration_start_utc": calibration[
                            "execution_decision_utc"
                        ].min(),
                        "strict_resolution_before_calibration": bool(
                            core["execution_label_end_utc"].max()
                            < calibration["execution_decision_utc"].min()
                        ),
                        "selected_threshold": threshold,
                        "threshold_selection": audit["selection"],
                        "threshold_audit": json.dumps(safe(audit), sort_keys=True),
                    }
                )
                x_train, medians = prepare_x(train, features)
                models = fit_bundle(
                    x_train, train["wait_delta"].to_numpy(dtype=float), seed + 50
                )
                for month in ("2025-03", "2025-04"):
                    valid = evaluation.loc[
                        evaluation["candidate_month"].eq(month)
                        & evaluation["side_name"].eq(side)
                    ].copy()
                    x_valid, _ = prepare_x(valid, features, medians)
                    prediction = predict_bundle(models, x_valid)
                    scored = pd.concat(
                        [valid.reset_index(drop=True), prediction], axis=1
                    )
                    scored["training_source"] = source_name
                    scored["feature_set"] = feature_name
                    scored["evaluation"] = f"{month}_frozen_global_book"
                    scored["calibrated_threshold"] = threshold
                    predictions.append(scored)
                    shift_records.extend(
                        feature_shift_records(
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
        raise CrossEraError("cross-era predictions contain non-finite values")
    calibration_frame = pd.DataFrame(calibration_records)
    if not calibration_frame["strict_resolution_before_calibration"].all():
        raise CrossEraError("cross-era calibration uses unresolved labels")
    metric_frame = economics(ledger)
    head_frame = head_metrics(ledger)
    bootstrap = bootstrap_top10(ledger)
    shift_frame = pd.DataFrame(shift_records)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        calibration_frame.to_csv(temporary / "calibration_audit.csv", index=False)
        metric_frame.to_csv(temporary / "policy_metrics.csv", index=False)
        head_frame.to_csv(temporary / "head_metrics.csv", index=False)
        bootstrap.to_csv(temporary / "daily_bootstrap_ci_top10.csv", index=False)
        shift_frame.to_csv(temporary / "feature_shift.csv", index=False)
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
        outputs = {
            name: sha256(temporary / name)
            for name in (
                "calibration_audit.csv",
                "policy_metrics.csv",
                "head_metrics.csv",
                "daily_bootstrap_ci_top10.csv",
                "feature_shift.csv",
                "action_predictions.parquet",
            )
        }
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_CROSS_ERA_DIAGNOSTIC_FROZEN_BOOK_NO_PROMOTION",
            "contract": {
                "training": "Apr-2023--Dec-2024 exact Wait10 labels with held-block OOF scores; all-period, 2024-only and Q4-2024 sources",
                "evaluation": "unchanged March/April 2025 pooled-global frozen books and fractional weights; no reranking or book reconstruction",
                "common_scores": "only score_base_alpha and score_residual_expected_ev, present with matching semantic roles in both eras",
                "common_transitions": "raw causal hourly state/transition fields from the same sealed calendar in both eras; historical fitted state-component probabilities are deliberately not projected",
                "abstention": "train-only day-cluster positive lower-bound rule; no reused-month threshold selection",
                "diagnostic": "historical base is frozen diagnostic lineage and results cannot promote an action",
            },
            "historical_rows": len(historical),
            "frozen_rows": len(evaluation),
            "prediction_rows": len(ledger),
            "frozen_identity_sha256": identity_digest(evaluation),
            "training_sources": list(historical_sources(historical)),
            "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "input_provenance": {
                "historical_manifest_sha256": sha256(
                    historical_root / "manifest.json"
                ),
                "historical_labels_sha256": historical_manifest["outputs_sha256"][
                    "action_labels.parquet"
                ],
                "current_manifest_sha256": sha256(current_root / "manifest.json"),
                "current_labels_sha256": current_manifest["outputs_sha256"][
                    "action_labels.parquet"
                ],
                "handoff_manifest_sha256": sha256(handoff_root / "manifest.json"),
                "handoff_sha256": handoff_manifest["outputs_sha256"]["handoff.parquet"],
                **calendar_provenance,
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(
            f"{sha256(temporary / 'manifest.json')}  manifest.json\n"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--historical-root", type=Path, default=HISTORICAL_ROOT)
    result.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    result.add_argument("--handoff-root", type=Path, default=HANDOFF_ROOT)
    result.add_argument("--calendar-root", type=Path, default=CALENDAR_ROOT)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            safe(
                run(
                    args.historical_root,
                    args.current_root,
                    args.handoff_root,
                    args.calendar_root,
                    args.output,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
