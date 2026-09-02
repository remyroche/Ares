#!/usr/bin/env python3
"""Prove 1h model cadence while allowing 1m bars only inside label replay."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data_perp/artifacts/regime_transition_hourly_cadence_audit_20260730_v2"
SOURCES = {
    "regime_feature_panel": ROOT
    / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet",
    "strict_regime_2026": ROOT
    / "data_perp/artifacts/strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3/regime_only_forward_2026_sidecar.parquet",
    "strict_transition_2026": ROOT
    / "data_perp/artifacts/strict_transition_v3_multihorizon_competing_risk_20260730_v2/forward_multihorizon_predictions.parquet",
    "full_2024_candidates": ROOT
    / "data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2/staged_candidates.parquet",
    "full_2024_paths": ROOT
    / "data_perp/artifacts/failure_2024_exact1m_paths_20260730_v1/paths.parquet",
    "full_2024_labels": ROOT
    / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
    "final_v3_historical_oof": ROOT
    / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3/historical_oof_scores.parquet",
    "final_v3_forward": ROOT
    / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3/frozen_2026_candidate_scores.parquet",
    "mapping_v2_forward": ROOT
    / "data_perp/artifacts/pre2026_mapping_resolution_ablations_20260730_v2/frozen_2026_selected_mapping_scores.parquet",
    "gam_mixture_forward": ROOT
    / "data_perp/artifacts/final_v3_gam_convex_mixture_ablation_20260730_v1/frozen_2026_mixture_top10_books.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamps(path: Path, column: str) -> pd.Series:
    values = pd.read_parquet(path, columns=[column])[column]
    return pd.to_datetime(values, utc=True, errors="raise")


def on_hour(values: pd.Series) -> bool:
    return bool(
        values.dt.minute.eq(0).all()
        and values.dt.second.eq(0).all()
        and values.dt.microsecond.eq(0).all()
    )


def run(output: Path = OUTPUT) -> dict[str, object]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    panel = pd.read_parquet(
        SOURCES["regime_feature_panel"],
        columns=["source_utc", "calendar_segment_id"],
    )
    panel["source_utc"] = pd.to_datetime(panel["source_utc"], utc=True)
    panel_diffs = panel.groupby("calendar_segment_id")["source_utc"].diff().dropna()
    panel_ok = (
        on_hour(panel["source_utc"])
        and not panel["source_utc"].duplicated().any()
        and panel_diffs.eq(pd.Timedelta(hours=1)).all()
    )

    regime = timestamps(SOURCES["strict_regime_2026"], "source_utc")
    regime_ok = on_hour(regime) and not regime.duplicated().any()

    transition = pd.read_parquet(
        SOURCES["strict_transition_2026"],
        columns=["source_utc", "horizon_hours"],
    )
    transition["source_utc"] = pd.to_datetime(transition["source_utc"], utc=True)
    horizon_sets = transition.groupby("source_utc")["horizon_hours"].agg(
        lambda values: tuple(sorted(values.astype(int)))
    )
    transition_ok = (
        on_hour(transition["source_utc"])
        and not transition.duplicated(["source_utc", "horizon_hours"]).any()
        and horizon_sets.map(lambda values: values == (1, 3, 6, 12)).all()
    )

    candidates = pd.read_parquet(
        SOURCES["full_2024_candidates"],
        columns=["signal_timestamp", "decision_timestamp", "candidate_id"],
    )
    signal = pd.to_datetime(candidates["signal_timestamp"], utc=True)
    decision = pd.to_datetime(candidates["decision_timestamp"], utc=True)
    candidate_ok = (
        on_hour(signal)
        and on_hour(decision)
        and decision.sub(signal).eq(pd.Timedelta(hours=1)).all()
        and not candidates["candidate_id"].duplicated().any()
    )

    label_times = pd.read_parquet(
        SOURCES["full_2024_labels"],
        columns=["__ts__", "__decision_ts__", "__label_end_ts__"],
    )
    label_signal = pd.to_datetime(label_times["__ts__"], utc=True)
    label_decision = pd.to_datetime(label_times["__decision_ts__"], utc=True)
    label_end = pd.to_datetime(label_times["__label_end_ts__"], utc=True)
    label_ok = (
        on_hour(label_signal)
        and on_hour(label_decision)
        and on_hour(label_end)
        and label_decision.sub(label_signal).eq(pd.Timedelta(hours=1)).all()
        and label_end.sub(label_decision).eq(pd.Timedelta(hours=12)).all()
    )

    path_rows = pq.ParquetFile(SOURCES["full_2024_paths"]).metadata.num_rows
    path_schema = pq.read_schema(SOURCES["full_2024_paths"])
    path_ok = (
        path_rows == len(candidates)
        and "execution_future_path" in path_schema.names
        and path_rows == len(label_times)
    )

    current_rows: dict[str, int] = {}
    current_unique_candidates: dict[str, int] = {}
    current_checks: dict[str, bool] = {}
    for role in (
        "final_v3_historical_oof",
        "final_v3_forward",
        "mapping_v2_forward",
        "gam_mixture_forward",
    ):
        frame = pd.read_parquet(
            SOURCES[role],
            columns=["candidate_id", "__ts__", "execution_label_end_utc"],
        )
        decision_ts = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        label_end_ts = pd.to_datetime(
            frame["execution_label_end_utc"], utc=True, errors="raise"
        )
        current_rows[role] = len(frame)
        current_unique_candidates[role] = int(frame["candidate_id"].nunique())
        current_checks[f"{role}_hourly_and_future_resolved"] = bool(
            on_hour(decision_ts)
            and on_hour(label_end_ts)
            and label_end_ts.gt(decision_ts).all()
        )

    checks = {
        "regime_feature_panel_hourly": bool(panel_ok),
        "strict_regime_assessment_hourly": bool(regime_ok),
        "strict_transition_assessment_hourly": bool(transition_ok),
        "candidate_signal_and_decision_hourly": bool(candidate_ok),
        "one_minute_path_is_nested_one_row_per_hourly_candidate": bool(path_ok),
        "label_horizon_signal_plus_1h_then_12h_path": bool(label_ok),
        **current_checks,
    }
    if not all(checks.values()):
        raise ValueError(f"cadence audit failed: {checks}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        manifest = {
            "schema": "regime_transition_hourly_model_cadence_audit_v2",
            "status": "PASSED",
            "model_sample_cadence": "1h",
            "assessment_sample_cadence": "1h",
            "exact_replay_bar_cadence": "1m",
            "cadence_separation": (
                "one-minute bars are nested future-path observations used only "
                "to construct labels; they are never independent model rows"
            ),
            "checks": checks,
            "counts": {
                "regime_panel_hours": len(panel),
                "strict_regime_2026_hours": len(regime),
                "strict_transition_2026_hours": transition["source_utc"].nunique(),
                "full_2024_hourly_candidates": len(candidates),
                "full_2024_nested_path_rows": path_rows,
                "full_2024_label_rows": len(label_times),
                "current_artifact_rows": current_rows,
                "current_artifact_unique_candidates": current_unique_candidates,
            },
            "sources": {
                role: {"path": str(path), "sha256": sha256(path)}
                for role, path in SOURCES.items()
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(
            f"{sha256(manifest_path)}  manifest.json\n"
        )
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
