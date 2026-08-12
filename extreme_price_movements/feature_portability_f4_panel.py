"""Build the compact, project-native candidate panel consumed by F4 MDA.

This is a data materialisation boundary, not an F4 evaluation.  It reuses the
TP6/SL4/H12 source loader and the exact side-local frozen F0 contracts, then
adds the F3 rank90/rank180, robust-z90/180, and delta4/delta24 fields in
bounded causal batches.
It stops strictly before the untouched November 2024 holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_portability import (
    causal_rolling_portability_transform_batches,
    estimate_causal_rolling_transform_memory,
)
from .tp6_portability_data import (
    SIDES,
    TP6PortabilityContract,
    all_frozen_base_features,
    load_tp6_population,
)


SCHEMA = "stage_a_f4_tp6_candidate_panel_v1"
PANEL_START = pd.Timestamp("2023-04-01", tz="UTC")
FINAL_OOS_START = pd.Timestamp("2024-11-01", tz="UTC")
F3_BATCH_SIZE = 1
F3_MAX_GENERATED_BYTES = 1024 * 1024 * 1024
F3_MAX_BATCH_WORKING_BYTES = 384 * 1024 * 1024
F3_DISALLOWED_DISPOSITIONS = frozenset({"ERA_SHORTCUT", "REJECTED_LINEAGE"})
R3_WEIGHT_COLUMNS = (
    "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50",
)
FROZEN_R3_BASE_PARAMS: Mapping[str, Any] = {
    "n_estimators": 140,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 350,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 8.0,
    "n_jobs": 1,
    "verbosity": -1,
    "objective": "multiclass",
    "num_class": 3,
}
F4_TRANSPORTS: tuple[dict[str, str], ...] = (
    {
        "name": "transport_a_2023q4_to_2024h1",
        "train_start": "2023-04-01T00:00:00+00:00",
        "evaluation_start": "2024-01-01T00:00:00+00:00",
        "evaluation_end": "2024-07-01T00:00:00+00:00",
    },
    {
        "name": "transport_b_2024h1_to_2024h2_to_date",
        "train_start": "2023-04-01T00:00:00+00:00",
        "evaluation_start": "2024-07-01T00:00:00+00:00",
        "evaluation_end": "2024-11-01T00:00:00+00:00",
    },
)


class F4CandidatePanelError(ValueError):
    """Raised when the panel cannot prove the TP6/F0/F3 contract."""


@dataclass(frozen=True)
class F4CandidatePanel:
    """The panel and side contracts needed by the standalone F4 MDA CLI."""

    panel: pd.DataFrame
    representation_contracts: Mapping[str, Mapping[str, Sequence[str]]]
    transports: Sequence[Mapping[str, str]]
    r3_cost_contract: Mapping[str, Any]
    frozen_r3_model_contract: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _normalise_dispositions(dispositions: pd.DataFrame) -> pd.Series:
    needed = {"feature", "disposition"}
    missing = sorted(needed.difference(dispositions.columns))
    if missing:
        raise F4CandidatePanelError(f"portability dispositions lack columns: {missing}")
    if dispositions["feature"].isna().any() or dispositions["feature"].duplicated().any():
        raise F4CandidatePanelError("portability dispositions require unique non-null features")
    return dispositions.set_index(dispositions["feature"].astype(str))["disposition"].astype(str)


def _f3_sources(
    frozen_features: Mapping[str, Sequence[str]], dispositions: pd.Series
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for side in SIDES:
        selected = [
            str(field) for field in frozen_features[side]
            if str(dispositions.get(str(field), "REJECTED_LINEAGE")) not in F3_DISALLOWED_DISPOSITIONS
        ]
        if not selected:
            raise F4CandidatePanelError(f"{side} has no lineage-safe F3 source fields")
        sources[side] = selected
    return sources


def _verify_source_frame(
    frame: pd.DataFrame,
    *,
    side: str,
    expected_features: Sequence[str],
    cost_bps: float,
) -> pd.DataFrame:
    required = [
        "candidate_id", "decision_ts", "label_available_ts", "side_name", "asset", "r3_class",
        "gross_bps", "net_bps", *R3_WEIGHT_COLUMNS, *expected_features,
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise F4CandidatePanelError(f"TP6 source omits F4 panel fields: {missing}")
    work = frame.copy()
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], utc=True, errors="coerce")
    work["label_available_ts"] = pd.to_datetime(work["label_available_ts"], utc=True, errors="coerce")
    if work[["decision_ts", "label_available_ts"]].isna().any().any() or work.empty:
        raise F4CandidatePanelError("TP6 source has empty or invalid UTC candidate/label timestamps")
    if not work["side_name"].astype(str).eq(side).all():
        raise F4CandidatePanelError(f"TP6 side loader returned rows outside {side}")
    if work["candidate_id"].isna().any() or work["candidate_id"].duplicated().any():
        raise F4CandidatePanelError("candidate IDs must remain unique within each TP6 side panel")
    if work["decision_ts"].lt(PANEL_START).any() or work["decision_ts"].ge(FINAL_OOS_START).any():
        raise F4CandidatePanelError("candidate panel must be [2023-04-01, 2024-11-01), never final November OOS")
    classes = pd.to_numeric(work["r3_class"], errors="coerce")
    if classes.isna().any() or set(classes.astype(int)).difference({0, 1, 2}):
        raise F4CandidatePanelError("TP6 R3 classes must be adverse=0, weak=1, clear=2")
    work["r3_class"] = classes.astype(np.int8)
    gross, net = (pd.to_numeric(work[column], errors="coerce") for column in ("gross_bps", "net_bps"))
    if not np.isfinite(gross.to_numpy(float)).all() or not np.isfinite(net.to_numpy(float)).all():
        raise F4CandidatePanelError("TP6 gross/net bps must be finite")
    if not np.allclose(gross.to_numpy(float) - net.to_numpy(float), float(cost_bps), atol=0.02, rtol=0.0):
        raise F4CandidatePanelError("TP6 gross/net rows do not charge the frozen cost exactly once")
    work["gross_bps"], work["net_bps"] = gross.astype(float), net.astype(float)
    return work


def _add_f3_transforms(frame: pd.DataFrame, *, sources: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    estimate = estimate_causal_rolling_transform_memory(
        rows=len(frame), source_features=len(sources), rank_windows=(90, 180), robust_z_windows=(90, 180),
        change_periods=(4, 24), include_relative_change=False, feature_batch_size=F3_BATCH_SIZE,
    )
    if estimate.materialized_output_bytes > F3_MAX_GENERATED_BYTES:
        raise F4CandidatePanelError(
            "F3 generated panel exceeds the declared memory contract "
            f"({estimate.materialized_output_bytes:,} > {F3_MAX_GENERATED_BYTES:,} bytes)"
        )
    generated: list[str] = []
    # Each source batch is narrow by design; collect those blocks then join
    # once.  Repeated `frame[field] = ...` inserts fragment a large panel and
    # turn an otherwise bounded vectorised transform into a quadratic copy
    # pattern.
    transform_blocks: list[pd.DataFrame] = []
    for batch in causal_rolling_portability_transform_batches(
        frame, feature_names=list(sources), timestamp_column="decision_ts", group_columns=("asset",),
        rank_windows=(90, 180), robust_z_windows=(90, 180), change_periods=(4, 24),
        include_relative_change=False, minimum_periods=30, feature_batch_size=F3_BATCH_SIZE,
        max_batch_working_bytes=F3_MAX_BATCH_WORKING_BYTES,
    ):
        transform_blocks.append(batch)
        generated.extend(map(str, batch.columns))
    expected_count = 6 * len(sources)
    if len(generated) != expected_count or len(generated) != len(set(generated)):
        raise F4CandidatePanelError(
            "F3 did not materialise exactly rank90/rank180/robust-z90/robust-z180/delta4/delta24 per source"
        )
    return pd.concat([frame, *transform_blocks], axis=1, copy=False), generated


def materialize_tp6_f4_candidate_panel(
    *,
    contract: TP6PortabilityContract,
    portability_dispositions: pd.DataFrame,
    load_population: Callable[..., pd.DataFrame] = load_tp6_population,
    frozen_features_provider: Callable[[TP6PortabilityContract], Mapping[str, Sequence[str]]] = all_frozen_base_features,
) -> F4CandidatePanel:
    """Build the F0/F3 candidate panel without evaluating any transport.

    The loader is deliberately injectable for small contract tests.  Production
    calls use ``load_tp6_population`` once per side, with only that side's F0
    fields loaded, then discard the side's raw source frame after F3 batches
    have been attached and pruned.
    """
    dispositions = _normalise_dispositions(portability_dispositions)
    frozen = {side: list(map(str, values)) for side, values in frozen_features_provider(contract).items()}
    if set(frozen) != set(SIDES) or any(not frozen[side] for side in SIDES):
        raise F4CandidatePanelError("actual F0 frozen feature provider must give non-empty long/short contracts")
    f3_sources = _f3_sources(frozen, dispositions)
    side_frames: list[pd.DataFrame] = []
    f3_contract: dict[str, list[str]] = {}
    for side in SIDES:
        source = load_population(
            contract=contract, columns=frozen[side], start=PANEL_START, end=FINAL_OOS_START,
            sides=(side,), valid_labels_only=True,
        )
        source = _verify_source_frame(source, side=side, expected_features=frozen[side], cost_bps=float(contract.cost_bps))
        source, generated = _add_f3_transforms(source, sources=f3_sources[side])
        f3_contract[side] = [*f3_sources[side], *generated]
        retain = [
            "candidate_id", "decision_ts", "label_available_ts", "side_name", "gross_bps", "net_bps", "r3_class",
            *R3_WEIGHT_COLUMNS, *frozen[side], *generated,
        ]
        side_frames.append(source.loc[:, list(dict.fromkeys(retain))].copy())
    panel = pd.concat(side_frames, ignore_index=True).sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    if panel["candidate_id"].duplicated().any() or panel["decision_ts"].ge(FINAL_OOS_START).any():
        raise F4CandidatePanelError("combined F4 panel violates candidate identity or final-OOS exclusion")
    representation_contracts: dict[str, dict[str, list[str]]] = {
        "F0_current_frozen": {side: list(frozen[side]) for side in SIDES},
        "F3_plus_relative": {side: list(f3_contract[side]) for side in SIDES},
    }
    r3_cost_contract: dict[str, Any] = {
        "class_column": "r3_class",
        "gross_bps_column": "gross_bps",
        "net_bps_column": "net_bps",
        "expected_cost_bps": float(contract.cost_bps),
        "robust_clear_columns": list(R3_WEIGHT_COLUMNS),
    }
    frozen_r3_model_contract: dict[str, Any] = {
        "model_id": "frozen_tp6_sl4_r3_side_local_lgbm_base_v1",
        "params": dict(FROZEN_R3_BASE_PARAMS),
        "random_seed": 17,
        "model_hpo_performed": False,
        "class_order": ["adverse", "weak", "clear"],
    }
    manifest = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_DEVELOPMENT_PANEL_ONLY",
        "source_window": {"start": PANEL_START.isoformat(), "end_exclusive": FINAL_OOS_START.isoformat()},
        "final_november_oos_consumed": False,
        "candidate_rows": int(len(panel)),
        "candidate_ids_sha256": _sha256_json(panel["candidate_id"].tolist()),
        "f0_feature_counts": {side: len(frozen[side]) for side in SIDES},
        "f3_source_counts": {side: len(f3_sources[side]) for side in SIDES},
        "f3_generated_counts": {side: len(f3_contract[side]) - len(f3_sources[side]) for side in SIDES},
        "f3_transform": {
            "families": [
                "causal_rank_w90", "causal_rank_w180", "causal_robust_z_w90", "causal_robust_z_w180",
                "causal_delta_p4", "causal_delta_p24",
            ],
            "grouping": "asset_within_side", "minimum_periods": 30,
            "feature_batch_size": F3_BATCH_SIZE, "max_generated_bytes": F3_MAX_GENERATED_BYTES,
        },
        "transports": list(F4_TRANSPORTS),
        "r3_cost_contract": r3_cost_contract,
        "frozen_r3_model_contract": frozen_r3_model_contract,
    }
    return F4CandidatePanel(
        panel=panel, representation_contracts=representation_contracts, transports=F4_TRANSPORTS,
        r3_cost_contract=r3_cost_contract, frozen_r3_model_contract=frozen_r3_model_contract, manifest=manifest,
    )


def write_tp6_f4_candidate_panel(result: F4CandidatePanel, output_dir: Path) -> dict[str, Path]:
    """Persist the exact inputs expected by the standalone F4 evidence CLI."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite F4 candidate panel directory: {output_dir}")
    output_dir.mkdir(parents=True)
    paths = {
        "panel": output_dir / "f4_candidate_panel.parquet",
        "representation_contracts": output_dir / "f4_representation_contracts.json",
        "transports": output_dir / "f4_transports.json",
        "r3_cost_contract": output_dir / "f4_r3_cost_contract.json",
        "frozen_r3_model_contract": output_dir / "f4_frozen_r3_model_contract.json",
        "manifest": output_dir / "f4_candidate_panel_manifest.json",
    }
    result.panel.to_parquet(paths["panel"], index=False, compression="zstd")
    for name in ("representation_contracts", "transports", "r3_cost_contract", "frozen_r3_model_contract", "manifest"):
        paths[name].write_text(json.dumps(getattr(result, name) if name != "manifest" else result.manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return paths


__all__ = [
    "F3_BATCH_SIZE", "F4CandidatePanel", "F4CandidatePanelError", "F4_TRANSPORTS",
    "FROZEN_R3_BASE_PARAMS", "PANEL_START", "FINAL_OOS_START",
    "materialize_tp6_f4_candidate_panel", "write_tp6_f4_candidate_panel",
]
