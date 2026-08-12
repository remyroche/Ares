#!/usr/bin/env python3
"""Run the sequential Stage-I S/O/R3 base-target ablation.

Round 1 screens all 60 materialised target arms as label/oracle contracts.
Round 2 fits only preregistered survivors on one large chronological,
label-availability-purged development holdout.  Round 3 compares frozen R3,
the best S arm, and the best O arm on that identical holdout and the three
declared training-weight contracts.  This fast funnel is development selection,
not final OOS evidence; target-specific MDA and 2024--2026 validation follow.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_base_target_ablation import (  # noqa: E402
    BaseTargetAblationError,
    DevelopmentModelCache,
    Round1Gates,
    TargetArm,
    file_sha256,
    pooled_global_tail_metrics,
    require_selected_feature_contract,
    robust_top10_lift_score,
    round1_screen,
    run_development_holdout_arm,
    target_column_for_arm,
    target_arm_grid,
    verify_completed_manifest,
)
from extreme_price_movements.stage_i_r3_contract import (  # noqa: E402
    require_r3_label_economics_contract,
    selector_validity_mask,
)
from extreme_price_movements.stage_i_target_promotion import (  # noqa: E402
    StageITargetPromotionError,
    decide_round3_promotion,
)


SCHEMA = "stage_i_base_target_ablation_v2"
IDENTITY = ["candidate_id", "__ts__", "__symbol__"]


def _model_cell_source_fingerprint() -> dict[str, Any]:
    """Bind cells to scientific code while excluding report presentation."""

    target_module = ROOT / "extreme_price_movements" / "stage_i_base_target_ablation.py"
    scientific_functions = (
        "_model_frame",
        "_prune_model_frame_to_selected_contract",
        "_prediction_metrics",
        "_population_audit",
        "_run_model_cell",
    )
    function_hashes = {
        name: sha256(inspect.getsource(globals()[name]).encode("utf-8")).hexdigest()
        for name in scientific_functions
    }
    payload = {
        "schema": "stage_i_target_model_source_fingerprint_v2",
        "scientific_function_sha256": function_hashes,
        "target_module_sha256": file_sha256(target_module),
        "presentation_code_excluded": True,
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    return payload


def _canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaseTargetAblationError(f"JSON object required: {path}")
    return value


def _require_labels(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    labels_path = root / "target_repair_labels.parquet"
    if not manifest_path.is_file() or not labels_path.is_file():
        raise BaseTargetAblationError("target-grid artifact is incomplete")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "stage_i_base_target_label_grid_v2"
        or manifest.get("status") != "complete"
        or int(manifest.get("geometries", -1)) != 15
        or int(manifest.get("target_arms", -1)) != 60
        or manifest.get("artifact_sha256", {}).get("target_repair_labels.parquet") != file_sha256(labels_path)
    ):
        raise BaseTargetAblationError("target-grid manifest/hash drift")
    materializer_fingerprint = manifest.get("materializer_source_fingerprint")
    if (
        not isinstance(materializer_fingerprint, dict)
        or not isinstance(manifest.get("source_materializer_code_contract_sha256"), str)
        or not isinstance(manifest.get("minute_source_inventory_sha256"), str)
    ):
        raise BaseTargetAblationError("target-grid lacks materializer/minute-source lineage")
    current_materializer_paths = (
        ROOT / "scripts" / "materialize_stage_i_base_target_grid.py",
        ROOT / "extreme_price_movements" / "stage_i_base_target_ablation.py",
    )
    current_materializer_fingerprint = {
        "schema": "stage_i_target_grid_materializer_source_v1",
        "files": {
            str(path.resolve()): file_sha256(path) for path in current_materializer_paths
        },
    }
    current_materializer_fingerprint["contract_sha256"] = _canonical_sha(
        current_materializer_fingerprint
    )
    if materializer_fingerprint != current_materializer_fingerprint:
        raise BaseTargetAblationError(
            "INVALID_SENTINEL_SEMANTICS: target-grid materializer source fingerprint drift"
        )
    regime = manifest.get("causal_regime_contract")
    if (
        not isinstance(regime, dict)
        or regime.get("causal_at_decision_time") is not True
        or regime.get("diagnostic_noncausal") is not False
    ):
        raise BaseTargetAblationError("target-grid regime is not proven causal at decision time")
    labels = pd.read_parquet(labels_path)
    valid = labels.target_valid.eq(True).to_numpy(dtype=bool)
    event = pd.to_numeric(labels.event, errors="coerce").to_numpy(dtype=np.float64)
    minute = pd.to_numeric(labels.event_minute, errors="coerce").to_numpy(dtype=np.float64)
    valid_event = event[valid]
    valid_minute = minute[valid]
    if (
        not np.isin(valid_event, (0.0, 1.0, 2.0)).all()
        or not np.array_equal(valid_event == 1.0, valid_minute == -1.0)
        or not np.any(valid_event == 1.0)
        or np.any((valid_minute[valid_event != 1.0] < 0.0) | (valid_minute[valid_event != 1.0] >= 720.0))
    ):
        raise BaseTargetAblationError(
            "INVALID_SENTINEL_SEMANTICS: event/minute invariant or timeout support failed"
        )
    gross = pd.to_numeric(labels.gross_bps, errors="coerce").to_numpy(dtype=np.float64)
    net = pd.to_numeric(labels.net_bps, errors="coerce").to_numpy(dtype=np.float64)
    # Both columns are persisted independently as float32.  Around the largest
    # barrier values their separate rounding can differ by two float32 ULPs
    # (observed maximum 0.000244 bps), even though the pre-serialization
    # materializer proves exact ``net = gross - 100`` in float64.  Keep the
    # artifact gate far below any economically meaningful unit while allowing
    # that deterministic storage error.
    if not np.allclose(net[valid], gross[valid] - 100.0, rtol=0.0, atol=5e-4):
        raise BaseTargetAblationError("target-grid net economics are not gross minus 100 bps")
    for (geometry, side), part in labels.loc[valid].groupby(
        ["geometry", "side_name"], observed=True, sort=True
    ):
        scalar = pd.to_numeric(part.S_target, errors="coerce").to_numpy(dtype=np.float64)
        if not np.any((scalar > 0.0) & (scalar < 1.0)):
            raise BaseTargetAblationError(
                f"{geometry}/{side}: scalar_S lacks side-local soft timeout support"
            )
        for alpha in ("0p25", "0p33", "0p5"):
            ordinal = pd.to_numeric(
                part[f"O_a{alpha}_target"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            unique = np.unique(ordinal[np.isfinite(ordinal)])
            if len(unique) <= 2 or not np.isin(unique, (1.0, 2.0, 3.0)).any():
                raise BaseTargetAblationError(
                    f"{geometry}/{side}: ordinal_O lacks side-local interior class support"
                )
    return labels, manifest


def _request(
    *, args: argparse.Namespace, selected: dict[str, Any], label_manifest: dict[str, Any]
) -> dict[str, Any]:
    new_boosters = 5 * int(args.round2_per_family) + 18
    legacy_boosters = 30 * int(args.round2_per_family) + 180
    payload = {
        "schema": "stage_i_base_target_ablation_request_v2",
        "round": int(args.round),
        "selected_feature_contract": selected,
        "target_label_manifest_sha256": file_sha256(args.label_grid_dir / "manifest.json"),
        "target_label_artifact_sha256": label_manifest["artifact_sha256"]["target_repair_labels.parquet"],
        "round1_gates": {
            "min_upper_support_rows": args.min_upper_support_rows,
            "max_timeout_prevalence": args.max_timeout_prevalence,
            "min_worst_regime_upper_rate": args.min_worst_regime_upper_rate,
            "min_oracle_top10_net_bps": args.min_oracle_top10_net_bps,
            "provenance": "explicit preregistered CLI values; never fitted post-results",
        },
        "development_split": {
            "schema": "single_large_chronological_holdout_v1",
            "evaluation_fraction_of_unique_timestamps": args.evaluation_fraction,
            "seed": args.development_seed,
            "purge": "train label_available_ts < held-out evaluation start",
            "status": "development_selection_not_final_oos",
        },
        "round2": {"single_holdout": True, "seed": args.development_seed, "per_family": args.round2_per_family},
        "round3": {"single_holdout": True, "seed": args.development_seed, "weights": ["uniform", "contract_certainty", "hybrid"]},
        "min_train_rows": int(args.min_train_rows),
        "causal_regime_column": args.regime_column,
        "selection": {
            "primary": "pooled_global_top10_net_bps_after_common_bps_mapping",
            "secondary": ["top1", "top5", "median_era", "worst_era", "worst_side", "regime_stability", "mapped_ev_monotonicity", "latest_era"],
            "robust_score": "0.5*pooled_top10_lift + 0.5*median_era_top10_lift - 0.5*MAD(era_lifts) - max(0,-worst_era_lift)",
        },
        "ranking": "one pooled global rank after prior-resolved side-local common-bps mapping",
        "inference_leakage": "path/target/certainty/economic fields excluded from X; certainty is training weight only",
        "runtime_optimization": {
            "schema": "stage_i_base_target_runtime_optimization_v1",
            "legacy_default_nominal_booster_fits": legacy_boosters,
            "single_holdout_nominal_booster_fits": new_boosters,
            "nominal_booster_fit_reduction_factor": legacy_boosters / new_boosters,
            "path_normalisation_passes": {"legacy": 15, "current": 1},
            "distinct_first_touch_passes": {"legacy": 30, "current": 8},
            "selector_feature_matrix_reads": 1,
            "ordinal_heads_batched": 4,
            "shared_lightgbm_bin_reference_per_side": True,
            "parallel_target_workers": int(getattr(args, "parallel_workers", 3)),
            "parallel_memory_budget_fraction": float(getattr(args, "parallel_memory_budget_fraction", .65)),
            "parallel_contract": "fresh processes over hash-verified read-only Arrow IPC memory maps; each booster n_jobs=1; sequential shared-bin fallback",
        },
    }
    payload["input_contract_sha256"] = _canonical_sha({
        "selected_feature_contract_sha256": selected["contract_sha256"],
        "target_label_manifest_sha256": payload["target_label_manifest_sha256"],
        "target_label_artifact_sha256": payload["target_label_artifact_sha256"],
        "round1_gates": payload["round1_gates"],
        "causal_regime_column": payload["causal_regime_column"],
        "min_train_rows": payload["min_train_rows"],
        "development_split": payload["development_split"],
    })
    payload["request_sha256"] = _canonical_sha(payload)
    return payload


def _round1_slices(labels: pd.DataFrame, arms: Iterable[TargetArm], regime_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        local = labels.loc[labels.geometry.eq(arm.geometry.key) & labels.target_valid.astype(bool)].copy()
        local["score"] = pd.to_numeric(local[target_column_for_arm(arm)], errors="coerce")
        for dimensions in (("side_name",), (regime_column,), ("side_name", regime_column)):
            for key, part in local.groupby(list(dimensions), observed=True, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                selected = part.sort_values(
                    ["score", "decision_ts", "__symbol__", "side_name", "candidate_id"],
                    ascending=[False, True, True, True, True], kind="mergesort",
                ).head(max(1, int(np.ceil(.10 * len(part)))))
                row = {
                    "arm": arm.name, "slice": "x".join(dimensions), "rows": int(len(part)),
                    "top10_rows": int(len(selected)), "top10_gross_bps": float(selected.gross_bps.mean()),
                    "top10_net_bps": float(selected.net_bps.mean()),
                }
                row.update(dict(zip(dimensions, map(str, key), strict=True)))
                rows.append(row)
    return pd.DataFrame(rows)


def _arm_from_name(name: str) -> TargetArm:
    matches = [arm for arm in target_arm_grid() if arm.name == name]
    if len(matches) != 1:
        raise BaseTargetAblationError(f"unknown target arm {name!r}")
    return matches[0]


def _round1_winners(metrics: pd.DataFrame, gates: pd.DataFrame, *, per_family: int) -> list[TargetArm]:
    allowed = set(gates.loc[gates.promotion_eligible.astype(bool), "arm"])
    top10 = metrics.loc[metrics.top_fraction.eq(.10) & metrics.arm.isin(allowed)].copy()
    winners: list[TargetArm] = []
    for family in ("scalar_S", "ordinal_O"):
        selected = top10.loc[top10.family.eq(family)].sort_values(
            ["net_bps_per_trade", "gross_bps_per_trade", "arm"], ascending=[False, False, True], kind="mergesort"
        ).head(per_family)
        winners.extend(_arm_from_name(name) for name in selected.arm)
    if not winners:
        raise BaseTargetAblationError("Round 1 gates rejected every promotable arm")
    return winners


def _model_frame(
    labels: pd.DataFrame, selector_dir: Path, arm: TargetArm | None, *, regime_column: str,
    selector_ledger: pd.DataFrame | None = None,
    selector_feature_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str, str, pd.DataFrame]:
    ledger = selector_ledger if selector_ledger is not None else pd.read_parquet(selector_dir / "selector_ledger.parquet")
    if selector_feature_frame is None:
        features = pd.read_parquet(selector_dir / "selector_features.parquet")
        if not ledger.loc[:, IDENTITY].reset_index(drop=True).equals(features.loc[:, IDENTITY].reset_index(drop=True)):
            raise BaseTargetAblationError("selector ledger/features identity drift")
        feature_only = features.drop(columns=IDENTITY)
        feature_only.columns = feature_only.columns.astype(str)
        feature_frame = pd.concat(
            [features.loc[:, IDENTITY].reset_index(drop=True), feature_only.reset_index(drop=True)], axis=1,
        )
    else:
        feature_frame = selector_feature_frame
    if arm is None:
        selector_manifest = _json(selector_dir / "manifest.json")
        integrity = selector_manifest.get("artifact_integrity")
        if not isinstance(integrity, dict):
            raise BaseTargetAblationError("R3 control selector lacks artifact-integrity contract")
        require_r3_label_economics_contract(
            ledger, str(integrity.get("r3_label_economics_contract_sha256", "")),
        )
        selector_valid = selector_validity_mask(ledger)
        # Causal regime and certainty remain context/weight metadata; the R3
        # control's target and economics both come from its frozen TP6/SL4
        # contract, never from the clipped S/O geometry.
        context = labels.loc[
            labels.geometry.eq("sl4_tp6"),
            IDENTITY + [regime_column, "contract_certainty", "target_valid"],
        ].copy()
        if len(context) != len(ledger):
            raise BaseTargetAblationError(
                "R3 control lacks one exact-H12 path-validity row per selector candidate"
            )
        population = ledger.loc[:, IDENTITY + ["side_name", "decision_ts"]].copy()
        population["r3_selector_label_valid"] = selector_valid
        population = population.merge(
            context.loc[:, IDENTITY + ["target_valid"]],
            on=IDENTITY, how="left", validate="one_to_one",
        )
        if population.target_valid.isna().any():
            raise BaseTargetAblationError("R3/path validity identity drift")
        population["exact_h12_path_valid"] = population.target_valid.astype(bool)
        population["label_valid"] = (
            population.r3_selector_label_valid.astype(bool)
            & population.exact_h12_path_valid.astype(bool)
        )
        source = ledger.loc[
            selector_valid,
            IDENTITY + [
                "side_name", "decision_ts", "label_available_ts", "r3_class",
                "exact_net_bps",
            ],
        ].copy().rename(columns={"exact_net_bps": "net_bps"})
        local = source.merge(context, on=IDENTITY, how="inner", validate="one_to_one")
        local = local.loc[local.target_valid.astype(bool)].drop(columns="target_valid")
        if len(local) != int(population.label_valid.sum()):
            raise BaseTargetAblationError("R3/common exact-H12 validity intersection drift")
        target_column, family = "r3_class", "R3_control"
    else:
        geometry_rows = labels.loc[labels.geometry.eq(arm.geometry.key)].copy()
        population = geometry_rows.loc[:, IDENTITY + ["side_name", "decision_ts"]].copy()
        population["label_valid"] = geometry_rows.target_valid.astype(bool).to_numpy()
        local = geometry_rows.loc[geometry_rows.target_valid.astype(bool)].copy()
        target_column, family = target_column_for_arm(arm), arm.family
    local = local.merge(feature_frame, on=IDENTITY, how="inner", validate="one_to_one")
    if local.empty:
        raise BaseTargetAblationError("model frame has no exact selector/label identity intersection")
    return local, target_column, family, population


def _load_selector_feature_frame(selector_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalise the exact selector matrix once for the full funnel."""

    ledger = pd.read_parquet(selector_dir / "selector_ledger.parquet")
    features = pd.read_parquet(selector_dir / "selector_features.parquet")
    if not ledger.loc[:, IDENTITY].reset_index(drop=True).equals(features.loc[:, IDENTITY].reset_index(drop=True)):
        raise BaseTargetAblationError("selector ledger/features identity drift")
    feature_only = features.drop(columns=IDENTITY)
    feature_only.columns = feature_only.columns.astype(str)
    frame = pd.concat(
        [features.loc[:, IDENTITY].reset_index(drop=True), feature_only.reset_index(drop=True)], axis=1,
    )
    return ledger, frame


def _prune_model_frame_to_selected_contract(
    frame: pd.DataFrame, *, target_column: str, selected_contract: dict[str, Any],
    regime_column: str,
) -> pd.DataFrame:
    """Drop unselected selector columns before IPC/checkpoint orchestration."""

    selected = {
        str(feature)
        for side in ("long", "short")
        for feature in selected_contract["sides"][side]["selected_features"]
    }
    required = {
        *IDENTITY, "side_name", "decision_ts", "label_available_ts", "net_bps",
        target_column, regime_column, *selected,
    }
    # Certainty is a training-only weight input. Uniform-only Round 2 does not
    # need it, but retaining one float column lets the exact frame be reused by
    # every Round-3 weight arm without another materialisation.
    if "contract_certainty" in frame:
        required.add("contract_certainty")
    if missing := sorted(required.difference(frame.columns)):
        raise BaseTargetAblationError(f"parallel model frame lacks selected inputs: {missing[:8]}")
    ordered = [column for column in frame.columns if column in required]
    return frame.loc[:, ordered].copy()


def _prediction_metrics(
    prediction: pd.DataFrame, *, regime_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Average deterministic development predictions before one pooled ranking.
    keys = IDENTITY + ["side_name", "decision_ts", "label_available_ts", "net_bps", regime_column, "fold_id", "arm", "family", "weight_mode"]
    average = prediction.groupby(keys, observed=True, as_index=False).agg(
        raw_score=("raw_score", "mean"), expected_net_bps=("expected_net_bps", "mean")
    )
    pooled = pooled_global_tail_metrics(average)
    population_mean = float(average.net_bps.mean())
    pooled["net_lift_bps"] = pooled.net_bps_per_trade - population_mean
    rows: list[dict[str, Any]] = []
    for dimension in ("side_name", regime_column):
        for value, part in average.groupby(dimension, observed=True, sort=True):
            local = pooled_global_tail_metrics(part)
            local["slice"] = dimension
            local["slice_value"] = str(value)
            rows.extend(local.to_dict(orient="records"))
    average["era"] = pd.to_datetime(average.decision_ts, utc=True).dt.strftime("%Y-%m")
    era_lifts: list[float] = []
    for era, part in average.groupby("era", observed=True, sort=True):
        local = pooled_global_tail_metrics(part)
        local["slice"] = "era"
        local["slice_value"] = str(era)
        local["net_lift_bps"] = local.net_bps_per_trade - float(part.net_bps.mean())
        top10 = local.loc[local.top_fraction.eq(.10)]
        if len(top10):
            era_lifts.append(float(top10.net_lift_bps.iloc[0]))
        rows.extend(local.to_dict(orient="records"))
    average["week"] = pd.to_datetime(average.decision_ts, utc=True).dt.strftime("%G-W%V")
    for week, part in average.groupby("week", observed=True, sort=True):
        local = pooled_global_tail_metrics(part)
        local["slice"] = "week"
        local["slice_value"] = str(week)
        local["net_lift_bps"] = local.net_bps_per_trade - float(part.net_bps.mean())
        rows.extend(local.to_dict(orient="records"))
    top10_lift = float(pooled.loc[pooled.top_fraction.eq(.10), "net_lift_bps"].iloc[0])
    pooled["robust_top10_lift_score"] = robust_top10_lift_score(top10_lift, era_lifts)
    # Common-bps monotonicity diagnostic, not a local reranking rule.
    finite = average.loc[np.isfinite(average.expected_net_bps)].copy()
    if len(finite) >= 10:
        finite["mapped_bin"] = pd.qcut(finite.expected_net_bps.rank(method="first"), 10, labels=False)
        curve = finite.groupby("mapped_bin", observed=True).agg(
            rows=("candidate_id", "size"), predicted_net_bps=("expected_net_bps", "mean"), realised_net_bps=("net_bps", "mean")
        ).reset_index()
        realised = curve.realised_net_bps.to_numpy(float)
        monotone_violations = int((np.diff(realised) < 0.0).sum())
        pooled["mapped_ev_monotonicity_violations"] = monotone_violations
    else:
        curve = pd.DataFrame(columns=["mapped_bin", "rows", "predicted_net_bps", "realised_net_bps"])
        pooled["mapped_ev_monotonicity_violations"] = np.nan
    slices = pd.DataFrame(rows)
    return pd.concat([pooled.assign(slice="pooled_global", slice_value="all"), slices], ignore_index=True), curve


def _population_audit(population: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    scored = prediction.drop_duplicates(IDENTITY)
    mapped = scored.loc[np.isfinite(scored.expected_net_bps)]
    rows: list[dict[str, Any]] = []
    pop = population.copy()
    pop["month"] = pd.to_datetime(pop.decision_ts, utc=True).dt.strftime("%Y-%m")
    for scope, column in (("global", None), ("side", "side_name"), ("month", "month")):
        groups = [("all", pop)] if column is None else list(pop.groupby(column, observed=True, sort=True))
        for value, part in groups:
            identities = set(map(tuple, part.loc[:, IDENTITY].itertuples(index=False, name=None)))
            valid_id = set(map(tuple, part.loc[part.label_valid.astype(bool), IDENTITY].itertuples(index=False, name=None)))
            scored_id = set(map(tuple, scored.loc[:, IDENTITY].itertuples(index=False, name=None))) & identities
            mapped_id = set(map(tuple, mapped.loc[:, IDENTITY].itertuples(index=False, name=None))) & identities
            rows.append({
                "scope": scope, "value": str(value), "original_candidate_rows": len(identities),
                "valid_complete_label_rows": len(valid_id), "invalid_or_incomplete_rows": len(identities - valid_id),
                "held_out_development_scored_rows": len(scored_id), "causally_mapped_rows": len(mapped_id),
                "mapping_coverage_of_original": len(mapped_id) / max(len(identities), 1),
                "global_topk_denominator": "causally_mapped rows; original/valid/held-out/mapped counts disclosed",
            })
    return pd.DataFrame(rows)


def _selection_scorecard(summary: pd.DataFrame, *, round_id: int) -> pd.DataFrame:
    source = summary.loc[summary["round"].eq(round_id)].copy()
    rows: list[dict[str, Any]] = []
    for (arm, weight), part in source.groupby(["arm", "weight_mode"], observed=True, sort=True):
        pooled = part.loc[part["slice"].eq("pooled_global")].set_index("top_fraction")
        if not {0.01, 0.05, 0.10}.issubset(pooled.index):
            raise BaseTargetAblationError(f"{arm}/{weight} lacks pooled 1/5/10 tails")
        def tail_slice(name: str) -> pd.DataFrame:
            return part.loc[part["slice"].eq(name) & part.top_fraction.eq(.10)]
        eras, sides, regimes = tail_slice("era"), tail_slice("side_name"), tail_slice("causal_regime")
        # A custom causal-regime column is persisted as its own slice name.
        if regimes.empty:
            regimes = part.loc[
                ~part["slice"].isin(["pooled_global", "era", "side_name"])
                & part.top_fraction.eq(.10)
            ]
        era_values = pd.to_numeric(eras.net_bps_per_trade, errors="coerce")
        latest = eras.sort_values("slice_value", kind="mergesort").tail(1)
        rows.append({
            "arm": str(arm), "weight_mode": str(weight),
            "pooled_top10_net_bps": float(pooled.loc[.10, "net_bps_per_trade"]),
            "pooled_top1_net_bps": float(pooled.loc[.01, "net_bps_per_trade"]),
            "pooled_top5_net_bps": float(pooled.loc[.05, "net_bps_per_trade"]),
            "median_era_top10_net_bps": float(era_values.median()) if len(era_values) else np.nan,
            "worst_era_top10_net_bps": float(era_values.min()) if len(era_values) else np.nan,
            "worst_side_top10_net_bps": float(sides.net_bps_per_trade.min()) if len(sides) else np.nan,
            "worst_regime_top10_net_bps": float(regimes.net_bps_per_trade.min()) if len(regimes) else np.nan,
            "latest_era_top10_net_bps": float(latest.net_bps_per_trade.iloc[0]) if len(latest) else np.nan,
            "mapped_ev_monotonicity_violations": float(pooled.loc[.10, "mapped_ev_monotonicity_violations"]),
            "robust_top10_lift_score": float(pooled.loc[.10, "robust_top10_lift_score"]),
        })
    return pd.DataFrame(rows)


def _ordered_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "pooled_top10_net_bps", "pooled_top1_net_bps", "pooled_top5_net_bps",
            "median_era_top10_net_bps", "worst_era_top10_net_bps",
            "worst_side_top10_net_bps", "worst_regime_top10_net_bps",
            "mapped_ev_monotonicity_violations", "latest_era_top10_net_bps",
            "robust_top10_lift_score", "arm", "weight_mode",
        ],
        ascending=[False, False, False, False, False, False, False, True, False, False, True, True],
        kind="mergesort",
    )


def _run_model_cell(
    *, root: Path, frame: pd.DataFrame, arm: TargetArm | None, target_column: str,
    family: str, selected_contract: dict[str, Any], development_seed: int,
    evaluation_fraction: float,
    min_train_rows: int, weight_mode: str, regime_column: str, resume: bool,
    experiment_input_sha256: str,
    population: pd.DataFrame,
    model_cache: DevelopmentModelCache | None,
) -> tuple[dict[str, Any], DevelopmentModelCache | None]:
    selected_features = {side: selected_contract["sides"][side]["selected_features"] for side in ("long", "short")}
    fixed_params = {side: selected_contract["sides"][side]["fixed_params"] for side in ("long", "short")}
    arm_name = "R3_frozen_control" if arm is None else arm.name
    request = {
        "schema": "stage_i_base_target_model_cell_request_v2", "arm": arm_name,
        "family": family, "development_seed": development_seed,
        "evaluation_fraction": evaluation_fraction, "weight_mode": weight_mode,
        "min_train_rows": min_train_rows, "regime_column": regime_column,
        "selected_contract_sha256": selected_contract["contract_sha256"],
        "experiment_input_sha256": experiment_input_sha256,
        "target": target_column,
        "source_fingerprint": _model_cell_source_fingerprint(),
    }
    request_sha = _canonical_sha(request)
    if resume:
        prior = verify_completed_manifest(root, request_sha)
        if prior is not None:
            return prior, model_cache
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"model cell exists without valid --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    prediction, reference, split_audit, model_cache = run_development_holdout_arm(
        frame, arm=arm, target_column=target_column, family=family,
        selected_features=selected_features, fixed_params=fixed_params,
        seed=development_seed, min_train_rows=min_train_rows,
        weight_mode=weight_mode, regime_column=regime_column,
        evaluation_fraction=evaluation_fraction, model_cache=model_cache,
    )
    prediction_path = root / "target_repair_development_predictions.parquet"
    prediction.to_parquet(prediction_path, index=False, compression="zstd")
    reference_path = root / "development_mapping_reference_predictions.parquet"
    reference.to_parquet(reference_path, index=False, compression="zstd")
    metrics, calibration = _prediction_metrics(prediction, regime_column=regime_column)
    metrics_path = root / "target_repair_results.parquet"
    calibration_path = root / "mapped_ev_decile_economics.parquet"
    population_path = root / "population_denominator_audit.parquet"
    metrics.to_parquet(metrics_path, index=False, compression="zstd")
    calibration.to_parquet(calibration_path, index=False, compression="zstd")
    _population_audit(population, prediction).to_parquet(
        population_path, index=False, compression="zstd"
    )
    fold_path = root / "development_split_audit.json"
    fold_path.write_text(json.dumps(split_audit, indent=2, sort_keys=True) + "\n")
    inventory = {
        path.name: file_sha256(path)
        for path in (prediction_path, reference_path, metrics_path, calibration_path, population_path, fold_path)
    }
    manifest = {
        "schema": "stage_i_base_target_model_cell_v2", "status": "complete",
        "request": request, "request_sha256": request_sha, "artifact_sha256": inventory,
        "strict_oof": False,
        "evidence_status": "single_holdout_development_selection_not_final_oos",
        "causal_mapping": "prior-resolved training-reference side-local to common expected-net bps",
        "ranking": "pooled global only after common-bps conversion",
        "promotion_eligible": bool(arm is None or arm.promotion_eligible),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, model_cache


def _frame_bytes(frame: pd.DataFrame) -> int:
    """Conservative in-memory size used by the parallel admission guard."""

    return int(frame.memory_usage(index=True, deep=True).sum())


def _available_memory_bytes() -> int | None:
    """Return currently available memory without requiring psutil."""

    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _peak_rss_bytes() -> int:
    """Normalize ru_maxrss to bytes on Darwin and Linux."""

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def parallel_memory_admission(
    jobs: Iterable[dict[str, Any]], *, workers: int,
    available_bytes: int | None = None, budget_fraction: float = 0.65,
) -> dict[str, Any]:
    """Decide whether read-only-mmap process execution has a safe memory envelope.

    LightGBM creates pandas copies, contiguous float matrices, train/evaluation
    slices, quantized bins, histograms, and prediction buffers in every worker.
    The envelope therefore includes a fixed interpreter/model allowance plus a
    6x raw-frame multiplier, raised to 8x above 200 selected features. Worker
    count is reduced dynamically; fewer than two safe workers falls back to the
    proven sequential shared-bin implementation.
    """

    materialized = list(jobs)
    if workers < 1 or workers > 4:
        raise BaseTargetAblationError("parallel target workers must be in [1,4]")
    if not 0.20 <= float(budget_fraction) <= 0.90:
        raise BaseTargetAblationError("parallel memory budget fraction must be in [0.20,0.90]")
    sizes = [_frame_bytes(job["frame"]) + _frame_bytes(job["population"]) for job in materialized]
    # Count distinct objects for disclosure. They are already resident when
    # available-memory is sampled, so they are not charged again as incremental
    # worker RSS.
    unique_parent: dict[tuple[int, str], int] = {}
    for job in materialized:
        for name in ("frame", "population"):
            value = job[name]
            unique_parent[(id(value), name)] = _frame_bytes(value)
    parent_bytes = int(sum(unique_parent.values()))
    largest_job_bytes = int(max(sizes, default=0))
    selected_feature_count = max((
        len({
            str(feature)
            for side in ("long", "short")
            for feature in job["selected_contract"]["sides"][side]["selected_features"]
        })
        for job in materialized
    ), default=0)
    worker_raw_multiplier = 8.0 if selected_feature_count > 200 else 6.0
    fixed_worker_overhead_bytes = 320 * 1024 * 1024
    estimated_worker_peak_bytes = int(
        fixed_worker_overhead_bytes + worker_raw_multiplier * largest_job_bytes
    )
    serialization_headroom_bytes = int(1.25 * largest_job_bytes)
    available = _available_memory_bytes() if available_bytes is None else int(available_bytes)
    budget_bytes = None if available is None else int(available * float(budget_fraction))
    max_workers_by_memory = 0 if budget_bytes is None else max(
        0, int((budget_bytes - serialization_headroom_bytes) // max(estimated_worker_peak_bytes, 1)),
    )
    active_workers = min(int(workers), len(materialized), max_workers_by_memory)
    estimated_peak_bytes = int(
        serialization_headroom_bytes + estimated_worker_peak_bytes * active_workers
    )
    admitted = bool(
        len(materialized) > 1
        and workers > 1
        and budget_bytes is not None
        and active_workers >= 2
        and estimated_peak_bytes <= budget_bytes
    )
    reason = "admitted"
    if len(materialized) <= 1 or workers <= 1:
        reason = "parallelism_not_requested_or_single_job"
    elif budget_bytes is None:
        reason = "available_memory_unknown_fail_closed"
    elif active_workers < 2 or estimated_peak_bytes > budget_bytes:
        reason = "insufficient_memory_budget"
    return {
        "schema": "stage_i_target_parallel_memory_admission_v1",
        "admitted": admitted,
        "reason": reason,
        "requested_workers": int(workers),
        "active_workers": int(active_workers),
        "jobs": int(len(materialized)),
        "parent_shared_input_bytes": parent_bytes,
        "largest_job_bytes": largest_job_bytes,
        "selected_feature_count": int(selected_feature_count),
        "worker_raw_multiplier": float(worker_raw_multiplier),
        "fixed_worker_overhead_bytes": int(fixed_worker_overhead_bytes),
        "serialization_headroom_bytes": int(serialization_headroom_bytes),
        "estimated_worker_peak_bytes": int(estimated_worker_peak_bytes),
        "max_workers_by_memory": int(max_workers_by_memory),
        "estimated_peak_bytes": estimated_peak_bytes,
        "available_bytes": available,
        "budget_fraction": float(budget_fraction),
        "budget_bytes": budget_bytes,
        "sharing": "hash_bound_read_only_arrow_ipc_memory_maps",
        "lightgbm_threads_per_model": 1,
        "development_seeds": sorted({int(job["development_seed"]) for job in materialized}),
        "seed_contract": "explicit deterministic seed, identical scientific comparison across arms",
    }


def _isolated_parallel_cell_worker(job_id: int, job: dict[str, Any]) -> dict[str, Any]:
    """Run one hash-bound cell without exposing partial output as complete."""

    # Prevent nested native thread pools.  `_fixed_params` independently forces
    # every LightGBM booster to n_jobs=1 as part of the model contract.
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    destination = Path(job["root"])
    # Completed cells are verified in place and never copied or rewritten.
    if job["resume"] and destination.is_dir():
        manifest, _ = _run_model_cell(**job, model_cache=None)
        return {"job_id": int(job_id), "manifest": manifest, "resumed": True}
    staging = destination.with_name(f".{destination.name}.inprogress.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"isolated target-cell staging directory already exists: {staging}")
    local = dict(job)
    local["root"] = staging
    local["resume"] = False
    try:
        manifest, _ = _run_model_cell(**local, model_cache=None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"target cell appeared while worker was running: {destination}")
        os.replace(staging, destination)
        failure = destination.with_name(f"{destination.name}.failure.json")
        if failure.exists():
            failure.unlink()
        return {"job_id": int(job_id), "manifest": manifest, "resumed": False}
    except BaseException as error:
        failed = destination.with_name(f".{destination.name}.failed.{os.getpid()}")
        if staging.exists():
            os.replace(staging, failed)
        failure_payload = {
            "schema": "stage_i_target_parallel_cell_failure_v1",
            "status": "failed",
            "job_id": int(job_id),
            "arm": job["arm"].name if job["arm"] is not None else "R3_frozen_control",
            "weight_mode": job["weight_mode"],
            "experiment_input_sha256": job["experiment_input_sha256"],
            "development_seed": int(job["development_seed"]),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "recoverable_partial_directory": str(failed) if failed.exists() else None,
        }
        failure_payload["failure_sha256"] = _canonical_sha(failure_payload)
        failure = destination.with_name(f"{destination.name}.failure.json")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps(failure_payload, indent=2, sort_keys=True) + "\n")
        raise


def _write_read_only_ipc(frame: pd.DataFrame, path: Path) -> str:
    """Write an uncompressed Arrow file suitable for cross-process mmap."""

    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    path.chmod(0o444)
    return file_sha256(path)


def _read_verified_ipc(path: Path, expected_sha256: str) -> pd.DataFrame:
    """Verify lineage, then expose Arrow buffers read-only to pandas."""

    import pyarrow as pa
    import pyarrow.ipc as ipc

    if file_sha256(path) != expected_sha256:
        raise BaseTargetAblationError(f"parallel shared-input hash drift: {path}")
    source = pa.memory_map(str(path), "r")
    # Keep the memory map alive through table->pandas conversion. Primitive
    # blocks can remain zero-copy; string/timestamp blocks use bounded copies.
    table = ipc.open_file(source).read_all()
    return table.to_pandas(split_blocks=True, self_destruct=False)


def _parallel_worker_from_spec(spec_path: Path) -> int:
    """Fresh-interpreter worker entry point; never forks an OpenMP runtime."""

    spec = _json(spec_path)
    result_path = Path(spec["result_path"])
    rss_before_bytes = _peak_rss_bytes()
    try:
        claimed_spec_sha = str(spec.get("spec_sha256", ""))
        unsigned_spec = dict(spec)
        unsigned_spec.pop("spec_sha256", None)
        if claimed_spec_sha != _canonical_sha(unsigned_spec):
            raise BaseTargetAblationError("parallel worker-spec hash drift")
        arm_name = spec["job"].pop("arm_name")
        job = dict(spec["job"])
        job["root"] = Path(job["root"])
        job["arm"] = None if arm_name == "R3_frozen_control" else _arm_from_name(arm_name)
        job["frame"] = _read_verified_ipc(
            Path(spec["frame_path"]), str(spec["frame_sha256"]),
        )
        job["population"] = _read_verified_ipc(
            Path(spec["population_path"]), str(spec["population_sha256"]),
        )
        result = _isolated_parallel_cell_worker(int(spec["job_id"]), job)
        result["worker_peak_rss_bytes"] = _peak_rss_bytes()
        result["worker_peak_rss_delta_bytes"] = max(
            0, int(result["worker_peak_rss_bytes"]) - rss_before_bytes,
        )
        payload: dict[str, Any] = {"ok": True, "result": result}
        exit_code = 0
    except BaseException as error:
        payload = {
            "ok": False, "job_id": int(spec.get("job_id", -1)),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    result_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


def _terminate_and_reap_workers(
    active: dict[int, tuple[int, subprocess.Popen[Any]]], *, timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Terminate, then kill and wait every worker before mmap cleanup."""

    live = [process for _, process in active.values() if process.poll() is None]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + max(float(timeout_seconds), .05)
    for process in live:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    killed = 0
    for process in live:
        if process.poll() is None:
            process.kill()
            killed += 1
    for process in live:
        process.wait()
    return {
        "workers_seen": int(len(active)), "live_workers_terminated": int(len(live)),
        "workers_killed_after_timeout": int(killed),
        "all_workers_reaped": all(process.poll() is not None for _, process in active.values()),
    }


def run_parallel_model_cells(
    jobs: list[dict[str, Any]], *, workers: int = 3,
    memory_budget_fraction: float = 0.65,
    available_memory_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute independent arms in bounded fresh workers with failure isolation.

    The caller must use the sequential shared-`DevelopmentModelCache` path when
    the returned admission says ``admitted=False``.  A failed arm does not
    cancel unrelated futures; all failures are reported after the healthy arm
    checkpoints have completed.
    """

    admission = parallel_memory_admission(
        jobs, workers=workers, available_bytes=available_memory_bytes,
        budget_fraction=memory_budget_fraction,
    )
    if not admission["admitted"]:
        return [], admission
    for job in jobs:
        destination = Path(job["root"])
        if destination.exists() and (
            not destination.is_dir() or any(destination.iterdir())
        ) and not job["resume"]:
            raise FileExistsError(f"model cell exists without valid --resume: {destination}")
    results: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    result_root = Path(tempfile.mkdtemp(prefix="stage_i_target_workers_"))
    active: dict[int, tuple[int, subprocess.Popen[Any]]] = {}
    logs: dict[int, Any] = {}
    cleanup_audit: dict[str, Any] = {}
    try:
        # Deduplicate frames shared by several weight cells. The uncompressed
        # Arrow files are immutable, hash-bound, and memory-mapped by workers.
        shared: dict[tuple[int, str], tuple[Path, str]] = {}
        specs: dict[int, Path] = {}
        for job_id, job in enumerate(jobs):
            references: dict[str, tuple[Path, str]] = {}
            for field in ("frame", "population"):
                key = (id(job[field]), field)
                if key not in shared:
                    path = result_root / f"shared_{len(shared):03d}_{field}.arrow"
                    shared[key] = (path, _write_read_only_ipc(job[field], path))
                references[field] = shared[key]
            serializable = {
                key: value for key, value in job.items()
                if key not in {"frame", "population", "arm"}
            }
            serializable["root"] = str(serializable["root"])
            serializable["arm_name"] = (
                "R3_frozen_control" if job["arm"] is None else job["arm"].name
            )
            spec = {
                "schema": "stage_i_target_parallel_worker_spec_v1",
                "job_id": int(job_id), "job": serializable,
                "frame_path": str(references["frame"][0]),
                "frame_sha256": references["frame"][1],
                "population_path": str(references["population"][0]),
                "population_sha256": references["population"][1],
                "result_path": str(result_root / f"{job_id}.result.json"),
            }
            spec["spec_sha256"] = _canonical_sha(spec)
            spec_path = result_root / f"{job_id}.spec.json"
            spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
            specs[job_id] = spec_path
        pending = list(sorted(specs))
        worker_env = os.environ.copy()
        worker_env.update({
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "STAGE_I_ORDINAL_WORKERS": "1",
        })
        while pending or active:
            while pending and len(active) < int(admission["active_workers"]):
                job_id = pending.pop(0)
                log = (result_root / f"{job_id}.log").open("wb")
                logs[job_id] = log
                # Block Ctrl-C across the tiny Popen→registry window. A pending
                # interrupt is delivered only after the child is registered,
                # so the outer finally can always terminate and reap it.
                previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
                try:
                    process = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--parallel-worker-spec", str(specs[job_id])],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        env=worker_env, cwd=str(ROOT),
                    )
                    active[int(process.pid)] = (int(job_id), process)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            # The bounded worker count is tiny; a short poll avoids platform
            # semaphore dependencies and keeps completion order irrelevant.
            completed_pid = next(
                (pid for pid, (_, process) in active.items() if process.poll() is not None),
                None,
            )
            if completed_pid is None:
                time.sleep(.05)
                continue
            job_id, process = active.pop(completed_pid)
            logs.pop(job_id).close()
            status = process.returncode
            result_path = result_root / f"{job_id}.result.json"
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "ok": False, "job_id": int(job_id),
                    "error_type": "WorkerExitedWithoutResult",
                    "error": f"worker exit status {status}",
                }
            if payload.get("ok"):
                results[job_id] = payload["result"]
            else:
                failures.append({
                    "job_id": int(job_id),
                    "error_type": str(payload.get("error_type", "WorkerFailure")),
                    "error": str(payload.get("error", f"worker exit status {status}")),
                })
    finally:
        cleanup_audit = _terminate_and_reap_workers(active)
        for log in logs.values():
            if not log.closed:
                log.close()
        shutil.rmtree(result_root, ignore_errors=True)
    admission["completed_jobs"] = int(len(results))
    admission["failed_jobs"] = failures
    admission["worker_cleanup"] = cleanup_audit
    measured_rss = [
        int(result.get("worker_peak_rss_bytes", 0)) for result in results.values()
        if int(result.get("worker_peak_rss_bytes", 0)) > 0
    ]
    measured_delta = [
        int(result.get("worker_peak_rss_delta_bytes", 0)) for result in results.values()
    ]
    admission["measured_worker_peak_rss_bytes"] = measured_rss
    admission["measured_worker_peak_rss_max_bytes"] = max(measured_rss, default=0)
    admission["measured_worker_peak_rss_delta_max_bytes"] = max(measured_delta, default=0)
    admission["measured_to_estimated_worker_peak_ratio"] = (
        max(measured_rss, default=0) / max(int(admission["estimated_worker_peak_bytes"]), 1)
    )
    if failures:
        raise BaseTargetAblationError(
            "parallel target cells failed after healthy cells were preserved: "
            + json.dumps(failures, sort_keys=True)
        )
    return [results[index] for index in sorted(results)], admission


def _execute_model_job_batch(
    jobs: list[dict[str, Any]], *, workers: int, memory_budget_fraction: float,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Run one dependency-free stage, preserving the sequential cache fallback."""

    _, admission = run_parallel_model_cells(
        jobs, workers=workers, memory_budget_fraction=memory_budget_fraction,
    )
    if not admission["admitted"]:
        model_cache: DevelopmentModelCache | None = None
        for job in jobs:
            _, model_cache = _run_model_cell(**job, model_cache=model_cache)
        admission["execution_mode"] = "sequential_shared_lightgbm_bin_cache"
        admission["completed_jobs"] = int(len(jobs))
        admission["failed_jobs"] = []
    else:
        admission["execution_mode"] = "bounded_fresh_processes_read_only_arrow_mmap"
    rows: list[pd.DataFrame] = []
    for job in jobs:
        cell = Path(job["root"])
        local = pd.read_parquet(cell / "target_repair_results.parquet")
        local["arm"] = "R3_frozen_control" if job["arm"] is None else job["arm"].name
        local["weight_mode"] = job["weight_mode"]
        rows.append(local)
    return rows, admission


def _report(root: Path, *, request: dict[str, Any], summary: pd.DataFrame, stage: str) -> Path:
    path = root / "BASE_TARGET_ABLATION_REPORT.md"
    lines = [
        "# Stage-I base-target ablation", "", f"Status: **{stage} complete**", "",
        "The experiment uses the exact minute-bar open indexed at signal timestamp +1h (entry_ts equals decision_ts; no additional minute), H12 paths, adverse same-minute precedence, and excludes invalid/incomplete targets. The physical barriers are ATR-based; the 100 bps cost is applied once to economic evaluation and never moves a barrier.", "",
        "All model features come from the hash-bound per-side selected base manifests. Path progress, dominance, contract certainty, realised net, and labels are not inference features. Contract certainty is available only as a training weight.", "",
        "Model comparison uses one shared large chronological development holdout. Training labels must resolve strictly before the held-out start. This is fast development selection, not final OOS evidence; target-specific MDA and frozen 2024--2026 validation remain mandatory.", "",
        "The bounded process path shares hash-verified read-only Arrow input maps across fresh workers and forces every LightGBM model to one thread. If its memory gate rejects parallelism, the sequential fallback reuses one numeric matrix, purged split, and LightGBM bin dataset per side across arms. Four ordinal cumulative heads retain their deterministic contract.", "",
        "Ranking is pooled globally after side-local scores have been causally mapped into common expected-net bps.", "",
        "## Results", "",
    ]
    if len(summary):
        # Keep report publication independent of Pandas' optional ``tabulate``
        # dependency.  The experiment must never fail after durable model
        # checkpoints merely because a presentation-only package is absent.
        try:
            rendered = summary.head(100).to_markdown(index=False)
        except ImportError:
            rendered = "```text\n" + summary.head(100).to_string(index=False) + "\n```"
        lines.append(rendered)
    else:
        lines.append("No model round was requested yet.")
    lines += ["", "## Frozen request", "", f"`{request['request_sha256']}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _causal_three_class_error_target(
    prediction: pd.DataFrame, reference: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Training-reference terciles: overestimate / right / underestimate."""

    keys = IDENTITY + ["side_name", "decision_ts", "label_available_ts", "net_bps", "fold_id"]
    work = prediction.groupby(keys, observed=True, as_index=False).agg(
        raw_score=("raw_score", "mean"), expected_net_bps=("expected_net_bps", "mean")
    )
    work["economic_residual_bps"] = work.net_bps - work.expected_net_bps
    work["meta_error_class"] = -1
    work["meta_target_valid"] = False
    thresholds: list[dict[str, Any]] = []
    reference = reference.copy()
    reference["economic_residual_bps"] = reference.net_bps - reference.expected_net_bps
    evaluation_start = pd.to_datetime(work.decision_ts, utc=True, errors="raise").min()
    available = pd.to_datetime(reference.label_available_ts, utc=True, errors="raise")
    for side in ("long", "short"):
        current = work.index[work.side_name.eq(side)]
        prior = reference.loc[
            reference.side_name.eq(side)
            & available.lt(evaluation_start)
            & np.isfinite(reference.economic_residual_bps)
        ]
        if len(prior) < 100:
            thresholds.append({
                "side": side, "evaluation_start": evaluation_start.isoformat(),
                "prior_rows": int(len(prior)), "status": "insufficient_prior_training_residual_support",
            })
            continue
        q33, q67 = np.quantile(prior.economic_residual_bps, (1 / 3, 2 / 3))
        residual = work.loc[current, "economic_residual_bps"].to_numpy(float)
        valid = np.isfinite(residual)
        classes = np.where(residual < q33, 0, np.where(residual > q67, 2, 1)).astype(np.int8)
        work.loc[current[valid], "meta_error_class"] = classes[valid]
        work.loc[current[valid], "meta_target_valid"] = True
        thresholds.append({
            "side": side, "evaluation_start": evaluation_start.isoformat(),
            "prior_rows": int(len(prior)), "q33_residual_bps": float(q33),
            "q67_residual_bps": float(q67), "status": "prior_resolved_training_thresholds",
            "prior_label_available_max": available.loc[prior.index].max().isoformat(),
        })
    work["meta_error_semantics"] = np.select(
        [work.meta_error_class.eq(0), work.meta_error_class.eq(1), work.meta_error_class.eq(2)],
        ["base_overestimating", "base_approximately_right", "base_underestimating"],
        default="invalid_or_burnin",
    )
    return work, thresholds


def _export_winner_bundles(
    *, output_dir: Path, labels: pd.DataFrame, summary: pd.DataFrame,
    selected_contract: dict[str, Any], label_manifest: dict[str, Any], regime_column: str,
) -> list[dict[str, Any]]:
    scorecard = _selection_scorecard(summary, round_id=3)
    bundles: list[dict[str, Any]] = []
    root = output_dir / "winner_bundles"
    for family in ("scalar_S", "ordinal_O"):
        family_names = {arm.name for arm in target_arm_grid() if arm.family == family}
        local = _ordered_scorecard(scorecard.loc[scorecard.arm.isin(family_names)])
        if local.empty:
            raise BaseTargetAblationError(f"Round 3 produced no {family} winner")
        winner = local.iloc[0]
        arm = _arm_from_name(str(winner.arm))
        weight_mode = str(winner.weight_mode)
        cell = output_dir / "round3" / arm.name / weight_mode
        prediction_path = cell / "target_repair_development_predictions.parquet"
        reference_path = cell / "development_mapping_reference_predictions.parquet"
        if not prediction_path.is_file():
            raise BaseTargetAblationError(f"{family} winner development artifact is absent")
        prediction = pd.read_parquet(prediction_path)
        reference = pd.read_parquet(reference_path)
        target = labels.loc[
            labels.geometry.eq(arm.geometry.key) & labels.target_valid.astype(bool),
            IDENTITY + [
                "side_name", "decision_ts", "label_available_ts", regime_column,
                "geometry", "target_valid", "gross_bps", "net_bps",
                "event", "event_minute", "favorable_progress", "adverse_progress",
                "dominance", "upper_fraction", "lower_fraction",
                "upper_floor_bound", "upper_cap_bound",
                "contract_certainty", target_column_for_arm(arm),
            ],
        ].copy().rename(columns={target_column_for_arm(arm): "target_value", "target_valid": "label_valid"})
        target["target_valid"] = target["label_valid"].astype(bool)
        if target.loc[:, IDENTITY].duplicated().any() or not target.label_valid.all():
            raise BaseTargetAblationError("winner target handoff identities/validity are invalid")
        target["target_family"] = family
        target["target_name"] = arm.name
        target["ordinal_alpha"] = arm.ordinal_alpha
        target["weight_mode"] = weight_mode
        target["sample_weight_base_component"] = np.where(
            weight_mode == "uniform", 1.0,
            0.5 + 0.5 * target.contract_certainty.to_numpy(float),
        ).astype(np.float32)
        target["sample_weight_requires_fold_local_fit"] = weight_mode == "hybrid"
        bundle_root = root / family
        bundle_root.mkdir(parents=True, exist_ok=True)
        target_path = bundle_root / "winner_target_handoff.parquet"
        target.to_parquet(target_path, index=False, compression="zstd")
        oof_path = bundle_root / "winner_base_development_predictions.parquet"
        prediction.to_parquet(oof_path, index=False, compression="zstd")
        meta, thresholds = _causal_three_class_error_target(prediction, reference)
        meta["base_target_name"] = arm.name
        meta["base_target_geometry"] = arm.geometry.key
        meta["base_target_weight_mode"] = weight_mode
        meta_path = bundle_root / "meta_three_class_target_handoff.parquet"
        meta.to_parquet(meta_path, index=False, compression="zstd")
        threshold_path = bundle_root / "meta_three_class_threshold_audit.json"
        threshold_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
        weight_contract = {
            "schema": "stage_i_base_target_training_weight_contract_v1",
            "mode": weight_mode,
            "uniform": "weight=1",
            "contract_certainty": "weight=.5+.5*contract_certainty, then fold-train mean normalization",
            "hybrid": "certainty * mild fold-local chronology * mild fold-local causal-environment balance * mild fold-local target-class balance; normalize train mean; cap [.25,4]",
            "fit_scope": "recomputed from each permitted training fold only",
            "inference_feature": False,
            "row_handoff": "contract_certainty is provided; final weights must not be prefit globally",
        }
        weight_contract["contract_sha256"] = _canonical_sha(weight_contract)
        weight_path = bundle_root / "training_weight_contract.json"
        weight_path.write_text(json.dumps(weight_contract, indent=2, sort_keys=True) + "\n")
        artifacts = {
            path.name: file_sha256(path)
            for path in (target_path, oof_path, meta_path, threshold_path, weight_path)
        }
        manifest = {
            "schema": "stage_i_base_target_winner_bundle_v1", "status": "complete",
            "family": family, "target_name": arm.name, "geometry": arm.geometry.key,
            "geometry_contract": arm.geometry.to_dict(),
            "weight_mode": weight_mode,
            "selection_primary_top10_net_bps": float(winner.pooled_top10_net_bps),
            "selection_robust_score": float(winner.robust_top10_lift_score),
            "selected_feature_contract_sha256": selected_contract["contract_sha256"],
            "target_grid_label_artifact_sha256": label_manifest["artifact_sha256"]["target_repair_labels.parquet"],
            "target_handoff_semantics": "identity-aligned winner target and winner-geometry gross/net/validity/availability; never old selector TP6/SL4 net",
            "supporting_label_semantics": "winner-geometry event/progress/dominance/barrier diagnostics are training-only support labels; never inference features and never old TP6/SL4 support labels",
            "meta_three_class_semantics": {"0": "base overestimating", "1": "base approximately right", "2": "base underestimating"},
            "meta_thresholds": "side-local prior-resolved training-reference residual q33/q67 only",
            "evidence_status": "single_holdout_development_selection_not_final_oos",
            "artifact_sha256": artifacts,
        }
        manifest["bundle_sha256"] = _canonical_sha(manifest)
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        bundles.append({"family": family, "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path), **manifest})
    return bundles


def _export_joint_target_finalists(
    *, output_dir: Path, decision: dict[str, Any], winner_bundles: list[dict[str, Any]],
    base_selection_dir: Path, selected_contract: dict[str, Any],
    label_manifest: dict[str, Any], scorecard_path: Path,
    shared_population_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the three target contracts that require joint meta evaluation.

    Base-only economics are useful diagnostics and shortlist ordering, but the
    user contract explicitly forbids using them as the terminal promotion
    gate.  R3 and the best S/O family configurations therefore all survive to
    their matching direct FQ3 meta stacks.
    """

    by_family = {str(item["family"]): item for item in winner_bundles}
    required = {"scalar_S", "ordinal_O"}
    if set(by_family) != required:
        raise BaseTargetAblationError(
            "joint target shortlist requires exactly one scalar-S and ordinal-O bundle"
        )
    if shared_population_contract is not None:
        required_shared = {
            "schema", "path", "manifest_sha256", "population_file_sha256",
            "contract_sha256", "population_sha256", "rows", "per_side",
        }
        if required_shared.difference(shared_population_contract):
            raise BaseTargetAblationError(
                "joint target shortlist needs a fully signed shared-population reference"
            )
        if (
            shared_population_contract.get("schema") != "stage_i_joint_finalist_shared_population_reference_v1"
            or int(shared_population_contract.get("rows", 0)) < 1
            or set(dict(shared_population_contract.get("per_side", {}))) != {"long", "short"}
        ):
            raise BaseTargetAblationError("joint target shortlist shared-population reference is invalid")
    base_manifests = {}
    for side in ("long", "short"):
        path = base_selection_dir / side / "manifest.json"
        oof = base_selection_dir / side / "selector_base_oof.parquet"
        if not path.is_file() or not oof.is_file():
            raise BaseTargetAblationError("R3 finalist lacks complete same-side base artifacts")
        base_manifests[side] = {
            "manifest_path": str(path.resolve()),
            "manifest_sha256": file_sha256(path),
            "base_oof_path": str(oof.resolve()),
            "base_oof_sha256": file_sha256(oof),
        }
    finalists: list[dict[str, Any]] = []
    for item in decision.get("finalists", ()):
        family = str(item["family"])
        if family == "R3_control":
            source = {
                "kind": "existing_completed_r3_base_selection",
                "base_selection_dir": str(base_selection_dir.resolve()),
                "side_artifacts": base_manifests,
            }
        else:
            bundle = by_family[family]
            source = {
                "kind": "target_specific_winner_bundle_requires_new_base_mda",
                "bundle_manifest_path": str(Path(bundle["manifest"]).resolve()),
                "bundle_manifest_sha256": str(bundle["manifest_sha256"]),
                "bundle_sha256": str(bundle["bundle_sha256"]),
            }
        finalists.append({**item, "source": source})
    payload = {
        "schema": "stage_i_base_target_joint_finalists_v2",
        "status": "complete",
        "promotion_scope": "three_finalists_require_matching_direct_fq3_meta_before_any_terminal_decision",
        "base_only_economics_are_diagnostic": True,
        "decision_sha256": str(decision["decision_sha256"]),
        "scorecard_sha256": file_sha256(scorecard_path),
        "selected_feature_contract_sha256": selected_contract["contract_sha256"],
        "target_grid_label_artifact_sha256": label_manifest["artifact_sha256"]["target_repair_labels.parquet"],
        "shared_population_contract_sha256": None if shared_population_contract is None else shared_population_contract.get("contract_sha256"),
        "shared_population_path": None if shared_population_contract is None else shared_population_contract.get("path"),
        "shared_population": None if shared_population_contract is None else dict(shared_population_contract),
        "finalists": finalists,
        "terminal_gate": "compare reconstructed joint base+meta stacks on identical rows after causal common-bps mapping",
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    root = output_dir / "winner_bundles" / "joint_finalists"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "target_finalist_contracts.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": file_sha256(path), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--label-grid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--regime-column", default="causal_regime")
    parser.add_argument("--min-upper-support-rows", type=int, default=100)
    parser.add_argument("--max-timeout-prevalence", type=float, default=.90)
    parser.add_argument("--min-worst-regime-upper-rate", type=float, default=.005)
    parser.add_argument("--min-oracle-top10-net-bps", type=float, default=0.0)
    parser.add_argument("--round2-per-family", type=int, default=3)
    parser.add_argument("--development-seed", type=int, default=11)
    parser.add_argument("--evaluation-fraction", type=float, default=.25)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument(
        "--parallel-workers", type=int, choices=(1, 2, 3, 4), default=3,
        help="bounded target-cell processes; 1 forces the shared-bin sequential path",
    )
    parser.add_argument(
        "--parallel-memory-budget-fraction", type=float, default=.65,
        help="fraction of currently available RAM usable by read-only-mmap target workers",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0.10 <= args.evaluation_fraction <= 0.50:
        raise BaseTargetAblationError("--evaluation-fraction must be in [0.10,0.50]")
    if not 0.20 <= args.parallel_memory_budget_fraction <= 0.90:
        raise BaseTargetAblationError("--parallel-memory-budget-fraction must be in [0.20,0.90]")
    selected = require_selected_feature_contract(
        selector_dir=args.selector_dir, base_selection_dir=args.base_selection_dir,
    )
    labels, label_manifest = _require_labels(args.label_grid_dir)
    request = _request(args=args, selected=selected, label_manifest=label_manifest)
    prior_root_manifest = args.output_dir / "run_manifest.json"
    if prior_root_manifest.is_file():
        prior = _json(prior_root_manifest)
        prior_input = (prior.get("request") or {}).get("input_contract_sha256")
        if prior_input != request["input_contract_sha256"]:
            raise BaseTargetAblationError("base-target root input/gate contract drift across sequential rounds")
    gates = Round1Gates(
        min_upper_support_rows=args.min_upper_support_rows,
        max_timeout_prevalence=args.max_timeout_prevalence,
        min_worst_regime_upper_rate=args.min_worst_regime_upper_rate,
        min_oracle_top10_net_bps=args.min_oracle_top10_net_bps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    round1_root = args.output_dir / "round1"
    round1_root.mkdir(exist_ok=True)
    metrics_path, gates_path, slices_path = (
        round1_root / "target_oracle_metrics.parquet",
        round1_root / "target_rejection_gates.parquet",
        round1_root / "target_side_regime_metrics.parquet",
    )
    round1_request = {
        "schema": "stage_i_base_target_round1_request_v2",
        "experiment_input_sha256": request["input_contract_sha256"],
        "arm_inventory_sha256": _canonical_sha([arm.to_dict() for arm in target_arm_grid()]),
    }
    round1_request_sha = _canonical_sha(round1_request)
    prior_round1 = verify_completed_manifest(round1_root, round1_request_sha)
    if prior_round1 is None:
        if any(path.exists() for path in (metrics_path, gates_path, slices_path)):
            raise BaseTargetAblationError("partial/unmanifested Round-1 checkpoint requires a fresh output directory")
        metrics, gate_frame = round1_screen(labels, gates=gates, regime_column=args.regime_column)
        slices = _round1_slices(labels, target_arm_grid(), args.regime_column)
        metrics.to_parquet(metrics_path, index=False, compression="zstd")
        gate_frame.to_parquet(gates_path, index=False, compression="zstd")
        slices.to_parquet(slices_path, index=False, compression="zstd")
        (round1_root / "manifest.json").write_text(json.dumps({
            "schema": "stage_i_base_target_round1_v1", "status": "complete",
            "request": round1_request, "request_sha256": round1_request_sha,
            "artifact_sha256": {
                path.name: file_sha256(path) for path in (metrics_path, gates_path, slices_path)
            },
        }, indent=2, sort_keys=True) + "\n")
    metrics, gate_frame = pd.read_parquet(metrics_path), pd.read_parquet(gates_path)
    winners = _round1_winners(metrics, gate_frame, per_family=args.round2_per_family)
    model_rows: list[pd.DataFrame] = []
    parallel_audits: list[dict[str, Any]] = []
    selector_ledger, selector_feature_frame = _load_selector_feature_frame(args.selector_dir)
    if args.round >= 2:
        round2_jobs: list[dict[str, Any]] = []
        for arm in winners:
            frame, target_column, family, population = _model_frame(
                labels, args.selector_dir, arm, regime_column=args.regime_column,
                selector_ledger=selector_ledger, selector_feature_frame=selector_feature_frame,
            )
            frame = _prune_model_frame_to_selected_contract(
                frame, target_column=target_column, selected_contract=selected,
                regime_column=args.regime_column,
            )
            cell = args.output_dir / "round2" / arm.name / "uniform"
            round2_jobs.append(dict(
                root=cell, frame=frame, arm=arm, target_column=target_column, family=family,
                selected_contract=selected, development_seed=args.development_seed,
                evaluation_fraction=args.evaluation_fraction,
                min_train_rows=args.min_train_rows, weight_mode="uniform",
                regime_column=args.regime_column, resume=args.resume,
                experiment_input_sha256=request["input_contract_sha256"],
                population=population,
            ))
        round2_rows, round2_parallel = _execute_model_job_batch(
            round2_jobs, workers=args.parallel_workers,
            memory_budget_fraction=args.parallel_memory_budget_fraction,
        )
        round2_parallel["stage"] = "round2"
        parallel_audits.append(round2_parallel)
        for local in round2_rows:
            local["round"] = 2
            model_rows.append(local)
    if args.round >= 3:
        if not model_rows:
            raise BaseTargetAblationError(
                "Round 3 requires the exact current Round-1 winner cells to be verified/run first"
            )
        round2 = pd.concat(model_rows, ignore_index=True)
        round2_scorecard = _selection_scorecard(round2, round_id=2)
        finalists: list[TargetArm | None] = [None]
        for family in ("scalar_S", "ordinal_O"):
            names = {
                _arm_from_name(name).name for name in round2_scorecard.arm.unique()
                if _arm_from_name(name).family == family
            }
            local = _ordered_scorecard(round2_scorecard.loc[round2_scorecard.arm.isin(names)])
            if len(local):
                finalists.append(_arm_from_name(str(local.arm.iloc[0])))
        round3_jobs: list[dict[str, Any]] = []
        for arm in finalists:
            frame, target_column, family, population = _model_frame(
                labels, args.selector_dir, arm, regime_column=args.regime_column,
                selector_ledger=selector_ledger, selector_feature_frame=selector_feature_frame,
            )
            frame = _prune_model_frame_to_selected_contract(
                frame, target_column=target_column, selected_contract=selected,
                regime_column=args.regime_column,
            )
            arm_name = "R3_frozen_control" if arm is None else arm.name
            for weight_mode in ("uniform", "contract_certainty", "hybrid"):
                cell = args.output_dir / "round3" / arm_name / weight_mode
                round3_jobs.append(dict(
                    root=cell, frame=frame, arm=arm, target_column=target_column, family=family,
                    selected_contract=selected, development_seed=args.development_seed,
                    evaluation_fraction=args.evaluation_fraction,
                    min_train_rows=args.min_train_rows, weight_mode=weight_mode,
                    regime_column=args.regime_column, resume=args.resume,
                    experiment_input_sha256=request["input_contract_sha256"],
                    population=population,
                ))
        round3_rows, round3_parallel = _execute_model_job_batch(
            round3_jobs, workers=args.parallel_workers,
            memory_budget_fraction=args.parallel_memory_budget_fraction,
        )
        round3_parallel["stage"] = "round3"
        parallel_audits.append(round3_parallel)
        for local in round3_rows:
            local["round"] = 3
            model_rows.append(local)
    summary = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    summary_path = args.output_dir / "target_repair_results.parquet"
    summary.to_parquet(summary_path, index=False, compression="zstd")
    scorecard = (
        _selection_scorecard(summary, round_id=args.round)
        if args.round >= 2 and len(summary) else pd.DataFrame()
    )
    scorecard_path = args.output_dir / "target_selection_scorecard.parquet"
    scorecard.to_parquet(scorecard_path, index=False, compression="zstd")
    promotion_decision: dict[str, Any] | None = None
    joint_target_finalists: dict[str, Any] | None = None
    promotion_path = args.output_dir / "target_joint_shortlist_decision.json"
    if args.round == 3:
        try:
            promotion_decision = decide_round3_promotion(
                scorecard,
                source_contract={
                    "experiment_request_sha256": request["request_sha256"],
                    "experiment_input_contract_sha256": request["input_contract_sha256"],
                    "selector_manifest_sha256": file_sha256(args.selector_dir / "manifest.json"),
                    "selected_feature_contract_sha256": selected["contract_sha256"],
                    "label_grid_manifest_sha256": file_sha256(args.label_grid_dir / "manifest.json"),
                    "label_grid_artifact_sha256": label_manifest["artifact_sha256"]["target_repair_labels.parquet"],
                    "scorecard_sha256": file_sha256(scorecard_path),
                    "runner_source_fingerprint": _model_cell_source_fingerprint(),
                    "promotion_gate_module_sha256": file_sha256(
                        ROOT / "extreme_price_movements" / "stage_i_target_promotion.py"
                    ),
                    "ranking": "one pooled global rank after prior-resolved side-local common-bps mapping",
                },
            )
        except StageITargetPromotionError as error:
            raise BaseTargetAblationError(f"Round-3 immutable joint shortlist failed closed: {error}") from error
        promotion_path.write_text(json.dumps(promotion_decision, indent=2, sort_keys=True) + "\n")
    winner_bundles = (
        _export_winner_bundles(
            output_dir=args.output_dir, labels=labels, summary=summary,
            selected_contract=selected, label_manifest=label_manifest,
            regime_column=args.regime_column,
        )
        if args.round == 3 else []
    )
    if promotion_decision is not None:
        joint_target_finalists = _export_joint_target_finalists(
            output_dir=args.output_dir, decision=promotion_decision,
            winner_bundles=winner_bundles,
            base_selection_dir=args.base_selection_dir,
            selected_contract=selected, label_manifest=label_manifest,
            scorecard_path=scorecard_path,
        )
    report = _report(args.output_dir, request=request, summary=summary, stage=f"Round {args.round}")
    parallel_audit_path = args.output_dir / "parallel_execution_audit.json"
    parallel_audit_path.write_text(json.dumps({
        "schema": "stage_i_target_parallel_execution_audit_v1",
        "stages": parallel_audits,
        "scientific_parity_contract": "same arms, seed, 75/25 whole-timestamp purged holdout, side-local common-bps map, and pooled-global ranking as sequential execution",
    }, indent=2, sort_keys=True) + "\n")
    artifact_paths = sorted(
        {
            metrics_path, gates_path, slices_path, summary_path, scorecard_path, report,
            parallel_audit_path, *( [promotion_path] if promotion_path.is_file() else [] ),
            *(
                path for stage_root in (
                    args.output_dir / "round1", args.output_dir / "round2",
                    args.output_dir / "round3", args.output_dir / "winner_bundles",
                ) if stage_root.exists() for path in stage_root.rglob("*") if path.is_file()
            ),
        },
        key=lambda path: str(path),
    )
    manifest = {
        "schema": SCHEMA, "status": "complete", "completed_round": int(args.round),
        "request": request, "request_sha256": request["request_sha256"],
        "round1_arm_count": 60, "round1_promotable_arm_count": 48,
        "round2_arms": [arm.name for arm in winners],
        "winner_bundles": winner_bundles,
        "target_promotion_decision": None,
        "target_joint_shortlist_decision": (
            {"path": str(promotion_path), "sha256": file_sha256(promotion_path), **promotion_decision}
            if promotion_decision is not None else None
        ),
        "selected_target_contract": None,
        "joint_target_finalists": joint_target_finalists,
        "artifact_sha256": {str(path.relative_to(args.output_dir)): file_sha256(path) for path in artifact_paths},
        "next_step": "carry R3 plus best S/O through matching direct three-class meta stacks; select only on joint-stack causal common-bps economics",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "complete", "round": args.round, "round2_arms": manifest["round2_arms"],
        "output_dir": str(args.output_dir.resolve()), "report": str(report.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--parallel-worker-spec":
        raise SystemExit(_parallel_worker_from_spec(Path(sys.argv[2])))
    raise SystemExit(main())
