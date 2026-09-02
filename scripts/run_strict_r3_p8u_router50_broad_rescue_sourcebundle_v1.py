#!/usr/bin/env python3
"""Source-bundle strict-OOF P8U rescue / Router blend ablation.

This recovery variant is intentionally narrow.  It reuses the immutable
outer-fold P8U probe models already fitted on prior-resolved rows, scores those
models across every target-free held candidate, and only then joins a sealed
held-outcome sidecar.  It exists because the raw historical label ledger used
for an otherwise-equivalent broad re-fit is temporarily unreadable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_p8u_opportunity_probe_router_recall_v1 as base  # noqa: E402
import run_strict_r3_p8u_router_conditioned_head_retrain_v1 as conditioned  # noqa: E402
import run_strict_r3_p8u_router50_broad_rescue_recall_v1 as shared  # noqa: E402


SCHEMA = "strict_r3_p8u_router50_broad_rescue_sourcebundle_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_outcomes(path: Path, definition: dict[str, Any]) -> pd.DataFrame:
    start = base._utc(definition["held_start"])
    end = base._exclusive_day_end(definition["held_end"])
    outcomes = pd.read_parquet(path, columns=[
        "candidate_id", "__decision_ts__", "label_valid", "policy_net_bps", "policy_label_available_ts",
    ])
    outcomes["__decision_ts__"] = pd.to_datetime(outcomes["__decision_ts__"], utc=True)
    outcomes["policy_label_available_ts"] = pd.to_datetime(outcomes["policy_label_available_ts"], utc=True)
    outcomes = outcomes.loc[outcomes["__decision_ts__"].ge(start) & outcomes["__decision_ts__"].lt(end)].copy()
    if outcomes["candidate_id"].duplicated().any():
        raise AssertionError(f"{definition['name']}: duplicate sealed held-outcome identity")
    return outcomes


def _fold(config: dict[str, Any], definition: dict[str, Any], outcome_path: Path) -> dict[str, Any]:
    fold = str(definition["name"])
    held_start = base._utc(definition["held_start"])
    held_end = base._exclusive_day_end(definition["held_end"])
    bundle_path = ROOT / config["source_head_bundles"][fold]
    bundle = conditioned._load_bundle(bundle_path)
    if str(bundle.get("fold")) != fold or base._utc(bundle["held_start"]) != held_start:
        raise AssertionError(f"{fold}: source bundle is not this exact outer OOS fold")
    models = list(bundle["probe_models"])
    head_specs = [dict(item) for item in bundle["head_specs"]]
    if len(models) != len(head_specs):
        raise AssertionError(f"{fold}: source bundle head/model count mismatch")

    predictive = base._assert_causal_feature_contract(base._P8U_FEATURE_CONTRACTS[config["feature_contract_keys"]["predictive"]])
    sidecar = base._assert_causal_feature_contract(base._P8U_FEATURE_CONTRACTS[config["probe_feature_sidecar_fields_key"]])
    # This reads the complete held target-free candidate universe and Router
    # rank first.  There is no outcome table in scope at this point.
    held, held_target_free, source_audit = base._read_panel(
        config, predictive, sidecar, start=held_start, end=held_end,
    )
    if not held["side_name"].eq("long").all():
        raise AssertionError(f"{fold}: non-long candidate entered source population")
    membership = bundle["category_model"].membership(held)
    p8u_score, _ = base._score_probe_stack(
        held, membership, models, head_specs, dict(bundle["combination"]),
    )
    router_rank = held["router_rank"].to_numpy(dtype=np.float32)
    p8u_rank = shared._timestamp_rank(held, p8u_score)
    router_ts_rank = shared._timestamp_rank(held, router_rank)
    # Only now attach the already sealed outcomes.  This operation is purely
    # for metrics and cannot affect source model scores or selections.
    outcomes = _fold_outcomes(outcome_path, definition)
    frame = held.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "router_rank"]].copy()
    frame["p8u_score"] = p8u_score.astype(np.float32)
    frame["p8u_timestamp_rank"] = p8u_rank
    frame["router_timestamp_rank"] = router_ts_rank
    frame = frame.merge(outcomes, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    if len(frame) != len(held) or not frame["candidate_id"].eq(held["candidate_id"]).all():
        raise AssertionError(f"{fold}: post-score held outcome join altered target-free identity/order")
    frame["label_valid"] = frame["label_valid"].fillna(False).astype(bool)
    retained = float(config["evaluation"]["retained_fraction"])
    selections: dict[str, np.ndarray] = {}
    metrics: list[dict[str, Any]] = []
    router = base._selection_mask(frame, router_rank, None, retained, 1.0)
    shared._assert_exact_capacity(frame, router, retained)
    selections["router_100"] = router
    metrics.append(shared._valid_selection_metrics(frame, router, fold=fold, family="capacity_rescue", arm="router_100", router_share=1.0))
    for share in config["evaluation"]["rescue_router_shares"]:
        share = float(share)
        if np.isclose(share, 1.0):
            continue
        selected = base._selection_mask(frame, router_rank, p8u_rank, retained, share)
        shared._assert_exact_capacity(frame, selected, retained)
        arm = f"router{int(round(share * 100)):02d}_p8u{int(round((1-share) * 100)):02d}_rescue"
        selections[arm] = selected
        metrics.append(shared._valid_selection_metrics(frame, selected, fold=fold, family="capacity_rescue", arm=arm, router_share=share))
    for weight in config["evaluation"]["blend_p8u_weights"]:
        weight = float(weight)
        blend = (1.0 - weight) * router_ts_rank + weight * p8u_rank
        selected = base._selection_mask(frame, blend, None, retained, 1.0)
        shared._assert_exact_capacity(frame, selected, retained)
        arm = f"router{int(round((1-weight) * 100)):02d}_p8u{int(round(weight * 100)):02d}_blend"
        selections[arm] = selected
        metrics.append(shared._valid_selection_metrics(frame, selected, fold=fold, family="score_blend", arm=arm, p8u_weight=weight))
    for arm, selected in selections.items():
        frame[f"{arm}_selected"] = selected
    source = {
        "fold": fold, "held_start": str(held_start), "held_end": str(held_end),
        "held_rows": int(len(frame)), "held_valid_rows": int(frame["label_valid"].sum()),
        "source_bundle": str(bundle_path.relative_to(ROOT)), "source_bundle_sha256": _sha256(bundle_path),
        "category_algorithm": str(bundle["category_model"].algorithm), "category_k": int(bundle["category_model"].k),
        "source_models_pretrained": True, "geometry_refit": False, "head_contract_retuned": False,
        "p8u_scored_full_target_free_universe": True,
        "outcomes_joined_after_source_model_scoring": True,
    }
    return {"metrics": pd.DataFrame(metrics), "predictions": frame, "source": source, "target_free": held_target_free, "source_audit": source_audit}


def _pooled(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in [item for item in predictions.columns if item.endswith("_selected")]:
        arm = column.removesuffix("_selected")
        family = "capacity_rescue" if arm == "router_100" or arm.endswith("_rescue") else "score_blend"
        router_share: float | None = None
        p8u_weight: float | None = None
        if arm == "router_100":
            router_share = 1.0
        elif arm.endswith("_rescue"):
            router_share = float(arm.removeprefix("router").split("_", 1)[0]) / 100.0
        elif arm.endswith("_blend"):
            # The compact arm name is e.g. router75_p8u25_blend; parse the
            # latter token rather than infer a rounded complement.
            p8u_weight = float(arm.split("_")[1].removeprefix("p8u")) / 100.0
        rows.append(shared._valid_selection_metrics(
            predictions, predictions[column].to_numpy(dtype=bool), fold="pooled_2025_q2_q4", family=family, arm=arm,
            router_share=router_share, p8u_weight=p8u_weight,
        ))
    return pd.DataFrame(rows)


def _report(output: Path, config_path: Path, source: pd.DataFrame, fold_metrics: pd.DataFrame, pooled: pd.DataFrame) -> None:
    order = pooled.sort_values(["recall_gt_50", "recall_gt_100", "recall_gt_200"], ascending=False, kind="stable")
    text = [
        "# Broad P8U Router-50% Rescue — Immutable Source-Bundle Variant\n",
        "## Scope and limitation\n",
        "This long-only strict-OOF study scores each fold's immutable, already prequentially fitted P8U specialist models across the **full target-free held universe**. It then compares exact Router 50% capacity, reserved P8U rescue capacity, and deterministic Router/P8U score blends. The raw historical training-label ledger was unavailable during this run, so the heads are **not re-fitted**. That limits this to a source-bundle rescue diagnostic, not a new P8U model-training result.\n",
        "Held policy outcomes come from a previously sealed candidate-level outcome sidecar. They are loaded only after P8U/Router target-free scoring and cannot affect model score, routing, capacity, or blend values. No live/canonical artifact changes.\n",
        "## Causal / source receipt\n",
        shared._format(source, ["fold", "held_start", "held_end", "held_rows", "held_valid_rows", "category_algorithm", "category_k", "source_models_pretrained", "geometry_refit", "head_contract_retuned", "p8u_scored_full_target_free_universe", "outcomes_joined_after_source_model_scoring"]),
        "## Pooled fixed-capacity metrics\n",
        "All arms retain exactly the Router's 50% timestamp-local capacity. Recall@50/100/200 is primary; net policy economics are a guardrail.\n",
        shared._format(order, ["family", "arm", "router_share", "p8u_weight", "selected_rows_valid", "recall_gt_50", "recall_gt_100", "recall_gt_200", "selected_hit_gt_50", "selected_mean_net_bps", "selected_cvar10_net_bps", "positive_economic_mass_recall"]),
        "## Strict-OOF metrics by quarter\n",
        shared._format(fold_metrics, ["fold", "family", "arm", "router_share", "p8u_weight", "selected_rows_valid", "recall_gt_50", "recall_gt_100", "recall_gt_200", "selected_mean_net_bps", "selected_cvar10_net_bps"]),
        f"Config: `{config_path.relative_to(ROOT)}`\n",
    ]
    (output / "P8U_ROUTER50_BROAD_RESCUE_SOURCEBUNDLE_REPORT.md").write_text("\n".join(text))


def run(config_path: Path, output: Path) -> None:
    config_path = config_path.resolve()
    config = base._load_config(config_path)
    if str(config.get("schema_version")) != SCHEMA:
        raise AssertionError("unexpected source-bundle rescue schema")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if not np.isclose(float(config["evaluation"]["retained_fraction"]), .5):
        raise AssertionError("this experiment must preserve exact Router top-50% capacity")
    outcome_path = ROOT / config["outcome_sidecar"]["path"]
    if not outcome_path.exists():
        raise FileNotFoundError(f"missing sealed outcome sidecar: {outcome_path}")
    output.mkdir(parents=True, exist_ok=False)
    results = [_fold(config, definition, outcome_path) for definition in config["outer_folds"]]
    metrics = pd.concat([item["metrics"] for item in results], ignore_index=True)
    predictions = pd.concat([item["predictions"] for item in results], ignore_index=True)
    source = pd.DataFrame([item["source"] for item in results])
    target_free = pd.concat([item["target_free"] for item in results], ignore_index=True)
    pooled = _pooled(predictions)
    base._write_parquet_exclusive(metrics, output / "fold_metrics.parquet")
    base._write_parquet_exclusive(pooled, output / "pooled_metrics.parquet")
    base._write_parquet_exclusive(predictions, output / "candidate_predictions.parquet")
    base._write_parquet_exclusive(target_free, output / "target_free_candidate_universe.parquet")
    base._write_parquet_exclusive(source, output / "fold_source_audit.parquet")
    correctness = {
        "schema": SCHEMA, "long_only": True,
        "target_free_universe_scored_before_outcome_sidecar_join": True,
        "full_universe_p8u_score": True, "source_models_are_outer_pretrained": True,
        "geometry_refit_per_fold": False, "head_contract_retuned": False,
        "held_outcomes_used_for_router_or_score": False,
        "all_arms_exact_timestamp_local_router50_capacity": True,
        "canonical_or_live_contract_changed": False,
    }
    base._write_json_exclusive(output / "correctness_report.json", correctness)
    manifest = {
        "schema": SCHEMA, "config_path": str(config_path.relative_to(ROOT)), "config_sha256": _sha256(config_path),
        "outcome_sidecar": str(outcome_path.relative_to(ROOT)), "outcome_sidecar_sha256": _sha256(outcome_path),
        "source_audits": [audit for item in results for audit in item["source_audit"]],
        "folds": source.to_dict(orient="records"), "decision": "OFFLINE_RESEARCH_ONLY",
    }
    base._write_json_exclusive(output / "run_manifest.json", manifest)
    _report(output, config_path, source, metrics, pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
