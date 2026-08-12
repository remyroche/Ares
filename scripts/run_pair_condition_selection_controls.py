#!/usr/bin/env python3
"""Equal-contract OOS refits for pair-condition selection controls.

The canonical v5 run already refits the full selection surface.  This utility
refits the four pair-valid discovery controls (random-supported,
geometry-only, no-model-utility, and no-feature-behavior) with the same folds,
residual target, LambdaRank parameters, side-local maps and global ranking.
The unary control remains discovery-only because it is not a valid pair
condition and therefore cannot be passed to the pair specialist contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pair_condition_specialists import (
    BASELINE_PRED,
    DISCOVERY_END,
    OUT as CANONICAL_OUT,
    SEED,
    _authoritative_metrics,
    _materialize_model_bank_manifest,
    _materialize_score_calibration,
    _materialize_side_artifacts,
    _run_outer,
    _write_json,
)
from extreme_price_movements.conditional_specialists import ConditionalSpecialistConfig


ARTIFACT_ROOT = ROOT / "data_perp/artifacts"
OUT = ARTIFACT_ROOT / "pair_condition_selection_controls_20260806"
CONTROL_NAMES = ("random_supported", "geometry_only", "no_model_utility", "no_feature_behavior")


def _load_frozen_contract() -> tuple[dict[str, dict[str, object]], dict[str, list[str]], dict[str, dict[str, list[str]]], dict[str, list[str]], dict[str, pd.DataFrame]]:
    root = CANONICAL_OUT
    manifests = {side: json.loads((root / f"condition_activation_manifest_{side}.json").read_text())["features"] for side in ("long", "short")}
    spines = {side: json.loads((root / f"condition_spine_manifest_{side}.json").read_text())["fields"] for side in ("long", "short")}
    pool = json.loads((root / "feature_pool_manifest.json").read_text())
    predictive = {side: list(pool["predictive_fields_by_side"][side]) for side in ("long", "short")}
    candidates = {side: pd.read_parquet(root / f"condition_candidates_{side}.parquet") for side in ("long", "short")}
    return manifests, spines, {}, predictive, candidates


def _control_candidates(side: str, candidates: pd.DataFrame, control: str) -> list[dict[str, object]]:
    selected = {
        str(c["condition_id"])
        for c in json.loads((CANONICAL_OUT / f"selected_conditions_{side}.json").read_text())["conditions"]
    }
    pool = candidates[~candidates.condition_id.astype(str).isin(selected)].copy()
    if control == "random_supported":
        rng = np.random.default_rng(SEED + (0 if side == "long" else 1))
        frame = pool.iloc[rng.permutation(len(pool))]
    elif control == "geometry_only":
        frame = candidates.sort_values(["effective_rows", "supported_month_count"], ascending=[False, False], kind="stable")
    elif control == "no_model_utility":
        frame = candidates.sort_values(["pair_interaction", "event_lift", "supported_month_count"], ascending=[False, False, False], kind="stable")
    elif control == "no_feature_behavior":
        model = pd.read_parquet(CANONICAL_OUT / f"condition_model_utility_portability_{side}.parquet")
        frame = candidates.merge(model[["condition_id", "portable_delta_top10_net_bps", "portable_delta_rank_ic"]], on="condition_id", how="left")
        frame["model_portability_score"] = frame.portable_delta_top10_net_bps.fillna(0.0) + 100.0 * frame.portable_delta_rank_ic.fillna(0.0)
        frame = frame.sort_values(["model_portability_score", "effective_rows"], ascending=[False, False], kind="stable")
    else:
        raise ValueError(f"unsupported pair control: {control}")
    return frame.head(3).to_dict("records")


def _feature_sets(side: str, conditions: list[dict[str, object]], predictive: list[str]) -> dict[str, list[str]]:
    portability = pd.read_parquet(CANONICAL_OUT / f"condition_feature_portability_{side}.parquet")
    result: dict[str, list[str]] = {}
    for condition in conditions:
        cid = str(condition["condition_id"])
        g = portability[portability.condition_id.eq(cid)].sort_values(
            ["portable_differential_rank_ic", "positive_month_fraction", "supported_months"],
            ascending=[False, False, False],
            kind="stable",
        )
        ordered = list(dict.fromkeys(g.feature.astype(str).tolist() + predictive))
        result[cid] = ordered[:40]
    return result


def _run_control(control: str) -> Path:
    out = OUT / control
    out.mkdir(parents=True, exist_ok=True)
    manifests, spines, _, predictive, candidates = _load_frozen_contract()
    selected = {side: _control_candidates(side, candidates[side], control) for side in ("long", "short")}
    feature_sets = {side: _feature_sets(side, selected[side], predictive[side]) for side in ("long", "short")}
    for side in ("long", "short"):
        _write_json(out / f"selected_conditions_{side}.json", {"schema": "selection_control_conditions_v1", "side": side, "control": control, "conditions": selected[side], "discovery_end_utc": DISCOVERY_END.isoformat()})
        _write_json(out / f"condition_feature_sets_{side}.json", {"schema": "selection_control_feature_sets_v1", "side": side, "control": control, "sets": feature_sets[side]})
    _write_json(out / "control_manifest.json", {"schema": "pair_condition_selection_control_refit_v1", "control": control, "target": "canonical ordinalized H12 net residual bps", "query": "4h x side", "mapping": "prequential_same_side_monotone_pava_20_bins_over_all_queries", "ranking": "mapped_common_bps_global_top_k", "selection_source": "discovery_only", "lomo": "not_run_for_control_refit"})

    # Reuse the base frame exactly as the canonical runner does.  Importing
    # here avoids a second copy in memory during module discovery.
    from scripts.run_pair_condition_specialists import _base_frame

    base = _base_frame()
    cfg = ConditionalSpecialistConfig(global_seed=SEED, condition_weight_exponent=1.5)
    predictions, _, _ = _run_outer(base, manifests, spines, feature_sets, selected, out, specialist_config=cfg)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    authoritative = _authoritative_metrics(out)
    _materialize_side_artifacts(out, predictions, authoritative, pd.DataFrame())
    _materialize_score_calibration(out, predictions)
    _materialize_model_bank_manifest(out, predictions)
    resource = {"base_rows": int(len(base)), "transport_rows": int(len(predictions)), "condition_weight_exponent": 1.5, "selection_control": control, "lomo": "not_run"}
    _write_json(out / "condition_resource_usage.json", resource)
    _write_json(out / "progress.json", {"status": "complete", "control": control, "folds": ["transport_long_2024_07_08", "transport_long_2024_09_10", "transport_long_2024_11_partial"]})
    return out


def run() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for control in CONTROL_NAMES:
        path = _run_control(control)
        metrics = pd.read_parquet(path / "global_metrics.parquet")
        rows.extend({"control": control, **rec} for rec in metrics[(metrics.scope == "global") & (metrics.period == "all")].to_dict("records"))
    result = pd.DataFrame(rows)
    result.to_parquet(OUT / "selection_control_refit_metrics.parquet", index=False)
    pooled = result[result["tail"].isin([.01, .05, .10])].pivot_table(index="control", columns="tail", values="net_bps", aggfunc="first").rename(columns={.01: "top1_net_bps", .05: "top5_net_bps", .10: "top10_net_bps"})
    pooled.to_csv(OUT / "selection_control_refit_summary.csv")
    lines = [
        "# Pair-condition selection-control refits",
        "",
        "All four controls use the canonical frozen target, 4-hour×side queries, fold boundaries, specialist/meta parameters, side-local EV maps and global common-bps ranking. Only discovery-time condition selection changes. The unary control is not refit because it is not a valid pair condition.",
        "",
        "## Global H12 net bps/trade",
        "",
        pooled.to_string(float_format=lambda x: f"{x:.2f}"),
        "",
        "Promotion remains gated by both top-5/top-10 improvement versus anchor and non-catastrophic worst month. These controls are diagnostic; no control is promoted automatically.",
    ]
    (OUT / "SELECTION_CONTROL_REFIT_REPORT.md").write_text("\n".join(lines) + "\n")
    _write_json(OUT / "run_manifest.json", {"schema": "pair_condition_selection_control_refit_v1", "controls": list(CONTROL_NAMES), "target": "canonical ordinalized H12 net residual bps", "selection_control_refits": True, "univariate": "discovery_only_not_pair_valid", "output": "selection_control_refit_metrics.parquet"})
    return OUT


if __name__ == "__main__":
    print(run())
