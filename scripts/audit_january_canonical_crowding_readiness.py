#!/usr/bin/env python3
"""Fail-closed provenance audit for adding January to canonical crowding work.

This script intentionally performs no scoring, label construction, calibration,
or outcome access.  It distinguishes a reconstructible candidate/covariate
universe from the *canonical* base-score and exact-economics contracts.  In
particular, ``historical_base_soft_oof`` is inventoried only as a prohibited
incompatible source and is never used to calculate a score band or result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/january_canonical_crowding_readiness_20260730_v1"
CANONICAL_BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"
CANONICAL_PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
JAN_NATIVE = ROOT / "data_perp/artifacts/january2025_native_first_touch_full_12h_paths_20260729_v1"
JAN_TWOLAYER = ROOT / "data_perp/artifacts/janfeb2025_execution_ev_exact1m_two_layer_oof_20260727_v1"
JAN_LEDGER = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if value is pd.NaT or (not isinstance(value, (str, bytes, bool)) and pd.isna(value)):
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _schema_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _quantile_code(values: pd.Series, bins: int = 4) -> pd.Series:
    """Outcome-free January-local group-size buckets; ties remain deterministic."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    result = pd.Series("missing", index=values.index, dtype="object")
    if finite.empty:
        return result
    edges = np.unique(np.quantile(finite.to_numpy(float), np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        result.loc[finite.index] = "constant"
        return result
    result.loc[finite.index] = pd.Series(
        np.searchsorted(edges[1:-1], finite.to_numpy(float), side="right"), index=finite.index
    ).map(lambda code: f"q{code}")
    return result


def assess_readiness(*, canonical_score_present: bool, exact_policy_labels_present: bool, covariates_present: bool) -> dict[str, Any]:
    """Single promotion gate: every canonical lineage prerequisite is required."""
    legal = canonical_score_present and exact_policy_labels_present and covariates_present
    return {
        "materialization_legal": legal,
        "status": "READY_FOR_BOUNDED_MATERIALIZATION" if legal else "NOT_READY_FAIL_CLOSED_NO_CANONICAL_JANUARY_SCORE_BRIDGE",
        "blocking_prerequisites": [
            name for name, present in {
                "canonical_base_score_same_recipe_and_replayable_fitted_state": canonical_score_present,
                "canonical_current_spread_exact_policy_12h_labels": exact_policy_labels_present,
                "canonical_causal_covariates": covariates_present,
            }.items() if not present
        ],
    }


def _load_january_universe() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for side in ("long", "short"):
        path = JAN_LEDGER / f"train_global_{side}_5_2025_01.parquet"
        frame = pd.read_parquet(path, columns=list(IDENTITY))
        if frame.candidate_id.duplicated().any():
            raise ValueError(f"January {side} ledger has duplicate candidate IDs")
        frames.append(frame)
    universe = pd.concat(frames, ignore_index=True)
    if universe.candidate_id.duplicated().any() or set(universe.side_name.unique()) != {"long", "short"}:
        raise ValueError("January identity/side contract fails")
    universe["__ts__"] = pd.to_datetime(universe["__ts__"], utc=True)
    universe["candidate_group_rows_timestamp_side"] = universe.groupby(["__ts__", "side_name"], observed=True)["candidate_id"].transform("size")
    universe["candidate_group_size_bin_january_local"] = _quantile_code(universe["candidate_group_rows_timestamp_side"])
    return universe


def _transition_availability() -> tuple[list[str], list[str]]:
    panel_manifest = json.loads((CANONICAL_PANEL / "manifest.json").read_text(encoding="utf-8"))
    canonical = sorted({
        item.replace("preentry_transition__", "").rsplit("__delta_", 1)[0]
        for item in panel_manifest["feature_groups"]["past_only_transition_deltas"]
    })
    available = _schema_names(JAN_LEDGER / "train_global_long_5_2025_01.parquet")
    aliases = {
        "meta_raw__chop_score": "__meta_raw__chop_score",
        "meta_raw__volatility_zscore": "__meta_raw__volatility_zscore",
        "regime_source_compression_score": "__regime_source_compression_score__",
        "regime_source_dirty_shock_avoid_score": "__regime_source_dirty_shock_avoid_score__",
        "regime_source_loud_breakout_impulse_score": "__regime_source_loud_breakout_impulse_score__",
        "regime_source_shock_impulse_score": "__regime_source_shock_impulse_score__",
    }
    present = [field for field in canonical if aliases.get(field, field) in available]
    return canonical, present


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    required = [CANONICAL_BASE / "manifest.json", CANONICAL_PANEL / "manifest.json", JAN_NATIVE / "manifest.json", JAN_TWOLAYER / "summary.json"]
    required += [JAN_LEDGER / f"train_global_{side}_5_2025_01.parquet" for side in ("long", "short")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("required canonical/January provenance inputs are absent")

    base_manifest = json.loads((CANONICAL_BASE / "manifest.json").read_text(encoding="utf-8"))
    native_manifest = json.loads((JAN_NATIVE / "manifest.json").read_text(encoding="utf-8"))
    two_layer = json.loads((JAN_TWOLAYER / "summary.json").read_text(encoding="utf-8"))
    canonical_months = sorted({Path(item).parent.name.rsplit("_", 1)[-1].replace("_", "-") for item in base_manifest["shard_manifests"]})
    universe = _load_january_universe()
    transition_required, transition_present = _transition_availability()
    twolayer_fields = _schema_names(JAN_TWOLAYER / "two_layer_direct_ev_strict_oof.parquet")

    # The strict-OOF score has enough fields to be tempting, but its manifest
    # documents a different label, score function, feature selection and model
    # state.  It is deliberately not read: no calibration or score-band bridge.
    prohibited = "historical_base_soft_oof"
    canonical_score_present = "2025-01" in canonical_months
    exact_policy_labels_present = False  # January native manifest expressly excludes these labels.
    covariates_present = set(transition_required) == set(transition_present)
    decision = assess_readiness(
        canonical_score_present=canonical_score_present,
        exact_policy_labels_present=exact_policy_labels_present,
        covariates_present=covariates_present,
    )

    support = (
        universe.groupby(["side_name", "candidate_group_size_bin_january_local"], observed=True)
        .agg(rows=("candidate_id", "size"), timestamps=("__ts__", "nunique"), assets=("__symbol__", "nunique"), mean_group_rows=("candidate_group_rows_timestamp_side", "mean"), min_group_rows=("candidate_group_rows_timestamp_side", "min"), max_group_rows=("candidate_group_rows_timestamp_side", "max"))
        .reset_index()
    )
    support["support_kind"] = "outcome_free_january_candidate_universe_crowding_only"
    support["high_score_support_status"] = "NOT_OBSERVABLE: canonical January base_oof_score absent; incompatible historical_base_soft_oof prohibited"
    inventory = pd.DataFrame([
        {"source": "canonical_base_oof", "rows_or_months": ",".join(canonical_months), "identity_side_asset": True, "candidate_group_size": False, "canonical_base_score": canonical_score_present, "exact_current_spread_policy_12h_label": False, "causal_transition_raw_fields": False, "legal_for_canonical_extension": False, "reason": "canonical OOF manifests cover Feb-Apr only; retained validation predictions are not a January score stream"},
        {"source": "january_native_first_touch_paths", "rows_or_months": native_manifest["rows"], "identity_side_asset": True, "candidate_group_size": True, "canonical_base_score": False, "exact_current_spread_policy_12h_label": False, "causal_transition_raw_fields": True, "legal_for_canonical_extension": False, "reason": "complete decision+12h native paths, explicitly not execution-EV labels or policy exits"},
        {"source": "january_raw_candidate_ledger", "rows_or_months": len(universe), "identity_side_asset": True, "candidate_group_size": True, "canonical_base_score": False, "exact_current_spread_policy_12h_label": False, "causal_transition_raw_fields": covariates_present, "legal_for_canonical_extension": False, "reason": "candidate universe and raw transition inputs exist; 12h prior history before January is still required for early-January deltas"},
        {"source": "janfeb_two_layer_strict_oof", "rows_or_months": two_layer["rows"]["strict_two_layer_direct_ev_oof"], "identity_side_asset": all(field in twolayer_fields for field in IDENTITY), "candidate_group_size": "candidate_group_size" in twolayer_fields, "canonical_base_score": False, "exact_current_spread_policy_12h_label": True, "causal_transition_raw_fields": False, "legal_for_canonical_extension": False, "reason": "explicitly incompatible: historical_base_soft_oof, sigmoid(execution_net_ev_12h/0.01), fold-local top-40 Spearman, seven-day expanding OOF; prohibited from pooling or calibration"},
    ])
    prerequisites = pd.DataFrame([
        {"prerequisite": "replayable fitted canonical base state for January", "status": "MISSING", "minimal_reconstruction_path": "retain the exact per-fold fitted model, preprocessing and AE/GMM state used by the frozen 31-long/8-short base contract; score January only after defining an admissible pre-January training/calibration state"},
        {"prerequisite": "canonical base OOF score stream and timestamp-side rank context", "status": "MISSING", "minimal_reconstruction_path": "materialize base_oof_score plus candidate-group/rank/cutoff context on the canonical January candidate IDs using the replayable fitted state; no conversion from historical_base_soft_oof"},
        {"prerequisite": "current-spread exact-policy H12 economics", "status": "MISSING", "minimal_reconstruction_path": "replay the canonical decision+12h execution policy from the complete January 1m paths with the canonical cost contract, then hash-bind it to January IDs"},
        {"prerequisite": "early-January causal t-3h/t-12h history", "status": "PARTIAL", "minimal_reconstruction_path": "materialize immediately preceding December raw feature history for each side/symbol; January raw fields cover the remainder, but not the leading horizon"},
        {"prerequisite": "one-to-one canonical population gate", "status": "PENDING_AFTER_ABOVE", "minimal_reconstruction_path": "build the January population gate and verify candidate_id, side, symbol, signal/decision timestamps and all hashes before any pooled February-to-March extension"},
    ])

    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        outputs = {
            "source_inventory.csv": inventory,
            "pre_outcome_crowding_support.csv": support,
            "missing_prerequisites.csv": prerequisites,
        }
        for name, table in outputs.items():
            table.to_csv(stage / name, index=False)
        manifest = {
            "schema": "january_canonical_crowding_readiness_v1",
            "status": decision["status"],
            "promotion_eligible": False,
            "materialization_legal": decision["materialization_legal"],
            "blocking_prerequisites": decision["blocking_prerequisites"],
            "contract": {
                "read_only_readiness_first": True,
                "outcome_access": False,
                "forbidden": ["historical_base_soft_oof pooling", "historical_base_soft_oof calibration bridge", "1m/100bps label pooling"],
                "canonical_score": "frozen 31-long/8-short exact same-run base OOF recipe; January stream absent",
                "q2_definition": "January-local q2 is the third deterministic quartile of timestamp-side candidate-group rows, reported only for outcome-free crowding support",
                "high_score_support": "not quantified: no canonical January base score exists and the only historical score is prohibited",
            },
            "sources": {
                "canonical_base_manifest": {"path": str(CANONICAL_BASE / "manifest.json"), "sha256": sha256(CANONICAL_BASE / "manifest.json")},
                "canonical_panel_manifest": {"path": str(CANONICAL_PANEL / "manifest.json"), "sha256": sha256(CANONICAL_PANEL / "manifest.json")},
                "january_native_manifest": {"path": str(JAN_NATIVE / "manifest.json"), "sha256": sha256(JAN_NATIVE / "manifest.json")},
                "january_two_layer_summary": {"path": str(JAN_TWOLAYER / "summary.json"), "sha256": sha256(JAN_TWOLAYER / "summary.json")},
            },
            "january_candidate_universe": {"rows": int(len(universe)), "unique_candidate_ids": int(universe.candidate_id.nunique()), "sides": sorted(universe.side_name.unique()), "assets": int(universe.__symbol__.nunique())},
            "transition_covariates": {"required_raw_fields": transition_required, "present_raw_fields": transition_present, "complete_within_january": covariates_present, "leading_history_gap": "December raw values are absent for January t-3h/t-12h transition deltas"},
            "outputs_sha256": {name: sha256(stage / name) for name in outputs},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(_safe(run(arguments.output_dir)), sort_keys=True))
