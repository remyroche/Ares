#!/usr/bin/env python3
"""Audit Feb--Apr base-top40/residual readiness without training a residual.

The only present historical OOF score stream predates the canonical 31/8 base
contract.  We materialize its deterministic top-40 diagnostic slice solely to
prove ranking mechanics, then gate it out of residual use until a fresh 31/8
base OOF has been trained on the accepted exact population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (
    BaseCandidatePopulationContract,
    candidate_identity_sha256,
    select_base_candidate_population,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
BASE_SCORE = "historical_base_soft_oof"
DEFAULT_ACCEPTED = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet"
DEFAULT_OOF = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_20260727_v2/two_layer_direct_ev_strict_oof.parquet"
DEFAULT_SUMMARY = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_20260727_v2/summary.json"
DEFAULT_BASE_AUDIT = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_20260727_v2/base_fold_audit.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_canonical_top40_residual_readiness_20260727_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _ranking_audit(source: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    grouped = source.groupby(["__ts__", "side_name"], sort=True).size()
    expected = grouped.map(lambda count: int(np.ceil(float(count) * 0.40)))
    selected_counts = selected.groupby(["__ts__", "side_name"], sort=True).size()
    aligned = expected.to_frame("expected").join(selected_counts.rename("selected"), how="left").fillna(0)
    valid_rank = (
        selected["base_candidate_rank_timestamp_side"]
        <= np.ceil(selected["base_candidate_group_rows"] * 0.40)
    ).all()
    return {
        "ranking_scope": "within_utc_timestamp_and_side",
        "top_fraction": 0.40,
        "tie_break": "score_desc_symbol_asc_stable_mergesort",
        "source_rows": int(len(source)),
        "selected_rows": int(len(selected)),
        "all_groups_exact_ceiling_fraction": bool((aligned["expected"] == aligned["selected"]).all()),
        "all_selected_ranks_within_ceiling": bool(valid_rank),
        "groups": int(len(grouped)),
    }


def _label_resolution_audit(oof: pd.DataFrame) -> dict[str, Any]:
    decision = pd.to_datetime(oof["execution_decision_utc"], utc=True, errors="raise")
    end = pd.to_datetime(oof["execution_label_end_utc"], utc=True, errors="raise")
    cutoff = pd.to_datetime(oof["base_oof_train_cutoff_utc"], utc=True, errors="raise")
    return {
        "decision_is_signal_plus_1h": bool((decision == pd.to_datetime(oof["__ts__"], utc=True) + pd.Timedelta(hours=1)).all()),
        "label_end_is_decision_plus_12h": bool((end == decision + pd.Timedelta(hours=12)).all()),
        "score_boundary_precedes_own_future_label_resolution": bool((cutoff <= end).all()),
        "score_boundary_is_not_after_scored_row_label": bool((cutoff <= decision).all()),
        "training_cutoff_column": "base_oof_train_cutoff_utc",
        "note": "Scored rows must retain future labels; the fold audit separately proves that training rows resolved before the score boundary.",
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    accepted = pd.read_parquet(args.accepted_population, columns=list(IDENTITY))
    oof = pd.read_parquet(args.oof, columns=[
        *IDENTITY, "execution_decision_utc", "execution_label_end_utc",
        "execution_net_ev_12h", "execution_soft_positive_12h", BASE_SCORE,
        "base_oof_fold_start_utc", "base_oof_train_cutoff_utc",
    ])
    oof["__ts__"] = pd.to_datetime(oof["__ts__"], utc=True, errors="raise")
    for frame in (accepted, oof):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if oof.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("historical base OOF identities are duplicated")
    in_accepted = oof.merge(accepted, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(in_accepted) != len(oof):
        raise ValueError("historical base OOF contains rows outside accepted exact population")
    if not np.isfinite(pd.to_numeric(oof[BASE_SCORE], errors="coerce")).all():
        raise ValueError("historical base OOF score is nonfinite")
    selected = select_base_candidate_population(
        oof,
        BaseCandidatePopulationContract(score_col=BASE_SCORE, top_fraction=0.40),
    )
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    base_audit = json.loads(args.base_audit.read_text(encoding="utf-8"))
    ranking = _ranking_audit(oof, selected)
    label_resolution = _label_resolution_audit(oof)
    trained_folds = [item for item in base_audit.get("folds", ()) if item.get("status") == "trained"]
    audit_purge = all(
        pd.Timestamp(item["max_train_label_end_utc"]) <= pd.Timestamp(item["fold_start_utc"])
        for item in trained_folds
    )
    label_resolution["trained_fold_count"] = int(len(trained_folds))
    label_resolution["base_fold_audit_resolved_training_purge"] = bool(audit_purge)
    base = summary.get("base", {})
    mismatch = {
        "fresh_canonical_31_8_base_oof_available": False,
        "feature_contract_matches_31_8": False,
        "feature_source": base.get("feature_selection"),
        "required_feature_contract": "frozen current canonical 31 long / 8 short feature manifests from accepted population gate",
        "hpo_provenance_available": False,
        "model_provenance": "HistGradientBoostingRegressor weekly reconstruction; no canonical LightGBM HPO winner/model manifest",
        "sample_weight_provenance_available": False,
        "reason": "existing Feb-Apr base score uses fold-local top-40 absolute-Spearman raw features, not fresh canonical 31/8 side-local feature/model/HPO contract",
    }
    args.output_dir.mkdir(parents=True)
    diagnostic_path = args.output_dir / "legacy_historical_top40_diagnostic_only.parquet"
    gate_path = args.output_dir / "residual_readiness_gate.json"
    selected.to_parquet(diagnostic_path, index=False, compression="zstd")
    gate = {
        "schema": "febapr_canonical_top40_residual_readiness_gate_v1",
        "accepted_population": {"path": str(args.accepted_population), "sha256": _sha256(args.accepted_population)},
        "historical_base_oof": {"path": str(args.oof), "sha256": _sha256(args.oof), "rows": int(len(oof)), "identity_overlap_with_accepted": int(len(in_accepted))},
        "legacy_diagnostic_top40": {"path": str(diagnostic_path), "sha256": _sha256(diagnostic_path), "identity_sha256": candidate_identity_sha256(selected)},
        "ranking": ranking,
        "target": {"base_target": "execution_soft_positive_12h = sigmoid(execution_net_ev_12h / 0.01)", "economic_target": "execution_net_ev_12h", "weights": "not persisted / unavailable"},
        "label_resolution": label_resolution,
        "legacy_base_provenance": {"summary_sha256": _sha256(args.summary), "base_fold_audit_sha256": _sha256(args.base_audit), "fold_count": len(base_audit.get("folds", ())), "side_local": bool(base.get("side_local"))},
        "canonical_mismatch": mismatch,
        "residual_reconstruction_ready": False,
        "forbidden": ["training or evaluating a residual on legacy_historical_top40_diagnostic_only.parquet", "calling the legacy fold-local top-40 feature screen canonical 31/8", "assuming unavailable sample weights/HPO/model provenance"],
        "required_before_residual": ["fresh side-local canonical 31/8 base OOF trained from accepted feature-store join keys", "per-fold resolved-label cutoff and persisted base model/HPO provenance", "canonical timestamp-side top40 materialized from that fresh OOF", "explicit residual target and sample-weight contract"],
    }
    _write_json(gate_path, gate)
    return {"diagnostic": diagnostic_path, "gate": gate_path}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--accepted-population", type=Path, default=DEFAULT_ACCEPTED)
    p.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--base-audit", type=Path, default=DEFAULT_BASE_AUDIT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
