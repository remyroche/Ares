#!/usr/bin/env python3
"""Validate the fresh Feb--Apr canonical 31/8 base OOF/top40 handoff.

Unlike the superseded legacy audit, this gate accepts the new canonical base
artifact only after recomputing its timestamp-side top40 membership and
checking native target, weight, feature/HPO/AE provenance and residual support.
It does not train a residual model.
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

from extreme_price_movements.base_candidate_population import BaseCandidatePopulationContract, select_base_candidate_population

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
DEFAULT_BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"
DEFAULT_TOP40 = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1"
DEFAULT_ACCEPTED = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_canonical_top40_residual_readiness_20260727_v3"


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


def month_start_residual_support(frame: pd.DataFrame) -> pd.DataFrame:
    """Count resolved canonical top40 rows before each month boundary."""

    work = frame.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["native_label_resolution_utc"] = pd.to_datetime(work["native_label_resolution_utc"], utc=True, errors="raise")
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for month in sorted(work["month"].unique()):
        start = pd.Period(month, freq="M").start_time.tz_localize("UTC")
        prior = work.loc[work["native_label_resolution_utc"].lt(start)]
        rows.append({
            "candidate_month": month,
            "month_start_utc": start,
            "prior_resolved_top40_rows": int(len(prior)),
            "prior_resolved_long_rows": int(prior["side_name"].eq("long").sum()),
            "prior_resolved_short_rows": int(prior["side_name"].eq("short").sum()),
            "monthly_residual_oof_supported": bool(len(prior) > 0),
        })
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_shard_contracts(
    *,
    base_manifest: Mapping[str, Any],
    provenance: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    """Verify that every materialized OOF shard carries the promoted 31/8 contract.

    The aggregate provenance records feature names and HPO trial IDs.  The
    shard manifests are the additional evidence for the fitted DAE/GMM state
    hashes, so inspect both rather than treating a name in a manifest as proof.
    """

    expected_count = {"long": 31, "short": 8}
    expected_hpo = {"long": "trial_141", "short": "trial_084"}
    records: list[dict[str, Any]] = []
    valid = True
    paths = [Path(str(value)) for value in base_manifest.get("shard_manifests", [])]
    if len(paths) != 6:
        valid = False
    for path in paths:
        if not path.is_file():
            valid = False
            continue
        shard = _read_json(path)
        side = str(shard.get("side_filter", "")).lower()
        month = str(shard.get("month_filter", ""))
        contracts = shard.get("contracts", {})
        contract = contracts.get(side, {}) if isinstance(contracts, dict) else {}
        matching = provenance.loc[
            provenance["side"].astype(str).str.lower().eq(side)
            & provenance["fold_id"].astype(str).eq(f"month_{month}")
        ]
        features = list(contract.get("features", [])) if isinstance(contract, dict) else []
        expected_features = list(matching.iloc[0]["features"]) if len(matching) == 1 else []
        ae_gmm_hashes = {
            key: str(contract.get(key, ""))
            for key in ("ae_gmm_state_sha256", "ae_gmm_state_metadata_sha256")
        }
        row_ok = bool(
            side in expected_count
            and len(matching) == 1
            and len(features) == expected_count.get(side)
            and features == expected_features
            and str(contract.get("hpo_trial_id", "")) == expected_hpo.get(side)
            and str(contract.get("hpo_trial_id", "")) == str(matching.iloc[0]["hpo_trial_id"])
            and all(len(value) == 64 for value in ae_gmm_hashes.values())
            and str(shard.get("label_purge", "")).startswith("native base label decision+24h < fold validation start")
        )
        valid &= row_ok
        records.append(
            {
                "path": str(path),
                "side": side,
                "fold_id": f"month_{month}",
                "contract_rows_match_provenance": row_ok,
                "feature_count": len(features),
                "hpo_trial_id": contract.get("hpo_trial_id"),
                **ae_gmm_hashes,
            }
        )
    # A fitted geometry must be immutable within side, rather than silently
    # varying with the month that was scored.
    for side in expected_count:
        states = {row["ae_gmm_state_sha256"] for row in records if row["side"] == side}
        metadata = {row["ae_gmm_state_metadata_sha256"] for row in records if row["side"] == side}
        if len(states) != 1 or len(metadata) != 1:
            valid = False
    return valid, {"shard_count": len(paths), "shards": records}


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    base_manifest = _read_json(args.base_dir / "manifest.json")
    top_manifest = _read_json(args.top40_dir / "manifest.json")
    base = pd.read_parquet(args.base_dir / "oof_predictions.parquet")
    top = pd.read_parquet(args.top40_dir / "population.parquet")
    accepted = pd.read_parquet(args.accepted_population, columns=["candidate_id", "side_name", "__symbol__", "__ts__"])
    for frame in (base, top, accepted):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        frame["candidate_id"] = frame["candidate_id"].astype(str)
    if len(base) != int(top_manifest["base_rows"]) or len(base) != len(accepted):
        raise ValueError("canonical base rows do not equal accepted population rows")
    if len(top) != int(top_manifest["selected_rows"]):
        raise ValueError("top40 row count does not match its manifest")
    if base.duplicated(list(IDENTITY), keep=False).any() or top.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("canonical base/top40 identities are duplicated")
    overlap = base.merge(accepted, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(overlap) != len(base):
        raise ValueError("canonical base OOF includes identities outside accepted population")
    recomputed = select_base_candidate_population(
        base.rename(columns={"base_oof_score": "score"}),
        BaseCandidatePopulationContract(score_col="score", top_fraction=0.40),
    )
    recomputed = recomputed.rename(columns={"base_candidate_rank_timestamp_side": "base_rank_timestamp_side", "base_candidate_group_rows": "base_group_rows", "base_candidate_rank_pct_timestamp_side": "base_rank_pct_timestamp_side"})
    compare_columns = [*IDENTITY, "base_rank_timestamp_side", "base_group_rows", "base_rank_pct_timestamp_side"]
    compare = recomputed.loc[:, compare_columns].merge(top.loc[:, compare_columns], on=list(IDENTITY), suffixes=("__recomputed", "__persisted"), how="outer", indicator=True, validate="one_to_one")
    rank_fields = ("base_rank_timestamp_side", "base_group_rows", "base_rank_pct_timestamp_side")
    rank_mismatches = 0
    for field in rank_fields:
        left = pd.to_numeric(compare[f"{field}__recomputed"], errors="coerce")
        right = pd.to_numeric(compare[f"{field}__persisted"], errors="coerce")
        rank_mismatches += int((~np.isclose(left, right, rtol=0.0, atol=1e-7, equal_nan=False)).sum())
    top_identity_match = bool(compare["_merge"].eq("both").all() and rank_mismatches == 0)
    numeric = top.loc[:, ["__first_touch_target_soft__", "__w__", "__first_touch_capture_net__", "execution_net_ev_12h"]].apply(pd.to_numeric, errors="coerce")
    native = {
        "target_column": "__first_touch_capture_net__",
        "soft_target_column": "__first_touch_target_soft__",
        "weight_column": "__w__",
        "all_native_values_finite": bool(np.isfinite(numeric.to_numpy(float)).all()),
        "positive_weight_rows": int(numeric["__w__"].gt(0.0).sum()),
        "nonpositive_weight_rows": int(numeric["__w__"].le(0.0).sum()),
    }
    provenance = pd.read_parquet(args.base_dir / "fold_provenance.parquet")
    if len(provenance) != 6:
        raise ValueError("expected six side-month canonical base folds")
    provenance["validation_start_utc"] = pd.to_datetime(provenance["validation_start_utc"], utc=True, errors="raise")
    provenance["train_base_label_resolution_max_utc"] = pd.to_datetime(provenance["train_base_label_resolution_max_utc"], utc=True, errors="raise")
    expected_count = {"long": 31, "short": 8}
    feature_ok = all(len(list(row.features)) == expected_count[str(row.side)] for row in provenance.itertuples())
    purge_ok = bool((provenance["train_base_label_resolution_max_utc"] < provenance["validation_start_utc"]).all())
    hpo_ok = bool(
        provenance.assign(side=provenance["side"].astype(str).str.lower()).apply(
            lambda row: str(row["hpo_trial_id"]) == {"long": "trial_141", "short": "trial_084"}[row["side"]], axis=1
        ).all()
    )
    shard_contracts_ok, shard_contracts = _audit_shard_contracts(base_manifest=base_manifest, provenance=provenance)
    support = month_start_residual_support(top)
    feb_supported = bool(support.loc[support["candidate_month"].eq("2025-02"), "monthly_residual_oof_supported"].item())
    args.output_dir.mkdir(parents=True)
    support_path = args.output_dir / "monthly_residual_support_boundary.csv"
    gate_path = args.output_dir / "residual_readiness_gate.json"
    support.to_csv(support_path, index=False)
    base_file_sha = _sha256(args.base_dir / "oof_predictions.parquet")
    base_hash_ok = bool(
        base_file_sha == str(base_manifest.get("outputs", {}).get("oof_predictions.parquet", ""))
        and base_file_sha == str(top_manifest.get("base_oof_sha256", ""))
    )
    base_oof_ok = bool(feature_ok and purge_ok and hpo_ok and shard_contracts_ok and base_hash_ok and top_identity_match and native["all_native_values_finite"] and native["nonpositive_weight_rows"] == 0)
    gate = {
        "schema": "febapr_canonical_top40_residual_readiness_gate_v3",
        "supersedes": "febapr2025_canonical_top40_residual_readiness_20260727_v2 legacy-only gate",
        "base": {"path": str(args.base_dir / "oof_predictions.parquet"), "sha256": base_file_sha, "manifest_and_top40_hash_match": base_hash_ok, "rows": int(len(base)), "accepted_identity_overlap": int(len(overlap))},
        "top40": {"path": str(args.top40_dir / "population.parquet"), "sha256": _sha256(args.top40_dir / "population.parquet"), "rows": int(len(top)), "exact_recomputed_identity_and_rank_match": top_identity_match, "rank_field_mismatches": int(rank_mismatches), "top_fraction": 0.40, "ranking_scope": "UTC timestamp x side", "tie_break": "score descending then symbol ascending, stable mergesort"},
        "canonical_base_provenance": {"six_side_month_folds": True, "feature_counts_match_31_8": bool(feature_ok), "hpo_trial_ids_valid": bool(hpo_ok), "native_label_purge_before_validation": bool(purge_ok), "ae_gmm_and_hpo_provenance": provenance.loc[:, ["side", "fold_id", "hpo_trial_id", "features", "train_base_label_resolution_max_utc", "validation_start_utc"]].to_dict(orient="records"), "shard_contracts_valid": bool(shard_contracts_ok), "shard_contracts": shard_contracts},
        "target_and_weight": native,
        "february_residual_support_boundary": {"path": str(support_path), "february_monthly_residual_oof_supported": feb_supported, "rule": "monthly residual OOF uses only prior resolved canonical top40 base-OOF rows; February has no earlier accepted month", "conclusion": "do not emit a February residual OOF score; March is the first monthly residual-evaluation boundary"},
        "base_and_top40_ready": base_oof_ok,
        "march_onward_residual_data_ready": bool(base_oof_ok and not feb_supported),
        "not_a_promotion_result": True,
        "remaining_before_residual": ["define/freeze residual model family and target transformation", "define/freeze residual training sample-weight use and HPO scope", "train only March onward under the recorded support boundary"],
    }
    _write_json(gate_path, gate)
    return {"gate": gate_path, "support": support_path}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    p.add_argument("--top40-dir", type=Path, default=DEFAULT_TOP40)
    p.add_argument("--accepted-population", type=Path, default=DEFAULT_ACCEPTED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p


if __name__ == "__main__":
    outputs = run(parser().parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
