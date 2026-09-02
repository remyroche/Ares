#!/usr/bin/env python3
"""Pre-register the short P0 -> O -> C -> K0 2×2 untouched experiment.

This producer deliberately performs *no* model fit and no evaluation.  It
freezes the four feature-contract combinations derived without 2025--26
policy-economics selection:

    A0  O45 / C59  frozen research control
    A1  O30 / C59  compact-O challenger
    A2  O45 / C40  compact-C challenger
    A3  O30 / C40  fully compact challenger

It also turns the source-lineage limitation discovered by the portability
audit into a machine-readable short-side live-readiness ledger.  There is no
sealed short runtime source contract today, so the ledger correctly grades no
feature A/B and all prospective production uses fail closed.  This is not a
claim about the separately deployed long stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round2 as r2  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round4_k0_refinement as r4  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_untouched_2x2_preregistration_v1"
PORTABILITY = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_feature_contract_portability_20260822_v3"
CONTRACTS = PORTABILITY / "predeclared_compact_contracts.json"
SOURCE_RELIABILITY = PORTABILITY / "feature_source_lineage_and_reliability.parquet"
PORTABILITY_SUMMARY = PORTABILITY / "feature_portability_summary.parquet"
C59_PREDICTIONS = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair_20260822_v1/C59_outer_oof_predictions.parquet"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_untouched_2x2_preregistration_20260822_v1"

# The frozen O/C research ledger ends in July.  Any evaluation beginning
# earlier than this would reopen evidence used to construct the contracts.
UNTOUCHED_START = pd.Timestamp("2026-08-01T00:00:00Z")
SIDE = "short"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_contracts(path: Path = CONTRACTS) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    contracts = payload["contracts"]
    expected = {
        "O45_canonical_frozen": 45,
        "O30_predeclared_stability_core": 30,
        "C59_canonical_frozen": 59,
        "C40_predeclared_stability_core": 40,
    }
    for name, size in expected.items():
        fields = tuple(contracts.get(name, ()))
        if len(fields) != size or len(set(fields)) != size:
            raise AssertionError(f"invalid predeclared contract {name}")
    if not set(contracts["O30_predeclared_stability_core"]).issubset(contracts["O45_canonical_frozen"]):
        raise AssertionError("O30 must be a strict portability-only subset of O45")
    if not set(contracts["C40_predeclared_stability_core"]).issubset(contracts["C59_canonical_frozen"]):
        raise AssertionError("C40 must be a strict portability-only subset of C59")
    return payload


def _arms(contracts: dict[str, Any]) -> list[dict[str, Any]]:
    values = contracts["contracts"]
    pairs = (
        ("A0", "frozen research control", "O45_canonical_frozen", "C59_canonical_frozen"),
        ("A1", "compact-O challenger", "O30_predeclared_stability_core", "C59_canonical_frozen"),
        ("A2", "compact-C challenger", "O45_canonical_frozen", "C40_predeclared_stability_core"),
        ("A3", "fully compact challenger", "O30_predeclared_stability_core", "C40_predeclared_stability_core"),
    )
    return [
        {
            "arm": arm, "description": description,
            "opportunity_contract": o_name, "conversion_contract": c_name,
            "opportunity_features": list(values[o_name]), "conversion_features": list(values[c_name]),
            "opportunity_feature_count": len(values[o_name]), "conversion_feature_count": len(values[c_name]),
        }
        for arm, description, o_name, c_name in pairs
    ]


def _frozen_stack() -> dict[str, Any]:
    c_target = next(target for target in r3.TARGETS if target.name == "C3_normalized_regret")
    # Explicit values, not a broad HPO space: all four arms must share them.
    o_model = r2._binary_config(r2.FROZEN_CONFIG, r3b.O_SEED)
    c_model = r3._model(c_target, r3b.C_SEED)
    return {
        "side": SIDE,
        "P0": "frozen F90 target-free candidate contract",
        "O": {
            "target": "mfe_6h_bps > 250", "target_name": "O250_H6",
            "weights": "uniform", "outer_probability_calibration": "Platt",
            "seed": r3b.O_SEED, "model_params": o_model.get_params(),
        },
        "C": {
            "target": c_target.name, "target_description": c_target.description,
            "conditional_population": "valid, prior-resolved O250/H6-positive training rows only",
            "weights": "uniform", "seed": r3b.C_SEED, "model_params": c_model.get_params(),
        },
        "K0": {
            "formula": "p(O) * mu1(C) + (1-p(O)) * mu0(P0_anchor)",
            "probability": "Platt fitted only on prior-resolved outer-OOF O scores",
            "mu1": {"kind": "isotonic", "k": 0},
            "mu0": {"kind": "anchor5", "k": 500},
            "admission": {"kind": "absolute_expected_policy_net_bps", "threshold_bps": 75.0},
            "policy_target": "frozen canonical short policy-net outcome; costs applied exactly once",
        },
        "shared_invariants": [
            "same target definitions, model parameters, seeds, weights, calibration and K0 across A0-A3",
            "strict chronological training: label_available_at < held decision fold start",
            "target-free candidate generation and complete candidate identities before labels join",
            "no 2025-2026 economics may select, re-rank, or modify a contract",
        ],
    }


def _readiness(arms: Sequence[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(SOURCE_RELIABILITY)
    portability = pd.read_parquet(PORTABILITY_SUMMARY)[["head", "feature", "hard_blacklist", "blacklist_reason"]]
    source = source.merge(portability, on=["head", "feature"], how="left", validate="one_to_one")
    if source.duplicated(["head", "feature"]).any() or len(source) != 104:
        raise AssertionError("source-readiness source must contain exactly one row per frozen O/C feature")
    # The audit established no sealed short current-hour runtime contract.  Do
    # not infer A/B merely because historical panels existed.  D means an
    # explicit historical portability failure; C means historical availability
    # exists but live ingestion/parity is still unproven.
    source["readiness_grade"] = np.where(source["hard_blacklist"].fillna(False), "D", "C")
    source["readiness_reason"] = np.where(
        source["readiness_grade"].eq("D"),
        "known historical coverage/variance portability failure",
        "historically available but no sealed short current-hour source/parity contract",
    )
    source["production_allowed"] = False
    source["production_disposition"] = "fail_closed_until_grade_A_or_B_and_short_runtime_parity_receipt"
    source["source_contract_status"] = "not_deployed_no_sealed_short_runtime_contract"
    used: list[dict[str, Any]] = []
    for arm in arms:
        for head, features in (("O", arm["opportunity_features"]), ("C", arm["conversion_features"])):
            local = source.loc[source["head"].eq(head) & source["feature"].isin(features)].copy()
            if len(local) != len(features):
                raise AssertionError(f"{arm['arm']}/{head} readiness ledger does not match frozen contract")
            local["arm"] = arm["arm"]
            local["contract_role"] = head
            used.append(local)
    by_arm = pd.concat(used, ignore_index=True)
    summary = by_arm.groupby("arm", as_index=False).agg(
        fields=("feature", "size"),
        grade_a_or_b=("readiness_grade", lambda value: int(value.isin(("A", "B")).sum())),
        grade_c=("readiness_grade", lambda value: int((value == "C").sum())),
        grade_d=("readiness_grade", lambda value: int((value == "D").sum())),
    )
    summary["production_ready"] = False
    summary["production_gate"] = "fail_closed: every required short field must be proven A/B by a sealed current-hour source and feature-parity receipt"
    return source.sort_values(["head", "source_group", "feature"], kind="stable").reset_index(drop=True), summary.sort_values("arm", kind="stable").reset_index(drop=True)


def validate_untouched_window(registry: dict[str, Any], evaluation_start: pd.Timestamp | str) -> pd.Timestamp:
    """Reject any retrospective/evidence-exhausted evaluation window."""
    start = pd.Timestamp(evaluation_start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    cutoff = pd.Timestamp(registry["untouched_protocol"]["earliest_evaluation_start"])
    if start < cutoff:
        raise ValueError(f"evaluation begins {start.isoformat()} before frozen untouched cutoff {cutoff.isoformat()}")
    return start


def validate_candidate_schema(registry: dict[str, Any], candidate_panel: Path, evaluation_start: pd.Timestamp | str) -> dict[str, Any]:
    """Schema-only preflight for a future held panel; never reads outcomes."""
    start = validate_untouched_window(registry, evaluation_start)
    names = set(ds.dataset(candidate_panel, format="parquet").schema.names)
    required = {"candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"}
    for arm in registry["arms"]:
        required.update(arm["opportunity_features"])
        required.update(arm["conversion_features"])
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"future target-free candidate panel lacks {len(missing)} preregistered fields: {missing}")
    return {
        "status": "schema_valid_target_free_panel_only",
        "evaluation_start": start.isoformat(),
        "candidate_panel": str(Path(candidate_panel).resolve()),
        "required_fields": len(required),
        "side": SIDE,
    }


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = [str(column) for column in frame.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def build(out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable preregistration already exists: {out}")
    contracts = _read_contracts()
    arms = _arms(contracts)
    frozen_stack = _frozen_stack()
    source, readiness_summary = _readiness(arms)
    latest = pd.to_datetime(pd.read_parquet(C59_PREDICTIONS, columns=["__decision_ts__"])["__decision_ts__"], utc=True).max()
    if latest >= UNTOUCHED_START:
        raise AssertionError("pre-registration cutoff must follow all source C59 decisions")
    registry = {
        "schema": SCHEMA,
        "status": "pre_registered_not_evaluated",
        "side": SIDE,
        "created_from": "target-free portability/redundancy and pre-existing chronological MDA only; no 2025-2026 policy economics used",
        "untouched_protocol": {
            "frozen_research_last_decision_ts": latest.isoformat(),
            "earliest_evaluation_start": UNTOUCHED_START.isoformat(),
            "prohibited": ["evaluation before cutoff", "contract modification after held outcomes", "feature-count HPO", "historical policy-economic arm selection"],
            "required": ["complete target-free candidate identities before label join", "strict label_available_at < held fold start", "same 15-minute? policy-label contract across all arms", "identical candidate IDs and portfolio eligibility across arms"],
        },
        "arms": arms,
        "frozen_stack": frozen_stack,
        "future_metrics": {
            "economics": ["net EV/trade", "trades", "total net bps", "CVaR10", "worst month", "worst week"],
            "opportunity": ["AUC", "PR-AUC", "Brier", "log loss", "calibration bins", "within-volatility Lift@20"],
            "conversion": ["conditional C monotonicity by score bins", "conditional MFE-bucket monotonicity", "rank IC"],
            "stability": ["monthly/weekly dispersion", "coverage", "source-readiness grade changes"],
        },
        "replacement_policy": {
            "automatic_promotion": False,
            "allowed_only_after_untouched_result": True,
            "head_replacement_requires": [
                "compact head is economically non-inferior to its matched broad-head control on the untouched block",
                "no material CVaR or worst-period degradation",
                "improved portability/source-readiness or materially simpler contract",
                "O calibration and C conditional monotonicity remain acceptable",
            ],
            "interpretation": "A1 isolates O compaction; A2 isolates C compaction; A3 tests their interaction. No arm is preselected as a winner.",
        },
        "sources": {
            "portability_contract_sha256": _sha256(CONTRACTS),
            "portability_manifest_sha256": _sha256(PORTABILITY / "run_manifest.json"),
            "source_reliability_sha256": _sha256(SOURCE_RELIABILITY),
            "C59_outer_oof_sha256": _sha256(C59_PREDICTIONS),
        },
    }
    registry["registry_sha256"] = _json_hash(registry)
    out.mkdir(parents=True)
    (out / "untouched_2x2_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    source.to_parquet(out / "short_feature_source_readiness.parquet", index=False, compression="zstd")
    readiness_summary.to_parquet(out / "arm_source_readiness_summary.parquet", index=False, compression="zstd")
    pd.DataFrame([
        {"arm": arm["arm"], "description": arm["description"], "O_fields": arm["opportunity_feature_count"], "C_fields": arm["conversion_feature_count"], "O_contract": arm["opportunity_contract"], "C_contract": arm["conversion_contract"]}
        for arm in arms
    ]).to_parquet(out / "arm_registry.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "scope": "pre-register a future untouched four-arm feature-contract experiment and short source-readiness grades; no fit, prediction, policy-economic comparison, or promotion",
        "registry_sha256": registry["registry_sha256"], "sources": registry["sources"],
        "source_readiness": "all fields are C or D because no sealed short current-hour runtime contract exists; production use must fail closed",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Short P0/O/C/K0 untouched 2×2 preregistration", "",
        "This receipt freezes four future feature-contract arms. It contains no model fit and no policy-economic result.", "",
        "## Frozen arms", "", _table(pd.read_parquet(out / "arm_registry.parquet")), "",
        "## Short live-source readiness", "", _table(readiness_summary), "",
        "All arms are research-only until every production field has an A/B grade backed by a sealed short current-hour source and training/inference parity receipt.", "",
        "## Frozen protocol", "", "```json", json.dumps(registry, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_UNTOUCHED_2X2_PREREGISTRATION.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--validate-evaluation-start", type=str)
    parser.add_argument("--candidate-panel", type=Path)
    args = parser.parse_args()
    if args.validate_evaluation_start:
        registry_path = args.out / "untouched_2x2_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"run preregistration first: {registry_path}")
        registry = json.loads(registry_path.read_text())
        if args.candidate_panel:
            print(json.dumps(validate_candidate_schema(registry, args.candidate_panel, args.validate_evaluation_start), indent=2))
        else:
            print(validate_untouched_window(registry, args.validate_evaluation_start).isoformat())
        return
    print(build(args.out))


if __name__ == "__main__":
    main()
