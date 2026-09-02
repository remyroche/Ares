#!/usr/bin/env python3
"""Independent v2 audit of sealed pre-2026 score-control/preregistration.

Only manifests, contracts, code text, and pre-2026 OOF outputs are opened.
In particular, this audit deliberately never opens a frozen-2026 candidate,
label, economics, replay, or portfolio file.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"
V3 = ART / "pre2026_oof_model_failure_incremental_value_20260730_v3"
V4 = ART / "pre2026_oof_model_failure_incremental_value_20260730_v4"
CONTROL = ART / "pre2026_oof_model_failure_incremental_value_score_control_20260730_v2"
CONTROL_V1 = ART / "pre2026_oof_model_failure_incremental_value_score_control_20260730_v1"
PREREG = ART / "frozen_2026_failure_value_correction_preregistration_20260730_v2"
PREVIOUS = ART / "pre2026_oof_model_failure_incremental_value_independent_review_20260730_v1"
OUT = ART / "pre2026_oof_model_failure_incremental_value_independent_review_20260730_v2_r2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path: Path) -> bool:
    return path.exists() and sha(path / "manifest.json") == (path / "manifest.sha256").read_text().split()[0]


def dump(path: Path, value: object) -> None:
    tmp = path.with_name("." + path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def all_output_hashes_match(path: Path) -> bool:
    manifest = json.loads((path / "manifest.json").read_text())
    return all((path / name).exists() and sha(path / name) == digest
               for name, digest in manifest.get("outputs_sha256", {}).items())


def metric_from_oof(frame: pd.DataFrame) -> float:
    if frame.target.iloc[0] == "selected_net_failure":
        return float(roc_auc_score(frame.actual_target, frame.prediction))
    return float(frame.prediction.corr(frame.actual_target, method="spearman"))


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prerequisites = {p.name: sealed(p) for p in [V3, V4, CONTROL, PREREG, PREVIOUS]}
    if not all(prerequisites.values()):
        raise RuntimeError(f"unsealed prerequisite: {prerequisites}")
    source_control = ROOT / "scripts" / "run_pre2026_score_only_incremental_control.py"
    source_model = ROOT / "scripts" / "run_pre2026_model_failure_incremental_value.py"
    source_prereg = ROOT / "scripts" / "preregister_frozen_2026_failure_value_corrections.py"
    control_contract = json.loads((CONTROL / "contract.json").read_text())
    control_manifest = json.loads((CONTROL / "manifest.json").read_text())
    prereg_contract = json.loads((PREREG / "contract.json").read_text())
    prereg_manifest = json.loads((PREREG / "manifest.json").read_text())
    v3_manifest_hash = sha(V3 / "manifest.json")
    v4_manifest_hash = sha(V4 / "manifest.json")
    control_manifest_hash = sha(CONTROL / "manifest.json")

    # Independently recompute every control fold metric from its sealed OOF
    # prediction rows; then independently recreate every paired context delta.
    control_metrics = pd.read_csv(CONTROL / "score_control_fold_metrics.csv")
    source_metrics = pd.read_csv(V3 / "fold_metrics.csv")
    metric_rows = []
    for target in ["incremental_selected_book_utility", "selected_net_failure"]:
        p = CONTROL / f"leave_era_oof_{target}_score_only_control.parquet"
        frame = pd.read_parquet(p, columns=["era", "side_name", "target", "prediction", "actual_target"])
        for era, pooled in frame.groupby("era", sort=True):
            for name, part in [("pooled", pooled), ("long", pooled[pooled.side_name.eq("long")]), ("short", pooled[pooled.side_name.eq("short")])]:
                observed = control_metrics[(control_metrics.target.eq(target)) &
                                           (control_metrics.era.eq(era)) &
                                           (control_metrics.scope.eq(name))].rank_metric
                recomputed = metric_from_oof(part)
                metric_rows.append({"target": target, "era": era, "scope": name,
                                    "reported_rank_metric": float(observed.iloc[0]) if len(observed) == 1 else np.nan,
                                    "recomputed_rank_metric": recomputed,
                                    "absolute_error": abs(float(observed.iloc[0]) - recomputed) if len(observed) == 1 else np.inf,
                                    "exact_metric_match": len(observed) == 1 and bool(np.isclose(float(observed.iloc[0]), recomputed, atol=1e-12, rtol=0))})
    metric_audit = pd.DataFrame(metric_rows)

    reported_delta = pd.read_csv(CONTROL / "context_vs_score_control_fold_deltas.csv")
    reported_gate = pd.read_csv(CONTROL / "context_incremental_gate.csv")
    paired = []
    for target in ["incremental_selected_book_utility", "selected_net_failure"]:
        baseline = control_metrics[(control_metrics.target.eq(target)) & control_metrics.scope.eq("pooled")][["era", "rank_metric"]].rename(columns={"rank_metric": "control"})
        for arm, context in source_metrics[(source_metrics.target.eq(target)) & source_metrics.scope.eq("pooled")].groupby("arm", sort=True):
            expected = context[["era", "rank_metric"]].rename(columns={"rank_metric": "context_rank_metric"}).merge(baseline, on="era", how="inner")
            expected["expected_delta"] = expected.context_rank_metric - expected.control
            got = reported_delta[(reported_delta.target.eq(target)) & reported_delta.arm.eq(arm)][["era", "rank_metric", "control_rank_metric", "incremental_rank_metric"]]
            joined = expected.merge(got, on="era", how="outer", indicator=True)
            values_match = (joined._merge.eq("both") &
                            np.isclose(joined.context_rank_metric, joined.rank_metric, atol=1e-12, rtol=0) &
                            np.isclose(joined.expected_delta, joined.incremental_rank_metric, atol=1e-12, rtol=0)).all()
            row = {"target": target, "arm": arm, "expected_matched_eras": len(expected), "reported_matched_eras": len(got),
                   "matched_fold_delta_math_exact": bool(values_match),
                   "max_delta_error": float((joined.expected_delta - joined.incremental_rank_metric).abs().max()) if len(joined) else np.nan}
            gate = reported_gate[(reported_gate.target.eq(target)) & reported_gate.arm.eq(arm)]
            if len(gate) != 1:
                row["gate_math_exact"] = False
            else:
                d = expected.expected_delta
                expected_gate = bool(len(d) >= 6 and d.median() > 0 and (d > 0).mean() >= .75 and d.min() >= -.02)
                gr = gate.iloc[0]
                row.update({"gate_math_exact": bool(gr.matched_eras == len(d) and np.isclose(gr.median_incremental_rank_metric, d.median(), atol=1e-12, rtol=0) and np.isclose(gr.min_incremental_rank_metric, d.min(), atol=1e-12, rtol=0) and np.isclose(gr.positive_era_fraction, (d > 0).mean(), atol=1e-12, rtol=0) and bool(gr.context_incremental_gate) == expected_gate),
                            "reported_context_gate": bool(gr.context_incremental_gate), "independently_recomputed_context_gate": expected_gate})
            paired.append(row)
    paired = pd.DataFrame(paired)

    declared_implementation = control_contract.get("implementation_sha256", {})
    v3_contract = json.loads((V3 / "contract.json").read_text())
    v3_source = source_model.read_text()
    prereg_implementation = prereg_contract.get("implementation_sha256", {})
    implementation = pd.DataFrame([
        {"source": str(source_control.resolve()), "declared": declared_implementation.get(str(source_control.resolve())), "actual": sha(source_control), "hash_matches": declared_implementation.get(str(source_control.resolve())) == sha(source_control)},
        {"source": str(source_model.resolve()), "declared": declared_implementation.get(str(source_model.resolve())), "actual": sha(source_model), "hash_matches": declared_implementation.get(str(source_model.resolve())) == sha(source_model)},
        {"source": str(source_prereg.resolve()), "declared": prereg_implementation.get(str(source_prereg.resolve())), "actual": sha(source_prereg), "hash_matches": prereg_implementation.get(str(source_prereg.resolve())) == sha(source_prereg)},
    ])
    prereg_control_hash = prereg_contract.get("prerequisite_artifacts", {}).get("score_only_control_manifest_sha256")
    prereg_source = source_prereg.read_text()
    binding = pd.DataFrame([{
        "control_v2_manifest_sealed": sealed(CONTROL), "control_v2_output_hashes_exact": all_output_hashes_match(CONTROL),
        "v3_manifest_binding_exact": control_manifest.get("inputs_sha256", {}).get(str((V3 / "manifest.json").resolve())) == v3_manifest_hash,
        "v4_manifest_binding_exact": control_manifest.get("inputs_sha256", {}).get(str((V4 / "manifest.json").resolve())) == v4_manifest_hash,
        "control_current_implementation_hashes_exact": bool(implementation.iloc[:2].hash_matches.all()),
        "v3_contract_declares_150k_cap": "150,000" in json.dumps(v3_contract),
        "v3_manifest_generator_source_hash_present": "implementation_sha256" in json.dumps(json.loads((V3 / "manifest.json").read_text())),
        "current_v3_generator_uses_v2_materialization_layout": "V2 / \"materialized_targets.parquet\"" in v3_source,
        "sealed_v3_layout_has_materialized_targets": (V3 / "materialized_targets.parquet").exists(),
        "current_v3_generator_writes_reference_not_materialized_target": "materialized_targets_reference.json" in v3_source,
        "prereg_manifest_sealed": sealed(PREREG), "prereg_output_hashes_exact": all_output_hashes_match(PREREG),
        "prereg_binds_current_control_v2": prereg_control_hash == control_manifest_hash,
        "prereg_binds_older_control_v1": CONTROL_V1.exists() and prereg_control_hash == sha(CONTROL_V1 / "manifest.json"),
        "prereg_generator_source_hash_present": str(source_prereg.resolve()) in prereg_implementation,
        "prereg_current_generator_hash_exact": bool(implementation.iloc[2].hash_matches),
        "prereg_source_static_no_candidate_parquet_read": "read_parquet" not in prereg_source and "pd.read_" not in prereg_source,
        "prereg_source_static_references_current_v2_control": "pre2026_oof_model_failure_incremental_value_score_control_20260730_v2" in prereg_source,
    }])
    no_2026_read = bool(binding.prereg_source_static_no_candidate_parquet_read.iloc[0])
    reconciliation = {
        "earlier_v1_verdict": "conditional score-only application only; not promotion",
        "new_score_control_result": "all eight context target/arm combinations fail the prespecified matched-era incremental gate against the score-only control",
        "reconciliation": "The new result tightens the earlier verdict: it is evidence to retain the score-only/raw-residual control and to exclude every tested context correction. It does not validate the raw residual as a tradable policy, remove the non-promotion safeguard, or repair nonchronological leave-era validation.",
    }
    review = {
        "no_2026_candidate_label_economics_replay_or_portfolio_file_opened": no_2026_read,
        "matched_fold_delta_arithmetic_exact": bool(paired.matched_fold_delta_math_exact.all() and paired.gate_math_exact.all()),
        "control_metrics_recomputed_exactly": bool(metric_audit.exact_metric_match.all()),
        "all_context_heads_rejected_by_reported_matched_score_control_gate": bool((~paired.reported_context_gate).all()),
        "matched_delta_authority": "NOT_AUTHORITATIVE_FOR_CONTEXT_VS_SCORE_INCREMENTALITY: the reported arithmetic is exact and score-control v2 is internally bound, but sealed v3 neither records the 150k cap nor a generator hash. The current v3 generator has a different v2-materialization/reference-output layout. Therefore identity of the context learner/cap to the control is unproven.",
        "implementation_hash_binding": "ADEQUATE_ONLY_FOR_SCORE_CONTROL_V2_INTERNAL_REPRODUCIBILITY: both current control-producing scripts hash-match the control contract; v3/v4 input manifests and recorded control output hashes match. It is not adequate to establish v3/context versus control learner identity.",
        "preregistration_hash_binding": "ADEQUATE_FOR_PROTOCOL_V2_BINDING_ONLY: preregistration v2 binds score-control v2 and its generator source hash, and it does not open a 2026 candidate/economics file. It cannot repair the unproven v3/context versus control learner identity.",
        "reconciliation": reconciliation["reconciliation"] + " This conclusion is now provisional only: the reported zero-head result is not an authoritative incremental comparison until joint regeneration proves identical learner implementation.",
        "frozen_2026_application_verdict": "NO_CONTEXT_HEAD_APPLICATION_AUTHORIZED: do not apply a context correction, assert that raw score-only beat context, tune on 2026, replay, or promote. First jointly regenerate v3-context and score-only OOF predictions with an identical sealed learner/cap implementation, then reseal preregistration against the corrected control. Preserve the prior non-promotion and chronological-validation safeguards.",
    }
    safeguards = [
        {"priority": 1, "safeguard": "Do not use the reported zero-context-head result to choose an application arm. It is an exact arithmetic reconciliation, not an authoritative matched-implementation comparison."},
        {"priority": 2, "safeguard": "Jointly regenerate context v3 and score-only control from one immutable all-row materialization, with one versioned model_predictions implementation. Seal its source/git hash, environment lock, feature lists, candidate-hash cap (150000), hashing implementation, float coercions, learner parameters and random states."},
        {"priority": 3, "safeguard": "For every target/arm/held-era/side, emit training candidate-ID digest before and after the cap, test candidate-ID digest, row counts and prediction-array digest. Require identical train/test rows and capped training identities for score-only versus each comparable context fold before computing a delta."},
        {"priority": 4, "safeguard": "Recompute control metrics and paired deltas from the jointly regenerated prediction arrays; only then apply the predeclared median/positive/minimum gate and state whether context adds value."},
        {"priority": 5, "safeguard": "The sealed preregistration v2 correctly binds the control-v2 manifest and its generator source. Supersede the earlier v1 protocol; after joint regeneration, issue one further preregistration only if its new comparison manifests differ."},
        {"priority": 6, "safeguard": "Keep model training and assessment rows at 1h. One-minute data remains nested label/path evidence or replay execution only."},
        {"priority": 7, "safeguard": "A later diagnostic must use a causal pooled global-top10 rule with deterministic ties and may not tune thresholds, mappings, weights, or choose an arm using 2026 outcomes."},
        {"priority": 8, "safeguard": "The result remains non-promotable: leave-era fitting includes later eras, so require chronological/OOS evidence before a policy change."},
    ]
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix="." + output.name + "."))
    try:
        metric_audit.to_csv(stage / "score_control_metric_recomputation_audit.csv", index=False)
        paired.to_csv(stage / "matched_fold_delta_and_gate_audit.csv", index=False)
        implementation.to_csv(stage / "implementation_hash_audit.csv", index=False)
        binding.to_csv(stage / "artifact_and_preregistration_binding_audit.csv", index=False)
        pd.DataFrame(safeguards).to_csv(stage / "required_v2_application_safeguards.csv", index=False)
        dump(stage / "review.json", review)
        contract = {"scope": "independent review of sealed pre-2026 v3/v4/control-v2 and preregistration-v1 artifacts; no 2026 data file opened", "cadence": "1h candidate/model rows; 1m nested label/path/replay only", "review_limit": "checks score-control arithmetic and binding; does not score or assess 2026"}
        dump(stage / "contract.json", contract)
        files = [p for p in stage.iterdir() if p.is_file()]
        manifest = {"schema": "pre2026_oof_model_failure_incremental_value_independent_review_v2", "status": "SEALED_SCORE_CONTROL_V2_AUDIT_NON_PROMOTION", "promotion_eligible": False, "review": review, "contract": contract,
                    "inputs_sha256": {str((p / "manifest.json").resolve()): sha(p / "manifest.json") for p in [V3, V4, CONTROL, PREREG, PREVIOUS]},
                    "outputs_sha256": {p.name: sha(p) for p in files}}
        dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
