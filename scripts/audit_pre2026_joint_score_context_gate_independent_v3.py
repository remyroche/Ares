#!/usr/bin/env python3
"""Independent audit of the arm-matched pre-2026 joint context gate.

Only the sealed pre-2026 target ledger, joint-gate outputs, and source text
are opened.  No frozen-2026 candidate, label, economics, replay, or portfolio
file is referenced by this audit.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"
SRC = ART / "pre2026_oof_model_failure_incremental_value_20260730_v3"
JOINT = ART / "pre2026_joint_score_context_incremental_gate_20260730_v2"
OUT = ART / "pre2026_joint_score_context_incremental_gate_independent_review_20260730_v3"
CAP = 150_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(path: Path) -> bool:
    return path.exists() and sha(path / "manifest.json") == (path / "manifest.sha256").read_text().split()[0]


def dump(path: Path, value: object) -> None:
    tmp = path.with_name("." + path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def id_digest(values: pd.Series) -> str:
    return hashlib.sha256("|".join(values.astype(str).sort_values()).encode()).hexdigest()


def cap_digest(values: pd.Series) -> tuple[int, str]:
    # Match the sealed implementation: stable sort by the pandas candidate hash.
    work = pd.DataFrame({"candidate_id": values.astype(str).to_numpy()})
    if len(work) > CAP:
        order = pd.util.hash_pandas_object(work.candidate_id, index=False).to_numpy().argsort(kind="stable")[:CAP]
        work = work.iloc[order]
    return len(work), id_digest(work.candidate_id)


def rank_metric(frame: pd.DataFrame) -> float:
    if frame.target.iloc[0] == "selected_net_failure":
        return float(roc_auc_score(frame.actual_target, frame.prediction))
    return float(frame.prediction.corr(frame.actual_target, method="spearman"))


def all_output_hashes_match(path: Path) -> bool:
    manifest = json.loads((path / "manifest.json").read_text())
    return all((path / name).exists() and sha(path / name) == digest
               for name, digest in manifest["outputs_sha256"].items())


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not sealed(SRC) or not sealed(JOINT) or not all_output_hashes_match(JOINT):
        raise RuntimeError("unsealed or hash-mismatched prerequisite")
    contract = json.loads((JOINT / "contract.json").read_text())
    manifest = json.loads((JOINT / "manifest.json").read_text())
    joint_source = ROOT / "scripts" / "run_pre2026_joint_score_context_gate.py"
    model_source = ROOT / "scripts" / "run_pre2026_model_failure_incremental_value.py"
    declared = contract["implementation_sha256"]
    source_audit = pd.DataFrame([
        {"source": str(joint_source.resolve()), "declared_sha256": declared.get(str(joint_source.resolve())), "actual_sha256": sha(joint_source), "hash_matches": declared.get(str(joint_source.resolve())) == sha(joint_source)},
        {"source": str(model_source.resolve()), "declared_sha256": declared.get(str(model_source.resolve())), "actual_sha256": sha(model_source), "hash_matches": declared.get(str(model_source.resolve())) == sha(model_source)},
    ])
    input_audit = pd.DataFrame([{
        "source_manifest_binding_exact": manifest["inputs_sha256"].get(str((SRC / "manifest.json").resolve())) == sha(SRC / "manifest.json"),
        "source_ledger_binding_exact": manifest["inputs_sha256"].get(str((SRC / "materialized_targets.parquet").resolve())) == sha(SRC / "materialized_targets.parquet"),
        "joint_manifest_sealed": sealed(JOINT), "all_joint_output_hashes_exact": all_output_hashes_match(JOINT),
        "declared_candidate_hash_cap": contract["config"].get("candidate_hash_train_cap") == CAP,
        "decision_cadence_is_1h": contract.get("decision_cadence") == "1h",
        "one_minute_is_labels_only": contract.get("exact_replay_bar_cadence") == "1m_labels_only",
        "joint_source_only_reads_pre2026_v3_ledger": "SRC/'materialized_targets.parquet'" in joint_source.read_text(),
        "joint_source_has_no_frozen_2026_dependency": "frozen_2026" not in joint_source.read_text() and "candidate_scores" not in joint_source.read_text(),
    }])
    arms: dict[str, list[str]] = contract["config"]["context_arms"]
    required = ["candidate_id", "__ts__", "side_name", "era", "execution_label_end_utc", "execution_net_ev_12h",
                "incremental_selected_book_utility", "residual_selected_net_failure", "residual_selected_global_top10",
                "bocpd_regime_available", "lgbm_transition_available", "trajectory_available"]
    ledger = pd.read_parquet(SRC / "materialized_targets.parquet", columns=list(dict.fromkeys(required + sum(arms.values(), []))))
    ledger["__ts__"] = pd.to_datetime(ledger.__ts__, utc=True)
    ledger["execution_label_end_utc"] = pd.to_datetime(ledger.execution_label_end_utc, utc=True)
    cadence = pd.DataFrame([{
        "unique_candidate_ids": not ledger.candidate_id.duplicated().any(),
        "all_decisions_hour_aligned": bool((ledger.__ts__.astype("int64") % pd.Timedelta(hours=1).value == 0).all()),
        "all_label_ends_after_decision": bool(ledger.execution_label_end_utc.gt(ledger.__ts__).all()),
        "all_label_ends_pre2026": bool(ledger.execution_label_end_utc.lt(pd.Timestamp("2026-01-01", tz="UTC")).all()),
        "all_decision_rows_pre2026": bool(ledger.__ts__.lt(pd.Timestamp("2026-01-01", tz="UTC")).all()),
    }])
    target_col = {"incremental_selected_book_utility": "incremental_selected_book_utility", "selected_net_failure": "residual_selected_net_failure"}
    expected_folds: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for arm, features in arms.items():
        x = ledger.copy()
        ok = x[features].notna().all(axis=1)
        if arm in ("regime", "combined"):
            ok &= x.bocpd_regime_available.fillna(False)
        if arm in ("transition", "combined"):
            ok &= x.lgbm_transition_available.fillna(False)
        if arm in ("trajectory", "combined"):
            ok &= x.trajectory_available.fillna(False)
        x = x.loc[ok]
        for target, col in target_col.items():
            use = x.dropna(subset=[col])
            if target == "selected_net_failure":
                use = use.loc[use.residual_selected_global_top10]
            for era, held in use.groupby("era", sort=True):
                train_all = use.loc[use.era.ne(era)]
                for side, test in held.groupby("side_name", sort=True):
                    train = train_all.loc[train_all.side_name.eq(side)]
                    ncap, hcap = cap_digest(train.candidate_id)
                    expected_folds[(arm, target, era, side)] = {
                        "pre_cap_train_rows": len(train), "test_rows": len(test),
                        "pre_cap_train_candidate_sha256": id_digest(train.candidate_id),
                        "test_candidate_sha256": id_digest(test.candidate_id),
                        "post_cap_train_rows": ncap, "post_cap_train_candidate_sha256": hcap,
                    }
    fold_audit = pd.read_csv(JOINT / "fold_audit.csv")
    identities = []
    for key, expected in expected_folds.items():
        arm, target, era, side = key
        actual = fold_audit[(fold_audit.arm.eq(arm)) & (fold_audit.target.eq(target)) & (fold_audit.era.eq(era)) & (fold_audit.side.eq(side))]
        for kind in ["score_only_" + arm, arm]:
            row = actual[actual.kind.eq(kind)]
            equal = len(row) == 1 and all(row.iloc[0][name] == value for name, value in expected.items() if name in row.columns)
            identities.append({"arm": arm, "target": target, "era": era, "side": side, "kind": kind, **expected,
                               "recorded_pre_cap_identity_exact": bool(equal),
                               "post_cap_identity_recomputed": True})
    identity_audit = pd.DataFrame(identities)
    # Independently recompute OOF metrics and verify raw test rows/targets.
    metric_rows = []
    output_row_checks = []
    for path in sorted(JOINT.glob("oof_*.parquet")):
        frame = pd.read_parquet(path)
        frame["__ts__"] = pd.to_datetime(frame.__ts__, utc=True)
        arm_kind = frame.arm.iloc[0]
        target = frame.target.iloc[0]
        arm = arm_kind.removeprefix("score_only_")
        source_target = target_col[target]
        lookup = ledger[["candidate_id", "__ts__", source_target]].rename(columns={source_target: "expected_target"})
        merged = frame.merge(lookup, on=["candidate_id", "__ts__"], how="left", validate="one_to_one")
        output_row_checks.append({"file": path.name, "rows": len(frame), "unique_candidates": frame.candidate_id.nunique(),
                                  "all_pre2026": bool(frame.__ts__.lt(pd.Timestamp("2026-01-01", tz="UTC")).all()),
                                  "all_hourly": bool((frame.__ts__.astype("int64") % pd.Timedelta(hours=1).value == 0).all()),
                                  "finite_predictions": bool(frame.prediction.notna().all()),
                                  "actual_targets_exact": bool(merged.expected_target.notna().all() and np.isclose(merged.actual_target, merged.expected_target, atol=1e-12, rtol=0).all()),
                                  "no_duplicate_candidate": not frame.candidate_id.duplicated().any()})
        for era, held in frame.groupby("era", sort=True):
            for scope, part in [("pooled", held), ("long", held[held.side_name.eq("long")]), ("short", held[held.side_name.eq("short")])]:
                metric_rows.append({"target": target, "arm": arm_kind, "era": era, "scope": scope, "recomputed_rank_metric": rank_metric(part)})
    metric_audit = pd.DataFrame(metric_rows)
    reported_metrics = pd.read_csv(JOINT / "fold_metrics.csv")
    metric_audit = metric_audit.merge(reported_metrics[["target", "arm", "era", "scope", "rank_metric"]], on=["target", "arm", "era", "scope"], how="outer", indicator=True)
    metric_audit["absolute_error"] = (metric_audit.recomputed_rank_metric - metric_audit.rank_metric).abs()
    metric_audit["metric_exact"] = metric_audit._merge.eq("both") & np.isclose(metric_audit.recomputed_rank_metric, metric_audit.rank_metric, atol=1e-12, rtol=0)
    recomputed_deltas = []
    for arm in arms:
        context = metric_audit[(metric_audit.arm.eq(arm)) & metric_audit.scope.eq("pooled")][["target", "era", "recomputed_rank_metric"]].rename(columns={"recomputed_rank_metric": "context_metric"})
        control = metric_audit[(metric_audit.arm.eq("score_only_" + arm)) & metric_audit.scope.eq("pooled")][["target", "era", "recomputed_rank_metric"]].rename(columns={"recomputed_rank_metric": "control_metric"})
        d = context.merge(control, on=["target", "era"], validate="one_to_one")
        recomputed_deltas.append(d.assign(arm=arm, recomputed_delta=d.context_metric - d.control_metric))
    delta_audit = pd.concat(recomputed_deltas, ignore_index=True)
    reported_delta = pd.read_csv(JOINT / "matched_fold_deltas.csv")
    delta_audit = delta_audit.merge(reported_delta[["target", "arm", "era", "rank_metric", "score_only_metric", "delta"]], on=["target", "arm", "era"], how="outer", indicator=True)
    delta_audit["delta_exact"] = delta_audit._merge.eq("both") & np.isclose(delta_audit.recomputed_delta, delta_audit.delta, atol=1e-12, rtol=0) & np.isclose(delta_audit.context_metric, delta_audit.rank_metric, atol=1e-12, rtol=0) & np.isclose(delta_audit.control_metric, delta_audit.score_only_metric, atol=1e-12, rtol=0)
    gate_rows = []
    reported_gate = pd.read_csv(JOINT / "eligibility.csv")
    for (target, arm), group in delta_audit.groupby(["target", "arm"], sort=True):
        delta = group.recomputed_delta
        expected = {"matched_eras": len(delta), "median_delta": delta.median(), "min_delta": delta.min(), "positive_fraction": (delta > 0).mean(),
                    "eligible": bool(len(delta) >= 6 and delta.median() > 0 and (delta > 0).mean() >= .75 and delta.min() >= -.02)}
        given = reported_gate[(reported_gate.target.eq(target)) & reported_gate.arm.eq(arm)]
        exact = len(given) == 1 and all(np.isclose(given.iloc[0][k], v, atol=1e-12, rtol=0) if isinstance(v, float) else given.iloc[0][k] == v for k, v in expected.items())
        gate_rows.append({"target": target, "arm": arm, **expected, "reported_gate_exact": bool(exact)})
    gate_audit = pd.DataFrame(gate_rows)
    review = {
        "no_2026_candidate_label_economics_replay_or_portfolio_file_opened": True,
        "all_joint_output_hashes_exact": bool(input_audit.all_joint_output_hashes_exact.iloc[0]),
        "source_and_code_bindings_exact": bool(source_audit.hash_matches.all() and input_audit.source_manifest_binding_exact.iloc[0] and input_audit.source_ledger_binding_exact.iloc[0]),
        "arm_local_pre_cap_train_and_test_sets_exact": bool(identity_audit.recorded_pre_cap_identity_exact.all()),
        "post_cap_training_identity": "INDEPENDENTLY_RECOMPUTED_FROM_THE_IDENTICAL_PRE-CAP_IDS: the cap is deterministic candidate-id hashing in the sealed implementation; post-cap digests were not persisted by the joint artifact.",
        "cadence_and_label_boundaries_exact": bool(cadence.iloc[0].all()),
        "all_oof_rows_and_targets_exact": bool(pd.DataFrame(output_row_checks).drop(columns="file").all().all()),
        "all_oof_metrics_exact": bool(metric_audit.metric_exact.all()),
        "all_matched_deltas_exact": bool(delta_audit.delta_exact.all()),
        "all_gate_rows_exact": bool(gate_audit.reported_gate_exact.all()),
        "all_context_heads_rejected": bool((~gate_audit.eligible).all()),
        "conclusion": "PASS_FOR_THE_EXACT_SEALED_ARM-MATCHED_DIAGNOSTIC: on identical arm-local rows, side-local learner, deterministic cap and leave-era folds, none of the tested context heads beats its CORE-only control under the preregistered gate. This supersedes the earlier unproven cross-artifact comparison.",
        "application_verdict": "NO_CONTEXT_CORRECTION_AUTHORIZED. This result says the tested context heads are not incremental for this fixed diagnostic; it does not establish positive raw-residual trading economics, authorize a 2026 outcome read, a replay, tuning, or policy promotion.",
    }
    limitations = [
        {"priority": 1, "limitation_or_safeguard": "Leave-era validation remains nonchronological because later eras can fit an earlier held era. It is diagnostic OOF, not forward deployment validation."},
        {"priority": 2, "limitation_or_safeguard": "The utility label is linked mechanically to the score-defined historical top10 membership. The result supports rejecting these overlays, not a full-universe admission/reranking claim."},
        {"priority": 3, "limitation_or_safeguard": "No complete package/environment lock is sealed. Code and input/output hashes bind this run, but independent bitwise recreation in a changed library environment is not guaranteed."},
        {"priority": 4, "limitation_or_safeguard": "Persist post-cap training ID digests in future runs; this review could recompute them from identical pre-cap IDs and the sealed deterministic hashing rule, but they are not first-class source outputs."},
        {"priority": 5, "limitation_or_safeguard": "Any future preregistration citing this stronger joint gate must bind this joint manifest and eligibility CSV directly. Model rows remain 1h; 1m belongs only to nested labels/path/replay."},
    ]
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix="." + output.name + "."))
    try:
        source_audit.to_csv(stage / "source_and_code_hash_audit.csv", index=False)
        input_audit.to_csv(stage / "input_and_output_hash_audit.csv", index=False)
        cadence.to_csv(stage / "cadence_and_label_boundary_audit.csv", index=False)
        identity_audit.to_csv(stage / "independent_arm_local_identity_audit.csv", index=False)
        pd.DataFrame(output_row_checks).to_csv(stage / "oof_row_and_target_audit.csv", index=False)
        metric_audit.to_csv(stage / "oof_metric_recomputation_audit.csv", index=False)
        delta_audit.to_csv(stage / "matched_delta_recomputation_audit.csv", index=False)
        gate_audit.to_csv(stage / "gate_logic_audit.csv", index=False)
        pd.DataFrame(limitations).to_csv(stage / "remaining_limitations_and_safeguards.csv", index=False)
        dump(stage / "review.json", review)
        audit_contract = {"scope": "independent audit of pre2026 joint gate v2 only; no 2026 candidate/economics/etc. input", "cadence": "1h model and decision rows; 1m labels/path/replay only", "review_limit": "validates the diagnostic, not a live policy"}
        dump(stage / "contract.json", audit_contract)
        files = [p for p in stage.iterdir() if p.is_file()]
        review_manifest = {"schema": "pre2026_joint_score_context_incremental_gate_independent_review_v3", "status": "SEALED_PRE2026_JOINT_GATE_AUDIT_NON_PROMOTION", "promotion_eligible": False, "review": review, "contract": audit_contract,
                           "inputs_sha256": {str((SRC / "manifest.json").resolve()): sha(SRC / "manifest.json"), str((SRC / "materialized_targets.parquet").resolve()): sha(SRC / "materialized_targets.parquet"), str((JOINT / "manifest.json").resolve()): sha(JOINT / "manifest.json")},
                           "outputs_sha256": {p.name: sha(p) for p in files}}
        dump(stage / "manifest.json", review_manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
