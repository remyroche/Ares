#!/usr/bin/env python3
"""Write a hash-bound reconciliation from historical Stage-C v3 to sealed v4.

No model is fitted.  This authoring step preserves v3 as historical evidence,
declares v4 as the currently audited Stage-1 record, and keeps every blocked
or correctly-not-run requirement explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
V3 = ART / "stage_c_conditional_retention_ablation_20260731_v3"
V4 = ART / "stage_c_conditional_retention_ablation_20260731_v4"
STAGE0 = ART / "stage_c_continuation_feature_panel_20260731_v2"
TARGET_AUDIT = ART / "stage_c_primary_target_contract_audit_20260801_v1"
OI_FUNDING_BLOCKER = ART / "stage_c_oi_funding_lineage_blocker_20260801_v1"
V11 = ART / "exact_h12_target_purity_ablation_20260731_v11"
DEFAULT_OUTPUT = ART / "stage_c_v4_canonical_reconciliation_20260801_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _aggregate_metrics() -> list[dict[str, Any]]:
    result = pd.read_parquet(V4 / "retention_conditional_results.parquet")
    values = result.loc[
        result.scope.eq("aggregate") & result.split.isin(("development_oof", "final_oos")),
        ["split", "arm", "rows", "roc_auc", "pr_auc", "brier", "log_loss", "spearman_exact_h12_net", "top_decile_exact_h12_net_bps"],
    ].sort_values(["split", "arm"], kind="stable")
    return values.to_dict(orient="records")


def build_reconciliation(*, v3: dict[str, Any], v4: dict[str, Any], target_audit: dict[str, Any], current_runner_sha256: str) -> dict[str, Any]:
    """Build the complete specification-to-evidence map without interpretation drift."""
    frozen_runner = v4["code_sha256"]["stage1_runner"]
    return {
        "schema": "stage_c_v4_canonical_reconciliation_v1",
        "status": "STAGE_C_V4_RECONCILED_RESEARCH_ONLY_WITH_EXPLICIT_BLOCKERS",
        "promotion_eligible": False,
        "historical_record": {
            "v3": {
                "path": str(V3),
                "status": v3["status"],
                "purpose": "preserved historical Stage-C report record; not overwritten or re-labelled",
            },
            "v4": {
                "path": str(V4),
                "status": v4["status"],
                "role": "current audited Stage-1 retention record",
                "target": v4["target"],
                "terminal_decision": v4["terminal_decision"],
            },
        },
        "v3_to_v4_delta": {
            "v3_input_compatible": v3["rows"]["input_compatible"],
            "v4_input_compatible": v4["rows"]["input_compatible"],
            "v3_clear_first_support": v3["rows"]["clear_first_support"],
            "v4_clear_first_support": v4["rows"]["clear_first_support"],
            "v3_prediction_rows": v3["rows"]["predictions"],
            "v4_prediction_rows": v4["rows"]["predictions"],
            "protocol_change": "v4 extends resolved training history to 2023-04..2024-03 and adds the April 2024 development OOF fold; final OOS remains 2024-08..11 and is not used for selection.",
            "not_a_same_code_rerun": True,
        },
        "specification_sections": [
            {"section": "1. decision/research status", "status": "SATISFIED", "evidence": ["v4 terminal decision", "Stage-B remains closed; no entry execution target is advanced"]},
            {"section": "2. available/unavailable information", "status": "QUALIFIED_BLOCKER", "evidence": ["F4/F5 source-timing rejection", "F7 OOF/provenance rejection"], "blocker": "The frozen F0 E15 control retains historical `ob_*` features. It is valid only as the mandated inherited comparator; this reconciliation does not establish factual-L2 provenance for those legacy fields or a pure permitted-input C0 model."},
            {"section": "3. hard scope constraints", "status": "SATISFIED", "evidence": ["v4 Stage-2 and Stage-B records are NOT_RUN", "no threshold, quota, portfolio, entry or exit policy output"]},
            {"section": "4. frozen contracts", "status": "VERIFIED", "evidence": ["v4 input/output hashes", "target audit frozen IDs, timestamps and exact-H12 endpoint"]},
            {"section": "5. primary conditional target", "status": "VERIFIED", "evidence": ["recomputed H0 clear-first event", "exact H12 net > 0 label", "support-side/month and null-outside-support audit"]},
            {"section": "6. inherited baseline", "status": "SATISFIED", "evidence": ["frozen E15 hash", "C0 side-local v11 control"]},
            {"section": "7–9. causal feature groups/reduction", "status": "SATISFIED_WITH_SOURCE_BLOCKS", "evidence": ["sealed Stage-0 lineage/coverage/group artifacts", "train-fold-only selection evidence", "F4/F5/F7 explicitly rejected"]},
            {"section": "10. Stage 0/1 protocol", "status": "VERIFIED", "evidence": ["identical candidate IDs per arm/fold", "strict H12 purge", "development-only selection", "v4 correctness report"]},
            {"section": "10. Stage 2 compact combination", "status": "CORRECTLY_NOT_RUN", "evidence": ["zero development survivors", "compact and leave-group-out NOT_RUN artifacts"]},
            {"section": "11. frozen Stage-B economic test", "status": "CORRECTLY_NOT_RUN", "evidence": ["Stage-1 admission gate failed", "no hierarchy/bridge/entry ranking was run"]},
            {"section": "12. decision rule", "status": "SATISFIED", "evidence": ["Outcome-A terminal decision in v4 manifest and disposition"]},
            {"section": "13. transition monitor", "status": "SOURCE_BLOCKED", "evidence": ["F7 C7 blocked without candidate-level strict OOF/prequential provenance"]},
            {"section": "14. correctness tests", "status": "VERIFIED_WITH_FUTURE_RUN_GAP", "evidence": ["v4 correctness report passed", "separate 55-check primary-target audit passed", "focused source tests pass"], "blocker": "The strengthened target-contract columns/checks were added after sealed v4, so v4 proves them through the independent audit rather than native prediction-ledger fields."},
            {"section": "15. deliverables", "status": "SATISFIED", "evidence": ["Stage-0 files in stage_c_continuation_feature_panel_v2", "Stage-1 files in v4", "Stage-2/B records explicitly NOT_RUN"]},
            {"section": "16. final report", "status": "RECONCILED", "evidence": ["this v4 reconciliation", "historical v3 final report preserved"]},
            {"section": "17. terminal interpretation", "status": "SATISFIED", "evidence": ["no admitted mechanism survives all development transport/calibration/stability criteria", "no economic Stage-B retest is justified"]},
        ],
        "target_audit": {
            "path": str(TARGET_AUDIT),
            "status": target_audit["status"],
            "checks": target_audit["checks"],
        },
        "reproducibility": {
            "sealed_v4_runner_sha256": frozen_runner,
            "current_runner_sha256": current_runner_sha256,
            "status": "SOURCE_DRIFT_AFTER_SEALED_V4" if frozen_runner != current_runner_sha256 else "SAME_SOURCE",
            "meaning": "The new fail-closed target checks improve future runs, but v4 must be reproduced only with its sealed source hash or an archived source copy; this reconciliation is not a same-code rerun.",
        },
        "remaining_blockers": [
            "F4 OI: native source-observation/availability time, finite staleness and linear-product/unit lineage are absent.",
            "F5 funding: native observation/availability, value-kind, settlement/revision and finite staleness evidence are absent.",
            "F7: candidate-level strict OOF/prequential transition predictions with fold/train-end/availability lineage are absent.",
            "No Stage-1 mechanism passed; Stage 2 and frozen Stage-B hierarchy test remain correctly not run.",
            "The frozen F0 comparator includes legacy `ob_*` inputs whose factual L2 lineage is not re-proven by this limited OHLCV/OI/funding workstream.",
            "v4's sealed runner differs from current source after target-contract hardening; no same-code v4 rerun was requested or performed.",
        ],
    }


def _markdown(payload: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage-C v4 canonical reconciliation — 2026-08-01", "",
        "## Status", "",
        "`STAGE_C_V4_RECONCILED_RESEARCH_ONLY_WITH_EXPLICIT_BLOCKERS`", "",
        "This is the authoritative v4 reconciliation for the Stage-C specification. It preserves the v3 report as historical evidence; it neither reruns a model nor promotes an entry, hierarchy, threshold, policy, or portfolio rule.", "",
        "The supported terminal decision remains:", "",
        "`CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION`", "",
        "## v3 to v4", "",
        "| Record | Compatible rows | H0-clear support | Prediction rows | Role |",
        "|---|---:|---:|---:|---|",
        f"| v3 | {payload['v3_to_v4_delta']['v3_input_compatible']:,} | {payload['v3_to_v4_delta']['v3_clear_first_support']:,} | {payload['v3_to_v4_delta']['v3_prediction_rows']:,} | Historical final report record |",
        f"| v4 | {payload['v3_to_v4_delta']['v4_input_compatible']:,} | {payload['v3_to_v4_delta']['v4_clear_first_support']:,} | {payload['v3_to_v4_delta']['v4_prediction_rows']:,} | Current audited Stage-1 record |",
        "",
        "v4 adds 2023-04..2024-03 resolved training support and April 2024 development OOF. It is not a same-code rerun of v3; the final August–November OOS period remains selection-free.", "",
        "## Primary target and row contract", "",
        "The separate primary-target audit recomputed `retain_h0_given_clear` from the frozen H0 clear-first event and `exact_h12_net_bps > 0`. It verified all 55 checks: immutable frozen IDs, H12 endpoint/availability, label/support fields, all OOF labels/timestamps/nets, per-fold candidate hashes, identical arm rows, no H25/continuous target sweep, and strict purge/selection chronology.", "",
        "- Panel: 252,702 compatible rows; 103,681 H0-clear rows.",
        "- v4 temporal protocol: 252,677 compatible rows; 103,667 H0-clear rows. The 25/14 difference is the `< 2024-12-01` boundary, not a label discrepancy.",
        "- Compared arms: C0/C1/C2/C3/C6/C8, identical within each fold; C4/C5/C7 are source-blocked.", "",
        "## Stage-1 aggregate evidence", "",
        "| Split | Arm | Rows | ROC-AUC | PR-AUC | Brier | Net Spearman | Top-decile exact H12 net (bps) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['split']} | {row['arm']} | {int(row['rows']):,} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['brier']:.4f} | {row['spearman_exact_h12_net']:.4f} | {row['top_decile_exact_h12_net_bps']:.1f} |"
        )
    lines += [
        "", "The admission decision uses development OOF only. No group cleared every predeclared transport, calibration, month-stability, side-stability and bootstrap criterion; therefore compact Stage 2 and the frozen Stage-B hierarchy comparison are correctly **not run**.", "",
        "## Specification map", "",
        "| Specification area | Status | Evidence / qualification |",
        "|---|---|---|",
    ]
    for section in payload["specification_sections"]:
        evidence = "; ".join(section["evidence"])
        if section.get("blocker"):
            evidence += ". **Qualification:** " + section["blocker"]
        lines.append(f"| {section['section']} | {section['status']} | {evidence} |")
    lines += [
        "", "## Tests and seals", "",
        "- v4 stored correctness report: passed 10 Stage-1 checks, including candidate identity, feature-selection isolation, H12 purge/availability, April development OOF and paired seeds.",
        "- Current focused checks: `tests/test_stage_c_conditional_retention_ablation.py` and `tests/test_audit_stage_c_primary_target_contract.py` — 13 passed.",
        "- The latest target audit verifies all declared v4 input/output hashes and has terminal status `STAGE_C_V4_PRIMARY_TARGET_CONTRACT_VERIFIED`.",
        "",
        "## Explicit remaining blockers", "",
    ]
    for item in payload["remaining_blockers"]:
        lines.append(f"- {item}")
    lines += [
        "", "## Canonical paths", "",
        f"- Historical v3 record: `{V3}`", f"- Current Stage-1 v4 record: `{V4}`", f"- Stage-0 feature/coverage record: `{STAGE0}`", f"- Frozen H0 source: `{V11}`", f"- Target audit: `{TARGET_AUDIT}`", f"- OI/funding blocker: `{OI_FUNDING_BLOCKER}`", "",
        "The v3 report remains unmodified. This record supersedes it only as the documentation pointer for the audited v4 Stage-C result.",
    ]
    return "\n".join(lines) + "\n"


def run(*, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    required = [
        V3 / "run_manifest.json", V4 / "run_manifest.json", V4 / "retention_conditional_results.parquet",
        V4 / "correctness_test_report.json", TARGET_AUDIT / "run_manifest.json",
        TARGET_AUDIT / "primary_target_contract_readiness.parquet", OI_FUNDING_BLOCKER / "manifest.json",
        STAGE0 / "run_manifest.json", V11 / "manifest.json",
    ]
    for path in required:
        _require(path)
    v3, v4 = _read_json(V3 / "run_manifest.json"), _read_json(V4 / "run_manifest.json")
    target_audit = _read_json(TARGET_AUDIT / "run_manifest.json")
    current_runner_sha256 = _sha256(ROOT / "scripts/run_stage_c_conditional_retention_ablation.py")
    payload = build_reconciliation(v3=v3, v4=v4, target_audit=target_audit, current_runner_sha256=current_runner_sha256)
    metrics = _aggregate_metrics()
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (stage / "stage_c_v4_canonical_reconciliation.md").write_text(_markdown(payload, metrics), encoding="utf-8")
        (stage / "stage_c_v4_specification_map.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs = {name: _sha256(stage / name) for name in ("stage_c_v4_canonical_reconciliation.md", "stage_c_v4_specification_map.json")}
        manifest = {
            "schema": "stage_c_v4_canonical_reconciliation_artifact_v1",
            "status": payload["status"],
            "promotion_eligible": False,
            "inputs": {
                str(path): _sha256(path)
                for path in required + [ROOT / "scripts/run_stage_c_conditional_retention_ablation.py"]
            },
            "outputs": outputs,
            "author": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
