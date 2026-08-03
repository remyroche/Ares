"""Synthetic evidence tests for deterministic final diagnostic authoring."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from scripts.assemble_root_cause_final_pack import PackContractError, sha256
from scripts.author_root_cause_diagnostic import author


REPO = Path(__file__).resolve().parents[1]


def _replace_parquet(directory: Path, manifest_path: Path, manifest_key: str, name: str, frame: pd.DataFrame) -> None:
    path = directory / name
    frame.to_parquet(path)
    manifest = json.loads(manifest_path.read_text())
    manifest[manifest_key][name] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _evidence(tmp_path: Path) -> dict[str, Path]:
    helper = runpy.run_path(str(REPO / "tests/test_assemble_root_cause_final_pack.py"))
    sources = helper["_sources"](tmp_path)
    fractions = (.01, .05, .10, .20)
    rows = []
    for oracle, gross, net in (
        ("O1_realised_gross_h12", 5.0, -2.0),
        ("O2_realised_net_h12", 5.0, -2.0),
        ("CURRENT_base_plus_residual_delta_OOF", -4.0, -6.0),
    ):
        for fraction in fractions:
            rows.append({
                "oracle": oracle, "top_fraction": fraction, "slice_kind": "pooled", "slice_value": "ALL",
                "mean_evaluation_gross_bps": gross, "mean_net_bps": net, "net_status": "AVAILABLE",
            })
    stage1_manifest = sources["pointer"].parent / "stage1" / "run_manifest.json"
    _replace_parquet(sources["pointer"].parent / "stage1", stage1_manifest, "outputs_sha256", "oracle_ladder_results.parquet", pd.DataFrame(rows))
    regret = pd.DataFrame([{"oracle": "O1_realised_gross_h12", "top_fraction": .1, "entry_regret_bps": 9.0, "oracle_net_status": "AVAILABLE"}])
    (sources["pointer"].parent / "stage1" / "oracle_regret_vs_current_oof.parquet").write_bytes(b"placeholder")
    manifest = json.loads(stage1_manifest.read_text())
    manifest["outputs_sha256"]["oracle_regret_vs_current_oof.parquet"] = sha256(sources["pointer"].parent / "stage1" / "oracle_regret_vs_current_oof.parquet")
    # Replace placeholder with a real parquet after recording the named output.
    regret.to_parquet(sources["pointer"].parent / "stage1" / "oracle_regret_vs_current_oof.parquet")
    manifest["outputs_sha256"]["oracle_regret_vs_current_oof.parquet"] = sha256(sources["pointer"].parent / "stage1" / "oracle_regret_vs_current_oof.parquet")
    stage1_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pointer = json.loads(sources["pointer"].read_text())
    pointer["stage1_manifest_sha256"] = sha256(stage1_manifest)
    sources["pointer"].write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")

    stage2_manifest = sources["stage2"] / "run_manifest.json"
    _replace_parquet(
        sources["stage2"], stage2_manifest, "outputs_sha256", "feature_information_results.parquet",
        pd.DataFrame([{"feature_name": "causal_x", "transported_ic_mean": .02, "top_bottom_decile_spread_mean_bps": 4.0}]),
    )
    stage3_manifest = sources["stage3"] / "run_manifest.json"
    _replace_parquet(
        sources["stage3"], stage3_manifest, "outputs_sha256", "model_learning_efficiency.parquet",
        pd.DataFrame([
            {
                "model_family": "prior", "seed": 1, "split": "later_oos",
                "evaluation_scope": "outer_heldout", "side": "long", "component": "base_directional",
                "base_directional__spearman_ic": .12, "base_directional__roc_auc": .55,
                "base_directional__pr_auc": .20, "base_directional__log_loss": .68,
                "base_directional__brier": .24, "base_directional__ece": .04,
                "base_directional__mae": .10,
            },
            {
                "model_family": "prior", "seed": 1, "split": "later_oos",
                "evaluation_scope": "outer_heldout", "side": "long", "component": "residual_economic",
                "residual_economic__spearman_ic": .08, "residual_economic__mae_bps": 100.0,
                "residual_economic__huber_bps": 90.0, "residual_economic__gross_mean_bps": -5.0,
                "residual_economic__net_mean_bps": -105.0, "residual_economic__gross_top10_bps": 2.0,
                "residual_economic__net_top10_bps": -98.0, "residual_economic__gross_top20_bps": -5.0,
                "residual_economic__net_top20_bps": -105.0,
            },
        ]),
    )
    _replace_parquet(sources["stage3"], stage3_manifest, "outputs_sha256", "metric_concordance.parquet", pd.DataFrame([{"x": 1}]))

    execution = pd.DataFrame([
        {"record_type": "waterfall", "stage": "B_executable_entry_gross", "slice": "full_population", "score": "none", "rows": 100, "value_bps_per_candidate": -10.0, "status": "OBSERVED", "detail": "fixture"},
        {"record_type": "waterfall", "stage": "cost_drag_D_minus_E", "slice": "full_population", "score": "none", "rows": 100, "value_bps_per_candidate": 7.0, "status": "IDENTIFIED", "detail": "fixture"},
        {"record_type": "waterfall", "stage": "entry_transfer_loss_A_minus_B", "slice": "full_population", "score": "none", "rows": 100, "value_bps_per_candidate": None, "status": "NOT_IDENTIFIABLE", "detail": "unavailable"},
    ])
    stage56_manifest = sources["stage56"] / "run_manifest.json"
    _replace_parquet(sources["stage56"], stage56_manifest, "outputs", "execution_waterfall.parquet", execution)
    policy = pd.DataFrame([{
        "population": "complete_upstream_population", "policy": "learned_action_overlay", "rows": 100,
        "gross_bps_per_candidate": None, "oracle_regret_bps_per_candidate": None, "status": "NOT_RUN_TWO_HEAD_SCOPE", "detail": "disabled",
    }])
    _replace_parquet(sources["stage56"], stage56_manifest, "outputs", "policy_regret.parquet", policy)

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    global_files = {
        "global_topk_learning_economics.parquet": pd.DataFrame([{
            "record_type": "global_topk_arm", "model_family": "production_like_lgbm", "seed": 1,
            "top_fraction": .1, "net_bps": -4.0, "selection_scope": "GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE",
        }]),
        "global_topk_learning_gaps.parquet": pd.DataFrame([
            {"record_type": "named_global_gap", "comparison": "causal_to_future", "net_gap_bps": 3.0, "gross_gap_bps": 3.0, "selection_scope": "GLOBAL_TOP_10"},
            {"record_type": "named_global_gap", "comparison": "production_to_causal", "net_gap_bps": 2.0, "gross_gap_bps": 2.0, "selection_scope": "GLOBAL_TOP_10"},
        ]),
        "causal_only_global_metric_concordance.parquet": pd.DataFrame([{
            "development_base_metric": "base_directional__spearman_ic", "later_global_economic_metric": "net_bps",
            "arms": 3, "spearman": -.5,
        }]),
    }
    for name, frame in global_files.items():
        frame.to_parquet(global_dir / name)
    global_manifest = {
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "selection_scope": "GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE",
        "stage3_manifest_sha256": sha256(stage3_manifest),
        "runner": {"path": "scripts/materialize_root_cause_global_learning_economics.py", "sha256": sha256(REPO / "scripts/materialize_root_cause_global_learning_economics.py")},
        "outputs_sha256": {name: sha256(global_dir / name) for name in global_files},
    }
    (global_dir / "run_manifest.json").write_text(json.dumps(global_manifest, indent=2, sort_keys=True) + "\n")
    sources["global"] = global_dir
    return sources


def _author(sources: dict[str, Path], output: Path) -> dict:
    return author(
        pointer_path=sources["pointer"], stage2_dir=sources["stage2"], stage3_dir=sources["stage3"],
        stage56_dir=sources["stage56"], global_learning_dir=sources["global"], output=output,
    )


def test_author_applies_explicit_rules_and_keeps_unproven_classes_unresolved(tmp_path: Path) -> None:
    sources = _evidence(tmp_path)
    result = _author(sources, tmp_path / "author")
    assert result["status"] == "COMPLETE_EVIDENCE_DRIVEN_DIAGNOSTIC_ONLY_NO_PROMOTION"
    classes = pd.read_parquet(tmp_path / "author/classification_evidence.parquet").set_index("classification")
    assert classes.loc["COST_DRAG_FAILURE", "status"] == "SUPPORTED"
    assert classes.loc["CAUSAL_FEATURE_INFORMATION_INSUFFICIENT", "status"] == "SUPPORTED"
    assert classes.loc["ML_LEARNING_EFFICIENCY_FAILURE", "status"] == "SUPPORTED"
    assert classes.loc["METRIC_SELECTION_MISALIGNMENT", "status"] == "SUPPORTED"
    assert classes.loc["TARGET_OR_POPULATION_FAILURE", "status"] == "UNRESOLVED"
    assert classes.loc["EXECUTION_TRANSFER_FAILURE", "status"] == "UNRESOLVED"
    assert classes.loc["POLICY_CONVERSION_FAILURE", "status"] == "UNRESOLVED"
    report = (tmp_path / "author/ROOT_CAUSE_DIAGNOSTIC_REPORT.md").read_text()
    assert "COST_DRAG_FAILURE | SUPPORTED" in report
    assert "Directional base-head metrics (later OOS)" in report
    assert "Stopped-gradient residual-head metrics (later OOS)" in report


def test_author_fails_closed_on_invalid_global_or_two_head_scope(tmp_path: Path) -> None:
    sources = _evidence(tmp_path)
    global_manifest_path = sources["global"] / "run_manifest.json"
    global_manifest = json.loads(global_manifest_path.read_text())
    global_manifest["selection_scope"] = "PER_TIMESTAMP"
    global_manifest_path.write_text(json.dumps(global_manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(PackContractError, match="global top-k"):
        _author(sources, tmp_path / "bad-global")
    assert not (tmp_path / "bad-global").exists()

    sources = _evidence(tmp_path / "scope")
    stage56_manifest_path = sources["stage56"] / "run_manifest.json"
    stage56_manifest = json.loads(stage56_manifest_path.read_text())
    stage56_manifest["architecture"].append("timing_head")
    stage56_manifest_path.write_text(json.dumps(stage56_manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(PackContractError, match="two-head"):
        _author(sources, tmp_path / "bad-scope")
    assert not (tmp_path / "bad-scope").exists()
