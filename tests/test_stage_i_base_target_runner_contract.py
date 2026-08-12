from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_base_target_ablation import (
    BaseTargetAblationError,
    file_sha256,
    require_selected_feature_contract,
)
from extreme_price_movements.stage_i_r3_contract import r3_label_economics_contract
from scripts.run_stage_i_base_target_ablation import (
    _export_joint_target_finalists,
    _model_frame,
    _model_cell_source_fingerprint,
    _report,
    _request,
    _run_model_cell,
)
from extreme_price_movements.stage_i_target_promotion import decide_round3_promotion


def test_report_does_not_require_optional_tabulate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        pd.DataFrame, "to_markdown",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("no tabulate")),
    )
    path = _report(
        tmp_path, request={"request_sha256": "a" * 64},
        summary=pd.DataFrame({"arm": ["control"], "net": [1.0]}), stage="Round 2",
    )
    text = path.read_text()
    assert "```text" in text and "control" in text


def test_model_cell_fingerprint_excludes_presentation_code() -> None:
    value = _model_cell_source_fingerprint()
    assert value["schema"] == "stage_i_target_model_source_fingerprint_v2"
    assert value["presentation_code_excluded"] is True
    assert set(value["scientific_function_sha256"]) == {
        "_model_frame", "_prune_model_frame_to_selected_contract",
        "_prediction_metrics", "_population_audit", "_run_model_cell",
    }


def _selector(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    root = tmp_path / "selector"; root.mkdir()
    ts = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    ledger = pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "__ts__": ts, "__symbol__": ["X"] * 3,
        "side_name": ["long", "long", "short"], "decision_ts": ts + pd.Timedelta(hours=1),
        "label_available_ts": ts + pd.Timedelta(hours=13),
        "r3_class": [0, 1, 2], "r3_metric_target": [-1., .5, 1.],
        "exact_net_bps": [-301., -17., 209.],
        "t2_tp6_sl4_event": [1., 0., 0.], "robust_clear_event_b25": [0., 0., 1.],
        "robust_clear_soft_b25_t50": [0., .5, 1.],
        "label_valid": [True] * 3, "target_invalid": [False] * 3, "path_complete": [True] * 3,
    })
    features = ledger[["candidate_id", "__ts__", "__symbol__"]].copy(); features["f"] = [1., 2., 3.]
    ledger.to_parquet(root / "selector_ledger.parquet", index=False)
    features.to_parquet(root / "selector_features.parquet", index=False)
    contract = r3_label_economics_contract(ledger)
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete", "artifact_integrity": {
            "schema": "stage_i_selector_artifact_integrity_v1",
            "selector_ledger_sha256": file_sha256(root / "selector_ledger.parquet"),
            "selector_features_sha256": file_sha256(root / "selector_features.parquet"),
            "r3_label_economics_contract_sha256": contract["contract_sha256"],
        },
    }))
    (root / "selector_feature_contract.json").write_text(json.dumps({
        "max_feature_columns": 0, "feature_columns": ["f"],
    }))
    return root, ledger


def test_r3_control_uses_its_frozen_exact_net_not_new_clipped_geometry(tmp_path: Path) -> None:
    selector, ledger = _selector(tmp_path)
    labels = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    labels["geometry"] = "sl4_tp6"; labels["target_valid"] = True
    labels["causal_regime"] = "stable"; labels["contract_certainty"] = 1.
    labels["net_bps"] = 9999.  # must never become R3 control economics
    frame, target, family, population = _model_frame(
        labels, selector, None, regime_column="causal_regime"
    )
    assert target == "r3_class" and family == "R3_control"
    assert frame.net_bps.tolist() == ledger.exact_net_bps.tolist()
    assert len(population) == len(ledger) and population.label_valid.all()


def test_selected_contract_rejects_training_only_or_out_of_contract_feature(tmp_path: Path) -> None:
    selector, _ = _selector(tmp_path)
    selection = tmp_path / "selection"
    selector_manifest = json.loads((selector / "manifest.json").read_text())
    for side in ("long", "short"):
        root = selection / side; root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({
            "schema": "stage_i_base_feature_selection_v1", "status": "complete", "side": side,
            "selector_sample_manifest_sha256": file_sha256(selector / "manifest.json"),
            "selector_artifact_integrity": selector_manifest["artifact_integrity"],
            "selected_features": ["exact_net_bps"],
            "selected_feature_contract": ["exact_net_bps"],
            "input_feature_contract": ["exact_net_bps"], "best_params": {"num_leaves": 3},
        }))
    with pytest.raises(BaseTargetAblationError, match="lineage"):
        require_selected_feature_contract(selector_dir=selector, base_selection_dir=selection)


def test_v2_request_disables_legacy_multifold_runtime_and_records_speedup(tmp_path: Path) -> None:
    label_root = tmp_path / "labels"; label_root.mkdir()
    (label_root / "manifest.json").write_text("{}")
    args = SimpleNamespace(
        round=3, label_grid_dir=label_root, min_upper_support_rows=100,
        max_timeout_prevalence=.9, min_worst_regime_upper_rate=.005,
        min_oracle_top10_net_bps=0., evaluation_fraction=.25,
        development_seed=11, round2_per_family=3, min_train_rows=500,
        regime_column="causal_regime",
    )
    selected = {"contract_sha256": "a" * 64}
    labels = {"artifact_sha256": {"target_repair_labels.parquet": "b" * 64}}
    request = _request(args=args, selected=selected, label_manifest=labels)
    assert request["development_split"]["schema"] == "single_large_chronological_holdout_v1"
    assert "folds" not in request["round2"] and "folds" not in request["round3"]
    optimization = request["runtime_optimization"]
    assert optimization["legacy_default_nominal_booster_fits"] == 270
    assert optimization["single_holdout_nominal_booster_fits"] == 33
    assert optimization["nominal_booster_fit_reduction_factor"] == pytest.approx(270 / 33)
    assert optimization["parallel_target_workers"] == 3
    assert optimization["parallel_memory_budget_fraction"] == pytest.approx(.65)
    assert "Arrow IPC" in optimization["parallel_contract"]


def test_model_cell_publishes_single_holdout_reference_and_rejects_stale_request(tmp_path: Path) -> None:
    rng = np.random.default_rng(45)
    rows = 240
    decision = pd.date_range("2025-01-01", periods=rows // 2, freq="h", tz="UTC").repeat(2)
    signal = rng.normal(size=rows)
    frame = pd.DataFrame({
        "candidate_id": [f"x-{item}" for item in range(rows)],
        "__ts__": decision - pd.Timedelta(hours=1), "__symbol__": ["X"] * rows,
        "side_name": np.tile(["long", "short"], rows // 2),
        "decision_ts": decision, "label_available_ts": decision + pd.Timedelta(hours=12),
        "net_bps": signal * 50 + rng.normal(size=rows),
        "causal_regime": np.where(signal > 0, "up", "down"),
        "target": 1 / (1 + np.exp(-signal)), "f": signal,
    })
    selected = {
        "contract_sha256": "c" * 64,
        "sides": {
            side: {"selected_features": ["f"], "fixed_params": {
                "num_leaves": 7, "n_estimators": 10, "learning_rate": .05,
                "min_child_samples": 5, "max_bin": 31,
            }}
            for side in ("long", "short")
        },
    }
    population = frame[["candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts"]].copy()
    population["label_valid"] = True
    root = tmp_path / "cell"
    manifest, cache = _run_model_cell(
        root=root, frame=frame, arm=None, target_column="target", family="scalar_S",
        selected_contract=selected, development_seed=11, evaluation_fraction=.25,
        min_train_rows=30, weight_mode="uniform", regime_column="causal_regime",
        resume=False, experiment_input_sha256="d" * 64, population=population,
        model_cache=None,
    )
    assert cache is not None
    assert manifest["schema"] == "stage_i_base_target_model_cell_v2"
    assert manifest["strict_oof"] is False
    assert (root / "target_repair_development_predictions.parquet").is_file()
    assert (root / "development_mapping_reference_predictions.parquet").is_file()
    with pytest.raises(BaseTargetAblationError, match="request drift"):
        _run_model_cell(
            root=root, frame=frame, arm=None, target_column="target", family="scalar_S",
            selected_contract=selected, development_seed=12, evaluation_fraction=.25,
            min_train_rows=30, weight_mode="uniform", regime_column="causal_regime",
            resume=True, experiment_input_sha256="d" * 64, population=population,
            model_cache=cache,
        )


def test_joint_target_shortlist_preserves_r3_s_and_o_until_meta(tmp_path: Path) -> None:
    scorecard = pd.DataFrame([
        {"arm": "R3_frozen_control", "weight_mode": "uniform", "pooled_top10_net_bps": -10., "pooled_top1_net_bps": -5., "pooled_top5_net_bps": -8., "robust_top10_lift_score": -1., "worst_era_top10_net_bps": -20., "worst_side_top10_net_bps": -15., "worst_regime_top10_net_bps": -18., "latest_era_top10_net_bps": -12., "mapped_ev_monotonicity_violations": 1.},
        {"arm": "S__sl2_tp7", "weight_mode": "uniform", "pooled_top10_net_bps": -30., "pooled_top1_net_bps": -20., "pooled_top5_net_bps": -25., "robust_top10_lift_score": -4., "worst_era_top10_net_bps": -40., "worst_side_top10_net_bps": -35., "worst_regime_top10_net_bps": -38., "latest_era_top10_net_bps": -32., "mapped_ev_monotonicity_violations": 2.},
        {"arm": "O_a0p25__sl2_tp7", "weight_mode": "uniform", "pooled_top10_net_bps": -25., "pooled_top1_net_bps": -18., "pooled_top5_net_bps": -22., "robust_top10_lift_score": -3., "worst_era_top10_net_bps": -36., "worst_side_top10_net_bps": -31., "worst_regime_top10_net_bps": -34., "latest_era_top10_net_bps": -29., "mapped_ev_monotonicity_violations": 2.},
    ])
    scorecard_path = tmp_path / "scorecard.parquet"
    scorecard.to_parquet(scorecard_path, index=False)
    decision = decide_round3_promotion(
        scorecard, source_contract={"scorecard_sha256": file_sha256(scorecard_path)}
    )
    base = tmp_path / "base"
    for side in ("long", "short"):
        cell = base / side; cell.mkdir(parents=True)
        (cell / "manifest.json").write_text(json.dumps({"status": "complete", "side": side}))
        pd.DataFrame({"candidate_id": [side]}).to_parquet(cell / "selector_base_oof.parquet", index=False)
    bundles = []
    for family in ("scalar_S", "ordinal_O"):
        path = tmp_path / f"{family}.json"
        path.write_text(json.dumps({"status": "complete", "family": family}))
        bundles.append({
            "family": family, "manifest": str(path),
            "manifest_sha256": file_sha256(path), "bundle_sha256": family.ljust(64, "0")[:64],
        })
    result = _export_joint_target_finalists(
        output_dir=tmp_path / "run", decision=decision, winner_bundles=bundles,
        base_selection_dir=base, selected_contract={"contract_sha256": "a" * 64},
        label_manifest={"artifact_sha256": {"target_repair_labels.parquet": "b" * 64}},
        scorecard_path=scorecard_path,
    )
    assert result["base_only_economics_are_diagnostic"] is True
    assert {item["family"] for item in result["finalists"]} == {
        "R3_control", "scalar_S", "ordinal_O",
    }
    assert all(item["must_advance_to_joint_base_meta_evaluation"] for item in result["finalists"])
