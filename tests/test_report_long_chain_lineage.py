import json
import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_long_chain_lineage.py"
_SPEC = importlib.util.spec_from_file_location("report_long_chain_lineage", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
audit_long_chain = _MODULE.audit_long_chain
write_report = _MODULE.write_report
is_path_value = _MODULE._is_path_value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _model_manifest(directory: Path) -> None:
    for name in ("labels.parquet", "features.csv", "params.json", "ae_gmm.joblib"):
        (directory / name).write_text(name, encoding="utf-8")
    _write_json(
        directory / "manifest.json",
        {
            "training_window": {"train": "2026-01-01/2026-03-31", "validation": "2026-04"},
            "oos_scope": "2026-05",
            "target_modes": ["net_return_long"],
            "labels_path": "labels.parquet",
            "feature_selection_recipe": "largest-authorized-fold",
            "feature_count": 2,
            "selected_features": ["alpha", "beta"],
            "hpo_scope": "training-validation-only",
            "n_trials_requested": 12,
            "fixed_params_json": "params.json",
            "ae_gmm_state_path": "ae_gmm.joblib",
            "cost_contract": {"round_trip_bps": 20},
            "final_refit_exclusion": "reported OOS excluded from final refit",
            "leakage_status": "pass",
        },
    )


def _policy_manifest(directory: Path, *, replay: bool = False) -> None:
    _write_json(
        directory / "manifest.json",
        {
            "training_window": "2026-01 through 2026-04",
            "oos_scope": "2026-05",
            "target": "net_return_long",
            "cost_contract": "20bps round trip",
            "final_refit_exclusion": "OOS excluded",
            "side_archetype_expected_ev_policy": {"fixed_target_net_ev": 0.007},
            "geometry": {"long_only": True} if replay else {"top_fraction": 0.10},
            "leakage_status": "pass",
        },
    )


def test_audit_writes_five_evidence_backed_rows(tmp_path: Path) -> None:
    base, meta, policy, replay = (tmp_path / name for name in ("base", "meta", "policy", "replay"))
    for directory in (base, meta):
        directory.mkdir()
        _model_manifest(directory)
    policy.mkdir()
    replay.mkdir()
    _policy_manifest(policy)
    _policy_manifest(replay, replay=True)

    report = audit_long_chain(base, meta, policy, replay)

    assert [row["layer"] for row in report["rows"]] == [
        "base", "meta_long_residual_expert", "side_archetype_ev_map", "simple_policy_geometry", "portfolio_policy",
    ]
    assert all(row["status"] == "PASS" for row in report["rows"])
    assert report["rows"][0]["feature_hash"]["value"]
    assert report["rows"][0]["ae_gmm_state"]["artifact"]["sha256"]

    outputs = write_report(report, tmp_path / "out")
    assert set(outputs) == {"json", "csv", "markdown"}
    assert json.loads(outputs["json"].read_text(encoding="utf-8"))["rows"][0]["status"] == "PASS"
    assert "meta_long_residual_expert" in outputs["markdown"].read_text(encoding="utf-8")


def test_missing_referenced_state_is_failed_not_passed(tmp_path: Path) -> None:
    base, meta, policy = (tmp_path / name for name in ("base", "meta", "policy"))
    for directory in (base, meta):
        directory.mkdir()
        _model_manifest(directory)
    (base / "ae_gmm.joblib").unlink()
    policy.mkdir()
    _policy_manifest(policy)

    report = audit_long_chain(base, meta, policy)
    base_row = report["rows"][0]
    portfolio_row = report["rows"][-1]

    assert base_row["status"] == "FAILED_VALIDATION"
    assert any(item["declared_path"] == "ae_gmm.joblib" for item in base_row["invalid_paths"])
    assert base_row["leakage_status"] == "PASS"
    assert portfolio_row["status"] == "MISSING_EVIDENCE"
    assert "authoritative_manifest" in portfolio_row["missing_evidence"]


def test_only_explicit_filesystem_fields_are_treated_as_paths() -> None:
    assert is_path_value("labels_path", "labels.parquet")
    assert is_path_value("model_bundle_path", "fold.joblib")
    assert is_path_value("manifest", "manifest.json")
    assert not is_path_value("timeframe", "15m")
    assert not is_path_value("train_end", "2026-03-31T23:00:00+00:00")
    assert not is_path_value("artifact_hash", "a" * 64)
    assert not is_path_value("model", "LGBMRegressor")
    assert not is_path_value("source_files", "train_global_long_5_2026_06.parquet")
