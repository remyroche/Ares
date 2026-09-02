from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_mc1_inference_package import (
    FEATURES,
    P8UMC1InferencePackage,
    fit_package,
    save_package,
)
from extreme_price_movements.inference.p8u_mc1_selector import (
    P8UMC1PackageSelector,
    _package_tree_sha256,
)
from extreme_price_movements.inference.p8u_production_contract import P8UPreproductionBundle, sha256_tree


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_panel(rows: int = 5_100) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    timestamp = pd.Timestamp("2025-08-01T00:00:00Z") + pd.to_timedelta(np.arange(rows) % 192, unit="h")
    out = pd.DataFrame({
        "candidate_id": [f"X{i:05d}|long" for i in range(rows)],
        "__decision_ts__": timestamp,
        "policy_net_bps": rng.normal(100.0, 80.0, rows),
    })
    for number, field in enumerate(FEATURES):
        out[field] = rng.normal(.5 + number / 10.0, .2, rows)
    out["final_score"] = np.clip(out["final_score"], 0.0, 1.0)
    return out


def _package(panel: pd.DataFrame, *, family: str, month: str) -> P8UMC1InferencePackage:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    package = fit_package(
        panel, family=family,
        train_start=start - pd.DateOffset(months=6),
        train_end_exclusive=start,
        held_start=start,
        held_end_exclusive=start + pd.offsets.MonthBegin(1),
        train_months=6,
        source_hashes={"unit": "test"}, policy_contract={"unit": "test"},
    )
    return package


def _shift(month: str, family: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    days = pd.date_range(start, start + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1), freq="D", tz="UTC")
    return pd.DataFrame({
        "decision_day": days,
        "recent_shift_bps": np.zeros(len(days), dtype=float),
        "max_policy_label_available_ts": days - pd.Timedelta(hours=1),
        "window_start": days - pd.Timedelta(days=21),
        "window_end_exclusive": days,
        "family": family,
    })


def _selector_root(tmp_path: Path) -> tuple[Path, Path]:
    panel = _training_panel()
    package_root = tmp_path / "artifacts" / "mc1"
    entries = []
    for month in ("2026-02", "2026-03"):
        for family in ("bcf", "current"):
            destination = package_root / "mc1_packages" / f"family={family}" / f"month={month}"
            save_package(_package(panel, family=family, month=month), _shift(month, family), destination)
            entries.append({
                "family": family,
                "month": month,
                "path": str(destination.relative_to(package_root)),
                "sha256": _package_tree_sha256(destination),
                "is_latest": month == "2026-03",
            })
    run = {
        "schema": "strict_r3_p8u_dual_mc1_prequential_v2",
        "mc1": {"train_months": 6, "features": list(FEATURES)},
    }
    correctness = {
        "all_target_free_scores_persisted_before_policy_join": True,
        "mc1_maps_are_separate_by_family": True,
        "mc1_training_window_is_exactly_six_complete_calendar_months": True,
        "all_mc1_labels_are_resolved_before_held_month": True,
        "all_mc1_models_and_maps_are_serialized": True,
        "all_serialized_static_scores_match_persisted_predictions": True,
        "prior21_shift_uses_only_prior_resolved_labels": True,
        "no_live_or_exchange_mutation": True,
    }
    index = {
        "feature_order": list(FEATURES),
        "train_months": 6,
        "families": entries,
        "latest_package_by_family": {
            family: next(item for item in entries if item["family"] == family and item["month"] == "2026-03")
            for family in ("bcf", "current")
        },
    }
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "run_manifest.json").write_text(json.dumps(run))
    (package_root / "correctness_report.json").write_text(json.dumps(correctness))
    (package_root / "mc1_package_index.json").write_text(json.dumps(index))
    config = {
        "schema": "strict_r3_p8u_dual_mc1_sixmonth_inference_config_v1",
        "status": "RESEARCH_PACKAGE_NOT_LIVE",
        "package_root": str(package_root.relative_to(tmp_path)),
        "run_manifest_sha256": _sha(package_root / "run_manifest.json"),
        "correctness_report_sha256": _sha(package_root / "correctness_report.json"),
        "feature_contract": list(FEATURES),
        "training": {"train_months": 6},
        "admission": {"order_submission": False},
    }
    config_path = tmp_path / "config" / "selector.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    return config_path, package_root


def test_selector_uses_exact_monthly_vintage_and_config_hash(tmp_path: Path):
    config, _root = _selector_root(tmp_path)
    selector = P8UMC1PackageSelector.load(
        config, root=tmp_path, expected_config_sha256=_sha(config),
    )
    feb = selector.select("2026-02-15T12:00:00Z")
    march = selector.select("2026-03-31T12:00:00Z")
    assert feb.month == "2026-02"
    assert march.month == "2026-03"
    assert feb.bcf.package.family == "bcf"
    assert feb.current.package.feature_names == FEATURES
    with pytest.raises(ValueError, match="no exact"):
        selector.select("2026-04-01T00:00:00Z")
    with pytest.raises(ValueError, match="config hash mismatch"):
        P8UMC1PackageSelector.load(config, root=tmp_path, expected_config_sha256="wrong")


def test_selector_fails_closed_after_package_tree_tamper(tmp_path: Path):
    config, package_root = _selector_root(tmp_path)
    selector = P8UMC1PackageSelector.load(config, root=tmp_path)
    target = package_root / "mc1_packages/family=bcf/month=2026-02/band_curve.json"
    target.write_text("tampered")
    with pytest.raises(ValueError, match="package hash mismatch"):
        selector.select("2026-02-15T12:00:00Z")


def test_selector_config_can_be_carried_as_hash_bound_bundle_artifact(tmp_path: Path):
    config, package_root = _selector_root(tmp_path)
    artifacts = {
        "mc1_selector_config": {"path": str(config.relative_to(tmp_path)), "type": "file", "sha256": _sha(config)},
        "mc1_package_root": {"path": str(package_root.relative_to(tmp_path)), "type": "tree", "sha256": sha256_tree(package_root)},
    }
    # The parent bundle uses its own generic sealed-tree framing.  The
    # selector additionally validates the receipt-specific tree framing.
    for family, role in (("bcf", "bcf_mc1_model"), ("current", "current_mc1_model")):
        path = package_root / "mc1_packages" / f"family={family}" / "month=2026-03"
        artifacts[role] = {"path": str(path.relative_to(tmp_path)), "type": "tree", "sha256": sha256_tree(path)}
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({
        "schema": "strict_r3_p8u_preproduction_bundle_v1",
        "side": "long",
        "routing": {"fraction": .5},
        "runtime": {"order_submission": False, "promotion_status": "blocked_preproduction"},
        "artifacts": artifacts,
    }))
    bundle = P8UPreproductionBundle.load(bundle_path, root=tmp_path)
    selector = P8UMC1PackageSelector.from_preproduction_bundle(bundle)
    assert selector.select("2026-03-03T00:00:00Z").month == "2026-03"


def test_hash_bound_bundle_rejects_changed_complete_mc1_package_index(tmp_path: Path):
    config, package_root = _selector_root(tmp_path)
    artifacts = {
        "mc1_selector_config": {"path": str(config.relative_to(tmp_path)), "type": "file", "sha256": _sha(config)},
        "mc1_package_root": {"path": str(package_root.relative_to(tmp_path)), "type": "tree", "sha256": sha256_tree(package_root)},
    }
    for family, role in (("bcf", "bcf_mc1_model"), ("current", "current_mc1_model")):
        path = package_root / "mc1_packages" / f"family={family}" / "month=2026-03"
        artifacts[role] = {"path": str(path.relative_to(tmp_path)), "type": "tree", "sha256": sha256_tree(path)}
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({
        "schema": "strict_r3_p8u_preproduction_bundle_v1",
        "side": "long",
        "routing": {"fraction": .5},
        "runtime": {"order_submission": False, "promotion_status": "blocked_preproduction"},
        "artifacts": artifacts,
    }))
    # Change a non-latest index member.  The selected latest leaves still
    # hash-match, so this proves the complete package root is actually bound.
    index_path = package_root / "mc1_package_index.json"
    payload = json.loads(index_path.read_text())
    payload["families"][0]["month"] = "tampered"
    index_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        P8UMC1PackageSelector.from_preproduction_bundle(
            P8UPreproductionBundle.load(bundle_path, root=tmp_path)
        )
