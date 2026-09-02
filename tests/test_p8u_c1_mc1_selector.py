from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.inference.p8u_c1_mc1_inference_package import FEATURES, fit_package, save_package
from extreme_price_movements.inference.p8u_c1_mc1_selector import (
    CONFIG_SCHEMA,
    P8UC1MC1PackageSelector,
    RUN_SCHEMA,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(member.relative_to(path).as_posix().encode())
        digest.update(member.read_bytes())
    return digest.hexdigest()


def _panel(rows: int = 5_100) -> pd.DataFrame:
    rng = np.random.default_rng(1729)
    time = pd.Timestamp("2025-08-01T00:00:00Z") + pd.to_timedelta(np.arange(rows) % 192, unit="h")
    frame = pd.DataFrame({
        "candidate_id": [f"c1-{idx}" for idx in range(rows)], "__decision_ts__": time,
        "policy_path_valid": True, "policy_net_bps": rng.normal(80, 120, rows),
        "policy_label_available_ts": time + pd.Timedelta(hours=12),
        "final_score": rng.uniform(.1, .99, rows),
    })
    for field in FEATURES:
        if field not in frame:
            frame[field] = rng.normal(.5, .2, rows)
    return frame


def _write_package(root: Path, family: str) -> tuple[str, str]:
    package = fit_package(
        _panel(), family=family,
        train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
        train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
        train_months=6, source_hashes={}, policy_contract={},
    )
    state = pd.DataFrame({
        "decision_day": pd.date_range("2025-08-01", "2025-08-31", freq="D", tz="UTC"),
        "recent_shift_bps": 0.0,
        "max_policy_label_available_ts": pd.Timestamp("2025-07-31T12:00:00Z"),
    })
    path = root / f"{family}_2025-08"
    save_package(package, state, path)
    return str(path.relative_to(root)), _tree(path)


def test_c1_selector_loads_hash_bound_matching_dual_vintage(tmp_path: Path) -> None:
    package_root = tmp_path / "packages"
    package_root.mkdir()
    bcf_path, bcf_hash = _write_package(package_root, "bcf")
    current_path, current_hash = _write_package(package_root, "current")
    index = {
        "feature_order": list(FEATURES),
        "families": [
            {"family": "bcf", "month": "2025-08", "path": bcf_path, "sha256": bcf_hash},
            {"family": "current", "month": "2025-08", "path": current_path, "sha256": current_hash},
        ],
    }
    (package_root / "mc1_package_index.json").write_text(json.dumps(index))
    run = package_root / "run_manifest.json"
    run.write_text(json.dumps({"schema": RUN_SCHEMA}))
    config = tmp_path / "selector.json"
    config.write_text(json.dumps({
        "schema": CONFIG_SCHEMA, "status": "SEALED_NO_ORDER_C1_LVA_MAPPER",
        "package_root": str(package_root.relative_to(tmp_path)),
        "feature_contract": list(FEATURES), "training": {"train_months": 6},
        "admission": {"order_submission": False}, "run_manifest_sha256": _sha(run),
    }))
    selector = P8UC1MC1PackageSelector.load(config.relative_to(tmp_path), root=tmp_path)
    selected = selector.select(pd.Timestamp("2025-08-14T00:00:00Z"))
    assert selected.bcf.family == "bcf"
    assert selected.current.family == "current"


def test_c1_selector_rejects_wrong_feature_contract(tmp_path: Path) -> None:
    path = tmp_path / "selector.json"
    path.write_text(json.dumps({
        "schema": CONFIG_SCHEMA, "status": "SEALED_NO_ORDER_C1_LVA_MAPPER",
        "feature_contract": ["wrong"], "training": {"train_months": 6},
        "admission": {"order_submission": False}, "package_root": "packages",
    }))
    try:
        P8UC1MC1PackageSelector.load(path.relative_to(tmp_path), root=tmp_path)
    except ValueError as exc:
        assert "feature order" in str(exc)
    else:
        raise AssertionError("C1 selector accepted a wrong feature contract")


def test_c1_selector_accepts_builder_layout_with_adjacent_run_manifest(tmp_path: Path) -> None:
    """The actual builder writes the run receipt beside ``packages/``."""
    package_root = tmp_path / "run" / "packages"
    package_root.mkdir(parents=True)
    bcf_path, bcf_hash = _write_package(package_root, "bcf")
    current_path, current_hash = _write_package(package_root, "current")
    (package_root / "mc1_package_index.json").write_text(json.dumps({
        "feature_order": list(FEATURES),
        "families": [
            {"family": "bcf", "month": "2025-08", "path": bcf_path, "sha256": bcf_hash},
            {"family": "current", "month": "2025-08", "path": current_path, "sha256": current_hash},
        ],
    }))
    run = package_root.parent / "run_manifest.json"
    run.write_text(json.dumps({"schema": RUN_SCHEMA}))
    config = tmp_path / "selector.json"
    config.write_text(json.dumps({
        "schema": CONFIG_SCHEMA, "status": "SEALED_NO_ORDER_C1_LVA_MAPPER",
        "package_root": str(package_root.relative_to(tmp_path)),
        "feature_contract": list(FEATURES), "training": {"train_months": 6},
        "admission": {"order_submission": False}, "run_manifest_sha256": _sha(run),
    }))
    assert P8UC1MC1PackageSelector.load(config.relative_to(tmp_path), root=tmp_path).select(
        pd.Timestamp("2025-08-14T00:00:00Z")
    ).bcf.family == "bcf"
