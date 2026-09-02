#!/usr/bin/env python3
"""Audit an immutable P8U six-month dual-MC1 inference-package receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_mc1_inference_package import (  # noqa: E402
    FEATURES,
    apply_shift,
    load_package,
)


SCHEMA = "strict_r3_p8u_dual_mc1_prequential_v2"
POLICY_FORBIDDEN = frozenset({
    "policy_path_valid", "policy_net_bps", "policy_gross_bps", "policy_exit_bar_15m",
    "policy_exit_price", "policy_entry_price", "policy_label_available_ts", "policy_exit_reason",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*")) if path.is_dir() else [path]
    for member in members:
        if not member.is_file():
            continue
        digest.update(str(member.relative_to(path) if path.is_dir() else member.name).encode("utf-8"))
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    index = json.loads((root / "mc1_package_index.json").read_text())
    failures: list[str] = []
    checks: dict[str, object] = {}
    _require(manifest.get("schema") == SCHEMA, "unexpected run-manifest schema", failures)
    mc1 = manifest.get("mc1", {})
    _require(int(mc1.get("train_months", -1)) == 6, "MC1 train_months is not six", failures)
    _require(tuple(mc1.get("features", ())) == FEATURES, "MC1 feature order differs from frozen six-field contract", failures)
    # Every target-free panel must remain outcome-free even after package build.
    target_free_ok = True
    panel_count = 0
    for path in sorted((root / "target_free_scores").rglob("*.parquet")):
        panel_count += 1
        target_free_ok &= not bool(POLICY_FORBIDDEN.intersection(pq.ParquetFile(path).schema_arrow.names))
    _require(panel_count > 0 and target_free_ok, "target-free score panels contain policy/outcome columns", failures)
    checks["target_free_panel_count"] = panel_count
    # Rehash every declared original input and code member.
    source_hashes = manifest.get("source_hashes", {})
    source_paths: dict[str, Path] = {
        "base_target_free_scores": Path(manifest["base_root"]),
        "policy_labels": Path(manifest["policy"]),
        "runner_source": ROOT / "scripts/run_strict_r3_p8u_dual_mc1_prequential_v2.py",
        "mc1_package_source": ROOT / "extreme_price_movements/inference/p8u_mc1_inference_package.py",
    }
    for index_meta, meta in enumerate(manifest.get("metas", ())):
        source_paths[f"meta_{index_meta}_{meta['arm']}_target_free_scores"] = Path(meta["root"])
    source_ok = True
    for name, path in source_paths.items():
        actual = _sha256(path)
        expected = source_hashes.get(name)
        source_ok &= actual == expected
        if actual != expected:
            failures.append(f"source hash mismatch for {name}: expected={expected}, actual={actual}")
    checks["source_hashes_match"] = source_ok
    # Verify fit temporal boundaries, serialization exactness, and shift state.
    total_packages = 0
    max_static_delta = 0.0
    max_shift_delta = 0.0
    for family in ("bcf", "current"):
        audit_frame = pd.read_parquet(root / f"{family}_mc1_fit_audit.parquet")
        prediction = pd.read_parquet(root / f"enhanced_{family}_mc1_predictions.parquet")
        _require(len(audit_frame) == 7, f"{family}: expected seven Feb-Aug six-month vintages", failures)
        for row in audit_frame.itertuples(index=False):
            held_start = pd.Timestamp(f"{row.month}-01", tz="UTC")
            _require(int(row.train_months) == 6, f"{family} {row.month}: non-six training month audit", failures)
            _require(pd.Timestamp(row.train_start) == held_start - pd.DateOffset(months=6), f"{family} {row.month}: train start is not held-6 months", failures)
            _require(pd.Timestamp(row.train_end_exclusive) == held_start, f"{family} {row.month}: train end is not held start", failures)
            _require(bool(row.shift_max_label_available_ts_lt_decision_day), f"{family} {row.month}: shift used not-yet-resolved labels", failures)
            package_path = root / str(row.package_path)
            _require(package_path.exists(), f"{family} {row.month}: missing package path", failures)
            _require(_sha256(package_path) == row.package_sha256, f"{family} {row.month}: package hash mismatch", failures)
            package = load_package(package_path)
            _require(package.family == family, f"{family} {row.month}: family mismatch inside package", failures)
            _require(package.feature_names == FEATURES and package.train_months == 6, f"{family} {row.month}: invalid feature/training contract", failures)
            held_end = held_start + pd.offsets.MonthBegin(1)
            held = prediction.loc[
                pd.to_datetime(prediction["__decision_ts__"], utc=True).ge(held_start)
                & pd.to_datetime(prediction["__decision_ts__"], utc=True).lt(held_end)
            ].copy()
            _require(len(held) == int(row.held_rows) and len(held) > 0, f"{family} {row.month}: held prediction identity/count mismatch", failures)
            static = package.predict_static(held)
            static_delta = float(np.max(np.abs(static - held["static_expected_bps"].to_numpy(float))))
            max_static_delta = max(max_static_delta, static_delta)
            _require(static_delta <= 1e-12, f"{family} {row.month}: serialized static score delta={static_delta}", failures)
            shift_state = pd.read_parquet(package_path / "prior21d_shift_state.parquet")
            state_days = pd.to_datetime(shift_state["decision_day"], utc=True)
            state_max = pd.to_datetime(shift_state["max_policy_label_available_ts"], utc=True, errors="coerce")
            _require((state_max.lt(state_days) | state_max.isna()).all(), f"{family} {row.month}: shift ledger contains future label", failures)
            expected = apply_shift(static, held["__decision_ts__"], shift_state)
            shift_delta = float(np.max(np.abs(expected - held["mc1_expected_bps"].to_numpy(float))))
            max_shift_delta = max(max_shift_delta, shift_delta)
            _require(shift_delta <= 1e-12, f"{family} {row.month}: serialized shift delta={shift_delta}", failures)
            total_packages += 1
    _require(total_packages == 14, "expected fourteen family/vintage packages", failures)
    checks.update({
        "package_count": total_packages,
        "max_serialized_static_prediction_delta": max_static_delta,
        "max_serialized_shift_prediction_delta": max_shift_delta,
        "declared_package_index_entries": len(index.get("families", ())),
    })
    _require(len(index.get("families", ())) == total_packages, "package index count differs from package audit", failures)
    dual = pd.read_parquet(root / "dual_predictions.parquet", columns=["candidate_id", "__decision_ts__"])
    _require(not dual.duplicated(["candidate_id", "__decision_ts__"]).any(), "dual MC1 panel has duplicate identities", failures)
    return {
        "schema": "strict_r3_p8u_dual_mc1_prequential_v2_audit_v1",
        "root": str(root),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "receipt_sha256": _sha256(root / "run_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new immutable audit directory")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable audit output exists: {args.out}")
    result = audit(args.root)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "correctness_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
