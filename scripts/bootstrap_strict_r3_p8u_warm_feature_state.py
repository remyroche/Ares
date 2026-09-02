#!/usr/bin/env python3
"""Bootstrap a same-contract P8U transform-state snapshot.

This one-time offline command is the only permitted way to make an initial
state bundle consumable by ``run_strict_r3_p8u_warm_feature_worker.py``.  It
runs the existing canonical incremental materializer over an explicitly
provided, target-free warm-up panel, compares every sealed P8U feature with a
full causal reference, and snapshots the state *only after* parity passes.

It does not score models, map EV, admit candidates, or interact with an
exchange. A state from any other P8U feature union cannot satisfy this tool:
the snapshot is hash-bound to the exact ordered plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_warm_feature_state import (  # noqa: E402
    P8UWarmFeatureConfig,
    assert_feature_output_contract,
    atomic_json,
    audit_feature_parity,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--activated-config-out",
        type=Path,
        help=(
            "Write an immutable successor worker config that binds this exact "
            "same-plan bootstrap bundle. Required for a resumable warm worker."
        ),
    )
    parser.add_argument(
        "--activated-state-name",
        help="Named state namespace for --activated-config-out.",
    )
    args = parser.parse_args()
    args.candidates = args.candidates.resolve()
    args.panel_state = args.panel_state.resolve()
    args.reference_features = args.reference_features.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.activated_config_out is not None:
        args.activated_config_out = args.activated_config_out.resolve()
    config = P8UWarmFeatureConfig.load(args.config, root=ROOT)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable P8U state-bootstrap output exists: {args.out_dir}")
    if not args.candidates.is_file() or not args.panel_state.is_file() or not args.reference_features.is_file():
        raise FileNotFoundError("bootstrap inputs must be immutable files")
    args.out_dir.mkdir(parents=True)
    cache = args.out_dir / "cache"
    features = args.out_dir / "features"
    materializer = [
        sys.executable,
        str(ROOT / "scripts/materialize_strict_r3_forward_features_incremental_v13.py"),
        "--candidates", str(args.candidates),
        "--panel-state", str(args.panel_state),
        "--cache-dir", str(cache),
        "--requested-features-json", str(config.payload["feature_plan_path"]),
        "--feature-cache-namespace", config.state_contract_id,
        "--bootstrap-state",
        "--bootstrap-state-retention-hours", str(
            int(config.payload["stateful_tail_hours"])
        ),
        "--emit-all-candidate-timestamps",
        "--side", "long",
        "--out-dir", str(features),
    ]
    log = args.out_dir / "materializer.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(materializer, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"P8U state bootstrap failed; inspect {log}")
    output = features / "canonical120_features.parquet"
    assert_feature_output_contract(output, config.feature_plan)
    parity = audit_feature_parity(
        incremental_features=output,
        reference_features=args.reference_features,
        required_features=config.feature_plan,
        out_dir=args.out_dir / "parity",
        atol=float(config.payload.get("parity_atol", 1e-6)),
        rtol=float(config.payload.get("parity_rtol", 1e-6)),
    )
    if parity["status"] != "pass":
        raise AssertionError("P8U state bootstrap did not pass all-feature parity")
    snapshot_dir = args.out_dir / "state_bundle"
    snapshot = [
        sys.executable,
        str(ROOT / "scripts/snapshot_strict_r3_feature_state_bundle.py"),
        "--cache-dir", str(cache),
        "--panel-state", str(args.panel_state),
        "--out-dir", str(snapshot_dir),
        "--contract-hash", config.feature_union_sha256,
        "--scope", config.state_contract_id,
        "--panel-tail-hours", str(int(config.payload["stateful_tail_hours"])),
        "--expected-state-timestamp", str(
            json.loads((features / "feature_manifest.json").read_text())["latest_signal_ts"]
        ),
    ]
    for kind in sorted(config.payload.get("required_state_kinds", [])):
        snapshot.extend(["--required-state-kind", str(kind)])
    snapshot_log = args.out_dir / "snapshot.log"
    with snapshot_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(snapshot, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"P8U state snapshot failed; inspect {snapshot_log}")
    manifest = json.loads((snapshot_dir / "state_bundle_manifest.json").read_text())
    activated_config = None
    if args.activated_config_out is not None:
        if not args.activated_state_name:
            raise ValueError("--activated-config-out requires --activated-state-name")
        if args.activated_config_out.exists():
            raise FileExistsError(f"activated config exists: {args.activated_config_out}")
        payload = dict(config.payload)
        payload.update({
            "state_name": str(args.activated_state_name),
            "initial_state_bundle": str(snapshot_dir.relative_to(ROOT)),
            "initial_state_bundle_manifest_sha256": sha256_file(
                snapshot_dir / "state_bundle_manifest.json"
            ),
            "bootstrap_required": False,
            "source_contract": (
                "P8U begins only from the exact same-plan bootstrap state "
                "bundle named below; legacy canonical120 bundles are rejected."
            ),
        })
        atomic_json(args.activated_config_out, payload)
        activated_config = {
            "path": str(args.activated_config_out),
            "sha256": sha256_file(args.activated_config_out),
            "state_name": str(args.activated_state_name),
        }
    receipt = {
        "schema": "strict_r3_p8u_warm_feature_state_bootstrap_v1",
        "status": "pass",
        "config": str(config.path),
        "config_sha256": sha256_file(config.path),
        "feature_union_sha256": config.feature_union_sha256,
        "state_contract_id": config.state_contract_id,
        "candidates": str(args.candidates),
        "candidates_sha256": sha256_file(args.candidates),
        "panel_state": str(args.panel_state),
        "panel_state_sha256": sha256_file(args.panel_state),
        "reference_features": str(args.reference_features),
        "reference_features_sha256": sha256_file(args.reference_features),
        "parity": parity,
        "state_bundle": str(snapshot_dir),
        "state_bundle_manifest_sha256": sha256_file(snapshot_dir / "state_bundle_manifest.json"),
        "latest_state_timestamp": manifest.get("latest_state_timestamp"),
        "activated_config": activated_config,
        "outcome_columns_consumed": [],
    }
    atomic_json(args.out_dir / "bootstrap_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
