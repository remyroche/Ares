#!/usr/bin/env python3
"""Create a byte-identical current-runtime receipt for v60's 19:00 state.

Only the state-manifest implementation hashes are updated.  The inventory and
all operator-state payload files must remain byte-identical, which lets the
stateful runner perform its already-sealed one-time re-receipt validation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data_perp/artifacts/strict_r3_successor_v60_live_20260817T190000Z_v1/feature_state/bundle"
OUT = ROOT / "data_perp/artifacts/strict_r3_v100_v60_1900_current_code_feature_state_successor_20260818T090000Z_v1/bundle"
OVERLAY_SOURCE = ROOT / "config/strict_r3_inference_overlay_long_20260801_v80_bcf_current_dual_newest_feature_runtime_exact_stream_audit.json"
OVERLAY_OUT = ROOT / "config/strict_r3_inference_overlay_long_20260801_v81_bcf_current_dual_v60_1900_state_reseal.json"
EXECUTION_SOURCE = ROOT / "config/strict_r3_kraken_live_execution_v78_v100_bcf_current_dual_newest_feature_runtime_exact_stream_audit.json"
EXECUTION_OUT = ROOT / "config/strict_r3_kraken_live_execution_v79_v101_bcf_current_dual_v60_1900_state_reseal.json"
AUTH_SOURCE = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v08_bcf_current_dual_newest_feature_runtime_exact_stream_audit.json"
AUTH_OUT = ROOT / "config/strict_r3_kraken_live_activation_authorization_20260818_v09_bcf_current_dual_v60_1900_state_reseal.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(bundle: Path) -> str:
    inventory = pd.read_parquet(bundle / "operator_state_inventory.parquet")
    digest = hashlib.sha256()
    for row in inventory.loc[:, ["relative_path", "sha256"]].sort_values(
        "relative_path", kind="stable"
    ).itertuples(index=False):
        digest.update(f"{row.relative_path}\0{row.sha256}\n".encode())
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    OUT.parent.mkdir(parents=True)
    shutil.copytree(SOURCE, OUT)
    manifest_path = OUT / "state_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation = dict(manifest["implementation_sha256"])
    for relative in implementation:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        implementation[relative] = sha(source)
    manifest["implementation_sha256"] = implementation
    manifest["reseal"] = {
        "schema": "strict_r3_feature_state_runtime_reseal_v1",
        "superseded_bundle": str(SOURCE.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(SOURCE / "state_bundle_manifest.json"),
        "operator_state_payload_sha256": payload_digest(SOURCE),
        "reason": "Current-runtime manifest re-receipt; operator payloads byte-identical.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if payload_digest(SOURCE) != payload_digest(OUT):
        raise AssertionError("reseal changed an operator-state payload")

    overlay = json.loads(OVERLAY_SOURCE.read_text())
    feature_state = dict(overlay["overrides"]["runtime"]["feature_state"])
    feature_state["one_time_state_reseal"] = {
        "superseded_bundle": str(SOURCE.relative_to(ROOT)),
        "resealed_bundle": str(OUT.relative_to(ROOT)),
        "superseded_manifest_sha256": sha(SOURCE / "state_bundle_manifest.json"),
        "resealed_manifest_sha256": sha(manifest_path),
        "operator_state_payload_sha256": payload_digest(SOURCE),
        "reason": "One-time v60 19:00 current-code receipt; operator payloads are byte-identical.",
    }
    overlay["overrides"]["runtime"]["feature_state"] = feature_state
    overlay["purpose"] = (
        "v81: v60 19:00 persisted feature-state re-receipt for recovery. "
        "Frozen models, Geometry/K9, features, admission, portfolio and rich exit are unchanged."
    )
    write_new(OVERLAY_OUT, overlay)

    authorization = json.loads(AUTH_SOURCE.read_text())
    authorization.update({
        "authorization_source": "User-approved v60 19:00 state recovery reseal; no strategy change.",
        "inference_bundle": str(OVERLAY_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY_OUT),
    })
    write_new(AUTH_OUT, authorization)

    execution = json.loads(EXECUTION_SOURCE.read_text())
    execution.update({
        "version_note": "v101: v60 19:00 state recovery successor; strategy unchanged.",
        "inference_bundle": str(OVERLAY_OUT.relative_to(ROOT)),
        "inference_bundle_sha256": sha(OVERLAY_OUT),
        "activation_authorization": str(AUTH_OUT.relative_to(ROOT)),
        "activation_authorization_sha256": sha(AUTH_OUT),
    })
    write_new(EXECUTION_OUT, execution)
    print(json.dumps({
        "state_bundle": str(OUT.relative_to(ROOT)),
        "state_manifest_sha256": sha(manifest_path),
        "operator_state_payload_sha256": payload_digest(OUT),
        "overlay": str(OVERLAY_OUT.relative_to(ROOT)),
        "overlay_sha256": sha(OVERLAY_OUT),
        "execution": str(EXECUTION_OUT.relative_to(ROOT)),
        "execution_sha256": sha(EXECUTION_OUT),
        "authorization": str(AUTH_OUT.relative_to(ROOT)),
        "authorization_sha256": sha(AUTH_OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
