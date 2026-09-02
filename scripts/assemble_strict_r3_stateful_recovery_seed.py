#!/usr/bin/env python3
"""Make a read-only adapter for a verified state checkpoint missing its root manifest.

This utility never alters the source checkpoint.  It creates an immutable
recovery-only wrapper whose payload directories are symlinks to the source and
whose manifest records the source hashes.  The wrapper is not a live receipt
and may only be consumed by ``rebuild_strict_r3_stateful_recovery.py`` before
a complete no-order recovery chain has been produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_stateful_recovery_seed_v1"
REQUIRED_DIRS = ("candidate_grid", "features", "cycle", "feature_state")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _link(*, source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    source = _resolve(args.source_run)
    out_dir = _resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"immutable recovery seed exists: {out_dir}")
    if (source / "run_manifest.json").exists():
        raise ValueError("source already has a root manifest; no recovery seed is needed")
    for relative in REQUIRED_DIRS:
        if not (source / relative).is_dir():
            raise FileNotFoundError(source / relative)
    for relative in (
        "candidate_grid/run_manifest.json",
        "cycle/score/predictions.parquet",
        "cycle/score/geometry_k9_state/run_manifest.json",
        "feature_state/bundle/state_bundle_manifest.json",
        "portfolio_reconciliation_state.json",
    ):
        if not (source / relative).is_file():
            raise FileNotFoundError(source / relative)

    candidate_manifest = json.loads(
        (source / "candidate_grid/run_manifest.json").read_text()
    )
    # Forward candidate-grid manifests describe a signal interval; their
    # timestamp-exact decision is its end-exclusive boundary. Older hourly
    # wrappers stored ``decision_ts`` directly, so retain that form first.
    decision_ts = str(
        candidate_manifest.get("decision_ts")
        or candidate_manifest.get("end_exclusive")
        or ""
    )
    if not decision_ts:
        raise ValueError("source candidate manifest lacks decision_ts")

    out_dir.mkdir(parents=True, exist_ok=False)
    # The successor prefix assembler must copy all four immutable input roles
    # (candidate population, eligibility, rejections and features) before it
    # appends a recovered hour.  The original adapter exposed only the first
    # three, which let scoring finish but made the final exact append proof
    # fail after the fact.  Link the verified source feature prefix directly;
    # this recovery-only wrapper never recomputes or changes it.
    for relative in ("candidate_grid", "features", "feature_state"):
        _link(source=source / relative, destination=out_dir / relative)
    # The v122 checkpoint persisted the exact-decision shadow portfolio bridge
    # at its root rather than under ``cycle/``.  Materialise that byte-identical
    # payload in the conventional predecessor location; every other cycle
    # component remains an immutable source symlink.
    cycle = out_dir / "cycle"
    cycle.mkdir()
    for child in sorted((source / "cycle").iterdir(), key=lambda item: item.name):
        _link(source=child, destination=cycle / child.name)
    portfolio_source = source / "portfolio_reconciliation_state.json"
    portfolio = json.loads(portfolio_source.read_text())
    if str(portfolio.get("schema")) != "strict_r3_shadow_portfolio_state_v3_adaptive_exit":
        raise ValueError("source reconciliation state is not a shadow portfolio state")
    if str(portfolio.get("as_of_ts")) != decision_ts:
        raise ValueError("source reconciliation state timestamp differs from candidate decision")
    (cycle / "next_portfolio_state.json").write_bytes(portfolio_source.read_bytes())
    manifest = {
        "schema": SCHEMA,
        "mode": "recovery_seed_only",
        "decision_ts": decision_ts,
        "source_run": str(source.relative_to(ROOT)),
        "source_has_root_manifest": False,
        "source_hashes": {
            "candidate_manifest": _sha(source / "candidate_grid/run_manifest.json"),
            "canonical120_features": _sha(
                source / "features/canonical120_features.parquet"
            ),
            "next_portfolio_state": _sha(portfolio_source),
            "predictions": _sha(source / "cycle/score/predictions.parquet"),
            "geometry_state_manifest": _sha(
                source / "cycle/score/geometry_k9_state/run_manifest.json"
            ),
            "feature_state_manifest": _sha(
                source / "feature_state/bundle/state_bundle_manifest.json"
            ),
        },
        "source_payload_access": "read_only_symlinks",
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "eligible_use": "stateful_recovery_bootstrap_only",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
