#!/usr/bin/env python3
"""Publish a read-only manifest across consecutive v60-derived recovery roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    roots = [path.resolve() for path in args.root]
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    hours: list[dict] = []
    root_receipts: list[dict] = []
    coverage: list[pd.DataFrame] = []
    positions: list[pd.DataFrame] = []
    summary: list[pd.DataFrame] = []
    k9_rows: list[dict] = []
    previous_final: Path | None = None
    for root in roots:
        manifest = _load(root / "run_manifest.json")
        parity = _load(root / "terminal_state_replay_parity_manifest.json")
        if not (
            manifest.get("schema") == "strict_r3_stateful_recovery_v1"
            and manifest.get("status") == "complete"
            and int(manifest.get("exchange_calls", -1)) == 0
            and manifest.get("order_submission_enabled") is False
            and parity.get("status") == "pass"
        ):
            raise AssertionError(f"root is not a completed zero-order parity-verified recovery: {root}")
        final = (ROOT / str(manifest["final_run"])).resolve()
        if previous_final is not None and str(manifest.get("bootstrap_run")) != _relative(previous_final):
            raise AssertionError("recovery roots are not immediate-predecessor contiguous")
        previous_final = final
        root_receipts.append({
            "root": _relative(root),
            "manifest_sha256": _sha(root / "run_manifest.json"),
            "parity_receipt": _relative(root / "terminal_state_replay_parity_manifest.json"),
            "parity_sha256": _sha(root / "terminal_state_replay_parity_manifest.json"),
            "final_run": _relative(final),
        })
        for hour in manifest.get("hours") or []:
            run = (ROOT / str(hour["run"])).resolve()
            geometry = _load(run / "cycle/score/geometry_k9_state/run_manifest.json")
            hours.append(hour)
            k9_rows.append({
                "decision_timestamp": hour["decision_ts"],
                "recovery_root": _relative(root),
                "geometry_bundle_sha256": hour["geometry_bundle_sha256"],
                "geometry_state_mode": hour["geometry_state_mode"],
                "geometry_state_input": geometry.get("input"),
                "geometry_state_output": geometry.get("output"),
                "geometry_state_manifest_sha256": _sha(run / "cycle/score/geometry_k9_state/run_manifest.json"),
                "run": hour["run"],
            })
        for rel, target in [
            ("per_hour_source_feature_coverage.csv", coverage),
            ("missed_hour_positions.csv", positions),
            ("missed_hour_summary.csv", summary),
        ]:
            frame = pd.read_csv(root / rel)
            frame.insert(0, "recovery_root", _relative(root))
            target.append(frame)

    ordered_hours = sorted(hours, key=lambda item: item["decision_ts"])
    decisions = [pd.Timestamp(item["decision_ts"]) for item in ordered_hours]
    if any(right - left != pd.Timedelta(hours=1) for left, right in zip(decisions, decisions[1:])):
        raise AssertionError("recovery chain skipped an hourly state")
    out.mkdir(parents=True)
    pd.concat(coverage, ignore_index=True).to_parquet(out / "per_hour_source_feature_coverage.parquet", index=False)
    pd.concat(coverage, ignore_index=True).to_csv(out / "per_hour_source_feature_coverage.csv", index=False)
    pd.concat(summary, ignore_index=True).to_parquet(out / "missed_hour_summary.parquet", index=False)
    pd.concat(summary, ignore_index=True).to_csv(out / "missed_hour_summary.csv", index=False)
    pd.concat(positions, ignore_index=True).to_parquet(out / "missed_hour_positions.parquet", index=False)
    pd.concat(positions, ignore_index=True).to_csv(out / "missed_hour_positions.csv", index=False)
    pd.DataFrame(k9_rows).sort_values("decision_timestamp").to_parquet(out / "k9_state_lineage.parquet", index=False)
    pd.DataFrame(k9_rows).sort_values("decision_timestamp").to_csv(out / "k9_state_lineage.csv", index=False)
    payload = {
        "schema": "strict_r3_stateful_recovery_chain_v1",
        "status": "complete",
        "source_bootstrap": "v60_only",
        "start_decision": ordered_hours[0]["decision_ts"],
        "end_decision": ordered_hours[-1]["decision_ts"],
        "recovered_hours": len(ordered_hours),
        "hours": ordered_hours,
        "recovery_roots": root_receipts,
        "final_run": root_receipts[-1]["final_run"],
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "candidate_identity_contract": "complete target-free 170-symbol universe before row-local eligibility",
        "parity_contract": "same-predecessor persisted-state re-advance; all feature/score/admission/portfolio output identities exact within 0.01 percent tolerance",
        "artifacts": {
            "coverage": _relative(out / "per_hour_source_feature_coverage.parquet"),
            "summary": _relative(out / "missed_hour_summary.parquet"),
            "positions": _relative(out / "missed_hour_positions.parquet"),
            "k9_lineage": _relative(out / "k9_state_lineage.parquet"),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "out": _relative(out), "hours": len(ordered_hours), "final_run": payload["final_run"]}, sort_keys=True))


if __name__ == "__main__":
    main()
