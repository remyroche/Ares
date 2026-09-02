#!/usr/bin/env python3
"""Guarded research-only handoff from CMI receipts to U/O/M Meta head screens.

The upstream CMI prescreens are long-running and immutable.  This controller
waits for their final receipts, checks the exact frozen target/query arm and
Base-Explanation-V1 top-tail contract, freezes one 80-feature input contract
per family, then launches the three strict-OOF objective screens.  It is
deliberately not a portfolio, MC1, admission, live, or exchange producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "config/strict_r3_p8u_uom_cmi_input_freeze_source_repaired_expandedbase_20260829_v1.json"
CONFIG = ROOT / "config/strict_r3_p8u_meta_target_query_grid_source_repaired_dec24_jul25_20260829_v2_expanded_base_lineage.json"
TRIALS = ROOT / "config/strict_r3_p8u_meta_lgbm_objective_stage3a_20260828_v1.json"
CONTRACT_BUILDER = ROOT / "scripts/build_strict_r3_p8u_cmi_meta_contracts_v1.py"
SCORER = ROOT / "scripts/run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py"
HELD_MONTHS = ("2025-05", "2025-06", "2025-07")
SCHEMA = "strict_r3_p8u_uom_objective_handoff_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _once(path: Path, value: Any) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _event(root: Path, **value: Any) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _freeze() -> dict[str, str]:
    data = json.loads(FREEZE.read_text())
    if data.get("schema") != "strict_r3_p8u_uom_cmi_input_freeze_v1":
        raise AssertionError("unexpected CMI freeze schema")
    arms = {str(key): str(value) for key, value in data["selected_arms"].items()}
    if set(arms) != {"under", "over", "magnitude"}:
        raise AssertionError("required Under/Over/Magnitude arms are not frozen")
    return arms


def _prescreen_root(family: str) -> Path:
    return ROOT / f"data_perp/artifacts/strict_r3_p8u_uom_cmi_{family}_full1400_baseexp_top15_source_repaired_20260829_v1"


def _audit(arms: dict[str, str]) -> tuple[bool, list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    for family, arm in sorted(arms.items()):
        source = _prescreen_root(family)
        required = (
            source / "run_manifest.json", source / "prescreen_contract.json",
            source / "prescreen_summary.parquet", source / "prescreen_correctness_report.json",
        )
        row: dict[str, Any] = {"family": family, "arm": arm, "source": str(source), "ready": False}
        if not all(path.is_file() for path in required):
            audit.append(row)
            continue
        manifest = json.loads((source / "run_manifest.json").read_text())
        contract = json.loads((source / "prescreen_contract.json").read_text())
        correctness = json.loads((source / "prescreen_correctness_report.json").read_text())
        checks = {
            "manifest_arm": str(manifest.get("arm", {}).get("name")) == arm,
            "contract_arm": str(contract.get("arm")) == arm,
            "top15": float(contract.get("base_top_fraction_for_cmi", -1.0)) == .15,
            "candidate_support": int(contract.get("candidate_count", 0)) >= 80,
            "causal_receipt": all(bool(value) for value in correctness.values()),
            "held_months": tuple(manifest.get("held_months", ())) == HELD_MONTHS,
        }
        row.update(checks)
        row["ready"] = bool(all(checks.values()))
        audit.append(row)
    return bool(audit and all(row["ready"] for row in audit)), audit


def _run(root: Path, *, poll_seconds: int) -> None:
    arms = _freeze()
    logs = root / "logs"; logs.mkdir()
    _once(root / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline research-only CMI-contract and strict-OOF U/O/M objective handoff; no MC1/admission/portfolio/live/execution/exchange mutation",
        "freeze": str(FREEZE), "freeze_sha256": _sha(FREEZE),
        "config": str(CONFIG), "config_sha256": _sha(CONFIG),
        "trials": str(TRIALS), "trials_sha256": _sha(TRIALS),
        "held_months": list(HELD_MONTHS), "feature_count": 80,
        "selection": "sealed full-1400 Base-Explanation-V1 top-15% conditional-MI prescreen; MDA intentionally not invoked",
    })
    while True:
        ready, audit = _audit(arms)
        _once_or_replace(root / "cmi_receipt_audit.json", {"ready": ready, "families": audit})
        _event(root, event="cmi_receipt_audit", ready=ready, ready_families=int(sum(row["ready"] for row in audit)))
        if ready:
            break
        time.sleep(poll_seconds)
    contracts: dict[str, Path] = {}
    for family in sorted(arms):
        out = root / "contracts" / family
        out.parent.mkdir(exist_ok=True)
        command = [sys.executable, str(CONTRACT_BUILDER), "--prescreen-root", str(_prescreen_root(family)), "--feature-count", "80", "--out", str(out)]
        subprocess.run(command, cwd=ROOT, check=True, text=True)
        contracts[family] = out / "contract.json"
        _event(root, event="contract_frozen", family=family, contract=str(contracts[family]))
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for family, arm in sorted(arms.items()):
        out = ROOT / f"data_perp/artifacts/strict_r3_p8u_uom_{family}_full1400_objective_source_repaired_20260829_v1"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite immutable objective root: {out}")
        log = (logs / f"{family}.log").open("w", encoding="utf-8")
        command = [
            sys.executable, str(SCORER), "--config", str(CONFIG), "--arm", arm,
            "--trials", str(TRIALS), "--out", str(out), "--feature-contract", str(contracts[family]),
            "--held-months", *HELD_MONTHS,
        ]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        processes.append((family, process, log))
        _event(root, event="objective_started", family=family, arm=arm, pid=process.pid, out=str(out))
    failed = False
    for family, process, log in processes:
        code = process.wait(); log.close()
        _event(root, event="objective_finished", family=family, returncode=code)
        failed = failed or code != 0
    if failed:
        raise SystemExit("at least one immutable U/O/M objective screen failed; inspect family logs")
    _once(root / "completion.json", {
        "all_three_objective_screens_completed": True,
        "next_gate": "audit head receipts, compare against the frozen control, and decide whether any family qualifies for untouched 2026 validation",
    })


def _once_or_replace(path: Path, value: Any) -> None:
    # This is bounded progress telemetry, never a research input/receipt.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    if args.poll_seconds < 15:
        raise ValueError("poll interval must be at least 15 seconds")
    args.out.mkdir(parents=True)
    _run(args.out.resolve(), poll_seconds=int(args.poll_seconds))


if __name__ == "__main__":
    main()
