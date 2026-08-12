#!/usr/bin/env python3
"""Validate evidence receipts required for governed agent-authored changes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "agents" / "receipts"
VALIDATION_STATUSES = {"passed", "not_run"}


def is_governed(path: Path) -> bool:
    if path.suffix != ".py" or path == Path("scripts/validate_agent_receipt.py"):
        return False
    return path.parts[:1] in {("extreme_price_movements",), ("scripts",)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_contracts(path: Path) -> set[str]:
    required = {"agents/dataset_contract.md"}
    path_text = path.as_posix()
    if path_text.startswith("extreme_price_movements/inference/") or any(
        token in path.name for token in ("portfolio", "policy", "replay")
    ):
        required.update({"agents/feature_pipeline_rules.md", "agents/backtest_protocol.md"})
    if path_text.startswith("scripts/") or any(
        token in path.name
        for token in ("lgbm", "training", "feature", "model", "stage_", "archetype")
    ):
        required.update(
            {
                "agents/leakage_prevention.md",
                "agents/model_validation_protocol.md",
                "agents/experiment_discipline.md",
            }
        )
    return required


def load_receipts() -> list[tuple[Path, dict[str, Any]]]:
    receipts: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.relative_to(ROOT)} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
        receipts.append((path, payload))
    return receipts


def validate_receipt(path: Path, receipt: dict[str, Any], governed: set[str]) -> list[str]:
    errors: list[str] = []
    label = path.relative_to(ROOT).as_posix()
    for key in ("task", "scope", "changed_paths", "contracts", "validation", "agent_plan"):
        if key not in receipt:
            errors.append(f"{label}: missing {key}")

    changed = receipt.get("changed_paths", [])
    if not isinstance(changed, list) or not all(isinstance(item, str) for item in changed):
        errors.append(f"{label}: changed_paths must be a list of strings")
        changed = []
    uncovered = governed - set(changed)
    if uncovered:
        errors.append(f"{label}: does not cover {', '.join(sorted(uncovered))}")

    contracts = receipt.get("contracts", [])
    hashes: dict[str, str] = {}
    if not isinstance(contracts, list):
        errors.append(f"{label}: contracts must be a list")
        contracts = []
    for item in contracts:
        if (
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str)
        ):
            hashes[item["path"]] = item["sha256"]
        else:
            errors.append(f"{label}: each contract needs path and sha256 strings")
    for changed_path in governed:
        for contract in required_contracts(Path(changed_path)):
            contract_path = ROOT / contract
            if hashes.get(contract) != sha256(contract_path):
                errors.append(f"{label}: missing or stale hash for {contract}")

    validation = receipt.get("validation", {})
    if not isinstance(validation, dict) or validation.get("status") not in VALIDATION_STATUSES:
        errors.append(f"{label}: validation.status must be passed or not_run")
    elif validation["status"] == "passed" and not validation.get("commands"):
        errors.append(f"{label}: passed validation requires at least one command")
    elif validation["status"] == "not_run" and not validation.get("not_run_reason"):
        errors.append(f"{label}: not_run validation requires not_run_reason")

    plan = receipt.get("agent_plan", {})
    subagents = plan.get("subagents", []) if isinstance(plan, dict) else None
    if not isinstance(subagents, list):
        errors.append(f"{label}: agent_plan.subagents must be a list")
    else:
        for index, agent in enumerate(subagents):
            if not isinstance(agent, dict):
                errors.append(f"{label}: subagents[{index}] must be an object")
                continue
            model = agent.get("model")
            if model == "luna":
                continue
            if (
                model == "gpt-5.6-terra"
                and isinstance(agent.get("terra_exception"), str)
                and agent["terra_exception"].strip()
            ):
                continue
            errors.append(f"{label}: subagents[{index}] must use luna or document a Terra exception")
    return errors


def main(argv: list[str]) -> int:
    paths = {
        Path(raw).resolve().relative_to(ROOT.resolve()).as_posix()
        for raw in argv
        if Path(raw).resolve().is_relative_to(ROOT.resolve())
    }
    governed = {path for path in paths if is_governed(Path(path))}
    if not governed:
        return 0
    try:
        receipts = load_receipts()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not receipts:
        print("Governed changes require an agents/receipts/*.json evidence receipt.", file=sys.stderr)
        return 1
    candidates = [
        (path, receipt)
        for path, receipt in receipts
        if governed.issubset(set(receipt.get("changed_paths", [])))
    ]
    if not candidates:
        print(
            "No evidence receipt covers all governed changes: " + ", ".join(sorted(governed)),
            file=sys.stderr,
        )
        return 1
    errors = [error for path, receipt in candidates for error in validate_receipt(path, receipt, governed)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
