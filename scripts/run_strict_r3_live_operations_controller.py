#!/usr/bin/env python3
"""Apply bounded, hash-bound operational recovery for the strict-R3 services.

The controller is deliberately narrower than a general-purpose "self-healing"
agent.  It may restart an *already sealed* missing producer or minute monitor
through an allow-listed launcher, after checking the launcher's hash, process
identity, restart budget, and current decision freshness.  It never changes a
model, source, policy, state lineage, or admission/execution gate.  Failures
at those boundaries are recorded as fail-closed incidents for a reviewed,
resealed successor.

This makes unattended operation safer without turning an operational watchdog
into an unbounded code-patching or stale-order mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
REPORTS = ROOT / "data_perp" / "reports"
SCHEMA = "strict_r3_live_operations_controller_config_v1"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ValueError(f"path escapes repository root: {path}")
    return resolved


def _pid_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    # PID files are intentionally kept in /private/tmp by the live launchers.
    if not path.is_absolute() or not str(path).startswith("/private/tmp/"):
        raise ValueError("controller PID paths must be absolute /private/tmp paths")
    return path


def _service_status(spec: dict[str, Any]) -> dict[str, Any]:
    pid_path = _pid_path(spec.get("pid_path"))
    expected = str(spec.get("expected_command_fragment") or "")
    if not expected:
        raise ValueError("service expected_command_fragment is required")
    result: dict[str, Any] = {
        "pid_path": str(pid_path), "expected_command_fragment": expected,
        "pid_file_present": pid_path.is_file(), "pid": None,
        "running": False, "identity_matches": False,
    }
    if not pid_path.is_file():
        return result
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        result["pid_file_invalid"] = True
        return result
    result["pid"] = pid
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        result["process_missing"] = True
        return result
    except PermissionError:
        # Permission ambiguity must never be treated as permission to replace a
        # possibly live service.
        result["process_permission_unknown"] = True
        return result
    result["running"] = True
    try:
        command = subprocess.check_output(
            ["/bin/ps", "-p", str(pid), "-o", "command="], text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["identity_unreadable"] = True
        return result
    result["command"] = command
    result["identity_matches"] = expected in command
    return result


def _validated_launcher(spec: dict[str, Any]) -> Path:
    launcher_text = str(spec.get("launcher") or "")
    expected_hash = str(spec.get("launcher_sha256") or "")
    if not launcher_text or not expected_hash:
        raise ValueError("enabled service restart requires launcher and launcher_sha256")
    launcher = _inside_root(ROOT / launcher_text)
    if not launcher.is_file() or _sha(launcher) != expected_hash:
        raise ValueError("sealed launcher is missing or hash-mismatched")
    return launcher


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = _load_json(path)
    if config.get("schema") != SCHEMA:
        raise ValueError(f"unsupported controller config schema: {config.get('schema')!r}")
    services = config.get("services")
    if not isinstance(services, dict) or set(services) != {"producer", "monitor"}:
        raise ValueError("controller config requires exactly producer and monitor service specs")
    for name, spec in services.items():
        if not isinstance(spec, dict):
            raise ValueError(f"invalid {name} service spec")
        _pid_path(spec.get("pid_path"))
        if not str(spec.get("expected_command_fragment") or ""):
            raise ValueError(f"{name} expected_command_fragment is required")
    budget = int(config.get("max_auto_restarts_per_24h", 0))
    if budget < 0:
        raise ValueError("max_auto_restarts_per_24h must be non-negative")
    verify_seconds = float(config.get("restart_verification_seconds", 2.0))
    if verify_seconds < 0.0 or verify_seconds > 15.0:
        raise ValueError("restart_verification_seconds must be between 0 and 15")
    return config, _sha(path)


def _validate_sealed_contract_files(config: dict[str, Any]) -> dict[str, str]:
    """Verify the immutable files that authorize a service restart.

    The mutable live state is intentionally not part of this mapping: it is
    verified by the producer/monitor at runtime.  Every model, execution and
    activation contract used by a restart belongs here instead.
    """
    declared = config.get("sealed_contract_files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("enabled controller requires non-empty sealed_contract_files")
    verified: dict[str, str] = {}
    for relative, expected in declared.items():
        path = _inside_root(ROOT / str(relative))
        if not path.is_file() or _sha(path) != str(expected):
            raise ValueError(f"sealed controller contract is missing or hash-mismatched: {relative}")
        verified[str(relative)] = str(expected)
    return verified


def _report_path(path: Path) -> Path:
    resolved = path.resolve()
    if REPORTS.resolve() not in resolved.parents or not resolved.is_file():
        raise ValueError("report must be an immutable strict-R3 report under data_perp/reports")
    return resolved


def _execution_boundary_crossed(report: dict[str, Any]) -> bool:
    relative = report.get("producer_receipt")
    if not relative:
        return False
    receipt = _inside_root(ROOT / str(relative))
    manifest = receipt / "run_manifest.json"
    if not manifest.is_file():
        return True
    payload = _load_json(manifest)
    return bool(payload.get("execution_attempt_started")) and payload.get("status") != "pass"


def _restart_count_since(now: pd.Timestamp) -> int:
    cutoff = now - pd.Timedelta(hours=24)
    count = 0
    for manifest in ARTIFACTS.glob("strict_r3_live_operations_controller_*/*run_manifest.json"):
        try:
            payload = _load_json(manifest)
            timestamp = _utc(payload["generated_at"])
        except (KeyError, ValueError):
            continue
        if timestamp >= cutoff and str(payload.get("action") or "").startswith("restart_"):
            count += 1
    return count


def _start_service(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    launcher = _validated_launcher(spec)
    subprocess.Popen(
        ["/bin/bash", str(launcher)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"service": name, "launcher": str(launcher.relative_to(ROOT)), "submitted": True}


def _classify_report(report: dict[str, Any]) -> tuple[str, str]:
    irregularities = [str(value) for value in report.get("irregularities") or []]
    if report.get("status") == "pass":
        return "healthy", "observe_next_candle"
    if _execution_boundary_crossed(report):
        return "execution_boundary_ambiguous", "block_and_escalate"
    if any(value.startswith("position_monitor_") for value in irregularities):
        return "monitor_health_failure", "restart_monitor_if_exact_launcher_is_authorized"
    if any(value.startswith("producer_") for value in irregularities):
        return "producer_pre_execution_failure", "ensure_producer_for_next_fresh_hour"
    if any(
        token in value
        for value in irregularities
        for token in (
            "source_", "feature", "lineage", "runtime", "calibration", "stage_artifact",
            "live_state", "entry_actions", "duplicate", "execution_receipt_missing",
        )
    ):
        return "contract_or_source_failure", "fail_closed_require_diagnosis_and_reseal"
    return "unclassified_operational_failure", "fail_closed_require_diagnosis"


def _choose_action(
    *,
    report: dict[str, Any] | None,
    decision: pd.Timestamp,
    now: pd.Timestamp,
    config: dict[str, Any],
    service_status: dict[str, dict[str, Any]],
) -> tuple[str, str, str]:
    """Return action, reason, and root-cause classification.

    A producer restart always uses its sealed launcher, which is required to
    start at the *next* fresh boundary.  The controller never replays an old
    decision and never operates a same-hour retry itself.
    """
    if report is None:
        if not service_status["monitor"]["running"]:
            return "restart_monitor", "minute_monitor_missing", "watchdog_service_missing"
        if not service_status["producer"]["running"]:
            return "restart_producer_next_fresh", "hourly_producer_missing", "watchdog_service_missing"
        return "observe", "no_report_yet_services_healthy", "watchdog_waiting"

    category, disposition = _classify_report(report)
    if category == "healthy":
        return "observe", "terminal_report_pass", category
    if category == "monitor_health_failure" and not service_status["monitor"]["running"]:
        return "restart_monitor", disposition, category
    if category == "producer_pre_execution_failure" and not service_status["producer"]["running"]:
        # Keep the service warm for the next boundary. It must not execute the
        # already failed decision from the controller path.
        return "restart_producer_next_fresh", disposition, category
    return "fail_closed", disposition, category


def _receipt_dir(runtime_tag: str, decision: pd.Timestamp, source_hash: str) -> Path:
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    return ARTIFACTS / f"strict_r3_live_operations_controller_{runtime_tag}_{tag}_{source_hash[:16]}"


def run_controller(
    *,
    runtime_tag: str,
    config_path: Path,
    decision: pd.Timestamp,
    now: pd.Timestamp,
    report_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_file = _inside_root(config_path)
    config, config_hash = _load_config(config_file)
    report: dict[str, Any] | None = None
    source_hash = config_hash
    if report_path is not None:
        resolved_report = _report_path(report_path)
        report = _load_json(resolved_report)
        if report.get("schema") != "strict_r3_live_candle_report_v2":
            raise ValueError("unsupported live-candle report schema")
        if _utc(report.get("decision_ts")) != decision:
            raise ValueError("report decision timestamp mismatch")
        source_hash = _sha(resolved_report)
    statuses = {name: _service_status(spec) for name, spec in dict(config["services"]).items()}
    action, reason, category = _choose_action(
        report=report, decision=decision, now=now, config=config, service_status=statuses,
    )
    permitted = bool(config.get("enabled")) and bool(config.get("auto_restart_authorized"))
    budget = int(config.get("max_auto_restarts_per_24h", 0))
    prior_restarts = _restart_count_since(now)
    outcome: dict[str, Any] = {}
    if action.startswith("restart_"):
        service = "producer" if action == "restart_producer_next_fresh" else "monitor"
        status = statuses[service]
        # A live but mismatched process is deliberately not killed: that could
        # be a just-resealed successor. Freeze entry and require a human review.
        if status["running"] and not status["identity_matches"]:
            action = "fail_closed"
            reason = f"{service}_process_identity_mismatch"
        elif not permitted:
            action = "restart_not_authorized"
        elif prior_restarts >= budget:
            action = "fail_closed"
            reason = "auto_restart_budget_exhausted"
        else:
            # Validate even in dry-run mode. A dry-run must prove that the
            # exact launcher which would be started is still hash-bound.
            try:
                verified_contracts = _validate_sealed_contract_files(config)
                _validated_launcher(dict(config["services"])[service])
                if dry_run:
                    outcome = {
                        "service": service, "dry_run": True,
                        "sealed_contracts_verified": verified_contracts,
                    }
                else:
                    outcome = _start_service(service, dict(config["services"])[service])
                    verify_seconds = float(config.get("restart_verification_seconds", 2.0))
                    if verify_seconds:
                        time.sleep(verify_seconds)
                    verified = _service_status(dict(config["services"])[service])
                    outcome["post_restart_status"] = verified
                    if not (verified["running"] and verified["identity_matches"]):
                        action = "fail_closed"
                        reason = f"{service}_restart_did_not_verify"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                action = "fail_closed"
                reason = f"{service}_restart_launch_failure: {type(exc).__name__}"
                outcome = {"service": service, "error": str(exc)}
    payload = {
        "schema": "strict_r3_live_operations_controller_v1",
        "generated_at": now.isoformat(),
        "decision_ts": decision.isoformat(),
        "runtime_tag": runtime_tag,
        "controller_config": str(config_file.relative_to(ROOT)),
        "controller_config_sha256": config_hash,
        "report": (
            str(report_path.resolve().relative_to(ROOT)) if report_path is not None else None
        ),
        "report_sha256": source_hash if report_path is not None else None,
        "root_cause_category": category,
        "action": action,
        "reason": reason,
        "auto_restart_authorized": permitted,
        "restart_budget_24h": budget,
        "restarts_last_24h_before_action": prior_restarts,
        "dry_run": dry_run,
        "service_status": statuses,
        "action_outcome": outcome,
        "safety": {
            "no_stale_decision_execution": True,
            "no_model_or_policy_mutation": True,
            "no_state_lineage_mutation": True,
            "execution_boundary_fail_closed": True,
            "unclassified_failures_fail_closed": True,
        },
    }
    target = _receipt_dir(runtime_tag, decision, source_hash) / "run_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = _load_json(target)
        if (
            existing.get("controller_config_sha256") != config_hash
            or existing.get("report_sha256") != payload["report_sha256"]
        ):
            raise ValueError("immutable controller receipt conflicts with current inputs")
        return existing
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(target)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tag", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    result = run_controller(
        runtime_tag=str(args.runtime_tag),
        config_path=args.config,
        decision=_utc(args.decision_ts).floor("h"),
        now=now,
        report_path=args.report,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({key: result[key] for key in ("action", "reason", "root_cause_category")}, sort_keys=True))


if __name__ == "__main__":
    main()
