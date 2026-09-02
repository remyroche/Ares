"""Focused safety tests for bounded strict-R3 operations auto-recovery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/run_strict_r3_live_operations_controller.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("strict_r3_operations_controller", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "Ares"
    artifacts = root / "data_perp" / "artifacts"
    reports = root / "data_perp" / "reports"
    artifacts.mkdir(parents=True)
    reports.mkdir(parents=True)
    (root / "scripts").mkdir()
    producer = root / "scripts" / "producer.sh"
    monitor = root / "scripts" / "monitor.sh"
    producer.write_text("#!/usr/bin/env bash\nexit 0\n")
    monitor.write_text("#!/usr/bin/env bash\nexit 0\n")
    inference = root / "config" / "inference.json"
    execution = root / "config" / "execution.json"
    inference.parent.mkdir()
    inference.write_text('{"contract":"inference"}\n')
    execution.write_text('{"contract":"execution"}\n')
    config = root / "controller.json"
    config.write_text(json.dumps({
        "schema": "strict_r3_live_operations_controller_config_v1",
        "enabled": True,
        "auto_restart_authorized": True,
        "max_auto_restarts_per_24h": 2,
        "sealed_contract_files": {
            "config/inference.json": _sha(inference),
            "config/execution.json": _sha(execution),
        },
        "services": {
            "producer": {
                "pid_path": "/private/tmp/test_strict_r3_producer.pid",
                "expected_command_fragment": "producer-marker",
                "launcher": "scripts/producer.sh",
                "launcher_sha256": _sha(producer),
            },
            "monitor": {
                "pid_path": "/private/tmp/test_strict_r3_monitor.pid",
                "expected_command_fragment": "monitor-marker",
                "launcher": "scripts/monitor.sh",
                "launcher_sha256": _sha(monitor),
            },
        },
    }))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)
    monkeypatch.setattr(module, "REPORTS", reports)
    return module, root, reports, config


def _status(*, producer=False, monitor=False):
    return {
        "producer": {"running": producer, "identity_matches": producer},
        "monitor": {"running": monitor, "identity_matches": monitor},
    }


def test_watchdog_restarts_only_missing_monitor_with_sealed_launcher(monkeypatch, tmp_path):
    module, _root, _reports, config = _setup(monkeypatch, tmp_path)
    launched = []
    calls = {"monitor": 0}

    def status(spec):
        if "producer" in str(spec.get("launcher")):
            return _status()["producer"]
        calls["monitor"] += 1
        # The monitor is absent before the restart and valid immediately after
        # its known launcher is submitted.
        return {"running": calls["monitor"] > 1, "identity_matches": calls["monitor"] > 1}

    monkeypatch.setattr(module, "_service_status", status)
    monkeypatch.setattr(module.subprocess, "Popen", lambda argv, **kwargs: launched.append((argv, kwargs)))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module.run_controller(
        runtime_tag="test", config_path=config,
        decision=pd.Timestamp("2026-08-24T10:00:00Z"),
        now=pd.Timestamp("2026-08-24T10:03:00Z"),
    )

    assert result["action"] == "restart_monitor"
    assert result["action_outcome"]["service"] == "monitor"
    assert launched and launched[0][0][1].endswith("monitor.sh")


def test_controller_never_restarts_from_source_or_lineage_incident(monkeypatch, tmp_path):
    module, _root, reports, config = _setup(monkeypatch, tmp_path)
    report = reports / "strict_r3_live_candle_test_20260824T100000Z_x.json"
    report.write_text(json.dumps({
        "schema": "strict_r3_live_candle_report_v2",
        "decision_ts": "2026-08-24T10:00:00+00:00",
        "status": "action_required",
        "irregularities": ["source_coverage_incomplete", "feature_runtime_parity_failure"],
        "producer_receipt": None,
    }))
    monkeypatch.setattr(module, "_service_status", lambda _spec: {"running": False, "identity_matches": False})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not launch")))

    result = module.run_controller(
        runtime_tag="test", config_path=config,
        decision=pd.Timestamp("2026-08-24T10:00:00Z"),
        now=pd.Timestamp("2026-08-24T10:04:00Z"), report_path=report,
    )

    assert result["action"] == "fail_closed"
    assert result["root_cause_category"] == "contract_or_source_failure"


def test_execution_boundary_failure_is_never_auto_restarted(monkeypatch, tmp_path):
    module, root, reports, config = _setup(monkeypatch, tmp_path)
    producer = root / "data_perp" / "artifacts" / "producer"
    producer.mkdir()
    (producer / "run_manifest.json").write_text(json.dumps({
        "status": "failed_closed", "execution_attempt_started": True,
    }))
    report = reports / "strict_r3_live_candle_test_20260824T100000Z_intent.json"
    report.write_text(json.dumps({
        "schema": "strict_r3_live_candle_report_v2",
        "decision_ts": "2026-08-24T10:00:00+00:00",
        "status": "action_required",
        "irregularities": ["producer_failed_closed: exchange uncertain"],
        "producer_receipt": str(producer.relative_to(root)),
    }))
    monkeypatch.setattr(module, "_service_status", lambda _spec: {"running": False, "identity_matches": False})

    result = module.run_controller(
        runtime_tag="test", config_path=config,
        decision=pd.Timestamp("2026-08-24T10:00:00Z"),
        now=pd.Timestamp("2026-08-24T10:04:00Z"), report_path=report,
    )

    assert result["action"] == "fail_closed"
    assert result["root_cause_category"] == "execution_boundary_ambiguous"


def test_disabled_config_records_without_starting_any_service(monkeypatch, tmp_path):
    module, _root, _reports, config = _setup(monkeypatch, tmp_path)
    payload = json.loads(config.read_text())
    payload["enabled"] = False
    payload["auto_restart_authorized"] = False
    config.write_text(json.dumps(payload))
    monkeypatch.setattr(module, "_service_status", lambda _spec: {"running": False, "identity_matches": False})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not launch")))

    result = module.run_controller(
        runtime_tag="test", config_path=config,
        decision=pd.Timestamp("2026-08-24T10:00:00Z"),
        now=pd.Timestamp("2026-08-24T10:03:00Z"),
    )

    assert result["action"] == "restart_not_authorized"
