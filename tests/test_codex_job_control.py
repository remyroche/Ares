from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codex_job_control.py"
SPEC = importlib.util.spec_from_file_location("codex_job_control", SCRIPT)
codex_job_control = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(codex_job_control)


def _entry(pid: int = 12345) -> dict:
    return {
        "pid": pid,
        "pgid": pid,
        "name": "test-job",
        "status": "running",
        "started_by": "codex_job_control",
        "cwd": str(ROOT),
        "argv": [
            "python3",
            "-u",
            "extreme_price_movements/run_pipeline.py",
            "download",
        ],
        "lstart": "Wed Jun 17 20:00:00 2026",
    }


def test_command_is_python_rejects_shell_wrapped_commands():
    assert codex_job_control.command_is_python(["python3", "-u", "script.py"])
    assert codex_job_control.command_is_python(
        ["env", "PYTHONUNBUFFERED=1", "PYTHONPATH=.", "python3", "-u", "script.py"]
    )
    assert not codex_job_control.command_is_python(
        ["/bin/zsh", "-lc", "python3 -u script.py"]
    )


def test_validate_entry_refuses_pid_reuse(monkeypatch):
    entry = _entry()
    monkeypatch.setattr(codex_job_control, "_process_alive", lambda pid: True)
    monkeypatch.setattr(codex_job_control, "_process_cwd", lambda pid: ROOT)
    monkeypatch.setattr(
        codex_job_control,
        "_process_lstart",
        lambda pid: "Wed Jun 17 21:00:00 2026",
    )

    with pytest.raises(codex_job_control.JobControlError, match="pid was reused"):
        codex_job_control.validate_entry_for_stop(entry)


def test_stop_dry_run_does_not_signal_registered_matching_job(
    monkeypatch, tmp_path, capsys
):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("CODEX_JOB_CONTROL_REGISTRY", str(registry_path))
    registry = codex_job_control._empty_registry()
    registry["jobs"]["12345"] = _entry()
    codex_job_control.save_registry(registry, registry_path)

    sent_signals = []
    monkeypatch.setattr(codex_job_control, "_process_alive", lambda pid: True)
    monkeypatch.setattr(codex_job_control, "_process_cwd", lambda pid: ROOT)
    monkeypatch.setattr(
        codex_job_control,
        "_process_lstart",
        lambda pid: "Wed Jun 17 20:00:00 2026",
    )
    monkeypatch.setattr(
        codex_job_control,
        "_process_command",
        lambda pid: "python3 -u extreme_price_movements/run_pipeline.py download",
    )
    monkeypatch.setattr(codex_job_control, "_current_pgid", lambda pid: pid)
    monkeypatch.setattr(
        codex_job_control,
        "_send_signal",
        lambda *args: sent_signals.append(args),
    )

    rc = codex_job_control.stop_jobs(
        Namespace(pid=12345, name=None, all=False, timeout=0.1, dry_run=True)
    )

    assert rc == 0
    assert sent_signals == []
    out = json.loads(capsys.readouterr().out)
    assert out["results"][0]["status"] == "dry_run"
    assert out["results"][0]["cwd_verified"] is True


def test_stop_signals_process_group_for_wrapper_started_job(
    monkeypatch, tmp_path, capsys
):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("CODEX_JOB_CONTROL_REGISTRY", str(registry_path))
    registry = codex_job_control._empty_registry()
    registry["jobs"]["12345"] = _entry()
    codex_job_control.save_registry(registry, registry_path)

    sent_signals = []
    monkeypatch.setattr(codex_job_control, "_process_alive", lambda pid: True)
    monkeypatch.setattr(codex_job_control, "_process_cwd", lambda pid: ROOT)
    monkeypatch.setattr(
        codex_job_control,
        "_process_lstart",
        lambda pid: "Wed Jun 17 20:00:00 2026",
    )
    monkeypatch.setattr(
        codex_job_control,
        "_process_command",
        lambda pid: "python3 -u extreme_price_movements/run_pipeline.py download",
    )
    monkeypatch.setattr(codex_job_control, "_current_pgid", lambda pid: pid)

    def fake_send(pid, pgid, sig):
        sent_signals.append((pid, pgid, sig.name))
        return f"pgid:{pgid}"

    monkeypatch.setattr(codex_job_control, "_send_signal", fake_send)
    monkeypatch.setattr(
        codex_job_control, "_wait_until_stopped", lambda pid, timeout: True
    )

    rc = codex_job_control.stop_jobs(
        Namespace(pid=12345, name=None, all=False, timeout=0.1, dry_run=False)
    )

    assert rc == 0
    assert sent_signals == [(12345, 12345, "SIGTERM")]
    out = json.loads(capsys.readouterr().out)
    assert out["results"][0]["status"] == "stopped"
    updated = codex_job_control.load_registry(registry_path)
    assert updated["jobs"]["12345"]["status"] == "stopped"
