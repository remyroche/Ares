"""Regression tests for the live producer's immutable-receipt recovery loop."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_strict_r3_live_hourly_entry_producer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hourly_producer_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_loop_for_calls(monkeypatch, module, outcomes, *, failed_retry_seconds=1.0):
    calls = []
    clock_values = [
        pd.Timestamp("2026-08-16T01:00:00Z"),
        pd.Timestamp("2026-08-16T01:00:02Z"),
        pd.Timestamp("2026-08-16T01:00:03Z"),
    ]
    clock = iter(clock_values)

    class TimestampClock:
        @staticmethod
        def now(*, tz):
            assert tz == "UTC"
            return next(clock, clock_values[-1])

    def fake_run_once(_args, *, decision):
        calls.append(decision)
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    sleeps = 0

    def fake_sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            raise SystemExit

    monkeypatch.setattr(module, "run_once", fake_run_once)
    monkeypatch.setattr(module, "_load_warm_bundle", lambda _path: {})
    monkeypatch.setattr(module, "_load_warm_live_executor", lambda _path: {})
    monkeypatch.setattr(module, "pd", SimpleNamespace(Timestamp=TimestampClock, Timedelta=pd.Timedelta))
    monkeypatch.setattr(module.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT), "--inference-bundle", "bundle.json",
            "--execution-bundle", "execution.json", "--live-state", "state.json",
            "--bootstrap-previous-run", "previous", "--loop", "--poll-seconds", "1",
            "--failed-retry-seconds", str(failed_retry_seconds),
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    return calls


def test_loop_retries_current_hour_after_pre_execution_receipt_contention(monkeypatch):
    module = _load_module()
    calls = _run_loop_for_calls(
        monkeypatch,
        module,
        [FileExistsError("receipt is temporarily in progress"), {"status": "pass"}],
    )
    assert len(calls) == 2
    assert calls[0] == calls[1] == pd.Timestamp("2026-08-16T01:00:00Z")


def test_loop_never_auto_retries_after_execution_stage_started(monkeypatch):
    module = _load_module()
    calls = _run_loop_for_calls(
        monkeypatch,
        module,
        [{"status": "failed_closed", "execution_attempt_started": True}],
    )
    assert len(calls) == 1


def test_dead_pre_execution_lease_is_terminalized_then_skipped(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    orphan = tmp_path / "receipt_v1"
    orphan.mkdir()
    (orphan / "producer_lease.json").write_text(
        '{"pid":99999999,"decision_ts":"2026-08-16T03:00:00+00:00"}'
    )

    candidate, attempt = module._next_receipt("receipt")

    assert candidate == tmp_path / "receipt_v2"
    assert attempt == 2
    terminal = __import__("json").loads((orphan / "run_manifest.json").read_text())
    assert terminal["status"] == "failed_closed"
    assert terminal["execution_attempt_started"] is False
    assert terminal["terminalization"]["kind"] == "dead_pre_execution_producer_lease"


def test_execution_intent_is_never_auto_retried_after_restart(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    receipt = tmp_path / "receipt_v1"
    receipt.mkdir()
    (receipt / "run_manifest.json").write_text(
        '{"status":"failed_closed","execution_attempt_started":true}'
    )

    with pytest.raises(FileExistsError, match="exchange-execution boundary"):
        module._next_receipt("receipt")


def test_runtime_reseal_never_replays_a_completed_live_decision(monkeypatch, tmp_path):
    module = _load_module()
    decision = pd.Timestamp("2026-08-16T02:00:00Z")
    receipt = (
        tmp_path / "strict_r3_live_hourly_producer_v51_20260816T020000Z_v1"
    )
    receipt.mkdir()
    (receipt / "run_manifest.json").write_text(
        '{"status":"pass","mode":"live","exchange_order_submission":true,'
        '"decision_ts":"2026-08-16T02:00:00+00:00"}'
    )
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)

    with pytest.raises(module.SuccessfulProducerReceiptExists) as caught:
        module.run_once(SimpleNamespace(), decision=decision)

    assert caught.value.receipt == receipt


def test_explicit_compatible_runtime_reseal_can_supply_exact_predecessor(
    monkeypatch, tmp_path,
):
    module = _load_module()
    old_hash = "old-runtime-only-hash"
    predecessor = tmp_path / "strict_r3_successor_v51_live_20260816T020000Z_v1"
    (predecessor / "feature_state/bundle").mkdir(parents=True)
    (predecessor / "cycle/score/geometry_k9_state").mkdir(parents=True)
    (predecessor / "feature_state/bundle/state_bundle_manifest.json").write_text("{}")
    (predecessor / "cycle/next_portfolio_state.json").write_text("{}")
    (predecessor / "cycle/score/geometry_k9_state/causal_geometry_k9_history.parquet").write_bytes(b"test")
    (predecessor / "cycle/score/geometry_k9_state/run_manifest.json").write_text(
        '{"next_decision_ts":"2026-08-16T03:00:00+00:00"}'
    )
    (predecessor / "run_manifest.json").write_text(
        '{"decision_ts":"2026-08-16T02:00:00+00:00",'
        '"completed_within_live_decision_window":true,'
        '"hashes":{"inference_bundle":"old-runtime-only-hash"}}'
    )
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    bootstrap = tmp_path / "bootstrap"

    selected = module._successful_predecessor(
        decision=pd.Timestamp("2026-08-16T03:00:00Z"),
        bundle_hash="new-hash",
        compatible_bundle_hashes={"new-hash", old_hash},
        bootstrap=bootstrap,
    )

    assert selected == predecessor


def test_existing_stale_direct_predecessor_fails_closed_without_archival_scan(
    monkeypatch, tmp_path,
):
    module = _load_module()
    bootstrap = tmp_path / "strict_r3_successor_v1_live_20260816T020000Z_v1"
    bootstrap.mkdir()
    (bootstrap / "run_manifest.json").write_text(
        '{"decision_ts":"2026-08-16T02:00:00+00:00",'
        '"completed_within_live_decision_window":true,'
        '"hashes":{"inference_bundle":"same"}}'
    )
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)

    with pytest.raises(RuntimeError, match="recover the missing hour"):
        module._successful_predecessor(
            decision=pd.Timestamp("2026-08-16T03:00:00Z"),
            bundle_hash="same",
            compatible_bundle_hashes={"same"},
            bootstrap=bootstrap,
        )


def test_successful_hour_advances_the_direct_bootstrap(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    run = tmp_path / "data_perp/artifacts/strict_r3_successor_v1_live_20260816T020000Z_v1"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}")
    args = SimpleNamespace(bootstrap_previous_run="old/run")

    updated = module._advance_direct_bootstrap(
        args,
        {"hourly_run": str(run.relative_to(tmp_path))},
    )

    assert updated == str(run.relative_to(tmp_path))
    assert args.bootstrap_previous_run == updated


def test_invalid_completed_hour_never_replaces_direct_bootstrap(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    args = SimpleNamespace(bootstrap_previous_run="old/run")

    assert module._advance_direct_bootstrap(args, {"hourly_run": "outside"}) is None
    assert args.bootstrap_previous_run == "old/run"


def test_warm_executor_path_reuses_the_preloaded_execution_boundary(
    monkeypatch, tmp_path,
):
    module = _load_module()
    recorded = {}

    def execute_verified_hour(**kwargs):
        recorded.update(kwargs)
        return {"mode": "live", "checks_passed": True}

    def atomic_json(path, payload):
        path.write_text(json.dumps(payload))

    monkeypatch.setattr(
        module,
        "_load_warm_live_executor",
        lambda _path: (
            object(), object(), execute_verified_hour, atomic_json, nullcontext,
        ),
    )
    out = tmp_path / "execution_receipt.json"
    result = module._execute_verified_hour_warm(
        execution_bundle=tmp_path / "execution.json",
        hourly_run=tmp_path / "hourly_run",
        state_path=tmp_path / "state.json",
        out=out,
        live_hour_audit=tmp_path / "audit",
        runtime_checkpoint=tmp_path / "checkpoint",
    )

    assert result["checks_passed"] is True
    assert recorded["submit_orders"] is True
    assert recorded["state_path"] == tmp_path / "state.json"
    assert json.loads(out.read_text())["mode"] == "live"


def test_runtime_reseal_ignores_historical_bridge_for_another_successor(
    monkeypatch, tmp_path,
):
    """An append-only bridge list must validate only the active successor."""
    module = _load_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    predecessor = tmp_path / "predecessor.json"
    predecessor_payload = {
        "schema": "strict_r3_inference_bundle_v1",
        "runtime_code_sha256": {"runtime.py": "old"},
        "static_contract": "unchanged",
    }
    predecessor.write_text(json.dumps(predecessor_payload))
    current_payload = {
        "schema": "strict_r3_inference_bundle_v1",
        "runtime_code_sha256": {"runtime.py": "new"},
        "static_contract": "unchanged",
    }
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({
        "runtime_reseal_predecessors": [
            # An earlier bridge is retained as lineage and intentionally has
            # no fields required for the active bridge.
            {"current_inference_bundle_sha256": "historical-successor"},
            {
                "current_inference_bundle_sha256": "active-successor",
                "predecessor_inference_bundle": "predecessor.json",
                "predecessor_inference_bundle_sha256": module._sha(predecessor),
                "allowed_runtime_code_paths": ["runtime.py"],
                "added_runtime_code_paths": [],
            },
        ],
    }))

    compatible = module._verified_runtime_reseal_predecessors(
        execution_bundle=execution,
        current_bundle=tmp_path / "current.json",
        current_bundle_hash="active-successor",
        current_payload=current_payload,
    )

    assert compatible == {"active-successor", module._sha(predecessor)}
