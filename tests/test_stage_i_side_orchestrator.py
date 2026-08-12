from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from extreme_price_movements.stage_i_side_orchestrator import (
    GIB,
    LEGACY_ORCHESTRATOR_SCHEMA,
    LEGACY_PRUNING_SCHEMA,
    LEGACY_SELECTOR_SCHEMA,
    SideOrchestratorRequest,
    _validate_completed_child,
    _lineage,
    _stable_hash,
    build_side_command,
    choose_execution_mode,
    orchestrate_sides,
    require_safe_single_worker_capacity,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_SELECTOR_SCHEMA,
    stage_i_tiered_pruning_contract,
)


def _selector(root: Path) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"status": "complete"}))
    (root / "selector_feature_contract.json").write_text(json.dumps({"features": ["x"]}))
    (root / "selector_features.parquet").write_bytes(b"fixture")
    (root / "selector_ledger.parquet").write_bytes(b"fixture")
    return root


class _FakeProcess:
    def __init__(self, code: int = 0) -> None:
        self.code = code

    def wait(self) -> int:
        return self.code


class _InterruptProcess:
    def __init__(self, *, interrupts: bool = False) -> None:
        self.interrupts = interrupts
        self.wait_calls = 0
        self.poll_calls = 0
        self.terminated = False
        self.killed = False

    def poll(self):
        self.poll_calls += 1
        if self.interrupts and self.poll_calls == 1:
            raise KeyboardInterrupt()
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        del timeout
        self.wait_calls += 1
        if self.interrupts and self.wait_calls == 1:
            raise KeyboardInterrupt()
        return 0


def test_memory_preflight_has_bounded_parallel_and_sequential_modes() -> None:
    assert choose_execution_mode(
        available_bytes=20 * GIB, worker_bytes=4 * GIB, reserve_bytes=4 * GIB,
    ) == "parallel_two_sides"
    assert choose_execution_mode(
        available_bytes=10 * GIB, worker_bytes=4 * GIB, reserve_bytes=4 * GIB,
    ) == "sequential_memory_fallback"
    require_safe_single_worker_capacity(
        available_bytes=8 * GIB, worker_bytes=4 * GIB, reserve_bytes=4 * GIB,
    )
    with pytest.raises(RuntimeError, match="insufficient safe memory"):
        require_safe_single_worker_capacity(
            available_bytes=7 * GIB, worker_bytes=4 * GIB, reserve_bytes=4 * GIB,
        )


def test_single_side_request_is_explicit_and_validated() -> None:
    request = SideOrchestratorRequest(
        layer="base", selector_dir="selector", output_dir="output", sides=("short",),
    )
    assert request.sides == ("short",)
    with pytest.raises(ValueError, match="sides must contain"):
        SideOrchestratorRequest(
            layer="base", selector_dir="selector", output_dir="output", sides=("flat",),
        )


def test_target_only_support_mode_is_forwarded_to_isolated_child() -> None:
    request = SideOrchestratorRequest(
        layer="base", selector_dir="selector", output_dir="output",
        mda_support_mode="target-only",
    )
    command = build_side_command(request, "long")
    assert command[command.index("--mda-support-mode") + 1] == "target-only"
    with pytest.raises(ValueError, match="mda_support_mode"):
        SideOrchestratorRequest(
            layer="base", selector_dir="selector", output_dir="output",
            mda_support_mode="unsupported",
        )


def test_orchestrator_isolates_sides_sets_four_threads_and_binds_resume(tmp_path: Path) -> None:
    selector = _selector(tmp_path / "selector")
    output = tmp_path / "output"
    request = SideOrchestratorRequest(
        layer="base", selector_dir=str(selector), output_dir=str(output),
        hpo_trials=12,
    )
    calls: list[tuple[list[str], dict[str, str]]] = []
    selector_manifest_sha = __import__("hashlib").sha256(
        (selector / "manifest.json").read_bytes()
    ).hexdigest()
    selector_contract_sha = __import__("hashlib").sha256(
        (selector / "selector_feature_contract.json").read_bytes()
    ).hexdigest()

    def fake_popen(command, *, cwd, env, stdout, stderr):
        del cwd, stdout, stderr
        command = list(command)
        calls.append((command, dict(env)))
        side = command[command.index("--side") + 1]
        side_root = output / side
        pruning_contract = stage_i_tiered_pruning_contract()
        (side_root / "manifest.json").write_text(json.dumps({
            "status": "complete", "side": side, "hpo_trials": 12,
            "hpo_patience": 15, "correlation_policy": "grouped-preserve",
            "dedicated_mda_reference_mode": "full-selector-side",
            "selector_sample_manifest_sha256": selector_manifest_sha,
            "selector_feature_contract_sha256": selector_contract_sha,
            "stage_i_selector_schema": STAGE_I_SELECTOR_SCHEMA,
            "stage_i_pruning_contract": pruning_contract,
            "stage_i_pruning_contract_sha256": sha256(
                json.dumps(
                    pruning_contract, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }))
        return _FakeProcess()

    result = orchestrate_sides(
        request, memory_provider=lambda: 32 * GIB, popen_factory=fake_popen,
    )
    assert result["execution_mode"] == "parallel_two_sides"
    assert [call[0][call[0].index("--side") + 1] for call in calls] == ["long", "short"]
    assert all(call[1]["EPM_LGBM_N_JOBS"] == "4" for call in calls)
    assert result["checkpoint_roots"] == {
        "long": str(output / "long"), "short": str(output / "short"),
    }
    # Same request is a no-op after validating both completed side manifests.
    resumed = orchestrate_sides(
        request, memory_provider=lambda: 1, popen_factory=lambda *a, **k: pytest.fail("rerun"),
    )
    assert resumed["request_sha256"] == result["request_sha256"]

    # A completed orchestration is bound to the exact validated child bytes.
    long_manifest = output / "long" / "manifest.json"
    tampered = json.loads(long_manifest.read_text())
    tampered["hpo_trials"] = 999
    long_manifest.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="manifest SHA drift"):
        orchestrate_sides(request, memory_provider=lambda: 1)

    changed = SideOrchestratorRequest(
        layer="base", selector_dir=str(selector), output_dir=str(output),
        hpo_trials=13,
    )
    with pytest.raises(ValueError, match="stale side-orchestrator checkpoint"):
        orchestrate_sides(changed, memory_provider=lambda: 32 * GIB)


def test_meta_command_requires_base_root_and_cannot_override_side(tmp_path: Path) -> None:
    request = SideOrchestratorRequest(
        layer="meta", selector_dir="/selector", output_dir="/output",
        base_selection_dir="/base",
    )
    command = build_side_command(request, "short")
    assert command[command.index("--base-selection-dir") + 1] == "/base"
    assert command[command.index("--side") + 1] == "short"
    with pytest.raises(ValueError, match="may not override"):
        SideOrchestratorRequest(
            layer="base", selector_dir="/selector", output_dir="/output",
            extra_args=("--side", "long"),
        )


def test_direct_fq3_meta_command_uses_bounded_runner_and_explicit_context() -> None:
    request = SideOrchestratorRequest(
        layer="meta", selector_dir="/selector", output_dir="/output",
        base_selection_dir="/base", meta_mode="direct_fq3",
        required_regime_features=("volatility_regime",),
        required_context_features=("market_context",),
        base_candidate_fraction=1.0, target_neutral_cache_dir="/cache",
    )
    command = build_side_command(request, "long")
    assert command[1].endswith("run_stage_i_direct_fq3_meta_feature_selection.py")
    assert command[command.index("--base-candidate-fraction") + 1] == "1.0"
    assert command[command.index("--required-regime-feature") + 1] == "volatility_regime"
    assert command[command.index("--required-context-feature") + 1] == "market_context"
    assert "--resume" in command and command[command.index("--side") + 1] == "long"
    assert "--target-winner-dir" not in command
    with pytest.raises(ValueError, match="must not be forwarded"):
        SideOrchestratorRequest(
            layer="meta", selector_dir="/selector", output_dir="/output",
            base_selection_dir="/base", target_winner_dir="/winner",
            meta_mode="direct_fq3", required_regime_features=("volatility_regime",),
            required_context_features=("market_context",),
        )


def test_direct_fq3_sidecar_is_declared_and_forwarded() -> None:
    request = SideOrchestratorRequest(
        layer="meta", selector_dir="/selector", output_dir="/output",
        base_selection_dir="/base", meta_mode="direct_fq3",
        required_regime_features=("volatility_regime",), required_context_features=("market_context",),
        feature_sidecar="/latent.parquet", feature_sidecar_fields=("meta_lgbm_gmm_ood_score",),
    )
    command = build_side_command(request, "long")
    assert command[command.index("--feature-sidecar") + 1] == "/latent.parquet"
    assert command[command.index("--feature-sidecar-field") + 1] == "meta_lgbm_gmm_ood_score"
    with pytest.raises(ValueError, match="requires one or more"):
        SideOrchestratorRequest(
            layer="meta", selector_dir="/selector", output_dir="/output", base_selection_dir="/base",
            meta_mode="direct_fq3", required_regime_features=("volatility_regime",),
            required_context_features=("market_context",), feature_sidecar="/latent.parquet",
        )


def test_completed_immutable_v1_run_is_validated_and_returned_without_reinterpretation(
    tmp_path: Path,
) -> None:
    selector = _selector(tmp_path / "selector")
    output = tmp_path / "output"
    output.mkdir()
    request = SideOrchestratorRequest(
        layer="base", selector_dir=str(selector), output_dir=str(output),
        hpo_trials=12,
    )
    lineage = _lineage(request)
    child_hashes: dict[str, str] = {}
    for side in ("long", "short"):
        child = output / side / "manifest.json"
        child.parent.mkdir()
        child.write_text(json.dumps({
            "schema": "stage_i_base_feature_selection_v1",
            "status": "complete",
            "side": side,
            "hpo_trials": 12,
            "hpo_patience": 15,
            "correlation_policy": "grouped-preserve",
            "dedicated_mda_reference_mode": "full-selector-side",
            "selector_sample_manifest_sha256": lineage["selector_manifest_sha256"],
            "selector_feature_contract_sha256": lineage["selector_feature_contract_sha256"],
            "stage_i_selector_schema": LEGACY_SELECTOR_SCHEMA,
            "pruning_history": [{
                "round": 1,
                "stage_i_depth_dependent_pruning": {
                    "schema": LEGACY_PRUNING_SCHEMA,
                    "mode": "aggressive_above_boundary",
                    "boundary": 200,
                    "keep_fraction": 0.70,
                },
            }],
        }))
        child_hashes[side] = sha256(child.read_bytes()).hexdigest()
    legacy_payload = {
        "schema": LEGACY_ORCHESTRATOR_SCHEMA,
        "request": __import__("dataclasses").asdict(request),
        "lineage": lineage,
        "commands": {
            side: build_side_command(request, side) for side in ("long", "short")
        },
    }
    prior = {
        **legacy_payload,
        "request_sha256": _stable_hash(legacy_payload),
        "status": "complete",
        "side_manifest_sha256": child_hashes,
    }
    (output / "side_orchestrator_manifest.json").write_text(json.dumps(prior))

    resumed = orchestrate_sides(
        request,
        memory_provider=lambda: pytest.fail("completed v1 must not relaunch"),
        popen_factory=lambda *args, **kwargs: pytest.fail("completed v1 must not relaunch"),
    )
    assert resumed["request_sha256"] == prior["request_sha256"]
    assert resumed["side_manifest_sha256"] == prior["side_manifest_sha256"]
    assert resumed["schema"] == LEGACY_ORCHESTRATOR_SCHEMA
    assert "pruning_schema" not in resumed

    # Child SHA validity cannot hide a drifted legacy pruning mode: the
    # orchestrator delegates to the same exact shared v5-history predicate as
    # the direct base/meta CLIs.
    long_child = output / "long" / "manifest.json"
    drifted = json.loads(long_child.read_text())
    drifted["pruning_history"][0]["stage_i_depth_dependent_pruning"][
        "mode"
    ] = "different_mode"
    long_child.write_text(json.dumps(drifted))
    prior["side_manifest_sha256"]["long"] = sha256(
        long_child.read_bytes()
    ).hexdigest()
    (output / "side_orchestrator_manifest.json").write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="legacy.*resume lineage drift"):
        orchestrate_sides(request, memory_provider=lambda: 1)


def test_direct_fq3_v2_manifest_omission_is_accepted_only_with_v6_pruning_lineage(
    tmp_path: Path,
) -> None:
    selector = _selector(tmp_path / "selector")
    output = tmp_path / "output"
    request = SideOrchestratorRequest(
        layer="meta", meta_mode="direct_fq3", selector_dir=str(selector),
        output_dir=str(output), base_selection_dir=str(tmp_path / "base"), hpo_trials=12,
        required_regime_features=("volatility_regime",),
        required_context_features=("market_context",),
    )
    lineage = _lineage(request)
    lineage["base_side_manifests"] = {"short": "b" * 64}
    child = output / "short" / "manifest.json"
    child.parent.mkdir(parents=True)
    pruning_contract = stage_i_tiered_pruning_contract()
    payload = {
        "schema": "stage_i_adapter_meta_feature_selection_v2",
        "status": "complete", "side": "short", "hpo_trials": 12,
        "hpo_patience": 15, "correlation_policy": "grouped-preserve",
        "dedicated_mda_reference_mode": "full-selector-side",
        "selector_sample_manifest_sha256": lineage["selector_manifest_sha256"],
        "selector_feature_contract_sha256": lineage["selector_feature_contract_sha256"],
        "base_selector_manifest_sha256": lineage["base_side_manifests"]["short"],
        "stage_i_pruning_contract": pruning_contract,
        "stage_i_pruning_contract_sha256": sha256(
            json.dumps(pruning_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        # Regression fixture: old direct-FQ3 writer omitted this one field.
    }
    child.write_text(json.dumps(payload))
    assert _validate_completed_child(
        request=request, side="short", child=child, lineage=lineage,
        expected_sha256=sha256(child.read_bytes()).hexdigest(),
    ) == sha256(child.read_bytes()).hexdigest()

    payload["schema"] = "unrecognized_direct_meta_manifest"
    child.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stage_i_selector_schema"):
        _validate_completed_child(
            request=request, side="short", child=child, lineage=lineage,
            expected_sha256=sha256(child.read_bytes()).hexdigest(),
        )


def test_orchestrator_interrupt_terminates_all_workers_and_persists_state(tmp_path: Path) -> None:
    selector = _selector(tmp_path / "selector")
    output = tmp_path / "output"
    request = SideOrchestratorRequest(
        layer="base", selector_dir=str(selector), output_dir=str(output), hpo_trials=12,
    )
    processes = [_InterruptProcess(interrupts=True), _InterruptProcess()]

    def fake_popen(*args, **kwargs):
        del args, kwargs
        return processes.pop(0)

    launched = []
    original = processes.copy()
    def tracked_popen(*args, **kwargs):
        process = fake_popen(*args, **kwargs)
        launched.append(process)
        return process

    with pytest.raises(KeyboardInterrupt):
        orchestrate_sides(
            request, memory_provider=lambda: 32 * GIB, popen_factory=tracked_popen,
        )
    assert len(launched) == 2
    assert all(process.terminated for process in launched)
    manifest = json.loads((output / "side_orchestrator_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["failure"]["type"] == "KeyboardInterrupt"
    for side in ("long", "short"):
        # Reopening for append proves the orchestrator closed its log handle.
        with (output / "_side_logs" / f"{side}.log").open("ab") as handle:
            handle.write(b"")


def test_parallel_failure_polls_fail_fast_and_terminates_sibling(tmp_path: Path) -> None:
    selector = _selector(tmp_path / "selector")
    output = tmp_path / "output"

    class PollProcess(_InterruptProcess):
        def __init__(self, codes):
            super().__init__()
            self.codes = iter(codes)
        def poll(self):
            return next(self.codes, None)

    long = PollProcess([None, None, None])
    short = PollProcess([7])
    queue = [long, short]
    with pytest.raises(RuntimeError, match="worker failure"):
        orchestrate_sides(
            SideOrchestratorRequest(
                layer="base", selector_dir=str(selector), output_dir=str(output),
            ),
            memory_provider=lambda: 32 * GIB,
            popen_factory=lambda *args, **kwargs: queue.pop(0),
        )
    assert long.terminated
    manifest = json.loads((output / "side_orchestrator_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["return_codes"]["short"] == 7
