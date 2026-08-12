"""Bounded side-local Stage-I selector orchestration.

Long and short remain independent processes/checkpoint roots.  Parallelism is
limited to two workers and four LightGBM threads each; a deterministic memory
preflight falls back to sequential execution before either worker is started.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from .stage_i_feature_selection import (
    legacy_stage_i_v5_pruning_history_is_exact,
)


GIB = 1024**3
ORCHESTRATOR_SCHEMA = "stage_i_bounded_side_orchestrator_v2"
SELECTOR_SCHEMA = "stage_i_grouped_stability_mda_v6"
PRUNING_SCHEMA = "stage_i_tiered_round_pruning_v2"
LEGACY_ORCHESTRATOR_SCHEMA = "stage_i_bounded_side_orchestrator_v1"
LEGACY_SELECTOR_SCHEMA = "stage_i_grouped_stability_mda_v5"
LEGACY_PRUNING_SCHEMA = "stage_i_depth_dependent_pruning_v1"
# The first direct-FQ3 runner revision wrote this adapter schema and the full
# v6 pruning contract, but accidentally omitted ``stage_i_selector_schema``
# from its child manifest.  Keep compatibility narrowly scoped to that exact,
# already-completed artifact shape; any other missing selector schema remains
# a hard lineage error.
DIRECT_FQ3_SELECTOR_MANIFEST_SCHEMA = "stage_i_adapter_meta_feature_selection_v2"
DIRECT_FQ3_RESUME_COMPLETE_SCHEMA = "stage_i_direct_fq3_resume_complete_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def available_memory_bytes() -> int:
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:
        # Unknown memory is handled conservatively by the caller.
        return 0
    # macOS may report a large reclaimable cache while the compressed-memory
    # subsystem is already consuming almost all swap. A new selector then
    # thrashes or fails after launch despite the nominal psutil value. Treat
    # less than 1 GiB of swap headroom as no safe capacity for a fresh child.
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "vm.swapusage"], check=True, capture_output=True,
                text=True, timeout=2,
            )
            match = re.search(r"free\s+=\s+([0-9.]+)M", result.stdout)
            if match and float(match.group(1)) < 1024.0:
                return 0
        except (OSError, subprocess.SubprocessError, ValueError):
            # psutil remains the portable fallback if macOS cannot disclose
            # swap state (for example under a restricted test sandbox).
            pass
    return available


def estimate_side_worker_bytes(selector_dir: str | Path) -> int:
    root = Path(selector_dir)
    compressed = sum(
        path.stat().st_size
        for name in ("selector_features.parquet", "selector_ledger.parquet")
        if (path := root / name).is_file()
    )
    # Parquet decompression, float32 working copies, LightGBM bins, path index,
    # and permutation batches coexist.  Four times compressed size plus a 4GiB
    # floor is intentionally conservative and deterministic.
    return max(4 * GIB, 4 * compressed)


def choose_execution_mode(
    *, available_bytes: int, worker_bytes: int, reserve_bytes: int = 4 * GIB,
) -> str:
    required = int(reserve_bytes) + 2 * int(worker_bytes)
    return "parallel_two_sides" if int(available_bytes) >= required else "sequential_memory_fallback"


def require_safe_single_worker_capacity(
    *, available_bytes: int, worker_bytes: int, reserve_bytes: int = 4 * GIB,
) -> None:
    """Fail before launch when even the sequential fallback is unsafe."""
    required = int(reserve_bytes) + int(worker_bytes)
    if int(available_bytes) < required:
        raise RuntimeError(
            "insufficient safe memory for one Stage-I selector worker: "
            f"available={int(available_bytes)}, required={required}; "
            "wait for memory/swap recovery rather than overcommitting"
        )


@dataclass(frozen=True)
class SideOrchestratorRequest:
    layer: str
    selector_dir: str
    output_dir: str
    base_selection_dir: str | None = None
    target_winner_dir: str | None = None
    meta_mode: str = "legacy"
    required_regime_features: tuple[str, ...] = ()
    required_context_features: tuple[str, ...] = ()
    base_candidate_fraction: float = 1.0
    target_neutral_cache_dir: str | None = None
    feature_sidecar: str | None = None
    feature_sidecar_fields: tuple[str, ...] = ()
    feature_sidecar_min_coverage: float = 0.90
    hpo_trials: int = 60
    hpo_patience: int = 15
    correlation_policy: str = "grouped-preserve"
    dedicated_mda_reference: str = "full-selector-side"
    mda_support_mode: str = "full"
    threads_per_worker: int = 4
    reserve_gib: float = 4.0
    worker_memory_gib: float | None = None
    extra_args: tuple[str, ...] = ()
    # A restart may need to rerun only an interrupted side.  Keeping this in
    # the immutable request makes that narrow run auditable, rather than
    # recreating a completed opposite-side selector just to satisfy the
    # bounded parent.
    sides: tuple[str, ...] = ("long", "short")

    def __post_init__(self) -> None:
        if self.layer not in {"base", "meta"}:
            raise ValueError("layer must be base or meta")
        if self.layer == "meta" and not self.base_selection_dir:
            raise ValueError("meta orchestration requires base_selection_dir")
        if self.meta_mode not in {"legacy", "direct_fq3"}:
            raise ValueError("meta_mode must be legacy or direct_fq3")
        if self.meta_mode == "direct_fq3" and (
            self.layer != "meta" or not self.required_regime_features or not self.required_context_features
        ):
            raise ValueError("direct_fq3 meta orchestration requires explicit regime/context fields")
        if self.meta_mode == "direct_fq3" and self.target_winner_dir:
            raise ValueError(
                "direct_fq3 meta consumes the target contract from its completed base selector; "
                "target_winner_dir must not be forwarded to the meta child"
            )
        if self.feature_sidecar and (self.layer != "meta" or self.meta_mode != "direct_fq3"):
            raise ValueError("feature_sidecar is supported only for direct_fq3 meta orchestration")
        if self.feature_sidecar and not self.feature_sidecar_fields:
            raise ValueError("feature_sidecar requires one or more declared feature_sidecar_fields")
        if self.feature_sidecar_fields and not self.feature_sidecar:
            raise ValueError("feature_sidecar_fields require feature_sidecar")
        if not 0.0 < float(self.feature_sidecar_min_coverage) <= 1.0:
            raise ValueError("feature_sidecar_min_coverage must lie in (0,1]")
        if self.mda_support_mode not in {"full", "target-only"}:
            raise ValueError("mda_support_mode must be full or target-only")
        if int(self.threads_per_worker) != 4:
            raise ValueError("Stage-I side orchestration is bounded at four threads per worker")
        if not self.sides or any(side not in {"long", "short"} for side in self.sides):
            raise ValueError("sides must contain one or both of long and short")
        if len(set(self.sides)) != len(self.sides):
            raise ValueError("sides must not contain duplicates")
        forbidden = {"--side", "--output-dir", "--selector-dir", "--base-selection-dir"}
        if any(str(value) in forbidden for value in self.extra_args):
            raise ValueError("extra_args may not override side/input/output isolation")


def _lineage(request: SideOrchestratorRequest) -> dict[str, Any]:
    selector = Path(request.selector_dir)
    required = [selector / "manifest.json", selector / "selector_feature_contract.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selector lineage is incomplete: {missing}")
    lineage: dict[str, Any] = {
        "selector_manifest_sha256": _sha256(required[0]),
        "selector_feature_contract_sha256": _sha256(required[1]),
    }
    features = selector / "selector_features.parquet"
    if features.is_file():
        lineage["selector_features_size"] = features.stat().st_size
        lineage["selector_features_mtime_ns"] = features.stat().st_mtime_ns
    if request.base_selection_dir:
        base = Path(request.base_selection_dir)
        lineage["base_side_manifests"] = {
            side: _sha256(base / side / "manifest.json")
            for side in request.sides
            if (base / side / "manifest.json").is_file()
        }
    if request.target_winner_dir:
        target = Path(request.target_winner_dir)
        lineage["target_winner_manifests"] = {
            str(path.relative_to(target)): _sha256(path)
            for path in sorted(target.rglob("*.json"))
            if path.is_file()
        }
    if request.feature_sidecar:
        sidecar = Path(request.feature_sidecar)
        if not sidecar.is_file():
            raise FileNotFoundError(f"feature sidecar is missing: {sidecar}")
        lineage["feature_sidecar"] = {
            "path": str(sidecar.resolve()), "sha256": _sha256(sidecar),
            "fields": list(request.feature_sidecar_fields),
            "minimum_coverage": float(request.feature_sidecar_min_coverage),
        }
    return lineage


def build_side_command(request: SideOrchestratorRequest, side: str) -> list[str]:
    script = (
        "run_stage_i_base_feature_selection.py" if request.layer == "base"
        else "run_stage_i_direct_fq3_meta_feature_selection.py"
        if request.meta_mode == "direct_fq3"
        else "run_stage_i_meta_feature_selection.py"
    )
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(root / "scripts" / script),
        "--selector-dir", request.selector_dir,
        "--output-dir", request.output_dir,
        "--side", str(side),
        "--hpo-trials", str(int(request.hpo_trials)),
        "--hpo-patience", str(int(request.hpo_patience)),
        "--correlation-policy", request.correlation_policy,
        "--dedicated-mda-reference", request.dedicated_mda_reference,
        "--mda-support-mode", request.mda_support_mode,
        "--resume",
    ]
    if request.base_selection_dir:
        command.extend(["--base-selection-dir", request.base_selection_dir])
    if request.target_winner_dir and request.meta_mode != "direct_fq3":
        command.extend(["--target-winner-dir", request.target_winner_dir])
    if request.meta_mode == "direct_fq3":
        command.extend(["--base-candidate-fraction", str(float(request.base_candidate_fraction))])
        if request.target_neutral_cache_dir:
            command.extend(["--target-neutral-cache-dir", request.target_neutral_cache_dir])
        for feature in request.required_regime_features:
            command.extend(["--required-regime-feature", feature])
        for feature in request.required_context_features:
            command.extend(["--required-context-feature", feature])
        if request.feature_sidecar:
            command.extend(["--feature-sidecar", request.feature_sidecar])
            command.extend(["--feature-sidecar-min-coverage", str(float(request.feature_sidecar_min_coverage))])
            for feature in request.feature_sidecar_fields:
                command.extend(["--feature-sidecar-field", feature])
    command.extend(map(str, request.extra_args))
    return command


def _validate_completed_child(
    *, request: SideOrchestratorRequest, side: str, child: Path,
    lineage: Mapping[str, Any], expected_sha256: str | None = None,
) -> str:
    """Replicate the immutable part of each child CLI's resume contract."""
    # Direct-FQ3 clean restarts retain an interrupted root as evidence and
    # publish a hash-bound pointer only after an isolated attempt completes.
    # Do not accept an arbitrary attempt directory or partial manifest.
    if not child.is_file() and request.layer == "meta" and request.meta_mode == "direct_fq3":
        root = child.parent
        pointer_path = root / "resume_complete.json"
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text())
            if pointer.get("schema") != DIRECT_FQ3_RESUME_COMPLETE_SCHEMA or pointer.get("side") != side:
                raise ValueError(f"completed {side} resume pointer lineage is invalid")
            relative = pointer.get("attempt_relative_path")
            if not isinstance(relative, str):
                raise ValueError(f"completed {side} resume pointer lacks attempt path")
            attempt_root = (root / relative).resolve()
            allowed_root = (root / "_resume_attempts").resolve()
            if allowed_root not in attempt_root.parents or attempt_root.parent != allowed_root:
                raise ValueError(f"completed {side} resume pointer escapes attempt root")
            candidate = attempt_root / "manifest.json"
            if not candidate.is_file() or _sha256(candidate) != pointer.get("attempt_manifest_sha256"):
                raise ValueError(f"completed {side} resume attempt manifest SHA drift")
            child = candidate
    if not child.is_file():
        raise ValueError(f"completed {side} child manifest is missing")
    actual_sha256 = _sha256(child)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ValueError(f"completed {side} child manifest SHA drift")
    try:
        payload = json.loads(child.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"completed {side} child manifest is unreadable") from exc
    actual_selector_schema = payload.get("stage_i_selector_schema")
    if (
        actual_selector_schema is None
        and request.layer == "meta"
        and request.meta_mode == "direct_fq3"
        and payload.get("schema") == DIRECT_FQ3_SELECTOR_MANIFEST_SCHEMA
        and isinstance(payload.get("stage_i_pruning_contract"), Mapping)
        and payload["stage_i_pruning_contract"].get("schema") == PRUNING_SCHEMA
    ):
        # This is a manifest-field omission, not a semantic reinterpretation:
        # the v6 pruning schema already pins the selector implementation.
        actual_selector_schema = SELECTOR_SCHEMA
    expected = {
        "status": "complete",
        "side": side,
        "hpo_trials": int(request.hpo_trials),
        "hpo_patience": int(request.hpo_patience),
        "correlation_policy": request.correlation_policy,
        "dedicated_mda_reference_mode": request.dedicated_mda_reference,
        "mda_support_mode": request.mda_support_mode,
        "selector_sample_manifest_sha256": lineage["selector_manifest_sha256"],
        "selector_feature_contract_sha256": lineage["selector_feature_contract_sha256"],
        "stage_i_selector_schema": SELECTOR_SCHEMA,
    }
    def _actual(key: str) -> Any:
        if key == "stage_i_selector_schema":
            return actual_selector_schema
        # ``full`` preserves legacy selector semantics.  Accept an old
        # completed full-support artifact during resume, while a target-only
        # request always requires its explicit manifest marker.
        if (
            key == "mda_support_mode"
            and payload.get(key) is None
            and request.mda_support_mode == "full"
        ):
            return "full"
        return payload.get(key)

    mismatches = {
        key: {"expected": value, "actual": _actual(key)}
        for key, value in expected.items()
        if _actual(key) != value
    }
    if request.layer == "meta":
        base_sha = (lineage.get("base_side_manifests") or {}).get(side)
        if base_sha is None or payload.get("base_selector_manifest_sha256") != base_sha:
            mismatches["base_selector_manifest_sha256"] = {
                "expected": base_sha,
                "actual": payload.get("base_selector_manifest_sha256"),
            }
    pruning_contract = payload.get("stage_i_pruning_contract")
    if (
        not isinstance(pruning_contract, Mapping)
        or pruning_contract.get("schema") != PRUNING_SCHEMA
        or payload.get("stage_i_pruning_contract_sha256")
        != _stable_hash(pruning_contract)
    ):
        mismatches["stage_i_pruning_contract"] = {
            "expected_schema": PRUNING_SCHEMA,
            "actual": pruning_contract,
        }
    if mismatches:
        raise ValueError(f"completed {side} child resume lineage drift: {mismatches}")
    return actual_sha256


def _validate_completed_legacy_v1_child(
    *, request: SideOrchestratorRequest, side: str, child: Path,
    lineage: Mapping[str, Any], expected_sha256: str,
) -> None:
    """Validate, but never reinterpret, a completed in-flight v1 child."""

    if not child.is_file() or _sha256(child) != str(expected_sha256):
        raise ValueError(f"completed legacy {side} child manifest SHA drift")
    payload = json.loads(child.read_text())
    expected = {
        "status": "complete",
        "side": side,
        "hpo_trials": int(request.hpo_trials),
        "hpo_patience": int(request.hpo_patience),
        "correlation_policy": request.correlation_policy,
        "dedicated_mda_reference_mode": request.dedicated_mda_reference,
        "selector_sample_manifest_sha256": lineage["selector_manifest_sha256"],
        "selector_feature_contract_sha256": lineage["selector_feature_contract_sha256"],
        "stage_i_selector_schema": LEGACY_SELECTOR_SCHEMA,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items() if payload.get(key) != value
    }
    if not legacy_stage_i_v5_pruning_history_is_exact(
        payload.get("pruning_history")
    ):
        mismatches["pruning_history"] = "legacy_v5_exact_contract_drift"
    if request.layer == "meta":
        base_sha = (lineage.get("base_side_manifests") or {}).get(side)
        if base_sha is None or payload.get("base_selector_manifest_sha256") != base_sha:
            mismatches["base_selector_manifest_sha256"] = {
                "expected": base_sha, "actual": payload.get("base_selector_manifest_sha256"),
            }
    if mismatches:
        raise ValueError(
            f"completed legacy {side} child resume lineage drift: {mismatches}"
        )


def orchestrate_sides(
    request: SideOrchestratorRequest,
    *,
    memory_provider: Callable[[], int] = available_memory_bytes,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> Mapping[str, Any]:
    sides = tuple(request.sides)
    output = Path(request.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "_side_logs"
    logs.mkdir(exist_ok=True)
    lineage = _lineage(request)
    request_payload = {
        "schema": ORCHESTRATOR_SCHEMA,
        "selector_schema": SELECTOR_SCHEMA,
        "pruning_schema": PRUNING_SCHEMA,
        "request": asdict(request),
        "lineage": lineage,
        "commands": {
            side: build_side_command(request, side) for side in sides
        },
    }
    request_sha = _stable_hash(request_payload)
    manifest_path = output / "side_orchestrator_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if (
            prior.get("schema") == LEGACY_ORCHESTRATOR_SCHEMA
            and prior.get("status") == "complete"
        ):
            legacy_payload = {
                "schema": LEGACY_ORCHESTRATOR_SCHEMA,
                "request": asdict(request),
                "lineage": lineage,
                "commands": {
                    side: build_side_command(request, side)
                    for side in sides
                },
            }
            if prior.get("request_sha256") != _stable_hash(legacy_payload):
                raise ValueError("stale completed legacy side-orchestrator request hash")
            stored_hashes = prior.get("side_manifest_sha256")
            if not isinstance(stored_hashes, Mapping):
                raise ValueError("complete legacy orchestrator lacks child manifest hashes")
            for side in sides:
                _validate_completed_legacy_v1_child(
                    request=request,
                    side=side,
                    child=output / side / "manifest.json",
                    lineage=lineage,
                    expected_sha256=str(stored_hashes.get(side, "")),
                )
            # Immutable v1 artifacts remain v1.  This is a no-op compatibility
            # return only; partial v1 checkpoints are never adopted by v2.
            return prior
        if prior.get("request_sha256") != request_sha:
            raise ValueError("stale side-orchestrator checkpoint request hash")
        if prior.get("status") == "complete":
            stored_hashes = prior.get("side_manifest_sha256")
            if not isinstance(stored_hashes, Mapping):
                raise ValueError("complete orchestrator manifest lacks child manifest hashes")
            for side in sides:
                child = output / side / "manifest.json"
                _validate_completed_child(
                    request=request, side=side, child=child, lineage=lineage,
                    expected_sha256=stored_hashes.get(side),
                )
            return prior

    worker_bytes = (
        int(float(request.worker_memory_gib) * GIB)
        if request.worker_memory_gib is not None
        else estimate_side_worker_bytes(request.selector_dir)
    )
    available = int(memory_provider())
    require_safe_single_worker_capacity(
        available_bytes=available,
        worker_bytes=worker_bytes,
        reserve_bytes=int(float(request.reserve_gib) * GIB),
    )
    mode = (
        "single_side"
        if len(sides) == 1
        else choose_execution_mode(
            available_bytes=available,
            worker_bytes=worker_bytes,
            reserve_bytes=int(float(request.reserve_gib) * GIB),
        )
    )
    manifest: dict[str, Any] = {
        **request_payload,
        "request_sha256": request_sha,
        "status": "running",
        "execution_mode": mode,
        "memory_preflight": {
            "available_bytes": available, "estimated_worker_bytes": worker_bytes,
            "reserve_bytes": int(float(request.reserve_gib) * GIB),
            "parallel_required_bytes": int(float(request.reserve_gib) * GIB) + 2 * worker_bytes,
        },
        "threads_per_worker": 4,
        "checkpoint_roots": {side: str(output / side) for side in sides},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update({
        "EPM_LGBM_N_JOBS": "4", "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "4",
    })

    def launch(side: str) -> tuple[Any, Any]:
        side_root = output / side
        if side_root.exists() and any(side_root.iterdir()) and not (
            side_root / "orchestrator_request.json"
        ).is_file():
            raise ValueError(
                f"unbound pre-existing {side} checkpoint root; refusing to adopt stale work"
            )
        side_root.mkdir(exist_ok=True)
        side_request = side_root / "orchestrator_request.json"
        if side_request.is_file():
            prior = json.loads(side_request.read_text())
            if prior.get("request_sha256") != request_sha:
                raise ValueError(f"stale {side} checkpoint request hash")
        else:
            side_request.write_text(json.dumps({
                "schema": ORCHESTRATOR_SCHEMA,
                "side": side, "request_sha256": request_sha,
                "command": request_payload["commands"][side],
            }, indent=2) + "\n")
        log_handle = (logs / f"{side}.log").open("ab")
        process = popen_factory(
            request_payload["commands"][side],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        )
        return process, log_handle

    def completed_direct_fq3_resume(side: str) -> bool:
        """Recognise a hash-verified clean-resume child without relaunching it.

        A direct-FQ3 child can retain an earlier side receipt as evidence and
        publish its completed manifest under ``_resume_attempts``.  Its receipt
        deliberately has the child-runner request hash rather than the parent
        orchestrator hash, so trying to launch it again would reject a valid
        completed artifact as stale before the pointer-aware validator runs.
        """
        if request.layer != "meta" or request.meta_mode != "direct_fq3":
            return False
        root = output / side
        if not (root / "resume_complete.json").is_file():
            return False
        _validate_completed_child(
            request=request,
            side=side,
            child=root / "manifest.json",
            lineage=lineage,
        )
        return True

    return_codes: dict[str, int] = {}
    active: dict[str, tuple[Any, Any]] = {}

    def stop_active_workers() -> None:
        for process, _handle in active.values():
            try:
                poll = getattr(process, "poll", None)
                if callable(poll) and poll() is not None:
                    continue
                terminate = getattr(process, "terminate", None)
                if callable(terminate):
                    terminate()
            except BaseException:
                pass
        for process, _handle in active.values():
            try:
                process.wait(timeout=10)
            except TypeError:
                # Lightweight test/process adapters may expose wait() only.
                try:
                    process.wait()
                except BaseException:
                    pass
            except subprocess.TimeoutExpired:
                try:
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()
                    process.wait(timeout=10)
                except BaseException:
                    pass
            except BaseException:
                pass

    def close_all_logs() -> None:
        for _process, handle in active.values():
            try:
                if not handle.closed:
                    handle.close()
            except Exception:
                pass

    try:
        if mode == "parallel_two_sides":
            for side in sides:
                if completed_direct_fq3_resume(side):
                    return_codes[side] = 0
                else:
                    active[side] = launch(side)
            if all(callable(getattr(process, "poll", None)) for process, _ in active.values()):
                pending = set(active)
                while pending:
                    for side in tuple(pending):
                        process, handle = active[side]
                        code = process.poll()
                        if code is None:
                            continue
                        return_codes[side] = int(code)
                        pending.remove(side)
                        handle.close()
                        if int(code) != 0:
                            stop_active_workers()
                            pending.clear()
                            break
                    if pending:
                        time.sleep(0.05)
            else:
                for side, (process, handle) in active.items():
                    return_codes[side] = int(process.wait())
                    handle.close()
        else:
            for side in sides:
                if completed_direct_fq3_resume(side):
                    return_codes[side] = 0
                    continue
                process, handle = launch(side)
                active[side] = (process, handle)
                return_codes[side] = int(process.wait())
                handle.close()
                if return_codes[side] != 0:
                    break
    except BaseException as exc:
        stop_active_workers()
        close_all_logs()
        manifest["return_codes"] = return_codes
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    finally:
        close_all_logs()
        manifest["return_codes"] = return_codes
    failed = {side: code for side, code in return_codes.items() if code != 0}
    if failed or len(return_codes) != len(sides):
        manifest["status"] = "failed"
        manifest["failure"] = failed or {"missing_side": "worker_not_started"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise RuntimeError(f"Stage-I side worker failure: {manifest['failure']}")
    child_hashes: dict[str, str] = {}
    for side in sides:
        child = output / side / "manifest.json"
        try:
            child_hashes[side] = _validate_completed_child(
                request=request, side=side, child=child, lineage=lineage,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{side} worker exited zero without a valid complete manifest"
            ) from exc
    manifest.update({"status": "complete", "side_manifest_sha256": child_hashes})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = [
    "ORCHESTRATOR_SCHEMA", "SideOrchestratorRequest", "available_memory_bytes",
    "build_side_command", "choose_execution_mode", "estimate_side_worker_bytes",
    "require_safe_single_worker_capacity",
    "orchestrate_sides",
]
