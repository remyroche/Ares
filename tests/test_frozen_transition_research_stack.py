from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_frozen_transition_research_stack.py"
)
SPEC = importlib.util.spec_from_file_location("frozen_stack", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _registry(tmp_path: Path, source: Path, eligible: bool = True) -> Path:
    path = tmp_path / "registry.json"
    payload = {
        "status": "FROZEN_CONTEXT_ONLY_NO_DIRECT_POLICY_CONTROL",
        "sources": {
            "source": {
                "path": source.name,
                "sha256": MODULE.sha256(source),
                "model_input_eligible": eligible,
            }
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_frozen_source_requires_exact_path_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen")
    registry = MODULE.load_registry(_registry(tmp_path, source))
    assert (
        MODULE.validate_frozen_source(
            registry, "source", source, root=tmp_path
        )
        == source.resolve()
    )
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        MODULE.validate_frozen_source(
            registry, "source", source, root=tmp_path
        )


def test_unregistered_override_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    override = tmp_path / "override.bin"
    source.write_bytes(b"frozen")
    override.write_bytes(b"frozen")
    registry = MODULE.load_registry(_registry(tmp_path, source))
    with pytest.raises(ValueError, match="path"):
        MODULE.validate_frozen_source(
            registry, "source", override, root=tmp_path
        )


def test_descriptive_hazard_cannot_be_requested_as_model_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen")
    registry = MODULE.load_registry(_registry(tmp_path, source, eligible=False))
    with pytest.raises(ValueError, match="not a model input"):
        MODULE.validate_frozen_source(
            registry,
            "source",
            source,
            root=tmp_path,
            require_model_input_eligible=True,
        )
