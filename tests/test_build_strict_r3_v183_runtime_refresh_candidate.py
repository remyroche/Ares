from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_strict_r3_v183_runtime_refresh_candidate.py"
SPEC = importlib.util.spec_from_file_location("runtime_refresh", MODULE_PATH)
assert SPEC and SPEC.loader
runtime_refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_refresh)


def test_static_overlay_comparison_ignores_only_runtime_hashes(tmp_path: Path, monkeypatch) -> None:
    base = {"schema": "strict_r3_inference_bundle_v6", "runtime": {"mode": "shadow-only"}}
    (tmp_path / "base.json").write_text('{"schema":"strict_r3_inference_bundle_v6","runtime":{"mode":"shadow-only"}}')
    monkeypatch.setattr(runtime_refresh, "ROOT", tmp_path)
    source = {
        "schema": "strict_r3_inference_bundle_overlay_v1",
        "base_bundle": "base.json",
        "overrides": {
            "runtime": {"feature_state": {"contract_sha256": "x"}},
            "runtime_code_sha256": {"module.py": "old"},
        },
    }
    candidate = {
        **source,
        "overrides": {
            **source["overrides"],
            "runtime_code_sha256": {"module.py": "new"},
        },
    }
    assert runtime_refresh._merged_static(source) == runtime_refresh._merged_static(candidate)
    assert runtime_refresh._merged_static(source)["runtime"]["feature_state"]["contract_sha256"] == "x"
