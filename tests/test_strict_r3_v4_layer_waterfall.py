from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_strict_r3_v4_layer_waterfall.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_v4_layer_waterfall", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _manifest(*, severe_target: str = "none", alpha: float = 0.0) -> dict[str, object]:
    return {
        "score_override_arm": "conditional_none",
        "overlay_arms": [{
            "name": "correctness_top30_k9temp025_no_memberships",
            "severe_target": severe_target,
            "severe_alpha": alpha,
            "use_correctness": True,
            "correctness_training_fraction": 0.30,
            "k9_soft_memberships": False,
        }],
    }


def test_selected_waterfall_contract_requires_correctness_only_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "predictions.parquet"
    source.touch()
    (tmp_path / "run_manifest.json").write_text(json.dumps(_manifest()))
    monkeypatch.setitem(MODULE.SOURCE, 2025, source)

    contract = MODULE._load_selected_conversion_contract(2025)

    assert contract["severe_active"] is False
    assert contract["score_modulator"] == "top30_policy_residual_correctness"


def test_selected_waterfall_contract_rejects_active_severe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "predictions.parquet"
    source.touch()
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(_manifest(severe_target="h12", alpha=0.5)),
    )
    monkeypatch.setitem(MODULE.SOURCE, 2025, source)

    with pytest.raises(ValueError, match="severe_target"):
        MODULE._load_selected_conversion_contract(2025)
