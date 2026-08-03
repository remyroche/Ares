import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"


def _sealed(path: Path) -> bool:
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() == (path / "manifest.sha256").read_text().split()[0]


def test_score_control_and_preregistration_bind_implementation_hashes():
    control = ART / "pre2026_oof_model_failure_incremental_value_score_control_20260730_v2"
    prereg = ART / "frozen_2026_failure_value_correction_preregistration_20260730_v2"
    assert _sealed(control) and _sealed(prereg)
    cc = json.loads((control / "contract.json").read_text())
    pc = json.loads((prereg / "contract.json").read_text())
    assert len(cc["implementation_sha256"]) == 2
    assert len(pc["implementation_sha256"]) == 1
    assert pc["application"]["authorized"] is False
    assert pc["context_incremental_gate"]["all_context_heads_rejected"] is True
