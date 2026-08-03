from pathlib import Path
import json


def test_published_residual_gate_has_warmup_and_strict_oof() -> None:
    root = Path(__file__).resolve().parents[1]
    gate = json.loads((root / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/coverage_economics_gate.json").read_text())
    assert gate["warmup_rows"] > 0
    assert gate["strict_oof_rows"] > 0
    assert gate["rows"] == gate["warmup_rows"] + gate["strict_oof_rows"]
    assert gate["residual_metrics_identical_rows"]["rows"] == gate["strict_oof_rows"]
