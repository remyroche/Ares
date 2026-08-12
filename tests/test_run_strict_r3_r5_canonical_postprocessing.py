from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_strict_r3_r5_canonical_postprocessing.py"


def test_postprocessing_funnel_wires_all_canonical_stages() -> None:
    text = SCRIPT.read_text()
    for script in (
        "ablate_strict_r3_cell_day_bayesian_ev_mapping.py",
        "run_strict_r3_cell_day_residual_trust_walkforward.py",
        "replay_strict_r3_forward_portfolio.py",
        "report_strict_r3_r5_canonical_waterfall.py",
    ):
        assert script in text
    assert '"--window-days", "28"' in text
    assert '"--schema", "current-v5"' in text
    assert '"--geometry-mode", "frozen"' in text
    assert '"--perp-leverage", "7"' in text
    assert '"--margin-slot-wallet-fraction", "0.10"' in text
    assert '"--disable-canonical-n5"' in text


def test_postprocessing_uses_posterior_integration_for_admission() -> None:
    text = SCRIPT.read_text()
    assert "strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1.json" in text
    assert '"--cell-day-trust-oof-predictions"' in text
    assert '"--cell-day-trust-integration"' in text
    assert "score_and_cell_day_admission_provenance.parquet" in text

