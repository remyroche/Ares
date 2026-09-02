from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_canonical_sr_e2_mc1_input_ablation.py"


def _module():
    return ast.parse(RUNNER.read_text(encoding="utf-8"))


def test_runner_declares_dual_50_family_map_contract() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "ADMISSION_BPS = 50.0" in text
    assert "C0_refit_core_postfeb" in text
    assert "C1_refit_core_plus_causal_sr" in text
    assert "C2_refit_core_plus_15m_e2" in text
    assert "C3_refit_core_plus_causal_sr_15m_e2" in text
    assert "BCF MC1 >= +50 AND current-v5 MC1 >= +50" in text
    assert "C4_refit_core_plus_causal_sr_oi_positioning" in text
    assert "strictly prior OI observations" in text


def test_target_free_view_rejects_policy_fields() -> None:
    function = next(
        node for node in ast.walk(_module())
        if isinstance(node, ast.FunctionDef) and node.name == "_target_free_prediction_view"
    )
    text = ast.unparse(function)
    assert "POLICY_FORBIDDEN.intersection(result.columns)" in text
    assert '"policy_net_bps"' not in text


def test_optional_oi_positioning_merge_preserves_candidate_identity() -> None:
    function = next(
        node for node in ast.walk(_module())
        if isinstance(node, ast.FunctionDef) and node.name == "_merge_causal_oi_positioning"
    )
    text = ast.unparse(function)
    assert "validate='one_to_one'" in text
    assert "changed target-free candidate identity or order" in text
    assert "POLICY_FORBIDDEN" not in text


def test_prequential_e2_code_uses_prior_resolved_labels() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'prepared["policy_label_available_ts"].lt(month)' in text
    assert 'prepared["__decision_ts__"].lt(month)' in text
    assert "E2_TRAIN_MONTHS = 4" in text
    assert 'full["day"] = full["__decision_ts__"].dt.normalize()' in text
    assert 'end - pd.Timedelta(nanoseconds=1)' in text


def test_no_live_execution_dependency() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    prohibited = ("submit_order", "kraken_private", "exchange-writing", "run_live")
    assert not any(token in text.lower() for token in prohibited)
