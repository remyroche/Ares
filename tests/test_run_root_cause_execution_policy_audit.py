from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_root_cause_execution_policy_audit import execution_waterfall, policy_regret, run


def _ledger(n: int = 20) -> pd.DataFrame:
    gross = np.arange(n, dtype=float) - 10.0
    fee = np.full(n, 2.0)
    return pd.DataFrame({
        "candidate_id": [f"id-{i}" for i in range(n)],
        "execution_adjusted_gross_h12_bps": gross,
        "fee_bps": fee,
        "net_h12_bps": gross - fee,
        "fixed_ex_ante_fee_bps": np.full(n, 1.0),
        "score_base_alpha": np.arange(n, dtype=float),
        "score_residual_alpha": np.arange(n, dtype=float)[::-1],
        "action_target_status": ["AVAILABLE_GROSS_ARMS_FIXED_EX_ANTE_FEE_TARGET_ONLY"] * 4 + ["UNAVAILABLE"] * (n - 4),
        "action_continue_execution_adjusted_gross_bps": [1., 2., 3., 4.] + [np.nan] * (n - 4),
        "action_exit_execution_adjusted_gross_bps": [2., 1., 5., 0.] + [np.nan] * (n - 4),
    })


def test_waterfall_fails_closed_for_unavailable_counterfactuals():
    result = execution_waterfall(_ledger())
    missing = result[result.stage.isin(["A_reference_ideal_entry_gross", "C_delayed_entry_gross"])]
    assert missing.value_bps_per_candidate.isna().all()
    assert missing.status.str.startswith("NOT_RUN").all()
    assert not (result.stage.eq("entry_transfer_loss_A_minus_B") & result.value_bps_per_candidate.eq(0)).any()


def test_top10_is_global_and_heads_are_separate():
    result = execution_waterfall(_ledger())
    top = result[result.record_type.eq("global_top10_observed")]
    assert set(top.score) == {"score_base_alpha", "score_residual_alpha"}
    assert set(top.rows) == {2}
    assert set(top.slice) == {"global_top10"}


def test_top10_ties_do_not_expand_exact_global_support():
    ledger = _ledger()
    ledger["score_base_alpha"] = 1.0
    result = execution_waterfall(ledger)
    top = result[(result.record_type == "global_top10_observed") & (result.score == "score_base_alpha")]
    assert set(top.rows) == {2}


def test_policy_reports_conditional_and_complete_population_without_action_head():
    result = policy_regret(_ledger())
    assert set(result.population) == {"conditional_eligible", "complete_upstream_population", "policy_control_gate"}
    oracle = result[(result.population == "conditional_eligible") & (result.policy == "hindsight_oracle")].iloc[0]
    assert oracle.gross_bps_per_candidate == 3.25
    learned = result[result.policy.str.contains("learned_action")]
    assert learned.status.eq("NOT_RUN_TWO_HEAD_SCOPE").all()
    controls = result[result.population == "policy_control_gate"]
    assert "independent_prefix_recomputation" in set(controls.policy)
    assert controls.loc[controls.policy == "later_sealed_oos", "status"].item() == "NOT_RUN_TWO_HEAD_SCOPE"


def test_end_to_end_reconciliation_and_artifacts(tmp_path: Path):
    source = tmp_path / "ledger.parquet"
    output = tmp_path / "out"
    _ledger().to_parquet(source, index=False)
    manifest = run(source, output)
    assert manifest["checks"]["gross_fee_net_reconciles"]
    assert manifest["selection_semantics"] == "GLOBAL_TOP_K_NOT_PER_TIMESTAMP"
    for name in ("execution_waterfall.parquet", "policy_regret.parquet", "correctness_test_report.json", "run_manifest.json"):
        assert (output / name).exists()
