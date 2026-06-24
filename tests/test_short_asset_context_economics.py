from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import diagnose_short_asset_context_economics as mod


def test_gate_economics_counts_loser_avoided_against_winner_cost() -> None:
    ret = np.asarray([0.10, -0.04, 0.03, -0.02, 0.00], dtype=np.float32)
    reject = np.asarray([True, True, False, False, True])

    econ = mod._gate_economics(ret, reject)

    assert econ["rows"] == 5
    assert econ["rejected_rows"] == 3
    assert np.isclose(econ["loser_loss_avoided_sum"], 0.04)
    assert np.isclose(econ["winner_profit_sacrificed_sum"], 0.10)
    assert np.isclose(econ["net_benefit"], -0.06)


def test_requirement_audit_exposes_negative_action_result() -> None:
    risk_metrics = {
        "rows": 100,
        "coverage": 0.8,
        "tail_loss_auc": 0.64,
        "tail_loss_pr_auc": 0.15,
    }
    two_d = pd.DataFrame({"cell": range(50)})
    decile_compare = pd.DataFrame(
        {
            "high_minus_low_mean_return": np.ones(10),
            "high_minus_low_hit_rate": -np.ones(10),
            "high_minus_low_lower_tail_q05": -np.ones(10),
        }
    )
    target_tests = pd.DataFrame(
        {
            "delta_rmse_improvement": [0.1, 0.1, 0.1, 0.1],
            "delta_mae_improvement": [0.1, 0.1, 0.1, 0.1],
        }
    )
    residual_tests = pd.DataFrame(
        {
            "delta_rmse_improvement": [0.1, 0.1],
            "delta_mae_improvement": [-0.1, -0.1],
        }
    )
    rules = pd.DataFrame(
        {
            "net_benefit": [-1.0, -2.0, -3.0],
            "bad_episodes_positive_net": [0, 0, 0],
        }
    )
    frontier = pd.DataFrame(
        {
            "retained_coverage": [0.99, 0.975, 0.95, 0.90, 0.80],
            "net_benefit": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    )

    audit = mod._requirement_audit(
        risk_metrics=risk_metrics,
        two_d=two_d,
        decile_compare=decile_compare,
        target_tests=target_tests,
        residual_tests=residual_tests,
        rules=rules,
        rule_episodes=pd.DataFrame({"row": range(12)}),
        frontier=frontier,
    )

    assert len(audit) == 6
    assert audit.loc[audit["step"].eq("predeclared_conditional_actions"), "status"].item() == "completed_rejected"
    assert "pooled_positive_net_rules=0/3" in audit.loc[
        audit["step"].eq("predeclared_conditional_actions"), "primary_metrics"
    ].item()
    assert audit.loc[audit["step"].eq("economic_frontier"), "status"].item() == "completed_rejected"
    assert "positive_net_points=0/5" in audit.loc[audit["step"].eq("economic_frontier"), "primary_metrics"].item()
