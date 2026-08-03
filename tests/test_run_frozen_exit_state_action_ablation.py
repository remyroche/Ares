from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_frozen_exit_state_action_ablation import (
    ContractError,
    _promotion_gate,
    variant_strategy,
)


def test_variant_strategy_changes_only_declared_geometry() -> None:
    base = {
        "sl_mult": 2.0,
        "trailing_activation_decay_start_bars": 0,
        "trailing_activation_decay_half_life_bars": 0.0,
        "trailing_activation_min_mult": 1.0,
        "fixed_trailing_gap_mult": 0.0,
        "giveback_beta": 0.8,
    }
    assert variant_strategy(base, "T4") == base
    d2 = variant_strategy(base, "D2")
    assert d2["trailing_activation_decay_start_bars"] == 120
    assert d2["trailing_activation_decay_half_life_bars"] == 120.0
    assert d2["trailing_activation_min_mult"] == 0.5
    assert d2["sl_mult"] == base["sl_mult"]
    w75 = variant_strategy(base, "W75")
    assert w75["giveback_beta"] == pytest.approx(0.6)
    assert w75["fixed_trailing_gap_mult"] == 0.0
    fixed = variant_strategy(
        {**base, "fixed_trailing_gap_mult": 1.2}, "W75"
    )
    assert fixed["fixed_trailing_gap_mult"] == pytest.approx(0.9)
    assert fixed["giveback_beta"] == base["giveback_beta"]
    assert variant_strategy(base, "P50") == base
    assert base["giveback_beta"] == 0.8
    with pytest.raises(ContractError):
        variant_strategy(base, "unknown")


def test_promotion_gate_requires_all_month_side_rows_and_both_ci_gates() -> None:
    scopes = ("global", "side_long", "side_short")
    months = ("2025-03", "2025-04")
    metric_rows = []
    bootstrap_rows = []
    for arm in ("T4", "D2", "W75", "P50"):
        for month in months:
            for scope in scopes:
                metric_rows.append(
                    {
                        "candidate_month": month,
                        "top_fraction": 0.10,
                        "scope": scope,
                        "arm": arm,
                        "net_bps": 1.0,
                    }
                )
                bootstrap_rows.append(
                    {
                        "candidate_month": month,
                        "top_fraction": 0.10,
                        "scope": scope,
                        "arm": arm,
                        "paired_delta_vs_deployed_ci_low_bps": 0.1,
                        "paired_delta_vs_fixed_12h_ci_low_bps": 0.1,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    passed = _promotion_gate(metrics, bootstrap)
    assert passed.passes_all_retrospective_diagnostic_gates.all()
    assert not passed.promotion_eligible.any()
    bootstrap.loc[
        (bootstrap.arm == "D2") & (bootstrap.scope == "side_short"),
        "paired_delta_vs_fixed_12h_ci_low_bps",
    ] = -0.1
    failed = _promotion_gate(metrics, bootstrap).set_index("arm")
    assert not failed.loc["D2", "passes_all_retrospective_diagnostic_gates"]
    assert failed.loc["T4", "passes_all_retrospective_diagnostic_gates"]
