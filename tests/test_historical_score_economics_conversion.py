from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_historical_score_economics_conversion_ledger import (
    _standardise,
    validate_ledger,
)
from scripts.run_base_ic_execution_ev_change_attribution import (
    _two_state_shapley,
    change_attribution,
)
from scripts.run_causal_score_economics_conversion_mapping import (
    _stable_select,
    causal_component_mapping,
)


def _raw_rows(days: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exits = ("trailing", "timeout", "full_sl", "adverse_exit")
    for day in range(days):
        signal = pd.Timestamp("2025-02-01", tz="UTC") + pd.Timedelta(days=day)
        for side_index, side in enumerate(("long", "short")):
            for rank in range(4):
                gross = 0.005 + 0.004 * rank - 0.001 * side_index
                cost = 0.01
                rows.append(
                    {
                        "__ts__": signal,
                        "__symbol__": f"ASSET{rank}",
                        "side_name": side,
                        "candidate_id": f"{day}-{side}-{rank}",
                        "execution_decision_utc": signal + pd.Timedelta(hours=1),
                        "execution_label_end_utc": signal + pd.Timedelta(hours=13),
                        "candidate_month": signal.strftime("%Y-%m"),
                        "execution_gross_ev_12h": gross,
                        "execution_net_ev_12h": gross - cost,
                        "execution_cost_return": cost,
                        "execution_exit_reason": exits[rank],
                        "execution_exit_minute": 60.0 + rank,
                        "execution_mfe_return_12h": gross + 0.01,
                        "execution_mae_return_12h": -0.01 - 0.001 * rank,
                        "execution_soft_positive_12h": 1.0
                        / (1.0 + np.exp(-(gross - cost) / 0.01)),
                        "raw_score": float(rank + 0.1 * side_index + 0.01 * day),
                    }
                )
    return pd.DataFrame(rows)


def _ledger(days: int = 3) -> pd.DataFrame:
    return _standardise(
        _raw_rows(days),
        source_family="canonical_base_exact1m_current_spread_cf",
        evidence_tier="test_exact_1m",
        path_frequency="exact_1m",
        cost_contract="test_variable_cost",
        promotion_eligible=True,
        exact_policy_parity=True,
        score_columns={"score_base_alpha": "raw_score"},
    )


def test_materializer_normalises_full_sl_and_enforces_exclusive_exits() -> None:
    ledger = _ledger(days=1)
    full_stop = ledger.loc[ledger["execution_exit_reason"].eq("full_sl")]
    assert full_stop["execution_exit_class"].eq("full_stop").all()
    assert full_stop["exit_is_full_stop"].all()
    assert full_stop["exit_is_adverse"].all()
    flags = ledger[
        [
            "exit_is_trailing",
            "exit_is_timeout",
            "exit_is_full_stop",
            "exit_is_adverse_exit",
        ]
    ]
    assert flags.sum(axis=1).eq(1).all()


def test_materializer_rejects_noncanonical_promotion_family() -> None:
    rows = _raw_rows(days=1)
    with pytest.raises(ValueError, match="noncanonical"):
        _standardise(
            rows,
            source_family="diagnostic_family",
            evidence_tier="diagnostic",
            path_frequency="exact_1m",
            cost_contract="test",
            promotion_eligible=True,
            exact_policy_parity=True,
            score_columns={"score_base_alpha": "raw_score"},
        )


def test_causal_mapping_uses_only_strictly_resolved_prior_rows() -> None:
    mapped, audit = causal_component_mapping(
        _ledger(days=3),
        score_column="score_base_alpha",
        window_days=21,
        minimum_reference_rows=1,
        minimum_side_rows=1,
        minimum_quantile_rows=1,
        side_shrinkage=1.0,
        cell_shrinkage=1.0,
    )
    first_day = mapped["execution_decision_utc"].dt.floor("D").min()
    assert not mapped.loc[
        mapped["execution_decision_utc"].dt.floor("D").eq(first_day),
        "mapped_eligible",
    ].any()
    assert mapped.loc[
        mapped["execution_decision_utc"].dt.floor("D").gt(first_day),
        "mapped_eligible",
    ].all()
    used = audit.loc[audit["reference_rows"].gt(0)]
    assert (
        pd.to_datetime(used["reference_label_end_max_utc"], utc=True)
        < pd.to_datetime(used["snapshot_utc"], utc=True)
    ).all()
    exit_sum = mapped.loc[
        mapped["mapped_eligible"],
        [
            "mapped_exit_probability_trailing",
            "mapped_exit_probability_timeout",
            "mapped_exit_probability_full_stop",
            "mapped_exit_probability_adverse_exit",
        ],
    ].sum(axis=1)
    assert np.allclose(exit_sum, 1.0)


def test_stable_global_selection_breaks_score_ties_by_candidate_id() -> None:
    ledger = _ledger(days=1)
    ledger["tied"] = 1.0
    selected = _stable_select(ledger, score_column="tied", fraction=0.25)
    expected = sorted(ledger["candidate_id"])[:2]
    assert selected["candidate_id"].tolist() == expected


def test_validate_ledger_rejects_nonexclusive_exit_flags() -> None:
    ledger = _ledger(days=1)
    ledger.loc[0, "exit_is_timeout"] = True
    with pytest.raises(ValueError, match="exactly one"):
        validate_ledger(ledger)


def test_two_state_shapley_reconciles_mixture_change() -> None:
    probability_a = np.array([0.4, 0.6])
    probability_b = np.array([0.7, 0.3])
    value_a = np.array([4.0, -2.0])
    value_b = np.array([3.0, -1.0])
    mix, payoff = _two_state_shapley(
        probability_a, value_a, probability_b, value_b
    )
    actual = float(np.dot(probability_b, value_b) - np.dot(probability_a, value_a))
    assert mix + payoff == pytest.approx(actual)


def test_month_change_attribution_reconciles_each_independent_lens() -> None:
    rows = []
    for month, gross, cost, opportunity, trailing, long in (
        ("2025-02", 50.0, 100.0, 0.5, 0.6, 0.7),
        ("2025-03", 20.0, 101.0, 0.4, 0.5, 0.6),
    ):
        rows.append(
            {
                "candidate_month": month,
                "mean_gross_bps": gross,
                "mean_cost_bps": cost,
                "mean_net_bps": gross - cost,
                "opportunity_rate": opportunity,
                "opportunity_conditional_gross_bps": 300.0,
                "no_opportunity_conditional_gross_bps": (
                    gross - opportunity * 300.0
                )
                / (1.0 - opportunity),
                "long_share": long,
                "short_share": 1.0 - long,
                "long_conditional_gross_bps": gross + 10.0,
                "short_conditional_gross_bps": (
                    gross - long * (gross + 10.0)
                )
                / (1.0 - long),
                "trailing_share": trailing,
                "timeout_share": 1.0 - trailing,
                "full_stop_share": 0.0,
                "adverse_exit_share": 0.0,
                "trailing_conditional_gross_bps": gross + 20.0,
                "timeout_conditional_gross_bps": (
                    gross - trailing * (gross + 20.0)
                )
                / (1.0 - trailing),
                "full_stop_conditional_gross_bps": 0.0,
                "adverse_exit_conditional_gross_bps": 0.0,
            }
        )
    result = change_attribution(pd.DataFrame(rows))
    assert set(result["lens"]) == {"opportunity", "exit", "side_book"}
    assert np.allclose(result["reconciliation_error_bps"], 0.0, atol=1e-10)
