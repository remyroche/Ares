from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_febapr2025_exact_policy_population import validate_population


def _population() -> pd.DataFrame:
    signal = pd.to_datetime(["2025-02-01T00:00:00Z", "2025-02-01T01:00:00Z"])
    decision = signal + pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "__ts__": signal,
            "__symbol__": ["BTC_USD:USD", "ETH_USD:USD"],
            "side_name": ["long", "short"],
            "candidate_id": ["a", "b"],
            "execution_decision_utc": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
            "execution_label_available_at_utc": decision + pd.Timedelta(hours=12),
            "candidate_month": ["2025-02", "2025-02"],
            "execution_gross_ev_12h": [0.02, 0.01],
            "execution_cost_return": [0.01, 0.01],
            "execution_net_ev_12h": [0.01, 0.0],
            "execution_exit_reason": ["timeout", "stop"],
            "execution_exit_minute": [720, 12],
            "execution_mfe_return_12h": [0.03, 0.01],
            "execution_mae_return_12h": [0.01, 0.02],
            "execution_soft_positive_12h": [0.7, 0.5],
            "feature_source_ledger": ["long.parquet", "short.parquet"],
            "has_exact_1m_path": [True, True],
            "has_feature_store_join_key": [True, True],
            "eligible_for_fresh_canonical_base_oof": [True, True],
        }
    )


def test_population_requires_unique_identities_and_causal_label_resolution() -> None:
    frame = _population()
    validate_population(frame)
    frame.loc[0, "execution_label_available_at_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="label availability"):
        validate_population(frame)


def test_population_rejects_missing_exact_path_or_unreconciled_target() -> None:
    frame = _population()
    frame.loc[0, "has_exact_1m_path"] = False
    with pytest.raises(ValueError, match="ineligible"):
        validate_population(frame)
    frame = _population()
    frame.loc[0, "execution_net_ev_12h"] = 0.02
    with pytest.raises(ValueError, match="gross-cost"):
        validate_population(frame)
