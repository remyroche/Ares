"""Regression tests for the offline-only causal S/R oracle diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_causal_sr_heads as heads
from scripts import run_causal_sr_oracle_entry_ablation as entry


def test_reaction_magnitude_metric_uses_its_own_mfe_target() -> None:
    """A perfect MFE prediction must not be scored against reaction strength."""
    events = pd.DataFrame({
        "y_reaction_strength": [4.0, 3.0, 2.0, 1.0],
        "reaction_MFE_atr": [1.0, 2.0, 3.0, 4.0],
        "sr_prior_strength": [4.0, 3.0, 2.0, 1.0],
        "sr_conditional_strength": [4.0, 3.0, 2.0, 1.0],
        "sr_reaction_magnitude_q50": [1.0, 2.0, 3.0, 4.0],
        "sr_accepted_break_probability": [0.1, 0.2, 0.8, 0.9],
        "y_accepted_break": [0, 0, 1, 1],
    })
    metrics = heads._fold_metrics(events, pd.Timestamp("2026-08-01", tz="UTC"))
    magnitude = next(item for item in metrics if item["head"] == "sr_reaction_magnitude_q50")
    assert magnitude["spearman"] == pytest.approx(1.0)


def test_entry_oracle_label_loader_excludes_unreadable_symbol_without_substitution(tmp_path) -> None:
    """Unreadable immutable labels remain absent from every diagnostic arm."""
    root = tmp_path / "labels"
    good = root / "policy_parts" / "symbol=GOOD_USD:USD"
    bad = root / "policy_parts" / "symbol=BAD_USD:USD"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    pd.DataFrame({
        "candidate_id": ["good"],
        "policy_path_valid": [True],
        "policy_gross_bps": [100.0],
        "policy_net_bps": [0.0],
        "policy_exit_bar_15m": [0],
        "policy_entry_price": [1.0],
        "policy_exit_price": [1.0],
        "policy_exit_reason": ["timeout"],
        "policy_label_available_ts": [pd.Timestamp("2026-01-01", tz="UTC")],
        "policy_cost_bps": [100.0],
    }).to_parquet(good / "policy_labels.parquet", index=False)
    (bad / "policy_labels.parquet").write_bytes(b"not-a-parquet-file")

    labels, unavailable = entry._labels(root)

    assert labels.candidate_id.tolist() == ["good"]
    assert unavailable == [{"symbol": "BAD_USD:USD", "reason": "unreadable_parquet:ArrowInvalid"}]
