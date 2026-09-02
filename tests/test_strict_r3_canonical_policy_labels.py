from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.strict_r3_canonical_policy_labels import (
    attach_canonical_policy_labels,
    load_canonical_policy_labels,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
        "policy_path_valid": [True, False],
        "policy_gross_bps": [150.0, float("nan")],
        "policy_net_bps": [50.0, float("nan")],
        "policy_exit_bar_15m": [4, float("nan")],
        "policy_exit_reason": ["trailing", "invalid_path"],
        "policy_entry_price": [1.0, float("nan")],
        "policy_exit_price": [1.01, float("nan")],
        "policy_label_available_ts": pd.to_datetime(["2026-01-01T12:00:00Z", "2026-01-01T13:00:00Z"]),
        "policy_outcome_source": ["existing_15m_or_exact", "unavailable"],
        "policy_cost_bps": [100.0, float("nan")],
    })


def test_invalid_paths_are_not_economic_labels(tmp_path) -> None:
    path = tmp_path / "labels.parquet"
    _labels().to_parquet(path, index=False)
    labels, audit = load_canonical_policy_labels(path)
    assert audit["valid_rows"] == 1
    assert labels.loc[labels.candidate_id.eq("b"), "policy_path_valid"].item() is False
    assert pd.isna(labels.loc[labels.candidate_id.eq("b"), "policy_net_bps"].item())


def test_attachment_replaces_legacy_policy_values(tmp_path) -> None:
    path = tmp_path / "labels.parquet"
    _labels().to_parquet(path, index=False)
    labels, _ = load_canonical_policy_labels(path)
    rows = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
        "policy_path_valid": [False, True],
        "policy_net_bps": [999.0, 999.0],
    })
    merged, audit = attach_canonical_policy_labels(rows, labels)
    assert audit["valid_rows"] == 1
    assert merged.loc[merged.candidate_id.eq("a"), "policy_net_bps"].item() == 50.0
    assert merged.loc[merged.candidate_id.eq("b"), "policy_path_valid"].item() is False


def test_attachment_rejects_changed_decision_identity(tmp_path) -> None:
    path = tmp_path / "labels.parquet"
    _labels().to_parquet(path, index=False)
    labels, _ = load_canonical_policy_labels(path)
    rows = pd.DataFrame({
        "candidate_id": ["a"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:01:00Z"]),
    })
    with pytest.raises(ValueError, match="decision-timestamp"):
        attach_canonical_policy_labels(rows, labels)


def test_missing_label_is_explicitly_invalid_when_coverage_is_audit_only(tmp_path) -> None:
    path = tmp_path / "labels.parquet"
    _labels().to_parquet(path, index=False)
    labels, _ = load_canonical_policy_labels(path)
    rows = pd.DataFrame({
        "candidate_id": ["a", "missing"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"]),
    })
    merged, audit = attach_canonical_policy_labels(
        rows, labels, require_complete_identity_coverage=False,
    )
    missing = merged.loc[merged.candidate_id.eq("missing")].iloc[0]
    assert audit["missing_rows"] == 1
    assert bool(missing.policy_path_valid) is False
    assert pd.isna(missing.policy_net_bps)
    assert pd.isna(missing.policy_label_available_ts)
