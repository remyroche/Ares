from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_historical_exact_h12_alignment_sidecar import (
    COST_MODEL_ID,
    EXECUTION_POLICY_ID,
    TARGET_ID,
)
from scripts.materialize_root_cause_exact_h12_execution_targets import (
    build_exact_h12_target_rows,
    materialize,
    validate_exact_h12_target_contract,
)


def _path(*, decision: pd.Timestamp, high_values: dict[int, float], low_values: dict[int, float], close_last: float = 1.0) -> str:
    timestamps = (decision.value + np.arange(720, dtype=np.int64) * pd.Timedelta(minutes=1).value).tolist()
    open_ = np.ones(720, dtype=float)
    high = np.ones(720, dtype=float)
    low = np.ones(720, dtype=float)
    close = np.ones(720, dtype=float)
    for index, value in high_values.items():
        high[index] = value
    for index, value in low_values.items():
        low[index] = value
    close[-1] = close_last
    high[-1] = max(high[-1], close_last)
    low[-1] = min(low[-1], close_last)
    return json.dumps({"timestamp": timestamps, "open": open_.tolist(), "high": high.tolist(), "low": low.tolist(), "close": close.tolist()})


def _alignment() -> pd.DataFrame:
    decision = pd.Timestamp("2024-01-02 12:00:00Z")
    rows = []
    for candidate_id, net in (("clean", 100.0), ("timeout", -80.0)):
        rows.append(
            {
                "candidate_id": candidate_id,
                "symbol": "TEST/USD:USD",
                "side": "long",
                "decision_ts": decision,
                "feature_cutoff_ts": decision - pd.Timedelta(hours=1),
                "entry_ts": decision,
                "label_end_ts": decision + pd.Timedelta(hours=12),
                "label_available_ts": decision + pd.Timedelta(hours=12),
                "target_id": TARGET_ID,
                "execution_policy_id": EXECUTION_POLICY_ID,
                "replay_execution_policy_id": EXECUTION_POLICY_ID,
                "cost_model_id": COST_MODEL_ID,
                "feature_set_id": "test_feature_set",
                "policy_archetype": "test",
                "execution_geometry_id": "geometry-id",
                "execution_geometry_key": "geometry-key",
                "execution_geometry_source": "test-source",
                "barrier_pct": 0.02,
                "execution_entry_price": 1.0,
                "exact_h12_gross_bps": net + 100.0,
                "row_cost_bps": 100.0,
                "exact_h12_net_bps": net,
                "source_row_number": 1,
                "source_shard_sha256": "sha",
            }
        )
    return pd.DataFrame(rows)


def _paths(decision: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["clean", "timeout"],
            "__ts__": [decision - pd.Timedelta(hours=1), decision - pd.Timedelta(hours=1)],
            "__symbol__": ["TEST/USD:USD", "TEST/USD:USD"],
            "side_name": ["long", "long"],
            "atr_1h": [0.01, 0.01],
            "decision_price": [1.0, 1.0],
            "execution_future_path": [
                _path(decision=decision, high_values={2: 1.02}, low_values={7: 0.97}),
                _path(decision=decision, high_values={index: 1.005 for index in range(720)}, low_values={index: 0.995 for index in range(720)}, close_last=1.0025),
            ],
        }
    )


def test_primary_net_target_is_authoritative_and_conditionals_are_stratified() -> None:
    alignment = _alignment()
    decision = alignment.decision_ts.iloc[0]
    primary, supportive = build_exact_h12_target_rows(
        _paths(decision), alignment.set_index("candidate_id"), include_full_auxiliary_support=False
    )
    clean_primary = primary.set_index("candidate_id").loc["clean"]
    assert clean_primary.execution_exact_h12_gross_bps == pytest.approx(200.0)
    assert clean_primary.execution_exact_h12_cost_bps == pytest.approx(100.0)
    assert clean_primary.execution_exact_h12_net_bps == pytest.approx(100.0)
    assert clean_primary.execution_exact_h12_net_positive == 1
    assert primary["execution_exact_h12_net_bps"].dtype == np.float64
    np.testing.assert_array_equal(
        primary["execution_exact_h12_gross_bps"].to_numpy()
        - primary["execution_exact_h12_cost_bps"].to_numpy(),
        primary["execution_exact_h12_net_bps"].to_numpy(),
    )

    labels = supportive.set_index("candidate_id")
    assert labels.loc["clean", "clean_economic_favorable_first"] == 1
    assert labels.loc["clean", "adverse_first"] == 0
    assert labels.loc["clean", "timeout"] == 0
    assert np.isfinite(labels.loc["clean", "conditional_peak_mfe_atr_given_meaningful_mfe"])
    assert np.isnan(labels.loc["timeout", "conditional_peak_mfe_atr_given_meaningful_mfe"])
    assert labels.loc["timeout", "timeout"] == 1
    assert labels.loc["timeout", "timeout_soft_timeout_viability"] == pytest.approx(1.0)
    assert np.isnan(labels.loc["clean", "timeout_soft_timeout_viability"])
    assert np.isnan(labels.loc["timeout", "conditional_exact_h12_net_bps_given_clean_economic_first"])


def test_contract_rejects_non_exact_h12_timing_before_label_build() -> None:
    alignment = _alignment()
    alignment.loc[0, "label_end_ts"] = alignment.loc[0, "label_end_ts"] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="exact H12|exact-H12|exact H12"):
        validate_exact_h12_target_contract(alignment)


def test_materializer_writes_a_one_to_one_target_pack(tmp_path) -> None:
    alignment = _alignment()
    paths = _paths(alignment.decision_ts.iloc[0])
    alignment_path = tmp_path / "alignment.parquet"
    path_file = tmp_path / "paths.parquet"
    output = tmp_path / "target-pack"
    alignment.to_parquet(alignment_path, index=False)
    paths.to_parquet(path_file, index=False)

    manifest = materialize(
        alignment_path=alignment_path,
        path_files=(path_file,),
        output=output,
        batch_rows=2,
        include_full_auxiliary_support=False,
    )

    assert manifest["rows"] == 2
    assert {
        "primary_labels.parquet",
        "supportive_labels.parquet",
        "label_dictionary.parquet",
        "support_report.parquet",
        "execution_target_contract.json",
        "manifest.json",
    } == {path.name for path in output.iterdir()}
    assert len(pd.read_parquet(output / "primary_labels.parquet")) == 2
    assert len(pd.read_parquet(output / "supportive_labels.parquet")) == 2
