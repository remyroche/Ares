from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from extreme_price_movements.causal_leaf_health_incremental import (
    allocate_h1_state,
    score_auxiliary_block,
    score_h1_block,
    score_h1_candidate,
    update_h1_block,
    update_h1_candidate,
)
from extreme_price_movements.causal_leaf_health_event_incremental import (
    _DIRECT_NAMES,
    _ENTROPY_MASS,
    _ENTROPY_MASS_LOG_MASS,
    REASONING_ENTROPY_FIELD,
    _candidate_paths,
    _entropy_statistics,
    _entropy_statistics_block,
    _final_health,
)
from extreme_price_movements.causal_leaf_health_scoped import _final_health as _scoped_final_health
from extreme_price_movements.causal_leaf_health_vectorized import _IDENTITY
from extreme_price_movements.strict_event_store import StrictEventStore


def _score(state):
    return score_h1_candidate(
        np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int8), np.asarray([0.5], dtype=np.float32),
        state["family_rows"], state["family_successes"], state["family_predictions"],
        state["family_nets"], state["family_expecteds"], state["family_false_positive_losses"],
        state["family_timestamps"], state["family_days"], state["family_symbols"],
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
    )


def test_incremental_h1_scores_before_update_and_keeps_exact_support_counts():
    state = allocate_h1_state(2, 2)
    before = _score(state)
    # The initial family prior is neutral and no support is visible.
    np.testing.assert_allclose(before[1, :3], [0.5, 0.0, 0.0])
    assert before[1, 10] == 0.5

    values = np.asarray([0.5], dtype=np.float32)
    codes = np.asarray([0], dtype=np.int32)
    update_h1_candidate(
        codes, values, 0.0, 0.8, -25.0, 5.0,
        np.int64(100), np.int32(1), np.int32(0),
        state["family_rows"], state["family_successes"], state["family_predictions"],
        state["family_nets"], state["family_expecteds"], state["family_false_positive_losses"],
        state["family_timestamps"], state["family_days"], state["family_symbols"],
        state["family_last_timestamp"], state["family_last_day"], state["family_asset_seen"],
    )
    # Same decision timestamp must not create a second timestamp/day support
    # observation, while a new asset remains a new symbol observation.
    update_h1_candidate(
        codes, values, 1.0, 0.6, 15.0, 5.0,
        np.int64(100), np.int32(1), np.int32(1),
        state["family_rows"], state["family_successes"], state["family_predictions"],
        state["family_nets"], state["family_expecteds"], state["family_false_positive_losses"],
        state["family_timestamps"], state["family_days"], state["family_symbols"],
        state["family_last_timestamp"], state["family_last_day"], state["family_asset_seen"],
    )
    after = _score(state)
    assert after[1, 2] == 2.0
    assert after[1, 3] == 1.0
    assert after[1, 4] == 1.0
    assert after[1, 5] == 2.0
    assert after[1, 9] == 12.5
    assert after[1, 10] == 0.5


def test_incremental_h1_block_matches_single_candidate_scoring():
    state = allocate_h1_state(2, 1)
    values = np.asarray([0.2, -0.4], dtype=np.float32)
    codes = np.asarray([0, 1], dtype=np.int32)
    directions = np.asarray([1, 0], dtype=np.int8)
    single = score_h1_candidate(
        codes, directions, values,
        state["family_rows"], state["family_successes"], state["family_predictions"],
        state["family_nets"], state["family_expecteds"], state["family_false_positive_losses"],
        state["family_timestamps"], state["family_days"], state["family_symbols"],
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
    )
    block = score_h1_block(
        codes, directions, values, np.asarray([0, 2], dtype=np.int64),
        state["family_rows"], state["family_successes"], state["family_predictions"],
        state["family_nets"], state["family_expecteds"], state["family_false_positive_losses"],
        state["family_timestamps"], state["family_days"], state["family_symbols"],
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
    )
    np.testing.assert_allclose(block[0], single)


def test_incremental_h1_update_block_matches_sequential_updates():
    sequential = allocate_h1_state(2, 2)
    blocked = allocate_h1_state(2, 2)
    codes = np.asarray([0, 1], dtype=np.int32)
    values = np.asarray([0.3, -0.2], dtype=np.float32)
    for index in range(2):
        update_h1_candidate(
            codes[index:index + 1], values[index:index + 1], float(index), 0.6, -10.0, 2.0,
            np.int64(100), np.int32(1), np.int32(index),
            sequential["family_rows"], sequential["family_successes"], sequential["family_predictions"],
            sequential["family_nets"], sequential["family_expecteds"], sequential["family_false_positive_losses"],
            sequential["family_timestamps"], sequential["family_days"], sequential["family_symbols"],
            sequential["family_last_timestamp"], sequential["family_last_day"], sequential["family_asset_seen"],
        )
    update_h1_block(
        codes, values, np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0]), np.asarray([0.6, 0.6]), np.asarray([-10.0, -10.0]), np.asarray([2.0, 2.0]),
        np.asarray([100, 100], dtype=np.int64), np.asarray([1, 1], dtype=np.int32), np.asarray([0, 1], dtype=np.int32),
        blocked["family_rows"], blocked["family_successes"], blocked["family_predictions"],
        blocked["family_nets"], blocked["family_expecteds"], blocked["family_false_positive_losses"],
        blocked["family_timestamps"], blocked["family_days"], blocked["family_symbols"],
        blocked["family_last_timestamp"], blocked["family_last_day"], blocked["family_asset_seen"],
    )
    for name in sequential:
        np.testing.assert_array_equal(blocked[name], sequential[name])


def test_incremental_auxiliary_block_keeps_full_contribution_denominator():
    # The positive contribution is selected for H3; the negative one is not.
    # Each direction remains normalised by its own full active mass.
    result = score_auxiliary_block(
        np.asarray([0, 1], dtype=np.int32), np.asarray([1, 0], dtype=np.int8),
        np.asarray([2.0, -3.0], dtype=np.float32), np.asarray([0, 2], dtype=np.int64),
        np.asarray([[0.2, 0.4], [0.8, 0.6]], dtype=np.float64),
        np.asarray([[0.5], [0.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(result[0, 1], [0.2, 0.4, 0.5, 2.0])
    np.testing.assert_allclose(result[0, 0], [0.8, 0.6, 0.0, 3.0])


def test_reasoning_entropy_statistics_are_additive_and_zero_for_zero_mass():
    """The full candidate entropy must span head-local contribution vectors."""

    mass_a, moment_a = _entropy_statistics(np.asarray([2.0, -1.0], dtype=np.float32))
    mass_b, moment_b = _entropy_statistics(np.asarray([-3.0], dtype=np.float32))
    total = float(mass_a + mass_b)
    entropy = np.log(total) - float(moment_a + moment_b) / total
    expected = -sum(value * np.log(value) for value in (2.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0))
    assert entropy == pytest.approx(expected)
    assert _entropy_statistics(np.asarray([0.0, -0.0], dtype=np.float32)) == (np.float32(0.0), np.float32(0.0))
    masses, moments = _entropy_statistics_block(
        np.asarray([2.0, -1.0, -3.0, 0.0], dtype=np.float32),
        np.asarray([0, 2, 4], dtype=np.int64),
    )
    np.testing.assert_allclose(masses, [3.0, 3.0])
    np.testing.assert_allclose(moments, [2.0 * np.log(2.0), 3.0 * np.log(3.0)])


def _candidate_part(
    root: Path, *, name: str, month: str, meta_partition: str, rows: list[dict[str, object]], index_rows: list[dict[str, object]],
) -> None:
    path = root / name
    pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
    first = rows[0]
    index_rows.append({
        "dataset": "candidate", "path": name, "contract": "contract-a",
        "side": str(first["side_name"]), "head": str(first["head_name"]),
        "month": month, "meta_partition": meta_partition,
    })


def test_partitioned_final_health_matches_global_reference_and_order(tmp_path: Path) -> None:
    """Monthly/transport slices must be a semantic replacement for the global join.

    The fixture deliberately repeats an identity across two head parts (the
    candidate source needs ``DISTINCT``), spans two transports, strict
    partitions and decision months, and verifies the special H4 active-mass
    recovery from H1 rather than an H4 cell.
    """

    candidate_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []

    def row(
        candidate_id: str, when: str, *, side: str, head: str, fold: str,
        transport: str, partition: str,
    ) -> dict[str, object]:
        timestamp = pd.Timestamp(when, tz="UTC")
        return {
            "candidate_id": candidate_id, "decision_ts": timestamp,
            "side_name": side, "head_name": head, "fold_id": fold,
            "transport": transport, "meta_partition": partition,
        }

    alpha = row("alpha", "2024-01-03 01:00", side="long", head="peak_mfe", fold="f0", transport="T_A", partition="inner_oof")
    # Same final identity in a different head part: source candidate rows are
    # head-level, while final health is one row per _IDENTITY.
    alpha_duplicate = {**alpha, "head_name": "future_slope"}
    beta = row("beta", "2024-02-02 00:00", side="short", head="peak_mfe", fold="f1", transport="T_A", partition="inner_oof")
    delta = row("delta", "2024-01-04 00:00", side="short", head="peak_mfe", fold="f2", transport="T_B", partition="inner_oof")
    gamma = row("gamma", "2024-01-05 00:00", side="long", head="future_slope", fold="f3", transport="T_B", partition="outer_test")
    _candidate_part(tmp_path, name="candidate_jan_a.parquet", month="2024-01", meta_partition="inner_oof", rows=[alpha, alpha_duplicate, delta], index_rows=index_rows)
    _candidate_part(tmp_path, name="candidate_feb_a.parquet", month="2024-02", meta_partition="inner_oof", rows=[beta], index_rows=index_rows)
    _candidate_part(tmp_path, name="candidate_jan_b.parquet", month="2024-01", meta_partition="outer_test", rows=[gamma], index_rows=index_rows)
    store = StrictEventStore(
        root=tmp_path, manifest_path=tmp_path / "manifest.json", manifest={}, part_index=pd.DataFrame(index_rows),
    )

    # Scope-local direct parts carry the complete conventional direct schema,
    # including zeroes for unrelated heads/directions.  Their sum is the
    # public final H1/H2/H3 surface.
    direct_base = pd.DataFrame([{field: 0.0 for field in (*_DIRECT_NAMES, _ENTROPY_MASS, _ENTROPY_MASS_LOG_MASS)} | {
        key: value for key, value in item.items() if key != "head_name"
    } for item in (alpha, beta, delta, gamma)])
    direct_other = direct_base.copy()
    h1_active = "base_health__h1__p_clear__positive__active_abs_contribution"
    direct_base.loc[direct_base["candidate_id"].eq("alpha"), h1_active] = 2.5
    direct_base.loc[direct_base["candidate_id"].eq("beta"), h1_active] = 1.5
    # A distinct direct field confirms the global per-identity SUM remains
    # intact after partitioning.
    h2_field = "base_health__h2__p_clear__positive__instability"
    direct_other.loc[direct_other["candidate_id"].eq("alpha"), h2_field] = -0.75
    direct_paths = (tmp_path / "direct_base.parquet", tmp_path / "direct_other.parquet")
    direct_base.to_parquet(direct_paths[0], index=False, compression="zstd")
    direct_other.to_parquet(direct_paths[1], index=False, compression="zstd")

    h4_metric = "base_health__h4__p_clear__positive__covariance_drift"
    h4_active = "base_health__h4__p_clear__positive__active_abs_contribution"
    h4_path = tmp_path / "h4.parquet"
    pd.DataFrame([
        {key: alpha[key] for key in _IDENTITY} | {h4_metric: 4.0, h4_active: 999.0},
        {key: gamma[key] for key in _IDENTITY} | {h4_metric: 8.0, h4_active: 888.0},
    ]).to_parquet(h4_path, index=False, compression="zstd")
    h5_metric = "base_health__h5__p_clear__positive__weighted_break"
    h5_path = tmp_path / "h5.parquet"
    pd.DataFrame([
        {key: beta[key] for key in _IDENTITY} | {h5_metric: 0.25},
        {key: gamma[key] for key in _IDENTITY} | {h5_metric: 0.75},
    ]).to_parquet(h5_path, index=False, compression="zstd")

    expected_path = tmp_path / "expected_global.parquet"
    connection = duckdb.connect()
    try:
        _scoped_final_health(
            connection, candidate_paths=_candidate_paths(store), scope_health=direct_paths,
            h4_path=h4_path, h4_names=(h4_metric, h4_active), h5_path=h5_path,
            h5_names=(h5_metric,), output=expected_path,
        )
    finally:
        connection.close()
    actual_path = tmp_path / "actual_partitioned.parquet"
    rows = _final_health(
        store=store, direct_paths=direct_paths, h4_path=h4_path, h4_names=(h4_metric, h4_active),
        h5_path=h5_path, h5_names=(h5_metric,), output=actual_path, memory_limit="128MB",
        temp_disk_limit="512MB", temp_dir=tmp_path / "final_work",
    )
    expected = pd.read_parquet(expected_path)
    actual = pd.read_parquet(actual_path)
    assert rows == len(expected) == 4
    pdt.assert_frame_equal(
        actual.drop(columns=[REASONING_ENTROPY_FIELD]),
        expected.drop(columns=[_ENTROPY_MASS, _ENTROPY_MASS_LOG_MASS]),
        check_like=False,
    )
    assert actual["candidate_id"].tolist() == ["alpha", "beta", "delta", "gamma"]
    alpha_row = actual.loc[actual["candidate_id"].eq("alpha")].iloc[0]
    assert alpha_row[h4_active] == np.float32(2.5)
    assert alpha_row[h2_field] == np.float32(-0.75)
    assert alpha_row[REASONING_ENTROPY_FIELD] == np.float32(0.0)
