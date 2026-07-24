from __future__ import annotations

import numpy as np
import pytest

from extreme_price_movements.inference.live_zscore_state import (
    CausalTransformStateContainer,
    CausalTransformStateContainerBusy,
    RollingZScoreState,
    causal_transform_state_container_path,
)

FEATURES = ("ret2h", "ret4h", "ret8h")
SYMBOLS = ("BTC/USD", "ETH/USD")
WINDOW = 4
WINSOR_QT = 0.02
SIGMA_K = 6.0


def _namespace() -> str:
    return CausalTransformStateContainer.namespace_key(
        scope="feature_store=causal-container-test",
        transform_contract="feature-contract-test-v1",
        symbols=SYMBOLS,
        window=WINDOW,
        winsor_qt=WINSOR_QT,
        sigma_k=SIGMA_K,
    )


def _state(feature_keys) -> RollingZScoreState:
    return RollingZScoreState(
        feature_keys,
        SYMBOLS,
        WINDOW,
        SIGMA_K,
        winsor_qt=WINSOR_QT,
    )


def _payload(row: int, feature_keys=FEATURES) -> dict[str, np.ndarray]:
    values = {
        "ret2h": (1.0 + row, 2.0 + row),
        "ret4h": (10.0 + row, 20.0 + row),
        "ret8h": (100.0 + row, 200.0 + row),
    }
    return {
        key: np.asarray(values[key], dtype=np.float32)
        for key in feature_keys
    }


def _timestamp(row: int) -> str:
    return f"2026-01-01T0{row}:00:00+00:00"


def _assert_feature_matches(
    actual: RollingZScoreState,
    expected: RollingZScoreState,
    feature_key: str,
) -> None:
    actual_index = actual._feature_index[feature_key]
    expected_index = expected._feature_index[feature_key]
    np.testing.assert_array_equal(
        actual.valid[actual_index], expected.valid[expected_index]
    )
    np.testing.assert_array_equal(
        actual.ptr[actual_index], expected.ptr[expected_index]
    )
    np.testing.assert_array_equal(
        actual.count[actual_index], expected.count[expected_index]
    )
    np.testing.assert_array_equal(
        actual.K_set[actual_index], expected.K_set[expected_index]
    )
    np.testing.assert_allclose(
        actual.buffer[actual_index], expected.buffer[expected_index], equal_nan=True
    )
    np.testing.assert_allclose(actual.K[actual_index], expected.K[expected_index])
    np.testing.assert_allclose(
        actual.sum_d[actual_index], expected.sum_d[expected_index]
    )
    np.testing.assert_allclose(
        actual.sum_d_sq[actual_index], expected.sum_d_sq[expected_index]
    )


def test_overlapping_worksets_persist_per_feature_without_lost_state(tmp_path):
    path = causal_transform_state_container_path(tmp_path / "causal_transform_state.npz")
    namespace = _namespace()
    reference = _state(FEATURES)
    reference.update(_payload(0), timestamp=_timestamp(0))

    # Workset A seeds ret2h/ret4h. ret8h must stay absent, not be represented by
    # a workset-specific state file that would later fork ret4h's history.
    writer_a = CausalTransformStateContainer.open(path)
    state_a = _state(FEATURES[:2])
    state_a.update(_payload(0, FEATURES[:2]), timestamp=_timestamp(0))
    writer_a.put_many(namespace, state_a)
    assert writer_a.flush() == 2
    writer_a.close()

    # Workset B shares ret4h and initializes ret8h from the earlier row.
    writer_b = CausalTransformStateContainer.open(path)
    state_b = writer_b.get_many(
        namespace,
        feature_keys=FEATURES[1:],
        symbols=SYMBOLS,
        window=WINDOW,
        winsor_qt=WINSOR_QT,
        sigma_k=SIGMA_K,
    )
    assert state_b is not None
    assert state_b.feature_last_timestamp("ret4h") == _timestamp(0)
    assert state_b.feature_last_timestamp("ret8h") is None
    state_b.update(_payload(0, ("ret8h",)), timestamp=_timestamp(0))
    reference.update(_payload(1), timestamp=_timestamp(1))
    state_b.update(_payload(1, FEATURES[1:]), timestamp=_timestamp(1))
    writer_b.put_many(namespace, state_b)
    writer_b.close()

    # A returns: ret2h needs row 1 while ret4h is already current at row 1.
    writer_a_again = CausalTransformStateContainer.open(path)
    state_a_again = writer_a_again.get_many(
        namespace,
        feature_keys=FEATURES[:2],
        symbols=SYMBOLS,
        window=WINDOW,
        winsor_qt=WINSOR_QT,
        sigma_k=SIGMA_K,
    )
    assert state_a_again is not None
    assert state_a_again.last_timestamp is None
    assert state_a_again.feature_last_timestamp("ret2h") == _timestamp(0)
    assert state_a_again.feature_last_timestamp("ret4h") == _timestamp(1)
    state_a_again.update(_payload(1, ("ret2h",)), timestamp=_timestamp(1))
    reference.update(_payload(2), timestamp=_timestamp(2))
    state_a_again.update(_payload(2, FEATURES[:2]), timestamp=_timestamp(2))
    writer_a_again.put_many(namespace, state_a_again)
    writer_a_again.close()

    # B sees ret4h's newer state and ret8h's older cursor. Updating ret8h only
    # cannot overwrite the ret4h row that A just committed.
    verifier = CausalTransformStateContainer.open(path)
    state_b_again = verifier.get_many(
        namespace,
        feature_keys=FEATURES[1:],
        symbols=SYMBOLS,
        window=WINDOW,
        winsor_qt=WINSOR_QT,
        sigma_k=SIGMA_K,
    )
    assert state_b_again is not None
    assert state_b_again.feature_last_timestamp("ret4h") == _timestamp(2)
    assert state_b_again.feature_last_timestamp("ret8h") == _timestamp(1)
    _assert_feature_matches(state_b_again, reference, "ret4h")
    state_b_again.update(_payload(2, ("ret8h",)), timestamp=_timestamp(2))
    verifier.put_many(namespace, state_b_again, feature_keys=("ret8h",))
    verifier.close()

    reopened = CausalTransformStateContainer.open(path)
    full_state = reopened.get_many(
        namespace,
        feature_keys=FEATURES,
        symbols=SYMBOLS,
        window=WINDOW,
        winsor_qt=WINSOR_QT,
        sigma_k=SIGMA_K,
    )
    assert full_state is not None
    for feature_key in FEATURES:
        _assert_feature_matches(full_state, reference, feature_key)
        assert full_state.feature_last_timestamp(feature_key) == _timestamp(2)
    reopened.close()


def test_namespace_metadata_must_match_transform_parameters(tmp_path):
    path = tmp_path / "causal_transform_state.container.sqlite"
    container = CausalTransformStateContainer.open(path)
    with pytest.raises(ValueError, match="namespace metadata is incompatible"):
        container.get_many(
            _namespace(),
            feature_keys=("ret2h",),
            symbols=SYMBOLS,
            window=WINDOW + 1,
            winsor_qt=WINSOR_QT,
            sigma_k=SIGMA_K,
        )
    container.abort()


def test_container_holds_one_writer_reservation(tmp_path):
    path = tmp_path / "causal_transform_state.container.sqlite"
    owner = CausalTransformStateContainer.open(path, lock_timeout_seconds=0.0)
    with pytest.raises(CausalTransformStateContainerBusy):
        CausalTransformStateContainer.open(path, lock_timeout_seconds=0.0)
    owner.abort()
