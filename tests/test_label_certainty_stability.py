import numpy as np
import pytest

from extreme_price_movements.label_certainty_stability import (
    CERTAINTY_PREFIX,
    PerturbationContract,
    assert_no_label_certainty_inference_features,
    build_label_certainty,
    materialize_perturbed_barrier_targets,
)


def _paths(rows=3, minutes=960):
    open_ = np.full((rows, minutes), 100.0)
    high = open_.copy()
    low = open_.copy()
    close = open_.copy()
    high[0, 10:] = 103.0
    low[1, 8:] = 97.0
    close[2, :] = np.linspace(100.0, 100.2, minutes)
    return open_, high, low, close


def test_certainty_materializes_components_and_weights():
    open_, high, low, close = _paths()
    contracts = (
        PerturbationContract("reference"),
        PerturbationContract("delay", entry_delay_minutes=1),
        PerturbationContract("long_atr", atr_source="long"),
    )
    variants, report = materialize_perturbed_barrier_targets(
        open_=open_, high=high, low=low, close=close,
        side_sign=np.array([1.0, 1.0, -1.0]), atr_reference=np.array([0.01, 0.01, 0.01]),
        cost_return=np.array([0.001, 0.001, 0.001]), atr_alternatives={"long": np.array([0.012, 0.012, 0.012])}, contracts=contracts,
    )
    result = build_label_certainty(variants)
    assert len(report) == 3
    assert len(result) == 3
    assert result[f"{CERTAINTY_PREFIX}score"].between(0.0, 1.0).all()
    assert result[f"{CERTAINTY_PREFIX}weight_c1"].between(0.5, 1.0).all()
    assert result[f"{CERTAINTY_PREFIX}training_only"].eq(1).all()


def test_missing_atr_source_fails_closed():
    open_, high, low, close = _paths()
    with pytest.raises(ValueError, match="ATR source"):
        materialize_perturbed_barrier_targets(
            open_=open_, high=high, low=low, close=close, side_sign=np.ones(3),
            atr_reference=np.full(3, 0.01), cost_return=np.zeros(3),
            contracts=(PerturbationContract("reference"), PerturbationContract("long", atr_source="long")),
        )


def test_certainty_is_rejected_from_inference_features():
    assert_no_label_certainty_inference_features(["causal_feature"])
    with pytest.raises(ValueError, match="training-only"):
        assert_no_label_certainty_inference_features([f"{CERTAINTY_PREFIX}score"])
