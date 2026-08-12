from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_cell_day_trust import (
    CellDayResidualTrustBundle,
)


class _Transform:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(frame), 1), dtype=np.float32)


class _Forest:
    def apply(self, design: np.ndarray) -> np.ndarray:
        return np.zeros((len(design), 1), dtype=np.int64)


def _bundle(*, support: float, q25: float, p100: float, mean: float = -100.0) -> CellDayResidualTrustBundle:
    # Tuple order: support, mean, q10, q25, p50, p100, p200.
    statistic = (support, mean, q25 - 50.0, q25, p100, p100, p100 / 2.0)
    return CellDayResidualTrustBundle(
        cutoff=pd.Timestamp("2026-01-01", tz="UTC"),
        fields=("base_score",), transform=_Transform(), edges=(), model=_Forest(),
        leaf_statistics=({0: statistic},), global_mean=mean,
        global_q10=q25 - 50.0, global_q25=q25,
        global_probabilities=(p100, p100, p100 / 2.0), residual_noise=100.0,
        manifest={"schema": "test"},
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["candidate"], "raw_expected_bps": [100.0],
        "base_score": [0.5],
    })


def test_r5_requires_all_three_corroboration_gates() -> None:
    for bundle in (
        _bundle(support=119.0, q25=-100.0, p100=0.80),
        _bundle(support=200.0, q25=-49.0, p100=0.80),
        _bundle(support=200.0, q25=-100.0, p100=0.64),
    ):
        output = bundle.score(_frame())
        assert not bool(output.loc[0, "trust_risk_corroborated"])
        assert output.loc[0, "trust_authority"] == 0.0
        assert output.loc[0, "auction_rank_adjustment_bps"] == 0.0


def test_r5_is_bounded_demotion_only() -> None:
    output = _bundle(support=300.0, q25=-100.0, p100=0.80).score(_frame())
    assert bool(output.loc[0, "trust_risk_corroborated"])
    assert 0.0 < output.loc[0, "trust_authority"] <= 0.10
    assert output.loc[0, "auction_rank_adjustment_bps"] < 0.0
    assert output.loc[0, "trust_corrected_expected_net_bps"] <= 100.0


def test_r5_exposes_posterior_expected_net_for_canonical_admission() -> None:
    output = _bundle(
        support=300.0, q25=-100.0, p100=0.80, mean=-25.0,
    ).score(_frame())
    assert output.loc[0, "trust_posterior_expected_bps"] == 75.0
