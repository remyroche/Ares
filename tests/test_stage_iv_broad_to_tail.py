from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iv_broad_to_tail import (
    StageIVPlan,
    pooled_global_stage_iv_metrics,
    prequential_tail_handoff,
    run_stage_iv_broad_to_tail_ablation,
)


class _ColumnModel:
    def __init__(self, column: str, target_mean: float) -> None:
        self.column = column
        self.target_mean = target_mean

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Retain a fitted-target component so the future-label isolation test
        # would fail if a future outcome were ever admitted to an earlier fold.
        return X[self.column].to_numpy(dtype=np.float32) + np.float32(self.target_mean * 0.001)


def _fitter_records(records: list[tuple[str, tuple[str, ...], int]]):
    def fit(X, y, _weight, layer, _params):
        records.append((layer, tuple(X.columns), len(X)))
        raw_columns = [column for column in X.columns if not column.startswith("__stage_iv_")]
        return _ColumnModel(raw_columns[0], float(np.mean(y)))

    return fit


def _plan(
    side: str,
    *,
    route: str = "both",
    tail_fraction: float = 0.30,
    final_target_shift: float = 0.0,
    final_net_shift: float = 0.0,
) -> StageIVPlan:
    n_timestamps = 160
    per_timestamp = 2
    n = n_timestamps * per_timestamp
    rng = np.random.default_rng(5 if side == "long" else 9)
    decision = pd.date_range("2024-01-01", periods=n_timestamps, freq="h", tz="UTC").repeat(per_timestamp)
    rank = np.tile(np.array([0.1, 0.9], dtype=np.float32), n_timestamps)
    # This gives both rows of a timestamp different scores, which is useful for
    # catching accidental per-timestamp top-x selection.
    frame = pd.DataFrame({
        "broad_signal": rank + rng.normal(0.0, 0.02, n),
        "tail_signal": rank + rng.normal(0.0, 0.02, n),
        "meta_context": rng.normal(0.0, 1.0, n),
    }).astype("float32")
    target = (rank * 100.0 + rng.normal(0.0, 2.0, n)).astype("float32")
    target[-1] += final_target_shift
    net = (rank * 160.0 - 100.0 + rng.normal(0.0, 3.0, n)).astype("float32")
    net[-1] += final_net_shift
    return StageIVPlan(
        side=side,
        candidate_ids=[f"{side}-{index}" for index in range(n)],
        frame=frame,
        base_target=target,
        exact_net_bps=net,
        decision_timestamps=decision,
        label_available_timestamps=decision + pd.Timedelta(hours=13),
        broad_feature_names=["broad_signal"],
        tail_feature_names=["tail_signal"],
        meta_feature_names=["meta_context"],
        broad_params={}, tail_params={}, meta_params={},
        tail_fraction=tail_fraction,
        broad_min_train_rows=18,
        tail_min_train_rows=10,
        meta_min_train_rows=8,
        min_handoff_history_rows=8,
        n_validation_folds=3,
        broad_output_route=route,
    )


def test_handoff_is_prior_global_score_threshold_not_per_timestamp() -> None:
    timestamp = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
         "2024-01-01T01:00:00Z", "2024-01-01T01:00:00Z",
         "2024-01-01T02:00:00Z", "2024-01-01T02:00:00Z"]
    )
    threshold, selected = prequential_tail_handoff(
        [1.0, 100.0, 2.0, 3.0, 51.0, 52.0], timestamp,
        tail_fraction=0.50, min_history_rows=2,
    )
    # At t=01 the predeclared global historic median is 50.5, so neither row
    # qualifies. A per-timestamp top-50% procedure would have selected score 3.
    assert threshold[2] == threshold[3] == 50.5
    assert not selected[2:4].any()
    # t=02 sees earlier score history only; both current rows use one threshold.
    assert threshold[4] == threshold[5]
    assert selected[4] and selected[5]


def test_stage_iv_strict_chain_has_independent_burnins_and_same_side_direct_handoffs() -> None:
    records: list[tuple[str, tuple[str, ...], int]] = []
    result = run_stage_iv_broad_to_tail_ablation([_plan("long")], fitter=_fitter_records(records))
    prediction = result.predictions
    assert prediction.broad_strict_oof_available.any()
    assert prediction.tail_strict_oof_available.any()
    assert prediction.meta_strict_oof_available.any()
    assert not prediction.broad_strict_oof_available.all()
    assert not prediction.tail_strict_oof_available.all()
    assert not prediction.meta_strict_oof_available.all()
    # A tail row cannot exist without a strict same-side broad predecessor;
    # a meta row cannot exist without a strict same-side tail predecessor.
    assert prediction.loc[prediction.tail_strict_oof_available, "broad_strict_oof_available"].all()
    assert prediction.loc[prediction.meta_strict_oof_available, "tail_strict_oof_available"].all()
    tail_columns = {columns for layer, columns, _ in records if layer == "tail"}
    meta_columns = {columns for layer, columns, _ in records if layer == "meta"}
    assert tail_columns == {("tail_signal", "__stage_iv_broad_same_side_oof_score")}
    assert meta_columns == {
        ("meta_context", "__stage_iv_tail_same_side_oof_score", "__stage_iv_broad_same_side_oof_score")
    }
    provenance = result.side_results["long"].fold_provenance
    assert {"broad", "tail", "meta"}.issubset(set(provenance.layer))
    assert (
        pd.to_datetime(provenance.train_max_label_available_ts, utc=True)
        < pd.to_datetime(provenance.validation_start_ts, utc=True)
    ).all()
    assert result.manifest["ranking"].startswith("pooled global")


@pytest.mark.parametrize(
    ("route", "tail_has_broad", "meta_has_broad"),
    [("neither", False, False), ("tail", True, False), ("meta", False, True), ("both", True, True)],
)
def test_broad_output_routes_are_explicit(route, tail_has_broad, meta_has_broad) -> None:
    records: list[tuple[str, tuple[str, ...], int]] = []
    run_stage_iv_broad_to_tail_ablation([_plan("short", route=route)], fitter=_fitter_records(records))
    tail_columns = next(columns for layer, columns, _ in records if layer == "tail")
    meta_columns = next(columns for layer, columns, _ in records if layer == "meta")
    assert ("__stage_iv_broad_same_side_oof_score" in tail_columns) is tail_has_broad
    assert ("__stage_iv_broad_same_side_oof_score" in meta_columns) is meta_has_broad
    assert "__stage_iv_tail_same_side_oof_score" in meta_columns


def test_later_label_cannot_change_earlier_stage_iv_strict_oof_scores() -> None:
    records_left: list[tuple[str, tuple[str, ...], int]] = []
    records_right: list[tuple[str, tuple[str, ...], int]] = []
    left = run_stage_iv_broad_to_tail_ablation(
        [_plan("long", final_net_shift=0.0)], fitter=_fitter_records(records_left)
    ).predictions
    right = run_stage_iv_broad_to_tail_ablation(
        [_plan("long", final_net_shift=50_000.0)], fitter=_fitter_records(records_right)
    ).predictions
    earlier = left.decision_ts.lt(left.decision_ts.max())
    columns = [
        "broad_same_side_oof_score", "tail_same_side_oof_score",
        "meta_same_side_residual_oof_score", "meta_reconstructed_expected_net_bps",
    ]
    np.testing.assert_allclose(left.loc[earlier, columns], right.loc[earlier, columns], equal_nan=True)


def test_later_base_target_cannot_change_earlier_broad_or_tail_oof_scores() -> None:
    left = run_stage_iv_broad_to_tail_ablation(
        [_plan("short", final_target_shift=0.0)], fitter=_fitter_records([])
    ).predictions
    right = run_stage_iv_broad_to_tail_ablation(
        [_plan("short", final_target_shift=50_000.0)], fitter=_fitter_records([])
    ).predictions
    earlier = left.decision_ts.lt(left.decision_ts.max())
    np.testing.assert_allclose(
        left.loc[earlier, ["broad_same_side_oof_score", "tail_same_side_oof_score"]],
        right.loc[earlier, ["broad_same_side_oof_score", "tail_same_side_oof_score"]],
        equal_nan=True,
    )


def test_metrics_rank_one_pooled_book_then_attribute_month_and_side() -> None:
    ledger = pd.DataFrame({
        "candidate_key": ["long::a", "short::b", "long::c", "short::d"],
        "side_name": ["long", "short", "long", "short"],
        "decision_ts": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
            "2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z",
        ]),
        "exact_net_bps": [20.0, -100.0, 500.0, 300.0],
        "cost_bps": [100.0] * 4,
        "score": [100.0, 90.0, 80.0, 70.0],
    })
    metric = pooled_global_stage_iv_metrics(ledger, score_column="score", layer="test", top_fractions=(0.25,))
    pooled = metric.loc[metric.scope.eq("pooled_global")].iloc[0]
    contribution = metric.loc[metric.scope.eq("selected_contribution")].iloc[0]
    assert pooled.net_bps_per_trade == 20.0
    assert pooled.gross_bps_per_trade == 120.0
    assert pooled.selection == "pooled_global_once_no_timestamp_month_or_side_rerank"
    assert (contribution.month, contribution.side) == ("2024-01", "long")


def test_invalid_x_is_rejected() -> None:
    with pytest.raises(ValueError, match="20%, 30%, 40%, or 50%"):
        run_stage_iv_broad_to_tail_ablation([_plan("long", tail_fraction=0.10)], fitter=_fitter_records([]))
