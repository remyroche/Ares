from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.tp6_transport_validation import (
    FINAL_OOS,
    TRANSPORT_A,
    TRANSPORT_B,
    TP6TransportValidationError,
    evaluate_transport,
    make_final_oos_spec,
    validate_feature_contract,
    write_transport_evaluation,
)


def _row(
    candidate_id: str,
    side_name: str,
    decision_ts: str,
    score_bps: float,
    net_bps: float,
    *,
    feature: float = 1.0,
    reference_offset_hours: float = -1.0,
) -> dict[str, object]:
    decision = pd.Timestamp(decision_ts, tz="UTC")
    return {
        "candidate_id": candidate_id,
        "side_name": side_name,
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "score_fit_resolved_before": decision + pd.Timedelta(hours=reference_offset_hours),
        "score_bps": score_bps,
        "net_bps": net_bps,
        "gross_bps": net_bps + 100.0,
        "feature": feature,
    }


def _transport_a_frame() -> pd.DataFrame:
    # Training is prior-resolved before 2024-01-01.  The four test rows span
    # two months and have a deliberately cross-side global score order.
    return pd.DataFrame(
        [
            _row("train", "long", "2023-12-30 10:00", 0.0, -20.0),
            _row("test_l1", "long", "2024-01-02 00:00", 90.0, 20.0),
            _row("test_l2", "long", "2024-01-03 00:00", 80.0, 10.0),
            _row("test_s1", "short", "2024-05-02 00:00", 70.0, -5.0),
            _row("test_s2", "short", "2024-05-03 00:00", 10.0, -30.0),
        ]
    )


def test_transport_windows_are_contiguous_and_final_oos_is_held_out() -> None:
    assert TRANSPORT_A.train_windows[0].start == pd.Timestamp("2023-04-01", tz="UTC")
    assert TRANSPORT_A.first_test_decision == pd.Timestamp("2024-01-01", tz="UTC")
    assert TRANSPORT_A.test_windows[0].end == pd.Timestamp("2024-07-01", tz="UTC")
    assert TRANSPORT_B.first_test_decision == pd.Timestamp("2024-07-01", tz="UTC")
    assert TRANSPORT_B.test_windows[0].end == pd.Timestamp("2024-11-01", tz="UTC")
    assert FINAL_OOS.first_test_decision == pd.Timestamp("2024-11-01", tz="UTC")
    generic = make_final_oos_spec(
        name="later_final", train_start="2024-01-01", test_start="2024-02-01", test_end="2024-03-01"
    )
    assert generic.train_windows[0].end == generic.first_test_decision


def test_global_common_bps_tail_is_not_ranked_per_side_or_timestamp_and_reports_empty_side() -> None:
    result = evaluate_transport(
        _transport_a_frame(),
        transport=TRANSPORT_A,
        score_column="score_bps",
        feature_columns=["feature"],
        prior_resolved_columns=["score_fit_resolved_before"],
        top_fractions=(0.50,),
    )
    global_row = result.metrics.query("scope == 'global'").iloc[0]
    assert global_row["trades"] == 2
    assert global_row["net_bps_per_trade"] == pytest.approx(15.0)
    # Both globally selected rows are long.  A per-side ranking would have
    # admitted a short; the explicit zero-trade short row guards against that.
    short_row = result.metrics.query("scope == 'side' and side_name == 'short'").iloc[0]
    assert short_row["trades"] == 0
    assert pd.isna(short_row["net_bps_per_trade"])
    assert set(result.metrics.loc[result.metrics.scope.eq("month"), "period"]) == {"2024-01", "2024-05"}
    assert set(result.metrics.loc[result.metrics.scope.eq("quarter"), "period"]) == {"2024Q1", "2024Q2"}
    assert set(result.transport_gates["gate"]) == {"minimum_trade_support", "net_bps_per_trade"}


def test_prior_resolved_and_label_availability_gates_fail_closed() -> None:
    bad_label = _transport_a_frame()
    bad_label.loc[0, "decision_ts"] = pd.Timestamp("2023-12-31 11:00", tz="UTC")
    bad_label.loc[0, "label_available_ts"] = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    with pytest.raises(TP6TransportValidationError, match="resolved before"):
        evaluate_transport(
            bad_label,
            transport=TRANSPORT_A,
            score_column="score_bps",
            feature_columns=["feature"],
            prior_resolved_columns=["score_fit_resolved_before"],
        )

    bad_reference = _transport_a_frame()
    bad_reference.loc[1, "score_fit_resolved_before"] = bad_reference.loc[1, "decision_ts"]
    with pytest.raises(TP6TransportValidationError, match="strictly earlier"):
        evaluate_transport(
            bad_reference,
            transport=TRANSPORT_A,
            score_column="score_bps",
            feature_columns=["feature"],
            prior_resolved_columns=["score_fit_resolved_before"],
        )


def test_feature_contract_requires_99pct_coverage_and_refuses_outcome_columns() -> None:
    frame = _transport_a_frame()
    frame.loc[0, "feature"] = float("nan")
    with pytest.raises(TP6TransportValidationError, match="coverage"):
        validate_feature_contract(frame, ["feature"])
    with pytest.raises(TP6TransportValidationError, match="outcome/control"):
        validate_feature_contract(_transport_a_frame(), ["net_bps"])


def test_writer_produces_transport_gates_for_ablation_results(tmp_path) -> None:
    result = evaluate_transport(
        _transport_a_frame(),
        transport=TRANSPORT_A,
        score_column="score_bps",
        feature_columns=["feature"],
        prior_resolved_columns=["score_fit_resolved_before"],
        top_fractions=(0.50,),
    )
    paths = write_transport_evaluation(result, tmp_path)
    assert paths["transport_gates"].name == "transport_gates.parquet"
    assert paths["transport_gates"].exists()
    persisted = pd.read_parquet(paths["transport_gates"])
    assert not persisted.empty
