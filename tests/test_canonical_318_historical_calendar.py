import pandas as pd
import pytest

from extreme_price_movements.canonical_318_historical_calendar import (
    HistoricalOOFCalendarSpec,
    build_historical_base_oof_calendar,
    validate_strict_oof_predictions,
)


def _identities(start: str = "2025-01-01", end: str = "2025-05-01") -> pd.DataFrame:
    rows = []
    for timestamp in pd.date_range(start, end, freq="h", inclusive="left", tz="UTC"):
        for side in ("long", "short"):
            for index in range(100):
                rows.append(
                    {
                        "__ts__": timestamp,
                        "__symbol__": f"S{index:03d}/USD:USD",
                        "side_name": side,
                        "candidate_id": f"{timestamp.isoformat()}|{side}|{index}",
                        "__decision_ts__": timestamp + pd.Timedelta(hours=1),
                        "__label_resolution_ts__": timestamp + pd.Timedelta(hours=25),
                    }
                )
    return pd.DataFrame(rows)


def test_monthly_calendar_scores_every_frozen_identity_once_and_purges_labels() -> None:
    source = _identities()
    spec = HistoricalOOFCalendarSpec(
        score_start=pd.Timestamp("2025-02-01", tz="UTC"),
        score_end=pd.Timestamp("2025-05-01", tz="UTC"),
        minimum_train_rows_per_side=1_000,
        maximum_fit_rows_per_side=2_000,
    )
    frozen, sampled, contract = build_historical_base_oof_calendar(source, spec=spec)
    expected = source.loc[source["__ts__"].ge(spec.score_start) & source["__ts__"].lt(spec.score_end)]
    assert len(frozen) == len(expected)
    assert not frozen["candidate_id"].duplicated().any()
    assert frozen.groupby("oof_fold")["__ts__"].agg(["min", "max"]).shape[0] == 3
    for fold in contract["calendar"]["folds"]:
        start = pd.Timestamp(fold["validation_start_utc"])
        local = sampled.loc[
            sampled["oof_fold"].eq(fold["fold"])
            & sampled["side_name"].eq(fold["side"])
        ]
        assert local["__label_resolution_ts__"].lt(start).all()
        assert local["__decision_ts__"].lt(start - pd.Timedelta(hours=24)).all()
        assert len(local) <= 2_000
    assert contract["sampling_contract"]["forbidden_inputs"][-1] == "evaluation_outcomes"


def test_fails_when_first_calendar_fold_lacks_per_side_resolved_support() -> None:
    source = _identities(start="2025-01-25")
    spec = HistoricalOOFCalendarSpec(
        score_start=pd.Timestamp("2025-02-01", tz="UTC"),
        score_end=pd.Timestamp("2025-05-01", tz="UTC"),
        minimum_train_rows_per_side=50_000,
        maximum_fit_rows_per_side=60_000,
    )
    with pytest.raises(ValueError, match=r"resolved\+embargoed training rows"):
        build_historical_base_oof_calendar(source, spec=spec)


def test_prediction_validator_rejects_wrong_fold_even_when_identity_matches() -> None:
    source = _identities()
    spec = HistoricalOOFCalendarSpec(
        score_start=pd.Timestamp("2025-02-01", tz="UTC"),
        score_end=pd.Timestamp("2025-03-01", tz="UTC"),
        minimum_train_rows_per_side=1_000,
        maximum_fit_rows_per_side=2_000,
    )
    frozen, _, _ = build_historical_base_oof_calendar(source, spec=spec)
    valid = frozen.loc[:, ["__ts__", "__symbol__", "side_name", "candidate_id", "oof_fold"]].copy()
    validate_strict_oof_predictions(frozen, valid)
    valid.loc[0, "oof_fold"] = "wrong_fold"
    with pytest.raises(ValueError, match="wrong OOF fold"):
        validate_strict_oof_predictions(frozen, valid)
