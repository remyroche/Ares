from scripts import compare_packb_short_feature_lists_common_cohort as gate


def test_rank_uses_mean_then_worst_then_stability() -> None:
    rows = [
        {
            "candidate_id": "a",
            "mean_objective": 0.4,
            "worst_fold_objective": 0.2,
            "objective_std": 0.1,
        },
        {
            "candidate_id": "b",
            "mean_objective": 0.4,
            "worst_fold_objective": 0.3,
            "objective_std": 0.2,
        },
        {
            "candidate_id": "c",
            "mean_objective": 0.3,
            "worst_fold_objective": 0.3,
            "objective_std": 0.01,
        },
    ]

    assert [row["candidate_id"] for row in gate._rank(rows)] == ["b", "a", "c"]


def test_aggregate_requires_three_folds() -> None:
    rows = [{"objective": 0.1}, {"objective": 0.2}]

    try:
        gate._aggregate("candidate", rows)
    except gate.CommonCohortGateError as exc:
        assert "exactly three folds" in str(exc)
    else:
        raise AssertionError("two-fold comparison should fail closed")


def test_aggregate_records_mean_worst_and_sample_std() -> None:
    result = gate._aggregate(
        "candidate",
        [{"objective": 0.1}, {"objective": 0.2}, {"objective": 0.3}],
    )

    assert result["candidate_id"] == "candidate"
    assert abs(result["mean_objective"] - 0.2) < 1e-12
    assert result["worst_fold_objective"] == 0.1
    assert abs(result["objective_std"] - 0.1) < 1e-12
