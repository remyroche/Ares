import numpy as np

from extreme_price_movements.bayesian_changepoint import (
    BOCPDConfig,
    bocpd_student_t,
    bocpd_student_t_run_summary,
    robust_scale_train_oos,
    synchronized_break_score,
)


def test_bocpd_spikes_on_mean_shift() -> None:
    values = np.concatenate([np.zeros(80), np.full(80, 4.0)]).astype(np.float32)
    scores = bocpd_student_t(values, BOCPDConfig(expected_run_hours=48, max_run_hours=96))
    assert int(np.nanargmax(scores)) == 80
    assert float(scores[80]) > 0.50


def test_bocpd_prefix_is_not_changed_by_future_values() -> None:
    prefix = np.linspace(-0.25, 0.25, 100, dtype=np.float32)
    left = np.concatenate([prefix, np.zeros(40, dtype=np.float32)])
    right = np.concatenate([prefix, np.full(40, 6.0, dtype=np.float32)])
    config = BOCPDConfig(expected_run_hours=48, max_run_hours=72)
    np.testing.assert_allclose(
        bocpd_student_t(left, config)[: len(prefix)],
        bocpd_student_t(right, config)[: len(prefix)],
        rtol=0.0,
        atol=0.0,
    )


def test_bocpd_run_summary_is_causal_and_bounded() -> None:
    prefix = np.zeros(40, dtype=np.float32)
    config = BOCPDConfig(expected_run_hours=24, max_run_hours=48)
    left = bocpd_student_t_run_summary(np.r_[prefix, np.ones(20)], config)
    right = bocpd_student_t_run_summary(np.r_[prefix, np.full(20, -5.0)], config)
    np.testing.assert_allclose(left[: len(prefix)], right[: len(prefix)], rtol=0.0, atol=0.0)
    assert np.isfinite(left).all()
    assert np.all((left[:, 0] >= 0.0) & (left[:, 0] <= 1.0))
    assert np.all((left[:, 3] >= 0.0) & (left[:, 3] <= 1.0))


def test_synchronized_break_requires_multiple_train_tail_events() -> None:
    train = np.tile(np.linspace(0.0, 0.10, 100, dtype=np.float32)[:, None], (1, 3))
    score = np.asarray([[0.02, 0.02, 0.02], [0.90, 0.92, 0.95]], dtype=np.float32)
    composite, count, thresholds = synchronized_break_score(train, score, individual_tail=0.95)
    assert thresholds.shape == (3,)
    assert int(count[0]) == 0
    assert int(count[1]) == 3
    assert float(composite[1]) > float(composite[0])


def test_scaler_uses_train_reference_only() -> None:
    train = np.arange(20, dtype=np.float32)
    score = np.asarray([1000.0], dtype=np.float32)
    train_scaled, score_scaled, median, _ = robust_scale_train_oos(train, score)
    assert float(median) == 9.5
    assert float(score_scaled[0]) == 8.0
    assert np.isfinite(train_scaled).all()
