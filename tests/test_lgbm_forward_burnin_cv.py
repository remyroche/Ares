import numpy as np
import pandas as pd

from extreme_price_movements import lgbm_pipeline as lp


def _patch_forward_defaults(monkeypatch):
    monkeypatch.setattr(lp, "LGBM_CV_MODE", "forward_burnin")
    monkeypatch.setattr(lp, "LGBM_PURGE_HOURS", 0.0)
    monkeypatch.setattr(lp, "LGBM_BASE_FORWARD_BURN_IN_DAYS", 365.0)
    monkeypatch.setattr(lp, "LGBM_META_FORWARD_VALIDATION_MONTHS", 6)
    monkeypatch.setattr(lp, "LGBM_AUX_FORWARD_VALIDATION_MONTHS", 6)
    monkeypatch.setattr(lp, "LGBM_FORWARD_MIN_TRAIN_ROWS", 5)
    monkeypatch.setattr(lp, "LGBM_FORWARD_MIN_VALID_ROWS", 1)
    monkeypatch.setattr(lp, "LGBM_AUX_FORWARD_MIN_VALID_ROWS", 1)
    monkeypatch.setattr(lp, "LGBM_FORWARD_BURNIN_STRICT", True)
    monkeypatch.setattr(lp, "LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK", False)
    monkeypatch.setattr(lp, "LGBM_FORWARD_SHORT_HISTORY_FALLBACK_FRAC", 0.70)


def test_forward_burnin_base_keeps_first_year_train_only(monkeypatch):
    _patch_forward_defaults(monkeypatch)
    timestamps = pd.date_range("2024-01-01", periods=520, freq="D", tz="UTC")
    y = (np.arange(len(timestamps)) % 2).astype(np.float32)

    splitter, y_split = lp._forward_burnin_splitter(
        y,
        True,
        42,
        timestamps=timestamps,
        n_splits=4,
        objective_mode="train_base",
    )
    folds = list(splitter.split(np.zeros(len(y_split)), y_split))
    cutoff = timestamps[0] + pd.Timedelta(days=365)

    assert folds
    for train_idx, valid_idx in folds:
        assert timestamps[valid_idx].min() >= cutoff
        assert timestamps[train_idx].max() < timestamps[valid_idx].min()
        assert not np.intersect1d(train_idx, valid_idx).size


def test_forward_burnin_meta_validates_only_last_six_months(monkeypatch):
    _patch_forward_defaults(monkeypatch)
    timestamps = pd.date_range("2024-01-01", periods=800, freq="D", tz="UTC")
    y = (np.arange(len(timestamps)) % 2).astype(np.float32)

    splitter, y_split = lp._forward_burnin_splitter(
        y,
        True,
        42,
        timestamps=timestamps,
        n_splits=3,
        objective_mode="train_meta",
    )
    folds = list(splitter.split(np.zeros(len(y_split)), y_split))
    cutoff = timestamps[-1] - pd.DateOffset(months=6)

    assert folds
    for train_idx, valid_idx in folds:
        assert timestamps[valid_idx].min() >= cutoff
        assert timestamps[train_idx].max() < timestamps[valid_idx].min()
        assert not np.intersect1d(train_idx, valid_idx).size


def test_forward_burnin_auxiliary_validates_only_last_six_months(monkeypatch):
    _patch_forward_defaults(monkeypatch)
    timestamps = pd.date_range("2024-01-01", periods=800, freq="D", tz="UTC")
    y = np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)

    splitter, y_split = lp._forward_burnin_splitter(
        y,
        False,
        42,
        timestamps=timestamps,
        n_splits=3,
        objective_mode="auxiliary_regression",
    )
    folds = list(splitter.split(np.zeros(len(y_split)), y_split))
    cutoff = timestamps[-1] - pd.DateOffset(months=6)

    assert folds
    for train_idx, valid_idx in folds:
        assert timestamps[valid_idx].min() >= cutoff
        assert timestamps[train_idx].max() < timestamps[valid_idx].min()
        assert not np.intersect1d(train_idx, valid_idx).size


def test_forward_burnin_auxiliary_rejects_undersupported_validation(
    monkeypatch,
):
    _patch_forward_defaults(monkeypatch)
    monkeypatch.setattr(lp, "LGBM_AUX_FORWARD_MIN_VALID_ROWS", 300)
    timestamps = pd.date_range("2024-01-01", periods=800, freq="D", tz="UTC")
    y = np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)

    with np.testing.assert_raises_regex(
        ValueError, "under-supported.*min_valid_rows=300"
    ):
        lp._forward_burnin_splitter(
            y,
            False,
            42,
            timestamps=timestamps,
            n_splits=3,
            objective_mode="auxiliary_regression",
        )


def test_time_spread_subsample_takes_beginning_middle_and_end():
    timestamps = pd.date_range("2024-01-01", periods=90, freq="D", tz="UTC")
    y = (np.arange(len(timestamps)) % 2).astype(np.float32)

    idx = lp._time_spread_subsample_indices(
        y,
        max_n=30,
        random_state=42,
        classifier=True,
        timestamps=timestamps,
    )

    assert len(idx) == 30
    assert int(np.sum(idx < 30)) >= 5
    assert int(np.sum((idx >= 30) & (idx < 60))) >= 5
    assert int(np.sum(idx >= 60)) >= 5


def test_forward_burnin_short_history_fallback_stays_chronological(monkeypatch):
    _patch_forward_defaults(monkeypatch)
    monkeypatch.setattr(lp, "LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK", True)
    timestamps = pd.date_range("2026-03-01", periods=30, freq="D", tz="UTC")
    y = (np.arange(len(timestamps)) % 2).astype(np.float32)

    splitter, y_split = lp._forward_burnin_splitter(
        y,
        True,
        42,
        timestamps=timestamps,
        n_splits=2,
        objective_mode="train_base",
    )
    folds = list(splitter.split(np.zeros(len(y_split)), y_split))
    fallback_cutoff = timestamps[int(np.floor(0.70 * len(timestamps)))]

    assert folds
    for train_idx, valid_idx in folds:
        assert timestamps[valid_idx].min() >= fallback_cutoff
        assert timestamps[train_idx].max() < timestamps[valid_idx].min()
        assert not np.intersect1d(train_idx, valid_idx).size


def test_latest_holdout_can_enable_local_short_history_fallback(monkeypatch):
    _patch_forward_defaults(monkeypatch)
    timestamps = pd.date_range("2025-04-01", periods=360, freq="D", tz="UTC")
    y = (np.arange(len(timestamps)) % 2).astype(np.float32)

    train_idx, valid_idx = lp._forward_burnin_latest_holdout_indices(
        y,
        True,
        42,
        timestamps=timestamps,
        objective_mode="train_base",
        allow_short_history_fallback=True,
    )

    fallback_cutoff = timestamps[int(np.floor(0.70 * len(timestamps)))]
    assert timestamps[valid_idx].min() >= fallback_cutoff
    assert timestamps[train_idx].max() < timestamps[valid_idx].min()
    assert not np.intersect1d(train_idx, valid_idx).size
    assert lp.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK is False


def test_distillation_skips_short_forward_cv_failure(monkeypatch):
    _patch_forward_defaults(monkeypatch)
    monkeypatch.setattr(lp, "LGBM_SKIP_DISTILLATION_ON_FORWARD_CV_FAILURE", True)

    def _raise_no_folds(*args, **kwargs):
        raise ValueError("forward_burnin CV produced no usable folds")

    monkeypatch.setattr(lp, "_cross_val_oof_lgbm", _raise_no_folds)
    X = pd.DataFrame({"x": np.arange(8, dtype=np.float32)})
    y = (np.arange(8) % 2).astype(np.float32)
    base_weight = np.ones(8, dtype=np.float32)

    weights, oof = lp._oof_distilled_sample_weights_lgbm(
        X,
        y,
        base_weight,
        ["x"],
        classifier=True,
        params={},
        timestamps=pd.date_range("2026-03-01", periods=8, freq="H", tz="UTC"),
        random_state=42,
        passes=2,
        label="test",
        objective_mode="train_meta",
    )

    assert np.allclose(weights, np.ones_like(weights))
    assert np.allclose(oof, np.full_like(oof, np.mean(y)))
