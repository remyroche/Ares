from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.run_label_feature_store_model_smoke as smoke


def test_ae_gmm_crossfit_availability_features_are_materialized(monkeypatch: pytest.MonkeyPatch) -> None:
    n_train = 1000
    n_valid = 10
    x_train = pd.DataFrame(
        np.arange(n_train * 3, dtype=np.float32).reshape(n_train, 3),
        columns=["f0", "f1", "f2"],
    )
    x_valid = pd.DataFrame(
        np.arange(n_valid * 3, dtype=np.float32).reshape(n_valid, 3),
        columns=["f0", "f1", "f2"],
    )
    side_train = np.where(np.arange(n_train) % 2 == 0, 1.0, -1.0).astype(np.float32)
    side_valid = np.where(np.arange(n_valid) % 2 == 0, 1.0, -1.0).astype(np.float32)
    train_metrics = pd.DataFrame(
        {
            "u_policy_net": np.linspace(-0.01, 0.01, n_train, dtype=np.float32),
            "mae_norm": np.zeros(n_train, dtype=np.float32),
            "is_timeout": np.zeros(n_train, dtype=np.float32),
            "side": side_train,
        }
    )
    valid_metrics = pd.DataFrame(
        {
            "u_policy_net": np.linspace(-0.01, 0.01, n_valid, dtype=np.float32),
            "mae_norm": np.zeros(n_valid, dtype=np.float32),
            "is_timeout": np.zeros(n_valid, dtype=np.float32),
            "side": side_valid,
        }
    )
    train_frame = pd.DataFrame({"__ts__": pd.date_range("2026-01-01", periods=n_train, freq="h")})
    fit_time_bucket_lengths: list[int] = []

    def fake_fit(*args, **kwargs):
        targets = kwargs.get("economic_targets") or {}
        assert "time_bucket" in targets
        time_bucket = np.asarray(targets["time_bucket"])
        assert time_bucket.dtype.kind in {"f", "i", "u"}
        assert np.isfinite(time_bucket).all()
        fit_time_bucket_lengths.append(int(time_bucket.size))
        return {
            "enabled": True,
            "gmm_n_components": 2,
            "gmm_reg_covar": 1e-4,
            "smooth_lambda": 0.9,
            "hpo_report_count": 1,
            "selected_config": {
                "economic_regime_separation": 0.1,
                "target_signature_score": 0.2,
                "path_aware_hpo": True,
                "path_cleanliness_score": 0.3,
                "clean_positive_contrast": 0.4,
                "bad_mae_contrast": 0.5,
                "timeout_contrast": 0.6,
                "temporal_concentration_hpo": True,
                "temporal_concentration_score": 0.7,
                "max_cluster_time_bucket_share": 0.8,
                "temporal_stability_score": 0.9,
                "switch_rate": 0.1,
                "side_balance_score": 0.9,
                "min_occupancy": 0.2,
                "max_occupancy": 0.8,
            },
        }

    def fake_transform(x, state, *, index=None, prefix=""):
        idx = pd.RangeIndex(len(x)) if index is None else index
        return pd.DataFrame(
            {
                "gmm_prob_0": np.full(len(idx), 0.25, dtype=np.float32),
                "cluster_speed": np.full(len(idx), 0.50, dtype=np.float32),
            },
            index=idx,
        )

    monkeypatch.setattr(smoke, "AE_GMM_CROSSFIT_TRAIN_FEATURES", True)
    monkeypatch.setattr(smoke, "AE_GMM_SIDE_CONTEXT_MODE", "long_short")
    monkeypatch.setattr(smoke, "AE_GMM_SMOKE_FEATURE_POLICY", "soft")
    monkeypatch.setattr(smoke, "fit_ae_gmm_state", fake_fit)
    monkeypatch.setattr(smoke, "transform_ae_gmm_features", fake_transform)
    monkeypatch.setattr(
        smoke,
        "_chronological_inner_oof_splits",
        lambda **kwargs: [(np.arange(0, 600, dtype=np.int64), np.arange(600, 1000, dtype=np.int64))],
    )

    full_valid_context: dict[str, object] = {}
    x_train_out, x_valid_out, emitted, diag = smoke._append_fold_ae_gmm_state_features(
        x_train=x_train,
        x_valid=x_valid,
        train_frame=train_frame,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        enabled=True,
        max_train_rows=1000,
        ae_max_iter=2,
        random_state=17,
        output_feature_subset=[
            "f0",
            "f1",
            "f2",
            "gmm_prob_0",
            "ae_gmm_oof_available",
            "long_gmm_prob_0",
            "long_ae_gmm_oof_available",
            "short_gmm_prob_0",
            "short_ae_gmm_oof_available",
        ],
        valid_context_output=full_valid_context,
    )

    assert "ae_gmm_oof_available" in emitted
    assert "long_ae_gmm_oof_available" in emitted
    assert "short_ae_gmm_oof_available" in emitted
    assert x_train_out["ae_gmm_oof_available"].sum() == pytest.approx(400.0)
    assert x_valid_out["ae_gmm_oof_available"].sum() == pytest.approx(float(n_valid))
    assert x_train_out["long_ae_gmm_oof_available"].sum() == pytest.approx(200.0)
    assert x_train_out["short_ae_gmm_oof_available"].sum() == pytest.approx(200.0)
    assert x_valid_out["long_ae_gmm_oof_available"].sum() == pytest.approx(5.0)
    assert x_valid_out["short_ae_gmm_oof_available"].sum() == pytest.approx(5.0)
    assert diag["ae_gmm_state_train_feature_scope"] == "inner_chronological_oof"
    assert diag["ae_gmm_state_validation_feature_scope"] == "frozen_outer_train_artifact"
    assert diag["ae_gmm_state_crossfit_coverage"] == pytest.approx(0.4)
    assert diag["ae_gmm_state_crossfit_transformed_rows"] == 400
    assert diag["ae_gmm_state_crossfit_uncovered_rows"] == 600
    assert fit_time_bucket_lengths
    assert "cluster_speed" not in x_valid_out
    assert isinstance(full_valid_context.get("frame"), pd.DataFrame)
    assert "cluster_speed" in full_valid_context["frame"].columns
