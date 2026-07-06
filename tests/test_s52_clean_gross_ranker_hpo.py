from __future__ import annotations

import pandas as pd

from scripts.run_gate3_side_soft_label_hpo import (
    _ae_gmm_fold_cache_payload,
    _ae_gmm_state_feature_random_state,
    _fold_cache_digest,
    _load_ae_gmm_fold_cache,
    _write_ae_gmm_fold_cache,
)
from scripts.run_s52_clean_gross_ranker_hpo import (
    _s52_topk_path_objective,
    _select_hpo_objective,
)


def test_s52_topk_path_objective_prefers_ev_weighted_clean_path_order() -> None:
    clean = {
        "objective": -1.0,
        "mean_top10_ev_weighted_first_touch_precision": 0.74,
        "mean_top20_ev_weighted_first_touch_precision": 0.61,
        "mean_top30_ev_weighted_first_touch_precision": 0.52,
        "mean_long_top10_ev_weighted_first_touch_precision": 0.66,
        "mean_short_top10_ev_weighted_first_touch_precision": 0.78,
        "mean_top10_mean_first_touch_gross": 0.004,
        "mean_top10_mean_first_touch_net": -0.006,
        "mean_top10_mean_ev": -0.002,
        "mean_top10_first_touch_bad_mae_1r_rate": 0.12,
        "mean_top10_mae_1r_before_mfe_1r_rate": 0.18,
        "mean_top10_mfe_1r_before_mae_1r_rate": 0.78,
        "mean_top10_timeout_rate": 0.02,
        "mean_top10_mean_underwater_bars_before_mfe": 5.0,
        "mean_top10_mean_underwater_fraction_before_mfe": 0.20,
        "mean_top10_mean_max_adverse_before_mfe_1r": 0.70,
    }
    dirty_high_gross = {
        **clean,
        "objective": 10.0,
        "mean_top10_ev_weighted_first_touch_precision": 0.68,
        "mean_top20_ev_weighted_first_touch_precision": 0.56,
        "mean_top30_ev_weighted_first_touch_precision": 0.50,
        "mean_top10_mean_first_touch_gross": 0.009,
        "mean_top10_first_touch_bad_mae_1r_rate": 0.35,
        "mean_top10_mae_1r_before_mfe_1r_rate": 0.42,
        "mean_top10_mfe_1r_before_mae_1r_rate": 0.55,
        "mean_top10_mean_underwater_bars_before_mfe": 12.0,
        "mean_top10_mean_underwater_fraction_before_mfe": 0.50,
        "mean_top10_mean_max_adverse_before_mfe_1r": 1.60,
    }

    assert _s52_topk_path_objective(clean) > _s52_topk_path_objective(dirty_high_gross)
    assert _select_hpo_objective(clean, "s52_topk_path") == _s52_topk_path_objective(clean)
    assert _select_hpo_objective(dirty_high_gross, "inherited_objective") == 10.0


def test_ae_gmm_fold_cache_roundtrips_augmented_matrices(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01", "2026-04-02"]),
            "__symbol__": ["BTC", "ETH"],
        }
    )
    valid = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-05-01"]),
            "__symbol__": ["BTC"],
        }
    )
    payload = _ae_gmm_fold_cache_payload(
        labels_path=tmp_path / "labels",
        feature_dir=tmp_path / "features",
        feature_list_csv=tmp_path / "features.csv",
        month="2026-05",
        fold_i=0,
        train=train,
        valid=valid,
        features=["f1", "f2"],
        include_ae_gmm_state_features=True,
        ae_gmm_state_feature_max_train_rows=60000,
        ae_gmm_state_feature_max_iter=64,
        seed=42,
        random_state=42,
    )
    digest = _fold_cache_digest(payload)
    x_train = pd.DataFrame({"f1": [1.0, 2.0], "ae_gmm_oof_available": [0.0, 1.0]})
    x_valid = pd.DataFrame({"f1": [3.0], "ae_gmm_oof_available": [1.0]})
    diag = {"ae_gmm_state_feature_status": "ok"}

    _write_ae_gmm_fold_cache(
        cache_dir=tmp_path / "cache",
        digest=digest,
        payload=payload,
        x_train=x_train,
        x_valid=x_valid,
        generated=["ae_gmm_oof_available"],
        ae_diag=diag,
    )
    loaded = _load_ae_gmm_fold_cache(
        cache_dir=tmp_path / "cache",
        digest=digest,
        expected_payload=payload,
    )

    assert loaded is not None
    loaded_train, loaded_valid, generated, loaded_diag = loaded
    assert loaded_train.round(6).equals(x_train.astype("float32"))
    assert loaded_valid.round(6).equals(x_valid.astype("float32"))
    assert generated == ["ae_gmm_oof_available"]
    assert loaded_diag["ae_gmm_state_feature_cache_status"] == "hit"


def test_ae_gmm_state_feature_seed_is_separate_from_model_seed() -> None:
    assert _ae_gmm_state_feature_random_state(fold_i=0, seed=42) == 42
    assert _ae_gmm_state_feature_random_state(fold_i=2, seed=42) == 244

    # Replaying a model with another model seed should still be able to request
    # the original deterministic AE/GMM fold features by keeping this seed fixed.
    model_seed = 4042
    assert _ae_gmm_state_feature_random_state(fold_i=2, seed=42) != (
        model_seed + 2 * 101
    )
