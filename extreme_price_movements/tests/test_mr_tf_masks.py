import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from extreme_price_movements.config import CFG
from extreme_price_movements.mr_tf_masks import (
    MR_MASK_COL,
    ROUTE_COL,
    ROUTE_SCORE_SOURCE_COL,
    TF_MASK_COL,
    append_mr_tf_route_features,
    apply_mr_tf_masks,
    compare_specialist_to_baseline,
    mr_tf_route_from_path,
    mr_tf_masks_enabled,
    overlay_mr_tf_route_predictions,
    optimize_mr_tf_mask_params,
    route_support_diagnostics,
)


def _synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "adx": [30.0, 10.0, 18.0, 35.0],
            "ema_gap": [0.20, 0.00, 0.00, -0.20],
            "ret_3h": [0.30, 0.00, -0.05, 0.20],
            "stretch_atr": [0.10, -1.20, 0.05, -1.40],
            "ret_2h": [0.00, 0.10, 0.00, -0.30],
        }
    )


def test_mr_tf_masks_disabled_by_default() -> None:
    assert mr_tf_masks_enabled({}) is False
    assert mr_tf_masks_enabled(CFG) is False


def test_mr_tf_mask_generation_is_deterministic_from_synthetic_rows() -> None:
    params = {
        "q_adx_tf": 0.65,
        "q_adx_mr": 0.45,
        "q_stretch_mr": 0.75,
        "N_tf": 3,
        "N_mr": 2,
        "ema_gap_min_tf": 0.0,
        "mom_min_tf": 0.0,
        "stretch_min_mr": 0.25,
        "reversal_min_mr": -0.25,
        "thresholds": {"adx_tf": 20.0, "adx_mr": 15.0, "stretch_mr": 0.40},
    }
    routed, diag = apply_mr_tf_masks(_synthetic_frame(), side="long", params=params)

    assert routed[ROUTE_COL].tolist() == ["tf", "mr", "mixed", "mixed"]
    assert routed[MR_MASK_COL].tolist() == [0, 1, 0, 0]
    assert routed[TF_MASK_COL].tolist() == [1, 0, 0, 0]
    assert diag["counts"] == {"mr": 1, "tf": 1, "mixed": 2}
    assert diag["source_columns"]["adx"] == "adx"


def test_mr_tf_mask_generation_can_use_existing_persistence_features() -> None:
    frame = _synthetic_frame().assign(
        hurst_proxy_24=[0.80, 0.20, 0.50, 0.55],
        return_autocorr_48=[0.60, -0.40, 0.00, 0.10],
        path_efficiency_24=[0.90, 0.20, 0.40, 0.45],
        choppiness_index_20=[20.0, 80.0, 50.0, 45.0],
        direction_entropy_20=[0.10, 0.90, 0.50, 0.55],
    )
    params = {
        "q_adx_tf": 0.65,
        "q_adx_mr": 0.45,
        "q_stretch_mr": 0.75,
        "q_persist_tf": 0.50,
        "q_persist_mr": 0.50,
        "N_tf": 3,
        "N_mr": 2,
        "ema_gap_min_tf": 0.0,
        "mom_min_tf": 0.0,
        "stretch_min_mr": 0.25,
        "reversal_min_mr": -0.25,
        "persistence_axis": "composite",
        "thresholds": {
            "adx_tf": 20.0,
            "adx_mr": 15.0,
            "stretch_mr": 0.40,
            "persist_tf": 0.60,
            "persist_mr": 0.40,
        },
    }

    routed, diag = apply_mr_tf_masks(frame, side="long", params=params)

    assert routed[ROUTE_COL].tolist() == ["tf", "mr", "mixed", "mixed"]
    assert diag["persistence_axis"] == "composite"
    assert diag["persistence_source_columns"]["hurst"] == "hurst_proxy_24"
    assert diag["persistence_source_columns"]["autocorr"] == "return_autocorr_48"
    assert diag["persistence_source_columns"]["efficiency"] == "path_efficiency_24"
    assert diag["persistence_source_columns"]["choppiness"] == "choppiness_index_20"
    assert diag["persistence_source_columns"]["entropy"] == "direction_entropy_20"


def test_mr_tf_mask_generation_can_use_route_specific_quality_families() -> None:
    frame = _synthetic_frame().assign(
        trend_snr=[1.20, 0.10, 0.20, 0.30],
        dist_oiw_signed_delta_12h_atr=[0.00, -0.80, 0.00, 0.80],
        loc_bb_channel_pos_48=[0.50, 0.05, 0.50, 0.95],
        rsi=[55.0, 20.0, 50.0, 85.0],
        vol_compression=[0.10, 0.80, 0.20, 0.40],
    )
    params = {
        "q_adx_tf": 0.65,
        "q_adx_mr": 0.45,
        "q_stretch_mr": 0.75,
        "q_tf_quality": 0.60,
        "q_mr_quality": 0.60,
        "N_tf": 3,
        "N_mr": 2,
        "ema_gap_min_tf": 0.0,
        "mom_min_tf": 0.0,
        "stretch_min_mr": 0.25,
        "reversal_min_mr": -0.25,
        "tf_quality_axis": "trend",
        "mr_quality_axis": "range_position",
        "thresholds": {
            "adx_tf": 20.0,
            "adx_mr": 15.0,
            "stretch_mr": 0.40,
            "tf_quality": 0.60,
            "mr_quality": 0.20,
        },
    }

    routed, diag = apply_mr_tf_masks(frame, side="long", params=params)

    assert routed[ROUTE_COL].tolist() == ["tf", "mr", "mixed", "mixed"]
    assert diag["tf_quality_axis"] == "trend"
    assert diag["mr_quality_axis"] == "range_position"
    assert diag["quality_source_columns"]["trend"] == "trend_snr"
    assert diag["quality_source_columns"]["range_position"] == "dist_oiw_signed_delta_12h_atr"


def test_mr_tf_route_replay_uses_persisted_thresholds() -> None:
    first, diag = apply_mr_tf_masks(_synthetic_frame(), side="long")
    replay, replay_diag = apply_mr_tf_masks(
        _synthetic_frame(),
        side="long",
        params=diag["params"],
    )

    assert replay[ROUTE_COL].tolist() == first[ROUTE_COL].tolist()
    assert replay_diag["params_hash"] == diag["params_hash"]


def test_mr_tf_optuna_objective_charges_support_loss() -> None:
    pytest.importorskip("optuna")
    n = 60
    frame = pd.DataFrame(
        {
            "adx": np.r_[np.full(20, 30.0), np.full(20, 5.0), np.full(20, 18.0)],
            "ema_gap": np.r_[np.full(20, 1.0), np.zeros(40)],
            "ret_3h": np.r_[np.full(20, 1.0), np.zeros(40)],
            "stretch_atr": np.r_[
                np.zeros(20),
                np.full(10, -1.0),
                np.full(10, -3.0),
                np.zeros(20),
            ],
            "ret_2h": np.r_[np.zeros(20), np.full(10, -0.10), np.full(10, 0.20), np.zeros(20)],
        }
    )
    y = np.r_[np.tile([1.0, 0.0], 10), np.tile([1.0, 0.0], 10), np.full(20, 1.0)]
    returns = np.r_[
        np.tile([1.0, -0.1], 10),
        np.tile([1.0, -0.1], 10),
        np.full(20, -1.0),
    ]
    cfg = {
        "mr_tf_masks": {
            "optuna_trials": 1,
            "optuna_patience": 1,
            "min_train_samples": 4,
            "support_loss_hurdle_ratio": 0.25,
            "support_value_power": 0.5,
            "min_earned_quality_uplift": 0.0,
            "optuna_use_numba": False,
        }
    }

    _, diag = optimize_mr_tf_mask_params(
        frame,
        y=y,
        returns=returns,
        side="long",
        cfg=cfg,
    )

    assert diag["selected"] is True
    assert diag["objective"] == "route_quality_proxy_support_cost_adjusted"
    obj_diag = diag["best_objective_diagnostics"]
    assert obj_diag["support_loss"] > 0.0
    assert obj_diag["required_uplift"] > 0.0
    assert obj_diag["earned_quality_uplift"] > 0.0


def test_route_support_prunes_mixed_and_requires_two_classes() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mr_mask = np.array([True, False, True, False])
    tf_single_class_mask = np.array([True, True, False, False])

    mr_support = route_support_diagnostics(y, mr_mask, min_train_samples=2)
    tf_support = route_support_diagnostics(
        y, tf_single_class_mask, min_train_samples=2
    )

    assert mr_support["ok"] is True
    assert mr_support["n"] == 2
    assert tf_support["ok"] is False
    assert tf_support["reason"] == "single_class_route"


def test_baseline_comparison_promotes_only_positive_route_uplift() -> None:
    returns = np.array([0.03, 0.02, -0.01, -0.02], dtype=np.float32)
    baseline = np.array([0.10, 0.20, 0.90, 0.80], dtype=np.float32)
    specialist = np.array([0.90, 0.80, 0.20, 0.10], dtype=np.float32)

    comparison = compare_specialist_to_baseline(
        specialist_pred=specialist,
        baseline_pred=baseline,
        returns=returns,
        margin=0.0,
        top_frac=0.30,
    )

    assert comparison["promoted"] is True
    assert comparison["uplift"] > 0.0


def test_route_path_and_feature_helpers_are_canonical() -> None:
    assert (
        mr_tf_route_from_path("meta_oof_short_strategy_mr_tbm_clf.parquet") == "mr"
    )
    assert (
        mr_tf_route_from_path("meta_oof_long_strategy_tf_tbm_clf.parquet") == "tf"
    )

    routed = append_mr_tf_route_features(
        pd.DataFrame(
            {
                ROUTE_COL: ["mr", "tf", "mixed", "unknown"],
                ROUTE_SCORE_SOURCE_COL: ["mr", "general", "general", "general"],
            }
        )
    )

    assert routed["mr_tf_route_mr"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert routed["mr_tf_route_tf"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert routed["mr_tf_route_known"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert routed["mr_tf_specialist_active"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_route_overlay_replaces_only_matching_rows() -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T02:00Z"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "CCC"],
            ROUTE_COL: ["mr", "tf", "mixed"],
            "oof_pred": [0.10, 0.20, 0.30],
            "oof_p_move": [0.11, 0.21, 0.31],
        }
    )
    route = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB"],
            "oof_pred": [0.90, 0.80],
            "oof_p_move": [0.91, 0.81],
        }
    )

    overlaid, diag = overlay_mr_tf_route_predictions(base, route, route="mr")

    assert diag["overlay_rows"] == 1
    assert overlaid["oof_pred"].tolist() == [0.90, 0.20, 0.30]
    assert overlaid["oof_p_move"].tolist() == [0.91, 0.21, 0.31]
    assert overlaid["mr_tf_general_oof_pred"].tolist() == [0.10, 0.20, 0.30]
    assert overlaid[ROUTE_SCORE_SOURCE_COL].tolist() == ["mr", "general", "general"]


def test_policy_meta_oof_loader_overlays_routes_without_new_strategy(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from extreme_price_movements.policy_optimiser import (
        _policy_oof_strategy_key,
        load_meta_oof_predictions,
    )

    assert (
        _policy_oof_strategy_key(
            Path("meta_oof_short_strategy_mr_tbm_clf.parquet"),
            meta=True,
        )
        == "strategy"
    )

    data_root = tmp_path / "data"
    run_id = "run"
    oof_dir = data_root / "artifacts" / run_id / "meta_oof"
    oof_dir.mkdir(parents=True)
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB"],
            ROUTE_COL: ["mr", "tf"],
            "oof_pred": [0.10, 0.20],
        }
    )
    route = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00Z"], utc=True),
            "symbol": ["AAA"],
            "oof_pred": [0.95],
        }
    )
    base.to_parquet(oof_dir / "meta_oof_short_strategy_tbm_clf.parquet")
    route.to_parquet(oof_dir / "meta_oof_short_strategy_mr_tbm_clf.parquet")

    loaded = load_meta_oof_predictions(str(data_root), run_id)

    assert sorted(loaded) == ["strategy"]
    frame = loaded["strategy"]
    assert frame["oof_pred"].tolist() == [0.95, 0.20]
    assert frame[ROUTE_SCORE_SOURCE_COL].tolist() == ["mr", "general"]
