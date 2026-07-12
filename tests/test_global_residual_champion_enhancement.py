from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from scripts.run_global_residual_champion_enhancement import (
    _daily_signed_autocorrelation,
    _daily_top10_persistence_targets,
    _daily_top10_surprise_target,
    _final_feature_union,
    _fit_encoder_state_features,
    _latent_state_partitions,
    _local_context_blocks,
    _localize_partition_outputs,
    _materialize_local_signature_predictions,
    _merge_state_features,
    _partition_state_feature_sets,
    _passes_incremental_guard,
    _prune_generated_state_blocks,
    _purged_fit_boundaries,
    _score_local_residual_correction,
    _split_signed_surprise_targets,
    _state_blocks,
)


def test_partition_output_localization_removes_cross_archetype_names() -> None:
    prefix = "encoder_hybrid_mlp__"
    token = "long_mixed"
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "side_name": ["long", "long"],
            "archetype_policy_key": ["mixed", "mixed"],
            f"{prefix}global_state_pred_signature_arch__{token}_negative_ev": [
                0.1,
                0.2,
            ],
            f"{prefix}global_state_expected_signature_arch__{token}_negative_ev": [
                0.3,
                0.4,
            ],
        }
    )
    localized = _localize_partition_outputs(
        frame,
        encoder_kind="hybrid_mlp",
        token=token,
    )
    assert f"{prefix}local_arch_signature_pred_negative_ev" in localized
    assert f"{prefix}local_arch_signature_expected_negative_ev" in localized
    assert not any("arch__" in name for name in localized.columns)


def test_state_materialization_prunes_unused_output_families() -> None:
    prefix = "encoder_supervised_mlp__"
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "side_name": ["long", "long"],
            "archetype_policy_key": ["mixed", "mixed"],
            f"{prefix}global_state_latent_0": [0.1, 0.2],
            f"{prefix}global_state_posterior_0": [0.8, 0.7],
            f"{prefix}global_state_posterior_0_delta_1h": [0.0, -0.1],
            f"{prefix}global_state_entropy": [0.2, 0.3],
        }
    )
    pruned = _prune_generated_state_blocks(
        frame,
        encoder_kind="supervised_mlp",
        required_blocks=["B3_static_state_posteriors"],
    )
    assert f"{prefix}global_state_posterior_0" in pruned
    assert f"{prefix}global_state_latent_0" not in pruned
    assert f"{prefix}global_state_posterior_0_delta_1h" not in pruned
    assert f"{prefix}global_state_entropy" not in pruned


def test_merge_routes_state_features_by_side_and_archetype() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    data = pd.DataFrame(
        {
            "__ts__": [timestamp, timestamp],
            "side_name": ["long", "long"],
            "archetype_policy_key": ["mixed", "breakout"],
        }
    )
    states = data.copy()
    states["local_state"] = [1.0, -1.0]
    merged = _merge_state_features(data, states)
    assert merged["local_state"].tolist() == [1.0, -1.0]


def test_latent_partition_discovery_uses_pre_cutoff_archetypes() -> None:
    cutoff = pd.Timestamp("2026-04-01", tz="UTC")
    data = pd.DataFrame(
        {
            "__ts__": [cutoff - pd.Timedelta(days=1), cutoff + pd.Timedelta(days=1)],
            "side_name": ["long", "long"],
            "archetype_policy_key": ["mixed", "future_only"],
        }
    )
    states = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-03-01", periods=40, freq="D", tz="UTC"),
            "side_name": ["long"] * 40,
            "target_signature_arch__long_mixed_signed_surprise": np.arange(40),
        }
    )
    partitions = _latent_state_partitions(data, states, fit_end=cutoff)
    assert [item["token"] for item in partitions] == ["long_mixed"]


def test_encoder_gmm_bundles_are_fit_per_side_archetype(tmp_path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=560, freq="h", tz="UTC")
    phase = np.sin(np.arange(len(timestamps), dtype=np.float32) / 17.0)
    states = pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": ["long"] * len(timestamps),
            "market_feature": phase,
            "market_feature_2": np.cos(np.arange(len(timestamps)) / 13.0),
            "target_signature_arch__long_mixed_signed_surprise": phase,
            "target_signature_arch__long_mixed_mean_ev": phase * 0.01,
            "target_signature_arch__long_breakout_signed_surprise": -phase,
            "target_signature_arch__long_breakout_mean_ev": -phase * 0.01,
        }
    )
    partitions = [
        {
            "token": "long_mixed",
            "side_name": "long",
            "archetype_policy_key": "mixed",
        },
        {
            "token": "long_breakout",
            "side_name": "long",
            "archetype_policy_key": "breakout",
        },
    ]
    generated, manifest = _fit_encoder_state_features(
        states,
        ["market_feature", "market_feature_2"],
        partitions,
        encoder_kind="hybrid_mlp",
        cutoff=pd.Timestamp("2025-01-23", tz="UTC"),
        fit_end=pd.Timestamp("2025-01-23", tz="UTC"),
        evaluation_end=pd.Timestamp("2025-01-24 08:00", tz="UTC"),
        output_dir=tmp_path,
        state_cache_dir=tmp_path,
        latent_dim=2,
        epochs=2,
        components=(2,),
        covariance_types=("diag",),
        reg_covars=(1e-3,),
        gmm_n_init=1,
        seed=17,
        reuse_existing_state=False,
    )
    assert manifest["fit_granularity"] == "side_x_archetype"
    assert set(manifest["partitions"]) == {"long_mixed", "long_breakout"}
    assert (tmp_path / "states/hybrid_mlp__long_mixed.joblib").exists()
    assert (tmp_path / "states/hybrid_mlp__long_breakout.joblib").exists()
    mixed_bundle = joblib.load(tmp_path / "states/hybrid_mlp__long_mixed.joblib")
    breakout_bundle = joblib.load(tmp_path / "states/hybrid_mlp__long_breakout.joblib")
    assert mixed_bundle["encoder"].manifest()["partition"]["token"] == "long_mixed"
    assert mixed_bundle["gmm"].manifest()["partition"]["token"] == "long_mixed"
    assert (
        breakout_bundle["encoder"].manifest()["partition"]["token"] == "long_breakout"
    )
    assert breakout_bundle["gmm"].manifest()["partition"]["token"] == "long_breakout"
    assert mixed_bundle["encoder"] is not breakout_bundle["encoder"]
    assert mixed_bundle["gmm"] is not breakout_bundle["gmm"]
    counts = generated.groupby("archetype_policy_key").size().to_dict()
    assert counts == {"breakout": 560, "mixed": 560}
    assert "encoder_hybrid_mlp__local_arch_signature_pred_signed_surprise" in generated
    posterior = generated.filter(
        regex=r"^encoder_hybrid_mlp__global_state_posterior_[0-9]+$"
    )
    assert not posterior.isna().any().any()
    assert np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5)
    cache_path = tmp_path / "states/hybrid_mlp__long_mixed.joblib"
    cache_bytes = cache_path.read_bytes()
    reuse_root = tmp_path / "reused"
    _, reused_manifest = _fit_encoder_state_features(
        states,
        ["market_feature", "market_feature_2"],
        partitions,
        encoder_kind="hybrid_mlp",
        cutoff=pd.Timestamp("2025-01-23", tz="UTC"),
        fit_end=pd.Timestamp("2025-01-23", tz="UTC"),
        evaluation_end=pd.Timestamp("2025-01-24 08:00", tz="UTC"),
        output_dir=reuse_root,
        state_cache_dir=tmp_path,
        latent_dim=2,
        epochs=2,
        components=(2,),
        covariance_types=("diag",),
        reg_covars=(1e-3,),
        gmm_n_init=1,
        seed=999,
        reuse_existing_state=True,
    )
    assert all(
        partition["cache_reused"]
        for partition in reused_manifest["partitions"].values()
    )
    assert cache_path.read_bytes() == cache_bytes
    assert (reuse_root / "states/hybrid_mlp__long_mixed.joblib").exists()


def test_partition_state_feature_selection_is_local_and_frozen(tmp_path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    first = np.sin(np.arange(len(timestamps), dtype=np.float32) / 11.0)
    second = np.cos(np.arange(len(timestamps), dtype=np.float32) / 7.0)
    states = pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": "long",
            "feature_for_mixed": first,
            "feature_for_breakout": second,
        }
    )
    for token, signal in (("long_mixed", first), ("long_breakout", second)):
        prefix = f"target_signature_arch__{token}_"
        states[f"{prefix}signed_surprise"] = signal
        states[f"{prefix}positive_surprise"] = np.maximum(signal, 0.0)
        states[f"{prefix}negative_surprise"] = np.maximum(-signal, 0.0)
        states[f"{prefix}mean_ev"] = signal * 0.01
        states[f"{prefix}negative_ev"] = np.maximum(-signal, 0.0)
    partitions = [
        {
            "token": "long_mixed",
            "side_name": "long",
            "archetype_policy_key": "mixed",
        },
        {
            "token": "long_breakout",
            "side_name": "long",
            "archetype_policy_key": "breakout",
        },
    ]
    selected, manifest = _partition_state_feature_sets(
        states,
        ["feature_for_mixed", "feature_for_breakout"],
        partitions,
        fit_end=pd.Timestamp("2025-01-11", tz="UTC"),
        max_features=1,
        output_dir=tmp_path,
    )
    assert selected["long_mixed"] == ["feature_for_mixed"]
    assert selected["long_breakout"] == ["feature_for_breakout"]
    assert manifest["fit_end_exclusive"] == "2025-01-11 00:00:00+00:00"
    assert (tmp_path / "partition_state_feature_relevance.csv").exists()


def test_state_blocks_keep_encoder_outputs_separate() -> None:
    states = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "side_name": ["long"] * 3,
            "universe__median__oi_drawdown_from_peak_24h": [0.0, -0.1, -0.2],
            "target_signature_global_negative_ev": [0.0, 0.1, 0.2],
        }
    )
    prefix = "encoder_hybrid_mlp__"
    generated = pd.DataFrame(
        {
            "__ts__": states["__ts__"],
            "side_name": states["side_name"],
            f"{prefix}global_state_latent_0": [0.1, 0.2, 0.3],
            f"{prefix}global_state_pred_signature_global_negative_ev": [0.1, 0.2, 0.3],
            f"{prefix}global_state_pred_signature_arch__long_mixed_negative_ev": [
                0.2,
                0.3,
                0.4,
            ],
            f"{prefix}global_state_posterior_0": [0.8, 0.7, 0.6],
            f"{prefix}global_state_entropy": [0.2, 0.3, 0.4],
            f"{prefix}global_state_posterior_0_delta_1": [0.0, -0.1, -0.1],
            f"{prefix}global_state_latent_speed": [0.0, 0.2, 0.1],
            f"{prefix}global_state_transition_entropy": [0.1, 0.2, 0.3],
            f"{prefix}global_state_expected_signature_global_negative_ev": [
                0.1,
                0.2,
                0.3,
            ],
        }
    )
    blocks = _state_blocks(states, generated, "hybrid_mlp")
    assert blocks["B1_lifecycle_market"] == [
        "universe__median__oi_drawdown_from_peak_24h"
    ]
    assert f"{prefix}global_state_latent_0" in blocks["B2_encoder_bottleneck_signature"]
    assert (
        f"{prefix}global_state_pred_signature_arch__long_mixed_negative_ev"
        not in blocks["B2_encoder_bottleneck_signature"]
    )
    assert f"{prefix}global_state_posterior_0" in blocks["B3_static_state_posteriors"]
    assert f"{prefix}global_state_entropy" in blocks["B4_state_uncertainty"]
    assert f"{prefix}global_state_posterior_0_delta_1" in blocks["B5_state_transitions"]
    assert (
        f"{prefix}global_state_latent_speed"
        not in blocks["B2_encoder_bottleneck_signature"]
    )
    assert (
        f"{prefix}global_state_transition_entropy" not in blocks["B4_state_uncertainty"]
    )
    assert f"{prefix}global_state_latent_speed" in blocks["B5_state_transitions"]
    assert f"{prefix}global_state_transition_entropy" in blocks["B5_state_transitions"]
    assert "B6_state_side_archetype_priors" not in blocks


def test_final_feature_union_always_keeps_archetype_context() -> None:
    assert _final_feature_union(
        ["base_score"],
        ["local_arch_signature", "posterior_prior"],
        [],
        {"optional": ["latent_0"]},
    ) == ["base_score", "local_arch_signature", "posterior_prior"]


def test_local_context_outputs_are_separate_optional_blocks() -> None:
    prefix = "encoder_hybrid_mlp__"
    blocks = _local_context_blocks(
        [
            f"{prefix}local_arch_signature_pred_mean_ev",
            f"{prefix}local_state_prior_ev",
        ],
        "hybrid_mlp",
    )
    assert blocks == {
        "B0_local_signature_heads": [f"{prefix}local_arch_signature_pred_mean_ev"],
        "B0_local_state_priors": [f"{prefix}local_state_prior_ev"],
    }
    assert _final_feature_union(
        ["base_score"],
        [],
        ["B0_local_state_priors"],
        blocks,
    ) == ["base_score", f"{prefix}local_state_prior_ev"]


def test_local_residual_correction_is_scaled_without_future_inputs() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=2, freq="h", tz="UTC"),
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
            "archetype_policy_key": ["mixed", "breakout"],
            "score_meta_base_soft_label": [0.2, 0.8],
        }
    )
    scored = _score_local_residual_correction(
        frame,
        correction=[0.2, -0.4],
        base_probability=[0.4, 0.7],
        scale=0.5,
        arm="test",
    )
    assert np.allclose(scored["score_alternative"], [0.5, 0.5])
    assert np.allclose(scored["hit_prob_alternative"], [0.5, 0.5])
    assert np.allclose(scored["hit_prob_current_reference"], [0.4, 0.7])
    calibrated = _score_local_residual_correction(
        frame,
        correction=[0.2, -0.4],
        base_probability=[0.4, 0.7],
        scale=0.5,
        arm="test_calibration",
        rank_mode="calibration_only",
    )
    assert np.allclose(calibrated["score_alternative"], [0.2, 0.8])
    assert np.allclose(calibrated["hit_prob_alternative"], [0.5, 0.5])
    asymmetric = _score_local_residual_correction(
        frame,
        correction=[0.1, -0.1],
        base_probability=[0.4, 0.7],
        scale=1.0,
        arm="test_asymmetric",
        rank_mode="calibration_only",
        favorable_correction=[0.2, 0.1],
        adverse_correction=[0.1, 0.2],
        favorable_scale=1.0,
        adverse_scale=2.0,
    )
    assert np.allclose(asymmetric["hit_prob_alternative"], [0.4, 0.4])


def test_daily_top10_surprise_target_is_local_and_support_shrunk() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")] * 10,
            "side_name": ["long"] * 10,
            "archetype_policy_key": ["mixed"] * 10,
        }
    )
    score = pd.Series(np.arange(10, dtype=np.float32))
    residual = np.zeros(10, dtype=np.float32)
    residual[-1] = 0.6
    target = _daily_top10_surprise_target(
        frame,
        score,
        residual,
        pd.Series(True, index=frame.index),
        support_shrinkage=1.0,
    )
    assert np.allclose(target, 0.2)


def test_signed_surprise_targets_preserve_favorable_and_adverse_magnitudes() -> None:
    target = pd.Series([-0.4, 0.0, 0.3, np.nan], dtype=np.float32)
    favorable, adverse = _split_signed_surprise_targets(target)
    assert np.allclose(favorable.iloc[:3], [0.0, 0.0, 0.3])
    assert np.allclose(adverse.iloc[:3], [0.4, 0.0, 0.0])
    assert np.allclose((favorable - adverse).iloc[:3], target.iloc[:3])
    assert np.isnan(favorable.iloc[3])
    assert np.isnan(adverse.iloc[3])


def test_daily_persistence_target_uses_prior_week_not_same_day() -> None:
    timestamps = np.repeat(
        pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC"), 10
    )
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": "long",
            "archetype_policy_key": "mixed",
        }
    )
    score = pd.Series(np.tile(np.arange(10, dtype=np.float32), 8))
    residual = np.zeros(len(frame), dtype=np.float32)
    residual[9::10] = 0.4
    favorable, adverse = _daily_top10_persistence_targets(
        frame,
        score,
        residual,
        pd.Series(True, index=frame.index),
        support_shrinkage=0.0,
    )
    assert favorable.iloc[:30].isna().all()
    assert np.allclose(favorable.iloc[30:40], 0.04)
    assert np.allclose(adverse.iloc[30:40], 0.0)


def test_purged_fit_boundaries_protect_daily_signature_targets() -> None:
    cutoff = pd.Timestamp("2026-04-01", tz="UTC")
    train_end, state_end = _purged_fit_boundaries(cutoff, 12.0)
    assert train_end == pd.Timestamp("2026-03-31 12:00", tz="UTC")
    assert state_end == pd.Timestamp("2026-03-31", tz="UTC")


def test_local_signature_predictions_are_selected_by_side_and_archetype() -> None:
    prefix = "encoder_hybrid_mlp__"
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "archetype_policy_key": ["mixed", "breakout"],
            f"{prefix}global_state_pred_signature_arch__long_mixed_negative_persistence_prev7d": [
                0.3,
                9.0,
            ],
            f"{prefix}global_state_pred_signature_arch__short_breakout_negative_persistence_prev7d": [
                8.0,
                0.7,
            ],
        }
    )
    local = _materialize_local_signature_predictions(frame, "hybrid_mlp")
    name = f"{prefix}local_arch_signature_negative_persistence_prev7d"
    assert np.allclose(local[name].to_numpy(), [0.3, 0.7])


def test_incremental_guard_requires_ev_and_tail_stability() -> None:
    passing = {
        "objective": 0.1,
        "top10_ev_delta": 0.001,
        "autocorrelation_guard_pass": True,
        "worst_week_ev_delta": -0.0005,
        "worst_month_ev_delta": 0.0,
    }
    assert _passes_incremental_guard(passing, 0.0)
    assert not _passes_incremental_guard({**passing, "top10_ev_delta": -0.0001}, 0.0)
    assert _passes_incremental_guard(
        {**passing, "top10_ev_delta": 0.0, "rank_mode": "calibration_only"},
        0.0,
    )
    assert not _passes_incremental_guard(
        {**passing, "autocorrelation_guard_pass": False}, 0.0
    )
    assert not _passes_incremental_guard(
        {**passing, "worst_week_ev_delta": -0.002}, 0.0
    )


def test_ev_first_incremental_guard_prioritizes_ev_with_persistence_cap() -> None:
    passing = {
        "revision_priority": "ev_first",
        "objective": 0.1,
        "top10_ev_delta": 0.0004,
        "worst_week_ev_delta": 0.0008,
        "worst_month_ev_delta": 0.0001,
        "mean_abs_signed_component_ac1": 0.51,
        "baseline_mean_abs_signed_component_ac1": 0.26,
        "mean_abs_side_archetype_signed_component_ac1": 0.22,
        "baseline_mean_abs_side_archetype_signed_component_ac1": 0.17,
        "worst_supported_side_archetype_ev_delta": -0.0002,
    }
    assert _passes_incremental_guard(passing, 0.0)
    assert not _passes_incremental_guard(
        {**passing, "mean_abs_signed_component_ac1": 0.61}, 0.0
    )
    assert not _passes_incremental_guard(
        {**passing, "mean_abs_side_archetype_signed_component_ac1": 0.28}, 0.0
    )


def test_signed_autocorrelation_keeps_positive_and_negative_components() -> None:
    rows: list[dict[str, object]] = []
    for day in range(12):
        timestamp = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(days=day)
        for rank in range(20):
            clean = float((day + rank) % 3 != 0)
            rows.append(
                {
                    "__ts__": timestamp,
                    "score": rank / 20.0,
                    "prob": 0.65,
                    "clean_exec": clean,
                }
            )
    metrics = _daily_signed_autocorrelation(pd.DataFrame(rows), "score", "prob")
    assert set(metrics) == {"signed_ac1", "positive_ac1", "negative_ac1"}
    assert np.isfinite(metrics["signed_ac1"])
