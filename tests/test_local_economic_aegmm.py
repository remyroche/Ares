from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.local_economic_aegmm as local_module
from extreme_price_movements.features_gmm_ae import ae_gmm_feature_columns
from extreme_price_movements.local_economic_aegmm import (
    EconomicAEGMMBlock,
    HierarchicalEconomicAEGMM,
    HierarchicalEconomicAEGMMConfig,
    LocalEconomicAEGMM,
    LocalEconomicAEGMMConfig,
    LocalEconomicAEGMMModelBundle,
    default_base_economic_aegmm_blocks,
    default_meta_economic_aegmm_blocks,
    local_economic_aegmm_feature_names,
)


class _MeanModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.mean(axis=1).to_numpy(dtype=np.float32)


def _frame(rows_per_group: int = 80) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    specifications = (
        ("long", "long_breakout", 0.75, 0.012),
        ("long", "long_mixed", 0.35, -0.004),
        ("short", "short_breakout_precision", 0.68, 0.009),
        ("short", "short_mixed", 0.25, -0.008),
    )
    for group_index, (side, archetype, clean_rate, ev_mean) in enumerate(
        specifications
    ):
        timestamp = pd.date_range(
            "2025-01-01", periods=rows_per_group, freq="15min", tz="UTC"
        )
        phase = np.linspace(-2.0, 2.0, rows_per_group, dtype=np.float32)
        clean = (np.arange(rows_per_group) % 100 < int(clean_rate * 100)).astype(
            np.float32
        )
        parts.append(
            pd.DataFrame(
                {
                    "__ts__": timestamp,
                    "__symbol__": f"S{group_index}",
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "score_base": np.clip(0.55 + 0.08 * phase, 0.0, 1.0),
                    "score_meta_base_soft_label": np.clip(
                        0.55 + 0.08 * phase, 0.0, 1.0
                    ),
                    "market_a": phase + np.float32(group_index),
                    "market_b": np.sin(phase * np.float32(1.7)),
                    "geometry_a": np.cos(phase) + np.float32(group_index * 0.2),
                    "geometry_b": phase * phase,
                    "clean_exec": clean,
                    "dirty_positive": ((clean < 0.5) & (phase > 0.0)).astype(
                        np.float32
                    ),
                    "full_path_bad_mae_1r": (clean < 0.5).astype(np.float32),
                    "timeout": (
                        (np.arange(rows_per_group) + group_index) % 13 == 0
                    ).astype(np.float32),
                    "ev_after_1pct": ev_mean + 0.003 * phase,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def fake_aegmm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_fit(x, *, economic_targets=None, **kwargs):
        calls.append(
            {
                "rows": len(x),
                "columns": list(x.columns),
                "targets": sorted((economic_targets or {}).keys()),
                "cluster_candidates": tuple(kwargs.get("cluster_candidates", ())),
                "smooth": tuple(kwargs.get("smooth_lambda_candidates", ())),
                "temporal_stability_hpo": kwargs.get("temporal_stability_hpo"),
                "component_complexity_penalty": kwargs.get(
                    "component_complexity_penalty"
                ),
                "max_train_rows": kwargs.get("max_train_rows"),
                "gmm_max_train_rows": kwargs.get("gmm_max_train_rows"),
                "final_refit_all_rows": kwargs.get("final_refit_all_rows"),
            }
        )
        return {
            "enabled": True,
            "feature_columns": list(x.columns),
            "gmm_n_components": 3,
            "sample_manifest": {
                "ae": {"segments": ["beginning", "middle", "end"]},
                "gmm": {"segments": ["beginning", "middle", "end"]},
            },
        }

    def fake_transform(x, state, *, index=None, prefix=""):
        output = pd.DataFrame(
            0.0,
            index=x.index if index is None else index,
            columns=ae_gmm_feature_columns(prefix),
            dtype=np.float32,
        )
        first = (
            pd.to_numeric(x.iloc[:, 0], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        p0 = 1.0 / (1.0 + np.exp(-first))
        p1 = (1.0 - p0) * np.float32(0.65)
        p2 = np.maximum(1.0 - p0 - p1, 0.0)
        posterior = np.column_stack((p0, p1, p2)).astype(np.float32)
        posterior /= posterior.sum(axis=1, keepdims=True)
        for cluster in range(3):
            output[f"{prefix}gmm_cluster_posterior_{cluster}"] = posterior[:, cluster]
            output[f"{prefix}gmm_prob_{cluster}"] = posterior[:, cluster]
        output[f"{prefix}gmm_cluster_id"] = np.argmax(posterior, axis=1).astype(
            np.float32
        )
        output[f"{prefix}gmm_posterior_max"] = posterior.max(axis=1)
        output[f"{prefix}gmm_posterior_margin"] = (
            np.sort(posterior, axis=1)[:, -1] - np.sort(posterior, axis=1)[:, -2]
        )
        output[f"{prefix}gmm_entropy"] = -np.sum(
            posterior * np.log(np.maximum(posterior, 1e-7)), axis=1
        )
        output[f"{prefix}cluster_entropy_norm"] = output[
            f"{prefix}gmm_entropy"
        ] / np.log(3.0)
        output[f"{prefix}mahalanobis_distance"] = np.abs(first)
        output[f"{prefix}expected_mahalanobis"] = np.abs(first) + 0.1
        output[f"{prefix}AE_reconstruction_error"] = np.square(first) * 0.01
        output[f"{prefix}dae_reconstruction_error_zscore"] = first * 0.1
        for latent in range(16):
            output[f"{prefix}dae_b16_{latent:02d}"] = first * np.float32(
                (latent + 1) / 16.0
            )
        return output

    monkeypatch.setattr(local_module, "fit_ae_gmm_state", fake_fit)
    monkeypatch.setattr(local_module, "transform_ae_gmm_features", fake_transform)
    return calls


def _blocks() -> tuple[EconomicAEGMMBlock, ...]:
    return (
        EconomicAEGMMBlock("market_state", ("market_a", "market_b")),
        EconomicAEGMMBlock("cross_sectional_geometry", ("geometry_a", "geometry_b")),
        EconomicAEGMMBlock(
            "joint_market_geometry",
            ("market_a", "market_b", "geometry_a", "geometry_b"),
        ),
    )


def test_default_meta_blocks_include_individual_and_joint_spaces() -> None:
    blocks = default_meta_economic_aegmm_blocks()
    assert [block.name for block in blocks] == [
        "market_state",
        "cross_sectional_geometry",
        "joint_market_geometry",
    ]
    market = set(blocks[0].features)
    geometry = set(blocks[1].features)
    assert set(blocks[2].features) == market | geometry
    assert market
    assert geometry


def test_default_base_block_is_asset_local_directional_only() -> None:
    blocks = default_base_economic_aegmm_blocks()
    assert len(blocks) == 1
    assert blocks[0].name == "base_directional_state"
    assert blocks[0].timestamp_level is False
    assert "price_down_oi_down_1h_rz" in blocks[0].features
    assert "asset_flush_exhaustion_score" in blocks[0].features
    assert not any(name.startswith("mkt_") for name in blocks[0].features)
    assert not any(name.startswith("meta_xsgeom_") for name in blocks[0].features)


def test_local_states_are_fitted_per_side_archetype_and_block(fake_aegmm) -> None:
    frame = _frame()
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=50,
            min_fit_rows=20,
            ae_max_train_rows=30,
            gmm_max_train_rows=60,
        ),
        blocks=_blocks(),
    ).fit(frame)
    assert len(model.side_models) == 6
    assert len(model.local_models) == 12
    assert set(key[0] for key in model.local_models) == {
        "market_state",
        "cross_sectional_geometry",
        "joint_market_geometry",
    }
    assert all(call["cluster_candidates"] == (3, 4, 5, 6, 7) for call in fake_aegmm)
    assert all(call["smooth"] == (0.0,) for call in fake_aegmm)
    assert all("hit_surprise" in call["targets"] for call in fake_aegmm)
    assert all("negative_tail" in call["targets"] for call in fake_aegmm)
    assert all("time_bucket" not in call["targets"] for call in fake_aegmm)
    assert all(call["temporal_stability_hpo"] is True for call in fake_aegmm)
    assert all(call["component_complexity_penalty"] == 0.06 for call in fake_aegmm)


def test_local_state_semantics_use_global_candidate_rank_before_partitioning(
    fake_aegmm, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(rows_per_group=40)
    # Every local archetype has one row per timestamp. A local rank would make
    # each row rank 1.0, while the actual candidate book has four rows.
    scores = {
        "long_breakout": 0.20,
        "long_mixed": 0.95,
        "short_breakout_precision": 0.70,
        "short_mixed": 0.45,
    }
    for archetype, value in scores.items():
        mask = frame["archetype_policy_key"].eq(archetype)
        frame.loc[mask, "score_base"] = np.float32(value)
    # A low-ranked bad row must not become a top-10 bad-tail target merely
    # because its local side/archetype partition is tiny.
    low = frame["archetype_policy_key"].eq("long_breakout")
    frame.loc[low, "clean_exec"] = 0.0
    frame.loc[low, "ev_after_1pct"] = -0.01

    captured: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    original = local_module._reference_descriptors

    def capture(
        frame_part: pd.DataFrame, config: LocalEconomicAEGMMConfig
    ) -> pd.DataFrame:
        descriptors = original(frame_part, config)
        captured.append((frame_part.copy(), descriptors.copy()))
        return descriptors

    monkeypatch.setattr(local_module, "_reference_descriptors", capture)
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=20,
            min_fit_rows=20,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)

    local_rows = [
        (part, descriptor)
        for part, descriptor in captured
        if part["archetype_policy_key"].eq("long_breakout").all()
    ]
    assert local_rows
    local_part, descriptor = local_rows[0]
    assert local_part["__local_econ_aegmm_global_rank_pct__"].eq(0.25).all()
    assert descriptor["rank_pct"].eq(0.25).all()
    assert descriptor["negative_tail"].eq(0.0).all()
    assert "__local_econ_aegmm_global_rank_pct__" not in frame.columns
    assert "__local_econ_aegmm_global_rank_pct__" not in model.required_input_features()


def test_local_state_tail_descriptors_and_semantics_prioritize_global_top10() -> None:
    timestamp = pd.Timestamp("2026-02-01 00:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": [timestamp, timestamp],
            "side_name": ["long", "long"],
            "archetype_policy_key": ["long_mixed", "long_mixed"],
            "score_base": [0.95, 0.10],
            "clean_exec": [0.0, 1.0],
            "dirty_positive": [1.0, 0.0],
            "first_touch_bad_mae_1r": [1.0, 0.0],
            "timeout": [0.0, 0.0],
            "ev_after_1pct": [-0.012, 0.010],
        }
    )
    descriptors = local_module._reference_descriptors(frame, LocalEconomicAEGMMConfig())
    assert descriptors.loc[0, "top10_ev"] < 0.0
    assert np.isnan(descriptors.loc[1, "top10_ev"])
    assert descriptors.loc[0, "top10_acute_adverse"] == 1.0
    assert np.isnan(descriptors.loc[1, "top10_acute_adverse"])
    semantic = local_module._semantic_for_cluster(
        {
            "ev": 0.01,
            "clean_positive": 0.90,
            "bad_mae": 0.01,
            "top10_ev": -0.012,
            "top10_clean_positive": 0.0,
            "top10_bad_mae": 1.0,
            "top10_acute_adverse": 1.0,
        }
    )
    assert semantic == "acute_adverse_false_positive"
    positive = local_module._semantic_for_cluster(
        {
            "ev": 0.01,
            "clean_positive": 0.90,
            "negative_tail": 0.10,
            "top10_ev": 0.01,
            "top10_clean_positive": 0.90,
            "top10_clean_negative_ev": 0.30,
            "top10_negative_tail": 0.10,
            "top10_hit_surprise": 0.10,
        }
    )
    assert positive == "clean_high_confidence"


def test_asset_level_block_disables_row_order_temporal_stability(fake_aegmm) -> None:
    frame = _frame()
    LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=50,
            min_fit_rows=20,
        ),
        blocks=(
            EconomicAEGMMBlock(
                "base_directional_state",
                ("market_a", "market_b"),
                timestamp_level=False,
            ),
        ),
    ).fit(frame)
    assert fake_aegmm
    assert all(call["temporal_stability_hpo"] is False for call in fake_aegmm)


def test_full_train_fit_preserves_every_resolved_row_for_final_state(
    fake_aegmm,
) -> None:
    frame = _frame()
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=50,
            min_fit_rows=20,
            full_train_fit=True,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    assert fake_aegmm
    assert all(int(call["max_train_rows"]) > 0 for call in fake_aegmm)
    assert all(int(call["gmm_max_train_rows"]) > 0 for call in fake_aegmm)
    assert all(call["final_refit_all_rows"] is True for call in fake_aegmm)
    manifest = model.manifest()
    assert manifest["full_train_fit"] is True
    assert (
        "refit on all resolved rows before fit cutoff"
        in manifest["fit_sample_contract"]
    )


def test_oos_transform_is_frozen_outcome_free_and_emits_joint_state(fake_aegmm) -> None:
    frame = _frame()
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=50,
            min_fit_rows=20,
        ),
        blocks=_blocks(),
    ).fit(frame)
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    first = model.transform_oos(safe.iloc[:30])
    second = model.transform_oos(safe.iloc[:30])
    pd.testing.assert_frame_equal(first, second)
    expected = local_economic_aegmm_feature_names(
        ["market_state", "cross_sectional_geometry", "joint_market_geometry"]
    )
    assert list(first.columns) == expected
    assert first["local_econ_aegmm_joint_market_geometry_enabled"].eq(1.0).all()
    assert first["local_econ_aegmm_joint_market_geometry_local_model"].eq(1.0).all()
    assert (
        first.filter(like="joint_market_geometry_gmm_cluster_posterior_")
        .sum(axis=1)
        .round(6)
        .eq(1.0)
        .all()
    )
    assert not set(model.required_input_features()) & {
        "clean_exec",
        "ev_after_1pct",
        "full_path_bad_mae_1r",
    }


def test_older_frozen_state_prior_schema_remains_transformable(fake_aegmm) -> None:
    frame = _frame()
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=50,
            min_fit_rows=20,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    frozen = next(iter(model.local_models.values()))
    # Simulate a v2 joblib object saved before top-tail priors were added.
    frozen.prior_matrix = frozen.prior_matrix[:, :11]
    del frozen.prior_names
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    generated = model.transform_oos(safe.iloc[:10])
    assert generated["local_econ_aegmm_market_state_expected_ev"].notna().all()
    assert generated["local_econ_aegmm_market_state_expected_top10_ev"].eq(0.0).all()


def test_timestamp_level_dynamics_are_shared_within_timestamp_and_causal(
    fake_aegmm, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(rows_per_group=20)
    source = (
        frame.loc[frame["archetype_policy_key"].eq("long_breakout")].iloc[:4].copy()
    )
    duplicate = source.copy()
    duplicate["__symbol__"] = "DUP"
    duplicate["market_a"] += np.float32(100.0)
    duplicated = pd.concat([source, duplicate], ignore_index=True)
    # Deliberately scramble candidate rows. The state sequence must instead be
    # one sorted timestamp-level series shared by both candidates at each bar.
    duplicated = duplicated.iloc[[2, 4, 0, 6, 1, 5, 3, 7]].reset_index(drop=True)
    fit_frame = pd.concat([frame, duplicated], ignore_index=True)

    original_transform = local_module.transform_ae_gmm_features

    def sequence_transform(*args, **kwargs):
        output = original_transform(*args, **kwargs)
        sequence = np.arange(len(output), dtype=np.float32)
        output["cluster_speed"] = sequence
        output["cluster_acceleration"] = sequence * np.float32(2.0)
        return output

    monkeypatch.setattr(local_module, "transform_ae_gmm_features", sequence_transform)
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=10,
            min_local_rows=4,
            min_fit_rows=4,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(fit_frame)
    safe = duplicated.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    generated = model.transform_oos(safe)
    speed = "local_econ_aegmm_market_state_cluster_speed"
    accel = "local_econ_aegmm_market_state_cluster_acceleration"
    timestamp = pd.to_datetime(safe["__ts__"], utc=True)
    assert generated.groupby(timestamp)[speed].nunique().eq(1).all()
    assert generated.groupby(timestamp)[accel].nunique().eq(1).all()
    ordered = (
        pd.DataFrame({"timestamp": timestamp, "speed": generated[speed].to_numpy()})
        .groupby("timestamp", sort=True)["speed"]
        .first()
        .to_numpy()
    )
    np.testing.assert_array_equal(ordered, np.arange(len(ordered), dtype=np.float32))
    assert "__ts__" in model.required_input_features()
    with pytest.raises(ValueError, match="received outcomes"):
        model.transform_oos(frame.iloc[:5])


def test_low_support_archetype_uses_side_fallback(fake_aegmm) -> None:
    frame = _frame(rows_per_group=80)
    model = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=100,
            min_fit_rows=20,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    transformed = model.transform_oos(safe.iloc[:10])
    assert transformed["local_econ_aegmm_market_state_enabled"].eq(1.0).all()
    assert transformed["local_econ_aegmm_market_state_local_model"].eq(0.0).all()


def test_primary_model_bundle_applies_frozen_state_directly(fake_aegmm) -> None:
    frame = _frame()
    state = LocalEconomicAEGMM(
        config=LocalEconomicAEGMMConfig(
            min_side_rows=50,
            min_local_rows=50,
            min_fit_rows=20,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    generated_name = "local_econ_aegmm_market_state_expected_ev"
    bundle = LocalEconomicAEGMMModelBundle(
        model=_MeanModel(),
        local_aegmm=state,
        selected_features=["market_a", generated_name],
        raw_selected_features=["market_a", generated_name],
        feature_medians={"market_a": 0.0, generated_name: 0.0},
        ood_state={},
    )
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    prediction = bundle.predict(safe.iloc[:20])
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()
    assert generated_name not in bundle.required_input_features()
    assert "market_a" in bundle.required_input_features()
    with pytest.raises(ValueError, match="input parity failure"):
        bundle.predict(safe.iloc[:5].drop(columns=["market_b"]))


def test_hierarchical_states_share_geometry_and_keep_local_economic_response(
    fake_aegmm,
) -> None:
    frame = _frame(rows_per_group=80)
    model = HierarchicalEconomicAEGMM(
        config=HierarchicalEconomicAEGMMConfig(
            min_fit_rows=20,
            min_response_side_rows=50,
            min_response_local_rows=50,
            min_response_side_tail_rows=0,
            min_response_local_tail_rows=0,
        ),
        blocks=_blocks(),
    ).fit(frame)

    # One observable market/cross-sectional geometry is fitted per block,
    # rather than independently for each of the four local partitions.
    assert len(model.shared_models) == 3
    assert len(fake_aegmm) == 3
    assert all(call["cluster_candidates"] == (3, 4, 5) for call in fake_aegmm)
    assert (
        sum(len(response.local_matrices) for response in model.responses.values()) == 12
    )

    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    generated = model.transform_oos(safe)
    posterior_cols = [
        name
        for name in generated
        if name.startswith("local_econ_aegmm_market_state_gmm_cluster_posterior_")
    ]
    timestamp = pd.to_datetime(safe["__ts__"], utc=True)
    # The state is shared across the candidate book at a timestamp. The
    # response is local, so expected EV need not be shared.
    assert generated.groupby(timestamp)[posterior_cols].nunique().eq(1).all().all()
    expected_ev = "local_econ_aegmm_market_state_expected_ev"
    long_breakout = generated.loc[
        safe["archetype_policy_key"].eq("long_breakout"), expected_ev
    ]
    long_mixed = generated.loc[
        safe["archetype_policy_key"].eq("long_mixed"), expected_ev
    ]
    assert float(long_breakout.mean()) > float(long_mixed.mean())
    assert generated["local_econ_aegmm_market_state_local_model"].eq(1.0).all()
    assert not set(model.required_input_features()) & {
        "clean_exec",
        "ev_after_1pct",
        "full_path_bad_mae_1r",
    }
    with pytest.raises(ValueError, match="received outcomes"):
        model.transform_oos(frame.iloc[:10])


def test_hierarchical_states_fall_back_to_side_response_when_local_tail_is_thin(
    fake_aegmm,
) -> None:
    frame = _frame(rows_per_group=40)
    model = HierarchicalEconomicAEGMM(
        config=HierarchicalEconomicAEGMMConfig(
            min_fit_rows=20,
            min_response_side_rows=40,
            min_response_local_rows=100,
            min_response_side_tail_rows=1,
            min_response_local_tail_rows=1,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    generated = model.transform_oos(safe.iloc[:20])
    assert generated["local_econ_aegmm_market_state_enabled"].eq(1.0).all()
    assert generated["local_econ_aegmm_market_state_local_model"].eq(0.0).all()


def test_hierarchical_oos_history_preserves_dynamic_state_sequence(
    fake_aegmm, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame(rows_per_group=40)
    original_transform = local_module.transform_ae_gmm_features

    def sequence_transform(*args, **kwargs):
        output = original_transform(*args, **kwargs)
        output["cluster_speed"] = np.arange(len(output), dtype=np.float32)
        return output

    monkeypatch.setattr(local_module, "transform_ae_gmm_features", sequence_transform)
    model = HierarchicalEconomicAEGMM(
        config=HierarchicalEconomicAEGMMConfig(
            min_fit_rows=20,
            min_response_side_rows=40,
            min_response_local_rows=40,
            min_response_side_tail_rows=0,
            min_response_local_tail_rows=0,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    # Use a complete candidate book for each timestamp, split in time.
    split_time = pd.to_datetime(safe["__ts__"], utc=True).sort_values().unique()[20]
    history = safe.loc[pd.to_datetime(safe["__ts__"], utc=True).lt(split_time)]
    target = safe.loc[pd.to_datetime(safe["__ts__"], utc=True).ge(split_time)]
    full = model.transform_oos(safe)
    with_history = model.transform_oos_with_history(history, target)
    columns = [
        "local_econ_aegmm_market_state_cluster_speed",
        "local_econ_aegmm_market_state_gmm_cluster_id",
    ]
    expected = full.loc[target.index, columns]
    pd.testing.assert_frame_equal(with_history.loc[:, columns], expected)


def test_primary_bundle_accepts_frozen_hierarchical_state(fake_aegmm) -> None:
    frame = _frame(rows_per_group=60)
    state = HierarchicalEconomicAEGMM(
        config=HierarchicalEconomicAEGMMConfig(
            min_fit_rows=20,
            min_response_side_rows=40,
            min_response_local_rows=40,
            min_response_side_tail_rows=0,
            min_response_local_tail_rows=0,
        ),
        blocks=(EconomicAEGMMBlock("market_state", ("market_a", "market_b")),),
    ).fit(frame)
    generated_name = "local_econ_aegmm_market_state_expected_ev"
    bundle = LocalEconomicAEGMMModelBundle(
        model=_MeanModel(),
        local_aegmm=state,
        selected_features=["market_a", generated_name],
        raw_selected_features=["market_a", generated_name],
        feature_medians={"market_a": 0.0, generated_name: 0.0},
        ood_state={},
    )
    safe = frame.drop(
        columns=[
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "ev_after_1pct",
        ]
    )
    prediction = bundle.predict(safe.iloc[:20])
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()
    assert generated_name not in bundle.required_input_features()
