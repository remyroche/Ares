from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from extreme_price_movements.alternative_latent_encoders import (
    AlternativeLatentEncoder,
    EncoderConfig,
    LatentMatrixCache,
    side_conditioned_corruption,
)


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    rows = 72
    sides = np.asarray(["long", "short"] * (rows // 2), dtype=object)
    groups = np.asarray(["a", "b", "c"] * (rows // 3), dtype=object)
    base = rng.normal(size=(rows, 7)).astype(np.float32)
    base += (sides == "long").astype(np.float32).reshape(-1, 1) * 0.25
    base[:, 0] = base[:, 1] * 0.6 + rng.normal(scale=0.1, size=rows)
    return base.astype(np.float32), sides, groups


def _config(kind: str) -> EncoderConfig:
    return EncoderConfig(
        kind=kind,
        latent_dim=4,
        hidden_dim=16,
        residual_blocks=1,
        epochs=2,
        pretrain_epochs=1,
        batch_size=24,
        n_clusters=3,
        cluster_weight=0.1,
        random_state=73,
        device="cpu",
    )


def test_side_conditioned_corruption_never_crosses_side_or_group() -> None:
    x, sides, groups = _data()
    corrupted, mask, donors = side_conditioned_corruption(
        x, sides, groups=groups, corruption_rate=0.8, random_state=9
    )

    assert mask.any()
    for row, donor in enumerate(donors):
        if donor >= 0:
            assert sides[row] == sides[donor]
            assert groups[row] == groups[donor]
            np.testing.assert_allclose(corrupted[row, ~mask[row]], x[row, ~mask[row]])


def test_feature_group_augmentations_protect_side_and_respect_frozen_regimes() -> None:
    x, sides, groups = _data()
    regimes = np.asarray(["r0", "r1"] * (len(x) // 2), dtype=object)
    feature_groups = {"signals": [0, 1], "side": [2], "other": [3, 4, 5, 6]}

    donor_view, _mask, donors = side_conditioned_corruption(
        x,
        sides,
        groups=groups,
        donor_regime_labels=regimes,
        feature_group_indices=feature_groups,
        group_donor_replacement_rate=1.0,
        random_state=4,
    )
    for row, donor in enumerate(donors):
        if donor >= 0:
            assert sides[row] == sides[donor]
            assert groups[row] == groups[donor]
            assert regimes[row] == regimes[donor]
            np.testing.assert_allclose(donor_view[row, [0, 1]], x[donor, [0, 1]])
    np.testing.assert_allclose(donor_view[:, 2], x[:, 2])

    masked, mask, _ = side_conditioned_corruption(
        x,
        sides,
        feature_group_indices=feature_groups,
        element_mask_rate=1.0,
        whole_feature_group_mask_rate=1.0,
        random_state=5,
    )
    assert mask[:, [0, 1, 3, 4, 5, 6]].all()
    np.testing.assert_allclose(masked[:, [0, 1, 3, 4, 5, 6]], 0.0)
    np.testing.assert_allclose(masked[:, 2], x[:, 2])

    noisy, noise_mask, _ = side_conditioned_corruption(
        x,
        sides,
        feature_group_indices=feature_groups,
        additive_noise_std=0.2,
        random_state=6,
    )
    assert noise_mask[:, [0, 1, 3, 4, 5, 6]].all()
    assert not np.allclose(noisy[:, [0, 1]], x[:, [0, 1]])
    np.testing.assert_allclose(noisy[:, 2], x[:, 2])


@pytest.mark.parametrize("kind", ["masked", "scarf", "vicreg", "idec", "vade"])
def test_torch_adapters_return_float32_native_contract(kind: str) -> None:
    x, sides, groups = _data()
    encoder = AlternativeLatentEncoder(_config(kind)).fit(x, sides=sides, groups=groups)
    native = encoder.transform_native(x, sides=sides)

    assert native.latent.shape == (len(x), 4)
    assert native.latent.dtype == np.float32
    assert native.reconstruction is not None
    assert native.reconstruction.shape == x.shape
    assert native.reconstruction_error is not None
    assert native.reconstruction_error.shape == (len(x),)
    assert np.isfinite(native.latent).all()
    if kind in {"idec", "vade"}:
        assert native.cluster_probabilities is not None
        assert native.cluster_probabilities.shape == (len(x), 3)
        np.testing.assert_allclose(native.cluster_probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_masked_adapter_is_deterministic_serializable_and_cacheable(tmp_path) -> None:
    x, sides, groups = _data()
    first = AlternativeLatentEncoder(_config("masked")).fit(x, sides=sides, groups=groups)
    second = AlternativeLatentEncoder(_config("masked")).fit(x, sides=sides, groups=groups)
    native = first.transform_native(x, sides=sides)
    np.testing.assert_allclose(native.latent, second.transform(x, sides=sides), atol=1e-6)

    state = first.to_state()
    json.dumps(state, allow_nan=False)
    restored = AlternativeLatentEncoder.from_state(state, device="cpu")
    np.testing.assert_allclose(native.latent, restored.transform(x, sides=sides), atol=1e-6)

    cache = LatentMatrixCache(tmp_path / "latent_cache")
    cached = first.transform_cached(x, sides=sides, groups=groups, cache=cache)
    assert len(list((tmp_path / "latent_cache").glob("*.npz"))) == 1
    np.testing.assert_allclose(native.latent, cached.latent, atol=1e-6)
    np.testing.assert_allclose(native.latent, first.transform_cached(x, sides=sides, groups=groups, cache=cache).latent, atol=1e-6)


def test_clustering_controls_accept_incumbent_and_pretrained_mixtures() -> None:
    x, sides, groups = _data()
    feature_groups = {"signals": [0, 1, 3, 4], "side": [2], "other": [5, 6]}
    incumbent = {
        "gmm_means": np.zeros((3, 4), dtype=np.float32),
        "gmm_covariances": np.ones((3, 4), dtype=np.float32),
        "gmm_weights": np.full(3, 1.0 / 3.0, dtype=np.float32),
    }
    idec = AlternativeLatentEncoder(
        EncoderConfig(
            **{
                **_config("idec").__dict__,
                "initialization": "incumbent means",
                "target_update_frequency": 2,
                "student_t_df": 2.5,
                "reconstruction_objective": "feature_group_weighted",
            }
        )
    ).fit(
        x,
        sides=sides,
        groups=groups,
        feature_group_indices=feature_groups,
        initialization_state=incumbent,
    )
    assert idec.to_state()["config"]["target_update_frequency"] == 2
    assert idec.transform_native(x, sides=sides).cluster_probabilities is not None

    vade_config = EncoderConfig(
        **{
            **_config("vade").__dict__,
            "initialization": "pretrained_dae_gmm",
            "kl_weight": 0.2,
            "kl_warmup_fraction": 0.5,
            "min_effective_occupancy": 0.05,
            "reconstruction_objective": "feature_group_weighted",
        }
    )
    self_initialized = AlternativeLatentEncoder(vade_config).fit(
        x,
        sides=sides,
        feature_group_indices=feature_groups,
    )
    assert self_initialized.transform_native(x, sides=sides).cluster_probabilities is not None
    vade = AlternativeLatentEncoder(vade_config).fit(
        x,
        sides=sides,
        feature_group_indices=feature_groups,
        initialization_state=incumbent,
    )
    native = vade.transform_native(x, sides=sides)
    assert native.cluster_probabilities is not None
    assert vade.to_state()["protected_feature_indices"] == [2]

    pretrained_vade = AlternativeLatentEncoder(
        EncoderConfig(
            **{
                **vade_config.__dict__,
                "initialization": "pretrained_vae_gmm",
                "epochs": 1,
            }
        )
    ).fit(
        x,
        sides=sides,
        feature_group_indices=feature_groups,
        initialization_state=vade.to_state(),
    )
    assert pretrained_vade.transform_native(x, sides=sides).cluster_probabilities is not None


def test_idec_pretraining_cache_reuses_only_reconstruction_torso() -> None:
    x, sides, _groups = _data()
    config = _config("idec")
    feature_groups = {"signals": [0, 1, 3, 4], "side": [2], "other": [5, 6]}
    pretraining = AlternativeLatentEncoder(config).fit_idec_pretraining_state(
        x,
        sides=sides,
        feature_group_indices=feature_groups,
    )
    reused = AlternativeLatentEncoder(config).fit(
        x,
        sides=sides,
        feature_group_indices=feature_groups,
        pretraining_state=pretraining,
    )
    native = reused.transform_native(x, sides=sides)

    assert reused.training_report["reused_idec_pretraining"] is True
    assert native.cluster_probabilities is not None
    np.testing.assert_allclose(native.cluster_probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_legacy_dae_adapter_uses_existing_framework() -> None:
    x, _sides, _groups = _data()
    x = np.vstack((x, x, x)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder = AlternativeLatentEncoder(
            EncoderConfig(kind="dae", latent_dim=8, epochs=1, dae_max_train_rows=len(x))
        ).fit(x)
    native = encoder.transform_native(x)

    assert native.latent.shape == (len(x), 8)
    assert native.latent.dtype == np.float32
    assert native.reconstruction_error is not None
