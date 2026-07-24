"""Reusable, outcome-free latent encoder adapters for AE/GMM experiments.

The adapters intentionally own preprocessing and fitted parameters.  Callers
must fit them only on the authorised cycle-reference rows, then reuse the
serialized state for every later transform.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

try:  # Kept optional so importing the pipeline never requires torch.
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover - exercised on minimal installations.
    torch = None
    nn = None
    F = None

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - sklearn is optional for IDEC only.
    KMeans = None
    GaussianMixture = None

from extreme_price_movements.features_denoising_ae import (
    fit_denoising_autoencoder_state,
    transform_denoising_autoencoder_features,
)

TORCH_ENCODERS = frozenset({"masked", "scarf", "vicreg", "idec", "vade"})
ENCODER_KINDS = frozenset({"dae", *TORCH_ENCODERS})


@dataclass(frozen=True)
class EncoderConfig:
    """Small, practical training configuration for a latent encoder."""

    kind: str = "masked"
    latent_dim: int = 16
    hidden_dim: int = 64
    residual_blocks: int = 2
    epochs: int = 20
    pretrain_epochs: int = 10
    pretraining_fraction: float | None = None
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    # ``corruption_rate`` is retained as the legacy element-wise donor rate.
    corruption_rate: float = 0.15
    element_mask_rate: float = 0.0
    whole_feature_group_mask_rate: float = 0.0
    additive_noise_std: float = 0.0
    group_donor_replacement_rate: float = 0.0
    ssl_objective: str = "masked_reconstruction"
    ssl_view_pair: str = "weak_strong"
    side_feature_group: str = "side"
    side_feature_indices: tuple[int, ...] = ()
    reconstruction_weight: float = 1.0
    reconstruction_objective: str = "current"
    contrastive_temperature: float = 0.2
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_gamma: float = 1.0
    n_clusters: int = 6
    target_update_frequency: int = 1
    student_t_df: float = 1.0
    initialization: str = "auto"
    cluster_weight: float = 0.5
    kl_weight: float | None = None
    kl_warmup_fraction: float = 0.0
    min_effective_occupancy: float = 0.0
    random_state: int = 42
    device: str = "auto"
    dae_max_train_rows: int = 5000


@dataclass
class NativeLatentOutput:
    """Native encoder output, suitable for a later GMM or diagnostics layer."""

    latent: np.ndarray
    reconstruction: np.ndarray | None = None
    reconstruction_error: np.ndarray | None = None
    cluster_probabilities: np.ndarray | None = None
    mean: np.ndarray | None = None
    logvar: np.ndarray | None = None

    def copy(self) -> "NativeLatentOutput":
        def clone(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else np.asarray(value, dtype=np.float32).copy()

        return NativeLatentOutput(
            latent=clone(self.latent),
            reconstruction=clone(self.reconstruction),
            reconstruction_error=clone(self.reconstruction_error),
            cluster_probabilities=clone(self.cluster_probabilities),
            mean=clone(self.mean),
            logvar=clone(self.logvar),
        )


class LatentMatrixCache:
    """A small npz cache for frozen encoder transforms.

    Cache keys include the serialized encoder state and raw input bytes, so a
    cache cannot silently mix representations fitted in different cycles.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def get(self, key: str) -> NativeLatentOutput | None:
        path = self.directory / f"{key}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            def read(name: str) -> np.ndarray | None:
                value = data.get(name)
                return None if value is None or value.size == 0 else value.astype(np.float32)

            latent = read("latent")
            if latent is None:
                return None
            return NativeLatentOutput(
                latent=latent,
                reconstruction=read("reconstruction"),
                reconstruction_error=read("reconstruction_error"),
                cluster_probabilities=read("cluster_probabilities"),
                mean=read("mean"),
                logvar=read("logvar"),
            )

    def put(self, key: str, value: NativeLatentOutput) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.npz"
        np.savez_compressed(
            path,
            latent=np.asarray(value.latent, dtype=np.float32),
            reconstruction=_cache_array(value.reconstruction),
            reconstruction_error=_cache_array(value.reconstruction_error),
            cluster_probabilities=_cache_array(value.cluster_probabilities),
            mean=_cache_array(value.mean),
            logvar=_cache_array(value.logvar),
        )


def _cache_array(value: np.ndarray | None) -> np.ndarray:
    return np.empty(0, dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32)


def side_conditioned_corruption(
    values: Any,
    sides: Sequence[Any] | None,
    *,
    groups: Sequence[Any] | None = None,
    feature_group_indices: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
    donor_regime_labels: Sequence[Any] | None = None,
    corruption_rate: float = 0.15,
    element_mask_rate: float = 0.0,
    whole_feature_group_mask_rate: float = 0.0,
    additive_noise_std: float = 0.0,
    group_donor_replacement_rate: float = 0.0,
    side_feature_group: str = "side",
    side_feature_indices: Sequence[int] | None = None,
    random_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create side-safe SSL views with row donors and column feature groups.

    ``groups`` remains the backward-compatible *row* donor partition.  Frozen
    ``donor_regime_labels`` add a second row partition. ``feature_group_indices``
    controls whole-column-group operators.  A group named ``side`` (or the
    explicit protected indices) is never altered by any operator.
    """

    x = _as_matrix(values)
    n, d = x.shape
    if sides is None:
        side_values = np.repeat("__all__", n)
    else:
        side_values = _as_labels(sides, n, "sides")
    group_values = (
        np.repeat("__all__", n)
        if groups is None
        else _as_labels(groups, n, "groups")
    )
    regime_values = (
        np.repeat("__all__", n)
        if donor_regime_labels is None
        else _as_labels(donor_regime_labels, n, "donor_regime_labels")
    )
    feature_groups = _normalize_feature_groups(feature_group_indices, d)
    protected = _protected_feature_indices(
        feature_groups, side_feature_group, side_feature_indices, d
    )
    mutable = np.ones(d, dtype=bool)
    mutable[protected] = False
    generator = rng if rng is not None else np.random.default_rng(random_state)
    out = x.copy()
    mask = np.zeros((n, d), dtype=bool)
    donors = np.full(n, -1, dtype=np.int64)
    partitions: dict[tuple[str, str, str], list[int]] = {}
    for i, (side, group, regime) in enumerate(
        zip(side_values, group_values, regime_values)
    ):
        partitions.setdefault((str(side), str(group), str(regime)), []).append(i)
    rate = float(np.clip(corruption_rate, 0.0, 1.0))
    element_mask_rate = float(np.clip(element_mask_rate, 0.0, 1.0))
    whole_feature_group_mask_rate = float(
        np.clip(whole_feature_group_mask_rate, 0.0, 1.0)
    )
    group_donor_replacement_rate = float(
        np.clip(group_donor_replacement_rate, 0.0, 1.0)
    )
    noise_std = max(0.0, float(additive_noise_std))
    for positions in partitions.values():
        pos = np.asarray(positions, dtype=np.int64)
        for row in pos:
            donor = -1
            if len(pos) >= 2 and (rate > 0.0 or group_donor_replacement_rate > 0.0):
                candidates = pos[pos != row]
                donor = int(candidates[generator.integers(len(candidates))])
                feature_mask = (generator.random(d) < rate) & mutable
                if feature_mask.any():
                    out[row, feature_mask] = x[donor, feature_mask]
                    mask[row, feature_mask] = True
                    donors[row] = donor
                for indices in feature_groups.values():
                    selected = indices[~np.isin(indices, protected)]
                    if len(selected) and generator.random() < group_donor_replacement_rate:
                        out[row, selected] = x[donor, selected]
                        mask[row, selected] = True
                        donors[row] = donor
            for indices in feature_groups.values():
                selected = indices[~np.isin(indices, protected)]
                if len(selected) and generator.random() < whole_feature_group_mask_rate:
                    out[row, selected] = 0.0
                    mask[row, selected] = True
            element_mask = (generator.random(d) < element_mask_rate) & mutable
            if element_mask.any():
                out[row, element_mask] = 0.0
                mask[row, element_mask] = True
            if noise_std > 0.0 and mutable.any():
                noise = generator.normal(0.0, noise_std, size=d).astype(np.float32)
                out[row, mutable] += noise[mutable]
                mask[row, mutable] = True
    return out.astype(np.float32, copy=False), mask, donors


if nn is not None:  # pragma: no branch

    class _ResidualBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.first = nn.Linear(width, width)
            self.second = nn.Linear(width, width)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values + self.second(F.gelu(self.first(values)))


    class _SideConditionedAE(nn.Module):
        def __init__(self, input_dim: int, config: EncoderConfig, side_vocab: int) -> None:
            super().__init__()
            self.side_embedding = nn.Embedding(side_vocab, min(8, max(2, config.hidden_dim // 8)))
            side_dim = self.side_embedding.embedding_dim
            self.encoder_in = nn.Linear(input_dim + side_dim, config.hidden_dim)
            self.encoder_blocks = nn.ModuleList(
                [_ResidualBlock(config.hidden_dim) for _ in range(config.residual_blocks)]
            )
            self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim)
            self.decoder_in = nn.Linear(config.latent_dim + side_dim, config.hidden_dim)
            self.decoder_blocks = nn.ModuleList(
                [_ResidualBlock(config.hidden_dim) for _ in range(config.residual_blocks)]
            )
            self.to_reconstruction = nn.Linear(config.hidden_dim, input_dim)
            # It is used only by IDEC, but registering it for every residual AE
            # keeps restore strict and avoids a separate non-serializable object.
            self.cluster_centers = nn.Parameter(
                torch.zeros(config.n_clusters, config.latent_dim), requires_grad=True
            )

        def encode(self, values: torch.Tensor, side_codes: torch.Tensor) -> torch.Tensor:
            side = self.side_embedding(side_codes)
            hidden = F.gelu(self.encoder_in(torch.cat((values, side), dim=1)))
            for block in self.encoder_blocks:
                hidden = F.gelu(block(hidden))
            return self.to_latent(hidden)

        def decode(self, latent: torch.Tensor, side_codes: torch.Tensor) -> torch.Tensor:
            side = self.side_embedding(side_codes)
            hidden = F.gelu(self.decoder_in(torch.cat((latent, side), dim=1)))
            for block in self.decoder_blocks:
                hidden = F.gelu(block(hidden))
            return self.to_reconstruction(hidden)

        def forward(self, values: torch.Tensor, side_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent = self.encode(values, side_codes)
            return self.decode(latent, side_codes), latent


    class _VaDENetwork(nn.Module):
        def __init__(self, input_dim: int, config: EncoderConfig, side_vocab: int) -> None:
            super().__init__()
            self.backbone = _SideConditionedAE(input_dim, config, side_vocab)
            self.to_mu = nn.Linear(config.latent_dim, config.latent_dim)
            self.to_logvar = nn.Linear(config.latent_dim, config.latent_dim)
            self.cluster_logits = nn.Parameter(torch.zeros(config.n_clusters))
            self.cluster_means = nn.Parameter(torch.zeros(config.n_clusters, config.latent_dim))
            self.cluster_logvars = nn.Parameter(torch.zeros(config.n_clusters, config.latent_dim))

        def encode_distribution(
            self, values: torch.Tensor, side_codes: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.backbone.encode(values, side_codes)
            return self.to_mu(hidden), torch.clamp(self.to_logvar(hidden), -8.0, 6.0)

        def forward(
            self, values: torch.Tensor, side_codes: torch.Tensor, sample: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode_distribution(values, side_codes)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
            return self.backbone.decode(latent, side_codes), mu, logvar


class AlternativeLatentEncoder:
    """A fitted encoder with stable NumPy transform and serialized-state APIs."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self.feature_center: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.feature_group_indices: dict[str, np.ndarray] = {}
        self.protected_feature_indices = np.empty(0, dtype=np.int64)
        self.side_mapping: dict[str, int] = {}
        self._model: Any = None
        self._legacy_state: dict[str, Any] | None = None
        self.training_report: dict[str, Any] = {}
        self._fitted = False

    @property
    def kind(self) -> str:
        return str(self.config.kind).lower().strip()

    @property
    def latent_dim(self) -> int:
        if self.kind == "dae" and self._legacy_state is not None:
            return int(self._legacy_state.get("selected_bottleneck", self.config.latent_dim))
        return int(self.config.latent_dim)

    def fit(
        self,
        values: Any,
        *,
        sides: Sequence[Any] | None = None,
        groups: Sequence[Any] | None = None,
        feature_group_indices: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
        donor_regime_labels: Sequence[Any] | None = None,
        initialization_state: Mapping[str, Any] | None = None,
        pretraining_state: Mapping[str, Any] | None = None,
    ) -> "AlternativeLatentEncoder":
        if self.kind not in ENCODER_KINDS:
            raise ValueError(f"Unsupported latent encoder kind: {self.config.kind!r}")
        x = _as_matrix(values)
        if len(x) < 2 or x.shape[1] < 1:
            raise ValueError("Latent encoder requires at least two rows and one feature")
        side_labels = _as_labels(sides, len(x), "sides") if sides is not None else np.repeat("__all__", len(x))
        _as_labels(groups, len(x), "groups") if groups is not None else None
        _as_labels(donor_regime_labels, len(x), "donor_regime_labels") if donor_regime_labels is not None else None
        self.feature_group_indices = _normalize_feature_groups(feature_group_indices, x.shape[1])
        self.protected_feature_indices = _protected_feature_indices(
            self.feature_group_indices,
            self.config.side_feature_group,
            self.config.side_feature_indices,
            x.shape[1],
        )
        self.feature_center, self.feature_scale = _robust_location_scale(x)
        clean = self._prepare(values)
        self.side_mapping = {name: i + 1 for i, name in enumerate(sorted(set(map(str, side_labels))))}
        if self.kind == "dae":
            self._fit_dae(clean)
        else:
            self._fit_torch(
                clean,
                side_labels,
                groups,
                donor_regime_labels,
                initialization_state,
                pretraining_state,
            )
        self._fitted = True
        return self

    def fit_idec_pretraining_state(
        self,
        values: Any,
        *,
        sides: Sequence[Any] | None = None,
        feature_group_indices: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
    ) -> dict[str, Any]:
        """Fit and serialize only the IDEC reconstruction pretraining phase.

        The state deliberately excludes cluster centers.  It can therefore be
        reused for every ``K`` and clustering-weight choice sharing the same
        architecture and reference sample without leaking a prior assignment.
        """
        if self.kind != "idec":
            raise ValueError("IDEC pretraining cache is only valid for IDEC encoders")
        _require_torch()
        x = _as_matrix(values)
        side_labels = (
            _as_labels(sides, len(x), "sides")
            if sides is not None
            else np.repeat("__all__", len(x))
        )
        self.feature_group_indices = _normalize_feature_groups(
            feature_group_indices, x.shape[1]
        )
        self.protected_feature_indices = _protected_feature_indices(
            self.feature_group_indices,
            self.config.side_feature_group,
            self.config.side_feature_indices,
            x.shape[1],
        )
        self.feature_center, self.feature_scale = _robust_location_scale(x)
        clean = self._prepare(values)
        self.side_mapping = {
            name: index + 1
            for index, name in enumerate(sorted(set(map(str, side_labels))))
        }
        _seed_torch(self.config.random_state)
        model = self._new_torch_model(clean.shape[1]).to(
            device=self._device(), dtype=torch.float32
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._pretrain_autoencoder(
            model,
            clean,
            self._side_codes(side_labels),
            optimizer,
            np.random.default_rng(self.config.random_state),
        )
        torch_state = _torch_state_to_lists(model.state_dict())
        torch_state.pop("cluster_centers", None)
        return {
            "schema_version": "idec_pretraining_v1",
            "architecture": {
                "input_dim": int(clean.shape[1]),
                "latent_dim": int(self.config.latent_dim),
                "hidden_dim": int(self.config.hidden_dim),
                "residual_blocks": int(self.config.residual_blocks),
                "pretraining_fraction": self.config.pretraining_fraction,
                "pretrain_epochs": int(self.config.pretrain_epochs),
            },
            "feature_center": _array_list(self.feature_center),
            "feature_scale": _array_list(self.feature_scale),
            "side_mapping": dict(self.side_mapping),
            "torch_state": torch_state,
        }

    def fit_transform(
        self,
        values: Any,
        *,
        sides: Sequence[Any] | None = None,
        groups: Sequence[Any] | None = None,
        feature_group_indices: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None = None,
        donor_regime_labels: Sequence[Any] | None = None,
        initialization_state: Mapping[str, Any] | None = None,
        pretraining_state: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        return self.fit(
            values,
            sides=sides,
            groups=groups,
            feature_group_indices=feature_group_indices,
            donor_regime_labels=donor_regime_labels,
            initialization_state=initialization_state,
            pretraining_state=pretraining_state,
        ).transform(values, sides=sides)

    def transform(self, values: Any, *, sides: Sequence[Any] | None = None) -> np.ndarray:
        return self.transform_native(values, sides=sides).latent

    def transform_native(self, values: Any, *, sides: Sequence[Any] | None = None) -> NativeLatentOutput:
        self._require_fitted()
        clean = self._prepare(values)
        if sides is None and set(self.side_mapping) != {"__all__"}:
            raise ValueError("This side-conditioned encoder requires sides at transform time")
        labels = _as_labels(sides, len(clean), "sides") if sides is not None else np.repeat("__all__", len(clean))
        if self.kind == "dae":
            return self._transform_dae(clean)
        return self._transform_torch(clean, labels)

    def cache_key(
        self, values: Any, *, sides: Sequence[Any] | None = None, groups: Sequence[Any] | None = None
    ) -> str:
        self._require_fitted()
        digest = hashlib.sha256()
        digest.update(_canonical_json(self.to_state()).encode("utf-8"))
        digest.update(_as_matrix(values).tobytes(order="C"))
        for label_set in (sides, groups):
            if label_set is not None:
                digest.update("\x1f".join(map(str, label_set)).encode("utf-8"))
        return digest.hexdigest()

    def transform_cached(
        self,
        values: Any,
        *,
        sides: Sequence[Any] | None = None,
        groups: Sequence[Any] | None = None,
        cache: LatentMatrixCache | MutableMapping[str, NativeLatentOutput] | None = None,
    ) -> NativeLatentOutput:
        key = self.cache_key(values, sides=sides, groups=groups)
        if cache is not None:
            cached = cache.get(key) if isinstance(cache, LatentMatrixCache) else cache.get(key)
            if cached is not None:
                return cached.copy()
        native = self.transform_native(values, sides=sides)
        if cache is not None:
            if isinstance(cache, LatentMatrixCache):
                cache.put(key, native)
            else:
                cache[key] = native.copy()
        return native

    def to_state(self) -> dict[str, Any]:
        self._require_fitted()
        state: dict[str, Any] = {
            "schema_version": "alternative_latent_encoder_v1",
            "config": asdict(self.config),
            "feature_center": _array_list(self.feature_center),
            "feature_scale": _array_list(self.feature_scale),
            "feature_group_indices": {
                name: indices.astype(int).tolist()
                for name, indices in self.feature_group_indices.items()
            },
            "protected_feature_indices": self.protected_feature_indices.astype(int).tolist(),
            "side_mapping": dict(self.side_mapping),
            "training_report": _json_safe(self.training_report),
        }
        if self.kind == "dae":
            state["legacy_state"] = _json_safe(self._legacy_state)
        else:
            state["torch_state"] = _torch_state_to_lists(self._model.state_dict())
        return state

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], *, device: str | None = None
    ) -> "AlternativeLatentEncoder":
        if str(state.get("schema_version")) != "alternative_latent_encoder_v1":
            raise ValueError("Unsupported alternative latent encoder state")
        raw_config = dict(state.get("config", {}))
        if device is not None:
            raw_config["device"] = device
        encoder = cls(EncoderConfig(**raw_config))
        encoder.feature_center = np.asarray(state.get("feature_center", []), dtype=np.float32)
        encoder.feature_scale = np.asarray(state.get("feature_scale", []), dtype=np.float32)
        encoder.feature_group_indices = _normalize_feature_groups(
            state.get("feature_group_indices", {}), encoder.feature_center.size
        )
        encoder.protected_feature_indices = np.asarray(
            state.get("protected_feature_indices", []), dtype=np.int64
        )
        encoder.side_mapping = {str(k): int(v) for k, v in dict(state.get("side_mapping", {})).items()}
        encoder.training_report = dict(state.get("training_report", {}))
        if encoder.feature_center.size == 0 or encoder.feature_center.size != encoder.feature_scale.size:
            raise ValueError("Encoder state has invalid preprocessing vectors")
        if encoder.kind == "dae":
            encoder._legacy_state = dict(state.get("legacy_state", {}))
        else:
            encoder._model = encoder._new_torch_model(int(encoder.feature_center.size))
            _load_torch_state_from_lists(encoder._model, state.get("torch_state", {}), encoder._device())
            encoder._model.eval()
        encoder._fitted = True
        return encoder

    def _fit_dae(self, clean: np.ndarray) -> None:
        state = fit_denoising_autoencoder_state(
            clean,
            random_state=self.config.random_state,
            max_train_rows=self.config.dae_max_train_rows,
            max_iter=max(1, self.config.epochs),
        )
        if not bool(state.get("enabled", False)):
            raise RuntimeError(f"Legacy DAE fit failed: {state.get('reason', 'no_model')}")
        requested = int(self.config.latent_dim)
        available = [8, 16]
        selected = min(available, key=lambda dim: (abs(dim - requested), dim))
        if f"b{selected}" not in state.get("models", {}):
            selected = 16 if "b16" in state.get("models", {}) else 8
        state["selected_bottleneck"] = selected
        self._legacy_state = state
        self.training_report = {"backend": "sklearn_dae", "selected_bottleneck": selected}

    def _transform_dae(self, clean: np.ndarray) -> NativeLatentOutput:
        assert self._legacy_state is not None
        features = transform_denoising_autoencoder_features(clean, self._legacy_state)
        dim = int(self._legacy_state["selected_bottleneck"])
        columns = [f"ae_b{dim}_{i:02d}" for i in range(dim)]
        latent = features.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)
        error = features[f"ae_b{dim}_reconstruction_error"].to_numpy(dtype=np.float32)
        return NativeLatentOutput(latent=latent, reconstruction_error=error)

    def _fit_torch(
        self,
        clean: np.ndarray,
        sides: Sequence[Any],
        groups: Sequence[Any] | None,
        donor_regime_labels: Sequence[Any] | None,
        initialization_state: Mapping[str, Any] | None,
        pretraining_state: Mapping[str, Any] | None,
    ) -> None:
        _require_torch()
        device = self._device()
        _seed_torch(self.config.random_state)
        model = self._new_torch_model(clean.shape[1]).to(device=device, dtype=torch.float32)
        side_codes = self._side_codes(sides)
        group_values = _as_labels(groups, len(clean), "groups") if groups is not None else None
        regime_values = (
            _as_labels(donor_regime_labels, len(clean), "donor_regime_labels")
            if donor_regime_labels is not None
            else None
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        rng = np.random.default_rng(self.config.random_state)
        if self.kind == "idec":
            if pretraining_state is None:
                self._pretrain_autoencoder(model, clean, side_codes, optimizer, rng)
            else:
                self._load_idec_pretraining_state(model, pretraining_state, clean.shape[1])
            self._fit_idec(model, clean, side_codes, optimizer, rng, initialization_state)
        elif self.kind == "vade":
            self._fit_vade(model, clean, side_codes, optimizer, rng, initialization_state)
        else:
            self._fit_ssl(
                model, clean, side_codes, group_values, regime_values, optimizer, rng
            )
        model.eval()
        self._model = model
        self.training_report = {
            "backend": "torch",
            "device": str(device),
            "epochs": int(self.config.epochs),
            "kind": self.kind,
            "rows": int(len(clean)),
            "feature_groups": list(self.feature_group_indices),
            "protected_feature_indices": self.protected_feature_indices.astype(int).tolist(),
            "reused_idec_pretraining": bool(
                self.kind == "idec" and pretraining_state is not None
            ),
        }

    def _load_idec_pretraining_state(
        self, model: Any, state: Mapping[str, Any], input_dim: int
    ) -> None:
        if str(state.get("schema_version")) != "idec_pretraining_v1":
            raise ValueError("Unsupported IDEC pretraining cache state")
        architecture = dict(state.get("architecture", {}))
        expected = {
            "input_dim": int(input_dim),
            "latent_dim": int(self.config.latent_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "residual_blocks": int(self.config.residual_blocks),
        }
        if any(int(architecture.get(key, -1)) != value for key, value in expected.items()):
            raise ValueError("IDEC pretraining cache architecture does not match encoder")
        cached_center = np.asarray(state.get("feature_center", ()), dtype=np.float32)
        cached_scale = np.asarray(state.get("feature_scale", ()), dtype=np.float32)
        if (
            self.feature_center is None
            or self.feature_scale is None
            or cached_center.shape != self.feature_center.shape
            or cached_scale.shape != self.feature_scale.shape
            or not np.allclose(cached_center, self.feature_center, rtol=0.0, atol=1e-6)
            or not np.allclose(cached_scale, self.feature_scale, rtol=0.0, atol=1e-6)
            or dict(state.get("side_mapping", {})) != dict(self.side_mapping)
        ):
            raise ValueError("IDEC pretraining cache does not match preprocessing contract")
        _load_partial_torch_state_from_lists(
            model,
            state.get("torch_state", {}),
            self._device(),
            excluded=("cluster_centers",),
        )

    def _fit_ssl(
        self,
        model: Any,
        clean: np.ndarray,
        side_codes: np.ndarray,
        groups: Sequence[Any] | None,
        donor_regime_labels: Sequence[Any] | None,
        optimizer: Any,
        rng: np.random.Generator,
    ) -> None:
        for _epoch in range(max(1, int(self.config.epochs))):
            for batch in _batches(len(clean), self.config.batch_size, rng):
                x = clean[batch]
                codes = side_codes[batch]
                batch_groups = None if groups is None else np.asarray(groups, dtype=object)[batch]
                batch_regimes = (
                    None
                    if donor_regime_labels is None
                    else np.asarray(donor_regime_labels, dtype=object)[batch]
                )
                corrupted, mask, _ = side_conditioned_corruption(
                    x,
                    np.asarray(side_codes)[batch],
                    groups=batch_groups,
                    donor_regime_labels=batch_regimes,
                    feature_group_indices=self.feature_group_indices,
                    corruption_rate=self.config.corruption_rate,
                    element_mask_rate=self.config.element_mask_rate,
                    whole_feature_group_mask_rate=self.config.whole_feature_group_mask_rate,
                    additive_noise_std=self.config.additive_noise_std,
                    group_donor_replacement_rate=self.config.group_donor_replacement_rate,
                    side_feature_group=self.config.side_feature_group,
                    side_feature_indices=self.protected_feature_indices,
                    rng=rng,
                )
                weak_first, _weak_first_mask, _ = side_conditioned_corruption(
                    x,
                    np.asarray(side_codes)[batch],
                    groups=batch_groups,
                    donor_regime_labels=batch_regimes,
                    feature_group_indices=self.feature_group_indices,
                    corruption_rate=0.0,
                    element_mask_rate=0.05,
                    whole_feature_group_mask_rate=0.0,
                    additive_noise_std=0.01,
                    group_donor_replacement_rate=0.0,
                    side_feature_group=self.config.side_feature_group,
                    side_feature_indices=self.protected_feature_indices,
                    rng=rng,
                )
                if self.kind != "masked" and self.config.ssl_view_pair == "weak_weak":
                    corrupted, mask, _ = side_conditioned_corruption(
                        x,
                        np.asarray(side_codes)[batch],
                        groups=batch_groups,
                        donor_regime_labels=batch_regimes,
                        feature_group_indices=self.feature_group_indices,
                        corruption_rate=0.0,
                        element_mask_rate=0.05,
                        whole_feature_group_mask_rate=0.0,
                        additive_noise_std=0.01,
                        group_donor_replacement_rate=0.0,
                        side_feature_group=self.config.side_feature_group,
                        side_feature_indices=self.protected_feature_indices,
                        rng=rng,
                    )
                elif self.kind != "masked" and self.config.ssl_view_pair != "weak_strong":
                    raise ValueError("ssl_view_pair must be weak_weak or weak_strong")
                xb, cb = self._torch_batch(x, codes)
                corrupt_b, _ = self._torch_batch(corrupted, codes)
                weak_first_b, _ = self._torch_batch(weak_first, codes)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, latent = model(corrupt_b, cb)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=xb.device)
                reconstruction_mask = (
                    None
                    if self.config.ssl_objective == "denoising_reconstruction"
                    else mask_t
                )
                recon_loss = self._reconstruction_loss(
                    reconstruction, xb, reconstruction_mask
                )
                if self.kind == "masked":
                    loss = recon_loss
                else:
                    clean_latent = model.encode(weak_first_b, cb)
                    if self.kind == "scarf":
                        ssl = _nt_xent(clean_latent, latent, self.config.contrastive_temperature)
                    else:
                        ssl = _vicreg_loss(clean_latent, latent, self.config)
                    loss = self.config.reconstruction_weight * recon_loss + ssl
                loss.backward()
                optimizer.step()

    def _pretrain_autoencoder(
        self,
        model: Any,
        clean: np.ndarray,
        side_codes: np.ndarray,
        optimizer: Any,
        rng: np.random.Generator,
    ) -> None:
        epochs = int(self.config.pretrain_epochs)
        if self.config.pretraining_fraction is not None:
            epochs = max(1, int(round(float(self.config.epochs) * float(self.config.pretraining_fraction))))
        for _epoch in range(max(1, epochs)):
            for batch in _batches(len(clean), self.config.batch_size, rng):
                xb, cb = self._torch_batch(clean[batch], side_codes[batch])
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = model(xb, cb)
                self._reconstruction_loss(reconstruction, xb).backward()
                optimizer.step()

    def _fit_idec(
        self,
        model: Any,
        clean: np.ndarray,
        side_codes: np.ndarray,
        optimizer: Any,
        rng: np.random.Generator,
        initialization_state: Mapping[str, Any] | None,
    ) -> None:
        with torch.no_grad():
            z = self._encode_all(model, clean, side_codes)
        count = int(self.config.n_clusters)
        if count < 2 or count > len(z):
            raise ValueError("IDEC n_clusters must be between 2 and the fitted row count")
        mode = _initialization_mode(self.config.initialization, "idec")
        if mode == "incumbent_means":
            if initialization_state is not None:
                centers = _initial_cluster_means(initialization_state, count, self.latent_dim)
            else:
                if GaussianMixture is None:
                    raise RuntimeError("IDEC GMM-means initialization requires scikit-learn")
                centers = GaussianMixture(
                    n_components=count,
                    covariance_type="diag",
                    reg_covar=0.003,
                    n_init=3,
                    random_state=self.config.random_state,
                ).fit(z).means_
        else:
            if KMeans is None:
                raise RuntimeError("IDEC kmeans++ initialization requires scikit-learn")
            centers = KMeans(
                n_clusters=count,
                init="k-means++",
                n_init=10,
                random_state=self.config.random_state,
            ).fit(z).cluster_centers_
        with torch.no_grad():
            model.cluster_centers.copy_(torch.as_tensor(centers, dtype=torch.float32, device=self._device()))
        centers_t = model.cluster_centers
        frequency = int(self.config.target_update_frequency)
        if frequency < 1:
            raise ValueError("target_update_frequency must be at least one")
        target = None
        for epoch in range(max(1, int(self.config.epochs))):
            if target is None or epoch % frequency == 0:
                with torch.no_grad():
                    q_all = _student_t_assignments(
                        torch.as_tensor(
                            self._encode_all(model, clean, side_codes),
                            device=self._device(),
                        ),
                        centers_t,
                        self.config.student_t_df,
                    )
                    target = _idec_target(q_all).detach()
            for batch in _batches(len(clean), self.config.batch_size, rng):
                xb, cb = self._torch_batch(clean[batch], side_codes[batch])
                optimizer.zero_grad(set_to_none=True)
                reconstruction, latent = model(xb, cb)
                q = _student_t_assignments(latent, centers_t, self.config.student_t_df)
                recon = self._reconstruction_loss(reconstruction, xb)
                kl = F.kl_div(torch.log(q.clamp_min(1e-8)), target[batch], reduction="batchmean")
                (recon + self.config.cluster_weight * kl).backward()
                optimizer.step()

    def _fit_vade(
        self,
        model: Any,
        clean: np.ndarray,
        side_codes: np.ndarray,
        optimizer: Any,
        rng: np.random.Generator,
        initialization_state: Mapping[str, Any] | None,
    ) -> None:
        if initialization_state is None and _initialization_mode(self.config.initialization, "vade") != "random":
            self._pretrain_and_initialize_vade(model, clean, side_codes, optimizer, rng)
        else:
            self._initialize_vade(model, initialization_state)
        total_epochs = max(1, int(self.config.epochs))
        for epoch in range(total_epochs):
            kl_scale = self._kl_scale(epoch, total_epochs)
            for batch in _batches(len(clean), self.config.batch_size, rng):
                xb, cb = self._torch_batch(clean[batch], side_codes[batch])
                optimizer.zero_grad(set_to_none=True)
                reconstruction, mu, logvar = model(xb, cb, sample=True)
                recon = self._reconstruction_loss(reconstruction, xb)
                kl, assignments = _vade_kl(model, mu, logvar)
                occupancy = _occupancy_penalty(
                    assignments, self.config.min_effective_occupancy
                )
                (recon + kl_scale * (kl + occupancy)).backward()
                optimizer.step()

    def _pretrain_and_initialize_vade(
        self,
        model: Any,
        clean: np.ndarray,
        side_codes: np.ndarray,
        optimizer: Any,
        rng: np.random.Generator,
    ) -> None:
        mode = _initialization_mode(self.config.initialization, "vade")
        epochs = int(self.config.pretrain_epochs)
        if self.config.pretraining_fraction is not None:
            epochs = max(1, int(round(float(self.config.epochs) * float(self.config.pretraining_fraction))))
        for _epoch in range(max(1, epochs)):
            for batch in _batches(len(clean), self.config.batch_size, rng):
                xb, cb = self._torch_batch(clean[batch], side_codes[batch])
                optimizer.zero_grad(set_to_none=True)
                sample = mode == "pretrained_vae_gmm"
                reconstruction, mu, logvar = model(xb, cb, sample=sample)
                loss = self._reconstruction_loss(reconstruction, xb)
                if sample:
                    standard_kl = -0.5 * torch.mean(
                        torch.sum(1.0 + logvar - mu.square() - torch.exp(logvar), dim=1)
                    )
                    loss = loss + 0.1 * standard_kl
                loss.backward()
                optimizer.step()
        means: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch in _fixed_batches(len(clean), self.config.batch_size):
                xb, cb = self._torch_batch(clean[batch], side_codes[batch])
                mu, _logvar = model.encode_distribution(xb, cb)
                means.append(mu.detach().cpu().numpy())
        latent = np.concatenate(means, axis=0).astype(np.float32, copy=False)
        if GaussianMixture is None:
            raise RuntimeError("VaDE pretrained GMM initialization requires scikit-learn")
        gmm = GaussianMixture(
            n_components=int(self.config.n_clusters),
            covariance_type="diag",
            reg_covar=0.003,
            n_init=3,
            random_state=self.config.random_state,
        ).fit(latent)
        with torch.no_grad():
            model.cluster_means.copy_(
                torch.as_tensor(gmm.means_, dtype=torch.float32, device=self._device())
            )
            model.cluster_logvars.copy_(
                torch.log(
                    torch.as_tensor(
                        np.maximum(gmm.covariances_, 1e-6),
                        dtype=torch.float32,
                        device=self._device(),
                    )
                )
            )
            model.cluster_logits.copy_(
                torch.log(
                    torch.as_tensor(
                        np.maximum(gmm.weights_, 1e-8),
                        dtype=torch.float32,
                        device=self._device(),
                    )
                )
            )

    def _initialize_vade(
        self, model: Any, initialization_state: Mapping[str, Any] | None
    ) -> None:
        mode = _initialization_mode(self.config.initialization, "vade")
        if mode == "random":
            return
        means, logvars, logits = _initial_vade_parameters(
            initialization_state,
            int(self.config.n_clusters),
            int(self.config.latent_dim),
            mode,
        )
        with torch.no_grad():
            model.cluster_means.copy_(torch.as_tensor(means, device=self._device()))
            model.cluster_logvars.copy_(torch.as_tensor(logvars, device=self._device()))
            model.cluster_logits.copy_(torch.as_tensor(logits, device=self._device()))

    def _kl_scale(self, epoch: int, total_epochs: int) -> float:
        weight = float(
            self.config.cluster_weight
            if self.config.kl_weight is None
            else self.config.kl_weight
        )
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("kl_weight must be finite and non-negative")
        if not 0.0 <= float(self.config.kl_warmup_fraction) <= 1.0:
            raise ValueError("kl_warmup_fraction must be in [0, 1]")
        warmup_epochs = int(
            np.ceil(total_epochs * float(self.config.kl_warmup_fraction))
        )
        if warmup_epochs <= 0:
            return weight
        return weight * min(1.0, float(epoch + 1) / float(warmup_epochs))

    def _reconstruction_loss(
        self, reconstruction: Any, target: Any, mask: Any | None = None
    ) -> Any:
        if self.config.reconstruction_objective == "current" or not self.feature_group_indices:
            squared = (reconstruction - target).square()
            return squared[mask].mean() if mask is not None and bool(mask.any()) else squared.mean()
        if self.config.reconstruction_objective != "feature_group_weighted":
            raise ValueError("reconstruction_objective must be current or feature_group_weighted")
        losses: list[Any] = []
        for indices in self.feature_group_indices.values():
            if not len(indices):
                continue
            index = torch.as_tensor(indices, dtype=torch.long, device=target.device)
            squared = (reconstruction.index_select(1, index) - target.index_select(1, index)).square()
            if mask is not None:
                local_mask = mask.index_select(1, index)
                if bool(local_mask.any()):
                    losses.append(squared[local_mask].mean())
                continue
            losses.append(squared.mean())
        return torch.stack(losses).mean() if losses else (reconstruction - target).square().mean()

    def _transform_torch(self, clean: np.ndarray, sides: Sequence[Any]) -> NativeLatentOutput:
        assert self._model is not None
        codes = self._side_codes(sides)
        latents: list[np.ndarray] = []
        reconstructions: list[np.ndarray] = []
        means: list[np.ndarray] = []
        logvars: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for batch in _fixed_batches(len(clean), self.config.batch_size):
                xb, cb = self._torch_batch(clean[batch], codes[batch])
                if self.kind == "vade":
                    reconstruction, mu, logvar = self._model(xb, cb, sample=False)
                    latent = mu
                    probability = _vade_assignments(self._model, mu)
                    means.append(mu.detach().cpu().numpy())
                    logvars.append(logvar.detach().cpu().numpy())
                    probs.append(probability.detach().cpu().numpy())
                else:
                    reconstruction, latent = self._model(xb, cb)
                    if self.kind == "idec" and hasattr(self._model, "cluster_centers"):
                        probs.append(
                            _student_t_assignments(
                                latent,
                                self._model.cluster_centers,
                                self.config.student_t_df,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                latents.append(latent.detach().cpu().numpy())
                reconstructions.append(reconstruction.detach().cpu().numpy())
        latent = np.concatenate(latents, axis=0).astype(np.float32) if latents else np.empty((0, self.latent_dim), dtype=np.float32)
        reconstruction = np.concatenate(reconstructions, axis=0).astype(np.float32) if reconstructions else np.empty_like(clean)
        return NativeLatentOutput(
            latent=latent,
            reconstruction=reconstruction,
            reconstruction_error=np.mean(np.square(reconstruction - clean), axis=1).astype(np.float32),
            cluster_probabilities=np.concatenate(probs, axis=0).astype(np.float32) if probs else None,
            mean=np.concatenate(means, axis=0).astype(np.float32) if means else None,
            logvar=np.concatenate(logvars, axis=0).astype(np.float32) if logvars else None,
        )

    def _new_torch_model(self, input_dim: int) -> Any:
        _require_torch()
        side_vocab = max(self.side_mapping.values(), default=0) + 1
        if self.kind == "vade":
            return _VaDENetwork(input_dim, self.config, side_vocab)
        return _SideConditionedAE(input_dim, self.config, side_vocab)

    def _encode_all(self, model: Any, clean: np.ndarray, side_codes: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch in _fixed_batches(len(clean), self.config.batch_size):
                xb, cb = self._torch_batch(clean[batch], side_codes[batch])
                parts.append(model.encode(xb, cb).detach().cpu().numpy())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _prepare(self, values: Any) -> np.ndarray:
        x = _as_matrix(values)
        if self.feature_center is None or self.feature_scale is None:
            raise RuntimeError("Encoder preprocessing is not fitted")
        if x.shape[1] != len(self.feature_center):
            raise ValueError(f"Expected {len(self.feature_center)} features, received {x.shape[1]}")
        filled = np.where(np.isfinite(x), x, self.feature_center.reshape(1, -1))
        return np.clip((filled - self.feature_center) / self.feature_scale, -8.0, 8.0).astype(np.float32)

    def _side_codes(self, sides: Sequence[Any]) -> np.ndarray:
        labels = _as_labels(sides, len(sides), "sides")
        unknown = sorted({str(label) for label in labels}.difference(self.side_mapping))
        if unknown:
            raise ValueError(f"Encoder received unseen side labels: {unknown}")
        return np.asarray([self.side_mapping[str(label)] for label in labels], dtype=np.int64)

    def _torch_batch(self, values: np.ndarray, side_codes: np.ndarray) -> tuple[Any, Any]:
        device = self._device()
        return (
            torch.as_tensor(values, dtype=torch.float32, device=device),
            torch.as_tensor(side_codes, dtype=torch.long, device=device),
        )

    def _device(self) -> Any:
        _require_torch()
        requested = self.config.device.lower()
        if requested == "auto":
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if requested == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        if requested not in {"cpu", "mps"}:
            raise ValueError("Only cpu, mps, and auto encoder devices are supported")
        return torch.device(requested)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Latent encoder is not fitted")


def build_encoder(config: EncoderConfig | None = None, **kwargs: Any) -> AlternativeLatentEncoder:
    """Construct an encoder, allowing concise keyword configuration."""

    if config is not None and kwargs:
        raise ValueError("Pass either config or keyword settings, not both")
    return AlternativeLatentEncoder(config or EncoderConfig(**kwargs))


def restore_encoder(state: Mapping[str, Any], *, device: str | None = None) -> AlternativeLatentEncoder:
    return AlternativeLatentEncoder.from_state(state, device=device)


def _as_matrix(values: Any) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Encoder values must be a two-dimensional numeric matrix")
    return np.nan_to_num(x, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32, copy=False)


def _as_labels(values: Sequence[Any], n: int, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=object).reshape(-1)
    if len(out) != n:
        raise ValueError(f"{name} length must match encoder rows")
    return out


def _normalize_feature_groups(
    feature_group_indices: Mapping[str, Sequence[int]] | Sequence[Sequence[int]] | None,
    feature_count: int,
) -> dict[str, np.ndarray]:
    if feature_group_indices is None:
        return {}
    raw_items = (
        feature_group_indices.items()
        if isinstance(feature_group_indices, Mapping)
        else enumerate(feature_group_indices)
    )
    result: dict[str, np.ndarray] = {}
    for raw_name, raw_indices in raw_items:
        values = np.asarray(list(raw_indices), dtype=np.int64).reshape(-1)
        values = np.unique(values)
        if values.size and (values.min() < 0 or values.max() >= feature_count):
            raise ValueError(f"Feature group {raw_name!r} has out-of-range indices")
        if values.size:
            result[str(raw_name)] = values
    return result


def _protected_feature_indices(
    feature_groups: Mapping[str, np.ndarray],
    side_feature_group: str,
    side_feature_indices: Sequence[int] | None,
    feature_count: int,
) -> np.ndarray:
    protected: list[int] = (
        []
        if side_feature_indices is None
        else np.asarray(side_feature_indices, dtype=np.int64).reshape(-1).tolist()
    )
    requested = str(side_feature_group).strip().lower()
    for name, indices in feature_groups.items():
        normalized = str(name).strip().lower()
        if normalized == requested or normalized in {"side", "side_features"}:
            protected.extend(np.asarray(indices, dtype=np.int64).tolist())
    out = np.unique(np.asarray(protected, dtype=np.int64))
    if out.size and (out.min() < 0 or out.max() >= feature_count):
        raise ValueError("side_feature_indices has out-of-range indices")
    return out


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0).astype(np.float32)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    mad = np.nanmedian(np.abs(values - center), axis=0).astype(np.float32) * np.float32(1.4826)
    std = np.nanstd(values, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return center, scale


def _batches(n: int, batch_size: int, rng: np.random.Generator) -> Sequence[np.ndarray]:
    return [part for part in np.array_split(rng.permutation(n), max(1, int(np.ceil(n / max(1, batch_size))))) if len(part)]


def _fixed_batches(n: int, batch_size: int) -> Sequence[np.ndarray]:
    return [np.arange(start, min(n, start + max(1, batch_size))) for start in range(0, n, max(1, batch_size))]


def _nt_xent(left: Any, right: Any, temperature: float) -> Any:
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    logits = left @ right.T / max(float(temperature), 1e-4)
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _vicreg_loss(left: Any, right: Any, config: EncoderConfig) -> Any:
    invariance = F.mse_loss(left, right)
    variance = left.new_tensor(0.0)
    covariance = left.new_tensor(0.0)
    for value in (left, right):
        std = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
        variance = variance + F.relu(float(config.vicreg_gamma) - std).mean()
        centered = value - value.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, len(value) - 1)
        covariance = covariance + _off_diagonal(cov).square().mean()
    return invariance + float(config.vicreg_variance_weight) * variance + float(config.vicreg_covariance_weight) * covariance


def _off_diagonal(matrix: Any) -> Any:
    return matrix.flatten()[:-1].view(matrix.shape[0] - 1, matrix.shape[0] + 1)[:, 1:].flatten()


def _student_t_assignments(latent: Any, centers: Any, degrees_of_freedom: float) -> Any:
    df = float(degrees_of_freedom)
    if not np.isfinite(df) or df <= 0.0:
        raise ValueError("student_t_df must be positive")
    distance = torch.sum((latent[:, None, :] - centers[None, :, :]).square(), dim=2)
    q = torch.pow(1.0 + distance / df, -0.5 * (df + 1.0))
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _idec_target(q: Any) -> Any:
    weight = q.square() / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _vade_assignments(model: Any, mu: Any) -> Any:
    diff = mu[:, None, :] - model.cluster_means[None, :, :]
    log_prob = -0.5 * (model.cluster_logvars + diff.square() / torch.exp(model.cluster_logvars)).sum(dim=2)
    return torch.softmax(log_prob + torch.log_softmax(model.cluster_logits, dim=0), dim=1)


def _vade_kl(model: Any, mu: Any, logvar: Any) -> tuple[Any, Any]:
    q = _vade_assignments(model, mu)
    prior_logvar = model.cluster_logvars[None, :, :]
    delta = mu[:, None, :] - model.cluster_means[None, :, :]
    gaussian_kl = 0.5 * (
        prior_logvar - logvar[:, None, :] + (torch.exp(logvar[:, None, :]) + delta.square()) / torch.exp(prior_logvar) - 1.0
    ).sum(dim=2)
    categorical_kl = q * (torch.log(q.clamp_min(1e-8)) - torch.log_softmax(model.cluster_logits, dim=0))
    return (q * gaussian_kl + categorical_kl).sum(dim=1).mean(), q


def _occupancy_penalty(assignments: Any, minimum: float) -> Any:
    threshold = float(minimum)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("min_effective_occupancy must be a fraction in [0, 1]")
    if threshold == 0.0:
        return assignments.new_tensor(0.0)
    occupancy = assignments.mean(dim=0)
    return torch.relu(threshold - occupancy).square().mean()


def _initialization_mode(value: str, kind: str) -> str:
    mode = str(value or "auto").strip().lower().replace(" ", "_")
    if mode == "auto":
        return "kmeans++" if kind == "idec" else "random"
    aliases = {
        "kmeans_pp": "kmeans++",
        "incumbent": "incumbent_means",
        "incumbent_gmm_means": "incumbent_means",
    }
    mode = aliases.get(mode, mode)
    allowed = (
        {"kmeans++", "incumbent_means"}
        if kind == "idec"
        else {"random", "pretrained_dae_gmm", "pretrained_vae_gmm"}
    )
    if mode not in allowed:
        raise ValueError(f"Unsupported {kind} initialization: {value!r}")
    return mode


def _state_array(
    state: Mapping[str, Any] | None, names: Sequence[str], *, required: bool
) -> np.ndarray | None:
    if state is None:
        if required:
            raise ValueError("Initialization state is required for the selected initializer")
        return None
    candidates: list[Mapping[str, Any]] = [state]
    nested = state.get("torch_state")
    if isinstance(nested, Mapping):
        candidates.append(nested)
    for source in candidates:
        for name in names:
            if name in source:
                return np.asarray(source[name], dtype=np.float32)
    if required:
        raise ValueError(f"Initialization state is missing one of {list(names)}")
    return None


def _initial_cluster_means(
    state: Mapping[str, Any] | None, n_clusters: int, latent_dim: int
) -> np.ndarray:
    means = _state_array(
        state, ("cluster_centers", "cluster_means", "gmm_means", "means"), required=True
    )
    assert means is not None
    if means.shape != (n_clusters, latent_dim):
        raise ValueError("Incumbent cluster means shape does not match encoder configuration")
    return means.astype(np.float32)


def _initial_vade_parameters(
    state: Mapping[str, Any] | None,
    n_clusters: int,
    latent_dim: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = _initial_cluster_means(state, n_clusters, latent_dim)
    logvars = _state_array(
        state, ("cluster_logvars", "gmm_logvars", "logvars"), required=False
    )
    if logvars is None:
        covariance = _state_array(
            state, ("gmm_covariances", "covariances", "cluster_covariances"), required=False
        )
        if covariance is None:
            logvars = np.zeros((n_clusters, latent_dim), dtype=np.float32)
        else:
            covariance = np.asarray(covariance, dtype=np.float32)
            if covariance.ndim == 3:
                covariance = np.diagonal(covariance, axis1=1, axis2=2)
            logvars = np.log(np.clip(covariance, 1e-6, None))
    weights = _state_array(
        state, ("gmm_weights", "weights", "cluster_weights"), required=False
    )
    logits = _state_array(state, ("cluster_logits",), required=False)
    if logits is None:
        if weights is None:
            logits = np.zeros(n_clusters, dtype=np.float32)
        else:
            logits = np.log(np.clip(np.asarray(weights, dtype=np.float32), 1e-8, None))
    if logvars.shape != (n_clusters, latent_dim):
        raise ValueError(f"{mode} initialization has incompatible covariance shape")
    if np.asarray(logits).shape != (n_clusters,):
        raise ValueError(f"{mode} initialization has incompatible mixture-weight shape")
    return means, np.asarray(logvars, dtype=np.float32), np.asarray(logits, dtype=np.float32)


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("Torch-backed latent encoders require the optional PyTorch dependency")


def _seed_torch(seed: int) -> None:
    _require_torch()
    torch.manual_seed(int(seed))
    if torch.backends.mps.is_available():  # MPS has no deterministic-algorithm guarantee for every op.
        torch.mps.manual_seed(int(seed))


def _array_list(value: np.ndarray | None) -> list[float]:
    return [] if value is None else np.asarray(value, dtype=np.float32).astype(float).tolist()


def _torch_state_to_lists(state: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): tensor.detach().cpu().numpy().astype(np.float32).tolist() for name, tensor in state.items()}


def _load_torch_state_from_lists(model: Any, raw_state: Any, device: Any) -> None:
    if not isinstance(raw_state, Mapping):
        raise ValueError("Encoder state has no torch parameter mapping")
    expected = model.state_dict()
    rebuilt: dict[str, Any] = {}
    for name, template in expected.items():
        if name not in raw_state:
            raise ValueError(f"Encoder state is missing parameter {name}")
        value = torch.as_tensor(raw_state[name], dtype=template.dtype, device=device)
        if tuple(value.shape) != tuple(template.shape):
            raise ValueError(f"Encoder state parameter shape mismatch for {name}")
        rebuilt[name] = value
    model.load_state_dict(rebuilt, strict=True)
    model.to(device=device, dtype=torch.float32)


def _load_partial_torch_state_from_lists(
    model: Any,
    raw_state: Any,
    device: Any,
    *,
    excluded: Sequence[str] = (),
) -> None:
    """Restore a compatible torch subset, leaving candidate-specific heads fresh."""
    if not isinstance(raw_state, Mapping):
        raise ValueError("Encoder pretraining state has no torch parameter mapping")
    excluded_names = {str(name) for name in excluded}
    rebuilt: dict[str, Any] = {}
    for name, template in model.state_dict().items():
        if name in excluded_names or name not in raw_state:
            continue
        value = torch.as_tensor(raw_state[name], dtype=template.dtype, device=device)
        if tuple(value.shape) != tuple(template.shape):
            raise ValueError(f"Encoder pretraining parameter shape mismatch for {name}")
        rebuilt[name] = value
    if not rebuilt:
        raise ValueError("Encoder pretraining state has no compatible parameters")
    model.load_state_dict(rebuilt, strict=False)
    model.to(device=device, dtype=torch.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
