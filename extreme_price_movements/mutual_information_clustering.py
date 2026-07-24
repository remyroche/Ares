"""Frozen-embedding mutual-information clustering heads.

This module implements the assignment part of invariant information
clustering (IIC) for already computed tabular embeddings.  It deliberately
does not own an encoder or create augmentations: callers provide aligned
``weak/weak`` or ``weak/strong`` embedding pairs and this module trains only
the small assignment heads on those frozen values.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CLUSTER_COUNTS: tuple[int, ...] = (4, 6, 8, 12)
OVERCLUSTER_COUNTS: tuple[int, ...] = (12, 24, 48)
_EPSILON = 1e-8


def torch_available() -> bool:
    """Return whether the optional PyTorch dependency can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class MutualInformationClusteringConfig:
    """Configuration for frozen-embedding IIC assignment heads.

    ``cluster_counts`` and ``overcluster_counts`` are intentionally limited
    to the small, pre-agreed search spaces.  An empty ``overcluster_counts``
    disables overclustering.  Reconstruction is only meaningful when a
    shared bottleneck is present, because embeddings themselves are frozen.
    """

    cluster_counts: tuple[int, ...] = CLUSTER_COUNTS
    overcluster_counts: tuple[int, ...] = ()
    shared_bottleneck_dim: int | None = None
    mutual_information_weight: float = 1.0
    marginal_balance_weight: float = 1.0
    reconstruction_weight: float = 0.0
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    random_state: int = 20260718
    device: str = "auto"

    def __post_init__(self) -> None:
        _validate_counts(self.cluster_counts, CLUSTER_COUNTS, "cluster_counts")
        _validate_counts(
            self.overcluster_counts, OVERCLUSTER_COUNTS, "overcluster_counts"
        )
        if self.shared_bottleneck_dim not in (None, 8, 16):
            raise ValueError("shared_bottleneck_dim must be None, 8, or 16")
        if self.reconstruction_weight > 0.0 and self.shared_bottleneck_dim is None:
            raise ValueError(
                "reconstruction_weight requires shared_bottleneck_dim to be 8 or 16"
            )
        for name in (
            "mutual_information_weight",
            "marginal_balance_weight",
            "reconstruction_weight",
            "learning_rate",
            "weight_decay",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.epochs) < 1 or int(self.batch_size) < 1:
            raise ValueError("epochs and batch_size must both be positive")
        if str(self.device).lower() not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', or 'mps'")


def _validate_counts(
    values: Sequence[int], allowed: Sequence[int], name: str
) -> None:
    selected = tuple(int(value) for value in values)
    if not selected:
        if name == "cluster_counts":
            raise ValueError("cluster_counts must contain at least one value")
        return
    if len(set(selected)) != len(selected):
        raise ValueError(f"{name} must not contain duplicates")
    invalid = sorted(set(selected).difference(allowed))
    if invalid:
        raise ValueError(f"{name} contains unsupported values: {invalid}")


def _as_embeddings(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional embedding matrix")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _head_name(kind: str, count: int) -> str:
    return f"{kind}_{int(count)}"


class FrozenEmbeddingMutualInformationClustering:
    """Train and apply deterministic IIC heads to frozen tabular embeddings."""

    schema_version = "frozen_embedding_iic_v1"

    def __init__(
        self, config: MutualInformationClusteringConfig | None = None
    ) -> None:
        self.config = config or MutualInformationClusteringConfig()
        self.input_dim_: int | None = None
        self.model_state_: dict[str, np.ndarray] | None = None
        self.device_: str | None = None
        self.training_report_: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.input_dim_ is not None and self.model_state_ is not None

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(
            [_head_name("cluster", count) for count in self.config.cluster_counts]
            + [
                _head_name("overcluster", count)
                for count in self.config.overcluster_counts
            ]
        )

    def fit(
        self,
        weak_embeddings: Any,
        paired_embeddings: Any,
    ) -> "FrozenEmbeddingMutualInformationClustering":
        """Fit heads from caller-supplied aligned augmentation pairs.

        ``paired_embeddings`` may be another weak augmentation or a strong
        augmentation.  No augmentation is generated internally and neither
        input is modified.
        """
        torch = _require_torch()
        weak = _as_embeddings(weak_embeddings, "weak_embeddings")
        paired = _as_embeddings(paired_embeddings, "paired_embeddings")
        if weak.shape != paired.shape:
            raise ValueError(
                "weak_embeddings and paired_embeddings must have identical shapes"
            )
        if len(weak) < 2:
            raise ValueError("at least two aligned embedding rows are required")

        device = _resolve_device(torch, self.config.device)
        _seed_everything(torch, self.config.random_state)
        model = _build_network(torch, self.config, weak.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        rng = np.random.default_rng(int(self.config.random_state))
        weak_tensor = torch.as_tensor(weak, dtype=torch.float32, device=device)
        paired_tensor = torch.as_tensor(paired, dtype=torch.float32, device=device)
        batch_size = min(max(2, int(self.config.batch_size)), len(weak))
        history: list[dict[str, float]] = []

        for epoch in range(int(self.config.epochs)):
            model.train()
            order = rng.permutation(len(weak))
            losses: list[float] = []
            mi_values: list[float] = []
            balance_values: list[float] = []
            reconstruction_values: list[float] = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                # A final singleton cannot form a stable batch joint matrix.
                if len(indices) < 2:
                    continue
                index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
                weak_batch = weak_tensor.index_select(0, index_tensor)
                paired_batch = paired_tensor.index_select(0, index_tensor)
                weak_logits, weak_reconstruction = model(weak_batch)
                paired_logits, paired_reconstruction = model(paired_batch)
                loss, mi, balance, reconstruction = _iic_loss(
                    torch,
                    weak_logits,
                    paired_logits,
                    weak_batch,
                    paired_batch,
                    weak_reconstruction,
                    paired_reconstruction,
                    self.config,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                mi_values.append(float(mi.detach().cpu()))
                balance_values.append(float(balance.detach().cpu()))
                reconstruction_values.append(float(reconstruction.detach().cpu()))
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(np.mean(losses)) if losses else float("nan"),
                    "mutual_information": float(np.mean(mi_values))
                    if mi_values
                    else float("nan"),
                    "marginal_balance": float(np.mean(balance_values))
                    if balance_values
                    else float("nan"),
                    "reconstruction": float(np.mean(reconstruction_values))
                    if reconstruction_values
                    else float("nan"),
                }
            )

        self.input_dim_ = int(weak.shape[1])
        self.device_ = str(device)
        self.model_state_ = {
            name: value.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, value in model.state_dict().items()
        }
        final_diagnostics = self.diagnostics(weak, paired)
        self.training_report_ = {
            "schema_version": self.schema_version,
            "input_dim": int(self.input_dim_),
            "rows": int(len(weak)),
            "device": self.device_,
            "epochs": int(self.config.epochs),
            "history_tail": history[-10:],
            "heads": {
                name: {
                    "mutual_information": float(values["mutual_information"]),
                    "conditional_entropy": float(values["conditional_entropy"]),
                    "occupancy": np.asarray(values["occupancy"], dtype=np.float32)
                    .astype(float)
                    .tolist(),
                }
                for name, values in final_diagnostics.items()
            },
        }
        return self

    def fit_pairs(
        self, weak_embeddings: Any, paired_embeddings: Any
    ) -> "FrozenEmbeddingMutualInformationClustering":
        """Alias for :meth:`fit` that makes augmentation-pair ownership clear."""
        return self.fit(weak_embeddings, paired_embeddings)

    def predict_proba(self, embeddings: Any) -> dict[str, np.ndarray]:
        """Return float32 assignment probabilities for every configured head."""
        outputs = self._forward(embeddings)
        return {
            name: _softmax_numpy(logits).astype(np.float32, copy=False)
            for name, logits in outputs.items()
        }

    def predict(self, embeddings: Any) -> dict[str, np.ndarray]:
        """Return integer maximum-probability assignments for every head."""
        return {
            name: probabilities.argmax(axis=1).astype(np.int64, copy=False)
            for name, probabilities in self.predict_proba(embeddings).items()
        }

    def diagnostics(
        self,
        embeddings: Any,
        paired_embeddings: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Emit assignments and pair-aware clustering diagnostics per head.

        When no paired embeddings are supplied, pair-specific outputs are NaN
        per row while occupancy remains available from ``embeddings``.
        """
        embeddings_array = _as_embeddings(embeddings, "embeddings")
        primary = self.predict_proba(embeddings_array)
        paired: dict[str, np.ndarray] | None = None
        if paired_embeddings is not None:
            paired_array = _as_embeddings(paired_embeddings, "paired_embeddings")
            if paired_array.shape != embeddings_array.shape:
                raise ValueError("paired_embeddings must match embeddings in shape")
            paired = self.predict_proba(paired_array)

        result: dict[str, dict[str, Any]] = {}
        for name, probabilities in primary.items():
            assignments = probabilities.argmax(axis=1).astype(np.int64, copy=False)
            entropy = -np.sum(
                probabilities * np.log(np.clip(probabilities, _EPSILON, 1.0)), axis=1
            )
            normalized_entropy = entropy / np.log(float(probabilities.shape[1]))
            ordered = np.partition(probabilities, -2, axis=1)
            margin = ordered[:, -1] - ordered[:, -2]
            occupancy = probabilities.mean(axis=0).astype(np.float32, copy=False)
            if paired is None:
                consistency = np.full(len(probabilities), np.nan, dtype=np.float32)
                conditional_entropy = float("nan")
                mutual_information = float("nan")
            else:
                pair_probabilities = paired[name]
                consistency = np.sum(
                    probabilities * pair_probabilities, axis=1
                ).astype(np.float32, copy=False)
                joint = _joint_numpy(probabilities, pair_probabilities)
                conditional_entropy = _conditional_entropy_numpy(joint)
                mutual_information = _mutual_information_numpy(joint)
            result[name] = {
                "probabilities": probabilities.astype(np.float32, copy=False),
                "assignments": assignments,
                "normalized_entropy": normalized_entropy.astype(np.float32, copy=False),
                "margin": margin.astype(np.float32, copy=False),
                "augmentation_consistency": consistency,
                "occupancy": occupancy,
                "conditional_entropy": conditional_entropy,
                "mutual_information": mutual_information,
            }
        return result

    def transform(
        self,
        embeddings: Any,
        paired_embeddings: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Alias for :meth:`diagnostics` for transformer-style call sites."""
        return self.diagnostics(embeddings, paired_embeddings)

    def fit_predict(
        self, weak_embeddings: Any, paired_embeddings: Any
    ) -> dict[str, dict[str, Any]]:
        """Fit on a caller-provided pair and return its diagnostics."""
        return self.fit(weak_embeddings, paired_embeddings).diagnostics(
            weak_embeddings, paired_embeddings
        )

    def to_state(self) -> dict[str, Any]:
        """Return a portable, NumPy-only artifact state."""
        if not self.is_fitted:
            raise RuntimeError("Mutual-information clustering heads are not fitted")
        return {
            "schema_version": self.schema_version,
            "config": asdict(self.config),
            "input_dim": int(self.input_dim_ or 0),
            "device": self.device_,
            "model_state": {
                name: np.asarray(value, dtype=np.float32).copy()
                for name, value in (self.model_state_ or {}).items()
            },
            "training_report": self.training_report_,
        }

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any]
    ) -> "FrozenEmbeddingMutualInformationClustering":
        """Restore an artifact produced by :meth:`to_state` or :meth:`save`."""
        if str(state.get("schema_version")) != cls.schema_version:
            raise ValueError("Unsupported mutual-information clustering state")
        config_payload = dict(state.get("config", {}) or {})
        for name in ("cluster_counts", "overcluster_counts"):
            if name in config_payload:
                config_payload[name] = tuple(config_payload[name])
        model = cls(MutualInformationClusteringConfig(**config_payload))
        input_dim = int(state.get("input_dim", 0) or 0)
        raw_state = dict(state.get("model_state", {}) or {})
        if input_dim < 1 or not raw_state:
            raise ValueError("State does not contain fitted assignment-head weights")
        model.input_dim_ = input_dim
        model.device_ = str(state.get("device") or "cpu")
        model.model_state_ = {
            str(name): np.asarray(value, dtype=np.float32).copy()
            for name, value in raw_state.items()
        }
        model.training_report_ = dict(state.get("training_report", {}) or {})
        return model

    def save(self, path: str | Path) -> None:
        """Serialize a portable model artifact without requiring torch on load."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(self.to_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenEmbeddingMutualInformationClustering":
        """Load an artifact written by :meth:`save`."""
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, Mapping):
            raise ValueError("Invalid mutual-information clustering artifact")
        return cls.from_state(state)

    def _forward(self, embeddings: Any) -> dict[str, np.ndarray]:
        torch = _require_torch()
        if not self.is_fitted:
            raise RuntimeError("Mutual-information clustering heads are not fitted")
        values = _as_embeddings(embeddings, "embeddings")
        if values.shape[1] != self.input_dim_:
            raise ValueError(
                f"embeddings have {values.shape[1]} columns; expected {self.input_dim_}"
            )
        device = _resolve_device(torch, self.config.device)
        network = _build_network(torch, self.config, int(self.input_dim_)).to(device)
        network.load_state_dict(
            {
                name: torch.as_tensor(value, dtype=torch.float32, device=device)
                for name, value in (self.model_state_ or {}).items()
            },
            strict=True,
        )
        network.eval()
        outputs: dict[str, list[np.ndarray]] = {name: [] for name in self.head_names}
        with torch.no_grad():
            for start in range(0, len(values), 4096):
                batch = torch.as_tensor(
                    values[start : start + 4096], dtype=torch.float32, device=device
                )
                logits, _ = network(batch)
                for name, value in logits.items():
                    outputs[name].append(
                        value.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
        return {
            name: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for name, parts in outputs.items()
        }


# A concise alias for call sites that prefer the IIC term.
FrozenEmbeddingIIC = FrozenEmbeddingMutualInformationClustering
MutualInformationClustering = FrozenEmbeddingMutualInformationClustering


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional environment.
        raise RuntimeError(
            "PyTorch is required to fit or apply mutual-information clustering heads"
        ) from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> Any:
    wants_mps = str(requested).lower() in {"auto", "mps"}
    mps_available = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if wants_mps and mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(torch: Any, random_state: int) -> None:
    random.seed(int(random_state))
    np.random.seed(int(random_state))
    torch.manual_seed(int(random_state))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(int(random_state))
        except Exception:
            pass


def _build_network(torch: Any, config: MutualInformationClusteringConfig, input_dim: int) -> Any:
    nn = torch.nn

    class _AssignmentNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if config.shared_bottleneck_dim is None:
                self.bottleneck = None
                representation_dim = input_dim
            else:
                self.bottleneck = nn.Sequential(
                    nn.Linear(input_dim, int(config.shared_bottleneck_dim)), nn.GELU()
                )
                representation_dim = int(config.shared_bottleneck_dim)
            self.heads = nn.ModuleDict(
                {
                    _head_name("cluster", count): nn.Linear(representation_dim, int(count))
                    for count in config.cluster_counts
                }
                | {
                    _head_name("overcluster", count): nn.Linear(
                        representation_dim, int(count)
                    )
                    for count in config.overcluster_counts
                }
            )
            self.decoder = (
                nn.Linear(representation_dim, input_dim)
                if config.reconstruction_weight > 0.0
                else None
            )

        def forward(self, values: Any) -> tuple[dict[str, Any], Any | None]:
            representation = (
                values if self.bottleneck is None else self.bottleneck(values)
            )
            reconstruction = (
                None if self.decoder is None else self.decoder(representation)
            )
            return (
                {name: head(representation) for name, head in self.heads.items()},
                reconstruction,
            )

    return _AssignmentNetwork()


def _iic_loss(
    torch: Any,
    weak_logits: Mapping[str, Any],
    paired_logits: Mapping[str, Any],
    weak_embeddings: Any,
    paired_embeddings: Any,
    weak_reconstruction: Any | None,
    paired_reconstruction: Any | None,
    config: MutualInformationClusteringConfig,
) -> tuple[Any, Any, Any, Any]:
    total_mi = torch.zeros((), dtype=torch.float32, device=weak_embeddings.device)
    total_balance = torch.zeros_like(total_mi)
    for name, logits in weak_logits.items():
        weak_probabilities = torch.softmax(logits, dim=1)
        paired_probabilities = torch.softmax(paired_logits[name], dim=1)
        joint = weak_probabilities.transpose(0, 1) @ paired_probabilities
        joint = joint / float(weak_probabilities.shape[0])
        joint = 0.5 * (joint + joint.transpose(0, 1))
        joint = torch.clamp(joint, min=_EPSILON)
        joint = joint / joint.sum()
        marginal_left = torch.clamp(joint.sum(dim=1), min=_EPSILON)
        marginal_right = torch.clamp(joint.sum(dim=0), min=_EPSILON)
        total_mi = total_mi + torch.sum(
            joint
            * (
                torch.log(joint)
                - torch.log(marginal_left).unsqueeze(1)
                - torch.log(marginal_right).unsqueeze(0)
            )
        )
        cluster_count = float(joint.shape[0])
        total_balance = total_balance + 0.5 * (
            torch.sum(marginal_left * torch.log(marginal_left * cluster_count))
            + torch.sum(marginal_right * torch.log(marginal_right * cluster_count))
        )
    head_count = float(max(len(weak_logits), 1))
    mean_mi = total_mi / head_count
    mean_balance = total_balance / head_count
    reconstruction = torch.zeros_like(mean_mi)
    if weak_reconstruction is not None and paired_reconstruction is not None:
        reconstruction = 0.5 * (
            torch.mean((weak_reconstruction - weak_embeddings) ** 2)
            + torch.mean((paired_reconstruction - paired_embeddings) ** 2)
        )
    loss = (
        -float(config.mutual_information_weight) * mean_mi
        + float(config.marginal_balance_weight) * mean_balance
        + float(config.reconstruction_weight) * reconstruction
    )
    return loss, mean_mi, mean_balance, reconstruction


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(centered, dtype=np.float32)
    return values / values.sum(axis=1, keepdims=True)


def _joint_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    joint = left.T @ right / float(len(left))
    joint = 0.5 * (joint + joint.T)
    joint = np.clip(joint, _EPSILON, None)
    return joint / joint.sum()


def _conditional_entropy_numpy(joint: np.ndarray) -> float:
    marginal_left = np.clip(joint.sum(axis=1, keepdims=True), _EPSILON, None)
    return float(-np.sum(joint * (np.log(joint) - np.log(marginal_left))))


def _mutual_information_numpy(joint: np.ndarray) -> float:
    left = np.clip(joint.sum(axis=1, keepdims=True), _EPSILON, None)
    right = np.clip(joint.sum(axis=0, keepdims=True), _EPSILON, None)
    return float(np.sum(joint * (np.log(joint) - np.log(left) - np.log(right))))
