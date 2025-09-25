"""Neural architecture helpers backed by PyTorch/Keras."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from tensorflow import keras
except Exception:  # pragma: no cover
    keras = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class LayerSpec:
    """Specification for a single neural layer."""

    type: str
    units: Optional[int] = None
    activation: Optional[str] = None
    dropout: Optional[float] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    additional_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuralArchitecture:
    """A portable architecture description that can build torch or keras models."""

    input_shape: List[int]
    layers: List[LayerSpec]
    task_type: str = "classification"
    preferred_backend: Optional[str] = None

    def build(self, backend: Optional[str] = None):
        """Create a model instance using the requested backend."""
        target = (backend or self.preferred_backend or self._default_backend()).lower()
        if target == "torch":
            if torch is None:
                raise RuntimeError("PyTorch backend requested but torch is not available.")
            return self._build_torch()
        if target == "keras":
            if keras is None:
                raise RuntimeError("Keras backend requested but tensorflow/keras is not available.")
            return self._build_keras()
        raise ValueError(f"Unsupported backend '{target}'.")

    def parameter_count(self) -> int:
        try:
            model = self.build()
        except Exception:
            return 0
        if torch is not None and isinstance(model, nn.Module):
            return sum(param.numel() for param in model.parameters())
        if keras is not None:
            return model.count_params()
        return 0

    def _default_backend(self) -> str:
        if self.preferred_backend:
            return self.preferred_backend
        if keras is not None:
            return "keras"
        if torch is not None:
            return "torch"
        raise RuntimeError("Neither PyTorch nor TensorFlow/Keras is available.")

    # ------------------------------------------------------------------
    def _build_torch(self) -> "nn.Module":
        modules: List[nn.Module] = []
        in_features = self.input_shape[0]

        for layer in self.layers:
            if layer.type.lower() == "dense":
                modules.append(nn.Linear(in_features, layer.units or in_features))
                in_features = layer.units or in_features
                modules.append(self._torch_activation(layer.activation))
                if layer.dropout:
                    modules.append(nn.Dropout(layer.dropout))
            elif layer.type.lower() == "lstm":
                modules.append(
                    nn.LSTM(
                        input_size=in_features,
                        hidden_size=layer.units or in_features,
                        batch_first=True,
                    )
                )
                in_features = layer.units or in_features
            elif layer.type.lower() == "conv2d":
                modules.append(
                    nn.Conv2d(
                        in_channels=in_features,
                        out_channels=layer.units or in_features,
                        kernel_size=layer.kernel_size or 3,
                        stride=layer.stride or 1,
                    )
                )
                in_features = layer.units or in_features
                modules.append(self._torch_activation(layer.activation))
            elif layer.type.lower() == "attention":
                modules.append(
                    nn.MultiheadAttention(
                        embed_dim=in_features,
                        num_heads=layer.additional_args.get("num_heads", 4),
                        batch_first=True,
                    )
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer.type}")

        if self.task_type == "classification":
            modules.append(nn.Softmax(dim=-1))
        else:
            modules.append(nn.Identity())

        model = nn.Sequential(*[m for m in modules if m is not None])
        return model

    def _build_keras(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for layer in self.layers:
            if layer.type.lower() == "dense":
                x = keras.layers.Dense(layer.units or x.shape[-1], activation=layer.activation)(x)
                if layer.dropout:
                    x = keras.layers.Dropout(layer.dropout)(x)
            elif layer.type.lower() == "lstm":
                x = keras.layers.LSTM(layer.units or x.shape[-1], return_sequences=True)(x)
            elif layer.type.lower() == "conv2d":
                x = keras.layers.Conv2D(
                    filters=layer.units or x.shape[-1],
                    kernel_size=layer.kernel_size or 3,
                    strides=layer.stride or 1,
                    activation=layer.activation,
                )(x)
            elif layer.type.lower() == "attention":
                x = keras.layers.MultiHeadAttention(
                    num_heads=layer.additional_args.get("num_heads", 4),
                    key_dim=layer.additional_args.get("key_dim", x.shape[-1]),
                )(x, x)
            else:
                raise ValueError(f"Unsupported layer type: {layer.type}")
        if self.task_type == "classification":
            outputs = keras.layers.Activation("softmax")(x)
        else:
            outputs = x
        return keras.Model(inputs=inputs, outputs=outputs)

    def _torch_activation(self, name: Optional[str]):
        if name is None:
            return nn.Identity()
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        return nn.Identity()

    def validate(self, max_params: Optional[int] = None, allowed_layers: Optional[List[str]] = None) -> bool:
        allowed_layers = [layer.lower() for layer in allowed_layers] if allowed_layers else None
        for layer in self.layers:
            if allowed_layers and layer.type.lower() not in allowed_layers:
                logger.warning("Layer %s not in allowed set", layer.type)
                return False
        if max_params is not None:
            try:
                if self.parameter_count() > max_params:
                    return False
            except Exception as exc:  # pragma: no cover - safe guard
                logger.error("Failed to compute parameter count: %s", exc)
                return False
        return True
