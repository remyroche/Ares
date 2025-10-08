"""Utilities for deterministic random number generation across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict
import os
import random

try:  # pragma: no cover - optional dependency import guard
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is required but guard defensively
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency import guard
    import torch as _torch
except ImportError:  # pragma: no cover - torch optional
    _torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency import guard
    import tensorflow as _tf
except ImportError:  # pragma: no cover - tensorflow optional
    _tf = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SeededRNGs:
    """Container for seeded random number generators."""

    seed: int
    python: random.Random
    numpy: "_np.random.Generator"  # type: ignore[name-defined]
    torch: Optional["_torch.Generator"] = None  # type: ignore[name-defined]
    tensorflow: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a mapping representation for dependency injection helpers."""
        return {
            "seed": self.seed,
            "python": self.python,
            "numpy": self.numpy,
            "torch": self.torch,
            "tensorflow": self.tensorflow,
        }


def _seed_python(seed: int) -> random.Random:
    random.seed(seed)
    py_rng = random.Random(seed)
    return py_rng


def _seed_numpy(seed: int) -> "_np.random.Generator":  # type: ignore[name-defined]
    if _np is None:  # pragma: no cover - numpy should exist but guard defensively
        raise ImportError("numpy is required for deterministic seeding")
    _np.random.seed(seed)  # Legacy API compatibility for code using global state
    return _np.random.default_rng(seed)


def _seed_torch(seed: int) -> Optional["_torch.Generator"]:  # type: ignore[name-defined]
    if _torch is None:
        return None
    _torch.manual_seed(seed)
    if hasattr(_torch, "cuda") and callable(getattr(_torch.cuda, "manual_seed_all", None)):
        try:  # pragma: no cover - GPU not available in tests
            _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    generator = _torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_tensorflow(seed: int) -> Optional[Any]:
    if _tf is None:
        return None
    try:  # pragma: no cover - TensorFlow rarely present in tests
        _tf.random.set_seed(seed)
        return _tf.random.Generator.from_seed(seed)
    except Exception:
        _tf.random.set_seed(seed)
        return None


def seed_rngs(seed: int) -> SeededRNGs:
    """Seed all supported RNG backends and return the seeded instances.

    Args:
        seed: Seed value applied across Python, NumPy, and supported ML frameworks.

    Returns:
        SeededRNGs dataclass containing seeded RNG handles.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    python_rng = _seed_python(seed)
    numpy_rng = _seed_numpy(seed)
    torch_rng = _seed_torch(seed)
    tensorflow_rng = _seed_tensorflow(seed)

    return SeededRNGs(
        seed=seed,
        python=python_rng,
        numpy=numpy_rng,
        torch=torch_rng,
        tensorflow=tensorflow_rng,
    )
