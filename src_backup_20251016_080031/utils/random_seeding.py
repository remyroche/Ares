"""Utilities for deterministic random number generation across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Callable
import os
import random

try:  # pragma: no cover - optional dependency import guard
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is required but guard defensively
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency import guard
    import pandas as _pd
except ImportError:  # pragma: no cover - pandas optional in some environments
    _pd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency import guard
    import torch as _torch
except ImportError:  # pragma: no cover - torch optional
    _torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency import guard
    import tensorflow as _tf
except ImportError:  # pragma: no cover - tensorflow optional
    _tf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency import guard
    from sklearn.utils import check_random_state as _sk_check_random_state
except ImportError:  # pragma: no cover - sklearn optional
    _sk_check_random_state = None  # type: ignore[assignment]


_PANDAS_RANDOM_STATE_ORIGINAL: Optional[Callable[..., Any]] = None
_SKLEARN_RANDOM_STATE_ORIGINAL: Optional[Callable[..., Any]] = None


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


def _configure_pandas_randomness(numpy_rng: "_np.random.Generator") -> None:  # type: ignore[name-defined]
    """Patch pandas helpers so ``random_state=None`` reuses the seeded RNG."""

    global _PANDAS_RANDOM_STATE_ORIGINAL

    if _pd is None:  # pragma: no cover - pandas optional dependency
        return

    try:
        from pandas.core import common as _pd_common  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - extremely defensive
        return

    original = _PANDAS_RANDOM_STATE_ORIGINAL or getattr(_pd_common, "random_state", None)
    if original is None:  # pragma: no cover - pandas API change
        return

    if _PANDAS_RANDOM_STATE_ORIGINAL is None:
        _PANDAS_RANDOM_STATE_ORIGINAL = original

    def _random_state(state: Any = None, *, _orig: Callable[..., Any] = _PANDAS_RANDOM_STATE_ORIGINAL) -> Any:
        if state is None:
            return numpy_rng
        return _orig(state)

    setattr(_random_state, "__ares_seed_wrapper__", True)  # type: ignore[attr-defined]
    _pd_common.random_state = _random_state  # type: ignore[assignment]

    # Some pandas utilities cache shared RNGs under private attributes. Best effort.
    for attr in ("_shared_random_state", "_shared_random_state_gen"):
        if hasattr(_pd_common, attr):
            try:
                setattr(_pd_common, attr, numpy_rng)
            except Exception:  # pragma: no cover - attribute may be read-only
                pass


def _configure_sklearn_randomness(seed: int) -> None:
    """Ensure scikit-learn helpers use the deterministic RNG when available."""

    global _SKLEARN_RANDOM_STATE_ORIGINAL

    if _sk_check_random_state is None or _np is None:  # pragma: no cover - optional deps
        return

    if _SKLEARN_RANDOM_STATE_ORIGINAL is None:
        _SKLEARN_RANDOM_STATE_ORIGINAL = _sk_check_random_state

    original = _SKLEARN_RANDOM_STATE_ORIGINAL

    def _wrapped(seed_like: Any) -> Any:
        if seed_like is None or seed_like is _np.random:
            return _np.random.RandomState(seed)
        return original(seed_like)

    setattr(_wrapped, "__ares_seed_wrapper__", True)  # type: ignore[attr-defined]

    try:
        import sklearn.utils as _sk_utils  # type: ignore[import-not-found]

        _sk_utils.check_random_state = _wrapped  # type: ignore[assignment]
    except Exception:  # pragma: no cover - extremely defensive
        return


def set_global_seed(seed: int) -> SeededRNGs:
    """Seed every supported backend and apply framework specific patches."""

    seeded = seed_rngs(seed)

    try:
        _configure_pandas_randomness(seeded.numpy)
    except Exception:  # pragma: no cover - defensive best effort
        pass

    try:
        _configure_sklearn_randomness(seed)
    except Exception:  # pragma: no cover - defensive best effort
        pass

    return seeded
