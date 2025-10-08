"""Runtime configuration for the pre-training pipeline.

This module centralises all filesystem and resource configuration for the
pre-training steps.  It exposes a :func:`get_pre_training_settings` helper that
returns a frozen dataclass describing the resolved configuration.  Values are
loaded from environment variables using :mod:`pydantic-settings` when available
and fall back to a lightweight manual parser otherwise.  All paths are resolved
relative to the repository root so callers can rely on absolute paths without
hard-coding developer-specific directories.

Environment variable prefix: ``ARES_PRETRAINING_`` with ``__`` as the nested
delimiter.  For example::

    export ARES_PRETRAINING_DATA__ROOT=/mnt/training-data
    export ARES_PRETRAINING_DATA__CACHE_DIR=/tmp/ares-cache
    export ARES_PRETRAINING_REGIME__CACHE_DIR=/srv/regime-cache

When no environment overrides are provided sane project-relative defaults are
used (``historical_data`` for raw data, ``data_cache`` for cache files, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, get_type_hints

# Import common utilities for enhanced path handling and validation
from src.utils.common_operations import ensure_directory, safe_file_exists, validate_finite, validate_positive
from src.utils.tprint import tprint_debug, tprint_info

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# ``pydantic`` is optional inside the execution environment used by the tests.
# We prefer to use it when available for robust environment parsing but fall
# back to a small manual loader otherwise.
try:  # pragma: no cover - executed when pydantic is available
    from pydantic import BaseModel, Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - exercised via fallback path in tests
    BaseModel = object  # type: ignore[misc,assignment]

    class BaseSettings:  # type: ignore[override]
        """Fallback shim with a minimal API compatible with pydantic and runtime type validation."""

        model_config: Dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            # Perform runtime type validation when pydantic is not available
            if hasattr(self.__class__, '__annotations__'):
                type_hints = get_type_hints(self.__class__)
                for key, value in kwargs.items():
                    if key in type_hints:
                        expected_type = type_hints[key]
                        if not self._validate_type(value, expected_type):
                            raise TypeError(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")

            for key, value in kwargs.items():
                setattr(self, key, value)

        def _validate_type(self, value: Any, expected_type: Type) -> bool:
            """Basic runtime type validation for fallback mode."""
            if expected_type is Any:
                return True

            # Handle Optional types
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                # Check if None is allowed in Union
                if type(None) in expected_type.__args__:
                    if value is None:
                        return True
                    # Remove None from args for further validation
                    non_none_args = tuple(arg for arg in expected_type.__args__ if arg is not type(None))
                    if len(non_none_args) == 1:
                        return self._validate_type(value, non_none_args[0])

            # Handle basic types
            if expected_type in (str, int, float, bool):
                return isinstance(value, expected_type)

            # Handle Path
            if expected_type is Path:
                return isinstance(value, (str, Path))

            # Handle Dict
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is dict:
                return isinstance(value, dict)

            # Handle List
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
                return isinstance(value, list)

            # For complex types or when validation fails, allow but log warning
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Type validation not fully implemented for {expected_type}, allowing value")
            return True

    def Field(  # type: ignore[override]
        default: Any = None, *, env: Optional[str] = None, alias: Optional[str] = None, **_: Any
    ) -> Any:
        # The fallback simply returns the provided default. Environment parsing
        # is handled manually when pydantic is not installed.
        return default

    def SettingsConfigDict(**kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        return dict(kwargs)

    _PYDANTIC_AVAILABLE = False


@dataclass(frozen=True)
class ResolvedPath:
    """A filesystem path with the original (raw) value preserved."""

    raw: str
    resolved: Path

    def as_dict(self) -> Dict[str, str]:
        return {"raw": self.raw, "resolved": str(self.resolved)}


@dataclass(frozen=True)
class PreTrainingDataPaths:
    """Resolved directories used by the pre-training pipeline."""

    root: ResolvedPath
    cache_dir: ResolvedPath
    artifacts_dir: ResolvedPath
    generated_dir: ResolvedPath
    config_dir: ResolvedPath
    outcomes_dir: ResolvedPath


@dataclass(frozen=True)
class PreTrainingRegimeResources:
    """Resolved regime resource configuration."""

    dataset_path: Optional[ResolvedPath]
    cache_dir: Optional[ResolvedPath]


@dataclass(frozen=True)
class PreTrainingMetricsSettings:
    """Configuration for metrics output."""

    output_dir: Optional[ResolvedPath]
    filename: str
    format: str


@dataclass(frozen=True)
class PreTrainingSettings:
    """Container for resolved pre-training configuration."""

    data: PreTrainingDataPaths
    regime: PreTrainingRegimeResources
    metrics: PreTrainingMetricsSettings
    extras: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the effective configuration."""

        def _maybe(path: Optional[ResolvedPath]) -> Optional[Dict[str, str]]:
            return path.as_dict() if path is not None else None

        return {
            "data": {
                "root": self.data.root.as_dict(),
                "cache_dir": self.data.cache_dir.as_dict(),
                "artifacts_dir": self.data.artifacts_dir.as_dict(),
                "generated_dir": self.data.generated_dir.as_dict(),
                "config_dir": self.data.config_dir.as_dict(),
                "outcomes_dir": self.data.outcomes_dir.as_dict(),
            },
            "regime": {
                "dataset_path": _maybe(self.regime.dataset_path),
                "cache_dir": _maybe(self.regime.cache_dir),
            },
            "metrics": {
                "output_dir": _maybe(self.metrics.output_dir),
                "filename": self.metrics.filename,
                "format": self.metrics.format,
            },
            "extras": dict(self.extras),
        }

    @property
    def data_root(self) -> Path:
        return self.data.root.resolved

    @property
    def cache_root(self) -> Path:
        return self.data.cache_dir.resolved

    @property
    def artifacts_root(self) -> Path:
        return self.data.artifacts_dir.resolved

    @property
    def generated_root(self) -> Path:
        return self.data.generated_dir.resolved

    @property
    def config_root(self) -> Path:
        return self.data.config_dir.resolved

    @property
    def outcomes_root(self) -> Path:
        return self.data.outcomes_dir.resolved

    def to_data_locator_config(self):  # type: ignore[override]
        """Create a :class:`~src.training.config.data_locator.DataLocatorConfig`."""

        from src.training.config.data_locator import DataLocatorConfig

        return DataLocatorConfig(
            base_data_dir=str(self.data_root),
            base_cache_dir=str(self.cache_root),
            base_artifacts_dir=str(self.artifacts_root),
            base_generated_dir=str(self.generated_root),
            base_config_dir=str(self.config_root),
        )


def _resolve_path(value: Any) -> Optional[ResolvedPath]:
    if value is None:
        return None
    raw = str(value)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)

    # Validate path exists or parent directories exist for creation
    if not path.exists():
        if not path.parent.exists():
            raise ValueError(
                f"Pre-training configuration path does not exist and parent directory cannot be created: {raw!r} "
                f"(resolved to: {path!r}). Please ensure the parent directory exists or the path is accessible."
            )
        # Path doesn't exist but parent exists - this is OK for output directories that will be created

    return ResolvedPath(raw=raw, resolved=path)


def _ensure_resolved(value: Any, *, name: str) -> ResolvedPath:
    resolved = _resolve_path(value)
    if resolved is None:
        raise ValueError(f"Pre-training configuration '{name}' must not be None")
    return resolved


def _load_with_pydantic() -> Dict[str, Any]:  # pragma: no cover - executed when available
    class _DataPathsModel(BaseModel):
        root: Path = Field(default=Path("historical_data"))
        cache_dir: Path = Field(default=Path("data_cache"))
        artifacts_dir: Path = Field(default=Path("artifacts"))
        generated_dir: Path = Field(default=Path("generated"))
        config_dir: Path = Field(default=Path("src/config"))
        outcomes_dir: Path = Field(default=Path("outcomes"))

    class _RegimeModel(BaseModel):
        dataset_path: Optional[Path] = Field(default=None)
        cache_dir: Optional[Path] = Field(default=None)

    class _MetricsModel(BaseModel):
        output_dir: Optional[Path] = Field(default=None)
        filename: str = Field(default="pre_training_metrics")
        format: str = Field(default="csv")

    class _SettingsModel(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="ARES_PRETRAINING_", env_nested_delimiter="__")

        data: _DataPathsModel = Field(default_factory=_DataPathsModel)
        regime: _RegimeModel = Field(default_factory=_RegimeModel)
        metrics: _MetricsModel = Field(default_factory=_MetricsModel)
        extras: Dict[str, Any] = Field(default_factory=dict)

    model = _SettingsModel()
    return {
        "data": model.data.model_dump(),
        "regime": model.regime.model_dump(),
        "metrics": model.metrics.model_dump(),
        "extras": dict(getattr(model, "extras", {})),
    }


def _load_without_pydantic() -> Dict[str, Any]:  # pragma: no cover - exercised in tests
    def _env(key: str, default: Optional[str]) -> Optional[str]:
        value = os.getenv(key)
        if value is None:
            return default
        value = value.strip()
        return value or None

    data = {
        "root": _env("ARES_PRETRAINING_DATA__ROOT", "historical_data"),
        "cache_dir": _env("ARES_PRETRAINING_DATA__CACHE_DIR", "data_cache"),
        "artifacts_dir": _env("ARES_PRETRAINING_DATA__ARTIFACTS_DIR", "artifacts"),
        "generated_dir": _env("ARES_PRETRAINING_DATA__GENERATED_DIR", "generated"),
        "config_dir": _env("ARES_PRETRAINING_DATA__CONFIG_DIR", "src/config"),
        "outcomes_dir": _env("ARES_PRETRAINING_DATA__OUTCOMES_DIR", "outcomes"),
    }

    regime = {
        "dataset_path": _env("ARES_PRETRAINING_REGIME__DATASET_PATH", None),
        "cache_dir": _env("ARES_PRETRAINING_REGIME__CACHE_DIR", None),
    }

    metrics = {
        "output_dir": _env("ARES_PRETRAINING_METRICS__OUTPUT_DIR", None),
        "filename": _env("ARES_PRETRAINING_METRICS__FILENAME", "pre_training_metrics")
        or "pre_training_metrics",
        "format": _env("ARES_PRETRAINING_METRICS__FORMAT", "csv") or "csv",
    }

    extras: Dict[str, Any] = {}
    return {"data": data, "regime": regime, "metrics": metrics, "extras": extras}


def _load_raw_settings() -> Dict[str, Any]:
    """Load raw settings with enhanced logging."""
    tprint_debug("🔧 Loading pre-training settings from environment")

    if _PYDANTIC_AVAILABLE:
        tprint_debug("📋 Using Pydantic settings loader")
        settings = _load_with_pydantic()
    else:
        tprint_debug("📋 Using fallback settings loader")
        settings = _load_without_pydantic()

    tprint_info(f"✅ Loaded {len(settings)} top-level configuration sections")
    return settings


def _build_settings(raw: Dict[str, Any]) -> PreTrainingSettings:
    """Build settings object with enhanced logging."""
    tprint_debug(f"🏗️ Building pre-training settings from {len(raw)} sections")

    data = raw.get("data", {})
    regime = raw.get("regime", {})
    metrics = raw.get("metrics", {})

    tprint_debug(f"📊 Configuration sections: data={len(data)} items, regime={len(regime)} items, metrics={len(metrics)} items")

    data_paths = PreTrainingDataPaths(
        root=_ensure_resolved(data.get("root"), name="data.root"),
        cache_dir=_ensure_resolved(data.get("cache_dir"), name="data.cache_dir"),
        artifacts_dir=_ensure_resolved(data.get("artifacts_dir"), name="data.artifacts_dir"),
        generated_dir=_ensure_resolved(data.get("generated_dir"), name="data.generated_dir"),
        config_dir=_ensure_resolved(data.get("config_dir"), name="data.config_dir"),
        outcomes_dir=_ensure_resolved(data.get("outcomes_dir"), name="data.outcomes_dir"),
    )

    regime_resources = PreTrainingRegimeResources(
        dataset_path=_resolve_path(regime.get("dataset_path")),
        cache_dir=_resolve_path(regime.get("cache_dir")),
    )

    metrics_settings = PreTrainingMetricsSettings(
        output_dir=_resolve_path(metrics.get("output_dir")),
        filename=str(metrics.get("filename", "pre_training_metrics")),
        format=str(metrics.get("format", "csv")),
    )

    return PreTrainingSettings(
        data=data_paths,
        regime=regime_resources,
        metrics=metrics_settings,
        extras=dict(raw.get("extras", {})),
    )


@lru_cache(maxsize=1)
def get_pre_training_settings() -> PreTrainingSettings:
    """Return the resolved pre-training configuration (cached)."""

    raw = _load_raw_settings()
    return _build_settings(raw)


__all__ = [
    "PreTrainingSettings",
    "PreTrainingDataPaths",
    "PreTrainingMetricsSettings",
    "PreTrainingRegimeResources",
    "ResolvedPath",
    "get_pre_training_settings",
]

