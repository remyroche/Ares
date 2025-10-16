"""Utility helpers for resolving project data directories.

This module provides the :class:`DataLocator` helper which centralises all
filesystem path resolution for training pipelines. Paths are resolved from
configuration keys while respecting the ``ARES_DATA_DIR`` and
``ARES_CACHE_DIR`` environment overrides when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional
import os

def _default_data_paths() -> Dict[str, str]:
    """Return default data path mappings relative to the data root."""

    # ``market_data`` is the canonical key used throughout the training
    # pipelines when referencing the historical feature set. Mapping it to ``.``
    # keeps compatibility with callers expecting the resolved directory itself.
    return {"market_data": "."}

def _default_cache_paths() -> Dict[str, str]:
    """Return default cache path mappings relative to the cache root."""

    return {"default": "."}

def _default_artifact_paths() -> Dict[str, str]:
    """Return default artifact path mappings relative to the artifact root."""

    return {
        "default": ".",
        # Historical behaviour stored labeler outcomes directly under an
        # ``outcomes`` directory at the repository root. Mapping the key via a
        # parent-relative path maintains compatibility without hard-coding the
        # directory in downstream components.
        "multi_horizon_outcomes": "../outcomes",
    }

def _default_generated_paths() -> Dict[str, str]:
    """Return default generated path mappings relative to the generated root."""

    return {
        "default": ".",
        "market_analysis": "market_analysis",
        "final_feature_selection": "market_analysis/final_feature_selection",
    }

def _default_config_paths() -> Dict[str, str]:
    """Return default configuration path mappings relative to the config root."""

    return {
        "multi_horizon_labeling": "multi_horizon_labeling_config.yaml",
    }

@dataclass
class DataLocatorConfig:
    """Configuration describing how logical keys map to filesystem paths."""

    data: Dict[str, str] = field(default_factory=_default_data_paths)
    cache: Dict[str, str] = field(default_factory=_default_cache_paths)
    artifacts: Dict[str, str] = field(default_factory=_default_artifact_paths)
    generated: Dict[str, str] = field(default_factory=_default_generated_paths)
    config: Dict[str, str] = field(default_factory=_default_config_paths)
    base_data_dir: Optional[str] = None
    base_cache_dir: Optional[str] = None
    base_artifacts_dir: Optional[str] = None
    base_generated_dir: Optional[str] = None
    base_config_dir: Optional[str] = None

class DataLocator:
    """Resolve filesystem paths for training artefacts.

    Parameters
    ----------
    config:
        Mapping of logical keys to relative filesystem paths.
    root:
        Repository root used when resolving relative directories. Defaults to
        ``Path.cwd()``.
    env:
        Optional mapping used to resolve environment variable overrides. The
        default uses :mod:`os.environ`.
    """

    def __init__(
        self,
        config: Optional[DataLocatorConfig] = None,
        *,
        root: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._config = config or DataLocatorConfig()
        self._root = Path(root or Path.cwd())
        self._env = env or os.environ

        self._base_data_dir = self._resolve_base(
            category="data",
            explicit=self._config.base_data_dir,
            env_override=self._env.get("ARES_DATA_DIR"),
            default="historical_data",
        )
        self._base_cache_dir = self._resolve_base(
            category="cache",
            explicit=self._config.base_cache_dir,
            env_override=self._env.get("ARES_CACHE_DIR"),
            default="data_cache",
        )
        self._base_artifacts_dir = self._resolve_base(
            category="artifacts",
            explicit=self._config.base_artifacts_dir,
            env_override=None,
            default="artifacts",
        )
        self._base_generated_dir = self._resolve_base(
            category="generated",
            explicit=self._config.base_generated_dir,
            env_override=None,
            default="generated",
        )
        self._base_config_dir = self._resolve_base(
            category="config",
            explicit=self._config.base_config_dir,
            env_override=self._env.get("ARES_CONFIG_DIR"),
            default="src/config",
        )

    @property
    def base_data_dir(self) -> Path:
        return self._base_data_dir

    @property
    def base_cache_dir(self) -> Path:
        return self._base_cache_dir

    @property
    def base_artifacts_dir(self) -> Path:
        return self._base_artifacts_dir

    @property
    def base_generated_dir(self) -> Path:
        return self._base_generated_dir

    @property
    def base_config_dir(self) -> Path:
        return self._base_config_dir

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def data_path(
        self,
        key: Optional[str] = None,
        *,
        default: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        """Resolve a data directory for ``key``."""

        return self._resolve("data", key, default, ensure_exists)

    def cache_path(
        self,
        key: Optional[str] = None,
        *,
        default: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        """Resolve a cache directory for ``key``."""

        return self._resolve("cache", key, default, ensure_exists)

    def artifacts_path(
        self,
        key: Optional[str] = None,
        *,
        default: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        """Resolve an artifacts directory for ``key``."""

        return self._resolve("artifacts", key, default, ensure_exists)

    def generated_path(
        self,
        key: Optional[str] = None,
        *,
        default: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        """Resolve a generated directory for ``key``."""

        return self._resolve("generated", key, default, ensure_exists)

    def config_path(
        self,
        key: Optional[str] = None,
        *,
        default: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        """Resolve a configuration path for ``key``."""

        return self._resolve("config", key, default, ensure_exists)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_base(
        self,
        *,
        category: str,
        explicit: Optional[str],
        env_override: Optional[str],
        default: str,
    ) -> Path:
        if env_override:
            return self._coerce_path(Path(env_override))
        if explicit:
            return self._coerce_relative(explicit)
        return self._coerce_relative(default)

    def _resolve(
        self,
        category: str,
        key: Optional[str],
        default: Optional[str],
        ensure_exists: bool,
    ) -> Path:
        mapping: Dict[str, str] = getattr(self._config, category)
        candidate: Optional[str] = None

        if key and mapping:
            candidate = mapping.get(key)
        if candidate is None and default is not None:
            candidate = default
        if candidate is None and key:
            candidate = key
        if candidate is None:
            candidate = "."

        resolved = self._coerce_relative(
            candidate,
            base=getattr(self, f"base_{category}_dir"),
        )
        if ensure_exists:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def resolved_paths(self) -> Dict[str, Dict[str, str]]:
        """Return a mapping of categories to resolved filesystem paths."""

        summary: Dict[str, Dict[str, str]] = {}
        for category in ("data", "cache", "artifacts", "generated", "config"):
            resolver = getattr(self, f"{category}_path")
            base_dir = getattr(self, f"base_{category}_dir")
            mapping: Dict[str, str] = {}
            mapping["root"] = str(base_dir)

            configured = getattr(self._config, category, {})
            for key in sorted(configured.keys()):
                try:
                    mapping[key] = str(resolver(key))
                except Exception:
                    # Fall back to the configured relative path if resolution fails
                    mapping[key] = str(configured[key])

            summary[category] = mapping

        return summary

    def _coerce_relative(self, value: str, *, base: Optional[Path] = None) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return self._coerce_path(path)
        base_dir = base or self._root
        return self._coerce_path(base_dir / path)

    def _coerce_path(self, path: Path) -> Path:
        try:
            return path.expanduser().resolve()
        except FileNotFoundError:
            # ``resolve`` with ``strict=True`` (default) raises when the path
            # does not yet exist. Using ``strict=False`` keeps behaviour
            # predictable for callers that intend to create the directory later.
            return path.expanduser().resolve(strict=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"DataLocator(base_data_dir={self.base_data_dir!s}, "
            f"base_cache_dir={self.base_cache_dir!s}, "
            f"base_artifacts_dir={self.base_artifacts_dir!s}, "
            f"base_generated_dir={self.base_generated_dir!s})"
        )

class _LocatorCategoryView:
    """Attribute-access helper that proxies lookups to a :class:`DataLocator`."""

    def __init__(self, locator: DataLocator, category: str) -> None:
        self._locator = locator
        self._category = category

    @property
    def root(self) -> Path:
        return getattr(self._locator, f"base_{self._category}_dir")

    def path(
        self,
        key: Optional[str] = None,
        *,
        default: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        resolver = getattr(self._locator, f"{self._category}_path")
        return resolver(key, default=default, ensure_exists=ensure_exists)

    def __getattr__(self, item: str) -> Path:
        if item == "root":
            return self.root
        return self.path(item)

    def __getitem__(self, item: str) -> Path:
        return self.path(item)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"LocatorCategoryView(category={self._category!r}, root={self.root!s})"

class LocatorPaths:
    """Collection of category views backed by a :class:`DataLocator`."""

    def __init__(self, locator: DataLocator) -> None:
        self._locator = locator
        self.data = _LocatorCategoryView(locator, "data")
        self.cache = _LocatorCategoryView(locator, "cache")
        self.artifacts = _LocatorCategoryView(locator, "artifacts")
        self.generated = _LocatorCategoryView(locator, "generated")
        self.config = _LocatorCategoryView(locator, "config")

    @property
    def locator(self) -> DataLocator:
        return self._locator

    def summary(self) -> Dict[str, Dict[str, str]]:
        return self._locator.resolved_paths()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"LocatorPaths(locator={self._locator!r})"
