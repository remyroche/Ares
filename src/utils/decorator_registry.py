from __future__ import annotations

"""Central registry for all decorators with metadata and versioning."""

import logging
from datetime import datetime
from typing import A, Callableny

logger = logging.getLogger(__name__)


class DecoratorMetadata:
    """Metadata for a registered decorator."""

    def __init__(
        self,
        name: str,
        decorator: Callable,
        version: str = "1.0",
        description: str = "",
        tags: list[str] = None,
        deprecated: bool = False,
    ):
        self.name = name
        self.decorator = decorator
        self.version = version
        self.description = description
        self.tags = tags or []
        self.deprecated = deprecated
        self.registered_at = datetime.now()
        self.usage_count = 0

    def __repr__(self):
        return f"DecoratorMetadata(name='{self.name}', version='{self.version}', deprecated={self.deprecated})"


class DecoratorRegistry:
    """Central registry for all decorators with metadata and versioning."""

    def __init__(self):
        self._decorators: dict[str, DecoratorMetadata] = {}
        self._aliases: dict[str, str] = {}
        self._version_history: dict[str, list[str]] = {}

    def register(
        self,
        name: str,
        decorator: Callable,
        version: str = "1.0",
        description: str = "",
        tags: list[str] = None,
        deprecated: bool = False,
        aliases: list[str] = None,
    ) -> None:
        """Register a decorator with version tracking."""
        if name in self._decorators:
            # Update existing decorator
            existing = self._decorators[name]
            if existing.version != version:
                if name not in self._version_history:
                    self._version_history[name] = []
                self._version_history[name].append(existing.version)
                logger.info(
                    f"Updated decorator {name} from version {existing.version} to {version}"
                )

        metadata = DecoratorMetadata(
            name, decorator, version, description, tags, deprecated
        )
        self._decorators[name] = metadata

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

        logger.debug(f"Registered decorator: {name} v{version}")

    def get(self, name: str, version: str = None) -> Callable:
        """Get decorator by name and optional version."""
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._decorators:
            msg = f"Decorator '{name}' not found in registry"
            raise KeyError(msg)

        metadata = self._decorators[name]

        if version and metadata.version != version:
            msg = f"Version mismatch for {name}: requested {version}, available {metadata.version}"
            raise ValueError(msg)

        # Increment usage count
        metadata.usage_count += 1

        return metadata.decorator

    def list_decorators(
        self, include_deprecated: bool = False, tags: list[str] = None
    ) -> list[DecoratorMetadata]:
        """List all registered decorators with optional filtering."""
        decorators = list(self._decorators.values())

        if not include_deprecated:
            decorators = [d for d in decorators if not d.deprecated]

        if tags:
            decorators = [d for d in decorators if any(tag in d.tags for tag in tags)]

        return sorted(decorators, key=lambda x: x.name)

    def get_usage_stats(self) -> dict[str, int]:
        """Get usage statistics for all decorators."""
        return {
            name: metadata.usage_count for name, metadata in self._decorators.items()
        }

    def deprecate(self, name: str, replacement: str = None) -> None:
        """Mark a decorator as deprecated."""
        if name in self._decorators:
            self._decorators[name].deprecated = True
            if replacement:
                logger.warning(
                    f"Decorator '{name}' is deprecated. Use '{replacement}' instead."
                )
        else:
            msg = f"Decorator '{name}' not found in registry"
            raise KeyError(msg)

    def remove(self, name: str) -> None:
        """Remove a decorator from the registry."""
        if name in self._decorators:
            del self._decorators[name]
            # Remove aliases
            aliases_to_remove = [
                alias for alias, target in self._aliases.items() if target == name
            ]
            for alias in aliases_to_remove:
                del self._aliases[alias]
            logger.info(f"Removed decorator: {name}")
        else:
            msg = f"Decorator '{name}' not found in registry"
            raise KeyError(msg)

    def search(self, query: str) -> list[DecoratorMetadata]:
        """Search decorators by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for metadata in self._decorators.values():
            if (
                query_lower in metadata.name.lower()
                or query_lower in metadata.description.lower()
                or any(query_lower in tag.lower() for tag in metadata.tags)
            ):
                results.append(metadata)

        return results

    def export_config(self) -> dict[str, Any]:
        """Export registry configuration for persistence."""
        return {
            "decorators": {
                name: {
                    "version": meta.version,
                    "description": meta.description,
                    "tags": meta.tags,
                    "deprecated": meta.deprecated,
                    "registered_at": meta.registered_at.isoformat(),
                    "usage_count": meta.usage_count,
                }
                for name, meta in self._decorators.items()
            },
            "aliases": self._aliases,
            "version_history": self._version_history,
        }


# Global registry instance
decorator_registry = DecoratorRegistry()


def register_decorator(
    name: str,
    version: str = "1.0",
    description: str = "",
    tags: list[str] = None,
    deprecated: bool = False,
    aliases: list[str] = None,
):
    """Decorator to register a decorator function in the registry."""

    def decorator(func: Callable) -> Callable:
        decorator_registry.register(
            name=name,
            decorator=func,
            version=version,
            description=description,
            tags=tags or [],
            deprecated=deprecated,
            aliases=aliases or [],
        )
        return func

    return decorator
