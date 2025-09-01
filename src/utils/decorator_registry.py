"""Central registry for all decorators with metadata and versioning."""

from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DecoratorMetadata:
    """Metadata for a registered decorator."""

    def __init__(self, name: str, decorator: Callable, version: str = "1.0",
                 description: str = "", tags: List[str] = None, deprecated: bool = False):
        self.name = name
        self.decorator = decorator
        self.version = version
        self.description = description
        self.tags = tags or []
        self.deprecated = deprecated
        self.registered_at = datetime.now()
        self.usage_count = 0

class DecoratorRegistry:
    """Central registry for all decorators with metadata and versioning."""

    def __init__(self):
        self._decorators: Dict[str, DecoratorMetadata] = {}
        self._aliases: Dict[str, str] = {}
        self._version_history: Dict[str, List[str]] = {}

    def register(self, name: str, decorator: Callable, version: str = "1.0",
                description: str = "", tags: List[str] = None, deprecated: bool = False,
                aliases: List[str] = None) -> None:
        """Register a decorator with version tracking."""
        if name in self._decorators:
            # Update existing decorator
            existing = self._decorators[name]
            if existing.version != version:
                if name not in self._version_history:
                    self._version_history[name] = []
                self._version_history[name].append(existing.version)
                logger.info(f"Updated decorator {name} from version {existing.version} to {version}")

        metadata = DecoratorMetadata(name, decorator, version, description, tags, deprecated)
        self._decorators[name] = metadata

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

        logger.debug(f"Registered decorator: {name} v{version}")

# Global registry instance
decorator_registry = DecoratorRegistry()
