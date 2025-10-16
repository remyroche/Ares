"""
Base Feature Registry Interface

Provides a shared interface for feature registries across
feature_generation and feature_engineering_roadmap systems.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

# Import common utilities
from ..utils import TPRINT_AVAILABLE, tprint

logger = logging.getLogger(__name__)

class BaseFeatureRegistry(ABC):
    """
    Abstract base class for feature registries.

    This interface ensures both feature_generation and feature_engineering_roadmap
    registries can be used interchangeably where appropriate.

    Implementations:
    - feature_generation/core/feature_registry.py -> FeatureRegistry
    - feature_engineering_roadmap/feature_registry.py -> FeatureRegistry
    """

    def __init__(self):
        """Initialize the registry."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseFeatureRegistry] Initializing {self.__class__.__name__}", color="cyan")
        self.logger = logger.getChild(self.__class__.__name__)

    @abstractmethod
    def register(self, feature: Any) -> None:
        """
        Register a feature or feature generator.

        Args:
            feature: Feature or generator to register

        Note:
            Implementation details vary:
            - feature_generation: Accepts FeatureGenerator instances
            - feature_engineering_roadmap: May not support dynamic registration
        """
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[Any]:
        """
        Get a feature or generator by name.

        Args:
            name: Name of the feature/generator

        Returns:
            Feature/generator if found, None otherwise
        """
        pass

    @abstractmethod
    def list_names(self) -> List[str]:
        """
        List all registered feature names.

        Returns:
            List of feature names
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry contents.

        Returns:
            Dictionary with registry statistics and information
        """
        pass

    def __len__(self) -> int:
        """
        Get number of registered features.

        Returns:
            Number of features in registry
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseFeatureRegistry] Getting registry length", color="cyan")
        return len(self.list_names())

    def __contains__(self, name: str) -> bool:
        """
        Check if feature exists in registry.

        Args:
            name: Feature name to check

        Returns:
            True if feature exists, False otherwise
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseFeatureRegistry] Checking if feature '{name}' exists", color="cyan")
        return self.get_by_name(name) is not None

    def validate(self) -> bool:
        """
        Validate registry integrity.

        Returns:
            True if valid, False otherwise

        Raises:
            RuntimeError: If validation fails with specific error details
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseFeatureRegistry] Validating registry integrity", color="cyan")

        try:
            # Basic validation: check if we can list names
            names = self.list_names()
            if not isinstance(names, list):
                error_msg = f"list_names() must return a list, got {type(names)}"
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [BaseFeatureRegistry] {error_msg}", color="red")
                raise RuntimeError(error_msg)

            # Check for duplicate names
            if len(names) != len(set(names)):
                error_msg = "Duplicate feature names found in registry"
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [BaseFeatureRegistry] {error_msg}", color="red")
                raise RuntimeError(error_msg)

            if TPRINT_AVAILABLE:
                tprint(f"✅ [BaseFeatureRegistry] Registry validation passed with {len(names)} features", color="green")
            return True

        except Exception as e:
            error_msg = f"Registry validation failed: {e}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [BaseFeatureRegistry] {error_msg}", color="red")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
