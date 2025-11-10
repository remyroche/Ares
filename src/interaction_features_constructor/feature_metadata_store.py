"""
Feature Metadata Store

Stores and retrieves metadata about selected features including:
- Feature dependencies
- Calculation steps
- Parameters (lookback periods, etc.)
- Category information
"""

import json
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

from src.interaction_features_constructor.feature_decomposer import FeatureDecomposer, FeatureComponents


class FeatureMetadataStore:
    """
    Stores metadata about selected features for easy reconstruction during live trading.
    """

    def __init__(self):
        """Initialize the metadata store."""
        self.decomposer = FeatureDecomposer()
        self.metadata: Dict[str, Any] = {}

    def create_from_selection(
        self,
        selected_features: List[str],
        symbol: str = None,
        exchange: str = None,
        timeframe: str = None,
        direction: str = None,
        model: str = None
    ) -> 'FeatureMetadataStore':
        """
        Create metadata from a list of selected features.

        Args:
            selected_features: List of selected feature names
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (e.g., binance)
            timeframe: Timeframe (e.g., 15m)
            direction: Trading direction (long/short)
            model: Model type (analyst/tactician)

        Returns:
            FeatureMetadataStore instance with populated metadata
        """
        # Decompose all features
        decomposed = self.decomposer.batch_decompose(selected_features)

        # Get all unique base features
        all_base_features = self.decomposer.get_all_base_features(selected_features)

        # Build metadata
        self.metadata = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'context': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'model': model
            },
            'selected_features': selected_features,
            'base_features_required': all_base_features,
            'feature_decomposition': {
                name: self._components_to_dict(comp)
                for name, comp in decomposed.items()
            },
            'statistics': {
                'total_selected_features': len(selected_features),
                'total_base_features_required': len(all_base_features),
                'simple_features': sum(1 for comp in decomposed.values() if not comp.operators),
                'complex_features': sum(1 for comp in decomposed.values() if comp.operators),
                'variant_features': sum(1 for comp in decomposed.values() if comp.variant_type),
                'timeframe_features': sum(1 for comp in decomposed.values() if comp.timeframe_multiplier),
            }
        }

        return self

    def _components_to_dict(self, components: FeatureComponents) -> Dict[str, Any]:
        """Convert FeatureComponents to dictionary for JSON serialization."""
        return {
            'feature_name': components.feature_name,
            'base_features': components.base_features,
            'variant_type': components.variant_type,
            'timeframe_multiplier': components.timeframe_multiplier,
            'operators': components.operators,
            'calculation_steps': components.calculation_steps,
            'dependencies': components.dependencies
        }

    def save(self, filepath: str) -> None:
        """
        Save metadata to a JSON file.

        Args:
            filepath: Path to save the metadata file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'FeatureMetadataStore':
        """
        Load metadata from a JSON file.

        Args:
            filepath: Path to the metadata file

        Returns:
            FeatureMetadataStore instance with loaded metadata
        """
        store = cls()

        with open(filepath, 'r') as f:
            store.metadata = json.load(f)

        return store

    def get_selected_features(self) -> List[str]:
        """Get the list of selected features."""
        return self.metadata.get('selected_features', [])

    def get_base_features_required(self) -> List[str]:
        """Get the list of base features required from feature_bank."""
        return self.metadata.get('base_features_required', [])

    def get_feature_components(self, feature_name: str) -> Dict[str, Any]:
        """
        Get the decomposed components for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with feature components
        """
        return self.metadata.get('feature_decomposition', {}).get(feature_name)

    def get_context(self) -> Dict[str, Any]:
        """Get the context information (symbol, exchange, etc.)."""
        return self.metadata.get('context', {})

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the selected features."""
        return self.metadata.get('statistics', {})

    def to_dict(self) -> Dict[str, Any]:
        """Get the full metadata as a dictionary."""
        return self.metadata

    def __repr__(self) -> str:
        """String representation of the metadata store."""
        stats = self.get_statistics()
        return (
            f"FeatureMetadataStore("
            f"total_features={stats.get('total_selected_features', 0)}, "
            f"base_features_required={stats.get('total_base_features_required', 0)}, "
            f"simple={stats.get('simple_features', 0)}, "
            f"complex={stats.get('complex_features', 0)})"
        )
