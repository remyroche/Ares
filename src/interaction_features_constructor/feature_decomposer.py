"""
Feature Decomposer

Parses complex feature names into their constituent parts to understand:
- Base features from feature_bank
- Variant transformations applied
- Cross-timeframe multipliers
- Mathematical operators used
- Dependencies
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureComponents:
    """Components of a decomposed feature"""
    feature_name: str
    base_features: List[str]  # Base features from feature_bank
    variant_type: Optional[str]  # base, volnorm, vwap, trend_adj
    timeframe_multiplier: Optional[int]  # 3, 6, 9, 27
    operators: List[str]  # Mathematical operators used
    calculation_steps: List[Dict]  # Ordered steps to calculate feature
    dependencies: List[str]  # All features needed to calculate this feature


class FeatureDecomposer:
    """
    Decomposes complex feature names into their constituent parts.

    Handles:
    - Base features: rsi_14_price_returns
    - Variant features: rsi_14_volnorm
    - Cross-timeframe features: rsi_14_base_3x_ratio
    - Complex interactions: fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio
    """

    # Variant suffixes in order of priority (most specific first)
    VARIANT_SUFFIXES = ['_volnorm', '_vwap', '_trend_adj', '_base']

    # Timeframe multiplier pattern
    TIMEFRAME_PATTERN = r'_(\d+)x_ratio'

    # Mathematical operators in feature names
    OPERATORS = {
        '_x_': 'multiply',
        '_div_': 'divide',
        '_minus_': 'subtract',
        '_plus_': 'add',
        '_log_': 'log',
        '_log_ratio_': 'log_ratio',
    }

    def __init__(self):
        """Initialize the feature decomposer."""
        pass

    def decompose(self, feature_name: str) -> FeatureComponents:
        """
        Decompose a feature name into its components.

        Args:
            feature_name: The feature name to decompose

        Returns:
            FeatureComponents with all decomposed information
        """
        # Check if this is a simple base feature (no operators)
        if not any(op in feature_name for op in self.OPERATORS.keys()):
            return self._decompose_simple_feature(feature_name)

        # Complex feature with operators
        return self._decompose_complex_feature(feature_name)

    def _decompose_simple_feature(self, feature_name: str) -> FeatureComponents:
        """
        Decompose a simple feature (no operators).

        Examples:
        - rsi_14_price_returns -> base feature
        - rsi_14_volnorm -> variant feature
        - rsi_14_base_3x_ratio -> cross-timeframe feature
        """
        # Extract variant type
        variant_type = self._extract_variant_type(feature_name)

        # Extract timeframe multiplier
        timeframe_multiplier = self._extract_timeframe_multiplier(feature_name)

        # Extract base feature name
        base_feature = self._extract_base_feature_name(feature_name)

        # Build calculation steps
        calculation_steps = []

        # Step 1: Get base feature
        calculation_steps.append({
            'step': 'get_base_feature',
            'feature': base_feature
        })

        # Step 2: Apply variant transformation if needed
        if variant_type and variant_type != 'base':
            calculation_steps.append({
                'step': 'apply_variant',
                'variant_type': variant_type,
                'input': base_feature
            })

        # Step 3: Apply timeframe ratio if needed
        if timeframe_multiplier:
            input_feature = f"{base_feature}_{variant_type}" if variant_type else base_feature
            calculation_steps.append({
                'step': 'apply_timeframe_ratio',
                'multiplier': timeframe_multiplier,
                'input': input_feature
            })

        return FeatureComponents(
            feature_name=feature_name,
            base_features=[base_feature],
            variant_type=variant_type,
            timeframe_multiplier=timeframe_multiplier,
            operators=[],
            calculation_steps=calculation_steps,
            dependencies=[base_feature]
        )

    def _decompose_complex_feature(self, feature_name: str) -> FeatureComponents:
        """
        Decompose a complex feature with mathematical operators.

        Example:
        fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio

        This becomes:
        - Base features: ['fibonacci_0.236_5_price_returns', 'wavelet_energy']
        - Operators: ['multiply']
        - Steps: [get fibonacci variant 27x ratio, get wavelet variant 6x ratio, multiply them]
        """
        # Split by operators to find constituent features
        parts = []
        operators_found = []

        # Find all operators and split accordingly
        remaining = feature_name
        for op_pattern, op_name in sorted(self.OPERATORS.items(), key=lambda x: -len(x[0])):
            if op_pattern in remaining:
                # Split by this operator
                splits = remaining.split(op_pattern)
                if len(splits) == 2:
                    parts.append(splits[0])
                    operators_found.append(op_name)
                    remaining = splits[1]
                elif len(splits) > 2:
                    # Multiple occurrences
                    for i, split in enumerate(splits[:-1]):
                        parts.append(split)
                        operators_found.append(op_name)
                    remaining = splits[-1]

        # Add the final remaining part
        if remaining:
            parts.append(remaining)

        # If no operators found but we're in this function, handle as simple
        if not operators_found:
            return self._decompose_simple_feature(feature_name)

        # Decompose each part
        constituent_features = []
        all_base_features = []
        all_dependencies = []
        calculation_steps = []

        for part in parts:
            part_decomposed = self._decompose_simple_feature(part)
            constituent_features.append(part_decomposed)
            all_base_features.extend(part_decomposed.base_features)
            all_dependencies.extend(part_decomposed.dependencies)

            # Add calculation steps for this part
            for step in part_decomposed.calculation_steps:
                if step not in calculation_steps:  # Avoid duplicates
                    calculation_steps.append(step)

        # Add operator steps
        for i, operator in enumerate(operators_found):
            calculation_steps.append({
                'step': 'apply_operator',
                'operator': operator,
                'left': parts[i],
                'right': parts[i + 1] if i + 1 < len(parts) else None
            })

        return FeatureComponents(
            feature_name=feature_name,
            base_features=list(set(all_base_features)),  # Remove duplicates
            variant_type=None,  # Complex features don't have a single variant type
            timeframe_multiplier=None,  # Complex features don't have a single multiplier
            operators=operators_found,
            calculation_steps=calculation_steps,
            dependencies=list(set(all_dependencies))  # Remove duplicates
        )

    def _extract_base_feature_name(self, feature_name: str) -> str:
        """Extract the base feature name by removing variant and timeframe suffixes."""
        base_name = feature_name

        # Remove timeframe suffix first (e.g., _3x_ratio, _27x_ratio)
        timeframe_match = re.search(self.TIMEFRAME_PATTERN, base_name)
        if timeframe_match:
            base_name = base_name[:timeframe_match.start()]

        # Remove variant suffix
        for suffix in self.VARIANT_SUFFIXES:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        return base_name

    def _extract_variant_type(self, feature_name: str) -> Optional[str]:
        """Extract the variant type from feature name."""
        # First remove timeframe suffix if present
        name_without_timeframe = re.sub(self.TIMEFRAME_PATTERN, '', feature_name)

        # Check for variant suffixes
        for suffix in self.VARIANT_SUFFIXES:
            if name_without_timeframe.endswith(suffix):
                return suffix[1:]  # Remove leading underscore

        return None

    def _extract_timeframe_multiplier(self, feature_name: str) -> Optional[int]:
        """Extract the timeframe multiplier from feature name."""
        match = re.search(self.TIMEFRAME_PATTERN, feature_name)
        if match:
            return int(match.group(1))
        return None

    def batch_decompose(self, feature_names: List[str]) -> Dict[str, FeatureComponents]:
        """
        Decompose multiple features at once.

        Args:
            feature_names: List of feature names to decompose

        Returns:
            Dictionary mapping feature name to its components
        """
        return {
            name: self.decompose(name)
            for name in feature_names
        }

    def get_all_base_features(self, feature_names: List[str]) -> List[str]:
        """
        Get all unique base features needed for a list of feature names.

        Args:
            feature_names: List of feature names

        Returns:
            List of unique base feature names from feature_bank
        """
        all_base_features = set()

        for name in feature_names:
            components = self.decompose(name)
            all_base_features.update(components.base_features)

        return sorted(list(all_base_features))
