"""
Tactician Feature Optimization Integration

This module provides integration between the complementary lookback optimizer
and the tactician training pipeline, ensuring features are optimized for
complementary information beyond analyst outputs.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from .complementary_lookback_optimizer import (
    ComplementaryLookbackOptimizer,
    ComplementaryOptimizationConfig,
    optimize_complementary_lookbacks
)
# Lazy import to avoid circular dependency
def _get_feature_generator_imports():
    try:
        from ...core.feature_generator import FeatureGenerator
        return FeatureGenerator
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"FeatureGenerator import failed: {e}")
        return None
from ....utils.tprint import tprint

logger = logging.getLogger(__name__)

class TacticianFeatureOptimizer:
    """
    Feature optimizer specifically designed for Tactician training.
    
    This optimizer ensures that features are optimized for complementary
    information beyond what the Analyst already provides, using regime-invariant
    optimization for consistent performance across market conditions.
    """

    def __init__(self, config: Optional[ComplementaryOptimizationConfig] = None):
        """
        Initialize the tactician feature optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or ComplementaryOptimizationConfig()
        self.logger = logger.getChild('TacticianFeatureOptimizer')
        
        # Initialize complementary optimizer
        self.optimizer = ComplementaryLookbackOptimizer(self.config)
        
        self.logger.info("✅ TacticianFeatureOptimizer initialized")
        tprint("🎯 TacticianFeatureOptimizer initialized with complementary optimization")

    def optimize_for_tactician_training(self,
                                      generators: List[Any],
                                      data: pd.DataFrame,
                                      tactician_targets: Dict[str, pd.Series],
                                      analyst_outputs: Optional[Dict[str, pd.Series]] = None,
                                      regime_assignments: Optional[pd.Series] = None) -> Dict[str, int]:
        """
        Optimize features for tactician training using complementary scoring.

        Args:
            generators: List of feature generators to optimize
            data: Input market data
            tactician_targets: Tactician target variables (y_success, r_H, etc.)
            analyst_outputs: Optional analyst outputs for complementary scoring
            regime_assignments: Optional regime assignments for regime-invariant optimization

        Returns:
            Dictionary mapping feature names to optimal lookback periods
        """
        self.logger.info("🎯 Optimizing features for tactician training using complementary scoring")
        tprint("🎯 Optimizing features for tactician training using complementary scoring")

        # Use primary tactician target (y_success) for optimization
        primary_target = tactician_targets.get('y_success')
        if primary_target is None:
            # Fallback to first available target
            primary_target = list(tactician_targets.values())[0]
            self.logger.warning(f"No 'y_success' target found, using {list(tactician_targets.keys())[0]}")

        # Extract analyst signals for complementary scoring
        analyst_signals = None
        if analyst_outputs is not None:
            # Use analyst OOF score as primary signal
            analyst_signals = analyst_outputs.get('analyst_oof_score')
            if analyst_signals is None:
                # Fallback to first available analyst output
                analyst_signals = list(analyst_outputs.values())[0]
                self.logger.info(f"Using {list(analyst_outputs.keys())[0]} as analyst signal")

        # Add tactician target to data for optimization
        data_with_target = data.copy()
        data_with_target['tactician_target'] = primary_target

        # Optimize using complementary scoring with Tactician mode
        final_lookbacks = self.optimizer.optimize_multiple_features(
            generators=generators,
            data=data_with_target,
            target_column='tactician_target',
            analyst_signals=analyst_signals,
            regime_series=regime_assignments
        )

        self.logger.info(f"✅ Optimized {len(final_lookbacks)} features for tactician training")
        tprint(f"✅ Optimized {len(final_lookbacks)} features for tactician training")
        return final_lookbacks

    def generate_tactician_features(self,
                                  feature_bank,
                                  data: pd.DataFrame,
                                  tactician_targets: Dict[str, pd.Series],
                                  analyst_outputs: Optional[Dict[str, pd.Series]] = None,
                                  regime_assignments: Optional[pd.Series] = None,
                                  categories: Optional[List[str]] = None,
                                  features: Optional[List[str]] = None,
                                  **kwargs) -> pd.DataFrame:
        """
        Generate features using the feature bank in Tactician mode.

        Args:
            feature_bank: FeatureBank instance
            data: Input market data
            tactician_targets: Tactician target variables
            analyst_outputs: Optional analyst outputs for complementary scoring
            regime_assignments: Optional regime assignments for regime-invariant optimization
            categories: Feature categories to generate
            features: Specific features to generate
            **kwargs: Additional arguments for feature generation

        Returns:
            DataFrame with generated features
        """
        tprint("🎯 Generating features in Tactician mode with complementary optimization")
        
        # Extract primary target for optimization
        primary_target = tactician_targets.get('y_success')
        if primary_target is None:
            primary_target = list(tactician_targets.values())[0]
            tprint(f"⚠️ No 'y_success' target found, using {list(tactician_targets.keys())[0]}")

        # Extract analyst signals
        analyst_signals = None
        if analyst_outputs is not None:
            analyst_signals = analyst_outputs.get('analyst_oof_score')
            if analyst_signals is None:
                analyst_signals = list(analyst_outputs.values())[0]

        # Call feature bank with Tactician mode enabled
        tactician_kwargs = {
            'tactician_mode': True,
            'analyst_signals': analyst_signals,
            'regime_series': regime_assignments,
            'lookback_optimization': True,
            'target_column': 'y_success',
            **kwargs
        }

        # Add tactician target to data
        data_with_target = data.copy()
        data_with_target['y_success'] = primary_target

        # Generate features using feature bank in Tactician mode
        features_df = feature_bank.generate_features(
            data=data_with_target,
            categories=categories,
            features=features,
            **tactician_kwargs
        )

        tprint(f"✅ Generated {len(features_df.columns)} features in Tactician mode")
        return features_df

    def optimize_with_multi_target_objectives(self,
                                             generators: List[Any],
                                             data: pd.DataFrame,
                                             tactician_targets: Dict[str, pd.Series],
                                             analyst_outputs: Optional[Dict[str, pd.Series]] = None,
                                             regime_assignments: Optional[pd.Series] = None,
                                             target_weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
        """
        Optimize features using multiple tactician targets with weighted objectives.

        Args:
            generators: List of feature generators to optimize
            data: Input market data
            tactician_targets: Multiple tactician target variables
            analyst_outputs: Optional analyst outputs for complementary scoring
            regime_assignments: Optional regime assignments for regime-invariant optimization
            target_weights: Optional weights for different targets

        Returns:
            Dictionary mapping feature names to optimal lookback periods
        """
        self.logger.info("🎯 Optimizing features with multi-target objectives")

        # Default target weights
        if target_weights is None:
            target_weights = {
                'y_success': 0.4,      # Primary: profit success
                'r_H': 0.3,            # Secondary: realized returns
                'time_to_hit': 0.2,    # Tertiary: timing
                'direction': 0.1       # Quaternary: direction
            }

        # Filter available targets
        available_targets = {k: v for k, v in tactician_targets.items() if k in target_weights}
        if not available_targets:
            self.logger.warning("No valid targets found, using first available")
            available_targets = tactician_targets

        # Extract analyst signals
        analyst_signals = None
        if analyst_outputs is not None:
            analyst_signals = analyst_outputs.get('analyst_oof_score')

        # Optimize for each target and combine results
        target_results = {}
        for target_name, target_series in available_targets.items():
            weight = target_weights.get(target_name, 1.0)
            self.logger.info(f"Optimizing for target: {target_name} (weight: {weight})")

            # Add target to data
            data_with_target = data.copy()
            data_with_target[f'tactician_target_{target_name}'] = target_series

            # Optimize for this target
            target_lookbacks = self.optimizer.optimize_multiple_features(
                generators=generators,
                data=data_with_target,
                target_column=f'tactician_target_{target_name}',
                analyst_signals=analyst_signals,
                regime_series=regime_assignments
            )

            # Store results with weights
            for feature_name, lookback in target_lookbacks.items():
                if feature_name not in target_results:
                    target_results[feature_name] = []
                target_results[feature_name].append((lookback, weight))

        # Combine results using weighted average
        final_lookbacks = {}
        for feature_name, results in target_results.items():
            if not results:
                continue

            # Calculate weighted average lookback
            total_weight = sum(weight for _, weight in results)
            if total_weight > 0:
                weighted_lookback = sum(lookback * weight for lookback, weight in results) / total_weight
                final_lookbacks[feature_name] = int(round(weighted_lookback))
            else:
                # Fallback to first result
                final_lookbacks[feature_name] = results[0][0]

        self.logger.info(f"✅ Optimized {len(final_lookbacks)} features with multi-target objectives")
        return final_lookbacks

    def get_optimization_report(self,
                              optimal_lookbacks: Dict[str, int],
                                             generators: List[Any],
                              data: pd.DataFrame,
                              tactician_targets: Dict[str, pd.Series],
                              analyst_outputs: Optional[Dict[str, pd.Series]] = None,
                              regime_assignments: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report.

        Args:
            optimal_lookbacks: Optimal lookback periods for each feature
            generators: List of feature generators
            data: Input market data
            tactician_targets: Tactician target variables
            analyst_outputs: Optional analyst outputs
            regime_assignments: Optional regime assignments

        Returns:
            Comprehensive optimization report
        """
        self.logger.info("📊 Generating optimization report")

        report = {
            'optimization_summary': self.optimizer.get_optimization_summary(optimal_lookbacks),
            'tactician_specific_metrics': {},
            'complementary_analysis': {},
            'regime_analysis': {},
            'recommendations': []
        }

        # Analyze complementary information
        if analyst_outputs is not None:
            analyst_signals = analyst_outputs.get('analyst_oof_score')
            if analyst_signals is not None:
                report['complementary_analysis'] = self._analyze_complementary_information(
                    generators, data, tactician_targets, analyst_signals, optimal_lookbacks
                )

        # Analyze regime performance
        if regime_assignments is not None:
            report['regime_analysis'] = self._analyze_regime_performance(
                generators, data, tactician_targets, regime_assignments, optimal_lookbacks
            )

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(
            optimal_lookbacks, report['complementary_analysis'], report['regime_analysis']
        )

        return report

    def _analyze_complementary_information(self,
                                         generators: List[Any],
                                         data: pd.DataFrame,
                                         tactician_targets: Dict[str, pd.Series],
                                         analyst_signals: pd.Series,
                                         optimal_lookbacks: Dict[str, int]) -> Dict[str, Any]:
        """Analyze complementary information provided by features."""
        complementary_analysis = {
            'high_complementary_features': [],
            'low_complementary_features': [],
            'analyst_alignment_scores': {},
            'complementary_info_gains': {}
        }

        primary_target = tactician_targets.get('y_success', list(tactician_targets.values())[0])

        for generator in generators:
            feature_name = generator.config.name
            if feature_name not in optimal_lookbacks:
                continue

            try:
                # Generate feature with optimal lookback
                optimal_lookback = optimal_lookbacks[feature_name]
                if generator.supports_lookback_optimization():
                    result = generator.generate_with_lookback(data, optimal_lookback)
                else:
                    result = generator.generate(data)

                if not result.success:
                    continue

                # Calculate complementary metrics
                valid_indices = ~(result.data.isna() | primary_target.isna() | analyst_signals.isna())
                if valid_indices.sum() < 10:
                    continue

                feature_clean = result.data[valid_indices]
                target_clean = primary_target[valid_indices]
                analyst_clean = analyst_signals[valid_indices]

                # Calculate correlations
                feature_target_corr = abs(feature_clean.corr(target_clean))
                feature_analyst_corr = abs(feature_clean.corr(analyst_clean))
                analyst_target_corr = abs(analyst_clean.corr(target_clean))

                # Calculate complementary information
                if analyst_target_corr > 0:
                    complementary_gain = max(0, feature_target_corr - feature_analyst_corr)
                else:
                    complementary_gain = feature_target_corr

                # Store metrics
                complementary_analysis['analyst_alignment_scores'][feature_name] = feature_analyst_corr
                complementary_analysis['complementary_info_gains'][feature_name] = complementary_gain

                # Categorize features
                if complementary_gain > 0.1:  # High complementary information
                    complementary_analysis['high_complementary_features'].append(feature_name)
                elif complementary_gain < 0.05:  # Low complementary information
                    complementary_analysis['low_complementary_features'].append(feature_name)

            except Exception as e:
                self.logger.warning(f"Error analyzing complementary information for {feature_name}: {e}")

        return complementary_analysis

    def _analyze_regime_performance(self,
                                             generators: List[Any],
                                  data: pd.DataFrame,
                                  tactician_targets: Dict[str, pd.Series],
                                  regime_assignments: pd.Series,
                                  optimal_lookbacks: Dict[str, int]) -> Dict[str, Any]:
        """Analyze feature performance across different regimes."""
        regime_analysis = {
            'regime_performance': {},
            'regime_consistency': {},
            'regime_recommendations': []
        }

        primary_target = tactician_targets.get('y_success', list(tactician_targets.values())[0])
        unique_regimes = regime_assignments.unique()

        for generator in generators:
            feature_name = generator.config.name
            if feature_name not in optimal_lookbacks:
                continue

            try:
                # Generate feature with optimal lookback
                optimal_lookback = optimal_lookbacks[feature_name]
                if generator.supports_lookback_optimization():
                    result = generator.generate_with_lookback(data, optimal_lookback)
                else:
                    result = generator.generate(data)

                if not result.success:
                    continue

                # Analyze performance per regime
                regime_performance = {}
                for regime in unique_regimes:
                    regime_mask = regime_assignments == regime
                    if regime_mask.sum() < 10:  # Need minimum samples
                        continue

                    regime_feature = result.data[regime_mask]
                    regime_target = primary_target[regime_mask]
                    
                    valid_indices = ~(regime_feature.isna() | regime_target.isna())
                    if valid_indices.sum() < 5:
                        continue

                    regime_corr = abs(regime_feature[valid_indices].corr(regime_target[valid_indices]))
                    if not np.isnan(regime_corr):
                        regime_performance[str(regime)] = regime_corr

                regime_analysis['regime_performance'][feature_name] = regime_performance

                # Calculate regime consistency
                if regime_performance:
                    regime_scores = list(regime_performance.values())
                    min_score = min(regime_scores)
                    max_score = max(regime_scores)
                    consistency = min_score / max_score if max_score > 0 else 0
                    regime_analysis['regime_consistency'][feature_name] = consistency

            except Exception as e:
                self.logger.warning(f"Error analyzing regime performance for {feature_name}: {e}")

        return regime_analysis

    def _generate_recommendations(self,
                                optimal_lookbacks: Dict[str, int],
                                complementary_analysis: Dict[str, Any],
                                regime_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Complementary information recommendations
        high_comp_features = complementary_analysis.get('high_complementary_features', [])
        low_comp_features = complementary_analysis.get('low_complementary_features', [])

        if high_comp_features:
            recommendations.append(
                f"High complementary features (provide unique information beyond analyst): {high_comp_features}"
            )

        if low_comp_features:
            recommendations.append(
                f"Low complementary features (may be redundant with analyst): {low_comp_features}"
            )

        # Regime consistency recommendations
        regime_consistency = regime_analysis.get('regime_consistency', {})
        inconsistent_features = [
            name for name, consistency in regime_consistency.items() 
            if consistency < 0.5
        ]

        if inconsistent_features:
            recommendations.append(
                f"Features with low regime consistency (may need regime-specific optimization): {inconsistent_features}"
            )

        # Lookback period recommendations
        lookbacks = list(optimal_lookbacks.values())
        if lookbacks:
            avg_lookback = np.mean(lookbacks)
            if avg_lookback > 50:
                recommendations.append("Consider reducing lookback periods for faster adaptation")
            elif avg_lookback < 10:
                recommendations.append("Consider increasing lookback periods for more stable features")

        return recommendations

# Convenience functions
def optimize_tactician_features(generators: List[Any],
                               data: pd.DataFrame,
                               tactician_targets: Dict[str, pd.Series],
                               analyst_outputs: Optional[Dict[str, pd.Series]] = None,
                               regime_assignments: Optional[pd.Series] = None,
                               config: Optional[ComplementaryOptimizationConfig] = None) -> Dict[str, int]:
    """
    Optimize features for tactician training using complementary scoring.

    Args:
        generators: List of feature generators
        data: Input market data
        tactician_targets: Tactician target variables
        analyst_outputs: Optional analyst outputs for complementary scoring
        regime_assignments: Optional regime assignments for regime-invariant optimization
        config: Optimization configuration

    Returns:
        Dictionary mapping feature names to optimal lookback periods
    """
    optimizer = TacticianFeatureOptimizer(config)
    return optimizer.optimize_for_tactician_training(
        generators, data, tactician_targets, analyst_outputs, regime_assignments
    )

def generate_tactician_features_with_optimization(feature_bank,
                                                data: pd.DataFrame,
                                                tactician_targets: Dict[str, pd.Series],
                                                analyst_outputs: Optional[Dict[str, pd.Series]] = None,
                                                regime_assignments: Optional[pd.Series] = None,
                                                categories: Optional[List[str]] = None,
                                                features: Optional[List[str]] = None,
                                                config: Optional[ComplementaryOptimizationConfig] = None,
                                                **kwargs) -> pd.DataFrame:
    """
    Generate features using the feature bank in Tactician mode with complementary optimization.

    Args:
        feature_bank: FeatureBank instance
        data: Input market data
        tactician_targets: Tactician target variables
        analyst_outputs: Optional analyst outputs for complementary scoring
        regime_assignments: Optional regime assignments for regime-invariant optimization
        categories: Feature categories to generate
        features: Specific features to generate
        config: Optimization configuration
        **kwargs: Additional arguments for feature generation

    Returns:
        DataFrame with generated features
    """
    optimizer = TacticianFeatureOptimizer(config)
    return optimizer.generate_tactician_features(
        feature_bank, data, tactician_targets, analyst_outputs, regime_assignments,
        categories, features, **kwargs
    )

def get_tactician_optimization_config(**kwargs) -> ComplementaryOptimizationConfig:
    """
    Create a tactician-specific optimization configuration.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Tactician-specific optimization configuration
    """
    # Set tactician-specific defaults
    tactician_defaults = {
        'analyst_alignment_penalty': 0.7,  # Higher penalty for analyst alignment
        'complementary_bonus': 2.0,        # Higher bonus for complementary info
        'regime_consistency_weight': 0.4,  # Higher weight for regime consistency
        'temporal_stability_weight': 0.3,  # Higher weight for temporal stability
    }
    
    # Merge with provided kwargs
    tactician_defaults.update(kwargs)
    
    return ComplementaryOptimizationConfig(**tactician_defaults)
