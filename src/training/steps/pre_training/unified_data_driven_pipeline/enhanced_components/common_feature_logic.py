"""
Common Feature Generation Logic for UnifiedDataDrivenPipeline

This module provides common logic for all feature generation (interaction and cross timeframe)
and lookback optimization across the unified pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class FeatureCreationMethod(Enum):
    """Enumeration of feature creation methods."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    LOG = "log"
    SQRT = "sqrt"
    POWER = "power"
    RATIO = "ratio"
    LOG_ADD = "log_add"
    LOG_SUBTRACT = "log_subtract"
    LOG_DIVIDE = "log_divide"
    EXP_ADD = "exp_add"
    EXP_MULTIPLY = "exp_multiply"
    ABS_ADD = "abs_add"
    ABS_MULTIPLY = "abs_multiply"
    SQUARE_ADD = "square_add"
    SQUARE_MULTIPLY = "square_multiply"
    CUBE_ADD = "cube_add"
    CUBE_MULTIPLY = "cube_multiply"
    SIN_ADD = "sin_add"
    COS_MULTIPLY = "cos_multiply"
    TAN_DIVIDE = "tan_divide"


class FeatureType(Enum):
    """Enumeration of feature types."""
    CROSS_TIMEFRAME = "cross_timeframe"
    INTERACTION = "interaction"
    NO_FEATURE = "no_feature"
    COMPARISON = "comparison"


@dataclass
class FeatureGenerationConfig:
    """Configuration for common feature generation."""
    
    # Lookback optimization settings
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 5
    num_informative_periods: int = 3  # Number of informative periods to generate
    
    # Feature creation methods
    creation_methods: List[FeatureCreationMethod] = None
    
    # Cross timeframe settings
    cross_timeframe_periods: List[int] = None
    
    # Interaction settings
    interaction_orders: List[int] = None
    max_interactions_per_pair: int = 5
    
    # Optimization settings
    utility_threshold: float = 0.1
    correlation_threshold: float = 0.95
    stability_threshold: float = 0.7
    
    def __post_init__(self):
        if self.creation_methods is None:
            self.creation_methods = [
                FeatureCreationMethod.ADD,
                FeatureCreationMethod.SUBTRACT,
                FeatureCreationMethod.MULTIPLY,
                FeatureCreationMethod.DIVIDE,
                FeatureCreationMethod.LOG,
                FeatureCreationMethod.SQRT,
                FeatureCreationMethod.POWER,
                FeatureCreationMethod.RATIO,
                FeatureCreationMethod.LOG_ADD,
                FeatureCreationMethod.LOG_SUBTRACT,
                FeatureCreationMethod.LOG_DIVIDE,
                FeatureCreationMethod.EXP_ADD,
                FeatureCreationMethod.EXP_MULTIPLY,
                FeatureCreationMethod.ABS_ADD,
                FeatureCreationMethod.ABS_MULTIPLY,
                FeatureCreationMethod.SQUARE_ADD,
                FeatureCreationMethod.SQUARE_MULTIPLY,
                FeatureCreationMethod.CUBE_ADD,
                FeatureCreationMethod.CUBE_MULTIPLY,
                FeatureCreationMethod.SIN_ADD,
                FeatureCreationMethod.COS_MULTIPLY,
                FeatureCreationMethod.TAN_DIVIDE
            ]
        
        if self.cross_timeframe_periods is None:
            self.cross_timeframe_periods = [5, 10, 20, 30, 50, 100, 200]
        
        if self.interaction_orders is None:
            self.interaction_orders = [2, 3]


class CommonFeatureGenerator:
    """Common feature generation logic for all feature types."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_features_with_creation_methods(
        self, 
        series1: pd.Series, 
        series2: Optional[pd.Series] = None,
        series3: Optional[pd.Series] = None,
        feature_type: FeatureType = FeatureType.INTERACTION,
        base_name: str = "feature",
        lookback_period: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate features using various creation methods.
        
        Args:
            series1: Primary series
            series2: Secondary series (for 2-way interactions)
            series3: Tertiary series (for 3-way interactions)
            feature_type: Type of feature being generated
            base_name: Base name for the features
            lookback_period: Lookback period for the features
            
        Returns:
            List of feature dictionaries with series, formula, and metadata
        """
        features = []
        
        try:
            if series2 is None:
                # Single series features
                features.extend(self._generate_single_series_features(
                    series1, feature_type, base_name, lookback_period
                ))
            elif series3 is None:
                # Two series features (2-way interactions)
                features.extend(self._generate_two_series_features(
                    series1, series2, feature_type, base_name, lookback_period
                ))
            else:
                # Three series features (3-way interactions)
                features.extend(self._generate_three_series_features(
                    series1, series2, series3, feature_type, base_name, lookback_period
                ))
            
            return features
            
        except Exception as e:
            tprint_debug(f"Error generating features with creation methods: {e}")
            return []
    
    def _generate_single_series_features(
        self, 
        series: pd.Series, 
        feature_type: FeatureType,
        base_name: str,
        lookback_period: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate features from a single series."""
        features = []
        
        try:
            # Basic transformations
            transformations = {
                'pct_change': lambda x: x.pct_change(),
                'log': lambda x: np.log(np.abs(x) + 1e-8) * np.sign(x),
                'sqrt': lambda x: np.sqrt(np.abs(x)) * np.sign(x),
                'square': lambda x: np.square(x),
                'cube': lambda x: np.power(x, 3),
                'abs': lambda x: np.abs(x),
                'rank': lambda x: x.rank(pct=True),
                'zscore': lambda x: (x - x.mean()) / (x.std() + 1e-8),
                'sin': lambda x: np.sin(x),
                'cos': lambda x: np.cos(x),
                'tan': lambda x: np.tan(x)
            }
            
            for method_name, transform_func in transformations.items():
                try:
                    transformed_series = transform_func(series)
                    formula = f"{method_name}({base_name})"
                    
                    features.append({
                        'series': transformed_series,
                        'formula': formula,
                        'method': method_name,
                        'feature_type': feature_type.value,
                        'lookback_period': lookback_period,
                        'parent_features': [base_name]
                    })
                except Exception as e:
                    tprint_debug(f"Error applying {method_name} transformation: {e}")
                    continue
            
            return features
            
        except Exception as e:
            tprint_debug(f"Error generating single series features: {e}")
            return []
    
    def _generate_two_series_features(
        self, 
        series1: pd.Series, 
        series2: pd.Series,
        feature_type: FeatureType,
        base_name: str,
        lookback_period: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate features from two series using various creation methods."""
        features = []
        
        try:
            for method in self.config.creation_methods:
                try:
                    if method == FeatureCreationMethod.ADD:
                        result_series = series1 + series2
                        formula = f"{base_name}_1 + {base_name}_2"
                    elif method == FeatureCreationMethod.SUBTRACT:
                        result_series = series1 - series2
                        formula = f"{base_name}_1 - {base_name}_2"
                    elif method == FeatureCreationMethod.MULTIPLY:
                        result_series = series1 * series2
                        formula = f"{base_name}_1 * {base_name}_2"
                    elif method == FeatureCreationMethod.DIVIDE:
                        result_series = series1 / (series2 + 1e-8)
                        formula = f"{base_name}_1 / ({base_name}_2 + 1e-8)"
                    elif method == FeatureCreationMethod.LOG:
                        result_series = np.log(np.abs(series1) + 1e-8) * np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{base_name}_1|) * log(|{base_name}_2|)"
                    elif method == FeatureCreationMethod.SQRT:
                        result_series = np.sqrt(np.abs(series1)) * np.sqrt(np.abs(series2))
                        formula = f"sqrt(|{base_name}_1|) * sqrt(|{base_name}_2|)"
                    elif method == FeatureCreationMethod.POWER:
                        result_series = np.power(np.abs(series1), 0.5) * np.power(np.abs(series2), 0.5)
                        formula = f"pow(|{base_name}_1|, 0.5) * pow(|{base_name}_2|, 0.5)"
                    elif method == FeatureCreationMethod.RATIO:
                        result_series = series1 / (series2 + 1e-8) * series2 / (series1 + 1e-8)
                        formula = f"({base_name}_1 / {base_name}_2) * ({base_name}_2 / {base_name}_1)"
                    elif method == FeatureCreationMethod.LOG_ADD:
                        result_series = np.log(np.abs(series1) + 1e-8) + np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{base_name}_1|) + log(|{base_name}_2|)"
                    elif method == FeatureCreationMethod.LOG_SUBTRACT:
                        result_series = np.log(np.abs(series1) + 1e-8) - np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{base_name}_1|) - log(|{base_name}_2|)"
                    elif method == FeatureCreationMethod.LOG_DIVIDE:
                        result_series = np.log(np.abs(series1) + 1e-8) / (np.log(np.abs(series2) + 1e-8) + 1e-8)
                        formula = f"log(|{base_name}_1|) / log(|{base_name}_2|)"
                    elif method == FeatureCreationMethod.EXP_ADD:
                        result_series = np.exp(series1) + np.exp(series2)
                        formula = f"exp({base_name}_1) + exp({base_name}_2)"
                    elif method == FeatureCreationMethod.EXP_MULTIPLY:
                        result_series = np.exp(series1) * np.exp(series2)
                        formula = f"exp({base_name}_1) * exp({base_name}_2)"
                    elif method == FeatureCreationMethod.ABS_ADD:
                        result_series = np.abs(series1) + np.abs(series2)
                        formula = f"abs({base_name}_1) + abs({base_name}_2)"
                    elif method == FeatureCreationMethod.ABS_MULTIPLY:
                        result_series = np.abs(series1) * np.abs(series2)
                        formula = f"abs({base_name}_1) * abs({base_name}_2)"
                    elif method == FeatureCreationMethod.SQUARE_ADD:
                        result_series = np.square(series1) + np.square(series2)
                        formula = f"{base_name}_1^2 + {base_name}_2^2"
                    elif method == FeatureCreationMethod.SQUARE_MULTIPLY:
                        result_series = np.square(series1) * np.square(series2)
                        formula = f"{base_name}_1^2 * {base_name}_2^2"
                    elif method == FeatureCreationMethod.CUBE_ADD:
                        result_series = np.power(series1, 3) + np.power(series2, 3)
                        formula = f"{base_name}_1^3 + {base_name}_2^3"
                    elif method == FeatureCreationMethod.CUBE_MULTIPLY:
                        result_series = np.power(series1, 3) * np.power(series2, 3)
                        formula = f"{base_name}_1^3 * {base_name}_2^3"
                    elif method == FeatureCreationMethod.SIN_ADD:
                        result_series = np.sin(series1) + np.sin(series2)
                        formula = f"sin({base_name}_1) + sin({base_name}_2)"
                    elif method == FeatureCreationMethod.COS_MULTIPLY:
                        result_series = np.cos(series1) * np.cos(series2)
                        formula = f"cos({base_name}_1) * cos({base_name}_2)"
                    elif method == FeatureCreationMethod.TAN_DIVIDE:
                        result_series = np.tan(series1) / (np.tan(series2) + 1e-8)
                        formula = f"tan({base_name}_1) / tan({base_name}_2)"
                    else:
                        continue
                    
                    features.append({
                        'series': result_series,
                        'formula': formula,
                        'method': method.value,
                        'feature_type': feature_type.value,
                        'lookback_period': lookback_period,
                        'parent_features': [f"{base_name}_1", f"{base_name}_2"]
                    })
                    
                except Exception as e:
                    tprint_debug(f"Error applying {method.value} to two series: {e}")
                    continue
            
            return features
            
        except Exception as e:
            tprint_debug(f"Error generating two series features: {e}")
            return []
    
    def _generate_three_series_features(
        self, 
        series1: pd.Series, 
        series2: pd.Series,
        series3: pd.Series,
        feature_type: FeatureType,
        base_name: str,
        lookback_period: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate features from three series using various creation methods."""
        features = []
        
        try:
            # Three-way interaction methods
            three_way_methods = [
                ('multiply', lambda x, y, z: x * y * z, f"{base_name}_1 * {base_name}_2 * {base_name}_3"),
                ('add', lambda x, y, z: x + y + z, f"{base_name}_1 + {base_name}_2 + {base_name}_3"),
                ('ratio', lambda x, y, z: (x * y) / (z + 1e-8), f"({base_name}_1 * {base_name}_2) / ({base_name}_3 + 1e-8)"),
                ('log_multiply', lambda x, y, z: np.log(np.abs(x) + 1e-8) * np.log(np.abs(y) + 1e-8) * np.log(np.abs(z) + 1e-8), 
                 f"log(|{base_name}_1|) * log(|{base_name}_2|) * log(|{base_name}_3|)"),
                ('exp_add', lambda x, y, z: np.exp(x) + np.exp(y) + np.exp(z), f"exp({base_name}_1) + exp({base_name}_2) + exp({base_name}_3)"),
                ('abs_multiply', lambda x, y, z: np.abs(x) * np.abs(y) * np.abs(z), f"abs({base_name}_1) * abs({base_name}_2) * abs({base_name}_3)")
            ]
            
            for method_name, transform_func, formula in three_way_methods:
                try:
                    result_series = transform_func(series1, series2, series3)
                    
                    features.append({
                        'series': result_series,
                        'formula': formula,
                        'method': method_name,
                        'feature_type': feature_type.value,
                        'lookback_period': lookback_period,
                        'parent_features': [f"{base_name}_1", f"{base_name}_2", f"{base_name}_3"]
                    })
                    
                except Exception as e:
                    tprint_debug(f"Error applying {method_name} to three series: {e}")
                    continue
            
            return features
            
        except Exception as e:
            tprint_debug(f"Error generating three series features: {e}")
            return []


class CommonLookbackOptimizer:
    """Common lookback optimization logic for all feature types."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_lookback_periods(
        self, 
        features: List[Dict[str, Any]], 
        targets: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize lookback periods by selecting the most informative periods.
        
        For single features: select the most informative period
        For cross timeframe features: select multiple informative periods
        
        Args:
            features: List of feature dictionaries
            targets: Target series for utility calculation
            
        Returns:
            List of optimized features
        """
        if not features:
            return features
        
        try:
            # Group features by base name and type
            feature_groups = self._group_features_by_type_and_name(features)
            optimized_features = []
            
            for group_key, group_features in feature_groups.items():
                feature_type, base_name = group_key
                
                if feature_type == FeatureType.CROSS_TIMEFRAME.value:
                    # For cross timeframe features, select multiple informative periods
                    optimized_group = self._optimize_cross_timeframe_lookbacks(
                        group_features, targets
                    )
                else:
                    # For single features, select the most informative period
                    optimized_group = self._optimize_single_feature_lookbacks(
                        group_features, targets
                    )
                
                optimized_features.extend(optimized_group)
            
            tprint_debug(f"Optimized {len(features)} features to {len(optimized_features)} features")
            return optimized_features
            
        except Exception as e:
            tprint_debug(f"Error optimizing lookback periods: {e}")
            return features
    
    def _group_features_by_type_and_name(self, features: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Group features by type and base name."""
        groups = defaultdict(list)
        
        for feature in features:
            feature_type = feature.get('feature_type', 'unknown')
            base_name = self._extract_base_name(feature.get('formula', ''))
            group_key = (feature_type, base_name)
            groups[group_key].append(feature)
        
        return dict(groups)
    
    def _extract_base_name(self, formula: str) -> str:
        """Extract base name from formula."""
        # Simple extraction - could be enhanced
        if '_' in formula:
            parts = formula.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:-1])  # Remove last part (likely period)
        return formula.split('(')[0] if '(' in formula else formula
    
    def _optimize_single_feature_lookbacks(
        self, 
        features: List[Dict[str, Any]], 
        targets: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """Optimize lookback periods for single features (select most informative)."""
        if len(features) <= 1:
            return features
        
        try:
            # Calculate utility scores
            for feature in features:
                feature['utility_score'] = self._calculate_utility_score(
                    feature['series'], targets
                )
            
            # Sort by utility score and select the best one
            features.sort(key=lambda x: x.get('utility_score', 0), reverse=True)
            best_feature = features[0]
            
            # Add metadata
            best_feature['optimization_type'] = 'single_best'
            best_feature['total_candidates'] = len(features)
            
            return [best_feature]
            
        except Exception as e:
            tprint_debug(f"Error optimizing single feature lookbacks: {e}")
            return features
    
    def _optimize_cross_timeframe_lookbacks(
        self, 
        features: List[Dict[str, Any]], 
        targets: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """Optimize lookback periods for cross timeframe features (select multiple informative)."""
        if len(features) <= 1:
            return features
        
        try:
            # Calculate utility scores
            for feature in features:
                feature['utility_score'] = self._calculate_utility_score(
                    feature['series'], targets
                )
            
            # Sort by utility score
            features.sort(key=lambda x: x.get('utility_score', 0), reverse=True)
            
            # Select top features, ensuring diversity in lookback periods
            selected_features = []
            used_periods = set()
            
            for feature in features:
                lookback_period = feature.get('lookback_period')
                
                # If we haven't used this period yet, or if it's significantly better
                if (lookback_period not in used_periods or 
                    len(selected_features) < self.config.num_informative_periods):
                    
                    selected_features.append(feature)
                    if lookback_period is not None:
                        used_periods.add(lookback_period)
                    
                    # Stop if we have enough diverse features
                    if len(selected_features) >= self.config.num_informative_periods:
                        break
            
            # Add metadata
            for i, feature in enumerate(selected_features):
                feature['optimization_type'] = 'cross_timeframe_multiple'
                feature['selection_rank'] = i + 1
                feature['total_candidates'] = len(features)
            
            return selected_features
            
        except Exception as e:
            tprint_debug(f"Error optimizing cross timeframe lookbacks: {e}")
            return features
    
    def _calculate_utility_score(
        self, 
        feature_series: pd.Series, 
        targets: Optional[pd.Series] = None
    ) -> float:
        """Calculate utility score for a feature."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(feature_series.var())
            
            # Align series
            aligned_feature = feature_series.dropna()
            aligned_targets = targets.reindex(aligned_feature.index).dropna()
            
            if len(aligned_feature) < 10 or len(aligned_targets) < 10:
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(aligned_feature, aligned_targets)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return abs(correlation)
            
        except Exception as e:
            tprint_debug(f"Error calculating utility score: {e}")
            return 0.0


def create_common_feature_generator(config: Optional[FeatureGenerationConfig] = None) -> CommonFeatureGenerator:
    """Create a common feature generator instance."""
    if config is None:
        config = FeatureGenerationConfig()
    return CommonFeatureGenerator(config)


def create_common_lookback_optimizer(config: Optional[FeatureGenerationConfig] = None) -> CommonLookbackOptimizer:
    """Create a common lookback optimizer instance."""
    if config is None:
        config = FeatureGenerationConfig()
    return CommonLookbackOptimizer(config)