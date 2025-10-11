"""
VectorBT Feature Registration System

This module provides comprehensive feature registration for VectorBT-enhanced
feature engineering components with advanced configuration and optimization.

Features:
- VectorBT feature generator registration
- Parameter optimization and validation
- Performance monitoring and caching
- Advanced feature selection and filtering
- Multi-timeframe analysis support
"""

import logging
from typing import Dict, List, Type, Any, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Import VectorBT base classes
from src.training.steps.feature_engineering.vectorbt_base import (
    VectorBTFeatureGenerator, VectorBTConfig, VectorBTTechnicalIndicators
)
from src.feature_generation.core.feature_bank import FeatureBank
from src.feature_generation.core.feature_registry import FeatureRegistry
from src.feature_generation.core.feature_generator import FeatureCategory, FeatureConfig

# Import VectorBT feature generators
from src.training.steps.feature_engineering.volatility.vectorbt_atr_volatility_ratio import (
    VectorBTATRVolatilityRatioGenerator
)
from src.training.steps.feature_engineering.trend.vectorbt_trend_coherence import (
    VectorBTTrendCoherenceGenerator
)
from src.training.steps.feature_engineering.price_action.vectorbt_bar_efficiency_ratio import (
    VectorBTBarEfficiencyRatioGenerator
)
from src.training.steps.feature_engineering.price_action.vectorbt_close_location_value import (
    VectorBTCloseLocationValueGenerator
)

# Import indicator suite
from src.training.steps.feature_engineering.vectorbt_indicators_suite import (
    VectorBTIndicatorSuite, VectorBTIndicatorSuiteConfig
)

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = logging.getLogger(__name__)


@dataclass
class VectorBTFeatureRegistrationConfig:
    """Configuration for VectorBT feature registration."""
    
    # Registration settings
    enable_auto_registration: bool = True
    enable_parameter_optimization: bool = True
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    
    # Feature selection settings
    enable_feature_selection: bool = True
    max_features_per_category: int = 50
    feature_importance_threshold: float = 0.01
    
    # Optimization settings
    optimization_runs: int = 100
    optimization_method: str = 'grid'  # 'grid', 'random', 'bayesian'
    n_jobs: int = -1
    
    # Performance settings
    enable_parallel_processing: bool = True
    chunk_size: int = 1000
    memory_efficient: bool = True
    
    # Validation settings
    enable_cross_validation: bool = True
    cv_folds: int = 5
    validation_metric: str = 'sharpe_ratio'
    
    # Caching settings
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds


class VectorBTFeatureRegistry:
    """
    Enhanced feature registry for VectorBT features.
    
    Provides comprehensive registration, optimization, and management
    of VectorBT-enhanced feature generators.
    """
    
    def __init__(self, config: Optional[VectorBTFeatureRegistrationConfig] = None):
        """Initialize VectorBT feature registry."""
        self.config = config or VectorBTFeatureRegistrationConfig()
        self.feature_bank = FeatureBank()
        self.registry = FeatureRegistry()
        self.indicator_suite = VectorBTIndicatorSuite()
        
        # Feature generator registry
        self.vectorbt_generators: Dict[str, Type[VectorBTFeatureGenerator]] = {}
        self.registered_features: Dict[str, Dict[str, Any]] = {}
        self.optimization_results: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with default VectorBT features
        self._register_default_vectorbt_features()
        
        tprint_info("🔧 VectorBT Feature Registry initialized")
        tprint_info(f"   → Auto registration: {self.config.enable_auto_registration}")
        tprint_info(f"   → Parameter optimization: {self.config.enable_parameter_optimization}")
        tprint_info(f"   → Performance monitoring: {self.config.enable_performance_monitoring}")
        tprint_info(f"   → Feature selection: {self.config.enable_feature_selection}")
    
    def _register_default_vectorbt_features(self) -> None:
        """Register default VectorBT feature generators."""
        try:
            # Register VectorBT feature generators
            vectorbt_features = [
                {
                    'name': 'vectorbt_atr_volatility_ratio',
                    'generator_class': VectorBTATRVolatilityRatioGenerator,
                    'category': FeatureCategory.VOLATILITY,
                    'description': 'VectorBT-enhanced ATR volatility ratio with comprehensive volatility analysis',
                    'default_lookback': 4,
                    'parameters': {
                        'short_window': 4,
                        'long_window': 20,
                        'additional_windows': [8, 14, 30],
                        'high_ratio_threshold': 1.5,
                        'extreme_ratio_threshold': 2.0,
                        'low_ratio_threshold': 0.5,
                        'bb_window': 20,
                        'bb_std': 2.0,
                        'kc_window': 20,
                        'kc_atr_multiplier': 2.0,
                        'include_atr_short': True,
                        'include_atr_long': True,
                        'include_atr_ratio': True,
                        'include_atr_grade': True,
                        'include_atr_class': True,
                        'include_bb_volatility': True,
                        'include_kc_volatility': True,
                        'include_volatility_regime': True,
                        'include_volatility_momentum': True
                    }
                },
                {
                    'name': 'vectorbt_trend_coherence',
                    'generator_class': VectorBTTrendCoherenceGenerator,
                    'category': FeatureCategory.TREND,
                    'description': 'VectorBT-enhanced trend coherence with comprehensive trend analysis',
                    'default_lookback': 8,
                    'parameters': {
                        'direction_window': 8,
                        'ema_period': 12,
                        'additional_ema_periods': [5, 8, 21, 34, 55],
                        'adx_period': 14,
                        'adx_threshold': 25.0,
                        'use_ichimoku': True,
                        'ichimoku_conversion': 9,
                        'ichimoku_base': 26,
                        'ichimoku_span_b': 52,
                        'use_psar': True,
                        'psar_af': 0.02,
                        'psar_max_af': 0.2,
                        'min_direction_consistency': 0.6,
                        'min_slope_threshold': 0.001,
                        'strong_trend_threshold': 0.8,
                        'include_direction_consistency': True,
                        'include_ema_slope': True,
                        'include_adx_strength': True,
                        'include_ichimoku_signals': True,
                        'include_psar_signals': True,
                        'include_trend_coherence_grade': True,
                        'include_trend_class': True,
                        'include_trend_regime': True,
                        'include_trend_persistence': True
                    }
                },
                {
                    'name': 'vectorbt_bar_efficiency_ratio',
                    'generator_class': VectorBTBarEfficiencyRatioGenerator,
                    'category': FeatureCategory.PRICE_ACTION,
                    'description': 'VectorBT-enhanced bar efficiency ratio with comprehensive price action analysis',
                    'default_lookback': 3,
                    'parameters': {
                        'window': 3,
                        'additional_windows': [2, 4, 6, 8, 10],
                        'high_efficiency_threshold': 0.6,
                        'low_efficiency_threshold': 0.3,
                        'extreme_efficiency_threshold': 0.8,
                        'enable_candlestick_patterns': True,
                        'pattern_window': 5,
                        'enable_momentum_analysis': True,
                        'momentum_window': 10,
                        'enable_volume_analysis': True,
                        'volume_window': 20,
                        'include_raw_efficiency': True,
                        'include_rolling_efficiency': True,
                        'include_efficiency_grade': True,
                        'include_efficiency_class': True,
                        'include_candlestick_patterns': True,
                        'include_price_action_momentum': True,
                        'include_volume_efficiency': True,
                        'include_efficiency_regime': True
                    }
                },
                {
                    'name': 'vectorbt_close_location_value',
                    'generator_class': VectorBTCloseLocationValueGenerator,
                    'category': FeatureCategory.PRICE_ACTION,
                    'description': 'VectorBT-enhanced Close Location Value with comprehensive price action analysis',
                    'default_lookback': 8,
                    'parameters': {
                        'window': 8,
                        'additional_windows': [4, 6, 10, 12, 16],
                        'positive_threshold': 0.2,
                        'negative_threshold': -0.2,
                        'volatility_threshold': 0.5,
                        'extreme_threshold': 0.5,
                        'enable_volume_analysis': True,
                        'volume_window': 20,
                        'volume_weighted_clv': True,
                        'enable_control_analysis': True,
                        'control_window': 10,
                        'enable_momentum_analysis': True,
                        'momentum_window': 5,
                        'include_raw_clv': True,
                        'include_rolling_clv': True,
                        'include_clv_volatility': True,
                        'include_clv_grade': True,
                        'include_clv_class': True,
                        'include_volume_clv': True,
                        'include_control_analysis': True,
                        'include_momentum_analysis': True,
                        'include_clv_regime': True
                    }
                }
            ]
            
            # Register each feature
            for feature_config in vectorbt_features:
                self.register_vectorbt_feature(
                    name=feature_config['name'],
                    generator_class=feature_config['generator_class'],
                    category=feature_config['category'],
                    description=feature_config['description'],
                    default_lookback=feature_config['default_lookback'],
                    parameters=feature_config['parameters']
                )
            
            tprint_success(f"✅ Registered {len(vectorbt_features)} VectorBT features")
            
        except Exception as e:
            tprint_error(f"❌ Error registering default VectorBT features: {e}")
    
    def register_vectorbt_feature(
        self,
        name: str,
        generator_class: Type[VectorBTFeatureGenerator],
        category: FeatureCategory,
        description: str,
        default_lookback: int = 20,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Register a VectorBT feature generator.
        
        Args:
            name: Feature name
            generator_class: VectorBT feature generator class
            category: Feature category
            description: Feature description
            default_lookback: Default lookback period
            parameters: Default parameters
            **kwargs: Additional parameters
            
        Returns:
            True if registration successful
        """
        try:
            # Create VectorBT configuration
            vectorbt_config = VectorBTConfig(
                enable_optimization=self.config.enable_parameter_optimization,
                optimization_runs=self.config.optimization_runs,
                optimization_method=self.config.optimization_method,
                enable_caching=self.config.enable_caching,
                cache_size=self.config.cache_size,
                enable_parallel=self.config.enable_parallel_processing,
                n_jobs=self.config.n_jobs
            )
            
            # Create feature configuration
            feature_config = FeatureConfig(
                name=name,
                category=category,
                description=description,
                required_columns=['open', 'high', 'low', 'close'],
                optional_columns=['volume'],
                default_lookback=default_lookback,
                min_lookback=1,
                max_lookback=100,
                parameters=parameters or {},
                matrix_optimized=True,
                gpu_accelerated=False,
                enable_feature_selection=self.config.enable_feature_selection
            )
            
            # Register with feature bank
            generator = generator_class(lookback=default_lookback, **kwargs)
            self.feature_bank.register_generator(generator)
            
            # Store in registry
            self.vectorbt_generators[name] = generator_class
            self.registered_features[name] = {
                'generator_class': generator_class,
                'category': category,
                'description': description,
                'default_lookback': default_lookback,
                'parameters': parameters or {},
                'vectorbt_config': vectorbt_config,
                'feature_config': feature_config
            }
            
            tprint_info(f"✅ Registered VectorBT feature: {name}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Error registering VectorBT feature {name}: {e}")
            return False
    
    def register_indicator_suite_features(
        self,
        data: pd.DataFrame,
        windows: Optional[List[int]] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Register features from the VectorBT indicator suite.
        
        Args:
            data: Sample data for feature generation
            windows: List of windows to use
            categories: List of categories to include
            
        Returns:
            Dictionary of registration results
        """
        try:
            tprint_info("🔧 Registering VectorBT indicator suite features")
            
            # Get all indicators
            all_indicators = self.indicator_suite.get_all_indicators(data, windows)
            
            # Filter by categories if specified
            if categories:
                filtered_indicators = {}
                for name, series in all_indicators.items():
                    # Simple category filtering based on name
                    for category in categories:
                        if category.lower() in name.lower():
                            filtered_indicators[name] = series
                            break
                all_indicators = filtered_indicators
            
            # Register each indicator as a feature
            registration_results = {}
            for name, series in all_indicators.items():
                try:
                    # Create a simple feature generator for the indicator
                    success = self._register_indicator_as_feature(name, series, data)
                    registration_results[name] = success
                except Exception as e:
                    tprint_warning(f"⚠️ Error registering indicator {name}: {e}")
                    registration_results[name] = False
            
            successful_registrations = sum(registration_results.values())
            tprint_success(f"✅ Registered {successful_registrations}/{len(all_indicators)} indicator features")
            
            return registration_results
            
        except Exception as e:
            tprint_error(f"❌ Error registering indicator suite features: {e}")
            return {}
    
    def _register_indicator_as_feature(
        self, 
        name: str, 
        series: pd.Series, 
        data: pd.DataFrame
    ) -> bool:
        """Register a single indicator as a feature."""
        try:
            # Create a simple feature configuration
            feature_config = FeatureConfig(
                name=f"indicator_{name}",
                category=FeatureCategory.CUSTOM,
                description=f"VectorBT indicator: {name}",
                required_columns=['open', 'high', 'low', 'close'],
                optional_columns=['volume'],
                default_lookback=20,
                min_lookback=1,
                max_lookback=100,
                parameters={},
                matrix_optimized=True,
                gpu_accelerated=False,
                enable_feature_selection=True
            )
            
            # Store in registry
            self.registered_features[f"indicator_{name}"] = {
                'generator_class': None,  # Direct series
                'category': FeatureCategory.CUSTOM,
                'description': f"VectorBT indicator: {name}",
                'default_lookback': 20,
                'parameters': {},
                'data': series,
                'is_indicator': True
            }
            
            return True
            
        except Exception as e:
            tprint_warning(f"⚠️ Error registering indicator {name}: {e}")
            return False
    
    def optimize_feature_parameters(
        self, 
        data: pd.DataFrame, 
        feature_name: str,
        target_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a specific feature.
        
        Args:
            data: Input data for optimization
            feature_name: Name of the feature to optimize
            target_metric: Target metric for optimization
            
        Returns:
            Optimized parameters
        """
        if not self.config.enable_parameter_optimization:
            return {}
        
        try:
            tprint_info(f"🔍 Optimizing parameters for {feature_name}")
            
            # Get feature configuration
            if feature_name not in self.registered_features:
                tprint_warning(f"⚠️ Feature {feature_name} not found in registry")
                return {}
            
            feature_info = self.registered_features[feature_name]
            
            # Skip optimization for indicators
            if feature_info.get('is_indicator', False):
                tprint_info(f"⚠️ Skipping optimization for indicator {feature_name}")
                return {}
            
            # Create generator instance
            generator_class = feature_info['generator_class']
            if generator_class is None:
                tprint_warning(f"⚠️ No generator class for {feature_name}")
                return {}
            
            generator = generator_class(
                lookback=feature_info['default_lookback'],
                **feature_info['parameters']
            )
            
            # Run optimization
            optimized_params = generator.optimize_parameters(data, target_metric)
            
            # Store optimization results
            self.optimization_results[feature_name] = {
                'optimized_parameters': optimized_params,
                'target_metric': target_metric,
                'optimization_time': pd.Timestamp.now()
            }
            
            tprint_success(f"✅ Optimized parameters for {feature_name}")
            return optimized_params
            
        except Exception as e:
            tprint_error(f"❌ Error optimizing parameters for {feature_name}: {e}")
            return {}
    
    def optimize_all_features(
        self, 
        data: pd.DataFrame, 
        target_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize parameters for all registered features.
        
        Args:
            data: Input data for optimization
            target_metric: Target metric for optimization
            
        Returns:
            Dictionary of optimization results
        """
        if not self.config.enable_parameter_optimization:
            return {}
        
        try:
            tprint_info("🔍 Optimizing parameters for all features")
            
            optimization_results = {}
            
            for feature_name in self.registered_features:
                if not self.registered_features[feature_name].get('is_indicator', False):
                    result = self.optimize_feature_parameters(data, feature_name, target_metric)
                    if result:
                        optimization_results[feature_name] = result
            
            tprint_success(f"✅ Optimized parameters for {len(optimization_results)} features")
            return optimization_results
            
        except Exception as e:
            tprint_error(f"❌ Error optimizing all features: {e}")
            return {}
    
    def get_feature_performance_metrics(
        self, 
        data: pd.DataFrame, 
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific feature.
        
        Args:
            data: Input data for analysis
            feature_name: Name of the feature
            
        Returns:
            Performance metrics
        """
        if not self.config.enable_performance_monitoring:
            return {}
        
        try:
            # Get feature configuration
            if feature_name not in self.registered_features:
                return {}
            
            feature_info = self.registered_features[feature_name]
            
            # Skip performance monitoring for indicators
            if feature_info.get('is_indicator', False):
                return {}
            
            # Create generator instance
            generator_class = feature_info['generator_class']
            if generator_class is None:
                return {}
            
            generator = generator_class(
                lookback=feature_info['default_lookback'],
                **feature_info['parameters']
            )
            
            # Generate features
            features = generator.generate_vectorbt_features(data)
            
            # Calculate performance metrics
            metrics = generator._calculate_performance_metrics(features)
            
            # Store performance metrics
            self.performance_metrics[feature_name] = {
                'metrics': metrics,
                'calculation_time': pd.Timestamp.now(),
                'feature_count': len(features)
            }
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating performance metrics for {feature_name}: {e}")
            return {}
    
    def get_all_performance_metrics(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all registered features.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.config.enable_performance_monitoring:
            return {}
        
        try:
            tprint_info("📊 Calculating performance metrics for all features")
            
            all_metrics = {}
            
            for feature_name in self.registered_features:
                if not self.registered_features[feature_name].get('is_indicator', False):
                    metrics = self.get_feature_performance_metrics(data, feature_name)
                    if metrics:
                        all_metrics[feature_name] = metrics
            
            tprint_success(f"✅ Calculated performance metrics for {len(all_metrics)} features")
            return all_metrics
            
        except Exception as e:
            tprint_error(f"❌ Error calculating performance metrics: {e}")
            return {}
    
    def select_best_features(
        self, 
        data: pd.DataFrame, 
        target_metric: str = 'sharpe_ratio',
        max_features: Optional[int] = None
    ) -> List[str]:
        """
        Select the best features based on performance metrics.
        
        Args:
            data: Input data for analysis
            target_metric: Target metric for selection
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        if not self.config.enable_feature_selection:
            return list(self.registered_features.keys())
        
        try:
            tprint_info("🎯 Selecting best features")
            
            # Get performance metrics for all features
            all_metrics = self.get_all_performance_metrics(data)
            
            # Calculate feature scores
            feature_scores = {}
            for feature_name, metrics in all_metrics.items():
                if 'stability_scores' in metrics:
                    # Use stability scores for selection
                    stability_scores = metrics['stability_scores']
                    if stability_scores:
                        # Average stability score
                        avg_stability = np.mean(list(stability_scores.values()))
                        feature_scores[feature_name] = avg_stability
                else:
                    feature_scores[feature_name] = 0.0
            
            # Sort features by score
            sorted_features = sorted(
                feature_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Select top features
            max_features = max_features or self.config.max_features_per_category
            selected_features = [
                feature_name for feature_name, score in sorted_features[:max_features]
                if score >= self.config.feature_importance_threshold
            ]
            
            tprint_success(f"✅ Selected {len(selected_features)} best features")
            return selected_features
            
        except Exception as e:
            tprint_error(f"❌ Error selecting best features: {e}")
            return list(self.registered_features.keys())
    
    def get_registered_features(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered features."""
        return self.registered_features.copy()
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features organized by category."""
        categories = {}
        for feature_name, feature_info in self.registered_features.items():
            category = feature_info['category'].value
            if category not in categories:
                categories[category] = []
            categories[category].append(feature_name)
        return categories
    
    def cleanup(self) -> None:
        """Clean up resources and caches."""
        if hasattr(self.feature_bank, 'cleanup'):
            self.feature_bank.cleanup()
        
        # Clear caches
        self.optimization_results.clear()
        self.performance_metrics.clear()
        
        tprint_info("🧹 VectorBT Feature Registry cleanup completed")


# Convenience functions
def create_vectorbt_feature_registry(
    config: Optional[VectorBTFeatureRegistrationConfig] = None
) -> VectorBTFeatureRegistry:
    """Create VectorBT feature registry instance."""
    return VectorBTFeatureRegistry(config)


def register_vectorbt_features_with_bank(
    feature_bank: FeatureBank,
    config: Optional[VectorBTFeatureRegistrationConfig] = None
) -> None:
    """Register VectorBT features with an existing feature bank."""
    registry = create_vectorbt_feature_registry(config)
    
    # Copy registered features to the provided feature bank
    for feature_name, feature_info in registry.get_registered_features().items():
        if not feature_info.get('is_indicator', False):
            generator_class = feature_info['generator_class']
            if generator_class:
                generator = generator_class(
                    lookback=feature_info['default_lookback'],
                    **feature_info['parameters']
                )
                feature_bank.register_generator(generator)
    
    tprint_success("✅ VectorBT features registered with feature bank")


def get_vectorbt_feature_generators() -> Dict[str, Type[VectorBTFeatureGenerator]]:
    """Get dictionary of VectorBT feature generator classes."""
    return {
        'vectorbt_atr_volatility_ratio': VectorBTATRVolatilityRatioGenerator,
        'vectorbt_trend_coherence': VectorBTTrendCoherenceGenerator,
        'vectorbt_bar_efficiency_ratio': VectorBTBarEfficiencyRatioGenerator,
        'vectorbt_close_location_value': VectorBTCloseLocationValueGenerator
    }