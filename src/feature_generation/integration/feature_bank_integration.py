"""
Feature Bank Integration

This module provides comprehensive integration between existing feature bank features
and regime-specific features for different ML tasks. It combines volume, trend, 
volatility, momentum, and other features with regime features to create exhaustive
feature sets for each ML task.

Feature Categories Available:
- Volume: Volume patterns, OBV, AD, MFI, VWAP, volume clustering
- Trend: Moving averages, ADX, trend strength, support/resistance
- Volatility: Bollinger Bands, ATR, various volatility measures
- Momentum: RSI, MACD, Stochastic, Williams %R, momentum oscillators
- Technical: Support/resistance, candlestick patterns, oscillators
- Regime: Statistical, structural, volatility, volume, entropy, complexity
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# Import existing feature generators
from ..categories.volume import (
    VolumeFeatureGenerator, VolumeSMAGenerator, VolumeEMAGenerator,
    VolumeRatioGenerator, VolumeROCGenerator, VolumeStdGenerator,
    VolumePercentileGenerator, VolumeTrendStrengthGenerator,
    VolumeOscillatorGenerator, VolumeMomentumGenerator
)

from ..categories.trend import (
    TrendFeatureGenerator, ADXGenerator, DirectionalSignalGenerator,
    TrendScoreGenerator, SMAGenerator, EMAGenerator, WMAGenerator,
    create_default_trend_generators
)

from ..categories.volatility import (
    VolatilityFeatureGenerator, VectorBTVolatilityFeatureGenerator,
    VectorBTBollingerBandsGenerator, VectorBTAverageTrueRangeGenerator,
    VectorBTGarmanKlassVolatilityGenerator, VectorBTParkinsonVolatilityGenerator,
    VectorBTRogersSatchellVolatilityGenerator, VectorBTYangZhangVolatilityGenerator,
    create_default_volatility_generators
)

from ..categories.momentum import (
    MomentumFeatureGenerator, AnalystMomentum5mGenerator, AnalystMomentum15mGenerator,
    AnalystMomentum1hGenerator, AnalystMomentumAlignmentGenerator,
    RSIGenerator, MACDGenerator, StochasticGenerator, WilliamsRGenerator,
    MomentumOscillatorGenerator
)

# Import regime features
from ..categories.regime_features import (
    RegimeStatisticalFeatureGenerator, RegimeStructuralTrendFeatureGenerator,
    RegimeVolatilityFeatureGenerator, RegimeVolumeFeatureGenerator,
    RegimeEntropyGenerator, RegimeComplexityGenerator,
    RegimeFractalDimensionGenerator, RegimeHurstExponentGenerator,
    RegimeMemoryStrengthGenerator, RegimeCrossAssetGenerator,
    RegimeTransitionProbabilityGenerator, RegimeFeatureIntegration
)

# Import clustering features
from ..categories.clustering_features import (
    ClusteringDistanceGenerator, ClusteringSeparationGenerator,
    ClusteringStabilityGenerator, ClusteringIntegration
)

# Import task integration
from .feature_task_integration import MLTask, FeatureTaskIntegrator


class FeatureBankCategory(Enum):
    """Enumeration of feature bank categories."""
    VOLUME = "volume"
    TREND = "trend"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    REGIME = "regime"
    CLUSTERING = "clustering"


@dataclass
class FeatureBankConfig:
    """Configuration for feature bank integration."""
    
    # Feature limits for each task
    hdbscan_max_features: int = 150
    hdbscan_min_features: int = 100
    regime_clustering_max_features: int = 80
    regime_clustering_min_features: int = 40
    models_training_max_features: int = 60
    models_training_min_features: int = 30
    ensemble_training_max_features: int = 40
    ensemble_training_min_features: int = 20
    
    # Feature category weights for each task
    hdbscan_weights: Dict[FeatureBankCategory, float] = None
    regime_clustering_weights: Dict[FeatureBankCategory, float] = None
    models_training_weights: Dict[FeatureBankCategory, float] = None
    ensemble_training_weights: Dict[FeatureBankCategory, float] = None
    
    # Feature selection settings
    enable_feature_selection: bool = True
    selection_method: str = "variance"  # "variance", "correlation", "mutual_info"
    
    def __post_init__(self):
        """Set default weights for each task."""
        if self.hdbscan_weights is None:
            self.hdbscan_weights = {
                FeatureBankCategory.CLUSTERING: 0.4,
                FeatureBankCategory.VOLUME: 0.2,
                FeatureBankCategory.TREND: 0.15,
                FeatureBankCategory.VOLATILITY: 0.15,
                FeatureBankCategory.MOMENTUM: 0.1
            }
        
        if self.regime_clustering_weights is None:
            self.regime_clustering_weights = {
                FeatureBankCategory.REGIME: 0.4,
                FeatureBankCategory.VOLUME: 0.2,
                FeatureBankCategory.TREND: 0.2,
                FeatureBankCategory.VOLATILITY: 0.15,
                FeatureBankCategory.MOMENTUM: 0.05
            }
        
        if self.models_training_weights is None:
            self.models_training_weights = {
                FeatureBankCategory.REGIME: 0.3,
                FeatureBankCategory.VOLUME: 0.2,
                FeatureBankCategory.TREND: 0.2,
                FeatureBankCategory.VOLATILITY: 0.2,
                FeatureBankCategory.MOMENTUM: 0.1
            }
        
        if self.ensemble_training_weights is None:
            self.ensemble_training_weights = {
                FeatureBankCategory.REGIME: 0.25,
                FeatureBankCategory.VOLUME: 0.2,
                FeatureBankCategory.TREND: 0.2,
                FeatureBankCategory.VOLATILITY: 0.2,
                FeatureBankCategory.MOMENTUM: 0.15
            }


class FeatureBankIntegrator:
    """
    Feature Bank Integrator.
    
    Combines existing feature bank features with regime-specific features
    to create comprehensive feature sets for each ML task.
    """
    
    def __init__(self, config: Optional[FeatureBankConfig] = None):
        self.config = config or FeatureBankConfig()
        self.feature_generators = self._initialize_feature_generators()
        self.task_integrator = FeatureTaskIntegrator()
    
    def _initialize_feature_generators(self) -> Dict[FeatureBankCategory, List[Any]]:
        """Initialize feature generators for each category."""
        generators = {}
        
        # Volume generators
        generators[FeatureBankCategory.VOLUME] = [
            VolumeFeatureGenerator(),
            VolumeSMAGenerator(),
            VolumeEMAGenerator(),
            VolumeRatioGenerator(),
            VolumeROCGenerator(),
            VolumeStdGenerator(),
            VolumePercentileGenerator(),
            VolumeTrendStrengthGenerator(),
            VolumeOscillatorGenerator(),
            VolumeMomentumGenerator()
        ]
        
        # Trend generators
        generators[FeatureBankCategory.TREND] = [
            TrendFeatureGenerator(),
            ADXGenerator(),
            DirectionalSignalGenerator(),
            TrendScoreGenerator(),
            SMAGenerator(),
            EMAGenerator(),
            WMAGenerator()
        ]
        
        # Volatility generators
        generators[FeatureBankCategory.VOLATILITY] = create_default_volatility_generators()
        
        # Momentum generators
        generators[FeatureBankCategory.MOMENTUM] = [
            MomentumFeatureGenerator(),
            AnalystMomentum5mGenerator(),
            AnalystMomentum15mGenerator(),
            AnalystMomentum1hGenerator(),
            AnalystMomentumAlignmentGenerator(),
            RSIGenerator(),
            MACDGenerator(),
            StochasticGenerator(),
            WilliamsRGenerator(),
            MomentumOscillatorGenerator()
        ]
        
        # Regime generators
        generators[FeatureBankCategory.REGIME] = [
            RegimeStatisticalFeatureGenerator(),
            RegimeStructuralTrendFeatureGenerator(),
            RegimeVolatilityFeatureGenerator(),
            RegimeVolumeFeatureGenerator(),
            RegimeEntropyGenerator(),
            RegimeComplexityGenerator(),
            RegimeFractalDimensionGenerator(),
            RegimeHurstExponentGenerator(),
            RegimeMemoryStrengthGenerator(),
            RegimeCrossAssetGenerator(),
            RegimeTransitionProbabilityGenerator()
        ]
        
        # Clustering generators
        generators[FeatureBankCategory.CLUSTERING] = [
            ClusteringDistanceGenerator(),
            ClusteringSeparationGenerator(),
            ClusteringStabilityGenerator()
        ]
        
        return generators
    
    def get_comprehensive_features_for_task(self, task: MLTask, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features combining feature bank and regime features for a specific task.
        
        Args:
            task: The ML task to get features for
            data: Market data DataFrame
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        # Get task-specific configuration
        task_config = self._get_task_config(task)
        
        # Generate features from each category
        all_features = {}
        feature_metadata = {}
        
        for category, generators in self.feature_generators.items():
            if category in task_config['weights']:
                weight = task_config['weights'][category]
                category_features = self._generate_category_features(
                    category, generators, data, weight, task_config
                )
                all_features.update(category_features['features'])
                feature_metadata[category.value] = category_features['metadata']
        
        # Select optimal features for the task
        selected_features = self._select_optimal_features(
            all_features, task, task_config
        )
        
        return {
            'features': selected_features['features'],
            'feature_names': selected_features['feature_names'],
            'feature_count': len(selected_features['feature_names']),
            'target_range': task_config['target_range'],
            'task': task,
            'category_breakdown': feature_metadata,
            'selection_method': selected_features['method'],
            'description': f'Comprehensive features for {task.value}'
        }
    
    def _get_task_config(self, task: MLTask) -> Dict[str, Any]:
        """Get task-specific configuration."""
        if task == MLTask.HDBSCAN_CLUSTERING:
            return {
                'weights': self.config.hdbscan_weights,
                'target_range': (self.config.hdbscan_min_features, self.config.hdbscan_max_features),
                'priority_categories': [FeatureBankCategory.CLUSTERING, FeatureBankCategory.VOLUME, FeatureBankCategory.TREND]
            }
        elif task == MLTask.REGIME_CLUSTERING:
            return {
                'weights': self.config.regime_clustering_weights,
                'target_range': (self.config.regime_clustering_min_features, self.config.regime_clustering_max_features),
                'priority_categories': [FeatureBankCategory.REGIME, FeatureBankCategory.VOLUME, FeatureBankCategory.TREND]
            }
        elif task == MLTask.REGIME_MODELS_TRAINING:
            return {
                'weights': self.config.models_training_weights,
                'target_range': (self.config.models_training_min_features, self.config.models_training_max_features),
                'priority_categories': [FeatureBankCategory.REGIME, FeatureBankCategory.VOLUME, FeatureBankCategory.TREND]
            }
        elif task == MLTask.REGIME_ENSEMBLE_TRAINING:
            return {
                'weights': self.config.ensemble_training_weights,
                'target_range': (self.config.ensemble_training_min_features, self.config.ensemble_training_max_features),
                'priority_categories': [FeatureBankCategory.REGIME, FeatureBankCategory.VOLUME, FeatureBankCategory.TREND]
            }
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _generate_category_features(self, category: FeatureBankCategory, generators: List[Any], 
                                 data: pd.DataFrame, weight: float, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features for a specific category."""
        category_features = {}
        successful_generators = 0
        failed_generators = 0
        
        for generator in generators:
            try:
                # Generate features using the generator
                if hasattr(generator, 'generate_features'):
                    features = generator.generate_features(data)
                    category_features.update(features)
                    successful_generators += 1
                elif hasattr(generator, 'generate'):
                    features = generator.generate(data)
                    if isinstance(features, dict):
                        category_features.update(features)
                    successful_generators += 1
            except Exception as e:
                failed_generators += 1
                warnings.warn(f"Failed to generate features from {generator.__class__.__name__}: {e}")
        
        return {
            'features': category_features,
            'metadata': {
                'category': category.value,
                'weight': weight,
                'successful_generators': successful_generators,
                'failed_generators': failed_generators,
                'feature_count': len(category_features),
                'generator_names': [gen.__class__.__name__ for gen in generators]
            }
        }
    
    def _select_optimal_features(self, all_features: Dict[str, np.ndarray], 
                               task: MLTask, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal features for the task."""
        if not all_features:
            return {
                'features': {},
                'feature_names': [],
                'method': 'none'
            }
        
        feature_names = list(all_features.keys())
        target_min, target_max = task_config['target_range']
        
        # If we have fewer features than the minimum, return all
        if len(feature_names) <= target_min:
            return {
                'features': all_features,
                'feature_names': feature_names,
                'method': 'all'
            }
        
        # If we have more features than the maximum, select the best ones
        if len(feature_names) > target_max:
            selected_features = self._select_best_features(
                all_features, target_max, task_config
            )
        else:
            selected_features = all_features
        
        return {
            'features': selected_features,
            'feature_names': list(selected_features.keys()),
            'method': 'selection' if len(feature_names) > target_max else 'all'
        }
    
    def _select_best_features(self, features: Dict[str, np.ndarray], 
                            max_features: int, task_config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Select the best features based on the selection method."""
        if self.config.selection_method == "variance":
            return self._select_by_variance(features, max_features)
        elif self.config.selection_method == "correlation":
            return self._select_by_correlation(features, max_features)
        else:
            return self._select_by_variance(features, max_features)
    
    def _select_by_variance(self, features: Dict[str, np.ndarray], max_features: int) -> Dict[str, np.ndarray]:
        """Select features by variance."""
        feature_variances = {}
        for name, values in features.items():
            try:
                variance = np.var(values)
                feature_variances[name] = variance
            except:
                feature_variances[name] = 0
        
        # Sort by variance and select top features
        sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
        selected_names = [name for name, _ in sorted_features[:max_features]]
        
        return {name: features[name] for name in selected_names}
    
    def _select_by_correlation(self, features: Dict[str, np.ndarray], max_features: int) -> Dict[str, np.ndarray]:
        """Select features by correlation with price (if available)."""
        # This is a simplified correlation-based selection
        # In practice, you'd want to correlate with a target variable
        return self._select_by_variance(features, max_features)
    
    def get_feature_breakdown_by_category(self, task: MLTask, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed breakdown of features by category for a task."""
        task_config = self._get_task_config(task)
        breakdown = {}
        
        for category, generators in self.feature_generators.items():
            if category in task_config['weights']:
                category_result = self._generate_category_features(
                    category, generators, data, task_config['weights'][category], task_config
                )
                breakdown[category.value] = {
                    'feature_count': len(category_result['features']),
                    'weight': task_config['weights'][category],
                    'successful_generators': category_result['metadata']['successful_generators'],
                    'failed_generators': category_result['metadata']['failed_generators'],
                    'feature_names': list(category_result['features'].keys())[:10]  # First 10 names
                }
        
        return breakdown


# Convenience functions
def get_comprehensive_hdbscan_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive features for HDBSCAN clustering."""
    integrator = FeatureBankIntegrator()
    return integrator.get_comprehensive_features_for_task(MLTask.HDBSCAN_CLUSTERING, data)


def get_comprehensive_regime_clustering_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive features for regime clustering."""
    integrator = FeatureBankIntegrator()
    return integrator.get_comprehensive_features_for_task(MLTask.REGIME_CLUSTERING, data)


def get_comprehensive_models_training_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive features for models training."""
    integrator = FeatureBankIntegrator()
    return integrator.get_comprehensive_features_for_task(MLTask.REGIME_MODELS_TRAINING, data)


def get_comprehensive_ensemble_training_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive features for ensemble training."""
    integrator = FeatureBankIntegrator()
    return integrator.get_comprehensive_features_for_task(MLTask.REGIME_ENSEMBLE_TRAINING, data)


def get_feature_breakdown_for_task(task: MLTask, data: pd.DataFrame) -> Dict[str, Any]:
    """Get feature breakdown by category for a specific task."""
    integrator = FeatureBankIntegrator()
    return integrator.get_feature_breakdown_by_category(task, data)


__all__ = [
    'FeatureBankIntegrator',
    'FeatureBankConfig',
    'FeatureBankCategory',
    'get_comprehensive_hdbscan_features',
    'get_comprehensive_regime_clustering_features',
    'get_comprehensive_models_training_features',
    'get_comprehensive_ensemble_training_features',
    'get_feature_breakdown_for_task'
]