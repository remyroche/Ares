"""
Enhanced Consensus System with Advanced Candle Features

This module provides an enhanced consensus system that integrates all advanced
candle features including series patterns, cross-timeframe analysis, and
multi-dimensional interactions.

Key Features:
- Series of candles (consecutive patterns, sequences)
- Cross-timeframe candle analysis
- Multi-dimensional interactions (volume, momentum, volatility)
- Pattern strength and quality assessment
- Enhanced consensus validation
- Comprehensive feature engineering
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Core imports
from .ml_candle_pattern_indicators import (
    MLIndicatorGenerator, IndicatorType, ModelType, IndicatorConfig,
    create_ml_indicator_generator
)
from .advanced_candle_features import (
    AdvancedCandleFeatureGenerator, AdvancedFeatureConfig, create_advanced_candle_feature_generator
)
from .model_comparison_pipeline import (
    ModelComparisonPipeline, ConsensusMethod, ConsensusConfig, create_model_comparison_pipeline
)
from .consensus_indicator_system import (
    ConsensusIndicatorSystem, ConsensusSystemConfig, create_consensus_system
)

logger = logging.getLogger(__name__)


class EnhancedConsensusConfig:
    """Configuration for enhanced consensus system with advanced features."""
    # Advanced feature configuration
    enable_series_features: bool = True
    enable_cross_timeframe: bool = True
    enable_multi_dimensional: bool = True
    enable_pattern_strength: bool = True
    enable_temporal_analysis: bool = True
    enable_pattern_categorization: bool = True
    enable_market_structure: bool = True
    
    # Consensus configuration
    min_models_required: int = 2
    min_agreement_threshold: float = 0.6
    min_confidence_threshold: float = 0.7
    consensus_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED
    
    # Feature integration
    feature_weighting: Dict[str, float] = None
    enable_feature_selection: bool = True
    max_features: int = 100
    
    # Quality control
    enable_quality_control: bool = True
    enable_pattern_validation: bool = True
    enable_consensus_validation: bool = True
    
    def __post_init__(self):
        if self.feature_weighting is None:
            self.feature_weighting = {
                'series_features': 0.25,
                'cross_timeframe': 0.20,
                'multi_dimensional': 0.25,
                'pattern_strength': 0.20,
                'temporal_analysis': 0.10
            }


class EnhancedConsensusSystem:
    """
    Enhanced consensus system with comprehensive candle feature analysis.
    
    This system integrates all advanced candle features to provide the most
    comprehensive and reliable trading indicators based on candlestick patterns.
    """
    
    def __init__(self, config: Optional[EnhancedConsensusConfig] = None):
        self.config = config or EnhancedConsensusConfig()
        
        # Initialize components
        self.advanced_feature_generator = None
        self.consensus_system = None
        self.comparison_pipeline = None
        
        # Feature storage
        self.feature_cache = {}
        self.feature_importance = {}
        self.consensus_indicators = {}
        
        # Performance tracking
        self.performance_stats = {
            'features_generated': 0,
            'consensus_indicators_generated': 0,
            'total_processing_time': 0.0,
            'feature_generation_time': 0.0,
            'consensus_generation_time': 0.0
        }
        
        self._initialize_components()
        logger.info("🚀 Enhanced Consensus System initialized")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize advanced feature generator
        advanced_config = AdvancedFeatureConfig(
            enable_series_features=self.config.enable_series_features,
            enable_cross_timeframe=self.config.enable_cross_timeframe,
            enable_multi_dimensional=self.config.enable_multi_dimensional,
            enable_pattern_strength=self.config.enable_pattern_strength,
            enable_temporal_analysis=self.config.enable_temporal_analysis,
            enable_pattern_categorization=self.config.enable_pattern_categorization,
            enable_market_structure=self.config.enable_market_structure
        )
        self.advanced_feature_generator = create_advanced_candle_feature_generator(
            advanced_config=advanced_config
        )
        
        # Initialize consensus system
        consensus_config = ConsensusSystemConfig(
            min_models_required=self.config.min_models_required,
            min_agreement_threshold=self.config.min_agreement_threshold,
            min_confidence_threshold=self.config.min_confidence_threshold,
            consensus_method=self.config.consensus_method,
            enable_quality_control=self.config.enable_quality_control
        )
        self.consensus_system = create_consensus_system(consensus_config)
        
        # Initialize comparison pipeline
        comparison_config = ConsensusConfig(
            consensus_method=self.config.consensus_method,
            min_agreement_threshold=self.config.min_agreement_threshold,
            min_models_required=self.config.min_models_required
        )
        self.comparison_pipeline = create_model_comparison_pipeline(comparison_config)
        
        logger.info("✅ All components initialized successfully")
    
    def train_enhanced_system(self, data: pd.DataFrame, 
                            target_column: str = 'future_return',
                            symbol: str = 'BTCUSDT',
                            model_types: List[ModelType] = None) -> Dict[str, Any]:
        """
        Train the enhanced consensus system with advanced features.
        
        Args:
            data: Historical OHLCV data
            target_column: Target variable column name
            symbol: Trading symbol
            model_types: List of model types to train
            
        Returns:
            Training results with advanced feature analysis
        """
        if model_types is None:
            model_types = [ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]
        
        logger.info(f"🔧 Training enhanced consensus system for {symbol}")
        start_time = time.time()
        
        training_results = {
            'symbol': symbol,
            'start_time': start_time,
            'data_samples': len(data),
            'model_types': [mt.value for mt in model_types],
            'advanced_features': {},
            'consensus_training': {},
            'feature_analysis': {},
            'performance_metrics': {}
        }
        
        try:
            # Step 1: Generate advanced features
            logger.info("📊 Step 1: Generating advanced candle features...")
            feature_start = time.time()
            
            advanced_features = self._generate_advanced_features(data)
            training_results['advanced_features'] = advanced_features
            
            feature_time = time.time() - feature_start
            self.performance_stats['feature_generation_time'] += feature_time
            self.performance_stats['features_generated'] += 1
            
            logger.info(f"✅ Advanced features generated in {feature_time:.2f}s")
            
            # Step 2: Train consensus system with enhanced data
            logger.info("🤝 Step 2: Training consensus system...")
            consensus_start = time.time()
            
            # Enhance data with advanced features
            enhanced_data = self._enhance_data_with_features(data, advanced_features)
            
            # Train consensus system
            consensus_results = self.consensus_system.train_consensus_system(
                enhanced_data, target_column, symbol, model_types
            )
            training_results['consensus_training'] = consensus_results
            
            consensus_time = time.time() - consensus_start
            self.performance_stats['consensus_generation_time'] += consensus_time
            self.performance_stats['consensus_indicators_generated'] += 1
            
            logger.info(f"✅ Consensus system trained in {consensus_time:.2f}s")
            
            # Step 3: Analyze feature importance and patterns
            logger.info("🔍 Step 3: Analyzing feature importance...")
            feature_analysis = self._analyze_feature_importance(enhanced_data, advanced_features)
            training_results['feature_analysis'] = feature_analysis
            
            # Step 4: Calculate performance metrics
            logger.info("📈 Step 4: Calculating performance metrics...")
            performance_metrics = self._calculate_performance_metrics(
                enhanced_data, advanced_features, consensus_results
            )
            training_results['performance_metrics'] = performance_metrics
            
            training_results['success'] = True
            
        except Exception as e:
            logger.error(f"❌ Enhanced system training failed: {e}")
            training_results['success'] = False
            training_results['error'] = str(e)
        
        training_results['end_time'] = time.time()
        training_results['total_time'] = training_results['end_time'] - start_time
        self.performance_stats['total_processing_time'] += training_results['total_time']
        
        logger.info(f"🎉 Enhanced system training completed in {training_results['total_time']:.2f}s")
        return training_results
    
    def _generate_advanced_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive advanced candle features."""
        feature_results = {
            'series_features': {},
            'cross_timeframe_features': {},
            'multi_dimensional_features': {},
            'pattern_strength_features': {},
            'temporal_features': {},
            'categorization_features': {},
            'market_structure_features': {},
            'combined_features': None,
            'feature_names': [],
            'feature_importance': {}
        }
        
        try:
            # Generate all types of advanced features
            if self.config.enable_series_features:
                series_features = self._generate_series_features(data)
                feature_results['series_features'] = series_features
            
            if self.config.enable_cross_timeframe:
                cross_timeframe_features = self._generate_cross_timeframe_features(data)
                feature_results['cross_timeframe_features'] = cross_timeframe_features
            
            if self.config.enable_multi_dimensional:
                multi_dimensional_features = self._generate_multi_dimensional_features(data)
                feature_results['multi_dimensional_features'] = multi_dimensional_features
            
            if self.config.enable_pattern_strength:
                pattern_strength_features = self._generate_pattern_strength_features(data)
                feature_results['pattern_strength_features'] = pattern_strength_features
            
            if self.config.enable_temporal_analysis:
                temporal_features = self._generate_temporal_features(data)
                feature_results['temporal_features'] = temporal_features
            
            if self.config.enable_pattern_categorization:
                categorization_features = self._generate_categorization_features(data)
                feature_results['categorization_features'] = categorization_features
            
            if self.config.enable_market_structure:
                market_structure_features = self._generate_market_structure_features(data)
                feature_results['market_structure_features'] = market_structure_features
            
            # Combine all features
            combined_features = self._combine_all_features(feature_results)
            feature_results['combined_features'] = combined_features
            
            # Generate feature names
            feature_names = self._generate_feature_names(feature_results)
            feature_results['feature_names'] = feature_names
            
            # Store in cache
            self.feature_cache[symbol] = feature_results
            
        except Exception as e:
            logger.error(f"Advanced feature generation failed: {e}")
            feature_results['error'] = str(e)
        
        return feature_results
    
    def _generate_series_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate series of candles features."""
        series_features = {}
        
        # Use the advanced feature generator
        series_data = self.advanced_feature_generator._generate_series_features(data)
        
        # Organize by feature type
        feature_names = [
            'consecutive_bullish_series',
            'consecutive_bearish_series',
            'alternating_pattern_series',
            'doji_series',
            'hammer_series',
            'engulfing_series',
            'momentum_series',
            'reversal_series',
            'series_strength',
            'series_consistency'
        ]
        
        for i, name in enumerate(feature_names):
            if i < len(series_data):
                series_features[name] = series_data[i]
        
        return series_features
    
    def _generate_cross_timeframe_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-timeframe features."""
        cross_timeframe_features = {}
        
        # Use the advanced feature generator
        cross_timeframe_data = self.advanced_feature_generator._generate_cross_timeframe_features(data)
        
        # Organize by timeframe and feature type
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        feature_types = ['trend', 'confluence', 'divergence']
        
        feature_idx = 0
        for tf in timeframes:
            for ft in feature_types:
                if feature_idx < len(cross_timeframe_data):
                    key = f"{tf}_{ft}"
                    cross_timeframe_features[key] = cross_timeframe_data[feature_idx]
                    feature_idx += 1
        
        return cross_timeframe_features
    
    def _generate_multi_dimensional_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate multi-dimensional interaction features."""
        multi_dimensional_features = {}
        
        # Use the advanced feature generator
        multi_dimensional_data = self.advanced_feature_generator._generate_multi_dimensional_features(data)
        
        # Organize by dimension and feature type
        dimensions = ['volume', 'momentum', 'volatility', 'trend']
        feature_types = ['correlation', 'interaction', 'breakout', 'exhaustion', 'divergence']
        
        feature_idx = 0
        for dim in dimensions:
            for ft in feature_types:
                if feature_idx < len(multi_dimensional_data):
                    key = f"{dim}_{ft}"
                    multi_dimensional_features[key] = multi_dimensional_data[feature_idx]
                    feature_idx += 1
        
        return multi_dimensional_features
    
    def _generate_pattern_strength_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate pattern strength and quality features."""
        pattern_strength_features = {}
        
        # Use the advanced feature generator
        pattern_strength_data = self.advanced_feature_generator._generate_pattern_strength_features(data)
        
        # Organize by strength metric
        strength_metrics = [
            'body_size_strength',
            'shadow_ratio_strength',
            'range_ratio_strength',
            'volume_ratio_strength',
            'momentum_ratio_strength',
            'pattern_quality',
            'pattern_reliability'
        ]
        
        for i, metric in enumerate(strength_metrics):
            if i < len(pattern_strength_data):
                pattern_strength_features[metric] = pattern_strength_data[i]
        
        return pattern_strength_features
    
    def _generate_temporal_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate temporal analysis features."""
        temporal_features = {}
        
        # Use the advanced feature generator
        temporal_data = self.advanced_feature_generator._generate_temporal_analysis_features(data)
        
        # Organize by temporal pattern
        temporal_patterns = [
            'time_of_day_patterns',
            'day_of_week_patterns',
            'seasonal_patterns',
            'cyclical_patterns',
            'trend_persistence'
        ]
        
        for i, pattern in enumerate(temporal_patterns):
            if i < len(temporal_data):
                temporal_features[pattern] = temporal_data[i]
        
        return temporal_features
    
    def _generate_categorization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate pattern categorization features."""
        categorization_features = {}
        
        # Use the advanced feature generator
        categorization_data = self.advanced_feature_generator._generate_pattern_categorization_features(data)
        
        # Organize by pattern category
        pattern_categories = [
            'bullish_patterns',
            'bearish_patterns',
            'reversal_patterns',
            'continuation_patterns'
        ]
        
        for i, category in enumerate(pattern_categories):
            if i < len(categorization_data):
                categorization_features[category] = categorization_data[i]
        
        return categorization_features
    
    def _generate_market_structure_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate market structure features."""
        market_structure_features = {}
        
        # Use the advanced feature generator
        market_structure_data = self.advanced_feature_generator._generate_market_structure_features(data)
        
        # Organize by market structure aspect
        structure_aspects = [
            'support_resistance',
            'trend_structure',
            'market_regime',
            'liquidity_zones'
        ]
        
        for i, aspect in enumerate(structure_aspects):
            if i < len(market_structure_data):
                market_structure_features[aspect] = market_structure_data[i]
        
        return market_structure_features
    
    def _combine_all_features(self, feature_results: Dict[str, Any]) -> np.ndarray:
        """Combine all features into a single feature matrix."""
        all_features = []
        
        # Combine features from all categories
        for category, features in feature_results.items():
            if isinstance(features, dict):
                for feature_name, feature_data in features.items():
                    if isinstance(feature_data, np.ndarray):
                        all_features.append(feature_data.reshape(-1, 1))
        
        if all_features:
            combined_features = np.hstack(all_features)
        else:
            combined_features = np.zeros((len(feature_results.get('series_features', {}).get('consecutive_bullish_series', [])), 1))
        
        return combined_features
    
    def _generate_feature_names(self, feature_results: Dict[str, Any]) -> List[str]:
        """Generate descriptive feature names."""
        feature_names = []
        
        for category, features in feature_results.items():
            if isinstance(features, dict):
                for feature_name, feature_data in features.items():
                    if isinstance(feature_data, np.ndarray):
                        feature_names.append(f"{category}_{feature_name}")
        
        return feature_names
    
    def _enhance_data_with_features(self, data: pd.DataFrame, 
                                  advanced_features: Dict[str, Any]) -> pd.DataFrame:
        """Enhance data with advanced features."""
        enhanced_data = data.copy()
        
        # Add advanced features as new columns
        if advanced_features.get('combined_features') is not None:
            combined_features = advanced_features['combined_features']
            feature_names = advanced_features.get('feature_names', [])
            
            # Add features as columns
            for i, feature_name in enumerate(feature_names):
                if i < combined_features.shape[1]:
                    enhanced_data[f'advanced_{feature_name}'] = combined_features[:, i]
        
        return enhanced_data
    
    def _analyze_feature_importance(self, enhanced_data: pd.DataFrame, 
                                  advanced_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance across all categories."""
        feature_analysis = {
            'category_importance': {},
            'top_features': [],
            'feature_correlations': {},
            'feature_stability': {}
        }
        
        try:
            # Analyze importance by category
            for category, features in advanced_features.items():
                if isinstance(features, dict) and features:
                    category_importance = []
                    
                    for feature_name, feature_data in features.items():
                        if isinstance(feature_data, np.ndarray) and len(feature_data) > 0:
                            # Calculate feature importance based on variance and correlation with price
                            variance = np.var(feature_data)
                            
                            # Correlation with price (if available)
                            if 'close' in enhanced_data.columns:
                                price_correlation = np.corrcoef(feature_data, enhanced_data['close'])[0, 1]
                                if not np.isnan(price_correlation):
                                    importance = variance * abs(price_correlation)
                                else:
                                    importance = variance
                            else:
                                importance = variance
                            
                            category_importance.append({
                                'feature': feature_name,
                                'importance': importance,
                                'variance': variance
                            })
                    
                    # Sort by importance
                    category_importance.sort(key=lambda x: x['importance'], reverse=True)
                    feature_analysis['category_importance'][category] = category_importance
            
            # Find top features across all categories
            all_features = []
            for category, features in feature_analysis['category_importance'].items():
                for feature in features:
                    all_features.append({
                        'category': category,
                        'feature': feature['feature'],
                        'importance': feature['importance']
                    })
            
            all_features.sort(key=lambda x: x['importance'], reverse=True)
            feature_analysis['top_features'] = all_features[:20]  # Top 20 features
            
            # Store feature importance
            self.feature_importance = feature_analysis
            
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            feature_analysis['error'] = str(e)
        
        return feature_analysis
    
    def _calculate_performance_metrics(self, enhanced_data: pd.DataFrame, 
                                     advanced_features: Dict[str, Any],
                                     consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        performance_metrics = {
            'feature_metrics': {},
            'consensus_metrics': {},
            'integration_metrics': {}
        }
        
        try:
            # Feature metrics
            if advanced_features.get('combined_features') is not None:
                combined_features = advanced_features['combined_features']
                
                performance_metrics['feature_metrics'] = {
                    'total_features': combined_features.shape[1],
                    'feature_variance': np.var(combined_features, axis=0).mean(),
                    'feature_correlation': np.corrcoef(combined_features.T).mean(),
                    'feature_stability': self._calculate_feature_stability(combined_features)
                }
            
            # Consensus metrics
            if consensus_results.get('consensus_analysis'):
                consensus_analysis = consensus_results['consensus_analysis']
                
                performance_metrics['consensus_metrics'] = {
                    'models_used': len(consensus_analysis.get('models_used', [])),
                    'consensus_method': consensus_analysis.get('consensus_method', 'unknown'),
                    'agreement_scores': consensus_analysis.get('agreement_scores', {}),
                    'confidence_scores': consensus_analysis.get('confidence_scores', {})
                }
            
            # Integration metrics
            performance_metrics['integration_metrics'] = {
                'feature_integration_success': True,
                'consensus_integration_success': consensus_results.get('success', False),
                'total_processing_time': self.performance_stats['total_processing_time'],
                'feature_generation_efficiency': self.performance_stats['features_generated'] / max(self.performance_stats['feature_generation_time'], 1),
                'consensus_generation_efficiency': self.performance_stats['consensus_indicators_generated'] / max(self.performance_stats['consensus_generation_time'], 1)
            }
            
        except Exception as e:
            logger.warning(f"Performance metrics calculation failed: {e}")
            performance_metrics['error'] = str(e)
        
        return performance_metrics
    
    def _calculate_feature_stability(self, features: np.ndarray) -> float:
        """Calculate feature stability over time."""
        if features.shape[0] < 10:
            return 0.0
        
        # Calculate rolling correlation with previous period
        stability_scores = []
        window_size = min(10, features.shape[0] // 2)
        
        for i in range(window_size, features.shape[0]):
            current_window = features[i-window_size:i]
            previous_window = features[i-window_size*2:i-window_size]
            
            if previous_window.shape[0] == current_window.shape[0]:
                # Calculate correlation between windows
                correlations = []
                for j in range(features.shape[1]):
                    if len(current_window[:, j]) > 1 and len(previous_window[:, j]) > 1:
                        corr = np.corrcoef(current_window[:, j], previous_window[:, j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if correlations:
                    stability_scores.append(np.mean(correlations))
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def generate_enhanced_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate enhanced consensus indicators with advanced features.
        
        Args:
            data: New OHLCV data for indicator generation
            
        Returns:
            Enhanced consensus indicators with feature analysis
        """
        logger.info("🔮 Generating enhanced consensus indicators...")
        start_time = time.time()
        
        try:
            # Generate advanced features for new data
            advanced_features = self._generate_advanced_features(data)
            
            # Enhance data with features
            enhanced_data = self._enhance_data_with_features(data, advanced_features)
            
            # Generate consensus indicators
            consensus_indicators = self.consensus_system.generate_consensus_indicators(enhanced_data)
            
            # Add feature analysis to results
            enhanced_results = {
                'consensus_indicators': consensus_indicators,
                'advanced_features': advanced_features,
                'feature_analysis': self._analyze_feature_importance(enhanced_data, advanced_features),
                'generation_time': time.time() - start_time,
                'success': True
            }
            
            # Store consensus indicators
            self.consensus_indicators = consensus_indicators
            
            logger.info(f"✅ Enhanced indicators generated in {enhanced_results['generation_time']:.2f}s")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced indicator generation failed: {e}")
            return {
                'error': str(e),
                'success': False,
                'generation_time': time.time() - start_time
            }
    
    def get_enhanced_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the enhanced system."""
        summary = {
            'system_config': {
                'enable_series_features': self.config.enable_series_features,
                'enable_cross_timeframe': self.config.enable_cross_timeframe,
                'enable_multi_dimensional': self.config.enable_multi_dimensional,
                'enable_pattern_strength': self.config.enable_pattern_strength,
                'min_models_required': self.config.min_models_required,
                'consensus_method': self.config.consensus_method.value
            },
            'performance_stats': self.performance_stats,
            'feature_importance': self.feature_importance,
            'consensus_indicators_count': len(self.consensus_indicators),
            'feature_cache_size': len(self.feature_cache)
        }
        
        return summary


def create_enhanced_consensus_system(config: Optional[EnhancedConsensusConfig] = None) -> EnhancedConsensusSystem:
    """Create an enhanced consensus system with advanced features."""
    return EnhancedConsensusSystem(config)


def test_enhanced_consensus_system():
    """Test function for the enhanced consensus system."""
    print("🧪 Testing Enhanced Consensus System...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create enhanced consensus system
    config = EnhancedConsensusConfig(
        enable_series_features=True,
        enable_cross_timeframe=True,
        enable_multi_dimensional=True,
        enable_pattern_strength=True,
        enable_temporal_analysis=True,
        enable_pattern_categorization=True,
        enable_market_structure=True
    )
    
    system = create_enhanced_consensus_system(config)
    
    # Train enhanced system
    print("🔧 Training enhanced consensus system...")
    training_results = system.train_enhanced_system(
        data, 
        symbol='BTCUSDT',
        model_types=[ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]
    )
    
    # Display training results
    print("\n📊 Enhanced System Training Results:")
    print(f"   Training time: {training_results['total_time']:.2f} seconds")
    print(f"   Data samples: {training_results['data_samples']}")
    print(f"   Model types: {training_results['model_types']}")
    print(f"   Success: {'✅' if training_results['success'] else '❌'}")
    
    if training_results['success']:
        # Display advanced features
        advanced_features = training_results['advanced_features']
        print(f"\n🔧 Advanced Features Generated:")
        for category, features in advanced_features.items():
            if isinstance(features, dict):
                print(f"   {category}: {len(features)} features")
        
        # Display feature analysis
        feature_analysis = training_results['feature_analysis']
        if 'top_features' in feature_analysis:
            print(f"\n🏆 Top Features:")
            for i, feature in enumerate(feature_analysis['top_features'][:5]):
                print(f"   {i+1}. {feature['category']}_{feature['feature']}: {feature['importance']:.4f}")
        
        # Display performance metrics
        performance_metrics = training_results['performance_metrics']
        if 'feature_metrics' in performance_metrics:
            fm = performance_metrics['feature_metrics']
            print(f"\n📈 Feature Metrics:")
            print(f"   Total features: {fm.get('total_features', 'N/A')}")
            print(f"   Feature variance: {fm.get('feature_variance', 'N/A'):.4f}")
            print(f"   Feature stability: {fm.get('feature_stability', 'N/A'):.4f}")
    
    # Generate enhanced indicators
    print("\n🔮 Generating enhanced indicators...")
    new_data = data.iloc[-100:]  # Use last 100 samples as new data
    enhanced_indicators = system.generate_enhanced_indicators(new_data)
    
    if enhanced_indicators['success']:
        print(f"   Indicators generated: {len(enhanced_indicators['consensus_indicators'])}")
        print(f"   Generation time: {enhanced_indicators['generation_time']:.2f}s")
        
        # Display feature analysis
        if 'feature_analysis' in enhanced_indicators:
            fa = enhanced_indicators['feature_analysis']
            if 'top_features' in fa:
                print(f"   Top features: {len(fa['top_features'])}")
    else:
        print(f"   Error: {enhanced_indicators.get('error', 'Unknown error')}")
    
    # Get system summary
    summary = system.get_enhanced_system_summary()
    print(f"\n📋 System Summary:")
    print(f"   Features generated: {summary['performance_stats']['features_generated']}")
    print(f"   Consensus indicators: {summary['performance_stats']['consensus_indicators_generated']}")
    print(f"   Total processing time: {summary['performance_stats']['total_processing_time']:.2f}s")
    
    print("\n🎉 Enhanced Consensus System test completed successfully!")
    return system, training_results


if __name__ == "__main__":
    test_enhanced_consensus_system()