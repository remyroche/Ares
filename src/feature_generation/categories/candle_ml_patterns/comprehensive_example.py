"""
Comprehensive Example: Advanced Candle Features with Consensus ML Indicators

This module provides a comprehensive example demonstrating all advanced candle features
including series patterns, cross-timeframe analysis, multi-dimensional interactions,
and pattern strength assessment integrated with consensus-based ML indicators.

Key Features Demonstrated:
- Series of candles (consecutive patterns, sequences)
- Cross-timeframe candle analysis
- Multi-dimensional interactions (volume, momentum, volatility)
- Pattern strength and quality assessment
- Consensus-based ML indicator generation
- Comprehensive feature analysis and validation
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Core imports
from .enhanced_consensus_system import (
    EnhancedConsensusSystem, EnhancedConsensusConfig, create_enhanced_consensus_system
)
from .advanced_candle_features import (
    AdvancedCandleFeatureGenerator, AdvancedFeatureConfig, create_advanced_candle_feature_generator
)
from .ml_candle_pattern_indicators import ModelType, IndicatorType

logger = logging.getLogger(__name__)


class ComprehensiveCandleAnalysis:
    """
    Comprehensive analysis of candle patterns with advanced features and ML consensus.
    
    This class demonstrates the complete workflow from basic candle pattern detection
    to advanced feature engineering and consensus-based ML indicator generation.
    """
    
    def __init__(self):
        self.enhanced_system = None
        self.advanced_feature_generator = None
        self.analysis_results = {}
        
        logger.info("🎯 Comprehensive Candle Analysis initialized")
    
    def run_complete_analysis(self, data: pd.DataFrame, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """
        Run complete comprehensive analysis.
        
        Args:
            data: Historical OHLCV data
            symbol: Trading symbol
            
        Returns:
            Complete analysis results
        """
        logger.info(f"🚀 Starting comprehensive candle analysis for {symbol}")
        start_time = time.time()
        
        analysis_results = {
            'symbol': symbol,
            'start_time': start_time,
            'data_samples': len(data),
            'analysis_steps': [],
            'results': {}
        }
        
        try:
            # Step 1: Advanced Feature Analysis
            logger.info("📊 Step 1: Advanced Feature Analysis...")
            feature_analysis = self._analyze_advanced_features(data)
            analysis_results['analysis_steps'].append('advanced_features')
            analysis_results['results']['advanced_features'] = feature_analysis
            
            # Step 2: Series Pattern Analysis
            logger.info("🔗 Step 2: Series Pattern Analysis...")
            series_analysis = self._analyze_series_patterns(data)
            analysis_results['analysis_steps'].append('series_patterns')
            analysis_results['results']['series_patterns'] = series_analysis
            
            # Step 3: Cross-Timeframe Analysis
            logger.info("⏰ Step 3: Cross-Timeframe Analysis...")
            cross_timeframe_analysis = self._analyze_cross_timeframe(data)
            analysis_results['analysis_steps'].append('cross_timeframe')
            analysis_results['results']['cross_timeframe'] = cross_timeframe_analysis
            
            # Step 4: Multi-Dimensional Analysis
            logger.info("🎯 Step 4: Multi-Dimensional Analysis...")
            multi_dimensional_analysis = self._analyze_multi_dimensional(data)
            analysis_results['analysis_steps'].append('multi_dimensional')
            analysis_results['results']['multi_dimensional'] = multi_dimensional_analysis
            
            # Step 5: Pattern Strength Analysis
            logger.info("💪 Step 5: Pattern Strength Analysis...")
            pattern_strength_analysis = self._analyze_pattern_strength(data)
            analysis_results['analysis_steps'].append('pattern_strength')
            analysis_results['results']['pattern_strength'] = pattern_strength_analysis
            
            # Step 6: Enhanced Consensus System Training
            logger.info("🤝 Step 6: Enhanced Consensus System Training...")
            consensus_analysis = self._train_enhanced_consensus_system(data, symbol)
            analysis_results['analysis_steps'].append('consensus_training')
            analysis_results['results']['consensus_training'] = consensus_analysis
            
            # Step 7: Comprehensive Validation
            logger.info("✅ Step 7: Comprehensive Validation...")
            validation_analysis = self._validate_comprehensive_analysis(data, analysis_results)
            analysis_results['analysis_steps'].append('validation')
            analysis_results['results']['validation'] = validation_analysis
            
            # Step 8: Performance Evaluation
            logger.info("📈 Step 8: Performance Evaluation...")
            performance_analysis = self._evaluate_performance(data, analysis_results)
            analysis_results['analysis_steps'].append('performance_evaluation')
            analysis_results['results']['performance_evaluation'] = performance_analysis
            
            analysis_results['success'] = True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {e}")
            analysis_results['success'] = False
            analysis_results['error'] = str(e)
        
        analysis_results['end_time'] = time.time()
        analysis_results['total_time'] = analysis_results['end_time'] - start_time
        
        logger.info(f"🎉 Comprehensive analysis completed in {analysis_results['total_time']:.2f}s")
        return analysis_results
    
    def _analyze_advanced_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze advanced candle features."""
        # Create advanced feature generator
        advanced_config = AdvancedFeatureConfig(
            enable_series_features=True,
            enable_cross_timeframe=True,
            enable_multi_dimensional=True,
            enable_pattern_strength=True,
            enable_temporal_analysis=True,
            enable_pattern_categorization=True,
            enable_market_structure=True
        )
        
        self.advanced_feature_generator = create_advanced_candle_feature_generator(
            advanced_config=advanced_config
        )
        
        # Generate features
        features = self.advanced_feature_generator._generate_feature(data)
        
        # Analyze feature characteristics
        feature_analysis = {
            'total_features': len(features),
            'feature_statistics': {
                'mean': features.mean(),
                'std': features.std(),
                'min': features.min(),
                'max': features.max(),
                'non_zero': (features != 0).sum()
            },
            'feature_distribution': self._analyze_feature_distribution(features),
            'feature_correlations': self._analyze_feature_correlations(features, data),
            'feature_temporal_patterns': self._analyze_temporal_patterns(features)
        }
        
        return feature_analysis
    
    def _analyze_series_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze series of candle patterns."""
        series_analysis = {}
        
        # Generate series features
        series_features = self.advanced_feature_generator._generate_series_features(data)
        
        # Analyze each series type
        series_types = [
            'consecutive_bullish_series',
            'consecutive_bearish_series',
            'alternating_pattern_series',
            'doji_series',
            'hammer_series',
            'engulfing_series',
            'momentum_series',
            'reversal_series'
        ]
        
        for i, series_type in enumerate(series_types):
            if i < len(series_features):
                feature_data = series_features[i]
                
                series_analysis[series_type] = {
                    'count': (feature_data > 0).sum(),
                    'max_length': feature_data.max(),
                    'avg_length': feature_data[feature_data > 0].mean() if (feature_data > 0).sum() > 0 else 0,
                    'frequency': (feature_data > 0).sum() / len(feature_data),
                    'strength_distribution': self._analyze_strength_distribution(feature_data)
                }
        
        # Overall series analysis
        series_analysis['overall'] = {
            'total_series_detected': sum((feature > 0).sum() for feature in series_features),
            'series_diversity': len([f for f in series_features if (f > 0).sum() > 0]),
            'series_consistency': self._calculate_series_consistency(series_features)
        }
        
        return series_analysis
    
    def _analyze_cross_timeframe(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cross-timeframe patterns."""
        cross_timeframe_analysis = {}
        
        # Generate cross-timeframe features
        cross_timeframe_features = self.advanced_feature_generator._generate_cross_timeframe_features(data)
        
        # Analyze by timeframe
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        feature_types = ['trend', 'confluence', 'divergence']
        
        feature_idx = 0
        for tf in timeframes:
            cross_timeframe_analysis[tf] = {}
            
            for ft in feature_types:
                if feature_idx < len(cross_timeframe_features):
                    feature_data = cross_timeframe_features[feature_idx]
                    
                    cross_timeframe_analysis[tf][ft] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'trend_strength': abs(feature_data.mean()),
                        'volatility': feature_data.std(),
                        'consistency': 1.0 - (feature_data.std() / (abs(feature_data.mean()) + 1e-8))
                    }
                    
                    feature_idx += 1
        
        # Overall cross-timeframe analysis
        cross_timeframe_analysis['overall'] = {
            'timeframe_confluence': self._calculate_timeframe_confluence(cross_timeframe_analysis),
            'timeframe_divergence': self._calculate_timeframe_divergence(cross_timeframe_analysis),
            'multi_timeframe_trend_strength': self._calculate_multi_timeframe_trend_strength(cross_timeframe_analysis)
        }
        
        return cross_timeframe_analysis
    
    def _analyze_multi_dimensional(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze multi-dimensional interactions."""
        multi_dimensional_analysis = {}
        
        # Generate multi-dimensional features
        multi_dimensional_features = self.advanced_feature_generator._generate_multi_dimensional_features(data)
        
        # Analyze by dimension
        dimensions = ['volume', 'momentum', 'volatility', 'trend']
        feature_types = ['correlation', 'interaction', 'breakout', 'exhaustion', 'divergence']
        
        feature_idx = 0
        for dim in dimensions:
            multi_dimensional_analysis[dim] = {}
            
            for ft in feature_types:
                if feature_idx < len(multi_dimensional_features):
                    feature_data = multi_dimensional_features[feature_idx]
                    
                    multi_dimensional_analysis[dim][ft] = {
                        'strength': abs(feature_data.mean()),
                        'consistency': 1.0 - (feature_data.std() / (abs(feature_data.mean()) + 1e-8)),
                        'frequency': (feature_data != 0).sum() / len(feature_data),
                        'correlation_with_price': self._calculate_price_correlation(feature_data, data['close'])
                    }
                    
                    feature_idx += 1
        
        # Overall multi-dimensional analysis
        multi_dimensional_analysis['overall'] = {
            'dimension_interaction_strength': self._calculate_dimension_interaction_strength(multi_dimensional_analysis),
            'cross_dimension_correlation': self._calculate_cross_dimension_correlation(multi_dimensional_analysis),
            'multi_dimensional_consensus': self._calculate_multi_dimensional_consensus(multi_dimensional_analysis)
        }
        
        return multi_dimensional_analysis
    
    def _analyze_pattern_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern strength and quality."""
        pattern_strength_analysis = {}
        
        # Generate pattern strength features
        pattern_strength_features = self.advanced_feature_generator._generate_pattern_strength_features(data)
        
        # Analyze each strength metric
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
            if i < len(pattern_strength_features):
                feature_data = pattern_strength_features[i]
                
                pattern_strength_analysis[metric] = {
                    'mean_strength': feature_data.mean(),
                    'max_strength': feature_data.max(),
                    'strength_distribution': self._analyze_strength_distribution(feature_data),
                    'high_quality_ratio': (feature_data > 0.7).sum() / len(feature_data),
                    'medium_quality_ratio': ((feature_data > 0.4) & (feature_data <= 0.7)).sum() / len(feature_data),
                    'low_quality_ratio': (feature_data <= 0.4).sum() / len(feature_data)
                }
        
        # Overall pattern strength analysis
        pattern_strength_analysis['overall'] = {
            'average_pattern_quality': np.mean([analysis['mean_strength'] for analysis in pattern_strength_analysis.values() if isinstance(analysis, dict)]),
            'pattern_consistency': self._calculate_pattern_consistency(pattern_strength_analysis),
            'quality_trend': self._calculate_quality_trend(pattern_strength_analysis)
        }
        
        return pattern_strength_analysis
    
    def _train_enhanced_consensus_system(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train the enhanced consensus system."""
        # Create enhanced consensus system
        config = EnhancedConsensusConfig(
            enable_series_features=True,
            enable_cross_timeframe=True,
            enable_multi_dimensional=True,
            enable_pattern_strength=True,
            enable_temporal_analysis=True,
            enable_pattern_categorization=True,
            enable_market_structure=True,
            min_models_required=2,
            min_agreement_threshold=0.6,
            consensus_method=ConsensusMethod.CONFIDENCE_WEIGHTED
        )
        
        self.enhanced_system = create_enhanced_consensus_system(config)
        
        # Train system
        training_results = self.enhanced_system.train_enhanced_system(
            data, 
            symbol=symbol,
            model_types=[ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]
        )
        
        return training_results
    
    def _validate_comprehensive_analysis(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the comprehensive analysis."""
        validation_results = {
            'feature_validation': {},
            'consensus_validation': {},
            'integration_validation': {},
            'overall_validation': {}
        }
        
        try:
            # Feature validation
            if 'advanced_features' in analysis_results['results']:
                af = analysis_results['results']['advanced_features']
                validation_results['feature_validation'] = {
                    'features_generated': af.get('total_features', 0) > 0,
                    'feature_quality': af.get('feature_statistics', {}).get('non_zero', 0) > 0,
                    'feature_diversity': len(af.get('feature_distribution', {})) > 0
                }
            
            # Consensus validation
            if 'consensus_training' in analysis_results['results']:
                ct = analysis_results['results']['consensus_training']
                validation_results['consensus_validation'] = {
                    'training_success': ct.get('success', False),
                    'models_trained': len(ct.get('consensus_training', {}).get('comparison_results', {}).get('model_results', {})),
                    'consensus_generated': 'consensus_analysis' in ct.get('consensus_training', {})
                }
            
            # Integration validation
            validation_results['integration_validation'] = {
                'advanced_features_integrated': 'advanced_features' in analysis_results['results'],
                'series_patterns_integrated': 'series_patterns' in analysis_results['results'],
                'cross_timeframe_integrated': 'cross_timeframe' in analysis_results['results'],
                'multi_dimensional_integrated': 'multi_dimensional' in analysis_results['results'],
                'pattern_strength_integrated': 'pattern_strength' in analysis_results['results']
            }
            
            # Overall validation
            validation_results['overall_validation'] = {
                'all_steps_completed': len(analysis_results['analysis_steps']) >= 7,
                'analysis_successful': analysis_results.get('success', False),
                'comprehensive_coverage': all([
                    validation_results['feature_validation'].get('features_generated', False),
                    validation_results['consensus_validation'].get('training_success', False),
                    validation_results['integration_validation'].get('advanced_features_integrated', False)
                ])
            }
            
        except Exception as e:
            logger.warning(f"Validation analysis failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _evaluate_performance(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall performance of the comprehensive analysis."""
        performance_results = {
            'processing_efficiency': {},
            'feature_effectiveness': {},
            'consensus_effectiveness': {},
            'overall_performance': {}
        }
        
        try:
            # Processing efficiency
            total_time = analysis_results.get('total_time', 0)
            data_samples = analysis_results.get('data_samples', 0)
            
            performance_results['processing_efficiency'] = {
                'total_processing_time': total_time,
                'samples_per_second': data_samples / max(total_time, 1),
                'analysis_steps_completed': len(analysis_results.get('analysis_steps', [])),
                'time_per_step': total_time / max(len(analysis_results.get('analysis_steps', [])), 1)
            }
            
            # Feature effectiveness
            if 'advanced_features' in analysis_results['results']:
                af = analysis_results['results']['advanced_features']
                performance_results['feature_effectiveness'] = {
                    'total_features': af.get('total_features', 0),
                    'feature_utilization': af.get('feature_statistics', {}).get('non_zero', 0) / max(af.get('total_features', 1), 1),
                    'feature_diversity': len(af.get('feature_distribution', {}))
                }
            
            # Consensus effectiveness
            if 'consensus_training' in analysis_results['results']:
                ct = analysis_results['results']['consensus_training']
                performance_results['consensus_effectiveness'] = {
                    'training_success': ct.get('success', False),
                    'models_used': len(ct.get('consensus_training', {}).get('comparison_results', {}).get('model_results', {})),
                    'consensus_quality': ct.get('consensus_training', {}).get('quality_validation', {}).get('validation_passed', False)
                }
            
            # Overall performance
            performance_results['overall_performance'] = {
                'comprehensive_analysis_score': self._calculate_comprehensive_score(analysis_results),
                'feature_engineering_score': self._calculate_feature_engineering_score(analysis_results),
                'consensus_ml_score': self._calculate_consensus_ml_score(analysis_results),
                'integration_score': self._calculate_integration_score(analysis_results)
            }
            
        except Exception as e:
            logger.warning(f"Performance evaluation failed: {e}")
            performance_results['error'] = str(e)
        
        return performance_results
    
    # Helper methods for analysis
    def _analyze_feature_distribution(self, features: pd.Series) -> Dict[str, Any]:
        """Analyze feature distribution."""
        return {
            'skewness': features.skew(),
            'kurtosis': features.kurtosis(),
            'percentiles': {
                '25th': features.quantile(0.25),
                '50th': features.quantile(0.50),
                '75th': features.quantile(0.75),
                '90th': features.quantile(0.90),
                '95th': features.quantile(0.95)
            }
        }
    
    def _analyze_feature_correlations(self, features: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze feature correlations with price."""
        correlations = {}
        
        if 'close' in data.columns:
            correlations['price_correlation'] = features.corr(data['close'])
        
        if 'volume' in data.columns:
            correlations['volume_correlation'] = features.corr(data['volume'])
        
        return correlations
    
    def _analyze_temporal_patterns(self, features: pd.Series) -> Dict[str, Any]:
        """Analyze temporal patterns in features."""
        # Simple temporal analysis
        window_size = min(50, len(features) // 4)
        if window_size < 10:
            return {}
        
        rolling_mean = features.rolling(window_size).mean()
        rolling_std = features.rolling(window_size).std()
        
        return {
            'temporal_trend': rolling_mean.iloc[-1] - rolling_mean.iloc[window_size],
            'temporal_volatility': rolling_std.mean(),
            'temporal_stability': 1.0 - (rolling_std.std() / (rolling_std.mean() + 1e-8))
        }
    
    def _analyze_strength_distribution(self, feature_data: np.ndarray) -> Dict[str, Any]:
        """Analyze strength distribution of features."""
        if len(feature_data) == 0:
            return {}
        
        return {
            'high_strength': (feature_data > 0.7).sum(),
            'medium_strength': ((feature_data > 0.4) & (feature_data <= 0.7)).sum(),
            'low_strength': (feature_data <= 0.4).sum(),
            'strength_consistency': 1.0 - (feature_data.std() / (abs(feature_data.mean()) + 1e-8))
        }
    
    def _calculate_series_consistency(self, series_features: List[np.ndarray]) -> float:
        """Calculate consistency across series features."""
        if not series_features:
            return 0.0
        
        # Calculate correlation between different series types
        correlations = []
        for i in range(len(series_features)):
            for j in range(i + 1, len(series_features)):
                corr = np.corrcoef(series_features[i], series_features[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_timeframe_confluence(self, cross_timeframe_analysis: Dict[str, Any]) -> float:
        """Calculate confluence between timeframes."""
        # Simplified confluence calculation
        confluence_scores = []
        
        for tf, features in cross_timeframe_analysis.items():
            if tf != 'overall' and isinstance(features, dict):
                tf_scores = []
                for ft, data in features.items():
                    if isinstance(data, dict) and 'trend_strength' in data:
                        tf_scores.append(data['trend_strength'])
                
                if tf_scores:
                    confluence_scores.append(np.mean(tf_scores))
        
        return np.mean(confluence_scores) if confluence_scores else 0.0
    
    def _calculate_timeframe_divergence(self, cross_timeframe_analysis: Dict[str, Any]) -> float:
        """Calculate divergence between timeframes."""
        # Simplified divergence calculation
        divergence_scores = []
        
        timeframes = [tf for tf in cross_timeframe_analysis.keys() if tf != 'overall']
        
        for i in range(len(timeframes)):
            for j in range(i + 1, len(timeframes)):
                tf1 = timeframes[i]
                tf2 = timeframes[j]
                
                if (tf1 in cross_timeframe_analysis and tf2 in cross_timeframe_analysis and
                    isinstance(cross_timeframe_analysis[tf1], dict) and 
                    isinstance(cross_timeframe_analysis[tf2], dict)):
                    
                    # Calculate divergence between timeframes
                    tf1_trend = cross_timeframe_analysis[tf1].get('trend', {}).get('trend_strength', 0)
                    tf2_trend = cross_timeframe_analysis[tf2].get('trend', {}).get('trend_strength', 0)
                    
                    divergence = abs(tf1_trend - tf2_trend)
                    divergence_scores.append(divergence)
        
        return np.mean(divergence_scores) if divergence_scores else 0.0
    
    def _calculate_multi_timeframe_trend_strength(self, cross_timeframe_analysis: Dict[str, Any]) -> float:
        """Calculate multi-timeframe trend strength."""
        trend_strengths = []
        
        for tf, features in cross_timeframe_analysis.items():
            if tf != 'overall' and isinstance(features, dict):
                trend_data = features.get('trend', {})
                if isinstance(trend_data, dict) and 'trend_strength' in trend_data:
                    trend_strengths.append(trend_data['trend_strength'])
        
        return np.mean(trend_strengths) if trend_strengths else 0.0
    
    def _calculate_price_correlation(self, feature_data: np.ndarray, price_data: pd.Series) -> float:
        """Calculate correlation with price."""
        if len(feature_data) != len(price_data):
            return 0.0
        
        corr = np.corrcoef(feature_data, price_data)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def _calculate_dimension_interaction_strength(self, multi_dimensional_analysis: Dict[str, Any]) -> float:
        """Calculate dimension interaction strength."""
        interaction_strengths = []
        
        for dim, features in multi_dimensional_analysis.items():
            if dim != 'overall' and isinstance(features, dict):
                dim_strengths = []
                for ft, data in features.items():
                    if isinstance(data, dict) and 'strength' in data:
                        dim_strengths.append(data['strength'])
                
                if dim_strengths:
                    interaction_strengths.append(np.mean(dim_strengths))
        
        return np.mean(interaction_strengths) if interaction_strengths else 0.0
    
    def _calculate_cross_dimension_correlation(self, multi_dimensional_analysis: Dict[str, Any]) -> float:
        """Calculate cross-dimension correlation."""
        # Simplified cross-dimension correlation
        dimensions = [dim for dim in multi_dimensional_analysis.keys() if dim != 'overall']
        
        if len(dimensions) < 2:
            return 0.0
        
        # Calculate average strength for each dimension
        dimension_strengths = []
        for dim in dimensions:
            features = multi_dimensional_analysis[dim]
            if isinstance(features, dict):
                strengths = []
                for ft, data in features.items():
                    if isinstance(data, dict) and 'strength' in data:
                        strengths.append(data['strength'])
                
                if strengths:
                    dimension_strengths.append(np.mean(strengths))
        
        if len(dimension_strengths) < 2:
            return 0.0
        
        # Calculate correlation between dimension strengths
        corr = np.corrcoef(dimension_strengths)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def _calculate_multi_dimensional_consensus(self, multi_dimensional_analysis: Dict[str, Any]) -> float:
        """Calculate multi-dimensional consensus."""
        # Simplified consensus calculation
        consensus_scores = []
        
        for dim, features in multi_dimensional_analysis.items():
            if dim != 'overall' and isinstance(features, dict):
                dim_consensus = []
                for ft, data in features.items():
                    if isinstance(data, dict) and 'consistency' in data:
                        dim_consensus.append(data['consistency'])
                
                if dim_consensus:
                    consensus_scores.append(np.mean(dim_consensus))
        
        return np.mean(consensus_scores) if consensus_scores else 0.0
    
    def _calculate_pattern_consistency(self, pattern_strength_analysis: Dict[str, Any]) -> float:
        """Calculate pattern consistency."""
        consistency_scores = []
        
        for metric, data in pattern_strength_analysis.items():
            if metric != 'overall' and isinstance(data, dict) and 'strength_distribution' in data:
                strength_dist = data['strength_distribution']
                if isinstance(strength_dist, dict) and 'strength_consistency' in strength_dist:
                    consistency_scores.append(strength_dist['strength_consistency'])
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_quality_trend(self, pattern_strength_analysis: Dict[str, Any]) -> float:
        """Calculate quality trend over time."""
        # Simplified quality trend calculation
        quality_scores = []
        
        for metric, data in pattern_strength_analysis.items():
            if metric != 'overall' and isinstance(data, dict) and 'mean_strength' in data:
                quality_scores.append(data['mean_strength'])
        
        if len(quality_scores) < 2:
            return 0.0
        
        # Calculate trend in quality scores
        x = np.arange(len(quality_scores))
        slope, _, _, _, _ = np.polyfit(x, quality_scores, 1)
        
        return slope
    
    def _calculate_comprehensive_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate comprehensive analysis score."""
        scores = []
        
        # Feature analysis score
        if 'advanced_features' in analysis_results['results']:
            af = analysis_results['results']['advanced_features']
            feature_score = min(af.get('total_features', 0) / 50, 1.0)  # Normalize to 50 features
            scores.append(feature_score)
        
        # Series patterns score
        if 'series_patterns' in analysis_results['results']:
            sp = analysis_results['results']['series_patterns']
            series_score = min(sp.get('overall', {}).get('series_diversity', 0) / 8, 1.0)  # Normalize to 8 types
            scores.append(series_score)
        
        # Consensus training score
        if 'consensus_training' in analysis_results['results']:
            ct = analysis_results['results']['consensus_training']
            consensus_score = 1.0 if ct.get('success', False) else 0.0
            scores.append(consensus_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_feature_engineering_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate feature engineering score."""
        scores = []
        
        # Advanced features score
        if 'advanced_features' in analysis_results['results']:
            af = analysis_results['results']['advanced_features']
            scores.append(min(af.get('total_features', 0) / 100, 1.0))
        
        # Series patterns score
        if 'series_patterns' in analysis_results['results']:
            sp = analysis_results['results']['series_patterns']
            scores.append(min(sp.get('overall', {}).get('series_diversity', 0) / 8, 1.0))
        
        # Multi-dimensional score
        if 'multi_dimensional' in analysis_results['results']:
            md = analysis_results['results']['multi_dimensional']
            scores.append(min(len(md) / 5, 1.0))  # 5 dimensions
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_consensus_ml_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate consensus ML score."""
        if 'consensus_training' not in analysis_results['results']:
            return 0.0
        
        ct = analysis_results['results']['consensus_training']
        if not ct.get('success', False):
            return 0.0
        
        # Calculate score based on consensus quality
        consensus_training = ct.get('consensus_training', {})
        quality_validation = consensus_training.get('quality_validation', {})
        
        scores = []
        
        # Training success
        scores.append(1.0)
        
        # Quality validation
        if quality_validation.get('validation_passed', False):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Consensus strength
        consensus_strength = quality_validation.get('consensus_strength', 0)
        scores.append(min(consensus_strength * 2, 1.0))
        
        return np.mean(scores)
    
    def _calculate_integration_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate integration score."""
        if 'validation' not in analysis_results['results']:
            return 0.0
        
        validation = analysis_results['results']['validation']
        integration_validation = validation.get('integration_validation', {})
        
        integration_checks = [
            integration_validation.get('advanced_features_integrated', False),
            integration_validation.get('series_patterns_integrated', False),
            integration_validation.get('cross_timeframe_integrated', False),
            integration_validation.get('multi_dimensional_integrated', False),
            integration_validation.get('pattern_strength_integrated', False)
        ]
        
        return sum(integration_checks) / len(integration_checks)
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report."""
        report = f"""
# Comprehensive Candle Analysis Report

## Overview
- **Symbol**: {analysis_results['symbol']}
- **Data Samples**: {analysis_results['data_samples']}
- **Total Time**: {analysis_results['total_time']:.2f} seconds
- **Success**: {'✅' if analysis_results['success'] else '❌'}

## Analysis Steps Completed
{chr(10).join(f"- {step}" for step in analysis_results['analysis_steps'])}

## Results Summary

### 1. Advanced Features Analysis
"""
        
        if 'advanced_features' in analysis_results['results']:
            af = analysis_results['results']['advanced_features']
            report += f"""
- **Total Features**: {af.get('total_features', 'N/A')}
- **Feature Statistics**:
  - Mean: {af.get('feature_statistics', {}).get('mean', 'N/A'):.4f}
  - Std: {af.get('feature_statistics', {}).get('std', 'N/A'):.4f}
  - Non-zero: {af.get('feature_statistics', {}).get('non_zero', 'N/A')}
"""
        
        report += """
### 2. Series Patterns Analysis
"""
        
        if 'series_patterns' in analysis_results['results']:
            sp = analysis_results['results']['series_patterns']
            report += f"""
- **Total Series Detected**: {sp.get('overall', {}).get('total_series_detected', 'N/A')}
- **Series Diversity**: {sp.get('overall', {}).get('series_diversity', 'N/A')}
- **Series Consistency**: {sp.get('overall', {}).get('series_consistency', 'N/A'):.4f}
"""
        
        report += """
### 3. Cross-Timeframe Analysis
"""
        
        if 'cross_timeframe' in analysis_results['results']:
            ct = analysis_results['results']['cross_timeframe']
            report += f"""
- **Timeframe Confluence**: {ct.get('overall', {}).get('timeframe_confluence', 'N/A'):.4f}
- **Timeframe Divergence**: {ct.get('overall', {}).get('timeframe_divergence', 'N/A'):.4f}
- **Multi-Timeframe Trend Strength**: {ct.get('overall', {}).get('multi_timeframe_trend_strength', 'N/A'):.4f}
"""
        
        report += """
### 4. Multi-Dimensional Analysis
"""
        
        if 'multi_dimensional' in analysis_results['results']:
            md = analysis_results['results']['multi_dimensional']
            report += f"""
- **Dimension Interaction Strength**: {md.get('overall', {}).get('dimension_interaction_strength', 'N/A'):.4f}
- **Cross-Dimension Correlation**: {md.get('overall', {}).get('cross_dimension_correlation', 'N/A'):.4f}
- **Multi-Dimensional Consensus**: {md.get('overall', {}).get('multi_dimensional_consensus', 'N/A'):.4f}
"""
        
        report += """
### 5. Pattern Strength Analysis
"""
        
        if 'pattern_strength' in analysis_results['results']:
            ps = analysis_results['results']['pattern_strength']
            report += f"""
- **Average Pattern Quality**: {ps.get('overall', {}).get('average_pattern_quality', 'N/A'):.4f}
- **Pattern Consistency**: {ps.get('overall', {}).get('pattern_consistency', 'N/A'):.4f}
- **Quality Trend**: {ps.get('overall', {}).get('quality_trend', 'N/A'):.4f}
"""
        
        report += """
### 6. Consensus Training
"""
        
        if 'consensus_training' in analysis_results['results']:
            ct = analysis_results['results']['consensus_training']
            report += f"""
- **Training Success**: {'✅' if ct.get('success', False) else '❌'}
- **Training Time**: {ct.get('total_time', 'N/A'):.2f}s
"""
        
        report += """
### 7. Performance Evaluation
"""
        
        if 'performance_evaluation' in analysis_results['results']:
            pe = analysis_results['results']['performance_evaluation']
            overall_perf = pe.get('overall_performance', {})
            report += f"""
- **Comprehensive Analysis Score**: {overall_perf.get('comprehensive_analysis_score', 'N/A'):.4f}
- **Feature Engineering Score**: {overall_perf.get('feature_engineering_score', 'N/A'):.4f}
- **Consensus ML Score**: {overall_perf.get('consensus_ml_score', 'N/A'):.4f}
- **Integration Score**: {overall_perf.get('integration_score', 'N/A'):.4f}
"""
        
        report += """
## Conclusion
The comprehensive candle analysis successfully demonstrates:
- Advanced feature engineering with series patterns, cross-timeframe analysis, and multi-dimensional interactions
- Pattern strength and quality assessment
- Consensus-based ML indicator generation
- Comprehensive validation and performance evaluation

The system provides a complete solution for generating reliable trading indicators based on candlestick patterns with advanced feature analysis and consensus validation.
"""
        
        return report


def create_sample_data(n_samples: int = 3000) -> pd.DataFrame:
    """Create realistic sample OHLCV data for comprehensive analysis."""
    np.random.seed(42)
    
    # Generate price series with trend, volatility, and patterns
    base_price = 100.0
    trend = np.linspace(0, 0.15, n_samples)  # 15% trend over period
    volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 4*np.pi, n_samples))  # Varying volatility
    noise = np.random.normal(0, volatility, n_samples)
    returns = trend + noise
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data with realistic patterns
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples) * (1 + 0.5 * np.sin(np.linspace(0, 2*np.pi, n_samples)))
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some realistic patterns
    for i in range(100, n_samples, 200):
        if i + 5 < n_samples:
            # Create hammer pattern
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] * 0.98
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'close'] * 1.01
            data.loc[data.index[i], 'open'] = data.loc[data.index[i], 'close'] * 0.999
    
    return data


def run_comprehensive_analysis():
    """Run the complete comprehensive candle analysis."""
    print("🚀 Comprehensive Candle Analysis with Advanced Features")
    print("=" * 70)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data(3000)
    print(f"   Generated {len(data)} samples")
    
    # Create comprehensive analysis
    analysis = ComprehensiveCandleAnalysis()
    
    # Run complete analysis
    print("\n🎯 Running comprehensive analysis...")
    results = analysis.run_complete_analysis(data, symbol='BTCUSDT')
    
    # Generate and display report
    print("\n📋 Generating comprehensive report...")
    report = analysis.generate_comprehensive_report(results)
    print(report)
    
    # Save results
    import json
    with open('comprehensive_analysis_results.json', 'w') as f:
        json.dump(results, f, default=str, indent=2)
    
    print(f"\n💾 Results saved to 'comprehensive_analysis_results.json'")
    print("\n🎉 Comprehensive candle analysis completed successfully!")
    
    return analysis, results


if __name__ == "__main__":
    analysis, results = run_comprehensive_analysis()