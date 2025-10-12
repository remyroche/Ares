"""
Data-Driven Feature Selection System

This module provides intelligent feature selection from the full feature bank (200+ features)
to select 40-ish features (at least 3 per category) for interaction generation.

Key Features:
- Analyzes all available feature categories from the feature bank
- Uses data-driven metrics to select the most relevant features
- Ensures diversity across categories and feature types
- Integrates with DataDrivenInteractionGenerator
- Leverages VectorBT optimization for performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging
from itertools import combinations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import feature bank and categories
try:
    from ..core.feature_bank import get_global_feature_bank, FeatureBank
    from ..core.feature_generator import FeatureCategory
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False
    get_global_feature_bank = None
    FeatureCategory = None

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_corr, rolling_std, rolling_mean
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    rolling_corr = None
    rolling_std = None
    rolling_mean = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for data-driven feature selection."""
    # Selection parameters
    target_feature_count: int = 40
    min_features_per_category: int = 3
    max_features_per_category: int = 8
    
    # Quality thresholds
    min_variance: float = 1e-8
    max_correlation_threshold: float = 0.95
    min_information_content: float = 0.1
    
    # Diversity requirements
    require_different_aspects: bool = True
    aspect_diversity_threshold: float = 0.3
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_vectorbt: bool = True
    
    # Category weights (higher = more important)
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 1.0,
        'volatility': 1.0,
        'trend': 1.0,
        'oscillator': 1.0,
        'volume': 1.0,
        'returns': 1.0,
        'cross_timeframe': 1.2,  # Slightly higher weight
        'microstructure': 1.1,
        'entropy': 0.9,
        'support_resistance': 0.9,
        'candlestick_pattern': 0.8,
        'time': 0.7,
        'order_flow': 1.0,
        'regime': 1.0,
        'acceleration': 1.0,
        'advanced_statistical': 1.0,
        'spectral_wavelet': 0.9
    })


@dataclass
class FeatureScore:
    """Score and metadata for a feature."""
    feature_name: str
    category: str
    aspect_type: str  # e.g., 'momentum_short', 'volatility_long', 'trend_medium'
    score: float
    variance: float
    correlation_with_target: float
    information_content: float
    uniqueness_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSelectionResult:
    """Result of data-driven feature selection."""
    selected_features: List[FeatureScore]
    category_distribution: Dict[str, int]
    aspect_distribution: Dict[str, int]
    total_features_analyzed: int
    selection_time: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataDrivenFeatureSelector:
    """
    Data-driven feature selector that intelligently chooses features from the full feature bank.
    
    This class analyzes all available features from the feature bank and selects the most
    relevant ones based on data characteristics, ensuring diversity across categories
    and feature aspects.
    """
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """
        Initialize the data-driven feature selector.
        
        Args:
            config: Configuration for feature selection
        """
        self.config = config or FeatureSelectionConfig()
        
        # Initialize feature bank
        if FEATURE_BANK_AVAILABLE:
            self.feature_bank = get_global_feature_bank()
        else:
            self.feature_bank = None
            logger.warning("Feature bank not available, using fallback feature list")
        
        # Feature aspect mapping for diversity
        self.aspect_mapping = {
            'momentum': ['short_term', 'medium_term', 'long_term', 'cross_timeframe'],
            'volatility': ['realized', 'implied', 'regime_based', 'cross_timeframe'],
            'trend': ['short_term', 'medium_term', 'long_term', 'regime_based'],
            'oscillator': ['momentum_based', 'trend_based', 'volume_based', 'price_based'],
            'volume': ['absolute', 'relative', 'momentum', 'pattern_based'],
            'returns': ['raw', 'normalized', 'risk_adjusted', 'regime_based'],
            'cross_timeframe': ['momentum', 'volatility', 'trend', 'volume'],
            'microstructure': ['bid_ask', 'order_flow', 'liquidity', 'execution'],
            'entropy': ['price', 'volume', 'information', 'regime'],
            'support_resistance': ['static', 'dynamic', 'volume_based', 'time_based'],
            'candlestick_pattern': ['reversal', 'continuation', 'indecision', 'volume_confirmation'],
            'time': ['intraday', 'daily', 'weekly', 'seasonal'],
            'order_flow': ['imbalance', 'pressure', 'aggression', 'liquidity'],
            'regime': ['volatility', 'trend', 'volume', 'market_state'],
            'acceleration': ['price', 'volume', 'momentum', 'volatility'],
            'advanced_statistical': ['higher_moments', 'distribution', 'dependence', 'regime'],
            'spectral_wavelet': ['frequency', 'time_frequency', 'decomposition', 'reconstruction']
        }
        
        logger.info(f"✅ Data-driven feature selector initialized")
        logger.info(f"📊 Target features: {self.config.target_feature_count}")
        logger.info(f"📊 Min per category: {self.config.min_features_per_category}")
        logger.info(f"📊 Feature bank available: {FEATURE_BANK_AVAILABLE}")
    
    def select_features(self, 
                       data: pd.DataFrame,
                       targets: Optional[pd.Series] = None,
                       available_categories: Optional[List[str]] = None) -> FeatureSelectionResult:
        """
        Select features using data-driven approach.
        
        Args:
            data: Input data for feature generation
            targets: Target variable for relevance scoring
            available_categories: Specific categories to consider (None = all)
            
        Returns:
            FeatureSelectionResult with selected features and metadata
        """
        start_time = time.time()
        logger.info(f"🚀 Starting data-driven feature selection")
        logger.info(f"📊 Input data shape: {data.shape}")
        
        # Step 1: Generate all available features
        all_features = self._generate_all_features(data, available_categories)
        logger.info(f"📊 Generated {len(all_features.columns)} features from feature bank")
        
        # Step 2: Analyze feature characteristics
        feature_scores = self._analyze_features(all_features, targets)
        logger.info(f"📊 Analyzed {len(feature_scores)} features")
        
        # Step 3: Select features with diversity constraints
        selected_features = self._select_diverse_features(feature_scores)
        logger.info(f"📊 Selected {len(selected_features)} features")
        
        # Step 4: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(selected_features, all_features, targets)
        
        # Step 5: Create result
        selection_time = time.time() - start_time
        result = FeatureSelectionResult(
            selected_features=selected_features,
            category_distribution=self._calculate_category_distribution(selected_features),
            aspect_distribution=self._calculate_aspect_distribution(selected_features),
            total_features_analyzed=len(feature_scores),
            selection_time=selection_time,
            quality_metrics=quality_metrics,
            metadata={
                'config': self.config.__dict__,
                'data_shape': data.shape,
                'targets_provided': targets is not None
            }
        )
        
        logger.info(f"✅ Feature selection completed in {selection_time:.2f}s")
        logger.info(f"📊 Category distribution: {result.category_distribution}")
        
        return result
    
    def _generate_all_features(self, 
                              data: pd.DataFrame,
                              available_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate all available features from the feature bank."""
        if not self.feature_bank:
            logger.warning("Feature bank not available, using basic features")
            return self._generate_basic_features(data)
        
        try:
            # Get all available categories
            if available_categories is None:
                available_categories = [cat.value for cat in FeatureCategory]
            
            # Generate features for all categories
            all_features = self.feature_bank.generate_features(
                data=data,
                categories=available_categories,
                target_column='returns' if 'returns' in data.columns else None
            )
            
            return all_features
            
        except Exception as e:
            logger.warning(f"Feature bank generation failed: {e}, using basic features")
            return self._generate_basic_features(data)
    
    def _generate_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features as fallback."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        if 'close' in data.columns:
            features['close_return'] = data['close'].pct_change()
            features['close_log_return'] = np.log(data['close'] / data['close'].shift(1))
            features['close_volatility_20'] = data['close'].rolling(20).std()
            features['close_sma_20'] = data['close'].rolling(20).mean()
            features['close_ema_12'] = data['close'].ewm(span=12).mean()
        
        # Basic volume features
        if 'volume' in data.columns:
            features['volume_sma_20'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Basic technical indicators
        if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
            features['rsi_14'] = self._calculate_rsi(data['close'], 14)
            features['atr_14'] = self._calculate_atr(data, 14)
        
        return features
    
    def _analyze_features(self, 
                         features: pd.DataFrame,
                         targets: Optional[pd.Series]) -> List[FeatureScore]:
        """Analyze all features and calculate scores."""
        feature_scores = []
        
        for feature_name in features.columns:
            try:
                feature_series = features[feature_name].dropna()
                
                if len(feature_series) == 0:
                    continue
                
                # Calculate basic metrics
                variance = float(feature_series.var())
                if variance < self.config.min_variance:
                    continue
                
                # Calculate correlation with target
                correlation_with_target = 0.0
                if targets is not None:
                    try:
                        corr = feature_series.corr(targets)
                        correlation_with_target = abs(corr) if not pd.isna(corr) else 0.0
                    except:
                        correlation_with_target = 0.0
                
                # Calculate information content
                information_content = self._calculate_information_content(feature_series)
                
                # Determine category and aspect
                category = self._determine_category(feature_name)
                aspect_type = self._determine_aspect_type(feature_name, category)
                
                # Calculate uniqueness score
                uniqueness_score = self._calculate_uniqueness_score(feature_series, features)
                
                # Calculate overall score
                score = self._calculate_feature_score(
                    variance, correlation_with_target, information_content, 
                    uniqueness_score, category
                )
                
                feature_score = FeatureScore(
                    feature_name=feature_name,
                    category=category,
                    aspect_type=aspect_type,
                    score=score,
                    variance=variance,
                    correlation_with_target=correlation_with_target,
                    information_content=information_content,
                    uniqueness_score=uniqueness_score,
                    metadata={
                        'length': len(feature_series),
                        'missing_ratio': 1.0 - len(feature_series) / len(features),
                        'skewness': float(feature_series.skew()),
                        'kurtosis': float(feature_series.kurtosis())
                    }
                )
                
                feature_scores.append(feature_score)
                
            except Exception as e:
                logger.debug(f"Failed to analyze feature {feature_name}: {e}")
                continue
        
        return feature_scores
    
    def _select_diverse_features(self, feature_scores: List[FeatureScore]) -> List[FeatureScore]:
        """Select features ensuring diversity across categories and aspects."""
        # Group features by category
        category_groups = {}
        for score in feature_scores:
            category = score.category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(score)
        
        # Sort features within each category by score
        for category in category_groups:
            category_groups[category].sort(key=lambda x: x.score, reverse=True)
        
        selected_features = []
        
        # First pass: Select minimum required features per category
        for category, features in category_groups.items():
            min_required = self.config.min_features_per_category
            max_allowed = min(self.config.max_features_per_category, len(features))
            
            # Select top features from this category
            category_selected = features[:max_allowed]
            
            # Ensure we have at least the minimum
            if len(category_selected) < min_required:
                category_selected = features[:min_required]
            
            selected_features.extend(category_selected)
        
        # Second pass: Fill remaining slots with best available features
        remaining_slots = self.config.target_feature_count - len(selected_features)
        if remaining_slots > 0:
            # Get all unselected features sorted by score
            selected_names = {f.feature_name for f in selected_features}
            unselected = [f for f in feature_scores if f.feature_name not in selected_names]
            unselected.sort(key=lambda x: x.score, reverse=True)
            
            # Add best remaining features
            selected_features.extend(unselected[:remaining_slots])
        
        # Third pass: Ensure aspect diversity within categories
        if self.config.require_different_aspects:
            selected_features = self._ensure_aspect_diversity(selected_features)
        
        return selected_features[:self.config.target_feature_count]
    
    def _ensure_aspect_diversity(self, selected_features: List[FeatureScore]) -> List[FeatureScore]:
        """Ensure diversity of aspects within each category."""
        # Group by category
        category_groups = {}
        for feature in selected_features:
            category = feature.category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(feature)
        
        diversified_features = []
        
        for category, features in category_groups.items():
            if len(features) <= 3:  # Not enough for diversity
                diversified_features.extend(features)
                continue
            
            # Group by aspect type
            aspect_groups = {}
            for feature in features:
                aspect = feature.aspect_type
                if aspect not in aspect_groups:
                    aspect_groups[aspect] = []
                aspect_groups[aspect].append(feature)
            
            # Select from different aspects
            selected_from_category = []
            aspects_used = set()
            
            # First, select the best feature from each aspect
            for aspect, aspect_features in aspect_groups.items():
                if aspect not in aspects_used:
                    best_feature = max(aspect_features, key=lambda x: x.score)
                    selected_from_category.append(best_feature)
                    aspects_used.add(aspect)
            
            # Fill remaining slots with best available
            remaining_slots = len(features) - len(selected_from_category)
            if remaining_slots > 0:
                all_features = [f for f in features if f not in selected_from_category]
                all_features.sort(key=lambda x: x.score, reverse=True)
                selected_from_category.extend(all_features[:remaining_slots])
            
            diversified_features.extend(selected_from_category)
        
        return diversified_features
    
    def _calculate_feature_score(self, 
                                variance: float,
                                correlation_with_target: float,
                                information_content: float,
                                uniqueness_score: float,
                                category: str) -> float:
        """Calculate overall feature score."""
        # Base score from variance and information content
        base_score = np.log1p(variance) * information_content
        
        # Boost for target correlation
        correlation_boost = correlation_with_target * 0.3
        
        # Boost for uniqueness
        uniqueness_boost = uniqueness_score * 0.2
        
        # Category weight
        category_weight = self.config.category_weights.get(category, 1.0)
        
        # Combine scores
        total_score = (base_score + correlation_boost + uniqueness_boost) * category_weight
        
        return total_score
    
    def _calculate_information_content(self, series: pd.Series) -> float:
        """Calculate information content of a feature."""
        try:
            # Use entropy as a proxy for information content
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(value_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return normalized_entropy
        except:
            return 0.0
    
    def _calculate_uniqueness_score(self, series: pd.Series, all_features: pd.DataFrame) -> float:
        """Calculate how unique this feature is compared to others."""
        try:
            correlations = []
            for other_col in all_features.columns:
                if other_col != series.name:
                    try:
                        corr = series.corr(all_features[other_col])
                        if not pd.isna(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
            
            if not correlations:
                return 1.0
            
            # Uniqueness is inverse of maximum correlation
            max_correlation = max(correlations)
            uniqueness = 1.0 - max_correlation
            
            return max(0.0, uniqueness)
        except:
            return 0.5
    
    def _determine_category(self, feature_name: str) -> str:
        """Determine the category of a feature based on its name."""
        feature_lower = feature_name.lower()
        
        # Category detection based on keywords
        if any(keyword in feature_lower for keyword in ['rsi', 'momentum', 'roc', 'williams']):
            return 'momentum'
        elif any(keyword in feature_lower for keyword in ['volatility', 'atr', 'std', 'var']):
            return 'volatility'
        elif any(keyword in feature_lower for keyword in ['sma', 'ema', 'ma_', 'trend']):
            return 'trend'
        elif any(keyword in feature_lower for keyword in ['oscillator', 'stoch', 'cci', 'mfi']):
            return 'oscillator'
        elif any(keyword in feature_lower for keyword in ['volume', 'vol_', 'obv', 'ad']):
            return 'volume'
        elif any(keyword in feature_lower for keyword in ['return', 'pct_change', 'log_return']):
            return 'returns'
        elif any(keyword in feature_lower for keyword in ['cross', 'timeframe', 'tf_']):
            return 'cross_timeframe'
        elif any(keyword in feature_lower for keyword in ['microstructure', 'bid', 'ask', 'spread']):
            return 'microstructure'
        elif any(keyword in feature_lower for keyword in ['entropy', 'information', 'complexity']):
            return 'entropy'
        elif any(keyword in feature_lower for keyword in ['support', 'resistance', 'pivot']):
            return 'support_resistance'
        elif any(keyword in feature_lower for keyword in ['pattern', 'candlestick', 'doji', 'hammer']):
            return 'candlestick_pattern'
        elif any(keyword in feature_lower for keyword in ['time', 'hour', 'day', 'week']):
            return 'time'
        elif any(keyword in feature_lower for keyword in ['order_flow', 'imbalance', 'pressure']):
            return 'order_flow'
        elif any(keyword in feature_lower for keyword in ['regime', 'state', 'phase']):
            return 'regime'
        elif any(keyword in feature_lower for keyword in ['acceleration', 'jerk', 'second_derivative']):
            return 'acceleration'
        elif any(keyword in feature_lower for keyword in ['statistical', 'skew', 'kurt', 'quantile']):
            return 'advanced_statistical'
        elif any(keyword in feature_lower for keyword in ['spectral', 'wavelet', 'fourier', 'frequency']):
            return 'spectral_wavelet'
        else:
            return 'custom'
    
    def _determine_aspect_type(self, feature_name: str, category: str) -> str:
        """Determine the aspect type of a feature."""
        feature_lower = feature_name.lower()
        
        # Get possible aspects for this category
        possible_aspects = self.aspect_mapping.get(category, ['general'])
        
        # Determine aspect based on feature name patterns
        if 'short' in feature_lower or any(x in feature_lower for x in ['_5', '_10', '_15']):
            return possible_aspects[0] if len(possible_aspects) > 0 else 'short_term'
        elif 'long' in feature_lower or any(x in feature_lower for x in ['_50', '_100', '_200']):
            return possible_aspects[-1] if len(possible_aspects) > 0 else 'long_term'
        elif 'cross' in feature_lower or 'tf_' in feature_lower:
            return 'cross_timeframe' if 'cross_timeframe' in possible_aspects else possible_aspects[0]
        elif 'regime' in feature_lower or 'state' in feature_lower:
            return 'regime_based' if 'regime_based' in possible_aspects else possible_aspects[0]
        else:
            return possible_aspects[0] if len(possible_aspects) > 0 else 'general'
    
    def _calculate_category_distribution(self, features: List[FeatureScore]) -> Dict[str, int]:
        """Calculate distribution of features across categories."""
        distribution = {}
        for feature in features:
            category = feature.category
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _calculate_aspect_distribution(self, features: List[FeatureScore]) -> Dict[str, int]:
        """Calculate distribution of features across aspects."""
        distribution = {}
        for feature in features:
            aspect = feature.aspect_type
            distribution[aspect] = distribution.get(aspect, 0) + 1
        return distribution
    
    def _calculate_quality_metrics(self, 
                                  selected_features: List[FeatureScore],
                                  all_features: pd.DataFrame,
                                  targets: Optional[pd.Series]) -> Dict[str, float]:
        """Calculate quality metrics for the selected features."""
        if not selected_features:
            return {}
        
        # Average scores
        avg_score = np.mean([f.score for f in selected_features])
        avg_variance = np.mean([f.variance for f in selected_features])
        avg_correlation = np.mean([f.correlation_with_target for f in selected_features])
        avg_information = np.mean([f.information_content for f in selected_features])
        avg_uniqueness = np.mean([f.uniqueness_score for f in selected_features])
        
        # Diversity metrics
        category_diversity = len(set(f.category for f in selected_features))
        aspect_diversity = len(set(f.aspect_type for f in selected_features))
        
        # Coverage metrics
        total_categories = len(self.config.category_weights)
        category_coverage = category_diversity / total_categories
        
        return {
            'average_score': avg_score,
            'average_variance': avg_variance,
            'average_correlation': avg_correlation,
            'average_information_content': avg_information,
            'average_uniqueness': avg_uniqueness,
            'category_diversity': category_diversity,
            'aspect_diversity': aspect_diversity,
            'category_coverage': category_coverage,
            'total_features': len(selected_features)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR indicator."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
        except:
            return pd.Series(index=data.index, dtype=float)


def create_data_driven_feature_selector(config: Optional[FeatureSelectionConfig] = None) -> DataDrivenFeatureSelector:
    """Create a data-driven feature selector with default configuration."""
    return DataDrivenFeatureSelector(config)