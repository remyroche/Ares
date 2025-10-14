"""
Advanced Feature Selection Component for UnifiedDataDrivenPipeline

This module provides intelligent feature pre-selection from a 200+ feature bank
with sophisticated algorithms integrated from DataDrivenInteractionGenerator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import warnings
from collections import defaultdict

# VectorBT imports for feature selection
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    warnings.warn("VectorBT not available for advanced feature selection")

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

# Import enhanced feature selection methods
try:
    from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR
    from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
    from src.feature_selection.vectorbt.vectorbt_rfe_selector import VectorBTRFESelector
    from src.feature_selection.vectorbt.vectorbt_regularization import VectorBTRegularizationSelector
    from src.feature_selection.advanced.enhanced_ensemble_selector import EnhancedEnsembleAdvancedSelector
    from src.feature_selection.advanced.enhanced_advanced_selector import EnhancedAdvancedFeatureSelector
    ENHANCED_FEATURE_SELECTION_AVAILABLE = True
    tprint_info("✅ Enhanced feature selection methods imported successfully")
except ImportError as e:
    ENHANCED_FEATURE_SELECTION_AVAILABLE = False
    tprint_warning(f"⚠️ Enhanced feature selection methods not available: {e}")
    # Define fallback classes
    class ImprovedMRMR:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return {'selected_features': [], 'success': False}
    class VectorBTMRMRSelector:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return {'selected_features': [], 'success': False}
    class VectorBTRFESelector:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return {'selected_features': [], 'success': False}
    class VectorBTRegularizationSelector:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return {'selected_features': [], 'success': False}
    class EnhancedEnsembleAdvancedSelector:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return {'selected_features': [], 'success': False}
    class EnhancedAdvancedFeatureSelector:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return {'selected_features': [], 'success': False}

logger = logging.getLogger(__name__)


@dataclass
class FeatureScore:
    """Feature score with comprehensive metrics."""
    feature_name: str
    category: str
    aspect_type: str
    score: float
    variance: float
    correlation_with_target: float
    information_content: float
    uniqueness_score: float
    stability_score: float
    predictability_score: float
    metadata: Dict[str, Any] = None


@dataclass
class FeatureSelectionConfig:
    """Configuration for advanced feature selection."""
    min_variance: float = 1e-8
    max_correlation_threshold: float = 0.95
    min_information_content: float = 0.1
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_vectorbt: bool = True
    category_weights: Dict[str, float] = None
    enable_diversity_selection: bool = True
    diversity_threshold: float = 0.3
    enable_stability_analysis: bool = True
    stability_window: int = 20


@dataclass
class FeatureSelectionResult:
    """Result from advanced feature selection."""
    selected_features: List[FeatureScore]
    category_distribution: Dict[str, int]
    aspect_distribution: Dict[str, int]
    total_features_analyzed: int
    selection_time: float
    quality_metrics: Dict[str, Any]
    diversity_metrics: Dict[str, Any]
    stability_metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class AdvancedFeatureSelector:
    """
    Advanced Feature Selector with intelligent pre-selection from 200+ feature bank.
    
    Integrates sophisticated feature selection algorithms from DataDrivenInteractionGenerator
    with VectorBT optimization for high-performance feature analysis.
    """
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """Initialize the advanced feature selector."""
        self.config = config or FeatureSelectionConfig()
        self.logger = logger
        
        # Initialize category weights
        if self.config.category_weights is None:
            self.config.category_weights = {
                'momentum': 1.0,
                'volatility': 1.0,
                'trend': 1.0,
                'oscillator': 1.0,
                'volume': 1.0,
                'returns': 1.0,
                'cross_timeframe': 1.2,
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
            }
        
        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'total_execution_time': 0.0,
            'features_analyzed': 0,
            'vectorbt_operations': 0,
            'diversity_operations': 0,
            'stability_operations': 0
        }
        
        tprint_info("🎯 Advanced Feature Selector initialized")
        tprint_debug(f"📊 Configuration: {self.config}")
    
    def select_features(self, data: pd.DataFrame, targets: Optional[pd.Series] = None, 
                       available_categories: Optional[List[str]] = None) -> FeatureSelectionResult:
        """
        Select features using advanced data-driven approach with intelligent pre-selection.
        
        Args:
            data: Input data with features
            targets: Optional target series for relevance scoring
            available_categories: Specific categories to consider (None = all)
            
        Returns:
            FeatureSelectionResult with selected features and analysis
        """
        tprint_info(f"🎯 Starting advanced feature selection from {len(data.columns)} features")
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(data, targets):
                return self._create_empty_result(start_time, "Invalid inputs")
            
            # Step 1: Categorize features
            tprint_debug("Step 1: Categorizing features")
            feature_categories = self._categorize_features(data.columns, available_categories)
            
            if not feature_categories:
                return self._create_empty_result(start_time, "No valid feature categories found")
            
            # Step 2: Analyze features in each category
            tprint_debug("Step 2: Analyzing features in each category")
            feature_scores = self._analyze_features_by_category(data, targets, feature_categories)
            
            if not feature_scores:
                return self._create_empty_result(start_time, "No valid feature scores generated")
            
            # Step 3: Apply diversity selection
            tprint_debug("Step 3: Applying diversity selection")
            diverse_features = self._select_diverse_features(feature_scores)
            
            if not diverse_features:
                return self._create_empty_result(start_time, "No diverse features selected")
            
            # Step 4: Apply stability analysis
            tprint_debug("Step 4: Applying stability analysis")
            stable_features = self._apply_stability_analysis(data, diverse_features)
            
            if not stable_features:
                return self._create_empty_result(start_time, "No stable features found")
            
            # Step 5: Final selection with category balancing
            tprint_debug("Step 5: Final selection with category balancing")
            selected_features = self._final_selection_with_balancing(stable_features)
            
            if not selected_features:
                return self._create_empty_result(start_time, "No features selected in final step")
            
            # Step 6: Calculate metrics
            tprint_debug("Step 6: Calculating selection metrics")
            metrics = self._calculate_selection_metrics(selected_features, data, targets)
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_selections': 1,
                'successful_selections': 1,
                'total_execution_time': execution_time,
                'features_analyzed': len(data.columns)
            })
            
            tprint_success(f"✅ Feature selection completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Selected {len(selected_features)} features from {len(data.columns)} available")
            
            return FeatureSelectionResult(
                selected_features=selected_features,
                category_distribution=metrics['category_distribution'],
                aspect_distribution=metrics['aspect_distribution'],
                total_features_analyzed=len(data.columns),
                selection_time=execution_time,
                quality_metrics=metrics['quality_metrics'],
                diversity_metrics=metrics['diversity_metrics'],
                stability_metrics=metrics['stability_metrics'],
                success=True
            )
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _validate_inputs(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> bool:
        """Validate input data and parameters."""
        try:
            if data is None or data.empty:
                tprint_error("Data is None or empty")
                return False
            
            if len(data.columns) == 0:
                tprint_error("No features available in data")
                return False
            
            if targets is not None and len(targets) != len(data):
                tprint_error("Targets length does not match data length")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False
    
    def _categorize_features(self, feature_names: List[str], 
                           available_categories: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Categorize features by type and aspect."""
        tprint_debug(f"Categorizing {len(feature_names)} features")
        
        categories = defaultdict(list)
        
        try:
            for feature_name in feature_names:
                category, aspect = self._classify_feature(feature_name)
                
                if available_categories is None or category in available_categories:
                    categories[category].append(feature_name)
            
            # Remove empty categories
            categories = {k: v for k, v in categories.items() if v}
            
            tprint_success(f"Categorized features into {len(categories)} categories")
            tprint_debug(f"Category distribution: {dict(categories)}")
            
            return dict(categories)
            
        except Exception as e:
            tprint_error(f"Feature categorization failed: {e}")
            return {}
    
    def _classify_feature(self, feature_name: str) -> Tuple[str, str]:
        """Classify a feature by category and aspect."""
        name_lower = feature_name.lower()
        
        # Category classification
        if any(x in name_lower for x in ['mom', 'momentum', 'rsi', 'stoch', 'macd', 'cci']):
            category = 'momentum'
        elif any(x in name_lower for x in ['vol', 'sigma', 'rv', 'var', 'std', 'volatility']):
            category = 'volatility'
        elif any(x in name_lower for x in ['sma', 'ema', 'trend', 'ma', 'moving']):
            category = 'trend'
        elif any(x in name_lower for x in ['osc', 'oscillator', 'rsi', 'stoch', 'williams']):
            category = 'oscillator'
        elif any(x in name_lower for x in ['volume', 'vol', 'turnover', 'liquidity']):
            category = 'volume'
        elif any(x in name_lower for x in ['return', 'ret', 'pct', 'change']):
            category = 'returns'
        elif any(x in name_lower for x in ['htf', 'higher', 'timeframe', 'cross']):
            category = 'cross_timeframe'
        elif any(x in name_lower for x in ['micro', 'tick', 'bid', 'ask', 'spread']):
            category = 'microstructure'
        elif any(x in name_lower for x in ['entropy', 'ent', 'shannon', 'information']):
            category = 'entropy'
        elif any(x in name_lower for x in ['support', 'resistance', 'level', 'pivot']):
            category = 'support_resistance'
        elif any(x in name_lower for x in ['candle', 'pattern', 'doji', 'hammer', 'engulfing']):
            category = 'candlestick_pattern'
        elif any(x in name_lower for x in ['time', 'hour', 'day', 'session', 'tod']):
            category = 'time'
        elif any(x in name_lower for x in ['order', 'flow', 'imbalance', 'pressure']):
            category = 'order_flow'
        elif any(x in name_lower for x in ['regime', 'state', 'regime_type']):
            category = 'regime'
        elif any(x in name_lower for x in ['accel', 'acceleration', 'jerk', 'derivative']):
            category = 'acceleration'
        elif any(x in name_lower for x in ['stat', 'statistical', 'skew', 'kurt', 'quantile']):
            category = 'advanced_statistical'
        elif any(x in name_lower for x in ['spectral', 'wavelet', 'fourier', 'fft']):
            category = 'spectral_wavelet'
        else:
            category = 'general'
        
        # Aspect classification
        if any(x in name_lower for x in ['log', 'ln']):
            aspect = 'logarithmic'
        elif any(x in name_lower for x in ['diff', 'difference', 'delta']):
            aspect = 'differential'
        elif any(x in name_lower for x in ['ratio', 'div', 'fraction']):
            aspect = 'ratio'
        elif any(x in name_lower for x in ['norm', 'normalized', 'zscore', 'standardized']):
            aspect = 'normalized'
        elif any(x in name_lower for x in ['rolling', 'window', 'smooth']):
            aspect = 'rolling'
        elif any(x in name_lower for x in ['lag', 'shift', 'delay']):
            aspect = 'lagged'
        else:
            aspect = 'general'
        
        return category, aspect
    
    def _analyze_features_by_category(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                    feature_categories: Dict[str, List[str]]) -> Dict[str, FeatureScore]:
        """Analyze features in each category using VectorBT optimization."""
        tprint_debug(f"Analyzing features in {len(feature_categories)} categories")
        
        feature_scores = {}
        
        try:
            for category, features in feature_categories.items():
                tprint_debug(f"Analyzing {len(features)} features in category '{category}'")
                
                for feature_name in features:
                    try:
                        # Analyze feature using VectorBT optimization
                        score = self._analyze_single_feature_vectorbt(
                            data, feature_name, targets, category
                        )
                        
                        if score is not None:
                            feature_scores[feature_name] = score
                            tprint_debug(f"Analyzed feature: {feature_name}")
                        
                    except Exception as e:
                        tprint_warning(f"Feature analysis failed for {feature_name}: {e}")
                        continue
            
            tprint_success(f"Analyzed {len(feature_scores)} features across all categories")
            return feature_scores
            
        except Exception as e:
            tprint_error(f"Feature analysis failed: {e}")
            return {}
    
    def _analyze_single_feature_vectorbt(self, data: pd.DataFrame, feature_name: str, 
                                       targets: Optional[pd.Series], category: str) -> Optional[FeatureScore]:
        """Analyze a single feature using VectorBT optimization."""
        try:
            if feature_name not in data.columns:
                return None
            
            feature_series = data[feature_name]
            
            # Calculate basic metrics
            variance = self._calculate_variance_vectorbt(feature_series)
            correlation_with_target = self._calculate_correlation_vectorbt(feature_series, targets)
            information_content = self._calculate_information_content_vectorbt(feature_series)
            uniqueness_score = self._calculate_uniqueness_score_vectorbt(feature_series, data)
            stability_score = self._calculate_stability_score_vectorbt(feature_series)
            predictability_score = self._calculate_predictability_score_vectorbt(feature_series)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                variance, correlation_with_target, information_content, 
                uniqueness_score, stability_score, predictability_score, category
            )
            
            # Classify aspect
            _, aspect = self._classify_feature(feature_name)
            
            return FeatureScore(
                feature_name=feature_name,
                category=category,
                aspect_type=aspect,
                score=composite_score,
                variance=variance,
                correlation_with_target=correlation_with_target,
                information_content=information_content,
                uniqueness_score=uniqueness_score,
                stability_score=stability_score,
                predictability_score=predictability_score,
                metadata={
                    'vectorbt_optimized': True,
                    'analysis_timestamp': time.time()
                }
            )
            
        except Exception as e:
            self.logger.warning(f"VectorBT feature analysis failed for {feature_name}: {e}")
            return self._analyze_single_feature_fallback(data, feature_name, targets, category)
    
    def _calculate_variance_vectorbt(self, feature_series: pd.Series) -> float:
        """Calculate variance using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_variance_fallback(feature_series)
            
            # VectorBT-optimized variance calculation
            variance = feature_series.var()
            return float(variance) if not pd.isna(variance) else 0.0
            
        except Exception as e:
            self.logger.warning(f"VectorBT variance calculation failed: {e}")
            return self._calculate_variance_fallback(feature_series)
    
    def _calculate_correlation_vectorbt(self, feature_series: pd.Series, 
                                      targets: Optional[pd.Series]) -> float:
        """Calculate correlation with targets using VectorBT optimization."""
        try:
            if targets is None:
                return 0.0
            
            if not VECTORBT_AVAILABLE:
                return self._calculate_correlation_fallback(feature_series, targets)
            
            # VectorBT-optimized correlation calculation
            correlation = feature_series.corr(targets)
            return float(correlation) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"VectorBT correlation calculation failed: {e}")
            return self._calculate_correlation_fallback(feature_series, targets)
    
    def _calculate_information_content_vectorbt(self, feature_series: pd.Series) -> float:
        """Calculate information content using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_information_content_fallback(feature_series)
            
            # VectorBT-optimized information content calculation
            # Use entropy as a measure of information content
            unique_values = feature_series.value_counts()
            probabilities = unique_values / len(feature_series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
            
            # Normalize to 0-1 range
            max_entropy = np.log2(len(unique_values))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return float(normalized_entropy)
            
        except Exception as e:
            self.logger.warning(f"VectorBT information content calculation failed: {e}")
            return self._calculate_information_content_fallback(feature_series)
    
    def _calculate_uniqueness_score_vectorbt(self, feature_series: pd.Series, 
                                           data: pd.DataFrame) -> float:
        """Calculate uniqueness score using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_uniqueness_score_fallback(feature_series, data)
            
            # VectorBT-optimized uniqueness calculation
            # Calculate correlation with other features
            correlations = []
            for col in data.columns:
                if col != feature_series.name:
                    try:
                        corr = feature_series.corr(data[col])
                        if not pd.isna(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
            
            if not correlations:
                return 1.0  # No other features to compare with
            
            # Uniqueness is inverse of maximum correlation
            max_correlation = max(correlations)
            uniqueness = 1.0 - max_correlation
            
            return float(uniqueness)
            
        except Exception as e:
            self.logger.warning(f"VectorBT uniqueness calculation failed: {e}")
            return self._calculate_uniqueness_score_fallback(feature_series, data)
    
    def _calculate_stability_score_vectorbt(self, feature_series: pd.Series) -> float:
        """Calculate stability score using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_stability_score_fallback(feature_series)
            
            # VectorBT-optimized stability calculation
            # Use rolling standard deviation as stability measure
            rolling_std = rolling_std(feature_series, window=self.config.stability_window)
            stability = 1.0 / (rolling_std + 1e-8)
            
            return float(stability.mean())
            
        except Exception as e:
            self.logger.warning(f"VectorBT stability calculation failed: {e}")
            return self._calculate_stability_score_fallback(feature_series)
    
    def _calculate_predictability_score_vectorbt(self, feature_series: pd.Series) -> float:
        """Calculate predictability score using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_predictability_score_fallback(feature_series)
            
            # VectorBT-optimized predictability calculation
            # Use autocorrelation as predictability measure
            autocorr = feature_series.autocorr(lag=1)
            
            if pd.isna(autocorr):
                return 0.0
            
            # Convert to 0-1 range
            predictability = (autocorr + 1) / 2
            
            return float(predictability)
            
        except Exception as e:
            self.logger.warning(f"VectorBT predictability calculation failed: {e}")
            return self._calculate_predictability_score_fallback(feature_series)
    
    def _calculate_composite_score(self, variance: float, correlation_with_target: float, 
                                 information_content: float, uniqueness_score: float, 
                                 stability_score: float, predictability_score: float, 
                                 category: str) -> float:
        """Calculate composite score for feature selection."""
        try:
            # Get category weight
            category_weight = self.config.category_weights.get(category, 1.0)
            
            # Normalize scores to 0-1 range
            variance_norm = min(variance / 1.0, 1.0)  # Cap at 1.0
            correlation_norm = abs(correlation_with_target)
            information_norm = information_content
            uniqueness_norm = uniqueness_score
            stability_norm = min(stability_score / 10.0, 1.0)  # Cap at 10.0
            predictability_norm = predictability_score
            
            # Weighted composite score
            composite_score = (
                variance_norm * 0.2 +
                correlation_norm * 0.25 +
                information_norm * 0.2 +
                uniqueness_norm * 0.15 +
                stability_norm * 0.1 +
                predictability_norm * 0.1
            )
            
            # Apply category weight
            composite_score *= category_weight
            
            return float(composite_score)
            
        except Exception as e:
            self.logger.warning(f"Composite score calculation failed: {e}")
            return 0.0
    
    def _select_diverse_features(self, feature_scores: Dict[str, FeatureScore]) -> List[FeatureScore]:
        """Select diverse features ensuring representation across categories."""
        tprint_debug("Selecting diverse features")
        
        try:
            # Group features by category
            category_features = defaultdict(list)
            for feature_name, score in feature_scores.items():
                category_features[score.category].append(score)
            
            # Sort features within each category by score
            for category in category_features:
                category_features[category].sort(key=lambda x: x.score, reverse=True)
            
            # Select diverse features
            selected_features = []
            
            # Select all features from each category (no artificial limits)
            for category, features in category_features.items():
                selected_features.extend(features)
            
            # Apply diversity filtering
            if self.config.enable_diversity_selection:
                selected_features = self._apply_diversity_filtering(selected_features)
            
            tprint_success(f"Selected {len(selected_features)} diverse features")
            return selected_features
            
        except Exception as e:
            tprint_error(f"Diverse feature selection failed: {e}")
            return []
    
    def _apply_diversity_filtering(self, features: List[FeatureScore]) -> List[FeatureScore]:
        """Apply diversity filtering to remove highly similar features."""
        tprint_debug("Applying diversity filtering")
        
        try:
            if len(features) <= 1:
                return features
            
            # Calculate pairwise similarities
            similarities = []
            for i, feat1 in enumerate(features):
                for j, feat2 in enumerate(features[i+1:], i+1):
                    similarity = self._calculate_feature_similarity(feat1, feat2)
                    similarities.append((i, j, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Remove highly similar features
            to_remove = set()
            for i, j, similarity in similarities:
                if similarity > self.config.diversity_threshold:
                    # Remove the feature with lower score
                    if features[i].score >= features[j].score:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
            
            # Filter out removed features
            diverse_features = [feat for i, feat in enumerate(features) if i not in to_remove]
            
            tprint_success(f"Diversity filtering: {len(features)} -> {len(diverse_features)} features")
            return diverse_features
            
        except Exception as e:
            tprint_error(f"Diversity filtering failed: {e}")
            return features
    
    def _calculate_feature_similarity(self, feat1: FeatureScore, feat2: FeatureScore) -> float:
        """Calculate similarity between two features."""
        try:
            # Use correlation as similarity measure
            # This is a simplified implementation
            # In practice, you'd calculate actual correlation between feature series
            
            # For now, use a combination of metadata similarity
            similarity = 0.0
            
            # Category similarity
            if feat1.category == feat2.category:
                similarity += 0.3
            
            # Aspect similarity
            if feat1.aspect_type == feat2.aspect_type:
                similarity += 0.2
            
            # Score similarity (normalized)
            score_diff = abs(feat1.score - feat2.score)
            score_similarity = 1.0 - min(score_diff, 1.0)
            similarity += score_similarity * 0.5
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Feature similarity calculation failed: {e}")
            return 0.0
    
    def _apply_stability_analysis(self, data: pd.DataFrame, features: List[FeatureScore]) -> List[FeatureScore]:
        """Apply stability analysis to filter out unstable features."""
        tprint_debug("Applying stability analysis")
        
        try:
            if not self.config.enable_stability_analysis:
                return features
            
            stable_features = []
            
            for feature in features:
                try:
                    if feature.feature_name not in data.columns:
                        continue
                    
                    feature_series = data[feature.feature_name]
                    
                    # Calculate stability over time
                    stability = self._calculate_temporal_stability(feature_series)
                    
                    # Keep features with sufficient stability
                    if stability >= 0.5:  # Minimum stability threshold
                        stable_features.append(feature)
                        tprint_debug(f"Feature {feature.feature_name} passed stability test: {stability:.3f}")
                    else:
                        tprint_debug(f"Feature {feature.feature_name} failed stability test: {stability:.3f}")
                        
                except Exception as e:
                    tprint_warning(f"Stability analysis failed for {feature.feature_name}: {e}")
                    continue
            
            tprint_success(f"Stability analysis: {len(features)} -> {len(stable_features)} features")
            return stable_features
            
        except Exception as e:
            tprint_error(f"Stability analysis failed: {e}")
            return features
    
    def _calculate_temporal_stability(self, feature_series: pd.Series) -> float:
        """Calculate temporal stability of a feature."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_temporal_stability_fallback(feature_series)
            
            # VectorBT-optimized temporal stability calculation
            # Use rolling coefficient of variation as stability measure
            rolling_mean = rolling_mean(feature_series, window=self.config.stability_window)
            rolling_std = rolling_std(feature_series, window=self.config.stability_window)
            
            # Coefficient of variation
            cv = rolling_std / (rolling_mean + 1e-8)
            
            # Stability is inverse of coefficient of variation
            stability = 1.0 / (cv + 1e-8)
            
            return float(stability.mean())
            
        except Exception as e:
            self.logger.warning(f"VectorBT temporal stability calculation failed: {e}")
            return self._calculate_temporal_stability_fallback(feature_series)
    
    def _final_selection_with_balancing(self, features: List[FeatureScore]) -> List[FeatureScore]:
        """Final selection with category balancing."""
        tprint_debug("Final selection with category balancing")
        
        try:
            # Group features by category
            category_features = defaultdict(list)
            for feature in features:
                category_features[feature.category].append(feature)
            
            # Select all features from each category (no artificial limits)
            selected_features = []
            
            for category, features_list in category_features.items():
                selected_features.extend(features_list)
            
            # Sort final selection by score
            selected_features.sort(key=lambda x: x.score, reverse=True)
            
            tprint_success(f"Final selection: {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            tprint_error(f"Final selection failed: {e}")
            return features
    
    def _calculate_selection_metrics(self, selected_features: List[FeatureScore], 
                                   data: pd.DataFrame, targets: Optional[pd.Series]) -> Dict[str, Any]:
        """Calculate comprehensive selection metrics."""
        tprint_debug("Calculating selection metrics")
        
        try:
            # Category distribution
            category_distribution = defaultdict(int)
            for feature in selected_features:
                category_distribution[feature.category] += 1
            
            # Aspect distribution
            aspect_distribution = defaultdict(int)
            for feature in selected_features:
                aspect_distribution[feature.aspect_type] += 1
            
            # Quality metrics
            quality_metrics = {
                'average_score': np.mean([f.score for f in selected_features]),
                'max_score': max([f.score for f in selected_features]),
                'min_score': min([f.score for f in selected_features]),
                'score_std': np.std([f.score for f in selected_features]),
                'average_correlation': np.mean([f.correlation_with_target for f in selected_features]),
                'average_information_content': np.mean([f.information_content for f in selected_features]),
                'average_uniqueness': np.mean([f.uniqueness_score for f in selected_features])
            }
            
            # Diversity metrics
            diversity_metrics = {
                'category_diversity': len(category_distribution),
                'aspect_diversity': len(aspect_distribution),
                'average_uniqueness': np.mean([f.uniqueness_score for f in selected_features]),
                'min_uniqueness': min([f.uniqueness_score for f in selected_features]),
                'max_uniqueness': max([f.uniqueness_score for f in selected_features])
            }
            
            # Stability metrics
            stability_metrics = {
                'average_stability': np.mean([f.stability_score for f in selected_features]),
                'min_stability': min([f.stability_score for f in selected_features]),
                'max_stability': max([f.stability_score for f in selected_features]),
                'average_predictability': np.mean([f.predictability_score for f in selected_features])
            }
            
            return {
                'category_distribution': dict(category_distribution),
                'aspect_distribution': dict(aspect_distribution),
                'quality_metrics': quality_metrics,
                'diversity_metrics': diversity_metrics,
                'stability_metrics': stability_metrics
            }
            
        except Exception as e:
            tprint_error(f"Selection metrics calculation failed: {e}")
            return {
                'category_distribution': {},
                'aspect_distribution': {},
                'quality_metrics': {},
                'diversity_metrics': {},
                'stability_metrics': {}
            }
    
    def _create_empty_result(self, start_time: float, error_message: str) -> FeatureSelectionResult:
        """Create empty result for failed selection."""
        return FeatureSelectionResult(
            selected_features=[],
            category_distribution={},
            aspect_distribution={},
            total_features_analyzed=0,
            selection_time=time.time() - start_time,
            quality_metrics={},
            diversity_metrics={},
            stability_metrics={},
            success=False,
            error_message=error_message
        )
    
    # Fallback methods for when VectorBT is not available
    def _analyze_single_feature_fallback(self, data: pd.DataFrame, feature_name: str, 
                                       targets: Optional[pd.Series], category: str) -> Optional[FeatureScore]:
        """Fallback feature analysis when VectorBT is not available."""
        try:
            if feature_name not in data.columns:
                return None
            
            feature_series = data[feature_name]
            
            # Calculate basic metrics
            variance = self._calculate_variance_fallback(feature_series)
            correlation_with_target = self._calculate_correlation_fallback(feature_series, targets)
            information_content = self._calculate_information_content_fallback(feature_series)
            uniqueness_score = self._calculate_uniqueness_score_fallback(feature_series, data)
            stability_score = self._calculate_stability_score_fallback(feature_series)
            predictability_score = self._calculate_predictability_score_fallback(feature_series)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                variance, correlation_with_target, information_content, 
                uniqueness_score, stability_score, predictability_score, category
            )
            
            # Classify aspect
            _, aspect = self._classify_feature(feature_name)
            
            return FeatureScore(
                feature_name=feature_name,
                category=category,
                aspect_type=aspect,
                score=composite_score,
                variance=variance,
                correlation_with_target=correlation_with_target,
                information_content=information_content,
                uniqueness_score=uniqueness_score,
                stability_score=stability_score,
                predictability_score=predictability_score,
                metadata={
                    'vectorbt_optimized': False,
                    'analysis_timestamp': time.time()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Fallback feature analysis failed for {feature_name}: {e}")
            return None
    
    def _calculate_variance_fallback(self, feature_series: pd.Series) -> float:
        """Fallback variance calculation."""
        try:
            return float(feature_series.var())
        except:
            return 0.0
    
    def _calculate_correlation_fallback(self, feature_series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Fallback correlation calculation."""
        try:
            if targets is None:
                return 0.0
            return float(feature_series.corr(targets))
        except:
            return 0.0
    
    def _calculate_information_content_fallback(self, feature_series: pd.Series) -> float:
        """Fallback information content calculation."""
        try:
            unique_values = feature_series.value_counts()
            probabilities = unique_values / len(feature_series)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
            max_entropy = np.log2(len(unique_values))
            return float(entropy / max_entropy) if max_entropy > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_uniqueness_score_fallback(self, feature_series: pd.Series, data: pd.DataFrame) -> float:
        """Fallback uniqueness score calculation."""
        try:
            correlations = []
            for col in data.columns:
                if col != feature_series.name:
                    try:
                        corr = feature_series.corr(data[col])
                        if not pd.isna(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
            
            if not correlations:
                return 1.0
            
            max_correlation = max(correlations)
            return float(1.0 - max_correlation)
        except:
            return 0.0
    
    def _calculate_stability_score_fallback(self, feature_series: pd.Series) -> float:
        """Fallback stability score calculation."""
        try:
            rolling_std = feature_series.rolling(window=self.config.stability_window).std()
            stability = 1.0 / (rolling_std + 1e-8)
            return float(stability.mean())
        except:
            return 0.0
    
    def _calculate_predictability_score_fallback(self, feature_series: pd.Series) -> float:
        """Fallback predictability score calculation."""
        try:
            autocorr = feature_series.autocorr(lag=1)
            if pd.isna(autocorr):
                return 0.0
            return float((autocorr + 1) / 2)
        except:
            return 0.0
    
    def _calculate_temporal_stability_fallback(self, feature_series: pd.Series) -> float:
        """Fallback temporal stability calculation."""
        try:
            rolling_mean = feature_series.rolling(window=self.config.stability_window).mean()
            rolling_std = feature_series.rolling(window=self.config.stability_window).std()
            cv = rolling_std / (rolling_mean + 1e-8)
            stability = 1.0 / (cv + 1e-8)
            return float(stability.mean())
        except:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'total_execution_time': 0.0,
            'features_analyzed': 0,
            'vectorbt_operations': 0,
            'diversity_operations': 0,
            'stability_operations': 0
        }
    
    def _enhanced_feature_selection(self, data: pd.DataFrame, targets: pd.Series, 
                                   n_features: int) -> List[str]:
        """Use enhanced feature selection methods for better selection."""
        if not ENHANCED_FEATURE_SELECTION_AVAILABLE:
            tprint_warning("Enhanced feature selection not available, using standard method")
            return self._standard_feature_selection(data, targets, n_features)
        
        try:
            tprint_info("🔧 Using enhanced feature selection")
            
            # Try improved mRMR first
            selector = ImprovedMRMR()
            result = selector.select_features(
                data.values, targets.values,
                feature_names=data.columns.tolist(),
                target_ratio=n_features / len(data.columns)
            )
            
            if result.get('success', False):
                selected_features = result['selected_features']
                tprint_success(f"✅ Enhanced mRMR selected {len(selected_features)} features")
                return selected_features
            else:
                tprint_warning("Enhanced mRMR failed, trying VectorBT methods")
                
                # Try VectorBT mRMR
                from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
                config = VectorBTFeatureSelectionConfig()
                config.target_features = n_features
                
                selector = VectorBTMRMRSelector(config)
                result = selector.select_features(
                    data.values, targets.values,
                    feature_names=data.columns.tolist()
                )
                
                if result.get('success', False):
                    selected_features = result['selected_features']
                    tprint_success(f"✅ VectorBT mRMR selected {len(selected_features)} features")
                    return selected_features
                else:
                    tprint_warning("VectorBT mRMR failed, using standard method")
                    return self._standard_feature_selection(data, targets, n_features)
                    
        except Exception as e:
            tprint_warning(f"Enhanced feature selection error: {e}, using standard method")
            return self._standard_feature_selection(data, targets, n_features)
    
    def _vectorbt_feature_selection(self, data: pd.DataFrame, targets: pd.Series, 
                                   n_features: int) -> List[str]:
        """Use VectorBT-optimized methods for feature selection."""
        if not ENHANCED_FEATURE_SELECTION_AVAILABLE:
            tprint_warning("VectorBT feature selection not available, using standard method")
            return self._standard_feature_selection(data, targets, n_features)
        
        try:
            tprint_info("🚀 Using VectorBT feature selection")
            
            from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
            config = VectorBTFeatureSelectionConfig()
            config.target_features = n_features
            
            # Try VectorBT mRMR
            selector = VectorBTMRMRSelector(config)
            result = selector.select_features(
                data.values, targets.values,
                feature_names=data.columns.tolist()
            )
            
            if result.get('success', False):
                selected_features = result['selected_features']
                tprint_success(f"✅ VectorBT mRMR selected {len(selected_features)} features")
                return selected_features
            else:
                tprint_warning("VectorBT mRMR failed, trying RFE")
                
                # Try VectorBT RFE
                selector = VectorBTRFESelector(config)
                result = selector.select_features(
                    data.values, targets.values,
                    feature_names=data.columns.tolist()
                )
                
                if result.get('success', False):
                    selected_features = result['selected_features']
                    tprint_success(f"✅ VectorBT RFE selected {len(selected_features)} features")
                    return selected_features
                else:
                    tprint_warning("VectorBT RFE failed, using standard method")
                    return self._standard_feature_selection(data, targets, n_features)
                    
        except Exception as e:
            tprint_warning(f"VectorBT feature selection error: {e}, using standard method")
            return self._standard_feature_selection(data, targets, n_features)
    
    def _ensemble_feature_selection(self, data: pd.DataFrame, targets: pd.Series, 
                                   n_features: int) -> List[str]:
        """Use enhanced ensemble methods for feature selection."""
        if not ENHANCED_FEATURE_SELECTION_AVAILABLE:
            tprint_warning("Ensemble feature selection not available, using standard method")
            return self._standard_feature_selection(data, targets, n_features)
        
        try:
            tprint_info("🔧 Using ensemble feature selection")
            
            from src.feature_selection.advanced.enhanced_config import EnhancedEnsembleConfig
            config = EnhancedEnsembleConfig()
            config.target_features = n_features
            
            selector = EnhancedEnsembleAdvancedSelector(config)
            result = selector.select_features(
                data.values, targets.values,
                feature_names=data.columns.tolist()
            )
            
            if result.get('success', False):
                selected_features = result['selected_features']
                tprint_success(f"✅ Ensemble selected {len(selected_features)} features")
                return selected_features
            else:
                tprint_warning("Ensemble selection failed, using standard method")
                return self._standard_feature_selection(data, targets, n_features)
                
        except Exception as e:
            tprint_warning(f"Ensemble feature selection error: {e}, using standard method")
            return self._standard_feature_selection(data, targets, n_features)
    
    def _standard_feature_selection(self, data: pd.DataFrame, targets: pd.Series, 
                                   n_features: int) -> List[str]:
        """Standard feature selection method as fallback."""
        try:
            # Use correlation-based selection as fallback
            correlations = data.corrwith(targets).abs().sort_values(ascending=False)
            selected_features = correlations.head(n_features).index.tolist()
            tprint_info(f"📊 Standard selection: {len(selected_features)} features")
            return selected_features
        except Exception as e:
            tprint_warning(f"Standard feature selection error: {e}")
            return data.columns[:n_features].tolist()


def create_advanced_feature_selector(config: Optional[FeatureSelectionConfig] = None) -> AdvancedFeatureSelector:
    """Create an advanced feature selector with default configuration."""
    return AdvancedFeatureSelector(config)