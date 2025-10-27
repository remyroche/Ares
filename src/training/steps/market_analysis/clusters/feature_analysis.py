"""
Feature Analysis and Enhancement for Regime Clustering.

This module analyzes current features and identifies missing regime separation features
to improve clustering quality and economic relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)


@dataclass
class FeatureAnalysisResult:
    """Results of feature analysis."""
    current_features: List[str]
    missing_features: List[str]
    feature_importance: Dict[str, float]
    regime_separation_score: float
    recommendations: List[str]
    enhanced_features: Optional[pd.DataFrame] = None


class RegimeFeatureAnalyzer:
    """Analyzes and enhances features for regime clustering."""
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize feature analyzer.
        
        Args:
            lookback_periods: Number of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        
    def analyze_current_features(
        self, 
        market_data: pd.DataFrame, 
        features: np.ndarray, 
        feature_names: List[str],
        labels: Optional[np.ndarray] = None
    ) -> FeatureAnalysisResult:
        """
        Analyze current features and identify missing regime separation features.
        
        Args:
            market_data: Market data with OHLCV columns
            features: Feature matrix
            feature_names: Names of current features
            labels: Optional cluster labels for analysis
            
        Returns:
            FeatureAnalysisResult with analysis and recommendations
        """
        try:
            tprint_info("Starting comprehensive feature analysis...")
            
            # Analyze current features
            current_analysis = self._analyze_existing_features(features, feature_names, labels)
            
            # Identify missing features
            missing_features = self._identify_missing_features(market_data, feature_names)
            
            # Calculate regime separation score
            separation_score = self._calculate_regime_separation_score(features, labels)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                current_analysis, missing_features, separation_score
            )
            
            # Create enhanced features
            enhanced_features = self._create_enhanced_features(market_data, features, feature_names)
            
            result = FeatureAnalysisResult(
                current_features=feature_names,
                missing_features=missing_features,
                feature_importance=current_analysis.get('importance', {}),
                regime_separation_score=separation_score,
                recommendations=recommendations,
                enhanced_features=enhanced_features
            )
            
            tprint_success(f"Feature analysis completed. Separation score: {separation_score:.3f}")
            return result
            
        except Exception as e:
            tprint_error(f"Feature analysis failed: {e}")
            return FeatureAnalysisResult(
                current_features=feature_names,
                missing_features=[],
                feature_importance={},
                regime_separation_score=0.0,
                recommendations=[f"Analysis failed: {e}"]
            )
    
    def _analyze_existing_features(
        self, 
        features: np.ndarray, 
        feature_names: List[str], 
        labels: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze existing features for regime separation capability."""
        try:
            analysis = {
                'feature_count': len(feature_names),
                'feature_types': self._categorize_features(feature_names),
                'importance': {},
                'correlations': {},
                'variance_ratios': {}
            }
            
            if labels is not None and len(np.unique(labels)) > 1:
                # Calculate feature importance for regime separation
                analysis['importance'] = self._calculate_feature_importance(features, labels, feature_names)
                
                # Calculate variance ratios
                analysis['variance_ratios'] = self._calculate_variance_ratios(features, labels, feature_names)
                
                # Calculate feature correlations
                analysis['correlations'] = self._calculate_feature_correlations(features, feature_names)
            
            return analysis
            
        except Exception as e:
            tprint_warning(f"Feature analysis failed: {e}")
            return {'feature_count': len(feature_names), 'feature_types': {}, 'importance': {}}
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'price_features': [],
            'volatility_features': [],
            'trend_features': [],
            'volume_features': [],
            'momentum_features': [],
            'regime_features': [],
            'technical_features': [],
            'other_features': []
        }
        
        for name in feature_names:
            name_lower = name.lower()
            
            if any(x in name_lower for x in ['close', 'open', 'high', 'low', 'price', 'return']):
                categories['price_features'].append(name)
            elif any(x in name_lower for x in ['vol', 'volatility', 'std', 'atr']):
                categories['volatility_features'].append(name)
            elif any(x in name_lower for x in ['sma', 'ema', 'trend', 'ma_', 'moving']):
                categories['trend_features'].append(name)
            elif any(x in name_lower for x in ['volume', 'vol_']):
                categories['volume_features'].append(name)
            elif any(x in name_lower for x in ['rsi', 'macd', 'momentum', 'roc', 'stoch']):
                categories['momentum_features'].append(name)
            elif any(x in name_lower for x in ['regime', 'persistence', 'transition']):
                categories['regime_features'].append(name)
            elif any(x in name_lower for x in ['bollinger', 'bb_', 'rsi', 'macd', 'stoch', 'williams']):
                categories['technical_features'].append(name)
            else:
                categories['other_features'].append(name)
        
        return categories
    
    def _calculate_feature_importance(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance for regime separation."""
        try:
            from sklearn.feature_selection import f_classif
            from sklearn.ensemble import RandomForestClassifier
            
            # F-test for feature importance
            f_scores, _ = f_classif(features, labels)
            f_importance = f_scores / (np.sum(f_scores) + 1e-8)
            
            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            rf_importance = rf.feature_importances_
            
            # Combined importance (weighted average)
            combined_importance = 0.6 * f_importance + 0.4 * rf_importance
            
            importance_dict = {}
            for i, name in enumerate(feature_names):
                importance_dict[name] = float(combined_importance[i])
            
            return importance_dict
            
        except Exception as e:
            tprint_warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_variance_ratios(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate variance ratios for each feature."""
        try:
            variance_ratios = {}
            unique_labels = np.unique(labels)
            
            for i, name in enumerate(feature_names):
                feature_data = features[:, i]
                
                # Calculate within-cluster variance
                within_var = 0.0
                total_samples = 0
                
                for label in unique_labels:
                    cluster_data = feature_data[labels == label]
                    if len(cluster_data) > 1:
                        cluster_var = np.var(cluster_data)
                        within_var += cluster_var * len(cluster_data)
                        total_samples += len(cluster_data)
                
                if total_samples > 0:
                    within_var /= total_samples
                
                # Calculate between-cluster variance
                overall_mean = np.mean(feature_data)
                between_var = 0.0
                
                for label in unique_labels:
                    cluster_data = feature_data[labels == label]
                    if len(cluster_data) > 0:
                        cluster_mean = np.mean(cluster_data)
                        between_var += len(cluster_data) * (cluster_mean - overall_mean) ** 2
                
                if total_samples > 0:
                    between_var /= total_samples
                
                # Variance ratio
                if within_var > 0:
                    variance_ratio = between_var / within_var
                else:
                    variance_ratio = 0.0
                
                variance_ratios[name] = variance_ratio
            
            return variance_ratios
            
        except Exception as e:
            tprint_warning(f"Variance ratio calculation failed: {e}")
            return {}
    
    def _calculate_feature_correlations(
        self, 
        features: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature correlations."""
        try:
            corr_matrix = np.corrcoef(features.T)
            correlations = {}
            
            for i, name in enumerate(feature_names):
                # Calculate average absolute correlation with other features
                other_corrs = np.abs(corr_matrix[i, :])
                other_corrs = np.delete(other_corrs, i)  # Remove self-correlation
                correlations[name] = float(np.mean(other_corrs))
            
            return correlations
            
        except Exception as e:
            tprint_warning(f"Correlation calculation failed: {e}")
            return {}
    
    def _identify_missing_features(
        self, 
        market_data: pd.DataFrame, 
        current_features: List[str]
    ) -> List[str]:
        """Identify missing features that could improve regime separation."""
        try:
            missing_features = []
            current_lower = [f.lower() for f in current_features]
            
            # Check for basic OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            
            if missing_columns:
                missing_features.extend([f"Missing basic data: {missing_columns}"])
            
            # Volatility features
            vol_features = [
                'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
                'atr_5', 'atr_10', 'atr_20', 'atr_60',
                'vol_regime_zscore', 'vol_regime_percentile', 'vol_regime_transition'
            ]
            for feature in vol_features:
                if feature.lower() not in current_lower:
                    missing_features.append(f"Volatility: {feature}")
            
            # Trend features
            trend_features = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'trend_strength', 'trend_consistency', 'trend_acceleration',
                'adx', 'dmi_plus', 'dmi_minus'
            ]
            for feature in trend_features:
                if feature.lower() not in current_lower:
                    missing_features.append(f"Trend: {feature}")
            
            # Momentum features
            momentum_features = [
                'rsi_14', 'rsi_21', 'rsi_50',
                'macd', 'macd_signal', 'macd_histogram',
                'stoch_k', 'stoch_d', 'williams_r',
                'roc_5', 'roc_10', 'roc_20',
                'momentum_5', 'momentum_10', 'momentum_20'
            ]
            for feature in momentum_features:
                if feature.lower() not in current_lower:
                    missing_features.append(f"Momentum: {feature}")
            
            # Volume features
            volume_features = [
                'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
                'volume_ratio', 'volume_regime', 'volume_regime_percentile',
                'obv', 'ad_line', 'cmf', 'mfi'
            ]
            for feature in volume_features:
                if feature.lower() not in current_lower:
                    missing_features.append(f"Volume: {feature}")
            
            # Regime-specific features
            regime_features = [
                'regime_persistence', 'regime_transition_probability',
                'regime_stability', 'regime_volatility_cluster',
                'regime_trend_cluster', 'regime_volume_cluster'
            ]
            for feature in regime_features:
                if feature.lower() not in current_lower:
                    missing_features.append(f"Regime: {feature}")
            
            # Economic features
            economic_features = [
                'vix_proxy', 'fear_greed_index', 'market_sentiment',
                'sector_rotation', 'market_breadth', 'advance_decline_ratio'
            ]
            for feature in economic_features:
                if feature.lower() not in current_lower:
                    missing_features.append(f"Economic: {feature}")
            
            return missing_features
            
        except Exception as e:
            tprint_warning(f"Missing feature identification failed: {e}")
            return []
    
    def _calculate_regime_separation_score(
        self, 
        features: np.ndarray, 
        labels: Optional[np.ndarray]
    ) -> float:
        """Calculate overall regime separation score."""
        try:
            if labels is None or len(np.unique(labels)) < 2:
                return 0.0
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(features, labels)
            
            # Calculate variance ratio
            variance_ratio = self._calculate_overall_variance_ratio(features, labels)
            
            # Calculate feature diversity
            feature_diversity = self._calculate_feature_diversity(features)
            
            # Combined score
            separation_score = 0.4 * silhouette + 0.4 * variance_ratio + 0.2 * feature_diversity
            
            return separation_score
            
        except Exception as e:
            tprint_warning(f"Separation score calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_variance_ratio(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate overall variance ratio."""
        try:
            unique_labels = np.unique(labels)
            
            # Within-cluster variance
            within_var = 0.0
            total_samples = 0
            
            for label in unique_labels:
                cluster_data = features[labels == label]
                if len(cluster_data) > 1:
                    cluster_var = np.var(cluster_data, axis=0).sum()
                    within_var += cluster_var * len(cluster_data)
                    total_samples += len(cluster_data)
            
            if total_samples > 0:
                within_var /= total_samples
            
            # Between-cluster variance
            overall_mean = np.mean(features, axis=0)
            between_var = 0.0
            
            for label in unique_labels:
                cluster_data = features[labels == label]
                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data, axis=0)
                    between_var += len(cluster_data) * np.sum((cluster_mean - overall_mean) ** 2)
            
            if total_samples > 0:
                between_var /= total_samples
            
            # Variance ratio
            if within_var > 0:
                return between_var / within_var
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_feature_diversity(self, features: np.ndarray) -> float:
        """Calculate feature diversity score."""
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            
            # Remove diagonal (self-correlation)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            off_diagonal_corrs = corr_matrix[mask]
            
            # Diversity is inverse of average correlation
            avg_correlation = np.mean(np.abs(off_diagonal_corrs))
            diversity = 1.0 - avg_correlation
            
            return max(0.0, diversity)
            
        except Exception:
            return 0.0
    
    def _generate_recommendations(
        self, 
        current_analysis: Dict[str, Any], 
        missing_features: List[str], 
        separation_score: float
    ) -> List[str]:
        """Generate recommendations for improving regime separation."""
        try:
            recommendations = []
            
            # Overall separation score
            if separation_score < 0.3:
                recommendations.append("CRITICAL: Very low regime separation score. Consider adding more discriminative features.")
            elif separation_score < 0.5:
                recommendations.append("WARNING: Low regime separation score. Feature enhancement recommended.")
            elif separation_score < 0.7:
                recommendations.append("GOOD: Moderate regime separation. Minor improvements possible.")
            else:
                recommendations.append("EXCELLENT: High regime separation score.")
            
            # Feature type analysis
            feature_types = current_analysis.get('feature_types', {})
            
            if len(feature_types.get('volatility_features', [])) < 3:
                recommendations.append("Add more volatility features (ATR, GARCH, volatility regimes)")
            
            if len(feature_types.get('trend_features', [])) < 3:
                recommendations.append("Add more trend features (multiple timeframes, trend strength indicators)")
            
            if len(feature_types.get('regime_features', [])) < 2:
                recommendations.append("Add regime-specific features (persistence, transitions, stability)")
            
            if len(feature_types.get('momentum_features', [])) < 3:
                recommendations.append("Add momentum indicators (RSI, MACD, Stochastic, Williams %R)")
            
            if len(feature_types.get('volume_features', [])) < 2:
                recommendations.append("Add volume-based features (OBV, AD Line, Volume Rate of Change)")
            
            # Feature importance analysis
            importance = current_analysis.get('importance', {})
            if importance:
                low_importance_features = [name for name, imp in importance.items() if imp < 0.01]
                if low_importance_features:
                    recommendations.append(f"Consider removing low-importance features: {low_importance_features[:5]}")
            
            # Missing features
            if missing_features:
                recommendations.append(f"Add missing features: {len(missing_features)} identified")
            
            return recommendations
            
        except Exception as e:
            tprint_warning(f"Recommendation generation failed: {e}")
            return [f"Recommendation generation failed: {e}"]
    
    def _create_enhanced_features(
        self, 
        market_data: pd.DataFrame, 
        current_features: np.ndarray, 
        feature_names: List[str]
    ) -> Optional[pd.DataFrame]:
        """Create enhanced features for better regime separation."""
        try:
            if 'close' not in market_data.columns:
                return None
            
            enhanced_data = market_data.copy()
            
            # Add enhanced volatility features
            if 'close' in enhanced_data.columns:
                returns = enhanced_data['close'].pct_change().dropna()
                
                # Multi-timeframe volatility
                for period in [5, 10, 20, 40, 60]:
                    enhanced_data[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
                
                # Volatility regime features
                vol_20 = enhanced_data['volatility_20']
                vol_mean_60 = vol_20.rolling(60).mean()
                vol_std_60 = vol_20.rolling(60).std()
                enhanced_data['vol_regime_zscore'] = (vol_20 - vol_mean_60) / (vol_std_60 + 1e-8)
                enhanced_data['vol_regime_percentile'] = vol_20.rolling(252).rank(pct=True)
                
                # Volatility clustering
                enhanced_data['vol_cluster_high'] = (vol_20 > vol_mean_60 + vol_std_60).astype(int)
                enhanced_data['vol_cluster_low'] = (vol_20 < vol_mean_60 - vol_std_60).astype(int)
            
            # Add trend features
            if 'close' in enhanced_data.columns:
                close = enhanced_data['close']
                
                # Multiple timeframes
                for period in [5, 10, 20, 50, 200]:
                    enhanced_data[f'sma_{period}'] = close.rolling(period).mean()
                    enhanced_data[f'ema_{period}'] = close.ewm(span=period).mean()
                
                # Trend strength
                sma_20 = enhanced_data['sma_20']
                sma_5 = enhanced_data['sma_5']
                enhanced_data['trend_strength'] = np.abs(sma_20 - sma_5) / (close + 1e-8)
                
                # Trend consistency
                enhanced_data['trend_consistency'] = (returns > 0).rolling(20).mean() - 0.5
            
            # Add momentum features
            if 'close' in enhanced_data.columns:
                # RSI
                delta = returns.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                enhanced_data['rsi_14'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                enhanced_data['macd'] = ema_12 - ema_26
                enhanced_data['macd_signal'] = enhanced_data['macd'].ewm(span=9).mean()
                enhanced_data['macd_histogram'] = enhanced_data['macd'] - enhanced_data['macd_signal']
            
            # Add volume features
            if 'volume' in enhanced_data.columns:
                volume = enhanced_data['volume']
                
                # Volume moving averages
                for period in [5, 10, 20]:
                    enhanced_data[f'volume_sma_{period}'] = volume.rolling(period).mean()
                
                # Volume ratio
                enhanced_data['volume_ratio'] = volume / (enhanced_data['volume_sma_20'] + 1e-8)
                
                # Volume regime
                enhanced_data['volume_regime'] = enhanced_data['volume_ratio'].rolling(252).rank(pct=True)
            
            # Add regime persistence features
            if 'vol_regime_percentile' in enhanced_data.columns:
                enhanced_data['regime_persistence'] = self._calculate_regime_persistence(
                    enhanced_data['vol_regime_percentile']
                )
            
            return enhanced_data
            
        except Exception as e:
            tprint_warning(f"Enhanced feature creation failed: {e}")
            return None
    
    def _calculate_regime_persistence(self, regime_signal: pd.Series, threshold: float = 0.5) -> pd.Series:
        """Calculate regime persistence."""
        try:
            # Binary regime indicator
            regime_binary = (regime_signal > threshold).astype(int)
            
            # Count consecutive periods
            persistence = pd.Series(0, index=regime_signal.index)
            count = 0
            prev_regime = -1
            
            for i, regime in enumerate(regime_binary):
                if regime == prev_regime:
                    count += 1
                else:
                    count = 1
                persistence.iloc[i] = count
                prev_regime = regime
            
            return persistence
            
        except Exception:
            return pd.Series(0, index=regime_signal.index)


def create_feature_analyzer(lookback_periods: int = 20) -> RegimeFeatureAnalyzer:
    """Create feature analyzer instance."""
    return RegimeFeatureAnalyzer(lookback_periods=lookback_periods)