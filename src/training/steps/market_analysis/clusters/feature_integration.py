"""
Feature Integration Module for Regime Clustering.

This module integrates enhanced features with the existing clustering pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from .feature_analysis import RegimeFeatureAnalyzer, create_feature_analyzer
from .feature_enhancement import AdvancedFeatureGenerator, create_feature_generator


class RegimeFeatureIntegrator:
    """Integrates enhanced features with regime clustering pipeline."""
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize feature integrator.
        
        Args:
            lookback_periods: Number of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        self.analyzer = create_feature_analyzer(lookback_periods)
        self.generator = create_feature_generator(lookback_periods)
        
    def integrate_enhanced_features(
        self,
        market_data: pd.DataFrame,
        existing_features: Optional[np.ndarray] = None,
        existing_feature_names: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Integrate enhanced features with existing clustering pipeline.
        
        Args:
            market_data: Market data with OHLCV columns
            existing_features: Optional existing feature matrix
            existing_feature_names: Optional existing feature names
            labels: Optional cluster labels for analysis
            
        Returns:
            Dictionary with integrated features and analysis results
        """
        try:
            tprint_info("Starting feature integration process...")
            
            # Step 1: Analyze current features
            analysis_result = self.analyzer.analyze_current_features(
                market_data, existing_features, existing_feature_names, labels
            )
            
            # Step 2: Generate enhanced features
            enhanced_features, enhanced_names = self.generator.generate_enhanced_features(
                market_data, existing_features, existing_feature_names
            )
            
            # Step 3: Combine features
            combined_features, combined_names = self._combine_features(
                existing_features, existing_feature_names,
                enhanced_features, enhanced_names
            )
            
            # Step 4: Feature selection and optimization
            optimized_features, optimized_names = self._optimize_features(
                combined_features, combined_names, labels
            )
            
            # Step 5: Create integration report
            integration_report = self._create_integration_report(
                analysis_result, enhanced_features, enhanced_names,
                optimized_features, optimized_names
            )
            
            result = {
                'features': optimized_features,
                'feature_names': optimized_names,
                'analysis_result': analysis_result,
                'enhanced_features': enhanced_features,
                'enhanced_names': enhanced_names,
                'integration_report': integration_report,
                'feature_count': len(optimized_names),
                'enhancement_ratio': len(enhanced_names) / max(len(existing_feature_names or []), 1)
            }
            
            tprint_success(f"Feature integration completed. {len(optimized_names)} total features.")
            return result
            
        except Exception as e:
            tprint_error(f"Feature integration failed: {e}")
            return {
                'features': existing_features or np.array([]).reshape(len(market_data), 0),
                'feature_names': existing_feature_names or [],
                'analysis_result': None,
                'enhanced_features': None,
                'enhanced_names': [],
                'integration_report': {'error': str(e)},
                'feature_count': len(existing_feature_names or []),
                'enhancement_ratio': 0.0
            }
    
    def _combine_features(
        self,
        existing_features: Optional[np.ndarray],
        existing_names: Optional[List[str]],
        enhanced_features: np.ndarray,
        enhanced_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Combine existing and enhanced features."""
        try:
            combined_features = []
            combined_names = []
            
            # Add existing features if available
            if existing_features is not None and existing_names is not None:
                combined_features.append(existing_features)
                combined_names.extend(existing_names)
            
            # Add enhanced features
            if enhanced_features.size > 0:
                combined_features.append(enhanced_features)
                combined_names.extend(enhanced_names)
            
            # Combine all features
            if combined_features:
                final_features = np.hstack(combined_features)
            else:
                final_features = np.array([]).reshape(0, 0)
            
            return final_features, combined_names
            
        except Exception as e:
            tprint_warning(f"Feature combination failed: {e}")
            return existing_features or np.array([]).reshape(0, 0), existing_names or []
    
    def _optimize_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        labels: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """Optimize features for regime clustering."""
        try:
            if features.size == 0 or labels is None or len(np.unique(labels)) < 2:
                return features, feature_names
            
            # Remove highly correlated features
            features, names = self._remove_correlated_features(features, feature_names)
            
            # Select most important features
            features, names = self._select_important_features(features, names, labels)
            
            # Remove low-variance features
            features, names = self._remove_low_variance_features(features, names)
            
            return features, names
            
        except Exception as e:
            tprint_warning(f"Feature optimization failed: {e}")
            return features, feature_names
    
    def _remove_correlated_features(
        self, 
        features: np.ndarray, 
        feature_names: List[str], 
        threshold: float = 0.95
    ) -> Tuple[np.ndarray, List[str]]:
        """Remove highly correlated features."""
        try:
            if features.shape[1] <= 1:
                return features, feature_names
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            
            # Find highly correlated pairs
            to_remove = set()
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if abs(corr_matrix[i, j]) > threshold:
                        # Remove the feature with lower index (arbitrary choice)
                        to_remove.add(i)
            
            # Keep features not in removal set
            keep_indices = [i for i in range(len(feature_names)) if i not in to_remove]
            
            if keep_indices:
                filtered_features = features[:, keep_indices]
                filtered_names = [feature_names[i] for i in keep_indices]
                return filtered_features, filtered_names
            else:
                return features, feature_names
                
        except Exception as e:
            tprint_warning(f"Correlation removal failed: {e}")
            return features, feature_names
    
    def _select_important_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        labels: np.ndarray,
        max_features: int = 50
    ) -> Tuple[np.ndarray, List[str]]:
        """Select most important features for regime separation."""
        try:
            if features.shape[1] <= max_features:
                return features, feature_names
            
            # Calculate feature importance
            from sklearn.feature_selection import f_classif
            from sklearn.ensemble import RandomForestClassifier
            
            # F-test scores
            f_scores, _ = f_classif(features, labels)
            
            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            rf_importance = rf.feature_importances_
            
            # Combined importance
            combined_importance = 0.6 * f_scores + 0.4 * rf_importance
            
            # Select top features
            top_indices = np.argsort(combined_importance)[-max_features:]
            
            selected_features = features[:, top_indices]
            selected_names = [feature_names[i] for i in top_indices]
            
            return selected_features, selected_names
            
        except Exception as e:
            tprint_warning(f"Feature selection failed: {e}")
            return features, feature_names
    
    def _remove_low_variance_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.01
    ) -> Tuple[np.ndarray, List[str]]:
        """Remove low-variance features."""
        try:
            if features.shape[1] == 0:
                return features, feature_names
            
            # Calculate feature variances
            variances = np.var(features, axis=0)
            
            # Keep features with variance above threshold
            keep_indices = np.where(variances > threshold)[0]
            
            if len(keep_indices) > 0:
                filtered_features = features[:, keep_indices]
                filtered_names = [feature_names[i] for i in keep_indices]
                return filtered_features, filtered_names
            else:
                return features, feature_names
                
        except Exception as e:
            tprint_warning(f"Low variance removal failed: {e}")
            return features, feature_names
    
    def _create_integration_report(
        self,
        analysis_result: Any,
        enhanced_features: np.ndarray,
        enhanced_names: List[str],
        optimized_features: np.ndarray,
        optimized_names: List[str]
    ) -> Dict[str, Any]:
        """Create integration report."""
        try:
            report = {
                'original_feature_count': len(analysis_result.current_features) if analysis_result else 0,
                'enhanced_feature_count': len(enhanced_names),
                'final_feature_count': len(optimized_names),
                'feature_enhancement_ratio': len(enhanced_names) / max(len(analysis_result.current_features) if analysis_result else 1, 1),
                'optimization_ratio': len(optimized_names) / max(len(enhanced_names), 1),
                'separation_score': analysis_result.regime_separation_score if analysis_result else 0.0,
                'recommendations': analysis_result.recommendations if analysis_result else [],
                'missing_features_added': len(enhanced_names),
                'feature_categories': self._categorize_enhanced_features(enhanced_names)
            }
            
            return report
            
        except Exception as e:
            tprint_warning(f"Report creation failed: {e}")
            return {'error': str(e)}
    
    def _categorize_enhanced_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize enhanced features by type."""
        categories = {
            'volatility': 0,
            'trend': 0,
            'momentum': 0,
            'volume': 0,
            'regime': 0,
            'economic': 0,
            'microstructure': 0,
            'other': 0
        }
        
        for name in feature_names:
            name_lower = name.lower()
            
            if any(x in name_lower for x in ['vol', 'volatility', 'atr']):
                categories['volatility'] += 1
            elif any(x in name_lower for x in ['sma', 'ema', 'trend', 'adx', 'dmi']):
                categories['trend'] += 1
            elif any(x in name_lower for x in ['rsi', 'macd', 'stoch', 'williams', 'roc', 'momentum']):
                categories['momentum'] += 1
            elif any(x in name_lower for x in ['volume', 'obv', 'ad_line', 'mfi', 'vpt']):
                categories['volume'] += 1
            elif any(x in name_lower for x in ['regime', 'persistence', 'transition', 'stability']):
                categories['regime'] += 1
            elif any(x in name_lower for x in ['bull', 'bear', 'fear', 'greed', 'sentiment', 'recession']):
                categories['economic'] += 1
            elif any(x in name_lower for x in ['spread', 'impact', 'flow', 'efficiency']):
                categories['microstructure'] += 1
            else:
                categories['other'] += 1
        
        return categories


def create_feature_integrator(lookback_periods: int = 20) -> RegimeFeatureIntegrator:
    """Create feature integrator instance."""
    return RegimeFeatureIntegrator(lookback_periods=lookback_periods)