"""
VectorBT Advanced Correlation Analyzer

This module provides advanced correlation analysis using VectorBT's
sophisticated financial correlation metrics and time series analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Import VectorBT with fallback
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available. Using fallback implementations.")

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error, tprint_performance
from src.utils.dependency_manager import DependencyManager

logger = logging.getLogger(__name__)

@dataclass
class VectorBTCorrelationConfig:
    """Configuration for VectorBT correlation analysis."""
    # Correlation methods
    enable_pearson: bool = True
    enable_spearman: bool = True
    enable_kendall: bool = True
    enable_rolling_correlation: bool = True
    enable_rank_correlation: bool = True
    
    # Rolling correlation parameters
    rolling_window: int = 30
    min_periods: int = 10
    
    # Time series correlation parameters
    enable_lagged_correlation: bool = True
    max_lags: int = 10
    lag_selection_method: str = 'aic'  # 'aic', 'bic', 'max'
    
    # Financial correlation parameters
    enable_returns_correlation: bool = True
    enable_volatility_correlation: bool = True
    enable_volume_correlation: bool = False
    
    # Clustering parameters
    enable_correlation_clustering: bool = True
    clustering_method: str = 'hierarchical'  # 'hierarchical', 'kmeans', 'dbscan'
    n_clusters: Optional[int] = None
    correlation_threshold: float = 0.8
    
    # Performance optimization
    enable_parallel: bool = True
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True

@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    feature_names: List[str]
    correlation_matrix: pd.DataFrame
    rolling_correlations: Dict[str, pd.DataFrame]
    lagged_correlations: Dict[str, Dict[str, float]]
    correlation_clusters: Dict[str, List[str]]
    correlation_strength: Dict[str, float]
    correlation_stability: Dict[str, float]
    financial_correlations: Dict[str, Dict[str, float]]
    analysis_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class VectorBTCorrelationAnalyzer:
    """Advanced correlation analyzer using VectorBT."""
    
    def __init__(self, config: Optional[VectorBTCorrelationConfig] = None):
        """Initialize VectorBT correlation analyzer."""
        self.config = config or VectorBTCorrelationConfig()
        self.logger = logger.getChild('VectorBTCorrelationAnalyzer')
        self.dependency_manager = DependencyManager()
        
        # Check VectorBT availability
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available. Using fallback implementations.")
            self.vectorbt_available = False
        else:
            self.vectorbt_available = True
            tprint_success("✅ VectorBT available for advanced correlation analysis")
        
        # Performance tracking
        self.performance_stats = {
            'analyses_performed': 0,
            'correlation_matrices_calculated': 0,
            'rolling_correlations_calculated': 0,
            'total_time': 0.0
        }
        
        tprint_success("🚀 VectorBTCorrelationAnalyzer initialized")
    
    def _calculate_basic_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate basic correlation matrices."""
        correlations = {}
        
        try:
            if self.config.enable_pearson:
                correlations['pearson'] = data.corr(method='pearson')
            
            if self.config.enable_spearman:
                correlations['spearman'] = data.corr(method='spearman')
            
            if self.config.enable_kendall:
                correlations['kendall'] = data.corr(method='kendall')
            
            self.performance_stats['correlation_matrices_calculated'] += len(correlations)
            return correlations
            
        except Exception as e:
            self.logger.warning(f"Basic correlation calculation failed: {e}")
            return {}
    
    def _calculate_rolling_correlations(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate rolling correlations using VectorBT."""
        if not self.vectorbt_available or len(data) < self.config.rolling_window:
            return self._calculate_rolling_correlations_fallback(data)
        
        try:
            rolling_correlations = {}
            
            if self.config.enable_rolling_correlation:
                # Use VectorBT's rolling correlation
                for i, col1 in enumerate(data.columns):
                    for j, col2 in enumerate(data.columns):
                        if i < j:  # Avoid duplicates
                            try:
                                rolling_corr = vbt.rolling_corr(
                                    data[col1], 
                                    data[col2], 
                                    window=self.config.rolling_window,
                                    min_periods=self.config.min_periods
                                )
                                
                                pair_name = f"{col1}_{col2}"
                                rolling_correlations[pair_name] = rolling_corr
                                
                            except Exception as e:
                                self.logger.warning(f"Rolling correlation failed for {col1}-{col2}: {e}")
                                continue
            
            self.performance_stats['rolling_correlations_calculated'] += len(rolling_correlations)
            return rolling_correlations
            
        except Exception as e:
            self.logger.warning(f"Rolling correlation calculation failed: {e}")
            return self._calculate_rolling_correlations_fallback(data)
    
    def _calculate_rolling_correlations_fallback(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Fallback rolling correlation calculation without VectorBT."""
        rolling_correlations = {}
        
        if len(data) < self.config.rolling_window:
            return rolling_correlations
        
        try:
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns):
                    if i < j:  # Avoid duplicates
                        rolling_corr = data[col1].rolling(
                            window=self.config.rolling_window,
                            min_periods=self.config.min_periods
                        ).corr(data[col2])
                        
                        pair_name = f"{col1}_{col2}"
                        rolling_correlations[pair_name] = rolling_corr
        
        except Exception as e:
            self.logger.warning(f"Fallback rolling correlation failed: {e}")
        
        return rolling_correlations
    
    def _calculate_lagged_correlations(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate lagged correlations using VectorBT."""
        if not self.vectorbt_available or len(data) < self.config.max_lags + 10:
            return self._calculate_lagged_correlations_fallback(data)
        
        try:
            lagged_correlations = {}
            
            if self.config.enable_lagged_correlation:
                for i, col1 in enumerate(data.columns):
                    for j, col2 in enumerate(data.columns):
                        if i != j:  # Don't correlate with itself
                            try:
                                # Use VectorBT's lagged correlation
                                lags = range(1, min(self.config.max_lags + 1, len(data) // 4))
                                lag_correlations = {}
                                
                                for lag in lags:
                                    if len(data) > lag:
                                        # Calculate correlation with lag
                                        corr = vbt.correlation(
                                            data[col1].iloc[lag:], 
                                            data[col2].iloc[:-lag]
                                        )
                                        lag_correlations[f"lag_{lag}"] = float(corr) if not pd.isna(corr) else 0.0
                                
                                # Find optimal lag
                                if lag_correlations:
                                    optimal_lag = max(lag_correlations.keys(), key=lambda k: abs(lag_correlations[k]))
                                    lagged_correlations[f"{col1}_{col2}"] = {
                                        'optimal_lag': optimal_lag,
                                        'optimal_correlation': lag_correlations[optimal_lag],
                                        'all_lags': lag_correlations
                                    }
                                
                            except Exception as e:
                                self.logger.warning(f"Lagged correlation failed for {col1}-{col2}: {e}")
                                continue
            
            return lagged_correlations
            
        except Exception as e:
            self.logger.warning(f"Lagged correlation calculation failed: {e}")
            return self._calculate_lagged_correlations_fallback(data)
    
    def _calculate_lagged_correlations_fallback(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Fallback lagged correlation calculation without VectorBT."""
        lagged_correlations = {}
        
        if len(data) < self.config.max_lags + 10:
            return lagged_correlations
        
        try:
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns):
                    if i != j:  # Don't correlate with itself
                        lags = range(1, min(self.config.max_lags + 1, len(data) // 4))
                        lag_correlations = {}
                        
                        for lag in lags:
                            if len(data) > lag:
                                corr = data[col1].iloc[lag:].corr(data[col2].iloc[:-lag])
                                if not pd.isna(corr):
                                    lag_correlations[f"lag_{lag}"] = float(corr)
                        
                        if lag_correlations:
                            optimal_lag = max(lag_correlations.keys(), key=lambda k: abs(lag_correlations[k]))
                            lagged_correlations[f"{col1}_{col2}"] = {
                                'optimal_lag': optimal_lag,
                                'optimal_correlation': lag_correlations[optimal_lag],
                                'all_lags': lag_correlations
                            }
        
        except Exception as e:
            self.logger.warning(f"Fallback lagged correlation failed: {e}")
        
        return lagged_correlations
    
    def _calculate_financial_correlations(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate financial-specific correlations."""
        financial_correlations = {}
        
        try:
            if self.config.enable_returns_correlation:
                # Calculate returns correlations
                returns = data.pct_change().dropna()
                if not returns.empty:
                    returns_corr = returns.corr()
                    financial_correlations['returns'] = returns_corr.to_dict()
            
            if self.config.enable_volatility_correlation:
                # Calculate volatility correlations
                volatility = data.rolling(window=20).std().dropna()
                if not volatility.empty:
                    vol_corr = volatility.corr()
                    financial_correlations['volatility'] = vol_corr.to_dict()
            
            if self.config.enable_volume_correlation and 'volume' in data.columns:
                # Calculate volume correlations
                volume = data['volume']
                vol_correlations = {}
                for col in data.columns:
                    if col != 'volume':
                        corr = volume.corr(data[col])
                        if not pd.isna(corr):
                            vol_correlations[col] = float(corr)
                financial_correlations['volume'] = vol_correlations
        
        except Exception as e:
            self.logger.warning(f"Financial correlation calculation failed: {e}")
        
        return financial_correlations
    
    def _cluster_correlated_features(self, correlation_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Cluster highly correlated features."""
        if not self.config.enable_correlation_clustering:
            return {}
        
        try:
            clusters = {}
            used_features = set()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_value = abs(correlation_matrix.loc[col1, col2])
                        if corr_value > self.config.correlation_threshold:
                            high_corr_pairs.append((col1, col2, corr_value))
            
            # Sort by correlation strength
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Create clusters
            cluster_id = 0
            for col1, col2, corr_value in high_corr_pairs:
                if col1 not in used_features and col2 not in used_features:
                    cluster_name = f"cluster_{cluster_id}"
                    clusters[cluster_name] = [col1, col2]
                    used_features.add(col1)
                    used_features.add(col2)
                    cluster_id += 1
                elif col1 in used_features and col2 not in used_features:
                    # Add col2 to existing cluster
                    for cluster_name, features in clusters.items():
                        if col1 in features:
                            features.append(col2)
                            used_features.add(col2)
                            break
                elif col2 in used_features and col1 not in used_features:
                    # Add col1 to existing cluster
                    for cluster_name, features in clusters.items():
                        if col2 in features:
                            features.append(col1)
                            used_features.add(col1)
                            break
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Correlation clustering failed: {e}")
            return {}
    
    def _calculate_correlation_strength(self, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation strength for each feature."""
        strength = {}
        
        try:
            for col in correlation_matrix.columns:
                # Calculate average absolute correlation with other features
                other_correlations = correlation_matrix[col].drop(col).abs()
                strength[col] = float(other_correlations.mean()) if not other_correlations.empty else 0.0
        
        except Exception as e:
            self.logger.warning(f"Correlation strength calculation failed: {e}")
        
        return strength
    
    def _calculate_correlation_stability(self, rolling_correlations: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate correlation stability over time."""
        stability = {}
        
        try:
            for pair_name, rolling_corr in rolling_correlations.items():
                if not rolling_corr.empty:
                    # Calculate standard deviation of rolling correlations
                    stability[pair_name] = float(rolling_corr.std()) if len(rolling_corr) > 1 else 0.0
        
        except Exception as e:
            self.logger.warning(f"Correlation stability calculation failed: {e}")
        
        return stability
    
    def analyze_correlations(self, 
                           data: Union[np.ndarray, pd.DataFrame],
                           feature_names: Optional[List[str]] = None) -> CorrelationResult:
        """Analyze correlations using VectorBT advanced methods."""
        tprint("🔍 Starting VectorBT correlation analysis")
        
        start_time = datetime.now()
        
        try:
            # Prepare data
            if isinstance(data, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                data_df = pd.DataFrame(data, columns=feature_names)
            else:
                data_df = data.copy()
                if feature_names is None:
                    feature_names = list(data_df.columns)
            
            # Calculate basic correlations
            basic_correlations = self._calculate_basic_correlations(data_df)
            correlation_matrix = basic_correlations.get('pearson', data_df.corr())
            
            # Calculate rolling correlations
            rolling_correlations = self._calculate_rolling_correlations(data_df)
            
            # Calculate lagged correlations
            lagged_correlations = self._calculate_lagged_correlations(data_df)
            
            # Calculate financial correlations
            financial_correlations = self._calculate_financial_correlations(data_df)
            
            # Cluster correlated features
            correlation_clusters = self._cluster_correlated_features(correlation_matrix)
            
            # Calculate correlation strength
            correlation_strength = self._calculate_correlation_strength(correlation_matrix)
            
            # Calculate correlation stability
            correlation_stability = self._calculate_correlation_stability(rolling_correlations)
            
            # Create result
            result = CorrelationResult(
                feature_names=feature_names,
                correlation_matrix=correlation_matrix,
                rolling_correlations=rolling_correlations,
                lagged_correlations=lagged_correlations,
                correlation_clusters=correlation_clusters,
                correlation_strength=correlation_strength,
                correlation_stability=correlation_stability,
                financial_correlations=financial_correlations,
                analysis_metadata={
                    'vectorbt_available': self.vectorbt_available,
                    'data_shape': data_df.shape,
                    'analysis_time': (datetime.now() - start_time).total_seconds(),
                    'config': self.config.__dict__
                }
            )
            
            # Update performance stats
            self.performance_stats['analyses_performed'] += 1
            self.performance_stats['total_time'] += (datetime.now() - start_time).total_seconds()
            
            tprint_success(f"✅ Correlation analysis completed: {len(feature_names)} features, "
                         f"{len(correlation_clusters)} clusters")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            tprint_error(f"❌ Analysis failed: {e}")
            raise
    
    def get_highly_correlated_pairs(self, 
                                  correlation_matrix: pd.DataFrame,
                                  threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Get pairs of highly correlated features."""
        pairs = []
        
        try:
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr_value = abs(correlation_matrix.loc[col1, col2])
                        if corr_value > threshold:
                            pairs.append((col1, col2, corr_value))
            
            # Sort by correlation strength
            pairs.sort(key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"High correlation pairs extraction failed: {e}")
        
        return pairs
    
    def get_correlation_summary(self, result: CorrelationResult) -> Dict[str, Any]:
        """Get summary of correlation analysis."""
        summary = {
            'total_features': len(result.feature_names),
            'correlation_clusters': len(result.correlation_clusters),
            'high_correlation_pairs': len(self.get_highly_correlated_pairs(result.correlation_matrix)),
            'avg_correlation_strength': np.mean(list(result.correlation_strength.values())) if result.correlation_strength else 0.0,
            'avg_correlation_stability': np.mean(list(result.correlation_stability.values())) if result.correlation_stability else 0.0,
            'financial_correlation_types': list(result.financial_correlations.keys())
        }
        
        return summary
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['analyses_performed'] > 0:
            stats['avg_time_per_analysis'] = stats['total_time'] / stats['analyses_performed']
        else:
            stats['avg_time_per_analysis'] = 0.0
        
        tprint_performance(f"📊 VectorBT Correlation Stats: {stats['analyses_performed']} analyses, "
                         f"{stats['avg_time_per_analysis']:.3f}s avg")
        
        return stats

def create_vectorbt_correlation_analyzer(config: Optional[VectorBTCorrelationConfig] = None) -> VectorBTCorrelationAnalyzer:
    """Create a VectorBT correlation analyzer."""
    return VectorBTCorrelationAnalyzer(config)