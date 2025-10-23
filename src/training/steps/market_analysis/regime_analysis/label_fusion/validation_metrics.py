"""
Comprehensive validation metrics for regime analysis and label fusion.

This module provides advanced validation metrics including economic validation,
temporal stability analysis, and regime quality assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation metrics."""
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    ECONOMIC = "economic"
    CLUSTERING = "clustering"
    REGIME_SPECIFIC = "regime_specific"


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    validation_type: ValidationType
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RegimeQualityMetrics:
    """Comprehensive regime quality metrics."""
    # Basic metrics
    n_regimes: int
    regime_sizes: List[int]
    regime_balance: float
    
    # Statistical metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # Temporal metrics
    persistence_score: float
    stability_score: float
    transition_rate: float
    
    # Economic metrics
    economic_consistency: float
    volatility_separation: float
    return_separation: float
    
    # Overall quality
    overall_quality: float
    
    # Validation status
    validation_passed: bool
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RegimeValidator:
    """Comprehensive regime validation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime validator."""
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('RegimeValidator')
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'min_regime_persistence': 0.7,
            'max_feature_noise_ratio': 0.3,
            'min_temporal_stability': 0.6,
            'min_samples_per_regime': 10,
            'max_regime_count': 15,
            'min_regime_balance': 0.1,
            'max_transition_rate': 0.3,
            'min_silhouette_score': 0.3,
            'max_davies_bouldin_score': 2.0,
            'min_calinski_harabasz_score': 100.0,
            'enable_economic_validation': True,
            'enable_temporal_validation': True,
            'enable_statistical_validation': True
        }
    
    def validate_regimes(self, regimes: np.ndarray, 
                        features: Optional[np.ndarray] = None,
                        market_data: Optional[pd.DataFrame] = None,
                        temporal_data: Optional[np.ndarray] = None) -> RegimeQualityMetrics:
        """
        Perform comprehensive regime validation.
        
        Args:
            regimes: Regime labels array
            features: Optional feature matrix
            market_data: Optional market data for economic validation
            temporal_data: Optional temporal data for stability analysis
            
        Returns:
            Comprehensive regime quality metrics
        """
        try:
            self.logger.info("Starting comprehensive regime validation")
            
            # Basic validation
            basic_metrics = self._validate_basic_properties(regimes)
            
            # Statistical validation
            statistical_metrics = self._validate_statistical_properties(regimes, features)
            
            # Temporal validation
            temporal_metrics = self._validate_temporal_properties(regimes, temporal_data)
            
            # Economic validation
            economic_metrics = self._validate_economic_properties(regimes, market_data)
            
            # Combine all metrics
            quality_metrics = self._combine_metrics(
                basic_metrics, statistical_metrics, temporal_metrics, economic_metrics
            )
            
            # Determine overall validation status
            quality_metrics.validation_passed = self._determine_validation_status(quality_metrics)
            
            self.logger.info(f"Regime validation completed. Passed: {quality_metrics.validation_passed}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Regime validation failed: {e}")
            return self._create_error_metrics(str(e))
    
    def _validate_basic_properties(self, regimes: np.ndarray) -> Dict[str, Any]:
        """Validate basic regime properties."""
        try:
            unique_regimes, counts = np.unique(regimes, return_counts=True)
            n_regimes = len(unique_regimes)
            
            # Check regime count bounds
            if n_regimes < 2:
                return {
                    'n_regimes': n_regimes,
                    'regime_sizes': counts.tolist(),
                    'regime_balance': 0.0,
                    'critical_issues': ['Too few regimes detected'],
                    'warnings': []
                }
            
            if n_regimes > self.config['max_regime_count']:
                return {
                    'n_regimes': n_regimes,
                    'regime_sizes': counts.tolist(),
                    'regime_balance': 0.0,
                    'critical_issues': [f'Too many regimes: {n_regimes} > {self.config["max_regime_count"]}'],
                    'warnings': []
                }
            
            # Calculate regime balance
            regime_balance = np.min(counts) / np.max(counts) if np.max(counts) > 0 else 0.0
            
            # Check minimum samples per regime
            min_samples = np.min(counts)
            warnings = []
            if min_samples < self.config['min_samples_per_regime']:
                warnings.append(f'Some regimes have fewer than {self.config["min_samples_per_regime"]} samples')
            
            # Check regime balance
            if regime_balance < self.config['min_regime_balance']:
                warnings.append(f'Regime balance {regime_balance:.3f} is below threshold')
            
            return {
                'n_regimes': n_regimes,
                'regime_sizes': counts.tolist(),
                'regime_balance': regime_balance,
                'critical_issues': [],
                'warnings': warnings
            }
            
        except Exception as e:
            self.logger.error(f"Basic validation failed: {e}")
            return {
                'n_regimes': 0,
                'regime_sizes': [],
                'regime_balance': 0.0,
                'critical_issues': [str(e)],
                'warnings': []
            }
    
    def _validate_statistical_properties(self, regimes: np.ndarray, 
                                       features: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate statistical properties of regimes."""
        try:
            if features is None or len(features) != len(regimes):
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'critical_issues': ['No features provided for statistical validation'],
                    'warnings': []
                }
            
            # Calculate clustering metrics
            if len(np.unique(regimes)) < 2:
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'critical_issues': ['Insufficient regimes for statistical validation'],
                    'warnings': []
                }
            
            silhouette = silhouette_score(features, regimes)
            calinski = calinski_harabasz_score(features, regimes)
            davies = davies_bouldin_score(features, regimes)
            
            # Check thresholds
            warnings = []
            critical_issues = []
            
            if silhouette < self.config['min_silhouette_score']:
                warnings.append(f'Silhouette score {silhouette:.3f} below threshold')
            
            if davies > self.config['max_davies_bouldin_score']:
                warnings.append(f'Davies-Bouldin score {davies:.3f} above threshold')
            
            if calinski < self.config['min_calinski_harabasz_score']:
                warnings.append(f'Calinski-Harabasz score {calinski:.3f} below threshold')
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies,
                'critical_issues': critical_issues,
                'warnings': warnings
            }
            
        except Exception as e:
            self.logger.error(f"Statistical validation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'critical_issues': [str(e)],
                'warnings': []
            }
    
    def _validate_temporal_properties(self, regimes: np.ndarray, 
                                    temporal_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate temporal properties of regimes."""
        try:
            if temporal_data is None or len(temporal_data) != len(regimes):
                return {
                    'persistence_score': 0.5,  # Neutral score
                    'stability_score': 0.5,
                    'transition_rate': 0.0,
                    'critical_issues': ['No temporal data provided'],
                    'warnings': []
                }
            
            # Calculate persistence score
            persistence_score = self._calculate_persistence_score(regimes)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(regimes, temporal_data)
            
            # Calculate transition rate
            transition_rate = self._calculate_transition_rate(regimes)
            
            # Check thresholds
            warnings = []
            critical_issues = []
            
            if persistence_score < self.config['min_regime_persistence']:
                warnings.append(f'Persistence score {persistence_score:.3f} below threshold')
            
            if stability_score < self.config['min_temporal_stability']:
                warnings.append(f'Stability score {stability_score:.3f} below threshold')
            
            if transition_rate > self.config['max_transition_rate']:
                warnings.append(f'Transition rate {transition_rate:.3f} above threshold')
            
            return {
                'persistence_score': persistence_score,
                'stability_score': stability_score,
                'transition_rate': transition_rate,
                'critical_issues': critical_issues,
                'warnings': warnings
            }
            
        except Exception as e:
            self.logger.error(f"Temporal validation failed: {e}")
            return {
                'persistence_score': 0.0,
                'stability_score': 0.0,
                'transition_rate': 1.0,
                'critical_issues': [str(e)],
                'warnings': []
            }
    
    def _validate_economic_properties(self, regimes: np.ndarray, 
                                    market_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Validate economic properties of regimes."""
        try:
            if market_data is None or not self.config['enable_economic_validation']:
                return {
                    'economic_consistency': 0.5,  # Neutral score
                    'volatility_separation': 0.5,
                    'return_separation': 0.5,
                    'critical_issues': [],
                    'warnings': ['Economic validation disabled or no market data provided']
                }
            
            # Calculate economic metrics
            economic_consistency = self._calculate_economic_consistency(regimes, market_data)
            volatility_separation = self._calculate_volatility_separation(regimes, market_data)
            return_separation = self._calculate_return_separation(regimes, market_data)
            
            warnings = []
            critical_issues = []
            
            if economic_consistency < 0.5:
                warnings.append(f'Economic consistency {economic_consistency:.3f} is low')
            
            if volatility_separation < 0.3:
                warnings.append(f'Volatility separation {volatility_separation:.3f} is low')
            
            if return_separation < 0.3:
                warnings.append(f'Return separation {return_separation:.3f} is low')
            
            return {
                'economic_consistency': economic_consistency,
                'volatility_separation': volatility_separation,
                'return_separation': return_separation,
                'critical_issues': critical_issues,
                'warnings': warnings
            }
            
        except Exception as e:
            self.logger.error(f"Economic validation failed: {e}")
            return {
                'economic_consistency': 0.0,
                'volatility_separation': 0.0,
                'return_separation': 0.0,
                'critical_issues': [str(e)],
                'warnings': []
            }
    
    def _calculate_persistence_score(self, regimes: np.ndarray) -> float:
        """Calculate regime persistence score."""
        if len(regimes) < 2:
            return 0.0
        
        # Calculate transition rate
        transitions = np.sum(regimes[1:] != regimes[:-1])
        transition_rate = transitions / (len(regimes) - 1)
        
        # Persistence score (higher is better, lower transition rate)
        persistence_score = 1.0 - transition_rate
        return max(0.0, min(1.0, persistence_score))
    
    def _calculate_stability_score(self, regimes: np.ndarray, temporal_data: np.ndarray) -> float:
        """Calculate temporal stability score."""
        try:
            if len(regimes) < 10:
                return 0.5  # Neutral score for short sequences
            
            # Sort by temporal data
            sort_indices = np.argsort(temporal_data)
            sorted_regimes = regimes[sort_indices]
            
            # Calculate regime duration statistics
            unique_regimes = np.unique(regimes)
            durations = []
            
            for regime in unique_regimes:
                regime_mask = sorted_regimes == regime
                # Find consecutive runs
                diff = np.diff(np.concatenate(([False], regime_mask, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                if len(starts) > 0:
                    regime_durations = ends - starts
                    durations.extend(regime_durations)
            
            if not durations:
                return 0.0
            
            # Stability score based on duration consistency
            durations = np.array(durations)
            if len(durations) < 2:
                return 0.5
            
            # Coefficient of variation (lower is more stable)
            cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 1.0
            stability_score = 1.0 / (1.0 + cv)
            
            return max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate stability score: {e}")
            return 0.0
    
    def _calculate_transition_rate(self, regimes: np.ndarray) -> float:
        """Calculate regime transition rate."""
        if len(regimes) < 2:
            return 0.0
        
        transitions = np.sum(regimes[1:] != regimes[:-1])
        return transitions / (len(regimes) - 1)
    
    def _calculate_economic_consistency(self, regimes: np.ndarray, 
                                      market_data: pd.DataFrame) -> float:
        """Calculate economic consistency of regimes."""
        try:
            # Group by regime and calculate economic metrics
            regime_groups = market_data.groupby(regimes)
            
            # Calculate consistency of key economic indicators
            consistency_scores = []
            
            for col in ['close', 'volume', 'volatility']:
                if col in market_data.columns:
                    regime_means = regime_groups[col].mean()
                    regime_stds = regime_groups[col].std()
                    
                    # Consistency based on coefficient of variation
                    cv = regime_stds / regime_means
                    consistency = 1.0 / (1.0 + cv.mean())
                    consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate economic consistency: {e}")
            return 0.0
    
    def _calculate_volatility_separation(self, regimes: np.ndarray, 
                                       market_data: pd.DataFrame) -> float:
        """Calculate volatility separation between regimes."""
        try:
            if 'volatility' not in market_data.columns:
                # Calculate volatility if not present
                if 'close' in market_data.columns:
                    returns = market_data['close'].pct_change().dropna()
                    volatility = returns.rolling(window=20).std()
                else:
                    return 0.5  # Neutral score
            
            # Group by regime and calculate volatility statistics
            regime_groups = market_data.groupby(regimes)
            regime_volatilities = regime_groups['volatility'].mean()
            
            if len(regime_volatilities) < 2:
                return 0.0
            
            # Calculate separation as coefficient of variation
            separation = regime_volatilities.std() / regime_volatilities.mean()
            return min(1.0, separation)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate volatility separation: {e}")
            return 0.0
    
    def _calculate_return_separation(self, regimes: np.ndarray, 
                                   market_data: pd.DataFrame) -> float:
        """Calculate return separation between regimes."""
        try:
            if 'close' not in market_data.columns:
                return 0.5  # Neutral score
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Align returns with regimes
            if len(returns) != len(regimes):
                min_len = min(len(returns), len(regimes))
                returns = returns.iloc[:min_len]
                regimes = regimes[:min_len]
            
            # Group by regime and calculate return statistics
            regime_groups = pd.DataFrame({'returns': returns, 'regime': regimes}).groupby('regime')
            regime_returns = regime_groups['returns'].mean()
            
            if len(regime_returns) < 2:
                return 0.0
            
            # Calculate separation as coefficient of variation
            separation = regime_returns.std() / abs(regime_returns.mean())
            return min(1.0, separation)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate return separation: {e}")
            return 0.0
    
    def _combine_metrics(self, basic_metrics: Dict[str, Any], 
                        statistical_metrics: Dict[str, Any],
                        temporal_metrics: Dict[str, Any],
                        economic_metrics: Dict[str, Any]) -> RegimeQualityMetrics:
        """Combine all validation metrics."""
        # Collect all issues and warnings
        all_critical_issues = []
        all_warnings = []
        
        all_critical_issues.extend(basic_metrics.get('critical_issues', []))
        all_critical_issues.extend(statistical_metrics.get('critical_issues', []))
        all_critical_issues.extend(temporal_metrics.get('critical_issues', []))
        all_critical_issues.extend(economic_metrics.get('critical_issues', []))
        
        all_warnings.extend(basic_metrics.get('warnings', []))
        all_warnings.extend(statistical_metrics.get('warnings', []))
        all_warnings.extend(temporal_metrics.get('warnings', []))
        all_warnings.extend(economic_metrics.get('warnings', []))
        
        # Calculate overall quality score
        quality_components = []
        
        # Basic quality (regime balance)
        quality_components.append(basic_metrics.get('regime_balance', 0.0))
        
        # Statistical quality (silhouette score)
        quality_components.append(statistical_metrics.get('silhouette_score', 0.0))
        
        # Temporal quality (persistence and stability)
        temporal_quality = np.mean([
            temporal_metrics.get('persistence_score', 0.0),
            temporal_metrics.get('stability_score', 0.0)
        ])
        quality_components.append(temporal_quality)
        
        # Economic quality
        economic_quality = np.mean([
            economic_metrics.get('economic_consistency', 0.5),
            economic_metrics.get('volatility_separation', 0.5),
            economic_metrics.get('return_separation', 0.5)
        ])
        quality_components.append(economic_quality)
        
        overall_quality = np.mean(quality_components)
        
        return RegimeQualityMetrics(
            n_regimes=basic_metrics.get('n_regimes', 0),
            regime_sizes=basic_metrics.get('regime_sizes', []),
            regime_balance=basic_metrics.get('regime_balance', 0.0),
            silhouette_score=statistical_metrics.get('silhouette_score', 0.0),
            calinski_harabasz_score=statistical_metrics.get('calinski_harabasz_score', 0.0),
            davies_bouldin_score=statistical_metrics.get('davies_bouldin_score', float('inf')),
            persistence_score=temporal_metrics.get('persistence_score', 0.0),
            stability_score=temporal_metrics.get('stability_score', 0.0),
            transition_rate=temporal_metrics.get('transition_rate', 1.0),
            economic_consistency=economic_metrics.get('economic_consistency', 0.0),
            volatility_separation=economic_metrics.get('volatility_separation', 0.0),
            return_separation=economic_metrics.get('return_separation', 0.0),
            overall_quality=overall_quality,
            validation_passed=False,  # Will be set by caller
            critical_issues=all_critical_issues,
            warnings=all_warnings
        )
    
    def _determine_validation_status(self, metrics: RegimeQualityMetrics) -> bool:
        """Determine if validation passes."""
        # Check for critical issues
        if metrics.critical_issues:
            return False
        
        # Check quality thresholds
        if metrics.overall_quality < 0.5:
            return False
        
        if metrics.regime_balance < self.config['min_regime_balance']:
            return False
        
        if metrics.silhouette_score < self.config['min_silhouette_score']:
            return False
        
        if metrics.davies_bouldin_score > self.config['max_davies_bouldin_score']:
            return False
        
        return True
    
    def _create_error_metrics(self, error_message: str) -> RegimeQualityMetrics:
        """Create error metrics when validation fails."""
        return RegimeQualityMetrics(
            n_regimes=0,
            regime_sizes=[],
            regime_balance=0.0,
            silhouette_score=0.0,
            calinski_harabasz_score=0.0,
            davies_bouldin_score=float('inf'),
            persistence_score=0.0,
            stability_score=0.0,
            transition_rate=1.0,
            economic_consistency=0.0,
            volatility_separation=0.0,
            return_separation=0.0,
            overall_quality=0.0,
            validation_passed=False,
            critical_issues=[error_message],
            warnings=[]
        )