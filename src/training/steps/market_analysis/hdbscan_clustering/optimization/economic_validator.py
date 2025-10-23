"""
Economic Validation System for Data-Driven Clustering

This module provides comprehensive economic validation for clustering parameters,
including return separation, volatility profiles, drawdowns, and strategy backtests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import roc_auc_score
from itertools import combinations
import warnings

logger = logging.getLogger(__name__)

@dataclass
class EconomicValidationConfig:
    """Configuration for economic validation."""
    # Return-based metrics
    enable_return_separation: bool = True
    return_horizons: List[int] = None  # [1, 5, 10, 20] for different forward return periods
    min_cluster_size_for_returns: int = 10
    
    # Volatility-based metrics
    enable_volatility_discrimination: bool = True
    volatility_windows: List[int] = None  # [5, 10, 20] for different volatility windows
    min_cluster_size_for_volatility: int = 10
    
    # Risk metrics
    enable_risk_metrics: bool = True
    var_levels: List[float] = None  # [0.01, 0.05, 0.10] for VaR levels
    cvar_levels: List[float] = None  # [0.01, 0.05, 0.10] for CVaR levels
    
    # Drawdown metrics
    enable_drawdown_metrics: bool = True
    drawdown_windows: List[int] = None  # [20, 50, 100] for different drawdown windows
    
    # Volume/liquidity metrics
    enable_volume_metrics: bool = True
    volume_windows: List[int] = None  # [5, 10, 20] for different volume windows
    
    # Strategy backtest metrics
    enable_strategy_backtest: bool = True
    strategy_lookforward: int = 5  # Days to hold position
    transaction_costs: float = 0.001  # Transaction cost as fraction
    
    # Statistical validation
    enable_statistical_tests: bool = True
    significance_level: float = 0.05
    multiple_testing_correction: str = 'bonferroni'  # 'bonferroni', 'fdr', 'none'
    
    # Economic scoring weights
    return_weight: float = 0.3
    volatility_weight: float = 0.2
    risk_weight: float = 0.2
    drawdown_weight: float = 0.15
    volume_weight: float = 0.1
    strategy_weight: float = 0.05

    def __post_init__(self):
        """Set default values after initialization."""
        if self.return_horizons is None:
            self.return_horizons = [1, 5, 10, 20]
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20]
        if self.var_levels is None:
            self.var_levels = [0.01, 0.05, 0.10]
        if self.cvar_levels is None:
            self.cvar_levels = [0.01, 0.05, 0.10]
        if self.drawdown_windows is None:
            self.drawdown_windows = [20, 50, 100]
        if self.volume_windows is None:
            self.volume_windows = [5, 10, 20]

@dataclass
class EconomicValidationResult:
    """Result of economic validation."""
    # Overall scores
    overall_economic_score: float
    return_separation_score: float
    volatility_discrimination_score: float
    risk_discrimination_score: float
    drawdown_discrimination_score: float
    volume_discrimination_score: float
    strategy_performance_score: float
    
    # Detailed metrics
    return_metrics: Dict[str, Any]
    volatility_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    drawdown_metrics: Dict[str, Any]
    volume_metrics: Dict[str, Any]
    strategy_metrics: Dict[str, Any]
    
    # Statistical validation
    statistical_tests: Dict[str, Any]
    
    # Metadata
    validation_config: EconomicValidationConfig
    n_clusters: int
    n_samples: int
    validation_time: float

class EconomicValidator:
    """
    Comprehensive economic validation system for clustering parameters.
    
    Validates clustering quality using financial performance signals including
    return separation, volatility profiles, risk metrics, and strategy backtests.
    """
    
    def __init__(self, config: Optional[EconomicValidationConfig] = None):
        """Initialize economic validator."""
        self.config = config or EconomicValidationConfig()
        
    def validate_clustering(self, 
                          cluster_labels: np.ndarray,
                          market_data: pd.DataFrame,
                          features: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None) -> EconomicValidationResult:
        """
        Validate clustering using comprehensive economic metrics.
        
        Args:
            cluster_labels: Cluster labels to validate
            market_data: Market data with price, volume, etc.
            features: Optional feature matrix
            feature_names: Optional feature names
            
        Returns:
            EconomicValidationResult with comprehensive validation metrics
        """
        try:
            import time
            start_time = time.time()
            
            logger.info("🔍 Starting comprehensive economic validation...")
            
            # Validate input data
            cluster_labels, market_data = self._validate_input(cluster_labels, market_data)
            
            # Calculate return metrics
            return_metrics = self._calculate_return_metrics(cluster_labels, market_data)
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(cluster_labels, market_data)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(cluster_labels, market_data)
            
            # Calculate drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics(cluster_labels, market_data)
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(cluster_labels, market_data)
            
            # Calculate strategy performance
            strategy_metrics = self._calculate_strategy_metrics(cluster_labels, market_data)
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(cluster_labels, market_data)
            
            # Calculate overall scores
            scores = self._calculate_overall_scores(
                return_metrics, volatility_metrics, risk_metrics,
                drawdown_metrics, volume_metrics, strategy_metrics
            )
            
            # Create result
            result = EconomicValidationResult(
                overall_economic_score=scores['overall'],
                return_separation_score=scores['return'],
                volatility_discrimination_score=scores['volatility'],
                risk_discrimination_score=scores['risk'],
                drawdown_discrimination_score=scores['drawdown'],
                volume_discrimination_score=scores['volume'],
                strategy_performance_score=scores['strategy'],
                return_metrics=return_metrics,
                volatility_metrics=volatility_metrics,
                risk_metrics=risk_metrics,
                drawdown_metrics=drawdown_metrics,
                volume_metrics=volume_metrics,
                strategy_metrics=strategy_metrics,
                statistical_tests=statistical_tests,
                validation_config=self.config,
                n_clusters=len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                n_samples=len(cluster_labels),
                validation_time=time.time() - start_time
            )
            
            logger.info(f"✅ Economic validation completed in {result.validation_time:.2f}s")
            logger.info(f"📊 Overall economic score: {result.overall_economic_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Economic validation failed: {e}")
            raise
    
    def _validate_input(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Validate input data."""
        try:
            # Check cluster labels
            if len(cluster_labels) == 0:
                raise ValueError("Empty cluster labels")
            
            # Check market data
            if len(market_data) == 0:
                raise ValueError("Empty market data")
            
            # Ensure same length
            if len(cluster_labels) != len(market_data):
                raise ValueError(f"Length mismatch: labels={len(cluster_labels)}, data={len(market_data)}")
            
            # Check required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove noise points for validation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 10:
                logger.warning("⚠️ Very few valid clusters for validation")
            
            return cluster_labels, market_data
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise
    
    def _calculate_return_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate return-based validation metrics."""
        try:
            if not self.config.enable_return_separation:
                return {'enabled': False}
            
            metrics = {'enabled': True, 'horizons': {}}
            
            # Calculate returns
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                returns = returns.iloc[1:]  # Remove first NaN
                
                # Align with cluster labels
                min_len = min(len(returns), len(cluster_labels))
                returns = returns.iloc[:min_len]
                labels = cluster_labels[:min_len]
                
                # Calculate metrics for each horizon
                for horizon in self.config.return_horizons:
                    if horizon >= len(returns):
                        continue
                    
                    # Calculate forward returns
                    forward_returns = returns.rolling(horizon).apply(
                        lambda x: (1 + x).prod() - 1, raw=False
                    ).dropna()
                    
                    # Align with labels
                    min_len_h = min(len(forward_returns), len(labels))
                    forward_returns = forward_returns.iloc[:min_len_h]
                    labels_h = labels[:min_len_h]
                    
                    # Calculate cluster return statistics
                    cluster_returns = {}
                    unique_labels = np.unique(labels_h)
                    unique_labels = unique_labels[unique_labels != -1]
                    
                    for label in unique_labels:
                        mask = labels_h == label
                        if mask.sum() >= self.config.min_cluster_size_for_returns:
                            cluster_returns[label] = forward_returns[mask].values
                    
                    if len(cluster_returns) < 2:
                        continue
                    
                    # Calculate separation metrics
                    separation_metrics = self._calculate_return_separation(cluster_returns, forward_returns)
                    
                    metrics['horizons'][f'h{horizon}'] = {
                        'cluster_returns': cluster_returns,
                        'separation_metrics': separation_metrics,
                        'n_clusters': len(cluster_returns)
                    }
                
                # Calculate overall return score
                all_separation_scores = []
                for horizon_data in metrics['horizons'].values():
                    if 'separation_metrics' in horizon_data:
                        all_separation_scores.append(horizon_data['separation_metrics']['overall_score'])
                
                metrics['overall_score'] = np.mean(all_separation_scores) if all_separation_scores else 0.0
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Return metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_return_separation(self, cluster_returns: Dict[int, np.ndarray], all_returns: pd.Series) -> Dict[str, float]:
        """Calculate return separation metrics between clusters."""
        try:
            if len(cluster_returns) < 2:
                return {'overall_score': 0.0}
            
            # Calculate pairwise separation
            cluster_pairs = list(combinations(cluster_returns.keys(), 2))
            pairwise_scores = []
            
            for label1, label2 in cluster_pairs:
                returns1 = cluster_returns[label1]
                returns2 = cluster_returns[label2]
                
                # Mean difference
                mean_diff = abs(np.mean(returns1) - np.mean(returns2))
                
                # Volatility-adjusted difference
                pooled_std = np.sqrt((np.var(returns1) + np.var(returns2)) / 2)
                vol_adjusted_diff = mean_diff / (pooled_std + 1e-10)
                
                # Sharpe ratio difference
                sharpe1 = np.mean(returns1) / (np.std(returns1) + 1e-10)
                sharpe2 = np.mean(returns2) / (np.std(returns2) + 1e-10)
                sharpe_diff = abs(sharpe1 - sharpe2)
                
                # Combined score
                pair_score = (vol_adjusted_diff + sharpe_diff) / 2
                pairwise_scores.append(pair_score)
            
            # Calculate overall metrics
            mean_separation = np.mean(pairwise_scores) if pairwise_scores else 0.0
            max_separation = np.max(pairwise_scores) if pairwise_scores else 0.0
            
            # Calculate return variance explained by clusters
            cluster_means = [np.mean(returns) for returns in cluster_returns.values()]
            total_variance = np.var(all_returns)
            between_cluster_variance = np.var(cluster_means)
            variance_explained = between_cluster_variance / (total_variance + 1e-10)
            
            return {
                'mean_separation': mean_separation,
                'max_separation': max_separation,
                'variance_explained': variance_explained,
                'overall_score': (mean_separation + variance_explained) / 2
            }
            
        except Exception as e:
            logger.warning(f"Return separation calculation failed: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_volatility_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based validation metrics."""
        try:
            if not self.config.enable_volatility_discrimination:
                return {'enabled': False}
            
            metrics = {'enabled': True, 'windows': {}}
            
            # Calculate volatility for different windows
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                for window in self.config.volatility_windows:
                    if window >= len(returns):
                        continue
                    
                    # Calculate rolling volatility
                    volatility = returns.rolling(window).std().dropna()
                    
                    # Align with cluster labels
                    min_len = min(len(volatility), len(cluster_labels))
                    volatility = volatility.iloc[:min_len]
                    labels = cluster_labels[:min_len]
                    
                    # Calculate cluster volatility statistics
                    cluster_volatilities = {}
                    unique_labels = np.unique(labels)
                    unique_labels = unique_labels[unique_labels != -1]
                    
                    for label in unique_labels:
                        mask = labels == label
                        if mask.sum() >= self.config.min_cluster_size_for_volatility:
                            cluster_volatilities[label] = volatility[mask].values
                    
                    if len(cluster_volatilities) < 2:
                        continue
                    
                    # Calculate volatility discrimination
                    discrimination_metrics = self._calculate_volatility_discrimination(cluster_volatilities, volatility)
                    
                    metrics['windows'][f'w{window}'] = {
                        'cluster_volatilities': cluster_volatilities,
                        'discrimination_metrics': discrimination_metrics,
                        'n_clusters': len(cluster_volatilities)
                    }
                
                # Calculate overall volatility score
                all_discrimination_scores = []
                for window_data in metrics['windows'].values():
                    if 'discrimination_metrics' in window_data:
                        all_discrimination_scores.append(window_data['discrimination_metrics']['overall_score'])
                
                metrics['overall_score'] = np.mean(all_discrimination_scores) if all_discrimination_scores else 0.0
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Volatility metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_volatility_discrimination(self, cluster_volatilities: Dict[int, np.ndarray], all_volatility: pd.Series) -> Dict[str, float]:
        """Calculate volatility discrimination metrics between clusters."""
        try:
            if len(cluster_volatilities) < 2:
                return {'overall_score': 0.0}
            
            # Calculate pairwise discrimination
            cluster_pairs = list(combinations(cluster_volatilities.keys(), 2))
            pairwise_scores = []
            
            for label1, label2 in cluster_pairs:
                vol1 = cluster_volatilities[label1]
                vol2 = cluster_volatilities[label2]
                
                # Mean difference
                mean_diff = abs(np.mean(vol1) - np.mean(vol2))
                
                # Relative difference
                mean_vol = np.mean(all_volatility)
                relative_diff = mean_diff / (mean_vol + 1e-10)
                
                # Coefficient of variation difference
                cv1 = np.std(vol1) / (np.mean(vol1) + 1e-10)
                cv2 = np.std(vol2) / (np.mean(vol2) + 1e-10)
                cv_diff = abs(cv1 - cv2)
                
                # Combined score
                pair_score = (relative_diff + cv_diff) / 2
                pairwise_scores.append(pair_score)
            
            # Calculate overall metrics
            mean_discrimination = np.mean(pairwise_scores) if pairwise_scores else 0.0
            max_discrimination = np.max(pairwise_scores) if pairwise_scores else 0.0
            
            # Calculate volatility variance explained by clusters
            cluster_means = [np.mean(vol) for vol in cluster_volatilities.values()]
            total_variance = np.var(all_volatility)
            between_cluster_variance = np.var(cluster_means)
            variance_explained = between_cluster_variance / (total_variance + 1e-10)
            
            return {
                'mean_discrimination': mean_discrimination,
                'max_discrimination': max_discrimination,
                'variance_explained': variance_explained,
                'overall_score': (mean_discrimination + variance_explained) / 2
            }
            
        except Exception as e:
            logger.warning(f"Volatility discrimination calculation failed: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_risk_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-based validation metrics."""
        try:
            if not self.config.enable_risk_metrics:
                return {'enabled': False}
            
            metrics = {'enabled': True, 'var_levels': {}, 'cvar_levels': {}}
            
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Align with cluster labels
                min_len = min(len(returns), len(cluster_labels))
                returns = returns.iloc[:min_len]
                labels = cluster_labels[:min_len]
                
                # Calculate VaR for each cluster
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels != -1]
                
                for var_level in self.config.var_levels:
                    cluster_var = {}
                    for label in unique_labels:
                        mask = labels == label
                        if mask.sum() >= 10:  # Minimum samples for VaR
                            cluster_returns = returns[mask].values
                            var_value = np.percentile(cluster_returns, var_level * 100)
                            cluster_var[label] = var_value
                    
                    if len(cluster_var) >= 2:
                        # Calculate VaR discrimination
                        var_values = list(cluster_var.values())
                        var_std = np.std(var_values)
                        var_mean = np.mean(var_values)
                        discrimination = var_std / (abs(var_mean) + 1e-10)
                        
                        metrics['var_levels'][f'var_{var_level}'] = {
                            'cluster_var': cluster_var,
                            'discrimination': discrimination
                        }
                
                # Calculate CVaR for each cluster
                for cvar_level in self.config.cvar_levels:
                    cluster_cvar = {}
                    for label in unique_labels:
                        mask = labels == label
                        if mask.sum() >= 10:  # Minimum samples for CVaR
                            cluster_returns = returns[mask].values
                            var_value = np.percentile(cluster_returns, cvar_level * 100)
                            cvar_value = np.mean(cluster_returns[cluster_returns <= var_value])
                            cluster_cvar[label] = cvar_value
                    
                    if len(cluster_cvar) >= 2:
                        # Calculate CVaR discrimination
                        cvar_values = list(cluster_cvar.values())
                        cvar_std = np.std(cvar_values)
                        cvar_mean = np.mean(cvar_values)
                        discrimination = cvar_std / (abs(cvar_mean) + 1e-10)
                        
                        metrics['cvar_levels'][f'cvar_{cvar_level}'] = {
                            'cluster_cvar': cluster_cvar,
                            'discrimination': discrimination
                        }
                
                # Calculate overall risk score
                all_discriminations = []
                for var_data in metrics['var_levels'].values():
                    all_discriminations.append(var_data['discrimination'])
                for cvar_data in metrics['cvar_levels'].values():
                    all_discriminations.append(cvar_data['discrimination'])
                
                metrics['overall_score'] = np.mean(all_discriminations) if all_discriminations else 0.0
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_drawdown_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drawdown-based validation metrics."""
        try:
            if not self.config.enable_drawdown_metrics:
                return {'enabled': False}
            
            metrics = {'enabled': True, 'windows': {}}
            
            if 'close' in market_data.columns:
                prices = market_data['close']
                
                for window in self.config.drawdown_windows:
                    if window >= len(prices):
                        continue
                    
                    # Calculate rolling drawdowns
                    rolling_max = prices.rolling(window).max()
                    drawdowns = (prices - rolling_max) / rolling_max
                    
                    # Align with cluster labels
                    min_len = min(len(drawdowns), len(cluster_labels))
                    drawdowns = drawdowns.iloc[:min_len]
                    labels = cluster_labels[:min_len]
                    
                    # Calculate cluster drawdown statistics
                    cluster_drawdowns = {}
                    unique_labels = np.unique(labels)
                    unique_labels = unique_labels[unique_labels != -1]
                    
                    for label in unique_labels:
                        mask = labels == label
                        if mask.sum() >= 10:  # Minimum samples for drawdown analysis
                            cluster_drawdowns[label] = drawdowns[mask].values
                    
                    if len(cluster_drawdowns) < 2:
                        continue
                    
                    # Calculate drawdown discrimination
                    discrimination_metrics = self._calculate_drawdown_discrimination(cluster_drawdowns)
                    
                    metrics['windows'][f'w{window}'] = {
                        'cluster_drawdowns': cluster_drawdowns,
                        'discrimination_metrics': discrimination_metrics,
                        'n_clusters': len(cluster_drawdowns)
                    }
                
                # Calculate overall drawdown score
                all_discrimination_scores = []
                for window_data in metrics['windows'].values():
                    if 'discrimination_metrics' in window_data:
                        all_discrimination_scores.append(window_data['discrimination_metrics']['overall_score'])
                
                metrics['overall_score'] = np.mean(all_discrimination_scores) if all_discrimination_scores else 0.0
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Drawdown metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_drawdown_discrimination(self, cluster_drawdowns: Dict[int, np.ndarray]) -> Dict[str, float]:
        """Calculate drawdown discrimination metrics between clusters."""
        try:
            if len(cluster_drawdowns) < 2:
                return {'overall_score': 0.0}
            
            # Calculate pairwise discrimination
            cluster_pairs = list(combinations(cluster_drawdowns.keys(), 2))
            pairwise_scores = []
            
            for label1, label2 in cluster_pairs:
                dd1 = cluster_drawdowns[label1]
                dd2 = cluster_drawdowns[label2]
                
                # Maximum drawdown difference
                max_dd1 = np.min(dd1)
                max_dd2 = np.min(dd2)
                max_dd_diff = abs(max_dd1 - max_dd2)
                
                # Average drawdown difference
                avg_dd1 = np.mean(dd1)
                avg_dd2 = np.mean(dd2)
                avg_dd_diff = abs(avg_dd1 - avg_dd2)
                
                # Drawdown duration difference (simplified)
                dd_duration1 = np.mean(dd1 < -0.01)  # Fraction of time in significant drawdown
                dd_duration2 = np.mean(dd2 < -0.01)
                duration_diff = abs(dd_duration1 - dd_duration2)
                
                # Combined score
                pair_score = (max_dd_diff + avg_dd_diff + duration_diff) / 3
                pairwise_scores.append(pair_score)
            
            # Calculate overall metrics
            mean_discrimination = np.mean(pairwise_scores) if pairwise_scores else 0.0
            max_discrimination = np.max(pairwise_scores) if pairwise_scores else 0.0
            
            return {
                'mean_discrimination': mean_discrimination,
                'max_discrimination': max_discrimination,
                'overall_score': (mean_discrimination + max_discrimination) / 2
            }
            
        except Exception as e:
            logger.warning(f"Drawdown discrimination calculation failed: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_volume_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based validation metrics."""
        try:
            if not self.config.enable_volume_metrics:
                return {'enabled': False}
            
            metrics = {'enabled': True, 'windows': {}}
            
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                
                for window in self.config.volume_windows:
                    if window >= len(volume):
                        continue
                    
                    # Calculate rolling volume statistics
                    volume_mean = volume.rolling(window).mean()
                    volume_std = volume.rolling(window).std()
                    volume_zscore = (volume - volume_mean) / (volume_std + 1e-10)
                    
                    # Align with cluster labels
                    min_len = min(len(volume_zscore), len(cluster_labels))
                    volume_zscore = volume_zscore.iloc[:min_len]
                    labels = cluster_labels[:min_len]
                    
                    # Calculate cluster volume statistics
                    cluster_volumes = {}
                    unique_labels = np.unique(labels)
                    unique_labels = unique_labels[unique_labels != -1]
                    
                    for label in unique_labels:
                        mask = labels == label
                        if mask.sum() >= 10:  # Minimum samples for volume analysis
                            cluster_volumes[label] = volume_zscore[mask].values
                    
                    if len(cluster_volumes) < 2:
                        continue
                    
                    # Calculate volume discrimination
                    discrimination_metrics = self._calculate_volume_discrimination(cluster_volumes, volume_zscore)
                    
                    metrics['windows'][f'w{window}'] = {
                        'cluster_volumes': cluster_volumes,
                        'discrimination_metrics': discrimination_metrics,
                        'n_clusters': len(cluster_volumes)
                    }
                
                # Calculate overall volume score
                all_discrimination_scores = []
                for window_data in metrics['windows'].values():
                    if 'discrimination_metrics' in window_data:
                        all_discrimination_scores.append(window_data['discrimination_metrics']['overall_score'])
                
                metrics['overall_score'] = np.mean(all_discrimination_scores) if all_discrimination_scores else 0.0
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Volume metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_volume_discrimination(self, cluster_volumes: Dict[int, np.ndarray], all_volume: pd.Series) -> Dict[str, float]:
        """Calculate volume discrimination metrics between clusters."""
        try:
            if len(cluster_volumes) < 2:
                return {'overall_score': 0.0}
            
            # Calculate pairwise discrimination
            cluster_pairs = list(combinations(cluster_volumes.keys(), 2))
            pairwise_scores = []
            
            for label1, label2 in cluster_pairs:
                vol1 = cluster_volumes[label1]
                vol2 = cluster_volumes[label2]
                
                # Mean difference
                mean_diff = abs(np.mean(vol1) - np.mean(vol2))
                
                # Relative difference
                mean_vol = np.mean(all_volume)
                relative_diff = mean_diff / (abs(mean_vol) + 1e-10)
                
                # Volatility difference
                vol_std1 = np.std(vol1)
                vol_std2 = np.std(vol2)
                vol_std_diff = abs(vol_std1 - vol_std2)
                
                # Combined score
                pair_score = (relative_diff + vol_std_diff) / 2
                pairwise_scores.append(pair_score)
            
            # Calculate overall metrics
            mean_discrimination = np.mean(pairwise_scores) if pairwise_scores else 0.0
            max_discrimination = np.max(pairwise_scores) if pairwise_scores else 0.0
            
            # Calculate volume variance explained by clusters
            cluster_means = [np.mean(vol) for vol in cluster_volumes.values()]
            total_variance = np.var(all_volume)
            between_cluster_variance = np.var(cluster_means)
            variance_explained = between_cluster_variance / (total_variance + 1e-10)
            
            return {
                'mean_discrimination': mean_discrimination,
                'max_discrimination': max_discrimination,
                'variance_explained': variance_explained,
                'overall_score': (mean_discrimination + variance_explained) / 2
            }
            
        except Exception as e:
            logger.warning(f"Volume discrimination calculation failed: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_strategy_metrics(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate strategy backtest metrics."""
        try:
            if not self.config.enable_strategy_backtest:
                return {'enabled': False}
            
            metrics = {'enabled': True}
            
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Align with cluster labels
                min_len = min(len(returns), len(cluster_labels))
                returns = returns.iloc[:min_len]
                labels = cluster_labels[:min_len]
                
                # Calculate cluster-based strategy returns
                strategy_returns = self._calculate_cluster_strategy_returns(
                    labels, returns, self.config.strategy_lookforward
                )
                
                if len(strategy_returns) > 0:
                    # Calculate strategy performance metrics
                    total_return = np.prod(1 + strategy_returns) - 1
                    sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)
                    max_drawdown = self._calculate_max_drawdown(strategy_returns)
                    
                    metrics.update({
                        'strategy_returns': strategy_returns,
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'overall_score': sharpe_ratio  # Use Sharpe as overall score
                    })
                else:
                    metrics['overall_score'] = 0.0
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Strategy metrics calculation failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_cluster_strategy_returns(self, labels: np.ndarray, returns: pd.Series, lookforward: int) -> np.ndarray:
        """Calculate returns from a simple cluster-based strategy."""
        try:
            strategy_returns = []
            
            for i in range(len(returns) - lookforward):
                current_label = labels[i]
                if current_label == -1:  # Skip noise
                    continue
                
                # Simple strategy: buy if cluster has positive expected return
                cluster_mask = labels == current_label
                cluster_returns = returns[cluster_mask]
                
                if len(cluster_returns) > 5:  # Minimum samples for reliable estimate
                    expected_return = np.mean(cluster_returns)
                    
                    if expected_return > 0:
                        # Buy and hold for lookforward periods
                        future_returns = returns.iloc[i+1:i+1+lookforward]
                        strategy_return = np.prod(1 + future_returns) - 1
                        strategy_returns.append(strategy_return)
            
            return np.array(strategy_returns)
            
        except Exception as e:
            logger.warning(f"Strategy returns calculation failed: {e}")
            return np.array([])
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except Exception as e:
            logger.warning(f"Max drawdown calculation failed: {e}")
            return 0.0
    
    def _perform_statistical_tests(self, cluster_labels: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests for cluster validity."""
        try:
            if not self.config.enable_statistical_tests:
                return {'enabled': False}
            
            tests = {'enabled': True}
            
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Align with cluster labels
                min_len = min(len(returns), len(cluster_labels))
                returns = returns.iloc[:min_len]
                labels = cluster_labels[:min_len]
                
                # Remove noise
                valid_mask = labels != -1
                valid_returns = returns[valid_mask]
                valid_labels = labels[valid_mask]
                
                if len(valid_labels) < 10:
                    return {'enabled': True, 'error': 'Insufficient valid samples'}
                
                # Perform ANOVA test
                try:
                    from scipy.stats import f_oneway
                    unique_labels = np.unique(valid_labels)
                    cluster_return_groups = [valid_returns[valid_labels == label] for label in unique_labels]
                    
                    if len(cluster_return_groups) >= 2:
                        f_stat, p_value = f_oneway(*cluster_return_groups)
                        tests['anova'] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < self.config.significance_level
                        }
                except Exception as e:
                    tests['anova'] = {'error': str(e)}
                
                # Perform Kruskal-Wallis test (non-parametric)
                try:
                    from scipy.stats import kruskal
                    unique_labels = np.unique(valid_labels)
                    cluster_return_groups = [valid_returns[valid_labels == label] for label in unique_labels]
                    
                    if len(cluster_return_groups) >= 2:
                        h_stat, p_value = kruskal(*cluster_return_groups)
                        tests['kruskal_wallis'] = {
                            'h_statistic': h_stat,
                            'p_value': p_value,
                            'significant': p_value < self.config.significance_level
                        }
                except Exception as e:
                    tests['kruskal_wallis'] = {'error': str(e)}
            
            return tests
            
        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_overall_scores(self, 
                                return_metrics: Dict[str, Any],
                                volatility_metrics: Dict[str, Any],
                                risk_metrics: Dict[str, Any],
                                drawdown_metrics: Dict[str, Any],
                                volume_metrics: Dict[str, Any],
                                strategy_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall economic validation scores."""
        try:
            scores = {}
            
            # Return score
            if return_metrics.get('enabled', False) and 'overall_score' in return_metrics:
                scores['return'] = return_metrics['overall_score']
            else:
                scores['return'] = 0.0
            
            # Volatility score
            if volatility_metrics.get('enabled', False) and 'overall_score' in volatility_metrics:
                scores['volatility'] = volatility_metrics['overall_score']
            else:
                scores['volatility'] = 0.0
            
            # Risk score
            if risk_metrics.get('enabled', False) and 'overall_score' in risk_metrics:
                scores['risk'] = risk_metrics['overall_score']
            else:
                scores['risk'] = 0.0
            
            # Drawdown score
            if drawdown_metrics.get('enabled', False) and 'overall_score' in drawdown_metrics:
                scores['drawdown'] = drawdown_metrics['overall_score']
            else:
                scores['drawdown'] = 0.0
            
            # Volume score
            if volume_metrics.get('enabled', False) and 'overall_score' in volume_metrics:
                scores['volume'] = volume_metrics['overall_score']
            else:
                scores['volume'] = 0.0
            
            # Strategy score
            if strategy_metrics.get('enabled', False) and 'overall_score' in strategy_metrics:
                scores['strategy'] = strategy_metrics['overall_score']
            else:
                scores['strategy'] = 0.0
            
            # Overall weighted score
            weights = {
                'return': self.config.return_weight,
                'volatility': self.config.volatility_weight,
                'risk': self.config.risk_weight,
                'drawdown': self.config.drawdown_weight,
                'volume': self.config.volume_weight,
                'strategy': self.config.strategy_weight
            }
            
            overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
            scores['overall'] = overall_score
            
            return scores
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return {'overall': 0.0, 'return': 0.0, 'volatility': 0.0, 'risk': 0.0, 
                   'drawdown': 0.0, 'volume': 0.0, 'strategy': 0.0}