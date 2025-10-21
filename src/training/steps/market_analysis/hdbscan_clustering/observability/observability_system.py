"""
Comprehensive Observability System for Data-Driven Clustering

This module provides comprehensive logging, monitoring, and alerting for:
- Core clustering metrics
- Economic validity metrics
- Stability metrics
- Volume/liquidity metrics
- Infrastructure metrics
"""

import logging
import json
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import clustering metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, jaccard_score
from scipy import stats
from scipy.stats import kruskal, ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class ClusteringMetrics:
    """Core clustering metrics."""
    n_clusters: int
    n_noise: int
    noise_percentage: float
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    label_churn_rate: float
    timestamp: datetime


@dataclass
class EconomicValidityMetrics:
    """Economic validity metrics."""
    return_separation_anova_f: float
    return_separation_anova_p: float
    return_separation_kw_stat: float
    return_separation_kw_p: float
    sharpe_spread: float
    volatility_discrimination: float
    var_deltas: Dict[str, float]
    cvar_deltas: Dict[str, float]
    max_drawdown_per_regime: Dict[str, float]
    overall_economic_score: float
    timestamp: datetime


@dataclass
class StabilityMetrics:
    """Stability and persistence metrics."""
    bootstrap_jaccard: float
    bootstrap_ari: float
    temporal_persistence: float
    median_regime_lifespan: float
    smoothing_reassignments_pct: float
    n_regime_transitions: int
    timestamp: datetime


@dataclass
class VolumeLiquidityMetrics:
    """Volume and liquidity metrics."""
    rvol_separation: float
    vol_price_correlation_by_regime: Dict[str, float]
    volume_ks_stat: float
    volume_ks_p: float
    liquidity_discrimination: float
    timestamp: datetime


@dataclass
class InfrastructureMetrics:
    """Infrastructure and performance metrics."""
    trial_time_distribution: Dict[str, float]
    cache_hit_rate: float
    cpu_utilization: float
    peak_rss_mb: float
    n_retries: int
    n_exceptions: int
    memory_usage_mb: float
    execution_time_s: float
    timestamp: datetime


@dataclass
class Alert:
    """Alert structure."""
    alert_type: str
    severity: str  # 'critical', 'warning', 'info'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    metadata: Dict[str, Any]


class ObservabilitySystem:
    """
    Comprehensive observability system for data-driven clustering.
    
    Tracks and logs:
    - Core clustering metrics
    - Economic validity metrics
    - Stability metrics
    - Volume/liquidity metrics
    - Infrastructure metrics
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 alerts_dir: str = "alerts",
                 enable_slack_alerts: bool = False,
                 slack_webhook_url: Optional[str] = None):
        """
        Initialize the observability system.
        
        Args:
            log_dir: Directory for log files
            alerts_dir: Directory for alert files
            enable_slack_alerts: Whether to enable Slack alerts
            slack_webhook_url: Slack webhook URL for alerts
        """
        self.log_dir = Path(log_dir)
        self.alerts_dir = Path(alerts_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert configuration
        self.enable_slack_alerts = enable_slack_alerts
        self.slack_webhook_url = slack_webhook_url
        
        # Metrics storage
        self.clustering_metrics: deque = deque(maxlen=1000)
        self.economic_metrics: deque = deque(maxlen=1000)
        self.stability_metrics: deque = deque(maxlen=1000)
        self.volume_metrics: deque = deque(maxlen=1000)
        self.infra_metrics: deque = deque(maxlen=1000)
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[callable] = []
        
        # Baseline metrics for comparison
        self.baseline_metrics: Dict[str, Any] = {}
        self.load_baseline_metrics()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up structured logging."""
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for clustering metrics
        clustering_handler = logging.FileHandler(
            self.log_dir / 'clustering_metrics.log'
        )
        clustering_handler.setFormatter(formatter)
        clustering_handler.setLevel(logging.INFO)
        
        # File handler for economic metrics
        economic_handler = logging.FileHandler(
            self.log_dir / 'economic_metrics.log'
        )
        economic_handler.setFormatter(formatter)
        economic_handler.setLevel(logging.INFO)
        
        # File handler for alerts
        alert_handler = logging.FileHandler(
            self.log_dir / 'alerts.log'
        )
        alert_handler.setFormatter(formatter)
        alert_handler.setLevel(logging.WARNING)
        
        # Add handlers to logger
        logger.addHandler(clustering_handler)
        logger.addHandler(economic_handler)
        logger.addHandler(alert_handler)
    
    def log_clustering_metrics(self, 
                              cluster_labels: np.ndarray,
                              features: np.ndarray,
                              previous_labels: Optional[np.ndarray] = None) -> ClusteringMetrics:
        """
        Log core clustering metrics.
        
        Args:
            cluster_labels: Current cluster labels
            features: Feature matrix
            previous_labels: Previous cluster labels for churn calculation
            
        Returns:
            ClusteringMetrics object
        """
        # Calculate basic metrics
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        n_noise = np.sum(cluster_labels == -1)
        noise_percentage = (n_noise / len(cluster_labels)) * 100
        
        # Calculate clustering quality metrics
        if n_clusters > 1 and n_clusters < len(features):
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 1:
                silhouette = silhouette_score(features[valid_mask], cluster_labels[valid_mask])
                dbi = davies_bouldin_score(features[valid_mask], cluster_labels[valid_mask])
                ch = calinski_harabasz_score(features[valid_mask], cluster_labels[valid_mask])
            else:
                silhouette = dbi = ch = 0.0
        else:
            silhouette = dbi = ch = 0.0
        
        # Calculate label churn rate
        if previous_labels is not None and len(previous_labels) == len(cluster_labels):
            churn_rate = np.mean(cluster_labels != previous_labels) * 100
        else:
            churn_rate = 0.0
        
        # Create metrics object
        metrics = ClusteringMetrics(
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_percentage=noise_percentage,
            silhouette_score=silhouette,
            davies_bouldin_score=dbi,
            calinski_harabasz_score=ch,
            label_churn_rate=churn_rate,
            timestamp=datetime.now()
        )
        
        # Store and log
        self.clustering_metrics.append(metrics)
        
        # Log structured data
        logger.info(f"CLUSTERING_METRICS: {json.dumps(asdict(metrics), default=str)}")
        
        # Check for alerts
        self._check_clustering_alerts(metrics)
        
        return metrics
    
    def log_economic_validity_metrics(self, 
                                    cluster_labels: np.ndarray,
                                    market_data: pd.DataFrame,
                                    features: np.ndarray) -> EconomicValidityMetrics:
        """
        Log economic validity metrics.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data
            features: Feature matrix
            
        Returns:
            EconomicValidityMetrics object
        """
        # Calculate return separation
        returns = market_data['returns'].dropna()
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        valid_returns = returns.iloc[valid_mask]
        
        if len(np.unique(valid_labels)) > 1 and len(valid_returns) > 0:
            # ANOVA test
            groups = [valid_returns[valid_labels == label].values 
                     for label in np.unique(valid_labels) if len(valid_returns[valid_labels == label]) > 0]
            
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                anova_f, anova_p = stats.f_oneway(*groups)
                kw_stat, kw_p = kruskal(*groups)
            else:
                anova_f = anova_p = kw_stat = kw_p = 0.0
        else:
            anova_f = anova_p = kw_stat = kw_p = 0.0
        
        # Calculate Sharpe spread across regimes
        sharpe_scores = []
        for label in np.unique(valid_labels):
            if label != -1:
                regime_returns = valid_returns[valid_labels == label]
                if len(regime_returns) > 1:
                    sharpe = np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0
                    sharpe_scores.append(sharpe)
        
        sharpe_spread = np.std(sharpe_scores) if len(sharpe_scores) > 1 else 0.0
        
        # Calculate volatility discrimination
        volatility = market_data['volatility'].dropna()
        if len(volatility) > 0 and len(valid_labels) > 0:
            vol_groups = [volatility.iloc[valid_mask][valid_labels == label].values 
                         for label in np.unique(valid_labels) if label != -1]
            if len(vol_groups) > 1 and all(len(g) > 0 for g in vol_groups):
                vol_f, vol_p = stats.f_oneway(*vol_groups)
                volatility_discrimination = vol_f if not np.isnan(vol_f) else 0.0
            else:
                volatility_discrimination = 0.0
        else:
            volatility_discrimination = 0.0
        
        # Calculate VaR/CVaR deltas
        var_deltas = {}
        cvar_deltas = {}
        for label in np.unique(valid_labels):
            if label != -1:
                regime_returns = valid_returns[valid_labels == label]
                if len(regime_returns) > 0:
                    var_95 = np.percentile(regime_returns, 5)
                    cvar_95 = np.mean(regime_returns[regime_returns <= var_95])
                    var_deltas[f'regime_{label}'] = var_95
                    cvar_deltas[f'regime_{label}'] = cvar_95
        
        # Calculate max drawdown per regime
        max_drawdown_per_regime = {}
        for label in np.unique(valid_labels):
            if label != -1:
                regime_returns = valid_returns[valid_labels == label]
                if len(regime_returns) > 0:
                    cumulative = np.cumprod(1 + regime_returns)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown_per_regime[f'regime_{label}'] = np.min(drawdown)
        
        # Calculate overall economic score
        overall_economic_score = self._calculate_economic_score(
            anova_f, sharpe_spread, volatility_discrimination, var_deltas, max_drawdown_per_regime
        )
        
        # Create metrics object
        metrics = EconomicValidityMetrics(
            return_separation_anova_f=anova_f,
            return_separation_anova_p=anova_p,
            return_separation_kw_stat=kw_stat,
            return_separation_kw_p=kw_p,
            sharpe_spread=sharpe_spread,
            volatility_discrimination=volatility_discrimination,
            var_deltas=var_deltas,
            cvar_deltas=cvar_deltas,
            max_drawdown_per_regime=max_drawdown_per_regime,
            overall_economic_score=overall_economic_score,
            timestamp=datetime.now()
        )
        
        # Store and log
        self.economic_metrics.append(metrics)
        
        # Log structured data
        logger.info(f"ECONOMIC_METRICS: {json.dumps(asdict(metrics), default=str)}")
        
        # Check for alerts
        self._check_economic_alerts(metrics)
        
        return metrics
    
    def log_stability_metrics(self, 
                            cluster_labels: np.ndarray,
                            previous_labels: Optional[np.ndarray] = None,
                            bootstrap_labels: Optional[List[np.ndarray]] = None) -> StabilityMetrics:
        """
        Log stability and persistence metrics.
        
        Args:
            cluster_labels: Current cluster labels
            previous_labels: Previous cluster labels
            bootstrap_labels: Bootstrap cluster labels for stability
            
        Returns:
            StabilityMetrics object
        """
        # Calculate bootstrap stability
        if bootstrap_labels and len(bootstrap_labels) > 1:
            # Calculate Jaccard similarity between bootstrap samples
            jaccard_scores = []
            ari_scores = []
            
            for i in range(len(bootstrap_labels)):
                for j in range(i + 1, len(bootstrap_labels)):
                    if len(bootstrap_labels[i]) == len(bootstrap_labels[j]):
                        jaccard = jaccard_score(bootstrap_labels[i], bootstrap_labels[j], average='macro')
                        ari = adjusted_rand_score(bootstrap_labels[i], bootstrap_labels[j])
                        jaccard_scores.append(jaccard)
                        ari_scores.append(ari)
            
            bootstrap_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
            bootstrap_ari = np.mean(ari_scores) if ari_scores else 0.0
        else:
            bootstrap_jaccard = bootstrap_ari = 0.0
        
        # Calculate temporal persistence
        if len(cluster_labels) > 1:
            # Calculate regime transitions
            transitions = np.sum(cluster_labels[1:] != cluster_labels[:-1])
            temporal_persistence = 1 - (transitions / (len(cluster_labels) - 1))
        else:
            temporal_persistence = 1.0
            transitions = 0
        
        # Calculate median regime lifespan
        if len(cluster_labels) > 0:
            regime_lifespans = []
            current_regime = cluster_labels[0]
            current_length = 1
            
            for i in range(1, len(cluster_labels)):
                if cluster_labels[i] == current_regime:
                    current_length += 1
                else:
                    regime_lifespans.append(current_length)
                    current_regime = cluster_labels[i]
                    current_length = 1
            
            regime_lifespans.append(current_length)
            median_regime_lifespan = np.median(regime_lifespans) if regime_lifespans else 0.0
        else:
            median_regime_lifespan = 0.0
        
        # Calculate smoothing reassignments percentage
        if previous_labels is not None and len(previous_labels) == len(cluster_labels):
            smoothing_reassignments = np.sum(cluster_labels != previous_labels)
            smoothing_reassignments_pct = (smoothing_reassignments / len(cluster_labels)) * 100
        else:
            smoothing_reassignments_pct = 0.0
        
        # Create metrics object
        metrics = StabilityMetrics(
            bootstrap_jaccard=bootstrap_jaccard,
            bootstrap_ari=bootstrap_ari,
            temporal_persistence=temporal_persistence,
            median_regime_lifespan=median_regime_lifespan,
            smoothing_reassignments_pct=smoothing_reassignments_pct,
            n_regime_transitions=transitions,
            timestamp=datetime.now()
        )
        
        # Store and log
        self.stability_metrics.append(metrics)
        
        # Log structured data
        logger.info(f"STABILITY_METRICS: {json.dumps(asdict(metrics), default=str)}")
        
        return metrics
    
    def log_volume_liquidity_metrics(self, 
                                   cluster_labels: np.ndarray,
                                   market_data: pd.DataFrame) -> VolumeLiquidityMetrics:
        """
        Log volume and liquidity metrics.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume information
            
        Returns:
            VolumeLiquidityMetrics object
        """
        # Calculate RVOL separation
        if 'volume' in market_data.columns and 'volume_ma' in market_data.columns:
            rvol = market_data['volume'] / market_data['volume_ma']
            valid_mask = cluster_labels != -1
            valid_labels = cluster_labels[valid_mask]
            valid_rvol = rvol.iloc[valid_mask]
            
            if len(np.unique(valid_labels)) > 1 and len(valid_rvol) > 0:
                rvol_groups = [valid_rvol[valid_labels == label].values 
                              for label in np.unique(valid_labels) if label != -1]
                if len(rvol_groups) > 1 and all(len(g) > 0 for g in rvol_groups):
                    rvol_f, rvol_p = stats.f_oneway(*rvol_groups)
                    rvol_separation = rvol_f if not np.isnan(rvol_f) else 0.0
                else:
                    rvol_separation = 0.0
            else:
                rvol_separation = 0.0
        else:
            rvol_separation = 0.0
        
        # Calculate volume-price correlation by regime
        vol_price_correlation_by_regime = {}
        if 'volume' in market_data.columns and 'returns' in market_data.columns:
            returns = market_data['returns'].dropna()
            volume = market_data['volume']
            
            for label in np.unique(cluster_labels):
                if label != -1:
                    regime_mask = cluster_labels == label
                    if np.sum(regime_mask) > 1:
                        regime_returns = returns.iloc[regime_mask]
                        regime_volume = volume.iloc[regime_mask]
                        if len(regime_returns) > 1 and len(regime_volume) > 1:
                            correlation = np.corrcoef(regime_returns, regime_volume)[0, 1]
                            vol_price_correlation_by_regime[f'regime_{label}'] = correlation if not np.isnan(correlation) else 0.0
                        else:
                            vol_price_correlation_by_regime[f'regime_{label}'] = 0.0
                    else:
                        vol_price_correlation_by_regime[f'regime_{label}'] = 0.0
        
        # Calculate volume KS test
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            valid_mask = cluster_labels != -1
            valid_labels = cluster_labels[valid_mask]
            valid_volume = volume.iloc[valid_mask]
            
            if len(np.unique(valid_labels)) > 1 and len(valid_volume) > 0:
                volume_groups = [valid_volume[valid_labels == label].values 
                               for label in np.unique(valid_labels) if label != -1]
                if len(volume_groups) > 1 and all(len(g) > 0 for g in volume_groups):
                    ks_stat, ks_p = ks_2samp(volume_groups[0], volume_groups[1])
                    volume_ks_stat = ks_stat if not np.isnan(ks_stat) else 0.0
                    volume_ks_p = ks_p if not np.isnan(ks_p) else 1.0
                else:
                    volume_ks_stat = volume_ks_p = 0.0
            else:
                volume_ks_stat = volume_ks_p = 0.0
        else:
            volume_ks_stat = volume_ks_p = 0.0
        
        # Calculate liquidity discrimination
        liquidity_discrimination = rvol_separation * (1 - volume_ks_p) if volume_ks_p > 0 else rvol_separation
        
        # Create metrics object
        metrics = VolumeLiquidityMetrics(
            rvol_separation=rvol_separation,
            vol_price_correlation_by_regime=vol_price_correlation_by_regime,
            volume_ks_stat=volume_ks_stat,
            volume_ks_p=volume_ks_p,
            liquidity_discrimination=liquidity_discrimination,
            timestamp=datetime.now()
        )
        
        # Store and log
        self.volume_metrics.append(metrics)
        
        # Log structured data
        logger.info(f"VOLUME_LIQUIDITY_METRICS: {json.dumps(asdict(metrics), default=str)}")
        
        return metrics
    
    def log_infrastructure_metrics(self, 
                                 trial_times: List[float],
                                 cache_hit_rate: float,
                                 n_retries: int = 0,
                                 n_exceptions: int = 0) -> InfrastructureMetrics:
        """
        Log infrastructure and performance metrics.
        
        Args:
            trial_times: List of trial execution times
            cache_hit_rate: Cache hit rate
            n_retries: Number of retries
            n_exceptions: Number of exceptions
            
        Returns:
            InfrastructureMetrics object
        """
        # Calculate trial time distribution
        if trial_times:
            trial_time_distribution = {
                'mean': np.mean(trial_times),
                'std': np.std(trial_times),
                'min': np.min(trial_times),
                'max': np.max(trial_times),
                'median': np.median(trial_times),
                'p95': np.percentile(trial_times, 95),
                'p99': np.percentile(trial_times, 99)
            }
        else:
            trial_time_distribution = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'p95': 0.0, 'p99': 0.0
            }
        
        # Get current system metrics
        cpu_utilization = psutil.cpu_percent()
        memory_info = psutil.Process().memory_info()
        peak_rss_mb = memory_info.rss / 1024 / 1024
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # Calculate execution time
        execution_time_s = np.sum(trial_times) if trial_times else 0.0
        
        # Create metrics object
        metrics = InfrastructureMetrics(
            trial_time_distribution=trial_time_distribution,
            cache_hit_rate=cache_hit_rate,
            cpu_utilization=cpu_utilization,
            peak_rss_mb=peak_rss_mb,
            n_retries=n_retries,
            n_exceptions=n_exceptions,
            memory_usage_mb=memory_usage_mb,
            execution_time_s=execution_time_s,
            timestamp=datetime.now()
        )
        
        # Store and log
        self.infra_metrics.append(metrics)
        
        # Log structured data
        logger.info(f"INFRASTRUCTURE_METRICS: {json.dumps(asdict(metrics), default=str)}")
        
        # Check for alerts
        self._check_infrastructure_alerts(metrics)
        
        return metrics
    
    def _calculate_economic_score(self, 
                                anova_f: float, 
                                sharpe_spread: float, 
                                volatility_discrimination: float,
                                var_deltas: Dict[str, float],
                                max_drawdown_per_regime: Dict[str, float]) -> float:
        """Calculate overall economic score."""
        # Normalize metrics to 0-1 scale
        anova_score = min(anova_f / 10.0, 1.0)  # Cap at 1.0
        sharpe_score = min(sharpe_spread / 2.0, 1.0)  # Cap at 1.0
        vol_score = min(volatility_discrimination / 10.0, 1.0)  # Cap at 1.0
        
        # Calculate VaR diversity score
        if var_deltas:
            var_values = list(var_deltas.values())
            var_diversity = np.std(var_values) / (np.mean(np.abs(var_values)) + 1e-8)
            var_score = min(var_diversity, 1.0)
        else:
            var_score = 0.0
        
        # Calculate drawdown diversity score
        if max_drawdown_per_regime:
            dd_values = list(max_drawdown_per_regime.values())
            dd_diversity = np.std(dd_values) / (np.mean(np.abs(dd_values)) + 1e-8)
            dd_score = min(dd_diversity, 1.0)
        else:
            dd_score = 0.0
        
        # Weighted combination
        economic_score = (
            0.3 * anova_score +
            0.25 * sharpe_score +
            0.25 * vol_score +
            0.1 * var_score +
            0.1 * dd_score
        )
        
        return economic_score
    
    def _check_clustering_alerts(self, metrics: ClusteringMetrics):
        """Check for clustering-related alerts."""
        # Noise percentage alert
        if metrics.noise_percentage > 40.0:
            self._create_alert(
                alert_type='high_noise_percentage',
                severity='critical',
                message=f"Noise percentage {metrics.noise_percentage:.1f}% exceeds threshold 40%",
                metric_name='noise_percentage',
                current_value=metrics.noise_percentage,
                threshold_value=40.0,
                metadata={'n_clusters': metrics.n_clusters, 'n_noise': metrics.n_noise}
            )
        
        # Label churn alert
        if metrics.label_churn_rate > 15.0:
            self._create_alert(
                alert_type='high_label_churn',
                severity='warning',
                message=f"Label churn rate {metrics.label_churn_rate:.1f}% exceeds threshold 15%",
                metric_name='label_churn_rate',
                current_value=metrics.label_churn_rate,
                threshold_value=15.0,
                metadata={'n_clusters': metrics.n_clusters}
            )
    
    def _check_economic_alerts(self, metrics: EconomicValidityMetrics):
        """Check for economic validity alerts."""
        # Economic score decrease alert
        if 'economic_score' in self.baseline_metrics:
            baseline_score = self.baseline_metrics['economic_score']
            if baseline_score > 0:
                score_decrease = (baseline_score - metrics.overall_economic_score) / baseline_score
                if score_decrease > 0.1:  # 10% decrease
                    self._create_alert(
                        alert_type='economic_score_decrease',
                        severity='critical',
                        message=f"Economic score decreased by {score_decrease:.1%} from baseline",
                        metric_name='overall_economic_score',
                        current_value=metrics.overall_economic_score,
                        threshold_value=baseline_score * 0.9,
                        metadata={'baseline_score': baseline_score, 'decrease_pct': score_decrease}
                    )
    
    def _check_infrastructure_alerts(self, metrics: InfrastructureMetrics):
        """Check for infrastructure alerts."""
        # CPU utilization alert
        if metrics.cpu_utilization > 80.0:
            self._create_alert(
                alert_type='high_cpu_utilization',
                severity='warning',
                message=f"CPU utilization {metrics.cpu_utilization:.1f}% exceeds threshold 80%",
                metric_name='cpu_utilization',
                current_value=metrics.cpu_utilization,
                threshold_value=80.0,
                metadata={'peak_rss_mb': metrics.peak_rss_mb}
            )
        
        # Memory usage alert
        if metrics.peak_rss_mb > 2000.0:
            self._create_alert(
                alert_type='high_memory_usage',
                severity='warning',
                message=f"Peak memory usage {metrics.peak_rss_mb:.1f}MB exceeds threshold 2000MB",
                metric_name='peak_rss_mb',
                current_value=metrics.peak_rss_mb,
                threshold_value=2000.0,
                metadata={'cpu_utilization': metrics.cpu_utilization}
            )
    
    def _create_alert(self, 
                     alert_type: str, 
                     severity: str, 
                     message: str, 
                     metric_name: str, 
                     current_value: float, 
                     threshold_value: float, 
                     metadata: Dict[str, Any] = None):
        """Create and process alert."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"ALERT: {json.dumps(asdict(alert), default=str)}")
        
        # Save alert to file
        self._save_alert(alert)
        
        # Send Slack alert if enabled
        if self.enable_slack_alerts and self.slack_webhook_url:
            self._send_slack_alert(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _save_alert(self, alert: Alert):
        """Save alert to file."""
        timestamp = alert.timestamp.strftime('%Y%m%d_%H%M%S')
        alert_file = self.alerts_dir / f"alert_{timestamp}_{alert.alert_type}.json"
        
        with open(alert_file, 'w') as f:
            json.dump(asdict(alert), f, indent=2, default=str)
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack."""
        try:
            import requests
            
            # Format message
            color = {'critical': 'danger', 'warning': 'warning', 'info': 'good'}[alert.severity]
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Clustering Alert: {alert.alert_type}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                            {"title": "Severity", "value": alert.severity, "short": True}
                        ],
                        "timestamp": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def load_baseline_metrics(self, baseline_file: str = "baseline_metrics.json"):
        """Load baseline metrics for comparison."""
        baseline_path = self.log_dir / baseline_file
        
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                self.baseline_metrics = json.load(f)
            logger.info(f"Loaded baseline metrics from {baseline_path}")
        else:
            logger.warning(f"Baseline file {baseline_path} not found")
            self.baseline_metrics = {}
    
    def save_baseline_metrics(self, baseline_file: str = "baseline_metrics.json"):
        """Save current metrics as baseline."""
        if self.economic_metrics:
            latest_economic = self.economic_metrics[-1]
            self.baseline_metrics['economic_score'] = latest_economic.overall_economic_score
            self.baseline_metrics['timestamp'] = latest_economic.timestamp.isoformat()
        
        baseline_path = self.log_dir / baseline_file
        with open(baseline_path, 'w') as f:
            json.dump(self.baseline_metrics, f, indent=2, default=str)
        
        logger.info(f"Saved baseline metrics to {baseline_path}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            'timestamp': datetime.now(),
            'clustering_metrics': {
                'latest': asdict(self.clustering_metrics[-1]) if self.clustering_metrics else None,
                'count': len(self.clustering_metrics),
                'avg_noise_percentage': np.mean([m.noise_percentage for m in self.clustering_metrics]) if self.clustering_metrics else 0,
                'avg_silhouette': np.mean([m.silhouette_score for m in self.clustering_metrics]) if self.clustering_metrics else 0,
                'avg_churn_rate': np.mean([m.label_churn_rate for m in self.clustering_metrics]) if self.clustering_metrics else 0
            },
            'economic_metrics': {
                'latest': asdict(self.economic_metrics[-1]) if self.economic_metrics else None,
                'count': len(self.economic_metrics),
                'avg_economic_score': np.mean([m.overall_economic_score for m in self.economic_metrics]) if self.economic_metrics else 0
            },
            'stability_metrics': {
                'latest': asdict(self.stability_metrics[-1]) if self.stability_metrics else None,
                'count': len(self.stability_metrics),
                'avg_bootstrap_jaccard': np.mean([m.bootstrap_jaccard for m in self.stability_metrics]) if self.stability_metrics else 0
            },
            'volume_metrics': {
                'latest': asdict(self.volume_metrics[-1]) if self.volume_metrics else None,
                'count': len(self.volume_metrics),
                'avg_rvol_separation': np.mean([m.rvol_separation for m in self.volume_metrics]) if self.volume_metrics else 0
            },
            'infrastructure_metrics': {
                'latest': asdict(self.infra_metrics[-1]) if self.infra_metrics else None,
                'count': len(self.infra_metrics),
                'avg_cpu_utilization': np.mean([m.cpu_utilization for m in self.infra_metrics]) if self.infra_metrics else 0,
                'avg_memory_usage': np.mean([m.peak_rss_mb for m in self.infra_metrics]) if self.infra_metrics else 0
            },
            'alerts': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts if a.severity == 'critical']),
                'warning': len([a for a in self.alerts if a.severity == 'warning']),
                'recent': [asdict(a) for a in self.alerts[-10:]] if self.alerts else []
            },
            'baseline_metrics': self.baseline_metrics
        }
        
        return summary


class PerformanceTracker:
    """Performance tracking utility."""
    
    def __init__(self):
        self.start_times = {}
        self.trial_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def start_trial(self, trial_id: str):
        """Start timing a trial."""
        self.start_times[trial_id] = time.perf_counter()
    
    def end_trial(self, trial_id: str):
        """End timing a trial."""
        if trial_id in self.start_times:
            trial_time = time.perf_counter() - self.start_times[trial_id]
            self.trial_times.append(trial_time)
            del self.start_times[trial_id]
            return trial_time
        return 0.0
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def get_trial_times(self) -> List[float]:
        """Get trial times."""
        return self.trial_times.copy()
    
    def reset(self):
        """Reset tracker."""
        self.start_times.clear()
        self.trial_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# Global observability system instance
_observability_system = None


def get_observability_system() -> ObservabilitySystem:
    """Get global observability system instance."""
    global _observability_system
    if _observability_system is None:
        _observability_system = ObservabilitySystem()
    return _observability_system


def log_clustering_metrics(cluster_labels: np.ndarray, 
                          features: np.ndarray, 
                          previous_labels: Optional[np.ndarray] = None) -> ClusteringMetrics:
    """Log clustering metrics using global system."""
    return get_observability_system().log_clustering_metrics(cluster_labels, features, previous_labels)


def log_economic_validity_metrics(cluster_labels: np.ndarray, 
                                 market_data: pd.DataFrame, 
                                 features: np.ndarray) -> EconomicValidityMetrics:
    """Log economic validity metrics using global system."""
    return get_observability_system().log_economic_validity_metrics(cluster_labels, market_data, features)


def log_stability_metrics(cluster_labels: np.ndarray, 
                         previous_labels: Optional[np.ndarray] = None,
                         bootstrap_labels: Optional[List[np.ndarray]] = None) -> StabilityMetrics:
    """Log stability metrics using global system."""
    return get_observability_system().log_stability_metrics(cluster_labels, previous_labels, bootstrap_labels)


def log_volume_liquidity_metrics(cluster_labels: np.ndarray, 
                                market_data: pd.DataFrame) -> VolumeLiquidityMetrics:
    """Log volume/liquidity metrics using global system."""
    return get_observability_system().log_volume_liquidity_metrics(cluster_labels, market_data)


def log_infrastructure_metrics(trial_times: List[float], 
                              cache_hit_rate: float, 
                              n_retries: int = 0, 
                              n_exceptions: int = 0) -> InfrastructureMetrics:
    """Log infrastructure metrics using global system."""
    return get_observability_system().log_infrastructure_metrics(trial_times, cache_hit_rate, n_retries, n_exceptions)


if __name__ == "__main__":
    # Example usage
    print("Observability system example")
    
    # Create observability system
    obs = ObservabilitySystem()
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    cluster_labels = np.random.randint(0, 3, n_samples)
    features = np.random.randn(n_samples, 50)
    
    market_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
        'volume': np.random.lognormal(5, 0.5, n_samples),
        'volume_ma': np.random.lognormal(5, 0.3, n_samples)
    })
    
    # Log metrics
    clustering_metrics = obs.log_clustering_metrics(cluster_labels, features)
    economic_metrics = obs.log_economic_validity_metrics(cluster_labels, market_data, features)
    stability_metrics = obs.log_stability_metrics(cluster_labels)
    volume_metrics = obs.log_volume_liquidity_metrics(cluster_labels, market_data)
    infra_metrics = obs.log_infrastructure_metrics([1.0, 2.0, 1.5], 0.8)
    
    # Get summary
    summary = obs.get_metrics_summary()
    print(f"Clustering metrics: {summary['clustering_metrics']['count']}")
    print(f"Economic metrics: {summary['economic_metrics']['count']}")
    print(f"Alerts: {summary['alerts']['total']}")