"""
Canary and Rollback System

This module implements a canary deployment system for clustering models that:
- Shadow-runs new clustering for a week
- Compares label churn on overlapping windows
- Promotes only if churn and economic score are inside bands
- Provides automated rollback triggers
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import clustering components
from sklearn.metrics import adjusted_rand_score, jaccard_score
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import umap

logger = logging.getLogger(__name__)


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    shadow_run_days: int = 7  # Days to shadow run
    overlap_window_days: int = 3  # Days for overlapping comparison
    max_label_churn_pct: float = 15.0  # Maximum allowed label churn percentage
    min_economic_score: float = 0.7  # Minimum economic score threshold
    max_economic_score_drop: float = 0.1  # Maximum economic score drop (10%)
    min_silhouette_score: float = 0.2  # Minimum silhouette score
    min_cluster_stability: float = 0.8  # Minimum cluster stability
    enable_auto_rollback: bool = True  # Whether to enable automatic rollback
    rollback_threshold: float = 0.8  # Rollback threshold (80% of metrics fail)
    notification_webhook: Optional[str] = None  # Webhook for notifications


@dataclass
class CanaryMetrics:
    """Metrics for canary evaluation."""
    timestamp: datetime
    label_churn_pct: float
    economic_score: float
    silhouette_score: float
    cluster_stability: float
    n_clusters: int
    n_noise: int
    data_quality_score: float
    performance_score: float


@dataclass
class CanaryResult:
    """Result of canary evaluation."""
    canary_id: str
    start_time: datetime
    end_time: datetime
    status: str  # 'running', 'passed', 'failed', 'rolled_back'
    metrics_history: List[CanaryMetrics]
    final_decision: str
    rollback_reason: Optional[str]
    promotion_confidence: float
    timestamp: datetime


@dataclass
class RollbackTrigger:
    """Rollback trigger definition."""
    trigger_type: str
    threshold: float
    current_value: float
    triggered: bool
    severity: str  # 'critical', 'warning', 'info'
    message: str
    timestamp: datetime


class CanaryRollbackSystem:
    """
    Canary deployment and rollback system for clustering models.
    
    Features:
    - Shadow-run new clustering for specified period
    - Compare label churn on overlapping windows
    - Economic score validation
    - Cluster stability monitoring
    - Automated rollback triggers
    - Promotion confidence scoring
    """
    
    def __init__(self, config: CanaryConfig = None):
        """
        Initialize canary rollback system.
        
        Args:
            config: Configuration object
        """
        self.config = config or CanaryConfig()
        
        # Canary tracking
        self.active_canaries: Dict[str, CanaryResult] = {}
        self.completed_canaries: List[CanaryResult] = []
        self.rollback_triggers: List[RollbackTrigger] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_canaries': 0,
            'successful_promotions': 0,
            'failed_canaries': 0,
            'rollbacks': 0,
            'avg_evaluation_time': 0.0
        }
        
        # Baseline metrics for comparison
        self.baseline_metrics: Dict[str, Any] = {}
        
    def start_canary(self, 
                    canary_id: str,
                    clustering_func: Callable,
                    market_data: pd.DataFrame,
                    features: np.ndarray,
                    feature_names: List[str] = None) -> CanaryResult:
        """
        Start a new canary deployment.
        
        Args:
            canary_id: Unique identifier for the canary
            clustering_func: Function that performs clustering
            market_data: Market data
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            CanaryResult
        """
        logger.info(f"Starting canary deployment: {canary_id}")
        
        # Create canary result
        canary_result = CanaryResult(
            canary_id=canary_id,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=self.config.shadow_run_days),
            status='running',
            metrics_history=[],
            final_decision='pending',
            rollback_reason=None,
            promotion_confidence=0.0,
            timestamp=datetime.now()
        )
        
        # Store canary
        self.active_canaries[canary_id] = canary_result
        
        # Update performance metrics
        self.performance_metrics['total_canaries'] += 1
        
        logger.info(f"Canary {canary_id} started, will run until {canary_result.end_time}")
        
        return canary_result
    
    def evaluate_canary_metrics(self, 
                              canary_id: str,
                              cluster_labels: np.ndarray,
                              market_data: pd.DataFrame,
                              features: np.ndarray,
                              previous_labels: Optional[np.ndarray] = None) -> CanaryMetrics:
        """
        Evaluate metrics for a canary deployment.
        
        Args:
            canary_id: Canary identifier
            cluster_labels: Current cluster labels
            market_data: Market data
            features: Feature matrix
            previous_labels: Previous cluster labels for churn calculation
            
        Returns:
            CanaryMetrics
        """
        # Calculate label churn percentage
        if previous_labels is not None and len(previous_labels) == len(cluster_labels):
            label_churn_pct = np.mean(cluster_labels != previous_labels) * 100
        else:
            label_churn_pct = 0.0
        
        # Calculate economic score
        economic_score = self._calculate_economic_score(cluster_labels, market_data, features)
        
        # Calculate silhouette score
        silhouette_score = self._calculate_silhouette_score(cluster_labels, features)
        
        # Calculate cluster stability
        cluster_stability = self._calculate_cluster_stability(cluster_labels, previous_labels)
        
        # Count clusters and noise
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        n_noise = np.sum(cluster_labels == -1)
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(cluster_labels, market_data)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(cluster_labels, features, market_data)
        
        # Create metrics
        metrics = CanaryMetrics(
            timestamp=datetime.now(),
            label_churn_pct=label_churn_pct,
            economic_score=economic_score,
            silhouette_score=silhouette_score,
            cluster_stability=cluster_stability,
            n_clusters=n_clusters,
            n_noise=n_noise,
            data_quality_score=data_quality_score,
            performance_score=performance_score
        )
        
        # Store metrics
        if canary_id in self.active_canaries:
            self.active_canaries[canary_id].metrics_history.append(metrics)
        
        return metrics
    
    def check_rollback_triggers(self, canary_id: str, metrics: CanaryMetrics) -> List[RollbackTrigger]:
        """
        Check for rollback triggers based on current metrics.
        
        Args:
            canary_id: Canary identifier
            metrics: Current metrics
            
        Returns:
            List of triggered rollback conditions
        """
        triggers = []
        
        # Check label churn threshold
        if metrics.label_churn_pct > self.config.max_label_churn_pct:
            trigger = RollbackTrigger(
                trigger_type='high_label_churn',
                threshold=self.config.max_label_churn_pct,
                current_value=metrics.label_churn_pct,
                triggered=True,
                severity='critical',
                message=f"Label churn {metrics.label_churn_pct:.1f}% exceeds threshold {self.config.max_label_churn_pct:.1f}%",
                timestamp=datetime.now()
            )
            triggers.append(trigger)
        
        # Check economic score threshold
        if metrics.economic_score < self.config.min_economic_score:
            trigger = RollbackTrigger(
                trigger_type='low_economic_score',
                threshold=self.config.min_economic_score,
                current_value=metrics.economic_score,
                triggered=True,
                severity='critical',
                message=f"Economic score {metrics.economic_score:.3f} below threshold {self.config.min_economic_score:.3f}",
                timestamp=datetime.now()
            )
            triggers.append(trigger)
        
        # Check silhouette score threshold
        if metrics.silhouette_score < self.config.min_silhouette_score:
            trigger = RollbackTrigger(
                trigger_type='low_silhouette_score',
                threshold=self.config.min_silhouette_score,
                current_value=metrics.silhouette_score,
                triggered=True,
                severity='warning',
                message=f"Silhouette score {metrics.silhouette_score:.3f} below threshold {self.config.min_silhouette_score:.3f}",
                timestamp=datetime.now()
            )
            triggers.append(trigger)
        
        # Check cluster stability threshold
        if metrics.cluster_stability < self.config.min_cluster_stability:
            trigger = RollbackTrigger(
                trigger_type='low_cluster_stability',
                threshold=self.config.min_cluster_stability,
                current_value=metrics.cluster_stability,
                triggered=True,
                severity='warning',
                message=f"Cluster stability {metrics.cluster_stability:.3f} below threshold {self.config.min_cluster_stability:.3f}",
                timestamp=datetime.now()
            )
            triggers.append(trigger)
        
        # Check economic score drop from baseline
        if 'economic_score' in self.baseline_metrics:
            baseline_economic = self.baseline_metrics['economic_score']
            economic_drop = (baseline_economic - metrics.economic_score) / baseline_economic
            if economic_drop > self.config.max_economic_score_drop:
                trigger = RollbackTrigger(
                    trigger_type='economic_score_drop',
                    threshold=self.config.max_economic_score_drop,
                    current_value=economic_drop,
                    triggered=True,
                    severity='critical',
                    message=f"Economic score dropped by {economic_drop:.1%} from baseline",
                    timestamp=datetime.now()
                )
                triggers.append(trigger)
        
        # Store triggers
        self.rollback_triggers.extend(triggers)
        
        return triggers
    
    def evaluate_canary_promotion(self, canary_id: str) -> Tuple[bool, str, float]:
        """
        Evaluate whether a canary should be promoted.
        
        Args:
            canary_id: Canary identifier
            
        Returns:
            Tuple of (should_promote, reason, confidence)
        """
        if canary_id not in self.active_canaries:
            return False, "Canary not found", 0.0
        
        canary = self.active_canaries[canary_id]
        
        if len(canary.metrics_history) == 0:
            return False, "No metrics available", 0.0
        
        # Calculate average metrics over the canary period
        avg_metrics = self._calculate_average_metrics(canary.metrics_history)
        
        # Check all promotion criteria
        promotion_criteria = []
        
        # Label churn check
        if avg_metrics.label_churn_pct <= self.config.max_label_churn_pct:
            promotion_criteria.append(True)
        else:
            promotion_criteria.append(False)
        
        # Economic score check
        if avg_metrics.economic_score >= self.config.min_economic_score:
            promotion_criteria.append(True)
        else:
            promotion_criteria.append(False)
        
        # Silhouette score check
        if avg_metrics.silhouette_score >= self.config.min_silhouette_score:
            promotion_criteria.append(True)
        else:
            promotion_criteria.append(False)
        
        # Cluster stability check
        if avg_metrics.cluster_stability >= self.config.min_cluster_stability:
            promotion_criteria.append(True)
        else:
            promotion_criteria.append(False)
        
        # Calculate promotion confidence
        confidence = np.mean(promotion_criteria)
        
        # Determine promotion decision
        should_promote = confidence >= self.config.rollback_threshold
        
        if should_promote:
            reason = f"All criteria met (confidence: {confidence:.1%})"
        else:
            failed_criteria = [i for i, passed in enumerate(promotion_criteria) if not passed]
            reason = f"Failed criteria: {failed_criteria} (confidence: {confidence:.1%})"
        
        return should_promote, reason, confidence
    
    def promote_canary(self, canary_id: str) -> bool:
        """
        Promote a canary to production.
        
        Args:
            canary_id: Canary identifier
            
        Returns:
            Success status
        """
        if canary_id not in self.active_canaries:
            logger.error(f"Canary {canary_id} not found")
            return False
        
        canary = self.active_canaries[canary_id]
        
        # Evaluate promotion
        should_promote, reason, confidence = self.evaluate_canary_promotion(canary_id)
        
        if should_promote:
            # Update canary status
            canary.status = 'passed'
            canary.final_decision = 'promoted'
            canary.promotion_confidence = confidence
            
            # Move to completed canaries
            self.completed_canaries.append(canary)
            del self.active_canaries[canary_id]
            
            # Update performance metrics
            self.performance_metrics['successful_promotions'] += 1
            
            logger.info(f"Canary {canary_id} promoted to production (confidence: {confidence:.1%})")
            
            # Send notification
            self._send_notification(f"Canary {canary_id} promoted: {reason}")
            
            return True
        else:
            logger.warning(f"Canary {canary_id} not promoted: {reason}")
            return False
    
    def rollback_canary(self, canary_id: str, reason: str) -> bool:
        """
        Rollback a canary deployment.
        
        Args:
            canary_id: Canary identifier
            reason: Rollback reason
            
        Returns:
            Success status
        """
        if canary_id not in self.active_canaries:
            logger.error(f"Canary {canary_id} not found")
            return False
        
        canary = self.active_canaries[canary_id]
        
        # Update canary status
        canary.status = 'rolled_back'
        canary.final_decision = 'rolled_back'
        canary.rollback_reason = reason
        
        # Move to completed canaries
        self.completed_canaries.append(canary)
        del self.active_canaries[canary_id]
        
        # Update performance metrics
        self.performance_metrics['rollbacks'] += 1
        
        logger.warning(f"Canary {canary_id} rolled back: {reason}")
        
        # Send notification
        self._send_notification(f"Canary {canary_id} rolled back: {reason}")
        
        return True
    
    def auto_rollback_check(self, canary_id: str) -> bool:
        """
        Perform automatic rollback check.
        
        Args:
            canary_id: Canary identifier
            
        Returns:
            Whether rollback was triggered
        """
        if not self.config.enable_auto_rollback:
            return False
        
        if canary_id not in self.active_canaries:
            return False
        
        canary = self.active_canaries[canary_id]
        
        if len(canary.metrics_history) == 0:
            return False
        
        # Get latest metrics
        latest_metrics = canary.metrics_history[-1]
        
        # Check for rollback triggers
        triggers = self.check_rollback_triggers(canary_id, latest_metrics)
        
        # Count critical triggers
        critical_triggers = [t for t in triggers if t.severity == 'critical']
        
        if critical_triggers:
            # Auto-rollback on critical triggers
            reason = f"Auto-rollback triggered by: {', '.join([t.trigger_type for t in critical_triggers])}"
            return self.rollback_canary(canary_id, reason)
        
        return False
    
    def _calculate_economic_score(self, 
                                cluster_labels: np.ndarray,
                                market_data: pd.DataFrame,
                                features: np.ndarray) -> float:
        """Calculate economic score for clustering."""
        if 'returns' not in market_data.columns:
            return 0.0
        
        returns = market_data['returns'].dropna()
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        
        if len(valid_labels) == 0 or len(returns) == 0:
            return 0.0
        
        # Align data lengths
        min_len = min(len(valid_labels), len(returns))
        valid_labels = valid_labels[:min_len]
        valid_returns = returns.iloc[:min_len]
        
        if len(np.unique(valid_labels)) < 2:
            return 0.0
        
        # Calculate return separation using ANOVA
        groups = [valid_returns[valid_labels == label].values 
                 for label in np.unique(valid_labels) if label != -1]
        
        if len(groups) < 2 or any(len(g) == 0 for g in groups):
            return 0.0
        
        try:
            from scipy import stats
            f_stat, p_value = stats.f_oneway(*groups)
            return f_stat if not np.isnan(f_stat) else 0.0
        except Exception as e:
            logger.error(f"Economic score calculation failed: {e}")
            return 0.0
    
    def _calculate_silhouette_score(self, cluster_labels: np.ndarray, features: np.ndarray) -> float:
        """Calculate silhouette score for clustering."""
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
            try:
                return silhouette_score(features[valid_mask], cluster_labels[valid_mask])
            except Exception as e:
                logger.error(f"Silhouette score calculation failed: {e}")
                return 0.0
        return 0.0
    
    def _calculate_cluster_stability(self, 
                                   current_labels: np.ndarray,
                                   previous_labels: Optional[np.ndarray]) -> float:
        """Calculate cluster stability."""
        if previous_labels is None or len(previous_labels) != len(current_labels):
            return 1.0  # No previous labels to compare
        
        try:
            return adjusted_rand_score(previous_labels, current_labels)
        except Exception as e:
            logger.error(f"Cluster stability calculation failed: {e}")
            return 0.0
    
    def _calculate_data_quality_score(self, 
                                    cluster_labels: np.ndarray,
                                    market_data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        # Simple data quality score based on cluster distribution
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        n_noise = np.sum(cluster_labels == -1)
        noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
        
        # Prefer more clusters and less noise
        cluster_score = min(n_clusters / 10.0, 1.0)  # Normalize to 0-1
        noise_score = 1.0 - noise_ratio  # Lower noise is better
        
        return 0.6 * cluster_score + 0.4 * noise_score
    
    def _calculate_performance_score(self, 
                                   cluster_labels: np.ndarray,
                                   features: np.ndarray,
                                   market_data: pd.DataFrame) -> float:
        """Calculate overall performance score."""
        economic_score = self._calculate_economic_score(cluster_labels, market_data, features)
        silhouette_score = self._calculate_silhouette_score(cluster_labels, features)
        data_quality_score = self._calculate_data_quality_score(cluster_labels, market_data)
        
        # Normalize scores
        economic_norm = min(economic_score / 10.0, 1.0)
        silhouette_norm = max(0, min(silhouette_score, 1.0))
        
        # Combined performance score
        performance_score = 0.4 * economic_norm + 0.4 * silhouette_norm + 0.2 * data_quality_score
        
        return performance_score
    
    def _calculate_average_metrics(self, metrics_history: List[CanaryMetrics]) -> CanaryMetrics:
        """Calculate average metrics over the canary period."""
        if not metrics_history:
            return CanaryMetrics(
                timestamp=datetime.now(),
                label_churn_pct=0.0,
                economic_score=0.0,
                silhouette_score=0.0,
                cluster_stability=0.0,
                n_clusters=0,
                n_noise=0,
                data_quality_score=0.0,
                performance_score=0.0
            )
        
        return CanaryMetrics(
            timestamp=datetime.now(),
            label_churn_pct=np.mean([m.label_churn_pct for m in metrics_history]),
            economic_score=np.mean([m.economic_score for m in metrics_history]),
            silhouette_score=np.mean([m.silhouette_score for m in metrics_history]),
            cluster_stability=np.mean([m.cluster_stability for m in metrics_history]),
            n_clusters=int(np.mean([m.n_clusters for m in metrics_history])),
            n_noise=int(np.mean([m.n_noise for m in metrics_history])),
            data_quality_score=np.mean([m.data_quality_score for m in metrics_history]),
            performance_score=np.mean([m.performance_score for m in metrics_history])
        )
    
    def _send_notification(self, message: str):
        """Send notification about canary status."""
        if self.config.notification_webhook:
            try:
                import requests
                
                payload = {
                    "text": f"Canary System: {message}",
                    "timestamp": datetime.now().isoformat()
                }
                
                response = requests.post(self.config.notification_webhook, json=payload)
                response.raise_for_status()
                
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
        else:
            logger.info(f"Notification: {message}")
    
    def get_canary_status(self, canary_id: str) -> Dict[str, Any]:
        """Get status of a specific canary."""
        if canary_id in self.active_canaries:
            canary = self.active_canaries[canary_id]
            return {
                'canary_id': canary_id,
                'status': canary.status,
                'start_time': canary.start_time,
                'end_time': canary.end_time,
                'metrics_count': len(canary.metrics_history),
                'promotion_confidence': canary.promotion_confidence,
                'final_decision': canary.final_decision
            }
        else:
            # Check completed canaries
            for canary in self.completed_canaries:
                if canary.canary_id == canary_id:
                    return {
                        'canary_id': canary_id,
                        'status': canary.status,
                        'start_time': canary.start_time,
                        'end_time': canary.end_time,
                        'metrics_count': len(canary.metrics_history),
                        'promotion_confidence': canary.promotion_confidence,
                        'final_decision': canary.final_decision,
                        'rollback_reason': canary.rollback_reason
                    }
        
        return {'error': f'Canary {canary_id} not found'}
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get canary system summary."""
        return {
            'timestamp': datetime.now(),
            'active_canaries': len(self.active_canaries),
            'completed_canaries': len(self.completed_canaries),
            'performance_metrics': self.performance_metrics,
            'rollback_triggers': len(self.rollback_triggers),
            'critical_triggers': len([t for t in self.rollback_triggers if t.severity == 'critical']),
            'config': asdict(self.config)
        }
    
    def save_canary_results(self, output_file: str = None):
        """Save canary results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"canary_results_{timestamp}.json"
        
        output_path = Path(output_file)
        
        # Prepare data for saving
        save_data = {
            'config': asdict(self.config),
            'active_canaries': {k: asdict(v) for k, v in self.active_canaries.items()},
            'completed_canaries': [asdict(c) for c in self.completed_canaries],
            'rollback_triggers': [asdict(t) for t in self.rollback_triggers],
            'performance_metrics': self.performance_metrics,
            'summary': self.get_system_summary()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        save_data = recursive_convert(save_data)
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Canary results saved to {output_path}")


def run_canary_deployment(canary_id: str,
                         clustering_func: Callable,
                         market_data: pd.DataFrame,
                         features: np.ndarray,
                         feature_names: List[str] = None,
                         config: CanaryConfig = None) -> CanaryResult:
    """
    Run a canary deployment.
    
    Args:
        canary_id: Unique identifier for the canary
        clustering_func: Function that performs clustering
        market_data: Market data
        features: Feature matrix
        feature_names: List of feature names
        config: Configuration object
        
    Returns:
        CanaryResult
    """
    canary_system = CanaryRollbackSystem(config)
    return canary_system.start_canary(canary_id, clustering_func, market_data, features, feature_names)


if __name__ == "__main__":
    # Example usage
    print("Canary and rollback system example")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    features = np.random.randn(n_samples, n_features)
    market_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, n_samples),
        'volume': np.random.lognormal(5, 0.5, n_samples)
    })
    
    # Define clustering function
    def clustering_func(features, market_data):
        # Simple clustering for demonstration
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=50, min_samples=10)
        return clusterer.fit_predict(features)
    
    # Run canary deployment
    config = CanaryConfig(
        shadow_run_days=1,  # Shortened for example
        max_label_churn_pct=15.0,
        min_economic_score=0.7,
        enable_auto_rollback=True
    )
    
    result = run_canary_deployment(
        canary_id="test_canary_001",
        clustering_func=clustering_func,
        market_data=market_data,
        features=features,
        config=config
    )
    
    print(f"Canary ID: {result.canary_id}")
    print(f"Status: {result.status}")
    print(f"Start time: {result.start_time}")
    print(f"End time: {result.end_time}")
    print(f"Promotion confidence: {result.promotion_confidence:.1%}")