"""
Risk Cluster Quality Assessor

This module provides a specialized assessor for Risk/Volatility regimes.
It evaluates how well clusters separate different risk profiles (calm, turbulent, crash-prone).

Metrics include:
- Risk Separation (Vol/Drawdown separation between regimes)
- Tail Risk Capture (Kurtosis/Skewness separation)
- Regime Stability (Temporal consistency of risk states)
- Drawdown Containment (Ability to isolate high-drawdown periods)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Reuse standard metrics where applicable
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

try:
    from src.utils.tprint import (
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
        tprint_debug
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)
    tprint_info = logging.info
    tprint_warning = logging.warning
    tprint_error = logging.error
    tprint_success = logging.info
    tprint_debug = logging.debug

logger = logging.getLogger(__name__)

@dataclass
class RiskQualityMetrics:
    """Comprehensive risk cluster quality metrics."""
    # Core Clustering
    silhouette_score: float = 0.0
    davies_bouldin_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    
    # Risk Separation Metrics
    volatility_separation_score: float = 0.0  # Ratio of Between/Within Volatility
    drawdown_separation_score: float = 0.0    # Separation of Max Drawdowns
    tail_risk_capture_score: float = 0.0      # Ability to isolate kurtosis/skew
    
    # Regime Stability
    temporal_smoothness: float = 0.0
    regime_persistence: float = 0.0
    flip_flop_ratio: float = 0.0
    
    # Detailed Breakdowns
    per_regime_risk_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    regime_transition_matrix: Dict[str, Any] = field(default_factory=dict)
    
    # Composite Score
    quality_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silhouette_score": self.silhouette_score,
            "davies_bouldin_score": self.davies_bouldin_score,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "volatility_separation_score": self.volatility_separation_score,
            "drawdown_separation_score": self.drawdown_separation_score,
            "tail_risk_capture_score": self.tail_risk_capture_score,
            "temporal_smoothness": self.temporal_smoothness,
            "regime_persistence": self.regime_persistence,
            "flip_flop_ratio": self.flip_flop_ratio,
            "per_regime_risk_metrics": self.per_regime_risk_metrics,
            "regime_transition_matrix": self.regime_transition_matrix,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp
        }

class RiskClusterQualityAssessor:
    """Assessor for Risk-Based Regimes."""

    def __init__(self, artifact_manager=None):
        self.logger = logger
        self.artifact_manager = artifact_manager

    def assess_quality(self,
                       regime_labels: np.ndarray,
                       feature_data: pd.DataFrame,
                       returns: pd.Series,
                       timestamps: Optional[pd.DatetimeIndex] = None) -> RiskQualityMetrics:
        """
        Assess quality of risk regimes.
        
        Args:
            regime_labels: Array of regime IDs
            feature_data: DataFrame of input features (risk features)
            returns: Series of returns (to calculate realized risk metrics)
            timestamps: DatetimeIndex for temporal analysis
        """
        tprint_info("🔍 Starting Risk Cluster Quality Assessment...")
        
        metrics = RiskQualityMetrics()
        
        # 1. Core Clustering Metrics (on feature space)
        # Usually features are standardized risk metrics
        if len(feature_data) > 10 and len(set(regime_labels)) > 1:
            try:
                # Sample if too large for silhouette
                if len(feature_data) > 5000:
                    idx = np.random.choice(len(feature_data), 5000, replace=False)
                    X_sample = feature_data.iloc[idx]
                    y_sample = regime_labels[idx]
                else:
                    X_sample = feature_data
                    y_sample = regime_labels
                    
                metrics.silhouette_score = float(silhouette_score(X_sample, y_sample))
                metrics.calinski_harabasz_score = float(calinski_harabasz_score(X_sample, y_sample))
                metrics.davies_bouldin_score = float(davies_bouldin_score(X_sample, y_sample))
            except Exception as e:
                tprint_warning(f"Clustering metrics calculation failed: {e}")

        # 2. Risk Separation Metrics (using actual returns)
        metrics.per_regime_risk_metrics = self._calculate_per_regime_risk(regime_labels, returns)
        
        # Volatility Separation: CV of regime volatilities / Mean Regime Vol
        vols = [m['volatility'] for m in metrics.per_regime_risk_metrics.values()]
        if vols and np.mean(vols) > 0:
            metrics.volatility_separation_score = np.std(vols) / np.mean(vols)
        
        # Drawdown Separation
        dds = [abs(m['max_drawdown']) for m in metrics.per_regime_risk_metrics.values()]
        if dds and np.mean(dds) > 0:
            metrics.drawdown_separation_score = np.std(dds) / np.mean(dds)

        # Tail Risk Capture (Kurtosis difference)
        kurts = [m['kurtosis'] for m in metrics.per_regime_risk_metrics.values()]
        if kurts:
            metrics.tail_risk_capture_score = float(np.max(kurts) - np.min(kurts))

        # 3. Temporal Stability
        if timestamps is not None:
            metrics.temporal_smoothness = self._calculate_smoothness(regime_labels)
            metrics.regime_persistence = self._calculate_persistence(regime_labels)
            metrics.flip_flop_ratio = self._calculate_flip_flop(regime_labels)

        # 4. Composite Quality Score
        # Weights: Vol Separation (30%), Drawdown Sep (20%), Silhouette (20%), Smoothness (30%)
        metrics.quality_score = (
            0.3 * min(1.0, metrics.volatility_separation_score * 2) + 
            0.2 * min(1.0, metrics.drawdown_separation_score * 2) +
            0.2 * max(0.0, (metrics.silhouette_score + 1) / 2) +
            0.3 * metrics.temporal_smoothness
        )

        tprint_success(f"✅ Risk Assessment Complete. Score: {metrics.quality_score:.3f}")
        return metrics

    def _calculate_per_regime_risk(self, labels: np.ndarray, returns: pd.Series) -> Dict[int, Dict[str, float]]:
        """Calculate risk metrics per regime."""
        results = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1: continue
            mask = labels == label
            regime_rets = returns[mask]
            
            if len(regime_rets) < 5:
                continue
                
            # Calculate metrics
            vol = float(regime_rets.std())
            
            # Max Drawdown
            cum_rets = (1 + regime_rets).cumprod()
            peak = cum_rets.expanding(min_periods=1).max()
            dd = (cum_rets - peak) / peak
            max_dd = float(dd.min())
            
            # VaR (5%)
            var_95 = float(np.percentile(regime_rets, 5))
            
            # CVaR (5%)
            cvar_95 = float(regime_rets[regime_rets <= var_95].mean())
            
            # Higher moments
            skew = float(regime_rets.skew())
            kurt = float(regime_rets.kurtosis())
            
            results[int(label)] = {
                "volatility": vol,
                "max_drawdown": max_dd,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "skewness": skew,
                "kurtosis": kurt,
                "count": int(len(regime_rets))
            }
            
        return results

    def _calculate_smoothness(self, labels: np.ndarray) -> float:
        if len(labels) < 2: return 1.0
        changes = np.sum(labels[1:] != labels[:-1])
        return 1.0 - (changes / (len(labels) - 1))

    def _calculate_persistence(self, labels: np.ndarray) -> float:
        if len(labels) < 2: return 0.0
        # Avg duration
        changes = np.where(labels[1:] != labels[:-1])[0]
        if len(changes) == 0: return float(len(labels))
        durations = np.diff(np.concatenate(([0], changes, [len(labels)])))
        return float(np.mean(durations))

    def _calculate_flip_flop(self, labels: np.ndarray) -> float:
        # A->B->A pattern
        if len(labels) < 3: return 0.0
        flip_flops = np.sum((labels[:-2] == labels[2:]) & (labels[:-2] != labels[1:-1]))
        return float(flip_flops / (len(labels) - 2))

    def generate_markdown_report(self, metrics: RiskQualityMetrics, symbol: str, output_dir: str):
        """Generate a readable Markdown report."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        filename = f"risk_cluster_quality_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = path / filename
        
        md = f"# Risk Cluster Quality Report: {symbol}\n\n"
        md += f"**Generated:** {metrics.timestamp}\n\n"
        md += f"**Overall Quality Score:** {metrics.quality_score:.4f} / 1.0\n\n"
        
        md += "## 1. Separation Metrics\n"
        md += f"- **Volatility Separation:** {metrics.volatility_separation_score:.4f} (Higher is better)\n"
        md += f"- **Drawdown Separation:** {metrics.drawdown_separation_score:.4f}\n"
        md += f"- **Tail Risk Capture:** {metrics.tail_risk_capture_score:.4f}\n"
        md += f"- **Silhouette Score:** {metrics.silhouette_score:.4f}\n\n"
        
        md += "## 2. Temporal Stability\n"
        md += f"- **Smoothness:** {metrics.temporal_smoothness:.4f}\n"
        md += f"- **Persistence:** {metrics.regime_persistence:.2f} bars\n"
        md += f"- **Flip-Flop Ratio:** {metrics.flip_flop_ratio:.4f}\n\n"
        
        md += "## 3. Per-Regime Risk Profile\n"
        md += "| Regime | Volatility | Max Drawdown | VaR (95%) | Kurtosis | Count |\n"
        md += "|--------|------------|--------------|-----------|----------|-------|\n"
        
        for r_id, m in sorted(metrics.per_regime_risk_metrics.items()):
            md += f"| {r_id} | {m['volatility']:.5f} | {m['max_drawdown']:.2%} | {m['var_95']:.4f} | {m['kurtosis']:.2f} | {m['count']} |\n"
            
        with open(filepath, 'w') as f:
            f.write(md)
            
        tprint_success(f"📝 Report saved: {filepath}")
        return str(filepath)
