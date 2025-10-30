"""
Hidden Markov Model (HMM) Regime Discovery Step

This module provides HMM-based regime discovery with temporal state transitions.
HMM enforces temporal realism in regime transitions, making it superior to GMM
for trading applications where market states evolve over time.

Key Features:
- Reduced to 4 regimes for better interpretability
- Temporal transition modeling (state-to-state probabilities)
- Economic validation per regime (Sharpe, win rate, return distributions)
- Correlation-based feature reduction
- Limited to 20 major PCs for better cohesion

Integrated with:
- clustering_optimization_goals.py for optimization targets
- cluster_quality_assessor.py for comprehensive quality assessment
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# HMM library
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None
    tprint_warning = print  # Fallback

from src.training.steps.base_step import BaseStep
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
    create_cluster_quality_assessor
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    ClusteringOptimizationGoals,
    OptimizationTargets,
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS
)
from src.utils.tprint import tprint, tprint_info, tprint_timer, tprint_success, tprint_warning, tprint_error
from src.utils.logger import system_logger

logger = system_logger.getChild('HMMRegimeDiscoveryStep')

class CorrelationBasedFeatureSelector:
    """Feature selector that removes highly correlated features."""
    
    def __init__(self, correlation_threshold: float = 0.85):
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.correlation_matrix = None
        
    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        tprint(f"🔍 Analyzing feature correlations (threshold: {self.correlation_threshold})", "INFO")
        
        # Calculate correlation matrix
        self.correlation_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        
        tprint(f"📊 Found {len(to_drop)} redundant features to remove", "INFO")
        tprint(f"📈 Original features: {len(features_df.columns)}", "INFO")
        tprint(f"📉 Reduced features: {len(features_df.columns) - len(to_drop)}", "INFO")
        
        # Show which volatility features are being removed
        volatility_to_drop = [f for f in to_drop if 'volatility' in f.lower()]
        if volatility_to_drop:
            tprint(f"⚠️ Removing {len(volatility_to_drop)} redundant volatility features:", "WARNING")
            for f in volatility_to_drop[:10]:  # Show first 10
                tprint(f"  - {f}", "WARNING")
            if len(volatility_to_drop) > 10:
                tprint(f"  ... and {len(volatility_to_drop) - 10} more", "WARNING")
        
        # Select features
        self.selected_features = [col for col in features_df.columns if col not in to_drop]
        reduced_df = features_df[self.selected_features].copy()
        
        tprint(f"✅ Feature reduction complete: {len(features_df.columns)} → {len(reduced_df.columns)}", "SUCCESS")
        
        return reduced_df

class HMMRegimeDiscoveryStep(BaseStep):
    """
    Hidden Markov Model-based regime discovery with temporal transitions.
    
    This approach:
    1. Removes highly correlated features (especially redundant volatility features)
    2. Uses HMM with 4 hidden states for better interpretability
    3. Enforces temporal transition realism (state-to-state probabilities)
    4. Evaluates economic performance per regime
    5. Limited to 20 major PCs for cohesion
    
    Advantages over GMM:
    - Temporal transitions are modeled explicitly
    - Regime persistence enforced through transition matrix
    - More interpretable with fewer states (4 vs 6)
    - Better suited for trading applications
    """
    
    def __init__(self, step_name: str = "hmm_regime_discovery", **kwargs):
        """
        Initialize HMM regime discovery step.
        
        Args:
            step_name: Name of the step (passed by launcher)
            **kwargs: Additional keyword arguments (n_states, correlation_threshold, random_state, 
                     optimization_goals, optimization_targets, covariance_type)
        """
        super().__init__(step_name)
        
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not available. Install with: pip install hmmlearn")
        
        # Load optimization goals and targets
        self.optimization_goals = kwargs.get('optimization_goals', DEFAULT_CLUSTERING_GOALS)
        self.optimization_targets = kwargs.get('optimization_targets', DEFAULT_OPTIMIZATION_TARGETS)
        
        # HMM-specific parameters - default to 4 states for interpretability
        self.n_states = kwargs.get('n_states', 4)  # Fixed at 4 for better interpretability
        self.correlation_threshold = kwargs.get('correlation_threshold', 0.85)
        self.random_state = kwargs.get('random_state', 42)
        # Use 'diag' for regularization (prevents overfitting to noisy features)
        # Options: 'diag' (regularized), 'tied' (shared cov), 'full' (flexible but overfits), 'spherical' (strict)
        self.covariance_type = kwargs.get('covariance_type', 'diag')  # Default: diag for regularization
        self.n_iter = kwargs.get('n_iter', 100)
        
        # Minimum occupancy constraints (CRITICAL for trading)
        self.min_regime_pct = kwargs.get('min_regime_pct', 0.05)  # 5% minimum
        self.min_regime_samples = kwargs.get('min_regime_samples', 50)  # 50 samples minimum
        self.merge_tiny_regimes = kwargs.get('merge_tiny_regimes', True)  # Auto-merge tiny regimes
        
        # Bootstrap validation settings
        self.bootstrap_iterations = kwargs.get('bootstrap_iterations', 1000)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        
        tprint(f"🎯 HMM Regime Discovery: {self.n_states} hidden states", "INFO")
        tprint(f"📊 Target quality thresholds:", "INFO")
        tprint(f"   - Min Silhouette: {self.optimization_targets.min_silhouette_score:.2f}", "INFO")
        tprint(f"   - Target CV Score: {self.optimization_targets.target_cv_score:.2f}", "INFO")
        tprint(f"   - Min Temporal Smoothness: {self.optimization_targets.min_temporal_smoothness:.2f}", "INFO")
        tprint(f"   - Target Economic Sharpe: {self.optimization_targets.min_sharpe:.2f}", "INFO")
        
        self.quality_assessor = create_cluster_quality_assessor()
        self.feature_selector = CorrelationBasedFeatureSelector(self.correlation_threshold)
        self.scaler = StandardScaler()
        self.pca = None
        self.hmm_model = None
        self.regime_labels = None
        self.quality_metrics = None
        self.transition_matrix = None
        self.economic_metrics = None
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HMM regime discovery step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Data timeframe
                - execution_mode: Execution mode (light/full)
                
        Returns:
            Dictionary containing execution results
        """
        try:
            symbol = config.get('symbol')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '1h')
            execution_mode = config.get('execution_mode', 'light')
            
            tprint(f"🚀 Starting HMM Regime Discovery for {symbol} on {exchange} ({timeframe})", "INFO")
            tprint(f"📊 Execution mode: {execution_mode}", "INFO")
            
            # Load data (this would typically load from artifacts)
            # For now, we'll create a placeholder that would be replaced with actual data loading
            data, features_df = self._load_data(symbol, exchange, timeframe)
            
            # Extract timestamps from data index
            timestamps = data.index if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex) else None
            if timestamps is None and hasattr(features_df, 'index') and isinstance(features_df.index, pd.DatetimeIndex):
                timestamps = features_df.index
            
            # Discover regimes with timestamps for temporal metrics
            results = self.discover_regimes(data, features_df, timestamps=timestamps)
            
            # Store timestamps in results for artifact saving
            if timestamps is not None:
                results['timestamps'] = timestamps
            
            # Save artifacts
            self._save_artifacts(results, symbol, exchange, timeframe)
            
            # Generate report
            self._generate_report(results, symbol, exchange, timeframe)
            
            return {
                'success': True,
                'n_regimes': results['n_regimes'],
                'quality_score': results['quality_metrics'].quality_score,
                'processing_time': results['processing_time'],
                'feature_reduction': results['feature_reduction_stats']
            }
            
        except Exception as e:
            logger.error(f"GMM regime discovery failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_data(self, symbol: str, exchange: str, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and features for regime discovery."""
        try:
            tprint("📂 Loading market data and features for HMM regime discovery", "INFO")
            
            # First, try to load market data from klines
            from src.utils.data.klines_parquet import get_klines_manager
            
            klines_manager = get_klines_manager(data_dir='historical_data')
            
            # Load market data
            tprint(f"📊 Loading market data for {symbol} ({timeframe})", "INFO")
            market_data = klines_manager.read_data(
                symbol=symbol,
                interval=timeframe,
                data_type="processed",
                start_date=None,  # Use all available data
                end_date=None
            )
            
            if market_data is None or len(market_data) == 0:
                tprint("❌ No market data found, falling back to placeholder data", "ERROR")
                return self._create_placeholder_data()
            
            tprint(f"✅ Loaded {len(market_data)} rows of market data", "SUCCESS")
            
            # Generate features directly from ALL market data (don't rely on limited artifacts)
            tprint(f"🔧 Generating features for ALL {len(market_data)} rows of market data", "INFO")
                features_df = self._generate_basic_features(market_data)
            
            tprint(f"✅ Generated features for {len(features_df)} samples", "SUCCESS")
            
            # Ensure alignment (should already be aligned since generated from same data)
            if len(market_data) != len(features_df):
                tprint(f"⚠️ Alignment issue after feature generation: market={len(market_data)}, features={len(features_df)}", "WARNING")
                # Align by index
                common_index = market_data.index.intersection(features_df.index)
                if len(common_index) > 0:
                    market_data = market_data.loc[common_index]
                    features_df = features_df.loc[common_index]
                    tprint(f"✅ Aligned to {len(common_index)} common timestamps", "SUCCESS")
            
            return market_data, features_df
            
        except Exception as e:
            tprint(f"❌ Failed to load data: {e}", "ERROR")
            tprint("⚠️ Falling back to placeholder data", "WARNING")
            return self._create_placeholder_data()
    
    def _create_placeholder_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create placeholder data when real data loading fails."""
        tprint("🔧 Creating placeholder data for testing", "INFO")
        
        # Create sample data
        n_samples = 480
        n_features = 50
        
        # Sample OHLCV data
        data = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 101,
            'low': np.random.randn(n_samples).cumsum() + 99,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.exponential(1000, n_samples)
        })
        
        # Sample features (including many redundant volatility features)
        features_data = {}
        for i in range(n_features):
            if i < 20:  # First 20 are volatility features (highly correlated)
                features_data[f'volatility_{i}'] = np.random.randn(n_samples) * 0.1
            else:
                features_data[f'feature_{i}'] = np.random.randn(n_samples)
        
        features_df = pd.DataFrame(features_data)
        
        return data, features_df
    
    def _generate_basic_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features from market data."""
        tprint("🔧 Generating basic technical features", "INFO")
        
        features = {}
        
        # Price-based features
        if 'close' in market_data.columns:
            close = market_data['close']
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['price_change'] = close.diff()
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f'sma_{window}'] = close.rolling(window).mean()
                features[f'ema_{window}'] = close.ewm(span=window).mean()
            
            # Volatility features
            features['volatility_5'] = close.rolling(5).std()
            features['volatility_10'] = close.rolling(10).std()
            features['volatility_20'] = close.rolling(20).std()
            
            # Price ratios
            if 'high' in market_data.columns and 'low' in market_data.columns:
                high = market_data['high']
                low = market_data['low']
                features['hl_ratio'] = high / low
                features['price_position'] = (close - low) / (high - low)
        
        # Volume features
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            features['volume_ma_5'] = volume.rolling(5).mean()
            features['volume_ma_20'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / volume.rolling(20).mean()
        
        # Create DataFrame
        features_df = pd.DataFrame(features, index=market_data.index)
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        tprint(f"✅ Generated {len(features_df.columns)} basic features", "SUCCESS")
        
        return features_df
    
    def _save_artifacts(self, results: Dict[str, Any], symbol: str, exchange: str, timeframe: str):
        """Save regime discovery artifacts."""
        try:
            tprint("💾 Saving HMM regime discovery artifacts", "INFO")
            
            # Set context for saving artifacts
            self.artifact_manager.set_context(
                step_name="hmm_regime_discovery",
                symbol=symbol,
                exchange=exchange,
                datetime=datetime.now(),
                information="regime_discovery",
                direction="long",
                model="Analyst"
            )
            
            # Save regime labels with more metadata
            if 'regime_labels' in results:
                timestamps_data = results.get('timestamps')
                if timestamps_data is None:
                    timestamps_data = pd.date_range(start='2020-01-01', periods=len(results['regime_labels']), freq='H')
                
                regime_labels_df = pd.DataFrame({
                    'regime_label': results['regime_labels'],
                    'timestamp': timestamps_data
                })
                
                # Add index for easier querying
                if isinstance(timestamps_data, (pd.DatetimeIndex, pd.Index)):
                    regime_labels_df.set_index('timestamp', inplace=True)
                
                self.artifact_manager.save_artifact(
                    artifact_name="regime_labels",
                    artifact_data=regime_labels_df,
                    artifact_type="data"
                )
                tprint("✅ Saved regime labels", "SUCCESS")
            
            # Save regime probabilities
            if 'regime_probabilities' in results:
                regime_probs_df = pd.DataFrame(
                    results['regime_probabilities'],
                    columns=[f'regime_{i}_prob' for i in range(results['regime_probabilities'].shape[1])]
                )
                self.artifact_manager.save_artifact(
                    artifact_name="regime_probabilities",
                    artifact_data=regime_probs_df,
                    artifact_type="data"
                )
                tprint("✅ Saved regime probabilities", "SUCCESS")
            
            # Save quality metrics
            if 'quality_metrics' in results:
                quality_metrics = results['quality_metrics']
                metrics_dict = {
                    'quality_score': quality_metrics.quality_score,
                    'silhouette_score': quality_metrics.silhouette_score,
                    'noise_ratio': quality_metrics.noise_ratio,
                    'balance_score': quality_metrics.balance_score,
                    'n_regimes': results.get('n_regimes', 0),
                    'processing_time': results.get('processing_time', 0.0)
                }
                
                # Add additional metrics if available
                if hasattr(quality_metrics, 'within_regime_cv') and quality_metrics.within_regime_cv is not None:
                    metrics_dict['within_regime_cv'] = quality_metrics.within_regime_cv
                if hasattr(quality_metrics, 'between_regime_cv') and quality_metrics.between_regime_cv is not None:
                    metrics_dict['between_regime_cv'] = quality_metrics.between_regime_cv
                if hasattr(quality_metrics, 'temporal_smoothness') and quality_metrics.temporal_smoothness is not None:
                    metrics_dict['temporal_smoothness'] = quality_metrics.temporal_smoothness
                
                metrics_df = pd.DataFrame([metrics_dict])
                self.artifact_manager.save_artifact(
                    artifact_name="quality_metrics",
                    artifact_data=metrics_df,
                    artifact_type="data"
                )
                tprint("✅ Saved quality metrics", "SUCCESS")
            
            # Save GMM model parameters
            if 'gmm_params' in results:
                gmm_params_df = pd.DataFrame([results['gmm_params']])
                self.artifact_manager.save_artifact(
                    artifact_name="gmm_parameters",
                    artifact_data=gmm_params_df,
                    artifact_type="data"
                )
                tprint("✅ Saved GMM parameters", "SUCCESS")
            
            # Save feature reduction statistics
            if 'feature_reduction_stats' in results:
                feature_stats_df = pd.DataFrame([results['feature_reduction_stats']])
                self.artifact_manager.save_artifact(
                    artifact_name="feature_reduction_stats",
                    artifact_data=feature_stats_df,
                    artifact_type="data"
                )
                tprint("✅ Saved feature reduction statistics", "SUCCESS")
            
            tprint("✅ All GMM regime discovery artifacts saved successfully", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to save artifacts: {e}", "ERROR")
            logger.error(f"Failed to save GMM regime discovery artifacts: {e}")
    
    def _generate_report(self, results: Dict[str, Any], symbol: str, exchange: str, timeframe: str):
        """Generate comprehensive HMM regime discovery report with detailed metrics."""
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            
            # Extract metrics
            quality_metrics = results.get('quality_metrics')
            if not quality_metrics:
                tprint("⚠️ No quality metrics available for report generation", "WARNING")
                return
            
            n_regimes = results.get('n_regimes', 0)
            regime_labels = results.get('regime_labels', np.array([]))
            processing_time = results.get('processing_time', 0.0)
            feature_stats = results.get('feature_reduction_stats', {})
            gmm_params = results.get('gmm_params', {})
            
            # Calculate regime distribution
            unique_labels, counts = np.unique(regime_labels, return_counts=True)
            total_samples = len(regime_labels)
            
            # Format conditional values to avoid f-string format specifier issues
            balance_score_str = f"{quality_metrics.balance_score:.3f}" if quality_metrics.balance_score is not None else 'N/A'
            
            # Format CV metrics
            if quality_metrics.within_regime_cv is not None and quality_metrics.between_regime_cv is not None:
                cv_ratio_value = quality_metrics.between_regime_cv / (quality_metrics.within_regime_cv + 1e-8)
                cv_ratio_str = f"{cv_ratio_value:.4f}"
                within_cv_str = f"{quality_metrics.within_regime_cv:.6f}"
                between_cv_str = f"{quality_metrics.between_regime_cv:.6f}"
                
                if cv_ratio_value > 2.0:
                    cv_interpretation = 'Excellent separation'
                elif cv_ratio_value > 1.0:
                    cv_interpretation = 'Good separation'
                else:
                    cv_interpretation = 'Fair separation'
            else:
                cv_ratio_str = 'N/A'
                within_cv_str = 'N/A'
                between_cv_str = 'N/A'
                cv_interpretation = 'N/A'
            
            # Format temporal metrics
            temporal_smoothness_str = f"{quality_metrics.temporal_smoothness:.4f}" if quality_metrics.temporal_smoothness is not None else 'N/A'
            if quality_metrics.temporal_smoothness is not None:
                if quality_metrics.temporal_smoothness > 0.8:
                    temporal_interpretation = 'Excellent temporal stability'
                elif quality_metrics.temporal_smoothness > 0.6:
                    temporal_interpretation = 'Good temporal stability'
                elif quality_metrics.temporal_smoothness > 0.4:
                    temporal_interpretation = 'Fair temporal stability'
                else:
                    temporal_interpretation = 'Poor temporal stability'
            else:
                temporal_interpretation = 'N/A'
            
            regime_persistence_str = f"{quality_metrics.regime_persistence:.2f}" if quality_metrics.regime_persistence is not None else 'N/A'
            
            # Format balance metrics
            balance_score_detailed_str = f"{quality_metrics.balance_score:.4f}" if quality_metrics.balance_score is not None else 'N/A'
            min_cluster_size_str = f"{quality_metrics.min_cluster_size_pct:.1%}" if quality_metrics.min_cluster_size_pct is not None else 'N/A'
            max_cluster_size_str = f"{quality_metrics.max_cluster_size_pct:.1%}" if quality_metrics.max_cluster_size_pct is not None else 'N/A'
            cluster_size_std_str = f"{quality_metrics.cluster_size_std:.2f}" if quality_metrics.cluster_size_std is not None else 'N/A'
            
            # Calculate target achievement status
            targets_met = []
            targets_failed = []
            
            # Check against optimization targets
            if quality_metrics.silhouette_score is not None:
                if quality_metrics.silhouette_score >= self.optimization_targets.min_silhouette_score:
                    targets_met.append(f"Silhouette Score ({quality_metrics.silhouette_score:.3f})")
                else:
                    targets_failed.append(f"Silhouette Score ({quality_metrics.silhouette_score:.3f} < {self.optimization_targets.min_silhouette_score:.2f})")
            
            if quality_metrics.temporal_smoothness is not None:
                if quality_metrics.temporal_smoothness >= self.optimization_targets.min_temporal_smoothness:
                    targets_met.append(f"Temporal Smoothness ({quality_metrics.temporal_smoothness:.3f})")
                else:
                    targets_failed.append(f"Temporal Smoothness ({quality_metrics.temporal_smoothness:.3f} < {self.optimization_targets.min_temporal_smoothness:.2f})")
            
            if self.optimization_targets.target_clusters[0] <= n_regimes <= self.optimization_targets.target_clusters[1]:
                targets_met.append(f"Cluster Count ({n_regimes})")
            else:
                targets_failed.append(f"Cluster Count ({n_regimes} outside {self.optimization_targets.target_clusters})")
            
            overall_status = "✅ TARGETS MET" if len(targets_failed) == 0 else "⚠️ PARTIAL SUCCESS"
            
            # Build comprehensive report
            report = f"""# HMM Regime Discovery Comprehensive Report

**Generated**: {timestamp.isoformat()}  
**Report ID**: `hmm_regime_discovery_{symbol}_{timeframe}_{timestamp_str}`
**Model**: Hidden Markov Model (HMM) with {self.n_states} States

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | {symbol} |
| **Exchange** | {exchange} |
| **Timeframe** | {timeframe} |
| **Processing Time** | {processing_time:.2f} seconds |
| **Success Status** | {overall_status} |
| **Regimes Discovered** | {n_regimes} |
| **Quality Score** | {quality_metrics.quality_score:.3f} |
| **Noise Ratio** | {quality_metrics.noise_ratio:.1%} |

### Optimization Targets Achievement

**Targets Met** ({len(targets_met)}/{len(targets_met) + len(targets_failed)}):
"""
            for target in targets_met:
                report += f"- ✅ {target}\n"
            
            if targets_failed:
                report += "\n**Targets Not Met**:\n"
                for target in targets_failed:
                    report += f"- ❌ {target}\n"
            
            report += "\n---\n"
            
            # Calculate average regime size
            avg_regime_size = total_samples / max(n_regimes, 1)
            
            report += f"""
## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: {n_regimes}
- **Noise Points**: {quality_metrics.noise_ratio:.1%} of total samples (GMM: 0.0% - no noise)
- **Average Regime Size**: {avg_regime_size:.0f} samples per regime
- **Balance Score**: {balance_score_str} (higher is better)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
"""
            
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                report += f"| **Regime {label}** | {count:,} | {percentage:.1f}% |\n"
            
            # Transition Matrix Section (HMM-specific)
            if 'transition_matrix' in results and results['transition_matrix'] is not None:
                transition_matrix = results['transition_matrix']
                report += f"""

---

## 🔄 HMM Transition Matrix

The transition matrix shows the probability of moving from one regime to another.
Higher diagonal values indicate regime persistence (states tend to stay).

| From \\ To | State 0 | State 1 | State 2 | State 3 |
|-----------|---------|---------|---------|---------|
"""
                for i in range(min(transition_matrix.shape[0], 4)):
                    row_str = f"| **State {i}** |"
                    for j in range(min(transition_matrix.shape[1], 4)):
                        prob = transition_matrix[i, j]
                        # Highlight diagonal (persistence) and high transitions
                        if i == j:
                            row_str += f" **{prob:.3f}** |"  # Bold for persistence
                        elif prob > 0.2:
                            row_str += f" *{prob:.3f}* |"  # Italics for significant transitions
                        else:
                            row_str += f" {prob:.3f} |"
                    report += row_str + "\n"
                
                # Analyze transition characteristics
                persistence_probs = np.diag(transition_matrix)[:4]
                avg_persistence = np.mean(persistence_probs)
                
                report += f"""

### Transition Analysis
- **Average Regime Persistence**: {avg_persistence:.1%} (probability of staying in same state)
- **Regime-Specific Persistence**:
"""
                for i, persist_prob in enumerate(persistence_probs):
                    report += f"  - State {i}: {persist_prob:.1%} persistence\n"
            
            # Economic Performance Section (HMM-specific)
            if 'economic_metrics' in results and results['economic_metrics']:
                economic_metrics = results['economic_metrics']
                report += f"""

---

## 💰 Economic Performance Per Regime

This section evaluates trading performance within each regime.

| Regime | Sharpe Ratio | Win Rate | Expected Return | Max Drawdown | Volatility Clustering |
|--------|--------------|----------|-----------------|--------------|----------------------|
"""
                for regime_id in sorted(economic_metrics.keys()):
                    metrics = economic_metrics[regime_id]
                    sharpe = metrics.get('sharpe', 0.0)
                    win_rate = metrics.get('win_rate', 0.0)
                    exp_ret = metrics.get('expected_return', 0.0)
                    max_dd = metrics.get('max_drawdown', 0.0)
                    vol_clust = metrics.get('volatility_clustering', 0.0)
                    
                    # Status emoji based on Sharpe
                    if sharpe >= 1.0:
                        status = "🟢"
                    elif sharpe >= 0.5:
                        status = "🟡"
                    else:
                        status = "🔴"
                    
                    # Add reliability indicator
                    n_samples = metrics.get('n_samples', 0)
                    if n_samples < 50:
                        reliability_emoji = "🔴"
                    elif n_samples < 100:
                        reliability_emoji = "🟡"
                    else:
                        reliability_emoji = "🟢"
                    
                    report += f"| {status} **Regime {regime_id}** {reliability_emoji} | {sharpe:.3f} | {win_rate:.1%} | {exp_ret:.4f} | {max_dd:.2%} | {vol_clust:.3f} |\n"
                
                # Statistical Reliability Legend
                report += """

**Reliability Legend**:
- 🟢 RELIABLE: N ≥ 100 samples (statistics are trustworthy)
- 🟡 MARGINAL: 50 ≤ N < 100 samples (use with caution)
- 🔴 UNRELIABLE: N < 50 samples (DO NOT TRADE on these stats)
"""
                
                # Bootstrap Confidence Intervals
                report += f"""

### Bootstrap Confidence Intervals (95% CI)

Statistical validation using block bootstrap ({self.bootstrap_iterations} iterations):

| Regime | Sharpe CI | Mean Return CI | Samples | Reliability |
|--------|-----------|----------------|---------|-------------|
"""
                for regime_id in sorted(economic_metrics.keys()):
                    metrics = economic_metrics[regime_id]
                    bootstrap_ci = metrics.get('bootstrap_ci', {})
                    
                    sharpe = metrics.get('sharpe', 0.0)
                    sharpe_lower = bootstrap_ci.get('sharpe_ci_lower', 0.0)
                    sharpe_upper = bootstrap_ci.get('sharpe_ci_upper', 0.0)
                    
                    mean = metrics.get('expected_return', 0.0)
                    mean_lower = bootstrap_ci.get('mean_return_ci_lower', 0.0)
                    mean_upper = bootstrap_ci.get('mean_return_ci_upper', 0.0)
                    
                    n_samples = metrics.get('n_samples', 0)
                    reliability = bootstrap_ci.get('reason', 'Unknown')
                    
                    report += f"| **Regime {regime_id}** | [{sharpe_lower:.2f}, {sharpe_upper:.2f}] | [{mean_lower:.6f}, {mean_upper:.6f}] | {n_samples} | {reliability} |\n"
                
                report += """

**Trading Rule**: Only act on regimes where the **lower bound of Sharpe CI > 0.5** OR **(Sharpe ≥ 1.0 AND mean return CI lower > 0)**

"""
                
                # Production-Ready Trading Status
                tradeable_regimes = results.get('tradeable_regimes', {})
                if tradeable_regimes:
                    report += """

### 🎯 Production-Ready Trading Status

Conservative evaluation for live trading based on strict statistical criteria:

| Regime | Status | Samples | Sharpe (CI Lower) | Mean Return (CI Lower) | Decision |
|--------|--------|---------|-------------------|------------------------|----------|
"""
                    for regime_id in sorted(tradeable_regimes.keys()):
                        status = tradeable_regimes[regime_id]
                        metrics = economic_metrics.get(regime_id, {})
                        n_samples = metrics.get('n_samples', 0)
                        sharpe = metrics.get('sharpe', 0.0)
                        bootstrap_ci = metrics.get('bootstrap_ci', {})
                        sharpe_lower = bootstrap_ci.get('sharpe_ci_lower', 0.0)
                        mean_lower = bootstrap_ci.get('mean_return_ci_lower', 0.0)
                        
                        # Status emoji
                        if status == 'LONG':
                            status_emoji = "🟢 LONG"
                            decision = "**Trade with 0.5x size** (scale by vol)"
                        elif status == 'FLAT':
                            status_emoji = "🟡 FLAT"
                            decision = "Hold flat (insufficient edge)"
                        else:
                            status_emoji = "🔴 NO TRADE"
                            decision = "Do NOT trade (unreliable)"
                        
                        report += f"| **Regime {regime_id}** | {status_emoji} | {n_samples} | {sharpe:.2f} ({sharpe_lower:.2f}) | {mean_lower:.6f} | {decision} |\n"
                    
                    report += """

**Production Rules**:
1. ✅ **N ≥ 100**: Sufficient sample size for statistical reliability
2. ✅ **Sharpe CI lower ≥ 0.5** OR **(Sharpe ≥ 1.0 AND Mean Return CI lower > 0)**: Edge survives conservative CI
3. ⚠️ **Conservative sizing**: Use 0.5x max position, scale by volatility
4. ⚠️ **Do NOT short**: Small negative regimes are unreliable — stay flat instead

"""
                
                # Detailed distribution stats
                report += f"""

### Return Distribution Details

"""
                for regime_id in sorted(economic_metrics.keys()):
                    metrics = economic_metrics[regime_id]
                    dist = metrics.get('return_distribution', {})
                    
                    report += f"""#### Regime {regime_id} Return Statistics
- **Mean Return**: {dist.get('mean', 0.0):.6f} ({dist.get('mean', 0.0)*100:.4f}%)
- **Median Return**: {dist.get('median', 0.0):.6f}
- **Std Dev**: {dist.get('std', 0.0):.6f}
- **Skewness**: {dist.get('skew', 0.0):.4f} {'(right-tailed)' if dist.get('skew', 0.0) > 0 else '(left-tailed)' if dist.get('skew', 0.0) < 0 else '(symmetric)'}
- **Kurtosis**: {dist.get('kurtosis', 0.0):.4f} {'(fat-tailed)' if dist.get('kurtosis', 0.0) > 0 else '(thin-tailed)' if dist.get('kurtosis', 0.0) < 0 else '(normal)'}
- **Range**: [{dist.get('min', 0.0):.4f}, {dist.get('max', 0.0):.4f}]
- **IQR**: [{dist.get('percentile_25', 0.0):.4f}, {dist.get('percentile_75', 0.0):.4f}]
- **Total Return**: {metrics.get('total_return', 0.0):.4f}
- **Samples**: {metrics.get('n_samples', 0)}

"""
            
            # Detailed Quality Metrics Section
            report += f"""
---

## 📈 Comprehensive Quality Metrics (from cluster_quality_assessor.py)

### Overall Quality Score: {quality_metrics.quality_score:.3f}

**Quality Score Breakdown:**
"""
            
            # Calculate component contributions (matching the weights used)
            from src.training.steps.market_analysis.clusters.cluster_quality_assessor import QualityThresholds
            
            components = []
            if quality_metrics.within_regime_cv is not None and quality_metrics.between_regime_cv is not None:
                cv_ratio = quality_metrics.between_regime_cv / (quality_metrics.within_regime_cv + 1e-8)
                cv_normalized = np.tanh(cv_ratio)
                components.append(('CV Ratio', cv_normalized, QualityThresholds.WEIGHT_CV_RATIO))
            
            if quality_metrics.silhouette_score is not None:
                silhouette_normalized = (quality_metrics.silhouette_score + 1) / 2
                components.append(('Silhouette Score', silhouette_normalized, QualityThresholds.WEIGHT_SILHOUETTE))
            
            if quality_metrics.temporal_smoothness is not None:
                components.append(('Temporal Smoothness', quality_metrics.temporal_smoothness, QualityThresholds.WEIGHT_TEMPORAL_SMOOTHNESS))
            
            if quality_metrics.balance_score is not None:
                components.append(('Balance Score', quality_metrics.balance_score, QualityThresholds.WEIGHT_BALANCE))
            
            noise_score = 1.0 - quality_metrics.noise_ratio
            components.append(('Noise Ratio (inverted)', noise_score, QualityThresholds.WEIGHT_NOISE_RATIO))
            
            report += "\n| Metric | Normalized Value | Weight | Contribution |\n"
            report += "|--------|------------------|--------|--------------|\n"
            
            for metric_name, value, weight in components:
                contribution = value * weight
                report += f"| **{metric_name}** | {value:.4f} | {weight:.2%} | {contribution:.4f} |\n"
            
            report += f"""
**Total Weight**: {sum(w for _, _, w in components):.2%}  
**Weighted Score**: {quality_metrics.quality_score:.4f}

---

### Core Clustering Metrics
"""
            
            if quality_metrics.silhouette_score is not None:
                report += f"- **Silhouette Score**: {quality_metrics.silhouette_score:.4f} (range: [-1, 1], higher is better)\n"
                silhouette_quality = "Excellent" if quality_metrics.silhouette_score > 0.5 else "Good" if quality_metrics.silhouette_score > 0.25 else "Fair" if quality_metrics.silhouette_score > 0 else "Poor"
                report += f"  - *Interpretation*: {silhouette_quality} cluster separation\n"
            
            if quality_metrics.calinski_harabasz_score is not None:
                report += f"- **Calinski-Harabasz Score**: {quality_metrics.calinski_harabasz_score:.2f} (higher is better)\n"
            
            if quality_metrics.davies_bouldin_score is not None:
                report += f"- **Davies-Bouldin Score**: {quality_metrics.davies_bouldin_score:.4f} (lower is better)\n"
            
            report += f"""
### Coefficient of Variation (CV) Metrics

**CV Ratio**: {cv_ratio_str}

- **Within-Regime CV**: {within_cv_str} (lower = more cohesive regimes)
- **Between-Regime CV**: {between_cv_str} (higher = better separation)
- **CV Ratio Interpretation**: {cv_interpretation}

### Temporal Metrics

- **Temporal Smoothness**: {temporal_smoothness_str} (range: [0, 1], higher = more stable over time)
  - *Interpretation*: {temporal_interpretation}

- **Regime Persistence**: {regime_persistence_str} periods (average duration)

### Balance Metrics

- **Balance Score**: {balance_score_detailed_str} (range: [0, 1], higher = more balanced)
- **Min Cluster Size**: {min_cluster_size_str} of total samples
- **Max Cluster Size**: {max_cluster_size_str} of total samples
- **Cluster Size Std Dev**: {cluster_size_std_str}

### Per-Regime Details
"""
            
            if quality_metrics.per_regime_metrics:
                for regime_id, regime_data in quality_metrics.per_regime_metrics.items():
                    report += f"""
#### 🎯 Regime {regime_id}
"""
                    for key, value in regime_data.items():
                        if isinstance(value, float):
                            report += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
                        elif isinstance(value, (int, np.integer)):
                            report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                        else:
                            report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            
            # Feature Reduction Statistics
            if feature_stats:
                report += f"""
---

## 🔧 Feature Engineering Details

### Correlation-Based Feature Reduction
- **Original Features**: {feature_stats.get('original_features', 0)}
- **Reduced Features**: {feature_stats.get('reduced_features', 0)}
- **Features Removed**: {feature_stats.get('removed_features', 0)} ({feature_stats.get('removed_features', 0) / max(feature_stats.get('original_features', 1), 1) * 100:.1f}%)
- **Correlation Threshold**: {feature_stats.get('correlation_threshold', 0.85):.2f}

### Dimensionality Reduction
"""
                if results.get('pca_explained_variance_ratio'):
                    pca_var = results['pca_explained_variance_ratio']
                    total_var_explained = sum(pca_var)
                    report += f"- **PCA Applied**: ✅ Yes\n"
                    report += f"- **Total Variance Explained**: {total_var_explained:.1%}\n"
                    report += f"- **Number of Principal Components**: {len(pca_var)}\n"
                    if len(pca_var) > 0:
                        report += f"- **Top 5 Components Variance**: {', '.join([f'{v:.1%}' for v in pca_var[:5]])}\n"
                else:
                    report += "- **PCA Applied**: ❌ No\n"
            
            # HMM Parameters
            hmm_params = results.get('hmm_params', {})
            if hmm_params:
                report += f"""
### HMM Model Parameters
- **Number of Hidden States**: {hmm_params.get('n_states', 'N/A')}
- **Covariance Type**: {hmm_params.get('covariance_type', 'N/A')}
- **Maximum Iterations**: {hmm_params.get('n_iter', 'N/A')}
- **Random State**: {hmm_params.get('random_state', 'N/A')}
- **Converged**: {'✅ Yes' if hmm_params.get('converged', False) else '❌ No'}

---

## 🎯 Optimization Goals & Targets

This GMM regime discovery run was guided by the following optimization goals from `clustering_optimization_goals.py`:

### Cluster Configuration Targets
- **Target Cluster Count**: {self.optimization_targets.target_clusters[0]}-{self.optimization_targets.target_clusters[1]} clusters
- **Minimum Cluster Size**: {self.optimization_targets.min_cluster_size_pct:.1%} of total samples
- **Maximum Cluster Size**: {self.optimization_targets.max_cluster_size_pct:.1%} of total samples

### Quality Targets
- **Minimum Silhouette Score**: {self.optimization_targets.min_silhouette_score:.2f}
- **Target Silhouette Score**: {self.optimization_targets.target_silhouette_score:.2f}
- **Minimum Temporal Smoothness**: {self.optimization_targets.min_temporal_smoothness:.2f}
- **Target Temporal Smoothness**: {self.optimization_targets.target_temporal_smoothness:.2f}
- **Minimum CV Score**: {self.optimization_targets.min_cv_score:.2f}
- **Target CV Score**: {self.optimization_targets.target_cv_score:.2f}

### Economic Targets (for future integration)
- **Minimum Sharpe Ratio**: {self.optimization_targets.min_sharpe:.2f}
- **Target Sharpe Ratio**: {self.optimization_targets.target_sharpe:.2f}
- **Max Drawdown Threshold**: {self.optimization_targets.max_drawdown_threshold:.1%}

---

## 📊 Quality Score Interpretation

**Score: {quality_metrics.quality_score:.3f}**

| Score Range | Interpretation |
|-------------|----------------|
| 0.70 - 1.00 | Excellent: Highly distinct regimes with strong temporal stability |
| 0.50 - 0.70 | Good: Clear regime separation with reasonable stability |
| 0.30 - 0.50 | Moderate: Some regime distinction, room for improvement |
| 0.00 - 0.30 | Poor: Weak regime separation, consider parameter tuning |

**Current Status**: {('Excellent' if quality_metrics.quality_score >= 0.70 else 'Good' if quality_metrics.quality_score >= 0.50 else 'Moderate' if quality_metrics.quality_score >= 0.30 else 'Poor')}

---

*Generated by HMM Regime Discovery at {timestamp.isoformat()}*

"""
            
            # Save report to file
            report_dir = Path("outcomes") / f"hmm_regime_discovery_{symbol}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"hmm_regime_discovery_report_{symbol}_{timestamp_str}.md"
            report_file.write_text(report)
            
            tprint(f"✅ Comprehensive HMM report saved: {report_file}", "SUCCESS")
            tprint(f"📊 Quality Score: {quality_metrics.quality_score:.3f} ({'Excellent' if quality_metrics.quality_score >= 0.70 else 'Good' if quality_metrics.quality_score >= 0.50 else 'Moderate' if quality_metrics.quality_score >= 0.30 else 'Poor'})", "SUCCESS")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive HMM report: {e}", exc_info=True)
            tprint(f"⚠️ Failed to generate report: {e}", "WARNING")
        
    def discover_regimes(self, 
                        data: pd.DataFrame, 
                        features_df: pd.DataFrame,
                        timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Discover regimes using GMM with correlation-based feature reduction.
        
        Args:
            data: Original OHLCV data
            features_df: Feature matrix
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary containing regime discovery results
        """
        tprint("🚀 Starting HMM Regime Discovery with Temporal Transition Modeling", "INFO")
        start_time = time.time()
        
        try:
            # Step 1: Remove highly correlated features
            with tprint_timer("Correlation-Based Feature Reduction"):
                reduced_features = self.feature_selector.fit_transform(features_df)
            
            # Step 2: Standardize features (normalize to mean=0, std=1)
            with tprint_timer("Feature Standardization"):
                scaled_features = self.scaler.fit_transform(reduced_features)
                scaled_df = pd.DataFrame(scaled_features, 
                                       columns=reduced_features.columns,
                                       index=reduced_features.index)
                
                # Verify normalization
                mean_check = np.abs(scaled_df.mean().mean())
                std_check = scaled_df.std().mean()
                tprint(f"✅ Feature normalization verified: mean={mean_check:.6f}, std={std_check:.3f}", "SUCCESS")
            
            # Step 3: PCA dimensionality reduction (ALWAYS apply if > 5 features)
            # Target: 10-20 components explaining ~60-80% variance to reduce noise
            min_features_for_pca = 5
            target_pcs = 10  # Target 10 PCs for optimal silhouette/CV
            variance_target = 0.60  # Explain at least 60% variance
            
            if scaled_df.shape[1] > min_features_for_pca:
                with tprint_timer("PCA Dimensionality Reduction"):
                    # Try variance-based PCA first (target 60% variance)
                    pca_variance = PCA(n_components=variance_target, random_state=self.random_state, svd_solver='full')
                    pca_features_var = pca_variance.fit_transform(scaled_features)
                    n_components_var = pca_features_var.shape[1]
                    
                    # Use target_pcs or variance-based, whichever gives fewer components
                    n_components_final = min(target_pcs, n_components_var, scaled_df.shape[1])
                    
                    # Final PCA with selected number of components
                    self.pca = PCA(n_components=n_components_final, random_state=self.random_state)
                    pca_features = self.pca.fit_transform(scaled_features)
                    
                    # Normalize PCA features to ensure mean=0, std=1
                    pca_scaler = StandardScaler()
                    pca_features_normalized = pca_scaler.fit_transform(pca_features)
                    
                    scaled_df = pd.DataFrame(pca_features_normalized, 
                                           columns=[f'PC_{i+1}' for i in range(pca_features_normalized.shape[1])],
                                           index=reduced_features.index)
                    
                    # Verify normalization after PCA
                    pc_mean_check = np.abs(scaled_df.mean().mean())
                    pc_std_check = scaled_df.std().mean()
                    
                    explained_var = self.pca.explained_variance_ratio_.sum()
                    tprint(f"📊 PCA applied: {scaled_features.shape[1]} → {n_components_final} components", "INFO")
                    tprint(f"📊 Explained variance: {explained_var:.1%} (target: ≥{variance_target:.0%})", "INFO")
                    tprint(f"✅ PCA features normalized: mean={pc_mean_check:.6f}, std={pc_std_check:.3f}", "SUCCESS")
                    tprint(f"🎯 Using PCA to reduce noise and improve regime discriminability", "SUCCESS")
            
            # Step 4: Fit HMM with temporal transitions
            with tprint_timer("HMM Model Fitting"):
                self.hmm_model = self._fit_hmm(scaled_df)
            
            # Step 5: Predict regimes using Viterbi algorithm (optimal state sequence)
            with tprint_timer("HMM State Prediction (Viterbi)"):
                # Use the final scaled_df (which may be PCA-transformed)
                features_for_prediction = scaled_df.values
                self.regime_labels = self.hmm_model.predict(features_for_prediction)
                
                # Get state probabilities (posterior probabilities given observations)
                regime_probs = self.hmm_model.predict_proba(features_for_prediction)
                
                # Extract transition matrix
                self.transition_matrix = self.hmm_model.transmat_
                tprint("📊 HMM Transition Matrix:", "INFO")
                for i in range(self.n_states):
                    tprint(f"   State {i} → " + " | ".join([f"S{j}: {self.transition_matrix[i,j]:.3f}" for j in range(self.n_states)]), "INFO")
            
            # Step 6: Assess quality
            with tprint_timer("Quality Assessment"):
                # Convert timestamps to DatetimeIndex if needed
                if timestamps is not None and not isinstance(timestamps, pd.DatetimeIndex):
                    if isinstance(timestamps, pd.Series):
                        timestamps = pd.DatetimeIndex(timestamps)
                    else:
                        timestamps = pd.DatetimeIndex(timestamps)
                
                self.quality_metrics = self._assess_quality(scaled_df, self.regime_labels, timestamps)
                
                # Log quality metrics against optimization targets
                tprint("📊 Quality Assessment vs Optimization Targets:", "INFO")
                if self.quality_metrics.silhouette_score is not None:
                    target_silhouette = self.optimization_targets.min_silhouette_score
                    status = "✅" if self.quality_metrics.silhouette_score >= target_silhouette else "❌"
                    tprint(f"   Silhouette: {self.quality_metrics.silhouette_score:.3f} (target: ≥{target_silhouette:.2f}) {status}", "INFO")
                
                if self.quality_metrics.temporal_smoothness is not None:
                    target_temporal = self.optimization_targets.min_temporal_smoothness
                    status = "✅" if self.quality_metrics.temporal_smoothness >= target_temporal else "❌"
                    tprint(f"   Temporal Smoothness: {self.quality_metrics.temporal_smoothness:.3f} (target: ≥{target_temporal:.2f}) {status}", "INFO")
                
                n_clusters = len(np.unique(self.regime_labels))
                tprint(f"   Cluster Count: {n_clusters} (using {self.n_states} HMM states)", "INFO")
            
            # Step 6.5: Post-process tiny regimes (CRITICAL FOR TRADING)
            if self.merge_tiny_regimes:
                with tprint_timer("Tiny Regime Post-Processing"):
                    self.regime_labels, regime_mapping = self._merge_tiny_regimes(
                        self.regime_labels, 
                        scaled_df,
                        min_samples=self.min_regime_samples,
                        min_pct=self.min_regime_pct
                    )
                    
                    if regime_mapping:
                        tprint(f"📊 Regime remapping applied: {regime_mapping}", "INFO")
            
            # Step 7: Evaluate economic performance per regime
            with tprint_timer("Economic Evaluation Per Regime"):
                self.economic_metrics = self._evaluate_regime_economics(data, self.regime_labels, timestamps)
                
                # Log economic metrics with reliability warnings
                tprint("📊 Economic Performance Per Regime:", "INFO")
                if self.economic_metrics:
                    for regime_id, metrics in self.economic_metrics.items():
                        sharpe = metrics.get('sharpe', 0.0)
                        win_rate = metrics.get('win_rate', 0.0)
                        n_samples = metrics.get('n_samples', 0)
                        
                        # Reliability check
                        if n_samples < 50:
                            reliability = "🔴 UNRELIABLE"
                        elif n_samples < 100:
                            reliability = "🟡 MARGINAL"
                        else:
                            reliability = "🟢 RELIABLE"
                        
                        status = "✅" if sharpe >= self.optimization_targets.min_sharpe else "❌"
                        tprint(f"   Regime {regime_id}: Sharpe={sharpe:.3f}, Win Rate={win_rate:.1%}, N={n_samples} {reliability} {status}", "INFO")
            
            # Step 7.5: Identify production-ready tradeable regimes
            with tprint_timer("Tradeable Regime Identification"):
                tradeable_regimes = self._identify_tradeable_regimes(
                    self.economic_metrics,
                    min_sharpe_ci_lower=0.5,
                    min_sharpe_point=1.0
                )
            
            # Step 8: Generate results
            processing_time = time.time() - start_time
            
            results = {
                'regime_labels': self.regime_labels,
                'regime_probabilities': regime_probs,
                'n_regimes': self.n_states,
                'n_noise_points': 0,  # HMM doesn't have noise points
                'noise_ratio': 0.0,
                'quality_metrics': self.quality_metrics,
                'processing_time': processing_time,
                'transition_matrix': self.transition_matrix,
                'economic_metrics': self.economic_metrics,
                'tradeable_regimes': tradeable_regimes,  # Production-ready regime status
                'feature_reduction_stats': {
                    'original_features': len(features_df.columns),
                    'reduced_features': len(reduced_features.columns),
                    'removed_features': len(features_df.columns) - len(reduced_features.columns),
                    'correlation_threshold': self.correlation_threshold
                },
                'hmm_params': {
                    'n_states': self.n_states,
                    'covariance_type': self.covariance_type,
                    'n_iter': self.n_iter,
                    'random_state': self.random_state,
                    'converged': self.hmm_model.monitor_.converged if hasattr(self.hmm_model, 'monitor_') else True
                },
                'pca_explained_variance_ratio': self.pca.explained_variance_ratio_.tolist() if self.pca else None
            }
            
            tprint(f"✅ HMM Regime Discovery Complete: {self.n_states} regimes in {processing_time:.2f}s", "SUCCESS")
            tprint(f"📊 Quality Score: {self.quality_metrics.quality_score:.3f}", "SUCCESS")
            
            return results
            
        except Exception as e:
            logger.error(f"HMM regime discovery failed: {e}")
            raise
    
    def _fit_hmm(self, features_df: pd.DataFrame) -> hmm.GaussianHMM:
        """
        Fit Hidden Markov Model with Gaussian emissions.
        
        Args:
            features_df: Normalized feature matrix (20 PCs)
            
        Returns:
            Fitted GaussianHMM model
        """
        logger.info(f"Fitting HMM with {self.n_states} hidden states, covariance_type={self.covariance_type}")
        
                try:
            # Create HMM model
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                        random_state=self.random_state,
                verbose=False
                    )
            
            # Fit model to data (HMM learns temporal transitions)
            features_array = features_df.values
            model.fit(features_array)
                    
            # Calculate quality metrics
            labels = model.predict(features_array)
            log_likelihood = model.score(features_array)
            
            try:
                    silhouette = silhouette_score(features_df, labels)
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
                silhouette = 0.0
            
            # Check convergence
            converged = model.monitor_.converged if hasattr(model, 'monitor_') else True
            
            logger.info(f"HMM fitting complete: log_likelihood={log_likelihood:.2f}, silhouette={silhouette:.3f}, converged={converged}")
            tprint(f"🏆 HMM fitted: n_states={self.n_states}, log_likelihood={log_likelihood:.2f}, silhouette={silhouette:.3f}", "SUCCESS")
            
            if not converged:
                tprint("⚠️ HMM did not fully converge, but proceeding with current solution", "WARNING")
            
            return model
                        
                except Exception as e:
            logger.error(f"HMM fitting failed: {e}", exc_info=True)
            raise
    
    def _evaluate_regime_economics(self, data: pd.DataFrame, regime_labels: np.ndarray, 
                                    timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate economic performance per regime.
        
        Calculates:
        - Sharpe ratio per regime
        - Win rate
        - Expected return per trade
        - Return distribution statistics
        - Volatility clustering signatures
        
        Args:
            data: Market OHLCV data
            regime_labels: Regime assignments
            timestamps: Optional timestamps
            
        Returns:
            Dictionary of economic metrics per regime
        """
        try:
            # Calculate returns
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
            else:
                tprint("⚠️ No 'close' price found, using placeholder returns", "WARNING")
                returns = pd.Series(np.random.randn(len(data)) * 0.01, index=data.index)
            
            economic_metrics = {}
            unique_regimes = np.unique(regime_labels)
            
            tprint(f"📊 Evaluating economic performance for {len(unique_regimes)} regimes", "INFO")
            
            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) < 2:
                    continue
        
                # Calculate Sharpe ratio (annualized for 1h data)
                mean_return = regime_returns.mean()
                std_return = regime_returns.std()
                
                if std_return > 0:
                    # Annualization: sqrt(24 * 365) for hourly data
                    sharpe = (mean_return / std_return) * np.sqrt(24 * 365)
                else:
                    sharpe = 0.0
                
                # Win rate
                wins = (regime_returns > 0).sum()
                total_trades = len(regime_returns)
                win_rate = wins / total_trades if total_trades > 0 else 0.0
                
                # Expected return per trade
                expected_return = mean_return
                
                # Return distribution stats
                return_distribution = {
                    'mean': float(mean_return),
                    'median': float(regime_returns.median()),
                    'std': float(std_return),
                    'skew': float(regime_returns.skew()) if len(regime_returns) > 2 else 0.0,
                    'kurtosis': float(regime_returns.kurtosis()) if len(regime_returns) > 2 else 0.0,
                    'min': float(regime_returns.min()),
                    'max': float(regime_returns.max()),
                    'percentile_25': float(regime_returns.quantile(0.25)),
                    'percentile_75': float(regime_returns.quantile(0.75))
                }
                
                # Volatility clustering (using ARCH/GARCH-like metric)
                abs_returns = regime_returns.abs()
                if len(abs_returns) > 10:
                    vol_autocorr = abs_returns.autocorr(lag=1) if len(abs_returns) > 1 else 0.0
                else:
                    vol_autocorr = 0.0
                
                # Maximum drawdown within regime
                cumulative_returns = (1 + regime_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
                
                # Bootstrap confidence intervals for Sharpe and mean return
                bootstrap_results = self._bootstrap_regime_stats(regime_returns)
                
                economic_metrics[int(regime_id)] = {
                    'sharpe': float(sharpe),
                    'win_rate': float(win_rate),
                    'expected_return': float(expected_return),
                    'return_distribution': return_distribution,
                    'volatility_clustering': float(vol_autocorr),
                    'max_drawdown': float(max_drawdown),
                    'total_return': float(regime_returns.sum()),
                    'n_samples': int(total_trades),
                    'bootstrap_ci': bootstrap_results  # Confidence intervals
                }
            
            return economic_metrics
            
        except Exception as e:
            logger.error(f"Economic evaluation failed: {e}", exc_info=True)
            return {}
    
    def _bootstrap_regime_stats(self, returns: pd.Series, n_iterations: int = None) -> Dict[str, Any]:
        """
        Bootstrap confidence intervals for regime statistics.
        
        Args:
            returns: Regime returns
            n_iterations: Bootstrap iterations (uses self.bootstrap_iterations if None)
            
        Returns:
            Dictionary with CIs for Sharpe, mean return, and reliability flag
        """
        try:
            if n_iterations is None:
                n_iterations = self.bootstrap_iterations
            
            n_samples = len(returns)
            
            # Insufficient samples - mark as unreliable
            if n_samples < 10:
                return {
                    'sharpe_ci_lower': 0.0,
                    'sharpe_ci_upper': 0.0,
                    'mean_return_ci_lower': 0.0,
                    'mean_return_ci_upper': 0.0,
                    'reliable': False,
                    'reason': f'N={n_samples} too small'
                }
            
            # Block bootstrap to preserve autocorrelation
            block_size = max(5, int(np.sqrt(n_samples)))
            
            sharpe_samples = []
            mean_samples = []
            
            for _ in range(n_iterations):
                # Sample with replacement (block bootstrap)
                n_blocks = max(1, n_samples // block_size)
                indices = []
                
                for _ in range(n_blocks):
                    start_idx = np.random.randint(0, max(1, n_samples - block_size + 1))
                    block_indices = list(range(start_idx, min(start_idx + block_size, n_samples)))
                    indices.extend(block_indices)
                
                # Trim to original length
                indices = indices[:n_samples]
                boot_returns = returns.iloc[indices].values
                
                # Calculate Sharpe for this bootstrap sample
                mean_ret = np.mean(boot_returns)
                std_ret = np.std(boot_returns)
                
                if std_ret > 0:
                    boot_sharpe = (mean_ret / std_ret) * np.sqrt(24 * 365)  # Annualized
                else:
                    boot_sharpe = 0.0
                
                sharpe_samples.append(boot_sharpe)
                mean_samples.append(mean_ret)
            
            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            sharpe_ci_lower = np.percentile(sharpe_samples, alpha/2 * 100)
            sharpe_ci_upper = np.percentile(sharpe_samples, (1 - alpha/2) * 100)
            mean_ci_lower = np.percentile(mean_samples, alpha/2 * 100)
            mean_ci_upper = np.percentile(mean_samples, (1 - alpha/2) * 100)
            
            # Stricter reliability assessment for trading
            # Production rule: N >= 100 AND (Sharpe_CI_lower >= 0.5 OR (Sharpe_point >= 1.0 AND mean_CI_lower > 0))
            reliable = n_samples >= 100  # Minimum sample size for production
            if n_samples < 50:
                reliability_reason = f'N={n_samples} < 50 (🔴 UNRELIABLE - DO NOT TRADE)'
            elif n_samples < 100:
                reliability_reason = f'N={n_samples} < 100 (🟡 MARGINAL - PAPER TRADE ONLY)'
            else:
                reliability_reason = f'N={n_samples} ≥ 100 (🟢 SUFFICIENT SAMPLE SIZE)'
            
            return {
                'sharpe_ci_lower': float(sharpe_ci_lower),
                'sharpe_ci_upper': float(sharpe_ci_upper),
                'mean_return_ci_lower': float(mean_ci_lower),
                'mean_return_ci_upper': float(mean_ci_upper),
                'reliable': reliable,
                'reason': reliability_reason,
                'n_bootstrap': n_iterations
            }
            
        except Exception as e:
            logger.error(f"Bootstrap validation failed: {e}", exc_info=True)
            return {
                'sharpe_ci_lower': 0.0,
                'sharpe_ci_upper': 0.0,
                'mean_return_ci_lower': 0.0,
                'mean_return_ci_upper': 0.0,
                'reliable': False,
                'reason': f'Bootstrap failed: {str(e)}'
            }
    
    def _identify_tradeable_regimes(self, economic_metrics: Dict[int, Dict[str, Any]], 
                                     min_sharpe_ci_lower: float = 0.5,
                                     min_sharpe_point: float = 1.0) -> Dict[int, str]:
        """
        Identify which regimes are safe to trade based on production criteria.
        
        Production Rules:
        - N >= 100 (sufficient sample size)
        - Sharpe_CI_lower >= 0.5 OR (Sharpe_point >= 1.0 AND mean_return_CI_lower > 0)
        
        Args:
            economic_metrics: Economic performance per regime
            min_sharpe_ci_lower: Minimum lower bound of Sharpe CI
            min_sharpe_point: Minimum point estimate of Sharpe (alternative criterion)
            
        Returns:
            Dictionary mapping regime_id to trade status ('LONG', 'FLAT', 'NO_TRADE')
        """
        tradeable_regimes = {}
        
        tprint("\n" + "="*80, "INFO")
        tprint("🎯 PRODUCTION-READY REGIME EVALUATION", "INFO")
        tprint("="*80, "INFO")
        
        for regime_id, metrics in economic_metrics.items():
            n_samples = metrics.get('n_samples', 0)
            sharpe = metrics.get('sharpe', 0.0)
            bootstrap = metrics.get('bootstrap_ci', {})
            sharpe_ci_lower = bootstrap.get('sharpe_ci_lower', 0.0)
            mean_ci_lower = bootstrap.get('mean_return_ci_lower', 0.0)
            
            # Apply production rules
            sufficient_samples = n_samples >= 100
            meets_strict_sharpe = sharpe_ci_lower >= min_sharpe_ci_lower
            meets_alternative = (sharpe >= min_sharpe_point) and (mean_ci_lower > 0)
            
            # Decision logic
            if not sufficient_samples:
                status = 'NO_TRADE'
                reason = f'N={n_samples} < 100 (INSUFFICIENT)'
                emoji = '🔴'
            elif meets_strict_sharpe or meets_alternative:
                if sharpe > 0 and mean_ci_lower > 0:
                    status = 'LONG'
                    reason = f'Sharpe={sharpe:.2f}, CI_lower={sharpe_ci_lower:.2f} (PROFITABLE)'
                    emoji = '🟢'
                else:
                    status = 'FLAT'
                    reason = 'Meets criteria but no clear edge'
                    emoji = '🟡'
            else:
                status = 'NO_TRADE'
                reason = f'Sharpe_CI_lower={sharpe_ci_lower:.2f} < {min_sharpe_ci_lower:.2f} (UNRELIABLE)'
                emoji = '🟡'
            
            tradeable_regimes[regime_id] = status
            
            tprint(f"{emoji} Regime {regime_id}: {status:12s} - {reason}", 
                   "SUCCESS" if status == 'LONG' else "WARNING" if status == 'FLAT' else "INFO")
        
        tprint("="*80 + "\n", "INFO")
        
        return tradeable_regimes
    
    def _merge_tiny_regimes(self, regime_labels: np.ndarray, features_df: pd.DataFrame,
                            min_samples: int = 50, min_pct: float = 0.05) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Merge or drop tiny regimes that are statistically unreliable.
        
        Rule: Regimes with N < min_samples OR < min_pct% of total are merged
        into their nearest neighbor by feature similarity (emission mean).
        
        Args:
            regime_labels: Original regime assignments
            features_df: Feature matrix
            min_samples: Minimum samples required
            min_pct: Minimum percentage of total samples required
            
        Returns:
            (remapped_labels, regime_mapping) where regime_mapping shows old→new mappings
        """
        try:
            unique_regimes, counts = np.unique(regime_labels, return_counts=True)
            total_samples = len(regime_labels)
            
            regime_mapping = {}
            tiny_regimes = []
            valid_regimes = []
            
            tprint(f"🔍 Checking regime sizes against thresholds: min={min_samples} samples, {min_pct:.1%} of total", "INFO")
            
            for regime_id, count in zip(unique_regimes, counts):
                pct = count / total_samples
                
                if count < min_samples or pct < min_pct:
                    tiny_regimes.append(int(regime_id))
                    tprint(f"⚠️ Regime {regime_id}: N={count} ({pct:.1%}) - TINY (will merge)", "WARNING")
                else:
                    valid_regimes.append(int(regime_id))
                    tprint(f"✅ Regime {regime_id}: N={count} ({pct:.1%}) - VALID", "SUCCESS")
            
            if not tiny_regimes:
                tprint("✅ All regimes meet minimum size requirements", "SUCCESS")
                return regime_labels, {}
            
            # Calculate regime centroids and covariance matrices for Mahalanobis distance
            regime_centroids = {}
            regime_covariances = {}
            
            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                regime_features = features_df[regime_mask].values
                regime_centroids[int(regime_id)] = np.mean(regime_features, axis=0)
                
                # Calculate covariance with regularization to avoid singularity
                if len(regime_features) > 1:
                    cov = np.cov(regime_features, rowvar=False)
                    # Add small diagonal term for numerical stability
                    cov += np.eye(cov.shape[0]) * 1e-6
                    regime_covariances[int(regime_id)] = cov
                else:
                    # Use identity for single-sample regimes
                    regime_covariances[int(regime_id)] = np.eye(regime_features.shape[1])
            
            # Merge each tiny regime into nearest valid regime using Mahalanobis distance
            remapped_labels = regime_labels.copy()
            
            for tiny_regime in tiny_regimes:
                if len(valid_regimes) == 0:
                    tprint(f"⚠️ No valid regimes to merge into! Keeping {tiny_regime} as-is", "WARNING")
                    continue
                
                tiny_centroid = regime_centroids[tiny_regime]
                
                min_distance = np.inf
                nearest_regime = valid_regimes[0]
                
                # Find nearest valid regime by Mahalanobis distance
                for valid_regime in valid_regimes:
                    valid_centroid = regime_centroids[valid_regime]
                    valid_cov = regime_covariances[valid_regime]
                    
                    try:
                        # Compute Mahalanobis distance using the valid regime's covariance
                        # D = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
                        diff = tiny_centroid - valid_centroid
                        cov_inv = np.linalg.inv(valid_cov)
                        mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                    except np.linalg.LinAlgError:
                        # Fall back to Euclidean if covariance is singular
                        tprint(f"⚠️ Singular covariance for regime {valid_regime}, using Euclidean distance", "WARNING")
                        mahal_dist = np.linalg.norm(tiny_centroid - valid_centroid)
                    
                    if mahal_dist < min_distance:
                        min_distance = mahal_dist
                        nearest_regime = valid_regime
                
                # Remap tiny regime to nearest valid regime
                remapped_labels[regime_labels == tiny_regime] = nearest_regime
                regime_mapping[tiny_regime] = nearest_regime
                
                tprint(f"🔄 Merged Regime {tiny_regime} → Regime {nearest_regime} (Mahalanobis distance: {min_distance:.3f})", "INFO")
            
            # Update counts after merging
            unique_after, counts_after = np.unique(remapped_labels, return_counts=True)
            tprint(f"📊 After merging: {len(unique_after)} regimes (was {len(unique_regimes)})", "SUCCESS")
            for regime_id, count in zip(unique_after, counts_after):
                pct = count / total_samples
                tprint(f"   Regime {regime_id}: N={count} ({pct:.1%})", "INFO")
            
            return remapped_labels, regime_mapping
            
        except Exception as e:
            logger.error(f"Tiny regime merging failed: {e}", exc_info=True)
            return regime_labels, {}
    
    def _assess_quality(self, features_df: pd.DataFrame, regime_labels: np.ndarray, 
                       timestamps: Optional[pd.Series] = None) -> ClusterQualityMetrics:
        """Assess clustering quality using the unified quality assessor."""
        try:
            # Convert to the format expected by ClusterQualityAssessor
            if timestamps is not None:
                timestamps_array = timestamps.values if hasattr(timestamps, 'values') else timestamps
            else:
                timestamps_array = None
            
            # Use the unified quality assessor
            quality_metrics = self.quality_assessor.assess_quality(
                regime_labels=regime_labels,
                feature_data=features_df,
                timestamps=timestamps_array
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return minimal metrics
            return ClusterQualityMetrics(
                n_regimes=len(np.unique(regime_labels)),
                quality_score=0.5,
                silhouette_score=0.0,
                noise_ratio=0.0
            )

def create_hmm_regime_discovery_step(
    n_states: int = 4,
    correlation_threshold: float = 0.85,
    random_state: int = 42,
    covariance_type: str = 'diag',  # Default: 'diag' for regularization (prevents overfitting)
    n_iter: int = 100,
    min_regime_pct: float = 0.05,
    min_regime_samples: int = 50,
    merge_tiny_regimes: bool = True,
    bootstrap_iterations: int = 500,
    confidence_level: float = 0.95,
    optimization_goals: Optional[ClusteringOptimizationGoals] = None,
    optimization_targets: Optional[OptimizationTargets] = None
) -> HMMRegimeDiscoveryStep:
    """
    Create an HMM regime discovery step with specified parameters.
    
    Args:
        n_states: Number of hidden states (default: 4 for interpretability)
        correlation_threshold: Threshold for feature correlation removal
        random_state: Random seed for reproducibility
        covariance_type: HMM covariance type (default: 'diag' for regularization)
                        Options: 'diag' (regularized), 'tied' (shared), 'full' (flexible), 'spherical' (strict)
        n_iter: Maximum iterations for HMM fitting
        min_regime_pct: Minimum percentage of samples per regime (default: 0.05 = 5%)
        min_regime_samples: Minimum samples per regime (default: 50)
        merge_tiny_regimes: Whether to merge tiny regimes (default: True)
        bootstrap_iterations: Bootstrap iterations for CI (default: 500)
        confidence_level: Confidence level for CIs (default: 0.95)
        optimization_goals: Optimization goals configuration
        optimization_targets: Optimization targets configuration
        
    Returns:
        HMMRegimeDiscoveryStep instance
    """
    kwargs = {
        'n_states': n_states,
        'correlation_threshold': correlation_threshold,
        'random_state': random_state,
        'covariance_type': covariance_type,
        'n_iter': n_iter,
        'min_regime_pct': min_regime_pct,
        'min_regime_samples': min_regime_samples,
        'merge_tiny_regimes': merge_tiny_regimes,
        'bootstrap_iterations': bootstrap_iterations,
        'confidence_level': confidence_level
    }
    
    if optimization_goals is not None:
        kwargs['optimization_goals'] = optimization_goals
    if optimization_targets is not None:
        kwargs['optimization_targets'] = optimization_targets
    
    return HMMRegimeDiscoveryStep(**kwargs)
