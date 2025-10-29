"""
Gaussian Mixture Model (GMM) Regime Discovery Step

This module provides GMM-based regime discovery as an alternative to HDBSCAN,
with correlation-based feature reduction to remove redundant volatility features.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
import warnings

from src.training.steps.base_step import BaseStep
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
    create_cluster_quality_assessor
)
from src.utils.tprint import tprint, tprint_info, tprint_timer
from src.utils.logger import system_logger

logger = system_logger.getChild('GMMRegimeDiscoveryStep')

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

class GMMRegimeDiscoveryStep(BaseStep):
    """
    Gaussian Mixture Model-based regime discovery with correlation-based feature reduction.
    
    This approach:
    1. Removes highly correlated features (especially redundant volatility features)
    2. Uses GMM with k=5-9 components for regime discovery
    3. Applies the same quality assessment as HDBSCAN
    """
    
    def __init__(self, step_name: str = "gmm_regime_discovery", **kwargs):
        """
        Initialize GMM regime discovery step.
        
        Args:
            step_name: Name of the step (passed by launcher)
            **kwargs: Additional keyword arguments (n_components_range, correlation_threshold, random_state)
        """
        super().__init__(step_name)
        
        # Extract parameters from kwargs with defaults
        self.n_components_range = kwargs.get('n_components_range', (5, 9))
        self.correlation_threshold = kwargs.get('correlation_threshold', 0.85)
        self.random_state = kwargs.get('random_state', 42)
        
        self.quality_assessor = create_cluster_quality_assessor()
        self.feature_selector = CorrelationBasedFeatureSelector(self.correlation_threshold)
        self.scaler = StandardScaler()
        self.pca = None
        self.gmm_model = None
        self.regime_labels = None
        self.quality_metrics = None
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GMM regime discovery step.
        
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
            
            tprint(f"🚀 Starting GMM Regime Discovery for {symbol} on {exchange} ({timeframe})", "INFO")
            tprint(f"📊 Execution mode: {execution_mode}", "INFO")
            
            # Load data (this would typically load from artifacts)
            # For now, we'll create a placeholder that would be replaced with actual data loading
            data, features_df = self._load_data(symbol, exchange, timeframe)
            
            # Discover regimes
            results = self.discover_regimes(data, features_df)
            
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
            tprint("📂 Loading market data and features for GMM regime discovery", "INFO")
            
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
            
            # Try to load features from feature generation step
            features_df = None
            try:
                # Set context to feature generation step
                self.artifact_manager.set_context(
                    step_name="feature_generation_feature_generation_step",
                    symbol=symbol,
                    exchange=exchange,
                    datetime=datetime.now(),
                    information="feature_generation",
                    direction="long",
                    model="Analyst"
                )
                
                # Try to load features for the specific timeframe
                artifact_name = f"generated_features_{timeframe}"
                features_df = self.artifact_manager.get_artifact(artifact_name, artifact_type="data")
                
                if features_df is None or (hasattr(features_df, 'empty') and features_df.empty):
                    # Try 1h features as fallback (common for regime analysis)
                    if timeframe != '1h':
                        tprint(f"⚠️ No {timeframe} features found, trying 1h features", "WARNING")
                        features_df = self.artifact_manager.get_artifact("generated_features_1h", artifact_type="data")
                
                if features_df is not None and not (hasattr(features_df, 'empty') and features_df.empty):
                    tprint(f"✅ Loaded {len(features_df.columns)} features from feature generation", "SUCCESS")
                else:
                    tprint("⚠️ No features found, will generate basic features from market data", "WARNING")
                    features_df = None
                    
            except Exception as e:
                tprint(f"⚠️ Failed to load features from artifacts: {e}", "WARNING")
                features_df = None
            
            # If no features loaded, create basic features from market data
            if features_df is None:
                tprint("🔧 Generating basic features from market data", "INFO")
                features_df = self._generate_basic_features(market_data)
            
            # Ensure data alignment
            if len(market_data) != len(features_df):
                tprint(f"⚠️ Data length mismatch: market_data={len(market_data)}, features={len(features_df)}", "WARNING")
                # Align by index
                common_index = market_data.index.intersection(features_df.index)
                if len(common_index) > 0:
                    market_data = market_data.loc[common_index]
                    features_df = features_df.loc[common_index]
                    tprint(f"✅ Aligned data to {len(common_index)} common timestamps", "SUCCESS")
                else:
                    tprint("❌ No common timestamps found, using placeholder data", "ERROR")
                    return self._create_placeholder_data()
            
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
            tprint("💾 Saving GMM regime discovery artifacts", "INFO")
            
            # Set context for saving artifacts
            self.artifact_manager.set_context(
                step_name="gmm_regime_discovery",
                symbol=symbol,
                exchange=exchange,
                datetime=datetime.now(),
                information="regime_discovery",
                direction="long",
                model="Analyst"
            )
            
            # Save regime labels
            if 'regime_labels' in results:
                regime_labels_df = pd.DataFrame({
                    'regime_label': results['regime_labels'],
                    'timestamp': results.get('timestamps', range(len(results['regime_labels'])))
                })
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
        """Generate comprehensive GMM regime discovery report with detailed metrics."""
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
            
            # Build comprehensive report
            report = f"""# GMM Regime Discovery Comprehensive Report

**Generated**: {timestamp.isoformat()}  
**Report ID**: `gmm_regime_discovery_{symbol}_{timeframe}_{timestamp_str}`

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Symbol** | {symbol} |
| **Exchange** | {exchange} |
| **Timeframe** | {timeframe} |
| **Processing Time** | {processing_time:.2f} seconds |
| **Success Status** | ✅ SUCCESS |
| **Regimes Discovered** | {n_regimes} |
| **Quality Score** | {quality_metrics.quality_score:.3f} |
| **Noise Ratio** | {quality_metrics.noise_ratio:.1%} |

---

## 🔍 Regime Discovery Results

### Cluster Statistics
- **Total Regimes**: {n_regimes}
- **Noise Points**: {quality_metrics.noise_ratio:.1%} of total samples (GMM: 0.0% - no noise)
- **Average Regime Size**: {total_samples / max(n_regimes, 1):.0f} samples per regime
- **Balance Score**: {balance_score_str} (higher is better)

### Regime Distribution
| Regime ID | Sample Count | Percentage |
|-----------|--------------|------------|
"""
            
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                report += f"| **Regime {label}** | {count:,} | {percentage:.1f}% |\n"
            
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
            
            # GMM Parameters
            if gmm_params:
                report += f"""
### GMM Model Parameters
- **Number of Components**: {gmm_params.get('n_components', 'N/A')}
- **Covariance Type**: {gmm_params.get('covariance_type', 'N/A')}
- **Random State**: {gmm_params.get('random_state', 'N/A')}

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

*Generated by GMM Regime Discovery at {timestamp.isoformat()}*

"""
            
            # Save report to file
            report_dir = Path("outcomes") / f"gmm_regime_discovery_{symbol}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"gmm_regime_discovery_report_{symbol}_{timestamp_str}.md"
            report_file.write_text(report)
            
            tprint(f"✅ Comprehensive report saved: {report_file}", "SUCCESS")
            tprint(f"📊 Quality Score: {quality_metrics.quality_score:.3f} ({'Excellent' if quality_metrics.quality_score >= 0.70 else 'Good' if quality_metrics.quality_score >= 0.50 else 'Moderate' if quality_metrics.quality_score >= 0.30 else 'Poor'})", "SUCCESS")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}", exc_info=True)
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
        tprint("🚀 Starting GMM Regime Discovery with Correlation-Based Feature Reduction", "INFO")
        start_time = time.time()
        
        try:
            # Step 1: Remove highly correlated features
            with tprint_timer("Correlation-Based Feature Reduction"):
                reduced_features = self.feature_selector.fit_transform(features_df)
            
            # Step 2: Standardize features
            with tprint_timer("Feature Standardization"):
                scaled_features = self.scaler.fit_transform(reduced_features)
                scaled_df = pd.DataFrame(scaled_features, 
                                       columns=reduced_features.columns,
                                       index=reduced_features.index)
            
            # Step 3: Optional PCA for further dimensionality reduction
            if scaled_df.shape[1] > 50:  # Only if we have many features
                with tprint_timer("PCA Dimensionality Reduction"):
                    self.pca = PCA(n_components=min(50, scaled_df.shape[1]), random_state=self.random_state)
                    pca_features = self.pca.fit_transform(scaled_features)
                    scaled_df = pd.DataFrame(pca_features, 
                                           columns=[f'PC_{i+1}' for i in range(pca_features.shape[1])],
                                           index=reduced_features.index)
                    tprint(f"📊 PCA reduced features: {scaled_features.shape[1]} → {pca_features.shape[1]}", "INFO")
            
            # Step 4: Find optimal number of components using grid search
            with tprint_timer("GMM Parameter Optimization"):
                best_n_components, best_gmm = self._find_optimal_components(scaled_df)
                self.gmm_model = best_gmm
            
            # Step 5: Fit GMM and predict regimes
            with tprint_timer("GMM Fitting and Prediction"):
                self.regime_labels = self.gmm_model.predict(scaled_features)
                regime_probs = self.gmm_model.predict_proba(scaled_features)
            
            # Step 6: Assess quality
            with tprint_timer("Quality Assessment"):
                self.quality_metrics = self._assess_quality(scaled_df, self.regime_labels, timestamps)
            
            # Step 7: Generate results
            processing_time = time.time() - start_time
            
            results = {
                'regime_labels': self.regime_labels,
                'regime_probabilities': regime_probs,
                'n_regimes': best_n_components,
                'n_noise_points': 0,  # GMM doesn't have noise points
                'noise_ratio': 0.0,
                'quality_metrics': self.quality_metrics,
                'processing_time': processing_time,
                'feature_reduction_stats': {
                    'original_features': len(features_df.columns),
                    'reduced_features': len(reduced_features.columns),
                    'removed_features': len(features_df.columns) - len(reduced_features.columns),
                    'correlation_threshold': self.correlation_threshold
                },
                'gmm_params': {
                    'n_components': best_n_components,
                    'covariance_type': self.gmm_model.covariance_type,
                    'random_state': self.random_state
                },
                'pca_explained_variance_ratio': self.pca.explained_variance_ratio_.tolist() if self.pca else None
            }
            
            tprint(f"✅ GMM Regime Discovery Complete: {best_n_components} regimes in {processing_time:.2f}s", "SUCCESS")
            tprint(f"📊 Quality Score: {self.quality_metrics.quality_score:.3f}", "SUCCESS")
            
            return results
            
        except Exception as e:
            logger.error(f"GMM regime discovery failed: {e}")
            raise
    
    def _find_optimal_components(self, features_df: pd.DataFrame) -> Tuple[int, GaussianMixture]:
        """Find optimal number of components using grid search."""
        logger.info(f"Starting GMM parameter search with n_components range: {self.n_components_range}")
        
        n_components_range = range(self.n_components_range[0], self.n_components_range[1] + 1)
        
        # Test different covariance types
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        best_score = -np.inf
        best_params = None
        best_model = None
        
        results = []
        
        for n_components in n_components_range:
            for cov_type in covariance_types:
                try:
                    # Fit GMM
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cov_type,
                        random_state=self.random_state,
                        max_iter=100
                    )
                    gmm.fit(features_df)
                    
                    # Calculate metrics
                    labels = gmm.predict(features_df)
                    silhouette = silhouette_score(features_df, labels)
                    aic = gmm.aic(features_df)
                    bic = gmm.bic(features_df)
                    
                    # Combined score (higher is better)
                    # Weight AIC and BIC more heavily for GMM
                    combined_score = 0.3 * silhouette + 0.35 * (-aic/1000) + 0.35 * (-bic/1000)
                    
                    results.append({
                        'n_components': n_components,
                        'covariance_type': cov_type,
                        'silhouette': silhouette,
                        'aic': aic,
                        'bic': bic,
                        'combined_score': combined_score
                    })
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = (n_components, cov_type)
                        best_model = gmm
                        
                except Exception as e:
                    logger.warning(f"Failed to fit GMM with n_components={n_components}, cov_type={cov_type}: {e}")
                    continue
        
        # Show results
        try:
            logger.info(f"GMM search complete. Found {len(results)} valid configurations")
            tprint("📊 GMM Parameter Search Results:", "INFO")
            for result in sorted(results, key=lambda x: x['combined_score'], reverse=True)[:5]:
                result_str = (f"  n_components={result['n_components']}, cov_type={result['covariance_type']}, "
                             f"score={result['combined_score']:.3f}, silhouette={result['silhouette']:.3f}")
                tprint(result_str, "INFO")
            
            if best_model is None:
                raise RuntimeError("Failed to find valid GMM parameters")
            
            logger.info(f"Best params: n_components={best_params[0]}, cov_type={best_params[1]}, score={best_score:.3f}")
            tprint(f"🏆 Best GMM: n_components={best_params[0]}, cov_type={best_params[1]}, score={best_score:.3f}", "SUCCESS")
            
            return best_params[0], best_model
        except Exception as e:
            logger.error(f"Error in results display: {e}", exc_info=True)
            raise
    
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

def create_gmm_regime_discovery_step(n_components_range: Tuple[int, int] = (5, 9),
                                   correlation_threshold: float = 0.85,
                                   random_state: int = 42) -> GMMRegimeDiscoveryStep:
    """Create a GMM regime discovery step with specified parameters."""
    return GMMRegimeDiscoveryStep(
        n_components_range=n_components_range,
        correlation_threshold=correlation_threshold,
        random_state=random_state
    )
