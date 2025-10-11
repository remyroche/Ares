"""
VectorBT Optimized Clustering Analysis

This module enhances the clustering analysis framework with VectorBT capabilities:
- VectorBT technical indicators for market regime detection
- Enhanced clustering features using VectorBT
- Signal-based regime validation
- Portfolio-level regime analysis
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Suppress VectorBT warnings
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)

class VectorBTClusteringOptimizer:
    """
    VectorBT-optimized clustering analysis framework.
    
    This class enhances the existing clustering analysis with VectorBT capabilities:
    - VectorBT technical indicators for market regime detection
    - Enhanced clustering features using VectorBT
    - Signal-based regime validation
    - Portfolio-level regime analysis
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize VectorBT clustering optimizer.
        
        Args:
            data: OHLCV data
        """
        self.data = data.copy()
        
        # Ensure proper index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # VectorBT configuration
        vbt.settings.set_theme("dark")
        
        logger.info("✅ VectorBT clustering optimizer initialized")
    
    def generate_clustering_features(self) -> pd.DataFrame:
        """
        Generate comprehensive features for clustering using VectorBT.
        
        Returns:
            DataFrame of clustering features
        """
        logger.info("🔧 Generating VectorBT clustering features...")
        
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        features = {}
        
        try:
            # Price-based features
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['volatility'] = close.rolling(20).std()
            features['price_range'] = high - low
            features['price_position'] = (close - low) / (high - low)
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                sma = vbt.MA.run(close, period).ma
                features[f'sma_{period}'] = sma
                features[f'price_vs_sma_{period}'] = (close / sma - 1)
                features[f'sma_slope_{period}'] = sma.pct_change()
            
            # MACD features
            macd = vbt.MACD.run(close)
            features['macd'] = macd.macd
            features['macd_signal'] = macd.signal
            features['macd_histogram'] = macd.histogram
            features['macd_divergence'] = macd.histogram.diff()
            
            # RSI features
            rsi = vbt.RSI.run(close).rsi
            features['rsi'] = rsi
            features['rsi_oversold'] = (rsi < 30).astype(int)
            features['rsi_overbought'] = (rsi > 70).astype(int)
            features['rsi_momentum'] = rsi.diff()
            
            # Bollinger Bands features
            bb = vbt.BBANDS.run(close)
            features['bb_upper'] = bb.upper
            features['bb_middle'] = bb.middle
            features['bb_lower'] = bb.lower
            features['bb_width'] = bb.width
            features['bb_percent'] = bb.percent
            features['bb_squeeze'] = (bb.width < bb.width.rolling(20).mean()).astype(int)
            
            # Stochastic features
            stoch = vbt.STOCH.run(high, low, close)
            features['stoch_k'] = stoch.k
            features['stoch_d'] = stoch.d
            features['stoch_oversold'] = ((stoch.k < 20) & (stoch.d < 20)).astype(int)
            features['stoch_overbought'] = ((stoch.k > 80) & (stoch.d > 80)).astype(int)
            
            # ATR features
            atr = vbt.ATR.run(high, low, close).atr
            features['atr'] = atr
            features['atr_percent'] = atr / close
            features['atr_ratio'] = atr / atr.rolling(20).mean()
            
            # ADX features
            adx = vbt.ADX.run(high, low, close)
            features['adx'] = adx.adx
            features['adx_pos'] = adx.plus_di
            features['adx_neg'] = adx.minus_di
            features['adx_trend_strength'] = (adx.adx > 25).astype(int)
            
            # Volume features
            features['volume_sma'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / features['volume_sma']
            features['volume_spike'] = (volume > features['volume_sma'] * 2).astype(int)
            
            # OBV features
            obv = vbt.OBV.run(close, volume).obv
            features['obv'] = obv
            features['obv_sma'] = obv.rolling(20).mean()
            features['obv_momentum'] = obv.diff()
            
            # AD features
            ad = vbt.AD.run(high, low, close, volume).ad
            features['ad'] = ad
            features['ad_sma'] = ad.rolling(20).mean()
            features['ad_momentum'] = ad.diff()
            
            # CMF features
            cmf = vbt.CMF.run(high, low, close, volume).cmf
            features['cmf'] = cmf
            features['cmf_positive'] = (cmf > 0).astype(int)
            
            # Momentum features
            for period in [5, 10, 20]:
                momentum = close / close.shift(period) - 1
                features[f'momentum_{period}'] = momentum
                features[f'roc_{period}'] = close.pct_change(period)
            
            # Trend features
            features['trend_sma_20_50'] = (features['sma_20'] > features['sma_50']).astype(int)
            features['trend_sma_50_100'] = (features['sma_50'] > features['sma_100']).astype(int)
            features['trend_consensus'] = (features['trend_sma_20_50'] + features['trend_sma_50_100']) / 2
            
            # Volatility regime features
            vol_ma = features['volatility'].rolling(50).mean()
            features['vol_regime_high'] = (features['volatility'] > vol_ma * 1.2).astype(int)
            features['vol_regime_low'] = (features['volatility'] < vol_ma * 0.8).astype(int)
            features['vol_regime_normal'] = (
                (features['volatility'] >= vol_ma * 0.8) & 
                (features['volatility'] <= vol_ma * 1.2)
            ).astype(int)
            
            # Market structure features
            features['higher_highs'] = (high > high.rolling(20).max().shift(1)).astype(int)
            features['lower_lows'] = (low < low.rolling(20).min().shift(1)).astype(int)
            features['breakout_high'] = (close > high.rolling(20).max().shift(1)).astype(int)
            features['breakdown_low'] = (close < low.rolling(20).min().shift(1)).astype(int)
            
            # Time-based features
            features['hour'] = self.data.index.hour
            features['day_of_week'] = self.data.index.dayofweek
            features['month'] = self.data.index.month
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
                features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
                features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
                features[f'returns_kurt_{window}'] = features['returns'].rolling(window).kurt()
            
            # Create DataFrame
            feature_df = pd.DataFrame(features)
            
            # Remove infinite values and fill NaN
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.info(f"✅ Generated {len(feature_df.columns)} clustering features")
            
        except Exception as e:
            logger.error(f"Error generating clustering features: {e}")
            return pd.DataFrame()
        
        return feature_df
    
    def perform_clustering(self, features: pd.DataFrame, 
                          method: str = 'kmeans', 
                          n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform clustering analysis using VectorBT-enhanced features.
        
        Args:
            features: Feature DataFrame
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative')
            n_clusters: Number of clusters
            
        Returns:
            Clustering results
        """
        logger.info(f"🔍 Performing {method} clustering with {n_clusters} clusters...")
        
        # Prepare features
        X = features.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        try:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X_scaled)
                
            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(X_scaled)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
            elif method == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(X_scaled)
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Calculate clustering metrics
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
            else:
                silhouette_avg = 0
                calinski_harabasz = 0
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(features, cluster_labels)
            
            # Generate regime signals
            regime_signals = self._generate_regime_signals(cluster_labels)
            
            results = {
                'cluster_labels': cluster_labels,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'cluster_analysis': cluster_analysis,
                'regime_signals': regime_signals,
                'method': method,
                'feature_names': list(X.columns)
            }
            
            logger.info(f"✅ Clustering completed with {n_clusters} clusters")
            logger.info(f"   Silhouette score: {silhouette_avg:.3f}")
            logger.info(f"   Calinski-Harabasz score: {calinski_harabasz:.3f}")
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {}
        
        return results
    
    def _analyze_clusters(self, features: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        analysis = {}
        
        try:
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise in DBSCAN
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_features = features[cluster_mask]
                
                # Basic statistics
                analysis[f'cluster_{cluster_id}'] = {
                    'size': cluster_mask.sum(),
                    'percentage': cluster_mask.sum() / len(cluster_labels) * 100,
                    'mean_returns': cluster_features['returns'].mean() if 'returns' in cluster_features.columns else 0,
                    'volatility': cluster_features['volatility'].mean() if 'volatility' in cluster_features.columns else 0,
                    'rsi_mean': cluster_features['rsi'].mean() if 'rsi' in cluster_features.columns else 0,
                    'volume_ratio_mean': cluster_features['volume_ratio'].mean() if 'volume_ratio' in cluster_features.columns else 0,
                    'trend_consensus_mean': cluster_features['trend_consensus'].mean() if 'trend_consensus' in cluster_features.columns else 0
                }
            
            # Overall analysis
            analysis['overall'] = {
                'total_clusters': len(unique_clusters),
                'noise_points': (cluster_labels == -1).sum() if -1 in cluster_labels else 0,
                'cluster_balance': self._calculate_cluster_balance(cluster_labels)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing clusters: {e}")
        
        return analysis
    
    def _calculate_cluster_balance(self, cluster_labels: np.ndarray) -> float:
        """Calculate cluster balance (lower is more balanced)."""
        unique, counts = np.unique(cluster_labels, return_counts=True)
        if len(unique) <= 1:
            return 0.0
        
        # Calculate coefficient of variation
        mean_size = np.mean(counts)
        std_size = np.std(counts)
        return std_size / mean_size if mean_size > 0 else 0.0
    
    def _generate_regime_signals(self, cluster_labels: np.ndarray) -> Dict[str, pd.Series]:
        """Generate regime-based trading signals."""
        signals = {}
        
        try:
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise
                    continue
                
                # Binary signal for each regime
                signals[f'regime_{cluster_id}'] = (cluster_labels == cluster_id).astype(int)
            
            # Transition signals
            for i in range(len(unique_clusters) - 1):
                for j in range(i + 1, len(unique_clusters)):
                    if unique_clusters[i] == -1 or unique_clusters[j] == -1:
                        continue
                    
                    transition_name = f'transition_{unique_clusters[i]}_to_{unique_clusters[j]}'
                    signals[transition_name] = (
                        (cluster_labels == unique_clusters[j]) & 
                        (np.roll(cluster_labels, 1) == unique_clusters[i])
                    ).astype(int)
            
        except Exception as e:
            logger.error(f"Error generating regime signals: {e}")
        
        return signals
    
    def validate_regimes(self, cluster_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Validate regimes using VectorBT backtesting.
        
        Args:
            cluster_results: Clustering results
            
        Returns:
            Regime validation results
        """
        logger.info("🔬 Validating regimes with VectorBT backtesting...")
        
        close = self.data['close']
        regime_signals = cluster_results.get('regime_signals', {})
        validation_results = {}
        
        try:
            for signal_name, signal in regime_signals.items():
                if signal.sum() == 0:
                    continue
                
                # Create entries and exits
                entries = signal == 1
                exits = signal.shift(1) == 1
                
                # Run backtest
                pf = vbt.Portfolio.from_signals(
                    close,
                    entries=entries,
                    exits=exits,
                    init_cash=10000,
                    fees=0.001,
                    freq='1H'
                )
                
                # Extract validation metrics
                validation_results[signal_name] = {
                    'total_return': pf.total_return(),
                    'sharpe_ratio': pf.sharpe_ratio(),
                    'max_drawdown': pf.max_drawdown(),
                    'win_rate': pf.trades.win_rate(),
                    'profit_factor': pf.trades.profit_factor(),
                    'total_trades': pf.trades.count(),
                    'avg_trade_duration': pf.trades.duration.mean(),
                    'regime_frequency': signal.sum() / len(signal),
                    'regime_persistence': self._calculate_regime_persistence(signal)
                }
            
            logger.info(f"✅ Validated {len(validation_results)} regimes")
            
        except Exception as e:
            logger.error(f"Error validating regimes: {e}")
            return {}
        
        return validation_results
    
    def _calculate_regime_persistence(self, signal: pd.Series) -> float:
        """Calculate regime persistence (average duration)."""
        try:
            # Find regime changes
            changes = signal.diff().fillna(0)
            regime_starts = (changes == 1).cumsum()
            
            # Calculate average duration
            durations = []
            for regime_id in regime_starts.unique():
                if regime_id == 0:
                    continue
                regime_mask = regime_starts == regime_id
                duration = regime_mask.sum()
                if duration > 0:
                    durations.append(duration)
            
            return np.mean(durations) if durations else 0.0
        except:
            return 0.0
    
    def optimize_clustering_parameters(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize clustering parameters using multiple methods.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Optimization results
        """
        logger.info("🎯 Optimizing clustering parameters...")
        
        X = features.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        optimization_results = {}
        
        try:
            # Test different numbers of clusters for KMeans
            kmeans_results = []
            for n_clusters in range(2, 11):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
                    
                    kmeans_results.append({
                        'n_clusters': n_clusters,
                        'silhouette_score': silhouette_avg,
                        'calinski_harabasz_score': calinski_harabasz,
                        'inertia': kmeans.inertia_
                    })
            
            # Test different parameters for DBSCAN
            dbscan_results = []
            for eps in [0.1, 0.3, 0.5, 0.7, 1.0]:
                for min_samples in [3, 5, 10, 15]:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(X_scaled)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    if n_clusters > 1:
                        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
                        
                        dbscan_results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'silhouette_score': silhouette_avg,
                            'calinski_harabasz_score': calinski_harabasz,
                            'noise_points': (cluster_labels == -1).sum()
                        })
            
            # Find best parameters
            best_kmeans = max(kmeans_results, key=lambda x: x['silhouette_score']) if kmeans_results else None
            best_dbscan = max(dbscan_results, key=lambda x: x['silhouette_score']) if dbscan_results else None
            
            optimization_results = {
                'kmeans_results': kmeans_results,
                'dbscan_results': dbscan_results,
                'best_kmeans': best_kmeans,
                'best_dbscan': best_dbscan,
                'recommended_method': 'kmeans' if (best_kmeans and best_dbscan and 
                                                 best_kmeans['silhouette_score'] > best_dbscan['silhouette_score']) else 'dbscan'
            }
            
            logger.info(f"✅ Parameter optimization completed")
            if best_kmeans:
                logger.info(f"   Best KMeans: {best_kmeans['n_clusters']} clusters, silhouette: {best_kmeans['silhouette_score']:.3f}")
            if best_dbscan:
                logger.info(f"   Best DBSCAN: eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}, silhouette: {best_dbscan['silhouette_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {}
        
        return optimization_results
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive VectorBT clustering analysis.
        
        Returns:
            Complete analysis results
        """
        logger.info("🔬 Running comprehensive VectorBT clustering analysis...")
        
        # Generate features
        features = self.generate_clustering_features()
        
        if features.empty:
            logger.error("No features generated")
            return {}
        
        # Optimize parameters
        optimization_results = self.optimize_clustering_parameters(features)
        
        # Perform clustering with best parameters
        best_method = optimization_results.get('recommended_method', 'kmeans')
        if best_method == 'kmeans' and optimization_results.get('best_kmeans'):
            n_clusters = optimization_results['best_kmeans']['n_clusters']
        elif best_method == 'dbscan' and optimization_results.get('best_dbscan'):
            n_clusters = 3  # Default for DBSCAN
        else:
            n_clusters = 3
        
        cluster_results = self.perform_clustering(features, best_method, n_clusters)
        
        # Validate regimes
        validation_results = self.validate_regimes(cluster_results)
        
        # Generate summary
        summary = self._generate_clustering_summary(cluster_results, validation_results, optimization_results)
        
        results = {
            'features': features,
            'optimization_results': optimization_results,
            'cluster_results': cluster_results,
            'validation_results': validation_results,
            'summary': summary,
            'data_info': {
                'start_date': self.data.index.min(),
                'end_date': self.data.index.max(),
                'total_periods': len(self.data),
                'feature_count': len(features.columns)
            }
        }
        
        logger.info("✅ Comprehensive VectorBT clustering analysis completed")
        return results
    
    def _generate_clustering_summary(self, cluster_results: Dict[str, Any], 
                                   validation_results: Dict[str, Any],
                                   optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clustering analysis summary."""
        summary = {
            'total_clusters': cluster_results.get('n_clusters', 0),
            'silhouette_score': cluster_results.get('silhouette_score', 0),
            'calinski_harabasz_score': cluster_results.get('calinski_harabasz_score', 0),
            'profitable_regimes': len([r for r in validation_results.values() if r['total_return'] > 0]),
            'best_regime': None,
            'optimization_summary': {}
        }
        
        # Find best regime
        if validation_results:
            best_regime = max(validation_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            summary['best_regime'] = {
                'name': best_regime[0],
                'sharpe_ratio': best_regime[1]['sharpe_ratio'],
                'total_return': best_regime[1]['total_return'],
                'win_rate': best_regime[1]['win_rate']
            }
        
        # Optimization summary
        if optimization_results:
            summary['optimization_summary'] = {
                'recommended_method': optimization_results.get('recommended_method', 'unknown'),
                'best_kmeans_clusters': optimization_results.get('best_kmeans', {}).get('n_clusters', 0),
                'best_dbscan_params': optimization_results.get('best_dbscan', {}),
                'total_methods_tested': len(optimization_results.get('kmeans_results', [])) + len(optimization_results.get('dbscan_results', []))
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = "vectorbt_clustering.json"):
        """Save analysis results to file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if key == 'features':
                # Convert DataFrame to dict
                serializable_results[key] = value.to_dict('records')
            elif key == 'cluster_results' and 'cluster_labels' in value:
                # Convert numpy array to list
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        import json
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        sample_data.loc[sample_data.index[i], 'high'] = max(sample_data.iloc[i][['open', 'high', 'low', 'close']])
        sample_data.loc[sample_data.index[i], 'low'] = min(sample_data.iloc[i][['open', 'high', 'low', 'close']])
    
    # Run VectorBT clustering analysis
    optimizer = VectorBTClusteringOptimizer(sample_data)
    results = optimizer.run_comprehensive_analysis()
    
    # Save results
    optimizer.save_results(results)
    
    print("✅ VectorBT clustering analysis completed!")
    print(f"Total clusters: {results['summary']['total_clusters']}")
    print(f"Silhouette score: {results['summary']['silhouette_score']:.3f}")
    print(f"Profitable regimes: {results['summary']['profitable_regimes']}")
    print(f"Best regime: {results['summary']['best_regime']}")