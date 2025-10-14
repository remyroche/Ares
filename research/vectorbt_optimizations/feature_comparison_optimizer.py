"""
VectorBT Optimized Feature Comparison

This module enhances the feature comparison framework with VectorBT capabilities:
- Advanced technical indicators as features
- VectorBT-based feature engineering
- Enhanced performance evaluation
- Signal-based feature validation
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import warnings

# Suppress VectorBT warnings
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)

class VectorBTFeatureOptimizer:
    """
    VectorBT-optimized feature comparison framework.
    
    This class enhances the existing feature comparison with VectorBT's capabilities:
    - Advanced technical indicators as features
    - VectorBT-based feature engineering
    - Signal-based feature validation
    - Enhanced performance metrics
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'returns'):
        """
        Initialize VectorBT feature optimizer.
        
        Args:
            data: OHLCV data
            target_col: Target column name
        """
        self.data = data.copy()
        self.target_col = target_col
        
        # Ensure proper index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # VectorBT configuration
        vbt.settings.set_theme("dark")
        
        logger.info("✅ VectorBT feature optimizer initialized")
    
    def generate_vectorbt_features(self) -> Dict[str, pd.Series]:
        """
        Generate comprehensive features using VectorBT technical indicators.
        
        Returns:
            Dictionary of VectorBT-based features
        """
        logger.info("🔧 Generating VectorBT technical features...")
        
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        features = {}
        
        try:
            # Price-based features
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['price_change'] = close.diff()
            features['price_range'] = high - low
            features['price_position'] = (close - low) / (high - low)
            
            # Moving averages
            for period in [5, 10, 20, 50, 100]:
                features[f'sma_{period}'] = vbt.MA.run(close, period).ma
                features[f'ema_{period}'] = vbt.MA.run(close, period, short_win=period).ma
                features[f'price_vs_sma_{period}'] = (close / features[f'sma_{period}'] - 1)
                features[f'price_vs_ema_{period}'] = (close / features[f'ema_{period}'] - 1)
            
            # MACD features
            macd = vbt.MACD.run(close)
            features['macd'] = macd.macd
            features['macd_signal'] = macd.signal
            features['macd_histogram'] = macd.histogram
            features['macd_divergence'] = macd.histogram.diff()
            
            # RSI features
            rsi = vbt.RSI.run(close)
            features['rsi'] = rsi.rsi
            features['rsi_oversold'] = (rsi.rsi < 30).astype(int)
            features['rsi_overbought'] = (rsi.rsi > 70).astype(int)
            features['rsi_momentum'] = rsi.rsi.diff()
            
            # Bollinger Bands features
            bb = vbt.BBANDS.run(close)
            features['bb_upper'] = bb.upper
            features['bb_middle'] = bb.middle
            features['bb_lower'] = bb.lower
            features['bb_width'] = bb.width
            features['bb_percent'] = bb.percent
            features['bb_squeeze'] = (bb.width < bb.width.rolling(20).mean()).astype(int)
            features['bb_breakout_upper'] = (close > bb.upper).astype(int)
            features['bb_breakout_lower'] = (close < bb.lower).astype(int)
            
            # Stochastic features
            stoch = vbt.STOCH.run(high, low, close)
            features['stoch_k'] = stoch.k
            features['stoch_d'] = stoch.d
            features['stoch_oversold'] = ((stoch.k < 20) & (stoch.d < 20)).astype(int)
            features['stoch_overbought'] = ((stoch.k > 80) & (stoch.d > 80)).astype(int)
            features['stoch_crossover'] = ((stoch.k > stoch.d) & (stoch.k.shift(1) <= stoch.d.shift(1))).astype(int)
            
            # Williams %R
            willr = vbt.WILLR.run(high, low, close)
            features['williams_r'] = willr.willr
            features['williams_oversold'] = (willr.willr < -80).astype(int)
            features['williams_overbought'] = (willr.willr > -20).astype(int)
            
            # ATR features
            atr = vbt.ATR.run(high, low, close)
            features['atr'] = atr.atr
            features['atr_percent'] = atr.atr / close
            features['atr_ratio'] = atr.atr / atr.atr.rolling(20).mean()
            
            # ADX features
            adx = vbt.ADX.run(high, low, close)
            features['adx'] = adx.adx
            features['adx_pos'] = adx.plus_di
            features['adx_neg'] = adx.minus_di
            features['adx_trend_strength'] = (adx.adx > 25).astype(int)
            features['adx_direction'] = (adx.plus_di > adx.minus_di).astype(int)
            
            # Volume features
            features['volume_sma'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / features['volume_sma']
            features['volume_spike'] = (volume > volume.rolling(20).mean() * 2).astype(int)
            
            # OBV features
            obv = vbt.OBV.run(close, volume)
            features['obv'] = obv.obv
            features['obv_sma'] = obv.obv.rolling(20).mean()
            features['obv_momentum'] = obv.obv.diff()
            features['obv_divergence'] = (obv.obv > obv.obv.rolling(20).mean()).astype(int)
            
            # AD features
            ad = vbt.AD.run(high, low, close, volume)
            features['ad'] = ad.ad
            features['ad_sma'] = ad.ad.rolling(20).mean()
            features['ad_momentum'] = ad.ad.diff()
            
            # CMF features
            cmf = vbt.CMF.run(high, low, close, volume)
            features['cmf'] = cmf.cmf
            features['cmf_positive'] = (cmf.cmf > 0).astype(int)
            features['cmf_negative'] = (cmf.cmf < 0).astype(int)
            
            # Volatility features
            features['volatility'] = close.rolling(20).std()
            features['volatility_percent'] = features['volatility'] / close
            features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
            
            # Price pattern features
            features['doji'] = vbt.DOJI.run(open=self.data['open'], high=high, low=low, close=close).doji
            features['hammer'] = vbt.HAMMER.run(open=self.data['open'], high=high, low=low, close=close).hammer
            features['shooting_star'] = vbt.SHOOTING_STAR.run(open=self.data['open'], high=high, low=low, close=close).shooting_star
            
            # Momentum features
            for period in [5, 10, 20]:
                features[f'momentum_{period}'] = close / close.shift(period) - 1
                features[f'roc_{period}'] = close.pct_change(period)
                features[f'price_acceleration_{period}'] = features[f'momentum_{period}'].diff()
            
            # Trend features
            features['trend_sma_20_50'] = (features['sma_20'] > features['sma_50']).astype(int)
            features['trend_sma_50_100'] = (features['sma_50'] > features['sma_100']).astype(int)
            features['trend_consensus'] = (features['trend_sma_20_50'] + features['trend_sma_50_100']) / 2
            
            # Support/Resistance features
            features['resistance'] = high.rolling(20).max()
            features['support'] = low.rolling(20).min()
            features['price_vs_resistance'] = close / features['resistance']
            features['price_vs_support'] = close / features['support']
            features['breakout_resistance'] = (close > features['resistance'].shift(1)).astype(int)
            features['breakdown_support'] = (close < features['support'].shift(1)).astype(int)
            
            # Time-based features
            features['hour'] = self.data.index.hour
            features['day_of_week'] = self.data.index.dayofweek
            features['month'] = self.data.index.month
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            features['is_market_open'] = ((features['hour'] >= 9) & (features['hour'] <= 16)).astype(int)
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = volume.shift(lag)
                features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
                features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
                features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
                features[f'returns_kurt_{window}'] = features['returns'].rolling(window).kurt()
                features[f'volume_mean_{window}'] = volume.rolling(window).mean()
                features[f'volume_std_{window}'] = volume.rolling(window).std()
            
            logger.info(f"✅ Generated {len(features)} VectorBT features")
            
        except Exception as e:
            logger.error(f"Error generating VectorBT features: {e}")
            return {}
        
        return features
    
    def create_signal_based_features(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Create signal-based features using VectorBT.
        
        Args:
            features: Base features
            
        Returns:
            Signal-based features
        """
        logger.info("📊 Creating signal-based features...")
        
        signal_features = {}
        
        try:
            close = self.data['close']
            
            # Trend signals
            if 'sma_20' in features and 'sma_50' in features:
                signal_features['trend_signal'] = (features['sma_20'] > features['sma_50']).astype(int)
                signal_features['trend_strength'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
            
            # MACD signals
            if 'macd' in features and 'macd_signal' in features:
                signal_features['macd_signal'] = (features['macd'] > features['macd_signal']).astype(int)
                signal_features['macd_strength'] = features['macd'] - features['macd_signal']
            
            # RSI signals
            if 'rsi' in features:
                signal_features['rsi_signal'] = (features['rsi'] > 50).astype(int)
                signal_features['rsi_strength'] = (features['rsi'] - 50) / 50
            
            # Bollinger Bands signals
            if 'bb_percent' in features:
                signal_features['bb_signal'] = (features['bb_percent'] > 0.8).astype(int) - (features['bb_percent'] < 0.2).astype(int)
                signal_features['bb_strength'] = features['bb_percent']
            
            # Volume signals
            if 'volume_ratio' in features:
                signal_features['volume_signal'] = (features['volume_ratio'] > 1.5).astype(int)
                signal_features['volume_strength'] = features['volume_ratio']
            
            # Combined signals
            signal_cols = [col for col in signal_features.keys() if col.endswith('_signal')]
            if signal_cols:
                signal_df = pd.DataFrame({col: signal_features[col] for col in signal_cols})
                signal_features['consensus_signal'] = signal_df.sum(axis=1)
                signal_features['signal_agreement'] = (signal_df.sum(axis=1) >= len(signal_cols) // 2).astype(int)
            
            logger.info(f"✅ Created {len(signal_features)} signal-based features")
            
        except Exception as e:
            logger.error(f"Error creating signal-based features: {e}")
            return {}
        
        return signal_features
    
    def evaluate_feature_performance(self, features: Dict[str, pd.Series], 
                                   target: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate feature performance using VectorBT metrics.
        
        Args:
            features: Feature dictionary
            target: Target variable
            
        Returns:
            Performance metrics for each feature
        """
        logger.info("📈 Evaluating feature performance...")
        
        performance = {}
        
        try:
            for feature_name, feature_values in features.items():
                if feature_values.isna().all():
                    continue
                
                # Align feature and target
                aligned_data = pd.DataFrame({
                    'feature': feature_values,
                    'target': target
                }).dropna()
                
                if len(aligned_data) < 100:  # Need sufficient data
                    continue
                
                feature_series = aligned_data['feature']
                target_series = aligned_data['target']
                
                # Basic statistics
                performance[feature_name] = {
                    'correlation': feature_series.corr(target_series),
                    'mutual_info': self._calculate_mutual_info(feature_series, target_series),
                    'variance': feature_series.var(),
                    'mean': feature_series.mean(),
                    'std': feature_series.std(),
                    'skewness': feature_series.skew(),
                    'kurtosis': feature_series.kurtosis(),
                    'nan_count': feature_series.isna().sum(),
                    'zero_count': (feature_series == 0).sum(),
                    'unique_values': feature_series.nunique()
                }
                
                # Signal-based performance (if binary)
                if feature_series.nunique() == 2:
                    signal_performance = self._evaluate_signal_performance(feature_series, target_series)
                    performance[feature_name].update(signal_performance)
                
                # Rolling correlation stability
                if len(feature_series) > 200:
                    rolling_corr = feature_series.rolling(100).corr(target_series.rolling(100))
                    performance[feature_name]['rolling_corr_mean'] = rolling_corr.mean()
                    performance[feature_name]['rolling_corr_std'] = rolling_corr.std()
            
            logger.info(f"✅ Evaluated performance for {len(performance)} features")
            
        except Exception as e:
            logger.error(f"Error evaluating feature performance: {e}")
            return {}
        
        return performance
    
    def _calculate_mutual_info(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between two series."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            return mutual_info_regression(x.values.reshape(-1, 1), y.values)[0]
        except:
            return 0.0
    
    def _evaluate_signal_performance(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Evaluate performance of binary signals."""
        try:
            # Create simple strategy
            strategy_returns = signals.shift(1) * returns
            
            if strategy_returns.std() == 0:
                return {'signal_sharpe': 0.0, 'signal_return': 0.0, 'signal_win_rate': 0.0}
            
            return {
                'signal_sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
                'signal_return': strategy_returns.mean() * 252,
                'signal_win_rate': (strategy_returns > 0).mean(),
                'signal_max_drawdown': self._calculate_max_drawdown(strategy_returns)
            }
        except:
            return {'signal_sharpe': 0.0, 'signal_return': 0.0, 'signal_win_rate': 0.0}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive VectorBT feature analysis.
        
        Returns:
            Complete analysis results
        """
        logger.info("🔬 Running comprehensive VectorBT feature analysis...")
        
        # Generate features
        features = self.generate_vectorbt_features()
        
        # Create signal-based features
        signal_features = self.create_signal_based_features(features)
        
        # Combine all features
        all_features = {**features, **signal_features}
        
        # Create target
        target = self.data['close'].pct_change().shift(-1)  # Next period returns
        
        # Evaluate performance
        performance = self.evaluate_feature_performance(all_features, target)
        
        # Create feature ranking
        feature_ranking = self._create_feature_ranking(performance)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(all_features, performance)
        
        results = {
            'features': all_features,
            'performance': performance,
            'feature_ranking': feature_ranking,
            'summary': summary,
            'target_info': {
                'target_name': self.target_col,
                'target_stats': {
                    'mean': target.mean(),
                    'std': target.std(),
                    'skewness': target.skew(),
                    'kurtosis': target.kurtosis()
                }
            }
        }
        
        logger.info("✅ Comprehensive VectorBT feature analysis completed")
        return results
    
    def _create_feature_ranking(self, performance: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create feature ranking based on performance metrics."""
        ranking_data = []
        
        for feature_name, metrics in performance.items():
            ranking_data.append({
                'feature': feature_name,
                'correlation': abs(metrics.get('correlation', 0)),
                'mutual_info': metrics.get('mutual_info', 0),
                'signal_sharpe': abs(metrics.get('signal_sharpe', 0)),
                'rolling_corr_stability': 1 - metrics.get('rolling_corr_std', 1),
                'variance': metrics.get('variance', 0),
                'unique_values': metrics.get('unique_values', 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Calculate composite score
        ranking_df['composite_score'] = (
            ranking_df['correlation'] * 0.3 +
            ranking_df['mutual_info'] * 0.25 +
            ranking_df['signal_sharpe'] * 0.2 +
            ranking_df['rolling_corr_stability'] * 0.15 +
            (ranking_df['variance'] > 0).astype(int) * 0.05 +
            (ranking_df['unique_values'] > 2).astype(int) * 0.05
        )
        
        return ranking_df.sort_values('composite_score', ascending=False)
    
    def _generate_summary_statistics(self, features: Dict[str, pd.Series], 
                                   performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'total_features': len(features),
            'high_correlation_features': len([f for f, p in performance.items() if abs(p.get('correlation', 0)) > 0.1]),
            'high_mutual_info_features': len([f for f, p in performance.items() if p.get('mutual_info', 0) > 0.01]),
            'signal_features': len([f for f, p in performance.items() if 'signal_sharpe' in p]),
            'avg_correlation': np.mean([abs(p.get('correlation', 0)) for p in performance.values()]),
            'avg_mutual_info': np.mean([p.get('mutual_info', 0) for p in performance.values()]),
            'feature_categories': {
                'price_based': len([f for f in features.keys() if 'price' in f or 'returns' in f]),
                'technical_indicators': len([f for f in features.keys() if any(ind in f for ind in ['sma', 'ema', 'rsi', 'macd', 'bb', 'stoch'])]),
                'volume_based': len([f for f in features.keys() if 'volume' in f or 'obv' in f or 'ad' in f]),
                'signal_based': len([f for f in features.keys() if 'signal' in f]),
                'time_based': len([f for f in features.keys() if any(t in f for t in ['hour', 'day', 'month', 'weekend'])]),
                'lagged': len([f for f in features.keys() if 'lag' in f]),
                'rolling': len([f for f in features.keys() if 'rolling' in f or 'mean' in f or 'std' in f])
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = "vectorbt_features.json"):
        """Save analysis results to file."""
        output_path = Path(filename)
        
        # Convert to serializable format
        serializable_results = {}
        for key, value in results.items():
            if key == 'features':
                serializable_results[key] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            elif key == 'feature_ranking':
                serializable_results[key] = value.to_dict('records')
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
    
    # Run VectorBT feature analysis
    optimizer = VectorBTFeatureOptimizer(sample_data)
    results = optimizer.run_comprehensive_analysis()
    
    # Save results
    optimizer.save_results(results)
    
    print("✅ VectorBT feature analysis completed!")
    print(f"Generated {results['summary']['total_features']} features")
    print(f"High correlation features: {results['summary']['high_correlation_features']}")
    print(f"Signal features: {results['summary']['signal_features']}")
    
    # Show top features
    top_features = results['feature_ranking'].head(10)
    print("\nTop 10 Features:")
    for _, row in top_features.iterrows():
        print(f"{row['feature']}: {row['composite_score']:.4f}")