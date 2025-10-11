"""
VectorBT Integration Examples

This module provides practical examples of integrating VectorBT optimizations
with the existing research framework, demonstrating real-world usage patterns
and best practices.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import warnings

# Suppress VectorBT warnings
warnings.filterwarnings('ignore', category=UserWarning, module='vectorbt')

logger = logging.getLogger(__name__)

class VectorBTIntegrationExamples:
    """
    Practical examples of VectorBT integration with the research framework.
    
    This class demonstrates how to integrate VectorBT optimizations with
    existing research components and workflows.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize integration examples.
        
        Args:
            data: OHLCV data
        """
        self.data = data.copy()
        
        # Ensure proper index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        logger.info("✅ VectorBT integration examples initialized")
    
    def example_1_enhanced_crypto_analysis(self) -> Dict[str, Any]:
        """
        Example 1: Enhanced Crypto Analysis with VectorBT
        
        This example shows how to enhance the existing crypto analysis
        with VectorBT's advanced technical indicators and backtesting.
        """
        logger.info("🔬 Example 1: Enhanced Crypto Analysis")
        
        from .crypto_analysis_optimizer import VectorBTCryptoOptimizer
        
        # Initialize VectorBT optimizer
        optimizer = VectorBTCryptoOptimizer()
        
        # Generate enhanced analysis
        results = optimizer.generate_enhanced_analysis(self.data)
        
        # Extract key metrics
        indicators = results['indicators']
        signals = results['signals']
        backtest = results['backtest_results']
        
        # Create summary
        summary = {
            'total_indicators': len(indicators),
            'total_signals': len(signals),
            'profitable_strategies': len([s for s in backtest.values() if s['total_return'] > 0]),
            'best_strategy': max(backtest.items(), key=lambda x: x[1]['sharpe_ratio']) if backtest else None,
            'avg_sharpe_ratio': np.mean([s['sharpe_ratio'] for s in backtest.values() if not pd.isna(s['sharpe_ratio'])]),
            'total_trades': sum(s['total_trades'] for s in backtest.values())
        }
        
        logger.info(f"✅ Enhanced crypto analysis completed: {summary['total_indicators']} indicators, {summary['total_signals']} signals")
        
        return {
            'results': results,
            'summary': summary,
            'example_type': 'enhanced_crypto_analysis'
        }
    
    def example_2_feature_engineering_pipeline(self) -> Dict[str, Any]:
        """
        Example 2: VectorBT Feature Engineering Pipeline
        
        This example demonstrates how to use VectorBT for comprehensive
        feature engineering in the research pipeline.
        """
        logger.info("🔧 Example 2: VectorBT Feature Engineering Pipeline")
        
        from .feature_comparison_optimizer import VectorBTFeatureOptimizer
        
        # Initialize feature optimizer
        optimizer = VectorBTFeatureOptimizer(self.data)
        
        # Generate comprehensive features
        results = optimizer.run_comprehensive_analysis()
        
        # Extract feature information
        features = results['features']
        performance = results['performance']
        ranking = results['feature_ranking']
        
        # Analyze feature categories
        feature_categories = {
            'price_based': len([f for f in features.keys() if 'price' in f or 'returns' in f]),
            'technical_indicators': len([f for f in features.keys() if any(ind in f for ind in ['sma', 'ema', 'rsi', 'macd', 'bb', 'stoch'])]),
            'volume_based': len([f for f in features.keys() if 'volume' in f or 'obv' in f or 'ad' in f]),
            'signal_based': len([f for f in features.keys() if 'signal' in f]),
            'time_based': len([f for f in features.keys() if any(t in f for t in ['hour', 'day', 'month', 'weekend'])]),
            'lagged': len([f for f in features.keys() if 'lag' in f]),
            'rolling': len([f for f in features.keys() if 'rolling' in f or 'mean' in f or 'std' in f])
        }
        
        # Top performing features
        top_features = ranking.head(10)['feature'].tolist()
        
        summary = {
            'total_features': len(features),
            'feature_categories': feature_categories,
            'top_features': top_features,
            'avg_correlation': np.mean([abs(p.get('correlation', 0)) for p in performance.values()]),
            'high_correlation_features': len([p for p in performance.values() if abs(p.get('correlation', 0)) > 0.1])
        }
        
        logger.info(f"✅ Feature engineering pipeline completed: {summary['total_features']} features generated")
        
        return {
            'results': results,
            'summary': summary,
            'example_type': 'feature_engineering_pipeline'
        }
    
    def example_3_profit_labeling_optimization(self) -> Dict[str, Any]:
        """
        Example 3: Profit Labeling Optimization with VectorBT
        
        This example shows how to optimize profit labeling using VectorBT's
        advanced backtesting and signal generation capabilities.
        """
        logger.info("🎯 Example 3: Profit Labeling Optimization")
        
        from .profit_labeling_optimizer import VectorBTProfitLabelingOptimizer
        
        # Initialize profit labeling optimizer
        optimizer = VectorBTProfitLabelingOptimizer(self.data)
        
        # Run comprehensive analysis
        results = optimizer.run_comprehensive_analysis()
        
        # Extract key metrics
        signals = results['signals']
        backtest = results['backtest_results']
        optimization = results['optimization_results']
        consistency = results['consistency_results']
        
        # Analyze signal performance
        signal_performance = {
            'total_signals': len(signals),
            'profitable_signals': len([s for s in backtest.values() if s['total_return'] > 0]),
            'best_signal': max(backtest.items(), key=lambda x: x[1]['sharpe_ratio']) if backtest else None,
            'avg_win_rate': np.mean([s['win_rate'] for s in backtest.values() if not pd.isna(s['win_rate'])]),
            'avg_profit_factor': np.mean([s['profit_factor'] for s in backtest.values() if not pd.isna(s['profit_factor'])])
        }
        
        # Optimization results
        opt_summary = {
            'best_target': optimization.get('best_params', {}).get('target', 0),
            'best_horizon': optimization.get('best_params', {}).get('horizon', 0),
            'best_score': optimization.get('best_params', {}).get('score', 0),
            'total_combinations_tested': len(optimization.get('all_results', []))
        }
        
        # Consistency analysis
        consistency_summary = {
            'total_regimes': len(consistency),
            'avg_consistency_score': np.mean([r['consistency_score'] for r in consistency.values()]),
            'high_consistency_signals': len([r for r in consistency.values() if r['consistency_score'] > 0.7])
        }
        
        summary = {
            'signal_performance': signal_performance,
            'optimization': opt_summary,
            'consistency': consistency_summary
        }
        
        logger.info(f"✅ Profit labeling optimization completed: {signal_performance['total_signals']} signals, {signal_performance['profitable_signals']} profitable")
        
        return {
            'results': results,
            'summary': summary,
            'example_type': 'profit_labeling_optimization'
        }
    
    def example_4_pattern_discovery_workflow(self) -> Dict[str, Any]:
        """
        Example 4: Pattern Discovery Workflow with VectorBT
        
        This example demonstrates a complete pattern discovery workflow
        using VectorBT's pattern recognition capabilities.
        """
        logger.info("🔍 Example 4: Pattern Discovery Workflow")
        
        from .price_patterns_optimizer import VectorBTPricePatternsOptimizer
        
        # Initialize pattern optimizer
        optimizer = VectorBTPricePatternsOptimizer(self.data)
        
        # Run comprehensive pattern analysis
        results = optimizer.run_comprehensive_analysis()
        
        # Extract pattern information
        patterns = results['patterns']
        validation = results['validation_results']
        ranking = results['pattern_ranking']
        
        # Analyze pattern categories
        pattern_categories = {
            'candlestick': len([p for p in patterns.keys() if any(c in p for c in ['doji', 'hammer', 'shooting', 'engulfing', 'star'])]),
            'technical': len([p for p in patterns.keys() if any(t in p for t in ['rsi', 'macd', 'bb', 'stoch'])]),
            'price_action': len([p for p in patterns.keys() if any(pa in p for pa in ['breakout', 'double', 'head', 'triangle', 'flag'])]),
            'volume': len([p for p in patterns.keys() if any(v in p for v in ['volume', 'obv', 'ad', 'cmf'])]),
            'momentum': len([p for p in patterns.keys() if 'momentum' in p or 'roc' in p]),
            'trend': len([p for p in patterns.keys() if 'trend' in p or 'cross' in p]),
            'support_resistance': len([p for p in patterns.keys() if any(sr in p for sr in ['support', 'resistance', 'bounce', 'rejection'])])
        }
        
        # Pattern performance analysis
        pattern_performance = {
            'total_patterns': len(patterns),
            'validated_patterns': len(validation),
            'profitable_patterns': len([p for p in validation.values() if p['total_return'] > 0]),
            'best_pattern': max(validation.items(), key=lambda x: x[1]['sharpe_ratio']) if validation else None,
            'avg_sharpe_ratio': np.mean([p['sharpe_ratio'] for p in validation.values() if not pd.isna(p['sharpe_ratio'])]),
            'avg_win_rate': np.mean([p['win_rate'] for p in validation.values() if not pd.isna(p['win_rate'])])
        }
        
        # Top patterns
        top_patterns = ranking.head(10)['pattern'].tolist()
        
        summary = {
            'pattern_categories': pattern_categories,
            'pattern_performance': pattern_performance,
            'top_patterns': top_patterns
        }
        
        logger.info(f"✅ Pattern discovery workflow completed: {pattern_performance['total_patterns']} patterns, {pattern_performance['profitable_patterns']} profitable")
        
        return {
            'results': results,
            'summary': summary,
            'example_type': 'pattern_discovery_workflow'
        }
    
    def example_5_clustering_analysis(self) -> Dict[str, Any]:
        """
        Example 5: Clustering Analysis with VectorBT
        
        This example shows how to perform market regime detection
        using VectorBT-enhanced clustering analysis.
        """
        logger.info("📊 Example 5: Clustering Analysis")
        
        from .clustering_optimizer import VectorBTClusteringOptimizer
        
        # Initialize clustering optimizer
        optimizer = VectorBTClusteringOptimizer(self.data)
        
        # Run comprehensive clustering analysis
        results = optimizer.run_comprehensive_analysis()
        
        # Extract clustering information
        features = results['features']
        clusters = results['cluster_results']
        validation = results['validation_results']
        optimization = results['optimization_results']
        
        # Clustering metrics
        clustering_metrics = {
            'total_clusters': clusters.get('n_clusters', 0),
            'silhouette_score': clusters.get('silhouette_score', 0),
            'calinski_harabasz_score': clusters.get('calinski_harabasz_score', 0),
            'feature_count': len(features.columns),
            'method': clusters.get('method', 'unknown')
        }
        
        # Regime performance
        regime_performance = {
            'total_regimes': len(validation),
            'profitable_regimes': len([r for r in validation.values() if r['total_return'] > 0]),
            'best_regime': max(validation.items(), key=lambda x: x[1]['sharpe_ratio']) if validation else None,
            'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in validation.values() if not pd.isna(r['sharpe_ratio'])]),
            'avg_win_rate': np.mean([r['win_rate'] for r in validation.values() if not pd.isna(r['win_rate'])])
        }
        
        # Optimization results
        opt_summary = {
            'recommended_method': optimization.get('recommended_method', 'unknown'),
            'best_kmeans_clusters': optimization.get('best_kmeans', {}).get('n_clusters', 0),
            'best_dbscan_params': optimization.get('best_dbscan', {}),
            'total_methods_tested': len(optimization.get('kmeans_results', [])) + len(optimization.get('dbscan_results', []))
        }
        
        summary = {
            'clustering_metrics': clustering_metrics,
            'regime_performance': regime_performance,
            'optimization': opt_summary
        }
        
        logger.info(f"✅ Clustering analysis completed: {clustering_metrics['total_clusters']} clusters, {regime_performance['profitable_regimes']} profitable regimes")
        
        return {
            'results': results,
            'summary': summary,
            'example_type': 'clustering_analysis'
        }
    
    def example_6_comprehensive_research_pipeline(self) -> Dict[str, Any]:
        """
        Example 6: Comprehensive Research Pipeline
        
        This example demonstrates how to integrate all VectorBT optimizations
        into a comprehensive research pipeline.
        """
        logger.info("🚀 Example 6: Comprehensive Research Pipeline")
        
        # Import all optimizers
        from .crypto_analysis_optimizer import VectorBTCryptoOptimizer
        from .feature_comparison_optimizer import VectorBTFeatureOptimizer
        from .profit_labeling_optimizer import VectorBTProfitLabelingOptimizer
        from .price_patterns_optimizer import VectorBTPricePatternsOptimizer
        from .clustering_optimizer import VectorBTClusteringOptimizer
        
        # Initialize all optimizers
        crypto_optimizer = VectorBTCryptoOptimizer()
        feature_optimizer = VectorBTFeatureOptimizer(self.data)
        profit_optimizer = VectorBTProfitLabelingOptimizer(self.data)
        pattern_optimizer = VectorBTPricePatternsOptimizer(self.data)
        clustering_optimizer = VectorBTClusteringOptimizer(self.data)
        
        # Run all analyses
        logger.info("Running crypto analysis...")
        crypto_results = crypto_optimizer.generate_enhanced_analysis(self.data)
        
        logger.info("Running feature analysis...")
        feature_results = feature_optimizer.run_comprehensive_analysis()
        
        logger.info("Running profit labeling analysis...")
        profit_results = profit_optimizer.run_comprehensive_analysis()
        
        logger.info("Running pattern discovery...")
        pattern_results = pattern_optimizer.run_comprehensive_analysis()
        
        logger.info("Running clustering analysis...")
        clustering_results = clustering_optimizer.run_comprehensive_analysis()
        
        # Compile comprehensive results
        comprehensive_results = {
            'crypto_analysis': {
                'indicators_count': len(crypto_results['indicators']),
                'signals_count': len(crypto_results['signals']),
                'profitable_strategies': len([s for s in crypto_results['backtest_results'].values() if s['total_return'] > 0]),
                'best_sharpe': max([s['sharpe_ratio'] for s in crypto_results['backtest_results'].values() if not pd.isna(s['sharpe_ratio'])], default=0)
            },
            'feature_analysis': {
                'features_count': len(feature_results['features']),
                'high_correlation_features': feature_results['summary']['high_correlation_features'],
                'top_feature': feature_results['feature_ranking'].iloc[0]['feature'] if not feature_results['feature_ranking'].empty else None
            },
            'profit_labeling': {
                'signals_count': len(profit_results['signals']),
                'profitable_signals': len([s for s in profit_results['backtest_results'].values() if s['total_return'] > 0]),
                'best_target': profit_results['optimization_results'].get('best_params', {}).get('target', 0),
                'best_horizon': profit_results['optimization_results'].get('best_params', {}).get('horizon', 0)
            },
            'pattern_discovery': {
                'patterns_count': len(pattern_results['patterns']),
                'profitable_patterns': len([p for p in pattern_results['validation_results'].values() if p['total_return'] > 0]),
                'best_pattern': pattern_results['summary'].get('best_pattern', {}).get('name', 'None')
            },
            'clustering_analysis': {
                'clusters_count': clustering_results['summary']['total_clusters'],
                'profitable_regimes': clustering_results['summary']['profitable_regimes'],
                'silhouette_score': clustering_results['summary']['silhouette_score']
            }
        }
        
        # Calculate overall metrics
        total_indicators = comprehensive_results['crypto_analysis']['indicators_count']
        total_features = comprehensive_results['feature_analysis']['features_count']
        total_signals = (comprehensive_results['crypto_analysis']['signals_count'] + 
                        comprehensive_results['profit_labeling']['signals_count'])
        total_patterns = comprehensive_results['pattern_discovery']['patterns_count']
        total_clusters = comprehensive_results['clustering_analysis']['clusters_count']
        
        overall_summary = {
            'total_indicators': total_indicators,
            'total_features': total_features,
            'total_signals': total_signals,
            'total_patterns': total_patterns,
            'total_clusters': total_clusters,
            'total_components': total_indicators + total_features + total_signals + total_patterns + total_clusters,
            'profitable_strategies': (comprehensive_results['crypto_analysis']['profitable_strategies'] +
                                    comprehensive_results['profit_labeling']['profitable_signals'] +
                                    comprehensive_results['pattern_discovery']['profitable_patterns'] +
                                    comprehensive_results['clustering_analysis']['profitable_regimes'])
        }
        
        logger.info(f"✅ Comprehensive research pipeline completed: {overall_summary['total_components']} total components")
        
        return {
            'comprehensive_results': comprehensive_results,
            'overall_summary': overall_summary,
            'individual_results': {
                'crypto': crypto_results,
                'features': feature_results,
                'profit': profit_results,
                'patterns': pattern_results,
                'clustering': clustering_results
            },
            'example_type': 'comprehensive_research_pipeline'
        }
    
    def example_7_performance_comparison(self) -> Dict[str, Any]:
        """
        Example 7: Performance Comparison
        
        This example compares the performance of VectorBT-optimized
        analysis with traditional methods.
        """
        logger.info("⚡ Example 7: Performance Comparison")
        
        import time
        
        # Traditional method simulation (simplified)
        start_time = time.time()
        
        # Simulate traditional technical analysis
        close = self.data['close']
        traditional_indicators = {
            'sma_20': close.rolling(20).mean(),
            'sma_50': close.rolling(50).mean(),
            'rsi': self._calculate_rsi_traditional(close),
            'macd': self._calculate_macd_traditional(close)
        }
        
        traditional_time = time.time() - start_time
        
        # VectorBT method
        start_time = time.time()
        
        from .crypto_analysis_optimizer import VectorBTCryptoOptimizer
        optimizer = VectorBTCryptoOptimizer()
        vectorbt_results = optimizer.generate_enhanced_analysis(self.data)
        
        vectorbt_time = time.time() - start_time
        
        # Performance comparison
        performance_comparison = {
            'traditional': {
                'indicators_count': len(traditional_indicators),
                'execution_time': traditional_time,
                'indicators_per_second': len(traditional_indicators) / traditional_time if traditional_time > 0 else 0
            },
            'vectorbt': {
                'indicators_count': len(vectorbt_results['indicators']),
                'execution_time': vectorbt_time,
                'indicators_per_second': len(vectorbt_results['indicators']) / vectorbt_time if vectorbt_time > 0 else 0
            }
        }
        
        # Calculate improvements
        speed_improvement = (performance_comparison['vectorbt']['indicators_per_second'] / 
                           performance_comparison['traditional']['indicators_per_second']) if performance_comparison['traditional']['indicators_per_second'] > 0 else 0
        
        feature_improvement = (performance_comparison['vectorbt']['indicators_count'] / 
                             performance_comparison['traditional']['indicators_count']) if performance_comparison['traditional']['indicators_count'] > 0 else 0
        
        summary = {
            'performance_comparison': performance_comparison,
            'speed_improvement': speed_improvement,
            'feature_improvement': feature_improvement,
            'overall_improvement': speed_improvement * feature_improvement
        }
        
        logger.info(f"✅ Performance comparison completed: {speed_improvement:.1f}x speed, {feature_improvement:.1f}x features")
        
        return {
            'results': summary,
            'example_type': 'performance_comparison'
        }
    
    def _calculate_rsi_traditional(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using traditional method."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd_traditional(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD using traditional method."""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def run_all_examples(self) -> Dict[str, Any]:
        """
        Run all integration examples.
        
        Returns:
            Dictionary containing all example results
        """
        logger.info("🚀 Running all VectorBT integration examples...")
        
        examples = {}
        
        try:
            # Run all examples
            examples['example_1'] = self.example_1_enhanced_crypto_analysis()
            examples['example_2'] = self.example_2_feature_engineering_pipeline()
            examples['example_3'] = self.example_3_profit_labeling_optimization()
            examples['example_4'] = self.example_4_pattern_discovery_workflow()
            examples['example_5'] = self.example_5_clustering_analysis()
            examples['example_6'] = self.example_6_comprehensive_research_pipeline()
            examples['example_7'] = self.example_7_performance_comparison()
            
            # Generate overall summary
            total_components = sum(
                example.get('summary', {}).get('total_indicators', 0) +
                example.get('summary', {}).get('total_features', 0) +
                example.get('summary', {}).get('total_signals', 0) +
                example.get('summary', {}).get('total_patterns', 0) +
                example.get('summary', {}).get('total_clusters', 0)
                for example in examples.values()
            )
            
            overall_summary = {
                'total_examples': len(examples),
                'total_components_generated': total_components,
                'examples_completed': len([e for e in examples.values() if 'summary' in e]),
                'performance_improvement': examples.get('example_7', {}).get('results', {}).get('overall_improvement', 0)
            }
            
            logger.info(f"✅ All examples completed: {overall_summary['total_examples']} examples, {overall_summary['total_components_generated']} components")
            
            return {
                'examples': examples,
                'overall_summary': overall_summary
            }
            
        except Exception as e:
            logger.error(f"Error running examples: {e}")
            return {'error': str(e)}


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
    
    # Run integration examples
    examples = VectorBTIntegrationExamples(sample_data)
    results = examples.run_all_examples()
    
    print("✅ VectorBT integration examples completed!")
    print(f"Total examples: {results['overall_summary']['total_examples']}")
    print(f"Total components: {results['overall_summary']['total_components_generated']}")
    print(f"Performance improvement: {results['overall_summary']['performance_improvement']:.1f}x")