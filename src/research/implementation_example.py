"""
Practical Implementation Example: Economic Relevance Research Framework

This script demonstrates how to use the economic relevance research framework
to determine which market dimensions have meaningful impact on price patterns.

Usage:
    python implementation_example.py --data_path /path/to/market_data.csv
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import argparse
import logging
from pathlib import Path

# Import research frameworks
from economic_relevance_research_framework import (
    EconomicRelevanceResearchOrchestrator,
    ResearchMethodologyConfig,
    PriceMovementPattern
)
from volatility_impact_research import VolatilityImpactResearchOrchestrator
from microstructure_impact_research import MicrostructureImpactResearchOrchestrator

# Import existing dimension analysis (assuming these exist in your codebase)
try:
    from src.research.clusters.dimension_economic_relevance import analyze_all_dimensions_economic_relevance
    from src.research.clusters.economic_metrics import EconomicValidator
except ImportError:
    print("⚠️  Note: Some existing modules not found. Using standalone implementation.")


def create_sample_market_data(n_periods: int = 2000) -> pd.DataFrame:
    """
    Create sample market data for demonstration.
    Replace this with your actual data loading function.
    """
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily returns
    
    # Add some autocorrelation and volatility clustering
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum
        if abs(returns[i-1]) > 0.03:  # Volatility clustering
            returns[i] += np.random.normal(0, 0.01)
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    
    # Volume with some correlation to volatility
    volume = 1000000 + 500000 * np.abs(returns) + np.random.normal(0, 100000, n_periods)
    volume = np.maximum(volume, 10000)  # Minimum volume
    
    market_data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': high,
        'low': low, 
        'close': prices,
        'volume': volume
    })
    
    market_data['open'].iloc[0] = prices[0]  # Fix first open
    market_data.index = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    return market_data


def create_sample_dimension_features(market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create sample dimension features for demonstration.
    Replace this with your actual feature engineering pipeline.
    """
    
    returns = market_data['close'].pct_change().fillna(0)
    prices = market_data['close']
    volume = market_data['volume']
    
    # Volatility dimension features
    volatility_features = pd.DataFrame({
        'realized_vol_5': returns.rolling(5).std(),
        'realized_vol_20': returns.rolling(20).std(),
        'realized_vol_50': returns.rolling(50).std(),
        'vol_of_vol': returns.rolling(20).std().rolling(20).std(),
        'vol_skew': returns.rolling(50).skew(),
        'parkinson_vol': np.sqrt(np.log(market_data['high'] / market_data['low']).rolling(20).mean())
    }, index=market_data.index).fillna(0)
    
    # Momentum dimension features  
    momentum_features = pd.DataFrame({
        'momentum_5': returns.rolling(5).mean(),
        'momentum_20': returns.rolling(20).mean(),
        'momentum_50': returns.rolling(50).mean(),
        'rsi_14': calculate_rsi(prices, 14),
        'momentum_acceleration': returns.rolling(5).mean().diff(),
        'trend_strength': (prices.rolling(20).mean() - prices.rolling(50).mean()) / prices.rolling(50).mean()
    }, index=market_data.index).fillna(0)
    
    # Liquidity dimension features
    liquidity_features = pd.DataFrame({
        'volume_ma_ratio': volume / volume.rolling(20).mean(),
        'volume_volatility': volume.rolling(20).std() / volume.rolling(20).mean(),
        'price_volume_trend': returns.rolling(10).sum() * (volume / volume.rolling(20).mean()),
        'volume_price_correlation': returns.rolling(50).corr(volume / volume.rolling(20).mean()),
        'liquidity_proxy': volume / (market_data['high'] - market_data['low']),
        'amihud_illiquidity': abs(returns) / volume
    }, index=market_data.index).fillna(0)
    
    # Microstructure dimension features (proxies from OHLCV)
    microstructure_features = pd.DataFrame({
        'spread_proxy': (market_data['high'] - market_data['low']) / market_data['close'],
        'order_flow_proxy': returns.rolling(5).mean() * volume,
        'market_depth_proxy': volume / returns.rolling(20).std(),
        'price_impact': abs(returns) / volume,
        'tick_size_proxy': (market_data['close'].diff().abs()).rolling(20).mean(),
        'microstructure_noise': returns.rolling(50).apply(lambda x: x.autocorr(1))
    }, index=market_data.index).fillna(0)
    
    # Correlation dimension features
    correlation_features = pd.DataFrame({
        'autocorr_1': returns.rolling(50).apply(lambda x: x.autocorr(1)),
        'autocorr_5': returns.rolling(50).apply(lambda x: x.autocorr(5)),
        'vol_price_corr': returns.rolling(50).corr(returns.rolling(20).std()),
        'volume_return_corr': returns.rolling(50).corr(volume / volume.rolling(20).mean()),
        'cross_autocorr': returns.rolling(50).apply(lambda x: x.corr(x.shift(1))),
        'lead_lag_effect': returns.rolling(50).corr(returns.shift(-1))
    }, index=market_data.index).fillna(0)
    
    return {
        'volatility': volatility_features,
        'momentum': momentum_features,
        'liquidity': liquidity_features,
        'microstructure': microstructure_features,
        'correlation': correlation_features
    }


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def run_comprehensive_economic_relevance_research(market_data: pd.DataFrame, 
                                                dimension_features: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run comprehensive economic relevance research.
    """
    
    print("🔬 Starting Comprehensive Economic Relevance Research")
    print("=" * 60)
    
    results = {}
    
    # 1. Main Economic Relevance Research
    print("\n📊 1. Main Economic Relevance Framework")
    print("-" * 40)
    
    config = ResearchMethodologyConfig(
        lookback_windows=[5, 10, 20, 50],
        prediction_horizons=[1, 5, 10],
        significance_level=0.05,
        min_sharpe_ratio=0.3,  # Lower threshold for demo
        min_prediction_accuracy=0.52,  # Lower threshold for demo
        bootstrap_samples=100  # Reduced for demo speed
    )
    
    orchestrator = EconomicRelevanceResearchOrchestrator(config)
    
    # Focus on key patterns for demo
    key_patterns = [
        PriceMovementPattern.MOMENTUM_PERSISTENCE,
        PriceMovementPattern.MEAN_REVERSION_SPEED,
        PriceMovementPattern.VOLATILITY_EXPANSION
    ]
    
    main_results = orchestrator.conduct_comprehensive_research(
        market_data=market_data,
        dimension_feature_groups=dimension_features,
        patterns_to_analyze=key_patterns
    )
    
    results['main_research'] = main_results
    
    # Generate main report
    main_report = orchestrator.generate_research_report(main_results)
    results['main_report'] = main_report
    
    print("✅ Main research completed")
    
    # 2. Volatility-Specific Research
    print("\n🌪️ 2. Volatility Impact Research")
    print("-" * 40)
    
    vol_orchestrator = VolatilityImpactResearchOrchestrator()
    vol_results = vol_orchestrator.conduct_comprehensive_volatility_research(market_data)
    vol_report = vol_orchestrator.generate_volatility_research_report(vol_results)
    
    results['volatility_research'] = vol_results
    results['volatility_report'] = vol_report
    
    print("✅ Volatility research completed")
    
    # 3. Microstructure-Specific Research  
    print("\n🔬 3. Microstructure Impact Research")
    print("-" * 40)
    
    micro_orchestrator = MicrostructureImpactResearchOrchestrator()
    micro_results = micro_orchestrator.conduct_comprehensive_microstructure_research(market_data)
    micro_report = micro_orchestrator.generate_microstructure_research_report(micro_results)
    
    results['microstructure_research'] = micro_results
    results['microstructure_report'] = micro_report
    
    print("✅ Microstructure research completed")
    
    return results


def analyze_research_results(results: Dict) -> Dict:
    """
    Analyze and summarize research results.
    """
    
    print("\n📈 Research Results Analysis")
    print("=" * 60)
    
    analysis = {}
    
    # 1. Overall Economic Relevance Summary
    main_results = results.get('main_research', {})
    
    total_tests = 0
    relevant_tests = 0
    
    for dimension_name, dimension_results in main_results.items():
        for pattern_name, pattern_results in dimension_results.items():
            for methodology_name, result in pattern_results.items():
                total_tests += 1
                if result.is_economically_relevant:
                    relevant_tests += 1
    
    overall_relevance_rate = (relevant_tests / total_tests * 100) if total_tests > 0 else 0
    
    analysis['overall_relevance_rate'] = overall_relevance_rate
    analysis['total_tests'] = total_tests
    analysis['relevant_tests'] = relevant_tests
    
    print(f"📊 Overall Economic Relevance Rate: {overall_relevance_rate:.1f}%")
    print(f"   - Total Tests: {total_tests}")
    print(f"   - Economically Relevant: {relevant_tests}")
    
    # 2. Top Performing Dimensions
    dimension_scores = {}
    
    for dimension_name, dimension_results in main_results.items():
        dimension_relevant = 0
        dimension_total = 0
        
        for pattern_results in dimension_results.values():
            for result in pattern_results.values():
                dimension_total += 1
                if result.is_economically_relevant:
                    dimension_relevant += 1
        
        if dimension_total > 0:
            dimension_scores[dimension_name] = dimension_relevant / dimension_total
    
    # Sort dimensions by relevance
    top_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top Performing Dimensions:")
    for i, (dim_name, score) in enumerate(top_dimensions[:3], 1):
        status = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        print(f"   {i}. {status} {dim_name.upper()}: {score:.1%} relevance rate")
    
    analysis['dimension_rankings'] = top_dimensions
    
    # 3. Pattern Analysis
    pattern_performance = {}
    
    for dimension_results in main_results.values():
        for pattern_name, pattern_results in dimension_results.items():
            if pattern_name not in pattern_performance:
                pattern_performance[pattern_name] = {'total': 0, 'relevant': 0}
            
            for result in pattern_results.values():
                pattern_performance[pattern_name]['total'] += 1
                if result.is_economically_relevant:
                    pattern_performance[pattern_name]['relevant'] += 1
    
    print(f"\n🎯 Pattern Predictability Analysis:")
    for pattern_name, performance in pattern_performance.items():
        if performance['total'] > 0:
            rate = performance['relevant'] / performance['total']
            status = "✅" if rate > 0.5 else "⚠️" if rate > 0.3 else "❌"
            print(f"   {status} {pattern_name.replace('_', ' ').title()}: {rate:.1%}")
    
    analysis['pattern_performance'] = pattern_performance
    
    return analysis


def generate_trading_recommendations(results: Dict, analysis: Dict) -> List[str]:
    """
    Generate practical trading recommendations based on research results.
    """
    
    recommendations = []
    
    overall_relevance_rate = analysis.get('overall_relevance_rate', 0)
    top_dimensions = analysis.get('dimension_rankings', [])
    
    print(f"\n💡 Trading Strategy Recommendations")
    print("=" * 60)
    
    if overall_relevance_rate > 60:
        recommendations.append("✅ STRONG FOUNDATION: Multiple dimensions show economic relevance")
        recommendations.append("   → Proceed with dimension-based regime modeling")
        recommendations.append("   → Develop multi-dimensional trading strategies")
        
        if top_dimensions:
            top_dim = top_dimensions[0][0]
            recommendations.append(f"   → Focus primary strategy development on {top_dim} dimension")
    
    elif overall_relevance_rate > 30:
        recommendations.append("⚠️ MODERATE FOUNDATION: Some dimensions show promise")
        recommendations.append("   → Selective use of top-performing dimensions")
        recommendations.append("   → Consider ensemble approaches combining multiple signals")
        
        if top_dimensions:
            top_2_dims = [dim[0] for dim in top_dimensions[:2]]
            recommendations.append(f"   → Focus on {' and '.join(top_2_dims)} dimensions")
    
    else:
        recommendations.append("❌ LIMITED FOUNDATION: Few dimensions show clear economic value")
        recommendations.append("   → Consider simpler approaches (volume/volatility focus)")
        recommendations.append("   → Investigate alternative feature engineering")
        recommendations.append("   → May need higher frequency data or additional data sources")
    
    # Specific dimension recommendations
    main_results = results.get('main_research', {})
    
    for dimension_name, dimension_results in main_results.items():
        relevant_patterns = []
        for pattern_name, pattern_results in dimension_results.items():
            for methodology_name, result in pattern_results.items():
                if result.is_economically_relevant:
                    relevant_patterns.append(pattern_name)
                    break
        
        if relevant_patterns:
            recommendations.append(f"\n📊 {dimension_name.upper()} DIMENSION:")
            for pattern in set(relevant_patterns):
                pattern_clean = pattern.replace('_', ' ').title()
                recommendations.append(f"   → Use for {pattern_clean} prediction")
    
    # Print recommendations
    for rec in recommendations:
        print(rec)
    
    return recommendations


def save_results(results: Dict, analysis: Dict, recommendations: List[str], output_dir: Path):
    """Save research results to files."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Save main report
    with open(output_dir / "economic_relevance_report.txt", "w") as f:
        f.write(results.get('main_report', ''))
    
    # Save volatility report
    with open(output_dir / "volatility_impact_report.txt", "w") as f:
        f.write(results.get('volatility_report', ''))
    
    # Save microstructure report
    with open(output_dir / "microstructure_impact_report.txt", "w") as f:
        f.write(results.get('microstructure_report', ''))
    
    # Save analysis summary
    with open(output_dir / "research_analysis_summary.txt", "w") as f:
        f.write("# Research Analysis Summary\n\n")
        f.write(f"Overall Economic Relevance Rate: {analysis.get('overall_relevance_rate', 0):.1f}%\n")
        f.write(f"Total Tests Conducted: {analysis.get('total_tests', 0)}\n")
        f.write(f"Economically Relevant Results: {analysis.get('relevant_tests', 0)}\n\n")
        
        f.write("## Dimension Rankings:\n")
        for dim_name, score in analysis.get('dimension_rankings', []):
            f.write(f"- {dim_name}: {score:.1%}\n")
        
        f.write("\n## Trading Recommendations:\n")
        for rec in recommendations:
            f.write(f"{rec}\n")
    
    print(f"\n💾 Results saved to: {output_dir}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Economic Relevance Research Implementation Example")
    parser.add_argument("--data_path", type=str, help="Path to market data CSV file")
    parser.add_argument("--output_dir", type=str, default="research_results", help="Output directory for results")
    parser.add_argument("--use_sample_data", action="store_true", help="Use generated sample data instead of file")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Economic Relevance Research Framework - Implementation Example")
    print("=" * 80)
    
    # 1. Load or generate data
    if args.use_sample_data or not args.data_path:
        print("📊 Generating sample market data...")
        market_data = create_sample_market_data(2000)
        print(f"   Generated {len(market_data)} periods of sample data")
    else:
        print(f"📊 Loading market data from {args.data_path}...")
        market_data = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
        print(f"   Loaded {len(market_data)} periods of market data")
    
    # 2. Create dimension features
    print("🔧 Creating dimension features...")
    dimension_features = create_sample_dimension_features(market_data)
    
    feature_counts = {name: len(features.columns) for name, features in dimension_features.items()}
    print(f"   Created features: {feature_counts}")
    
    # 3. Run comprehensive research
    try:
        results = run_comprehensive_economic_relevance_research(market_data, dimension_features)
        
        # 4. Analyze results
        analysis = analyze_research_results(results)
        
        # 5. Generate recommendations
        recommendations = generate_trading_recommendations(results, analysis)
        
        # 6. Save results
        output_dir = Path(args.output_dir)
        save_results(results, analysis, recommendations, output_dir)
        
        print("\n🎉 Economic relevance research completed successfully!")
        print(f"📊 Overall relevance rate: {analysis.get('overall_relevance_rate', 0):.1f}%")
        
        if analysis.get('dimension_rankings'):
            top_dim = analysis['dimension_rankings'][0]
            print(f"🏆 Top performing dimension: {top_dim[0]} ({top_dim[1]:.1%} relevance)")
        
    except Exception as e:
        print(f"❌ Research failed with error: {e}")
        logging.exception("Research execution failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())