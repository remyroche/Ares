"""
Complete Pattern Discovery Implementation Script

This script demonstrates the complete pattern discovery framework with all 18+
mathematical pattern definitions and ML-based discovery methods.

Usage:
    python run_complete_pattern_discovery.py --data_path /path/to/data.csv
    python run_complete_pattern_discovery.py --use_sample_data
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import pattern discovery frameworks
from pattern_discovery_framework import PatternDiscoveryOrchestrator
from advanced_pattern_definitions import AdvancedPatternDiscoveryOrchestrator
from ml_pattern_discovery import MLPatternDiscoveryOrchestrator

def create_comprehensive_sample_data(n_periods: int = 2000) -> pd.DataFrame:
    """Create comprehensive sample data with embedded patterns."""

    np.random.seed(42)
    print(f"📊 Generating {n_periods} periods of sample market data with embedded patterns...")

    # Base price evolution
    returns = np.random.normal(0.0005, 0.02, n_periods)

    # Embed momentum persistence patterns (every 80-120 periods)
    print("   🚀 Embedding momentum persistence patterns...")
    for i in range(50, n_periods, 100):
        if i + 15 < n_periods:
            momentum_direction = np.random.choice([-1, 1])
            base_momentum = 0.008 * momentum_direction

            for j in range(12):  # 12-period momentum persistence
                if i + j < n_periods:
                    decay_factor = max(0.4, 1 - j * 0.05)  # Gradual decay
                    returns[i + j] = base_momentum * decay_factor + np.random.normal(0, 0.008)

    # Embed mean reversion patterns (every 150-200 periods)
    print("   🔄 Embedding mean reversion patterns...")
    for i in range(80, n_periods, 175):
        if i + 15 < n_periods:
            # Create oversold/overbought condition
            extreme_move = 0.04 * np.random.choice([-1, 1])
            returns[i] = extreme_move

            # Gradual reversion over next periods
            for j in range(1, 10):
                if i + j < n_periods:
                    reversion_strength = extreme_move * -0.4 * np.exp(-j * 0.3)
                    returns[i + j] = reversion_strength + np.random.normal(0, 0.012)

    # Embed volatility expansion patterns (every 200-300 periods)
    print("   📈 Embedding volatility expansion patterns...")
    for i in range(150, n_periods, 250):
        if i + 25 < n_periods:
            # Low volatility period
            for j in range(8):
                if i + j < n_periods:
                    returns[i + j] = np.random.normal(0, 0.003)  # Very low vol

            # High volatility expansion
            for j in range(8, 18):
                if i + j < n_periods:
                    returns[i + j] = np.random.normal(0, 0.045)  # High vol expansion

    # Embed trend continuation patterns (every 120-180 periods)
    print("   📊 Embedding trend continuation patterns...")
    for i in range(100, n_periods, 150):
        if i + 25 < n_periods:
            trend_direction = np.random.choice([-1, 1])
            base_trend = 0.003 * trend_direction

            for j in range(20):  # 20-period trend
                if i + j < n_periods:
                    trend_strength = base_trend * (1 + j * 0.02)  # Slightly accelerating
                    returns[i + j] = trend_strength + np.random.normal(0, 0.015)

    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC with realistic patterns
    print("   📋 Generating OHLC data...")
    high = prices * (1 + np.abs(np.random.normal(0, 0.008, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.008, n_periods)))

    # Volume with correlation to volatility and patterns
    base_volume = 1000000
    vol_factor = abs(returns) * 50000000  # Volume increases with volatility
    pattern_factor = np.random.normal(0, 200000, n_periods)  # Random component
    volume = base_volume + vol_factor + pattern_factor
    volume = np.maximum(volume, 50000)  # Minimum volume

    market_data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })

    market_data['open'].iloc[0] = prices[0]
    market_data.index = pd.date_range('2020-01-01', periods=n_periods, freq='D')

    print(f"   ✅ Generated market data: {market_data.shape}")
    return market_data

def run_basic_pattern_discovery(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Run basic pattern discovery (5 core patterns)."""

    print("\n🔬 BASIC PATTERN DISCOVERY (5 Core Patterns)")
    print("=" * 60)

    orchestrator = PatternDiscoveryOrchestrator()

    # Discover basic patterns
    basic_results = orchestrator.discover_all_patterns(market_data['close'])

    # Generate report
    basic_report = orchestrator.generate_pattern_report(basic_results)

    # Export ML targets
    basic_targets = orchestrator.export_pattern_labels(basic_results)

    # Summary
    valid_basic = sum(1 for result in basic_results.values() if result.is_valid_pattern)
    print(f"📊 Basic Pattern Results: {valid_basic}/{len(basic_results)} valid patterns")

    for pattern_name, result in basic_results.items():
        status = "✅" if result.is_valid_pattern else "❌"
        print(f"   {status} {pattern_name}: {result.frequency:.3f} frequency")

    return {
        'results': basic_results,
        'report': basic_report,
        'targets': basic_targets
    }

def run_advanced_pattern_discovery(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Run advanced pattern discovery (7 sophisticated patterns)."""

    print("\n⚡ ADVANCED PATTERN DISCOVERY (7 Sophisticated Patterns)")
    print("=" * 60)

    orchestrator = AdvancedPatternDiscoveryOrchestrator()

    # Discover advanced patterns
    advanced_results = orchestrator.discover_all_advanced_patterns(market_data)

    # Summary
    valid_advanced = sum(1 for result in advanced_results.values() if result.is_valid_pattern)
    print(f"📊 Advanced Pattern Results: {valid_advanced}/{len(advanced_results)} valid patterns")

    for pattern_name, result in advanced_results.items():
        status = "✅" if result.is_valid_pattern else "❌"
        print(f"   {status} {pattern_name}: {result.frequency:.3f} frequency")

    return {
        'results': advanced_results
    }

def run_ml_pattern_discovery(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Run ML-based pattern discovery."""

    print("\n🤖 ML-BASED PATTERN DISCOVERY")
    print("=" * 60)

    orchestrator = MLPatternDiscoveryOrchestrator()

    # Discover ML patterns
    ml_results = orchestrator.discover_all_ml_patterns(market_data)

    # Generate report
    ml_report = orchestrator.generate_ml_pattern_report(ml_results)

    # Summary
    total_ml_patterns = sum(len(patterns) for patterns in ml_results.values())
    significant_ml_patterns = sum(
        sum(1 for p in patterns if p.is_significant_pattern)
        for patterns in ml_results.values()
    )

    print(f"📊 ML Pattern Results: {significant_ml_patterns}/{total_ml_patterns} significant patterns")

    for method_name, patterns in ml_results.items():
        significant_count = sum(1 for p in patterns if p.is_significant_pattern)
        print(f"   {method_name}: {significant_count}/{len(patterns)} significant")

    return {
        'results': ml_results,
        'report': ml_report
    }

def generate_comprehensive_analysis(basic_results: Dict[str, Any],
                                  advanced_results: Dict[str, Any],
                                  ml_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive analysis of all discovered patterns."""

    print("\n📈 COMPREHENSIVE PATTERN ANALYSIS")
    print("=" * 60)

    analysis = {
        'summary': {},
        'pattern_catalog': {},
        'ml_targets': {},
        'recommendations': []
    }

    # Combine all valid patterns
    all_valid_patterns = {}

    # Basic patterns
    for name, result in basic_results['results'].items():
        if result.is_valid_pattern:
            all_valid_patterns[f"basic_{name}"] = result

    # Advanced patterns
    for name, result in advanced_results['results'].items():
        if result.is_valid_pattern:
            all_valid_patterns[f"advanced_{name}"] = result

    # ML patterns
    for method_name, patterns in ml_results['results'].items():
        for pattern in patterns:
            if pattern.is_significant_pattern:
                all_valid_patterns[f"ml_{pattern.pattern_id}"] = pattern

    # Summary statistics
    analysis['summary'] = {
        'total_patterns_tested': (
            len(basic_results['results']) +
            len(advanced_results['results']) +
            sum(len(patterns) for patterns in ml_results['results'].values())
        ),
        'valid_patterns_found': len(all_valid_patterns),
        'basic_patterns_valid': sum(1 for result in basic_results['results'].values() if result.is_valid_pattern),
        'advanced_patterns_valid': sum(1 for result in advanced_results['results'].values() if result.is_valid_pattern),
        'ml_patterns_significant': sum(
            sum(1 for p in patterns if p.is_significant_pattern)
            for patterns in ml_results['results'].values()
        )
    }

    print(f"📊 PATTERN DISCOVERY SUMMARY:")
    print(f"   Total Patterns Tested: {analysis['summary']['total_patterns_tested']}")
    print(f"   Valid Patterns Found: {analysis['summary']['valid_patterns_found']}")
    print(f"   Success Rate: {analysis['summary']['valid_patterns_found']/analysis['summary']['total_patterns_tested']*100:.1f}%")

    # Pattern frequency analysis
    print(f"\n📈 PATTERN FREQUENCY ANALYSIS:")

    pattern_frequencies = []
    for name, pattern in all_valid_patterns.items():
        if hasattr(pattern, 'frequency'):
            freq = pattern.frequency
        else:
            freq = pattern.pattern_labels.sum() / len(pattern.pattern_labels)

        pattern_frequencies.append((name, freq))

    # Sort by frequency
    pattern_frequencies.sort(key=lambda x: x[1], reverse=True)

    print("   Top 5 Most Frequent Patterns:")
    for i, (name, freq) in enumerate(pattern_frequencies[:5], 1):
        print(f"   {i}. {name}: {freq:.3f} ({freq*100:.1f}%)")

    # Generate ML targets
    if basic_results.get('targets') is not None and len(basic_results['targets']) > 0:
        analysis['ml_targets'] = basic_results['targets']
        print(f"\n🎯 ML TARGETS GENERATED:")
        print(f"   Target Matrix Shape: {basic_results['targets'].shape}")
        print(f"   Available Targets: {list(basic_results['targets'].columns)}")

    # Recommendations
    analysis['recommendations'] = generate_recommendations(analysis['summary'], all_valid_patterns)

    return analysis

def generate_recommendations(summary: Dict[str, Any],
                           all_valid_patterns: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on pattern discovery results."""

    recommendations = []

    success_rate = summary['valid_patterns_found'] / summary['total_patterns_tested']

    print(f"\n💡 RECOMMENDATIONS:")

    if success_rate > 0.6:
        recommendations.append("✅ EXCELLENT PATTERN DISCOVERY SUCCESS")
        recommendations.append("   → Multiple valid patterns found across all categories")
        recommendations.append("   → Proceed with full pattern-based ML model development")
        recommendations.append("   → Use patterns for market dimension relevance testing")
        print("   ✅ Excellent success rate - proceed with full implementation")

    elif success_rate > 0.4:
        recommendations.append("⚠️ MODERATE PATTERN DISCOVERY SUCCESS")
        recommendations.append("   → Some valid patterns found")
        recommendations.append("   → Focus on highest-frequency patterns for ML training")
        recommendations.append("   → Consider parameter optimization for failed patterns")
        print("   ⚠️ Moderate success - focus on best patterns")

    else:
        recommendations.append("❌ LIMITED PATTERN DISCOVERY SUCCESS")
        recommendations.append("   → Few valid patterns found")
        recommendations.append("   → Consider alternative data sources or timeframes")
        recommendations.append("   → Focus on ML-based discovery methods")
        print("   ❌ Limited success - consider data/parameter adjustments")

    # Specific recommendations based on pattern types
    basic_valid = summary['basic_patterns_valid']
    advanced_valid = summary['advanced_patterns_valid']
    ml_valid = summary['ml_patterns_significant']

    if basic_valid >= 3:
        recommendations.append("📊 BASIC PATTERNS: Strong foundation detected")
        recommendations.append("   → Use basic patterns as primary ML targets")
        print("   📊 Strong basic pattern foundation")

    if advanced_valid >= 3:
        recommendations.append("⚡ ADVANCED PATTERNS: Sophisticated patterns detected")
        recommendations.append("   → Incorporate advanced patterns for enhanced strategies")
        print("   ⚡ Advanced patterns add sophistication")

    if ml_valid >= 2:
        recommendations.append("🤖 ML DISCOVERY: Data-driven patterns found")
        recommendations.append("   → Validate ML patterns for economic significance")
        print("   🤖 ML methods discovered additional patterns")

    # Next steps
    recommendations.extend([
        "\n🚀 NEXT STEPS:",
        "1. Test pattern predictability using market dimension features",
        "2. Validate economic significance through backtesting",
        "3. Integrate patterns with existing market analysis pipeline",
        "4. Develop pattern-specific trading strategies"
    ])

    for rec in recommendations[recommendations.index("\n🚀 NEXT STEPS:"):]:
        print(rec)

    return recommendations

def save_comprehensive_results(results: Dict[str, Any], output_dir: Path):
    """Save all results to files."""

    output_dir.mkdir(exist_ok=True)

    print(f"\n💾 Saving results to {output_dir}")

    # Save basic pattern report
    if 'basic' in results and 'report' in results['basic']:
        with open(output_dir / "basic_pattern_report.md", "w") as f:
            f.write(results['basic']['report'])
        print("   ✅ Basic pattern report saved")

    # Save ML pattern report
    if 'ml' in results and 'report' in results['ml']:
        with open(output_dir / "ml_pattern_report.md", "w") as f:
            f.write(results['ml']['report'])
        print("   ✅ ML pattern report saved")

    # Save ML targets
    if 'analysis' in results and 'ml_targets' in results['analysis']:
        targets_df = results['analysis']['ml_targets']
        if not targets_df.empty:
            targets_df.to_csv(output_dir / "ml_targets.csv")
            print("   ✅ ML targets saved to CSV")

    # Save comprehensive analysis
    if 'analysis' in results:
        analysis_summary = {
            'summary': results['analysis']['summary'],
            'recommendations': results['analysis']['recommendations']
        }

        with open(output_dir / "comprehensive_analysis.json", "w") as f:
            json.dump(analysis_summary, f, indent=2)
        print("   ✅ Comprehensive analysis saved")

    # Save pattern catalog
    pattern_catalog = {}

    if 'basic' in results:
        for name, result in results['basic']['results'].items():
            if result.is_valid_pattern:
                pattern_catalog[f"basic_{name}"] = {
                    'frequency': result.frequency,
                    'predictability': result.predictability_score,
                    'description': result.definition.description,
                    'formula': result.definition.mathematical_formula
                }

    if 'advanced' in results:
        for name, result in results['advanced']['results'].items():
            if result.is_valid_pattern:
                pattern_catalog[f"advanced_{name}"] = {
                    'frequency': result.frequency,
                    'predictability': result.predictability_score,
                    'description': result.definition.description,
                    'formula': result.definition.mathematical_formula
                }

    with open(output_dir / "pattern_catalog.json", "w") as f:
        json.dump(pattern_catalog, f, indent=2)
    print("   ✅ Pattern catalog saved")

def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(description="Complete Pattern Discovery Implementation")
    parser.add_argument("--data_path", type=str, help="Path to market data CSV file")
    parser.add_argument("--output_dir", type=str, default="pattern_discovery_results", help="Output directory")
    parser.add_argument("--use_sample_data", action="store_true", help="Use generated sample data")
    parser.add_argument("--skip_ml", action="store_true", help="Skip ML-based discovery (faster)")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🚀 COMPLETE PATTERN DISCOVERY FRAMEWORK")
    print("=" * 80)
    print("🎯 Mathematical Pattern Precision + ML-Based Discovery")
    print("=" * 80)

    # Load or generate data
    if args.use_sample_data or not args.data_path:
        market_data = create_comprehensive_sample_data(2000)
    else:
        print(f"📊 Loading market data from {args.data_path}...")
        market_data = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
        print(f"   Loaded {len(market_data)} periods")

    results = {}

    try:
        # 1. Basic Pattern Discovery
        results['basic'] = run_basic_pattern_discovery(market_data)

        # 2. Advanced Pattern Discovery
        results['advanced'] = run_advanced_pattern_discovery(market_data)

        # 3. ML-Based Pattern Discovery
        if not args.skip_ml:
            results['ml'] = run_ml_pattern_discovery(market_data)
        else:
            print("\n🤖 ML-based discovery skipped")
            results['ml'] = {'results': {}, 'report': 'Skipped'}

        # 4. Comprehensive Analysis
        results['analysis'] = generate_comprehensive_analysis(
            results['basic'], results['advanced'], results['ml']
        )

        # 5. Save Results
        output_dir = Path(args.output_dir)
        save_comprehensive_results(results, output_dir)

        print(f"\n🎉 PATTERN DISCOVERY COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        summary = results['analysis']['summary']
        print(f"📊 FINAL RESULTS:")
        print(f"   Total Patterns: {summary['total_patterns_tested']}")
        print(f"   Valid Patterns: {summary['valid_patterns_found']}")
        print(f"   Success Rate: {summary['valid_patterns_found']/summary['total_patterns_tested']*100:.1f}%")

        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   📋 Pattern reports: *.md files")
        print(f"   🎯 ML targets: ml_targets.csv")
        print(f"   📊 Analysis: comprehensive_analysis.json")

        return 0

    except Exception as e:
        print(f"❌ Pattern discovery failed: {e}")
        logging.exception("Pattern discovery execution failed")
        return 1

if __name__ == "__main__":
    exit(main())
