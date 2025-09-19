"""
Pure Price Action Pattern Discovery - Complete Implementation

This script demonstrates the complete pure price action pattern discovery framework
with both binary labels and gradient-based intensity measurements.

Focus: WHAT price does, not WHY it moves.

Usage:
    python run_pure_pattern_discovery.py --data_path /path/to/data.csv
    python run_pure_pattern_discovery.py --use_sample_data
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from core_patterns import PurePricePatternOrchestrator
from gradient_targets import GradientPatternTargetGenerator
from lstm_discovery import LSTMPricePatternDiscovery
from matrix_profile_discovery import MatrixProfileOrchestrator


def create_pure_price_sample_data(n_periods: int = 1500) -> pd.Series:
    """Create sample price data with embedded pure price action patterns."""
    
    np.random.seed(42)
    print(f"📊 Generating {n_periods} periods of pure price action sample data...")
    
    # Base price evolution
    returns = np.random.normal(0.0005, 0.015, n_periods)
    
    # Embed pure price action patterns
    
    # 1. Momentum persistence patterns
    print("   🚀 Embedding momentum persistence patterns...")
    for i in range(50, n_periods, 120):
        if i + 12 < n_periods:
            momentum_direction = np.random.choice([-1, 1])
            base_momentum = 0.006 * momentum_direction
            
            for j in range(12):
                if i + j < n_periods:
                    decay_factor = max(0.4, 1 - j * 0.05)
                    returns[i + j] = base_momentum * decay_factor + np.random.normal(0, 0.008)
    
    # 2. Price reversion patterns
    print("   🔄 Embedding price reversion patterns...")
    for i in range(80, n_periods, 150):
        if i + 15 < n_periods:
            # Create extreme deviation
            extreme_move = 0.035 * np.random.choice([-1, 1])
            returns[i] = extreme_move
            
            # Gradual reversion
            for j in range(1, 10):
                if i + j < n_periods:
                    reversion_strength = extreme_move * -0.35 * np.exp(-j * 0.25)
                    returns[i + j] = reversion_strength + np.random.normal(0, 0.010)
    
    # 3. Trend acceleration patterns
    print("   ⚡ Embedding trend acceleration patterns...")
    for i in range(100, n_periods, 180):
        if i + 10 < n_periods:
            direction = np.random.choice([-1, 1])
            base_return = 0.003 * direction
            
            for j in range(8):
                if i + j < n_periods:
                    acceleration_factor = 1 + j * 0.15  # Accelerating
                    returns[i + j] = base_return * acceleration_factor + np.random.normal(0, 0.008)
    
    # 4. Range breakout patterns
    print("   📊 Embedding range breakout patterns...")
    for i in range(150, n_periods, 200):
        if i + 25 < n_periods:
            # Consolidation period
            for j in range(15):
                if i + j < n_periods:
                    returns[i + j] = np.random.normal(0, 0.005)  # Low volatility
            
            # Breakout
            breakout_direction = np.random.choice([-1, 1])
            breakout_return = 0.025 * breakout_direction
            
            for j in range(15, 22):
                if i + j < n_periods:
                    continuation_factor = max(0.7, 1 - (j - 15) * 0.1)
                    returns[i + j] = breakout_return * continuation_factor + np.random.normal(0, 0.012)
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    price_series = pd.Series(prices, index=pd.date_range('2020-01-01', periods=n_periods, freq='D'))
    
    print(f"   ✅ Generated price series: min={prices.min():.2f}, max={prices.max():.2f}")
    return price_series


def run_core_pattern_discovery(prices: pd.Series) -> Dict[str, Any]:
    """Run core pure price action pattern discovery."""
    
    print("\n🎯 CORE PURE PRICE PATTERN DISCOVERY")
    print("=" * 60)
    
    orchestrator = PurePricePatternOrchestrator()
    
    # Discover core patterns
    core_results = orchestrator.discover_all_pure_patterns(prices)
    
    # Export binary labels
    binary_targets = orchestrator.export_binary_labels(core_results)
    
    # Export intensity gradients
    intensity_targets = orchestrator.export_intensity_gradients(core_results)
    
    # Combined targets
    combined_targets = orchestrator.export_combined_targets(core_results)
    
    # Summary
    valid_patterns = sum(1 for result in core_results.values() if result.is_valid_pattern)
    print(f"📊 Core Pattern Results: {valid_patterns}/{len(core_results)} valid patterns")
    
    for pattern_name, result in core_results.items():
        if result.is_valid_pattern:
            max_intensity = result.intensity.max()
            print(f"   ✅ {pattern_name}: {result.frequency:.3f} frequency, {max_intensity:.3f} max intensity")
        else:
            print(f"   ❌ {pattern_name}: {result.frequency:.3f} frequency (invalid)")
    
    return {
        'results': core_results,
        'binary_targets': binary_targets,
        'intensity_targets': intensity_targets,
        'combined_targets': combined_targets
    }


def run_gradient_target_generation(prices: pd.Series) -> Dict[str, Any]:
    """Run gradient-based target generation."""
    
    print("\n📈 GRADIENT-BASED TARGET GENERATION")
    print("=" * 60)
    
    generator = GradientPatternTargetGenerator()
    
    # Generate gradient targets
    gradient_results = generator.generate_all_gradient_targets(prices)
    
    # Export ML-ready targets
    ml_exports = generator.export_ml_ready_targets(gradient_results)
    
    # Generate report
    gradient_report = generator.generate_gradient_report(gradient_results)
    
    # Summary
    print(f"📊 Gradient Target Results: {len(gradient_results)} patterns processed")
    
    for pattern_name, measurement in gradient_results.items():
        binary_count = measurement.binary_labels.sum()
        intensity_count = measurement.intensity_statistics['non_zero_count']
        correlation = measurement.correlation_with_outcomes
        
        print(f"   📈 {pattern_name}:")
        print(f"      Binary: {binary_count} occurrences")
        print(f"      Intensity: {intensity_count} non-zero ({intensity_count-binary_count:+d} additional)")
        print(f"      Future correlation: {correlation:.3f}")
    
    return {
        'gradient_results': gradient_results,
        'ml_exports': ml_exports,
        'gradient_report': gradient_report
    }


def run_lstm_pattern_discovery(prices: pd.Series) -> Dict[str, Any]:
    """Run LSTM-based pattern discovery."""
    
    print("\n🤖 LSTM-BASED PATTERN DISCOVERY")
    print("=" * 60)
    
    discoverer = LSTMPricePatternDiscovery(sequence_length=25, latent_dim=8)
    
    try:
        # Discover LSTM patterns
        lstm_patterns = discoverer.discover_lstm_patterns(prices)
        
        # Generate report
        lstm_report = discoverer.generate_lstm_pattern_report(lstm_patterns)
        
        # Summary
        significant_patterns = sum(1 for p in lstm_patterns if p.is_significant)
        print(f"📊 LSTM Discovery Results: {significant_patterns}/{len(lstm_patterns)} significant patterns")
        
        for pattern in lstm_patterns:
            if pattern.is_significant:
                print(f"   ✅ {pattern.pattern_id}: {pattern.frequency:.3f} frequency")
                print(f"      {pattern.description}")
        
        return {
            'lstm_patterns': lstm_patterns,
            'lstm_report': lstm_report
        }
        
    except Exception as e:
        print(f"   ⚠️ LSTM discovery failed: {e}")
        print("   Note: This is expected in simulation mode")
        return {
            'lstm_patterns': [],
            'lstm_report': "LSTM discovery requires TensorFlow/PyTorch implementation"
        }


def run_matrix_profile_discovery(prices: pd.Series) -> Dict[str, Any]:
    """Run matrix profile-based pattern discovery."""
    
    print("\n📊 MATRIX PROFILE PATTERN DISCOVERY")
    print("=" * 60)
    
    orchestrator = MatrixProfileOrchestrator()
    
    try:
        # Run matrix profile analysis
        mp_results = orchestrator.run_complete_matrix_profile_analysis(
            prices, motif_lengths=[15, 20, 25]
        )
        
        # Export targets
        mp_targets = orchestrator.export_matrix_profile_targets(mp_results)
        
        # Summary
        total_patterns = sum(len(patterns) for patterns in mp_results.values())
        significant_patterns = sum(
            sum(1 for p in patterns if p.is_significant)
            for patterns in mp_results.values()
        )
        
        print(f"📊 Matrix Profile Results: {significant_patterns}/{total_patterns} significant patterns")
        
        for length_key, patterns in mp_results.items():
            significant_count = sum(1 for p in patterns if p.is_significant)
            print(f"   📈 {length_key}: {significant_count}/{len(patterns)} significant")
        
        return {
            'mp_results': mp_results,
            'mp_targets': mp_targets
        }
        
    except Exception as e:
        print(f"   ⚠️ Matrix profile discovery failed: {e}")
        print("   Note: This is expected in simulation mode")
        return {
            'mp_results': {},
            'mp_targets': {}
        }


def generate_comprehensive_analysis(core_results: Dict[str, Any],
                                  gradient_results: Dict[str, Any],
                                  lstm_results: Dict[str, Any],
                                  mp_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive analysis of all pure price pattern discoveries."""
    
    print("\n📈 COMPREHENSIVE PURE PRICE PATTERN ANALYSIS")
    print("=" * 60)
    
    analysis = {
        'summary': {},
        'target_comparison': {},
        'ml_readiness': {},
        'recommendations': []
    }
    
    # Summary statistics
    core_valid = sum(1 for result in core_results['results'].values() if result.is_valid_pattern)
    lstm_significant = len([p for p in lstm_results.get('lstm_patterns', []) if p.is_significant])
    mp_significant = sum(
        sum(1 for p in patterns if p.is_significant)
        for patterns in mp_results.get('mp_results', {}).values()
    )
    
    analysis['summary'] = {
        'core_patterns_valid': core_valid,
        'lstm_patterns_significant': lstm_significant,
        'matrix_profile_patterns_significant': mp_significant,
        'total_valid_patterns': core_valid + lstm_significant + mp_significant
    }
    
    print(f"📊 PURE PRICE PATTERN SUMMARY:")
    print(f"   Core Patterns Valid: {core_valid}")
    print(f"   LSTM Patterns Significant: {lstm_significant}")
    print(f"   Matrix Profile Patterns Significant: {mp_significant}")
    print(f"   Total Valid Patterns: {analysis['summary']['total_valid_patterns']}")
    
    # Target comparison (binary vs gradient)
    if not core_results['binary_targets'].empty and not core_results['intensity_targets'].empty:
        binary_df = core_results['binary_targets']
        intensity_df = core_results['intensity_targets']
        
        target_comparison = {}
        for col in binary_df.columns:
            binary_count = binary_df[col].sum()
            intensity_col = f"{col}_intensity"
            
            if intensity_col in intensity_df.columns:
                intensity_count = (intensity_df[intensity_col] > 0).sum()
                additional_weak = intensity_count - binary_count
                
                target_comparison[col] = {
                    'binary_patterns': int(binary_count),
                    'intensity_patterns': int(intensity_count),
                    'additional_weak_patterns': int(additional_weak),
                    'enhancement_ratio': intensity_count / binary_count if binary_count > 0 else 0
                }
        
        analysis['target_comparison'] = target_comparison
        
        print(f"\n📈 BINARY vs GRADIENT TARGET COMPARISON:")
        for pattern, stats in target_comparison.items():
            print(f"   {pattern}:")
            print(f"      Binary: {stats['binary_patterns']} patterns")
            print(f"      Gradient: {stats['intensity_patterns']} patterns ({stats['additional_weak_patterns']:+d} additional)")
            print(f"      Enhancement: {stats['enhancement_ratio']:.1f}x more patterns captured")
    
    # ML readiness assessment
    ml_readiness = {
        'binary_classification_ready': not core_results['binary_targets'].empty,
        'regression_ready': not core_results['intensity_targets'].empty,
        'multi_task_ready': not core_results['combined_targets'].empty,
        'pattern_count': len(core_results['binary_targets'].columns) if not core_results['binary_targets'].empty else 0
    }
    
    analysis['ml_readiness'] = ml_readiness
    
    print(f"\n🎯 ML READINESS ASSESSMENT:")
    print(f"   Binary Classification: {'✅' if ml_readiness['binary_classification_ready'] else '❌'}")
    print(f"   Regression Targets: {'✅' if ml_readiness['regression_ready'] else '❌'}")
    print(f"   Multi-Task Learning: {'✅' if ml_readiness['multi_task_ready'] else '❌'}")
    print(f"   Available Patterns: {ml_readiness['pattern_count']}")
    
    # Generate recommendations
    analysis['recommendations'] = generate_recommendations(analysis)
    
    return analysis


def generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on analysis results."""
    
    recommendations = []
    
    total_patterns = analysis['summary']['total_valid_patterns']
    core_patterns = analysis['summary']['core_patterns_valid']
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if total_patterns >= 5:
        recommendations.append("✅ STRONG PURE PRICE PATTERN FOUNDATION")
        recommendations.append("   → Multiple valid patterns discovered")
        recommendations.append("   → Proceed with pattern-based ML model development")
        recommendations.append("   → Use both binary and gradient targets for enhanced training")
        print("   ✅ Strong foundation - proceed with full ML implementation")
        
    elif total_patterns >= 3:
        recommendations.append("⚠️ MODERATE PURE PRICE PATTERN FOUNDATION")
        recommendations.append("   → Some valid patterns found")
        recommendations.append("   → Focus on highest-frequency patterns")
        recommendations.append("   → Consider parameter optimization for failed patterns")
        print("   ⚠️ Moderate foundation - focus on best patterns")
        
    else:
        recommendations.append("❌ LIMITED PURE PRICE PATTERN FOUNDATION")
        recommendations.append("   → Few valid patterns discovered")
        recommendations.append("   → Consider different timeframes or longer data series")
        recommendations.append("   → Adjust pattern sensitivity parameters")
        print("   ❌ Limited foundation - consider data/parameter adjustments")
    
    # Specific recommendations based on target types
    if analysis['ml_readiness']['regression_ready']:
        recommendations.append("\n📈 GRADIENT TARGET BENEFITS:")
        recommendations.append("   → Use intensity gradients for regression models")
        recommendations.append("   → Scale trading positions by pattern intensity")
        recommendations.append("   → Implement confidence-based trading strategies")
        print("   📈 Gradient targets enable enhanced ML training")
    
    if core_patterns >= 3:
        recommendations.append("\n🎯 CORE PATTERN APPLICATIONS:")
        recommendations.append("   → Test which market dimensions predict these patterns")
        recommendations.append("   → Develop pattern-specific trading strategies")
        recommendations.append("   → Validate economic significance through backtesting")
        print("   🎯 Core patterns ready for dimension relevance testing")
    
    # Next steps
    recommendations.extend([
        "\n🚀 NEXT STEPS:",
        "1. Test pattern predictability using market dimension features",
        "2. Implement LSTM and Matrix Profile discovery (requires TensorFlow/stumpy)",
        "3. Validate economic significance of discovered patterns",
        "4. Develop pattern-based trading strategies"
    ])
    
    for rec in recommendations[recommendations.index("\n🚀 NEXT STEPS:"):]:
        print(rec)
    
    return recommendations


def save_pure_pattern_results(results: Dict[str, Any], output_dir: Path):
    """Save pure price pattern results."""
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving pure price pattern results to {output_dir}")
    
    # Save binary targets
    if not results['core']['binary_targets'].empty:
        results['core']['binary_targets'].to_csv(output_dir / "binary_pattern_targets.csv")
        print("   ✅ Binary pattern targets saved")
    
    # Save intensity targets  
    if not results['core']['intensity_targets'].empty:
        results['core']['intensity_targets'].to_csv(output_dir / "intensity_pattern_targets.csv")
        print("   ✅ Intensity pattern targets saved")
    
    # Save combined targets
    if not results['core']['combined_targets'].empty:
        results['core']['combined_targets'].to_csv(output_dir / "combined_pattern_targets.csv")
        print("   ✅ Combined pattern targets saved")
    
    # Save gradient report
    if 'gradient_report' in results['gradient']:
        with open(output_dir / "gradient_targets_report.md", "w") as f:
            f.write(results['gradient']['gradient_report'])
        print("   ✅ Gradient targets report saved")
    
    # Save analysis summary
    analysis_summary = {
        'summary': results['analysis']['summary'],
        'target_comparison': results['analysis']['target_comparison'],
        'ml_readiness': results['analysis']['ml_readiness'],
        'recommendations': results['analysis']['recommendations']
    }
    
    with open(output_dir / "pure_pattern_analysis.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)
    print("   ✅ Analysis summary saved")
    
    # Save pattern definitions
    core_patterns = results['core']['results']
    pattern_definitions = {}
    
    for name, result in core_patterns.items():
        if result.is_valid_pattern:
            pattern_definitions[name] = {
                'description': result.definition.description,
                'formula': result.definition.mathematical_formula,
                'frequency': result.frequency,
                'parameters': result.definition.parameters
            }
    
    with open(output_dir / "pure_pattern_definitions.json", "w") as f:
        json.dump(pattern_definitions, f, indent=2)
    print("   ✅ Pattern definitions saved")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Pure Price Action Pattern Discovery")
    parser.add_argument("--data_path", type=str, help="Path to price data CSV file")
    parser.add_argument("--output_dir", type=str, default="pure_pattern_results", help="Output directory")
    parser.add_argument("--use_sample_data", action="store_true", help="Use generated sample data")
    parser.add_argument("--skip_advanced", action="store_true", help="Skip LSTM/Matrix Profile (faster)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎯 PURE PRICE ACTION PATTERN DISCOVERY FRAMEWORK")
    print("=" * 80)
    print("Focus: WHAT price does (not WHY it moves)")
    print("Output: Binary labels + Intensity gradients for ML")
    print("=" * 80)
    
    # Load or generate data
    if args.use_sample_data or not args.data_path:
        prices = create_pure_price_sample_data(1500)
    else:
        print(f"📊 Loading price data from {args.data_path}...")
        data = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
        if 'close' in data.columns:
            prices = data['close']
        else:
            prices = data.iloc[:, 0]  # First column
        print(f"   Loaded {len(prices)} price points")
    
    results = {}
    
    try:
        # 1. Core Pattern Discovery
        results['core'] = run_core_pattern_discovery(prices)
        
        # 2. Gradient Target Generation
        results['gradient'] = run_gradient_target_generation(prices)
        
        # 3. Advanced ML Discovery (if not skipped)
        if not args.skip_advanced:
            results['lstm'] = run_lstm_pattern_discovery(prices)
            results['matrix_profile'] = run_matrix_profile_discovery(prices)
        else:
            print("\n🤖 Advanced ML discovery skipped")
            results['lstm'] = {'lstm_patterns': [], 'lstm_report': 'Skipped'}
            results['matrix_profile'] = {'mp_results': {}, 'mp_targets': {}}
        
        # 4. Comprehensive Analysis
        results['analysis'] = generate_comprehensive_analysis(
            results['core'], results['gradient'], results['lstm'], results['matrix_profile']
        )
        
        # 5. Save Results
        output_dir = Path(args.output_dir)
        save_pure_pattern_results(results, output_dir)
        
        print(f"\n🎉 PURE PRICE PATTERN DISCOVERY COMPLETED!")
        print("=" * 80)
        
        summary = results['analysis']['summary']
        print(f"📊 FINAL RESULTS:")
        print(f"   Total Valid Patterns: {summary['total_valid_patterns']}")
        print(f"   Core Patterns: {summary['core_patterns_valid']}")
        print(f"   LSTM Patterns: {summary['lstm_patterns_significant']}")
        print(f"   Matrix Profile Patterns: {summary['matrix_profile_patterns_significant']}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   🎯 Binary targets: binary_pattern_targets.csv")
        print(f"   📈 Intensity targets: intensity_pattern_targets.csv")
        print(f"   🔗 Combined targets: combined_pattern_targets.csv")
        print(f"   📊 Analysis: pure_pattern_analysis.json")
        
        return 0
        
    except Exception as e:
        print(f"❌ Pure pattern discovery failed: {e}")
        logging.exception("Pattern discovery execution failed")
        return 1


if __name__ == "__main__":
    exit(main())