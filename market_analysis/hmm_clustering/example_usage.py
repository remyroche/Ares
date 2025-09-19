#!/usr/bin/env python3
"""
Example Usage of Enhanced HMM Clustering for Market Analysis

This script demonstrates how to use the enhanced HMM clustering system
with all common utilities integrated for optimal performance.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import the enhanced HMM clustering
from enhanced_hmm_clustering import (
    EnhancedHMMClustering, 
    HMMClusteringConfig, 
    RegimeType,
    run_hmm_clustering_analysis
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_custom_config():
    """Create a custom configuration for HMM clustering with robust settings."""
    return HMMClusteringConfig(
        # HMM Parameters
        n_components=4,  # 4 market regimes (reduced for stability)
        covariance_type="full",
        n_iter=150,  # Reduced for faster convergence
        random_state=42,
        
        # Feature Engineering
        lookback_windows=[5, 10, 20, 50],  # Reduced for better performance
        technical_indicators=[
            "rsi", "macd", "bollinger_bands", "atr", "stochastic"
        ],
        
        # Optimization - more conservative settings
        use_gpu=False,  # Disabled for compatibility
        use_memory_optimization=False,  # Disabled for testing
        use_cpu_optimization=False,  # Disabled for compatibility
        
        # Cross-validation
        cv_folds=3,  # Reduced for faster execution
        test_size=0.2,
        purged_cv=True,
        
        # Feature Selection
        feature_selection_method="mrmr",
        max_features=25,  # Reduced for better performance
        
        # Data Processing
        min_data_points=1000,  # Reduced threshold
        max_missing_ratio=0.1,  # More tolerant
        
        # Regime Analysis
        min_regime_duration=15,
        regime_stability_threshold=0.7  # More tolerant
    )

def analyze_regime_transitions(result):
    """Analyze regime transitions and create visualizations."""
    try:
        regime_labels = result.regime_labels
        regime_probabilities = result.regime_probabilities
        
        # Create transition matrix
        n_regimes = len(np.unique(regime_labels))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            current_regime = regime_labels[i]
            next_regime = regime_labels[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize transition matrix with safety check
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
        
        # Create visualization with error handling
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            plt.figure(figsize=(12, 8))
        
        # Plot 1: Regime sequence
        plt.subplot(2, 2, 1)
        plt.plot(regime_labels, alpha=0.7)
        plt.title('Regime Sequence Over Time')
        plt.xlabel('Time')
        plt.ylabel('Regime')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Regime probabilities
        plt.subplot(2, 2, 2)
        for i in range(n_regimes):
            plt.plot(regime_probabilities[:, i], label=f'Regime {i}', alpha=0.7)
        plt.title('Regime Probabilities Over Time')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Transition matrix heatmap
        plt.subplot(2, 2, 3)
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Regime Transition Matrix')
        plt.xlabel('Next Regime')
        plt.ylabel('Current Regime')
        
        # Plot 4: Regime distribution
        plt.subplot(2, 2, 4)
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        plt.bar(unique_regimes, counts)
        plt.title('Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        results_dir = Path('market_analysis/hmm_clustering/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Regime analysis plot saved to {results_dir / 'regime_analysis.png'}")
        
        # Only show plot if in interactive environment
        try:
            plt.show()
        except Exception:
            logger.info("Non-interactive environment detected, plot saved only")
                
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualization")
        except Exception as viz_e:
            logger.error(f"Visualization error: {viz_e}")
        
        return transition_matrix
        
    except Exception as e:
        logger.error(f"Failed to analyze regime transitions: {e}")
        return None

def analyze_feature_importance(result):
    """Analyze and visualize feature importance."""
    try:
        feature_importance = result.feature_importance
        
        if not feature_importance:
            logger.warning("No feature importance data available")
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Top 20 features
        top_features = sorted_features[:20]
        features, importances = zip(*top_features)
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features for Regime Detection')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        results_dir = Path('market_analysis/hmm_clustering/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {results_dir / 'feature_importance.png'}")
        
        # Only show plot if in interactive environment
        try:
            plt.show()
        except Exception:
            logger.info("Non-interactive environment detected, plot saved only")
        
        # Print feature importance summary
        print("\nFeature Importance Summary:")
        print("=" * 50)
        for feature, importance in sorted_features[:10]:
            print(f"{feature:30s}: {importance:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to analyze feature importance: {e}")
        return

def analyze_regime_characteristics(result):
    """Analyze and display regime characteristics."""
    try:
        characteristics = result.regime_characteristics
        
        print("\nRegime Characteristics Analysis:")
        print("=" * 60)
        
        for regime, char in characteristics.items():
            print(f"\n{regime.upper()}:")
            print(f"  Count: {char.get('count', 0):,}")
            print(f"  Percentage: {char.get('percentage', 0):.2f}%")
            print(f"  Mean Returns: {char.get('mean_returns', 0):.4f}")
            print(f"  Volatility: {char.get('volatility', 0):.4f}")
            print(f"  Mean Price: {char.get('mean_price', 0):.2f}")
            
            # Technical indicators
            for key, value in char.items():
                if key.endswith('_mean') and not key.startswith('mean_'):
                    indicator = key.replace('_mean', '')
                    print(f"  {indicator}: {value:.4f}")
        
        # Performance metrics
        metrics = result.performance_metrics
        print(f"\nPerformance Metrics:")
        print(f"  Regime Stability: {metrics.get('regime_stability', 0):.4f}")
        print(f"  Regime Balance: {metrics.get('regime_balance', 0):.4f}")
        print(f"  Average Confidence: {metrics.get('avg_confidence', 0):.4f}")
        print(f"  Average Regime Duration: {metrics.get('avg_regime_duration', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Failed to analyze regime characteristics: {e}")
        return

def run_comprehensive_analysis():
    """Run a comprehensive HMM clustering analysis."""
    try:
        logger.info("Starting comprehensive HMM clustering analysis")
        
        # Create custom configuration
        config = create_custom_config()
        
        # Run analysis for different symbols and timeframes with error handling
        symbols = ["BTCUSDT"]  # Reduced for demo
        intervals = ["1h"]  # Reduced for demo
        
        results = {}
        
        for symbol in symbols:
            for interval in intervals:
                logger.info(f"Analyzing {symbol} {interval}")
                
                try:
                    result = run_hmm_clustering_analysis(
                        symbol=symbol,
                        interval=interval,
                        config=config,
                        save_results=True
                    )
                    
                    if result is not None:
                        results[f"{symbol}_{interval}"] = result
                        
                        # Analyze results
                        print(f"\n{'='*60}")
                        print(f"ANALYSIS RESULTS FOR {symbol} {interval}")
                        print(f"{'='*60}")
                        
                        analyze_regime_characteristics(result)
                        analyze_feature_importance(result)
                        
                        # Create visualizations with error handling
                        try:
                            transition_matrix = analyze_regime_transitions(result)
                        except Exception as viz_e:
                            logger.warning(f"Visualization failed: {viz_e}")
                    else:
                        logger.warning(f"No result returned for {symbol} {interval}")
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol} {interval}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Compare results across symbols and timeframes
        compare_results(results)
        
        logger.info("Comprehensive analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run comprehensive analysis: {e}")
        raise

def compare_results(results):
    """Compare results across different symbols and timeframes."""
    try:
        if not results:
            logger.warning("No results to compare")
            return
        
        print(f"\n{'='*80}")
        print("CROSS-SYMBOL AND TIMEFRAME COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        comparison_data = []
        
        for key, result in results.items():
            symbol, interval = key.split('_')
            
            comparison_data.append({
                'Symbol': symbol,
                'Interval': interval,
                'Processing Time': result.processing_time,
                'Regime Stability': result.performance_metrics.get('regime_stability', 0),
                'Regime Balance': result.performance_metrics.get('regime_balance', 0),
                'Avg Confidence': result.performance_metrics.get('avg_confidence', 0),
                'Avg Duration': result.performance_metrics.get('avg_regime_duration', 0),
                'Memory Usage (MB)': result.memory_usage.get('total_mb', 0)
            })
        
        # Create DataFrame and display
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Create comparison visualization
        if len(comparison_data) > 1:
            create_comparison_visualization(comparison_df)
        
    except Exception as e:
        logger.error(f"Failed to compare results: {e}")
        return

def create_comparison_visualization(comparison_df):
    """Create visualization comparing results across symbols and timeframes."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Processing time comparison
        plt.subplot(2, 3, 1)
        sns.barplot(data=comparison_df, x='Symbol', y='Processing Time', hue='Interval')
        plt.title('Processing Time Comparison')
        plt.xticks(rotation=45)
        
        # Plot 2: Regime stability comparison
        plt.subplot(2, 3, 2)
        sns.barplot(data=comparison_df, x='Symbol', y='Regime Stability', hue='Interval')
        plt.title('Regime Stability Comparison')
        plt.xticks(rotation=45)
        
        # Plot 3: Regime balance comparison
        plt.subplot(2, 3, 3)
        sns.barplot(data=comparison_df, x='Symbol', y='Regime Balance', hue='Interval')
        plt.title('Regime Balance Comparison')
        plt.xticks(rotation=45)
        
        # Plot 4: Average confidence comparison
        plt.subplot(2, 3, 4)
        sns.barplot(data=comparison_df, x='Symbol', y='Avg Confidence', hue='Interval')
        plt.title('Average Confidence Comparison')
        plt.xticks(rotation=45)
        
        # Plot 5: Average duration comparison
        plt.subplot(2, 3, 5)
        sns.barplot(data=comparison_df, x='Symbol', y='Avg Duration', hue='Interval')
        plt.title('Average Regime Duration Comparison')
        plt.xticks(rotation=45)
        
        # Plot 6: Memory usage comparison
        plt.subplot(2, 3, 6)
        sns.barplot(data=comparison_df, x='Symbol', y='Memory Usage (MB)', hue='Interval')
        plt.title('Memory Usage Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('market_analysis/hmm_clustering/results/comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Failed to create comparison visualization: {e}")
        return

def run_single_symbol_analysis():
    """Run analysis for a single symbol with detailed output."""
    try:
        logger.info("Running single symbol analysis")
        
        # Create configuration
        config = HMMClusteringConfig(
            n_components=4,
            lookback_windows=[5, 10, 20, 50],
            technical_indicators=["rsi", "macd", "bollinger_bands", "atr"],
            use_gpu=True,
            use_memory_optimization=True,
            max_features=25
        )
        
        # Run analysis with error handling
        try:
            result = run_hmm_clustering_analysis(
                symbol="BTCUSDT",
                interval="1h",
                config=config,
                save_results=True
            )
            
            if result is None:
                logger.error("Analysis returned None result")
                return None
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Detailed analysis
        print(f"\n{'='*80}")
        print("DETAILED HMM CLUSTERING ANALYSIS")
        print(f"{'='*80}")
        
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        print(f"Memory Usage: {result.memory_usage}")
        print(f"Number of Regimes: {result.config.n_components}")
        print(f"Features Used: {len(result.feature_importance)}")
        
        # Regime analysis
        analyze_regime_characteristics(result)
        
        # Feature importance
        analyze_feature_importance(result)
        
        # Transition analysis
        transition_matrix = analyze_regime_transitions(result)
        
        if transition_matrix is not None:
            print(f"\nTransition Matrix:")
            print(transition_matrix)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to run single symbol analysis: {e}")
        raise

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("market_analysis/hmm_clustering/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Enhanced HMM Clustering for Market Analysis")
    print("=" * 50)
    
    # Choose analysis type with better error handling
    try:
        print("Available analysis types:")
        print("1: Single Symbol Analysis")
        print("2: Comprehensive Analysis (multiple symbols/timeframes)")
        
        analysis_type = input("Choose analysis type (1 or 2, default=1): ").strip()
        
        if analysis_type == "2":
            print("Running comprehensive analysis...")
            run_comprehensive_analysis()
        else:
            print("Running single symbol analysis...")
            run_single_symbol_analysis()
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}")
        import traceback
        traceback.print_exc()