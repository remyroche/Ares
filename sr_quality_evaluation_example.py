#!/usr/bin/env python3
"""
SR Quality Evaluation Example Script

This script demonstrates how to use the comprehensive SR quality evaluation system
to pre-evaluate SR levels before ML training.

Usage:
    python sr_quality_evaluation_example.py --symbol BTCUSDT --exchange binance

Features:
- Comprehensive SR detection using 10 different methods
- Multi-factor strength evaluation with historical validation
- Quality classification into Elite/Strong/Moderate/Weak/Rejected categories
- Multiple training datasets (elite-only, strong+, moderate+, all-qualified, weighted)
- Bayesian parameter optimization for optimal settings
- Complete validation and reporting
"""

import asyncio
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SROptimizationStep
from src.utils.logger import system_logger
import logging
import time


async def run_sr_quality_evaluation(symbol: str, exchange: str, config_path: str = None):
    """Run comprehensive SR quality evaluation."""

    logger = system_logger.getChild('SRQualityEvaluation')
    logger.info(f'🎯 Starting SR Quality Evaluation for {symbol} on {exchange}')

    try:
        # Initialize the SR optimization step
        step = SROptimizationStep({})

        # Load configuration
        config = load_evaluation_config(config_path)

        # Load market data (you would replace this with actual data loading)
        market_data = await load_market_data(symbol, exchange)

        if market_data is None or market_data.empty:
            logger.error('No market data available')
            return None

        # Prepare inputs for quality evaluation
        training_input = {
            'config': config,
            'validated_data': market_data,
            'sr_levels': {'support_levels': [], 'resistance_levels': []}  # Will be auto-detected
        }

        pipeline_state = {
            'dataframe': market_data
        }

        # Run comprehensive quality evaluation
        logger.info('🚀 Running comprehensive SR quality evaluation...')
        quality_results = step.evaluate_sr_quality_for_ml_training(
            market_data=market_data,
            sr_levels=training_input['sr_levels'],
            quality_thresholds=config.get('sr_optimization', {}).get('quality_thresholds', {})
        )

        if quality_results:
            # Display results
            display_quality_results(quality_results)

            # Save detailed results
            save_results(quality_results, symbol, exchange)

            logger.info('✅ SR Quality Evaluation completed successfully!')
            return quality_results
        else:
            logger.error('❌ SR Quality Evaluation failed')
            return None

    except Exception as e:
        logger.error(f'❌ SR Quality Evaluation failed: {e}')
        return None


def load_evaluation_config(config_path: str = None) -> dict:
    """Load configuration for SR quality evaluation."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)

    # Default configuration
    return {
        'sr_optimization': {
            'enable_quality_evaluation': True,
            'quality_thresholds': {
                'elite_threshold': 0.85,
                'strong_threshold': 0.70,
                'moderate_threshold': 0.50,
                'weak_threshold': 0.30,
                'historical_min_tests': 5,
                'confluence_min_levels': 2,
                'volume_min_ratio': 1.2
            }
        }
    }


async def load_market_data(symbol: str, exchange: str) -> pd.DataFrame:
    """Load market data for evaluation."""
    try:
        # This is a placeholder - replace with actual data loading logic
        # For demonstration, we'll create sample data
        logger = system_logger.getChild('DataLoader')
        logger.info(f'📊 Loading market data for {symbol} on {exchange}')

        # Create sample OHLCV data (replace with actual data loading)
        n_samples = 5000  # 5k candles
        base_price = 50000.0  # Sample BTC price

        # Generate realistic price action
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
        prices = []
        current_price = base_price

        for i in range(n_samples):
            # Add some trend and volatility
            trend = 0.0001 * np.sin(i / 100)  # Slow trend
            noise = np.random.normal(0, 0.002)  # 0.2% volatility
            current_price *= (1 + trend + noise)
            prices.append(current_price)

        # Create OHLCV DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n_samples)
        })

        df.set_index('timestamp', inplace=True)
        logger.info(f'✅ Loaded {len(df)} candles of market data')
        return df

    except Exception as e:
        logger = system_logger.getChild('DataLoader')
        logger.error(f'Failed to load market data: {e}')
        return pd.DataFrame()


def display_quality_results(results: dict):
    """Display quality evaluation results in a readable format."""
    logger = system_logger.getChild('ResultsDisplay')

    print("\n" + "="*80)
    print("🎯 SR QUALITY EVALUATION RESULTS")
    print("="*80)

    # Quality classification summary
    classification = results.get('quality_classification', {})
    summary = classification.get('summary', {})

    print(f"📊 Total Levels Evaluated: {results.get('total_levels_evaluated', 0)}")
    print(f"🏆 Elite Levels (0.85+): {summary.get('elite_count', 0)} ({summary.get('elite_percentage', 0):.1f}%)")
    print(f"💪 Strong Levels (0.70+): {summary.get('strong_count', 0)} ({summary.get('strong_percentage', 0):.1f}%)")
    print(f"📈 Moderate Levels (0.50+): {summary.get('moderate_count', 0)}")
    print(f"⚠️ Weak Levels (0.30+): {summary.get('weak_count', 0)}")
    print(f"❌ Rejected Levels (<0.30): {summary.get('rejected_count', 0)}")

    # Top performing detection methods
    evaluation_summary = results.get('evaluation_summary', {})
    top_methods = evaluation_summary.get('top_performing_methods', [])

    if top_methods:
        print(f"\n🔧 Top Detection Methods:")
        for i, method in enumerate(top_methods[:3], 1):
            print(f"  {i}. {method['method']} (Quality: {method['average_quality']:.3f})")

    # Training datasets available
    training_data = results.get('training_data', {})
    print(f"\n🎓 Available Training Datasets:")
    for dataset_name in training_data.keys():
        dataset = training_data[dataset_name]
        if isinstance(dataset, pd.DataFrame) and not dataset.empty:
            print(f"  ✅ {dataset_name}: {len(dataset)} samples")
        else:
            print(f"  ❌ {dataset_name}: No data")

    # Recommendations
    recommendations = evaluation_summary.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    print("\n" + "="*80)


def save_results(results: dict, symbol: str, exchange: str):
    """Save evaluation results to file."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sr_quality_evaluation_{symbol}_{exchange}_{timestamp}.json'

        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        filepath = reports_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger = system_logger.getChild('ResultsSaver')
        logger.info(f'💾 Results saved to: {filepath}')

        # Also save a summary report
        summary_filename = f'sr_quality_summary_{symbol}_{exchange}_{timestamp}.txt'
        summary_filepath = reports_dir / summary_filename

        with open(summary_filepath, 'w') as f:
            f.write("SR Quality Evaluation Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Exchange: {exchange}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

            classification = results.get('quality_classification', {})
            summary = classification.get('summary', {})

            f.write("Quality Classification:\n")
            f.write(f"  Elite: {summary.get('elite_count', 0)}\n")
            f.write(f"  Strong: {summary.get('strong_count', 0)}\n")
            f.write(f"  Moderate: {summary.get('moderate_count', 0)}\n")
            f.write(f"  Weak: {summary.get('weak_count', 0)}\n")
            f.write(f"  Rejected: {summary.get('rejected_count', 0)}\n\n")

            training_data = results.get('training_data', {})
            f.write("Training Datasets:\n")
            for name in training_data.keys():
                dataset = training_data[name]
                if isinstance(dataset, pd.DataFrame):
                    f.write(f"  {name}: {len(dataset)} samples\n")

        logger.info(f'📄 Summary saved to: {summary_filepath}')

    except Exception as e:
        logger = system_logger.getChild('ResultsSaver')
        logger.error(f'Failed to save results: {e}')


def main():
    """Main entry point for SR quality evaluation."""
    parser = argparse.ArgumentParser(description='SR Quality Evaluation for ML Training')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--exchange', required=True, help='Exchange name (e.g., binance)')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output-dir', default='reports', help='Output directory for results')

    args = parser.parse_args()

    # Run the evaluation
    results = asyncio.run(run_sr_quality_evaluation(
        symbol=args.symbol,
        exchange=args.exchange,
        config_path=args.config
    ))

    if results:
        print("✅ SR Quality Evaluation completed successfully!")
        print(f"📊 Results saved to reports directory")
    else:
        print("❌ SR Quality Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
