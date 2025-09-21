#!/usr/bin/env python3
"""
Run Clustering Pipeline

Main script to run the complete regime clustering pipeline from HMM discovery
to ML-ready outputs. This script can be used as a standalone tool or imported
as a module.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .clustering_pipeline import create_clustering_pipeline
from .regime_consolidator import ConsolidationConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_clustering_pipeline(
    hmm_results_file: str,
    symbol: str,
    timeframe: str,
    target_clusters: int = 20,
    min_cluster_size_pct: float = 0.03,
    max_cluster_size_pct: float = 0.08,
    coverage_target: float = 0.95,
    output_dir: str = "training/steps/market_analysis/clustering",
    save_outputs: bool = True
) -> dict:
    """
    Run the complete regime clustering pipeline.
    
    Args:
        hmm_results_file: Path to HMM discovery results JSON file
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Data timeframe (e.g., '1h', '15m')
        target_clusters: Number of target clusters
        min_cluster_size_pct: Minimum cluster size as percentage
        max_cluster_size_pct: Maximum cluster size as percentage
        coverage_target: Target coverage by top clusters
        output_dir: Output directory for results
        save_outputs: Whether to save outputs to files
        
    Returns:
        Dictionary with pipeline results
    """
    
    try:
        logger.info(f"Starting regime clustering pipeline for {symbol} {timeframe}")
        logger.info(f"HMM results file: {hmm_results_file}")
        logger.info(f"Target clusters: {target_clusters}")
        logger.info(f"Cluster size range: {min_cluster_size_pct:.1%} - {max_cluster_size_pct:.1%}")
        logger.info(f"Coverage target: {coverage_target:.1%}")
        
        # Create pipeline
        pipeline = create_clustering_pipeline(
            target_clusters=target_clusters,
            min_cluster_size_pct=min_cluster_size_pct,
            max_cluster_size_pct=max_cluster_size_pct,
            coverage_target=coverage_target,
            output_base_dir=output_dir
        )
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            hmm_results_file=hmm_results_file,
            symbol=symbol,
            timeframe=timeframe,
            save_outputs=save_outputs
        )
        
        # Validate results
        validation = pipeline.validate_pipeline_results(results)
        
        if not validation['is_valid']:
            logger.warning("Pipeline validation warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        # Print summary
        summary = pipeline.get_pipeline_summary(results)
        print(summary)
        
        # Log recommendations
        if validation['recommendations']:
            logger.info("Recommendations:")
            for rec in validation['recommendations']:
                logger.info(f"  - {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

def run_from_outcomes_dir(
    outcomes_dir: str,
    symbol: str,
    timeframe: str,
    target_clusters: int = 20,
    min_cluster_size_pct: float = 0.03,
    max_cluster_size_pct: float = 0.08,
    coverage_target: float = 0.95,
    output_dir: str = "training/steps/market_analysis/clustering",
    save_outputs: bool = True
) -> dict:
    """
    Run pipeline using the latest HMM discovery results from outcomes directory.
    
    Args:
        outcomes_dir: Directory containing HMM discovery outcome files
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Data timeframe (e.g., '1h', '15m')
        target_clusters: Number of target clusters
        min_cluster_size_pct: Minimum cluster size as percentage
        max_cluster_size_pct: Maximum cluster size as percentage
        coverage_target: Target coverage by top clusters
        output_dir: Output directory for results
        save_outputs: Whether to save outputs to files
        
    Returns:
        Dictionary with pipeline results
    """
    
    try:
        logger.info(f"Running pipeline from outcomes directory for {symbol} {timeframe}")
        logger.info(f"Outcomes directory: {outcomes_dir}")
        
        # Create pipeline
        pipeline = create_clustering_pipeline(
            target_clusters=target_clusters,
            min_cluster_size_pct=min_cluster_size_pct,
            max_cluster_size_pct=max_cluster_size_pct,
            coverage_target=coverage_target,
            output_base_dir=output_dir
        )
        
        # Run pipeline from outcomes directory
        results = pipeline.run_from_outcomes_dir(
            outcomes_dir=outcomes_dir,
            symbol=symbol,
            timeframe=timeframe,
            save_outputs=save_outputs
        )
        
        # Validate results
        validation = pipeline.validate_pipeline_results(results)
        
        if not validation['is_valid']:
            logger.warning("Pipeline validation warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        # Print summary
        summary = pipeline.get_pipeline_summary(results)
        print(summary)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description="Run regime clustering pipeline from HMM discovery to ML-ready outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with specific HMM results file
  python run_clustering_pipeline.py --hmm-results /path/to/hmm_results.json --symbol BTCUSDT --timeframe 1h
  
  # Run from outcomes directory (finds latest results)
  python run_clustering_pipeline.py --outcomes-dir /path/to/outcomes --symbol ETHUSDT --timeframe 15m
  
  # Custom clustering parameters
  python run_clustering_pipeline.py --hmm-results /path/to/hmm_results.json --symbol BTCUSDT --timeframe 1h \\
    --target-clusters 25 --min-cluster-size 0.02 --max-cluster-size 0.10 --coverage-target 0.98
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--hmm-results',
        type=str,
        help='Path to HMM discovery results JSON file'
    )
    input_group.add_argument(
        '--outcomes-dir',
        type=str,
        help='Directory containing HMM discovery outcome files'
    )
    
    # Required parameters
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., BTCUSDT, ETHUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        required=True,
        help='Data timeframe (e.g., 1h, 15m, 4h)'
    )
    
    # Clustering parameters
    parser.add_argument(
        '--target-clusters',
        type=int,
        default=20,
        help='Number of target clusters (default: 20)'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=float,
        default=0.03,
        help='Minimum cluster size as percentage (default: 0.03 = 3%%)'
    )
    parser.add_argument(
        '--max-cluster-size',
        type=float,
        default=0.08,
        help='Maximum cluster size as percentage (default: 0.08 = 8%%)'
    )
    parser.add_argument(
        '--coverage-target',
        type=float,
        default=0.95,
        help='Target coverage by top clusters (default: 0.95 = 95%%)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training/steps/market_analysis/clustering',
        help='Output directory for results (default: training/steps/market_analysis/clustering)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save outputs to files'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        if args.hmm_results:
            # Run with specific HMM results file
            results = run_clustering_pipeline(
                hmm_results_file=args.hmm_results,
                symbol=args.symbol,
                timeframe=args.timeframe,
                target_clusters=args.target_clusters,
                min_cluster_size_pct=args.min_cluster_size,
                max_cluster_size_pct=args.max_cluster_size,
                coverage_target=args.coverage_target,
                output_dir=args.output_dir,
                save_outputs=not args.no_save
            )
        else:
            # Run from outcomes directory
            results = run_from_outcomes_dir(
                outcomes_dir=args.outcomes_dir,
                symbol=args.symbol,
                timeframe=args.timeframe,
                target_clusters=args.target_clusters,
                min_cluster_size_pct=args.min_cluster_size,
                max_cluster_size_pct=args.max_cluster_size,
                coverage_target=args.coverage_target,
                output_dir=args.output_dir,
                save_outputs=not args.no_save
            )
        
        logger.info("Pipeline completed successfully!")
        
        # Print key metrics
        consolidation = results['consolidation_result']
        dataset = results['training_dataset']
        
        print(f"\nKey Metrics:")
        print(f"  Final Clusters: {consolidation['final_cluster_count']}")
        print(f"  Coverage: {consolidation['coverage_percentage']:.2%}")
        print(f"  Top Clusters Coverage: {consolidation['top_clusters_coverage']:.2%}")
        print(f"  Training Samples: {dataset['n_samples']:,}")
        
        if 'saved_files' in results:
            print(f"\nOutput Files:")
            for output_type, file_path in results['saved_files'].items():
                print(f"  {output_type}: {file_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())