#!/usr/bin/env python3
"""
CLI Interface for Statsmodel Clustering

This module provides a comprehensive command-line interface for running statsmodel clustering
analysis with data downloading, model training, and result visualization.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import statsmodel clustering components with error handling
try:
    from src.training.steps.market_analysis.statsmodel_clustering.core import (
        MarkovRegressionAdapter,
        create_enhanced_markov_regression_adapter,
        BaseDataDownloader,
        StandardDataDownloader,
        create_data_downloader,
        download_clustering_data
    )
    CORE_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core imports not available: {e}")
    MarkovRegressionAdapter = None
    create_enhanced_markov_regression_adapter = None
    BaseDataDownloader = None
    StandardDataDownloader = None
    create_data_downloader = None
    download_clustering_data = None
    CORE_IMPORTS_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
    from src.utils.logger import system_logger
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    system_logger = None


class StatsmodelClusteringCLI:
    """Command-line interface for statsmodel clustering operations."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.logger = system_logger.getChild("StatsmodelClusteringCLI") if system_logger else None
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Statsmodel Clustering Analysis CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Download data for ETHUSDT
  python cli.py download --symbol ETHUSDT --exchange BINANCE --timeframe 1h

  # Run clustering analysis
  python cli.py cluster --symbol ETHUSDT --data-file data.parquet --regimes 3

  # Run complete pipeline
  python cli.py pipeline --symbol ETHUSDT --exchange BINANCE --timeframe 1h --regimes 3

  # Optimize parameters
  python cli.py optimize --symbol ETHUSDT --data-file data.parquet --trials 50
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Download command
        download_parser = subparsers.add_parser('download', help='Download market data')
        download_parser.add_argument('--symbol', type=str, default='ETHUSDT',
                                help='Trading symbol (default: ETHUSDT)')
        download_parser.add_argument('--exchange', type=str, default='BINANCE',
                                help='Exchange name (default: BINANCE)')
        download_parser.add_argument('--timeframe', type=str, default='1h',
                                help='Timeframe (default: 1h)')
        download_parser.add_argument('--years', type=int, default=2,
                                help='Years of historical data (default: 2)')
        download_parser.add_argument('--data-dir', type=str, default='data_cache',
                                help='Data directory (default: data_cache)')
        download_parser.add_argument('--force', action='store_true',
                                help='Force re-download even if data exists')
        download_parser.add_argument('--output', type=str,
                                help='Output file path (optional)')
        
        # Cluster command
        cluster_parser = subparsers.add_parser('cluster', help='Run clustering analysis')
        cluster_parser.add_argument('--symbol', type=str, default='ETHUSDT',
                                help='Trading symbol (default: ETHUSDT)')
        cluster_parser.add_argument('--data-file', type=str, required=False,
                                help='Input data file path (optional, will download if not provided)')
        cluster_parser.add_argument('--regimes', type=int, default=5,
                                help='Number of regimes (default: 5)')
        cluster_parser.add_argument('--pca-components', type=int, default=12,
                                help='PCA components (default: 12)')
        cluster_parser.add_argument('--output-dir', type=str, default='outcomes',
                                help='Output directory (default: outcomes)')
        cluster_parser.add_argument('--config', type=str,
                                help='Configuration file path (optional)')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
        pipeline_parser.add_argument('--symbol', type=str, default='ETHUSDT',
                                 help='Trading symbol (default: ETHUSDT)')
        pipeline_parser.add_argument('--exchange', type=str, default='BINANCE',
                                 help='Exchange name (default: BINANCE)')
        pipeline_parser.add_argument('--timeframe', type=str, default='1h',
                                 help='Timeframe (default: 1h)')
        pipeline_parser.add_argument('--years', type=int, default=2,
                                 help='Years of historical data (default: 2)')
        pipeline_parser.add_argument('--regimes', type=int, default=5,
                                 help='Number of regimes (default: 5)')
        pipeline_parser.add_argument('--data-dir', type=str, default='data_cache',
                                 help='Data directory (default: data_cache)')
        pipeline_parser.add_argument('--output-dir', type=str, default='outcomes',
                                 help='Output directory (default: outcomes)')
        pipeline_parser.add_argument('--force-download', action='store_true',
                                 help='Force re-download even if data exists')
        pipeline_parser.add_argument('--config', type=str,
                                 help='Configuration file path (optional)')
        
        # Optimize command
        optimize_parser = subparsers.add_parser('optimize', help='Optimize clustering parameters')
        optimize_parser.add_argument('--symbol', type=str, default='ETHUSDT',
                                  help='Trading symbol (default: ETHUSDT)')
        optimize_parser.add_argument('--data-file', type=str, required=True,
                                  help='Input data file path')
        optimize_parser.add_argument('--trials', type=int, default=50,
                                  help='Number of optimization trials (default: 50)')
        optimize_parser.add_argument('--output-dir', type=str, default='outcomes',
                                  help='Output directory (default: outcomes)')
        optimize_parser.add_argument('--config', type=str,
                                  help='Configuration file path (optional)')
        
        # Global arguments
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose output')
        parser.add_argument('--log-level', type=str, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Log level (default: INFO)')
        
        return parser
    
    async def handle_download(self, args: argparse.Namespace) -> bool:
        """Handle the download command."""
        try:
            if not CORE_IMPORTS_AVAILABLE:
                tprint_error("❌ Core imports not available - cannot download data")
                return False
                
            tprint_info(f"📥 Downloading data for {args.symbol} on {args.exchange} ({args.timeframe})")
            
            config = {
                'symbol': args.symbol,
                'exchange': args.exchange,
                'timeframe': args.timeframe,
                'lookback_years': args.years,
                'data_dir': args.data_dir,
                'force_download': args.force,
                'downloader_type': 'standard'
            }
            
            downloader = create_data_downloader(config)
            success, data, error = await downloader.download_data()
            
            if success and data is not None:
                tprint_success(f"✅ Successfully downloaded {len(data)} records")
                
                # Save to custom output path if specified
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    data.to_parquet(output_path, index=True, compression='snappy')
                    tprint_info(f"💾 Data saved to {output_path}")
                
                # Print statistics
                self._print_data_stats(data)
                return True
            else:
                tprint_error(f"❌ Download failed: {error}")
                return False
                
        except Exception as e:
            tprint_error(f"❌ Download error: {e}")
            return False
    
    async def handle_cluster(self, args: argparse.Namespace) -> bool:
        """Handle the cluster command."""
        try:
            if not CORE_IMPORTS_AVAILABLE:
                tprint_error("❌ Core imports not available - cannot run clustering")
                return False
                
            tprint_info(f"🔬 Running clustering analysis for {args.symbol}")
            
            # Load or download data
            data = None
            data_path = None
            
            if args.data_file and Path(args.data_file).exists():
                # Load existing data file
                data_path = Path(args.data_file)
                data = pd.read_parquet(data_path)
                tprint_info(f"📊 Loaded {len(data)} records from {data_path}")
            else:
                # Download data using BaseDataDownloader
                tprint_info("📥 No data file provided, downloading data...")
                
                # Create downloader config
                download_config = {
                    'symbol': args.symbol,
                    'exchange': 'BINANCE',  # Default exchange
                    'timeframe': '1h',  # Default timeframe
                    'lookback_years': 0.08,  # ~30 days
                    'data_dir': 'data_cache',
                    'downloader_type': 'standard'
                }
                
                downloader = create_data_downloader(download_config)
                success, downloaded_data, error = await downloader.download_data()
                
                if success and downloaded_data is not None:
                    data = downloaded_data
                    tprint_info(f"📊 Downloaded {len(data)} records")
                else:
                    tprint_error(f"❌ Data download failed: {error}")
                    return False
            
            # Load configuration if provided
            config = self._load_config(args.config) if args.config else {}
            
            # Create adapter
            adapter = create_enhanced_markov_regression_adapter(
                k_regimes=args.regimes,
                enable_pca=True,
                pca_components=args.pca_components,
                enable_diagnostics=True,
                enable_hardware_optimization=True,
                **config
            )
            
            # Prepare data for clustering
            features = self._prepare_features(data)
            
            # Fit model
            tprint_info("🔄 Fitting clustering model...")
            result = adapter.fit(features)
            
            if result.success:
                tprint_success(f"✅ Clustering completed successfully")
                self._print_clustering_results(result)
                
                # Save results
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self._save_results(result, output_dir, args.symbol)
                
                return True
            else:
                tprint_error(f"❌ Clustering failed: {result.error_message}")
                return False
                
        except Exception as e:
            tprint_error(f"❌ Clustering error: {e}")
            return False
    
    async def handle_pipeline(self, args: argparse.Namespace) -> bool:
        """Handle the pipeline command."""
        try:
            tprint_info(f"🚀 Running complete pipeline for {args.symbol}")
            
            # Step 1: Download data
            tprint_info("📥 Step 1: Downloading data...")
            download_success = await self.handle_download(args)
            if not download_success:
                return False
            
            # Get data file path
            config = {
                'symbol': args.symbol,
                'exchange': args.exchange,
                'timeframe': args.timeframe,
                'data_dir': args.data_dir
            }
            downloader = create_data_downloader(config)
            data_file = downloader.get_output_path()
            
            # Step 2: Run clustering
            tprint_info("🔬 Step 2: Running clustering analysis...")
            cluster_args = argparse.Namespace(
                symbol=args.symbol,
                data_file=str(data_file),
                regimes=args.regimes,
                pca_components=12,  # Default value
                output_dir=args.output_dir,
                config=args.config
            )
            cluster_success = await self.handle_cluster(cluster_args)
            if not cluster_success:
                return False
            
            tprint_success("✅ Pipeline completed successfully")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Pipeline error: {e}")
            return False
    
    async def handle_optimize(self, args: argparse.Namespace) -> bool:
        """Handle the optimize command."""
        try:
            tprint_info(f"⚡ Optimizing clustering parameters for {args.symbol}")
            
            # Load data
            data_path = Path(args.data_file)
            if not data_path.exists():
                tprint_error(f"❌ Data file not found: {data_path}")
                return False
            
            data = pd.read_parquet(data_path)
            tprint_info(f"📊 Loaded {len(data)} records from {data_path}")
            
            # Load configuration if provided
            config = self._load_config(args.config) if args.config else {}
            
            # Prepare features
            features = self._prepare_features(data)
            
            # Run optimization (placeholder for now)
            tprint_info("🔄 Running parameter optimization...")
            tprint_warning("⚠️ Optimization not yet implemented - using default parameters")
            
            # Create adapter with default parameters
            adapter = create_enhanced_markov_regression_adapter(
                k_regimes=3,
                enable_pca=True,
                pca_components=12,
                enable_diagnostics=True,
                enable_hardware_optimization=True,
                **config
            )
            
            # Fit model
            result = adapter.fit(features)
            
            if result.success:
                tprint_success(f"✅ Optimization completed")
                self._print_clustering_results(result)
                
                # Save results
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self._save_results(result, output_dir, args.symbol)
                
                return True
            else:
                tprint_error(f"❌ Optimization failed: {result.error_message}")
                return False
                
        except Exception as e:
            tprint_error(f"❌ Optimization error: {e}")
            return False
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load config: {e}")
            return {}
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering."""
        # Basic feature engineering
        features = pd.DataFrame(index=data.index)
        
        # Print available columns for debugging
        tprint_info(f"📊 Available columns: {list(data.columns)}")
        
        # Map common column names to standard names
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'close' in col_lower and 'close' not in column_mapping:
                column_mapping['close'] = col
            elif 'high' in col_lower and 'high' not in column_mapping:
                column_mapping['high'] = col
            elif 'low' in col_lower and 'low' not in column_mapping:
                column_mapping['low'] = col
            elif 'open' in col_lower and 'open' not in column_mapping:
                column_mapping['open'] = col
            elif 'volume' in col_lower and 'volume' not in column_mapping:
                column_mapping['volume'] = col
        
        tprint_info(f"📊 Column mapping: {column_mapping}")
        
        # Check if we have the required columns
        if 'close' not in column_mapping:
            tprint_error("❌ No 'close' column found in data")
            raise KeyError("No 'close' column found in data")
        
        # Use mapped columns
        close_col = column_mapping['close']
        high_col = column_mapping.get('high')
        low_col = column_mapping.get('low')
        open_col = column_mapping.get('open')
        volume_col = column_mapping.get('volume')
        
        # Price-based features
        features['returns'] = data[close_col].pct_change()
        features['log_returns'] = np.log(data[close_col] / data[close_col].shift(1))
        
        if high_col and low_col:
            features['high_low_ratio'] = data[high_col] / data[low_col]
        
        if open_col:
            features['close_open_ratio'] = data[close_col] / data[open_col]
        
        # Volume features
        if volume_col:
            features['volume_ratio'] = data[volume_col] / data[volume_col].rolling(20).mean()
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Trend features
        features['sma_5'] = data[close_col].rolling(5).mean()
        features['sma_20'] = data[close_col].rolling(20).mean()
        features['sma_ratio'] = features['sma_5'] / features['sma_20']
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _print_data_stats(self, data: pd.DataFrame):
        """Print data statistics."""
        tprint_info("📊 Data Statistics:")
        print(f"  Records: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Date range: {data.index.min()} to {data.index.max()}")
        print(f"  Missing values: {data.isnull().sum().sum()}")
    
    def _print_clustering_results(self, result):
        """Print clustering results."""
        tprint_info("📊 Clustering Results:")
        print(f"  Number of regimes: {result.n_regimes}")
        print(f"  Log likelihood: {result.log_likelihood:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        if result.transition_matrix is not None:
            print(f"  Transition matrix shape: {result.transition_matrix.shape}")
        
        if result.diagnostics:
            print(f"  Diagnostics available: Yes")
    
    def _save_results(self, result, output_dir: Path, symbol: str):
        """Save clustering results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save basic results
        results_file = output_dir / f"{symbol}_clustering_results_{timestamp}.json"
        results_data = {
            'symbol': symbol,
            'timestamp': timestamp,
            'n_regimes': result.n_regimes,
            'log_likelihood': result.log_likelihood,
            'aic': result.aic,
            'bic': result.bic,
            'processing_time': result.processing_time,
            'feature_names': result.feature_names,
            'success': result.success
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        tprint_info(f"💾 Results saved to {results_file}")
        
        # Save regime labels if available
        if len(result.cluster_labels) > 0:
            labels_file = output_dir / f"{symbol}_regime_labels_{timestamp}.csv"
            labels_df = pd.DataFrame({
                'regime': result.cluster_labels
            }, index=result.metadata.get('data_index', pd.RangeIndex(len(result.cluster_labels))))
            labels_df.to_csv(labels_file)
            tprint_info(f"💾 Regime labels saved to {labels_file}")
    
    async def run(self, args: Optional[list] = None) -> int:
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        # Set up logging
        if self.logger:
            import logging
            log_level = getattr(logging, parsed_args.log_level)
            logging.basicConfig(level=log_level)
        
        # Handle command
        try:
            if parsed_args.command == 'download':
                success = await self.handle_download(parsed_args)
            elif parsed_args.command == 'cluster':
                success = await self.handle_cluster(parsed_args)
            elif parsed_args.command == 'pipeline':
                success = await self.handle_pipeline(parsed_args)
            elif parsed_args.command == 'optimize':
                success = await self.handle_optimize(parsed_args)
            else:
                tprint_error(f"❌ Unknown command: {parsed_args.command}")
                return 1
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            tprint_warning("⚠️ Operation cancelled by user")
            return 1
        except Exception as e:
            tprint_error(f"❌ Unexpected error: {e}")
            return 1


async def main():
    """Main entry point."""
    cli = StatsmodelClusteringCLI()
    return await cli.run()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)