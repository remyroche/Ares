#!/usr/bin/env python3
"""
Automated Cryptocurrency Data Processor
Coordinates the download and analysis of cryptocurrency data
"""

import asyncio
import traceback
import sys
import gc
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Add the main project to path for Ares utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data_downloader import BinanceDataDownloader
from data_analyzer import CryptoPriceAnalyzer
from config import ASSETS, DATA_CONFIG, ANALYSIS_CONFIG

# Import Ares utilities for optimization
try:
    from src.utils.math_validation import safe_divide, safe_correlation
    from src.utils.parquet_utils import ParquetUtils
    from src.utils.common_operations import ensure_directory
    from src.core.decorators import handles_errors
    from src.utils.logger import system_logger
    ARES_UTILS_AVAILABLE = True
    logger = system_logger.getChild(__name__)
except ImportError as e:
    ARES_UTILS_AVAILABLE = False
    # Fallback logging
    import logging
    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)
    logger = get_logger(__name__)
    logger.warning(f"⚠️ Ares utilities not fully available: {e}")

# Try to import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer  
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False
    M1GPUManager = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None

class AutomatedCryptoProcessor:
    """Enhanced automated processor for cryptocurrency data analysis with Ares utilities"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        """
        Initialize the enhanced processor with Ares utilities
        
        Args:
            data_dir: Directory to store raw data
            output_dir: Directory to store analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.assets = ASSETS.copy()
        
        # Initialize Ares utilities if available
        self.parquet_utils = ParquetUtils() if ARES_UTILS_AVAILABLE else None
        self.hardware_optimizations = {}
        
        # Initialize hardware optimizations
        if HARDWARE_OPT_AVAILABLE:
            try:
                self.gpu_manager = M1GPUManager() if M1GPUManager else None
                self.memory_optimizer = M1MemoryOptimizer() if M1MemoryOptimizer else None
                self.cpu_optimizer = M1CPUOptimizer() if M1CPUOptimizer else None
                
                self.hardware_optimizations = {
                    "gpu_available": self.gpu_manager is not None,
                    "memory_opt_available": self.memory_optimizer is not None,
                    "cpu_opt_available": self.cpu_optimizer is not None
                }
            except Exception as e:
                logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Create directories using Ares utilities if available
        if ARES_UTILS_AVAILABLE:
            ensure_directory(str(self.data_dir))
            ensure_directory(str(self.output_dir))
            ensure_directory(str(self.output_dir / "reports"))
            ensure_directory(str(self.output_dir / "csv"))
            ensure_directory(str(self.output_dir / "charts"))
        else:
            # Fallback directory creation
            self.data_dir.mkdir(exist_ok=True)
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "reports").mkdir(exist_ok=True)
            (self.output_dir / "csv").mkdir(exist_ok=True)
            (self.output_dir / "charts").mkdir(exist_ok=True)
        
        # Initialize components
        self.downloader = BinanceDataDownloader()
        self.analyzer = None
        
        # Log initialization status
        logger.info(f"✅ Enhanced processor initialized with data_dir={data_dir}, output_dir={output_dir}")
        logger.info(f"🔧 Ares utilities available: {ARES_UTILS_AVAILABLE}")
        logger.info(f"⚡ Hardware optimizations available: {HARDWARE_OPT_AVAILABLE}")
        
        if HARDWARE_OPT_AVAILABLE:
            for opt_name, available in self.hardware_optimizations.items():
                status = "✅" if available else "❌"
                logger.info(f"   {status} {opt_name}")
    
    def _optimize_dataframe_memory(self, df):
        """Optimize DataFrame memory usage using available optimizations"""
        if not ARES_UTILS_AVAILABLE:
            return df
            
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            # Apply memory optimization if available
            if self.memory_optimizer:
                # Use M1 memory optimization
                df_optimized = df.copy()
                
                # Optimize numeric columns
                for col in df_optimized.select_dtypes(include=[np.number]).columns:
                    if df_optimized[col].dtype == 'float64':
                        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                    elif df_optimized[col].dtype == 'int64':
                        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                
                optimized_memory = df_optimized.memory_usage(deep=True).sum()
                reduction = safe_divide(original_memory - optimized_memory, original_memory, 0.0) * 100
                
                logger.info(f"🧠 Memory optimized: {reduction:.1f}% reduction")
                return df_optimized
            else:
                return df
                
        except Exception as e:
            logger.warning(f"⚠️ Memory optimization failed: {e}")
            return df
    
    async def download_data(self, years: int = 2, api_key: str = "", api_secret: str = "", use_existing: bool = True) -> Optional[Path]:
        """
        Enhanced data download with Ares utilities optimization
        
        Args:
            years: Number of years of historical data to download
            api_key: Binance API key (optional)
            api_secret: Binance API secret (optional)
            use_existing: Whether to check for and use existing data files
            
        Returns:
            Path to the downloaded data file, or None if failed
        """
        try:
            # Check for existing data first if requested
            if use_existing:
                existing_files = list(self.data_dir.glob("crypto_*.parquet"))
                if existing_files:
                    latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
                    
                    # Validate existing file using Ares utilities
                    if self.parquet_utils:
                        validation_result = self.parquet_utils.validate_parquet_file(str(latest_file))
                        if validation_result["valid"]:
                            logger.info(f"✅ Using existing validated data: {latest_file}")
                            logger.info(f"📊 File size: {validation_result.get('file_size', 0) / 1024 / 1024:.1f}MB")
                            logger.info(f"📈 Records: {validation_result.get('shape', [0])[0]:,}")
                            return latest_file
                        else:
                            logger.warning(f"⚠️ Existing file validation failed: {validation_result.get('error', 'Unknown')}")
                    else:
                        # Basic validation fallback
                        try:
                            test_df = pd.read_parquet(latest_file, nrows=10)
                            if not test_df.empty:
                                logger.info(f"✅ Using existing data: {latest_file}")
                                return latest_file
                        except Exception as e:
                            logger.warning(f"⚠️ Existing file validation failed: {e}")
            
            logger.info(f"🔄 Downloading fresh data for {len(self.assets)} assets, {years} years")
            
            # Apply memory optimization for large downloads
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.optimize_for_operation("data_download")
                    logger.info("🧠 Memory optimization applied for download")
                except Exception as e:
                    logger.warning(f"⚠️ Memory optimization failed: {e}")
            
            # Define date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Download data
            df = self.downloader.download_multiple_assets(
                self.assets, 
                start_date, 
                end_date, 
                interval=DATA_CONFIG["interval"]
            )
            
            if df.empty:
                logger.error("❌ No data downloaded")
                return None
            
            # Apply memory optimization to the downloaded data
            df_optimized = self._optimize_dataframe_memory(df)
            
            # Save to Parquet file with enhanced utilities
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.data_dir / f"crypto_enhanced_{DATA_CONFIG['interval']}_{timestamp}.parquet"
            
            if self.parquet_utils:
                # Use Ares parquet utilities for optimal saving
                save_result = self.parquet_utils.save_dataframe_to_parquet(
                    df_optimized,
                    str(output_file),
                    compression="snappy",
                    validate_after_save=True
                )
                
                if save_result["success"]:
                    logger.info(f"✅ Enhanced data saved: {output_file}")
                    logger.info(f"📊 Compression ratio: {save_result.get('compression_ratio', 'N/A')}")
                else:
                    logger.error(f"❌ Enhanced save failed: {save_result['error']}")
                    # Fallback to standard save
                    df_optimized.to_parquet(output_file, compression="snappy", engine="pyarrow", index=True)
                    logger.info(f"💾 Fallback save completed: {output_file}")
            else:
                # Standard parquet save
                df_optimized.to_parquet(output_file, compression="snappy", engine="pyarrow", index=True)
                logger.info(f"💾 Data saved: {output_file}")
            
            logger.info(f"📈 Downloaded {len(df_optimized):,} records for {df_optimized['symbol'].nunique()} assets")
            
            return output_file
            
        except Exception as e:
            logger.exception(f"❌ Error during enhanced data download: {e}")
            return None
    
    def analyze_data(self, data_file: Path) -> Dict[str, Any]:
        """
        Analyze the downloaded cryptocurrency data
        
        Args:
            data_file: Path to the data file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting data analysis on {data_file}")
            
            # Initialize analyzer
            self.analyzer = CryptoPriceAnalyzer(data_file)
            
            # Load data
            if not self.analyzer.load_data():
                logger.error("Failed to load data for analysis")
                return {"success": False, "error": "Failed to load data"}
            
            # Run analysis
            self.analyzer.analyze_all_assets()
            
            # Generate summary report
            summary = self.analyzer.generate_summary_report(save_to_file=True, output_dir=str(self.output_dir))
            
            # Create visualizations if configured
            if ANALYSIS_CONFIG["generate_plots"]:
                charts_dir = self.output_dir / "charts"
                charts_dir.mkdir(exist_ok=True)
                self.analyzer.create_visualizations(str(charts_dir))
                logger.info(f"Charts saved to {charts_dir}")
            
            # Save CSV files if configured
            if ANALYSIS_CONFIG["save_csv"]:
                csv_dir = self.output_dir / "csv"
                csv_dir.mkdir(exist_ok=True)
                
                # Save summary data
                if summary and "basic_summary" in summary:
                    summary["basic_summary"].to_csv(csv_dir / "price_movement_metrics.csv", index=False)
                    logger.info("Basic metrics saved to price_movement_metrics.csv")
                
                if summary and "barrier_summary" in summary:
                    summary["barrier_summary"].to_csv(csv_dir / "triple_barrier_profits.csv", index=False)
                    logger.info("Triple barrier results saved to triple_barrier_profits.csv")
                
                if summary and "volume_summary" in summary and not summary["volume_summary"].empty:
                    summary["volume_summary"].to_csv(csv_dir / "volume_analysis.csv", index=False)
                    logger.info("Volume analysis saved to volume_analysis.csv")
            
            # Save comprehensive JSON results
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            comprehensive_results = {
                "timestamp": timestamp,
                "analysis_summary": {
                    "total_assets": len(self.analyzer.results),
                    "total_records": len(self.analyzer.df) if self.analyzer.df is not None else 0,
                    "date_range": {
                        "start": str(self.analyzer.df.index.min()) if self.analyzer.df is not None else None,
                        "end": str(self.analyzer.df.index.max()) if self.analyzer.df is not None else None
                    }
                },
                "detailed_results": self.analyzer.results
            }
            
            json_file = self.output_dir / f"comprehensive_results_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            logger.info(f"Comprehensive JSON results saved to {json_file}")
            
            # Generate detailed reports if configured
            if ANALYSIS_CONFIG["save_detailed_reports"]:
                self.generate_detailed_reports()
            
            logger.info("Analysis completed successfully")
            return {
                "success": True,
                "summary": summary,
                "results": self.analyzer.results
            }
            
        except Exception as e:
            logger.exception(f"Error during data analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_detailed_reports(self):
        """Generate detailed text reports for each asset"""
        try:
            reports_dir = self.output_dir / "reports"
            
            # Generate individual asset reports
            for symbol, result in self.analyzer.results.items():
                report_file = reports_dir / f"{symbol}_detailed_report.txt"
                
                with open(report_file, 'w') as f:
                    f.write(f"DETAILED ANALYSIS REPORT FOR {symbol}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Basic metrics
                    f.write("BASIC METRICS:\n")
                    f.write("-" * 20 + "\n")
                    basic = result["basic_metrics"]
                    for key, value in basic.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # Triple barrier results
                    f.write("TRIPLE BARRIER ANALYSIS:\n")
                    f.write("-" * 30 + "\n")
                    for barrier_name, barrier_data in result["triple_barrier_profits"].items():
                        barrier_level = int(barrier_name.split("_")[1].replace("bp", "")) / 1000
                        f.write(f"\nBarrier Level: {barrier_level:.1%}\n")
                        for key, value in barrier_data.items():
                            if isinstance(value, float):
                                f.write(f"  {key}: {value:.6f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    f.write("\n")
                    
                    # Volume analysis
                    f.write("VOLUME ANALYSIS:\n")
                    f.write("-" * 20 + "\n")
                    volume = result["volume_analysis"]
                    for key, value in volume.items():
                        if key == "volume_percentiles":
                            f.write(f"{key}:\n")
                            for pkey, pvalue in value.items():
                                f.write(f"  {pkey}: {pvalue:.2f}\n")
                        elif isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # Movement statistics
                    f.write("MOVEMENT STATISTICS:\n")
                    f.write("-" * 25 + "\n")
                    movement = result["movement_statistics"]
                    for key, value in movement.items():
                        if key == "movement_percentiles":
                            f.write(f"{key}:\n")
                            for pkey, pvalue in value.items():
                                f.write(f"  {pkey}: {pvalue:.6f}\n")
                        elif isinstance(value, float):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # Intraday patterns
                    f.write("INTRADAY PATTERNS:\n")
                    f.write("-" * 20 + "\n")
                    intraday = result["intraday_patterns"]
                    f.write(f"Peak Hours: {intraday['peak_hours']}\n")
                    f.write(f"Best Trading Hours: {intraday['best_trading_hours']}\n")
                    f.write("\n")
                
                logger.info(f"Detailed report saved for {symbol}")
            
            # Generate summary comparison report
            summary_file = reports_dir / "assets_comparison_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("CRYPTOCURRENCY ASSETS COMPARISON SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Rank by total return
                returns = [(symbol, result["basic_metrics"]["total_return"]) 
                          for symbol, result in self.analyzer.results.items()]
                returns.sort(key=lambda x: x[1], reverse=True)
                
                f.write("RANKING BY TOTAL RETURN:\n")
                f.write("-" * 30 + "\n")
                for i, (symbol, return_val) in enumerate(returns, 1):
                    f.write(f"{i:2d}. {symbol:10s}: {return_val:8.4f} ({return_val*100:6.2f}%)\n")
                f.write("\n")
                
                # Rank by volatility
                volatilities = [(symbol, result["basic_metrics"]["volatility"]) 
                               for symbol, result in self.analyzer.results.items()]
                volatilities.sort(key=lambda x: x[1], reverse=True)
                
                f.write("RANKING BY VOLATILITY:\n")
                f.write("-" * 25 + "\n")
                for i, (symbol, vol) in enumerate(volatilities, 1):
                    f.write(f"{i:2d}. {symbol:10s}: {vol:8.4f} ({vol*100:6.2f}%)\n")
                f.write("\n")
                
                # Rank by average daily range
                daily_ranges = [(symbol, result["basic_metrics"]["avg_daily_range"]) 
                               for symbol, result in self.analyzer.results.items()]
                daily_ranges.sort(key=lambda x: x[1], reverse=True)
                
                f.write("RANKING BY AVERAGE DAILY RANGE:\n")
                f.write("-" * 35 + "\n")
                for i, (symbol, range_val) in enumerate(daily_ranges, 1):
                    f.write(f"{i:2d}. {symbol:10s}: {range_val:8.4f} ({range_val*100:6.2f}%)\n")
                f.write("\n")
                
                # Volume rankings
                volumes = [(symbol, result["volume_analysis"]["total_volume"]) 
                          for symbol, result in self.analyzer.results.items()]
                volumes.sort(key=lambda x: x[1], reverse=True)
                
                f.write("RANKING BY TOTAL VOLUME:\n")
                f.write("-" * 30 + "\n")
                for i, (symbol, volume) in enumerate(volumes, 1):
                    f.write(f"{i:2d}. {symbol:10s}: {volume:15,.0f}\n")
                f.write("\n")
            
            logger.info("Detailed reports generated successfully")
            
        except Exception as e:
            logger.exception(f"Error generating detailed reports: {e}")
    
    async def process_all_assets(self, years: int = 2, api_key: str = "", api_secret: str = "") -> Dict[str, Any]:
        """
        Complete processing pipeline: download data and run analysis
        
        Args:
            years: Number of years of historical data
            api_key: Binance API key (optional)
            api_secret: Binance API secret (optional)
            
        Returns:
            Dictionary containing processing results and summary
        """
        try:
            logger.info("🚀 Starting enhanced cryptocurrency analysis pipeline")
            
            # Step 1: Enhanced data acquisition
            data_file = await self.download_data(years, api_key, api_secret, use_existing=True)
            if not data_file:
                return {
                    "success": False,
                    "error": "Data download failed",
                    "summary": {
                        "total_assets": len(self.assets),
                        "successfully_processed": 0,
                        "success_rate": 0.0
                    },
                    "assets_processed": [],
                    "assets_failed": [{"asset": asset, "error": "Data download failed"} for asset in self.assets],
                    "all_metrics": {}
                }
            
            # Step 2: Analyze data
            analysis_result = self.analyze_data(data_file)
            if not analysis_result["success"]:
                return {
                    "success": False,
                    "error": f"Data analysis failed: {analysis_result.get('error', 'Unknown error')}",
                    "summary": {
                        "total_assets": len(self.assets),
                        "successfully_processed": 0,
                        "success_rate": 0.0
                    },
                    "assets_processed": [],
                    "assets_failed": [{"asset": asset, "error": "Data analysis failed"} for asset in self.assets],
                    "all_metrics": {}
                }
            
            # Extract results
            processed_assets = list(analysis_result["results"].keys())
            failed_assets = [asset for asset in self.assets if asset not in processed_assets]
            
            # Create metrics summary
            all_metrics = {}
            for asset in processed_assets:
                result = analysis_result["results"][asset]
                all_metrics[asset] = {
                    "price_metrics": result["basic_metrics"],
                    "volume_metrics": result["volume_analysis"],
                    "movement_stats": result["movement_statistics"]
                }
            
            success_rate = len(processed_assets) / len(self.assets) * 100
            
            logger.info(f"✅ Enhanced pipeline completed successfully: {len(processed_assets)}/{len(self.assets)} assets processed ({success_rate:.1f}%)")
            
            return {
                "success": True,
                "enhanced_processing": True,
                "optimization_status": {
                    "ares_utilities_available": ARES_UTILS_AVAILABLE,
                    "hardware_optimizations_available": HARDWARE_OPT_AVAILABLE,
                    "parquet_utils_enabled": self.parquet_utils is not None,
                    "memory_optimization_enabled": self.memory_optimizer is not None,
                    "gpu_acceleration_enabled": self.gpu_manager is not None,
                    "cpu_optimization_enabled": self.cpu_optimizer is not None
                },
                "summary": {
                    "total_assets": len(self.assets),
                    "successfully_processed": len(processed_assets),
                    "success_rate": success_rate
                },
                "assets_processed": processed_assets,
                "assets_failed": [{"asset": asset, "error": "Processing failed"} for asset in failed_assets],
                "all_metrics": all_metrics,
                "data_file": str(data_file),
                "analysis_summary": analysis_result.get("summary", {})
            }
            
        except Exception as e:
            logger.exception(f"Error in complete processing pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": {
                    "total_assets": len(self.assets),
                    "successfully_processed": 0,
                    "success_rate": 0.0
                },
                "assets_processed": [],
                "assets_failed": [{"asset": asset, "error": str(e)} for asset in self.assets],
                "all_metrics": {}
            }
    
    def cleanup(self):
        """Enhanced cleanup using Ares utilities"""
        try:
            # Cleanup hardware resources
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.cleanup()
                    logger.info("🧠 Memory optimization cleanup completed")
                except Exception as e:
                    logger.warning(f"⚠️ Memory cleanup failed: {e}")
            
            if self.gpu_manager:
                try:
                    self.gpu_manager.cleanup()
                    logger.info("🚀 GPU cleanup completed")
                except Exception as e:
                    logger.warning(f"⚠️ GPU cleanup failed: {e}")
            
            # Close any open connections
            if hasattr(self.downloader, 'session'):
                self.downloader.session.close()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("🧹 Enhanced cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Error during enhanced cleanup: {e}")
