"""
Refactored DataCollectionStep with reduced complexity and type hints.
This version breaks down the massive _log_detailed_data_extract method into smaller,
focused methods using the Strategy pattern for different data types.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Protocol
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class DataType(Enum):
    """Types of data to analyze"""
    KLINES = "klines"
    AGGTRADES = "aggtrades"
    FUTURES = "futures"
    SPOT = "spot"


@dataclass
class DataQualityMetrics:
    """Container for data quality metrics"""
    shape: Tuple[int, int]
    file_size: int
    columns: List[str]
    dtypes: Dict[str, str]
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
    missing_values: Dict[str, int]
    duplicate_count: int
    infinite_values: Dict[str, int]
    zero_price_count: Dict[str, int]
    value_ranges: Dict[str, Tuple[float, float, float]]  # min, max, mean


@dataclass
class DataExtractConfig:
    """Configuration for data extraction logging"""
    show_sample_rows: int = 5
    check_duplicates: bool = True
    check_missing: bool = True
    check_infinite: bool = True
    check_zero_prices: bool = True
    log_value_ranges: bool = True
    log_memory_usage: bool = True


class DataAnalyzer(ABC):
    """Abstract base class for data analysis strategies"""
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, file_path: Path) -> DataQualityMetrics:
        """Analyze the dataframe and return quality metrics"""
        pass
    
    @abstractmethod
    def format_report(self, metrics: DataQualityMetrics, logger: logging.Logger) -> None:
        """Format and log the analysis report"""
        pass


class KlinesDataAnalyzer(DataAnalyzer):
    """Analyzer for klines (OHLCV) data"""
    
    def analyze(self, df: pd.DataFrame, file_path: Path) -> DataQualityMetrics:
        """Analyze klines data"""
        metrics = DataQualityMetrics(
            shape=df.shape,
            file_size=file_path.stat().st_size,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            date_range=None,
            missing_values={},
            duplicate_count=0,
            infinite_values={},
            zero_price_count={},
            value_ranges={}
        )
        
        # Analyze timestamp range
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                metrics.date_range = (df["timestamp"].min(), df["timestamp"].max())
                metrics.duplicate_count = df.duplicated(subset=["timestamp"]).sum()
            except Exception:
                pass
        
        # Analyze missing values
        metrics.missing_values = df.isnull().sum().to_dict()
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for infinite values
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                metrics.infinite_values[col] = inf_count
        
        # Check for zero prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    metrics.zero_price_count[col] = zero_count
        
        # Calculate value ranges
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                metrics.value_ranges[col] = (
                    float(col_data.min()),
                    float(col_data.max()),
                    float(col_data.mean())
                )
        
        return metrics
    
    def format_report(self, metrics: DataQualityMetrics, logger: logging.Logger) -> None:
        """Format and log klines analysis report"""
        logger.info("📊 Klines Data Analysis:")
        logger.info(f"   Shape: {metrics.shape}")
        logger.info(f"   File size: {metrics.file_size:,} bytes")
        logger.info(f"   Columns: {metrics.columns}")
        
        if metrics.date_range:
            start, end = metrics.date_range
            days = (end - start).days
            logger.info(f"   Date range: {start} to {end} ({days} days)")
        
        # Report data quality issues
        if any(metrics.missing_values.values()):
            logger.warning("   ⚠️ Missing values found:")
            for col, count in metrics.missing_values.items():
                if count > 0:
                    pct = (count / metrics.shape[0]) * 100
                    logger.warning(f"      - {col}: {count} ({pct:.2f}%)")
        else:
            logger.info("   ✅ No missing values")
        
        if metrics.duplicate_count > 0:
            logger.warning(f"   ⚠️ Found {metrics.duplicate_count} duplicate timestamps")
        else:
            logger.info("   ✅ No duplicate timestamps")
        
        if metrics.infinite_values:
            logger.warning("   ⚠️ Infinite values found:")
            for col, count in metrics.infinite_values.items():
                logger.warning(f"      - {col}: {count}")
        
        if metrics.zero_price_count:
            logger.warning("   ⚠️ Zero prices found:")
            for col, count in metrics.zero_price_count.items():
                logger.warning(f"      - {col}: {count}")
        
        # Report value ranges
        if metrics.value_ranges:
            logger.info("   📈 Value ranges:")
            for col, (min_val, max_val, mean_val) in metrics.value_ranges.items():
                logger.info(f"      - {col}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")


class AggtradesDataAnalyzer(DataAnalyzer):
    """Analyzer for aggregated trades data"""
    
    def analyze(self, df: pd.DataFrame, file_path: Path) -> DataQualityMetrics:
        """Analyze aggtrades data"""
        metrics = DataQualityMetrics(
            shape=df.shape,
            file_size=file_path.stat().st_size,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            date_range=None,
            missing_values={},
            duplicate_count=0,
            infinite_values={},
            zero_price_count={},
            value_ranges={}
        )
        
        # Analyze timestamp range
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                metrics.date_range = (df["timestamp"].min(), df["timestamp"].max())
                
                # For aggtrades, check duplicates on trade ID if available
                if "trade_id" in df.columns:
                    metrics.duplicate_count = df.duplicated(subset=["trade_id"]).sum()
                else:
                    metrics.duplicate_count = df.duplicated(subset=["timestamp"]).sum()
            except Exception:
                pass
        
        # Analyze missing values
        metrics.missing_values = df.isnull().sum().to_dict()
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for zero prices
        if "price" in df.columns:
            zero_count = (df["price"] == 0).sum()
            if zero_count > 0:
                metrics.zero_price_count["price"] = zero_count
        
        # Calculate value ranges
        for col in ["price", "quantity", "quote_quantity"]:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    metrics.value_ranges[col] = (
                        float(col_data.min()),
                        float(col_data.max()),
                        float(col_data.mean())
                    )
        
        return metrics
    
    def format_report(self, metrics: DataQualityMetrics, logger: logging.Logger) -> None:
        """Format and log aggtrades analysis report"""
        logger.info("📊 Aggtrades Data Analysis:")
        logger.info(f"   Shape: {metrics.shape}")
        logger.info(f"   File size: {metrics.file_size:,} bytes")
        logger.info(f"   Columns: {metrics.columns}")
        
        if metrics.date_range:
            start, end = metrics.date_range
            hours = (end - start).total_seconds() / 3600
            logger.info(f"   Date range: {start} to {end} ({hours:.2f} hours)")
        
        # Report data quality
        if metrics.duplicate_count > 0:
            logger.warning(f"   ⚠️ Found {metrics.duplicate_count} duplicate trades")
        else:
            logger.info("   ✅ No duplicate trades")
        
        if metrics.zero_price_count:
            logger.warning(f"   ⚠️ Found {metrics.zero_price_count['price']} zero prices")
        
        # Report trade statistics
        if metrics.value_ranges:
            logger.info("   📈 Trade statistics:")
            for col, (min_val, max_val, mean_val) in metrics.value_ranges.items():
                logger.info(f"      - {col}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")


class DataCollectionLoggerRefactored:
    """Refactored data collection logger with reduced complexity"""
    
    def __init__(
        self,
        config: Optional[DataExtractConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the logger.
        
        Args:
            config: Configuration for data extraction
            logger: Logger instance
        """
        self.config = config or DataExtractConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize analyzers
        self.analyzers: Dict[DataType, DataAnalyzer] = {
            DataType.KLINES: KlinesDataAnalyzer(),
            DataType.AGGTRADES: AggtradesDataAnalyzer(),
        }
    
    async def log_detailed_data_extract(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Log detailed data extract with reduced complexity.
        
        This refactored method delegates to specialized analyzers for each data type.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            logger: Logger instance (optional)
        """
        log = logger or self.logger
        
        log.info("=" * 80)
        log.info("📊 DETAILED DATA EXTRACT FOR TROUBLESHOOTING")
        log.info("=" * 80)
        log.info(f"Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
        log.info(f"Data Directory: {data_dir}")
        log.info("=" * 80)
        
        # Define files to analyze
        files_to_analyze = [
            (DataType.KLINES, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"),
            (DataType.AGGTRADES, f"aggtrades_{exchange}_{symbol}_consolidated.parquet"),
        ]
        
        # Analyze each file
        for data_type, filename in files_to_analyze:
            file_path = Path(data_dir) / filename
            
            if file_path.exists():
                await self._analyze_file(data_type, file_path, log)
            else:
                log.warning(f"⚠️ File not found: {file_path}")
        
        # Log summary
        self._log_summary(log)
    
    async def _analyze_file(
        self,
        data_type: DataType,
        file_path: Path,
        logger: logging.Logger
    ) -> None:
        """Analyze a single data file"""
        logger.info(f"\n🔍 Analyzing {data_type.value} data: {file_path.name}")
        
        try:
            # Load the data
            df = pd.read_parquet(file_path)
            
            # Get appropriate analyzer
            analyzer = self.analyzers.get(data_type)
            if not analyzer:
                logger.warning(f"⚠️ No analyzer available for {data_type}")
                return
            
            # Analyze the data
            metrics = analyzer.analyze(df, file_path)
            
            # Log sample data if configured
            if self.config.show_sample_rows > 0:
                self._log_sample_data(df, logger)
            
            # Format and log the report
            analyzer.format_report(metrics, logger)
            
            # Log memory usage if configured
            if self.config.log_memory_usage:
                self._log_memory_usage(df, logger)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_path}: {e}")
    
    def _log_sample_data(self, df: pd.DataFrame, logger: logging.Logger) -> None:
        """Log sample rows from the dataframe"""
        n_rows = min(self.config.show_sample_rows, len(df))
        
        if n_rows > 0:
            logger.info(f"\n   📋 Sample data (first {n_rows} rows):")
            sample_df = df.head(n_rows)
            
            for idx, row in sample_df.iterrows():
                formatted_row = self._format_row(row)
                logger.info(f"      Row {idx}: {formatted_row}")
            
            # Also log last few rows
            if len(df) > n_rows * 2:
                logger.info(f"\n   📋 Sample data (last {n_rows} rows):")
                sample_df_last = df.tail(n_rows)
                
                for idx, row in sample_df_last.iterrows():
                    formatted_row = self._format_row(row)
                    logger.info(f"      Row {idx}: {formatted_row}")
    
    def _format_row(self, row: pd.Series) -> Dict[str, str]:
        """Format a row for logging"""
        formatted = {}
        
        for col, val in row.items():
            if pd.isna(val):
                formatted[col] = "NaN"
            elif isinstance(val, (int, np.integer)):
                formatted[col] = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted[col] = f"{val:.6f}"
            elif isinstance(val, pd.Timestamp):
                formatted[col] = val.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted[col] = str(val)
        
        return formatted
    
    def _log_memory_usage(self, df: pd.DataFrame, logger: logging.Logger) -> None:
        """Log memory usage of the dataframe"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        logger.info(f"\n   💾 Memory usage:")
        logger.info(f"      Total: {total_memory / 1024 / 1024:.2f} MB")
        
        # Log per-column memory if not too many columns
        if len(df.columns) <= 20:
            for col, mem in memory_usage.items():
                if col != "Index":
                    logger.info(f"      - {col}: {mem / 1024:.2f} KB")
    
    def _log_summary(self, logger: logging.Logger) -> None:
        """Log analysis summary"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 DATA EXTRACT ANALYSIS COMPLETE")
        logger.info("=" * 80)


# Example usage with the existing DataCollectionStep
class DataCollectionStepAdapter:
    """Adapter to integrate refactored logger with existing DataCollectionStep"""
    
    def __init__(self, original_step):
        self.original_step = original_step
        self.logger = DataCollectionLoggerRefactored()
    
    async def _log_detailed_data_extract(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        logger: Any
    ) -> None:
        """Delegate to refactored logger"""
        await self.logger.log_detailed_data_extract(
            symbol, exchange, timeframe, data_dir, logger
        )