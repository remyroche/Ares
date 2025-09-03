"""Step 1: Data Collection - Refactored to use BaseStep.

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""

from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import os
import pandas as pd
from datetime import datetime, timedelta

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors


class DataCollectionStep(BaseStep):
    """Step 1: Data Collection using standardized base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data collection step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "01", "data_collection")
        
        # Step-specific configuration
        self.lookback_years = config.get("lookback_years", 2)
        self.data_sources = config.get("data_sources", ["binance"])
        self.intervals = config.get("intervals", ["1m"])
        self.data_downloader = None
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Try to import data downloader
        try:
            from src.training.steps.data_downloader import download_all_data_with_consolidation
            self.data_downloader = download_all_data_with_consolidation
            self.logger.info("✅ Data downloader initialized")
        except ImportError:
            self.logger.warning("⚠️ Data downloader not available, will use mock data")
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check required parameters
        if not training_input.get("symbol"):
            errors.append("Symbol is required")
        
        if not training_input.get("exchange"):
            errors.append("Exchange is required")
        
        # Validate symbol format
        symbol = training_input.get("symbol", "")
        if symbol and not symbol.isupper():
            errors.append(f"Symbol should be uppercase, got: {symbol}")
        
        # Validate exchange
        valid_exchanges = ["binance", "bybit", "okx", "kraken"]
        exchange = training_input.get("exchange", "").lower()
        if exchange and exchange not in valid_exchanges:
            errors.append(f"Invalid exchange: {exchange}. Valid: {valid_exchanges}")
        
        # Validate timeframe
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        timeframe = training_input.get("timeframe", "1m")
        if timeframe not in valid_timeframes:
            errors.append(f"Invalid timeframe: {timeframe}. Valid: {valid_timeframes}")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="data collection execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data collection logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        # Extract parameters
        symbol = training_input["symbol"]
        exchange = training_input["exchange"]
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")
        
        self.logger.info(f"📥 Collecting data for {symbol} on {exchange} ({timeframe})")
        
        # Build output paths
        output_dir = Path(data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        consolidated_file = output_dir / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        
        # Check if data already exists
        if consolidated_file.exists() and not training_input.get("force_download", False):
            self.logger.info(f"📁 Found existing data: {consolidated_file}")
            
            # Load and validate existing data
            try:
                data = pd.read_parquet(consolidated_file)
                self.logger.info(f"✅ Loaded {len(data)} rows of existing data")
                
                # Update pipeline state
                pipeline_state["raw_market_data"] = consolidated_file
                pipeline_state["data_shape"] = data.shape
                pipeline_state["data_columns"] = list(data.columns)
                pipeline_state["data_date_range"] = {
                    "start": str(data.index.min() if not data.empty else None),
                    "end": str(data.index.max() if not data.empty else None)
                }
                
                return pipeline_state
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load existing data: {e}")
        
        # Download new data
        if self.data_downloader:
            try:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * self.lookback_years)
                
                self.logger.info(
                    f"📥 Downloading data from {start_date.date()} to {end_date.date()}"
                )
                
                # Call data downloader
                success = await self.data_downloader(
                    symbol=symbol,
                    exchange=exchange,
                    interval=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=str(output_dir)
                )
                
                if success and consolidated_file.exists():
                    # Load downloaded data
                    data = pd.read_parquet(consolidated_file)
                    self.logger.info(f"✅ Downloaded {len(data)} rows of data")
                    
                    # Update pipeline state
                    pipeline_state["raw_market_data"] = str(consolidated_file)
                    pipeline_state["data_shape"] = data.shape
                    pipeline_state["data_columns"] = list(data.columns)
                    pipeline_state["data_date_range"] = {
                        "start": str(data.index.min() if not data.empty else None),
                        "end": str(data.index.max() if not data.empty else None)
                    }
                else:
                    raise RuntimeError("Data download failed")
                    
            except Exception as e:
                self.logger.error(f"❌ Data download failed: {e}")
                # Fall back to mock data
                data = self._generate_mock_data(symbol, exchange, timeframe)
                data.to_parquet(consolidated_file)
                pipeline_state["raw_market_data"] = str(consolidated_file)
                pipeline_state["data_shape"] = data.shape
                pipeline_state["data_columns"] = list(data.columns)
                pipeline_state["is_mock_data"] = True
        else:
            # Generate mock data
            self.logger.info("📊 Generating mock data (downloader not available)")
            data = self._generate_mock_data(symbol, exchange, timeframe)
            data.to_parquet(consolidated_file)
            pipeline_state["raw_market_data"] = str(consolidated_file)
            pipeline_state["data_shape"] = data.shape
            pipeline_state["data_columns"] = list(data.columns)
            pipeline_state["is_mock_data"] = True
        
        # Add metadata
        pipeline_state["data_collection_metadata"] = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "lookback_years": self.lookback_years,
            "download_timestamp": datetime.now().isoformat()
        }
        
        return pipeline_state
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if raw market data path exists
        if "raw_market_data" not in pipeline_state:
            errors.append("No raw_market_data in pipeline state")
            return False, errors
        
        # Check if file exists
        data_path = Path(pipeline_state["raw_market_data"])
        if not data_path.exists():
            errors.append(f"Data file does not exist: {data_path}")
            return False, errors
        
        # Validate data shape
        if "data_shape" in pipeline_state:
            rows, cols = pipeline_state["data_shape"]
            if rows < 100:
                errors.append(f"Insufficient data rows: {rows} (minimum 100 required)")
            if cols < 5:
                errors.append(f"Insufficient columns: {cols} (minimum 5 required)")
        
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        if "data_columns" in pipeline_state:
            missing_cols = set(required_columns) - set(pipeline_state["data_columns"])
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
        
        return len(errors) == 0, errors
    
    def _generate_mock_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> pd.DataFrame:
        """Generate mock market data for testing.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            
        Returns:
            Mock DataFrame with OHLCV data
        """
        import numpy as np
        
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days of data
        
        # Map timeframe to frequency
        freq_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "1H", "4h": "4H", "1d": "1D"
        }
        freq = freq_map.get(timeframe, "1min")
        
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate price data
        np.random.seed(42)  # For reproducibility
        n_points = len(timestamps)
        
        # Brownian motion for price
        returns = np.random.normal(0.0001, 0.01, n_points)
        price = 50000 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        data = pd.DataFrame({
            "timestamp": timestamps,
            "open": price * (1 + np.random.uniform(-0.001, 0.001, n_points)),
            "high": price * (1 + np.random.uniform(0, 0.005, n_points)),
            "low": price * (1 - np.random.uniform(0, 0.005, n_points)),
            "close": price,
            "volume": np.random.uniform(100, 1000, n_points)
        })
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)
        
        # Set timestamp as index
        data.set_index("timestamp", inplace=True)
        
        self.logger.info(f"📊 Generated {len(data)} rows of mock data")
        
        return data
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["symbol", "exchange", "timeframe"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ["raw_market_data", "data_shape", "data_columns", "data_date_range"]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return []  # First step has no dependencies