# src/training/steps/step2_market_regime_classification.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.logger import system_logger


def convert_trade_data_to_ohlcv(trade_data: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """
    Convert trade data to OHLCV format.
    
    Args:
        trade_data: DataFrame with columns ['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id']
        timeframe: Timeframe for resampling (e.g., '1h', '1m', '1d')
    
    Returns:
        DataFrame with OHLCV columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    """
    try:
        # Make a copy to avoid modifying original data
        df = trade_data.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index for resampling
        df = df.set_index('timestamp')
        
        # Resample to the specified timeframe and calculate OHLCV
        ohlcv = df.resample(timeframe).agg({
            'price': ['first', 'max', 'min', 'last'],
            'quantity': 'sum'
        })
        
        # Flatten column names
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Reset index to create timestamp column
        ohlcv = ohlcv.reset_index()
        
        # Rename the index column to 'timestamp'
        ohlcv = ohlcv.rename(columns={ohlcv.index.name: 'timestamp'})
        
        # Remove any rows with NaN values
        ohlcv = ohlcv.dropna()
        
        return ohlcv
        
    except Exception as e:
        system_logger.error(f"Error converting trade data to OHLCV: {e}")
        raise


class MarketRegimeClassificationStep:
    """Step 2: Market Regime Classification using UnifiedRegimeClassifier."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.regime_classifier = None

    async def initialize(self) -> None:
        """Initialize the market regime classification step."""
        self.logger.info("Initializing Market Regime Classification Step...")

        # Initialize the unified regime classifier (will be re-initialized with exchange/symbol in execute)
        self.regime_classifier = None

        self.logger.info(
            "Market Regime Classification Step initialized successfully",
        )

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute market regime classification.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing regime classification results
        """
        self.logger.info("🔄 Executing Market Regime Classification...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        
        # Initialize the unified regime classifier with exchange and symbol
        if self.regime_classifier is None:
            self.regime_classifier = UnifiedRegimeClassifier(self.config, exchange, symbol)
            await self.regime_classifier.initialize()
        
        # Try to load pre-consolidated data first
        consolidated_file = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
        data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
        
        historical_data = None
        
        # Try consolidated parquet file first
        if os.path.exists(consolidated_file):
            self.logger.info(f"📁 Loading pre-consolidated data from: {consolidated_file}")
            try:
                from src.training.enhanced_training_manager_optimized import MemoryEfficientDataManager
                trade_data = MemoryEfficientDataManager().load_from_parquet(consolidated_file)
                self.logger.info(f"✅ Loaded consolidated trade data: {len(trade_data)} records")
                
                # Convert trade data to OHLCV format
                self.logger.info("🔄 Converting trade data to OHLCV format...")
                # Normalize timestamp just in case
                if 'timestamp' in trade_data.columns:
                    trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'], utc=True, errors='coerce')
                    trade_data = trade_data.dropna(subset=['timestamp']).sort_values('timestamp')
                historical_data = convert_trade_data_to_ohlcv(trade_data, timeframe="1h")
                self.logger.info(f"✅ Converted to OHLCV: {len(historical_data)} records")
                
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load consolidated data: {e}")
                historical_data = None
        
        # Fallback to pickle file if consolidated data not available
        if historical_data is None:
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found: {data_file_path}")
            
            self.logger.info(f"📁 Loading data from pickle file: {data_file_path}")
            import pickle
            
            with open(data_file_path, "rb") as f:
                payload = pickle.load(f)
            
            # Expect a dict with keys like 'klines', 'agg_trades', 'futures'
            if isinstance(payload, dict):
                historical_data = payload.get("klines")
                if historical_data is None:
                    # Fallback: prefer any non-empty DataFrame in payload
                    historical_data = next((df for df in payload.values() if isinstance(df, pd.DataFrame) and not df.empty), None)
                if historical_data is None:
                    raise ValueError(f"No usable DataFrame found inside {data_file_path}")
            elif isinstance(payload, pd.DataFrame):
                historical_data = payload
            else:
                raise ValueError(f"Unsupported pickle payload type: {type(payload)} from {data_file_path}")
            
            # Ensure 'timestamp' column exists
            if isinstance(historical_data.index, pd.DatetimeIndex) and 'timestamp' not in historical_data.columns:
                historical_data = historical_data.copy()
                historical_data['timestamp'] = historical_data.index
            
            self.logger.info(f"✅ Loaded historical OHLCV data: {len(historical_data)} records")

        # Perform regime classification
        regime_results = await self._classify_market_regimes(
            historical_data,
            symbol,
            exchange,
        )

        # Save regime classification results
        regime_file_path = (
            f"{data_dir}/{exchange}_{symbol}_regime_classification.json"
        )
        os.makedirs(os.path.dirname(regime_file_path), exist_ok=True)

        with open(regime_file_path, "w") as f:
            json.dump(regime_results, f, indent=2)

        self.logger.info(
            f"✅ Market regime classification completed. Results saved to {regime_file_path}",
        )

        # Update pipeline state
        pipeline_state["regime_classification"] = regime_results
        pipeline_state["regime_file_path"] = regime_file_path

        return {
            "regime_classification": regime_results,
            "regime_file_path": regime_file_path,
            "duration": 0.0,  # Will be calculated in actual implementation
            "status": "SUCCESS",
        }

    async def _classify_market_regimes(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """
        Classify market regimes using the unified regime classifier.

        Args:
            data: Historical market data
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing regime classification results
        """
        try:
            self.logger.info(
                f"Classifying market regimes for {symbol} on {exchange}...",
            )

            # Prepare data for regime classification
            # The unified regime classifier expects specific column names
            required_columns = ["open", "high", "low", "close", "volume", "timestamp"]

            # Rename columns if needed
            column_mapping = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Timestamp": "timestamp",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})

            # Ensure we have the required columns
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns for regime classification: {missing_columns}",
                )

            # Sort by timestamp
            if "timestamp" in data.columns:
                data = data.sort_values("timestamp")

            # Perform regime classification
            regime_results = await self.regime_classifier.classify_regimes(data)

            # Process and format results
            formatted_results = {
                "symbol": symbol,
                "exchange": exchange,
                "classification_date": datetime.now().isoformat(),
                "total_records": len(data),
                "regime_distribution": {},
                "regime_sequence": [],
                "regime_transitions": [],
                "confidence_scores": {},
                "metadata": {
                    "classifier_version": "unified_regime_classifier",
                    "classification_method": "HMM_ensemble",
                },
            }

            # Extract regime information from classifier results
            if isinstance(regime_results, dict):
                # If the classifier returns a dict with regime information
                if "regimes" in regime_results:
                    regimes = regime_results["regimes"]
                    if isinstance(regimes, list):
                        # The regime sequence from the classifier may be shorter due to feature calculation
                        # We need to map it back to the original data length
                        original_data_length = len(data)
                        regime_sequence_length = len(regimes)
                        
                        if regime_sequence_length < original_data_length:
                            self.logger.info(
                                f"Regime sequence length ({regime_sequence_length}) is shorter than original data length ({original_data_length}). "
                                f"Mapping regimes to original data length..."
                            )
                            
                            # Create a regime sequence that matches the original data length
                            # We'll use interpolation to map the regimes back to the full data
                            if regime_sequence_length > 0:
                                # Create indices for the regime sequence
                                regime_indices = np.linspace(0, original_data_length - 1, regime_sequence_length, dtype=int)
                                
                                # Create a full-length regime sequence
                                full_regime_sequence = []
                                for i in range(original_data_length):
                                    # Find the closest regime index
                                    closest_idx = np.argmin(np.abs(regime_indices - i))
                                    full_regime_sequence.append(regimes[closest_idx])
                                
                                formatted_results["regime_sequence"] = full_regime_sequence
                                self.logger.info(f"Mapped regime sequence to original data length: {len(full_regime_sequence)}")
                            else:
                                # Fallback: use default regime
                                formatted_results["regime_sequence"] = ["SIDEWAYS"] * original_data_length
                                self.logger.warning("No regimes available, using default SIDEWAYS regime")
                        else:
                            # Regime sequence is long enough, truncate if needed
                            formatted_results["regime_sequence"] = regimes[:original_data_length]

                        # Calculate regime distribution
                        from collections import Counter
                        regime_counts = Counter(formatted_results["regime_sequence"])
                        formatted_results["regime_distribution"] = dict(regime_counts)

                        # Calculate regime transitions
                        transitions = []
                        for i in range(1, len(formatted_results["regime_sequence"])):
                            if formatted_results["regime_sequence"][i] != formatted_results["regime_sequence"][i - 1]:
                                transitions.append(
                                    {
                                        "from_regime": formatted_results["regime_sequence"][i - 1],
                                        "to_regime": formatted_results["regime_sequence"][i],
                                        "transition_index": i,
                                    },
                                )
                        formatted_results["regime_transitions"] = transitions

                # Extract confidence scores if available
                if "confidence_scores" in regime_results:
                    formatted_results["confidence_scores"] = regime_results[
                        "confidence_scores"
                    ]

            self.logger.info(
                f"Regime classification completed. Found {len(formatted_results['regime_distribution'])} distinct regimes",
            )

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the market regime classification step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = MarketRegimeClassificationStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception as e:
        print(f"❌ Market regime classification failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
