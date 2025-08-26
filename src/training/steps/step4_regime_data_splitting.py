#!/usr/bin/env python3
"""Step 4: Regime Data Splitting.

This module splits data by HMM regimes for regime-specific processing.
Supports 10+ regimes with efficient memory management and parallel processing.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
    monitor_feature_engineering,
)

logger = system_logger.getChild("Step4RegimeDataSplitting")


class RegimeDataSplittingStep:
    """Step 4: Regime Data Splitting with support for 10+ regimes."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("RegimeDataSplittingStep")
        self.start_time = None
        self.step_timings = {}

    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing Regime Data Splitting Step...")
        self.logger.info("📋 Step 4 Configuration:")
        self.logger.info(f"   - Max regimes supported: 20")
        self.logger.info(f"   - Parallel processing: Enabled")
        self.logger.info(f"   - Memory management: Optimized")
        self.logger.info("✅ Regime Data Splitting Step initialized successfully")

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("split_data_by_regimes")
    @quality_gate(
        min_quality_score=0.8,
        max_correlation=0.95,
        required_grade="B"
    )
    @comprehensive_data_validation
    @memory_efficient
    async def split_data_by_regimes(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> bool:
        """Split data by HMM regimes for regime-specific processing."""
        step_start = time.time()
        self.logger.info(f"🔀 Splitting data by regimes for {symbol} on {exchange} ({timeframe})")
        
        try:
            # Load HMM regime data
            regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                return False
            
            # Get unique regime IDs
            regime_ids = regime_data['composite_cluster_id'].unique()
            num_regimes = len(regime_ids)
            
            self.logger.info(f"📊 Found {num_regimes} regimes: {sorted(regime_ids)}")
            
            # Validate regime count
            if num_regimes < 3:
                self.logger.error(f"❌ Too few regimes: {num_regimes} (minimum 3 required)")
                return False
            
            if num_regimes > 20:
                self.logger.warning(f"⚠️ Many regimes detected: {num_regimes} (maximum 20 supported)")
                # Continue but with memory optimization
            
            # Create regime-specific directories
            regime_base_dir = Path(data_dir) / "training" / "regimes" / f"{exchange}_{symbol}_{timeframe}"
            regime_base_dir.mkdir(parents=True, exist_ok=True)
            
            # Split data by regimes with parallel processing
            success = await self._process_regimes_parallel(
                regime_data, regime_ids, regime_base_dir, num_regimes
            )
            
            if success:
                self._log_step_timing("Regime Data Splitting", step_start)
                self.logger.info(f"✅ Successfully split data into {num_regimes} regimes")
                
                # Save regime metadata
                await self._save_regime_metadata(regime_ids, regime_base_dir)
                
                return True
            else:
                self.logger.error("❌ Failed to split data by regimes")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Error in regime data splitting: {e}")
            return False

    async def _load_regime_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load HMM regime data."""
        try:
            # Load unified data
            unified_data_path = Path(data_dir) / "unified" / exchange / symbol / timeframe
            if not unified_data_path.exists():
                self.logger.error(f"❌ Unified data path not found: {unified_data_path}")
                return None
            
            # Load regime clusters
            regime_file = Path(data_dir) / "hmm_regimes" / f"{exchange}_{symbol}_{timeframe}_composite_clusters.parquet"
            if not regime_file.exists():
                self.logger.error(f"❌ Regime file not found: {regime_file}")
                return None
            
            # Load data
            unified_files = list(unified_data_path.glob("*.parquet"))
            if not unified_files:
                self.logger.error(f"❌ No unified data files found in {unified_data_path}")
                return None
            
            # Load and concatenate unified data
            unified_data = []
            for file_path in sorted(unified_files):
                df = pd.read_parquet(file_path)
                unified_data.append(df)
            
            unified_df = pd.concat(unified_data, ignore_index=True)
            regime_df = pd.read_parquet(regime_file)
            
            # Merge unified data with regime information
            merged_data = pd.merge(
                unified_df, 
                regime_df[['timestamp', 'composite_cluster_id']], 
                on='timestamp', 
                how='inner'
            )
            
            self.logger.info(f"✅ Loaded {len(merged_data)} data points with regime information")
            return merged_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading regime data: {e}")
            return None

    async def _process_regimes_parallel(
        self, 
        data: pd.DataFrame, 
        regime_ids: List[int], 
        base_dir: Path, 
        num_regimes: int
    ) -> bool:
        """Process regimes in parallel with memory management."""
        
        # Determine batch size based on number of regimes
        if num_regimes <= 5:
            batch_size = 3
        elif num_regimes <= 10:
            batch_size = 2
        else:
            batch_size = 1  # Process one at a time for 10+ regimes
        
        self.logger.info(f"🔄 Processing {num_regimes} regimes in batches of {batch_size}")
        
        # Split regime IDs into batches
        regime_batches = [regime_ids[i:i + batch_size] for i in range(0, len(regime_ids), batch_size)]
        
        all_success = True
        
        for batch_idx, regime_batch in enumerate(regime_batches):
            self.logger.info(f"📦 Processing batch {batch_idx + 1}/{len(regime_batches)}: regimes {regime_batch}")
            
            # Process batch in parallel
            tasks = [
                self._process_single_regime(data, regime_id, base_dir) 
                for regime_id in regime_batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check batch results
            for regime_id, result in zip(regime_batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Error processing regime {regime_id}: {result}")
                    all_success = False
                elif not result:
                    self.logger.error(f"❌ Failed to process regime {regime_id}")
                    all_success = False
                else:
                    self.logger.info(f"✅ Successfully processed regime {regime_id}")
            
            # Clear memory after each batch
            del tasks
            del batch_results
            
        return all_success

    async def _process_single_regime(
        self, 
        data: pd.DataFrame, 
        regime_id: int, 
        base_dir: Path
    ) -> bool:
        """Process a single regime."""
        try:
            # Filter data for this regime
            regime_data = data[data['composite_cluster_id'] == regime_id].copy()
            
            if len(regime_data) < 50:
                self.logger.warning(f"⚠️ Regime {regime_id} has only {len(regime_data)} data points")
            
            # Create regime directory
            regime_dir = base_dir / f"regime_{regime_id}"
            regime_dir.mkdir(exist_ok=True)
            
            # Save regime data
            regime_file = regime_dir / "regime_data.parquet"
            regime_data.to_parquet(regime_file, index=False)
            
            # Save regime statistics
            stats = {
                "regime_id": regime_id,
                "data_points": len(regime_data),
                "date_range": {
                    "start": regime_data['timestamp'].min().isoformat(),
                    "end": regime_data['timestamp'].max().isoformat()
                },
                "price_stats": {
                    "mean": float(regime_data['close'].mean()),
                    "std": float(regime_data['close'].std()),
                    "min": float(regime_data['close'].min()),
                    "max": float(regime_data['close'].max())
                }
            }
            
            stats_file = regime_dir / "regime_stats.json"
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"✅ Regime {regime_id}: {len(regime_data)} data points saved")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error processing regime {regime_id}: {e}")
            return False

    async def _save_regime_metadata(self, regime_ids: List[int], base_dir: Path) -> None:
        """Save metadata about all regimes."""
        try:
            metadata = {
                "total_regimes": len(regime_ids),
                "regime_ids": sorted(regime_ids),
                "created_at": time.time(),
                "regime_structure": {
                    "base_dir": str(base_dir),
                    "regime_pattern": "regime_{regime_id}/regime_data.parquet"
                }
            }
            
            metadata_file = base_dir / "regime_metadata.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Regime metadata saved: {metadata_file}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving regime metadata: {e}")


@with_tracing_span("execute_regime_data_splitting")
@quality_gate(
    min_quality_score=0.8,
    max_correlation=0.95,
    required_grade="B"
)
@comprehensive_data_validation
@handle_errors
@memory_efficient
@resource_monitor
@secure_data_processing
@validate_data_structure
@monitor_feature_engineering()
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    config: dict[str, Any] = None,
) -> bool:
    """Run Step 4: Regime Data Splitting.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force_rerun: Force rerun flag
        config: Configuration dictionary
        
    Returns:
        bool: Success status
    """
    logger.info("🚀 Starting Step 4: Regime Data Splitting")
    
    try:
        # Initialize step
        step = RegimeDataSplittingStep(config or {})
        await step.initialize()
        
        # Execute regime data splitting
        success = await step.split_data_by_regimes(symbol, exchange, timeframe, data_dir)
        
        if success:
            logger.info("✅ Step 4: Regime Data Splitting completed successfully")
        else:
            logger.error("❌ Step 4: Regime Data Splitting failed")
        
        return success
        
    except Exception as e:
        logger.exception(f"❌ Error in Step 4: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        test_config = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m"
        }
        
        success = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            force_rerun=False,
            config=test_config
        )
        
        print(f"Test result: {success}")

    asyncio.run(test())