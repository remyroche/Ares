#!/usr/bin/env python3
"""Step 4: Triple Barrier Method.

This module applies the triple barrier method to create trading signals and labels.
It uses the optimized triple barrier labeling component and integrates with the pipeline.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

# Handle optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
from src.utils.logger import system_logger

logger = system_logger.getChild("Step4TripleBarrierMethod")


class TripleBarrierMethodStep:
    """Step 4: Triple Barrier Method with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("TripleBarrierMethodStep")
        self.start_time = None
        self.step_timings = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize triple barrier method components."""
        self.logger.info("🔧 Initializing triple barrier method components...")
        try:
            from .step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
                OptimizedTripleBarrierLabeling
            )
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling()
            self.logger.info("✅ Optimized triple barrier labeler initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import OptimizedTripleBarrierLabeling: {e}")
            self.logger.info("📝 Proceeding without optimized triple barrier labeler")
            self.triple_barrier_labeler = None

    async def initialize(self) -> None:
        """Initialize the triple barrier method step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing Triple Barrier Method Step...")
        self.logger.info("📋 Step 4 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info("✅ Triple Barrier Method Step initialized successfully")

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("execute_triple_barrier_method")
    @quality_gate(
        min_quality_score=0.7,
        max_correlation=0.95,
        required_grade="C"
    )
    @comprehensive_data_validation
    @handle_errors
    @memory_efficient
    @resource_monitor
    @secure_data_processing
    @validate_data_structure
    async def execute_triple_barrier_method(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "data_cache",
        force_rerun: bool = False,
    ) -> bool:
        """Execute the triple barrier method step.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe for data
            data_dir: Data directory
            force_rerun: Force rerun the step

        Returns:
            True if successful, False otherwise
        """
        step_start = time.time()
        self.logger.info(f"🚀 Executing Triple Barrier Method for {symbol} on {exchange}")

        try:
            # Load data from previous steps
            unified_data_path = Path(data_dir) / "unified" / exchange / symbol / timeframe
            if not unified_data_path.exists():
                self.logger.error(f"❌ Unified data not found at {unified_data_path}")
                return False

            # Load the unified data
            data_files = list(unified_data_path.glob("*.parquet"))
            if not data_files:
                self.logger.error(f"❌ No parquet files found in {unified_data_path}")
                return False

            # Load the most recent data file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"📁 Loading data from {latest_file}")

            data = pd.read_parquet(latest_file)
            self.logger.info(f"✅ Loaded data with shape: {data.shape}")

            # Apply triple barrier method
            if self.triple_barrier_labeler:
                # Use optimized triple barrier labeling
                labeled_data = await self._apply_optimized_triple_barrier(data)
            else:
                # Fallback to basic implementation
                labeled_data = await self._apply_basic_triple_barrier(data)

            if labeled_data is None:
                self.logger.error("❌ Failed to generate triple barrier labels")
                return False

            # Save results
            output_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Combine data with labels
            result_data = data.copy()
            result_data['label'] = labeled_data['label']
            result_data['potential_profit_pct'] = labeled_data['potential_profit_pct']
            
            # Create enhanced labels that include profit information
            result_data = self._create_enhanced_labels(result_data)
            
            result_data.to_parquet(output_path)
            self.logger.info(f"✅ Triple barrier labels saved to {output_path}")

            self._log_step_timing("Triple Barrier Method", step_start)
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error in triple barrier method: {e}")
            return False

    async def _apply_optimized_triple_barrier(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply optimized triple barrier labeling with profit tracking."""
        try:
            # Configure triple barrier parameters
            profit_take_multiplier = self.config.get("triple_barrier", {}).get("profit_take_multiplier", 0.002)
            stop_loss_multiplier = self.config.get("triple_barrier", {}).get("stop_loss_multiplier", 0.001)
            time_barrier_minutes = self.config.get("triple_barrier", {}).get("time_barrier_minutes", 30)
            max_lookahead = self.config.get("triple_barrier", {}).get("max_lookahead", 100)

            # Create triple barrier labeler with configuration
            from .step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
                OptimizedTripleBarrierLabeling
            )
            
            labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=profit_take_multiplier,
                stop_loss_multiplier=stop_loss_multiplier,
                time_barrier_minutes=time_barrier_minutes,
                max_lookahead=max_lookahead,
                binary_classification=True
            )

            # Apply triple barrier labeling with profit tracking
            labeled_data = labeler.apply_triple_barrier_labeling_vectorized(data)
            
            # Extract labels and profit percentages
            labels = labeled_data['label']
            profit_pcts = labeled_data['potential_profit_pct']
            
            self.logger.info(f"✅ Generated {len(labels)} triple barrier labels with profit tracking")
            self.logger.info(f"   - Buy signals: {(labels == 1).sum()}")
            self.logger.info(f"   - Sell signals: {(labels == -1).sum()}")
            self.logger.info(f"   - Hold signals: {(labels == 0).sum()}")
            
            # Log profit statistics
            if len(labeled_data) > 0:
                buy_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
                sell_profits = labeled_data[labeled_data['label'] == -1]['potential_profit_pct']
                
                self.logger.info("💰 Profit tracking statistics:")
                self.logger.info(f"   - BUY signals avg profit: {buy_profits.mean():.4f} ({buy_profits.std():.4f} std)")
                self.logger.info(f"   - SELL signals avg profit: {sell_profits.mean():.4f} ({sell_profits.std():.4f} std)")
                self.logger.info(f"   - Overall avg profit: {labeled_data['potential_profit_pct'].mean():.4f}")
            
            return labeled_data

        except Exception as e:
            self.logger.exception(f"❌ Error in optimized triple barrier: {e}")
            return None

    async def _apply_basic_triple_barrier(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply basic triple barrier labeling as fallback with profit tracking."""
        try:
            self.logger.warning("⚠️ Using basic triple barrier implementation with profit tracking")
            
            # Simple triple barrier implementation with profit tracking
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            profit_take_multiplier = self.config.get("triple_barrier", {}).get("profit_take_multiplier", 0.002)
            stop_loss_multiplier = self.config.get("triple_barrier", {}).get("stop_loss_multiplier", 0.001)
            max_lookahead = self.config.get("triple_barrier", {}).get("max_lookahead", 100)
            
            labels = np.zeros(len(close_prices), dtype=np.int8)
            profit_pcts = np.zeros(len(close_prices), dtype=np.float64)
            
            for i in range(len(close_prices) - 1):
                entry_price = close_prices[i]
                profit_barrier = entry_price * (1 + profit_take_multiplier)
                stop_barrier = entry_price * (1 - stop_loss_multiplier)
                
                # Look ahead for barrier hits
                for j in range(i + 1, min(i + max_lookahead, len(close_prices))):
                    if high_prices[j] >= profit_barrier:
                        labels[i] = 1  # Buy signal
                        profit_pcts[i] = profit_take_multiplier  # Profit take hit
                        break
                    elif low_prices[j] <= stop_barrier:
                        labels[i] = -1  # Sell signal
                        profit_pcts[i] = -stop_loss_multiplier  # Stop loss hit
                        break
                    # If no barrier hit, label remains 0 (hold) and profit_pct remains 0.0
            
            # Create result DataFrame with both labels and profit percentages
            result_data = data.copy()
            result_data['label'] = labels
            result_data['potential_profit_pct'] = profit_pcts
            
            # Filter out HOLD samples for binary classification
            original_count = len(result_data)
            result_data = result_data[result_data['label'] != 0].copy()
            filtered_count = len(result_data)
            
            # Create enhanced labels that include profit information
            result_data = self._create_enhanced_labels(result_data)
            
            self.logger.info(f"✅ Generated {len(labels)} basic triple barrier labels with profit tracking")
            self.logger.info(f"   - Buy signals: {(labels == 1).sum()}")
            self.logger.info(f"   - Sell signals: {(labels == -1).sum()}")
            self.logger.info(f"   - Hold signals: {(labels == 0).sum()}")
            self.logger.info(f"   - Filtered samples: {filtered_count} (from {original_count})")
            
            # Log profit statistics
            if len(result_data) > 0:
                buy_profits = result_data[result_data['label'] == 1]['potential_profit_pct']
                sell_profits = result_data[result_data['label'] == -1]['potential_profit_pct']
                
                self.logger.info("💰 Basic profit tracking statistics:")
                self.logger.info(f"   - BUY signals avg profit: {buy_profits.mean():.4f}")
                self.logger.info(f"   - SELL signals avg profit: {sell_profits.mean():.4f}")
                self.logger.info(f"   - Overall avg profit: {result_data['potential_profit_pct'].mean():.4f}")
            
            return result_data

        except Exception as e:
            self.logger.exception(f"❌ Error in basic triple barrier: {e}")
            return None

    def _create_enhanced_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced labels that include profit information alongside direction labels.
        
        This method creates additional columns that combine direction and profit information
        for more comprehensive trading signal analysis.
        
        Args:
            data: DataFrame with 'label' and 'potential_profit_pct' columns
            
        Returns:
            DataFrame with enhanced label columns
        """
        try:
            enhanced_data = data.copy()
            
            # Create profit-binned labels (categorize profits into bins)
            profit_bins = [-np.inf, -0.005, -0.002, 0, 0.002, 0.005, np.inf]
            profit_labels = ['Large Loss', 'Medium Loss', 'Small Loss', 'No Profit', 'Small Profit', 'Large Profit']
            enhanced_data['profit_category'] = pd.cut(
                enhanced_data['potential_profit_pct'], 
                bins=profit_bins, 
                labels=profit_labels, 
                include_lowest=True
            )
            
            # Create combined direction-profit labels
            enhanced_data['direction_profit_label'] = enhanced_data.apply(
                lambda row: f"{'BUY' if row['label'] == 1 else 'SELL'}_{row['profit_category']}", 
                axis=1
            )
            
            # Create profit-weighted labels (for regression tasks)
            enhanced_data['profit_weighted_label'] = enhanced_data['label'] * enhanced_data['potential_profit_pct']
            
            # Create risk-adjusted labels (profit divided by time to barrier hit)
            # For now, we'll use a simple approach - can be enhanced later
            enhanced_data['risk_adjusted_profit'] = enhanced_data['potential_profit_pct'].abs()
            
            # Create confidence scores based on profit magnitude
            max_profit = enhanced_data['potential_profit_pct'].abs().max()
            if max_profit > 0:
                enhanced_data['signal_confidence'] = enhanced_data['potential_profit_pct'].abs() / max_profit
            else:
                enhanced_data['signal_confidence'] = 0.0
            
            # Log enhanced labeling statistics
            self.logger.info("🎯 Enhanced labeling statistics:")
            self.logger.info(f"   - Profit categories: {enhanced_data['profit_category'].value_counts().to_dict()}")
            self.logger.info(f"   - Direction-profit combinations: {enhanced_data['direction_profit_label'].value_counts().to_dict()}")
            self.logger.info(f"   - Average signal confidence: {enhanced_data['signal_confidence'].mean():.4f}")
            self.logger.info(f"   - Risk-adjusted profit range: {enhanced_data['risk_adjusted_profit'].min():.4f} to {enhanced_data['risk_adjusted_profit'].max():.4f}")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create enhanced labels: {e}")
            return data


async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run the triple barrier method step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = {}

    # Add step-specific configuration
    step_config = {
        "SYMBOL": symbol,
        "EXCHANGE": exchange,
        "TIMEFRAME": timeframe,
        "DATA_DIR": data_dir,
        "triple_barrier": {
            "profit_take_multiplier": 0.002,
            "stop_loss_multiplier": 0.001,
            "time_barrier_minutes": 30,
            "max_lookahead": 100,
        },
        **config
    }

    step = TripleBarrierMethodStep(step_config)
    await step.initialize()
    
    return await step.execute_triple_barrier_method(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun,
    )


if __name__ == "__main__":
    # Test the step
    async def test():
        success = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache"
        )
        print(f"Step 4 result: {success}")

    asyncio.run(test())