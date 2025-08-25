#!/usr/bin/env python3
"""Step 5: Labeling.

This module creates comprehensive labels for the training data, combining triple barrier
labels with additional labeling strategies and meta-labeling features.
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
)
from src.utils.centralized_decorators import quality_gate
from src.utils.logger import system_logger
from src.utils.centralized_decorators import monitor_feature_engineering

logger = system_logger.getChild("Step5Labeling")


class LabelingStep:
    """Step 5: Labeling with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("LabelingStep")
        self.start_time = None
        self.step_timings = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize labeling components."""
        self.logger.info("🔧 Initializing labeling components...")
        try:
            # Import meta-labeling system if available
            from src.analyst.meta_labeling_system import MetaLabelingSystem
            self.meta_labeling_system = MetaLabelingSystem(self.config)
            self.logger.info("✅ Meta-labeling system initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import MetaLabelingSystem: {e}")
            self.logger.info("📝 Proceeding without meta-labeling system")
            self.meta_labeling_system = None

    async def initialize(self) -> None:
        """Initialize the labeling step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing Labeling Step...")
        self.logger.info("📋 Step 5 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info("✅ Labeling Step initialized successfully")

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("execute_labeling")
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
    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "data_cache",
        force_rerun: bool = False,
    ) -> bool:
        """Execute the labeling step.

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
        self.logger.info(f"🚀 Executing Labeling for {symbol} on {exchange}")

        try:
            # Load triple barrier labels from previous step
            triple_barrier_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet"
            if not triple_barrier_path.exists():
                self.logger.error(f"❌ Triple barrier labels not found at {triple_barrier_path}")
                return False

            self.logger.info(f"📁 Loading triple barrier labels from {triple_barrier_path}")
            data = pd.read_parquet(triple_barrier_path)
            self.logger.info(f"✅ Loaded data with shape: {data.shape}")

            # Generate comprehensive labels
            labeled_data = await self._generate_comprehensive_labels(data, symbol, exchange, timeframe)

            if labeled_data is None:
                self.logger.error("❌ Failed to generate comprehensive labels")
                return False

            # Save results
            output_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_labeled_data.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            labeled_data.to_parquet(output_path)
            self.logger.info(f"✅ Labeled data saved to {output_path}")

            # Save labeling metadata
            metadata_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_labeling_metadata.json"
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "total_samples": len(labeled_data),
                "label_distribution": labeled_data['label'].value_counts().to_dict(),
                "triple_barrier_distribution": labeled_data['triple_barrier_label'].value_counts().to_dict(),
                "created_at": pd.Timestamp.now().isoformat(),
                "labeling_config": self.config.get("labeling", {})
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Labeling metadata saved to {metadata_path}")

            self._log_step_timing("Labeling", step_start)
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error in labeling: {e}")
            return False

    async def _generate_comprehensive_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Generate comprehensive labels combining multiple labeling strategies."""
        try:
            result_data = data.copy()
            
            # 1. Triple barrier labels (already present)
            if 'triple_barrier_label' not in result_data.columns:
                self.logger.error("❌ Triple barrier labels not found in data")
                return None
            
            # 2. Generate meta-labels if meta-labeling system is available
            if self.meta_labeling_system:
                try:
                    await self.meta_labeling_system.initialize()
                    
                    # Generate analyst labels
                    analyst_labels = await self.meta_labeling_system._generate_analyst_labels(
                        data, symbol, exchange, timeframe
                    )
                    if analyst_labels is not None:
                        result_data['analyst_label'] = analyst_labels
                        self.logger.info("✅ Generated analyst labels")
                    
                    # Generate tactician labels
                    tactician_labels = await self.meta_labeling_system._generate_tactician_labels(
                        data, symbol, exchange, timeframe
                    )
                    if tactician_labels is not None:
                        result_data['tactician_label'] = tactician_labels
                        self.logger.info("✅ Generated tactician labels")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Meta-labeling failed: {e}")
            
            # 3. Generate trend-based labels
            trend_labels = await self._generate_trend_labels(data)
            if trend_labels is not None:
                result_data['trend_label'] = trend_labels
                self.logger.info("✅ Generated trend labels")
            
            # 4. Generate volatility-based labels
            volatility_labels = await self._generate_volatility_labels(data)
            if volatility_labels is not None:
                result_data['volatility_label'] = volatility_labels
                self.logger.info("✅ Generated volatility labels")
            
            # 5. Create composite label (primary label for training)
            composite_label = await self._create_composite_label(result_data)
            result_data['label'] = composite_label
            
            # 6. Add label metadata
            result_data['label_confidence'] = await self._calculate_label_confidence(result_data)
            result_data['label_source'] = await self._determine_label_source(result_data)
            
            self.logger.info(f"✅ Generated comprehensive labels with {len(result_data.columns)} columns")
            self.logger.info(f"   - Label distribution: {result_data['label'].value_counts().to_dict()}")
            
            return result_data

        except Exception as e:
            self.logger.exception(f"❌ Error generating comprehensive labels: {e}")
            return None

    async def _generate_trend_labels(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate trend-based labels."""
        try:
            # Simple trend detection using moving averages
            window_short = 10
            window_long = 30
            
            if len(data) < window_long:
                self.logger.warning(f"⚠️ Insufficient data for trend labels: {len(data)} < {window_long}")
                return pd.Series(0, index=data.index)
            
            # Calculate moving averages
            ma_short = data['close'].rolling(window=window_short).mean()
            ma_long = data['close'].rolling(window=window_long).mean()
            
            # Generate trend labels
            trend_labels = np.zeros(len(data), dtype=np.int8)
            
            # Bullish trend: short MA > long MA
            bullish_mask = (ma_short > ma_long) & (ma_short.notna()) & (ma_long.notna())
            trend_labels[bullish_mask] = 1
            
            # Bearish trend: short MA < long MA
            bearish_mask = (ma_short < ma_long) & (ma_short.notna()) & (ma_long.notna())
            trend_labels[bearish_mask] = -1
            
            return pd.Series(trend_labels, index=data.index)

        except Exception as e:
            self.logger.warning(f"⚠️ Error generating trend labels: {e}")
            return None

    async def _generate_volatility_labels(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate volatility-based labels."""
        try:
            # Calculate rolling volatility
            window = 20
            if len(data) < window:
                self.logger.warning(f"⚠️ Insufficient data for volatility labels: {len(data)} < {window}")
                return pd.Series(0, index=data.index)
            
            # Calculate returns
            returns = data['close'].pct_change()
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=window).std()
            
            # Generate volatility labels based on percentile
            volatility_labels = np.zeros(len(data), dtype=np.int8)
            
            # High volatility: top 25%
            high_vol_threshold = volatility.quantile(0.75)
            high_vol_mask = (volatility > high_vol_threshold) & (volatility.notna())
            volatility_labels[high_vol_mask] = 1
            
            # Low volatility: bottom 25%
            low_vol_threshold = volatility.quantile(0.25)
            low_vol_mask = (volatility < low_vol_threshold) & (volatility.notna())
            volatility_labels[low_vol_mask] = -1
            
            return pd.Series(volatility_labels, index=data.index)

        except Exception as e:
            self.logger.warning(f"⚠️ Error generating volatility labels: {e}")
            return None

    async def _create_composite_label(self, data: pd.DataFrame) -> pd.Series:
        """Create composite label from multiple labeling strategies."""
        try:
            # Start with triple barrier labels as base
            composite_label = data['triple_barrier_label'].copy()
            
            # If we have analyst labels, use them to enhance the composite
            if 'analyst_label' in data.columns:
                # Combine triple barrier with analyst labels
                # Analyst labels can override triple barrier in certain conditions
                analyst_override_mask = (
                    (data['analyst_label'] != 0) & 
                    (data['triple_barrier_label'] == 0)
                )
                composite_label[analyst_override_mask] = data['analyst_label'][analyst_override_mask]
            
            # If we have trend labels, use them for additional context
            if 'trend_label' in data.columns:
                # Use trend labels to enhance hold signals
                trend_enhancement_mask = (
                    (composite_label == 0) & 
                    (data['trend_label'] != 0)
                )
                composite_label[trend_enhancement_mask] = data['trend_label'][trend_enhancement_mask]
            
            return composite_label

        except Exception as e:
            self.logger.warning(f"⚠️ Error creating composite label: {e}")
            # Fallback to triple barrier labels
            return data['triple_barrier_label']

    async def _calculate_label_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for labels."""
        try:
            confidence = np.ones(len(data), dtype=np.float32)
            
            # Higher confidence when multiple labeling strategies agree
            if 'analyst_label' in data.columns:
                agreement_mask = (data['label'] == data['analyst_label']) & (data['analyst_label'] != 0)
                confidence[agreement_mask] += 0.2
            
            if 'trend_label' in data.columns:
                agreement_mask = (data['label'] == data['trend_label']) & (data['trend_label'] != 0)
                confidence[agreement_mask] += 0.1
            
            if 'volatility_label' in data.columns:
                agreement_mask = (data['label'] == data['volatility_label']) & (data['volatility_label'] != 0)
                confidence[agreement_mask] += 0.1
            
            # Cap confidence at 1.0
            confidence = np.minimum(confidence, 1.0)
            
            return pd.Series(confidence, index=data.index)

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating label confidence: {e}")
            return pd.Series(1.0, index=data.index)

    async def _determine_label_source(self, data: pd.DataFrame) -> pd.Series:
        """Determine the source of each label."""
        try:
            sources = []
            
            for idx in range(len(data)):
                if data['label'].iloc[idx] == data['triple_barrier_label'].iloc[idx]:
                    if 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                        sources.append("triple_barrier+analyst")
                    else:
                        sources.append("triple_barrier")
                elif 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                    sources.append("analyst")
                elif 'trend_label' in data.columns and data['label'].iloc[idx] == data['trend_label'].iloc[idx]:
                    sources.append("trend")
                else:
                    sources.append("composite")
            
            return pd.Series(sources, index=data.index)

        except Exception as e:
            self.logger.warning(f"⚠️ Error determining label source: {e}")
            return pd.Series("unknown", index=data.index)


async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run the labeling step.

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
        "labeling": {
            "enable_meta_labeling": True,
            "enable_trend_labels": True,
            "enable_volatility_labels": True,
            "composite_label_strategy": "weighted_combination",
        },
        **config
    }

    step = LabelingStep(step_config)
    await step.initialize()
    
    return await step.execute_labeling(
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
        print(f"Step 5 result: {success}")

    asyncio.run(test())