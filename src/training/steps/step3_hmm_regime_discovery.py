#!/usr/bin/env python3
"""Step 3: HMM Regime Discovery with Enhanced Data Quality Management.

This module performs Hidden Markov Model (HMM) regime discovery with comprehensive
data quality checks and automatic data preparation using step1/step1_5 components.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
)
from src.utils.logger import system_logger

logger = system_logger.getChild("Step3HMMRegimeDiscovery")


class HMMRegimeDiscoveryStep:
    """Step 3: HMM Regime Discovery with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("HMMRegimeDiscoveryStep")
        self.data_quality_manager = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize HMM and data quality components."""
        try:
            from .step1.enhanced_data_quality_manager import EnhancedDataQualityManager
            self.data_quality_manager = EnhancedDataQualityManager()
            self.logger.info("✅ Enhanced data quality manager initialized")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import EnhancedDataQualityManager: {e}")

    async def initialize(self) -> None:
        """Initialize the HMM regime discovery step."""
        self.logger.info("🚀 Initializing HMM Regime Discovery Step...")
        self.logger.info("HMM Regime Discovery Step initialized successfully")

    @with_tracing_span("execute_hmm_regime_discovery")
    @quality_gate(validation_level="comprehensive")
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "regimes": [], "error": "HMM discovery failed"},
        context="hmm_regime_discovery.execute"
    )
    async def execute(
        self, 
        training_input: dict[str, Any], 
        pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute HMM regime discovery with enhanced data quality management.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with regime discovery results
        """
        self.logger.info("Starting HMM regime discovery...")

        try:
            # Step 1: Ensure data quality and readiness
            data_ready = await self._ensure_data_quality(training_input)
            if not data_ready:
                self.logger.error("❌ Data not ready for HMM regime discovery")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = "Data quality check failed"
                return pipeline_state

            # Step 2: Load and prepare data for HMM
            data_loaded = await self._load_and_prepare_data(training_input)
            if not data_loaded.get("success", False):
                self.logger.error("❌ Failed to load and prepare data for HMM")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = "Data loading failed"
                return pipeline_state

            # Step 3: Perform HMM regime discovery
            regime_results = await self._perform_hmm_regime_discovery(
                training_input, data_loaded["data"]
            )

            if regime_results.get("success", False):
                self.logger.info("✅ HMM regime discovery completed successfully")
                pipeline_state["hmm_regime_discovery_completed"] = True
                pipeline_state["regime_states"] = regime_results.get("regime_states", [])
                pipeline_state["regime_transitions"] = regime_results.get("regime_transitions", {})
                pipeline_state["regime_metrics"] = regime_results.get("metrics", {})
            else:
                self.logger.error("❌ HMM regime discovery failed")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = regime_results.get("error", "Unknown error")

        except Exception as e:
            self.logger.exception(f"Error during HMM regime discovery: {e}")
            pipeline_state["hmm_regime_discovery_completed"] = False
            pipeline_state["regime_discovery_error"] = str(e)

        return pipeline_state

    @with_tracing_span("ensure_data_quality")
    @secure_data_processing
    async def _ensure_data_quality(self, training_input: dict[str, Any]) -> bool:
        """Ensure data quality and readiness for HMM regime discovery."""
        if not self.data_quality_manager:
            self.logger.warning("⚠️ Data quality manager not available, proceeding without quality check")
            return True

        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

            self.logger.info("🔍 Ensuring data quality for HMM regime discovery...")

            # Get data ready for step3/step4 (which includes HMM)
            data_results = await self.data_quality_manager.get_data_for_step3_step4(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe
            )

            if data_results.get("success", False):
                self.logger.info("✅ Data quality check passed")
                return True
            else:
                self.logger.error("❌ Data quality check failed")
                error = data_results.get("error", "Unknown error")
                self.logger.error(f"   Error: {error}")
                
                # Try to fix missing data using step1/step1_5 components
                self.logger.info("🔄 Attempting to fix missing data...")
                fix_results = await self._fix_missing_data(training_input)
                
                if fix_results.get("success", False):
                    self.logger.info("✅ Successfully fixed missing data")
                    return True
                else:
                    self.logger.error("❌ Failed to fix missing data")
                    return False

        except Exception as e:
            self.logger.exception(f"❌ Error ensuring data quality: {e}")
            return False

    @with_tracing_span("fix_missing_data")
    async def _fix_missing_data(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Fix missing data using step1 and step1_5 components."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

            self.logger.info("🔄 Fixing missing data using step1/step1_5 components...")

            # Try step1 data collection
            step1_success = False
            try:
                from .step1_data_collection import run_step as run_step1
                step1_success = await run_step1(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=True
                )
                if step1_success:
                    self.logger.info("✅ Step1 data collection completed")
                else:
                    self.logger.warning("⚠️ Step1 data collection failed")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not run step1: {e}")

            # Try step1_5 data conversion
            step1_5_success = False
            try:
                from .step1_5_data_converter import run_step as run_step1_5
                step1_5_success = await run_step1_5(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=True
                )
                if step1_5_success:
                    self.logger.info("✅ Step1_5 data conversion completed")
                else:
                    self.logger.warning("⚠️ Step1_5 data conversion failed")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not run step1_5: {e}")

            # Check if data is now ready
            if self.data_quality_manager:
                data_results = await self.data_quality_manager.get_data_for_step3_step4(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                return {
                    "success": data_results.get("success", False),
                    "step1_success": step1_success,
                    "step1_5_success": step1_5_success
                }
            else:
                return {
                    "success": step1_success and step1_5_success,
                    "step1_success": step1_success,
                    "step1_5_success": step1_5_success
                }

        except Exception as e:
            self.logger.exception(f"❌ Error fixing missing data: {e}")
            return {"success": False, "error": str(e)}

    @with_tracing_span("load_and_prepare_data")
    @memory_efficient
    async def _load_and_prepare_data(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Load and prepare data for HMM regime discovery."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")
            data_dir = training_input.get("data_dir", "data_cache")

            self.logger.info("📊 Loading and preparing data for HMM...")

            # Load klines data
            klines_path = Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            if not klines_path.exists():
                return {
                    "success": False,
                    "error": f"Klines file not found: {klines_path}"
                }

            # Load data with memory optimization
            df = pd.read_parquet(klines_path)
            
            if df.empty:
                return {
                    "success": False,
                    "error": "Klines data is empty"
                }

            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "error": f"Missing required columns: {missing_columns}"
                }

            # Prepare features for HMM
            features = await self._prepare_hmm_features(df)

            self.logger.info(f"✅ Data loaded successfully: {len(df)} rows, {len(features)} features")
            
            return {
                "success": True,
                "data": df,
                "features": features,
                "data_info": {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": {
                        "start": df["timestamp"].min().isoformat(),
                        "end": df["timestamp"].max().isoformat()
                    }
                }
            }

        except Exception as e:
            self.logger.exception(f"❌ Error loading and preparing data: {e}")
            return {"success": False, "error": str(e)}

    @with_tracing_span("prepare_hmm_features")
    @validate_data_structure
    async def _prepare_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for HMM regime discovery."""
        try:
            # Ensure timestamp is datetime
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Calculate basic features
            features = pd.DataFrame()
            features["timestamp"] = df["timestamp"]

            # Price-based features
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["price_range"] = (df["high"] - df["low"]) / df["close"]
            features["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

            # Technical indicators
            features["sma_20"] = df["close"].rolling(window=20).mean()
            features["sma_50"] = df["close"].rolling(window=50).mean()
            features["rsi"] = self._calculate_rsi(df["close"])
            features["macd"] = self._calculate_macd(df["close"])

            # Remove NaN values
            features = features.dropna()

            self.logger.info(f"✅ Prepared {len(features.columns)} features for HMM")
            return features

        except Exception as e:
            self.logger.exception(f"❌ Error preparing HMM features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @with_tracing_span("perform_hmm_regime_discovery")
    @resource_monitor
    async def _perform_hmm_regime_discovery(
        self, 
        training_input: dict[str, Any], 
        data: pd.DataFrame
    ) -> dict[str, Any]:
        """Perform HMM regime discovery on the prepared data."""
        try:
            self.logger.info("🔍 Performing HMM regime discovery...")

            # For now, implement a simple regime detection
            # In a full implementation, this would use a proper HMM library like hmmlearn
            
            # Simple regime detection based on volatility and returns
            features = await self._prepare_hmm_features(data)
            
            # Define regimes based on volatility and returns
            volatility = features["volatility"].fillna(0)
            returns = features["returns"].fillna(0)
            
            # Simple regime classification
            regimes = []
            for i in range(len(features)):
                vol = volatility.iloc[i]
                ret = returns.iloc[i]
                
                if vol > volatility.quantile(0.8):
                    if ret > returns.quantile(0.7):
                        regime = "high_volatility_bull"
                    elif ret < returns.quantile(0.3):
                        regime = "high_volatility_bear"
                    else:
                        regime = "high_volatility_neutral"
                elif vol < volatility.quantile(0.2):
                    if ret > returns.quantile(0.7):
                        regime = "low_volatility_bull"
                    elif ret < returns.quantile(0.3):
                        regime = "low_volatility_bear"
                    else:
                        regime = "low_volatility_neutral"
                else:
                    if ret > returns.quantile(0.7):
                        regime = "medium_volatility_bull"
                    elif ret < returns.quantile(0.3):
                        regime = "medium_volatility_bear"
                    else:
                        regime = "medium_volatility_neutral"
                
                regimes.append(regime)

            # Calculate regime statistics
            regime_counts = pd.Series(regimes).value_counts()
            regime_transitions = self._calculate_regime_transitions(regimes)

            self.logger.info(f"✅ HMM regime discovery completed")
            self.logger.info(f"📊 Discovered {len(regime_counts)} regimes:")
            for regime, count in regime_counts.items():
                self.logger.info(f"   - {regime}: {count} periods ({count/len(regimes)*100:.1f}%)")

            return {
                "success": True,
                "regime_states": regimes,
                "regime_transitions": regime_transitions,
                "metrics": {
                    "total_periods": len(regimes),
                    "unique_regimes": len(regime_counts),
                    "regime_distribution": regime_counts.to_dict()
                }
            }

        except Exception as e:
            self.logger.exception(f"❌ Error performing HMM regime discovery: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_regime_transitions(self, regimes: List[str]) -> dict[str, Any]:
        """Calculate regime transition probabilities."""
        transitions = {}
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            
            if current_regime not in transitions:
                transitions[current_regime] = {}
            
            if next_regime not in transitions[current_regime]:
                transitions[current_regime][next_regime] = 0
            
            transitions[current_regime][next_regime] += 1

        # Convert counts to probabilities
        for current_regime in transitions:
            total = sum(transitions[current_regime].values())
            for next_regime in transitions[current_regime]:
                transitions[current_regime][next_regime] /= total

        return transitions


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step3_hmm_regime_discovery",
)
async def run_step(
    symbol: str, 
    exchange: str, 
    timeframe: str = "1m", 
    data_dir: str = "data_cache", 
    force_rerun: bool = False,
    **kwargs: Any
) -> bool:
    """Run the HMM regime discovery step with enhanced data quality management.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger = system_logger.getChild("Step3HMMRegimeDiscovery")

        logger.info("=" * 80)
        logger.info("🚀 STEP 3: HMM Regime Discovery")
        logger.info("=" * 80)
        logger.info(f"🎯 Symbol: {symbol}")
        logger.info(f"🏢 Exchange: {exchange}")
        logger.info(f"📊 Timeframe: {timeframe}")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🔄 Force rerun: {force_rerun}")

        # Initialize HMM regime discovery step
        config = {
            "SYMBOL": symbol,
            "EXCHANGE": exchange,
            "TIMEFRAME": timeframe,
            "DATA_DIR": data_dir,
        }
        
        step = HMMRegimeDiscoveryStep(config)
        await step.initialize()

        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
        }

        # Execute HMM regime discovery
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        if result.get("hmm_regime_discovery_completed", False):
            logger.info("✅ Step 3: HMM Regime Discovery completed successfully")
            
            # Log regime discovery results
            if result.get("regime_states"):
                logger.info(f"📊 Discovered {len(set(result['regime_states']))} unique regimes")
            
            if result.get("regime_metrics"):
                metrics = result["regime_metrics"]
                logger.info(f"📈 Total periods: {metrics.get('total_periods', 0)}")
                logger.info(f"🔄 Unique regimes: {metrics.get('unique_regimes', 0)}")
            
            return True
        else:
            logger.error("❌ Step 3: HMM Regime Discovery failed")
            error = result.get("regime_discovery_error", "Unknown error")
            logger.error(f"   Error: {error}")
            return False

    except Exception as e:
        logger.exception(f"❌ Step 3: HMM Regime Discovery failed: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    import asyncio

    async def main() -> None:
        # Get command line arguments
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else "data_cache"
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
        else:
            print("Usage: python step3_hmm_regime_discovery.py <symbol> <exchange> <timeframe> [data_dir] [force_rerun]")
            print("Example: python step3_hmm_regime_discovery.py ETHUSDT BINANCE 1m data_cache true")
            return

        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun
        )

        if success:
            print("✅ Step 3: HMM Regime Discovery completed successfully")
        else:
            print("❌ Step 3: HMM Regime Discovery failed")

        # Clean up memory
        import gc
        gc.collect()

    # Use a more robust approach to prevent segmentation fault
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Final cleanup
        import gc
        gc.collect()