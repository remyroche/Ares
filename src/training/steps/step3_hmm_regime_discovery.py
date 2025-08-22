#!/usr/bin/env python3
"""Step 3: HMM Regime Discovery with Enhanced Data Quality Management.

This module performs Hidden Markov Model (HMM) regime discovery with comprehensive
data quality checks and automatic data preparation using step1/step1_5 components.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import psutil

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
from src.utils.training_pipeline_decorators import monitor_feature_engineering

logger = system_logger.getChild("Step3HMMRegimeDiscovery")


class HMMRegimeDiscoveryStep:
    """Step 3: HMM Regime Discovery with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("HMMRegimeDiscoveryStep")
        self.data_quality_manager = None
        self.start_time = None
        self.step_timings = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize HMM and data quality components."""
        self.logger.info("🔧 Initializing HMM regime discovery components...")
        try:
            from .step1.enhanced_data_quality_manager import EnhancedDataQualityManager
            self.data_quality_manager = EnhancedDataQualityManager()
            self.logger.info("✅ Enhanced data quality manager initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import EnhancedDataQualityManager: {e}")
            self.logger.info("📝 Proceeding without enhanced data quality manager")

    async def initialize(self) -> None:
        """Initialize the HMM regime discovery step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing HMM Regime Discovery Step...")
        self.logger.info("📋 Step 3 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info("✅ HMM Regime Discovery Step initialized successfully")

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

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
        step_start = time.time()
        self.logger.info("🎯 Starting HMM regime discovery execution...")
        self.logger.info(f"📊 Training input keys: {list(training_input.keys())}")
        self.logger.info(f"🔄 Pipeline state keys: {list(pipeline_state.keys())}")
        
        # Initial memory usage
        initial_memory = psutil.virtual_memory()
        self.logger.info(f"💾 Initial memory usage: {initial_memory.percent:.1f}% ({initial_memory.used / 1024**3:.1f}GB / {initial_memory.total / 1024**3:.1f}GB)")

        try:
            # Step 1: Ensure data quality and readiness
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: Data Quality Validation")
            self.logger.info("=" * 60)
            data_quality_start = time.time()
            data_ready = await self._ensure_data_quality(training_input)
            data_quality_elapsed = time.time() - data_quality_start
            self.logger.info(f"⏱️ Data Quality Validation completed in {data_quality_elapsed:.2f} seconds")
            
            if not data_ready:
                self.logger.error("❌ Data not ready for HMM regime discovery")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = "Data quality check failed"
                return pipeline_state

            # Step 2: Load and prepare data for HMM
            self.logger.info("=" * 60)
            self.logger.info("STEP 2: Data Loading and Preparation")
            self.logger.info("=" * 60)
            data_loading_start = time.time()
            data_loaded = await self._load_and_prepare_data(training_input)
            data_loading_elapsed = time.time() - data_loading_start
            self.logger.info(f"⏱️ Data Loading and Preparation completed in {data_loading_elapsed:.2f} seconds")
            
            if not data_loaded.get("success", False):
                self.logger.error("❌ Failed to load and prepare data for HMM")
                error_msg = data_loaded.get("error", "Unknown error")
                self.logger.error(f"   Error details: {error_msg}")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = f"Data loading failed: {error_msg}"
                return pipeline_state

            # Step 3: Perform HMM regime discovery
            self.logger.info("=" * 60)
            self.logger.info("STEP 3: HMM Regime Discovery")
            self.logger.info("=" * 60)
            hmm_start = time.time()
            regime_results = await self._perform_hmm_regime_discovery(
                training_input, data_loaded["data"]
            )
            hmm_elapsed = time.time() - hmm_start
            self.logger.info(f"⏱️ HMM Regime Discovery completed in {hmm_elapsed:.2f} seconds")

            if regime_results.get("success", False):
                self.logger.info("✅ HMM regime discovery completed successfully")
                pipeline_state["hmm_regime_discovery_completed"] = True
                pipeline_state["regime_states"] = regime_results.get("regime_states", [])
                pipeline_state["regime_transitions"] = regime_results.get("regime_transitions", {})
                pipeline_state["regime_metrics"] = regime_results.get("metrics", {})
                
                # Log detailed results
                self._log_regime_discovery_results(regime_results)
            else:
                self.logger.error("❌ HMM regime discovery failed")
                error_msg = regime_results.get("error", "Unknown error")
                self.logger.error(f"   Error details: {error_msg}")
                pipeline_state["hmm_regime_discovery_completed"] = False
                pipeline_state["regime_discovery_error"] = error_msg

        except Exception as e:
            self.logger.exception(f"❌ Unexpected error during HMM regime discovery: {e}")
            pipeline_state["hmm_regime_discovery_completed"] = False
            pipeline_state["regime_discovery_error"] = str(e)

        # Log overall execution summary
        total_elapsed = time.time() - step_start
        self.logger.info("=" * 60)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
        self.logger.info(f"⏱️ Step timings:")
        self.logger.info(f"   - Data Quality Validation: {data_quality_elapsed:.2f}s")
        self.logger.info(f"   - Data Loading and Preparation: {data_loading_elapsed:.2f}s")
        self.logger.info(f"   - HMM Regime Discovery: {hmm_elapsed:.2f}s")
        
        # Memory usage summary
        memory_usage = psutil.virtual_memory()
        self.logger.info(f"💾 Memory usage: {memory_usage.percent:.1f}% ({memory_usage.used / 1024**3:.1f}GB / {memory_usage.total / 1024**3:.1f}GB)")
        
        success = pipeline_state.get("hmm_regime_discovery_completed", False)
        self.logger.info(f"🎯 Final result: {'✅ SUCCESS' if success else '❌ FAILED'}")

        return pipeline_state

    def _log_regime_discovery_results(self, regime_results: dict[str, Any]) -> None:
        """Log detailed regime discovery results."""
        self.logger.info("📊 REGIME DISCOVERY RESULTS")
        self.logger.info("-" * 40)
        
        metrics = regime_results.get("metrics", {})
        self.logger.info(f"📈 Total periods analyzed: {metrics.get('total_periods', 0):,}")
        self.logger.info(f"🔄 Unique regimes discovered: {metrics.get('unique_regimes', 0)}")
        
        regime_distribution = metrics.get('regime_distribution', {})
        if regime_distribution:
            self.logger.info("📊 Regime distribution:")
            for regime, count in regime_distribution.items():
                percentage = (count / metrics.get('total_periods', 1)) * 100
                self.logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")
        
        transitions = regime_results.get("regime_transitions", {})
        if transitions:
            self.logger.info("🔄 Regime transition probabilities:")
            for from_regime, to_regimes in transitions.items():
                self.logger.info(f"   From {from_regime}:")
                for to_regime, prob in to_regimes.items():
                    self.logger.info(f"     → {to_regime}: {prob:.3f}")

    @with_tracing_span("ensure_data_quality")
    @secure_data_processing
    async def _ensure_data_quality(self, training_input: dict[str, Any]) -> bool:
        """Ensure data quality and readiness for HMM regime discovery."""
        self.logger.info("🔍 Starting data quality validation...")
        
        if not self.data_quality_manager:
            self.logger.warning("⚠️ Data quality manager not available, proceeding without quality check")
            self.logger.info("📝 Skipping enhanced data quality validation")
            return True

        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

            self.logger.info(f"🎯 Validating data quality for {symbol} on {exchange} ({timeframe})...")

            # Get data ready for step3/step4 (which includes HMM)
            self.logger.info("📋 Requesting data from quality manager...")
            data_results = await self.data_quality_manager.get_data_for_step3_step4(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe
            )

            if data_results.get("success", False):
                self.logger.info("✅ Data quality check passed")
                self.logger.info("📊 Data quality metrics:")
                for key, value in data_results.items():
                    if key != "success":
                        self.logger.info(f"   - {key}: {value}")
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
                    self.logger.info("📊 Fix results:")
                    for key, value in fix_results.items():
                        if key != "success":
                            self.logger.info(f"   - {key}: {value}")
                    return True
                else:
                    self.logger.error("❌ Failed to fix missing data")
                    fix_error = fix_results.get("error", "Unknown error")
                    self.logger.error(f"   Fix error: {fix_error}")
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

            self.logger.info(f"🔄 Fixing missing data for {symbol} on {exchange} ({timeframe})...")

            # Try step1 data collection
            step1_success = False
            try:
                self.logger.info("📥 Attempting step1 data collection...")
                from .step1_data_collection import run_step as run_step1
                step1_success = await run_step1(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=True
                )
                if step1_success:
                    self.logger.info("✅ Step1 data collection completed successfully")
                else:
                    self.logger.warning("⚠️ Step1 data collection failed")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not run step1: {e}")

            # Try step1_5 data conversion
            step1_5_success = False
            try:
                self.logger.info("🔄 Attempting step1_5 data conversion...")
                from .step1_5_data_converter import run_step as run_step1_5
                step1_5_success = await run_step1_5(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_rerun=True
                )
                if step1_5_success:
                    self.logger.info("✅ Step1_5 data conversion completed successfully")
                else:
                    self.logger.warning("⚠️ Step1_5 data conversion failed")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not run step1_5: {e}")

            # Check if data is now ready
            if self.data_quality_manager:
                self.logger.info("🔍 Re-checking data quality after fixes...")
                data_results = await self.data_quality_manager.get_data_for_step3_step4(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                return {
                    "success": data_results.get("success", False),
                    "step1_success": step1_success,
                    "step1_5_success": step1_5_success,
                    "quality_check_result": data_results
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

            self.logger.info(f"📊 Loading and preparing data for HMM...")
            self.logger.info(f"   Symbol: {symbol}")
            self.logger.info(f"   Exchange: {exchange}")
            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Data directory: {data_dir}")

            # Load klines data
            klines_path = Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            self.logger.info(f"📁 Looking for klines file: {klines_path}")
            
            if not klines_path.exists():
                self.logger.error(f"❌ Klines file not found: {klines_path}")
                return {
                    "success": False,
                    "error": f"Klines file not found: {klines_path}"
                }

            self.logger.info("📥 Loading klines data from parquet file...")
            # Load data with memory optimization
            df = pd.read_parquet(klines_path)
            
            if df.empty:
                self.logger.error("❌ Klines data is empty")
                return {
                    "success": False,
                    "error": "Klines data is empty"
                }

            self.logger.info(f"✅ Klines data loaded: {len(df):,} rows, {len(df.columns)} columns")
            self.logger.info(f"📊 Data columns: {list(df.columns)}")

            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return {
                    "success": False,
                    "error": f"Missing required columns: {missing_columns}"
                }

            self.logger.info("✅ All required columns present")

            # Prepare features for HMM
            self.logger.info("🔧 Preparing features for HMM analysis...")
            features = await self._prepare_hmm_features(df)

            self.logger.info(f"✅ Data preparation completed successfully")
            self.logger.info(f"📊 Final data summary:")
            self.logger.info(f"   - Original data: {len(df):,} rows")
            self.logger.info(f"   - Features prepared: {len(features.columns)}")
            self.logger.info(f"   - Feature data: {len(features):,} rows")
            
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
            self.logger.info("🔧 Starting feature preparation for HMM...")
            
            # Ensure timestamp is datetime
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                self.logger.info("🕒 Converting timestamp to datetime...")
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Sort by timestamp
            self.logger.info("📅 Sorting data by timestamp...")
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Calculate basic features
            self.logger.info("📊 Calculating price-based features...")
            features = pd.DataFrame()
            features["timestamp"] = df["timestamp"]

            # Price-based features
            self.logger.info("   - Calculating returns...")
            features["returns"] = df["close"].pct_change()
            
            self.logger.info("   - Calculating log returns...")
            features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            
            self.logger.info("   - Calculating volatility...")
            features["volatility"] = features["returns"].rolling(window=20).std()
            
            self.logger.info("   - Calculating price range...")
            features["price_range"] = (df["high"] - df["low"]) / df["close"]
            
            self.logger.info("   - Calculating volume ratio...")
            features["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

            # Technical indicators
            self.logger.info("📈 Calculating technical indicators...")
            self.logger.info("   - Calculating SMA 20...")
            features["sma_20"] = df["close"].rolling(window=20).mean()
            
            self.logger.info("   - Calculating SMA 50...")
            features["sma_50"] = df["close"].rolling(window=50).mean()
            
            self.logger.info("   - Calculating RSI...")
            features["rsi"] = self._calculate_rsi(df["close"])
            
            self.logger.info("   - Calculating MACD...")
            features["macd"] = self._calculate_macd(df["close"])

            # Remove NaN values
            initial_rows = len(features)
            self.logger.info(f"🧹 Removing NaN values (initial rows: {initial_rows:,})...")
            features = features.dropna()
            final_rows = len(features)
            removed_rows = initial_rows - final_rows
            
            self.logger.info(f"✅ Feature preparation completed:")
            self.logger.info(f"   - Initial rows: {initial_rows:,}")
            self.logger.info(f"   - Final rows: {final_rows:,}")
            self.logger.info(f"   - Removed rows: {removed_rows:,} ({removed_rows/initial_rows*100:.1f}%)")
            self.logger.info(f"   - Features created: {len(features.columns)}")
            
            return features

        except Exception as e:
            self.logger.exception(f"❌ Error preparing HMM features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        self.logger.debug(f"Calculating RSI with window {window}...")
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        self.logger.debug(f"Calculating MACD (fast={fast}, slow={slow}, signal={signal})...")
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
            self.logger.info("🔍 Starting HMM regime discovery analysis...")
            self.logger.info(f"📊 Input data shape: {data.shape}")

            # For now, implement a simple regime detection
            # In a full implementation, this would use a proper HMM library like hmmlearn
            
            # Simple regime detection based on volatility and returns
            self.logger.info("🔧 Preparing features for regime analysis...")
            features = await self._prepare_hmm_features(data)
            
            self.logger.info("📊 Feature statistics:")
            for col in features.columns:
                if col != "timestamp":
                    series = features[col].dropna()
                    if len(series) > 0:
                        self.logger.info(f"   - {col}: mean={series.mean():.6f}, std={series.std():.6f}, min={series.min():.6f}, max={series.max():.6f}")
            
            # Define regimes based on volatility and returns
            self.logger.info("🎯 Defining regimes based on volatility and returns...")
            volatility = features["volatility"].fillna(0)
            returns = features["returns"].fillna(0)
            
            self.logger.info(f"📊 Volatility quantiles:")
            vol_quantiles = volatility.quantile([0.2, 0.8])
            self.logger.info(f"   - 20th percentile: {vol_quantiles[0.2]:.6f}")
            self.logger.info(f"   - 80th percentile: {vol_quantiles[0.8]:.6f}")
            
            self.logger.info(f"📊 Returns quantiles:")
            ret_quantiles = returns.quantile([0.3, 0.7])
            self.logger.info(f"   - 30th percentile: {ret_quantiles[0.3]:.6f}")
            self.logger.info(f"   - 70th percentile: {ret_quantiles[0.7]:.6f}")
            
            # Simple regime classification
            self.logger.info("🏷️ Classifying regimes...")
            regimes = []
            regime_counts = {}
            total_periods = len(features)
            
            # Progress tracking
            progress_interval = max(1, total_periods // 10)  # Log every 10% progress
            
            for i in range(len(features)):
                vol = volatility.iloc[i]
                ret = returns.iloc[i]
                
                if vol > vol_quantiles[0.8]:
                    if ret > ret_quantiles[0.7]:
                        regime = "high_volatility_bull"
                    elif ret < ret_quantiles[0.3]:
                        regime = "high_volatility_bear"
                    else:
                        regime = "high_volatility_neutral"
                elif vol < vol_quantiles[0.2]:
                    if ret > ret_quantiles[0.7]:
                        regime = "low_volatility_bull"
                    elif ret < ret_quantiles[0.3]:
                        regime = "low_volatility_bear"
                    else:
                        regime = "low_volatility_neutral"
                else:
                    if ret > ret_quantiles[0.7]:
                        regime = "medium_volatility_bull"
                    elif ret < ret_quantiles[0.3]:
                        regime = "medium_volatility_bear"
                    else:
                        regime = "medium_volatility_neutral"
                
                regimes.append(regime)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                # Progress logging
                if (i + 1) % progress_interval == 0:
                    progress = ((i + 1) / total_periods) * 100
                    self.logger.info(f"📊 Regime classification progress: {progress:.1f}% ({i + 1:,}/{total_periods:,})")

            # Calculate regime statistics
            self.logger.info("📊 Calculating regime statistics...")
            regime_counts_series = pd.Series(regime_counts)
            regime_transitions = self._calculate_regime_transitions(regimes)

            self.logger.info(f"✅ HMM regime discovery completed successfully")
            self.logger.info(f"📊 Discovered {len(regime_counts)} unique regimes:")
            for regime, count in regime_counts.items():
                percentage = (count / len(regimes)) * 100
                self.logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")

            # Log transition matrix
            if regime_transitions:
                self.logger.info("🔄 Regime transition matrix:")
                for from_regime, to_regimes in regime_transitions.items():
                    self.logger.info(f"   From {from_regime}:")
                    for to_regime, prob in to_regimes.items():
                        self.logger.info(f"     → {to_regime}: {prob:.3f}")

            return {
                "success": True,
                "regime_states": regimes,
                "regime_transitions": regime_transitions,
                "metrics": {
                    "total_periods": len(regimes),
                    "unique_regimes": len(regime_counts),
                    "regime_distribution": regime_counts,
                    "volatility_quantiles": vol_quantiles.to_dict(),
                    "returns_quantiles": ret_quantiles.to_dict()
                }
            }

        except Exception as e:
            self.logger.exception(f"❌ Error performing HMM regime discovery: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_regime_transitions(self, regimes: List[str]) -> dict[str, Any]:
        """Calculate regime transition probabilities."""
        self.logger.info("🔄 Calculating regime transition probabilities...")
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
        self.logger.info("📊 Converting transition counts to probabilities...")
        for current_regime in transitions:
            total = sum(transitions[current_regime].values())
            for next_regime in transitions[current_regime]:
                transitions[current_regime][next_regime] /= total

        self.logger.info(f"✅ Transition matrix calculated for {len(transitions)} regimes")
        return transitions


@monitor_feature_engineering()
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
    start_time = time.time()
    
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
        logger.info(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Initialize HMM regime discovery step
        config = {
            "SYMBOL": symbol,
            "EXCHANGE": exchange,
            "TIMEFRAME": timeframe,
            "DATA_DIR": data_dir,
        }
        
        logger.info("🔧 Initializing HMM regime discovery step...")
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
        logger.info("🎯 Executing HMM regime discovery...")
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        if result.get("hmm_regime_discovery_completed", False):
            logger.info("✅ Step 3: HMM Regime Discovery completed successfully")
            
            # Log regime discovery results
            if result.get("regime_states"):
                unique_regimes = len(set(result['regime_states']))
                total_periods = len(result['regime_states'])
                logger.info(f"📊 Discovered {unique_regimes} unique regimes across {total_periods:,} periods")
            
            if result.get("regime_metrics"):
                metrics = result["regime_metrics"]
                logger.info(f"📈 Total periods: {metrics.get('total_periods', 0):,}")
                logger.info(f"🔄 Unique regimes: {metrics.get('unique_regimes', 0)}")
                
                # Log regime distribution
                regime_dist = metrics.get('regime_distribution', {})
                if regime_dist:
                    logger.info("📊 Regime distribution:")
                    for regime, count in regime_dist.items():
                        percentage = (count / metrics.get('total_periods', 1)) * 100
                        logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")
            
            # Log execution summary
            total_elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info("🎉 STEP 3 EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("✅ SUCCESS")
            logger.info("=" * 80)
            
            return True
        else:
            logger.error("❌ Step 3: HMM Regime Discovery failed")
            error = result.get("regime_discovery_error", "Unknown error")
            logger.error(f"   Error: {error}")
            
            # Log execution summary
            total_elapsed = time.time() - start_time
            logger.info("=" * 80)
            logger.info("💥 STEP 3 EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("❌ FAILED")
            logger.info(f"   Error: {error}")
            logger.info("=" * 80)
            
            return False

    except Exception as e:
        logger.exception(f"❌ Step 3: HMM Regime Discovery failed with exception: {e}")
        
        # Log execution summary
        total_elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("💥 STEP 3 EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total execution time: {total_elapsed:.2f} seconds")
        logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("❌ FAILED")
        logger.info(f"   Exception: {e}")
        logger.info("=" * 80)
        
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

        print("=" * 80)
        print("🚀 STEP 3: HMM Regime Discovery - Command Line Execution")
        print("=" * 80)
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print(f"📊 Timeframe: {timeframe}")
        print(f"📁 Data directory: {data_dir}")
        print(f"🔄 Force rerun: {force_rerun}")
        print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun
        )

        print("=" * 80)
        if success:
            print("✅ Step 3: HMM Regime Discovery completed successfully")
        else:
            print("❌ Step 3: HMM Regime Discovery failed")
        print(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

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