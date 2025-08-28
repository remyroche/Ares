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
    ensure_data_integrity,
    monitor_step_execution,
    secure_step_execution,
    validate_pipeline_step
)
from src.utils.logger import system_logger

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

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="hmm_regime_discovery_initialization"
    )
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

    @validate_pipeline_step(
        step_name="hmm_regime_discovery",
        validation_level="CRITICAL",
        enable_rollback=True,
        max_retries=2
    )
    @ensure_data_integrity(
        check_schema=True,
        check_constraints=True,
        validate_relationships=True
    )
    @monitor_step_execution(
        enable_timing=True,
        enable_memory_monitoring=True,
        enable_progress_tracking=True
    )
    @secure_step_execution(
        error_handling=True,
        rollback_on_failure=True,
        data_validation=True,
        resource_cleanup=True
    )
    @with_tracing_span("execute_hmm_regime_discovery")
    @quality_gate(
        min_quality_score=0.7,
        max_correlation=0.95,
        required_grade="C"
    )
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
        if PSUTIL_AVAILABLE:
            initial_memory = psutil.virtual_memory()
            self.logger.info(f"💾 Initial memory usage: {initial_memory.percent:.1f}% ({initial_memory.used / 1024**3:.1f}GB / {initial_memory.total / 1024**3:.1f}GB)")
        else:
            self.logger.info("💾 Memory monitoring not available (psutil not installed)")

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
        if PSUTIL_AVAILABLE:
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
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="data_quality_validation"
    )
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
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Data fix failed"},
        context="fix_missing_data"
    )
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
    @comprehensive_data_validation
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Data loading failed"},
        context="load_and_prepare_data"
    )
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
    @monitor_feature_engineering()
    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="prepare_hmm_features"
    )
    async def _prepare_hmm_features(self, df: Any) -> Any:
        """Prepare comprehensive features for HMM regime discovery including momentum, S/R, volume, and volatility."""
        try:
            self.logger.info("🔧 Starting comprehensive feature preparation for HMM...")
            
            # Ensure timestamp is datetime
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                self.logger.info("🕒 Converting timestamp to datetime...")
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Sort by timestamp
            self.logger.info("📅 Sorting data by timestamp...")
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Calculate comprehensive features
            self.logger.info("📊 Calculating comprehensive features for HMM...")
            features = pd.DataFrame()
            features["timestamp"] = df["timestamp"]

            # === 1. MOMENTUM FEATURES ===
            self.logger.info("🚀 Calculating momentum features...")
            
            # Price momentum
            self.logger.info("   - Price momentum (5, 10, 20 periods)...")
            features["price_momentum_5"] = df["close"].pct_change(5)
            features["price_momentum_10"] = df["close"].pct_change(10)
            features["price_momentum_20"] = df["close"].pct_change(20)
            
            # Volume momentum
            self.logger.info("   - Volume momentum...")
            features["volume_momentum_5"] = df["volume"].pct_change(5)
            features["volume_momentum_10"] = df["volume"].pct_change(10)
            features["volume_momentum_20"] = df["volume"].pct_change(20)
            
            # RSI momentum
            self.logger.info("   - RSI momentum...")
            features["rsi"] = self._calculate_rsi(df["close"])
            features["rsi_momentum"] = features["rsi"].diff(5)
            
            # MACD momentum
            self.logger.info("   - MACD momentum...")
            features["macd"] = self._calculate_macd(df["close"])
            features["macd_momentum"] = features["macd"].diff(5)

            # === 2. VOLATILITY FEATURES ===
            self.logger.info("📈 Calculating volatility features...")
            
            # Multiple timeframe volatility
            self.logger.info("   - Multi-timeframe volatility...")
            features["volatility_5"] = df["close"].pct_change().rolling(window=5).std()
            features["volatility_10"] = df["close"].pct_change().rolling(window=10).std()
            features["volatility_20"] = df["close"].pct_change().rolling(window=20).std()
            
            # EWMA volatility (smoother)
            self.logger.info("   - EWMA volatility...")
            features["ewma_volatility_20"] = df["close"].pct_change().ewm(span=20).std()
            
            # Volatility acceleration and momentum
            self.logger.info("   - Volatility acceleration and momentum...")
            features["volatility_acceleration"] = features["volatility_20"].diff()
            features["volatility_momentum"] = features["volatility_20"] - features["volatility_20"].shift(5)
            
            # ATR-based volatility
            self.logger.info("   - ATR volatility...")
            features["atr"] = self._calculate_atr(df)
            features["atr_normalized"] = features["atr"] / df["close"]

            # === 3. VOLUME FEATURES ===
            self.logger.info("📊 Calculating volume features...")
            
            # Volume ratios
            self.logger.info("   - Volume ratios...")
            features["volume_ratio_5"] = df["volume"] / df["volume"].rolling(window=5).mean()
            features["volume_ratio_10"] = df["volume"] / df["volume"].rolling(window=10).mean()
            features["volume_ratio_20"] = df["volume"] / df["volume"].rolling(window=20).mean()
            
            # Volume change
            self.logger.info("   - Volume change...")
            features["volume_change"] = df["volume"].pct_change()
            
            # Volume-price relationship
            self.logger.info("   - Volume-price relationship...")
            features["volume_price_trend"] = (df["close"] - df["close"].shift(1)) * df["volume"]
            features["volume_price_trend_ratio"] = features["volume_price_trend"] / features["volume_price_trend"].rolling(20).mean()

            # === 4. SUPPORT/RESISTANCE FEATURES ===
            self.logger.info("🎯 Calculating support/resistance features...")
            
            # Pivot points
            self.logger.info("   - Pivot points...")
            features["pivot_point"] = (df["high"] + df["low"] + df["close"]) / 3
            features["support_1"] = 2 * features["pivot_point"] - df["high"]
            features["resistance_1"] = 2 * features["pivot_point"] - df["low"]
            
            # Distance to support/resistance
            self.logger.info("   - Distance to S/R levels...")
            features["distance_to_support"] = (df["close"] - features["support_1"]) / df["close"]
            features["distance_to_resistance"] = (features["resistance_1"] - df["close"]) / df["close"]
            
            # S/R strength indicators
            self.logger.info("   - S/R strength indicators...")
            features["sr_strength"] = self._calculate_sr_strength(df)
            
            # Bollinger Bands (for S/R context)
            self.logger.info("   - Bollinger Bands...")
            bb_features = self._calculate_bollinger_bands(df["close"])
            features = pd.concat([features, bb_features], axis=1)

            # === 5. ADDITIONAL TECHNICAL FEATURES ===
            self.logger.info("🔧 Calculating additional technical features...")
            
            # Moving averages
            self.logger.info("   - Moving averages...")
            features["sma_20"] = df["close"].rolling(window=20).mean()
            features["sma_50"] = df["close"].rolling(window=50).mean()
            features["ema_12"] = df["close"].ewm(span=12).mean()
            features["ema_26"] = df["close"].ewm(span=26).mean()
            
            # Price position relative to MAs
            self.logger.info("   - Price position relative to MAs...")
            features["price_vs_sma20"] = (df["close"] - features["sma_20"]) / features["sma_20"]
            features["price_vs_sma50"] = (df["close"] - features["sma_50"]) / features["sma_50"]
            
            # ADX for trend strength
            self.logger.info("   - ADX trend strength...")
            features["adx"] = self._calculate_adx(df)

            # === 6. FEATURE INTERACTIONS ===
            self.logger.info("🔄 Calculating feature interactions...")
            
            # Momentum × Volume interactions
            self.logger.info("   - Momentum × Volume interactions...")
            features["momentum_volume_interaction"] = features["price_momentum_10"] * features["volume_ratio_10"]
            
            # Volatility × Volume interactions
            self.logger.info("   - Volatility × Volume interactions...")
            features["volatility_volume_interaction"] = features["volatility_20"] * features["volume_ratio_20"]
            
            # RSI × Momentum interactions
            self.logger.info("   - RSI × Momentum interactions...")
            features["rsi_momentum_interaction"] = features["rsi"] * features["price_momentum_10"]

            # === 7. CLEANUP AND VALIDATION ===
            self.logger.info("🧹 Cleaning and validating features...")
            
            # Remove timestamp column for HMM analysis
            hmm_features = features.drop("timestamp", axis=1)
            
            # Handle NaN values intelligently
            initial_rows = len(hmm_features)
            self.logger.info(f"   - Initial rows: {initial_rows:,}")
            
            # Forward fill for technical indicators
            technical_cols = ["rsi", "macd", "adx", "bb_position", "bb_width"]
            for col in technical_cols:
                if col in hmm_features.columns:
                    hmm_features[col] = hmm_features[col].ffill()
            
            # Fill remaining NaN with 0
            hmm_features = hmm_features.fillna(0)
            
            # Final validation
            final_rows = len(hmm_features)
            removed_rows = initial_rows - final_rows
            
            self.logger.info(f"✅ Comprehensive feature preparation completed:")
            self.logger.info(f"   - Initial rows: {initial_rows:,}")
            self.logger.info(f"   - Final rows: {final_rows:,}")
            self.logger.info(f"   - Removed rows: {removed_rows:,} ({removed_rows/initial_rows*100:.1f}%)")
            self.logger.info(f"   - Features created: {len(hmm_features.columns)}")
            
            # Log feature categories
            self._log_feature_categories(hmm_features)
            
            return hmm_features

        except Exception as e:
            self.logger.exception(f"❌ Error preparing HMM features: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.Series(),
        context="calculate_rsi"
    )
    def _calculate_rsi(self, prices: Any, window: int = 14) -> Any:
        """Calculate Relative Strength Index."""
        self.logger.debug(f"Calculating RSI with window {window}...")
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.Series(),
        context="calculate_macd"
    )
    def _calculate_macd(self, prices: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Any:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        self.logger.debug(f"Calculating MACD (fast={fast}, slow={slow}, signal={signal})...")
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.Series(),
        context="calculate_atr"
    )
    def _calculate_atr(self, df: Any, window: int = 14) -> Any:
        """Calculate Average True Range (ATR)."""
        self.logger.debug(f"Calculating ATR with window {window}...")
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.Series(),
        context="calculate_bollinger_bands"
    )
    def _calculate_bollinger_bands(self, prices: Any, window: int = 20, num_std: float = 2) -> Any:
        """Calculate Bollinger Bands."""
        self.logger.debug(f"Calculating Bollinger Bands (window={window}, std={num_std})...")
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        bb_upper = sma + (std * num_std)
        bb_lower = sma - (std * num_std)
        bb_width = (bb_upper - bb_lower) / sma
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        
        bb_features = pd.DataFrame({
            "bb_upper": bb_upper,
            "bb_middle": sma,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "bb_position": bb_position
        })
        
        return bb_features

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.Series(),
        context="calculate_adx"
    )
    def _calculate_adx(self, df: Any, window: int = 14) -> Any:
        """Calculate Average Directional Index (ADX)."""
        self.logger.debug(f"Calculating ADX with window {window}...")
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.Series(),
        context="calculate_sr_strength"
    )
    def _calculate_sr_strength(self, df: Any, window: int = 20) -> Any:
        """Calculate support/resistance strength indicator."""
        self.logger.debug(f"Calculating S/R strength with window {window}...")
        
        # Calculate price swings
        high_swing = df["high"].rolling(window=window, center=True).max()
        low_swing = df["low"].rolling(window=window, center=True).min()
        
        # Calculate strength based on how close price is to swing levels
        current_price = df["close"]
        high_strength = (high_swing - current_price) / high_swing
        low_strength = (current_price - low_swing) / low_swing
        
        # Combined strength indicator
        sr_strength = (high_strength + low_strength) / 2
        return sr_strength

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="log_feature_categories"
    )
    def _log_feature_categories(self, features: Any) -> None:
        """Log feature categories for analysis."""
        try:
            feature_categories = {
                "momentum": [],
                "volatility": [],
                "volume": [],
                "support_resistance": [],
                "technical": [],
                "interactions": []
            }
            
            for col in features.columns:
                if "momentum" in col.lower():
                    feature_categories["momentum"].append(col)
                elif "volatility" in col.lower():
                    feature_categories["volatility"].append(col)
                elif "volume" in col.lower():
                    feature_categories["volume"].append(col)
                elif any(sr_term in col.lower() for sr_term in ["support", "resistance", "pivot", "sr_", "bb_"]):
                    feature_categories["support_resistance"].append(col)
                elif any(tech_term in col.lower() for tech_term in ["rsi", "macd", "adx", "atr", "sma", "ema"]):
                    feature_categories["technical"].append(col)
                elif "interaction" in col.lower():
                    feature_categories["interactions"].append(col)
                else:
                    feature_categories["technical"].append(col)
            
            self.logger.info("📊 Feature categories:")
            for category, cols in feature_categories.items():
                if cols:
                    self.logger.info(f"   - {category.capitalize()}: {len(cols)} features")
                    if len(cols) <= 5:  # Show all if 5 or fewer
                        self.logger.info(f"     {cols}")
                    else:  # Show first 3 and last 2
                        self.logger.info(f"     {cols[:3]} ... {cols[-2:]}")
        
        except Exception as e:
            self.logger.warning(f"Could not log feature categories: {e}")

    @with_tracing_span("perform_hmm_regime_discovery")
    @resource_monitor
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "HMM regime discovery failed"},
        context="perform_hmm_regime_discovery"
    )
    async def _perform_hmm_regime_discovery(
        self, 
        training_input: dict[str, Any], 
        data: Any
    ) -> dict[str, Any]:
        """Perform HMM regime discovery using hmmlearn with comprehensive features."""
        try:
            self.logger.info("🔍 Starting HMM regime discovery analysis...")
            self.logger.info(f"📊 Input data shape: {data.shape}")

            # Prepare comprehensive features
            self.logger.info("🔧 Preparing comprehensive features for HMM analysis...")
            features = await self._prepare_hmm_features(data)
            
            if features.empty:
                self.logger.error("❌ No features available for HMM analysis")
                return {"success": False, "error": "No features available"}

            self.logger.info(f"📊 Features prepared: {len(features.columns)} features, {len(features)} samples")
            
            # Log feature statistics
            self.logger.info("📊 Feature statistics:")
            for col in features.columns:
                series = features[col].dropna()
                if len(series) > 0:
                    self.logger.info(f"   - {col}: mean={series.mean():.6f}, std={series.std():.6f}, min={series.min():.6f}, max={series.max():.6f}")

            # Try to import hmmlearn
            try:
                from hmmlearn import hmm
                HMM_AVAILABLE = True
                self.logger.info("✅ hmmlearn library available")
            except ImportError:
                HMM_AVAILABLE = False
                self.logger.warning("⚠️ hmmlearn not available, falling back to simple regime detection")

            if HMM_AVAILABLE:
                # Use proper HMM implementation
                return await self._perform_hmmlearn_regime_discovery(features)
            else:
                # Fallback to simple regime detection
                return await self._perform_simple_regime_discovery(features)

        except Exception as e:
            self.logger.exception(f"❌ Error performing HMM regime discovery: {e}")
            return {"success": False, "error": str(e)}

    @with_tracing_span("perform_hmmlearn_regime_discovery")
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "HMMLearn regime discovery failed"},
        context="perform_hmmlearn_regime_discovery"
    )
    async def _perform_hmmlearn_regime_discovery(self, features: Any) -> dict[str, Any]:
        """Perform HMM regime discovery using hmmlearn library."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
            
            self.logger.info("🧠 Using hmmlearn for HMM regime discovery...")
            
            # Scale features for HMM
            self.logger.info("📊 Scaling features for HMM...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Configure HMM parameters
            n_states = 4  # BULL, BEAR, SIDEWAYS, VOLATILE
            n_iter = 100
            random_state = 42
            
            self.logger.info(f"🎯 Training HMM with {n_states} states, {n_iter} iterations...")
            
            # Train Gaussian HMM
            hmm_model = hmm.GaussianHMM(
                n_components=n_states,
                n_iter=n_iter,
                random_state=random_state,
                covariance_type="full",
                init_params="stmc",  # Initialize all parameters
                params="stmc"  # Train all parameters
            )
            
            # Fit the model
            hmm_model.fit(features_scaled)
            
            # Get state sequence
            state_sequence = hmm_model.predict(features_scaled)
            
            # Get state probabilities
            state_probs = hmm_model.predict_proba(features_scaled)
            
            # Interpret states based on feature characteristics
            self.logger.info("🔍 Interpreting HMM states...")
            state_interpretation = self._interpret_hmm_states(features, state_sequence, state_probs)
            
            # Map states to regime names
            regime_states = []
            for state in state_sequence:
                regime_name = state_interpretation["state_to_regime_map"].get(state, f"regime_{state}")
                regime_states.append(regime_name)
            
            # Calculate regime statistics
            regime_counts = {}
            for regime in regime_states:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Calculate transition matrix
            regime_transitions = self._calculate_regime_transitions(regime_states)
            
            # Calculate additional metrics
            metrics = {
                "total_periods": len(regime_states),
                "unique_regimes": len(regime_counts),
                "regime_distribution": regime_counts,
                "hmm_score": hmm_model.score(features_scaled),
                "state_interpretation": state_interpretation["state_analysis"]
            }
            
            self.logger.info(f"✅ HMMLearn regime discovery completed successfully")
            self.logger.info(f"📊 Discovered {len(regime_counts)} unique regimes:")
            for regime, count in regime_counts.items():
                percentage = (count / len(regime_states)) * 100
                self.logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")
            
            # Log HMM model score
            self.logger.info(f"📈 HMM model score: {metrics['hmm_score']:.4f}")
            
            # Log transition matrix
            if regime_transitions:
                self.logger.info("🔄 Regime transition matrix:")
                for from_regime, to_regimes in regime_transitions.items():
                    self.logger.info(f"   From {from_regime}:")
                    for to_regime, prob in to_regimes.items():
                        self.logger.info(f"     → {to_regime}: {prob:.3f}")
            
            return {
                "success": True,
                "regime_states": regime_states,
                "regime_transitions": regime_transitions,
                "metrics": metrics,
                "hmm_model": hmm_model,
                "scaler": scaler,
                "state_probs": state_probs
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Error in HMMLearn regime discovery: {e}")
            return {"success": False, "error": str(e)}

    @with_tracing_span("perform_simple_regime_discovery")
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Simple regime discovery failed"},
        context="perform_simple_regime_discovery"
    )
    async def _perform_simple_regime_discovery(self, features: Any) -> dict[str, Any]:
        """Perform simple regime discovery based on volatility and momentum."""
        try:
            self.logger.info("📊 Using simple regime detection (fallback method)...")
            
            # Use key features for regime classification
            volatility = features.get("volatility_20", features.get("volatility", pd.Series([0] * len(features))))
            momentum = features.get("price_momentum_10", pd.Series([0] * len(features)))
            volume_ratio = features.get("volume_ratio_10", pd.Series([1] * len(features)))
            
            # Fill NaN values
            volatility = volatility.fillna(0)
            momentum = momentum.fillna(0)
            volume_ratio = volume_ratio.fillna(1)
            
            # Calculate quantiles for classification
            vol_quantiles = volatility.quantile([0.2, 0.8])
            mom_quantiles = momentum.quantile([0.3, 0.7])
            vol_quantiles = volume_ratio.quantile([0.3, 0.7])
            
            self.logger.info(f"📊 Volatility quantiles: {vol_quantiles.to_dict()}")
            self.logger.info(f"📊 Momentum quantiles: {mom_quantiles.to_dict()}")
            self.logger.info(f"📊 Volume ratio quantiles: {vol_quantiles.to_dict()}")
            
            # Classify regimes
            regimes = []
            regime_counts = {}
            total_periods = len(features)
            
            progress_interval = max(1, total_periods // 10)
            
            for i in range(total_periods):
                vol = volatility.iloc[i] if hasattr(volatility, 'iloc') else volatility[i]
                mom = momentum.iloc[i] if hasattr(momentum, 'iloc') else momentum[i]
                vol_ratio = volume_ratio.iloc[i] if hasattr(volume_ratio, 'iloc') else volume_ratio[i]
                
                # Classify based on volatility and momentum
                if vol > vol_quantiles[0.8]:
                    if mom > mom_quantiles[0.7]:
                        regime = "high_volatility_bull"
                    elif mom < mom_quantiles[0.3]:
                        regime = "high_volatility_bear"
                    else:
                        regime = "high_volatility_neutral"
                elif vol < vol_quantiles[0.2]:
                    if mom > mom_quantiles[0.7]:
                        regime = "low_volatility_bull"
                    elif mom < mom_quantiles[0.3]:
                        regime = "low_volatility_bear"
                    else:
                        regime = "low_volatility_neutral"
                else:
                    if mom > mom_quantiles[0.7]:
                        regime = "medium_volatility_bull"
                    elif mom < mom_quantiles[0.3]:
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
            regime_transitions = self._calculate_regime_transitions(regimes)
            
            metrics = {
                "total_periods": len(regimes),
                "unique_regimes": len(regime_counts),
                "regime_distribution": regime_counts,
                "method": "simple_classification"
            }
            
            self.logger.info(f"✅ Simple regime discovery completed")
            self.logger.info(f"📊 Discovered {len(regime_counts)} unique regimes:")
            for regime, count in regime_counts.items():
                percentage = (count / len(regimes)) * 100
                self.logger.info(f"   - {regime}: {count:,} periods ({percentage:.1f}%)")
            
            return {
                "success": True,
                "regime_states": regimes,
                "regime_transitions": regime_transitions,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Error in simple regime discovery: {e}")
            return {"success": False, "error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return={"state_to_regime_map": {}, "state_analysis": {}},
        context="interpret_hmm_states"
    )
    def _interpret_hmm_states(self, features: Any, state_sequence: Any, state_probs: Any) -> dict[str, Any]:
        """Interpret HMM states based on feature characteristics."""
        try:
            self.logger.info("🔍 Interpreting HMM states...")
            
            # Analyze each state's characteristics
            state_analysis = {}
            state_to_regime_map = {}
            
            unique_states = sorted(set(state_sequence))
            
            for state in unique_states:
                # Get data points for this state
                state_mask = state_sequence == state
                state_data = features[state_mask]
                
                if len(state_data) == 0:
                    continue
                
                # Calculate state characteristics
                state_char = {
                    "count": len(state_data),
                    "percentage": len(state_data) / len(features) * 100
                }
                
                # Analyze key features for this state
                key_features = [
                    "price_momentum_10", "volatility_20", "volume_ratio_10", 
                    "rsi", "adx", "bb_position"
                ]
                
                for feature in key_features:
                    if feature in state_data.columns:
                        feature_data = state_data[feature].dropna()
                        if len(feature_data) > 0:
                            state_char[f"{feature}_mean"] = feature_data.mean()
                            state_char[f"{feature}_std"] = feature_data.std()
                
                state_analysis[state] = state_char
                
                # Map state to regime based on characteristics
                regime_name = self._map_state_to_regime(state_char)
                state_to_regime_map[state] = regime_name
                
                self.logger.info(f"   State {state} → {regime_name}: {len(state_data)} periods ({state_char['percentage']:.1f}%)")
            
            return {
                "state_to_regime_map": state_to_regime_map,
                "state_analysis": state_analysis
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Error interpreting HMM states: {e}")
            return {"state_to_regime_map": {}, "state_analysis": {}}

    @handle_errors(
        exceptions=(Exception,),
        default_return="unknown_regime",
        context="map_state_to_regime"
    )
    def _map_state_to_regime(self, state_char: dict[str, Any]) -> str:
        """Map state characteristics to regime name."""
        try:
            # Extract key characteristics
            momentum = state_char.get("price_momentum_10_mean", 0)
            volatility = state_char.get("volatility_20_mean", 0)
            volume_ratio = state_char.get("volume_ratio_10_mean", 1)
            rsi = state_char.get("rsi_mean", 50)
            adx = state_char.get("adx_mean", 25)
            
            # Classify based on characteristics
            if volatility > 0.02:  # High volatility
                if momentum > 0.001:  # Positive momentum
                    return "high_volatility_bull"
                elif momentum < -0.001:  # Negative momentum
                    return "high_volatility_bear"
                else:
                    return "high_volatility_neutral"
            elif volatility < 0.01:  # Low volatility
                if momentum > 0.001:
                    return "low_volatility_bull"
                elif momentum < -0.001:
                    return "low_volatility_bear"
                else:
                    return "low_volatility_neutral"
            else:  # Medium volatility
                if momentum > 0.001:
                    return "medium_volatility_bull"
                elif momentum < -0.001:
                    return "medium_volatility_bear"
                else:
                    return "medium_volatility_neutral"
                    
        except Exception as e:
            self.logger.warning(f"Error mapping state to regime: {e}")
            return "unknown_regime"

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="calculate_regime_transitions"
    )
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