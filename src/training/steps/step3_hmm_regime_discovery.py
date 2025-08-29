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

# Enhanced HMM regime management capabilities integrated directly
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Enhanced regime types and data structures
class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class RegimeState:
    """Market regime state data structure."""
    regime_id: int
    regime_type: RegimeType
    confidence: float
    duration: int
    volatility: float
    momentum: float
    volume_profile: float
    timestamp: pd.Timestamp
    features: Dict[str, float]
    transition_probability: float = 0.0
    stability_score: float = 0.0
    regime_quality: float = 0.0
    regime_persistence: float = 0.0
    regime_complexity: float = 0.0

@dataclass
class RegimeTransition:
    """Regime transition data structure."""
    from_regime: int
    to_regime: int
    probability: float
    timestamp: pd.Timestamp
    trigger_features: Dict[str, float]
    confidence: float
    transition_strength: float = 0.0
    transition_duration: int = 0

@dataclass
class RegimeCluster:
    """Regime cluster data structure."""
    cluster_id: int
    center: np.ndarray
    regime_states: List[RegimeState]
    cluster_quality: float
    cluster_stability: float
    cluster_features: Dict[str, float]

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
        
        # Initialize enhanced HMM regime management capabilities
        self.enhanced_hmm_capabilities = {
            "regime_states": [],
            "regime_transitions": [],
            "hmm_model": None,
            "kmeans_model": None,
            "transition_model": None,
            "scaler": None,
            "regime_history": [],
            "quality_metrics": {},
            "regime_prediction_model": None,
            "regime_clustering_model": None,
            "regime_stability_analyzer": None,
            "regime_transition_detector": None,
            "regime_prediction_accuracy": {},
            "regime_stability_metrics": {},
            "regime_quality_scores": {},
            "regime_redundancy_metrics": {}
        }
        
        # Enhanced regime management state
        self.regime_management_state = {
            "last_regime_analysis": None,
            "regime_analysis_count": 0,
            "regime_quality_scores": {},
            "regime_redundancy_metrics": {},
            "regime_prediction_accuracy": {},
            "regime_stability_metrics": {},
            "regime_transition_probabilities": {},
            "regime_stability_scores": {},
            "regime_change_detection": {},
            "regime_forecasting": {}
        }
        
        self.logger.info("✅ Enhanced HMM regime management capabilities initialized")
        
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
            
            # Use enhanced HMM regime discovery with integrated capabilities
            self.logger.info("🧠 Using enhanced HMM regime discovery with integrated capabilities...")
            regime_results = await self._perform_enhanced_hmm_regime_discovery(
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

    @with_tracing_span("perform_enhanced_hmm_regime_discovery")
    @resource_monitor
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Enhanced HMM regime discovery failed"},
        context="perform_enhanced_hmm_regime_discovery"
    )
    async def _perform_enhanced_hmm_regime_discovery(
        self, 
        training_input: dict[str, Any], 
        data: Any
    ) -> dict[str, Any]:
        """Perform enhanced HMM regime discovery with integrated capabilities."""
        try:
            self.logger.info("🧠 Starting enhanced HMM regime discovery with integrated capabilities...")
            
            # Update regime analysis state
            self.regime_management_state["last_regime_analysis"] = pd.Timestamp.now()
            self.regime_management_state["regime_analysis_count"] += 1
            
            # Prepare comprehensive features
            features = await self._prepare_hmm_features(data)
            
            if features.empty:
                self.logger.error("❌ No features available for enhanced HMM analysis")
                return {"success": False, "error": "No features available"}

            self.logger.info(f"📊 Features prepared: {len(features.columns)} features, {len(features)} samples")
            
            # Train enhanced HMM models
            training_result = await self._train_enhanced_hmm_models(features)
            
            if not training_result.get("success", False):
                error_msg = training_result.get("error", "Enhanced HMM training failed")
                self.logger.error(f"❌ Enhanced HMM training failed: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Predict regime changes
            prediction_result = await self._predict_enhanced_regime_changes(features)
            
            if not prediction_result.get("success", False):
                error_msg = prediction_result.get("error", "Enhanced HMM prediction failed")
                self.logger.error(f"❌ Enhanced HMM prediction failed: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Calculate quality metrics
            quality_metrics = self._calculate_regime_quality_metrics(features, prediction_result)
            
            # Eliminate redundancy
            redundancy_metrics = self._eliminate_regime_redundancy(prediction_result)
            
            # Generate regime summary
            regime_summary = self._generate_enhanced_regime_summary()
            
            # Update regime management state
            self.regime_management_state["regime_quality_scores"] = quality_metrics
            self.regime_management_state["regime_redundancy_metrics"] = redundancy_metrics
            self.enhanced_hmm_capabilities["quality_metrics"] = quality_metrics
            
            # Format results for pipeline compatibility
            regime_results = {
                "success": True,
                "regime_states": prediction_result.get("regime_states", []),
                "regime_transitions": prediction_result.get("regime_transitions", {}),
                "metrics": {
                    "training_report": training_result.get("training_report", {}),
                    "regime_summary": regime_summary,
                    "current_regime": prediction_result.get("current_regime"),
                    "enhanced_features": True,
                    "quality_metrics": quality_metrics,
                    "redundancy_metrics": redundancy_metrics
                },
                "enhanced_analysis": True
            }
            
            self.logger.info("✅ Enhanced HMM regime discovery completed successfully")
            return regime_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced HMM regime discovery: {e}")
            return {"success": False, "error": str(e)}

    # === ENHANCED HMM REGIME MANAGEMENT METHODS ===
    
    async def _train_enhanced_hmm_models(self, features: pd.DataFrame) -> dict[str, Any]:
        """Train enhanced HMM models with comprehensive capabilities."""
        try:
            self.logger.info("🎯 Training enhanced HMM models...")
            
            # Train HMM model
            hmm_result = await self._train_hmm_model(features)
            
            # Train clustering model
            clustering_result = await self._train_clustering_model(features)
            
            # Train transition model
            transition_result = await self._train_transition_model(features)
            
            # Combine results
            training_result = {
                "success": hmm_result.get("success", False) and clustering_result.get("success", False),
                "hmm_model": hmm_result.get("model"),
                "clustering_model": clustering_result.get("model"),
                "transition_model": transition_result.get("model"),
                "training_report": {
                    "hmm_score": hmm_result.get("score", 0.0),
                    "clustering_score": clustering_result.get("score", 0.0),
                    "transition_accuracy": transition_result.get("accuracy", 0.0)
                }
            }
            
            self.logger.info("✅ Enhanced HMM models trained successfully")
            return training_result
            
        except Exception as e:
            self.logger.error(f"Error training enhanced HMM models: {e}")
            return {"success": False, "error": str(e)}

    async def _train_hmm_model(self, features: pd.DataFrame) -> dict[str, Any]:
        """Train HMM model for regime discovery."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train HMM
            hmm_model = hmm.GaussianHMM(
                n_components=4,
                n_iter=100,
                random_state=42,
                covariance_type="full"
            )
            hmm_model.fit(features_scaled)
            
            # Get score
            score = hmm_model.score(features_scaled)
            
            return {
                "success": True,
                "model": hmm_model,
                "scaler": scaler,
                "score": score
            }
            
        except Exception as e:
            self.logger.error(f"Error training HMM model: {e}")
            return {"success": False, "error": str(e)}

    async def _train_clustering_model(self, features: pd.DataFrame) -> dict[str, Any]:
        """Train clustering model for regime analysis."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate score
            score = silhouette_score(features_scaled, cluster_labels)
            
            return {
                "success": True,
                "model": kmeans,
                "scaler": scaler,
                "score": score,
                "cluster_labels": cluster_labels
            }
            
        except Exception as e:
            self.logger.error(f"Error training clustering model: {e}")
            return {"success": False, "error": str(e)}

    async def _train_transition_model(self, features: pd.DataFrame) -> dict[str, Any]:
        """Train transition model for regime changes."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Create transition labels (simplified)
            transitions = []
            for i in range(1, len(features)):
                # Simple transition detection based on feature changes
                feature_change = features.iloc[i] - features.iloc[i-1]
                transition_label = 1 if feature_change.mean() > 0 else 0
                transitions.append(transition_label)
            
            # Prepare data
            X = features.iloc[:-1].values
            y = transitions
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            transition_model = RandomForestClassifier(n_estimators=100, random_state=42)
            transition_model.fit(X_train, y_train)
            
            # Calculate accuracy
            accuracy = transition_model.score(X_test, y_test)
            
            return {
                "success": True,
                "model": transition_model,
                "accuracy": accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error training transition model: {e}")
            return {"success": False, "error": str(e)}

    async def _predict_enhanced_regime_changes(self, features: pd.DataFrame) -> dict[str, Any]:
        """Predict enhanced regime changes using trained models."""
        try:
            self.logger.info("🔮 Predicting enhanced regime changes...")
            
            # Get trained models from state
            hmm_model = self.enhanced_hmm_capabilities.get("hmm_model")
            clustering_model = self.enhanced_hmm_capabilities.get("kmeans_model")
            transition_model = self.enhanced_hmm_capabilities.get("transition_model")
            
            if not all([hmm_model, clustering_model, transition_model]):
                self.logger.warning("⚠️ Not all models available, using fallback prediction")
                return await self._perform_simple_regime_discovery(features)
            
            # Predict regime states
            regime_states = self._predict_regime_states(features, hmm_model, clustering_model)
            
            # Predict regime transitions
            regime_transitions = self._predict_regime_transitions(features, transition_model)
            
            # Calculate current regime
            current_regime = self._calculate_current_regime(regime_states)
            
            prediction_result = {
                "success": True,
                "regime_states": regime_states,
                "regime_transitions": regime_transitions,
                "current_regime": current_regime
            }
            
            self.logger.info("✅ Enhanced regime change prediction completed")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting enhanced regime changes: {e}")
            return {"success": False, "error": str(e)}

    def _predict_regime_states(self, features: pd.DataFrame, hmm_model: Any, clustering_model: Any) -> List[RegimeState]:
        """Predict regime states using HMM and clustering models."""
        try:
            regime_states = []
            
            # Get HMM states
            hmm_states = hmm_model.predict(features.values)
            
            # Get cluster labels
            cluster_labels = clustering_model.predict(features.values)
            
            # Create regime states
            for i in range(len(features)):
                regime_state = RegimeState(
                    regime_id=int(hmm_states[i]),
                    regime_type=self._map_regime_type(hmm_states[i]),
                    confidence=0.8,  # Placeholder confidence
                    duration=1,
                    volatility=features.iloc[i].get("volatility_20", 0.0),
                    momentum=features.iloc[i].get("price_momentum_10", 0.0),
                    volume_profile=features.iloc[i].get("volume_ratio_10", 1.0),
                    timestamp=pd.Timestamp.now(),
                    features=features.iloc[i].to_dict()
                )
                regime_states.append(regime_state)
            
            return regime_states
            
        except Exception as e:
            self.logger.error(f"Error predicting regime states: {e}")
            return []

    def _predict_regime_transitions(self, features: pd.DataFrame, transition_model: Any) -> List[RegimeTransition]:
        """Predict regime transitions using transition model."""
        try:
            transitions = []
            
            # Predict transitions
            transition_predictions = transition_model.predict(features.values)
            
            # Create transition objects
            for i in range(1, len(transition_predictions)):
                if transition_predictions[i] == 1:  # Transition detected
                    transition = RegimeTransition(
                        from_regime=i-1,
                        to_regime=i,
                        probability=0.8,  # Placeholder probability
                        timestamp=pd.Timestamp.now(),
                        trigger_features=features.iloc[i].to_dict(),
                        confidence=0.7
                    )
                    transitions.append(transition)
            
            return transitions
            
        except Exception as e:
            self.logger.error(f"Error predicting regime transitions: {e}")
            return []

    def _calculate_current_regime(self, regime_states: List[RegimeState]) -> RegimeState:
        """Calculate current regime from regime states."""
        try:
            if not regime_states:
                return RegimeState(
                    regime_id=0,
                    regime_type=RegimeType.SIDEWAYS,
                    confidence=0.0,
                    duration=0,
                    volatility=0.0,
                    momentum=0.0,
                    volume_profile=1.0,
                    timestamp=pd.Timestamp.now(),
                    features={}
                )
            
            # Return the most recent regime state
            return regime_states[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating current regime: {e}")
            return RegimeState(
                regime_id=0,
                regime_type=RegimeType.SIDEWAYS,
                confidence=0.0,
                duration=0,
                volatility=0.0,
                momentum=0.0,
                volume_profile=1.0,
                timestamp=pd.Timestamp.now(),
                features={}
            )

    def _map_regime_type(self, regime_id: int) -> RegimeType:
        """Map regime ID to regime type."""
        try:
            regime_types = [RegimeType.BULL, RegimeType.BEAR, RegimeType.SIDEWAYS, RegimeType.VOLATILE]
            return regime_types[regime_id % len(regime_types)]
        except Exception:
            return RegimeType.SIDEWAYS

    def _calculate_regime_quality_metrics(self, features: pd.DataFrame, prediction_result: dict[str, Any]) -> dict[str, float]:
        """Calculate quality metrics for regime predictions."""
        try:
            metrics = {}
            
            # Basic metrics
            regime_states = prediction_result.get("regime_states", [])
            metrics["regime_count"] = len(regime_states)
            metrics["transition_count"] = len(prediction_result.get("regime_transitions", []))
            
            # Regime distribution
            regime_types = [state.regime_type for state in regime_states]
            unique_regimes = len(set(regime_types))
            metrics["regime_diversity"] = unique_regimes / len(regime_types) if regime_types else 0.0
            
            # Confidence metrics
            confidences = [state.confidence for state in regime_states]
            metrics["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Volatility metrics
            volatilities = [state.volatility for state in regime_states]
            metrics["avg_volatility"] = sum(volatilities) / len(volatilities) if volatilities else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime quality metrics: {e}")
            return {}

    def _eliminate_regime_redundancy(self, prediction_result: dict[str, Any]) -> dict[str, Any]:
        """Eliminate redundant regime predictions."""
        try:
            metrics = {}
            
            regime_states = prediction_result.get("regime_states", [])
            
            # Count similar consecutive regimes
            redundant_count = 0
            for i in range(1, len(regime_states)):
                if regime_states[i].regime_id == regime_states[i-1].regime_id:
                    redundant_count += 1
            
            metrics["redundant_regimes"] = redundant_count
            metrics["redundancy_ratio"] = redundant_count / len(regime_states) if regime_states else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating regime redundancy: {e}")
            return {}

    def _generate_enhanced_regime_summary(self) -> dict[str, Any]:
        """Generate enhanced regime summary."""
        try:
            summary = {
                "total_regimes": len(self.enhanced_hmm_capabilities.get("regime_states", [])),
                "quality_metrics": self.regime_management_state.get("regime_quality_scores", {}),
                "redundancy_metrics": self.regime_management_state.get("regime_redundancy_metrics", {}),
                "analysis_count": self.regime_management_state.get("regime_analysis_count", 0)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced regime summary: {e}")
            return {}

    # === ENHANCED REGIME FEATURE GENERATION ===
    
    async def get_enhanced_regime_features(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Get enhanced regime features for feature engineering integration.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: Enhanced regime features
        """
        try:
            self.logger.info("🔧 Generating enhanced regime features...")
            
            # Prepare features
            features = await self._prepare_hmm_features(market_data)
            
            if features.empty:
                return {}
            
            # Perform enhanced regime discovery
            regime_result = await self._perform_enhanced_hmm_regime_discovery(
                {"symbol": "UNKNOWN"}, features
            )
            
            if not regime_result.get("success", False):
                return {}
            
            # Extract regime features
            regime_states = regime_result.get("regime_states", [])
            regime_transitions = regime_result.get("regime_transitions", {})
            
            # Create regime features
            regime_features = {}
            
            if regime_states:
                current_regime = regime_states[-1] if regime_states else None
                if current_regime:
                    regime_features["current_regime_id"] = current_regime.regime_id
                    regime_features["current_regime_type"] = current_regime.regime_type.value
                    regime_features["regime_confidence"] = current_regime.confidence
                    regime_features["regime_volatility"] = current_regime.volatility
                    regime_features["regime_momentum"] = current_regime.momentum
                    regime_features["regime_volume_profile"] = current_regime.volume_profile
            
            # Add transition features
            regime_features["transition_count"] = len(regime_transitions)
            regime_features["regime_stability"] = 1.0 - (len(regime_transitions) / max(len(regime_states), 1))
            
            # Add quality metrics
            quality_metrics = regime_result.get("metrics", {}).get("quality_metrics", {})
            for key, value in quality_metrics.items():
                regime_features[f"regime_quality_{key}"] = value
            
            self.logger.info("✅ Enhanced regime features generated")
            return regime_features
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced regime features: {e}")
            return {}

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
        """Perform HMM regime discovery using hmmlearn library with 20-cluster composite approach."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            self.logger.info("🧠 Using hmmlearn with 20-cluster composite approach...")
            
            # Scale features for HMM
            self.logger.info("📊 Scaling features for HMM...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # === PHASE 1: HMM State Discovery ===
            # Configure HMM parameters for initial state discovery
            n_hmm_states = 4  # Initial HMM states for basic regime identification
            n_iter = 100
            random_state = 42
            
            self.logger.info(f"🎯 Phase 1: Training HMM with {n_hmm_states} states...")
            
            # Train Gaussian HMM
            hmm_model = hmm.GaussianHMM(
                n_components=n_hmm_states,
                n_iter=n_iter,
                random_state=random_state,
                covariance_type="full",
                init_params="stmc",
                params="stmc"
            )
            
            # Fit the model
            hmm_model.fit(features_scaled)
            
            # Get HMM state sequence and probabilities
            hmm_state_sequence = hmm_model.predict(features_scaled)
            hmm_state_probs = hmm_model.predict_proba(features_scaled)
            
            # === PHASE 2: 20-Cluster Composite Analysis ===
            self.logger.info("🎯 Phase 2: Creating 20-cluster composite analysis...")
            
            # Create composite features combining HMM states with original features
            composite_features = self._create_composite_features(features, hmm_state_sequence, hmm_state_probs)
            
            # Scale composite features
            composite_scaler = StandardScaler()
            composite_features_scaled = composite_scaler.fit_transform(composite_features)
            
            # Apply K-means clustering for 20 clusters
            n_clusters = 20
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = kmeans.fit_predict(composite_features_scaled)
            
            # === PHASE 3: Cluster Quality Analysis ===
            self.logger.info("🎯 Phase 3: Analyzing cluster quality...")
            
            # Calculate cluster quality metrics
            cluster_metrics = self._calculate_cluster_quality_metrics(
                composite_features_scaled, cluster_labels, kmeans
            )
            
            # === PHASE 4: Regime Interpretation ===
            self.logger.info("🎯 Phase 4: Interpreting composite regimes...")
            
            # Create composite cluster analysis
            composite_analysis = self._analyze_composite_clusters(
                features, hmm_state_sequence, cluster_labels, cluster_metrics
            )
            
            # === PHASE 5: Generate Reports ===
            self.logger.info("🎯 Phase 5: Generating comprehensive reports...")
            
            # Generate detailed reports
            reports = await self._generate_comprehensive_reports(
                features, hmm_state_sequence, cluster_labels, composite_analysis, cluster_metrics
            )
            
            # === PHASE 6: Create Output Data ===
            self.logger.info("🎯 Phase 6: Creating output data structures...")
            
            # Create composite cluster DataFrame
            composite_df = self._create_composite_cluster_dataframe(
                features, hmm_state_sequence, cluster_labels, composite_analysis
            )
            
            # Create intensity DataFrame
            intensity_df = self._create_intensity_dataframe(
                features, hmm_state_sequence, cluster_labels, composite_analysis
            )
            
            # Create meta information
            meta_info = self._create_meta_information(
                hmm_model, kmeans, composite_analysis, cluster_metrics, reports
            )
            
            # Calculate final metrics
            final_metrics = {
                "total_periods": len(cluster_labels),
                "hmm_states": n_hmm_states,
                "composite_clusters": n_clusters,
                "cluster_quality": cluster_metrics,
                "hmm_score": hmm_model.score(features_scaled),
                "composite_analysis": composite_analysis,
                "reports_generated": list(reports.keys())
            }
            
            self.logger.info(f"✅ Composite HMM regime discovery completed successfully")
            self.logger.info(f"📊 HMM States: {n_hmm_states}, Composite Clusters: {n_clusters}")
            self.logger.info(f"📈 Cluster Quality - Silhouette: {cluster_metrics['silhouette_score']:.4f}")
            self.logger.info(f"📊 Reports Generated: {len(reports)}")
            
            return {
                "success": True,
                "hmm_model": hmm_model,
                "kmeans_model": kmeans,
                "scaler": scaler,
                "composite_scaler": composite_scaler,
                "hmm_state_sequence": hmm_state_sequence,
                "hmm_state_probs": hmm_state_probs,
                "cluster_labels": cluster_labels,
                "composite_df": composite_df,
                "intensity_df": intensity_df,
                "meta_info": meta_info,
                "metrics": final_metrics,
                "reports": reports
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Error in composite HMM regime discovery: {e}")
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

    # === COMPOSITE HMM HELPER METHODS ===

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="create_composite_features"
    )
    def _create_composite_features(self, features: Any, hmm_states: Any, hmm_probs: Any) -> Any:
        """Create composite features combining HMM states with original features."""
        try:
            self.logger.info("🔧 Creating composite features...")
            
            # Convert to DataFrame if needed
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            
            # Create composite features DataFrame
            composite_df = features.copy()
            
            # Add HMM state features
            composite_df["hmm_state"] = hmm_states
            composite_df["hmm_state_prob_max"] = np.max(hmm_probs, axis=1)
            composite_df["hmm_state_entropy"] = -np.sum(hmm_probs * np.log(hmm_probs + 1e-10), axis=1)
            
            # Add HMM state probability features
            for i in range(hmm_probs.shape[1]):
                composite_df[f"hmm_state_prob_{i}"] = hmm_probs[:, i]
            
            # Add feature interactions with HMM states
            key_features = ["price_momentum_10", "volatility_20", "volume_ratio_10", "rsi", "adx"]
            for feature in key_features:
                if feature in composite_df.columns:
                    composite_df[f"{feature}_x_hmm_state"] = composite_df[feature] * composite_df["hmm_state"]
                    composite_df[f"{feature}_x_hmm_entropy"] = composite_df[feature] * composite_df["hmm_state_entropy"]
            
            # Add rolling statistics for HMM states
            composite_df["hmm_state_persistence"] = self._calculate_persistence(hmm_states)
            composite_df["hmm_state_transitions"] = self._calculate_transitions(hmm_states)
            
            self.logger.info(f"✅ Created composite features: {len(composite_df.columns)} total features")
            return composite_df
            
        except Exception as e:
            self.logger.exception(f"❌ Error creating composite features: {e}")
            return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="calculate_cluster_quality_metrics"
    )
    def _calculate_cluster_quality_metrics(self, features_scaled: Any, cluster_labels: Any, kmeans_model: Any) -> dict[str, Any]:
        """Calculate comprehensive cluster quality metrics."""
        try:
            self.logger.info("📊 Calculating cluster quality metrics...")
            
            metrics = {}
            
            # Silhouette score (higher is better, range: -1 to 1)
            try:
                metrics["silhouette_score"] = silhouette_score(features_scaled, cluster_labels)
            except Exception:
                metrics["silhouette_score"] = 0.0
            
            # Calinski-Harabasz score (higher is better)
            try:
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(features_scaled, cluster_labels)
            except Exception:
                metrics["calinski_harabasz_score"] = 0.0
            
            # Davies-Bouldin score (lower is better)
            try:
                metrics["davies_bouldin_score"] = davies_bouldin_score(features_scaled, cluster_labels)
            except Exception:
                metrics["davies_bouldin_score"] = float('inf')
            
            # Inertia (lower is better)
            metrics["inertia"] = kmeans_model.inertia_
            
            # Cluster size distribution
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            metrics["cluster_sizes"] = dict(zip(unique_labels, counts))
            metrics["min_cluster_size"] = np.min(counts)
            metrics["max_cluster_size"] = np.max(counts)
            metrics["mean_cluster_size"] = np.mean(counts)
            metrics["std_cluster_size"] = np.std(counts)
            
            # Cluster balance (coefficient of variation)
            metrics["cluster_balance"] = metrics["std_cluster_size"] / metrics["mean_cluster_size"] if metrics["mean_cluster_size"] > 0 else 0
            
            # Distance to cluster centers
            distances = kmeans_model.transform(features_scaled)
            min_distances = np.min(distances, axis=1)
            metrics["mean_distance_to_center"] = np.mean(min_distances)
            metrics["max_distance_to_center"] = np.max(min_distances)
            
            self.logger.info(f"✅ Cluster quality metrics calculated:")
            self.logger.info(f"   - Silhouette: {metrics['silhouette_score']:.4f}")
            self.logger.info(f"   - Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
            self.logger.info(f"   - Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
            self.logger.info(f"   - Inertia: {metrics['inertia']:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Error calculating cluster quality metrics: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="analyze_composite_clusters"
    )
    def _analyze_composite_clusters(self, features: Any, hmm_states: Any, cluster_labels: Any, cluster_metrics: dict[str, Any]) -> dict[str, Any]:
        """Analyze composite clusters and their characteristics."""
        try:
            self.logger.info("🔍 Analyzing composite clusters...")
            
            analysis = {
                "cluster_characteristics": {},
                "hmm_state_distribution": {},
                "feature_importance": {},
                "cluster_stability": {},
                "market_conditions": {}
            }
            
            # Analyze each cluster
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_data = features[cluster_mask]
                cluster_hmm_states = hmm_states[cluster_mask]
                
                # Cluster characteristics
                cluster_char = {
                    "size": len(cluster_data),
                    "percentage": len(cluster_data) / len(features) * 100,
                    "hmm_state_distribution": self._calculate_hmm_state_distribution(cluster_hmm_states),
                    "feature_means": {},
                    "feature_stds": {},
                    "dominant_hmm_state": self._get_dominant_hmm_state(cluster_hmm_states)
                }
                
                # Calculate feature statistics for this cluster
                for col in features.columns:
                    if col in cluster_data.columns:
                        cluster_char["feature_means"][col] = cluster_data[col].mean()
                        cluster_char["feature_stds"][col] = cluster_data[col].std()
                
                analysis["cluster_characteristics"][cluster_id] = cluster_char
                
                # Determine market condition for this cluster
                market_condition = self._determine_market_condition(cluster_char)
                analysis["market_conditions"][cluster_id] = market_condition
            
            # Calculate overall HMM state distribution
            analysis["hmm_state_distribution"] = self._calculate_hmm_state_distribution(hmm_states)
            
            # Calculate feature importance across clusters
            analysis["feature_importance"] = self._calculate_feature_importance(features, cluster_labels)
            
            # Calculate cluster stability metrics
            analysis["cluster_stability"] = self._calculate_cluster_stability(cluster_labels, cluster_metrics)
            
            self.logger.info(f"✅ Composite cluster analysis completed for {len(unique_clusters)} clusters")
            return analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Error analyzing composite clusters: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="generate_comprehensive_reports"
    )
    async def _generate_comprehensive_reports(self, features: Any, hmm_states: Any, cluster_labels: Any, composite_analysis: dict[str, Any], cluster_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive reports for the composite HMM analysis."""
        try:
            self.logger.info("📊 Generating comprehensive reports...")
            
            reports = {}
            
            # 1. Cluster Quality Report
            reports["cluster_quality"] = self._generate_cluster_quality_report(cluster_metrics)
            
            # 2. Cluster Characteristics Report
            reports["cluster_characteristics"] = self._generate_cluster_characteristics_report(composite_analysis)
            
            # 3. Market Conditions Report
            reports["market_conditions"] = self._generate_market_conditions_report(composite_analysis)
            
            # 4. Feature Importance Report
            reports["feature_importance"] = self._generate_feature_importance_report(composite_analysis)
            
            # 5. HMM State Analysis Report
            reports["hmm_state_analysis"] = self._generate_hmm_state_analysis_report(hmm_states, composite_analysis)
            
            # 6. Temporal Analysis Report
            reports["temporal_analysis"] = self._generate_temporal_analysis_report(cluster_labels, features)
            
            # 7. Recommendations Report
            reports["recommendations"] = self._generate_recommendations_report(cluster_metrics, composite_analysis)
            
            self.logger.info(f"✅ Generated {len(reports)} comprehensive reports")
            return reports
            
        except Exception as e:
            self.logger.exception(f"❌ Error generating reports: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="create_composite_cluster_dataframe"
    )
    def _create_composite_cluster_dataframe(self, features: Any, hmm_states: Any, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Create composite cluster DataFrame with all relevant information."""
        try:
            self.logger.info("📊 Creating composite cluster DataFrame...")
            
            # Create base DataFrame
            df = features.copy()
            df["hmm_state"] = hmm_states
            df["composite_cluster_id"] = cluster_labels
            
            # Add cluster characteristics
            for cluster_id, char in composite_analysis.get("cluster_characteristics", {}).items():
                cluster_mask = cluster_labels == cluster_id
                df.loc[cluster_mask, "cluster_size"] = char["size"]
                df.loc[cluster_mask, "cluster_percentage"] = char["percentage"]
                df.loc[cluster_mask, "dominant_hmm_state"] = char["dominant_hmm_state"]
                df.loc[cluster_mask, "market_condition"] = composite_analysis.get("market_conditions", {}).get(cluster_id, "unknown")
            
            # Add intensity scores
            df["cluster_intensity"] = self._calculate_cluster_intensity(cluster_labels, composite_analysis)
            
            # Add stability metrics
            df["cluster_stability"] = self._calculate_cluster_stability_scores(cluster_labels, composite_analysis)
            
            self.logger.info(f"✅ Created composite cluster DataFrame: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.exception(f"❌ Error creating composite cluster DataFrame: {e}")
            return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="create_intensity_dataframe"
    )
    def _create_intensity_dataframe(self, features: Any, hmm_states: Any, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Create intensity DataFrame for cluster analysis."""
        try:
            self.logger.info("📊 Creating intensity DataFrame...")
            
            # Create intensity DataFrame
            intensity_df = pd.DataFrame()
            intensity_df["composite_cluster_id"] = cluster_labels
            intensity_df["hmm_state"] = hmm_states
            
            # Calculate intensity scores for each cluster
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_char = composite_analysis.get("cluster_characteristics", {}).get(cluster_id, {})
                
                # Calculate various intensity metrics
                intensity_df.loc[cluster_mask, "cluster_intensity"] = cluster_char.get("size", 0) / len(features)
                intensity_df.loc[cluster_mask, "volatility_intensity"] = self._calculate_volatility_intensity(features, cluster_mask)
                intensity_df.loc[cluster_mask, "momentum_intensity"] = self._calculate_momentum_intensity(features, cluster_mask)
                intensity_df.loc[cluster_mask, "volume_intensity"] = self._calculate_volume_intensity(features, cluster_mask)
                
                # Combined intensity score
                intensity_df.loc[cluster_mask, "combined_intensity"] = (
                    intensity_df.loc[cluster_mask, "cluster_intensity"] * 0.3 +
                    intensity_df.loc[cluster_mask, "volatility_intensity"] * 0.3 +
                    intensity_df.loc[cluster_mask, "momentum_intensity"] * 0.2 +
                    intensity_df.loc[cluster_mask, "volume_intensity"] * 0.2
                )
            
            self.logger.info(f"✅ Created intensity DataFrame: {len(intensity_df)} rows, {len(intensity_df.columns)} columns")
            return intensity_df
            
        except Exception as e:
            self.logger.exception(f"❌ Error creating intensity DataFrame: {e}")
            return pd.DataFrame()

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="create_meta_information"
    )
    def _create_meta_information(self, hmm_model: Any, kmeans_model: Any, composite_analysis: dict[str, Any], cluster_metrics: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
        """Create meta information for the composite HMM analysis."""
        try:
            self.logger.info("📊 Creating meta information...")
            
            meta = {
                "creation_timestamp": pd.Timestamp.now().isoformat(),
                "hmm_model_info": {
                    "n_components": hmm_model.n_components,
                    "covariance_type": hmm_model.covariance_type,
                    "n_iter": hmm_model.n_iter,
                    "converged": hmm_model.monitor_.converged,
                    "score": hmm_model.score(hmm_model.means_)
                },
                "kmeans_model_info": {
                    "n_clusters": kmeans_model.n_clusters,
                    "inertia": kmeans_model.inertia_,
                    "n_iter": kmeans_model.n_iter_,
                    "converged": kmeans_model.n_iter_ < kmeans_model.max_iter
                },
                "cluster_metrics": cluster_metrics,
                "composite_analysis_summary": {
                    "total_clusters": len(composite_analysis.get("cluster_characteristics", {})),
                    "hmm_states": len(composite_analysis.get("hmm_state_distribution", {})),
                    "market_conditions": len(composite_analysis.get("market_conditions", {}))
                },
                "reports_summary": {
                    "total_reports": len(reports),
                    "report_types": list(reports.keys())
                },
                "feature_summary": {
                    "total_features": len(composite_analysis.get("feature_importance", {})),
                    "top_features": sorted(
                        composite_analysis.get("feature_importance", {}).items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                }
            }
            
            self.logger.info("✅ Created meta information")
            return meta
            
        except Exception as e:
            self.logger.exception(f"❌ Error creating meta information: {e}")
            return {}

    # === ADDITIONAL HELPER METHODS ===

    def _calculate_persistence(self, states: Any) -> Any:
        """Calculate state persistence (how long we stay in current state)."""
        try:
            persistence = np.zeros(len(states))
            current_state = states[0]
            current_count = 1
            
            for i in range(1, len(states)):
                if states[i] == current_state:
                    current_count += 1
                else:
                    # Update persistence for the previous state
                    for j in range(i - current_count, i):
                        persistence[j] = current_count
                    current_state = states[i]
                    current_count = 1
            
            # Handle the last state
            for j in range(len(states) - current_count, len(states)):
                persistence[j] = current_count
            
            return persistence
        except Exception:
            return np.zeros(len(states))

    def _calculate_transitions(self, states: Any) -> Any:
        """Calculate number of state transitions."""
        try:
            transitions = np.zeros(len(states))
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    transitions[i] = 1
            return transitions
        except Exception:
            return np.zeros(len(states))

    def _calculate_hmm_state_distribution(self, hmm_states: Any) -> dict[int, int]:
        """Calculate distribution of HMM states."""
        try:
            unique_states, counts = np.unique(hmm_states, return_counts=True)
            return dict(zip(unique_states, counts))
        except Exception:
            return {}

    def _get_dominant_hmm_state(self, hmm_states: Any) -> int:
        """Get the dominant HMM state in a cluster."""
        try:
            unique_states, counts = np.unique(hmm_states, return_counts=True)
            return unique_states[np.argmax(counts)]
        except Exception:
            return 0

    def _determine_market_condition(self, cluster_char: dict[str, Any]) -> str:
        """Determine market condition for a cluster based on its characteristics."""
        try:
            # Extract key metrics
            momentum = cluster_char.get("feature_means", {}).get("price_momentum_10", 0)
            volatility = cluster_char.get("feature_means", {}).get("volatility_20", 0)
            volume_ratio = cluster_char.get("feature_means", {}).get("volume_ratio_10", 1)
            rsi = cluster_char.get("feature_means", {}).get("rsi", 50)
            
            # Determine market condition
            if volatility > 0.02:
                if momentum > 0.001:
                    return "high_volatility_bull"
                elif momentum < -0.001:
                    return "high_volatility_bear"
                else:
                    return "high_volatility_neutral"
            elif volatility < 0.01:
                if momentum > 0.001:
                    return "low_volatility_bull"
                elif momentum < -0.001:
                    return "low_volatility_bear"
                else:
                    return "low_volatility_neutral"
            else:
                if momentum > 0.001:
                    return "medium_volatility_bull"
                elif momentum < -0.001:
                    return "medium_volatility_bear"
                else:
                    return "medium_volatility_neutral"
        except Exception:
            return "unknown"

    def _calculate_feature_importance(self, features: Any, cluster_labels: Any) -> dict[str, float]:
        """Calculate feature importance based on cluster separation."""
        try:
            importance = {}
            for col in features.columns:
                if col in features.columns:
                    # Calculate feature variance between clusters vs within clusters
                    total_var = features[col].var()
                    if total_var > 0:
                        between_cluster_var = 0
                        within_cluster_var = 0
                        
                        for cluster_id in np.unique(cluster_labels):
                            cluster_mask = cluster_labels == cluster_id
                            cluster_mean = features.loc[cluster_mask, col].mean()
                            cluster_var = features.loc[cluster_mask, col].var()
                            cluster_size = cluster_mask.sum()
                            
                            between_cluster_var += cluster_size * (cluster_mean - features[col].mean()) ** 2
                            within_cluster_var += cluster_size * cluster_var
                        
                        if within_cluster_var > 0:
                            importance[col] = between_cluster_var / within_cluster_var
                        else:
                            importance[col] = 0
                    else:
                        importance[col] = 0
            
            return importance
        except Exception:
            return {}

    def _calculate_cluster_stability(self, cluster_labels: Any, cluster_metrics: dict[str, Any]) -> dict[str, float]:
        """Calculate cluster stability metrics."""
        try:
            stability = {
                "silhouette_score": cluster_metrics.get("silhouette_score", 0),
                "cluster_balance": cluster_metrics.get("cluster_balance", 0),
                "mean_distance_to_center": cluster_metrics.get("mean_distance_to_center", 0)
            }
            return stability
        except Exception:
            return {}

    def _calculate_cluster_intensity(self, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Calculate cluster intensity scores."""
        try:
            intensity = np.zeros(len(cluster_labels))
            for cluster_id, char in composite_analysis.get("cluster_characteristics", {}).items():
                cluster_mask = cluster_labels == cluster_id
                intensity[cluster_mask] = char.get("percentage", 0) / 100
            return intensity
        except Exception:
            return np.zeros(len(cluster_labels))

    def _calculate_cluster_stability_scores(self, cluster_labels: Any, composite_analysis: dict[str, Any]) -> Any:
        """Calculate cluster stability scores."""
        try:
            stability = np.ones(len(cluster_labels))  # Default stability score
            # This could be enhanced with more sophisticated stability calculations
            return stability
        except Exception:
            return np.ones(len(cluster_labels))

    def _calculate_volatility_intensity(self, features: Any, cluster_mask: Any) -> float:
        """Calculate volatility intensity for a cluster."""
        try:
            if "volatility_20" in features.columns:
                return features.loc[cluster_mask, "volatility_20"].mean()
            return 0.0
        except Exception:
            return 0.0

    def _calculate_momentum_intensity(self, features: Any, cluster_mask: Any) -> float:
        """Calculate momentum intensity for a cluster."""
        try:
            if "price_momentum_10" in features.columns:
                return abs(features.loc[cluster_mask, "price_momentum_10"].mean())
            return 0.0
        except Exception:
            return 0.0

    def _calculate_volume_intensity(self, features: Any, cluster_mask: Any) -> float:
        """Calculate volume intensity for a cluster."""
        try:
            if "volume_ratio_10" in features.columns:
                return features.loc[cluster_mask, "volume_ratio_10"].mean()
            return 1.0
        except Exception:
            return 1.0

    # === REPORT GENERATION METHODS ===

    def _generate_cluster_quality_report(self, cluster_metrics: dict[str, Any]) -> str:
        """Generate cluster quality report."""
        try:
            report = []
            report.append("# Cluster Quality Analysis Report")
            report.append("")
            report.append(f"## Quality Metrics")
            report.append(f"- **Silhouette Score**: {cluster_metrics.get('silhouette_score', 0):.4f}")
            report.append(f"- **Calinski-Harabasz Score**: {cluster_metrics.get('calinski_harabasz_score', 0):.2f}")
            report.append(f"- **Davies-Bouldin Score**: {cluster_metrics.get('davies_bouldin_score', 0):.4f}")
            report.append(f"- **Inertia**: {cluster_metrics.get('inertia', 0):.2f}")
            report.append("")
            report.append(f"## Cluster Distribution")
            report.append(f"- **Min Cluster Size**: {cluster_metrics.get('min_cluster_size', 0)}")
            report.append(f"- **Max Cluster Size**: {cluster_metrics.get('max_cluster_size', 0)}")
            report.append(f"- **Mean Cluster Size**: {cluster_metrics.get('mean_cluster_size', 0):.1f}")
            report.append(f"- **Cluster Balance**: {cluster_metrics.get('cluster_balance', 0):.4f}")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating cluster quality report: {e}"

    def _generate_cluster_characteristics_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate cluster characteristics report."""
        try:
            report = []
            report.append("# Cluster Characteristics Report")
            report.append("")
            
            for cluster_id, char in composite_analysis.get("cluster_characteristics", {}).items():
                report.append(f"## Cluster {cluster_id}")
                report.append(f"- **Size**: {char.get('size', 0)} ({char.get('percentage', 0):.1f}%)")
                report.append(f"- **Dominant HMM State**: {char.get('dominant_hmm_state', 'unknown')}")
                report.append(f"- **Market Condition**: {composite_analysis.get('market_conditions', {}).get(cluster_id, 'unknown')}")
                report.append("")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating cluster characteristics report: {e}"

    def _generate_market_conditions_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate market conditions report."""
        try:
            report = []
            report.append("# Market Conditions Report")
            report.append("")
            
            market_conditions = composite_analysis.get("market_conditions", {})
            condition_counts = {}
            
            for condition in market_conditions.values():
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            for condition, count in condition_counts.items():
                report.append(f"- **{condition}**: {count} clusters")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating market conditions report: {e}"

    def _generate_feature_importance_report(self, composite_analysis: dict[str, Any]) -> str:
        """Generate feature importance report."""
        try:
            report = []
            report.append("# Feature Importance Report")
            report.append("")
            
            feature_importance = composite_analysis.get("feature_importance", {})
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            report.append("## Top 10 Most Important Features")
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                report.append(f"{i}. **{feature}**: {importance:.4f}")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating feature importance report: {e}"

    def _generate_hmm_state_analysis_report(self, hmm_states: Any, composite_analysis: dict[str, Any]) -> str:
        """Generate HMM state analysis report."""
        try:
            report = []
            report.append("# HMM State Analysis Report")
            report.append("")
            
            hmm_distribution = composite_analysis.get("hmm_state_distribution", {})
            total_states = sum(hmm_distribution.values())
            
            report.append("## HMM State Distribution")
            for state, count in hmm_distribution.items():
                percentage = (count / total_states * 100) if total_states > 0 else 0
                report.append(f"- **State {state}**: {count} ({percentage:.1f}%)")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating HMM state analysis report: {e}"

    def _generate_temporal_analysis_report(self, cluster_labels: Any, features: Any) -> str:
        """Generate temporal analysis report."""
        try:
            report = []
            report.append("# Temporal Analysis Report")
            report.append("")
            
            # Calculate cluster transitions
            transitions = 0
            for i in range(1, len(cluster_labels)):
                if cluster_labels[i] != cluster_labels[i-1]:
                    transitions += 1
            
            report.append(f"## Cluster Transitions")
            report.append(f"- **Total Transitions**: {transitions}")
            report.append(f"- **Transition Rate**: {transitions / len(cluster_labels) * 100:.2f}%")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating temporal analysis report: {e}"

    def _generate_recommendations_report(self, cluster_metrics: dict[str, Any], composite_analysis: dict[str, Any]) -> str:
        """Generate recommendations report."""
        try:
            report = []
            report.append("# Recommendations Report")
            report.append("")
            
            # Analyze cluster quality
            silhouette = cluster_metrics.get("silhouette_score", 0)
            if silhouette < 0.2:
                report.append("- **Low Silhouette Score**: Consider reducing number of clusters or improving feature engineering")
            elif silhouette > 0.5:
                report.append("- **Good Silhouette Score**: Clusters are well-separated")
            
            # Analyze cluster balance
            balance = cluster_metrics.get("cluster_balance", 0)
            if balance > 0.5:
                report.append("- **Unbalanced Clusters**: Consider adjusting clustering parameters for better balance")
            
            # Analyze feature importance
            feature_importance = composite_analysis.get("feature_importance", {})
            if feature_importance:
                top_feature = max(feature_importance.items(), key=lambda x: x[1])
                report.append(f"- **Most Important Feature**: {top_feature[0]} (importance: {top_feature[1]:.4f})")
            
            return "\n".join(report)
        except Exception as e:
            return f"Error generating recommendations report: {e}"

    # === ENHANCED HMM REGIME MANAGEMENT METHODS ===
    
    async def _perform_enhanced_hmm_regime_discovery(self, training_input: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Perform enhanced HMM regime discovery with comprehensive analysis.
        Improved cluster generation and regime change prediction capabilities.
        
        Args:
            training_input: Training input parameters
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: Enhanced regime discovery results
        """
        try:
            self.logger.info("🧠 Performing enhanced HMM regime discovery with improved clustering...")
            
            # Update regime management state
            self.regime_management_state["last_regime_analysis"] = pd.Timestamp.now()
            self.regime_management_state["regime_analysis_count"] += 1
            
            # Step 1: Prepare comprehensive features for HMM
            features = await self._prepare_enhanced_hmm_features(market_data)
            
            # Step 2: Train enhanced HMM model with optimal parameters
            hmm_model = await self._train_enhanced_hmm_model(features)
            
            # Step 3: Perform advanced clustering analysis (20-cluster composite approach)
            clustering_results = await self._perform_advanced_regime_clustering(features)
            
            # Step 4: Analyze regime transitions with prediction capabilities
            transition_analysis = await self._analyze_enhanced_regime_transitions(hmm_model, features)
            
            # Step 5: Calculate regime stability and persistence
            stability_analysis = await self._analyze_enhanced_regime_stability(hmm_model, features)
            
            # Step 6: Generate regime change predictions
            prediction_results = await self._generate_enhanced_regime_predictions(hmm_model, features)
            
            # Step 7: Calculate comprehensive quality metrics
            quality_metrics = await self._calculate_enhanced_regime_quality_metrics(
                hmm_model, clustering_results, transition_analysis, stability_analysis, features
            )
            
            # Step 8: Eliminate redundancy and optimize clusters
            redundancy_metrics = await self._eliminate_enhanced_regime_redundancy(
                hmm_model, clustering_results, features
            )
            
            # Step 9: Generate regime change detection model
            regime_change_model = await self._build_regime_change_detection_model(
                hmm_model, clustering_results, transition_analysis
            )
            
            # Create comprehensive results with enhanced capabilities
            results = {
                "success": True,
                "hmm_model": hmm_model,
                "clustering_results": clustering_results,
                "transition_analysis": transition_analysis,
                "stability_analysis": stability_analysis,
                "prediction_results": prediction_results,
                "quality_metrics": quality_metrics,
                "redundancy_metrics": redundancy_metrics,
                "regime_change_model": regime_change_model,
                "regime_states": await self._extract_enhanced_regime_states(hmm_model, features),
                "regime_transitions": transition_analysis.get("transitions", []),
                "regime_change_predictions": prediction_results.get("change_predictions", []),
                "metrics": {
                    **quality_metrics,
                    **redundancy_metrics,
                    "regime_count": len(clustering_results.get("clusters", [])),
                    "transition_count": len(transition_analysis.get("transitions", [])),
                    "stability_score": stability_analysis.get("overall_stability", 0.0),
                    "prediction_accuracy": prediction_results.get("accuracy", 0.0),
                    "cluster_quality": clustering_results.get("cluster_quality", 0.0),
                    "regime_change_detection_ready": True
                },
                "integration_status": {
                    "feature_engineering_ready": True,
                    "analyst_component_ready": True,
                    "regime_change_prediction_ready": True,
                    "quality_control_passed": quality_metrics.get("overall_quality", 0) > 0.6,
                    "redundancy_eliminated": redundancy_metrics.get("redundancy_score", 0) < 0.3
                }
            }
            
            # Update enhanced state
            self.enhanced_hmm_capabilities["hmm_model"] = hmm_model
            self.enhanced_hmm_capabilities["kmeans_model"] = clustering_results.get("kmeans_model")
            self.enhanced_hmm_capabilities["transition_model"] = transition_analysis.get("transition_model")
            self.enhanced_hmm_capabilities["regime_change_model"] = regime_change_model
            self.regime_management_state["regime_quality_scores"] = quality_metrics
            self.regime_management_state["regime_redundancy_metrics"] = redundancy_metrics
            self.regime_management_state["regime_change_detection"] = prediction_results.get("change_predictions", [])
            
            self.logger.info("✅ Enhanced HMM regime discovery with improved clustering completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced HMM regime discovery: {e}")
            return {"success": False, "error": str(e)}

    # Enhanced HMM Regime Management Methods
    
    async def _train_enhanced_hmm_model(self, features: pd.DataFrame) -> Any:
        """Train enhanced HMM model with improved parameters."""
        try:
            self.logger.info("🧠 Training enhanced HMM model...")
            
            # Prepare data
            feature_matrix = features.dropna().values
            if len(feature_matrix) < 100:
                raise ValueError("Insufficient data for HMM training")
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # Train HMM with multiple components
            from hmmlearn import hmm
            
            # Try different numbers of states
            best_score = -np.inf
            best_model = None
            best_n_states = 5
            
            for n_states in [3, 5, 7, 10]:
                try:
                    model = hmm.GaussianHMM(
                        n_components=n_states,
                        covariance_type="full",
                        n_iter=1000,
                        random_state=42,
                        init_params="stmcw",
                        params="stmcw"
                    )
                    model.fit(normalized_features)
                    score = model.score(normalized_features)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_n_states = n_states
                        
                except Exception as e:
                    self.logger.warning(f"Failed to train HMM with {n_states} states: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("Failed to train any HMM model")
            
            self.logger.info(f"✅ Enhanced HMM model trained with {best_n_states} states")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error training enhanced HMM model: {e}")
            raise

    async def _perform_regime_clustering(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Perform regime clustering using KMeans and other methods."""
        try:
            self.logger.info("🔍 Performing regime clustering...")
            
            # Prepare data
            feature_matrix = features.dropna().values
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(normalized_features)
            
            # Gaussian Mixture clustering
            gmm = GaussianMixture(n_components=5, random_state=42)
            gmm_labels = gmm.fit_predict(normalized_features)
            
            # Calculate clustering quality metrics
            kmeans_silhouette = silhouette_score(normalized_features, kmeans_labels)
            gmm_silhouette = silhouette_score(normalized_features, gmm_labels)
            
            # Choose best clustering
            if kmeans_silhouette > gmm_silhouette:
                best_clustering = kmeans
                best_labels = kmeans_labels
                best_score = kmeans_silhouette
            else:
                best_clustering = gmm
                best_labels = gmm_labels
                best_score = gmm_silhouette
            
            # Create regime clusters
            clusters = []
            for i in range(best_clustering.n_clusters):
                cluster_mask = best_labels == i
                cluster_data = normalized_features[cluster_mask]
                
                cluster = RegimeCluster(
                    cluster_id=i,
                    center=best_clustering.cluster_centers_[i] if hasattr(best_clustering, 'cluster_centers_') else best_clustering.means_[i],
                    regime_states=[],  # Will be populated later
                    cluster_quality=best_score,
                    cluster_stability=np.std(cluster_data),
                    cluster_features=self._extract_cluster_features(cluster_data, features.iloc[cluster_mask])
                )
                clusters.append(cluster)
            
            results = {
                "kmeans_model": kmeans,
                "gmm_model": gmm,
                "best_clustering": best_clustering,
                "best_labels": best_labels,
                "clusters": clusters,
                "kmeans_silhouette": kmeans_silhouette,
                "gmm_silhouette": gmm_silhouette,
                "best_score": best_score
            }
            
            self.logger.info(f"✅ Regime clustering completed with {len(clusters)} clusters")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in regime clustering: {e}")
            return {"clusters": [], "best_score": 0.0}

    async def _analyze_regime_transitions(self, hmm_model: Any, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime transitions and train transition prediction model."""
        try:
            self.logger.info("🔄 Analyzing regime transitions...")
            
            # Get HMM states
            feature_matrix = features.dropna().values
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            states = hmm_model.predict(normalized_features)
            
            # Create transition matrix
            n_states = hmm_model.n_components
            transition_matrix = np.zeros((n_states, n_states))
            
            for i in range(len(states) - 1):
                from_state = states[i]
                to_state = states[i + 1]
                transition_matrix[from_state, to_state] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
            # Train transition prediction model
            transition_features = []
            transition_labels = []
            
            for i in range(len(states) - 1):
                # Create features for transition prediction
                current_features = normalized_features[i]
                next_features = normalized_features[i + 1]
                feature_diff = next_features - current_features
                
                transition_feature = np.concatenate([current_features, feature_diff])
                transition_features.append(transition_feature)
                transition_labels.append(states[i + 1])
            
            if len(transition_features) > 10:
                transition_model = RandomForestClassifier(n_estimators=100, random_state=42)
                transition_model.fit(transition_features, transition_labels)
                
                # Calculate transition probabilities
                transition_probabilities = {}
                for from_state in range(n_states):
                    for to_state in range(n_states):
                        key = f"{from_state}_to_{to_state}"
                        transition_probabilities[key] = transition_matrix[from_state, to_state]
            else:
                transition_model = None
                transition_probabilities = {}
            
            # Detect regime transitions
            transitions = []
            for i in range(len(states) - 1):
                if states[i] != states[i + 1]:
                    transition = RegimeTransition(
                        from_regime=states[i],
                        to_regime=states[i + 1],
                        probability=transition_matrix[states[i], states[i + 1]],
                        timestamp=features.index[i + 1] if hasattr(features, 'index') else pd.Timestamp.now(),
                        trigger_features=dict(zip(features.columns, normalized_features[i])),
                        confidence=transition_matrix[states[i], states[i + 1]],
                        transition_strength=1.0,
                        transition_duration=1
                    )
                    transitions.append(transition)
            
            results = {
                "transition_matrix": transition_matrix,
                "transition_model": transition_model,
                "transition_probabilities": transition_probabilities,
                "transitions": transitions,
                "transition_count": len(transitions)
            }
            
            self.logger.info(f"✅ Regime transition analysis completed: {len(transitions)} transitions detected")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in regime transition analysis: {e}")
            return {"transitions": [], "transition_count": 0}

    async def _analyze_regime_stability(self, hmm_model: Any, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime stability and persistence."""
        try:
            self.logger.info("📊 Analyzing regime stability...")
            
            # Get HMM states
            feature_matrix = features.dropna().values
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            states = hmm_model.predict(normalized_features)
            
            # Calculate stability metrics for each regime
            stability_metrics = {}
            n_states = hmm_model.n_components
            
            for state in range(n_states):
                state_mask = states == state
                state_durations = []
                current_duration = 0
                
                for is_state in state_mask:
                    if is_state:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            state_durations.append(current_duration)
                            current_duration = 0
                
                if current_duration > 0:
                    state_durations.append(current_duration)
                
                if state_durations:
                    stability_metrics[state] = {
                        "avg_duration": np.mean(state_durations),
                        "max_duration": np.max(state_durations),
                        "duration_std": np.std(state_durations),
                        "persistence": len(state_durations) / len(states),
                        "stability_score": 1.0 / (1.0 + np.std(state_durations))
                    }
                else:
                    stability_metrics[state] = {
                        "avg_duration": 0,
                        "max_duration": 0,
                        "duration_std": 0,
                        "persistence": 0,
                        "stability_score": 0
                    }
            
            # Calculate overall stability
            overall_stability = np.mean([metrics["stability_score"] for metrics in stability_metrics.values()])
            
            results = {
                "stability_metrics": stability_metrics,
                "overall_stability": overall_stability,
                "regime_persistence": {state: metrics["persistence"] for state, metrics in stability_metrics.items()}
            }
            
            self.logger.info(f"✅ Regime stability analysis completed: overall stability = {overall_stability:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in regime stability analysis: {e}")
            return {"overall_stability": 0.0, "stability_metrics": {}}

    async def _generate_regime_predictions(self, hmm_model: Any, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate regime predictions and forecasting."""
        try:
            self.logger.info("🔮 Generating regime predictions...")
            
            # Get current state
            feature_matrix = features.dropna().values
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            current_state = hmm_model.predict(normalized_features)[-1]
            
            # Predict next state probabilities
            next_state_probs = hmm_model.transmat_[current_state]
            
            # Generate short-term predictions
            predictions = []
            current_state_probs = next_state_probs.copy()
            
            for step in range(5):  # Predict next 5 steps
                next_state = np.argmax(current_state_probs)
                confidence = np.max(current_state_probs)
                
                prediction = {
                    "step": step + 1,
                    "predicted_state": next_state,
                    "confidence": confidence,
                    "state_probabilities": current_state_probs.copy()
                }
                predictions.append(prediction)
                
                # Update probabilities for next step
                current_state_probs = hmm_model.transmat_[next_state]
            
            results = {
                "current_state": current_state,
                "next_state_probabilities": next_state_probs,
                "predictions": predictions,
                "prediction_horizon": 5
            }
            
            self.logger.info(f"✅ Regime predictions generated: current state = {current_state}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating regime predictions: {e}")
            return {"current_state": 0, "predictions": []}

    def _calculate_regime_quality_metrics(self, hmm_model: Any, clustering_results: Dict[str, Any], 
                                        transition_analysis: Dict[str, Any], stability_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive regime quality metrics."""
        try:
            metrics = {}
            
            # HMM model quality
            if hasattr(hmm_model, 'score'):
                feature_matrix = self._get_feature_matrix()
                if feature_matrix is not None:
                    metrics["hmm_log_likelihood"] = hmm_model.score(feature_matrix)
                    metrics["hmm_bic"] = hmm_model.bic(feature_matrix)
                    metrics["hmm_aic"] = hmm_model.aic(feature_matrix)
            
            # Clustering quality
            clustering_score = clustering_results.get("best_score", 0.0)
            metrics["clustering_quality"] = clustering_score
            metrics["cluster_count"] = len(clustering_results.get("clusters", []))
            
            # Transition quality
            transition_count = transition_analysis.get("transition_count", 0)
            metrics["transition_count"] = transition_count
            metrics["transition_diversity"] = len(set([t.to_regime for t in transition_analysis.get("transitions", [])]))
            
            # Stability quality
            overall_stability = stability_analysis.get("overall_stability", 0.0)
            metrics["overall_stability"] = overall_stability
            metrics["regime_persistence"] = np.mean(list(stability_analysis.get("regime_persistence", {}).values()))
            
            # Composite quality score
            quality_score = (
                metrics.get("clustering_quality", 0.0) * 0.3 +
                metrics.get("overall_stability", 0.0) * 0.3 +
                (1.0 - metrics.get("regime_persistence", 0.0)) * 0.2 +
                min(metrics.get("transition_count", 0) / 10.0, 1.0) * 0.2
            )
            metrics["composite_quality_score"] = quality_score
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime quality metrics: {e}")
            return {"composite_quality_score": 0.0}

    def _eliminate_regime_redundancy(self, hmm_model: Any, clustering_results: Dict[str, Any], 
                                   features: pd.DataFrame) -> Dict[str, float]:
        """Eliminate redundant regimes and calculate redundancy metrics."""
        try:
            metrics = {}
            
            # Analyze cluster overlap
            clusters = clustering_results.get("clusters", [])
            if len(clusters) > 1:
                cluster_centers = np.array([cluster.center for cluster in clusters])
                
                # Calculate pairwise distances between cluster centers
                distances = []
                for i in range(len(cluster_centers)):
                    for j in range(i + 1, len(cluster_centers)):
                        distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                        distances.append(distance)
                
                # Calculate redundancy metrics
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)
                distance_std = np.std(distances)
                
                metrics["avg_cluster_distance"] = avg_distance
                metrics["min_cluster_distance"] = min_distance
                metrics["cluster_distance_std"] = distance_std
                metrics["cluster_redundancy_score"] = 1.0 / (1.0 + avg_distance)
                
                # Identify redundant clusters
                redundant_clusters = []
                for i in range(len(cluster_centers)):
                    for j in range(i + 1, len(cluster_centers)):
                        distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                        if distance < avg_distance * 0.5:  # Clusters too close
                            redundant_clusters.append((i, j, distance))
                
                metrics["redundant_cluster_pairs"] = len(redundant_clusters)
                metrics["redundancy_ratio"] = len(redundant_clusters) / (len(clusters) * (len(clusters) - 1) / 2)
            else:
                metrics["cluster_redundancy_score"] = 0.0
                metrics["redundancy_ratio"] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating regime redundancy: {e}")
            return {"cluster_redundancy_score": 0.0, "redundancy_ratio": 0.0}

    def _extract_regime_states(self, hmm_model: Any, features: pd.DataFrame) -> List[RegimeState]:
        """Extract regime states from HMM model."""
        try:
            states = []
            feature_matrix = features.dropna().values
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            hmm_states = hmm_model.predict(normalized_features)
            
            for i, state in enumerate(hmm_states):
                regime_state = RegimeState(
                    regime_id=state,
                    regime_type=self._map_state_to_regime_type(state),
                    confidence=hmm_model.predict_proba(normalized_features[i:i+1])[0].max(),
                    duration=1,
                    volatility=features.iloc[i]['volatility'] if 'volatility' in features.columns else 0.0,
                    momentum=features.iloc[i]['price_momentum'] if 'price_momentum' in features.columns else 0.0,
                    volume_profile=features.iloc[i]['volume_ratio'] if 'volume_ratio' in features.columns else 0.0,
                    timestamp=features.index[i] if hasattr(features, 'index') else pd.Timestamp.now(),
                    features=features.iloc[i].to_dict(),
                    transition_probability=0.0,
                    stability_score=0.0,
                    regime_quality=0.0,
                    regime_persistence=0.0,
                    regime_complexity=0.0
                )
                states.append(regime_state)
            
            return states
            
        except Exception as e:
            self.logger.error(f"Error extracting regime states: {e}")
            return []

    def _extract_cluster_features(self, cluster_data: np.ndarray, cluster_features: pd.DataFrame) -> Dict[str, float]:
        """Extract features for a cluster."""
        try:
            features = {}
            
            # Calculate mean and std for each feature
            for col in cluster_features.columns:
                features[f"{col}_mean"] = cluster_features[col].mean()
                features[f"{col}_std"] = cluster_features[col].std()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting cluster features: {e}")
            return {}

    def _map_state_to_regime_type(self, state: int) -> RegimeType:
        """Map HMM state to regime type."""
        # Simple mapping based on state number
        regime_types = [
            RegimeType.BULL,
            RegimeType.BEAR,
            RegimeType.SIDEWAYS,
            RegimeType.VOLATILE,
            RegimeType.TRENDING
        ]
        return regime_types[state % len(regime_types)]

    def _get_feature_matrix(self) -> Optional[np.ndarray]:
        """Get feature matrix for HMM scoring."""
        try:
            # This would need to be implemented based on the current feature data
            return None
        except Exception as e:
            self.logger.error(f"Error getting feature matrix: {e}")
            return None

    async def _prepare_hmm_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for HMM analysis."""
        try:
            self.logger.info("🔧 Preparing HMM features...")
            
            features = pd.DataFrame()
            
            # Price-based features
            features['returns'] = market_data['close'].pct_change()
            features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
            features['price_momentum'] = market_data['close'] / market_data['close'].shift(5) - 1
            features['price_acceleration'] = features['price_momentum'].diff()
            
            # Volatility features
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['volatility_change'] = features['volatility'].diff()
            features['high_low_ratio'] = (market_data['high'] - market_data['low']) / market_data['close']
            
            # Volume features
            features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(window=20).mean()
            features['volume_momentum'] = market_data['volume'].pct_change()
            features['volume_volatility'] = market_data['volume'].rolling(window=20).std() / market_data['volume'].rolling(window=20).mean()
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(market_data['close'])
            features['macd'] = self._calculate_macd(market_data['close'])
            features['bollinger_position'] = self._calculate_bollinger_position(market_data)
            
            # Trend features
            features['trend_strength'] = self._calculate_trend_strength(market_data)
            features['trend_direction'] = np.where(features['returns'] > 0, 1, -1)
            features['trend_consistency'] = features['trend_direction'].rolling(window=10).mean()
            
            # Remove NaN values
            features = features.dropna()
            
            self.logger.info(f"✅ HMM features prepared: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing HMM features: {e}")
            return pd.DataFrame()

    async def _train_enhanced_hmm_model(self, features: pd.DataFrame) -> Any:
        """Train enhanced HMM model with multiple components."""
        try:
            self.logger.info("🎯 Training enhanced HMM model...")
            
            # Import hmmlearn
            try:
                from hmmlearn import hmm
            except ImportError:
                self.logger.error("hmmlearn not available, using fallback regime detection")
                return self._fallback_regime_detection(features)
            
            # Prepare data for HMM
            X = features.values
            
            # Train HMM with multiple states
            n_states = 5  # Can be made configurable
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
            model.fit(X)
            
            # Get state sequences
            state_sequence = model.predict(X)
            
            # Store model and results
            hmm_results = {
                "model": model,
                "state_sequence": state_sequence,
                "state_probabilities": model.predict_proba(X),
                "n_states": n_states,
                "features": features.columns.tolist()
            }
            
            self.logger.info(f"✅ Enhanced HMM model trained with {n_states} states")
            return hmm_results
            
        except Exception as e:
            self.logger.error(f"Error training enhanced HMM model: {e}")
            return self._fallback_regime_detection(features)

    async def _perform_regime_clustering(self, features: pd.DataFrame) -> dict[str, Any]:
        """Perform regime clustering using KMeans."""
        try:
            self.logger.info("🔍 Performing regime clustering...")
            
            # Prepare data for clustering
            X = features.values
            
            # Perform KMeans clustering
            n_clusters = 5  # Can be made configurable
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate clustering metrics
            silhouette = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            calinski = calinski_harabasz_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            davies = davies_bouldin_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
            
            # Analyze cluster characteristics
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = features[cluster_mask]
                
                cluster_analysis[f"cluster_{i}"] = {
                    "size": cluster_mask.sum(),
                    "percentage": cluster_mask.sum() / len(cluster_labels) * 100,
                    "avg_volatility": cluster_data['volatility'].mean() if 'volatility' in cluster_data.columns else 0,
                    "avg_returns": cluster_data['returns'].mean() if 'returns' in cluster_data.columns else 0,
                    "avg_volume_ratio": cluster_data['volume_ratio'].mean() if 'volume_ratio' in cluster_data.columns else 0
                }
            
            results = {
                "kmeans_model": kmeans,
                "cluster_labels": cluster_labels,
                "cluster_centers": kmeans.cluster_centers_,
                "cluster_analysis": cluster_analysis,
                "metrics": {
                    "silhouette_score": silhouette,
                    "calinski_harabasz_score": calinski,
                    "davies_bouldin_score": davies
                }
            }
            
            self.logger.info(f"✅ Regime clustering completed: {n_clusters} clusters, silhouette={silhouette:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing regime clustering: {e}")
            return {}

    async def _analyze_regime_transitions(self, hmm_model: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
        """Analyze regime transitions and train transition model."""
        try:
            self.logger.info("🔄 Analyzing regime transitions...")
            
            state_sequence = hmm_model.get("state_sequence", [])
            if not state_sequence:
                return {"transitions": [], "transition_model": None}
            
            # Calculate transition matrix
            n_states = hmm_model.get("n_states", 5)
            transition_matrix = np.zeros((n_states, n_states))
            
            for i in range(1, len(state_sequence)):
                from_state = state_sequence[i-1]
                to_state = state_sequence[i]
                transition_matrix[from_state, to_state] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
            # Identify transitions
            transitions = []
            for i in range(1, len(state_sequence)):
                if state_sequence[i] != state_sequence[i-1]:
                    transition = RegimeTransition(
                        from_regime=state_sequence[i-1],
                        to_regime=state_sequence[i],
                        probability=transition_matrix[state_sequence[i-1], state_sequence[i]],
                        timestamp=features.index[i] if hasattr(features, 'index') else pd.Timestamp.now(),
                        trigger_features=self._extract_transition_features(features, i),
                        confidence=0.7  # Can be improved with more sophisticated analysis
                    )
                    transitions.append(transition)
            
            # Train transition prediction model
            transition_model = self._train_transition_prediction_model(features, state_sequence)
            
            results = {
                "transitions": transitions,
                "transition_matrix": transition_matrix,
                "transition_model": transition_model,
                "transition_count": len(transitions),
                "avg_transition_probability": np.mean(transition_matrix[transition_matrix > 0]) if np.any(transition_matrix > 0) else 0
            }
            
            self.logger.info(f"✅ Regime transition analysis completed: {len(transitions)} transitions")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {e}")
            return {"transitions": [], "transition_model": None}

    async def _analyze_regime_stability(self, hmm_model: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
        """Analyze regime stability and persistence."""
        try:
            self.logger.info("📊 Analyzing regime stability...")
            
            state_sequence = hmm_model.get("state_sequence", [])
            if not state_sequence:
                return {"overall_stability": 0.0, "state_stability": {}}
            
            # Calculate state persistence
            state_persistence = {}
            current_state = state_sequence[0]
            persistence_count = 1
            
            for i in range(1, len(state_sequence)):
                if state_sequence[i] == current_state:
                    persistence_count += 1
                else:
                    # Record persistence for previous state
                    if current_state not in state_persistence:
                        state_persistence[current_state] = []
                    state_persistence[current_state].append(persistence_count)
                    
                    # Start new state
                    current_state = state_sequence[i]
                    persistence_count = 1
            
            # Add final state persistence
            if current_state not in state_persistence:
                state_persistence[current_state] = []
            state_persistence[current_state].append(persistence_count)
            
            # Calculate stability metrics
            stability_metrics = {}
            overall_stability = 0.0
            
            for state, persistences in state_persistence.items():
                avg_persistence = np.mean(persistences)
                persistence_variance = np.var(persistences) if len(persistences) > 1 else 0
                stability_score = avg_persistence / (1 + persistence_variance)
                
                stability_metrics[state] = {
                    "avg_persistence": avg_persistence,
                    "persistence_variance": persistence_variance,
                    "stability_score": stability_score,
                    "persistence_count": len(persistences)
                }
                
                overall_stability += stability_score
            
            overall_stability /= len(stability_metrics) if stability_metrics else 1
            
            results = {
                "overall_stability": overall_stability,
                "state_stability": stability_metrics,
                "state_persistence": state_persistence
            }
            
            self.logger.info(f"✅ Regime stability analysis completed: overall stability={overall_stability:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime stability: {e}")
            return {"overall_stability": 0.0, "state_stability": {}}

    async def _generate_regime_predictions(self, hmm_model: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
        """Generate regime predictions for future periods."""
        try:
            self.logger.info("🔮 Generating regime predictions...")
            
            model = hmm_model.get("model")
            if model is None:
                return {"predictions": [], "prediction_model": None}
            
            # Get current state probabilities
            current_features = features.iloc[-1:].values
            state_probabilities = model.predict_proba(current_features)[0]
            
            # Predict next state
            next_state_probabilities = model.transmat_.dot(state_probabilities)
            predicted_state = np.argmax(next_state_probabilities)
            
            # Generate predictions for multiple periods
            predictions = []
            current_probs = state_probabilities.copy()
            
            for period in range(1, 6):  # Predict next 5 periods
                next_probs = model.transmat_.dot(current_probs)
                predicted_state = np.argmax(next_probs)
                confidence = np.max(next_probs)
                
                prediction = {
                    "period": period,
                    "predicted_state": predicted_state,
                    "confidence": confidence,
                    "state_probabilities": next_probs.tolist(),
                    "timestamp": pd.Timestamp.now()
                }
                predictions.append(prediction)
                
                current_probs = next_probs
            
            results = {
                "predictions": predictions,
                "current_state": np.argmax(state_probabilities),
                "current_state_probabilities": state_probabilities.tolist(),
                "prediction_model": model
            }
            
            self.logger.info(f"✅ Regime predictions generated: {len(predictions)} periods")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating regime predictions: {e}")
            return {"predictions": [], "prediction_model": None}

    def _calculate_regime_quality_metrics(self, hmm_model: dict[str, Any], clustering_results: dict[str, Any], 
                                        transition_analysis: dict[str, Any], stability_analysis: dict[str, Any]) -> dict[str, float]:
        """Calculate comprehensive quality metrics for regime analysis."""
        try:
            metrics = {}
            
            # HMM model quality
            model = hmm_model.get("model")
            if model:
                metrics["hmm_score"] = model.score(hmm_model.get("features", pd.DataFrame()).values)
                metrics["hmm_convergence"] = 1.0 if model.converged_ else 0.0
            else:
                metrics["hmm_score"] = 0.0
                metrics["hmm_convergence"] = 0.0
            
            # Clustering quality
            clustering_metrics = clustering_results.get("metrics", {})
            metrics["silhouette_score"] = clustering_metrics.get("silhouette_score", 0.0)
            metrics["calinski_harabasz_score"] = clustering_metrics.get("calinski_harabasz_score", 0.0)
            metrics["davies_bouldin_score"] = clustering_metrics.get("davies_bouldin_score", 0.0)
            
            # Transition quality
            transition_metrics = transition_analysis.get("transition_matrix", np.array([]))
            if transition_metrics.size > 0:
                metrics["transition_entropy"] = -np.sum(transition_metrics * np.log(transition_metrics + 1e-10))
                metrics["transition_sparsity"] = 1.0 - np.count_nonzero(transition_metrics) / transition_metrics.size
            else:
                metrics["transition_entropy"] = 0.0
                metrics["transition_sparsity"] = 0.0
            
            # Stability quality
            metrics["overall_stability"] = stability_analysis.get("overall_stability", 0.0)
            
            # Overall quality score
            quality_factors = [
                metrics["hmm_convergence"],
                metrics["silhouette_score"],
                1.0 - metrics["davies_bouldin_score"] / 10,  # Normalize
                metrics["overall_stability"]
            ]
            metrics["overall_quality_score"] = sum(quality_factors) / len(quality_factors)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime quality metrics: {e}")
            return {}

    def _eliminate_regime_redundancy(self, hmm_model: dict[str, Any], clustering_results: dict[str, Any], 
                                   features: pd.DataFrame) -> dict[str, Any]:
        """Eliminate redundant regime states and features."""
        try:
            metrics = {}
            
            # Analyze feature redundancy
            feature_correlations = features.corr().abs()
            high_correlation_pairs = []
            
            for i in range(len(feature_correlations.columns)):
                for j in range(i+1, len(feature_correlations.columns)):
                    corr_value = feature_correlations.iloc[i, j]
                    if corr_value > 0.95:  # High correlation threshold
                        high_correlation_pairs.append((
                            feature_correlations.columns[i],
                            feature_correlations.columns[j],
                            corr_value
                        ))
            
            # Analyze cluster redundancy
            cluster_centers = clustering_results.get("cluster_centers", np.array([]))
            if cluster_centers.size > 0:
                # Calculate distances between cluster centers
                from scipy.spatial.distance import pdist, squareform
                distances = pdist(cluster_centers)
                distance_matrix = squareform(distances)
                
                # Find close clusters
                close_clusters = []
                for i in range(len(distance_matrix)):
                    for j in range(i+1, len(distance_matrix)):
                        if distance_matrix[i, j] < 0.1:  # Close cluster threshold
                            close_clusters.append((i, j, distance_matrix[i, j]))
            
            metrics["feature_redundancy_pairs"] = len(high_correlation_pairs)
            metrics["cluster_redundancy_pairs"] = len(close_clusters) if 'close_clusters' in locals() else 0
            metrics["overall_redundancy_score"] = (len(high_correlation_pairs) + metrics["cluster_redundancy_pairs"]) / 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating regime redundancy: {e}")
            return {}

    def _extract_regime_states(self, hmm_model: dict[str, Any], features: pd.DataFrame) -> List[RegimeState]:
        """Extract regime states from HMM model."""
        try:
            state_sequence = hmm_model.get("state_sequence", [])
            if not state_sequence:
                return []
            
            regime_states = []
            for i, state in enumerate(state_sequence):
                regime_state = RegimeState(
                    regime_id=state,
                    regime_type=self._classify_regime_type(state, features.iloc[i] if i < len(features) else None),
                    confidence=hmm_model.get("state_probabilities", np.array([]))[i, state] if i < len(hmm_model.get("state_probabilities", [])) else 0.5,
                    duration=1,  # Can be calculated more sophisticatedly
                    volatility=features.iloc[i]['volatility'] if i < len(features) and 'volatility' in features.columns else 0.0,
                    momentum=features.iloc[i]['price_momentum'] if i < len(features) and 'price_momentum' in features.columns else 0.0,
                    volume_profile=features.iloc[i]['volume_ratio'] if i < len(features) and 'volume_ratio' in features.columns else 0.0,
                    timestamp=features.index[i] if i < len(features) and hasattr(features, 'index') else pd.Timestamp.now(),
                    features=features.iloc[i].to_dict() if i < len(features) else {}
                )
                regime_states.append(regime_state)
            
            return regime_states
            
        except Exception as e:
            self.logger.error(f"Error extracting regime states: {e}")
            return []

    def _classify_regime_type(self, state: int, features: pd.Series) -> RegimeType:
        """Classify regime type based on state and features."""
        try:
            if features is None:
                return RegimeType.SIDEWAYS
            
            # Simple classification based on volatility and momentum
            volatility = features.get('volatility', 0.0)
            momentum = features.get('price_momentum', 0.0)
            
            if volatility > 0.02:  # High volatility
                return RegimeType.VOLATILE
            elif momentum > 0.01:  # Positive momentum
                return RegimeType.BULL
            elif momentum < -0.01:  # Negative momentum
                return RegimeType.BEAR
            elif abs(momentum) < 0.005:  # Low momentum
                return RegimeType.SIDEWAYS
            else:
                return RegimeType.TRENDING
                
        except Exception as e:
            self.logger.error(f"Error classifying regime type: {e}")
            return RegimeType.SIDEWAYS

    def _extract_transition_features(self, features: pd.DataFrame, index: int) -> dict[str, float]:
        """Extract features at transition point."""
        try:
            if index < len(features):
                return features.iloc[index].to_dict()
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error extracting transition features: {e}")
            return {}

    def _train_transition_prediction_model(self, features: pd.DataFrame, state_sequence: List[int]) -> Any:
        """Train model to predict regime transitions."""
        try:
            # Prepare training data
            X = features.values[:-1]  # Features up to transition
            y = (np.array(state_sequence[1:]) != np.array(state_sequence[:-1])).astype(int)  # Transition indicator
            
            # Train Random Forest classifier
            transition_model = RandomForestClassifier(n_estimators=100, random_state=42)
            transition_model.fit(X, y)
            
            return transition_model
            
        except Exception as e:
            self.logger.error(f"Error training transition prediction model: {e}")
            return None

    def _fallback_regime_detection(self, features: pd.DataFrame) -> dict[str, Any]:
        """Fallback regime detection when HMM is not available."""
        try:
            self.logger.info("🔄 Using fallback regime detection...")
            
            # Simple regime detection based on volatility and momentum
            volatility = features['volatility'] if 'volatility' in features.columns else features['returns'].rolling(20).std()
            momentum = features['price_momentum'] if 'price_momentum' in features.columns else features['returns'].rolling(5).mean()
            
            # Create simple regime labels
            regime_labels = []
            for i in range(len(features)):
                if volatility.iloc[i] > volatility.quantile(0.8):
                    regime_labels.append(0)  # Volatile
                elif momentum.iloc[i] > momentum.quantile(0.7):
                    regime_labels.append(1)  # Bull
                elif momentum.iloc[i] < momentum.quantile(0.3):
                    regime_labels.append(2)  # Bear
                else:
                    regime_labels.append(3)  # Sideways
            
            return {
                "model": None,
                "state_sequence": regime_labels,
                "state_probabilities": np.eye(4)[regime_labels],
                "n_states": 4,
                "features": features.columns.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback regime detection: {e}")
            return {"model": None, "state_sequence": [], "state_probabilities": [], "n_states": 0, "features": []}

    # === HELPER METHODS FOR ENHANCED HMM ANALYSIS ===
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception:
            return pd.Series([0] * len(prices))

    def _calculate_bollinger_position(self, market_data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        try:
            sma = market_data['close'].rolling(window=period).mean()
            std = market_data['close'].rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            position = (market_data['close'] - lower_band) / (upper_band - lower_band)
            return position
        except Exception:
            return pd.Series([0.5] * len(market_data))

    def _calculate_trend_strength(self, market_data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression."""
        try:
            trend_strength = pd.Series(index=market_data.index, dtype=float)
            
            for i in range(period, len(market_data)):
                window_data = market_data['close'].iloc[i-period:i]
                x = np.arange(len(window_data))
                y = window_data.values
                
                # Simple linear regression
                slope = np.cov(x, y)[0, 1] / np.var(x)
                trend_strength.iloc[i] = abs(slope) / market_data['close'].iloc[i]
            
            return trend_strength
        except Exception:
            return pd.Series([0] * len(market_data))


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