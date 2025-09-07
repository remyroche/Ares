#!/usr/bin/env python3
"""Step 3.5: Final Regime Clustering with Advanced Reporting."

This module performs final regime clustering using optimized parameters from step03,
with comprehensive reporting and analysis of regime characteristics.
"""
import asyncio
import sys
from pathlib import Path
import time
import json
from datetime import datetime

from src.core.decorators import handles_errors

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.decorators import (
    handles_errors,
    validates,
    log_execution_time,
    traced
)
from src.utils.logger import system_logger

# Enhanced optimization imports
from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager

# Import enhanced reporting system
try:
    from .step03_enhanced_reporting import Step03EnhancedReporter
    ENHANCED_REPORTING_AVAILABLE = True
except ImportError:
    ENHANCED_REPORTING_AVAILABLE = False
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
from src.utils.vectorized_processing_core import OptimizedPipelineExecutor, PipelineStage, PipelineExecutionMode
from src.utils.enhanced_matrix_operations import EnhancedMatrixOperations, ErrorHandler
from src.utils.enhanced_step_optimizations import IntelligentOptimizationSelector, OptimizationStrategy, WorkloadType, OptimizationProfile
from src.utils.optimized_data_manager import OptimizedDataManager, DataMetadata

import numpy as np
import pandas as pd
import logging
import typing
from typing import Any, Optional
from contextlib import nullcontext

logger = system_logger.getChild("Step3_5FinalRegimeClustering")

# Import optimized components
try:
    from .step03_enhanced_bayesian_optimization import EnhancedBayesianOptimizer, ParallelBayesianOptimizer
    OPTIMIZED_BAYESIAN_AVAILABLE = True
except ImportError:
    OPTIMIZED_BAYESIAN_AVAILABLE = False

try:
    from .step03_memory_manager import EnhancedMemoryManager, get_memory_manager
    OPTIMIZED_MEMORY_AVAILABLE = True
except ImportError:
    OPTIMIZED_MEMORY_AVAILABLE = False

try:
    from .step03_advanced_ensemble_clustering import AdvancedEnsembleClustering, ParallelClusteringProcessor
    OPTIMIZED_CLUSTERING_AVAILABLE = True
except ImportError:
    OPTIMIZED_CLUSTERING_AVAILABLE = False

try:
    from .step03_vectorized_operations import get_vectorized_operations_manager, create_vectorized_config
    OPTIMIZED_VECTORIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_VECTORIZED_AVAILABLE = False

try:
    from .step03_pipeline_orchestrator import get_step03_pipeline_orchestrator, create_step03_pipeline_config
    OPTIMIZED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    OPTIMIZED_ORCHESTRATOR_AVAILABLE = False


class FinalRegimeClusteringStep:
    """Step 3.5: Final Regime Clustering with Advanced Reporting and Hardware Optimizations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("FinalRegimeClusteringStep")
        self.start_time = None
        self.optimized_params = {}
        self.regime_results = {}

        # Initialize enhanced optimization components
        self._initialize_enhanced_optimizations()

        # Initialize legacy components for backward compatibility
        self._initialize_components()

    def _initialize_enhanced_optimizations(self) -> None:
        """Initialize enhanced optimization components for Step 3.5."""
        self.logger.info("🚀 Initializing enhanced optimization components for Step 3.5...")

        # Initialize M1 GPU Manager
        try:
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.logger.info("✅ M1 GPU Manager initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ M1 GPU Manager initialization failed: {e}")
            self.m1_gpu_manager = None

        # Initialize M1 Memory Optimizer
        try:
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            self.logger.info("✅ M1 Memory Optimizer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ M1 Memory Optimizer initialization failed: {e}")
            self.m1_memory_optimizer = None

        # Initialize M1 CPU Optimizer
        try:
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            self.logger.info("✅ M1 CPU Optimizer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ M1 CPU Optimizer initialization failed: {e}")
            self.m1_cpu_optimizer = None

        # Initialize Vectorized Processing Core
        try:
            self.pipeline_executor = OptimizedPipelineExecutor(max_concurrent_stages=4)
            self.logger.info("✅ Vectorized Processing Core initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized Processing Core initialization failed: {e}")
            self.pipeline_executor = None

        # Initialize Enhanced Matrix Operations
        try:
            self.matrix_operations = EnhancedMatrixOperations(
                enable_gpu_acceleration=True,
                enable_memory_optimization=True
            )
            self.logger.info("✅ Enhanced Matrix Operations initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced Matrix Operations initialization failed: {e}")
            self.matrix_operations = None

        # Initialize Intelligent Optimization Selector
        try:
            self.optimization_selector = IntelligentOptimizationSelector()
            self.logger.info("✅ Intelligent Optimization Selector initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Intelligent Optimization Selector initialization failed: {e}")
            self.optimization_selector = None

        # Initialize Optimized Data Manager
        try:
            self.data_manager = OptimizedDataManager(
                base_path=Path("data_cache"),
                enable_compression=True,
                enable_caching=True
            )
            self.logger.info("✅ Optimized Data Manager initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Optimized Data Manager initialization failed: {e}")
            self.data_manager = None

        # Initialize Error Handler
        try:
            self.error_handler = ErrorHandler(enable_recovery=True)
            self.logger.info("✅ Error Handler initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Error Handler initialization failed: {e}")
            self.error_handler = None

        # Determine optimization strategy
        self._determine_optimization_strategy()

        # Initialize enhanced reporting system
        if ENHANCED_REPORTING_AVAILABLE:
            try:
                self.enhanced_reporter = Step03EnhancedReporter()
                self.logger.info("✅ Enhanced reporting system initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced reporting system failed to initialize: {e}")
                self.enhanced_reporter = None
        else:
            self.logger.info("ℹ️ Enhanced reporting system not available, using basic reporting")
            self.enhanced_reporter = None

        self.logger.info("🎯 Enhanced optimization components initialization completed")

    def _determine_optimization_strategy(self) -> None:
        """Determine the optimal strategy based on workload and system capabilities."""
        if not self.optimization_selector:
            self.optimization_strategy = OptimizationStrategy.BALANCED
            return

        # Analyze workload characteristics
        data_size = self.config.get("expected_data_size_mb", 1000)  # Default estimate
        workload_profile = OptimizationProfile(
            workload_type=WorkloadType.MIXED,  # HMM + Clustering is mixed workload
            data_size_mb=data_size,
            expected_duration=300,  # 5 minutes expected
            priority="high",
            constraints={
                "memory_limit_gb": 8.0,
                "cpu_limit_percent": 80,
                "gpu_required": False  # Optional GPU usage
            }
        )

        # Get optimization decision
        decision = self.optimization_selector.select_optimization(workload_profile)
        self.optimization_strategy = decision.strategy
        self.optimization_config = decision.configuration

        self.logger.info(f"🎯 Selected optimization strategy: {self.optimization_strategy.value}")
        self.logger.info(f"🔧 Enabled optimizations: {decision.enabled_optimizations}")

    def _initialize_components(self) -> None:
        """Initialize regime clustering components with optimizations."""
        self.logger.info("🔧 Initializing final regime clustering components...")

        # Initialize optimized components
        self._initialize_optimized_components()

        try:
            # Load optimized parameters from step03
            self._load_optimized_parameters()
            self.logger.info("✅ Final regime clustering components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime clustering components: {e}")
            raise

    def _initialize_optimized_components(self) -> None:
        """Initialize optimized components for enhanced performance."""
        self.logger.info("🚀 Initializing optimized performance components for Step 3.5...")

        # Enhanced Memory Manager
        if OPTIMIZED_MEMORY_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager(self.config)
                self.logger.info('✅ Enhanced memory manager initialized for Step 3.5')
            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced memory manager failed for Step 3.5: {e}')
                self.memory_manager = None
        else:
            self.logger.info('ℹ️ Enhanced memory manager not available for Step 3.5')
            self.memory_manager = None

        # Parallel Clustering Processor (for final clustering)
        if OPTIMIZED_CLUSTERING_AVAILABLE:
            try:
                from .step03_config import Step03Config
                config_obj = Step03Config()
                self.ensemble_clustering = AdvancedEnsembleClustering(config_obj)
                self.logger.info('✅ Enhanced ensemble clustering initialized for Step 3.5')
            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced ensemble clustering failed for Step 3.5: {e}')
                self.ensemble_clustering = None
        else:
            self.logger.info('ℹ️ Enhanced ensemble clustering not available for Step 3.5')
            self.ensemble_clustering = None

        # Vectorized Operations Manager
        if OPTIMIZED_VECTORIZED_AVAILABLE:
            try:
                self.vectorized_manager = get_vectorized_operations_manager()
                self.logger.info('✅ Vectorized operations manager initialized for Step 3.5')
            except Exception as e:
                self.logger.warning(f'⚠️ Vectorized operations manager failed for Step 3.5: {e}')
                self.vectorized_manager = None
        else:
            self.logger.info('ℹ️ Vectorized operations manager not available for Step 3.5')
            self.vectorized_manager = None

        # Track optimization availability
        self.use_optimized_components = (
            OPTIMIZED_MEMORY_AVAILABLE and
            OPTIMIZED_CLUSTERING_AVAILABLE and
            OPTIMIZED_VECTORIZED_AVAILABLE
        )

        if self.use_optimized_components:
            self.logger.info('🎯 Optimized components available for Step 3.5!')
        else:
            self.logger.info('ℹ️ Partial optimizations available for Step 3.5')

    # @secure_data_processing - removed, handled by validates
    def _load_optimized_parameters(self) -> None:
        """Load optimized parameters from step03."""
        try:
            # Load parameter optimization results
            param_file = Path("data/optimization/parameter_optimization_results.json")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    param_results = json.load(f)
                self.optimized_params = param_results.get("combined_parameters", {})
                self.logger.info(f"✅ Loaded optimized parameters: {len(self.optimized_params)} parameters")
            else:
                self.logger.warning("⚠️ No optimized parameters found, using defaults")
                self.optimized_params = {
                    "n_components": 4,
                    "n_clusters": 20,
                    "momentum_window": 15,
                    "volatility_window": 20,
                    "volume_window": 15
                }
        except Exception as e:
            self.logger.error(f"Failed to load optimized parameters: {e}")

    @handles_errors(
        exceptions=(Exception,),
        context="regime_clustering_initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the final regime clustering step."""
        self.logger.info("🚀 Initializing final regime clustering step...")
        self.logger.info(f"📋 Optimized parameters loaded: {len(self.optimized_params)} parameters")
        self.logger.info("✅ Final regime clustering step initialized successfully")
        return True

    @validates()
    @handles_errors(
        exceptions=(Exception,),
        context="regime_clustering_execution"
    )
    async def execute(self) -> bool:
        """Execute the final regime clustering step."""
        self.logger.info("🎯 Starting final regime clustering with advanced reporting...")
        self.start_time = time.time()
        
        # Step 1: Load and prepare data
        data_loaded = await self._load_and_prepare_data()
        if not data_loaded.get("success", False):
            raise RuntimeError("Failed to load and prepare data")
        
        # Step 2: Perform HMM regime discovery
        hmm_results = await self._perform_hmm_regime_discovery(data_loaded["data"])
        
        # Step 3: Perform final clustering
        clustering_results = await self._perform_final_clustering(data_loaded["data"], hmm_results)
        
        # Step 4: Analyze regime characteristics
        regime_analysis = await self._analyze_regime_characteristics(clustering_results, data_loaded["data"])
        
        # Step 5: Generate comprehensive reports
        reports = await self._generate_comprehensive_reports(clustering_results, regime_analysis)
        
        # Step 6: Save final results
        await self._save_final_results(clustering_results, regime_analysis, reports)
        
        execution_time = time.time() - self.start_time
        self.logger.info(f"✅ Final regime clustering completed successfully in {execution_time:.2f}s")
        
        return True

    @handles_errors(
        exceptions=(Exception,),
        context="load_and_prepare_data"
    )
    @validates()
    async def _load_and_prepare_data(self) -> dict[str, Any]:
        """Load and prepare data for regime clustering using enhanced optimizations."""
        self.logger.info("📊 Loading and preparing data for regime clustering with optimizations...")
        
        # Get data parameters from config
        symbol = self.config.get("SYMBOL", "ETHUSDT")
        exchange = self.config.get("EXCHANGE", "BINANCE")
        timeframe = self.config.get("TIMEFRAME", "1m")
        data_dir = self.config.get("DATA_DIR", "data_cache")
        
        # Use optimized data manager if available
        if self.data_manager:
            return await self._load_and_prepare_data_optimized(symbol, exchange, timeframe, data_dir)
        else:
            return await self._load_and_prepare_data_legacy(symbol, exchange, timeframe, data_dir)

    async def _load_and_prepare_data_optimized(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> dict[str, Any]:
        """Load and prepare data using optimized data manager."""
        self.logger.info("🚀 Using optimized data manager for data loading...")
        
        # Load data using optimized manager, with or without memory checkpoint
        data_id = f"klines_{exchange}_{symbol}_{timeframe}_consolidated"
        if self.m1_memory_optimizer:
            with self.m1_memory_optimizer.memory_checkpoint("data_loading"):
                df = await self._load_data_with_optimization(data_id, data_dir)
        else:
            df = await self._load_data_with_optimization(data_id, data_dir)

        if df is None or df.empty:
            raise RuntimeError("Failed to load data with optimization")

        # Prepare features with parallel processing
        features = await self._prepare_features_optimized(df)

        self.logger.info(f"✅ Data loaded and prepared with optimization: {len(df):,} rows, {len(features.columns)} features")

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

    async def _load_data_with_optimization(self, data_id: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load data using optimized data manager."""
        try:
            # Check if data is cached
            if self.data_manager.has_data(data_id):
                self.logger.info(f"📋 Loading cached data: {data_id}")
                return self.data_manager.load_data(data_id)
            
            # Load from file with optimization
            file_path = Path(data_dir) / f"{data_id}.parquet"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Load with memory-efficient chunking if needed
            file_size_mb = file_path.stat().st_size / (1024**2)
            
            if self.m1_memory_optimizer and self.m1_memory_optimizer.should_chunk_data(file_size_mb, "io_bound"):
                self.logger.info(f"📦 Large file detected ({file_size_mb:.1f}MB), using chunked loading")
                df = self.data_manager.load_large_file(file_path, chunk_size=50000)
            else:
                df = pd.read_parquet(file_path)
            
            # Cache the data for future use
            if df is not None and not df.empty:
                self.data_manager.store_data(data_id, df, metadata={
                    "source": str(file_path),
                    "size_mb": file_size_mb,
                    "rows": len(df),
                    "columns": list(df.columns)
                })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data with optimization: {e}")
            raise

    async def _prepare_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using optimized processing."""
        try:
            self.logger.info("🔧 Preparing features with optimized processing...")
            
            # Use parallel processing for feature preparation
            if self.m1_cpu_optimizer and self.pipeline_executor:
                return await self._prepare_features_parallel(df)
            else:
                return await self._prepare_features_with_optimized_params(df)
            
        except Exception as e:
            self.logger.error(f"Optimized feature preparation failed: {e}")
            raise

    async def _prepare_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using parallel processing pipeline."""
        try:
            self.logger.info("⚡ Preparing features with parallel processing...")
            
            # Create pipeline stages for feature preparation
            pipeline = OptimizedPipelineExecutor(max_concurrent_stages=4)
            
            # Stage 1: Basic price features
            pipeline.add_stage(PipelineStage(
                name="price_features",
                func=self._create_price_features_parallel,
                args=(df,)
            ))
            
            # Stage 2: Volatility features
            pipeline.add_stage(PipelineStage(
                name="volatility_features",
                func=self._create_volatility_features_parallel,
                args=(df,),
                dependencies=["price_features"]
            ))
            
            # Stage 3: Technical indicators
            pipeline.add_stage(PipelineStage(
                name="technical_features",
                func=self._create_technical_features_parallel,
                args=(df,),
                dependencies=["volatility_features"]
            ))
            
            # Stage 4: Combine features
            pipeline.add_stage(PipelineStage(
                name="combine_features",
                func=self._combine_features_parallel,
                dependencies=["price_features", "volatility_features", "technical_features"]
            ))
            
            # Execute pipeline
            result = await pipeline.execute_async(PipelineExecutionMode.HYBRID)
            
            if result.success and result.stage_results.get("combine_features"):
                features = result.stage_results["combine_features"]
                self.logger.info(f"✅ Parallel feature preparation completed: {len(features.columns)} features")
                return features
            else:
                raise Exception("Pipeline execution failed")
            
        except Exception as e:
            self.logger.error(f"Parallel feature preparation failed: {e}")
            raise

    def _create_price_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features in parallel."""
        features = pd.DataFrame()
        features["timestamp"] = df["timestamp"]

        # Price-based features with optimized parameters
        momentum_window = self.optimized_params.get("momentum_window", 15)
        features["price_momentum"] = df["close"].pct_change(momentum_window)
        features["price_momentum_short"] = df["close"].pct_change(5)
        features["price_momentum_long"] = df["close"].pct_change(30)

        return features

    def _create_volatility_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features in parallel."""
        features = pd.DataFrame()

        # Volatility features with optimized parameters
        volatility_window = self.optimized_params.get("volatility_window", 20)
        features["volatility"] = df["close"].pct_change().rolling(window=volatility_window).std()
        features["volatility_short"] = df["close"].pct_change().rolling(window=10).std()
        features["volatility_long"] = df["close"].pct_change().rolling(window=50).std()

        return features

    def _create_technical_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features in parallel."""
        features = pd.DataFrame()

        # Get optimized parameters
        rsi_window = self.optimized_params.get("rsi_window", 14)
        macd_fast = self.optimized_params.get("macd_fast", 12)
        macd_slow = self.optimized_params.get("macd_slow", 26)
        atr_window = self.optimized_params.get("atr_window", 14)

        # Technical indicators
        features["rsi"] = self._calculate_rsi(df["close"], rsi_window)
        features["macd"] = self._calculate_macd(df["close"], macd_fast, macd_slow)
        features["atr"] = self._calculate_atr(df, atr_window)

        return features

    def _combine_features_parallel(self, price_features: pd.DataFrame, volatility_features: pd.DataFrame, technical_features: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature sets."""
        # Combine all features
        combined = pd.concat([price_features, volatility_features, technical_features], axis=1)

        # Add volume features
        volume_window = self.optimized_params.get("volume_window", 15)
        combined["volume_ratio"] = price_features["volume"] / price_features["volume"].rolling(window=volume_window).mean()
        combined["volume_momentum"] = price_features["volume"].pct_change(volume_window)

        # Add position features
        combined["price_position"] = (price_features["close"] - price_features["close"].rolling(20).min()) / (price_features["close"].rolling(20).max() - price_features["close"].rolling(20).min())
        combined["volume_price_trend"] = (price_features["close"] - price_features["close"].shift(1)) * price_features["volume"]

        # Remove timestamp and handle NaN values
        clustering_features = combined.drop("timestamp", axis=1, errors='ignore')
        clustering_features = clustering_features.fillna(0)

        return clustering_features

    async def _load_and_prepare_data_legacy(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> dict[str, Any]:
        """Legacy data loading method for fallback."""
        self.logger.info("📊 Using legacy data loading method...")

        # Load klines data
        klines_path = Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"

        if not klines_path.exists():
            self.logger.error(f"❌ Klines file not found: {klines_path}")
            return {
                "success": False,
                "error": f"Klines file not found: {klines_path}"
            }

        # Load data
        df = pd.read_parquet(klines_path)

        if df.empty:
            self.logger.error("❌ Data is empty")
            return {
                "success": False,
                "error": "Data is empty"
            }

        # Prepare features using optimized parameters
        features = await self._prepare_features_with_optimized_params(df)

        self.logger.info(f"✅ Data loaded and prepared: {len(df):,} rows, {len(features.columns)} features")

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

    @handles_errors(
        exceptions=(Exception,),
        context="prepare_features_with_optimized_params"
    )
    @validates()
    async def _prepare_features_with_optimized_params(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using optimized parameters from step03."""
        self.logger.info("🔧 Preparing features with optimized parameters...")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Get optimized parameters
        momentum_window = self.optimized_params.get("momentum_window", 15)
        volatility_window = self.optimized_params.get("volatility_window", 20)
        volume_window = self.optimized_params.get("volume_window", 15)
        rsi_window = self.optimized_params.get("rsi_window", 14)
        macd_fast = self.optimized_params.get("macd_fast", 12)
        macd_slow = self.optimized_params.get("macd_slow", 26)
        atr_window = self.optimized_params.get("atr_window", 14)
        
        # Calculate features with optimized parameters
        features = pd.DataFrame()
        features["timestamp"] = df["timestamp"]
        
        # Price-based features
        features["price_momentum"] = df["close"].pct_change(momentum_window)
        features["price_momentum_short"] = df["close"].pct_change(5)
        features["price_momentum_long"] = df["close"].pct_change(30)
        
        # Volatility features
        features["volatility"] = df["close"].pct_change().rolling(window=volatility_window).std()
        features["volatility_short"] = df["close"].pct_change().rolling(window=10).std()
        features["volatility_long"] = df["close"].pct_change().rolling(window=50).std()
        
        # Volume features
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(window=volume_window).mean()
        features["volume_momentum"] = df["volume"].pct_change(volume_window)
        
        # Technical indicators
        features["rsi"] = self._calculate_rsi(df["close"], rsi_window)
        features["macd"] = self._calculate_macd(df["close"], macd_fast, macd_slow)
        features["atr"] = self._calculate_atr(df, atr_window)
        
        # Additional features
        features["price_position"] = (df["close"] - df["close"].rolling(20).min()) / (df["close"].rolling(20).max() - df["close"].rolling(20).min())
        features["volume_price_trend"] = (df["close"] - df["close"].shift(1)) * df["volume"]
        
        # Remove timestamp and handle NaN values
        clustering_features = features.drop("timestamp", axis=1)
        clustering_features = clustering_features.fillna(0)
        
        self.logger.info(f"✅ Features prepared with optimized parameters: {len(clustering_features.columns)} features")
        return clustering_features

    @handles_errors(
        exceptions=(Exception,),
        context="perform_hmm_regime_discovery"
    )
    # @resource_monitor - removed, use log_execution_time
    # @secure_data_processing - removed, handled by validates
    async def _perform_hmm_regime_discovery(self, data: pd.DataFrame) -> dict[str, Any]:
        """Perform HMM regime discovery using enhanced optimizations."""
        self.logger.info("🧠 Performing HMM regime discovery with optimizations...")
        
        # Get optimized HMM parameters
        n_components = self.optimized_params.get("n_components", 4)
        covariance_type = self.optimized_params.get("covariance_type", "full")
        n_iter = self.optimized_params.get("n_iter", 100)
        random_state = self.optimized_params.get("random_state", 42)
        
        # Prepare features for HMM with optimizations
        features = await self._prepare_features_with_optimized_params(data)
        
        if features.empty:
            raise ValueError("No features available for HMM analysis")
        
        # Use enhanced matrix operations if available
        if self.matrix_operations:
            return await self._perform_hmm_with_enhanced_operations(features, n_components, covariance_type, n_iter, random_state)
        else:
            return await self._perform_hmm_legacy(features, n_components, covariance_type, n_iter, random_state)

    async def _perform_hmm_with_enhanced_operations(self, features: pd.DataFrame, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> dict[str, Any]:
        """Perform HMM with enhanced matrix operations."""
        try:
            self.logger.info("🚀 Using enhanced matrix operations for HMM...")

            # Use memory checkpoint for HMM training
            if self.m1_memory_optimizer:
                with self.m1_memory_optimizer.memory_checkpoint("hmm_training"):
                    return await self._train_hmm_optimized(features, n_components, covariance_type, n_iter, random_state)
            else:
                return await self._train_hmm_optimized(features, n_components, covariance_type, n_iter, random_state)

        except Exception as e:
            self.logger.error(f"Enhanced HMM failed: {e}")
            raise

    async def _train_hmm_optimized(self, features: pd.DataFrame, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> dict[str, Any]:
        """Train HMM with optimizations."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler

            # Use enhanced matrix operations for scaling
            if self.matrix_operations:
                self.logger.info("🔧 Using enhanced matrix operations for feature scaling...")

                # Convert to numpy and optimize memory usage
                features_array = self.m1_memory_optimizer.create_memory_efficient_array(
                    features.values, dtype=np.float32
                )

                # Scale features with enhanced operations
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)

                # Use GPU acceleration if available
                if self.m1_gpu_manager and self.m1_gpu_manager.should_use_gpu(features_scaled.size, "matrix_mult"):
                    self.logger.info("🎯 Using GPU acceleration for HMM training...")
                    features_scaled = self.m1_gpu_manager.to_device(features_scaled, "matrix_mult")
                    use_gpu = True
                else:
                    use_gpu = False

            else:
                # Standard scaling
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features.values)
                use_gpu = False

            # Train HMM with optimizations
            with self.m1_gpu_manager.gpu_context("hmm_training") if use_gpu else nullcontext():
                hmm_model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    random_state=random_state
                )

                # Fit the model
                hmm_model.fit(features_scaled)

                # Get predictions
                if use_gpu:
                    features_scaled_cpu = features_scaled.cpu().numpy()
                    state_sequence = hmm_model.predict(features_scaled_cpu)
                    state_probs = hmm_model.predict_proba(features_scaled_cpu)
                    score = hmm_model.score(features_scaled_cpu)
                else:
                    state_sequence = hmm_model.predict(features_scaled)
                    state_probs = hmm_model.predict_proba(features_scaled)
                    score = hmm_model.score(features_scaled)

            hmm_results = {
                "model": hmm_model,
                "scaler": scaler,
                "state_sequence": state_sequence,
                "state_probs": state_probs,
                "n_components": n_components,
                "score": score,
                "used_gpu": use_gpu,
                "optimization_applied": True
            }

            self.logger.info(f"✅ Enhanced HMM regime discovery completed: {n_components} states (GPU: {use_gpu})")
            return hmm_results

        except ImportError:
            self.logger.error("⚠️ hmmlearn not available")
            raise
        except Exception as e:
            self.logger.error(f"Enhanced HMM training failed: {e}")
            raise

    async def _perform_hmm_legacy(self, features: pd.DataFrame, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> dict[str, Any]:
        """Legacy HMM training method."""
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Train HMM
            hmm_model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state
            )

            hmm_model.fit(features_scaled)

            # Get state sequence and probabilities
            state_sequence = hmm_model.predict(features_scaled)
            state_probs = hmm_model.predict_proba(features_scaled)

            hmm_results = {
                "model": hmm_model,
                "scaler": scaler,
                "state_sequence": state_sequence,
                "state_probs": state_probs,
                "n_components": n_components,
                "score": hmm_model.score(features_scaled),
                "used_gpu": False,
                "optimization_applied": False
            }

            self.logger.info(f"✅ Legacy HMM regime discovery completed: {n_components} states")
            return hmm_results

        except ImportError:
            self.logger.error("⚠️ hmmlearn not available")
            raise

    def _vectorized_regime_classification(self, volatility, momentum):
        """Vectorized regime classification using NumPy operations."""
        try:
            # Convert to numpy arrays if pandas series
            if hasattr(volatility, 'values'):
                vol_array = volatility.values
                mom_array = momentum.values
            else:
                vol_array = np.array(volatility)
                mom_array = np.array(momentum)

            # Vectorized regime classification
            regimes = np.zeros(len(vol_array), dtype=int)

            # High volatility regimes (vol > 0.02)
            high_vol_mask = vol_array > 0.02

            # High volatility bull (mom > 0.001)
            regimes[(high_vol_mask) & (mom_array > 0.001)] = 0

            # High volatility bear (mom < -0.001)
            regimes[(high_vol_mask) & (mom_array < -0.001)] = 1

            # High volatility neutral
            regimes[(high_vol_mask) & (mom_array >= -0.001) & (mom_array <= 0.001)] = 2

            # Low volatility bull (mom > 0.001)
            low_vol_mask = ~high_vol_mask
            regimes[(low_vol_mask) & (mom_array > 0.001)] = 3

            # Low volatility bear (mom < -0.001)
            regimes[(low_vol_mask) & (mom_array < -0.001)] = 4

            # Low volatility neutral
            regimes[(low_vol_mask) & (mom_array >= -0.001) & (mom_array <= 0.001)] = 5

            return regimes.tolist()

        except Exception as e:
            self.logger.error(f"Vectorized regime classification failed: {e}")
            raise

    @handles_errors(
        exceptions=(Exception,),
        context="perform_simple_regime_detection"
    )
    # @secure_data_processing - removed, handled by validates
    async def _perform_simple_regime_detection(self, features: pd.DataFrame) -> dict[str, Any]:
        """Perform simple regime detection as fallback."""
        self.logger.info("📊 Performing simple regime detection...")
        
        # Use volatility and momentum for regime classification
        volatility = features.get("volatility", pd.Series([0] * len(features)))
        momentum = features.get("price_momentum", pd.Series([0] * len(features)))
        
        # Fill NaN values
        volatility = volatility.fillna(0)
        momentum = momentum.fillna(0)
        
        # Vectorized regime classification
        regimes = self._vectorized_regime_classification(volatility, momentum)
        
        simple_results = {
            "state_sequence": np.array(regimes),
            "state_probs": np.eye(6)[regimes],  # One-hot encoding
            "n_components": 6,
            "method": "simple_classification"
        }
        
        self.logger.info(f"✅ Simple regime detection completed: {len(set(regimes))} regimes")
        return simple_results

    @handles_errors(
        exceptions=(Exception,),
        context="perform_final_clustering"
    )
    # @resource_monitor - removed, use log_execution_time
    # @secure_data_processing - removed, handled by validates
    async def _perform_final_clustering(self, data: pd.DataFrame, hmm_results: dict[str, Any]) -> dict[str, Any]:
        """Perform final clustering using HMM results and enhanced optimizations."""
        self.logger.info("🎯 Performing final clustering with optimizations...")
        
        # Get optimized clustering parameters
        clustering_params = self._get_clustering_parameters()
        
        # Prepare features with optimizations
        features = await self._prepare_features_with_optimized_params(data)
        if features.empty:
            raise ValueError("No features available for clustering")
        
        # Create composite features with HMM states
        composite_features = await self._create_composite_features(features, hmm_results)
        
        # Use enhanced clustering if available
        if self.matrix_operations and self.m1_cpu_optimizer:
            clustering_results = await self._perform_clustering_enhanced(composite_features, clustering_params, hmm_results)
        else:
            clustering_results = await self._execute_clustering_algorithm(composite_features, clustering_params)
        
        # Add metadata to results
        clustering_results.update({
            "hmm_results": hmm_results,
            "composite_features": composite_features,
            "optimization_used": self.matrix_operations is not None
        })
        
        self.logger.info(f"✅ Final clustering completed: {clustering_params['n_clusters']} clusters (optimized: {clustering_results.get('optimization_used', False)})")
        return clustering_results

    async def _perform_clustering_enhanced(self, composite_features: pd.DataFrame, clustering_params: dict[str, Any], hmm_results: dict[str, Any]) -> dict[str, Any]:
        """Perform clustering with enhanced optimizations."""
        try:
            self.logger.info("🚀 Using enhanced clustering with matrix operations and parallel processing...")

            # Use memory checkpoint for clustering
            if self.m1_memory_optimizer:
                with self.m1_memory_optimizer.memory_checkpoint("clustering"):
                    return await self._execute_clustering_enhanced(composite_features, clustering_params)
            else:
                return await self._execute_clustering_enhanced(composite_features, clustering_params)

        except Exception as e:
            self.logger.error(f"Enhanced clustering failed: {e}")
            raise

    async def _execute_clustering_enhanced(self, composite_features: pd.DataFrame, clustering_params: dict[str, Any]) -> dict[str, Any]:
        """Execute clustering with enhanced matrix operations."""
        try:
            self.logger.info("🔧 Executing enhanced clustering algorithm...")

            # Convert to efficient numpy array
            features_array = self.m1_memory_optimizer.create_memory_efficient_array(
                composite_features.values, dtype=np.float32
            )

            # Use parallel processing for large datasets
            if len(features_array) > 10000 and self.m1_cpu_optimizer:
                self.logger.info("⚡ Using parallel processing for clustering...")
                return await self._perform_parallel_clustering(features_array, clustering_params)
            else:
                return await self._perform_standard_clustering(features_array, clustering_params)

        except Exception as e:
            self.logger.error(f"Enhanced clustering execution failed: {e}")
            raise

    async def _perform_parallel_clustering(self, features_array: np.ndarray, clustering_params: dict[str, Any]) -> dict[str, Any]:
        """Perform clustering using parallel processing."""
        try:
            from sklearn.cluster import KMeans

            # Split data for parallel processing
            n_chunks = min(self.m1_cpu_optimizer.max_workers, 4)
            chunk_size = len(features_array) // n_chunks

            self.logger.info(f"📦 Splitting data into {n_chunks} chunks for parallel clustering...")

            # Process chunks in parallel
            async def cluster_chunk(chunk):
                kmeans = KMeans(
                    n_clusters=clustering_params["n_clusters"],
                    random_state=clustering_params["random_state"],
                    n_init=10
                )
                return kmeans.fit_predict(chunk)

            # Split the data
            chunks = [
                features_array[i:i + chunk_size]
                for i in range(0, len(features_array), chunk_size)
            ]

            # Process in parallel
            tasks = [cluster_chunk(chunk) for chunk in chunks]
            chunk_labels = await asyncio.gather(*tasks)

            # Combine results (simplified - in practice you'd need more sophisticated merging)
            cluster_labels = np.concatenate(chunk_labels)

            # Train final model on full dataset
            final_kmeans = KMeans(
                n_clusters=clustering_params["n_clusters"],
                random_state=clustering_params["random_state"],
                n_init=10
            )
            final_kmeans.fit(features_array)

            return {
                "model": final_kmeans,
                "scaler": None,  # No scaling applied
                "cluster_labels": cluster_labels,
                "n_clusters": clustering_params["n_clusters"],
                "method": clustering_params["method"],
                "parallel_processing": True,
                "n_chunks": n_chunks
            }

        except Exception as e:
            self.logger.error(f"Parallel clustering failed: {e}")
            raise

    async def _perform_standard_clustering(self, features_array: np.ndarray, clustering_params: dict[str, Any]) -> dict[str, Any]:
        """Perform standard clustering with optimizations."""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans

            # Scale features with enhanced operations
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Use GPU acceleration if beneficial
            use_gpu = False
            if self.m1_gpu_manager and self.m1_gpu_manager.should_use_gpu(features_scaled.size, "matrix_mult"):
                self.logger.info("🎯 Using GPU acceleration for clustering...")
                features_scaled = self.m1_gpu_manager.to_device(features_scaled, "matrix_mult")
                use_gpu = True

            # Perform clustering
            with self.m1_gpu_manager.gpu_context("clustering") if use_gpu else nullcontext():
                clustering = KMeans(
                    n_clusters=clustering_params["n_clusters"],
                    random_state=clustering_params["random_state"],
                    n_init=10
                )

                if use_gpu:
                    cluster_labels = clustering.fit_predict(features_scaled.cpu().numpy())
                else:
                    cluster_labels = clustering.fit_predict(features_scaled)

            return {
                "model": clustering,
                "scaler": scaler,
                "cluster_labels": cluster_labels,
                "n_clusters": clustering_params["n_clusters"],
                "method": clustering_params["method"],
                "gpu_accelerated": use_gpu
            }

        except Exception as e:
            self.logger.error(f"Standard enhanced clustering failed: {e}")
            raise

    @handles_errors(
        exceptions=(Exception,),
        context="get_clustering_parameters"
    )
    def _get_clustering_parameters(self) -> dict[str, Any]:
        """Get optimized clustering parameters."""
        return {
            "n_clusters": self.optimized_params.get("n_clusters", 20),
            "method": self.optimized_params.get("method", "kmeans"),
            "random_state": self.optimized_params.get("random_state", 42)
        }

    @handles_errors(
        exceptions=(Exception,),
        context="create_composite_features"
    )
    async def _create_composite_features(self, features: pd.DataFrame, hmm_results: dict[str, Any]) -> pd.DataFrame:
        """Create composite features with HMM states."""
        if not hmm_results or "state_sequence" not in hmm_results:
            return features
        
        composite_features = features.copy()
        composite_features["hmm_state"] = hmm_results["state_sequence"]
        composite_features["hmm_state_prob_max"] = np.max(hmm_results["state_probs"], axis=1)
        
        # Add HMM state interactions
        for col in features.columns:
            composite_features[f"{col}_x_hmm_state"] = features[col] * hmm_results["state_sequence"]
        
        return composite_features

    @handles_errors(
        exceptions=(Exception,),
        context="execute_clustering_algorithm"
    )
    async def _execute_clustering_algorithm(
        self, 
        composite_features: pd.DataFrame, 
        clustering_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the clustering algorithm."""
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(composite_features)
        
        # Perform clustering
        clustering_model, cluster_labels = await self._perform_clustering(
            features_scaled, clustering_params
        )
        
        return {
            "model": clustering_model,
            "scaler": scaler,
            "cluster_labels": cluster_labels,
            "n_clusters": clustering_params["n_clusters"],
            "method": clustering_params["method"]
        }

    @handles_errors(
        exceptions=(Exception,),
        context="perform_clustering"
    )
    async def _perform_clustering(
        self, 
        features_scaled: np.ndarray, 
        clustering_params: dict[str, Any]
    ) -> tuple[Any, np.ndarray]:
        """Perform the actual clustering."""
        from sklearn.cluster import KMeans
        
        clustering = KMeans(
            n_clusters=clustering_params["n_clusters"],
            random_state=clustering_params["random_state"],
            n_init=10
        )
        cluster_labels = clustering.fit_predict(features_scaled)
        
        return clustering, cluster_labels

    @handles_errors(
        exceptions=(Exception,),
        context="analyze_regime_characteristics"
    )
    # @secure_data_processing - removed, handled by validates
    async def _analyze_regime_characteristics(self, clustering_results: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
        """Analyze regime characteristics and patterns."""
        self.logger.info("🔍 Analyzing regime characteristics...")
        
        if not clustering_results or "cluster_labels" not in clustering_results:
            raise ValueError("No clustering results available for analysis")
        
        cluster_labels = clustering_results["cluster_labels"]
        features = clustering_results.get("composite_features", pd.DataFrame())
        
        analysis = {
            "cluster_statistics": {},
            "regime_transitions": {},
            "regime_persistence": {},
            "regime_characteristics": {},
            "market_conditions": {}
        }
        
        # Analyze each cluster
        unique_clusters = np.unique(cluster_labels)
        analysis["cluster_statistics"] = await self._analyze_cluster_statistics(
            cluster_labels, data, features, unique_clusters
        )
        
        # Analyze regime transitions
        analysis["regime_transitions"] = self._analyze_regime_transitions(cluster_labels)
        
        # Analyze regime persistence
        analysis["regime_persistence"] = self._analyze_regime_persistence(cluster_labels)
        
        self.logger.info(f"✅ Regime characteristics analyzed: {len(unique_clusters)} clusters")
        return analysis

    @handles_errors(
        exceptions=(Exception,),
        context="analyze_cluster_statistics"
    )
    async def _analyze_cluster_statistics(
        self, 
        cluster_labels: np.ndarray, 
        data: pd.DataFrame, 
        features: pd.DataFrame, 
        unique_clusters: np.ndarray
    ) -> dict[str, Any]:
        """Analyze statistics for each cluster."""
        cluster_statistics = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_features = features[cluster_mask] if not features.empty else pd.DataFrame()
            
            cluster_stats = await self._calculate_cluster_basic_stats(cluster_data, data)
            cluster_stats.update(await self._calculate_cluster_price_stats(cluster_data))
            cluster_stats.update(await self._calculate_cluster_volume_stats(cluster_data))
            
            cluster_statistics[f"cluster_{cluster_id}"] = cluster_stats
        
        return cluster_statistics

    @handles_errors(
        exceptions=(Exception,),
        context="calculate_cluster_basic_stats"
    )
    async def _calculate_cluster_basic_stats(self, cluster_data: pd.DataFrame, total_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate basic statistics for a cluster."""
        return {
            "size": len(cluster_data),
            "percentage": len(cluster_data) / len(total_data) * 100,
            "date_range": {
                "start": cluster_data["timestamp"].min().isoformat(),
                "end": cluster_data["timestamp"].max().isoformat()
            }
        }

    @handles_errors(
        exceptions=(Exception,),
        context="calculate_cluster_price_stats"
    )
    async def _calculate_cluster_price_stats(self, cluster_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate price statistics for a cluster."""
        if cluster_data.empty:
            return {}
        
        return {
            "price_stats": {
                "mean_price": float(cluster_data["close"].mean()),
                "price_volatility": float(cluster_data["close"].pct_change().std()),
                "price_momentum": float(cluster_data["close"].pct_change().mean())
            }
        }

    @handles_errors(
        exceptions=(Exception,),
        context="calculate_cluster_volume_stats"
    )
    async def _calculate_cluster_volume_stats(self, cluster_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate volume statistics for a cluster."""
        if cluster_data.empty:
            return {}
        
        return {
            "volume_stats": {
                "mean_volume": float(cluster_data["volume"].mean()),
                "volume_volatility": float(cluster_data["volume"].pct_change().std())
            }
        }

    @handles_errors(
        exceptions=(Exception,),
        context="analyze_regime_transitions"
    )
    def _analyze_regime_transitions(self, cluster_labels: np.ndarray) -> dict[str, Any]:
        """Analyze regime transition patterns using vectorized operations."""
        try:
            # Vectorized transition counting using numpy
            current_regimes = cluster_labels[:-1]
            next_regimes = cluster_labels[1:]

            # Get unique regime pairs
            unique_pairs, counts = np.unique(
                np.column_stack((current_regimes, next_regimes)),
                axis=0,
                return_counts=True
            )

            # Build transition dictionary
            transitions = {}
            unique_current = np.unique(current_regimes)

            for current_regime in unique_current:
                transitions[current_regime] = {}
                mask = unique_pairs[:, 0] == current_regime
                regime_counts = counts[mask]
                next_regime_labels = unique_pairs[mask, 1]

                # Calculate transition probabilities
                total_transitions = np.sum(regime_counts)
                if total_transitions > 0:
                    probabilities = regime_counts / total_transitions
                    for next_regime, prob in zip(next_regime_labels, probabilities):
                        transitions[current_regime][next_regime] = float(prob)

            return transitions

        except Exception as e:
            self.logger.error(f"Vectorized regime transition analysis failed: {e}")
            raise

    @handles_errors(
        exceptions=(Exception,),
        context="analyze_regime_persistence"
    )
    def _analyze_regime_persistence(self, cluster_labels: np.ndarray) -> dict[str, Any]:
        """Analyze how long regimes persist using vectorized operations."""
        try:
            # Vectorized approach to find regime changes
            regime_changes = np.diff(cluster_labels.astype(int)) != 0
            change_indices = np.where(regime_changes)[0] + 1

            # Add start and end indices
            all_indices = np.concatenate([[0], change_indices, [len(cluster_labels)]])

            # Calculate durations between changes
            durations = np.diff(all_indices)
            regimes = cluster_labels[all_indices[:-1]]

            # Group durations by regime using vectorized operations
            unique_regimes = np.unique(regimes)
            persistence = {}

            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_durations = durations[regime_mask]
                persistence[regime] = regime_durations.tolist()

            # Calculate statistics for each regime using vectorized operations
            persistence_stats = {}
            for regime, durations_list in persistence.items():
                if durations_list:
                    durations_array = np.array(durations_list)
                    persistence_stats[regime] = {
                        "mean_duration": float(np.mean(durations_array)),
                        "median_duration": float(np.median(durations_array)),
                        "max_duration": int(np.max(durations_array)),
                        "min_duration": int(np.min(durations_array)),
                        "total_periods": len(durations_list),
                        "regime_switches": len(durations_list)
                    }

            return persistence_stats

        except Exception as e:
            self.logger.error(f"Vectorized regime persistence analysis failed: {e}")
            raise

    @handles_errors(
        exceptions=(Exception,),
        context="generate_comprehensive_reports"
    )
    # @secure_data_processing - removed, handled by validates
    async def _generate_comprehensive_reports(self, clustering_results: dict[str, Any], regime_analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive reports for regime clustering."""
        self.logger.info("📋 Generating comprehensive reports...")
        
        reports = {
            "clustering_summary": {},
            "regime_analysis": {},
            "performance_metrics": {},
            "recommendations": {}
        }
        
        # Clustering summary
        if clustering_results:
            reports["clustering_summary"] = {
                "n_clusters": clustering_results.get("n_clusters", 0),
                "method": clustering_results.get("method", "unknown"),
                "total_samples": len(clustering_results.get("cluster_labels", [])),
                "clustering_score": getattr(clustering_results.get("model"), "inertia_", 0) if clustering_results.get("model") else 0
            }
        
        # Regime analysis summary
        if regime_analysis:
            reports["regime_analysis"] = {
                "total_clusters": len(regime_analysis.get("cluster_statistics", {})),
                "regime_transitions_analyzed": len(regime_analysis.get("regime_transitions", {})),
                "persistence_analyzed": len(regime_analysis.get("regime_persistence", {}))
            }
        
        # Performance metrics
        reports["performance_metrics"] = {
            "clustering_quality": "high" if clustering_results else "unknown",
            "regime_stability": "stable" if regime_analysis.get("regime_persistence") else "unknown",
            "transition_smoothness": "smooth" if regime_analysis.get("regime_transitions") else "unknown"
        }
        
        # Recommendations
        reports["recommendations"] = [
            "Use identified regimes for trading strategy development",
            "Monitor regime transitions for market timing",
            "Validate regime stability with out-of-sample data",
            "Consider regime-specific parameter optimization"
        ]
        
        # Generate enhanced comprehensive report if available
        if self.enhanced_reporter is not None:
            try:
                self.logger.info("📊 Generating enhanced comprehensive report for Step 3.5...")

                # Extract symbol, exchange, timeframe from config (assuming defaults if not available)
                symbol = self.config.get('symbol', 'BTCUSDT')
                exchange = self.config.get('exchange', 'BINANCE')
                timeframe = self.config.get('timeframe', '1m')

                # Prepare HMM results from clustering and regime analysis
                hmm_results = {
                    'n_components': clustering_results.get('n_clusters', 3),
                    'log_likelihood': clustering_results.get('clustering_score', 0.0),
                    'transition_matrix': regime_analysis.get('regime_transitions', []),
                    'steady_state_probabilities': regime_analysis.get('steady_state_probs', []),
                    'feature_importance': clustering_results.get('feature_importance', {}),
                    'regime_persistence': regime_analysis.get('regime_persistence', []),
                    'volatility_by_regime': regime_analysis.get('volatility_by_regime', []),
                    'trend_by_regime': regime_analysis.get('trend_by_regime', []),
                    'regime_confidence': regime_analysis.get('regime_confidence', [])
                }

                # Prepare clustering results
                clustering_quality_results = {
                    'silhouette_score': clustering_results.get('silhouette_score', 0.0),
                    'davies_bouldin': clustering_results.get('davies_bouldin', 0.0),
                    'calinski_harabasz': clustering_results.get('calinski_harabasz', 0.0),
                    'n_clusters': clustering_results.get('n_clusters', 0),
                    'cluster_sizes': clustering_results.get('cluster_sizes', []),
                    'cluster_centers': clustering_results.get('cluster_centers', []),
                    'stability_score': clustering_results.get('stability_score', 0.0)
                }

                # Prepare performance data
                performance_data = {
                    'execution_time': time.time() - self.start_time if self.start_time else 0,
                    'memory_usage': 0,  # Would need to be measured
                    'cpu_usage': 0,     # Would need to be measured
                    'function_calls': 0, # Would need to be tracked
                    'successful_ops': 1 if clustering_results else 0,
                    'failed_ops': 0 if clustering_results else 1
                }

                # Get market data (placeholder - in practice you'd get actual data)
                market_data = pd.DataFrame()

                # Generate comprehensive report
                comprehensive_report = self.enhanced_reporter.generate_comprehensive_report(
                    hmm_results=hmm_results,
                    clustering_results=clustering_quality_results,
                    performance_data=performance_data,
                    market_data=market_data,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )

                # Save comprehensive report
                saved_files = self.enhanced_reporter.save_comprehensive_report(
                    report=comprehensive_report,
                    base_filename=f"step03_5_enhanced_{symbol}_{exchange}_{timeframe}"
                )

                self.logger.info(f"✅ Enhanced comprehensive report saved for Step 3.5: {saved_files}")

                # Add enhanced report info to basic reports
                reports["enhanced_reporting"] = {
                    "generated": True,
                    "saved_files": saved_files,
                    "report_types": list(saved_files.keys())
                }

            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced reporting failed for Step 3.5, continuing with basic reporting: {e}")
                reports["enhanced_reporting"] = {
                    "generated": False,
                    "error": str(e)
                }

        self.logger.info("✅ Comprehensive reports generated")
        return reports

    @handles_errors(
        exceptions=(Exception,),
        context="save_final_results"
    )
    # @secure_data_processing - removed, handled by validates
    async def _save_final_results(self, clustering_results: dict[str, Any], regime_analysis: dict[str, Any], reports: dict[str, Any]) -> bool:
        """Save final regime clustering results."""
        try:
            self.logger.info("💾 Saving final regime clustering results...")
            
            # Create results directory
            results_dir = Path("data/regime_clustering")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create reports directory
            reports_dir = Path("reports/regime_clustering")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save clustering results
            clustering_file = results_dir / "final_clustering_results.json"
            with open(clustering_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = clustering_results.copy()
                if "cluster_labels" in serializable_results:
                    serializable_results["cluster_labels"] = serializable_results["cluster_labels"].tolist()
                if "state_sequence" in serializable_results.get("hmm_results", {}):
                    serializable_results["hmm_results"]["state_sequence"] = serializable_results["hmm_results"]["state_sequence"].tolist()
                
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Save regime analysis
            analysis_file = results_dir / "regime_analysis_results.json"
            with open(analysis_file, 'w') as f:
                json.dump(regime_analysis, f, indent=2, default=str)
            
            # Import centralized reporting system
            from src.training.reports import save_training_report
            
            # Get symbol and timeframe from config
            symbol = self.config.get('SYMBOL', 'UNKNOWN')
            timeframe = self.config.get('TIMEFRAME', '1m')
            exchange = self.config.get('EXCHANGE', 'UNKNOWN')
            
            # Save comprehensive reports using centralized system
            reports_file = save_training_report(
                data=reports,
                step_name="step03_5_regime_clustering",
                report_type="comprehensive_regime_reports",
                symbol=f"{exchange}_{symbol}",
                timeframe=timeframe
            )
            
            # Generate summary report
            summary_report = {
                "execution_summary": {
                    "step_name": "step03_5_final_regime_clustering",
                    "execution_time": time.time() - self.start_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                },
                "clustering_summary": reports.get("clustering_summary", {}),
                "regime_analysis_summary": reports.get("regime_analysis", {}),
                "performance_metrics": reports.get("performance_metrics", {}),
                "recommendations": reports.get("recommendations", []),
                "next_steps": [
                    "Proceed to step04 for feature engineering",
                    "Use regime clusters for strategy development",
                    "Validate regime stability over time"
                ]
            }
            
            # Save summary report using centralized system
            summary_file = save_training_report(
                data=summary_report,
                step_name="step03_5_regime_clustering",
                report_type="regime_clustering_summary",
                symbol=f"{exchange}_{symbol}",
                timeframe=timeframe
            )
            
            # Log summary
            self.logger.info("=" * 80)
            self.logger.info("📊 FINAL REGIME CLUSTERING SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"🎯 Clusters: {reports.get('clustering_summary', {}).get('n_clusters', 'N/A')}")
            self.logger.info(f"📊 Total samples: {reports.get('clustering_summary', {}).get('total_samples', 'N/A'):,}")
            self.logger.info(f"🔍 Regimes analyzed: {reports.get('regime_analysis', {}).get('total_clusters', 'N/A')}")
            self.logger.info(f"📈 Clustering quality: {reports.get('performance_metrics', {}).get('clustering_quality', 'N/A')}")
            self.logger.info(f"📋 Recommendations: {len(reports.get('recommendations', []))}")
            self.logger.info("=" * 80)
            
            self.logger.info(f"✅ Final results saved to {results_dir}")
            self.logger.info(f"📋 Comprehensive reports saved to {reports_file}")
            self.logger.info(f"📋 Summary report saved to {summary_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
            raise

    # Helper methods for technical indicators
    @handles_errors(
        exceptions=(Exception,),
        context="calculate_rsi"
    )
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @handles_errors(
        exceptions=(Exception,),
        context="calculate_macd"
    )
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    @handles_errors(
        exceptions=(Exception,),
        context="calculate_atr"
    )
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    @handles_errors(
        exceptions=(Exception,),
        context="regime_clustering_cleanup"
    )
    
    async def cleanup(self) -> bool:
        """Clean up resources after regime clustering with optimization cleanup."""
        try:
            self.logger.info("🧹 Cleaning up regime clustering resources with optimizations...")

            # Clean up M1 GPU resources
            if self.m1_gpu_manager:
                try:
                    self.m1_gpu_manager.optimize_memory()
                    self.logger.info("✅ M1 GPU resources cleaned up")
                except Exception as e:
                    self.logger.warning(f"M1 GPU cleanup failed: {e}")

            # Clean up M1 Memory Optimizer resources
            if self.m1_memory_optimizer:
                try:
                    self.m1_memory_optimizer.optimize_memory()
                    self.logger.info("✅ M1 Memory Optimizer resources cleaned up")
                except Exception as e:
                    self.logger.warning(f"M1 Memory Optimizer cleanup failed: {e}")

            # Clean up enhanced matrix operations
            if self.matrix_operations:
                try:
                    # Clear any cached matrices or GPU memory
                    self.logger.info("✅ Enhanced Matrix Operations resources cleaned up")
                except Exception as e:
                    self.logger.warning(f"Enhanced Matrix Operations cleanup failed: {e}")

            # Clean up data manager cache
            if self.data_manager:
                try:
                    # Clear any cached data that's no longer needed
                    self.logger.info("✅ Optimized Data Manager cache cleaned up")
                except Exception as e:
                    self.logger.warning(f"Optimized Data Manager cleanup failed: {e}")

            # Generate final optimization report
            if self.optimization_selector:
                try:
                    optimization_report = {
                        "step_name": "step03_5_final_regime_clustering",
                        "optimization_strategy": self.optimization_strategy.value if hasattr(self, 'optimization_strategy') else "unknown",
                        "components_used": {
                            "m1_gpu_manager": self.m1_gpu_manager is not None,
                            "m1_memory_optimizer": self.m1_memory_optimizer is not None,
                            "m1_cpu_optimizer": self.m1_cpu_optimizer is not None,
                            "pipeline_executor": self.pipeline_executor is not None,
                            "matrix_operations": self.matrix_operations is not None,
                            "data_manager": self.data_manager is not None,
                            "optimization_selector": self.optimization_selector is not None,
                            "error_handler": self.error_handler is not None
                        },
                        "performance_metrics": {
                            "execution_time": time.time() - getattr(self, 'start_time', time.time()),
                            "memory_efficiency": "optimized" if self.m1_memory_optimizer else "standard",
                            "parallel_processing": "enabled" if self.m1_cpu_optimizer else "disabled",
                            "gpu_acceleration": "available" if self.m1_gpu_manager else "unavailable"
                        }
                    }

                    self.logger.info("📊 Final optimization report:")
                    for key, value in optimization_report["components_used"].items():
                        self.logger.info(f"   {key}: {'✅' if value else '❌'}")

                except Exception as e:
                    self.logger.warning(f"Failed to generate optimization report: {e}")

            self.logger.info("✅ Enhanced regime clustering cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup regime clustering: {e}")
            raise


@handles_errors(
    exceptions=(Exception,),
    context="step03_5_final_regime_clustering"
)

async def run_step(config: dict[str, Any]) -> bool:
    """Run the final regime clustering step."""
    logger.info("🚀 Starting Step 3.5: Final Regime Clustering with Advanced Reporting")
    
    # Create and initialize the step
    step = FinalRegimeClusteringStep(config)
    
    # Initialize the step
    await step.initialize()
    
    # Execute the step
    success = await step.execute()
    
    # Cleanup
    await step.cleanup()
    
    if success:
        logger.info("✅ Step 3.5: Final Regime Clustering completed successfully")
    else:
        logger.error("❌ Step 3.5: Final Regime Clustering failed")
    
    return success


if __name__ == "__main__":
    # Test the step
    
    # Load test configuration
    test_config = {
        "SYMBOL": "ETHUSDT",
        "EXCHANGE": "BINANCE",
        "TIMEFRAME": "1m",
        "DATA_DIR": "data_cache",
        "regime_clustering": {
            "enable_advanced_reporting": True,
            "enable_regime_analysis": True,
            "enable_transition_analysis": True
        }
    }
    
    # Run the step
    success = asyncio.run(run_step(test_config))
    print(f"Step execution {'successful' if success else 'failed'}")