# src/training/steps/step6_feature_engineering.py

"""Step 6: Complete Feature Engineering with Standardized Data Quality Management."
This step creates comprehensive features including both basic and advanced features,
with regime-aware optimization after HMM regime discovery.
"""
import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from src.core.decorators import cached, circuit_breaker, handles_errors, log_call, log_execution_time, validates

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Common utilities
from src.utils.common_operations import ensure_directory, safe_json_dump

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "hashlib",
    "src.training.steps.vectorized_advanced_feature_engineering",
    "src.tactician.sr_breakout_predictor",
    "src.training.optimized_feature_selection_manager",
    "src.utils.training_pipeline_decorators",
    "src.utils.logger",
    "src.utils.error_handler",
    "src.utils.decorators",
    "src.utils.enhanced_mlflow_integration"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
vectorized_feature_engineering = PipelineStandards.safe_import("src.training.steps.vectorized_advanced_feature_engineering", None)
sr_breakout_predictor = PipelineStandards.safe_import("src.tactician.sr_breakout_predictor", None)
optimized_feature_selection = PipelineStandards.safe_import("src.training.optimized_feature_selection_manager", None)
training_pipeline_decorators = PipelineStandards.safe_import("src.utils.training_pipeline_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
error_handler = PipelineStandards.safe_import("src.utils.error_handler", None)
decorators = PipelineStandards.safe_import("src.utils.decorators", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
numpy = PipelineStandards.safe_import("numpy", None)
pandas = PipelineStandards.safe_import("pandas", None)

# Fallback functions if imports fail
def create_fallback_logger():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if training_pipeline_decorators is None:
    circuit_breaker_protection = create_fallback_decorator()
    debug_training_step = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    prevent_data_leakage = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_step_output = create_fallback_decorator()
    validate_step_prerequisites = create_fallback_decorator()
    monitor_feature_engineering = create_fallback_decorator()
else:
    circuit_breaker_protection = training_pipeline_decorators.circuit_breaker_protection
    debug_training_step = training_pipeline_decorators.debug_training_step
    memory_efficient = training_pipeline_decorators.memory_efficient
    prevent_data_leakage = training_pipeline_decorators.prevent_data_leakage
    quality_gate = training_pipeline_decorators.quality_gate
    resource_monitor = training_pipeline_decorators.resource_monitor
    secure_data_processing = training_pipeline_decorators.secure_data_processing
    validate_step_output = training_pipeline_decorators.validate_step_output
    validate_step_prerequisites = training_pipeline_decorators.validate_step_prerequisites
    monitor_feature_engineering = training_pipeline_decorators.monitor_feature_engineering

if error_handler is None:
    handle_errors = create_fallback_decorator()
else:
    handle_errors = error_handler.handle_errors

if decorators is None:
    guard_dataframe_nulls = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
else:
    guard_dataframe_nulls = decorators.guard_dataframe_nulls
    with_tracing_span = decorators.with_tracing_span

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_dataframe = lambda *args, **kwargs: "fallback_dataframe"
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: "fallback_dataframe"
    log_step_report = lambda *args, **kwargs: "fallback_report"
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: "fallback_artifact"
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_dataframe = enhanced_mlflow.log_step_dataframe
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_report = enhanced_mlflow.log_step_report
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name


@validates(
    required_directories=["data/training", "data/hmm_regimes"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy", "hashlib"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        "data_validation": {
            "check_negative_prices": True,
            "check_price_relationships": True,
            "max_missing_ratio": 0.1,
            "min_data_points": 100
        }
    },
    context="Complete Feature Engineering",
)
# @secure_data_processing - removed, handled by validates(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
# @prevent_data_leakage - removed, handled by validates
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@log_execution_time(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=10.0,
    auto_cleanup=True,
)
@cached(
    chunk_size=5000, streaming_processing=True, memory_pool=True, cleanup_frequency=5,
)
# @quality_gate - removed, handled by validates
    data_quality_threshold=0.9,
    feature_quality_threshold=0.8,
    model_quality_threshold=0.7,
    validation_checks=["data_integrity", "feature_quality", "feature_stability"],
)
@circuit_breaker(
    max_execution_time=7200,  # 2 hours
    max_memory_usage_gb=16.0,
    max_cpu_usage_percent=90.0,
    error_threshold=3,
    recovery_timeout=600,
)
@log_call(
    enable_debug_logging=True,
    save_intermediate_results=True,
    enable_profiling=True,
    debug_output_dir="debug_output/step06",
)
@monitor_feature_engineering(
    track_feature_importance=True,
    monitor_feature_correlations=True,
    track_feature_stability=True,
    save_feature_analysis=True,
)
@validates(
    output_validation_rules={
        # Align with regime-specific outputs under data/training/regime_features
        "required_files": ["regime_features"],  # presence of regime_features directory
        "required_columns": ["timestamp"],
        "min_rows": 100,  # per regime file checked separately in validator
        "max_missing_ratio": 0.20,
    },
    validation_timeout=600,
)
# @with_enhanced_mlflow_logging - removed, use traced"step6_feature_engineering")
@handles_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step6_feature_engineering",
)
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = None,
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Step 6: Complete Feature Engineering with Standardized Data Quality Management.
    
    This step creates comprehensive features including both basic and advanced features,
    with regime-aware optimization after HMM regime discovery.
    
    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = system_logger.getChild("Step6FeatureEngineering")
    
    # Use standardized path construction
    if data_dir is None:
        data_dir = pipeline_standards.build_path("processed_data", exchange, symbol)
    
    logger.info("=" * 80)
    logger.info("🚀 STEP 6: Complete Feature Engineering with Standardized Data Quality Management")
    logger.info("=" * 80)
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"🏢 Exchange: {exchange}")
    logger.info(f"📊 Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"🔄 Force rerun: {force_rerun}")
    logger.info("=" * 80)
    
    try:
        # Check for existing artifacts first
        logger.info("🔍 Checking for existing feature artifacts...")
        artifacts_exist = _check_feature_artifacts_exist(symbol, exchange, data_dir)
        
        if artifacts_exist and not force_rerun:
            logger.info("📦 Loading existing feature artifacts (use --force-rerun to regenerate)")
            return True

        if artifacts_exist and force_rerun:
            logger.info("🔄 Force rerun enabled - regenerating features")
            logger.info("🗑️  Existing artifacts will be overwritten")
        else:
            logger.info("🔧 No existing artifacts found - generating features")
            logger.info("🆕 Starting fresh feature engineering pipeline")

        # 1) Load unified data from step1_5
        logger.info("📊 Loading unified data from step1_5...")
        unified_data = await _load_unified_data(symbol, exchange, timeframe, data_dir)
        if unified_data is None or unified_data.empty:
            logger.error("❌ Failed to load unified data")
            return False

        # Note: Data validation is now handled by decorators (@validates, # @secure_data_processing - removed, handled by validates)
        logger.info("✅ Data validation passed (handled by decorators)")

        # 2) Load regime information from step03
        logger.info("📊 Loading regime information from step03...")
        regime_data = await _load_regime_data(symbol, exchange, timeframe)
        if regime_data is not None:
            logger.info(f"✅ Loaded regime data with {len(regime_data)} regimes")
        else:
            logger.warning("⚠️ No regime data found - proceeding without regime-aware features")

        # 3) Load labeled data from step05
        logger.info("📊 Loading labeled data from step05...")
        labeled_data = await _load_labeled_data(symbol, exchange, timeframe)
        if labeled_data is None or labeled_data.empty:
            logger.error("❌ Failed to load labeled data")
            return False

        # 4) Initialize vectorized feature engineering
        logger.info("🔧 Initializing vectorized feature engineering...")
        feature_engineer = VectorizedAdvancedFeatureEngineering(
            config={
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "enable_regime_aware_features": regime_data is not None,
                "enable_advanced_features": True,
                "enable_basic_features": True,
            }
        )

        # 5) Create comprehensive features
        logger.info("🔧 Creating comprehensive features...")
        features_result = await _create_comprehensive_features(
            unified_data, labeled_data, regime_data, feature_engineer, symbol, exchange, timeframe
        )
        
        if not features_result:
            logger.error("❌ Failed to create comprehensive features")
            return False

        # 6) Monitor feature generation
        logger.info("📊 Monitoring feature generation...")
        feature_stats = _monitor_feature_generation(features_result["features_full"])
        logger.info(f"📈 Feature generation stats: {feature_stats}")

        # 7) Save feature artifacts
        logger.info("💾 Saving feature artifacts...")
        save_success = await _save_feature_artifacts(features_result, symbol, exchange, timeframe, data_dir)
        
        if not save_success:
            logger.error("❌ Failed to save feature artifacts")
            return False

        logger.info("✅ Step 6: Complete Feature Engineering completed successfully")
        logger.info("📊 Features created with comprehensive engineering")
        logger.info("=" * 80)
        return True
            
    except Exception as e:
        logger.exception(f"❌ Unexpected error in Step 6: {e}")
        return False


def _monitor_feature_generation(features: pd.DataFrame) -> dict:
    """Monitor feature generation process."""
    stats = {
        "total_features": len(features.columns),
        "numeric_features": len(features.select_dtypes(include=[np.number]).columns),
        "missing_values": features.isnull().sum().sum(),
        "memory_usage_mb": features.memory_usage(deep=True).sum() / 1024 / 1024,
        "feature_categories": _categorize_features(features.columns),
        "data_shape": features.shape
    }
    
    system_logger.info(f"📊 Feature Generation Stats: {stats}")
    return stats


def _categorize_features(feature_columns: List[str]) -> dict:
    """Categorize features by type."""
    categories = {
        "price_features": [],
        "volume_features": [],
        "vwap_features": [],
        "technical_indicators": [],
        "statistical_features": [],
        "regime_features": [],
        "lagged_features": [],
        "rolling_features": [],
        "vectorized_features": [],
        "other_features": []
    }
    
    for col in feature_columns:
        if any(keyword in col.lower() for keyword in ["return", "price", "close", "open", "high", "low"]):
            categories["price_features"].append(col)
        elif "volume" in col.lower():
            categories["volume_features"].append(col)
        elif "vwap" in col.lower():
            categories["vwap_features"].append(col)
        elif any(keyword in col.lower() for keyword in ["sma", "ema", "rsi", "macd", "bb", "atr", "stoch"]):
            categories["technical_indicators"].append(col)
        elif any(keyword in col.lower() for keyword in ["mean", "std", "skew", "kurt", "zscore"]):
            categories["statistical_features"].append(col)
        elif "regime" in col.lower():
            categories["regime_features"].append(col)
        elif "lag" in col.lower():
            categories["lagged_features"].append(col)
        elif any(keyword in col.lower() for keyword in ["rolling", "window"]):
            categories["rolling_features"].append(col)
        elif "vectorized" in col.lower():
            categories["vectorized_features"].append(col)
        else:
            categories["other_features"].append(col)
    
    return {k: len(v) for k, v in categories.items()}


async def _load_unified_data(symbol: str, exchange: str, timeframe: str, data_dir: str) -> pd.DataFrame:
    """Load unified data from step1_5."""
    try:
        from src.training.steps.unified_data_loader import load_unified_data
        
        unified_data = await load_unified_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            columns=["timestamp", "open", "high", "low", "close", "volume", "exchange", "symbol", "timeframe"],
        )
        
        if unified_data is None or unified_data.empty:
            return None
            
        return unified_data
        
    except Exception as e:
        system_logger.error(f"Failed to load unified data: {e}")
        return None


async def _load_regime_data(symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
    """Load unified regime data with labels from step04/step08."""
    try:
        # Try to load unified regime dataset first (new approach)
        unified_regime_file = Path(f"data/training/{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet")
        
        if unified_regime_file.exists():
            regime_data = pd.read_parquet(unified_regime_file)
            system_logger.info(f"✅ Loaded unified regime dataset: {regime_data.shape}")
            system_logger.info(f"   Regime column: composite_cluster_id")
            system_logger.info(f"   Unique regimes: {regime_data['composite_cluster_id'].nunique()}")
            return regime_data
        
        # Fallback to old approach for backward compatibility
        regime_file = Path(f"data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet")
        
        if regime_file.exists():
            regime_data = pd.read_parquet(regime_file)
            system_logger.info(f"⚠️ Loaded legacy regime data: {regime_data.shape}")
            system_logger.info(f"   Note: Consider running step04/step08 for unified approach")
            return regime_data
        else:
            system_logger.warning(f"⚠️ No regime data found (neither unified nor legacy)")
            system_logger.warning(f"   Expected files:")
            system_logger.warning(f"     - {unified_regime_file}")
            system_logger.warning(f"     - {regime_file}")
            return None
            
    except Exception as e:
        system_logger.error(f"Failed to load regime data: {e}")
        return None


async def _load_labeled_data(symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
    """Load labeled data from step05."""
    try:
        labeled_file = Path(f"data/training/{exchange}_{symbol}_{timeframe}_labeled_data.parquet")
        
        if labeled_file.exists():
            labeled_data = pd.read_parquet(labeled_file)
            system_logger.info(f"Loaded labeled data: {labeled_data.shape}")
            return labeled_data
        else:
            system_logger.warning(f"Labeled file not found: {labeled_file}")
            return None
            
    except Exception as e:
        system_logger.error(f"Failed to load labeled data: {e}")
        return None


async def _create_comprehensive_features(
    unified_data: pd.DataFrame,
    labeled_data: pd.DataFrame,
    regime_data: pd.DataFrame,
    feature_engineer: VectorizedAdvancedFeatureEngineering,
    symbol: str,
    exchange: str,
    timeframe: str
) -> Dict[str, Any]:
    """Create comprehensive features using vectorized feature engineering."""
    
    try:
        # Create proper config for SR features
        config = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "sr_breakout_predictor": {
                "enable_detailed_reporting": True,
                "sr_proximity_threshold": 0.02,
                "breakout_confidence_threshold": 0.6,
                "sr_detection_method": "fractal",
                "min_sr_strength": 0.3,
                "max_sr_levels": 10,
                "enable_dbscan_clustering": True,
                "enable_enhanced_strength": True,
                "enable_sr_features": True,
                "replace_existing_sr": False
            }
        }
        
        # Merge data
        merged_data = unified_data.copy()
        
        if regime_data is not None:
            # Check if this is unified regime data or legacy regime data
            if "composite_cluster_id" in regime_data.columns:
                # Unified regime dataset - regime info is already included
                if "composite_cluster_id" not in merged_data.columns:
                    # Merge regime information from unified dataset
                    regime_columns = ["composite_cluster_id"]
                    if "timestamp" in regime_data.columns:
                        regime_columns.insert(0, "timestamp")
                    
                    merged_data = merged_data.merge(
                        regime_data[regime_columns], 
                        on="timestamp" if "timestamp" in regime_columns else merged_data.index,
                        how="left"
                    )
                    system_logger.info("✅ Merged regime data from unified dataset")
                else:
                    system_logger.info("✅ Regime data already present in unified dataset")
            else:
                # Legacy regime data - merge regime information
                regime_columns = ["regime"] if "regime" in regime_data.columns else []
                if "timestamp" in regime_data.columns:
                    regime_columns.insert(0, "timestamp")
                
                if regime_columns:
                    merged_data = merged_data.merge(
                        regime_data[regime_columns], 
                        on="timestamp", 
                        how="left"
                    )
                    system_logger.info("✅ Merged legacy regime data")
                else:
                    system_logger.warning("⚠️ No valid regime columns found in regime data")
        
        if labeled_data is not None:
            # Merge labeled data
            label_columns = [col for col in labeled_data.columns if col.startswith("label_")]
            if label_columns:
                merged_data = merged_data.merge(
                    labeled_data[["timestamp"] + label_columns], 
                    on="timestamp", 
                    how="left"
                )
                system_logger.info(f"✅ Merged labeled data with {len(label_columns)} label columns")
        
        # Create features using parallel processing
        system_logger.info("🔧 Creating features with parallel processing...")
        
        # Create basic features
        features_df = await _create_basic_features(merged_data)
        
        # Add technical indicators
        features_df = _add_technical_indicators(features_df)
        
        # Add statistical features
        features_df = _add_statistical_features(features_df)
        
        # Add lagged features
        features_df = _create_lagged_features(features_df)
        
        # Add rolling window features
        features_df = _create_rolling_window_features(features_df)
        
        # Add regime-aware features if regime data is available
        if regime_data is not None:
            features_df = _add_regime_aware_features(features_df, merged_data)
        
        # Add HMM feature enhancement if regime data is available
        if regime_data is not None:
            features_df = _enhance_hmm_features(features_df, regime_data)
        
        # Add comprehensive S/R features using centralized logic
        if config.get("sr_breakout_predictor", {}).get("enable_sr_features", True):
            features_df = await _add_sr_features(features_df, merged_data, config)
        else:
            system_logger.info("⏭️ Skipping SR feature generation (disabled in config)")
        
        # Add SR-aware feature selection
        features_df = await _add_sr_aware_feature_selection(features_df, merged_data, config)
        
        # Add SR detection optimization features
        features_df = await _add_sr_optimization_features(features_df, merged_data, config)
        
        # Better integration with vectorized advanced features
        features_df = await _enhanced_integration_with_vectorized_features(features_df, feature_engineer, symbol, exchange, timeframe)
        
        # Validate and clean features
        features_df = _validate_and_clean_features(features_df)
        
        # Split into train/validation
        split_point = int(len(features_df) * 0.8)
        features_train = features_df.iloc[:split_point]
        features_val = features_df.iloc[split_point:]
        
        return {
            "features_train": features_train,
            "features_val": features_val,
            "features_full": features_df,
            "metadata": {
                "total_features": len(features_df.columns),
                "train_samples": len(features_train),
                "val_samples": len(features_val),
                "feature_columns": list(features_df.columns),
                "regime_aware": regime_data is not None,
                "sr_features_enabled": config.get("sr_breakout_predictor", {}).get("enable_sr_features", True),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        system_logger.error(f"Failed to create comprehensive features: {e}")
        return None


def _create_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create basic features."""
    features = data.copy()
    
    # Price-based features
    features["returns"] = data["close"].pct_change()
    features["log_returns"] = np.log(data["close"] / data["close"].shift(1))
    features["price_range"] = (data["high"] - data["low"]) / data["close"]
    features["body_size"] = abs(data["close"] - data["open"]) / data["close"]
    
    # VWAP calculation and VWAP-based features
    features["vwap"] = (data["close"] * data["volume"]).rolling(window=20).sum() / data["volume"].rolling(window=20).sum()
    features["vwap_returns"] = features["vwap"].pct_change()
    features["vwap_log_returns"] = np.log(features["vwap"] / features["vwap"].shift(1))
    
    # VWAP vs Price features
    features["price_vwap_ratio"] = data["close"] / features["vwap"]
    features["price_vwap_deviation"] = (data["close"] - features["vwap"]) / features["vwap"]
    features["price_vwap_spread"] = data["close"] - features["vwap"]
    
    # VWAP momentum features
    features["vwap_momentum_5"] = features["vwap"] / features["vwap"].shift(5) - 1
    features["vwap_momentum_10"] = features["vwap"] / features["vwap"].shift(10) - 1
    features["vwap_momentum_20"] = features["vwap"] / features["vwap"].shift(20) - 1
    
    # VWAP acceleration features
    features["vwap_acceleration_5"] = features["vwap_momentum_5"] - features["vwap_momentum_5"].shift(5)
    features["vwap_acceleration_10"] = features["vwap_momentum_10"] - features["vwap_momentum_10"].shift(10)
    features["vwap_acceleration_20"] = features["vwap_momentum_20"] - features["vwap_momentum_20"].shift(20)
    
    # VWAP volatility features
    features["vwap_volatility_5"] = features["vwap_returns"].rolling(window=5).std()
    features["vwap_volatility_10"] = features["vwap_returns"].rolling(window=10).std()
    features["vwap_volatility_20"] = features["vwap_returns"].rolling(window=20).std()
    
    # VWAP momentum volatility features
    features["vwap_momentum_volatility_5"] = features["vwap_momentum_5"].rolling(window=5).std()
    features["vwap_momentum_volatility_10"] = features["vwap_momentum_10"].rolling(window=10).std()
    features["vwap_momentum_volatility_20"] = features["vwap_momentum_20"].rolling(window=20).std()
    
    # Volume features
    features["volume_ratio"] = data["volume"] / data["volume"].rolling(window=20).mean()
    features["volume_std"] = data["volume"].rolling(window=20).std()
    
    # Volatility features
    features["volatility"] = features["returns"].rolling(window=20).std()
    features["volatility_ratio"] = features["volatility"] / features["volatility"].rolling(window=50).mean()
    
    return features


def _create_lagged_features(features: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Create lagged versions of important features."""
    important_features = ["returns", "volume_ratio", "volatility", "rsi"]
    
    # Add VWAP-based features to important features list
    if "vwap_returns" in features.columns:
        important_features.extend(["vwap_returns", "vwap_momentum_20", "vwap_volatility_20"])
    if "price_vwap_ratio" in features.columns:
        important_features.extend(["price_vwap_ratio", "price_vwap_deviation"])
    
    for feature in important_features:
        if feature in features.columns:
            for lag in lags:
                features[f"{feature}_lag_{lag}"] = features[feature].shift(lag)
    
    system_logger.info(f"✅ Created {len(important_features) * len(lags)} lagged features")
    return features


def _create_rolling_window_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create advanced rolling window features."""
    windows = [5, 10, 20, 50]
    
    for window in windows:
        # Rolling statistics
        features[f"returns_skew_{window}"] = features["returns"].rolling(window).skew()
        features[f"returns_kurt_{window}"] = features["returns"].rolling(window).kurt()
        
        # Rolling quantiles
        features[f"returns_q25_{window}"] = features["returns"].rolling(window).quantile(0.25)
        features[f"returns_q75_{window}"] = features["returns"].rolling(window).quantile(0.75)
        
        # Rolling extremes
        features[f"returns_max_{window}"] = features["returns"].rolling(window).max()
        features[f"returns_min_{window}"] = features["returns"].rolling(window).min()
        
        # Volume rolling features
        if "volume_ratio" in features.columns:
            features[f"volume_skew_{window}"] = features["volume_ratio"].rolling(window).skew()
            features[f"volume_kurt_{window}"] = features["volume_ratio"].rolling(window).kurt()
    
    # VWAP-based rolling window features
    if "vwap_returns" in features.columns:
        for window in windows:
            # VWAP returns rolling statistics
            features[f"vwap_returns_skew_{window}"] = features["vwap_returns"].rolling(window).skew()
            features[f"vwap_returns_kurt_{window}"] = features["vwap_returns"].rolling(window).kurt()
            features[f"vwap_returns_q25_{window}"] = features["vwap_returns"].rolling(window).quantile(0.25)
            features[f"vwap_returns_q75_{window}"] = features["vwap_returns"].rolling(window).quantile(0.75)
            features[f"vwap_returns_max_{window}"] = features["vwap_returns"].rolling(window).max()
            features[f"vwap_returns_min_{window}"] = features["vwap_returns"].rolling(window).min()
    
    # VWAP momentum rolling window features
    if "vwap_momentum_20" in features.columns:
        for window in [5, 10, 20]:
            features[f"vwap_momentum_skew_{window}"] = features["vwap_momentum_20"].rolling(window).skew()
            features[f"vwap_momentum_kurt_{window}"] = features["vwap_momentum_20"].rolling(window).kurt()
            features[f"vwap_momentum_q25_{window}"] = features["vwap_momentum_20"].rolling(window).quantile(0.25)
            features[f"vwap_momentum_q75_{window}"] = features["vwap_momentum_20"].rolling(window).quantile(0.75)
    
    system_logger.info(f"✅ Created {len(windows) * 7 + (len(windows) * 6 if 'vwap_returns' in features.columns else 0) + (3 * 5 if 'vwap_momentum_20' in features.columns else 0)} rolling window features")
    return features


def _add_regime_aware_features(features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Add regime-aware features."""
    if "regime" not in data.columns:
        return features
    
    # Regime-specific features
    features["regime"] = data["regime"]
    features["regime_change"] = data["regime"].diff()
    features["regime_duration"] = data["regime"].groupby((data["regime"] != data["regime"].shift()).cumsum()).cumcount()
    
    # Regime-specific statistics
    for regime in data["regime"].unique():
        if pd.notna(regime):
            regime_mask = data["regime"] == regime
            features[f"regime_{regime}_returns_mean"] = features["returns"].where(regime_mask).rolling(20).mean()
            features[f"regime_{regime}_volatility_mean"] = features["volatility"].where(regime_mask).rolling(20).mean()
            
            # VWAP-based regime features
            if "vwap_returns" in features.columns:
                features[f"regime_{regime}_vwap_returns_mean"] = features["vwap_returns"].where(regime_mask).rolling(20).mean()
                features[f"regime_{regime}_vwap_volatility_mean"] = features["vwap_volatility_20"].where(regime_mask).rolling(20).mean()
                features[f"regime_{regime}_vwap_momentum_mean"] = features["vwap_momentum_20"].where(regime_mask).rolling(20).mean()
    
    return features


def _enhance_hmm_features(features: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    """Enhance features with HMM feature enhancer."""
    try:
        from src.training.steps.hmm_feature_enhancer import HMMFeatureEnhancer
        
        # Initialize HMM feature enhancer
        enhancer = HMMFeatureEnhancer()
        
        # Merge regime data with features for enhancement
        enhanced_features = features.copy()
        
        # Add regime information if not already present
        if "composite_cluster_id" not in enhanced_features.columns and "regime" in regime_data.columns:
            enhanced_features = enhanced_features.merge(
                regime_data[["timestamp", "regime"]].rename(columns={"regime": "composite_cluster_id"}),
                on="timestamp",
                how="left"
            )
        
        # Enhance features with HMM feature enhancer
        enhanced_features = enhancer.enhance_hmm_features(enhanced_features)
        
        system_logger.info(f"✅ Enhanced features with HMM feature enhancer: {len(enhanced_features.columns)} total features")
        return enhanced_features
        
    except Exception as e:
        system_logger.error(f"Failed to enhance HMM features: {e}")
        return features


def _add_technical_indicators(features: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators."""
    # Moving averages
    features["sma_20"] = features["close"].rolling(window=20).mean()
    features["sma_50"] = features["close"].rolling(window=50).mean()
    features["ema_12"] = features["close"].ewm(span=12).mean()
    features["ema_26"] = features["close"].ewm(span=26).mean()
    
    # VWAP-based moving averages
    if "vwap" in features.columns:
        features["vwap_sma_20"] = features["vwap"].rolling(window=20).mean()
        features["vwap_sma_50"] = features["vwap"].rolling(window=50).mean()
        features["vwap_ema_12"] = features["vwap"].ewm(span=12).mean()
        features["vwap_ema_26"] = features["vwap"].ewm(span=26).mean()
        
        # VWAP Bollinger Bands
        vwap_bb_middle = features["vwap"].rolling(window=20).mean()
        vwap_bb_std = features["vwap"].rolling(window=20).std()
        features["vwap_bb_upper"] = vwap_bb_middle + (vwap_bb_std * 2)
        features["vwap_bb_lower"] = vwap_bb_middle - (vwap_bb_std * 2)
        features["vwap_bb_width"] = features["vwap_bb_upper"] - features["vwap_bb_lower"]
        features["vwap_bb_position"] = (features["vwap"] - features["vwap_bb_lower"]) / features["vwap_bb_width"]
        
        # VWAP RSI
        vwap_delta = features["vwap"].diff()
        vwap_gain = (vwap_delta.where(vwap_delta > 0, 0)).rolling(window=14).mean()
        vwap_loss = (-vwap_delta.where(vwap_delta < 0, 0)).rolling(window=14).mean()
        vwap_rs = vwap_gain / vwap_loss
        features["vwap_rsi"] = 100 - (100 / (1 + vwap_rs))
        
        # VWAP MACD
        features["vwap_macd"] = features["vwap_ema_12"] - features["vwap_ema_26"]
        features["vwap_macd_signal"] = features["vwap_macd"].ewm(span=9).mean()
        features["vwap_macd_histogram"] = features["vwap_macd"] - features["vwap_macd_signal"]
    
    # RSI
    delta = features["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD
    features["macd"] = features["ema_12"] - features["ema_26"]
    features["macd_signal"] = features["macd"].ewm(span=9).mean()
    features["macd_histogram"] = features["macd"] - features["macd_signal"]
    
    # Bollinger Bands
    features["bb_middle"] = features["close"].rolling(window=20).mean()
    bb_std = features["close"].rolling(window=20).std()
    features["bb_upper"] = features["bb_middle"] + (bb_std * 2)
    features["bb_lower"] = features["bb_middle"] - (bb_std * 2)
    features["bb_width"] = features["bb_upper"] - features["bb_lower"]
    features["bb_position"] = (features["close"] - features["bb_lower"]) / features["bb_width"]
    
    return features


def _add_statistical_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add statistical features."""
    # Rolling statistics
    for window in [5, 10, 20, 50]:
        features[f"returns_mean_{window}"] = features["returns"].rolling(window=window).mean()
        features[f"returns_std_{window}"] = features["returns"].rolling(window=window).std()
        features[f"returns_skew_{window}"] = features["returns"].rolling(window=window).skew()
        features[f"returns_kurt_{window}"] = features["returns"].rolling(window=window).kurt()
    
    # VWAP-based rolling statistics
    if "vwap_returns" in features.columns:
        for window in [5, 10, 20, 50]:
            features[f"vwap_returns_mean_{window}"] = features["vwap_returns"].rolling(window=window).mean()
            features[f"vwap_returns_std_{window}"] = features["vwap_returns"].rolling(window=window).std()
            features[f"vwap_returns_skew_{window}"] = features["vwap_returns"].rolling(window=window).skew()
            features[f"vwap_returns_kurt_{window}"] = features["vwap_returns"].rolling(window=window).kurt()
    
    # VWAP momentum rolling statistics
    if "vwap_momentum_20" in features.columns:
        for window in [5, 10, 20]:
            features[f"vwap_momentum_mean_{window}"] = features["vwap_momentum_20"].rolling(window=window).mean()
            features[f"vwap_momentum_std_{window}"] = features["vwap_momentum_20"].rolling(window=window).std()
            features[f"vwap_momentum_skew_{window}"] = features["vwap_momentum_20"].rolling(window=window).skew()
            features[f"vwap_momentum_kurt_{window}"] = features["vwap_momentum_20"].rolling(window=window).kurt()
    
    # Z-score features
    features["returns_zscore"] = (features["returns"] - features["returns"].rolling(20).mean()) / features["returns"].rolling(20).std()
    features["volume_zscore"] = (features["volume"] - features["volume"].rolling(20).mean()) / features["volume"].rolling(20).std()
    
    # VWAP Z-score features
    if "vwap_returns" in features.columns:
        features["vwap_returns_zscore"] = (features["vwap_returns"] - features["vwap_returns"].rolling(20).mean()) / features["vwap_returns"].rolling(20).std()
        features["vwap_momentum_zscore"] = (features["vwap_momentum_20"] - features["vwap_momentum_20"].rolling(20).mean()) / features["vwap_momentum_20"].rolling(20).std()
    
    return features


async def _add_sr_features(
    features: pd.DataFrame, 
    market_data: pd.DataFrame,
    config: dict[str, Any]
) -> pd.DataFrame:
    """Add comprehensive S/R features using all features from SR breakout predictor."""
    try:
        # Check for existing SR features to avoid redundancy
        existing_sr_features = [col for col in features.columns if any(keyword in col.lower() for keyword in [
            "sr_", "support", "resistance", "pivot", "breakout", "proximity"
        ])]
        
        if existing_sr_features:
            system_logger.info(f"⚠️ Found {len(existing_sr_features)} existing SR features, will enhance rather than replace")
            system_logger.info(f"   Existing features: {existing_sr_features[:5]}...")
        
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
        
        # Initialize S/R predictor with optimized parameters
        sr_config = config.copy()
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        sr_predictor = SRBreakoutPredictor(sr_config)
        await sr_predictor.initialize()
        
        # Get comprehensive S/R context with all advanced features
        current_price = market_data['close'].iloc[-1]
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        
        # Calculate comprehensive S/R features using all available methods
        sr_features = await sr_predictor.calculate_comprehensive_sr_features(market_data)
        
        # Add all S/R context features including advanced analysis
        context_features = {
            # Basic SR features
            "sr_support_proximity": sr_context.get("support_proximity", 1.0),
            "sr_resistance_proximity": sr_context.get("resistance_proximity", 1.0),
            "sr_support_strength": sr_context.get("support_strength", 0.5),
            "sr_resistance_strength": sr_context.get("resistance_strength", 0.5),
            "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
            "sr_nearest_support": sr_context.get("nearest_support", current_price),
            "sr_nearest_resistance": sr_context.get("nearest_resistance", current_price),
            "sr_total_support_levels": len(sr_context.get("support_levels", [])),
            "sr_total_resistance_levels": len(sr_context.get("resistance_levels", [])),
            
            # Enhanced strength features
            "sr_enhanced_support_strength": np.mean([level.get("enhanced_strength", 0.5) for level in sr_context.get("support_levels", [])]) if sr_context.get("support_levels") else 0.5,
            "sr_enhanced_resistance_strength": np.mean([level.get("enhanced_strength", 0.5) for level in sr_context.get("resistance_levels", [])]) if sr_context.get("resistance_levels") else 0.5,
            
            # Clustering features
            "sr_clusters_detected": sr_context.get("clustering_result", {}).get("n_clusters", 0),
            "sr_noise_points": sr_context.get("clustering_result", {}).get("noise_points", 0),
            "sr_clustering_quality": 1.0 if sr_context.get("clustering_result", {}).get("n_clusters", 0) > 0 else 0.0,
            
            # Advanced analysis features
            "sr_fibonacci_levels": len(sr_context.get("fibonacci_levels", {})),
            "sr_elliott_waves": len(sr_context.get("elliott_wave_levels", {}).get("wave_levels", {})),
            "sr_order_flow_poc": 1.0 if sr_context.get("order_flow_analysis", {}).get("poc") else 0.0,
            "sr_order_flow_hvns": len(sr_context.get("order_flow_analysis", {}).get("volume_profile", {}).get("high_volume_nodes", [])),
            "sr_order_flow_imbalances": len(sr_context.get("order_flow_analysis", {}).get("imbalances", [])),
        }
        
        # Add pivot levels features (as percentages relative to current price)
        pivot_levels = sr_context.get("pivot_levels", {})
        if pivot_levels and current_price > 0:
            context_features.update({
                "sr_pivot_level_pct": (pivot_levels.get("pivot", current_price) - current_price) / current_price,
                "sr_support_1_pct": (pivot_levels.get("s1", current_price) - current_price) / current_price,
                "sr_support_2_pct": (pivot_levels.get("s2", current_price) - current_price) / current_price,
                "sr_resistance_1_pct": (pivot_levels.get("r1", current_price) - current_price) / current_price,
                "sr_resistance_2_pct": (pivot_levels.get("r2", current_price) - current_price) / current_price,
            })
        
        # Add all features to DataFrame with conflict resolution
        all_sr_features = {**sr_features, **context_features}
        features_added = 0
        
        for feature_name, feature_value in all_sr_features.items():
            new_feature_name = f"sr_{feature_name}"
            
            # Check if feature already exists
            if new_feature_name in features.columns:
                if config.get("sr_breakout_predictor", {}).get("replace_existing_sr", False):
                    system_logger.info(f"🔄 Replacing existing feature: {new_feature_name}")
                else:
                    system_logger.warning(f"⚠️ Feature {new_feature_name} already exists, skipping")
                    continue
                
            if isinstance(feature_value, pd.Series) and len(feature_value) == len(features):
                features[new_feature_name] = feature_value
                features_added += 1
            elif isinstance(feature_value, (int, float)):
                # If it's a scalar, broadcast to all rows'
                features[new_feature_name] = feature_value
                features_added += 1
        
        system_logger.info(f"✅ Added {features_added} new S/R features (avoided {len(existing_sr_features)} existing)")
        
        # Generate detailed report if enabled
        if sr_predictor.reporting_enabled:
            await sr_predictor.generate_manual_report(market_data, sr_context)
        
        # Cleanup
        await sr_predictor.cleanup()
        
        return features
        
    except Exception as e:
        system_logger.warning(f"S/R feature integration failed: {e}")
        return features

async def _add_sr_aware_feature_selection(
    features: pd.DataFrame,
    market_data: pd.DataFrame,
    config: dict[str, Any]
) -> pd.DataFrame:
    """Add SR-aware feature selection and engineering."""
    try:
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
        
        # Initialize SRBreakoutPredictor with optimized parameters
        sr_config = config.copy()
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        sr_predictor = SRBreakoutPredictor(sr_config)
        await sr_predictor.initialize()
        
        # Get SR context for feature selection
        current_price = market_data['close'].iloc[-1]
        sr_context = await sr_predictor.get_sr_context(market_data, current_price)
        
        # Create SR-aware features based on proximity
        support_proximity = sr_context.get("support_proximity", 1.0)
        resistance_proximity = sr_context.get("resistance_proximity", 1.0)
        
        # Add proximity-based feature weights (using percentages)
        features["sr_proximity_weight"] = 1.0 / (1.0 + min(support_proximity, resistance_proximity))
        
        # Add SR strength features (already as percentages/ratios)
        features["sr_combined_strength"] = (
            sr_context.get("support_strength", 0.5) + 
            sr_context.get("resistance_strength", 0.5)
        ) / 2
        
        # Add SR zone features (using percentages)
        sr_zone_width = sr_context.get("sr_zone_width", 0.0)
        if sr_zone_width > 0 and current_price > 0:
            zone_position_pct = (current_price - sr_context.get("nearest_support", current_price)) / current_price / sr_zone_width
        else:
            zone_position_pct = 0.5
        features["sr_zone_position_pct"] = zone_position_pct
        
        # Add SR momentum features (as percentage returns)
        features["sr_momentum_pct"] = market_data['close'].pct_change().iloc[-5:].mean()
        
        # Add SR volatility features (as percentage returns)
        features["sr_volatility_pct"] = market_data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Add SR volume features (as ratio/percentage)
        features["sr_volume_ratio"] = market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
        
        # Add SR trend features (as percentage change)
        features["sr_trend_pct"] = (market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20]
        
        await sr_predictor.cleanup()
        system_logger.info("✅ Added SR-aware feature selection features")
        return features
        
    except Exception as e:
        system_logger.warning(f"SR-aware feature selection failed: {e}")
        return features

async def _add_sr_optimization_features(
    features: pd.DataFrame,
    market_data: pd.DataFrame,
    config: dict[str, Any]
) -> pd.DataFrame:
    """Add SR detection optimization features using all optimization capabilities."""
    try:
        from src.tactician.sr_detection_optimization import setup_sr_detection_optimizer
    except Exception as e:
        pass  # TODO: Handle exception properly
import copy
import numpy as np
import pandas as pd
        
# Initialize SR detection optimizer
optimizer = await setup_sr_detection_optimizer(config)
if not optimizer:
            system_logger.warning("⚠️ SR detection optimizer not available, skipping optimization features")
            return features
        
        # Get optimized parameters if available
        optimized_params = optimizer.get_optimized_parameters()
        if optimized_params:
            system_logger.info("✅ Using optimized SR parameters")
            
            # Add optimization-based features
            features["sr_optimized_method_weights"] = np.mean(list(optimized_params.get("method_weights", {}).values()))
            features["sr_optimized_strength_weights"] = np.mean(list(optimized_params.get("strength_weights", {}).values()))
            features["sr_optimized_dbscan_eps"] = optimized_params.get("dbscan_params", {}).get("eps", 0.01)
            features["sr_optimized_dbscan_min_samples"] = optimized_params.get("dbscan_params", {}).get("min_samples", 3)
            features["sr_optimized_fibonacci_sensitivity"] = optimized_params.get("advanced_params", {}).get("fibonacci_sensitivity", 0.7)
            features["sr_optimized_elliott_confidence"] = optimized_params.get("advanced_params", {}).get("elliott_confidence_threshold", 0.6)
            features["sr_optimized_order_flow_threshold"] = optimized_params.get("advanced_params", {}).get("order_flow_hvn_threshold", 1.5)
            
            # Add timeframe optimization features
            timeframe_weights = optimized_params.get("timeframe_weights", {})
            for tf, weight in timeframe_weights.items():
                features[f"sr_optimized_tf_{tf}_weight"] = weight
        else:
            system_logger.info("ℹ️ No optimized parameters available, using default values")
            # Add default optimization features
            features["sr_optimized_method_weights"] = 0.25  # Default average
            features["sr_optimized_strength_weights"] = 0.2  # Default average
            features["sr_optimized_dbscan_eps"] = 0.01
            features["sr_optimized_dbscan_min_samples"] = 3
            features["sr_optimized_fibonacci_sensitivity"] = 0.7
            features["sr_optimized_elliott_confidence"] = 0.6
            features["sr_optimized_order_flow_threshold"] = 1.5
        
        # Add optimization score if available (keeping only the main optimization score)
        if hasattr(optimizer, 'best_result') and optimizer.best_result:
            features["sr_optimization_score"] = optimizer.best_result.optimization_score
        else:
            # Add default optimization score
            features["sr_optimization_score"] = 0.5
        
        system_logger.info("✅ Added SR optimization features")
        return features
        
    except Exception as e:
        system_logger.warning(f"SR optimization feature integration failed: {e}")
        return features


async def _enhanced_integration_with_vectorized_features(
    features: pd.DataFrame, 
    feature_engineer: VectorizedAdvancedFeatureEngineering,
    symbol: str,
    exchange: str,
    timeframe: str
) -> pd.DataFrame:
    """Better integration with vectorized advanced feature engineering."""
    try:
        # Initialize vectorized feature engineering with more configuration
        vectorized_config = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "enable_regime_aware_features": "regime" in features.columns,
            "enable_advanced_features": True,
            "enable_basic_features": False,  # Already done in step06
            "feature_engineering_parameters": {
                "enable_wavelet_transforms": True,
                "enable_multi_timeframe": True,
                "enable_profit_based_features": True
            }
        }
        
        # Initialize the feature engineer
        await feature_engineer.initialize()
        
        # Get additional features
        additional_features = await feature_engineer.engineer_features(
            price_data=features[["open", "high", "low", "close"]],
            volume_data=features[["volume"]]
        )
        
        # Merge additional features
        for key, value in additional_features.items():
            if isinstance(value, pd.Series):
                features[f"vectorized_{key}"] = value
        
        system_logger.info(f"✅ Integrated {len(additional_features)} vectorized features")
        return features
        
    except Exception as e:
        system_logger.warning(f"Vectorized feature integration failed: {e}")
        return features


def _validate_and_clean_features(features: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean features."""
    # Remove constant features
    constant_features = features.columns[features.nunique() <= 1]
    if len(constant_features) > 0:
        features = features.drop(columns=constant_features)
        system_logger.info(f"🗑️ Removed {len(constant_features)} constant features")
    
    # Remove highly correlated features
    correlation_matrix = features.select_dtypes(include=[np.number]).corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_features = [column for column in upper_tri.columns 
    if any(upper_tri[column] > 0.95)]
    if len(high_corr_features) > 0:
        features = features.drop(columns=high_corr_features)
        system_logger.info(f"🗑️ Removed {len(high_corr_features)} highly correlated features")
    
    # Handle infinite values
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaN values
    features = features.fillna(method="ffill").fillna(method="bfill").fillna(0)
    
    system_logger.info(f"✅ Feature validation and cleaning completed. Final shape: {features.shape}")
    return features


async def _save_feature_artifacts(
    features_result: Dict[str, Any], 
    symbol: str, 
    exchange: str, 
    timeframe: str, 
    data_dir: str
) -> bool:
    """Save feature artifacts."""
    try:
        output_dir = ensure_directory("data/training")
        
        # Save feature files
        train_file_path = output_dir / f"{exchange}_{symbol}_{timeframe}_features_train.parquet"
        val_file_path = output_dir / f"{exchange}_{symbol}_{timeframe}_features_val.parquet"
        metadata_file_path = output_dir / f"{exchange}_{symbol}_{timeframe}_feature_metadata.json"
        
        features_result["features_train"].to_parquet(train_file_path)
        features_result["features_val"].to_parquet(val_file_path)
        
        # Save metadata
        safe_json_dump(features_result["metadata"], metadata_file_path, indent=2, default=str)
        
        # Log artifacts to MLflow with standardized naming
        try:
            # Create a config dict for MLflow logging
            config = {
                "trading_symbol": symbol,
                "exchange_name": exchange,
                "lookback_years": 2,  # Default value
            }
            
            # Log training features DataFrame with standardized naming
            train_artifact_name = log_step_dataframe_with_standardized_name(
                config=config,
                step_name="step6_feature_engineering",
                df=features_result["features_train"],
                artifact_type="features_train",
                additional_metadata={
                    "artifact_type": "training_features",
                    "feature_count": len(features_result["features_train"].columns),
                    "sample_count": len(features_result["features_train"]),
                    "timeframe": timeframe,
                }
            )
            system_logger.info(f"✅ Logged training features: {train_artifact_name}")
            
            # Log validation features DataFrame with standardized naming
            val_artifact_name = log_step_dataframe_with_standardized_name(
                config=config,
                step_name="step6_feature_engineering",
                df=features_result["features_val"],
                artifact_type="features_val",
                additional_metadata={
                    "artifact_type": "validation_features",
                    "feature_count": len(features_result["features_val"].columns),
                    "sample_count": len(features_result["features_val"]),
                    "timeframe": timeframe,
                }
            )
            system_logger.info(f"✅ Logged validation features: {val_artifact_name}")
            
            # Log feature metadata with standardized naming
            metadata_artifact_name = log_step_artifact_with_standardized_name(
                config=config,
                step_name="step6_feature_engineering",
                artifact_path=str(metadata_file_path),
                artifact_type="feature_metadata",
                additional_metadata={
                    "metadata_keys": list(features_result["metadata"].keys()),
                    "feature_count": features_result["metadata"].get("feature_count", 0),
                    "timeframe": timeframe,
                }
            )
            system_logger.info(f"✅ Logged feature metadata: {metadata_artifact_name}")
            
            # Log feature engineering report
            report_data = {
                "feature_engineering_summary": {
                    "total_features": len(features_result["features_train"].columns),
                    "training_samples": len(features_result["features_train"]),
                    "validation_samples": len(features_result["features_val"]),
                    "feature_categories": features_result["metadata"].get("feature_categories", {}),
                    "feature_importance": features_result["metadata"].get("feature_importance", {}),
                },
                "metadata": features_result["metadata"],
                "training_input": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                },
                "execution_timestamp": datetime.now().isoformat(),
            }
            
            report_name = log_step_report(
                config=config,
                step_name="step6_feature_engineering",
                report_data=report_data,
                report_type="feature_engineering_report",
                additional_metadata={
                    "total_features": len(features_result["features_train"].columns),
                    "feature_categories": len(features_result["metadata"].get("feature_categories", {})),
                    "timeframe": timeframe,
                }
            )
            system_logger.info(f"✅ Logged feature engineering report: {report_name}")
            
            # Log feature engineering metrics
            if "metadata" in features_result and "metrics" in features_result["metadata"]:
                metrics = features_result["metadata"]["metrics"]
                numeric_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        numeric_metrics[f"step6_{key}"] = float(value)
                
                if numeric_metrics:
                    log_step_metrics(
                        config=config,
                        step_name="step6_feature_engineering",
                        metrics=numeric_metrics,
                        additional_metadata={
                            "metrics_type": "feature_engineering",
                            "feature_count": len(features_result["features_train"].columns),
                            "timeframe": timeframe,
                        }
                    )
            
            system_logger.info("✅ Feature artifacts logged to MLflow with standardized naming successfully")
            
        except Exception as e:
            system_logger.warning(f"⚠️ MLflow logging failed for step 6: {e}")
            # Don't fail the step if MLflow logging fails'
        
        system_logger.info(f"✅ Saved feature artifacts to {output_dir}")
        return True
        
    except Exception as e:
        system_logger.error(f"Failed to save feature artifacts: {e}")
        return False


def _check_feature_artifacts_exist(symbol: str, exchange: str, data_dir: str) -> bool:
    """Check if feature artifacts already exist."""
    try:
        output_dir = Path("data/training")
        
        train_file = output_dir / f"{exchange}_{symbol}_1m_features_train.parquet"
        val_file = output_dir / f"{exchange}_{symbol}_1m_features_val.parquet"
        metadata_file = output_dir / f"{exchange}_{symbol}_1m_feature_metadata.json"
        
        return train_file.exists() and val_file.exists() and metadata_file.exists()
        
    except Exception:
        return False


# Export the main function for external use
__all__ = ["run_step"]