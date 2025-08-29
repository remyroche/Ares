# src/training/steps/step6_feature_engineering.py

"""Step 6: Complete Feature Engineering (Simple + Advanced).
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

import numpy as np
import pandas as pd

# Import vectorized advanced feature engineering
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)

# Import SR breakout predictor for comprehensive SR features
from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor

# Import optimized feature selection manager
from src.training.optimized_feature_selection_manager import (
    OptimizedFeatureSelectionManager,
)

# Import training pipeline decorators
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection,
    debug_training_step,
    memory_efficient,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_step_output,
    validate_step_prerequisites,
    monitor_feature_engineering,
)

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span


@validate_step_prerequisites(
    required_directories=["data/training", "data/hmm_regimes"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy", "hashlib"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    },
    context="Complete Feature Engineering",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=10.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=5000, streaming_processing=True, memory_pool=True, cleanup_frequency=5,
)
@quality_gate(
    data_quality_threshold=0.9,
    feature_quality_threshold=0.8,
    model_quality_threshold=0.7,
    validation_checks=["data_integrity", "feature_quality", "feature_stability"],
)
@circuit_breaker_protection(
    max_execution_time=7200,  # 2 hours
    max_memory_usage_gb=16.0,
    max_cpu_usage_percent=90.0,
    error_threshold=3,
    recovery_timeout=600,
)
@debug_training_step(
    enable_debug_logging=True,
    save_intermediate_results=True,
    enable_profiling=True,
    debug_output_dir="debug_output/step6",
)
@monitor_feature_engineering(
    track_feature_importance=True,
    monitor_feature_correlations=True,
    track_feature_stability=True,
    save_feature_analysis=True,
)
@validate_step_output(
    output_validation_rules={
        "required_files": ["features_train.parquet", "features_val.parquet", "feature_metadata.json"],
        "required_columns": ["timestamp", "features"],
        "min_rows": 1000,
        "max_missing_ratio": 0.05,
    },
    validation_timeout=600,
)
@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step6_feature_engineering",
)
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Step 6: Complete Feature Engineering (Simple + Advanced).
    
    This step creates comprehensive features including both basic and advanced features,
    with regime-aware optimization after HMM regime discovery.
    
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
    logger = system_logger.getChild("Step6FeatureEngineering")
    
    logger.info("=" * 80)
    logger.info("🚀 STEP 6: Complete Feature Engineering (Simple + Advanced)")
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

        # 2) Validate input data
        logger.info("🔍 Validating input data...")
        if not _validate_input_data(unified_data):
            logger.error("❌ Input data validation failed")
            return False

        # 3) Load regime information from step3
        logger.info("📊 Loading regime information from step3...")
        regime_data = await _load_regime_data(symbol, exchange, timeframe)
        if regime_data is not None:
            logger.info(f"✅ Loaded regime data with {len(regime_data)} regimes")
        else:
            logger.warning("⚠️ No regime data found - proceeding without regime-aware features")

        # 4) Load labeled data from step5
        logger.info("📊 Loading labeled data from step5...")
        labeled_data = await _load_labeled_data(symbol, exchange, timeframe)
        if labeled_data is None or labeled_data.empty:
            logger.error("❌ Failed to load labeled data")
            return False

        # 5) Initialize vectorized feature engineering
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

        # 6) Create comprehensive features
        logger.info("🔧 Creating comprehensive features...")
        features_result = await _create_comprehensive_features(
            unified_data, labeled_data, regime_data, feature_engineer, symbol, exchange, timeframe
        )
        
        if not features_result:
            logger.error("❌ Failed to create comprehensive features")
            return False

        # 7) Monitor feature generation
        logger.info("📊 Monitoring feature generation...")
        feature_stats = _monitor_feature_generation(features_result["features_full"])
        logger.info(f"📈 Feature generation stats: {feature_stats}")

        # 8) Save feature artifacts
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


def _validate_input_data(data: pd.DataFrame) -> bool:
    """Validate input data before feature generation."""
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        system_logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check data quality
    if len(data) < 100:
        system_logger.error("Insufficient data for feature generation")
        return False
    
    # Check for negative prices
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        system_logger.error("Negative or zero prices detected")
        return False
    
    # Check for logical price relationships
    if not ((data["high"] >= data["low"]) & 
            (data["high"] >= data["open"]) & 
            (data["high"] >= data["close"]) &
            (data["low"] <= data["open"]) & 
            (data["low"] <= data["close"])).all():
        system_logger.error("Invalid price relationships detected")
        return False
    
    # Check for excessive missing values
    missing_ratio = data[required_columns].isnull().sum().sum() / (len(data) * len(required_columns))
    if missing_ratio > 0.1:  # More than 10% missing
        system_logger.error(f"Too many missing values: {missing_ratio:.2%}")
        return False
    
    system_logger.info("✅ Input data validation passed")
    return True


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
    """Load regime data from step3."""
    try:
        regime_file = Path(f"data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet")
        
        if regime_file.exists():
            regime_data = pd.read_parquet(regime_file)
            system_logger.info(f"Loaded regime data: {regime_data.shape}")
            return regime_data
        else:
            system_logger.warning(f"Regime file not found: {regime_file}")
            return None
            
    except Exception as e:
        system_logger.error(f"Failed to load regime data: {e}")
        return None


async def _load_labeled_data(symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
    """Load labeled data from step5."""
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
        # Merge data
        merged_data = unified_data.copy()
        
        if regime_data is not None:
            # Merge regime information
            merged_data = merged_data.merge(
                regime_data[["timestamp", "regime"]], 
                on="timestamp", 
                how="left"
            )
            system_logger.info("✅ Merged regime data")
        
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
        features_df = _create_basic_features(merged_data)
        
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
    
    system_logger.info(f"✅ Created {len(windows) * 7} rolling window features")
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
    
    # Z-score features
    features["returns_zscore"] = (features["returns"] - features["returns"].rolling(20).mean()) / features["returns"].rolling(20).std()
    features["volume_zscore"] = (features["volume"] - features["volume"].rolling(20).mean()) / features["volume"].rolling(20).std()
    
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
            "enable_basic_features": False,  # Already done in step6
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
        output_dir = Path("data/training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature files
        features_result["features_train"].to_parquet(
            output_dir / f"{exchange}_{symbol}_{timeframe}_features_train.parquet"
        )
        features_result["features_val"].to_parquet(
            output_dir / f"{exchange}_{symbol}_{timeframe}_features_val.parquet"
        )
        
        # Save metadata
        with open(output_dir / f"{exchange}_{symbol}_{timeframe}_feature_metadata.json", "w") as f:
            json.dump(features_result["metadata"], f, indent=2, default=str)
        
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