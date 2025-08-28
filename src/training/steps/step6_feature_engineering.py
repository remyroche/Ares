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

import numpy as np
import pandas as pd

# Import vectorized advanced feature engineering
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)

# Import optimized feature engineering for enhanced functionality
try:
    from src.utils.optimized_feature_engineering import OptimizedFeatureEngineering
    OPTIMIZED_FE_AVAILABLE = True
except ImportError:
    OPTIMIZED_FE_AVAILABLE = False

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

        # 2) Load regime information from step3
        logger.info("📊 Loading regime information from step3...")
        regime_data = await _load_regime_data(symbol, exchange, timeframe)
        if regime_data is not None:
            logger.info(f"✅ Loaded regime data with {len(regime_data)} regimes")
        else:
            logger.warning("⚠️ No regime data found - proceeding without regime-aware features")

        # 3) Load labeled data from step5
        logger.info("📊 Loading labeled data from step5...")
        labeled_data = await _load_labeled_data(symbol, exchange, timeframe)
        if labeled_data is None or labeled_data.empty:
            logger.error("❌ Failed to load labeled data")
            return False

        # 4) Initialize feature engineering (optimized if available)
        logger.info("🔧 Initializing feature engineering...")
        
        if OPTIMIZED_FE_AVAILABLE:
            logger.info("🚀 Using optimized feature engineering...")
            feature_engineer = OptimizedFeatureEngineering(
                config={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "enable_regime_features": regime_data is not None,
                    "enable_sr_features": True,
                    "enable_wavelet_features": True,
                    "enable_interaction_features": True,
                    "min_feature_quality": 0.3,
                    "max_correlation_threshold": 0.95,
                }
            )
        else:
            logger.info("🔧 Using vectorized advanced feature engineering...")
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

        # 6) Save feature artifacts
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
    feature_engineer: Any,  # Can be either OptimizedFeatureEngineering or VectorizedAdvancedFeatureEngineering
    symbol: str,
    exchange: str,
    timeframe: str
) -> Dict[str, Any]:
    """Create comprehensive features using optimized or vectorized feature engineering."""
    
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
        
        # Create features using the appropriate feature engineering method
        if hasattr(feature_engineer, 'generate_features') and callable(getattr(feature_engineer, 'generate_features')):
            # Use optimized feature engineering if available
            system_logger.info("🚀 Using optimized feature engineering...")
            features_df = await feature_engineer.generate_features(
                merged_data,
                include_sr_analysis=True,
                include_regime_analysis=regime_data is not None
            )
        else:
            # Fallback to basic feature creation
            system_logger.info("🔧 Using basic feature engineering...")
            features_df = _create_basic_features(merged_data)
            
            # Add advanced features if regime data is available
            if regime_data is not None:
                features_df = _add_regime_aware_features(features_df, merged_data)
            
            # Add technical indicators
            features_df = _add_technical_indicators(features_df)
            
            # Add statistical features
            features_df = _add_statistical_features(features_df)
            
            # Add HMM feature enhancement if regime data is available
            if regime_data is not None:
                features_df = _enhance_hmm_features(features_df, regime_data)
        
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
                "optimized_engineering": hasattr(feature_engineer, 'generate_features'),
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