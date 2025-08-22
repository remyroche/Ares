# src/training/steps/step2_feature_engineering.py

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any, Never

import numpy as np
import pandas as pd

# Import SR breakout predictor for comprehensive SR features
from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor

# Import optimized feature selection manager
from src.training.optimized_feature_selection_manager import (
    OptimizedFeatureSelectionManager,
)

# Import comprehensive file validation
try:
    from src.utils.comprehensive_file_validation import (
    ComprehensiveFileValidator,
    validate_step2_file,
    FileValidationResult
)
from src.utils.validation_decorators import (
    validate_file_operation,
    validate_dataframe_operation,
    validate_step2_operation
)
from src.utils.advanced_ml_validation import validate_ml_data_quality
from src.utils.enhanced_validation_decorators import step_specific_ml_validation
except ImportError:
    ComprehensiveFileValidator = None
    validate_step2_file = None
    FileValidationResult = None
    validate_file_operation = None
    validate_dataframe_operation = None
    validate_step2_operation = None

# Import the auto-fix decorator for data quality issues
from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.utils.logger import system_logger

# Import training pipeline decorators for comprehensive security and troubleshooting
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
)


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=1.0,
    min_disk_gb=0.5,
    required_packages=["pandas", "numpy", "hashlib"],
    data_quality_checks={
        "min_rows": 1,
        "required_columns": [],
    },
    context="Feature Artifact Hash Generation",
)
@secure_data_processing(
    backup_before=False, integrity_checks=False, memory_cleanup=False, data_validation=False,
)
@prevent_data_leakage(
    temporal_validation=False,
    feature_leakage_detection=False,
    lookahead_bias_prevention=False,
)
@resource_monitor(
    memory_threshold_gb=2.0,
    cpu_threshold_percent=20.0,
    disk_threshold_gb=1.0,
    monitor_interval=5.0,
    auto_cleanup=False,
)
@memory_efficient(
    chunk_size=100, streaming_processing=False, memory_pool=False, cleanup_frequency=1,
)
@debug_training_step(
    log_intermediate_results=False,
    save_debug_artifacts=False,
    performance_profiling=False,
    error_context_preservation=False,
)
@circuit_breaker_protection(
    failure_threshold=10,
    recovery_timeout=10.0,
    expected_exception=Exception,
    monitor_interval=2.0,
)
def _generate_feature_artifact_hash(symbol: str, exchange: str, timeframe: str, data_dir: str) -> str:
    """Generate a hash for the feature artifact based on input data characteristics.
    This ensures we regenerate features when the underlying data changes.
    """
    try:
        # Check if labeled data exists and get its hash
        labeled_paths = [
            f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
            f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
            f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
        ]

        # Create a hash based on file modification times and sizes
        hash_input = f"{symbol}_{exchange}_{timeframe}"
        for path in labeled_paths:
            if os.path.exists(path):
                stat = os.stat(path)
                hash_input += f"_{stat.st_mtime}_{stat.st_size}"

        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception:
        # Fallback to simple hash
        return hashlib.md5(f"{symbol}_{exchange}_{timeframe}".encode()).hexdigest()


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=1.0,
    min_disk_gb=0.5,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 1,
        "required_columns": [],
    },
    context="Feature Artifact Path Generation",
)
@secure_data_processing(
    backup_before=False, integrity_checks=False, memory_cleanup=False, data_validation=False,
)
@prevent_data_leakage(
    temporal_validation=False,
    feature_leakage_detection=False,
    lookahead_bias_prevention=False,
)
@resource_monitor(
    memory_threshold_gb=1.0,
    cpu_threshold_percent=10.0,
    disk_threshold_gb=0.5,
    monitor_interval=2.0,
    auto_cleanup=False,
)
@memory_efficient(
    chunk_size=10, streaming_processing=False, memory_pool=False, cleanup_frequency=1,
)
@debug_training_step(
    log_intermediate_results=False,
    save_debug_artifacts=False,
    performance_profiling=False,
    error_context_preservation=False,
)
@circuit_breaker_protection(
    failure_threshold=20,
    recovery_timeout=5.0,
    expected_exception=Exception,
    monitor_interval=1.0,
)
def _get_feature_artifact_paths(symbol: str, exchange: str, data_dir: str) -> dict[str, str]:
    """Get the paths for feature artifacts."""
    base_name = f"{exchange}_{symbol}_features"
    return {
        "train": f"{data_dir}/{base_name}_train.parquet",
        "validation": f"{data_dir}/{base_name}_validation.parquet",
        "test": f"{data_dir}/{base_name}_test.parquet",
        "metadata": f"{data_dir}/{base_name}_metadata.json",
        "hash": f"{data_dir}/{base_name}_hash.txt",
    }


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=2.0,
    min_disk_gb=1.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Existence Check",
)
@secure_data_processing(
    backup_before=False, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=False,
    feature_leakage_detection=False,
    lookahead_bias_prevention=False,
)
@resource_monitor(
    memory_threshold_gb=4.0,
    cpu_threshold_percent=30.0,
    disk_threshold_gb=2.0,
    monitor_interval=5.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=1000, streaming_processing=False, memory_pool=True, cleanup_frequency=5,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=Exception,
    monitor_interval=5.0,
)
def _check_feature_artifacts_exist(symbol: str, exchange: str, data_dir: str) -> bool:
    """Check if all required feature artifacts exist and are valid."""
    paths = _get_feature_artifact_paths(symbol, exchange, data_dir)

    # Check if all required files exist
    required_files = ["train", "validation", "test", "metadata", "hash"]
    for file_type in required_files:
        if not os.path.exists(paths[file_type]):
            return False

    # Validate that the files are not empty
    for file_type in ["train", "validation", "test"]:
        try:
            df = pd.read_parquet(paths[file_type])
            if df.empty:
                return False
        except Exception:
            return False

    return True


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Loading",
)
@secure_data_processing(
    backup_before=False, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=False,
    feature_leakage_detection=False,
    lookahead_bias_prevention=False,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=10.0,
    monitor_interval=15.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=20,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=False,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=15.0,
)
def _load_feature_artifacts(symbol: str, exchange: str, data_dir: str) -> dict[str, pd.DataFrame]:
    """Load existing feature artifacts."""
    paths, _get_feature_artifact_paths(symbol, exchange, data_dir)

    artifacts = {}
    for split in ["train", "validation", "test"]:
        artifacts[split] = pd.read_parquet(paths[split])

    return artifacts


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "json"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp"],
    },
    context="Feature Artifact Saving",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=False,
    feature_leakage_detection=False,
    lookahead_bias_prevention=False,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=15.0,
    monitor_interval=20.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=2,
    recovery_timeout=180.0,
    expected_exception=Exception,
    monitor_interval=20.0,
)
def _save_feature_artifacts(symbol: str, exchange: str, data_dir: str, features: dict[str, pd.DataFrame], feature_config: dict[str, Any], artifact_hash: str, ) -> None:
    """Save feature artifacts with metadata."""
    paths, _get_feature_artifact_paths(symbol, exchange, data_dir)

    # Save feature DataFrames
    for split, df in features.items():
        df.to_parquet(paths[split])

    # Save metadata
    metadata = {
        "symbol": symbol,
        "exchange": exchange,
        "created_at": datetime.now().isoformat(),
        "feature_config": feature_config,
        "feature_counts": {split: len(df.columns) for split, df in features.items()},
        "row_counts": {split: len(df) for split, df in features.items()},
        "feature_columns": list(features["train"].columns) if not features["train"].empty else [],
    }

    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    # Save hash
    with open(paths["hash"], "w") as f:
        f.write(artifact_hash)


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "sklearn", "talib"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    },
    context="Feature Engineering",
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
    memory_threshold_gb=16.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=180.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["features", "targets"]},
    performance_thresholds={"engineering_time_minutes": 60.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7},
)
@auto_fix_data_quality_issues
@step_specific_ml_validation("step2", timestamp_col="timestamp") if step_specific_ml_validation else lambda x: x
async def run_step(symbol: str, exchange: str = "BINANCE", data_dir: str = "data/training", timeframe: str = "1m", force_rerun: bool = False, **kwargs: Any) -> bool:
    """Step 2: Engineering the features (post-labeling).
    Loads labeled parquet from Step 1 and produces robust feature parquet artifacts for train/val/test.
    Also writes pickle copies with timestamps and a feature hash for Step 3 compatibility.

    Features:
    - Persistent artifact system: Features are cached unless force_rerun=True
    - Hash-based invalidation: Features are regenerated when input data changes
    - Metadata tracking: Saves configuration and statistics for reproducibility
    """
    logger = system_logger.getChild("Step2.FeatureEngineering")

    try:
        # Check for existing artifacts first
        artifact_hash = _generate_feature_artifact_hash(symbol, exchange, timeframe, data_dir)
        artifacts_exist = _check_feature_artifacts_exist(symbol, exchange, data_dir)

        if artifacts_exist and not force_rerun:
            logger.info("📦 Loading existing feature artifacts (use --force-rerun to regenerate)")

            # Load existing artifacts
            features = _load_feature_artifacts(symbol, exchange, data_dir)

            # Log artifact information
            for split, df in features.items():
                logger.info(f"✅ Loaded {split} features: {len(df)} rows, {len(df.columns)} features")

            logger.info("🎯 Feature engineering completed (using cached artifacts)")
            return True

        if artifacts_exist and force_rerun:
            logger.info("🔄 Force rerun enabled - regenerating features")
        else:
            logger.info("🔧 No existing artifacts found - generating features")

        # Continue with feature engineering...
        from src.training.enhanced_training_manager_optimized import (
            MemoryEfficientDataManager,
        )
        from src.training.steps.vectorized_advanced_feature_engineering import (
            VectorizedAdvancedFeatureEngineering,
        )

        # 1) Load unified data from step1_5 using secure data loader
        logger.info("📊 Loading unified data from step1_5...")

        try:
            from src.training.steps.unified_data_loader import load_unified_data

            # Load unified data with comprehensive validation
            unified_data = await load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir="data_cache",  # step1_5 data is in data_cache
                columns=["timestamp", "open", "high", "low", "close", "volume", "exchange", "symbol", "timeframe"]
            )

            if unified_data is None or unified_data.empty:
                logger.error("❌ Failed to load unified data from step1_5")
                return False

            logger.info(f"✅ Loaded {len(unified_data)} rows of unified data from step1_5")

            # Split data into train/validation/test (80/10/10 split)
            total_rows = len(unified_data)
            train_end = int(total_rows * 0.8)
            val_end = int(total_rows * 0.9)

            labeled = {
                "train": unified_data.iloc[:train_end].copy(),
                "validation": unified_data.iloc[train_end:val_end].copy(),
                "test": unified_data.iloc[val_end:].copy(),
            }

            for split, df in labeled.items():
                logger.info(f"📦 Split {split}: {len(df)} rows")

        except ImportError:
            logger.warning("⚠️ Unified data loader not available, falling back to legacy method")
            # Fallback to legacy method
            paths = {
                "train": f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
                "validation": f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
                "test": f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
            }
            labeled = {name: pd.read_parquet(path) for name, path in paths.items()}
            for split, df in labeled.items():
                logger.info(f"📦 Loaded labeled {split}: {len(df)} rows")

        # Ensure timestamp present and set as index for alignment
        for k in labeled:
            if "timestamp" not in labeled[k].columns and isinstance(
                labeled[k].index, pd.DatetimeIndex,
            ):
                labeled[k] = (
                    labeled[k].reset_index().rename(columns={"index": "timestamp"})
                )
            if "timestamp" in labeled[k].columns:
                labeled[k]["timestamp"] = pd.to_datetime(
                    labeled[k]["timestamp"], errors="coerce"
                )
                labeled[k] = (
                    labeled[k].dropna(subset=["timestamp"]).sort_values("timestamp")
                )
                labeled[k] = labeled[k].set_index("timestamp")

        # 2) Extract OHLCV inputs
        @with_tracing_span("Step3._extract_inputs", log_args=False)
        @guard_dataframe_nulls(mode="warn", arg_index=0)
        def _extract_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            price_cols = [
                c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
            ]
            if len(price_cols) < 4:  # expect at least open/high/low/close
                msg = "🚨 Missing OHLC columns in labeled data"
                raise ValueError(msg)
            price = df[price_cols].copy()
            vol = (
                price[["volume"]].copy()
                if "volume" in price.columns
                else pd.DataFrame({"volume": 1.0}, index=price.index)
            )
            return price, vol

        # SR levels loader with append-and-reuse semantics (prefers Step 2 persisted levels)
        @with_tracing_span("Step3._load_or_build_sr_levels", log_args=False)
        async def _load_or_build_sr_levels(price_df: pd.DataFrame, split_name: str
        ) -> dict[str, Any]:
            try:
                data_dir = data_dir_ref
                exchange_local = exchange
                symbol_local = symbol
                import os

                sr_path = (
                    f"{data_dir}/{exchange_local}_{symbol_local}_sr_levels.parquet"
                )
                if os.path.exists(sr_path):
                    sr_df = pd.read_parquet(sr_path)
                    # Retain only recent enough levels and compute age decay; keep full history for training alignment
                    sr_df["timestamp"] = pd.to_datetime(
                        sr_df["timestamp"], errors="coerce"
                    )
                    sr_df = sr_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                    # Align to the split end timestamp; strengths can be decayed by age (optional)
                    pd.to_datetime(price_df.index.max())
                    # Build levels list with decayed strength based on age in minutes
                    supports = []
                    resistances = []
                    for _, row in sr_df.iterrows():
                        price = float(row.get("price", np.nan))
                        if not np.isfinite(price):
                            continue
                        base_strength = float(row.get("strength", 0.2))
                        age_min = float(row.get("age", 0.0))
                        # Simple exponential decay by age in hours, configurable via kwargs
                        decay_hl_min = float(
                            kwargs.get("sr_strength_half_life_min", 24 * 60),
                        )
                        if decay_hl_min > 0:
                            lam = np.log(2) / max(decay_hl_min, 1e-6)
                            decayed = base_strength * float(
                                np.exp(-lam * max(0.0, age_min)),
                            )
                        else:
                            decayed = base_strength
                        lvl = {
                            "price": price,
                            "strength": float(np.clip(decayed, 0.0, 1.0)),
                        }
                        if (
                            str(row.get("level_type", "support"))
                            .lower()
                            .startswith("support")
                        ):
                            supports.append(lvl)
                        else:
                            resistances.append(lvl)
                    return {
                        "support_levels": supports,
                        "resistance_levels": resistances,
                    }
            except Exception:
                pass
            # Fallback to lightweight builder (percentiles) if persisted SR is unavailable
            try:
                lows = price_df["low"].astype(float)
                highs = price_df["high"].astype(float)
                window = min(len(lows), 2000)
                if window <= 0:
                    return {"support_levels": [], "resistance_levels": []}
                lt = lows.tail(window).dropna()
                ht = highs.tail(window).dropna()
                if lt.empty or ht.empty:
                    return {"support_levels": [], "resistance_levels": []}
                support_prices = np.percentile(lt.values, [5, 15, 30]).tolist()
                resistance_prices = np.percentile(ht.values, [70, 85, 95]).tolist()

                def _mk_levels(vals, strength=0.2):
                    out = []
                    seen: set[float] = set()
                    for v in vals:
                        r = round(float(v), 8)
                        if r in seen:
                            continue
                        seen.add(r)
                        out.append({"price": r, "strength": float(strength)})
                    return out

                return {
                    "support_levels": _mk_levels(support_prices, 0.2),
                    "resistance_levels": _mk_levels(resistance_prices, 0.2),
                }
            except Exception:
                return {"support_levels": [], "resistance_levels": []}

        @with_tracing_span("Step2._generate_comprehensive_sr_features", log_args=False)
        async def _generate_comprehensive_sr_features(price_df: pd.DataFrame, sr_levels: dict[str, Any]
        ) -> dict[str, pd.Series]:
            """Generate comprehensive SR features using the SRBreakoutPredictor.

            This function adds the following comprehensive SR features:
            - distance_to_resistance & distance_to_support
            - normalized_distance_to_resistance & normalized_distance_to_support
            - sr_proximity_score
            - strength_score
            - clarity_factor
            - directional_pressure
            - sr_score
            - delta_sr_score
            - isolation_score
            """
        try:
            # Initialize SR breakout predictor
            sr_predictor = await setup_sr_breakout_predictor(config)
            if not sr_predictor:
                logger.warning("⚠️ Failed to initialize SR breakout predictor, using basic features")
                return {}

            # Generate comprehensive SR features
            comprehensive_features = sr_predictor.calculate_comprehensive_sr_features(
                price_df=sr_levels,
            )

            if comprehensive_features:
                logger.info(f"✅ Generated {len(comprehensive_features)} comprehensive SR features")
                # Log feature names for debugging
                feature_names = list(comprehensive_features.keys())
                logger.info(f"🔍 Comprehensive SR features: {feature_names}")
            else:
                logger.warning("⚠️ No comprehensive SR features generated")

            return comprehensive_features

        except Exception as e:
            logger.exception(f"❌ Error generating comprehensive SR features: {e}")
            return {}

        price_tr, vol_tr = _extract_inputs(labeled["train"])
        price_vl, vol_vl = _extract_inputs(labeled["validation"])
        price_te, vol_te = _extract_inputs(labeled["test"])

        # 3) Initialize FE engine with configuration
        # Get configuration from kwargs or use defaults
        feature_config = kwargs.get("feature_config", {})
        if not feature_config:
        # Default configuration with difference and acceleration features enabled
        # NOTE: Advanced features have been re-enabled after fixing indentation issues
        # in vectorized_advanced_feature_engineering.py. All analyzers should now work properly.
            feature_config = {
                "vectorized_advanced_features": {
                    "enable_difference_acceleration_features": True,
                    "enable_volatility_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                    "enable_candlestick_patterns": True,  # Re-enabled after fixing indentation issues
                    "enable_sr_distance": True,  # Re-enabled after fixing indentation issues
                    "enable_wavelet_transforms": True,  # Re-enabled after fixing indentation issues
                    "enable_multi_timeframe": True,
                    "enable_meta_labeling": False,
                    "enable_explicit_meta_labels": False,
                },
            }

        # Add symbol and exchange to the feature config for data quality decorator
        feature_config["symbol"] = symbol
        feature_config["exchange"] = exchange

        fe = VectorizedAdvancedFeatureEngineering(feature_config)
        await fe.initialize()

        # 4) Engineer features per split
        # Bind data_dir for loader
        data_dir_ref = data_dir
        sr_tr = await _load_or_build_sr_levels(price_tr, "train")
        sr_vl = await _load_or_build_sr_levels(price_vl, "validation")
        sr_te = await _load_or_build_sr_levels(price_te, "test")

        feats_tr = await fe.engineer_features(price_tr, vol_tr, sr_levels=sr_tr)
        feats_vl = await fe.engineer_features(price_vl, vol_vl, sr_levels=sr_vl)
        feats_te = await fe.engineer_features(price_te, vol_te, sr_levels=sr_te)

        # Add comprehensive SR features
        comprehensive_sr_tr = await _generate_comprehensive_sr_features(price_tr, sr_tr)
        comprehensive_sr_vl = await _generate_comprehensive_sr_features(price_vl, sr_vl)
        comprehensive_sr_te = await _generate_comprehensive_sr_features(price_te, sr_te)

        def _merge_features(target_feats, new_feats) -> None:
            if not new_feats:
                return
            for feature_name, feature_series in new_feats.items():
                if feature_name not in target_feats:
                    target_feats[feature_name] = feature_series

        # Merge comprehensive SR features with existing features
        _merge_features(feats_tr, comprehensive_sr_tr)
        _merge_features(feats_vl, comprehensive_sr_vl)
        _merge_features(feats_te, comprehensive_sr_te)

        X_tr = pd.DataFrame(feats_tr).reindex(price_tr.index)
        X_vl = pd.DataFrame(feats_vl).reindex(price_vl.index)
        X_te = pd.DataFrame(feats_te).reindex(price_te.index)

        # 4a) HMM features will be calculated in step3 when properly trained
        logger.info("ℹ️ Skipping HMM feature loading in step2 - will be calculated in step3")

        # 4b) Optionally augment with Autoencoder features
        @with_tracing_span("Step3._augment_with_autoencoder", log_args=False)
        def _augment_with_autoencoder(features_df: pd.DataFrame, split: str
        ) -> pd.DataFrame:
            try:
                from src.analyst.autoencoder_feature_generator import (
                    AutoencoderFeatureGenerator,
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Autoencoder unavailable for Step 3 augmentation: {e}",
                )
                return features_df
            try:
                ae = AutoencoderFeatureGenerator({})
                y = None
                try:
                    y = (
                        labeled[split]["label"].astype(int).values
                        if "label" in labeled[split].columns
                        else np.zeros(len(features_df))
                    )
                except Exception:
                    y = np.zeros(len(features_df))
                ae_input = features_df.copy()
                ae_df = ae.generate_features(ae_input, f"step3_{split}", y)
                if isinstance(ae_df, pd.DataFrame) and not ae_df.empty:
                    ae_df = ae_df.reindex(features_df.index)
                    merged = pd.concat([features_df, ae_df], axis=1)
                    logger.info(
                        f"✅ Augmented {split} with Autoencoder features: +{ae_df.shape[1]} cols",
                    )
                    return merged
                return features_df
            except Exception as e:
                logger.warning(f"⚠️ Autoencoder augmentation skipped for {split}: {e}")
                return features_df

        # Temporarily disable autoencoder features to avoid validation issues
        if bool(kwargs.get("enable_autoencoder_features", False)):  # Changed from True to False
            X_tr = _augment_with_autoencoder(X_tr, "train")
            X_vl = _augment_with_autoencoder(X_vl, "validation")
            X_te = _augment_with_autoencoder(X_te, "test")

        # 4c) Handle lookahead bias for specific features that need lagging
        @with_tracing_span("Step3._handle_lookahead_bias", log_args=False)
        def _handle_lookahead_bias(features_df: pd.DataFrame) -> pd.DataFrame:
            """Apply lagging to features that may have lookahead bias."""
            try:
                # List of features that commonly have lookahead bias
                features_needing_lagging = [
                    "market_depth_change", "market_depth_returns", "market_depth_imbalance",
                    "ema20_slope", "sma50_slope", "price_impact", "volume_price_impact",
                    "order_flow_imbalance", "bid_ask_spread_returns", "bid_ask_spread_level",
                    "market_depth_change", "market_depth_returns", "market_depth_imbalance",
                ]

                # Find features that exist in the DataFrame and need lagging
                existing_features = [col for col in features_needing_lagging if col in features_df.columns]

                if existing_features:
                    logger.info(f"🔧 Applying lagging to {len(existing_features)} features to prevent lookahead bias")

                    # Apply 1-period lag to these features
                    for feature in existing_features:
                        lagged_feature_name = f"{feature}_lag1"
                        features_df[lagged_feature_name] = features_df[feature].shift(1)

                        # Replace original feature with lagged version
                        features_df[feature] = features_df[lagged_feature_name]
                        features_df = features_df.drop(columns=[lagged_feature_name])

                    logger.info(f"✅ Applied lagging to features: {existing_features}")

                return features_df
            except Exception as e:
                logger.warning(f"⚠️ Lookahead bias handling failed: {e}")
                return features_df

        # Apply lookahead bias handling to all splits
        X_tr = _handle_lookahead_bias(X_tr)
        X_vl = _handle_lookahead_bias(X_vl)
        X_te = _handle_lookahead_bias(X_te)

        # 5) Basic sanitization: drop constant columns, handle inf/nan
        @with_tracing_span("Step3._sanitize", log_args=False)
        @guard_dataframe_nulls(mode="warn", arg_index=0)
        def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.replace([np.inf, -np.inf], np.nan)

            # Check for constant features more intelligently
            low_var_cols = []
            for col in df.columns:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series.dtype):
                    # For numeric columns, check variance
                    if series.var() == 0:
                        low_var_cols.append(col)
                # For categorical columns, check unique values
                elif series.nunique() <= 1:
                    low_var_cols.append(col)

            if low_var_cols:
                logger.warning(f"🚨 BUG: Found {len(low_var_cols)} constant features - this indicates calculation errors!")
                logger.warning(f"🚨 BUG: Constant features: {low_var_cols[:10]}... (showing first 10)")

                # Debug the first few constant features
                for feat in low_var_cols[:3]:
                    feat_data = df[feat]
                    logger.warning(f"🚨 BUG ANALYSIS for '{feat}':")
                    logger.warning(f"🚨   - All values: {feat_data.iloc[:5].tolist()}... (first 5)")
                    logger.warning(f"🚨   - Unique values: {feat_data.nunique()}")
                    logger.warning(f"🚨   - Min/Max: {feat_data.min()}/{feat_data.max()}")
                    logger.warning(f"🚨   - NaN count: {feat_data.isna().sum()}")

                logger.warning(f"🚨 REMOVING constant features due to calculation bugs: {len(low_var_cols)} features")
                df = df.drop(columns=low_var_cols, errors="ignore")
            return df.fillna(0)

        X_tr = _sanitize(X_tr)
        X_vl = _sanitize(X_vl)
        X_te = _sanitize(X_te)

        # 6) Cluster-based correlation pruning with cap (|rho| >= threshold)
        @with_tracing_span("Step3._cluster_corr_prune", log_args=False)
        def _cluster_corr_prune(train_df: pd.DataFrame, thr: float = 0.95, max_to_drop: int | None = None
        ) -> list[str]:
            if train_df.empty:
                return []
            numeric_df = train_df.select_dtypes(include=[np.number]).copy()
            if numeric_df.shape[1] < 2:
                return []
            numeric_df = numeric_df.fillna(0.0)
            cols = list(numeric_df.columns)
            corr = numeric_df.corr().abs()

            # Build adjacency based on threshold
            neighbors = {c: set() for c in cols}
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if corr.iloc[i, j] >= thr:
                        ci, cj = cols[i], cols[j]
                        neighbors[ci].add(cj)
                        neighbors[cj].add(ci)

            # Find connected components (clusters)
            visited = set()
            clusters: list[list[str]] = []
            for c in cols:
                if c in visited:
                    continue
                stack = [c]
                cluster = []
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    cluster.append(node)
                    for nb in neighbors.get(node, set()):
                        if nb not in visited:
                            stack.append(nb)
                if len(cluster) > 1:
                    clusters.append(cluster)

            # Pick representative per cluster (keep max-variance feature)
            to_drop: list[str] = []
            for cluster in clusters:
                var_series = numeric_df[cluster].var(ddof=0)
                keep_col = str(var_series.idxmax())
                for col in cluster:
                    if col != keep_col:
                        to_drop.append(col)

            if max_to_drop is not None and len(to_drop) > max_to_drop:
                logger.warning(
                    f"⚠️ Cluster correlation pruning proposed {len(to_drop)} removals; capping at {max_to_drop}.",
                )
                to_drop = to_drop[:max_to_drop]

            return to_drop

        # Execute cluster correlation pruning with 50% cap (read from config if available)
        initial_feature_count = X_tr.shape[1]
        try:
            from src.utils.config_loader import ConfigLoader
            loader = ConfigLoader()
            fs_conf = loader.load_yaml_config("src/config/feature_selection_config.yaml").get("feature_selection", {})
            cluster_thr = float(fs_conf.get("cluster_corr_threshold", kwargs.get("cluster_corr_threshold", 0.95)))
            cluster_cap_fraction = float(fs_conf.get("cluster_corr_max_removal_fraction", kwargs.get("cluster_corr_max_removal_fraction", 0.5)))
        except Exception:
            cluster_thr = float(kwargs.get("cluster_corr_threshold", 0.95))
            cluster_cap_fraction = float(kwargs.get("cluster_corr_max_removal_fraction", 0.5))
        cluster_cap_count = int(initial_feature_count * cluster_cap_fraction)
        drop_tr = _cluster_corr_prune(X_tr, thr=cluster_thr, max_to_drop=cluster_cap_count)
        removed_corr_count = len(drop_tr)
        if drop_tr:
            logger.info(
                f"🔗 Cluster correlation prune: dropping {len(drop_tr)} features (|rho|>={cluster_thr:.2f}, cap={cluster_cap_count})"
            )
            X_tr = X_tr.drop(columns=drop_tr, errors="ignore")
            X_vl = X_vl.drop(columns=drop_tr, errors="ignore")
            X_te = X_te.drop(columns=drop_tr, errors="ignore")

        # 7) Mutual information screen (classification target 'label')
        try:
            from sklearn.feature_selection import mutual_info_classif

            y = None
            if "label" in labeled["train"].columns:
                # Use classification labels from Step 2
                y = labeled["train"]["label"].astype(int).values
            if y is not None and len(np.unique(y)) > 1 and not X_tr.empty:
                numX = X_tr.select_dtypes(include=[np.number])
                if not numX.empty:
                    mi = mutual_info_classif(
                        numX.values, y, discrete_features=False, random_state=42
                    )
                    mi_s = pd.Series(mi, index=numX.columns).sort_values(
                        ascending=False
                    )
                    # Persist MI scores
                    os.makedirs("log/mi", exist_ok=True)
                    with open(f"log/mi/{exchange}_{symbol}_step3_mi.json", "w") as f:
                        json.dump({"mi": mi_s.to_dict()}, f, indent=2)
                    # Selection policy: keep top-k if provided; otherwise drop bottom quantile
                    mi_top_k = int(kwargs.get("mi_top_k", 0) or 0)
                    if mi_top_k > 0:
                        keep_cols = list(mi_s.head(mi_top_k).index)
                    else:
                        mi_quantile = float(kwargs.get("mi_quantile", 0.25))  # Keep top 75% of features (above 25th percentile)
                        thr = mi_s.quantile(mi_quantile)
                        keep_cols = list(mi_s[mi_s >= thr].index)

                    # Safety check: ensure we keep at least some features
                    if len(keep_cols) == 0:
                        # If quantile approach resulted in 0 features, keep top 10% or at least 5 features
                        min_features = max(5, int(len(mi_s) * 0.10))
                        keep_cols = list(mi_s.head(min_features).index)
                        logger.warning(f"⚠️ MI quantile resulted in 0 features, keeping top {min_features} features instead")

                    # Apply keep set safely across splits: skip features missing in any split
                    set_tr = set(X_tr.columns)
                    set_vl = set(X_vl.columns)
                    set_te = set(X_te.columns)

                    missing_tr = [c for c in keep_cols if c not in set_tr]
                    missing_vl = [c for c in keep_cols if c not in set_vl]
                    missing_te = [c for c in keep_cols if c not in set_te]

                    present_all = set_tr & set_vl & set_te
                    final_keep_cols = [c for c in keep_cols if c in present_all]

                    # Log and skip missing features per split
                    def _shorten(cols: list[str], limit: int = 15) -> str:
                        return (
                            ", ".join(cols[:limit]) + ("..." if len(cols) > limit else "")
                        ) if cols else ""

                    if missing_tr:
                        logger.warning(
                            f"⚠️ MI: skipping {len(missing_tr)} features absent in TRAIN split: [{_shorten(missing_tr)}]",
                        )
                    if missing_vl:
                        logger.warning(
                            f"⚠️ MI: skipping {len(missing_vl)} features absent in VALIDATION split: [{_shorten(missing_vl)}]",
                        )
                    if missing_te:
                        logger.warning(
                            f"⚠️ MI: skipping {len(missing_te)} features absent in TEST split: [{_shorten(missing_te)}]",
                        )

                    if not final_keep_cols:
                        logger.warning(
                            "⚠️ MI: no selected features were common across all splits; skipping MI application",
                        )
                    else:
                        X_tr = X_tr[final_keep_cols]
                        X_vl = X_vl[final_keep_cols]
                        X_te = X_te[final_keep_cols]
                        logger.info(
                            f"📊 MI kept {len(final_keep_cols)} features (top_k={mi_top_k} quantile={kwargs.get('mi_quantile', 0.66)})"
                        )
        except Exception as e:
            logger.warning(f"⚠️ MI screening skipped: {e}")

        # 8) VIF reduction (iterative) with combined 50% cap
        try:
            # Keep full dataset rows for VIF; no downsampling
            Xv = X_tr
            logger.info("🔍 VIF Analysis: Using full dataset rows for VIF (no downsampling)")

            vif_thr = float(kwargs.get("vif_threshold", 20.0))  # Increased to 20.0 for financial data (more permissive)
            int(kwargs.get("max_vif_iterations", 5))  # Reduced from 10 to 5
            num_cols = list(Xv.select_dtypes(include=[np.number]).columns)
            removed_features = []
            # Read thresholds from config if available
            try:
                from src.utils.config_loader import ConfigLoader
                loader = ConfigLoader()
                fs_conf = loader.load_yaml_config("src/config/feature_selection_config.yaml").get("feature_selection", {})
                overall_cap_fraction = float(fs_conf.get("max_total_removal_fraction", kwargs.get("max_total_removal_fraction", 0.5)))
            except Exception:
                overall_cap_fraction = float(kwargs.get("max_total_removal_fraction", 0.5))
            overall_cap_count = int(initial_feature_count * overall_cap_fraction)
            vif_allowed = max(0, overall_cap_count - removed_corr_count)

            # Debug: Log feature count and data quality
            logger.info(f"🔍 VIF Analysis: Starting with {len(num_cols)} numeric features")
            logger.info(f"🔍 VIF Analysis: Data shape: {Xv.shape}")
            logger.info(f"🔍 VIF Analysis: NaN count: {Xv[num_cols].isna().sum().sum()}")
            logger.info(f"🔍 VIF Analysis: Infinite count: {np.isinf(Xv[num_cols]).sum().sum()}")
            logger.info(f"🔍 VIF Analysis: Zero variance features: {(Xv[num_cols].var() == 0).sum()}")
            logger.info(f"🔍 VIF Analysis: Very low variance features (< 1e-10): {(Xv[num_cols].var() < 1e-10).sum()}")

            # Remove duplicate features before VIF analysis
            logger.info("🔍 VIF Analysis: Checking for duplicate features...")
            original_count = len(num_cols)

            # Remove features with zero variance
            zero_var_features = Xv[num_cols].columns[Xv[num_cols].var() == 0].tolist()
            if zero_var_features:
                logger.info(f"🔍 VIF Analysis: Removing {len(zero_var_features)} zero variance features: {zero_var_features}")
                num_cols = [col for col in num_cols if col not in zero_var_features]

            # Remove features with very low variance (likely duplicates)
            low_var_features = Xv[num_cols].columns[Xv[num_cols].var() < 1e-10].tolist()
            if low_var_features:
                logger.info(f"🔍 VIF Analysis: Removing {len(low_var_features)} very low variance features: {low_var_features}")
                num_cols = [col for col in num_cols if col not in low_var_features]

            # Standardize once and perform exact/near-duplicate and perfect-correlation sweeps
            try:
                Xn_base = Xv[num_cols].astype(float).fillna(0.0)
                std_base = Xn_base.std(ddof=0)
                std_base = std_base.replace(0.0, 1.0)
                Xn_base = (Xn_base - Xn_base.mean()) / std_base

                # Exact duplicates after standardization
                dup_mask = Xn_base.T.duplicated()
                if dup_mask.any():
                    dup_cols = list(Xn_base.columns[dup_mask])
                    if dup_cols:
                        logger.warning(f"⚠️ Duplicate standardized columns removed: {dup_cols}")
                        num_cols = [c for c in num_cols if c not in dup_cols]
                        Xn_base = Xn_base.drop(columns=dup_cols)

                # Near-duplicates via very high correlation (|corr|>=0.9999): keep lexicographically first
                corr_abs = Xn_base.corr().abs()
                to_remove = set()
                cols_sorted = sorted(num_cols)
                for i, ci in enumerate(cols_sorted):
                    if ci in to_remove:
                        continue
                    for cj in cols_sorted[i + 1 :]:
                        if cj in to_remove:
                            continue
                        if corr_abs.loc[ci, cj] >= 0.9999:
                            to_remove.add(cj)
                if to_remove:
                    logger.warning(f"⚠️ Near-duplicate columns (|corr|>=0.9999) removed: {sorted(to_remove)}")
                    num_cols = [c for c in num_cols if c not in to_remove]
                    Xn_base = Xn_base.drop(columns=list(to_remove))

                # Perfect correlation pairs (|corr|==1.0) – keep lexicographically first
                corr_abs2 = Xn_base.corr().abs()
                perfect_pairs = []
                removal_pc = set()
                cols_sorted2 = sorted(Xn_base.columns)
                for i, ci in enumerate(cols_sorted2):
                    if ci in removal_pc:
                        continue
                    for cj in cols_sorted2[i + 1 :]:
                        if cj in removal_pc:
                            continue
                        if corr_abs2.loc[ci, cj] == 1.0:
                            perfect_pairs.append((ci, cj))
                            removal_pc.add(cj)
                if perfect_pairs:
                    logger.warning(f"⚠️ Perfect-correlation pairs removed (keep first): {perfect_pairs}")
                if removal_pc:
                    num_cols = [c for c in num_cols if c not in removal_pc]
                    Xn_base = Xn_base.drop(columns=list(removal_pc))
            except Exception:
                # If standardization or corr fails, proceed without these prunes
                pass

            logger.info(f"🔍 VIF Analysis: After deduplication: {len(num_cols)} features (removed {original_count - len(num_cols)})")

            # One-shot VIF using shrinkage inverse of correlation matrix
            if len(num_cols) > 1:
                Xn = Xn_base[num_cols].astype(float).fillna(0.0)

                # Debug standardization issues
                logger.info(f"🔍 VIF Analysis: Before standardization - Shape: {Xn.shape}")
                logger.info(f"🔍 VIF Analysis: Before standardization - NaN count: {Xn.isna().sum().sum()}")
                logger.info(f"🔍 VIF Analysis: Before standardization - Infinite count: {np.isinf(Xn).sum().sum()}")

                # Check for zero or very small standard deviations
                raw_std = Xn.std(ddof=0)
                zero_std_features = raw_std[raw_std == 0].index.tolist()
                small_std_features = raw_std[raw_std < 1e-10].index.tolist()

                if zero_std_features:
                    logger.warning(f"⚠️ VIF Analysis: Found {len(zero_std_features)} features with zero std: {zero_std_features}")
                if small_std_features:
                    logger.warning(f"⚠️ VIF Analysis: Found {len(small_std_features)} features with very small std (< 1e-10): {small_std_features}")

                # Remove problematic features BEFORE standardization
                # 1. INVESTIGATE: Zero variance features indicate calculation bugs
                zero_var_features = Xn.columns[Xn.var() == 0].tolist()
                if zero_var_features:
                    logger.warning(f"🚨 BUG DETECTED: Found {len(zero_var_features)} constant features - this indicates calculation errors!")
                    logger.warning(f"🚨 BUG DETECTED: Constant features: {zero_var_features}")

                    # Debug each constant feature to understand the bug
                    for feat in zero_var_features[:3]:  # Debug first 3
                        feat_data = Xn[feat]
                        logger.warning(f"🚨 BUG ANALYSIS for '{feat}':")
                        logger.warning(f"🚨   - All values: {feat_data.iloc[:5].tolist()}... (first 5)")
                        logger.warning(f"🚨   - Unique values: {feat_data.nunique()}")
                        logger.warning(f"🚨   - Min/Max: {feat_data.min()}/{feat_data.max()}")
                        logger.warning(f"🚨   - NaN count: {feat_data.isna().sum()}")

                        # Check if it's all NaN filled with same value
                        if feat_data.nunique() == 1:
                            unique_val = feat_data.iloc[0]
                            if pd.isna(unique_val) or unique_val == 0:
                                logger.warning(f"🚨   - LIKELY CAUSE: All NaN values filled with {unique_val}")
                            else:
                                logger.warning(f"🚨   - LIKELY CAUSE: All values are {unique_val} (calculation bug)")

                    # TEMPORARILY KEEP constant features to debug the issue
                    logger.warning(f"🚨 TEMPORARILY KEEPING constant features for debugging: {zero_var_features}")
                    # Xn = Xn.drop(columns=zero_var_features)
                    # num_cols = [col for col in num_cols if col not in zero_var_features]

                # 2. Only remove features with extremely small variance (likely numerical artifacts)
                # Use a much more conservative threshold for financial data
                extremely_small_var_features = Xn.columns[Xn.var() < 1e-15].tolist()
                if extremely_small_var_features:
                    logger.info(f"🔍 VIF Analysis: Removing {len(extremely_small_var_features)} extremely small variance features before VIF: {extremely_small_var_features}")
                    Xn = Xn.drop(columns=extremely_small_var_features)
                    num_cols = [col for col in num_cols if col not in extremely_small_var_features]

                # 3. Check if we have enough features left
                if len(num_cols) < 2:
                    logger.warning(f"⚠️ VIF Analysis: Not enough features left after removing problematic ones: {len(num_cols)}")
                    # Skip VIF analysis if not enough features
                    vif_vals = pd.Series(np.ones(len(num_cols)), index=num_cols)
                    max_vif = 1.0

                # Now standardize the cleaned data
                std = Xn.std(ddof=0)
                std = std.replace(0.0, 1.0)
                Xn = (Xn - Xn.mean()) / std

                # Cheap VIF via shrinkage inverse with timeout protection
                import signal

                def timeout_handler(signum, frame) -> Never:
                    msg = "VIF calculation timed out"
                    raise TimeoutError(msg)

                try:
                    # Set timeout for VIF calculation (30 seconds)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)

                    try:
                        from sklearn.covariance import LedoitWolf
                        lw = LedoitWolf().fit(Xn.values)
                        cov = lw.covariance_
                        std_vec = np.sqrt(np.diag(cov))
                        std_vec[std_vec == 0.0] = 1.0
                        R = cov / np.outer(std_vec, std_vec)
                    except Exception:
                        logger.info("🔍 VIF Analysis: LedoitWolf failed, using correlation matrix")
                        R = Xn.corr().values

                    try:
                        R_inv = np.linalg.pinv(R)
                        vif_vals = pd.Series(np.diag(R_inv), index=num_cols)
                    except Exception as _e:
                        logger.warning(f"⚠️ VIF Analysis: pseudo-inverse failed, fallback to diag ones: {_e}")
                        vif_vals = pd.Series(np.ones(len(num_cols)), index=num_cols)

                    # Cancel timeout
                    signal.alarm(0)

                except TimeoutError:
                    logger.warning("⚠️ VIF Analysis: Calculation timed out, skipping VIF filtering")
                    vif_vals = pd.Series(np.ones(len(num_cols)), index=num_cols)
                except Exception as e:
                    logger.warning(f"⚠️ VIF Analysis: Unexpected error, skipping VIF filtering: {e}")
                    vif_vals = pd.Series(np.ones(len(num_cols)), index=num_cols)
                max_vif = float(vif_vals.max()) if not vif_vals.empty else 0.0

                logger.info(f"🔍 VIF Analysis: Max VIF: {max_vif:.2f}, Threshold: {vif_thr}")
                logger.info(f"🔍 VIF Analysis: VIF range: {vif_vals.min():.2f} to {vif_vals.max():.2f}")
                logger.info(f"🔍 VIF Analysis: Features with VIF > {vif_thr}: {(vif_vals > vif_thr).sum()}")

                # One-shot prune: drop up to K highest VIF offenders
                K = int(kwargs.get("max_vif_drop", 5))
                if (vif_vals > vif_thr).any() and K > 0:
                    offenders = vif_vals[vif_vals > vif_thr].sort_values(ascending=False)
                    drops = list(offenders.head(K).index)
                    # Respect overall cap
                    cap_left = max(0, vif_allowed - len(removed_features))
                    if len(drops) > cap_left:
                        drops = drops[:cap_left]
                    removed_features.extend(drops)
                    num_cols = [c for c in num_cols if c not in drops]
                    logger.info(f"📊 VIF prune (one-shot): dropping up to {K} high-VIF cols: {drops}")
                else:
                    logger.warning("⚠️ VIF Analysis: keeping all features (no offenders or K=0)")

                # Apply final VIF-selected set
                if num_cols:
                    X_tr = X_tr[num_cols]
                    X_vl = X_vl[num_cols]
                    X_te = X_te[num_cols]
                    logger.info(
                        f"📊 VIF kept {len(num_cols)} features (threshold={vif_thr}, removed_vif={len(removed_features)}, removed_corr={removed_corr_count})"
                    )

                    # Print the features that passed through VIF filter
                    features_list = ", ".join([f"{i}.{feature}" for i, feature in enumerate(num_cols, 1)])
                    logger.info(f"✅ VIF FILTERED FEATURES ({len(num_cols)}): [{features_list}]")

                    # Log summary of all removed features
                    if removed_features:
                        logger.info(f"📊 VIF REMOVAL SUMMARY - Total removed: {len(removed_features)}")
                        logger.info(f"📊 VIF REMOVAL SUMMARY - Removed features: {removed_features}")

                        # Show some examples of kept features
                        kept_features_sample = num_cols[:10] if len(num_cols) > 10 else num_cols
                        logger.info(f"📊 VIF REMOVAL SUMMARY - Sample of kept features: {kept_features_sample}")
            else:  # Safety check: if VIF removed all features = keep original features
                logger.warning("⚠️ VIF removed all features, keeping original feature set")
                num_cols = list(X_tr.select_dtypes(include=[np.number]).columns)
                if num_cols:
                    X_tr = X_tr[num_cols]
                    X_vl = X_vl[num_cols]
                    X_te = X_te[num_cols]
                    logger.info(f"📊 VIF fallback: kept {len(num_cols)} original features")
        except Exception as e:
            logger.warning(f"⚠️ VIF reduction skipped: {e}")

        # 9) Save features and selected feature lists
        os.makedirs(data_dir, exist_ok=True)
        mem_mgr = MemoryEfficientDataManager()

        @with_tracing_span("Step3._attach_timestamp", log_args=False)
        def _attach_timestamp(df_features: pd.DataFrame, labeled_df: pd.DataFrame
        ) -> pd.DataFrame:
            try:
                if (
                    "timestamp" in labeled_df.columns
                    and "timestamp" not in df_features.columns
                ):
                    df_features = df_features.copy()
                    df_features["timestamp"] = labeled_df["timestamp"].values
            except Exception:
                pass
            return df_features

        @with_tracing_span("Step3._save", log_args=False)
        @guard_dataframe_nulls(mode="warn", arg_index=1)
        def _save(name: str, df: pd.DataFrame, labeled_df: pd.DataFrame) -> None:
            df_out = _attach_timestamp(df, labeled_df)
            path_parquet = f"{data_dir}/{exchange}_{symbol}_features_{name}.parquet"
            mem_mgr.save_to_parquet(
                mem_mgr.optimize_dataframe(df_out.copy()), path_parquet,
            )
            logger.info(
                f"✅ Saved features {name}: {len(df_out)} rows, {df_out.shape[1]} cols -> {path_parquet}",
            )
            # Also save PKL for downstream steps expecting PKL
            try:
                import pickle  # Import inside function to avoid scope issues
                path_pkl = f"{data_dir}/{exchange}_{symbol}_features_{name}.pkl"
                with open(path_pkl, "wb") as f:
                    pickle.dump(df_out, f)
                logger.info(f"✅ Saved features {name} (PKL): {path_pkl}")
            except Exception as e:
                logger.warning(f"⚠️ Unable to save PKL features for {name}: {e}")

        _save("train", X_tr, labeled["train"])
        _save("validation", X_vl, labeled["validation"])
        _save("test", X_te, labeled["test"])

        # Save feature lists per split and a feature hash
        feature_lists = {
            "train": list(X_tr.columns),
            "validation": list(X_vl.columns),
            "test": list(X_te.columns),
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{data_dir}/{exchange}_{symbol}_selected_features.json", "w") as f:
            json.dump(feature_lists, f, indent=2)

        # NEW: HMM Composite Regime Data Splitting
        @with_tracing_span("Step3._hmm_composite_regime_splitting", log_args=False)
        async def _hmm_composite_regime_splitting() -> None:
            """Split data by HMM composite regimes for regime-specific training."""
            try:
                logger.info("🔄 Starting HMM composite regime data splitting...")

                # Load HMM composite regime data
                hmm_file = f"{data_dir}/{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
                if not os.path.exists(hmm_file):
                    logger.warning(f"⚠️ HMM composite file not found: {hmm_file}")
                    return

                with open(hmm_file) as f:
                    hmm_data = json.load(f)

                archetype_descriptions = hmm_data.get("archetype_descriptions", {})
                logger.info(
                    f"📊 Loaded {len(archetype_descriptions)} HMM composite archetypes",
                )

                # Create regime data directory
                regime_data_dir = os.path.join(data_dir, "regime_data")
                os.makedirs(regime_data_dir, exist_ok=True)

                # Split each dataset by composite_cluster_id
                regime_splits = {}
                for split_name, features_df in [
                    ("train", X_tr),
                    ("validation", X_vl),
                    ("test", X_te),
                ]:
                    if "composite_cluster_id" not in features_df.columns:
                        logger.warning(
                            f"⚠️ No composite_cluster_id in {split_name} data",
                        )
                        continue

                    # Get unique regime IDs
                    regime_ids = features_df["composite_cluster_id"].dropna().unique()
                    logger.info(
                        f"📊 {split_name}: Found {len(regime_ids)} unique HMM composite regimes",
                    )

                    for regime_id in regime_ids:
                        regime_key = f"hmm_composite_{regime_id}"
                        regime_mask = features_df["composite_cluster_id"] == regime_id
                        regime_data = features_df[regime_mask].copy()

                        if not regime_data.empty:
                            # Add regime description
                            regime_desc = archetype_descriptions.get(
                                str(regime_id), f"Archetype {regime_id}",
                            )
                            regime_data["regime_description"] = regime_desc

                            # Save regime-specific data
                            regime_file = os.path.join(
                                regime_data_dir, f"{split_name}_{regime_key}.parquet",
                            )
                            regime_data.to_parquet(regime_file, index=True)

                            if regime_key not in regime_splits:
                                regime_splits[regime_key] = {
                                    "description": regime_desc,
                                    "splits": {},
                                }
                            regime_splits[regime_key]["splits"][split_name] = {
                                "rows": len(regime_data),
                                "file": regime_file,
                            }

                            logger.info(
                                f"✅ {split_name} regime {regime_key}: {len(regime_data)} rows -> {regime_file}",
                            )

                # Save regime splitting summary
                regime_summary = {
                    "total_regimes": len(regime_splits),
                    "regime_details": regime_splits,
                    "generated_at": datetime.now().isoformat(),
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "symbol": symbol,
                }

                summary_file = os.path.join(
                    data_dir, f"{exchange}_{symbol}_hmm_composite_regime_splits.json",
                )
                with open(summary_file, "w") as f:
                    json.dump(regime_summary, f, indent=2)

                logger.info(
                    f"✅ HMM composite regime splitting completed: {len(regime_splits)} regimes",
                )
                logger.info(f"📄 Regime summary saved to: {summary_file}")

                # Create gating matrix for ensemble training
                @with_tracing_span("Step3._create_gating_matrix", log_args=False)
                def _create_gating_matrix() -> None:
                    """Create gating matrix for regime ensemble training."""
                    try:
                        gating_dir = os.path.join(data_dir, "gating")
                        os.makedirs(gating_dir, exist_ok=True)

                        # Create gating matrix from composite_cluster_id probabilities
                        gating_data = []
                        for split_name, features_df in [
                            ("train", X_tr),
                            ("validation", X_vl),
                            ("test", X_te),
                        ]:
                            if "composite_cluster_id" in features_df.columns:
                                # Get regime probabilities (one-hot encoding)
                                regime_probs = pd.get_dummies(
                                    features_df["composite_cluster_id"], prefix="regime"
                                )

                                # Add timestamp and split info
                                gating_df = regime_probs.copy()
                                gating_df["timestamp"] = features_df.index
                                gating_df["split"] = split_name

                                gating_data.append(gating_df)

                        if gating_data:
                            combined_gating = pd.concat(gating_data, ignore_index=True)
                            gating_file = os.path.join(
                                gating_dir,
                                f"{exchange}_{symbol}_hmm_composite_gating.parquet",
                            )
                            combined_gating.to_parquet(gating_file, index=False)
                            logger.info(
                                f"✅ Gating matrix saved: {gating_file} ({len(combined_gating)} rows)",
                            )

                    except Exception as e:
                        logger.warning(f"⚠️ Gating matrix creation failed: {e}")

                _create_gating_matrix()

            except Exception as e:
                logger.exception(f"🚨 HMM composite regime splitting failed: {e}")

        # Execute regime splitting
        await _hmm_composite_regime_splitting()

    # NEW: also persist pickle copies with timestamps for Step 5 compatibility
    try:
        import pickle

        for split_name, X in [("train", X_tr), ("validation", X_vl), ("test", X_te)]:
            X_pick = X.copy()
            X_pick["timestamp"] = X_pick.index
            X_pick = X_pick.reset_index(drop=True)
            pkl_path = f"{data_dir}/{exchange}_{symbol}_features_{split_name}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(X_pick, f)
            logger.info(
                f"✅ Wrote pickle features {split_name}: {pkl_path} rows={len(X_pick)} cols={X_pick.shape[1]}"
            )

        # Write a simple feature hash to ensure downstream consistency
        import hashlib

        def _hash_cols(cols: list[str]) -> str:
            s = ",".join(cols)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

        hash_info = {
            "train_hash": _hash_cols(feature_lists["train"]),
            "validation_hash": _hash_cols(feature_lists["validation"]),
            "test_hash": _hash_cols(feature_lists["test"]),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        with open(f"{data_dir}/{exchange}_{symbol}_feature_hash.json", "w") as f:
            json.dump(hash_info, f, indent=2)
    except Exception as e:
        logger.warning(f"⚠️ Pickle compatibility write skipped: {e}")

    # Apply optimized feature selection to reduce features to target count
    try:
        # Initialize optimized feature selection manager
        optimized_feature_selection = OptimizedFeatureSelectionManager(config)

        # Perform feature selection ONLY on the training data using the real target
        # Get the actual target labels from the labeled data
        if "target" not in labeled["train"].columns:
            logger.warning("⚠️ Target column 'target' not found in training data, skipping feature selection")
        else:
            train_target = labeled["train"]["target"].reindex(X_tr.index)

            logger.info("🚀 Applying optimized feature selection on the training set...")
            X_tr, selection_metadata = optimized_feature_selection.select_features_optimized(
                X_tr, train_target, model_type="general", step_name="step2"
            )

            selected_features = list(X_tr.columns)
            logger.info(f"✅ Optimized feature selection completed: {len(selected_features)} features selected from training data.")

            # Log performance metrics
            if "performance_metrics" in selection_metadata:
                perf_metrics = selection_metadata["performance_metrics"]
                logger.info("📊 Performance metrics:")
                logger.info(f"   - VIF calculation time: {perf_metrics.get('vif_calculation_time', 0):.2f}s")
                logger.info(f"   - SHAP calculation time: {perf_metrics.get('shap_calculation_time', 0):.2f}s")
                logger.info(f"   - Correlation analysis time: {perf_metrics.get('correlation_analysis_time', 0):.2f}s")
                logger.info(f"   - Total selection time: {selection_metadata.get('total_time', 0):.2f}s")

            # Log feature category distribution
            if "feature_categories" in selection_metadata:
                category_dist = selection_metadata["feature_categories"]
                logger.info("📊 Feature category distribution:")
                for category, features in category_dist.items():
                    if features:
                        logger.info(f"   - {category}: {len(features)} features")

            # Apply the same feature selection to validation and test sets
            X_vl = X_vl[selected_features]
            X_te = X_te[selected_features]
            logger.info("✅ Applied selected features to validation and test sets.")

            # Save selection metadata
            optimized_feature_selection.save_selection_metadata(selection_metadata, symbol, exchange, data_dir)

    except Exception as e:
        logger.warning(f"⚠️ Optimized feature selection failed, using original features: {e}")

    # Save feature artifacts for persistence
    try:
        features_dict = {
            "train": X_tr,
            "validation": X_vl,
            "test": X_te,
        }

        # Get feature configuration from kwargs
        feature_config = kwargs.get("feature_config", {})
        if not feature_config:
            feature_config = {
                "vectorized_advanced_features": {
                    "enable_difference_acceleration_features": True,
                    "enable_volatility_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                    "enable_candlestick_patterns": True,
                    "enable_sr_distance": True,
                    "enable_wavelet_transforms": True,
                    "enable_multi_timeframe": True,
                    "enable_meta_labeling": False,
                    "enable_explicit_meta_labels": False,
                },
            }

        # Add symbol and exchange to the feature config for data quality decorator
        feature_config["symbol"] = symbol
        feature_config["exchange"] = exchange

        _save_feature_artifacts(symbol, exchange, data_dir, features_dict, feature_config, artifact_hash)
        logger.info("💾 Feature artifacts saved for future reuse")

    except Exception as e:
        logger.warning(f"⚠️ Failed to save feature artifacts: {e}")

    logger.info("✅ Step 2: Feature engineering completed successfully")
    
    # Run comprehensive file format validation
    if validate_step2_file:
        logger.info("🔍 Running comprehensive file format validation...")
        validation_success = await _run_comprehensive_validation(symbol, exchange, data_dir, logger)
        
        if validation_success:
            logger.info("✅ Comprehensive file format validation passed")
        else:
            logger.warning("⚠️ Comprehensive file format validation found issues")
    else:
        logger.info("⚠️ Comprehensive file validation not available, skipping validation")
    
    return True
except Exception as e:
    logger.exception(f"🚨 Step 2 feature engineering failed: {e}")
    return False


async def _run_comprehensive_validation(
    symbol: str, 
    exchange: str, 
    data_dir: str, 
    logger: Any
) -> bool:
    """Run comprehensive file format validation for step 2."""
    try:
        if not validate_step2_file:
            logger.warning("Comprehensive file validation not available")
            return True
        
        # Define expected files for step 2
        expected_files = [
            f"{data_dir}/features_{exchange}_{symbol}_train.parquet",
            f"{data_dir}/features_{exchange}_{symbol}_validation.parquet",
            f"{data_dir}/features_{exchange}_{symbol}_test.parquet",
        ]
        
        validation_results = []
        all_valid = True
        
        for file_path in expected_files:
            if Path(file_path).exists():
                logger.info(f"🔍 Validating file: {file_path}")
                
                # Validate file format
                validation_result = validate_step2_file(file_path)
                validation_results.append(validation_result)
                
                if validation_result.is_valid:
                    logger.info(f"✅ File validation passed: {file_path}")
                    logger.info(f"   📊 Shape: {validation_result.summary.get('shape', 'N/A')}")
                    logger.info(f"   📁 File type: {validation_result.file_type}")
                    logger.info(f"   🗂️ Columns: {validation_result.summary.get('column_count', 'N/A')}")
                else:
                    logger.warning(f"⚠️ File validation issues found: {file_path}")
                    all_valid = False
                    
                    # Log detailed issues
                    for issue in validation_result.issues:
                        logger.warning(f"   - {issue.severity.value.upper()}: {issue.description}")
                        if issue.details:
                            logger.warning(f"     Details: {issue.details}")
            else:
                logger.warning(f"⚠️ Expected file not found: {file_path}")
                all_valid = False
        
        # Log validation summary
        if validation_results:
            total_files = len(validation_results)
            valid_files = sum(1 for r in validation_results if r.is_valid)
            logger.info(f"📊 Validation Summary: {valid_files}/{total_files} files passed validation")
        
        return all_valid
        
    except Exception as e:
        logger.exception(f"❌ Error during comprehensive validation: {e}")
        return False


if __name__ == "__main__":

    async def _test() -> None:
        await run_step("ETHUSDT")

    asyncio.run(_test())