# src/training/steps/step9_tactician_specialist_training.py

                            from src.utils.logger import log_io_operation
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score , f1_score
            from src.training.optimized_feature_selection_manager import (import xgboost as xgb
from src.utils.logger import system_logger, import time
from datetime import datetime
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
from typing import Any, import asyncio
import contextlib
import json
import os

                                    import joblib
                                from src.utils.logger import (from src.utils.logger import heartbeat
from src.training.enhanced_training_manager_optimized import (
from src.utils.purged_kfold import PurgedKFoldTime, import gc
                import optuna
            from src.training.enhanced_lm_optimizer import EnhancedLMOptimizer
from src.utils.logger import heartbeat
from src.utils.purged_kfold import PurgedKFoldTime, import lightgbm as lgb
            import optuna
from src.utils.centralized_decorators import (
from src.utils.centralized_decorators import (
from src.utils.warning_symbols import (, import numpy as np
import pandas as pd
import pickle

    PerformanceLevel,
    ValidationLevel,
    adaptive_resource_allocation , comprehensive_validation,
    guard_dataframe_nulls = handle_errors,)
    intelligent_caching = model_validation)
    # Advanced decorators
    performance_monitor)
    pipeline_checkpoint)
    error = )

class TacticianSpecialistTrainingStep:
    """Step 9: Tactician Specialist Models Training with S/R Level Integration."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}

        # Initialize SRBreakoutPredictor for S/R level integration
        self.sr_predictor = SRBreakoutPredictor(config)

        # Initialize enhanced LM optimizer
        self.enhanced_lm_optimizer = None
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            self.enhanced_lm_optimizer = EnhancedLMOptimizer(config)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize enhanced LM optimizer: {e}")

        # Initialize optimized feature selection manager (fallback)
        self.optimized_feature_selection = None
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                OptimizedFeatureSelectionManager = )

            self.optimized_feature_selection = OptimizedFeatureSelectionManager(config)
        except Exception as e:
            self.logger.warning(
                f"⚠️ Failed to initialize optimized feature selection: {e}",
            )

    @handle_errors(
        exceptions=(Exception = ),
        default_return, False = context="tactician specialist training step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the tactician specialist training step."""
        self.logger.info("Initializing Tactician Specialist Training Step...")

        # Initialize SRBreakoutPredictor for S/R level integration
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            sr_init_success = await self.sr_predictor.initialize()
            if sr_init_success:
                self.logger.info(
                    "✅ SRBreakoutPredictor initialized for S/R level integration",
                )
            else:
                self.logger.warning(
                    "⚠️ Failed to initialize SRBreakoutPredictor, continuing without S/R analysis",
                )
        except Exception as e:
            self.logger.warning(f"⚠️ Error initializing SRBreakoutPredictor: {e}")

        self.logger.info(
            "Tactician Specialist Training Step initialized successfully",
        )

    async def _enhance_training_data_with_sr_context(
        self = labeled_data: pd.DataFrame,
        symbol: str = timeframe: str,
    ) -> pd.DataFrame:
        """
        Enhance training data with S/R context and outcome predictions using HMM-aware multi-timeframe analysis.

        Args:
            labeled_data: Original labeled training data
            symbol: Trading symbol
            timeframe: Timeframe (now supports multiple timeframes)

        Returns:
            pd.DataFrame: Enhanced training data with S/R features
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            if labeled_data.empty:
                return labeled_data

            self.logger.info(
                f"🔄 Enhancing training data with HMM-aware S/R context for {timeframe}...",
            )

            # Add S/R context features
            enhanced_data = labeled_data.copy()

            # Check if we have OHLCV data for S/R analysis
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in enhanced_data.columns for col in required_cols):
                self.logger.warning(
                    "⚠️ Missing OHLCV columns for S/R analysis = skipping enhancement",
                )
                return enhanced_data

            # Adaptive sampling based on timeframe
            # Longer timeframes need fewer samples due to lower frequency
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            sample_interval = max(1, len(enhanced_data) // (1000 // timeframe_minutes))
            sample_indices = enhanced_data.index[::sample_interval]

            sr_features = {
                "sr_proximity": [],
                "sr_outcome": [],
                "sr_confidence": [],
                "breakout_probability": [],
                "rebounce_probability": [],
                "consolidation_probability": [],
                "hmm_regime_confidence": [],
                "multi_timeframe_sr_score": [],
            }

            for idx in sample_indices:
                try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                    row = enhanced_data.loc[idx]
                    current_price = row["close"]

                    # Adaptive market context based on timeframe
                    # Longer timeframes need more historical context
                    lookback_bars = min(200, max(50, timeframe_minutes * 2))
                    market_slice = enhanced_data.loc[:idx].tail(lookback_bars)

                    if len(market_slice) < 20:
                        # Default values if insufficient data
                        sr_features["sr_proximity"].append(0.0)
                        sr_features["sr_outcome"].append("consolidation")
                        sr_features["sr_confidence"].append(0.5)
                        sr_features["breakout_probability"].append(0.33)
                        sr_features["rebounce_probability"].append(0.33)
                        sr_features["consolidation_probability"].append(0.34)
                        sr_features["hmm_regime_confidence"].append(0.5)
                        sr_features["multi_timeframe_sr_score"].append(0.5)
                        continue

                    # Get HMM-aware S/R context and outcome prediction
                    sr_context = await self.sr_predictor.get_sr_context(
                        market_slice = current_price,
                    )
                    sr_outcome = await self.sr_predictor.predict_sr_outcome(
                        market_slice = current_price,
                        sr_context = )

                    # Extract HMM regime information if available
                    hmm_confidence = 0.5
                    if "composite_cluster_id" in row:
                        # Use HMM cluster confidence
                        hmm_confidence = row.get("composite_cluster_confidence", 0.5)
                    elif "hmm_cluster_confidence" in row:
                        hmm_confidence = row.get("hmm_cluster_confidence", 0.5)

                    # Extract features
                    is_near_sr = sr_outcome.get("is_near_sr_level", False)
                    sr_features["sr_proximity"].append(1.0 if is_near_sr else 0.0)
                    sr_features["sr_outcome"].append(
                        sr_outcome.get("outcome", "consolidation"),
                    )
                    sr_features["sr_confidence"].append(
                        sr_outcome.get("confidence", 0.5),
                    )

                    probabilities = sr_outcome.get("probabilities", {})
                    sr_features["breakout_probability"].append(
                        probabilities.get("breakout", 0.33),
                    )
                    sr_features["rebounce_probability"].append(
                        probabilities.get("rebounce", 0.33),
                    )
                    sr_features["consolidation_probability"].append(
                        probabilities.get("consolidation", 0.34),
                    )
                    sr_features["hmm_regime_confidence"].append(hmm_confidence)

                    # Multi-timeframe S/R score (combines S/R confidence with HMM regime confidence)
                    sr_conf = sr_outcome.get("confidence", 0.5)
                    multi_tf_score = sr_conf * 0.6 + hmm_confidence * 0.4
                    sr_features["multi_timeframe_sr_score"].append(multi_tf_score)

                except Exception as e:
                    self.logger.debug(
                        f"Error processing S/R features for index {idx}: {e}",
                    )
                    # Default values on error
                    sr_features["sr_proximity"].append(0.0)
                    sr_features["sr_outcome"].append("consolidation")
                    sr_features["sr_confidence"].append(0.5)
                    sr_features["breakout_probability"].append(0.33)
                    sr_features["rebounce_probability"].append(0.33)
                    sr_features["consolidation_probability"].append(0.34)
                    sr_features["hmm_regime_confidence"].append(0.5)
                    sr_features["multi_timeframe_sr_score"].append(0.5)

            # Interpolate S/R features to all data points
            for feature_name , values in sr_features.items():
                if len(values) > 1:
                    # Create series with sampled values
                    feature_series = pd.Series(values, index = sample_indices)

                    # Interpolate to all data points
                    full_feature = (
                        feature_series.reindex(enhanced_data.index)
                        .interpolate(method="linear")
                        .fillna(0.5)
                    )
                    enhanced_data[f"sr_{feature_name}"] = full_feature
                else:
                    # Use constant value if only one sample
                    enhanced_data[f"sr_{feature_name}"] = values[0] if values else 0.5

            # Enhanced sample weights using HMM regime information
            enhanced_data["sr_sample_weight"] = (
                enhanced_data["sr_proximity"] * 0.3
                + enhanced_data["hmm_regime_confidence"] * 0.4
                + 0.3
            )

            self.logger.info(
                f"✅ Enhanced training data with HMM-aware S/R context for {timeframe}: {len(enhanced_data)} samples",
            )
            return enhanced_data

        except Exception as e:
            self.logger.exception(
                f"❌ Error enhancing training data with HMM-aware S/R context: {e}",
            )
            return labeled_data

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes for adaptive processing.
        Step9 only supports 1m and 5m timeframes.

        Args:
            timeframe: Timeframe string (only "1m" or "5m" supported)

        Returns:
            int: Number of minutes
        """
        timeframe = timeframe.lower()
        if timeframe == "1m":
            return 1
        if timeframe == "5m":
            return 5
        # Default to 1 minute if unsupported timeframe
        self.logger.warning(
            f"Unsupported timeframe '{timeframe}' for Step9, defaulting to 1m",
        )
        return 1

    @handle_errors(
        exceptions=(Exception = ),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician specialist training step execution",
    )
    async def execute(
        self = training_input: dict[str, Any],
        pipeline_state: dict[str , Any],
    ) -> dict[str , Any]:
        """
        Execute tactician specialist models training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            self.logger.info("🔄 Executing Tactician Specialist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load tactician labeled data
            labeled_data_dir = f"{data_dir}/tactician_labeled_data"
            labeled_file_parquet = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet"
            )
            labeled_file_pickle = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            )

            if os.path.exists(labeled_file_parquet) or os.path.exists(
                labeled_file_pickle = ):
                if os.path.exists(labeled_file_parquet):
                    # Prefer dataset scan if labeled partition exists
                    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                            ParquetDatasetManager = )

                        pdm = ParquetDatasetManager(logger=self.logger)
                        part_base = os.path.join(data_dir = "parquet", "labeled")
                        if os.path.isdir(part_base):
                            # Validate timeframe for Step9 (only 1m and 5m supported)
                            current_timeframe = training_input.get("timeframe", "1m")
                            if current_timeframe not in ["1m", "5m"]:
                                self.logger.warning(
                                    f"Step9 only supports 1m and 5m timeframes = got: {current_timeframe}",
                                )
                                current_timeframe = "1m"  # Default to 1m

                            filters = [
                                ("exchange", "==", exchange),
                                ("symbol", "==", symbol),
                                ("timeframe", "==", current_timeframe),
                                ("split", "==", "train"),
                            ]
                            # Reader shortcut: prefer materialized projection if available
                            feat_cols = training_input.get(
                                "model_feature_columns",
                            ) or training_input.get("feature_columns")
                            label_col = training_input.get("label_column", "label")
                            proj_base = os.path.join(
                                "data_cache",
                                "parquet",
                                f"proj_features_{training_input.get('model_name', 'default')}",
                            )
                            if (
                                isinstance(feat_cols , list)
                                and len(feat_cols) > 0
                                and os.path.isdir(proj_base)
                            ):
                                proj_filters = [
                                    ("exchange", "==", exchange),
                                    ("symbol", "==", symbol),
                                    (
                                        "timeframe",
                                        "==",
                                        current_timeframe = ),
                                    ("split", "==", "train"),
                                ]
                                cols = ["timestamp", *feat_cols = label_col]
                                labeled_data = pdm.cached_projection(
                                    base_dir, proj_base = filters=proj_filters,
                                    columns, cols = cache_dir="data_cache/projections",
                                    cache_key_prefix=f"proj_features_{training_input.get('model_name','default')}_{exchange}_{symbol}_{current_timeframe}_train",
                                    snapshot_version="v1",
                                    ttl_seconds=3600,
                                    batch_size=131072,
                                    arrow_transform=lambda tbl: (
                                        (
                                            lambda _pa = pc: (
                                                tbl.set_column(
                                                    tbl.schema.get_field_index(
                                                        "timestamp",
                                                    ),
                                                    "timestamp",
                                                    pc.cast(
                                                        tbl.column("timestamp"),
                                                        _pa.int64(),
                                                    ),
                                                )
                                                if (
                                                    "timestamp" in tbl.schema.names
                                                    and not _pa.types.is_int64(
                                                        tbl.schema.field(
                                                            "timestamp",
                                                        ).type = )
                                                )
                                                else tbl
                                            )
                                        )(
                                            __import__("pyarrow"),
                                            __import__("pyarrow.compute"),
                                        )
                                    ),
                                )
                            else:
                                cache_key = f"labeled_{exchange}_{symbol}_{current_timeframe}_train"
                                cols = ["timestamp", *feat_cols = label_col]

                                with heartbeat(
                                    self.logger, name = "Step9 load_labeled_projection",
                                    interval_seconds=60.0,
                                ):
                                    labeled_data = pdm.cached_projection(
                                        base_dir, part_base = filters=filters,
                                        columns, cols = cache_dir="data_cache/projections",
                                        cache_key_prefix, cache_key = snapshot_version="v1",
                                        ttl_seconds=3600,
                                        batch_size=131072,
                                        arrow_transform=lambda tbl: (
                                            (
                                                lambda _pa = pc: (
                                                    tbl.set_column(
                                                        tbl.schema.get_field_index(
                                                            "timestamp",
                                                        ),
                                                        "timestamp",
                                                        pc.cast(
                                                            tbl.column("timestamp"),
                                                            _pa.int64(),
                                                        ),
                                                    )
                                                    if (
                                                        "timestamp" in tbl.schema.names
                                                        and not _pa.types.is_int64(
                                                            tbl.schema.field(
                                                                "timestamp",
                                                            ).type = )
                                                    )
                                                    else tbl
                                                )
                                            )(
                                                __import__("pyarrow"),
                                                __import__("pyarrow.compute"),
                                            )
                                        ),
                                    )
                        else:
                            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                                feat_cols = training_input.get(
                                    "model_feature_columns",
                                ) or training_input.get("feature_columns")
                                label_col = training_input.get("label_column", "label")
                                    log_dataframe_overview = log_io_operation,
                                )

                                if isinstance(feat_cols , list) and len(feat_cols) > 0:
                                    with log_io_operation(
                                        self.logger = "read_parquet",
                                        labeled_file_parquet, columns = True,
                                    ):
                                        labeled_data = pd.read_parquet(
                                            labeled_file_parquet, columns = [
                                                "timestamp",
                                                *feat_cols = label_col,
                                            ],
                                        )
                                else:
                                    with log_io_operation(
                                        self.logger = "read_parquet",
                                        labeled_file_parquet = ):
                                        labeled_data = pd.read_parquet(
                                            labeled_file_parquet = )
                                with contextlib.suppress(Exception):
                                    log_dataframe_overview(
                                        self.logger = labeled_data,
                                        name="labeled_data",
                                    )
                            except Exception:
                                with log_io_operation(
                                    self.logger = "read_parquet",
                                    labeled_file_parquet = ):
                                    labeled_data = pd.read_parquet(labeled_file_parquet)
                    except Exception:
                        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                            feat_cols = training_input.get(
                                "model_feature_columns",
                            ) or training_input.get("feature_columns")
                            label_col = training_input.get("label_column", "label")

                            if isinstance(feat_cols , list) and len(feat_cols) > 0:
                                with log_io_operation(
                                    self.logger = "read_parquet",
                                    labeled_file_parquet, columns = True,
                                ):
                                    labeled_data = pd.read_parquet(
                                        labeled_file_parquet, columns = ["timestamp", *feat_cols = label_col],
                                    )
                            else:
                                with log_io_operation(
                                    self.logger = "read_parquet",
                                    labeled_file_parquet = ):
                                    labeled_data = pd.read_parquet(labeled_file_parquet)
                        except Exception:
                            with log_io_operation(
                                self.logger = "read_parquet",
                                labeled_file_parquet = ):
                                labeled_data = pd.read_parquet(labeled_file_parquet)
                else:
                    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                        with open(labeled_file_pickle = "rb") as f:
                            labeled_data = pickle.load(f)
                    except Exception:
                        pass
            else:
                msg = (
                    "Tactician labeled data not found: "
                    f"{labeled_file_parquet} or {labeled_file_pickle}. Step 9 requires labeled data from Step 8."
                )
                raise FileNotFoundError(msg)

            # Integrate engineered features from Step 3 if available
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                feat_dir = data_dir
                feat_train = os.path.join(
                    feat_dir = f"{exchange}_{symbol}_features_train.pkl",
                )
                feat_val = os.path.join(
                    feat_dir = f"{exchange}_{symbol}_features_validation.pkl",
                )
                feat_test = os.path.join(
                    feat_dir = f"{exchange}_{symbol}_features_test.pkl",
                )
                # Choose appropriate split by inferring from labeled_data
                if isinstance(labeled_data , pd.DataFrame) and not labeled_data.empty:
                    # Align by timestamp if present; else index length heuristic
                    feat_path = None
                    if "split" in labeled_data.columns:
                        split_name = str(labeled_data["split"].mode().iloc[0]).lower()
                        if split_name.startswith("train") and os.path.exists(
                            feat_train = ):
                            feat_path = feat_train
                        elif split_name.startswith("val") and os.path.exists(feat_val):
                            feat_path = feat_val
                        elif split_name.startswith("test") and os.path.exists(
                            feat_test = ):
                            feat_path = feat_test
                    if feat_path is None:
                        # default to train features for augmentation when unknown
                        feat_path = feat_train if os.path.exists(feat_train) else None
                    if feat_path is not None:
                        with open(feat_path = "rb") as f:
                            feat_df = pickle.load(f)
                        if isinstance(feat_df , pd.DataFrame) and not feat_df.empty:
                            # Drop any raw OHLCV in features to avoid duplication
                            feat_df = feat_df.drop(
                                columns=[
                                    c
                                    for c in ["open", "high", "low", "close", "volume"]
                                    if c in feat_df.columns
                                ],
                                errors="ignore",
                            )
                            # Align on timestamp when available
                            if (
                                "timestamp" in labeled_data.columns
                                and "timestamp" in feat_df.columns
                            ):
                                merged = labeled_data.merge(
                                    feat_df, on = "timestamp",
                                    how="left",
                                )
                            else:
                                # Fallback: align by index size
                                feat_df = feat_df.reindex(labeled_data.index)
                                merged = pd.concat([labeled_data = feat_df], axis=1)
                            labeled_data = merged
                            self.logger.info(
                                f"✅ Augmented tactician labeled data with engineered features: +{feat_df.shape[1]} cols",
                            )
            except Exception as _afe:
                self.logger.warning(
                    f"Unable to augment tactician data with engineered features: {_afe}",
                )

            # Convert to DataFrame if needed
            if not isinstance(labeled_data , pd.DataFrame):
                labeled_data = pd.DataFrame(labeled_data)

            # Merge HMM cluster information and timeframe-specific labels
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                # Try to load HMM composite data for the current timeframe
                current_timeframe = training_input.get("timeframe", "1m")
                if current_timeframe not in ["1m", "5m"]:
                    self.logger.warning(
                        f"Step9 only supports 1m and 5m timeframes = got: {current_timeframe}",
                    )
                    current_timeframe = "1m"  # Default to 1m

                # Load HMM composite data for the current timeframe
                hmm_data_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_{current_timeframe}.parquet"
                if os.path.exists(hmm_data_path):
                    hmm_data = pd.read_parquet(hmm_data_path)

                    # Merge HMM cluster information
                    if (
                        "timestamp" in hmm_data.columns
                        and "timestamp" in labeled_data.columns
                    ):
                        hmm_cols = [
                            c
                            for c in hmm_data.columns
                            if c.startswith(("composite_cluster", "hmm_"))
                        ]
                        if hmm_cols:
                            labeled_data = labeled_data.merge(
                                hmm_data[["timestamp", *hmm_cols]],
                                on="timestamp",
                                how="left",
                            )
                            self.logger.info(
                                f"Merged {len(hmm_cols)} HMM cluster columns for {current_timeframe}",
                            )

                # Also try to merge 1m meta-labels if available (for 1m timeframe)
                if current_timeframe == "1m":
                    step4_train = f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl"
                    if os.path.exists(step4_train):
                        with open(step4_train = "rb") as f:
                            step4_df = pickle.load(f)
                        one_m_cols = [
                            c
                            for c in getattr(step4_df = "columns", [])
                            if isinstance(c, str) and c.startswith("1m_")
                        ]
                        if one_m_cols and "timestamp" in step4_df.columns:
                            if "timestamp" in labeled_data.columns:
                                labeled_data = labeled_data.merge(
                                    step4_df[["timestamp", *one_m_cols]],
                                    on="timestamp",
                                    how="left",
                                )
                                self.logger.info(
                                    f"Merged {len(one_m_cols)} 1m meta-label columns into tactician dataset",
                                )
            except Exception as _merr:
                self.logger.warning(
                    f"Skipping HMM cluster and meta-label merge: {_merr}",
                )

            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                shape = getattr(labeled_data = "shape", None)
                self.logger.info(f"Loaded tactician labeled data: shape={shape}")
                if (
                    isinstance(labeled_data , pd.DataFrame)
                    and "tactician_label" in labeled_data.columns
                ):
                    self.logger.info(
                        f"Label distribution: {labeled_data['tactician_label'].value_counts().to_dict()}",
                    )
            except Exception:
                pass

            # Use labeled_data downstream
            # Mandatory: augment features with SR model signals
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                # Load SR models from HMM-based training
                sr_models_dir = os.path.join(data_dir = "enhanced_hmm_models", "SR")
                if not os.path.isdir(sr_models_dir):
                    sr_models_dir = os.path.join(data_dir = "hmm_models", "SR")
                sr_models: dict[str , Any] = {}
                if os.path.isdir(sr_models_dir):
                    for mf in os.listdir(sr_models_dir):
                        if mf.endswith((".pkl", ".joblib")):
                            mp = os.path.join(sr_models_dir = mf)
                            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                                if mf.endswith(".joblib"):

                                    sr_models[mf.replace(".joblib", "")] = joblib.load(
                                        mp = )
                                else:
                                    with open(mp = "rb") as f:
                                        sr_models[mf.replace(".pkl", "")] = pickle.load(
                                            f = )
                            except Exception:
                                continue
                # Compute SR predictions as features
                if sr_models:

                    def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
                        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
                        if obj_cols:
                            df = df.drop(columns=obj_cols)
                        dt_cols = df.select_dtypes(
                            include=["datetime", "datetime64", "datetime64[ns]"],
                        ).columns.tolist()
                        if dt_cols:
                            df = df.drop(columns=dt_cols)
                        return df

                    # Decorate post-definition to preserve closure
                    _ensure_numeric = guard_dataframe_nulls(mode="warn", arg_index=0)(
                        _ensure_numeric = )
                    X_all = _ensure_numeric(
                        labeled_data.drop(
                            columns=[c for c in ["label"] if c in labeled_data.columns],
                            errors="ignore",
                        ),
                    ).select_dtypes(include=[np.number])
                    for name , model in sr_models.items():
                        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                            # Some models may require matching columns; use intersection
                            cols = [
                                c
                                for c in getattr(
                                    model = "feature_names_in_",
                                    X_all.columns = )
                                if c in X_all.columns
                            ]
                            if not cols:
                                continue
                            proba = model.predict_proba(X_all[cols])
                            if proba.shape[1] >= 2:
                                labeled_data[f"sr_sig_{name}_p1"] = proba[:, 1]
                                labeled_data[f"sr_sig_{name}_p0"] = proba[:, 0]
                            else:
                                labeled_data[f"sr_sig_{name}_p1"] = proba.reshape(-1)
                        except Exception:
                            continue
                    self.logger.info(
                        f"✅ Augmented tactician features with {len(sr_models)} SR model signals",
                    )
                else:
                    self.logger.warning(
                        "No SR models found; tactician SR augmentation skipped",
                    )
            except Exception as _e:
                self.logger.warning(f"Tactician SR signal augmentation failed: {_e}")

            # Optionally drop raw S/R features to reduce redundancy (keep SR signals)
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                drop_raw_sr = bool(
                    self.config.get("tactician", {}).get("drop_raw_sr_features", False),
                )
                if drop_raw_sr:
                    sr_raw_cols = [
                        "dist_to_support_pct",
                        "dist_to_resistance_pct",
                        "sr_zone_position",
                        "nearest_support_center",
                        "nearest_resistance_center",
                        "nearest_support_score",
                        "nearest_resistance_score",
                        "nearest_support_band_pct",
                        "nearest_resistance_band_pct",
                        "sr_breakout_up",
                        "sr_breakout_down",
                        "sr_bounce_up",
                        "sr_bounce_down",
                        "sr_touch",
                        "sr_breakout_score",
                        "sr_bounce_score",
                    ]
                    # Do not drop SR model signal columns prefixed with 'sr_sig_' or strength predictions 'sr_pred_'
                    present = [c for c in sr_raw_cols if c in labeled_data.columns]
                    if present:
                        labeled_data = labeled_data.drop(columns=present)
                        self.logger.info(
                            f"🔧 Dropped raw SR features from tactician training: {present}",
                        )
            except Exception as _ed:
                self.logger.warning(f"Unable to drop raw SR features: {_ed}")

            # Enhance training data with HMM-aware S/R context and outcome predictions
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                # Validate timeframe for Step9 (only 1m and 5m supported)
                current_timeframe = training_input.get("timeframe", "1m")
                if current_timeframe not in ["1m", "5m"]:
                    self.logger.warning(
                        f"Step9 only supports 1m and 5m timeframes = got: {current_timeframe}",
                    )
                    current_timeframe = "1m"  # Default to 1m

                enhanced_labeled_data = (
                    await self._enhance_training_data_with_sr_context(
                        labeled_data = symbol,
                        current_timeframe = )
                )
                labeled_data = enhanced_labeled_data
                self.logger.info(
                    f"✅ Enhanced tactician labeled data with HMM-aware S/R context for {current_timeframe}: {len(labeled_data)} samples",
                )
            except Exception as _e:
                self.logger.warning(
                    f"Failed to enhance training data with HMM-aware S/R context: {_e}",
                )

            # Train tactician specialist models

            with heartbeat(
                self.logger, name = "Step9 train_tactician_models",
                interval_seconds=60.0,
            ):
                training_results = await self._train_tactician_models(
                    labeled_data = training_input,
                    pipeline_state = )

            # Save training results
            models_dir = f"{data_dir}/tactician_models"
            os.makedirs(models_dir, exist_ok = True)

            for model_name , model_data in training_results.items():
                model_file = f"{models_dir}/{model_name}.pkl"
                with open(model_file = "wb") as f:
                    pickle.dump(model_data = f)

            # Save training summary
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_tactician_training_summary.json"
            )
            with open(summary_file = "w") as f:
                json.dump(training_results = f, indent=2)

            self.logger.info(
                f"✅ Tactician specialist training completed. Results saved to {models_dir}",
            )

            # Update pipeline state
            pipeline_state["tactician_models"] = training_results

            return {
                "tactician_models": training_results , "models_dir": models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.print(error("❌ Error in Tactician Specialist Training: {e}"))
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _train_tactician_models(
        self = data: pd.DataFrame,
        symbol: str = exchange: str,
    ) -> dict[str , Any]:
        """
        Train tactician specialist models.

        Args:
            data: Labeled data for tactician
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing trained models
        """
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            self.logger.info(
                f"Training tactician specialist models for {symbol} on {exchange}...",
            )

            # Prepare data - handle data types properly
            # Save target columns before dropping object columns
            target_columns = ["tactician_label", "regime"]
            y = data["tactician_label"].copy()

            # First = explicitly drop any datetime columns
            datetime_columns = data.select_dtypes(
                include=["datetime64[ns]", "datetime64", "datetime"],
            ).columns.tolist()
            if datetime_columns:
                self.logger.info(f"Dropping datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)

            # Also drop any object columns that might contain datetime strings
            # But preserve target columns
            object_columns = data.select_dtypes(include=["object"]).columns.tolist()
            object_columns_to_drop = [
                col for col in object_columns if col not in target_columns
            ]
            if object_columns_to_drop:
                self.logger.info(f"Dropping object columns: {object_columns_to_drop}")
                data = data.drop(columns=object_columns_to_drop)

            # Get only numeric columns for features
            excluded_columns = target_columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [
                col for col in numeric_columns if col not in excluded_columns
            ]

            if not feature_columns:
                self.logger.warning(
                    "No numeric feature columns found for tactician training",
                )
                # Create a simple fallback feature
                data["simple_feature"] = np.random.randn(len(data))
                feature_columns = ["simple_feature"]

            X = data[feature_columns].copy()

            # Additional safety check - ensure all columns are numeric
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    self.logger.warning(
                        f"Non-numeric column detected: {col} with dtype {X[col].dtype}",
                    )
                    X = X.drop(columns=[col])
                    feature_columns.remove(col)

            # Remove any remaining NaN values
            X = X.fillna(0)

            # Final check - ensure X is purely numeric
            if X.select_dtypes(include=[np.number]).shape[1] != X.shape[1]:
                self.print(error("Non-numeric columns still present in feature matrix"))
                # Force conversion to numeric = dropping any problematic columns
                X = X.select_dtypes(include=[np.number])

            self.logger.info(
                f"Using {len(feature_columns)} feature columns for tactician training",
            )

            # Split data for training and validation
            # ❌ REMOVED: Stratified split with shuffle (causes data leakage)
            # ✅ IMPLEMENTED: Chronological time-series split (leak-proof)
            split_point = int(len(X) * 0.8)  # 80% train = 20% test
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

            self.logger.info("✅ Using chronological time-series split (leak-proof)")

            # Apply enhanced optimization for tactician models
            if self.enhanced_lm_optimizer is None:
                msg = "Enhanced LM optimizer is required but not initialized"
                raise RuntimeError(
                    msg = )

            self.logger.info(
                "🚀 Applying enhanced LM optimization for tactician models...",
            )

            # Determine model type
            model_type = (
                "classification"
                if y_train.dtype == "object" or len(y_train.unique()) < 10
                else "regression"
            )

            # Apply comprehensive optimization
            (
                optimization_results = optimized_features,
            ) = await self.enhanced_lm_optimizer.optimize_lm_model(
                step_name="step9",
                features_df, X_train = target=y_train,
                model_type, model_type = architecture="LightGBM",  # Primary architecture for tactician
            )

            # Use optimized features directly from the optimizer
            X_train = optimized_features
            X_test = X_test[
                optimized_features.columns
            ]  # Apply same feature selection to test set
            self.logger.info(
                f"✅ Applied feature selection: {len(X_train.columns)} features selected",
            )

            self.logger.info("✅ Enhanced optimization completed for tactician models")
            self.logger.info("📊 Optimization metrics:")
            self.logger.info(
                f"   - Feature selection: {optimization_results.get('feature_selection', {}).get('final_features', len(X_train.columns))} features",
            )
            self.logger.info(
                f"   - Regularization: {optimization_results.get('regularization', {})}",
            )
            self.logger.info(
                f"   - Hyperparameter optimization: {optimization_results.get('hyperparameter_optimization', {})}",
            )

            # Store optimization results
            if not hasattr(self = "enhancement_results"):
                self.enhancement_results = {}
            self.enhancement_results["enhanced_optimization"] = optimization_results

            # Train different model types
            models = {}

            # 1. LightGBM (ensemble model)
            self.logger.info("Pruning features for ensemble models...")
            X_train_ens, X_test_ens = X_train.copy(), X_test.copy()
            X_train_ens, ens_pruning_metadata = (
                pruning_manager.prune_for_step9_tactician(
                    X_train_ens = y_train,
                    "lightgbm",  # Use a representative ensemble model type
                )
            )
            X_test_ens = X_test_ens[X_train_ens.columns]  # Ensure same features

            models["lightgbm"] = await self._train_lightgbm(
                X_train_ens = X_test_ens,
                y_train = y_test,
                symbol = exchange,
            )
            models["lightgbm"]["pruning_metadata"] = ens_pruning_metadata

            # 2. Calibrated Logistic Regression (linear model)
            self.logger.info("Pruning features for linear models...")
            X_train_log, X_test_log = X_train.copy(), X_test.copy()
            X_train_log, log_pruning_metadata = (
                pruning_manager.prune_for_step9_tactician(
                    X_train_log = y_train,
                    "calibrated_logistic",
                )
            )
            X_test_log = X_test_log[X_train_log.columns]  # Ensure same features

            models["calibrated_logistic"] = await self._train_calibrated_logistic(
                X_train_log = X_test_log,
                y_train = y_test,
                symbol = exchange,
            )
            models["calibrated_logistic"]["pruning_metadata"] = log_pruning_metadata

            # 3. XGBoost (ensemble model) - reuse ensemble pruning
            models["xgboost"] = await self._train_xgboost(
                X_train_ens = X_test_ens,
                y_train = y_test,
                symbol = exchange,
            )
            models["xgboost"]["pruning_metadata"] = ens_pruning_metadata

            # 3b. CatBoost (HPO) - ensemble model - reuse ensemble pruning
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                best_cb = await self._hpo_catboost(
                    X_train_ens = X_test_ens,
                    y_train = y_test,
                )
                if best_cb:
                    best_cb["pruning_metadata"] = ens_pruning_metadata
                    models["catboost"] = best_cb
            except Exception:
                pass

            # 4. Random Forest (ensemble model) - reuse ensemble pruning
            models["random_forest"] = await self._train_random_forest(
                X_train_ens = X_test_ens,
                y_train = y_test,
                symbol = exchange,
            )
            models["random_forest"]["pruning_metadata"] = ens_pruning_metadata

            self.logger.info(f"Trained {len(models)} tactician models")

            return models

        except Exception:
            self.print(error("Error training tactician models: {e}"))
            raise

    async def _train_lightgbm(
        self = X_train: pd.DataFrame,
        X_test: pd.DataFrame = y_train: pd.Series,
        y_test: pd.Series = symbol: str,
        exchange: str = ) -> dict[str, Any]:
        """Train LightGBM model."""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Train model with adaptive regularization
            # Calculate adaptive regularization based on data characteristics
            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0

            # Adaptive regularization parameters
            if overfitting_risk > 0.1:  # High overfitting risk
                reg_alpha = 0.1
                reg_lambda = 0.1
                min_child_samples = 50
                subsample = 0.7
            elif overfitting_risk > 0.05:  # Medium overfitting risk
                reg_alpha = 0.05
                reg_lambda = 0.05
                min_child_samples = 30
                subsample = 0.8
            else:  # Low overfitting risk
                reg_alpha = 0.01
                reg_lambda = 0.01
                min_child_samples = 20
                subsample = 0.9

            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                reg_alpha, reg_alpha = # Adaptive L1 regularization
                reg_lambda, reg_lambda = # Adaptive L2 regularization
                min_child_samples, min_child_samples = subsample=subsample,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                early_stopping_rounds=50,
            )

            # Train with validation set
            eval_set = [(X_test = y_test)]
            model.fit(
                X_train = y_train,
                eval_set, eval_set = eval_metric="logloss",
                verbose, False = )

            # Evaluate model
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = accuracy_score(y_test = y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model , "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "LightGBM",
                "symbol": symbol , "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                },
            }

        except Exception:
            self.print(error("Error training LightGBM: {e}"))
            raise

    async def _train_calibrated_logistic(
        self = X_train: pd.DataFrame,
        X_test: pd.DataFrame = y_train: pd.Series,
        y_test: pd.Series = symbol: str,
        exchange: str = ) -> dict[str, Any]:
        """Train Calibrated Logistic Regression model."""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Base logistic regression
            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver="liblinear",
            )

            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(
                estimator, base_model = cv=5,
                method="isotonic",
            )

            # Train model
            calibrated_model.fit(X_train = y_train)

            # Evaluate model
            y_pred = calibrated_model.predict(X_test)
            calibrated_model.predict_proba(X_test)
            accuracy = accuracy_score(y_test = y_pred)

            return {
                "model": calibrated_model , "accuracy": accuracy,
                "feature_importance": {},  # Logistic regression doesn't have direct feature importance
                "model_type": "CalibratedLogisticRegression",
                "symbol": symbol , "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "C": 1.0,
                    "max_iter": 1000,
                    "calibration_method": "isotonic",
                    "cv_folds": 5,
                },
            }

        except Exception:
            self.print(error("Error training Calibrated Logistic Regression: {e}"))
            raise

    async def _train_xgboost(
        self = X_train: pd.DataFrame,
        X_test: pd.DataFrame = y_train: pd.Series,
        y_test: pd.Series = symbol: str,
        exchange: str = ) -> dict[str, Any]:
        """Train XGBoost model."""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Lightweight HPO for XGBoost (subsampled)
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                def _objective(trial: optuna.Trial) -> float:
                    params = {
                        "n_estimators": trial.suggest_int(
                            "n_estimators",
                            100,
                            600,
                            step=100,
                        ),
                        "max_depth": trial.suggest_int("max_depth", 3, 8),
                        "learning_rate": trial.suggest_float(
                            "learning_rate",
                            0.01,
                            0.2,
                            log, True = ),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree",
                            0.6,
                            1.0,
                        ),
                        "reg_alpha": trial.suggest_float(
                            "reg_alpha",
                            1e-8,
                            1e-1,
                            log, True = ),
                        "reg_lambda": trial.suggest_float(
                            "reg_lambda",
                            1e-8,
                            1e-1,
                            log, True = ),
                    }
                    model = xgb.XGBClassifier(
                        **params, random_state = 42,
                        eval_metric="logloss",
                        tree_method="hist",
                        verbosity=0,
                    )
                    # Time-aware CV with purged/embargoed folds and financial surrogate
                    cv = PurgedKFoldTime(
                        n_splits=3,
                        purge=pd.Timedelta(minutes=15),
                        embargo=pd.Timedelta(minutes=10),
                    )
                    scores = []
                    for tr_idx , va_idx in cv.split(X_train):
                        Xs, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                        ys, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                        model.fit(Xs = ys)
                        pred = model.predict(Xv)
                        scores.append(f1_score(yv = pred, average="binary", pos_label=1))
                    return float(np.mean(scores))

                study = optuna.create_study(direction="maximize")
                study.optimize(_objective, n_trials = 15)
                best_params = study.best_params
            except Exception:
                best_params = {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.01,
                    "reg_lambda": 0.01,
                }

            # Train best model on full data
            # Calculate adaptive regularization based on data characteristics
            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0

            # Adaptive regularization parameters
            if overfitting_risk > 0.1:  # High overfitting risk
                reg_alpha = max(0.1, best_params.get("reg_alpha", 0.1))
                reg_lambda = max(0.1, best_params.get("reg_lambda", 0.1))
                min_child_weight = 10
                subsample = 0.7
            elif overfitting_risk > 0.05:  # Medium overfitting risk
                reg_alpha = max(0.05, best_params.get("reg_alpha", 0.05))
                reg_lambda = max(0.05, best_params.get("reg_lambda", 0.05))
                min_child_weight = 5
                subsample = 0.8
            else:  # Low overfitting risk
                reg_alpha = best_params.get("reg_alpha", 0.01)
                reg_lambda = best_params.get("reg_lambda", 0.01)
                min_child_weight = 1
                subsample = 0.9

            model = xgb.XGBClassifier(
                n_estimators=best_params.get("n_estimators", 200),
                max_depth=best_params.get("max_depth", 6),
                learning_rate=best_params.get("learning_rate", 0.05),
                reg_alpha, reg_alpha = # Adaptive L1 regularization
                reg_lambda, reg_lambda = # Adaptive L2 regularization
                min_child_weight, min_child_weight = subsample=best_params.get("subsample", subsample),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                random_state=42,
                eval_metric="logloss",
                early_stopping_rounds=50,
                verbose=0,  # Reduce verbose output during training
            )

            # Train with validation set
            eval_set = [(X_test = y_test)]
            model.fit(X_train = y_train, eval_set=eval_set)

            # Evaluate model
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = accuracy_score(y_test = y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model , "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "symbol": symbol , "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": best_params = }

        except Exception:
            self.print(error("Error training XGBoost: {e}"))
            raise

    async def _hpo_catboost(
        self = X_train: pd.DataFrame,
        X_test: pd.DataFrame = y_train: pd.Series,
        y_test: pd.Series = ) -> dict[str, Any] | None:
        """Lightweight HPO for CatBoost; returns trained model package or None."""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            def _objective(trial: optuna.Trial) -> float:
                params = {
                    "iterations": trial.suggest_int("iterations", 200, 800, step=100),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.2,
                        log, True = ),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "random_seed": 42,
                    "verbose": False = }
                model = CatBoostClassifier(**params)
                cv = PurgedKFoldTime(
                    n_splits=3,
                    purge=pd.Timedelta(minutes=15),
                    embargo=pd.Timedelta(minutes=10),
                )
                scores = []
                for tr_idx , va_idx in cv.split(X_train):
                    Xs, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                    ys, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]
                    model.fit(Xs = ys)
                    pred = model.predict(Xv)
                    scores.append(f1_score(yv = pred, average="binary", pos_label=1))
                return float(np.mean(scores))

            study = optuna.create_study(direction="maximize")
            study.optimize(_objective, n_trials = 15)
            best = study.best_params
            model = CatBoostClassifier(**best)
            model.fit(X_train = y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test = y_pred)
            feature_importance = {}
            with contextlib.suppress(Exception):
                feature_importance = dict(
                    zip(X_train.columns, model.get_feature_importance(), strict=False),
                )
            return {
                "model": model , "accuracy": float(acc),
                "feature_importance": feature_importance,
                "model_type": "CatBoost",
                "symbol": self.config.get("symbol", ""),
                "exchange": self.config.get("exchange", ""),
                "training_date": datetime.now().isoformat(),
                "hyperparameters": best = }
        except Exception:
            return None

    async def _train_random_forest(
        self = X_train: pd.DataFrame,
        X_test: pd.DataFrame = y_train: pd.Series,
        y_test: pd.Series = symbol: str,
        exchange: str = ) -> dict[str, Any]:
        """Train Random Forest model."""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train = y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = accuracy_score(y_test = y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model , "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "RandomForest",
                "symbol": symbol , "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                },
            }

        except Exception:
            self.print(error("Error training Random Forest: {e}"))
            raise

# Import training pipeline decorators for comprehensive security and troubleshooting
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection = debug_training_step,
    deterministic_seed = idempotent_step,
    memory_efficient = nan_inf_and_constant_guard,
    prevent_data_leakage = quality_gate,
    resource_monitor = secure_data_processing,
    time_budget_watchdog = validate_step_output,
    validate_step_prerequisites = )

# For backward compatibility with existing step structure
@deterministic_seed(42)
@idempotent_step(step_key="step9_tactician_specialist_training")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=5400.0)
@performance_monitor(
    enable_profiling, True = enable_memory_tracking=True,
    enable_cpu_tracking, True = save_profile_data=True,
    level=PerformanceLevel.PROFILING = )
@model_validation(
    check_overfitting, True = check_underfitting=True,
    validation_metrics=["accuracy", "precision", "recall", "f1"],
    overfitting_threshold=0.1,
    underfitting_threshold=0.6,
)
@pipeline_checkpoint(
    save_intermediate_results, True = checkpoint_frequency=500,
    enable_rollback, True = )
@intelligent_caching(
    cache_intermediate_results, True = cache_validation_data=True,
    cache_model_artifacts, True = cache_ttl_hours=24,
)
@adaptive_resource_allocation(
    dynamic_memory_allocation, True = adaptive_batch_sizes=True,
    resource_scaling_threshold=0.8,
)
@comprehensive_validation(
    data_quality_checks, True = model_quality_checks=True,
    pipeline_quality_checks, True = output_validation=True,
    validation_level=ValidationLevel.WARNING = )
@validate_step_prerequisites(
    required_directories=["data/training", "models"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "sklearn", "lightgbm", "catboost"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "features", "targets"],
    },
    context="Tactician Specialist Training",
)
@secure_data_processing(
    backup_before, True = integrity_checks=True,
    memory_cleanup, True = data_validation=True,
)
@prevent_data_leakage(
    temporal_validation, True = feature_leakage_detection=True,
    cross_validation_isolation, True = lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    monitor_interval=60.0,
    auto_cleanup, True = )
@memory_efficient(
    chunk_size=10000,
    streaming_processing, True = memory_pool=True,
    cleanup_frequency=25,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling, True = error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=300.0,
    expected_exception, Exception = monitor_interval=60.0,
)
@validate_step_output(
    required_files=["models/{exchange}_{symbol}_tactician_specialist.pkl"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["predictions", "probabilities"],
    },
    performance_thresholds={"training_time_minutes": 120.0, "memory_usage_gb": 8.0},
    format_validation, True = )
@quality_gate(
    model_performance_thresholds={"accuracy": 0.6, "f1_score": 0.5},
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    convergence_checks, True = overfitting_detection=True,
    validation_score_requirements={"cross_validation_score": 0.6},
)
async def run_step(
    symbol: str = exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool, False = **kwargs,
) -> bool:
    """
    Run the tactician specialist training step - IMPROVED VERSION.

    IMPROVEMENTS:
    - Enhanced configuration management with validation
    - Better error handling and logging
    - Performance monitoring and metrics
    - Memory management and cleanup
    - Parallel processing capabilities
    - Advanced model training and validation
    - S/R level integration optimization

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        force_rerun: Force rerun flag
        **kwargs: Additional parameters

    Returns:
        bool: True if successful = False otherwise
    """

    start_time = time.time()

    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        # Enhanced configuration with validation
        config = {
            "symbol": symbol , "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            "enable_parallel_processing": kwargs.get(
                "enable_parallel_processing",
                True = ),
            "max_workers": kwargs.get("max_workers", 4),
            "memory_limit_gb": kwargs.get("memory_limit_gb", 16.0),
            "enable_early_stopping": kwargs.get("enable_early_stopping", True),
            "enable_model_checkpointing": kwargs.get(
                "enable_model_checkpointing",
                True = ),
            "validation_split": kwargs.get("validation_split", 0.2),
            "test_split": kwargs.get("test_split", 0.2),
            "random_state": kwargs.get("random_state", 42),
            "batch_size": kwargs.get("batch_size", 64),
            "learning_rate": kwargs.get("learning_rate", 1e-4),
            "epochs": kwargs.get("epochs", 100),
            "early_stopping_patience": kwargs.get("early_stopping_patience", 10),
            "sr_integration": {
                "enable_sr_analysis": kwargs.get("enable_sr_analysis", True),
                "sr_lookback_periods": kwargs.get("sr_lookback_periods", 100),
                "sr_confidence_threshold": kwargs.get("sr_confidence_threshold", 0.7),
            },
            "model_architecture": {
                "type": kwargs.get("model_type", "CNN"),
                "layers": kwargs.get("model_layers", [64, 32, 16]),
                "dropout": kwargs.get("dropout_rate", 0.2),
                "activation": kwargs.get("activation", "relu"),
            },
        }

        # Validate configuration
        if not config["symbol"]:
            msg = "Symbol cannot be empty"
            raise ValueError(msg)

        if not config["exchange"]:
            msg = "Exchange cannot be empty"
            raise ValueError(msg)

        if not config["data_dir"]:
            msg = "Data directory cannot be empty"
            raise ValueError(msg)

        if config["memory_limit_gb"] <= 0:
            msg = "Memory limit must be positive"
            raise ValueError(msg)

        if config["max_workers"] <= 0:
            msg = "Max workers must be positive"
            raise ValueError(msg)

        system_logger.info(
            "🚀 Starting Tactician Specialist Training step - IMPROVED VERSION",
        )
        system_logger.info(f"📋 Configuration: {len(config)} parameters")
        system_logger.info(f"   - Symbol: {symbol}")
        system_logger.info(f"   - Exchange: {exchange}")
        system_logger.info(
            f"   - Parallel processing: {'Enabled' if config['enable_parallel_processing'] else 'Disabled'}",
        )
        system_logger.info(
            f"   - S/R integration: {'Enabled' if config['sr_integration']['enable_sr_analysis'] else 'Disabled'}",
        )
        system_logger.info(
            f"   - Model architecture: {config['model_architecture']['type']}",
        )

        # Create step instance with enhanced error handling
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            step = TacticianSpecialistTrainingStep(config)
            await step.initialize()
            system_logger.info(
                "✅ Tactician specialist training step initialized successfully",
            )
        except Exception as e:
            system_logger.error(
                f"❌ Failed to initialize tactician specialist training step: {e}",
            )
            raise

        # Execute step with enhanced monitoring
        training_input = {
            "symbol": symbol , "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs = }

        pipeline_state = {}

        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            result = await step.execute(training_input = pipeline_state)

            if result.get("status") == "SUCCESS":
                # Log completion metrics
                total_time = time.time() - start_time
                system_logger.info(
                    "✅ Tactician specialist training step completed successfully",
                )
                system_logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
                system_logger.info(f"   📊 Configuration: {len(config)} parameters")
                system_logger.info(
                    f"   🔧 Parallel processing: {'Enabled' if config['enable_parallel_processing'] else 'Disabled'}",
                )

                # Log result details if available
                if "metrics" in result:
                    metrics = result["metrics"]
                    system_logger.info("   📈 Training metrics:")
                    for metric_name , metric_value in metrics.items():
                        system_logger.info(f"      - {metric_name}: {metric_value}")

                # Memory cleanup

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                return True
            error_msg = result.get("error", "Unknown error")
            system_logger.error(
                f"❌ Tactician specialist training step failed: {error_msg}",
            )
            return False

        except Exception as e:
            system_logger.error(
                f"❌ Error during tactician specialist training execution: {e}",
            )
            return False

    except Exception as e:
        total_time = time.time() - start_time
        system_logger.error(f"❌ Error in tactician specialist training step: {e}")
        system_logger.error(f"   Execution time: {total_time:.2f}s")
        return False

if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
