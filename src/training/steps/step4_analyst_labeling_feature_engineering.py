# src/training/steps/step4_analyst_labeling_feature_engineering.py

import asyncio
import json
import os
import pickle
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
from src.training.steps.unified_data_loader import get_unified_data_loader


class AnalystLabelingFeatureEngineeringStep:
    """Step 4: Analyst Labeling and Feature Engineering using Vectorized Orchestrator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("AnalystLabelingFeatureEngineeringStep")
        self.orchestrator = None

        # Standardized column groups
        self._RAW_PRICE_COLUMNS = [
            "open",
            "high",
            "low",
            "close",
            "avg_price",
            "min_price",
            "max_price",
        ]
        self._RAW_VOLUME_COLUMNS = [
            "volume",
            "trade_volume",
            "trade_count",
            "volume_ratio",
        ]
        self._RAW_MICROSTRUCTURE_COLUMNS = [
            "market_depth",
            "bid_ask_spread",
        ]
        # Any raw-like context that must never leak into features directly
        self._RAW_CONTEXT_COLUMNS = (
            self._RAW_PRICE_COLUMNS
            + self._RAW_VOLUME_COLUMNS
            + self._RAW_MICROSTRUCTURE_COLUMNS
            + ["funding_rate"]
        )
        # Metadata/non-feature columns
        self._METADATA_COLUMNS = [
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_month",
            "quarter",
            "exchange",
            "symbol",
            "timeframe",
            "split",
        ]

    def _zscore(self, series: pd.Series, window: int = 50) -> pd.Series:
        roll = series.rolling(window=window, min_periods=max(5, window // 5))
        z = (series - roll.mean()) / roll.std().replace(0, np.nan)
        return z.replace([np.inf, -np.inf], np.nan)

    async def _build_pipeline_a_stationary(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline A: create stationary features directly from raw series made stationary."""
        try:
            df = price_data.copy()
            features = pd.DataFrame(index=df.index)

            # Core stationary transforms
            if "close" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["close_returns"] = df["close"].pct_change()

            # Prefer trade_volume if present, else volume (readable control flow)
            if "trade_volume" in df.columns:
                vol_col = "trade_volume"
            elif "volume" in df.columns:
                vol_col = "volume"
            else:
                vol_col = None
            if vol_col is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["volume_returns"] = df[vol_col].pct_change()

            # Funding rates dynamics
            if "funding_rate" in df.columns:
                features["funding_rate_change"] = df["funding_rate"].diff()
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["funding_rate_returns"] = df["funding_rate"].pct_change()

            # Volatility of stationary series
            if "close_returns" in features.columns:
                features["returns_volatility_20"] = features["close_returns"].rolling(20).std()
            if "volume_returns" in features.columns:
                features["volume_returns_volatility_20"] = features["volume_returns"].rolling(20).std()

            # Simple interactive features
            if "close_returns" in features.columns and "volume_returns" in features.columns:
                features["returns_x_volume_returns"] = (
                    features["close_returns"] * features["volume_returns"]
                )

            # Additional upstream standardization to returns/changes for available raw series
            if "trade_count" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["trade_count_returns"] = df["trade_count"].pct_change()
            if "trade_volume" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["trade_volume_returns"] = df["trade_volume"].pct_change()
            if "market_depth" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["market_depth_returns"] = df["market_depth"].pct_change()
            if "bid_ask_spread" in df.columns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    features["bid_ask_spread_returns"] = df["bid_ask_spread"].pct_change()

            # Cleanup
            num = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            features[num.columns] = num
            features = features.fillna(0)
            # Drop any all-NaN columns defensively
            features = features.dropna(axis=1, how="all")
            return features
        except Exception as e:
            self.logger.warning(f"Pipeline A failed: {e}")
            return pd.DataFrame(index=price_data.index)

    async def _build_pipeline_b_ohlcv(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline B: compute raw OHLCV indicators then transform to stationary features."""
        try:
            required = ["open", "high", "low", "close"]
            if not all(c in price_data.columns for c in required):
                self.logger.warning("Pipeline B skipped due to missing OHLCV columns")
                return pd.DataFrame(index=price_data.index)

            df = price_data.copy()
            feats = pd.DataFrame(index=df.index)

            close = df["close"].astype(float)
            open_ = df["open"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # SMA distances (stationary)
            sma20 = close.rolling(20, min_periods=1).mean()
            sma50 = close.rolling(50, min_periods=1).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                feats["sma20_distance"] = (close / sma20) - 1.0
                feats["sma50_distance"] = (close / sma50) - 1.0

            # Garman–Klass volatility -> returns of vol
            with np.errstate(divide="ignore", invalid="ignore"):
                log_hl = np.log(high / low)
                log_co = np.log(close / open_)
            gk_var = (log_hl ** 2) / (2 * np.log(2)) - (2 * np.log(2) - 1) * (log_co ** 2)
            gk_vol = np.sqrt(gk_var.clip(lower=0))
            with np.errstate(divide="ignore", invalid="ignore"):
                feats["gk_vol_returns"] = (gk_vol / gk_vol.shift(1)) - 1.0

            # Simple candlestick strength (engulfing bullish/bearish) -> z-scored
            body = (close - open_).abs()
            body_prev = body.shift(1)
            # Bullish engulfing
            is_current_bullish = close > open_
            is_previous_bearish = df["close"].shift(1) < df["open"].shift(1)
            is_bull_engulf = (open_ < df["close"].shift(1)) & (close > df["open"].shift(1))
            bull_cond = is_current_bullish & is_previous_bearish & is_bull_engulf
            bull_strength = (body / (body_prev.replace(0, np.nan))).where(bull_cond, 0.0)
            feats["bullish_engulf_strength_z"] = self._zscore(bull_strength, window=50)
            # Bearish engulfing
            is_current_bearish = close < open_
            is_previous_bullish = df["close"].shift(1) > df["open"].shift(1)
            is_bear_engulf = (open_ > df["close"].shift(1)) & (close < df["open"].shift(1))
            bear_cond = is_current_bearish & is_previous_bullish & is_bear_engulf
            bear_strength = (body / (body_prev.replace(0, np.nan))).where(bear_cond, 0.0)
            feats["bearish_engulf_strength_z"] = self._zscore(bear_strength, window=50)

            # Cleanup
            num = feats.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            feats[num.columns] = num
            feats = feats.fillna(0)
            feats = feats.dropna(axis=1, how="all")
            return feats
        except Exception as e:
            self.logger.warning(f"Pipeline B failed: {e}")
            return pd.DataFrame(index=price_data.index)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst labeling and feature engineering step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst labeling and feature engineering step."""
        self.logger.info(
            "Initializing Analyst Labeling and Feature Engineering Step...",
        )

        # Initialize the vectorized labeling orchestrator
        from src.training.steps.vectorized_labelling_orchestrator import (
            VectorizedLabellingOrchestrator,
        )

        self.orchestrator = VectorizedLabellingOrchestrator(self.config)
        await self.orchestrator.initialize()

        self.logger.info(
            "Analyst Labeling and Feature Engineering Step initialized successfully",
        )

    async def _validate_and_enhance_features(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and enhance features for the 240+ feature set."""
        try:
            self.logger.info("🔍 Validating and enhancing features for 240+ feature set...")
            
            # Separate features from labels
            feature_columns = [col for col in labeled_data.columns if col != 'label']
            features_df = labeled_data[feature_columns]
            
            # CRITICAL: Enhanced feature validation with detailed reporting
            constant_features = []
            low_variance_features = []
            valid_features = []
            problematic_features = []
            
            self.logger.info(f"📊 Starting feature validation for {len(feature_columns)} features...")
            
            for col in feature_columns:
                try:
                    # Skip non-numeric columns during variance checks (e.g., datetime/timestamp/strings)
                    if not np.issubdtype(features_df[col].dtype, np.number):
                        # Coerce categoricals to string and skip validation for them
                        try:
                            if str(features_df[col].dtype).startswith("category"):
                                features_df[col] = features_df[col].astype(str)
                        except Exception:
                            pass
                        continue
                    feature_values = features_df[col].dropna()
                    
                    if len(feature_values) == 0:
                        constant_features.append(f"{col} (all NaN values)")
                        continue
                    
                    # Check for constant features
                    unique_values = feature_values.unique()
                    if len(unique_values) <= 1:
                        constant_value = unique_values[0] if len(unique_values) == 1 else "NaN"
                        constant_features.append(f"{col} (constant value: {constant_value})")
                        continue
                    
                    # Check for very low variance features
                    variance = feature_values.var()
                    if variance < 1e-8:  # Very low variance threshold
                        low_variance_features.append(f"{col} (variance: {variance:.2e})")
                        continue
                    
                    # Check for binary features with only one value
                    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                        # Binary feature with both 0 and 1 - this is perfect!
                        valid_features.append(col)
                    elif len(unique_values) == 2:
                        # Binary feature with unexpected values
                        problematic_features.append(f"{col} (binary feature with values: {unique_values})")
                        continue
                    else:
                        # Continuous feature with sufficient variation
                        valid_features.append(col)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error validating feature '{col}': {e}")
                    problematic_features.append(f"{col} (validation error: {e})")
                    continue
            
            # Log comprehensive feature quality metrics
            self.logger.info(f"📊 Feature quality analysis:")
            self.logger.info(f"   📊 Total features: {len(feature_columns)}")
            self.logger.info(f"   ✅ Valid features: {len(valid_features)}")
            self.logger.info(f"   {'✅' if len(constant_features) == 0 else '🚨'} Constant features: {len(constant_features)} (SHOULD BE 0)")
            self.logger.info(f"   {'✅' if len(low_variance_features) == 0 else '⚠️'} Low-variance features: {len(low_variance_features)}")
            self.logger.info(f"   {'✅' if len(problematic_features) == 0 else '⚠️'} Problematic features: {len(problematic_features)}")
            
            # CRITICAL: If we have constant features, this indicates a serious issue
            if constant_features:
                self.logger.error(
                    f"🚨 CRITICAL: {len(constant_features)} constant features: "
                    + ", ".join([cf.split(" (" )[0] for cf in constant_features[:50]])
                    + (" ..." if len(constant_features) > 50 else "")
                )
                
                # Provide diagnostic information
                self.logger.error(f"🚨 DIAGNOSTIC INFORMATION:")
                self.logger.error(f"   - Check if time-series data is being processed correctly")
                self.logger.error(f"   - Verify feature engineering is not flattening arrays")
                self.logger.error(f"   - Ensure proper data length validation")
                self.logger.error(f"   - Review the orchestrator feature combination logic")
                
                # Remove constant features to prevent model failure
                features_to_remove = [col.split(" (constant value:")[0] for col in constant_features]
                labeled_data = labeled_data.drop(columns=features_to_remove)
                self.logger.warning(f"🗑️ Removed {len(features_to_remove)} constant features to prevent model failure")
            
            # Remove low variance features
            if low_variance_features:
                low_var_features_to_remove = [col.split(" (variance:")[0] for col in low_variance_features]
                labeled_data = labeled_data.drop(columns=low_var_features_to_remove)
                self.logger.info(f"🗑️ Removed {len(low_var_features_to_remove)} low-variance features")
            
            # Remove problematic features
            if problematic_features:
                names = [col.split(" (")[0] for col in problematic_features]
                labeled_data = labeled_data.drop(columns=names)
                self.logger.warning(
                    f"🗑️ Removed {len(names)} problematic features: " + ", ".join(names[:50]) + (" ..." if len(names) > 50 else "")
                )
            
            # Check for NaN values and handle them
            nan_counts = labeled_data.isnull().sum()
            features_with_nans = nan_counts[nan_counts > 0]
            
            if len(features_with_nans) > 0:
                self.logger.warning(f"⚠️ Found {len(features_with_nans)} features with NaN values")
                # Fill NaN values with 0 for numerical features
                numeric_columns = labeled_data.select_dtypes(include=[np.number]).columns
                labeled_data[numeric_columns] = labeled_data[numeric_columns].fillna(0)
                self.logger.info("✅ Filled NaN values with 0")
            
            # Check for infinite values
            inf_counts = np.isinf(labeled_data.select_dtypes(include=[np.number])).sum()
            features_with_infs = inf_counts[inf_counts > 0]
            
            if len(features_with_infs) > 0:
                self.logger.warning(f"⚠️ Found {len(features_with_infs)} features with infinite values")
                # Replace infinite values with large finite values
                labeled_data = labeled_data.replace([np.inf, -np.inf], [1e6, -1e6])
                self.logger.info("✅ Replaced infinite values with finite bounds")
            
            # Remove raw OHLCV columns to prevent data leakage
            raw_ohlcv_columns = [c for c in self._RAW_CONTEXT_COLUMNS if c in ["open","high","low","close","volume","avg_price","min_price","max_price"]]
            ohlcv_columns_found = [col for col in raw_ohlcv_columns if col in labeled_data.columns]
            if ohlcv_columns_found:
                labeled_data = labeled_data.drop(columns=ohlcv_columns_found)
                self.logger.warning(f"🚨 CRITICAL: Found raw OHLCV columns in features: {ohlcv_columns_found}")
                self.logger.warning(f"🚨 Removed raw OHLCV columns to prevent data leakage!")
                self.logger.warning(f"🚨 This indicates the orchestrator is including raw price data!")
            
            # Final validation check
            remaining_features = [col for col in labeled_data.columns if col != 'label']
            self.logger.info(f"✅ Final feature validation complete:")
            self.logger.info(f"   📊 Remaining features: {len(remaining_features)}")
            self.logger.info(f"   📊 Total samples: {len(labeled_data)}")
            self.logger.info(f"   🗑️ Raw OHLCV columns removed: {ohlcv_columns_found if ohlcv_columns_found else 'None'}")
            
            if len(remaining_features) < 10:
                self.logger.error(f"🚨 CRITICAL: Only {len(remaining_features)} features remaining!")
                self.logger.error(f"🚨 This indicates a severe feature engineering failure!")
                self.logger.error(f"🚨 The model will likely fail to train properly!")
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Feature validation failed: {e}")
            return labeled_data

    async def _log_feature_engineering_summary(self, labeled_data: pd.DataFrame) -> None:
        """Log concise feature engineering summary."""
        try:
            feature_columns = [col for col in labeled_data.columns if col != 'label']
            
            self.logger.info(f"📊 Feature engineering completed: {len(feature_columns)} features, {len(labeled_data)} samples")
            
            # Log only basic label distribution if available
            if 'label' in labeled_data.columns:
                label_distribution = labeled_data['label'].value_counts()
                self.logger.info(f"🎯 Label distribution: {dict(label_distribution)}")
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering summary logging failed: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst labeling and feature engineering step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst labeling and feature engineering step using vectorized orchestrator.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            dict: Updated pipeline state
        """
        self.logger.info("Starting analyst labeling and feature engineering...")

        try:
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            timeframe = training_input.get("timeframe", "1m")

            # Use unified data loader to get comprehensive data for feature engineering
            self.logger.info("🔄 Loading unified data...")
            data_loader = get_unified_data_loader(self.config)

            # Load unified data with optimizations for ML training (use configured lookback)
            lookback_days = self.config.get("lookback_days", 180)
            price_data = await data_loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=lookback_days,
                use_streaming=True,  # Enable streaming for large datasets
            )

            if price_data is None or price_data.empty:
                self.logger.error(f"No unified data found for {symbol} on {exchange}")
                return {
                    "status": "FAILED",
                    "error": f"No unified data found for {symbol} on {exchange}",
                }

            # Log data information
            data_info = data_loader.get_data_info(price_data)
            self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")

            # Ensure price_data is a DataFrame
            if not isinstance(price_data, pd.DataFrame):
                price_data = pd.DataFrame(price_data)

            # Validate that we have proper OHLCV data for triple barrier labeling
            required_ohlcv_columns = ["open", "high", "low", "close", "volume"]
            missing_ohlcv = [
                col for col in required_ohlcv_columns if col not in price_data.columns
            ]

            if missing_ohlcv:
                self.logger.error(f"Missing required OHLCV columns: {missing_ohlcv}")
                self.logger.error(
                    "Cannot perform proper triple barrier labeling without OHLCV data."
                )
                self.logger.error(f"Available columns: {list(price_data.columns)}")
                return {
                    "status": "FAILED",
                    "error": f"Missing OHLCV columns: {missing_ohlcv}",
                }

            self.logger.info("✅ Validated OHLCV data")

            # Dual-pipeline feature engineering
            self.logger.info("🔀 Building Pipeline A (Stationary) and Pipeline B (OHLCV->Stationary)...")
            features_a = await self._build_pipeline_a_stationary(price_data)
            features_b = await self._build_pipeline_b_ohlcv(price_data)
            combined_features = pd.concat([features_a, features_b], axis=1)

            # Label using optimized triple barrier on raw OHLCV, then align features to labeled index
            try:
                from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
                    OptimizedTripleBarrierLabeling,
                )
                tb_config = self.config.get("vectorized_labelling_orchestrator", {})
                labeler = OptimizedTripleBarrierLabeling(
                    profit_take_multiplier=tb_config.get("profit_take_multiplier", 0.002),
                    stop_loss_multiplier=tb_config.get("stop_loss_multiplier", 0.001),
                    time_barrier_minutes=tb_config.get("time_barrier_minutes", 30),
                    max_lookahead=tb_config.get("max_lookahead", 100),
                )
                labeled_ohlcv = labeler.apply_triple_barrier_labeling_vectorized(
                    price_data[["open", "high", "low", "close", "volume"]].copy()
                )
                # Align feature rows to labeled rows (binary classification removes HOLD rows)
                combined_features = combined_features.loc[labeled_ohlcv.index]
                labeled_data = pd.concat([combined_features, labeled_ohlcv[["label"]]], axis=1)
            except Exception as e:
                self.logger.warning(f"Triple barrier labeling failed, using fallback labels: {e}")
                fallback = self._create_fallback_labeled_data(price_data)
                labeled_data = fallback.get("data", price_data)

            # Final feature sanity: drop raw OHLCV if somehow present
            raw_cols = list(set(self._RAW_CONTEXT_COLUMNS))
            labeled_data = labeled_data.drop(columns=[c for c in raw_cols if c in labeled_data.columns], errors="ignore")

            # Validate feature quality
            labeled_data = await self._validate_and_enhance_features(labeled_data)

            # Drop datetime/timestamp and metadata columns
            try:
                datetime_cols = [
                    c for c in labeled_data.columns if str(labeled_data[c].dtype).startswith("datetime64") or "timestamp" in c.lower()
                ]
                if datetime_cols:
                    self.logger.info(f"Removing datetime columns prior to saving/validation: {datetime_cols}")
                    labeled_data = labeled_data.drop(columns=datetime_cols)
            except Exception:
                pass

            meta_cols = [c for c in labeled_data.columns if c in self._METADATA_COLUMNS]
            if meta_cols:
                self.logger.info(f"Removing metadata columns from features: {meta_cols}")
                labeled_data = labeled_data.drop(columns=meta_cols)
 
            # Final guard: drop metadata columns again prior to splitting/saving
            try:
                drop_meta = [c for c in self._METADATA_COLUMNS if c in labeled_data.columns]
                if drop_meta:
                    labeled_data = labeled_data.drop(columns=drop_meta)
            except Exception:
                pass

            # Split the data into train/validation/test sets (80/10/10 split)
            total_rows = len(labeled_data)
            train_end = int(total_rows * 0.8)
            val_end = int(total_rows * 0.9)

            train_data = labeled_data.iloc[:train_end]
            validation_data = labeled_data.iloc[train_end:val_end]
            test_data = labeled_data.iloc[val_end:]

            # Persist and log selected feature lists per split (traceability)
            try:
                selected_features = {
                    "train": [c for c in train_data.columns if c != "label"],
                    "validation": [c for c in validation_data.columns if c != "label"],
                    "test": [c for c in test_data.columns if c != "label"],
                }
                trace_path = f"{data_dir}/{exchange}_{symbol}_selected_features.json"
                with open(trace_path, "w") as jf:
                    json.dump(selected_features, jf, indent=2)
                self.logger.info(f"🔎 Saved feature lists to {trace_path}")
            except Exception as e:
                self.logger.warning(f"Could not persist selected feature lists: {e}")
            # Save feature files that the validator expects
            feature_files = [
                (f"{data_dir}/{exchange}_{symbol}_features_train.pkl", train_data),
                (
                    f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                    validation_data,
                ),
                (f"{data_dir}/{exchange}_{symbol}_features_test.pkl", test_data),
            ]

            # For features files, drop raw OHLCV/trade columns to avoid leakage
            raw_cols = list(set(self._RAW_CONTEXT_COLUMNS))
            for file_path, data in feature_files:
                features_df = data.drop(columns=[c for c in raw_cols if c in data.columns], errors="ignore")
                with open(file_path, "wb") as f:
                    pickle.dump(features_df, f)
            self.logger.info(f"✅ Saved feature data files")

            # Also save Parquet versions with downcasting for efficiency
            try:
                from src.training.enhanced_training_manager_optimized import (
                    MemoryEfficientDataManager,
                )

                mem_mgr = MemoryEfficientDataManager()
                parquet_files = [
                    (
                        f"{data_dir}/{exchange}_{symbol}_features_train.parquet",
                        mem_mgr.optimize_dataframe(train_data.copy()),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_features_validation.parquet",
                        mem_mgr.optimize_dataframe(validation_data.copy()),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_features_test.parquet",
                        mem_mgr.optimize_dataframe(test_data.copy()),
                    ),
                ]
                for file_path, df in parquet_files:
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )
                        import json as _json

                        feature_cols = list(df.columns)
                        if "label" in feature_cols:
                            feature_cols.remove("label")
                        metadata = {
                            "schema_version": "1",
                            "feature_list": _json.dumps(feature_cols),
                            "feature_hash": str(hash(tuple(sorted(feature_cols))))[:16],
                            "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                            "generator_commit": training_input.get(
                                "generator_commit", ""
                            ),
                        }
                        # Arrow-native cast on timestamp before write
                        import pyarrow as _pa, pyarrow.compute as pc

                        table = _pa.Table.from_pandas(df, preserve_index=False)
                        if "timestamp" in table.schema.names and not _pa.types.is_int64(
                            table.schema.field("timestamp").type
                        ):
                            table = table.set_column(
                                table.schema.get_field_index("timestamp"),
                                "timestamp",
                                pc.cast(table.column("timestamp"), _pa.int64()),
                            )
                        df = table.to_pandas(types_mapper=pd.ArrowDtype)
                        # Drop raw OHLCV for feature parquet
                        drop_cols = [c for c in self._RAW_CONTEXT_COLUMNS if c in df.columns]
                        if drop_cols:
                            df = df.drop(columns=drop_cols)
                        ParquetDatasetManager(self.logger).write_flat_parquet(
                            df,
                            file_path,
                            compression="snappy",
                        )
                    except Exception:
                        from src.utils.logger import (
                            log_io_operation,
                            log_dataframe_overview,
                        )

                        with log_io_operation(
                            self.logger, "to_parquet", file_path, compression="snappy"
                        ):
                            df.to_parquet(file_path, compression="snappy", index=False)
                        try:
                            log_dataframe_overview(
                                self.logger, df, name=f"features_{split_name}"
                            )
                        except Exception:
                            pass
                    pass  # Reduced logging
                # Also write partitioned dataset for features
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    base_dir = os.path.join(data_dir, "parquet", "features")
                    for split_name, df in (
                        ("train", mem_mgr.optimize_dataframe(train_data.copy())),
                        (
                            "validation",
                            mem_mgr.optimize_dataframe(validation_data.copy()),
                        ),
                        ("test", mem_mgr.optimize_dataframe(test_data.copy())),
                    ):
                        df = df.copy()
                        df["exchange"] = exchange
                        df["symbol"] = symbol
                        df["timeframe"] = training_input.get("timeframe", "1m")
                        df["split"] = split_name
                        import json as _json

                        feat_cols = list(df.columns)
                        if "label" in feat_cols:
                            feat_cols.remove("label")
                        # Drop raw OHLCV for feature parquet (partitioned)
                        drop_cols = [c for c in self._RAW_CONTEXT_COLUMNS if c in df.columns]
                        if drop_cols:
                            df = df.drop(columns=drop_cols)
                        pdm.write_partitioned_dataset(
                            df=df,
                            base_dir=base_dir,
                            partition_cols=[
                                "exchange",
                                "symbol",
                                "timeframe",
                                "split",
                                "year",
                                "month",
                                "day",
                            ],
                            schema_name="features",
                            compression="snappy",
                            metadata={
                                "schema_version": "1",
                                "feature_list": _json.dumps(feat_cols),
                                "feature_hash": str(hash(tuple(sorted(feat_cols))))[
                                    :16
                                ],
                                "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                                "generator_commit": training_input.get(
                                    "generator_commit", ""
                                ),
                            },
                        )
                except Exception as _e:
                    self.logger.warning(f"Partitioned features write skipped: {_e}")
            except Exception as e:
                self.logger.error(f"Could not save Parquet features: {e}")

            # Also save labeled data files for compatibility
            labeled_files = [
                (f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl", train_data),
                (
                    f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl",
                    validation_data,
                ),
                (f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl", test_data),
            ]

            for file_path, data in labeled_files:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
            self.logger.info(f"✅ Saved labeled data files")

            # Parquet for labeled data too
            try:
                parquet_labeled = [
                    (
                        f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
                        mem_mgr.optimize_dataframe(train_data.copy()),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
                        mem_mgr.optimize_dataframe(validation_data.copy()),
                    ),
                    (
                        f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
                        mem_mgr.optimize_dataframe(test_data.copy()),
                    ),
                ]
                for file_path, df in parquet_labeled:
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )
                        import json as _json

                        metadata = {
                            "schema_version": "1",
                            "feature_list": _json.dumps(
                                [c for c in df.columns if c != "label"]
                            ),
                            "feature_hash": str(
                                hash(
                                    tuple(
                                        sorted([c for c in df.columns if c != "label"])
                                    )
                                )
                            )[:16],
                            "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                            "generator_commit": training_input.get(
                                "generator_commit", ""
                            ),
                        }
                        ParquetDatasetManager(self.logger).write_flat_parquet(
                            df,
                            file_path,
                            compression="snappy",
                        )
                    except Exception:
                        from src.utils.logger import log_io_operation

                        with log_io_operation(
                            self.logger, "to_parquet", file_path, compression="snappy"
                        ):
                            df.to_parquet(file_path, compression="snappy", index=False)
                    pass  # Reduced logging
                # Also write partitioned dataset for labeled data
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    base_dir = os.path.join(data_dir, "parquet", "labeled")
                    for split_name, df in (
                        ("train", mem_mgr.optimize_dataframe(train_data.copy())),
                        (
                            "validation",
                            mem_mgr.optimize_dataframe(validation_data.copy()),
                        ),
                        ("test", mem_mgr.optimize_dataframe(test_data.copy())),
                    ):
                        df = df.copy()
                        df["exchange"] = exchange
                        df["symbol"] = symbol
                        df["timeframe"] = training_input.get("timeframe", "1m")
                        df["split"] = split_name
                        import json as _json

                        pdm.write_partitioned_dataset(
                            df=df,
                            base_dir=base_dir,
                            partition_cols=[
                                "exchange",
                                "symbol",
                                "timeframe",
                                "split",
                                "year",
                                "month",
                                "day",
                            ],
                            schema_name="labeled",
                            compression="snappy",
                            metadata={
                                "schema_version": "1",
                                "feature_list": _json.dumps(
                                    [c for c in df.columns if c != "label"]
                                ),
                                "feature_hash": str(
                                    hash(
                                        tuple(
                                            sorted(
                                                [c for c in df.columns if c != "label"]
                                            )
                                        )
                                    )
                                )[:16],
                                "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                                "generator_commit": training_input.get(
                                    "generator_commit", ""
                                ),
                            },
                        )

                    # Materialize model-specific feature projection per split for faster downstream reads
                    try:
                        import json as _json

                        feat_cols = self.config.get(
                            "model_feature_columns"
                        ) or self.config.get("feature_columns")
                        label_col = self.config.get("label_column", "label")
                        if isinstance(feat_cols, list) and len(feat_cols) > 0:
                            cols = ["timestamp", *feat_cols, label_col]
                            model_name = self.config.get("model_name", "default")
                            out_dir = os.path.join(
                                "data_cache", "parquet", f"proj_features_{model_name}"
                            )
                            for split_name in ("train", "validation", "test"):
                                filters = [
                                    ("exchange", "==", exchange),
                                    ("symbol", "==", symbol),
                                    (
                                        "timeframe",
                                        "==",
                                        training_input.get("timeframe", "1m"),
                                    ),
                                    ("split", "==", split_name),
                                ]
                                meta = {
                                    "schema_version": "1",
                                    "feature_list": _json.dumps(feat_cols),
                                    "feature_hash": str(hash(tuple(sorted(feat_cols))))[
                                        :16
                                    ],
                                    "training_window": f"{training_input.get('t0_ms','')}..{training_input.get('t1_ms','')}",
                                    "generator_commit": training_input.get(
                                        "generator_commit", ""
                                    ),
                                    "split": split_name,
                                }
                                pdm.materialize_projection(
                                    base_dir=base_dir,
                                    filters=filters,
                                    columns=cols,
                                    output_dir=out_dir,
                                    partition_cols=[
                                        "exchange",
                                        "symbol",
                                        "timeframe",
                                        "split",
                                        "year",
                                        "month",
                                        "day",
                                    ],
                                    compression="snappy",
                                    metadata=meta,
                                )
                            self.logger.info(f"✅ Materialized model projections")
                    except Exception as _proj_err:
                        self.logger.warning(
                            f"Feature projection materialization skipped: {_proj_err}"
                        )
                except Exception as _e:
                    self.logger.warning(f"Partitioned labeled write skipped: {_e}")
            except Exception as e:
                self.logger.error(f"Could not save Parquet labeled data: {e}")

            # Integrate UnifiedDataManager to create time-based train/validation/test splits
            try:
                from src.training.data_manager import UnifiedDataManager

                lookback_days = training_input.get(
                    "lookback_days",
                    self.config.get("lookback_days", 180),
                )

                labeled_full = labeled_data.copy()
                # Ensure datetime index for time-based splits
                if "timestamp" in labeled_full.columns:
                    labeled_full["timestamp"] = pd.to_datetime(
                        labeled_full["timestamp"],
                        errors="coerce",
                    )
                    labeled_full = labeled_full.dropna(
                        subset=["timestamp"],
                    )  # drop rows with invalid timestamps
                    labeled_full = labeled_full.set_index("timestamp").sort_index()
                elif not pd.api.types.is_datetime64_any_dtype(labeled_full.index):
                    # Fallback: create a synthetic datetime index to preserve ordering
                    self.logger.warning(
                        "No timestamp column found; creating synthetic datetime index for splits",
                    )
                    labeled_full = labeled_full.copy()
                    labeled_full.index = pd.date_range(
                        end=pd.Timestamp.utcnow(),
                        periods=len(labeled_full),
                        freq="T",
                    )

                data_manager = UnifiedDataManager(
                    data_dir=data_dir,
                    symbol=symbol,
                    exchange=exchange,
                    lookback_days=lookback_days,
                )
                db_result = data_manager.create_unified_database(labeled_full)
                pipeline_state["unified_database"] = db_result
                self.logger.info("✅ Created unified database splits")
            except Exception as e:
                self.logger.exception(
                    f"❌ UnifiedDataManager failed to create splits: {e}",
                )

            # Update pipeline state with results
            pipeline_state.update(
                {
                    "labeled_data": labeled_data,
                    "feature_engineering_metadata": {},
                    "feature_engineering_completed": True,
                    "labeling_completed": True,
                },
            )

            self.logger.info(
                "✅ Analyst labeling and feature engineering completed successfully",
            )
            self.logger.info("Training specialist models for regime: combined (single unified feature set)")
            return {"status": "SUCCESS", "data": {"data": labeled_data}}

        except Exception as e:
            self.logger.exception(
                f"❌ Error in analyst labeling and feature engineering: {e}",
            )
            return {"status": "FAILED", "error": str(e)}

    def _create_fallback_labeled_data(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Create fallback labeled data when orchestrator fails."""
        try:
            # Create simple labeled data with basic features
            labeled_data = price_data.copy()

            # Add basic features
            if "close" in labeled_data.columns:
                labeled_data["returns"] = labeled_data["close"].pct_change()
                labeled_data["volatility"] = (
                    labeled_data["returns"].rolling(window=20).std()
                )
                labeled_data["sma_20"] = labeled_data["close"].rolling(window=20).mean()
                labeled_data["sma_50"] = labeled_data["close"].rolling(window=50).mean()

            # Add simple labels (binary classification: -1 for sell, 1 for buy)
            labeled_data["label"] = -1  # Default to sell signal

            # Create simple buy/sell signals based on moving averages
            if "sma_20" in labeled_data.columns and "sma_50" in labeled_data.columns:
                # Use -1 for sell signal, 1 for buy signal (binary classification)
                labeled_data.loc[
                    labeled_data["sma_20"] > labeled_data["sma_50"],
                    "label",
                ] = 1  # Buy signal
                # Keep -1 for when sma_20 <= sma_50 (sell signal)

            # Remove raw OHLCV columns to prevent data leakage
            raw_ohlcv_columns = [c for c in self._RAW_CONTEXT_COLUMNS if c in ["open","high","low","close","volume","avg_price","min_price","max_price"]]
            columns_to_remove = [col for col in raw_ohlcv_columns if col in labeled_data.columns]
            if columns_to_remove:
                labeled_data = labeled_data.drop(columns=columns_to_remove)
                self.logger.info(f"🗑️ Removed raw OHLCV columns to prevent data leakage: {columns_to_remove}")

            # Remove NaN values
            labeled_data = labeled_data.dropna()

            return {
                "data": labeled_data,
                "metadata": {
                    "labeling_method": "fallback_simple",
                    "features_added": ["returns", "volatility", "sma_20", "sma_50"],
                    "raw_ohlcv_removed": columns_to_remove,
                    "label_distribution": labeled_data["label"]
                    .value_counts()
                    .to_dict(),
                },
            }

        except Exception as e:
            self.logger.error(f"Error creating fallback labeled data: {e}")
            # Return original data with basic label (binary classification)
            price_data_copy = price_data.copy()
            price_data_copy["label"] = -1  # Default to sell signal
            
            # Remove raw OHLCV columns even in error case
            raw_ohlcv_columns = [c for c in self._RAW_CONTEXT_COLUMNS if c in ["open","high","low","close","volume","avg_price","min_price","max_price"]]
            columns_to_remove = [col for col in raw_ohlcv_columns if col in price_data_copy.columns]
            if columns_to_remove:
                price_data_copy = price_data_copy.drop(columns=columns_to_remove)
                self.logger.info(f"🗑️ Removed raw OHLCV columns in error case: {columns_to_remove}")
            
            return {
                "data": price_data_copy,
                "metadata": {"labeling_method": "fallback_basic", "error": str(e), "raw_ohlcv_removed": columns_to_remove},
            }


class DeprecatedAnalystLabelingFeatureEngineeringStep:
    """Step 4: Analyst Labeling and Feature Engineering (DEPRECATED)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild(
            "DeprecatedAnalystLabelingFeatureEngineeringStep",
        )

    async def initialize(self) -> None:
        """Initialize the analyst labeling and feature engineering step."""
        self.logger.info(
            "Initializing Deprecated Analyst Labeling and Feature Engineering Step...",
        )
        self.logger.info(
            "Deprecated Analyst Labeling and Feature Engineering Step initialized successfully",
        )

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst labeling and feature engineering step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            dict: Updated pipeline state
        """
        self.logger.info(
            "Executing deprecated analyst labeling and feature engineering step...",
        )

        # This step is deprecated - return current state
        return pipeline_state

    async def _apply_triple_barrier_labeling(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> pd.DataFrame:
        """
        Apply Triple Barrier Method for labeling.

        Args:
            data: Market data for the regime
            regime_name: Name of the regime

        Returns:
            DataFrame with labels added
        """
        self.logger.info(
            f"Applying Triple Barrier Method for regime: {regime_name}",
        )

        # Ensure we have required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                msg = f"Missing required column: {col}"
                raise ValueError(msg)

        # Calculate features for labeling
        data = self._calculate_features(data)

        # Apply Triple Barrier Method
        labeled_data = data.copy()

        # Define barrier parameters based on regime
        profit_take_multiplier = 0.002  # 0.2%
        stop_loss_multiplier = 0.001  # 0.1%
        # Use environment-aware default consistent with vectorized orchestrator
        try:
            if os.environ.get("BLANK_TRAINING_MODE", "0") == "1":
                time_barrier_minutes = 90
            elif os.environ.get("FULL_TRAINING_MODE", "0") == "1":
                time_barrier_minutes = 360
            else:
                time_barrier_minutes = 30
        except Exception:
            time_barrier_minutes = 30

        # Apply triple barrier labeling
        labels = []
        for i in range(len(data)):
            if i >= len(data) - 1:  # Skip last point
                labels.append(0)
                continue

            entry_price = data.iloc[i]["close"]
            entry_time = data.index[i]

            # Calculate barriers
            profit_take_barrier = entry_price * (1 + profit_take_multiplier)
            stop_loss_barrier = entry_price * (1 - stop_loss_multiplier)
            time_barrier = entry_time + timedelta(minutes=time_barrier_minutes)

            # Check if any barrier is hit
            label = -1  # Default to sell signal for binary classification

            for j in range(
                i + 1,
                min(i + 100, len(data)),
            ):  # Look ahead up to 100 points
                current_time = data.index[j]
                current_price = data.iloc[j]["high"]  # Use high for profit take
                current_low = data.iloc[j]["low"]  # Use low for stop loss

                # Check time barrier
                if current_time > time_barrier:
                    label = -1  # Time barrier hit - default to sell signal
                    break

                # Check profit take barrier
                if current_price >= profit_take_barrier:
                    label = 1  # Profit take hit - positive
                    break

                # Check stop loss barrier
                if current_low <= stop_loss_barrier:
                    label = -1  # Stop loss hit - negative
                    break

            labels.append(label)

        labeled_data["label"] = labels

        # Calculate label distribution
        label_counts = pd.Series(labels).value_counts()
        self.logger.info(
            f"Label distribution for {regime_name}: {dict(label_counts)}",
        )

        return labeled_data

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for the data.

        Args:
            data: Market data

        Returns:
            DataFrame with features added
        """
        try:
            # Calculate RSI
            data["rsi"] = self._calculate_rsi(data["close"])

            # Calculate MACD
            macd, signal = self._calculate_macd(data["close"])
            data["macd"] = macd
            data["macd_signal"] = signal
            data["macd_histogram"] = macd - signal

            # Calculate Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data["close"])
            data["bb_upper"] = bb_upper
            data["bb_lower"] = bb_lower
            data["bb_width"] = bb_upper - bb_lower
            data["bb_position"] = (data["close"] - bb_lower) / (bb_upper - bb_lower)

            # Calculate ATR
            data["atr"] = self._calculate_atr(data)

            # Calculate price-based features
            data["price_change"] = data["close"].pct_change()
            data["price_change_abs"] = data["price_change"].abs()
            data["high_low_ratio"] = data["high"] / data["low"]
            data["volume_price_ratio"] = data["volume"] / data["close"]

            # Calculate moving averages
            data["sma_5"] = data["close"].rolling(window=5).mean()
            data["sma_10"] = data["close"].rolling(window=10).mean()
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["ema_12"] = data["close"].ewm(span=12).mean()
            data["ema_26"] = data["close"].ewm(span=26).mean()

            # Calculate momentum features
            data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            data["momentum_10"] = data["close"] / data["close"].shift(10) - 1
            data["momentum_20"] = data["close"] / data["close"].shift(20) - 1

            # Add candlestick pattern features using advanced feature engineering
            # Legacy S/R/Candle code removed - using simplified approach
            data = data  # Keep original data for now

            # Fill NaN values
            return data.fillna(method="bfill").fillna(0)

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    async def _add_candlestick_pattern_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add candlestick pattern features using advanced feature engineering.

        Args:
            data: Market data with OHLCV

        Returns:
            DataFrame with candlestick pattern features added
        """
        try:
            self.logger.info("Adding candlestick pattern features...")

            # Import advanced feature engineering
            from src.analyst.advanced_feature_engineering import (
                AdvancedFeatureEngineering,
            )

            # Initialize advanced feature engineering
            config = {
                "advanced_features": {
                    # Legacy S/R/Candle code removed,
                    "enable_volatility_regime_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                },
                # Legacy S/R/Candle code removed
            }

            # Initialize advanced feature engineering
            advanced_fe = AdvancedFeatureEngineering(config)
            await advanced_fe.initialize()

            # Prepare data for feature engineering
            price_data = data[["open", "high", "low", "close"]].copy()
            volume_data = data[["volume"]].copy()

            # Get advanced features including candlestick patterns
            advanced_features = await advanced_fe.engineer_features(
                price_data=price_data,
                volume_data=volume_data,
                order_flow_data=None,
            )

            # Convert features to DataFrame and align with original data
            if advanced_features:
                # Align per-row features to data length; skip scalars
                aligned = pd.DataFrame(index=data.index)
                n = len(data)
                for name, val in advanced_features.items():
                    arr = None
                    if isinstance(val, pd.Series):
                        arr = val.values
                    elif isinstance(val, np.ndarray):
                        if val.ndim == 1:
                            arr = val
                        elif val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1):
                            arr = val.reshape(-1)
                    elif isinstance(val, list):
                        tmp = np.asarray(val)
                        if tmp.ndim == 1:
                            arr = tmp
                        elif tmp.ndim == 2 and (tmp.shape[0] == 1 or tmp.shape[1] == 1):
                            arr = tmp.reshape(-1)
                    if arr is None:
                        continue
                    if len(arr) > n:
                        arr = arr[-n:]
                    elif len(arr) < n:
                        pad = n - len(arr)
                        arr = np.concatenate([np.full(pad, np.nan), arr])
                    try:
                        aligned[name] = pd.to_numeric(arr, errors="coerce")
                    except Exception:
                        pass

                # Drop fully-NaN and constant columns
                if not aligned.empty:
                    aligned = aligned.dropna(axis=1, how="all")
                    nunique = aligned.nunique(dropna=True)
                    const_cols = nunique[nunique <= 1].index.tolist()
                    if const_cols:
                        self.logger.warning(f"Dropping {len(const_cols)} constant candlestick features")
                        aligned = aligned.drop(columns=const_cols)

                # Add aligned features to original data
                for col in aligned.columns:
                    if col not in data.columns:
                        data[col] = aligned[col]

                self.logger.info(
                    f"Added {len(aligned.columns)} candlestick pattern features",
                )
            else:
                self.logger.error("No candlestick pattern features generated")

            return data

        except Exception as e:
            self.logger.error(f"Error adding candlestick pattern features: {e}")
            return data

    async def _perform_feature_engineering(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> pd.DataFrame:
        """
        Perform feature engineering for the regime.

        Args:
            data: Labeled data with features
            regime_name: Name of the regime

        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info(
                f"Performing feature engineering for regime: {regime_name}",
            )

            # Select feature columns including candlestick patterns
            feature_columns = [
                "rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_lower",
                "bb_width",
                "bb_position",
                "atr",
                "price_change",
                "price_change_abs",
                "high_low_ratio",
                "volume_price_ratio",
                "sma_5",
                "sma_10",
                "sma_20",
                "ema_12",
                "ema_26",
                "momentum_5",
                "momentum_10",
                "momentum_20",
            ]

            # Add candlestick pattern features
            # Legacy S/R/Candle code removed

            # Combine all feature columns
            # Legacy S/R/Candle code removed

            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]

            if not available_features:
                msg = f"No features available for regime {regime_name}"
                raise ValueError(msg)

            # Create feature matrix
            feature_data = data[available_features].copy()

            # Remove rows with NaN values
            feature_data = feature_data.dropna()

            # Standardize features
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            feature_data_scaled = pd.DataFrame(
                feature_data_scaled,
                columns=available_features,
                index=feature_data.index,
            )

            # Add label column
            feature_data_scaled["label"] = data.loc[feature_data.index, "label"]

            self.logger.info(
                f"Feature engineering completed for {regime_name}: {len(feature_data_scaled)} samples, {len(available_features)} features",
            )

            return feature_data_scaled

        except Exception as e:
            self.logger.exception(
                f"Error in feature engineering for {regime_name}: {e}",
            )
            raise

    async def _train_regime_encoders(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> dict[str, Any]:
        """
        Train regime-specific encoders.

        Args:
            data: Feature-engineered data
            regime_name: Name of the regime

        Returns:
            Dict containing trained encoders
        """
        try:
            self.logger.info(f"Training encoders for regime: {regime_name}")

            # Separate features and labels
            feature_columns = [col for col in data.columns if col != "label"]
            X = data[feature_columns]
            y = data["label"]

            # Train PCA encoder for dimensionality reduction
            pca = PCA(n_components=min(10, len(feature_columns)))
            pca.fit_transform(X)

            # Train autoencoder for feature learning
            from sklearn.neural_network import MLPRegressor

            # Ensure we have enough data and features for autoencoder
            if len(X) < 10 or len(feature_columns) < 2:
                self.logger.warning(
                    f"Insufficient data for autoencoder training: {len(X)} samples, {len(feature_columns)} features"
                )
                # Create a simple identity encoder as fallback
                autoencoder = None
            else:
                try:
                    # Use a simpler architecture for better convergence
                    hidden_size = max(2, min(len(feature_columns) // 2, 10))
                    autoencoder = MLPRegressor(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=500,  # Reduced for faster training
                        random_state=42,
                        alpha=0.01,  # L2 regularization
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                    )
                    autoencoder.fit(X, X)
                    self.logger.info(
                        f"Autoencoder trained successfully with {hidden_size} hidden units"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Autoencoder training failed: {e}, using fallback"
                    )
                    autoencoder = None

            # Store encoders
            encoders = {
                "pca": pca,
                "autoencoder": autoencoder,
                "feature_columns": feature_columns,
                "n_features": len(feature_columns),
                "n_samples": len(X),
            }

            # Log encoder information
            pca_info = f"PCA components={pca.n_components_}"
            autoencoder_info = f"Autoencoder layers={autoencoder.n_layers_ if autoencoder else 'None (fallback)'}"
            self.logger.info(
                f"Encoders trained for {regime_name}: {pca_info}, {autoencoder_info}",
            )

            return encoders

        except Exception as e:
            self.logger.error(f"Error training encoders for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    exchange: str = "BINANCE",
    force_rerun: bool = False,
) -> bool:
    """
    Run analyst labeling and feature engineering step using vectorized orchestrator.

    Args:
        symbol: Trading symbol
        exchange_name: Exchange name (deprecated, use exchange)
        data_dir: Data directory path
        timeframe: Timeframe for data
        exchange: Exchange name

    Returns:
        bool: True if successful, False otherwise
    """
    print(
        "🚀 Running analyst labeling and feature engineering step with vectorized orchestrator..."
    )

    # Use exchange parameter if provided, otherwise use exchange_name for backward compatibility
    actual_exchange = exchange if exchange != "BINANCE" else exchange_name

    try:
        # Create step instance
        config = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
        }
        step = AnalystLabelingFeatureEngineeringStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": actual_exchange,
            "data_dir": data_dir,
            "timeframe": timeframe,
            "force_rerun": force_rerun,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS" if isinstance(result, dict) else True

    except Exception as e:
        print(f"Analyst labeling and feature engineering failed: {e}")
        return False


# For backward compatibility with existing step structure
async def deprecated_run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
) -> bool:
    """
    DEPRECATED: Run analyst labeling and feature engineering step.

    This function is deprecated and should not be used in new training pipelines.
    """
    print(
        "⚠️  WARNING: This step is deprecated and should not be used in new training pipelines.",
    )
    return True


if __name__ == "__main__":
    # Test the step
    async def test():
        """Test the analyst labeling and feature engineering step."""
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step test result: {result}")

    asyncio.run(test())
