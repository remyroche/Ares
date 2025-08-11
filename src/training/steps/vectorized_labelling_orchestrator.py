# src/training/steps/vectorized_labelling_orchestrator.py

"""
Vectorized Labelling Orchestrator for comprehensive feature engineering and labeling pipeline.
Coordinates optimized_triple_barrier_labeling.py, vectorized_advanced_feature_engineering.py 
and autoencoder_feature_generator.py with advanced preprocessing and feature selection.
"""

import numpy as np
import logging
from datetime import datetime
import os
import pandas as pd
import pywt
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class VectorizedLabellingOrchestrator:
    """
    Comprehensive vectorized labeling orchestrator that coordinates all feature generation 
    and labeling components with advanced preprocessing and feature selection.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLabellingOrchestrator")
        self.feature_error_logger = logging.getLogger("Ares.FeatureError")
        if not self.feature_error_logger.handlers:
            try:
                fh = logging.FileHandler("log/feature_errors.log")
                fh.setLevel(logging.INFO)
                fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                fh.setFormatter(fmt)
                self.feature_error_logger.addHandler(fh)
                self.feature_error_logger.propagate = False
            except Exception:
                pass

        # Configuration
        self.orchestrator_config = config.get("vectorized_labelling_orchestrator", {})
        self.enable_stationary_checks = self.orchestrator_config.get("enable_stationary_checks", True)
        self.enable_data_normalization = self.orchestrator_config.get("enable_data_normalization", True)
        self.enable_lookahead_bias_handling = self.orchestrator_config.get("enable_lookahead_bias_handling", True)
        self.enable_feature_selection = self.orchestrator_config.get("enable_feature_selection", True)
        self.enable_memory_efficient_types = self.orchestrator_config.get("enable_memory_efficient_types", True)
        self.enable_parquet_saving = self.orchestrator_config.get("enable_parquet_saving", True)
        # Strict feature shapes mode: treat scalar features as errors
        self.strict_feature_shapes = bool(
            self.orchestrator_config.get("strict_feature_shapes", False)
            or os.getenv("CI") == "1"
        )

        # Feature selection configuration
        self.feature_selection_config = self.orchestrator_config.get("feature_selection", {})
        self.vif_threshold = self.feature_selection_config.get("vif_threshold", 5.0)
        self.mutual_info_threshold = self.feature_selection_config.get("mutual_info_threshold", 0.01)
        self.lightgbm_importance_threshold = self.feature_selection_config.get("lightgbm_importance_threshold", 0.01)

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Initialize components
        self.triple_barrier_labeler = None
        self.advanced_feature_engineer = None
        self.autoencoder_generator = None
        self.stationarity_checker = None
        self.feature_selector = None
        self.data_normalizer = None

        self.is_initialized = False

        # Debug snapshots for logging
        self._debug_raw_ohlcv: pd.DataFrame | None = None
        self._debug_price_returns: pd.DataFrame | None = None

    def _log_feature_sample(self, stage: str, df: pd.DataFrame, step_no: str) -> None:
        try:
            import os
            os.makedirs("log/features_samples", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_stage = stage.replace(" ", "_")
            fname = f"log/features_samples/{ts}_{step_no}_{safe_stage}.log"
            # Merge raw/returns for visibility when available
            sample = df.copy()
            if self._debug_raw_ohlcv is not None:
                for c in ["open","high","low","close","volume"]:
                    if c in self._debug_raw_ohlcv.columns and c not in sample.columns:
                        sample[c] = self._debug_raw_ohlcv[c]
            if self._debug_price_returns is not None:
                for c in self._debug_price_returns.columns:
                    if c not in sample.columns:
                        sample[c] = self._debug_price_returns[c]
            # Reorder: raw, returns, then features
            cols_raw = [c for c in ["open","high","low","close","volume"] if c in sample.columns]
            cols_ret = [c for c in sample.columns if c.endswith("_returns")]
            other = [c for c in sample.columns if c not in cols_raw + cols_ret]
            sample = sample[cols_raw + cols_ret + other]
            with open(fname, "w") as f:
                f.write(f"Stage: {stage} | Step: {step_no} | Shape: {sample.shape}\n")
                f.write(sample.head(50).to_string())
            self.logger.info(f"Feature sample written: {fname}")
        except Exception as e:
            self.logger.warning(f"Failed to write feature sample for {stage}: {e}")

    def _log_feature_errors(self, stage: str, df: pd.DataFrame) -> None:
        try:
            numeric = df.select_dtypes(include=[np.number])
            nan_counts = numeric.isna().sum()
            inf_counts = np.isinf(numeric).sum()
            any_nan = int(nan_counts.sum())
            any_inf = int(inf_counts.sum())
            if any_nan or any_inf:
                self.feature_error_logger.info(
                    f"Stage={stage} | NaN_total={any_nan} | Inf_total={any_inf} | NaN_cols={nan_counts[nan_counts>0].to_dict()} | Inf_cols={inf_counts[inf_counts>0].to_dict()}"
                )
        except Exception:
            pass

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="vectorized labelling orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize vectorized labeling orchestrator components."""
        try:
            self.logger.info("🚀 Initializing vectorized labeling orchestrator...")

            # Initialize triple barrier labeler using the proper implementation
            from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=self.orchestrator_config.get("profit_take_multiplier", 0.002),
                stop_loss_multiplier=self.orchestrator_config.get("stop_loss_multiplier", 0.001),
                time_barrier_minutes=self.orchestrator_config.get("time_barrier_minutes", 30),
                max_lookahead=self.orchestrator_config.get("max_lookahead", 100),
            )

            # Initialize advanced feature engineer with error handling
            try:
                from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
                self.advanced_feature_engineer = VectorizedAdvancedFeatureEngineering(self.config)
                await self.advanced_feature_engineer.initialize()
            except Exception as e:
                self.logger.warning(f"Failed to initialize advanced feature engineer: {e}")
                self.advanced_feature_engineer = None

            # Initialize autoencoder generator with error handling
            try:
                from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
                self.autoencoder_generator = AutoencoderFeatureGenerator(self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize autoencoder generator: {e}")
                self.autoencoder_generator = None

            # Initialize stationarity checker
            self.stationarity_checker = VectorizedStationarityChecker(self.config)

            # Initialize feature selector
            self.feature_selector = VectorizedFeatureSelector(self.config)

            # Initialize data normalizer
            self.data_normalizer = VectorizedDataNormalizer(self.config)

            self.is_initialized = True
            self.logger.info("✅ Vectorized labeling orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized labeling orchestrator: {e}",
            )
            # Set basic components even if advanced ones fail
            from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling()
            self.stationarity_checker = VectorizedStationarityChecker(self.config)
            self.feature_selector = VectorizedFeatureSelector(self.config)
            self.data_normalizer = VectorizedDataNormalizer(self.config)
            self.is_initialized = True
            self.logger.warning("⚠️ Vectorized labeling orchestrator initialized with fallback components")
            return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="vectorized labeling orchestration",
    )
    async def orchestrate_labeling_and_feature_engineering(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        sr_levels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Orchestrate the complete labeling and feature engineering pipeline.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            if not self.is_initialized:
                self.logger.error("Vectorized labeling orchestrator not initialized")
                return {}

            self.logger.info("🎯 Starting comprehensive vectorized labeling and feature engineering orchestration...")

            # Systematic leakage guard: capture baseline columns present before feature generation
            baseline_cols: set[str] = set(price_data.columns)
            try:
                if volume_data is not None:
                    baseline_cols |= set(getattr(volume_data, 'columns', []))
                if order_flow_data is not None:
                    baseline_cols |= set(getattr(order_flow_data, 'columns', []))
            except Exception:
                pass

            # 1. Stationary checks
            if self.enable_stationary_checks:
                self.logger.info("📊 Performing stationary checks...")
                stationary_data = await self.stationarity_checker.check_and_transform_stationarity(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                price_data = stationary_data.get("price_data", price_data)
                volume_data = stationary_data.get("volume_data", volume_data)
                order_flow_data = stationary_data.get("order_flow_data", order_flow_data)
                # Capture raw and returns for logging
                try:
                    self._debug_raw_ohlcv = price_data[["open","high","low","close","volume"]].copy()
                except Exception:
                    self._debug_raw_ohlcv = None
                try:
                    ret_cols = [c for c in price_data.columns if c.endswith("_returns")]
                    self._debug_price_returns = price_data[ret_cols].copy() if ret_cols else None
                except Exception:
                    self._debug_price_returns = None
                self._log_feature_sample("Stationarity", price_data, "04_01")
                self._log_feature_errors("Stationarity", price_data)

            # 2. Advanced feature engineering
            self.logger.info("🔧 Generating advanced features...")
            if self.advanced_feature_engineer is not None:
                advanced_features = await self.advanced_feature_engineer.engineer_features(
                    price_data,
                    volume_data,
                    order_flow_data,
                    sr_levels,
                )
            else:
                self.logger.warning("Advanced feature engineer not available, using basic features")
                advanced_features = {}
            # Preview engineered features as DataFrame for logging
            try:
                preview_df = pd.DataFrame(index=price_data.index)
                for k,v in list(advanced_features.items())[:200]:
                    if isinstance(v, pd.Series):
                        preview_df[k] = v
                    elif isinstance(v, np.ndarray) and v.ndim==1 and len(v)==len(preview_df.index):
                        preview_df[k] = v
                self._log_feature_sample("EngineerFeatures", preview_df, "04_02")
                self._log_feature_errors("EngineerFeatures", preview_df)
                # Categorize and log feature creation summary
                try:
                    cats = self._categorize_features(advanced_features)
                    cats_path = os.path.join("log", "features_samples")
                    os.makedirs(cats_path, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(os.path.join(cats_path, f"{ts}_04_02_EngineerFeatures_Categories.txt"), "w") as f:
                        for k, v in cats.items():
                            f.write(f"{k}: {v}\n")
                    self.logger.info(f"Feature categories: {cats}")
                except Exception:
                    pass
                # Format and alignment diagnostics
                try:
                    formatted_preview, fmt_report = self._format_and_align_features(advanced_features, preview_df.index)
                    self._log_feature_format_report("EngineerFeatures", fmt_report, "04_02")
                except Exception:
                    pass
            except Exception:
                pass

            # 3. Triple barrier labeling
            self.logger.info("🏷️ Applying triple barrier labeling...")
            labeled_data = self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(
                price_data.copy()
            )

            # 4. Combine features and labels
            self.logger.info("🔗 Combining features and labels...")
            combined_data = self._combine_features_and_labels_vectorized(
                labeled_data,
                advanced_features,
            )
            self._log_feature_sample("Combine", combined_data, "04_03")
            self._log_feature_errors("Combine", combined_data)
            try:
                self._log_dataframe_columns("Combine", combined_data, "04_03")
            except Exception:
                pass

            # Drop stationarity helper columns prior to selection/normalization
            combined_data = self._remove_stationarity_transform_columns(combined_data)

            # 5. Feature selection
            if self.enable_feature_selection:
                self.logger.info("🎯 Performing feature selection...")
                selected = await self.feature_selector.select_optimal_features(
                    combined_data,
                    labeled_data["label"] if "label" in labeled_data.columns else None,
                )
                if isinstance(selected, pd.DataFrame) and selected.shape[1] > 0:
                    combined_data = selected
                else:
                    self.logger.warning("Feature selection returned 0 columns; keeping pre-selection data.")
                
                # Remove raw OHLCV columns after feature selection
                combined_data = self._remove_raw_ohlcv_columns(combined_data)
                self._log_feature_sample("FeatureSelection", combined_data, "04_04")
                self._log_feature_errors("FeatureSelection", combined_data)
                try:
                    self._log_dataframe_columns("FeatureSelection", combined_data, "04_04")
                except Exception:
                    pass

            # 6. Data normalization
            if self.enable_data_normalization:
                self.logger.info("📏 Normalizing data...")
                normalized_data = await self.data_normalizer.normalize_data(
                    combined_data,
                )
                combined_data = normalized_data
                
                # Remove raw OHLCV columns after normalization
                combined_data = self._remove_raw_ohlcv_columns(combined_data)
                combined_data = self._remove_stationarity_transform_columns(combined_data)
                self._log_feature_sample("Normalization", combined_data, "04_05")
                self._log_feature_errors("Normalization", combined_data)
                try:
                    self._log_dataframe_columns("Normalization", combined_data, "04_05")
                except Exception:
                    pass

            # 7. Autoencoder feature generation
            self.logger.info("🤖 Generating autoencoder features...")
            if self.autoencoder_generator is not None:
                # Remove label column before passing to autoencoder to prevent data leakage
                autoencoder_input_data = combined_data.copy()
                if "label" in autoencoder_input_data.columns:
                    autoencoder_input_data = autoencoder_input_data.drop(columns=["label"])
                    self.logger.info("🗑️ Removed 'label' column from autoencoder input to prevent data leakage")
                
                autoencoder_features = self.autoencoder_generator.generate_features(
                    autoencoder_input_data,
                    "vectorized_regime",
                    labeled_data["label"].values if "label" in labeled_data.columns else np.zeros(len(combined_data)),
                )
            else:
                self.logger.warning("Autoencoder generator not available, skipping autoencoder feature generation")
                autoencoder_features = combined_data
            try:
                ae_df = autoencoder_features if isinstance(autoencoder_features, pd.DataFrame) else combined_data
                self._log_feature_sample("Autoencoder", ae_df, "04_06")
                self._log_feature_errors("Autoencoder", ae_df)
                try:
                    self._log_dataframe_columns("Autoencoder", ae_df, "04_06")
                except Exception:
                    pass
            except Exception:
                pass

            # 8. Final data preparation
            self.logger.info("🎨 Preparing final data...")
            final_data = self._prepare_final_data_vectorized(
                autoencoder_features,
                labeled_data,
            )
            
            # Remove raw OHLCV columns to prevent data leakage
            final_data = self._remove_raw_ohlcv_columns(final_data)
            final_data = self._remove_stationarity_transform_columns(final_data)
            # Also ensure datetime/timestamp columns are removed before returning
            final_data = self._remove_datetime_columns(final_data)
            self._log_feature_sample("FinalData", final_data, "04_07")
            self._log_feature_errors("FinalData", final_data)
            try:
                self._log_dataframe_columns("FinalData", final_data, "04_07")
            except Exception:
                pass

            # Systematic leakage guard: drop any baseline columns observed before feature generation
            try:
                drop_baseline = [c for c in final_data.columns if c in baseline_cols and c != 'label']
                if drop_baseline:
                    self.logger.warning(f"🚨 Removing baseline/raw columns carried into features: {drop_baseline[:20]}" + (" ..." if len(drop_baseline) > 20 else ""))
                    final_data = final_data.drop(columns=drop_baseline)
            except Exception:
                pass

            # 9. Memory optimization
            if self.enable_memory_efficient_types:
                self.logger.info("💾 Optimizing memory usage...")
                final_data = self._optimize_memory_usage_vectorized(final_data)

            # 10. Save data
            if self.enable_parquet_saving:
                self.logger.info("💾 Saving data as Parquet...")
                self._save_data_as_parquet(final_data)

            self.logger.info(
                f"🎉 Vectorized labeling and feature engineering completed! Final shape: {final_data.shape}",
            )

            return {
                "data": final_data,
                "metadata": {
                    "total_features": len(final_data.columns),
                    "total_samples": len(final_data),
                    "feature_engineering_completed": self.advanced_feature_engineer is not None,
                    "labeling_completed": True,
                    "autoencoder_features_generated": self.autoencoder_generator is not None,
                    "stationary_checks_performed": self.enable_stationary_checks,
                    "data_normalized": self.enable_data_normalization,
                    "feature_selection_performed": self.enable_feature_selection,
                    "memory_optimized": self.enable_memory_efficient_types,
                    "saved_as_parquet": self.enable_parquet_saving,
                }
            }

        except Exception as e:
            try:
                self.logger.error(f"Error in vectorized labeling orchestration: {e}")
            finally:
                # Ensure we always return a consistent structure
                return {}

    def _combine_features_and_labels_vectorized(
        self,
        labeled_data: pd.DataFrame,
        advanced_features: dict[str, Any],
    ) -> pd.DataFrame:
        """Combine features and labels using vectorized operations."""
        try:
            # Ensure OHLCV data is present first
            labeled_data = self._ensure_ohlcv_data(labeled_data)
            
            # Remove metadata columns first
            labeled_data = self._remove_metadata_columns(labeled_data)
            
            # Remove datetime columns from labeled_data to prevent dtype conflicts
            labeled_data = self._remove_datetime_columns(labeled_data)
            
            # If no advanced features, return labeled data as is
            if not advanced_features:
                return labeled_data
            
            # Build a features DataFrame aligned to labeled_data index
            target_index = labeled_data.index
            num_rows = len(target_index)
            features_df = pd.DataFrame(index=target_index)

            def _as_1d_array(value: Any) -> Optional[np.ndarray]:
                """Normalize a feature value to a 1D numpy array if possible, else None."""
                if isinstance(value, pd.Series):
                    return value.values.reshape(-1)
                if isinstance(value, np.ndarray):
                    # Handle multi-dimensional arrays more flexibly
                    if value.ndim == 1:
                        return value
                    elif value.ndim == 2:
                        # For 2D arrays, try to flatten them
                        if value.shape[0] == 1 or value.shape[1] == 1:
                            return value.reshape(-1)
                        # If it's a 2D array with multiple columns, take the first column
                        # This handles cases where features might be stored as 2D arrays
                        return value[:, 0] if value.shape[1] > 0 else None
                    elif value.ndim > 2:
                        # For higher dimensional arrays, try to flatten them
                        # This handles wavelet features and other multi-dimensional features
                        try:
                            return value.reshape(-1)
                        except Exception:
                            # If flattening fails, try to take the first element along extra dimensions
                            return value.reshape(value.shape[0], -1)[:, 0] if value.shape[0] > 0 else None
                    return None
                if isinstance(value, list):
                    try:
                        arr = np.asarray(value)
                        if arr.ndim == 1:
                            return arr
                        elif arr.ndim == 2:
                            if arr.shape[0] == 1 or arr.shape[1] == 1:
                                return arr.reshape(-1)
                            return arr[:, 0] if arr.shape[1] > 0 else None
                        elif arr.ndim > 2:
                            return arr.reshape(-1)
                        return None
                    except Exception:
                        return None
                # Enhanced scalar handling: broadcast numeric scalars across all rows
                if isinstance(value, (int, float)):
                    try:
                        if np.isnan(value) or np.isinf(value):
                            return None
                    except Exception:
                        pass
                    # Previously we broadcasted numeric scalars which created constant features.
                    # Skip numeric scalars entirely to avoid constant/leaky columns.
                    return None
                # Skip non-numeric scalars (strings, bools) to avoid creating categorical leakage
                if isinstance(value, (str, bool)):
                    return None
                # For other types, try to convert to array
                try:
                    if hasattr(value, '__len__') and len(value) > 1:
                        # If it has length > 1, it might be a feature array
                        arr = np.asarray(value)
                        if arr.ndim == 1:
                            return arr
                        elif arr.ndim == 2:
                            return arr[:, 0] if arr.shape[1] > 0 else None
                        elif arr.ndim > 2:
                            return arr.reshape(-1)
                    return None
                except Exception:
                    return None

            added_columns: list[str] = []
            skipped_scalars: list[str] = []
            scalar_offenders: list[str] = []
            trimmed_aligned: list[str] = []
            padded_aligned: list[str] = []

            for feature_name, feature_value in advanced_features.items():
                arr = _as_1d_array(feature_value)
                if arr is None:
                    # Skip scalar/non-array features to avoid constant columns
                    skipped_scalars.append(feature_name)
                    scalar_offenders.append(feature_name)
                    # Add debugging to understand what's being skipped
                    if len(skipped_scalars) <= 5:  # Only log first 5 to avoid spam
                        self.logger.debug(f"Skipping feature '{feature_name}': type={type(feature_value)}, "
                                        f"shape={getattr(feature_value, 'shape', 'N/A') if hasattr(feature_value, 'shape') else 'N/A'}")
                    continue

                # Align array length to labeled_data length
                if len(arr) > num_rows:
                    # Keep the most recent values (align to the end)
                    arr = arr[-num_rows:]
                    trimmed_aligned.append(feature_name)
                elif len(arr) < num_rows:
                    # Left-pad with NaN so the tail aligns to the label index
                    pad_size = num_rows - len(arr)
                    arr = np.concatenate([np.full(pad_size, np.nan), arr])
                    padded_aligned.append(feature_name)

                # Add to DataFrame
                try:
                    features_df[feature_name] = pd.to_numeric(arr, errors="coerce")
                    added_columns.append(feature_name)
                except Exception:
                    # If a column still fails, skip it
                    continue

            # Drop columns that are entirely NaN (e.g., failed conversions)
            if not features_df.empty:
                features_df = features_df.dropna(axis=1, how="all")

            # Remove constant columns (nunique <= 1) to avoid useless features
            if not features_df.empty:
                nunique = features_df.nunique(dropna=True)
                constant_cols = nunique[nunique <= 1].index.tolist()
                if constant_cols:
                    self.logger.warning(f"Dropping {len(constant_cols)} constant features")
                    # Log names of constant features (capped)
                    max_list = 50
                    preview = constant_cols[:max_list]
                    self.logger.warning(f"Constant features: {preview}{' ...' if len(constant_cols) > max_list else ''}")
                    features_df = features_df.drop(columns=constant_cols)

            # Combine with labeled data
            combined_data = pd.concat([labeled_data, features_df], axis=1)

            # Remove duplicate columns if any
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

            # Remove raw OHLCV columns to prevent data leakage
            combined_data = self._remove_raw_ohlcv_columns(combined_data)

            # Log a brief summary for diagnostics (without spam)
            try:
                self.logger.info(
                    f"Combined features: added={len(added_columns)}, "
                    f"trimmed={len(trimmed_aligned)}, padded={len(padded_aligned)}, "
                    f"skipped_scalars={len(skipped_scalars)}"
                )
                total_attempted = max(1, len(advanced_features))
                skip_ratio = len(skipped_scalars) / total_attempted
                if skip_ratio > 0.5:
                    self.logger.warning(
                        f"⚠️ High scalar skip ratio ({skip_ratio:.1%}). This suggests a provider returned non-array features; "
                        f"review feature generators. Sample skipped: {skipped_scalars[:10]}"
                    )
                # Strict mode: treat any scalar offenders as an error
                if self.strict_feature_shapes and scalar_offenders:
                    raise ValueError(
                        f"Strict feature shape check failed: scalar features detected: {scalar_offenders[:20]}"
                    )
            except Exception:
                pass

            return combined_data

        except Exception as e:
            # Provide more diagnostics for common alignment issues
            try:
                self.logger.error(
                    f"Error combining features and labels: {e}. "
                    f"labeled_data.shape={labeled_data.shape if isinstance(labeled_data, pd.DataFrame) else 'n/a'}"
                )
            except Exception:
                self.logger.error(f"Error combining features and labels: {e}")
            return labeled_data

    def _ensure_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV data is present in the dataset."""
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in required_ohlcv if col not in data.columns]
        
        if missing_ohlcv:
            self.logger.warning(f"⚠️ Missing OHLCV columns: {missing_ohlcv}")
            # Try to create minimal OHLCV from available data
            if 'avg_price' in data.columns:
                if 'open' not in data.columns:
                    data['open'] = data['avg_price']
                if 'high' not in data.columns:
                    data['high'] = data['avg_price']
                if 'low' not in data.columns:
                    data['low'] = data['avg_price']
                if 'close' not in data.columns:
                    data['close'] = data['avg_price']
            
            if 'trade_volume' in data.columns and 'volume' not in data.columns:
                data['volume'] = data['trade_volume']
        
        return data

    def _remove_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove metadata columns that are not actual features."""
        metadata_columns = ['year', 'exchange', 'symbol', 'timeframe', 'month', 'day', 'day_of_month', 'quarter']
        columns_to_remove = [col for col in metadata_columns if col in data.columns]
        
        if columns_to_remove:
            self.logger.info(f"🗑️ Removing metadata columns: {columns_to_remove}")
            data = data.drop(columns=columns_to_remove)
        
        return data

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            # Get columns with datetime dtype
            datetime_columns = []
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                    datetime_columns.append(col)
            
            # Remove datetime columns
            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)
            
            # Also remove any timestamp columns that might cause issues
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing datetime columns: {e}")
            return data

    def _remove_raw_ohlcv_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove raw OHLCV columns to prevent data leakage in ML training."""
        try:
            # Define raw OHLCV and context columns that should not be used as features
            raw_ohlcv_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_volume",
                "trade_count",
                "avg_price",
                "min_price",
                "max_price",
                # Treat these as context inputs; engineered variants should be used instead
                "funding_rate",
                "volume_ratio",
            ]
            
            # Find columns that match raw OHLCV names
            ohlcv_columns_found = [col for col in raw_ohlcv_columns if col in data.columns]
            
            # Remove raw OHLCV columns
            if ohlcv_columns_found:
                self.logger.warning(f"🚨 CRITICAL: Found raw OHLCV columns in features: {ohlcv_columns_found}")
                self.logger.warning(f"🚨 Removing raw OHLCV columns to prevent data leakage!")
                data = data.drop(columns=ohlcv_columns_found)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing raw OHLCV columns: {e}")
            return data

    def _remove_stationarity_transform_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove intermediate stationarity helper columns that are not final engineered features."""
        try:
            to_drop = [
                c for c in data.columns
                if c.endswith("_log") or c.endswith("_returns") or c.endswith("_log_returns") or c.endswith("_diff") or c.endswith("_detrended")
            ]
            if to_drop:
                self.logger.info(f"Removing stationarity helper columns: {to_drop[:20]}" + (" ..." if len(to_drop) > 20 else ""))
                data = data.drop(columns=to_drop)
            return data
        except Exception as e:
            self.logger.error(f"Error removing stationarity helper columns: {e}")
            return data

    def _prepare_final_data_vectorized(
        self,
        autoencoder_features: pd.DataFrame,
        labeled_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare final data using vectorized operations."""
        try:
            # Combine autoencoder features with labeled data
            final_data = pd.concat([autoencoder_features, labeled_data], axis=1)

            # Remove duplicate columns
            final_data = final_data.loc[:, ~final_data.columns.duplicated()]

            # Ensure label column is present
            if "label" not in final_data.columns and "label" in labeled_data.columns:
                final_data["label"] = labeled_data["label"]

            # Remove infinite values
            final_data = final_data.replace([np.inf, -np.inf], np.nan)

            # Fill remaining NaN values
            final_data = final_data.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # Remove columns with all NaN values
            final_data = final_data.dropna(axis=1, how="all")

            # Drop stationarity helper columns (keep engineered features only)
            label_only = final_data[["label"]] if "label" in final_data.columns else None
            features_only = final_data.drop(columns=["label"]) if "label" in final_data.columns else final_data
            features_only = self._remove_stationarity_transform_columns(features_only)
            final_data = pd.concat([features_only, label_only], axis=1) if label_only is not None else features_only

            return final_data

        except Exception as e:
            self.logger.error(f"Error preparing final data: {e}")
            return autoencoder_features

    def _optimize_memory_usage_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage using efficient data types."""
        try:
            optimized_data = data.copy()

            # Optimize numeric columns
            for col in optimized_data.select_dtypes(include=[np.number]).columns:
                col_min = optimized_data[col].min()
                col_max = optimized_data[col].max()
                
                if col_min >= 0 and col_max <= 255:
                    optimized_data[col] = optimized_data[col].astype(np.uint8)
                elif col_min >= -32768 and col_max <= 32767:
                    optimized_data[col] = optimized_data[col].astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
                else:
                    optimized_data[col] = optimized_data[col].astype(np.float32)

            # Optimize categorical columns
            for col in optimized_data.select_dtypes(include=['object']).columns:
                if optimized_data[col].nunique() < 255:
                    optimized_data[col] = optimized_data[col].astype('category')

            return optimized_data

        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {e}")
            return data

    def _save_data_as_parquet(self, data: pd.DataFrame) -> None:
        """Save data as Parquet file."""
        try:
            import os
            from datetime import datetime

            # Create output directory
            output_dir = "data/vectorized_features"
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vectorized_features_{timestamp}.parquet"
            filepath = os.path.join(output_dir, filename)

            # Save as Parquet
            data.to_parquet(filepath, index=True, compression='snappy')
            self.logger.info(f"💾 Data saved as Parquet: {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving data as Parquet: {e}")

    def _log_feature_dict_summary(self, stage: str, features: dict[str, Any], step_no: str) -> None:
        try:
            os.makedirs("log/features_samples", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_stage = stage.replace(" ", "_")
            fname = f"log/features_samples/{ts}_{step_no}_{safe_stage}_Features.txt"
            # Build summary
            total = len(features)
            type_counts: dict[str, int] = {}
            lines: list[str] = []
            for k, v in features.items():
                t = type(v).__name__
                type_counts[t] = type_counts.get(t, 0) + 1
                # Summarize value
                summary = ""
                try:
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        summary = f"value={float(v):.6g}"
                    elif isinstance(v, pd.Series):
                        summary = f"series(len={len(v)}, nan={int(v.isna().sum())})"
                    elif isinstance(v, np.ndarray):
                        summary = f"ndarray(shape={v.shape})"
                    elif isinstance(v, (list, tuple)):
                        summary = f"{type(v).__name__}(len={len(v)})"
                    else:
                        # Fallback to string repr trimmed
                        s = str(v)
                        summary = (s[:80] + "...") if len(s) > 80 else s
                except Exception:
                    summary = "<unavailable>"
                lines.append(f"{k}: type={t} {summary}")
            with open(fname, "w") as f:
                f.write(f"Stage: {stage} | Step: {step_no} | Total features: {total}\n")
                if type_counts:
                    f.write("Type counts: " + ", ".join([f"{k}={v}" for k, v in sorted(type_counts.items())]) + "\n")
                # Cap lines to avoid huge files
                cap = 1000
                for line in lines[:cap]:
                    f.write(line + "\n")
                if len(lines) > cap:
                    f.write(f"... ({len(lines) - cap} more)\n")
            self.logger.info(f"Feature list written: {fname}")
        except Exception as e:
            self.logger.warning(f"Failed to write feature list for {stage}: {e}")

    def _log_dataframe_columns(self, stage: str, df: pd.DataFrame, step_no: str) -> None:
        try:
            os.makedirs("log/features_samples", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_stage = stage.replace(" ", "_")
            fname = f"log/features_samples/{ts}_{step_no}_{safe_stage}_Columns.txt"
            cols = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()
            numeric = df.select_dtypes(include=[np.number])
            nan_total = int(numeric.isna().sum().sum()) if not numeric.empty else 0
            inf_total = int(np.isinf(numeric).sum().sum()) if not numeric.empty else 0
            with open(fname, "w") as f:
                f.write(f"Stage: {stage} | Step: {step_no} | Columns: {len(cols)}\n")
                f.write(f"NaN_total={nan_total} | Inf_total={inf_total}\n")
                for c in cols:
                    f.write(f"{c}: {dtypes.get(c, 'unknown')}\n")
            self.logger.info(f"Column inventory written: {fname}")
        except Exception as e:
            self.logger.warning(f"Failed to write column inventory for {stage}: {e}")

    def _categorize_features(self, features: dict[str, Any]) -> dict[str, int]:
        categories: dict[str, int] = {
            "wavelet": 0,
            "momentum": 0,
            "volatility": 0,
            "correlation": 0,
            "liquidity": 0,
            "candlestick": 0,
            "microstructure": 0,
            "sr_distance": 0,
            "other": 0,
        }
        for name in features.keys():
            lname = name.lower()
            if any(x in lname for x in ["db", "cmor", "morl", "wavelet", "cwt", "dwt"]):
                categories["wavelet"] += 1
            elif any(x in lname for x in ["momentum", "roc"]):
                categories["momentum"] += 1
            elif "volatility" in lname:
                categories["volatility"] += 1
            elif any(x in lname for x in ["corr", "correlation"]):
                categories["correlation"] += 1
            elif any(x in lname for x in ["liquidity", "depth", "spread"]):
                categories["liquidity"] += 1
            elif any(x in lname for x in ["engulf", "hammer", "doji", "marubozu", "tweezer", "spinning_top", "shooting_star"]):
                categories["candlestick"] += 1
            elif any(x in lname for x in ["order_flow", "price_impact", "market_depth", "volume_price_impact"]):
                categories["microstructure"] += 1
            elif any(x in lname for x in ["sr_distance", "support_resistance"]):
                categories["sr_distance"] += 1
            else:
                categories["other"] += 1
        return categories

    def _format_and_align_features(self, features: dict[str, Any], target_index: pd.Index) -> tuple[dict[str, pd.Series], dict[str, Any]]:
        """Ensure each feature is a well-formed pd.Series aligned to target_index.
        Returns formatted_features and a report dict with diagnostics.
        """
        formatted: dict[str, pd.Series] = {}
        report = {
            "input_features": len(features),
            "converted_arrays": 0,
            "coerced_types": 0,
            "trimmed": 0,
            "padded": 0,
            "filled_nan": 0,
            "non_numeric_dropped": [],
        }
        num_rows = len(target_index)
        for name, value in features.items():
            try:
                series: pd.Series | None = None
                if isinstance(value, pd.Series):
                    series = value.copy()
                elif isinstance(value, np.ndarray):
                    series = pd.Series(value)
                    report["converted_arrays"] += 1
                elif isinstance(value, list):
                    series = pd.Series(value)
                    report["converted_arrays"] += 1
                else:
                    # Skip scalars and unsupported types
                    continue

                # Align length by trim/pad as needed
                if len(series) > num_rows:
                    series = series.iloc[-num_rows:]
                    report["trimmed"] += 1
                elif len(series) < num_rows:
                    pad = num_rows - len(series)
                    series = pd.concat([pd.Series([np.nan] * pad), series], ignore_index=True)
                    report["padded"] += 1

                # Set target index and enforce numeric dtype
                series.index = target_index
                series = pd.to_numeric(series, errors="coerce")
                # Fill NaNs by forward/backward fill then zero
                before_nan = int(series.isna().sum())
                if before_nan:
                    series = series.fillna(method="ffill").fillna(method="bfill").fillna(0)
                    report["filled_nan"] += before_nan

                # Cast to float32 for efficiency
                series = series.astype(np.float32)

                formatted[name] = series
            except Exception:
                # Drop problematic features silently but record
                report["non_numeric_dropped"].append(name)
                continue
        return formatted, report

    def _log_feature_format_report(self, stage: str, report: dict[str, Any], step_no: str) -> None:
        try:
            os.makedirs("log/features_samples", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_stage = stage.replace(" ", "_")
            fname = f"log/features_samples/{ts}_{step_no}_{safe_stage}_FormatReport.json"
            import json as _json
            with open(fname, "w") as f:
                f.write(_json.dumps(report, indent=2, default=str))
            self.logger.info(f"Feature format report written: {fname}")
        except Exception as e:
            self.logger.warning(f"Failed to write feature format report: {e}")


class VectorizedStationarityChecker:
    """Check and transform data for stationarity using vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedStationarityChecker")

    async def check_and_transform_stationarity(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Check and transform data for stationarity using vectorized operations."""
        try:
            self.logger.info("🔍 Checking data stationarity...")

            # Check price data stationarity
            price_stationary = self._check_price_stationarity_vectorized(price_data)
            if not price_stationary:
                self.logger.info("📈 Transforming price data for stationarity...")
                price_data = self._transform_price_stationarity_vectorized(price_data)

            # Check volume data stationarity
            volume_stationary = self._check_volume_stationarity_vectorized(volume_data)
            if not volume_stationary:
                self.logger.info("📊 Transforming volume data for stationarity...")
                volume_data = self._transform_volume_stationarity_vectorized(volume_data)

            # Check order flow data stationarity
            order_flow_stationary = True
            if order_flow_data is not None:
                order_flow_stationary = self._check_order_flow_stationarity_vectorized(order_flow_data)
                if not order_flow_stationary:
                    self.logger.info("🔄 Transforming order flow data for stationarity...")
                    order_flow_data = self._transform_order_flow_stationarity_vectorized(order_flow_data)

            return {
                "price_data": price_data,
                "volume_data": volume_data,
                "order_flow_data": order_flow_data,
                "price_stationary": price_stationary,
                "volume_stationary": volume_stationary,
                "order_flow_stationary": order_flow_stationary,
            }

        except Exception as e:
            self.logger.error(f"Error checking and transforming stationarity: {e}")
            return {
                "price_data": price_data,
                "volume_data": volume_data,
                "order_flow_data": order_flow_data,
                "price_stationary": False,
                "volume_stationary": False,
                "order_flow_stationary": False,
            }

    def _check_price_stationarity_vectorized(self, price_data: pd.DataFrame) -> bool:
        """Check price data stationarity using vectorized operations."""
        try:
            # Calculate returns
            returns = price_data["close"].pct_change().dropna()
            
            # Augmented Dickey-Fuller test (simplified)
            # For vectorized implementation, we'll use a simplified approach
            # In practice, you might want to use statsmodels.tsa.stattools.adfuller
            
            # Check for trend
            trend = np.polyfit(range(len(returns)), returns, 1)[0]
            trend_threshold = 0.001
            
            # Check for autocorrelation
            autocorr = returns.autocorr()
            autocorr_threshold = 0.1
            
            # Check for variance stability
            rolling_std = returns.rolling(20).std()
            variance_ratio = rolling_std.iloc[-1] / rolling_std.iloc[0] if len(rolling_std) > 0 else 1.0
            variance_threshold = 2.0
            
            is_stationary = (
                abs(trend) < trend_threshold and
                abs(autocorr) < autocorr_threshold and
                variance_ratio < variance_threshold
            )
            
            return is_stationary

        except Exception as e:
            self.logger.error(f"Error checking price stationarity: {e}")
            return False

    def _transform_price_stationarity_vectorized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Transform price data for stationarity using vectorized operations."""
        try:
            transformed_data = price_data.copy()
            
            # Ensure we preserve the original OHLCV columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in transformed_data.columns for col in required_columns):
                self.logger.warning(f"Missing required OHLCV columns. Available: {transformed_data.columns.tolist()}")
                return price_data
            
            # Calculate returns (first difference) - ADD to existing data, don't replace
            transformed_data["close_returns"] = transformed_data["close"].pct_change()
            transformed_data["open_returns"] = transformed_data["open"].pct_change()
            transformed_data["high_returns"] = transformed_data["high"].pct_change()
            transformed_data["low_returns"] = transformed_data["low"].pct_change()
            
            # Log transformation for variance stabilization - ADD to existing data
            transformed_data["close_log"] = np.log(transformed_data["close"])
            transformed_data["open_log"] = np.log(transformed_data["open"])
            transformed_data["high_log"] = np.log(transformed_data["high"])
            transformed_data["low_log"] = np.log(transformed_data["low"])
            
            # Remove trend using rolling mean - ADD to existing data
            window = 20
            transformed_data["close_detrended"] = (
                transformed_data["close"] - transformed_data["close"].rolling(window).mean()
            )
            
            # Ensure original OHLCV columns are still present
            self.logger.info(f"✅ Transformed price data shape: {transformed_data.shape}, columns: {transformed_data.columns.tolist()}")
            
            return transformed_data

        except Exception as e:
            self.logger.error(f"Error transforming price stationarity: {e}")
            return price_data

    def _check_volume_stationarity_vectorized(self, volume_data: pd.DataFrame) -> bool:
        """Check volume data stationarity using vectorized operations."""
        try:
            if "volume" not in volume_data.columns:
                return True
                
            volume = volume_data["volume"]
            
            # Check for trend
            trend = np.polyfit(range(len(volume)), volume, 1)[0]
            trend_threshold = 0.001
            
            # Check for autocorrelation
            autocorr = volume.autocorr()
            autocorr_threshold = 0.1
            
            # Check for variance stability
            rolling_std = volume.rolling(20).std()
            variance_ratio = rolling_std.iloc[-1] / rolling_std.iloc[0] if len(rolling_std) > 0 else 1.0
            variance_threshold = 2.0
            
            is_stationary = (
                abs(trend) < trend_threshold and
                abs(autocorr) < autocorr_threshold and
                variance_ratio < variance_threshold
            )
            
            return is_stationary

        except Exception as e:
            self.logger.error(f"Error checking volume stationarity: {e}")
            return False

    def _transform_volume_stationarity_vectorized(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Transform volume data for stationarity using vectorized operations."""
        try:
            transformed_data = volume_data.copy()
            
            # Ensure we preserve the original volume column
            if "volume" not in transformed_data.columns:
                self.logger.warning(f"Missing volume column. Available: {transformed_data.columns.tolist()}")
                return volume_data
            
            # Log transformation - ADD to existing data, don't replace
            transformed_data["volume_log"] = np.log(transformed_data["volume"])
            
            # Remove trend - ADD to existing data
            window = 20
            transformed_data["volume_detrended"] = (
                transformed_data["volume"] - transformed_data["volume"].rolling(window).mean()
            )
            
            # Normalize volume - ADD to existing data
            transformed_data["volume_normalized"] = (
                transformed_data["volume"] / transformed_data["volume"].rolling(window).mean()
            )
            
            # Ensure original volume column is still present
            self.logger.info(f"✅ Transformed volume data shape: {transformed_data.shape}, columns: {transformed_data.columns.tolist()}")
            
            return transformed_data

        except Exception as e:
            self.logger.error(f"Error transforming volume stationarity: {e}")
            return volume_data

    def _check_order_flow_stationarity_vectorized(self, order_flow_data: pd.DataFrame) -> bool:
        """Check order flow data stationarity using vectorized operations."""
        try:
            # Simplified check for order flow data
            # In practice, you might want more sophisticated checks
            
            # Check for any numeric columns
            numeric_cols = order_flow_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return True
                
            # Check first numeric column for stationarity
            first_col = numeric_cols[0]
            data = order_flow_data[first_col]
            
            # Check for trend
            trend = np.polyfit(range(len(data)), data, 1)[0]
            trend_threshold = 0.001
            
            # Check for autocorrelation
            autocorr = data.autocorr()
            autocorr_threshold = 0.1
            
            is_stationary = (
                abs(trend) < trend_threshold and
                abs(autocorr) < autocorr_threshold
            )
            
            return is_stationary

        except Exception as e:
            self.logger.error(f"Error checking order flow stationarity: {e}")
            return False

    def _transform_order_flow_stationarity_vectorized(self, order_flow_data: pd.DataFrame) -> pd.DataFrame:
        """Transform order flow data for stationarity using vectorized operations."""
        try:
            transformed_data = order_flow_data.copy()
            
            # Apply transformations to numeric columns
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Log transformation (if positive)
                if (transformed_data[col] > 0).all():
                    transformed_data[f"{col}_log"] = np.log(transformed_data[col])
                
                # Remove trend
                window = 20
                transformed_data[f"{col}_detrended"] = (
                    transformed_data[col] - transformed_data[col].rolling(window).mean()
                )
                
                # Normalize
                transformed_data[f"{col}_normalized"] = (
                    transformed_data[col] / transformed_data[col].rolling(window).mean()
                )
            
            return transformed_data

        except Exception as e:
            self.logger.error(f"Error transforming order flow stationarity: {e}")
            return order_flow_data


class VectorizedFeatureSelector:
    """Vectorized feature selector using multiple selection methods."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedFeatureSelector")

        # Feature selection configuration
        self.feature_selection_config = config.get("feature_selection", {})
        self.vif_threshold = self.feature_selection_config.get("vif_threshold", 10.0)  # Increased from 5.0
        self.mutual_info_threshold = self.feature_selection_config.get("mutual_info_threshold", 0.001)  # Decreased from 0.01
        self.lightgbm_importance_threshold = self.feature_selection_config.get("lightgbm_importance_threshold", 0.001)  # Decreased from 0.01
        self.min_features_to_keep = self.feature_selection_config.get("min_features_to_keep", 10)  # New parameter
        self.correlation_threshold = self.feature_selection_config.get("correlation_threshold", 0.98)  # Increased from 0.95
        self.max_removal_percentage = self.feature_selection_config.get("max_removal_percentage", 0.3)
        
        # PCA combination configuration
        pca_config = self.feature_selection_config.get("pca_combination", {})
        self.pca_high_vif_threshold = pca_config.get("high_vif_threshold", 20.0)
        self.pca_correlation_threshold = pca_config.get("correlation_threshold", 0.7)
        self.pca_variance_explained_threshold = pca_config.get("variance_explained_threshold", 0.95)
        self.pca_scaling_method = pca_config.get("scaling_method", "standard")
        
        # Enable/disable flags
        self.enable_constant_removal = self.feature_selection_config.get("enable_constant_removal", True)
        self.enable_correlation_removal = self.feature_selection_config.get("enable_correlation_removal", True)
        self.enable_vif_removal = self.feature_selection_config.get("enable_vif_removal", True)
        self.enable_mutual_info_removal = self.feature_selection_config.get("enable_mutual_info_removal", True)
        self.enable_importance_removal = self.feature_selection_config.get("enable_importance_removal", True)
        
        # Safety settings
        self.enable_safety_checks = self.feature_selection_config.get("enable_safety_checks", True)
        self.return_original_on_failure = self.feature_selection_config.get("return_original_on_failure", True)

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            # Get columns with datetime dtype
            datetime_columns = []
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                    datetime_columns.append(col)
            
            # Remove datetime columns
            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)
            
            # Also remove any timestamp columns that might cause issues
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing datetime columns: {e}")
            return data

    async def select_optimal_features(
        self,
        data: pd.DataFrame,
        labels: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Select optimal features using vectorized operations."""
        try:
            self.logger.info("🎯 Starting feature selection...")
            initial_features = len(data.columns)
            original_data = data.copy()  # Keep original for fallback
            
            # Remove datetime columns first
            data = self._remove_datetime_columns(data)
            
            # Handle NaN values in feature selection
            self.logger.info("Handling NaN values in feature selection...")
            data = data.fillna(method="ffill").fillna(method="bfill").fillna(0)

            # Remove constant features (if enabled)
            if self.enable_constant_removal:
                constant_features = []
                for col in data.columns:
                    if data[col].nunique() <= 1:
                        constant_features.append(col)
                
                if constant_features:
                    self.logger.info(f"Removed {len(constant_features)} constant features: {constant_features}")
                    data = data.drop(columns=constant_features)

            # Safety check after constant removal
            if self.enable_safety_checks and len(data.columns) < self.min_features_to_keep:
                self.logger.warning(f"Too few features after constant removal ({len(data.columns)}). Skipping further selection.")
                return data

            # Remove highly correlated features (only if enabled and we have enough features)
            if (self.enable_correlation_removal and 
                len(data.columns) > self.min_features_to_keep and 
                len(data.columns) > 2):  # Need at least 2 features for correlation
                
                correlated_features = self._remove_correlated_features_vectorized(data)
                if correlated_features:
                    # Check removal percentage
                    removal_percentage = len(correlated_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(correlated_features) >= self.min_features_to_keep):
                        self.logger.info(f"Removed {len(correlated_features)} highly correlated features")
                        try:
                            self.logger.info(f"Correlated features removed: {sorted(correlated_features)[:20]}" + (" ..." if len(correlated_features) > 20 else ""))
                        except Exception:
                            pass
                        data = data.drop(columns=correlated_features)
                    else:
                        self.logger.info(f"Skipping correlated feature removal (removal %: {removal_percentage:.2f} > threshold: {self.max_removal_percentage:.2f})")

            # Combine high VIF features using PCA instead of removal (if enabled and we have enough features)
            if (self.enable_vif_removal and 
                len(data.columns) > self.min_features_to_keep and 
                len(data.columns) > 2):  # Need at least 2 features for VIF
                
                self.logger.info("🔄 Applying PCA-based feature combination for highly collinear features...")
                data = self._combine_high_vif_features_with_pca(data)
                
                # After PCA combination, also remove any remaining extremely high VIF features
                # (those that couldn't be combined due to insufficient correlation)
                remaining_high_vif_features = self._remove_high_vif_features_vectorized(data)
                if remaining_high_vif_features:
                    # Check removal percentage
                    removal_percentage = len(remaining_high_vif_features) / len(data.columns)
                    
                    # For remaining high VIF features, be more conservative
                    vif_removal_threshold = 0.3
                    
                    if (removal_percentage <= vif_removal_threshold and 
                        len(data.columns) - len(remaining_high_vif_features) >= self.min_features_to_keep):
                        self.logger.info(f"Removed {len(remaining_high_vif_features)} remaining high VIF features (removal %: {removal_percentage:.2f})")
                        try:
                            self.logger.info(f"High VIF features removed: {sorted(remaining_high_vif_features)[:20]}" + (" ..." if len(remaining_high_vif_features) > 20 else ""))
                        except Exception:
                            pass
                        data = data.drop(columns=remaining_high_vif_features)
                    else:
                        self.logger.warning(f"⚠️ Remaining high VIF features detected but skipping removal (removal %: {removal_percentage:.2f} > threshold: {vif_removal_threshold:.2f})")
                        # Still remove some high VIF features to reduce multicollinearity
                        if len(remaining_high_vif_features) > 0:
                            # Remove the top 30% of remaining high VIF features
                            features_to_remove = remaining_high_vif_features[:max(1, len(remaining_high_vif_features)//3)]
                            if len(data.columns) - len(features_to_remove) >= self.min_features_to_keep:
                                self.logger.info(f"Removing top 30% of remaining high VIF features: {len(features_to_remove)} features")
                                try:
                                    self.logger.info(f"High VIF features (top slice) removed: {sorted(features_to_remove)[:20]}" + (" ..." if len(features_to_remove) > 20 else ""))
                                except Exception:
                                    pass
                                data = data.drop(columns=features_to_remove)

            # Remove low mutual information features (if enabled, labels provided, and we have enough features)
            if (self.enable_mutual_info_removal and 
                labels is not None and len(labels) > 0 and 
                len(data.columns) > self.min_features_to_keep):
                
                low_mi_features = self._remove_low_mutual_info_features_vectorized(data, labels)
                if low_mi_features:
                    # Check removal percentage
                    removal_percentage = len(low_mi_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(low_mi_features) >= self.min_features_to_keep):
                        self.logger.info(f"Removed {len(low_mi_features)} low mutual information features")
                        try:
                            self.logger.info(f"Low MI features removed: {sorted(low_mi_features)[:20]}" + (" ..." if len(low_mi_features) > 20 else ""))
                        except Exception:
                            pass
                        data = data.drop(columns=low_mi_features)
                    else:
                        self.logger.info(f"Skipping mutual info feature removal (removal %: {removal_percentage:.2f} > threshold: {self.max_removal_percentage:.2f})")

            # Remove low importance features (if enabled, labels provided, and we have enough features)
            if (self.enable_importance_removal and 
                labels is not None and len(labels) > 0 and 
                len(data.columns) > self.min_features_to_keep):
                
                low_importance_features = self._remove_low_importance_features_vectorized(data, labels)
                if low_importance_features:
                    # Check removal percentage
                    removal_percentage = len(low_importance_features) / len(data.columns)
                    if (removal_percentage <= self.max_removal_percentage and 
                        len(data.columns) - len(low_importance_features) >= self.min_features_to_keep):
                        self.logger.info(f"Removed {len(low_importance_features)} low importance features")
                        try:
                            self.logger.info(f"Low-importance features removed: {sorted(low_importance_features)[:20]}" + (" ..." if len(low_importance_features) > 20 else ""))
                        except Exception:
                            pass
                        data = data.drop(columns=low_importance_features)
                    else:
                        self.logger.info(f"Skipping importance feature removal (removal %: {removal_percentage:.2f} > threshold: {self.max_removal_percentage:.2f})")

            final_features = len(data.columns)
            self.logger.info(f"Feature selection completed. Initial features: {initial_features}, Final features: {final_features}")
            
            # Final safety check
            if final_features == 0:
                self.logger.error("No features remaining after selection!")
                if self.return_original_on_failure:
                    self.logger.info("Returning original data as fallback.")
                    return self._remove_datetime_columns(original_data)
                else:
                    raise ValueError("No features remaining after selection and fallback disabled")
            
            return data

        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            if self.return_original_on_failure:
                self.logger.info("Returning original data as fallback due to error.")
                return self._remove_datetime_columns(original_data)
            else:
                raise

    def _remove_correlated_features_vectorized(self, data: pd.DataFrame) -> List[str]:
        """Remove highly correlated features using vectorized operations."""
        try:
            # Handle NaN values by imputing with median for correlation calculation
            from sklearn.impute import SimpleImputer
            
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            correlation_matrix = data_imputed.corr()
            upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            high_correlation = np.abs(correlation_matrix) > self.correlation_threshold
            high_correlation = high_correlation & upper_triangle
            
            to_drop = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if high_correlation.iloc[i, j]:
                        to_drop.append(correlation_matrix.columns[j])
            
            return list(set(to_drop))

        except Exception as e:
            self.logger.error(f"Error removing correlated features: {e}")
            return []

    def _combine_high_vif_features_with_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Combine highly collinear features using PCA instead of removal."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.impute import SimpleImputer
            from sklearn.decomposition import PCA
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Calculate VIF scores for all features
            vif_scores = {}
            for col in data_imputed.columns:
                other_cols = [c for c in data_imputed.columns if c != col]
                if len(other_cols) > 0:
                    X = data_imputed[other_cols]
                    y = data_imputed[col]
                    
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
                    y_pred = reg.predict(X)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    vif = 1 / (1 - r_squared) if r_squared != 1 else np.inf
                    vif_scores[col] = vif
            
            # Use a high threshold to ensure only highly correlated features are combined
            high_vif_features = [col for col, vif in vif_scores.items() if vif > self.pca_high_vif_threshold]
            
            if not high_vif_features:
                self.logger.info(f"✅ No features with VIF > {self.pca_high_vif_threshold} found for PCA combination")
                return data
            
            self.logger.info(f"🔍 VIF Analysis: Found {len(high_vif_features)} features with VIF > {self.pca_high_vif_threshold} for PCA combination")
            
            # Log the highly collinear features
            sorted_vif = sorted([(col, vif_scores[col]) for col in high_vif_features], 
                              key=lambda x: x[1], reverse=True)
            for col, vif in sorted_vif:
                if vif == np.inf:
                    self.logger.warning(f"⚠️ Feature '{col}' has infinite VIF (perfect multicollinearity)")
                else:
                    self.logger.info(f"📊 High VIF feature - {col}: {vif:.2f}")
            
            # Create correlation matrix for high VIF features
            high_vif_data = data_imputed[high_vif_features]
            correlation_matrix = high_vif_data.corr().abs()
            
            # Use hierarchical clustering to group highly correlated features
            # Convert correlation matrix to distance matrix (1 - correlation)
            distance_matrix = 1 - correlation_matrix.values
            
            # Apply hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,  # Let it determine optimal number
                distance_threshold=1 - self.pca_correlation_threshold,  # Features with correlation > threshold will be grouped
                linkage='complete'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group features by clusters
            feature_clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in feature_clusters:
                    feature_clusters[label] = []
                feature_clusters[label].append(high_vif_features[i])
            
            # Filter out single-feature clusters (no need for PCA)
            multi_feature_clusters = {k: v for k, v in feature_clusters.items() if len(v) > 1}
            
            if not multi_feature_clusters:
                self.logger.info("✅ No multi-feature clusters found for PCA combination")
                return data
            
            self.logger.info(f"🔍 Found {len(multi_feature_clusters)} clusters of highly collinear features for PCA combination")
            
            # Apply PCA to each cluster
            result_data = data.copy()
            features_to_remove = []
            pca_components_added = 0
            
            for cluster_id, features in multi_feature_clusters.items():
                self.logger.info(f"📊 Processing cluster {cluster_id}: {features}")
                
                # Prepare data for PCA
                cluster_data = data_imputed[features]
                
                # Log original feature statistics before scaling
                self.logger.info(f"📊 Cluster {cluster_id} - Original feature statistics:")
                for feature in features:
                    mean_val = cluster_data[feature].mean()
                    std_val = cluster_data[feature].std()
                    range_val = cluster_data[feature].max() - cluster_data[feature].min()
                    self.logger.info(f"   {feature}: mean={mean_val:.4f}, std={std_val:.4f}, range={range_val:.4f}")
                
                # Scale the data - crucial for PCA to work correctly
                # This ensures features with larger numerical ranges don't disproportionately influence PCA
                if self.pca_scaling_method == "standard":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    self.logger.info(f"📊 Using StandardScaler (z-score normalization) for cluster {cluster_id}")
                elif self.pca_scaling_method == "robust":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    self.logger.info(f"📊 Using RobustScaler (robust to outliers) for cluster {cluster_id}")
                elif self.pca_scaling_method == "minmax":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    self.logger.info(f"📊 Using MinMaxScaler (0-1 scaling) for cluster {cluster_id}")
                else:
                    # Default to StandardScaler if invalid method specified
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    self.logger.warning(f"⚠️ Invalid scaling method '{self.pca_scaling_method}', using StandardScaler")
                
                cluster_data_scaled = scaler.fit_transform(cluster_data)
                
                # Validate scaling results
                cluster_data_scaled_df = pd.DataFrame(cluster_data_scaled, columns=features, index=cluster_data.index)
                self.logger.info(f"📊 Cluster {cluster_id} - Scaled feature statistics:")
                for feature in features:
                    mean_val = cluster_data_scaled_df[feature].mean()
                    std_val = cluster_data_scaled_df[feature].std()
                    min_val = cluster_data_scaled_df[feature].min()
                    max_val = cluster_data_scaled_df[feature].max()
                    self.logger.info(f"   {feature}: mean={mean_val:.6f}, std={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}]")
                
                # Verify scaling was successful based on the scaling method used
                scaling_validation = True
                if self.pca_scaling_method == "standard":
                    # For StandardScaler: should be ~0 mean, ~1 std
                    for feature in features:
                        mean_val = abs(cluster_data_scaled_df[feature].mean())
                        std_val = abs(cluster_data_scaled_df[feature].std() - 1.0)
                        if mean_val > 1e-3 or std_val > 1e-2:
                            self.logger.warning(f"⚠️ StandardScaler validation failed for {feature}: mean={mean_val:.2e}, std_deviation_from_1={std_val:.2e}")
                            scaling_validation = False
                elif self.pca_scaling_method == "robust":
                    # For RobustScaler: should be ~0 median, ~1 IQR-based scale
                    for feature in features:
                        median_val = abs(cluster_data_scaled_df[feature].median())
                        if median_val > 1e-3:
                            self.logger.warning(f"⚠️ RobustScaler validation failed for {feature}: median={median_val:.2e}")
                            scaling_validation = False
                elif self.pca_scaling_method == "minmax":
                    # For MinMaxScaler: should be in [0, 1] range
                    for feature in features:
                        min_val = cluster_data_scaled_df[feature].min()
                        max_val = cluster_data_scaled_df[feature].max()
                        if min_val < -1e-3 or max_val > 1 + 1e-3:
                            self.logger.warning(f"⚠️ MinMaxScaler validation failed for {feature}: range=[{min_val:.2e}, {max_val:.2e}]")
                            scaling_validation = False
                
                if not scaling_validation:
                    self.logger.warning(f"⚠️ Scaling validation failed for cluster {cluster_id}, but continuing with PCA")
                else:
                    self.logger.info(f"✅ Scaling validation passed for cluster {cluster_id}")
                
                # Apply PCA - keep components that explain the configured percentage of variance
                pca = PCA(n_components=self.pca_variance_explained_threshold)
                pca_components = pca.fit_transform(cluster_data_scaled)
                
                # Get number of components
                n_components = pca_components.shape[1]
                
                if n_components == 0:
                    self.logger.warning(f"⚠️ PCA for cluster {cluster_id} produced 0 components, skipping")
                    continue
                
                # Create new feature names
                cluster_name = f"pca_cluster_{cluster_id}"
                for i in range(n_components):
                    component_name = f"{cluster_name}_pc{i+1}"
                    
                    # Add the PCA component to the result data
                    result_data[component_name] = pca_components[:, i]
                    pca_components_added += 1
                
                # Mark original features for removal
                features_to_remove.extend(features)
                
                # Log the variance explained
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                self.logger.info(f"📊 Cluster {cluster_id} PCA: {n_components} components explain {cumulative_variance[-1]:.3f} of variance")
                for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
                    self.logger.info(f"   PC{i+1}: {var_ratio:.3f} variance, cumulative: {cum_var:.3f}")
            
            # Remove original highly collinear features
            if features_to_remove:
                result_data = result_data.drop(columns=features_to_remove)
                self.logger.info(f"🔄 Removed {len(features_to_remove)} highly collinear features")
                self.logger.info(f"🔄 Added {pca_components_added} PCA meta-features")
                self.logger.info(f"🔄 Net change: {pca_components_added - len(features_to_remove)} features")
            
            return result_data

        except Exception as e:
            self.logger.error(f"Error combining high VIF features with PCA: {e}")
            return data

    def _remove_high_vif_features_vectorized(self, data: pd.DataFrame) -> List[str]:
        """Remove features with high Variance Inflation Factor using vectorized operations."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            vif_scores = {}
            for col in data_imputed.columns:
                # Use other features to predict this feature
                other_cols = [c for c in data_imputed.columns if c != col]
                if len(other_cols) > 0:
                    X = data_imputed[other_cols]
                    y = data_imputed[col]
                    
                    # Fit linear regression
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
                    # Calculate R-squared
                    y_pred = reg.predict(X)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calculate VIF
                    vif = 1 / (1 - r_squared) if r_squared != 1 else np.inf
                    vif_scores[col] = vif
            
            # Log VIF scores for debugging
            high_vif_features = [col for col, vif in vif_scores.items() if vif > self.vif_threshold]
            
            if high_vif_features:
                self.logger.info(f"🔍 VIF Analysis: Found {len(high_vif_features)} features with VIF > {self.vif_threshold}")
                # Log the top 5 highest VIF features
                sorted_vif = sorted(vif_scores.items(), key=lambda x: x[1], reverse=True)
                top_vif = sorted_vif[:5]
                for col, vif in top_vif:
                    if vif == np.inf:
                        self.logger.warning(f"⚠️ Feature '{col}' has infinite VIF (perfect multicollinearity)")
                    else:
                        self.logger.info(f"📊 VIF scores - {col}: {vif:.2f}")
            else:
                self.logger.info(f"✅ VIF Analysis: All features have VIF <= {self.vif_threshold}")
            
            return high_vif_features

        except Exception as e:
            self.logger.error(f"Error removing high VIF features: {e}")
            return []

    def _remove_low_mutual_info_features_vectorized(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> List[str]:
        """Remove features with low mutual information using vectorized operations."""
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(data_imputed, labels, random_state=42)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(data_imputed.columns, mi_scores))
            
            # Return features with low mutual information
            low_mi_features = [
                col for col, score in feature_importance.items()
                if score < self.mutual_info_threshold
            ]
            
            return low_mi_features

        except Exception as e:
            self.logger.error(f"Error removing low mutual information features: {e}")
            return []

    def _remove_low_importance_features_vectorized(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> List[str]:
        """Remove features with low LightGBM importance using vectorized operations."""
        try:
            import lightgbm as lgb
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values by imputing with median
            imputer = SimpleImputer(strategy='median')
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
            
            # Train LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbose=-1,
                verbosity=-1,
            )
            # Suppress LightGBM training output
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(data_imputed, labels)
            
            # Get feature importance
            feature_importance = dict(zip(data_imputed.columns, model.feature_importances_))
            
            # Return features with low importance
            low_importance_features = [
                col for col, importance in feature_importance.items()
                if importance < self.lightgbm_importance_threshold
            ]
            
            return low_importance_features

        except Exception as e:
            self.logger.error(f"Error removing low importance features: {e}")
            return []


class VectorizedDataNormalizer:
    """Normalize data using various scaling methods with vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedDataNormalizer")
        
        # Configuration
        self.normalization_config = config.get("data_normalization", {})
        self.scaling_method = self.normalization_config.get("scaling_method", "robust")
        self.outlier_handling = self.normalization_config.get("outlier_handling", "clip")

    async def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using vectorized operations."""
        try:
            self.logger.info("📏 Normalizing data...")
            
            # Remove datetime columns first
            data = self._remove_datetime_columns(data)
            
            # Handle outliers
            data = self._clip_outliers_vectorized(data)
            
            # Apply robust scaling
            data = self._apply_robust_scaling_vectorized(data)
            
            self.logger.info("✅ Data normalization completed")
            return data

        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            return data

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            # Get columns with datetime dtype
            datetime_columns = []
            for col in data.columns:
                if data[col].dtype == 'datetime64[ns]' or 'datetime' in str(data[col].dtype).lower():
                    datetime_columns.append(col)
            
            # Remove datetime columns
            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)
            
            # Also remove any timestamp columns that might cause issues
            timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error removing datetime columns: {e}")
            return data

    def _clip_outliers_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using vectorized operations."""
        try:
            clipped_data = data.copy()
            
            for col in clipped_data.select_dtypes(include=[np.number]).columns:
                Q1 = clipped_data[col].quantile(0.25)
                Q3 = clipped_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                clipped_data[col] = clipped_data[col].clip(lower=lower_bound, upper=upper_bound)
            
            return clipped_data

        except Exception as e:
            self.logger.error(f"Error clipping outliers: {e}")
            return data

    def _remove_outliers_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using vectorized operations."""
        try:
            cleaned_data = data.copy()
            
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Create mask for outliers
                outlier_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                cleaned_data = cleaned_data[~outlier_mask]
            
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error removing outliers: {e}")
            return data

    def _apply_robust_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply robust scaling using vectorized operations."""
        try:
            from sklearn.preprocessing import RobustScaler
            
            scaler = RobustScaler()
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 0:
                scaled_data = scaler.fit_transform(numeric_data)
                data[numeric_data.columns] = scaled_data
            
            return data

        except Exception as e:
            self.logger.error(f"Error applying robust scaling: {e}")
            return data

    def _apply_standard_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling using vectorized operations."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 0:
                scaled_data = scaler.fit_transform(numeric_data)
                data[numeric_data.columns] = scaled_data
            
            return data

        except Exception as e:
            self.logger.error(f"Error applying standard scaling: {e}")
            return data

    def _apply_minmax_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling using vectorized operations."""
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            scaler = MinMaxScaler()
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 0:
                scaled_data = scaler.fit_transform(numeric_data)
                data[numeric_data.columns] = scaled_data
            
            return data

        except Exception as e:
            self.logger.error(f"Error applying min-max scaling: {e}")
            return data
