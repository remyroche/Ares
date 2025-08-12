# src/training/steps/step6_analyst_enhancement.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset

# Import shap with error handling
try:
    import shap
except ImportError:
    shap = None

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.config import CONFIG
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

# Suppress Optuna's verbose logging to keep the output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


"""
Compatibility shim for NumPy RNG unpickling across versions.
We avoid nested functions to keep the shim picklable.
"""
_NUMPY_RNG_UNPICKLE_PATCHED = False
_NP_ORIGINAL_BITGEN_CTOR = None  # type: ignore[var-annotated]


def _normalized_numpy_bitgen_ctor(bit_generator_name, state=None, *args, **kwargs):  # type: ignore[override]
    """Module-level normalized ctor to avoid creating a closure (picklable)."""
    global _NP_ORIGINAL_BITGEN_CTOR
    name_candidate = bit_generator_name
    try:
        if hasattr(name_candidate, "__name__"):
            name_candidate = name_candidate.__name__
        elif isinstance(name_candidate, str) and name_candidate.startswith("<class "):
            name_candidate = name_candidate.split(".")[-1].split("'>")[0]
    except Exception:
        pass

    effective_state = kwargs.get("state", state)
    try:
        return _NP_ORIGINAL_BITGEN_CTOR(name_candidate, effective_state)  # type: ignore[misc]
    except (TypeError, ValueError):
        try:
            return _NP_ORIGINAL_BITGEN_CTOR(name_candidate)  # type: ignore[misc]
        except Exception as ctor_exc:  # noqa: BLE001
            try:
                import numpy as _np

                bitgen_cls = getattr(_np.random, name_candidate, None)
                if bitgen_cls is None and name_candidate == "MT19937":
                    try:
                        import numpy.random._mt19937 as _mt  # type: ignore[attr-defined]

                        bitgen_cls = getattr(_mt, "MT19937", None)
                    except Exception:
                        bitgen_cls = None
                if bitgen_cls is not None:
                    return bitgen_cls()
            except Exception:
                pass
            raise ctor_exc


def _enable_numpy_rng_unpickle_compat(logger=None) -> None:
    """Enable compatibility for unpickling NumPy RNG BitGenerators (idempotent)."""
    global _NUMPY_RNG_UNPICKLE_PATCHED, _NP_ORIGINAL_BITGEN_CTOR
    if _NUMPY_RNG_UNPICKLE_PATCHED:
        return
    try:
        import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]

        original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
        if original_ctor is None:
            _NUMPY_RNG_UNPICKLE_PATCHED = True
            return

        _NP_ORIGINAL_BITGEN_CTOR = original_ctor
        np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]
        _NUMPY_RNG_UNPICKLE_PATCHED = True
        if logger is not None:
            logger.info("Applied NumPy RNG unpickle compatibility shim")
    except Exception as _shim_exc:  # noqa: BLE001
        _NUMPY_RNG_UNPICKLE_PATCHED = True
        if logger is not None:
            logger.warning(
                warning(
                    f"NumPy RNG unpickle compatibility shim not applied: {_shim_exc}"
                )
            )


class AnalystEnhancementStep:
    """
    Step 6: Analyst Models Enhancement.

    This step refines the trained analyst models through a sequential process:
    1.  **Hyperparameter Optimization (HPO):** Uses Optuna with early pruning to find the best hyperparameters efficiently.
    2.  **Feature Selection:** Employs robust feature selection methods that work around SHAP/Keras compatibility issues.
    3.  **Final Retraining:** Trains a new model from scratch using the best hyperparameters and the optimal feature set.
    4.  **Advanced Optimization (Optional):** Applies techniques like quantization, structured pruning (WANDA),
        and knowledge distillation for further efficiency and performance gains, especially for neural network models.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initializes the AnalystEnhancementStep.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the step.
        """
        self.config = config
        self.logger = system_logger
        # --- Mac M1/M2/M3 (Apple Silicon) Specific Setup ---
        # Use 'mps' for PyTorch to leverage Apple's Metal Performance Shaders for GPU acceleration.
        # Fallback to 'cpu' if MPS is not available or hangs.
        self.device = self._safe_get_device()
        self.logger.info(f"Using device: {self.device.upper()} for PyTorch operations.")

        # Explicit feature isolation: non-feature columns to exclude from selection
        self._METADATA_COLUMNS: list[str] = [
            "timestamp",
            "exchange",
            "symbol",
            "timeframe",
            "split",
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_month",
            "quarter",
        ]
        self._LABEL_COLUMNS: set[str] = {"label", "target", "y", "class", "signal", "prediction"}

    def _safe_get_device(self) -> str:
        """Safely determine the best device to use with timeout protection."""
        try:
            # Use threading with timeout to prevent hanging
            import threading
            import queue

            result_queue = queue.Queue()

            def check_mps():
                try:
                    is_available = torch.backends.mps.is_available()
                    result_queue.put(("mps" if is_available else "cpu", None))
                except Exception as e:
                    result_queue.put(("cpu", e))

            # Start the check in a separate thread
            thread = threading.Thread(target=check_mps)
            thread.daemon = True
            thread.start()

            # Wait for result with timeout
            try:
                device, error = result_queue.get(timeout=10)  # 10 second timeout
                if error:
                    self.logger.error(failed("MPS check failed: {error}, using CPU"))
                    return "cpu"
                return device
            except queue.Empty:
                self.logger.error(
                    timeout("MPS availability check timed out, using CPU")
                )
                return "cpu"

        except Exception as e:
            self.logger.error(error("Error checking MPS availability: {e}, using CPU"))
            return "cpu"

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst enhancement step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst enhancement step."""
        self.logger.info("Initializing Analyst Enhancement Step...")
        self.logger.info("Analyst Enhancement Step initialized successfully.")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst enhancement step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Executes the full analyst model enhancement pipeline for each regime.

        Args:
            training_input (Dict[str, Any]): Input parameters, including symbol, exchange, and data directories.
            pipeline_state (Dict[str, Any]): The current state of the pipeline.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the enhancement process.
        """
        self.logger.info("🚀 Starting Step 6: Analyst Enhancement - Model Optimization and Feature Selection")
        self.logger.info("🔄 Executing Analyst Enhancement...")
        try:
            print("[Step6] Executing Analyst Enhancement...")
        except Exception:
            pass
        start_time = datetime.now()

        try:
            data_dir = training_input.get("data_dir", "data/training")
            models_dir = os.path.join(data_dir, "analyst_models")
            # Use the main data_dir for regime data, not processed_data_dir
            regime_data_dir = data_dir

            self.logger.info(f"📁 Data directory: {data_dir}")
            self.logger.info(f"📁 Models directory: {models_dir}")
            self.logger.info(f"📁 Regime data directory: {regime_data_dir}")

            self.logger.info("🔄 Loading analyst models from previous step...")
            self.logger.info({"msg": "Load models start", "dir": models_dir})
            try:
                print(f"[Step6] Load models start dir={models_dir}")
            except Exception:
                pass
            analyst_models = self._load_models(models_dir)
            self.logger.info(
                {"msg": "Load models complete", "count": len(analyst_models or {})}
            )
            try:
                print(f"[Step6] Load models complete count={len(analyst_models or {})}")
            except Exception:
                pass
            if not analyst_models:
                msg = f"No analyst models found in {models_dir}"
                raise ValueError(msg)
            if isinstance(analyst_models, dict):
                try:
                    regimes_count = len(analyst_models)
                    counts_per_regime = {
                        k: (len(v) if isinstance(v, dict) else "n/a")
                        for k, v in analyst_models.items()
                    }
                    self.logger.info(
                        f"Loaded analyst models summary: regimes={regimes_count}, models_per_regime={counts_per_regime}",
                    )
                except Exception:
                    pass

            # Log performance metrics before enhancement
            try:
                data_loader = get_unified_data_loader(self.config)
                perf_metrics = data_loader.get_performance_metrics()
                self.logger.info(f"📊 Performance before enhancement:")
                self.logger.info(
                    f"   Memory Usage: {perf_metrics['memory_usage']['percent']:.1f}%"
                )
                self.logger.info(
                    f"   Cache Size: {perf_metrics['cache_stats']['cache_size']}/{perf_metrics['cache_stats']['max_cache_size']}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Could not get performance metrics: {e}")

            # Enable parallel processing for model enhancement
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            import gc

            self.logger.info("🔄 Setting up parallel processing for model enhancement...")
            enhanced_models_summary = {}

            # Process regimes in parallel for better efficiency
            async def enhance_regime_models(regime_name, regime_models):
                self.logger.info(f"🚀 Starting enhancement for regime: {regime_name}")
                self.logger.info(f"📊 Regime {regime_name} has {len(regime_models)} models to enhance")
                try:
                    print(f"[Step6] Regime start {regime_name}")
                except Exception:
                    pass

                try:
                    self.logger.info(f"📂 Loading training data for regime: {regime_name}")
                    X_train, y_train, X_val, y_val = self._load_regime_data(
                        regime_data_dir,
                        regime_name,
                    )
                    self.logger.info(f"✅ Loaded data for regime {regime_name}: train={X_train.shape}, val={X_val.shape}")
                except FileNotFoundError as e:
                    self.logger.error(error("⚠️ {e} — skipping regime '{regime_name}'"))
                    return regime_name, {}

                # Memory cleanup before processing
                self.logger.info(f"🧹 Performing memory cleanup for regime: {regime_name}")
                gc.collect()

                enhanced_regime_models = {}
                self.logger.info(f"🔄 Starting model enhancement loop for regime: {regime_name}")
                for i, (model_name, model_data) in enumerate(regime_models.items(), 1):
                    self.logger.info(f"🔧 Enhancing model {i}/{len(regime_models)}: {model_name} for {regime_name}...")

                    enhanced_model_package = await self._enhance_single_model(
                        model_data,
                        model_name,
                        regime_name,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                    )
                    enhanced_regime_models[model_name] = enhanced_model_package
                    self.logger.info(f"✅ Completed enhancement for {model_name} in regime {regime_name}")

                    # Memory cleanup after each model
                    self.logger.info(f"🧹 Memory cleanup after {model_name}")
                    gc.collect()

                try:
                    print(f"[Step6] Regime complete {regime_name}")
                except Exception:
                    pass

                return regime_name, enhanced_regime_models

            # Create tasks for parallel processing
            self.logger.info(f"🔄 Creating parallel processing tasks for {len(analyst_models)} regimes...")
            tasks = []
            for regime_name, regime_models in analyst_models.items():
                task = enhance_regime_models(regime_name, regime_models)
                tasks.append(task)

            # Execute tasks with limited concurrency to avoid memory issues
            max_concurrent = min(3, len(tasks))  # Limit to 3 concurrent regimes
            self.logger.info(f"⚡ Processing {len(tasks)} regimes with max {max_concurrent} concurrent tasks")
            
            for batch_idx, i in enumerate(range(0, len(tasks), max_concurrent), 1):
                batch = tasks[i : i + max_concurrent]
                self.logger.info(f"🔄 Processing batch {batch_idx}: regimes {i+1}-{min(i+max_concurrent, len(tasks))}")
                results = await asyncio.gather(*batch, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(
                            f"❌ Error in parallel regime processing: {result}"
                        )
                    else:
                        regime_name, enhanced_regime_models = result
                        enhanced_models_summary[regime_name] = enhanced_regime_models
                        self.logger.info(f"✅ Completed batch processing for regime: {regime_name}")

                # Memory cleanup between batches
                self.logger.info(f"🧹 Memory cleanup after batch {batch_idx}")
                gc.collect()

            self.logger.info("💾 Saving enhanced models...")
            enhanced_models_dir = self._save_enhanced_models(
                enhanced_models_summary,
                data_dir,
                training_input,
            )

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"✅ Analyst enhancement completed in {duration:.2f}s. Results saved to {enhanced_models_dir}",
            )
            try:
                print(
                    f"[Step6] Completed in {duration:.2f}s. saved={enhanced_models_dir}"
                )
            except Exception:
                pass

            pipeline_state["enhanced_analyst_models"] = enhanced_models_summary
            return {
                "status": "SUCCESS",
                "enhanced_models_dir": enhanced_models_dir,
                "duration": duration,
            }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"❌ Error in Analyst Enhancement after {duration:.2f}s: {e}",
                exc_info=True,
            )
            return {"status": "FAILED", "error": str(e), "duration": duration}

    def _load_models(self, models_dir: str) -> dict[str, Any]:
        """Loads all analyst models from the specified directory."""
        # Ensure NumPy RNG pickles created under different versions can be loaded
        _enable_numpy_rng_unpickle_compat(self.logger)
        analyst_models = {}
        if not os.path.exists(models_dir):
            return analyst_models

        for regime_dir in os.listdir(models_dir):
            regime_path = os.path.join(models_dir, regime_dir)
            if os.path.isdir(regime_path):
                regime_models = {}
                for model_file in os.listdir(regime_path):
                    if model_file.endswith(".pkl") or model_file.endswith(".joblib"):
                        model_name = model_file.replace(".pkl", "")
                        model_name = model_name.replace(".joblib", "")
                        model_path = os.path.join(regime_path, model_file)
                        try:
                            if model_file.endswith(".joblib"):
                                regime_models[model_name] = joblib.load(model_path)
                            else:
                                with open(model_path, "rb") as f:
                                    regime_models[model_name] = pickle.load(f)
                        except (ValueError, TypeError) as e:
                            # Detect legacy RNG state incompatibility and force rebuild
                            error_text = str(e)
                            legacy_rng_issue = (
                                "legacy MT19937" in error_text
                                or "MT19937 state" in error_text
                                or "BitGenerator" in error_text
                            )
                            if legacy_rng_issue:
                                self.logger.warning(
                                    f"Detected legacy RNG state incompatibility for {model_name}; forcing safe rebuild (skip joblib retry)",
                                )
                                # Try to recreate the model from the training data
                                self.logger.info(
                                    f"Attempting to recreate {model_name}...",
                                )
                            else:
                                # Otherwise, attempt a joblib retry before rebuild
                                self.logger.warning(
                                    f"Failed to load {model_name} with pickle: {e}",
                                )
                                self.logger.info(
                                    f"Attempting to load {model_name} with joblib...",
                                )
                                try:
                                    import joblib

                                    regime_models[model_name] = joblib.load(model_path)
                                    continue  # successful joblib load; skip rebuild
                                except Exception as joblib_error:
                                    self.logger.exception(
                                        f"Failed to load {model_name} with joblib: {joblib_error}",
                                    )
                                    self.logger.info(
                                        f"Attempting to recreate {model_name}...",
                                    )
                                try:
                                    # Load the training data to recreate the model
                                    data_dir = self.config.get(
                                        "data_dir",
                                        "data/training",
                                    )
                                    symbol = self.config.get("symbol", "ETHUSDT")
                                    exchange = self.config.get("exchange", "BINANCE")

                                    # Load the training data
                                    train_data_base = (
                                        f"{data_dir}/{exchange}_{symbol}_train_data"
                                    )
                                    parquet_path = f"{train_data_base}.parquet"
                                    pickle_path = f"{train_data_base}.pkl"
                                    if os.path.exists(parquet_path):
                                        # Prefer materialized projection, then dataset scan
                                        try:
                                            from src.training.enhanced_training_manager_optimized import (
                                                ParquetDatasetManager,
                                            )

                                            pdm = ParquetDatasetManager(
                                                logger=self.logger
                                            )
                                            # 1) Reader shortcut: model-specific materialized projection
                                            feat_cols = self.config.get(
                                                "model_feature_columns"
                                            ) or self.config.get("feature_columns")
                                            label_col = self.config.get(
                                                "label_column", "label"
                                            )
                                            proj_base = os.path.join(
                                                "data_cache",
                                                "parquet",
                                                f"proj_features_{model_name}",
                                            )
                                            if (
                                                os.path.isdir(proj_base)
                                                and isinstance(feat_cols, list)
                                                and len(feat_cols) > 0
                                            ):
                                                proj_filters = [
                                                    ("exchange", "==", exchange),
                                                    ("symbol", "==", symbol),
                                                    (
                                                        "timeframe",
                                                        "==",
                                                        self.config.get(
                                                            "timeframe", "1m"
                                                        ),
                                                    ),
                                                    ("split", "==", "train"),
                                                ]
                                                t0 = self.config.get(
                                                    "t0_ms"
                                                ) or self.config.get(
                                                    "start_timestamp_ms"
                                                )
                                                t1 = self.config.get(
                                                    "t1_ms"
                                                ) or self.config.get("end_timestamp_ms")
                                                if t0 is not None:
                                                    proj_filters.append(
                                                        ("timestamp", ">=", int(t0))
                                                    )
                                                if t1 is not None:
                                                    proj_filters.append(
                                                        ("timestamp", "<", int(t1))
                                                    )
                                                cols = [
                                                    "timestamp",
                                                    *feat_cols,
                                                    label_col,
                                                ]
                                                train_data = pdm.cached_projection(
                                                    base_dir=proj_base,
                                                    filters=proj_filters,
                                                    columns=cols,
                                                    cache_dir="data_cache/projections",
                                                    cache_key_prefix=f"proj_features_{model_name}_{exchange}_{symbol}_{self.config.get('timeframe','1m')}_train",
                                                    snapshot_version="v1",
                                                    ttl_seconds=3600,
                                                    batch_size=131072,
                                                )
                                            else:
                                                # 2) Partitioned base labeled dataset scan
                                                dataset_base = os.path.join(
                                                    data_dir, "parquet", "labeled"
                                                )
                                            if os.path.isdir(dataset_base):
                                                filters = [
                                                    ("exchange", "==", exchange),
                                                    ("symbol", "==", symbol),
                                                    (
                                                        "timeframe",
                                                        "==",
                                                        self.config.get(
                                                            "timeframe", "1m"
                                                        ),
                                                    ),
                                                    ("split", "==", "train"),
                                                ]
                                                # Optional time filters from config
                                                t0 = self.config.get(
                                                    "t0_ms"
                                                ) or self.config.get(
                                                    "start_timestamp_ms"
                                                )
                                                t1 = self.config.get(
                                                    "t1_ms"
                                                ) or self.config.get("end_timestamp_ms")
                                                if t0 is not None:
                                                    filters.append(
                                                        ("timestamp", ">=", int(t0))
                                                    )
                                                if t1 is not None:
                                                    filters.append(
                                                        ("timestamp", "<", int(t1))
                                                    )
                                                # Project features + label if provided in config
                                                columns = None
                                                if (
                                                    isinstance(feat_cols, list)
                                                    and len(feat_cols) > 0
                                                ):
                                                    columns = list(
                                                        dict.fromkeys(
                                                            [
                                                                "timestamp",
                                                                *feat_cols,
                                                                label_col,
                                                            ]
                                                        )
                                                    )
                                                cache_key = f"labeled_{exchange}_{symbol}_{self.config.get('timeframe','1m')}_train"

                                                def _arrow_preprocess(tbl):
                                                    import pyarrow as _pa
                                                    import pyarrow.compute as pc

                                                    cols = list(tbl.schema.names)
                                                    # Ensure timestamp is int64
                                                    if (
                                                        "timestamp" in cols
                                                        and not _pa.types.is_int64(
                                                            tbl.schema.field(
                                                                "timestamp"
                                                            ).type
                                                        )
                                                    ):
                                                        tbl = tbl.set_column(
                                                            cols.index("timestamp"),
                                                            "timestamp",
                                                            pc.cast(
                                                                tbl.column("timestamp"),
                                                                _pa.int64(),
                                                            ),
                                                        )
                                                    return tbl

                                                train_data = pdm.cached_projection(
                                                    base_dir=dataset_base,
                                                    filters=filters,
                                                    columns=columns or [],
                                                    cache_dir="data_cache/projections",
                                                    cache_key_prefix=cache_key,
                                                    snapshot_version="v1",
                                                    ttl_seconds=3600,
                                                    batch_size=131072,
                                                    arrow_transform=_arrow_preprocess,
                                                )
                                            else:
                                                try:
                                                    feat_cols = self.config.get(
                                                        "model_feature_columns"
                                                    ) or self.config.get(
                                                        "feature_columns"
                                                    )
                                                    label_col = self.config.get(
                                                        "label_column", "label"
                                                    )
                                                    from src.utils.logger import (
                                                        log_io_operation,
                                                        log_dataframe_overview,
                                                    )

                                                    if (
                                                        isinstance(feat_cols, list)
                                                        and len(feat_cols) > 0
                                                    ):
                                                        with log_io_operation(
                                                            self.logger,
                                                            "read_parquet",
                                                            parquet_path,
                                                            columns=True,
                                                        ):
                                                            train_data = (
                                                                pd.read_parquet(
                                                                    parquet_path,
                                                                    columns=[
                                                                        "timestamp",
                                                                        *feat_cols,
                                                                        label_col,
                                                                    ],
                                                                )
                                                            )
                                                    else:
                                                        with log_io_operation(
                                                            self.logger,
                                                            "read_parquet",
                                                            parquet_path,
                                                        ):
                                                            train_data = (
                                                                pd.read_parquet(
                                                                    parquet_path
                                                                )
                                                            )
                                                    try:
                                                        log_dataframe_overview(
                                                            self.logger,
                                                            train_data,
                                                            name="train_data",
                                                        )
                                                    except Exception:
                                                        pass
                                                except Exception:
                                                    from src.utils.logger import (
                                                        log_io_operation,
                                                    )

                                                    with log_io_operation(
                                                        self.logger,
                                                        "read_parquet",
                                                        parquet_path,
                                                    ):
                                                        train_data = pd.read_parquet(
                                                            parquet_path
                                                        )
                                        except Exception:
                                            try:
                                                feat_cols = self.config.get(
                                                    "model_feature_columns"
                                                ) or self.config.get("feature_columns")
                                                label_col = self.config.get(
                                                    "label_column", "label"
                                                )
                                                from src.utils.logger import (
                                                    log_io_operation,
                                                )

                                                if (
                                                    isinstance(feat_cols, list)
                                                    and len(feat_cols) > 0
                                                ):
                                                    with log_io_operation(
                                                        self.logger,
                                                        "read_parquet",
                                                        parquet_path,
                                                        columns=True,
                                                    ):
                                                        train_data = pd.read_parquet(
                                                            parquet_path,
                                                            columns=[
                                                                "timestamp",
                                                                *feat_cols,
                                                                label_col,
                                                            ],
                                                        )
                                                else:
                                                    with log_io_operation(
                                                        self.logger,
                                                        "read_parquet",
                                                        parquet_path,
                                                    ):
                                                        train_data = pd.read_parquet(
                                                            parquet_path
                                                        )
                                            except Exception:
                                                from src.utils.logger import (
                                                    log_io_operation,
                                                )

                                                with log_io_operation(
                                                    self.logger,
                                                    "read_parquet",
                                                    parquet_path,
                                                ):
                                                    train_data = pd.read_parquet(
                                                        parquet_path
                                                    )
                                    elif os.path.exists(pickle_path):
                                        with open(pickle_path, "rb") as f:
                                            train_data = pickle.load(f)

                                        # Extract features and target
                                        if isinstance(train_data, pd.DataFrame):
                                            if "label" in train_data.columns:
                                                X = train_data.drop("label", axis=1)
                                                y = train_data["label"]

                                                # Recreate the model based on the model name
                                                if (
                                                    "random_forest"
                                                    in model_name.lower()
                                                ):
                                                    from sklearn.ensemble import (
                                                        RandomForestClassifier,
                                                    )

                                                    regime_models[model_name] = (
                                                        RandomForestClassifier(
                                                            n_estimators=100,
                                                            random_state=42,
                                                        )
                                                    )
                                                    regime_models[model_name].fit(X, y)
                                                elif "lightgbm" in model_name.lower():
                                                    from lightgbm import LGBMClassifier

                                                    regime_models[model_name] = (
                                                        LGBMClassifier(
                                                            n_estimators=100,
                                                            random_state=42,
                                                        )
                                                    )
                                                    regime_models[model_name].fit(X, y)
                                                elif "xgboost" in model_name.lower():
                                                    from xgboost import XGBClassifier

                                                    regime_models[model_name] = (
                                                        XGBClassifier(
                                                            n_estimators=100,
                                                            random_state=42,
                                                            verbose=0,  # Reduce verbose output during training
                                                        )
                                                    )
                                                    regime_models[model_name].fit(X, y)
                                                elif "svm" in model_name.lower():
                                                    from sklearn.svm import SVC

                                                    regime_models[model_name] = SVC(
                                                        random_state=42,
                                                    )
                                                    regime_models[model_name].fit(X, y)
                                                else:
                                                    # Default to Random Forest
                                                    from sklearn.ensemble import (
                                                        RandomForestClassifier,
                                                    )

                                                    regime_models[model_name] = (
                                                        RandomForestClassifier(
                                                            n_estimators=100,
                                                            random_state=42,
                                                        )
                                                    )
                                                    regime_models[model_name].fit(X, y)

                                                self.logger.info(
                                                    f"Successfully recreated {model_name}",
                                                )
                                            else:
                                                msg = "No label column found in training data"
                                                raise ValueError(
                                                    msg,
                                                )
                                        else:
                                            msg = "Training data is not a DataFrame"
                                            raise ValueError(
                                                msg,
                                            )
                                    else:
                                        msg = (
                                            "Training data not found: "
                                            f"{parquet_path} or {pickle_path}"
                                        )
                                        raise FileNotFoundError(
                                            msg,
                                        )
                                except Exception as recreate_error:
                                    self.logger.exception(
                                        f"Failed to recreate {model_name}: {recreate_error}",
                                    )
                                    # Skip this model and continue
                                    self.logger.warning(
                                        f"Skipping {model_name} due to loading issues",
                                    )
                                    continue
                analyst_models[regime_dir] = regime_models
        return analyst_models

    async def _load_regime_data(
        self,
        data_dir: str,
        regime_name: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads training and validation data for a specific regime using optimized unified data loader."""
        try:
            self.logger.info(
                f"Loading data for regime '{regime_name}' using unified data loader..."
            )

            symbol = self.config.get("symbol", "ETHUSDT")
            exchange = self.config.get("exchange", "BINANCE")
            timeframe = self.config.get("timeframe", "1m")

            # Try to load from unified data loader first (more efficient)
            try:
                data_loader = get_unified_data_loader(self.config)
                historical_data = await data_loader.load_unified_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    lookback_days=180,
                    use_streaming=True,  # Enable streaming for large datasets
                )

                if historical_data is not None and not historical_data.empty:
                    # Filter data by regime if regime information is available
                    if "regime" in historical_data.columns:
                        regime_data = historical_data[
                            historical_data["regime"] == regime_name
                        ]
                    else:
                        # If no regime column, use all data (fallback)
                        regime_data = historical_data

                    if not regime_data.empty:
                        self.logger.info(
                            f"✅ Loaded {len(regime_data)} rows for regime '{regime_name}' using unified data loader"
                        )

                        # Split into train/validation (80/20)
                        split_idx = int(len(regime_data) * 0.8)
                        train_data = regime_data.iloc[:split_idx]
                        val_data = regime_data.iloc[split_idx:]

                        # Extract features and target
                        if "label" in regime_data.columns:
                            X_train = train_data.drop(
                                ["label", "timestamp"], axis=1, errors="ignore"
                            )
                            y_train = train_data["label"]
                            X_val = val_data.drop(
                                ["label", "timestamp"], axis=1, errors="ignore"
                            )
                            y_val = val_data["label"]
                        else:
                            # Create synthetic target if no label column
                            X_train = train_data.drop(
                                ["timestamp"], axis=1, errors="ignore"
                            )
                            y_train = pd.Series(
                                np.random.choice([0, 1], size=len(train_data))
                            )
                            X_val = val_data.drop(
                                ["timestamp"], axis=1, errors="ignore"
                            )
                            y_val = pd.Series(
                                np.random.choice([0, 1], size=len(val_data))
                            )

                        return X_train, y_train, X_val, y_val

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Unified data loader failed for regime '{regime_name}': {e}, falling back to pickle files"
                )

            # Fallback to original pickle file loading
            regime_data_dir = os.path.join(data_dir, "regime_data")
            data_path = os.path.join(
                regime_data_dir,
                f"{exchange}_{symbol}_{regime_name}_data.pkl",
            )

            if not os.path.exists(data_path):
                msg = f"Data file for regime '{regime_name}' not found in {regime_data_dir}. Step 6 requires regime data from Step 3."
                raise FileNotFoundError(msg)

            # Load the combined data
            with open(data_path, "rb") as f:
                data = pickle.load(f)

                # Ensure data is a DataFrame
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)

                self.logger.info(
                    f"Loaded data shape: {data.shape}, columns: {list(data.columns)}",
                )

                # Remove non-numeric columns that XGBoost doesn't accept
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                data = data[numeric_columns]

                self.logger.info(
                    f"After numeric filtering: {data.shape}, columns: {list(data.columns)}",
                )

                # Check for target column with different possible names
                target_column = None
                target_candidates = [
                    "label",
                    "target",
                    "y",
                    "class",
                    "signal",
                    "prediction",
                ]

                for possible_target in target_candidates:
                    if possible_target in data.columns:
                        target_column = possible_target
                        self.logger.info(f"Found target column: {target_column}")
                        break

                if target_column is None:
                    self.logger.warning(
                        f"No target column found in regime data. Available columns: {list(data.columns)}",
                    )

                    # Try to create a meaningful target from available data
                    target_created = self._create_target_from_data(data, regime_name)

                    if target_created:
                        target_column = "label"
                        self.logger.info(
                            "Successfully created target column from available data",
                        )
                    else:
                        self.logger.warning(
                            "Creating dummy target - this may not be suitable for training",
                        )
                        data["label"] = np.random.choice([0, 1], size=len(data))
                        target_column = "label"
                # Rename target column to 'label' for consistency
                elif target_column != "label":
                    data["label"] = data[target_column]
                    data = data.drop(columns=[target_column])

                # Split features and target
                X = data.drop("label", axis=1)
                y = data["label"]

                # Validate target distribution
                unique_targets = y.unique()
                # Mode-aware, structured target distribution logging
                blank_mode = (
                    os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
                    or bool(CONFIG.get("BLANK_TRAINING_MODE", False))
                    or bool(CONFIG.get("blank_training_mode", False))
                )
                target_dist = dict(y.value_counts())
                self.logger.info(
                    f"Target distribution: {target_dist}",
                    extra={
                        "mode": "blank" if blank_mode else "full",
                        "target_distribution": target_dist,
                        "unique_classes": sorted(list(set(y.unique().tolist()))),
                        "note": (
                            "BLANK MODE: Skewed or single-class targets can be normal with limited data"
                            if blank_mode
                            else "",
                        ),
                        "next_steps": (
                            "Optional in blank mode: consider proxy or median-based target for quick diversity"
                            if blank_mode
                            else "Review labeling thresholds/event rates if distribution is degenerate",
                        ),
                    },
                )

                if len(unique_targets) <= 1:
                    # Emit mode-aware structured warning
                    self.logger.warning(
                        f"⚠️ Target has only {len(unique_targets)} unique values: {unique_targets}",
                        extra={
                            "mode": "blank" if blank_mode else "full",
                            "unique_values": unique_targets.tolist()
                            if hasattr(unique_targets, "tolist")
                            else list(unique_targets),
                            "unique_count": int(len(unique_targets)),
                            "note": (
                                "BLANK MODE: Often normal with limited data"
                                if blank_mode
                                else "Consider revisiting labeling thresholds or creating a proxy target"
                            ),
                            "next_steps": (
                                "Optional: use proxy/median-based target or adjust quick-test settings"
                                if blank_mode
                                else "Check label generation, event rates, or use proxy/median-based target",
                            ),
                        },
                    )
                    # Create a more diverse target if possible
                    if len(data.columns) > 1:
                        # Use the first numeric column as a proxy target
                        proxy_column = data.columns[0]
                        if proxy_column != "label":
                            proxy_values = data[proxy_column]
                            # Create binary target based on median
                            median_val = proxy_values.median()
                            y = (proxy_values > median_val).astype(int)
                            self.logger.info(
                                f"Created proxy target from {proxy_column} (median: {median_val})",
                            )
                            self.logger.info(
                                f"New target distribution: {dict(y.value_counts())}",
                            )

                # Split into train and validation
                train_size = int(0.8 * len(data))
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:]
                y_val = y[train_size:]

                self.logger.info(
                    f"Data loaded and split: X_train shape {X_train.shape}, X_val shape {X_val.shape}",
                )
                self.logger.info(
                    f"Target classes in training: {y_train.unique()}, in validation: {y_val.unique()}",
                )

                return X_train, y_train, X_val, y_val

        except Exception as e:
            self.logger.error(
                error("Error loading regime data for '{regime_name}': {e}")
            )
            raise

    def _create_target_from_data(self, data: pd.DataFrame, regime_name: str) -> bool:
        """
        Attempts to create a meaningful target column from available data.

        Args:
            data: The regime data DataFrame
            regime_name: Name of the regime

        Returns:
            bool: True if target was successfully created, False otherwise
        """
        try:
            # Look for price-related columns that could be used to create targets
            price_columns = [
                col
                for col in data.columns
                if any(
                    price_term in col.lower()
                    for price_term in ["close", "price", "value"]
                )
            ]

            if price_columns:
                # Use the first price column to create a target
                price_col = price_columns[0]
                price_values = data[price_col]

                # Create a simple momentum-based target
                if len(price_values) > 1:
                    # Calculate price changes
                    price_changes = price_values.pct_change().fillna(0)

                    # Create binary target based on positive/negative momentum
                    threshold = price_changes.std() * 0.1  # Small threshold
                    target = (price_changes > threshold).astype(int)

                    # Ensure we have at least 2 classes
                    if target.nunique() >= 2:
                        data["label"] = target
                        self.logger.info(
                            f"Created momentum-based target from {price_col}",
                        )
                        return True

            # Look for volume-related columns
            volume_columns = [col for col in data.columns if "volume" in col.lower()]

            if volume_columns:
                volume_col = volume_columns[0]
                volume_values = data[volume_col]

                # Create target based on volume spikes
                if len(volume_values) > 1:
                    volume_median = volume_values.median()
                    target = (volume_values > volume_median).astype(int)

                    if target.nunique() >= 2:
                        data["label"] = target
                        self.logger.info(
                            f"Created volume-based target from {volume_col}",
                        )
                        return True

            # Look for any numeric column with good variance
            for col in data.columns:
                if col != "label" and data[col].dtype in ["int64", "float64"]:
                    values = data[col]
                    if values.nunique() >= 2 and values.std() > 0:
                        # Create target based on above/below median
                        median_val = values.median()
                        target = (values > median_val).astype(int)

                        if target.nunique() >= 2:
                            data["label"] = target
                            self.logger.info(
                                f"Created target from {col} (median-based)",
                            )
                            return True

            return False

        except Exception as e:
            self.logger.error(error("Error creating target from data: {e}"))
            return False

    async def _enhance_single_model(
        self,
        model_data: dict[str, Any],
        model_name: str,
        regime_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Applies the full enhancement pipeline to a single model."""
        self.logger.info(f"🔧 Starting enhancement pipeline for {model_name} in {regime_name}")
        
        # Support both legacy dict payloads and direct sklearn model instances
        if isinstance(model_data, dict):
            original_accuracy = model_data.get("accuracy", "N/A")
            initial_model = model_data.get("model")
        else:
            original_accuracy = "N/A"
            initial_model = model_data
        self.logger.info(f"📊 Original model accuracy: {original_accuracy}")
        self.logger.info(f"📊 Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        self.logger.info(f"📊 Target distribution: {y_train.value_counts().to_dict()}")

        # Check if we have valid targets for training
        if y_train.nunique() <= 1:
            self.logger.warning(
                f"⚠️ Target has only {y_train.nunique()} unique values: {y_train.unique()}",
            )
            self.logger.warning(
                "⚠️ Skipping model enhancement due to insufficient target diversity",
            )
            return {
                "model": initial_model,  # Return original model if available
                "selected_features": list(X_train.columns),
                "accuracy": original_accuracy,
                "enhancement_metadata": {
                    "enhancement_date": datetime.now().isoformat(),
                    "original_accuracy": original_accuracy,
                    "hpo_score": 0.0,
                    "final_accuracy": original_accuracy,
                    "improvement": 0.0,
                    "best_params": {},
                    "feature_selection_method": "None - insufficient target diversity",
                    "original_feature_count": len(X_train.columns),
                    "selected_feature_count": len(X_train.columns),
                    "shap_summary": {},
                    "enhancement_applied": False,
                    "reason": f"Insufficient target diversity (only {y_train.nunique()} unique values)",
                },
            }

        # Enforce feature list isolation (exclude metadata/non-features)
        allow_features = [
            c for c in X_train.columns
            if c not in self._METADATA_COLUMNS and c not in self._LABEL_COLUMNS
        ]
        if len(allow_features) != X_train.shape[1]:
            self.logger.info(f"Feature isolation excluded {X_train.shape[1]-len(allow_features)} non-feature columns")
        X_train = X_train[allow_features]
        X_val = X_val[allow_features]

        # Apply VIF loop with CPA clustering before any model-based selection
        try:
            X_train, X_val, vif_summary = self._vif_loop_reduce_features(X_train, X_val)
            self.logger.info(
                {
                    "msg": "VIF loop completed",
                    "train_shape": X_train.shape,
                    "val_shape": X_val.shape,
                    "removed": len(vif_summary.get("removed_features", [])),
                    "cpa_clusters": vif_summary.get("cpa_count", 0),
                }
            )
        except Exception as e:
            self.logger.warning(f"VIF reduction skipped due to error: {e}")

        # --- 1. Hyperparameter Optimization with Pruning ---
        self.logger.info(f"🎯 Starting Hyperparameter Optimization for {model_name} in {regime_name}")
        self.logger.info(
            {
                "msg": "HPO start",
                "model": model_name,
                "regime": regime_name,
            }
        )
        try:
            print(f"[Enhancement] HPO start for {model_name} ({regime_name})")
        except Exception:
            pass
        best_params, hpo_score = await self._apply_hyperparameter_optimization(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        self.logger.info(
            {
                "msg": "HPO result",
                "model": model_name,
                "regime": regime_name,
                "best_score": float(hpo_score)
                if isinstance(hpo_score, (int, float))
                else None,
                "best_params": best_params,
            }
        )
        try:
            print(
                f"[Enhancement] HPO result for {model_name} ({regime_name}): best_score={hpo_score}"
            )
        except Exception:
            pass

        # --- 2. Feature Selection ---
        self.logger.info(
            {
                "msg": "Feature selection start",
                "model": model_name,
                "regime": regime_name,
            }
        )
        try:
            print(
                f"[Enhancement] Feature selection start for {model_name} ({regime_name})"
            )
        except Exception:
            pass
        temp_model = self._get_model_instance(model_name, best_params)
        temp_model.fit(X_train, y_train)

        (
            optimal_features,
            feature_selection_summary,
        ) = await self._select_optimal_features(
            temp_model,
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        self.logger.info(
            {
                "msg": "Feature selection result",
                "model": model_name,
                "regime": regime_name,
                "selected_feature_count": len(optimal_features),
                "summary": feature_selection_summary.get(
                    "summary", feature_selection_summary
                ),
            }
        )
        try:
            print(
                f"[Enhancement] Feature selection result for {model_name} ({regime_name}): selected={len(optimal_features)}"
            )
        except Exception:
            pass

        X_train_optimal = X_train[optimal_features]
        X_val_optimal = X_val[optimal_features]

        # --- 3. Final Retraining ---
        self.logger.info(
            {
                "msg": "Final retraining start",
                "model": model_name,
                "regime": regime_name,
                "selected_feature_count": len(optimal_features),
            }
        )
        try:
            print(
                f"[Enhancement] Final retraining start for {model_name} ({regime_name}) with {len(optimal_features)} features"
            )
        except Exception:
            pass
        final_model = self._get_model_instance(model_name, best_params)
        final_model.fit(X_train_optimal, y_train)

        final_accuracy = accuracy_score(y_val, final_model.predict(X_val_optimal))
        self.logger.info(
            {
                "msg": "Final retraining complete",
                "model": model_name,
                "regime": regime_name,
                "final_accuracy": float(final_accuracy),
            }
        )
        try:
            print(
                f"[Enhancement] Final retraining complete for {model_name} ({regime_name}): accuracy={float(final_accuracy):.4f}"
            )
        except Exception:
            pass

        # --- 4. Advanced Optimizations ---
        if model_name == "neural_network":
            # Apply advanced optimizations only for PyTorch models (skip for sklearn MLPClassifier)
            if isinstance(final_model, torch.nn.Module):
                final_model = self._apply_quantization(final_model)
                final_model = self._apply_wanda_pruning(final_model, X_train_optimal)
                final_model = self._apply_knowledge_distillation(
                    final_model,
                    X_train_optimal,
                    y_train,
                )

        return {
            "model": final_model,
            "selected_features": optimal_features,
            "accuracy": final_accuracy,
            "enhancement_metadata": {
                "enhancement_date": datetime.now().isoformat(),
                "original_accuracy": original_accuracy,
                "hpo_score": hpo_score,
                "final_accuracy": final_accuracy,
                "improvement": final_accuracy - original_accuracy
                if isinstance(original_accuracy, float)
                else "N/A",
                "best_params": best_params,
                "feature_selection_method": feature_selection_summary.get(
                    "method",
                    "robust_fallback",
                ),
                "original_feature_count": len(X_train.columns),
                "selected_feature_count": len(optimal_features),
                "feature_selection_summary": feature_selection_summary,
            },
        }

    def _get_model_instance(self, model_name: str, params: dict[str, Any]):
        """Factory function to get a model instance from its name and parameters."""
        if model_name in ["xgboost", "lightgbm"] and self.device == "mps":
            # LightGBM does not support 'mps' device. Force CPU for tree learners on Apple Silicon.
            params.pop("device", None)

        if model_name == "random_forest":
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        if model_name == "lightgbm":
            # Ensure CPU execution to avoid 'Unknown device type mps'
            safe_params = params.copy()
            safe_params.pop("device", None)
            safe_params["device_type"] = "cpu"
            return lgb.LGBMClassifier(**safe_params, random_state=42, n_jobs=-1)
        if model_name == "xgboost":
            # Remove eval_metric and device from params if they exist to avoid duplicate/unsupported parameters
            xgb_params = params.copy()
            if "eval_metric" in xgb_params:
                del xgb_params["eval_metric"]
            if "device" in xgb_params:
                del xgb_params["device"]

            return xgb.XGBClassifier(
                **xgb_params,
                random_state=42,
                tree_method="hist" if self.device == "cpu" else "auto",
                use_label_encoder=False,
                eval_metric="logloss",
                verbose=0,  # Reduce verbose output during training
            )
        if model_name == "svm":
            from sklearn.svm import SVC

            return SVC(**params, random_state=42, probability=True)
        if model_name == "neural_network":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                **params,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
        msg = f"Model {model_name} not supported."
        raise ValueError(msg)

    async def _apply_hyperparameter_optimization(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[dict[str, Any], float]:
        """Performs hyperparameter optimization using Optuna with early pruning."""
        self.logger.info(f"🚀 Running Optuna HPO with pruning for {model_name}...")

        # Track progress
        trial_count = 0
        # Determine blank mode (support both ENV and CONFIG flag)
        try:
            is_blank_env = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        except Exception:
            is_blank_env = False
        try:
            is_blank_cfg = bool(CONFIG.get("BLANK_TRAINING_MODE", False))
        except Exception:
            is_blank_cfg = False
        blank_mode = is_blank_env or is_blank_cfg

        default_trials = min(50, self.config.get("n_trials", 50))
        # In BLANK mode, restrict trials aggressively for speed across all models
        if blank_mode:
            total_trials = 3
        else:
            total_trials = default_trials
        self.logger.info(
            {
                "msg": "HPO trial plan",
                "model": model_name,
                "total_trials": total_trials,
                "blank_mode": blank_mode,
            }
        )
        try:
            print(
                f"[HPO] Plan for {model_name}: total_trials={total_trials}, blank_mode={blank_mode}"
            )
        except Exception:
            pass

        def objective(trial: optuna.trial.Trial) -> float:
            nonlocal trial_count
            trial_count += 1
            self.logger.info(
                {
                    "msg": "HPO trial start",
                    "model": model_name,
                    "trial": trial_count,
                    "of": total_trials,
                }
            )
            try:
                print(f"[HPO] {model_name} trial {trial_count}/{total_trials} start")
            except Exception:
                pass
            pruning_callback = None

            # Validate target data before proceeding
            if y_train.nunique() <= 1:
                self.logger.warning(
                    f"Target has only {y_train.nunique()} unique values, skipping optimization",
                )
                return 0.0

            if model_name == "lightgbm":
                # Align pruning metric and study direction for LightGBM using logloss (minimize)
                # Determine binary vs multiclass
                n_classes = int(len(set(pd.concat([y_train, y_val]).unique())))
                is_multiclass = n_classes > 2
                lgb_objective = "multiclass" if is_multiclass else "binary"
                lgb_metric = "multi_logloss" if is_multiclass else "binary_logloss"

                # Enable pruning in both blank and full modes
                pruning_callback = optuna.integration.LightGBMPruningCallback(
                    trial,
                    lgb_metric,
                )
                params = {
                    "objective": lgb_objective,
                    "metric": lgb_metric,
                    "verbosity": -1,
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),  # Reduced max to prevent getting stuck
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        1e-3,
                        0.3,
                        log=True,
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda",
                        1e-8,
                        10.0,
                        log=True,
                    ),
                    "early_stopping_rounds": 50,  # Add explicit early stopping
                }
            elif model_name == "xgboost":
                pruning_callback = optuna.integration.XGBoostPruningCallback(
                    trial,
                    "validation_0-logloss",
                )
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        1e-3,
                        0.3,
                        log=True,
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.6,
                        1.0,
                    ),
                    # Add base_score to prevent the error when all targets are the same
                    "base_score": 0.5,
                }
            elif model_name == "svm":
                # SVM doesn't support iterative pruning
                params = {
                    "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                    "kernel": trial.suggest_categorical(
                        "kernel",
                        ["rbf", "linear", "poly"],
                    ),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                }
                self.logger.info(
                    f"🔧 SVM Trial {trial_count}: C={params['C']:.3f}, kernel={params['kernel']}, gamma={params['gamma']}"
                )
            elif model_name == "neural_network":
                # Neural network doesn't support iterative pruning
                params = {
                    "hidden_layer_sizes": trial.suggest_categorical(
                        "hidden_layer_sizes",
                        [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
                    ),
                    "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    "learning_rate_init": trial.suggest_float(
                        "learning_rate_init",
                        1e-4,
                        1e-1,
                        log=True,
                    ),
                    "max_iter": trial.suggest_int("max_iter", 200, 1000),
                }
            else:  # RandomForest doesn't support iterative pruning.
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                }

            model = self._get_model_instance(model_name, params)

            # Train the model with appropriate parameters based on model type
            if model_name == "lightgbm":
                # LightGBM supports callbacks. Train with eval_set so pruning can observe logloss.
                use_pruning = pruning_callback is not None
                fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
                if use_pruning:
                    fit_kwargs["callbacks"] = [pruning_callback]
                # Suppress LightGBM training output
                import warnings
                import signal
                import os
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("LightGBM training timed out")
                
                # Set a timeout of 5 minutes for training
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes timeout
                
                # Suppress LightGBM output by redirecting stdout temporarily
                import sys
                from io import StringIO
                
                try:
                    # Redirect stdout to suppress LightGBM output
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train, **fit_kwargs)
                        
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                    signal.alarm(0)  # Cancel the alarm
            elif model_name == "xgboost":
                # XGBoost doesn't support callbacks parameter, use eval_set only
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                # SVM, Neural Network, and Random Forest don't support eval_set
                if model_name == "svm":
                    self.logger.info(
                        {
                            "msg": "Training model",
                            "model": model_name,
                            "trial": trial_count,
                        }
                    )
                model.fit(X_train, y_train)

            preds = model.predict(X_val)
            accuracy = accuracy_score(y_val, preds)
            # Return metric aligned with study direction: for LightGBM use logloss (minimize); others use accuracy (maximize)
            if model_name == "svm":
                self.logger.info(
                    {
                        "msg": "SVM trial result",
                        "trial": trial_count,
                        "accuracy": float(accuracy),
                    }
                )
                try:
                    print(
                        f"[HPO] SVM trial {trial_count} accuracy={float(accuracy):.4f}"
                    )
                except Exception:
                    pass
            if model_name == "lightgbm":
                from sklearn.metrics import log_loss

                # Ensure labels ordering covers all classes present
                labels_sorted = list(sorted(pd.unique(pd.concat([y_train, y_val]))))
                # Use predict_proba to compute logloss
                y_proba = model.predict_proba(X_val)
                try:
                    loss = log_loss(y_val, y_proba, labels=labels_sorted)
                except Exception:
                    # Fallback: if labels parameter causes issues, omit it
                    loss = log_loss(y_val, y_proba)
                return float(loss)
            return float(accuracy)

        # Align study direction with the objective metric
        study_direction = "minimize" if model_name == "lightgbm" else "maximize"
        study = optuna.create_study(
            direction=study_direction,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        # Add progress callback
        def progress_callback(study, trial):
            completed_trials = len(study.trials)
            if completed_trials % 1 == 0:  # Log every trial for better visibility
                self.logger.info(
                    {
                        "msg": "HPO progress",
                        "model": model_name,
                        "completed": completed_trials,
                        "total": total_trials,
                    }
                )
                try:
                    print(
                        f"[HPO] Progress {model_name}: {completed_trials}/{total_trials}"
                    )
                except Exception:
                    pass

        # Bound parallelism to avoid CPU thrashing or potential thread deadlocks on macOS
        try:
            import platform
        except Exception:
            platform = None
        try:
            is_macos = platform.system() == "Darwin" if platform else False
        except Exception:
            is_macos = False
        # Prefer conservative parallelism for SVM and on macOS to avoid thread contention
        parallel_jobs = (
            1 if (model_name == "svm" or is_macos) else min(4, os.cpu_count() or 4)
        )

        # Visibility around optimize lifecycle
        self.logger.info(
            {
                "msg": "HPO optimize start",
                "model": model_name,
                "total_trials": total_trials,
                "n_jobs": parallel_jobs,
            }
        )
        try:
            print(
                f"[HPO] optimize start {model_name}: trials={total_trials} n_jobs={parallel_jobs}"
            )
        except Exception:
            pass

        await asyncio.to_thread(
            study.optimize,
            objective,
            n_trials=total_trials,
            n_jobs=parallel_jobs,
            callbacks=[progress_callback] if model_name == "svm" else None,
        )

        self.logger.info(
            {
                "msg": "HPO optimize finished",
                "model": model_name,
                "completed_trials": len(study.trials),
            }
        )
        try:
            print(
                f"[HPO] optimize finished {model_name}: completed={len(study.trials)}/{total_trials}"
            )
        except Exception:
            pass

        if not study.best_trial:
            self.logger.warning(
                "Optuna study found no best trial, possibly due to all trials being pruned. Returning empty params.",
            )
            return {}, 0.0

        self.logger.info(
            {
                "msg": "HPO complete",
                "model": model_name,
                "best_score": float(study.best_value),
            }
        )
        # Extra visibility for terminal users
        try:
            print(
                f"[HPO] {model_name} complete. best_score={float(study.best_value):.6f}",
            )
        except Exception:
            pass
        if model_name == "svm":
            self.logger.info(
                {
                    "msg": "Best SVM parameters",
                    "params": study.best_params,
                }
            )
            # Provide a concise summary with parameters in the console as well
            try:
                print(
                    f"[HPO] {model_name} best_params={study.best_params}",
                )
            except Exception:
                pass
        return study.best_params, study.best_value

    async def _select_optimal_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[list[str], dict]:
        """Selects the most important features using enhanced tiered strategy with stability selection and look-ahead bias prevention."""
        self.logger.info("🎯 Selecting optimal features using enhanced tiered strategy with stability selection...")
        
        # Echo to console to make progress more visible during potentially long SHAP runs
        try:
            print(
                f"[Features] Starting enhanced feature selection for {model_name} (180 features) with stability selection",
            )
        except Exception:
            pass

        # Enforce feature isolation at selection start
        feature_names = [c for c in X_val.columns.tolist() if c not in self._METADATA_COLUMNS and c not in self._LABEL_COLUMNS]
        # Align training/validation to explicit feature list
        X_train = X_train[feature_names]
        X_val = X_val[feature_names]
        total_features = len(feature_names)
        
        self.logger.info(f"📊 Total features available: {total_features}")
        
        # Enhanced tiered feature selection strategy for large feature sets
        if total_features > 200:
            # Use tiered selection with stability selection for large feature sets
            optimal_features, selection_summary = await self._execute_stable_tiered_feature_selection(
                model, model_name, X_train, y_train, X_val, y_val, feature_names
            )
        else:
            # Use traditional selection with stability for smaller feature sets
            optimal_features, selection_summary = await self._execute_stable_traditional_feature_selection(
                model, model_name, X_train, y_train, X_val, y_val, feature_names
            )

        self.logger.info(
            f"✅ Selected {len(optimal_features)} optimal features from {total_features} total features",
        )
        return optimal_features, selection_summary

    async def _execute_stable_tiered_feature_selection(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list,
    ) -> tuple[list[str], dict]:
        """Execute stable tiered feature selection with bootstrapping to prevent selection instability."""
        
        # Load tiered selection configuration
        feature_config = self.config.get("feature_interactions", {})
        selection_tiers = feature_config.get("feature_selection_tiers", {})
        stability_config = feature_config.get("stability_selection", {})
        
        # Get stability selection parameters
        n_bootstrap_samples = stability_config.get("n_bootstrap_samples", 50)
        stability_threshold = stability_config.get("stability_threshold", 0.7)
        min_features_per_tier = stability_config.get("min_features_per_tier", 5)
        
        # Get tiered selection parameters
        tier_1_count = selection_tiers.get("tier_1_base_features", 80)
        tier_2_count = selection_tiers.get("tier_2_normalized_features", 40)
        tier_3_count = selection_tiers.get("tier_3_interaction_features", 60)
        tier_4_count = selection_tiers.get("tier_4_lagged_features", 40)
        tier_5_count = selection_tiers.get("tier_5_causality_features", 20)
        total_max_features = selection_tiers.get("total_max_features", 240)
        
        self.logger.info(f"🎯 Stable tiered feature selection targets (180 features):")
        self.logger.info(f"   Tier 1 (Core): {tier_1_count} features")
        self.logger.info(f"   Tier 2 (Normalized): {tier_2_count} features")
        self.logger.info(f"   Tier 3 (Interactions): {tier_3_count} features")
        self.logger.info(f"   Tier 4 (Lagged): {tier_4_count} features")
        self.logger.info(f"   Tier 5 (Causality): {tier_5_count} features")
        self.logger.info(f"   Total Max: {total_max_features} features")
        self.logger.info(f"   Stability: {n_bootstrap_samples} bootstrap samples, threshold: {stability_threshold}")
        
        # Categorize features by tier
        feature_categories = self._categorize_features_by_tier(feature_names)
        
        selected_features = []
        selection_summary = {
            "method": "stable_tiered_selection",
            "total_features": len(feature_names),
            "selected_features": 0,
            "tier_breakdown": {},
            "selection_details": {},
            "stability_metrics": {}
        }
        
        # Tier 1: Core features with stability selection
        tier_1_features = await self._select_stable_tier_1_features(
            model, model_name, X_train, y_train, X_val, y_val, 
            feature_categories["tier_1"], tier_1_count, n_bootstrap_samples, stability_threshold, min_features_per_tier
        )
        selected_features.extend(tier_1_features)
        selection_summary["tier_breakdown"]["tier_1_core"] = len(tier_1_features)
        self.logger.info(f"   ✅ Tier 1: Selected {len(tier_1_features)} stable core features")
        
        # Tier 2: Normalized features with stability selection
        tier_2_features = await self._select_stable_tier_2_features(
            model, model_name, X_train, y_train, X_val, y_val,
            feature_categories["tier_2"], tier_2_count, n_bootstrap_samples, stability_threshold, min_features_per_tier
        )
        selected_features.extend(tier_2_features)
        selection_summary["tier_breakdown"]["tier_2_normalized"] = len(tier_2_features)
        self.logger.info(f"   ✅ Tier 2: Selected {len(tier_2_features)} stable normalized features")
        
        # Tier 3: Interaction features with stability selection
        tier_3_features = await self._select_stable_tier_3_features(
            model, model_name, X_train, y_train, X_val, y_val,
            feature_categories["tier_3"], tier_3_count, n_bootstrap_samples, stability_threshold, min_features_per_tier
        )
        selected_features.extend(tier_3_features)
        selection_summary["tier_breakdown"]["tier_3_interactions"] = len(tier_3_features)
        self.logger.info(f"   ✅ Tier 3: Selected {len(tier_3_features)} stable interaction features")
        
        # Tier 4: Lagged features with stability selection
        tier_4_features = await self._select_stable_tier_4_features(
            model, model_name, X_train, y_train, X_val, y_val,
            feature_categories["tier_4"], tier_4_count, n_bootstrap_samples, stability_threshold, min_features_per_tier
        )
        selected_features.extend(tier_4_features)
        selection_summary["tier_breakdown"]["tier_4_lagged"] = len(tier_4_features)
        self.logger.info(f"   ✅ Tier 4: Selected {len(tier_4_features)} stable lagged features")
        
        # Tier 5: Causality features with stability selection
        tier_5_features = await self._select_stable_tier_5_features(
            model, model_name, X_train, y_train, X_val, y_val,
            feature_categories["tier_5"], tier_5_count, n_bootstrap_samples, stability_threshold, min_features_per_tier
        )
        selected_features.extend(tier_5_features)
        selection_summary["tier_breakdown"]["tier_5_causality"] = len(tier_5_features)
        self.logger.info(f"   ✅ Tier 5: Selected {len(tier_5_features)} stable causality features")
        
        # Apply final pruning if we exceed total_max_features
        if len(selected_features) > total_max_features:
            selected_features = await self._apply_stable_final_pruning(
                selected_features, X_val[selected_features], y_val, total_max_features, n_bootstrap_samples, stability_threshold
            )
            self.logger.info(f"   🔧 Final pruning: Reduced to {len(selected_features)} stable features")
        
        selection_summary["selected_features"] = len(selected_features)
        selection_summary["reduction_percentage"] = ((len(feature_names) - len(selected_features)) / len(feature_names) * 100)
        
        return selected_features, selection_summary

    async def _execute_stable_traditional_feature_selection(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list,
    ) -> tuple[list[str], dict]:
        """Execute stable traditional feature selection with bootstrapping."""
        
        # Load stability configuration
        feature_config = self.config.get("feature_interactions", {})
        stability_config = feature_config.get("stability_selection", {})
        
        # Get stability selection parameters
        n_bootstrap_samples = stability_config.get("n_bootstrap_samples", 50)
        stability_threshold = stability_config.get("stability_threshold", 0.7)
        
        # Ensure we keep at least 10 features or 50% of original features, whichever is larger
        min_features = max(10, len(feature_names) // 2)
        max_features = min(20, len(feature_names))  # Don't select more than 20 features

        try:
            # Try SHAP first with stability selection
            optimal_features, shap_summary = await self._try_stable_shap_feature_selection(
                model,
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
                feature_names,
                min_features,
                max_features,
                n_bootstrap_samples,
                stability_threshold,
            )
            if optimal_features:
                return optimal_features, {"method": "stable_shap", **shap_summary}
        except Exception as e:
            self.logger.warning(
                f"Stable SHAP analysis failed: {e}. Trying alternative methods...",
            )

        # Fallback to robust feature selection methods with stability
        optimal_features, fallback_summary = await self._robust_stable_feature_selection(
            model,
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names,
            min_features,
            max_features,
            n_bootstrap_samples,
            stability_threshold,
        )

        return optimal_features, {"method": "stable_robust", **fallback_summary}

    async def _select_stable_tier_1_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tier_1_features: list,
        count: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
        min_features_per_tier: int,
    ) -> list[str]:
        """Select core features with stability selection using bootstrapping."""
        if not tier_1_features:
            return []
        
        # Get available features
        available_features = [f for f in tier_1_features if f in X_val.columns]
        if not available_features:
            return []
        
        # Perform stability selection
        stable_features = await self._perform_stability_selection(
            model, model_name, X_train, y_train, available_features, 
            count, n_bootstrap_samples, stability_threshold, min_features_per_tier
        )
        
        return stable_features

    async def _select_stable_tier_2_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tier_2_features: list,
        count: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
        min_features_per_tier: int,
    ) -> list[str]:
        """Select normalized features with stability selection."""
        if not tier_2_features:
            return []
        
        available_features = [f for f in tier_2_features if f in X_val.columns]
        if not available_features:
            return []
        
        # For normalized features, use stability selection with stability-based criteria
        stable_features = await self._perform_stability_selection(
            model, model_name, X_train, y_train, available_features, 
            count, n_bootstrap_samples, stability_threshold, min_features_per_tier,
            selection_criteria="stability"  # Prefer stable features for normalized tier
        )
        
        return stable_features

    async def _select_stable_tier_3_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tier_3_features: list,
        count: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
        min_features_per_tier: int,
    ) -> list[str]:
        """Select interaction features with stability selection."""
        if not tier_3_features:
            return []
        
        available_features = [f for f in tier_3_features if f in X_val.columns]
        if not available_features:
            return []
        
        # For interaction features, use stability selection with significance-based criteria
        stable_features = await self._perform_stability_selection(
            model, model_name, X_train, y_train, available_features, 
            count, n_bootstrap_samples, stability_threshold, min_features_per_tier,
            selection_criteria="significance"  # Prefer significant interactions
        )
        
        return stable_features

    async def _select_stable_tier_4_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tier_4_features: list,
        count: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
        min_features_per_tier: int,
    ) -> list[str]:
        """Select lagged features with stability selection."""
        if not tier_4_features:
            return []
        
        available_features = [f for f in tier_4_features if f in X_val.columns]
        if not available_features:
            return []
        
        # For lagged features, use stability selection with temporal criteria
        stable_features = await self._perform_stability_selection(
            model, model_name, X_train, y_train, available_features, 
            count, n_bootstrap_samples, stability_threshold, min_features_per_tier,
            selection_criteria="temporal"  # Prefer temporally stable features
        )
        
        return stable_features

    async def _select_stable_tier_5_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tier_5_features: list,
        count: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
        min_features_per_tier: int,
    ) -> list[str]:
        """Select causality features with stability selection."""
        if not tier_5_features:
            return []
        
        available_features = [f for f in tier_5_features if f in X_val.columns]
        if not available_features:
            return []
        
        # For causality features, use stability selection with market logic criteria
        stable_features = await self._perform_stability_selection(
            model, model_name, X_train, y_train, available_features, 
            count, n_bootstrap_samples, stability_threshold, min_features_per_tier,
            selection_criteria="market_logic"  # Prefer market-logic consistent features
        )
        
        return stable_features

    async def _perform_stability_selection(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        available_features: list,
        count: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
        min_features_per_tier: int,
        selection_criteria: str = "importance"
    ) -> list[str]:
        """Perform stability selection using bootstrapping to ensure feature selection stability."""
        
        self.logger.info(f"🔄 Performing stability selection for {len(available_features)} features with {n_bootstrap_samples} bootstrap samples...")
        
        # Initialize feature selection frequency counter
        feature_selection_freq = {feature: 0 for feature in available_features}
        
        # Perform bootstrap sampling and feature selection
        for i in range(n_bootstrap_samples):
            try:
                # Create bootstrap sample (with replacement) from training data only
                bootstrap_indices = np.random.choice(
                    len(X_train), 
                    size=len(X_train), 
                    replace=True
                )
                
                X_bootstrap = X_train.iloc[bootstrap_indices][available_features]
                y_bootstrap = y_train.iloc[bootstrap_indices]
                
                # Perform feature selection on bootstrap sample
                selected_features_bootstrap = await self._select_features_single_bootstrap(
                    model, model_name, X_bootstrap, y_bootstrap, available_features, 
                    count, selection_criteria
                )
                
                # Count selected features
                for feature in selected_features_bootstrap:
                    feature_selection_freq[feature] += 1
                    
            except Exception as e:
                self.logger.warning(f"Bootstrap sample {i+1} failed: {e}")
                continue
        
        # Calculate selection stability for each feature
        feature_stability = {
            feature: freq / n_bootstrap_samples 
            for feature, freq in feature_selection_freq.items()
        }
        
        # Select features that meet stability threshold
        stable_features = [
            feature for feature, stability in feature_stability.items()
            if stability >= stability_threshold
        ]
        
        # Ensure minimum number of features per tier
        if len(stable_features) < min_features_per_tier:
            # Add top features by stability score to meet minimum
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:min_features_per_tier]]
        
        # Limit to requested count
        if len(stable_features) > count:
            # Sort by stability and take top features
            stable_features = sorted(
                stable_features, 
                key=lambda x: feature_stability[x], 
                reverse=True
            )[:count]
        
        self.logger.info(f"   📊 Stability selection results:")
        self.logger.info(f"      Selected: {len(stable_features)} stable features")
        self.logger.info(f"      Average stability: {np.mean([feature_stability[f] for f in stable_features]):.3f}")
        self.logger.info(f"      Min stability: {min([feature_stability[f] for f in stable_features]):.3f}")
        
        return stable_features

    async def _select_features_single_bootstrap(
        self,
        model: Any,
        model_name: str,
        X_bootstrap: pd.DataFrame,
        y_bootstrap: pd.Series,
        available_features: list,
        count: int,
        selection_criteria: str
    ) -> list[str]:
        """Select features for a single bootstrap sample."""
        
        try:
            # Use model-based importance if available
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                feature_importance = model.feature_importances_
                feature_importance_dict = dict(zip(X_bootstrap.columns, feature_importance))
                tier_importance = {f: feature_importance_dict.get(f, 0) for f in available_features}
                selected_features = sorted(tier_importance.items(), key=lambda x: x[1], reverse=True)[:count]
                return [f[0] for f in selected_features]
            else:
                # Use criteria-based selection for non-tree models
                if selection_criteria == "stability":
                    # Select based on feature stability (lower variance for normalized features)
                    feature_variance = X_bootstrap[available_features].var()
                    stable_features = feature_variance.nsmallest(count).index.tolist()
                    return stable_features
                elif selection_criteria == "significance":
                    # Select based on absolute mean (higher values indicate more significant interactions)
                    feature_abs_mean = X_bootstrap[available_features].abs().mean()
                    significant_features = feature_abs_mean.nlargest(count).index.tolist()
                    return significant_features
                elif selection_criteria == "temporal":
                    # Select based on variance (higher variance indicates more temporal information)
                    feature_variance = X_bootstrap[available_features].var()
                    temporal_features = feature_variance.nlargest(count).index.tolist()
                    return temporal_features
                elif selection_criteria == "market_logic":
                    # Select based on absolute mean (causality features should have meaningful values)
                    feature_abs_mean = X_bootstrap[available_features].abs().mean()
                    causality_features = feature_abs_mean.nlargest(count).index.tolist()
                    return causality_features
                else:
                    # Default to variance-based selection
                    feature_variance = X_bootstrap[available_features].var()
                    top_features = feature_variance.nlargest(count).index.tolist()
                    return top_features
        except Exception:
            # Fallback to variance-based selection
            feature_variance = X_bootstrap[available_features].var()
            top_features = feature_variance.nlargest(count).index.tolist()
            return top_features

    async def _apply_stable_final_pruning(
        self,
        selected_features: list,
        X_val_subset: pd.DataFrame,
        y_val: pd.Series,
        max_features: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
    ) -> list[str]:
        """Apply final pruning with stability selection to meet maximum feature count."""
        if len(selected_features) <= max_features:
            return selected_features
        
        # Perform stability selection for final pruning
        stable_features = await self._perform_stability_selection(
            None, "final_pruning", X_val_subset, y_val, selected_features,
            max_features, n_bootstrap_samples, stability_threshold, 5
        )
        
        return stable_features

    async def _try_stable_shap_feature_selection(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list,
        min_features: int,
        max_features: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
    ) -> tuple[list[str], dict]:
        """Attempts stable SHAP-based feature selection with bootstrapping."""
        
        self.logger.info(f"🔍 Performing stable SHAP feature selection with {n_bootstrap_samples} bootstrap samples...")
        
        # Load SHAP analysis configuration
        feature_config = self.config.get("feature_interactions", {})
        stability_config = feature_config.get("stability_selection", {})
        shap_config = stability_config.get("shap_analysis", {})
        
        # Get adaptive sample size based on dataset size
        validation_sample_size = self._get_adaptive_shap_sample_size(
            len(X_val), shap_config
        )
        
        self.logger.info(f"   📊 Using {validation_sample_size} validation samples for SHAP analysis")
        
        # Initialize feature selection frequency counter
        feature_selection_freq = {feature: 0 for feature in feature_names}
        shap_values_all = []
        
        # Perform bootstrap sampling and SHAP analysis
        for i in range(n_bootstrap_samples):
            try:
                # Create bootstrap sample from training data only (prevent look-ahead bias)
                bootstrap_indices = np.random.choice(
                    len(X_train), 
                    size=len(X_train), 
                    replace=True
                )
                
                X_bootstrap = X_train.iloc[bootstrap_indices]
                y_bootstrap = y_train.iloc[bootstrap_indices]
                
                # Sample validation set for SHAP analysis with adaptive size
                sample_idx = np.random.RandomState(42 + i).choice(
                    len(X_val),
                    size=min(validation_sample_size, len(X_val)),  # Adaptive sample size
                    replace=False,
                )
                X_val_sample = X_val.iloc[sample_idx]
                y_val_sample = y_val.iloc[sample_idx]
                
                # Calculate SHAP values for bootstrap sample
                shap_importance = await self._calculate_shap_importance_single_bootstrap(
                    model, model_name, X_bootstrap, y_bootstrap, X_val_sample, y_val_sample
                )
                
                if shap_importance is not None:
                    # Select top features based on SHAP importance
                    top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:max_features]
                    
                    # Count selected features
                    for feature, importance in top_features:
                        feature_selection_freq[feature] += 1
                        shap_values_all.append((feature, importance))
                        
            except Exception as e:
                self.logger.warning(f"SHAP bootstrap sample {i+1} failed: {e}")
                continue
        
        # Calculate selection stability for each feature
        feature_stability = {
            feature: freq / n_bootstrap_samples 
            for feature, freq in feature_selection_freq.items()
        }
        
        # Select features that meet stability threshold
        stable_features = [
            feature for feature, stability in feature_stability.items()
            if stability >= stability_threshold
        ]
        
        # Ensure minimum number of features
        if len(stable_features) < min_features:
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:min_features]]
        
        # Limit to maximum features
        if len(stable_features) > max_features:
            stable_features = sorted(
                stable_features, 
                key=lambda x: feature_stability[x], 
                reverse=True
            )[:max_features]
        
        # Calculate average SHAP importance for selected features
        feature_shap_avg = {}
        for feature in stable_features:
            shap_values = [shap_val for f, shap_val in shap_values_all if f == feature]
            if shap_values:
                feature_shap_avg[feature] = np.mean(shap_values)
        
        self.logger.info(f"   📊 Stable SHAP selection results:")
        self.logger.info(f"      Selected: {len(stable_features)} stable features")
        self.logger.info(f"      Average stability: {np.mean([feature_stability[f] for f in stable_features]):.3f}")
        self.logger.info(f"      Validation samples used: {validation_sample_size}")
        
        return stable_features, {
            "method": "stable_shap",
            "stability_scores": feature_stability,
            "shap_importance": feature_shap_avg,
            "bootstrap_samples": n_bootstrap_samples,
            "stability_threshold": stability_threshold,
            "validation_sample_size": validation_sample_size
        }

    def _get_adaptive_shap_sample_size(self, total_samples: int, shap_config: dict) -> int:
        """Calculate adaptive sample size for SHAP analysis based on dataset size."""
        
        # Get configuration parameters
        default_size = shap_config.get("validation_sample_size", 2000)
        min_size = shap_config.get("min_sample_size", 1000)
        max_size = shap_config.get("max_sample_size", 5000)
        enable_adaptive = shap_config.get("enable_adaptive_sampling", True)
        
        if not enable_adaptive:
            return min(default_size, total_samples)
        
        # Adaptive sizing based on dataset size
        if total_samples <= 10000:
            # Small dataset: use 20% of data, but at least min_size
            sample_size = max(min_size, int(total_samples * 0.2))
        elif total_samples <= 50000:
            # Medium dataset: use 10% of data
            sample_size = int(total_samples * 0.1)
        elif total_samples <= 200000:
            # Large dataset: use 5% of data
            sample_size = int(total_samples * 0.05)
        else:
            # Very large dataset: use 2% of data, but cap at max_size
            sample_size = min(max_size, int(total_samples * 0.02))
        
        # Ensure we don't exceed total samples
        sample_size = min(sample_size, total_samples)
        
        # Ensure we meet minimum requirements
        sample_size = max(min_size, sample_size)
        
        return sample_size

    async def _calculate_shap_importance_single_bootstrap(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, float] | None:
        """Calculate SHAP importance for a single bootstrap sample."""
        
        try:
            if model_name in ["lightgbm", "xgboost", "random_forest"]:
                # Try TreeExplainer with proper import
                try:
                    from shap.explainers import TreeExplainer

                    explainer = TreeExplainer(model)
                    shap_values = explainer.shap_values(X_val)

                    # Normalize SHAP outputs to a 1D importance vector
                    if isinstance(shap_values, list):
                        shap_array = np.asarray(shap_values)
                    else:
                        shap_array = np.asarray(shap_values)

                    # Handle different SHAP output shapes
                    if shap_array.ndim == 2:
                        feature_importance = np.mean(np.abs(shap_array), axis=0)
                    elif shap_array.ndim == 3:
                        feature_importance = np.mean(np.abs(shap_array), axis=(0, 1))
                    else:
                        return None

                    return dict(zip(X_val.columns, feature_importance))

                except (ImportError, AttributeError):
                    # Fallback to permutation importance
                    from sklearn.inspection import permutation_importance
                    feature_importance = permutation_importance(
                        model,
                        X_val,
                        y_val,
                        n_repeats=3,
                        random_state=42,
                    ).importances_mean
                    return dict(zip(X_val.columns, feature_importance))

            elif model_name == "svm":
                # Use KernelExplainer for SVM models
                try:
                    from shap.explainers import KernelExplainer
                    
                    # Use training data as background
                    explainer = KernelExplainer(model.predict, X_train.iloc[:100])  # Sample background
                    shap_values = explainer.shap_values(X_val.iloc[:50])  # Sample validation
                    
                    feature_importance = np.mean(np.abs(shap_values), axis=0)
                    return dict(zip(X_val.columns, feature_importance))
                    
                except Exception:
                    return None

            else:
                # For other models, use permutation importance
                from sklearn.inspection import permutation_importance
                feature_importance = permutation_importance(
                    model,
                    X_val,
                    y_val,
                    n_repeats=3,
                    random_state=42,
                ).importances_mean
                return dict(zip(X_val.columns, feature_importance))

        except Exception:
            return None

    async def _robust_stable_feature_selection(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list,
        min_features: int,
        max_features: int,
        n_bootstrap_samples: int,
        stability_threshold: float,
    ) -> tuple[list[str], dict]:
        """Fallback to robust feature selection methods with stability selection."""
        
        self.logger.info(f"🔄 Performing robust stable feature selection with {n_bootstrap_samples} bootstrap samples...")
        
        # Initialize feature selection frequency counter
        feature_selection_freq = {feature: 0 for feature in feature_names}
        
        # Perform bootstrap sampling and feature selection
        for i in range(n_bootstrap_samples):
            try:
                # Create bootstrap sample from training data only
                bootstrap_indices = np.random.choice(
                    len(X_train), 
                    size=len(X_train), 
                    replace=True
                )
                
                X_bootstrap = X_train.iloc[bootstrap_indices]
                y_bootstrap = y_train.iloc[bootstrap_indices]
                
                # Sample validation set
                sample_idx = np.random.RandomState(42 + i).choice(
                    len(X_val),
                    size=min(500, len(X_val)),
                    replace=False,
                )
                X_val_sample = X_val.iloc[sample_idx]
                y_val_sample = y_val.iloc[sample_idx]
                
                # Perform robust feature selection on bootstrap sample
                selected_features_bootstrap = await self._robust_feature_selection_single_bootstrap(
                    model, model_name, X_bootstrap, y_bootstrap, X_val_sample, y_val_sample, 
                    feature_names, min_features, max_features
                )
                
                # Count selected features
                for feature in selected_features_bootstrap:
                    feature_selection_freq[feature] += 1
                    
            except Exception as e:
                self.logger.warning(f"Robust bootstrap sample {i+1} failed: {e}")
                continue
        
        # Calculate selection stability for each feature
        feature_stability = {
            feature: freq / n_bootstrap_samples 
            for feature, freq in feature_selection_freq.items()
        }
        
        # Select features that meet stability threshold
        stable_features = [
            feature for feature, stability in feature_stability.items()
            if stability >= stability_threshold
        ]
        
        # Ensure minimum number of features
        if len(stable_features) < min_features:
            sorted_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:min_features]]
        
        # Limit to maximum features
        if len(stable_features) > max_features:
            stable_features = sorted(
                stable_features, 
                key=lambda x: feature_stability[x], 
                reverse=True
            )[:max_features]
        
        self.logger.info(f"   📊 Robust stable selection results:")
        self.logger.info(f"      Selected: {len(stable_features)} stable features")
        self.logger.info(f"      Average stability: {np.mean([feature_stability[f] for f in stable_features]):.3f}")
        
        return stable_features, {
            "method": "robust_stable",
            "stability_scores": feature_stability,
            "bootstrap_samples": n_bootstrap_samples,
            "stability_threshold": stability_threshold
        }

    async def _robust_feature_selection_single_bootstrap(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list,
        min_features: int,
        max_features: int,
    ) -> list[str]:
        """Perform robust feature selection for a single bootstrap sample."""
        
        try:
            # Try multiple feature selection methods
            methods = []
            
            # Method 1: Variance-based selection
            try:
                feature_variance = X_val[feature_names].var()
                variance_features = feature_variance.nlargest(max_features).index.tolist()
                methods.append(variance_features)
            except Exception:
                pass
            
            # Method 2: Correlation-based selection
            try:
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(score_func=f_classif, k=max_features)
                selector.fit(X_train[feature_names], y_train)
                correlation_features = [feature_names[i] for i in selector.get_support(indices=True)]
                methods.append(correlation_features)
            except Exception:
                pass
            
            # Method 3: Mutual information-based selection
            try:
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(X_train[feature_names], y_train, random_state=42)
                mi_features = [feature_names[i] for i in np.argsort(mi_scores)[-max_features:]]
                methods.append(mi_features)
            except Exception:
                pass
            
            # Method 4: Model-based importance (if available)
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    feature_importance_dict = dict(zip(X_train.columns, feature_importance))
                    model_features = sorted(
                        feature_importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:max_features]
                    model_features = [f[0] for f in model_features if f[0] in feature_names]
                    methods.append(model_features)
            except Exception:
                pass
            
            # Combine methods using voting
            if methods:
                feature_votes = {}
                for method_features in methods:
                    for feature in method_features:
                        feature_votes[feature] = feature_votes.get(feature, 0) + 1
                
                # Select features with highest votes
                selected_features = sorted(
                    feature_votes.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:max_features]
                return [f[0] for f in selected_features]
            else:
                # Fallback to variance-based selection
                feature_variance = X_val[feature_names].var()
                return feature_variance.nlargest(max_features).index.tolist()
                
        except Exception:
            # Final fallback
            return feature_names[:max_features]

    def _categorize_features_by_tier(self, feature_names: list) -> dict:
        """Categorize features into tiers based on naming patterns."""
        categories = {
            "tier_1": [],  # Core features
            "tier_2": [],  # Normalized features
            "tier_3": [],  # Interaction features
            "tier_4": [],  # Lagged features
            "tier_5": [],  # Causality features
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            # Tier 1: Core technical and liquidity features
            if any(keyword in feature_lower for keyword in [
                "rsi", "macd", "bb", "atr", "adx", "sma", "ema", "cci", "mfi", "roc",
                "volume", "spread", "liquidity", "price_impact", "kyle", "amihud"
            ]):
                categories["tier_1"].append(feature)
            
            # Tier 2: Normalized features
            elif any(keyword in feature_lower for keyword in [
                "_z_score", "_change", "_pct_change", "_acceleration", "_bounded",
                "_log", "_normalized"
            ]):
                categories["tier_2"].append(feature)
            
            # Tier 3: Interaction features
            elif "_x_" in feature_lower or "_div_" in feature_lower:
                categories["tier_3"].append(feature)
            
            # Tier 4: Lagged features
            elif "_lag" in feature_lower:
                categories["tier_4"].append(feature)
            
            # Tier 5: Causality features
            elif any(keyword in feature_lower for keyword in [
                "_predicts_", "_causality", "_divergence", "_stress", "_extreme"
            ]):
                categories["tier_5"].append(feature)
            
            # Default to tier 1 for uncategorized features
            else:
                categories["tier_1"].append(feature)
        
        return categories

    def _save_enhanced_models(
        self,
        enhanced_models: dict,
        data_dir: str,
        training_input: dict,
    ) -> str:
        """Saves the enhanced models and a JSON summary report."""
        enhanced_models_dir = os.path.join(data_dir, "enhanced_analyst_models")
        os.makedirs(enhanced_models_dir, exist_ok=True)

        json_summary = {}

        for regime_name, models in enhanced_models.items():
            regime_models_dir = os.path.join(enhanced_models_dir, regime_name)
            os.makedirs(regime_models_dir, exist_ok=True)
            json_summary[regime_name] = {}

            for model_name, model_data in models.items():
                model_file = os.path.join(regime_models_dir, f"{model_name}.joblib")
                joblib.dump(model_data["model"], model_file)

                summary_data = model_data.copy()
                summary_data.pop("model", None)
                summary_data["model_path"] = model_file
                json_summary[regime_name][model_name] = summary_data

        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        summary_file = os.path.join(
            data_dir,
            f"{exchange}_{symbol}_analyst_enhancement_summary.json",
        )
        with open(summary_file, "w") as f:
            json.dump(json_summary, f, indent=2, default=str)

        return enhanced_models_dir

    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Applies dynamic quantization to a PyTorch model for CPU/MPS inference."""
        self.logger.info("Applying dynamic quantization to the model...")
        # Move model to CPU for quantization, as it's primarily a CPU-based feature set in PyTorch
        model.to("cpu")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        self.logger.info(
            "Dynamic quantization complete. Model is now smaller and may run faster on CPU.",
        )
        return quantized_model

    def _apply_wanda_pruning(
        self,
        model: torch.nn.Module,
        calibration_data: pd.DataFrame,
        sparsity: float = 0.5,
    ) -> torch.nn.Module:
        """
        Applies structured pruning using a simplified WANDA (Weight and Activation-based) method.
        This implementation demonstrates the core concept.
        """
        self.logger.info(f"Applying WANDA-style pruning with {sparsity} sparsity...")
        model.to(self.device)

        # Convert calibration data to tensors
        calib_tensor = torch.tensor(calibration_data.values, dtype=torch.float32).to(
            self.device,
        )

        # 1. Collect activations
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = torch.sqrt(torch.mean(input[0] ** 2, dim=0))

            return hook

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))

        model(calib_tensor)  # Forward pass to trigger hooks
        for hook in hooks:
            hook.remove()

        # 2. Calculate importance and prune
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                W = module.weight.data
                act_norm = activations[name]

                # WANDA Importance Score: |Weight| * ||Activation||
                importance_scores = torch.abs(W) * act_norm

                # Prune the weights with the lowest importance scores
                prune.l1_unstructured(
                    module,
                    name="weight",
                    amount=sparsity,
                    importance_scores=importance_scores,
                )
                # Make pruning permanent
                prune.remove(module, "weight")

        self.logger.info("WANDA-style pruning complete.")
        return model

    def _apply_knowledge_distillation(
        self,
        teacher_model: torch.nn.Module,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> torch.nn.Module:
        """
        Uses knowledge distillation to train a smaller 'student' model to mimic the teacher.
        """
        self.logger.info("Applying knowledge distillation...")
        teacher_model.to(self.device).eval()

        # 1. Define a smaller student model
        input_dim = X_train.shape[1]
        student_model = nn.Sequential(
            nn.Linear(input_dim, 64),  # Smaller hidden layer
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(self.device)

        # 2. Setup training
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)

        # Prepare data
        train_dataset = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Distillation parameters
        T = 2.0  # Temperature for softening probabilities
        alpha = 0.3  # Weight for student's own loss

        # 3. Training loop
        student_model.train()
        for epoch in range(5):  # A short training for demonstration
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Get teacher's logits (outputs before softmax)
                with torch.no_grad():
                    teacher_logits = teacher_model(data)

                # Get student's logits
                student_logits = student_model(data)

                # Calculate losses
                loss_hard = F.cross_entropy(student_logits, targets)  # Standard loss
                loss_soft = nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                ) * (T * T)  # Scaling factor

                # Combine losses
                loss = alpha * loss_hard + (1.0 - alpha) * loss_soft

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.logger.info(f"Distillation Epoch {epoch+1}, Loss: {loss.item():.4f}")

        self.logger.info(
            "Knowledge distillation complete. Returning the trained student model.",
        )
        return student_model.eval()

    def _compute_vif_scores(self, X: pd.DataFrame) -> dict[str, float]:
        """Compute approximate VIF via linear regression R^2 for each feature."""
        from sklearn.linear_model import LinearRegression
        vif_scores: dict[str, float] = {}
        cols = X.columns.tolist()
        # Impute NaNs with median to stabilize
        X_imputed = X.copy()
        for c in cols:
            med = X_imputed[c].median()
            if pd.isna(med):
                med = 0.0
            X_imputed[c] = X_imputed[c].fillna(med)
        for col in cols:
            others = [c for c in cols if c != col]
            if not others:
                vif_scores[col] = 1.0
                continue
            reg = LinearRegression(n_jobs=-1) if hasattr(LinearRegression, "n_jobs") else LinearRegression()
            try:
                reg.fit(X_imputed[others], X_imputed[col])
                y = X_imputed[col].values
                y_pred = reg.predict(X_imputed[others])
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
                vif = float(np.inf) if r2 >= 0.999999 else (1.0 / max(1e-12, 1.0 - r2))
            except Exception:
                vif = 1.0
            vif_scores[col] = vif
        return vif_scores

    def _build_cpa_transform(self, X: pd.DataFrame, cluster_cols: list[str], name_prefix: str) -> tuple[pd.Series, dict]:
        """Create a simple CPA (first principal component) from a cluster of correlated features.
        Returns the component series (aligned to X index) and a transform dict to apply on validation.
        """
        # Standardize using train stats
        means = X[cluster_cols].mean()
        stds = X[cluster_cols].std().replace(0, np.nan)
        Z = (X[cluster_cols] - means) / stds
        Z = Z.fillna(0.0)
        try:
            # SVD on standardized data
            U, S, VT = np.linalg.svd(Z.values, full_matrices=False)
            weights = VT[0, :]
        except Exception:
            # Fallback: equal weights
            k = len(cluster_cols)
            weights = np.ones(k) / max(1, k)
        pc1 = pd.Series(np.dot(Z.values, weights), index=X.index, name=f"{name_prefix}_pc1")
        transform = {
            "name": pc1.name,
            "cols": cluster_cols,
            "means": means.to_dict(),
            "stds": stds.to_dict(),
            "weights": weights.tolist(),
        }
        return pc1.astype(np.float32), transform

    def _apply_cpa_transforms(self, X: pd.DataFrame, transforms: list[dict]) -> pd.DataFrame:
        """Apply stored CPA transforms to dataframe X, drop original cluster cols."""
        X_new = X.copy()
        for t in transforms:
            cols = [c for c in t["cols"] if c in X_new.columns]
            if len(cols) < 1:
                continue
            means = pd.Series(t["means"]).reindex(cols).fillna(0.0)
            stds = pd.Series(t["stds"]).reindex(cols).replace(0, np.nan)
            Z = (X_new[cols] - means) / stds
            Z = Z.fillna(0.0)
            w = np.array(t["weights"])[: len(cols)]
            comp = pd.Series(np.dot(Z.values, w), index=X_new.index, name=t["name"]).astype(np.float32)
            X_new[t["name"]] = comp
            # Drop originals
            X_new = X_new.drop(columns=cols, errors="ignore")
        return X_new

    def _vif_loop_reduce_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Iteratively reduce multicollinearity using VIF thresholds with CPA for very high VIF clusters.
        - If max VIF > 20: build CPA on correlated cluster (|corr|>=0.7), replace originals with PC1
        - Else if max VIF > 10: remove that feature
        Returns reduced X_train, X_val and a summary dict with removed features and CPA clusters.
        """
        removed: list[str] = []
        cpa_transforms: list[dict] = []
        cpa_clusters: list[list[str]] = []
        X_tr = X_train.copy()
        X_vl = X_val.copy()
        # Remove zero-variance columns upfront
        zero_var = [c for c in X_tr.columns if X_tr[c].nunique(dropna=True) <= 1]
        if zero_var:
            removed.extend(zero_var)
            X_tr = X_tr.drop(columns=zero_var, errors="ignore")
            X_vl = X_vl.drop(columns=zero_var, errors="ignore")
        # Main loop
        max_iters = 50
        for _ in range(max_iters):
            if X_tr.shape[1] <= 2:
                break
            vif = self._compute_vif_scores(X_tr)
            if not vif:
                break
            worst_feature, worst_vif = max(vif.items(), key=lambda kv: (float("inf") if np.isinf(kv[1]) else kv[1]))
            if worst_vif is None or (not np.isfinite(worst_vif)):
                # Remove problematic feature
                removed.append(worst_feature)
                X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                X_vl = X_vl.drop(columns=[worst_feature], errors="ignore")
                continue
            if worst_vif > 20.0:
                # Build cluster via correlation with threshold
                try:
                    corr = X_tr.corr().abs()
                    candidates = corr.columns[(corr[worst_feature] >= 0.7)].tolist()
                    # Ensure at least 2
                    cluster = list(dict.fromkeys([worst_feature] + [c for c in candidates if c != worst_feature]))
                    if len(cluster) >= 2:
                        pc1, transform = self._build_cpa_transform(X_tr, cluster, name_prefix=f"cpa_{len(cpa_transforms)+1}")
                        X_tr[pc1.name] = pc1
                        cpa_transforms.append(transform)
                        cpa_clusters.append(cluster)
                        # Drop originals
                        X_tr = X_tr.drop(columns=cluster, errors="ignore")
                        X_vl = X_vl.drop(columns=cluster, errors="ignore")
                        # Apply transform to validation
                        X_vl = self._apply_cpa_transforms(X_vl, [transform])
                        continue
                    else:
                        # Fall back to removal if cluster too small
                        removed.append(worst_feature)
                        X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                        X_vl = X_vl.drop(columns=[worst_feature], errors="ignore")
                        continue
                except Exception:
                    # On failure, remove worst feature
                    removed.append(worst_feature)
                    X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                    X_vl = X_vl.drop(columns=[worst_feature], errors="ignore")
                    continue
            elif worst_vif > 10.0:
                removed.append(worst_feature)
                X_tr = X_tr.drop(columns=[worst_feature], errors="ignore")
                X_vl = X_vl.drop(columns=[worst_feature], errors="ignore")
                continue
            else:
                break
        summary = {
            "removed_features": removed,
            "cpa_clusters": cpa_clusters,
            "cpa_count": len(cpa_clusters),
        }
        # Log once
        if removed:
            self.logger.info(f"VIF reduction removed {len(removed)} features (threshold>10): {removed[:50]}{' ...' if len(removed)>50 else ''}")
        if cpa_clusters:
            self.logger.info(f"CPA created for {len(cpa_clusters)} clusters (threshold>20)")
        return X_tr, X_vl, summary


async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the analyst enhancement step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
        }

        step = AnalystEnhancementStep(config)
        await step.initialize()

        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS" if isinstance(result, dict) else True

    except Exception as e:
        print(failed("Analyst enhancement failed: {e}"))
        return False
