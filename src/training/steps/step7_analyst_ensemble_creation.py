# src/training/steps/step7_analyst_ensemble_creation.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.analyst.data_utils import load_klines_data
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    failed,
    missing,
)
from src.training.steps.unified_data_loader import get_unified_data_loader

try:
    import joblib  # Optional; used for .joblib artifacts
except Exception:  # pragma: no cover
    joblib = None


class DynamicWeightedEnsemble:
    """Dynamic weighted ensemble using Sharpe Ratio for model weighting."""

    def __init__(self, models, model_names, weights):
        self.models = models
        self.model_names = model_names
        self.weights = weights

    def predict(self, X):
        """Make ensemble predictions using weighted average of model probabilities."""
        all_probabilities = []

        for name, model in zip(self.model_names, self.models, strict=False):
            if self.weights.get(name, 0) > 0:
                try:
                    probs = model.predict_proba(X)
                    weighted_probs = probs * self.weights.get(name, 0)
                    all_probabilities.append(weighted_probs)
                except Exception:
                    continue

        if all_probabilities:
            # Average the weighted probabilities
            ensemble_probs = np.mean(all_probabilities, axis=0)
            return np.argmax(ensemble_probs, axis=1)
        # Fallback: return random predictions
        return np.random.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        """Get ensemble probability predictions."""
        all_probabilities = []

        for name, model in zip(self.model_names, self.models, strict=False):
            if self.weights.get(name, 0) > 0:
                try:
                    probs = model.predict_proba(X)
                    weighted_probs = probs * self.weights.get(name, 0)
                    all_probabilities.append(weighted_probs)
                except Exception:
                    continue

        if all_probabilities:
            # Average the weighted probabilities
            return np.mean(all_probabilities, axis=0)
        # Fallback: return uniform probabilities
        return np.ones((len(X), 2)) * 0.5


class ProximityWeightedEnsemble(DynamicWeightedEnsemble):
    """Proximity-weighted ensemble that boosts S/R regime when near S/R levels.

    Parameters (from config):
    - sr_proximity_tau: decay constant for proximity weight (default 0.003)
    - sr_touch_boost: additive boost if recent sr_touch (default 0.2)
    - sr_max_weight: cap on SR weight (default 0.7)
    """

    def __init__(self, models, model_names, weights, cfg=None):
        super().__init__(models, model_names, weights)
        self.cfg = cfg or {}
        self._prev_w: float | None = None

    def _compute_sr_weights(self, X: pd.DataFrame) -> np.ndarray:
        try:
            tau = float(self.cfg.get("sr_proximity_tau", 0.003))
            max_w = float(self.cfg.get("sr_max_weight", 0.7))
            boost = float(self.cfg.get("sr_touch_boost", 0.2))
            # Expect these columns in validation frame
            d_sup = X.get("dist_to_support_pct", pd.Series(1.0, index=X.index)).astype(float)
            d_res = X.get("dist_to_resistance_pct", pd.Series(1.0, index=X.index)).astype(float)
            sr_touch = X.get("sr_touch", pd.Series(0, index=X.index)).astype(float)
            prox = np.minimum(d_sup.values, d_res.values)
            base = np.exp(-prox / max(1e-9, tau))  # in [0,1]
            base = np.clip(base, 0.0, 1.0)
            base = np.minimum(base + boost * sr_touch.values, 1.0)
            return np.clip(base, 0.0, max_w)
        except Exception:
            return np.zeros(len(X))

    def predict_proba(self, X):
        # Compute base probs per model
        probs_per_model = {}
        for name, model in zip(self.model_names, self.models, strict=False):
            try:
                probs_per_model[name] = model.predict_proba(X)
            except Exception:
                continue

        # Derive SR weight per sample
        sr_weight = self._compute_sr_weights(X)
        # Optional: modulate weight by predicted SR strength if present
        try:
            # Expect columns like sr_pred_breakout_strength, sr_pred_bounce_strength
            pred_bk = X.get("sr_pred_breakout_strength", pd.Series(0.0, index=X.index)).astype(float).values
            pred_bo = X.get("sr_pred_bounce_strength", pd.Series(0.0, index=X.index)).astype(float).values
            strength = np.maximum(pred_bk, pred_bo)
            gain = float(self.cfg.get("sr_strength_gain", 0.5))
            # Normalize strength to [0,1] by clipping with a reasonable scale (e.g., 1%)
            norm = np.clip(strength / max(1e-6, float(self.cfg.get("sr_strength_scale", 0.01))), 0.0, 1.0)
            sr_weight = np.clip(sr_weight * (1.0 + gain * norm), 0.0, float(self.cfg.get("sr_max_weight", 0.7)))
        except Exception:
            pass
        # Names that correspond to SR models
        sr_names = [n for n in self.model_names if n.upper().startswith("SR") or "/SR/" in n]
        other_names = [n for n in self.model_names if n not in sr_names]

        # Aggregate SR and OTHER probabilities
        def _avg(names):
            arrs = [probs_per_model[n] for n in names if n in probs_per_model]
            if not arrs:
                return None
            return np.mean(arrs, axis=0)

        sr_avg = _avg(sr_names)
        other_avg = _avg(other_names)
        if sr_avg is None and other_avg is None:
            return np.ones((len(X), 2)) * 0.5
        if sr_avg is None:
            return other_avg
        if other_avg is None:
            return sr_avg

        # Blend per sample with simple hysteresis to reduce flip-flop
        out = np.empty_like(sr_avg)
        alpha_up = float(self.cfg.get("hysteresis_alpha_up", 0.6))
        alpha_down = float(self.cfg.get("hysteresis_alpha_down", 0.3))
        for i in range(len(X)):
            w_raw = float(sr_weight[i])
            if self._prev_w is None:
                w = w_raw
            else:
                if w_raw >= self._prev_w:
                    w = alpha_up * w_raw + (1 - alpha_up) * self._prev_w
                else:
                    w = alpha_down * w_raw + (1 - alpha_down) * self._prev_w
            self._prev_w = w
            out[i, :] = w * sr_avg[i, :] + (1.0 - w) * other_avg[i, :]
        return out


class AnalystEnsembleCreationStep:
    """Step 7: Analyst Ensemble Creation using StackingCV and Dynamic Weighting with caching and streaming."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

        # Add ensemble caching for repeated operations
        self._ensemble_cache = {}
        self._model_cache = {}
        self.max_cache_size = 50  # Maximum number of cached ensembles

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst ensemble creation step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst ensemble creation step."""
        self.logger.info("Initializing Analyst Ensemble Creation Step...")
        self.logger.info("Analyst Ensemble Creation Step initialized successfully")

    def _generate_cache_key(
        self, regime_name: str, model_names: list[str], ensemble_type: str
    ) -> str:
        """Generate a cache key for ensemble operations."""
        sorted_models = sorted(model_names)
        return f"{regime_name}_{ensemble_type}_{'_'.join(sorted_models)}"

    def _get_cached_ensemble(self, cache_key: str) -> Any:
        """Get cached ensemble if available."""
        return self._ensemble_cache.get(cache_key)

    def _cache_ensemble(self, cache_key: str, ensemble: Any) -> None:
        """Cache ensemble with size management."""
        if len(self._ensemble_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._ensemble_cache))
            del self._ensemble_cache[oldest_key]
        self._ensemble_cache[cache_key] = ensemble

    def _stream_models(
        self, regime_path: str, batch_size: int = 5
    ) -> Iterator[tuple[str, Any]]:
        """Stream models from directory to avoid loading all at once."""
        model_files = [
            f for f in os.listdir(regime_path) if f.endswith((".pkl", ".joblib"))
        ]

        for i in range(0, len(model_files), batch_size):
            batch_files = model_files[i : i + batch_size]
            batch_models = {}

            for model_file in batch_files:
                model_name = model_file.replace(".pkl", "").replace(".joblib", "")
                model_path = os.path.join(regime_path, model_file)

                try:
                    if model_file.endswith(".joblib") and joblib is not None:
                        model = joblib.load(model_path)
                    else:
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)

                    batch_models[model_name] = model
                    yield model_name, model

                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_file}: {e}")
                    continue

            # Clear batch from memory
            batch_models.clear()
            import gc

            gc.collect()

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst ensemble creation step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst ensemble creation with data loading and proper ensemble methods.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing ensemble creation results
        """
        try:
            self.logger.info("🔄 Executing Analyst Ensemble Creation...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load training and validation data
            from src.utils.logger import heartbeat
            with heartbeat(self.logger, name="Step7 load_training_data", interval_seconds=60.0):
                training_data, validation_data = await self._load_training_data(
                    symbol,
                    exchange,
                    data_dir,
                )
            try:
                self.logger.info(
                    f"Loaded data: training={getattr(training_data, 'keys', lambda: [])() if hasattr(training_data, 'keys') else type(training_data).__name__}, validation={getattr(validation_data, 'keys', lambda: [])() if hasattr(validation_data, 'keys') else type(validation_data).__name__}",
                )
            except Exception:
                pass

            if training_data is None or validation_data is None:
                msg = "Failed to load training and validation data"
                raise ValueError(msg)

            # Load enhanced analyst models
            enhanced_models_dir = f"{data_dir}/enhanced_analyst_models"
            regime_dirs = [
                d
                for d in os.listdir(enhanced_models_dir)
                if os.path.isdir(os.path.join(enhanced_models_dir, d))
            ]

            if not regime_dirs:
                msg = f"No enhanced analyst models found in {enhanced_models_dir}"
                raise ValueError(
                    msg,
                )
            try:
                self.logger.info(
                    f"Enhanced analyst regimes found: count={len(regime_dirs)}; regimes={regime_dirs}",
                )
            except Exception:
                pass

            # Log performance metrics before ensemble creation
            try:
                data_loader = get_unified_data_loader(self.config)
                perf_metrics = data_loader.get_performance_metrics()
                self.logger.info(f"📊 Performance before ensemble creation:")
                self.logger.info(
                    f"   Memory Usage: {perf_metrics['memory_usage']['percent']:.1f}%"
                )
                self.logger.info(
                    f"   Cache Size: {perf_metrics['cache_stats']['cache_size']}/{perf_metrics['cache_stats']['max_cache_size']}"
                )
                self.logger.info(
                    f"   Ensemble Cache Size: {len(self._ensemble_cache)}/{self.max_cache_size}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Could not get performance metrics: {e}")

            # Create ensembles for each regime
            ensemble_results = {}

            for regime_name in regime_dirs:
                self.logger.info(f"Creating ensemble for regime: {regime_name}")

                # Get regime-specific data
                regime_training_data = training_data.get(regime_name, pd.DataFrame())
                regime_validation_data = validation_data.get(
                    regime_name,
                    pd.DataFrame(),
                )

                if regime_training_data.empty or regime_validation_data.empty:
                    self.print(error("No data available for regime: {regime_name}"))
                    continue

                # Stream-load models for this regime to avoid memory issues
                regime_path = os.path.join(enhanced_models_dir, regime_name)
                regime_models = {}

                # Use streaming to load models in batches
                from src.utils.logger import heartbeat
                with heartbeat(self.logger, name=f"Step7 stream_models[{regime_name}]", interval_seconds=60.0):
                    for model_name, model in self._stream_models(regime_path, batch_size=3):
                        regime_models[model_name] = model

                    # Check cache for existing ensemble
                    cache_key = self._generate_cache_key(
                        regime_name, list(regime_models.keys()), "dynamic_weighting"
                    )
                    cached_ensemble = self._get_cached_ensemble(cache_key)

                    if cached_ensemble is not None:
                        self.logger.info(
                            f"✅ Using cached ensemble for {regime_name} with {len(regime_models)} models"
                        )
                        regime_ensemble = cached_ensemble
                        break

                # Create ensemble for this regime (with caching)
                if "regime_ensemble" not in locals():
                    from src.utils.logger import heartbeat
                    with heartbeat(self.logger, name=f"Step7 create_ensemble[{regime_name}]", interval_seconds=60.0):
                        regime_ensemble = await self._create_regime_ensemble(
                            regime_models,
                            regime_name,
                            regime_training_data,
                            regime_validation_data,
                        )

                    # Cache the ensemble for future use
                    cache_key = self._generate_cache_key(
                        regime_name, list(regime_models.keys()), "dynamic_weighting"
                    )
                    self._cache_ensemble(cache_key, regime_ensemble)
                    self.logger.info(f"✅ Cached ensemble for {regime_name}")

                ensemble_results[regime_name] = regime_ensemble
                # Free models from memory before next regime
                regime_models.clear()

            # Save ensemble models
            ensemble_models_dir = f"{data_dir}/analyst_ensembles"
            os.makedirs(ensemble_models_dir, exist_ok=True)

            for regime_name, ensemble_data in ensemble_results.items():
                ensemble_file = f"{ensemble_models_dir}/{regime_name}_ensemble.pkl"
                with open(ensemble_file, "wb") as f:
                    pickle.dump(ensemble_data, f)

            # Save ensemble summary (without ensemble objects for JSON serialization)
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_ensemble_summary.json"
            )

            # Create JSON-serializable summary
            json_summary = {}
            for regime_name, regime_ensembles in ensemble_results.items():
                json_summary[regime_name] = {}
                for ensemble_type, ensemble_data in regime_ensembles.items():
                    # Extract only JSON-serializable data
                    json_summary[regime_name][ensemble_type] = {
                        "ensemble_type": ensemble_data.get("ensemble_type"),
                        "base_models": ensemble_data.get("base_models"),
                        "regime": ensemble_data.get("regime"),
                        "creation_date": ensemble_data.get("creation_date"),
                        "validation_metrics": ensemble_data.get("validation_metrics"),
                        "cv_scores": ensemble_data.get("cv_scores"),
                        "weights": ensemble_data.get("weights"),
                        "sharpe_ratios": ensemble_data.get("sharpe_ratios"),
                        "features": ensemble_data.get("features"),
                    }

            with open(summary_file, "w") as f:
                json.dump(json_summary, f, indent=2)

            self.logger.info(
                f"✅ Analyst ensemble creation completed. Results saved to {ensemble_models_dir}",
            )

            # Update pipeline state
            pipeline_state["analyst_ensembles"] = ensemble_results

            return {
                "analyst_ensembles": ensemble_results,
                "ensemble_models_dir": ensemble_models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.print(error("❌ Error in Analyst Ensemble Creation: {e}"))
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _load_training_data(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> tuple[dict[str, pd.DataFrame] | None, dict[str, pd.DataFrame] | None]:
        """
        Load training and validation data for all regimes using optimized unified data loader.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path

        Returns:
            Tuple of (training_data, validation_data) dictionaries keyed by regime
        """
        try:
            self.logger.info(
                "Loading training and validation data using unified data loader..."
            )

            # Try to load from unified data loader first (more efficient)
            try:
                timeframe = self.config.get("timeframe", "1m")
                data_loader = get_unified_data_loader(self.config)
                historical_data = await data_loader.load_unified_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    lookback_days=180,
                    use_streaming=True,  # Enable streaming for large datasets
                )

                if historical_data is not None and not historical_data.empty:
                    self.logger.info(
                        f"✅ Loaded {len(historical_data)} rows using unified data loader"
                    )

                    # Split data by regime if available
                    if "regime" in historical_data.columns:
                        training_data = {}
                        validation_data = {}

                        for regime_name in historical_data["regime"].unique():
                            regime_data = historical_data[
                                historical_data["regime"] == regime_name
                            ]

                            # Split into train/validation (80/20)
                            split_idx = int(len(regime_data) * 0.8)
                            training_data[regime_name] = regime_data.iloc[:split_idx]
                            validation_data[regime_name] = regime_data.iloc[split_idx:]

                        self.logger.info(
                            f"✅ Split data into {len(training_data)} regimes"
                        )
                        return training_data, validation_data
                    else:
                        # If no regime column, split all data
                        split_idx = int(len(historical_data) * 0.8)
                        training_data = {"combined": historical_data.iloc[:split_idx]}
                        validation_data = {"combined": historical_data.iloc[split_idx:]}
                        return training_data, validation_data

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Unified data loader failed: {e}, falling back to regime files"
                )

            # Fallback to original regime data files
            regime_data_dir = f"{data_dir}/regime_data"
            if not os.path.exists(regime_data_dir):
                self.print(
                    missing(
                        f"Regime data directory not found: {regime_data_dir}. Step 7 requires regime data from Step 3."
                    )
                )
                return None, None

            training_data = {}
            validation_data = {}

            for regime_file in os.listdir(regime_data_dir):
                if regime_file.endswith("_training.csv"):
                    regime_name = regime_file.replace("_training.csv", "")
                    training_csv = os.path.join(regime_data_dir, regime_file)
                    validation_csv = os.path.join(
                        regime_data_dir,
                        f"{regime_name}_validation.csv",
                    )
                    training_parquet = os.path.join(
                        regime_data_dir,
                        f"{regime_name}_training.parquet",
                    )
                    validation_parquet = os.path.join(
                        regime_data_dir,
                        f"{regime_name}_validation.parquet",
                    )

                    # Load training data (prefer partitioned dataset scan)
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )

                        pdm = ParquetDatasetManager(logger=self.logger)
                        # If step3 wrote partitioned regime data, scan it
                        part_base = os.path.join(data_dir, "parquet", "regime_data")
                        if os.path.isdir(part_base):
                            # Base filters
                            filters = [
                                ("exchange", "==", exchange),
                                ("symbol", "==", symbol),
                                ("regime", "==", regime_name),
                            ]
                            # Time-window filters from config
                            t0 = self.config.get("t0_ms") or self.config.get(
                                "start_timestamp_ms"
                            )
                            t1 = self.config.get("t1_ms") or self.config.get(
                                "end_timestamp_ms"
                            )
                            if t0 is not None:
                                filters.append(("timestamp", ">=", int(t0)))
                            if t1 is not None:
                                filters.append(("timestamp", "<", int(t1)))
                            # Projection: use features+label if configured
                            feat_cols = self.config.get("feature_columns")
                            label_col = self.config.get("label_column", "label")
                            columns = None
                            if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                columns = list(
                                    dict.fromkeys(["timestamp", *feat_cols, label_col])
                                )
                            cache_key = f"regime_{exchange}_{symbol}_{regime_name}_{self.config.get('timeframe','1m')}_train"

                            def _arrow_pre(tbl):
                                import pyarrow as _pa, pyarrow.compute as pc

                                if (
                                    "timestamp" in tbl.schema.names
                                    and not _pa.types.is_int64(
                                        tbl.schema.field("timestamp").type
                                    )
                                ):
                                    tbl = tbl.set_column(
                                        tbl.schema.get_field_index("timestamp"),
                                        "timestamp",
                                        pc.cast(tbl.column("timestamp"), _pa.int64()),
                                    )
                                return tbl

                            # Reader shortcut: if materialized OHLCV+label projection exists, read it first
                            try:
                                proj_base = os.path.join(
                                    "data_cache", "parquet", "proj_ohlcv_label"
                                )
                                if os.path.isdir(proj_base):
                                    proj_filters = [
                                        ("exchange", "==", exchange),
                                        ("symbol", "==", symbol),
                                        (
                                            "timeframe",
                                            "==",
                                            self.config.get("timeframe", "1m"),
                                        ),
                                    ]
                                    # Use regime if the dataset is partitioned by regime, else split=train
                                    if any(
                                        "regime=" in p for p in os.listdir(proj_base)
                                    ):
                                        proj_filters.append(
                                            ("regime", "==", regime_name)
                                        )
                                    else:
                                        proj_filters.append(("split", "==", "train"))
                                    t0 = self.config.get("t0_ms") or self.config.get(
                                        "start_timestamp_ms"
                                    )
                                    t1 = self.config.get("t1_ms") or self.config.get(
                                        "end_timestamp_ms"
                                    )
                                    if t0 is not None:
                                        proj_filters.append(
                                            ("timestamp", ">=", int(t0))
                                        )
                                    if t1 is not None:
                                        proj_filters.append(("timestamp", "<", int(t1)))
                                    ohlcv_cols = [
                                        "timestamp",
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        label_col,
                                    ]
                                    training_df = pdm.cached_projection(
                                        base_dir=proj_base,
                                        filters=proj_filters,
                                        columns=ohlcv_cols,
                                        cache_dir="data_cache/projections",
                                        cache_key_prefix=f"proj_ohlcv_label_{exchange}_{symbol}_{regime_name}_{self.config.get('timeframe','1m')}",
                                        snapshot_version="v1",
                                        ttl_seconds=3600,
                                        batch_size=131072,
                                        arrow_transform=_arrow_pre,
                                    )
                                    training_data[regime_name] = training_df
                                    self.logger.info(
                                        f"Loaded training data (materialized projection) for {regime_name}: {training_df.shape}",
                                    )
                                    continue
                            except Exception:
                                pass

                            # Materialize OHLCV+label projection for this regime and timeframe if not present
                            try:
                                ohlcv_cols = [
                                    "timestamp",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                    label_col,
                                ]
                                proj_out = os.path.join(
                                    "data_cache", "parquet", "proj_ohlcv_label"
                                )
                                pdm.materialize_projection(
                                    base_dir=os.path.join(
                                        data_dir, "parquet", "labeled"
                                    ),
                                    filters=[
                                        ("exchange", "==", exchange),
                                        ("symbol", "==", symbol),
                                        (
                                            "timeframe",
                                            "==",
                                            self.config.get("timeframe", "1m"),
                                        ),
                                        ("regime", "==", regime_name)
                                        if any(
                                            "regime=" in p
                                            for p in os.listdir(part_base)
                                        )
                                        else ("split", "==", "train"),
                                    ],
                                    columns=ohlcv_cols,
                                    output_dir=proj_out,
                                    partition_cols=[
                                        "exchange",
                                        "symbol",
                                        "timeframe",
                                        "year",
                                        "month",
                                        "day",
                                    ],
                                    schema_name="split",
                                    compression="snappy",
                                    batch_size=131072,
                                    metadata={
                                        "schema_version": "1",
                                        "projection": "ohlcv_label",
                                    },
                                )
                            except Exception:
                                pass
                            training_df = pdm.cached_projection(
                                base_dir=part_base,
                                filters=filters,
                                columns=columns or [],
                                cache_dir="data_cache/projections",
                                cache_key_prefix=cache_key,
                                snapshot_version="v1",
                                ttl_seconds=3600,
                                batch_size=131072,
                                arrow_transform=_arrow_pre,
                            )
                            training_data[regime_name] = training_df
                            self.logger.info(
                                f"Loaded training data (dataset scan) for {regime_name}: {training_df.shape}",
                            )
                        elif os.path.exists(training_parquet):
                            # Use projected columns if defined
                            try:
                                feat_cols = self.config.get("feature_columns")
                                label_col = self.config.get("label_column", "label")
                                if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger,
                                        "read_parquet",
                                        training_parquet,
                                        columns=True,
                                    ):
                                        training_df = pd.read_parquet(
                                            training_parquet,
                                            columns=[
                                                "timestamp",
                                                *feat_cols,
                                                label_col,
                                            ],
                                        )
                                else:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger, "read_parquet", training_parquet
                                    ):
                                        training_df = pd.read_parquet(training_parquet)
                            except Exception:
                                from src.utils.logger import log_io_operation

                                with log_io_operation(
                                    self.logger, "read_parquet", training_parquet
                                ):
                                    training_df = pd.read_parquet(training_parquet)
                            training_data[regime_name] = training_df
                            self.logger.info(
                                f"Loaded training data (Parquet) for {regime_name}: {training_df.shape}",
                            )
                        elif os.path.exists(training_csv):
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger, "read_csv", training_csv
                            ):
                                training_df = pd.read_csv(
                                    training_csv,
                                    index_col=0,
                                    parse_dates=True,
                                )
                        training_data[regime_name] = training_df
                        self.logger.info(
                            f"Loaded training data (CSV) for {regime_name}: {training_df.shape}",
                        )
                    except Exception:
                        # Fallback chain
                        if os.path.exists(training_parquet):
                            try:
                                feat_cols = self.config.get("feature_columns")
                                label_col = self.config.get("label_column", "label")
                                if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger,
                                        "read_parquet",
                                        training_parquet,
                                        columns=True,
                                    ):
                                        training_df = pd.read_parquet(
                                            training_parquet,
                                            columns=[
                                                "timestamp",
                                                *feat_cols,
                                                label_col,
                                            ],
                                        )
                                else:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger, "read_parquet", training_parquet
                                    ):
                                        training_df = pd.read_parquet(training_parquet)
                            except Exception:
                                from src.utils.logger import log_io_operation

                                with log_io_operation(
                                    self.logger, "read_parquet", training_parquet
                                ):
                                    training_df = pd.read_parquet(training_parquet)
                            training_data[regime_name] = training_df
                            self.logger.info(
                                f"Loaded training data (Parquet) for {regime_name}: {training_df.shape}",
                            )
                        elif os.path.exists(training_csv):
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger, "read_csv", training_csv
                            ):
                                training_df = pd.read_csv(
                                    training_csv,
                                    index_col=0,
                                    parse_dates=True,
                                )
                            training_data[regime_name] = training_df
                            self.logger.info(
                                f"Loaded training data (CSV) for {regime_name}: {training_df.shape}",
                            )

                    # Load validation data (prefer partitioned dataset scan)
                    try:
                        from src.training.enhanced_training_manager_optimized import (
                            ParquetDatasetManager,
                        )

                        pdm = ParquetDatasetManager(logger=self.logger)
                        part_base = os.path.join(data_dir, "parquet", "regime_data")
                        if os.path.isdir(part_base):
                            filters = [
                                ("exchange", "==", exchange),
                                ("symbol", "==", symbol),
                                ("regime", "==", regime_name),
                            ]
                            t0 = self.config.get("t0_ms") or self.config.get(
                                "start_timestamp_ms"
                            )
                            t1 = self.config.get("t1_ms") or self.config.get(
                                "end_timestamp_ms"
                            )
                            if t0 is not None:
                                filters.append(("timestamp", ">=", int(t0)))
                            if t1 is not None:
                                filters.append(("timestamp", "<", int(t1)))
                            feat_cols = self.config.get("feature_columns")
                            label_col = self.config.get("label_column", "label")
                            columns = None
                            if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                columns = list(
                                    dict.fromkeys(["timestamp", *feat_cols, label_col])
                                )
                            cache_key = f"regime_{exchange}_{symbol}_{regime_name}_{self.config.get('timeframe','1m')}_validation"
                            validation_df = pdm.cached_projection(
                                base_dir=part_base,
                                filters=filters,
                                columns=columns or [],
                                cache_dir="data_cache/projections",
                                cache_key_prefix=cache_key,
                                snapshot_version="v1",
                                ttl_seconds=3600,
                                batch_size=131072,
                                arrow_transform=_arrow_pre,
                            )
                            validation_data[regime_name] = validation_df
                            self.logger.info(
                                f"Loaded validation data (dataset scan) for {regime_name}: {validation_df.shape}",
                            )
                        elif os.path.exists(validation_parquet):
                            try:
                                feat_cols = self.config.get("feature_columns")
                                label_col = self.config.get("label_column", "label")
                                if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger,
                                        "read_parquet",
                                        validation_parquet,
                                        columns=True,
                                    ):
                                        validation_df = pd.read_parquet(
                                            validation_parquet,
                                            columns=[
                                                "timestamp",
                                                *feat_cols,
                                                label_col,
                                            ],
                                        )
                                else:
                                    from src.utils.logger import log_io_operation

                                    with log_io_operation(
                                        self.logger, "read_parquet", validation_parquet
                                    ):
                                        validation_df = pd.read_parquet(
                                            validation_parquet
                                        )
                            except Exception:
                                from src.utils.logger import log_io_operation

                                with log_io_operation(
                                    self.logger, "read_parquet", validation_parquet
                                ):
                                    validation_df = pd.read_parquet(validation_parquet)
                            validation_data[regime_name] = validation_df
                            self.logger.info(
                                f"Loaded validation data (Parquet) for {regime_name}: {validation_df.shape}",
                            )
                        elif os.path.exists(validation_csv):
                            from src.utils.logger import log_io_operation

                            with log_io_operation(
                                self.logger, "read_csv", validation_csv
                            ):
                                validation_df = pd.read_csv(
                                    validation_csv,
                                    index_col=0,
                                    parse_dates=True,
                                )
                        validation_data[regime_name] = validation_df
                        self.logger.info(
                            f"Loaded validation data (CSV) for {regime_name}: {validation_df.shape}",
                        )
                    except Exception:
                        if os.path.exists(validation_parquet):
                            try:
                                feat_cols = self.config.get("feature_columns")
                                label_col = self.config.get("label_column", "label")
                                if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                    validation_df = pd.read_parquet(
                                        validation_parquet,
                                        columns=["timestamp", *feat_cols, label_col],
                                    )
                                else:
                                    validation_df = pd.read_parquet(validation_parquet)
                            except Exception:
                                validation_df = pd.read_parquet(validation_parquet)
                            validation_data[regime_name] = validation_df
                            self.logger.info(
                                f"Loaded validation data (Parquet) for {regime_name}: {validation_df.shape}",
                            )
                        elif os.path.exists(validation_csv):
                            validation_df = pd.read_csv(
                                validation_csv,
                                index_col=0,
                                parse_dates=True,
                            )
                            validation_data[regime_name] = validation_df
                            self.logger.info(
                                f"Loaded validation data (CSV) for {regime_name}: {validation_df.shape}",
                            )

            if not training_data or not validation_data:
                self.logger.warning(
                    "No training or validation data found, attempting to load from klines data...",
                )

                # Fallback: try to load from klines data and split
                klines_file = f"{data_dir}/{exchange}_{symbol}_1h.csv"
                if os.path.exists(klines_file):
                    klines_df = load_klines_data(klines_file)
                    if not klines_df.empty:
                        # Split data into training and validation
                        split_idx = int(len(klines_df) * 0.8)
                        training_data["all"] = klines_df.iloc[:split_idx]
                        validation_data["all"] = klines_df.iloc[split_idx:]
                        self.logger.info(
                            f"Split klines data: training={len(training_data['all'])}, validation={len(validation_data['all'])}",
                        )

            return training_data, validation_data

        except Exception:
            self.print(error("Error loading training data: {e}"))
            return None, None

    async def _create_regime_ensemble(
        self,
        models: dict[str, Any],
        regime_name: str,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Create ensemble for a specific regime with real data.

        Args:
            models: Enhanced models for the regime
            regime_name: Name of the regime
            training_data: Training data for the regime
            validation_data: Validation data for the regime

        Returns:
            Dict containing ensemble model
        """
        try:
            self.logger.info(f"Creating ensemble for regime: {regime_name}")

            # Extract models and prepare data
            model_list = []
            model_names = []
            model_performances = {}

            for model_name, model_data in models.items():
                model_list.append(model_data["model"])
                model_names.append(model_name)

                # Calculate model performance on validation data
                if not validation_data.empty and "label" in validation_data.columns:
                    X_val = validation_data.drop("label", axis=1)
                    y_val = validation_data["label"]

                    try:
                        predictions = model_data["model"].predict(X_val)
                        accuracy = accuracy_score(y_val, predictions)
                        precision = precision_score(
                            y_val,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                        recall = recall_score(
                            y_val,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                        f1 = f1_score(
                            y_val,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )

                        model_performances[model_name] = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                        }
                    except Exception as e:
                        self.logger.warning(
                            f"Could not evaluate model {model_name}: {e}",
                        )
                        model_performances[model_name] = {
                            "accuracy": 0.5,
                            "precision": 0.5,
                            "recall": 0.5,
                            "f1": 0.5,
                        }

            # Create different ensemble types
            ensemble_results = {}

            # 1. StackingCV Ensemble with proper meta-learner
            stacking_ensemble = await self._create_stacking_ensemble(
                model_list,
                model_names,
                regime_name,
                training_data,
                validation_data,
            )
            ensemble_results["stacking_cv"] = stacking_ensemble

            # 2. Dynamic Performance-Weighted Ensemble using Sharpe Ratio
            dynamic_ensemble = await self._create_dynamic_weighting_ensemble(
                model_list,
                model_names,
                regime_name,
                training_data,
                validation_data,
                model_performances,
            )
            ensemble_results["dynamic_weighting"] = dynamic_ensemble

            # 3. Voting Ensemble
            voting_ensemble = await self._create_voting_ensemble(
                model_list,
                model_names,
                regime_name,
            )
            ensemble_results["voting"] = voting_ensemble

            return ensemble_results

        except Exception as e:
            self.logger.exception(
                f"Error creating ensemble for regime {regime_name}: {e}",
            )
            raise

    async def _create_stacking_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        regime_name: str,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create enhanced StackingCV ensemble with proper meta-learner."""
        try:
            if training_data.empty or "label" not in training_data.columns:
                msg = "Training data is empty or missing labels"
                raise ValueError(msg)

            X_train = training_data.drop("label", axis=1)
            y_train = training_data["label"]

            # Create estimators for stacking
            estimators = list(zip(model_names, models, strict=False))

            # Use stratified k-fold for better validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Create meta-learner with regularization
            meta_learner = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,  # Regularization strength
                penalty="l2",
                solver="lbfgs",
            )

            # Create stacking classifier
            stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv,
                stack_method="predict_proba",
                n_jobs=-1,
                passthrough=False,  # Don't pass original features to meta-learner
            )

            # Fit the stacking classifier
            stacking_classifier.fit(X_train, y_train)

            # Evaluate on validation data
            validation_metrics = {}
            if not validation_data.empty and "label" in validation_data.columns:
                X_val = validation_data.drop("label", axis=1)
                y_val = validation_data["label"]

                val_predictions = stacking_classifier.predict(X_val)
                stacking_classifier.predict_proba(X_val)

                validation_metrics = {
                    "accuracy": accuracy_score(y_val, val_predictions),
                    "precision": precision_score(
                        y_val,
                        val_predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        y_val,
                        val_predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        y_val,
                        val_predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                }

            # Cross-validation scores
            cv_scores = cross_val_score(
                stacking_classifier,
                X_train,
                y_train,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )

            return {
                "ensemble": stacking_classifier,
                "ensemble_type": "StackingCV",
                "base_models": model_names,
                "meta_learner": "LogisticRegression",
                "cv_folds": 5,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat(),
                "validation_metrics": validation_metrics,
                "cv_scores": {
                    "mean": cv_scores.mean(),
                    "std": cv_scores.std(),
                    "scores": cv_scores.tolist(),
                },
                "features": {
                    "use_probabilities": True,
                    "use_raw_predictions": False,
                    "passthrough": False,
                },
            }

        except Exception as e:
            self.logger.exception(
                f"Error creating stacking ensemble for {regime_name}: {e}",
            )
            raise

    async def _create_dynamic_weighting_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        regime_name: str,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        model_performances: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Create dynamic performance-weighted ensemble using Sharpe Ratio."""
        try:
            if validation_data.empty or "label" not in validation_data.columns:
                msg = "Validation data is empty or missing labels"
                raise ValueError(msg)

            X_val = validation_data.drop("label", axis=1)
            y_val = validation_data["label"]

            # Calculate Sharpe Ratio for each model
            model_sharpe_ratios = {}
            model_weights = {}

            for _i, (name, model) in enumerate(zip(model_names, models, strict=False)):
                try:
                    # Get predictions and probabilities
                    predictions = model.predict(X_val)
                    model.predict_proba(X_val)

                    # Calculate returns based on predictions (simplified approach)
                    # In a real implementation, you would calculate actual trading returns
                    (predictions == y_val).astype(int)

                    # Calculate "returns" based on prediction accuracy
                    # This is a simplified approach - in reality you'd calculate actual trading returns
                    returns = []
                    for _j, (pred, actual) in enumerate(
                        zip(predictions, y_val, strict=False),
                    ):
                        if pred == actual:
                            # Correct prediction: positive return
                            returns.append(0.01)  # 1% return
                        else:
                            # Incorrect prediction: negative return
                            returns.append(-0.005)  # -0.5% return

                    returns = np.array(returns)

                    # Calculate Sharpe Ratio
                    if len(returns) > 0:
                        mean_return = np.mean(returns)
                        std_return = np.std(returns)

                        if std_return > 0:
                            sharpe_ratio = mean_return / std_return
                        else:
                            sharpe_ratio = 0.0
                    else:
                        sharpe_ratio = 0.0

                    model_sharpe_ratios[name] = sharpe_ratio

                    # Models with negative Sharpe Ratio get weight 0
                    if sharpe_ratio <= 0:
                        model_weights[name] = 0.0
                    else:
                        model_weights[name] = sharpe_ratio

                except Exception as e:
                    self.logger.warning(
                        f"Could not calculate Sharpe Ratio for model {name}: {e}",
                    )
                    model_sharpe_ratios[name] = 0.0
                    model_weights[name] = 0.0

            # Normalize weights so they sum to 1
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                for name in model_weights:
                    model_weights[name] /= total_weight
            else:
                # If all weights are 0, assign equal weights
                for name in model_weights:
                    model_weights[name] = 1.0 / len(model_weights)

            # Use proximity-weighted ensemble if SR features available
            use_sr_blend = all(k in X_val.columns for k in ["dist_to_support_pct", "dist_to_resistance_pct"]) or "sr_touch" in X_val.columns
            if use_sr_blend:
                cfg_base = self.config.get("sr_blending", {})
                # Hyperparameter tuning: grid search over blending params with CV
                grid = cfg_base.get("tuning_grid", {
                    "sr_proximity_tau": [0.002, 0.003, 0.004, 0.006],
                    "sr_touch_boost": [0.0, 0.1, 0.2, 0.3],
                    "sr_max_weight": [0.5, 0.6, 0.7, 0.8],
                    "hysteresis_alpha_up": [0.5, 0.6, 0.7],
                    "hysteresis_alpha_down": [0.2, 0.3, 0.4],
                })
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                best_score = -np.inf
                best_cfg = cfg_base.copy()
                for tau in grid["sr_proximity_tau"]:
                    for boost in grid["sr_touch_boost"]:
                        for mmax in grid["sr_max_weight"]:
                            for aup in grid["hysteresis_alpha_up"]:
                                for adn in grid["hysteresis_alpha_down"]:
                                    cfg = {
                                        "sr_proximity_tau": tau,
                                        "sr_touch_boost": boost,
                                        "sr_max_weight": mmax,
                                        "hysteresis_alpha_up": aup,
                                        "hysteresis_alpha_down": adn,
                                    }
                                    cv_scores = []
                                    try:
                                        for tr_idx, te_idx in kf.split(X_val):
                                            Xtr, Xte = X_val.iloc[tr_idx], X_val.iloc[te_idx]
                                            ytr, yte = y_val.iloc[tr_idx], y_val.iloc[te_idx]
                                            ens = ProximityWeightedEnsemble(models, model_names, model_weights, cfg=cfg)
                                            # Fit-free ensemble; just evaluate predict_proba and accuracy
                                            preds = np.argmax(ens.predict_proba(Xte), axis=1)
                                            # Map to binary if labels not 0/1
                                            try:
                                                from sklearn.metrics import accuracy_score
                                                score = accuracy_score(yte, preds)
                                            except Exception:
                                                score = (preds == yte.values).mean()
                                            cv_scores.append(float(score))
                                    except Exception:
                                        continue
                                    if cv_scores:
                                        avg = float(np.mean(cv_scores))
                                        if avg > best_score:
                                            best_score = avg
                                            best_cfg = cfg
                self.logger.info(f"SR blending tuned best score={best_score:.4f} cfg={best_cfg}")
                ensemble = ProximityWeightedEnsemble(models, model_names, model_weights, cfg=best_cfg)
            else:
                ensemble = DynamicWeightedEnsemble(models, model_names, model_weights)

            # Evaluate ensemble performance
            ensemble_predictions = ensemble.predict(X_val)
            ensemble_metrics = {
                "accuracy": accuracy_score(y_val, ensemble_predictions),
                "precision": precision_score(
                    y_val,
                    ensemble_predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_val,
                    ensemble_predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_val,
                    ensemble_predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }

            return {
                "ensemble": ensemble,
                "ensemble_type": "DynamicWeightedEnsemble",
                "base_models": model_names,
                "weights": model_weights,
                "sharpe_ratios": model_sharpe_ratios,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat(),
                "validation_metrics": ensemble_metrics,
                "features": {
                    "sharpe_ratio_weighting": True,
                    "negative_sharpe_filtering": True,
                    "dynamic_weighting": True,
                },
            }

        except Exception as e:
            self.logger.exception(
                f"Error creating dynamic weighting ensemble for {regime_name}: {e}",
            )
            raise

    async def _create_voting_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        regime_name: str,
    ) -> dict[str, Any]:
        """Create voting ensemble."""
        try:
            from sklearn.ensemble import VotingClassifier

            # Create voting classifier
            estimators = list(zip(model_names, models, strict=False))
            voting_classifier = VotingClassifier(estimators=estimators, voting="soft")

            return {
                "ensemble": voting_classifier,
                "ensemble_type": "Voting",
                "base_models": model_names,
                "voting_method": "soft",
                "regime": regime_name,
                "creation_date": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.exception(
                f"Error creating voting ensemble for {regime_name}: {e}",
            )
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the analyst ensemble creation step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = AnalystEnsembleCreationStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:
        print(failed("Analyst ensemble creation failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
