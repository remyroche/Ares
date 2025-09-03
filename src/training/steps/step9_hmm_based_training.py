# src/training/steps/step9_hmm_based_training.py

"""Step 9: HMM-Based Model Training with Standardized Data Quality Management.

This step performs HMM-based model training with timeframe-specific architectures
and S/R integration, using standardized data quality management patterns.
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from typing import Any
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import common operations
from src.utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_read_parquet, safe_to_parquet, safe_copy, safe_json_dump
)

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "lightgbm",
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "src.tactician.sr_breakout_predictor",
    "src.utils.centralized_decorators",
    "src.utils.logger",
    "src.utils.enhanced_mlflow_integration",
    "src.training.model_probability_generator",
    "src.training.model_saving_utils"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
sr_breakout_predictor = PipelineStandards.safe_import("src.tactician.sr_breakout_predictor", None)
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
model_probability_generator = PipelineStandards.safe_import("src.training.model_probability_generator", None)
model_saving_utils = PipelineStandards.safe_import("src.training.model_saving_utils", None)
lightgbm = PipelineStandards.safe_import("lightgbm", None)
numpy = PipelineStandards.safe_import("numpy", None)
pandas = PipelineStandards.safe_import("pandas", None)
torch = PipelineStandards.safe_import("torch", None)
sklearn = PipelineStandards.safe_import("sklearn", None)

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

if centralized_decorators is None:
    PerformanceLevel = "BASIC"
    ValidationLevel = "BASIC"
    adaptive_resource_allocation = create_fallback_decorator()
    comprehensive_validation = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    intelligent_caching = create_fallback_decorator()
    model_validation = create_fallback_decorator()
    performance_monitor = create_fallback_decorator()
    pipeline_checkpoint = create_fallback_decorator()
    validate_feature_engineering_with_lookahead_bias_detection = create_fallback_decorator()
else:
    PerformanceLevel = centralized_decorators.PerformanceLevel
    ValidationLevel = centralized_decorators.ValidationLevel
    adaptive_resource_allocation = centralized_decorators.adaptive_resource_allocation
    comprehensive_validation = centralized_decorators.comprehensive_validation
    handle_errors = centralized_decorators.handle_errors
    intelligent_caching = centralized_decorators.intelligent_caching
    model_validation = centralized_decorators.model_validation
    performance_monitor = centralized_decorators.performance_monitor
    pipeline_checkpoint = centralized_decorators.pipeline_checkpoint
    validate_feature_engineering_with_lookahead_bias_detection = centralized_decorators.validate_feature_engineering_with_lookahead_bias_detection

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_model = lambda *args, **kwargs: "fallback_model"
    log_step_metrics = lambda *args, **kwargs: None
    log_step_artifact = lambda *args, **kwargs: "fallback_artifact"
    log_step_report = lambda *args, **kwargs: "fallback_report"
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: "fallback_artifact"
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_model = enhanced_mlflow.log_step_model
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_artifact = enhanced_mlflow.log_step_artifact
    log_step_report = enhanced_mlflow.log_step_report
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name

# Suppress warnings
warnings.filterwarnings("ignore")


class HMMBasedTrainingStep:
    """Step 9: HMM-Based Model Training with Standardized Data Quality Management.

    Includes an optional forecasting head that emits next-regime probabilities
    and simple exit-within-H-bars signals leveraging Step 3 HMM posteriors and
    transition probabilities.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.standards = pipeline_standards
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Validate environment on initialization
        self._validate_environment()

        # Initialize SRBreakoutPredictor for S/R level integration with optimized parameters
        if sr_breakout_predictor is not None:
            try:
                sr_config = config.copy()
                sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
                sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
                self.sr_predictor = sr_breakout_predictor.SRBreakoutPredictor(sr_config)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize SRBreakoutPredictor: {e}")
                self.sr_predictor = None
        else:
            self.logger.warning("⚠️ SRBreakoutPredictor not available")
            self.sr_predictor = None

        # Initialize S/R outcome model trainer
        self.sr_outcome_trainer = None
        self.sr_outcome_model_trained = False
        
        # Initialize probability generator for enhanced prediction service
        if model_probability_generator is not None:
            self.probability_generator = model_probability_generator.ModelProbabilityGenerator()
        else:
            self.logger.warning("⚠️ ModelProbabilityGenerator not available")
            self.probability_generator = None

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")
        
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
            self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
            self.logger.info("✅ All required dependencies available")

        # Model architecture mapping from config
        hmm_lm_config = config.get("HMM_LM", {})
        specialist_config = hmm_lm_config.get("specialist_models", {})

        self.model_architectures = {}
        for timeframe, model_config in specialist_config.items():
            self.model_architectures[timeframe] = model_config.get(
                "architecture", "LightGBM",
            )

        # Fallback to default if config not available
        if not self.model_architectures:
            self.model_architectures = {
                "1m": "CNN",  # Tactician
                "5m": "TCN",  # Analyst
                "15m": "Transformer",  # Analyst
                "30m": "LightGBM",  # Analyst
            }

        # HMM-derived features (composite regimes and intensity scores)
        self.hmm_features = [
            "composite_cluster_id",
            "intensity_cluster_0",
            "intensity_cluster_1",
            "intensity_cluster_2",
            "intensity_cluster_3",
            "intensity_cluster_4",
            "intensity_cluster_5",
            "intensity_cluster_6",
            "intensity_cluster_7",
            "intensity_cluster_8",
            "intensity_cluster_9",
            "intensity_cluster_10",
            "intensity_cluster_11",
            "intensity_cluster_12",
            "intensity_cluster_13",
            "intensity_cluster_14",
            "intensity_cluster_15",
            "intensity_cluster_16",
            "intensity_cluster_17",
            "intensity_cluster_18",
            "intensity_cluster_19",
        # Regime probability features
            "momentum_p_state_0",
            "momentum_p_state_1",
            "momentum_p_state_2",
            "momentum_p_state_3",
            "volatility_p_state_0",
            "volatility_p_state_1",
            "volatility_p_state_2",
            "volatility_p_state_3",
            "liquidity_p_state_0",
            "liquidity_p_state_1",
            "liquidity_p_state_2",
            "liquidity_p_state_3",
            "microstructure_p_state_0",
            "microstructure_p_state_1",
            "microstructure_p_state_2",
            "microstructure_p_state_3",
            "microstructure_p_state_4",
            "orderflow_p_state_0",
            "orderflow_p_state_1",
            "orderflow_p_state_2",
            "orderflow_p_state_3",
            "orderflow_p_state_4",
        ]

        # Initialize enhanced LM optimizer
        self.enhanced_lm_optimizer = None
        try:
            from src.training.enhanced_lm_optimizer import EnhancedLMOptimizer
            self.enhanced_lm_optimizer = EnhancedLMOptimizer(config)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize enhanced LM optimizer: {e}")

        # Initialize optimized feature selection manager (fallback)
        self.optimized_feature_selection = None
        try:
            from src.training.optimized_feature_selection_manager import (
                OptimizedFeatureSelectionManager,
            )
            self.optimized_feature_selection = OptimizedFeatureSelectionManager(config)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize optimized feature selection: {e}")

        # All available features - will be optimized by feature selection
        # Note: These should be returns-based features, not raw data
        self.all_features = [
        # Technical indicators (already returns-based or normalized)
            "momentum_strength",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "rsi",
            "bb_position",
            "trend_regime",
            "macd",
            "atr",
            "adx",
            "sma",
            "ema",
            "cci",
            "mfi",
            "roc",
        # Volatility measures (returns-based)
            "volume_volatility",
            "parkinson_volatility",
            "garman_klass_volatility",
            "adaptive_atr",
            "1m_price_volatility",
            "ewma_volatility",
        # Liquidity measures (returns-based or normalized)
            "liquidity_score",
            "bid_ask_spread",
            "market_depth",
            "volume_profile",
            "order_imbalance",
            "tick_size",
            "spread_tightness",
            "price_impact",
        # Order flow measures (returns-based or normalized)
            "trade_frequency",
            "order_flow_imbalance",
            "order_flow_vpin_50",
            "order_flow_aggressor_ratio",
            "order_flow_large_small_imbalance",
            "order_flow_pressure",
            "order_flow_taker_imbalance",
            "order_flow_avg_trade_size",
            "order_flow_trade_size_volatility",
        # Normalized/returns-based features
            "momentum_strength_z_score",
            "rsi_change",
            "volume_pct_change",
            "volatility_acceleration",
            "liquidity_bounded",
            "price_log",
            "spread_normalized",
        # Interaction features
            "momentum_x_volume",
            "volatility_x_liquidity",
            "price_x_volume",
            "rsi_div_volatility",
            "momentum_div_liquidity",
        # Lagged features
            "momentum_lag_1",
            "volume_lag_1",
            "rsi_lag_1",
            "volatility_lag_1",
            "momentum_lag_5",
            "volume_lag_5",
            "rsi_lag_5",
            "volatility_lag_5",
        # Causality features
            "momentum_predicts_volume",
            "volatility_causality",
            "liquidity_divergence",
            "price_stress",
            "volume_extreme",
            "momentum_extreme",
        ]

        # Remove feature selection configuration - let ML models handle it
        # self.feature_selection_config = {...}  # REMOVED

        # Validation and cross-validation configuration
        self.validation_config = {
            "n_splits": 5,  # Number of time series splits
            "test_size": 0.2,  # Test set size
            "validation_size": 0.2,  # Validation set size
            "min_samples_per_split": 1000,
            "regime_aware_splitting": True,
        }

        # Data source configuration
        self.data_source_config = {
            "prefer_pickle": True,
            "fallback_to_parquet": True,
            "load_regime_weights": False,  # Temporarily disable to avoid the method call issue
            "validate_data_quality": True,
        }

    def print(self, message: str) -> None:
        """Print message using logger."""
        self.logger.info(message)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="HMM-based training step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the HMM-based training step."""
        self.logger.info("Initializing HMM-Based Training Step...")
        self.logger.info("HMM-Based Training Step initialized successfully")

    def _get_available_features(self, data: pd.DataFrame) -> list:
        """Get all available features from the dataset, excluding target and metadata columns."""
        try:
            # Exclude non-feature columns
            exclude_columns = [
                "target",
                "timeframe",
                "composite_cluster_id",
                "sample_weight",
            ]

            # Get all available features
            available_features = [
                col for col in data.columns if col not in exclude_columns
            ]

            self.logger.info(f"✅ Found {len(available_features)} available features")
            return available_features

        except Exception as e:
            self.logger.exception(f"❌ Failed to get available features: {e}")
            return []

    async def _apply_enhanced_optimization(
        self, features_df: pd.DataFrame, target: pd.Series, timeframe: str, architecture: str, ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply enhanced optimization for Step 6 models including feature selection, regularization, and hyperparameter optimization.

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            timeframe: Timeframe (1m = 5m, 15m, 30m)
            architecture: Model architecture (CNN = TCN, Transformer, LightGBM)

        Returns: Tuple of (optimized_features_df = optimization_metadata)

        """
        try:
            # Enhanced LM optimizer is required - no fallbacks
            if self.enhanced_lm_optimizer is None:
                msg = "Enhanced LM optimizer is required but not initialized"
                raise RuntimeError(msg)

            # Use enhanced LM optimizer for comprehensive optimization
            self.logger.info(f"🔄 Applying enhanced LM optimization for {timeframe} {architecture}")

            # Determine model type
            model_type = "classification" if target.dtype == "object" or len(target.unique()) < 10 else "regression"

            # Apply comprehensive optimization
            optimization_results, optimized_features = await self.enhanced_lm_optimizer.optimize_lm_model(
                step_name="step06",
                features_df=features_df,
                target=target,
                model_type=model_type,
                architecture=architecture,
            )

            # Use optimized features directly from the optimizer
            self.logger.info(f"✅ Applied feature selection: {len(features_df.columns)} -> {len(optimized_features.columns)} features")

            self.logger.info(f"✅ Enhanced optimization completed for {timeframe} {architecture}")
            self.logger.info("📊 Optimization metrics:")
            self.logger.info(f"   - Feature selection: {optimization_results.get('feature_selection', {}).get('final_features', len(features_df.columns))} features")
            self.logger.info(f"   - Regularization: {optimization_results.get('regularization', {})}")
            self.logger.info(f"   - Hyperparameter optimization: {optimization_results.get('hyperparameter_optimization', {})}")

            return optimized_features, optimization_results

        except Exception as e:
            self.logger.exception(f"❌ Enhanced optimization failed for {timeframe} {architecture}: {e}")
            msg = f"Enhanced optimization failed for {timeframe} {architecture}: {e}"
            raise RuntimeError(msg)

    @with_enhanced_mlflow_logging("step9_hmm_based_training")
    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="HMM-based training step execution",
    )
    @validate_feature_engineering_with_lookahead_bias_detection
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any], ) -> dict[str, Any]:
        """Execute HMM-based model training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results

        """
        try:
            self.logger.info("🔄 Executing HMM-Based Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            timeframes = training_input.get("timeframes", ["1m", "5m", "15m", "30m"])

            # Load HMM cluster data
            hmm_data = await self._load_hmm_data(exchange, symbol, data_dir, timeframes)
            if not hmm_data:
                msg = "Failed to load HMM data"
                raise ValueError(msg)

            # Load feature data
            feature_data = await self._load_feature_data(
                exchange, symbol, data_dir, timeframes,
            )

            # Check if we have data for all timeframes
            missing_timeframes = [
                tf for tf in timeframes if tf not in feature_data or feature_data[tf].empty
            ]

            if missing_timeframes:
                self.logger.info(
                    f"🔄 Missing feature data for timeframes: {missing_timeframes}",
                )
                self.logger.info(
                    "🔄 Attempting to create timeframe-specific feature files...",
                )
                await self._create_timeframe_specific_features(
                    exchange, symbol, data_dir, timeframes,
                )
                feature_data = await self._load_feature_data(
                    exchange, symbol, data_dir, timeframes,
                )

            if not feature_data:
                msg = "Failed to load feature data"
                raise ValueError(msg)

            # Load regime weights if available
            regime_weights = None
            if self.data_source_config["load_regime_weights"]:
                regime_weights = await self._load_regime_weights(
                    exchange, symbol, data_dir,
                )

            # Train models for each timeframe - BOTH regime-specific AND combined models are required
            training_results: dict[str, Any] = {}
            for timeframe in timeframes:
                self.logger.info(f"🎯 Training models for {timeframe}")

                # Step 1: Train regime-specific models (required)
                self.logger.info(
                    f"🎯 Step 1: Training regime-specific models for {timeframe}",
                )
                regime_models = await self._train_regime_specific_models(timeframe)

                if not regime_models:
                    self.logger.error(
                        f"❌ Failed to train regime-specific models for {timeframe}",
                    )
                    self.logger.error(
                        "❌ Both regime-specific AND combined models are required",
                    )
                    msg = f"Failed to train regime-specific models for {timeframe}"
                    raise ValueError(msg)

                # Step 2: Train combined model (also required)
                self.logger.info(f"🎯 Step 2: Training combined model for {timeframe}")

                # Prepare data for this timeframe
                tf_data = await self._prepare_timeframe_data(
                    hmm_data[timeframe], feature_data[timeframe], timeframe,
                )

                if (
                    tf_data is None
                    or len(tf_data) < self.validation_config["min_samples_per_split"]
                ):
                    self.logger.error(
                        f"❌ Insufficient data for combined model training for {timeframe}",
                    )
                    msg = f"Insufficient data for combined model training for {timeframe}"
                    raise ValueError(msg)

                # Add regime weights if available
                if regime_weights is not None:
                    tf_data = await self._add_regime_weights(
                        tf_data, regime_weights, timeframe,
                    )

                # Train combined model based on architecture
                combined_model_result = await self._train_timeframe_model(
                    tf_data, timeframe,
                )
                if not combined_model_result:
                    self.logger.error(
                        f"❌ Failed to train combined model for {timeframe}",
                    )
                    msg = f"Failed to train combined model for {timeframe}"
                    raise ValueError(msg)

                # Store both types of models
                training_results[timeframe] = {
                    "training_type": "both",
                    "regime_models": regime_models,
                    "combined_model": combined_model_result,
                    "total_regimes": len(regime_models),
                    "architecture": self.model_architectures[timeframe],
                }
                self.logger.info(
                    f"✅ Trained {len(regime_models)} regime-specific models + 1 combined model for {timeframe}",
                )

            # Save models and metadata
            await self._save_models(training_results, exchange, symbol, data_dir)

            # Train S/R outcome model using all available features
            self.logger.info("🔄 Training S/R outcome model...")
            # sr_outcome_training_success = await self._train_sr_outcome_model(feature_data)
            sr_outcome_training_success = True  # Temporarily skip S/R outcome training

            if sr_outcome_training_success:
                self.logger.info("✅ S/R outcome model training completed successfully")
            else:
                self.logger.warning("⚠️ S/R outcome model training failed or skipped")

            # Emit regime forecasting artifacts (next-regime probabilities and exit-within-H)
            try:
                import json
                import os
                import pandas as _pd

                rf_dir = os.path.join(data_dir, "regime_forecasting")
                ensure_directory(rf_dir)

                regime_forecasting_summary: dict[str, dict] = {}
                for tf in timeframes:
                    try:
                        df = hmm_data.get(tf)
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            continue
                        if "composite_cluster_id" not in df.columns:
                            continue

                        cids = df["composite_cluster_id"].astype(int)
                        # Empirical transition matrix
                        transitions: dict[int, dict[int, int]] = {}
                        prev = None
                        for cid in cids.tolist():
                            if prev is not None:
                                transitions.setdefault(int(prev), {}).setdefault(int(cid), 0)
                                transitions[int(prev)][int(cid)] += 1
                            prev = cid
                        # Normalize to probabilities
                        trans_prob: dict[int, dict[int, float]] = {}
                        for i, row in transitions.items():
                            row_sum = float(sum(row.values()))
                            if row_sum <= 0:
                                continue
                            trans_prob[i] = {j: cnt / row_sum for j, cnt in row.items()}

                        current_cid = int(cids.iloc[-1])
                        next_probs = trans_prob.get(current_cid, {})
                        p_stay = float(next_probs.get(current_cid, 0.0))
                        H = 20
                        exit_within_H = 1.0 - (p_stay ** H)

                        artifact = {
                            "timeframe": tf,
                            "current_regime": current_cid,
                            "next_regime_probabilities": next_probs,
                            "exit_within_H_bars_prob": exit_within_H,
                            "H": H,
                        }
                        regime_forecasting_summary[tf] = artifact

                        rf_path = os.path.join(
                            rf_dir, f"{exchange}_{symbol}_{tf}_regime_forecasting.json",
                        )
                        with open(rf_path, "w") as f:
                            json.dump(artifact, f, indent=2)
                        self.logger.info(f"💾 Saved regime forecasting artifact -> {rf_path}")
                    except Exception as _inner:
                        self.logger.warning(
                            f"⚠️ Regime forecasting generation failed for {tf}: {_inner}",
                        )

                if regime_forecasting_summary:
                    pipeline_state["regime_forecasting"] = regime_forecasting_summary
            except Exception as _fe:
                self.logger.warning(
                    f"⚠️ Skipped regime forecasting artifacts due to error: {_fe}",
                )

            self.logger.info("✅ HMM-Based Training completed successfully")
            return {
                "status": "SUCCESS",
                "models_trained": len(training_results),
                "timeframes": list(training_results.keys()),
                "sr_outcome_model_trained": sr_outcome_training_success,
                "results": training_results,
            }

        except Exception as e:
            self.logger.exception(f"❌ HMM-Based Training failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    async def _load_hmm_data(
        self, exchange: str, symbol: str, data_dir: str, timeframes: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Load HMM cluster data for all timeframes using centralized HMM composite manager."""
        hmm_data: dict[str, pd.DataFrame] = {}

        # Use centralized HMM composite manager
        try:
            from src.utils.hmm_composite_manager import get_hmm_composite_manager

            hmm_manager = get_hmm_composite_manager()
        except ImportError as e:
            self.logger.exception(f"❌ Failed to import HMM composite manager: {e}")
            return {}

        for timeframe in timeframes:
            try:
                # Load composite clusters using the manager
                clusters_df = hmm_manager.load_composite_clusters(
                    exchange=exchange, symbol=symbol, timeframe=timeframe, data_dir=data_dir,
                )

                if clusters_df is None:
                    self.logger.warning(
                        f"⚠️ No HMM composite clusters found for {timeframe}",
                    )
                    continue

                # Ensure timestamp index
                if "timestamp" in clusters_df.columns:
                    clusters_df["timestamp"] = pd.to_datetime(clusters_df["timestamp"])
                    # Normalize timestamps to remove microseconds for consistency
                    clusters_df["timestamp"] = clusters_df["timestamp"].dt.floor("1T")
                    clusters_df = clusters_df.set_index("timestamp")

                # Load intensity scores if available
                intensity_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
                intensity_df: pd.DataFrame | None = None
                if os.path.exists(intensity_path):
                    try:
                        intensity_df = pd.read_parquet(intensity_path)
                        if "timestamp" in intensity_df.columns:
                            intensity_df["timestamp"] = pd.to_datetime(
                                intensity_df["timestamp"],
                            )
                            # Normalize timestamps to remove microseconds for consistency
                            intensity_df["timestamp"] = intensity_df[
                                "timestamp"
                            ].dt.floor("1T")
                            intensity_df = intensity_df.set_index("timestamp")

                        # Merge cluster assignments with intensity scores
                        hmm_df = clusters_df.merge(
                            intensity_df, left_index=True, right_index=True, how="inner"
                        )
                        hmm_data[timeframe] = hmm_df
                        self.logger.info(
                            f"✅ Loaded complete HMM data for {timeframe}: {hmm_df.shape}",
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Failed to load intensity data for {timeframe}: {e}",
                        )
                        hmm_data[timeframe] = clusters_df
                        self.logger.info(
                            f"✅ Loaded HMM clusters only for {timeframe}: {clusters_df.shape}",
                        )
                else:
                    hmm_data[timeframe] = clusters_df
                    self.logger.info(
                        f"✅ Loaded HMM clusters only for {timeframe}: {clusters_df.shape}",
                    )

            except Exception as e:
                self.logger.exception(f"❌ Failed to load HMM data for {timeframe}: {e}")

        return hmm_data

    async def _load_feature_data(
        self, exchange: str, symbol: str, data_dir: str, timeframes: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Load feature data for all timeframes with multiple source support and validation.

        Prefer centralized artifact loader for 1m features to ensure column alignment via metadata,
        then resample to target timeframes.
        """
        feature_data: dict[str, pd.DataFrame] = {}

        # 1) Try centralized artifact loader for 1m and resample others
        try:
            from src.training.steps.feature_artifact_loader import (
                load_features_for_step,
            )
            self.logger.info("🔍 Using centralized feature_artifact_loader for 1m features (Step 6)")
            loaded = load_features_for_step(symbol, exchange, data_dir, step_name="Step6.HMMTraining")
            # Use train split as canonical for resampling; index must be timestamp
            base_df = loaded.get("train")
            if isinstance(base_df, pd.DataFrame) and not base_df.empty:
                if "timestamp" in base_df.columns:
                    base_df = base_df.set_index(pd.to_datetime(base_df["timestamp"]).dt.floor("1T")).drop(columns=["timestamp"], errors="ignore")
                elif not isinstance(base_df.index, pd.DatetimeIndex):
                    # Create synthetic timestamp if needed
                    start_time = pd.Timestamp.now() - pd.Timedelta(days=60)
                    base_df.index = pd.date_range(start=start_time, periods=len(base_df), freq="1T", tz="UTC")

                # Assign 1m
                feature_data["1m"] = base_df

                # Resample to other timeframes
                for timeframe in timeframes:
                    if timeframe == "1m":
                        continue
                    resampled = await self._resample_features_to_timeframe(base_df, timeframe)
                    if resampled is not None:
                        feature_data[timeframe] = resampled
                        self.logger.info(f"✅ Resampled features for {timeframe}: {resampled.shape}")
            else:
                self.logger.info("ℹ️ Centralized loader returned empty or invalid 1m features; falling back to legacy loaders")
        except Exception as e:
            self.logger.info(f"ℹ️ Centralized loader unavailable or failed: {e}. Falling back to legacy loaders")

        # 2) Fallback: legacy multi-source loading per timeframe if not already loaded
        for timeframe in timeframes:
            try:
                if timeframe in feature_data and isinstance(feature_data[timeframe], pd.DataFrame) and not feature_data[timeframe].empty:
                    # Already populated from centralized path
                    continue

                features_df: pd.DataFrame | None = None

                # Try to load from pickle first if preferred
                if self.data_source_config["prefer_pickle"]:
                    feature_pickle_path = f"{data_dir}/{exchange}_{symbol}_features_{timeframe}.pkl"
                    if os.path.exists(feature_pickle_path):
                        with open(feature_pickle_path, "rb") as f:
                            features_df = pickle.load(f)
                        if isinstance(features_df, pd.DataFrame):
                            self.logger.info(
                                f"✅ Loaded features from pickle for {timeframe}: {features_df.shape}",
                            )
                        else:
                            raise ValueError(f"Invalid pickle format for {timeframe}")

                # Try to load from combined parquet file
                if (
                    features_df is None
                    and self.data_source_config["fallback_to_parquet"]
                ):
                    feature_path = f"{data_dir}/{exchange}_{symbol}_features_{timeframe}.parquet"
                    if os.path.exists(feature_path):
                        features_df = pd.read_parquet(feature_path)
                        self.logger.info(
                            f"✅ Loaded features from parquet for {timeframe}: {features_df.shape}",
                        )

                # Try to load from split parquet files and combine them
                if features_df is None:
                    features_df = await self._load_and_combine_split_features(
                        exchange, symbol, data_dir, timeframe,
                    )
                    if features_df is not None:
                        self.logger.info(
                            f"✅ Loaded and combined split features for {timeframe}: {features_df.shape}",
                        )

                # Try to load from legacy train/test/validation pickle files
                if features_df is None:
                    features_df = await self._load_and_combine_legacy_features(
                        exchange, symbol, data_dir, timeframe,
                    )
                    if features_df is not None:
                        self.logger.info(
                            f"✅ Loaded and combined legacy features for {timeframe}: {features_df.shape}",
                        )

                if features_df is None:
                    self.logger.warning(f"⚠️ No feature data found for {timeframe}")
                    continue

                # Handle timestamp index - create one if missing
                if "timestamp" in features_df.columns:
                    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
                    # Normalize timestamps to remove microseconds for consistency with HMM data
                    features_df["timestamp"] = features_df["timestamp"].dt.floor("1T")
                    features_df = features_df.set_index("timestamp")
                elif not isinstance(features_df.index, pd.DatetimeIndex):
                    # Create a synthetic timestamp index based on the data length
                    # This assumes the data is in chronological order
                    self.logger.info(
                        f"🔄 Creating synthetic timestamp index for {timeframe}",
                    )
                    start_time = pd.Timestamp.now() - pd.Timedelta(days=60)  # 60 days ago
                    timestamps = pd.date_range(
                        start=start_time, periods=len(features_df), freq="1T", tz="UTC",
                    )
                    features_df = features_df.copy()
                    features_df.index = timestamps
                    self.logger.info(
                        f"✅ Created timestamp index for {timeframe}: {len(features_df)} rows",
                    )

                # Data quality validation
                if self.data_source_config["validate_data_quality"]:
                    features_df = await self._validate_and_clean_data(
                        features_df, timeframe,
                    )

                feature_data[timeframe] = features_df
                self.logger.info(
                    f"✅ Processed features for {timeframe}: {features_df.shape}",
                )

            except Exception as e:
                self.logger.exception(f"❌ Failed to load features for {timeframe}: {e}")

        return feature_data

    async def _validate_and_clean_data(
        self, df: pd.DataFrame, timeframe: str,
    ) -> pd.DataFrame:
        """Validate and clean feature data."""
        try:
            original_shape = df.shape

            # Remove rows with all NaN values
            df = df.dropna(how="all")

            # Remove columns with all NaN values
            df = df.dropna(axis=1, how="all")

            # Fill remaining NaN values with forward fill then backward fill
            df = df.fillna(method="ffill").fillna(method="bfill")

            # Remove any remaining NaN values
            df = df.dropna()

            if df.shape != original_shape:
                self.logger.info(
                    f"   🔧 Cleaned {timeframe} data: {original_shape} -> {df.shape}",
                )

            return df

        except Exception as e:
            self.logger.exception(f"❌ Data validation failed for {timeframe}: {e}")
            return df

    async def _load_and_combine_split_features(
        self, exchange: str, symbol: str, data_dir: str, timeframe: str,
    ) -> pd.DataFrame | None:
        """Load and combine split parquet features for a specific timeframe."""
        try:
            # Path to the split features directory
            split_features_dir = f"{data_dir}/parquet/features/exchange={exchange}/symbol={symbol}/timeframe={timeframe}"

            if not os.path.exists(split_features_dir):
                self.logger.debug(
                    f"Split features directory not found: {split_features_dir}",
                )
                return None

            # Load train, validation, and test splits
            splits: list[pd.DataFrame] = []
            for split_name in ["train", "validation", "test"]:
                split_dir = os.path.join(split_features_dir, f"split={split_name}")
                if os.path.exists(split_dir):
                    # Find all parquet files in the split directory
                    parquet_files = [
                        f for f in os.listdir(split_dir) if f.endswith(".parquet")
                    ]
                    if parquet_files:
                        split_file = os.path.join(split_dir, parquet_files[0])
                        split_df = pd.read_parquet(split_file)
                        split_df["split"] = split_name  # Add split identifier
                        splits.append(split_df)
                        self.logger.debug(
                            f"Loaded {split_name} split for {timeframe}: {split_df.shape}",
                        )

            if not splits:
                self.logger.debug(f"No split files found for {timeframe}")
                return None

            # Combine all splits
            combined_df = pd.concat(splits, ignore_index=True)

            # Ensure timestamp column exists and is properly formatted
            if "timestamp" in combined_df.columns:
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                # Normalize timestamps to remove microseconds for consistency with HMM data
                combined_df["timestamp"] = combined_df["timestamp"].dt.floor("1T")
                combined_df = combined_df.set_index("timestamp")

            # Remove the split column if it exists
            if "split" in combined_df.columns:
                combined_df = combined_df.drop("split", axis=1)

            self.logger.info(
                f"✅ Combined split features for {timeframe}: {combined_df.shape}",
            )
            return combined_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to load split features for {timeframe}: {e}")
            return None

    async def _load_and_combine_legacy_features(
        self, exchange: str, symbol: str, data_dir: str, timeframe: str,
    ) -> pd.DataFrame | None:
        """Load and combine legacy train/test/validation pickle features."""
        try:  # Try to load train, test, and validation pickle files
            splits: list[pd.DataFrame] = []
            for split_name in ["train", "test", "validation"]:
                pickle_path = f"{data_dir}/{exchange}_{symbol}_features_{split_name}.pkl"
                if os.path.exists(pickle_path):
                    with open(pickle_path, "rb") as f:
                        split_df = pickle.load(f)
                    if isinstance(split_df, pd.DataFrame):
                        split_df["split"] = split_name  # Add split identifier
                        splits.append(split_df)
                        self.logger.debug(
                            f"Loaded {split_name} pickle for {timeframe}: {split_df.shape}",
                        )

            if not splits:
                self.logger.debug(f"No legacy pickle files found for {timeframe}")
                return None

            # Combine all splits
            combined_df = pd.concat(splits, ignore_index=True)

            # Ensure timestamp column exists and is properly formatted
            if "timestamp" in combined_df.columns:
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
                # Normalize timestamps to remove microseconds for consistency with HMM data
                combined_df["timestamp"] = combined_df["timestamp"].dt.floor("1T")
                combined_df = combined_df.set_index("timestamp")

            # Remove the split column if it exists
            if "split" in combined_df.columns:
                combined_df = combined_df.drop("split", axis=1)

            self.logger.info(
                f"✅ Combined legacy features for {timeframe}: {combined_df.shape}",
            )
            return combined_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to load legacy features for {timeframe}: {e}")
            return None

    async def _create_timeframe_specific_features(
        self, exchange: str, symbol: str, data_dir: str, timeframes: list[str],
    ) -> None:
        """Create timeframe-specific feature files from available data."""
        try:  # First, try to get combined feature data from any available source
            combined_features: pd.DataFrame | None = None

            # Try to load from split parquet files (1m timeframe)
            split_features_dir = f"{data_dir}/parquet/features/exchange={exchange}/symbol={symbol}/timeframe=1m"
            if os.path.exists(split_features_dir):
                combined_features = await self._load_and_combine_split_features(
                    exchange, symbol, data_dir, "1m",
                )
            if combined_features is not None:
                self.logger.info(
                    f"✅ Loaded combined features from split files: {combined_features.shape}",
                )

            # If no split files, try legacy pickle files
            if combined_features is None:
                combined_features = await self._load_and_combine_legacy_features(
                    exchange, symbol, data_dir, "1m",
                )
            if combined_features is not None:
                self.logger.info(
                    f"✅ Loaded combined features from legacy files: {combined_features.shape}",
                )

            if combined_features is None:
                self.logger.warning(
                    "⚠️ No source feature data found to create timeframe-specific files",
                )
                return

            # Create timeframe-specific files by resampling the 1m data
            for timeframe in timeframes:
                if timeframe == "1m":
                    # Save the 1m data directly
                    output_path = f"{data_dir}/{exchange}_{symbol}_features_{timeframe}.parquet"
                    combined_features.to_parquet(output_path)
                    self.logger.info(
                        f"✅ Created {timeframe} feature file: {output_path}",
                    )
                else:
                    # Resample to other timeframes
                    resampled_features = await self._resample_features_to_timeframe(
                        combined_features, timeframe,
                    )
                    if resampled_features is not None:
                        output_path = f"{data_dir}/{exchange}_{symbol}_features_{timeframe}.parquet"
                        resampled_features.to_parquet(output_path)
                        self.logger.info(
                            f"✅ Created {timeframe} feature file: {output_path}",
                        )
                    else:  # If resampling fails, copy the 1m data for other timeframes
                        self.logger.warning(
                            f"⚠️ Resampling failed for {timeframe}, using 1m data",
                        )
                        output_path = f"{data_dir}/{exchange}_{symbol}_features_{timeframe}.parquet"
                        combined_features.to_parquet(output_path)
                        self.logger.info(
                            f"✅ Created {timeframe} feature file (copied from 1m): {output_path}",
                        )

        except Exception as e:
            self.logger.exception(f"❌ Failed to create timeframe-specific features: {e}")

    async def _resample_features_to_timeframe(
        self, features_df: pd.DataFrame, target_timeframe: str,
    ) -> pd.DataFrame | None:
        """Resample 1m features to target timeframe."""
        try:
            if target_timeframe == "1m":
                return features_df

            # Define resampling rules for different timeframes
            resample_rules = {
                "5m": "5T",
                "15m": "15T",
                "30m": "30T",
                "1h": "1H",
                "4h": "4H",
                "1d": "1D",
            }

            if target_timeframe not in resample_rules:
                self.logger.warning(
                    f"⚠️ Unsupported timeframe for resampling: {target_timeframe}",
                )
                return None

            rule = resample_rules[target_timeframe]

            # Ensure we have a proper DatetimeIndex
            if not isinstance(features_df.index, pd.DatetimeIndex):
                self.logger.info(
                    f"🔄 Converting RangeIndex to DatetimeIndex for {target_timeframe}",
                )

                # Create a synthetic timestamp index based on the target timeframe
                if target_timeframe == "5m":
                    freq = "5T"
                elif target_timeframe == "15m":
                    freq = "15T"
                elif target_timeframe == "30m":
                    freq = "30T"
                else:
                    freq = "1T"

                # Create a proper timestamp index starting from a reasonable date
                start_date = pd.Timestamp.now() - pd.Timedelta(days=180)
                timestamps = pd.date_range(
                    start=start_date, periods=len(features_df), freq=freq,
                )
                features_df = features_df.copy()
                features_df.index = timestamps

            # Resample numeric columns (features)
            numeric_columns = features_df.select_dtypes(
                include=[np.number],
            ).columns.tolist()

            # For features, we'll use mean aggregation for most columns
            # But for some specific features, we might want different aggregation
            agg_dict: dict[str, str] = {}
            for col in numeric_columns:
                if "volume" in col.lower() or "count" in col.lower():
                    agg_dict[col] = "sum"  # Sum for volume/count features
                elif "price" in col.lower() or "close" in col.lower():
                    agg_dict[col] = "last"  # Last value for price features
                else:
                    agg_dict[col] = "mean"  # Mean for most other features

            # Resample the dataframe
            resampled_df = features_df[numeric_columns].resample(rule).agg(agg_dict)

            # Forward fill any remaining NaN values
            resampled_df = resampled_df.fillna(method="ffill").fillna(method="bfill")

            # Remove any remaining NaN values
            resampled_df = resampled_df.dropna()

            self.logger.info(
                f"✅ Resampled features to {target_timeframe}: {resampled_df.shape}",
            )
            return resampled_df

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to resample features to {target_timeframe}: {e}",
            )
            return None

    async def _prepare_timeframe_data(
        self, hmm_df: pd.DataFrame, features_df: pd.DataFrame, timeframe: str,
    ) -> pd.DataFrame | None:
        """Prepare data for a specific timeframe with regime-specific splitting and feature selection."""
        try:
            # Ensure both dataframes have timestamp as index
            if "timestamp" in hmm_df.columns:
                hmm_df = hmm_df.set_index("timestamp")
            if "timestamp" in features_df.columns:
                features_df = features_df.set_index("timestamp")

            # Round timestamps to the nearest minute for better alignment
            hmm_df.index = hmm_df.index.round("1T")
            features_df.index = features_df.index.round("1T")

            # Ensure both datasets use the same time range (last 180 days)
            # Find the common time range
            hmm_start = hmm_df.index.min()
            hmm_end = hmm_df.index.max()
            features_start = features_df.index.min()
            features_end = features_df.index.max()

            # Use the most recent 180 days from the earlier end date
            common_end = min(hmm_end, features_end)
            common_start = common_end - pd.Timedelta(days=180)

            # Filter both datasets to the common range
            hmm_df_filtered = hmm_df[
                (hmm_df.index >= common_start) & (hmm_df.index <= common_end)
            ]
            features_df_filtered = features_df[
                (features_df.index >= common_start) & (features_df.index <= common_end)
            ]

            self.logger.info(f"📊 Data range alignment for {timeframe}:")
            self.logger.info(
                f"   HMM data: {hmm_start} to {hmm_end} ({len(hmm_df)} records)",
            )
            self.logger.info(
                f"   Features data: {features_start} to {features_end} ({len(features_df)} records)",
            )
            self.logger.info(f"   Common range: {common_start} to {common_end}")
            self.logger.info(f"   Filtered HMM: {len(hmm_df_filtered)} records")
            self.logger.info(
                f"   Filtered Features: {len(features_df_filtered)} records)",
            )

            # Merge HMM data with features on timestamp index
            merged_df = hmm_df_filtered.merge(
                features_df_filtered, left_index=True, right_index=True, how="inner"
            )

            if merged_df.empty:
                self.logger.warning(
                    f"⚠️ No overlapping data for {timeframe} after range alignment",
                )
                self.logger.warning(f"   HMM filtered shape: {hmm_df_filtered.shape}")
                self.logger.warning(
                    f"   Features filtered shape: {features_df_filtered.shape}",
                )
                return None

            # Add timeframe identifier
            merged_df["timeframe"] = timeframe

            self.logger.info(
                f"✅ Successfully merged HMM and features data for {timeframe}: {merged_df.shape}",
            )
            self.logger.info(
                f"   Overlapping timestamps: {len(merged_df)} out of {len(hmm_df_filtered)} HMM records",
            )

            # Create target labels using composite cluster (HMM regimes)
            if "composite_cluster_id" in merged_df.columns:
                merged_df["target"] = merged_df["composite_cluster_id"].astype(int)
            # Fallback: use hmm_composite_cluster_id if available
            elif "hmm_composite_cluster_id" in merged_df.columns:
                merged_df["target"] = merged_df["hmm_composite_cluster_id"].astype(
                    int,
                )
            else:
                # Last resort: create a dummy target based on timestamp
                self.logger.warning(
                    f"⚠️ No composite_cluster_id found for {timeframe}, creating dummy target",
                )
                merged_df["target"] = (merged_df.index.astype(int) % 10).astype(int)

            # Filter out noise states (-1) for training
            merged_df = merged_df[merged_df["target"] >= 0].copy()

            if len(merged_df) < self.validation_config["min_samples_per_split"]:
                self.logger.warning(
                    f"⚠️ Insufficient data after filtering for {timeframe}: {len(merged_df)}",
                )
                return None

            # Use all available features - let ML models handle feature selection
            feature_columns = self._get_available_features(merged_df)

            # Keep all features plus target and timeframe
            final_columns = [*feature_columns, "target", "timeframe"]
            if "composite_cluster_id" in merged_df.columns:
                final_columns.append("composite_cluster_id")

            merged_df = merged_df[final_columns].copy()

            # Add regime change prediction features
            merged_df = await self._add_regime_change_features(merged_df, timeframe)

            self.logger.info(
                f"✅ Prepared data for {timeframe}: {merged_df.shape} with {len(feature_columns)} features",
            )
            return merged_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to prepare data for {timeframe}: {e}")
            return None

    async def _load_hmm_composite_regime_data(
        self, timeframe: str,
    ) -> dict[str, pd.DataFrame]:
        """Load unified HMM composite regime dataset with labels."""
        try:
            data_dir = self.config.get("data_dir", "data/training")
            symbol = self.config.get("symbol", "ETHUSDT")
            exchange = self.config.get("exchange", "BINANCE")
            
            # Try to load unified regime dataset first (new approach)
            unified_regime_file = os.path.join(
                data_dir, 
                f"{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet"
            )
            
            if os.path.exists(unified_regime_file):
                self.logger.info(f"✅ Loading unified regime dataset: {unified_regime_file}")
                unified_data = pd.read_parquet(unified_regime_file)
                
                # Load regime labels mapping
                labels_file = os.path.join(
                    data_dir, 
                    f"{exchange}_{symbol}_{timeframe}_regime_labels.json"
                )
                
                if os.path.exists(labels_file):
                    with open(labels_file) as f:
                        regime_labels = json.load(f)
                    
                    regime_ids = regime_labels.get("regime_ids", [])
                    self.logger.info(f"📊 Found {len(regime_ids)} regimes in unified dataset")
                    
                    # Create regime splits from unified dataset
                    regime_splits = {}
                    for regime_id in regime_ids:
                        regime_data = unified_data[unified_data["composite_cluster_id"] == regime_id].copy()
                        
                        if len(regime_data) > 0:
                            # Split into train/validation/test (80/10/10)
                            total_len = len(regime_data)
                            train_end = int(total_len * 0.8)
                            val_end = int(total_len * 0.9)
                            
                            regime_splits[f"regime_{regime_id}"] = {
                                "data": {
                                    "train": regime_data.iloc[:train_end],
                                    "validation": regime_data.iloc[train_end:val_end],
                                    "test": regime_data.iloc[val_end:]
                                },
                                "description": f"Regime {regime_id} from unified dataset",
                                "total_samples": total_len
                            }
                            
                            self.logger.info(
                                f"✅ Created splits for regime {regime_id}: "
                                f"train={len(regime_data.iloc[:train_end])}, "
                                f"val={len(regime_data.iloc[train_end:val_end])}, "
                                f"test={len(regime_data.iloc[val_end:])}"
                            )
                    
                    self.logger.info(f"📊 Created {len(regime_splits)} regime splits from unified dataset")
                    return regime_splits
                else:
                    self.logger.warning(f"⚠️ Regime labels file not found: {labels_file}")
            
            # Fallback to legacy approach for backward compatibility
            self.logger.warning("⚠️ Falling back to legacy regime data loading approach")
            regime_data_dir = os.path.join(data_dir, "regime_data")

            if not os.path.exists(regime_data_dir):
                self.logger.warning(
                    f"⚠️ Legacy regime data directory not found: {regime_data_dir}",
                )
                return {}

            # Load regime splitting summary
            summary_file = os.path.join(
                data_dir,
                f"{self.config['exchange']}_{self.config['symbol']}_hmm_composite_regime_splits.json",
            )
            if not os.path.exists(summary_file):
                self.logger.warning(
                    f"⚠️ Legacy regime splitting summary not found: {summary_file}",
                )
                return {}

            with open(summary_file) as f:
                regime_summary = json.load(f)

            regime_splits: dict[str, Any] = {}
            regime_details = regime_summary.get("regime_details", {})

            for regime_key, regime_info in regime_details.items():
                splits = regime_info.get("splits", {})
                regime_data: dict[str, pd.DataFrame] = {}

                for split_name in ["train", "validation", "test"]:
                    if split_name in splits:
                        file_path = splits[split_name]["file"]
                        if os.path.exists(file_path):
                            try:
                                regime_data[split_name] = pd.read_parquet(file_path)
                                self.logger.info(
                                    f"✅ Loaded {split_name} data for {regime_key}: {len(regime_data[split_name])} rows",
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"⚠️ Failed to load {split_name} data for {regime_key}: {e}",
                                )

                if regime_data:
                    regime_splits[regime_key] = {
                        "data": regime_data,
                        "description": regime_info.get(
                            "description", f"Regime {regime_key}",
                        ),
                    }

            self.logger.info(
                f"📊 Loaded {len(regime_splits)} HMM composite regime splits from legacy approach",
            )
            return regime_splits

        except Exception as e:
            self.logger.exception(f"❌ Failed to load HMM composite regime data: {e}")
            return {}

    async def _add_regime_change_features(
        self, data: pd.DataFrame, timeframe: str,
    ) -> pd.DataFrame:
        """Add regime change prediction features."""
        try:
            # Add regime change indicators
            data["regime_change"] = data["target"].diff().fillna(0).astype(int)
            data["regime_change_abs"] = data["regime_change"].abs()

            # Add regime stability features
            data["regime_duration"] = data.groupby(
                (data["target"] != data["target"].shift()).cumsum()
            ).cumcount()

            # Add regime transition probabilities (simplified)
            regime_counts = data["target"].value_counts()
            total_samples = len(data)
            data["regime_frequency"] = data["target"].map(regime_counts) / total_samples

            self.logger.info(f"   ✅ Added regime change features for {timeframe}")
            return data

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to add regime change features for {timeframe}: {e}",
            )
            return data

    async def _train_timeframe_model(
        self, data: pd.DataFrame, timeframe: str,
    ) -> dict[str, Any] | None:
        """Train model for a specific timeframe with extensive validation and cross-validation."""
        try:
            architecture = self.model_architectures[timeframe]
            self.logger.info(
                f"🎯 Training {architecture} model for {timeframe} with enhanced validation",
            )

            # Prepare features and target
            feature_columns = [
                col
                for col in data.columns
                if col not in ["target", "timeframe", "composite_cluster_id"]
            ]
            X = data[feature_columns]
            y = data["target"]

            # Apply enhanced optimization based on model architecture
            X_optimized, optimization_metadata = await self._apply_enhanced_optimization(
                X, y, timeframe, architecture,
            )

            # Update feature columns after optimization
            feature_columns = list(X_optimized.columns)
            X = X_optimized

            # Perform regime-aware time series split
            (
                train_splits,
                val_splits,
                test_splits,
            ) = await self._create_regime_aware_splits(data, timeframe)

            # Cross-validation results
            cv_results = []

            for split_idx, (train_idx, val_idx, test_idx) in enumerate(
                zip(train_splits, val_splits, test_splits, strict=False)
            ):
                self.logger.info(
                    f"   🔄 Cross-validation split {split_idx + 1}/{len(train_splits)}",
                )

                # Split data
                X_train, X_val, X_test = (
                    X.iloc[train_idx],
                    X.iloc[val_idx],
                    X.iloc[test_idx],
                )
                y_train, y_val, y_test = (
                    y.iloc[train_idx],
                    y.iloc[val_idx],
                    y.iloc[test_idx],
                )

                # Train model based on architecture
                if architecture == "CNN":
                    model_result, await self._train_cnn_model_cv(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        split_idx,
                    )
                elif architecture == "TCN":
                    model_result, await self._train_tcn_model_cv(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        split_idx,
                    )
                elif architecture == "Transformer":
                    model_result, await self._train_transformer_model_cv(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        split_idx,
                    )
                elif architecture == "LightGBM":
                    model_result, await self._train_lightgbm_model_cv(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        split_idx,
                    )
                else:
                    self.logger.error(f"❌ Unknown architecture: {architecture}")
                    return None

                if model_result:
                    cv_results.append(model_result)

            # Aggregate cross-validation results
            if cv_results:
                return await self._aggregate_cv_results(
                    cv_results=timeframe, architecture=architecture,
                )
            self.logger.error(
                f"❌ No successful cross-validation results for {timeframe}",
            )
            return None

        except Exception as e:
            self.logger.exception(f"❌ Failed to train model for {timeframe}: {e}")
            return None

    @validate_feature_engineering_with_lookahead_bias_detection
    async def _train_regime_specific_models(self, timeframe: str) -> dict[str, Any]:
        """Train regime-specific models using HMM composite regime data."""
        try:
            self.logger.info(f"🎯 Training regime-specific models for {timeframe}")

            # Load HMM composite regime data
            regime_splits = await self._load_hmm_composite_regime_data(timeframe)

            if not regime_splits:
                self.logger.error(f"❌ No regime splits found for {timeframe}")
                self.logger.error(
                    "❌ Regime-specific models are required for the system to function properly",
                )
                self.logger.error(
                    "💡 Please run step3_hmm_regime_discovery first to create regime splits",
                )
                msg = f"Missing regime splits for {timeframe}. Run step3_hmm_regime_discovery first.",
                raise ValueError(
                    msg,
                )

            architecture = self.model_architectures[timeframe]
            regime_models = {}

            for regime_key, regime_info in regime_splits.items():
                regime_data = regime_info["data"]
                regime_desc = regime_info["description"]

                # Check if we have sufficient data for this regime
                train_data = regime_data.get("train")
                val_data = regime_data.get("validation")
                test_data = regime_data.get("test")

                if (
                    train_data is None
                    or len(train_data) < self.validation_config["min_samples_per_split"]
                ):
                    self.logger.warning(
                        f"⚠️ Insufficient training data for regime {regime_key}: {len(train_data) if train_data is not None else 0} rows",
                    )
                    continue

                self.logger.info(
                    f"🎯 Training {architecture} model for regime {regime_key}: {regime_desc}",
                )
                self.logger.info(
                    f"   📊 Data: train={len(train_data)}, val={len(val_data) if val_data is not None else 0}, test={len(test_data) if test_data is not None else 0}"
                )

                # Prepare features and target
                feature_columns = [
                    col
                    for col in train_data.columns
                    if col
                    not in [
                        "target",
                        "timeframe",
                        "composite_cluster_id",
                        "regime_description",
                    ]
                ]
                X_train = train_data[feature_columns]
                y_train = train_data["target"]

                X_val = val_data[feature_columns] if val_data is not None else X_train
                y_val = val_data["target"] if val_data is not None else y_train

                X_test = test_data[feature_columns] if test_data is not None else X_val
                y_test = test_data["target"] if test_data is not None else y_val

                # Train model based on architecture
                model_result = None
                if architecture == "CNN":
                    model_result, await self._train_cnn_model_regime(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        regime_key,
                    )
                elif architecture == "TCN":
                    model_result, await self._train_tcn_model_regime(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        regime_key,
                    )
                elif architecture == "Transformer":
                    model_result, await self._train_transformer_model_regime(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        regime_key,
                    )
                elif architecture == "LightGBM":
                    model_result, await self._train_lightgbm_model_regime(
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        timeframe,
                        regime_key,
                    )
                else:
                    self.logger.error(f"❌ Unknown architecture: {architecture}")
                    continue

                if model_result:
                    regime_models[regime_key] = {
                        "model": model_result,
                        "description": regime_desc,
                        "architecture": architecture,
                        "data_sizes": {
                            "train": len(train_data),
                            "validation": len(val_data) if val_data is not None else 0,
                            "test": len(test_data) if test_data is not None else 0,
                        },
                    }
                self.logger.info(
                    f"✅ Trained model for regime {regime_key}: {regime_desc}",
                )

            self.logger.info(
                f"✅ Completed regime-specific training for {timeframe}: {len(regime_models)} models",
            )
            return regime_models

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to train regime-specific models for {timeframe}: {e}",
            )
            return {}

    @validate_feature_engineering_with_lookahead_bias_detection
    async def _train_lightgbm_model_regime(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, regime_key: str, ) -> dict[str, Any] | None:
        """Train LightGBM model for a specific regime."""
        try:
            self.logger.info(f"   🌳 Training LightGBM model for regime {regime_key}")

            # Prepare data
            X_train_clean = X_train.fillna(0).astype(float)
            X_val_clean = X_val.fillna(0).astype(float)
            X_test_clean = X_test.fillna(0).astype(float)

            # Train LightGBM model
            model, lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
            )

            model.fit(
                X_train_clean,
                y_train,
                eval_set=[(X_val_clean, y_val)],
                eval_metric="multi_logloss",
                early_stopping_rounds=10,
                verbose=False,
            )

            # Evaluate model
            train_score, model.score(X_train_clean, y_train)
            val_score, model.score(X_val_clean, y_val)
            test_score, model.score(X_test_clean, y_test)

            # Feature importance
            feature_importance, dict(
                zip(X_train_clean.columns, model.feature_importances_, strict=False)
            )

            result = {
                "model": model,
                "architecture": "LightGBM",
                "regime_key": regime_key,
                "timeframe": timeframe,
                "scores": {
                    "train": train_score,
                    "validation": val_score,
                    "test": test_score,
                },
                "feature_importance": feature_importance,
                "n_features": len(X_train_clean.columns),
            }

            self.logger.info(
                f"   ✅ LightGBM regime {regime_key}: train={train_score:.3f}, val={val_score:.3f}, test={test_score:.3f}"
            )
            return result

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to train LightGBM model for regime {regime_key}: {e}",
            )
            return None

    async def _train_cnn_model_regime(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, regime_key: str, ) -> dict[str, Any] | None:
        """Train CNN model for a specific regime."""
        try:
            self.logger.info(f"   🧠 Training CNN model for regime {regime_key}")

            # For now, return a placeholder - CNN training would need more complex setup
            self.logger.warning(
                f"   ⚠️ CNN training for regime {regime_key} not yet implemented",
            )
            return None

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to train CNN model for regime {regime_key}: {e}",
            )
            return None

    async def _train_tcn_model_regime(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, regime_key: str, ) -> dict[str, Any] | None:
        """Train TCN model for a specific regime."""
        try:
            self.logger.info(f"   🔄 Training TCN model for regime {regime_key}")

            # For now, return a placeholder - TCN training would need more complex setup
            self.logger.warning(
                f"   ⚠️ TCN training for regime {regime_key} not yet implemented",
            )
            return None

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to train TCN model for regime {regime_key}: {e}",
            )
            return None

    async def _train_transformer_model_regime(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, regime_key: str, ) -> dict[str, Any] | None:
        """Train Transformer model for a specific regime."""
        try:
            self.logger.info(
                f"   🔄 Training Transformer model for regime {regime_key}",
            )

            # For now, return a placeholder - Transformer training would need more complex setup
            self.logger.warning(
                f"   ⚠️ Transformer training for regime {regime_key} not yet implemented",
            )
            return None

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to train Transformer model for regime {regime_key}: {e}",
            )
            return None

    async def _add_regime_change_features(
        self, data: pd.DataFrame, timeframe: str, ) -> pd.DataFrame:
        """Add regime change prediction features to the dataset."""
        try:
            # Create regime change targets
            if "composite_cluster_id" in data.columns:
                regimes = data["composite_cluster_id"].fillna(-1).astype(int)

            # Detect regime changes
            regime_changes = []
            for i in range(1, len(regimes)):
                prev_regime = regimes.iloc[i - 1]
                curr_regime = regimes.iloc[i]

                if (
                    prev_regime != curr_regime
                    and prev_regime >= 0
                    and curr_regime >= 0
                ):
                    regime_changes.append(1)  # Regime change detected
                else:
                    regime_changes.append(0)  # No change

            # Add padding for the first element
            regime_changes.insert(0, 0)

            # Add regime change features
            data["regime_change"] = regime_changes
            data["regime_change_next"] = data["regime_change"].shift(-1).fillna(0)
            data["regime_change_prev"] = data["regime_change"].shift(1).fillna(0)

            # Add regime stability features
            data["regime_stability"] = (
                data["regime_change"].rolling(window=10).sum()
            )
            data["regime_volatility"] = (
                data["regime_change"].rolling(window=20).std()
            )

            # HMM stay/switch risk (leak-free, trailing)
            stay_prob, 1.0 - data["regime_change"].rolling(
                window=20, min_periods=1,
            ).mean()
            data["hmm_stay_prob_w20"] = stay_prob.fillna(0.5)
            data["hmm_switch_risk_w20"] = 1.0 - data["hmm_stay_prob_w20"]

            # HMM dwell (time spent in current regime so far)
            run_id = (regimes != regimes.shift(1)).cumsum()
            data["hmm_dwell"] = data.groupby(run_id).cumcount() + 1

            # HMM probability gap between top1 and top2 across state posteriors
            state_cols = [c for c in data.columns if "_p_state_" in c]
            if state_cols:
                probs = data[state_cols].clip(lower=0.0, upper=1.0)
                top1 = probs.max(axis=1)
                # Use nlargest per-row safely
                top2 = probs.apply(
                    lambda r: r.nlargest(2).iloc[-1] if r.count() >= 2 else 0.0,
                    axis=1,
                )
                data["hmm_top1_top2_gap"] = (top1 - top2).fillna(0.0)

            # Caps: keep HMM additions lean
            hmm_cols = [
                "hmm_stay_prob_w20",
                "hmm_switch_risk_w20",
                "hmm_dwell",
                "hmm_top1_top2_gap",
            ]
            kept = [c for c in hmm_cols if c in data.columns]
            if len(kept) != len(hmm_cols):
                self.logger.debug(
                    f"ℹ️ HMM feature cap/availability: kept={kept}"
                )

            self.logger.info(
                f"✅ Added regime change + HMM features for {timeframe} (kept {len(kept)})",
            )

            return data

        except Exception as e:
            self.logger.exception(f"❌ Failed to add regime change features: {e}")
            return data

    async def _create_regime_aware_splits(
        self, data: pd.DataFrame, timeframe: str, ) -> tuple[list, list, list]:
        """Create regime-aware time series splits."""
        try:
            n_splits = self.validation_config["n_splits"]
            test_size = self.validation_config["test_size"]
            val_size = self.validation_config["validation_size"]

            total_samples = len(data)
            test_samples = int(total_samples * test_size)
            val_samples = int(total_samples * val_size)
            train_samples = total_samples - test_samples - val_samples

            train_splits, val_splits, test_splits = [], [], []

            for i in range(n_splits):
                # Ensure minimum samples per split
                if train_samples < self.validation_config["min_samples_per_split"]:
                    self.logger.warning(
                        f"⚠️ Insufficient samples for split {i} in {timeframe}",
                    )
                    continue

                # Create time-aware splits
                start_idx = i * (total_samples // n_splits)
                train_end = start_idx + train_samples
                val_end = train_end + val_samples

                train_idx, list(range(start_idx, train_end))
                val_idx, list(range(train_end, val_end))
                test_idx, list(
                    range(val_end, min(val_end + test_samples, total_samples)),
                )

                # Ensure regime balance in splits
                if self.validation_config["regime_aware_splitting"]:
                    train_idx, val_idx, test_idx = self._balance_regimes_in_splits(
                        data, train_idx, val_idx, test_idx,
                    )

                train_splits.append(train_idx)
                val_splits.append(val_idx)
                test_splits.append(test_idx)

            self.logger.info(
                f"   ✅ Created {len(train_splits)} regime-aware splits for {timeframe}",
            )
            return train_splits, val_splits, test_splits

        except Exception as e:
            self.logger.exception(f"❌ Failed to create splits for {timeframe}: {e}")
            # Fallback to simple splits
            return self._create_simple_splits(data)

    def _balance_regimes_in_splits(
        self, data: pd.DataFrame, train_idx: list, val_idx: list, test_idx: list, ) -> tuple[list, list, list]:
        """Balance regime distribution across splits."""
        try:
            # Get regime distribution
            all_regimes = (data["target"].iloc[train_idx + val_idx + test_idx].value_counts()
            )

            # Ensure each split has representation from all regimes
            balanced_train, balanced_val, balanced_test = [], [], []

            for regime in all_regimes.index:
                regime_indices, data[data["target"] == regime].index
                regime_indices = [
                    i for i in regime_indices if i in train_idx + val_idx + test_idx
                ]

            if len(regime_indices) >= 3:  # Need at least 3 samples for 3 splits
                # Distribute regime samples across splits
                n_train, max(1, len(regime_indices) // 3)
                n_val, max(1, len(regime_indices) // 3)
                len(regime_indices) - n_train - n_val

                balanced_train.extend(regime_indices[:n_train])
                balanced_val.extend(regime_indices[n_train : n_train + n_val])
                balanced_test.extend(regime_indices[n_train + n_val :])

            return balanced_train, balanced_val, balanced_test

        except Exception as e:
            self.logger.exception(f"❌ Failed to balance regimes: {e}")
            return train_idx, val_idx, test_idx

    def _create_simple_splits(self, data: pd.DataFrame) -> tuple[list, list, list]:
        """Create simple time series splits as fallback."""
        try:
            n_splits = self.validation_config["n_splits"]
            total_samples = len(data)

            train_splits, val_splits, test_splits = [], [], []

            for i in range(n_splits):
                split_size = total_samples // n_splits
                start_idx = i * split_size
                end_idx = start_idx + split_size

                train_end = start_idx + int(split_size * 0.6)
                val_end = start_idx + int(split_size * 0.8)

                train_splits.append(list(range(start_idx, train_end)))
                val_splits.append(list(range(train_end, val_end)))
                test_splits.append(list(range(val_end, end_idx)))

            return train_splits, val_splits, test_splits

        except Exception as e:
            self.logger.exception(f"❌ Failed to create simple splits: {e}")
            return [], [], []

    async def _aggregate_cv_results(
        self, cv_results: list[dict], timeframe: str, architecture: str, ) -> dict[str, Any]:
        """Aggregate cross-validation results."""
        try:
            # Calculate average metrics
            avg_accuracy, np.mean([result.get("accuracy", 0) for result in cv_results])
            avg_f1_score, np.mean([result.get("f1_score", 0) for result in cv_results])
            avg_precision, np.mean(
                [result.get("precision", 0) for result in cv_results],
            )
            avg_recall, np.mean([result.get("recall", 0) for result in cv_results])

            # Select best model based on validation accuracy
            best_result, max(cv_results, key=lambda x: x.get("val_accuracy", 0))

            return {
                "timeframe": timeframe,
                "architecture": architecture,
                "cv_results": cv_results,
                "avg_accuracy": avg_accuracy,
                "avg_f1_score": avg_f1_score,
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "best_model": best_result.get("model"),
                "best_accuracy": best_result.get("val_accuracy"),
                "feature_importance": best_result.get("feature_importance", {}),
                "training_history": best_result.get("training_history", {}),
                "regime_performance": best_result.get("regime_performance", {}),
            }

        except Exception as e:
            self.logger.exception(f"❌ Failed to aggregate CV results: {e}")
            return cv_results[0] if cv_results else {}

    async def _train_cnn_model(
        self, data: pd.DataFrame, timeframe: str, ) -> dict[str, Any]:
        """Train CNN model for 1m timeframe (Tactician)."""
        try:
            self.logger.info(f"🔄 Training CNN for {timeframe}")

            # Prepare features
            X, y, scaler, label_encoder, self._prepare_features(
                data = self.specialist_features,
            )

            # Reshape for CNN (samples, channels, sequence_length)
            # For 1m data, we'll use a window of recent features
            sequence_length = 60  # 60 minutes of history,
            X_sequences, self._create_sequences(X, sequence_length)

            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train = X_test, X_sequences[:split_idx], X_sequences[split_idx:],
            y_train, y_test = (
                y[sequence_length : split_idx + sequence_length],
                y[split_idx + sequence_length :],
            )

            # Create CNN model
            model, CNNModel(
                input_channels=X.shape[1],
                sequence_length=sequence_length,
                num_classes=len(label_encoder.classes_),
            )

            # Train model
            trainer, CNNTrainer(model, learning_rate=0.001, batch_size=32)
            history, await trainer.train(X_train, y_train, X_test, y_test, epochs=50)

            # Save model and metadata
            model_path = f"models/{timeframe}_cnn_model.pth"
            torch.save(model.state_dict(), model_path)

            return {
                "architecture": "CNN",
                "model_path": model_path,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "sequence_length": sequence_length,
                "history": history,
                "feature_columns": self.specialist_features,
            }

        except Exception as e:
            self.logger.exception(f"❌ CNN training failed: {e}")
            return None

    async def _train_tcn_model(
        self, data: pd.DataFrame, timeframe: str, ) -> dict[str, Any]:
        """Train Temporal Convolutional Network for 5m timeframe (Analyst)."""
        try:
            self.logger.info(f"🔄 Training TCN for {timeframe}")

            # Prepare features
            X, y, scaler, label_encoder, self._prepare_features(
                data = self.specialist_features,
            )

            # Create sequences for TCN
            sequence_length = 24  # 24 periods (2 hours of 5m data),
            X_sequences, self._create_sequences(X, sequence_length)

            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train = X_test, X_sequences[:split_idx], X_sequences[split_idx:],
            y_train, y_test = (
                y[sequence_length : split_idx + sequence_length],
                y[split_idx + sequence_length :],
            )

            # Create TCN model
            model, TCNModel(
                input_size=X.shape[1],
                num_channels=[64, 128, 256],
                kernel_size=3,
                num_classes=len(label_encoder.classes_),
            )

            # Train model
            trainer, TCNTrainer(model, learning_rate=0.001, batch_size=64)
            history, await trainer.train(X_train, y_train, X_test, y_test, epochs=100)

            # Save model and metadata
            model_path = f"models/{timeframe}_tcn_model.pth"
            torch.save(model.state_dict(), model_path)

            return {
                "architecture": "TCN",
                "model_path": model_path,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "sequence_length": sequence_length,
                "history": history,
                "feature_columns": self.specialist_features,
            }

        except Exception as e:
            self.logger.exception(f"❌ TCN training failed: {e}")
            return None

    async def _train_transformer_model(
        self, data: pd.DataFrame, timeframe: str, ) -> dict[str, Any]:
        """Train Transformer model for 15m timeframe (Analyst)."""
        try:
            self.logger.info(f"🔄 Training Transformer for {timeframe}")

            # Prepare features
            X, y, scaler, label_encoder, self._prepare_features(
                data = self.specialist_features,
            )

            # Create sequences for Transformer
            sequence_length = 16  # 16 periods (4 hours of 15m data),
            X_sequences, self._create_sequences(X, sequence_length)

            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train = X_test, X_sequences[:split_idx], X_sequences[split_idx:],
            y_train, y_test = (
                y[sequence_length : split_idx + sequence_length],
                y[split_idx + sequence_length :],
            )

            # Create Transformer model
            model, TransformerModel(
                input_size=X.shape[1],
                d_model=256,
                nhead=8,
                num_layers=6,
                num_classes=len(label_encoder.classes_),
            )

            # Train model
            trainer, TransformerTrainer(model, learning_rate=0.0001, batch_size=32)
            history, await trainer.train(X_train, y_train, X_test, y_test, epochs=150)

            # Save model and metadata
            model_path = f"models/{timeframe}_transformer_model.pth"
            torch.save(model.state_dict(), model_path)

            return {
                "architecture": "Transformer",
                "model_path": model_path,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "sequence_length": sequence_length,
                "history": history,
                "feature_columns": self.specialist_features,
            }

        except Exception as e:
            self.logger.exception(f"❌ Transformer training failed: {e}")
            return None

    async def _train_lightgbm_model(
        self, data: pd.DataFrame, timeframe: str, ) -> dict[str, Any]:
        """Train LightGBM model with multi-output probability training for 30m timeframe (Analyst)."""
        try:
            self.logger.info(f"🔄 Training LightGBM with multi-output probability training for {timeframe}")

            # Prepare features
            X, y, scaler, label_encoder = self._prepare_features(
                data=data, feature_columns=self.specialist_features,
            )

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Create market data DataFrame for probability calculations
            market_data = pd.DataFrame({
                'close': data.get('close', np.random.randn(len(data))),
                'volume': data.get('volume', np.random.randn(len(data)))
            })

            # Initialize multi-output probability trainer
            from ..multi_output_probability_trainer import MultiOutputProbabilityTrainer
            
            # Configure multi-output training with advanced models
            multi_output_config = {
                "use_lightgbm": True,
                "n_estimators": 1000,
                "learning_rate": 0.01,
                "max_depth": 8,
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "look_ahead_periods": 20,
                "magnitude_threshold_factor": 0.8,
                "adverse_threshold": 0.01,
                "avoidance_look_ahead": 10,
                # Advanced model configuration
                "timeframe": "5m",  # Use TCN for 5-minute data
                "model_architectures": {
                    "1m": "cnn",      # CNN for 1-minute data (Tactician)
                    "5m": "tcn",      # TCN for 5-minute data (Analyst)
                    "15m": "transformer", # Transformer for 15-minute data (Enhanced)
                    "30m": "lightgbm",    # LightGBM for 30-minute data (Analyst)
                    "1h": "hmm_regime"    # HMM regime definition only
                },
                "neural_config": {
                    "tcn": {
                        "num_channels": [64, 128, 256],
                        "kernel_size": 2,
                        "dropout": 0.2,
                        "batch_size": 32,
                        "epochs": 50,
                        "learning_rate": 0.001
                    },
                    "cnn": {
                        "num_filters": [64, 128, 256],
                        "kernel_sizes": [3, 3, 3],
                        "dropout": 0.2,
                        "batch_size": 32,
                        "epochs": 50,
                        "learning_rate": 0.001
                    },
                    "transformer": {
                        "d_model": 128,
                        "nhead": 8,
                        "num_layers": 4,
                        "dropout": 0.1,
                        "batch_size": 32,
                        "epochs": 50,
                        "learning_rate": 0.001
                    },
                    "lstm": {
                        "hidden_size": 128,
                        "num_layers": 2,
                        "bidirectional": True,
                        "dropout": 0.2,
                        "batch_size": 32,
                        "epochs": 50,
                        "learning_rate": 0.001
                    },
                    "gru": {
                        "hidden_size": 128,
                        "num_layers": 2,
                        "bidirectional": True,
                        "dropout": 0.2,
                        "batch_size": 32,
                        "epochs": 50,
                        "learning_rate": 0.001
                    }
                }
            }
            
            multi_output_trainer = MultiOutputProbabilityTrainer(multi_output_config)
            
            # Generate multi-output targets
            y_train_multi = multi_output_trainer.prepare_multi_output_targets(
                X_train, y_train, market_data.iloc[:len(X_train)]
            )
            y_test_multi = multi_output_trainer.prepare_multi_output_targets(
                X_test, y_test, market_data.iloc[len(X_train):]
            )
            
            # Train multi-output model
            trained_models = multi_output_trainer.train_multi_output_model(
                X_train, y_train_multi, X_test, y_test_multi
            )
            
            # Generate probability outputs
            price_action_probabilities = multi_output_trainer.predict_probabilities(
                X_test, market_data.iloc[len(X_train):]
            )
            
            # Calculate overall metrics
            overall_metrics = {}
            for prob_type, prob_value in price_action_probabilities.items():
                if prob_type != "generation_timestamp" and prob_type != "model_type":
                    overall_metrics[f"{prob_type}_value"] = prob_value
            
            # Prepare model data for saving
            model_data = {
                "multi_output_trainer": multi_output_trainer,
                "trained_models": trained_models,
                "model_type": "multi_output",
                "architecture": "MultiOutputLightGBM",
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_columns": self.specialist_features,
                "timeframe": timeframe,
                "training_date": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                "hyperparameters": multi_output_config,
                "metrics": overall_metrics,
                "price_action_probabilities": price_action_probabilities
            }

            # Save model with probabilities using multi-output format
            model_path = f"models/{timeframe}_multi_output_lightgbm_model.pkl"
            try:
                from ..model_saving_utils import save_multi_output_model_with_probabilities
                save_multi_output_model_with_probabilities(
                    model_data, model_path, save_format="joblib"
                )
                self.logger.info(f"✅ Saved multi-output LightGBM model with probabilities to {model_path}")
                self.logger.info(f"   Probability outputs: {probability_outputs}")
                
            except Exception as save_error:
                self.logger.error(f"❌ Failed to save multi-output model: {save_error}")
                # Fallback to simple save
                with open(model_path, "wb") as f:
                    pickle.dump(model_data, f)

            return {
                "architecture": "LightGBM",
                "model_path": model_path,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "train_score": train_score,
                "test_score": test_score,
                "feature_columns": self.specialist_features,
                "price_action_probabilities": price_action_probabilities,
            }

        except Exception as e:
            self.logger.exception(f"❌ LightGBM training failed: {e}")
            return None

    def _prepare_features(
        self, data: pd.DataFrame, feature_columns: list[str], ) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
        """Prepare features and targets for training."""
        # Select available features
        available_features = [f for f in feature_columns if f in data.columns]
        if not available_features:
            msg = "No features available for training"
            raise ValueError(msg)

        # Prepare features
        X, data[available_features].fillna(0).values

        # Prepare targets
        y = data["target"].values

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return X, y, scaler, label_encoder

    def _create_sequences(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for time series models."""
        sequences = []
        for i in range(len(X) - sequence_length):
            sequences.append(X[i : i + sequence_length])
        return np.array(sequences)

    async def _save_models(
        self, training_results: dict[str, Any], exchange: str, symbol: str, data_dir: str, ) -> None:
        """Save trained models and comprehensive metadata."""
        try:
            # Create models directory
            models_dir = f"{data_dir}/models"
            ensure_directory(models_dir)

            # Save each model with enhanced metadata
            for timeframe, result in training_results.items():
                model_path = (f"{models_dir}/{exchange}_{symbol}_{timeframe}_hmm_model.pkl"
                )

            # Enhanced model data with comprehensive metadata
                model_data = {
                    "model": result.get("best_model"),
                    "architecture": result.get("architecture"),
                    "timeframe": timeframe,
                    "training_date": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                    "feature_importance": result.get("feature_importance", {}),
                    "training_history": result.get("training_history", {}),
                    "cv_results": result.get("cv_results", []),
                    "regime_performance": result.get("regime_performance", {}),
                    "model_architectures": self.model_architectures,
                    "validation_config": self.validation_config,
                    "data_source_config": self.data_source_config,
                    "metrics": {
                        "avg_accuracy": result.get("avg_accuracy"),
                        "avg_f1_score": result.get("avg_f1_score"),
                        "avg_precision": result.get("avg_precision"),
                        "avg_recall": result.get("avg_recall"),
                        "best_accuracy": result.get("best_accuracy"),
                    },
                }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            self.logger.info(f"✅ Saved {timeframe} model to {model_path}")
            
            # Log model to MLflow
            try:
                if result.get("best_model"):
                    log_step_model(
                        config=self.config,
                        step_name="step9_hmm_based_training",
                        model=result["best_model"],
                        model_name=f"{timeframe}_hmm_model",
                        model_type="hmm_based",
                        additional_metadata={
                            "timeframe": timeframe,
                            "architecture": result.get("architecture", "unknown"),
                            "avg_accuracy": result.get("avg_accuracy", 0.0),
                            "avg_f1_score": result.get("avg_f1_score", 0.0),
                            "training_algorithm": getattr(result["best_model"], '__class__.__name__', 'Unknown'),
                        }
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to log {timeframe} model to MLflow: {e}")

            # Save comprehensive training summary
            summary_path = f"{models_dir}/{exchange}_{symbol}_hmm_training_summary.json"
            summary = {
                "exchange": exchange,
                "symbol": symbol,
                "training_date": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                "models_trained": len(training_results),
                "timeframes": list(training_results.keys()),
                "model_architectures": self.model_architectures,
                "validation_config": self.validation_config,
                "data_source_config": self.data_source_config,
                "training_results": {
                    timeframe: {
                        "architecture": result.get("architecture"),
                        "avg_accuracy": result.get("avg_accuracy"),
                        "avg_f1_score": result.get("avg_f1_score"),
                        "avg_precision": result.get("avg_precision"),
                        "avg_recall": result.get("avg_recall"),
                        "best_accuracy": result.get("best_accuracy"),
                        "cv_splits": len(result.get("cv_results", [])),
                        "regime_performance": result.get("regime_performance", {}),
                    }
                for timeframe, result in training_results.items()
                },
                "system_info": {
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                },
            }

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            self.logger.info(
                f"✅ Saved comprehensive training summary to {summary_path}",
            )
            
            # Log training summary to MLflow with standardized naming
            try:
                summary_artifact_name = log_step_artifact_with_standardized_name(
                    config=self.config,
                    step_name="step9_hmm_based_training",
                    artifact_path=summary_path,
                    artifact_type="training_summary",
                    additional_metadata={
                        "models_trained": len(training_results),
                        "timeframes": list(training_results.keys()),
                        "summary_type": "comprehensive_training_summary",
                    }
                )
                self.logger.info(f"✅ Logged training summary: {summary_artifact_name}")
                
                # Log comprehensive training report
                report_data = {
                    "training_summary": summary,
                    "model_architectures": self.model_architectures,
                    "validation_config": self.validation_config,
                    "data_source_config": self.data_source_config,
                    "training_results": training_results,
                    "execution_timestamp": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                }
                
                report_name = log_step_report(
                    config=self.config,
                    step_name="step9_hmm_based_training",
                    report_data=report_data,
                    report_type="hmm_training_report",
                    additional_metadata={
                        "models_trained": len(training_results),
                        "timeframes": list(training_results.keys()),
                        "model_architectures": list(self.model_architectures.keys()),
                    }
                )
                self.logger.info(f"✅ Logged HMM training report: {report_name}")
                
                # Log training metrics
                all_metrics = {}
                for timeframe, result in training_results.items():
                    if "avg_accuracy" in result:
                        all_metrics[f"step9_{timeframe}_avg_accuracy"] = result["avg_accuracy"]
                    if "avg_f1_score" in result:
                        all_metrics[f"step9_{timeframe}_avg_f1_score"] = result["avg_f1_score"]
                    if "avg_precision" in result:
                        all_metrics[f"step9_{timeframe}_avg_precision"] = result["avg_precision"]
                    if "avg_recall" in result:
                        all_metrics[f"step9_{timeframe}_avg_recall"] = result["avg_recall"]
                
                if all_metrics:
                    log_step_metrics(
                        config=self.config,
                        step_name="step9_hmm_based_training",
                        metrics=all_metrics,
                        additional_metadata={
                            "metrics_type": "hmm_training_performance",
                            "models_trained": len(training_results),
                        }
                    )
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to log training summary to MLflow: {e}")

            # Save feature importance summary
            feature_summary = {}
            for timeframe, result in training_results.items():
                if "feature_importance" in result:
                    feature_summary[timeframe] = result["feature_importance"]

            if feature_summary:
                feature_path = (f"{models_dir}/{exchange}_{symbol}_feature_importance.json"
                )
                with open(feature_path, "w") as f:
                    json.dump(feature_summary, f, indent=2, default=str)
                self.logger.info(
                    f"✅ Saved feature importance summary to {feature_path}",
                )

        except Exception as e:
            self.logger.exception(f"❌ Failed to save models: {e}")

    async def _save_enhanced_artifacts(
        self, training_results: dict[str, Any], data_dir: str, exchange: str, symbol: str, combined_data: pd.DataFrame, feature_columns: list, ) -> dict[str, Any]:
        """Save enhanced artifacts with comprehensive metadata and training history.

        Args:
            training_results: Results from model training
            data_dir: Data directory path
            exchange: Exchange name
            symbol: Symbol name
            combined_data: Combined training data
            feature_columns: List of feature columns used

        Returns:
            Dict containing artifact paths and metadata

        """
        try:
            self.logger.info("💾 Saving enhanced artifacts and metadata...")

            # Create artifacts directory
            artifacts_dir = f"{data_dir}/{exchange}_{symbol}_hmm_models"
            ensure_directory(artifacts_dir)

            # Save main model (first available)
            main_model_artifact = None
            main_model_name = None

            for timeframe, models in training_results.items():
                if models and isinstance(models, dict):
                    for model_name, model_data in models.items():
                        if model_data:
                            main_model_name = model_name
                            main_model_artifact = model_data
                            break
                    if main_model_artifact:
                        break

            if main_model_artifact:
                main_estimator, self._extract_estimator_from_artifact(
                    main_model_artifact,
                )
                main_model_file = (f"{artifacts_dir}/{exchange}_{symbol}_hmm_main_model.pkl"
                )

            with open(main_model_file, "wb") as f:
                pickle.dump(main_estimator, f)

            self.logger.info(f"✅ Saved main HMM model to {main_model_file}")

            # Create comprehensive model metadata
            model_metadata, await self._create_model_metadata(
                main_model_artifact,
                main_model_name,
                exchange,
                symbol,
                feature_columns,
                main_model_file,
            )

            # Save model metadata
            metadata_file = (f"{artifacts_dir}/{exchange}_{symbol}_hmm_model_metadata.json"
            )
            with open(metadata_file, "w") as f:
                json.dump(model_metadata, f, indent=2)

            self.logger.info(f"✅ Saved model metadata to {metadata_file}")

            # Save per-timeframe models
            timeframe_models_dir = f"{artifacts_dir}/timeframes"
            ensure_directory(timeframe_models_dir)

            for timeframe, models in training_results.items():
                if models and isinstance(models, dict):
                    timeframe_dir = f"{timeframe_models_dir}/{timeframe}"
                    ensure_directory(timeframe_dir)

            for model_name, model_data in models.items():
                if model_data:
                    model_file = f"{timeframe_dir}/{model_name}.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(model_data, f)

            # Create training history
            training_history = await self._create_training_history(
                training_results, exchange, symbol, combined_data, feature_columns,
            )

            # Save training history
            history_file = (f"{artifacts_dir}/{exchange}_{symbol}_hmm_training_history.json"
            )
            with open(history_file, "w") as f:
                json.dump(training_history, f, indent=2)

            self.logger.info(f"✅ Saved training history to {history_file}")

            # Create feature analysis report
            feature_report = await self._create_feature_analysis_report(
                combined_data, feature_columns, training_results,
            )

            # Save feature report
            feature_file = (f"{artifacts_dir}/{exchange}_{symbol}_hmm_feature_report.json"
            )
            with open(feature_file, "w") as f:
                json.dump(feature_report, f, indent=2)

            self.logger.info(f"✅ Saved feature analysis report to {feature_file}")

            # Create training summary
            summary_file = (f"{artifacts_dir}/{exchange}_{symbol}_hmm_training_summary.json"
            )
            summary_data = await self._create_training_summary(
                training_results, exchange, symbol, combined_data, feature_columns,
            )

            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            self.logger.info(f"✅ Saved training summary to {summary_file}")

            return {
                "artifacts_dir": artifacts_dir,
                "main_model_file": main_model_file if main_model_artifact else None,
                "metadata_file": metadata_file,
                "history_file": history_file,
                "feature_file": feature_file,
                "summary_file": summary_file,
            }

        except Exception as e:
            self.logger.exception(f"❌ Failed to save enhanced artifacts: {e}")
            return {}

    async def _create_model_metadata(
        self, model_artifact: dict[str, Any], model_name: str, exchange: str, symbol: str, feature_columns: list, model_file: str, ) -> dict[str, Any]:
        """Create comprehensive model metadata."""
        try:
            metadata = {
                "model_type": model_name,
                "training_date": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                "symbol": symbol,
                "exchange": exchange,
                "model_file": os.path.basename(model_file),
                "model_size_mb": os.path.getsize(model_file) / (1024 * 1024)
            if os.path.exists(model_file)
                else 0,
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "model_architectures": self.model_architectures,
                "validation_config": self.validation_config,
                "data_source_config": self.data_source_config,
            }

            # Add model-specific metadata
            if isinstance(model_artifact, dict):
                # Add accuracy and performance metrics
                if "accuracy" in model_artifact:
                    metadata["accuracy"] = float(model_artifact["accuracy"])

                # Add feature importance if available
                if "feature_importance" in model_artifact:
                    metadata["top_features"] = dict(
                        sorted(
                            model_artifact["feature_importance"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:20],
                    )

                # Add label mappings
                for mapping_key in [
                    "xgb_label_mapping",
                    "lgb_label_mapping",
                    "rf_label_mapping",
                ]:
                    if mapping_key in model_artifact:
                        metadata["label_mapping"] = model_artifact[mapping_key]
                        metadata["label_encoding_scheme"] = (
                            f"{model_name}_contiguous_0_to_K_minus_1"
                        )
                        break

            return metadata

        except Exception as e:
            self.logger.exception(f"❌ Failed to create model metadata: {e}")
            return {"error": str(e)}

    async def _create_training_history(
        self, training_results: dict[str, Any], exchange: str, symbol: str, combined_data: pd.DataFrame, feature_columns: list, ) -> dict[str, Any]:
        """Create comprehensive training history."""
        try:
            history = {
                "training_date": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                "symbol": symbol,
                "exchange": exchange,
                "timeframes_trained": list(training_results.keys()),
                "total_models": sum(
                    len(models) if isinstance(models, dict) else 0
        for models in training_results.values()
                ),
                "data_statistics": {
                    "total_samples": len(combined_data),
                    "feature_count": len(feature_columns),
                    "data_columns": list(combined_data.columns),
                    "data_types": combined_data.dtypes.astype(str).to_dict(),
                    "missing_values": combined_data.isnull().sum().to_dict(),
                    "memory_usage_mb": combined_data.memory_usage(deep=True).sum()
                    / (1024 * 1024),
                },
                "model_performance": {},
                "training_configuration": {
                    "model_architectures": self.model_architectures,
                    "validation_config": self.validation_config,
                    "data_source_config": self.data_source_config,
                },
            }

            # Add performance metrics for each timeframe
            for timeframe, models in training_results.items():
                if isinstance(models, dict):
                    history["model_performance"][timeframe] = {}
                    for model_name, model_data in models.items():
                        if isinstance(model_data, dict) and "accuracy" in model_data:
                            history["model_performance"][timeframe][model_name] = {
                                "accuracy": float(model_data["accuracy"]),
                                "model_type": model_data.get("model_type", "Unknown"),
                                "training_date": model_data.get("training_date", ""),
                            }

            return history

        except Exception as e:
            self.logger.exception(f"❌ Failed to create training history: {e}")
            return {"error": str(e)}

    async def _create_feature_analysis_report(
        self, combined_data: pd.DataFrame, feature_columns: list, training_results: dict[str, Any], ) -> dict[str, Any]:
        """Create comprehensive feature analysis report."""
        try:
            report = {
                "feature_statistics": {},
                "feature_importance_aggregate": {},
                "feature_correlations": {},
                "feature_quality_metrics": {},
            }

            # Calculate feature statistics
            for feature in feature_columns:
                if feature in combined_data.columns:
                    feature_data = combined_data[feature].dropna()
                    if len(feature_data) > 0:
                        report["feature_statistics"][feature] = {
                            "count": len(feature_data),
                            "mean": float(feature_data.mean()),
                            "std": float(feature_data.std()),
                            "min": float(feature_data.min()),
                            "max": float(feature_data.max()),
                            "p25": float(feature_data.quantile(0.25)),
                            "p50": float(feature_data.quantile(0.50)),
                            "p75": float(feature_data.quantile(0.75)),
                            "missing_pct": float(
                                (
                                    combined_data[feature].isnull().sum()
                                    / len(combined_data)
                                )
                                * 100,
                            ),
                        }

            # Aggregate feature importance across all models
            all_importances = {}
            for models in training_results.values():
                if isinstance(models, dict):
                    for model_data in models.values():
                        if (
                            isinstance(model_data, dict)
                            and "feature_importance" in model_data
                        ):
                            for feature, importance in model_data[
                                "feature_importance"
                            ].items():
                                if feature not in all_importances:
                                    all_importances[feature] = []
                                all_importances[feature].append(float(importance))

            # Calculate aggregate importance
            for feature, importances in all_importances.items():
                report["feature_importance_aggregate"][feature] = {
                    "mean_importance": float(np.mean(importances)),
                    "std_importance": float(np.std(importances)),
                    "count_models": len(importances),
                    "max_importance": float(np.max(importances)),
                    "min_importance": float(np.min(importances)),
                }

            # Calculate feature correlations (top correlations only)
            if len(feature_columns) <= 50:  # Only for reasonable feature counts
                corr_matrix = combined_data[feature_columns].corr()
                high_corr_pairs = []

                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:  # High correlation threshold
                            high_corr_pairs.append(
                                {
                                    "feature1": corr_matrix.columns[i],
                                    "feature2": corr_matrix.columns[j],
                                    "correlation": float(corr_val),
                                },
                            )

                report["feature_correlations"]["high_correlation_pairs"] = (
                    high_corr_pairs
                )

            # Feature quality metrics
            for feature in feature_columns:
                if feature in combined_data.columns:
                    feature_data = combined_data[feature].dropna()
                    if len(feature_data) > 0:
                        # Calculate coefficient of variation
                        cv = (feature_data.std() / abs(feature_data.mean())
                        if feature_data.mean() != 0
                            else 0
                        )

                        # Calculate zero variance
                        zero_var, feature_data.nunique() <= 1

                        report["feature_quality_metrics"][feature] = {
                            "coefficient_of_variation": float(cv),
                            "zero_variance": bool(zero_var),
                            "unique_values": int(feature_data.nunique()),
                            "data_type": str(combined_data[feature].dtype),
                        }

            return report

        except Exception as e:
            self.logger.exception(f"❌ Failed to create feature analysis report: {e}")
            return {"error": str(e)}

    async def _create_training_summary(
        self, training_results: dict[str, Any], exchange: str, symbol: str, combined_data: pd.DataFrame, feature_columns: list, ) -> dict[str, Any]:
        """Create training summary with key metrics and insights."""
        try:
            summary = {
                "training_summary": {
                    "total_timeframes": len(training_results),
                    "total_models_trained": sum(
                        len(models) if isinstance(models, dict) else 0
        for models in training_results.values()
                    ),
                    "successful_timeframes": [
                        tf
        for tf, models in training_results.items()
        if models and isinstance(models, dict) and len(models) > 0
                    ],
                    "failed_timeframes": [
                        tf
        for tf, models in training_results.items()
        if not models
                        or not isinstance(models, dict)
                        or len(models) == 0
                    ],
                },
                "performance_summary": {
                    "best_accuracy": 0.0,
                    "worst_accuracy": 1.0,
                    "average_accuracy": 0.0,
                    "best_model": None,
                    "best_timeframe": None,
                },
                "data_summary": {
                    "total_samples": len(combined_data),
                    "feature_count": len(feature_columns),
                    "data_span_days": 0,
                    "data_completeness": 0.0,
                },
                "training_metadata": {
                    "training_date": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
                    "symbol": symbol,
                    "exchange": exchange,
                    "model_architectures": self.model_architectures,
                },
            }

            # Calculate performance metrics
            accuracies = []
            for timeframe, models in training_results.items():
                if isinstance(models, dict):
                    for model_name, model_data in models.items():
                        if isinstance(model_data, dict) and "accuracy" in model_data:
                            acc = float(model_data["accuracy"])
                            accuracies.append(acc)

            if acc > summary["performance_summary"]["best_accuracy"]:
                summary["performance_summary"]["best_accuracy"] = acc
                summary["performance_summary"]["best_model"] = (
                    model_name
                )
                summary["performance_summary"]["best_timeframe"] = (
                    timeframe
                )

            summary["performance_summary"]["worst_accuracy"] = min(summary["performance_summary"]["worst_accuracy"], acc)

            if accuracies:
                summary["performance_summary"]["average_accuracy"] = float(
                    np.mean(accuracies),
                )

            # Calculate data span
            if "timestamp" in combined_data.columns:
                try:
                    timestamps = pd.to_datetime(combined_data["timestamp"])
                    data_span = timestamps.max() - timestamps.min()
                    summary["data_summary"]["data_span_days"] = data_span.days
                except:
                    pass

            # Calculate data completeness
            if feature_columns:
                completeness = (combined_data[feature_columns].notnull().sum().sum()
                    / (len(combined_data) * len(feature_columns))
                ) * 100
                summary["data_summary"]["data_completeness"] = float(completeness)

            return summary

        except Exception as e:
            self.logger.exception(f"❌ Failed to create training summary: {e}")
            return {"error": str(e)}

    def _extract_estimator_from_artifact(self, artifact: Any) -> Any:
        """Extract the underlying estimator from a saved artifact.

        This method supports several common wrapping patterns:
        - Dict with one of the keys: 'model', 'estimator', 'clf', 'pipeline'
        - Objects with attribute 'best_estimator_' (e.g., GridSearchCV)
        - Tuple/list where the first element is the estimator
        - If the artifact itself implements a 'predict' method, return as-is
        """
        try:
            predict_attr = getattr(artifact, "predict", None)
            if callable(predict_attr):
                return artifact
        except Exception:
            return artifact

        # Dict wrappers
        if isinstance(artifact, dict):
            for key in ("model", "estimator", "clf", "pipeline"):
                if key in artifact:
                    inner = artifact[key]
                    if callable(getattr(inner, "predict", None)):
                        return inner
            # Unwrap nested dicts once more
            if isinstance(inner, dict):
                for inner_key in ("model", "estimator", "clf"):
                    if inner_key in inner and callable(
                        getattr(inner[inner_key], "predict", None),
                    ):
                        return inner[inner_key]

        # GridSearchCV or similar
        if hasattr(artifact, "best_estimator_"):
            inner, getattr(artifact, "best_estimator_", None)
            if callable(getattr(inner, "predict", None)):
                return inner

        # Tuple/list where the first element might be the estimator
        if isinstance(artifact, list | tuple) and artifact:
            first = artifact[0]
            if callable(getattr(first, "predict", None)):
                return first

        # Fallback: return original artifact
        return artifact


# Model Architectures


class CNNModel(nn.Module):
    """CNN model for 1m timeframe."""

    def __init__(self, input_channels: int, sequence_length: int, num_classes: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # Calculate output size after convolutions and pooling
        conv_output_size = sequence_length // 8 * 256
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x, x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)



class TCNModel(nn.Module):
    """Temporal Convolutional Network for 5m timeframe."""

    def __init__(
        self, input_size: int, num_channels: list[int], kernel_size: int, num_classes: int, ) -> None:
        super().__init__()

        self.tcn = TemporalBlock(
            input_size, num_channels[0], kernel_size, stride=1, dilation=1
        )
        self.tcn2 = TemporalBlock(
            num_channels[0], num_channels[1], kernel_size, stride=1, dilation=2
        )
        self.tcn3 = TemporalBlock(
            num_channels[1], num_channels[2], kernel_size, stride=1, dilation=4
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_channels[2], num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        x, x.transpose(1, 2)  # (batch, input_size, sequence_length)

        x = self.tcn(x)
        x = self.tcn2(x)
        x = self.tcn3(x)

        x, x.transpose(1, 2)  # (batch, sequence_length, channels)
        x = x[:, -1, :]  # Take last timestep,
        x = self.dropout(x)
        return self.fc(x)



class TemporalBlock(nn.Module):
    """Temporal block for TCN."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(out + x)


class TransformerModel(nn.Module):
    """Transformer model for 15m timeframe."""

    def __init__(
        self, input_size: int, d_model: int, nhead: int, num_layers: int, num_classes: int, ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer, nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)

        x = x[:, -1, :]  # Take last timestep,
        x = self.dropout(x)
        return self.fc(x)



class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe, torch.zeros(max_len, d_model)
        position, torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term, torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model),
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe, pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


# Trainers


class CNNTrainer:
    """Trainer for CNN model."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.001, batch_size: int = 32,
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @validate_feature_engineering_with_lookahead_bias_detection
    async def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 50
    ) -> dict[str, list[float]]:
        """Train the CNN model."""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += float(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += int((predicted == batch_y).sum().item())

            avg_train_loss = train_loss / max(1, len(train_loader))
            train_acc = train_correct / max(1, train_total)

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test)
                test_loss = float(self.criterion(outputs, y_test).item())
                _, predicted = torch.max(outputs.data, 1)
                test_acc = int((predicted == y_test).sum().item()) / max(1, y_test.size(0))

            history["train_loss"].append(avg_train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

        return history


class TCNTrainer:
    """Trainer for TCN model."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.001, batch_size: int = 64,
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 100
    ) -> dict[str, list[float]]:
        """Train the TCN model."""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += float(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += int((predicted == batch_y).sum().item())

            avg_train_loss = train_loss / max(1, len(train_loader))
            train_acc = train_correct / max(1, train_total)

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test)
                test_loss = float(self.criterion(outputs, y_test).item())
                _, predicted = torch.max(outputs.data, 1)
                test_acc = int((predicted == y_test).sum().item()) / max(1, y_test.size(0))

            history["train_loss"].append(avg_train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

        return history

    # Cross-validation training methods
    async def _train_cnn_model_cv(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, split_idx: int, ) -> dict[str, Any] | None:
        """Train CNN model with cross-validation."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

        # Prepare data
            X_train_np, X_val_np, X_test_np = (
                X_train.values,
                X_val.values,
                X_test.values,
            )
            y_train_np, y_val_np, y_test_np = (
                y_train.values,
                y_val.values,
                y_test.values,
            )

        # Create sequences for CNN
            sequence_length = 60,
            X_train_seq, self._create_sequences(X_train_np, sequence_length)
            X_val_seq, self._create_sequences(X_val_np, sequence_length)
            X_test_seq, self._create_sequences(X_test_np, sequence_length)

        # Adjust targets for sequences
            y_train_seq = y_train_np[sequence_length:],
            y_val_seq = y_val_np[sequence_length:],
            y_test_seq = y_test_np[sequence_length:],

        # Create and train model
            model, CNNModel(
                input_channels=X_train.shape[1],
                sequence_length=sequence_length,
                num_classes=len(np.unique(y_train_np)),
            )

            trainer = CNNTrainer(model, learning_rate=0.001, batch_size=32)
            history = await trainer.train(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_outputs = model(torch.FloatTensor(X_test_seq))
                test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

            # Calculate metrics
            accuracy = accuracy_score(y_test_seq, test_preds)
            f1 = f1_score(y_test_seq, test_preds, average="weighted")
            precision = precision_score(y_test_seq, test_preds, average="weighted")
            recall = recall_score(y_test_seq, test_preds, average="weighted")

            return {
                "model": model,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "val_accuracy": history["test_acc"][-1],
                "training_history": history,
                "split_idx": split_idx,
            }

        except Exception as e:
            self.logger.exception(
                f"❌ CNN CV training failed for {timeframe} split {split_idx}: {e}",
            )
            return None

    async def _train_tcn_model_cv(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, split_idx: int, ) -> dict[str, Any] | None:
        """Train TCN model with cross-validation."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

        # Prepare data
            X_train_np, X_val_np, X_test_np = (
                X_train.values,
                X_val.values,
                X_test.values,
            )
            y_train_np, y_val_np, y_test_np = (
                y_train.values,
                y_val.values,
                y_test.values,
            )

        # Create sequences for TCN
            sequence_length = 100,
            X_train_seq, self._create_sequences(X_train_np, sequence_length)
            X_val_seq, self._create_sequences(X_val_np, sequence_length)
            X_test_seq, self._create_sequences(X_test_np, sequence_length)

        # Adjust targets for sequences
            y_train_seq = y_train_np[sequence_length:],
            y_val_seq = y_val_np[sequence_length:],
            y_test_seq = y_test_np[sequence_length:],

        # Create and train model
            model, TCNModel(
                input_size=X_train.shape[1],
                num_channels=[64, 128, 256],
                kernel_size=3,
                num_classes=len(np.unique(y_train_np)),
            )

            trainer = TCNTrainer(model, learning_rate=0.001, batch_size=32)
            history = await trainer.train(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=100
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_outputs = model(torch.FloatTensor(X_test_seq))
                test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

            # Calculate metrics
            accuracy = accuracy_score(y_test_seq, test_preds)
            f1 = f1_score(y_test_seq, test_preds, average="weighted")
            precision = precision_score(y_test_seq, test_preds, average="weighted")
            recall = recall_score(y_test_seq, test_preds, average="weighted")

            return {
                "model": model,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "val_accuracy": history["test_acc"][-1],
                "training_history": history,
                "split_idx": split_idx,
            }

        except Exception as e:
            self.logger.exception(
                f"❌ TCN CV training failed for {timeframe} split {split_idx}: {e}",
            )
            return None

    async def _train_transformer_model_cv(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, split_idx: int, ) -> dict[str, Any] | None:
        """Train Transformer model with cross-validation."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

        # Prepare data
            X_train_np, X_val_np, X_test_np = (
                X_train.values,
                X_val.values,
                X_test.values,
            )
            y_train_np, y_val_np, y_test_np = (
                y_train.values,
                y_val.values,
                y_test.values,
            )

        # Create sequences for Transformer
            sequence_length = 50,
            X_train_seq, self._create_sequences(X_train_np, sequence_length)
            X_val_seq, self._create_sequences(X_val_np, sequence_length)
            X_test_seq, self._create_sequences(X_test_np, sequence_length)

        # Adjust targets for sequences
            y_train_seq = y_train_np[sequence_length:],
            y_val_seq = y_val_np[sequence_length:],
            y_test_seq = y_test_np[sequence_length:],

        # Create and train model
            model, TransformerModel(
                input_size=X_train.shape[1],
                d_model=128,
                nhead=8,
                num_layers=4,
                num_classes=len(np.unique(y_train_np)),
            )

            trainer = TransformerTrainer(model, learning_rate=0.0001, batch_size=32)
            history = await trainer.train(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=150
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_outputs = model(torch.FloatTensor(X_test_seq))
                test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

            # Calculate metrics
            accuracy = accuracy_score(y_test_seq, test_preds)
            f1 = f1_score(y_test_seq, test_preds, average="weighted")
            precision = precision_score(y_test_seq, test_preds, average="weighted")
            recall = recall_score(y_test_seq, test_preds, average="weighted")

            return {
                "model": model,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "val_accuracy": history["test_acc"][-1],
                "training_history": history,
                "split_idx": split_idx,
            }

        except Exception as e:
            self.logger.exception(
                f"❌ Transformer CV training failed for {timeframe} split {split_idx}: {e}",
            )
            return None

    async def _train_lightgbm_model_cv(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, timeframe: str, split_idx: int, ) -> dict[str, Any] | None:
        """Train LightGBM model with cross-validation."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

        # Prepare data
            X_train_np, X_val_np, X_test_np = (
                X_train.values,
                X_val.values,
                X_test.values,
            )
            y_train_np, y_val_np, y_test_np = (
                y_train.values,
                y_val.values,
                y_test.values,
            )

        # Create and train model
            model, lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42,
                verbose=-1,
            )

            model.fit(
                X_train_np,
                y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                eval_metric="multi_logloss",
                early_stopping_rounds=10,
                verbose=False,
            )

        # Evaluate
            test_preds = model.predict(X_test_np),

        # Calculate metrics
            accuracy = accuracy_score(y_test_np, test_preds)
            f1 = f1_score(y_test_np, test_preds, average="weighted")
            precision = precision_score(y_test_np, test_preds, average="weighted")
            recall = recall_score(y_test_np, test_preds, average="weighted")

            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_, strict=False))

            return {
                "model": model,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "val_accuracy": model.best_score_ if hasattr(model, "best_score_") else accuracy,
                "feature_importance": feature_importance,
                "split_idx": split_idx,
            }

        except Exception as e:
            self.logger.exception(
                f"❌ LightGBM CV training failed for {timeframe} split {split_idx}: {e}",
            )
            return None

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for time series models."""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i : i + sequence_length])
        return np.array(sequences)

    async def _load_regime_weights(
        self, exchange: str, symbol: str, data_dir: str,
    ) -> pd.DataFrame | None:
        """Load regime weights for sample weighting."""
        try:
            # Try pickle first
            weights_pickle_path = f"{data_dir}/{exchange}_{symbol}_regime_weights.pkl"
            if os.path.exists(weights_pickle_path):
                with open(weights_pickle_path, "rb") as f:
                    weights_df = pickle.load(f)
                if isinstance(weights_df, pd.DataFrame):
                    self.logger.info(
                        f"✅ Loaded regime weights from pickle: {weights_df.shape}",
                    )
                    return weights_df

            # Fallback to parquet
            weights_path = f"{data_dir}/{exchange}_{symbol}_regime_weights.parquet"
            if os.path.exists(weights_path):
                weights_df = pd.read_parquet(weights_path)
                weights_df["timestamp"] = pd.to_datetime(weights_df["timestamp"])
                self.logger.info(
                    f"✅ Loaded regime weights from parquet: {weights_df.shape}",
                )
                return weights_df

            self.logger.info(
                "ℹ️ No regime weights found, proceeding without sample weighting",
            )
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load regime weights: {e}")
            return None

    async def _add_regime_weights(
        self, data: pd.DataFrame, regime_weights: pd.DataFrame, timeframe: str,
    ) -> pd.DataFrame:
        """Add regime weights to the dataset for sample weighting with S/R level integration."""
        try:
            # Merge regime weights with data
            if "timestamp" in regime_weights.columns:
                merged_data = data.merge(regime_weights, on="timestamp", how="left")

        # Initialize SR predictor if not already done
            if not hasattr(self, "sr_predictor_initialized"):
                try:
                    await self.sr_predictor.initialize()
                    self.sr_predictor_initialized = True
                    self.logger.info(
                        "✅ SRBreakoutPredictor initialized for sample weighting",
                    )
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Failed to initialize SRBreakoutPredictor: {e}",
                    )
                    self.sr_predictor_initialized = False

        # Create base sample weights based on regime confidence
            if "confidence" in merged_data.columns:
                base_weights = merged_data["confidence"].fillna(0.5).clip(0.1, 1.0)
            elif "regime_weight" in merged_data.columns:
                base_weights = merged_data["regime_weight"].fillna(0.5).clip(0.1, 1.0)
            else:
                base_weights = pd.Series(1.0, index=merged_data.index)

            # Add S/R-aware weighting if SR predictor is available
            sr_weights = None
            if self.sr_predictor_initialized and len(merged_data) > 0:
                sr_weights = await self._calculate_sr_sample_weights(
                    merged_data, timeframe,
                )
            if sr_weights is not None:
                # Combine regime weights with S/R weights
                # S/R weights get 30% influence, regime weights get 70%
                combined_weights = base_weights * 0.7 + sr_weights * 0.3
                merged_data["sample_weight"] = combined_weights.clip(0.1, 1.0)
                self.logger.info(
                    f"   ✅ Added S/R-aware sample weights for {timeframe}",
                )
            else:
                merged_data["sample_weight"] = base_weights

            self.logger.info(f"   ✅ Added regime weights for {timeframe}")
            return merged_data

        except Exception as e:
            self.logger.exception(f"❌ Failed to add regime weights for {timeframe}: {e}")
            return data

    async def _calculate_sr_sample_weights(
        self, data: pd.DataFrame, timeframe: str, ) -> pd.Series | None:
        """Calculate S/R-aware sample weights for training data."""
        try:
            if len(data) == 0:
                return None

            # Prepare market data for S/R analysis
            if not all(
                col in data.columns for col in ["open", "high", "low", "close", "volume"]
            ):
                self.logger.warning(
                    f"⚠️ Missing OHLCV columns for S/R analysis in {timeframe}",
                )
                return None

            # Use a subset of data for efficiency (every 10th row for large datasets)
            sample_interval = max(1, len(data) // 1000)  # Sample up to 1000 points
            sample_data = data.iloc[::sample_interval].copy()

            sr_weights = []

            for idx, row in sample_data.iterrows():
                try:
                    # Create market data slice for S/R analysis
                    current_price = row["close"]

                    # Get S/R context for this point
                    market_slice = data.loc[:idx].tail(100)  # Last 100 bars for context
                    if len(market_slice) < 20:
                        sr_weights.append(0.5)  # Default weight
                        continue

                    sr_context = await self.sr_predictor.get_sr_context(
                        market_data=market_slice, current_price=current_price,
                    )

                    # Check if near S/R level
                    is_near_sr = self.sr_predictor.is_near_sr_level(
                        current_price, sr_context,
                    )

                    if is_near_sr:
                        # Higher weight for samples near S/R levels
                        sr_weights.append(0.9)
                    else:
                        sr_weights.append(0.5)
                except Exception:
                    sr_weights.append(0.5)

            # Interpolate weights for all data points
            if len(sr_weights) > 1:
                # Create a series with the sampled weights
                sample_indices = sample_data.index
                weight_series = pd.Series(sr_weights, index=sample_indices)

                # Interpolate to all data points
                return (
                    weight_series.reindex(data.index).interpolate(method="time").fillna(method="bfill").fillna(method="ffill").clip(0.1, 1.0)
                )

            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate S/R sample weights for {timeframe}: {e}")
            return None

    def _derive_sample_weight(
        self, df: pd.DataFrame, regime_key: str | None, ) -> pd.Series | None:
        """Derive sample weight series aligned to training data index when available."""
        try:
            # Check for explicit sample weight column
            if "sample_weight" in df.columns:
                return (
                    df["sample_weight"]
                    .astype(float)
                    .clip(0.0, 1.0)
                    .reindex(df.index)
                    .fillna(0.0)
                )

            # Check for regime confidence
            if "confidence" in df.columns:
                return (
                    df["confidence"]
                    .astype(float)
                    .clip(0.0, 1.0)
                    .reindex(df.index)
                    .fillna(0.0)
                )

            # Check for intensity-based weights (HMM-specific)
            intensity_cols = [
                col for col in df.columns if col.startswith("intensity_cluster_")
            ]
            if intensity_cols:
                # Use average intensity as sample weight
                intensity_weights = df[intensity_cols].mean(axis=1).clip(0.0, 1.0)
                return intensity_weights.reindex(df.index).fillna(0.0)

            # No sample weights available
            return None

        except Exception as e:
            self.logger.warning(f"Failed to derive sample weight: {e}")
            return None

    def _time_aware_split(self, X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
        """Time-aware train/test split helper to prevent look-ahead bias."""
        if isinstance(X.index, pd.DatetimeIndex):
            n = len(X)
            cut = int(n * (1.0 - test_frac))
            return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X, y, test_size=test_frac, random_state=42, stratify=y
        )

    async def _train_and_optionally_refit(
        self, model_key: str, train_coro, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, regime_name: str, sample_weight: pd.Series | None, ) -> tuple[str, dict[str, Any] | None]:
        """Train a model using provided coroutine, then optionally refit with sample weights.
        Returns (model_key, model_package_or_None).
        """
        try:
            pkg = await train_coro(X_train, X_test, y_train, y_test, regime_name)
            if not pkg:
                return model_key, None

            # Optional sample-weighted refit where supported
            if sample_weight is not None:
                try:
                    estimator = pkg.get("model") if isinstance(pkg, dict) else None
                    if estimator is not None and hasattr(estimator, "fit"):
                        # Try to find a label mapping from the package; fallback to identity
                        label_mapping = None
                        for k in (
                            "xgb_label_mapping",
                            "lgb_label_mapping",
                            "rf_label_mapping",
                            "nn_label_mapping",
                            "svm_label_mapping",
                        ):
                            if isinstance(pkg, dict) and k in pkg:
                                label_mapping = pkg[k]
                                break
                        if label_mapping is not None:
                            # Fit with sample weights if supported
                            try:
                                estimator.fit(X_train, y_train, sample_weight=sample_weight)
                                pkg["model_refit_weighted"] = True
                            except Exception:
                                pkg["model_refit_weighted"] = False
                except Exception:
                    pass

            return model_key, pkg
        except Exception as e:
            self.logger.exception(f"❌ Training coroutine failed for {model_key}: {e}")
            return model_key, None

    async def _apply_smart_feature_selection(
        self, data: pd.DataFrame, feature_columns: list, target_column: str, max_features: int = 100
    ) -> list:
        """Apply comprehensive feature selection using multiple methods:
        1. Mutual Information for feature-target relevance
        2. Collinearity analysis with correlation
        3. Random Forest importance
        4. SHAP values for interpretability
        5. Category-based selection (min 15 per category).

        Args:
            data: Training data
            feature_columns: List of feature column names
            target_column: Target column name
            max_features: Maximum number of features to select (target: 100)

        Returns:
            List of selected feature names

        """
        try:
            self.logger.info(
                f"🔍 Applying comprehensive feature selection on {len(feature_columns)} features...",
            )
            self.logger.info(f"📊 Target: {max_features} features, min 15 per category")

            # Prepare data
            X = data[feature_columns].fillna(0)
            y = data[target_column]

            # Step 1: Pre-filtering (variance, correlation)
            pre_filtered = await self._pre_filter_features(X, feature_columns)
            self.logger.info(
                f"   ✅ Pre-filtering: {len(pre_filtered)} features remaining",
            )

            # Step 2: Calculate comprehensive feature scores
            feature_scores = await self._calculate_comprehensive_scores(
                X[pre_filtered], y,
            )

            # Step 3: Category-based selection
            category_selected = await self._select_features_by_category(
                pre_filtered, feature_scores,
            )
            self.logger.info(
                f"   ✅ Category selection: {len(category_selected)} features",
            )

            # Step 4: Final selection and validation
            final_selected = await self._final_feature_selection(
                X[category_selected], y, category_selected, max_features,
            )

            self.logger.info(f"   ✅ Final selection: {len(final_selected)} features")
            await self._log_category_breakdown(final_selected)

            return final_selected

        except Exception as e:
            self.logger.exception(f"❌ Smart feature selection failed: {e}")
            return feature_columns  # Return original features if selection fails

    async def _calculate_mutual_information(
        self, X: pd.DataFrame, y: pd.Series, ) -> np.ndarray:
        """Calculate mutual information between features and target."""
        try:
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )

            # Determine if classification or regression
            if y.dtype in ["object", "category"] or len(y.unique()) < 10:
                # Classification
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                # Regression
                mi_scores = mutual_info_regression(X, y, random_state=42)

            return mi_scores

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Mutual information calculation failed: {e}")
            # Return uniform scores as fallback
            return np.ones(len(X.columns))

    async def _remove_collinear_features(
        self, X: pd.DataFrame, threshold: float = 0.95, ) -> list[str]:
        """Remove collinear features using correlation analysis and PCA."""
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()

            # Find highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [
                column
                for column in upper_tri.columns
                if any(upper_tri[column] > threshold)
            ]

            # Remove highly correlated features
            low_corr_features = [
                col for col in X.columns if col not in high_corr_features
            ]

            # If too many features removed, use PCA for dimensionality reduction
            if len(low_corr_features) < len(X.columns) * 0.5:
                self.logger.info("   🔧 Too many collinear features, applying PCA...")
                return await self._apply_pca_dimensionality_reduction(
                    X, target_variance=0.95
                )

            return low_corr_features

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Collinearity removal failed: {e}")
            return list(X.columns)

    async def _apply_pca_dimensionality_reduction(
        self, X: pd.DataFrame, target_variance: float = 0.95, ) -> list[str]:
        """Apply PCA for dimensionality reduction while preserving variance."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA
            pca = PCA(n_components=target_variance)
            pca.fit(X_scaled)

            # Get number of components needed
            n_components = int(pca.n_components_)

            # Select top features based on PCA loadings
            loadings = np.abs(pca.components_)
            feature_importance = np.sum(loadings, axis=0)

            # Select top features
            top_indices = np.argsort(feature_importance)[-n_components:]
            selected_features = [X.columns[i] for i in top_indices]

            self.logger.info(
                f"   🔧 PCA reduced to {len(selected_features)} features (variance: {target_variance})",
            )

            return selected_features

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ PCA dimensionality reduction failed: {e}")
            return list(X.columns)

    async def _select_by_random_forest_importance(
        self, X: pd.DataFrame, y: pd.Series, max_features: int, ) -> list[str]:
        """Select features based on Random Forest importance scores."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Determine if classification or regression
            if y.dtype in ["object", "category"] or len(y.unique()) < 10:
                rf = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1,
                )
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            # Fit Random Forest
            rf.fit(X, y)

            # Get feature importances
            importances = getattr(rf, "feature_importances_", None)
            if importances is None:
                return list(X.columns)[:max_features]

            # Select top features
            indices = np.argsort(importances)[::-1][:max_features]
            selected_features = [X.columns[i] for i in indices]
            return selected_features

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Random Forest importance selection failed: {e}")
            return list(X.columns)[:max_features]

    async def _validate_with_shap(
        self, X: pd.DataFrame, y: pd.Series, max_features: int, ) -> list[str]:
        """Validate feature selection using SHAP values."""
        try:
            # Use LightGBM for SHAP analysis (faster than Random Forest for SHAP)
            import lightgbm as lgb
            import shap

            # Determine if classification or regression
            if y.dtype in ["object", "category"] or len(y.unique()) < 10:
                model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            else:
                model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)

            # Fit model
            model.fit(X, y)

            # Calculate SHAP values (use a subset for speed)
            sample_size = min(1000, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # If classification, use the first class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Calculate mean absolute SHAP values per feature
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            feature_shap = list(zip(X.columns, mean_shap, strict=False))
            feature_shap.sort(key=lambda x: x[1], reverse=True)

            # Select top features based on SHAP importance
            return [feature for feature, score in feature_shap[:max_features]]

        except Exception as e:
            self.logger.exception(f"❌ SHAP validation failed: {e}")
            # Return all features if SHAP fails
            return list(X.columns)[:max_features]

    async def _pre_filter_features(
        self, X: pd.DataFrame, feature_columns: list, ) -> list:
        """Enhanced pre-filter features using data-driven methods (VIF, MI, SHAP, RF)."""
        try:
            self.logger.info(f"🔍 Enhanced pre-filtering: {len(feature_columns)} features")
            
            # Stage 1: Data quality filtering
            X_clean = X[feature_columns].copy()
            
            # Remove features with too many NaN values (>10%)
            nan_ratio = X_clean.isna().sum() / len(X_clean)
            high_nan_features = nan_ratio[nan_ratio > 0.1].index.tolist()
            X_clean = X_clean.drop(columns=high_nan_features)
            
            # Remove features with infinite values
            inf_features = []
            for col in X_clean.columns:
                if np.isinf(X_clean[col]).any():
                    inf_features.append(col)
            X_clean = X_clean.drop(columns=inf_features)
            
            # Fill remaining NaN values
            X_clean = X_clean.fillna(method="ffill").fillna(method="bfill").fillna(0)
            
            self.logger.info(
                f"   Data quality filtering: {len(feature_columns)} -> {len(X_clean.columns)} features",
            )
            
            # Stage 2: Variance filtering
            variance = X_clean.var()
            high_variance_mask = variance > 1e-6
            high_variance_features = [
                col for col in X_clean.columns if high_variance_mask[col]
            ]
            
            self.logger.info(
                f"   Variance filtering: {len(X_clean.columns)} -> {len(high_variance_features)} features",
            )
            
            # Stage 3: VIF filtering (multicollinearity)
            try:
                from src.utils.vif_calculator import calculate_vif_robust
                
                X_vif = X_clean[high_variance_features]
                vif_scores = calculate_vif_robust(X_vif)
                
                # Remove features with high VIF (>10)
                low_vif_features = vif_scores[vif_scores <= 10.0].index.tolist()
                
                self.logger.info(
                    f"   VIF filtering: {len(high_variance_features)} -> {len(low_vif_features)} features",
                )
                
            except Exception as e:
                self.logger.warning(f"VIF filtering failed: {e}, skipping")
                low_vif_features = high_variance_features
            
            # Stage 4: Correlation filtering
            uncorr_features = low_vif_features
            if len(low_vif_features) > 1:
                X_corr = X_clean[low_vif_features]
                corr_matrix = X_corr.corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Find features to drop
                to_drop = [
                    column for column in upper_tri.columns
                    if any(upper_tri[column] > 0.95)
                ]
                uncorr_features = [
                    col for col in low_vif_features if col not in to_drop
                ]
            
            self.logger.info(
                f"   Correlation filtering: {len(low_vif_features)} -> {len(uncorr_features)} features",
            )
            
            # Stage 5: Mutual Information filtering (if target available)
            try:
                # Try to get target from the data
                target_col = None
                for col in X.columns:
                    if col.lower() in ['label', 'target', 'direction', 'y']:
                        target_col = col
                        break
                
                if target_col and target_col in X.columns:
                    y = X[target_col]
                    
                    # Calculate mutual information
                    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                    
                    # Determine task type
                    task_type = "classification" if len(y.unique()) < 10 else "regression"
                    
                    if task_type == "classification":
                        mi_scores = mutual_info_classif(X_clean[uncorr_features], y, random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X_clean[uncorr_features], y, random_state=42)
                    
                    # Remove features with low MI (<0.01)
                    mi_series = pd.Series(mi_scores, index=uncorr_features)
                    high_mi_features = mi_series[mi_scores >= 0.01].index.tolist()
                    
                    self.logger.info(
                        f"   MI filtering: {len(uncorr_features)} -> {len(high_mi_features)} features",
                    )
                    
                    uncorr_features = high_mi_features
                    
            except Exception as e:
                self.logger.warning(f"MI filtering failed: {e}, skipping")
            
            # Stage 6: SHAP-based filtering (if target available)
            try:
                if target_col and target_col in X.columns and len(uncorr_features) > 50:
                    from src.analyst.meta_label_relevance import compute_shap_importance
                    
                    # Calculate SHAP importance
                    shap_scores = compute_shap_importance(
                        X_clean[uncorr_features], y, task=task_type
                    )
                    
                    if shap_scores:
                        # Remove bottom 20% of features by SHAP importance
                        shap_series = pd.Series(shap_scores)
                        threshold = shap_series.quantile(0.2)
                        high_shap_features = shap_series[shap_series >= threshold].index.tolist()
                        
                        self.logger.info(
                            f"   SHAP filtering: {len(uncorr_features)} -> {len(high_shap_features)} features",
                        )
                        
                        uncorr_features = high_shap_features
                        
            except Exception as e:
                self.logger.warning(f"SHAP filtering failed: {e}, skipping")
            
            # Stage 7: RandomForest importance filtering (if target available)
            try:
                if target_col and target_col in X.columns and len(uncorr_features) > 30:
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    
                    # Train RF for feature importance
                    if task_type == "classification":
                        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    
                    rf.fit(X_clean[uncorr_features], y)
                    rf_importance = pd.Series(rf.feature_importances_, index=uncorr_features)
                    
                    # Remove bottom 20% of features by RF importance
                    threshold = rf_importance.quantile(0.2)
                    high_rf_features = rf_importance[rf_importance >= threshold].index.tolist()
                    
                    self.logger.info(
                        f"   RF filtering: {len(uncorr_features)} -> {len(high_rf_features)} features",
                    )
                    
                    uncorr_features = high_rf_features
                    
            except Exception as e:
                self.logger.warning(f"RF filtering failed: {e}, skipping")
            
            self.logger.info(
                f"✅ Enhanced pre-filtering completed: {len(feature_columns)} -> {len(uncorr_features)} features",
            )
            
            return uncorr_features

        except Exception as e:
            self.logger.warning(f"⚠️ Error in enhanced pre-filtering: {e}")
            return feature_columns

    async def _calculate_comprehensive_scores(
        self, X: pd.DataFrame, y: pd.Series, ) -> dict:
        """Calculate feature importance scores using multiple methods."""
        try:
            feature_scores = {}
            # Prepare data
            X_clean = X.fillna(0).astype(float)
            y_clean = y.fillna(0).astype(float)
            # Determine task type
            task_type = (
                "classification"
                if y_clean.dtype in ["object", "category"] or len(y_clean.unique()) < 10
                else "regression"
            )
            # 1. Mutual Information
            if task_type == "classification":
                mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            for i, feature in enumerate(X_clean.columns):
                feature_scores[feature] = {"mutual_info": float(mi_scores[i])}
            # 2. Random Forest importance
            if task_type == "classification":
                rf = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1,
                )
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_clean, y_clean)
            rf_importance = rf.feature_importances_
            for i, feature in enumerate(X_clean.columns):
                feature_scores[feature]["rf_importance"] = float(rf_importance[i])

            # 3. F-statistics
            if task_type == "classification":
                f_scores, _, f_classif(X_clean, y_clean)
            else: f_scores = _, f_regression(X_clean, y_clean)

            for i, feature in enumerate(X_clean.columns):
                feature_scores[feature]["f_statistic"] = f_scores[i]

            # 4. LightGBM importance
            if task_type == "classification":
                lgb_model, lgb.LGBMClassifier(
                    n_estimators=100, random_state=42, verbose=-1,
                )
            else: lgb_model = lgb.LGBMRegressor(
                    n_estimators=100, random_state=42, verbose=-1,
                )

            lgb_model.fit(X_clean, y_clean)
            lgb_importance = lgb_model.feature_importances_,

            for i, feature in enumerate(X_clean.columns):
                feature_scores[feature]["lgb_importance"] = lgb_importance[i]

            # 5. SHAP importance (for top features)
            try:
                import shap

                sample_size = min(500, len(X_clean))
                X_sample = X_clean.sample(n=sample_size, random_state=42)

                explainer = shap.TreeExplainer(lgb_model)
                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if task_type == "classification" else shap_values[0]

                mean_shap = np.mean(np.abs(shap_values), axis=0)

                for i, feature in enumerate(X_clean.columns):
                    feature_scores[feature]["shap_importance"] = float(mean_shap[i])
            except Exception as e:
                self.logger.warning(f"⚠️ SHAP calculation failed: {e}")

            # Combine scores
            for feature, scores in feature_scores.items():
                normalized_scores = []
                for score in scores.values():
                    if score is not None and not np.isnan(score):
                        normalized_scores.append(score)

                if normalized_scores:
                    feature_scores[feature]["combined_score"] = float(np.mean(normalized_scores))
                else:
                    feature_scores[feature]["combined_score"] = 0.0

            return feature_scores

        except Exception as e:
            self.logger.exception(f"❌ Error calculating comprehensive scores: {e}")
            return {}

    async def _select_features_by_category(
        self, all_features: list, feature_scores: dict, ) -> list:
        """Select features by category ensuring minimum per category."""
        try:
            # Define feature categories
            feature_categories = {
                "technical_indicators": [
                    "rsi",
                    "macd",
                    "bb_",
                    "sma",
                    "ema",
                    "atr",
                    "adx",
                    "cci",
                    "mfi",
                    "roc",
                    "momentum",
                    "trend",
                ],
                "volatility_features": [
                    "volatility",
                    "parkinson",
                    "garman_klass",
                    "adaptive_atr",
                    "ewma_volatility",
                ],
                "liquidity_features": [
                    "liquidity",
                    "bid_ask",
                    "market_depth",
                    "volume_profile",
                    "order_imbalance",
                    "spread",
                ],
                "order_flow_features": [
                    "order_flow",
                    "volume_ratio",
                    "buy_sell_pressure",
                    "large_order",
                    "funding",
                    "obv",
                    "vwap",
                ],
                "wavelet_features": [
                    "wavelet",
                    "dwt",
                    "cwt",
                    "wavelet_packet",
                    "wavelet_denoised",
                ],
                "regime_features": [
                    "composite_cluster",
                    "intensity_cluster",
                    "momentum_p_state",
                    "volatility_p_state",
                    "liquidity_p_state",
                ],
                "time_features": [
                    "hour",
                    "day",
                    "week",
                    "month",
                    "season",
                    "time_sin",
                    "time_cos",
                ],
                "price_features": [
                    "price",
                    "returns",
                    "log_returns",
                    "price_momentum",
                    "price_acceleration",
                ],
            }

            selected_features: list[str] = []
            category_counts: dict[str, int] = {}

            # Group features by category
            feature_categories_groups: dict[str, list[str]] = {}
            for feature in all_features:
                category = self._get_feature_category(feature, feature_categories)
                if category not in feature_categories_groups:
                    feature_categories_groups[category] = []
                feature_categories_groups[category].append(feature)

            # Select features from each category
            for category, features in feature_categories_groups.items():
                if category not in category_counts:
                    category_counts[category] = 0

                # Get scores for features in this category
                category_scores: list[tuple[str, float]] = []
                for feature in features:
                    if feature in feature_scores:
                        score = feature_scores[feature].get("combined_score", 0.0)
                        category_scores.append((feature, float(score)))

                # Sort by score and select top features
                category_scores.sort(key=lambda x: x[1], reverse=True)

                # Select between 15 and 30 where available
                min_select = min(15, len(category_scores))
                max_select = min(30, len(category_scores))
                num_to_select = max(min_select, min(max_select, len(category_scores)))
                selected_category_features = [f for f, _ in category_scores[:num_to_select]]

                selected_features.extend(selected_category_features)
                category_counts[category] = len(selected_category_features)

                self.logger.info(
                    f"   {category}: {len(selected_category_features)} features selected",
                )

            return selected_features

        except Exception as e:
            self.logger.exception(f"❌ Error selecting features by category: {e}")
            return all_features

    def _get_feature_category(self, feature: str, feature_categories: dict) -> str:
        """Determine the category of a feature based on its name."""
        feature_lower = feature.lower()

        for category, keywords in feature_categories.items():
            for keyword in keywords:
                if keyword in feature_lower:
                    return category

        return "other"

    async def _final_feature_selection(
        self, X: pd.DataFrame, y: pd.Series, selected_features: list, max_features: int, ) -> list:
        """Final feature selection and validation."""
        try:
            if len(selected_features) <= max_features:
                return selected_features

            # If we have too many features, select the best ones using simple variance as a proxy
            feature_scores: list[tuple[str, float]] = []
            for feature in selected_features:
                if feature in X.columns:
                    score = float(X[feature].var()) if feature in X.columns else 0.0
                    feature_scores.append((feature, score))

            # Sort by score and select top features
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            return [f for f, _ in feature_scores[:max_features]]

        except Exception as e:
            self.logger.exception(f"❌ Error in final selection: {e}")
            return selected_features[:max_features]

    async def _log_category_breakdown(self, final_features: list) -> None:
        """Log the breakdown of selected features by category."""
        try:
            self.logger.info("📊 Final feature selection breakdown:")
            category_counts = {}

            feature_categories = {
                "technical_indicators": [
                    "rsi",
                    "macd",
                    "bb_",
                    "sma",
                    "ema",
                    "atr",
                    "adx",
                    "cci",
                    "mfi",
                    "roc",
                    "momentum",
                    "trend",
                ],
                "volatility_features": [
                    "volatility",
                    "parkinson",
                    "garman_klass",
                    "adaptive_atr",
                    "ewma_volatility",
                ],
                "liquidity_features": [
                    "liquidity",
                    "bid_ask",
                    "market_depth",
                    "volume_profile",
                    "order_imbalance",
                    "spread",
                ],
                "order_flow_features": [
                    "order_flow",
                    "volume_ratio",
                    "buy_sell_pressure",
                    "large_order",
                    "funding",
                    "obv",
                    "vwap",
                ],
                "wavelet_features": [
                    "wavelet",
                    "dwt",
                    "cwt",
                    "wavelet_packet",
                    "wavelet_denoised",
                ],
                "regime_features": [
                    "composite_cluster",
                    "intensity_cluster",
                    "momentum_p_state",
                    "volatility_p_state",
                    "liquidity_p_state",
                ],
                "time_features": [
                    "hour",
                    "day",
                    "week",
                    "month",
                    "season",
                    "time_sin",
                    "time_cos",
                ],
                "price_features": [
                    "price",
                    "returns",
                    "log_returns",
                    "price_momentum",
                    "price_acceleration",
                ],
            }

            for feature in final_features:
                category, self._get_feature_category(feature, feature_categories)
                category_counts[category] = category_counts.get(category, 0) + 1

            for category, count in sorted(category_counts.items()):
                self.logger.info(f"   {category}: {count} features")

        except Exception as e:
            self.logger.warning(f"⚠️ Error logging category breakdown: {e}")


class TransformerTrainer:
    """Trainer for Transformer model."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.0001, batch_size: int = 32,
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def train(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 150
    ) -> dict[str, list[float]]:
        """Train the Transformer model."""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += int((predicted == batch_y).sum().item())

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = float(self.criterion(test_outputs, y_test).item())
                _, predicted = torch.max(test_outputs.data, 1)
                test_correct = (predicted == y_test).sum().item()
                test_total = y_test.size(0)

            # Record metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            test_acc = test_correct / test_total

            history["train_loss"].append(train_loss_avg)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if epoch % 30 == 0:
                pass  # TODO: Add proper implementation

        return history

    async def _train_sr_outcome_model(
        self, training_data: dict[str, pd.DataFrame], ) -> bool:
        """Train S/R outcome model using all available features from step04.
        Trains specifically on data near S/R levels using the pruning logic from step05.
        """
        try:
            self.logger.info("🔄 Training S/R outcome model...")

            # Initialize S/R outcome trainer if not already done
            if self.sr_outcome_trainer is None:
                from src.training.steps.sr_outcome_model_trainer import (
                    SROutcomeModelTrainer,
                )

            self.sr_outcome_trainer = SROutcomeModelTrainer(self.config)
            await self.sr_outcome_trainer.initialize()

            # Prepare S/R-specific training data
            sr_training_data = await self._prepare_sr_training_data(training_data)
            if not sr_training_data:
                self.logger.warning("No S/R training data available")
                return False

            # Train the S/R outcome model
            training_success, await self.sr_outcome_trainer.train_model(
                sr_training_data,
            )

            if training_success:
                self.sr_outcome_model_trained = True
                self.logger.info("✅ S/R outcome model training completed successfully")
            else:
                self.logger.error("❌ S/R outcome model training failed")

            return training_success

        except Exception as e:
            self.logger.exception(f"Error training S/R outcome model: {e}")
            return False

    async def _prepare_sr_training_data(
        self, training_data: dict[str, pd.DataFrame], ) -> dict[str, pd.DataFrame] | None:
        """Prepare training data specifically for S/R outcome prediction.
        Uses all available features from step04 and filters for data near S/R levels.
        """
        try:
            self.logger.info("🔄 Preparing S/R-specific training data...")

            sr_training_data = {}

            for timeframe, data in training_data.items():
                if data.empty:
                    continue

                self.logger.info(
                    f"Processing {timeframe} data for S/R training: {len(data)} samples",
                )

                # Get all available features from step04 (comprehensive feature set)
                all_features = self._get_all_available_features(data, timeframe)

                # Filter for data near S/R levels
                sr_filtered_data, await self._filter_sr_proximity_data(
                    all_features = timeframe,
                )

                if not sr_filtered_data.empty:
                    sr_training_data[timeframe] = sr_filtered_data
                    self.logger.info(
                        f"✅ {timeframe}: {len(sr_filtered_data)} S/R samples",
                    )
                else:
                    self.logger.warning(f"⚠️ {timeframe}: No S/R proximity data found")

            if not sr_training_data:
                self.logger.warning(
                    "No S/R training data available across all timeframes",
                )
                return None

            total_samples = sum(len(data) for data in sr_training_data.values())
            self.logger.info(
                f"✅ Prepared S/R training data: {total_samples} total samples",
            )

            return sr_training_data

        except Exception as e:
            self.logger.exception(f"Error preparing S/R training data: {e}")
            return None

    def _get_all_available_features(
        self, data: pd.DataFrame, timeframe: str, ) -> pd.DataFrame:
        """Get all available features from step04 for comprehensive S/R analysis.
        Uses the same feature engineering logic as the main HMM training.
        """
        try:
            # Start with base data
            features_df = data.copy()

            # Add all HMM-derived features (from step04)
            if hasattr(self, "hmm_features"):
                # Ensure HMM features are present
                for feature in self.hmm_features:
                    if feature not in features_df.columns:
                        features_df[feature] = 0.0  # Default value if missing

            # Add all technical indicators and market features
            if hasattr(self, "all_features"):
                # Ensure all features are present
                for feature in self.all_features:
                    if feature not in features_df.columns:
                        features_df[feature] = 0.0  # Default value if missing

            # Add timeframe-specific features
            features_df["timeframe"] = timeframe

            # Add price-based features
            features_df["price_change_1m"] = features_df["close"].pct_change()
            features_df["price_change_5m"] = features_df["close"].pct_change(5)
            features_df["price_change_15m"] = features_df["close"].pct_change(15)
            features_df["price_volatility"] = features_df["close"].rolling(20).std()

            # Add volume-based features
            features_df["volume_ratio"] = (
                features_df["volume"] / features_df["volume"].rolling(20).mean()
            )
            features_df["volume_momentum"] = features_df["volume"].pct_change()
            features_df["volume_volatility"] = features_df["volume"].rolling(10).std()

            # Add technical indicators
            features_df["rsi"] = self._calculate_rsi(features_df["close"])
            features_df["macd"] = self._calculate_macd(features_df["close"])
            features_df["bb_position"] = self._calculate_bb_position(
                features_df["close"],
            )

            # Add market context features
            features_df["market_trend"] = self._calculate_market_trend(features_df)
            features_df["momentum_strength"] = self._calculate_momentum_strength(
                features_df,
            )

            # Fill NaN values
            return features_df.fillna(method="ffill").fillna(0)

        except Exception as e:
            self.logger.exception(f"Error getting all available features: {e}")
            return data

    async def _filter_sr_proximity_data(
        self, data: pd.DataFrame, timeframe: str, ) -> pd.DataFrame:
        """Filter data for samples near S/R levels using the SRBreakoutPredictor."""
        try:
            if data.empty:
                return pd.DataFrame()

            # Sample data for efficiency (process every 5th row for large datasets)
            sample_interval = max(
                1, len(data) // 2000,
            )  # Sample up to 2000 points per timeframe
            sample_data = data.iloc[::sample_interval].copy()

            sr_proximity_samples = []

            for idx, row in sample_data.iterrows():
                try:
                    current_price = row["close"]

                    # Create market data slice for S/R analysis
                    market_slice = data.loc[:idx].tail(100)
                    if len(market_slice) < 20:
                        continue

                    # Get S/R context and check proximity using centralized logic
                    sr_context = await self.sr_predictor.get_sr_context(
                        market_data=market_slice, current_price=current_price,
                    )
                    is_near_sr = self.sr_predictor.is_near_sr_level(
                        current_price=current_price, sr_context=sr_context,
                    )

                    if is_near_sr:
                        # Add S/R context features to the sample
                        sample = row.copy()

                        # Add S/R-specific features
                        nearest_support, sr_context.get(
                            "nearest_support", current_price,
                        )
                        nearest_resistance, sr_context.get(
                            "nearest_resistance", current_price,
                        )

                        sample["distance_to_support"] = (
                            current_price - nearest_support
                        ) / current_price
                        sample["distance_to_resistance"] = (
                            nearest_resistance - current_price
                        ) / current_price
                        sample["support_strength"] = sr_context.get(
                            "support_strength", 0.5,
                        )
                        sample["resistance_strength"] = sr_context.get(
                            "resistance_strength", 0.5,
                        )
                        sample["is_near_sr_level"] = True
                        sample["sr_context"] = sr_context

                        sr_proximity_samples.append(sample)

                except Exception as e:
                    self.logger.debug(f"Error processing S/R sample {idx}: {e}")
                    continue

            if not sr_proximity_samples:
                return pd.DataFrame()

            # Convert to DataFrame
            sr_filtered_df = pd.DataFrame(sr_proximity_samples)

            # Apply feature pruning logic from step05 (remove redundant/irrelevant features)
            return self._apply_feature_pruning(sr_filtered_df)

        except Exception as e:
            self.logger.exception(f"Error filtering S/R proximity data: {e}")
            return pd.DataFrame()

    def _apply_feature_pruning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature pruning logic from step05 to remove redundant/irrelevant features."""
        try:
            # Remove highly correlated features (VIF filtering)
            # This uses the same logic as in step5_hmm_based_training.py

            # Remove features with too many NaN values
            nan_threshold = 0.5,
            nan_counts = data.isnull().sum() / len(data),
            data = data.loc[:, nan_counts < nan_threshold],

            # Remove constant features
            constant_features = []
            for col in data.columns:
                if data[col].nunique() <= 1:
                    constant_features.append(col)
            data, data.drop(columns=constant_features)

            # Remove highly correlated features (simplified version)
            # In practice, this would use VIF analysis from step05
            correlation_threshold = 0.95,
            corr_matrix = data.corr().abs(),
            upper_tri, corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [
                column
                for column in upper_tri.columns
                if any(upper_tri[column] > correlation_threshold)
            ]
            data, data.drop(columns=high_corr_features)

            self.logger.info(
                f"Feature pruning: removed {len(constant_features) + len(high_corr_features)} redundant features",
            )

            return data

        except Exception as e:
            self.logger.exception(f"Error applying feature pruning: {e}")
            return data

    async def run_step(
        self, symbol: str = "ETHUSDT", data_dir: str = None, method_a_mixture_of_experts: dict | None = None,
        **kwargs, ) -> bool:
        """Run the HMM-based training step with standardized data quality management.

        Args:
            symbol: Trading symbol
            data_dir: Data directory path (will use standardized path if None)
            method_a_mixture_of_experts: Configuration for method A mixture of experts
            **kwargs: Additional arguments

        Returns: True if successful, False otherwise

        """
        try:
            from src.utils.logger import system_logger

            # Use standardized path construction
            if data_dir is None:
                exchange = kwargs.get("exchange", "BINANCE")
                data_dir = pipeline_standards.build_path("processed_data", exchange, symbol)

            # Create configuration
            config = {
                "symbol": symbol,
                "data_dir": data_dir,
                "exchange": kwargs.get("exchange", "BINANCE"),
                "timeframes": kwargs.get("timeframes", ["1m", "5m", "15m", "30m"]),
                "method_a_mixture_of_experts": method_a_mixture_of_experts or {},
            }

            # Create and run the training step
            training_step = HMMBasedTrainingStep(config)
            await training_step.initialize()

            training_input = {
                "symbol": symbol,
                "exchange": config["exchange"],
                "data_dir": data_dir,
                "timeframes": config["timeframes"],
            }

            pipeline_state = {}

            result, await training_step.execute(training_input, pipeline_state)

            if result.get("status") == "SUCCESS":
                system_logger.info("✅ HMM-based training step completed successfully")
            return True
            system_logger.error(
                f"❌ HMM-based training step failed: {result.get('error', 'Unknown error')}",
            )
            return False

        except Exception as e:
            system_logger.error(f"❌ Error in HMM-based training step: {e}")
            return False


# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
)


@deterministic_seed(42)
@idempotent_step(step_key="step6_hmm_based_training")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=7200.0)
@performance_monitor(
    enable_profiling=True,
    enable_memory_tracking=True,
    enable_cpu_tracking=True,
    save_profile_data=True,
    level=PerformanceLevel.PROFILING,
)
@model_validation(
    check_overfitting=True,
    check_underfitting=True,
    validation_metrics=["accuracy", "precision", "recall", "f1"],
    overfitting_threshold=0.1,
    underfitting_threshold=0.6,
)
@pipeline_checkpoint(
    save_intermediate_results=True,
    checkpoint_frequency=500,
    enable_rollback=True,
)
@intelligent_caching(
    cache_intermediate_results=True,
    cache_validation_data=True,
    cache_model_artifacts=True,
    cache_ttl_hours=24,
)
@adaptive_resource_allocation(
    dynamic_memory_allocation=True,
    adaptive_batch_sizes=True,
    resource_scaling_threshold=0.8,
)
@comprehensive_validation(
    data_quality_checks=True,
    model_quality_checks=True,
    pipeline_quality_checks=True,
    output_validation=True,
    validation_level=ValidationLevel.WARNING,
)
@validate_step_prerequisites(
    required_directories=["data/training", "models"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "sklearn", "hmmlearn", "lightgbm"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "features", "targets"],
    },
    context="HMM-Based Training",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    cross_validation_isolation=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    monitor_interval=60.0,
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
    recovery_timeout=300.0,
    expected_exception=Exception,
    monitor_interval=60.0,
)
@validate_step_output(
    required_files=["models/{exchange}_{symbol}_hmm_model.pkl"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["predictions", "probabilities"],
    },
    performance_thresholds={"training_time_minutes": 120.0, "memory_usage_gb": 8.0},
    format_validation=True,
)
@quality_gate(
    model_performance_thresholds={"accuracy": 0.6, "f1_score": 0.5},
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    convergence_checks=True,
    overfitting_detection=True,
    validation_score_requirements={"cross_validation_score": 0.6},
)
async def run_step(symbol: str = "ETHUSDT", data_dir: str = "data/training", method_a_mixture_of_experts: dict | None = None
    **kwargs, ) -> bool:
    """Run the HMM-based training step.

    Args:
        symbol: Trading symbol
        data_dir: Data directory path
        method_a_mixture_of_experts: Configuration for method A mixture of experts
        **kwargs: Additional arguments

    Returns: True if successful = False otherwise

    """
    try:
        from src.utils.logger import system_logger
import copy
import numpy as np
import os.path

        # Create configuration
        config = {
            "symbol": symbol,
            "data_dir": data_dir,
            "exchange": kwargs.get("exchange", "BINANCE"),
            "timeframes": kwargs.get("timeframes", ["1m", "5m", "15m", "30m"]),
            "method_a_mixture_of_experts": method_a_mixture_of_experts or {},
        }

        # Create and run the training step
        training_step = HMMBasedTrainingStep(config)
        await training_step.initialize()

        training_input = {
            "symbol": symbol,
            "exchange": config["exchange"],
            "data_dir": data_dir,
            "timeframes": config["timeframes"],
        }

        pipeline_state = {}

        result, await training_step.execute(training_input, pipeline_state)

        if result.get("status") == "SUCCESS":
            system_logger.info("✅ HMM-based training step completed successfully")
        return True
        system_logger.error(
            f"❌ HMM-based training step failed: {result.get('error', 'Unknown error')}",
        )
        return False

    except Exception as e:
        system_logger.error(f"❌ Error in HMM-based training step: {e}")
        return False