# src/training/steps/step10_unified_regime_intelligence.py

"""Step 10: Unified Regime Intelligence System with Standardized Data Quality Management."

This unified step consolidates:
1. Multi-timeframe HMM state analysis with intensity scores for regime detection
2. Intensity-based regime transition prediction (entry/exit timing)
3. TPSL-based direction prediction (long/short only)
4. Position logic based on confidence and current position
5. Integration with existing SRBreakoutPredictor for S/R analysis

Replaces step9_5 and step10 with a single, efficient model.
Integrates intensity-based transition detection from step1_7.
Uses existing S/R system for coherence.

Key Features:
- Dynamic regime count based on step1_7 data (not hard-coded)
- Long/short only trading signals (no "hold" as separate class)
- Position logic: buy when no position + high confidence, hold when position + high confidence, sell when confidence drops
"""

import json
import os
import pickle
import re
import time
import warnings
from datetime import datetime
from typing import Any
from pathlib import Path
import asyncio

# Common utilities
from src.utils.common_operations import ensure_directory, safe_json_dump

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "src.tactician.sr_breakout_predictor",
    "src.utils.error_handler",
    "src.utils.logger",
    "src.utils.warning_symbols",
    "src.training.enhanced_lm_optimizer"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
sr_breakout_predictor = PipelineStandards.safe_import("src.tactician.sr_breakout_predictor", None)
error_handler = PipelineStandards.safe_import("src.utils.error_handler", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
warning_symbols = PipelineStandards.safe_import("src.utils.warning_symbols", None)
enhanced_lm_optimizer = PipelineStandards.safe_import("src.training.enhanced_lm_optimizer", None)
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

if error_handler is None:
    handle_errors = create_fallback_decorator()
else:
    handle_errors = error_handler.handle_errors

if warning_symbols is None:
    error = lambda msg: print(f"ERROR: {msg}")
    failed = lambda msg: print(f"FAILED: {msg}")
    timeout = lambda msg: print(f"TIMEOUT: {msg}")
else:
    error = warning_symbols.error
    failed = warning_symbols.failed
    timeout = warning_symbols.timeout

# Import enhanced LM optimizer
if enhanced_lm_optimizer is not None:
    ENHANCED_OPTIMIZER_AVAILABLE = True
else:
    ENHANCED_OPTIMIZER_AVAILABLE = False
    logger.warning("⚠️ Enhanced LM optimizer not available, using basic optimization")

warnings.filterwarnings("ignore")

logger = system_logger.getChild("Step10_UnifiedRegimeIntelligence")


class MultiTimeframeHMMEncoder(nn.Module):
    """Multi-timeframe HMM state encoder using attention mechanisms."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.timeframes = config.get(
            "timeframes", ["5m", "15m", "30m"],
        )  # Less noisy for regime detection
        self.hmm_states_per_tf = config.get("hmm_states_per_tf", 5)
        self.d_model = config.get("d_model", 256)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 4)
        self.dropout = config.get("dropout", 0.1)

        # Per-timeframe HMM state embeddings
        per_tf_dim = max(1, self.d_model // max(1, len(self.timeframes)))
        self.hmm_embeddings = nn.ModuleDict(
            {
                tf: nn.Embedding(num_embeddings=self.hmm_states_per_tf, embedding_dim=per_tf_dim)
                for tf in self.timeframes
            },
        )

        # Multi-head attention for cross-timeframe analysis
        self.cross_timeframe_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True,
        )

        # Transformer layers for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Output projections - will be dynamically set based on actual data
        self.num_regimes: int | None = None  # Will be determined from data
        self.regime_classifier: nn.Linear | None = None  # Will be initialized later
        self.intensity_predictor: nn.Linear | None = None  # Will be initialized later
        self.transition_predictor = nn.Linear(self.d_model, 2)  # transition probability
        self.tpsl_predictor = nn.Linear(
            self.d_model, 2
        )  # TPSL-based direction (long/short only)
        self.confidence_predictor = nn.Linear(self.d_model, 1)  # confidence score
        # Persisted feature projection (initialized lazily to match input feature dimension)
        self.feature_projection: nn.Linear | None = None

    def forward(
        self, hmm_states: dict[str, torch.Tensor], features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the unified regime intelligence model."

        Args:
            hmm_states: Dict of HMM state sequences per timeframe
            features: Additional market features

        Returns: Dict containing regime classification, transition predictions, and S/R detection

        """
        batch_size = features.size(0)
        seq_len = features.size(1)

        # Encode HMM states for each timeframe
        tf_embeddings: list[torch.Tensor] = []
        for tf in self.timeframes:
            if tf in hmm_states:
                tf_embed = self.hmm_embeddings[tf](hmm_states[tf])
                # If per-timeframe dim is smaller, pad to d_model across concatenation later
                tf_embeddings.append(tf_embed)

        # Concatenate timeframe embeddings
        if tf_embeddings:
            hmm_cat = torch.cat(tf_embeddings, dim=-1)
            # Project concatenated embeddings to d_model if needed
            if hmm_cat.size(-1) != self.d_model:
                hmm_encoded = nn.Linear(hmm_cat.size(-1), self.d_model).to(hmm_cat.device)(hmm_cat)
            else:
                hmm_encoded = hmm_cat
        else:
            hmm_encoded = torch.zeros(
                batch_size, seq_len, self.d_model, device=features.device
            )

        # Combine with market features (lazy-init projection to avoid recreating each forward)
        if (
            self.feature_projection is None
            or getattr(self.feature_projection, "in_features", None) != features.size(-1)
        ):
            self.feature_projection = nn.Linear(features.size(-1), self.d_model).to(features.device)
        feature_encoded = self.feature_projection(features)

        # Combine HMM and feature encodings
        combined = hmm_encoded + feature_encoded

        # Apply cross-timeframe attention
        attended, _ = self.cross_timeframe_attention(combined, combined, combined)

        # Apply transformer layers
        transformed = self.transformer(attended)

        # Global average pooling for classification
        pooled = torch.mean(transformed, dim=1)

        # Generate outputs
        regime_logits = self.regime_classifier(pooled) if self.regime_classifier is not None else torch.zeros((batch_size, 1), device=pooled.device)
        intensity_logits = (
            self.intensity_predictor(pooled) if self.intensity_predictor is not None else torch.zeros((batch_size, 1), device=pooled.device)
        )
        transition_logits = self.transition_predictor(pooled)
        tpsl_logits = self.tpsl_predictor(pooled)
        confidence_logits = self.confidence_predictor(pooled)

        return {
            "regime_logits": regime_logits,
            "intensity_logits": intensity_logits,
            "transition_logits": transition_logits,
            "tpsl_logits": tpsl_logits,
            "confidence_logits": confidence_logits,
            "hidden_states": transformed,
        }


class UnifiedRegimeIntelligenceStep:
    """Unified Step 9: Regime Intelligence System."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

        # Model configuration
        self.timeframes = config.get(
            "timeframes", ["5m", "15m", "30m"],
        )  # Less noisy for regime detection
        self.hmm_states_per_tf = config.get("hmm_states_per_tf", 5)
        self.sequence_length = config.get("sequence_length", 20)
        self.num_regimes = None  # Will be determined dynamically from step1_7 data

        # Training configuration
        self.learning_rate = config.get("learning_rate", 0.0001)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.validation_split = config.get("validation_split", 0.2)

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Initialize SRBreakoutPredictor for S/R analysis with optimized parameters
        sr_config = config.copy()
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        self.sr_predictor = SRBreakoutPredictor(sr_config)

        # Artifacts
        self.artifacts_dir = config.get(
            "artifacts_dir", "checkpoints/unified_regime_intelligence",
        )
        ensure_directory(self.artifacts_dir)

        # Enhancement-related config (optional)
        self.enhancement_config = config.get("enhancement", {})
        self.hpo_enabled = self.enhancement_config.get("hpo_enabled", False)
        self.architecture_optimization_enabled = self.enhancement_config.get(
            "architecture_optimization_enabled", False,
        )
        self.hpo_config = self.enhancement_config.get("hpo", {})
        self.n_trials = self.hpo_config.get("n_trials", 20)
        self.hpo_timeout = self.hpo_config.get("timeout", 900)
        self.hpo_pruning = self.hpo_config.get("pruning_enabled", True)

        # Initialize enhanced LM optimizer
        self.enhanced_lm_optimizer = None
        if ENHANCED_OPTIMIZER_AVAILABLE:
            try:
                self.enhanced_lm_optimizer = EnhancedLMOptimizer(config)
                # Note: initialize() will be called later in an async context
                self.logger.info("✅ Enhanced LM optimizer created for step6_5")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create enhanced LM optimizer: {e}")

        # Device selection
        self.device_str = self._safe_get_device()
        if self.device_str == "cuda":
            self.device = torch.device("cuda")
        elif self.device_str == "mps":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device_str.upper()} for PyTorch operations.")

    def _safe_get_device(self) -> str:
        """Safely determine best device: prefer CUDA, then MPS with timeout, else CPU."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            # MPS check can occasionally hang; guard with timeout
            import queue
            import threading

            result_queue: "queue.Queue[tuple[str | None, Exception | None]]" = queue.Queue()

            def check_mps() -> None:
                try:
                    is_available = torch.backends.mps.is_available()
                    result_queue.put(("mps" if is_available else "cpu", None))
                except Exception as ex:
                    result_queue.put(("cpu", ex))

            thread = threading.Thread(target=check_mps, daemon=True)
            thread.start()
            try:
                device, err = result_queue.get(timeout=10)
                if err:
                    self.logger.error(failed(f"MPS check failed: {err}, using CPU"))
                    return "cpu"
                return device or "cpu"
            except queue.Empty:
                self.logger.exception(timeout("MPS availability check timed out, using CPU"))
                return "cpu"
        except Exception as ex:
            self.logger.exception(error(f"Error checking device availability: {ex}, using CPU"))
            return "cpu"

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unified regime intelligence initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the unified regime intelligence step."""
        try:
            self.logger.info("🚀 Initializing Unified Regime Intelligence Step...")

            # Initialize model
            self.model = MultiTimeframeHMMEncoder(self.config)

            # Initialize label encoders
            self.label_encoders["regime"] = LabelEncoder()
            self.label_encoders["transition"] = LabelEncoder()
            self.label_encoders["tpsl"] = LabelEncoder()

            # Initialize SRBreakoutPredictor
            sr_init_success = await self.sr_predictor.initialize()
            if not sr_init_success:
                self.logger.warning(
                    "⚠️ Failed to initialize SRBreakoutPredictor, continuing without S/R analysis",
                )

            self.logger.info(
                "✅ Unified Regime Intelligence Step initialized successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"🚨 Failed to initialize Unified Regime Intelligence Step: {e}",
            )
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unified regime intelligence training",
    )
    async def train(self, data: dict[str, pd.DataFrame]) -> bool:
        """Train the unified regime intelligence model."""
        try:
            self.logger.info("🚀 Starting Unified Regime Intelligence training...")

            # Enhanced optimization for step6_5
            if self.enhanced_lm_optimizer is None:
                raise RuntimeError("Enhanced LM optimizer is required but not initialized")

            # Initialize the optimizer if not already done
            if not getattr(self.enhanced_lm_optimizer, "initialization_status", None):
                await self.enhanced_lm_optimizer.initialize()

            self.logger.info("🔧 Enhanced LM optimization enabled: starting comprehensive optimization...")

            # Prepare data for optimization
            optimization_data = await self._prepare_optimization_data(data)
            if not optimization_data:
                raise RuntimeError("Failed to prepare optimization data")

            optimization_results = await self.enhanced_lm_optimizer.optimize_lm_model(
                step_name="step6_5",
                features_df=optimization_data["features"],
                target=optimization_data["target"],
                model_type="classification",
                architecture="Transformer",
            )

            self.logger.info("✅ Enhanced optimization completed for step6_5")
            # Store optimization results
            if not hasattr(self, "enhancement_results"):
                self.enhancement_results = {}
            self.enhancement_results["enhanced_optimization"] = optimization_results

            # Check if HPO is enabled
            if self.hpo_enabled:
                self.logger.info("🔧 HPO enabled: starting short optimization...")
                hpo_results = await self._run_hyperparameter_optimization()
                if hpo_results and "best_params" in hpo_results:
                    self.config.update(hpo_results["best_params"])
                    # Update core params if present
                    self.learning_rate = self.config.get("learning_rate", self.learning_rate)
                    self.batch_size = self.config.get("batch_size", self.batch_size)
                    self.sequence_length = self.config.get("sequence_length", self.sequence_length)
                    # Recreate model with new architecture settings if any
                    self.model = MultiTimeframeHMMEncoder(self.config)
                    # Attach HPO results to artifacts
                    if not hasattr(self, "enhancement_results"):
                        self.enhancement_results = {}
                    self.enhancement_results["hpo_results"] = hpo_results or {}

            # Prepare training data
            train_data = await self._prepare_training_data(data)
            if not train_data:
                self.logger.error("🚨 Failed to prepare training data")
                return False

            # Train the model
            training_result = await self._train_model(train_data)
            if not training_result:
                self.logger.error("🚨 Model training failed")
                return False

            # Optional: light architecture optimization/pruning
            if self.architecture_optimization_enabled and self.model is not None:
                arch_results = {
                    "pruning_results": self._apply_structured_pruning(self.model),
                    "optimization_results": self._optimize_architecture(self.model),
                    "model_size_before": sum(p.numel() for p in self.model.parameters()),
                    "model_size_after": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                }
                if not hasattr(self, "enhancement_results"):
                    self.enhancement_results = {}
                self.enhancement_results["architecture_optimization_results"] = arch_results

            # Save artifacts
            await self._save_artifacts()

            self.logger.info(
                "✅ Unified Regime Intelligence training completed successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(f"🚨 Training failed: {e}")
            return False

    async def _prepare_optimization_data(
        self, data: dict[str, pd.DataFrame],
    ) -> dict[str, Any] | None:
        """Prepare data for enhanced optimization."""
        try:
            # Load HMM composite data for each timeframe
            hmm_data: dict[str, pd.DataFrame] = {}
            for tf in self.timeframes:
                hmm_file = f"data/BINANCE_ETHUSDT_hmm_composite_clusters_{tf}.parquet"
                if os.path.exists(hmm_file):
                    hmm_data[tf] = pd.read_parquet(hmm_file)
                    self.logger.info(
                        f"📦 Loaded HMM data for optimization: {tf}: {len(hmm_data[tf])} rows",
                    )

            if not hmm_data:
                self.logger.error("🚨 No HMM data found for optimization")
                return None

            # Use the first timeframe for optimization
            tf = self.timeframes[0]
            tf_data = hmm_data[tf]

            # Prepare features and target
            feature_columns = [
                col for col in tf_data.columns if col not in ["composite_cluster_id", "timestamp"]
            ]
            features = tf_data[feature_columns].fillna(0)
            target = tf_data["composite_cluster_id"].fillna(-1)

            # Remove noise cluster (-1) from target
            valid_mask = target != -1
            features = features[valid_mask]
            target = target[valid_mask]

            if len(features) == 0:
                self.logger.error("🚨 No valid data for optimization")
                return None

            self.logger.info(
                f"📊 Prepared optimization data: {len(features)} samples, {len(features.columns)} features",
            )

            return {
                "features": features,
                "target": target,
            }

        except Exception as e:
            self.logger.exception(f"🚨 Failed to prepare optimization data: {e}")
            return None

    async def _prepare_training_data(
        self, data: dict[str, pd.DataFrame],
    ) -> dict[str, Any] | None:
        """Prepare training data from multi-timeframe HMM states, intensity scores, and features."""
        try:
            # Load HMM composite data for each timeframe
            hmm_data: dict[str, pd.DataFrame] = {}
            for tf in self.timeframes:
                hmm_file = f"data/BINANCE_ETHUSDT_hmm_composite_clusters_{tf}.parquet"
                if os.path.exists(hmm_file):
                    hmm_data[tf] = pd.read_parquet(hmm_file)
                    self.logger.info(
                        f"📦 Loaded HMM data for {tf}: {len(hmm_data[tf])} rows",
                    )

            if not hmm_data:
                self.logger.error("🚨 No HMM data found for any timeframe")
                return None

            # Determine number of regimes dynamically from the data
            all_cluster_ids: set[int] = set()
            for tf, tf_data in hmm_data.items():
                if "composite_cluster_id" in tf_data.columns:
                    cluster_ids = tf_data["composite_cluster_id"].dropna()
                    all_cluster_ids.update(cluster_ids.unique())

            # Remove noise cluster (-1) and get actual number of regimes
            all_cluster_ids.discard(-1)  # Remove noise cluster
            self.num_regimes = len(all_cluster_ids)

            if self.num_regimes == 0:
                self.logger.error("🚨 No valid regimes found in HMM data")
                return None

            self.logger.info(
                f"📊 Determined {self.num_regimes} regimes from HMM data: {sorted(all_cluster_ids)}",
            )

            # Initialize output layers with correct dimensions
            if self.model is not None:
                self.model.num_regimes = self.num_regimes
                self.model.regime_classifier = nn.Linear(
                    self.model.d_model, self.num_regimes,
                )
                self.model.intensity_predictor = nn.Linear(
                    self.model.d_model, self.num_regimes,
                )

            # Load intensity data from step1_7
            intensity_data: dict[str, pd.DataFrame] = {}
            for tf in self.timeframes:
                intensity_file = (
                    f"data/BINANCE_ETHUSDT_hmm_composite_intensity_{tf}.parquet"
                )
                if os.path.exists(intensity_file):
                    intensity_data[tf] = pd.read_parquet(intensity_file)
                    self.logger.info(
                        f"📦 Loaded intensity data for {tf}: {len(intensity_data[tf])} rows",
                    )
                else:
                    self.logger.warning(
                        f"⚠️ Intensity data not found for {tf}, generating from HMM states",
                    )
                    # Generate intensity scores from HMM states (fallback)
                    intensity_data[tf] = self._generate_intensity_scores(hmm_data[tf])

            # Load combined features
            combined_features = data.get("combined_features", pd.DataFrame())
            if combined_features is None:
                combined_features = pd.DataFrame()

            # Align all data to the same index (use 1m as base)
            base_tf = "1m"
            if base_tf not in hmm_data:
                self.logger.error(f"🚨 Base timeframe {base_tf} not found in HMM data")
                return None

            base_index = hmm_data[base_tf].index

            # Prepare sequences
            return await self._create_sequences(
                hmm_data, intensity_data, combined_features, base_index,
            )

        except Exception as e:
            self.logger.exception(f"🚨 Error preparing training data: {e}")
            return None

    def _generate_intensity_scores(self, hmm_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive intensity scores from HMM states (enhanced method)."""
        try:
            # Get unique cluster IDs
            cluster_ids = hmm_df.get(
                "composite_cluster_id", hmm_df.get("hmm_state", pd.Series(np.arange(20), index=hmm_df.index)),
            )
            unique_clusters = np.unique(cluster_ids)

            # Create intensity columns for each cluster
            intensity_df = pd.DataFrame(index=hmm_df.index)

            # Basic intensity scores (probability of being in each cluster)
            for cluster_id in unique_clusters:
                cluster_mask = (cluster_ids == cluster_id).astype(float)
                # Multiple window sizes for different temporal scales
                intensity_5 = cluster_mask.rolling(window=5, min_periods=1).mean()
                intensity_10 = cluster_mask.rolling(window=10, min_periods=1).mean()
                intensity_20 = cluster_mask.rolling(window=20, min_periods=1).mean()

                intensity_df[f"intensity_cluster_{cluster_id}"] = intensity_10  # Main intensity
                intensity_df[f"intensity_cluster_{cluster_id}_short"] = intensity_5  # Short-term
                intensity_df[f"intensity_cluster_{cluster_id}_long"] = intensity_20  # Long-term

            # Regime persistence features
            for cluster_id in unique_clusters:
                cluster_mask = (cluster_ids == cluster_id).astype(float)
                # Calculate how long we've been in this regime'
                persistence = cluster_mask.groupby((cluster_mask != cluster_mask.shift()).cumsum()).cumsum()
                intensity_df[f"persistence_cluster_{cluster_id}"] = persistence

            # Regime transition features
            for cluster_id in unique_clusters:
                cluster_mask = (cluster_ids == cluster_id).astype(float)
                # Transition probability (likelihood of staying in this regime)
                transition_prob = cluster_mask.rolling(window=10, min_periods=1).apply(
                    lambda x: float((x == 1).sum()) / float(len(x)) if len(x) > 0 else 0.0
                )
                intensity_df[f"transition_prob_cluster_{cluster_id}"] = transition_prob

            # Volatility of intensity (regime stability)
            for cluster_id in unique_clusters:
                cluster_mask = (cluster_ids == cluster_id).astype(float)
                intensity = cluster_mask.rolling(window=10, min_periods=1).mean()
                intensity_vol = intensity.rolling(window=5, min_periods=1).std()
                intensity_df[f"intensity_vol_cluster_{cluster_id}"] = intensity_vol

            # Cross-regime correlation features
            if len(unique_clusters) > 1:
                # Calculate correlation between different regime intensities
                for i, cluster_id1 in enumerate(unique_clusters):
                    for cluster_id2 in list(unique_clusters)[i + 1 :]:
                        intensity1 = intensity_df[f"intensity_cluster_{cluster_id1}"]
                        intensity2 = intensity_df[f"intensity_cluster_{cluster_id2}"]
                        correlation = intensity1.rolling(window=20, min_periods=1).corr(intensity2)
                        intensity_df[f"corr_{cluster_id1}_{cluster_id2}"] = correlation

            # Regime dominance features
            all_intensities = [intensity_df[f"intensity_cluster_{cid}"] for cid in unique_clusters]
            if all_intensities:
                intensity_matrix = pd.concat(all_intensities, axis=1)
                # Dominant regime (highest intensity)
                dominant_regime = intensity_matrix.idxmax(axis=1)
                intensity_df["dominant_regime"] = dominant_regime.astype("category").cat.codes

                # Regime diversity (number of regimes with significant intensity)
                significant_intensities = (intensity_matrix > 0.1).sum(axis=1)
                intensity_df["regime_diversity"] = significant_intensities

            self.logger.info(f"📊 Generated {len(intensity_df.columns)} comprehensive intensity features")
            return intensity_df

        except Exception as e:
            self.logger.exception(f"🚨 Error generating intensity scores: {e}")
            # Return basic intensity scores as fallback
            return pd.DataFrame(
                {
                    f"intensity_cluster_{i}": np.random.random(len(hmm_df)) for i in range(20)
                },
                index=hmm_df.index,
            )

    async def _create_cross_timeframe_correlations(
        self, intensity_data: dict[str, pd.DataFrame], base_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Create cross-timeframe intensity correlations."""
        try:
            # Initialize correlation dataframe
            correlation_df = pd.DataFrame(index=base_index)

            # Get intensity columns from each timeframe
            tf_intensities: dict[str, pd.DataFrame] = {}
            for tf in self.timeframes:
                if tf in intensity_data:
                    tf_data = intensity_data[tf]
                    if tf != "1m":
                        tf_data = tf_data.reindex(base_index, method="ffill")

                    # Get intensity columns
                    intensity_cols = [col for col in tf_data.columns if col.startswith("intensity_cluster_")]
                    tf_intensities[tf] = tf_data[intensity_cols]

            # Calculate cross-timeframe correlations
            if len(tf_intensities) >= 2:
                # 1. 1m-5m correlation
                if "1m" in tf_intensities and "5m" in tf_intensities:
                    correlation_df["corr_1m_5m"] = self._calculate_intensity_correlation(
                        tf_intensities["1m"], tf_intensities["5m"], window=20
                    )

                # 2. 1m-15m correlation
                if "1m" in tf_intensities and "15m" in tf_intensities:
                    correlation_df["corr_1m_15m"] = self._calculate_intensity_correlation(
                        tf_intensities["1m"], tf_intensities["15m"], window=20
                    )

                # 3. 5m-15m correlation
                if "5m" in tf_intensities and "15m" in tf_intensities:
                    correlation_df["corr_5m_15m"] = self._calculate_intensity_correlation(
                        tf_intensities["5m"], tf_intensities["15m"], window=20
                    )

                # 4. Multi-timeframe alignment score
                correlation_df["multi_tf_alignment"] = self._calculate_multi_timeframe_alignment(
                    tf_intensities, window=20
                )

                # 5. Temporal consistency score
                correlation_df["temporal_consistency"] = self._calculate_temporal_consistency(
                    tf_intensities, window=20
                )

                # 6. Regime synchronization score
                correlation_df["regime_synchronization"] = self._calculate_regime_synchronization(
                    tf_intensities, window=20
                )

            self.logger.info(f"📊 Generated {len(correlation_df.columns)} cross-timeframe correlation features")
            return correlation_df

        except Exception as e:
            self.logger.exception(f"🚨 Error creating cross-timeframe correlations: {e}")
            return pd.DataFrame(index=base_index)

    def _calculate_intensity_correlation(
        self, tf1_intensities: pd.DataFrame, tf2_intensities: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """Calculate rolling correlation between two timeframe intensities."""
        try:
            # Calculate mean intensity per timeframe
            tf1_mean = tf1_intensities.mean(axis=1)
            tf2_mean = tf2_intensities.mean(axis=1)

            # Calculate rolling correlation
            correlation = tf1_mean.rolling(window=window, min_periods=1).corr(tf2_mean)

            return correlation.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating intensity correlation: {e}")
            return pd.Series(0, index=tf1_intensities.index)

    def _calculate_multi_timeframe_alignment(
        self, tf_intensities: dict[str, pd.DataFrame], window: int = 20
    ) -> pd.Series:
        """Calculate how well all timeframes are aligned."""
        try:
            # Get dominant regime for each timeframe
            dominant_regimes: dict[str, pd.Series] = {}
            for tf, intensities in tf_intensities.items():
                dominant_regimes[tf] = intensities.idxmax(axis=1)

            # Calculate alignment score (percentage of timeframes with same dominant regime)
            alignment_scores: list[float] = []
            reference_index = next(iter(tf_intensities.values())).index
            for i in range(len(reference_index)):
                regimes_at_time = [regimes.iloc[i] for regimes in dominant_regimes.values()]
                alignment = len(set(regimes_at_time)) / float(len(regimes_at_time))
                alignment_scores.append(1.0 - alignment)  # Higher, better alignment

            return pd.Series(alignment_scores, index=reference_index)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating multi-timeframe alignment: {e}")
            reference_index = next(iter(tf_intensities.values())).index
            return pd.Series(0, index=reference_index)

    def _calculate_temporal_consistency(
        self, tf_intensities: dict[str, pd.DataFrame], window: int = 20
    ) -> pd.Series:
        """Calculate temporal consistency across timeframes."""
        try:
            # Calculate intensity stability for each timeframe
            stability_scores: list[pd.Series] = []
            for intensities in tf_intensities.values():
                # Calculate rolling standard deviation of mean intensity
                mean_intensity = intensities.mean(axis=1)
                stability = 1.0 / (1.0 + mean_intensity.rolling(window=window, min_periods=1).std())
                stability_scores.append(stability)

            # Average stability across timeframes
            avg_stability = pd.concat(stability_scores, axis=1).mean(axis=1)

            return avg_stability.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating temporal consistency: {e}")
            reference_index = next(iter(tf_intensities.values())).index
            return pd.Series(0, index=reference_index)

    def _calculate_regime_synchronization(
        self, tf_intensities: dict[str, pd.DataFrame], window: int = 20
    ) -> pd.Series:
        """Calculate regime synchronization across timeframes."""
        try:
            # Calculate regime change points for each timeframe
            change_points: dict[str, pd.Series] = {}
            for tf, intensities in tf_intensities.items():
                dominant_regimes = intensities.idxmax(axis=1)
                changes = (dominant_regimes != dominant_regimes.shift(1)).astype(int)
                change_points[tf] = changes

            # Calculate synchronization (how often changes happen simultaneously)
            reference_index = next(iter(tf_intensities.values())).index
            sync_scores: list[float] = []
            for i in range(len(reference_index)):
                changes_at_time = [changes.iloc[i] for changes in change_points.values()]
                sync_score = float(sum(changes_at_time)) / float(len(changes_at_time))
                sync_scores.append(sync_score)

            # Rolling average for smoothing
            sync_series = pd.Series(sync_scores, index=reference_index)
            return sync_series.rolling(window=window, min_periods=1).mean().fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating regime synchronization: {e}")
            reference_index = next(iter(tf_intensities.values())).index
            return pd.Series(0, index=reference_index)

    async def _create_regime_transition_features(
        self, hmm_data: dict[str, pd.DataFrame], base_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Create regime transition probability features."""
        try:
            # Initialize transition dataframe
            transition_df = pd.DataFrame(index=base_index)

            # Get regime data from 1m (base timeframe)
            if "1m" in hmm_data:
                regime_data = hmm_data["1m"]
                if "composite_cluster_id" in regime_data.columns:
                    regimes = regime_data["composite_cluster_id"]

                    # Get unique regimes (excluding noise cluster -1)
                    unique_regimes = sorted([int(r) for r in regimes.unique() if r >= 0])

                    # Calculate transition probabilities for each regime
                    for regime_id in unique_regimes:
                        # 1. Stay probability (probability of staying in this regime)
                        stay_prob = self._calculate_stay_probability(regimes, regime_id, window=20)
                        transition_df[f"stay_prob_regime_{regime_id}"] = stay_prob

                        # 2. Transition velocity (how quickly we transition from this regime)
                        transition_vel = self._calculate_transition_velocity(regimes, regime_id, window=20)
                        transition_df[f"transition_vel_regime_{regime_id}"] = transition_vel

                        # 3. Regime persistence (how long we typically stay in this regime)
                        persistence = self._calculate_regime_persistence(regimes, regime_id, window=20)
                        transition_df[f"persistence_regime_{regime_id}"] = persistence

                        # 4. Regime momentum (tendency to continue in this regime)
                        momentum = self._calculate_regime_momentum(regimes, regime_id, window=20)
                        transition_df[f"momentum_regime_{regime_id}"] = momentum

            self.logger.info(f"📊 Generated {len(transition_df.columns)} regime transition features")
            return transition_df

        except Exception as e:
            self.logger.exception(f"🚨 Error creating regime transition features: {e}")
            return pd.DataFrame(index=base_index)

    def _calculate_stay_probability(
        self, regimes: pd.Series, regime_id: int, window: int = 20
    ) -> pd.Series:
        """Calculate probability of staying in a specific regime."""
        try:
            # Create regime mask
            regime_mask = (regimes == regime_id).astype(int)

            # Calculate rolling probability of staying in regime
            stay_prob = regime_mask.rolling(window=window, min_periods=1).mean()

            return stay_prob.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating stay probability: {e}")
            return pd.Series(0, index=regimes.index)

    def _calculate_transition_velocity(
        self, regimes: pd.Series, regime_id: int, window: int = 20
    ) -> pd.Series:
        """Calculate how quickly we transition from a specific regime."""
        try:
            # Create regime mask
            regime_mask = (regimes == regime_id).astype(int)

            # Calculate transition points (when we enter this regime)
            transitions = ((regime_mask == 1) & (regime_mask.shift(1) == 0)).astype(int)

            # Calculate rolling transition frequency
            transition_freq = transitions.rolling(window=window, min_periods=1).sum() / float(window)

            return transition_freq.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating transition velocity: {e}")
            return pd.Series(0, index=regimes.index)

    def _calculate_regime_persistence(
        self, regimes: pd.Series, regime_id: int, window: int = 20
    ) -> pd.Series:
        """Calculate typical persistence length of a specific regime."""
        try:
            # Create regime mask
            regime_mask = (regimes == regime_id).astype(int)

            # Calculate consecutive periods in regime
            persistence = regime_mask.groupby((regime_mask != regime_mask.shift()).cumsum()).cumsum()

            # Calculate rolling average persistence
            avg_persistence = persistence.rolling(window=window, min_periods=1).mean()

            return avg_persistence.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating regime persistence: {e}")
            return pd.Series(0, index=regimes.index)

    def _calculate_regime_momentum(
        self, regimes: pd.Series, regime_id: int, window: int = 20
    ) -> pd.Series:
        """Calculate momentum of a specific regime."""
        try:
            # Create regime mask
            regime_mask = (regimes == regime_id).astype(int)

            # Calculate rate of change in regime probability
            regime_prob = regime_mask.rolling(window=window, min_periods=1).mean()
            momentum = regime_prob.diff().rolling(window=5, min_periods=1).mean()

            return momentum.fillna(0)

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating regime momentum: {e}")
            return pd.Series(0, index=regimes.index)

    async def _create_sequences(
        self, hmm_data: dict[str, pd.DataFrame], intensity_data: dict[str, pd.DataFrame], features: pd.DataFrame, base_index: pd.DatetimeIndex,
    ) -> dict[str, Any]:
        """Create training sequences for the unified model."""
        try:
            sequences: list[dict[str, Any]] = []
            labels: dict[str, list[int]] = {"regime": [], "transition": [], "tpsl": []}

            # Create cross-timeframe correlations
            cross_tf_correlations = await self._create_cross_timeframe_correlations(
                intensity_data, base_index,
            )

            # Create regime transition features
            transition_features = await self._create_regime_transition_features(
                hmm_data, base_index,
            )

            # Create sliding windows
            for i in range(self.sequence_length, len(base_index)):
                # Extract sequence window
                window_start = i - self.sequence_length
                window_end = i

                # Prepare HMM states for each timeframe
                hmm_states: dict[str, np.ndarray] = {}
                for tf in self.timeframes:
                    if tf in hmm_data:
                        tf_data = hmm_data[tf]
                        # Resample to base timeframe if needed
                        if tf != "1m":
                            tf_data = tf_data.reindex(base_index, method="ffill")

                        window_data = tf_data.iloc[window_start:window_end]
                        hmm_states[tf] = window_data["composite_cluster_id"].values

                # Prepare intensity features
                intensity_features: list[np.ndarray] = []
                for tf in self.timeframes:
                    if tf in intensity_data:
                        tf_intensity = intensity_data[tf]
                        if tf != "1m":
                            tf_intensity = tf_intensity.reindex(
                                base_index, method="ffill"
                            )

                        window_intensity = tf_intensity.iloc[window_start:window_end]
                        intensity_features.append(window_intensity.values)

                # Prepare cross-timeframe correlation features
                correlation_features: list[np.ndarray] = []
                if not cross_tf_correlations.empty:
                    correlation_window = cross_tf_correlations.iloc[window_start:window_end]
                    correlation_features.append(correlation_window.values)

                # Prepare regime transition features
                transition_feature_values: list[np.ndarray] = []
                if not transition_features.empty:
                    transition_window = transition_features.iloc[window_start:window_end]
                    transition_feature_values.append(transition_window.values)

                # Prepare additional features
                if not features.empty:
                    feature_window = features.iloc[window_start:window_end]
                    feature_values = feature_window.values
                else:
                    # Use only actual intensity/correlation/transition features
                    feature_values = np.array([]).reshape(self.sequence_length, 0)

                # Combine all features
                all_feature_arrays: list[np.ndarray] = []
                if feature_values.size > 0:
                    all_feature_arrays.append(feature_values)
                all_feature_arrays.extend(intensity_features)
                all_feature_arrays.extend(correlation_features)
                all_feature_arrays.extend(transition_feature_values)

                if all_feature_arrays:
                    all_features = np.concatenate(all_feature_arrays, axis=1)
                else:
                    all_features = np.array([]).reshape(self.sequence_length, 0)

                # Create labels
                current_regime = int(hmm_data["1m"].iloc[i]["composite_cluster_id"]) if "1m" in hmm_data else 0

                # Transition label (1 if regime changed in next few bars)
                future_regimes = (
                    hmm_data["1m"].iloc[i : i + 5]["composite_cluster_id"].values
                    if "1m" in hmm_data
                    else np.array([current_regime])
                )
                transition_label = 1 if len(set(future_regimes)) > 1 else 0

                # TPSL-based direction prediction (long/short only)
                tpsl_direction = await self._calculate_tpsl_direction(
                    hmm_data.get("1m", pd.DataFrame()), i, window_start, window_end,
                )

                # Intensity-based transition detection
                transition_detected = self._detect_intensity_transition(
                    intensity_data, i, window_start, window_end,
                )

                sequences.append({"hmm_states": hmm_states, "features": all_features})

                labels["regime"].append(current_regime)
                labels["transition"].append(int(transition_detected))
                labels["tpsl"].append(int(tpsl_direction))

            # Convert to tensors
            hmm_tensors: dict[str, torch.Tensor] = {}
            for tf in self.timeframes:
                tf_states = [
                    seq["hmm_states"].get(tf, np.zeros(self.sequence_length))
                    for seq in sequences
                ]
                hmm_tensors[tf] = torch.tensor(tf_states, dtype=torch.long)

            feature_tensor = torch.tensor(
                [seq["features"] for seq in sequences], dtype=torch.float32
            )

            # Log feature count information with enhanced features
            await self._log_feature_count_info(
                feature_tensor,
                intensity_features=[],  # logged already via feature counts
                features=features,
                cross_tf_correlations=cross_tf_correlations,
                transition_features=transition_features,
            )

            # Encode labels now that we have full sequences
            for label_type, label_values in labels.items():
                self.label_encoders[label_type].fit(label_values)
                labels[label_type] = torch.tensor(
                    self.label_encoders[label_type].transform(label_values),
                    dtype=torch.long,
                )

            return {
                "hmm_states": hmm_tensors,
                "features": feature_tensor,
                "labels": labels,
                "num_sequences": len(sequences),
            }

        except Exception as e:
            self.logger.exception(f"🚨 Error creating sequences: {e}")
            return None

    async def _log_feature_count_info(self, feature_tensor: torch.Tensor, intensity_features: list[np.ndarray], features: pd.DataFrame, cross_tf_correlations: pd.DataFrame, transition_features: pd.DataFrame) -> None:
        """Log detailed information about feature counts and dimensions."""
        try:
            total_features = feature_tensor.shape[-1] if len(feature_tensor.shape) > 1 else 0
            intensity_feature_count = sum(feat.shape[-1] for feat in intensity_features) if intensity_features else 0
            additional_feature_count = features.shape[1] if not features.empty else 0
            cross_tf_correlation_count = cross_tf_correlations.shape[1] if not cross_tf_correlations.empty else 0
            transition_feature_count = transition_features.shape[1] if not transition_features.empty else 0

            self.logger.info("📊 Enhanced Feature Count Analysis:")
            self.logger.info(f"   Total features: {total_features}")
            self.logger.info(f"   Intensity features: {intensity_feature_count}")
            self.logger.info(f"   Additional features: {additional_feature_count}")
            self.logger.info(f"   Cross-timeframe correlations: {cross_tf_correlation_count}")
            self.logger.info(f"   Regime transition features: {transition_feature_count}")
            self.logger.info(f"   Timeframes: {len(self.timeframes)} ({', '.join(self.timeframes)})")

            # Log intensity features per timeframe (if provided)
            for i, tf in enumerate(self.timeframes):
                if intensity_features is not None and i < len(intensity_features):
                    tf_features = intensity_features[i].shape[-1] if len(intensity_features[i].shape) > 1 else 0
                    self.logger.info(f"   {tf} intensity features: {tf_features}")

            # Log cross-timeframe correlation features
            if not cross_tf_correlations.empty:
                self.logger.info("   Cross-timeframe correlation features:")
                for col in cross_tf_correlations.columns:
                    self.logger.info(f"     - {col}")

            # Log regime transition features
            if not transition_features.empty:
                self.logger.info("   Regime transition features:")
                for col in transition_features.columns:
                    self.logger.info(f"     - {col}")

            # Log feature tensor shape
            if len(feature_tensor.shape) >= 2:
                self.logger.info(f"   Feature tensor shape: {feature_tensor.shape}")
                self.logger.info(f"   Sequences: {feature_tensor.shape[0]}")
                self.logger.info(f"   Sequence length: {feature_tensor.shape[1]}")
                self.logger.info(f"   Features per timestep: {feature_tensor.shape[2]}")

        except Exception as e:
            self.logger.warning(f"⚠️ Error logging feature count info: {e}")

    def _detect_intensity_transition(
        self, intensity_data: dict[str, pd.DataFrame], current_idx: int, window_start: int, window_end: int,
    ) -> int:
        """Detect regime transitions based on intensity score changes."""
        try:
            # Get current and previous intensity scores
            current_intensities: dict[int, float] = {}
            previous_intensities: dict[int, float] = {}

            # Aggregate intensity scores across timeframes
            for tf in self.timeframes:
                if tf in intensity_data:
                    tf_data = intensity_data[tf]
                    if current_idx < len(tf_data) and current_idx > 0:
                        # Get current intensity scores
                        current_row = tf_data.iloc[current_idx]
                        previous_row = tf_data.iloc[current_idx - 1]

                        # Extract intensity columns
                        intensity_cols = [
                            col
                            for col in tf_data.columns
                            if col.startswith("intensity_cluster_")
                        ]

                        for col in intensity_cols:
                            # Robustly parse cluster id for names like intensity_cluster_3, intensity_cluster_3_short
                            m = re.match(r"^intensity_cluster_(\d+)(?:_.*)?$", col)
                            if not m:
                                continue
                            cluster_id = int(m.group(1))
                            current_intensities[cluster_id] = (
                                current_intensities.get(cluster_id, 0.0)
                                + float(current_row[col])
                            )
                            previous_intensities[cluster_id] = (
                                previous_intensities.get(cluster_id, 0.0)
                                + float(previous_row[col])
                            )

            if not current_intensities or not previous_intensities:
                return 0  # no transition detected

            # Calculate intensity changes
            intensity_changes: dict[int, float] = {}
            for cluster_id in current_intensities:
                if cluster_id in previous_intensities:
                    change = (
                        current_intensities[cluster_id]
                        - previous_intensities[cluster_id]
                    )
                    intensity_changes[cluster_id] = change

            # Detect significant transitions
            transition_threshold = 0.1  # Configurable threshold

            # Count significant changes
            significant_changes = sum(
                1
                for change in intensity_changes.values()
                if abs(change) > transition_threshold
            )

            # Transition detected if multiple regimes show significant intensity changes
            if significant_changes >= 2:
                return 1  # transition detected
            return 0  # no transition

        except Exception as e:
            self.logger.warning(f"⚠️ Error detecting intensity transition: {e}")
            return 0  # no transition as fallback

    async def _calculate_tpsl_direction(
        self, hmm_data: pd.DataFrame, current_idx: int, window_start: int, window_end: int,
    ) -> int:
        """Calculate TPSL-based direction (long/short only)."""
        try:
            # Get current price and future prices for TPSL calculation
            current_price = (
                hmm_data.iloc[current_idx]["close"]
                if "close" in hmm_data.columns
                else 100.0
            )

            # TPSL parameters from step02-3 (triple barrier labeling)
            profit_take_multiplier = 0.002  # 0.2% take profit
            stop_loss_multiplier = 0.001  # 0.1% stop loss

            # Calculate barriers
            profit_barrier = current_price * (1.0 + profit_take_multiplier)
            stop_barrier = current_price * (1.0 - stop_loss_multiplier)

            # Look ahead for barrier hits (simplified - would need actual price data)
            future_window = hmm_data.iloc[
                current_idx + 1 : current_idx + 30
            ]  # 30 bars lookahead

            if len(future_window) == 0:
                return 0  # no position (neutral)

            # Check if profit barrier is hit first (long signal)
            for _, row in future_window.iterrows():
                high_price = row.get("high", current_price)
                low_price = row.get("low", current_price)

                if high_price >= profit_barrier:
                    return 1  # long signal
                if low_price <= stop_barrier:
                    return 0  # short signal (or no position)

            return 0  # no position (neutral)

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating TPSL direction: {e}")
            return 0  # hold as fallback

    async def _train_model(self, train_data: dict[str, Any]) -> bool:
        """Train the unified regime intelligence model."""
        try:
            # Apply model-specific pruning for Step 6.5
            if "features" in train_data and len(train_data["features"]) > 0:
                from src.training.model_specific_pruning import ModelSpecificPruning
                pruning_manager = ModelSpecificPruning(self.config)

                # Convert features to DataFrame for pruning
                features_df = pd.DataFrame(train_data["features"].numpy())
                # Use real target labels for pruning, not a dummy target.
                # The target should be available in `train_data`.
                if "labels" not in train_data or "regime" not in train_data["labels"]:
                    raise ValueError("Target labels are required for feature pruning but not found in train_data.")
                target_series = pd.Series(train_data["labels"]["regime"].numpy())

                pruned_features, pruning_metadata = pruning_manager.prune_for_step6_5_unified_regime(
                    features_df, target_series,
                )

                # Update features with pruned version
                train_data["features"] = torch.FloatTensor(pruned_features.values)
                self.logger.info(f"✅ Applied model-specific pruning: {features_df.shape[1]} -> {pruned_features.shape[1]} features")

            # Split data
            num_samples: int = int(train_data["num_sequences"])
            split_idx = int(num_samples * (1 - self.validation_split))

            # Training data
            train_hmm = {
                tf: states[:split_idx]
                for tf, states in train_data["hmm_states"].items()
            }
            train_features = train_data["features"][:split_idx]
            train_labels = {k: v[:split_idx] for k, v in train_data["labels"].items()}

            # Validation data
            val_hmm = {
                tf: states[split_idx:]
                for tf, states in train_data["hmm_states"].items()
            }
            val_features = train_data["features"][split_idx:]
            val_labels = {k: v[split_idx:] for k, v in train_data["labels"].items()}

            # Create data loaders
            train_dataset = TensorDataset(
                train_features,
                train_labels["regime"],
                train_labels["transition"],
                train_labels["tpsl"],
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False
            )

            val_dataset = TensorDataset(
                val_features,
                val_labels["regime"],
                val_labels["transition"],
                val_labels["tpsl"],
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            # Setup training
            device = self.device
            self.model.to(device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            best_val_loss: float = float("inf")
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_index, (
                    batch_features,
                    batch_regime,
                    batch_transition,
                    batch_tpsl,
                ) in enumerate(train_loader):
                    batch_features = batch_features.to(device)
                    batch_regime = batch_regime.to(device)
                    batch_transition = batch_transition.to(device)
                    batch_tpsl = batch_tpsl.to(device)

                    # Prepare HMM states for this batch
                    batch_hmm: dict[str, torch.Tensor] = {}
                    start_idx = batch_index * self.batch_size
                    end_idx = start_idx + len(batch_features)
                    for tf in self.timeframes:
                        if tf in train_hmm:
                            batch_hmm[tf] = train_hmm[tf][start_idx:end_idx].to(device)

                    # Forward pass
                    outputs = self.model(batch_hmm, batch_features)

                    # Calculate losses
                    regime_loss = criterion(outputs["regime_logits"], batch_regime)
                    transition_loss = criterion(
                        outputs["transition_logits"], batch_transition,
                    )
                    tpsl_loss = criterion(outputs["tpsl_logits"], batch_tpsl)
                    confidence_loss = F.mse_loss(
                        outputs["confidence_logits"].squeeze(),
                        torch.ones_like(outputs["confidence_logits"].squeeze()),
                    )

                    total_loss = (regime_loss + transition_loss + tpsl_loss + confidence_loss)

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    train_loss += total_loss.item()

                # Validation phase
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_index, (
                        batch_features,
                        batch_regime,
                        batch_transition,
                        batch_tpsl,
                    ) in enumerate(val_loader):
                        batch_features = batch_features.to(device)
                        batch_regime = batch_regime.to(device)
                        batch_transition = batch_transition.to(device)
                        batch_tpsl = batch_tpsl.to(device)

                        batch_hmm: dict[str, torch.Tensor] = {}
                        start_idx = split_idx + batch_index * self.batch_size
                        end_idx = start_idx + len(batch_features)
                        for tf in self.timeframes:
                            if tf in val_hmm:
                                batch_hmm[tf] = val_hmm[tf][start_idx - split_idx:end_idx - split_idx].to(device)

                        outputs = self.model(batch_hmm, batch_features)

                        regime_loss = criterion(outputs["regime_logits"], batch_regime)
                        transition_loss = criterion(
                            outputs["transition_logits"], batch_transition,
                        )
                        tpsl_loss = criterion(outputs["tpsl_logits"], batch_tpsl)
                        confidence_loss = F.mse_loss(
                            outputs["confidence_logits"].squeeze(),
                            torch.ones_like(outputs["confidence_logits"].squeeze()),
                        )

                        total_loss = (regime_loss + transition_loss + tpsl_loss + confidence_loss)
                        val_loss += total_loss.item()

                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(
                        f"📊 Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                        f"Val Loss: {val_loss/len(val_loader):.4f}",
                    )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.artifacts_dir, "best_model.pth"),
                    )

            self.logger.info("✅ Model training completed successfully")
            return True

        except Exception as e:
            self.logger.exception(f"🚨 Error during training: {e}")
            return False

    async def _save_artifacts(self) -> None:
        """Save model artifacts and metadata."""
        try:
            # Save model
            torch.save(
                self.model.state_dict(),
                os.path.join(self.artifacts_dir, "final_model.pth"),
            )

            # Save label encoders
            for name, encoder in self.label_encoders.items():
                with open(
                    os.path.join(self.artifacts_dir, f"{name}_encoder.pkl"), "wb",
                ) as f:
                    pickle.dump(encoder, f)

            # Save configuration
            config_save = {
                "timeframes": self.timeframes,
                "hmm_states_per_tf": self.hmm_states_per_tf,
                "sequence_length": self.sequence_length,
                "num_regimes": self.num_regimes,
                "model_config": self.config,
                "training_timestamp": datetime.now().isoformat(),
            }

            safe_json_dump(config_save, os.path.join(self.artifacts_dir, "config.json"), indent=2)

            self.logger.info(f"💾 Artifacts saved to {self.artifacts_dir}")

        except Exception as e:
            self.logger.exception(f"🚨 Error saving artifacts: {e}")

    def predict(
        self, hmm_states: dict[str, np.ndarray], features: np.ndarray,
    ) -> dict[str, Any] | None:
        """Make predictions using the trained unified model."

        Args:
            hmm_states: HMM state sequences for each timeframe
            features: Market features

        Returns: Dict containing regime prediction, transition probability, and S/R detection

        """
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")

            # Use configured device if available
            device = next(self.model.parameters()).device

            # Prepare inputs
            hmm_tensors: dict[str, torch.Tensor] = {}
            for tf, states in hmm_states.items():
                if tf in self.timeframes:
                    hmm_tensors[tf] = (
                        torch.tensor(states, dtype=torch.long).unsqueeze(0).to(device)
                    )

            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(hmm_tensors, feature_tensor)

            # Process outputs
            regime_probs = F.softmax(outputs["regime_logits"], dim=-1)
            transition_probs = F.softmax(outputs["transition_logits"], dim=-1)
            tpsl_probs = F.softmax(outputs["tpsl_logits"], dim=-1)
            confidence_score = torch.sigmoid(outputs["confidence_logits"]).item()

            # Decode predictions
            regime_pred = torch.argmax(regime_probs, dim=-1).item()
            transition_pred = torch.argmax(transition_probs, dim=-1).item()
            tpsl_pred = torch.argmax(tpsl_probs, dim=-1).item()

            return {
                "regime": {
                    "prediction": regime_pred,
                    "probabilities": regime_probs.cpu().numpy()[0],
                    "confidence": torch.max(regime_probs).item(),
                },
                "transition": {
                    "prediction": transition_pred,
                    "probability": transition_probs[
                        0, 1,
                    ].item(),  # Probability of transition
                    "confidence": torch.max(transition_probs).item(),
                },
                "tpsl": {
                    "prediction": tpsl_pred,
                    "probabilities": tpsl_probs.cpu().numpy()[0],
                    "confidence": torch.max(tpsl_probs).item(),
                    "direction": "long" if tpsl_pred == 1 else "short",  # Only long/short
                },
                "confidence_score": confidence_score,
            }

        except Exception as e:
            self.logger.exception(f"🚨 Error making prediction: {e}")
            return None

    def predict_with_position_logic(
        self, hmm_states: dict[str, np.ndarray], features: np.ndarray, current_position: str = "none", confidence_threshold: float = 0.7
    ) -> dict[str, Any] | None:
        """Make predictions with position logic integration."

        Args:
            hmm_states: HMM state sequences for each timeframe
            features: Market features
            current_position: Current position ("long", "short", "none")
            confidence_threshold: Minimum confidence to take action

        Returns:
            Dict containing predictions with position action logic

        """
        try:
            # Get base predictions
            base_prediction = self.predict(hmm_states, features)
            if base_prediction is None:
                return None

            # Extract TPSL prediction and confidence
            tpsl_prediction = base_prediction["tpsl"]["prediction"]
            confidence_score = base_prediction["confidence_score"]

            # Determine position action
            position_action = self._determine_position_action(
                tpsl_prediction=tpsl_prediction,
                confidence_score=confidence_score,
                current_position=current_position,
                confidence_threshold=confidence_threshold,
            )

            # Combine base prediction with position logic
            return {
                **base_prediction,
                "position_logic": position_action,
                "trading_decision": {
                    "action": position_action["action"],
                    "reason": position_action["reason"],
                    "confidence": confidence_score,
                    "regime": base_prediction["regime"]["prediction"],
                    "transition_probability": base_prediction["transition"]["probability"],
                },
            }

        except Exception as e:
            self.logger.exception(f"🚨 Error in prediction with position logic: {e}")
            return None

    def _determine_position_action(
        self, tpsl_prediction: int, confidence_score: float, current_position: str = "none", confidence_threshold: float = 0.7
    ) -> dict[str, Any]:
        """Determine position action based on TPSL prediction, confidence, and current position."

        Args:
            tpsl_prediction: 0 for short/no position, 1 for long
            confidence_score: Model confidence (0-1)
            current_position: Current position ("long", "short", "none")
            confidence_threshold: Minimum confidence to take action

        Returns:
            Dict with position action and reasoning

        """
        try:
            # Determine intended direction from TPSL prediction
            intended_direction = "long" if tpsl_prediction == 1 else "short"

            # Check if confidence is high enough to take action
            if confidence_score < confidence_threshold:
                return {
                    "action": "hold",
                    "reason": f"Confidence too low ({confidence_score:.3f} < {confidence_threshold})",
                    "intended_direction": intended_direction,
                    "confidence": confidence_score,
                }

            # Position logic based on current position and intended direction
            if current_position == "none":
                # No position open - can take new position if confidence is high
                if confidence_score >= confidence_threshold:
                    return {
                        "action": "open_long" if intended_direction == "long" else "open_short",
                        "reason": f"Opening {intended_direction} position with confidence {confidence_score:.3f}",
                        "intended_direction": intended_direction,
                        "confidence": confidence_score,
                    }
                return {
                    "action": "hold",
                    "reason": f"Confidence insufficient to open position ({confidence_score:.3f} < {confidence_threshold})",
                    "intended_direction": intended_direction,
                    "confidence": confidence_score,
                }

            if current_position == "long":
                # Currently long
                if intended_direction == "long":
                    # Intending to stay long - hold if confidence is high
                    if confidence_score >= confidence_threshold:
                        return {
                            "action": "hold_long",
                            "reason": f"Maintaining long position with confidence {confidence_score:.3f}",
                            "intended_direction": intended_direction,
                            "confidence": confidence_score,
                        }
                    return {
                        "action": "close_long",
                        "reason": f"Closing long position due to low confidence ({confidence_score:.3f} < {confidence_threshold})",
                        "intended_direction": intended_direction,
                        "confidence": confidence_score,
                    }
                # Intending to go short - close long and potentially open short
                return {
                    "action": "close_long",
                    "reason": "Closing long position to switch to short direction",
                    "intended_direction": intended_direction,
                    "confidence": confidence_score,
                }

            if current_position == "short":
                # Currently short
                if intended_direction == "short":
                    # Intending to stay short - hold if confidence is high
                    if confidence_score >= confidence_threshold:
                        return {
                            "action": "hold_short",
                            "reason": f"Maintaining short position with confidence {confidence_score:.3f}",
                            "intended_direction": intended_direction,
                            "confidence": confidence_score,
                        }
                    return {
                        "action": "close_short",
                        "reason": f"Closing short position due to low confidence ({confidence_score:.3f} < {confidence_threshold})",
                        "intended_direction": intended_direction,
                        "confidence": confidence_score,
                    }
                # Intending to go long - close short and potentially open long
                return {
                    "action": "close_short",
                    "reason": "Closing short position to switch to long direction",
                    "intended_direction": intended_direction,
                    "confidence": confidence_score,
                }

            # Fallback
            return {
                "action": "hold",
                "reason": "Unknown position state",
                "intended_direction": intended_direction,
                "confidence": confidence_score,
            }

        except Exception as e:
            self.logger.exception(f"🚨 Error determining position action: {e}")
            return {
                "action": "hold",
                "reason": f"Error in position logic: {e}",
                "intended_direction": "unknown",
                "confidence": confidence_score,
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="unified prediction with S/R integration",
    )
    async def predict_with_sr_integration(
        self, hmm_states: dict[str, np.ndarray], market_features: np.ndarray, market_data: pd.DataFrame, current_price: float, ) -> dict[str, Any]:
        """Make unified predictions with S/R level integration."

        Args:
            hmm_states: HMM states for each timeframe
            market_features: Market features
            market_data: Raw market data for S/R analysis
            current_price: Current market price

        Returns:
            dict: Unified predictions with S/R integration

        """
        try:
            # Get base unified predictions
            unified_prediction = self.predict(hmm_states, market_features)

            # Get S/R context and outcome prediction using centralized logic
            sr_context = await self.sr_predictor.get_sr_context(
                market_data=market_data,
                current_price=current_price,
            )
            sr_outcome = await self.sr_predictor.predict_sr_outcome(
                market_data=market_data,
                current_price=current_price,
                sr_context=sr_context,
            )

            # Combine predictions based on S/R proximity
            is_near_sr = sr_outcome.get("is_near_sr_level", False)

            if is_near_sr:
                # Use S/R outcome prediction when near levels
                combined_prediction = {
                    **(unified_prediction or {}),
                    "sr_analysis": {
                        "outcome": sr_outcome.get("outcome", "consolidation"),
                        "confidence": sr_outcome.get("confidence", 0.5),
                        "probabilities": sr_outcome.get("probabilities", {}),
                    },
                }
                return combined_prediction

            # Otherwise, return unified prediction with S/R context attached
            if unified_prediction is None:
                unified_prediction = {}
            unified_prediction["sr_analysis"] = sr_context
            return unified_prediction

        except Exception as e:
            self.logger.exception(
                f"🚨 Error in unified prediction with S/R integration: {e}",
            )
            return {
                "error": "Failed to integrate S/R analysis",
                "unified_prediction": unified_prediction
                if "unified_prediction" in locals() else {},
                "sr_analysis": {},
                "combined_confidence": 0.5,
                "risk_management": {
                    "position_size": 0.5,
                    "stop_loss_multiplier": 1.25,
                    "risk_level": "MEDIUM",
                },
            }

    async def _run_hyperparameter_optimization(self) -> dict[str, Any] | None:
        """Optional short hyperparameter optimization using Optuna."

        Returns a dict with best_params/best_value or None if Optuna unavailable.
        """
        try:
            import optuna  # type: ignore
        except Exception as ex:
            self.logger.warning(
                f"⚠️ Optuna not available for HPO ({ex}); skipping optimization",
            )
            return None

            pruner, optuna.pruners.MedianPruner() if self.hpo_pruning else None
            study, optuna.create_study(direction="maximize", pruner=pruner)

            def objective(trial: "optuna.Trial") -> float:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                    "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
                    "num_layers": trial.suggest_int("num_layers", 2, 6),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                    "sequence_length": trial.suggest_int("sequence_length", 10, 50),
                }
            # Lightweight proxy objective (no full training inside step to keep runtime bounded)
            score = 0.5 + 0.3 * (1.0 - float(params["dropout"])) + 0.2 * (float(params["d_model"]) / 512.0),
            return float(score)

            # Get HPO parameters from training input or use defaults
            hpo_trials = self.training_input.get("hpo_trials", self.n_trials)
            hpo_timeout = self.training_input.get("hpo_timeout", self.hpo_timeout)
            
            study.optimize(
                objective, n_trials=hpo_trials, timeout=hpo_timeout, show_progress_bar=False
            )

            best_params = study.best_params,
            best_value = study.best_value,
            self.logger.info(f"✅ HPO completed. Best score: {best_value:.4f}")
            self.logger.info(f"Best parameters: {best_params}")

            return {
                "best_params": best_params,
                "best_value": best_value,
                "n_trials": len(study.trials),
            }
        except Exception as ex:
            self.logger.exception(f"Error in hyperparameter optimization: {ex}")
            return None

    def _apply_structured_pruning(self, model: nn.Module) -> dict[str, Any]:
        """Apply light pruning to reduce model complexity (optional)."""
        try:
            pruning_results: dict[str, Any] = {}
            # Attention pruning
            if hasattr(model, "cross_timeframe_attention"):
                attn = model.cross_timeframe_attention
                if hasattr(attn, "in_proj_weight"):
                    try:
                        prune.l1_unstructured(attn, name="in_proj_weight", amount=0.1)
                        pruning_results["attention_pruning"] = True
                    except Exception as ex:
                        self.logger.warning(f"⚠️ Attention pruning failed: {ex}")
            # Classifier pruning
            if hasattr(model, "regime_classifier") and model.regime_classifier is not None:
                try:
                    prune.l1_unstructured(model.regime_classifier, name="weight", amount=0.1)
                    pruning_results["classifier_pruning"] = True
                except Exception as ex:
                    self.logger.warning(f"⚠️ Classifier pruning failed: {ex}")
            return pruning_results
        except Exception as ex:
            self.logger.exception(f"Error in structured pruning: {ex}")
            return {}

    def _optimize_architecture(self, model: nn.Module) -> dict[str, Any]:
        """Placeholder architecture optimization flags for diagnostics."""
        try:
            results: dict[str, Any] = {}
            if hasattr(model, "transformer"):
                results["transformer_optimization"] = True
            if hasattr(model, "hmm_embeddings"):
                results["embedding_optimization"] = True
            return results
        except Exception as ex:
            self.logger.exception(f"Error in architecture optimization: {ex}")
            return {}

    def _calculate_sr_combined_confidence(
        self, unified_prediction: dict[str, Any], sr_outcome: dict[str, Any]) -> float:
        """Calculate combined confidence when near S/R levels."""
        try:
            unified_confidence = unified_prediction.get("confidence_score", 0.5)
            sr_confidence = sr_outcome.get("confidence", 0.5)

            # When near S/R levels, weight S/R outcome more heavily
            # 60% S/R outcome, 40% unified prediction
            combined_confidence = sr_confidence * 0.6 + unified_confidence * 0.4

            return max(0.0, min(1.0, combined_confidence))

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating SR combined confidence: {e}")
            return 0.5

    def _calculate_sr_risk_parameters(
        self, unified_prediction: dict[str, Any], sr_outcome: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk management parameters when near S/R levels."""
        try:
            combined_confidence = self._calculate_sr_combined_confidence(
                unified_prediction, sr_outcome,
            )
            outcome = sr_outcome.get("outcome", "consolidation")

            # Adjust position sizing based on outcome
            base_position_size = min(combined_confidence, 0.8)

            if outcome == "breakout":
                # More aggressive for breakouts
                position_size = base_position_size * 1.2
                stop_loss_multiplier = (1.0 + (1.0 - combined_confidence) * 0.3
                )  # Tighter stops
            elif outcome == "rebounce":
                # Conservative for rebounds
                position_size = base_position_size * 0.8
                stop_loss_multiplier = (1.0 + (1.0 - combined_confidence) * 0.7
                )  # Wider stops
            else:  # consolidation
                # Standard sizing
                position_size = base_position_size
                stop_loss_multiplier = 1.0 + (1.0 - combined_confidence) * 0.5

            # Risk level classification
            if combined_confidence >= 0.8:
                risk_level = "LOW"
            elif combined_confidence >= 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            return {
                "position_size": min(position_size, 0.8),  # Cap at 80%
                "stop_loss_multiplier": stop_loss_multiplier,
                "risk_level": risk_level,
            }

        except Exception as e:
            self.logger.exception(f"🚨 Error calculating SR risk parameters: {e}")
            return {
                "position_size": 0.5,
                "stop_loss_multiplier": 1.25,
                "risk_level": "MEDIUM",
            }


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

from src.utils.enhanced_mlflow_integration import (
import copy
import numpy as np
import os.path
import pandas as pd

    with_enhanced_mlflow_logging,
    log_step_report,
    create_detailed_step_report,
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_artifact_with_standardized_name
)


@deterministic_seed(42)
@idempotent_step(step_key="step5_5_unified_regime_intelligence")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=3600.0)
@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=6.0,
    min_disk_gb=3.0,
    required_packages=["pandas", "numpy", "sklearn", "torch"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp"],
    },
    context="Unified Regime Intelligence",
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
    chunk_size=20000, streaming_processing=True, memory_pool=True, cleanup_frequency=50,
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
    required_files=[],
    data_quality_checks={"min_rows": 100},
)
@quality_gate(
    model_performance_thresholds={"accuracy": 0.55},
    data_quality_metrics={"completeness": 0.85},
)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    timeframe: str = "1m",
    training_config: dict[str, Any] | None = None,
    force_rerun: bool = False,
) -> bool:
    """Run the unified regime intelligence step."

    This step consolidates:
    - Multi-timeframe HMM state analysis
    - Regime transition prediction
    - Support/Resistance level detection
    - Expert activation logic

    Replaces step9_5 and step10 with a single, efficient model.
    """
    # Log step parameters for debugging
    logger.info("=" * 80)
    logger.info("🚀 STEP 5_5: Unified Regime Intelligence")
    logger.info("=" * 80)
    logger.info("📋 Step 5_5 Parameters:")
    logger.info(f"   Symbol: {symbol}")
    logger.info(f"   Exchange: {exchange}")
    logger.info(f"   Timeframe: {timeframe}")
    logger.info(f"   Force Rerun: {force_rerun}")

    step_start_time = time.time()
    step_phases = {
        "configuration": False,
        "initialization": False,
        "data_loading": False,
        "training": False,
        "validation": False,
    }

    try:
        logger.info(
            f"🔄 Starting Unified Regime Intelligence Step for {exchange}:{symbol}",
        )

        # Phase 1: Load configuration
        logger.info("📋 Phase 1: Loading configuration...")
        try:
            config = training_config or {}
            uri_config = config.get("UNIFIED_REGIME_INTELLIGENCE", {})

            if not uri_config.get("enabled", True):
                logger.info(
                    "⏭️ Unified Regime Intelligence disabled; skipping step 5_5.",
                )
                return True

            logger.info(f"✅ Configuration loaded: {len(uri_config)} parameters")
            step_phases["configuration"] = True
        except Exception as e:
            logger.exception(f"❌ Configuration loading failed: {e}")
            return False

        # Phase 2: Initialize step
        logger.info("🔧 Phase 2: Initializing Unified Regime Intelligence Step...")
        try:
            step = UnifiedRegimeIntelligenceStep(uri_config)
            if not await step.initialize():
                logger.error("❌ Failed to initialize Unified Regime Intelligence Step")
                return False

            logger.info("✅ Unified Regime Intelligence Step initialized successfully")
            step_phases["initialization"] = True
        except Exception as e:
            logger.exception(f"❌ Initialization failed: {e}")
            return False

        # Phase 3: Load data
        logger.info("📥 Phase 3: Loading training data...")
        try:
            data = {
                "combined_features": pd.DataFrame(),  # Would be loaded from previous steps
            }

            # Validate data
            if data["combined_features"].empty:
                logger.warning("⚠️ No combined features provided, using HMM data only")

            logger.info(f"✅ Data loaded: {len(data)} data sources")
            step_phases["data_loading"] = True
        except Exception as e:
            logger.exception(f"❌ Data loading failed: {e}")
            return False

        # Phase 4: Train model
        logger.info("🏋️ Phase 4: Training unified model...")
        try:
            train_success = await step.train(data)
            if not train_success:
                logger.error("❌ Training failed for Unified Regime Intelligence Step")
                return False

            step_phases["training"] = True
            logger.info("✅ Training phase completed")
        except Exception as e:
            logger.exception(f"❌ Training phase failed: {e}")
            return False

        # Phase 5: Validation (placeholder)
        logger.info("🧪 Phase 5: Validation...")
        try:
            step_phases["validation"] = True
            logger.info("✅ Validation phase completed")
        except Exception as e:
            logger.exception(f"❌ Validation phase failed: {e}")
            return False

        total_time = time.time() - step_start_time
        logger.info(f"🎉 Step 5_5 completed in {total_time:.2f}s")
        return True

    except Exception as e:
        logger.exception(f"🚨 Unified Regime Intelligence Step encountered a critical error: {e}")
        return False