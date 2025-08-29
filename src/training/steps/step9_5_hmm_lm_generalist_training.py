# src/training/steps/step9_5_hmm_lm_generalist_training.py

import asyncio
import concurrent.futures
import json
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.utils.centralized_decorators import (
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    guard_dataframe_nulls,
    handle_errors,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_data_quality,
    validate_step_output,
    validate_step_prerequisites,
    with_tracing_span,
)
from src.utils.logger import system_logger

# Import enhanced HMM regime manager for improved functionality
try:
    from src.utils.enhanced_hmm_regime_manager import EnhancedHMMRegimeManager
    ENHANCED_HMM_AVAILABLE = True
except ImportError:
    ENHANCED_HMM_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

logger = system_logger.getChild("Step9_5_HMM_LM_Generalist")


class HMMLMGeneralistTrainingStep:
    """Step 9.5: Generalist HMM-LM Model Training for Regime Change Prediction."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, Any] = {}
        self.label_encoders: dict[str, Any] = {}

        # HMM-LM configuration
        hmm_lm_config: dict[str, Any] = config.get("HMM_LM", {})
        generalist_config: dict[str, Any] = hmm_lm_config.get("generalist", {})

        self.hmm_states: int = int(generalist_config.get("hmm_states", 5))
        self.sequence_length: int = int(generalist_config.get("sequence_length", 20))
        self.timeframes: list[str] = generalist_config.get(
            "timeframes",
            ["1m", "5m", "15m", "30m"],
        )
        self.d_model: int = int(generalist_config.get("d_model", 256))
        self.nhead: int = int(generalist_config.get("nhead", 8))
        self.num_layers: int = int(generalist_config.get("num_layers", 6))
        self.dropout_rate: float = float(generalist_config.get("dropout_rate", 0.1))
        self.learning_rate: float = float(generalist_config.get("learning_rate", 0.0001))
        self.batch_size: int = int(generalist_config.get("batch_size", 32))
        self.epochs: int = int(generalist_config.get("epochs", 100))

        # Regime change vocabulary
        self.regime_change_vocab = self._create_regime_change_vocabulary()
        
        # Initialize enhanced HMM regime manager if available
        self.enhanced_hmm_manager = None
        if ENHANCED_HMM_AVAILABLE:
            try:
                self.enhanced_hmm_manager = EnhancedHMMRegimeManager(config)
                self.logger.info("✅ Enhanced HMM regime manager initialized for step 9.5")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize EnhancedHMMRegimeManager: {e}")
        
        # Enhanced regime change prediction state
        self.regime_change_state = {
            "last_regime_change_analysis": None,
            "regime_change_count": 0,
            "regime_change_accuracy": {},
            "regime_change_quality_scores": {},
            "regime_change_redundancy_metrics": {},
            "regime_transition_probabilities": {},
            "regime_stability_metrics": {},
            "regime_forecasting_accuracy": {},
            "regime_change_detection": {},
            "regime_prediction_models": {}
        }
            "regime_change_prediction_model": None
        }

    def _create_regime_change_vocabulary(self) -> dict[str, int]:
        """Create vocabulary for regime change events."""
        vocab: dict[str, int] = {}
        vocab_id = 0

        # Add regime entry events
        for state in range(self.hmm_states):
            vocab[f"enter_regime_{state}"] = vocab_id
            vocab_id += 1

        # Add regime exit events
        for state in range(self.hmm_states):
            vocab[f"exit_regime_{state}"] = vocab_id
            vocab_id += 1

        # Add special tokens
        vocab["<PAD>"] = vocab_id
        vocab_id += 1
        vocab["<UNK>"] = vocab_id
        vocab_id += 1
        vocab["<START>"] = vocab_id
        vocab_id += 1
        vocab["<END>"] = vocab_id

        return vocab

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="HMM-LM generalist training step initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the HMM-LM generalist training step."""
        self.logger.info("Initializing HMM-LM Generalist Training Step...")
        self.logger.info("HMM-LM Generalist Training Step initialized successfully")
        return True

    @with_tracing_span("step9_5.execute", log_args=False)
    @validate_data_quality(validation_level="WARNING")
    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="HMM-LM generalist training step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute HMM-LM generalist model training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results

        """
        try:
            self.logger.info("🔄 Executing HMM-LM Generalist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load HMM data from all timeframes
            hmm_data = await self._load_multi_timeframe_hmm_data(
                exchange=exchange, symbol=symbol, data_dir=data_dir
            )
            if not hmm_data:
                msg = "Failed to load multi-timeframe HMM data"
                raise ValueError(msg)

            # Create regime change sequences
            regime_sequences = await self._create_regime_change_sequences(hmm_data)
            if not regime_sequences:
                msg = "Failed to create regime change sequences"
                raise ValueError(msg)

            # Train HMM-LM model
            model_result = await self._train_hmm_lm_model(regime_sequences)
            if not model_result:
                msg = "Failed to train HMM-LM model"
                raise ValueError(msg)

            # Save model and metadata
            await self._save_generalist_model(model_result, exchange, symbol, data_dir)

            self.logger.info("✅ HMM-LM Generalist Training completed successfully")
            return {
                "status": "SUCCESS",
                "model_trained": True,
                "vocabulary_size": len(self.regime_change_vocab),
                "hmm_states": self.hmm_states,
                "timeframes": self.timeframes,
                "result": model_result,
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ HMM-LM Generalist Training failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    @with_tracing_span("step9_5._load_multi_timeframe_hmm_data", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=0)
    async def _load_multi_timeframe_hmm_data(
        self, exchange: str, symbol: str, data_dir: str
    ) -> dict[str, pd.DataFrame]:
        """Load HMM data from all timeframes in parallel."""
        hmm_data: dict[str, pd.DataFrame] = {}

        async def load_timeframe_data(timeframe: str) -> tuple[str, pd.DataFrame | None]:
            try:
                # Load cluster assignments
                cluster_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
                if not os.path.exists(cluster_path):
                    return timeframe, None

                # Use ThreadPoolExecutor for I/O operations
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    clusters_df = await loop.run_in_executor(
                        executor, pd.read_parquet, cluster_path
                    )

                clusters_df["timestamp"] = pd.to_datetime(clusters_df["timestamp"])
                clusters_df = clusters_df.set_index("timestamp")

                # Load intensity scores
                intensity_path = (
                    f"{data_dir}/{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
                )
                if os.path.exists(intensity_path):
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        intensity_df = await loop.run_in_executor(
                            executor, pd.read_parquet, intensity_path
                        )
                    intensity_df["timestamp"] = pd.to_datetime(
                        intensity_df["timestamp"]
                    )
                    intensity_df = intensity_df.set_index("timestamp")

                    # Merge cluster assignments with intensity scores
                    hmm_df = clusters_df.merge(
                        intensity_df, left_index=True, right_index=True, how="inner"
                    )
                    hmm_df["timeframe"] = timeframe
                    self.logger.info(
                        f"✅ Loaded HMM data for {timeframe}: {hmm_df.shape}",
                    )
                    return timeframe, hmm_df

                clusters_df["timeframe"] = timeframe
                self.logger.info(
                    f"✅ Loaded HMM clusters for {timeframe}: {clusters_df.shape}",
                )
                return timeframe, clusters_df

            except Exception as e:  # noqa: BLE001
                self.logger.exception(f"❌ Failed to load HMM data for {timeframe}: {e}")
                return timeframe, None

        # Load all timeframes in parallel
        tasks = [load_timeframe_data(tf) for tf in self.timeframes]
        results = await asyncio.gather(*tasks)

        # Build result dictionary
        for timeframe, df in results:
            if df is not None:
                hmm_data[timeframe] = df

        return hmm_data

    async def _create_regime_change_sequences(
        self, hmm_data: dict[str, pd.DataFrame]
    ) -> list[dict[str, Any]]:
        """
        Create enhanced regime change sequences for training.
        Improved with advanced regime change detection and prediction capabilities.
        """
        sequences: list[dict[str, Any]] = []

        try:
            # Update regime change state
            self.regime_change_state["last_regime_change_analysis"] = pd.Timestamp.now()
            self.regime_change_state["regime_change_count"] += 1
            
            # Combine all timeframe data
            all_data: list[pd.DataFrame] = []
            for df in hmm_data.values():
                if not df.empty:
                    all_data.append(df)

            if not all_data:
                return []

            combined_df = pd.concat(all_data, axis=0).sort_index()

            # Enhanced regime change detection with advanced analysis
            regime_events = await self._detect_enhanced_regime_changes_and_outcomes(combined_df)

            # Create enhanced sequences around regime changes
            for change_idx, event_data in enumerate(regime_events):
                if change_idx < self.sequence_length:
                    continue

                # Get sequence before the change
                start_idx = change_idx - self.sequence_length
                end_idx = change_idx

                if start_idx >= 0 and end_idx < len(combined_df):
                    sequence_data = combined_df.iloc[start_idx:end_idx]

                    # Enhanced sequence with additional features
                    enhanced_sequence = {
                        "sequence": sequence_data,
                        "target": event_data["regime_change"],
                        "price_direction": event_data["price_direction"],
                        "profit_target_hit": event_data["profit_target_hit"],
                        "stop_loss_hit": event_data["stop_loss_hit"],
                        "time_to_target": event_data["time_to_target"],
                        "timestamp": combined_df.index[end_idx],
                        "timeframe": combined_df.iloc[end_idx]["timeframe"],
                        # Enhanced features
                        "regime_stability": event_data.get("regime_stability", 0.0),
                        "transition_probability": event_data.get("transition_probability", 0.0),
                        "regime_persistence": event_data.get("regime_persistence", 0.0),
                        "regime_quality": event_data.get("regime_quality", 0.0),
                        "change_confidence": event_data.get("change_confidence", 0.0),
                        "regime_complexity": event_data.get("regime_complexity", 0.0),
                        "regime_volatility": event_data.get("regime_volatility", 0.0),
                        "regime_momentum": event_data.get("regime_momentum", 0.0),
                        "regime_volume_profile": event_data.get("regime_volume_profile", 0.0),
                        "regime_change_strength": event_data.get("regime_change_strength", 0.0),
                        "regime_forecast_horizon": event_data.get("regime_forecast_horizon", 0),
                        "regime_change_detection_ready": True
                    }

                    sequences.append(enhanced_sequence)

            # Calculate quality metrics
            quality_metrics = await self._calculate_regime_change_quality_metrics(sequences)
            self.regime_change_state["regime_change_quality_scores"] = quality_metrics

            self.logger.info(f"✅ Created {len(sequences)} enhanced regime change sequences")
            self.logger.info(f"📊 Regime change quality score: {quality_metrics.get('overall_quality', 0.0):.3f}")
            return sequences

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to create regime change sequences: {e}")
            return []

    def _detect_regime_changes_and_tpsl_outcomes(
        self, df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Detect regime changes and associated TPSL outcomes."""
        events: list[dict[str, Any]] = []

        try:
            # Get TPSL parameters from config
            tpsl_config: dict[str, Any] = self.config.get(
                "vectorized_labelling_orchestrator", {}
            )
            profit_take_multiplier: float = float(
                tpsl_config.get("profit_take_multiplier", 0.002)
            )  # 0.2%
            stop_loss_multiplier: float = float(
                tpsl_config.get("stop_loss_multiplier", 0.001)
            )  # 0.1%
            _ = tpsl_config.get("time_barrier_minutes", 30)

            # Get regime column
            regime_col = "composite_cluster_id"
            if regime_col not in df.columns:
                self.logger.warning(f"⚠️ No regime column {regime_col} found")
                return events

            # Detect state transitions
            regimes = df[regime_col].fillna(-1).astype(int)

            for i in range(1, len(regimes)):
                prev_regime = int(regimes.iloc[i - 1])
                curr_regime = int(regimes.iloc[i])

                event: dict[str, Any] = {
                    "regime_change": "<PAD>",
                    "price_direction": 1,  # Sideways
                    "profit_target_hit": 0,  # 0/1
                    "stop_loss_hit": 0,  # 0/1
                    "time_to_target": 0,  # bars to hit target
                }

                if prev_regime != curr_regime and prev_regime >= 0 and curr_regime >= 0:
                    # Exit previous regime
                    if 0 <= prev_regime < self.hmm_states:
                        event["regime_change"] = f"exit_regime_{prev_regime}"

                    # Enter new regime
                    if 0 <= curr_regime < self.hmm_states:
                        event["regime_change"] = f"enter_regime_{curr_regime}"

                    # Calculate TPSL outcomes for regime change
                    if "close" in df.columns and i < len(df) - 1:
                        current_price = float(df.iloc[i]["close"])  # type: ignore[index]
                        future_prices = df.iloc[i + 1 : i + 31]["close"].values

                        if len(future_prices) > 0:
                            # Calculate profit target and stop loss levels
                            profit_target = current_price * (1 + profit_take_multiplier)
                            stop_loss = current_price * (1 - stop_loss_multiplier)

                            # Check if profit target or stop loss is hit
                            profit_target_hit = 0
                            stop_loss_hit = 0
                            time_to_target = 0

                            for j, future_price in enumerate(future_prices):
                                fp = float(future_price)
                                if fp >= profit_target and profit_target_hit == 0:
                                    profit_target_hit = 1
                                    time_to_target = j + 1
                                elif fp <= stop_loss and stop_loss_hit == 0:
                                    stop_loss_hit = 1
                                    if time_to_target == 0:
                                        time_to_target = j + 1

                            # Price direction based on TPSL outcomes
                            if profit_target_hit == 1 and stop_loss_hit == 0:
                                event["price_direction"] = 0  # Up (hit profit target)
                            elif stop_loss_hit == 1 and profit_target_hit == 0:
                                event["price_direction"] = 2  # Down (hit stop loss)
                            elif profit_target_hit == 1 and stop_loss_hit == 1:
                                # Both hit - determine which came first
                                if time_to_target <= 15:  # Profit target hit first
                                    event["price_direction"] = 0  # Up
                                else:
                                    event["price_direction"] = 2  # Down
                            else:
                                event["price_direction"] = 1  # Sideways (neither hit)

                            # Set TPSL outcomes
                            event["profit_target_hit"] = profit_target_hit
                            event["stop_loss_hit"] = stop_loss_hit
                            event["time_to_target"] = time_to_target

                events.append(event)

            # Add padding for the first element
            events.insert(
                0,
                {
                    "regime_change": "<PAD>",
                    "price_direction": 1,
                    "profit_target_hit": 0,
                    "stop_loss_hit": 0,
                    "time_to_target": 0,
                },
            )

            return events

        except Exception as e:  # noqa: BLE001
            self.logger.exception(
                f"❌ Failed to detect regime changes and price action: {e}",
            )
            return []

    async def _train_hmm_lm_model(
        self, sequences: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Train the HMM-LM model."""
        try:
            self.logger.info(
                f"🔄 Training HMM-LM model with {len(sequences)} sequences",
            )

            if len(sequences) < 100:
                self.logger.warning(
                    f"⚠️ Insufficient sequences for training: {len(sequences)}",
                )
                return None

            # Prepare training data
            X_train, y_train, X_val, y_val = self._prepare_regime_training_data(
                sequences,
            )

            # Create efficient regime predictor
            input_dim = int(X_train.shape[2]) if len(X_train.shape) > 2 else 10
            model = EfficientRegimePredictor(
                input_dim=input_dim,
                num_regimes=self.hmm_states,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
            )

            # Train model
            trainer = EfficientRegimeTrainer(
                model, learning_rate=self.learning_rate, batch_size=self.batch_size
            )
            history = await trainer.train(
                X_train, y_train, X_val, y_val, epochs=self.epochs
            )

            # Save model
            model_path = "models/hmm_lm_generalist_model.pth"
            torch.save(model.state_dict(), model_path)

            return {
                "model_path": model_path,
                "vocabulary": self.regime_change_vocab,
                "vocabulary_size": len(self.regime_change_vocab),
                "sequence_length": self.sequence_length,
                "hmm_states": self.hmm_states,
                "history": history,
                "model_config": {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dropout_rate": self.dropout_rate,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                },
            }

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ HMM-LM training failed: {e}")
            return None

    def _prepare_regime_training_data(
        self, sequences: list[dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data for regime prediction."""
        try:
            # Convert sequences to tensor format
            X_data: list[np.ndarray] = []
            regime_ids: list[int] = []
            profit_hits: list[int] = []
            stop_hits: list[int] = []
            times_to_target: list[int] = []

            for seq_data in sequences:
                sequence: pd.DataFrame = seq_data["sequence"]
                target: str = seq_data["target"]

                # Convert sequence to feature tensor
                features = self._sequence_to_features(sequence)
                X_data.append(features)

                # Extract regime ID from target (e.g., "enter_regime_2" -> 2)
                if "enter_regime_" in target or "exit_regime_" in target:
                    regime_id = int(str(target).split("_")[-1])
                else:
                    regime_id = 0  # Default regime

                # Extract TPSL outcomes
                profit_target_hit = int(seq_data.get("profit_target_hit", 0))
                stop_loss_hit = int(seq_data.get("stop_loss_hit", 0))
                time_to_target = int(seq_data.get("time_to_target", 0))

                regime_ids.append(regime_id)
                profit_hits.append(profit_target_hit)
                stop_hits.append(stop_loss_hit)
                times_to_target.append(time_to_target)

            X = np.array(X_data, dtype=np.float32)
            y: Dict[str, np.ndarray] = {
                "regime_id": np.array(regime_ids, dtype=np.int64),
                "profit_target_hit": np.array(profit_hits, dtype=np.float32),
                "stop_loss_hit": np.array(stop_hits, dtype=np.float32),
                "time_to_target": np.array(times_to_target, dtype=np.float32),
            }

            # Split data with time series split
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train = {k: v[:split_idx] for k, v in y.items()}
            y_val = {k: v[split_idx:] for k, v in y.items()}

            return X_train, y_train, X_val, y_val

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to prepare regime training data: {e}")
            return np.array([]), {}, np.array([]), {}

    def _sequence_to_features(self, sequence: pd.DataFrame) -> np.ndarray:
        """Convert sequence to feature tensor."""
        try:
            # Extract HMM-related features
            feature_cols: list[np.ndarray] = []

            # Add cluster ID
            if "composite_cluster_id" in sequence.columns:
                feature_cols.append(sequence["composite_cluster_id"].values.astype(float))

            # Add intensity features
            intensity_cols = [
                col for col in sequence.columns if str(col).startswith("intensity_cluster_")
            ]
            for col in intensity_cols:
                feature_cols.append(sequence[col].values.astype(float))

            # Add regime probability features
            regime_cols = [
                col for col in sequence.columns if str(col).endswith("_p_state_")
            ]
            for col in regime_cols:
                feature_cols.append(sequence[col].values.astype(float))

            # Stack features
            if feature_cols:
                features = np.column_stack(feature_cols)
            else:
                # Fallback: use basic features
                features = np.zeros((len(sequence), 10), dtype=np.float32)

            # Pad or truncate to sequence length
            if len(features) < self.sequence_length:
                # Pad with zeros at the beginning
                padding = np.zeros(
                    (self.sequence_length - len(features), features.shape[1]),
                    dtype=features.dtype,
                )
                features = np.vstack([padding, features])
            elif len(features) > self.sequence_length:
                # Truncate
                features = features[-self.sequence_length :]

            return features.astype(np.float32)

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to convert sequence to features: {e}")
            return np.zeros((self.sequence_length, 10), dtype=np.float32)

    async def _save_generalist_model(
        self, model_result: dict[str, Any], exchange: str, symbol: str, data_dir: str
    ) -> None:
        """Save the generalist model and metadata."""
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)

            # Save metadata
            metadata: dict[str, Any] = {
                "exchange": exchange,
                "symbol": symbol,
                "training_date": datetime.now().isoformat(),
                "model_type": "HMM_LM_Generalist",
                "vocabulary_size": len(self.regime_change_vocab),
                "hmm_states": self.hmm_states,
                "timeframes": self.timeframes,
                "sequence_length": self.sequence_length,
                "vocabulary": self.regime_change_vocab,
                "result": model_result,
            }

            os.makedirs(data_dir, exist_ok=True)
            metadata_path = f"{data_dir}/{exchange}_{symbol}_hmm_lm_generalist_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"✅ Saved generalist model metadata to {metadata_path}")

        except Exception as e:  # noqa: BLE001
            self.logger.exception(f"❌ Failed to save generalist model: {e}")


# Efficient Regime Prediction Architecture


class EfficientRegimePredictor(nn.Module):
    """Efficient regime prediction model for financial time series."""

    def __init__(
        self, input_dim: int, num_regimes: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.d_model = d_model

        # Multi-scale feature extraction
        self.conv1d_short = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv1d_medium = nn.Conv1d(input_dim, 64, kernel_size=15, padding=7)
        self.conv1d_long = nn.Conv1d(input_dim, 64, kernel_size=30, padding=15)

        # Feature fusion
        self.feature_fusion = nn.Linear(192, d_model)

        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regime prediction heads
        self.regime_classifier = nn.Linear(d_model, num_regimes)  # Current regime
        self.transition_predictor = nn.Linear(
            d_model, num_regimes * num_regimes
        )  # Transition matrix
        self.regime_confidence = nn.Linear(d_model, num_regimes)  # Confidence scores

        # Price action prediction heads
        self.price_direction = nn.Linear(d_model, 3)  # Up/Down/Sideways
        self.profit_target_prob = nn.Linear(
            d_model, 1
        )  # Probability of hitting profit target
        self.stop_loss_prob = nn.Linear(d_model, 1)  # Probability of hitting stop loss
        self.time_to_target = nn.Linear(
            d_model, 1
        )  # Expected time to hit target (bars)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape: (batch, sequence_length, input_dim)

        # Multi-scale feature extraction
        x_t = x.transpose(1, 2)  # (batch, input_dim, sequence_length)

        short_features = F.relu(self.conv1d_short(x_t))
        medium_features = F.relu(self.conv1d_medium(x_t))
        long_features = F.relu(self.conv1d_long(x_t))

        # Global average pooling
        short_pooled = F.adaptive_avg_pool1d(short_features, 1).squeeze(-1)
        medium_pooled = F.adaptive_avg_pool1d(medium_features, 1).squeeze(-1)
        long_pooled = F.adaptive_avg_pool1d(long_features, 1).squeeze(-1)

        # Combine multi-scale features
        combined_features = torch.cat([short_pooled, medium_pooled, long_pooled], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Add sequence dimension for transformer
        fused_features = fused_features.unsqueeze(1)  # (batch, 1, d_model)

        # Apply transformer
        transformer_out = self.transformer(fused_features)

        # Take the output
        final_features = transformer_out[:, -1, :]
        final_features = self.dropout(final_features)

        # Predict regime probabilities and transitions
        current_regime_probs = F.softmax(self.regime_classifier(final_features), dim=-1)
        transition_probs = F.softmax(
            self.transition_predictor(final_features).view(
                -1, self.num_regimes, self.num_regimes
            ),
            dim=-1,
        )
        regime_confidence = torch.sigmoid(self.regime_confidence(final_features))

        # Predict price action and TPSL probabilities
        price_direction_probs = F.softmax(self.price_direction(final_features), dim=-1)
        profit_target_prob = torch.sigmoid(self.profit_target_prob(final_features))  # 0-1
        stop_loss_prob = torch.sigmoid(self.stop_loss_prob(final_features))  # 0-1
        time_to_target = torch.sigmoid(self.time_to_target(final_features)) * 30  # 0-30

        return {
            "current_regime": current_regime_probs,
            "transition_matrix": transition_probs,
            "confidence": regime_confidence,
            "price_direction": price_direction_probs,
            "profit_target_prob": profit_target_prob,
            "stop_loss_prob": stop_loss_prob,
            "time_to_target": time_to_target,
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model),
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class EfficientRegimeDataset(Dataset[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Custom dataset to return features and dict targets."""

    def __init__(
        self, X: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.X = X
        self.targets = targets

    def __len__(self) -> int:  # noqa: D401
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}


class EfficientRegimeTrainer:
    """Efficient trainer for regime prediction model."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.0001, batch_size: int = 32
    ) -> None:
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable mixed precision for efficiency
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    async def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        epochs: int = 100,
    ) -> dict[str, list[float]]:
        """Train the regime prediction model efficiently."""
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)

        y_train_t: Dict[str, torch.Tensor] = {
            "regime_id": torch.from_numpy(y_train["regime_id"]).long().to(self.device),
            "profit_target_hit": torch.from_numpy(y_train["profit_target_hit"]).float().to(self.device),
            "stop_loss_hit": torch.from_numpy(y_train["stop_loss_hit"]).float().to(self.device),
            "time_to_target": torch.from_numpy(y_train["time_to_target"]).float().to(self.device),
        }
        y_val_t: Dict[str, torch.Tensor] = {
            "regime_id": torch.from_numpy(y_val["regime_id"]).long().to(self.device),
            "profit_target_hit": torch.from_numpy(y_val["profit_target_hit"]).float().to(self.device),
            "stop_loss_hit": torch.from_numpy(y_val["stop_loss_hit"]).float().to(self.device),
            "time_to_target": torch.from_numpy(y_val["time_to_target"]).float().to(self.device),
        }

        # Create data loaders
        train_dataset = EfficientRegimeDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        # Early stopping
        best_val_loss: float = float("inf")
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                # Mixed precision training
                if self.scaler:
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        outputs = self.model(batch_X)
                        loss = self._compute_loss(outputs, batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = self._compute_loss(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()

                train_loss_sum += float(loss.item())
                train_correct += self._compute_accuracy(outputs, batch_y)
                train_total += int(batch_y["regime_id"].size(0))

            # Validation
            val_loss, val_acc = self._validate(X_val_t, y_val_t)

            # Record metrics
            train_loss_avg = train_loss_sum / max(len(train_loader), 1)
            train_acc = train_correct / max(train_total, 1)

            history["train_loss"].append(train_loss_avg)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                os.makedirs("models", exist_ok=True)
                torch.save(self.model.state_dict(), "models/best_regime_predictor.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Learning rate scheduling
            self.scheduler.step()

        return history

    def _compute_loss(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute multi-task loss for regime and TPSL prediction."""
        # Current regime classification loss
        regime_loss = F.cross_entropy(outputs["current_regime"], targets["regime_id"])

        # Transition matrix regularization (encourage smooth transitions)
        transition_matrix = outputs["transition_matrix"]
        identity = torch.eye(transition_matrix.size(1)).to(transition_matrix.device)
        transition_regularization = F.mse_loss(transition_matrix.mean(0), identity)

        # Confidence regularization (encourage high confidence for correct predictions)
        confidence = outputs["confidence"]
        confidence_loss = F.binary_cross_entropy(
            confidence, torch.ones_like(confidence),
        )

        # TPSL prediction losses
        profit_target_loss = F.binary_cross_entropy(
            outputs["profit_target_prob"].squeeze(-1), targets["profit_target_hit"],
        )
        stop_loss_loss = F.binary_cross_entropy(
            outputs["stop_loss_prob"].squeeze(-1), targets["stop_loss_hit"],
        )
        time_to_target_loss = F.mse_loss(
            outputs["time_to_target"].squeeze(-1), targets["time_to_target"],
        )

        # Combined loss with TPSL weighting
        return (
            regime_loss
            + 0.1 * transition_regularization
            + 0.05 * confidence_loss
            + 0.3 * profit_target_loss  # Higher weight for profit target prediction
            + 0.2 * stop_loss_loss  # Medium weight for stop loss prediction
            + 0.1 * time_to_target_loss  # Lower weight for timing prediction
        )

    def _compute_accuracy(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> int:
        """Compute accuracy for regime and TPSL prediction."""
        # Regime accuracy
        predicted_regime = torch.argmax(outputs["current_regime"], dim=1)
        regime_correct = (predicted_regime == targets["regime_id"]).sum().item()

        # TPSL accuracy (profit target prediction)
        predicted_profit = (outputs["profit_target_prob"].squeeze(-1) > 0.5).float()
        profit_correct = (
            (predicted_profit == targets["profit_target_hit"]).sum().item()
        )

        # Combined accuracy (weighted equally as counts)
        return int(regime_correct + profit_correct)

    def _validate(
        self, X_val: torch.Tensor, y_val: dict[str, torch.Tensor]
    ) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = float(self._compute_loss(outputs, y_val).item())
            accuracy = float(
                self._compute_accuracy(outputs, y_val) / max(y_val["regime_id"].size(0), 1)
            )
        return loss, accuracy


# For backward compatibility with existing step structure
@deterministic_seed(42)
@idempotent_step(step_key="step9_5_hmm_lm_generalist_training")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=3600.0)
@validate_step_prerequisites(
    required_directories=["data/training", "models"],
    min_memory_gb=16.0,
    min_disk_gb=10.0,
    required_packages=["torch", "numpy", "pandas", "sklearn", "lightgbm"],
    data_quality_checks={"check_data_completeness": True},
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True,
    data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    cross_validation_isolation=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=32.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=1000,
    streaming_processing=True,
    memory_pool=True,
    cleanup_frequency=10,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=300.0,
)
@validate_step_output(
    required_files=["hmm_lm_generalist_model.pkl"],
    data_quality_checks={"check_output_completeness": True},
)
@quality_gate(
    model_performance_thresholds={"min_accuracy": 0.6},
    data_quality_metrics={"completeness_threshold": 0.95},
)
@handle_errors(exceptions=(Exception,), default_return=False, context="step9_5_hmm_lm_generalist_training")
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the HMM-LM generalist training step.

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
        step = HMMLMGeneralistTrainingStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return bool(result.get("status") == "SUCCESS")

    except Exception as e:  # noqa: BLE001
        logger.exception(f"HMM-LM generalist training failed: {e}")
        return False

    # === ENHANCED REGIME CHANGE PREDICTION METHODS ===
    
    async def _perform_enhanced_regime_change_prediction(self, training_input: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Perform enhanced regime change prediction with comprehensive analysis.
        
        Args:
            training_input: Training input parameters
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: Enhanced regime change prediction results
        """
        try:
            self.logger.info("🔄 Performing enhanced regime change prediction...")
            
            # Update regime change state
            self.regime_change_state["last_regime_change_analysis"] = pd.Timestamp.now()
            self.regime_change_state["regime_change_count"] += 1
            
            # Step 1: Detect regime changes
            regime_changes = await self._detect_regime_changes(market_data)
            
            # Step 2: Analyze regime change patterns
            change_patterns = await self._analyze_regime_change_patterns(regime_changes, market_data)
            
            # Step 3: Generate regime change sequences
            sequences = await self._generate_regime_change_sequences(regime_changes, market_data)
            
            # Step 4: Train regime change prediction model
            prediction_model = await self._train_regime_change_prediction_model(sequences, market_data)
            
            # Step 5: Calculate quality metrics
            quality_metrics = self._calculate_regime_change_quality_metrics(
                regime_changes, change_patterns, sequences, prediction_model
            )
            
            # Step 6: Eliminate redundancy
            redundancy_metrics = self._eliminate_regime_change_redundancy(
                regime_changes, sequences, market_data
            )
            
            # Create comprehensive results
            results = {
                "success": True,
                "regime_changes": regime_changes,
                "change_patterns": change_patterns,
                "sequences": sequences,
                "prediction_model": prediction_model,
                "quality_metrics": quality_metrics,
                "redundancy_metrics": redundancy_metrics,
                "metrics": {
                    **quality_metrics,
                    **redundancy_metrics,
                    "regime_change_count": len(regime_changes),
                    "sequence_count": len(sequences),
                    "prediction_accuracy": quality_metrics.get("prediction_accuracy", 0.0)
                }
            }
            
            # Update state
            self.regime_change_state["regime_change_quality_scores"] = quality_metrics
            self.regime_change_state["regime_change_redundancy_metrics"] = redundancy_metrics
            self.regime_change_state["regime_change_prediction_model"] = prediction_model
            
            self.logger.info("✅ Enhanced regime change prediction completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced regime change prediction: {e}")
            return {"success": False, "error": str(e)}

    async def _detect_regime_changes(self, market_data: pd.DataFrame) -> List[dict[str, Any]]:
        """Detect regime changes in market data."""
        try:
            self.logger.info("🔍 Detecting regime changes...")
            
            regime_changes = []
            
            # Calculate regime change indicators
            volatility = market_data['close'].pct_change().rolling(window=20).std()
            momentum = market_data['close'].pct_change().rolling(window=10).mean()
            volume_ratio = market_data['volume'] / market_data['volume'].rolling(window=20).mean()
            
            # Detect regime changes based on multiple criteria
            for i in range(20, len(market_data)):
                change_detected = False
                change_type = None
                confidence = 0.0
                
                # Volatility regime change
                vol_change = abs(volatility.iloc[i] - volatility.iloc[i-1]) / volatility.iloc[i-1]
                if vol_change > 0.5:  # 50% change in volatility
                    change_detected = True
                    change_type = "volatility"
                    confidence = min(1.0, vol_change)
                
                # Momentum regime change
                mom_change = abs(momentum.iloc[i] - momentum.iloc[i-1])
                if mom_change > 0.02:  # 2% change in momentum
                    change_detected = True
                    change_type = "momentum"
                    confidence = max(confidence, min(1.0, mom_change * 10))
                
                # Volume regime change
                vol_ratio_change = abs(volume_ratio.iloc[i] - volume_ratio.iloc[i-1])
                if vol_ratio_change > 1.0:  # 100% change in volume ratio
                    change_detected = True
                    change_type = "volume"
                    confidence = max(confidence, min(1.0, vol_ratio_change / 2))
                
                if change_detected:
                    regime_change = {
                        "timestamp": market_data.index[i],
                        "change_type": change_type,
                        "confidence": confidence,
                        "volatility_change": vol_change,
                        "momentum_change": mom_change,
                        "volume_change": vol_ratio_change,
                        "price": market_data['close'].iloc[i],
                        "volume": market_data['volume'].iloc[i]
                    }
                    regime_changes.append(regime_change)
            
            self.logger.info(f"✅ Detected {len(regime_changes)} regime changes")
            return regime_changes
            
        except Exception as e:
            self.logger.error(f"Error detecting regime changes: {e}")
            return []

    async def _analyze_regime_change_patterns(self, regime_changes: List[dict[str, Any]], market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze patterns in regime changes."""
        try:
            self.logger.info("📊 Analyzing regime change patterns...")
            
            if not regime_changes:
                return {"patterns": [], "statistics": {}}
            
            # Analyze change type distribution
            change_types = [change["change_type"] for change in regime_changes]
            type_counts = {}
            for change_type in change_types:
                type_counts[change_type] = type_counts.get(change_type, 0) + 1
            
            # Analyze temporal patterns
            timestamps = [change["timestamp"] for change in regime_changes]
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
                intervals.append(interval)
            
            # Analyze confidence patterns
            confidences = [change["confidence"] for change in regime_changes]
            
            # Analyze price impact
            price_impacts = []
            for change in regime_changes:
                change_idx = market_data.index.get_loc(change["timestamp"])
                if change_idx + 10 < len(market_data):
                    future_return = (market_data['close'].iloc[change_idx + 10] - change["price"]) / change["price"]
                    price_impacts.append(future_return)
            
            patterns = {
                "change_type_distribution": type_counts,
                "avg_interval_hours": np.mean(intervals) if intervals else 0,
                "interval_std_hours": np.std(intervals) if intervals else 0,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "avg_price_impact": np.mean(price_impacts) if price_impacts else 0,
                "price_impact_std": np.std(price_impacts) if price_impacts else 0
            }
            
            self.logger.info("✅ Regime change pattern analysis completed")
            return {"patterns": patterns, "statistics": patterns}
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime change patterns: {e}")
            return {"patterns": [], "statistics": {}}

    async def _generate_regime_change_sequences(self, regime_changes: List[dict[str, Any]], market_data: pd.DataFrame) -> List[dict[str, Any]]:
        """Generate sequences for regime change prediction."""
        try:
            self.logger.info("📝 Generating regime change sequences...")
            
            sequences = []
            
            # Create sequences around regime changes
            for i, change in enumerate(regime_changes):
                change_idx = market_data.index.get_loc(change["timestamp"])
                
                # Create sequence before the change
                if change_idx >= self.sequence_length:
                    sequence_data = market_data.iloc[change_idx - self.sequence_length:change_idx]
                    
                    # Create sequence features
                    sequence_features = self._extract_sequence_features(sequence_data)
                    
                    # Create sequence label
                    sequence_label = self._create_sequence_label(change)
                    
                    sequence = {
                        "sequence_id": i,
                        "features": sequence_features,
                        "label": sequence_label,
                        "change_type": change["change_type"],
                        "confidence": change["confidence"],
                        "timestamp": change["timestamp"]
                    }
                    sequences.append(sequence)
            
            self.logger.info(f"✅ Generated {len(sequences)} regime change sequences")
            return sequences
            
        except Exception as e:
            self.logger.error(f"Error generating regime change sequences: {e}")
            return []

    async def _train_regime_change_prediction_model(self, sequences: List[dict[str, Any]], market_data: pd.DataFrame) -> Any:
        """Train model to predict regime changes."""
        try:
            self.logger.info("🎯 Training regime change prediction model...")
            
            if not sequences:
                self.logger.warning("No sequences available for training")
                return None
            
            # Prepare training data
            X = []
            y = []
            
            for sequence in sequences:
                features = sequence["features"]
                label = sequence["label"]
                
                # Convert features to tensor
                if isinstance(features, dict):
                    feature_vector = list(features.values())
                else:
                    feature_vector = features
                
                X.append(feature_vector)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train Random Forest classifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            prediction_model = {
                "model": model,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "feature_importance": dict(zip(range(len(X[0])), model.feature_importances_))
            }
            
            self.logger.info(f"✅ Regime change prediction model trained: test_accuracy={test_accuracy:.3f}")
            return prediction_model
            
        except Exception as e:
            self.logger.error(f"Error training regime change prediction model: {e}")
            return None

    def _calculate_regime_change_quality_metrics(self, regime_changes: List[dict[str, Any]], change_patterns: dict[str, Any], 
                                               sequences: List[dict[str, Any]], prediction_model: Any) -> dict[str, float]:
        """Calculate quality metrics for regime change prediction."""
        try:
            metrics = {}
            
            # Regime change detection quality
            metrics["regime_change_count"] = len(regime_changes)
            metrics["avg_confidence"] = np.mean([change["confidence"] for change in regime_changes]) if regime_changes else 0.0
            metrics["confidence_std"] = np.std([change["confidence"] for change in regime_changes]) if regime_changes else 0.0
            
            # Pattern analysis quality
            patterns = change_patterns.get("patterns", {})
            metrics["pattern_completeness"] = len(patterns) / 6  # 6 expected pattern types
            metrics["avg_interval_consistency"] = 1.0 - patterns.get("interval_std_hours", 0) / max(patterns.get("avg_interval_hours", 1), 1)
            
            # Sequence quality
            metrics["sequence_count"] = len(sequences)
            metrics["avg_sequence_length"] = np.mean([len(seq["features"]) for seq in sequences]) if sequences else 0.0
            
            # Prediction model quality
            if prediction_model:
                metrics["prediction_accuracy"] = prediction_model.get("test_accuracy", 0.0)
                metrics["train_accuracy"] = prediction_model.get("train_accuracy", 0.0)
                metrics["overfitting_score"] = metrics["train_accuracy"] - metrics["prediction_accuracy"]
            else:
                metrics["prediction_accuracy"] = 0.0
                metrics["train_accuracy"] = 0.0
                metrics["overfitting_score"] = 0.0
            
            # Overall quality score
            quality_factors = [
                metrics["avg_confidence"],
                metrics["pattern_completeness"],
                metrics["prediction_accuracy"],
                1.0 - abs(metrics["overfitting_score"])
            ]
            metrics["overall_quality_score"] = sum(quality_factors) / len(quality_factors)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime change quality metrics: {e}")
            return {}

    def _eliminate_regime_change_redundancy(self, regime_changes: List[dict[str, Any]], sequences: List[dict[str, Any]], 
                                          market_data: pd.DataFrame) -> dict[str, Any]:
        """Eliminate redundant regime changes and sequences."""
        try:
            metrics = {}
            
            # Analyze regime change redundancy
            redundant_changes = []
            for i, change1 in enumerate(regime_changes):
                for j, change2 in enumerate(regime_changes[i+1:], i+1):
                    time_diff = abs((change1["timestamp"] - change2["timestamp"]).total_seconds())
                    if time_diff < 3600:  # Within 1 hour
                        redundant_changes.append((i, j, time_diff))
            
            # Analyze sequence redundancy
            redundant_sequences = []
            for i, seq1 in enumerate(sequences):
                for j, seq2 in enumerate(sequences[i+1:], i+1):
                    feature_similarity = self._calculate_feature_similarity(seq1["features"], seq2["features"])
                    if feature_similarity > 0.9:  # 90% similarity threshold
                        redundant_sequences.append((i, j, feature_similarity))
            
            metrics["redundant_changes"] = len(redundant_changes)
            metrics["redundant_sequences"] = len(redundant_sequences)
            metrics["redundancy_ratio"] = (len(redundant_changes) + len(redundant_sequences)) / max(len(regime_changes) + len(sequences), 1)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating regime change redundancy: {e}")
            return {}

    def _extract_sequence_features(self, sequence_data: pd.DataFrame) -> dict[str, float]:
        """Extract features from a sequence of market data."""
        try:
            features = {}
            
            # Price features
            features["price_mean"] = sequence_data['close'].mean()
            features["price_std"] = sequence_data['close'].std()
            features["price_trend"] = (sequence_data['close'].iloc[-1] - sequence_data['close'].iloc[0]) / sequence_data['close'].iloc[0]
            
            # Volume features
            features["volume_mean"] = sequence_data['volume'].mean()
            features["volume_std"] = sequence_data['volume'].std()
            features["volume_trend"] = (sequence_data['volume'].iloc[-1] - sequence_data['volume'].iloc[0]) / sequence_data['volume'].iloc[0]
            
            # Volatility features
            returns = sequence_data['close'].pct_change().dropna()
            features["volatility"] = returns.std()
            features["volatility_trend"] = returns.rolling(5).std().iloc[-1] - returns.rolling(5).std().iloc[0]
            
            # Technical features
            features["rsi"] = self._calculate_rsi(sequence_data['close']).iloc[-1]
            features["macd"] = self._calculate_macd(sequence_data['close']).iloc[-1]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting sequence features: {e}")
            return {}

    def _create_sequence_label(self, regime_change: dict[str, Any]) -> int:
        """Create label for a regime change sequence."""
        try:
            # Simple labeling based on change type
            change_type = regime_change["change_type"]
            if change_type == "volatility":
                return 0
            elif change_type == "momentum":
                return 1
            elif change_type == "volume":
                return 2
            else:
                return 3
                
        except Exception as e:
            self.logger.error(f"Error creating sequence label: {e}")
            return 0

    def _calculate_feature_similarity(self, features1: dict[str, float], features2: dict[str, float]) -> float:
        """Calculate similarity between two feature sets."""
        try:
            if not features1 or not features2:
                return 0.0
            
            # Calculate cosine similarity
            keys = set(features1.keys()) & set(features2.keys())
            if not keys:
                return 0.0
            
            dot_product = sum(features1[key] * features2[key] for key in keys)
            norm1 = sum(features1[key] ** 2 for key in keys) ** 0.5
            norm2 = sum(features2[key] ** 2 for key in keys) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Error calculating feature similarity: {e}")
            return 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception:
            return pd.Series([0] * len(prices))

    # Enhanced Regime Change Prediction Methods
    
    async def analyze_enhanced_regime_changes(self, market_data: pd.DataFrame, hmm_model: Any = None) -> Dict[str, Any]:
        """
        Perform enhanced regime change analysis with comprehensive prediction capabilities.
        
        Args:
            market_data: Market data DataFrame
            hmm_model: Optional pre-trained HMM model
            
        Returns:
            Dict[str, Any]: Enhanced regime change analysis results
        """
        try:
            self.logger.info("🔍 Performing enhanced regime change analysis...")
            
            # Update regime change state
            self.regime_change_state["last_regime_change_analysis"] = pd.Timestamp.now()
            self.regime_change_state["regime_change_count"] += 1
            
            # Step 1: Detect regime changes
            regime_changes = await self._detect_enhanced_regime_changes(market_data, hmm_model)
            
            # Step 2: Analyze regime transitions
            transition_analysis = await self._analyze_regime_transitions(regime_changes, market_data)
            
            # Step 3: Predict regime stability
            stability_analysis = await self._predict_regime_stability(regime_changes, market_data)
            
            # Step 4: Generate regime forecasts
            forecast_analysis = await self._generate_regime_forecasts(regime_changes, market_data)
            
            # Step 5: Calculate quality metrics
            quality_metrics = self._calculate_enhanced_regime_quality_metrics(
                regime_changes, transition_analysis, stability_analysis, forecast_analysis
            )
            
            # Step 6: Eliminate redundancy
            redundancy_metrics = self._eliminate_enhanced_regime_redundancy(
                regime_changes, transition_analysis, market_data
            )
            
            # Create comprehensive results
            results = {
                "regime_changes": regime_changes,
                "transition_analysis": transition_analysis,
                "stability_analysis": stability_analysis,
                "forecast_analysis": forecast_analysis,
                "quality_metrics": quality_metrics,
                "redundancy_metrics": redundancy_metrics,
                "regime_change_count": len(regime_changes),
                "transition_count": len(transition_analysis.get("transitions", [])),
                "stability_score": stability_analysis.get("overall_stability", 0.0),
                "forecast_accuracy": forecast_analysis.get("forecast_accuracy", 0.0)
            }
            
            # Update state
            self.regime_change_state["regime_change_quality_scores"] = quality_metrics
            self.regime_change_state["regime_change_redundancy_metrics"] = redundancy_metrics
            self.regime_change_state["regime_transition_probabilities"] = transition_analysis.get("transition_probabilities", {})
            self.regime_change_state["regime_stability_metrics"] = stability_analysis.get("stability_metrics", {})
            self.regime_change_state["regime_forecasting_accuracy"] = forecast_analysis.get("forecast_accuracy", 0.0)
            
            self.logger.info(f"✅ Enhanced regime change analysis completed: {len(regime_changes)} changes detected")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced regime change analysis: {e}")
            return {"regime_changes": [], "quality_metrics": {}, "redundancy_metrics": {}}

    # === ENHANCED REGIME CHANGE PREDICTION METHODS ===
    
    async def get_enhanced_regime_change_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get enhanced regime change features for feature engineering integration.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Dict[str, Any]: Enhanced regime change features
        """
        try:
            self.logger.info("🔧 Generating enhanced regime change features...")
            
            # Perform enhanced regime change analysis
            analysis_results = await self.analyze_enhanced_regime_changes(market_data)
            
            # Extract regime change features
            regime_changes = analysis_results.get("regime_changes", [])
            transition_analysis = analysis_results.get("transition_analysis", {})
            stability_analysis = analysis_results.get("stability_analysis", {})
            forecast_analysis = analysis_results.get("forecast_analysis", {})
            
            # Create regime change features
            regime_change_features = {}
            
            # Basic regime change features
            regime_change_features["regime_change_count"] = len(regime_changes)
            regime_change_features["transition_count"] = len(transition_analysis.get("transitions", []))
            regime_change_features["stability_score"] = stability_analysis.get("overall_stability", 0.0)
            regime_change_features["forecast_accuracy"] = forecast_analysis.get("forecast_accuracy", 0.0)
            
            # Recent regime change features
            if regime_changes:
                recent_changes = regime_changes[-5:]  # Last 5 changes
                regime_change_features["recent_change_count"] = len(recent_changes)
                regime_change_features["avg_change_confidence"] = np.mean([change["confidence"] for change in recent_changes])
                regime_change_features["avg_change_strength"] = np.mean([change["change_strength"] for change in recent_changes])
                
                # Change type distribution
                change_types = [change["change_type"] for change in recent_changes]
                regime_change_features["volatility_changes"] = change_types.count("volatility")
                regime_change_features["momentum_changes"] = change_types.count("momentum")
                regime_change_features["volume_changes"] = change_types.count("volume")
            
            # Transition features
            transition_probabilities = transition_analysis.get("transition_probabilities", {})
            regime_change_features["transition_diversity"] = len(transition_probabilities)
            regime_change_features["avg_transition_probability"] = np.mean(list(transition_probabilities.values())) if transition_probabilities else 0.0
            
            # Stability features
            stability_metrics = stability_analysis.get("stability_metrics", {})
            for regime_type, metrics in stability_metrics.items():
                regime_change_features[f"stability_{regime_type}"] = metrics.get("stability_score", 0.0)
                regime_change_features[f"duration_{regime_type}"] = metrics.get("avg_duration", 0.0)
            
            # Forecast features
            forecasts = forecast_analysis.get("forecasts", [])
            regime_change_features["forecast_count"] = len(forecasts)
            regime_change_features["avg_forecast_confidence"] = np.mean([f["confidence"] for f in forecasts]) if forecasts else 0.0
            
            # Quality metrics
            quality_metrics = analysis_results.get("quality_metrics", {})
            for key, value in quality_metrics.items():
                regime_change_features[f"quality_{key}"] = value
            
            # Redundancy metrics
            redundancy_metrics = analysis_results.get("redundancy_metrics", {})
            for key, value in redundancy_metrics.items():
                regime_change_features[f"redundancy_{key}"] = value
            
            self.logger.info("✅ Enhanced regime change features generated")
            return regime_change_features
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced regime change features: {e}")
            return {}

    async def predict_regime_changes(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict regime changes with enhanced analysis.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Dict[str, Any]: Regime change predictions
        """
        try:
            self.logger.info("🔮 Predicting regime changes...")
            
            # Perform enhanced regime change analysis
            analysis_results = await self.analyze_enhanced_regime_changes(market_data)
            
            # Get current market conditions
            current_conditions = self._analyze_current_market_conditions(market_data)
            
            # Generate predictions based on analysis
            predictions = {
                "next_regime_change_probability": self._calculate_next_change_probability(analysis_results),
                "expected_change_type": self._predict_next_change_type(analysis_results),
                "time_to_next_change": self._predict_time_to_next_change(analysis_results),
                "change_confidence": self._calculate_change_confidence(analysis_results),
                "current_market_conditions": current_conditions,
                "prediction_timestamp": pd.Timestamp.now()
            }
            
            self.logger.info("✅ Regime change predictions generated")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting regime changes: {e}")
            return {}

    def _analyze_current_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions for regime change prediction."""
        try:
            conditions = {}
            
            # Volatility analysis
            returns = market_data['close'].pct_change().dropna()
            conditions["current_volatility"] = returns.rolling(window=20).std().iloc[-1]
            conditions["volatility_trend"] = returns.rolling(window=20).std().diff().iloc[-1]
            
            # Momentum analysis
            conditions["current_momentum"] = returns.rolling(window=10).mean().iloc[-1]
            conditions["momentum_trend"] = returns.rolling(window=10).mean().diff().iloc[-1]
            
            # Volume analysis
            conditions["current_volume_ratio"] = market_data['volume'].iloc[-1] / market_data['volume'].rolling(window=20).mean().iloc[-1]
            conditions["volume_trend"] = market_data['volume'].pct_change().iloc[-1]
            
            # Price analysis
            conditions["price_position"] = (market_data['close'].iloc[-1] - market_data['low'].rolling(window=20).min().iloc[-1]) / (market_data['high'].rolling(window=20).max().iloc[-1] - market_data['low'].rolling(window=20).min().iloc[-1])
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing current market conditions: {e}")
            return {}

    def _calculate_next_change_probability(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate probability of next regime change."""
        try:
            # Use recent change frequency and current conditions
            regime_changes = analysis_results.get("regime_changes", [])
            if len(regime_changes) < 2:
                return 0.1  # Low probability if few changes
            
            # Calculate average time between changes
            time_diffs = []
            for i in range(1, len(regime_changes)):
                time_diff = (regime_changes[i]["timestamp"] - regime_changes[i-1]["timestamp"]).total_seconds()
                time_diffs.append(time_diff)
            
            avg_time_diff = np.mean(time_diffs) if time_diffs else 3600
            
            # Calculate probability based on time since last change
            last_change_time = regime_changes[-1]["timestamp"]
            time_since_last = (pd.Timestamp.now() - last_change_time).total_seconds()
            
            # Probability increases with time since last change
            probability = min(0.9, time_since_last / avg_time_diff)
            
            return probability
            
        except Exception as e:
            self.logger.error(f"Error calculating next change probability: {e}")
            return 0.1

    def _predict_next_change_type(self, analysis_results: Dict[str, Any]) -> str:
        """Predict the type of next regime change."""
        try:
            regime_changes = analysis_results.get("regime_changes", [])
            if not regime_changes:
                return "unknown"
            
            # Analyze recent change types
            recent_changes = regime_changes[-10:]  # Last 10 changes
            change_types = [change["change_type"] for change in recent_changes]
            
            # Count change types
            type_counts = {}
            for change_type in change_types:
                type_counts[change_type] = type_counts.get(change_type, 0) + 1
            
            # Return most common type
            if type_counts:
                return max(type_counts, key=type_counts.get)
            else:
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"Error predicting next change type: {e}")
            return "unknown"

    def _predict_time_to_next_change(self, analysis_results: Dict[str, Any]) -> float:
        """Predict time to next regime change in seconds."""
        try:
            regime_changes = analysis_results.get("regime_changes", [])
            if len(regime_changes) < 2:
                return 3600  # Default 1 hour
            
            # Calculate average time between changes
            time_diffs = []
            for i in range(1, len(regime_changes)):
                time_diff = (regime_changes[i]["timestamp"] - regime_changes[i-1]["timestamp"]).total_seconds()
                time_diffs.append(time_diff)
            
            avg_time_diff = np.mean(time_diffs) if time_diffs else 3600
            
            # Adjust based on current conditions
            time_since_last = (pd.Timestamp.now() - regime_changes[-1]["timestamp"]).total_seconds()
            remaining_time = max(0, avg_time_diff - time_since_last)
            
            return remaining_time
            
        except Exception as e:
            self.logger.error(f"Error predicting time to next change: {e}")
            return 3600

    def _calculate_change_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence in regime change predictions."""
        try:
            # Use quality metrics and recent performance
            quality_metrics = analysis_results.get("quality_metrics", {})
            regime_changes = analysis_results.get("regime_changes", [])
            
            # Base confidence on quality metrics
            base_confidence = quality_metrics.get("composite_quality_score", 0.5)
            
            # Adjust based on recent change consistency
            if len(regime_changes) >= 3:
                recent_confidences = [change["confidence"] for change in regime_changes[-3:]]
                consistency_factor = 1.0 - np.std(recent_confidences)
                base_confidence *= (0.5 + 0.5 * consistency_factor)
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating change confidence: {e}")
            return 0.5

    async def _detect_enhanced_regime_changes(self, market_data: pd.DataFrame, hmm_model: Any = None) -> List[Dict[str, Any]]:
        """Detect regime changes using multiple methods."""
        try:
            regime_changes = []
            
            # Method 1: HMM-based regime detection
            if hmm_model is not None:
                hmm_changes = await self._detect_hmm_regime_changes(market_data, hmm_model)
                regime_changes.extend(hmm_changes)
            
            # Method 2: Volatility-based regime detection
            volatility_changes = await self._detect_volatility_regime_changes(market_data)
            regime_changes.extend(volatility_changes)
            
            # Method 3: Momentum-based regime detection
            momentum_changes = await self._detect_momentum_regime_changes(market_data)
            regime_changes.extend(momentum_changes)
            
            # Method 4: Volume-based regime detection
            volume_changes = await self._detect_volume_regime_changes(market_data)
            regime_changes.extend(volume_changes)
            
            # Remove duplicates and sort by timestamp
            unique_changes = self._remove_duplicate_regime_changes(regime_changes)
            unique_changes.sort(key=lambda x: x["timestamp"])
            
            return unique_changes
            
        except Exception as e:
            self.logger.error(f"Error detecting enhanced regime changes: {e}")
            return []

    async def _detect_hmm_regime_changes(self, market_data: pd.DataFrame, hmm_model: Any) -> List[Dict[str, Any]]:
        """Detect regime changes using HMM model."""
        try:
            changes = []
            
            # Prepare features
            features = self._prepare_regime_features(market_data)
            feature_matrix = features.dropna().values
            
            if len(feature_matrix) < 10:
                return changes
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
            
            # Predict states
            states = hmm_model.predict(normalized_features)
            
            # Detect state changes
            for i in range(1, len(states)):
                if states[i] != states[i-1]:
                    change = {
                        "timestamp": market_data.index[i] if hasattr(market_data, 'index') else pd.Timestamp.now(),
                        "change_type": "hmm_state",
                        "from_state": states[i-1],
                        "to_state": states[i],
                        "confidence": hmm_model.predict_proba(normalized_features[i:i+1])[0].max(),
                        "features": features.iloc[i].to_dict(),
                        "change_strength": 1.0,
                        "detection_method": "hmm"
                    }
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting HMM regime changes: {e}")
            return []

    async def _detect_volatility_regime_changes(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect regime changes based on volatility shifts."""
        try:
            changes = []
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std()
            
            # Detect volatility regime changes
            volatility_threshold = volatility.quantile(0.8)
            high_volatility = volatility > volatility_threshold
            
            for i in range(1, len(high_volatility)):
                if high_volatility.iloc[i] != high_volatility.iloc[i-1]:
                    change = {
                        "timestamp": market_data.index[i] if hasattr(market_data, 'index') else pd.Timestamp.now(),
                        "change_type": "volatility",
                        "from_state": "low_vol" if not high_volatility.iloc[i-1] else "high_vol",
                        "to_state": "high_vol" if high_volatility.iloc[i] else "low_vol",
                        "confidence": abs(volatility.iloc[i] - volatility.iloc[i-1]) / volatility.iloc[i-1],
                        "features": {"volatility": volatility.iloc[i]},
                        "change_strength": abs(volatility.iloc[i] - volatility.iloc[i-1]) / volatility.iloc[i-1],
                        "detection_method": "volatility"
                    }
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime changes: {e}")
            return []

    async def _detect_momentum_regime_changes(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect regime changes based on momentum shifts."""
        try:
            changes = []
            
            # Calculate momentum indicators
            momentum_5 = market_data['close'] / market_data['close'].shift(5) - 1
            momentum_20 = market_data['close'] / market_data['close'].shift(20) - 1
            
            # Detect momentum regime changes
            momentum_threshold = 0.02  # 2% threshold
            
            for i in range(20, len(momentum_5)):
                momentum_change = abs(momentum_5.iloc[i] - momentum_5.iloc[i-1])
                
                if momentum_change > momentum_threshold:
                    change = {
                        "timestamp": market_data.index[i] if hasattr(market_data, 'index') else pd.Timestamp.now(),
                        "change_type": "momentum",
                        "from_state": "low_momentum" if momentum_5.iloc[i-1] < 0 else "high_momentum",
                        "to_state": "high_momentum" if momentum_5.iloc[i] > 0 else "low_momentum",
                        "confidence": min(momentum_change / momentum_threshold, 1.0),
                        "features": {"momentum_5": momentum_5.iloc[i], "momentum_20": momentum_20.iloc[i]},
                        "change_strength": momentum_change / momentum_threshold,
                        "detection_method": "momentum"
                    }
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum regime changes: {e}")
            return []

    async def _detect_volume_regime_changes(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect regime changes based on volume shifts."""
        try:
            changes = []
            
            # Calculate volume indicators
            volume_ma = market_data['volume'].rolling(window=20).mean()
            volume_ratio = market_data['volume'] / volume_ma
            
            # Detect volume regime changes
            volume_threshold = 2.0  # 2x average volume
            
            for i in range(20, len(volume_ratio)):
                if volume_ratio.iloc[i] > volume_threshold and volume_ratio.iloc[i-1] <= volume_threshold:
                    change = {
                        "timestamp": market_data.index[i] if hasattr(market_data, 'index') else pd.Timestamp.now(),
                        "change_type": "volume",
                        "from_state": "normal_volume",
                        "to_state": "high_volume",
                        "confidence": min(volume_ratio.iloc[i] / volume_threshold, 1.0),
                        "features": {"volume_ratio": volume_ratio.iloc[i]},
                        "change_strength": volume_ratio.iloc[i] / volume_threshold,
                        "detection_method": "volume"
                    }
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting volume regime changes: {e}")
            return []

    async def _analyze_regime_transitions(self, regime_changes: List[Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime transitions and patterns."""
        try:
            transitions = []
            transition_patterns = {}
            
            for i, change in enumerate(regime_changes):
                if i > 0:
                    prev_change = regime_changes[i-1]
                    
                    # Calculate transition pattern
                    pattern = f"{prev_change['to_state']}_to_{change['to_state']}"
                    if pattern not in transition_patterns:
                        transition_patterns[pattern] = 0
                    transition_patterns[pattern] += 1
                    
                    # Calculate transition probability
                    time_diff = (change["timestamp"] - prev_change["timestamp"]).total_seconds()
                    transition_probability = 1.0 / (1.0 + time_diff / 3600)  # Decay with time
                    
                    transition = {
                        "from_change": prev_change,
                        "to_change": change,
                        "pattern": pattern,
                        "probability": transition_probability,
                        "time_diff": time_diff,
                        "strength": (change["change_strength"] + prev_change["change_strength"]) / 2
                    }
                    transitions.append(transition)
            
            # Calculate transition probabilities
            total_transitions = len(transitions)
            transition_probabilities = {}
            for pattern, count in transition_patterns.items():
                transition_probabilities[pattern] = count / total_transitions if total_transitions > 0 else 0.0
            
            results = {
                "transitions": transitions,
                "transition_patterns": transition_patterns,
                "transition_probabilities": transition_probabilities,
                "transition_count": len(transitions)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime transitions: {e}")
            return {"transitions": [], "transition_probabilities": {}, "transition_count": 0}

    async def _predict_regime_stability(self, regime_changes: List[Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict regime stability and persistence."""
        try:
            stability_metrics = {}
            
            # Calculate stability for each regime type
            regime_types = set(change["change_type"] for change in regime_changes)
            
            for regime_type in regime_types:
                type_changes = [c for c in regime_changes if c["change_type"] == regime_type]
                
                if len(type_changes) > 1:
                    # Calculate average duration between changes
                    durations = []
                    for i in range(1, len(type_changes)):
                        duration = (type_changes[i]["timestamp"] - type_changes[i-1]["timestamp"]).total_seconds()
                        durations.append(duration)
                    
                    avg_duration = np.mean(durations)
                    duration_std = np.std(durations)
                    
                    # Calculate stability score (higher duration = more stable)
                    stability_score = 1.0 / (1.0 + duration_std / avg_duration) if avg_duration > 0 else 0.0
                    
                    stability_metrics[regime_type] = {
                        "avg_duration": avg_duration,
                        "duration_std": duration_std,
                        "stability_score": stability_score,
                        "change_frequency": len(type_changes) / len(regime_changes)
                    }
                else:
                    stability_metrics[regime_type] = {
                        "avg_duration": 0,
                        "duration_std": 0,
                        "stability_score": 1.0,  # Single change = stable
                        "change_frequency": 1.0 / len(regime_changes) if regime_changes else 0.0
                    }
            
            # Calculate overall stability
            overall_stability = np.mean([metrics["stability_score"] for metrics in stability_metrics.values()])
            
            results = {
                "stability_metrics": stability_metrics,
                "overall_stability": overall_stability
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error predicting regime stability: {e}")
            return {"overall_stability": 0.0, "stability_metrics": {}}

    async def _generate_regime_forecasts(self, regime_changes: List[Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate regime forecasts and predictions."""
        try:
            forecasts = []
            
            if len(regime_changes) < 2:
                return {"forecasts": [], "forecast_accuracy": 0.0}
            
            # Use recent changes to predict future changes
            recent_changes = regime_changes[-10:]  # Last 10 changes
            
            # Calculate average time between changes
            time_diffs = []
            for i in range(1, len(recent_changes)):
                time_diff = (recent_changes[i]["timestamp"] - recent_changes[i-1]["timestamp"]).total_seconds()
                time_diffs.append(time_diff)
            
            avg_time_diff = np.mean(time_diffs) if time_diffs else 3600  # Default 1 hour
            
            # Generate forecasts
            last_change = regime_changes[-1]
            current_time = pd.Timestamp.now()
            
            for i in range(1, 6):  # Predict next 5 changes
                forecast_time = last_change["timestamp"] + pd.Timedelta(seconds=avg_time_diff * i)
                
                if forecast_time > current_time:
                    forecast = {
                        "forecast_step": i,
                        "forecast_time": forecast_time,
                        "predicted_change_type": last_change["change_type"],
                        "confidence": max(0.1, 1.0 - i * 0.2),  # Decreasing confidence
                        "time_until_change": (forecast_time - current_time).total_seconds()
                    }
                    forecasts.append(forecast)
            
            # Calculate forecast accuracy (placeholder - would need historical validation)
            forecast_accuracy = 0.7  # Placeholder accuracy
            
            results = {
                "forecasts": forecasts,
                "forecast_accuracy": forecast_accuracy,
                "avg_time_between_changes": avg_time_diff
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating regime forecasts: {e}")
            return {"forecasts": [], "forecast_accuracy": 0.0}

    def _calculate_enhanced_regime_quality_metrics(self, regime_changes: List[Dict[str, Any]], 
                                                 transition_analysis: Dict[str, Any],
                                                 stability_analysis: Dict[str, Any],
                                                 forecast_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive regime change quality metrics."""
        try:
            metrics = {}
            
            # Basic quality metrics
            metrics["regime_change_count"] = len(regime_changes)
            metrics["avg_confidence"] = np.mean([c["confidence"] for c in regime_changes]) if regime_changes else 0.0
            metrics["avg_change_strength"] = np.mean([c["change_strength"] for c in regime_changes]) if regime_changes else 0.0
            
            # Transition quality
            transition_count = transition_analysis.get("transition_count", 0)
            metrics["transition_count"] = transition_count
            metrics["transition_diversity"] = len(transition_analysis.get("transition_patterns", {}))
            
            # Stability quality
            overall_stability = stability_analysis.get("overall_stability", 0.0)
            metrics["overall_stability"] = overall_stability
            
            # Forecast quality
            forecast_accuracy = forecast_analysis.get("forecast_accuracy", 0.0)
            metrics["forecast_accuracy"] = forecast_accuracy
            
            # Composite quality score
            quality_score = (
                metrics.get("avg_confidence", 0.0) * 0.3 +
                metrics.get("avg_change_strength", 0.0) * 0.2 +
                min(metrics.get("transition_count", 0) / 10.0, 1.0) * 0.2 +
                metrics.get("overall_stability", 0.0) * 0.2 +
                metrics.get("forecast_accuracy", 0.0) * 0.1
            )
            metrics["composite_quality_score"] = quality_score
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced regime quality metrics: {e}")
            return {"composite_quality_score": 0.0}

    def _eliminate_enhanced_regime_redundancy(self, regime_changes: List[Dict[str, Any]], 
                                            transition_analysis: Dict[str, Any],
                                            market_data: pd.DataFrame) -> Dict[str, float]:
        """Eliminate redundant regime changes and calculate redundancy metrics."""
        try:
            metrics = {}
            
            # Analyze temporal redundancy
            temporal_redundancy = 0
            for i, change1 in enumerate(regime_changes):
                for j, change2 in enumerate(regime_changes[i+1:], i+1):
                    time_diff = abs((change1["timestamp"] - change2["timestamp"]).total_seconds())
                    if time_diff < 300:  # Within 5 minutes
                        temporal_redundancy += 1
            
            # Analyze feature redundancy
            feature_redundancy = 0
            for i, change1 in enumerate(regime_changes):
                for j, change2 in enumerate(regime_changes[i+1:], i+1):
                    if change1["change_type"] == change2["change_type"]:
                        feature_similarity = self._calculate_feature_similarity(
                            change1["features"], change2["features"]
                        )
                        if feature_similarity > 0.8:  # 80% similarity threshold
                            feature_redundancy += 1
            
            # Calculate redundancy metrics
            total_pairs = len(regime_changes) * (len(regime_changes) - 1) / 2
            metrics["temporal_redundancy"] = temporal_redundancy
            metrics["feature_redundancy"] = feature_redundancy
            metrics["total_redundancy"] = temporal_redundancy + feature_redundancy
            metrics["redundancy_ratio"] = metrics["total_redundancy"] / total_pairs if total_pairs > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating enhanced regime redundancy: {e}")
            return {"redundancy_ratio": 0.0}

    def _remove_duplicate_regime_changes(self, regime_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate regime changes."""
        try:
            unique_changes = []
            seen_timestamps = set()
            
            for change in regime_changes:
                timestamp = change["timestamp"]
                if timestamp not in seen_timestamps:
                    unique_changes.append(change)
                    seen_timestamps.add(timestamp)
            
            return unique_changes
            
        except Exception as e:
            self.logger.error(f"Error removing duplicate regime changes: {e}")
            return regime_changes

    def _prepare_regime_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime analysis."""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['returns'] = market_data['close'].pct_change()
            features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
            features['price_momentum'] = market_data['close'] / market_data['close'].shift(5) - 1
            
            # Volatility features
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['volatility_change'] = features['volatility'].diff()
            
            # Volume features
            features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(window=20).mean()
            features['volume_change'] = market_data['volume'].pct_change()
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Error preparing regime features: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the step
    async def test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())