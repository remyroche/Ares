# src/training/steps/step9_5_hmm_lm_generalist_training.py

import asyncio
import concurrent.futures
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, failed, success
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span

# Suppress warnings
warnings.filterwarnings("ignore")

logger = system_logger.getChild("Step9_5_HMM_LM_Generalist")


class HMMLMGeneralistTrainingStep:
    """Step 9.5: Generalist HMM-LM Model Training for Regime Change Prediction."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}

        # HMM-LM configuration
        hmm_lm_config = config.get("HMM_LM", {})
        generalist_config = hmm_lm_config.get("generalist", {})

        self.hmm_states = generalist_config.get("hmm_states", 5)
        self.sequence_length = generalist_config.get("sequence_length", 20)
        self.timeframes = generalist_config.get(
            "timeframes", ["1m", "5m", "15m", "30m"]
        )
        self.d_model = generalist_config.get("d_model", 256)
        self.nhead = generalist_config.get("nhead", 8)
        self.num_layers = generalist_config.get("num_layers", 6)
        self.dropout_rate = generalist_config.get("dropout_rate", 0.1)
        self.learning_rate = generalist_config.get("learning_rate", 0.0001)
        self.batch_size = generalist_config.get("batch_size", 32)
        self.epochs = generalist_config.get("epochs", 100)

        # Regime change vocabulary
        self.regime_change_vocab = self._create_regime_change_vocabulary()

    def _create_regime_change_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary for regime change events."""
        vocab = {}
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
    async def initialize(self) -> None:
        """Initialize the HMM-LM generalist training step."""
        self.logger.info("Initializing HMM-LM Generalist Training Step...")
        self.logger.info("HMM-LM Generalist Training Step initialized successfully")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="HMM-LM generalist training step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute HMM-LM generalist model training.

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
                exchange, symbol, data_dir
            )
            if not hmm_data:
                raise ValueError("Failed to load multi-timeframe HMM data")

            # Create regime change sequences
            regime_sequences = await self._create_regime_change_sequences(hmm_data)
            if not regime_sequences:
                raise ValueError("Failed to create regime change sequences")

            # Train HMM-LM model
            model_result = await self._train_hmm_lm_model(regime_sequences)
            if not model_result:
                raise ValueError("Failed to train HMM-LM model")

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

        except Exception as e:
            self.logger.error(f"❌ HMM-LM Generalist Training failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    async def _load_multi_timeframe_hmm_data(
        self, exchange: str, symbol: str, data_dir: str
    ) -> Dict[str, pd.DataFrame]:
        """Load HMM data from all timeframes in parallel."""
        hmm_data = {}

        async def load_timeframe_data(
            timeframe: str,
        ) -> Tuple[str, Optional[pd.DataFrame]]:
            try:
                # Load cluster assignments
                cluster_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
                if not os.path.exists(cluster_path):
                    return timeframe, None

                # Use ThreadPoolExecutor for I/O operations
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    clusters_df = await loop.run_in_executor(
                        executor, pd.read_parquet, cluster_path
                    )

                clusters_df["timestamp"] = pd.to_datetime(clusters_df["timestamp"])
                clusters_df = clusters_df.set_index("timestamp")

                # Load intensity scores
                intensity_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
                if os.path.exists(intensity_path):
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
                        f"✅ Loaded HMM data for {timeframe}: {hmm_df.shape}"
                    )
                    return timeframe, hmm_df
                else:
                    clusters_df["timeframe"] = timeframe
                    self.logger.info(
                        f"✅ Loaded HMM clusters for {timeframe}: {clusters_df.shape}"
                    )
                    return timeframe, clusters_df

            except Exception as e:
                self.logger.error(f"❌ Failed to load HMM data for {timeframe}: {e}")
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
        self, hmm_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Create regime change sequences for training."""
        sequences = []

        try:
            # Combine all timeframe data
            all_data = []
            for timeframe, df in hmm_data.items():
                if not df.empty:
                    all_data.append(df)

            if not all_data:
                return []

            combined_df = pd.concat(all_data, axis=0).sort_index()

            # Detect regime changes and TPSL outcomes
            regime_events = self._detect_regime_changes_and_tpsl_outcomes(combined_df)

            # Create sequences around regime changes
            for change_idx, event_data in enumerate(regime_events):
                if change_idx < self.sequence_length:
                    continue

                # Get sequence before the change
                start_idx = change_idx - self.sequence_length
                end_idx = change_idx

                if start_idx >= 0 and end_idx < len(combined_df):
                    sequence_data = combined_df.iloc[start_idx:end_idx]

                    sequences.append(
                        {
                            "sequence": sequence_data,
                            "target": event_data["regime_change"],
                            "price_direction": event_data["price_direction"],
                            "price_magnitude": event_data["price_magnitude"],
                            "volatility_change": event_data["volatility_change"],
                            "timing": event_data["timing"],
                            "timestamp": combined_df.index[end_idx],
                            "timeframe": combined_df.iloc[end_idx]["timeframe"],
                        }
                    )

            self.logger.info(f"✅ Created {len(sequences)} regime change sequences")
            return sequences

        except Exception as e:
            self.logger.error(f"❌ Failed to create regime change sequences: {e}")
            return []

    def _detect_regime_changes_and_tpsl_outcomes(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect regime changes and associated TPSL outcomes."""
        events = []

        try:
            # Get TPSL parameters from config
            tpsl_config = self.config.get("vectorized_labelling_orchestrator", {})
            profit_take_multiplier = tpsl_config.get(
                "profit_take_multiplier", 0.002
            )  # 0.2%
            stop_loss_multiplier = tpsl_config.get(
                "stop_loss_multiplier", 0.001
            )  # 0.1%
            time_barrier_minutes = tpsl_config.get("time_barrier_minutes", 30)

            # Get regime column
            regime_col = "composite_cluster_id"
            if regime_col not in df.columns:
                self.logger.warning(f"⚠️ No regime column {regime_col} found")
                return events

            # Detect state transitions
            regimes = df[regime_col].fillna(-1).astype(int)

            for i in range(1, len(regimes)):
                prev_regime = regimes.iloc[i - 1]
                curr_regime = regimes.iloc[i]

                event = {
                    "regime_change": "<PAD>",
                    "price_direction": 1,  # Sideways
                    "profit_target_hit": 0,  # 0/1
                    "stop_loss_hit": 0,  # 0/1
                    "time_to_target": 0,  # bars to hit target
                }

                if prev_regime != curr_regime and prev_regime >= 0 and curr_regime >= 0:
                    # Exit previous regime
                    if prev_regime < self.hmm_states:
                        event["regime_change"] = f"exit_regime_{prev_regime}"

                    # Enter new regime
                    if curr_regime < self.hmm_states:
                        event["regime_change"] = f"enter_regime_{curr_regime}"

                    # Calculate TPSL outcomes for regime change
                    if "close" in df.columns and i < len(df) - 1:
                        current_price = df.iloc[i]["close"]
                        future_prices = df.iloc[i + 1 : i + 31][
                            "close"
                        ]  # Look ahead 30 bars

                        if len(future_prices) > 0:
                            # Calculate profit target and stop loss levels
                            profit_target = current_price * (1 + profit_take_multiplier)
                            stop_loss = current_price * (1 - stop_loss_multiplier)

                            # Check if profit target or stop loss is hit
                            profit_target_hit = 0
                            stop_loss_hit = 0
                            time_to_target = 0

                            for j, future_price in enumerate(future_prices):
                                if (
                                    future_price >= profit_target
                                    and profit_target_hit == 0
                                ):
                                    profit_target_hit = 1
                                    time_to_target = j + 1
                                elif future_price <= stop_loss and stop_loss_hit == 0:
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

        except Exception as e:
            self.logger.error(
                f"❌ Failed to detect regime changes and price action: {e}"
            )
            return []

    async def _train_hmm_lm_model(
        self, sequences: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Train the HMM-LM model."""
        try:
            self.logger.info(
                f"🔄 Training HMM-LM model with {len(sequences)} sequences"
            )

            if len(sequences) < 100:
                self.logger.warning(
                    f"⚠️ Insufficient sequences for training: {len(sequences)}"
                )
                return None

            # Prepare training data
            X_train, y_train, X_val, y_val = self._prepare_regime_training_data(
                sequences
            )

            # Create efficient regime predictor
            input_dim = X_train.shape[2] if len(X_train.shape) > 2 else 10
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

        except Exception as e:
            self.logger.error(f"❌ HMM-LM training failed: {e}")
            return None

    def _prepare_regime_training_data(
        self, sequences: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for regime prediction."""
        try:
            # Convert sequences to tensor format
            X_data = []
            y_data = []

            for seq_data in sequences:
                sequence = seq_data["sequence"]
                target = seq_data["target"]

                # Convert sequence to feature tensor
                features = self._sequence_to_features(sequence)
                X_data.append(features)

                # Extract regime ID from target (e.g., "enter_regime_2" -> 2)
                if "enter_regime_" in target:
                    regime_id = int(target.split("_")[-1])
                elif "exit_regime_" in target:
                    regime_id = int(target.split("_")[-1])
                else:
                    regime_id = 0  # Default regime

                # Extract TPSL outcomes
                profit_target_hit = seq_data.get("profit_target_hit", 0)
                stop_loss_hit = seq_data.get("stop_loss_hit", 0)
                time_to_target = seq_data.get("time_to_target", 0)

                y_data.append(
                    {
                        "regime_id": regime_id,
                        "profit_target_hit": profit_target_hit,
                        "stop_loss_hit": stop_loss_hit,
                        "time_to_target": time_to_target,
                    }
                )

            X = np.array(X_data)
            y = np.array(y_data)

            # Split data with time series split
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            return X_train, y_train, X_val, y_val

        except Exception as e:
            self.logger.error(f"❌ Failed to prepare regime training data: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])

    def _sequence_to_features(self, sequence: pd.DataFrame) -> np.ndarray:
        """Convert sequence to feature tensor."""
        try:
            # Extract HMM-related features
            feature_cols = []

            # Add cluster ID
            if "composite_cluster_id" in sequence.columns:
                feature_cols.append(sequence["composite_cluster_id"].values)

            # Add intensity features
            intensity_cols = [
                col for col in sequence.columns if col.startswith("intensity_cluster_")
            ]
            for col in intensity_cols:
                feature_cols.append(sequence[col].values)

            # Add regime probability features
            regime_cols = [col for col in sequence.columns if col.endswith("_p_state_")]
            for col in regime_cols:
                feature_cols.append(sequence[col].values)

            # Stack features
            if feature_cols:
                features = np.column_stack(feature_cols)
            else:
                # Fallback: use basic features
                features = np.zeros((len(sequence), 10))

            # Pad or truncate to sequence length
            if len(features) < self.sequence_length:
                # Pad with zeros
                padding = np.zeros(
                    (self.sequence_length - len(features), features.shape[1])
                )
                features = np.vstack([padding, features])
            elif len(features) > self.sequence_length:
                # Truncate
                features = features[-self.sequence_length :]

            return features

        except Exception as e:
            self.logger.error(f"❌ Failed to convert sequence to features: {e}")
            return np.zeros((self.sequence_length, 10))

    async def _save_generalist_model(
        self, model_result: Dict[str, Any], exchange: str, symbol: str, data_dir: str
    ) -> None:
        """Save the generalist model and metadata."""
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)

            # Save metadata
            metadata = {
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

            metadata_path = (
                f"{data_dir}/{exchange}_{symbol}_hmm_lm_generalist_metadata.json"
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.info(f"✅ Saved generalist model metadata to {metadata_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save generalist model: {e}")


# Efficient Regime Prediction Architecture


class EfficientRegimePredictor(nn.Module):
    """Efficient regime prediction model for financial time series."""

    def __init__(
        self,
        input_dim: int,
        num_regimes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
    ):
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

    def forward(self, x):
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
        profit_target_prob = torch.sigmoid(
            self.profit_target_prob(final_features)
        )  # 0-1 probability
        stop_loss_prob = torch.sigmoid(
            self.stop_loss_prob(final_features)
        )  # 0-1 probability
        time_to_target = (
            torch.sigmoid(self.time_to_target(final_features)) * 30
        )  # 0-30 bars

        return {
            "current_regime": current_regime_probs,
            "transition_matrix": transition_probs,
            "confidence": regime_confidence,
            "price_direction": price_direction_probs,  # Up/Down/Sideways probabilities
            "profit_target_prob": profit_target_prob,  # Probability of hitting profit target
            "stop_loss_prob": stop_loss_prob,  # Probability of hitting stop loss
            "time_to_target": time_to_target,  # Expected time to hit target (bars)
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class EfficientRegimeTrainer:
    """Efficient trainer for regime prediction model."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.0001, batch_size: int = 32
    ):
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
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """Train the regime prediction model efficiently."""

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Early stopping
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                # Mixed precision training
                if self.scaler:
                    with torch.cuda.amp.autocast():
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

                train_loss += loss.item()
                train_correct += self._compute_accuracy(outputs, batch_y)
                train_total += batch_y.size(0)

            # Validation
            val_loss, val_acc = self._validate(X_val, y_val)

            # Record metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(train_loss_avg)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "models/best_regime_predictor.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Learning rate scheduling
            self.scheduler.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        return history

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
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
            confidence, torch.ones_like(confidence)
        )

        # TPSL prediction losses
        profit_target_loss = F.binary_cross_entropy(
            outputs["profit_target_prob"], targets["profit_target_hit"].float()
        )
        stop_loss_loss = F.binary_cross_entropy(
            outputs["stop_loss_prob"], targets["stop_loss_hit"].float()
        )
        time_to_target_loss = F.mse_loss(
            outputs["time_to_target"], targets["time_to_target"].float()
        )

        # Combined loss with TPSL weighting
        total_loss = (
            regime_loss
            + 0.1 * transition_regularization
            + 0.05 * confidence_loss
            + 0.3 * profit_target_loss  # Higher weight for profit target prediction
            + 0.2 * stop_loss_loss  # Medium weight for stop loss prediction
            + 0.1 * time_to_target_loss  # Lower weight for timing prediction
        )
        return total_loss

    def _compute_accuracy(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> int:
        """Compute accuracy for regime and TPSL prediction."""
        # Regime accuracy
        predicted_regime = torch.argmax(outputs["current_regime"], dim=1)
        regime_correct = (predicted_regime == targets["regime_id"]).sum().item()

        # TPSL accuracy (profit target prediction)
        predicted_profit = (outputs["profit_target_prob"] > 0.5).float()
        profit_correct = (
            (predicted_profit == targets["profit_target_hit"].float()).sum().item()
        )

        # Combined accuracy (weighted)
        total_correct = regime_correct + profit_correct
        return total_correct

    def _validate(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self._compute_loss(outputs, y_val).item()
            accuracy = self._compute_accuracy(outputs, y_val) / y_val.size(0)
        return loss, accuracy


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the HMM-LM generalist training step.

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

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception as e:
        logger.error(f"HMM-LM generalist training failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
