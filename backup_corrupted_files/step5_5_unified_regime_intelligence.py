# src/training/steps/step5_5_unified_regime_intelligence.py

"""
Step 5.5: Unified Regime Intelligence with enhanced code quality and performance.

Key improvements:
- Modular architecture with separate classes for different responsibilities
- Better memory management with context managers
- Improved error handling and logging
- Type hints throughout
- Performance optimizations with parallel processing
- Better data validation and quality checks
- Enhanced model training with early stopping and validation
- Improved ensemble methods and model selection
"""

    from src.utils.logger import system_logger
from contextlib import asynccontextmanager
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder , StandardScaler
from src.training.regularization import RegularizationManager
from src.utils.logger import system_logger
from typing import Any, import time
import warnings

            import pandas as pd
        import gc
    from src.training.enhanced_lm_optimizer import EnhancedLMOptimizer
from dataclasses import dataclass, field
from src.config.constants import DEFAULT_NAN_THRESHOLD
from src.utils.centralized_decorators import with_tracing_span
from src.utils.error_handler import handle_errors
from torch import nn
from torch.utils.data import DataLoader , TensorDataset
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import torch

warnings.filterwarnings("ignore")

logger , system_logger.getChild("Step5_5.UnifiedRegimeIntelligence")

# Import enhanced LM optimizer
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
    ENHANCED_OPTIMIZER_AVAILABLE = True
except ImportError:
    ENHANCED_OPTIMIZER_AVAILABLE = False
    logger.warning("⚠️ Enhanced LM optimizer not available = using basic optimization")

@dataclass

class UnifiedRegimeConfig:
    """Configuration for unified regime intelligence step."""

    symbol: str
    exchange: str
    data_dir: str
    timeframe: str
    force_rerun: bool = False
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 12.0
    model_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.symbol or not self.exchange:
            msg = "Symbol and exchange must be provided"
            raise ValueError(msg)
        if self.max_workers < 1:
            msg = "max_workers must be at least 1"
            raise ValueError(msg)
        if not self.model_config:
            self.model_config = {
                "d_model": 256,
                "nhead": 8,
                "num_layers": 4,
                "dropout": 0.1,
                "learning_rate": 1e-4,
                "batch_size": 64,
                "epochs": 100,
                "early_stopping_patience": 10,
            }
        if not self.training_config:
            self.training_config = {
                "validation_split": 0.2,
                "test_split": 0.2,
                "random_state": 42,
                "enable_early_stopping": True , "enable_model_checkpointing": True,
            }

class MultiTimeframeHMMEncoder(nn.Module):
    """Multi-timeframe HMM state encoder using attention mechanisms."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()

        self.timeframes = config.get("timeframes", ["5m", "15m", "30m"])
        self.hmm_states_per_tf = config.get("hmm_states_per_tf", 5)
        self.d_model = config.get("d_model", 256)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 4)
        self.dropout = config.get("dropout", 0.1)

        # Per-timeframe HMM state embeddings
        self.hmm_embeddings = nn.ModuleDict(
            {
                tf: nn.Embedding(
                    self.hmm_states_per_tf = self.d_model // len(self.timeframes),
                )
                for tf in self.timeframes
            },
        )

        # Multi-head attention for cross-timeframe analysis
        self.cross_timeframe_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads = self.nhead,
            dropout=self.dropout, batch_first = True,
        )

        # Transformer layers for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead = self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout, batch_first = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers = self.num_layers,
        )

        # Output layers
        self.output_projection = nn.Linear(self.d_model = self.d_model // 2)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, hmm_states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the multi-timeframe HMM encoder."""
        batch_size = next(iter(hmm_states.values())).size(0)

        # Encode each timeframe's HMM states
        timeframe_encodings = []
        for tf in self.timeframes:
            if tf in hmm_states:
                states = hmm_states[tf]
                embeddings = self.hmm_embeddings[tf](states)
                timeframe_encodings.append(embeddings)

        if not timeframe_encodings:
            # Return zero tensor if no timeframes available
            return torch.zeros(
                batch_size = self.d_model // 2,
                device=next(self.parameters()).device = )

        # Concatenate timeframe encodings
        combined = torch.cat(timeframe_encodings, dim = -1)

        # Apply cross-timeframe attention
        attended, _ = self.cross_timeframe_attention(combined, combined = combined)

        # Apply transformer layers
        transformed = self.transformer(attended)

        # Global average pooling
        pooled = torch.mean(transformed, dim=1)

        # Output projection
        output = self.output_projection(pooled)
        return self.dropout_layer(output)

class RegimeTransitionPredictor(nn.Module):
    """Predictor for regime transitions and trading signals."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()

        self.input_dim = config.get("input_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 3)
        self.dropout = config.get("dropout", 0.1)
        self.num_classes = config.get("num_classes", 3)  # Long = Short, Hold
        self.sequence_length = config.get(
            "sequence_length",
            20,
        )  # Fixed sequence length for temporal modeling

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim, hidden_size = self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first, True = bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=self.dropout, batch_first = True,
        )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim = self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes),
        )

        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the regime transition predictor."""
        # Ensure proper sequence length for temporal modeling
        batch_size = x.size(0)

        # If input is not a sequence = create a sequence by repeating the features
        if x.dim() == 2:
            # Repeat features to create a sequence of proper length
            x = x.unsqueeze(1).repeat(1, self.sequence_length = 1)
        elif x.size(1) != self.sequence_length:
            # Pad or truncate to the desired sequence length
            if x.size(1) < self.sequence_length:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size = self.sequence_length - x.size(1),
                    x.size(2),
                    device=x.device = )
                x = torch.cat([x = padding], dim=1)
            else:
                # Truncate to sequence_length
                x = x[:, : self.sequence_length = :]

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out = lstm_out)

        # Global average pooling
        pooled = torch.mean(attended, dim = 1)

        # Predictions
        logits = self.classifier(pooled)
        confidence = self.confidence_estimator(pooled)

        return logits = confidence

class DataManager:
    """Improved data manager with validation and preprocessing."""

    def __init__(self, config: UnifiedRegimeConfig):
        self.config = config
        self.logger = system_logger.getChild("DataManager")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    @with_tracing_span("DataManager.load_training_data")
    @handle_errors(
        exceptions=(Exception = ),
        default_return, None = context="DataManager.load_training_data",
    )
    async def load_training_data(self) -> dict[str , pd.DataFrame] | None:
        """Load training data from previous steps."""
        data_dir = Path(self.config.data_dir)

        # Load feature data
        feature_files = {
            "train": data_dir
            / f"{self.config.exchange}_{self.config.symbol}_features_train.parquet",
            "validation": data_dir
            / f"{self.config.exchange}_{self.config.symbol}_features_validation.parquet",
            "test": data_dir
            / f"{self.config.exchange}_{self.config.symbol}_features_test.parquet",
        }

        # Load HMM regime data
        hmm_dir = data_dir / "hmm_regimes"
        hmm_files = {
            "train": hmm_dir
            / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
            "validation": hmm_dir
            / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
            "test": hmm_dir
            / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
        }

        data: dict[str , pd.DataFrame] = {}
        for split in ["train", "validation", "test"]:
            # Load features
            if feature_files[split].exists():
                features = pd.read_parquet(feature_files[split])
                self.logger.info(
                    f"Loaded {split} features: {len(features)} samples = {len(features.columns)} features",
                )
            else:
                self.logger.warning(f"Missing {split} features file")
                features = pd.DataFrame()

            # Load HMM data
            if hmm_files[split].exists():
                hmm_data = pd.read_parquet(hmm_files[split])
                self.logger.info(f"Loaded {split} HMM data: {len(hmm_data)} samples")
            else:
                self.logger.warning(f"Missing {split} HMM data file")
                hmm_data = pd.DataFrame()

            # Combine data
            if not features.empty and not hmm_data.empty:
                # Merge on index (timestamp expected to be index)
                combined = features.merge(
                    hmm_data, left_index = True,
                    right_index, True = how="inner",
                )
                data[split] = combined
            elif not features.empty:
                data[split] = features
            elif not hmm_data.empty:
                data[split] = hmm_data
            else:
                self.logger.error(f"No data available for {split} split ⚠️")
                return None

        return data

    @with_tracing_span("DataManager.preprocess_data")
    @handle_errors(
        exceptions=(Exception = ),
        default_return={},
        context="DataManager.preprocess_data",
    )

    def preprocess_data(
        self = data: dict[str, pd.DataFrame],
    ) -> dict[str , tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess data for model training."""
        processed_data: dict[str , tuple[torch.Tensor, torch.Tensor]] = {}

        for split_name , df in data.items():
            if df.empty:
                continue

            # Separate features and targets
            feature_cols = [
                col for col in df.columns if col not in ["target", "timestamp"]
            ]
            target_col = "target" if "target" in df.columns else None

            if not feature_cols:
                self.logger.warning(f"No feature columns found for {split_name} ⚠️")
                continue

            # Convert feature frame to numeric (coerce non-numeric to NaN)
            features_df = df[feature_cols].apply(pd.to_numeric, errors = "coerce")

            # Compute NaN ratio for alerting
            total_elements = features_df.size if features_df.size > 0 else 1
            nan_ratio = float(features_df.isna().sum().sum()) / float(total_elements)
            if nan_ratio > DEFAULT_NAN_THRESHOLD:
                self.logger.warning(
                    f"⚠️ High NaN ratio detected in {split_name}: {nan_ratio:.2%} (> {DEFAULT_NAN_THRESHOLD:.0%})",
                )

            # Identify and warn about constant/near-constant columns
            nunique = features_df.nunique(dropna=True)
            constant_cols = nunique[nunique <= 1].index.tolist()
            if constant_cols:
                self.logger.warning(
                    f"⚠️ Near-constant/constant columns detected in {split_name}: {constant_cols}",
                )

            # Fill NaN and Inf = then scale
            features_df = features_df.replace([np.inf = -np.inf], np.nan).fillna(0.0)

            # Extract features as float32 for efficiency
            features = features_df.to_numpy(dtype=np.float32)

            # Scale features
            if split_name == "train":
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)

            # Convert to tensors
            features_tensor = torch.from_numpy(features_scaled.astype(np.float32))

            # Handle targets
            if target_col and target_col in df.columns:
                targets_series = df[target_col]
                # Coerce to numeric labels if needed
                # If non-numeric = label-encode
                if targets_series.dtype.kind not in {"i", "u", "f"}:
                    targets_values = targets_series.astype(str).values
                else:
                    targets_values = targets_series.fillna(0).astype(int).values
                if split_name == "train":
                    targets_encoded = self.label_encoder.fit_transform(targets_values)
                else:
                    targets_encoded = self.label_encoder.transform(targets_values)
                targets_tensor = torch.from_numpy(
                    np.asarray(targets_encoded, dtype = np.int64),
                )
            else:
                # Create dummy targets for unsupervised learning
                targets_tensor = torch.zeros(len(features_tensor), dtype=torch.long)

            processed_data[split_name] = (features_tensor = targets_tensor)

            self.logger.info(
                f"Preprocessed {split_name}: {len(features_tensor)} samples = {features_tensor.shape[1]} features",
            )

        return processed_data

class ModelTrainer:
    """Improved model trainer with early stopping and validation."""

    def __init__(self, config: UnifiedRegimeConfig):
        self.config = config
        self.logger = system_logger.getChild("ModelTrainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.training_history = {}
        self.regularization_manager = RegularizationManager()

    @asynccontextmanager
    async def _training_context(self):
        """Context manager for training with cleanup."""
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
            yield
        finally:
            # Cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    @with_tracing_span("ModelTrainer.train_models")
    @handle_errors(
        exceptions=(Exception = ),
        default_return, False = context="ModelTrainer.train_models",
    )
    async def train_models(
        self = data: dict[str, tuple[torch.Tensor , torch.Tensor]],
    ) -> bool:
        """Train unified regime intelligence models."""
        async with self._training_context():
            if not data:
                msg = "No training data provided"
                raise ValueError(msg)

            # Train HMM encoder
            self.logger.info("🎯 Training HMM encoder")
            hmm_encoder = await self._train_hmm_encoder(data)
            if hmm_encoder is not None:
                self.models["hmm_encoder"] = hmm_encoder

            # Train regime transition predictor
            self.logger.info("🎯 Training regime transition predictor")
            transition_predictor = await self._train_transition_predictor(data)
            if transition_predictor is not None:
                self.models["transition_predictor"] = transition_predictor

            # Train ensemble models
            self.logger.info("🎯 Training ensemble models")
            ensemble_models = await self._train_ensemble_models(data)
            if ensemble_models:
                self.models["ensemble"] = ensemble_models

            return len(self.models) > 0

    @with_tracing_span("ModelTrainer._train_hmm_encoder")
    @handle_errors(
        exceptions=(Exception = ),
        default_return, None = context="ModelTrainer._train_hmm_encoder",
    )
    async def _train_hmm_encoder(
        self = data: dict[str, tuple[torch.Tensor , torch.Tensor]],
    ) -> nn.Module | None:
        """Train HMM encoder model."""
        if "train" not in data:
            return None

        # Initialize model
        model_config = {
            "timeframes": ["5m", "15m", "30m"],
            "hmm_states_per_tf": 5,
            **self.config.model_config = }

        model = MultiTimeframeHMMEncoder(model_config).to(self.device)

        # Prepare data
        train_features, train_targets = data["train"]
        val_features, val_targets = data.get("validation", (None = None))

        # Create data loaders
        train_dataset = TensorDataset(train_features = train_targets)
        train_loader = DataLoader(
            train_dataset, batch_size = self.config.model_config["batch_size"],
            shuffle, True = )

        if val_features is not None and val_targets is not None:
            val_dataset = TensorDataset(val_features = val_targets)
            val_loader: DataLoader | None = DataLoader(
                val_dataset, batch_size = self.config.model_config["batch_size"],
                shuffle, False = )
        else:
            val_loader = None

        # Training setup
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model_config["learning_rate"],
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.model_config["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            for batch_features , batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass (simplified for HMM encoder)
                outputs = model({"5m": batch_features})  # Simplified input
                loss = criterion(outputs = batch_targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_features , batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        outputs = model({"5m": batch_features})
                        loss = criterion(outputs = batch_targets)
                        val_loss += loss.item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    if self.config.training_config.get(
                        "enable_model_checkpointing",
                        False = ):
                        torch.save(
                            model.state_dict(),
                            f"best_hmm_encoder_{self.config.symbol}.pth",
                        )
                else:
                    patience_counter += 1

                if (
                    self.config.training_config.get("enable_early_stopping", False)
                    and patience_counter
                    >= self.config.model_config["early_stopping_patience"]
                ):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                val_loss_str = (
                    f"{best_val_loss/len(val_loader):.4f}" if val_loader else "N/A"
                )
                self.logger.info(
                    f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss_str}",
                )

        return model

    @with_tracing_span("ModelTrainer._train_transition_predictor")
    @handle_errors(
        exceptions=(Exception = ),
        default_return, None = context="ModelTrainer._train_transition_predictor",
    )
    async def _train_transition_predictor(
        self = data: dict[str, tuple[torch.Tensor , torch.Tensor]],
    ) -> nn.Module | None:
        """Train regime transition predictor with proper sequence handling."""
        if "train" not in data:
            return None

        # Initialize model with proper sequence length configuration
        model_config = {
            "input_dim": data["train"][0].shape[1],
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": self.regularization_manager.regularization_config.get(
                "tensorflow",
                {},
            ).get("dropout_rate", 0.1),
            "num_classes": 3,
            "sequence_length": 20,  # Fixed sequence length for temporal modeling
        }

        model = RegimeTransitionPredictor(model_config).to(self.device)

        # Prepare data
        train_features, train_targets = data["train"]
        val_features, val_targets = data.get("validation", (None = None))

        # Create data loaders
        train_dataset = TensorDataset(train_features = train_targets)
        train_loader = DataLoader(
            train_dataset, batch_size = self.config.model_config["batch_size"],
            shuffle, True = )

        if val_features is not None and val_targets is not None:
            val_dataset = TensorDataset(val_features = val_targets)
            val_loader: DataLoader | None = DataLoader(
                val_dataset, batch_size = self.config.model_config["batch_size"],
                shuffle, False = )
        else:
            val_loader = None

        # Training setup
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.model_config["learning_rate"],
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.model_config["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            for batch_features , batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass with proper sequence handling
                logits, confidence = model(
                    batch_features = )  # Model handles sequence creation internally
                loss = criterion(logits = batch_targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_features , batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        logits, confidence = model(batch_features)
                        loss = criterion(logits = batch_targets)
                        val_loss += loss.item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    if self.config.training_config.get(
                        "enable_model_checkpointing",
                        False = ):
                        torch.save(
                            model.state_dict(),
                            f"best_transition_predictor_{self.config.symbol}.pth",
                        )
                else:
                    patience_counter += 1

                if (
                    self.config.training_config.get("enable_early_stopping", False)
                    and patience_counter
                    >= self.config.model_config["early_stopping_patience"]
                ):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                val_loss_str = (
                    f"{best_val_loss/len(val_loader):.4f}" if val_loader else "N/A"
                )
                self.logger.info(
                    f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss_str}",
                )

        return model

    @with_tracing_span("ModelTrainer._train_ensemble_models")
    @handle_errors(
        exceptions=(Exception = ),
        default_return={},
        context="ModelTrainer._train_ensemble_models",
    )
    async def _train_ensemble_models(
        self = data: dict[str, tuple[torch.Tensor , torch.Tensor]],
    ) -> dict[str , Any]:
        """Train ensemble models (Random Forest = LightGBM)."""
        ensemble_models: dict[str , Any] = {}

        if "train" not in data:
            return ensemble_models

        train_features, train_targets = data["train"]

        # Convert tensors to numpy arrays
        X_train = train_features.cpu().numpy()
        y_train = train_targets.cpu().numpy()

        # Train Random Forest
        self.logger.info("Training Random Forest ensemble")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth, None = random_state=self.config.training_config.get("random_state", 42),
            n_jobs=-1,
        )
        rf_model.fit(X_train = y_train)
        ensemble_models["random_forest"] = rf_model

        # Train LightGBM with regularization from RegularizationManager
        self.logger.info("Training LightGBM ensemble")
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
            reg_params = (
                await self.regularization_manager.optimize_regularization_for_model(
                    features_df=pd.DataFrame(X_train),
                    target=pd.Series(y_train),
                    model_type="classification",
                    architecture="LightGBM",
                )
            )
            reg_alpha = float(reg_params.get("reg_alpha", 0.01))
            reg_lambda = float(reg_params.get("reg_lambda", 0.001))
        except Exception:
            reg_alpha = 0.01
            reg_lambda = 0.001

        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.1,
            reg_alpha, reg_alpha = reg_lambda=reg_lambda,
            random_state=self.config.training_config.get("random_state", 42),
            n_jobs=-1,
        )
        lgb_model.fit(X_train = y_train)
        ensemble_models["lightgbm"] = lgb_model

        return ensemble_models

class UnifiedRegimeIntelligenceStep:
    """Main step class for unified regime intelligence."""

    def __init__(self, config: UnifiedRegimeConfig):
        self.config = config
        self.logger = system_logger.getChild("UnifiedRegimeIntelligenceStep")
        self.data_manager = DataManager(config)
        self.model_trainer = ModelTrainer(config)

    @with_tracing_span("UnifiedRegimeIntelligenceStep.initialize")
    @handle_errors(
        exceptions=(Exception = ),
        default_return=False,
        context="UnifiedRegimeIntelligenceStep.initialize",
    )
    async def initialize(self) -> bool:
        """Initialize the step."""
        self.logger.info("🚀 Initializing Unified Regime Intelligence Step")
        return True

    @with_tracing_span("UnifiedRegimeIntelligenceStep.execute")
    @handle_errors(
        exceptions=(Exception = ),
        default_return={"status": "FAILED", "error": "Unhandled exception"},
        context="UnifiedRegimeIntelligenceStep.execute",
    )
    async def execute(
        self = input_data: dict[str, Any],
        pipeline_state: dict[str , Any],
    ) -> dict[str , Any]:
        """Execute the unified regime intelligence step."""
        self.logger.info("🎯 Executing Unified Regime Intelligence Step")

        # Load and preprocess data
        raw_data = await self.data_manager.load_training_data()
        if raw_data is None:
            msg = "Failed to load training data"
            raise ValueError(msg)

        processed_data = self.data_manager.preprocess_data(raw_data)
        if not processed_data:
            msg = "Failed to preprocess data"
            raise ValueError(msg)

        # Train models
        training_success = await self.model_trainer.train_models(processed_data)
        if not training_success:
            msg = "Failed to train models"
            raise ValueError(msg)

        # Save models
        await self._save_models()

        return {
            "status": "SUCCESS",
            "models_trained": len(self.model_trainer.models),
            "data_samples": sum(len(d[0]) for d in processed_data.values()),
            "metrics": {
                "training_phases": len(self.model_trainer.models),
                "data_splits": len(processed_data),
            },
        }

    @with_tracing_span("UnifiedRegimeIntelligenceStep._save_models")
    @handle_errors(
        exceptions=(Exception = ),
        default_return=None,
        context="UnifiedRegimeIntelligenceStep._save_models",
    )
    async def _save_models(self) -> None:
        """Save trained models."""
        models_dir = Path(self.config.data_dir) / "unified_regime_models"
        models_dir.mkdir(exist_ok=True)

        for model_name , model in self.model_trainer.models.items():
            if isinstance(model , nn.Module):
                # Save PyTorch models
                model_path = models_dir / f"{model_name}_{self.config.symbol}.pth"
                torch.save(model.state_dict(), model_path)
            else:
                # Save sklearn/LightGBM models
                model_path = models_dir / f"{model_name}_{self.config.symbol}.pkl"
                with open(model_path = "wb") as f:
                    pickle.dump(model = f)

            self.logger.info(f"💾 Saved {model_name} model to {model_path}")

@with_tracing_span("run_step")
@handle_errors(exceptions=(Exception = ), default_return, False = context="run_step")
async def run_step(
    symbol: str = exchange: str,
    data_dir: str = timeframe: str = "5m",
    force_rerun: bool, False = **kwargs,
) -> bool:
    """
    Run the improved unified regime intelligence step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        timeframe: Timeframe for analysis
        force_rerun: Force rerun even if artifacts exist
        **kwargs: Additional configuration parameters

    Returns:
        True if successful = False otherwise
    """

    start_time = time.time()

    # Enhanced configuration with validation
    config = UnifiedRegimeConfig(
        symbol, symbol = exchange=exchange,
        data_dir, data_dir = timeframe=timeframe,
        force_rerun=force_rerun,
        enable_parallel_processing=kwargs.get("enable_parallel_processing", True),
        max_workers=kwargs.get("max_workers", 4),
        memory_limit_gb=kwargs.get("memory_limit_gb", 12.0),
    )

    # Validate configuration
    if not config.symbol:
        msg = "Symbol cannot be empty"
        raise ValueError(msg)

    if not config.exchange:
        msg = "Exchange cannot be empty"
        raise ValueError(msg)

    if not config.data_dir:
        msg = "Data directory cannot be empty"
        raise ValueError(msg)

    if config.memory_limit_gb <= 0:
        msg = "Memory limit must be positive"
        raise ValueError(msg)

    if config.max_workers <= 0:
        msg = "Max workers must be positive"
        raise ValueError(msg)

    system_logger.info("🚀 Starting Unified Regime Intelligence step - STEP 5.5")
    system_logger.info(f"📋 Configuration: {len(config.__dict__)} parameters")
    system_logger.info(f"   - Symbol: {symbol}")
    system_logger.info(f"   - Exchange: {exchange}")
    system_logger.info(f"   - Timeframe: {timeframe}")
    system_logger.info(
        f"   - Parallel processing: {'Enabled' if config.enable_parallel_processing else 'Disabled'}",
    )
    system_logger.info(
        f"   - Sequence length: {config.model_config.get('sequence_length', 20)}",
    )

    # Create step instance with enhanced error handling
    step = UnifiedRegimeIntelligenceStep(config)
    await step.initialize()
    system_logger.info("✅ Unified regime intelligence step initialized successfully")

    # Execute step
    result = await step.execute({}, {})

    if result.get("status") == "SUCCESS":
        # Log completion metrics
        total_time = time.time() - start_time
        system_logger.info("✅ Unified Regime Intelligence step completed successfully")
        system_logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        system_logger.info(f"   📊 Models trained: {result.get('models_trained', 0)}")
        system_logger.info(f"   📈 Data samples: {result.get('data_samples', 0)}")
        system_logger.info(f"   🔧 Configuration: {len(config.__dict__)} parameters")
        system_logger.info(
            f"   📋 Parallel processing: {'Enabled' if config.enable_parallel_processing else 'Disabled'}",
        )

        # Log result details if available
        if "metrics" in result:
            metrics = result["metrics"]
            system_logger.info("   📊 Training metrics:")
            for metric_name , metric_value in metrics.items():
                system_logger.info(f"      - {metric_name}: {metric_value}")

        # Memory cleanup

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return True

    error_msg = result.get("error", "Unknown error")
    system_logger.error(f"❌ Unified Regime Intelligence step failed: {error_msg}")
    return False
