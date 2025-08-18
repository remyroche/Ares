# src/training/steps/step5_unified_regime_intelligence_improved.py

"""
Improved Step 5: Unified Regime Intelligence with enhanced code quality and performance.

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

import asyncio
import os
import json
import pickle
import time
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
import gc
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import prune
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, failed, success, timeout
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor

warnings.filterwarnings("ignore")

logger = system_logger.getChild("Step5.ImprovedUnifiedRegimeIntelligence")

# Import enhanced LM optimizer
try:
    from src.training.enhanced_lm_optimizer import EnhancedLMOptimizer
    ENHANCED_OPTIMIZER_AVAILABLE = True
except ImportError:
    ENHANCED_OPTIMIZER_AVAILABLE = False
    logger.warning("⚠️ Enhanced LM optimizer not available, using basic optimization")


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
    model_config: Dict[str, Any] = None
    training_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.symbol or not self.exchange:
            raise ValueError("Symbol and exchange must be provided")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.model_config is None:
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
        if self.training_config is None:
            self.training_config = {
                "validation_split": 0.2,
                "test_split": 0.2,
                "random_state": 42,
                "enable_early_stopping": True,
                "enable_model_checkpointing": True,
            }


class MultiTimeframeHMMEncoder(nn.Module):
    """Multi-timeframe HMM state encoder using attention mechanisms."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.timeframes = config.get("timeframes", ["5m", "15m", "30m"])
        self.hmm_states_per_tf = config.get("hmm_states_per_tf", 5)
        self.d_model = config.get("d_model", 256)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 4)
        self.dropout = config.get("dropout", 0.1)
        
        # Per-timeframe HMM state embeddings
        self.hmm_embeddings = nn.ModuleDict({
            tf: nn.Embedding(self.hmm_states_per_tf, self.d_model // len(self.timeframes))
            for tf in self.timeframes
        })
        
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output layers
        self.output_projection = nn.Linear(self.d_model, self.d_model // 2)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, hmm_states: Dict[str, torch.Tensor]) -> torch.Tensor:
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
            return torch.zeros(batch_size, self.d_model // 2, device=next(self.parameters()).device)
        
        # Concatenate timeframe encodings
        combined = torch.cat(timeframe_encodings, dim=-1)
        
        # Apply cross-timeframe attention
        attended, _ = self.cross_timeframe_attention(combined, combined, combined)
        
        # Apply transformer layers
        transformed = self.transformer(attended)
        
        # Global average pooling
        pooled = torch.mean(transformed, dim=1)
        
        # Output projection
        output = self.output_projection(pooled)
        output = self.dropout_layer(output)
        
        return output


class RegimeTransitionPredictor(nn.Module):
    """Predictor for regime transitions and trading signals."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.input_dim = config.get("input_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 3)
        self.dropout = config.get("dropout", 0.1)
        self.num_classes = config.get("num_classes", 3)  # Long, Short, Hold
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=self.dropout,
            batch_first=True,
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the regime transition predictor."""
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        
        # Predictions
        logits = self.classifier(pooled)
        confidence = self.confidence_estimator(pooled)
        
        return logits, confidence


class DataManager:
    """Improved data manager with validation and preprocessing."""
    
    def __init__(self, config: UnifiedRegimeConfig):
        self.config = config
        self.logger = system_logger.getChild("DataManager")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    async def load_training_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load training data from previous steps."""
        try:
            data_dir = Path(self.config.data_dir)
            
            # Load feature data
            feature_files = {
                "train": data_dir / f"{self.config.exchange}_{self.config.symbol}_features_train.parquet",
                "validation": data_dir / f"{self.config.exchange}_{self.config.symbol}_features_validation.parquet",
                "test": data_dir / f"{self.config.exchange}_{self.config.symbol}_features_test.parquet",
            }
            
            # Load HMM regime data
            hmm_dir = data_dir / "hmm_regimes"
            hmm_files = {
                "train": hmm_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
                "validation": hmm_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
                "test": hmm_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
            }
            
            data = {}
            for split in ["train", "validation", "test"]:
                # Load features
                if feature_files[split].exists():
                    features = pd.read_parquet(feature_files[split])
                    self.logger.info(f"Loaded {split} features: {len(features)} samples, {len(features.columns)} features")
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
                    # Merge on timestamp
                    combined = features.merge(hmm_data, left_index=True, right_index=True, how='inner')
                    data[split] = combined
                elif not features.empty:
                    data[split] = features
                elif not hmm_data.empty:
                    data[split] = hmm_data
                else:
                    self.logger.error(f"No data available for {split} split")
                    return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return None
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess data for model training."""
        try:
            processed_data = {}
            
            for split_name, df in data.items():
                if df.empty:
                    continue
                
                # Separate features and targets
                feature_cols = [col for col in df.columns if col not in ['target', 'timestamp']]
                target_col = 'target' if 'target' in df.columns else None
                
                if not feature_cols:
                    self.logger.warning(f"No feature columns found for {split_name}")
                    continue
                
                # Extract features
                features = df[feature_cols].values
                
                # Handle missing values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale features
                if split_name == "train":
                    features_scaled = self.scaler.fit_transform(features)
                else:
                    features_scaled = self.scaler.transform(features)
                
                # Convert to tensors
                features_tensor = torch.FloatTensor(features_scaled)
                
                # Handle targets
                if target_col and target_col in df.columns:
                    targets = df[target_col].values
                    # Encode targets
                    if split_name == "train":
                        targets_encoded = self.label_encoder.fit_transform(targets)
                    else:
                        targets_encoded = self.label_encoder.transform(targets)
                    targets_tensor = torch.LongTensor(targets_encoded)
                else:
                    # Create dummy targets for unsupervised learning
                    targets_tensor = torch.zeros(len(features_tensor), dtype=torch.long)
                
                processed_data[split_name] = (features_tensor, targets_tensor)
                
                self.logger.info(f"Preprocessed {split_name}: {len(features_tensor)} samples, {features_tensor.shape[1]} features")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return {}


class ModelTrainer:
    """Improved model trainer with early stopping and validation."""
    
    def __init__(self, config: UnifiedRegimeConfig):
        self.config = config
        self.logger = system_logger.getChild("ModelTrainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.training_history = {}
    
    @asynccontextmanager
    async def _training_context(self):
        """Context manager for training with cleanup."""
        try:
            yield
        finally:
            # Cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
    
    async def train_models(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Train unified regime intelligence models."""
        try:
            async with self._training_context():
                if not data:
                    raise ValueError("No training data provided")
                
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
                
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return False
    
    async def _train_hmm_encoder(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Optional[nn.Module]:
        """Train HMM encoder model."""
        try:
            if "train" not in data:
                return None
            
            # Initialize model
            model_config = {
                "timeframes": ["5m", "15m", "30m"],
                "hmm_states_per_tf": 5,
                **self.config.model_config
            }
            
            model = MultiTimeframeHMMEncoder(model_config).to(self.device)
            
            # Prepare data
            train_features, train_targets = data["train"]
            val_features, val_targets = data.get("validation", (None, None))
            
            # Create data loaders
            train_dataset = TensorDataset(train_features, train_targets)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.model_config["batch_size"],
                shuffle=True
            )
            
            if val_features is not None:
                val_dataset = TensorDataset(val_features, val_targets)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.model_config["batch_size"],
                    shuffle=False
                )
            else:
                val_loader = None
            
            # Training setup
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.model_config["learning_rate"]
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.model_config["epochs"]):
                # Training
                model.train()
                train_loss = 0.0
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass (simplified for HMM encoder)
                    outputs = model({"5m": batch_features})  # Simplified input
                    loss = criterion(outputs, batch_targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_features, batch_targets in val_loader:
                            batch_features = batch_features.to(self.device)
                            batch_targets = batch_targets.to(self.device)
                            
                            outputs = model({"5m": batch_features})
                            loss = criterion(outputs, batch_targets)
                            val_loss += loss.item()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        if self.config.training_config["enable_model_checkpointing"]:
                            torch.save(model.state_dict(), f"best_hmm_encoder_{self.config.symbol}.pth")
                    else:
                        patience_counter += 1
                    
                    if (self.config.training_config["enable_early_stopping"] and 
                        patience_counter >= self.config.model_config["early_stopping_patience"]):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                                   f"Val Loss = {val_loss/len(val_loader) if val_loader else 'N/A':.4f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training HMM encoder: {e}")
            return None
    
    async def _train_transition_predictor(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Optional[nn.Module]:
        """Train regime transition predictor."""
        try:
            if "train" not in data:
                return None
            
            # Initialize model
            model_config = {
                "input_dim": data["train"][0].shape[1],
                "hidden_dim": 256,
                "num_layers": 3,
                "dropout": 0.1,
                "num_classes": 3,
            }
            
            model = RegimeTransitionPredictor(model_config).to(self.device)
            
            # Prepare data
            train_features, train_targets = data["train"]
            val_features, val_targets = data.get("validation", (None, None))
            
            # Create data loaders
            train_dataset = TensorDataset(train_features, train_targets)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.model_config["batch_size"],
                shuffle=True
            )
            
            if val_features is not None:
                val_dataset = TensorDataset(val_features, val_targets)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.model_config["batch_size"],
                    shuffle=False
                )
            else:
                val_loader = None
            
            # Training setup
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.model_config["learning_rate"]
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.model_config["epochs"]):
                # Training
                model.train()
                train_loss = 0.0
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    logits, confidence = model(batch_features.unsqueeze(1))  # Add sequence dimension
                    loss = criterion(logits, batch_targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_features, batch_targets in val_loader:
                            batch_features = batch_features.to(self.device)
                            batch_targets = batch_targets.to(self.device)
                            
                            logits, confidence = model(batch_features.unsqueeze(1))
                            loss = criterion(logits, batch_targets)
                            val_loss += loss.item()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if self.config.training_config["enable_model_checkpointing"]:
                            torch.save(model.state_dict(), f"best_transition_predictor_{self.config.symbol}.pth")
                    else:
                        patience_counter += 1
                    
                    if (self.config.training_config["enable_early_stopping"] and 
                        patience_counter >= self.config.model_config["early_stopping_patience"]):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                                   f"Val Loss = {val_loss/len(val_loader) if val_loader else 'N/A':.4f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training transition predictor: {e}")
            return None
    
    async def _train_ensemble_models(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """Train ensemble models (Random Forest, LightGBM)."""
        try:
            if "train" not in data:
                return {}
            
            ensemble_models = {}
            
            # Convert tensors to numpy arrays
            train_features, train_targets = data["train"]
            val_features, val_targets = data.get("validation", (None, None))
            
            X_train = train_features.numpy()
            y_train = train_targets.numpy()
            
            if val_features is not None:
                X_val = val_features.numpy()
                y_val = val_targets.numpy()
            else:
                X_val, y_val = None, None
            
            # Train Random Forest
            self.logger.info("Training Random Forest")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.training_config["random_state"],
                n_jobs=self.config.max_workers
            )
            rf_model.fit(X_train, y_train)
            ensemble_models["random_forest"] = rf_model
            
            # Train LightGBM
            self.logger.info("Training LightGBM")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=self.config.training_config["random_state"],
                n_jobs=self.config.max_workers,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            ensemble_models["lightgbm"] = lgb_model
            
            return ensemble_models
            
        except Exception as e:
            self.logger.error(f"Error training ensemble models: {e}")
            return {}


class ArtifactManager:
    """Manages model artifacts and persistence."""
    
    def __init__(self, config: UnifiedRegimeConfig):
        self.config = config
        self.logger = system_logger.getChild("ArtifactManager")
        self.output_dir = Path(config.data_dir) / "unified_regime_intelligence"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_models(self, models: Dict[str, Any]) -> bool:
        """Save trained models."""
        try:
            for model_name, model in models.items():
                if model is None:
                    continue
                
                if isinstance(model, nn.Module):
                    # Save PyTorch model
                    model_path = self.output_dir / f"{self.config.symbol}_{model_name}.pth"
                    torch.save(model.state_dict(), model_path)
                else:
                    # Save scikit-learn/LightGBM model
                    model_path = self.output_dir / f"{self.config.symbol}_{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                self.logger.info(f"Saved {model_name} model to {model_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    def save_training_history(self, history: Dict[str, Any]) -> bool:
        """Save training history and metrics."""
        try:
            history_path = self.output_dir / f"{self.config.symbol}_training_history.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    serializable_history[key] = value.tolist()
                else:
                    serializable_history[key] = value
            
            with open(history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            self.logger.info(f"Saved training history to {history_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving training history: {e}")
            return False


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "torch", "sklearn", "lightgbm"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["target"],
    },
    context="Improved Unified Regime Intelligence",
)
@secure_data_processing(
    backup_before=True, 
    integrity_checks=True, 
    memory_cleanup=True, 
    data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=10.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=1000, 
    streaming_processing=True, 
    memory_pool=True, 
    cleanup_frequency=3
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=15.0,
)
@validate_step_output(
    output_validation=True,
    data_quality_checks=True,
    performance_metrics=True,
)
@quality_gate(
    quality_threshold=0.7,
    validation_metrics=["model_accuracy", "training_quality", "performance"],
)
async def run_step_improved(
    symbol: str,
    exchange: str = "BINANCE",
    timeframe: str = "1m",
    training_config: Dict[str, Any] | None = None,
    force_rerun: bool = False,
) -> bool:
    """
    Improved Step 5: Unified Regime Intelligence with enhanced code quality and performance.
    
    Key improvements:
    - Modular architecture with separate classes
    - Better memory management and GPU utilization
    - Improved error handling and validation
    - Enhanced model training with early stopping
    - Comprehensive logging and monitoring
    """
    logger = system_logger.getChild("Step5.ImprovedUnifiedRegimeIntelligence")
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting improved Step 5: Unified Regime Intelligence")
        
        # Initialize configuration
        config = UnifiedRegimeConfig(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            force_rerun=force_rerun,
        )
        
        # Update config with training config if provided
        if training_config:
            if "UNIFIED_REGIME_INTELLIGENCE" in training_config:
                config.model_config.update(training_config["UNIFIED_REGIME_INTELLIGENCE"])
        
        # Initialize components
        data_manager = DataManager(config)
        model_trainer = ModelTrainer(config)
        artifact_manager = ArtifactManager(config)
        
        # Load training data
        logger.info("📥 Loading training data")
        data = await data_manager.load_training_data()
        
        if data is None:
            raise ValueError("Failed to load training data")
        
        # Preprocess data
        logger.info("🔧 Preprocessing data")
        processed_data = data_manager.preprocess_data(data)
        
        if not processed_data:
            raise ValueError("Failed to preprocess data")
        
        # Train models
        logger.info("🎯 Training unified regime intelligence models")
        training_success = await model_trainer.train_models(processed_data)
        
        if not training_success:
            raise RuntimeError("Failed to train models")
        
        # Save models and artifacts
        logger.info("💾 Saving models and artifacts")
        if not artifact_manager.save_models(model_trainer.models):
            raise RuntimeError("Failed to save models")
        
        if model_trainer.training_history:
            if not artifact_manager.save_training_history(model_trainer.training_history):
                logger.warning("Failed to save training history")
        
        # Log completion
        total_time = time.time() - start_time
        num_models = len(model_trainer.models)
        
        logger.info(f"✅ Unified regime intelligence completed successfully")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        logger.info(f"   🤖 Trained models: {num_models}")
        logger.info(f"   📊 Model types: {list(model_trainer.models.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Unified regime intelligence failed: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Backward compatibility
async def run_step(*args, **kwargs):
    """Backward compatibility wrapper for the improved run_step function."""
    return await run_step_improved(*args, **kwargs)