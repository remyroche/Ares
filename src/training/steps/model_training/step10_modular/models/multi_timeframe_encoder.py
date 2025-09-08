from ..standardized_parquet_handler import standardized_parquet_handler
"""Multi-Timeframe HMM Encoder Model.

This module contains the core neural network architecture for the unified
regime intelligence system, handling multi-timeframe HMM state encoding
with attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from src.utils.logger import system_logger

logger = system_logger.getChild('MultiTimeframeHMMEncoder')


class MultiTimeframeHMMEncoder(nn.Module):
    """Multi-timeframe HMM state encoder using attention mechanisms.

    This model processes HMM states from multiple timeframes and market features
    to produce unified regime intelligence predictions including:
    - Regime classification
    - Intensity prediction
    - Transition probability
    - TPSL-based direction (long/short)
    - Confidence scoring
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the multi-timeframe HMM encoder.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()

        # Model configuration
        self.timeframes = config.get("timeframes", ["1m", "5m", "15m", "30m", "1h"])
        self.hmm_states_per_tf = config.get("hmm_states_per_tf", 5)
        self.d_model = config.get("d_model", 256)
        self.nhead = config.get("nhead", 8)
        self.num_layers = config.get("num_layers", 4)
        self.dropout = config.get("dropout", 0.1)

        logger.info("🚀 Initializing MultiTimeframeHMMEncoder")
        logger.info(f"   Timeframes: {self.timeframes}")
        logger.info(f"   HMM states per TF: {self.hmm_states_per_tf}")
        logger.info(f"   Model dimension: {self.d_model}")

        # Per-timeframe HMM state embeddings
        per_tf_dim = max(1, self.d_model // max(1, len(self.timeframes)))
        self.hmm_embeddings = nn.ModuleDict({
            tf: nn.Embedding(num_embeddings=self.hmm_states_per_tf, embedding_dim=per_tf_dim)
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers,
        )

        # Output projections - will be dynamically set based on actual data
        self.num_regimes: int | None = None  # Will be determined from data
        self.regime_classifier: nn.Linear | None = None  # Will be initialized later
        self.intensity_predictor: nn.Linear | None = None  # Will be initialized later
        self.transition_predictor = nn.Linear(self.d_model, 2)  # transition probability
        self.tpsl_predictor = nn.Linear(self.d_model, 2)  # TPSL-based direction (long/short only)
        self.confidence_predictor = nn.Linear(self.d_model, 1)  # confidence score

        # Persisted feature projection (initialized lazily to match input feature dimension)
        self.feature_projection: nn.Linear | None = None

        logger.info("✅ MultiTimeframeHMMEncoder initialized successfully")

    def forward(self, hmm_states: Dict[str, torch.Tensor], features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the unified regime intelligence model.

        Args:
            hmm_states: Dict of HMM state sequences per timeframe
            features: Additional market features

        Returns:
            Dict containing regime classification, transition predictions, and other outputs
        """
        batch_size = features.size(0)
        seq_len = features.size(1)

        # Encode HMM states for each timeframe
        tf_embeddings: list[torch.Tensor] = []
        for tf in self.timeframes:
            if tf in hmm_states:
                tf_embed = self.hmm_embeddings[tf](hmm_states[tf])
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
                batch_size, seq_len, self.d_model, device=features.device,
            )

        # Combine with market features (lazy-init projection to avoid recreating each forward)
        if (self.feature_projection is None or
            getattr(self.feature_projection, "in_features", None) != features.size(-1)):
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
        regime_logits = (self.regime_classifier(pooled)
                        if self.regime_classifier is not None
                        else torch.zeros((batch_size, 1), device=pooled.device))
        intensity_logits = (self.intensity_predictor(pooled)
                           if self.intensity_predictor is not None
                           else torch.zeros((batch_size, 1), device=pooled.device))
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

    def initialize_output_layers(self, num_regimes: int, num_intensity_features: int) -> None:
        """Initialize output layers based on data characteristics.

        Args:
            num_regimes: Number of regimes detected in the data
            num_intensity_features: Number of intensity features
        """
        self.num_regimes = num_regimes
        self.regime_classifier = nn.Linear(self.d_model, num_regimes)
        self.intensity_predictor = nn.Linear(self.d_model, num_intensity_features)

        logger.info(f"✅ Output layers initialized: {num_regimes} regimes, {num_intensity_features} intensity features")
