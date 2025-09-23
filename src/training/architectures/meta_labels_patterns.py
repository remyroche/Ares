"""
Meta-Labels and Patterns System for Enhanced ML Model Training

This module implements meta-labels and pattern recognition systems to improve
ML model training with higher-level abstractions and pattern-based learning.

Key features:
- Meta-label generation from market patterns
- Pattern recognition and classification
- Meta-learning for pattern adaptation
- Hierarchical pattern representation
- Pattern-based feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optimized imports
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context
)

logger = get_logger('MetaLabelsPatterns')

class PatternType(Enum):
    """Types of market patterns."""
    TREND = "trend"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"

class MetaLabelType(Enum):
    """Types of meta-labels."""
    PATTERN_LABEL = "pattern_label"
    REGIME_LABEL = "regime_label"
    TRANSITION_LABEL = "transition_label"
    CONFIDENCE_LABEL = "confidence_label"
    UNCERTAINTY_LABEL = "uncertainty_label"

@dataclass
class MetaLabelsConfig:
    """Configuration for meta-labels and patterns system."""
    
    # Input configuration
    input_features: int = 50
    sequence_length: int = 60
    
    # Pattern recognition
    pattern_types: List[PatternType] = field(default_factory=lambda: [
        PatternType.TREND, PatternType.REVERSAL, PatternType.CONSOLIDATION,
        PatternType.BREAKOUT, PatternType.VOLATILITY, PatternType.MOMENTUM
    ])
    pattern_window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 40])
    
    # Meta-label configuration
    meta_label_types: List[MetaLabelType] = field(default_factory=lambda: [
        MetaLabelType.PATTERN_LABEL, MetaLabelType.REGIME_LABEL,
        MetaLabelType.TRANSITION_LABEL, MetaLabelType.CONFIDENCE_LABEL
    ])
    
    # Pattern clustering
    num_pattern_clusters: int = 10
    pattern_embedding_dim: int = 32
    clustering_method: str = 'kmeans'  # 'kmeans', 'hierarchical', 'dbscan'
    
    # Meta-learning
    meta_learning_rate: float = 0.001
    meta_batch_size: int = 32
    meta_epochs: int = 100
    
    # Pattern similarity
    similarity_threshold: float = 0.8
    pattern_memory_size: int = 1000
    
    # Training configuration
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0


class PatternRecognizer(nn.Module):
    """Neural network for pattern recognition."""
    
    def __init__(self, config: MetaLabelsConfig):
        super().__init__()
        self.config = config
        
        # Pattern recognition network
        self.pattern_net = nn.Sequential(
            nn.Linear(config.input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(config.pattern_types))
        )
        
        # Pattern embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(config.input_features, 128),
            nn.ReLU(),
            nn.Linear(128, config.pattern_embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through pattern recognizer."""
        # Pattern classification
        pattern_logits = self.pattern_net(x)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        # Pattern embedding
        pattern_embedding = self.embedding_net(x)
        
        return {
            'pattern_logits': pattern_logits,
            'pattern_probs': pattern_probs,
            'pattern_embedding': pattern_embedding
        }


class MetaLabelGenerator(nn.Module):
    """Meta-label generator for enhanced training."""
    
    def __init__(self, config: MetaLabelsConfig):
        super().__init__()
        self.config = config
        
        # Meta-label networks
        self.meta_label_nets = nn.ModuleDict()
        
        for label_type in config.meta_label_types:
            if label_type == MetaLabelType.PATTERN_LABEL:
                self.meta_label_nets[label_type.value] = nn.Sequential(
                    nn.Linear(config.input_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, len(config.pattern_types))
                )
            elif label_type == MetaLabelType.REGIME_LABEL:
                self.meta_label_nets[label_type.value] = nn.Sequential(
                    nn.Linear(config.input_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3)  # Low, Medium, High volatility
                )
            elif label_type == MetaLabelType.TRANSITION_LABEL:
                self.meta_label_nets[label_type.value] = nn.Sequential(
                    nn.Linear(config.input_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)  # Binary: transition or not
                )
            elif label_type == MetaLabelType.CONFIDENCE_LABEL:
                self.meta_label_nets[label_type.value] = nn.Sequential(
                    nn.Linear(config.input_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)  # Confidence score [0,1]
                )
            elif label_type == MetaLabelType.UNCERTAINTY_LABEL:
                self.meta_label_nets[label_type.value] = nn.Sequential(
                    nn.Linear(config.input_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)  # Uncertainty score [0,1]
                )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate meta-labels for input data."""
        meta_labels = {}
        
        for label_type in self.config.meta_label_types:
            label_key = label_type.value
            if label_key in self.meta_label_nets:
                label_output = self.meta_label_nets[label_key](x)
                
                if label_type in [MetaLabelType.TRANSITION_LABEL, MetaLabelType.CONFIDENCE_LABEL, MetaLabelType.UNCERTAINTY_LABEL]:
                    # Sigmoid for binary/continuous outputs
                    meta_labels[label_key] = torch.sigmoid(label_output)
                else:
                    # Softmax for categorical outputs
                    meta_labels[label_key] = F.softmax(label_output, dim=-1)
        
        return meta_labels


class PatternMemory(nn.Module):
    """Pattern memory for storing and retrieving similar patterns."""
    
    def __init__(self, config: MetaLabelsConfig):
        super().__init__()
        self.config = config
        
        # Pattern memory storage
        self.pattern_memory = torch.zeros(config.pattern_memory_size, config.pattern_embedding_dim)
        self.pattern_labels = torch.zeros(config.pattern_memory_size, dtype=torch.long)
        self.memory_index = 0
        self.memory_full = False
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(config.pattern_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def add_pattern(self, pattern_embedding: torch.Tensor, pattern_label: torch.Tensor):
        """Add a new pattern to memory."""
        if not self.memory_full:
            self.pattern_memory[self.memory_index] = pattern_embedding
            self.pattern_labels[self.memory_index] = pattern_label
            self.memory_index += 1
            
            if self.memory_index >= self.config.pattern_memory_size:
                self.memory_full = True
                self.memory_index = 0
        else:
            # Replace oldest pattern
            self.pattern_memory[self.memory_index] = pattern_embedding
            self.pattern_labels[self.memory_index] = pattern_label
            self.memory_index = (self.memory_index + 1) % self.config.pattern_memory_size
    
    def find_similar_patterns(self, pattern_embedding: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k most similar patterns in memory."""
        if not self.memory_full and self.memory_index == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        
        # Calculate similarities
        memory_size = self.config.pattern_memory_size if self.memory_full else self.memory_index
        similarities = torch.cosine_similarity(
            pattern_embedding.unsqueeze(0),
            self.pattern_memory[:memory_size],
            dim=1
        )
        
        # Get top-k similar patterns
        top_k_indices = torch.topk(similarities, min(k, memory_size)).indices
        
        return self.pattern_memory[top_k_indices], self.pattern_labels[top_k_indices]
    
    def compute_pattern_similarity(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two patterns."""
        combined = torch.cat([pattern1, pattern2], dim=-1)
        return torch.sigmoid(self.similarity_net(combined))


class PatternClustering:
    """Pattern clustering for meta-label generation."""
    
    def __init__(self, config: MetaLabelsConfig):
        self.config = config
        self.clusterer = None
        self.pattern_centers = None
        
    def fit_clusters(self, pattern_embeddings: np.ndarray) -> np.ndarray:
        """Fit clustering model to pattern embeddings."""
        if self.config.clustering_method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.config.num_pattern_clusters, random_state=42)
            cluster_labels = self.clusterer.fit_predict(pattern_embeddings)
            self.pattern_centers = self.clusterer.cluster_centers_
        else:
            raise ValueError(f"Unsupported clustering method: {self.config.clustering_method}")
        
        return cluster_labels
    
    def predict_clusters(self, pattern_embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new pattern embeddings."""
        if self.clusterer is None:
            raise ValueError("Clustering model not fitted")
        
        return self.clusterer.predict(pattern_embeddings)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        return self.pattern_centers


class MetaLearningSystem(nn.Module):
    """Meta-learning system for pattern adaptation."""
    
    def __init__(self, config: MetaLabelsConfig):
        super().__init__()
        self.config = config
        
        # Meta-learner network
        self.meta_learner = nn.Sequential(
            nn.Linear(config.input_features + config.pattern_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.input_features)  # Output adapted features
        )
        
        # Pattern adaptation network
        self.pattern_adapter = nn.Sequential(
            nn.Linear(config.pattern_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.pattern_embedding_dim)
        )
        
    def forward(self, x: torch.Tensor, pattern_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through meta-learning system."""
        # Combine input and pattern embedding
        combined_input = torch.cat([x, pattern_embedding], dim=-1)
        
        # Meta-learning adaptation
        adapted_features = self.meta_learner(combined_input)
        
        # Pattern adaptation
        adapted_pattern = self.pattern_adapter(pattern_embedding)
        
        return {
            'adapted_features': adapted_features,
            'adapted_pattern': adapted_pattern,
            'meta_representation': combined_input
        }


class MetaLabelsPatternsSystem(nn.Module):
    """Complete meta-labels and patterns system."""
    
    def __init__(self, config: MetaLabelsConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.pattern_recognizer = PatternRecognizer(config)
        self.meta_label_generator = MetaLabelGenerator(config)
        self.pattern_memory = PatternMemory(config)
        self.meta_learning_system = MetaLearningSystem(config)
        
        # Pattern clustering
        self.pattern_clustering = PatternClustering(config)
        
        # Output layers
        self.output_layers = nn.ModuleDict({
            'pattern_classification': nn.Linear(len(config.pattern_types), len(config.pattern_types)),
            'meta_label_prediction': nn.Linear(config.input_features, sum([
                1 if label_type in [MetaLabelType.TRANSITION_LABEL, MetaLabelType.CONFIDENCE_LABEL, MetaLabelType.UNCERTAINTY_LABEL]
                else (3 if label_type == MetaLabelType.REGIME_LABEL else len(config.pattern_types))
                for label_type in config.meta_label_types
            ]))
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through meta-labels and patterns system."""
        # Pattern recognition
        pattern_outputs = self.pattern_recognizer(x)
        
        # Meta-label generation
        meta_labels = self.meta_label_generator(x)
        
        # Pattern memory operations
        pattern_embedding = pattern_outputs['pattern_embedding']
        similar_patterns, similar_labels = self.pattern_memory.find_similar_patterns(
            pattern_embedding, k=5
        )
        
        # Meta-learning adaptation
        meta_learning_outputs = self.meta_learning_system(x, pattern_embedding)
        
        # Final outputs
        outputs = {
            'pattern_classification': pattern_outputs['pattern_probs'],
            'pattern_embedding': pattern_embedding,
            'meta_labels': meta_labels,
            'similar_patterns': similar_patterns,
            'similar_labels': similar_labels,
            'adapted_features': meta_learning_outputs['adapted_features'],
            'adapted_pattern': meta_learning_outputs['adapted_pattern']
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss for meta-labels and patterns system."""
        losses = {}
        
        # Pattern classification loss
        if 'pattern_labels' in targets:
            pattern_loss = F.cross_entropy(outputs['pattern_classification'], targets['pattern_labels'])
            losses['pattern_loss'] = pattern_loss
        
        # Meta-label losses
        for label_type in self.config.meta_label_types:
            label_key = label_type.value
            if label_key in targets and label_key in outputs['meta_labels']:
                if label_type in [MetaLabelType.TRANSITION_LABEL, MetaLabelType.CONFIDENCE_LABEL, MetaLabelType.UNCERTAINTY_LABEL]:
                    # MSE loss for continuous outputs
                    label_loss = F.mse_loss(outputs['meta_labels'][label_key], targets[label_key])
                else:
                    # Cross-entropy loss for categorical outputs
                    label_loss = F.cross_entropy(outputs['meta_labels'][label_key], targets[label_key])
                
                losses[f'{label_key}_loss'] = label_loss
        
        # Pattern similarity loss
        if 'similar_patterns' in outputs and len(outputs['similar_patterns']) > 0:
            similarity_loss = 0
            for i in range(len(outputs['similar_patterns'])):
                similarity = self.pattern_memory.compute_pattern_similarity(
                    outputs['pattern_embedding'],
                    outputs['similar_patterns'][i]
                )
                similarity_loss += F.mse_loss(similarity, torch.tensor(1.0, device=similarity.device))
            
            if len(outputs['similar_patterns']) > 0:
                similarity_loss /= len(outputs['similar_patterns'])
            
            losses['similarity_loss'] = similarity_loss
        
        # Meta-learning loss
        if 'adapted_features' in outputs and 'original_features' in targets:
            adaptation_loss = F.mse_loss(outputs['adapted_features'], targets['original_features'])
            losses['adaptation_loss'] = adaptation_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def update_pattern_memory(self, pattern_embeddings: torch.Tensor, pattern_labels: torch.Tensor):
        """Update pattern memory with new patterns."""
        for i in range(pattern_embeddings.size(0)):
            self.pattern_memory.add_pattern(pattern_embeddings[i], pattern_labels[i])
    
    def fit_pattern_clusters(self, pattern_embeddings: np.ndarray) -> np.ndarray:
        """Fit pattern clustering model."""
        return self.pattern_clustering.fit_clusters(pattern_embeddings)


class MetaLabelsPatternsTrainer:
    """Trainer for meta-labels and patterns system."""
    
    def __init__(self, model: MetaLabelsPatternsSystem, config: MetaLabelsConfig):
        self.model = model
        self.config = config
        self.logger = get_logger('MetaLabelsPatternsTrainer')
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=15,
            verbose=True
        )
        
    @traced(span_name='train_epoch')
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total_loss': 0}
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss
            losses = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Update pattern memory
            if 'pattern_embeddings' in outputs and 'pattern_labels' in targets:
                self.model.update_pattern_memory(
                    outputs['pattern_embeddings'],
                    targets['pattern_labels']
                )
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0) + value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    @traced(span_name='validate_epoch')
    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {'total_loss': 0}
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = self.model(data)
                losses = self.model.compute_loss(outputs, targets)
                
                for key, value in losses.items():
                    epoch_losses[key] = epoch_losses.get(key, 0) + value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, 
              epochs: int = 100) -> Dict[str, List[float]]:
        """Complete training loop."""
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_losses['total_loss'])
            
            # Record history
            history['train_loss'].append(train_losses['total_loss'])
            history['val_loss'].append(val_losses['total_loss'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            if epoch % 20 == 0:
                self.logger.info(f'Epoch {epoch}: Train Loss: {train_losses["total_loss"]:.4f}, '
                               f'Val Loss: {val_losses["total_loss"]:.4f}, '
                               f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        return history


# Factory functions
def create_meta_labels_patterns_system(config: Optional[MetaLabelsConfig] = None) -> MetaLabelsPatternsSystem:
    """Create meta-labels and patterns system with default configuration."""
    if config is None:
        config = MetaLabelsConfig()
    
    return MetaLabelsPatternsSystem(config)


def create_meta_labels_patterns_trainer(model: MetaLabelsPatternsSystem, config: Optional[MetaLabelsConfig] = None) -> MetaLabelsPatternsTrainer:
    """Create meta-labels and patterns trainer."""
    if config is None:
        config = MetaLabelsConfig()
    
    return MetaLabelsPatternsTrainer(model, config)


# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Meta-Labels and Patterns System')
    
    # Test configuration
    config = MetaLabelsConfig(
        input_features=50,
        sequence_length=60
    )
    
    tprint(f'📊 Meta-Labels Configuration:')
    tprint(f'   → Input features: {config.input_features}')
    tprint(f'   → Sequence length: {config.sequence_length}')
    tprint(f'   → Pattern types: {[pt.value for pt in config.pattern_types]}')
    tprint(f'   → Meta-label types: {[mlt.value for mlt in config.meta_label_types]}')
    tprint(f'   → Number of pattern clusters: {config.num_pattern_clusters}')
    
    # Test model creation
    try:
        model = create_meta_labels_patterns_system(config)
        trainer = create_meta_labels_patterns_trainer(model, config)
        
        # Test forward pass
        batch_size = 32
        test_input = torch.randn(batch_size, config.sequence_length, config.input_features)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        tprint('✅ Meta-labels and patterns system created successfully')
        tprint(f'   → Output keys: {list(outputs.keys())}')
        tprint(f'   → Pattern classification shape: {outputs["pattern_classification"].shape}')
        tprint(f'   → Pattern embedding shape: {outputs["pattern_embedding"].shape}')
        tprint(f'   → Meta-labels: {list(outputs["meta_labels"].keys())}')
        tprint(f'   → Similar patterns shape: {outputs["similar_patterns"].shape}')
        
        # Test loss computation
        test_targets = {
            'pattern_labels': torch.randint(0, len(config.pattern_types), (batch_size,)),
            'pattern_embeddings': torch.randn(batch_size, config.pattern_embedding_dim),
            'pattern_labels': torch.randint(0, len(config.pattern_types), (batch_size,)),
            'regime_label': torch.randint(0, 3, (batch_size,)),
            'transition_label': torch.randn(batch_size, 1),
            'confidence_label': torch.randn(batch_size, 1),
            'original_features': torch.randn(batch_size, config.input_features)
        }
        
        losses = model.compute_loss(outputs, test_targets)
        tprint(f'   → Loss components: {list(losses.keys())}')
        tprint(f'   → Total loss: {losses["total_loss"].item():.4f}')
        
    except Exception as e:
        tprint(f'❌ Error creating meta-labels and patterns system: {e}')
    
    tprint('✅ Meta-Labels and Patterns System test completed!')