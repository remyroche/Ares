"""
ML-Enhanced Market Discovery Framework

This module extends the existing clusters framework with advanced ML techniques for:
1. Automated feature discovery and synthesis
2. Implicit market dimension discovery using deep learning
3. Adaptive regime identification with neural networks
4. Predictive regime transition modeling
5. Automated hyperparameter optimization for clustering

Key ML Enhancements:
- Deep autoencoders for non-linear dimension reduction
- LSTM/Transformer models for temporal regime patterns
- Reinforcement learning for adaptive clustering parameters
- Neural architecture search for optimal feature combinations
- Ensemble methods for robust regime prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import FastICA, NMF
from sklearn.ensemble import IsolationForest
from sklearn.cluster import OPTICS, SpectralClustering
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.utils.logger import system_logger

# Import existing framework components
from .dimension_analyzer import MarketDimensionAnalyzer, MarketDimension
from .feature_importance import RegimeFeatureImportance, ImportanceMethod
from .regime_clusterer import RegimeClusterer, ClusteringMethod
from .validation_metrics import RegimeValidationMetrics


class MLDiscoveryMethod(Enum):
    """ML methods for enhanced market discovery."""
    AUTOENCODER = "autoencoder"
    VARIATIONAL_AUTOENCODER = "vae"
    LSTM_ENCODER = "lstm_encoder"
    TRANSFORMER_ENCODER = "transformer_encoder"
    MANIFOLD_LEARNING = "manifold_learning"
    DEEP_CLUSTERING = "deep_clustering"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    REINFORCEMENT_CLUSTERING = "rl_clustering"
    ENSEMBLE_DISCOVERY = "ensemble_discovery"


@dataclass
class MLDiscoveryConfig:
    """Configuration for ML-enhanced discovery."""
    # Neural network parameters
    hidden_dims: List[int] = None
    latent_dim: int = 10
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    dropout_rate: float = 0.2
    
    # Autoencoder parameters
    encoder_layers: List[int] = None
    decoder_layers: List[int] = None
    activation: str = "relu"
    use_batch_norm: bool = True
    
    # LSTM parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    sequence_length: int = 20
    
    # Transformer parameters
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dim: int = 128
    
    # Manifold learning parameters
    manifold_method: str = "tsne"  # tsne, isomap, lle
    manifold_neighbors: int = 15
    manifold_components: int = 2
    
    # Optimization parameters
    use_optuna: bool = True
    optuna_trials: int = 100
    optuna_timeout: int = 3600  # 1 hour
    
    # Ensemble parameters
    ensemble_methods: List[str] = None
    ensemble_voting: str = "soft"
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    random_state: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
        if self.encoder_layers is None:
            self.encoder_layers = [256, 128, 64]
        if self.decoder_layers is None:
            self.decoder_layers = [64, 128, 256]
        if self.ensemble_methods is None:
            self.ensemble_methods = ["autoencoder", "lstm_encoder", "manifold_learning"]
        
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class AutoencoderDiscovery(nn.Module):
    """Deep autoencoder for discovering implicit market dimensions."""
    
    def __init__(self, input_dim: int, config: MLDiscoveryConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.encoder_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(self._get_activation())
            encoder_layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = config.latent_dim
        
        for hidden_dim in config.decoder_layers:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(self._get_activation())
            decoder_layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self):
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "tanh":
            return nn.Tanh()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class LSTMRegimeEncoder(nn.Module):
    """LSTM-based encoder for temporal regime patterns."""
    
    def __init__(self, input_dim: int, config: MLDiscoveryConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(config.lstm_hidden_size, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout and final linear layer
        encoded = self.dropout(last_hidden)
        encoded = self.fc(encoded)
        
        return encoded


class TransformerRegimeEncoder(nn.Module):
    """Transformer-based encoder for complex temporal dependencies."""
    
    def __init__(self, input_dim: int, config: MLDiscoveryConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.transformer_dim)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            config.sequence_length, config.transformer_dim
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.transformer_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.transformer_dim, config.latent_dim)
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Final projection
        encoded = self.output_projection(pooled)
        
        return encoded


class MLEnhancedDiscovery:
    """Main class for ML-enhanced market discovery."""
    
    def __init__(self, config: MLDiscoveryConfig = None):
        self.config = config or MLDiscoveryConfig()
        self.logger = system_logger.getChild('MLEnhancedDiscovery')
        
        # Initialize models dictionary
        self.models = {}
        self.fitted_models = {}
        self.discovery_results = {}
        
        # Set device
        self.device = torch.device(self.config.device)
        self.logger.info(f"Using device: {self.device}")
    
    def discover_implicit_dimensions(
        self, 
        features: pd.DataFrame, 
        method: MLDiscoveryMethod = MLDiscoveryMethod.AUTOENCODER
    ) -> Dict[str, Any]:
        """Discover implicit market dimensions using ML techniques."""
        
        self.logger.info(f"🧠 Discovering implicit dimensions using {method.value}")
        
        # Prepare data
        X = self._prepare_features(features)
        
        if method == MLDiscoveryMethod.AUTOENCODER:
            return self._autoencoder_discovery(X, features.columns)
        elif method == MLDiscoveryMethod.VARIATIONAL_AUTOENCODER:
            return self._vae_discovery(X, features.columns)
        elif method == MLDiscoveryMethod.LSTM_ENCODER:
            return self._lstm_discovery(X, features.columns)
        elif method == MLDiscoveryMethod.TRANSFORMER_ENCODER:
            return self._transformer_discovery(X, features.columns)
        elif method == MLDiscoveryMethod.MANIFOLD_LEARNING:
            return self._manifold_discovery(X, features.columns)
        elif method == MLDiscoveryMethod.ENSEMBLE_DISCOVERY:
            return self._ensemble_discovery(X, features.columns)
        else:
            raise ValueError(f"Unknown discovery method: {method}")
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML processing."""
        # Handle missing values
        X = features.fillna(features.mean()).values
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X
    
    def _autoencoder_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Discover dimensions using deep autoencoder."""
        
        input_dim = X.shape[1]
        model = AutoencoderDiscovery(input_dim, self.config).to(self.device)
        
        # Prepare data loader
        tensor_X = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(tensor_X, tensor_X)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        model.train()
        losses = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for batch_X, _ in dataloader:
                optimizer.zero_grad()
                
                reconstructed, encoded = model(batch_X)
                loss = criterion(reconstructed, batch_X)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        # Extract learned representations
        model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            _, encoded_features = model(tensor_X)
            encoded_features = encoded_features.cpu().numpy()
        
        # Analyze learned dimensions
        dimension_analysis = self._analyze_learned_dimensions(
            encoded_features, X, feature_names
        )
        
        return {
            'method': 'autoencoder',
            'encoded_features': encoded_features,
            'reconstruction_loss': losses[-1],
            'training_losses': losses,
            'dimension_analysis': dimension_analysis,
            'model': model,
            'latent_dim': self.config.latent_dim
        }
    
    def _lstm_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Discover temporal patterns using LSTM encoder."""
        
        # Create sequences
        sequences = self._create_sequences(X, self.config.sequence_length)
        
        if len(sequences) == 0:
            self.logger.warning("Not enough data for sequence creation")
            return self._fallback_discovery(X, feature_names)
        
        input_dim = sequences.shape[2]
        model = LSTMRegimeEncoder(input_dim, self.config).to(self.device)
        
        # Prepare data
        tensor_sequences = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(tensor_sequences)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training with reconstruction task
        decoder = nn.Linear(self.config.latent_dim, input_dim).to(self.device)
        optimizer = optim.Adam(
            list(model.parameters()) + list(decoder.parameters()),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        
        model.train()
        decoder.train()
        losses = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for (batch_sequences,) in dataloader:
                optimizer.zero_grad()
                
                # Encode sequences
                encoded = model(batch_sequences)
                
                # Reconstruct last timestep
                reconstructed = decoder(encoded)
                target = batch_sequences[:, -1, :]  # Last timestep
                
                loss = criterion(reconstructed, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                self.logger.info(f"LSTM Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        # Extract encoded features
        model.eval()
        with torch.no_grad():
            tensor_sequences = torch.FloatTensor(sequences).to(self.device)
            encoded_features = model(tensor_sequences).cpu().numpy()
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(
            encoded_features, sequences, feature_names
        )
        
        return {
            'method': 'lstm_encoder',
            'encoded_features': encoded_features,
            'sequences': sequences,
            'temporal_analysis': temporal_analysis,
            'training_losses': losses,
            'model': model,
            'sequence_length': self.config.sequence_length
        }
    
    def _manifold_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Discover non-linear manifold structure."""
        
        method = self.config.manifold_method.lower()
        
        if method == "tsne":
            manifold = TSNE(
                n_components=self.config.manifold_components,
                random_state=self.config.random_state,
                perplexity=min(30, len(X) // 4)
            )
        elif method == "isomap":
            manifold = Isomap(
                n_components=self.config.manifold_components,
                n_neighbors=self.config.manifold_neighbors
            )
        elif method == "lle":
            manifold = LocallyLinearEmbedding(
                n_components=self.config.manifold_components,
                n_neighbors=self.config.manifold_neighbors,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown manifold method: {method}")
        
        # Fit manifold
        try:
            embedded = manifold.fit_transform(X)
        except Exception as e:
            self.logger.warning(f"Manifold learning failed: {e}")
            return self._fallback_discovery(X, feature_names)
        
        # Analyze manifold structure
        manifold_analysis = self._analyze_manifold_structure(
            embedded, X, feature_names
        )
        
        return {
            'method': f'manifold_{method}',
            'embedded_features': embedded,
            'manifold_analysis': manifold_analysis,
            'manifold_model': manifold,
            'n_components': self.config.manifold_components
        }
    
    def _ensemble_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Combine multiple discovery methods."""
        
        ensemble_results = {}
        
        for method_name in self.config.ensemble_methods:
            try:
                if method_name == "autoencoder":
                    result = self._autoencoder_discovery(X, feature_names)
                elif method_name == "lstm_encoder":
                    result = self._lstm_discovery(X, feature_names)
                elif method_name == "manifold_learning":
                    result = self._manifold_discovery(X, feature_names)
                else:
                    continue
                
                ensemble_results[method_name] = result
                
            except Exception as e:
                self.logger.warning(f"Method {method_name} failed: {e}")
                continue
        
        # Combine results
        combined_analysis = self._combine_ensemble_results(ensemble_results)
        
        return {
            'method': 'ensemble',
            'individual_results': ensemble_results,
            'combined_analysis': combined_analysis,
            'successful_methods': list(ensemble_results.keys())
        }
    
    def _create_sequences(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for temporal modeling."""
        if len(X) < sequence_length:
            return np.array([])
        
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        
        return np.array(sequences)
    
    def _analyze_learned_dimensions(
        self, 
        encoded_features: np.ndarray, 
        original_features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze the learned latent dimensions."""
        
        analysis = {
            'latent_statistics': {},
            'feature_correlations': {},
            'dimension_interpretability': {}
        }
        
        # Latent dimension statistics
        for i in range(encoded_features.shape[1]):
            dim_values = encoded_features[:, i]
            analysis['latent_statistics'][f'dim_{i}'] = {
                'mean': float(np.mean(dim_values)),
                'std': float(np.std(dim_values)),
                'skewness': float(pd.Series(dim_values).skew()),
                'kurtosis': float(pd.Series(dim_values).kurtosis())
            }
        
        # Correlations with original features
        for i in range(encoded_features.shape[1]):
            correlations = []
            for j, feature_name in enumerate(feature_names):
                corr = np.corrcoef(encoded_features[:, i], original_features[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append((feature_name, float(corr)))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            analysis['feature_correlations'][f'dim_{i}'] = correlations[:10]  # Top 10
        
        return analysis
    
    def _analyze_temporal_patterns(
        self,
        encoded_features: np.ndarray,
        sequences: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in encoded features."""
        
        analysis = {
            'temporal_stability': {},
            'regime_transitions': {},
            'pattern_clustering': {}
        }
        
        # Temporal stability analysis
        for i in range(encoded_features.shape[1]):
            dim_values = encoded_features[:, i]
            
            # Calculate autocorrelation
            autocorr = pd.Series(dim_values).autocorr(lag=1)
            
            # Calculate volatility
            volatility = np.std(np.diff(dim_values))
            
            analysis['temporal_stability'][f'dim_{i}'] = {
                'autocorrelation': float(autocorr) if not np.isnan(autocorr) else 0.0,
                'volatility': float(volatility),
                'trend_strength': float(np.corrcoef(range(len(dim_values)), dim_values)[0, 1])
            }
        
        # Detect regime transitions
        from sklearn.cluster import KMeans
        if len(encoded_features) > 10:
            kmeans = KMeans(n_clusters=min(5, len(encoded_features) // 10), random_state=42)
            regime_labels = kmeans.fit_predict(encoded_features)
            
            # Transition analysis
            transitions = np.diff(regime_labels) != 0
            transition_rate = np.mean(transitions)
            
            analysis['regime_transitions'] = {
                'n_regimes': len(np.unique(regime_labels)),
                'transition_rate': float(transition_rate),
                'average_regime_duration': float(1 / (transition_rate + 1e-6))
            }
        
        return analysis
    
    def _analyze_manifold_structure(
        self,
        embedded: np.ndarray,
        original_features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze the discovered manifold structure."""
        
        analysis = {
            'manifold_properties': {},
            'cluster_structure': {},
            'density_analysis': {}
        }
        
        # Manifold properties
        if embedded.shape[1] >= 2:
            # Calculate convex hull area (2D case)
            if embedded.shape[1] == 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(embedded)
                    analysis['manifold_properties']['convex_hull_area'] = float(hull.volume)
                except:
                    analysis['manifold_properties']['convex_hull_area'] = 0.0
        
        # Cluster structure analysis
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clustering.fit_predict(embedded)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        analysis['cluster_structure'] = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'silhouette_score': float(silhouette_score(embedded, cluster_labels)) 
                              if n_clusters > 1 else 0.0
        }
        
        return analysis
    
    def _combine_ensemble_results(self, ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple discovery methods."""
        
        combined = {
            'consensus_dimensions': [],
            'method_agreement': {},
            'best_method': None,
            'combined_features': None
        }
        
        if not ensemble_results:
            return combined
        
        # Find best method based on reconstruction quality or other metrics
        best_score = float('inf')
        best_method = None
        
        for method, result in ensemble_results.items():
            if 'reconstruction_loss' in result:
                score = result['reconstruction_loss']
                if score < best_score:
                    best_score = score
                    best_method = method
        
        combined['best_method'] = best_method
        
        # Combine features from all methods
        all_features = []
        for method, result in ensemble_results.items():
            if 'encoded_features' in result:
                all_features.append(result['encoded_features'])
            elif 'embedded_features' in result:
                all_features.append(result['embedded_features'])
        
        if all_features:
            combined['combined_features'] = np.concatenate(all_features, axis=1)
        
        return combined
    
    def _fallback_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Fallback to simple PCA when advanced methods fail."""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(self.config.latent_dim, X.shape[1]))
        encoded_features = pca.fit_transform(X)
        
        return {
            'method': 'pca_fallback',
            'encoded_features': encoded_features,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'n_components': pca.n_components_
        }
    
    def optimize_clustering_parameters(
        self,
        features: pd.DataFrame,
        target_regimes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Use ML to optimize clustering parameters."""
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, using default parameters")
            return {}
        
        X = self._prepare_features(features)
        
        def objective(trial):
            # Suggest hyperparameters
            method = trial.suggest_categorical('method', ['kmeans', 'gmm', 'spectral'])
            n_clusters = trial.suggest_int('n_clusters', 2, 15)
            
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == 'gmm':
                from sklearn.mixture import GaussianMixture
                covariance_type = trial.suggest_categorical('covariance_type', 
                                                          ['full', 'tied', 'diag', 'spherical'])
                clusterer = GaussianMixture(n_components=n_clusters, 
                                          covariance_type=covariance_type,
                                          random_state=42)
            elif method == 'spectral':
                gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)
                clusterer = SpectralClustering(n_clusters=n_clusters, 
                                             gamma=gamma,
                                             random_state=42)
            
            # Fit and evaluate
            try:
                labels = clusterer.fit_predict(X)
                
                # Use silhouette score as objective
                score = silhouette_score(X, labels)
                
                # If we have target regimes, also consider ARI
                if target_regimes is not None:
                    ari = adjusted_rand_score(target_regimes, labels)
                    score = 0.7 * score + 0.3 * ari
                
                return score
            except:
                return -1.0  # Bad score for failed clustering
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.optuna_trials, 
                      timeout=self.config.optuna_timeout)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials]
        }
    
    def predict_regime_transitions(
        self,
        features: pd.DataFrame,
        regime_labels: np.ndarray,
        prediction_horizon: int = 5
    ) -> Dict[str, Any]:
        """Train ML model to predict regime transitions."""
        
        X = self._prepare_features(features)
        
        # Create sequences for prediction
        sequences = self._create_sequences(X, self.config.sequence_length)
        regime_sequences = self._create_sequences(regime_labels, self.config.sequence_length)
        
        if len(sequences) == 0:
            return {'error': 'Not enough data for sequence creation'}
        
        # Prepare targets (future regime changes)
        targets = []
        for i in range(len(regime_sequences) - prediction_horizon):
            current_regime = regime_sequences[i][-1]  # Last regime in sequence
            future_regimes = regime_labels[i + self.config.sequence_length:
                                         i + self.config.sequence_length + prediction_horizon]
            
            # Check if regime will change in next prediction_horizon steps
            will_change = np.any(future_regimes != current_regime)
            targets.append(int(will_change))
        
        # Align sequences with targets
        sequences = sequences[:len(targets)]
        targets = np.array(targets)
        
        # Build prediction model
        input_dim = sequences.shape[2]
        
        class TransitionPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.2)
                self.fc = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                lstm_out, (h_n, _) = self.lstm(x)
                return self.fc(h_n[-1])
        
        model = TransitionPredictor().to(self.device)
        
        # Train model
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        # Training loop
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_X = torch.FloatTensor(X_test).to(self.device)
            predictions = model(test_X).cpu().numpy()
            
            # Calculate accuracy
            pred_labels = (predictions > 0.5).astype(int).flatten()
            accuracy = np.mean(pred_labels == y_test)
        
        return {
            'model': model,
            'accuracy': float(accuracy),
            'prediction_horizon': prediction_horizon,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }


def create_ml_enhanced_pipeline(
    market_data: pd.DataFrame,
    config: MLDiscoveryConfig = None
) -> Dict[str, Any]:
    """Create complete ML-enhanced discovery pipeline."""
    
    logger = system_logger.getChild('MLEnhancedPipeline')
    logger.info("🚀 Starting ML-Enhanced Market Discovery Pipeline")
    
    if config is None:
        config = MLDiscoveryConfig()
    
    # Initialize ML discovery
    ml_discovery = MLEnhancedDiscovery(config)
    
    # Initialize existing framework components
    dimension_analyzer = MarketDimensionAnalyzer()
    feature_importance = RegimeFeatureImportance()
    regime_clusterer = RegimeClusterer()
    validator = RegimeValidationMetrics()
    
    results = {}
    
    try:
        # Step 1: Traditional dimension analysis
        logger.info("📊 Step 1: Traditional dimension analysis")
        traditional_results = dimension_analyzer.analyze_all_dimensions(market_data)
        results['traditional_analysis'] = traditional_results
        
        # Step 2: ML-enhanced feature discovery
        logger.info("🧠 Step 2: ML-enhanced implicit dimension discovery")
        
        # Try multiple ML methods
        ml_results = {}
        for method in [MLDiscoveryMethod.AUTOENCODER, 
                      MLDiscoveryMethod.LSTM_ENCODER,
                      MLDiscoveryMethod.MANIFOLD_LEARNING]:
            try:
                method_result = ml_discovery.discover_implicit_dimensions(
                    market_data, method
                )
                ml_results[method.value] = method_result
                logger.info(f"✅ {method.value} discovery completed")
            except Exception as e:
                logger.warning(f"❌ {method.value} failed: {e}")
        
        results['ml_discovery'] = ml_results
        
        # Step 3: Optimize clustering parameters
        logger.info("🎯 Step 3: ML-optimized clustering parameters")
        optimization_results = ml_discovery.optimize_clustering_parameters(market_data)
        results['parameter_optimization'] = optimization_results
        
        # Step 4: Enhanced clustering with ML features
        logger.info("🔄 Step 4: Enhanced clustering with ML features")
        
        # Get best ML-discovered features
        best_ml_features = None
        best_method = None
        
        for method, method_result in ml_results.items():
            if 'encoded_features' in method_result:
                best_ml_features = method_result['encoded_features']
                best_method = method
                break
            elif 'embedded_features' in method_result:
                best_ml_features = method_result['embedded_features']
                best_method = method
                break
        
        if best_ml_features is not None:
            # Combine traditional and ML features
            traditional_features = market_data.select_dtypes(include=[np.number]).fillna(0)
            
            # Create combined feature set
            ml_feature_names = [f"{best_method}_dim_{i}" 
                               for i in range(best_ml_features.shape[1])]
            ml_df = pd.DataFrame(best_ml_features, 
                               columns=ml_feature_names,
                               index=traditional_features.index[:len(best_ml_features)])
            
            combined_features = pd.concat([traditional_features.iloc[:len(ml_df)], ml_df], axis=1)
            
            # Run clustering on combined features
            clustering_results = regime_clusterer.run_all_methods(combined_features.values)
            results['enhanced_clustering'] = clustering_results
            
            # Validate results
            best_method_result = regime_clusterer.get_best_method()
            if best_method_result:
                validation_results = validator.validate_all_metrics(
                    combined_features, best_method_result[1].labels
                )
                results['validation'] = validation_results
        
        # Step 5: Regime transition prediction
        if 'enhanced_clustering' in results:
            logger.info("🔮 Step 5: Regime transition prediction")
            best_labels = results['enhanced_clustering']['best_labels']
            transition_results = ml_discovery.predict_regime_transitions(
                market_data, best_labels
            )
            results['transition_prediction'] = transition_results
        
        logger.info("✅ ML-Enhanced Discovery Pipeline Completed Successfully")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        results['error'] = str(e)
    
    return results


# Example usage and integration
if __name__ == "__main__":
    # This would be called from your main research workflow
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create sample market data with regime structure
    data = np.random.randn(n_samples, n_features)
    
    # Add some regime structure
    regime_changes = [200, 400, 700]
    for i, change_point in enumerate(regime_changes):
        if i == 0:
            data[:change_point, :10] += 2  # High momentum regime
        elif i == 1:
            data[change_point:regime_changes[i+1] if i+1 < len(regime_changes) else None, 10:20] += 1.5  # High volatility
        else:
            data[change_point:, 20:30] -= 1  # Mean reversion regime
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    market_data = pd.DataFrame(data, columns=feature_names)
    
    # Run ML-enhanced pipeline
    config = MLDiscoveryConfig(
        latent_dim=8,
        epochs=50,
        sequence_length=20,
        use_optuna=True,
        optuna_trials=20
    )
    
    results = create_ml_enhanced_pipeline(market_data, config)
    
    print("🎯 ML-Enhanced Discovery Results:")
    print(f"Traditional dimensions analyzed: {len(results.get('traditional_analysis', {}))}")
    print(f"ML methods successful: {len(results.get('ml_discovery', {}))}")
    
    if 'parameter_optimization' in results and results['parameter_optimization']:
        print(f"Best clustering parameters: {results['parameter_optimization']['best_params']}")
    
    if 'transition_prediction' in results:
        print(f"Transition prediction accuracy: {results['transition_prediction']['accuracy']:.3f}")