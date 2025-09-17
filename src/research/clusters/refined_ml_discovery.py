"""
Refined ML Discovery Framework - Pure Regime Discovery Focus

This module focuses exclusively on discovering market regimes using ML techniques,
removing transition prediction and redundant feature engineering components.

Core Focus:
1. Non-linear dimension reduction for regime discovery (Autoencoders)
2. Manifold learning for regime structure discovery
3. Adaptive clustering optimization
4. Regime quality assessment

Removed Components:
- Regime transition prediction (not needed for discovery)
- Time series feature engineering (use existing feature_engineering/)
- Financial domain features (use existing feature_engineering/)
- Polynomial features (adds noise, not regime-relevant)

Key Question: Are LSTM/Transformers relevant for regime discovery?
Answer: Only if they help identify regime-defining patterns, not for prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.utils.logger import system_logger


class RegimeDiscoveryMethod(Enum):
    """Methods focused purely on regime discovery."""
    AUTOENCODER = "autoencoder"
    VARIATIONAL_AUTOENCODER = "vae"
    MANIFOLD_TSNE = "manifold_tsne"
    MANIFOLD_ISOMAP = "manifold_isomap"
    MANIFOLD_LLE = "manifold_lle"
    DEEP_CLUSTERING = "deep_clustering"
    ENSEMBLE_DISCOVERY = "ensemble_discovery"


@dataclass
class RegimeDiscoveryConfig:
    """Configuration for regime discovery."""
    # Autoencoder parameters
    latent_dim: int = 8
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    dropout_rate: float = 0.2
    
    # Manifold learning parameters
    manifold_components: int = 3
    manifold_neighbors: int = 15
    tsne_perplexity: int = 30
    
    # Deep clustering parameters
    n_clusters_range: Tuple[int, int] = (2, 15)
    
    # General parameters
    device: str = "auto"
    random_state: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32, 16]
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class RegimeAutoencoder(nn.Module):
    """Autoencoder specifically designed for regime discovery."""
    
    def __init__(self, input_dim: int, config: RegimeDiscoveryConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Encoder: compress market data to regime-defining latent space
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Latent space - this should capture regime-defining factors
        encoder_layers.append(nn.Linear(prev_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: reconstruct market data from regime factors
        decoder_layers = []
        prev_dim = config.latent_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode market data to regime-defining latent factors."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode regime factors back to market data."""
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class VariationalRegimeAutoencoder(nn.Module):
    """VAE for probabilistic regime discovery."""
    
    def __init__(self, input_dim: int, config: RegimeDiscoveryConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, config.latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = config.latent_dim
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


class DeepClusteringModel(nn.Module):
    """Deep clustering model for direct regime discovery."""
    
    def __init__(self, input_dim: int, n_clusters: int, config: RegimeDiscoveryConfig):
        super().__init__()
        self.n_clusters = n_clusters
        
        # Feature extraction network
        feature_layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final feature layer
        feature_layers.append(nn.Linear(prev_dim, config.latent_dim))
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Clustering layer
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, config.latent_dim))
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Calculate distances to cluster centers
        distances = torch.cdist(features.unsqueeze(0), self.cluster_centers.unsqueeze(0)).squeeze(0)
        
        # Soft assignments
        assignments = torch.softmax(-distances, dim=1)
        
        return features, assignments, distances


class RefinedMLDiscovery:
    """Refined ML discovery focused purely on regime identification."""
    
    def __init__(self, config: RegimeDiscoveryConfig = None):
        self.config = config or RegimeDiscoveryConfig()
        self.logger = system_logger.getChild('RefinedMLDiscovery')
        self.device = torch.device(self.config.device)
        
        self.logger.info(f"🎯 Refined ML Discovery initialized (device: {self.device})")
    
    def discover_regimes(
        self, 
        features: pd.DataFrame, 
        method: RegimeDiscoveryMethod = RegimeDiscoveryMethod.AUTOENCODER
    ) -> Dict[str, Any]:
        """Discover market regimes using specified ML method."""
        
        self.logger.info(f"🔍 Discovering regimes using {method.value}")
        
        # Prepare data
        X = self._prepare_features(features)
        
        if method == RegimeDiscoveryMethod.AUTOENCODER:
            return self._autoencoder_regime_discovery(X, features.columns)
        elif method == RegimeDiscoveryMethod.VARIATIONAL_AUTOENCODER:
            return self._vae_regime_discovery(X, features.columns)
        elif method == RegimeDiscoveryMethod.MANIFOLD_TSNE:
            return self._manifold_regime_discovery(X, features.columns, "tsne")
        elif method == RegimeDiscoveryMethod.MANIFOLD_ISOMAP:
            return self._manifold_regime_discovery(X, features.columns, "isomap")
        elif method == RegimeDiscoveryMethod.MANIFOLD_LLE:
            return self._manifold_regime_discovery(X, features.columns, "lle")
        elif method == RegimeDiscoveryMethod.DEEP_CLUSTERING:
            return self._deep_clustering_regime_discovery(X, features.columns)
        elif method == RegimeDiscoveryMethod.ENSEMBLE_DISCOVERY:
            return self._ensemble_regime_discovery(X, features.columns)
        else:
            raise ValueError(f"Unknown regime discovery method: {method}")
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML processing."""
        # Handle missing values
        X = features.fillna(features.median()).values
        
        # Standardize features for regime discovery
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X
    
    def _autoencoder_regime_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Discover regimes using autoencoder latent space."""
        
        input_dim = X.shape[1]
        model = RegimeAutoencoder(input_dim, self.config).to(self.device)
        
        # Prepare data
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
            
            if epoch % 20 == 0 and self.config.verbose:
                self.logger.info(f"Epoch {epoch}, Reconstruction Loss: {avg_loss:.6f}")
        
        # Extract regime-defining latent features
        model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            _, regime_features = model(tensor_X)
            regime_features = regime_features.cpu().numpy()
        
        # Analyze regime structure in latent space
        regime_analysis = self._analyze_regime_structure(regime_features, X, feature_names)
        
        return {
            'method': 'autoencoder_regime_discovery',
            'regime_features': regime_features,
            'reconstruction_loss': losses[-1],
            'regime_analysis': regime_analysis,
            'model': model,
            'latent_dim': self.config.latent_dim,
            'success': True
        }
    
    def _vae_regime_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Discover regimes using VAE probabilistic latent space."""
        
        input_dim = X.shape[1]
        model = VariationalRegimeAutoencoder(input_dim, self.config).to(self.device)
        
        # Prepare data
        tensor_X = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(tensor_X)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        def vae_loss(recon_x, x, mu, logvar):
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kl_loss
        
        model.train()
        losses = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for (batch_X,) in dataloader:
                optimizer.zero_grad()
                
                recon_x, mu, logvar, z = model(batch_X)
                loss = vae_loss(recon_x, batch_X, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 20 == 0 and self.config.verbose:
                self.logger.info(f"VAE Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        # Extract probabilistic regime features
        model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            _, mu, logvar, regime_features = model(tensor_X)
            regime_features = regime_features.cpu().numpy()
            mu = mu.cpu().numpy()
            logvar = logvar.cpu().numpy()
        
        # Analyze probabilistic regime structure
        regime_analysis = self._analyze_regime_structure(regime_features, X, feature_names)
        regime_analysis['probabilistic_analysis'] = {
            'latent_mean': np.mean(mu, axis=0).tolist(),
            'latent_std': np.std(mu, axis=0).tolist(),
            'uncertainty': np.mean(np.exp(0.5 * logvar), axis=0).tolist()
        }
        
        return {
            'method': 'vae_regime_discovery',
            'regime_features': regime_features,
            'latent_mu': mu,
            'latent_logvar': logvar,
            'regime_analysis': regime_analysis,
            'model': model,
            'success': True
        }
    
    def _manifold_regime_discovery(self, X: np.ndarray, feature_names: List[str], method: str) -> Dict[str, Any]:
        """Discover regimes using manifold learning."""
        
        if method == "tsne":
            manifold = TSNE(
                n_components=self.config.manifold_components,
                perplexity=min(self.config.tsne_perplexity, len(X) // 4),
                random_state=self.config.random_state
            )
        elif method == "isomap":
            manifold = Isomap(
                n_components=self.config.manifold_components,
                n_neighbors=min(self.config.manifold_neighbors, len(X) // 2)
            )
        elif method == "lle":
            manifold = LocallyLinearEmbedding(
                n_components=self.config.manifold_components,
                n_neighbors=min(self.config.manifold_neighbors, len(X) // 2),
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown manifold method: {method}")
        
        try:
            regime_features = manifold.fit_transform(X)
            
            # Analyze manifold regime structure
            regime_analysis = self._analyze_regime_structure(regime_features, X, feature_names)
            
            return {
                'method': f'manifold_{method}_regime_discovery',
                'regime_features': regime_features,
                'regime_analysis': regime_analysis,
                'manifold_model': manifold,
                'n_components': self.config.manifold_components,
                'success': True
            }
        
        except Exception as e:
            self.logger.warning(f"Manifold {method} failed: {e}")
            return {
                'method': f'manifold_{method}_regime_discovery',
                'error': str(e),
                'success': False
            }
    
    def _deep_clustering_regime_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Discover regimes using deep clustering."""
        
        best_score = -1
        best_result = None
        
        # Try different numbers of clusters
        for n_clusters in range(self.config.n_clusters_range[0], self.config.n_clusters_range[1] + 1):
            try:
                input_dim = X.shape[1]
                model = DeepClusteringModel(input_dim, n_clusters, self.config).to(self.device)
                
                # Prepare data
                tensor_X = torch.FloatTensor(X).to(self.device)
                
                # Training
                optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
                
                model.train()
                for epoch in range(self.config.epochs // 2):  # Fewer epochs for speed
                    optimizer.zero_grad()
                    
                    features, assignments, distances = model(tensor_X)
                    
                    # Clustering loss: minimize intra-cluster distances
                    cluster_loss = torch.mean(torch.min(distances, dim=1)[0])
                    
                    # Regularization: encourage diverse clusters
                    assignment_entropy = -torch.sum(assignments * torch.log(assignments + 1e-8), dim=1).mean()
                    
                    loss = cluster_loss - 0.1 * assignment_entropy
                    loss.backward()
                    optimizer.step()
                
                # Extract regime assignments
                model.eval()
                with torch.no_grad():
                    features, assignments, _ = model(tensor_X)
                    regime_labels = torch.argmax(assignments, dim=1).cpu().numpy()
                    regime_features = features.cpu().numpy()
                
                # Evaluate clustering quality
                if len(np.unique(regime_labels)) > 1:
                    score = silhouette_score(X, regime_labels)
                    
                    if score > best_score:
                        best_score = score
                        regime_analysis = self._analyze_regime_structure(regime_features, X, feature_names)
                        regime_analysis['regime_labels'] = regime_labels
                        regime_analysis['silhouette_score'] = score
                        
                        best_result = {
                            'method': 'deep_clustering_regime_discovery',
                            'regime_features': regime_features,
                            'regime_labels': regime_labels,
                            'regime_analysis': regime_analysis,
                            'n_clusters': n_clusters,
                            'silhouette_score': score,
                            'model': model,
                            'success': True
                        }
            
            except Exception as e:
                self.logger.warning(f"Deep clustering with {n_clusters} clusters failed: {e}")
                continue
        
        if best_result is None:
            return {
                'method': 'deep_clustering_regime_discovery',
                'error': 'All clustering attempts failed',
                'success': False
            }
        
        return best_result
    
    def _ensemble_regime_discovery(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Ensemble regime discovery combining multiple methods."""
        
        methods = [
            RegimeDiscoveryMethod.AUTOENCODER,
            RegimeDiscoveryMethod.MANIFOLD_TSNE,
            RegimeDiscoveryMethod.MANIFOLD_ISOMAP
        ]
        
        results = {}
        successful_results = []
        
        for method in methods:
            try:
                result = self.discover_regimes(pd.DataFrame(X, columns=feature_names), method)
                if result.get('success', False):
                    results[method.value] = result
                    successful_results.append(result)
            except Exception as e:
                self.logger.warning(f"Ensemble method {method.value} failed: {e}")
                results[method.value] = {'error': str(e), 'success': False}
        
        if not successful_results:
            return {
                'method': 'ensemble_regime_discovery',
                'error': 'All ensemble methods failed',
                'success': False
            }
        
        # Combine regime features from successful methods
        combined_features = []
        for result in successful_results:
            combined_features.append(result['regime_features'])
        
        ensemble_features = np.concatenate(combined_features, axis=1)
        
        # Analyze ensemble regime structure
        ensemble_analysis = self._analyze_regime_structure(ensemble_features, X, feature_names)
        
        return {
            'method': 'ensemble_regime_discovery',
            'regime_features': ensemble_features,
            'individual_results': results,
            'regime_analysis': ensemble_analysis,
            'n_successful_methods': len(successful_results),
            'success': True
        }
    
    def _analyze_regime_structure(
        self, 
        regime_features: np.ndarray, 
        original_features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze the discovered regime structure."""
        
        analysis = {
            'regime_characteristics': {},
            'clustering_potential': {},
            'feature_relationships': {}
        }
        
        try:
            # Regime feature statistics
            for i in range(regime_features.shape[1]):
                dim_values = regime_features[:, i]
                analysis['regime_characteristics'][f'regime_dim_{i}'] = {
                    'mean': float(np.mean(dim_values)),
                    'std': float(np.std(dim_values)),
                    'range': [float(np.min(dim_values)), float(np.max(dim_values))],
                    'skewness': float(pd.Series(dim_values).skew()),
                    'kurtosis': float(pd.Series(dim_values).kurtosis())
                }
            
            # Clustering potential assessment
            from sklearn.cluster import KMeans
            clustering_scores = {}
            
            for n_clusters in range(2, min(8, len(regime_features) // 10)):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(regime_features)
                    score = silhouette_score(regime_features, labels)
                    clustering_scores[n_clusters] = float(score)
                except:
                    continue
            
            if clustering_scores:
                best_k = max(clustering_scores.keys(), key=lambda k: clustering_scores[k])
                analysis['clustering_potential'] = {
                    'scores_by_k': clustering_scores,
                    'best_k': best_k,
                    'best_score': clustering_scores[best_k],
                    'regime_separability': 'high' if clustering_scores[best_k] > 0.3 else 
                                         'medium' if clustering_scores[best_k] > 0.1 else 'low'
                }
            
            # Relationship to original features
            if len(feature_names) <= 50:  # Avoid computation explosion
                correlations = []
                for i in range(regime_features.shape[1]):
                    regime_dim = regime_features[:, i]
                    dim_correlations = []
                    
                    for j, feature_name in enumerate(feature_names):
                        corr = np.corrcoef(regime_dim, original_features[:, j])[0, 1]
                        if not np.isnan(corr):
                            dim_correlations.append((feature_name, float(corr)))
                    
                    # Sort by absolute correlation
                    dim_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                    correlations.append(dim_correlations[:5])  # Top 5 per dimension
                
                analysis['feature_relationships'] = {
                    f'regime_dim_{i}': correlations[i] 
                    for i in range(len(correlations))
                }
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis


# Simplified integration focusing on regime discovery
def discover_market_regimes(
    market_data: pd.DataFrame,
    methods: Optional[List[RegimeDiscoveryMethod]] = None,
    config: RegimeDiscoveryConfig = None
) -> Dict[str, Any]:
    """Discover market regimes using refined ML approach."""
    
    if methods is None:
        methods = [
            RegimeDiscoveryMethod.AUTOENCODER,
            RegimeDiscoveryMethod.MANIFOLD_TSNE,
            RegimeDiscoveryMethod.ENSEMBLE_DISCOVERY
        ]
    
    discovery = RefinedMLDiscovery(config)
    
    results = {}
    best_result = None
    best_score = -1
    
    for method in methods:
        try:
            result = discovery.discover_regimes(market_data, method)
            results[method.value] = result
            
            if result.get('success', False):
                # Get clustering potential score
                regime_analysis = result.get('regime_analysis', {})
                clustering_potential = regime_analysis.get('clustering_potential', {})
                score = clustering_potential.get('best_score', 0)
                
                if score > best_score:
                    best_score = score
                    best_result = result
        
        except Exception as e:
            results[method.value] = {'error': str(e), 'success': False}
    
    return {
        'individual_results': results,
        'best_result': best_result,
        'best_score': best_score,
        'recommendation': 'use_regime_specific_models' if best_score > 0.3 else 
                         'use_regime_features' if best_score > 0.1 else 
                         'single_model_approach'
    }


# Example usage
if __name__ == "__main__":
    # Generate sample market data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate regime structure
    data = np.random.randn(n_samples, 20)
    
    # Add regime patterns
    regime_1 = slice(0, 300)
    regime_2 = slice(300, 700)
    regime_3 = slice(700, 1000)
    
    data[regime_1, :5] += 2    # High momentum regime
    data[regime_2, 5:10] += 1.5  # High volatility regime
    data[regime_3, 10:15] -= 1   # Mean reversion regime
    
    market_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(20)])
    
    # Discover regimes
    results = discover_market_regimes(market_data)
    
    print("🎯 Refined Regime Discovery Results:")
    print(f"Best Score: {results['best_score']:.3f}")
    print(f"Recommendation: {results['recommendation']}")
    
    if results['best_result']:
        best_analysis = results['best_result']['regime_analysis']
        clustering_potential = best_analysis.get('clustering_potential', {})
        print(f"Best K: {clustering_potential.get('best_k', 'unknown')}")
        print(f"Regime Separability: {clustering_potential.get('regime_separability', 'unknown')}")