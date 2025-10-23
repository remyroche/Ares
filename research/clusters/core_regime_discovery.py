"""
Core Regime Discovery - Focused ML Approach

This module provides the essential ML capabilities for regime discovery,
removing all unnecessary components and focusing purely on finding distinct
market regimes in feature space.

Core Principle: Regimes are STRUCTURAL patterns in market data, not temporal sequences.

Key Components:
1. Autoencoder - Non-linear dimension reduction to find regime-defining factors
2. Manifold Learning - Discover geometric structure of regimes  
3. Adaptive Clustering - Find optimal regime boundaries
4. Quality Assessment - Validate regime separability

Removed Components (as requested):
❌ Regime transition prediction (not discovery)
❌ Time series features (use existing feature_engineering/)
❌ Financial domain features (use existing feature_engineering/) 
❌ Polynomial features (adds noise)
❌ LSTM/Transformers (temporal modeling not needed for structural discovery)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from src.utils.logger import system_logger


class CoreDiscoveryMethod(Enum):
    """Core methods for regime discovery."""
    AUTOENCODER = "autoencoder"
    MANIFOLD_TSNE = "manifold_tsne"  
    MANIFOLD_ISOMAP = "manifold_isomap"
    ADAPTIVE_CLUSTERING = "adaptive_clustering"


@dataclass
class CoreDiscoveryConfig:
    """Simplified configuration for core regime discovery."""
    # Autoencoder parameters
    latent_dim: int = 6  # Reduced for focus
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    
    # Manifold parameters
    manifold_components: int = 3
    tsne_perplexity: int = 30
    
    # Clustering parameters
    min_clusters: int = 2
    max_clusters: int = 10
    
    # General
    random_state: int = 42
    device: str = "auto"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 16]  # Simpler architecture
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class RegimeAutoencoder(nn.Module):
    """Simple autoencoder for regime factor discovery."""
    
    def __init__(self, input_dim: int, config: CoreDiscoveryConfig):
        super().__init__()
        
        # Simple encoder: market features → regime factors
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.latent_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Simple decoder: regime factors → market features
        layers = []
        prev_dim = config.latent_dim
        
        for hidden_dim in reversed(config.hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class CoreRegimeDiscovery:
    """Core regime discovery using essential ML methods."""
    
    def __init__(self, config: CoreDiscoveryConfig = None):
        self.config = config or CoreDiscoveryConfig()
        self.logger = system_logger.getChild('CoreRegimeDiscovery')
        self.device = torch.device(self.config.device)
    
    def discover_regimes(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Discover market regimes using core ML methods."""
        
        self.logger.info("🎯 Starting Core Regime Discovery")
        
        # Prepare features
        X = self._prepare_features(features)
        
        results = {}
        
        # Method 1: Autoencoder regime factors
        self.logger.info("🔧 Discovering regime factors with autoencoder")
        autoencoder_result = self._autoencoder_discovery(X)
        results['autoencoder'] = autoencoder_result
        
        # Method 2: Manifold structure discovery
        self.logger.info("📊 Discovering regime structure with manifold learning")
        manifold_result = self._manifold_discovery(X)
        results['manifold'] = manifold_result
        
        # Method 3: Direct clustering optimization
        self.logger.info("🎯 Optimizing regime clustering")
        clustering_result = self._adaptive_clustering(X)
        results['clustering'] = clustering_result
        
        # Combine results and recommend best approach
        recommendation = self._analyze_and_recommend(results)
        
        return {
            'methods': results,
            'recommendation': recommendation,
            'summary': self._create_summary(results, recommendation)
        }
    
    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime discovery."""
        # Handle missing values
        X = features.fillna(features.median()).values
        
        # Standardize for regime discovery
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X
    
    def _autoencoder_discovery(self, X: np.ndarray) -> Dict[str, Any]:
        """Discover regime factors using autoencoder."""
        
        try:
            input_dim = X.shape[1]
            model = RegimeAutoencoder(input_dim, self.config).to(self.device)
            
            # Training
            tensor_X = torch.FloatTensor(X).to(self.device)
            dataset = TensorDataset(tensor_X)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            model.train()
            final_loss = 0
            
            for epoch in range(self.config.epochs):
                epoch_loss = 0
                for (batch_X,) in dataloader:
                    optimizer.zero_grad()
                    reconstructed, encoded = model(batch_X)
                    loss = criterion(reconstructed, batch_X)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                final_loss = epoch_loss / len(dataloader)
                
                if epoch % 25 == 0:
                    self.logger.info(f"  Epoch {epoch}, Loss: {final_loss:.6f}")
            
            # Extract regime factors
            model.eval()
            with torch.no_grad():
                _, regime_factors = model(tensor_X)
                regime_factors = regime_factors.cpu().numpy()
            
            # Evaluate regime separability
            regime_quality = self._evaluate_regime_quality(regime_factors)
            
            return {
                'regime_factors': regime_factors,
                'reconstruction_loss': final_loss,
                'regime_quality': regime_quality,
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"Autoencoder discovery failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _manifold_discovery(self, X: np.ndarray) -> Dict[str, Any]:
        """Discover regime structure using manifold learning."""
        
        results = {}
        
        # t-SNE for local structure
        try:
            tsne = TSNE(
                n_components=self.config.manifold_components,
                perplexity=min(self.config.tsne_perplexity, len(X) // 4),
                random_state=self.config.random_state
            )
            tsne_embedding = tsne.fit_transform(X)
            tsne_quality = self._evaluate_regime_quality(tsne_embedding)
            
            results['tsne'] = {
                'embedding': tsne_embedding,
                'regime_quality': tsne_quality,
                'success': True
            }
        except Exception as e:
            results['tsne'] = {'error': str(e), 'success': False}
        
        # Isomap for global structure  
        try:
            isomap = Isomap(
                n_components=self.config.manifold_components,
                n_neighbors=min(15, len(X) // 4)
            )
            isomap_embedding = isomap.fit_transform(X)
            isomap_quality = self._evaluate_regime_quality(isomap_embedding)
            
            results['isomap'] = {
                'embedding': isomap_embedding,
                'regime_quality': isomap_quality,
                'success': True
            }
        except Exception as e:
            results['isomap'] = {'error': str(e), 'success': False}
        
        return results
    
    def _adaptive_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """Find optimal clustering for regime discovery."""
        
        best_score = -1
        best_result = None
        scores_by_k = {}
        
        for k in range(self.config.min_clusters, self.config.max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Multiple quality metrics
                silhouette = silhouette_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)
                
                # Composite score
                composite_score = 0.6 * silhouette + 0.4 * min(1.0, calinski_harabasz / 1000)
                scores_by_k[k] = {
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'composite': composite_score
                }
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_result = {
                        'n_clusters': k,
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'composite_score': composite_score
                    }
            
            except Exception as e:
                scores_by_k[k] = {'error': str(e)}
        
        return {
            'best_result': best_result,
            'scores_by_k': scores_by_k,
            'best_score': best_score,
            'success': best_result is not None
        }
    
    def _evaluate_regime_quality(self, regime_representation: np.ndarray) -> Dict[str, Any]:
        """Evaluate the quality of regime representation."""
        
        quality_metrics = {}
        
        # Try different cluster numbers
        best_silhouette = -1
        best_k = 2
        
        for k in range(2, min(8, len(regime_representation) // 10)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(regime_representation)
                silhouette = silhouette_score(regime_representation, labels)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
            except:
                continue
        
        quality_metrics['best_silhouette_score'] = best_silhouette
        quality_metrics['best_k'] = best_k
        
        # Regime separability assessment
        if best_silhouette > 0.4:
            quality_metrics['regime_separability'] = 'excellent'
        elif best_silhouette > 0.3:
            quality_metrics['regime_separability'] = 'good'
        elif best_silhouette > 0.1:
            quality_metrics['regime_separability'] = 'moderate'
        else:
            quality_metrics['regime_separability'] = 'poor'
        
        return quality_metrics
    
    def _analyze_and_recommend(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results and provide recommendation."""
        
        # Collect quality scores
        method_scores = {}
        
        # Autoencoder score
        if results['autoencoder'].get('success'):
            ae_quality = results['autoencoder']['regime_quality']
            method_scores['autoencoder'] = ae_quality.get('best_silhouette_score', 0)
        
        # Manifold scores
        if 'tsne' in results['manifold'] and results['manifold']['tsne'].get('success'):
            tsne_quality = results['manifold']['tsne']['regime_quality']
            method_scores['tsne'] = tsne_quality.get('best_silhouette_score', 0)
        
        if 'isomap' in results['manifold'] and results['manifold']['isomap'].get('success'):
            isomap_quality = results['manifold']['isomap']['regime_quality']
            method_scores['isomap'] = isomap_quality.get('best_silhouette_score', 0)
        
        # Direct clustering score
        if results['clustering'].get('success'):
            clustering_score = results['clustering'].get('best_score', 0)
            method_scores['direct_clustering'] = clustering_score
        
        # Find best method
        if not method_scores:
            return {
                'best_method': 'none',
                'recommendation': 'single_model_approach',
                'reason': 'No methods produced viable regimes'
            }
        
        best_method = max(method_scores.keys(), key=lambda k: method_scores[k])
        best_score = method_scores[best_method]
        
        # Generate recommendation
        if best_score > 0.4:
            recommendation = 'train_separate_models'
            reason = f'Excellent regime separation (score: {best_score:.3f})'
        elif best_score > 0.3:
            recommendation = 'train_separate_models'
            reason = f'Good regime separation (score: {best_score:.3f})'
        elif best_score > 0.1:
            recommendation = 'use_regime_features'
            reason = f'Moderate regime separation - use as features (score: {best_score:.3f})'
        else:
            recommendation = 'single_model_approach'
            reason = f'Poor regime separation (score: {best_score:.3f})'
        
        return {
            'best_method': best_method,
            'best_score': best_score,
            'method_scores': method_scores,
            'recommendation': recommendation,
            'reason': reason
        }
    
    def _create_summary(self, results: Dict[str, Any], recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of regime discovery results."""
        
        summary = {
            'regime_discovery_success': False,
            'recommended_approach': recommendation['recommendation'],
            'confidence': 'low',
            'key_insights': []
        }
        
        best_score = recommendation.get('best_score', 0)
        best_method = recommendation.get('best_method', 'none')
        
        if best_score > 0.3:
            summary['regime_discovery_success'] = True
            summary['confidence'] = 'high'
            summary['key_insights'].append(f"Strong regime structure discovered using {best_method}")
            
            if best_method == 'autoencoder':
                summary['key_insights'].append("Non-linear regime factors identified")
            elif 'manifold' in best_method:
                summary['key_insights'].append("Geometric regime structure discovered")
            elif best_method == 'direct_clustering':
                summary['key_insights'].append("Clear clustering boundaries found")
        
        elif best_score > 0.1:
            summary['regime_discovery_success'] = True
            summary['confidence'] = 'medium'
            summary['key_insights'].append("Moderate regime structure - can be used as features")
        
        else:
            summary['key_insights'].append("No clear regime structure found - use single model")
        
        # Add method-specific insights
        if results['autoencoder'].get('success'):
            ae_loss = results['autoencoder'].get('reconstruction_loss', 0)
            summary['key_insights'].append(f"Autoencoder reconstruction loss: {ae_loss:.4f}")
        
        if results['clustering'].get('success'):
            best_k = results['clustering']['best_result'].get('n_clusters', 0)
            summary['key_insights'].append(f"Optimal number of regimes: {best_k}")
        
        return summary


# Simple interface for integration
def discover_market_regimes_core(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Simple interface for core regime discovery."""
    
    discovery = CoreRegimeDiscovery()
    return discovery.discover_regimes(market_data)


# Example usage
if __name__ == "__main__":
    # Generate test data with regime structure
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create regime structure
    data = np.random.randn(n_samples, n_features)
    
    # Regime 1: High correlation between features 0-4
    regime_1 = slice(0, 300)
    factor_1 = np.random.randn(300)
    for i in range(5):
        data[regime_1, i] = factor_1 + np.random.randn(300) * 0.3
    
    # Regime 2: High correlation between features 5-9  
    regime_2 = slice(300, 700)
    factor_2 = np.random.randn(400)
    for i in range(5, 10):
        data[regime_2, i] = factor_2 + np.random.randn(400) * 0.3
    
    # Regime 3: High correlation between features 10-14
    regime_3 = slice(700, 1000)
    factor_3 = np.random.randn(300)
    for i in range(10, 15):
        data[regime_3, i] = factor_3 + np.random.randn(300) * 0.3
    
    market_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Discover regimes
    results = discover_market_regimes_core(market_data)
    
    print("🎯 Core Regime Discovery Results:")
    print(f"Success: {results['summary']['regime_discovery_success']}")
    print(f"Recommendation: {results['recommendation']['recommendation']}")
    print(f"Best Method: {results['recommendation']['best_method']}")
    print(f"Best Score: {results['recommendation']['best_score']:.3f}")
    print(f"Confidence: {results['summary']['confidence']}")
    
    print("\nKey Insights:")
    for insight in results['summary']['key_insights']:
        print(f"  • {insight}")