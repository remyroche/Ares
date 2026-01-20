"""
GMM Integration for Layer 2.5 Chaser (No Feature Selection)

This module provides direct GMM feature processing for Layer 2.5 Chaser,
including State, Shock, and Cluster features without selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import pickle
import warnings

# GMM imports
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, FastICA
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

# Project imports
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.training.steps.market_analysis.gmm_based_features import RobustGMM

# Suppress warnings
warnings.filterwarnings('ignore')


class Layer25GMMIntegration:
    """
    Direct GMM integration for Layer 2.5 Chaser without feature selection.
    
    This class processes all Layer 2.5 features through three GMM pipelines:
    1. GMM State Features: Regime probabilities, velocities, accelerations
    2. GMM Shock Features: High-conviction regime transitions  
    3. GMM Cluster Features: Overextended cluster detection
    
    All generated features are included without selection.
    """
    
    def __init__(
        self,
        n_components: int = 8,
        random_state: int = 42,
        enable_state_features: bool = True,
        enable_shock_features: bool = True,
        enable_cluster_features: bool = True,
        cache_dir: str = "artifacts/layer25_gmm_cache",
        verbose: bool = True
    ):
        """
        Initialize GMM integration for Layer 2.5.
        
        Args:
            n_components: Number of GMM components
            random_state: Random state for reproducibility
            enable_state_features: Generate GMM state features
            enable_shock_features: Generate GMM shock features
            enable_cluster_features: Generate GMM cluster features
            cache_dir: Directory for caching GMM models
            verbose: Enable verbose logging
        """
        self.n_components = n_components
        self.random_state = random_state
        self.enable_state_features = enable_state_features
        self.enable_shock_features = enable_shock_features
        self.enable_cluster_features = enable_cluster_features
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GMM models storage
        self.gmm_models = {}
        
        # Initialize transformers
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95, random_state=random_state)
        self.ica = FastICA(n_components=min(20, n_components), random_state=random_state)
        
        # Feature cache
        self._processed_features_cache = {}
        
    def _prepare_features_for_gmm(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare Layer 2.5 features for GMM processing.
        
        Args:
            X: Layer 2.5 features DataFrame
            
        Returns:
            Prepared numpy array for GMM
        """
        # Handle missing values
        X_clean = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant features
        constant_mask = X_clean.var(axis=0) > 1e-8
        X_clean = X_clean.loc[:, constant_mask]
        
        if self.verbose:
            tprint_info(f"🧹 Prepared features: {X.shape} → {X_clean.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Optional dimensionality reduction for high-dimensional data
        if X_scaled.shape[1] > 50:
            X_scaled = self.pca.fit_transform(X_scaled)
            if self.verbose:
                tprint_info(f"📊 PCA reduced to {X_scaled.shape[1]} components")
        
        return X_scaled
    
    def _generate_gmm_state_features(self, X: pd.DataFrame, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate GMM State Features: Regime probabilities, velocities, accelerations.
        
        Args:
            X: Layer 2.5 features
            returns: Returns series for anchoring (optional)
            
        Returns:
            DataFrame with GMM state features
        """
        if not self.enable_state_features:
            return pd.DataFrame(index=X.index)
        
        tprint_info("🧠 Generating GMM State Features...")
        
        # Prepare features
        X_prepared = self._prepare_features_for_gmm(X)
        
        # Fit GMM
        gmm = RobustGMM(n_components=self.n_components, random_state=self.random_state)
        gmm.fit(X_prepared)
        self.gmm_models['state'] = gmm
        
        # Get probabilities and statistics
        probs, z_fam, ent = gmm.predict(X_prepared)
        
        # Create state features DataFrame
        state_features = pd.DataFrame(index=X.index)
        
        # 1. Regime probabilities
        for i in range(probs.shape[1]):
            state_features[f'gmm_state_prob_{i}'] = probs[:, i]
        
        # 2. Regime velocities (first derivative of probabilities)
        prob_velocities = np.zeros_like(probs)
        prob_velocities[1:] = np.diff(probs, axis=0)
        
        for i in range(prob_velocities.shape[1]):
            state_features[f'gmm_state_velocity_{i}'] = prob_velocities[:, i]
        
        # 3. Regime accelerations (second derivative)
        prob_accelerations = np.zeros_like(probs)
        prob_accelerations[2:] = np.diff(prob_velocities, axis=0)
        
        for i in range(prob_accelerations.shape[1]):
            state_features[f'gmm_state_acceleration_{i}'] = prob_accelerations[:, i]
        
        # 4. Regime entropy and familiarity
        state_features['gmm_state_entropy'] = ent
        state_features['gmm_state_familiarity'] = z_fam
        
        # 5. Dominant regime and confidence
        dominant_regime = np.argmax(probs, axis=1)
        max_prob = np.max(probs, axis=1)
        
        state_features['gmm_state_dominant_regime'] = dominant_regime
        state_features['gmm_state_confidence'] = max_prob
        
        # 6. Regime transition indicators
        regime_changes = np.diff(dominant_regime) != 0
        state_features['gmm_state_transition'] = np.concatenate([[0], regime_changes.astype(int)])
        
        # 7. Regime persistence (how long current regime has been active)
        persistence = np.zeros(len(dominant_regime))
        current_persist = 0
        for i in range(len(dominant_regime)):
            if i == 0 or dominant_regime[i] != dominant_regime[i-1]:
                current_persist = 0
            current_persist += 1
            persistence[i] = current_persist
        
        state_features['gmm_state_persistence'] = persistence
        
        # 8. If returns provided, calculate regime-specific returns
        if returns is not None and len(returns) == len(X):
            returns_aligned = returns.reindex(X.index).fillna(0)
            
            for i in range(probs.shape[1]):
                regime_returns = returns_aligned * probs[:, i]
                state_features[f'gmm_state_regime_return_{i}'] = regime_returns
            
            # Regime risk (volatility within each regime)
            for i in range(probs.shape[1]):
                regime_mask = probs[:, i] > 0.5
                if regime_mask.sum() > 10:
                    regime_vol = returns_aligned.rolling(20).std().fillna(0) * regime_mask
                    state_features[f'gmm_state_regime_volatility_{i}'] = regime_vol
        
        if self.verbose:
            tprint_success(f"✅ Generated {len(state_features.columns)} GMM state features")
        
        return state_features
    
    def _generate_gmm_shock_features(self, X: pd.DataFrame, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate GMM Shock Features: High-conviction regime transitions.
        
        Args:
            X: Layer 2.5 features
            returns: Returns series for shock detection
            
        Returns:
            DataFrame with GMM shock features
        """
        if not self.enable_shock_features:
            return pd.DataFrame(index=X.index)
        
        tprint_info("⚡ Generating GMM Shock Features...")
        
        # Use state GMM if already fitted
        if 'state' not in self.gmm_models:
            X_prepared = self._prepare_features_for_gmm(X)
            gmm = RobustGMM(n_components=self.n_components, random_state=self.random_state)
            gmm.fit(X_prepared)
            self.gmm_models['state'] = gmm
        else:
            gmm = self.gmm_models['state']
            X_prepared = self._prepare_features_for_gmm(X)
        
        # Get probabilities
        probs, _, _ = gmm.predict(X_prepared)
        
        # Create shock features DataFrame
        shock_features = pd.DataFrame(index=X.index)
        
        # 1. Regime change magnitude (probability shift)
        prob_changes = np.linalg.norm(np.diff(probs, axis=0), axis=1)
        shock_features['gmm_shock_magnitude'] = np.concatenate([[0], prob_changes])
        
        # 2. Regime change direction (which regime is gaining)
        dominant_regime = np.argmax(probs, axis=1)
        regime_changes = np.diff(dominant_regime) != 0
        shock_features['gmm_shock_direction'] = np.concatenate([[0], regime_changes.astype(int)])
        
        # 3. High-conviction shocks (large probability shifts)
        high_conviction_threshold = np.percentile(prob_changes, 90)
        high_conviction_shocks = (prob_changes > high_conviction_threshold).astype(int)
        shock_features['gmm_shock_high_conviction'] = np.concatenate([[0], high_conviction_shocks])
        
        # 4. Shock persistence (how long shock effects last)
        shock_persistence = np.zeros(len(X))
        persist_counter = 0
        for i in range(1, len(X)):
            if prob_changes[i-1] > high_conviction_threshold:
                persist_counter = 5  # Shock effect lasts 5 periods
            if persist_counter > 0:
                shock_persistence[i] = persist_counter
                persist_counter -= 1
        
        shock_features['gmm_shock_persistence'] = shock_persistence
        
        # 5. Multiple regime transitions (regime instability)
        transition_window = 10
        regime_transitions = pd.Series(regime_changes).rolling(transition_window).sum().fillna(0)
        shock_features['gmm_shock_instability'] = regime_transitions
        
        # 6. Entropy shocks (sudden changes in regime uncertainty)
        _, _, ent = gmm.predict(X_prepared)
        entropy_changes = np.abs(np.diff(ent))
        entropy_shocks = (entropy_changes > np.percentile(entropy_changes, 90)).astype(int)
        shock_features['gmm_shock_entropy'] = np.concatenate([[0], entropy_shocks])
        
        # 7. If returns provided, calculate shock impact on returns
        if returns is not None and len(returns) == len(X):
            returns_aligned = returns.reindex(X.index).fillna(0)
            
            # Return shock (absolute return during regime transitions)
            return_shock = returns_aligned.abs() * regime_changes.astype(float)
            shock_features['gmm_shock_return_impact'] = return_shock
            
            # Volatility shock (volatility spike during transitions)
            vol_aligned = returns_aligned.rolling(5).std().fillna(0)
            vol_shock = vol_aligned * regime_changes.astype(float)
            shock_features['gmm_shock_volatility_impact'] = vol_shock
        
        if self.verbose:
            tprint_success(f"✅ Generated {len(shock_features.columns)} GMM shock features")
        
        return shock_features
    
    def _generate_gmm_cluster_features(self, X: pd.DataFrame, returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate GMM Cluster Features: Overextended cluster detection.
        
        Args:
            X: Layer 2.5 features
            returns: Returns series for cluster analysis
            
        Returns:
            DataFrame with GMM cluster features
        """
        if not self.enable_cluster_features:
            return pd.DataFrame(index=X.index)
        
        tprint_info("🎯 Generating GMM Cluster Features...")
        
        # Prepare features with ICA for better cluster separation
        X_prepared = self._prepare_features_for_gmm(X)
        
        # Apply ICA for independent components
        X_ica = self.ica.fit_transform(X_prepared)
        
        # Fit GMM on ICA components
        gmm = RobustGMM(n_components=self.n_components, random_state=self.random_state)
        gmm.fit(X_ica)
        self.gmm_models['cluster'] = gmm
        
        # Get probabilities and assignments
        probs, _, _ = gmm.predict(X_ica)
        cluster_assignments = np.argmax(probs, axis=1)
        
        # Create cluster features DataFrame
        cluster_features = pd.DataFrame(index=X.index)
        
        # 1. Cluster probabilities
        for i in range(probs.shape[1]):
            cluster_features[f'gmm_cluster_prob_{i}'] = probs[:, i]
        
        # 2. Cluster dominance and confidence
        cluster_features['gmm_cluster_dominant'] = cluster_assignments
        cluster_features['gmm_cluster_confidence'] = np.max(probs, axis=1)
        
        # 3. Cluster size (how many samples belong to each cluster)
        cluster_sizes = np.bincount(cluster_assignments, minlength=self.n_components)
        for i in range(self.n_components):
            cluster_features[f'gmm_cluster_size_{i}'] = cluster_sizes[i]
        
        # 4. Cluster density (local density around each point)
        nbrs = NearestNeighbors(n_neighbors=10).fit(X_ica)
        distances, indices = nbrs.kneighbors(X_ica)
        
        # Calculate local density (inverse of average distance to neighbors)
        local_density = 1.0 / (np.mean(distances, axis=1) + 1e-8)
        cluster_features['gmm_cluster_density'] = local_density
        
        # 5. Overextended detection (clusters with low density but high confidence)
        density_threshold = np.percentile(local_density, 25)
        confidence_threshold = np.percentile(np.max(probs, axis=1), 75)
        
        overextended = (local_density < density_threshold) & (np.max(probs, axis=1) > confidence_threshold)
        cluster_features['gmm_cluster_overextended'] = overextended.astype(int)
        
        # 6. Cluster isolation (how isolated a point is from other clusters)
        cluster_isolation = np.zeros(len(X))
        for i in range(len(X)):
            current_cluster = cluster_assignments[i]
            # Count neighbors in same cluster
            same_cluster_neighbors = np.sum(cluster_assignments[indices[i]] == current_cluster)
            cluster_isolation[i] = 1.0 - (same_cluster_neighbors / 10.0)  # 10 neighbors
        
        cluster_features['gmm_cluster_isolation'] = cluster_isolation
        
        # 7. Cluster transition probability
        cluster_transitions = np.zeros(len(X))
        for i in range(1, len(X)):
            # Probability of transitioning from previous cluster to current
            prev_cluster = cluster_assignments[i-1]
            curr_cluster = cluster_assignments[i]
            if prev_cluster != curr_cluster:
                cluster_transitions[i] = probs[i, curr_cluster]  # Confidence in new cluster
        
        cluster_features['gmm_cluster_transition_prob'] = cluster_transitions
        
        # 8. If returns provided, calculate cluster-specific performance
        if returns is not None and len(returns) == len(X):
            returns_aligned = returns.reindex(X.index).fillna(0)
            
            # Cluster-specific returns
            for i in range(self.n_components):
                cluster_mask = cluster_assignments == i
                cluster_returns = returns_aligned * cluster_mask.astype(float)
                cluster_features[f'gmm_cluster_return_{i}'] = cluster_returns
            
            # Cluster risk (returns volatility within cluster)
            for i in range(self.n_components):
                cluster_mask = cluster_assignments == i
                if cluster_mask.sum() > 10:
                    cluster_vol = returns_aligned.rolling(20).std().fillna(0) * cluster_mask.astype(float)
                    cluster_features[f'gmm_cluster_volatility_{i}'] = cluster_vol
        
        if self.verbose:
            tprint_success(f"✅ Generated {len(cluster_features.columns)} GMM cluster features")
        
        return cluster_features
    
    def process_layer25_features(
        self, 
        X: pd.DataFrame, 
        returns: Optional[pd.Series] = None,
        cache_key: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process Layer 2.5 features through all GMM pipelines.
        
        Args:
            X: Layer 2.5 features DataFrame
            returns: Returns series for GMM anchoring
            cache_key: Optional cache key for storing results
            
        Returns:
            Tuple of (enhanced_features, metadata)
        """
        # Check cache
        if cache_key and cache_key in self._processed_features_cache:
            if self.verbose:
                tprint_info(f"💾 Using cached GMM features for {cache_key}")
            return self._processed_features_cache[cache_key]
        
        tprint_info(f"🚀 Processing Layer 2.5 features through GMM pipelines...")
        tprint_info(f"📊 Input features: {X.shape}")
        
        # Generate GMM features
        start_time = pd.Timestamp.now()
        
        # 1. State features
        state_features = self._generate_gmm_state_features(X, returns)
        
        # 2. Shock features  
        shock_features = self._generate_gmm_shock_features(X, returns)
        
        # 3. Cluster features
        cluster_features = self._generate_gmm_cluster_features(X, returns)
        
        # Combine all GMM features (no selection)
        all_gmm_features = []
        
        if not state_features.empty:
            all_gmm_features.append(state_features)
        
        if not shock_features.empty:
            all_gmm_features.append(shock_features)
        
        if not cluster_features.empty:
            all_gmm_features.append(cluster_features)
        
        # Combine with original features
        if all_gmm_features:
            gmm_combined = pd.concat(all_gmm_features, axis=1)
            enhanced_features = pd.concat([X, gmm_combined], axis=1)
        else:
            enhanced_features = X.copy()
        
        # Create metadata
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        metadata = {
            'original_features': X.shape[1],
            'gmm_state_features': len(state_features.columns),
            'gmm_shock_features': len(shock_features.columns),
            'gmm_cluster_features': len(cluster_features.columns),
            'total_gmm_features': len(state_features.columns) + len(shock_features.columns) + len(cluster_features.columns),
            'final_features': enhanced_features.shape[1],
            'processing_time_seconds': processing_time,
            'n_components': self.n_components
        }
        
        # Cache results
        if cache_key:
            self._processed_features_cache[cache_key] = (enhanced_features, metadata)
        
        if self.verbose:
            tprint_success(f"✅ GMM processing complete:")
            tprint_info(f"   📊 Original: {metadata['original_features']} → Final: {metadata['final_features']} features")
            tprint_info(f"   ⏱️  Processing time: {processing_time:.2f}s")
            tprint_info(f"   🧠 State: {metadata['gmm_state_features']}, ⚡ Shock: {metadata['gmm_shock_features']}, 🎯 Cluster: {metadata['gmm_cluster_features']}")
        
        return enhanced_features, metadata
    
    def get_regime_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get regime probabilities from the fitted GMM model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with regime probabilities
        """
        if 'state' not in self.gmm_models:
            tprint_warning("⚠️ No state GMM model fitted. Call process_layer25_features first.")
            return pd.DataFrame()
        
        X_prepared = self._prepare_features_for_gmm(X)
        probs, _, _ = self.gmm_models['state'].predict(X_prepared)
        
        regime_probs = pd.DataFrame(
            probs, 
            index=X.index,
            columns=[f"regime_{i}" for i in range(probs.shape[1])]
        )
        
        return regime_probs
    
    def save_models(self, filepath: str):
        """Save fitted GMM models and configuration."""
        save_data = {
            'gmm_models': self.gmm_models,
            'scaler': self.scaler,
            'pca': self.pca,
            'ica': self.ica,
            'config': {
                'n_components': self.n_components,
                'random_state': self.random_state,
                'enable_state_features': self.enable_state_features,
                'enable_shock_features': self.enable_shock_features,
                'enable_cluster_features': self.enable_cluster_features,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            tprint_success(f"💾 GMM models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load fitted GMM models and configuration."""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.gmm_models = save_data['gmm_models']
            self.scaler = save_data['scaler']
            self.pca = save_data['pca']
            self.ica = save_data['ica']
            
            # Restore configuration
            config = save_data['config']
            self.n_components = config['n_components']
            self.random_state = config['random_state']
            self.enable_state_features = config['enable_state_features']
            self.enable_shock_features = config['enable_shock_features']
            self.enable_cluster_features = config['enable_cluster_features']
            
            if self.verbose:
                tprint_success(f"📥 GMM models loaded from {filepath}")
                
        except Exception as e:
            tprint_error(f"❌ Failed to load GMM models: {e}")


# Convenience function for direct integration
def enhance_layer25_with_gmm(
    X: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    n_components: int = 8,
    cache_models: bool = True,
    model_cache_path: str = "artifacts/layer25_gmm_models.pkl"
) -> Tuple[pd.DataFrame, Layer25GMMIntegration]:
    """
    Convenience function to enhance Layer 2.5 features with GMM (no selection).
    
    Args:
        X: Layer 2.5 features
        returns: Returns series for GMM anchoring
        n_components: Number of GMM components
        cache_models: Whether to cache fitted models
        model_cache_path: Path for model cache
        
    Returns:
        Tuple of (enhanced_features, gmm_integration)
    """
    # Create GMM integration
    integration = Layer25GMMIntegration(
        n_components=n_components,
        verbose=True
    )
    
    # Process features
    enhanced_features, metadata = integration.process_layer25_features(
        X, returns, cache_key="layer25_enhanced"
    )
    
    # Save models if requested
    if cache_models:
        integration.save_models(model_cache_path)
    
    return enhanced_features, integration