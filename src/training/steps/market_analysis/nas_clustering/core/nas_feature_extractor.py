"""
NAS Feature Extractor

Neural Architecture Search feature extraction for market analysis.
"""

print("🔍 [NAS_FEATURE_EXTRACTOR] Loading NAS Feature Extractor module")
print("🔍 [NAS_FEATURE_EXTRACTOR] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/nas_feature_extractor.py")
print("🔍 [NAS_FEATURE_EXTRACTOR] Purpose: Neural Architecture Search feature extraction for market analysis")
print("🔍 [NAS_FEATURE_EXTRACTOR] Status: Starting module import")

import numpy as np
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ NumPy imported successfully")

import pandas as pd
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ Pandas imported successfully")

from typing import Dict, List, Any, Optional, Tuple
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ Typing imports completed")

import logging
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ Logging imported successfully")

from sklearn.preprocessing import StandardScaler
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ StandardScaler imported successfully")

from sklearn.decomposition import PCA
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ PCA imported successfully")

import torch
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ PyTorch imported successfully")

import torch.nn as nn
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ PyTorch neural network module imported successfully")

# Import existing feature selection framework
print("🔍 [NAS_FEATURE_EXTRACTOR] Attempting to import feature selection framework...")
try:
    from src.feature_selection.core.framework import (
        select_features, run_comprehensive_feature_selection
    )
    from src.feature_selection.methods import (
        MRMRSelector, ElasticNetStabilitySelector, RecursiveFeatureEliminator
    )
    FEATURE_SELECTION_AVAILABLE = True
    print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ Feature selection framework imported successfully")
    print("🔍 [NAS_FEATURE_EXTRACTOR] Available methods: select_features, run_comprehensive_feature_selection")
    print("🔍 [NAS_FEATURE_EXTRACTOR] Available selectors: MRMRSelector, ElasticNetStabilitySelector, RecursiveFeatureEliminator")
except ImportError as e:
    FEATURE_SELECTION_AVAILABLE = False
    print(f"🔍 [NAS_FEATURE_EXTRACTOR] ❌ Feature selection framework import failed: {e}")
    print("🔍 [NAS_FEATURE_EXTRACTOR] Falling back to sklearn feature selection")
    # Fallback to sklearn
    from sklearn.feature_selection import SelectKBest, f_regression
    print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ Sklearn feature selection imported as fallback")

logger = logging.getLogger(__name__)
print("🔍 [NAS_FEATURE_EXTRACTOR] ✓ Logger initialized")
print("🔍 [NAS_FEATURE_EXTRACTOR] All imports completed successfully")

class NASFeatureExtractor:
    """Neural Architecture Search Feature Extractor for market analysis."""

    def __init__(self,
                 feature_dim: int = 128,
                 n_components: int = 64,
                 selection_k: int = 32,
                 device: str = 'cpu'):
        """Initialize NAS Feature Extractor.

        Args:
            feature_dim: Dimension of extracted features
            n_components: Number of PCA components
            selection_k: Number of features to select
            device: Computation device
        """
        print("🔍 [NAS_FEATURE_EXTRACTOR_INIT] Initializing NASFeatureExtractor")
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT] Parameters received:")
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT]   - feature_dim: {feature_dim}")
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT]   - n_components: {n_components}")
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT]   - selection_k: {selection_k}")
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT]   - device: {device}")

        self.feature_dim = feature_dim
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT] ✓ feature_dim set to: {self.feature_dim}")

        self.n_components = n_components
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT] ✓ n_components set to: {self.n_components}")

        self.selection_k = selection_k
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT] ✓ selection_k set to: {self.selection_k}")

        self.device = device
        print(f"🔍 [NAS_FEATURE_EXTRACTOR_INIT] ✓ device set to: {self.device}")

        # Feature processing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

        # Use existing feature selection framework if available
        if FEATURE_SELECTION_AVAILABLE:
            self.feature_selector = None  # Will be created dynamically
            self.feature_selection_method = 'comprehensive'
        else:
            # Fallback to sklearn
            from sklearn.feature_selection import SelectKBest, f_regression
            self.feature_selector = SelectKBest(score_func=f_regression, k=selection_k)
            self.feature_selection_method = 'sklearn_fallback'

        # Neural feature extraction
        self.feature_network = self._build_feature_network()
        self.is_fitted = False

        logger.info(f"NASFeatureExtractor initialized with feature_dim={feature_dim}")

    def _build_feature_network(self) -> nn.Module:
        """Build neural network for feature extraction."""
        return nn.Sequential(
            nn.Linear(self.selection_k, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.feature_dim),
            nn.Tanh()
        ).to(self.device)

    def extract_features(self,
                        data: np.ndarray,
                        target: Optional[np.ndarray] = None,
                        fit_transform: bool = True) -> np.ndarray:
        """Extract features from input data using NAS methodology.

        Args:
            data: Input data array (n_samples, n_features)
            target: Target values for supervised feature selection
            fit_transform: Whether to fit transformers on this data

        Returns:
            Extracted features array (n_samples, feature_dim)
        """
        logger.info(f"Extracting features from data shape: {data.shape}")

        try:
            # Step 1: Standardize features
            if fit_transform:
                data_scaled = self.scaler.fit_transform(data)
            else:
                data_scaled = self.scaler.transform(data)

            # Step 2: Feature selection using existing framework
            if target is not None and fit_transform:
                data_selected = self._perform_feature_selection(data_scaled, target, fit_transform)
            elif target is not None:
                data_selected = self._perform_feature_selection(data_scaled, target, fit_transform)
            else:
                data_selected = data_scaled[:, :self.selection_k]

            # Step 3: PCA dimensionality reduction
            if fit_transform:
                data_pca = self.pca.fit_transform(data_selected)
            else:
                data_pca = self.pca.transform(data_selected)

            # Step 4: Neural feature extraction
            data_tensor = torch.tensor(data_pca, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                features = self.feature_network(data_tensor)

            features_np = features.cpu().numpy()

            if fit_transform:
                self.is_fitted = True

            logger.info(f"Feature extraction completed. Output shape: {features_np.shape}")
            return features_np

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return basic features as fallback
            return self._extract_basic_features(data)

    def _extract_basic_features(self, data: np.ndarray) -> np.ndarray:
        """Extract basic statistical features as fallback."""
        logger.warning("Using basic feature extraction as fallback")

        # Basic statistical features
        features = []

        # Mean and std
        features.append(np.mean(data, axis=1))
        features.append(np.std(data, axis=1))

        # Min and max
        features.append(np.min(data, axis=1))
        features.append(np.max(data, axis=1))

        # Percentiles
        features.append(np.percentile(data, 25, axis=1))
        features.append(np.percentile(data, 75, axis=1))

        # Combine features
        basic_features = np.column_stack(features)

        # Pad or truncate to target dimension
        if basic_features.shape[1] < self.feature_dim:
            padding = np.zeros((basic_features.shape[0], self.feature_dim - basic_features.shape[1]))
            basic_features = np.column_stack([basic_features, padding])
        else:
            basic_features = basic_features[:, :self.feature_dim]

        return basic_features

    def _perform_feature_selection(self, data: np.ndarray, target: np.ndarray, fit_transform: bool) -> np.ndarray:
        """Perform feature selection using the existing framework."""
        try:
            if FEATURE_SELECTION_AVAILABLE and fit_transform:
                # Use comprehensive feature selection
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]

                # Use the existing framework for feature selection
                selection_result = select_features(
                    data, target,
                    method='comprehensive',
                    max_features=self.selection_k,
                    feature_names=feature_names
                )

                if selection_result.get('success', False) and selection_result.get('selected_features'):
                    selected_indices = selection_result.get('selected_indices', [])
                    if selected_indices:
                        return data[:, selected_indices]

                # Fallback to top k features if selection fails
                logger.warning("Feature selection failed, using top k features")
                return data[:, :self.selection_k]

            elif FEATURE_SELECTION_AVAILABLE and not fit_transform:
                # For transform, we need to use the fitted selector
                if hasattr(self, 'selected_indices') and self.selected_indices is not None:
                    return data[:, self.selected_indices]
                else:
                    return data[:, :self.selection_k]

            else:
                # Fallback to sklearn
                if fit_transform:
                    data_selected = self.feature_selector.fit_transform(data, target)
                else:
                    data_selected = self.feature_selector.transform(data)
                return data_selected

        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return data[:, :self.selection_k]

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            logger.warning("Feature extractor not fitted yet")
            return np.array([])

        try:
            # Get PCA component importance
            pca_importance = np.abs(self.pca.components_).mean(axis=0)

            # Get feature selection scores
            if hasattr(self.feature_selector, 'scores_'):
                selection_scores = self.feature_selector.scores_
            else:
                selection_scores = np.ones(self.selection_k)

            # Combine importance scores
            importance = pca_importance * selection_scores
            return importance / np.sum(importance)  # Normalize

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return np.ones(self.feature_dim) / self.feature_dim

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return [f"nas_feature_{i}" for i in range(self.feature_dim)]

    def save_model(self, filepath: str):
        """Save the feature extractor model."""
        try:
            import joblib

            model_data = {
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_selector': self.feature_selector,
                'feature_network_state': self.feature_network.state_dict(),
                'feature_dim': self.feature_dim,
                'n_components': self.n_components,
                'selection_k': self.selection_k,
                'device': self.device,
                'is_fitted': self.is_fitted
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Feature extractor saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save feature extractor: {e}")

    def load_model(self, filepath: str):
        """Load the feature extractor model."""
        try:
            import joblib

            model_data = joblib.load(filepath)

            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.feature_selector = model_data['feature_selector']
            self.feature_dim = model_data['feature_dim']
            self.n_components = model_data['n_components']
            self.selection_k = model_data['selection_k']
            self.device = model_data['device']
            self.is_fitted = model_data['is_fitted']

            # Rebuild and load network
            self.feature_network = self._build_feature_network()
            self.feature_network.load_state_dict(model_data['feature_network_state'])

            logger.info(f"Feature extractor loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load feature extractor: {e}")
            raise
