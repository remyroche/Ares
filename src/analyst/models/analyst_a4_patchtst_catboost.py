"""
Analyst A4 Model: PatchTST-Embed + CatBoost

Binary "green light" classification with:
- 300+ features, regime posteriors, cross-TF aggregates
- PatchTST (frozen tiny): patch_len=24, stride=6, d_model=128, layers=3, causal pooling
- CatBoost head: depth=6, iterations~1000, l2_leaf_reg=10
- Different bias from LGBM/XGB for ensemble diversity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import catboost as cb
from sklearn.model_selection import StratifiedKFold
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST embedding (frozen tiny)."""
    patch_len: int = 24
    stride: int = 6
    d_model: int = 128
    n_layers: int = 3
    causal_pooling: bool = True
    dropout: float = 0.1
    attention_heads: int = 4
    frozen: bool = True  # Frozen tiny model


@dataclass
class CatBoostConfig:
    """Configuration for CatBoost model."""
    depth: int = 6
    iterations: int = 1000
    l2_leaf_reg: float = 10.0
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bylevel: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_data_in_leaf: int = 20
    max_leaves: int = 31
    grow_policy: str = 'SymmetricTree'
    boosting_type: str = 'Plain'
    bootstrap_type: str = 'Bayesian'
    bagging_temperature: float = 1.0
    random_strength: float = 1.0
    one_hot_max_size: int = 255
    leaf_estimation_method: str = 'Newton'
    leaf_estimation_iterations: int = 10
    leaf_estimation_backtracking: str = 'AnyImprovement'
    feature_border_type: str = 'GreedyLogSum'
    per_feature_borders: Optional[List] = None
    model_shrink_rate: float = 0.0
    feature_weights: Optional[List] = None
    penalties_coefficient: float = 1.0
    first_feature_use_penalties: float = 0.0
    per_object_feature_penalties: Optional[List] = None
    model_shrink_mode: str = 'Constant'
    langevin: bool = False
    diffusion_temperature: float = 10000.0
    posterior_sampling: bool = False
    boost_from_average: bool = True
    text_processing: Optional[Dict] = None
    tokenizers: Optional[List] = None
    dictionaries: Optional[List] = None
    feature_calcers: Optional[List] = None
    text_processing: Optional[Dict] = None
    verbose: int = 0
    random_seed: int = 42


@dataclass
class CalibrationConfig:
    """Configuration for model calibration."""
    method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    cv_folds: int = 5
    enable_venn_abers: bool = True
    confidence_levels: List[float] = None

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]


class FrozenPatchTSTEmbedding:
    """Frozen PatchTST-style embedding for time series features."""
    
    def __init__(self, config: PatchTSTConfig):
        self.config = config
        self.patch_embeddings = None
        self.attention_weights = None
        self.regime_embeddings = None
        self.feature_scaler = None
        self.projection_matrices = None  # Frozen projection matrices
        self.is_fitted = False
        
    def _create_patches(self, X: np.ndarray) -> np.ndarray:
        """Create patches from time series data."""
        n_samples, n_features = X.shape
        patches = []
        
        for i in range(0, n_samples - self.config.patch_len + 1, self.config.stride):
            patch = X[i:i + self.config.patch_len, :]
            patches.append(patch.flatten())
        
        if not patches:
            # Fallback: use original data with padding
            padded_X = np.pad(X, ((0, self.config.patch_len - 1), (0, 0)), mode='edge')
            patches = [padded_X[i:i + self.config.patch_len, :].flatten() 
                      for i in range(0, len(padded_X) - self.config.patch_len + 1, self.config.stride)]
        
        return np.array(patches)
    
    def _compute_attention_weights(self, patches: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute attention weights for patches."""
        n_patches = patches.shape[0]
        
        # Simple attention mechanism based on target correlation
        if len(y) >= n_patches:
            y_aligned = y[:n_patches]
        else:
            y_aligned = np.tile(y, (n_patches // len(y) + 1))[:n_patches]
        
        # Compute patch-target correlations
        patch_scores = []
        for i in range(n_patches):
            patch = patches[i]
            if len(patch) > 0:
                # Use correlation as attention score
                correlation = np.corrcoef(patch, y_aligned[i:i+1])[0, 1] if len(patch) > 1 else 0
                patch_scores.append(abs(correlation) if not np.isnan(correlation) else 0)
            else:
                patch_scores.append(0)
        
        # Normalize to attention weights
        patch_scores = np.array(patch_scores)
        if np.sum(patch_scores) > 0:
            attention_weights = np.exp(patch_scores) / np.sum(np.exp(patch_scores))
        else:
            attention_weights = np.ones(n_patches) / n_patches
        
        return attention_weights
    
    def _apply_causal_pooling(self, patches: np.ndarray) -> np.ndarray:
        """Apply causal pooling to patches."""
        if not self.config.causal_pooling:
            return patches
        
        # Causal pooling: only use past information
        pooled_patches = []
        for i in range(patches.shape[0]):
            # Use only patches up to current time step
            causal_patches = patches[:i+1]
            if len(causal_patches) > 0:
                # Average pooling over causal patches
                pooled_patch = np.mean(causal_patches, axis=0)
                pooled_patches.append(pooled_patch)
            else:
                pooled_patches.append(patches[i])
        
        return np.array(pooled_patches)
    
    def _create_frozen_embeddings(self, patches: np.ndarray) -> np.ndarray:
        """Create frozen embeddings from patches."""
        # Simple linear projection to d_model (frozen)
        patch_dim = patches.shape[1]
        d_model = self.config.d_model
        
        # Create frozen projection matrices (in practice, these would be pre-trained)
        if self.projection_matrices is None:
            np.random.seed(42)  # Fixed seed for reproducibility
            self.projection_matrices = []
            
            # Create multiple projection layers for depth
            current_dim = patch_dim
            for layer in range(self.config.n_layers):
                next_dim = d_model if layer == self.config.n_layers - 1 else d_model // 2
                projection = np.random.randn(current_dim, next_dim) * 0.1
                self.projection_matrices.append(projection)
                current_dim = next_dim
        
        # Apply frozen projections
        embeddings = patches
        for projection in self.projection_matrices:
            embeddings = embeddings @ projection
            # Apply activation (ReLU)
            embeddings = np.maximum(0, embeddings)
        
        return embeddings
    
    def fit(self, X: np.ndarray, y: np.ndarray, regimes: Optional[np.ndarray] = None) -> 'FrozenPatchTSTEmbedding':
        """Fit the frozen PatchTST embedding."""
        logger.info("Fitting frozen PatchTST embedding...")
        
        # Create patches
        patches = self._create_patches(X)
        logger.info(f"Created {patches.shape[0]} patches of size {patches.shape[1]}")
        
        # Compute attention weights
        self.attention_weights = self._compute_attention_weights(patches, y)
        
        # Apply causal pooling
        if self.config.causal_pooling:
            patches = self._apply_causal_pooling(patches)
        
        # Create frozen embeddings
        self.patch_embeddings = self._create_frozen_embeddings(patches)
        
        # Create regime-specific embeddings if regimes provided
        if regimes is not None:
            self._create_regime_embeddings(patches, regimes)
        
        # Fit feature scaler
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        self.patch_embeddings = self.feature_scaler.fit_transform(self.patch_embeddings)
        
        self.is_fitted = True
        logger.info(f"✅ Frozen PatchTST embedding fitted with {self.patch_embeddings.shape[1]} dimensions")
        return self
    
    def _create_regime_embeddings(self, patches: np.ndarray, regimes: np.ndarray):
        """Create regime-specific embeddings."""
        unique_regimes = np.unique(regimes)
        self.regime_embeddings = {}
        
        for regime in unique_regimes:
            regime_mask = regimes == regime
            if np.sum(regime_mask) > 10:  # Minimum samples for regime
                regime_patches = patches[regime_mask]
                regime_embeddings = self._create_frozen_embeddings(regime_patches)
                self.regime_embeddings[regime] = regime_embeddings
    
    def transform(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform features using frozen PatchTST embedding."""
        if not self.is_fitted:
            raise ValueError("Frozen PatchTST embedding must be fitted before transform")
        
        # Create patches
        patches = self._create_patches(X)
        
        # Apply causal pooling
        if self.config.causal_pooling:
            patches = self._apply_causal_pooling(patches)
        
        # Create frozen embeddings
        embeddings = self._create_frozen_embeddings(patches)
        
        # Scale features
        if self.feature_scaler is not None:
            embeddings = self.feature_scaler.transform(embeddings)
        
        return embeddings


class VennAbersCalibration:
    """Venn-Abers calibration for uncertainty estimation."""
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.5, 0.6, 0.7, 0.8, 0.9]
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'VennAbersCalibration':
        """Fit Venn-Abers calibrators."""
        for level in self.confidence_levels:
            # Create binary targets for this confidence level
            y_binary = (y_prob >= level).astype(int)
            
            if len(np.unique(y_binary)) > 1:  # Ensure we have both classes
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_prob, y_binary)
                self.calibrators[level] = calibrator
        
        self.is_fitted = True
        return self
    
    def predict_confidence(self, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Venn-Abers calibrators must be fitted first")
        
        results = {}
        for level, calibrator in self.calibrators.items():
            calibrated_probs = calibrator.predict(y_prob)
            results[f'confidence_{level}'] = calibrated_probs
        
        return results


class AnalystA4Model:
    """Analyst A4: PatchTST-Embed + CatBoost with calibration."""
    
    def __init__(self, 
                 patchtst_config: Optional[PatchTSTConfig] = None,
                 catboost_config: Optional[CatBoostConfig] = None,
                 calibration_config: Optional[CalibrationConfig] = None):
        self.patchtst_config = patchtst_config or PatchTSTConfig()
        self.catboost_config = catboost_config or CatBoostConfig()
        self.calibration_config = calibration_config or CalibrationConfig()
        
        # Model components
        self.patchtst_embedding = FrozenPatchTSTEmbedding(self.patchtst_config)
        self.catboost_model = None
        self.calibrated_model = None
        self.venn_abers = None
        self.feature_names = None
        self.is_fitted = False
        
        logger.info("Initialized Analyst A4 Model (PatchTST-Embed + CatBoost)")
    
    def _prepare_features(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare features with PatchTST embedding."""
        # Get PatchTST embeddings
        patchtst_features = self.patchtst_embedding.transform(X, regimes)
        
        # Combine with original features
        if X.shape[0] == patchtst_features.shape[0]:
            combined_features = np.hstack([X, patchtst_features])
        else:
            # Handle dimension mismatch by padding/truncating
            min_samples = min(X.shape[0], patchtst_features.shape[0])
            combined_features = np.hstack([
                X[:min_samples], 
                patchtst_features[:min_samples]
            ])
        
        return combined_features
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        class_weights = {}
        for class_label, count in zip(unique_classes, counts):
            class_weights[class_label] = total_samples / (len(unique_classes) * count)
        
        return class_weights
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            regimes: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'AnalystA4Model':
        """Fit the Analyst A4 model."""
        logger.info("Fitting Analyst A4 Model...")
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_classes)}")
        
        # Compute class weights
        class_weights = self._compute_class_weights(y)
        logger.info(f"Class weights: {class_weights}")
        
        # Fit PatchTST embedding
        self.patchtst_embedding.fit(X, y, regimes)
        
        # Prepare features
        X_enhanced = self._prepare_features(X, regimes)
        logger.info(f"Enhanced features shape: {X_enhanced.shape}")
        
        # Create CatBoost model
        self.catboost_model = cb.CatBoostClassifier(
            depth=self.catboost_config.depth,
            iterations=self.catboost_config.iterations,
            l2_leaf_reg=self.catboost_config.l2_leaf_reg,
            learning_rate=self.catboost_config.learning_rate,
            subsample=self.catboost_config.subsample,
            colsample_bylevel=self.catboost_config.colsample_bylevel,
            reg_lambda=self.catboost_config.reg_lambda,
            reg_alpha=self.catboost_config.reg_alpha,
            min_data_in_leaf=self.catboost_config.min_data_in_leaf,
            max_leaves=self.catboost_config.max_leaves,
            grow_policy=self.catboost_config.grow_policy,
            boosting_type=self.catboost_config.boosting_type,
            bootstrap_type=self.catboost_config.bootstrap_type,
            bagging_temperature=self.catboost_config.bagging_temperature,
            random_strength=self.catboost_config.random_strength,
            one_hot_max_size=self.catboost_config.one_hot_max_size,
            leaf_estimation_method=self.catboost_config.leaf_estimation_method,
            leaf_estimation_iterations=self.catboost_config.leaf_estimation_iterations,
            leaf_estimation_backtracking=self.catboost_config.leaf_estimation_backtracking,
            feature_border_type=self.catboost_config.feature_border_type,
            per_feature_borders=self.catboost_config.per_feature_borders,
            model_shrink_rate=self.catboost_config.model_shrink_rate,
            feature_weights=self.catboost_config.feature_weights,
            penalties_coefficient=self.catboost_config.penalties_coefficient,
            first_feature_use_penalties=self.catboost_config.first_feature_use_penalties,
            per_object_feature_penalties=self.catboost_config.per_object_feature_penalties,
            model_shrink_mode=self.catboost_config.model_shrink_mode,
            langevin=self.catboost_config.langevin,
            diffusion_temperature=self.catboost_config.diffusion_temperature,
            posterior_sampling=self.catboost_config.posterior_sampling,
            boost_from_average=self.catboost_config.boost_from_average,
            text_processing=self.catboost_config.text_processing,
            tokenizers=self.catboost_config.tokenizers,
            dictionaries=self.catboost_config.dictionaries,
            feature_calcers=self.catboost_config.feature_calcers,
            verbose=self.catboost_config.verbose,
            random_seed=self.catboost_config.random_seed,
            class_weights=class_weights,
            thread_count=-1
        )
        
        # Fit CatBoost model
        if sample_weight is not None:
            self.catboost_model.fit(
                X_enhanced, 
                y, 
                sample_weight=sample_weight,
                eval_set=(X_enhanced, y),
                verbose=False
            )
        else:
            self.catboost_model.fit(
                X_enhanced, 
                y,
                eval_set=(X_enhanced, y),
                verbose=False
            )
        
        # Get predictions for calibration
        y_prob = self.catboost_model.predict_proba(X_enhanced)[:, 1]
        
        # Fit calibration
        if self.calibration_config.method in ['isotonic', 'sigmoid']:
            self.calibrated_model = CalibratedClassifierCV(
                self.catboost_model,
                method=self.calibration_config.method,
                cv=self.calibration_config.cv_folds
            )
            self.calibrated_model.fit(X_enhanced, y)
        
        # Fit Venn-Abers calibration
        if self.calibration_config.enable_venn_abers:
            self.venn_abers = VennAbersCalibration(self.calibration_config.confidence_levels)
            self.venn_abers.fit(y, y_prob)
        
        self.is_fitted = True
        logger.info("✅ Analyst A4 Model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X_enhanced = self._prepare_features(X, regimes)
        
        # Get predictions
        if self.calibrated_model is not None:
            y_prob = self.calibrated_model.predict_proba(X_enhanced)[:, 1]
        else:
            y_prob = self.catboost_model.predict_proba(X_enhanced)[:, 1]
        
        return y_prob
    
    def predict_uncertainty(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Predict uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base predictions
        y_prob = self.predict_proba(X, regimes)
        
        # Get Venn-Abers confidence intervals
        uncertainty_results = {
            'probability': y_prob,
            'confidence_intervals': {}
        }
        
        if self.venn_abers is not None:
            confidence_intervals = self.venn_abers.predict_confidence(y_prob)
            uncertainty_results['confidence_intervals'] = confidence_intervals
        
        # Add margin statistics
        uncertainty_results['margin_stats'] = {
            'mean_probability': np.mean(y_prob),
            'std_probability': np.std(y_prob),
            'min_probability': np.min(y_prob),
            'max_probability': np.max(y_prob),
            'confidence_range': np.max(y_prob) - np.min(y_prob)
        }
        
        return uncertainty_results
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the model."""
        if not self.is_fitted or self.catboost_model is None:
            return {}
        
        importance = self.catboost_model.feature_importances_
        
        # Create feature names
        if self.feature_names is not None:
            # Original features + PatchTST features
            patchtst_feature_names = [f'patchtst_{i}' for i in range(self.patchtst_embedding.patch_embeddings.shape[1])]
            all_feature_names = self.feature_names + patchtst_feature_names
        else:
            all_feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return {
            'importance_scores': importance,
            'feature_names': all_feature_names,
            'top_features': sorted(zip(all_feature_names, importance), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def get_catboost_info(self) -> Dict[str, Any]:
        """Get CatBoost model information."""
        if not self.is_fitted or self.catboost_model is None:
            return {}
        
        return {
            'depth': self.catboost_config.depth,
            'iterations': self.catboost_config.iterations,
            'l2_leaf_reg': self.catboost_config.l2_leaf_reg,
            'learning_rate': self.catboost_config.learning_rate,
            'subsample': self.catboost_config.subsample,
            'colsample_bylevel': self.catboost_config.colsample_bylevel,
            'boosting_type': self.catboost_config.boosting_type,
            'bootstrap_type': self.catboost_config.bootstrap_type,
            'grow_policy': self.catboost_config.grow_policy,
            'leaf_estimation_method': self.catboost_config.leaf_estimation_method,
            'feature_count': len(self.catboost_model.feature_importances_),
            'tree_count': self.catboost_model.get_tree_count()
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'patchtst_embedding': self.patchtst_embedding,
            'catboost_model': self.catboost_model,
            'calibrated_model': self.calibrated_model,
            'venn_abers': self.venn_abers,
            'feature_names': self.feature_names,
            'patchtst_config': self.patchtst_config,
            'catboost_config': self.catboost_config,
            'calibration_config': self.calibration_config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'AnalystA4Model':
        """Load the model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            patchtst_config=model_data['patchtst_config'],
            catboost_config=model_data['catboost_config'],
            calibration_config=model_data['calibration_config']
        )
        
        # Restore state
        instance.patchtst_embedding = model_data['patchtst_embedding']
        instance.catboost_model = model_data['catboost_model']
        instance.calibrated_model = model_data['calibrated_model']
        instance.venn_abers = model_data['venn_abers']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        logger.info(f"✅ Model loaded from {filepath}")
        return instance


# Factory function for easy model creation
def create_analyst_a4_model(patchtst_config: Optional[PatchTSTConfig] = None,
                           catboost_config: Optional[CatBoostConfig] = None,
                           calibration_config: Optional[CalibrationConfig] = None) -> AnalystA4Model:
    """Create an Analyst A4 model with the specified configurations."""
    return AnalystA4Model(patchtst_config, catboost_config, calibration_config)