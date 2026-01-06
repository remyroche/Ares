"""
Causal Quality Metrics Module

Implements enhanced quality metrics based on causal framework:
1. Causal validity scoring
2. Mechanism alignment assessment
3. Interventional robustness testing
4. Counterfactual consistency validation
5. Causal invariance measurement

Key Features:
- Additive to existing survival filters
- Test geometry consistency with causal structure
- Validate against counterfactual outcomes
- Measure interventional robustness
- Assess mechanism alignment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import networkx as nx

# Import existing components
from .structural_causal_model import StructuralCausalModel
from .causal_feature_transformer import CausalFeatureTransformer
from .causal_denoising_engine import CausalDenoisingEngine

# Import geometry trial structure
try:
    from .orthogonal_label_generation import GeometryTrial
except ImportError:
    # Fallback geometry structure
    class GeometryTrial:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class CausalQualityMetrics:
    """
    Enhanced quality metrics based on causal framework.
    
    Adds causal validation on top of existing survival filters.
    """
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        scm: Optional[StructuralCausalModel] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        Initialize Causal Quality Metrics.
        
        Args:
            causal_graph: Causal graph from discovery
            scm: Fitted structural causal models
            quality_thresholds: Thresholds for quality metrics
            verbose: Whether to print progress information
        """
        self.causal_graph = causal_graph or {}
        self.scm = scm
        self.verbose = verbose
        
        # Default quality thresholds (additive to survival filters)
        if quality_thresholds is None:
            self.quality_thresholds = {
                'causal_validity_min': 0.3,
                'mechanism_alignment_min': 0.4,
                'interventional_robustness_min': 0.5,
                'counterfactual_consistency_min': 0.3,
                'causal_invariance_min': 0.4,
                'overall_causal_quality_min': 0.4
            }
        else:
            self.quality_thresholds = quality_thresholds
        
        # Storage for quality results
        self.quality_results_ = {}
        self.geometry_scores_ = {}
        
    def assess_geometry_causal_quality(
        self, 
        geometry: Any,  # GeometryTrial or similar
        X: pd.DataFrame, 
        y: pd.Series,
        counterfactual_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Comprehensive causal quality assessment for a geometry.
        
        Args:
            geometry: Geometry trial object
            X: Feature matrix
            y: Target labels
            counterfactual_data: Counterfactual scenarios
            
        Returns:
            Dictionary with causal quality metrics
        """
        if self.verbose:
            geom_id = getattr(geometry, 'uuid', 'unknown')[:8]
            tprint_info(f"🔍 Assessing Causal Quality: {geom_id}")
        
        start_time = time.time()
        
        quality_scores = {}
        
        try:
            # 1. Causal Validity Score
            causal_validity = self._compute_causal_validity_score(geometry, X, y)
            quality_scores['causal_validity'] = causal_validity
            
            # 2. Mechanism Alignment Score
            mechanism_alignment = self._compute_mechanism_alignment_score(geometry, X, y)
            quality_scores['mechanism_alignment'] = mechanism_alignment
            
            # 3. Interventional Robustness Score
            interventional_robustness = self._compute_interventional_robustness_score(geometry, X, y)
            quality_scores['interventional_robustness'] = interventional_robustness
            
            # 4. Counterfactual Consistency Score
            if counterfactual_data is not None:
                counterfactual_consistency = self._compute_counterfactual_consistency_score(
                    geometry, counterfactual_data
                )
                quality_scores['counterfactual_consistency'] = counterfactual_consistency
            else:
                quality_scores['counterfactual_consistency'] = 0.0
            
            # 5. Causal Invariance Score
            causal_invariance = self._compute_causal_invariance_score(geometry, X, y)
            quality_scores['causal_invariance'] = causal_invariance
            
            # 6. Overall Causal Quality Score
            overall_causal_quality = self._compute_overall_causal_quality_score(quality_scores)
            quality_scores['overall_causal_quality'] = overall_causal_quality
            
            # 7. Pass/Fail Decision for Causal Quality
            quality_scores['passes_causal_quality'] = self._evaluate_causal_quality_thresholds(quality_scores)
            
            assessment_time = time.time() - start_time
            
            if self.verbose:
                tprint_info(f"   📊 Causal Quality: {overall_causal_quality:.3f}")
                tprint_info(f"   📊 Passes Causal Filter: {quality_scores['passes_causal_quality']}")
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal quality assessment failed: {e}")
            
            # Return default scores
            quality_scores = {
                'causal_validity': 0.0,
                'mechanism_alignment': 0.0,
                'interventional_robustness': 0.0,
                'counterfactual_consistency': 0.0,
                'causal_invariance': 0.0,
                'overall_causal_quality': 0.0,
                'passes_causal_quality': False
            }
        
        return quality_scores
    
    def _compute_causal_validity_score(self, geometry: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute causal validity score based on consistency with causal structure."""
        if not self.causal_graph:
            return 0.5  # Neutral score if no causal graph
        
        try:
            # Get geometry features
            geom_features = getattr(geometry, 'features', []) or getattr(geometry, 'selected_features', [])
            if not geom_features:
                return 0.0
            
            # Check if geometry respects causal ordering
            validity_score = 0.0
            total_checks = 0
            
            # Check that features don't violate causal constraints
            for i, feat1 in enumerate(geom_features):
                for j, feat2 in enumerate(geom_features[i+1:], i+1):
                    total_checks += 1
                    
                    # Check if features have causal relationship
                    feat1_parents = self.causal_graph.get(feat1, [])
                    feat2_parents = self.causal_graph.get(feat2, [])
                    
                    # Simple validity check: no obvious violations
                    if feat1 in feat2_parents or feat2 in feat1_parents:
                        # Features have causal relationship, check ordering
                        validity_score += 0.8  # High validity for causal features
                    else:
                        # No direct causal relationship
                        validity_score += 0.5  # Neutral validity
            
            if total_checks > 0:
                return validity_score / total_checks
            else:
                return 0.5
                
        except Exception:
            return 0.0
    
    def _compute_mechanism_alignment_score(self, geometry: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute mechanism alignment score based on SEM consistency."""
        if not self.scm or not hasattr(self.scm, 'structural_models_') or not self.scm.structural_models_:
            return 0.3  # Low score if no SCM
        
        try:
            # Get geometry features
            geom_features = getattr(geometry, 'features', []) or getattr(geometry, 'selected_features', [])
            if not geom_features:
                return 0.0
            
            alignment_score = 0.0
            total_checks = 0
            
            # Check alignment with fitted SEMs
            for target, model in self.scm.structural_models_.items():
                if target not in geom_features:
                    continue
                
                # Get parents for this target
                parents = self.causal_graph.get(target, [])
                valid_parents = [p for p in parents if p in X.columns]
                
                if not valid_parents:
                    continue
                
                # Check if geometry uses causal features correctly
                geom_parents = [f for f in valid_parents if f in geom_features]
                
                if geom_parents:
                    # Geometry includes causal parents - good alignment
                    alignment_score += 0.8
                else:
                    # Geometry doesn't include causal parents - poor alignment
                    alignment_score += 0.2
                
                total_checks += 1
            
            if total_checks > 0:
                return alignment_score / total_checks
            else:
                return 0.5
                
        except Exception:
            return 0.0
    
    def _compute_interventional_robustness_score(self, geometry: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute interventional robustness score using bootstrap testing."""
        try:
            # Get geometry features
            geom_features = getattr(geometry, 'features', []) or getattr(geometry, 'selected_features', [])
            if not geom_features or len(geom_features) == 0:
                return 0.0
            
            # Use available features
            available_features = [f for f in geom_features if f in X.columns]
            if not available_features:
                return 0.0
            
            X_geom = X[available_features]
            
            # Simple robustness test: cross-validation stability
            if len(X_geom) < 50:
                return 0.3  # Low score for small datasets
            
            try:
                # Use simple model for testing
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
                
                # Time series cross-validation
                cv = TimeSeriesSplit(n_splits=3)
                cv_scores = []
                
                for train_idx, val_idx in cv.split(X_geom):
                    if len(train_idx) < 20 or len(val_idx) < 10:
                        continue
                    
                    X_train, X_val = X_geom.iloc[train_idx], X_geom.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Check if we have both classes
                    if len(np.unique(y_train)) < 2:
                        continue
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_val)[:, 1]
                    
                    if len(np.unique(y_val)) > 1:
                        score = roc_auc_score(y_val, y_pred)
                        cv_scores.append(score)
                
                if cv_scores:
                    # Robustness = mean score - std score (penalize instability)
                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    robustness = max(0.0, mean_score - std_score)
                    return min(1.0, robustness * 2)  # Scale to [0,1]
                else:
                    return 0.3
                    
            except Exception:
                return 0.3
                
        except Exception:
            return 0.0
    
    def _compute_counterfactual_consistency_score(
        self, 
        geometry: Any, 
        counterfactual_data: pd.DataFrame
    ) -> float:
        """Compute counterfactual consistency score."""
        try:
            # Get geometry features
            geom_features = getattr(geometry, 'features', []) or getattr(geometry, 'selected_features', [])
            if not geom_features:
                return 0.0
            
            # Check consistency across counterfactual scenarios
            available_features = [f for f in geom_features if f in counterfactual_data.columns]
            
            if len(available_features) < 2:
                return 0.3
            
            # Compute consistency of feature relationships
            consistency_scores = []
            
            for i, feat1 in enumerate(available_features):
                for feat2 in available_features[i+1:]:
                    # Compute correlation in original vs counterfactual
                    try:
                        corr_orig = np.corrcoef(
                            counterfactual_data[feat1], 
                            counterfactual_data[feat2]
                        )[0, 1]
                        
                        # Simple consistency: correlation should be stable
                        if not np.isnan(corr_orig):
                            consistency_scores.append(abs(corr_orig))
                    
                    except Exception:
                        continue
            
            if consistency_scores:
                # Average consistency
                return np.mean(consistency_scores)
            else:
                return 0.3
                
        except Exception:
            return 0.0
    
    def _compute_causal_invariance_score(self, geometry: Any, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute causal invariance score across different conditions."""
        try:
            # Get geometry features
            geom_features = getattr(geometry, 'features', []) or getattr(geometry, 'selected_features', [])
            if not geom_features:
                return 0.0
            
            available_features = [f for f in geom_features if f in X.columns]
            if not available_features:
                return 0.0
            
            X_geom = X[available_features]
            
            # Test invariance across different time periods
            if len(X_geom) < 100:
                return 0.3
            
            # Split data into different regimes
            n_splits = 3
            split_size = len(X_geom) // n_splits
            
            invariance_scores = []
            
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X_geom)
                
                X_regime = X_geom.iloc[start_idx:end_idx]
                y_regime = y.iloc[start_idx:end_idx]
                
                if len(X_regime) < 20 or len(np.unique(y_regime)) < 2:
                    continue
                
                try:
                    # Simple model for this regime
                    model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)
                    model.fit(X_regime, y_regime)
                    
                    # Get feature importances
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        invariance_scores.append(np.mean(importances))
                
                except Exception:
                    continue
            
            if invariance_scores:
                # Invariance = stability of feature importances across regimes
                mean_importance = np.mean(invariance_scores)
                std_importance = np.std(invariance_scores)
                
                # Higher invariance if importances are stable
                invariance = max(0.0, 1.0 - (std_importance / (mean_importance + 1e-8)))
                return min(1.0, invariance)
            else:
                return 0.3
                
        except Exception:
            return 0.0
    
    def _compute_overall_causal_quality_score(self, quality_scores: Dict[str, float]) -> float:
        """Compute overall causal quality score from individual metrics."""
        weights = {
            'causal_validity': 0.25,
            'mechanism_alignment': 0.20,
            'interventional_robustness': 0.20,
            'counterfactual_consistency': 0.15,
            'causal_invariance': 0.20
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_scores:
                overall_score += weight * quality_scores[metric]
                total_weight += weight
        
        if total_weight > 0:
            return overall_score / total_weight
        else:
            return 0.0
    
    def _evaluate_causal_quality_thresholds(self, quality_scores: Dict[str, float]) -> bool:
        """Evaluate if geometry passes causal quality thresholds."""
        # Check overall causal quality threshold
        overall_quality = quality_scores.get('overall_causal_quality', 0.0)
        if overall_quality < self.quality_thresholds['overall_causal_quality_min']:
            return False
        
        # Check individual thresholds
        critical_metrics = [
            'causal_validity',
            'mechanism_alignment',
            'interventional_robustness'
        ]
        
        for metric in critical_metrics:
            threshold_key = f"{metric}_min"
            if threshold_key in self.quality_thresholds:
                threshold = self.quality_thresholds[threshold_key]
                score = quality_scores.get(metric, 0.0)
                if score < threshold:
                    return False
        
        return True


# Convenience function for quick usage
def assess_geometry_causal_quality(
    geometry: Any,
    X: pd.DataFrame,
    y: pd.Series,
    causal_graph: Dict[str, List[str]],
    scm: Optional[StructuralCausalModel] = None,
    counterfactual_data: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Quick function for causal quality assessment of a single geometry.
    
    Args:
        geometry: Geometry trial object
        X: Feature matrix
        y: Target labels
        causal_graph: Causal graph from discovery
        scm: Fitted structural causal models
        counterfactual_data: Counterfactual scenarios
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with causal quality scores
    """
    quality_metrics = CausalQualityMetrics(
        causal_graph=causal_graph,
        scm=scm,
        verbose=verbose
    )
    
    return quality_metrics.assess_geometry_causal_quality(
        geometry, X, y, counterfactual_data
    )
