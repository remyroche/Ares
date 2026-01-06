"""
Interventionist Sampling Module

Implements interventionist sampling for causal discovery and feature engineering.

Key Features:
1. Structural shock detection and intervention
2. Causal intervention sampling
3. Event generation from structural breaks
4. Integration with existing event systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class CausalInterventionSampler:
    """
    Implements interventionist sampling for causal discovery.
    
    Generates structural shocks and interventions to discover
    causal relationships and create causal events.
    """
    
    def __init__(
        self,
        shock_threshold: float = 2.0,
        intervention_strength: float = 1.0,
        min_shock_duration: int = 5,
        max_shock_duration: int = 20,
        n_interventions: int = 100,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Causal Intervention Sampler.
        
        Args:
            shock_threshold: Threshold for shock detection
            intervention_strength: Strength of causal interventions
            min_shock_duration: Minimum duration of shocks
            max_shock_duration: Maximum duration of shocks
            n_interventions: Number of interventions to generate
            random_state: Random seed
            verbose: Whether to print progress information
        """
        self.shock_threshold = shock_threshold
        self.intervention_strength = intervention_strength
        self.min_shock_duration = min_shock_duration
        self.max_shock_duration = max_shock_duration
        self.n_interventions = n_interventions
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        np.random.seed(random_state)
        
        # Storage for results
        self.structural_shocks_ = {}
        self.interventions_ = {}
        self.causal_graph_ = {}
        
    def detect_structural_shocks(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: str = "zscore"
    ) -> Dict[str, np.ndarray]:
        """
        Detect structural shocks in the data.
        
        Args:
            data: Input data
            features: Features to analyze
            method: Method for shock detection ("zscore", "quantile", "pca")
            
        Returns:
            Dictionary of structural shocks
        """
        try:
            if self.verbose:
                tprint_info("📊 Detecting Structural Shocks...")
            
            if features is None:
                features = data.columns.tolist()
            
            structural_shocks = {}
            
            for feature in features:
                if feature not in data.columns:
                    continue
                
                feature_data = data[feature].dropna()
                
                if len(feature_data) < 50:
                    continue
                
                if method == "zscore":
                    # Z-score based shock detection
                    rolling_mean = feature_data.rolling(window=20, min_periods=10).mean()
                    rolling_std = feature_data.rolling(window=20, min_periods=10).std()
                    z_scores = np.abs(feature_data - rolling_mean) / (rolling_std + 1e-8)
                    shock_mask = z_scores > self.shock_threshold
                    
                elif method == "quantile":
                    # Quantile-based shock detection
                    lower_threshold = feature_data.quantile(0.01)
                    upper_threshold = feature_data.quantile(0.99)
                    shock_mask = (feature_data < lower_threshold) | (feature_data > upper_threshold)
                    
                elif method == "pca":
                    # PCA-based shock detection (for multivariate)
                    if len(features) > 1:
                        # Use PCA on multiple features
                        pca_data = data[features].dropna()
                        if len(pca_data) > 50:
                            pca = PCA(n_components=1)
                            pca_scores = pca.fit_transform(pca_data).flatten()
                            
                            # Detect shocks in PCA scores
                            rolling_mean = pd.Series(pca_scores).rolling(window=20, min_periods=10).mean()
                            rolling_std = pd.Series(pca_scores).rolling(window=20, min_periods=10).std()
                            z_scores = np.abs(pca_scores - rolling_mean) / (rolling_std + 1e-8)
                            shock_mask = pd.Series(z_scores, index=pca_data.index) > self.shock_threshold
                        else:
                            shock_mask = pd.Series(False, index=feature_data.index)
                    else:
                        shock_mask = pd.Series(False, index=feature_data.index)
                
                else:
                    raise ValueError(f"Unknown shock detection method: {method}")
                
                # Filter shocks by minimum duration
                if isinstance(shock_mask, pd.Series):
                    shock_mask = self._filter_shocks_by_duration(shock_mask)
                
                structural_shocks[feature] = shock_mask.values if hasattr(shock_mask, 'values') else shock_mask
                
                if self.verbose and len(structural_shocks) <= 5:  # Show first few
                    n_shocks = shock_mask.sum() if hasattr(shock_mask, 'sum') else np.sum(shock_mask)
                    tprint_info(f"   - {feature}: {n_shocks} shocks detected")
            
            self.structural_shocks_ = structural_shocks
            
            if self.verbose:
                total_shocks = sum(mask.sum() if hasattr(mask, 'sum') else np.sum(mask) for mask in structural_shocks.values())
                tprint_success(f"✅ Structural shock detection complete:")
                tprint_info(f"   - Features analyzed: {len(structural_shocks)}")
                tprint_info(f"   - Total shocks: {total_shocks}")
                tprint_info(f"   - Method: {method}")
            
            return structural_shocks
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Structural shock detection failed: {e}")
            return {}
    
    def _filter_shocks_by_duration(self, shock_mask: pd.Series) -> pd.Series:
        """
        Filter shocks by minimum duration.
        
        Args:
            shock_mask: Boolean mask of shocks
            
        Returns:
            Filtered shock mask
        """
        try:
            filtered_mask = shock_mask.copy()
            
            # Find shock periods
            shock_periods = []
            in_shock = False
            start_idx = None
            
            for i, is_shock in enumerate(shock_mask):
                if is_shock and not in_shock:
                    # Start of shock period
                    in_shock = True
                    start_idx = i
                elif not is_shock and in_shock:
                    # End of shock period
                    in_shock = False
                    if start_idx is not None:
                        shock_periods.append((start_idx, i - 1))
                        start_idx = None
            
            # Handle case where shock continues to end
            if in_shock and start_idx is not None:
                shock_periods.append((start_idx, len(shock_mask) - 1))
            
            # Filter by duration
            for start, end in shock_periods:
                duration = end - start + 1
                if duration < self.min_shock_duration:
                    filtered_mask.iloc[start:end+1] = False
                elif duration > self.max_shock_duration:
                    # Keep only the most extreme part of long shocks
                    feature_data = shock_mask.index[start:end+1]
                    # This is simplified - in practice, you'd keep the peak
                    pass
            
            return filtered_mask
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Shock duration filtering failed: {e}")
            return shock_mask
    
    def generate_causal_interventions(
        self,
        data: pd.DataFrame,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        intervention_features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate causal interventions for discovery.
        
        Args:
            data: Input data
            causal_graph: Causal graph (if available)
            intervention_features: Features to intervene on
            
        Returns:
            Dictionary of interventions
        """
        try:
            if self.verbose:
                tprint_info("🔬 Generating Causal Interventions...")
            
            if intervention_features is None:
                intervention_features = data.columns.tolist()
            
            interventions = {}
            
            for i in range(self.n_interventions):
                # Select random feature for intervention
                feature = np.random.choice(intervention_features)
                
                if feature not in data.columns:
                    continue
                
                # Select random time point
                max_time_idx = len(data) - self.max_shock_duration - 1
                if max_time_idx <= 0:
                    continue
                
                start_idx = np.random.randint(0, max_time_idx)
                duration = np.random.randint(self.min_shock_duration, self.max_shock_duration + 1)
                end_idx = start_idx + duration
                
                # Generate intervention
                intervention = self._create_intervention(
                    data, feature, start_idx, end_idx, causal_graph
                )
                
                if intervention:
                    interventions[f"intervention_{i}"] = intervention
            
            self.interventions_ = interventions
            
            if self.verbose:
                tprint_success(f"✅ Generated {len(interventions)} causal interventions:")
                tprint_info(f"   - Features: {intervention_features}")
                tprint_info(f"   - Duration range: {self.min_shock_duration}-{self.max_shock_duration}")
                tprint_info(f"   - Strength: {self.intervention_strength}")
            
            return interventions
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Intervention generation failed: {e}")
            return {}
    
    def _create_intervention(
        self,
        data: pd.DataFrame,
        feature: str,
        start_idx: int,
        end_idx: int,
        causal_graph: Optional[Dict[str, List[str]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single intervention.
        
        Args:
            data: Input data
            feature: Feature to intervene on
            start_idx: Start index of intervention
            end_idx: End index of intervention
            causal_graph: Causal graph
            
        Returns:
            Intervention dictionary
        """
        try:
            # Get original values
            original_values = data[feature].iloc[start_idx:end_idx].values
            
            # Create intervention effect
            if causal_graph and feature in causal_graph:
                # Use causal information for intervention
                parents = causal_graph[feature]
                if parents:
                    # Intervention based on parent effects
                    parent_effect = 0
                    for parent in parents[:3]:  # Limit to top 3 parents
                        if parent in data.columns:
                            parent_values = data[parent].iloc[start_idx:end_idx].values
                            parent_effect += np.mean(parent_values) * self.intervention_strength
                    
                    intervention_values = original_values + parent_effect
                else:
                    intervention_values = original_values * (1 + self.intervention_strength)
            else:
                # Simple multiplicative intervention
                intervention_values = original_values * (1 + self.intervention_strength)
            
            # Create intervention dictionary
            intervention = {
                'feature': feature,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': end_idx - start_idx,
                'original_values': original_values,
                'intervention_values': intervention_values,
                'effect_size': np.mean(intervention_values - original_values),
                'type': 'causal_intervention',
                'strength': self.intervention_strength
            }
            
            return intervention
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Intervention creation failed: {e}")
            return None
    
    def combine_with_existing_events(
        self,
        cusum_events: Dict[Any, Dict[str, Any]],
        structural_shocks: Dict[str, np.ndarray],
        X: pd.DataFrame,
        separate_types: bool = True
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Combine structural shocks with existing CUSUM events.
        
        Args:
            cusum_events: Existing CUSUM events
            structural_shocks: Detected structural shocks
            X: Feature matrix for index alignment
            separate_types: Whether to keep event types separate
            
        Returns:
            Combined events dictionary
        """
        try:
            if self.verbose:
                tprint_info("🔄 Combining Structural Shocks with CUSUM Events...")
            
            combined_events = cusum_events.copy()
            
            if separate_types:
                # Keep event types separate
                for feature, shock_mask in structural_shocks.items():
                    if shock_mask.sum() == 0:
                        continue
                    
                    # Get shock indices
                    shock_indices = X.index[shock_mask]
                    
                    for idx in shock_indices:
                        if idx not in combined_events:
                            # New event from structural shock
                            combined_events[idx] = {
                                'type': 'structural_shock',
                                'feature': feature,
                                'strength': self._calculate_shock_strength(feature, shock_mask, X),
                                'source': 'structural_break',
                                'error_zscore': self._get_error_zscore_at_point(feature, idx, X),
                                'parents': self.causal_graph_.get(feature, [])
                            }
                        else:
                            # Enhance existing event with structural information
                            existing_event = combined_events[idx]
                            if existing_event.get('type') == 'cusum_filter':
                                existing_event['enhanced_by'] = 'structural_shock'
                                existing_event['structural_feature'] = feature
                                existing_event['structural_strength'] = self._calculate_shock_strength(feature, shock_mask, X)
                                existing_event['structural_parents'] = self.causal_graph_.get(feature, [])
            else:
                # Mix event types (original behavior)
                for feature, shock_mask in structural_shocks.items():
                    if shock_mask.sum() == 0:
                        continue
                    
                    # Get shock indices
                    shock_indices = X.index[shock_mask]
                    
                    for idx in shock_indices:
                        if idx not in combined_events:
                            # New event from structural shock
                            combined_events[idx] = {
                                'type': 'causal_intervention',
                                'feature': feature,
                                'strength': self._calculate_shock_strength(feature, shock_mask, X),
                                'source': 'structural_break',
                                'error_zscore': self._get_error_zscore_at_point(feature, idx, X),
                                'parents': self.causal_graph_.get(feature, [])
                            }
                        else:
                            # Enhance existing event with causal information
                            existing_event = combined_events[idx]
                            if existing_event.get('type') == 'cusum_filter':
                                existing_event['enhanced_by'] = 'causal_intervention'
                                existing_event['causal_feature'] = feature
                                existing_event['causal_strength'] = self._calculate_shock_strength(feature, shock_mask, X)
                                existing_event['causal_parents'] = self.causal_graph_.get(feature, [])
            
            if self.verbose:
                n_causal_events = sum(1 for event in combined_events.values() 
                                    if event.get('type') == 'causal_intervention')
                n_structural_events = sum(1 for event in combined_events.values() 
                                        if event.get('type') == 'structural_shock')
                n_enhanced_events = sum(1 for event in combined_events.values() 
                                     if 'enhanced_by' in event)
                
                tprint_success(f"✅ Event combination complete:")
                tprint_info(f"   - Causal events: {n_causal_events}")
                tprint_info(f"   - Structural events: {n_structural_events}")
                tprint_info(f"   - Enhanced events: {n_enhanced_events}")
                tprint_info(f"   - Total events: {len(combined_events)}")
            
            return combined_events
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Event combination failed: {e}")
            return cusum_events
    
    def _calculate_shock_strength(
        self,
        feature: str,
        shock_mask: np.ndarray,
        X: pd.DataFrame
    ) -> float:
        """
        Calculate strength of structural shock.
        
        Args:
            feature: Feature name
            shock_mask: Shock mask
            X: Feature matrix
            
        Returns:
            Shock strength
        """
        try:
            if feature not in X.columns:
                return 0.0
            
            feature_data = X[feature].values
            shock_data = feature_data[shock_mask]
            normal_data = feature_data[~shock_mask]
            
            if len(shock_data) == 0 or len(normal_data) == 0:
                return 0.0
            
            # Strength as difference in means
            shock_mean = np.mean(shock_data)
            normal_mean = np.mean(normal_data)
            normal_std = np.std(normal_data)
            
            if normal_std == 0:
                return 0.0
            
            strength = abs(shock_mean - normal_mean) / normal_std
            return strength
            
        except Exception:
            return 0.0
    
    def _get_error_zscore_at_point(
        self,
        feature: str,
        idx: Any,
        X: pd.DataFrame
    ) -> float:
        """
        Get error z-score at a specific point.
        
        Args:
            feature: Feature name
            idx: Index point
            X: Feature matrix
            
        Returns:
            Z-score at point
        """
        try:
            if feature not in X.columns:
                return 0.0
            
            feature_data = X[feature]
            
            # Get rolling statistics
            window = min(20, len(feature_data) // 4)
            if window < 5:
                return 0.0
            
            rolling_mean = feature_data.rolling(window=window, min_periods=5).mean()
            rolling_std = feature_data.rolling(window=window, min_periods=5).std()
            
            if idx in rolling_mean.index and idx in rolling_std.index:
                mean_val = rolling_mean[idx]
                std_val = rolling_std[idx]
                actual_val = feature_data[idx]
                
                if std_val == 0:
                    return 0.0
                
                z_score = abs(actual_val - mean_val) / std_val
                return z_score
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def run_interventionist_sampling(
        self,
        data: pd.DataFrame,
        cusum_events: Dict[Any, Dict[str, Any]],
        causal_graph: Optional[Dict[str, List[str]]] = None,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete interventionist sampling pipeline.
        
        Args:
            data: Input data
            cusum_events: Existing CUSUM events
            causal_graph: Causal graph
            features: Features to analyze
            
        Returns:
            Dictionary with sampling results
        """
        try:
            if self.verbose:
                tprint_info("🚀 Starting Interventionist Sampling Pipeline...")
            
            # Store causal graph
            self.causal_graph_ = causal_graph or {}
            
            # Step 1: Detect structural shocks
            structural_shocks = self.detect_structural_shocks(data, features)
            
            # Step 2: Generate causal interventions
            interventions = self.generate_causal_interventions(data, causal_graph, features)
            
            # Step 3: Combine with existing events
            combined_events = self.combine_with_existing_events(
                cusum_events, structural_shocks, data
            )
            
            # Compile results
            results = {
                'structural_shocks': structural_shocks,
                'interventions': interventions,
                'combined_events': combined_events,
                'n_structural_shocks': sum(mask.sum() for mask in structural_shocks.values()),
                'n_interventions': len(interventions),
                'n_combined_events': len(combined_events),
                'causal_graph': causal_graph
            }
            
            if self.verbose:
                tprint_success("✅ Interventionist Sampling Complete:")
                tprint_info(f"   - Structural shocks: {results['n_structural_shocks']}")
                tprint_info(f"   - Interventions: {results['n_interventions']}")
                tprint_info(f"   - Combined events: {results['n_combined_events']}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Interventionist sampling failed: {e}")
            return {
                'structural_shocks': {},
                'interventions': {},
                'combined_events': cusum_events,
                'error': str(e)
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of interventionist sampling.
        
        Returns:
            Summary dictionary
        """
        return {
            'shock_threshold': self.shock_threshold,
            'intervention_strength': self.intervention_strength,
            'n_interventions': self.n_interventions,
            'min_shock_duration': self.min_shock_duration,
            'max_shock_duration': self.max_shock_duration,
            'has_structural_shocks': len(self.structural_shocks_) > 0,
            'has_interventions': len(self.interventions_) > 0,
            'causal_graph_size': len(self.causal_graph_)
        }

# Convenience functions
def quick_interventionist_sampling(
    data: pd.DataFrame,
    cusum_events: Dict[Any, Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Quick interventionist sampling with default parameters.
    
    Args:
        data: Input data
        cusum_events: Existing CUSUM events
        **kwargs: Additional parameters
        
    Returns:
        Sampling results
    """
    sampler = CausalInterventionSampler(**kwargs)
    return sampler.run_interventionist_sampling(data, cusum_events)

def create_structural_shock_events(
    data: pd.DataFrame,
    features: List[str],
    threshold: float = 2.0,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Create structural shock events.
    
    Args:
        data: Input data
        features: Features to analyze
        threshold: Shock threshold
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of structural shocks
    """
    sampler = CausalInterventionSampler(shock_threshold=threshold, **kwargs)
    return sampler.detect_structural_shocks(data, features)
