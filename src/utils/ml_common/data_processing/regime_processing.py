"""
Regime Processing Utilities

Common regime processing patterns shared across all training modules.
Uses existing data utilities for consistency and efficiency.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

# Use existing utilities
from src.utils.data.unified_data_utils import UnifiedDataUtils
from src.utils.data.quality.data_quality import DataQualityFramework
from src.utils.logger import system_logger

logger = system_logger.getChild('RegimeProcessor')


class RegimeProcessor:
    """Common regime processing utilities."""
    
    @staticmethod
    def analyze_regimes(
        regime_labels: np.ndarray, 
        min_samples: int = 1000,
        enable_regime_merging: bool = True,
        regime_merge_threshold: int = 500
    ) -> Dict[str, Any]:
        """
        Analyze regime distribution and characteristics.
        
        Args:
            regime_labels: Array of regime labels for each sample
            min_samples: Minimum samples required per regime
            enable_regime_merging: Whether to enable regime merging
            regime_merge_threshold: Threshold for regime merging
            
        Returns:
            Dictionary containing regime analysis results
        """
        unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
        
        regime_analysis = {
            'unique_regimes': unique_regimes,
            'regime_counts': regime_counts,
            'total_samples': len(regime_labels),
            'regime_proportions': regime_counts / len(regime_labels)
        }
        
        # Identify regimes with sufficient data
        sufficient_regimes = unique_regimes[regime_counts >= min_samples]
        insufficient_regimes = unique_regimes[regime_counts < min_samples]
        
        # 🔧 CRITICAL FIX: If no regimes meet minimum threshold, use adaptive threshold
        if len(sufficient_regimes) == 0 and len(unique_regimes) > 0:
            logger.warning(f"⚠️ 🚨 NO regimes meet minimum threshold of {min_samples} samples!")
            logger.warning(f"⚠️ 🚨 Regime distribution: {dict(zip(unique_regimes, regime_counts))}")
            
            # Use adaptive threshold: 50% of largest regime or 10% of total samples, whichever is smaller
            adaptive_threshold = min(
                int(regime_counts.max() * 0.5),  # 50% of largest regime
                int(len(regime_labels) * 0.1)    # 10% of total samples
            )
            adaptive_threshold = max(adaptive_threshold, 100)  # But at least 100 samples
            
            logger.warning(f"⚠️ 🔧 Using adaptive threshold: {adaptive_threshold} samples")
            sufficient_regimes = unique_regimes[regime_counts >= adaptive_threshold]
            insufficient_regimes = unique_regimes[regime_counts < adaptive_threshold]
            
            # Update regime analysis with adaptive threshold info
            regime_analysis['adaptive_threshold_used'] = True
            regime_analysis['adaptive_threshold'] = adaptive_threshold
            regime_analysis['original_threshold'] = min_samples
        else:
            regime_analysis['adaptive_threshold_used'] = False
        
        regime_analysis['sufficient_regimes'] = sufficient_regimes
        regime_analysis['insufficient_regimes'] = insufficient_regimes
        
        # Identify regimes to merge
        if enable_regime_merging:
            merge_candidates = unique_regimes[regime_counts < regime_merge_threshold]
            regime_analysis['merge_candidates'] = merge_candidates
        else:
            regime_analysis['merge_candidates'] = []
        
        logger.info(f"📊 Regime analysis:")
        logger.info(f"📊 - Total regimes: {len(unique_regimes)}")
        logger.info(f"📊 - Regime distribution: {dict(zip(unique_regimes, regime_counts))}")
        logger.info(f"📊 - Sufficient data: {len(sufficient_regimes)} regimes")
        logger.info(f"📊 - Insufficient data: {len(insufficient_regimes)} regimes")
        logger.info(f"📊 - Merge candidates: {len(regime_analysis['merge_candidates'])}")
        if regime_analysis.get('adaptive_threshold_used', False):
            logger.warning(f"⚠️ 🔧 Used adaptive threshold: {regime_analysis['adaptive_threshold']} (original: {regime_analysis['original_threshold']})")
        
        return regime_analysis
    
    @staticmethod
    def prepare_regime_data(
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        regime_analysis: Dict[str, Any],
        hmm_states: Optional[np.ndarray] = None,
        min_samples: int = 1000,
        enable_data_augmentation: bool = True,
        augmentation_method: str = "smote",
        augmentation_ratio: float = 1.0
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Prepare data for each regime with HMM state integration.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            regime_analysis: Results from regime analysis
            hmm_states: Optional HMM cluster/regime states
            min_samples: Minimum samples required per regime
            enable_data_augmentation: Whether to enable data augmentation
            augmentation_method: Method for data augmentation
            augmentation_ratio: Ratio for data augmentation
            
        Returns:
            Dictionary containing prepared data for each regime
        """
        regime_data = {}
        
        for regime in regime_analysis['unique_regimes']:
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            # Add HMM states as features if available
            regime_hmm_states = None
            if hmm_states is not None:
                regime_hmm_states = hmm_states[regime_mask]
                # One-hot encode HMM states
                hmm_features = pd.get_dummies(regime_hmm_states, prefix='hmm_state').values
                regime_X = np.hstack([regime_X, hmm_features])
            
            # Check if regime has sufficient data
            if len(regime_X) >= min_samples:
                # Sufficient data - use as is
                regime_data[regime] = {
                    'X': regime_X,
                    'y': regime_y,
                    'samples': len(regime_X),
                    'augmented': False,
                    'hmm_states': regime_hmm_states
                }
            elif enable_data_augmentation and len(regime_X) > 100:
                # Insufficient data but enough for augmentation
                augmented_X, augmented_y = RegimeProcessor.augment_regime_data(
                    regime_X, regime_y, augmentation_method, augmentation_ratio
                )
                regime_data[regime] = {
                    'X': augmented_X,
                    'y': augmented_y,
                    'samples': len(augmented_X),
                    'augmented': True,
                    'hmm_states': regime_hmm_states
                }
            else:
                # Too little data - mark for global model fallback
                regime_data[regime] = {
                    'X': regime_X,
                    'y': regime_y,
                    'samples': len(regime_X),
                    'augmented': False,
                    'use_global': True,
                    'hmm_states': regime_hmm_states
                }
            
            logger.debug(f"📊 Regime {regime}: {regime_data[regime]['samples']} samples, "
                        f"augmented: {regime_data[regime]['augmented']}, "
                        f"use_global: {regime_data[regime].get('use_global', False)}")
        
        return regime_data
    
    @staticmethod
    def augment_regime_data(
        X: np.ndarray, 
        y: np.ndarray, 
        method: str = "smote",
        augmentation_ratio: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data for regimes with insufficient samples.
        Uses existing data utilities for consistency.
        
        Args:
            X: Input features
            y: Target values
            method: Augmentation method (smote, adasyn)
            augmentation_ratio: Ratio for augmentation
            
        Returns:
            Tuple of augmented features and targets
        """
        try:
            # Use existing data utilities for augmentation
            data_utils = UnifiedDataUtils()
            
            # Convert to DataFrame for processing
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)
            
            # Use existing data processing capabilities
            if method == "smote":
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42, sampling_strategy=augmentation_ratio)
                    X_aug, y_aug = smote.fit_resample(X_df, y_series)
                    return X_aug.values, y_aug.values
                except ImportError:
                    logger.warning("⚠️ SMOTE not available, skipping augmentation")
                    return X, y
            elif method == "adasyn":
                try:
                    from imblearn.over_sampling import ADASYN
                    adasyn = ADASYN(random_state=42, sampling_strategy=augmentation_ratio)
                    X_aug, y_aug = adasyn.fit_resample(X_df, y_series)
                    return X_aug.values, y_aug.values
                except ImportError:
                    logger.warning("⚠️ ADASYN not available, skipping augmentation")
                    return X, y
            else:
                logger.warning(f"⚠️ Unknown augmentation method: {method}")
                return X, y
                
        except Exception as e:
            logger.warning(f"⚠️ Data augmentation failed: {e}")
            return X, y
    
    @staticmethod
    def calculate_regime_durations(regime_labels: np.ndarray) -> np.ndarray:
        """
        Calculate duration of current regime for each sample.
        
        Args:
            regime_labels: Array of regime labels
            
        Returns:
            Array of regime durations for each sample
        """
        durations = np.zeros(len(regime_labels))
        current_regime = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                # Regime changed, update durations for previous regime
                durations[i-current_duration:i] = current_duration
                current_regime = regime_labels[i]
                current_duration = 1
        
        # Update durations for the last regime
        durations[-current_duration:] = current_duration
        
        return durations
    
    @staticmethod
    def calculate_regime_momentum(regime_labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Calculate momentum features within each regime.
        
        Args:
            regime_labels: Array of regime labels
            X: Input features
            
        Returns:
            Array of regime momentum features
        """
        momentum_features = []
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            
            if len(regime_X) > 1:
                # Calculate momentum as difference between consecutive samples
                regime_momentum = np.diff(regime_X, axis=0)
                # Pad with zeros for the first sample
                regime_momentum = np.vstack([np.zeros((1, regime_momentum.shape[1])), regime_momentum])
            else:
                regime_momentum = np.zeros((1, X.shape[1]))
            
            momentum_features.append(regime_momentum)
        
        # Combine momentum features
        combined_momentum = np.vstack(momentum_features)
        return combined_momentum