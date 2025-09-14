"""
Feature Preparation Utilities

Common feature preparation patterns shared across all training modules.
Uses existing data utilities for consistency and efficiency.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

# Use existing utilities
from src.utils.data.unified_data_utils import UnifiedDataUtils
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.logger import system_logger

logger = system_logger.getChild('FeaturePreparator')


class FeaturePreparator:
    """Common feature preparation utilities."""
    
    @staticmethod
    def add_hmm_features(X: np.ndarray, hmm_states: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Add HMM states as features.
        
        Args:
            X: Input features
            hmm_states: HMM cluster/regime states
            
        Returns:
            Tuple of enhanced features and new feature names
        """
        if hmm_states is None:
            return X, []
        
        logger.info("🔄 Adding HMM states as features...")
        hmm_features = pd.get_dummies(hmm_states, prefix='hmm_state').values
        enhanced_X = np.hstack([X, hmm_features])
        
        hmm_feature_names = [f"hmm_state_{i}" for i in range(hmm_features.shape[1])]
        
        logger.info(f"📊 Added {hmm_features.shape[1]} HMM features")
        return enhanced_X, hmm_feature_names
    
    @staticmethod
    def create_regime_features(regime_labels: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Create regime-aware features.
        
        Args:
            regime_labels: Array of regime labels
            X: Input features
            
        Returns:
            Tuple of regime features and feature names
        """
        logger.info("🔄 Creating regime features...")
        regime_features = []
        feature_names = []
        
        # One-hot encoding of regime
        regime_onehot = pd.get_dummies(regime_labels, prefix='regime')
        regime_features.append(regime_onehot.values)
        feature_names.extend([f"regime_{i}" for i in range(regime_onehot.shape[1])])
        
        # Regime transition features
        regime_transitions = np.diff(regime_labels, prepend=regime_labels[0])
        regime_features.append(regime_transitions.reshape(-1, 1))
        feature_names.append("regime_transition")
        
        # Regime duration features
        regime_durations = FeaturePreparator.calculate_regime_durations(regime_labels)
        regime_features.append(regime_durations.reshape(-1, 1))
        feature_names.append("regime_duration")
        
        # Regime momentum features
        regime_momentum = FeaturePreparator.calculate_regime_momentum(regime_labels, X)
        regime_features.append(regime_momentum)
        feature_names.extend([f"regime_momentum_{i}" for i in range(regime_momentum.shape[1])])
        
        combined_regime_features = np.hstack(regime_features)
        
        logger.info(f"📊 Created {combined_regime_features.shape[1]} regime features")
        return combined_regime_features, feature_names
    
    @staticmethod
    def calculate_regime_durations(regime_labels: np.ndarray) -> np.ndarray:
        """
        Calculate duration of current regime for each sample using vectorized operations.

        Args:
            regime_labels: Array of regime labels

        Returns:
            Array of regime durations for each sample
        """
        # VECTORIZED: Calculate regime durations without loops
        # Find where regime changes occur
        regime_changes = np.diff(regime_labels, prepend=regime_labels[0])
        change_indices = np.where(regime_changes != 0)[0]

        if len(change_indices) == 0:
            # All same regime
            return np.full(len(regime_labels), len(regime_labels))

        # Calculate durations for each regime segment
        durations = np.zeros(len(regime_labels))

        # Add start and end indices
        segment_starts = np.concatenate([[0], change_indices])
        segment_ends = np.concatenate([change_indices, [len(regime_labels)]])

        for start, end in zip(segment_starts, segment_ends):
            duration = end - start
            durations[start:end] = duration

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
    
    @staticmethod
    def prepare_combined_features(
        X: np.ndarray,
        regime_labels: np.ndarray,
        hmm_states: Optional[np.ndarray] = None,
        analyst_outputs: Optional[np.ndarray] = None,
        analyst_output_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare combined features with HMM states, analyst outputs, and regime features.
        
        Args:
            X: Input features
            regime_labels: Array of regime labels
            hmm_states: Optional HMM cluster/regime states
            analyst_outputs: Optional analyst model outputs
            analyst_output_names: Names of analyst output features
            feature_names: Names of input features
            
        Returns:
            Tuple of combined features and feature names
        """
        features = [X]
        new_feature_names = feature_names.copy() if feature_names else []
        
        # Add HMM states as features if available
        if hmm_states is not None:
            hmm_X, hmm_names = FeaturePreparator.add_hmm_features(X, hmm_states)
            features.append(hmm_X[:, X.shape[1]:])  # Only the HMM features
            new_feature_names.extend(hmm_names)
        
        # Add Analyst outputs as features if available
        if analyst_outputs is not None:
            logger.info("🔄 Adding Analyst outputs as features...")
            features.append(analyst_outputs)
            if analyst_output_names:
                new_feature_names.extend(analyst_output_names)
            logger.info(f"📊 Added {analyst_outputs.shape[1]} Analyst features")
        
        # Add regime features
        regime_X, regime_names = FeaturePreparator.create_regime_features(regime_labels, X)
        features.append(regime_X)
        new_feature_names.extend(regime_names)
        
        # Combine all features
        combined_features = np.hstack(features)
        
        logger.info(f"📊 Combined features: {combined_features.shape[1]} total features")
        logger.info(f"📊 - Original features: {X.shape[1]}")
        if hmm_states is not None:
            logger.info(f"📊 - HMM features: {hmm_X.shape[1] - X.shape[1]}")
        if analyst_outputs is not None:
            logger.info(f"📊 - Analyst features: {analyst_outputs.shape[1]}")
        logger.info(f"📊 - Regime features: {regime_X.shape[1]}")
        
        return combined_features, new_feature_names
    
    @staticmethod
    def get_analyst_outputs(
        X: np.ndarray,
        regime_labels: np.ndarray,
        analyst_ensembles: Dict[int, Any],
        analyst_output_names: List[str],
        analyst_threshold: float = 0.6
    ) -> np.ndarray:
        """
        Get Analyst outputs for all samples.
        
        Args:
            X: Input features
            regime_labels: Array of regime labels
            analyst_ensembles: Pre-trained Analyst ensemble models
            analyst_output_names: Names of analyst output features
            analyst_threshold: Threshold for filtering analyst outputs
            
        Returns:
            Array of analyst outputs for all samples
        """
        analyst_outputs = np.zeros((len(X), len(analyst_output_names)))
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            
            if regime in analyst_ensembles:
                try:
                    ensemble_manager = analyst_ensembles[regime]['ensemble_manager']
                    regime_outputs = ensemble_manager.predict(regime_X)
                    
                    # Apply threshold filtering if confidence is available
                    if regime_outputs.shape[1] > 1:
                        confidence_scores = regime_outputs[:, 1]  # Assuming confidence is second column
                        valid_mask = confidence_scores >= analyst_threshold
                        
                        # Only use outputs above threshold
                        analyst_outputs[regime_mask] = regime_outputs
                        analyst_outputs[regime_mask][~valid_mask] = 0  # Zero out low confidence outputs
                        
                        logger.debug(f"📊 Regime {regime}: {np.sum(valid_mask)}/{len(regime_X)} samples above threshold")
                    else:
                        # No confidence scores, use all outputs
                        analyst_outputs[regime_mask] = regime_outputs
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get Analyst outputs for regime {regime}: {e}")
                    continue
            else:
                logger.warning(f"⚠️ No Analyst ensemble found for regime {regime}")
        
        return analyst_outputs