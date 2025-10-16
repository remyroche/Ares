"""
Feature Preparation Utilities

Common feature preparation patterns shared across all training modules.
Uses existing data utilities for consistency and efficiency.
"""

import numpy as np
import pandas as pd
import time
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

    @staticmethod
    def prepare_features(
        data: Union[np.ndarray, pd.DataFrame],
        feature_config: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None,
        regime_labels: Optional[np.ndarray] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_outputs: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Comprehensive feature preparation pipeline.

        Args:
            data: Input data (DataFrame or numpy array)
            feature_config: Feature preparation configuration
            target_column: Name of target column (if applicable)
            regime_labels: Array of regime labels
            hmm_states: Optional HMM cluster/regime states
            analyst_outputs: Optional analyst model outputs
            feature_names: Names of input features

        Returns:
            Tuple of (prepared_features, feature_names, preparation_metadata)
        """
        logger.info("🔄 Starting comprehensive feature preparation...")

        start_time = time.time()
        preparation_metadata = {
            'start_time': start_time,
            'operations_performed': [],
            'feature_counts': {},
            'warnings': [],
            'errors': []
        }

        try:
            # Convert input to numpy array if needed
            if isinstance(data, pd.DataFrame):
                if target_column and target_column in data.columns:
                    # Separate target column
                    target_data = data[target_column].values
                    feature_data = data.drop(columns=[target_column]).values
                    if feature_names is None:
                        feature_names = [str(col) for col in data.drop(columns=[target_column]).columns.tolist()]
                    preparation_metadata['target_separated'] = True
                else:
                    feature_data = data.values
                    if feature_names is None:
                        feature_names = [str(col) for col in data.columns.tolist()]
                    preparation_metadata['target_separated'] = False
            else:
                feature_data = data.copy()
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(feature_data.shape[1])]
                preparation_metadata['target_separated'] = False

            preparation_metadata['feature_counts']['original'] = feature_data.shape[1]
            preparation_metadata['operations_performed'].append('data_conversion')

            # Apply feature configuration if provided
            if feature_config:
                feature_data, feature_names = FeaturePreparator._apply_feature_config(
                    feature_data, feature_names, feature_config, preparation_metadata
                )

            # Add regime-based features if regime labels are provided
            if regime_labels is not None:
                logger.info("🔄 Adding regime-based features...")
                regime_features, regime_names = FeaturePreparator.create_regime_features(
                    regime_labels, feature_data
                )
                feature_data = np.hstack([feature_data, regime_features])
                feature_names.extend(regime_names)
                preparation_metadata['feature_counts']['after_regime'] = feature_data.shape[1]
                preparation_metadata['operations_performed'].append('regime_features')

            # Add HMM features if available
            if hmm_states is not None:
                logger.info("🔄 Adding HMM features...")
                enhanced_features, hmm_names = FeaturePreparator.add_hmm_features(
                    feature_data, hmm_states
                )
                feature_data = enhanced_features
                feature_names.extend(hmm_names)
                preparation_metadata['feature_counts']['after_hmm'] = feature_data.shape[1]
                preparation_metadata['operations_performed'].append('hmm_features')

            # Add analyst outputs if available
            if analyst_outputs is not None:
                logger.info("🔄 Adding analyst output features...")
                feature_data = np.hstack([feature_data, analyst_outputs])
                analyst_names = [f"analyst_output_{i}" for i in range(analyst_outputs.shape[1])]
                feature_names.extend(analyst_names)
                preparation_metadata['feature_counts']['after_analyst'] = feature_data.shape[1]
                preparation_metadata['operations_performed'].append('analyst_features')

            # Final validation and cleanup
            feature_data, feature_names = FeaturePreparator._validate_and_cleanup_features(
                feature_data, feature_names, preparation_metadata
            )

            # Calculate final statistics
            preparation_metadata['feature_counts']['final'] = feature_data.shape[1]
            preparation_metadata['sample_count'] = feature_data.shape[0]
            preparation_metadata['end_time'] = time.time()
            preparation_metadata['total_duration'] = preparation_metadata['end_time'] - start_time
            preparation_metadata['success'] = True

            logger.info(f"✅ Feature preparation completed in {preparation_metadata['total_duration']:.3f}s")
            logger.info(f"📊 Final feature count: {preparation_metadata['feature_counts']['final']}")

            return feature_data, feature_names, preparation_metadata

        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            preparation_metadata['errors'].append(str(e))
            preparation_metadata['success'] = False
            preparation_metadata['end_time'] = time.time()

            # Return original data on failure
            if isinstance(data, pd.DataFrame):
                fallback_data = data.values
                fallback_names = [str(col) for col in data.columns.tolist()]
            else:
                fallback_data = data.copy()
                fallback_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]

            return fallback_data, fallback_names, preparation_metadata

    @staticmethod
    def _apply_feature_config(
        data: np.ndarray,
        feature_names: List[str],
        config: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply feature configuration transformations."""
        try:
            # Feature scaling
            if config.get('scale_features', False):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
                metadata['operations_performed'].append('feature_scaling')

            # Feature selection by importance
            if 'feature_importance_threshold' in config:
                threshold = config['feature_importance_threshold']
                # Simple variance-based selection as fallback
                feature_variances = np.var(data, axis=0)
                selected_indices = feature_variances > threshold

                if np.sum(selected_indices) > 0:
                    data = data[:, selected_indices]
                    feature_names = [name for i, name in enumerate(feature_names) if selected_indices[i]]
                    metadata['operations_performed'].append('feature_selection')
                    metadata['features_removed'] = np.sum(~selected_indices)

            # Dimensionality reduction
            if config.get('apply_pca', False) and 'n_components' in config:
                from sklearn.decomposition import PCA
                n_components = min(config['n_components'], data.shape[1])
                pca = PCA(n_components=n_components)
                data = pca.fit_transform(data)
                feature_names = [f"pca_component_{i}" for i in range(n_components)]
                metadata['operations_performed'].append('pca')
                metadata['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()

            return data, feature_names

        except Exception as e:
            logger.warning(f"⚠️ Feature config application failed: {e}")
            metadata['warnings'].append(f"Feature config failed: {e}")
            return data, feature_names

    @staticmethod
    def _validate_and_cleanup_features(
        data: np.ndarray,
        feature_names: List[str],
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Validate and cleanup prepared features."""
        try:
            original_shape = data.shape

            # Ensure data is numeric and convert if needed
            try:
                # Convert to float64 to ensure compatibility with isfinite
                if data.dtype == object or not np.issubdtype(data.dtype, np.number):
                    logger.info("🔄 Converting non-numeric data to float64...")
                    # Try to convert object arrays or non-numeric types
                    data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
                elif data.dtype != np.float64:
                    # Convert numeric types to float64 for consistency
                    data = data.astype(np.float64)
            except Exception as conv_error:
                logger.warning(f"⚠️ Data conversion failed: {conv_error}. Attempting fallback...")
                # Fallback: try to handle mixed types
                try:
                    data = np.array(data, dtype=np.float64)
                except (ValueError, TypeError):
                    # Last resort: convert via pandas
                    data = pd.DataFrame(data).select_dtypes(include=[np.number]).values.astype(np.float64)

            # Remove features with all NaN or infinite values
            valid_features = np.isfinite(data).all(axis=0)
            if not valid_features.all():
                invalid_count = np.sum(~valid_features)
                logger.warning(f"⚠️ 🚨 Removing {invalid_count} features with invalid values")
                data = data[:, valid_features]
                feature_names = [name for i, name in enumerate(feature_names) if valid_features[i]]
                metadata['warnings'].append(f"Removed {invalid_count} invalid features")

            # Remove constant features (zero variance)
            if data.shape[1] > 1:
                try:
                    feature_variances = np.var(data, axis=0)
                    # Ensure variances are finite
                    finite_variances = np.isfinite(feature_variances)
                    non_constant = (feature_variances > 1e-10) & finite_variances
                    if not non_constant.all():
                        constant_count = np.sum(~non_constant)
                        logger.warning(f"⚠️ 🚨 Removing {constant_count} constant/invalid variance features")
                        data = data[:, non_constant]
                        feature_names = [name for i, name in enumerate(feature_names) if non_constant[i]]
                        metadata['warnings'].append(f"Removed {constant_count} constant/invalid features")
                except Exception as var_error:
                    logger.warning(f"⚠️ Variance calculation failed: {var_error}. Skipping constant feature removal.")
                    metadata['warnings'].append(f"Variance calculation failed: {var_error}")

            # Ensure feature names match data dimensions
            if len(feature_names) != data.shape[1]:
                logger.warning("⚠️ Feature names count mismatch - generating new names")
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                metadata['warnings'].append("Feature names regenerated due to mismatch")

            metadata['cleanup_performed'] = True
            metadata['shape_change'] = f"{original_shape} -> {data.shape}"

            return data, feature_names

        except Exception as e:
            logger.warning(f"⚠️ Feature validation failed: {e}")
            metadata['warnings'].append(f"Validation failed: {e}")
            return data, feature_names
