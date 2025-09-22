"""
HMM-Specific Temporal Protection and Bias Prevention

This module provides specialized temporal protection for HMM models,
consolidating lookahead protection and temporal validation functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

from .lookahead_protection import LookaheadProtection
from ..config.base_training_config import HMMTrainingConfig

logger = logging.getLogger(__name__)

class HMMTemporalProtection:
    """
    Specialized temporal protection for HMM model training and prediction.
    Integrates lookahead bias detection with HMM-specific temporal constraints.
    """

    def __init__(self, config: Optional[HMMTrainingConfig] = None):
        """
        Initialize HMM temporal protection.

        Args:
            config: HMM training configuration
        """
        self.config = config or HMMTrainingConfig()
        self.lookahead_protection = LookaheadProtection()
        self.logger = logger.getChild('HMMTemporalProtection')

        # HMM-specific temporal constraints
        self.hmm_temporal_constraints = {
            'max_lookback_days': 30,  # Maximum lookback for HMM features
            'min_prediction_horizon': timedelta(minutes=15),  # Minimum prediction horizon
            'state_transition_buffer': timedelta(minutes=5),  # Buffer for state transitions
            'regime_stability_period': timedelta(hours=1),  # Minimum regime stability
            'feature_freshness_threshold': timedelta(minutes=30)  # Feature freshness threshold
        }

    def validate_hmm_temporal_constraints(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        prediction_timestamp: Optional[datetime] = None,
        timestamp_col: str = 'timestamp'
    ) -> Dict[str, Any]:
        """
        Validate HMM-specific temporal constraints.

        Args:
            features_df: DataFrame containing features
            target_df: DataFrame containing targets
            prediction_timestamp: Timestamp when prediction is made
            timestamp_col: Timestamp column name

        Returns:
            Temporal constraint validation results
        """
        self.logger.info("⏰ Validating HMM temporal constraints")

        if prediction_timestamp is None:
            prediction_timestamp = datetime.now()

        validation_results = {
            'temporal_constraints_valid': True,
            'constraint_violations': [],
            'feature_temporal_analysis': {},
            'target_temporal_analysis': {},
            'state_transition_analysis': {},
            'regime_stability_analysis': {},
            'recommendations': []
        }

        try:
            # Set current timestamp for lookahead protection
            self.lookahead_protection.set_current_timestamp(prediction_timestamp)

            # 1. Feature temporal analysis
            validation_results['feature_temporal_analysis'] = self._analyze_feature_temporal_quality(
                features_df, prediction_timestamp, timestamp_col
            )

            # 2. Target temporal analysis
            validation_results['target_temporal_analysis'] = self._analyze_target_temporal_quality(
                target_df, prediction_timestamp, timestamp_col
            )

            # 3. State transition temporal analysis
            validation_results['state_transition_analysis'] = self._analyze_state_transition_temporal(
                features_df, target_df, timestamp_col
            )

            # 4. Regime stability analysis
            validation_results['regime_stability_analysis'] = self._analyze_regime_stability_temporal(
                features_df, target_df, timestamp_col
            )

            # 5. Aggregate constraint violations
            validation_results['constraint_violations'] = self._aggregate_temporal_violations(
                validation_results
            )

            # 6. Overall temporal validity
            validation_results['temporal_constraints_valid'] = (
                len(validation_results['constraint_violations']) == 0
            )

            # 7. Generate recommendations
            validation_results['recommendations'] = self._generate_temporal_recommendations(
                validation_results
            )

            self.logger.info(f"✅ HMM temporal constraints validation completed - "
                           f"{'Valid' if validation_results['temporal_constraints_valid'] else 'Invalid'}")

            return validation_results

        except Exception as e:
            self.logger.error(f"❌ HMM temporal constraints validation failed: {e}")
            return {
                'temporal_constraints_valid': False,
                'constraint_violations': [f'Validation failed: {str(e)}'],
                'error': str(e)
            }

    def detect_hmm_lookahead_bias(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        current_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Detect HMM-specific lookahead bias patterns.

        Args:
            X: Feature matrix
            y: Target values (HMM states)
            timestamps: Timestamps for data points
            feature_names: Names of features
            hmm_states: Optional HMM state assignments
            current_timestamp: Current timestamp for bias detection

        Returns:
            HMM-specific lookahead bias detection results
        """
        self.logger.info("🔍 Detecting HMM-specific lookahead bias")

        if current_timestamp is None:
            current_timestamp = datetime.now()

        self.lookahead_protection.set_current_timestamp(current_timestamp)

        # Create DataFrames for analysis
        features_df = self._create_feature_dataframe(X, feature_names, timestamps, y)
        target_df = features_df.copy()  # For HMM, features and targets are related

        # Perform comprehensive lookahead bias detection
        bias_results = self.lookahead_protection.detect_data_leakage(
            features_df=features_df,
            target_df=target_df,
            timestamp_col='timestamp'
        )

        # Add HMM-specific bias analysis
        hmm_bias_analysis = self._analyze_hmm_specific_bias_patterns(
            X, y, timestamps, hmm_states
        )

        # Combine results
        combined_results = {
            'general_bias_detection': bias_results,
            'hmm_specific_bias_analysis': hmm_bias_analysis,
            'overall_bias_detected': (
                bias_results.get('leakage_detected', False) or
                hmm_bias_analysis.get('bias_detected', False)
            ),
            'bias_confidence': max(
                bias_results.get('bias_score', 0.0),
                hmm_bias_analysis.get('bias_confidence', 0.0)
            )
        }

        # Generate recommendations
        combined_results['recommendations'] = self._generate_bias_recommendations(combined_results)

        self.logger.info(f"✅ HMM lookahead bias detection completed - "
                       f"{'Bias detected' if combined_results['overall_bias_detected'] else 'No bias detected'}")

        return combined_results

    def create_temporal_data_filters(
        self,
        df: pd.DataFrame,
        current_timestamp: Optional[datetime] = None,
        timestamp_col: str = 'timestamp',
        filter_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create temporal data filters for HMM training data.

        Args:
            df: DataFrame to filter
            current_timestamp: Current timestamp
            timestamp_col: Timestamp column name
            filter_config: Filtering configuration

        Returns:
            Dictionary of filtered DataFrames for different temporal constraints
        """
        self.logger.info("🔧 Creating temporal data filters for HMM training")

        if current_timestamp is None:
            current_timestamp = datetime.now()

        if filter_config is None:
            filter_config = {
                'remove_future_data': True,
                'max_lookback_days': self.hmm_temporal_constraints['max_lookback_days'],
                'enforce_regime_stability': True,
                'remove_stale_features': True
            }

        filtered_datasets = {
            'original': df.copy(),
            'future_filtered': None,
            'temporal_constraints_applied': None,
            'regime_stable': None,
            'fresh_features': None
        }

        try:
            # 1. Remove future data
            if filter_config.get('remove_future_data', True):
                filtered_datasets['future_filtered'] = self.lookahead_protection.automated_future_data_filtering(
                    df=df,
                    current_time=current_timestamp,
                    timestamp_col=timestamp_col
                )

            # 2. Apply temporal constraints
            base_df = filtered_datasets['future_filtered'] or df
            filtered_datasets['temporal_constraints_applied'] = self._apply_temporal_constraints(
                base_df, current_timestamp, timestamp_col, filter_config
            )

            # 3. Filter for regime stability
            if filter_config.get('enforce_regime_stability', True):
                base_df = filtered_datasets['temporal_constraints_applied'] or base_df
                filtered_datasets['regime_stable'] = self._filter_regime_stable_data(
                    base_df, timestamp_col
                )

            # 4. Filter for fresh features
            if filter_config.get('remove_stale_features', True):
                base_df = filtered_datasets['regime_stable'] or filtered_datasets['temporal_constraints_applied'] or base_df
                filtered_datasets['fresh_features'] = self._filter_fresh_features(
                    base_df, current_timestamp, timestamp_col
                )

            # Log filtering results
            self._log_filtering_results(filtered_datasets)

            return filtered_datasets

        except Exception as e:
            self.logger.error(f"❌ Temporal data filtering failed: {e}")
            return {'original': df, 'error': str(e)}

    def _analyze_feature_temporal_quality(
        self,
        features_df: pd.DataFrame,
        prediction_timestamp: datetime,
        timestamp_col: str
    ) -> Dict[str, Any]:
        """Analyze temporal quality of features for HMM training."""
        analysis = {
            'temporal_range': {},
            'feature_freshness': {},
            'temporal_consistency': {},
            'violations': []
        }

        if timestamp_col not in features_df.columns:
            analysis['violations'].append(f"Timestamp column '{timestamp_col}' not found")
            return analysis

        # Convert timestamps if needed
        features_df = self.lookahead_protection._ensure_timestamp_format(features_df, timestamp_col)
        timestamps = features_df[timestamp_col].dropna()

        if len(timestamps) == 0:
            analysis['violations'].append("No valid timestamps found")
            return analysis

        # Temporal range analysis
        analysis['temporal_range'] = {
            'earliest': timestamps.min(),
            'latest': timestamps.max(),
            'total_span': timestamps.max() - timestamps.min(),
            'prediction_time': prediction_timestamp,
            'max_lookback_exceeded': (prediction_timestamp - timestamps.min()).days > self.hmm_temporal_constraints['max_lookback_days']
        }

        # Feature freshness analysis
        freshness_scores = []
        for col in features_df.columns:
            if col != timestamp_col:
                feature_data = features_df[col]
                valid_timestamps = timestamps[~feature_data.isnull()]

                if len(valid_timestamps) > 0:
                    freshness = prediction_timestamp - valid_timestamps.max()
                    freshness_scores.append({
                        'feature': col,
                        'last_update': valid_timestamps.max(),
                        'freshness': freshness,
                        'is_fresh': freshness <= self.hmm_temporal_constraints['feature_freshness_threshold']
                    })

        analysis['feature_freshness'] = freshness_scores

        # Temporal consistency analysis
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            analysis['temporal_consistency'] = {
                'mean_interval': time_diffs.mean(),
                'std_interval': time_diffs.std(),
                'is_consistent': time_diffs.std() < time_diffs.mean() * 0.5
            }

        # Check for violations
        if analysis['temporal_range']['max_lookback_exceeded']:
            analysis['violations'].append("Feature lookback exceeds maximum allowed days")

        stale_features = [f['feature'] for f in freshness_scores if not f['is_fresh']]
        if stale_features:
            analysis['violations'].append(f"Stale features detected: {stale_features}")

        if not analysis['temporal_consistency']['is_consistent']:
            analysis['violations'].append("Inconsistent temporal intervals detected")

        return analysis

    def _analyze_target_temporal_quality(
        self,
        target_df: pd.DataFrame,
        prediction_timestamp: datetime,
        timestamp_col: str
    ) -> Dict[str, Any]:
        """Analyze temporal quality of targets for HMM training."""
        analysis = {
            'target_range': {},
            'prediction_horizon': {},
            'target_freshness': {},
            'violations': []
        }

        if timestamp_col not in target_df.columns:
            analysis['violations'].append(f"Timestamp column '{timestamp_col}' not found")
            return analysis

        # Convert timestamps if needed
        target_df = self.lookahead_protection._ensure_timestamp_format(target_df, timestamp_col)
        timestamps = target_df[timestamp_col].dropna()

        if len(timestamps) == 0:
            analysis['violations'].append("No valid target timestamps found")
            return analysis

        # Target range analysis
        analysis['target_range'] = {
            'earliest': timestamps.min(),
            'latest': timestamps.max(),
            'total_span': timestamps.max() - timestamps.min(),
            'prediction_time': prediction_timestamp
        }

        # Prediction horizon analysis
        if 'target' in target_df.columns:
            prediction_horizon = timestamps.max() - prediction_timestamp
            analysis['prediction_horizon'] = {
                'horizon': prediction_horizon,
                'is_sufficient': prediction_horizon >= self.hmm_temporal_constraints['min_prediction_horizon'],
                'horizon_minutes': prediction_horizon.total_seconds() / 60
            }

        # Target freshness analysis
        if 'target' in target_df.columns:
            target_data = target_df['target']
            valid_timestamps = timestamps[~target_data.isnull()]

            if len(valid_timestamps) > 0:
                target_freshness = prediction_timestamp - valid_timestamps.max()
                analysis['target_freshness'] = {
                    'last_target_update': valid_timestamps.max(),
                    'freshness': target_freshness,
                    'is_fresh': target_freshness <= self.hmm_temporal_constraints['feature_freshness_threshold']
                }

        # Check for violations
        if not analysis['prediction_horizon'].get('is_sufficient', True):
            analysis['violations'].append("Insufficient prediction horizon")

        if not analysis['target_freshness'].get('is_fresh', True):
            analysis['violations'].append("Target data is stale")

        return analysis

    def _analyze_state_transition_temporal(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        timestamp_col: str
    ) -> Dict[str, Any]:
        """Analyze temporal aspects of state transitions."""
        analysis = {
            'transition_temporal_analysis': {},
            'transition_stability': {},
            'violations': []
        }

        if 'target' not in target_df.columns:
            analysis['violations'].append("No target column found for state transition analysis")
            return analysis

        # Get state transitions with timestamps
        target_states = target_df['target'].values
        timestamps = target_df[timestamp_col].values

        if len(target_states) < 2:
            return analysis

        # Analyze transition timing
        transition_times = []
        transition_states = []

        for i in range(len(target_states) - 1):
            if target_states[i] != target_states[i + 1]:
                transition_time = timestamps[i + 1] - timestamps[i]
                transition_times.append(transition_time)
                transition_states.append((target_states[i], target_states[i + 1]))

        if transition_times:
            analysis['transition_temporal_analysis'] = {
                'mean_transition_time': np.mean(transition_times),
                'min_transition_time': np.min(transition_times),
                'max_transition_time': np.max(transition_times),
                'transition_frequency': len(transition_times) / len(target_states)
            }

            # Check transition stability
            min_transition_time = np.min(transition_times)
            analysis['transition_stability'] = {
                'is_stable': min_transition_time >= self.hmm_temporal_constraints['state_transition_buffer'],
                'min_transition_time': min_transition_time
            }

        # Check for violations
        if not analysis['transition_stability'].get('is_stable', True):
            analysis['violations'].append("State transitions too rapid - consider temporal aggregation")

        return analysis

    def _analyze_regime_stability_temporal(
        self,
        features_df: pd.DataFrame,
        target_df: pd.DataFrame,
        timestamp_col: str
    ) -> Dict[str, Any]:
        """Analyze temporal stability of regimes."""
        analysis = {
            'regime_temporal_stability': {},
            'regime_duration_analysis': {},
            'violations': []
        }

        if 'target' not in target_df.columns:
            return analysis

        # Analyze regime durations
        target_states = target_df['target'].values
        timestamps = target_df[timestamp_col].values

        if len(target_states) == 0:
            return analysis

        # Calculate regime durations
        regime_durations = []
        current_regime = target_states[0]
        regime_start = timestamps[0]

        for i in range(1, len(target_states)):
            if target_states[i] != current_regime:
                duration = timestamps[i] - regime_start
                regime_durations.append({
                    'regime': current_regime,
                    'duration': duration,
                    'start_time': regime_start,
                    'end_time': timestamps[i]
                })

                current_regime = target_states[i]
                regime_start = timestamps[i]

        # Add final regime
        final_duration = timestamps[-1] - regime_start
        regime_durations.append({
            'regime': current_regime,
            'duration': final_duration,
            'start_time': regime_start,
            'end_time': timestamps[-1]
        })

        # Analyze regime stability
        if regime_durations:
            durations = [rd['duration'] for rd in regime_durations]
            analysis['regime_duration_analysis'] = {
                'mean_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'duration_variance': np.var(durations)
            }

            # Check stability
            min_duration = np.min(durations)
            analysis['regime_temporal_stability'] = {
                'is_stable': min_duration >= self.hmm_temporal_constraints['regime_stability_period'],
                'min_regime_duration': min_duration,
                'regime_count': len(regime_durations)
            }

        # Check for violations
        if not analysis['regime_temporal_stability'].get('is_stable', True):
            analysis['violations'].append("Regime durations too short - consider temporal smoothing")

        return analysis

    def _aggregate_temporal_violations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Aggregate temporal constraint violations."""
        violations = []

        # Feature violations
        feature_analysis = validation_results.get('feature_temporal_analysis', {})
        violations.extend(feature_analysis.get('violations', []))

        # Target violations
        target_analysis = validation_results.get('target_temporal_analysis', {})
        violations.extend(target_analysis.get('violations', []))

        # State transition violations
        state_analysis = validation_results.get('state_transition_analysis', {})
        violations.extend(state_analysis.get('violations', []))

        # Regime stability violations
        regime_analysis = validation_results.get('regime_stability_analysis', {})
        violations.extend(regime_analysis.get('violations', []))

        return violations

    def _generate_temporal_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate temporal recommendations based on validation results."""
        recommendations = []

        violations = validation_results.get('constraint_violations', [])
        if not violations:
            recommendations.append("✅ All temporal constraints satisfied")
            return recommendations

        # Specific recommendations based on violation types
        violation_text = ' '.join(violations).lower()

        if 'lookback' in violation_text:
            recommendations.append("Reduce feature lookback window or use more recent data")

        if 'freshness' in violation_text or 'stale' in violation_text:
            recommendations.append("Ensure features are updated within freshness threshold")

        if 'transition' in violation_text:
            recommendations.append("Consider temporal aggregation for state transitions")

        if 'regime' in violation_text:
            recommendations.append("Apply temporal smoothing to stabilize regime detection")

        if 'horizon' in violation_text:
            recommendations.append("Increase prediction horizon or adjust temporal constraints")

        # General temporal recommendations
        recommendations.extend([
            "Implement automated temporal validation in data pipeline",
            "Add temporal integrity checks to feature engineering",
            "Consider time-series cross-validation for temporal stability"
        ])

        return recommendations

    def _create_feature_dataframe(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]],
        timestamps: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        """Create DataFrame for feature analysis."""
        # Create DataFrame with features
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['timestamp'] = pd.to_datetime(timestamps)

        return df

    def _analyze_hmm_specific_bias_patterns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        hmm_states: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze HMM-specific bias patterns."""
        analysis = {
            'bias_detected': False,
            'bias_confidence': 0.0,
            'bias_patterns': [],
            'temporal_bias_analysis': {},
            'state_transition_bias': {}
        }

        try:
            # Analyze temporal patterns in HMM states
            if len(y) > 1:
                # Check for unrealistic state transitions
                state_changes = np.sum(y[1:] != y[:-1])
                change_rate = state_changes / len(y)

                if change_rate > 0.8:  # More than 80% state changes
                    analysis['bias_patterns'].append("Unrealistically frequent state transitions")
                    analysis['bias_detected'] = True
                    analysis['bias_confidence'] += 0.3

                # Analyze state transition timing
                if len(timestamps) > 1:
                    time_diffs = np.diff(pd.to_datetime(timestamps))
                    if len(time_diffs) > 0:
                        mean_time_diff = time_diffs.mean()
                        analysis['temporal_bias_analysis'] = {
                            'mean_state_duration': mean_time_diff,
                            'state_change_rate': change_rate
                        }

            # Analyze feature-state correlations
            if X.shape[1] > 0:
                # Check for features that perfectly predict state changes
                for i in range(X.shape[1]):
                    feature_state_corr = np.corrcoef(X[:, i], y)[0, 1]
                    if abs(feature_state_corr) > 0.95:  # Very high correlation
                        analysis['bias_patterns'].append(f"Feature {i} highly correlated with HMM states")
                        analysis['bias_detected'] = True
                        analysis['bias_confidence'] += 0.4

        except Exception as e:
            self.logger.warning(f"⚠️ HMM-specific bias analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _generate_bias_recommendations(self, bias_results: Dict[str, Any]) -> List[str]:
        """Generate bias-specific recommendations."""
        recommendations = []

        if not bias_results.get('overall_bias_detected', False):
            recommendations.append("✅ No HMM-specific bias detected")
            return recommendations

        # General bias recommendations
        recommendations.append("🔧 Implement stricter temporal validation")

        # Specific recommendations based on bias patterns
        bias_patterns = bias_results.get('hmm_specific_bias_analysis', {}).get('bias_patterns', [])

        for pattern in bias_patterns:
            if 'transition' in pattern.lower():
                recommendations.append("Consider temporal smoothing for state transitions")
            if 'correlated' in pattern.lower():
                recommendations.append("Review feature engineering for potential target leakage")

        # Add general recommendations
        recommendations.extend([
            "Implement rolling window validation for temporal stability",
            "Add temporal cross-validation to training pipeline",
            "Consider feature lag engineering for temporal relationships"
        ])

        return recommendations

    def _apply_temporal_constraints(
        self,
        df: pd.DataFrame,
        current_timestamp: datetime,
        timestamp_col: str,
        filter_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply temporal constraints to DataFrame."""
        filtered_df = df.copy()

        # Apply max lookback constraint
        if filter_config.get('max_lookback_days', 0) > 0:
            max_lookback = current_timestamp - timedelta(days=filter_config['max_lookback_days'])
            filtered_df = filtered_df[filtered_df[timestamp_col] >= max_lookback]

        return filtered_df

    def _filter_regime_stable_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Filter data to ensure regime stability."""
        if 'target' not in df.columns:
            return df

        # Simple regime stability filter - keep only data where regime lasts long enough
        target_states = df['target'].values
        timestamps = df[timestamp_col].values

        # Find stable regime periods
        stable_indices = []

        for i in range(len(target_states)):
            # Check if current regime is stable (same state for sufficient duration)
            current_state = target_states[i]
            current_time = timestamps[i]

            # Look ahead to see if regime persists
            regime_persists = True
            for j in range(i + 1, min(i + 10, len(target_states))):  # Check next 10 steps
                time_diff = timestamps[j] - current_time
                if time_diff > self.hmm_temporal_constraints['regime_stability_period']:
                    break
                if target_states[j] != current_state:
                    regime_persists = False
                    break

            if regime_persists:
                stable_indices.append(i)

        if stable_indices:
            return df.iloc[stable_indices].copy()
        else:
            return df  # Return original if no stable periods found

    def _filter_fresh_features(
        self,
        df: pd.DataFrame,
        current_timestamp: datetime,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Filter data to ensure feature freshness."""
        freshness_threshold = self.hmm_temporal_constraints['feature_freshness_threshold']

        # Remove rows where features are too old
        valid_mask = df[timestamp_col] >= (current_timestamp - freshness_threshold)

        return df[valid_mask].copy()

    def _log_filtering_results(self, filtered_datasets: Dict[str, pd.DataFrame]):
        """Log filtering results."""
        original_count = len(filtered_datasets.get('original', pd.DataFrame()))
        final_count = len(filtered_datasets.get('fresh_features') or
                         filtered_datasets.get('regime_stable') or
                         filtered_datasets.get('temporal_constraints_applied') or
                         pd.DataFrame())

        self.logger.info(f"📊 Temporal filtering results: {original_count} → {final_count} rows "
                       f"({(final_count/original_count*100):.1f}% retained)")

# Global instance
_hmm_temporal_protection_instance = None

def get_hmm_temporal_protection(config: Optional[HMMTrainingConfig] = None) -> HMMTemporalProtection:
    """Get global HMM temporal protection instance."""
    global _hmm_temporal_protection_instance
    if _hmm_temporal_protection_instance is None:
        _hmm_temporal_protection_instance = HMMTemporalProtection(config)
    return _hmm_temporal_protection_instance

# Export key classes and functions
__all__ = ['HMMTemporalProtection', 'get_hmm_temporal_protection']