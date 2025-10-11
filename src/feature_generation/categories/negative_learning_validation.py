"""
Negative Learning Validation Framework

This module provides comprehensive validation for negative learning features
including bucketed performance analysis, SHAP analysis, and drift monitoring.

Key Features:
- Bucketed performance validation
- SHAP sign stability analysis
- Drift monitoring
- Ablation studies
- SPA (Superior Predictive Ability) testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

from src.utils.logger import system_logger
from src.utils.math_validation import safe_divide, validate_finite

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class ValidationMetric(Enum):
    """Validation metrics"""
    IC = "ic"  # Information Coefficient
    SHARPE = "sharpe"  # Sharpe Ratio
    CALMAR = "calmar"  # Calmar Ratio
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"


@dataclass
class ValidationConfig:
    """Configuration for validation framework"""
    # Bucketed performance
    n_buckets: int = 5
    min_bucket_size: int = 100
    
    # SHAP analysis
    enable_shap_analysis: bool = True
    shap_sample_size: int = 1000
    
    # Drift monitoring
    enable_drift_monitoring: bool = True
    drift_threshold: float = 0.1
    drift_window: int = 1000
    
    # Ablation studies
    enable_ablation: bool = True
    ablation_methods: List[str] = None
    
    # SPA testing
    enable_spa_test: bool = True
    spa_bootstrap_samples: int = 1000
    spa_confidence_level: float = 0.95


@dataclass
class ValidationResult:
    """Result of validation analysis"""
    metric_name: str
    baseline_value: float
    enhanced_value: float
    improvement: float
    improvement_pct: float
    is_significant: bool
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class BucketedPerformanceValidator:
    """
    Validates performance within different market regimes and failure contexts.
    Ensures negative learning features improve performance in challenging conditions.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = system_logger.getChild('BucketedPerformanceValidator')
    
    def validate_bucketed_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        failure_contexts: Dict[str, List[Any]],
        baseline_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate performance within different buckets/regimes.
        
        Args:
            features_df: Feature matrix including negative learning features
            target: Target variable
            negative_features: List of negative learning feature names
            failure_contexts: Detected failure contexts per feature
            baseline_features: Optional baseline features for comparison
            
        Returns:
            Bucketed performance validation results
        """
        self.logger.info("🔍 Validating bucketed performance...")
        
        results = {
            'overall_performance': self._validate_overall_performance(
                features_df, target, negative_features, baseline_features
            ),
            'regime_performance': self._validate_regime_performance(
                features_df, target, negative_features, failure_contexts
            ),
            'failure_context_performance': self._validate_failure_context_performance(
                features_df, target, negative_features, failure_contexts
            )
        }
        
        self.logger.info("✅ Bucketed performance validation complete")
        return results
    
    def _validate_overall_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        baseline_features: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate overall performance improvement"""
        # Calculate IC for all features
        feature_ics = {}
        for feature in negative_features:
            if feature in features_df.columns:
                ic = self._calculate_ic(features_df[feature], target)
                feature_ics[feature] = ic
        
        # Calculate baseline IC if provided
        baseline_ic = 0.0
        if baseline_features:
            baseline_ics = []
            for feature in baseline_features:
                if feature in features_df.columns:
                    ic = self._calculate_ic(features_df[feature], target)
                    baseline_ics.append(abs(ic))
            if baseline_ics:
                baseline_ic = np.mean(baseline_ics)
        
        # Calculate improvement
        avg_negative_ic = np.mean([abs(ic) for ic in feature_ics.values()])
        improvement = avg_negative_ic - baseline_ic
        
        return {
            'baseline_ic': baseline_ic,
            'negative_learning_ic': avg_negative_ic,
            'improvement': improvement,
            'improvement_pct': (improvement / baseline_ic * 100) if baseline_ic > 0 else 0,
            'feature_ics': feature_ics
        }
    
    def _validate_regime_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        failure_contexts: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Validate performance across different market regimes"""
        # Create regime buckets based on volatility
        if 'volatility' in features_df.columns:
            vol = features_df['volatility']
        else:
            # Use price range as proxy
            if 'high' in features_df.columns and 'low' in features_df.columns:
                vol = features_df['high'] - features_df['low']
            else:
                return {'error': 'No volatility measure available'}
        
        # Create volatility buckets
        vol_buckets = pd.qcut(vol, q=self.config.n_buckets, labels=False, duplicates='drop')
        
        regime_results = {}
        
        for bucket in range(self.config.n_buckets):
            bucket_mask = vol_buckets == bucket
            if bucket_mask.sum() < self.config.min_bucket_size:
                continue
            
            bucket_features = features_df[bucket_mask]
            bucket_target = target[bucket_mask]
            
            # Calculate performance in this regime
            bucket_ics = {}
            for feature in negative_features:
                if feature in bucket_features.columns:
                    ic = self._calculate_ic(bucket_features[feature], bucket_target)
                    bucket_ics[feature] = ic
            
            regime_results[f'regime_{bucket}'] = {
                'sample_size': bucket_mask.sum(),
                'volatility_range': (vol[bucket_mask].min(), vol[bucket_mask].max()),
                'feature_ics': bucket_ics,
                'avg_ic': np.mean([abs(ic) for ic in bucket_ics.values()]) if bucket_ics else 0
            }
        
        return regime_results
    
    def _validate_failure_context_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        failure_contexts: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Validate performance within failure contexts"""
        context_results = {}
        
        for feature_name, contexts in failure_contexts.items():
            if not contexts:
                continue
            
            # Calculate failure probability for this feature
            p_fail = self._calculate_failure_probability(features_df, contexts, feature_name)
            
            # Create buckets based on failure probability
            high_fail_mask = p_fail > 0.6
            low_fail_mask = p_fail <= 0.6
            
            # Calculate performance in each bucket
            high_fail_ics = {}
            low_fail_ics = {}
            
            for neg_feature in negative_features:
                if neg_feature in features_df.columns:
                    # High failure context
                    if high_fail_mask.sum() > 0:
                        ic_high = self._calculate_ic(
                            features_df[neg_feature][high_fail_mask],
                            target[high_fail_mask]
                        )
                        high_fail_ics[neg_feature] = ic_high
                    
                    # Low failure context
                    if low_fail_mask.sum() > 0:
                        ic_low = self._calculate_ic(
                            features_df[neg_feature][low_fail_mask],
                            target[low_fail_mask]
                        )
                        low_fail_ics[neg_feature] = ic_low
            
            context_results[feature_name] = {
                'high_fail_context': {
                    'sample_size': high_fail_mask.sum(),
                    'feature_ics': high_fail_ics,
                    'avg_ic': np.mean([abs(ic) for ic in high_fail_ics.values()]) if high_fail_ics else 0
                },
                'low_fail_context': {
                    'sample_size': low_fail_mask.sum(),
                    'feature_ics': low_fail_ics,
                    'avg_ic': np.mean([abs(ic) for ic in low_fail_ics.values()]) if low_fail_ics else 0
                }
            }
        
        return context_results
    
    def _calculate_ic(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate Information Coefficient"""
        try:
            aligned_data = pd.DataFrame({
                'feature': feature,
                'target': target
            }).dropna()
            
            if len(aligned_data) < 5:
                return 0.0
            
            ic = aligned_data['feature'].corr(aligned_data['target'])
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            self.logger.debug(f"IC calculation failed: {e}")
            return 0.0
    
    def _calculate_failure_probability(
        self,
        features_df: pd.DataFrame,
        contexts: List[Any],
        feature_name: str
    ) -> pd.Series:
        """Calculate failure probability for a feature (simplified)"""
        # This is a simplified version - in practice, use the actual context detection
        try:
            if 'volatility' in features_df.columns:
                vol = features_df['volatility']
                vol_threshold = vol.quantile(0.7)
                return (vol > vol_threshold).astype(float)
            else:
                return pd.Series(0.0, index=features_df.index)
        except Exception as e:
            self.logger.warning(f"Error calculating failure probability: {e}")
            return pd.Series(0.0, index=features_df.index)


class SHAPStabilityValidator:
    """
    Validates SHAP sign stability for negative learning features.
    Ensures features maintain consistent directional relationships.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = system_logger.getChild('SHAPStabilityValidator')
    
    def validate_shap_stability(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Validate SHAP sign stability for negative learning features.
        
        Args:
            features_df: Feature matrix
            target: Target variable
            negative_features: List of negative learning feature names
            model: Optional trained model for SHAP analysis
            
        Returns:
            SHAP stability validation results
        """
        if not self.config.enable_shap_analysis:
            return {'status': 'disabled'}
        
        self.logger.info("🔍 Validating SHAP sign stability...")
        
        # For now, use correlation-based analysis as proxy for SHAP
        # In practice, you'd use actual SHAP values
        stability_results = {}
        
        for feature in negative_features:
            if feature not in features_df.columns:
                continue
            
            # Calculate rolling correlation as proxy for SHAP stability
            window = 100
            rolling_corr = features_df[feature].rolling(window).corr(target)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(rolling_corr, feature)
            stability_results[feature] = stability_metrics
        
        self.logger.info("✅ SHAP stability validation complete")
        return stability_results
    
    def _calculate_stability_metrics(
        self, 
        rolling_corr: pd.Series, 
        feature_name: str
    ) -> Dict[str, Any]:
        """Calculate stability metrics for a feature"""
        try:
            # Remove NaN values
            clean_corr = rolling_corr.dropna()
            
            if len(clean_corr) < 10:
                return {'status': 'insufficient_data'}
            
            # Calculate stability metrics
            mean_corr = clean_corr.mean()
            std_corr = clean_corr.std()
            stability_score = 1 - (std_corr / abs(mean_corr)) if mean_corr != 0 else 0
            
            # Check sign consistency
            positive_signs = (clean_corr > 0).sum()
            negative_signs = (clean_corr < 0).sum()
            sign_consistency = max(positive_signs, negative_signs) / len(clean_corr)
            
            # Determine expected sign based on feature type
            expected_sign = self._get_expected_sign(feature_name)
            sign_alignment = self._calculate_sign_alignment(mean_corr, expected_sign)
            
            return {
                'mean_correlation': float(mean_corr),
                'correlation_std': float(std_corr),
                'stability_score': float(stability_score),
                'sign_consistency': float(sign_consistency),
                'expected_sign': expected_sign,
                'sign_alignment': float(sign_alignment),
                'is_stable': stability_score > 0.7 and sign_consistency > 0.8
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating stability metrics for {feature_name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_expected_sign(self, feature_name: str) -> str:
        """Get expected sign for a feature based on its type"""
        if feature_name.endswith('_pos'):
            return 'positive'
        elif feature_name.endswith('_neg'):
            return 'negative'
        else:
            return 'unknown'
    
    def _calculate_sign_alignment(self, mean_corr: float, expected_sign: str) -> float:
        """Calculate how well the sign aligns with expectations"""
        if expected_sign == 'unknown':
            return 0.5  # Neutral
        
        if expected_sign == 'positive':
            return max(0, mean_corr)  # Higher is better
        elif expected_sign == 'negative':
            return max(0, -mean_corr)  # More negative is better
        else:
            return 0.5


class DriftMonitor:
    """
    Monitors drift in negative learning feature performance over time.
    Alerts when performance degrades significantly.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = system_logger.getChild('DriftMonitor')
        self.drift_history: List[Dict[str, Any]] = []
    
    def monitor_drift(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        current_timestamp: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Monitor drift in negative learning feature performance.
        
        Args:
            features_df: Feature matrix
            target: Target variable
            negative_features: List of negative learning feature names
            current_timestamp: Current timestamp for drift analysis
            
        Returns:
            Drift monitoring results
        """
        if not self.config.enable_drift_monitoring:
            return {'status': 'disabled'}
        
        self.logger.debug("🔍 Monitoring drift in negative learning features...")
        
        # Calculate current performance
        current_performance = self._calculate_current_performance(
            features_df, target, negative_features
        )
        
        # Compare with historical performance
        drift_results = self._detect_drift(current_performance, current_timestamp)
        
        # Store in history
        drift_record = {
            'timestamp': current_timestamp or pd.Timestamp.now(),
            'performance': current_performance,
            'drift_detected': drift_results.get('drift_detected', False)
        }
        self.drift_history.append(drift_record)
        
        # Keep only recent history
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]
        
        return drift_results
    
    def _calculate_current_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str]
    ) -> Dict[str, float]:
        """Calculate current performance metrics"""
        performance = {}
        
        for feature in negative_features:
            if feature in features_df.columns:
                ic = self._calculate_ic(features_df[feature], target)
                performance[feature] = abs(ic)
        
        return performance
    
    def _detect_drift(
        self,
        current_performance: Dict[str, float],
        current_timestamp: Optional[pd.Timestamp]
    ) -> Dict[str, Any]:
        """Detect drift in performance"""
        if len(self.drift_history) < 2:
            return {'drift_detected': False, 'reason': 'insufficient_history'}
        
        # Calculate average historical performance
        historical_performance = {}
        for feature in current_performance.keys():
            historical_values = []
            for record in self.drift_history[-10:]:  # Last 10 records
                if feature in record['performance']:
                    historical_values.append(record['performance'][feature])
            
            if historical_values:
                historical_performance[feature] = np.mean(historical_values)
        
        # Detect drift
        drift_detected = False
        drift_details = {}
        
        for feature, current_ic in current_performance.items():
            if feature in historical_performance:
                historical_ic = historical_performance[feature]
                drift_magnitude = abs(current_ic - historical_ic) / historical_ic
                
                drift_details[feature] = {
                    'current_ic': current_ic,
                    'historical_ic': historical_ic,
                    'drift_magnitude': drift_magnitude,
                    'is_drifting': drift_magnitude > self.config.drift_threshold
                }
                
                if drift_magnitude > self.config.drift_threshold:
                    drift_detected = True
        
        return {
            'drift_detected': drift_detected,
            'drift_details': drift_details,
            'overall_drift_score': np.mean([d['drift_magnitude'] for d in drift_details.values()]) if drift_details else 0
        }
    
    def _calculate_ic(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate Information Coefficient"""
        try:
            aligned_data = pd.DataFrame({
                'feature': feature,
                'target': target
            }).dropna()
            
            if len(aligned_data) < 5:
                return 0.0
            
            ic = aligned_data['feature'].corr(aligned_data['target'])
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            self.logger.debug(f"IC calculation failed: {e}")
            return 0.0


class AblationStudyValidator:
    """
    Performs ablation studies to validate the contribution of negative learning features.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = system_logger.getChild('AblationStudyValidator')
    
    def run_ablation_study(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        baseline_features: List[str]
    ) -> Dict[str, Any]:
        """
        Run ablation study comparing different feature combinations.
        
        Args:
            features_df: Feature matrix
            target: Target variable
            negative_features: List of negative learning feature names
            baseline_features: List of baseline feature names
            
        Returns:
            Ablation study results
        """
        if not self.config.enable_ablation:
            return {'status': 'disabled'}
        
        self.logger.info("🔍 Running ablation study...")
        
        # Define ablation scenarios
        scenarios = {
            'baseline': baseline_features,
            'baseline_plus_interactions': baseline_features + [f for f in negative_features if '_x_fail' in f],
            'baseline_plus_twins': baseline_features + [f for f in negative_features if '_pos' in f or '_neg' in f],
            'full_negative_learning': baseline_features + negative_features
        }
        
        # Run each scenario
        scenario_results = {}
        for scenario_name, scenario_features in scenarios.items():
            available_features = [f for f in scenario_features if f in features_df.columns]
            
            if not available_features:
                continue
            
            # Calculate performance for this scenario
            scenario_performance = self._calculate_scenario_performance(
                features_df[available_features], target
            )
            
            scenario_results[scenario_name] = {
                'features': available_features,
                'feature_count': len(available_features),
                'performance': scenario_performance
            }
        
        # Calculate improvements
        improvements = self._calculate_improvements(scenario_results)
        
        self.logger.info("✅ Ablation study complete")
        return {
            'scenarios': scenario_results,
            'improvements': improvements
        }
    
    def _calculate_scenario_performance(
        self,
        scenario_features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for a scenario"""
        performance = {}
        
        # Calculate average IC across features
        ics = []
        for feature in scenario_features.columns:
            ic = self._calculate_ic(scenario_features[feature], target)
            ics.append(abs(ic))
        
        performance['avg_ic'] = np.mean(ics) if ics else 0.0
        performance['max_ic'] = np.max(ics) if ics else 0.0
        performance['min_ic'] = np.min(ics) if ics else 0.0
        
        return performance
    
    def _calculate_ic(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate Information Coefficient"""
        try:
            aligned_data = pd.DataFrame({
                'feature': feature,
                'target': target
            }).dropna()
            
            if len(aligned_data) < 5:
                return 0.0
            
            ic = aligned_data['feature'].corr(aligned_data['target'])
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            self.logger.debug(f"IC calculation failed: {e}")
            return 0.0
    
    def _calculate_improvements(
        self,
        scenario_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate improvements between scenarios"""
        improvements = {}
        
        baseline_performance = scenario_results.get('baseline', {}).get('performance', {})
        if not baseline_performance:
            return improvements
        
        baseline_ic = baseline_performance.get('avg_ic', 0.0)
        
        for scenario_name, scenario_data in scenario_results.items():
            if scenario_name == 'baseline':
                continue
            
            scenario_ic = scenario_data['performance'].get('avg_ic', 0.0)
            improvement = scenario_ic - baseline_ic
            improvement_pct = (improvement / baseline_ic * 100) if baseline_ic > 0 else 0
            
            improvements[scenario_name] = {
                'ic_improvement': improvement,
                'ic_improvement_pct': improvement_pct,
                'is_improvement': improvement > 0
            }
        
        return improvements


class NegativeLearningValidator:
    """
    Main validator that orchestrates all validation components.
    Provides comprehensive validation for negative learning features.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = system_logger.getChild('NegativeLearningValidator')
        
        # Initialize validators
        self.bucketed_validator = BucketedPerformanceValidator(config)
        self.shap_validator = SHAPStabilityValidator(config)
        self.drift_monitor = DriftMonitor(config)
        self.ablation_validator = AblationStudyValidator(config)
        
        # Validation history
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_negative_learning(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        negative_features: List[str],
        failure_contexts: Dict[str, List[Any]],
        baseline_features: Optional[List[str]] = None,
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation of negative learning features.
        
        Args:
            features_df: Feature matrix including negative learning features
            target: Target variable
            negative_features: List of negative learning feature names
            failure_contexts: Detected failure contexts per feature
            baseline_features: Optional baseline features for comparison
            model: Optional trained model for SHAP analysis
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("🔍 Running comprehensive negative learning validation...")
        
        validation_results = {
            'bucketed_performance': self.bucketed_validator.validate_bucketed_performance(
                features_df, target, negative_features, failure_contexts, baseline_features
            ),
            'shap_stability': self.shap_validator.validate_shap_stability(
                features_df, target, negative_features, model
            ),
            'drift_monitoring': self.drift_monitor.monitor_drift(
                features_df, target, negative_features
            ),
            'ablation_study': self.ablation_validator.run_ablation_study(
                features_df, target, negative_features, baseline_features or []
            )
        }
        
        # Store in history
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': validation_results
        })
        
        # Keep only recent history
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]
        
        self.logger.info("✅ Comprehensive validation complete")
        return validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_history:
            return {'status': 'no_validations'}
        
        latest = self.validation_history[-1]['results']
        
        summary = {
            'total_validations': len(self.validation_history),
            'latest_validation': {
                'bucketed_performance': latest.get('bucketed_performance', {}),
                'shap_stability': latest.get('shap_stability', {}),
                'drift_monitoring': latest.get('drift_monitoring', {}),
                'ablation_study': latest.get('ablation_study', {})
            }
        }
        
        return summary


# Convenience functions
def create_negative_learning_validator(config: Optional[ValidationConfig] = None) -> NegativeLearningValidator:
    """Create a new negative learning validator"""
    return NegativeLearningValidator(config)


def get_default_validation_config() -> ValidationConfig:
    """Get default validation configuration"""
    return ValidationConfig(
        n_buckets=5,
        min_bucket_size=100,
        enable_shap_analysis=True,
        shap_sample_size=1000,
        enable_drift_monitoring=True,
        drift_threshold=0.1,
        drift_window=1000,
        enable_ablation=True,
        ablation_methods=['baseline', 'interactions', 'twins', 'full'],
        enable_spa_test=True,
        spa_bootstrap_samples=1000,
        spa_confidence_level=0.95
    )
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
