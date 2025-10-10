"""
Feature Acceleration and Window Dilation

This module implements feature acceleration (Δ over k or 2nd difference) and 
window dilation (3× lookback) comparisons to expose turning points and 
capture different regime/trend complements.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import entropy
from scipy.stats import normaltest, jarque_bera
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
from itertools import combinations
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

class FeatureAccelerationDilation:
    """
    Feature acceleration and window dilation comparison system.
    """
    
    def __init__(self, 
                 acceleration_lags: List[int] = [1, 3],
                 dilation_factors: List[float] = [2.0, 3.0],
                 mi_threshold: float = 0.6,
                 correlation_threshold: float = 0.9,
                 conditional_mi_threshold: float = 0.6,
                 rank_std_threshold: float = 0.25,
                 fqs_improvement_threshold: float = 0.05,
                 enable_matrix_ops: bool = True):
        """
        Initialize feature acceleration and dilation system.
        
        Args:
            acceleration_lags: Lags for acceleration calculation (k values)
            dilation_factors: Window dilation factors (2×, 3×, etc.)
            mi_threshold: MI threshold for signal acceptance
            correlation_threshold: Correlation threshold for uniqueness
            conditional_mi_threshold: Conditional MI threshold for incremental value
            rank_std_threshold: Rank standard deviation threshold for stability
            fqs_improvement_threshold: FQS improvement threshold for acceptance
            enable_matrix_ops: Whether to enable matrix operations
        """
        self.acceleration_lags = acceleration_lags
        self.dilation_factors = dilation_factors
        self.mi_threshold = mi_threshold
        self.correlation_threshold = correlation_threshold
        self.conditional_mi_threshold = conditional_mi_threshold
        self.rank_std_threshold = rank_std_threshold
        self.fqs_improvement_threshold = fqs_improvement_threshold
        self.enable_matrix_ops = enable_matrix_ops
        
        # Initialize matrix operations if available
        if enable_matrix_ops:
            try:
                from src.utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations(enable_gpu=True, enable_parallel=True)
                self.matrix_available = True
            except ImportError:
                self.matrix_ops = None
                self.matrix_available = False
                logger.warning("Matrix operations not available, using standard operations")
        else:
            self.matrix_ops = None
            self.matrix_available = False
    
    def generate_acceleration_features(self, X: pd.DataFrame, 
                                     base_features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate acceleration features for base features.
        
        Args:
            X: Feature matrix
            base_features: List of base features to accelerate (if None, auto-detect)
            
        Returns:
            Dictionary with acceleration features by lag
        """
        logger.info("Generating acceleration features...")
        
        if base_features is None:
            base_features = self._identify_acceleration_candidates(X)
        
        acceleration_features = {}
        
        for lag in self.acceleration_lags:
            lag_features = {}
            
            for feature in base_features:
                if feature not in X.columns:
                    continue
                
                # Check if feature is suitable for acceleration
                if not self._is_suitable_for_acceleration(X[feature]):
                    continue
                
                # Generate acceleration feature
                accel_feature = self._calculate_acceleration(X[feature], lag)
                if accel_feature is not None:
                    lag_features[f"{feature}_accel_{lag}"] = accel_feature
            
            if lag_features:
                acceleration_features[f"lag_{lag}"] = pd.DataFrame(lag_features, index=X.index)
        
        logger.info(f"Generated acceleration features for {len(base_features)} base features")
        return acceleration_features
    
    def generate_dilation_features(self, X: pd.DataFrame, 
                                 base_features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate window dilation features for base features.
        
        Args:
            X: Feature matrix
            base_features: List of base features to dilate (if None, auto-detect)
            
        Returns:
            Dictionary with dilation features by factor
        """
        logger.info("Generating window dilation features...")
        
        if base_features is None:
            base_features = self._identify_dilation_candidates(X)
        
        dilation_features = {}
        
        for factor in self.dilation_factors:
            factor_features = {}
            
            for feature in base_features:
                if feature not in X.columns:
                    continue
                
                # Check if feature is suitable for dilation
                if not self._is_suitable_for_dilation(X[feature]):
                    continue
                
                # Generate dilation feature
                dilated_feature = self._calculate_dilation(X[feature], factor)
                if dilated_feature is not None:
                    factor_features[f"{feature}_dil_{factor}x"] = dilated_feature
            
            if factor_features:
                dilation_features[f"factor_{factor}"] = pd.DataFrame(factor_features, index=X.index)
        
        logger.info(f"Generated dilation features for {len(base_features)} base features")
        return dilation_features
    
    def evaluate_acceleration_features(self, X: pd.DataFrame, y: pd.Series,
                                     acceleration_features: Dict[str, pd.DataFrame],
                                     base_features: List[str]) -> Dict[str, Any]:
        """
        Evaluate acceleration features against base features.
        
        Args:
            X: Feature matrix
            y: Target variable
            acceleration_features: Dictionary of acceleration features
            base_features: List of base features
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating acceleration features...")
        
        results = {
            'acceleration_evaluations': {},
            'accepted_features': [],
            'rejected_features': [],
            'evaluation_metrics': {}
        }
        
        for lag_key, accel_df in acceleration_features.items():
            lag = int(lag_key.split('_')[1])
            lag_results = {}
            
            for accel_feature in accel_df.columns:
                base_feature = accel_feature.replace(f'_accel_{lag}', '')
                
                if base_feature not in base_features:
                    continue
                
                # Calculate evaluation metrics
                evaluation = self._evaluate_feature_pair(
                    X[base_feature], 
                    accel_df[accel_feature], 
                    y, 
                    base_feature, 
                    accel_feature
                )
                
                lag_results[accel_feature] = evaluation
                
                # Apply acceptance gates
                if self._should_accept_acceleration(evaluation):
                    results['accepted_features'].append(accel_feature)
                else:
                    results['rejected_features'].append(accel_feature)
            
            results['acceleration_evaluations'][lag_key] = lag_results
        
        return results
    
    def evaluate_dilation_features(self, X: pd.DataFrame, y: pd.Series,
                                 dilation_features: Dict[str, pd.DataFrame],
                                 base_features: List[str]) -> Dict[str, Any]:
        """
        Evaluate dilation features against base features.
        
        Args:
            X: Feature matrix
            y: Target variable
            dilation_features: Dictionary of dilation features
            base_features: List of base features
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating dilation features...")
        
        results = {
            'dilation_evaluations': {},
            'accepted_features': [],
            'rejected_features': [],
            'evaluation_metrics': {}
        }
        
        for factor_key, dil_df in dilation_features.items():
            factor = float(factor_key.split('_')[1])
            factor_results = {}
            
            for dil_feature in dil_df.columns:
                base_feature = dil_feature.replace(f'_dil_{factor}x', '')
                
                if base_feature not in base_features:
                    continue
                
                # Calculate evaluation metrics
                evaluation = self._evaluate_feature_pair(
                    X[base_feature], 
                    dil_df[dil_feature], 
                    y, 
                    base_feature, 
                    dil_feature
                )
                
                factor_results[dil_feature] = evaluation
                
                # Apply acceptance gates
                if self._should_accept_dilation(evaluation):
                    results['accepted_features'].append(dil_feature)
                else:
                    results['rejected_features'].append(dil_feature)
            
            results['dilation_evaluations'][factor_key] = factor_results
        
        return results
    
    def _identify_acceleration_candidates(self, X: pd.DataFrame) -> List[str]:
        """Identify features suitable for acceleration."""
        candidates = []
        
        for feature in X.columns:
            if self._is_suitable_for_acceleration(X[feature]):
                candidates.append(feature)
        
        return candidates
    
    def _identify_dilation_candidates(self, X: pd.DataFrame) -> List[str]:
        """Identify features suitable for dilation."""
        candidates = []
        
        for feature in X.columns:
            if self._is_suitable_for_dilation(X[feature]):
                candidates.append(feature)
        
        return candidates
    
    def _is_suitable_for_acceleration(self, feature_series: pd.Series) -> bool:
        """Check if feature is suitable for acceleration."""
        # Check for sufficient data
        if len(feature_series.dropna()) < 50:
            return False
        
        # Check for autocorrelation (persistence)
        try:
            autocorr = feature_series.autocorr(lag=1)
            if pd.isna(autocorr) or autocorr < 0.2:
                return False
        except:
            return False
        
        # Check for bounded features (e.g., RSI-like)
        if feature_series.min() >= 0 and feature_series.max() <= 100:
            return True
        
        # Check for continuous features
        if feature_series.dtype in ['float64', 'float32']:
            return True
        
        return False
    
    def _is_suitable_for_dilation(self, feature_series: pd.Series) -> bool:
        """Check if feature is suitable for dilation."""
        # Check for sufficient data
        if len(feature_series.dropna()) < 100:
            return False
        
        # Check if feature appears to be window-based (has rolling characteristics)
        feature_name = feature_series.name or ""
        
        # Look for window indicators in name
        window_indicators = ['_w', '_ma_', '_ema_', '_std_', '_vol_', '_vwap_', '_bb_']
        if any(indicator in feature_name.lower() for indicator in window_indicators):
            return True
        
        # Check for rolling characteristics
        try:
            # Calculate rolling statistics to see if feature has window-like behavior
            rolling_std = feature_series.rolling(20).std()
            if rolling_std.std() > 0:  # Has variation in rolling behavior
                return True
        except:
            pass
        
        return False
    
    def _calculate_acceleration(self, feature_series: pd.Series, lag: int) -> Optional[pd.Series]:
        """Calculate acceleration feature."""
        try:
            # Center first if bounded (e.g., RSI-50)
            if feature_series.min() >= 0 and feature_series.max() <= 100:
                centered = feature_series - 50
            else:
                centered = feature_series
            
            # Winsorize to reduce noise
            winsorized = self._winsorize(centered, limits=(0.01, 0.99))
            
            # Calculate acceleration: accel_k = base_t - base_{t-k}
            acceleration = winsorized - winsorized.shift(lag)
            
            # Optional re-scaling
            acceleration = self._rescale_acceleration(acceleration)
            
            return acceleration
            
        except Exception as e:
            logger.warning(f"Failed to calculate acceleration for {feature_series.name}: {e}")
            return None
    
    def _calculate_dilation(self, feature_series: pd.Series, factor: float) -> Optional[pd.Series]:
        """Calculate window dilation feature."""
        try:
            # Extract window size from feature name if possible
            feature_name = feature_series.name or ""
            original_window = self._extract_window_size(feature_name)
            
            if original_window is None:
                # Default to 20 if we can't determine window size
                original_window = 20
            
            # Calculate new window size
            new_window = int(original_window * factor)
            
            # Generate dilated feature based on feature type
            if 'ma_' in feature_name.lower():
                # Moving average dilation
                dilated = feature_series.rolling(new_window).mean()
            elif 'ema_' in feature_name.lower():
                # EMA dilation (adjust span)
                span = int(original_window * factor)
                dilated = feature_series.ewm(span=span).mean()
            elif 'std_' in feature_name.lower():
                # Standard deviation dilation
                dilated = feature_series.rolling(new_window).std()
            elif 'vol_' in feature_name.lower() or 'volatility' in feature_name.lower():
                # Volatility dilation
                returns = feature_series.pct_change()
                dilated = returns.rolling(new_window).std()
            else:
                # Generic rolling mean dilation
                dilated = feature_series.rolling(new_window).mean()
            
            return dilated
            
        except Exception as e:
            logger.warning(f"Failed to calculate dilation for {feature_series.name}: {e}")
            return None
    
    def _extract_window_size(self, feature_name: str) -> Optional[int]:
        """Extract window size from feature name."""
        import re
        
        # Look for patterns like _5, _10, _20, etc.
        patterns = [
            r'_(\d+)$',  # _20 at end
            r'_(\d+)_',  # _20_ in middle
            r'w(\d+)',   # w20
            r'(\d+)_',   # 20_ at start
        ]
        
        for pattern in patterns:
            match = re.search(pattern, feature_name)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _winsorize(self, series: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        """Winsorize series to reduce noise."""
        try:
            lower_limit = series.quantile(limits[0])
            upper_limit = series.quantile(limits[1])
            return series.clip(lower=lower_limit, upper=upper_limit)
        except:
            return series
    
    def _rescale_acceleration(self, acceleration: pd.Series) -> pd.Series:
        """Rescale acceleration feature."""
        try:
            # Z-score normalization
            return (acceleration - acceleration.mean()) / (acceleration.std() + 1e-8)
        except:
            return acceleration
    
    def _evaluate_feature_pair(self, base_feature: pd.Series, variant_feature: pd.Series, 
                              y: pd.Series, base_name: str, variant_name: str) -> Dict[str, Any]:
        """Evaluate a feature pair (base vs variant)."""
        try:
            # Align data
            common_idx = base_feature.index.intersection(variant_feature.index).intersection(y.index)
            base_aligned = base_feature.loc[common_idx]
            variant_aligned = variant_feature.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(base_aligned.isna() | variant_aligned.isna() | y_aligned.isna())
            base_clean = base_aligned[valid_mask]
            variant_clean = variant_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            if len(base_clean) < 50:
                return {'error': 'Insufficient data'}
            
            # Calculate metrics
            metrics = {}
            
            # 1. Mutual Information
            try:
                from sklearn.feature_selection import mutual_info_regression
                base_mi = mutual_info_regression(base_clean.values.reshape(-1, 1), y_clean.values)[0]
                variant_mi = mutual_info_regression(variant_clean.values.reshape(-1, 1), y_clean.values)[0]
                metrics['base_mi'] = base_mi
                metrics['variant_mi'] = variant_mi
                metrics['mi_ratio'] = variant_mi / (base_mi + 1e-8)
            except:
                metrics['base_mi'] = 0
                metrics['variant_mi'] = 0
                metrics['mi_ratio'] = 0
            
            # 2. Correlation between base and variant
            try:
                correlation = base_clean.corr(variant_clean)
                metrics['correlation'] = correlation if not pd.isna(correlation) else 0
            except:
                metrics['correlation'] = 0
            
            # 3. Conditional MI (incremental value)
            try:
                from sklearn.feature_selection import mutual_info_regression
                # Combined features
                combined = np.column_stack([base_clean.values, variant_clean.values])
                combined_mi = mutual_info_regression(combined, y_clean.values)[0]
                conditional_mi = combined_mi - base_mi
                metrics['conditional_mi'] = conditional_mi
            except:
                metrics['conditional_mi'] = 0
            
            # 4. Permutation importance (simplified)
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.inspection import permutation_importance
                
                # Base model
                base_model = RandomForestRegressor(n_estimators=50, random_state=42)
                base_model.fit(base_clean.values.reshape(-1, 1), y_clean.values)
                base_perm_imp = permutation_importance(base_model, base_clean.values.reshape(-1, 1), y_clean.values, n_repeats=5)
                metrics['base_perm_imp'] = base_perm_imp.importances_mean[0]
                
                # Variant model
                variant_model = RandomForestRegressor(n_estimators=50, random_state=42)
                variant_model.fit(variant_clean.values.reshape(-1, 1), y_clean.values)
                variant_perm_imp = permutation_importance(variant_model, variant_clean.values.reshape(-1, 1), y_clean.values, n_repeats=5)
                metrics['variant_perm_imp'] = variant_perm_imp.importances_mean[0]
                
                # Combined model
                combined_model = RandomForestRegressor(n_estimators=50, random_state=42)
                combined_model.fit(combined, y_clean.values)
                combined_perm_imp = permutation_importance(combined_model, combined, y_clean.values, n_repeats=5)
                metrics['combined_perm_imp'] = combined_perm_imp.importances_mean[0]
                
            except:
                metrics['base_perm_imp'] = 0
                metrics['variant_perm_imp'] = 0
                metrics['combined_perm_imp'] = 0
            
            # 5. Model performance improvement
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_squared_error
                
                # Base model
                base_model = RandomForestRegressor(n_estimators=50, random_state=42)
                base_model.fit(base_clean.values.reshape(-1, 1), y_clean.values)
                base_pred = base_model.predict(base_clean.values.reshape(-1, 1))
                base_mse = mean_squared_error(y_clean.values, base_pred)
                
                # Combined model
                combined_model = RandomForestRegressor(n_estimators=50, random_state=42)
                combined_model.fit(combined, y_clean.values)
                combined_pred = combined_model.predict(combined)
                combined_mse = mean_squared_error(y_clean.values, combined_pred)
                
                # Improvement
                mse_improvement = (base_mse - combined_mse) / base_mse
                metrics['mse_improvement'] = mse_improvement
                
            except:
                metrics['mse_improvement'] = 0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error evaluating feature pair {base_name} vs {variant_name}: {e}")
            return {'error': str(e)}
    
    def _should_accept_acceleration(self, evaluation: Dict[str, Any]) -> bool:
        """Check if acceleration feature should be accepted."""
        if 'error' in evaluation:
            return False
        
        # Signal gates
        mi_ratio = evaluation.get('mi_ratio', 0)
        conditional_mi = evaluation.get('conditional_mi', 0)
        
        signal_ok = (mi_ratio >= (1 - 0.1) and  # MI within 10% of base
                    conditional_mi >= self.conditional_mi_threshold)
        
        # Uniqueness gates
        correlation = abs(evaluation.get('correlation', 1))
        uniqueness_ok = correlation <= self.correlation_threshold
        
        # Practicality gates
        mse_improvement = evaluation.get('mse_improvement', 0)
        practicality_ok = mse_improvement >= 0.01  # At least 1% improvement
        
        return signal_ok and uniqueness_ok and practicality_ok
    
    def _should_accept_dilation(self, evaluation: Dict[str, Any]) -> bool:
        """Check if dilation feature should be accepted."""
        if 'error' in evaluation:
            return False
        
        # Signal gates
        mi_ratio = evaluation.get('mi_ratio', 0)
        conditional_mi = evaluation.get('conditional_mi', 0)
        
        signal_ok = (mi_ratio >= 1.0 or  # Better than base
                    conditional_mi >= self.conditional_mi_threshold)
        
        # Uniqueness gates
        correlation = abs(evaluation.get('correlation', 1))
        uniqueness_ok = correlation <= self.correlation_threshold
        
        # Practicality gates
        mse_improvement = evaluation.get('mse_improvement', 0)
        practicality_ok = mse_improvement >= 0.01  # At least 1% improvement
        
        return signal_ok and uniqueness_ok and practicality_ok
    
    def run_complete_evaluation(self, X: pd.DataFrame, y: pd.Series,
                              base_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete acceleration and dilation evaluation.
        
        Args:
            X: Feature matrix
            y: Target variable
            base_features: List of base features to evaluate
            
        Returns:
            Complete evaluation results
        """
        logger.info("Running complete acceleration and dilation evaluation...")
        
        # Generate acceleration features
        acceleration_features = self.generate_acceleration_features(X, base_features)
        
        # Generate dilation features
        dilation_features = self.generate_dilation_features(X, base_features)
        
        # Evaluate acceleration features
        acceleration_results = self.evaluate_acceleration_features(X, y, acceleration_features, base_features or [])
        
        # Evaluate dilation features
        dilation_results = self.evaluate_dilation_features(X, y, dilation_features, base_features or [])
        
        # Compile results
        results = {
            'acceleration_features': acceleration_features,
            'dilation_features': dilation_features,
            'acceleration_evaluation': acceleration_results,
            'dilation_evaluation': dilation_results,
            'summary': {
                'total_acceleration_features': sum(len(df.columns) for df in acceleration_features.values()),
                'total_dilation_features': sum(len(df.columns) for df in dilation_features.values()),
                'accepted_acceleration': len(acceleration_results['accepted_features']),
                'accepted_dilation': len(dilation_results['accepted_features']),
                'rejected_acceleration': len(acceleration_results['rejected_features']),
                'rejected_dilation': len(dilation_results['rejected_features'])
            }
        }
        
        logger.info(f"Complete evaluation finished. Accepted: {results['summary']['accepted_acceleration']} accel, {results['summary']['accepted_dilation']} dilation")
        
        return results