"""
Conditional Mutual Information (CMI) Feature Selection for Meta-Labeling.

This module implements I(X;Y|P) - Conditional Mutual Information between feature X and label Y
given base model predictions P. Features with CMI ≈ 0 are redundant because the base model's
predictions already capture all information X has about the outcome.

Formula: I(X;Y|P) = H(X,P) + H(Y,P) - H(X,Y,P) - H(P)

Usage:
- Filter redundant features before model training
- Data-driven threshold: 25th percentile of CMI values
- Reduces feature dimensionality while preserving conditional information
"""

import numpy as np
import time
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from scipy.stats import entropy
import warnings
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

EPS = 1e-12

class ConditionalMutualInformationSelector:
    """
    Conditional Mutual Information (CMI) feature selector.
    
    Measures I(X;Y|P) - new information feature X provides about label Y 
    given base model predictions P.
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        threshold_percentile: float = 25.0,
        min_samples: int = 100,
        random_state: int = 42
    ):
        """
        Initialize CMI selector.
        
        Args:
            n_bins: Number of bins for discretizing continuous variables
            threshold_percentile: Percentile threshold for feature removal (default: 25%)
            min_samples: Minimum samples required for reliable estimation
            random_state: Random seed for reproducibility
        """
        self.n_bins = n_bins
        self.threshold_percentile = threshold_percentile
        self.min_samples = min_samples
        self.random_state = random_state
        self.cmi_scores_ = None
        self.threshold_ = None
        self.selected_features_ = None
        
    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize continuous data into bins using quantiles.
        
        Args:
            data: Continuous data array
            
        Returns:
            Discretized data (integer bins)
        """
        if len(data) < self.min_samples:
            warnings.warn(f"Insufficient samples ({len(data)}) for reliable discretization")
            
        # Use quantile-based binning for robustness
        try:
            bins = np.quantile(data[~np.isnan(data)], np.linspace(0, 1, self.n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
            discretized = np.digitize(data, bins) - 1
            discretized = np.clip(discretized, 0, self.n_bins - 1)
            return discretized
        except Exception as e:
            tprint_warning(f"Discretization failed: {e}. Using equal-width bins.")
            return np.clip(
                np.digitize(data, np.linspace(data.min(), data.max(), self.n_bins + 1)) - 1,
                0, self.n_bins - 1
            )
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """
        Compute Shannon entropy for discrete data.
        
        Args:
            data: Discrete data array
            
        Returns:
            Entropy value
        """
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return 0.0
            
        # Count frequencies
        unique, counts = np.unique(clean_data, return_counts=True)
        probs = counts / len(clean_data)
        
        # Compute entropy
        return entropy(probs, base=2)
    
    def _compute_joint_entropy(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute joint entropy for two discrete variables.
        
        Args:
            data1: First discrete variable
            data2: Second discrete variable
            
        Returns:
            Joint entropy value
        """
        # Remove NaN values (both must be non-NaN)
        mask = ~np.isnan(data1) & ~np.isnan(data2)
        clean_data1 = data1[mask]
        clean_data2 = data2[mask]
        
        if len(clean_data1) == 0:
            return 0.0
            
        # Create joint distribution
        joint_pairs = np.column_stack([clean_data1, clean_data2])
        unique_pairs, counts = np.unique(joint_pairs, axis=0, return_counts=True)
        probs = counts / len(clean_data1)
        
        return entropy(probs, base=2)
    
    def _compute_3way_joint_entropy(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> float:
        """
        Compute 3-way joint entropy for X, Y, P.
        
        Args:
            X, Y, P: Discrete data arrays
            
        Returns:
            Joint entropy value
        """
        # Remove NaN values (all must be non-NaN)
        mask = ~np.isnan(X) & ~np.isnan(Y) & ~np.isnan(P)
        clean_X = X[mask]
        clean_Y = Y[mask]
        clean_P = P[mask]
        
        if len(clean_X) == 0:
            return 0.0
        
        # Create triple joint distribution
        triple_data = np.column_stack([clean_X, clean_Y, clean_P])
        unique_triples, counts = np.unique(triple_data, axis=0, return_counts=True)
        probs = counts / len(triple_data)
        
        return entropy(probs, base=2)
    
    def _compute_cmi(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> float:
        """
        Compute Conditional Mutual Information I(X;Y|P).
        
        Formula: I(X;Y|P) = H(X,P) + H(Y,P) - H(X,Y,P) - H(P)
        
        Args:
            X: Feature values (discretized)
            Y: Label values (discretized)
            P: Base model predictions (discretized)
            
        Returns:
            CMI value in bits
        """
        # Compute entropies
        h_xp = self._compute_joint_entropy(X, P)
        h_yp = self._compute_joint_entropy(Y, P)
        
        # Compute 3-way joint entropy
        if len(set(X)) * len(set(Y)) * len(set(P)) < 1000:
            h_xyp = self._compute_3way_joint_entropy(X, Y, P)
        else:
            h_xyp = 0.0
        
        h_p = self._compute_entropy(P)
        
        # Compute CMI
        cmi = h_xp + h_yp - h_xyp - h_p
        
        # Ensure non-negative (numerical issues)
        return max(0.0, cmi)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: pd.Series) -> "ConditionalMutualInformationSelector":
        """
        Fit CMI selector to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (binary or continuous)
            base_predictions: Base model predictions (probabilities or scores)
            
        Returns:
            Self
        """
        start_time = time.time()
        tprint_info("🔍 Computing Conditional Mutual Information (CMI) scores...")
        tprint_info(f"📊 Input data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Data quality checks
        if len(X) < self.min_samples:
            tprint_error(f"❌ Insufficient samples: {len(X)} < {self.min_samples}")
            raise ValueError(f"Insufficient samples: {len(X)} < {self.min_samples}")
        
        if len(X) != len(y) or len(X) != len(base_predictions):
            tprint_error(f"❌ Data length mismatch: X={len(X)}, y={len(y)}, base={len(base_predictions)}")
            raise ValueError("X, y, and base_predictions must have same length")
        
        # Check for missing values
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        missing_base = base_predictions.isnull().sum()
        
        if missing_X > 0:
            tprint_warning(f"⚠️ Found {missing_X} missing values in features, will fill with 0")
        if missing_y > 0:
            tprint_warning(f"⚠️ Found {missing_y} missing values in target, will fill with 0.5")
        if missing_base > 0:
            tprint_warning(f"⚠️ Found {missing_base} missing values in base predictions, will fill with 0.5")
        
        tprint_info(f"🔧 Discretizing continuous variables into {self.n_bins} bins...")
        """
        Fit CMI selector to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (binary or continuous)
            base_predictions: Base model predictions (probabilities or scores)
            
        Returns:
            Self
        """
        tprint_info("🔍 Computing Conditional Mutual Information (CMI) scores...")
        
        if len(X) < self.min_samples:
            raise ValueError(f"Insufficient samples: {len(X)} < {self.min_samples}")
        
        if len(X) != len(y) or len(X) != len(base_predictions):
            raise ValueError("X, y, and base_predictions must have same length")
        
        # Discretize continuous variables
        discretize_start = time.time()
        y_disc = self._discretize(y.values)
        p_disc = self._discretize(base_predictions.values)
        
        discretize_time = time.time() - discretize_start
        tprint_info(f"⏱️  Discretization completed in {discretize_time:.2f}s")
        tprint_info(f"📈 Target discretization: {len(np.unique(y_disc))} unique values")
        tprint_info(f"📈 Base prediction discretization: {len(np.unique(p_disc))} unique values")
        p_disc = self._discretize(base_predictions.values)
        
        # Compute CMI for each feature
        tprint_info(f"🔄 Computing CMI for {n_features} features...")
        cmi_start = time.time()
        successful_computations = 0
        failed_computations = 0
        
        for i, col in enumerate(X.columns):
            if i % 50 == 0 or i == n_features - 1:
                progress = (i + 1) / n_features * 100
                tprint_info(f"   Progress: {i+1}/{n_features} ({progress:.1f}%) - {col}")
                
            try:
                x_disc = self._discretize(X[col].values)
                cmi = self._compute_cmi(x_disc, y_disc, p_disc)
                cmi_scores[col] = cmi
                successful_computations += 1
            except Exception as e:
                tprint_warning(f"   ❌ Failed to compute CMI for {col}: {e}")
                cmi_scores[col] = 0.0
                failed_computations += 1
        
        cmi_time = time.time() - cmi_start
        tprint_info(f"⏱️  CMI computation completed in {cmi_time:.2f}s")
        tprint_info(f"✅ Successful: {successful_computations}, Failed: {failed_computations}")
        # Convert to Series
        self.cmi_scores_ = pd.Series(cmi_scores)
        
        tprint_info(f"📊 CMI Statistics:")
        tprint_info(f"   Mean: {self.cmi_scores_.mean():.6f} bits")
        tprint_info(f"   Std:  {self.cmi_scores_.std():.6f} bits")
        tprint_info(f"   Min:  {self.cmi_scores_.min():.6f} bits")
        tprint_info(f"   Max:  {self.cmi_scores_.max():.6f} bits")
        
        # Set data-driven threshold (25th percentile)
        threshold_start = time.time()
        self.threshold_ = np.percentile(self.cmi_scores_.values, self.threshold_percentile)
        threshold_time = time.time() - threshold_start
        
        tprint_info(f"🎯 Data-driven threshold (25th percentile): {self.threshold_:.6f} bits")
        tprint_info(f"⏱️  Threshold computation: {threshold_time:.3f}s")
        
        # Select features
        selection_start = time.time()
        self.selected_features_ = self.cmi_scores_[self.cmi_scores_ > self.threshold_].index.tolist()
        selection_time = time.time() - selection_start
        
        total_time = time.time() - start_time
        tprint_success(f"✅ CMI Selection completed in {total_time:.2f}s")
        tprint_success(f"📊 Selected {len(self.selected_features_)}/{n_features} features ({len(self.selected_features_)/n_features:.1%})")
        tprint_info(f"⏱️  Feature selection: {selection_time:.3f}s")
        
        tprint_success(f"✅ CMI Selection: {len(self.selected_features_)}/{n_features} features kept")
        # Enhanced detailed reporting
        if cfg.get("detailed_feature_reporting", True):
            self._print_detailed_cmi_report(X.columns.tolist(), n_features)
        tprint_info(f"📊 Threshold (25th percentile): {self.threshold_:.6f} bits")
        tprint_info(f"📈 CMI range: {self.cmi_scores_.min():.6f} - {self.cmi_scores_.max():.6f} bits")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using selected features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed matrix with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            tprint_warning(f"Missing features: {missing_features}")
            available_features = [f for f in self.selected_features_ if f in X.columns]
        else:
            available_features = self.selected_features_
        
        return X[available_features].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, base_predictions: pd.Series) -> pd.DataFrame:
        """
        Fit selector and transform data in one step.
        
        Args:
            X: Feature matrix
            y: Target labels
            base_predictions: Base model predictions
            
        Returns:
            Transformed matrix with selected features
        """
        return self.fit(X, y, base_predictions).transform(X)
    
    def get_feature_scores(self) -> pd.Series:
        """
        Get CMI scores for all features.
        
        Returns:
            Series of CMI scores indexed by feature names
        """
        if self.cmi_scores_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.cmi_scores_.copy()
    
    def get_selected_features(self) -> List[str]:
        """
        Get list of selected feature names.
        
        Returns:
            List of selected feature names
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.selected_features_.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of selection results.
        
        Returns:
            Dictionary with selection statistics
        """
        if self.cmi_scores_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return {
            'n_original_features': len(self.cmi_scores_),
            'n_selected_features': len(self.selected_features_),
            'selection_ratio': len(self.selected_features_) / len(self.cmi_scores_),
            'threshold': self.threshold_,
            'cmi_mean': self.cmi_scores_.mean(),
            'cmi_std': self.cmi_scores_.std(),
            'cmi_max': self.cmi_scores_.max(),
            'cmi_min': self.cmi_scores_.min(),
            'threshold_percentile': self.threshold_percentile
        }


def cmi_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    base_predictions: pd.Series,
    n_bins: int = 10,
    threshold_percentile: float = 25.0,
    min_samples: int = 100,
    random_state: int = 42
) -> Tuple[pd.DataFrame, ConditionalMutualInformationSelector]:
    """
    Convenience function for CMI feature selection.
    
    Args:
        X: Feature matrix
        y: Target labels
        base_predictions: Base model predictions
        n_bins: Number of discretization bins
        threshold_percentile: Percentile threshold for removal
        min_samples: Minimum samples for reliable estimation
        random_state: Random seed
        
    Returns:
        Tuple of (selected_features_df, fitted_selector)
    """
    selector = ConditionalMutualInformationSelector(
        n_bins=n_bins,
        threshold_percentile=threshold_percentile,
        min_samples=min_samples,
        random_state=random_state
    )
    
    X_selected = selector.fit_transform(X, y, base_predictions)
    
    return X_selected, selector

    def _print_detailed_cmi_report(self, all_features: List[str], n_features: int) -> None:
        """
        Print detailed CMI feature selection report with kept vs discarded features.
        
        Args:
            all_features: List of all feature names considered
            n_features: Total number of features processed
        """
        if self.cmi_scores_ is None or self.selected_features_ is None:
            tprint_warning("⚠️ No CMI scores available for detailed reporting")
            return
        
        max_display = min(50, n_features)  # Limit output for large feature sets
        
        tprint_info("🔍 CMI Feature Selection Report:")
        tprint_info(f"📊 Threshold: {self.threshold_:.6f} bits (25th percentile)")
        tprint_info(f"📈 CMI range: {self.cmi_scores_.min():.6f} - {self.cmi_scores_.max():.6f} bits")
        
        # Separate kept and discarded features
        kept_features = self.selected_features_
        discarded_features = [f for f in all_features if f not in kept_features]
        
        tprint_info(f"✅ KEPT ({len(kept_features)} features):")
        
        # Show kept features with scores
        kept_scores = self.cmi_scores_[kept_features].sort_values(ascending=False)
        for i, (feature, score) in enumerate(kept_scores.head(max_display).items()):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "✅"
            tprint_info(f"   {status} {feature}: {score:.6f} bits (above threshold)")
        
        if len(kept_features) > max_display:
            tprint_info(f"   ... and {len(kept_features) - max_display} more kept features")
        
        tprint_info(f"❌ DISCARDED ({len(discarded_features)} features):")
        
        # Show discarded features with reasons
        discarded_scores = self.cmi_scores_[discarded_features].sort_values(ascending=True)
        for i, (feature, score) in enumerate(discarded_scores.head(max_display).items()):
            reason = f"below threshold ({self.threshold_:.6f} bits)"
            tprint_info(f"   ❌ {feature}: {score:.6f} bits ({reason})")
        
        if len(discarded_features) > max_display:
            tprint_info(f"   ... and {len(discarded_features) - max_display} more discarded features")
        
        # Summary statistics
        kept_mean = kept_scores.mean()
        discarded_mean = discarded_scores.mean()
        
        tprint_info(f"📊 Selection Summary:")
        tprint_info(f"   📈 Kept features avg CMI: {kept_mean:.6f} bits")
        tprint_info(f"   📉 Discarded features avg CMI: {discarded_mean:.6f} bits")
        tprint_info(f"   🎯 Quality improvement: {((kept_mean - discarded_mean) / discarded_mean * 100):.1f}% higher CMI in kept features")
        
        # Top performers
        top_performers = kept_scores.head(3)
        tprint_info(f"🏆 Top 3 Performers:")
        for i, (feature, score) in enumerate(top_performers.items()):
            tprint_info(f"   {i+1}. {feature}: {score:.6f} bits")
        
        # Worst performers (if any discarded)
        if len(discarded_features) > 0:
            worst_performers = discarded_scores.head(3)
            tprint_info(f"🗑️  Worst 3 Performers:")
            for i, (feature, score) in enumerate(worst_performers.items()):
                tprint_info(f"   {i+1}. {feature}: {score:.6f} bits")

def fast_cmi_proxy(X, y, base_predictions, top_percentile=0.5):
    """
    Fast CMI approximation using correlation and mutual information.
    
    Args:
        X: Feature matrix
        y: Target series  
        base_predictions: Base model predictions
        top_percentile: Percentile of features to keep
        
    Returns:
        Tuple of (filtered features, selected indices)
    """
    # Stage 1: Correlation-based pre-filtering (O(f))
    corr_scores = np.abs([np.corrcoef(X.iloc[:, i], y)[0, 1] for i in range(X.shape[1])])
    corr_scores = np.nan_to_num(corr_scores, nan=0.0)
    
    # Select top percentile by correlation
    top_corr_idx = np.argsort(corr_scores)[-int(top_percentile * X.shape[1]):]
    X_filtered = X.iloc[:, top_corr_idx]
    
    # Stage 2: Conditional correlation filtering
    conditional_scores = []
    for i in range(len(top_corr_idx)):
        feature_col = X_filtered.columns[i]
        feature_values = X_filtered[feature_col].values
        
        # Calculate conditional correlation
        residual_y = y - base_predictions
        residual_feature = feature_values - base_predictions
        
        if np.std(residual_feature) > EPS:
            conditional_corr = np.corrcoef(residual_feature, residual_y)[0, 1]
            conditional_scores.append(abs(conditional_corr))
        else:
            conditional_scores.append(0.0)
    
    # Select top features by conditional correlation
    final_top_idx = np.argsort(conditional_scores)[-int(0.4 * len(conditional_scores)):]
    selected_features = X_filtered.iloc[:, final_top_idx]
    selected_indices = top_corr_idx[final_top_idx]
    
    return selected_features, selected_indices

