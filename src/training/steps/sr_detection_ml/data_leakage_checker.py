"""
Data Leakage Checker for SR ML System

Identifies and prevents data leakage issues that cause unrealistic performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataLeakageChecker:
    """
    Check for data leakage issues in SR ML training.
    
    Common leakage sources:
    1. Future information in features
    2. Target variable leaking into features
    3. Train/test contamination
    4. Temporal ordering violations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_for_leakage(
        self,
        raw_data: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        best_target: str
    ) -> Dict[str, any]:
        """
        Comprehensive data leakage check.
        
        Args:
            raw_data: Raw training data
            feature_cols: List of feature column names
            target_cols: List of target column names
            best_target: Selected target variable
        
        Returns:
            Dictionary with leakage check results
        """
        self.logger.info("🔍 Running comprehensive data leakage checks...")
        
        issues = []
        warnings = []
        
        # Check 1: Target in features
        target_leakage = self._check_target_in_features(
            raw_data, feature_cols, best_target
        )
        if target_leakage['has_leakage']:
            issues.append(target_leakage['message'])
        
        # Check 2: Future information
        future_leakage = self._check_future_information(
            raw_data, feature_cols, target_cols
        )
        if future_leakage['warnings']:
            warnings.extend(future_leakage['warnings'])
        
        # Check 3: Perfect correlations (suspicious)
        perfect_corr = self._check_perfect_correlations(
            raw_data[feature_cols + [best_target]]
        )
        if perfect_corr['has_perfect']:
            issues.append(perfect_corr['message'])
        
        # Check 4: Dataset size adequacy
        size_check = self._check_dataset_size(
            raw_data, len(feature_cols)
        )
        if size_check['issues']:
            warnings.extend(size_check['issues'])
        
        # Check 5: Temporal ordering
        temporal_check = self._check_temporal_ordering(raw_data)
        if not temporal_check['is_valid']:
            issues.append(temporal_check['message'])
        
        return {
            'has_critical_issues': len(issues) > 0,
            'critical_issues': issues,
            'warnings': warnings,
            'checks_performed': 5
        }
    
    def _check_target_in_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target: str
    ) -> Dict:
        """Check if target variable is directly in features."""
        # Check for exact target column in features
        if target in feature_cols:
            return {
                'has_leakage': True,
                'message': f"🚨 CRITICAL: Target '{target}' found in feature columns!"
            }
        
        # Check for high correlation (>0.99) between target and any feature
        target_vals = data[target].values
        
        for feat in feature_cols:
            feat_vals = data[feat].values
            
            # Skip if either has all NaN
            if np.all(np.isnan(feat_vals)) or np.all(np.isnan(target_vals)):
                continue
            
            # Calculate correlation
            valid_mask = ~(np.isnan(feat_vals) | np.isnan(target_vals))
            if valid_mask.sum() > 10:
                corr = np.corrcoef(
                    feat_vals[valid_mask],
                    target_vals[valid_mask]
                )[0, 1]
                
                if abs(corr) > 0.99:
                    return {
                        'has_leakage': True,
                        'message': f"🚨 CRITICAL: Feature '{feat}' has correlation {corr:.4f} with target (>0.99)!"
                    }
        
        return {'has_leakage': False, 'message': None}
    
    def _check_future_information(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str]
    ) -> Dict:
        """Check for features that use future information."""
        warnings = []
        
        # Check if any features have keywords suggesting future data
        future_keywords = ['future', 'forward', 'ahead', 'next']
        
        for feat in feature_cols:
            if any(kw in feat.lower() for kw in future_keywords):
                warnings.append(
                    f"⚠️ Feature '{feat}' may contain future information (keyword match)"
                )
        
        return {'warnings': warnings}
    
    def _check_perfect_correlations(self, data: pd.DataFrame) -> Dict:
        """Check for perfect correlations (r=1.0) suggesting leakage."""
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Find perfect correlations (excluding diagonal)
        np.fill_diagonal(corr_matrix.values, 0)
        
        perfect_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= 0.999:
                    perfect_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if perfect_pairs:
            pairs_str = ', '.join([f"({p[0]}, {p[1]})" for p in perfect_pairs[:3]])
            return {
                'has_perfect': True,
                'message': f"🚨 CRITICAL: {len(perfect_pairs)} perfect correlations found: {pairs_str}"
            }
        
        return {'has_perfect': False, 'message': None}
    
    def _check_dataset_size(
        self,
        data: pd.DataFrame,
        n_features: int
    ) -> Dict:
        """Check if dataset size is adequate."""
        issues = []
        
        n_samples = len(data)
        ratio = n_samples / n_features if n_features > 0 else 0
        
        if n_samples < 1000:
            issues.append(
                f"⚠️ Small dataset: {n_samples} samples (recommended: >1000 for stability)"
            )
        
        if ratio < 10:
            issues.append(
                f"⚠️ Low sample-to-feature ratio: {ratio:.1f}:1 (recommended: >10:1)"
            )
        
        return {'issues': issues}
    
    def _check_temporal_ordering(self, data: pd.DataFrame) -> Dict:
        """Check temporal ordering of data."""
        if 'date' not in data.columns:
            return {'is_valid': True, 'message': None}
        
        dates = data['date']
        
        # Check if dates are sorted
        if not dates.is_monotonic_increasing:
            return {
                'is_valid': False,
                'message': "🚨 CRITICAL: Data is not temporally ordered!"
            }
        
        return {'is_valid': True, 'message': None}

