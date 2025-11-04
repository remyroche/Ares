"""
Multicollinearity Remover for SR ML System

Detects and removes perfectly or highly correlated features to prevent
model instability and overfitting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Set

logger = logging.getLogger(__name__)


class MulticollinearityRemover:
    """
    Detect and remove multicollinear features from dataset.
    
    Handles:
    1. Perfect correlations (r >= 0.999)
    2. High correlations (r >= 0.95) 
    3. Provides reporting on removed features
    """
    
    def __init__(self, perfect_threshold: float = 0.999, high_threshold: float = 0.95):
        """
        Initialize multicollinearity remover.
        
        Args:
            perfect_threshold: Correlation threshold for perfect multicollinearity (default: 0.999)
            high_threshold: Correlation threshold for high multicollinearity (default: 0.95)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.perfect_threshold = perfect_threshold
        self.high_threshold = high_threshold
        
        self.removed_features = []
        self.correlation_pairs = []
    
    def detect_and_remove(
        self,
        X: pd.DataFrame,
        remove_perfect_only: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and remove multicollinear features.
        
        Args:
            X: Feature DataFrame
            remove_perfect_only: If True, only removes perfect correlations (>= 0.999)
                                If False, also removes high correlations (>= 0.95)
        
        Returns:
            Tuple of (cleaned_DataFrame, report_dict)
        """
        self.logger.info("🔍 Detecting multicollinearity...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find correlated pairs
        perfect_pairs = self._find_correlated_pairs(corr_matrix, self.perfect_threshold)
        
        if not remove_perfect_only:
            high_pairs = self._find_correlated_pairs(corr_matrix, self.high_threshold)
            # Remove perfect pairs from high pairs (avoid duplicates)
            high_pairs = [p for p in high_pairs if p not in perfect_pairs]
        else:
            high_pairs = []
        
        total_pairs = len(perfect_pairs) + len(high_pairs)
        
        if total_pairs == 0:
            self.logger.info("✅ No significant multicollinearity detected")
            return X, {
                'removed_count': 0,
                'perfect_correlations': 0,
                'high_correlations': 0,
                'removed_features': []
            }
        
        self.logger.warning(f"⚠️ Found {len(perfect_pairs)} perfect correlations (>= {self.perfect_threshold})")
        if high_pairs:
            self.logger.warning(f"⚠️ Found {len(high_pairs)} high correlations (>= {self.high_threshold})")
        
        # Determine which features to remove
        features_to_remove = self._select_features_to_remove(
            X, perfect_pairs + high_pairs, corr_matrix
        )
        
        # Log some examples
        if perfect_pairs:
            self.logger.info(f"   Examples of perfect correlations:")
            for i, (f1, f2, corr) in enumerate(perfect_pairs[:3], 1):
                self.logger.info(f"      {i}. {f1} ↔ {f2} (r={corr:.4f})")
        
        if high_pairs and not remove_perfect_only:
            self.logger.info(f"   Examples of high correlations:")
            for i, (f1, f2, corr) in enumerate(high_pairs[:3], 1):
                self.logger.info(f"      {i}. {f1} ↔ {f2} (r={corr:.4f})")
        
        # Remove features
        X_cleaned = X.drop(columns=features_to_remove)
        
        self.logger.info(f"🧹 Removed {len(features_to_remove)} features due to multicollinearity")
        self.logger.info(f"   Remaining features: {len(X_cleaned.columns)}")
        
        # Store results
        self.removed_features = features_to_remove
        self.correlation_pairs = perfect_pairs + high_pairs
        
        report = {
            'removed_count': len(features_to_remove),
            'perfect_correlations': len(perfect_pairs),
            'high_correlations': len(high_pairs),
            'removed_features': features_to_remove,
            'correlation_pairs': self.correlation_pairs[:10]  # Top 10 for reporting
        }
        
        return X_cleaned, report
    
    def _find_correlated_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features with correlation above threshold.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold
        
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        pairs = []
        
        # Iterate through upper triangle only (avoid duplicates and diagonal)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if corr_val >= threshold:
                    pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        # Sort by correlation (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def _select_features_to_remove(
        self,
        X: pd.DataFrame,
        corr_pairs: List[Tuple[str, str, float]],
        corr_matrix: pd.DataFrame
    ) -> List[str]:
        """
        Select which features to remove from correlated pairs.
        
        Strategy:
        1. For each correlated pair, keep the feature with higher average correlation to target
        2. If tie, keep the feature that appears in fewer correlation pairs
        3. If still tie, keep the first one alphabetically (consistency)
        
        Args:
            X: Feature DataFrame
            corr_pairs: List of correlated pairs
            corr_matrix: Full correlation matrix
        
        Returns:
            List of feature names to remove
        """
        features_to_remove = set()
        pair_counts = {}  # Count how many times each feature appears in pairs
        
        # Count pair appearances
        for f1, f2, _ in corr_pairs:
            pair_counts[f1] = pair_counts.get(f1, 0) + 1
            pair_counts[f2] = pair_counts.get(f2, 0) + 1
        
        # For each pair, decide which to remove
        for f1, f2, corr_val in corr_pairs:
            # Skip if either already marked for removal
            if f1 in features_to_remove or f2 in features_to_remove:
                continue
            
            # Calculate average absolute correlation with all other features
            f1_avg_corr = corr_matrix[f1].abs().mean()
            f2_avg_corr = corr_matrix[f2].abs().mean()
            
            # Remove the feature with higher average correlation (more redundant)
            if f1_avg_corr > f2_avg_corr:
                features_to_remove.add(f1)
            elif f2_avg_corr > f1_avg_corr:
                features_to_remove.add(f2)
            else:
                # Tie: remove feature that appears in more pairs
                if pair_counts[f1] > pair_counts[f2]:
                    features_to_remove.add(f1)
                elif pair_counts[f2] > pair_counts[f1]:
                    features_to_remove.add(f2)
                else:
                    # Still tie: alphabetical (consistency)
                    features_to_remove.add(f1 if f1 > f2 else f2)
        
        return list(features_to_remove)
    
    def get_removal_report(self) -> str:
        """
        Get formatted report of removed features.
        
        Returns:
            Markdown-formatted report string
        """
        if not self.removed_features:
            return "No features removed due to multicollinearity."
        
        report = f"## Multicollinearity Removal Report\n\n"
        report += f"**Total features removed:** {len(self.removed_features)}\n\n"
        report += f"**Correlation pairs found:** {len(self.correlation_pairs)}\n\n"
        
        report += f"### Top 10 Correlated Pairs:\n\n"
        report += "| Feature 1 | Feature 2 | Correlation |\n"
        report += "|-----------|-----------|-------------|\n"
        
        for f1, f2, corr in self.correlation_pairs[:10]:
            removed_marker = ""
            if f1 in self.removed_features:
                removed_marker = " ❌"
            elif f2 in self.removed_features:
                removed_marker = " ❌"
            
            report += f"| {f1} | {f2} | {corr:.4f}{removed_marker} |\n"
        
        report += f"\n### Removed Features:\n\n"
        for feat in self.removed_features[:20]:  # Show top 20
            report += f"- {feat}\n"
        
        if len(self.removed_features) > 20:
            report += f"\n... and {len(self.removed_features) - 20} more\n"
        
        return report

