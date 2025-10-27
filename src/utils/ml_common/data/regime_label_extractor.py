"""
Simplified Regime Label Extractor.

This module provides a clean, hierarchical approach to extracting regime labels
from pipeline artifacts, replacing the complex fallback logic.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
import re

logger = logging.getLogger(__name__)


class RegimeLabelExtractor:
    """
    Simplified regime label extractor with clear hierarchy and fast fail.
    """
    
    def __init__(self, min_samples: int = 6, min_regimes: int = 2):
        """
        Initialize regime label extractor.
        
        Args:
            min_samples: Minimum number of samples required
            min_regimes: Minimum number of unique regimes required
        """
        self.min_samples = min_samples
        self.min_regimes = min_regimes
        
        # Define extraction hierarchy (ordered by preference)
        self.extraction_paths = [
            'optimal_regime_clustering_result.clustering_result.cluster_assignments',
            'regime_clustering_result.cluster_assignments',
            'regime_discovery_result.regime_assignments',
            'regime_discovery_result.cluster_assignments',
            'regime_assignments',
            'cluster_assignments',
            'assignments'
        ]
    
    def extract_regime_labels(self, artifacts: Dict[str, Any]) -> np.ndarray:
        """
        Extract regime labels from artifacts using hierarchical approach.
        
        Args:
            artifacts: Pipeline artifacts dictionary
            
        Returns:
            Regime labels as numpy array
            
        Raises:
            ValueError: If no valid regime labels found
        """
        logger.info("Starting regime label extraction")
        
        # Try each extraction path in order
        for path in self.extraction_paths:
            try:
                labels = self._get_nested_value(artifacts, path)
                if labels is not None:
                    labels = self._parse_labels(labels)
                    if self._validate_labels(labels):
                        logger.info(f"Successfully extracted regime labels from: {path}")
                        logger.info(f"Labels shape: {labels.shape}, unique regimes: {np.unique(labels)}")
                        return labels
            except Exception as e:
                logger.debug(f"Failed to extract from {path}: {e}")
                continue
        
        # If no valid labels found, raise error (fast fail)
        raise ValueError(
            f"No valid regime labels found in artifacts. "
            f"Tried paths: {self.extraction_paths}. "
            f"Available artifact keys: {list(artifacts.keys())}"
        )
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[Any]:
        """
        Get nested value from dictionary using dot notation.
        
        Args:
            data: Dictionary to search
            path: Dot-separated path (e.g., 'a.b.c')
            
        Returns:
            Value at path or None if not found
        """
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _parse_labels(self, labels: Any) -> Optional[np.ndarray]:
        """
        Parse labels from various formats to numpy array.
        
        Args:
            labels: Labels in various formats
            
        Returns:
            Parsed labels as numpy array or None if parsing fails
        """
        if labels is None:
            return None
        
        # If already numpy array, return as is
        if isinstance(labels, np.ndarray):
            return labels.astype(int)
        
        # If list, convert to numpy array
        if isinstance(labels, list):
            return np.array(labels, dtype=int)
        
        # If string, try to parse
        if isinstance(labels, str):
            return self._parse_string_labels(labels)
        
        # If other type, try to convert
        try:
            return np.array(labels, dtype=int)
        except (ValueError, TypeError):
            return None
    
    def _parse_string_labels(self, labels_str: str) -> Optional[np.ndarray]:
        """
        Parse labels from string representation.
        
        Args:
            labels_str: String representation of labels
            
        Returns:
            Parsed labels as numpy array or None if parsing fails
        """
        try:
            # Remove brackets and whitespace
            clean_str = labels_str.strip('[]').strip()
            
            # Handle ellipsis case (e.g., "[2 2 2 ... 4 6 6]")
            if '...' in clean_str:
                logger.warning("Found ellipsis in regime labels string - this may indicate data truncation")
                # Extract numbers using regex
                numbers = re.findall(r'\d+', clean_str)
                if numbers:
                    return np.array([int(x) for x in numbers], dtype=int)
                else:
                    return None
            
            # Split by spaces and convert to integers
            if clean_str:
                values = [int(x) for x in clean_str.split() if x.strip()]
                return np.array(values, dtype=int)
            else:
                return None
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse string labels: {e}")
            return None
    
    def _validate_labels(self, labels: Optional[np.ndarray]) -> bool:
        """
        Validate regime labels.
        
        Args:
            labels: Labels to validate
            
        Returns:
            True if labels are valid
        """
        if labels is None:
            return False
        
        # Check minimum samples
        if len(labels) < self.min_samples:
            logger.warning(f"Insufficient samples: {len(labels)} < {self.min_samples}")
            return False
        
        # Check minimum regimes
        unique_regimes = np.unique(labels)
        if len(unique_regimes) < self.min_regimes:
            logger.warning(f"Insufficient regimes: {len(unique_regimes)} < {self.min_regimes}")
            return False
        
        # Check for valid regime values (non-negative integers)
        if not np.all(labels >= 0):
            logger.warning("Found negative regime labels")
            return False
        
        if not np.all(labels == labels.astype(int)):
            logger.warning("Found non-integer regime labels")
            return False
        
        # Check for reasonable regime distribution
        regime_counts = np.bincount(labels)
        min_regime_count = np.min(regime_counts[regime_counts > 0])
        if min_regime_count < 2:
            logger.warning(f"Some regimes have very few samples (min: {min_regime_count})")
        
        logger.info(f"Regime distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        return True
    
    def create_synthetic_labels(self, n_samples: int, n_regimes: int = 3) -> np.ndarray:
        """
        Create synthetic regime labels for testing.
        
        Args:
            n_samples: Number of samples
            n_regimes: Number of regimes
            
        Returns:
            Synthetic regime labels
        """
        logger.info(f"Creating synthetic regime labels: {n_samples} samples, {n_regimes} regimes")
        
        # Create regime labels with some temporal structure
        np.random.seed(42)
        
        # Create regime changes at random intervals
        change_points = np.sort(np.random.choice(n_samples, size=n_regimes-1, replace=False))
        change_points = np.concatenate([[0], change_points, [n_samples]])
        
        labels = np.zeros(n_samples, dtype=int)
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            labels[start_idx:end_idx] = i
        
        logger.info(f"Synthetic labels created: {np.unique(labels, return_counts=True)}")
        return labels


def extract_regime_labels_fast_fail(artifacts: Dict[str, Any], 
                                  min_samples: int = 6, 
                                  min_regimes: int = 2) -> np.ndarray:
    """
    Fast-fail regime label extraction function.
    
    Args:
        artifacts: Pipeline artifacts dictionary
        min_samples: Minimum number of samples required
        min_regimes: Minimum number of unique regimes required
        
    Returns:
        Regime labels as numpy array
        
    Raises:
        ValueError: If no valid regime labels found
    """
    extractor = RegimeLabelExtractor(min_samples, min_regimes)
    return extractor.extract_regime_labels(artifacts)