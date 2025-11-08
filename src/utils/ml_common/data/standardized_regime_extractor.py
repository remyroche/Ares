"""
Standardized Regime Label Extractor.

This module provides a clean, standardized interface for extracting regime labels
from pipeline state artifacts, with fast-fail behavior and clear error messages.
"""

import numpy as np
from typing import Dict, Any, Optional

try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('StandardizedRegimeExtractor')
except ImportError:
    import logging
    logger = logging.getLogger('StandardizedRegimeExtractor')

try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(msg, color=None):
        logger.info(msg)


class RegimeLabelExtractionError(Exception):
    """Exception raised when regime labels cannot be extracted."""
    pass


class StandardizedRegimeExtractor:
    """
    Standardized extractor for regime labels from pipeline artifacts.
    
    This class provides a clean, hierarchical approach to extracting regime labels
    with fast-fail behavior and detailed logging.
    """
    
    def __init__(self, min_samples: int = 10, min_regimes: int = 2):
        """
        Initialize the standardized regime extractor.
        
        Args:
            min_samples: Minimum number of samples required
            min_regimes: Minimum number of unique regimes required
        """
        self.min_samples = min_samples
        self.min_regimes = min_regimes
        self.logger = logger
    
    def extract(self, pipeline_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract regime labels from pipeline state with standardized hierarchy.
        
        Extraction hierarchy:
        1. artifacts['optimal_regime_clustering_result']['labels']
        2. artifacts['regime_clustering_result']['cluster_assignments']
        3. artifacts['gmm_regime_discovery_result']['labels']
        4. artifacts['hmm_regime_discovery_result']['labels']
        5. Direct keys: 'regime_labels', 'cluster_assignments', 'labels'
        
        Args:
            pipeline_state: Pipeline state dictionary containing artifacts
            
        Returns:
            Regime labels as numpy array
            
        Raises:
            RegimeLabelExtractionError: If labels cannot be extracted or validation fails
        """
        tprint("🔍 [REGIME_EXTRACTOR] Starting standardized regime label extraction", color="cyan")
        
        # Get artifacts from pipeline state
        artifacts = pipeline_state.get('artifacts', {})
        if not artifacts:
            raise RegimeLabelExtractionError(
                "No artifacts found in pipeline state. "
                "Ensure regime discovery step has been executed."
            )
        
        tprint(f"📋 [REGIME_EXTRACTOR] Available artifacts: {list(artifacts.keys())}", color="blue")
        
        # Try extraction hierarchy
        regime_labels = self._try_extraction_hierarchy(artifacts)
        
        if regime_labels is None:
            available_keys = list(artifacts.keys())
            raise RegimeLabelExtractionError(
                f"Could not extract regime labels from any known artifact structure. "
                f"Available artifacts: {available_keys}. "
                f"Expected one of: 'optimal_regime_clustering_result', "
                f"'regime_clustering_result', 'gmm_regime_discovery_result', "
                f"'rolling_hmm_regime_discovery_result', 'hmm_regime_discovery_result', "
                f"or direct labels."
            )
        
        # Validate extracted labels
        regime_labels = self._validate_and_convert(regime_labels)
        
        tprint(f"✅ [REGIME_EXTRACTOR] Successfully extracted {len(regime_labels)} regime labels", color="green")
        tprint(f"📊 [REGIME_EXTRACTOR] Unique regimes: {np.unique(regime_labels)}", color="blue")
        
        return regime_labels
    
    def _try_extraction_hierarchy(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Try extraction hierarchy in order of preference.
        
        Args:
            artifacts: Artifacts dictionary
            
        Returns:
            Regime labels if found, None otherwise
        """
        # 1. Try optimal_regime_clustering_result (highest priority)
        labels = self._extract_from_optimal_clustering(artifacts)
        if labels is not None:
            tprint("✅ [REGIME_EXTRACTOR] Extracted from optimal_regime_clustering_result", color="green")
            return labels
        
        # 2. Try regime_clustering_result
        labels = self._extract_from_regime_clustering(artifacts)
        if labels is not None:
            tprint("✅ [REGIME_EXTRACTOR] Extracted from regime_clustering_result", color="green")
            return labels
        
        # 3. Try GMM regime discovery result
        labels = self._extract_from_gmm_discovery(artifacts)
        if labels is not None:
            tprint("✅ [REGIME_EXTRACTOR] Extracted from gmm_regime_discovery_result", color="green")
            return labels
        
        # 4. Try Rolling HMM regime discovery result (new addition)
        labels = self._extract_from_rolling_hmm_discovery(artifacts)
        if labels is not None:
            tprint("✅ [REGIME_EXTRACTOR] Extracted from rolling_hmm_regime_discovery_result", color="green")
            return labels
        
        # 5. Try HMM regime discovery result
        labels = self._extract_from_hmm_discovery(artifacts)
        if labels is not None:
            tprint("✅ [REGIME_EXTRACTOR] Extracted from hmm_regime_discovery_result", color="green")
            return labels
        
        # 6. Try direct keys
        labels = self._extract_from_direct_keys(artifacts)
        if labels is not None:
            tprint("✅ [REGIME_EXTRACTOR] Extracted from direct artifact keys", color="green")
            return labels
        
        return None
    
    def _extract_from_optimal_clustering(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract from optimal_regime_clustering_result."""
        result = artifacts.get('optimal_regime_clustering_result')
        if not result:
            return None
        
        # Try 'labels' key first (preferred)
        if 'labels' in result:
            return result['labels']
        
        # Try 'cluster_assignments'
        if 'cluster_assignments' in result:
            return result['cluster_assignments']
        
        # Try nested clustering_result
        clustering_result = result.get('clustering_result')
        if isinstance(clustering_result, dict):
            if 'labels' in clustering_result:
                return clustering_result['labels']
            if 'cluster_assignments' in clustering_result:
                return clustering_result['cluster_assignments']
        
        return None
    
    def _extract_from_regime_clustering(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract from regime_clustering_result."""
        result = artifacts.get('regime_clustering_result')
        if not result:
            return None
        
        if 'cluster_assignments' in result:
            return result['cluster_assignments']
        if 'labels' in result:
            return result['labels']
        if 'regime_assignments' in result:
            return result['regime_assignments']
        
        return None
    
    def _extract_from_gmm_discovery(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract from gmm_regime_discovery_result."""
        result = artifacts.get('gmm_regime_discovery_result')
        if not result:
            return None
        
        if 'labels' in result:
            return result['labels']
        if 'cluster_assignments' in result:
            return result['cluster_assignments']
        
        return None
    
    def _extract_from_hmm_discovery(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract from hmm_regime_discovery_result."""
        result = artifacts.get('hmm_regime_discovery_result')
        if not result:
            return None
        
        if 'labels' in result:
            return result['labels']
        if 'state_sequence' in result:
            return result['state_sequence']
        
        return None
    
    def _extract_from_rolling_hmm_discovery(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract from rolling_hmm_regime_discovery_result."""
        result = artifacts.get('rolling_hmm_regime_discovery_result')
        if not result:
            return None
        
        # Try artifacts dict first
        artifacts_dict = result.get('artifacts', {})
        if 'labels' in artifacts_dict:
            labels_df = artifacts_dict['labels']
            if hasattr(labels_df, 'regime_label'):
                return labels_df['regime_label'].values
            elif hasattr(labels_df, 'values'):
                return labels_df.values.flatten()
        
        # Try direct result keys
        if 'regime_labels' in result:
            return result['regime_labels']
        if 'labels' in result:
            return result['labels']
        if 'state_sequence' in result:
            return result['state_sequence']
        
        return None
    
    def _extract_from_direct_keys(self, artifacts: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract from direct artifact keys."""
        direct_keys = ['regime_labels', 'cluster_assignments', 'labels', 'regime_assignments']
        
        for key in direct_keys:
            if key in artifacts:
                return artifacts[key]
        
        return None
    
    def _validate_and_convert(self, regime_labels: Any) -> np.ndarray:
        """
        Validate and convert regime labels to numpy array.
        
        Args:
            regime_labels: Raw regime labels (can be list, array, or other types)
            
        Returns:
            Validated numpy array of regime labels
            
        Raises:
            RegimeLabelExtractionError: If validation fails
        """
        # Convert to numpy array if needed
        if not isinstance(regime_labels, np.ndarray):
            if isinstance(regime_labels, (list, tuple)):
                regime_labels = np.array(regime_labels)
            else:
                raise RegimeLabelExtractionError(
                    f"Regime labels must be array-like, got {type(regime_labels)}"
                )
        
        # Check minimum samples
        if len(regime_labels) < self.min_samples:
            raise RegimeLabelExtractionError(
                f"Insufficient regime labels: {len(regime_labels)} < {self.min_samples} required"
            )
        
        # Check for NaN values
        if np.any(np.isnan(regime_labels)):
            nan_count = np.sum(np.isnan(regime_labels))
            raise RegimeLabelExtractionError(
                f"Regime labels contain {nan_count} NaN values. Clean data required."
            )
        
        # Check unique regimes
        unique_regimes = np.unique(regime_labels)
        if len(unique_regimes) < self.min_regimes:
            raise RegimeLabelExtractionError(
                f"Insufficient unique regimes: {len(unique_regimes)} < {self.min_regimes} required. "
                f"Found regimes: {unique_regimes}"
            )
        
        # Ensure integer labels
        regime_labels = regime_labels.astype(int)
        
        return regime_labels


def extract_regime_labels_standardized(
    pipeline_state: Dict[str, Any],
    min_samples: int = 10,
    min_regimes: int = 2
) -> np.ndarray:
    """
    Convenience function to extract regime labels with standardized interface.
    
    Args:
        pipeline_state: Pipeline state dictionary
        min_samples: Minimum number of samples required
        min_regimes: Minimum number of unique regimes required
        
    Returns:
        Regime labels as numpy array
        
    Raises:
        RegimeLabelExtractionError: If extraction or validation fails
    """
    extractor = StandardizedRegimeExtractor(min_samples=min_samples, min_regimes=min_regimes)
    return extractor.extract(pipeline_state)

