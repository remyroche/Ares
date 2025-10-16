"""
Schema validation for pre-training pipeline.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class SchemaValidationException(Exception):
    """Exception raised when schema validation fails."""
    pass


@dataclass
class RawOHLCV:
    """Raw OHLCV data schema."""
    open: Any
    high: Any
    low: Any
    close: Any
    volume: Any
    timestamp: Any = None


@dataclass
class EngineeredFeatures:
    """Engineered features schema."""
    features: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def validate_raw_ohlcv(data: RawOHLCV) -> RawOHLCV:
    """Validate raw OHLCV data."""
    if data.open is None or data.high is None or data.low is None or data.close is None or data.volume is None:
        raise SchemaValidationException("Raw OHLCV data cannot have None values")
    
    return data


def validate_engineered_features(features: EngineeredFeatures) -> bool:
    """Validate engineered features."""
    if features.features is None:
        raise SchemaValidationException("Engineered features cannot be None")
    
    return True


def report_hypothesis_count(config: Any) -> Dict[str, Any]:
    """Report hypothesis count statistics."""
    try:
        # Default hypothesis count statistics
        hypothesis_stats = {
            'total_hypotheses': 0,
            'validated_hypotheses': 0,
            'rejected_hypotheses': 0,
            'pending_hypotheses': 0,
            'validation_rate': 0.0,
            'rejection_rate': 0.0
        }
        
        # If config has hypothesis tracking, update stats
        if hasattr(config, 'hypothesis_tracking'):
            tracking = config.hypothesis_tracking
            hypothesis_stats.update({
                'total_hypotheses': getattr(tracking, 'total', 0),
                'validated_hypotheses': getattr(tracking, 'validated', 0),
                'rejected_hypotheses': getattr(tracking, 'rejected', 0),
                'pending_hypotheses': getattr(tracking, 'pending', 0)
            })
            
            # Calculate rates
            total = hypothesis_stats['total_hypotheses']
            if total > 0:
                hypothesis_stats['validation_rate'] = hypothesis_stats['validated_hypotheses'] / total
                hypothesis_stats['rejection_rate'] = hypothesis_stats['rejected_hypotheses'] / total
        
        logger.info(f"Hypothesis statistics: {hypothesis_stats}")
        return hypothesis_stats
        
    except Exception as e:
        logger.warning(f"Error reporting hypothesis count: {e}")
        return {
            'total_hypotheses': 0,
            'validated_hypotheses': 0,
            'rejected_hypotheses': 0,
            'pending_hypotheses': 0,
            'validation_rate': 0.0,
            'rejection_rate': 0.0,
            'error': str(e)
        }


def enforce_feature_temporal_alignment(features: Any, timestamps: Any) -> bool:
    """Enforce temporal alignment between features and timestamps."""
    try:
        if features is None or timestamps is None:
            logger.warning("Features or timestamps are None, skipping temporal alignment")
            return False
        
        # Check if features and timestamps have the same length
        if hasattr(features, '__len__') and hasattr(timestamps, '__len__'):
            if len(features) != len(timestamps):
                logger.warning(f"Feature length ({len(features)}) != timestamp length ({len(timestamps)})")
                return False
        
        # Additional temporal alignment checks can be added here
        logger.info("Temporal alignment check passed")
        return True
        
    except Exception as e:
        logger.warning(f"Error enforcing temporal alignment: {e}")
        return False


def schema_metadata() -> Dict[str, Any]:
    """Get schema metadata information."""
    return {
        'version': '1.0.0',
        'description': 'Pre-training validation schemas',
        'features': [
            'RawOHLCV validation',
            'EngineeredFeatures validation',
            'Hypothesis tracking',
            'Temporal alignment enforcement'
        ],
        'last_updated': '2025-10-16',
        'author': 'Ares Trading System'
    }


def validate_labeled_dataset(dataset: Any) -> bool:
    """Validate labeled dataset structure and content."""
    try:
        if dataset is None:
            logger.warning("Dataset is None, validation failed")
            return False
        
        # Check if dataset has required attributes
        required_attrs = ['features', 'labels', 'timestamps']
        for attr in required_attrs:
            if not hasattr(dataset, attr):
                logger.warning(f"Dataset missing required attribute: {attr}")
                return False
        
        # Check if features and labels have the same length
        if hasattr(dataset, 'features') and hasattr(dataset, 'labels'):
            if len(dataset.features) != len(dataset.labels):
                logger.warning(f"Features length ({len(dataset.features)}) != labels length ({len(dataset.labels)})")
                return False
        
        # Check for non-empty dataset
        if len(dataset.features) == 0:
            logger.warning("Dataset is empty")
            return False
        
        logger.info("Labeled dataset validation passed")
        return True
        
    except Exception as e:
        logger.warning(f"Error validating labeled dataset: {e}")
        return False
