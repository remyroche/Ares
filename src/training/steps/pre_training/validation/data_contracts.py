"""
Data contracts for pre-training pipeline validation.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass


class DataContractValidationError(Exception):
    """Exception raised when data contract validation fails."""
    pass


@dataclass
class FeatureArtifact:
    """Feature artifact data contract."""
    name: str
    data: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultiHorizonLabelingResult:
    """Multi-horizon labeling result data contract."""
    labels: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def validate_feature_artifact(artifact: FeatureArtifact) -> bool:
    """Validate a feature artifact."""
    if not artifact.name:
        raise DataContractValidationError("Feature artifact name cannot be empty")
    
    if artifact.data is None:
        raise DataContractValidationError("Feature artifact data cannot be None")
    
    return True


def validate_multi_horizon_labeling_result(result: MultiHorizonLabelingResult) -> bool:
    """Validate a multi-horizon labeling result."""
    if result.labels is None:
        raise DataContractValidationError("Multi-horizon labeling result labels cannot be None")
    
    return True


@dataclass
class SelectionArtifact:
    """Selection artifact data contract."""
    name: str
    data: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def validate_selection_artifact(artifact: SelectionArtifact) -> bool:
    """Validate a selection artifact."""
    if not artifact.name:
        raise DataContractValidationError("Selection artifact name cannot be empty")
    
    if artifact.data is None:
        raise DataContractValidationError("Selection artifact data cannot be None")
    
    return True
