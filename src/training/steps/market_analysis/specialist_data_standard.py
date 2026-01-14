"""
Specialist Data Standard - Configuration and Validation

This module defines the standard data structures, validation rules,
and configuration for all specialist models to ensure consistency
and ensemble compatibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SpecialistType(Enum):
    """Enumeration of specialist types."""
    VOLUME_FORCE = "volume_force"
    MOMENTUM_PERSISTENCE = "momentum_persistence"
    VOLATILITY_BURST = "volatility_burst"
    BREAKOUT_BOUNCE = "breakout_bounce"
    LIQUIDITY_REGIME = "liquidity_regime"
    SMC_REGIME = "smc_regime"
    PATH_REGIME = "path_regime"
    REVERSION_REGIME = "reversion_regime"
    RISK_REGIME = "risk_regime"
    MACRO_REGIME = "macro_regime"
    MESO_REGIME = "meso_regime"
    SPECTRAL = "spectral"
    MICROSTRUCTURE = "microstructure"
    CANDLESTICK = "candlestick"


class RequirementStatus(Enum):
    """Enumeration of requirement compliance status."""
    COMPLIANT = "COMPLIANT"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    NON_COMPLIANT = "NON_COMPLIANT"


@dataclass
class SpecialistRequirements:
    """Requirements definition for specialist models."""
    min_mi_score: float = 0.02
    min_hsic_score: float = 0.1
    min_auc_score: float = 0.55
    min_accuracy_score: float = 0.51
    min_features: int = 5
    max_features: int = 100
    min_samples: int = 1000
    max_high_correlation_pairs: int = 3
    target_ensemble_correlation: float = 0.3
    binary_output_required: bool = True
    max_correlation_threshold: float = 0.7


@dataclass
class SpecialistMetrics:
    """Metrics for specialist model evaluation."""
    mi_score: float = 0.0
    hsic_score: float = 0.0
    auc_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    orthogonal_features: int = 0
    high_correlation_pairs: int = 0
    total_features: int = 0
    binary_output: bool = False
    requirements_met: int = 0
    compliance_status: RequirementStatus = RequirementStatus.NON_COMPLIANT


@dataclass
class SpecialistArtifact:
    """Standard structure for specialist artifacts."""
    specialist_name: str
    specialist_type: SpecialistType
    data: pd.DataFrame
    metadata: Dict[str, Any]
    metrics: SpecialistMetrics
    requirements: SpecialistRequirements = field(default_factory=SpecialistRequirements)
    
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """Validate artifact structure compliance."""
        issues = []
        
        # Check required columns
        required_columns = ['specialist_prediction', 'specialist_probability', 'target_label']
        for col in required_columns:
            if col not in self.data.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check data quality
        if len(self.data) < self.requirements.min_samples:
            issues.append(f"Insufficient samples: {len(self.data)} < {self.requirements.min_samples}")
        
        # Check binary output
        if self.requirements.binary_output_required and not self.metrics.binary_output:
            issues.append("Binary output requirement not met")
        
        # Check MI score
        if self.metrics.mi_score < self.requirements.min_mi_score:
            issues.append(f"MI score too low: {self.metrics.mi_score} < {self.requirements.min_mi_score}")
        
        # Check correlation
        if self.metrics.high_correlation_pairs > self.requirements.max_high_correlation_pairs:
            issues.append(f"Too many high correlation pairs: {self.metrics.high_correlation_pairs}")
        
        return len(issues) == 0, issues
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary for the specialist."""
        is_valid, issues = self.validate_structure()
        
        return {
            'specialist_name': self.specialist_name,
            'specialist_type': self.specialist_type.value,
            'is_compliant': is_valid,
            'compliance_status': self.metrics.compliance_status.value,
            'requirements_met': self.metrics.requirements_met,
            'total_requirements': 3,
            'issues': issues,
            'metrics': {
                'mi_score': self.metrics.mi_score,
                'hsic_score': self.metrics.hsic_score,
                'auc_score': self.metrics.auc_score,
                'accuracy_score': self.metrics.accuracy_score,
                'orthogonal_features': self.metrics.orthogonal_features,
                'high_correlation_pairs': self.metrics.high_correlation_pairs,
                'total_features': self.metrics.total_features,
                'binary_output': self.metrics.binary_output
            }
        }


class SpecialistDataValidator:
    """Validator for specialist data structures."""
    
    def __init__(self, requirements: SpecialistRequirements = None):
        self.requirements = requirements or SpecialistRequirements()
    
    def validate_prediction_data(self, df: pd.DataFrame, specialist_name: str) -> Tuple[bool, List[str]]:
        """
        Validate prediction data structure.
        
        Args:
            df: DataFrame to validate
            specialist_name: Name of the specialist
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check DataFrame structure
        if not isinstance(df, pd.DataFrame):
            issues.append("Data is not a pandas DataFrame")
            return False, issues
        
        # Check if this is a feature-only specialist (no predictions/probabilities)
        has_predictions = any(col in df.columns for col in ['specialist_prediction', 'specialist_probability'])
        
        # Check required columns
        if has_predictions:
            # For specialists with predictions, target_label is required
            required_columns = ['target_label']
            optional_columns = ['specialist_prediction', 'specialist_probability']
        else:
            # For feature-only specialists, target_label is optional
            required_columns = []
            optional_columns = ['target_label', 'specialist_prediction', 'specialist_probability']
        
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check optional columns and warn if missing
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            if not has_predictions:
                # Feature-only specialists don't need target/prediction columns - this is normal
                logger.debug(f"Feature-only specialist {specialist_name} missing optional columns: {missing_optional}")
            else:
                issues.append(f"Missing optional columns: {missing_optional}")
        
        # Check data quality
        if len(df) < self.requirements.min_samples:
            issues.append(f"Insufficient samples: {len(df)} < {self.requirements.min_samples}")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        for col, count in nan_counts.items():
            if count > len(df) * 0.1:  # More than 10% NaN
                issues.append(f"Too many NaN values in {col}: {count} ({count/len(df):.1%})")
        
        # Check binary output
        if 'specialist_prediction' in df.columns:
            unique_vals = df['specialist_prediction'].nunique()
            if unique_vals > 10:
                issues.append(f"Prediction not binary: {unique_vals} unique values")
        
        # Check target validity
        if 'target_label' in df.columns:
            unique_targets = df['target_label'].nunique()
            if unique_targets > 2:
                issues.append(f"Target not binary: {unique_targets} unique values")
            
            # Check target balance
            pos_rate = df['target_label'].mean()
            if pos_rate < 0.01 or pos_rate > 0.99:
                issues.append(f"Target imbalance: {pos_rate:.3f}")
        
        return len(issues) == 0, issues
    
    def validate_metadata(self, metadata: Dict[str, Any], specialist_name: str) -> Tuple[bool, List[str]]:
        """
        Validate metadata structure.
        
        Args:
            metadata: Metadata dictionary
            specialist_name: Name of the specialist
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Required metadata fields
        required_fields = ['specialist_name', 'symbol', 'exchange', 'timeframe', 'timestamp']
        for field in required_fields:
            if field not in metadata:
                issues.append(f"Missing required metadata field: {field}")
        
        # Check data types
        if 'mi_score' in metadata and not isinstance(metadata['mi_score'], (int, float)):
            issues.append("mi_score must be numeric")
        
        if 'n_samples' in metadata and not isinstance(metadata['n_samples'], int):
            issues.append("n_samples must be integer")
        
        return len(issues) == 0, issues


class SpecialistStandardFactory:
    """Factory for creating standardized specialist artifacts."""
    
    def __init__(self, requirements: SpecialistRequirements = None):
        self.requirements = requirements or SpecialistRequirements()
        self.validator = SpecialistDataValidator(requirements)
    
    def create_standard_artifact(self, specialist_name: str, specialist_type: SpecialistType,
                                data: pd.DataFrame, metadata: Dict[str, Any],
                                metrics: Dict[str, Any]) -> SpecialistArtifact:
        """
        Create a standardized specialist artifact.
        
        Args:
            specialist_name: Name of the specialist
            specialist_type: Type of specialist
            data: Prediction data
            metadata: Artifact metadata
            metrics: Performance metrics
            
        Returns:
            Standardized specialist artifact
        """
        # Validate data and metadata
        data_valid, data_issues = self.validator.validate_prediction_data(data, specialist_name)
        metadata_valid, metadata_issues = self.validator.validate_metadata(metadata, specialist_name)
        
        if not data_valid:
            logger.warning(f"Data validation issues for {specialist_name}: {data_issues}")
        
        if not metadata_valid:
            logger.warning(f"Metadata validation issues for {specialist_name}: {metadata_issues}")
        
        # Create metrics object
        specialist_metrics = SpecialistMetrics(
            mi_score=metrics.get('mi_score', 0.0),
            hsic_score=metrics.get('hsic_score', 0.0),
            auc_score=metrics.get('auc_score'),
            accuracy_score=metrics.get('accuracy_score'),
            orthogonal_features=metrics.get('orthogonal_features', 0),
            high_correlation_pairs=metrics.get('high_correlation_pairs', 0),
            total_features=metrics.get('total_features', 0),
            binary_output=metrics.get('binary_output', False),
            requirements_met=metrics.get('requirements_met', 0)
        )
        
        # Determine compliance status
        specialist_metrics.compliance_status = self._determine_compliance_status(specialist_metrics)
        
        # Create artifact
        artifact = SpecialistArtifact(
            specialist_name=specialist_name,
            specialist_type=specialist_type,
            data=data,
            metadata=metadata,
            metrics=specialist_metrics,
            requirements=self.requirements
        )
        
        return artifact
    
    def _determine_compliance_status(self, metrics: SpecialistMetrics) -> RequirementStatus:
        """Determine compliance status based on metrics."""
        requirements_met = 0
        
        # Check MI requirement
        if metrics.mi_score >= self.requirements.min_mi_score:
            requirements_met += 1
        
        # Check orthogonality requirement
        if metrics.high_correlation_pairs <= self.requirements.max_high_correlation_pairs:
            requirements_met += 1
        
        # Check binary output requirement
        if not self.requirements.binary_output_required or metrics.binary_output:
            requirements_met += 1
        
        metrics.requirements_met = requirements_met
        
        if requirements_met == 3:
            return RequirementStatus.COMPLIANT
        elif requirements_met >= 2:
            return RequirementStatus.NEEDS_IMPROVEMENT
        else:
            return RequirementStatus.NON_COMPLIANT


# Standard configurations for different specialist types
SPECIALIST_CONFIGURATIONS = {
    SpecialistType.VOLUME_FORCE: {
        'target_columns': ['vol_force_breakout', 'vol_force_volatility', 'vol_force_trend'],
        'feature_columns': ['volume', 'high', 'low', 'close'],
        'mi_target': 0.02,
        'correlation_threshold': 0.7
    },
    SpecialistType.MOMENTUM_PERSISTENCE: {
        'target_columns': ['momentum_persistence_label'],
        'feature_columns': ['close', 'volume', 'high', 'low'],
        'mi_target': 0.02,
        'correlation_threshold': 0.7
    },
    SpecialistType.SMC_REGIME: {
        'target_columns': ['smc_target'],
        'feature_columns': ['close', 'volume', 'high', 'low'],
        'mi_target': 0.02,
        'correlation_threshold': 0.7
    }
}


def get_specialist_configuration(specialist_type: SpecialistType) -> Dict[str, Any]:
    """Get standard configuration for a specialist type."""
    return SPECIALIST_CONFIGURATIONS.get(specialist_type, {
        'target_columns': ['target_label'],
        'feature_columns': ['close', 'volume', 'high', 'low'],
        'mi_target': 0.02,
        'correlation_threshold': 0.7
    })
