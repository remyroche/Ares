"""
Feature Validation Module

This module provides utilities to validate that feature sets match
between training and inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum


@dataclass
class FeatureValidationResult:
    """Result of feature validation."""
    is_valid: bool
    missing_features: List[str]
    extra_features: List[str]
    mismatched_types: Dict[str, Tuple[str, str]]  # feature: (expected_type, actual_type)
    warnings: List[str]
    errors: List[str]


class FeatureValidator:
    """
    Validator for ensuring feature sets match between training and inference.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.expected_features: Optional[Set[str]] = None
        self.expected_feature_types: Optional[Dict[str, str]] = None
    
    def set_expected_features(
        self,
        features: List[str],
        feature_types: Optional[Dict[str, str]] = None
    ):
        """
        Set expected features from training.
        
        Args:
            features: List of expected feature names
            feature_types: Optional dictionary mapping feature names to types
        """
        self.expected_features = set(features)
        self.expected_feature_types = feature_types or {}
        self.logger.info(f"Set expected features: {len(self.expected_features)} features")
    
    def validate_features(
        self,
        data: pd.DataFrame,
        strict: bool = False
    ) -> FeatureValidationResult:
        """
        Validate that provided features match expected features.
        
        Args:
            data: DataFrame with features to validate
            strict: If True, fail on missing features. If False, just warn.
            
        Returns:
            FeatureValidationResult with validation details
        """
        if self.expected_features is None:
            return FeatureValidationResult(
                is_valid=False,
                missing_features=[],
                extra_features=[],
                mismatched_types={},
                warnings=[],
                errors=["Expected features not set. Call set_expected_features() first."]
            )
        
        actual_features = set(data.columns)
        missing_features = list(self.expected_features - actual_features)
        extra_features = list(actual_features - self.expected_features)
        
        mismatched_types = {}
        warnings = []
        errors = []
        
        # Check feature types if provided
        if self.expected_feature_types:
            for feature in self.expected_features & actual_features:
                if feature in self.expected_feature_types:
                    expected_type = self.expected_feature_types[feature]
                    actual_type = str(data[feature].dtype)
                    
                    # Check if types match (allowing some flexibility)
                    if not self._types_match(expected_type, actual_type):
                        mismatched_types[feature] = (expected_type, actual_type)
                        warnings.append(
                            f"Feature '{feature}' type mismatch: expected {expected_type}, got {actual_type}"
                        )
        
        # Determine validity
        is_valid = True
        
        if missing_features:
            error_msg = f"Missing {len(missing_features)} expected features: {missing_features[:5]}..."
            if strict:
                errors.append(error_msg)
                is_valid = False
            else:
                warnings.append(error_msg)
        
        if extra_features:
            warnings.append(f"Found {len(extra_features)} extra features: {extra_features[:5]}...")
        
        if mismatched_types:
            if strict:
                errors.append(f"Type mismatches found for {len(mismatched_types)} features")
                is_valid = False
        
        if errors:
            for error in errors:
                self.logger.error(error)
        
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
        
        if is_valid:
            self.logger.info(f"Feature validation passed: {len(actual_features)} features validated")
        
        return FeatureValidationResult(
            is_valid=is_valid,
            missing_features=missing_features,
            extra_features=extra_features,
            mismatched_types=mismatched_types,
            warnings=warnings,
            errors=errors
        )
    
    def _types_match(self, expected: str, actual: str) -> bool:
        """Check if types match (allowing some flexibility)."""
        # Normalize types
        expected_norm = expected.lower()
        actual_norm = actual.lower()
        
        # Exact match
        if expected_norm == actual_norm:
            return True
        
        # Numeric type groups
        numeric_types = {'int', 'int64', 'int32', 'int16', 'int8', 'float', 'float64', 'float32', 'float16'}
        if expected_norm in numeric_types and actual_norm in numeric_types:
            return True
        
        # Object types
        if expected_norm in {'object', 'str', 'string'} and actual_norm in {'object', 'str', 'string'}:
            return True
        
        # Bool types
        if expected_norm in {'bool', 'boolean'} and actual_norm in {'bool', 'boolean', 'int8'}:
            return True
        
        return False
    
    def compare_feature_sets(
        self,
        training_features: List[str],
        inference_features: List[str]
    ) -> FeatureValidationResult:
        """
        Compare two feature sets directly.
        
        Args:
            training_features: Features used in training
            inference_features: Features used in inference
            
        Returns:
            FeatureValidationResult with comparison details
        """
        training_set = set(training_features)
        inference_set = set(inference_features)
        
        missing_features = list(training_set - inference_set)
        extra_features = list(inference_set - training_set)
        
        is_valid = len(missing_features) == 0
        
        warnings = []
        errors = []
        
        if missing_features:
            error_msg = f"Missing {len(missing_features)} training features in inference: {missing_features[:5]}..."
            errors.append(error_msg)
        
        if extra_features:
            warnings.append(f"Found {len(extra_features)} extra features in inference: {extra_features[:5]}...")
        
        if is_valid:
            self.logger.info(f"Feature sets match: {len(inference_set)} features")
        else:
            self.logger.error(f"Feature sets mismatch: {len(missing_features)} missing, {len(extra_features)} extra")
        
        return FeatureValidationResult(
            is_valid=is_valid,
            missing_features=missing_features,
            extra_features=extra_features,
            mismatched_types={},
            warnings=warnings,
            errors=errors
        )


def validate_feature_set(
    expected_features: List[str],
    actual_features: List[str],
    strict: bool = False,
    logger: Optional[logging.Logger] = None
) -> FeatureValidationResult:
    """
    Convenience function to validate feature sets.
    
    Args:
        expected_features: Expected feature names
        actual_features: Actual feature names
        strict: If True, fail on mismatches
        logger: Optional logger
        
    Returns:
        FeatureValidationResult
    """
    validator = FeatureValidator(logger=logger)
    return validator.compare_feature_sets(expected_features, actual_features)


def compare_feature_sets(
    training_features: List[str],
    inference_features: List[str],
    logger: Optional[logging.Logger] = None
) -> FeatureValidationResult:
    """
    Convenience function to compare feature sets.
    
    Args:
        training_features: Training feature names
        inference_features: Inference feature names
        logger: Optional logger
        
    Returns:
        FeatureValidationResult
    """
    validator = FeatureValidator(logger=logger)
    return validator.compare_feature_sets(training_features, inference_features)
