"""
Vector Integrity Validation

This module implements vector integrity validation to ensure semantic consistency
across feature families, including timestamp alignment, asset ID consistency,
and scaling/normalization validation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class VectorIntegrityConfig:
    """Configuration for vector integrity validation."""
    
    # Timestamp validation
    max_timestamp_gap_minutes: int = 5        # Maximum gap between timestamps
    require_continuous_timestamps: bool = True # Require continuous timestamps
    timezone_aware: bool = True              # Require timezone awareness
    
    # Asset ID validation
    require_asset_id_consistency: bool = True # Require consistent asset IDs
    max_asset_id_changes: int = 1            # Maximum asset ID changes per window
    
    # Scaling validation
    require_consistent_scaling: bool = True   # Require consistent scaling
    max_scale_variance: float = 0.1          # Maximum scale variance
    min_scale_consistency: float = 0.8       # Minimum scale consistency
    
    # Feature family validation
    enable_feature_family_validation: bool = True # Enable family validation
    max_family_correlation: float = 0.9      # Maximum correlation within family
    min_family_diversity: float = 0.1        # Minimum diversity within family
    
    # Data quality validation
    max_missing_ratio: float = 0.05          # Maximum missing data ratio
    max_outlier_ratio: float = 0.01          # Maximum outlier ratio
    min_data_quality_score: float = 0.8       # Minimum data quality score
    
    # Logging
    verbose: bool = True


@dataclass
class IntegrityViolation:
    """Represents an integrity violation."""
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_features: List[str]
    recommendation: str


@dataclass
class VectorIntegrityResult:
    """Result of vector integrity validation."""
    
    # Validation results
    is_valid: bool = True
    integrity_score: float = 1.0
    
    # Violations
    violations: List[IntegrityViolation] = field(default_factory=list)
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    
    # Feature family analysis
    feature_families: Dict[str, List[str]] = field(default_factory=dict)
    family_scores: Dict[str, float] = field(default_factory=dict)
    
    # Data quality metrics
    timestamp_consistency: float = 1.0
    asset_id_consistency: float = 1.0
    scaling_consistency: float = 1.0
    data_quality_score: float = 1.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class VectorIntegrityValidator:
    """
    Vector Integrity Validator for semantic consistency.
    
    Validates:
    1. Timestamp alignment and continuity
    2. Asset ID consistency
    3. Scaling/normalization consistency
    4. Feature family integrity
    5. Data quality metrics
    """
    
    def __init__(self, config: Optional[VectorIntegrityConfig] = None):
        """Initialize the vector integrity validator."""
        self.config = config or VectorIntegrityConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.verbose:
            tprint("🔍 Initializing VectorIntegrityValidator")
    
    def validate_vector_integrity(self, 
                                features: pd.DataFrame,
                                metadata: Optional[Dict[str, Any]] = None) -> VectorIntegrityResult:
        """
        Validate vector integrity for semantic consistency.
        
        Args:
            features: Input feature matrix
            metadata: Optional metadata (asset IDs, timestamps, etc.)
            
        Returns:
            VectorIntegrityResult
        """
        if self.config.verbose:
            tprint("🔍 Validating vector integrity")
        
        result = VectorIntegrityResult()
        
        # Validate timestamps
        timestamp_result = self._validate_timestamps(features, metadata)
        result.timestamp_consistency = timestamp_result['score']
        result.violations.extend(timestamp_result['violations'])
        
        # Validate asset IDs
        asset_result = self._validate_asset_ids(features, metadata)
        result.asset_id_consistency = asset_result['score']
        result.violations.extend(asset_result['violations'])
        
        # Validate scaling
        scaling_result = self._validate_scaling(features)
        result.scaling_consistency = scaling_result['score']
        result.violations.extend(scaling_result['violations'])
        
        # Validate feature families
        family_result = self._validate_feature_families(features)
        result.feature_families = family_result['families']
        result.family_scores = family_result['scores']
        result.violations.extend(family_result['violations'])
        
        # Validate data quality
        quality_result = self._validate_data_quality(features)
        result.data_quality_score = quality_result['score']
        result.violations.extend(quality_result['violations'])
        
        # Calculate overall integrity score
        result.integrity_score = self._calculate_integrity_score(result)
        
        # Count violations by severity
        self._count_violations(result)
        
        # Determine overall validity
        result.is_valid = self._determine_validity(result)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        if self.config.verbose:
            tprint(f"📊 Integrity score: {result.integrity_score:.4f}")
            tprint(f"✅ Valid: {result.is_valid}")
            tprint(f"⚠️ Violations: {len(result.violations)}")
        
        return result
    
    def _validate_timestamps(self, 
                           features: pd.DataFrame,
                           metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate timestamp consistency."""
        violations = []
        score = 1.0
        
        try:
            if not hasattr(features.index, 'to_pydatetime'):
                violations.append(IntegrityViolation(
                    violation_type="timestamp",
                    severity="critical",
                    description="DataFrame index is not datetime",
                    affected_features=[],
                    recommendation="Ensure DataFrame has datetime index"
                ))
                return {'score': 0.0, 'violations': violations}
            
            # Check timestamp continuity
            if self.config.require_continuous_timestamps:
                time_diffs = features.index.to_series().diff().dt.total_seconds() / 60  # minutes
                max_gap = time_diffs.max()
                
                if max_gap > self.config.max_timestamp_gap_minutes:
                    violations.append(IntegrityViolation(
                        violation_type="timestamp",
                        severity="high",
                        description=f"Timestamp gap too large: {max_gap:.1f} minutes",
                        affected_features=[],
                        recommendation="Ensure continuous timestamps"
                    ))
                    score -= 0.3
            
            # Check timezone awareness
            if self.config.timezone_aware and features.index.tz is None:
                violations.append(IntegrityViolation(
                    violation_type="timestamp",
                    severity="medium",
                    description="Timestamps not timezone-aware",
                    affected_features=[],
                    recommendation="Add timezone information to timestamps"
                ))
                score -= 0.1
            
        except Exception as e:
            violations.append(IntegrityViolation(
                violation_type="timestamp",
                severity="critical",
                description=f"Timestamp validation failed: {e}",
                affected_features=[],
                recommendation="Fix timestamp issues"
            ))
            score = 0.0
        
        return {'score': max(0.0, score), 'violations': violations}
    
    def _validate_asset_ids(self, 
                          features: pd.DataFrame,
                          metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate asset ID consistency."""
        violations = []
        score = 1.0
        
        try:
            if not metadata or 'asset_ids' not in metadata:
                if self.config.require_asset_id_consistency:
                    violations.append(IntegrityViolation(
                        violation_type="asset_id",
                        severity="medium",
                        description="Asset IDs not provided",
                        affected_features=[],
                        recommendation="Provide asset ID metadata"
                    ))
                    score -= 0.2
                return {'score': max(0.0, score), 'violations': violations}
            
            asset_ids = metadata['asset_ids']
            
            # Check asset ID consistency
            if len(asset_ids) != len(features):
                violations.append(IntegrityViolation(
                    violation_type="asset_id",
                    severity="critical",
                    description="Asset ID count mismatch",
                    affected_features=[],
                    recommendation="Ensure asset ID count matches feature count"
                ))
                score = 0.0
                return {'score': 0.0, 'violations': violations}
            
            # Check for asset ID changes
            asset_changes = (pd.Series(asset_ids) != pd.Series(asset_ids).shift()).sum()
            if asset_changes > self.config.max_asset_id_changes:
                violations.append(IntegrityViolation(
                    violation_type="asset_id",
                    severity="high",
                    description=f"Too many asset ID changes: {asset_changes}",
                    affected_features=[],
                    recommendation="Reduce asset ID changes"
                ))
                score -= 0.3
            
        except Exception as e:
            violations.append(IntegrityViolation(
                violation_type="asset_id",
                severity="critical",
                description=f"Asset ID validation failed: {e}",
                affected_features=[],
                recommendation="Fix asset ID issues"
            ))
            score = 0.0
        
        return {'score': max(0.0, score), 'violations': violations}
    
    def _validate_scaling(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate scaling consistency."""
        violations = []
        score = 1.0
        
        try:
            if not self.config.require_consistent_scaling:
                return {'score': 1.0, 'violations': violations}
            
            # Calculate scale variance across features
            feature_scales = features.std()
            scale_variance = feature_scales.std() / feature_scales.mean()
            
            if scale_variance > self.config.max_scale_variance:
                violations.append(IntegrityViolation(
                    violation_type="scaling",
                    severity="medium",
                    description=f"Scale variance too high: {scale_variance:.3f}",
                    affected_features=[],
                    recommendation="Normalize feature scales"
                ))
                score -= 0.2
            
            # Check scale consistency over time
            rolling_scales = features.rolling(20).std()
            scale_consistency = 1.0 - rolling_scales.std().mean() / rolling_scales.mean().mean()
            
            if scale_consistency < self.config.min_scale_consistency:
                violations.append(IntegrityViolation(
                    violation_type="scaling",
                    severity="medium",
                    description=f"Scale consistency too low: {scale_consistency:.3f}",
                    affected_features=[],
                    recommendation="Improve scale consistency over time"
                ))
                score -= 0.2
            
        except Exception as e:
            violations.append(IntegrityViolation(
                violation_type="scaling",
                severity="medium",
                description=f"Scaling validation failed: {e}",
                affected_features=[],
                recommendation="Fix scaling issues"
            ))
            score -= 0.1
        
        return {'score': max(0.0, score), 'violations': violations}
    
    def _validate_feature_families(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature family integrity."""
        violations = []
        families = {}
        family_scores = {}
        
        try:
            if not self.config.enable_feature_family_validation:
                return {'families': families, 'scores': family_scores, 'violations': violations}
            
            # Group features by family (based on naming convention)
            for col in features.columns:
                family_name = self._extract_family_name(col)
                if family_name not in families:
                    families[family_name] = []
                families[family_name].append(col)
            
            # Validate each family
            for family_name, family_features in families.items():
                if len(family_features) < 2:
                    continue
                
                family_data = features[family_features]
                
                # Check correlation within family
                corr_matrix = family_data.corr()
                max_corr = corr_matrix.where(
                    ~np.eye(corr_matrix.shape[0], dtype=bool), 0
                ).max().max()
                
                if max_corr > self.config.max_family_correlation:
                    violations.append(IntegrityViolation(
                        violation_type="feature_family",
                        severity="medium",
                        description=f"Family {family_name} too correlated: {max_corr:.3f}",
                        affected_features=family_features,
                        recommendation="Reduce correlation within family"
                    ))
                
                # Check diversity within family
                diversity = self._calculate_family_diversity(family_data)
                family_scores[family_name] = diversity
                
                if diversity < self.config.min_family_diversity:
                    violations.append(IntegrityViolation(
                        violation_type="feature_family",
                        severity="low",
                        description=f"Family {family_name} lacks diversity: {diversity:.3f}",
                        affected_features=family_features,
                        recommendation="Increase feature diversity within family"
                    ))
        
        except Exception as e:
            violations.append(IntegrityViolation(
                violation_type="feature_family",
                severity="medium",
                description=f"Feature family validation failed: {e}",
                affected_features=[],
                recommendation="Fix feature family issues"
            ))
        
        return {'families': families, 'scores': family_scores, 'violations': violations}
    
    def _validate_data_quality(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics."""
        violations = []
        score = 1.0
        
        try:
            # Check missing data ratio
            missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
            if missing_ratio > self.config.max_missing_ratio:
                violations.append(IntegrityViolation(
                    violation_type="data_quality",
                    severity="high",
                    description=f"Too much missing data: {missing_ratio:.3f}",
                    affected_features=[],
                    recommendation="Handle missing data"
                ))
                score -= 0.3
            
            # Check outlier ratio
            outlier_ratio = self._calculate_outlier_ratio(features)
            if outlier_ratio > self.config.max_outlier_ratio:
                violations.append(IntegrityViolation(
                    violation_type="data_quality",
                    severity="medium",
                    description=f"Too many outliers: {outlier_ratio:.3f}",
                    affected_features=[],
                    recommendation="Handle outliers"
                ))
                score -= 0.2
            
            # Check data quality score
            quality_score = self._calculate_data_quality_score(features)
            if quality_score < self.config.min_data_quality_score:
                violations.append(IntegrityViolation(
                    violation_type="data_quality",
                    severity="medium",
                    description=f"Low data quality score: {quality_score:.3f}",
                    affected_features=[],
                    recommendation="Improve data quality"
                ))
                score -= 0.2
            
        except Exception as e:
            violations.append(IntegrityViolation(
                violation_type="data_quality",
                severity="medium",
                description=f"Data quality validation failed: {e}",
                affected_features=[],
                recommendation="Fix data quality issues"
            ))
            score -= 0.1
        
        return {'score': max(0.0, score), 'violations': violations}
    
    def _extract_family_name(self, feature_name: str) -> str:
        """Extract family name from feature name."""
        # Simple heuristic: use prefix before first underscore
        parts = feature_name.split('_')
        return parts[0] if parts else 'unknown'
    
    def _calculate_family_diversity(self, family_data: pd.DataFrame) -> float:
        """Calculate diversity within a feature family."""
        try:
            # Calculate pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(family_data.T, metric='correlation')
            return np.mean(distances)
        except:
            return 0.0
    
    def _calculate_outlier_ratio(self, features: pd.DataFrame) -> float:
        """Calculate outlier ratio in features."""
        try:
            outlier_count = 0
            total_count = 0
            
            for col in features.columns:
                data = features[col].dropna()
                if len(data) == 0:
                    continue
                
                # Use IQR method for outlier detection
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                outlier_count += outliers
                total_count += len(data)
            
            return outlier_count / total_count if total_count > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_data_quality_score(self, features: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        try:
            # Combine multiple quality metrics
            missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
            outlier_ratio = self._calculate_outlier_ratio(features)
            
            # Quality score as combination of metrics
            quality_score = (
                (1.0 - missing_ratio) * 0.5 +
                (1.0 - outlier_ratio) * 0.3 +
                0.2  # Base score
            )
            
            return max(0.0, min(1.0, quality_score))
        except:
            return 0.0
    
    def _calculate_integrity_score(self, result: VectorIntegrityResult) -> float:
        """Calculate overall integrity score."""
        try:
            # Weighted combination of all scores
            integrity_score = (
                result.timestamp_consistency * 0.2 +
                result.asset_id_consistency * 0.2 +
                result.scaling_consistency * 0.2 +
                result.data_quality_score * 0.2 +
                np.mean(list(result.family_scores.values())) * 0.2 if result.family_scores else 0.2
            )
            
            return max(0.0, min(1.0, integrity_score))
        except:
            return 0.0
    
    def _count_violations(self, result: VectorIntegrityResult) -> None:
        """Count violations by severity."""
        for violation in result.violations:
            if violation.severity == "critical":
                result.critical_violations += 1
            elif violation.severity == "high":
                result.high_violations += 1
            elif violation.severity == "medium":
                result.medium_violations += 1
            else:
                result.low_violations += 1
    
    def _determine_validity(self, result: VectorIntegrityResult) -> bool:
        """Determine overall validity based on violations."""
        # Invalid if critical violations exist
        if result.critical_violations > 0:
            return False
        
        # Invalid if too many high severity violations
        if result.high_violations > 3:
            return False
        
        # Invalid if integrity score too low
        if result.integrity_score < 0.7:
            return False
        
        return True
    
    def _generate_recommendations(self, result: VectorIntegrityResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if result.critical_violations > 0:
            recommendations.append("Address critical violations immediately")
        
        if result.high_violations > 0:
            recommendations.append("Address high severity violations")
        
        if result.integrity_score < 0.8:
            recommendations.append("Improve overall integrity score")
        
        if result.timestamp_consistency < 0.9:
            recommendations.append("Improve timestamp consistency")
        
        if result.scaling_consistency < 0.8:
            recommendations.append("Improve scaling consistency")
        
        if result.data_quality_score < 0.8:
            recommendations.append("Improve data quality")
        
        return recommendations
