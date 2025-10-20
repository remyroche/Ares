"""
Comprehensive Validation Orchestrator

This module orchestrates all validation components to address the critical issues:
1. Label leakage prevention with nested OOF validation
2. Hierarchical validation to prevent objective function collapse
3. Anchored optimization to prevent recency bias
4. Interpretability feedback for interaction generation
5. Vector integrity validation for semantic consistency
6. Forward validation with walk-forward holdout testing
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Import all validation components
from .nested_oof_validator import NestedOOFValidator, NestedOOFConfig
from .hierarchical_validator import HierarchicalValidator, HierarchicalValidationConfig, ValidationStage
from .anchored_optimizer import AnchoredOptimizer, AnchoredOptimizationConfig
from .interpretability_feedback import InterpretabilityFeedbackLoop, InterpretabilityFeedbackConfig
from .vector_integrity_validator import VectorIntegrityValidator, VectorIntegrityConfig
from .forward_validator import ForwardValidator, ForwardValidationConfig
from .enhanced_vectorized_validator import EnhancedVectorizedValidator, EnhancedVectorizedConfig

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


@dataclass
class ComprehensiveValidationConfig:
    """Configuration for comprehensive validation."""
    
    # Component configurations
    nested_oof_config: Optional[NestedOOFConfig] = None
    hierarchical_config: Optional[HierarchicalValidationConfig] = None
    anchored_config: Optional[AnchoredOptimizationConfig] = None
    interpretability_config: Optional[InterpretabilityFeedbackConfig] = None
    vector_integrity_config: Optional[VectorIntegrityConfig] = None
    forward_config: Optional[ForwardValidationConfig] = None
    enhanced_vectorized_config: Optional[EnhancedVectorizedConfig] = None
    
    # Validation stages
    enable_label_leakage_prevention: bool = True
    enable_hierarchical_validation: bool = True
    enable_anchored_optimization: bool = True
    enable_interpretability_feedback: bool = True
    enable_vector_integrity: bool = True
    enable_forward_validation: bool = True
    enable_enhanced_vectorized: bool = True
    
    # Integration parameters
    validation_order: List[str] = field(default_factory=lambda: [
        "enhanced_vectorized",
        "vector_integrity",
        "nested_oof",
        "hierarchical",
        "anchored_optimization",
        "interpretability_feedback",
        "forward_validation"
    ])
    
    # Overall validation criteria
    min_overall_score: float = 0.7
    require_all_components: bool = False
    allow_partial_failure: bool = True
    
    # Logging
    verbose: bool = True


@dataclass
class ComprehensiveValidationResult:
    """Result of comprehensive validation."""
    
    # Component results
    enhanced_vectorized_result: Optional[Any] = None
    vector_integrity_result: Optional[Any] = None
    nested_oof_result: Optional[Any] = None
    hierarchical_results: Dict[ValidationStage, Any] = field(default_factory=dict)
    anchored_optimization_result: Optional[Any] = None
    interpretability_result: Optional[Any] = None
    forward_validation_result: Optional[Any] = None
    
    # Enhanced metrics
    performance_improvement: float = 1.0
    memory_efficiency: float = 1.0
    time_efficiency: float = 1.0
    vectorbt_optimization_gains: Dict[str, float] = field(default_factory=dict)
    vectorization_strategy: Optional[str] = None
    
    # Overall metrics
    overall_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    # Validation status
    passed_validation: bool = False
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Performance metrics
    validation_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0


class ComprehensiveValidator:
    """
    Comprehensive validation orchestrator.
    
    Addresses all critical pipeline issues:
    1. Label leakage prevention
    2. Economic validation overuse
    3. Recency bias prevention
    4. Interpretability feedback
    5. Vector integrity
    6. Forward validation
    """
    
    def __init__(self, config: Optional[ComprehensiveValidationConfig] = None):
        """Initialize the comprehensive validator."""
        self.config = config or ComprehensiveValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize component validators
        self._initialize_validators()
        
        if self.config.verbose:
            tprint("🔧 Initializing ComprehensiveValidator")
    
    def _initialize_validators(self) -> None:
        """Initialize all validation components."""
        # Nested OOF Validator
        if self.config.enable_label_leakage_prevention:
            self.nested_oof_validator = NestedOOFValidator(self.config.nested_oof_config)
        else:
            self.nested_oof_validator = None
        
        # Hierarchical Validator
        if self.config.enable_hierarchical_validation:
            self.hierarchical_validator = HierarchicalValidator(self.config.hierarchical_config)
        else:
            self.hierarchical_validator = None
        
        # Anchored Optimizer
        if self.config.enable_anchored_optimization:
            self.anchored_optimizer = AnchoredOptimizer(self.config.anchored_config)
        else:
            self.anchored_optimizer = None
        
        # Interpretability Feedback
        if self.config.enable_interpretability_feedback:
            self.interpretability_feedback = InterpretabilityFeedbackLoop(self.config.interpretability_config)
        else:
            self.interpretability_feedback = None
        
        # Vector Integrity Validator
        if self.config.enable_vector_integrity:
            self.vector_integrity_validator = VectorIntegrityValidator(self.config.vector_integrity_config)
        else:
            self.vector_integrity_validator = None
        
        # Forward Validator
        if self.config.enable_forward_validation:
            self.forward_validator = ForwardValidator(self.config.forward_config)
        else:
            self.forward_validator = None
        
        # Enhanced Vectorized Validator
        if self.config.enable_enhanced_vectorized:
            self.enhanced_vectorized_validator = EnhancedVectorizedValidator(self.config.enhanced_vectorized_config)
        else:
            self.enhanced_vectorized_validator = None
    
    def validate_pipeline(self, 
                         data: pd.DataFrame,
                         targets: pd.Series,
                         pipeline: callable,
                         metadata: Optional[Dict[str, Any]] = None) -> ComprehensiveValidationResult:
        """
        Perform comprehensive pipeline validation.
        
        Args:
            data: Input features
            targets: Target labels
            pipeline: Trained pipeline
            metadata: Optional metadata
            
        Returns:
            ComprehensiveValidationResult
        """
        if self.config.verbose:
            tprint("🔧 Starting comprehensive pipeline validation")
        
        result = ComprehensiveValidationResult()
        start_time = datetime.now()
        
        try:
            # Execute validation components in order
            for component in self.config.validation_order:
                if self.config.verbose:
                    tprint(f"🔄 Executing {component} validation")
                
                component_result = self._execute_component_validation(
                    component, data, targets, pipeline, metadata
                )
                
                # Store component result
                setattr(result, f"{component}_result", component_result)
                
                # Calculate component score
                component_score = self._calculate_component_score(component, component_result)
                result.component_scores[component] = component_score
                
                # Check for critical issues
                critical_issues = self._extract_critical_issues(component, component_result)
                result.critical_issues.extend(critical_issues)
                
                # Check for warnings
                warnings = self._extract_warnings(component, component_result)
                result.warnings.extend(warnings)
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result.component_scores)
            
            # Determine validation status
            result.passed_validation = self._determine_validation_status(result)
            
            # Generate recommendations
            result.recommendations = self._generate_comprehensive_recommendations(result)
            
            # Calculate performance metrics
            end_time = datetime.now()
            result.validation_time_seconds = (end_time - start_time).total_seconds()
            result.memory_usage_mb = self._calculate_memory_usage()
            
            if self.config.verbose:
                tprint_success(f"✅ Comprehensive validation completed")
                tprint(f"📊 Overall score: {result.overall_score:.4f}")
                tprint(f"✅ Passed: {result.passed_validation}")
                tprint(f"⚠️ Critical issues: {len(result.critical_issues)}")
                tprint(f"⚠️ Warnings: {len(result.warnings)}")
        
        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            result.critical_issues.append(f"Validation failed: {e}")
            result.passed_validation = False
        
        return result
    
    def _execute_component_validation(self, 
                                    component: str,
                                    data: pd.DataFrame,
                                    targets: pd.Series,
                                    pipeline: callable,
                                    metadata: Optional[Dict[str, Any]]) -> Any:
        """Execute validation for a specific component."""
        try:
            if component == "vector_integrity":
                return self.vector_integrity_validator.validate_vector_integrity(data, metadata)
            
            elif component == "nested_oof":
                return self.nested_oof_validator.perform_nested_validation(
                    data, targets, pipeline, pipeline
                )
            
            elif component == "hierarchical":
                # Perform hierarchical validation for each stage
                results = {}
                early_result = self.hierarchical_validator.validate_early_stage(data, targets)
                results[ValidationStage.EARLY] = early_result
                
                mid_result = self.hierarchical_validator.validate_mid_stage(data, targets, early_result)
                results[ValidationStage.MID] = mid_result
                
                late_result = self.hierarchical_validator.validate_late_stage(data, targets, mid_result)
                results[ValidationStage.LATE] = late_result
                
                return results
            
            elif component == "anchored_optimization":
                return self.anchored_optimizer.optimize_with_anchoring(
                    data, targets, pipeline, pipeline
                )
            
            elif component == "interpretability_feedback":
                return self.interpretability_feedback.iterative_pruning(
                    data, targets, self.interpretability_feedback.analyze_interpretability(data, targets)
                )
            
            elif component == "forward_validation":
                return self.forward_validator.perform_forward_validation(
                    data, targets, pipeline
                )
            
            elif component == "enhanced_vectorized":
                return self.enhanced_vectorized_validator.validate_pipeline_enhanced(
                    data, targets, pipeline, metadata
                )
            
            else:
                self.logger.warning(f"Unknown component: {component}")
                return None
        
        except Exception as e:
            self.logger.warning(f"Component {component} validation failed: {e}")
            return None
    
    def _calculate_component_score(self, component: str, component_result: Any) -> float:
        """Calculate score for a validation component."""
        try:
            if component_result is None:
                return 0.0
            
            if component == "vector_integrity":
                return getattr(component_result, 'integrity_score', 0.0)
            
            elif component == "nested_oof":
                return getattr(component_result, 'final_ic', 0.0)
            
            elif component == "hierarchical":
                # Average score across stages
                scores = []
                for stage_result in component_result.values():
                    if hasattr(stage_result, 'weighted_score'):
                        scores.append(stage_result.weighted_score)
                return np.mean(scores) if scores else 0.0
            
            elif component == "anchored_optimization":
                return getattr(component_result, 'mean_ic', 0.0)
            
            elif component == "interpretability_feedback":
                return getattr(component_result, 'final_score', 0.0)
            
            elif component == "forward_validation":
                return getattr(component_result, 'forward_ic', 0.0)
            
            elif component == "enhanced_vectorized":
                return getattr(component_result, 'overall_score', 0.0)
            
            else:
                return 0.0
        
        except Exception as e:
            self.logger.warning(f"Score calculation failed for {component}: {e}")
            return 0.0
    
    def _extract_critical_issues(self, component: str, component_result: Any) -> List[str]:
        """Extract critical issues from component result."""
        issues = []
        
        try:
            if component_result is None:
                issues.append(f"{component}: Validation failed")
                return issues
            
            if component == "vector_integrity":
                if hasattr(component_result, 'critical_violations') and component_result.critical_violations > 0:
                    issues.append(f"{component}: {component_result.critical_violations} critical violations")
            
            elif component == "nested_oof":
                if hasattr(component_result, 'label_leakage_detected') and component_result.label_leakage_detected:
                    issues.append(f"{component}: Label leakage detected")
            
            elif component == "hierarchical":
                for stage, stage_result in component_result.items():
                    if hasattr(stage_result, 'passed_thresholds') and not stage_result.passed_thresholds:
                        issues.append(f"{component}: {stage.value} stage failed thresholds")
            
            elif component == "anchored_optimization":
                if hasattr(component_result, 'recency_bias_detected') and component_result.recency_bias_detected:
                    issues.append(f"{component}: Recency bias detected")
            
            elif component == "interpretability_feedback":
                if hasattr(component_result, 'converged') and not component_result.converged:
                    issues.append(f"{component}: Interpretability feedback did not converge")
            
            elif component == "forward_validation":
                if hasattr(component_result, 'passed_forward_validation') and not component_result.passed_forward_validation:
                    issues.append(f"{component}: Forward validation failed")
        
        except Exception as e:
            self.logger.warning(f"Issue extraction failed for {component}: {e}")
        
        return issues
    
    def _extract_warnings(self, component: str, component_result: Any) -> List[str]:
        """Extract warnings from component result."""
        warnings = []
        
        try:
            if component_result is None:
                return warnings
            
            if component == "vector_integrity":
                if hasattr(component_result, 'high_violations') and component_result.high_violations > 0:
                    warnings.append(f"{component}: {component_result.high_violations} high severity violations")
            
            elif component == "nested_oof":
                if hasattr(component_result, 'isolation_violations') and component_result.isolation_violations:
                    warnings.append(f"{component}: {len(component_result.isolation_violations)} isolation violations")
            
            elif component == "hierarchical":
                for stage, stage_result in component_result.items():
                    if hasattr(stage_result, 'recommendations') and stage_result.recommendations:
                        warnings.append(f"{component}: {stage.value} stage has {len(stage_result.recommendations)} recommendations")
            
            elif component == "anchored_optimization":
                if hasattr(component_result, 'regime_instability_detected') and component_result.regime_instability_detected:
                    warnings.append(f"{component}: Regime instability detected")
            
            elif component == "interpretability_feedback":
                if hasattr(component_result, 'improvement') and component_result.improvement < 0:
                    warnings.append(f"{component}: Negative improvement in interpretability")
            
            elif component == "forward_validation":
                if hasattr(component_result, 'issues') and component_result.issues:
                    warnings.append(f"{component}: {len(component_result.issues)} forward validation issues")
        
        except Exception as e:
            self.logger.warning(f"Warning extraction failed for {component}: {e}")
        
        return warnings
    
    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate overall validation score."""
        try:
            if not component_scores:
                return 0.0
            
            # Weighted average of component scores
            weights = {
                "enhanced_vectorized": 0.25,
                "vector_integrity": 0.15,
                "nested_oof": 0.15,
                "hierarchical": 0.15,
                "anchored_optimization": 0.10,
                "interpretability_feedback": 0.10,
                "forward_validation": 0.10
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for component, score in component_scores.items():
                weight = weights.get(component, 0.1)
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _determine_validation_status(self, result: ComprehensiveValidationResult) -> bool:
        """Determine overall validation status."""
        try:
            # Check overall score
            if result.overall_score < self.config.min_overall_score:
                return False
            
            # Check critical issues
            if result.critical_issues:
                if self.config.require_all_components:
                    return False
                elif not self.config.allow_partial_failure:
                    return False
            
            return True
        
        except Exception as e:
            self.logger.warning(f"Validation status determination failed: {e}")
            return False
    
    def _generate_comprehensive_recommendations(self, result: ComprehensiveValidationResult) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Overall score recommendations
        if result.overall_score < 0.8:
            recommendations.append("Improve overall validation score")
        
        # Component-specific recommendations
        for component, score in result.component_scores.items():
            if score < 0.7:
                recommendations.append(f"Improve {component} validation score")
        
        # Critical issues recommendations
        if result.critical_issues:
            recommendations.append("Address critical validation issues")
        
        # Performance recommendations
        if result.validation_time_seconds > 300:  # 5 minutes
            recommendations.append("Optimize validation performance")
        
        if result.memory_usage_mb > 1000:  # 1GB
            recommendations.append("Optimize memory usage")
        
        return recommendations
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
