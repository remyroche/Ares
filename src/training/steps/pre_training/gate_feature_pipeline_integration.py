"""
Gate Feature Pipeline Integration

This module provides integration of gate features with the pre-training pipeline,
ensuring quality gates are applied throughout the feature generation and selection process.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager, GateFeatureConfig, GateFeatureResult,
    GateStatus, GateFeatureType, get_gate_manager, create_gate_manager
)
from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


class GateFeaturePipelineIntegration(BasePreTrainingComponent):
    """
    Integration component for gate features in the pre-training pipeline.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the gate feature pipeline integration."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Initialize gate feature manager
        gate_config = self.config.custom_params.get('gate_protection', {}) if hasattr(self.config, 'custom_params') else {}
        self.gate_manager = create_gate_manager(gate_config)
        
        # Integration settings
        self.enable_gate_integration = gate_config.get('enabled', True)
        self.gate_evaluation_frequency = gate_config.get('evaluation_frequency', 1)
        self.gate_failure_threshold = gate_config.get('failure_threshold', 0.5)
        
        self.evaluation_count = 0
        self.gate_results_history = []
    
    def process(self, data: Dict[str, Any]) -> ComponentResult:
        """
        Process data through gate feature integration.
        
        Args:
            data: Input data containing features and targets
            
        Returns:
            ComponentResult with processed data and gate evaluation results
        """
        try:
            tprint_info("🛡️ Starting gate feature pipeline integration...")
            
            # Extract features and targets
            features = data.get('features')
            targets = data.get('targets')
            
            if features is None or targets is None:
                return ComponentResult(
                    success=False,
                    data=data,
                    error_message="Missing features or targets in input data"
                )
            
            # Convert to DataFrame/Series if needed
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            if not isinstance(targets, pd.Series):
                targets = pd.Series(targets)
            
            # Evaluate gate features
            gate_results = self._evaluate_gate_features(features, targets)
            
            # Check if gates passed
            gate_success = self._check_gate_success(gate_results)
            
            if not gate_success:
                tprint_warning("⚠️ Gate features failed - applying corrective measures...")
                features, targets = self._apply_corrective_measures(features, targets, gate_results)
            
            # Select gate features for integration
            selected_gate_features = self._select_gate_features(features, targets)
            
            # Integrate gate features
            enhanced_data = self._integrate_gate_features(data, selected_gate_features, gate_results)
            
            # Update evaluation count
            self.evaluation_count += 1
            
            tprint_success(f"✅ Gate feature integration completed (evaluation #{self.evaluation_count})")
            
            return ComponentResult(
                success=True,
                data=enhanced_data,
                metadata={
                    'gate_results': gate_results,
                    'selected_gate_features': selected_gate_features,
                    'evaluation_count': self.evaluation_count,
                    'gate_success': gate_success
                }
            )
            
        except Exception as e:
            self.logger.error(f"Gate feature integration failed: {e}")
            return ComponentResult(
                success=False,
                data=data,
                error_message=f"Gate feature integration failed: {str(e)}"
            )
    
    def _evaluate_gate_features(self, features: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate gate features for the given data."""
        if not self.enable_gate_integration:
            return []
        
        # Only evaluate at specified frequency
        if self.evaluation_count % self.gate_evaluation_frequency != 0:
            return []
        
        return self.gate_manager.evaluate_gate_features(features, targets)
    
    def _check_gate_success(self, gate_results: List[GateFeatureResult]) -> bool:
        """Check if gate features passed successfully."""
        if not gate_results:
            return True
        
        # Calculate success rate
        total_gates = len(gate_results)
        failed_gates = len([r for r in gate_results if r.status == GateStatus.FAILED])
        success_rate = 1.0 - (failed_gates / total_gates)
        
        return success_rate >= self.gate_failure_threshold
    
    def _apply_corrective_measures(self, features: pd.DataFrame, targets: pd.Series, 
                                 gate_results: List[GateFeatureResult]) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply corrective measures based on gate results."""
        corrected_features = features.copy()
        corrected_targets = targets.copy()
        
        for result in gate_results:
            if result.status == GateStatus.FAILED:
                if result.gate_type == GateFeatureType.QUALITY_GATE:
                    # Handle quality issues
                    corrected_features = self._handle_quality_issues(corrected_features, result)
                elif result.gate_type == GateFeatureType.CORRELATION_GATE:
                    # Handle correlation issues
                    corrected_features = self._handle_correlation_issues(corrected_features, result)
                elif result.gate_type == GateFeatureType.VARIANCE_GATE:
                    # Handle variance issues
                    corrected_features = self._handle_variance_issues(corrected_features, result)
        
        return corrected_features, corrected_targets
    
    def _handle_quality_issues(self, features: pd.DataFrame, result: GateFeatureResult) -> pd.DataFrame:
        """Handle quality gate failures."""
        # Remove features with high NaN ratios
        nan_ratios = features.isnull().sum() / len(features)
        high_nan_features = nan_ratios > 0.5
        if high_nan_features.any():
            features_to_remove = high_nan_features[high_nan_features].index.tolist()
            features = features.drop(columns=features_to_remove)
            tprint_warning(f"Removed {len(features_to_remove)} features with high NaN ratios")
        
        # Fill remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def _handle_correlation_issues(self, features: pd.DataFrame, result: GateFeatureResult) -> pd.DataFrame:
        """Handle correlation gate failures."""
        # Remove highly correlated features
        corr_matrix = features.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2, _ in high_corr_pairs:
            if feat1 not in features_to_remove:
                features_to_remove.add(feat2)
        
        if features_to_remove:
            features = features.drop(columns=list(features_to_remove))
            tprint_warning(f"Removed {len(features_to_remove)} highly correlated features")
        
        return features
    
    def _handle_variance_issues(self, features: pd.DataFrame, result: GateFeatureResult) -> pd.DataFrame:
        """Handle variance gate failures."""
        # Remove low variance features
        variances = features.var()
        low_variance_features = variances < 1e-8
        
        if low_variance_features.any():
            features_to_remove = low_variance_features[low_variance_features].index.tolist()
            features = features.drop(columns=features_to_remove)
            tprint_warning(f"Removed {len(features_to_remove)} low variance features")
        
        return features
    
    def _select_gate_features(self, features: pd.DataFrame, targets: pd.Series) -> List[str]:
        """Select gate features for integration."""
        if not self.enable_gate_integration:
            return []
        
        return self.gate_manager.select_gate_features(features, targets)
    
    def _integrate_gate_features(self, data: Dict[str, Any], gate_features: List[str], 
                               gate_results: List[GateFeatureResult]) -> Dict[str, Any]:
        """Integrate gate features into the data pipeline."""
        enhanced_data = data.copy()
        
        # Add gate feature metadata
        enhanced_data['gate_features'] = {
            'selected_features': gate_features,
            'gate_results': [
                {
                    'feature_name': r.feature_name,
                    'gate_type': r.gate_type.value,
                    'status': r.status.value,
                    'score': r.score,
                    'threshold': r.threshold,
                    'message': r.message
                }
                for r in gate_results
            ],
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_count': self.evaluation_count
        }
        
        # Add gate feature status
        enhanced_data['gate_status'] = self.gate_manager.get_gate_status()
        
        # Store gate results history
        self.gate_results_history.extend(gate_results)
        
        return enhanced_data
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get gate feature statistics."""
        if not self.gate_results_history:
            return {}
        
        total_evaluations = len(self.gate_results_history)
        passed_gates = len([r for r in self.gate_results_history if r.status == GateStatus.PASSED])
        failed_gates = len([r for r in self.gate_results_history if r.status == GateStatus.FAILED])
        warning_gates = len([r for r in self.gate_results_history if r.status == GateStatus.WARNING])
        
        return {
            'total_evaluations': total_evaluations,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'warning_gates': warning_gates,
            'success_rate': passed_gates / total_evaluations if total_evaluations > 0 else 0.0,
            'gate_manager_status': self.gate_manager.get_gate_status()
        }
    
    def reset_gate_statistics(self) -> None:
        """Reset gate feature statistics."""
        self.evaluation_count = 0
        self.gate_results_history.clear()
        tprint_info("🔄 Gate feature statistics reset")


def create_gate_feature_integration(config: Optional[ComponentConfig] = None) -> GateFeaturePipelineIntegration:
    """
    Factory function to create a gate feature pipeline integration component.
    
    Args:
        config: Optional component configuration
        
    Returns:
        GateFeaturePipelineIntegration instance
    """
    return GateFeaturePipelineIntegration(config)


def integrate_gate_features_with_pipeline(pipeline_data: Dict[str, Any], 
                                        gate_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to integrate gate features with pipeline data.
    
    Args:
        pipeline_data: Pipeline data containing features and targets
        gate_config: Optional gate feature configuration
        
    Returns:
        Enhanced pipeline data with gate feature integration
    """
    integration = create_gate_feature_integration()
    if gate_config:
        integration.gate_manager = create_gate_manager(gate_config)
    
    result = integration.process(pipeline_data)
    return result.data if result.success else pipeline_data
