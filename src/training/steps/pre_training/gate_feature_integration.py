"""
Gate Feature Integration for Pre-Training Pipeline

This module provides integration patches for the existing pre-training pipeline
to ensure gate features are protected during final_feature_selection.

Integration Points:
1. analyst_pre_ml_orchestration
2. tactician_pre_ml_orchestration
3. final_feature_selection_pipeline
4. final_feature_selection_step
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from functools import wraps

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_warning, tprint_success, tprint_error
from .gate_feature_protection import (
    GateFeatureProtector, 
    GateFeatureConfig, 
    GateAwareFeatureSelector,
    create_gate_aware_selector
)


def patch_final_feature_selection_pipeline():
    """
    Patch the final_feature_selection_pipeline to protect gate features.
    
    This function modifies the MultiStageFeatureSelector class to use
    gate-aware feature selection.
    """
    try:
        from .final_feature_selection_pipeline import MultiStageFeatureSelector
        
        # Store original methods
        original_correlation_filtering = MultiStageFeatureSelector._correlation_filtering
        original_rfe_selection = MultiStageFeatureSelector._recursive_feature_elimination
        original_variance_filtering = MultiStageFeatureSelector._variance_filtering
        
        def patched_correlation_filtering(self, X, y, stage_name="correlation_filtering"):
            """Patched correlation filtering with gate protection."""
            tprint(f"🛡️ Applying gate-aware correlation filtering...")
            
            # Initialize gate protector
            gate_protector = GateFeatureProtector()
            
            # Protect gate features
            protected_X, protection_info = gate_protector.protect_gate_features(
                X, y, "correlation_filtering"
            )
            
            # Apply original correlation filtering to protected features
            result = original_correlation_filtering(self, protected_X, y, stage_name)
            
            # Log protection results
            if protection_info.get('valid_gate_count', 0) > 0:
                tprint(f"✅ Protected {protection_info['valid_gate_count']} gate features")
            
            return result
        
        def patched_rfe_selection(self, X, y):
            """Patched RFE selection with gate protection."""
            tprint(f"🛡️ Applying gate-aware RFE selection...")
            
            # Initialize gate protector
            gate_protector = GateFeatureProtector()
            
            # Protect gate features
            protected_X, protection_info = gate_protector.protect_gate_features(
                X, y, "rfe"
            )
            
            # Apply original RFE to protected features
            result = original_rfe_selection(self, protected_X, y)
            
            # Log protection results
            if protection_info.get('valid_gate_count', 0) > 0:
                tprint(f"✅ Protected {protection_info['valid_gate_count']} gate features")
            
            return result
        
        def patched_variance_filtering(self, X, y, stage_name="variance_filtering"):
            """Patched variance filtering with gate protection."""
            tprint(f"🛡️ Applying gate-aware variance filtering...")
            
            # Initialize gate protector
            gate_protector = GateFeatureProtector()
            
            # Protect gate features
            protected_X, protection_info = gate_protector.protect_gate_features(
                X, y, "variance_filtering"
            )
            
            # Apply original variance filtering to protected features
            result = original_variance_filtering(self, protected_X, y, stage_name)
            
            # Log protection results
            if protection_info.get('valid_gate_count', 0) > 0:
                tprint(f"✅ Protected {protection_info['valid_gate_count']} gate features")
            
            return result
        
        # Apply patches
        MultiStageFeatureSelector._correlation_filtering = patched_correlation_filtering
        MultiStageFeatureSelector._recursive_feature_elimination = patched_rfe_selection
        MultiStageFeatureSelector._variance_filtering = patched_variance_filtering
        
        tprint("✅ Successfully patched final_feature_selection_pipeline for gate protection")
        
    except ImportError as e:
        tprint_error(f"Failed to patch final_feature_selection_pipeline: {e}")


def patch_final_feature_selection_step():
    """
    Patch the final_feature_selection_step to use gate-aware selection.
    """
    try:
        from .final_feature_selection_step import FinalFeatureSelectionStep
        
        # Store original method
        original_run = FinalFeatureSelectionStep.run
        
        def patched_run(self, data, target, config=None):
            """Patched run method with gate protection."""
            tprint("🛡️ Running final feature selection with gate protection...")
            
            # Initialize gate protector
            gate_protector = GateFeatureProtector()
            
            # Identify gate features
            gate_features = gate_protector.identify_gate_features(data)
            
            if gate_features:
                tprint(f"🔍 Found {len(gate_features)} gate features to protect")
                
                # Add gate protection to config
                if config is None:
                    config = {}
                
                config['gate_protection'] = {
                    'enabled': True,
                    'gate_features': list(gate_features.keys()),
                    'protection_config': gate_protector.config.__dict__
                }
            
            # Run original method with gate protection
            result = original_run(self, data, target, config)
            
            # Log gate protection results
            if hasattr(result, 'gate_protection_info'):
                tprint(f"✅ Gate protection completed: {result.gate_protection_info}")
            
            return result
        
        # Apply patch
        FinalFeatureSelectionStep.run = patched_run
        
        tprint("✅ Successfully patched final_feature_selection_step for gate protection")
        
    except ImportError as e:
        tprint_error(f"Failed to patch final_feature_selection_step: {e}")


def patch_analyst_pre_ml_orchestration():
    """
    Patch analyst_pre_ml_orchestration to include gate features.
    """
    try:
        from .analyst_pre_ml_orchestration import AnalystPreMLOrchestration
        
        # Store original method
        original_run = AnalystPreMLOrchestration.run
        
        def patched_run(self, data, target, config=None):
            """Patched run method with gate feature generation."""
            tprint("🛡️ Running analyst pre-ML orchestration with gate features...")
            
            # Run original orchestration
            result = original_run(self, data, target, config)
            
            # Generate gate features if not already present
            if 'features' in result:
                gate_protector = GateFeatureProtector()
                gate_features = gate_protector.identify_gate_features(result['features'])
                
                if not gate_features:
                    tprint("🔧 Generating gate features for analyst...")
                    
                    # Generate gate features using negative learning
                    from src.feature_generation.categories.negative_learning import NegativeLearningPlugin
                    
                    gate_plugin = NegativeLearningPlugin()
                    gate_plugin.fit(result['features'], target)
                    enhanced_features = gate_plugin.transform(result['features'])
                    
                    # Update result with enhanced features
                    result['features'] = enhanced_features
                    result['gate_features'] = gate_plugin.get_negative_features()
                    
                    tprint(f"✅ Generated {len(result['gate_features'])} gate features for analyst")
            
            return result
        
        # Apply patch
        AnalystPreMLOrchestration.run = patched_run
        
        tprint("✅ Successfully patched analyst_pre_ml_orchestration for gate features")
        
    except ImportError as e:
        tprint_error(f"Failed to patch analyst_pre_ml_orchestration: {e}")


def patch_tactician_pre_ml_orchestration():
    """
    Patch tactician_pre_ml_orchestration to include gate features.
    """
    try:
        from .tactician_pre_ml_orchestration import TacticianPreMLOrchestration
        
        # Store original method
        original_run = TacticianPreMLOrchestration.run
        
        def patched_run(self, data, target, analyst_outputs=None, config=None):
            """Patched run method with gate feature generation."""
            tprint("🛡️ Running tactician pre-ML orchestration with gate features...")
            
            # Run original orchestration
            result = original_run(self, data, target, analyst_outputs, config)
            
            # Generate gate features if not already present
            if 'features' in result:
                gate_protector = GateFeatureProtector()
                gate_features = gate_protector.identify_gate_features(result['features'])
                
                if not gate_features:
                    tprint("🔧 Generating gate features for tactician...")
                    
                    # Generate gate features using negative learning
                    from src.feature_generation.categories.negative_learning import NegativeLearningPlugin
                    
                    gate_plugin = NegativeLearningPlugin()
                    gate_plugin.fit(result['features'], target)
                    enhanced_features = gate_plugin.transform(result['features'])
                    
                    # Update result with enhanced features
                    result['features'] = enhanced_features
                    result['gate_features'] = gate_plugin.get_negative_features()
                    
                    tprint(f"✅ Generated {len(result['gate_features'])} gate features for tactician")
            
            return result
        
        # Apply patch
        TacticianPreMLOrchestration.run = patched_run
        
        tprint("✅ Successfully patched tactician_pre_ml_orchestration for gate features")
        
    except ImportError as e:
        tprint_error(f"Failed to patch tactician_pre_ml_orchestration: {e}")


def apply_all_gate_protection_patches():
    """Apply all gate protection patches to the pre-training pipeline."""
    tprint("🛡️ Applying gate feature protection patches...")
    
    # Apply patches in order
    patch_final_feature_selection_pipeline()
    patch_final_feature_selection_step()
    patch_analyst_pre_ml_orchestration()
    patch_tactician_pre_ml_orchestration()
    
    tprint("✅ All gate protection patches applied successfully")


def create_gate_aware_pipeline_config():
    """Create configuration for gate-aware pipeline."""
    return {
        'gate_protection': {
            'enabled': True,
            'max_gate_features_per_base': 3,
            'min_gate_ic_improvement': 0.005,
            'min_gate_stability': 0.4,
            'gate_correlation_threshold': 0.95,
            'gate_importance_weight': 1.5,
            'gate_regime_bonus': 0.1,
            'validate_gate_contribution': True,
            'min_gate_contribution': 0.01,
            'enable_gate_interaction_validation': True
        },
        'feature_selection': {
            'correlation_thresholds': [0.92, 0.96, 0.98],
            'enable_rfe': True,
            'enable_variance_filtering': True,
            'enable_gate_protection': True
        }
    }


class GateFeaturePipelineManager:
    """
    Manager for gate feature integration in the pre-training pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or create_gate_aware_pipeline_config()
        self.logger = system_logger.getChild('GateFeaturePipelineManager')
        
        # Apply patches
        apply_all_gate_protection_patches()
    
    def run_analyst_pipeline_with_gates(
        self, 
        data: pd.DataFrame, 
        target: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run analyst pipeline with gate feature protection."""
        
        # Merge configs
        merged_config = {**self.config, **(config or {})}
        
        # Run analyst pre-ML orchestration
        from .analyst_pre_ml_orchestration import AnalystPreMLOrchestration
        
        analyst_orchestrator = AnalystPreMLOrchestration(merged_config)
        result = analyst_orchestrator.run(data, target, merged_config)
        
        # Ensure gate features are protected
        if 'features' in result:
            gate_protector = GateFeatureProtector()
            gate_features = gate_protector.identify_gate_features(result['features'])
            
            if gate_features:
                result['gate_features'] = list(gate_features.keys())
                result['gate_protection_applied'] = True
        
        return result
    
    def run_tactician_pipeline_with_gates(
        self, 
        data: pd.DataFrame, 
        target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run tactician pipeline with gate feature protection."""
        
        # Merge configs
        merged_config = {**self.config, **(config or {})}
        
        # Run tactician pre-ML orchestration
        from .tactician_pre_ml_orchestration import TacticianPreMLOrchestration
        
        tactician_orchestrator = TacticianPreMLOrchestration(merged_config)
        result = tactician_orchestrator.run(data, target, analyst_outputs, merged_config)
        
        # Ensure gate features are protected
        if 'features' in result:
            gate_protector = GateFeatureProtector()
            gate_features = gate_protector.identify_gate_features(result['features'])
            
            if gate_features:
                result['gate_features'] = list(gate_features.keys())
                result['gate_protection_applied'] = True
        
        return result
    
    def get_gate_protection_summary(self) -> Dict[str, Any]:
        """Get summary of gate protection status."""
        return {
            'patches_applied': True,
            'config': self.config,
            'gate_protection_enabled': self.config.get('gate_protection', {}).get('enabled', True)
        }


# Convenience functions for easy integration
def enable_gate_protection():
    """Enable gate feature protection for the entire pipeline."""
    apply_all_gate_protection_patches()
    tprint("✅ Gate feature protection enabled")


def create_gate_aware_manager(config: Optional[Dict[str, Any]] = None):
    """Create a gate-aware pipeline manager."""
    return GateFeaturePipelineManager(config)


# Auto-apply patches when module is imported
if __name__ != "__main__":
    try:
        apply_all_gate_protection_patches()
    except Exception as e:
        tprint_warning(f"Failed to auto-apply gate protection patches: {e}")