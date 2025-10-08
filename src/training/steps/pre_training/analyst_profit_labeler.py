"""
Analyst Profit Labeler - Specialized Multi-Horizon Labeling for Analyst Models

This module provides a specialized profit labeling component for Analyst models,
using the VolatilityAwareMultiHorizonLabeler with Analyst-specific configurations.

Key Features:
- 60m timeframe optimization for strategic decision-making
- Multi-horizon profit labeling (1h, 4h, 12h, 24h horizons)
- Volatility-aware target bands
- Enhanced label quality scoring
- Per-regime/cluster optimization support
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from src.training.steps.pre_training.components.contracts import PipelineState
from src.training.steps.pre_training.components.component_factory import register_component

# Import the volatility-aware labeler
try:
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler,
        VolatilityAwareConfig,
        LabelingResult,
        LabelDefinitionType,
        create_enhanced_analyst_labeler,
    )
    VOLATILITY_LABELER_AVAILABLE = True
except (ImportError, SyntaxError):
    VOLATILITY_LABELER_AVAILABLE = False
    VolatilityAwareMultiHorizonLabeler = None
    VolatilityAwareConfig = None
    LabelingResult = None
    LabelDefinitionType = None
    create_enhanced_analyst_labeler = None


@dataclass
class AnalystProfitLabelerConfig:
    """Configuration for Analyst profit labeling."""
    
    # Timeframe settings (Analyst operates on 60m)
    timeframe: str = "60m"
    base_period_minutes: int = 60
    
    # Horizon settings for Analyst (strategic decision-making)
    horizons: List[int] = field(default_factory=lambda: [60, 240, 720, 1440])  # 1h, 4h, 12h, 24h in minutes
    
    # Profit targets (percentage)
    target_profits: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0])
    
    # Volatility-aware settings
    use_volatility_normalization: bool = True
    volatility_window: int = 20
    
    # Label quality thresholds
    min_label_quality: float = 0.6
    min_predictability: float = 0.55
    
    # Per-regime optimization
    enable_regime_adaptation: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


class AnalystProfitLabeler:
    """
    Analyst Profit Labeler - Specialized labeling for Analyst models.
    
    This class wraps the VolatilityAwareMultiHorizonLabeler with Analyst-specific
    configurations and provides a simplified interface for Analyst model training.
    """
    
    def __init__(self, config: Optional[AnalystProfitLabelerConfig] = None):
        """Initialize the Analyst profit labeler."""
        self.config = config or AnalystProfitLabelerConfig()
        self.logger = system_logger.getChild('AnalystProfitLabeler')
        
        if not VOLATILITY_LABELER_AVAILABLE:
            raise RuntimeError(
                "VolatilityAwareMultiHorizonLabeler is not available. "
                "Please ensure the profit_labeling module is properly installed."
            )
        
        # Create the underlying labeler with Analyst-specific config
        self.labeler = self._create_labeler()
        
        tprint_success(f"✅ AnalystProfitLabeler initialized (timeframe: {self.config.timeframe})")
    
    def _create_labeler(self) -> Any:
        """Create and configure the VolatilityAwareMultiHorizonLabeler for Analyst."""
        # Create Analyst-specific configuration
        labeler_config = VolatilityAwareConfig()
        
        # Set label definition type to Analyst
        labeler_config.label_definition_type = LabelDefinitionType.ANALYST
        labeler_config.enable_enhanced_labels = True
        
        # Configure timeframe and horizons
        labeler_config.timeframe = self.config.timeframe
        labeler_config.multi_target.horizons = self.config.horizons
        labeler_config.multi_target.target_profits = self.config.target_profits
        
        # Configure volatility settings
        labeler_config.volatility.enabled = self.config.use_volatility_normalization
        labeler_config.volatility.window = self.config.volatility_window
        
        # Configure quality scoring
        labeler_config.quality_scoring.min_quality_threshold = self.config.min_label_quality
        labeler_config.quality_scoring.min_predictability = self.config.min_predictability
        
        # Configure regime adaptation
        labeler_config.regime_config.enabled = self.config.enable_regime_adaptation
        
        # Apply custom parameters
        if self.config.custom_params:
            for key, value in self.config.custom_params.items():
                if hasattr(labeler_config, key):
                    setattr(labeler_config, key, value)
        
        # Create the enhanced Analyst labeler
        return create_enhanced_analyst_labeler(config=labeler_config)
    
    def generate_labels(
        self,
        data: pd.DataFrame,
        regime_assignments: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> LabelingResult:
        """
        Generate Analyst profit labels for the input data.
        
        Args:
            data: Input market data (OHLCV format)
            regime_assignments: Optional regime assignments for regime-aware labeling
            **kwargs: Additional parameters for the labeler
            
        Returns:
            LabelingResult with labels, confidence scores, and quality metrics
        """
        tprint_info(f"📈 Generating Analyst profit labels for {len(data)} samples...")
        
        try:
            # Add regime assignments to kwargs if provided
            if regime_assignments is not None:
                kwargs['regime_assignments'] = regime_assignments
            
            # Generate labels using the underlying labeler
            result = self.labeler.generate_labels(data, **kwargs)
            
            tprint_success(
                f"✅ Analyst labels generated: {result.n_samples} samples, "
                f"{result.n_targets} targets, {result.n_horizons} horizons"
            )
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate Analyst labels: {e}")
            raise
    
    def get_label_summary(self, result: LabelingResult) -> Dict[str, Any]:
        """Get a summary of the labeling results."""
        summary = {
            'n_samples': result.n_samples,
            'n_targets': result.n_targets,
            'n_horizons': result.n_horizons,
            'processing_time': result.processing_time,
            'quality_scores': {}
        }
        
        # Add quality scores
        if result.quality_scores:
            for target_name, quality in result.quality_scores.items():
                summary['quality_scores'][target_name] = {
                    'overall_quality': quality.overall_quality,
                    'predictability': quality.predictability,
                    'stability': quality.stability,
                    'balance': quality.balance
                }
        
        return summary


@register_component('analyst_profit_labeler')
class AnalystProfitLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Analyst Profit Labeler.
    
    This component integrates the AnalystProfitLabeler with the pre-training pipeline
    and handles proper error handling, reporting, and pipeline state management.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Analyst profit labeler component."""
        super().__init__(config)
        self.logger = system_logger.getChild('AnalystProfitLabelerComponent')
        
        # Create Analyst-specific configuration
        analyst_config = AnalystProfitLabelerConfig()
        
        # Override with custom parameters if provided
        if self.config and self.config.custom_params:
            custom_params = self.config.custom_params
            
            # Update timeframe if provided
            if 'timeframe' in custom_params:
                analyst_config.timeframe = custom_params['timeframe']
                # Update base period based on timeframe
                if analyst_config.timeframe.endswith('m'):
                    analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1])
                elif analyst_config.timeframe.endswith('h'):
                    analyst_config.base_period_minutes = int(analyst_config.timeframe[:-1]) * 60
            
            # Update other parameters
            for key in ['horizons', 'target_profits', 'min_label_quality', 'min_predictability']:
                if key in custom_params:
                    setattr(analyst_config, key, custom_params[key])
            
            # Store all custom params for the underlying labeler
            analyst_config.custom_params = custom_params
        
        # Create the labeler
        try:
            self.labeler = AnalystProfitLabeler(analyst_config)
            tprint_success("✅ AnalystProfitLabelerComponent initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize AnalystProfitLabelerComponent: {e}")
            raise
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']
    
    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute Analyst profit labeling as a component.
        
        Args:
            data: Input data (typically market data DataFrame)
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with labeling results and artifacts
        """
        try:
            tprint_info("🚀 Starting Analyst Profit Labeling execution...")
            
            # Extract data from pipeline state if not provided
            if data is None:
                data = pipeline_state.get('prepared_data')
                if data is None:
                    raise ValueError("No input data provided and no prepared_data in pipeline state")
            
            # Extract regime assignments if available
            regime_assignments = pipeline_state.get('regime_assignments')
            if regime_assignments is not None:
                tprint_info(f"📊 Using regime assignments: {len(regime_assignments)} regimes")
            
            # Generate labels
            labeling_result = self.labeler.generate_labels(
                data=data,
                regime_assignments=regime_assignments
            )
            
            # Create artifacts
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': labeling_result.labels,
                    'labels': labeling_result.labels,
                    'confidence_scores': labeling_result.confidence_scores,
                    'eligibility_masks': labeling_result.eligibility_masks,
                    'quality_scores': labeling_result.quality_scores,
                    'normalization_factors': labeling_result.normalization_factors,
                    'processing_time': labeling_result.processing_time,
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                    'method': 'analyst_profit_labeling',
                },
                'labeling_report': {
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'method': 'analyst_profit_labeling',
                    'timeframe': self.labeler.config.timeframe,
                    'summary': self.labeler.get_label_summary(labeling_result),
                    'horizons': self.labeler.config.horizons,
                    'target_profits': self.labeler.config.target_profits,
                }
            }
            
            # Create result
            result = ComponentResult(
                success=True,
                data=labeling_result.labels,
                artifacts=artifacts,
                metadata={
                    'component': 'analyst_profit_labeler',
                    'timeframe': self.labeler.config.timeframe,
                    'n_samples': labeling_result.n_samples,
                    'n_targets': labeling_result.n_targets,
                    'n_horizons': labeling_result.n_horizons,
                }
            )
            
            tprint_success("✅ Analyst Profit Labeling completed successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Analyst Profit Labeling failed: {e}")
            
            result = ComponentResult(
                success=False,
                error_message=str(e),
                metadata={'component': 'analyst_profit_labeler'}
            )
            return result


# Convenience function for external usage
async def execute_analyst_profit_labeling(
    data: pd.DataFrame,
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[AnalystProfitLabelerConfig] = None,
    **kwargs
) -> LabelingResult:
    """
    Execute Analyst profit labeling.
    
    Args:
        data: Input market data (OHLCV format)
        regime_assignments: Optional regime assignments
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        LabelingResult with labels and quality metrics
    """
    labeler = AnalystProfitLabeler(config)
    return labeler.generate_labels(data, regime_assignments, **kwargs)