"""
Standardized Labeling Interface

This module provides a standardized interface for passing labels and weights
between components in the pre-training pipeline, ensuring consistent data
flow and proper error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


class LabelingFormat(Enum):
    """Supported labeling formats."""
    STANDARDIZED = "standardized"
    MULTI_HORIZON = "multi_horizon"
    TRIPLE_BARRIER = "triple_barrier"


@dataclass
class LabelingMetadata:
    """Metadata for labeling results."""
    source_component: str
    creation_time: str
    pipeline_ready: bool
    symbol: str
    exchange: str
    timeframe: str
    n_samples: int
    n_targets: int
    n_horizons: int
    error: Optional[str] = None


@dataclass
class StandardizedLabelingResult:
    """Standardized labeling result that all components can use."""
    labels: pd.DataFrame
    weights: Dict[str, float]
    target_columns: List[str]
    quality_scores: Dict[str, Any]
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    metadata: LabelingMetadata
    
    def is_valid(self) -> bool:
        """Check if the labeling result is valid."""
        return (
            not self.labels.empty and
            len(self.target_columns) > 0 and
            self.metadata.pipeline_ready and
            self.metadata.error is None
        )
    
    def get_best_target(self) -> Optional[str]:
        """Get the best target based on weights."""
        if not self.weights or not self.target_columns:
            # No weights available, use first available target
            available_targets = [col for col in self.labels.columns if col not in ['timestamp', 'symbol']]
            return available_targets[0] if available_targets else None

        # Priority order based on horizon weights (higher weight = higher priority)
        target_priority = []
        
        for target in self.target_columns:
            if target in self.labels.columns:
                # Determine horizon type from target name
                if 'immediate' in target.lower() or 'small' in target.lower():
                    horizon_weight = self.weights.get('small', 0.0)
                elif 'short' in target.lower() or 'medium' in target.lower():
                    horizon_weight = self.weights.get('medium', 0.0)
                elif 'leverage' in target.lower() or 'high' in target.lower():
                    horizon_weight = self.weights.get('high', 0.0)
                else:
                    # Default to small horizon if unclear
                    horizon_weight = self.weights.get('small', 0.0)
                
                target_priority.append((target, horizon_weight))

        # Sort by weight (descending) and return the highest weighted target
        if target_priority:
            target_priority.sort(key=lambda x: x[1], reverse=True)
            return target_priority[0][0]

        return None


class StandardizedLabelingInterface:
    """Interface for standardized labeling data exchange between components."""
    
    @staticmethod
    def create_from_multi_horizon_result(
        multi_horizon_result: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> StandardizedLabelingResult:
        """Create standardized result from multi_horizon_profit_labeler output."""
        try:
            tprint_info("🔄 Converting multi_horizon_profit_labeler result to standardized format")
            
            # Extract data from multi_horizon_result
            labeled_data = multi_horizon_result.get('labeled_data', pd.DataFrame())
            horizon_weights = multi_horizon_result.get('horizon_weights', {})
            target_columns = multi_horizon_result.get('target_columns', [])
            quality_scores = multi_horizon_result.get('quality_scores', {})
            confidence_scores = multi_horizon_result.get('confidence_scores', pd.DataFrame())
            eligibility_masks = multi_horizon_result.get('eligibility_masks', pd.DataFrame())
            
            # Create metadata
            metadata = LabelingMetadata(
                source_component='multi_horizon_profit_labeler',
                creation_time=datetime.now().isoformat(),
                pipeline_ready=True,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=len(labeled_data) if not labeled_data.empty else 0,
                n_targets=len(target_columns),
                n_horizons=len(horizon_weights)
            )
            
            result = StandardizedLabelingResult(
                labels=labeled_data,
                weights=horizon_weights,
                target_columns=target_columns,
                quality_scores=quality_scores,
                confidence_scores=confidence_scores,
                eligibility_masks=eligibility_masks,
                metadata=metadata
            )
            
            tprint_success("✅ Successfully created standardized labeling result")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to create standardized result: {e}")
            # Return empty result with error
            metadata = LabelingMetadata(
                source_component='multi_horizon_profit_labeler',
                creation_time=datetime.now().isoformat(),
                pipeline_ready=False,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=0,
                n_targets=0,
                n_horizons=0,
                error=str(e)
            )
            
            return StandardizedLabelingResult(
                labels=pd.DataFrame(),
                weights={},
                target_columns=[],
                quality_scores={},
                confidence_scores=pd.DataFrame(),
                eligibility_masks=pd.DataFrame(),
                metadata=metadata
            )
    
    @staticmethod
    def create_from_standardized_output(
        standardized_output: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> StandardizedLabelingResult:
        """Create standardized result from standardized output format."""
        try:
            tprint_info("🔄 Processing standardized output format")
            
            # Extract data from standardized output
            labels = standardized_output.get('labels', pd.DataFrame())
            weights = standardized_output.get('weights', {})
            target_columns = standardized_output.get('target_columns', [])
            quality_scores = standardized_output.get('quality_scores', {})
            confidence_scores = standardized_output.get('confidence_scores', pd.DataFrame())
            eligibility_masks = standardized_output.get('eligibility_masks', pd.DataFrame())
            
            # Create metadata
            metadata = LabelingMetadata(
                source_component=standardized_output.get('metadata', {}).get('source_component', 'unknown'),
                creation_time=standardized_output.get('metadata', {}).get('creation_time', datetime.now().isoformat()),
                pipeline_ready=standardized_output.get('metadata', {}).get('pipeline_ready', True),
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=len(labels) if not labels.empty else 0,
                n_targets=len(target_columns),
                n_horizons=len(weights)
            )
            
            result = StandardizedLabelingResult(
                labels=labels,
                weights=weights,
                target_columns=target_columns,
                quality_scores=quality_scores,
                confidence_scores=confidence_scores,
                eligibility_masks=eligibility_masks,
                metadata=metadata
            )
            
            tprint_success("✅ Successfully processed standardized output")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to process standardized output: {e}")
            # Return empty result with error
            metadata = LabelingMetadata(
                source_component='unknown',
                creation_time=datetime.now().isoformat(),
                pipeline_ready=False,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=0,
                n_targets=0,
                n_horizons=0,
                error=str(e)
            )
            
            return StandardizedLabelingResult(
                labels=pd.DataFrame(),
                weights={},
                target_columns=[],
                quality_scores={},
                confidence_scores=pd.DataFrame(),
                eligibility_masks=pd.DataFrame(),
                metadata=metadata
            )
    
    @staticmethod
    def extract_from_pipeline_state(
        pipeline_state: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[StandardizedLabelingResult]:
        """Extract standardized labeling result from pipeline state."""
        try:
            tprint_info("🔍 Extracting labeling result from pipeline state")
            
            # Try standardized output format first
            if 'standardized_output' in pipeline_state:
                tprint_info("📋 Found standardized output in pipeline state")
                return StandardizedLabelingInterface.create_from_standardized_output(
                    pipeline_state['standardized_output'], symbol, exchange, timeframe
                )
            
            # Try multi_horizon_labeling_result format
            if 'multi_horizon_labeling_result' in pipeline_state:
                tprint_info("📊 Found multi_horizon_labeling_result in pipeline state")
                return StandardizedLabelingInterface.create_from_multi_horizon_result(
                    pipeline_state['multi_horizon_labeling_result'], symbol, exchange, timeframe
                )
            
            # Try artifacts
            artifacts = pipeline_state.get('artifacts', {})
            if 'standardized_output' in artifacts:
                tprint_info("📋 Found standardized output in artifacts")
                return StandardizedLabelingInterface.create_from_standardized_output(
                    artifacts['standardized_output'], symbol, exchange, timeframe
                )
            
            if 'multi_horizon_labeling_result' in artifacts:
                tprint_info("📊 Found multi_horizon_labeling_result in artifacts")
                return StandardizedLabelingInterface.create_from_multi_horizon_result(
                    artifacts['multi_horizon_labeling_result'], symbol, exchange, timeframe
                )
            
            tprint_warning("⚠️ No labeling result found in pipeline state")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract labeling result from pipeline state: {e}")
            return None
    
    @staticmethod
    def validate_result(result: StandardizedLabelingResult) -> bool:
        """Validate a standardized labeling result."""
        try:
            if not result.is_valid():
                tprint_warning("⚠️ Labeling result is not valid")
                return False
            
            # Additional validation checks
            if result.labels.empty:
                tprint_warning("⚠️ Labels DataFrame is empty")
                return False
            
            if not result.target_columns:
                tprint_warning("⚠️ No target columns specified")
                return False
            
            # Check if target columns exist in labels
            missing_targets = [col for col in result.target_columns if col not in result.labels.columns]
            if missing_targets:
                tprint_warning(f"⚠️ Missing target columns: {missing_targets}")
                return False
            
            tprint_success("✅ Labeling result validation passed")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Validation failed: {e}")
            return False
