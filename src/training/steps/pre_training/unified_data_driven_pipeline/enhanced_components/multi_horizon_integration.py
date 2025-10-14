"""
Multi-Horizon Profit Labeling Integration for UnifiedDataDrivenPipeline

This module integrates multi-horizon profit labeling functionality from FeatureLookbackOptimizationComponent
into the UnifiedDataDrivenPipeline, including:
- Multi-horizon profit labeler integration
- Target column selection and alignment
- Direction-specific target generation
- Labeling result normalization and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import multi-horizon profit labeler
try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler import (
        MultiHorizonProfitLabeler, MultiHorizonConfig
    )
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False
    MultiHorizonProfitLabeler = None
    MultiHorizonConfig = None

# Import labeling result types
try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler.types import (
        MultiHorizonLabelingResult, LabelingMetadata
    )
    LABELING_TYPES_AVAILABLE = True
except ImportError:
    LABELING_TYPES_AVAILABLE = False
    MultiHorizonLabelingResult = None
    LabelingMetadata = None

logger = logging.getLogger(__name__)


class TargetDirection(Enum):
    """Target directions for labeling."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


@dataclass
class TargetColumnInfo:
    """Information about a target column."""
    column_name: str
    direction: TargetDirection
    horizon: int
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class MultiHorizonIntegrationResult:
    """Result of multi-horizon integration."""
    target_columns: Dict[str, str]  # direction -> target_column
    labeling_result: Optional[Any]
    target_info: List[TargetColumnInfo]
    integration_success: bool
    error_message: Optional[str] = None


class MultiHorizonIntegration:
    """
    Multi-horizon profit labeling integration for UnifiedDataDrivenPipeline.
    
    Features:
    - Multi-horizon profit labeler integration
    - Target column selection and alignment
    - Direction-specific target generation
    - Labeling result normalization and validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-horizon integration."""
        self.config = config or {}
        
        # Initialize multi-horizon labeler
        self._initialize_multi_horizon_labeler()
        
        # Performance tracking
        self.performance_stats = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'total_execution_time': 0.0,
            'labeling_operations': 0,
            'target_selections': 0
        }
        
        tprint_success("✅ Multi-Horizon Integration initialized")
    
    def _initialize_multi_horizon_labeler(self):
        """Initialize the multi-horizon profit labeler."""
        if MULTI_HORIZON_AVAILABLE:
            try:
                # Create default configuration
                labeler_config = MultiHorizonConfig(
                    horizons=[1, 2, 3, 5, 10, 20],
                    profit_threshold=0.001,
                    stop_loss_threshold=0.001,
                    enable_direction_specific=True
                )
                
                self.multi_horizon_labeler = MultiHorizonProfitLabeler(labeler_config)
                tprint_success("✅ Multi-horizon profit labeler initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Multi-horizon labeler initialization failed: {e}")
                self.multi_horizon_labeler = None
        else:
            self.multi_horizon_labeler = None
            tprint_warning("⚠️ Multi-horizon profit labeler not available")
    
    def integrate_multi_horizon_labeling(
        self,
        data: pd.DataFrame,
        pipeline_state: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False
    ) -> MultiHorizonIntegrationResult:
        """
        Integrate multi-horizon profit labeling into the pipeline.
        
        Args:
            data: Input market data
            pipeline_state: Current pipeline state
            force_refresh: Whether to force refresh of labeling results
            
        Returns:
            MultiHorizonIntegrationResult with target columns and labeling info
        """
        tprint_info("🚀 Starting multi-horizon profit labeling integration")
        start_time = time.time()
        
        try:
            self.performance_stats['total_integrations'] += 1
            
            # Try to load existing labeling results first
            labeling_result = None
            if not force_refresh:
                labeling_result = self._load_existing_labeling_results(pipeline_state)
            
            # Generate new labeling results if needed
            if labeling_result is None:
                tprint_info("🧪 Generating new multi-horizon labeling results")
                labeling_result = self._generate_labeling_results(data)
            
            if labeling_result is None:
                tprint_error("❌ Failed to generate labeling results")
                return MultiHorizonIntegrationResult(
                    target_columns={},
                    labeling_result=None,
                    target_info=[],
                    integration_success=False,
                    error_message="Failed to generate labeling results"
                )
            
            # Extract target columns
            target_columns = self._extract_target_columns(labeling_result)
            
            # Generate target info
            target_info = self._generate_target_info(target_columns, labeling_result)
            
            # Validate integration
            if not target_columns:
                tprint_error("❌ No target columns extracted from labeling results")
                return MultiHorizonIntegrationResult(
                    target_columns={},
                    labeling_result=labeling_result,
                    target_info=target_info,
                    integration_success=False,
                    error_message="No target columns extracted"
                )
            
            execution_time = time.time() - start_time
            self.performance_stats['successful_integrations'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['labeling_operations'] += 1
            self.performance_stats['target_selections'] += len(target_columns)
            
            tprint_success(f"✅ Multi-horizon integration completed in {execution_time:.3f}s")
            tprint_info(f"📊 Extracted {len(target_columns)} target columns")
            
            return MultiHorizonIntegrationResult(
                target_columns=target_columns,
                labeling_result=labeling_result,
                target_info=target_info,
                integration_success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats['failed_integrations'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            tprint_error(f"❌ Multi-horizon integration failed: {e}")
            return MultiHorizonIntegrationResult(
                target_columns={},
                labeling_result=None,
                target_info=[],
                integration_success=False,
                error_message=str(e)
            )
    
    def _load_existing_labeling_results(self, pipeline_state: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Load existing labeling results from pipeline state."""
        if not pipeline_state:
            return None
        
        try:
            # Try to get from pipeline state
            labeling_result = pipeline_state.get('multi_horizon_labeling_result')
            if labeling_result:
                tprint_debug("📊 Found labeling results in pipeline state")
                return self._normalize_labeling_result(labeling_result)
            
            # Try to get from artifacts
            artifacts = pipeline_state.get('artifacts', {})
            if isinstance(artifacts, dict):
                artifact_result = artifacts.get('multi_horizon_labeling_result')
                if artifact_result:
                    tprint_debug("📊 Found labeling results in artifacts")
                    return self._normalize_labeling_result(artifact_result)
            
            return None
            
        except Exception as e:
            tprint_debug(f"Failed to load existing labeling results: {e}")
            return None
    
    def _generate_labeling_results(self, data: pd.DataFrame) -> Optional[Any]:
        """Generate new multi-horizon labeling results."""
        if not self.multi_horizon_labeler:
            tprint_warning("⚠️ Multi-horizon labeler not available")
            return None
        
        try:
            tprint_debug("🧪 Generating multi-horizon labeling results")
            
            # Ensure we have required columns
            required_columns = ['close', 'open', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                tprint_error(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Generate labeling results
            labeling_result = self.multi_horizon_labeler.label_data(data)
            
            if labeling_result is None:
                tprint_error("❌ Labeling result is None")
                return None
            
            tprint_success("✅ Multi-horizon labeling results generated")
            return labeling_result
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate labeling results: {e}")
            return None
    
    def _normalize_labeling_result(self, raw_result: Any) -> Optional[Any]:
        """Normalize labeling result to standard format."""
        try:
            if raw_result is None:
                return None
            
            # If it's already in the correct format, return as is
            if hasattr(raw_result, 'target_columns') and hasattr(raw_result, 'metadata'):
                return raw_result
            
            # If it's a dictionary, try to extract the relevant parts
            if isinstance(raw_result, dict):
                if 'multi_horizon_labeling_result' in raw_result:
                    return self._normalize_labeling_result(raw_result['multi_horizon_labeling_result'])
                
                # Check if it has the expected structure
                if 'target_columns' in raw_result or 'labels' in raw_result:
                    return raw_result
            
            return raw_result
            
        except Exception as e:
            tprint_debug(f"Failed to normalize labeling result: {e}")
            return raw_result
    
    def _extract_target_columns(self, labeling_result: Any) -> Dict[str, str]:
        """Extract target columns from labeling result."""
        target_columns = {}
        
        try:
            if hasattr(labeling_result, 'target_columns'):
                # Direct access to target_columns attribute
                raw_targets = labeling_result.target_columns
            elif isinstance(labeling_result, dict) and 'target_columns' in labeling_result:
                # Dictionary with target_columns key
                raw_targets = labeling_result['target_columns']
            else:
                tprint_warning("⚠️ No target_columns found in labeling result")
                return {}
            
            # Extract direction-specific target columns
            if isinstance(raw_targets, dict):
                # Look for long/short specific columns
                for direction in ['long', 'short']:
                    direction_key = f"{direction}_target"
                    if direction_key in raw_targets:
                        target_columns[direction] = raw_targets[direction_key]
                    elif f"{direction}_returns" in raw_targets:
                        target_columns[direction] = raw_targets[f"{direction}_returns"]
                    elif f"{direction}_labels" in raw_targets:
                        target_columns[direction] = raw_targets[f"{direction}_labels"]
                
                # If no direction-specific columns found, look for generic ones
                if not target_columns:
                    for key, value in raw_targets.items():
                        if 'target' in key.lower() or 'return' in key.lower() or 'label' in key.lower():
                            # Default to long direction
                            target_columns['long'] = value
                            break
            
            tprint_debug(f"📊 Extracted target columns: {target_columns}")
            return target_columns
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract target columns: {e}")
            return {}
    
    def _generate_target_info(self, target_columns: Dict[str, str], labeling_result: Any) -> List[TargetColumnInfo]:
        """Generate target information for each target column."""
        target_info = []
        
        try:
            for direction, column_name in target_columns.items():
                # Determine direction enum
                direction_enum = TargetDirection.LONG if direction == 'long' else TargetDirection.SHORT
                
                # Extract metadata if available
                metadata = {}
                if hasattr(labeling_result, 'metadata'):
                    metadata = labeling_result.metadata
                elif isinstance(labeling_result, dict) and 'metadata' in labeling_result:
                    metadata = labeling_result['metadata']
                
                # Calculate confidence (simplified)
                confidence = 0.8  # Default confidence
                if isinstance(metadata, dict):
                    confidence = metadata.get('confidence', 0.8)
                
                # Determine horizon (simplified)
                horizon = 1  # Default horizon
                if 'horizon' in column_name.lower():
                    try:
                        # Try to extract horizon from column name
                        import re
                        horizon_match = re.search(r'(\d+)', column_name)
                        if horizon_match:
                            horizon = int(horizon_match.group(1))
                    except:
                        pass
                
                target_info.append(TargetColumnInfo(
                    column_name=column_name,
                    direction=direction_enum,
                    horizon=horizon,
                    confidence=confidence,
                    metadata=metadata
                ))
            
            return target_info
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate target info: {e}")
            return []
    
    def select_optimal_target_columns(
        self,
        data: pd.DataFrame,
        target_columns: Dict[str, str],
        direction: str = 'long'
    ) -> Optional[str]:
        """
        Select optimal target column for a specific direction.
        
        Args:
            data: Input data
            target_columns: Available target columns
            direction: Direction to select for
            
        Returns:
            Optimal target column name or None
        """
        try:
            if direction not in target_columns:
                tprint_warning(f"⚠️ No target column for direction {direction}")
                return None
            
            target_column = target_columns[direction]
            
            # Validate target column exists in data
            if target_column not in data.columns:
                tprint_error(f"❌ Target column {target_column} not found in data")
                return None
            
            # Check data quality
            target_series = data[target_column].dropna()
            if len(target_series) < 10:
                tprint_warning(f"⚠️ Target column {target_column} has insufficient data")
                return None
            
            # Calculate basic statistics
            mean_val = target_series.mean()
            std_val = target_series.std()
            
            if std_val == 0:
                tprint_warning(f"⚠️ Target column {target_column} has zero variance")
                return None
            
            tprint_success(f"✅ Selected optimal target column: {target_column} (direction: {direction})")
            tprint_debug(f"📊 Target stats: mean={mean_val:.4f}, std={std_val:.4f}")
            
            return target_column
            
        except Exception as e:
            tprint_error(f"❌ Failed to select optimal target column: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'total_execution_time': 0.0,
            'labeling_operations': 0,
            'target_selections': 0
        }


def create_multi_horizon_integration(config: Optional[Dict[str, Any]] = None) -> MultiHorizonIntegration:
    """Create a multi-horizon integration with default configuration."""
    return MultiHorizonIntegration(config)