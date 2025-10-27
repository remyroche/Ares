"""
Analyst Side Information Handler

This module handles the emission and processing of Analyst side information
for CMI complementarity computation in the Tactician pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getChild('AnalystSideInfo')

@dataclass
class AnalystSideInfoResult:
    """Result of Analyst side information processing."""
    
    analyst_outputs: Optional[pd.DataFrame]
    feature_importance: Dict[str, float]
    regime_labels: Optional[pd.Series]
    metadata: Dict[str, Any]
    processing_time: float
    warnings: List[str]

class AnalystSideInfoHandler:
    """
    Handler for Analyst side information in CMI complementarity computation.
    
    This class manages the emission and processing of Analyst outputs
    to provide side information for Tactician feature selection.
    """
    
    def __init__(self):
        """Initialize the Analyst side information handler."""
        self.logger = logger.getChild('AnalystSideInfoHandler')
        
        # Processing history
        self.processing_history = []
        self.performance_stats = {
            'total_processing': 0,
            'avg_processing_time': 0.0,
            'successful_processing': 0,
            'failed_processing': 0
        }
        
        self.logger.info("✅ Analyst Side Information Handler initialized")
    
    def emit_analyst_side_info(
        self,
        pipeline_state: Dict[str, Any],
        targets: Optional[pd.Series] = None,
        data_index: Optional[pd.Index] = None
    ) -> AnalystSideInfoResult:
        """
        Emit Analyst side information for CMI complementarity computation.
        
        Args:
            pipeline_state: Current pipeline state
            targets: Target values (optional)
            data_index: Data index (optional)
            
        Returns:
            AnalystSideInfoResult with processed side information
        """
        start_time = datetime.now()
        warnings_list = []
        
        self.logger.info("🔄 Emitting Analyst side information")
        
        try:
            # Extract Analyst outputs from pipeline state
            analyst_outputs = self._extract_analyst_outputs(pipeline_state)
            
            # Extract feature importance
            feature_importance = self._extract_feature_importance(pipeline_state)
            
            # Extract regime labels
            regime_labels = self._extract_regime_labels(pipeline_state)
            
            # Process metadata
            metadata = self._process_metadata(pipeline_state, targets, data_index)
            
            # Compute processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AnalystSideInfoResult(
                analyst_outputs=analyst_outputs,
                feature_importance=feature_importance,
                regime_labels=regime_labels,
                metadata=metadata,
                processing_time=processing_time,
                warnings=warnings_list
            )
            
            # Update performance stats
            self._update_performance_stats(processing_time, success=True)
            
            self.logger.info(f"✅ Analyst side information emitted in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to emit Analyst side information: {e}")
            warnings_list.append(f"Emission failed: {str(e)}")
            
            # Return empty result on failure
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(processing_time, success=False)
            
            return AnalystSideInfoResult(
                analyst_outputs=None,
                feature_importance={},
                regime_labels=None,
                metadata={},
                processing_time=processing_time,
                warnings=warnings_list
            )
    
    def _extract_analyst_outputs(self, pipeline_state: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract Analyst outputs from pipeline state."""
        try:
            # Look for Analyst outputs in various possible locations
            analyst_outputs = None
            
            # Check for direct Analyst outputs
            if 'analyst_outputs' in pipeline_state:
                analyst_outputs = pipeline_state['analyst_outputs']
            
            # Check for Analyst predictions
            elif 'analyst_predictions' in pipeline_state:
                analyst_outputs = pipeline_state['analyst_predictions']
            
            # Check for Analyst model outputs
            elif 'analyst_model_outputs' in pipeline_state:
                analyst_outputs = pipeline_state['analyst_model_outputs']
            
            # Check for Analyst features
            elif 'analyst_features' in pipeline_state:
                analyst_outputs = pipeline_state['analyst_features']
            
            # Convert to DataFrame if needed
            if analyst_outputs is not None:
                if isinstance(analyst_outputs, np.ndarray):
                    analyst_outputs = pd.DataFrame(analyst_outputs)
                elif isinstance(analyst_outputs, pd.Series):
                    analyst_outputs = pd.DataFrame(analyst_outputs)
                elif not isinstance(analyst_outputs, pd.DataFrame):
                    analyst_outputs = pd.DataFrame([analyst_outputs])
            
            return analyst_outputs
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract Analyst outputs: {e}")
            return None
    
    def _extract_feature_importance(self, pipeline_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract feature importance from pipeline state."""
        try:
            feature_importance = {}
            
            # Check for feature importance in various locations
            if 'feature_importance' in pipeline_state:
                feature_importance = pipeline_state['feature_importance']
            elif 'feature_scores' in pipeline_state:
                feature_importance = pipeline_state['feature_scores']
            elif 'analyst_feature_importance' in pipeline_state:
                feature_importance = pipeline_state['analyst_feature_importance']
            
            # Ensure it's a dictionary
            if not isinstance(feature_importance, dict):
                if isinstance(feature_importance, (list, np.ndarray)):
                    # Convert list/array to dict with generic names
                    feature_importance = {
                        f'feature_{i}': float(score) 
                        for i, score in enumerate(feature_importance)
                    }
                else:
                    feature_importance = {}
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract feature importance: {e}")
            return {}
    
    def _extract_regime_labels(self, pipeline_state: Dict[str, Any]) -> Optional[pd.Series]:
        """Extract regime labels from pipeline state."""
        try:
            regime_labels = None
            
            # Check for regime labels in various locations
            if 'regime_labels' in pipeline_state:
                regime_labels = pipeline_state['regime_labels']
            elif 'regime_clusters' in pipeline_state:
                regime_labels = pipeline_state['regime_clusters']
            elif 'cluster_labels' in pipeline_state:
                regime_labels = pipeline_state['cluster_labels']
            
            # Convert to Series if needed
            if regime_labels is not None:
                if isinstance(regime_labels, np.ndarray):
                    regime_labels = pd.Series(regime_labels)
                elif not isinstance(regime_labels, pd.Series):
                    regime_labels = pd.Series([regime_labels])
            
            return regime_labels
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract regime labels: {e}")
            return None
    
    def _process_metadata(
        self,
        pipeline_state: Dict[str, Any],
        targets: Optional[pd.Series],
        data_index: Optional[pd.Index]
    ) -> Dict[str, Any]:
        """Process metadata for side information."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_stage': pipeline_state.get('stage', 'unknown'),
            'data_shape': None,
            'target_info': None,
            'index_info': None
        }
        
        try:
            # Data shape information
            if 'data' in pipeline_state:
                data = pipeline_state['data']
                if hasattr(data, 'shape'):
                    metadata['data_shape'] = data.shape
                elif hasattr(data, '__len__'):
                    metadata['data_shape'] = (len(data),)
            
            # Target information
            if targets is not None:
                metadata['target_info'] = {
                    'length': len(targets),
                    'dtype': str(targets.dtype),
                    'min': float(targets.min()) if hasattr(targets, 'min') else None,
                    'max': float(targets.max()) if hasattr(targets, 'max') else None,
                    'mean': float(targets.mean()) if hasattr(targets, 'mean') else None
                }
            
            # Index information
            if data_index is not None:
                metadata['index_info'] = {
                    'length': len(data_index),
                    'dtype': str(data_index.dtype),
                    'start': str(data_index[0]) if len(data_index) > 0 else None,
                    'end': str(data_index[-1]) if len(data_index) > 0 else None
                }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to process metadata: {e}")
        
        return metadata
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics."""
        self.performance_stats['total_processing'] += 1
        
        if success:
            self.performance_stats['successful_processing'] += 1
        else:
            self.performance_stats['failed_processing'] += 1
        
        # Update average processing time
        total = self.performance_stats['total_processing']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_processing': self.performance_stats['total_processing'],
            'successful_processing': self.performance_stats['successful_processing'],
            'failed_processing': self.performance_stats['failed_processing'],
            'success_rate': (
                self.performance_stats['successful_processing'] / 
                max(1, self.performance_stats['total_processing'])
            ),
            'avg_processing_time': self.performance_stats['avg_processing_time']
        }
    
    def save_side_info(self, result: AnalystSideInfoResult, filepath: str):
        """Save side information result to file."""
        try:
            save_data = {
                'analyst_outputs': result.analyst_outputs.to_dict() if result.analyst_outputs is not None else None,
                'feature_importance': result.feature_importance,
                'regime_labels': result.regime_labels.to_dict() if result.regime_labels is not None else None,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'warnings': result.warnings
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Side information saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save side information: {e}")
    
    def load_side_info(self, filepath: str) -> AnalystSideInfoResult:
        """Load side information result from file."""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Reconstruct result
            result = AnalystSideInfoResult(
                analyst_outputs=pd.DataFrame(save_data['analyst_outputs']) if save_data['analyst_outputs'] else None,
                feature_importance=save_data['feature_importance'],
                regime_labels=pd.Series(save_data['regime_labels']) if save_data['regime_labels'] else None,
                metadata=save_data['metadata'],
                processing_time=save_data['processing_time'],
                warnings=save_data['warnings']
            )
            
            self.logger.info(f"✅ Side information loaded from {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load side information: {e}")
            return AnalystSideInfoResult(
                analyst_outputs=None,
                feature_importance={},
                regime_labels=None,
                metadata={},
                processing_time=0.0,
                warnings=[f"Load failed: {str(e)}"]
            )

# Convenience functions
def create_analyst_side_info_handler() -> AnalystSideInfoHandler:
    """Create Analyst side information handler instance."""
    return AnalystSideInfoHandler()

def emit_analyst_side_info(
    pipeline_state: Dict[str, Any],
    targets: Optional[pd.Series] = None,
    data_index: Optional[pd.Index] = None
) -> AnalystSideInfoResult:
    """Emit Analyst side information."""
    handler = create_analyst_side_info_handler()
    return handler.emit_analyst_side_info(pipeline_state, targets, data_index)
