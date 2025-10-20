"""
Feature Generation Labeling Integration Step

This step integrates labeling into the feature generation pipeline.
It ensures features are aligned with labels and handles label-related preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from src.training.steps.base_step import BaseStep
from src.training.steps.pre_training.standardized_labeling_interface import (
    assert_labels_sigma_scaled,
    validate_dataframe_schema
)


class FeatureGenerationLabelingIntegrationStep(BaseStep):
    """
    Integrates labeling with feature generation.
    
    Responsibilities:
    - Load and validate labels
    - Align labels with features
    - Handle label preprocessing
    - Generate label statistics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the labeling integration step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_labeling_integration_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute labeling integration.
        
        Args:
            config: Configuration containing:
                - label_type: Type of labels to use
                - label_column: Name of label column
                - normalize_labels: Whether to normalize labels
        
        Returns:
            Dictionary containing:
                - success: bool
                - labeled_data_path: str
                - label_statistics: Dict
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🏷️ Starting labeling integration")
            
            # Load validated data
            validated_data = self._load_dataframe('validated_market_data')
            if validated_data is None:
                return {
                    'success': False,
                    'error': 'No validated market data found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Load labels (from labeling step or artifact)
            label_column = config.get('label_column', 'label')
            labels = None
            
            # Try to load from labeling artifact
            try:
                labels_data = self._load_dataframe('profit_labels')
                if labels_data is not None and label_column in labels_data.columns:
                    labels = labels_data[label_column]
            except:
                self.logger.warning("Could not load profit labels artifact")
            
            # If labels are already in the data, use them
            if labels is None and label_column in validated_data.columns:
                labels = validated_data[label_column]
            
            if labels is None:
                return {
                    'success': False,
                    'error': f'No labels found. Expected column: {label_column}',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Align labels with data
            if len(labels) != len(validated_data):
                self.logger.warning(f"Label length mismatch: {len(labels)} vs {len(validated_data)}")
                # Align by index
                common_index = validated_data.index.intersection(labels.index)
                validated_data = validated_data.loc[common_index]
                labels = labels.loc[common_index]
            
            # Add labels to data
            labeled_data = validated_data.copy()
            labeled_data[label_column] = labels
            
            # Normalize labels if requested
            if config.get('normalize_labels', True):
                # Check if already normalized
                try:
                    assert_labels_sigma_scaled(labeled_data, label_column)
                    self.logger.info("Labels are already sigma-scaled")
                except:
                    # Normalize labels to sigma-scaled
                    mean_val = labels.mean()
                    std_val = labels.std()
                    if std_val > 0:
                        labeled_data[label_column] = (labels - mean_val) / std_val
                        self.logger.info(f"Normalized labels: mean={mean_val:.4f}, std={std_val:.4f}")
            
            # Generate label statistics
            label_stats = {
                'count': len(labels),
                'mean': float(labels.mean()),
                'std': float(labels.std()),
                'min': float(labels.min()),
                'max': float(labels.max()),
                'median': float(labels.median()),
                'q25': float(labels.quantile(0.25)),
                'q75': float(labels.quantile(0.75)),
                'missing': int(labels.isnull().sum()),
                'unique_values': int(labels.nunique())
            }
            
            # Save labeled data
            labeled_path = self._save_dataframe(
                labeled_data,
                'labeled_data',
                metadata=label_stats
            )
            
            # Save label statistics
            stats_path = self._save_metadata(
                label_stats,
                'label_statistics'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'labeled_data_path': labeled_path,
                'label_statistics': label_stats,
                'artifacts': [labeled_path, stats_path],
                'metrics': {
                    'label_count': len(labels),
                    'label_mean': label_stats['mean'],
                    'label_std': label_stats['std'],
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Labeling integration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
