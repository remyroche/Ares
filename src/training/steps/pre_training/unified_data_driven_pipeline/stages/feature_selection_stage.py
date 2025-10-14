"""
Feature Selection Stage for Unified Data-Driven Pipeline

This module handles feature selection and optimization for the unified pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

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

# Import core components
from ..core.intelligent_feature_selector import (
    IntelligentFeatureSelector, FeatureSelectionConfig, 
    FeatureSelectionResult, create_intelligent_feature_selector
)

# Import enhanced components
from ..enhanced_components.advanced_feature_selection import (
    AdvancedFeatureSelector, FeatureSelectionConfig as AdvancedFeatureSelectionConfig
)
from ..enhanced_components.detailed_pipeline_reporter import DetailedPipelineReporter


@dataclass
class FeatureSelectionStageResult:
    """Result from feature selection stage."""
    
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_metadata: Dict[str, Any]
    selection_time: float
    memory_usage: float
    quality_score: float
    warnings: List[str]
    errors: List[str]


class FeatureSelectionStage:
    """Feature selection stage for the unified pipeline."""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """Initialize the feature selection stage.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize feature selection components
        self.intelligent_selector = create_intelligent_feature_selector()
        self.advanced_selector = AdvancedFeatureSelector()
        self.detailed_reporter = DetailedPipelineReporter(outcomes_dir="outcomes")
        
        tprint_info("🎯 Feature selection stage initialized")
    
    def select_features(self, 
                       data: pd.DataFrame, 
                       targets: Optional[pd.Series] = None,
                       feature_columns: Optional[List[str]] = None,
                       timeframe: str = "15m") -> FeatureSelectionStageResult:
        """Select features using multiple selection methods.
        
        Args:
            data: Input DataFrame with features
            targets: Optional target series for supervised learning
            feature_columns: Optional list of feature columns to consider
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            FeatureSelectionStageResult with selected features
        """
        start_time = time.time()
        warnings = []
        errors = []
        
        try:
            tprint_info(f"🎯 Selecting features for timeframe: {timeframe}")
            
            # Start detailed reporting
            self.detailed_reporter.start_step("feature_selection", len(data.columns))
            
            # Step 1: Intelligent pre-selection
            tprint_info("Step 1: Intelligent pre-selection")
            pre_selected_features = self._pre_select_features(data, targets, timeframe)
            
            # Step 2: Advanced feature selection
            tprint_info("Step 2: Advanced feature selection")
            final_selection = self._advanced_feature_selection(
                data, targets, pre_selected_features, timeframe
            )
            
            # Step 3: Calculate feature scores
            tprint_info("Step 3: Calculating feature scores")
            feature_scores = self._calculate_feature_scores(
                data, targets, final_selection
            )
            
            # Step 4: Validate selection
            tprint_info("Step 4: Validating selection")
            validated_selection = self._validate_selection(
                data, final_selection, feature_scores
            )
            
            # Calculate metrics
            selection_time = time.time() - start_time
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            quality_score = self._calculate_selection_quality(
                data, validated_selection, feature_scores
            )
            
            # Create result
            result = FeatureSelectionStageResult(
                selected_features=validated_selection,
                feature_scores=feature_scores,
                selection_metadata={
                    'total_features_considered': len(data.columns),
                    'pre_selected_count': len(pre_selected_features),
                    'final_selected_count': len(validated_selection),
                    'timeframe': timeframe,
                    'selection_methods': ['intelligent_pre_selection', 'advanced_selection']
                },
                selection_time=selection_time,
                memory_usage=memory_usage,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
            # End detailed reporting
            self.detailed_reporter.end_step(
                "feature_selection", 
                len(validated_selection),
                selection_time,
                memory_usage,
                True
            )
            
            tprint_success(f"✅ Selected {len(validated_selection)} features in {selection_time:.2f}s")
            tprint_info(f"📊 Memory usage: {memory_usage:.2f} MB")
            tprint_info(f"📈 Quality score: {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            selection_time = time.time() - start_time
            error_msg = f"Feature selection failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return FeatureSelectionStageResult(
                selected_features=[],
                feature_scores={},
                selection_metadata={},
                selection_time=selection_time,
                memory_usage=0.0,
                quality_score=0.0,
                warnings=warnings,
                errors=[error_msg]
            )
    
    def _pre_select_features(self, 
                           data: pd.DataFrame, 
                           targets: Optional[pd.Series],
                           timeframe: str) -> List[str]:
        """Pre-select features using intelligent selection.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            timeframe: Target timeframe
            
        Returns:
            List of pre-selected feature names
        """
        try:
            # Use intelligent feature selector for pre-selection
            selection_result = self.intelligent_selector.select_features(
                data, targets=targets, timeframe=timeframe
            )
            
            if selection_result and hasattr(selection_result, 'selected_features'):
                return selection_result.selected_features
            else:
                # Fallback to basic selection
                return self._basic_feature_selection(data, targets)
                
        except Exception as e:
            tprint_warning(f"⚠️ Intelligent pre-selection failed: {e}")
            return self._basic_feature_selection(data, targets)
    
    def _advanced_feature_selection(self, 
                                  data: pd.DataFrame, 
                                  targets: Optional[pd.Series],
                                  pre_selected: List[str],
                                  timeframe: str) -> List[str]:
        """Perform advanced feature selection.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            pre_selected: Pre-selected features
            timeframe: Target timeframe
            
        Returns:
            List of selected feature names
        """
        try:
            # Filter data to pre-selected features
            if pre_selected:
                filtered_data = data[pre_selected]
            else:
                filtered_data = data
            
            # Use advanced feature selector
            selection_result = self.advanced_selector.select_features(
                filtered_data, targets=targets, timeframe=timeframe
            )
            
            if selection_result and hasattr(selection_result, 'selected_features'):
                return selection_result.selected_features
            else:
                # Fallback to pre-selected features
                return pre_selected
                
        except Exception as e:
            tprint_warning(f"⚠️ Advanced feature selection failed: {e}")
            return pre_selected
    
    def _basic_feature_selection(self, 
                               data: pd.DataFrame, 
                               targets: Optional[pd.Series]) -> List[str]:
        """Basic feature selection as fallback.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            
        Returns:
            List of selected feature names
        """
        try:
            # Select numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove columns with too many NaN values
            nan_threshold = 0.5
            valid_cols = []
            for col in numeric_cols:
                nan_ratio = data[col].isnull().sum() / len(data)
                if nan_ratio < nan_threshold:
                    valid_cols.append(col)
            
            # Select top features by variance
            if valid_cols:
                variances = data[valid_cols].var().sort_values(ascending=False)
                top_features = variances.head(min(20, len(variances))).index.tolist()
                return top_features
            else:
                return numeric_cols[:10]  # Fallback to first 10 numeric columns
                
        except Exception as e:
            tprint_warning(f"⚠️ Basic feature selection failed: {e}")
            return data.columns.tolist()[:10]  # Ultimate fallback
    
    def _calculate_feature_scores(self, 
                                data: pd.DataFrame, 
                                targets: Optional[pd.Series],
                                selected_features: List[str]) -> Dict[str, float]:
        """Calculate scores for selected features.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            selected_features: List of selected feature names
            
        Returns:
            Dictionary of feature names to scores
        """
        try:
            scores = {}
            
            for feature in selected_features:
                if feature in data.columns:
                    # Calculate basic score based on variance and correlation
                    variance_score = data[feature].var()
                    
                    if targets is not None and not targets.empty:
                        # Calculate correlation with targets
                        correlation = data[feature].corr(targets)
                        correlation_score = abs(correlation) if not np.isnan(correlation) else 0.0
                    else:
                        correlation_score = 0.0
                    
                    # Combine scores
                    combined_score = variance_score * (1 + correlation_score)
                    scores[feature] = combined_score
                else:
                    scores[feature] = 0.0
            
            return scores
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature scoring failed: {e}")
            return {feature: 1.0 for feature in selected_features}
    
    def _validate_selection(self, 
                          data: pd.DataFrame, 
                          selected_features: List[str],
                          feature_scores: Dict[str, float]) -> List[str]:
        """Validate the feature selection.
        
        Args:
            data: Input DataFrame
            selected_features: List of selected features
            feature_scores: Dictionary of feature scores
            
        Returns:
            List of validated feature names
        """
        try:
            validated = []
            
            for feature in selected_features:
                if feature in data.columns:
                    # Check if feature has valid data
                    if not data[feature].isnull().all():
                        # Check if feature has sufficient variance
                        if data[feature].var() > 1e-8:
                            validated.append(feature)
                        else:
                            tprint_debug(f"Removed low variance feature: {feature}")
                    else:
                        tprint_debug(f"Removed all-NaN feature: {feature}")
                else:
                    tprint_debug(f"Removed missing feature: {feature}")
            
            return validated
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature validation failed: {e}")
            return selected_features
    
    def _calculate_selection_quality(self, 
                                   data: pd.DataFrame, 
                                   selected_features: List[str],
                                   feature_scores: Dict[str, float]) -> float:
        """Calculate quality score for the feature selection.
        
        Args:
            data: Input DataFrame
            selected_features: List of selected features
            feature_scores: Dictionary of feature scores
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            if not selected_features:
                return 0.0
            
            # Calculate various quality metrics
            feature_count_score = min(1.0, len(selected_features) / 20.0)  # Prefer around 20 features
            
            # Calculate score diversity
            if feature_scores:
                scores = list(feature_scores.values())
                score_std = np.std(scores) if len(scores) > 1 else 0.0
                diversity_score = min(1.0, score_std / np.mean(scores) if np.mean(scores) > 0 else 0.0)
            else:
                diversity_score = 0.0
            
            # Calculate data quality
            selected_data = data[selected_features]
            nan_ratio = selected_data.isnull().sum().sum() / (selected_data.shape[0] * selected_data.shape[1])
            data_quality_score = 1.0 - nan_ratio
            
            # Combine scores
            quality_score = (feature_count_score + diversity_score + data_quality_score) / 3.0
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            tprint_warning(f"⚠️ Selection quality calculation failed: {e}")
            return 0.0
    
    def get_selection_summary(self, result: FeatureSelectionStageResult) -> Dict[str, Any]:
        """Get a summary of feature selection results.
        
        Args:
            result: FeatureSelectionStageResult to summarize
            
        Returns:
            Dictionary with selection summary
        """
        return {
            'selected_count': len(result.selected_features),
            'quality_score': result.quality_score,
            'selection_time': result.selection_time,
            'memory_usage_mb': result.memory_usage,
            'warnings_count': len(result.warnings),
            'errors_count': len(result.errors),
            'metadata': result.selection_metadata
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        tprint_debug("🧹 Cleaning up feature selection stage")
        # Add any cleanup logic here if needed


def create_feature_selection_stage(config: Any, logger: Optional[logging.Logger] = None) -> FeatureSelectionStage:
    """Create a feature selection stage instance.
    
    Args:
        config: Pipeline configuration
        logger: Optional logger instance
        
    Returns:
        FeatureSelectionStage instance
    """
    return FeatureSelectionStage(config, logger)