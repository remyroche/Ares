"""
Feature Generation Stage for Unified Data-Driven Pipeline

This module handles feature generation and engineering for the unified pipeline.
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
from ..enhanced_components.common_feature_logic import (
    CommonFeatureGenerator, FeatureGenerationConfig, 
    create_common_feature_generator
)
from ..enhanced_components.enhanced_feature_generator import (
    EnhancedFeatureGenerator, FeatureGenerationConfig as EnhancedFeatureGenerationConfig
)
from ..enhanced_components.lightgbm_featuretools_generator import (
    LightGBMFeatureToolsGenerator, LightGBMFeatureToolsConfig
)
from ..enhanced_components.detailed_pipeline_reporter import DetailedPipelineReporter


@dataclass
class FeatureGenerationResult:
    """Result from feature generation stage."""
    
    generated_features: pd.DataFrame
    feature_metadata: Dict[str, Any]
    generation_time: float
    memory_usage: float
    feature_count: int
    quality_score: float
    warnings: List[str]
    errors: List[str]


class FeatureGenerationStage:
    """Feature generation stage for the unified pipeline."""
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        """Initialize the feature generation stage.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize feature generation components
        self.common_generator = create_common_feature_generator()
        self.enhanced_generator = EnhancedFeatureGenerator()
        self.lightgbm_generator = LightGBMFeatureToolsGenerator()
        self.detailed_reporter = DetailedPipelineReporter(outcomes_dir="outcomes")
        
        tprint_info("🔧 Feature generation stage initialized")
    
    def generate_features(self, 
                         data: pd.DataFrame, 
                         targets: Optional[pd.Series] = None,
                         feature_columns: Optional[List[str]] = None,
                         timeframe: str = "15m") -> FeatureGenerationResult:
        """Generate features using multiple generation methods.
        
        Args:
            data: Input DataFrame with OHLCV data
            targets: Optional target series for supervised learning
            feature_columns: Optional list of feature columns to use
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            FeatureGenerationResult with generated features
        """
        start_time = time.time()
        warnings = []
        errors = []
        
        try:
            tprint_info(f"🔧 Generating features for timeframe: {timeframe}")
            
            # Start detailed reporting
            self.detailed_reporter.start_step("feature_generation", len(data.columns))
            
            # Step 1: Generate common features
            tprint_info("Step 1: Generating common features")
            common_features = self._generate_common_features(data, timeframe)
            
            # Step 2: Generate enhanced features
            tprint_info("Step 2: Generating enhanced features")
            enhanced_features = self._generate_enhanced_features(data, targets, timeframe)
            
            # Step 3: Generate LightGBM + FeatureTools features
            tprint_info("Step 3: Generating LightGBM + FeatureTools features")
            lightgbm_features = self._generate_lightgbm_features(data, targets, timeframe)
            
            # Step 4: Combine all features
            tprint_info("Step 4: Combining all features")
            combined_features = self._combine_features(
                data, common_features, enhanced_features, lightgbm_features
            )
            
            # Step 5: Apply feature transformations
            tprint_info("Step 5: Applying feature transformations")
            transformed_features = self._apply_transformations(combined_features, timeframe)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            memory_usage = transformed_features.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            feature_count = len(transformed_features.columns)
            quality_score = self._calculate_quality_score(transformed_features)
            
            # Create result
            result = FeatureGenerationResult(
                generated_features=transformed_features,
                feature_metadata={
                    'common_features_count': len(common_features.columns) if common_features is not None else 0,
                    'enhanced_features_count': len(enhanced_features.columns) if enhanced_features is not None else 0,
                    'lightgbm_features_count': len(lightgbm_features.columns) if lightgbm_features is not None else 0,
                    'total_features': feature_count,
                    'timeframe': timeframe,
                    'generation_methods': ['common', 'enhanced', 'lightgbm']
                },
                generation_time=generation_time,
                memory_usage=memory_usage,
                feature_count=feature_count,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
            # End detailed reporting
            self.detailed_reporter.end_step(
                "feature_generation", 
                feature_count,
                generation_time,
                memory_usage,
                True
            )
            
            tprint_success(f"✅ Generated {feature_count} features in {generation_time:.2f}s")
            tprint_info(f"📊 Memory usage: {memory_usage:.2f} MB")
            tprint_info(f"📈 Quality score: {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Feature generation failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return FeatureGenerationResult(
                generated_features=pd.DataFrame(),
                feature_metadata={},
                generation_time=generation_time,
                memory_usage=0.0,
                feature_count=0,
                quality_score=0.0,
                warnings=warnings,
                errors=[error_msg]
            )
    
    def _generate_common_features(self, 
                                 data: pd.DataFrame, 
                                 timeframe: str) -> Optional[pd.DataFrame]:
        """Generate common features using CommonFeatureGenerator.
        
        Args:
            data: Input DataFrame
            timeframe: Target timeframe
            
        Returns:
            DataFrame with common features or None if failed
        """
        try:
            return self.common_generator.generate_features(data, timeframe=timeframe)
        except Exception as e:
            tprint_warning(f"⚠️ Common feature generation failed: {e}")
            return None
    
    def _generate_enhanced_features(self, 
                                   data: pd.DataFrame, 
                                   targets: Optional[pd.Series],
                                   timeframe: str) -> Optional[pd.DataFrame]:
        """Generate enhanced features using EnhancedFeatureGenerator.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            timeframe: Target timeframe
            
        Returns:
            DataFrame with enhanced features or None if failed
        """
        try:
            return self.enhanced_generator.generate_features(
                data, targets=targets, timeframe=timeframe
            )
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced feature generation failed: {e}")
            return None
    
    def _generate_lightgbm_features(self, 
                                   data: pd.DataFrame, 
                                   targets: Optional[pd.Series],
                                   timeframe: str) -> Optional[pd.DataFrame]:
        """Generate LightGBM + FeatureTools features.
        
        Args:
            data: Input DataFrame
            targets: Optional target series
            timeframe: Target timeframe
            
        Returns:
            DataFrame with LightGBM features or None if failed
        """
        try:
            return self.lightgbm_generator.generate_features(
                data, targets=targets, timeframe=timeframe
            )
        except Exception as e:
            tprint_warning(f"⚠️ LightGBM feature generation failed: {e}")
            return None
    
    def _combine_features(self, 
                         original_data: pd.DataFrame,
                         common_features: Optional[pd.DataFrame],
                         enhanced_features: Optional[pd.DataFrame],
                         lightgbm_features: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Combine all generated features.
        
        Args:
            original_data: Original input data
            common_features: Common features DataFrame
            enhanced_features: Enhanced features DataFrame
            lightgbm_features: LightGBM features DataFrame
            
        Returns:
            Combined DataFrame with all features
        """
        try:
            # Start with original data
            combined = original_data.copy()
            
            # Add common features
            if common_features is not None and not common_features.empty:
                combined = pd.concat([combined, common_features], axis=1)
                tprint_debug(f"Added {len(common_features.columns)} common features")
            
            # Add enhanced features
            if enhanced_features is not None and not enhanced_features.empty:
                combined = pd.concat([combined, enhanced_features], axis=1)
                tprint_debug(f"Added {len(enhanced_features.columns)} enhanced features")
            
            # Add LightGBM features
            if lightgbm_features is not None and not lightgbm_features.empty:
                combined = pd.concat([combined, lightgbm_features], axis=1)
                tprint_debug(f"Added {len(lightgbm_features.columns)} LightGBM features")
            
            return combined
            
        except Exception as e:
            tprint_error(f"❌ Feature combination failed: {e}")
            return original_data.copy()
    
    def _apply_transformations(self, 
                              features: pd.DataFrame, 
                              timeframe: str) -> pd.DataFrame:
        """Apply feature transformations.
        
        Args:
            features: Features DataFrame
            timeframe: Target timeframe
            
        Returns:
            Transformed features DataFrame
        """
        try:
            # Apply basic transformations
            transformed = features.copy()
            
            # Remove infinite values
            transformed = transformed.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with forward fill then backward fill
            transformed = transformed.fillna(method='ffill').fillna(method='bfill')
            
            # Remove columns with all NaN values
            transformed = transformed.dropna(axis=1, how='all')
            
            return transformed
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature transformation failed: {e}")
            return features.copy()
    
    def _calculate_quality_score(self, features: pd.DataFrame) -> float:
        """Calculate quality score for generated features.
        
        Args:
            features: Features DataFrame
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            if features.empty:
                return 0.0
            
            # Calculate various quality metrics
            nan_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
            inf_ratio = np.isinf(features.select_dtypes(include=[np.number])).sum().sum() / (features.shape[0] * features.shape[1])
            constant_ratio = (features.nunique() == 1).sum() / len(features.columns)
            
            # Calculate quality score (higher is better)
            quality_score = 1.0 - (nan_ratio + inf_ratio + constant_ratio)
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            tprint_warning(f"⚠️ Quality score calculation failed: {e}")
            return 0.0
    
    def get_generation_summary(self, result: FeatureGenerationResult) -> Dict[str, Any]:
        """Get a summary of feature generation results.
        
        Args:
            result: FeatureGenerationResult to summarize
            
        Returns:
            Dictionary with generation summary
        """
        return {
            'feature_count': result.feature_count,
            'quality_score': result.quality_score,
            'generation_time': result.generation_time,
            'memory_usage_mb': result.memory_usage,
            'warnings_count': len(result.warnings),
            'errors_count': len(result.errors),
            'metadata': result.feature_metadata
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        tprint_debug("🧹 Cleaning up feature generation stage")
        # Add any cleanup logic here if needed


def create_feature_generation_stage(config: Any, logger: Optional[logging.Logger] = None) -> FeatureGenerationStage:
    """Create a feature generation stage instance.
    
    Args:
        config: Pipeline configuration
        logger: Optional logger instance
        
    Returns:
        FeatureGenerationStage instance
    """
    return FeatureGenerationStage(config, logger)