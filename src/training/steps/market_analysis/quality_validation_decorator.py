"""
Quality Validation Decorator for Market Analysis Steps

This module provides a comprehensive quality validation decorator that replaces
the basic validation decorators with proper data quality tools from src/utils/data/quality/.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd

from src.utils.logger import system_logger

logger = system_logger.getChild('QualityValidationDecorator')

def validate_data_quality(threshold: float = 0.8, 
                         context: str = "general",
                         step_name: Optional[str] = None,
                         data_type: str = "klines"):
    """
    Comprehensive data quality validation decorator using proper quality tools.
    
    Args:
        threshold: Minimum quality score threshold (0.0-1.0)
        context: Validation context (e.g., 'market_analysis', 'feature_engineering')
        step_name: Name of the pipeline step
        data_type: Type of data ('klines', 'aggtrades', 'futures')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract DataFrame from arguments
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            
            # If no DataFrame in args, check kwargs
            if df is None:
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break
            
            # Perform quality validation if DataFrame found
            if df is not None:
                try:
                    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
                    from src.utils.data.quality.data_quality import DataQualityFramework
                    from src.utils.data.quality.data_cleaning import DataCleaner
                    
                    # Initialize quality tools
                    quality_scorer = get_quality_scorer()
                    quality_framework = DataQualityFramework()
                    data_cleaner = DataCleaner(data_type=data_type)
                    
                    # Perform comprehensive quality assessment
                    quality_assessment = quality_scorer.assess_data_quality(
                        df,
                        context=context,
                        step_name=step_name or func.__name__,
                        data_type=data_type
                    )
                    
                    # Check if quality meets threshold
                    quality_score = quality_assessment.overall_score / 100.0  # Convert to 0-1 scale
                    
                    if quality_score < threshold:
                        logger.warning(f"⚠️ Data quality below threshold: {quality_score:.3f} < {threshold}")
                        logger.warning(f"   Issues: {quality_assessment.issues}")
                        
                        # Attempt data cleaning for poor quality data
                        if quality_assessment.level.value == 'poor':
                            logger.info("🔧 Attempting data cleaning to improve quality...")
                            cleaned_df = data_cleaner.clean_dataframe(df)
                            
                            if cleaned_df is not None and not cleaned_df.empty:
                                # Re-assess quality after cleaning
                                cleaned_assessment = quality_scorer.assess_data_quality(
                                    cleaned_df,
                                    context=context,
                                    step_name=f"{step_name or func.__name__}_cleaned",
                                    data_type=data_type
                                )
                                
                                if cleaned_assessment.overall_score > quality_assessment.overall_score:
                                    logger.info(f"✅ Data cleaning improved quality: {cleaned_assessment.overall_score:.2f}")
                                    # Replace DataFrame in arguments
                                    for i, arg in enumerate(args):
                                        if isinstance(arg, pd.DataFrame):
                                            args = list(args)
                                            args[i] = cleaned_df
                                            args = tuple(args)
                                            break
                                    else:
                                        for key, value in kwargs.items():
                                            if isinstance(value, pd.DataFrame):
                                                kwargs[key] = cleaned_df
                                                break
                                else:
                                    logger.warning("⚠️ Data cleaning did not improve quality")
                    
                    # Log quality assessment results
                    logger.info(f"📊 Data quality assessment: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})")
                    
                except ImportError as e:
                    logger.warning(f"⚠️ Comprehensive quality tools not available, using fallback: {e}")
                    # Fallback to basic validation
                    _fallback_quality_check(df, threshold, func.__name__)
                except Exception as e:
                    logger.error(f"❌ Error in quality validation: {e}")
                    # Continue with original function if validation fails
            
            # Execute the original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def validate_feature_engineering_with_lookahead_bias_detection(
    threshold: float = 0.8,
    context: str = "feature_engineering",
    step_name: Optional[str] = None
):
    """
    Enhanced feature engineering validation with lookahead bias detection.
    
    Args:
        threshold: Minimum quality score threshold
        context: Validation context
        step_name: Name of the pipeline step
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract DataFrame from arguments
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            
            if df is None:
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break
            
            # Perform feature engineering specific validation
            if df is not None:
                try:
                    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
                    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
                    
                    # Initialize quality tools
                    advanced_metrics = AdvancedQualityMetrics()
                    quality_scorer = get_quality_scorer()
                    
                    # Perform comprehensive quality assessment
                    quality_assessment = quality_scorer.assess_data_quality(
                        df,
                        context=context,
                        step_name=step_name or func.__name__,
                        data_type="features"
                    )
                    
                    # Check for lookahead bias indicators
                    lookahead_issues = []
                    for metric in quality_assessment.metrics:
                        if 'temporal' in metric.name.lower() or 'future' in metric.name.lower():
                            if metric.severity in ['error', 'critical']:
                                lookahead_issues.append(metric.message)
                    
                    if lookahead_issues:
                        logger.warning(f"⚠️ Potential lookahead bias detected: {lookahead_issues}")
                    
                    # Check quality threshold
                    quality_score = quality_assessment.overall_score / 100.0
                    if quality_score < threshold:
                        logger.warning(f"⚠️ Feature quality below threshold: {quality_score:.3f} < {threshold}")
                    
                    logger.info(f"📊 Feature engineering quality: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})")
                    
                except ImportError as e:
                    logger.warning(f"⚠️ Advanced quality tools not available: {e}")
                    # Fallback to basic validation
                    _fallback_quality_check(df, threshold, func.__name__)
                except Exception as e:
                    logger.error(f"❌ Error in feature engineering validation: {e}")
            
            # Execute the original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def _fallback_quality_check(df: pd.DataFrame, threshold: float, func_name: str) -> None:
    """Fallback quality check using basic validation."""
    try:
        # Basic quality checks
        if df.empty:
            logger.error(f"❌ DataFrame is empty in {func_name}")
            return
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] > 0 and df.shape[1] > 0 else 0
        if missing_ratio > 0.1:
            logger.warning(f"⚠️ High missing value ratio: {missing_ratio:.2%}")
        
        # Check for duplicates
        duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        if duplicate_ratio > 0.05:
            logger.warning(f"⚠️ High duplicate ratio: {duplicate_ratio:.2%}")
        
        # Simple quality score calculation
        quality_score = 1.0 - (missing_ratio + duplicate_ratio)
        if quality_score < threshold:
            logger.warning(f"⚠️ Basic quality check failed: {quality_score:.3f} < {threshold}")
        
        logger.info(f"📊 Basic quality check: {quality_score:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Error in fallback quality check: {e}")

# Convenience decorators for common use cases
def validate_market_data_quality(func: Callable) -> Callable:
    """Decorator for market data quality validation."""
    return validate_data_quality(
        threshold=0.8,
        context="market_analysis",
        data_type="klines"
    )(func)

def validate_feature_data_quality(func: Callable) -> Callable:
    """Decorator for feature data quality validation."""
    return validate_data_quality(
        threshold=0.7,
        context="feature_engineering",
        data_type="features"
    )(func)

def validate_label_data_quality(func: Callable) -> Callable:
    """Decorator for label data quality validation."""
    return validate_data_quality(
        threshold=0.9,
        context="labeling",
        data_type="labels"
    )(func)