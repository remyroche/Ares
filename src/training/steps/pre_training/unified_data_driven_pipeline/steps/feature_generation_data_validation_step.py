"""
Feature Generation Data Validation Step

This step performs comprehensive data validation and quality assessment
as the first step in the unified data-driven pipeline.
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import UnifiedPipelineConfig
from src.training.steps.pre_training.unified_data_driven_pipeline.stages.data_validation_stage import (
    EnhancedStatisticalFramework,
    EnhancedSchemaValidator,
    AdvancedInputValidator
)


@dataclass
class DataValidationResult:
    """Result of data validation step."""
    
    success: bool
    data_quality_score: float
    validation_metadata: Dict[str, Any]
    quality_report: Dict[str, Any]
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None


class FeatureGenerationDataValidationStep:
    """Data validation step for feature generation pipeline."""
    
    def __init__(self, config: UnifiedPipelineConfig, logger: Optional[logging.Logger] = None):
        """Initialize the data validation step.
        
        Args:
            config: Unified pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize validation components
        self.statistical_framework = EnhancedStatisticalFramework()
        self.schema_validator = EnhancedSchemaValidator()
        self.input_validator = AdvancedInputValidator()
    
    async def execute(self, 
                     market_data: pd.DataFrame,
                     artifacts_dir: str,
                     **kwargs) -> DataValidationResult:
        """Execute data validation step.
        
        Args:
            market_data: Input market data
            artifacts_dir: Directory to save artifacts
            **kwargs: Additional arguments
            
        Returns:
            DataValidationResult with validation results
        """
        self.logger.info("🔍 Starting data validation step...")
        
        try:
            # Create artifacts directory
            artifacts_path = Path(artifacts_dir) / "feature_generation_data_validation_step"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Schema validation
            self.logger.info("📋 Performing schema validation...")
            schema_result = self.schema_validator.validate_schema(market_data)
            
            # 2. Statistical validation
            self.logger.info("📊 Performing statistical validation...")
            statistical_result = self.statistical_framework.analyze_data_quality(market_data)
            
            # 3. Input validation
            self.logger.info("✅ Performing input validation...")
            input_result = self.input_validator.validate_inputs(market_data)
            
            # 4. Calculate overall quality score
            quality_score = self._calculate_quality_score(schema_result, statistical_result, input_result)
            
            # 5. Generate quality report
            quality_report = self._generate_quality_report(
                schema_result, statistical_result, input_result, quality_score
            )
            
            # 6. Save artifacts
            artifacts = await self._save_artifacts(
                artifacts_path, schema_result, statistical_result, 
                input_result, quality_report, market_data
            )
            
            # 7. Create validation metadata
            validation_metadata = {
                "step_name": "feature_generation_data_validation_step",
                "timestamp": datetime.now().isoformat(),
                "data_shape": market_data.shape,
                "quality_score": quality_score,
                "schema_valid": schema_result.get("valid", False),
                "statistical_valid": statistical_result.get("valid", False),
                "input_valid": input_result.get("valid", False)
            }
            
            self.logger.info(f"✅ Data validation completed with quality score: {quality_score:.3f}")
            
            return DataValidationResult(
                success=True,
                data_quality_score=quality_score,
                validation_metadata=validation_metadata,
                quality_report=quality_report,
                artifacts=artifacts
            )
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return DataValidationResult(
                success=False,
                data_quality_score=0.0,
                validation_metadata={},
                quality_report={},
                error_message=str(e)
            )
    
    def _calculate_quality_score(self, 
                                schema_result: Dict[str, Any],
                                statistical_result: Dict[str, Any],
                                input_result: Dict[str, Any]) -> float:
        """Calculate overall data quality score.
        
        Args:
            schema_result: Schema validation results
            statistical_result: Statistical validation results
            input_result: Input validation results
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        scores = []
        
        # Schema validation score
        if schema_result.get("valid", False):
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Statistical validation score
        statistical_score = statistical_result.get("quality_score", 0.0)
        scores.append(statistical_score)
        
        # Input validation score
        if input_result.get("valid", False):
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_quality_report(self,
                                schema_result: Dict[str, Any],
                                statistical_result: Dict[str, Any],
                                input_result: Dict[str, Any],
                                quality_score: float) -> Dict[str, Any]:
        """Generate comprehensive quality report.
        
        Args:
            schema_result: Schema validation results
            statistical_result: Statistical validation results
            input_result: Input validation results
            quality_score: Overall quality score
            
        Returns:
            Quality report dictionary
        """
        return {
            "overall_quality_score": quality_score,
            "schema_validation": {
                "valid": schema_result.get("valid", False),
                "issues": schema_result.get("issues", []),
                "warnings": schema_result.get("warnings", [])
            },
            "statistical_validation": {
                "valid": statistical_result.get("valid", False),
                "quality_score": statistical_result.get("quality_score", 0.0),
                "missing_data_percentage": statistical_result.get("missing_data_percentage", 0.0),
                "outlier_percentage": statistical_result.get("outlier_percentage", 0.0),
                "correlation_issues": statistical_result.get("correlation_issues", [])
            },
            "input_validation": {
                "valid": input_result.get("valid", False),
                "issues": input_result.get("issues", []),
                "warnings": input_result.get("warnings", [])
            },
            "recommendations": self._generate_recommendations(
                schema_result, statistical_result, input_result, quality_score
            )
        }
    
    def _generate_recommendations(self,
                                 schema_result: Dict[str, Any],
                                 statistical_result: Dict[str, Any],
                                 input_result: Dict[str, Any],
                                 quality_score: float) -> List[str]:
        """Generate recommendations based on validation results.
        
        Args:
            schema_result: Schema validation results
            statistical_result: Statistical validation results
            input_result: Input validation results
            quality_score: Overall quality score
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Data quality is below recommended threshold. Consider data cleaning.")
        
        if statistical_result.get("missing_data_percentage", 0) > 0.05:
            recommendations.append("High missing data percentage detected. Consider imputation strategies.")
        
        if statistical_result.get("outlier_percentage", 0) > 0.1:
            recommendations.append("High outlier percentage detected. Consider outlier treatment.")
        
        if not schema_result.get("valid", False):
            recommendations.append("Schema validation failed. Check data structure and types.")
        
        if not input_result.get("valid", False):
            recommendations.append("Input validation failed. Check data format and requirements.")
        
        return recommendations
    
    async def _save_artifacts(self,
                             artifacts_path: Path,
                             schema_result: Dict[str, Any],
                             statistical_result: Dict[str, Any],
                             input_result: Dict[str, Any],
                             quality_report: Dict[str, Any],
                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """Save validation artifacts.
        
        Args:
            artifacts_path: Path to save artifacts
            schema_result: Schema validation results
            statistical_result: Statistical validation results
            input_result: Input validation results
            quality_report: Quality report
            market_data: Market data
            
        Returns:
            Dictionary of saved artifact paths
        """
        artifacts = {}
        
        # Save quality report
        quality_report_path = artifacts_path / "data_quality_report.json"
        with open(quality_report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        artifacts["data_quality_report"] = str(quality_report_path)
        
        # Save validation metadata
        validation_metadata = {
            "schema_validation": schema_result,
            "statistical_validation": statistical_result,
            "input_validation": input_result,
            "timestamp": datetime.now().isoformat()
        }
        
        validation_metadata_path = artifacts_path / "validation_metadata.json"
        with open(validation_metadata_path, 'w') as f:
            json.dump(validation_metadata, f, indent=2)
        artifacts["validation_metadata"] = str(validation_metadata_path)
        
        # Save data statistics
        data_stats = {
            "shape": market_data.shape,
            "columns": list(market_data.columns),
            "dtypes": market_data.dtypes.to_dict(),
            "memory_usage": market_data.memory_usage(deep=True).to_dict(),
            "describe": market_data.describe().to_dict()
        }
        
        data_stats_path = artifacts_path / "data_statistics.json"
        with open(data_stats_path, 'w') as f:
            json.dump(data_stats, f, indent=2, default=str)
        artifacts["data_statistics"] = str(data_stats_path)
        
        # Save detailed validation artifacts
        validation_artifacts_dir = artifacts_path / "validation_artifacts"
        validation_artifacts_dir.mkdir(exist_ok=True)
        
        # Save correlation matrix if available
        if "correlation_matrix" in statistical_result:
            corr_matrix = pd.DataFrame(statistical_result["correlation_matrix"])
            corr_matrix_path = validation_artifacts_dir / "correlation_matrix.parquet"
            corr_matrix.to_parquet(corr_matrix_path)
            artifacts["correlation_matrix"] = str(corr_matrix_path)
        
        return artifacts


# Command handler for ares_launcher integration
async def handle_feature_generation_data_validation_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DataValidationResult:
    """Handle feature_generation_data_validation_step command.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        direction: Direction type
        intensity: Pipeline intensity
        lookback_days: Lookback period in days
        start_date: Start date for data
        end_date: End date for data
        exchange: Exchange name
        custom_overrides: Custom configuration overrides
        **kwargs: Additional arguments
        
    Returns:
        DataValidationResult
    """
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.simplified_config import (
        SimplifiedConfig
    )
    
    # Create configuration
    simplified_config = SimplifiedConfig()
    simplified_config.set_intensity(intensity)
    
    if custom_overrides:
        simplified_config.apply_custom_overrides(custom_overrides)
    
    config = simplified_config.get_config()
    
    # Create step instance
    step = FeatureGenerationDataValidationStep(config)
    
    # Load market data (placeholder - would integrate with actual data loading)
    # This would typically load data based on symbol, timeframe, start_date, end_date
    market_data = pd.DataFrame()  # Placeholder
    
    # Execute step
    return await step.execute(market_data, "artifacts")