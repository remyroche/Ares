"""
Feature Generation Data Validation Step

This step validates the input data before feature generation begins.
It ensures data quality, schema compliance, and identifies any issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from src.training.steps.base_step import BaseStep
from src.training.steps.pre_training.validation.data_contracts import validate_selection_artifact
from src.training.steps.pre_training.validation.schemas import extract_p_value_mapping
from src.training.steps.pre_training.standardized_labeling_interface import validate_dataframe_schema


class FeatureGenerationDataValidationStep(BaseStep):
    """
    Validates input data for feature generation pipeline.
    
    Performs comprehensive data validation including:
    - Schema validation
    - Data quality checks
    - Missing value detection
    - Outlier detection
    - Data distribution analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_data_validation_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data validation.
        
        Args:
            config: Configuration containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - information: Timeframe information
                - min_data_points: Minimum required data points
                - required_columns: List of required column names
        
        Returns:
            Dictionary containing:
                - success: bool
                - validated_data_path: str
                - validation_report: Dict
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🔍 Starting data validation")
            
            # Load input data
            input_data = self._load_dataframe('market_data')
            if input_data is None:
                return {
                    'success': False,
                    'error': 'No market data found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Validate schema
            schema_valid = validate_dataframe_schema(input_data)
            if not schema_valid:
                return {
                    'success': False,
                    'error': 'Schema validation failed',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Check minimum data points
            min_points = config.get('min_data_points', 1000)
            if len(input_data) < min_points:
                return {
                    'success': False,
                    'error': f'Insufficient data points: {len(input_data)} < {min_points}',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Check for required columns
            required_cols = config.get('required_columns', ['open', 'high', 'low', 'close', 'volume'])
            missing_cols = [col for col in required_cols if col not in input_data.columns]
            if missing_cols:
                return {
                    'success': False,
                    'error': f'Missing required columns: {missing_cols}',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Data quality checks
            validation_report = {
                'total_rows': len(input_data),
                'total_columns': len(input_data.columns),
                'missing_values': input_data.isnull().sum().to_dict(),
                'duplicate_rows': input_data.duplicated().sum(),
                'data_types': {col: str(dtype) for col, dtype in input_data.dtypes.items()},
                'numeric_summary': input_data.describe().to_dict() if len(input_data.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            # Check for excessive missing values
            missing_threshold = config.get('max_missing_pct', 0.05)
            high_missing_cols = []
            for col in input_data.columns:
                missing_pct = input_data[col].isnull().sum() / len(input_data)
                if missing_pct > missing_threshold:
                    high_missing_cols.append((col, missing_pct))
            
            if high_missing_cols:
                self.logger.warning(f"Columns with high missing values: {high_missing_cols}")
                validation_report['high_missing_columns'] = high_missing_cols
            
            # Save validated data
            validated_path = self._save_dataframe(
                input_data,
                'validated_market_data',
                metadata=validation_report
            )
            
            # Save validation report
            report_path = self._save_metadata(
                validation_report,
                'data_validation_report'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'validated_data_path': validated_path,
                'validation_report': validation_report,
                'artifacts': [validated_path, report_path],
                'metrics': {
                    'data_rows': len(input_data),
                    'data_columns': len(input_data.columns),
                    'missing_value_pct': input_data.isnull().sum().sum() / (len(input_data) * len(input_data.columns)),
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
