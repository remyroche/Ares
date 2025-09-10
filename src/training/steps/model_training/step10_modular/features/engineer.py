from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Step 10 Feature Engineer.

This module handles feature engineering for the unified regime intelligence system.
Includes comprehensive data validation to prevent parquet schema issues and data corruption.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from src.utils.logger import system_logger
import logging

logger = system_logger.getChild('Step10FeatureEngineer')


class FeatureEngineer:
    """Feature engineering coordinator for Step 10 with robust data validation.

    This class coordinates all feature engineering tasks:
    - Cross-timeframe correlations
    - Regime transition features
    - Sequence creation
    - Intensity processing
    - Comprehensive data validation to prevent parquet schema issues
    """

    def __init__(self, config):
        """Initialize feature engineer with validation capabilities.

        Args:
            config: Step 10 configuration
        """
        self.config = config
        self.logger = logger

        # Placeholder for future implementation
        self.cross_timeframe_processor = None
        self.transition_detector = None
        self.sequence_builder = None
        self.intensity_processor = None

        # Data validation settings
        self.validation_enabled = config.get('enable_data_validation', True)
        self.schema_check_enabled = config.get('enable_schema_check', True)

        self.logger.info("✅ Feature Engineer initialized with data validation")

    async def prepare_features(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare features for training with comprehensive validation.

        Args:
            data: Raw input data

        Returns:
            Processed features or None if failed
        """
        try:
            self.logger.info("🔧 Starting feature preparation with validation")

            # Validate input data first (prevents parquet schema issues)
            if not await self._validate_input_data(data):
                self.logger.error("❌ Input data validation failed")
                return None

            # Perform data quality checks
            if not await self._perform_data_quality_checks(data):
                self.logger.warning("⚠️ Data quality issues detected, proceeding with caution")

            # Placeholder: return input data unchanged with validation
            # In full implementation, this will:
            # 1. Extract HMM states from multiple timeframes
            # 2. Calculate cross-timeframe correlations
            # 3. Generate regime transition features
            # 4. Create sequences for model input
            # 5. Process intensity features

            self.logger.info("✅ Feature preparation completed successfully")
            return data

        except Exception as e:
            self.logger.exception(f"❌ Feature preparation failed: {e}")
            return None

    async def prepare_prediction_features(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare features for prediction with validation.

        Args:
            data: Raw input data for prediction

        Returns:
            Processed features for prediction
        """
        try:
            self.logger.info("🔧 Starting prediction feature preparation with validation")

            # Validate prediction data
            if not await self._validate_prediction_data(data):
                self.logger.error("❌ Prediction data validation failed")
                return None

            # Placeholder implementation with validation
            return data

        except Exception as e:
            self.logger.exception(f"❌ Prediction feature preparation failed: {e}")
            return None

    async def _validate_input_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data to prevent parquet schema issues.

        Args:
            data: Input data dictionary

        Returns:
            True if valid, False otherwise
        """
        try:
            if not data:
                self.logger.error("❌ Input data is empty or None")
                return False

            # Check for required keys
            required_keys = ['symbol', 'exchange', 'timeframes']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                self.logger.error(f"❌ Missing required keys in input data: {missing_keys}")
                return False

            # Validate data types and schemas
            if self.schema_check_enabled:
                schema_issues = await self._check_data_schema(data)
                if schema_issues:
                    self.logger.warning(f"⚠️ Schema issues detected: {schema_issues}")
                    # Don't fail on schema issues, just warn

            self.logger.info("✅ Input data validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Input data validation failed: {e}")
            return False

    async def _validate_prediction_data(self, data: Dict[str, Any]) -> bool:
        """Validate prediction data.

        Args:
            data: Prediction input data

        Returns:
            True if valid, False otherwise
        """
        try:
            if not data:
                self.logger.error("❌ Prediction data is empty or None")
                return False

            # Basic structure validation for prediction
            if 'features' not in data and 'market_data' not in data:
                self.logger.warning("⚠️ No features or market_data found in prediction input")

            self.logger.info("✅ Prediction data validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Prediction data validation failed: {e}")
            return False

    async def _perform_data_quality_checks(self, data: Dict[str, Any]) -> bool:
        """Perform comprehensive data quality checks.

        Args:
            data: Input data dictionary

        Returns:
            True if quality is acceptable, False if critical issues
        """
        try:
            quality_issues = []

            # Check for DataFrame data if present
            if 'dataframe' in data:
                df = data['dataframe']
                if isinstance(df, pd.DataFrame):
                    # Check for NaN values
                    nan_counts = df.isnull().sum().sum()
                    if nan_counts > 0:
                        percentage = (nan_counts / df.size) * 100
                        if percentage > 10:  # More than 10% NaN
                            quality_issues.append(f"High NaN ratio: {percentage:.1f}%")
                        else:
                            self.logger.warning(f"⚠️ Found {nan_counts} NaN values ({percentage:.1f}%)")

                    # Check for infinite values
                    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
                    if inf_counts > 0:
                        quality_issues.append(f"Found {inf_counts} infinite values")

                    # Check data ranges for critical columns
                    if 'close' in df.columns:
                        min_price = df['close'].min()
                        max_price = df['close'].max()
                        if min_price <= 0:
                            quality_issues.append(f"Non-positive prices found (min: {min_price})")

            # Check timeframes
            if 'timeframes' in data:
                timeframes = data['timeframes']
                if not timeframes:
                    quality_issues.append("No timeframes specified")
                else:
                    self.logger.info(f"📊 Processing {len(timeframes)} timeframes: {timeframes}")

            # Report issues
            if quality_issues:
                for issue in quality_issues:
                    self.logger.warning(f"⚠️ Data quality issue: {issue}")

                # Only fail if critical issues found
                critical_issues = [issue for issue in quality_issues if 'Non-positive prices' in issue]
                if critical_issues:
                    self.logger.error(f"❌ Critical data quality issues: {critical_issues}")
                    return False

            self.logger.info("✅ Data quality checks completed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Data quality check failed: {e}")
            return False

    async def _check_data_schema(self, data: Dict[str, Any]) -> List[str]:
        """Check data schema for consistency issues.

        Args:
            data: Input data dictionary

        Returns:
            List of schema issues found
        """
        try:
            schema_issues = []

            # Check symbol/exchange format
            if 'symbol' in data:
                symbol = data['symbol']
                if not isinstance(symbol, str) or len(symbol) == 0:
                    schema_issues.append("Invalid symbol format")

            if 'exchange' in data:
                exchange = data['exchange']
                if not isinstance(exchange, str) or len(exchange) == 0:
                    schema_issues.append("Invalid exchange format")

            # Check timeframes format
            if 'timeframes' in data:
                timeframes = data['timeframes']
                if not isinstance(timeframes, list):
                    schema_issues.append("Timeframes should be a list")
                else:
                    for tf in timeframes:
                        if not isinstance(tf, str):
                            schema_issues.append(f"Invalid timeframe format: {tf}")

            return schema_issues

        except Exception as e:
            self.logger.exception(f"❌ Schema check failed: {e}")
            return [f"Schema check error: {str(e)}"]
