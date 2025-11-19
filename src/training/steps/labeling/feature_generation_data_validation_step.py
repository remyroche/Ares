"""
Feature Generation Data Validation Step.

This step validates data quality before feature generation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class FeatureGenerationDataValidationStep(BaseStep):
    """
    Feature Generation Data Validation Step.

    Validates data quality and integrity before feature generation.
    """

    def __init__(self, step_name: str = "feature_generation_data_validation_step"):
        """Initialize the feature generation data validation step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('FeatureGenerationDataValidation')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature generation data validation.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"🔍 Starting data validation for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Load and validate data
            market_data = await self._load_market_data(config)
            if market_data is None or len(market_data) == 0:
                raise ValueError("No market data available for validation")

            # Perform validation checks
            validation_results = self._validate_data_quality(market_data, config)

            artifacts = {
                'data_validation_results': {
                    'validation_checks': validation_results,
                    'data_shape': market_data.shape if hasattr(market_data, 'shape') else None,
                    'data_columns': list(market_data.columns) if hasattr(market_data, 'columns') else None,
                    'validation_passed': all(result['passed'] for result in validation_results.values()),
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'execution_mode': config.get('execution_mode', 'light'),
                        'created_at': datetime.now().isoformat()
                    }
                }
            }

            metrics = {
                'validation_checks_performed': len(validation_results),
                'validation_passed': all(result['passed'] for result in validation_results.values()),
                'data_rows': len(market_data),
                'data_columns': len(market_data.columns) if hasattr(market_data, 'columns') else 0,
                'execution_mode': config.get('execution_mode', 'light'),
                'success': True
            }

            tprint(f"✅ Data validation completed: {metrics['validation_checks_performed']} checks, {'PASSED' if metrics['validation_passed'] else 'FAILED'}", "SUCCESS")
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"Data validation failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    async def _load_market_data(self, config: Dict[str, Any]) -> Any:
        """Load market data for validation."""
        try:
            from src.utils.data.klines_parquet import get_klines_manager

            klines_manager = get_klines_manager(data_dir=config.get('data_dir', 'historical_data'))

            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed"
            )

            return market_data

        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return None

    def _validate_data_quality(self, market_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive data quality validation."""
        validation_results = {}

        # Check for basic requirements
        validation_results['has_data'] = {
            'passed': market_data is not None and len(market_data) > 0,
            'message': f"Data has {len(market_data)} rows" if market_data is not None else "No data found"
        }

        # Check for required columns
        if hasattr(market_data, 'columns'):
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            validation_results['required_columns'] = {
                'passed': len(missing_columns) == 0,
                'message': f"Missing columns: {missing_columns}" if missing_columns else "All required columns present"
            }

            # Check for NaN values
            if len(market_data) > 0:
                nan_ratios = market_data[required_columns].isna().mean()
                high_nan_columns = nan_ratios[nan_ratios > 0.1].index.tolist()
                validation_results['nan_values'] = {
                    'passed': len(high_nan_columns) == 0,
                    'message': f"High NaN ratios in: {high_nan_columns}" if high_nan_columns else "Acceptable NaN ratios"
                }

                # Check for reasonable price values
                price_checks = []
                for col in ['open', 'high', 'low', 'close']:
                    if col in market_data.columns:
                        invalid_prices = (market_data[col] <= 0).sum()
                        price_checks.append(invalid_prices == 0)

                validation_results['price_values'] = {
                    'passed': all(price_checks),
                    'message': "All price values are positive" if all(price_checks) else "Some invalid price values found"
                }
        else:
            validation_results['required_columns'] = {
                'passed': False,
                'message': "Data does not have columns attribute"
            }

        return validation_results

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_feature_generation_data_validation_step():
    """Register the feature generation data validation step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_data_validation_step", FeatureGenerationDataValidationStep)
    tprint("✅ Feature generation data validation step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_data_validation_step()
