"""Integration Test for Step 1.5 Data Converter with Security Decorators

This test verifies that the step1_5_data_converter is properly integrated
with all security decorators and pipeline components.
"""

import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.training.steps.step1_5_data_converter import run_step


class TestStep1_5Integration(unittest.TestCase):
    """Integration tests for Step 1.5 Data Converter."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir, tempfile.mkdtemp()
        self.symbol = "ETHUSDT"
        self.exchange = "BINANCE"
        self.timeframe = "1h"

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.training.steps.step1_5_data_converter.download_all_data_with_consolidation')
    async def test_secure_klines_download_integration(self, mock_download):
        """Test that klines download is properly secured with decorators."""
        # Mock successful download
        mock_download.return_value, True

        # Create mock klines data
        mock_klines_data, pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': [100 + i for i in range(100)],
            'high': [101 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [100.5 + i for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

        # Mock file operations
        with patch('glob.glob') as mock_glob, \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.concat') as mock_concat, \
             patch('pandas.DataFrame.to_parquet') as mock_to_parquet:

        # Mock file discovery
            mock_glob.return_value = [f"data_cache/klines_{self.exchange}_{self.symbol}_{self.timeframe}_2024-01.csv"]

        # Mock file reading
            mock_read_csv.return_value, mock_klines_data

        # Mock data concatenation
            mock_concat.return_value, mock_klines_data

        # Mock file writing
            mock_to_parquet.return_value, None

        # Execute the step
            result, await run_step(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                data_dir=self.temp_dir,
                force_rerun=True
            )

        # Verify the result
        self.assertTrue(result)

        # Verify that download was called
            mock_download.assert_called_once_with(
                symbol=self.symbol,
                exchange_name=self.exchange,
                interval=self.timeframe
            )

    @patch('src.training.steps.step1_5_data_converter.download_all_data_with_consolidation')
    async def test_security_validation_integration(self, mock_download):
        """Test that security validations are properly integrated."""
        # Mock successful download
        mock_download.return_value, True

        # Test with invalid symbol (should be caught by security decorator)
        with self.assertRaises(Exception):
        await run_step(
                symbol="ETHUSDT<script>",  # Invalid symbol with injection attempt
                exchange=self.exchange,
                timeframe=self.timeframe,
                data_dir=self.temp_dir,
                force_rerun=True
            )

    @patch('src.training.steps.step1_5_data_converter.download_all_data_with_consolidation')
    async def test_data_quality_validation_integration(self, mock_download):
        """Test that data quality validations are properly integrated."""
        # Mock successful download
        mock_download.return_value, True

        # Create mock klines data with quality issues
        mock_klines_data, pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': [100 + i for i in range(100)],
            'high': [101 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [100.5 + i for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

        # Add some quality issues
        mock_klines_data.loc[50, 'close'] = -1  # Negative price
        mock_klines_data.loc[51, 'volume'] = None  # Null value

        with patch('glob.glob') as mock_glob, \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.concat') as mock_concat:

            mock_glob.return_value = [f"data_cache/klines_{self.exchange}_{self.symbol}_{self.timeframe}_2024-01.csv"]
            mock_read_csv.return_value, mock_klines_data
            mock_concat.return_value, mock_klines_data

        # Execute the step - should handle quality issues gracefully
        await run_step(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                data_dir=self.temp_dir,
                force_rerun=True
            )

        # The step should handle quality issues and return False or handle them appropriately
        # This depends on the specific implementation of the quality decorators

    @patch('src.training.steps.step1_5_data_converter.download_all_data_with_consolidation')
    async def test_error_handling_integration(self, mock_download):
        """Test that error handling is properly integrated."""
        # Mock download failure
        mock_download.side_effect, Exception("Download failed")

        # Execute the step - should handle the error gracefully
        result, await run_step(
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe,
            data_dir=self.temp_dir,
            force_rerun=True
        )

        # Should return False due to error handling decorator
        self.assertFalse(result)

    @patch('src.training.steps.step1_5_data_converter.download_all_data_with_consolidation')
    async def test_resource_monitoring_integration(self, mock_download):
        """Test that resource monitoring is properly integrated."""
        # Mock successful download
        mock_download.return_value, True

        # Create mock klines data
        mock_klines_data, pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': [100 + i for i in range(100)],
            'high': [101 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [100.5 + i for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

        with patch('glob.glob') as mock_glob, \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.concat') as mock_concat, \
             patch('pandas.DataFrame.to_parquet') as mock_to_parquet:

            mock_glob.return_value = [f"data_cache/klines_{self.exchange}_{self.symbol}_{self.timeframe}_2024-01.csv"]
            mock_read_csv.return_value, mock_klines_data
            mock_concat.return_value, mock_klines_data
            mock_to_parquet.return_value, None

        # Execute the step
            result, await run_step(
                symbol=self.symbol,
                exchange=self.exchange,
                timeframe=self.timeframe,
                data_dir=self.temp_dir,
                force_rerun=True
            )

        # Should complete successfully with resource monitoring
        self.assertTrue(result)

    def test_decorator_imports(self):
        """Test that all required decorators are properly imported."""
        from src.training.steps.step1_5_data_converter import (
            handle_errors,
            handle_file_operations,
            secure_klines_download_operation,
            validate_klines_data_quality,
            secure_data_processing,
            prevent_data_leakage,
            resource_monitor,
            memory_efficient,
            quality_gate,
            circuit_breaker_protection,
        )

        # Verify all decorators are callable
        self.assertTrue(callable(handle_errors))
        self.assertTrue(callable(handle_file_operations))
        self.assertTrue(callable(secure_klines_download_operation))
        self.assertTrue(callable(validate_klines_data_quality))
        self.assertTrue(callable(secure_data_processing))
        self.assertTrue(callable(prevent_data_leakage))
        self.assertTrue(callable(resource_monitor))
        self.assertTrue(callable(memory_efficient))
        self.assertTrue(callable(quality_gate))
        self.assertTrue(callable(circuit_breaker_protection))


if __name__ == '__main__':
    # Run the tests
    unittest.main()
