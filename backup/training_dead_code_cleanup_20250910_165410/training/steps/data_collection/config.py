from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Configuration module for data collection steps."""

CONFIG = {
    'validation_enabled': True,
    'data_quality_threshold': 0.8,
    'min_records': 500,
    'max_gap_ratio': 0.2,
    'max_gap_hours': 48,
    'required_columns': ['open', 'high', 'low', 'close', 'volume'],
    'timeframe': '1m',
    'exchanges': ['BINANCE'],
    'symbols': ['ETHUSDT', 'BTCUSDT'],
    'data_retention_days': 180,
    'validation_timeout': 300,
    'max_file_size_mb': 100,
    'parallel_processing': True,
    'max_workers': 4,
    'cache_enabled': True,
    'log_level': 'INFO',
    'metrics_enabled': True,
    'error_handling': {
        'retry_attempts': 3,
        'retry_delay': 5,
        'continue_on_error': True
    },
    'data_quality': {
        'check_duplicates': True,
        'check_missing_values': True,
        'check_data_types': True,
        'check_time_continuity': True,
        'min_quality_score': 85
    }
}