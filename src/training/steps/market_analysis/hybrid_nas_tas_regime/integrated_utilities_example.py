"""
Integrated Utilities Example

This example demonstrates how to use all the integrated utilities from the enhanced
hybrid NAS-TAS regime detection system. It shows best practices for leveraging
the common operations, math validation, serialization, data utilities, matrix operations,
ML common utilities, and hardware optimization tools.

Usage:
    python integrated_utilities_example.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import the enhanced utilities
from .shared_utils import (
    # Common operations
    get_logger, setup_basic_logging, safe_log_metric, safe_log_params, safe_log_artifact,
    get_current_datetime, format_datetime, ensure_directory, safe_json_dump, safe_json_load,
    create_empty_dataframe, validate_dataframe, validate_dataframe_columns,
    safe_dataframe_operation, safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
    safe_drop_columns, safe_rename_columns, optimize_dataframe_dtypes,
    calculate_data_quality_metrics, get_dataframe_info, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std, safe_float, safe_int,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_lower, safe_upper, safe_join,
    safe_append, safe_extend, safe_dict_get, safe_dict_items, safe_sleep, timed_operation,
    format_bytes, chunked_iterable, parallel_map, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, safe_rolling, safe_groupby_operation,
    safe_apply_function, safe_filter_dataframe, create_summary_statistics,
    safe_to_parquet, safe_read_parquet, list_parquet_files, get_latest_outcome_file,
    load_latest_optimal_regime_clustering_outcome, safe_copy, safe_deepcopy,
    safe_resample, align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
    sanitize_string, memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space, JSONSerializer, PickleSerializer,
    ParquetSerializer, UniversalSerializer, initialize_enhanced_utilities,

    # M1 optimization utilities
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, integrate_with_m1_optimizers,

    # Data utilities integration
    get_unified_data_utils, get_feature_engineer, get_data_quality_framework,
    process_market_data
)

# Setup logging
setup_basic_logging()
logger = get_logger(__name__)

class IntegratedUtilitiesExample:
    """Example class demonstrating integrated utilities usage."""

    def __init__(self):
        """Initialize the example with integrated utilities."""
        logger.info("🚀 Initializing Integrated Utilities Example")

        # Initialize all utilities
        self.utilities_status = initialize_enhanced_utilities()
        logger.info(f"✅ Utilities initialization status: {self.utilities_status}")

        # Initialize M1 optimization
        self.m1_status = integrate_with_m1_optimizers()
        logger.info(f"✅ M1 optimization status: {self.m1_status}")

    @timed_operation
    def create_sample_market_data(self) -> pd.DataFrame:
        """Create sample market data using enhanced utilities."""
        logger.info("📊 Creating sample market data")

        # Create empty DataFrame with proper columns
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        df = create_empty_dataframe(columns)

        # Generate sample data
        start_date = get_current_datetime() - timedelta(days=100)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

        for symbol in symbols:
            for i in range(100):
                timestamp = start_date + timedelta(days=i)
                open_price = 100 + np.random.randn() * 10
                close_price = open_price + np.random.randn() * 5
                high_price = max(open_price, close_price) + abs(np.random.randn()) * 2
                low_price = min(open_price, close_price) - abs(np.random.randn()) * 2
                volume = np.random.randint(100000, 1000000)

                row = {
                    'timestamp': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume,
                    'symbol': symbol
                }

                df = safe_dataframe_operation(df, lambda x: x.append(row, ignore_index=True), df)

        logger.info(f"✅ Created sample data with shape: {df.shape}")
        return df

    @memory_checkpoint("data_preprocessing")
    def preprocess_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data using integrated utilities."""
        logger.info("🔧 Preprocessing market data")

        # Validate DataFrame
        if not validate_dataframe(df):
            raise ValueError("Invalid DataFrame")

        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not validate_dataframe_columns(df, required_columns):
            raise ValueError("Missing required columns")

        # Convert timestamp column
        df = safe_timestamp_conversion(df, 'timestamp')

        # Optimize data types
        df = optimize_dataframe_dtypes(df)

        # Fill missing values
        df = safe_fillna(df, method='forward')
        df = safe_fillna(df, value=0)

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        logger.info(f"✅ Removed {initial_rows - final_rows} duplicate rows")

        # Add derived features
        df = safe_dataframe_operation(df, self._add_derived_features, df)

        # Validate final schema
        if not validate_dataframe_schema(df, required_columns + ['returns', 'volatility']):
            logger.warning("⚠️ Schema validation failed for derived features")

        logger.info(f"✅ Data preprocessing completed. Final shape: {df.shape}")
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to DataFrame."""
        try:
            # Calculate returns
            df['returns'] = df.groupby('symbol')['close'].transform(
                lambda x: x.pct_change()
            )

            # Calculate volatility (rolling standard deviation)
            df['volatility'] = df.groupby('symbol')['returns'].transform(
                lambda x: x.rolling(20).std()
            )

            # Calculate moving averages
            df['ma_20'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20).mean()
            )

            df['ma_50'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(50).mean()
            )

            # Calculate RSI
            df['rsi'] = df.groupby('symbol').apply(
                lambda x: self._calculate_rsi(x['close'])
            ).reset_index(0, drop=True)

            return df
        except Exception as e:
            logger.error(f"Error adding derived features: {e}")
            return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for a price series."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.fillna(50)  # Fill NaN with neutral RSI value
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    @gpu_context("feature_engineering")
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using ML common utilities."""
        logger.info("🔬 Engineering features with ML common utilities")

        try:
            # Use unified data utilities for feature engineering
            data_utils = get_unified_data_utils()
            feature_engineer = get_feature_engineer()

            # Process data with quality validation
            processed_df = data_utils.process_and_validate(
                data=df,
                validate_quality=True,
                clean_missing_values=True,
                detect_outliers=True,
                optimize_dtypes=True
            )

            # Engineer additional features
            engineered_df = feature_engineer.engineer_features(
                processed_df,
                feature_types=['basic', 'technical', 'statistical']
            )

            logger.info(f"✅ Feature engineering completed. Shape: {engineered_df.shape}")
            return engineered_df

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return df

    def perform_cross_validation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform cross validation analysis using ML common utilities."""
        logger.info("🔍 Performing cross validation analysis")

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Prepare data for ML
            df_ml = df.copy()

            # Create target variable (1 if next day return > 0, 0 otherwise)
            df_ml['target'] = (df_ml.groupby('symbol')['close'].shift(-1) > df_ml['close']).astype(int)

            # Remove rows with NaN targets
            df_ml = df_ml.dropna(subset=['target'])

            # Split features and target
            feature_columns = ['returns', 'volatility', 'ma_20', 'ma_50', 'rsi', 'volume']
            X = df_ml[feature_columns]
            y = df_ml['target']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            # Create and train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Perform cross validation analysis
            cv_result = perform_cross_validation_analysis(train_data, model, cv_method='time_series')

            # Detect model issues
            model_issues = detect_model_issues(model, train_data, test_data)

            return {
                'cross_validation': cv_result,
                'model_issues': model_issues,
                'model_score': model.score(X_test, y_test)
            }

        except Exception as e:
            logger.error(f"Error in cross validation analysis: {e}")
            return {'error': str(e)}

    def run_matrix_operations_example(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run matrix operations example."""
        logger.info("🔢 Running matrix operations example")

        try:
            # Extract numeric data for matrix operations
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            data_matrix = df[numeric_cols].values

            # Compute correlation matrix
            corr_matrix = compute_correlation_matrix(data_matrix)

            # Compute covariance matrix
            cov_matrix = compute_covariance_matrix(data_matrix)

            # Perform batch operations
            batch_corr_matrices = batch_compute_correlation_matrices([data_matrix])

            # Validate correlation matrix
            is_valid_corr = validate_correlation_matrix(corr_matrix)

            return {
                'correlation_matrix_shape': corr_matrix.shape,
                'covariance_matrix_shape': cov_matrix.shape,
                'is_valid_correlation': is_valid_corr,
                'batch_results_count': len(batch_corr_matrices)
            }

        except Exception as e:
            logger.error(f"Error in matrix operations: {e}")
            return {'error': str(e)}

    def save_and_load_data_example(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Save and load data using serialization utilities."""
        logger.info("💾 Running save/load data example")

        try:
            # Ensure output directory exists
            ensure_directory('data/output')

            # Save using different serialization methods
            json_path = 'data/output/sample_data.json'
            parquet_path = 'data/output/sample_data.parquet'
            pickle_path = 'data/output/sample_data.pkl'

            # JSON serialization
            json_success = safe_json_dump(df.to_dict('records'), json_path)

            # Parquet serialization
            parquet_success = safe_to_parquet(df, parquet_path)

            # Pickle serialization
            pickle_success = PickleSerializer.save(df, pickle_path)

            # Load data back
            loaded_json = safe_json_load(json_path)
            loaded_parquet = safe_read_parquet(parquet_path)
            loaded_pickle = PickleSerializer.load(pickle_path)

            return {
                'json_save_success': json_success,
                'parquet_save_success': parquet_success,
                'pickle_save_success': pickle_success,
                'json_load_success': loaded_json is not None,
                'parquet_load_success': loaded_parquet is not None,
                'pickle_load_success': loaded_pickle is not None,
                'data_shapes_match': (len(df) == len(loaded_parquet) if loaded_parquet is not None else False)
            }

        except Exception as e:
            logger.error(f"Error in save/load operations: {e}")
            return {'error': str(e)}

    @timed_operation
    def run_comprehensive_example(self):
        """Run the comprehensive integrated utilities example."""
        logger.info("🚀 Starting comprehensive integrated utilities example")

        results = {
            'timestamp': get_current_datetime(),
            'utilities_status': self.utilities_status,
            'm1_status': self.m1_status,
            'steps': {}
        }

        try:
            # Step 1: Create sample data
            df = self.create_sample_market_data()
            results['steps']['data_creation'] = {'success': True, 'shape': df.shape}

            # Step 2: Preprocess data
            preprocessed_df = self.preprocess_market_data(df)
            results['steps']['data_preprocessing'] = {'success': True, 'shape': preprocessed_df.shape}

            # Step 3: Feature engineering
            engineered_df = self.engineer_features(preprocessed_df)
            results['steps']['feature_engineering'] = {'success': True, 'shape': engineered_df.shape}

            # Step 4: Cross validation analysis
            cv_results = self.perform_cross_validation_analysis(engineered_df)
            results['steps']['cross_validation'] = {'success': True, 'cv_score': cv_results.get('model_score', 0)}

            # Step 5: Matrix operations
            matrix_results = self.run_matrix_operations_example(engineered_df)
            results['steps']['matrix_operations'] = {'success': True, 'details': matrix_results}

            # Step 6: Save and load data
            save_load_results = self.save_and_load_data_example(engineered_df)
            results['steps']['save_load_operations'] = {'success': True, 'details': save_load_results}

            # Calculate data quality metrics
            quality_metrics = calculate_data_quality_metrics(engineered_df)
            results['data_quality'] = quality_metrics

            # Log memory usage
            memory_usage = get_memory_usage()
            results['memory_usage'] = format_bytes(memory_usage)

            # Log execution summary
            successful_steps = sum(1 for step in results['steps'].values() if step.get('success', False))
            logger.info(f"✅ Completed {successful_steps}/{len(results['steps'])} steps successfully")

            # Save results
            results_path = 'data/output/integrated_utilities_results.json'
            ensure_directory('data/output')
            safe_json_dump(results, results_path)
            safe_log_artifact('integrated_utilities_results', results_path)

            return results

        except Exception as e:
            logger.error(f"❌ Error in comprehensive example: {e}")
            results['error'] = str(e)
            return results

def main():
    """Main function to run the integrated utilities example."""
    logger.info("🚀 Starting Integrated Utilities Example")

    try:
        # Create example instance
        example = IntegratedUtilitiesExample()

        # Run comprehensive example
        results = example.run_comprehensive_example()

        # Log summary
        logger.info("📋 Example Results Summary:")
        logger.info(f"   - Timestamp: {results['timestamp']}")
        logger.info(f"   - Utilities Status: {results['utilities_status']['overall_status']}")
        logger.info(f"   - M1 Status: {results['m1_status']['integration_status']}")
        logger.info(f"   - Memory Usage: {results['memory_usage']}")

        successful_steps = sum(1 for step in results['steps'].values() if step.get('success', False))
        logger.info(f"   - Successful Steps: {successful_steps}/{len(results['steps'])}")

        if 'data_quality' in results:
            quality = results['data_quality']
            logger.info(f"   - Data Quality: {quality['missing_percentage']:.2f}% missing, {quality['duplicate_percentage']:.2f}% duplicates")

        logger.info("✅ Integrated utilities example completed successfully!")
        return results

    except Exception as e:
        logger.error(f"❌ Error running integrated utilities example: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    main()