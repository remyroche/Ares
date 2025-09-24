"""
Enhanced Data Operations for Perfect NAS Regime System

Integrates with utils/data/ and utils/common_operations.py for comprehensive data handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Import enhanced utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    safe_file_exists, timed_operation, format_bytes,
    safe_dataframe_operation, validate_dataframe_columns,
    safe_convert_dtypes, safe_merge_dataframes,
    safe_drop_columns, safe_rename_columns,
    safe_timestamp_conversion, validate_timestamp_column,
    optimize_dataframe_dtypes, calculate_data_quality_metrics,
    get_dataframe_info, create_data_quality_report,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    safe_resample, align_dataframes, validate_dataframe_schema
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    validate_positive, validate_range, safe_correlation,
    validate_numeric_array, MathValidationError
)
from src.utils.serialization_utils import UniversalSerializer

# Import data utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    from src.utils.data.processing.data_processing import DataProcessor
    from src.utils.data.quality.data_quality import DataQualityAnalyzer
    from src.utils.data.validation.validators import DataValidator
    DATA_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data utilities not available: {e}")
    DATA_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedDataOperations:
    """
    Enhanced data operations for Perfect NAS Regime System.
    
    Integrates with existing data utilities for:
    - Comprehensive data loading and processing
    - Quality analysis and validation
    - Safe mathematical operations
    - Serialization and persistence
    - Klines parquet management
    """
    
    def __init__(self, data_dir: str = "historical_data", enable_validation: bool = True):
        """Initialize enhanced data operations.
        
        Args:
            data_dir: Base directory for data storage
            enable_validation: Enable data validation
        """
        self.data_dir = data_dir
        self.enable_validation = enable_validation
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize serialization
        self.serializer = UniversalSerializer()
        
        # Initialize data utilities if available
        if DATA_UTILS_AVAILABLE:
            try:
                self.klines_manager = get_klines_manager(data_dir)
                self.data_processor = DataProcessor()
                self.quality_analyzer = DataQualityAnalyzer()
                self.data_validator = DataValidator()
                self.logger.info("✅ Enhanced data operations initialized with full utilities")
            except Exception as e:
                self.logger.warning(f"Data utilities initialization failed: {e}")
                self._initialize_fallback_utilities()
        else:
            self.logger.warning("Data utilities not available - using fallback implementations")
            self._initialize_fallback_utilities()
    
    def _initialize_fallback_utilities(self):
        """Initialize fallback utilities when data utils are not available."""
        self.klines_manager = None
        self.data_processor = None
        self.quality_analyzer = None
        self.data_validator = None
    
    @timed_operation
    def load_market_data(self, symbol: str, interval: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        data_type: str = "processed") -> Optional[pd.DataFrame]:
        """Load market data using enhanced klines parquet manager."""
        try:
            if self.klines_manager:
                data = self.klines_manager.read_data(
                    symbol, interval, start_date, end_date, data_type
                )
                
                if data is not None and self.enable_validation:
                    # Validate loaded data
                    validation_result = self.validate_market_data(data)
                    if not validation_result['is_valid']:
                        self.logger.warning(f"Data validation failed for {symbol} {interval}")
                        self.logger.warning(f"Errors: {validation_result.get('errors', [])}")
                
                return data
            else:
                # Fallback to manual loading
                return self._fallback_load_data(symbol, interval, start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Failed to load market data for {symbol} {interval}: {e}")
            return None
    
    def validate_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data using comprehensive validation."""
        try:
            if self.data_validator:
                return self.data_validator.validate_dataframe(data, 'market_data')
            else:
                # Enhanced fallback validation
                return self._fallback_validate_data(data)
                
        except Exception as e:
            self.logger.warning(f"Data validation failed: {e}")
            return {'is_valid': False, 'errors': [str(e)]}
    
    def _fallback_validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced fallback validation for market data."""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'data_quality_score': 1.0
            }
            
            # Check required columns for OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                validation_result['is_valid'] = False
                validation_result['data_quality_score'] = 0.0
                return validation_result
            
            # Check for negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_prices = (data[col] < 0).sum()
                    if negative_prices > 0:
                        validation_result['warnings'].append(f"Found {negative_prices} negative prices in {col}")
                        validation_result['data_quality_score'] -= 0.1
            
            # Check price relationships
            invalid_relationships = 0
            for i in range(len(data)):
                if (data.iloc[i]['high'] < data.iloc[i]['low'] or 
                    data.iloc[i]['high'] < data.iloc[i]['open'] or 
                    data.iloc[i]['high'] < data.iloc[i]['close'] or
                    data.iloc[i]['low'] > data.iloc[i]['open'] or 
                    data.iloc[i]['low'] > data.iloc[i]['close']):
                    invalid_relationships += 1
            
            if invalid_relationships > 0:
                validation_result['warnings'].append(f"Found {invalid_relationships} records with invalid price relationships")
                validation_result['data_quality_score'] -= 0.2
            
            # Check for zero volume
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > 0:
                validation_result['warnings'].append(f"Found {zero_volume} records with zero volume")
                validation_result['data_quality_score'] -= 0.05
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                validation_result['warnings'].append(f"Found {missing_values} missing values")
                validation_result['data_quality_score'] -= 0.1
            
            validation_result['data_quality_score'] = max(0.0, validation_result['data_quality_score'])
            
            return validation_result
            
        except Exception as e:
            return {'is_valid': False, 'errors': [str(e)], 'data_quality_score': 0.0}
    
    @timed_operation
    def process_market_data(self, data: pd.DataFrame, 
                           features: List[str] = None) -> pd.DataFrame:
        """Process market data with enhanced feature engineering."""
        try:
            if data is None or data.empty:
                return data
            
            processed_data = data.copy()
            
            # Add basic technical indicators
            if features is None:
                features = ['returns', 'volatility', 'momentum', 'rsi', 'macd']
            
            # Calculate returns
            if 'returns' in features:
                processed_data['returns'] = safe_divide(
                    processed_data['close'].pct_change(), 1.0
                )
            
            # Calculate volatility
            if 'volatility' in features:
                processed_data['volatility'] = processed_data['returns'].rolling(20).std()
            
            # Calculate momentum
            if 'momentum' in features:
                processed_data['momentum'] = processed_data['close'].pct_change(5)
            
            # Calculate RSI (simplified)
            if 'rsi' in features:
                delta = processed_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = safe_divide(gain, loss)
                processed_data['rsi'] = 100 - safe_divide(100, 1 + rs)
            
            # Calculate MACD (simplified)
            if 'macd' in features:
                exp1 = processed_data['close'].ewm(span=12).mean()
                exp2 = processed_data['close'].ewm(span=26).mean()
                processed_data['macd'] = exp1 - exp2
                processed_data['macd_signal'] = processed_data['macd'].ewm(span=9).mean()
                processed_data['macd_histogram'] = processed_data['macd'] - processed_data['macd_signal']
            
            # Optimize data types
            processed_data = optimize_dataframe_dtypes(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.warning(f"Market data processing failed: {e}")
            return data
    
    def save_processed_data(self, data: pd.DataFrame, symbol: str, 
                           interval: str, data_type: str = "processed") -> bool:
        """Save processed data using klines parquet manager."""
        try:
            if self.klines_manager:
                return self.klines_manager.write_data(data, symbol, interval, data_type, overwrite=True)
            else:
                # Fallback saving
                return self._fallback_save_data(data, symbol, interval)
                
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            return False
    
    def _fallback_save_data(self, data: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Fallback data saving."""
        try:
            ensure_directory(self.data_dir)
            filename = f"{symbol.lower()}_{interval}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = Path(self.data_dir) / filename
            
            return safe_to_parquet(data, filepath)
            
        except Exception as e:
            self.logger.error(f"Fallback data saving failed: {e}")
            return False
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data quality report."""
        try:
            if self.quality_analyzer:
                return self.quality_analyzer.analyze_dataframe(data)
            else:
                # Enhanced fallback quality report
                return create_data_quality_report(data)
                
        except Exception as e:
            self.logger.warning(f"Data quality report generation failed: {e}")
            return {'error': str(e)}
    
    def save_operations_state(self, filepath: str) -> bool:
        """Save current operations state."""
        try:
            state = {
                'data_dir': self.data_dir,
                'enable_validation': self.enable_validation,
                'data_utils_available': DATA_UTILS_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
            
            return self.serializer.save(state, filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save operations state: {e}")
            return False
    
    def load_operations_state(self, filepath: str) -> bool:
        """Load operations state."""
        try:
            state = self.serializer.load(filepath)
            if state is None:
                return False
            
            # Restore settings if available
            if 'enable_validation' in state:
                self.enable_validation = state['enable_validation']
            
            self.logger.info("✅ Data operations state loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load operations state: {e}")
            return False
    
    def _fallback_load_data(self, symbol: str, interval: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Fallback data loading when klines manager is not available."""
        try:
            # This would typically load from a different source or file format
            # For now, return None to indicate no data available
            self.logger.warning(f"Fallback data loading not implemented for {symbol} {interval}")
            return None
            
        except Exception as e:
            self.logger.error(f"Fallback data loading failed: {e}")
            return None