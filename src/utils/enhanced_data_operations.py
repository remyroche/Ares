#!/usr/bin/env python3
"""
Enhanced Data Operations

This module provides comprehensive data operations utilities that integrate with
the existing common_operations framework and provide enhanced functionality for
data formatting, analysis, adding/removing data, and data access with proper
validation and error handling.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import json
import os

from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load, safe_fillna,
    create_empty_dataframe, safe_rolling, safe_mean, safe_std
)
from src.utils.enhanced_data_validation import (
    DataQualityValidator, DataAccessValidator, EnhancedDataFormatter
)
from src.utils.enhanced_error_handler import (
    EnhancedErrorHandler, create_error_handler_decorator
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose
)
from src.core.domain.decorators import (
    validate_data_quality, monitor_step_execution, 
    ensure_data_integrity, validate_pipeline_step
)

logger = logging.getLogger(__name__)

class DataLoader:
    """Enhanced data loader with validation and error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataLoader")
        self.validator = DataQualityValidator(config)
        self.access_validator = DataAccessValidator(config)
        self.error_handler = EnhancedErrorHandler(config)
    
    @compose(
        error_boundary(name="load_data"),
        traced(span_name="load_data"),
        validate_data_quality(context='data_loading')
    )
    async def load_data(
        self,
        file_path: str,
        file_type: str = 'parquet',
        **kwargs
    ) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Load data with comprehensive validation."""
        try:
            self.logger.info(f"📁 Loading data from: {file_path}")
            
            # Validate data access
            access_passed, access_results = await self.access_validator.validate_data_access(
                'read', file_path
            )
            if not access_passed:
                self.logger.error(f"❌ Data access validation failed: {access_results.get('issues', [])}")
                return False, None
            
            # Check file existence
            if not safe_file_exists(file_path):
                self.logger.error(f"❌ File not found: {file_path}")
                return False, None
            
            # Load data based on file type
            if file_type.lower() == 'parquet':
                data = await self._load_parquet(file_path, **kwargs)
            elif file_type.lower() == 'csv':
                data = await self._load_csv(file_path, **kwargs)
            elif file_type.lower() == 'json':
                data = await self._load_json(file_path, **kwargs)
            else:
                self.logger.error(f"❌ Unsupported file type: {file_type}")
                return False, None
            
            if data is None or data.empty:
                self.logger.error("❌ Failed to load data or data is empty")
                return False, None
            
            # Validate loaded data
            validation_passed, validation_results = await self.validator.validate_ohlc_data(data)
            if not validation_passed:
                self.logger.warning(f"⚠️ Data validation issues: {validation_results.get('issues', [])}")
                # Continue with warnings for now
            
            self.logger.info(f"✅ Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            return True, data
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading data: {e}")
            return False, None
    
    async def _load_parquet(self, file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load parquet file."""
        try:
            return pd.read_parquet(file_path, **kwargs)
        except Exception as e:
            self.logger.exception(f"❌ Error loading parquet file: {e}")
            return None
    
    async def _load_csv(self, file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load CSV file."""
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            self.logger.exception(f"❌ Error loading CSV file: {e}")
            return None
    
    async def _load_json(self, file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load JSON file."""
        try:
            data = safe_json_load(file_path)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                self.logger.error("❌ Invalid JSON structure for DataFrame conversion")
                return None
        except Exception as e:
            self.logger.exception(f"❌ Error loading JSON file: {e}")
            return None

class DataSaver:
    """Enhanced data saver with validation and error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataSaver")
        self.validator = DataQualityValidator(config)
        self.access_validator = DataAccessValidator(config)
        self.error_handler = EnhancedErrorHandler(config)
    
    @compose(
        error_boundary(name="save_data"),
        traced(span_name="save_data"),
        validate_data_quality(context='data_saving')
    )
    async def save_data(
        self,
        data: pd.DataFrame,
        file_path: str,
        file_type: str = 'parquet',
        create_backup: bool = True,
        **kwargs
    ) -> bool:
        """Save data with comprehensive validation and backup."""
        try:
            self.logger.info(f"💾 Saving data to: {file_path}")
            
            # Validate data access
            access_passed, access_results = await self.access_validator.validate_data_access(
                'write', file_path
            )
            if not access_passed:
                self.logger.error(f"❌ Data access validation failed: {access_results.get('issues', [])}")
                return False
            
            # Validate data quality
            validation_passed, validation_results = await self.validator.validate_ohlc_data(data)
            if not validation_passed:
                self.logger.warning(f"⚠️ Data validation issues: {validation_results.get('issues', [])}")
                # Continue with warnings for now
            
            # Create backup if requested and file exists
            if create_backup and safe_file_exists(file_path):
                backup_path = f"{file_path}.backup.{int(datetime.now().timestamp())}"
                await self._create_backup(file_path, backup_path)
            
            # Ensure directory exists
            ensure_directory(Path(file_path).parent)
            
            # Save data based on file type
            if file_type.lower() == 'parquet':
                success = await self._save_parquet(data, file_path, **kwargs)
            elif file_type.lower() == 'csv':
                success = await self._save_csv(data, file_path, **kwargs)
            elif file_type.lower() == 'json':
                success = await self._save_json(data, file_path, **kwargs)
            else:
                self.logger.error(f"❌ Unsupported file type: {file_type}")
                return False
            
            if success:
                self.logger.info(f"✅ Data saved successfully: {len(data)} rows, {len(data.columns)} columns")
                return True
            else:
                self.logger.error("❌ Failed to save data")
                return False
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving data: {e}")
            return False
    
    async def _create_backup(self, original_path: str, backup_path: str) -> bool:
        """Create backup of existing file."""
        try:
            import shutil
            shutil.copy2(original_path, backup_path)
            self.logger.info(f"📋 Backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error creating backup: {e}")
            return False
    
    async def _save_parquet(self, data: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """Save as parquet file."""
        try:
            data.to_parquet(file_path, **kwargs)
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error saving parquet file: {e}")
            return False
    
    async def _save_csv(self, data: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """Save as CSV file."""
        try:
            data.to_csv(file_path, **kwargs)
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error saving CSV file: {e}")
            return False
    
    async def _save_json(self, data: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """Save as JSON file."""
        try:
            json_data = data.to_dict('records')
            safe_json_dump(json_data, file_path, **kwargs)
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error saving JSON file: {e}")
            return False

class DataAnalyzer:
    """Enhanced data analyzer with comprehensive analysis capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataAnalyzer")
        self.validator = DataQualityValidator(config)
        self.error_handler = EnhancedErrorHandler(config)
    
    @compose(
        error_boundary(name="analyze_data"),
        traced(span_name="analyze_data"),
        monitor_step_execution(step_name="data_analysis")
    )
    async def analyze_data(
        self,
        data: pd.DataFrame,
        analysis_type: str = 'comprehensive',
        **kwargs
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Analyze data with comprehensive metrics."""
        try:
            self.logger.info(f"📊 Analyzing data with type: {analysis_type}")
            
            # Validate data quality
            validation_passed, validation_results = await self.validator.validate_ohlc_data(data)
            if not validation_passed:
                self.logger.error(f"❌ Data validation failed: {validation_results.get('issues', [])}")
                return False, None
            
            # Perform analysis based on type
            if analysis_type == 'comprehensive':
                analysis_results = await self._comprehensive_analysis(data, **kwargs)
            elif analysis_type == 'price':
                analysis_results = await self._price_analysis(data, **kwargs)
            elif analysis_type == 'volume':
                analysis_results = await self._volume_analysis(data, **kwargs)
            elif analysis_type == 'regime':
                analysis_results = await self._regime_analysis(data, **kwargs)
            else:
                self.logger.error(f"❌ Unknown analysis type: {analysis_type}")
                return False, None
            
            if analysis_results:
                self.logger.info("✅ Data analysis completed successfully")
                return True, analysis_results
            else:
                self.logger.error("❌ Data analysis failed")
                return False, None
            
        except Exception as e:
            self.logger.exception(f"❌ Error analyzing data: {e}")
            return False, None
    
    async def _comprehensive_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        try:
            analysis_results = {
                'basic_stats': {},
                'price_analysis': {},
                'volume_analysis': {},
                'technical_indicators': {},
                'data_quality': {}
            }
            
            # Basic statistics
            analysis_results['basic_stats'] = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'date_range': {
                    'start': data.index.min() if hasattr(data.index, 'min') else None,
                    'end': data.index.max() if hasattr(data.index, 'max') else None
                },
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
            
            # Price analysis
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                analysis_results['price_analysis'] = await self._price_analysis(data, **kwargs)
            
            # Volume analysis
            if 'volume' in data.columns:
                analysis_results['volume_analysis'] = await self._volume_analysis(data, **kwargs)
            
            # Technical indicators
            analysis_results['technical_indicators'] = await self._calculate_technical_indicators(data, **kwargs)
            
            # Data quality metrics
            analysis_results['data_quality'] = {
                'quality_score': self.validator._calculate_quality_score(data),
                'completeness': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
                'consistency': self._calculate_consistency_score(data)
            }
            
            return analysis_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error in comprehensive analysis: {e}")
            return {}
    
    async def _price_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze price data."""
        try:
            price_analysis = {}
            
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Price statistics
                price_analysis['price_stats'] = {
                    'open': {
                        'min': data['open'].min(),
                        'max': data['open'].max(),
                        'mean': data['open'].mean(),
                        'std': data['open'].std()
                    },
                    'high': {
                        'min': data['high'].min(),
                        'max': data['high'].max(),
                        'mean': data['high'].mean(),
                        'std': data['high'].std()
                    },
                    'low': {
                        'min': data['low'].min(),
                        'max': data['low'].max(),
                        'mean': data['low'].mean(),
                        'std': data['low'].std()
                    },
                    'close': {
                        'min': data['close'].min(),
                        'max': data['close'].max(),
                        'mean': data['close'].mean(),
                        'std': data['close'].std()
                    }
                }
                
                # Price movements
                price_analysis['price_movements'] = {
                    'daily_returns': data['close'].pct_change().dropna(),
                    'volatility': data['close'].pct_change().std(),
                    'max_drawdown': self._calculate_max_drawdown(data['close']),
                    'price_range': data['high'].max() - data['low'].min()
                }
                
                # OHLC relationships
                price_analysis['ohlc_relationships'] = {
                    'high_low_spread': (data['high'] - data['low']).mean(),
                    'open_close_spread': (data['close'] - data['open']).mean(),
                    'body_size': abs(data['close'] - data['open']).mean(),
                    'wick_size': ((data['high'] - data[['open', 'close']].max(axis=1)) + 
                                 (data[['open', 'close']].min(axis=1) - data['low'])).mean()
                }
            
            return price_analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Error in price analysis: {e}")
            return {}
    
    async def _volume_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze volume data."""
        try:
            volume_analysis = {}
            
            if 'volume' in data.columns:
                # Volume statistics
                volume_analysis['volume_stats'] = {
                    'min': data['volume'].min(),
                    'max': data['volume'].max(),
                    'mean': data['volume'].mean(),
                    'std': data['volume'].std(),
                    'median': data['volume'].median()
                }
                
                # Volume patterns
                volume_analysis['volume_patterns'] = {
                    'zero_volume_count': (data['volume'] == 0).sum(),
                    'volume_trend': self._calculate_volume_trend(data['volume']),
                    'volume_volatility': data['volume'].std() / data['volume'].mean() if data['volume'].mean() > 0 else 0
                }
                
                # Volume-price relationship
                if 'close' in data.columns:
                    volume_analysis['volume_price_relationship'] = {
                        'volume_price_correlation': data['volume'].corr(data['close']),
                        'volume_weighted_price': (data['volume'] * data['close']).sum() / data['volume'].sum() if data['volume'].sum() > 0 else 0
                    }
            
            return volume_analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Error in volume analysis: {e}")
            return {}
    
    async def _regime_analysis(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze regime-specific data."""
        try:
            regime_analysis = {}
            
            # Regime identification based on price movements
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                
                # Simple regime classification
                regime_analysis['regime_classification'] = {
                    'bull_market_threshold': 0.02,  # 2% daily return
                    'bear_market_threshold': -0.02,  # -2% daily return
                    'bull_periods': (returns > 0.02).sum(),
                    'bear_periods': (returns < -0.02).sum(),
                    'sideways_periods': ((returns >= -0.02) & (returns <= 0.02)).sum()
                }
                
                # Regime-specific statistics
                bull_returns = returns[returns > 0.02]
                bear_returns = returns[returns < -0.02]
                sideways_returns = returns[(returns >= -0.02) & (returns <= 0.02)]
                
                regime_analysis['regime_stats'] = {
                    'bull_market': {
                        'count': len(bull_returns),
                        'mean_return': bull_returns.mean() if len(bull_returns) > 0 else 0,
                        'volatility': bull_returns.std() if len(bull_returns) > 0 else 0
                    },
                    'bear_market': {
                        'count': len(bear_returns),
                        'mean_return': bear_returns.mean() if len(bear_returns) > 0 else 0,
                        'volatility': bear_returns.std() if len(bear_returns) > 0 else 0
                    },
                    'sideways_market': {
                        'count': len(sideways_returns),
                        'mean_return': sideways_returns.mean() if len(sideways_returns) > 0 else 0,
                        'volatility': sideways_returns.std() if len(sideways_returns) > 0 else 0
                    }
                }
            
            return regime_analysis
            
        except Exception as e:
            self.logger.exception(f"❌ Error in regime analysis: {e}")
            return {}
    
    async def _calculate_technical_indicators(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            indicators = {}
            
            if 'close' in data.columns:
                # Moving averages
                indicators['moving_averages'] = {
                    'sma_20': data['close'].rolling(window=20).mean(),
                    'sma_50': data['close'].rolling(window=50).mean(),
                    'ema_20': data['close'].ewm(span=20).mean(),
                    'ema_50': data['close'].ewm(span=50).mean()
                }
                
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                sma_20 = data['close'].rolling(window=20).mean()
                std_20 = data['close'].rolling(window=20).std()
                indicators['bollinger_bands'] = {
                    'upper': sma_20 + (std_20 * 2),
                    'middle': sma_20,
                    'lower': sma_20 - (std_20 * 2)
                }
            
            return indicators
            
        except Exception as e:
            self.logger.exception(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except Exception:
            return 0.0
    
    def _calculate_volume_trend(self, volume: pd.Series) -> str:
        """Calculate volume trend."""
        try:
            if len(volume) < 2:
                return 'insufficient_data'
            
            first_half = volume[:len(volume)//2].mean()
            second_half = volume[len(volume)//2:].mean()
            
            if second_half > first_half * 1.1:
                return 'increasing'
            elif second_half < first_half * 0.9:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'unknown'
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        try:
            score = 1.0
            
            # Check for consistent data types
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Check if all values are of the same type
                    unique_types = set(type(val).__name__ for val in data[col].dropna())
                    if len(unique_types) > 1:
                        score -= 0.1
            
            # Check for consistent value ranges
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].std() > data[col].mean() * 10:  # High variance
                    score -= 0.05
            
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

class DataManager:
    """Comprehensive data manager that orchestrates all data operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataManager")
        self.loader = DataLoader(config)
        self.saver = DataSaver(config)
        self.analyzer = DataAnalyzer(config)
        self.formatter = EnhancedDataFormatter(config)
        self.error_handler = EnhancedErrorHandler(config)
    
    @compose(
        error_boundary(name="manage_data"),
        traced(span_name="manage_data"),
        monitor_step_execution(step_name="data_management")
    )
    async def process_data_pipeline(
        self,
        input_path: str,
        output_path: str,
        operations: List[str],
        **kwargs
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Process data through a complete pipeline."""
        try:
            self.logger.info(f"🔄 Processing data pipeline: {operations}")
            
            # Load data
            load_success, data = await self.loader.load_data(input_path, **kwargs)
            if not load_success or data is None:
                self.logger.error("❌ Failed to load input data")
                return False, None
            
            # Process data through operations
            processed_data = data.copy()
            pipeline_results = {
                'input_path': input_path,
                'output_path': output_path,
                'operations': operations,
                'results': {}
            }
            
            for operation in operations:
                self.logger.info(f"🔄 Executing operation: {operation}")
                
                if operation == 'format':
                    format_success, processed_data = await self.formatter.format_data(
                        processed_data, **kwargs
                    )
                    pipeline_results['results']['format'] = format_success
                
                elif operation == 'analyze':
                    analysis_success, analysis_results = await self.analyzer.analyze_data(
                        processed_data, **kwargs
                    )
                    pipeline_results['results']['analyze'] = analysis_success
                    if analysis_results:
                        pipeline_results['analysis_results'] = analysis_results
                
                elif operation == 'validate':
                    validation_passed, validation_results = await self.formatter.validator.validate_ohlc_data(
                        processed_data
                    )
                    pipeline_results['results']['validate'] = validation_passed
                    if validation_results:
                        pipeline_results['validation_results'] = validation_results
                
                else:
                    self.logger.warning(f"⚠️ Unknown operation: {operation}")
                    pipeline_results['results'][operation] = False
            
            # Save processed data
            save_success = await self.saver.save_data(processed_data, output_path, **kwargs)
            pipeline_results['results']['save'] = save_success
            
            # Determine overall success
            overall_success = all(pipeline_results['results'].values())
            
            if overall_success:
                self.logger.info("✅ Data pipeline completed successfully")
            else:
                self.logger.error("❌ Data pipeline completed with errors")
            
            return overall_success, pipeline_results
            
        except Exception as e:
            self.logger.exception(f"❌ Error in data pipeline: {e}")
            return False, None

# Export main classes
__all__ = [
    'DataLoader',
    'DataSaver',
    'DataAnalyzer',
    'DataManager'
]