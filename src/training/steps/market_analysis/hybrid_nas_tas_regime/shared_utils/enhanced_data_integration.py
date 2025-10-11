"""
Enhanced Data Integration Module

This module provides comprehensive data processing capabilities by integrating
with existing data utilities from src/utils/data/.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime, timedelta
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing data utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetProcessor
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    from src.utils.data.feature_engineer import FeatureEngineer
    from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
    from src.utils.data.gap_detector import GapDetector
    from src.utils.data.quality.data_quality import DataQualityAnalyzer
    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
    from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
    from src.utils.data.optimized_parquet_storage import OptimizedParquetStorage
    from src.utils.data.processing.data_processing import DataProcessor
    from src.utils.data.processing.transformers import DataTransformer
except ImportError as e:
    logging.warning(f"Some data utilities not available: {e}")
    KlinesParquetProcessor = None
    UnifiedDataUtils = None
    HistoricalDataDownloader = None
    FeatureEngineer = None
    BasicReturnsEngineer = None
    GapDetector = None
    DataQualityAnalyzer = None
    AdvancedQualityMetrics = None
    ComprehensiveQualityScorer = None
    OptimizedParquetStorage = None
    DataProcessor = None
    DataTransformer = None

# Import utility integration
from .enhanced_utility_integration import EnhancedUtilityIntegration, UtilityIntegrationConfig

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class DataIntegrationConfig:
    """Configuration for data integration."""
    enable_klines_parquet: bool = True
    enable_unified_data_utils: bool = True
    enable_historical_downloader: bool = True
    enable_feature_engineering: bool = True
    enable_returns_engineering: bool = True
    enable_gap_detection: bool = True
    enable_data_quality: bool = True
    enable_advanced_quality_metrics: bool = True
    enable_comprehensive_quality_scoring: bool = True
    enable_optimized_storage: bool = True
    enable_parquet_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    enable_schema_validation: bool = True
    enable_data_consistency_checks: bool = True
    enable_feature_selection: bool = True
    enable_data_transformation: bool = True
    enable_temporal_analysis: bool = True
    enable_regime_detection: bool = True
    enable_anomaly_detection: bool = True
    enable_correlation_analysis: bool = True
    enable_statistical_analysis: bool = True
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    enable_compression: bool = True


class EnhancedDataIntegration:
    """
    Enhanced data integration that consolidates functionality from existing data utilities.
    """
    
    def __init__(self, config: DataIntegrationConfig, utility_integration: EnhancedUtilityIntegration = None):
        """Initialize enhanced data integration."""
        self.config = config
        self.utility_integration = utility_integration or EnhancedUtilityIntegration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize data components
        self._initialize_data_components()
        
        # Performance tracking
        self.performance_metrics = {
            'processing_times': [],
            'memory_usage': [],
            'data_quality_scores': [],
            'processing_errors': []
        }
        
        self.logger.info("✅ Enhanced data integration initialized")
    
    def _initialize_data_components(self):
        """Initialize data processing components based on configuration."""
        try:
            # Initialize klines parquet processor
            if self.config.enable_klines_parquet and KlinesParquetProcessor:
                self.klines_processor = KlinesParquetProcessor()
                self.logger.info("✅ Klines parquet processor initialized")
            
            # Initialize unified data utils
            if self.config.enable_unified_data_utils and UnifiedDataUtils:
                self.unified_data_utils = UnifiedDataUtils()
                self.logger.info("✅ Unified data utils initialized")
            
            # Initialize historical data downloader
            if self.config.enable_historical_downloader and HistoricalDataDownloader:
                self.historical_downloader = HistoricalDataDownloader()
                self.logger.info("✅ Historical data downloader initialized")
            
            # Initialize feature engineer
            if self.config.enable_feature_engineering and FeatureEngineer:
                self.feature_engineer = FeatureEngineer()
                self.logger.info("✅ Feature engineer initialized")
            
            # Initialize returns engineer
            if self.config.enable_returns_engineering and BasicReturnsEngineer:
                self.returns_engineer = BasicReturnsEngineer()
                self.logger.info("✅ Returns engineer initialized")
            
            # Initialize gap detector
            if self.config.enable_gap_detection and GapDetector:
                self.gap_detector = GapDetector()
                self.logger.info("✅ Gap detector initialized")
            
            # Initialize data quality analyzer
            if self.config.enable_data_quality and DataQualityAnalyzer:
                self.data_quality_analyzer = DataQualityAnalyzer()
                self.logger.info("✅ Data quality analyzer initialized")
            
            # Initialize advanced quality metrics
            if self.config.enable_advanced_quality_metrics and AdvancedQualityMetrics:
                self.advanced_quality_metrics = AdvancedQualityMetrics()
                self.logger.info("✅ Advanced quality metrics initialized")
            
            # Initialize comprehensive quality scorer
            if self.config.enable_comprehensive_quality_scoring and ComprehensiveQualityScorer:
                self.comprehensive_quality_scorer = ComprehensiveQualityScorer()
                self.logger.info("✅ Comprehensive quality scorer initialized")
            
            # Initialize optimized parquet storage
            if self.config.enable_optimized_storage and OptimizedParquetStorage:
                self.optimized_storage = OptimizedParquetStorage()
                self.logger.info("✅ Optimized parquet storage initialized")
            
            # Initialize data processor
            if self.config.enable_data_processing and DataProcessor:
                self.data_processor = DataProcessor()
                self.logger.info("✅ Data processor initialized")
            
            # Initialize data transformer
            if self.config.enable_data_transformation and DataTransformer:
                self.data_transformer = DataTransformer()
                self.logger.info("✅ Data transformer initialized")
            
            self.logger.info("✅ All data components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize data components: {e}")
            raise
    
    # =============================================================================
    # MARKET DATA PROCESSING
    # =============================================================================
    
    def process_market_data(self, data: pd.DataFrame, symbol: str = "BTCUSDT", timeframe: str = "15m") -> pd.DataFrame:
        """Process market data using enhanced data integration."""
        try:
            start_time = time.time()
            
            # Validate input data
            if not self.utility_integration.validate_dataframe_columns(data, ['open', 'high', 'low', 'close']):
                self.logger.warning("⚠️ Missing required OHLC columns, adding defaults")
                data = self._add_default_ohlc_columns(data)
            
            # Process using klines processor if available
            if hasattr(self, 'klines_processor'):
                processed_data = self.klines_processor.process_klines(data, symbol, timeframe)
            else:
                processed_data = data.copy()
            
            # Apply data quality checks
            if self.config.enable_data_quality:
                quality_metrics = self.calculate_data_quality_metrics(processed_data)
                self.performance_metrics['data_quality_scores'].append(quality_metrics)
            
            # Detect and handle gaps
            if self.config.enable_gap_detection and hasattr(self, 'gap_detector'):
                gap_analysis = self.gap_detector.detect_gaps(processed_data)
                if gap_analysis.get('gaps_detected', False):
                    self.logger.warning(f"⚠️ {gap_analysis['gap_count']} gaps detected in data")
                    processed_data = self._handle_data_gaps(processed_data, gap_analysis)
            
            # Optimize data types
            processed_data = self.utility_integration.optimize_dataframe_dtypes(processed_data)
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            
            self.logger.info(f"✅ Market data processed successfully in {processing_time:.2f}s")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Market data processing failed: {e}")
            self.performance_metrics['processing_errors'].append(str(e))
            return data
    
    def _add_default_ohlc_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add default OHLC columns if missing."""
        if 'open' not in data.columns:
            data['open'] = data.get('close', 100.0)
        if 'high' not in data.columns:
            data['high'] = data['open'] * 1.02
        if 'low' not in data.columns:
            data['low'] = data['open'] * 0.98
        if 'close' not in data.columns:
            data['close'] = data['open']
        if 'volume' not in data.columns:
            data['volume'] = 1000.0
        return data
    
    def _handle_data_gaps(self, data: pd.DataFrame, gap_analysis: Dict[str, Any]) -> pd.DataFrame:
        """Handle data gaps using appropriate methods."""
        try:
            # Simple gap handling - forward fill
            data = data.fillna(method='ffill')
            self.logger.info("✅ Data gaps handled using forward fill")
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Gap handling failed: {e}")
            return data
    
    # =============================================================================
    # FEATURE ENGINEERING
    # =============================================================================
    
    def engineer_features(self, data: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
        """Engineer features using enhanced data integration."""
        try:
            if feature_types is None:
                feature_types = ['momentum', 'volatility', 'volume', 'technical']
            
            start_time = time.time()
            engineered_data = data.copy()
            
            # Use feature engineer if available
            if hasattr(self, 'feature_engineer'):
                for feature_type in feature_types:
                    try:
                        if feature_type == 'momentum':
                            features = self.feature_engineer.calculate_momentum_features(data)
                        elif feature_type == 'volatility':
                            features = self.feature_engineer.calculate_volatility_features(data)
                        elif feature_type == 'volume':
                            features = self.feature_engineer.calculate_volume_features(data)
                        elif feature_type == 'technical':
                            features = self.feature_engineer.calculate_technical_indicators(data)
                        else:
                            continue
                        
                        if not features.empty:
                            engineered_data = pd.concat([engineered_data, features], axis=1)
                            self.logger.info(f"✅ {feature_type} features engineered: {features.shape[1]} features")
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to engineer {feature_type} features: {e}")
                        continue
            else:
                # Fallback to basic feature engineering
                engineered_data = self._basic_feature_engineering(data, feature_types)
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            
            self.logger.info(f"✅ Feature engineering completed in {processing_time:.2f}s")
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return data
    
    def _basic_feature_engineering(self, data: pd.DataFrame, feature_types: List[str]) -> pd.DataFrame:
        """Basic feature engineering fallback."""
        try:
            engineered_data = data.copy()
            
            for feature_type in feature_types:
                if feature_type == 'momentum':
                    # Simple momentum features
                    engineered_data['price_change'] = engineered_data['close'].pct_change()
                    engineered_data['momentum_5'] = engineered_data['close'].pct_change(5)
                    engineered_data['momentum_10'] = engineered_data['close'].pct_change(10)
                
                elif feature_type == 'volatility':
                    # Simple volatility features
                    engineered_data['volatility_5'] = engineered_data['close'].rolling(5).std()
                    engineered_data['volatility_10'] = engineered_data['close'].rolling(10).std()
                    engineered_data['high_low_ratio'] = engineered_data['high'] / engineered_data['low']
                
                elif feature_type == 'volume':
                    # Simple volume features
                    engineered_data['volume_change'] = engineered_data['volume'].pct_change()
                    engineered_data['volume_ma_5'] = engineered_data['volume'].rolling(5).mean()
                    engineered_data['volume_ma_10'] = engineered_data['volume'].rolling(10).mean()
                
                elif feature_type == 'technical':
                    # Simple technical indicators
                    engineered_data['sma_5'] = engineered_data['close'].rolling(5).mean()
                    engineered_data['sma_10'] = engineered_data['close'].rolling(10).mean()
                    engineered_data['rsi_14'] = self._calculate_rsi(engineered_data['close'], 14)
            
            return engineered_data
            
        except Exception as e:
            self.logger.error(f"❌ Basic feature engineering failed: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    # =============================================================================
    # RETURNS ENGINEERING
    # =============================================================================
    
    def engineer_returns(self, data: pd.DataFrame, return_types: List[str] = None) -> pd.DataFrame:
        """Engineer returns using enhanced data integration."""
        try:
            if return_types is None:
                return_types = ['simple', 'log', 'excess']
            
            start_time = time.time()
            returns_data = pd.DataFrame(index=data.index)
            
            # Use returns engineer if available
            if hasattr(self, 'returns_engineer'):
                for return_type in return_types:
                    try:
                        if return_type == 'simple':
                            returns = self.returns_engineer.calculate_simple_returns(data['close'])
                        elif return_type == 'log':
                            returns = self.returns_engineer.calculate_log_returns(data['close'])
                        elif return_type == 'excess':
                            returns = self.returns_engineer.calculate_excess_returns(data['close'])
                        else:
                            continue
                        
                        if not returns.empty:
                            returns_data[f'{return_type}_returns'] = returns
                            self.logger.info(f"✅ {return_type} returns engineered")
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to engineer {return_type} returns: {e}")
                        continue
            else:
                # Fallback to basic returns calculation
                returns_data = self._basic_returns_engineering(data, return_types)
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            
            self.logger.info(f"✅ Returns engineering completed in {processing_time:.2f}s")
            return returns_data
            
        except Exception as e:
            self.logger.error(f"❌ Returns engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _basic_returns_engineering(self, data: pd.DataFrame, return_types: List[str]) -> pd.DataFrame:
        """Basic returns engineering fallback."""
        try:
            returns_data = pd.DataFrame(index=data.index)
            
            for return_type in return_types:
                if return_type == 'simple':
                    returns_data['simple_returns'] = data['close'].pct_change()
                elif return_type == 'log':
                    returns_data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                elif return_type == 'excess':
                    simple_returns = data['close'].pct_change()
                    returns_data['excess_returns'] = simple_returns - simple_returns.mean()
            
            return returns_data
            
        except Exception as e:
            self.logger.error(f"❌ Basic returns engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    # =============================================================================
    # DATA QUALITY ANALYSIS
    # =============================================================================
    
    def calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        try:
            start_time = time.time()
            
            # Use utility integration for basic quality metrics
            basic_metrics = self.utility_integration.calculate_data_quality_metrics(data)
            
            # Use advanced quality metrics if available
            if hasattr(self, 'advanced_quality_metrics'):
                advanced_metrics = self.advanced_quality_metrics.calculate_metrics(data)
            else:
                advanced_metrics = {}
            
            # Use comprehensive quality scorer if available
            if hasattr(self, 'comprehensive_quality_scorer'):
                quality_score = self.comprehensive_quality_scorer.score_data_quality(data)
            else:
                quality_score = self._calculate_basic_quality_score(basic_metrics)
            
            # Combine metrics
            quality_metrics = {
                'basic_metrics': basic_metrics,
                'advanced_metrics': advanced_metrics,
                'quality_score': quality_score,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Data quality metrics calculated: score={quality_score:.3f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Data quality metrics calculation failed: {e}")
            return {'error': str(e), 'quality_score': 0.0}
    
    def _calculate_basic_quality_score(self, basic_metrics: Dict[str, Any]) -> float:
        """Calculate basic quality score from metrics."""
        try:
            # Weight different quality aspects
            completeness_score = 1.0 - (basic_metrics.get('missing_percentage', 0) / 100)
            uniqueness_score = 1.0 - (basic_metrics.get('duplicate_percentage', 0) / 100)
            
            # Combine scores
            quality_score = (completeness_score + uniqueness_score) / 2
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    def validate_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data consistency using enhanced checks."""
        try:
            consistency_results = {
                'is_consistent': True,
                'issues': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Check for basic consistency issues
            if 'high' in data.columns and 'low' in data.columns:
                invalid_hl = (data['high'] < data['low']).sum()
                if invalid_hl > 0:
                    consistency_results['issues'].append(f"High < Low in {invalid_hl} rows")
                    consistency_results['is_consistent'] = False
            
            if 'open' in data.columns and 'close' in data.columns:
                # Check for extreme price movements
                price_changes = data['close'].pct_change().abs()
                extreme_moves = (price_changes > 0.5).sum()
                if extreme_moves > 0:
                    consistency_results['warnings'].append(f"Extreme price movements in {extreme_moves} rows")
            
            # Check for data gaps
            if 'timestamp' in data.columns:
                time_diffs = pd.to_datetime(data['timestamp']).diff()
                expected_freq = pd.Timedelta(minutes=15)  # Assuming 15m data
                irregular_gaps = (time_diffs != expected_freq).sum()
                if irregular_gaps > 0:
                    consistency_results['warnings'].append(f"Irregular time gaps in {irregular_gaps} rows")
            
            # Check for missing values in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            for col in critical_columns:
                if col in data.columns:
                    missing_count = data[col].isnull().sum()
                    if missing_count > 0:
                        consistency_results['issues'].append(f"Missing values in {col}: {missing_count}")
                        consistency_results['is_consistent'] = False
            
            # Generate recommendations
            if not consistency_results['is_consistent']:
                consistency_results['recommendations'].append("Review and clean data before processing")
            if consistency_results['warnings']:
                consistency_results['recommendations'].append("Consider investigating warnings")
            
            self.logger.info(f"✅ Data consistency validation completed: {consistency_results['is_consistent']}")
            return consistency_results
            
        except Exception as e:
            self.logger.error(f"❌ Data consistency validation failed: {e}")
            return {'is_consistent': False, 'issues': [str(e)], 'warnings': [], 'recommendations': []}
    
    # =============================================================================
    # DATA STORAGE AND OPTIMIZATION
    # =============================================================================
    
    def save_optimized_data(self, data: pd.DataFrame, file_path: Union[str, Path], 
                          compression: str = "snappy", optimize: bool = True) -> bool:
        """Save data with optimization."""
        try:
            if self.config.enable_optimized_storage and hasattr(self, 'optimized_storage'):
                success = self.optimized_storage.save_data(data, file_path, compression=compression, optimize=optimize)
            else:
                # Fallback to utility integration
                success = self.utility_integration.safe_to_parquet(data, file_path, compression=compression)
            
            if success:
                self.logger.info(f"✅ Data saved successfully to {file_path}")
            else:
                self.logger.warning(f"⚠️ Failed to save data to {file_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Data saving failed: {e}")
            return False
    
    def load_optimized_data(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Load optimized data."""
        try:
            if self.config.enable_optimized_storage and hasattr(self, 'optimized_storage'):
                data = self.optimized_storage.load_data(file_path)
            else:
                # Fallback to utility integration
                data = self.utility_integration.safe_read_parquet(file_path)
            
            if data is not None:
                self.logger.info(f"✅ Data loaded successfully from {file_path}")
            else:
                self.logger.warning(f"⚠️ Failed to load data from {file_path}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {e}")
            return None
    
    # =============================================================================
    # PERFORMANCE MONITORING
    # =============================================================================
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            metrics = {
                'processing_times': {
                    'mean': np.mean(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0,
                    'std': np.std(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0,
                    'min': np.min(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0,
                    'max': np.max(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0,
                    'count': len(self.performance_metrics['processing_times'])
                },
                'data_quality_scores': {
                    'mean': np.mean([score.get('quality_score', 0) for score in self.performance_metrics['data_quality_scores']]) if self.performance_metrics['data_quality_scores'] else 0,
                    'std': np.std([score.get('quality_score', 0) for score in self.performance_metrics['data_quality_scores']]) if self.performance_metrics['data_quality_scores'] else 0,
                    'count': len(self.performance_metrics['data_quality_scores'])
                },
                'processing_errors': {
                    'count': len(self.performance_metrics['processing_errors']),
                    'errors': self.performance_metrics['processing_errors']
                },
                'memory_usage': self.utility_integration.get_memory_usage() if hasattr(self.utility_integration, 'get_memory_usage') else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def get_available_data_utilities(self) -> List[str]:
        """Get list of available data utilities."""
        utilities = []
        
        if self.config.enable_klines_parquet and hasattr(self, 'klines_processor'):
            utilities.append('klines_parquet_processing')
        
        if self.config.enable_unified_data_utils and hasattr(self, 'unified_data_utils'):
            utilities.append('unified_data_utils')
        
        if self.config.enable_historical_downloader and hasattr(self, 'historical_downloader'):
            utilities.append('historical_data_downloader')
        
        if self.config.enable_feature_engineering and hasattr(self, 'feature_engineer'):
            utilities.append('feature_engineering')
        
        if self.config.enable_returns_engineering and hasattr(self, 'returns_engineer'):
            utilities.append('returns_engineering')
        
        if self.config.enable_gap_detection and hasattr(self, 'gap_detector'):
            utilities.append('gap_detection')
        
        if self.config.enable_data_quality and hasattr(self, 'data_quality_analyzer'):
            utilities.append('data_quality_analysis')
        
        if self.config.enable_advanced_quality_metrics and hasattr(self, 'advanced_quality_metrics'):
            utilities.append('advanced_quality_metrics')
        
        if self.config.enable_comprehensive_quality_scoring and hasattr(self, 'comprehensive_quality_scorer'):
            utilities.append('comprehensive_quality_scoring')
        
        if self.config.enable_optimized_storage and hasattr(self, 'optimized_storage'):
            utilities.append('optimized_storage')
        
        return utilities
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and available utilities."""
        return {
            'config': self.config.__dict__,
            'available_utilities': self.get_available_data_utilities(),
            'performance_metrics': self.get_performance_metrics(),
            'data_components': {
                'klines_processor': hasattr(self, 'klines_processor'),
                'unified_data_utils': hasattr(self, 'unified_data_utils'),
                'historical_downloader': hasattr(self, 'historical_downloader'),
                'feature_engineer': hasattr(self, 'feature_engineer'),
                'returns_engineer': hasattr(self, 'returns_engineer'),
                'gap_detector': hasattr(self, 'gap_detector'),
                'data_quality_analyzer': hasattr(self, 'data_quality_analyzer'),
                'advanced_quality_metrics': hasattr(self, 'advanced_quality_metrics'),
                'comprehensive_quality_scorer': hasattr(self, 'comprehensive_quality_scorer'),
                'optimized_storage': hasattr(self, 'optimized_storage'),
                'data_processor': hasattr(self, 'data_processor'),
                'data_transformer': hasattr(self, 'data_transformer')
            }
        }


def create_enhanced_data_integration(config: DataIntegrationConfig = None, 
                                   utility_integration: EnhancedUtilityIntegration = None) -> EnhancedDataIntegration:
    """Create an enhanced data integration instance."""
    if config is None:
        config = DataIntegrationConfig()
    
    return EnhancedDataIntegration(config, utility_integration)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
