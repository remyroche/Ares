"""
Enhanced Data Integration for Hybrid NAS-TAS Regime System

This module integrates data utilities from src/utils/data/ for enhanced
data processing, quality control, and feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import time
from dataclasses import dataclass, field
from enum import Enum

# Import enhanced utility integration
from .enhanced_utility_integration import EnhancedUtilityIntegration, UtilityIntegrationConfig

# Import data utilities (conditional imports)
try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    KLINES_PARQUET_AVAILABLE = True
except ImportError:
    KLINES_PARQUET_AVAILABLE = False
    KlinesParquetManager = None

try:
    from src.utils.data.unified_data_utils import UnifiedDataUtils
    UNIFIED_DATA_UTILS_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_UTILS_AVAILABLE = False
    UnifiedDataUtils = None

try:
    from src.utils.data.feature_engineer import FeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    FeatureEngineer = None

try:
    from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
    BASIC_RETURNS_ENGINEER_AVAILABLE = True
except ImportError:
    BASIC_RETURNS_ENGINEER_AVAILABLE = False
    BasicReturnsEngineer = None

try:
    from src.utils.data.gap_detector import GapDetector
    GAP_DETECTOR_AVAILABLE = True
except ImportError:
    GAP_DETECTOR_AVAILABLE = False
    GapDetector = None

try:
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader
    HISTORICAL_DATA_DOWNLOADER_AVAILABLE = True
except ImportError:
    HISTORICAL_DATA_DOWNLOADER_AVAILABLE = False
    HistoricalDataDownloader = None

try:
    from src.utils.data.optimized_parquet_storage import OptimizedParquetStorage
    OPTIMIZED_PARQUET_STORAGE_AVAILABLE = True
except ImportError:
    OPTIMIZED_PARQUET_STORAGE_AVAILABLE = False
    OptimizedParquetStorage = None

# Import data quality utilities
try:
    from src.utils.data.quality.data_quality import DataQuality
    DATA_QUALITY_AVAILABLE = True
except ImportError:
    DATA_QUALITY_AVAILABLE = False
    DataQuality = None

try:
    from src.utils.data.quality.advanced_quality_metrics import AdvancedQualityMetrics
    ADVANCED_QUALITY_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_QUALITY_METRICS_AVAILABLE = False
    AdvancedQualityMetrics = None

try:
    from src.utils.data.quality.comprehensive_quality_scorer import ComprehensiveQualityScorer
    COMPREHENSIVE_QUALITY_SCORER_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_QUALITY_SCORER_AVAILABLE = False
    ComprehensiveQualityScorer = None

# Setup logging
logger = logging.getLogger(__name__)


class DataIntegrationStatus(Enum):
    """Status of data integration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class DataIntegrationConfig:
    """Configuration for data integration."""
    # Data sources
    enable_klines_parquet: bool = True
    enable_unified_data_utils: bool = True
    enable_historical_downloader: bool = True
    
    # Feature engineering
    enable_feature_engineering: bool = True
    enable_returns_engineering: bool = True
    enable_gap_detection: bool = True
    
    # Data quality
    enable_data_quality: bool = True
    enable_advanced_quality_metrics: bool = True
    enable_comprehensive_quality_scoring: bool = True
    
    # Storage optimization
    enable_optimized_storage: bool = True
    enable_parquet_optimization: bool = True
    
    # Performance
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    
    # Data validation
    enable_schema_validation: bool = True
    enable_data_consistency_checks: bool = True


class EnhancedDataIntegration:
    """
    Enhanced data integration manager for hybrid NAS-TAS regime system.
    
    This class integrates all available data utilities from src/utils/data/
    to provide enhanced data processing, quality control, and feature engineering.
    """
    
    def __init__(self, config: Optional[DataIntegrationConfig] = None, utility_config: Optional[UtilityIntegrationConfig] = None):
        """Initialize the enhanced data integration."""
        self.config = config or DataIntegrationConfig()
        self.utility_integration = EnhancedUtilityIntegration(utility_config)
        self.logger = logger.getChild('EnhancedDataIntegration')
        
        # Initialize integration status
        self.integration_status = self._check_integration_status()
        
        # Initialize data managers
        self._initialize_data_managers()
        
        self.logger.info("📊 Enhanced Data Integration initialized")
        self.logger.info(f"📈 Integration Status: {self.integration_status}")
    
    def _check_integration_status(self) -> Dict[str, DataIntegrationStatus]:
        """Check the status of all data integrations."""
        status = {}
        
        # Check data sources
        status['klines_parquet'] = DataIntegrationStatus.AVAILABLE if KLINES_PARQUET_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        status['unified_data_utils'] = DataIntegrationStatus.AVAILABLE if UNIFIED_DATA_UTILS_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        status['historical_downloader'] = DataIntegrationStatus.AVAILABLE if HISTORICAL_DATA_DOWNLOADER_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        
        # Check feature engineering
        status['feature_engineer'] = DataIntegrationStatus.AVAILABLE if FEATURE_ENGINEER_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        status['returns_engineer'] = DataIntegrationStatus.AVAILABLE if BASIC_RETURNS_ENGINEER_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        status['gap_detector'] = DataIntegrationStatus.AVAILABLE if GAP_DETECTOR_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        
        # Check data quality
        status['data_quality'] = DataIntegrationStatus.AVAILABLE if DATA_QUALITY_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        status['advanced_quality_metrics'] = DataIntegrationStatus.AVAILABLE if ADVANCED_QUALITY_METRICS_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        status['comprehensive_quality_scorer'] = DataIntegrationStatus.AVAILABLE if COMPREHENSIVE_QUALITY_SCORER_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        
        # Check storage
        status['optimized_storage'] = DataIntegrationStatus.AVAILABLE if OPTIMIZED_PARQUET_STORAGE_AVAILABLE else DataIntegrationStatus.UNAVAILABLE
        
        return status
    
    def _initialize_data_managers(self):
        """Initialize data managers."""
        # Initialize data sources
        if self.config.enable_klines_parquet and KLINES_PARQUET_AVAILABLE:
            self.klines_manager = KlinesParquetManager()
        else:
            self.klines_manager = None
            
        if self.config.enable_unified_data_utils and UNIFIED_DATA_UTILS_AVAILABLE:
            self.unified_data_utils = UnifiedDataUtils()
        else:
            self.unified_data_utils = None
            
        if self.config.enable_historical_downloader and HISTORICAL_DATA_DOWNLOADER_AVAILABLE:
            self.historical_downloader = HistoricalDataDownloader()
        else:
            self.historical_downloader = None
        
        # Initialize feature engineering
        if self.config.enable_feature_engineering and FEATURE_ENGINEER_AVAILABLE:
            self.feature_engineer = FeatureEngineer()
        else:
            self.feature_engineer = None
            
        if self.config.enable_returns_engineering and BASIC_RETURNS_ENGINEER_AVAILABLE:
            self.returns_engineer = BasicReturnsEngineer()
        else:
            self.returns_engineer = None
            
        if self.config.enable_gap_detection and GAP_DETECTOR_AVAILABLE:
            self.gap_detector = GapDetector()
        else:
            self.gap_detector = None
        
        # Initialize data quality
        if self.config.enable_data_quality and DATA_QUALITY_AVAILABLE:
            self.data_quality = DataQuality()
        else:
            self.data_quality = None
            
        if self.config.enable_advanced_quality_metrics and ADVANCED_QUALITY_METRICS_AVAILABLE:
            self.advanced_quality_metrics = AdvancedQualityMetrics()
        else:
            self.advanced_quality_metrics = None
            
        if self.config.enable_comprehensive_quality_scoring and COMPREHENSIVE_QUALITY_SCORER_AVAILABLE:
            self.quality_scorer = ComprehensiveQualityScorer()
        else:
            self.quality_scorer = None
        
        # Initialize storage
        if self.config.enable_optimized_storage and OPTIMIZED_PARQUET_STORAGE_AVAILABLE:
            self.optimized_storage = OptimizedParquetStorage()
        else:
            self.optimized_storage = None
    
    # =============================================================================
    # DATA LOADING AND PROCESSING
    # =============================================================================
    
    def load_klines_data(self, symbol: str, timeframe: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Load klines data using enhanced data utilities."""
        if self.klines_manager:
            try:
                data = self.klines_manager.get_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Apply data quality checks
                if self.config.enable_data_quality_checks:
                    quality_report = self.calculate_data_quality_metrics(data)
                    self.logger.info(f"📊 Data quality report: {quality_report}")
                
                return data
            except Exception as e:
                self.logger.error(f"❌ Error loading klines data: {e}")
                return None
        else:
            self.logger.warning("⚠️ Klines manager not available")
            return None
    
    def process_market_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Process market data with enhanced utilities."""
        if data is None or data.empty:
            self.logger.warning("⚠️ No data to process")
            return data
        
        # Apply data validation
        if self.config.enable_schema_validation:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not self.utility_integration.validate_dataframe_columns(data, required_columns):
                self.logger.error("❌ Data validation failed - missing required columns")
                return data
        
        # Optimize data types
        data = self.utility_integration.optimize_dataframe_dtypes(data)
        
        # Apply data quality checks
        if self.config.enable_data_quality_checks:
            quality_metrics = self.calculate_data_quality_metrics(data)
            self.logger.info(f"📊 Data quality metrics: {quality_metrics}")
        
        # Apply gap detection if enabled
        if self.config.enable_gap_detection and self.gap_detector:
            gaps = self.gap_detector.detect_gaps(data)
            if gaps:
                self.logger.warning(f"⚠️ Detected {len(gaps)} gaps in data")
        
        return data
    
    def calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        if self.quality_scorer:
            return self.quality_scorer.calculate_quality_score(data)
        elif self.advanced_quality_metrics:
            return self.advanced_quality_metrics.calculate_metrics(data)
        elif self.data_quality:
            return self.data_quality.assess_quality(data)
        else:
            # Fallback to utility integration
            return self.utility_integration.calculate_data_quality_metrics(data)
    
    # =============================================================================
    # FEATURE ENGINEERING
    # =============================================================================
    
    def engineer_features(self, data: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
        """Engineer features using enhanced feature engineering utilities."""
        if feature_types is None:
            feature_types = ['momentum', 'volatility', 'volume', 'trend']
        
        if self.feature_engineer:
            try:
                features = self.feature_engineer.create_features(data, feature_types)
                self.logger.info(f"✅ Engineered {len(features.columns)} features")
                return features
            except Exception as e:
                self.logger.error(f"❌ Error engineering features: {e}")
                return data
        else:
            self.logger.warning("⚠️ Feature engineer not available")
            return data
    
    def engineer_returns(self, data: pd.DataFrame, return_types: List[str] = None) -> pd.DataFrame:
        """Engineer returns using enhanced returns engineering utilities."""
        if return_types is None:
            return_types = ['simple', 'log', 'normalized']
        
        if self.returns_engineer:
            try:
                returns = self.returns_engineer.create_returns(data, return_types)
                self.logger.info(f"✅ Engineered {len(returns.columns)} return features")
                return returns
            except Exception as e:
                self.logger.error(f"❌ Error engineering returns: {e}")
                return data
        else:
            self.logger.warning("⚠️ Returns engineer not available")
            return data
    
    def detect_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect gaps in data using enhanced gap detection utilities."""
        if self.gap_detector:
            try:
                gaps = self.gap_detector.detect_gaps(data)
                self.logger.info(f"🔍 Detected {len(gaps)} gaps in data")
                return gaps
            except Exception as e:
                self.logger.error(f"❌ Error detecting gaps: {e}")
                return []
        else:
            self.logger.warning("⚠️ Gap detector not available")
            return []
    
    # =============================================================================
    # DATA STORAGE AND OPTIMIZATION
    # =============================================================================
    
    def save_optimized_data(self, data: pd.DataFrame, filepath: str, optimization_level: str = "medium") -> bool:
        """Save data with optimization using enhanced storage utilities."""
        if self.optimized_storage:
            try:
                success = self.optimized_storage.save_data(data, filepath, optimization_level)
                if success:
                    self.logger.info(f"✅ Data saved with optimization: {filepath}")
                return success
            except Exception as e:
                self.logger.error(f"❌ Error saving optimized data: {e}")
                return False
        else:
            # Fallback to utility integration
            return self.utility_integration.save_parquet(data, filepath)
    
    def load_optimized_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load data with optimization using enhanced storage utilities."""
        if self.optimized_storage:
            try:
                data = self.optimized_storage.load_data(filepath)
                if data is not None:
                    self.logger.info(f"✅ Data loaded with optimization: {filepath}")
                return data
            except Exception as e:
                self.logger.error(f"❌ Error loading optimized data: {e}")
                return None
        else:
            # Fallback to utility integration
            return self.utility_integration.load_parquet(filepath)
    
    # =============================================================================
    # DATA VALIDATION AND CONSISTENCY
    # =============================================================================
    
    def validate_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data consistency using enhanced validation utilities."""
        validation_results = {
            'is_consistent': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.1:
            validation_results['issues'].append(f"High missing value ratio: {missing_ratio:.2%}")
            validation_results['is_consistent'] = False
        
        # Check for duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > 0.05:
            validation_results['warnings'].append(f"High duplicate ratio: {duplicate_ratio:.2%}")
        
        # Check for outliers (basic check)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_ratio = outliers / len(data)
                if outlier_ratio > 0.1:
                    validation_results['warnings'].append(f"High outlier ratio in {col}: {outlier_ratio:.2%}")
        
        # Check data types
        for col in data.columns:
            if data[col].dtype == 'object' and col not in ['timestamp', 'datetime']:
                validation_results['recommendations'].append(f"Consider converting {col} to numeric type")
        
        return validation_results
    
    def clean_data(self, data: pd.DataFrame, cleaning_options: Dict[str, Any] = None) -> pd.DataFrame:
        """Clean data using enhanced cleaning utilities."""
        if cleaning_options is None:
            cleaning_options = {
                'remove_duplicates': True,
                'fill_missing': True,
                'remove_outliers': False,
                'normalize_types': True
            }
        
        cleaned_data = data.copy()
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', True):
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            if removed_duplicates > 0:
                self.logger.info(f"🧹 Removed {removed_duplicates} duplicate rows")
        
        # Fill missing values
        if cleaning_options.get('fill_missing', True):
            for col in cleaned_data.columns:
                if cleaned_data[col].dtype in ['int64', 'float64']:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                else:
                    cleaned_data[col] = cleaned_data[col].fillna('')
        
        # Remove outliers (optional)
        if cleaning_options.get('remove_outliers', False):
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                cleaned_data = cleaned_data[~((cleaned_data[col] < (Q1 - 1.5 * IQR)) | 
                                             (cleaned_data[col] > (Q3 + 1.5 * IQR)))]
        
        # Normalize data types
        if cleaning_options.get('normalize_types', True):
            cleaned_data = self.utility_integration.optimize_dataframe_dtypes(cleaned_data)
        
        self.logger.info(f"✅ Data cleaned: {len(data)} -> {len(cleaned_data)} rows")
        return cleaned_data
    
    # =============================================================================
    # DATA ANALYSIS AND INSIGHTS
    # =============================================================================
    
    def analyze_data_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns using enhanced analysis utilities."""
        analysis = {
            'basic_stats': {},
            'patterns': {},
            'insights': [],
            'recommendations': []
        }
        
        # Basic statistics
        analysis['basic_stats'] = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # Pattern analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis['patterns']['correlation_matrix'] = data[numeric_columns].corr().to_dict()
            analysis['patterns']['descriptive_stats'] = data[numeric_columns].describe().to_dict()
        
        # Time series patterns (if applicable)
        if 'timestamp' in data.columns or 'datetime' in data.columns:
            time_col = 'timestamp' if 'timestamp' in data.columns else 'datetime'
            if pd.api.types.is_datetime64_any_dtype(data[time_col]):
                analysis['patterns']['time_series_info'] = {
                    'start_date': data[time_col].min(),
                    'end_date': data[time_col].max(),
                    'frequency': pd.infer_freq(data[time_col]),
                    'gaps': self.detect_gaps(data)
                }
        
        # Generate insights
        if analysis['basic_stats']['missing_values']:
            missing_columns = [col for col, count in analysis['basic_stats']['missing_values'].items() if count > 0]
            if missing_columns:
                analysis['insights'].append(f"Missing values found in columns: {missing_columns}")
        
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                analysis['insights'].append(f"High correlation found between {len(high_corr_pairs)} pairs of variables")
        
        return analysis
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_integration_status(self) -> Dict[str, DataIntegrationStatus]:
        """Get the status of all data integrations."""
        return self.integration_status
    
    def get_available_data_utilities(self) -> List[str]:
        """Get list of available data utilities."""
        available = []
        for utility, status in self.integration_status.items():
            if status == DataIntegrationStatus.AVAILABLE:
                available.append(utility)
        return available
    
    def get_unavailable_data_utilities(self) -> List[str]:
        """Get list of unavailable data utilities."""
        unavailable = []
        for utility, status in self.integration_status.items():
            if status == DataIntegrationStatus.UNAVAILABLE:
                unavailable.append(utility)
        return unavailable
    
    def cleanup_data_resources(self) -> bool:
        """Clean up data resources."""
        try:
            if self.optimized_storage and hasattr(self.optimized_storage, 'cleanup'):
                self.optimized_storage.cleanup()
            
            self.logger.info("🧹 Data resources cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error during data cleanup: {e}")
            return False


# Factory function for easy initialization
def create_enhanced_data_integration(
    config: Optional[DataIntegrationConfig] = None,
    utility_config: Optional[UtilityIntegrationConfig] = None
) -> EnhancedDataIntegration:
    """Create an enhanced data integration instance."""
    return EnhancedDataIntegration(config, utility_config)


# Convenience functions for common data operations
def load_klines_data_enhanced(symbol: str, timeframe: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """Enhanced klines data loading."""
    integration = create_enhanced_data_integration()
    return integration.load_klines_data(symbol, timeframe, start_date, end_date)


def process_market_data_enhanced(data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Enhanced market data processing."""
    integration = create_enhanced_data_integration()
    return integration.process_market_data(data, symbol, timeframe)


def engineer_features_enhanced(data: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
    """Enhanced feature engineering."""
    integration = create_enhanced_data_integration()
    return integration.engineer_features(data, feature_types)


def calculate_data_quality_enhanced(data: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced data quality calculation."""
    integration = create_enhanced_data_integration()
    return integration.calculate_data_quality_metrics(data)