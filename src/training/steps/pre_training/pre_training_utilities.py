"""
Pre-Training Utilities and Abstractions

This module provides standardized utilities and abstractions for pre-training steps,
leveraging BaseStep's comprehensive tool suite to eliminate code duplication and
ensure consistent patterns across all feature generation and pre-training operations.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import gc
import time

from src.training.steps.base_step import BaseStep

# ============================================================================
# Pre-Training Specific Data Structures
# ============================================================================

@dataclass
class PreTrainingConfig:
    """Standardized configuration for pre-training steps."""
    symbol: str = 'ETHUSDT'
    exchange: str = 'binance'
    timeframe: str = '15m'
    direction: str = 'long'
    model: str = 'Analyst'
    lookback_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    enable_hardware_optimization: bool = True
    enable_data_preview: bool = True
    enable_memory_monitoring: bool = True
    chunk_size: int = 10000
    max_memory_usage: float = 0.8
    quality_threshold: float = 0.7

@dataclass
class FeatureGenerationResult:
    """Standardized result structure for feature generation operations."""
    success: bool
    features: pd.DataFrame
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    generation_metrics: Dict[str, Any]
    optimization_stats: Dict[str, Any]
    quality_score: float
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

@dataclass
class DataValidationResult:
    """Standardized result structure for data validation operations."""
    success: bool
    quality_score: float
    quality_level: str
    validation_metadata: Dict[str, Any]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

@dataclass
class OptimizationResult:
    """Standardized result structure for optimization operations."""
    success: bool
    optimized_data: pd.DataFrame
    optimization_metrics: Dict[str, Any]
    memory_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

# ============================================================================
# Pre-Training Utility Mixin
# ============================================================================

class PreTrainingUtilitiesMixin:
    """Mixin class providing standardized utilities for pre-training steps."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_training_config: Optional[PreTrainingConfig] = None
        self._feature_generation_cache: Dict[str, Any] = {}
        self._validation_cache: Dict[str, Any] = {}
    
    # ========================================================================
    # Configuration Management
    # ========================================================================
    
    def _initialize_pre_training_config(self, config: Dict[str, Any]) -> PreTrainingConfig:
        """Initialize standardized pre-training configuration."""
        self.tprint_operation_start("⚙️ Initializing pre-training configuration")
        
        pre_training_config = PreTrainingConfig(
            symbol=config.get('symbol', 'ETHUSDT'),
            exchange=config.get('exchange', 'binance'),
            timeframe=config.get('timeframe', '15m'),
            direction=config.get('direction', 'long'),
            model=config.get('model', 'Analyst'),
            lookback_days=config.get('lookback_days'),
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            enable_hardware_optimization=config.get('enable_hardware_optimization', True),
            enable_data_preview=config.get('enable_data_preview', True),
            enable_memory_monitoring=config.get('enable_memory_monitoring', True),
            chunk_size=config.get('chunk_size', 10000),
            max_memory_usage=config.get('max_memory_usage', 0.8),
            quality_threshold=config.get('quality_threshold', 0.7)
        )
        
        self._pre_training_config = pre_training_config
        self.tprint_operation_end("✅ Pre-training configuration initialized")
        
        return pre_training_config
    
    def _get_pre_training_config(self) -> PreTrainingConfig:
        """Get the current pre-training configuration."""
        if self._pre_training_config is None:
            raise ValueError("Pre-training configuration not initialized. Call _initialize_pre_training_config first.")
        return self._pre_training_config
    
    # ========================================================================
    # Data Loading and Validation
    # ========================================================================
    
    async def _load_data_standardized(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load data using standardized pre-training patterns."""
        self.tprint_operation_start("📊 Loading data with standardized patterns")
        
        pre_config = self._initialize_pre_training_config(config)
        
        # Set context for enhanced operations
        self._set_context(
            symbol=pre_config.symbol,
            exchange=pre_config.exchange,
            direction=pre_config.direction,
            model=pre_config.model
        )
        
        # Load data using BaseStep's klines integration
        data = self._load_klines_with_context(pre_config.timeframe)
        
        if data is None or data.empty:
            # Try fallback loading methods
            data = await self._load_data_fallback(pre_config)
        
        if data is None or data.empty:
            raise ValueError(f"No data found for {pre_config.symbol} {pre_config.timeframe}")
        
        # Apply date filtering if specified
        if pre_config.lookback_days or pre_config.start_date or pre_config.end_date:
            data = self._apply_date_filtering(data, pre_config)
        
        # Validate data using BaseStep utilities
        self._validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume'])
        
        # Use BaseStep's data preview
        if pre_config.enable_data_preview:
            self.tprint_data_preview(data, f"loaded_data_{pre_config.symbol}_{pre_config.timeframe}", max_rows=5)
        
        self.tprint_operation_end("✅ Data loaded and validated")
        return data
    
    async def _load_data_fallback(self, config: PreTrainingConfig) -> Optional[pd.DataFrame]:
        """Fallback data loading methods."""
        self.tprint_warning("⚠️ Primary data loading failed, trying fallback methods")
        
        try:
            # Try loading from consolidated files
            consolidated_path = f"historical_data/features_{config.exchange}_{config.symbol}_consolidated.parquet"
            if self._safe_file_exists(consolidated_path):
                self.tprint_info(f"📁 Loading from consolidated file: {consolidated_path}")
                return self._safe_read_parquet(consolidated_path)
            
            # Try loading from 1m consolidated file
            consolidated_1m_path = f"historical_data/{config.exchange}/{config.symbol.lower()}/processed/{config.symbol.lower()}_1m/features_{config.symbol.lower()}_1m_consolidated.parquet"
            if self._safe_file_exists(consolidated_1m_path):
                self.tprint_info(f"📁 Loading from 1m consolidated file: {consolidated_1m_path}")
                return self._safe_read_parquet(consolidated_1m_path)
            
            return None
            
        except Exception as e:
            self.tprint_error(f"❌ Fallback data loading failed: {e}")
            return None
    
    def _apply_date_filtering(self, data: pd.DataFrame, config: PreTrainingConfig) -> pd.DataFrame:
        """Apply date filtering to data."""
        if 'timestamp' not in data.columns or len(data) == 0:
            return data
        
        # Get the actual data range
        data_start = data['timestamp'].min()
        data_end = data['timestamp'].max()
        
        self.tprint_info(f"📊 Data range: {data_start} to {data_end}")
        
        # Apply lookback days filter
        if config.lookback_days and config.lookback_days > 0:
            end_date = data_end
            start_date = end_date - pd.Timedelta(days=config.lookback_days)
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
            self.tprint_info(f"📊 Applied lookback filter: {start_date} to {end_date} ({config.lookback_days} days)")
        
        # Apply specific date range filter
        elif config.start_date or config.end_date:
            if config.start_date:
                start_dt = pd.to_datetime(config.start_date, utc=True)
                data = data[data['timestamp'] >= start_dt]
            if config.end_date:
                end_dt = pd.to_datetime(config.end_date, utc=True)
                data = data[data['timestamp'] <= end_dt]
            self.tprint_info(f"📊 Applied date filters: {config.start_date} to {config.end_date}")
        
        # Use default 30-day window if no filters specified
        else:
            end_date = data_end
            start_date = end_date - pd.Timedelta(days=30)
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
            self.tprint_info(f"📊 Using default 30-day window: {start_date} to {end_date}")
        
        return data
    
    async def _validate_data_standardized(self, data: pd.DataFrame, config: PreTrainingConfig) -> DataValidationResult:
        """Validate data using standardized pre-training patterns."""
        self.tprint_operation_start("🔍 Validating data with standardized patterns")
        
        try:
            # Basic validation checks
            basic_checks = {
                'has_data': not len(data) == 0,
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_all_nan': not data.isnull().all().any(),
                'sufficient_rows': len(data) >= 100
            }
            
            success = all(basic_checks.values())
            quality_score = sum(basic_checks.values()) / len(basic_checks) * 100
            
            # Determine quality level
            if quality_score >= 90:
                quality_level = "excellent"
            elif quality_score >= 75:
                quality_level = "good"
            elif quality_score >= 60:
                quality_level = "fair"
            elif quality_score >= 40:
                quality_level = "poor"
            else:
                quality_level = "critical"
            
            # Generate recommendations
            recommendations = []
            if not basic_checks['has_data']:
                recommendations.append("Ensure data is loaded correctly")
            if not basic_checks['has_required_columns']:
                recommendations.append("Verify data contains required OHLCV columns")
            if not basic_checks['sufficient_rows']:
                recommendations.append("Increase data size or adjust lookback period")
            
            # Use BaseStep's data preview
            if config.enable_data_preview:
                self.tprint_data_preview(data, "validated_data", max_rows=5)
            
            result = DataValidationResult(
                success=success,
                quality_score=quality_score,
                quality_level=quality_level,
                validation_metadata={
                    'basic_checks': basic_checks,
                    'method': 'standardized_pre_training'
                },
                issues=[] if success else ['Basic validation failed'],
                warnings=[],
                recommendations=recommendations,
                artifacts=['validated_dataframe']
            )
            
            self.tprint_operation_end("✅ Data validation completed")
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Data validation failed: {e}")
            return DataValidationResult(
                success=False,
                quality_score=0.0,
                quality_level="error",
                validation_metadata={'error': str(e)},
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Check data format and try again"],
                error_message=str(e)
            )
    
    # ========================================================================
    # Feature Generation
    # ========================================================================
    
    async def _generate_features_standardized(self, data: pd.DataFrame, config: PreTrainingConfig) -> FeatureGenerationResult:
        """Generate features using standardized pre-training patterns."""
        self.tprint_operation_start("🔧 Generating features with standardized patterns")
        
        try:
            # Check cache first
            cache_key = f"{config.symbol}_{config.timeframe}_{config.direction}"
            if cache_key in self._feature_generation_cache:
                self.tprint_info("📋 Using cached feature generation result")
                return self._feature_generation_cache[cache_key]
            
            # Generate features using BaseStep's hardware optimization
            if config.enable_hardware_optimization:
                features = await self._generate_features_with_hardware_optimization(data, config)
            else:
                features = await self._generate_features_basic(data, config)
            
            # Validate generated features
            feature_validation = await self._validate_features_standardized(features, config)
            
            if not feature_validation.success:
                raise ValueError(f"Feature validation failed: {feature_validation.error_message}")
            
            # Generate feature categories
            feature_categories = self._categorize_features(features.columns.tolist())
            
            # Calculate generation metrics
            generation_metrics = {
                'feature_count': len(features.columns),
                'row_count': len(features),
                'generation_timestamp': self._get_current_datetime(),
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'direction': config.direction,
                'model': config.model
            }
            
            # Calculate optimization stats
            optimization_stats = {
                'memory_usage': self._get_memory_analytics(),
                'performance_metrics': self._get_performance_metrics(),
                'hardware_stats': self._get_hardware_stats()
            }
            
            result = FeatureGenerationResult(
                success=True,
                features=features,
                feature_names=features.columns.tolist(),
                feature_categories=feature_categories,
                generation_metrics=generation_metrics,
                optimization_stats=optimization_stats,
                quality_score=feature_validation.quality_score,
                artifacts=['generated_features', 'feature_metadata']
            )
            
            # Cache the result
            self._feature_generation_cache[cache_key] = result
            
            self.tprint_operation_end("✅ Features generated successfully")
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Feature generation failed: {e}")
            return FeatureGenerationResult(
                success=False,
                features=pd.DataFrame(),
                feature_names=[],
                feature_categories={},
                generation_metrics={},
                optimization_stats={},
                quality_score=0.0,
                error_message=str(e)
            )
    
    async def _generate_features_with_hardware_optimization(self, data: pd.DataFrame, config: PreTrainingConfig) -> pd.DataFrame:
        """Generate features with hardware optimization."""
        self.tprint_info("⚙️ Using hardware optimization for feature generation")
        
        # Use BaseStep's hardware optimization
        optimized_data = self._optimize_dataframe_with_hardware(data)
        
        # Generate features in chunks for memory efficiency
        chunk_size = config.chunk_size
        features_list = []
        
        for i in range(0, len(optimized_data), chunk_size):
            chunk = optimized_data.iloc[i:i + chunk_size]
            
            # Process chunk with hardware optimization
            chunk_features = await self._process_chunk_with_hardware_optimization(chunk, config)
            features_list.append(chunk_features)
            
            # Monitor memory usage
            if config.enable_memory_monitoring:
                self._monitor_memory_usage()
        
        # Combine all chunks
        combined_features = pd.concat(features_list, ignore_index=True)
        
        # Final optimization
        final_features = self._optimize_dataframe_with_hardware(combined_features)
        
        return final_features
    
    async def _generate_features_basic(self, data: pd.DataFrame, config: PreTrainingConfig) -> pd.DataFrame:
        """Generate features using basic methods."""
        self.tprint_info("🔧 Using basic feature generation")
        
        # Basic feature generation logic
        features = data.copy()
        
        # Add basic technical indicators
        features['sma_20'] = features['close'].rolling(window=20).mean()
        features['sma_50'] = features['close'].rolling(window=50).mean()
        features['rsi_14'] = self._calculate_rsi(features['close'], 14)
        features['bb_upper'] = features['sma_20'] + (features['close'].rolling(window=20).std() * 2)
        features['bb_lower'] = features['sma_20'] - (features['close'].rolling(window=20).std() * 2)
        
        # Add price-based features
        features['price_change'] = features['close'].pct_change()
        features['volume_change'] = features['volume'].pct_change()
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        return features
    
    async def _process_chunk_with_hardware_optimization(self, chunk: pd.DataFrame, config: PreTrainingConfig) -> pd.DataFrame:
        """Process a chunk of data with hardware optimization."""
        # Use BaseStep's hardware optimization decorators
        @self.memory_optimized
        @self.cpu_optimized
        def process_chunk_internal(data_chunk):
            # Basic feature generation for chunk
            features = data_chunk.copy()
            
            # Add technical indicators
            features['sma_20'] = features['close'].rolling(window=20).mean()
            features['rsi_14'] = self._calculate_rsi(features['close'], 14)
            features['price_change'] = features['close'].pct_change()
            
            return features
        
        return process_chunk_internal(chunk)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features into logical groups."""
        categories = {
            'price_features': [],
            'volume_features': [],
            'technical_indicators': [],
            'statistical_features': [],
            'other_features': []
        }
        
        for feature in feature_names:
            if any(keyword in feature.lower() for keyword in ['open', 'high', 'low', 'close', 'price']):
                categories['price_features'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['volume', 'vol']):
                categories['volume_features'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['sma', 'ema', 'rsi', 'bb', 'macd', 'stoch']):
                categories['technical_indicators'].append(feature)
            elif any(keyword in feature.lower() for keyword in ['std', 'var', 'mean', 'median', 'skew', 'kurt']):
                categories['statistical_features'].append(feature)
            else:
                categories['other_features'].append(feature)
        
        return categories
    
    async def _validate_features_standardized(self, features: pd.DataFrame, config: PreTrainingConfig) -> DataValidationResult:
        """Validate features using standardized patterns."""
        try:
            # Basic validation
            basic_checks = {
                'has_features': len(features.columns) > 0,
                'has_data': len(features) > 0,
                'no_all_nan': not features.isnull().all().any(),
                'sufficient_features': len(features.columns) >= 5
            }
            
            success = all(basic_checks.values())
            quality_score = sum(basic_checks.values()) / len(basic_checks) * 100
            
            # Determine quality level
            if quality_score >= 90:
                quality_level = "excellent"
            elif quality_score >= 75:
                quality_level = "good"
            elif quality_score >= 60:
                quality_level = "fair"
            else:
                quality_level = "poor"
            
            return DataValidationResult(
                success=success,
                quality_score=quality_score,
                quality_level=quality_level,
                validation_metadata={'basic_checks': basic_checks},
                issues=[] if success else ['Feature validation failed'],
                recommendations=[] if success else ['Check feature generation logic']
            )
            
        except Exception as e:
            return DataValidationResult(
                success=False,
                quality_score=0.0,
                quality_level="error",
                validation_metadata={'error': str(e)},
                issues=[f"Feature validation error: {str(e)}"],
                error_message=str(e)
            )
    
    # ========================================================================
    # Artifact Management
    # ========================================================================
    
    async def _save_artifacts_standardized(self, result: FeatureGenerationResult, config: PreTrainingConfig) -> List[str]:
        """Save artifacts using standardized patterns."""
        self.tprint_operation_start("💾 Saving artifacts with standardized patterns")
        
        artifacts = []
        
        try:
            # Save features
            self._save_dataframe(result.features, 'generated_features')
            artifacts.append('generated_features')
            
            # Save metadata
            metadata = {
                'feature_names': result.feature_names,
                'feature_categories': result.feature_categories,
                'generation_metrics': result.generation_metrics,
                'optimization_stats': result.optimization_stats,
                'quality_score': result.quality_score,
                'config': config.__dict__
            }
            self._save_metadata(metadata, 'feature_metadata')
            artifacts.append('feature_metadata')
            
            # Save individual feature categories
            for category, features in result.feature_categories.items():
                if features:
                    category_data = result.features[features]
                    self._save_dataframe(category_data, f'features_{category}')
                    artifacts.append(f'features_{category}')
            
            self.tprint_operation_end("✅ Artifacts saved successfully")
            return artifacts
            
        except Exception as e:
            self.tprint_error(f"❌ Artifact saving failed: {e}")
            return []
    
    # ========================================================================
    # Performance Monitoring
    # ========================================================================
    
    def _monitor_performance_standardized(self, operation_name: str, config: PreTrainingConfig):
        """Monitor performance using standardized patterns."""
        if not config.enable_memory_monitoring:
            return
        
        self.tprint_operation_start(f"📊 Monitoring performance for {operation_name}")
        
        # Get performance metrics
        performance_metrics = self._get_performance_metrics()
        memory_analytics = self._get_memory_analytics()
        hardware_stats = self._get_hardware_stats()
        
        # Log performance summary
        self.tprint_performance_summary({
            'operation': operation_name,
            'performance_metrics': performance_metrics,
            'memory_analytics': memory_analytics,
            'hardware_stats': hardware_stats
        })
        
        # Check memory usage
        memory_usage = memory_analytics.get('memory_usage_percent', 0)
        if memory_usage > config.max_memory_usage * 100:
            self.tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}%")
            self._aggressive_garbage_collection()
        
        self.tprint_operation_end(f"✅ Performance monitoring completed for {operation_name}")

# ============================================================================
# Pre-Training Step Base Class
# ============================================================================

class PreTrainingStepBase(BaseStep, PreTrainingUtilitiesMixin):
    """Base class for all pre-training steps with standardized utilities."""
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize pre-training step with comprehensive utilities."""
        super().__init__(step_name, config)
        
        # Initialize pre-training specific components
        self._pre_training_config: Optional[PreTrainingConfig] = None
        self._feature_generation_cache: Dict[str, Any] = {}
        self._validation_cache: Dict[str, Any] = {}
        
        self.tprint_info("🔧 Initializing pre-training step with standardized utilities")
        self.tprint_success("✅ Pre-training step initialized successfully")
    
    async def execute_standardized(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pre-training step using standardized patterns."""
        self.tprint_step_start("🚀 Starting standardized pre-training execution")
        
        try:
            # Initialize configuration
            pre_config = self._initialize_pre_training_config(config)
            
            # Load and validate data
            data = await self._load_data_standardized(config)
            validation_result = await self._validate_data_standardized(data, pre_config)
            
            if not validation_result.success:
                return {
                    'success': False,
                    'artifacts': [],
                    'metrics': validation_result.__dict__,
                    'error': validation_result.error_message
                }
            
            # Generate features
            feature_result = await self._generate_features_standardized(data, pre_config)
            
            if not feature_result.success:
                return {
                    'success': False,
                    'artifacts': [],
                    'metrics': feature_result.__dict__,
                    'error': feature_result.error_message
                }
            
            # Save artifacts
            artifacts = await self._save_artifacts_standardized(feature_result, pre_config)
            
            # Generate comprehensive report
            report = await self._generate_comprehensive_report_standardized(feature_result, pre_config)
            
            self.tprint_step_end("✅ Standardized pre-training completed successfully")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': {
                    'validation_result': validation_result.__dict__,
                    'feature_result': feature_result.__dict__,
                    'report': report
                }
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Standardized pre-training failed: {e}")
            self.tprint_exception(e)
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _generate_comprehensive_report_standardized(self, feature_result: FeatureGenerationResult, config: PreTrainingConfig) -> Dict[str, Any]:
        """Generate comprehensive report using standardized patterns."""
        self.tprint_operation_start("📊 Generating comprehensive report")
        
        # Get performance metrics
        performance_metrics = self._get_performance_metrics()
        memory_analytics = self._get_memory_analytics()
        hardware_stats = self._get_hardware_stats()
        
        # Generate report
        report = {
            'execution_summary': {
                'timestamp': self._get_current_datetime(),
                'step_name': self.step_name,
                'config': config.__dict__,
                'success': feature_result.success
            },
            'feature_summary': {
                'feature_count': len(feature_result.feature_names),
                'feature_categories': feature_result.feature_categories,
                'quality_score': feature_result.quality_score
            },
            'performance_summary': {
                'performance_metrics': performance_metrics,
                'memory_analytics': memory_analytics,
                'hardware_stats': hardware_stats
            },
            'artifacts': feature_result.artifacts
        }
        
        # Save report
        self._save_metadata(report, 'comprehensive_report')
        
        # Use BaseStep's structured logging
        self.tprint_execution_summary(report)
        
        self.tprint_operation_end("✅ Comprehensive report generated")
        return report

# ============================================================================
# Utility Functions
# ============================================================================

def create_pre_training_step(step_name: str, config: Optional[Dict[str, Any]] = None) -> PreTrainingStepBase:
    """Factory function to create a pre-training step with standardized utilities."""
    return PreTrainingStepBase(step_name, config)

def validate_pre_training_config(config: Dict[str, Any]) -> bool:
    """Validate pre-training configuration."""
    required_fields = ['symbol', 'exchange', 'timeframe']
    return all(field in config for field in required_fields)

def get_pre_training_defaults() -> Dict[str, Any]:
    """Get default configuration for pre-training steps."""
    return {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'model': 'Analyst',
        'enable_hardware_optimization': True,
        'enable_data_preview': True,
        'enable_memory_monitoring': True,
        'chunk_size': 10000,
        'max_memory_usage': 0.8,
        'quality_threshold': 0.7
    }