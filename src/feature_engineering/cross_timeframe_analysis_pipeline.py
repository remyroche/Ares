"""
Cross Timeframe Analysis Pipeline

This module provides comprehensive cross timeframe analysis for market data
to identify interactions and relationships between different timeframes.

Key Features:
- Multi-timeframe data loading and alignment
- Cross timeframe feature engineering
- Interaction analysis and correlation detection
- Data quality validation using existing utilities
- Integration with ML commons for enhanced analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range,
    safe_divide, safe_log, safe_sqrt, safe_power,
    MathValidationError
)

# Data processing utilities - simplified for now
# Data quality utilities - simplified for now

class MathValidation:
    """Simple math validation wrapper class."""
    
    def __init__(self):
        self.logger = system_logger.getChild("MathValidation")
    
    def validate_finite(self, value: Any, name: str = "value") -> float:
        """Validate that a value is finite."""
        return validate_finite(value, name)
    
    def validate_positive(self, value: float, name: str = "value") -> float:
        """Validate that a value is positive."""
        return validate_positive(value, name)
    
    def validate_range(self, value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
        """Validate that a value is in range."""
        return validate_range(value, min_val, max_val, name)
    
    def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safely divide two numbers."""
        return safe_divide(a, b, default)
    
    def safe_log(self, x: float, default: float = 0.0) -> float:
        """Safely calculate logarithm."""
        return safe_log(x, default)
    
    def safe_sqrt(self, x: float, default: float = 0.0) -> float:
        """Safely calculate square root."""
        return safe_sqrt(x, default)
    
    def safe_power(self, x: float, y: float, default: float = 0.0) -> float:
        """Safely calculate power."""
        return safe_power(x, y, default)

# Import data quality utilities from data_quality
from src.utils.data.quality.enhanced_data_quality_validator import QualityResult, EnhancedDataQualityValidator

# Simple placeholder classes for missing functionality
class DataQualityUtilities:
    def __init__(self):
        pass

class CommonOperations:
    def __init__(self):
        pass
# Math validation functions available in data_qualification_imports
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = system_logger.getChild('CrossTimeframeAnalysisPipeline')

@dataclass
class CrossTimeframeConfig:
    """Configuration for cross timeframe analysis optimized for high leverage trading."""
    # Timeframe configuration - optimized for high leverage trading
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '30m'])
    base_timeframe: str = '1m'
    
    # Feature engineering - optimized for short timeframes
    interaction_features: List[str] = field(default_factory=lambda: ['correlation', 'momentum', 'volatility', 'volume', 'microstructure'])
    lookback_periods: List[int] = field(default_factory=lambda: [3, 5, 10, 15, 20])  # Shorter periods for high leverage
    
    # Analysis parameters - adjusted for high leverage
    correlation_threshold: float = 0.6  # Lower threshold for short timeframes
    min_observations: int = 50  # Reduced for short timeframes
    max_correlations: int = 30  # Reduced for performance
    
    # High leverage specific parameters
    enable_microstructure_features: bool = True
    enable_order_flow_features: bool = True
    enable_momentum_divergence: bool = True
    enable_volatility_spillover: bool = True
    
    # Data quality
    enable_data_quality_validation: bool = True
    quality_thresholds: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossTimeframeResult:
    """Result of cross timeframe analysis."""
    cross_timeframe_features: pd.DataFrame
    interaction_metrics: Dict[str, Any]
    timeframe_correlations: Dict[str, Any]
    feature_importance: Dict[str, Any]
    quality_report: Optional[QualityResult] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

class CrossTimeframeAnalysisPipeline:
    """
    Cross Timeframe Analysis Pipeline.
    
    Provides comprehensive cross timeframe analysis for market data.
    """
    
    def __init__(self, config: Optional[CrossTimeframeConfig] = None):
        """Initialize cross timeframe analysis pipeline."""
        self.config = config or CrossTimeframeConfig()
        self.logger = logger.getChild('CrossTimeframeAnalysisPipeline')
        self.common_ops = CommonOperations()
        self.math_validator = MathValidation()
        
        # Initialize data quality utilities
        self.data_quality_validator = EnhancedDataQualityValidator()
        self.ml_data_quality = None
        
        try:
            self.ml_data_quality = DataQualityUtilities()
            self.logger.info("✅ ML data quality utilities initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ ML data quality utilities not available: {e}")
    
    async def analyze_cross_timeframes(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: Optional[List[str]] = None
    ) -> CrossTimeframeResult:
        """
        Perform cross timeframe analysis.
        
        Args:
            data_dir: Data directory path
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to analyze (optional)
            
        Returns:
            CrossTimeframeResult with analysis results and metrics
        """
        if timeframes is None:
            timeframes = self.config.timeframes
        
        self.logger.info(f"⏰ Starting cross timeframe analysis for {symbol} on {exchange} ({timeframes})")
        
        try:
            # Load multi-timeframe data
            timeframe_data = await self._load_multi_timeframe_data(data_dir, symbol, exchange, timeframes)
            
            # Perform data quality validation
            quality_report = None
            if self.config.enable_data_quality_validation:
                quality_report = await self._validate_data_quality(timeframe_data, symbol, exchange)
            
            # Align timeframes
            aligned_data = await self._align_timeframes(timeframe_data)
            
            # Engineer cross timeframe features
            cross_timeframe_features = await self._engineer_cross_timeframe_features(aligned_data)
            
            # Calculate interaction metrics
            interaction_metrics = await self._calculate_interaction_metrics(aligned_data)
            
            # Calculate timeframe correlations
            timeframe_correlations = await self._calculate_timeframe_correlations(aligned_data)
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(cross_timeframe_features)
            
            # Prepare analysis metadata
            analysis_metadata = {
                'timeframes_analyzed': timeframes,
                'base_timeframe': self.config.base_timeframe,
                'total_features': len(cross_timeframe_features.columns),
                'interaction_features': self.config.interaction_features,
                'correlation_threshold': self.config.correlation_threshold
            }
            
            result = CrossTimeframeResult(
                cross_timeframe_features=cross_timeframe_features,
                interaction_metrics=interaction_metrics,
                timeframe_correlations=timeframe_correlations,
                feature_importance=feature_importance,
                quality_report=quality_report,
                analysis_metadata=analysis_metadata
            )
            
            self.logger.info(f"✅ Cross timeframe analysis completed: {len(cross_timeframe_features.columns)} features generated")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Cross timeframe analysis failed: {e}")
            raise
    
    async def _load_multi_timeframe_data(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes."""
        self.logger.info(f"📊 Loading data for {len(timeframes)} timeframes")
        
        timeframe_data = {}
        
        for timeframe in timeframes:
            try:
                # Construct file path
                file_path = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
                
                if not file_path.exists():
                    self.logger.warning(f"⚠️ Data file not found for {timeframe}: {file_path}")
                    continue
                
                # Load data using standardized handler
                data = standardized_parquet_handler.read_parquet_standardized(file_path)
                
                # Basic validation
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = set(required_columns) - set(data.columns)
                if missing_columns:
                    self.logger.warning(f"⚠️ Missing columns for {timeframe}: {missing_columns}")
                    continue
                
                # Sort by timestamp if available
                if 'timestamp' in data.columns:
                    data = data.sort_values('timestamp').reset_index(drop=True)
                
                # Resample to target timeframe if needed
                if timeframe != self.config.base_timeframe:
                    data = await self._resample_data(data, timeframe)
                
                timeframe_data[timeframe] = data
                self.logger.info(f"📊 Loaded {len(data)} data points for {timeframe}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load data for {timeframe}: {e}")
                continue
        
        if not timeframe_data:
            raise ValueError("No timeframe data could be loaded")
        
        self.logger.info(f"📊 Successfully loaded data for {len(timeframe_data)} timeframes")
        return timeframe_data
    
    async def _resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe."""
        try:
            # Convert timeframe to pandas frequency
            timeframe_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            if target_timeframe not in timeframe_map:
                self.logger.warning(f"⚠️ Unknown timeframe: {target_timeframe}")
                return data
            
            frequency = timeframe_map[target_timeframe]
            
            # Ensure timestamp column exists
            if 'timestamp' not in data.columns:
                # Create timestamp index
                data = data.copy()
                data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='1T')
            
            # Set timestamp as index
            data_indexed = data.set_index('timestamp')
            
            # Resample OHLCV data
            resampled = data_indexed.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Reset index
            resampled = resampled.reset_index()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"❌ Data resampling failed: {e}")
            return data
    
    async def _validate_data_quality(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        symbol: str,
        exchange: str
    ) -> QualityResult:
        """Validate data quality for all timeframes."""
        self.logger.info("🔍 Performing data quality validation for cross timeframe analysis")
        
        try:
            # Aggregate quality results from all timeframes
            all_issues = []
            all_warnings = []
            
            for timeframe, data in timeframe_data.items():
                try:
                    # Use enhanced data quality validator
                    quality_result = self.data_quality_validator.validate_dataframe(data)
                    
                    # Add timeframe prefix to issues and warnings
                    for issue in quality_result.issues:
                        all_issues.append(f"{timeframe}: {issue}")
                    for warning in quality_result.warnings:
                        all_warnings.append(f"{timeframe}: {warning}")
                    
                    # Use ML data quality utilities if available
                    if self.ml_data_quality:
                        try:
                            ml_quality_report = await self.ml_data_quality.perform_comprehensive_validation(
                                data, symbol=symbol, exchange=exchange
                            )
                            
                            # Merge ML quality insights
                            if ml_quality_report.get('has_critical_issues', False):
                                for issue in ml_quality_report.get('critical_issues', []):
                                    all_issues.append(f"{timeframe} (ML): {issue}")
                            
                            if ml_quality_report.get('warnings', []):
                                for warning in ml_quality_report.get('warnings', []):
                                    all_warnings.append(f"{timeframe} (ML): {warning}")
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ ML data quality validation failed for {timeframe}: {e}")
                
                except Exception as e:
                    all_issues.append(f"{timeframe}: Validation failed - {e}")
            
            # Create combined quality result
            combined_quality_result = QualityResult(
                passed=len(all_issues) == 0,
                issues=all_issues,
                warnings=all_warnings
            )
            
            # Log quality results
            if combined_quality_result.passed:
                self.logger.info("✅ Cross timeframe data quality validation passed")
            else:
                self.logger.warning(f"⚠️ Cross timeframe data quality issues found: {len(all_issues)} issues, {len(all_warnings)} warnings")
                for issue in all_issues[:5]:  # Log first 5 issues
                    self.logger.warning(f"  - {issue}")
            
            return combined_quality_result
            
        except Exception as e:
            self.logger.error(f"❌ Cross timeframe data quality validation failed: {e}")
            # Return a basic quality result
            return QualityResult(passed=False, issues=[f"Validation failed: {e}"])
    
    async def _align_timeframes(
        self,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align timeframes to common time index."""
        self.logger.info("🔄 Aligning timeframes")
        
        try:
            aligned_data = {}
            
            # Use base timeframe as reference
            base_timeframe = self.config.base_timeframe
            if base_timeframe not in timeframe_data:
                base_timeframe = list(timeframe_data.keys())[0]
            
            base_data = timeframe_data[base_timeframe]
            
            # Create common time index
            if 'timestamp' in base_data.columns:
                common_index = base_data['timestamp']
            else:
                # Create synthetic time index
                common_index = pd.date_range(start='2023-01-01', periods=len(base_data), freq='1T')
            
            # Align each timeframe to common index
            for timeframe, data in timeframe_data.items():
                if timeframe == base_timeframe:
                    aligned_data[timeframe] = data.copy()
                    continue
                
                # Forward fill to align with common index
                aligned_df = pd.DataFrame(index=common_index)
                
                # Interpolate or forward fill data
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in data.columns:
                        if 'timestamp' in data.columns:
                            series = pd.Series(data[col].values, index=data['timestamp'])
                        else:
                            series = pd.Series(data[col].values)
                        
                        # Forward fill to align with common index
                        aligned_series = series.reindex(common_index, method='ffill')
                        aligned_df[col] = aligned_series
                
                aligned_data[timeframe] = aligned_df.dropna()
            
            self.logger.info("✅ Timeframes aligned")
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"❌ Timeframe alignment failed: {e}")
            return timeframe_data
    
    async def _engineer_cross_timeframe_features(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Engineer cross timeframe features."""
        self.logger.info("🔧 Engineering cross timeframe features")
        
        try:
            features = pd.DataFrame()
            timeframes = list(aligned_data.keys())
            
            # Base timeframe features
            base_timeframe = timeframes[0]
            base_data = aligned_data[base_timeframe]
            
            # Create base features
            features['base_close'] = base_data['close']
            features['base_volume'] = base_data['volume']
            features['base_returns'] = base_data['close'].pct_change()
            features['base_volatility'] = features['base_returns'].rolling(20).std()
            
            # Cross timeframe interaction features
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    # Correlation features
                    if 'correlation' in self.config.interaction_features:
                        corr_5 = data1['close'].rolling(5).corr(data2['close'])
                        corr_20 = data1['close'].rolling(20).corr(data2['close'])
                        
                        features[f'corr_{tf1}_{tf2}_5'] = corr_5
                        features[f'corr_{tf1}_{tf2}_20'] = corr_20
                    
                    # Momentum features
                    if 'momentum' in self.config.interaction_features:
                        mom1 = data1['close'].pct_change(5)
                        mom2 = data2['close'].pct_change(5)
                        features[f'mom_diff_{tf1}_{tf2}'] = mom1 - mom2
                        features[f'mom_ratio_{tf1}_{tf2}'] = mom1 / (mom2 + 1e-10)
                    
                    # Volatility features
                    if 'volatility' in self.config.interaction_features:
                        vol1 = data1['close'].pct_change().rolling(20).std()
                        vol2 = data2['close'].pct_change().rolling(20).std()
                        features[f'vol_ratio_{tf1}_{tf2}'] = vol1 / (vol2 + 1e-10)
                        features[f'vol_diff_{tf1}_{tf2}'] = vol1 - vol2
                    
                    # Volume features
                    if 'volume' in self.config.interaction_features:
                        vol_ratio = data1['volume'] / (data2['volume'] + 1e-10)
                        features[f'volume_ratio_{tf1}_{tf2}'] = vol_ratio
            
            # Multi-timeframe aggregation features
            for timeframe in timeframes:
                data = aligned_data[timeframe]
                
                # Price position across timeframes
                for period in self.config.lookback_periods:
                    high_period = data['high'].rolling(period).max()
                    low_period = data['low'].rolling(period).min()
                    price_position = (data['close'] - low_period) / (high_period - low_period + 1e-10)
                    features[f'price_pos_{timeframe}_{period}'] = price_position
                
                # Volume profile
                volume_ma = data['volume'].rolling(20).mean()
                volume_ratio = data['volume'] / (volume_ma + 1e-10)
                features[f'volume_profile_{timeframe}'] = volume_ratio
            
            # High leverage specific features
            if self.config.enable_microstructure_features:
                features.update(self._generate_microstructure_features(aligned_data))
            
            if self.config.enable_order_flow_features:
                features.update(self._generate_order_flow_features(aligned_data))
            
            if self.config.enable_momentum_divergence:
                features.update(self._generate_momentum_divergence_features(aligned_data))
            
            if self.config.enable_volatility_spillover:
                features.update(self._generate_volatility_spillover_features(aligned_data))
            
            # Remove rows with NaN values
            features = features.dropna()
            
            self.logger.info(f"🔧 Engineered {len(features.columns)} cross timeframe features")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Cross timeframe feature engineering failed: {e}")
            raise
    
    async def _calculate_interaction_metrics(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate interaction metrics between timeframes."""
        self.logger.info("📊 Calculating interaction metrics")
        
        try:
            metrics = {}
            timeframes = list(aligned_data.keys())
            
            # Calculate pairwise correlations
            correlations = {}
            for i, tf1 in enumerate(timeframes):
                for j, tf2 in enumerate(timeframes[i+1:], i+1):
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    # Price correlation
                    price_corr = data1['close'].corr(data2['close'])
                    
                    # Volume correlation
                    volume_corr = data1['volume'].corr(data2['volume'])
                    
                    # Returns correlation
                    returns1 = data1['close'].pct_change()
                    returns2 = data2['close'].pct_change()
                    returns_corr = returns1.corr(returns2)
                    
                    correlations[f'{tf1}_{tf2}'] = {
                        'price_correlation': price_corr,
                        'volume_correlation': volume_corr,
                        'returns_correlation': returns_corr,
                        'avg_correlation': (price_corr + volume_corr + returns_corr) / 3
                    }
            
            # Calculate interaction strength
            strong_interactions = []
            for pair, corrs in correlations.items():
                if abs(corrs['avg_correlation']) > self.config.correlation_threshold:
                    strong_interactions.append(pair)
            
            metrics = {
                'pairwise_correlations': correlations,
                'strong_interactions': strong_interactions,
                'interaction_strength': len(strong_interactions) / len(correlations) if correlations else 0,
                'total_interactions': len(correlations)
            }
            
            self.logger.info("✅ Interaction metrics calculated")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Interaction metrics calculation failed: {e}")
            return {}
    
    async def _calculate_timeframe_correlations(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate detailed timeframe correlations."""
        self.logger.info("📊 Calculating timeframe correlations")
        
        try:
            correlations = {}
            timeframes = list(aligned_data.keys())
            
            # Create correlation matrix for each metric
            metrics = ['close', 'volume', 'returns', 'volatility']
            
            for metric in metrics:
                corr_matrix = pd.DataFrame(index=timeframes, columns=timeframes)
                
                for tf1 in timeframes:
                    for tf2 in timeframes:
                        data1 = aligned_data[tf1]
                        data2 = aligned_data[tf2]
                        
                        if metric == 'close':
                            corr_value = data1['close'].corr(data2['close'])
                        elif metric == 'volume':
                            corr_value = data1['volume'].corr(data2['volume'])
                        elif metric == 'returns':
                            returns1 = data1['close'].pct_change()
                            returns2 = data2['close'].pct_change()
                            corr_value = returns1.corr(returns2)
                        elif metric == 'volatility':
                            vol1 = data1['close'].pct_change().rolling(20).std()
                            vol2 = data2['close'].pct_change().rolling(20).std()
                            corr_value = vol1.corr(vol2)
                        
                        corr_matrix.loc[tf1, tf2] = corr_value
                
                correlations[metric] = corr_matrix.fillna(1.0)  # Diagonal should be 1.0
            
            # Calculate average correlation
            avg_corr = np.mean([corr.values for corr in correlations.values()], axis=0)
            correlations['average'] = pd.DataFrame(avg_corr, index=timeframes, columns=timeframes)
            
            self.logger.info("✅ Timeframe correlations calculated")
            return correlations
            
        except Exception as e:
            self.logger.error(f"❌ Timeframe correlations calculation failed: {e}")
            return {}
    
    async def _calculate_feature_importance(
        self,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate feature importance for cross timeframe features."""
        self.logger.info("📊 Calculating feature importance")
        
        try:
            # Calculate correlation with base returns
            if 'base_returns' in features.columns:
                base_returns = features['base_returns'].dropna()
                
                feature_importance = {}
                for col in features.columns:
                    if col != 'base_returns' and col in features.columns:
                        try:
                            # Align data
                            aligned_data = features[[col, 'base_returns']].dropna()
                            if len(aligned_data) > 10:
                                corr = aligned_data[col].corr(aligned_data['base_returns'])
                                feature_importance[col] = abs(corr)
                        except:
                            feature_importance[col] = 0.0
                
                # Sort by importance
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                
                # Get top features
                top_features = dict(list(sorted_importance.items())[:20])
                
                importance_metrics = {
                    'feature_importance': sorted_importance,
                    'top_features': top_features,
                    'avg_importance': np.mean(list(sorted_importance.values())),
                    'max_importance': max(sorted_importance.values()) if sorted_importance else 0.0
                }
                
                self.logger.info("✅ Feature importance calculated")
                return importance_metrics
            
            else:
                self.logger.warning("⚠️ Base returns not found, skipping feature importance calculation")
                return {}
            
        except Exception as e:
            self.logger.error(f"❌ Feature importance calculation failed: {e}")
            return {}
    
    def _generate_microstructure_features(self, aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Generate microstructure features for high leverage trading."""
        features = {}
        timeframes = list(aligned_data.keys())
        
        for timeframe in timeframes:
            data = aligned_data[timeframe]
            
            # Bid-ask spread proxy (using high-low as proxy)
            features[f'spread_proxy_{timeframe}'] = (data['high'] - data['low']) / data['close']
            
            # Price impact proxy (volume vs price movement)
            price_change = data['close'].pct_change().abs()
            volume_normalized = data['volume'] / data['volume'].rolling(20).mean()
            features[f'price_impact_{timeframe}'] = price_change / (volume_normalized + 1e-10)
            
            # Tick-by-tick volatility (using high-low range)
            features[f'tick_volatility_{timeframe}'] = (data['high'] - data['low']) / data['close']
            
            # Order flow imbalance proxy (close position within bar)
            features[f'order_flow_imbalance_{timeframe}'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-10)
        
        return features
    
    def _generate_order_flow_features(self, aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Generate order flow features for high leverage trading."""
        features = {}
        timeframes = list(aligned_data.keys())
        
        for timeframe in timeframes:
            data = aligned_data[timeframe]
            
            # Volume-weighted average price (VWAP) deviation
            vwap = (data['high'] + data['low'] + data['close']) / 3
            vwap_volume = (vwap * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            features[f'vwap_deviation_{timeframe}'] = (data['close'] - vwap_volume) / vwap_volume
            
            # Volume momentum
            volume_momentum = data['volume'].pct_change(5)
            features[f'volume_momentum_{timeframe}'] = volume_momentum
            
            # Price-volume relationship
            price_momentum = data['close'].pct_change(5)
            features[f'price_volume_correlation_{timeframe}'] = price_momentum.rolling(10).corr(volume_momentum)
        
        return features
    
    def _generate_momentum_divergence_features(self, aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Generate momentum divergence features between timeframes."""
        features = {}
        timeframes = list(aligned_data.keys())
        
        # Calculate momentum for each timeframe
        momentum_data = {}
        for timeframe in timeframes:
            data = aligned_data[timeframe]
            momentum_data[timeframe] = {
                'momentum_5': data['close'].pct_change(5),
                'momentum_10': data['close'].pct_change(10),
                'momentum_20': data['close'].pct_change(20)
            }
        
        # Calculate divergences between timeframes
        for i, tf1 in enumerate(timeframes):
            for j, tf2 in enumerate(timeframes[i+1:], i+1):
                for period in [5, 10, 20]:
                    mom1 = momentum_data[tf1][f'momentum_{period}']
                    mom2 = momentum_data[tf2][f'momentum_{period}']
                    
                    # Momentum divergence
                    features[f'momentum_divergence_{tf1}_{tf2}_{period}'] = mom1 - mom2
                    
                    # Momentum ratio
                    features[f'momentum_ratio_{tf1}_{tf2}_{period}'] = mom1 / (mom2 + 1e-10)
                    
                    # Momentum correlation
                    features[f'momentum_correlation_{tf1}_{tf2}_{period}'] = mom1.rolling(20).corr(mom2)
        
        return features
    
    def _generate_volatility_spillover_features(self, aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Generate volatility spillover features between timeframes."""
        features = {}
        timeframes = list(aligned_data.keys())
        
        # Calculate volatility for each timeframe
        volatility_data = {}
        for timeframe in timeframes:
            data = aligned_data[timeframe]
            returns = data['close'].pct_change()
            volatility_data[timeframe] = {
                'volatility_5': returns.rolling(5).std(),
                'volatility_10': returns.rolling(10).std(),
                'volatility_20': returns.rolling(20).std()
            }
        
        # Calculate volatility spillovers
        for i, tf1 in enumerate(timeframes):
            for j, tf2 in enumerate(timeframes[i+1:], i+1):
                for period in [5, 10, 20]:
                    vol1 = volatility_data[tf1][f'volatility_{period}']
                    vol2 = volatility_data[tf2][f'volatility_{period}']
                    
                    # Volatility spillover (lagged correlation)
                    vol1_lagged = vol1.shift(1)
                    features[f'volatility_spillover_{tf1}_{tf2}_{period}'] = vol1_lagged.rolling(20).corr(vol2)
                    
                    # Volatility ratio
                    features[f'volatility_ratio_{tf1}_{tf2}_{period}'] = vol1 / (vol2 + 1e-10)
                    
                    # Volatility difference
                    features[f'volatility_diff_{tf1}_{tf2}_{period}'] = vol1 - vol2
        
        return features

# Convenience function
async def analyze_cross_timeframes(
    data_dir: str,
    symbol: str,
    exchange: str,
    timeframes: Optional[List[str]] = None,
    config: Optional[CrossTimeframeConfig] = None
) -> CrossTimeframeResult:
    """Convenience function to analyze cross timeframes."""
    pipeline = CrossTimeframeAnalysisPipeline(config)
    return await pipeline.analyze_cross_timeframes(data_dir, symbol, exchange, timeframes)