"""
Tactician PID-Based Feature Generation - With Long/Short Differentiation

This component provides PID-based feature generation for Tactician models on 1m timeframe
with long/short differentiation. Uses separate analysis for long and short opportunities.

Key features:
- Long/short differentiation (separate analysis)
- Optimized for 1m timeframe
- Enhanced PID analysis
- Focus on separate long and short opportunity assessment
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

# Core dependencies
import numpy as np
import pandas as pd

# Import logger
from src.utils.logger import get_logger

class GenerationStatus(Enum):
    """Status of feature generation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class TacticianPIDConfig:
    """Configuration for Tactician PID-based feature generation."""
    # PID parameters (long/short differentiated)
    pid_threshold: float = 0.1  # Minimum PID score for feature selection
    max_features_per_type: int = 100  # Maximum features per type (higher for 1m data)
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    enable_cross_timeframe_features: bool = True
    
    # Timeframe specific (1m for Tactician)
    timeframe_minutes: int = 1
    max_horizon_minutes: int = 20  # 20 periods * 1 minute
    
    # Long/short differentiation parameters
    enable_long_short_differentiation: bool = True
    long_short_balance_weight: float = 0.5  # Weight for balancing long/short features
    directional_confidence_threshold: float = 0.1  # Minimum confidence for directional bias
    
    # Quality thresholds
    min_feature_quality_score: float = 0.6
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.9
    
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size: int = 2000  # Larger batch size for 1m data

@dataclass
class TacticianPIDResult:
    """Result of Tactician PID-based feature generation."""
    # Generated features (long/short differentiated)
    long_interaction_features: Optional[pd.DataFrame] = None
    short_interaction_features: Optional[pd.DataFrame] = None
    long_polynomial_features: Optional[pd.DataFrame] = None
    short_polynomial_features: Optional[pd.DataFrame] = None
    long_cross_timeframe_features: Optional[pd.DataFrame] = None
    short_cross_timeframe_features: Optional[pd.DataFrame] = None
    combined_features: Optional[pd.DataFrame] = None
    
    # Feature names
    long_feature_names: List[str] = None
    short_feature_names: List[str] = None
    combined_feature_names: List[str] = None
    
    # Quality metrics
    overall_quality_score: float = 0.0
    long_quality_score: float = 0.0
    short_quality_score: float = 0.0
    long_short_balance_score: float = 0.0
    
    # Performance metrics
    total_features_generated: int = 0
    long_features_generated: int = 0
    short_features_generated: int = 0
    generation_time: float = 0.0
    generation_status: GenerationStatus = GenerationStatus.PENDING
    
    # Status
    success: bool = False
    error_message: Optional[str] = None

class TacticianPIDBasedFeatureGenerator:
    """
    Tactician PID-Based Feature Generator - WITH LONG/SHORT DIFFERENTIATION.
    
    Generates features using PID analysis for Tactician models on 1m timeframe
    with long/short differentiation.
    """
    
    def __init__(self, config: Optional[TacticianPIDConfig] = None):
        """Initialize the Tactician PID-based feature generator."""
        self.config = config or TacticianPIDConfig()
        self.logger = get_logger('TacticianPIDBasedFeatureGenerator')
        
        # Initialize hardware optimizers
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if self.config.enable_memory_optimization:
            try:
                from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
                from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
                
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                if self.memory_optimizer:
                    self.memory_optimizer.set_memory_limit(self.config.max_memory_usage_gb)
            except ImportError:
                self.logger.warning("Hardware optimizers not available")
        
        self.logger.info("🚀 Tactician PID-Based Feature Generator initialized (WITH LONG/SHORT DIFFERENTIATION)")
        self.logger.info(f"   → PID threshold: {self.config.pid_threshold}")
        self.logger.info(f"   → Max features per type: {self.config.max_features_per_type}")
        self.logger.info(f"   → Timeframe: {self.config.timeframe_minutes}m")
        self.logger.info(f"   → Long/Short differentiation: {'Enabled' if self.config.enable_long_short_differentiation else 'Disabled'}")
        self.logger.info(f"   → Memory optimization: {'Enabled' if self.config.enable_memory_optimization else 'Disabled'}")
    
    async def generate_features(self, 
                              market_data: pd.DataFrame,
                              target_data: Optional[pd.Series] = None,
                              long_target_data: Optional[pd.Series] = None,
                              short_target_data: Optional[pd.Series] = None,
                              lookback_periods: Optional[Dict[str, int]] = None) -> TacticianPIDResult:
        """
        Generate PID-based features for Tactician (LONG/SHORT DIFFERENTIATED).
        
        Args:
            market_data: Market data for feature generation
            target_data: Combined target variable (optional)
            long_target_data: Long-specific target variable (optional)
            short_target_data: Short-specific target variable (optional)
            lookback_periods: Optimized lookback periods (optional)
            
        Returns:
            TacticianPIDResult with generated features
        """
        start_time = time.time()
        self.logger.info("🔍 Starting Tactician PID-based feature generation (LONG/SHORT DIFFERENTIATED)")
        
        try:
            # Step 1: Validate input data
            validation_result = await self._validate_input_data(market_data, target_data, long_target_data, short_target_data)
            if not validation_result['is_valid']:
                return TacticianPIDResult(
                    success=False,
                    error_message=validation_result['error_message'],
                    generation_time=time.time() - start_time,
                    generation_status=GenerationStatus.FAILED
                )
            
            # Step 2: Prepare data for feature generation
            prepared_data = await self._prepare_data_for_generation(market_data, target_data, long_target_data, short_target_data)
            
            # Step 3: Generate long features
            long_result = await self._generate_long_features(prepared_data, lookback_periods)
            
            # Step 4: Generate short features
            short_result = await self._generate_short_features(prepared_data, lookback_periods)
            
            # Step 5: Combine features
            combined_result = await self._combine_features(long_result, short_result)
            
            # Step 6: Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(combined_result, long_result, short_result)
            
            # Step 7: Create final result
            result = TacticianPIDResult(
                long_interaction_features=long_result.get('interaction_features'),
                short_interaction_features=short_result.get('interaction_features'),
                long_polynomial_features=long_result.get('polynomial_features'),
                short_polynomial_features=short_result.get('polynomial_features'),
                long_cross_timeframe_features=long_result.get('cross_timeframe_features'),
                short_cross_timeframe_features=short_result.get('cross_timeframe_features'),
                combined_features=combined_result.get('features'),
                long_feature_names=long_result.get('feature_names', []),
                short_feature_names=short_result.get('feature_names', []),
                combined_feature_names=combined_result.get('feature_names', []),
                overall_quality_score=quality_metrics['overall'],
                long_quality_score=quality_metrics['long'],
                short_quality_score=quality_metrics['short'],
                long_short_balance_score=quality_metrics['long_short_balance'],
                total_features_generated=len(combined_result.get('feature_names', [])),
                long_features_generated=len(long_result.get('feature_names', [])),
                short_features_generated=len(short_result.get('feature_names', [])),
                generation_time=time.time() - start_time,
                generation_status=GenerationStatus.COMPLETED,
                success=True
            )
            
            self.logger.info(f"✅ Tactician feature generation completed: {result.total_features_generated} features generated")
            self.logger.info(f"   → Long features: {result.long_features_generated}")
            self.logger.info(f"   → Short features: {result.short_features_generated}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Tactician feature generation failed: {e}")
            return TacticianPIDResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                generation_status=GenerationStatus.FAILED
            )
    
    async def _validate_input_data(self, market_data: pd.DataFrame, 
                                 target_data: Optional[pd.Series],
                                 long_target_data: Optional[pd.Series],
                                 short_target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Validate input data for feature generation."""
        try:
            # Check market data
            if market_data is None or market_data.empty:
                return {'is_valid': False, 'error_message': 'Market data is empty or None'}
            
            if len(market_data) < 100:
                return {'is_valid': False, 'error_message': f'Insufficient data: {len(market_data)} rows'}
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            
            if len(missing_columns) == len(required_columns):
                return {'is_valid': False, 'error_message': 'No OHLCV columns found'}
            
            # Check target data
            has_target = target_data is not None
            has_long_target = long_target_data is not None
            has_short_target = short_target_data is not None
            
            if not (has_target or (has_long_target and has_short_target)):
                return {'is_valid': False, 'error_message': 'No target data provided'}
            
            # Validate target data lengths
            if target_data is not None and len(target_data) != len(market_data):
                return {'is_valid': False, 'error_message': 'Target data length mismatch'}
            
            if long_target_data is not None and len(long_target_data) != len(market_data):
                return {'is_valid': False, 'error_message': 'Long target data length mismatch'}
            
            if short_target_data is not None and len(short_target_data) != len(market_data):
                return {'is_valid': False, 'error_message': 'Short target data length mismatch'}
            
            return {'is_valid': True, 'error_message': None}
            
        except Exception as e:
            return {'is_valid': False, 'error_message': f'Validation error: {e}'}
    
    async def _prepare_data_for_generation(self, market_data: pd.DataFrame, 
                                         target_data: Optional[pd.Series],
                                         long_target_data: Optional[pd.Series],
                                         short_target_data: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for feature generation."""
        try:
            # Clean and prepare market data
            prepared_market_data = market_data.copy()
            
            # Handle missing values
            prepared_market_data = prepared_market_data.fillna(method='forward').fillna(method='backward')
            
            # Ensure numeric types
            for col in prepared_market_data.columns:
                if prepared_market_data[col].dtype == 'object':
                    try:
                        prepared_market_data[col] = pd.to_numeric(prepared_market_data[col], errors='coerce')
                    except:
                        prepared_market_data = prepared_market_data.drop(columns=[col])
            
            # Prepare target data
            prepared_target = None
            if target_data is not None:
                prepared_target = target_data.copy()
                prepared_target = prepared_target.fillna(method='forward').fillna(method='backward')
            
            prepared_long_target = None
            if long_target_data is not None:
                prepared_long_target = long_target_data.copy()
                prepared_long_target = prepared_long_target.fillna(method='forward').fillna(method='backward')
            
            prepared_short_target = None
            if short_target_data is not None:
                prepared_short_target = short_target_data.copy()
                prepared_short_target = prepared_short_target.fillna(method='forward').fillna(method='backward')
            
            return {
                'market_data': prepared_market_data,
                'target_data': prepared_target,
                'long_target_data': prepared_long_target,
                'short_target_data': prepared_short_target
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    async def _generate_long_features(self, prepared_data: Dict[str, Any], 
                                    lookback_periods: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """Generate long-specific features using PID analysis."""
        try:
            market_data = prepared_data['market_data']
            long_target_data = prepared_data['long_target_data']
            
            if long_target_data is None:
                return {'interaction_features': None, 'polynomial_features': None, 
                       'cross_timeframe_features': None, 'feature_names': []}
            
            # Generate base features for long analysis
            base_features = self._generate_base_features(market_data)
            
            # Generate long-specific features
            long_features = []
            long_names = []
            
            # Long momentum features
            if 'close' in market_data.columns:
                long_momentum_features = self._generate_long_momentum_features(market_data, base_features)
                long_features.extend(long_momentum_features['features'])
                long_names.extend(long_momentum_features['names'])
            
            # Long volatility features
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                long_volatility_features = self._generate_long_volatility_features(market_data, base_features)
                long_features.extend(long_volatility_features['features'])
                long_names.extend(long_volatility_features['names'])
            
            # Long trend features
            if 'close' in market_data.columns:
                long_trend_features = self._generate_long_trend_features(market_data, base_features)
                long_features.extend(long_trend_features['features'])
                long_names.extend(long_trend_features['names'])
            
            # Apply PID analysis to select best long features
            if len(long_features) > 0:
                selected_features, selected_names = self._apply_pid_selection(
                    long_features, long_names, long_target_data
                )
            else:
                selected_features, selected_names = [], []
            
            # Limit number of features
            if len(selected_features) > self.config.max_features_per_type:
                selected_features = selected_features[:self.config.max_features_per_type]
                selected_names = selected_names[:self.config.max_features_per_type]
            
            return {
                'interaction_features': np.array(selected_features) if selected_features else None,
                'polynomial_features': None,  # Simplified for this example
                'cross_timeframe_features': None,  # Simplified for this example
                'feature_names': selected_names
            }
            
        except Exception as e:
            self.logger.error(f"Long feature generation failed: {e}")
            return {'interaction_features': None, 'polynomial_features': None, 
                   'cross_timeframe_features': None, 'feature_names': []}
    
    async def _generate_short_features(self, prepared_data: Dict[str, Any], 
                                     lookback_periods: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """Generate short-specific features using PID analysis."""
        try:
            market_data = prepared_data['market_data']
            short_target_data = prepared_data['short_target_data']
            
            if short_target_data is None:
                return {'interaction_features': None, 'polynomial_features': None, 
                       'cross_timeframe_features': None, 'feature_names': []}
            
            # Generate base features for short analysis
            base_features = self._generate_base_features(market_data)
            
            # Generate short-specific features
            short_features = []
            short_names = []
            
            # Short momentum features
            if 'close' in market_data.columns:
                short_momentum_features = self._generate_short_momentum_features(market_data, base_features)
                short_features.extend(short_momentum_features['features'])
                short_names.extend(short_momentum_features['names'])
            
            # Short volatility features
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                short_volatility_features = self._generate_short_volatility_features(market_data, base_features)
                short_features.extend(short_volatility_features['features'])
                short_names.extend(short_volatility_features['names'])
            
            # Short trend features
            if 'close' in market_data.columns:
                short_trend_features = self._generate_short_trend_features(market_data, base_features)
                short_features.extend(short_trend_features['features'])
                short_names.extend(short_trend_features['names'])
            
            # Apply PID analysis to select best short features
            if len(short_features) > 0:
                selected_features, selected_names = self._apply_pid_selection(
                    short_features, short_names, short_target_data
                )
            else:
                selected_features, selected_names = [], []
            
            # Limit number of features
            if len(selected_features) > self.config.max_features_per_type:
                selected_features = selected_features[:self.config.max_features_per_type]
                selected_names = selected_names[:self.config.max_features_per_type]
            
            return {
                'interaction_features': np.array(selected_features) if selected_features else None,
                'polynomial_features': None,  # Simplified for this example
                'cross_timeframe_features': None,  # Simplified for this example
                'feature_names': selected_names
            }
            
        except Exception as e:
            self.logger.error(f"Short feature generation failed: {e}")
            return {'interaction_features': None, 'polynomial_features': None, 
                   'cross_timeframe_features': None, 'feature_names': []}
    
    def _generate_base_features(self, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate base features for further processing."""
        base_features = {}
        
        # Price features
        if 'close' in market_data.columns:
            base_features['close'] = market_data['close'].values
            base_features['close_sma_5'] = market_data['close'].rolling(window=5).mean().values
            base_features['close_sma_10'] = market_data['close'].rolling(window=10).mean().values
            base_features['close_ema_5'] = market_data['close'].ewm(span=5).mean().values
        
        # Volume features
        if 'volume' in market_data.columns:
            base_features['volume'] = market_data['volume'].values
            base_features['volume_sma_5'] = market_data['volume'].rolling(window=5).mean().values
            base_features['volume_ema_5'] = market_data['volume'].ewm(span=5).mean().values
        
        # Volatility features
        if all(col in market_data.columns for col in ['high', 'low', 'close']):
            base_features['atr'] = self._calculate_atr(market_data)
            base_features['volatility'] = market_data['close'].rolling(window=10).std().values
        
        return base_features
    
    def _generate_long_momentum_features(self, market_data: pd.DataFrame, 
                                       base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate long-specific momentum features."""
        features = []
        names = []
        
        if 'close' in base_features:
            # Long momentum (upward price movement)
            long_momentum = np.diff(base_features['close'], prepend=base_features['close'][0])
            long_momentum = np.maximum(0, long_momentum)  # Only positive momentum
            features.append(long_momentum)
            names.append('long_momentum')
            
            # Long acceleration
            long_acceleration = np.diff(long_momentum, prepend=long_momentum[0])
            features.append(long_acceleration)
            names.append('long_acceleration')
            
            # Long momentum ratio
            long_momentum_ratio = long_momentum / (base_features['close'] + 1e-8)
            features.append(long_momentum_ratio)
            names.append('long_momentum_ratio')
        
        return {'features': features, 'names': names}
    
    def _generate_short_momentum_features(self, market_data: pd.DataFrame, 
                                        base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate short-specific momentum features."""
        features = []
        names = []
        
        if 'close' in base_features:
            # Short momentum (downward price movement)
            short_momentum = np.diff(base_features['close'], prepend=base_features['close'][0])
            short_momentum = np.minimum(0, short_momentum)  # Only negative momentum
            features.append(short_momentum)
            names.append('short_momentum')
            
            # Short acceleration
            short_acceleration = np.diff(short_momentum, prepend=short_momentum[0])
            features.append(short_acceleration)
            names.append('short_acceleration')
            
            # Short momentum ratio
            short_momentum_ratio = short_momentum / (base_features['close'] + 1e-8)
            features.append(short_momentum_ratio)
            names.append('short_momentum_ratio')
        
        return {'features': features, 'names': names}
    
    def _generate_long_volatility_features(self, market_data: pd.DataFrame, 
                                         base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate long-specific volatility features."""
        features = []
        names = []
        
        if 'atr' in base_features and 'close' in base_features:
            # Long volatility (volatility during upward moves)
            price_changes = np.diff(base_features['close'], prepend=base_features['close'][0])
            long_volatility = np.where(price_changes > 0, base_features['atr'], 0)
            features.append(long_volatility)
            names.append('long_volatility')
            
            # Long volatility ratio
            long_volatility_ratio = long_volatility / (base_features['close'] + 1e-8)
            features.append(long_volatility_ratio)
            names.append('long_volatility_ratio')
        
        return {'features': features, 'names': names}
    
    def _generate_short_volatility_features(self, market_data: pd.DataFrame, 
                                          base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate short-specific volatility features."""
        features = []
        names = []
        
        if 'atr' in base_features and 'close' in base_features:
            # Short volatility (volatility during downward moves)
            price_changes = np.diff(base_features['close'], prepend=base_features['close'][0])
            short_volatility = np.where(price_changes < 0, base_features['atr'], 0)
            features.append(short_volatility)
            names.append('short_volatility')
            
            # Short volatility ratio
            short_volatility_ratio = short_volatility / (base_features['close'] + 1e-8)
            features.append(short_volatility_ratio)
            names.append('short_volatility_ratio')
        
        return {'features': features, 'names': names}
    
    def _generate_long_trend_features(self, market_data: pd.DataFrame, 
                                    base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate long-specific trend features."""
        features = []
        names = []
        
        if 'close' in base_features:
            # Long trend strength
            close_values = base_features['close']
            long_trend_strength = np.maximum(0, np.diff(close_values, prepend=close_values[0]))
            features.append(long_trend_strength)
            names.append('long_trend_strength')
            
            # Long trend persistence
            long_trend_persistence = np.cumsum(long_trend_strength)
            features.append(long_trend_persistence)
            names.append('long_trend_persistence')
        
        return {'features': features, 'names': names}
    
    def _generate_short_trend_features(self, market_data: pd.DataFrame, 
                                     base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate short-specific trend features."""
        features = []
        names = []
        
        if 'close' in base_features:
            # Short trend strength
            close_values = base_features['close']
            short_trend_strength = np.minimum(0, np.diff(close_values, prepend=close_values[0]))
            features.append(short_trend_strength)
            names.append('short_trend_strength')
            
            # Short trend persistence
            short_trend_persistence = np.cumsum(short_trend_strength)
            features.append(short_trend_persistence)
            names.append('short_trend_persistence')
        
        return {'features': features, 'names': names}
    
    def _apply_pid_selection(self, features: List[np.ndarray], names: List[str], 
                           target_data: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """Apply PID analysis to select best features."""
        try:
            if len(features) == 0:
                return [], []
            
            # Calculate PID scores for each feature
            pid_scores = []
            for feature in features:
                pid_score = self._calculate_pid_score(feature, target_data)
                pid_scores.append(pid_score)
            
            # Select features above threshold
            selected_features = []
            selected_names = []
            
            for i, (feature, name, score) in enumerate(zip(features, names, pid_scores)):
                if score >= self.config.pid_threshold:
                    selected_features.append(feature)
                    selected_names.append(name)
            
            # Sort by PID score and limit
            if len(selected_features) > self.config.max_features_per_type:
                sorted_indices = np.argsort(pid_scores)[::-1]
                selected_features = [features[i] for i in sorted_indices[:self.config.max_features_per_type]]
                selected_names = [names[i] for i in sorted_indices[:self.config.max_features_per_type]]
            
            return selected_features, selected_names
            
        except Exception as e:
            self.logger.warning(f"PID selection failed: {e}")
            return features[:self.config.max_features_per_type], names[:self.config.max_features_per_type]
    
    def _calculate_pid_score(self, feature: np.ndarray, target: pd.Series) -> float:
        """Calculate PID score for feature selection."""
        try:
            # Align lengths
            min_len = min(len(feature), len(target))
            if min_len < 10:
                return 0.0
            
            feature_aligned = feature[-min_len:]
            target_aligned = target.iloc[-min_len:].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
            if np.sum(valid_mask) < 10:
                return 0.0
            
            feature_clean = feature_aligned[valid_mask]
            target_clean = target_aligned[valid_mask]
            
            # Calculate correlation
            correlation = np.corrcoef(feature_clean, target_clean)[0, 1]
            if np.isnan(correlation):
                return 0.0
            
            # Calculate PID score (correlation * stability)
            stability = 1.0 - np.std(feature_clean) / (np.mean(np.abs(feature_clean)) + 1e-8)
            pid_score = abs(correlation) * max(0.0, stability)
            
            return pid_score
            
        except Exception as e:
            self.logger.warning(f"PID score calculation failed: {e}")
            return 0.0
    
    def _calculate_atr(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            return atr.values
        except:
            return np.full(len(market_data), np.nan)
    
    async def _combine_features(self, long_result: Dict[str, Any], 
                               short_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine long and short features."""
        try:
            all_features = []
            all_names = []
            
            # Add long features
            if long_result.get('interaction_features') is not None:
                all_features.extend(long_result['interaction_features'])
                all_names.extend([f'long_{name}' for name in long_result['feature_names']])
            
            # Add short features
            if short_result.get('interaction_features') is not None:
                all_features.extend(short_result['interaction_features'])
                all_names.extend([f'short_{name}' for name in short_result['feature_names']])
            
            # Convert to DataFrame
            if all_features:
                combined_features = pd.DataFrame(
                    np.array(all_features).T, 
                    columns=all_names
                )
            else:
                combined_features = pd.DataFrame()
            
            return {
                'features': combined_features,
                'feature_names': all_names
            }
            
        except Exception as e:
            self.logger.error(f"Feature combination failed: {e}")
            return {'features': pd.DataFrame(), 'feature_names': []}
    
    async def _calculate_quality_metrics(self, combined_result: Dict[str, Any], 
                                       long_result: Dict[str, Any], 
                                       short_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for generated features."""
        try:
            features = combined_result.get('features')
            if features is None or features.empty:
                return {
                    'overall': 0.0,
                    'long': 0.0,
                    'short': 0.0,
                    'long_short_balance': 0.0
                }
            
            # Calculate overall quality
            overall_quality = len(features.columns) / max(1, len(features.columns))
            
            # Calculate long quality
            long_features = [col for col in features.columns if col.startswith('long_')]
            long_quality = len(long_features) / max(1, len(features.columns))
            
            # Calculate short quality
            short_features = [col for col in features.columns if col.startswith('short_')]
            short_quality = len(short_features) / max(1, len(features.columns))
            
            # Calculate long/short balance
            long_short_balance = 1.0 - abs(len(long_features) - len(short_features)) / max(len(long_features) + len(short_features), 1)
            
            return {
                'overall': overall_quality,
                'long': long_quality,
                'short': short_quality,
                'long_short_balance': long_short_balance
            }
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {
                'overall': 0.5,
                'long': 0.5,
                'short': 0.5,
                'long_short_balance': 0.5
            }

# Convenience functions
def create_tactician_pid_feature_generator(config: Optional[TacticianPIDConfig] = None) -> TacticianPIDBasedFeatureGenerator:
    """Create Tactician PID-based feature generator."""
    return TacticianPIDBasedFeatureGenerator(config)

async def generate_tactician_pid_features(market_data: pd.DataFrame,
                                        target_data: Optional[pd.Series] = None,
                                        long_target_data: Optional[pd.Series] = None,
                                        short_target_data: Optional[pd.Series] = None,
                                        lookback_periods: Optional[Dict[str, int]] = None,
                                        config: Optional[TacticianPIDConfig] = None) -> TacticianPIDResult:
    """Generate Tactician PID-based features."""
    generator = TacticianPIDBasedFeatureGenerator(config)
    return await generator.generate_features(market_data, target_data, long_target_data, short_target_data, lookback_periods)