"""
Analyst PID-Based Feature Generation - No Long/Short Differentiation

This component provides PID-based feature generation for Analyst models on 5m timeframe
without long/short differentiation. Uses unified approach for overall opportunity assessment.

Key features:
- No long/short differentiation (unified approach)
- Optimized for 5m timeframe
- Simplified PID analysis
- Focus on overall opportunity assessment
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Core dependencies
import numpy as np
import pandas as pd

# Import base component
from ...market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

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
class AnalystPIDConfig:
    """Configuration for Analyst PID-based feature generation."""
    # PID parameters (unified approach)
    pid_threshold: float = 0.1  # Minimum PID score for feature selection
    max_features_per_type: int = 50  # Maximum features per type
    enable_interaction_features: bool = True
    enable_polynomial_features: bool = True
    enable_cross_timeframe_features: bool = True
    
    # Timeframe specific (5m for Analyst)
    timeframe_minutes: int = 5
    max_horizon_minutes: int = 20  # 4 periods * 5 minutes
    
    # Quality thresholds
    min_feature_quality_score: float = 0.6
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.9
    
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size: int = 1000

@dataclass
class AnalystPIDResult:
    """Result of Analyst PID-based feature generation."""
    # Generated features
    interaction_features: Optional[pd.DataFrame] = None
    polynomial_features: Optional[pd.DataFrame] = None
    cross_timeframe_features: Optional[pd.DataFrame] = None
    combined_features: Optional[pd.DataFrame] = None
    
    # Feature names
    interaction_feature_names: List[str] = None
    polynomial_feature_names: List[str] = None
    cross_timeframe_feature_names: List[str] = None
    combined_feature_names: List[str] = None
    
    # Quality metrics
    overall_quality_score: float = 0.0
    feature_diversity_score: float = 0.0
    redundancy_score: float = 0.0
    stability_score: float = 0.0
    
    # Performance metrics
    total_features_generated: int = 0
    generation_time: float = 0.0
    generation_status: GenerationStatus = GenerationStatus.PENDING
    
    # Status
    success: bool = False
    error_message: Optional[str] = None

class AnalystPIDBasedFeatureGenerator:
    """
    Analyst PID-Based Feature Generator - NO LONG/SHORT DIFFERENTIATION.
    
    Generates features using PID analysis for Analyst models on 5m timeframe
    without long/short differentiation.
    """
    
    def __init__(self, config: Optional[AnalystPIDConfig] = None):
        """Initialize the Analyst PID-based feature generator."""
        self.config = config or AnalystPIDConfig()
        self.logger = get_logger('AnalystPIDBasedFeatureGenerator')
        
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
        
        self.logger.info("🚀 Analyst PID-Based Feature Generator initialized (NO LONG/SHORT DIFFERENTIATION)")
        self.logger.info(f"   → PID threshold: {self.config.pid_threshold}")
        self.logger.info(f"   → Max features per type: {self.config.max_features_per_type}")
        self.logger.info(f"   → Timeframe: {self.config.timeframe_minutes}m")
        self.logger.info(f"   → Memory optimization: {'Enabled' if self.config.enable_memory_optimization else 'Disabled'}")
    
    async def generate_features(self, 
                              market_data: pd.DataFrame,
                              target_data: Optional[pd.Series] = None,
                              lookback_periods: Optional[Dict[str, int]] = None) -> AnalystPIDResult:
        """
        Generate PID-based features for Analyst (UNIFIED APPROACH).
        
        Args:
            market_data: Market data for feature generation
            target_data: Target variable (optional, will use correlation-based PID if not provided)
            lookback_periods: Optimized lookback periods (optional)
            
        Returns:
            AnalystPIDResult with generated features
        """
        start_time = time.time()
        self.logger.info("🔍 Starting Analyst PID-based feature generation (UNIFIED APPROACH)")
        
        try:
            # Step 1: Validate input data
            validation_result = await self._validate_input_data(market_data, target_data)
            if not validation_result['is_valid']:
                return AnalystPIDResult(
                    success=False,
                    error_message=validation_result['error_message'],
                    generation_time=time.time() - start_time,
                    generation_status=GenerationStatus.FAILED
                )
            
            # Step 2: Prepare data for feature generation
            prepared_data = await self._prepare_data_for_generation(market_data, target_data)
            
            # Step 3: Generate interaction features
            interaction_result = await self._generate_interaction_features(
                prepared_data, lookback_periods
            )
            
            # Step 4: Generate polynomial features
            polynomial_result = await self._generate_polynomial_features(
                prepared_data, lookback_periods
            )
            
            # Step 5: Generate cross-timeframe features
            cross_timeframe_result = await self._generate_cross_timeframe_features(
                prepared_data, lookback_periods
            )
            
            # Step 6: Combine all features
            combined_result = await self._combine_features(
                interaction_result, polynomial_result, cross_timeframe_result
            )
            
            # Step 7: Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(combined_result)
            
            # Step 8: Create final result
            result = AnalystPIDResult(
                interaction_features=interaction_result.get('features'),
                polynomial_features=polynomial_result.get('features'),
                cross_timeframe_features=cross_timeframe_result.get('features'),
                combined_features=combined_result.get('features'),
                interaction_feature_names=interaction_result.get('feature_names', []),
                polynomial_feature_names=polynomial_result.get('feature_names', []),
                cross_timeframe_feature_names=cross_timeframe_result.get('feature_names', []),
                combined_feature_names=combined_result.get('feature_names', []),
                overall_quality_score=quality_metrics['overall'],
                feature_diversity_score=quality_metrics['diversity'],
                redundancy_score=quality_metrics['redundancy'],
                stability_score=quality_metrics['stability'],
                total_features_generated=len(combined_result.get('feature_names', [])),
                generation_time=time.time() - start_time,
                generation_status=GenerationStatus.COMPLETED,
                success=True
            )
            
            self.logger.info(f"✅ Analyst feature generation completed: {result.total_features_generated} features generated")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Analyst feature generation failed: {e}")
            return AnalystPIDResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                generation_status=GenerationStatus.FAILED
            )
    
    async def _validate_input_data(self, market_data: pd.DataFrame, 
                                 target_data: Optional[pd.Series]) -> Dict[str, Any]:
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
            
            # Check target data if provided
            if target_data is not None:
                if len(target_data) != len(market_data):
                    return {'is_valid': False, 'error_message': 'Target data length mismatch'}
                
                if target_data.isna().all():
                    return {'is_valid': False, 'error_message': 'Target data contains only NaN values'}
            
            return {'is_valid': True, 'error_message': None}
            
        except Exception as e:
            return {'is_valid': False, 'error_message': f'Validation error: {e}'}
    
    async def _prepare_data_for_generation(self, market_data: pd.DataFrame, 
                                         target_data: Optional[pd.Series]) -> Dict[str, Any]:
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
            
            return {
                'market_data': prepared_market_data,
                'target_data': prepared_target
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    async def _generate_interaction_features(self, prepared_data: Dict[str, Any], 
                                           lookback_periods: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """Generate interaction features using PID analysis (UNIFIED APPROACH)."""
        try:
            if not self.config.enable_interaction_features:
                return {'features': None, 'feature_names': []}
            
            market_data = prepared_data['market_data']
            target_data = prepared_data['target_data']
            
            # Generate base features for interaction
            base_features = self._generate_base_features(market_data)
            
            # Generate interaction features
            interaction_features = []
            interaction_names = []
            
            # Price-volume interactions
            if 'close' in market_data.columns and 'volume' in market_data.columns:
                price_volume_interactions = self._generate_price_volume_interactions(
                    market_data, base_features
                )
                interaction_features.extend(price_volume_interactions['features'])
                interaction_names.extend(price_volume_interactions['names'])
            
            # Price momentum interactions
            if 'close' in market_data.columns:
                momentum_interactions = self._generate_momentum_interactions(
                    market_data, base_features
                )
                interaction_features.extend(momentum_interactions['features'])
                interaction_names.extend(momentum_interactions['names'])
            
            # Volatility interactions
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                volatility_interactions = self._generate_volatility_interactions(
                    market_data, base_features
                )
                interaction_features.extend(volatility_interactions['features'])
                interaction_names.extend(volatility_interactions['names'])
            
            # Apply PID analysis to select best features
            if target_data is not None and len(interaction_features) > 0:
                selected_features, selected_names = self._apply_pid_selection(
                    interaction_features, interaction_names, target_data
                )
            else:
                # Use correlation-based selection if no target
                selected_features, selected_names = self._apply_correlation_selection(
                    interaction_features, interaction_names
                )
            
            # Limit number of features
            if len(selected_features) > self.config.max_features_per_type:
                selected_features = selected_features[:self.config.max_features_per_type]
                selected_names = selected_names[:self.config.max_features_per_type]
            
            return {
                'features': np.array(selected_features) if selected_features else None,
                'feature_names': selected_names
            }
            
        except Exception as e:
            self.logger.error(f"Interaction feature generation failed: {e}")
            return {'features': None, 'feature_names': []}
    
    async def _generate_polynomial_features(self, prepared_data: Dict[str, Any], 
                                          lookback_periods: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """Generate polynomial features using PID analysis (UNIFIED APPROACH)."""
        try:
            if not self.config.enable_polynomial_features:
                return {'features': None, 'feature_names': []}
            
            market_data = prepared_data['market_data']
            target_data = prepared_data['target_data']
            
            # Generate base features for polynomial expansion
            base_features = self._generate_base_features(market_data)
            
            # Generate polynomial features
            polynomial_features = []
            polynomial_names = []
            
            # Price polynomial features
            if 'close' in market_data.columns:
                price_polynomials = self._generate_price_polynomials(market_data, base_features)
                polynomial_features.extend(price_polynomials['features'])
                polynomial_names.extend(price_polynomials['names'])
            
            # Volume polynomial features
            if 'volume' in market_data.columns:
                volume_polynomials = self._generate_volume_polynomials(market_data, base_features)
                polynomial_features.extend(volume_polynomials['features'])
                polynomial_names.extend(volume_polynomials['names'])
            
            # Technical indicator polynomials
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                technical_polynomials = self._generate_technical_polynomials(market_data, base_features)
                polynomial_features.extend(technical_polynomials['features'])
                polynomial_names.extend(technical_polynomials['names'])
            
            # Apply PID analysis to select best features
            if target_data is not None and len(polynomial_features) > 0:
                selected_features, selected_names = self._apply_pid_selection(
                    polynomial_features, polynomial_names, target_data
                )
            else:
                # Use correlation-based selection if no target
                selected_features, selected_names = self._apply_correlation_selection(
                    polynomial_features, polynomial_names
                )
            
            # Limit number of features
            if len(selected_features) > self.config.max_features_per_type:
                selected_features = selected_features[:self.config.max_features_per_type]
                selected_names = selected_names[:self.config.max_features_per_type]
            
            return {
                'features': np.array(selected_features) if selected_features else None,
                'feature_names': selected_names
            }
            
        except Exception as e:
            self.logger.error(f"Polynomial feature generation failed: {e}")
            return {'features': None, 'feature_names': []}
    
    async def _generate_cross_timeframe_features(self, prepared_data: Dict[str, Any], 
                                                lookback_periods: Optional[Dict[str, int]]) -> Dict[str, Any]:
        """Generate cross-timeframe features using PID analysis (UNIFIED APPROACH)."""
        try:
            if not self.config.enable_cross_timeframe_features:
                return {'features': None, 'feature_names': []}
            
            market_data = prepared_data['market_data']
            target_data = prepared_data['target_data']
            
            # Generate cross-timeframe features
            cross_timeframe_features = []
            cross_timeframe_names = []
            
            # Multi-period features
            multi_period_features = self._generate_multi_period_features(market_data)
            cross_timeframe_features.extend(multi_period_features['features'])
            cross_timeframe_names.extend(multi_period_features['names'])
            
            # Trend features
            trend_features = self._generate_trend_features(market_data)
            cross_timeframe_features.extend(trend_features['features'])
            cross_timeframe_names.extend(trend_features['names'])
            
            # Momentum features
            momentum_features = self._generate_momentum_features(market_data)
            cross_timeframe_features.extend(momentum_features['features'])
            cross_timeframe_names.extend(momentum_features['names'])
            
            # Apply PID analysis to select best features
            if target_data is not None and len(cross_timeframe_features) > 0:
                selected_features, selected_names = self._apply_pid_selection(
                    cross_timeframe_features, cross_timeframe_names, target_data
                )
            else:
                # Use correlation-based selection if no target
                selected_features, selected_names = self._apply_correlation_selection(
                    cross_timeframe_features, cross_timeframe_names
                )
            
            # Limit number of features
            if len(selected_features) > self.config.max_features_per_type:
                selected_features = selected_features[:self.config.max_features_per_type]
                selected_names = selected_names[:self.config.max_features_per_type]
            
            return {
                'features': np.array(selected_features) if selected_features else None,
                'feature_names': selected_names
            }
            
        except Exception as e:
            self.logger.error(f"Cross-timeframe feature generation failed: {e}")
            return {'features': None, 'feature_names': []}
    
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
    
    def _generate_price_volume_interactions(self, market_data: pd.DataFrame, 
                                           base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate price-volume interaction features."""
        features = []
        names = []
        
        if 'close' in base_features and 'volume' in base_features:
            # Price-volume ratio
            pv_ratio = base_features['close'] / (base_features['volume'] + 1e-8)
            features.append(pv_ratio)
            names.append('price_volume_ratio')
            
            # Price-volume momentum
            pv_momentum = np.diff(pv_ratio, prepend=pv_ratio[0])
            features.append(pv_momentum)
            names.append('price_volume_momentum')
            
            # Price-volume correlation
            pv_corr = self._rolling_correlation(base_features['close'], base_features['volume'], 10)
            features.append(pv_corr)
            names.append('price_volume_correlation')
        
        return {'features': features, 'names': names}
    
    def _generate_momentum_interactions(self, market_data: pd.DataFrame, 
                                       base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate momentum interaction features."""
        features = []
        names = []
        
        if 'close' in base_features:
            # Price momentum
            price_momentum = np.diff(base_features['close'], prepend=base_features['close'][0])
            features.append(price_momentum)
            names.append('price_momentum')
            
            # Price acceleration
            price_acceleration = np.diff(price_momentum, prepend=price_momentum[0])
            features.append(price_acceleration)
            names.append('price_acceleration')
            
            # Price momentum ratio
            momentum_ratio = price_momentum / (base_features['close'] + 1e-8)
            features.append(momentum_ratio)
            names.append('price_momentum_ratio')
        
        return {'features': features, 'names': names}
    
    def _generate_volatility_interactions(self, market_data: pd.DataFrame, 
                                        base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate volatility interaction features."""
        features = []
        names = []
        
        if 'atr' in base_features and 'close' in base_features:
            # ATR ratio
            atr_ratio = base_features['atr'] / (base_features['close'] + 1e-8)
            features.append(atr_ratio)
            names.append('atr_ratio')
            
            # Volatility momentum
            vol_momentum = np.diff(base_features['atr'], prepend=base_features['atr'][0])
            features.append(vol_momentum)
            names.append('volatility_momentum')
        
        return {'features': features, 'names': names}
    
    def _generate_price_polynomials(self, market_data: pd.DataFrame, 
                                   base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate price polynomial features."""
        features = []
        names = []
        
        if 'close' in base_features:
            close_values = base_features['close']
            
            # Quadratic features
            close_squared = close_values ** 2
            features.append(close_squared)
            names.append('close_squared')
            
            # Cubic features
            close_cubed = close_values ** 3
            features.append(close_cubed)
            names.append('close_cubed')
            
            # Square root features
            close_sqrt = np.sqrt(np.abs(close_values))
            features.append(close_sqrt)
            names.append('close_sqrt')
            
            # Logarithmic features
            close_log = np.log(np.abs(close_values) + 1e-8)
            features.append(close_log)
            names.append('close_log')
        
        return {'features': features, 'names': names}
    
    def _generate_volume_polynomials(self, market_data: pd.DataFrame, 
                                    base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate volume polynomial features."""
        features = []
        names = []
        
        if 'volume' in base_features:
            volume_values = base_features['volume']
            
            # Quadratic features
            volume_squared = volume_values ** 2
            features.append(volume_squared)
            names.append('volume_squared')
            
            # Cubic features
            volume_cubed = volume_values ** 3
            features.append(volume_cubed)
            names.append('volume_cubed')
            
            # Square root features
            volume_sqrt = np.sqrt(np.abs(volume_values))
            features.append(volume_sqrt)
            names.append('volume_sqrt')
        
        return {'features': features, 'names': names}
    
    def _generate_technical_polynomials(self, market_data: pd.DataFrame, 
                                       base_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate technical indicator polynomial features."""
        features = []
        names = []
        
        if 'atr' in base_features:
            atr_values = base_features['atr']
            
            # ATR polynomial features
            atr_squared = atr_values ** 2
            features.append(atr_squared)
            names.append('atr_squared')
            
            atr_cubed = atr_values ** 3
            features.append(atr_cubed)
            names.append('atr_cubed')
        
        return {'features': features, 'names': names}
    
    def _generate_multi_period_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate multi-period features."""
        features = []
        names = []
        
        if 'close' in market_data.columns:
            # Multi-period moving averages
            for period in [5, 10, 20]:
                sma = market_data['close'].rolling(window=period).mean()
                features.append(sma.values)
                names.append(f'close_sma_{period}')
                
                ema = market_data['close'].ewm(span=period).mean()
                features.append(ema.values)
                names.append(f'close_ema_{period}')
        
        return {'features': features, 'names': names}
    
    def _generate_trend_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trend features."""
        features = []
        names = []
        
        if 'close' in market_data.columns:
            # Trend strength
            trend_strength = market_data['close'].rolling(window=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
            )
            features.append(trend_strength.values)
            names.append('trend_strength')
            
            # Trend direction
            trend_direction = np.sign(trend_strength)
            features.append(trend_direction.values)
            names.append('trend_direction')
        
        return {'features': features, 'names': names}
    
    def _generate_momentum_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate momentum features."""
        features = []
        names = []
        
        if 'close' in market_data.columns:
            # Momentum indicators
            for period in [5, 10, 20]:
                momentum = market_data['close'].pct_change(periods=period)
                features.append(momentum.values)
                names.append(f'momentum_{period}')
                
                # Rate of change
                roc = market_data['close'] / market_data['close'].shift(period) - 1
                features.append(roc.values)
                names.append(f'roc_{period}')
        
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
    
    def _apply_correlation_selection(self, features: List[np.ndarray], names: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Apply correlation-based selection when no target is available."""
        try:
            if len(features) == 0:
                return [], []
            
            # Calculate variance as proxy for information content
            variances = []
            for feature in features:
                clean_feature = feature[~np.isnan(feature)]
                if len(clean_feature) > 0:
                    variance = np.var(clean_feature)
                    variances.append(variance if not np.isnan(variance) else 0.0)
                else:
                    variances.append(0.0)
            
            # Select features with highest variance
            sorted_indices = np.argsort(variances)[::-1]
            selected_features = [features[i] for i in sorted_indices[:self.config.max_features_per_type]]
            selected_names = [names[i] for i in sorted_indices[:self.config.max_features_per_type]]
            
            return selected_features, selected_names
            
        except Exception as e:
            self.logger.warning(f"Correlation selection failed: {e}")
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
    
    def _rolling_correlation(self, x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling correlation between two arrays."""
        try:
            correlations = []
            for i in range(len(x)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                
                x_window = x[start_idx:end_idx]
                y_window = y[start_idx:end_idx]
                
                if len(x_window) >= 2:
                    corr = np.corrcoef(x_window, y_window)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                else:
                    correlations.append(0.0)
            
            return np.array(correlations)
        except:
            return np.zeros(len(x))
    
    async def _combine_features(self, interaction_result: Dict[str, Any], 
                               polynomial_result: Dict[str, Any], 
                               cross_timeframe_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all generated features."""
        try:
            all_features = []
            all_names = []
            
            # Add interaction features
            if interaction_result.get('features') is not None:
                all_features.extend(interaction_result['features'])
                all_names.extend(interaction_result['feature_names'])
            
            # Add polynomial features
            if polynomial_result.get('features') is not None:
                all_features.extend(polynomial_result['features'])
                all_names.extend(polynomial_result['feature_names'])
            
            # Add cross-timeframe features
            if cross_timeframe_result.get('features') is not None:
                all_features.extend(cross_timeframe_result['features'])
                all_names.extend(cross_timeframe_result['feature_names'])
            
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
    
    async def _calculate_quality_metrics(self, combined_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for generated features."""
        try:
            features = combined_result.get('features')
            if features is None or features.empty:
                return {
                    'overall': 0.0,
                    'diversity': 0.0,
                    'redundancy': 1.0,
                    'stability': 0.0
                }
            
            # Calculate diversity score
            diversity_score = len(features.columns) / max(1, len(features.columns))
            
            # Calculate redundancy score
            correlation_matrix = features.corr().abs()
            redundancy_score = correlation_matrix.mean().mean()
            
            # Calculate stability score
            stability_scores = []
            for col in features.columns:
                col_values = features[col].dropna()
                if len(col_values) > 1:
                    stability = 1.0 - (col_values.std() / (col_values.mean() + 1e-8))
                    stability_scores.append(max(0.0, stability))
            
            stability_score = np.mean(stability_scores) if stability_scores else 0.0
            
            # Calculate overall quality score
            overall_score = (diversity_score + (1.0 - redundancy_score) + stability_score) / 3
            
            return {
                'overall': overall_score,
                'diversity': diversity_score,
                'redundancy': redundancy_score,
                'stability': stability_score
            }
            
        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {
                'overall': 0.5,
                'diversity': 0.5,
                'redundancy': 0.5,
                'stability': 0.5
            }

# Convenience functions
def create_analyst_pid_feature_generator(config: Optional[AnalystPIDConfig] = None) -> AnalystPIDBasedFeatureGenerator:
    """Create Analyst PID-based feature generator."""
    return AnalystPIDBasedFeatureGenerator(config)

async def generate_analyst_pid_features(market_data: pd.DataFrame,
                                      target_data: Optional[pd.Series] = None,
                                      lookback_periods: Optional[Dict[str, int]] = None,
                                      config: Optional[AnalystPIDConfig] = None) -> AnalystPIDResult:
    """Generate Analyst PID-based features."""
    generator = AnalystPIDBasedFeatureGenerator(config)
    return await generator.generate_features(market_data, target_data, lookback_periods)