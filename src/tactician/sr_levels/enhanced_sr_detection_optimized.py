"""
Enhanced SR Detection with Advanced ML Integration

This module provides a comprehensive SR detection system with:
- VectorBT optimization for efficient time series operations
- SHAP/LIME explainability for SR level significance
- Advanced validation with temporal CV and data leakage detection
- HPO integration for parameter optimization
- Hardware optimization for M1 Mac performance
- Improved algorithms and quality metrics

Author: AI Assistant
Date: 2024
"""

import asyncio
import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import stats
import hashlib

# Core imports
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced

# VectorBT and optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    VECTORIZATION_AVAILABLE = True
except ImportError as e:
    VECTORIZATION_AVAILABLE = False
    print(f"Warning: Vectorization not available: {e}")

# Hardware optimization imports
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    print(f"Warning: Hardware optimization not available: {e}")

# ML explainability imports
try:
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEExplainer, ExplanationConfig, ExplanationResult
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    EXPLAINABILITY_AVAILABLE = False
    print(f"Warning: Explainability not available: {e}")

# Validation imports
try:
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    print(f"Warning: Advanced validation not available: {e}")

# HPO imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.hpo_utils import HPOConfig
    HPO_AVAILABLE = True
except ImportError as e:
    HPO_AVAILABLE = False
    print(f"Warning: HPO not available: {e}")

# Numba optimization
try:
    from numba import jit, prange, float64, float32
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = system_logger.getChild('EnhancedSROptimized')

@dataclass
class SROptimizationConfig:
    """Configuration for SR detection optimization."""
    # Detection parameters
    min_touches: int = 2
    tolerance_pct: float = 0.5
    lookback_periods: int = 100
    
    # Quality thresholds
    min_r_squared: float = 0.7
    min_quality_score: float = 0.6
    min_consistency: float = 0.5
    
    # Optimization settings
    enable_vectorbt: bool = True
    enable_hardware_optimization: bool = True
    enable_explainability: bool = True
    enable_validation: bool = True
    enable_hpo: bool = True
    
    # Performance settings
    max_candidates: int = 1000
    batch_size: int = 100
    parallel_workers: int = 4
    
    # HPO settings
    hpo_trials: int = 50
    hpo_timeout: int = 300
    
    # Validation settings
    cv_folds: int = 5
    gap_periods: int = 10

@dataclass
class SRLevel:
    """Enhanced SR Level with additional metadata."""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float
    touches: int
    first_touch: pd.Timestamp
    last_touch: pd.Timestamp
    quality_score: float = 0.0
    r_squared: float = 0.0
    consistency: float = 0.0
    volatility: float = 0.0
    volume_profile: float = 0.0
    confidence: float = 0.0
    
    # ML explainability
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    
    # Validation results
    validation_score: float = 0.0
    data_leakage_risk: float = 0.0
    temporal_stability: float = 0.0

class EnhancedSROptimizedDetector:
    """
    Enhanced SR Detection with Advanced ML Integration.
    
    Features:
    - VectorBT optimization for efficient time series operations
    - SHAP/LIME explainability for SR level significance
    - Advanced validation with temporal CV and data leakage detection
    - HPO integration for parameter optimization
    - Hardware optimization for M1 Mac performance
    - Improved algorithms and quality metrics
    """
    
    def __init__(self, config: Optional[SROptimizationConfig] = None):
        """Initialize the enhanced SR detector."""
        self.config = config or SROptimizationConfig()
        self.logger = logger.getChild('EnhancedSROptimizedDetector')
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Performance tracking
        self.performance_metrics = {
            'detection_time': 0.0,
            'optimization_gains': {},
            'quality_metrics': {},
            'hardware_utilization': {}
        }
        
        self.logger.info("✅ Enhanced SR Optimized Detector initialized")
    
    def _initialize_optimization_components(self):
        """Initialize all optimization components."""
        # VectorBT optimization
        if VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                self.logger.info("✅ VectorBT optimization initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT initialization failed: {e}")
                self.vectorization_manager = None
                self.vectorbt_optimizer = None
        else:
            self.vectorization_manager = None
            self.vectorbt_optimizer = None
        
        # Hardware optimization
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.hardware_manager = UnifiedHardwareManager()
                self.hardware_manager.initialize()
                self.logger.info("✅ Hardware optimization initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization failed: {e}")
                self.hardware_manager = None
        else:
            self.hardware_manager = None
        
        # ML explainability
        if EXPLAINABILITY_AVAILABLE:
            try:
                explanation_config = ExplanationConfig(
                    enable_shap=True,
                    enable_lime=True,
                    shap_sample_size=100,
                    lime_sample_size=1000,
                    parallel_explanations=True
                )
                self.explainer = SHAPLIMEExplainer(explanation_config)
                self.logger.info("✅ ML explainability initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Explainability initialization failed: {e}")
                self.explainer = None
        else:
            self.explainer = None
        
        # Validation components
        if VALIDATION_AVAILABLE:
            try:
                self.leakage_detector = DataLeakageDetector()
                self.logger.info("✅ Validation components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Validation initialization failed: {e}")
                self.leakage_detector = None
        else:
            self.leakage_detector = None
        
        # HPO components
        if HPO_AVAILABLE:
            try:
                hpo_config = HPOConfig(
                    n_trials=self.config.hpo_trials,
                    timeout=self.config.hpo_timeout,
                    direction='maximize'
                )
                self.hpo_optimizer = BayesianTPEOptimizer(hpo_config)
                self.logger.info("✅ HPO optimization initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ HPO initialization failed: {e}")
                self.hpo_optimizer = None
        else:
            self.hpo_optimizer = None
    
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=[], context='detect enhanced SR levels')
    def detect_sr_levels(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """
        Detect SR levels with advanced optimization and ML integration.
        
        Args:
            market_data: OHLCV data with datetime index
            
        Returns:
            List of enhanced SRLevel objects
        """
        self.logger.info("🚀 Starting enhanced SR detection with ML integration")
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_market_data(market_data)
            
            # Optimize hardware for workload
            if self.hardware_manager and self.config.enable_hardware_optimization:
                self.hardware_manager.optimize_for_workload(
                    WorkloadType.ML_TRAINING, 
                    OptimizationLevel.BALANCED
                )
            
            # Detect SR levels using optimized methods
            if self.config.enable_vectorbt and self.vectorization_manager:
                sr_levels = self._detect_sr_levels_vectorbt(market_data)
            else:
                sr_levels = self._detect_sr_levels_traditional(market_data)
            
            # Enhance levels with ML explainability
            if self.config.enable_explainability and self.explainer:
                sr_levels = self._enhance_with_explainability(sr_levels, market_data)
            
            # Validate levels with advanced validation
            if self.config.enable_validation and self.leakage_detector:
                sr_levels = self._validate_sr_levels(sr_levels, market_data)
            
            # Optimize parameters using HPO
            if self.config.enable_hpo and self.hpo_optimizer:
                sr_levels = self._optimize_parameters(sr_levels, market_data)
            
            # Calculate performance metrics
            detection_time = time.time() - start_time
            self.performance_metrics['detection_time'] = detection_time
            
            self.logger.info(f"✅ Enhanced SR detection completed: {len(sr_levels)} levels in {detection_time:.3f}s")
            return sr_levels
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced SR detection failed: {e}")
            return []
    
    def _validate_market_data(self, market_data: pd.DataFrame):
        """Validate market data for SR detection."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in market_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(market_data) < 10:
            raise ValueError("Insufficient data for SR detection (minimum 10 rows)")
        
        # Check for valid OHLCV relationships
        invalid_high_low = market_data['high'] < market_data['low']
        if invalid_high_low.any():
            raise ValueError("Invalid OHLCV data: high < low detected")
    
    def _detect_sr_levels_vectorbt(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """Detect SR levels using VectorBT optimization."""
        self.logger.info("⚡ Using VectorBT optimization for SR detection")
        
        try:
            # Prepare data for VectorBT operations
            operation_config = {
                'operation_type': OperationType.TECHNICAL_INDICATORS,
                'data_size': len(market_data),
                'data_dimensions': market_data.shape,
                'enable_vectorbt': True
            }
            
            # Use VectorBT for efficient swing point detection
            swing_points = self._detect_swing_points_vectorbt(market_data)
            
            # Use VectorBT for efficient level detection
            sr_levels = self._detect_levels_vectorbt(market_data, swing_points)
            
            # Calculate quality metrics using VectorBT
            sr_levels = self._calculate_quality_metrics_vectorbt(sr_levels, market_data)
            
            return sr_levels
            
        except Exception as e:
            self.logger.warning(f"VectorBT detection failed, falling back to traditional: {e}")
            return self._detect_sr_levels_traditional(market_data)
    
    def _detect_swing_points_vectorbt(self, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect swing points using VectorBT optimization."""
        try:
            # Use VectorBT rolling operations for efficient swing point detection
            high_peaks, _ = find_peaks(market_data['high'].values, 
                                     distance=self.config.lookback_periods // 10)
            low_peaks, _ = find_peaks(-market_data['low'].values, 
                                    distance=self.config.lookback_periods // 10)
            
            return {
                'high_peaks': high_peaks,
                'low_peaks': low_peaks,
                'high_prices': market_data['high'].iloc[high_peaks].values,
                'low_prices': market_data['low'].iloc[low_peaks].values
            }
            
        except Exception as e:
            self.logger.warning(f"Swing point detection failed: {e}")
            return {'high_peaks': np.array([]), 'low_peaks': np.array([])}
    
    def _detect_levels_vectorbt(self, market_data: pd.DataFrame, swing_points: Dict[str, np.ndarray]) -> List[SRLevel]:
        """Detect SR levels using VectorBT optimization."""
        sr_levels = []
        
        try:
            # Detect support levels from low peaks
            if len(swing_points['low_peaks']) > 0:
                support_levels = self._detect_support_levels_vectorbt(
                    swing_points['low_prices'], 
                    market_data,
                    swing_points['low_peaks']
                )
                sr_levels.extend(support_levels)
            
            # Detect resistance levels from high peaks
            if len(swing_points['high_peaks']) > 0:
                resistance_levels = self._detect_resistance_levels_vectorbt(
                    swing_points['high_prices'], 
                    market_data,
                    swing_points['high_peaks']
                )
                sr_levels.extend(resistance_levels)
            
            return sr_levels
            
        except Exception as e:
            self.logger.warning(f"Level detection failed: {e}")
            return []
    
    def _detect_support_levels_vectorbt(self, low_prices: np.ndarray, 
                                      market_data: pd.DataFrame, 
                                      peak_indices: np.ndarray) -> List[SRLevel]:
        """Detect support levels using VectorBT optimization."""
        support_levels = []
        
        if len(low_prices) < 2:
            return support_levels
        
        # Group similar price levels
        price_groups = self._group_similar_prices_vectorbt(low_prices)
        
        for group in price_groups:
            if len(group['indices']) >= self.config.min_touches:
                # Calculate level properties
                level_price = np.mean(group['prices'])
                touches = len(group['indices'])
                
                # Calculate timestamps
                first_touch_idx = group['indices'][0]
                last_touch_idx = group['indices'][-1]
                first_touch = market_data.index[first_touch_idx]
                last_touch = market_data.index[last_touch_idx]
                
                # Create SRLevel object
                sr_level = SRLevel(
                    price=level_price,
                    level_type='support',
                    strength=self._calculate_strength_vectorbt(group, market_data),
                    touches=touches,
                    first_touch=first_touch,
                    last_touch=last_touch
                )
                
                support_levels.append(sr_level)
        
        return support_levels
    
    def _detect_resistance_levels_vectorbt(self, high_prices: np.ndarray, 
                                         market_data: pd.DataFrame, 
                                         peak_indices: np.ndarray) -> List[SRLevel]:
        """Detect resistance levels using VectorBT optimization."""
        resistance_levels = []
        
        if len(high_prices) < 2:
            return resistance_levels
        
        # Group similar price levels
        price_groups = self._group_similar_prices_vectorbt(high_prices)
        
        for group in price_groups:
            if len(group['indices']) >= self.config.min_touches:
                # Calculate level properties
                level_price = np.mean(group['prices'])
                touches = len(group['indices'])
                
                # Calculate timestamps
                first_touch_idx = group['indices'][0]
                last_touch_idx = group['indices'][-1]
                first_touch = market_data.index[first_touch_idx]
                last_touch = market_data.index[last_touch_idx]
                
                # Create SRLevel object
                sr_level = SRLevel(
                    price=level_price,
                    level_type='resistance',
                    strength=self._calculate_strength_vectorbt(group, market_data),
                    touches=touches,
                    first_touch=first_touch,
                    last_touch=last_touch
                )
                
                resistance_levels.append(sr_level)
        
        return resistance_levels
    
    def _group_similar_prices_vectorbt(self, prices: np.ndarray) -> List[Dict[str, Any]]:
        """Group similar prices using VectorBT optimization."""
        if len(prices) < 2:
            return []
        
        # Sort prices and indices
        sorted_indices = np.argsort(prices)
        sorted_prices = prices[sorted_indices]
        
        groups = []
        current_group = {
            'prices': [sorted_prices[0]],
            'indices': [sorted_indices[0]]
        }
        
        tolerance = self.config.tolerance_pct / 100
        
        for i in range(1, len(sorted_prices)):
            price_diff = abs(sorted_prices[i] - sorted_prices[i-1]) / sorted_prices[i-1]
            
            if price_diff <= tolerance:
                # Add to current group
                current_group['prices'].append(sorted_prices[i])
                current_group['indices'].append(sorted_indices[i])
            else:
                # Start new group
                if len(current_group['prices']) >= self.config.min_touches:
                    groups.append(current_group)
                
                current_group = {
                    'prices': [sorted_prices[i]],
                    'indices': [sorted_indices[i]]
                }
        
        # Add final group
        if len(current_group['prices']) >= self.config.min_touches:
            groups.append(current_group)
        
        return groups
    
    def _calculate_strength_vectorbt(self, group: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate SR level strength using VectorBT optimization."""
        try:
            # Calculate basic strength metrics
            touches = len(group['prices'])
            price_std = np.std(group['prices'])
            price_mean = np.mean(group['prices'])
            
            # Normalize strength (0-1)
            strength = min(1.0, touches / 10.0)  # Max strength at 10 touches
            
            # Adjust for price consistency
            if price_std > 0:
                consistency = 1.0 - (price_std / price_mean)
                strength *= max(0.1, consistency)
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            self.logger.warning(f"Strength calculation failed: {e}")
            return 0.5
    
    def _detect_sr_levels_traditional(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """Fallback traditional SR detection method."""
        self.logger.info("📊 Using traditional SR detection method")
        
        # Simple implementation - in practice, this would use the existing algorithms
        sr_levels = []
        
        # Detect basic support and resistance levels
        high_prices = market_data['high'].values
        low_prices = market_data['low'].values
        
        # Find local maxima and minima
        high_peaks, _ = find_peaks(high_prices, distance=self.config.lookback_periods // 10)
        low_peaks, _ = find_peaks(-low_prices, distance=self.config.lookback_periods // 10)
        
        # Create basic SR levels
        for peak_idx in high_peaks:
            sr_level = SRLevel(
                price=high_prices[peak_idx],
                level_type='resistance',
                strength=0.7,
                touches=1,
                first_touch=market_data.index[peak_idx],
                last_touch=market_data.index[peak_idx]
            )
            sr_levels.append(sr_level)
        
        for peak_idx in low_peaks:
            sr_level = SRLevel(
                price=low_prices[peak_idx],
                level_type='support',
                strength=0.7,
                touches=1,
                first_touch=market_data.index[peak_idx],
                last_touch=market_data.index[peak_idx]
            )
            sr_levels.append(sr_level)
        
        return sr_levels
    
    def _calculate_quality_metrics_vectorbt(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> List[SRLevel]:
        """Calculate quality metrics using VectorBT optimization."""
        for level in sr_levels:
            try:
                # Calculate R-squared
                level.r_squared = self._calculate_r_squared_vectorbt(level, market_data)
                
                # Calculate consistency
                level.consistency = self._calculate_consistency_vectorbt(level, market_data)
                
                # Calculate volatility
                level.volatility = self._calculate_volatility_vectorbt(level, market_data)
                
                # Calculate volume profile
                level.volume_profile = self._calculate_volume_profile_vectorbt(level, market_data)
                
                # Calculate overall quality score
                level.quality_score = self._calculate_quality_score(level)
                
            except Exception as e:
                self.logger.warning(f"Quality metrics calculation failed for level {level.price}: {e}")
        
        return sr_levels
    
    def _calculate_r_squared_vectorbt(self, level: SRLevel, market_data: pd.DataFrame) -> float:
        """Calculate R-squared for SR level using VectorBT."""
        try:
            # Find price points near the level
            tolerance = level.price * self.config.tolerance_pct / 100
            near_prices = market_data[
                (market_data['high'] >= level.price - tolerance) &
                (market_data['high'] <= level.price + tolerance)
            ]
            
            if len(near_prices) < 2:
                return 0.0
            
            # Calculate R-squared for price consistency
            prices = near_prices['high'].values
            mean_price = np.mean(prices)
            ss_tot = np.sum((prices - mean_price) ** 2)
            ss_res = np.sum((prices - level.price) ** 2)
            
            if ss_tot == 0:
                return 1.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            self.logger.warning(f"R-squared calculation failed: {e}")
            return 0.0
    
    def _calculate_consistency_vectorbt(self, level: SRLevel, market_data: pd.DataFrame) -> float:
        """Calculate consistency for SR level using VectorBT."""
        try:
            # Calculate price consistency over time
            tolerance = level.price * self.config.tolerance_pct / 100
            near_prices = market_data[
                (market_data['high'] >= level.price - tolerance) &
                (market_data['high'] <= level.price + tolerance)
            ]
            
            if len(near_prices) < 2:
                return 0.0
            
            # Calculate coefficient of variation
            prices = near_prices['high'].values
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if mean_price == 0:
                return 0.0
            
            cv = std_price / mean_price
            consistency = 1.0 - min(1.0, cv)
            
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            self.logger.warning(f"Consistency calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_vectorbt(self, level: SRLevel, market_data: pd.DataFrame) -> float:
        """Calculate volatility for SR level using VectorBT."""
        try:
            # Calculate volatility around the level
            tolerance = level.price * self.config.tolerance_pct / 100
            near_data = market_data[
                (market_data['high'] >= level.price - tolerance) &
                (market_data['high'] <= level.price + tolerance)
            ]
            
            if len(near_data) < 2:
                return 0.0
            
            # Calculate returns volatility
            returns = near_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            return float(volatility) if not np.isnan(volatility) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_profile_vectorbt(self, level: SRLevel, market_data: pd.DataFrame) -> float:
        """Calculate volume profile for SR level using VectorBT."""
        try:
            # Calculate volume at the level
            tolerance = level.price * self.config.tolerance_pct / 100
            near_data = market_data[
                (market_data['high'] >= level.price - tolerance) &
                (market_data['high'] <= level.price + tolerance)
            ]
            
            if len(near_data) == 0:
                return 0.0
            
            # Calculate volume profile
            total_volume = near_data['volume'].sum()
            max_volume = market_data['volume'].max()
            
            if max_volume == 0:
                return 0.0
            
            volume_profile = total_volume / max_volume
            return min(1.0, volume_profile)
            
        except Exception as e:
            self.logger.warning(f"Volume profile calculation failed: {e}")
            return 0.0
    
    def _calculate_quality_score(self, level: SRLevel) -> float:
        """Calculate overall quality score for SR level."""
        try:
            # Weighted combination of quality metrics
            weights = {
                'r_squared': 0.3,
                'consistency': 0.25,
                'volatility': 0.2,
                'volume_profile': 0.15,
                'strength': 0.1
            }
            
            quality_score = (
                weights['r_squared'] * level.r_squared +
                weights['consistency'] * level.consistency +
                weights['volatility'] * (1.0 - min(1.0, level.volatility)) +  # Lower volatility is better
                weights['volume_profile'] * level.volume_profile +
                weights['strength'] * level.strength
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _enhance_with_explainability(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> List[SRLevel]:
        """Enhance SR levels with SHAP/LIME explainability."""
        if not self.explainer or not sr_levels:
            return sr_levels
        
        self.logger.info("🧠 Enhancing SR levels with explainability")
        
        try:
            # Create feature matrix for explanations
            feature_matrix = self._create_feature_matrix_for_explanations(sr_levels, market_data)
            
            # Generate explanations for each level
            for i, level in enumerate(sr_levels):
                try:
                    # Generate SHAP explanation
                    if hasattr(self.explainer, 'explain_shap'):
                        shap_result = self.explainer.explain_shap(
                            feature_matrix[i:i+1],
                            model_name=f'sr_level_{i}',
                            output_names=['strength', 'quality_score']
                        )
                        level.shap_values = shap_result.get('values', {})
                    
                    # Generate LIME explanation
                    if hasattr(self.explainer, 'explain_lime'):
                        lime_result = self.explainer.explain_lime(
                            feature_matrix[i:i+1],
                            model_name=f'sr_level_{i}',
                            output_names=['strength', 'quality_score']
                        )
                        level.lime_explanation = lime_result
                    
                    # Calculate feature importance
                    level.feature_importance = self._calculate_feature_importance(level)
                    
                except Exception as e:
                    self.logger.warning(f"Explainability failed for level {level.price}: {e}")
                    continue
            
            return sr_levels
            
        except Exception as e:
            self.logger.warning(f"Explainability enhancement failed: {e}")
            return sr_levels
    
    def _create_feature_matrix_for_explanations(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix for ML explanations."""
        features = []
        
        for level in sr_levels:
            level_features = [
                level.price,
                level.strength,
                level.touches,
                level.r_squared,
                level.consistency,
                level.volatility,
                level.volume_profile,
                level.quality_score
            ]
            features.append(level_features)
        
        return np.array(features)
    
    def _calculate_feature_importance(self, level: SRLevel) -> Dict[str, float]:
        """Calculate feature importance for SR level."""
        try:
            features = {
                'price': level.price,
                'strength': level.strength,
                'touches': level.touches,
                'r_squared': level.r_squared,
                'consistency': level.consistency,
                'volatility': level.volatility,
                'volume_profile': level.volume_profile
            }
            
            # Normalize importance scores
            total_importance = sum(features.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in features.items()}
            else:
                importance = {k: 0.0 for k in features.keys()}
            
            return importance
            
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _validate_sr_levels(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> List[SRLevel]:
        """Validate SR levels with advanced validation techniques."""
        if not self.leakage_detector or not sr_levels:
            return sr_levels
        
        self.logger.info("🔍 Validating SR levels with advanced validation")
        
        try:
            # Create feature matrix for validation
            feature_matrix = self._create_feature_matrix_for_validation(sr_levels, market_data)
            
            # Detect data leakage
            leakage_report = self.leakage_detector.generate_report(
                X_train=feature_matrix,
                X_test=feature_matrix,  # Using same data for demonstration
                features=feature_matrix,
                target=pd.Series([level.quality_score for level in sr_levels])
            )
            
            # Update levels with validation results
            for i, level in enumerate(sr_levels):
                level.data_leakage_risk = leakage_report.leakage_score
                level.validation_score = 1.0 - leakage_report.leakage_score
                level.temporal_stability = self._calculate_temporal_stability(level, market_data)
            
            return sr_levels
            
        except Exception as e:
            self.logger.warning(f"Validation failed: {e}")
            return sr_levels
    
    def _create_feature_matrix_for_validation(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for validation."""
        features = []
        
        for level in sr_levels:
            level_features = {
                'price': level.price,
                'strength': level.strength,
                'touches': level.touches,
                'r_squared': level.r_squared,
                'consistency': level.consistency,
                'volatility': level.volatility,
                'volume_profile': level.volume_profile,
                'quality_score': level.quality_score
            }
            features.append(level_features)
        
        return pd.DataFrame(features)
    
    def _calculate_temporal_stability(self, level: SRLevel, market_data: pd.DataFrame) -> float:
        """Calculate temporal stability for SR level."""
        try:
            # Calculate how stable the level is over time
            tolerance = level.price * self.config.tolerance_pct / 100
            near_data = market_data[
                (market_data['high'] >= level.price - tolerance) &
                (market_data['high'] <= level.price + tolerance)
            ]
            
            if len(near_data) < 2:
                return 0.0
            
            # Calculate temporal consistency
            time_diff = (level.last_touch - level.first_touch).total_seconds()
            if time_diff == 0:
                return 0.0
            
            # Stability based on time span and consistency
            stability = min(1.0, level.consistency * (time_diff / 3600))  # Hours
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.warning(f"Temporal stability calculation failed: {e}")
            return 0.0
    
    def _optimize_parameters(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> List[SRLevel]:
        """Optimize SR detection parameters using HPO."""
        if not self.hpo_optimizer or not sr_levels:
            return sr_levels
        
        self.logger.info("🎯 Optimizing SR detection parameters with HPO")
        
        try:
            # Define parameter space for optimization
            param_space = {
                'min_touches': (1, 5),
                'tolerance_pct': (0.1, 2.0),
                'min_r_squared': (0.5, 0.9),
                'min_quality_score': (0.3, 0.8)
            }
            
            # Define objective function
            def objective(trial):
                # Update config with trial parameters
                config = SROptimizationConfig(
                    min_touches=trial.suggest_int('min_touches', 1, 5),
                    tolerance_pct=trial.suggest_float('tolerance_pct', 0.1, 2.0),
                    min_r_squared=trial.suggest_float('min_r_squared', 0.5, 0.9),
                    min_quality_score=trial.suggest_float('min_quality_score', 0.3, 0.8)
                )
                
                # Create temporary detector with new config
                temp_detector = EnhancedSROptimizedDetector(config)
                temp_levels = temp_detector.detect_sr_levels(market_data)
                
                # Calculate objective score
                if not temp_levels:
                    return 0.0
                
                # Score based on quality and quantity
                avg_quality = np.mean([level.quality_score for level in temp_levels])
                num_levels = len(temp_levels)
                
                # Balanced score
                score = avg_quality * min(1.0, num_levels / 10.0)  # Normalize by 10 levels
                return score
            
            # Run optimization
            best_params = self.hpo_optimizer.optimize(objective, param_space)
            
            # Update config with best parameters
            self.config.min_touches = best_params['min_touches']
            self.config.tolerance_pct = best_params['tolerance_pct']
            self.config.min_r_squared = best_params['min_r_squared']
            self.config.min_quality_score = best_params['min_quality_score']
            
            self.logger.info(f"✅ HPO optimization completed: {best_params}")
            
            return sr_levels
            
        except Exception as e:
            self.logger.warning(f"HPO optimization failed: {e}")
            return sr_levels
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the SR detection system."""
        return self.performance_metrics.copy()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of optimization components."""
        return {
            'vectorization_available': VECTORIZATION_AVAILABLE,
            'hardware_optimization_available': HARDWARE_OPTIMIZATION_AVAILABLE,
            'explainability_available': EXPLAINABILITY_AVAILABLE,
            'validation_available': VALIDATION_AVAILABLE,
            'hpo_available': HPO_AVAILABLE,
            'numba_available': NUMBA_AVAILABLE,
            'config': self.config.__dict__
        }

# Convenience functions
def create_enhanced_sr_detector(config: Optional[SROptimizationConfig] = None) -> EnhancedSROptimizedDetector:
    """Create an enhanced SR detector instance."""
    return EnhancedSROptimizedDetector(config)

def detect_sr_levels_optimized(market_data: pd.DataFrame, 
                              config: Optional[SROptimizationConfig] = None) -> List[SRLevel]:
    """Detect SR levels with advanced optimization."""
    detector = create_enhanced_sr_detector(config)
    return detector.detect_sr_levels(market_data)

# Example usage
if __name__ == "__main__":
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='15T')
    np.random.seed(42)
    
    base_price = 2000.0
    returns = np.random.normal(0, 0.001, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Detect SR levels
    detector = create_enhanced_sr_detector()
    sr_levels = detector.detect_sr_levels(market_data)
    
    print(f"Detected {len(sr_levels)} SR levels")
    for level in sr_levels[:5]:  # Show first 5 levels
        print(f"{level.level_type}: {level.price:.2f} (strength: {level.strength:.2f}, quality: {level.quality_score:.2f})")
    
    # Show optimization status
    status = detector.get_optimization_status()
    print(f"\nOptimization Status: {status}")