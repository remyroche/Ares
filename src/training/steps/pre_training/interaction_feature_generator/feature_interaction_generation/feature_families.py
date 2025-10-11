"""
Feature Family Builders with Cost-Aware Optimization

This module implements the final feature generation using optimized lookback
specifications, supporting both discrete and blended approaches while maintaining
production constraints and latency requirements.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import configuration and decision results
from .config import LookbackOptimizationConfig, FamilyType
from .decision import DecisionResult, LookbackSpec, DecisionType

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """Result of feature generation for a single family."""
    family: FamilyType
    feature_name: str
    feature_values: np.ndarray
    lookback_spec: LookbackSpec
    generation_time: float
    memory_usage_mb: float
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'family': self.family.value,
            'feature_name': self.feature_name,
            'feature_values': self.feature_values.tolist(),
            'lookback_spec': self.lookback_spec.to_dict(),
            'generation_time': self.generation_time,
            'memory_usage_mb': self.memory_usage_mb,
            'quality_score': self.quality_score
        }


class FeatureFamilyBuilder:
    """Base class for feature family builders."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def build_feature(self, data: pd.DataFrame, lookback_spec: LookbackSpec, 
                     feature_name: str) -> FeatureResult:
        """Build feature based on lookback specification."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            if lookback_spec.decision_type == DecisionType.DISCRETE:
                feature_values = self._build_discrete_feature(data, lookback_spec)
            elif lookback_spec.decision_type == DecisionType.BLEND:
                feature_values = self._build_blend_feature(data, lookback_spec)
            elif lookback_spec.decision_type == DecisionType.DEFAULT:
                feature_values = self._build_default_feature(data, lookback_spec)
            else:  # INACTIVE
                feature_values = np.zeros(len(data))
            
            generation_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            quality_score = self._calculate_quality_score(feature_values)
            
            return FeatureResult(
                family=self._get_family_type(),
                feature_name=feature_name,
                feature_values=feature_values,
                lookback_spec=lookback_spec,
                generation_time=generation_time,
                memory_usage_mb=memory_usage,
                quality_score=quality_score
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Feature generation failed: {e}")
            return FeatureResult(
                family=self._get_family_type(),
                feature_name=feature_name,
                feature_values=np.zeros(len(data)),
                lookback_spec=lookback_spec,
                generation_time=generation_time,
                memory_usage_mb=0.0,
                quality_score=0.0
            )
    
    def _build_discrete_feature(self, data: pd.DataFrame, lookback_spec: LookbackSpec) -> np.ndarray:
        """Build discrete feature with single lookback."""
        if lookback_spec.primary_lookback is None:
            return np.zeros(len(data))
        
        lookback = int(round(lookback_spec.primary_lookback))
        return self._compute_feature(data, lookback)
    
    def _build_blend_feature(self, data: pd.DataFrame, lookback_spec: LookbackSpec) -> np.ndarray:
        """Build blended feature with multiple lookbacks."""
        if (lookback_spec.primary_lookback is None or 
            lookback_spec.secondary_lookback is None or
            lookback_spec.blend_weights is None):
            return np.zeros(len(data))
        
        lookback1 = int(round(lookback_spec.primary_lookback))
        lookback2 = int(round(lookback_spec.secondary_lookback))
        w1, w2 = lookback_spec.blend_weights
        
        feature1 = self._compute_feature(data, lookback1)
        feature2 = self._compute_feature(data, lookback2)
        
        return w1 * feature1 + w2 * feature2
    
    def _build_default_feature(self, data: pd.DataFrame, lookback_spec: LookbackSpec) -> np.ndarray:
        """Build default feature."""
        if lookback_spec.primary_lookback is None:
            return np.zeros(len(data))
        
        lookback = int(round(lookback_spec.primary_lookback))
        return self._compute_feature(data, lookback)
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute the actual feature values (to be implemented by subclasses)."""
        # Default implementation - subclasses should override this
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Basic momentum calculation as fallback
        close_prices = data['close'].values
        if len(close_prices) < lookback:
            return np.zeros(len(close_prices))
        
        # Simple momentum calculation
        momentum = np.zeros_like(close_prices)
        for i in range(lookback, len(close_prices)):
            momentum[i] = (close_prices[i] - close_prices[i - lookback]) / close_prices[i - lookback]
        
        return momentum
    
    def _get_family_type(self) -> FamilyType:
        """Get the family type (to be implemented by subclasses)."""
        # Default implementation - subclasses should override this
        return FamilyType.MOMENTUM
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

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
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _calculate_quality_score(self, feature_values: np.ndarray) -> float:
        """Calculate quality score for generated feature."""
        try:
            # Remove NaN and infinite values
            clean_values = feature_values[np.isfinite(feature_values)]
            
            if len(clean_values) < 10:
                return 0.0
            
            # Calculate various quality metrics
            variance = np.var(clean_values)
            skewness = abs(stats.skew(clean_values))
            kurtosis = abs(stats.kurtosis(clean_values))
            
            # Normalize metrics to [0, 1] range
            variance_score = min(1.0, variance / 0.01)  # Higher variance is better
            skewness_score = max(0.0, 1.0 - skewness / 3.0)  # Lower skewness is better
            kurtosis_score = max(0.0, 1.0 - kurtosis / 10.0)  # Lower kurtosis is better
            
            # Weighted combination
            quality_score = (0.5 * variance_score + 
                           0.3 * skewness_score + 
                           0.2 * kurtosis_score)
            
            return float(quality_score)
            
        except Exception:
            return 0.0


class MomentumFeatureBuilder(FeatureFamilyBuilder):
    """Builder for momentum features."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.MOMENTUM
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute momentum feature."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Simple momentum calculation
        returns = data['close'].pct_change(lookback)
        return returns.fillna(0).values


class VolatilityFeatureBuilder(FeatureFamilyBuilder):
    """Builder for volatility features."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.VOLATILITY
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute EW volatility feature."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # EW volatility calculation
        returns = data['close'].pct_change()
        alpha = 2 / (lookback + 1)
        ew_var = returns.ewm(alpha=alpha).var()
        ew_vol = np.sqrt(ew_var)
        return ew_vol.fillna(0).values


class GKFeatureBuilder(FeatureFamilyBuilder):
    """Builder for Garman-Klass volatility features."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.GK
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute Garman-Klass volatility feature."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return np.zeros(len(data))
        
        # GK volatility calculation
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        
        gk_var = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        
        # Rolling mean
        gk_vol = np.sqrt(self._vectorbt_rolling_operation(gk_var, "mean", lookback))
        return gk_vol.fillna(0).values


class VWAPRollFeatureBuilder(FeatureFamilyBuilder):
    """Builder for VWAP rolling features."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.VWAP_ROLL
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute VWAP rolling feature."""
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            return np.zeros(len(data))
        
        # VWAP calculation
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=lookback).sum() / data['volume'].rolling(window=lookback).sum()
        
        # VWAP ratio
        vwap_ratio = data['close'] / vwap
        return vwap_ratio.fillna(1.0).values


class RSIFeatureBuilder(FeatureFamilyBuilder):
    """Builder for RSI features."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.RSI
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute RSI feature."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # RSI calculation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = self._vectorbt_rolling_operation(gain, "mean", lookback)
        avg_loss = self._vectorbt_rolling_operation(loss, "mean", lookback)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0).values


class AutocorrFeatureBuilder(FeatureFamilyBuilder):
    """Builder for autocorrelation features."""
    
    def _get_family_type(self) -> FamilyType:
        return FamilyType.AUTOCORR
    
    def _compute_feature(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Compute autocorrelation feature."""
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        # Autocorrelation calculation
        returns = data['close'].pct_change()
        
        # Rolling autocorrelation
        autocorr_values = np.zeros(len(data))
        
        for i in range(lookback, len(data)):
            window = returns.iloc[i-lookback:i+1]
            if len(window) > 1:
                corr = window.autocorr(lag=1)
                autocorr_values[i] = corr if not np.isnan(corr) else 0.0
        
        return autocorr_values


class FeatureFamilyFactory:
    """Factory for creating feature family builders."""
    
    @staticmethod
    def create_builder(family: FamilyType, config: LookbackOptimizationConfig) -> FeatureFamilyBuilder:
        """Create appropriate builder for family type."""
        builders = {
            FamilyType.MOMENTUM: MomentumFeatureBuilder,
            FamilyType.VOLATILITY: VolatilityFeatureBuilder,
            FamilyType.GK: GKFeatureBuilder,
            FamilyType.VWAP_ROLL: VWAPRollFeatureBuilder,
            FamilyType.RSI: RSIFeatureBuilder,
            FamilyType.AUTOCORR: AutocorrFeatureBuilder
        }
        
        builder_class = builders.get(family)
        if builder_class is None:
            raise ValueError(f"No builder available for family: {family}")
        
        return builder_class(config)


class MultiFamilyFeatureGenerator:
    """Generate features for multiple families using optimized lookbacks."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_features(self, data: pd.DataFrame, 
                         decisions: Dict[FamilyType, DecisionResult],
                         feature_names: Optional[Dict[FamilyType, str]] = None) -> Dict[FamilyType, FeatureResult]:
        """Generate features for all families using their decisions."""
        results = {}
        
        if feature_names is None:
            feature_names = {family: f"{family.value}_feature" for family in FamilyType}
        
        for family, decision in decisions.items():
            try:
                tprint_info(f"Generating {family.value} feature...")
                
                # Create builder for this family
                builder = FeatureFamilyFactory.create_builder(family, self.config)
                
                # Generate feature
                feature_result = builder.build_feature(
                    data, decision.lookback_spec, feature_names[family]
                )
                
                results[family] = feature_result
                
                tprint_info(f"Generated {family.value} feature in {feature_result.generation_time:.3f}s")
                tprint_info(f"Quality score: {feature_result.quality_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate {family.value} feature: {e}")
                continue
        
        return results
    
    def generate_all_symbols_features(self, 
                                    data: Dict[str, pd.DataFrame],
                                    decisions: Dict[str, Dict[FamilyType, DecisionResult]],
                                    feature_names: Optional[Dict[FamilyType, str]] = None) -> Dict[str, Dict[FamilyType, FeatureResult]]:
        """Generate features for all symbols and families."""
        all_results = {}
        
        for symbol, symbol_data in data.items():
            symbol_decisions = decisions.get(symbol, {})
            
            if not symbol_decisions:
                self.logger.warning(f"No decisions available for symbol {symbol}")
                continue
            
            try:
                symbol_results = self.generate_features(symbol_data, symbol_decisions, feature_names)
                all_results[symbol] = symbol_results
                
            except Exception as e:
                self.logger.error(f"Failed to generate features for symbol {symbol}: {e}")
                continue
        
        return all_results
    
    def create_feature_matrix(self, feature_results: Dict[FamilyType, FeatureResult]) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix from feature results."""
        features = []
        feature_names = []
        
        for family, result in feature_results.items():
            if result.feature_values is not None and len(result.feature_values) > 0:
                features.append(result.feature_values)
                feature_names.append(result.feature_name)
        
        if features:
            feature_matrix = np.column_stack(features)
        else:
            feature_matrix = np.array([]).reshape(0, 0)
        
        return feature_matrix, feature_names
    
    def generate_feature_report(self, all_results: Dict[str, Dict[FamilyType, FeatureResult]]) -> Dict[str, Any]:
        """Generate comprehensive feature generation report."""
        report = {
            'summary': {
                'total_symbols': len(all_results),
                'total_features': 0,
                'average_generation_time': 0.0,
                'average_quality_score': 0.0,
                'total_memory_usage_mb': 0.0
            },
            'family_summary': {},
            'symbol_summary': {},
            'quality_issues': []
        }
        
        all_generation_times = []
        all_quality_scores = []
        total_memory = 0.0
        
        # Initialize family summary
        for family in FamilyType:
            report['family_summary'][family.value] = {
                'total_generated': 0,
                'average_generation_time': 0.0,
                'average_quality_score': 0.0,
                'total_memory_usage_mb': 0.0
            }
        
        for symbol, symbol_results in all_results.items():
            symbol_generation_times = []
            symbol_quality_scores = []
            symbol_memory = 0.0
            
            for family, result in symbol_results.items():
                # Update family summary
                family_summary = report['family_summary'][family.value]
                family_summary['total_generated'] += 1
                family_summary['average_generation_time'] += result.generation_time
                family_summary['average_quality_score'] += result.quality_score
                family_summary['total_memory_usage_mb'] += result.memory_usage_mb
                
                # Collect metrics
                all_generation_times.append(result.generation_time)
                all_quality_scores.append(result.quality_score)
                total_memory += result.memory_usage_mb
                
                symbol_generation_times.append(result.generation_time)
                symbol_quality_scores.append(result.quality_score)
                symbol_memory += result.memory_usage_mb
                
                # Check for quality issues
                if result.quality_score < 0.3:
                    report['quality_issues'].append(
                        f"{symbol}-{family.value}: Low quality score {result.quality_score:.3f}"
                    )
            
            # Update symbol summary
            report['symbol_summary'][symbol] = {
                'total_features': len(symbol_results),
                'average_generation_time': np.mean(symbol_generation_times) if symbol_generation_times else 0.0,
                'average_quality_score': np.mean(symbol_quality_scores) if symbol_quality_scores else 0.0,
                'total_memory_usage_mb': symbol_memory
            }
        
        # Calculate averages
        if all_generation_times:
            report['summary']['average_generation_time'] = np.mean(all_generation_times)
        if all_quality_scores:
            report['summary']['average_quality_score'] = np.mean(all_quality_scores)
        
        report['summary']['total_memory_usage_mb'] = total_memory
        report['summary']['total_features'] = sum(len(symbol_results) for symbol_results in all_results.values())
        
        # Calculate family averages
        for family_summary in report['family_summary'].values():
            if family_summary['total_generated'] > 0:
                family_summary['average_generation_time'] /= family_summary['total_generated']
                family_summary['average_quality_score'] /= family_summary['total_generated']
        
        return report