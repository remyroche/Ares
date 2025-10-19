"""
Variant Generator for Feature Interaction Generation

Generates feature variants including vol-normalized, VWAP-based, and combined transforms
for multi-timeframe analysis in the three-phase interaction generation pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
import logging

from src.utils.tprint import tprint

# Import optimization utilities
from .advanced_memory_manager import AdvancedMemoryManager, MemoryConfig
from .enhanced_vectorbt_manager import EnhancedVectorBTManager, VectorBTConfig
from .m1_parallel_processor import M1ParallelProcessor, ParallelConfig
from .data_structure_optimizer import DataStructureOptimizer, OptimizationConfig

# Import existing transforms
try:
    from src.features_common.transforms.scaling_normalization import RobustScaler, StandardScaler
    from src.features_common.transforms.base_scaler import BaseScaler
    TRANSFORMS_AVAILABLE = True
except ImportError:
    TRANSFORMS_AVAILABLE = False
    warnings.warn("Feature transforms not available. Install with: pip install scikit-learn")

# Import VectorBT for efficient operations
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_sum
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available for variant generation")

logger = logging.getLogger(__name__)

@dataclass
class VariantConfig:
    """Configuration for variant generation."""
    
    # Volatility normalization settings
    vol_window: int = 20
    vol_method: str = 'std'  # 'std', 'atr', 'robust_std'
    vol_min_threshold: float = 1e-8
    vol_winsorize_quantiles: Tuple[float, float] = (0.05, 0.95)
    
    # VWAP settings
    vwap_window: int = 20
    volume_min_threshold: float = 1e-6
    volume_epsilon: float = 1e-8
    
    # Winsorization settings
    winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)
    
    # Multi-timeframe settings
    base_timeframe: str = '15m'
    ht_lag_periods: int = 1  # Lag HTF features by 1 bar
    
    # Memory optimization
    chunk_size: int = 10000
    enable_memory_optimization: bool = True

class FeatureVariantGenerator:
    """
    Generates feature variants for interaction analysis.
    
    Creates four variant types:
    1. Raw (baseline)
    2. Vol-normalized (divided by rolling volatility)
    3. VWAP-based (volume-weighted transforms)
    4. Combined (vol-normalized + VWAP-based)
    """
    
    def __init__(self, config: Optional[VariantConfig] = None):
        """Initialize the variant generator with advanced optimizations."""
        self.config = config or VariantConfig()
        self.logger = logger
        
        # Initialize optimization components
        self.memory_manager = AdvancedMemoryManager(MemoryConfig())
        self.vectorbt_manager = EnhancedVectorBTManager(VectorBTConfig())
        self.parallel_processor = M1ParallelProcessor(ParallelConfig())
        self.data_optimizer = DataStructureOptimizer(OptimizationConfig())
        
        # Initialize scalers if available
        if TRANSFORMS_AVAILABLE:
            self.robust_scaler = RobustScaler()
            self.standard_scaler = StandardScaler()
        else:
            self.robust_scaler = None
            self.standard_scaler = None
            
        # Cache for computed variants
        self._variant_cache = {}
        
    def _winsorize_series(self, series: pd.Series, 
                         quantiles: Optional[Tuple[float, float]] = None) -> pd.Series:
        """Winsorize a series to handle outliers."""
        if quantiles is None:
            quantiles = self.config.winsorize_quantiles
            
        lower_quantile, upper_quantile = quantiles
        lower_bound = series.quantile(lower_quantile)
        upper_bound = series.quantile(upper_quantile)
        
        return series.clip(lower_bound, upper_bound)
    
    def _compute_rolling_volatility(self, series: pd.Series, 
                                  price_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Compute rolling volatility using specified method."""
        if self.config.vol_method == 'std':
            vol = rolling_std(series, window=self.config.vol_window) if VECTORBT_AVAILABLE else series.rolling(self.config.vol_window).std()
        elif self.config.vol_method == 'atr' and price_data is not None:
            # True Range calculation
            high = price_data.get('high', series)
            low = price_data.get('low', series)
            close = price_data.get('close', series)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            if VECTORBT_AVAILABLE:
                vol = rolling_mean(true_range, window=self.config.vol_window)
            else:
                vol = true_range.rolling(self.config.vol_window).mean()
        elif self.config.vol_method == 'robust_std':
            # Use robust standard deviation (MAD-based)
            median = series.rolling(self.config.vol_window).median()
            mad = abs(series - median).rolling(self.config.vol_window).median()
            vol = mad * 1.4826  # Convert MAD to standard deviation
        else:
            # Fallback to standard deviation
            vol = rolling_std(series, window=self.config.vol_window) if VECTORBT_AVAILABLE else series.rolling(self.config.vol_window).std()
            
        # Ensure minimum threshold
        vol = vol.clip(lower=self.config.vol_min_threshold)
        return vol
    
    def _compute_vwap(self, series: pd.Series, 
                     volume: pd.Series) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        # Guard against low volume
        volume_clipped = volume.clip(lower=self.config.volume_min_threshold)
        
        if VECTORBT_AVAILABLE:
            vwap = rolling_sum(series * volume_clipped, window=self.config.vwap_window) / rolling_sum(volume_clipped, window=self.config.vwap_window)
        else:
            vwap = (series * volume_clipped).rolling(self.config.vwap_window).sum() / volume_clipped.rolling(self.config.vwap_window).sum()
            
        return vwap.fillna(method='ffill')
    
    def generate_vol_normalized_variants(self, 
                                       features_df: pd.DataFrame,
                                       price_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate volatility-normalized variants."""
        tprint("📊 Generating volatility-normalized variants...")
        
        vol_normalized_features = {}
        
        for col in features_df.columns:
            series = features_df[col].copy()
            
            # Compute rolling volatility
            vol = self._compute_rolling_volatility(series, price_data)
            
            # Volatility normalization with winsorization
            vol_norm = series / (vol + self.config.volume_epsilon)
            vol_norm = self._winsorize_series(vol_norm, self.config.vol_winsorize_quantiles)
            
            # Add suffix to indicate variant
            vol_normalized_features[f"{col}_vol_norm"] = vol_norm
            
        result_df = pd.DataFrame(vol_normalized_features, index=features_df.index)
        tprint(f"✅ Generated {len(result_df.columns)} volatility-normalized features")
        
        return result_df
    
    def generate_vwap_variants(self, 
                             features_df: pd.DataFrame,
                             volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate VWAP-based variants."""
        tprint("📊 Generating VWAP-based variants...")
        
        if volume is None:
            tprint("⚠️ No volume data provided, skipping VWAP variants")
            return pd.DataFrame(index=features_df.index)
            
        vwap_features = {}
        
        for col in features_df.columns:
            series = features_df[col].copy()
            
            # Compute VWAP
            vwap = self._compute_vwap(series, volume)
            
            # VWAP-based transforms
            vwap_ratio = series / (vwap + self.config.volume_epsilon)
            vwap_diff = series - vwap
            
            # Winsorize to handle outliers
            vwap_ratio = self._winsorize_series(vwap_ratio)
            vwap_diff = self._winsorize_series(vwap_diff)
            
            vwap_features[f"{col}_vwap_ratio"] = vwap_ratio
            vwap_features[f"{col}_vwap_diff"] = vwap_diff
            
        result_df = pd.DataFrame(vwap_features, index=features_df.index)
        tprint(f"✅ Generated {len(result_df.columns)} VWAP-based features")
        
        return result_df
    
    def generate_combined_variants(self, 
                                 features_df: pd.DataFrame,
                                 vol_normalized_df: pd.DataFrame,
                                 vwap_df: pd.DataFrame) -> pd.DataFrame:
        """Generate combined variants (vol-normalized + VWAP)."""
        tprint("📊 Generating combined variants...")
        
        combined_features = {}
        
        # Combine vol-normalized with VWAP variants
        for vol_col in vol_normalized_df.columns:
            base_feature = vol_col.replace('_vol_norm', '')
            
            if base_feature in features_df.columns:
                # Find corresponding VWAP variants
                vwap_ratio_col = f"{base_feature}_vwap_ratio"
                vwap_diff_col = f"{base_feature}_vwap_diff"
                
                vol_series = vol_normalized_df[vol_col]
                
                if vwap_ratio_col in vwap_df.columns:
                    vwap_ratio = vwap_df[vwap_ratio_col]
                    # Combined: vol-normalized * VWAP ratio
                    combined_features[f"{base_feature}_vol_vwap_combined"] = vol_series * vwap_ratio
                
                if vwap_diff_col in vwap_df.columns:
                    vwap_diff = vwap_df[vwap_diff_col]
                    # Combined: vol-normalized + VWAP difference
                    combined_features[f"{base_feature}_vol_vwap_sum"] = vol_series + vwap_diff
                    
        result_df = pd.DataFrame(combined_features, index=features_df.index)
        tprint(f"✅ Generated {len(result_df.columns)} combined features")
        
        return result_df
    
    def generate_multi_timeframe_variants(self, 
                                        features_df: pd.DataFrame,
                                        timeframes: List[str],
                                        price_data: Optional[pd.DataFrame] = None,
                                        volume: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        """Generate variants across multiple timeframes."""
        tprint(f"📊 Generating multi-timeframe variants for {len(timeframes)} timeframes...")
        
        timeframe_variants = {}
        
        for tf in timeframes:
            tprint(f"🕐 Processing timeframe: {tf}")
            
            # Resample features to higher timeframe
            tf_features = self._resample_to_timeframe(features_df, tf)
            
            if tf_features.empty:
                tprint(f"⚠️ No data for timeframe {tf}, skipping")
                continue
                
            # Generate variants for this timeframe
            variants = self.generate_all_variants(
                tf_features, 
                price_data=self._resample_to_timeframe(price_data, tf) if price_data is not None else None,
                volume=self._resample_to_timeframe(volume, tf) if volume is not None else None
            )
            
            # Add timeframe prefix
            prefixed_variants = {}
            for variant_type, df in variants.items():
                for col in df.columns:
                    prefixed_variants[f"{tf}_{col}"] = df[col]
                    
            timeframe_variants[tf] = pd.DataFrame(prefixed_variants, index=tf_features.index)
            
        tprint(f"✅ Generated variants for {len(timeframe_variants)} timeframes")
        return timeframe_variants
    
    def _resample_to_timeframe(self, 
                             data: pd.DataFrame, 
                             target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe."""
        if data is None or data.empty:
            return pd.DataFrame()
            
        try:
            # Convert target timeframe to pandas frequency
            tf_map = {
                '15m': '15T',
                '30m': '30T', 
                '1h': '1H',
                '2h': '2H',
                '4h': '4H',
                '6h': '6H',
                '12h': '12H',
                '1d': '1D'
            }
            
            freq = tf_map.get(target_timeframe, target_timeframe)
            
            # Resample with appropriate aggregation
            if isinstance(data, pd.Series):
                resampled = data.resample(freq).last()  # Last value for series
            else:
                # For DataFrames, use appropriate aggregation per column type
                agg_dict = {}
                for col in data.columns:
                    if data[col].dtype in ['float64', 'float32']:
                        agg_dict[col] = 'mean'  # Average for continuous features
                    else:
                        agg_dict[col] = 'last'  # Last value for discrete features
                        
                resampled = data.resample(freq).agg(agg_dict)
                
            # Apply lag to prevent lookahead bias
            if self.config.ht_lag_periods > 0:
                resampled = resampled.shift(self.config.ht_lag_periods)
                
            return resampled
            
        except Exception as e:
            tprint(f"⚠️ Failed to resample to {target_timeframe}: {e}")
            return pd.DataFrame()
    
    def generate_all_variants(self, 
                            features_df: pd.DataFrame,
                            price_data: Optional[pd.DataFrame] = None,
                            volume: Optional[pd.Series] = None) -> Dict[str, pd.DataFrame]:
        """Generate all variant types for a single timeframe."""
        tprint(f"🔄 [VARIANT] Generating variants for {len(features_df.columns)} features")
        
        variants = {
            'raw': features_df.copy()
        }
        tprint(f"✅ [VARIANT] Generated raw variants: {len(features_df.columns)} features")
        
        # Generate volatility-normalized variants
        tprint("🔄 [VARIANT] Generating volatility-normalized variants")
        vol_norm_df = self.generate_vol_normalized_variants(features_df, price_data)
        if not vol_norm_df.empty:
            variants['vol_normalized'] = vol_norm_df
            tprint(f"✅ [VARIANT] Generated vol-normalized variants: {len(vol_norm_df.columns)} features")
        else:
            tprint("⚠️ [VARIANT] No volatility-normalized variants generated")
            
        # Generate VWAP variants
        tprint("🔄 [VARIANT] Generating VWAP variants")
        vwap_df = self.generate_vwap_variants(features_df, volume)
        if not vwap_df.empty:
            variants['vwap'] = vwap_df
            tprint(f"✅ [VARIANT] Generated VWAP variants: {len(vwap_df.columns)} features")
        else:
            tprint("⚠️ [VARIANT] No VWAP variants generated (no volume data)")
            
        # Generate combined variants
        if not vol_norm_df.empty and not vwap_df.empty:
            tprint("🔄 [VARIANT] Generating combined variants")
            combined_df = self.generate_combined_variants(features_df, vol_norm_df, vwap_df)
            if not combined_df.empty:
                variants['combined'] = combined_df
                tprint(f"✅ [VARIANT] Generated combined variants: {len(combined_df.columns)} features")
            else:
                tprint("⚠️ [VARIANT] No combined variants generated")
        else:
            tprint("⚠️ [VARIANT] Skipping combined variants (missing vol-normalized or VWAP data)")
                
        tprint(f"🎉 [VARIANT] Generated {len(variants)} variant types with total {sum(len(v.columns) for v in variants.values())} features")
        return variants
    
    def normalize_variants(self, 
                         variants_dict: Dict[str, pd.DataFrame],
                         method: str = 'zscore') -> Dict[str, pd.DataFrame]:
        """Normalize variants within each timeframe/variant type."""
        tprint(f"📊 Normalizing variants using {method} method...")
        
        normalized_variants = {}
        
        for variant_type, df in variants_dict.items():
            if df.empty:
                continue
                
            if method == 'zscore' and self.standard_scaler:
                normalized_df = self.standard_scaler.fit_transform(df)
            elif method == 'robust' and self.robust_scaler:
                normalized_df = self.robust_scaler.fit_transform(df)
            else:
                # Manual z-score normalization
                normalized_df = (df - df.mean()) / (df.std() + 1e-8)
                
            normalized_variants[variant_type] = normalized_df
            
        tprint(f"✅ Normalized {len(normalized_variants)} variant types")
        return normalized_variants
    
    def combine_all_variants(self, 
                           variants_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all variants into a single DataFrame."""
        tprint("📊 Combining all variants...")
        
        all_features = []
        variant_metadata = {}
        
        for variant_type, df in variants_dict.items():
            if not df.empty:
                all_features.append(df)
                
                # Track metadata
                for col in df.columns:
                    variant_metadata[col] = {
                        'variant_type': variant_type,
                        'original_feature': col.split('_vol_norm')[0].split('_vwap')[0].split('_vol_vwap')[0]
                    }
                    
        if not all_features:
            tprint("⚠️ No variants to combine")
            return pd.DataFrame()
            
        combined_df = pd.concat(all_features, axis=1)
        tprint(f"✅ Combined {len(combined_df.columns)} features from {len(variants_dict)} variant types")
        
        # Store metadata
        self._variant_cache['metadata'] = variant_metadata
        
        return combined_df
    
    def get_variant_metadata(self) -> Dict[str, Any]:
        """Get metadata about generated variants."""
        return self._variant_cache.get('metadata', {})

# Import tprint for logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
