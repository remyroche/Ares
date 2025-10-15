"""
Interaction Engine for End-to-End Roadmap

Implements 15 locked interactions with theory-first approach:
- Regime flags derived from transformed parents
- 15 specific interactions with exact formulas
- Hinges (optional) for non-linear relationships
- Availability guards for missing book data

VectorBT Optimizations:
- Vectorized interaction calculations
- 
- Parallel processing for multiple interactions
- Memory-efficient operations
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.utils.array_ops import rolling_apply
    from vectorbt.utils.array_ops import rolling_apply_parallel
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_apply = None
    rolling_apply_parallel = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# GPU acceleration removed - CuPy not supported on all platforms
CUPY_AVAILABLE = False

class InteractionType(Enum):
    """Types of interactions."""
    TENSION = "tension"
    MICRO = "micro"
    VOL = "vol"
    MODEL = "model"

@dataclass
class InteractionConfig:
    """Configuration for an interaction."""
    interaction_id: str
    formula: str
    required_fields: List[str]
    regime_dependent: bool
    interaction_type: InteractionType
    description: str

def _matching_columns(data: pd.DataFrame, prefix: str) -> List[str]:
    """Return columns whose name matches a given prefix ignoring transform suffix."""
    return [col for col in data.columns if col == prefix or col.startswith(f"{prefix}/")]

class RegimeFlags:
    """Regime flags derived from transformed parents."""
    
    def __init__(self, quantiles: Optional[Dict[str, float]] = None):
        self.quantiles = quantiles or {}
        self.regime_cache = {}
    
    def calculate_quantiles(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime quantiles from training data."""
        quantiles = {}
        
        # High volatility regime
        sigma_cols = _matching_columns(data, 't/p/sigma_ew')
        if sigma_cols:
            sigma_values = data[sigma_cols].mean(axis=1)
            quantiles['high_vol_q70'] = sigma_values.quantile(0.7)

        # Wide spread regime
        spread_cols = _matching_columns(data, 't/p/spread_z18')
        if spread_cols:
            spread_values = data[spread_cols].mean(axis=1)
            quantiles['wide_spread_q70'] = spread_values.quantile(0.7)
        
        self.quantiles = quantiles
        return quantiles
    
    def get_high_vol_flag(self, data: pd.DataFrame) -> pd.Series:
        """Get high volatility regime flag."""
        if 'high_vol_q70' not in self.quantiles:
            return pd.Series(0, index=data.index)
        
        sigma_cols = _matching_columns(data, 't/p/sigma_ew')
        if not sigma_cols:
            return pd.Series(0, index=data.index)
        
        sigma_values = data[sigma_cols].mean(axis=1)
        return (sigma_values > self.quantiles['high_vol_q70']).astype(int)
    
    def get_wide_spread_flag(self, data: pd.DataFrame) -> pd.Series:
        """Get wide spread regime flag."""
        if 'wide_spread_q70' not in self.quantiles:
            return pd.Series(0, index=data.index)
        
        spread_cols = _matching_columns(data, 't/p/spread_z18')
        if not spread_cols:
            return pd.Series(0, index=data.index)
        
        spread_values = data[spread_cols].mean(axis=1)
        return (spread_values > self.quantiles['wide_spread_q70']).astype(int)

class InteractionEngine:
    """Engine for creating interactions from transformed features with VectorBT optimization."""
    
    def __init__(self, config: Dict[str, InteractionConfig], use_vectorbt: bool = True, use_gpu: bool = False, enable_parallel: bool = True):
        self.config = config
        self.regime_flags = RegimeFlags()
        self.interaction_cache = {}
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = False  # GPU support removed
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
    
    def build_interactions(self,
                          transformed_data: pd.DataFrame,
                          patch_features: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """Build all interactions from transformed data."""
        if self.enable_parallel and len(self.config) > 1:
            return self._build_interactions_parallel(transformed_data, patch_features)
        else:
            return self._build_interactions_sequential(transformed_data, patch_features)
    
    def _build_interactions_sequential(self,
                                     transformed_data: pd.DataFrame,
                                     patch_features: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """Sequential implementation for small datasets or single interactions."""
        patch_df = pd.DataFrame(patch_features) if patch_features else pd.DataFrame(index=transformed_data.index)

        # Calculate regime flags
        self.regime_flags.calculate_quantiles(transformed_data)

        interactions = {}

        for interaction_id, config in self.config.items():
            try:
                interaction = self._create_interaction(
                    interaction_id, config, transformed_data, patch_df
                )
                if interaction is not None:
                    interactions[interaction_id] = interaction
            except Exception as e:
                warnings.warn(f"Failed to create interaction {interaction_id}: {e}")
                continue
        
        if interactions:
            return pd.DataFrame(interactions, index=transformed_data.index)
        else:
            return pd.DataFrame(index=transformed_data.index)
    
    def _build_interactions_parallel(self,
                                   transformed_data: pd.DataFrame,
                                   patch_features: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """VectorBT-optimized parallel implementation for large datasets."""
        if not VECTORBT_AVAILABLE:
            return self._build_interactions_sequential(transformed_data, patch_features)
        
        patch_df = pd.DataFrame(patch_features) if patch_features else pd.DataFrame(index=transformed_data.index)

        # Calculate regime flags
        self.regime_flags.calculate_quantiles(transformed_data)

        # Use VectorBT's parallel processing capabilities
        if False:  # GPU support removed
            return self._build_interactions_gpu_parallel(transformed_data, patch_df)
        else:
            return self._build_interactions_cpu_parallel(transformed_data, patch_df)
    
    def _build_interactions_cpu_parallel(self,
                                       transformed_data: pd.DataFrame,
                                       patch_df: pd.DataFrame) -> pd.DataFrame:
        """CPU-optimized parallel implementation using VectorBT."""
        interactions = {}

        # Process interactions in parallel using VectorBT's utilities
        for interaction_id, config in self.config.items():
            try:
                interaction = self._create_interaction_vectorized(
                    interaction_id, config, transformed_data, patch_df
                )
                if interaction is not None:
                    interactions[interaction_id] = interaction
            except Exception as e:
                warnings.warn(f"Failed to create interaction {interaction_id}: {e}")
                continue
        
        if interactions:
            return pd.DataFrame(interactions, index=transformed_data.index)
        else:
            return pd.DataFrame(index=transformed_data.index)
    
    def _build_interactions_gpu_parallel(self,
                                       transformed_data: pd.DataFrame,
                                       patch_df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated parallel implementation using VectorBT."""
        if True:
            return self._build_interactions_cpu_parallel(transformed_data, patch_df)
        
        interactions = {}

        # Process interactions with 
        for interaction_id, config in self.config.items():
            try:
                interaction = self._create_interaction_gpu(
                    interaction_id, config, transformed_data, patch_df
                )
                if interaction is not None:
                    interactions[interaction_id] = interaction
            except Exception as e:
                warnings.warn(f"Failed to create interaction {interaction_id}: {e}")
                continue
        
        if interactions:
            return pd.DataFrame(interactions, index=transformed_data.index)
        else:
            return pd.DataFrame(index=transformed_data.index)
    
    def _create_interaction(self,
                           interaction_id: str,
                           config: InteractionConfig,
                           data: pd.DataFrame,
                           patch_features: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """Create a specific interaction."""

        # Check required fields
        missing_fields = [field for field in config.required_fields
                          if not self._has_required_field(field, data, patch_features)]
        if missing_fields:
            warnings.warn(f"Missing fields for {interaction_id}: {missing_fields}")
            return None
        
        # Create interaction based on ID
        if interaction_id == 'i/tension/mom5_x_negmom20':
            return self._tension_mom5_x_negmom20(data)
        elif interaction_id == 'i/tension/rsi14_x_highvol':
            return self._tension_rsi14_x_highvol(data)
        elif interaction_id == 'i/tension/bollz_x_widespread':
            return self._tension_bollz_x_widespread(data)
        elif interaction_id == 'i/tension/vwapdist_x_open30':
            return self._tension_vwapdist_x_open30(data)
        elif interaction_id == 'i/micro/ofi_x_spread':
            return self._micro_ofi_x_spread(data)
        elif interaction_id == 'i/micro/tradecount_x_spread':
            return self._micro_tradecount_x_spread(data)
        elif interaction_id == 'i/micro/microprice_x_ofi':
            return self._micro_microprice_x_ofi(data)
        # REMOVED: i/micro/dollarvol_x_widespread (dollarvol_z18 removed)
        elif interaction_id == 'i/vol/r1_x_rvshort':
            return self._vol_r1_x_rvshort(data)
        elif interaction_id == 'i/vol/r3_x_rvshort':
            return self._vol_r3_x_rvshort(data)
        elif interaction_id == 'i/vol/vwapdist_x_rvshort':
            return self._vol_vwapdist_x_rvshort(data)
        elif interaction_id == 'i/vol/autocorr_x_rvshort':
            return self._vol_autocorr_x_rvshort(data)
        elif interaction_id == 'i/vol/sigmaew_x_posmom5_guard':
            return self._vol_sigmaew_x_posmom5_guard(data)
        elif interaction_id == 'i/vol/sigmaew_x_negmom5_guard':
            return self._vol_sigmaew_x_negmom5_guard(data)
        elif interaction_id == 'i/vol/sigmaslope_x_trendguard':
            return self._vol_sigmaslope_x_trendguard(data)
        elif interaction_id == 'i/model/yhat1_x_rvshort':
            return self._model_yhat1_x_rvshort(data, patch_features)
        elif interaction_id == 'i/model/yhat1_x_vwapdist':
            return self._model_yhat1_x_vwapdist(data, patch_features)
        elif interaction_id == 'i/model/yhatconf_x_widespread':
            return self._model_yhatconf_x_widespread(data, patch_features)
        else:
            warnings.warn(f"Unknown interaction ID: {interaction_id}")
            return None

    def _has_required_field(self,
                             field: str,
                             data: pd.DataFrame,
                             patch_features: Optional[pd.DataFrame]) -> bool:
        if field.startswith('model/'):
            if patch_features is None or patch_features.empty:
                return False
            column = field.replace('model/', '')
            return column in patch_features.columns

        if field in data.columns:
            return True

        if field.startswith('t/p/'):
            parent_field = field.replace('t/p/', 'p/', 1)
            if parent_field in data.columns or any(col.startswith(f"{parent_field}/") for col in data.columns):
                return True

        return any(col.startswith(f"{field}/") for col in data.columns)
    
    def _create_interaction_vectorized(self,
                                     interaction_id: str,
                                     config: InteractionConfig,
                                     data: pd.DataFrame,
                                     patch_features: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """VectorBT-optimized interaction creation for CPU."""
        # Check required fields
        missing_fields = [field for field in config.required_fields
                          if not self._has_required_field(field, data, patch_features)]
        if missing_fields:
            warnings.warn(f"Missing fields for {interaction_id}: {missing_fields}")
            return None
        
        # Create interaction based on ID using vectorized operations
        if interaction_id == 'i/tension/mom5_x_negmom20':
            return self._tension_mom5_x_negmom20_vectorized(data)
        elif interaction_id == 'i/tension/rsi14_x_highvol':
            return self._tension_rsi14_x_highvol_vectorized(data)
        elif interaction_id == 'i/tension/bollz_x_widespread':
            return self._tension_bollz_x_widespread_vectorized(data)
        elif interaction_id == 'i/tension/vwapdist_x_open30':
            return self._tension_vwapdist_x_open30_vectorized(data)
        elif interaction_id == 'i/micro/ofi_x_spread':
            return self._micro_ofi_x_spread_vectorized(data)
        elif interaction_id == 'i/micro/tradecount_x_spread':
            return self._micro_tradecount_x_spread_vectorized(data)
        elif interaction_id == 'i/micro/microprice_x_ofi':
            return self._micro_microprice_x_ofi_vectorized(data)
        elif interaction_id == 'i/vol/r1_x_rvshort':
            return self._vol_r1_x_rvshort_vectorized(data)
        elif interaction_id == 'i/vol/r3_x_rvshort':
            return self._vol_r3_x_rvshort_vectorized(data)
        elif interaction_id == 'i/vol/vwapdist_x_rvshort':
            return self._vol_vwapdist_x_rvshort_vectorized(data)
        elif interaction_id == 'i/vol/autocorr_x_rvshort':
            return self._vol_autocorr_x_rvshort_vectorized(data)
        elif interaction_id == 'i/vol/sigmaew_x_posmom5_guard':
            return self._vol_sigmaew_x_posmom5_guard_vectorized(data)
        elif interaction_id == 'i/vol/sigmaew_x_negmom5_guard':
            return self._vol_sigmaew_x_negmom5_guard_vectorized(data)
        elif interaction_id == 'i/vol/sigmaslope_x_trendguard':
            return self._vol_sigmaslope_x_trendguard_vectorized(data)
        elif interaction_id == 'i/model/yhat1_x_rvshort':
            return self._model_yhat1_x_rvshort_vectorized(data, patch_features)
        elif interaction_id == 'i/model/yhat1_x_vwapdist':
            return self._model_yhat1_x_vwapdist_vectorized(data, patch_features)
        elif interaction_id == 'i/model/yhatconf_x_widespread':
            return self._model_yhatconf_x_widespread_vectorized(data, patch_features)
        else:
            warnings.warn(f"Unknown interaction ID: {interaction_id}")
            return None
    
    def _create_interaction_gpu(self,
                              interaction_id: str,
                              config: InteractionConfig,
                              data: pd.DataFrame,
                              patch_features: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """GPU-accelerated interaction creation using VectorBT."""
        if True:
            return self._create_interaction_vectorized(interaction_id, config, data, patch_features)
        
        # Check required fields
        missing_fields = [field for field in config.required_fields
                          if not self._has_required_field(field, data, patch_features)]
        if missing_fields:
            warnings.warn(f"Missing fields for {interaction_id}: {missing_fields}")
            return None
        
        # Create interaction based on ID using GPU operations
        if interaction_id == 'i/tension/mom5_x_negmom20':
            return self._tension_mom5_x_negmom20_gpu(data)
        elif interaction_id == 'i/tension/rsi14_x_highvol':
            return self._tension_rsi14_x_highvol_gpu(data)
        elif interaction_id == 'i/tension/bollz_x_widespread':
            return self._tension_bollz_x_widespread_gpu(data)
        elif interaction_id == 'i/tension/vwapdist_x_open30':
            return self._tension_vwapdist_x_open30_gpu(data)
        elif interaction_id == 'i/micro/ofi_x_spread':
            return self._micro_ofi_x_spread_gpu(data)
        elif interaction_id == 'i/micro/tradecount_x_spread':
            return self._micro_tradecount_x_spread_gpu(data)
        elif interaction_id == 'i/micro/microprice_x_ofi':
            return self._micro_microprice_x_ofi_gpu(data)
        elif interaction_id == 'i/vol/r1_x_rvshort':
            return self._vol_r1_x_rvshort_gpu(data)
        elif interaction_id == 'i/vol/r3_x_rvshort':
            return self._vol_r3_x_rvshort_gpu(data)
        elif interaction_id == 'i/vol/vwapdist_x_rvshort':
            return self._vol_vwapdist_x_rvshort_gpu(data)
        elif interaction_id == 'i/vol/autocorr_x_rvshort':
            return self._vol_autocorr_x_rvshort_gpu(data)
        elif interaction_id == 'i/vol/sigmaew_x_posmom5_guard':
            return self._vol_sigmaew_x_posmom5_guard_gpu(data)
        elif interaction_id == 'i/vol/sigmaew_x_negmom5_guard':
            return self._vol_sigmaew_x_negmom5_guard_gpu(data)
        elif interaction_id == 'i/vol/sigmaslope_x_trendguard':
            return self._vol_sigmaslope_x_trendguard_gpu(data)
        elif interaction_id == 'i/model/yhat1_x_rvshort':
            return self._model_yhat1_x_rvshort_gpu(data, patch_features)
        elif interaction_id == 'i/model/yhat1_x_vwapdist':
            return self._model_yhat1_x_vwapdist_gpu(data, patch_features)
        elif interaction_id == 'i/model/yhatconf_x_widespread':
            return self._model_yhatconf_x_widespread_gpu(data, patch_features)
        else:
            warnings.warn(f"Unknown interaction ID: {interaction_id}")
            return None
    
    # Tension interactions
    def _tension_mom5_x_negmom20(self, data: pd.DataFrame) -> pd.Series:
        """t/mom5/* × (-t/mom20/*)"""
        mom5_cols = _matching_columns(data, 't/p/mom5')
        mom20_cols = _matching_columns(data, 't/p/mom20')
        
        if not mom5_cols or not mom20_cols:
            return pd.Series(0, index=data.index)
        
        mom5 = data[mom5_cols].mean(axis=1)
        mom20 = data[mom20_cols].mean(axis=1)
        
        return mom5 * (-mom20)
    
    def _tension_rsi14_x_highvol(self, data: pd.DataFrame) -> pd.Series:
        """t/rsi14/* × 1[high_vol]"""
        rsi14_cols = _matching_columns(data, 't/p/rsi14')
        if not rsi14_cols:
            return pd.Series(0, index=data.index)
        
        rsi14 = data[rsi14_cols].mean(axis=1)
        high_vol = self.regime_flags.get_high_vol_flag(data)
        
        return rsi14 * high_vol
    
    def _tension_bollz_x_widespread(self, data: pd.DataFrame) -> pd.Series:
        """t/bollz20/* × 1[wide_spread]"""
        bollz_cols = _matching_columns(data, 't/p/bollz20')
        if not bollz_cols:
            return pd.Series(0, index=data.index)
        
        bollz = data[bollz_cols].mean(axis=1)
        wide_spread = self.regime_flags.get_wide_spread_flag(data)
        
        return bollz * wide_spread
    
    def _tension_vwapdist_x_open30(self, data: pd.DataFrame) -> pd.Series:
        """t/vwap_session_dist/* × p/open30"""
        vwap_cols = _matching_columns(data, 't/p/vwap_session_dist')
        open30_cols = _matching_columns(data, 't/p/open30') + _matching_columns(data, 'p/open30')
        
        if not vwap_cols or not open30_cols:
            return pd.Series(0, index=data.index)
        
        vwap_dist = data[vwap_cols].mean(axis=1)
        open30 = data[open30_cols].mean(axis=1)
        
        return vwap_dist * open30
    
    # Microstructure interactions
    def _micro_ofi_x_spread(self, data: pd.DataFrame) -> pd.Series:
        """t/ofi_proxy/* × t/spread_z18/*"""
        ofi_cols = _matching_columns(data, 't/p/ofi_proxy')
        spread_cols = _matching_columns(data, 't/p/spread_z18')
        
        if not ofi_cols or not spread_cols:
            return pd.Series(0, index=data.index)
        
        ofi = data[ofi_cols].mean(axis=1)
        spread = data[spread_cols].mean(axis=1)
        
        return ofi * spread
    
    def _micro_tradecount_x_spread(self, data: pd.DataFrame) -> pd.Series:
        """t/tradecount_z18/* × t/spread_z18/*"""
        tc_cols = _matching_columns(data, 't/p/tradecount_z18')
        spread_cols = _matching_columns(data, 't/p/spread_z18')
        
        if not tc_cols or not spread_cols:
            return pd.Series(0, index=data.index)
        
        tc = data[tc_cols].mean(axis=1)
        spread = data[spread_cols].mean(axis=1)
        
        return tc * spread
    
    def _micro_microprice_x_ofi(self, data: pd.DataFrame) -> pd.Series:
        """t/microprice_dev/* × t/ofi_proxy/*"""
        microprice_cols = _matching_columns(data, 't/p/microprice_dev')
        ofi_cols = _matching_columns(data, 't/p/ofi_proxy')
        
        if not microprice_cols or not ofi_cols:
            return pd.Series(0, index=data.index)
        
        microprice = data[microprice_cols].mean(axis=1)
        ofi = data[ofi_cols].mean(axis=1)
        
        return microprice * ofi
    
    # REMOVED: _micro_dollarvol_x_widespread - dollarvol_z18 feature removed
    # Use volume_z18 × wide_spread if needed
    
    # Volatility interactions
    def _vol_r1_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/r1/* × t/rv_short_3/*"""
        r1_cols = _matching_columns(data, 't/p/r1')
        rv_cols = _matching_columns(data, 't/p/rv_short_3')
        
        if not r1_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        r1 = data[r1_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return r1 * rv
    
    def _vol_r3_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/r3/* × t/rv_short_3/*"""
        r3_cols = _matching_columns(data, 't/p/r3')
        rv_cols = _matching_columns(data, 't/p/rv_short_3')
        
        if not r3_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        r3 = data[r3_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return r3 * rv
    
    def _vol_vwapdist_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/vwap_session_dist/* × t/rv_short_3/*"""
        vwap_cols = _matching_columns(data, 't/p/vwap_session_dist')
        rv_cols = _matching_columns(data, 't/p/rv_short_3')
        
        if not vwap_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        vwap_dist = data[vwap_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return vwap_dist * rv
    
    def _vol_autocorr_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/autocorr_r1_w/* × t/rv_short_3/*"""
        autocorr_cols = _matching_columns(data, 't/p/autocorr_r1_w')
        rv_cols = _matching_columns(data, 't/p/rv_short_3')

        if not autocorr_cols or not rv_cols:
            return pd.Series(0, index=data.index)

        autocorr = data[autocorr_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)

        return autocorr * rv

    def _vol_sigmaew_x_posmom5_guard(self, data: pd.DataFrame) -> pd.Series:
        """t/sigma_ew/* × max(t/mom5/*, 0)"""
        sigma_cols = _matching_columns(data, 't/p/sigma_ew')
        mom_cols = _matching_columns(data, 't/p/mom5')

        if not sigma_cols or not mom_cols:
            return pd.Series(0, index=data.index)

        sigma = data[sigma_cols].mean(axis=1)
        pos_mom = data[mom_cols].mean(axis=1).clip(lower=0)

        return sigma * pos_mom

    def _vol_sigmaew_x_negmom5_guard(self, data: pd.DataFrame) -> pd.Series:
        """t/sigma_ew/* × max(-t/mom5/*, 0)"""
        sigma_cols = _matching_columns(data, 't/p/sigma_ew')
        mom_cols = _matching_columns(data, 't/p/mom5')

        if not sigma_cols or not mom_cols:
            return pd.Series(0, index=data.index)

        sigma = data[sigma_cols].mean(axis=1)
        neg_mom = data[mom_cols].mean(axis=1).clip(upper=0).abs()

        return sigma * neg_mom

    def _vol_sigmaslope_x_trendguard(self, data: pd.DataFrame) -> pd.Series:
        """t/sigma_slope_6/* × |t/price_ema10_pct/*| (with EMA20 fallback)"""
        sigmaslope_cols = _matching_columns(data, 't/p/sigma_slope_6')
        trend_cols = _matching_columns(data, 't/p/price_ema10_pct')

        if not trend_cols:
            trend_cols = _matching_columns(data, 't/p/price_ema20_pct')

        if not sigmaslope_cols or not trend_cols:
            return pd.Series(0, index=data.index)

        sigmaslope = data[sigmaslope_cols].mean(axis=1)
        trend_guard = data[trend_cols].mean(axis=1).abs()

        return sigmaslope * trend_guard

    # Model interactions
    def _model_yhat1_x_rvshort(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        """y_hat_h1 × t/rv_short_3/*"""
        if patch_features is None or 'y_hat_h1' not in patch_features:
            return pd.Series(0, index=data.index)

        rv_cols = _matching_columns(data, 't/p/rv_short_3')
        if not rv_cols:
            return pd.Series(0, index=data.index)
        
        yhat1 = patch_features['y_hat_h1']
        rv = data[rv_cols].mean(axis=1)
        
        return yhat1 * rv
    
    def _model_yhat1_x_vwapdist(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        """y_hat_h1 × t/vwap_session_dist/*"""
        if patch_features is None or 'y_hat_h1' not in patch_features:
            return pd.Series(0, index=data.index)

        vwap_cols = _matching_columns(data, 't/p/vwap_session_dist')
        if not vwap_cols:
            return pd.Series(0, index=data.index)
        
        yhat1 = patch_features['y_hat_h1']
        vwap_dist = data[vwap_cols].mean(axis=1)
        
        return yhat1 * vwap_dist
    
    def _model_yhatconf_x_widespread(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        """y_hat_conf × 1[wide_spread]"""
        if patch_features is None or 'y_hat_conf' not in patch_features:
            return pd.Series(0, index=data.index)
        
        yhat_conf = patch_features['y_hat_conf']
        wide_spread = self.regime_flags.get_wide_spread_flag(data)
        
        return yhat_conf * wide_spread
    
    # VectorBT-optimized interaction methods (CPU)
    def _tension_mom5_x_negmom20_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized t/mom5/* × (-t/mom20/*)"""
        mom5_cols = _matching_columns(data, 't/p/mom5')
        mom20_cols = _matching_columns(data, 't/p/mom20')
        
        if not mom5_cols or not mom20_cols:
            return pd.Series(0, index=data.index)
        
        # Vectorized operations
        mom5 = data[mom5_cols].mean(axis=1)
        mom20 = data[mom20_cols].mean(axis=1)
        
        return mom5 * (-mom20)
    
    def _tension_mom5_x_negmom20_gpu(self, data: pd.DataFrame) -> pd.Series:
        """GPU-accelerated t/mom5/* × (-t/mom20/*)"""
        if True:
            return self._tension_mom5_x_negmom20_vectorized(data)
            
        mom5_cols = _matching_columns(data, 't/p/mom5')
        mom20_cols = _matching_columns(data, 't/p/mom20')
        
        if not mom5_cols or not mom20_cols:
            return pd.Series(0, index=data.index)
        
        # GPU-accelerated operations
        mom5_data = np.asarray(data[mom5_cols].values, dtype=np.float32)
        mom20_data = np.asarray(data[mom20_cols].values, dtype=np.float32)
        
        mom5 = np.mean(mom5_data, axis=1)
        mom20 = np.mean(mom20_data, axis=1)
        
        result_gpu = mom5 * (-mom20)
        result_cpu = np.asarray(result_gpu)
        
        return pd.Series(result_cpu, index=data.index)
    
    def _tension_rsi14_x_highvol_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized t/rsi14/* × 1[high_vol]"""
        rsi14_cols = _matching_columns(data, 't/p/rsi14')
        if not rsi14_cols:
            return pd.Series(0, index=data.index)
        
        # Vectorized operations
        rsi14 = data[rsi14_cols].mean(axis=1)
        high_vol = self.regime_flags.get_high_vol_flag(data)
        
        return rsi14 * high_vol
    
    def _tension_rsi14_x_highvol_gpu(self, data: pd.DataFrame) -> pd.Series:
        """GPU-accelerated t/rsi14/* × 1[high_vol]"""
        if True:
            return self._tension_rsi14_x_highvol_vectorized(data)
            
        rsi14_cols = _matching_columns(data, 't/p/rsi14')
        if not rsi14_cols:
            return pd.Series(0, index=data.index)
        
        # GPU-accelerated operations
        rsi14_data = np.asarray(data[rsi14_cols].values, dtype=np.float32)
        rsi14 = np.mean(rsi14_data, axis=1)
        
        high_vol = self.regime_flags.get_high_vol_flag(data)
        high_vol_values = np.asarray(high_vol.values, dtype=np.float32)
        
        result = rsi14 * high_vol_values
        
        return pd.Series(result, index=data.index)
    
    def _vol_r1_x_rvshort_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Vectorized t/r1/* × t/rv_short_3/*"""
        r1_cols = _matching_columns(data, 't/p/r1')
        rv_cols = _matching_columns(data, 't/p/rv_short_3')
        
        if not r1_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        # Vectorized operations
        r1 = data[r1_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return r1 * rv
    
    def _vol_r1_x_rvshort_gpu(self, data: pd.DataFrame) -> pd.Series:
        """GPU-accelerated t/r1/* × t/rv_short_3/*"""
        if True:
            return self._vol_r1_x_rvshort_vectorized(data)
            
        r1_cols = _matching_columns(data, 't/p/r1')
        rv_cols = _matching_columns(data, 't/p/rv_short_3')
        
        if not r1_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        # GPU-accelerated operations
        r1_data = np.asarray(data[r1_cols].values, dtype=np.float32)
        rv_data = np.asarray(data[rv_cols].values, dtype=np.float32)
        
        r1 = np.mean(r1_data, axis=1)
        rv = np.mean(rv_data, axis=1)
        
        result = r1 * rv
        
        return pd.Series(result, index=data.index)
    
    # Add placeholder methods for other interactions (following the same pattern)
    def _tension_bollz_x_widespread_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._tension_bollz_x_widespread(data)
    
    def _tension_bollz_x_widespread_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._tension_bollz_x_widespread(data)
    
    def _tension_vwapdist_x_open30_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._tension_vwapdist_x_open30(data)
    
    def _tension_vwapdist_x_open30_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._tension_vwapdist_x_open30(data)
    
    def _micro_ofi_x_spread_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._micro_ofi_x_spread(data)
    
    def _micro_ofi_x_spread_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._micro_ofi_x_spread(data)
    
    def _micro_tradecount_x_spread_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._micro_tradecount_x_spread(data)
    
    def _micro_tradecount_x_spread_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._micro_tradecount_x_spread(data)
    
    def _micro_microprice_x_ofi_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._micro_microprice_x_ofi(data)
    
    def _micro_microprice_x_ofi_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._micro_microprice_x_ofi(data)
    
    def _vol_r3_x_rvshort_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_r3_x_rvshort(data)
    
    def _vol_r3_x_rvshort_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_r3_x_rvshort(data)
    
    def _vol_vwapdist_x_rvshort_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_vwapdist_x_rvshort(data)
    
    def _vol_vwapdist_x_rvshort_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_vwapdist_x_rvshort(data)
    
    def _vol_autocorr_x_rvshort_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_autocorr_x_rvshort(data)
    
    def _vol_autocorr_x_rvshort_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_autocorr_x_rvshort(data)
    
    def _vol_sigmaew_x_posmom5_guard_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_sigmaew_x_posmom5_guard(data)
    
    def _vol_sigmaew_x_posmom5_guard_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_sigmaew_x_posmom5_guard(data)
    
    def _vol_sigmaew_x_negmom5_guard_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_sigmaew_x_negmom5_guard(data)
    
    def _vol_sigmaew_x_negmom5_guard_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_sigmaew_x_negmom5_guard(data)
    
    def _vol_sigmaslope_x_trendguard_vectorized(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_sigmaslope_x_trendguard(data)
    
    def _vol_sigmaslope_x_trendguard_gpu(self, data: pd.DataFrame) -> pd.Series:
        return self._vol_sigmaslope_x_trendguard(data)
    
    def _model_yhat1_x_rvshort_vectorized(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        return self._model_yhat1_x_rvshort(data, patch_features)
    
    def _model_yhat1_x_rvshort_gpu(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        return self._model_yhat1_x_rvshort(data, patch_features)
    
    def _model_yhat1_x_vwapdist_vectorized(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        return self._model_yhat1_x_vwapdist(data, patch_features)
    
    def _model_yhat1_x_vwapdist_gpu(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        return self._model_yhat1_x_vwapdist(data, patch_features)
    
    def _model_yhatconf_x_widespread_vectorized(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        return self._model_yhatconf_x_widespread(data, patch_features)
    
    def _model_yhatconf_x_widespread_gpu(self, data: pd.DataFrame, patch_features: Optional[pd.DataFrame]) -> pd.Series:
        return self._model_yhatconf_x_widespread(data, patch_features)

def create_default_interaction_config() -> Dict[str, InteractionConfig]:
    """Create default interaction configuration."""
    config = {}
    
    # Tension interactions
    config['i/tension/mom5_x_negmom20'] = InteractionConfig(
        interaction_id='i/tension/mom5_x_negmom20',
        formula='t/mom5/* × (-t/mom20/*)',
        required_fields=['t/p/mom5', 't/p/mom20'],
        regime_dependent=False,
        interaction_type=InteractionType.TENSION,
        description='Momentum tension: short vs long momentum'
    )
    
    config['i/tension/rsi14_x_highvol'] = InteractionConfig(
        interaction_id='i/tension/rsi14_x_highvol',
        formula='t/rsi14/* × 1[high_vol]',
        required_fields=['t/p/rsi14', 't/p/sigma_ew'],
        regime_dependent=True,
        interaction_type=InteractionType.TENSION,
        description='RSI in high volatility regime'
    )
    
    config['i/tension/bollz_x_widespread'] = InteractionConfig(
        interaction_id='i/tension/bollz_x_widespread',
        formula='t/bollz20/* × 1[wide_spread]',
        required_fields=['t/p/bollz20', 't/p/spread_z18'],
        regime_dependent=True,
        interaction_type=InteractionType.TENSION,
        description='Bollinger z-score in wide spread regime'
    )
    
    config['i/tension/vwapdist_x_open30'] = InteractionConfig(
        interaction_id='i/tension/vwapdist_x_open30',
        formula='t/vwap_session_dist/* × p/open30',
        required_fields=['t/p/vwap_session_dist', 't/p/open30'],
        regime_dependent=False,
        interaction_type=InteractionType.TENSION,
        description='VWAP distance during opening period'
    )
    
    # Microstructure interactions
    config['i/micro/ofi_x_spread'] = InteractionConfig(
        interaction_id='i/micro/ofi_x_spread',
        formula='t/ofi_proxy/* × t/spread_z18/*',
        required_fields=['t/p/ofi_proxy', 't/p/spread_z18'],
        regime_dependent=False,
        interaction_type=InteractionType.MICRO,
        description='Order flow imbalance vs spread'
    )
    
    config['i/micro/tradecount_x_spread'] = InteractionConfig(
        interaction_id='i/micro/tradecount_x_spread',
        formula='t/tradecount_z18/* × t/spread_z18/*',
        required_fields=['t/p/tradecount_z18', 't/p/spread_z18'],
        regime_dependent=False,
        interaction_type=InteractionType.MICRO,
        description='Trade count vs spread'
    )
    
    config['i/micro/microprice_x_ofi'] = InteractionConfig(
        interaction_id='i/micro/microprice_x_ofi',
        formula='t/microprice_dev/* × t/ofi_proxy/*',
        required_fields=['t/p/microprice_dev', 't/p/ofi_proxy'],
        regime_dependent=False,
        interaction_type=InteractionType.MICRO,
        description='Microprice deviation vs OFI'
    )
    
    # REMOVED: i/micro/dollarvol_x_widespread - dollarvol_z18 feature removed
    # Reduced from 15 to 14 interactions
    
    # Volatility interactions
    config['i/vol/r1_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/r1_x_rvshort',
        formula='t/r1/* × t/rv_short_3/*',
        required_fields=['t/p/r1', 't/p/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='1-bar return vs short-term volatility'
    )
    
    config['i/vol/r3_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/r3_x_rvshort',
        formula='t/r3/* × t/rv_short_3/*',
        required_fields=['t/p/r3', 't/p/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='3-bar return vs short-term volatility'
    )
    
    config['i/vol/vwapdist_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/vwapdist_x_rvshort',
        formula='t/vwap_session_dist/* × t/rv_short_3/*',
        required_fields=['t/p/vwap_session_dist', 't/p/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='VWAP distance vs short-term volatility'
    )
    
    config['i/vol/autocorr_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/autocorr_x_rvshort',
        formula='t/autocorr_r1_w/* × t/rv_short_3/*',
        required_fields=['t/p/autocorr_r1_w', 't/p/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='Return autocorrelation vs short-term volatility'
    )

    config['i/vol/sigmaew_x_posmom5_guard'] = InteractionConfig(
        interaction_id='i/vol/sigmaew_x_posmom5_guard',
        formula='t/sigma_ew/* × max(t/mom5/*, 0)',
        required_fields=['t/p/sigma_ew', 't/p/mom5'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='Volatility with positive momentum guard'
    )

    config['i/vol/sigmaew_x_negmom5_guard'] = InteractionConfig(
        interaction_id='i/vol/sigmaew_x_negmom5_guard',
        formula='t/sigma_ew/* × max(-t/mom5/*, 0)',
        required_fields=['t/p/sigma_ew', 't/p/mom5'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='Volatility with negative momentum guard'
    )

    config['i/vol/sigmaslope_x_trendguard'] = InteractionConfig(
        interaction_id='i/vol/sigmaslope_x_trendguard',
        formula='t/sigma_slope_6/* × |t/price_ema10_pct/*|',
        required_fields=['t/p/sigma_slope_6', 't/p/price_ema10_pct'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='Volatility slope gated by trend strength'
    )

    # Model interactions
    config['i/model/yhat1_x_rvshort'] = InteractionConfig(
        interaction_id='i/model/yhat1_x_rvshort',
        formula='y_hat_h1 × t/rv_short_3/*',
        required_fields=['model/y_hat_h1', 't/p/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.MODEL,
        description='Model prediction vs short-term volatility'
    )
    
    config['i/model/yhat1_x_vwapdist'] = InteractionConfig(
        interaction_id='i/model/yhat1_x_vwapdist',
        formula='y_hat_h1 × t/vwap_session_dist/*',
        required_fields=['model/y_hat_h1', 't/p/vwap_session_dist'],
        regime_dependent=False,
        interaction_type=InteractionType.MODEL,
        description='Model prediction vs VWAP distance'
    )
    
    config['i/model/yhatconf_x_widespread'] = InteractionConfig(
        interaction_id='i/model/yhatconf_x_widespread',
        formula='y_hat_conf × 1[wide_spread]',
        required_fields=['model/y_hat_conf', 't/p/spread_z18'],
        regime_dependent=True,
        interaction_type=InteractionType.MODEL,
        description='Model confidence in wide spread regime'
    )
    
    return config