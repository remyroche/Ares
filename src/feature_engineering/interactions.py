"""
Interaction Engine for End-to-End Roadmap

Implements 15 locked interactions with theory-first approach:
- Regime flags derived from transformed parents
- 15 specific interactions with exact formulas
- Hinges (optional) for non-linear relationships
- Availability guards for missing book data
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings


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


class RegimeFlags:
    """Regime flags derived from transformed parents."""
    
    def __init__(self, quantiles: Optional[Dict[str, float]] = None):
        self.quantiles = quantiles or {}
        self.regime_cache = {}
    
    def calculate_quantiles(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime quantiles from training data."""
        quantiles = {}
        
        # High volatility regime
        if 't/sigma_ew' in data.columns:
            sigma_cols = [col for col in data.columns if col.startswith('t/sigma_ew')]
            if sigma_cols:
                sigma_values = data[sigma_cols].mean(axis=1)
                quantiles['high_vol_q70'] = sigma_values.quantile(0.7)
        
        # Wide spread regime
        if 't/spread_z18' in data.columns:
            spread_cols = [col for col in data.columns if col.startswith('t/spread_z18')]
            if spread_cols:
                spread_values = data[spread_cols].mean(axis=1)
                quantiles['wide_spread_q70'] = spread_values.quantile(0.7)
        
        self.quantiles = quantiles
        return quantiles
    
    def get_high_vol_flag(self, data: pd.DataFrame) -> pd.Series:
        """Get high volatility regime flag."""
        if 'high_vol_q70' not in self.quantiles:
            return pd.Series(0, index=data.index)
        
        sigma_cols = [col for col in data.columns if col.startswith('t/sigma_ew')]
        if not sigma_cols:
            return pd.Series(0, index=data.index)
        
        sigma_values = data[sigma_cols].mean(axis=1)
        return (sigma_values > self.quantiles['high_vol_q70']).astype(int)
    
    def get_wide_spread_flag(self, data: pd.DataFrame) -> pd.Series:
        """Get wide spread regime flag."""
        if 'wide_spread_q70' not in self.quantiles:
            return pd.Series(0, index=data.index)
        
        spread_cols = [col for col in data.columns if col.startswith('t/spread_z18')]
        if not spread_cols:
            return pd.Series(0, index=data.index)
        
        spread_values = data[spread_cols].mean(axis=1)
        return (spread_values > self.quantiles['wide_spread_q70']).astype(int)


class InteractionEngine:
    """Engine for creating interactions from transformed features."""
    
    def __init__(self, config: Dict[str, InteractionConfig]):
        self.config = config
        self.regime_flags = RegimeFlags()
        self.interaction_cache = {}
    
    def build_interactions(self, 
                          transformed_data: pd.DataFrame,
                          patch_features: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """Build all interactions from transformed data."""
        
        # Calculate regime flags
        self.regime_flags.calculate_quantiles(transformed_data)
        
        interactions = {}
        
        for interaction_id, config in self.config.items():
            try:
                interaction = self._create_interaction(
                    interaction_id, config, transformed_data, patch_features
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
                           patch_features: Optional[Dict[str, pd.Series]] = None) -> Optional[pd.Series]:
        """Create a specific interaction."""
        
        # Check required fields
        missing_fields = set(config.required_fields) - set(data.columns)
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
        elif interaction_id == 'i/micro/dollarvol_x_widespread':
            return self._micro_dollarvol_x_widespread(data)
        elif interaction_id == 'i/vol/r1_x_rvshort':
            return self._vol_r1_x_rvshort(data)
        elif interaction_id == 'i/vol/r3_x_rvshort':
            return self._vol_r3_x_rvshort(data)
        elif interaction_id == 'i/vol/vwapdist_x_rvshort':
            return self._vol_vwapdist_x_rvshort(data)
        elif interaction_id == 'i/vol/autocorr_x_rvshort':
            return self._vol_autocorr_x_rvshort(data)
        elif interaction_id == 'i/model/yhat1_x_rvshort':
            return self._model_yhat1_x_rvshort(data, patch_features)
        elif interaction_id == 'i/model/yhat1_x_vwapdist':
            return self._model_yhat1_x_vwapdist(data, patch_features)
        elif interaction_id == 'i/model/yhatconf_x_widespread':
            return self._model_yhatconf_x_widespread(data, patch_features)
        else:
            warnings.warn(f"Unknown interaction ID: {interaction_id}")
            return None
    
    # Tension interactions
    def _tension_mom5_x_negmom20(self, data: pd.DataFrame) -> pd.Series:
        """t/mom5/* × (-t/mom20/*)"""
        mom5_cols = [col for col in data.columns if 't/mom5' in col]
        mom20_cols = [col for col in data.columns if 't/mom20' in col]
        
        if not mom5_cols or not mom20_cols:
            return pd.Series(0, index=data.index)
        
        mom5 = data[mom5_cols].mean(axis=1)
        mom20 = data[mom20_cols].mean(axis=1)
        
        return mom5 * (-mom20)
    
    def _tension_rsi14_x_highvol(self, data: pd.DataFrame) -> pd.Series:
        """t/rsi14/* × 1[high_vol]"""
        rsi14_cols = [col for col in data.columns if 't/rsi14' in col]
        if not rsi14_cols:
            return pd.Series(0, index=data.index)
        
        rsi14 = data[rsi14_cols].mean(axis=1)
        high_vol = self.regime_flags.get_high_vol_flag(data)
        
        return rsi14 * high_vol
    
    def _tension_bollz_x_widespread(self, data: pd.DataFrame) -> pd.Series:
        """t/bollz20/* × 1[wide_spread]"""
        bollz_cols = [col for col in data.columns if 't/bollz20' in col]
        if not bollz_cols:
            return pd.Series(0, index=data.index)
        
        bollz = data[bollz_cols].mean(axis=1)
        wide_spread = self.regime_flags.get_wide_spread_flag(data)
        
        return bollz * wide_spread
    
    def _tension_vwapdist_x_open30(self, data: pd.DataFrame) -> pd.Series:
        """t/vwap_session_dist/* × p/open30"""
        vwap_cols = [col for col in data.columns if 't/vwap_session_dist' in col]
        open30_cols = [col for col in data.columns if 'p/open30' in col]
        
        if not vwap_cols or not open30_cols:
            return pd.Series(0, index=data.index)
        
        vwap_dist = data[vwap_cols].mean(axis=1)
        open30 = data[open30_cols].mean(axis=1)
        
        return vwap_dist * open30
    
    # Microstructure interactions
    def _micro_ofi_x_spread(self, data: pd.DataFrame) -> pd.Series:
        """t/ofi_proxy/* × t/spread_z18/*"""
        ofi_cols = [col for col in data.columns if 't/ofi_proxy' in col]
        spread_cols = [col for col in data.columns if 't/spread_z18' in col]
        
        if not ofi_cols or not spread_cols:
            return pd.Series(0, index=data.index)
        
        ofi = data[ofi_cols].mean(axis=1)
        spread = data[spread_cols].mean(axis=1)
        
        return ofi * spread
    
    def _micro_tradecount_x_spread(self, data: pd.DataFrame) -> pd.Series:
        """t/tradecount_z18/* × t/spread_z18/*"""
        tc_cols = [col for col in data.columns if 't/tradecount_z18' in col]
        spread_cols = [col for col in data.columns if 't/spread_z18' in col]
        
        if not tc_cols or not spread_cols:
            return pd.Series(0, index=data.index)
        
        tc = data[tc_cols].mean(axis=1)
        spread = data[spread_cols].mean(axis=1)
        
        return tc * spread
    
    def _micro_microprice_x_ofi(self, data: pd.DataFrame) -> pd.Series:
        """t/microprice_dev/* × t/ofi_proxy/*"""
        microprice_cols = [col for col in data.columns if 't/microprice_dev' in col]
        ofi_cols = [col for col in data.columns if 't/ofi_proxy' in col]
        
        if not microprice_cols or not ofi_cols:
            return pd.Series(0, index=data.index)
        
        microprice = data[microprice_cols].mean(axis=1)
        ofi = data[ofi_cols].mean(axis=1)
        
        return microprice * ofi
    
    def _micro_dollarvol_x_widespread(self, data: pd.DataFrame) -> pd.Series:
        """t/dollarvol_z18/* × 1[wide_spread]"""
        dollarvol_cols = [col for col in data.columns if 't/dollarvol_z18' in col]
        if not dollarvol_cols:
            return pd.Series(0, index=data.index)
        
        dollarvol = data[dollarvol_cols].mean(axis=1)
        wide_spread = self.regime_flags.get_wide_spread_flag(data)
        
        return dollarvol * wide_spread
    
    # Volatility interactions
    def _vol_r1_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/r1/* × t/rv_short_3/*"""
        r1_cols = [col for col in data.columns if 't/r1' in col]
        rv_cols = [col for col in data.columns if 't/rv_short_3' in col]
        
        if not r1_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        r1 = data[r1_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return r1 * rv
    
    def _vol_r3_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/r3/* × t/rv_short_3/*"""
        r3_cols = [col for col in data.columns if 't/r3' in col]
        rv_cols = [col for col in data.columns if 't/rv_short_3' in col]
        
        if not r3_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        r3 = data[r3_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return r3 * rv
    
    def _vol_vwapdist_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/vwap_session_dist/* × t/rv_short_3/*"""
        vwap_cols = [col for col in data.columns if 't/vwap_session_dist' in col]
        rv_cols = [col for col in data.columns if 't/rv_short_3' in col]
        
        if not vwap_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        vwap_dist = data[vwap_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return vwap_dist * rv
    
    def _vol_autocorr_x_rvshort(self, data: pd.DataFrame) -> pd.Series:
        """t/autocorr_r1_w/* × t/rv_short_3/*"""
        autocorr_cols = [col for col in data.columns if 't/autocorr_r1_w' in col]
        rv_cols = [col for col in data.columns if 't/rv_short_3' in col]
        
        if not autocorr_cols or not rv_cols:
            return pd.Series(0, index=data.index)
        
        autocorr = data[autocorr_cols].mean(axis=1)
        rv = data[rv_cols].mean(axis=1)
        
        return autocorr * rv
    
    # Model interactions
    def _model_yhat1_x_rvshort(self, data: pd.DataFrame, patch_features: Optional[Dict[str, pd.Series]]) -> pd.Series:
        """y_hat_h1 × t/rv_short_3/*"""
        if not patch_features or 'y_hat_h1' not in patch_features:
            return pd.Series(0, index=data.index)
        
        rv_cols = [col for col in data.columns if 't/rv_short_3' in col]
        if not rv_cols:
            return pd.Series(0, index=data.index)
        
        yhat1 = patch_features['y_hat_h1']
        rv = data[rv_cols].mean(axis=1)
        
        return yhat1 * rv
    
    def _model_yhat1_x_vwapdist(self, data: pd.DataFrame, patch_features: Optional[Dict[str, pd.Series]]) -> pd.Series:
        """y_hat_h1 × t/vwap_session_dist/*"""
        if not patch_features or 'y_hat_h1' not in patch_features:
            return pd.Series(0, index=data.index)
        
        vwap_cols = [col for col in data.columns if 't/vwap_session_dist' in col]
        if not vwap_cols:
            return pd.Series(0, index=data.index)
        
        yhat1 = patch_features['y_hat_h1']
        vwap_dist = data[vwap_cols].mean(axis=1)
        
        return yhat1 * vwap_dist
    
    def _model_yhatconf_x_widespread(self, data: pd.DataFrame, patch_features: Optional[Dict[str, pd.Series]]) -> pd.Series:
        """y_hat_conf × 1[wide_spread]"""
        if not patch_features or 'y_hat_conf' not in patch_features:
            return pd.Series(0, index=data.index)
        
        yhat_conf = patch_features['y_hat_conf']
        wide_spread = self.regime_flags.get_wide_spread_flag(data)
        
        return yhat_conf * wide_spread


def create_default_interaction_config() -> Dict[str, InteractionConfig]:
    """Create default interaction configuration."""
    config = {}
    
    # Tension interactions
    config['i/tension/mom5_x_negmom20'] = InteractionConfig(
        interaction_id='i/tension/mom5_x_negmom20',
        formula='t/mom5/* × (-t/mom20/*)',
        required_fields=['t/mom5', 't/mom20'],
        regime_dependent=False,
        interaction_type=InteractionType.TENSION,
        description='Momentum tension: short vs long momentum'
    )
    
    config['i/tension/rsi14_x_highvol'] = InteractionConfig(
        interaction_id='i/tension/rsi14_x_highvol',
        formula='t/rsi14/* × 1[high_vol]',
        required_fields=['t/rsi14', 't/sigma_ew'],
        regime_dependent=True,
        interaction_type=InteractionType.TENSION,
        description='RSI in high volatility regime'
    )
    
    config['i/tension/bollz_x_widespread'] = InteractionConfig(
        interaction_id='i/tension/bollz_x_widespread',
        formula='t/bollz20/* × 1[wide_spread]',
        required_fields=['t/bollz20', 't/spread_z18'],
        regime_dependent=True,
        interaction_type=InteractionType.TENSION,
        description='Bollinger z-score in wide spread regime'
    )
    
    config['i/tension/vwapdist_x_open30'] = InteractionConfig(
        interaction_id='i/tension/vwapdist_x_open30',
        formula='t/vwap_session_dist/* × p/open30',
        required_fields=['t/vwap_session_dist', 'p/open30'],
        regime_dependent=False,
        interaction_type=InteractionType.TENSION,
        description='VWAP distance during opening period'
    )
    
    # Microstructure interactions
    config['i/micro/ofi_x_spread'] = InteractionConfig(
        interaction_id='i/micro/ofi_x_spread',
        formula='t/ofi_proxy/* × t/spread_z18/*',
        required_fields=['t/ofi_proxy', 't/spread_z18'],
        regime_dependent=False,
        interaction_type=InteractionType.MICRO,
        description='Order flow imbalance vs spread'
    )
    
    config['i/micro/tradecount_x_spread'] = InteractionConfig(
        interaction_id='i/micro/tradecount_x_spread',
        formula='t/tradecount_z18/* × t/spread_z18/*',
        required_fields=['t/tradecount_z18', 't/spread_z18'],
        regime_dependent=False,
        interaction_type=InteractionType.MICRO,
        description='Trade count vs spread'
    )
    
    config['i/micro/microprice_x_ofi'] = InteractionConfig(
        interaction_id='i/micro/microprice_x_ofi',
        formula='t/microprice_dev/* × t/ofi_proxy/*',
        required_fields=['t/microprice_dev', 't/ofi_proxy'],
        regime_dependent=False,
        interaction_type=InteractionType.MICRO,
        description='Microprice deviation vs OFI'
    )
    
    config['i/micro/dollarvol_x_widespread'] = InteractionConfig(
        interaction_id='i/micro/dollarvol_x_widespread',
        formula='t/dollarvol_z18/* × 1[wide_spread]',
        required_fields=['t/dollarvol_z18', 't/spread_z18'],
        regime_dependent=True,
        interaction_type=InteractionType.MICRO,
        description='Dollar volume in wide spread regime'
    )
    
    # Volatility interactions
    config['i/vol/r1_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/r1_x_rvshort',
        formula='t/r1/* × t/rv_short_3/*',
        required_fields=['t/r1', 't/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='1-bar return vs short-term volatility'
    )
    
    config['i/vol/r3_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/r3_x_rvshort',
        formula='t/r3/* × t/rv_short_3/*',
        required_fields=['t/r3', 't/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='3-bar return vs short-term volatility'
    )
    
    config['i/vol/vwapdist_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/vwapdist_x_rvshort',
        formula='t/vwap_session_dist/* × t/rv_short_3/*',
        required_fields=['t/vwap_session_dist', 't/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='VWAP distance vs short-term volatility'
    )
    
    config['i/vol/autocorr_x_rvshort'] = InteractionConfig(
        interaction_id='i/vol/autocorr_x_rvshort',
        formula='t/autocorr_r1_w/* × t/rv_short_3/*',
        required_fields=['t/autocorr_r1_w', 't/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.VOL,
        description='Return autocorrelation vs short-term volatility'
    )
    
    # Model interactions
    config['i/model/yhat1_x_rvshort'] = InteractionConfig(
        interaction_id='i/model/yhat1_x_rvshort',
        formula='y_hat_h1 × t/rv_short_3/*',
        required_fields=['t/rv_short_3'],
        regime_dependent=False,
        interaction_type=InteractionType.MODEL,
        description='Model prediction vs short-term volatility'
    )
    
    config['i/model/yhat1_x_vwapdist'] = InteractionConfig(
        interaction_id='i/model/yhat1_x_vwapdist',
        formula='y_hat_h1 × t/vwap_session_dist/*',
        required_fields=['t/vwap_session_dist'],
        regime_dependent=False,
        interaction_type=InteractionType.MODEL,
        description='Model prediction vs VWAP distance'
    )
    
    config['i/model/yhatconf_x_widespread'] = InteractionConfig(
        interaction_id='i/model/yhatconf_x_widespread',
        formula='y_hat_conf × 1[wide_spread]',
        required_fields=['t/spread_z18'],
        regime_dependent=True,
        interaction_type=InteractionType.MODEL,
        description='Model confidence in wide spread regime'
    )
    
    return config