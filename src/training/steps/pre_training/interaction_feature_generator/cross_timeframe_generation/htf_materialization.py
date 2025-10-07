"""
HTF Materialization System

Implements materialization of selected HTF features with:
- Same FeatureBank functions and TransformRouter as base features
- RIH features with incremental state maintenance
- EHU features with session-based updates
- Consistent naming convention
- State persistence and recovery
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
import pickle
import json
from pathlib import Path

from ..feature_interaction_generation.feature_engineering import (
    FeatureRegistry,
    FeatureFamily,
    TransformRouter,
    create_default_transform_config,
)

from . import htf_base_features


class UpdateStyle(Enum):
    """Update style for HTF features."""
    EHU = "ehu"  # End-of-Hour Update
    RIH = "rih"  # Real-time Incremental Update


@dataclass
class HTFFeatureState:
    """State for an HTF feature."""
    feature_name: str
    lookback: int
    update_style: UpdateStyle
    last_update: datetime
    current_value: float
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class MaterializedHTF:
    """Materialized HTF feature."""
    feature_name: str
    family: str
    lookback: int
    update_style: UpdateStyle
    feature_series: pd.Series
    transform_applied: str
    state: HTFFeatureState
    metadata: Dict[str, Any]


class RIHStateManager:
    """Manages incremental state for RIH features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.states = {}  # feature_name -> HTFFeatureState
    
    def initialize_state(self, 
                        feature_name: str,
                        lookback: int,
                        family: str) -> HTFFeatureState:
        """Initialize state for a new RIH feature."""
        state_data = self._create_initial_state_data(family, lookback)
        
        state = HTFFeatureState(
            feature_name=feature_name,
            lookback=lookback,
            update_style=UpdateStyle.RIH,
            last_update=datetime.now(),
            current_value=0.0,
            state_data=state_data,
            metadata={'family': family, 'initialized': True}
        )
        
        self.states[feature_name] = state
        return state
    
    def _create_initial_state_data(self, family: str, lookback: int) -> Dict[str, Any]:
        """Create initial state data based on feature family."""
        if family in ['trend_level_vol']:
            # EMA/EWσ state
            return {
                'ema_state': 0.0,
                'var_state': 1.0,
                'count': 0,
                'alpha': 1 - np.exp(-np.log(2) / (lookback / 2))
            }
        elif family == 'oscillators':
            # RSI/Stoch state
            return {
                'gain_state': 0.0,
                'loss_state': 0.0,
                'count': 0,
                'period': lookback
            }
        elif family == 'anchors':
            # VWAP state
            return {
                'vwap_state': 0.0,
                'volume_state': 0.0,
                'count': 0
            }
        else:
            # Default state
            return {
                'sum_state': 0.0,
                'count': 0,
                'lookback': lookback
            }
    
    def update_state(self, 
                    feature_name: str,
                    new_value: float,
                    timestamp: datetime) -> float:
        """Update RIH state with new value."""
        if feature_name not in self.states:
            self.logger.warning(f"State not found for {feature_name}")
            return 0.0
        
        state = self.states[feature_name]
        family = state.metadata.get('family', 'unknown')
        
        # Update based on family type
        if family in ['trend_level_vol']:
            updated_value = self._update_ema_state(state, new_value)
        elif family == 'oscillators':
            updated_value = self._update_oscillator_state(state, new_value)
        elif family == 'anchors':
            updated_value = self._update_vwap_state(state, new_value)
        else:
            updated_value = self._update_default_state(state, new_value)
        
        # Update state metadata
        state.current_value = updated_value
        state.last_update = timestamp
        
        return updated_value
    
    def _update_ema_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update EMA state."""
        alpha = state.state_data['alpha']
        ema_state = state.state_data['ema_state']
        var_state = state.state_data['var_state']
        count = state.state_data['count']
        
        # Update count
        state.state_data['count'] = count + 1
        
        # Update EMA
        new_ema = (1 - alpha) * ema_state + alpha * new_value
        state.state_data['ema_state'] = new_ema
        
        # Update variance (Welford's algorithm)
        if count > 0:
            delta = new_value - new_ema
            new_var = (1 - alpha) * var_state + alpha * delta**2
            state.state_data['var_state'] = new_var
        
        return new_ema
    
    def _update_oscillator_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update oscillator state (RSI/Stoch)."""
        period = state.state_data['period']
        gain_state = state.state_data['gain_state']
        loss_state = state.state_data['loss_state']
        count = state.state_data['count']
        
        # Update count
        state.state_data['count'] = count + 1
        
        # Calculate gain and loss
        if count > 0:
            prev_value = state.state_data.get('prev_value', new_value)
            change = new_value - prev_value
            gain = max(0, change)
            loss = max(0, -change)
            
            # Update EW averages
            alpha = 1.0 / period
            new_gain = (1 - alpha) * gain_state + alpha * gain
            new_loss = (1 - alpha) * loss_state + alpha * loss
            
            state.state_data['gain_state'] = new_gain
            state.state_data['loss_state'] = new_loss
            
            # Calculate RSI
            if new_loss > 0:
                rs = new_gain / new_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0
            
            state.state_data['prev_value'] = new_value
            return rsi
        else:
            state.state_data['prev_value'] = new_value
            return 50.0  # Neutral RSI
    
    def _update_vwap_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update VWAP state."""
        vwap_state = state.state_data['vwap_state']
        volume_state = state.state_data['volume_state']
        count = state.state_data['count']
        
        # Update count
        state.state_data['count'] = count + 1
        
        # For VWAP, we need price and volume
        # This is simplified - in practice, you'd pass both price and volume
        price = new_value
        volume = 1.0  # Simplified
        
        # Update VWAP
        new_vwap = (vwap_state * volume_state + price * volume) / (volume_state + volume)
        state.state_data['vwap_state'] = new_vwap
        state.state_data['volume_state'] = volume_state + volume
        
        return new_vwap
    
    def _update_default_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update default state (rolling sum)."""
        sum_state = state.state_data['sum_state']
        count = state.state_data['count']
        lookback = state.state_data['lookback']
        
        # Update count
        state.state_data['count'] = count + 1
        
        # Rolling sum
        new_sum = sum_state + new_value
        
        # Keep only last 'lookback' values
        if count >= lookback:
            # Remove oldest value (simplified)
            new_sum -= state.state_data.get('oldest_value', 0)
        
        state.state_data['sum_state'] = new_sum
        state.state_data['oldest_value'] = new_value
        
        return new_sum
    
    def get_state(self, feature_name: str) -> Optional[HTFFeatureState]:
        """Get current state for a feature."""
        return self.states.get(feature_name)
    
    def save_states(self, filepath: str):
        """Save all states to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.states, f)
    
    def load_states(self, filepath: str):
        """Load states from disk."""
        with open(filepath, 'rb') as f:
            self.states = pickle.load(f)


class EHUStateManager:
    """Manages state for EHU features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.states = {}  # feature_name -> HTFFeatureState
    
    def initialize_state(self, 
                        feature_name: str,
                        lookback: int,
                        family: str) -> HTFFeatureState:
        """Initialize state for a new EHU feature."""
        state = HTFFeatureState(
            feature_name=feature_name,
            lookback=lookback,
            update_style=UpdateStyle.EHU,
            last_update=datetime.now(),
            current_value=0.0,
            state_data={'family': family, 'initialized': True},
            metadata={'family': family, 'initialized': True}
        )
        
        self.states[feature_name] = state
        return state
    
    def update_state(self, 
                    feature_name: str,
                    new_value: float,
                    timestamp: datetime) -> float:
        """Update EHU state with new value."""
        if feature_name not in self.states:
            self.logger.warning(f"State not found for {feature_name}")
            return 0.0
        
        state = self.states[feature_name]
        state.current_value = new_value
        state.last_update = timestamp
        
        return new_value
    
    def get_state(self, feature_name: str) -> Optional[HTFFeatureState]:
        """Get current state for a feature."""
        return self.states.get(feature_name)
    
    def save_states(self, filepath: str):
        """Save all states to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.states, f)
    
    def load_states(self, filepath: str):
        """Load states from disk."""
        with open(filepath, 'rb') as f:
            self.states = pickle.load(f)


class HTFFeatureGenerator:
    """Generates HTF features using the same FeatureBank functions."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.feature_registry = FeatureRegistry()
        self.rih_manager = RIHStateManager()
        self.ehu_manager = EHUStateManager()
    
    def generate_htf_feature(self, 
                           data: pd.DataFrame,
                           feature_name: str,
                           family: str,
                           lookback: int,
                           update_style: UpdateStyle) -> MaterializedHTF:
        """
        Generate HTF feature using same FeatureBank functions.
        
        Args:
            data: OHLCV data
            feature_name: Base feature name
            family: Feature family
            lookback: HTF lookback in minutes
            update_style: EHU or RIH
            
        Returns:
            Materialized HTF feature
        """
        # Get base feature computation function
        base_feature_func = htf_base_features.get_base_feature_func(feature_name)
        
        # Compute base feature
        base_series = base_feature_func(data)
        
        # Resample to HTF
        htf_series = htf_base_features.resample_to_htf(base_series, lookback, family)
        
        # Apply transform
        transform_router = self._create_transform_router([feature_name])
        transformed_data = transform_router.fit_transform(
            pd.DataFrame({feature_name: htf_series}),
            pd.DataFrame({feature_name: htf_series})
        )
        
        transformed_series = transformed_data[feature_name]['train']
        
        # Create state
        if update_style == UpdateStyle.RIH:
            state = self.rih_manager.initialize_state(feature_name, lookback, family)
        else:
            state = self.ehu_manager.initialize_state(feature_name, lookback, family)
        
        # Create materialized HTF
        materialized_htf = MaterializedHTF(
            feature_name=f"t/{feature_name}_htf{lookback}/ewz12",
            family=family,
            lookback=lookback,
            update_style=update_style,
            feature_series=transformed_series,
            transform_applied="ewz12",
            state=state,
            metadata={
                'base_feature': feature_name,
                'lookback_minutes': lookback,
                'created_at': datetime.now(),
                'data_length': len(transformed_series)
            }
        )
        
        return materialized_htf
    
    def _create_transform_router(self, feature_names: List[str]) -> TransformRouter:
        """Create transform router for features."""
        config = create_default_transform_config(feature_names)
        return TransformRouter(config)


class HTFMaterialization:
    """Main HTF materialization system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.htf_generator = HTFFeatureGenerator(config)
        self.materialized_features = {}  # feature_name -> MaterializedHTF
    
    def materialize_htfs(self, 
                        sessionized_data: Dict[str, Any],
                        selected_features: List[Any]) -> Dict[str, MaterializedHTF]:
        """
        Materialize selected HTF features.
        
        Args:
            sessionized_data: Sessionized and aligned data
            selected_features: Selected features from knapsack selection
            
        Returns:
            Dictionary of materialized HTF features
        """
        self.logger.info("Starting HTF materialization")
        
        aligned_data = sessionized_data['aligned_data']
        materialized_features = {}
        
        for feature in selected_features:
            try:
                # Determine update style
                update_style = UpdateStyle.EHU if feature.update_style == 'ehu' else UpdateStyle.RIH
                
                # Generate HTF feature
                materialized_htf = self.htf_generator.generate_htf_feature(
                    aligned_data,
                    feature.feature_name,
                    feature.family,
                    feature.lookback,
                    update_style
                )
                
                # Store materialized feature
                materialized_features[materialized_htf.feature_name] = materialized_htf
                self.materialized_features[materialized_htf.feature_name] = materialized_htf
                
            except Exception as e:
                self.logger.warning(f"Failed to materialize {feature.feature_name}: {e}")
                continue
        
        self.logger.info(f"HTF materialization completed: {len(materialized_features)} features materialized")
        return materialized_features
    
    def update_htf_features(self, 
                          new_data: pd.DataFrame,
                          timestamp: datetime) -> Dict[str, float]:
        """
        Update HTF features with new data.
        
        Args:
            new_data: New OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Dictionary of updated feature values
        """
        updated_values = {}
        
        for feature_name, materialized_htf in self.materialized_features.items():
            try:
                if materialized_htf.update_style == UpdateStyle.RIH:
                    # Update RIH feature incrementally
                    # This would involve computing the base feature and updating state
                    # For now, we'll simulate the update
                    updated_value = self.htf_generator.rih_manager.update_state(
                        materialized_htf.state.feature_name,
                        new_data['close'].iloc[-1],  # Simplified
                        timestamp
                    )
                else:
                    # EHU features are updated at HTF close
                    # For now, we'll simulate the update
                    updated_value = self.htf_generator.ehu_manager.update_state(
                        materialized_htf.state.feature_name,
                        new_data['close'].iloc[-1],  # Simplified
                        timestamp
                    )
                
                updated_values[feature_name] = updated_value
                
            except Exception as e:
                self.logger.warning(f"Failed to update {feature_name}: {e}")
                continue
        
        return updated_values
    
    def get_feature_values(self) -> Dict[str, float]:
        """Get current values of all materialized features."""
        values = {}
        for feature_name, materialized_htf in self.materialized_features.items():
            values[feature_name] = materialized_htf.state.current_value
        return values
    
    def save_materialized_features(self, filepath: str):
        """Save materialized features to disk."""
        # Save feature data
        feature_data = {}
        for name, feature in self.materialized_features.items():
            feature_data[name] = {
                'feature_name': feature.feature_name,
                'family': feature.family,
                'lookback': feature.lookback,
                'update_style': feature.update_style.value,
                'feature_series': feature.feature_series.to_dict(),
                'transform_applied': feature.transform_applied,
                'state': {
                    'feature_name': feature.state.feature_name,
                    'lookback': feature.state.lookback,
                    'update_style': feature.state.update_style.value,
                    'last_update': feature.state.last_update.isoformat(),
                    'current_value': feature.state.current_value,
                    'state_data': feature.state.state_data,
                    'metadata': feature.state.metadata
                },
                'metadata': feature.metadata
            }
        
        with open(filepath, 'w') as f:
            json.dump(feature_data, f, indent=2, default=str)
    
    def load_materialized_features(self, filepath: str):
        """Load materialized features from disk."""
        with open(filepath, 'r') as f:
            feature_data = json.load(f)
        
        self.materialized_features = {}
        for name, data in feature_data.items():
            # Reconstruct MaterializedHTF object
            state = HTFFeatureState(
                feature_name=data['state']['feature_name'],
                lookback=data['state']['lookback'],
                update_style=UpdateStyle(data['state']['update_style']),
                last_update=datetime.fromisoformat(data['state']['last_update']),
                current_value=data['state']['current_value'],
                state_data=data['state']['state_data'],
                metadata=data['state']['metadata']
            )
            
            materialized_htf = MaterializedHTF(
                feature_name=data['feature_name'],
                family=data['family'],
                lookback=data['lookback'],
                update_style=UpdateStyle(data['update_style']),
                feature_series=pd.Series(data['feature_series']),
                transform_applied=data['transform_applied'],
                state=state,
                metadata=data['metadata']
            )
            
            self.materialized_features[name] = materialized_htf
    
    def get_materialization_summary(self) -> Dict[str, Any]:
        """Get summary of materialized features."""
        ehu_count = sum(1 for f in self.materialized_features.values() if f.update_style == UpdateStyle.EHU)
        rih_count = sum(1 for f in self.materialized_features.values() if f.update_style == UpdateStyle.RIH)
        
        family_counts = {}
        for feature in self.materialized_features.values():
            family = feature.family
            family_counts[family] = family_counts.get(family, 0) + 1
        
        return {
            'total_features': len(self.materialized_features),
            'ehu_count': ehu_count,
            'rih_count': rih_count,
            'family_counts': family_counts,
            'feature_names': list(self.materialized_features.keys())
        }