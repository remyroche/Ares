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

# Import existing feature generation components
import sys
sys.path.append('src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation')
from feature_engineering_roadmap.feature_registry import FeatureRegistry
from feature_engineering_roadmap.transforms import TransformRouter, create_default_transform_config
from .htf_utils import (
    build_htf_family_catalog,
    format_transform_suffix,
    resample_htf_series,
)
from ..feature_interaction_generation.feature_engineering import (
    FeatureRegistry,
    FeatureFamily,
    TransformRouter,
    create_default_transform_config,
)

from . import htf_base_features

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


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
        tprint_debug("[RIH] Initialized RIHStateManager with empty state store")
    
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
        tprint_success(
            f"[RIH] Initialized state for {feature_name} (family={family}, lookback={lookback})"
        )
        return state
    
    def _create_initial_state_data(self, family: str, lookback: int) -> Dict[str, Any]:
        """Create initial state data based on feature family."""
        if family in ['trend_level_vol']:
            # EMA/EWσ state
            tprint_debug(f"[RIH] Creating EMA state for {family} with lookback={lookback}")
            return {
                'ema_state': 0.0,
                'var_state': 1.0,
                'count': 0,
                'alpha': 1 - np.exp(-np.log(2) / (lookback / 2))
            }
        elif family == 'oscillators':
            # RSI/Stoch state
            tprint_debug(f"[RIH] Creating oscillator state with lookback={lookback}")
            return {
                'gain_state': 0.0,
                'loss_state': 0.0,
                'count': 0,
                'period': lookback
            }
        elif family == 'anchors':
            # VWAP state
            tprint_debug("[RIH] Creating VWAP state")
            return {
                'vwap_state': 0.0,
                'volume_state': 0.0,
                'count': 0
            }
        else:
            # Default state
            tprint_debug(f"[RIH] Creating default rolling state with lookback={lookback}")
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
            tprint_warning(f"[RIH] State not found for {feature_name}; returning 0.0")
            return 0.0

        state = self.states[feature_name]
        family = state.metadata.get('family', 'unknown')
        tprint_debug(
            f"[RIH] Updating state for {feature_name} (family={family}) with value={new_value}"
        )
        
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
        tprint_debug(
            f"[RIH] Updated state for {feature_name}; current_value={updated_value:.4f}"
        )

        return updated_value
    
    def _update_ema_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update EMA state."""
        alpha = state.state_data['alpha']
        ema_state = state.state_data['ema_state']
        var_state = state.state_data['var_state']
        count = state.state_data['count']
        
        # Update count
        state.state_data['count'] = count + 1
        tprint_debug(
            f"[RIH] EMA update #{count + 1} for {state.feature_name}: alpha={alpha:.4f}, new_value={new_value}"
        )
        
        # Update EMA
        new_ema = (1 - alpha) * ema_state + alpha * new_value
        state.state_data['ema_state'] = new_ema
        
        # Update variance (Welford's algorithm)
        if count > 0:
            delta = new_value - new_ema
            new_var = (1 - alpha) * var_state + alpha * delta**2
            state.state_data['var_state'] = new_var
            tprint_debug(
                f"[RIH] EMA variance update for {state.feature_name}: new_var={new_var:.6f}"
            )

        return new_ema
    
    def _update_oscillator_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update oscillator state (RSI/Stoch)."""
        period = state.state_data['period']
        gain_state = state.state_data['gain_state']
        loss_state = state.state_data['loss_state']
        count = state.state_data['count']
        
        # Update count
        state.state_data['count'] = count + 1
        tprint_debug(
            f"[RIH] Oscillator update #{count + 1} for {state.feature_name} with new_value={new_value}"
        )
        
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

            tprint_debug(
                f"[RIH] RSI computed for {state.feature_name}: gain={new_gain:.4f}, loss={new_loss:.4f}, rsi={rsi:.2f}"
            )

            state.state_data['prev_value'] = new_value
            return rsi
        else:
            state.state_data['prev_value'] = new_value
            tprint_debug(
                f"[RIH] First oscillator observation for {state.feature_name}; defaulting RSI to 50.0"
            )
            return 50.0  # Neutral RSI
    
    def _update_vwap_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update VWAP state."""
        vwap_state = state.state_data['vwap_state']
        volume_state = state.state_data['volume_state']
        count = state.state_data['count']
        
        # Update count
        state.state_data['count'] = count + 1
        tprint_debug(
            f"[RIH] VWAP update #{count + 1} for {state.feature_name} with price={new_value}"
        )
        
        # For VWAP, we need price and volume
        # This is simplified - in practice, you'd pass both price and volume
        price = new_value
        volume = 1.0  # Simplified
        
        # Update VWAP
        new_vwap = (vwap_state * volume_state + price * volume) / (volume_state + volume)
        state.state_data['vwap_state'] = new_vwap
        state.state_data['volume_state'] = volume_state + volume
        tprint_debug(
            f"[RIH] VWAP state for {state.feature_name}: vwap={new_vwap:.4f}, volume={volume_state + volume}"
        )

        return new_vwap
    
    def _update_default_state(self, state: HTFFeatureState, new_value: float) -> float:
        """Update default state (rolling sum)."""
        sum_state = state.state_data['sum_state']
        count = state.state_data['count']
        lookback = state.state_data['lookback']
        
        # Update count
        state.state_data['count'] = count + 1
        tprint_debug(
            f"[RIH] Default rolling update #{count + 1} for {state.feature_name} with value={new_value}"
        )
        
        # Rolling sum
        new_sum = sum_state + new_value
        
        # Keep only last 'lookback' values
        if count >= lookback:
            # Remove oldest value (simplified)
            new_sum -= state.state_data.get('oldest_value', 0)
        
        state.state_data['sum_state'] = new_sum
        state.state_data['oldest_value'] = new_value
        tprint_debug(
            f"[RIH] Rolling sum for {state.feature_name}: sum={new_sum:.4f}"
        )

        return new_sum
    
    def get_state(self, feature_name: str) -> Optional[HTFFeatureState]:
        """Get current state for a feature."""
        tprint_debug(f"[RIH] Retrieving state for {feature_name}")
        return self.states.get(feature_name)

    def save_states(self, filepath: str):
        """Save all states to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.states, f)
        tprint_success(f"[RIH] Saved {len(self.states)} states to {filepath}")

    def load_states(self, filepath: str):
        """Load states from disk."""
        with open(filepath, 'rb') as f:
            self.states = pickle.load(f)
        tprint_success(f"[RIH] Loaded {len(self.states)} states from {filepath}")


class EHUStateManager:
    """Manages state for EHU features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.states = {}  # feature_name -> HTFFeatureState
        tprint_debug("[EHU] Initialized EHUStateManager with empty state store")
    
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
        tprint_success(
            f"[EHU] Initialized state for {feature_name} (family={family}, lookback={lookback})"
        )
        return state
    
    def update_state(self, 
                    feature_name: str,
                    new_value: float,
                    timestamp: datetime) -> float:
        """Update EHU state with new value."""
        if feature_name not in self.states:
            self.logger.warning(f"State not found for {feature_name}")
            tprint_warning(f"[EHU] State not found for {feature_name}; returning 0.0")
            return 0.0

        state = self.states[feature_name]
        state.current_value = new_value
        state.last_update = timestamp
        tprint_debug(
            f"[EHU] Updated state for {feature_name}; current_value={new_value:.4f}"
        )

        return new_value
    
    def get_state(self, feature_name: str) -> Optional[HTFFeatureState]:
        """Get current state for a feature."""
        tprint_debug(f"[EHU] Retrieving state for {feature_name}")
        return self.states.get(feature_name)

    def save_states(self, filepath: str):
        """Save all states to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.states, f)
        tprint_success(f"[EHU] Saved {len(self.states)} states to {filepath}")

    def load_states(self, filepath: str):
        """Load states from disk."""
        with open(filepath, 'rb') as f:
            self.states = pickle.load(f)
        tprint_success(f"[EHU] Loaded {len(self.states)} states from {filepath}")


class HTFFeatureGenerator:
    """Generates HTF features using the same FeatureBank functions."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.feature_registry = FeatureRegistry()
        self.rih_manager = RIHStateManager()
        self.ehu_manager = EHUStateManager()
        self.htf_families, self.base_feature_to_family = build_htf_family_catalog(
            self.feature_registry
        )
        tprint_debug(
            "[HTFGen] Initialized HTFFeatureGenerator with"
            f" {len(self.base_feature_to_family)} registered base features"
        )
    
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
        if feature_name not in self.base_feature_to_family:
            raise ValueError(f"Feature '{feature_name}' is not registered for HTF materialization")

        expected_family = self.base_feature_to_family[feature_name]
        if family != expected_family:
            self.logger.warning(
                "Family mismatch for %s: expected %s, received %s",
                feature_name,
                expected_family,
                family,
            )
            tprint_warning(
                f"[HTFGen] Family mismatch for {feature_name}: expected {expected_family}, received {family}"
            )

        metadata = self.feature_registry.get_feature_metadata(feature_name)
        base_series = self.feature_registry.compute_feature(feature_name, data)
        htf_series = resample_htf_series(base_series, lookback, metadata.family)
        tprint_debug(
            f"[HTFGen] Generated base series for {feature_name} (lookback={lookback}, style={update_style.value})"
        )

        transformed_series, transform_suffix = self._apply_transforms(
            feature_name,
            lookback,
            htf_series,
        )
        tprint_success(
            f"[HTFGen] Applied transforms for {feature_name}; suffix={transform_suffix}"
        )
        
        # Create state
        if update_style == UpdateStyle.RIH:
            state = self.rih_manager.initialize_state(feature_name, lookback, family)
        else:
            state = self.ehu_manager.initialize_state(feature_name, lookback, family)
        
        # Create materialized HTF
        materialized_htf = MaterializedHTF(
            feature_name=f"t/{feature_name}_htf{lookback}/{transform_suffix}",
            family=family,
            lookback=lookback,
            update_style=update_style,
            feature_series=transformed_series,
            transform_applied=transform_suffix,
            state=state,
            metadata={
                'base_feature': feature_name,
                'lookback_minutes': lookback,
                'created_at': datetime.now(),
                'data_length': len(transformed_series)
            }
        )

        tprint_success(
            f"[HTFGen] Materialized HTF feature {materialized_htf.feature_name}"
        )
        return materialized_htf

    def _apply_transforms(
        self,
        feature_name: str,
        lookback: int,
        htf_series: pd.Series,
    ) -> Tuple[pd.Series, str]:
        """Apply default transforms and return the series with its suffix."""
        transform_config = create_default_transform_config([feature_name])
        transform_router = TransformRouter(transform_config)
        tprint_debug(
            f"[HTFGen] Applying transforms for {feature_name} with config keys:"
            f" {list(transform_config.keys())}"
        )

        transformed = transform_router.fit_transform(
            pd.DataFrame({feature_name: htf_series}),
            pd.DataFrame({feature_name: htf_series}),
        )

        transformed_df = transformed.get(feature_name, {}).get('train')
        if transformed_df is None or transformed_df.empty:
            empty_series = pd.Series(index=htf_series.index, dtype=float)
            tprint_warning(
                f"[HTFGen] Transform output empty for {feature_name}; returning empty series"
            )
            return empty_series, format_transform_suffix(transform_config[feature_name])

        transformed_series = transformed_df.iloc[:, 0]
        suffix = format_transform_suffix(transform_config[feature_name])
        transformed_series.name = f"t/{feature_name}_htf{lookback}/{suffix}"
        tprint_debug(
            f"[HTFGen] Transform produced series for {feature_name} with suffix {suffix}"
        )
        return transformed_series, suffix

    def _create_transform_router(self, feature_names: List[str]) -> TransformRouter:
        """Create transform router for features."""
        config = create_default_transform_config(feature_names)
        tprint_debug(
            f"[HTFGen] Created TransformRouter for features: {feature_names}"
        )
        return TransformRouter(config)


class HTFMaterialization:
    """Main HTF materialization system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.htf_generator = HTFFeatureGenerator(config)
        self.materialized_features = {}  # feature_name -> MaterializedHTF
        tprint_debug("[HTFMat] Initialized HTFMaterialization manager")
    
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
        tprint("🔧 Starting HTF materialization")
        tprint(f"   → Selected features: {len(selected_features)}")
        tprint(f"   → Sessionized data keys: {list(sessionized_data.keys())}")

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
                tprint_success(
                    f"[HTFMat] Materialized {materialized_htf.feature_name} ({update_style.value.upper()})"
                )

            except Exception as e:
                self.logger.warning(f"Failed to materialize {feature.feature_name}: {e}")
                tprint_error(f"[HTFMat] Failed to materialize {feature.feature_name}: {e}")
                continue

        self.logger.info(f"HTF materialization completed: {len(materialized_features)} features materialized")
        tprint_success(
            f"[HTFMat] Completed HTF materialization with {len(materialized_features)} features"
        )
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
                tprint_debug(
                    f"[HTFMat] Updating {feature_name} ({materialized_htf.update_style.value.upper()})"
                )
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
                tprint_debug(
                    f"[HTFMat] Updated {feature_name}; current_value={updated_value:.4f}"
                )

            except Exception as e:
                self.logger.warning(f"Failed to update {feature_name}: {e}")
                tprint_error(f"[HTFMat] Failed to update {feature_name}: {e}")
                continue

        return updated_values
    
    def get_feature_values(self) -> Dict[str, float]:
        """Get current values of all materialized features."""
        values = {}
        for feature_name, materialized_htf in self.materialized_features.items():
            values[feature_name] = materialized_htf.state.current_value
        tprint_debug(f"[HTFMat] Retrieved values for {len(values)} features")
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
        tprint_success(
            f"[HTFMat] Saved {len(feature_data)} materialized features to {filepath}"
        )
    
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
        tprint_success(
            f"[HTFMat] Loaded {len(self.materialized_features)} materialized features from {filepath}"
        )
    
    def get_materialization_summary(self) -> Dict[str, Any]:
        """Get summary of materialized features."""
        ehu_count = sum(1 for f in self.materialized_features.values() if f.update_style == UpdateStyle.EHU)
        rih_count = sum(1 for f in self.materialized_features.values() if f.update_style == UpdateStyle.RIH)
        
        family_counts = {}
        for feature in self.materialized_features.values():
            family = feature.family
            family_counts[family] = family_counts.get(family, 0) + 1

        summary = {
            'total_features': len(self.materialized_features),
            'ehu_count': ehu_count,
            'rih_count': rih_count,
            'family_counts': family_counts,
            'feature_names': list(self.materialized_features.keys())
        }
        tprint_debug(
            "[HTFMat] Materialization summary: "
            f"total={summary['total_features']}, EHU={summary['ehu_count']}, RIH={summary['rih_count']}"
        )
        return summary
