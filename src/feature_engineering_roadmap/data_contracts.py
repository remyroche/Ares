"""
Data Contracts for End-to-End Roadmap System

Defines data structures and validation for:
- Input bars (per asset)
- Feature store (wide, by timestamp)
- Artifacts registry
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum


class BarType(Enum):
    """Types of bar data available."""
    OHLCV = "ohlcv"
    BOOK = "book"
    CONTEXT = "context"


@dataclass
class InputBar:
    """Input bar data contract per asset."""
    timestamp: datetime  # tz-aware, exchange calendar aligned
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Optional book data
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trade_count: Optional[int] = None
    
    # Optional context data
    index_close: Optional[float] = None  # for beta calculation
    sector_id: Optional[str] = None
    
    # Session information
    session_id: Optional[Union[str, datetime]] = None  # date or explicit calendar session key
    
    def __post_init__(self):
        """Validate bar data after initialization."""
        if self.high < max(self.open, self.close):
            raise ValueError("High must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Low must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")
        if self.bid is not None and self.ask is not None and self.bid >= self.ask:
            raise ValueError("Bid must be < ask")


@dataclass
class FeatureStore:
    """Feature store contract - wide format by timestamp."""
    timestamp: pd.DatetimeIndex
    features: pd.DataFrame  # Columns named with registry paths
    spec_hash: str  # For bit-for-bit reproduction
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate feature store data."""
        if len(self.timestamp) != len(self.features):
            raise ValueError("Timestamp and features must have same length")
        
        # Check for forward-filled engineered features (not allowed)
        if self.features.isnull().any().any():
            # NaN is allowed, but we should log a warning
            pass
        
        # Validate spec_hash is present
        if not self.spec_hash:
            raise ValueError("spec_hash is required for reproducibility")


@dataclass
class TransformParams:
    """Transform parameters for reproducibility."""
    transform_type: str  # 'ewz', 'tod_rank', 'signed_log', 'winsor'
    params: Dict[str, Any]
    spec_hash: str
    
    def __post_init__(self):
        """Validate transform parameters."""
        valid_types = ['ewz', 'tod_rank', 'signed_log', 'winsor']
        if self.transform_type not in valid_types:
            raise ValueError(f"Transform type must be one of {valid_types}")


@dataclass
class LookbackChoice:
    """Lookback choice for a feature family."""
    family: str
    selected_lookback: int
    selection_criteria: str  # 'ic', 'auc', 'simplicity'
    confidence_score: float
    spec_hash: str


@dataclass
class InteractionConfig:
    """Interaction configuration."""
    interaction_id: str
    formula: str
    required_fields: List[str]
    regime_dependent: bool
    spec_hash: str


@dataclass
class ModelArtifact:
    """Model artifact with training metadata."""
    model_type: str  # 'lightgbm', 'patch', 'gru'
    model_object: Any
    training_metadata: Dict[str, Any]
    feature_importance: Dict[str, float]
    spec_hash: str
    
    def __post_init__(self):
        """Validate model artifact."""
        valid_types = ['lightgbm', 'patch', 'gru']
        if self.model_type not in valid_types:
            raise ValueError(f"Model type must be one of {valid_types}")


@dataclass
class ArtifactsRegistry:
    """Registry of all artifacts for reproducibility."""
    transform_params: Dict[str, TransformParams]
    lookback_choices: Dict[str, LookbackChoice]
    interaction_configs: Dict[str, InteractionConfig]
    model_artifacts: Dict[str, ModelArtifact]
    rotation_metadata: Optional[Dict[str, Any]] = None
    patch_weights: Optional[Dict[str, Any]] = None
    residual_std: Optional[float] = None
    spec_hash: str = ""
    
    def __post_init__(self):
        """Validate artifacts registry."""
        if not self.spec_hash:
            # Generate spec_hash from all components
            import hashlib
            content = str(sorted(self.__dict__.items()))
            self.spec_hash = hashlib.md5(content.encode()).hexdigest()


class DataContractValidator:
    """Validator for data contracts."""
    
    @staticmethod
    def validate_input_bars(bars: List[InputBar]) -> bool:
        """Validate list of input bars."""
        if not bars:
            raise ValueError("No bars provided")
        
        # Check chronological order
        timestamps = [bar.timestamp for bar in bars]
        if timestamps != sorted(timestamps):
            raise ValueError("Bars must be in chronological order")
        
        # Check for duplicates
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("Duplicate timestamps found")
        
        return True
    
    @staticmethod
    def validate_feature_store(store: FeatureStore) -> bool:
        """Validate feature store."""
        if store.features.empty:
            raise ValueError("Feature store cannot be empty")
        
        # Check column naming convention (registry paths)
        for col in store.features.columns:
            if not (col.startswith('p/') or col.startswith('t/') or col.startswith('i/')):
                raise ValueError(f"Feature column '{col}' must follow registry path convention")
        
        return True
    
    @staticmethod
    def validate_artifacts_registry(registry: ArtifactsRegistry) -> bool:
        """Validate artifacts registry."""
        if not registry.spec_hash:
            raise ValueError("Artifacts registry must have spec_hash")
        
        # Validate all transform params have spec_hash
        for name, params in registry.transform_params.items():
            if not params.spec_hash:
                raise ValueError(f"Transform params '{name}' missing spec_hash")
        
        return True


def create_input_bars_from_dataframe(df: pd.DataFrame, 
                                   symbol: str,
                                   exchange: str) -> List[InputBar]:
    """Create InputBar objects from DataFrame."""
    bars = []
    
    for idx, row in df.iterrows():
        bar = InputBar(
            timestamp=row['timestamp'] if 'timestamp' in row else idx,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            bid=row.get('bid'),
            ask=row.get('ask'),
            bid_size=row.get('bid_size'),
            ask_size=row.get('ask_size'),
            trade_count=row.get('trade_count'),
            index_close=row.get('index_close'),
            sector_id=row.get('sector_id'),
            session_id=row.get('session_id')
        )
        bars.append(bar)
    
    return bars


def create_feature_store(timestamp: pd.DatetimeIndex,
                        features: pd.DataFrame,
                        metadata: Dict[str, Any]) -> FeatureStore:
    """Create FeatureStore from components."""
    import hashlib

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
    
    # Generate spec_hash
    content = f"{timestamp.tolist()}{features.to_dict()}{metadata}"
    spec_hash = hashlib.md5(content.encode()).hexdigest()
    
    return FeatureStore(
        timestamp=timestamp,
        features=features,
        spec_hash=spec_hash,
        metadata=metadata
    )
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
