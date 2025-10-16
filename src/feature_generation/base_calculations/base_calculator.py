"""
Base Calculator for Feature Generation

This module provides base calculation methods that can be used by different
feature generators, including price returns and returns-based VWAP calculations.
"""

import logging
import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.indicators.basic import RSI, MACD, ATR, BBANDS, STOCH, OBV, MA
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    RSI = None
    MACD = None
    ATR = None
    BBANDS = None
    STOCH = None
    OBV = None
    MA = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

logger = logging.getLogger(__name__)

class BaseCalculationType(Enum):
    """Types of base calculations."""
    PRICE_RETURNS = "price_returns"
    RETURNS_VWAP = "returns_vwap"
    PRICE_LEVELS = "price_levels"
    VOLUME_WEIGHTED = "volume_weighted"
    VOLUME_RETURNS = "volume_returns"

@dataclass
class BaseCalculationConfig:
    """Configuration for base calculations."""
    calculation_type: BaseCalculationType
    lookback_period: int = 1
    vwap_period: int = 20
    volume_column: str = "volume"
    price_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    open_column: str = "open"
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class BaseCalculator(ABC):
    """
    Abstract base class for base calculations.

    This class provides the interface for different base calculation methods
    that can be used by feature generators.
    """

    def __init__(self, config: BaseCalculationConfig):
        """
        Initialize the base calculator.

        Args:
            config: Base calculation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the base values.

        Args:
            data: Input data DataFrame

        Returns:
            Calculated base values as pandas Series
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            data: Input data DataFrame

        Raises:
            ValueError: If data validation fails
        """
        if data.empty:
            raise ValueError("Input data is empty")

        required_columns = self.get_required_columns()
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get required columns for this calculation.

        Returns:
            List of required column names
        """
        pass

class PriceReturnsCalculator(BaseCalculator):
    """
    Calculator for price returns.

    This calculator computes price returns (percentage changes) which can be used
    as a base for various technical indicators.
    """

    def __init__(self, config: Optional[BaseCalculationConfig] = None):
        """
        Initialize the price returns calculator.

        Args:
            config: Base calculation configuration
        """
        if config is None:
            config = BaseCalculationConfig(
                calculation_type=BaseCalculationType.PRICE_RETURNS,
                lookback_period=1
            )
        super().__init__(config)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price returns.

        Args:
            data: Input data DataFrame

        Returns:
            Price returns as pandas Series
        """
        self.validate_data(data)

        price_column = self.config.price_column
        lookback_period = self.config.lookback_period

        if lookback_period == 1:
            returns = data[price_column].pct_change()
        else:
            returns = data[price_column].pct_change(periods=lookback_period)

        return returns

    def get_required_columns(self) -> List[str]:
        """Get required columns for price returns calculation."""
        return [self.config.price_column]

class ReturnsVWAPCalculator(BaseCalculator):
    """
    Calculator for returns-based VWAP.

    This calculator computes VWAP (Volume Weighted Average Price) and then
    calculates returns based on the VWAP values.
    """

    def __init__(self, config: Optional[BaseCalculationConfig] = None):
        """
        Initialize the returns VWAP calculator.

        Args:
            config: Base calculation configuration
        """
        if config is None:
            config = BaseCalculationConfig(
                calculation_type=BaseCalculationType.RETURNS_VWAP,
                vwap_period=20,
                lookback_period=1
            )
        super().__init__(config)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate returns-based VWAP.

        Args:
            data: Input data DataFrame

        Returns:
            Returns-based VWAP as pandas Series
        """
        self.validate_data(data)

        # Calculate VWAP
        vwap = self._calculate_vwap(data)

        # Calculate returns based on VWAP
        lookback_period = self.config.lookback_period
        if lookback_period == 1:
            returns_vwap = vwap.pct_change()
        else:
            returns_vwap = vwap.pct_change(periods=lookback_period)

        return returns_vwap

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price).

        Args:
            data: Input data DataFrame

        Returns:
            VWAP as pandas Series
        """
        high = data[self.config.high_column]
        low = data[self.config.low_column]
        close = data[self.config.price_column]
        volume = data[self.config.volume_column]

        # Calculate typical price
        typical_price = (high + low + close) / 3

        # Calculate VWAP
        vwap_period = self.config.vwap_period
        vwap = (typical_price * volume).rolling(window=vwap_period).sum() / self._vectorbt_rolling_operation(volume, "sum", vwap_period)

        return vwap

    def get_required_columns(self) -> List[str]:
        """Get required columns for returns VWAP calculation."""
        return [
            self.config.high_column,
            self.config.low_column,
            self.config.price_column,
            self.config.volume_column
        ]

class PriceLevelsCalculator(BaseCalculator):
    """
    Calculator for price levels (raw prices).

    This calculator provides raw price levels which can be used as a base
    for various technical indicators.
    """

    def __init__(self, config: Optional[BaseCalculationConfig] = None):
        """
        Initialize the price levels calculator.

        Args:
            config: Base calculation configuration
        """
        if config is None:
            config = BaseCalculationConfig(
                calculation_type=BaseCalculationType.PRICE_LEVELS,
                price_column="close"
            )
        super().__init__(config)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price levels.

        Args:
            data: Input data DataFrame

        Returns:
            Price levels as pandas Series
        """
        self.validate_data(data)
        return data[self.config.price_column]

    def get_required_columns(self) -> List[str]:
        """Get required columns for price levels calculation."""
        return [self.config.price_column]

class VolumeWeightedCalculator(BaseCalculator):
    """
    Calculator for volume-weighted calculations.

    This calculator provides volume-weighted values which can be used as a base
    for various technical indicators.
    """

    def __init__(self, config: Optional[BaseCalculationConfig] = None):
        """
        Initialize the volume-weighted calculator.

        Args:
            config: Base calculation configuration
        """
        if config is None:
            config = BaseCalculationConfig(
                calculation_type=BaseCalculationType.VOLUME_WEIGHTED,
                price_column="close",
                volume_column="volume"
            )
        super().__init__(config)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume-weighted values.

        Args:
            data: Input data DataFrame

        Returns:
            Volume-weighted values as pandas Series
        """
        self.validate_data(data)

        price = data[self.config.price_column]
        volume = data[self.config.volume_column]

        # Calculate volume-weighted price
        volume_weighted = (price * volume) / volume.rolling(window=self.config.vwap_period).sum()

        return volume_weighted

    def get_required_columns(self) -> List[str]:
        """Get required columns for volume-weighted calculation."""
        return [self.config.price_column, self.config.volume_column]

class VolumeReturnsCalculator(BaseCalculator):
    """
    Calculator for volume returns calculations.

    This calculator provides volume returns (percentage changes in volume) which can be used as a base
    for various volume-based technical indicators.
    """

    def __init__(self, config: Optional[BaseCalculationConfig] = None):
        """
        Initialize the volume returns calculator.

        Args:
            config: Base calculation configuration
        """
        if config is None:
            config = BaseCalculationConfig(
                calculation_type=BaseCalculationType.VOLUME_RETURNS,
                volume_column="volume",
                lookback_period=1
            )
        super().__init__(config)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume returns.

        Args:
            data: Input data DataFrame

        Returns:
            Volume returns as pandas Series
        """
        self.validate_data(data)

        volume = data[self.config.volume_column]

        # Calculate volume returns (percentage change)
        volume_returns = volume.pct_change(periods=self.config.lookback_period)

        return volume_returns

    def get_required_columns(self) -> List[str]:
        """Get required columns for volume returns calculation."""
        return [self.config.volume_column]

# Factory functions
def create_base_calculator(calculation_type: Union[str, BaseCalculationType], **kwargs) -> BaseCalculator:
    """
    Create a base calculator of the specified type.

    Args:
        calculation_type: Type of base calculation
        **kwargs: Additional configuration parameters

    Returns:
        Base calculator instance
    """
    if isinstance(calculation_type, str):
        try:
            calculation_type = BaseCalculationType(calculation_type)
        except ValueError:
            raise ValueError(f"Invalid calculation type: {calculation_type}")

    # Create configuration
    config = BaseCalculationConfig(
        calculation_type=calculation_type,
        **kwargs
    )

    # Create appropriate calculator
    if calculation_type == BaseCalculationType.PRICE_RETURNS:
        return PriceReturnsCalculator(config)
    elif calculation_type == BaseCalculationType.RETURNS_VWAP:
        return ReturnsVWAPCalculator(config)
    elif calculation_type == BaseCalculationType.PRICE_LEVELS:
        return PriceLevelsCalculator(config)
    elif calculation_type == BaseCalculationType.VOLUME_WEIGHTED:
        return VolumeWeightedCalculator(config)
    elif calculation_type == BaseCalculationType.VOLUME_RETURNS:
        return VolumeReturnsCalculator(config)
    else:
        raise ValueError(f"Unsupported calculation type: {calculation_type}")

def get_base_calculator(calculation_type: Union[str, BaseCalculationType], **kwargs) -> BaseCalculator:
    """
    Get a base calculator instance.

    Args:
        calculation_type: Type of base calculation
        **kwargs: Additional configuration parameters

    Returns:
        Base calculator instance
    """
    return create_base_calculator(calculation_type, **kwargs)

# Convenience functions for common calculations
def calculate_price_returns(data: pd.DataFrame,
                          price_column: str = "close",
                          lookback_period: int = 1) -> pd.Series:
    """
    Calculate price returns.

    Args:
        data: Input data DataFrame
        price_column: Name of the price column
        lookback_period: Lookback period for returns calculation

    Returns:
        Price returns as pandas Series
    """
    calculator = PriceReturnsCalculator(BaseCalculationConfig(
        calculation_type=BaseCalculationType.PRICE_RETURNS,
        price_column=price_column,
        lookback_period=lookback_period
    ))
    return calculator.calculate(data)

def calculate_returns_vwap(data: pd.DataFrame,
                         vwap_period: int = 20,
                         lookback_period: int = 1,
                         high_column: str = "high",
                         low_column: str = "low",
                         close_column: str = "close",
                         volume_column: str = "volume") -> pd.Series:
    """
    Calculate returns-based VWAP.

    Args:
        data: Input data DataFrame
        vwap_period: Period for VWAP calculation
        lookback_period: Lookback period for returns calculation
        high_column: Name of the high column
        low_column: Name of the low column
        close_column: Name of the close column
        volume_column: Name of the volume column

    Returns:
        Returns-based VWAP as pandas Series
    """
    calculator = ReturnsVWAPCalculator(BaseCalculationConfig(
        calculation_type=BaseCalculationType.RETURNS_VWAP,
        vwap_period=vwap_period,
        lookback_period=lookback_period,
        high_column=high_column,
        low_column=low_column,
        close_column=close_column,
        volume_column=volume_column
    ))
    return calculator.calculate(data)

def calculate_price_levels(data: pd.DataFrame, price_column: str = "close") -> pd.Series:
    """
    Calculate price levels.

    Args:
        data: Input data DataFrame
        price_column: Name of the price column

    Returns:
        Price levels as pandas Series
    """
    calculator = PriceLevelsCalculator(BaseCalculationConfig(
        calculation_type=BaseCalculationType.PRICE_LEVELS,
        price_column=price_column
    ))
    return calculator.calculate(data)

def calculate_volume_weighted(data: pd.DataFrame,
                            price_column: str = "close",
                            volume_column: str = "volume",
                            period: int = 20) -> pd.Series:
    """
    Calculate volume-weighted values.

    Args:
        data: Input data DataFrame
        price_column: Name of the price column
        volume_column: Name of the volume column
        period: Period for volume weighting

    Returns:
        Volume-weighted values as pandas Series
    """
    calculator = VolumeWeightedCalculator(BaseCalculationConfig(
        calculation_type=BaseCalculationType.VOLUME_WEIGHTED,
        price_column=price_column,
        volume_column=volume_column,
        vwap_period=period
    ))
    return calculator.calculate(data)

def calculate_volume_returns(data: pd.DataFrame,
                           volume_column: str = "volume",
                           lookback_period: int = 1) -> pd.Series:
    """
    Calculate volume returns.

    Args:
        data: Input data DataFrame
        volume_column: Name of the volume column
        lookback_period: Lookback period for returns calculation

    Returns:
        Volume returns as pandas Series
    """
    calculator = VolumeReturnsCalculator(BaseCalculationConfig(
        calculation_type=BaseCalculationType.VOLUME_RETURNS,
        volume_column=volume_column,
        lookback_period=lookback_period
    ))
    return calculator.calculate(data)
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
