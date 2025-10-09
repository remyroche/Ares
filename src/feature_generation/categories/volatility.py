"""
Volatility Feature Generator

This module provides feature generators for volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
Supports different base calculations: price returns, returns-based VWAP, etc.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class VolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility-based features with batch processing and optimization."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None, base_calculation: Optional[BaseCalculationType] = None):
        self.period = period
        self.base_calculation = base_calculation
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Volatility measure over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                "period": period
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'VolatilityFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'volatility_{self.period}')

        close_prices = data['close'].astype(float).values
        state = self.get_state()
        history = state.get('close_history') or []

        if history:
            try:
                history_array = np.asarray(history, dtype=float)
            except Exception:
                history_array = np.array(history, dtype=float)
            combined_closes = np.concatenate([history_array, close_prices])
        else:
            combined_closes = close_prices

        combined_volatility = self._calculate_volatility(combined_closes, period=self.period)
        volatility = combined_volatility[-len(close_prices):] if len(close_prices) else np.array([])

        return pd.Series(volatility, index=data.index, name=f'volatility_{self.period}')
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window=period-1).std().values
        return np.concatenate([[np.nan], volatility])

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        if not data.empty:
            closes = data['close'].astype(float)
            history_window = max(self.period, 1)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'last_close': float(closes.iloc[-1]),
                'close_history': close_history
            }
            if not feature_data.empty:
                last_vol = feature_data.iloc[-1]
                if pd.notna(last_vol):
                    state_update['last_volatility'] = float(last_vol)
            self.update_state(state_update)

        super()._finalize_state(data, feature_data)
    
    @classmethod
    def generate_batch_features(cls,
                               data: pd.DataFrame,
                               periods: List[int] = [5, 10, 20, 30],
                               volatility_types: List[str] = ["returns", "price"],
                               **kwargs) -> Dict[str, pd.Series]:
        """
        Generate volatility features for multiple periods and types in batch.
        
        Args:
            data: Input data DataFrame
            periods: List of periods to calculate
            volatility_types: List of volatility calculation types
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to Series
        """
        features = {}
        close_prices = data['close']
        
        for period in periods:
            for vol_type in volatility_types:
                if vol_type == "returns":
                    # Calculate volatility based on returns
                    returns = close_prices.pct_change().dropna()
                    volatility = returns.rolling(window=period).std()
                    # Pad with NaN to match original length
                    volatility = pd.concat([pd.Series([np.nan] * (len(close_prices) - len(volatility)), index=close_prices.index[:len(close_prices) - len(volatility)]), volatility])
                elif vol_type == "price":
                    # Calculate volatility based on price levels
                    volatility = close_prices.rolling(window=period).std()
                else:
                    continue
                
                feature_name = f"volatility_{vol_type}_{period}"
                features[feature_name] = volatility
        
        return features

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class BollingerBandsGenerator(VectorizedFeatureGenerator):
    """Generator for Bollinger Bands with different base calculations and batch processing."""
    
    def __init__(self,
                 period: int = 20,
                 std_dev: float = 2.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 band_type: str = "upper",  # "upper", "lower", "middle"
                 **base_kwargs):
        """
        Initialize Bollinger Bands generator.
        
        Args:
            period: Bollinger Bands period
            std_dev: Standard deviation multiplier
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            band_type: Type of band to generate ("upper", "lower", "middle")
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"bb_{band_type}_{period}_{std_dev}_{base_calculation.value}",
            category=FeatureCategory.VOLATILITY,
            description=f"Bollinger Bands {band_type} with period={period}, std={std_dev} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'std_dev': std_dev,
                'base_calculation': base_calculation.value,
                'band_type': band_type,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.std_dev = std_dev
        self.base_calculation = base_calculation
        self.band_type = band_type
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Bollinger Bands based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate Bollinger Bands on base values
        sma = base_values.rolling(window=self.period).mean()
        std = base_values.rolling(window=self.period).std()
        
        if self.band_type == "upper":
            band = sma + (std * self.std_dev)
        elif self.band_type == "lower":
            band = sma - (std * self.std_dev)
        elif self.band_type == "middle":
            band = sma
        else:
            raise ValueError(f"Invalid band_type: {self.band_type}")
        
        return band
    
    @classmethod
    def generate_batch_features(cls, 
                               data: pd.DataFrame,
                               periods: List[int] = [10, 20, 30],
                               std_devs: List[float] = [1.5, 2.0, 2.5],
                               band_types: List[str] = ["upper", "lower", "middle"],
                               base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                               **base_kwargs) -> Dict[str, pd.Series]:
        """
        Generate Bollinger Bands features for multiple periods, std_devs, and band_types in batch.
        
        Args:
            data: Input data DataFrame
            periods: List of periods to calculate
            std_devs: List of standard deviation multipliers
            band_types: List of band types to generate
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
            
        Returns:
            Dictionary mapping feature names to Series
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        base_values = base_calculator.calculate(data)
        
        features = {}
        
        # Vectorized calculation for all combinations
        for period in periods:
            # Calculate rolling mean and std for this period
            rolling_mean = base_values.rolling(window=period).mean()
            rolling_std = base_values.rolling(window=period).std()
            
            for std_dev in std_devs:
                for band_type in band_types:
                    if band_type == "upper":
                        band = rolling_mean + (rolling_std * std_dev)
                    elif band_type == "lower":
                        band = rolling_mean - (rolling_std * std_dev)
                    elif band_type == "middle":
                        band = rolling_mean
                    else:
                        continue
                    
                    feature_name = f"bb_{band_type}_{period}_{std_dev}_{base_calculation.value}"
                    features[feature_name] = band
        
        return features

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class ATRGenerator(VectorizedFeatureGenerator):
    """Generator for Average True Range with different base calculations and batch processing."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize ATR generator.
        
        Args:
            period: ATR period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"atr_{period}_{base_calculation.value}",
            category=FeatureCategory.VOLATILITY,
            description=f"Average True Range over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate ATR based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional ATR calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=self.period).mean()
            return atr
        else:
            # For other base calculations, calculate ATR on the base values
            base_values = self.base_calculator.calculate(data)
            
            # Calculate rolling standard deviation as volatility measure
            atr = base_values.rolling(window=self.period).std()
            return atr
    
    @classmethod
    def generate_batch_features(cls,
                               data: pd.DataFrame,
                               periods: List[int] = [7, 14, 21],
                               base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                               **base_kwargs) -> Dict[str, pd.Series]:
        """
        Generate ATR features for multiple periods in batch.
        
        Args:
            data: Input data DataFrame
            periods: List of periods to calculate
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
            
        Returns:
            Dictionary mapping feature names to Series
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        features = {}
        
        if base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional ATR calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR for each period
            for period in periods:
                atr = true_range.rolling(window=period).mean()
                feature_name = f"atr_{period}_{base_calculation.value}"
                features[feature_name] = atr
        else:
            # For other base calculations, calculate ATR on the base values
            base_calculator = create_base_calculator(base_calculation, **base_kwargs)
            base_values = base_calculator.calculate(data)
            
            # Calculate rolling standard deviation for each period
            for period in periods:
                atr = base_values.rolling(window=period).std()
                feature_name = f"atr_{period}_{base_calculation.value}"
                features[feature_name] = atr
        
        return features


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class VolatilityBandsGenerator(VectorizedFeatureGenerator):
    """Generator for Volatility Bands with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 std_multiplier: float = 2.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Volatility Bands generator.
        
        Args:
            period: Period for volatility calculation
            std_multiplier: Standard deviation multiplier for bands
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"volatility_bands_{period}_{std_multiplier}_{base_calculation.value}",
            category=FeatureCategory.VOLATILITY,
            description=f"Volatility Bands with period={period}, multiplier={std_multiplier} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'std_multiplier': std_multiplier,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.std_multiplier = std_multiplier
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Volatility Bands upper band based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate moving average and standard deviation
        sma = base_values.rolling(window=self.period).mean()
        volatility = base_values.rolling(window=self.period).std()
        
        # Return upper band as the main feature
        # Lower band would be: sma - (volatility * multiplier)
        # Middle line would be: sma
        upper_band = sma + (volatility * self.std_multiplier)
        
        return upper_band


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class GARCHFeatureGenerator(VectorizedFeatureGenerator):
    """Generator for GARCH-based volatility features."""

    def __init__(self,
                 p: int = 1,
                 q: int = 1,
                 forecast_horizon: int = 1,
                 **garch_kwargs):
        """
        Initialize GARCH generator.

        Args:
            p: GARCH lag order
            q: ARCH lag order
            forecast_horizon: Number of steps to forecast
            **garch_kwargs: Additional parameters for GARCH model
        """
        config = FeatureConfig(
            name=f"garch_{p}_{q}_h{forecast_horizon}",
            category=FeatureCategory.VOLATILITY,
            description=f"GARCH({p},{q}) volatility model with {forecast_horizon}-step forecast using vectorized rolling windows",
            required_columns=["close"],
            default_lookback=252,  # Use 1 year of data for GARCH fitting
            min_lookback=100,      # Minimum 100 data points for reliable GARCH
            max_lookback=1000,
            parameters={
                'p': p,
                'q': q,
                'forecast_horizon': forecast_horizon,
                **garch_kwargs
            },
            dependencies=["arch"]  # Require arch library for GARCH models
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.p = p
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.garch_kwargs = garch_kwargs

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate GARCH-based volatility features using vectorized calculations."""
        return self._generate_garch_vectorized(data)

    def _generate_garch_vectorized(self, data: pd.DataFrame) -> pd.Series:
        """Generate GARCH-based volatility features using vectorized rolling window approach."""
        from arch import arch_model

        close_prices = data['close'].dropna()
        if len(close_prices) < self.config.min_lookback:
            # Instead of all NaN, provide fallback volatility estimate
            fallback_volatility = close_prices.pct_change().std() if len(close_prices) > 1 else 0.02
            return pd.Series([fallback_volatility] * len(data), index=data.index, name=self.config.name)

        # Calculate returns
        returns = 100 * close_prices.pct_change().dropna()

        if len(returns) < 50:  # Need minimum data for GARCH
            # Fallback to rolling volatility for insufficient data
            fallback_volatility = close_prices.pct_change().rolling(window=min(20, len(returns))).std().fillna(0.02)
            # Pad to match data length
            pad_length = len(data) - len(fallback_volatility)
            result = pd.Series([0.02] * pad_length + fallback_volatility.tolist(), index=data.index, name=self.config.name)
            return result

        try:
            # Use vectorized rolling window with pandas
            window_size = min(252, len(returns))  # Use up to 252 days for fitting

            def fit_garch_window(window_returns: pd.Series) -> float:
                """Fit GARCH model on a window and return volatility forecast."""
                if len(window_returns) < 50:  # Minimum data requirement
                    # Fallback to simple volatility estimate
                    return window_returns.std() if len(window_returns) > 1 else 0.02

                try:
                    # Fit GARCH model
                    model = arch_model(window_returns, p=self.p, q=self.q, **self.garch_kwargs)
                    model_fit = model.fit(disp='off')

                    # Generate forecast
                    forecast = model_fit.forecast(horizon=self.forecast_horizon)
                    volatility_forecast = forecast.variance.iloc[-1].values[0] if self.forecast_horizon == 1 else forecast.variance.iloc[-1].values[0]
                    return volatility_forecast

                except Exception:
                    # Fallback to rolling volatility when GARCH fails
                    return window_returns.rolling(window=min(20, len(window_returns))).std().iloc[-1] if len(window_returns) > 1 else 0.02

            # Apply GARCH fitting to rolling windows
            # For vectorized processing, we'll use expanding windows with proper alignment
            volatility_forecasts = []

            # Process in chunks for better performance
            chunk_size = min(100, len(returns) - window_size + 1)

            for i in range(0, len(returns) - window_size + 1, chunk_size):
                end_idx = min(i + chunk_size, len(returns) - window_size + 1)

                for j in range(i, end_idx):
                    start_idx = j
                    end_idx_window = start_idx + window_size

                    if end_idx_window > len(returns):
                        break

                    window_returns = returns.iloc[start_idx:end_idx_window]

                    try:
                        model = arch_model(window_returns, p=self.p, q=self.q, **self.garch_kwargs)
                        model_fit = model.fit(disp='off')
                        forecast = model_fit.forecast(horizon=self.forecast_horizon)
                        volatility_forecast = forecast.variance.iloc[-1].values[0] if self.forecast_horizon == 1 else forecast.variance.iloc[-1].values[0]
                        volatility_forecasts.append(volatility_forecast)
                    except Exception:
                        # Fallback volatility when GARCH fitting fails for this window
                        window_vol = window_returns.std() if len(window_returns) > 1 else 0.02
                        volatility_forecasts.append(window_vol)

            # Pad the beginning with fallback volatility to match data length
            pad_length = len(data) - len(volatility_forecasts)
            # Use fallback volatility (0.02) for padding instead of NaN
            fallback_vol = 0.02
            volatility_series = pd.Series([fallback_vol] * pad_length + volatility_forecasts,
                                        index=data.index, name=self.config.name)

            return volatility_series

        except Exception as e:
            logging.getLogger(__name__).warning(f"⚠️ Vectorized GARCH calculation failed: {e}, using fallback volatility")
            # Fallback to simple rolling volatility when GARCH completely fails
            close_prices = data['close'].dropna()
            if len(close_prices) > 1:
                fallback_volatility = close_prices.pct_change().rolling(window=min(20, len(close_prices))).std().fillna(0.02)
                # Pad to match data length
                pad_length = len(data) - len(fallback_volatility)
                result = pd.Series([0.02] * pad_length + fallback_volatility.tolist(), index=data.index, name=self.config.name)
                return result
            else:
                return pd.Series([0.02] * len(data), index=data.index, name=self.config.name)



def create_volatility_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of volatility feature generators."""
    if periods is None:
        periods = {
            'bb': [20],
            'atr': [14],
            'volatility': [10, 20],
            'volatility_bands': [20],
            'garch': [(1, 1, 1), (1, 1, 5)]  # GARCH(p,q,h) configurations
        }
    
    generators = []
    
    # Bollinger Bands generators
    for period in periods.get('bb', [20]):
        generators.append(BollingerBandsGenerator(period))
    
    # ATR generators
    for period in periods.get('atr', [14]):
        generators.append(ATRGenerator(period))
    
    # Volatility Bands generators
    for period in periods.get('volatility_bands', [20]):
        generators.append(VolatilityBandsGenerator(period))

    # Basic volatility generators
    for period in periods.get('volatility', [10, 20]):
        generators.append(VolatilityFeatureGenerator(period))

    # GARCH generators
    for garch_config in periods.get('garch', [(1, 1, 1), (1, 1, 5)]):
        p, q, h = garch_config
        generators.append(GARCHFeatureGenerator(p=p, q=q, forecast_horizon=h))

    # Analyst Features - Volatility structure generators
    class AnalystVolatilityRatio5m15mGenerator(VectorizedFeatureGenerator):
        """Generator for volatility ratio between 5m and 15m timeframes."""

        def __init__(self):
            config = FeatureConfig(
                name="analyst_vol_ratio_5m_15m",
                category=FeatureCategory.VOLATILITY,
                description="Analyst volatility ratio between 5m and 15m timeframes",
                required_columns=["close"],
                default_lookback=60,
                min_lookback=20,
                max_lookback=200,
                parameters={}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            """Generate volatility ratio feature."""
            returns = data['close'].pct_change()

            # 5m volatility (20 periods)
            vol_5m = returns.rolling(20).std()

            # 15m volatility (60 periods)
            vol_15m = returns.rolling(60).std()

            # Volatility ratio
            vol_ratio = vol_5m / vol_15m.replace(0, 1)
            return vol_ratio

    class AnalystVolatilityRegimeDeviationGenerator(VectorizedFeatureGenerator):
        """Generator for volatility regime deviation feature."""

        def __init__(self):
            config = FeatureConfig(
                name="analyst_vol_regime_deviation",
                category=FeatureCategory.VOLATILITY,
                description="Analyst current volatility relative to regime average",
                required_columns=["close"],
                default_lookback=100,
                min_lookback=50,
                max_lookback=300,
                parameters={}
            )
            super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
            """Generate volatility regime deviation feature."""
            returns = data['close'].pct_change()
            current_vol = returns.rolling(20).std()

            if regime_data is not None and 'regime' in regime_data.columns:
                # Calculate regime-specific volatility averages
                regime_vol_avgs = {}
                for regime in regime_data['regime'].unique():
                    regime_mask = regime_data['regime'] == regime
                    regime_vol_avgs[regime] = current_vol[regime_mask].mean()

                # Current regime deviation
                current_regime = regime_data['regime'].iloc[-1] if len(regime_data) > 0 else None
                if current_regime is not None and current_regime in regime_vol_avgs:
                    regime_avg_vol = regime_vol_avgs[current_regime]
                    regime_deviation = current_vol.iloc[-1] / regime_avg_vol if regime_avg_vol > 0 else 1.0
                else:
                    regime_deviation = 1.0
            else:
                # Default to 1.0 if no regime data available
                regime_deviation = 1.0

            # Create series with the same index as input data
            regime_deviation_series = pd.Series([regime_deviation] * len(data), index=data.index, name=self.config.name)
            return regime_deviation_series

    generators.append(AnalystVolatilityRatio5m15mGenerator())
    generators.append(AnalystVolatilityRegimeDeviationGenerator())
    
    # NEW FEATURES - Advanced Volatility Analysis
    # Realized volatility generators
    for window in periods.get('realized_volatility', [20]):
        generators.append(RealizedVolatilityGenerator(window))
    
    # Parkinson volatility generators
    for window in periods.get('parkinson_volatility', [20]):
        generators.append(ParkinsonVolatilityGenerator(window))
    
    # Garman-Klass volatility generators
    for window in periods.get('garman_klass_volatility', [20]):
        generators.append(GarmanKlassVolatilityGenerator(window))
    
    # Rogers-Satchell volatility generators
    for window in periods.get('rogers_satchell_volatility', [20]):
        generators.append(RogersSatchellVolatilityGenerator(window))
    
    # Vol of vol generators
    for vol_window in periods.get('vol_of_vol_windows', [20]):
        for vol_of_vol_window in periods.get('vol_of_vol_periods', [10]):
            generators.append(VolOfVolGenerator(vol_window, vol_of_vol_window))
    
    # Downside semivolatility generators
    for window in periods.get('downside_semivol', [20]):
        generators.append(DownsideSemivolGenerator(window))

    return generators

# NEW FEATURES - Advanced Volatility Analysis

class RealizedVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for realized volatility (standard deviation of returns over window)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"realized_volatility_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Realized volatility (std of returns) over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate realized volatility
        realized_vol = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 1:
                realized_vol[i] = np.std(valid_returns, ddof=1)
        
        return pd.Series(realized_vol, index=data.index)

class ParkinsonVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for Parkinson volatility estimator (uses OHLC)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"parkinson_volatility_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Parkinson volatility estimator over {window} periods",
            required_columns=["high", "low"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        high = data['high'].values
        low = data['low'].values
        
        if len(high) < self.window:
            return pd.Series(np.full(len(high), np.nan), index=data.index)
        
        # Calculate Parkinson volatility
        parkinson_vol = np.full(len(high), np.nan)
        for i in range(self.window - 1, len(high)):
            window_high = high[i - self.window + 1:i + 1]
            window_low = low[i - self.window + 1:i + 1]
            
            # Parkinson estimator: sqrt(1/(4*ln(2)) * sum(ln(H/L)^2))
            log_hl = np.log(window_high / window_low)
            parkinson_vol[i] = np.sqrt(np.mean(log_hl ** 2) / (4 * np.log(2)))
        
        return pd.Series(parkinson_vol, index=data.index)

class GarmanKlassVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for Garman-Klass volatility estimator (uses OHLC)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"garman_klass_volatility_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Garman-Klass volatility estimator over {window} periods",
            required_columns=["open", "high", "low", "close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        open_price = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if len(open_price) < self.window:
            return pd.Series(np.full(len(open_price), np.nan), index=data.index)
        
        # Calculate Garman-Klass volatility
        gk_vol = np.full(len(open_price), np.nan)
        for i in range(self.window - 1, len(open_price)):
            window_open = open_price[i - self.window + 1:i + 1]
            window_high = high[i - self.window + 1:i + 1]
            window_low = low[i - self.window + 1:i + 1]
            window_close = close[i - self.window + 1:i + 1]
            
            # Garman-Klass estimator
            log_hl = np.log(window_high / window_low)
            log_co = np.log(window_close / window_open)
            
            gk_vol[i] = np.sqrt(np.mean(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2))
        
        return pd.Series(gk_vol, index=data.index)

class RogersSatchellVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for Rogers-Satchell volatility estimator (uses OHLC)."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"rogers_satchell_volatility_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Rogers-Satchell volatility estimator over {window} periods",
            required_columns=["open", "high", "low", "close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        open_price = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if len(open_price) < self.window:
            return pd.Series(np.full(len(open_price), np.nan), index=data.index)
        
        # Calculate Rogers-Satchell volatility
        rs_vol = np.full(len(open_price), np.nan)
        for i in range(self.window - 1, len(open_price)):
            window_open = open_price[i - self.window + 1:i + 1]
            window_high = high[i - self.window + 1:i + 1]
            window_low = low[i - self.window + 1:i + 1]
            window_close = close[i - self.window + 1:i + 1]
            
            # Rogers-Satchell estimator
            log_ho = np.log(window_high / window_open)
            log_hc = np.log(window_high / window_close)
            log_lo = np.log(window_low / window_open)
            log_lc = np.log(window_low / window_close)
            
            rs_vol[i] = np.sqrt(np.mean(log_ho * log_hc + log_lo * log_lc))
        
        return pd.Series(rs_vol, index=data.index)

class VolOfVolGenerator(VectorizedFeatureGenerator):
    """Generator for volatility of volatility (rolling std of realized volatility)."""
    
    def __init__(self, vol_window: int = 20, vol_of_vol_window: int = 10):
        config = FeatureConfig(
            name=f"vol_of_vol_{vol_window}_{vol_of_vol_window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Volatility of volatility over {vol_of_vol_window} periods (vol window {vol_window})",
            required_columns=["close"],
            default_lookback=vol_window + vol_of_vol_window,
            min_lookback=vol_window + vol_of_vol_window,
            max_lookback=(vol_window + vol_of_vol_window) * 3,  # Allow up to 3x window for optimization
            parameters={'vol_window': vol_window, 'vol_of_vol_window': vol_of_vol_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.vol_window = vol_window
        self.vol_of_vol_window = vol_of_vol_window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.vol_window + self.vol_of_vol_window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate realized volatility
        realized_vol = np.full(len(close), np.nan)
        for i in range(self.vol_window, len(close)):
            window_returns = returns[i - self.vol_window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 1:
                realized_vol[i] = np.std(valid_returns, ddof=1)
        
        # Calculate volatility of volatility
        vol_of_vol = np.full(len(close), np.nan)
        for i in range(self.vol_window + self.vol_of_vol_window, len(close)):
            vol_window = realized_vol[i - self.vol_of_vol_window + 1:i + 1]
            valid_vol = vol_window[np.isfinite(vol_window)]
            
            if len(valid_vol) > 1:
                vol_of_vol[i] = np.std(valid_vol, ddof=1)
        
        return pd.Series(vol_of_vol, index=data.index)

class DownsideSemivolGenerator(VectorizedFeatureGenerator):
    """Generator for downside semivolatility (std of min(returns, 0))."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"downside_semivol_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Downside semivolatility over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 3,  # Allow up to 3x window for optimization
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate downside semivolatility
        downside_semivol = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 1:
                # Only negative returns
                negative_returns = np.minimum(valid_returns, 0)
                downside_semivol[i] = np.std(negative_returns, ddof=1)
        
        return pd.Series(downside_semivol, index=data.index)

def create_default_volatility_generators() -> List[FeatureGenerator]:
    return create_volatility_generators()