"""
Streamlined Time Features

Focus on the most important time features:
- Hourly patterns (intraday trading patterns)
- Cyclical encodings (for machine learning compatibility)
- Intraday patterns (market open, close, lunch effects)
"""
import pandas as pd
import numpy as np
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory

# Basic Hour Features
class HourGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour",
            category=FeatureCategory.TIME,
            description="Hour of day (0-23) - captures intraday trading patterns",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(data.index.hour, index=data.index)

# Cyclical Encodings for Machine Learning
class HourSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of hour (cyclical) - ML compatible",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        return pd.Series(np.sin(2 * np.pi * hour / 24), index=data.index)

class HourCosGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of hour (cyclical) - ML compatible",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        return pd.Series(np.cos(2 * np.pi * hour / 24), index=data.index)

# Intraday Pattern Features
class MarketOpenGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_open",
            category=FeatureCategory.TIME,
            description="Market open indicator (first 2 hours of trading)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # Market open: 9-11 AM (assuming 9 AM market open)
        market_open = ((hour >= 9) & (hour < 11)).astype(int)
        return pd.Series(market_open, index=data.index)

class LunchHourGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="lunch_hour",
            category=FeatureCategory.TIME,
            description="Lunch hour indicator (12-2 PM) - reduced activity period",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # Lunch hour: 12-2 PM
        lunch_hour = ((hour >= 12) & (hour < 14)).astype(int)
        return pd.Series(lunch_hour, index=data.index)

class MarketCloseGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="market_close",
            category=FeatureCategory.TIME,
            description="Market close indicator (last 2 hours of trading)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # Market close: 3-5 PM (assuming 5 PM market close)
        market_close = ((hour >= 15) & (hour < 17)).astype(int)
        return pd.Series(market_close, index=data.index)

class AfterHoursGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="after_hours",
            category=FeatureCategory.TIME,
            description="After hours indicator (outside normal trading hours)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # After hours: before 9 AM or after 5 PM
        after_hours = ((hour < 9) | (hour >= 17)).astype(int)
        return pd.Series(after_hours, index=data.index)

class HighActivityHoursGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="high_activity_hours",
            category=FeatureCategory.TIME,
            description="High activity hours (10 AM - 2 PM) - peak trading period",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        # High activity: 10 AM - 2 PM (excluding lunch hour)
        high_activity = ((hour >= 10) & (hour < 12)) | ((hour >= 14) & (hour < 16))
        return pd.Series(high_activity.astype(int), index=data.index)

# Day of Week Cyclical Encoding (important for weekly patterns)
class DayOfWeekSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of day of week (cyclical) - weekly patterns",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        day_of_week = data.index.dayofweek
        return pd.Series(np.sin(2 * np.pi * day_of_week / 7), index=data.index)

class DayOfWeekCosGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of day of week (cyclical) - weekly patterns",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        day_of_week = data.index.dayofweek
        return pd.Series(np.cos(2 * np.pi * day_of_week / 7), index=data.index)

def create_default_time_generators() -> List[FeatureGenerator]:
    """Create streamlined time feature generators focusing on hourly patterns and intraday effects."""
    return [
        # Basic hour features
        HourGenerator(),
        
        # Cyclical encodings (ML compatible)
        HourSinGenerator(),
        HourCosGenerator(),
        DayOfWeekSinGenerator(),
        DayOfWeekCosGenerator(),
        
        # Intraday pattern features
        MarketOpenGenerator(),
        LunchHourGenerator(),
        MarketCloseGenerator(),
        AfterHoursGenerator(),
        HighActivityHoursGenerator(),
    ]