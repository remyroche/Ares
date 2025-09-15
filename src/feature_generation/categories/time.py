"""Time features"""
import pandas as pd
import numpy as np
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory

class HourGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour",
            category=FeatureCategory.TIME,
            description="Hour of day (0-23)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(data.index.hour, index=data.index)

class DayOfWeekGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week",
            category=FeatureCategory.TIME,
            description="Day of week (0=Monday, 6=Sunday)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(data.index.dayofweek, index=data.index)

class MonthGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="month",
            category=FeatureCategory.TIME,
            description="Month (1-12)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(data.index.month, index=data.index)

class QuarterGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="quarter",
            category=FeatureCategory.TIME,
            description="Quarter (1-4)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series(data.index.quarter, index=data.index)

class HourSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="hour_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of hour (cyclical)",
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
            description="Cosine transformation of hour (cyclical)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        hour = data.index.hour
        return pd.Series(np.cos(2 * np.pi * hour / 24), index=data.index)

class DayOfWeekSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="day_of_week_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of day of week (cyclical)",
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
            description="Cosine transformation of day of week (cyclical)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        day_of_week = data.index.dayofweek
        return pd.Series(np.cos(2 * np.pi * day_of_week / 7), index=data.index)

class MonthSinGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="month_sin",
            category=FeatureCategory.TIME,
            description="Sine transformation of month (cyclical)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        month = data.index.month
        return pd.Series(np.sin(2 * np.pi * month / 12), index=data.index)

class MonthCosGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="month_cos",
            category=FeatureCategory.TIME,
            description="Cosine transformation of month (cyclical)",
            required_columns=[],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        month = data.index.month
        return pd.Series(np.cos(2 * np.pi * month / 12), index=data.index)

def create_default_time_generators() -> List[FeatureGenerator]:
    generators = []
    
    # Basic time features
    generators.extend([
        HourGenerator(),
        DayOfWeekGenerator(),
        MonthGenerator(),
        QuarterGenerator(),
    ])
    
    # Cyclical time features
    generators.extend([
        HourSinGenerator(),
        HourCosGenerator(),
        DayOfWeekSinGenerator(),
        DayOfWeekCosGenerator(),
        MonthSinGenerator(),
        MonthCosGenerator(),
    ])
    
    return generators