import pandas as pd
from typing import Any, Dict, Callable, List

class FeatureGenerator:
    def __init__(self, custom_features: List[Callable[[pd.DataFrame], pd.DataFrame]] = None):
        self.feature_functions = [
            self.price_features,
            self.moving_averages,
            self.volatility_features,
            self.volume_features,
            self.technical_indicators,
        ]
        if custom_features:
            self.feature_functions.extend(custom_features)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)
        for func in self.feature_functions:
            feat = func(data)
            features = features.join(feat, how='outer')
        features = features.fillna(0)  # Default, can be replaced by handle_missing_data
        return features

    def generate_labels(self, data: pd.DataFrame) -> pd.Series:
        # Example: simple trend-following label
        labels = (data['close'].shift(-1) > data['close']).astype(int)
        return labels.fillna(0)

    def price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'price_change': data['close'].pct_change(),
            'high_low_ratio': data['high'] / data['low'],
            'open_close_ratio': data['open'] / data['close'],
        }, index=data.index)

    def moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'ma_5': data['close'].rolling(5).mean(),
            'ma_10': data['close'].rolling(10).mean(),
            'ma_20': data['close'].rolling(20).mean(),
        }, index=data.index)

    def volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'volatility_5': data['close'].rolling(5).std(),
            'volatility_10': data['close'].rolling(10).std(),
        }, index=data.index)

    def volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        vol_ma_5 = data['volume'].rolling(5).mean()
        return pd.DataFrame({
            'volume_ma_5': vol_ma_5,
            'volume_ratio': data['volume'] / vol_ma_5,
        }, index=data.index)

    def technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'rsi': self._calculate_rsi(data['close']),
            'macd': self._calculate_macd(data['close']),
        }, index=data.index)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd