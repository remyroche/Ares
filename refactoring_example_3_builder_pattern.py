
# REFACTORING PATTERN: Builder Pattern
# For complex feature engineering with many steps

class FeatureBuilder:
    """Builder for complex feature engineering pipelines"""
    
    def __init__(self, data):
        self.data = data
        self.features = {}
    
    def add_technical_indicators(self):
        """Add technical indicators"""
        self.features['technical'] = self._compute_technical_indicators()
        return self
    
    def add_market_microstructure(self):
        """Add market microstructure features"""
        self.features['microstructure'] = self._compute_microstructure()
        return self
    
    def add_regime_features(self):
        """Add regime-based features"""
        self.features['regime'] = self._compute_regime_features()
        return self
    
    def build(self):
        """Combine all features"""
        return pd.concat(list(self.features.values()), axis=1)

# Usage:
features = (FeatureBuilder(data)
            .add_technical_indicators()
            .add_market_microstructure()
            .add_regime_features()
            .build())
