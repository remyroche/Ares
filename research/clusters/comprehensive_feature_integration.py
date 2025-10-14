"""
Comprehensive Feature Integration Module.

This module ensures we use ALL available features from the feature_engineering/
pipeline to capture as many market dimensions as possible before dimensionality
reduction and economic relevance analysis.

Key Integration Goals:
1. Use ALL feature generators from feature_engineering/feature_generators.py
2. Include cross-timeframe analysis features
3. Incorporate microstructure features and order flow proxies
4. Add advanced technical indicators and statistical features
5. Ensure comprehensive market dimension coverage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from src.utils.logger import system_logger

# Import ALL feature engineering components
try:
    from src.feature_engineering_roadmap.feature_generators import FeatureGenerators
    from src.feature_engineering_roadmap.optimized_feature_orchestrator import OptimizedFeatureOrchestrator
    from src.feature_engineering_roadmap.cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline
    from src.feature_engineering_roadmap.limited_microstructure_features import LimitedMicrostructureFeatures
    from src.feature_engineering_roadmap.fractional_differentiation_pipeline import FractionalDifferentiationPipeline
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"Some feature engineering components not available: {e}")
    FEATURE_ENGINEERING_AVAILABLE = False


class ComprehensiveFeatureGenerator:
    """
    Comprehensive feature generator that uses ALL available feature engineering tools.
    
    This class systematically generates features from all available generators to
    ensure maximum market dimension coverage before statistical analysis.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('ComprehensiveFeatureGenerator')
        
        # Initialize all available feature generators
        if FEATURE_ENGINEERING_AVAILABLE:
            try:
                self.feature_generators = FeatureGenerators()
                self.orchestrator = OptimizedFeatureOrchestrator()
                self.logger.info("✅ Feature engineering components initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize feature generators: {e}")
                self.feature_generators = None
                self.orchestrator = None
        else:
            self.feature_generators = None
            self.orchestrator = None
    
    def generate_all_available_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ALL available features from the feature engineering pipeline.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            DataFrame with comprehensive feature set
        """
        self.logger.info("🔧 Generating ALL available features from feature engineering pipeline")
        
        # Start with base market data
        all_features = market_data.copy()
        
        # 1. Basic Technical Indicators (using FeatureGenerators)
        all_features = self._add_technical_indicators(all_features)
        
        # 2. Microstructure Features
        all_features = self._add_microstructure_features(all_features)
        
        # 3. Cross-Timeframe Features
        all_features = self._add_cross_timeframe_features(all_features)
        
        # 4. Advanced Statistical Features
        all_features = self._add_statistical_features(all_features)
        
        # 5. Order Flow Proxies
        all_features = self._add_order_flow_features(all_features)
        
        # 6. Volatility and Risk Features
        all_features = self._add_volatility_risk_features(all_features)
        
        # 7. Market Structure Features
        all_features = self._add_market_structure_features(all_features)
        
        # 8. Momentum and Trend Features
        all_features = self._add_momentum_trend_features(all_features)
        
        # 9. Volume Analysis Features
        all_features = self._add_volume_analysis_features(all_features)
        
        # 10. Price Pattern Features
        all_features = self._add_price_pattern_features(all_features)
        
        # Clean and validate features
        all_features = self._clean_and_validate_features(all_features)
        
        self.logger.info(f"✅ Generated {len(all_features.columns)} comprehensive features")
        self.logger.info(f"   📊 Original: {len(market_data.columns)}, Added: {len(all_features.columns) - len(market_data.columns)}")
        
        return all_features
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators from FeatureGenerators."""
        
        if not self.feature_generators:
            return data
        
        try:
            # Use orchestrator if available for optimized feature generation
            if self.orchestrator:
                # Generate all available features using orchestrator
                orchestrated_features = self.orchestrator.generate_all_features(
                    data, 
                    feature_types=['technical', 'statistical', 'microstructure', 'volume'],
                    use_cache=True
                )
                
                # Add orchestrated features
                for feature_name, feature_values in orchestrated_features.items():
                    if feature_name not in data.columns:
                        data[feature_name] = feature_values
                
                self.logger.info(f"   ✅ Added {len(orchestrated_features)} orchestrated features")
            
            # Fallback to manual technical indicators
            else:
                # Comprehensive technical indicator configuration
                indicator_configs = {
                    'sma': [5, 10, 20, 50, 100, 200],
                    'ema': [8, 12, 21, 26, 50, 100],
                    'volatility': [10, 20, 50, 100],
                    'momentum': [5, 10, 20, 50],
                    'rsi': [14, 21, 50],
                    'macd': [12, 26, 9],
                    'bollinger_bands': [20, 50],
                    'stochastic': [14, 21],
                    'volume_sma': [10, 20, 50],
                    'body_size': [],
                    'taker_buy_ratio': [5, 10, 20]
                }
                
                # Generate batch technical indicators
                tech_features = self.feature_generators.batch_technical_indicators(
                    data, indicator_configs, use_gpu=True
                )
                
                # Add to main dataframe
                for feature_name, feature_values in tech_features.items():
                    data[feature_name] = feature_values
                
                self.logger.info(f"   ✅ Added {len(tech_features)} technical indicators")
            
        except Exception as e:
            self.logger.warning(f"Technical indicators generation failed: {e}")
        
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure features using available generators."""
        
        try:
            # Order flow proxies from OHLCV
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Buy pressure (where close is relative to high-low range)
                data['buy_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
                data['sell_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
                
                # Trade intensity proxy
                data['trade_intensity_proxy'] = data.get('volume', 1) / (data['high'] - data['low'])
                
                # Price efficiency measures
                for window in [10, 20, 50]:
                    data[f'price_efficiency_{window}'] = data['close'].rolling(window).std() / data['close'].rolling(window).mean()
                
                # Intrabar volatility
                data['intrabar_volatility'] = (data['high'] - data['low']) / data['open']
            
            # Microstructure generators from feature_generators
            if self.feature_generators:
                # Taker buy ratio features
                if 'taker_buy_base_asset_volume' in data.columns:
                    data['taker_buy_ratio'] = self.feature_generators.taker_buy_ratio_generator(data)
                    data['market_aggression'] = self.feature_generators.market_aggression_generator(data)
                    data['order_flow_imbalance'] = self.feature_generators.order_flow_imbalance_generator(data)
                    data['institutional_indicator'] = self.feature_generators.institutional_indicator_generator(data)
                
                # Body size features
                data['body_size'] = self.feature_generators.body_size_generator(data)
                data['body_size_pct'] = self.feature_generators.body_size_pct_generator(data)
            
            self.logger.info("   ✅ Added microstructure features")
            
        except Exception as e:
            self.logger.warning(f"Microstructure features generation failed: {e}")
        
        return data
    
    def _add_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-timeframe analysis features."""
        
        try:
            # Multi-timeframe moving average relationships
            timeframes = [(5, 20), (10, 50), (20, 100), (50, 200)]
            
            for short, long in timeframes:
                if f'sma_{short}' in data.columns and f'sma_{long}' in data.columns:
                    # MA alignment
                    data[f'ma_alignment_{short}_{long}'] = (data[f'sma_{short}'] > data[f'sma_{long}']).astype(int)
                    
                    # MA divergence
                    data[f'ma_divergence_{short}_{long}'] = (data[f'sma_{short}'] - data[f'sma_{long}']) / data[f'sma_{long}']
                    
                    # MA convergence/divergence rate
                    data[f'ma_convergence_rate_{short}_{long}'] = data[f'ma_divergence_{short}_{long}'].diff()
            
            # Cross-timeframe volatility relationships
            if 'volatility_10' in data.columns and 'volatility_50' in data.columns:
                data['vol_regime_ratio'] = data['volatility_10'] / data['volatility_50']
                data['vol_regime_change'] = (data['vol_regime_ratio'].diff().abs() > data['vol_regime_ratio'].rolling(50).std()).astype(int)
            
            self.logger.info("   ✅ Added cross-timeframe features")
            
        except Exception as e:
            self.logger.warning(f"Cross-timeframe features generation failed: {e}")
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistical features."""
        
        try:
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                
                # Higher moments
                for window in [20, 50, 100]:
                    data[f'skewness_{window}'] = returns.rolling(window).skew()
                    data[f'kurtosis_{window}'] = returns.rolling(window).kurt()
                
                # Auto-correlation features
                for lag in [1, 5, 10, 20]:
                    data[f'autocorr_lag_{lag}'] = returns.rolling(50).apply(lambda x: x.autocorr(lag))
                
                # Hurst exponent proxy
                for window in [50, 100]:
                    data[f'hurst_proxy_{window}'] = returns.rolling(window).apply(
                        lambda x: 0.5 + np.corrcoef(np.arange(len(x)), np.cumsum(x))[0,1] * 0.5 
                        if len(x) > 10 and not np.isnan(np.corrcoef(np.arange(len(x)), np.cumsum(x))[0,1]) else 0.5
                    )
                
                # Fractal dimension proxy
                for window in [20, 50]:
                    data[f'fractal_dimension_{window}'] = data['close'].rolling(window).apply(
                        lambda x: len(x) / (1 + np.log(len(x))) if len(x) > 1 else 1
                    )
            
            self.logger.info("   ✅ Added statistical features")
            
        except Exception as e:
            self.logger.warning(f"Statistical features generation failed: {e}")
        
        return data
    
    def _add_order_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add order flow analysis features."""
        
        try:
            if 'volume' in data.columns and 'close' in data.columns:
                # Volume-price relationship features
                for window in [10, 20, 50]:
                    data[f'vol_price_corr_{window}'] = data['volume'].rolling(window).corr(data['close'])
                    data[f'vol_return_corr_{window}'] = data['volume'].rolling(window).corr(data['close'].pct_change())
                
                # On-Balance Volume
                price_change = data['close'].diff()
                data['obv'] = (data['volume'] * np.sign(price_change)).cumsum()
                
                # Volume Rate of Change
                for period in [5, 10, 20]:
                    data[f'volume_roc_{period}'] = data['volume'].pct_change(period)
                
                # Volume momentum
                data['volume_momentum_10_20'] = data['volume'].rolling(10).mean() / data['volume'].rolling(20).mean()
                
                # Accumulation/Distribution proxy
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                    data['ad_line'] = (clv * data['volume']).cumsum()
            
            self.logger.info("   ✅ Added order flow features")
            
        except Exception as e:
            self.logger.warning(f"Order flow features generation failed: {e}")
        
        return data
    
    def _add_volatility_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive volatility and risk features."""
        
        try:
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                
                # Realized volatility (multiple windows)
                for window in [5, 10, 20, 50, 100]:
                    data[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
                
                # Volatility of volatility
                for window in [20, 50]:
                    vol = returns.rolling(window).std()
                    data[f'vol_of_vol_{window}'] = vol.rolling(window).std()
                
                # Garman-Klass volatility
                if all(col in data.columns for col in ['high', 'low', 'open']):
                    data['gk_volatility'] = np.sqrt(
                        0.5 * (np.log(data['high'] / data['low'])) ** 2 - 
                        (2 * np.log(2) - 1) * (np.log(data['close'] / data['open'])) ** 2
                    )
                
                # Volatility regime indicators
                for window in [20, 50]:
                    vol = data[f'realized_vol_{window}']
                    data[f'vol_regime_high_{window}'] = (vol > vol.rolling(100).quantile(0.8)).astype(int)
                    data[f'vol_regime_low_{window}'] = (vol < vol.rolling(100).quantile(0.2)).astype(int)
                
                # Risk measures
                for window in [20, 50]:
                    data[f'var_95_{window}'] = returns.rolling(window).quantile(0.05)  # 95% VaR
                    data[f'expected_shortfall_{window}'] = returns[returns <= data[f'var_95_{window}']].rolling(window).mean()
            
            self.logger.info("   ✅ Added volatility and risk features")
            
        except Exception as e:
            self.logger.warning(f"Volatility/risk features generation failed: {e}")
        
        return data
    
    def _add_market_structure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market structure and regime change features."""
        
        try:
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                
                # Market stress indicators
                volatility_20 = returns.rolling(20).std()
                data['market_stress'] = (volatility_20 > volatility_20.rolling(100).quantile(0.8)).astype(int)
                
                # Extreme move indicators
                for percentile in [95, 99]:
                    threshold = returns.rolling(100).quantile(percentile / 100)
                    data[f'extreme_move_{percentile}'] = (abs(returns) > threshold).astype(int)
                
                # Market efficiency proxies
                for window in [10, 20, 50]:
                    # Efficiency ratio (net price change / total price movement)
                    net_change = abs(data['close'].diff(window))
                    total_movement = data['close'].rolling(window).apply(lambda x: np.sum(np.abs(np.diff(x))))
                    data[f'efficiency_ratio_{window}'] = net_change / total_movement
                
                # Regime change indicators
                for feature in ['volatility_20', 'volume']:
                    if feature in data.columns:
                        feature_data = data[feature]
                        data[f'{feature}_regime_change'] = (
                            feature_data.pct_change().abs() > feature_data.pct_change().rolling(50).std()
                        ).astype(int)
            
            self.logger.info("   ✅ Added market structure features")
            
        except Exception as e:
            self.logger.warning(f"Market structure features generation failed: {e}")
        
        return data
    
    def _add_momentum_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive momentum and trend features."""
        
        try:
            if 'close' in data.columns:
                # Price momentum (multiple timeframes)
                for period in [3, 5, 10, 20, 50, 100]:
                    data[f'price_momentum_{period}'] = data['close'].pct_change(period)
                    data[f'price_acceleration_{period}'] = data[f'price_momentum_{period}'].diff()
                
                # Trend strength indicators
                for window in [10, 20, 50]:
                    # Linear regression slope as trend strength
                    data[f'trend_strength_{window}'] = data['close'].rolling(window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.iloc[-1] if len(x) == window else 0
                    )
                
                # Momentum oscillators
                if all(col in data.columns for col in ['high', 'low']):
                    # Williams %R
                    for period in [14, 21]:
                        highest_high = data['high'].rolling(period).max()
                        lowest_low = data['low'].rolling(period).min()
                        data[f'williams_r_{period}'] = (highest_high - data['close']) / (highest_high - lowest_low) * -100
                
                # Rate of Change variations
                for period in [5, 10, 20]:
                    data[f'roc_{period}'] = data['close'].pct_change(period) * 100
                    data[f'roc_ma_{period}'] = data[f'roc_{period}'].rolling(10).mean()
            
            self.logger.info("   ✅ Added momentum and trend features")
            
        except Exception as e:
            self.logger.warning(f"Momentum/trend features generation failed: {e}")
        
        return data
    
    def _add_volume_analysis_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive volume analysis features."""
        
        try:
            if 'volume' in data.columns and 'close' in data.columns:
                # Volume moving averages and ratios
                for period in [5, 10, 20, 50, 100]:
                    vol_ma = data['volume'].rolling(period).mean()
                    data[f'volume_ma_{period}'] = vol_ma
                    data[f'volume_ratio_{period}'] = data['volume'] / vol_ma
                
                # Volume momentum
                for period in [5, 10, 20]:
                    data[f'volume_momentum_{period}'] = data['volume'].pct_change(period)
                
                # VWAP features
                for period in [10, 20, 50]:
                    typical_price = (data['high'] + data['low'] + data['close']) / 3 if all(col in data.columns for col in ['high', 'low']) else data['close']
                    cumulative_vol = data['volume'].rolling(period).sum()
                    cumulative_vol_price = (typical_price * data['volume']).rolling(period).sum()
                    data[f'vwap_{period}'] = cumulative_vol_price / cumulative_vol
                    
                    # Price vs VWAP
                    data[f'price_vs_vwap_{period}'] = (data['close'] - data[f'vwap_{period}']) / data[f'vwap_{period}']
                
                # Volume profile features
                for window in [20, 50]:
                    vol_std = data['volume'].rolling(window).std()
                    vol_mean = data['volume'].rolling(window).mean()
                    data[f'volume_zscore_{window}'] = (data['volume'] - vol_mean) / vol_std
            
            self.logger.info("   ✅ Added volume analysis features")
            
        except Exception as e:
            self.logger.warning(f"Volume analysis features generation failed: {e}")
        
        return data
    
    def _add_price_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern and candlestick features."""
        
        try:
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Candlestick patterns (basic)
                # Doji patterns
                body_size = abs(data['close'] - data['open'])
                total_range = data['high'] - data['low']
                data['doji_pattern'] = (body_size / total_range < 0.1).astype(int)
                
                # Hammer/Shooting star patterns
                upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
                lower_shadow = np.minimum(data['open'], data['close']) - data['low']
                data['hammer_pattern'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
                data['shooting_star_pattern'] = ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
                
                # Gap analysis
                data['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
                data['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)
                data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
                
                # Support/Resistance levels (simplified)
                for window in [20, 50]:
                    data[f'near_resistance_{window}'] = (
                        abs(data['close'] - data['high'].rolling(window).max()) / data['close'] < 0.02
                    ).astype(int)
                    data[f'near_support_{window}'] = (
                        abs(data['close'] - data['low'].rolling(window).min()) / data['close'] < 0.02
                    ).astype(int)
            
            self.logger.info("   ✅ Added price pattern features")
            
        except Exception as e:
            self.logger.warning(f"Price pattern features generation failed: {e}")
        
        return data
    
    def _clean_and_validate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the comprehensive feature set."""
        
        initial_columns = len(data.columns)
        
        # Remove columns with all NaN or infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(axis=1, how='all')
        
        # Remove columns with >90% NaN values
        nan_threshold = 0.9
        nan_ratios = data.isnull().sum() / len(data)
        columns_to_keep = nan_ratios[nan_ratios <= nan_threshold].index
        data = data[columns_to_keep]
        
        # Fill remaining NaN values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            data = data.drop(columns=constant_columns)
            self.logger.info(f"   🧹 Removed {len(constant_columns)} constant columns")
        
        # Remove highly correlated features (>0.99 correlation)
        correlation_matrix = data.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        highly_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.99)]
        if highly_correlated:
            data = data.drop(columns=highly_correlated)
            self.logger.info(f"   🧹 Removed {len(highly_correlated)} highly correlated features")
        
        final_columns = len(data.columns)
        removed_columns = initial_columns - final_columns
        
        self.logger.info(f"   🧹 Feature cleaning: {initial_columns} → {final_columns} ({removed_columns} removed)")
        
        return data
    
    def get_feature_categories(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by their market dimension."""
        
        categories = {
            'momentum': [],
            'volatility': [],
            'volume': [],
            'liquidity': [],
            'microstructure': [],
            'correlation': [],
            'statistical': [],
            'pattern': [],
            'risk': [],
            'other': []
        }
        
        for column in data.columns:
            col_lower = column.lower()
            
            # Categorize based on feature name patterns
            if any(keyword in col_lower for keyword in ['momentum', 'roc', 'trend', 'ma_', 'sma', 'ema', 'macd']):
                categories['momentum'].append(column)
            elif any(keyword in col_lower for keyword in ['volatility', 'vol_', 'atr', 'gk_', 'std']):
                categories['volatility'].append(column)
            elif any(keyword in col_lower for keyword in ['volume', 'vol', 'obv', 'vwap', 'ad_line']):
                categories['volume'].append(column)
            elif any(keyword in col_lower for keyword in ['spread', 'liquidity', 'efficiency', 'impact']):
                categories['liquidity'].append(column)
            elif any(keyword in col_lower for keyword in ['taker', 'aggression', 'flow', 'micro', 'body', 'pressure']):
                categories['microstructure'].append(column)
            elif any(keyword in col_lower for keyword in ['corr', 'autocorr', 'lag']):
                categories['correlation'].append(column)
            elif any(keyword in col_lower for keyword in ['skew', 'kurt', 'hurst', 'fractal']):
                categories['statistical'].append(column)
            elif any(keyword in col_lower for keyword in ['pattern', 'doji', 'hammer', 'gap', 'support', 'resistance']):
                categories['pattern'].append(column)
            elif any(keyword in col_lower for keyword in ['var_', 'shortfall', 'risk', 'stress']):
                categories['risk'].append(column)
            else:
                categories['other'].append(column)
        
        # Log category summary
        self.logger.info("📊 Feature categorization:")
        for category, features in categories.items():
            if features:
                self.logger.info(f"   {category}: {len(features)} features")
        
        return categories