# src/analyst/advanced_feature_engineering.py

"""
Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features, regime detection,
and adaptive indicators for improved prediction accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import savgol_filter

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering with market microstructure analysis,
    regime detection, and adaptive indicators for improved performance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AdvancedFeatureEngineering")
        
        # Configuration
        self.feature_config = config.get("advanced_features", {})
        self.enable_volatility_modeling = self.feature_config.get("enable_volatility_regime_modeling", True)
        self.enable_correlation_analysis = self.feature_config.get("enable_correlation_analysis", True)
        self.enable_momentum_analysis = self.feature_config.get("enable_momentum_analysis", True)
        self.enable_liquidity_analysis = self.feature_config.get("enable_liquidity_analysis", True)
        
        # Initialize components
        self.volatility_model = None
        self.correlation_analyzer = None
        self.momentum_analyzer = None
        self.liquidity_analyzer = None
        
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="advanced feature engineering initialization",
    )
    async def initialize(self) -> bool:
        """Initialize advanced feature engineering components."""
        try:
            self.logger.info("🚀 Initializing advanced feature engineering...")
            
            # Initialize volatility modeling
            if self.enable_volatility_modeling:
                self.volatility_model = VolatilityRegimeModel(self.config)
                await self.volatility_model.initialize()
            
            # Initialize correlation analysis
            if self.enable_correlation_analysis:
                self.correlation_analyzer = CorrelationAnalyzer(self.config)
                await self.correlation_analyzer.initialize()
            
            # Initialize momentum analysis
            if self.enable_momentum_analysis:
                self.momentum_analyzer = MomentumAnalyzer(self.config)
                await self.momentum_analyzer.initialize()
            
            # Initialize liquidity analysis
            if self.enable_liquidity_analysis:
                self.liquidity_analyzer = LiquidityAnalyzer(self.config)
                await self.liquidity_analyzer.initialize()
            
            self.is_initialized = True
            self.logger.info("✅ Advanced feature engineering initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing advanced feature engineering: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="advanced feature engineering",
    )
    async def engineer_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Engineer advanced features for improved prediction accuracy.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            
        Returns:
            Dictionary containing engineered features
        """
        try:
            if not self.is_initialized:
                self.logger.error("Advanced feature engineering not initialized")
                return {}
            
            features = {}
            
            # Market microstructure features
            microstructure_features = await self._engineer_microstructure_features(
                price_data, volume_data, order_flow_data
            )
            features.update(microstructure_features)
            
            # Volatility regime features
            if self.volatility_model:
                volatility_features = await self.volatility_model.model_volatility(price_data)
                features.update(volatility_features)
            
            # Correlation analysis features
            if self.correlation_analyzer:
                correlation_features = await self.correlation_analyzer.analyze_correlations(price_data)
                features.update(correlation_features)
            
            # Momentum analysis features
            if self.momentum_analyzer:
                momentum_features = await self.momentum_analyzer.analyze_momentum(price_data)
                features.update(momentum_features)
            
            # Liquidity analysis features
            if self.liquidity_analyzer:
                liquidity_features = await self.liquidity_analyzer.analyze_liquidity(
                    price_data, volume_data, order_flow_data
                )
                features.update(liquidity_features)
            
            # Adaptive indicators
            adaptive_features = self._engineer_adaptive_indicators(price_data)
            features.update(adaptive_features)
            
            # Feature selection and dimensionality reduction
            selected_features = self._select_optimal_features(features)
            
            self.logger.info(f"✅ Engineered {len(selected_features)} advanced features")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error engineering advanced features: {e}")
            return {}

    async def _engineer_microstructure_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Engineer market microstructure features."""
        try:
            features = {}
            
            # Price impact analysis
            features.update(self._calculate_price_impact(price_data, volume_data))
            
            # Order flow imbalance
            if order_flow_data is not None:
                features.update(self._calculate_order_flow_imbalance(order_flow_data))
            
            # Volume profile analysis
            features.update(self._calculate_volume_profile(price_data, volume_data))
            
            # Bid-ask spread analysis
            if order_flow_data is not None:
                features.update(self._calculate_spread_analysis(order_flow_data))
            
            # Market depth analysis
            if order_flow_data is not None:
                features.update(self._calculate_market_depth(order_flow_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering microstructure features: {e}")
            return {}

    def _calculate_price_impact(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price impact metrics."""
        try:
            # Calculate price changes
            price_changes = price_data['close'].pct_change()
            
            # Calculate volume-weighted price impact
            volume_weighted_impact = (price_changes * volume_data['volume']).rolling(20).mean()
            
            # Calculate Kyle's lambda (price impact parameter)
            kyle_lambda = np.abs(price_changes).rolling(50).mean() / volume_data['volume'].rolling(50).mean()
            
            # Calculate Amihud illiquidity measure
            amihud_illiquidity = np.abs(price_changes) / volume_data['volume']
            amihud_illiquidity = amihud_illiquidity.rolling(20).mean()
            
            return {
                'price_impact': volume_weighted_impact.iloc[-1] if not volume_weighted_impact.empty else 0.0,
                'kyle_lambda': kyle_lambda.iloc[-1] if not kyle_lambda.empty else 0.0,
                'amihud_illiquidity': amihud_illiquidity.iloc[-1] if not amihud_illiquidity.empty else 0.0,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return {}

    def _calculate_order_flow_imbalance(self, order_flow_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate order flow imbalance metrics."""
        try:
            # Calculate buy/sell pressure
            buy_volume = order_flow_data.get('buy_volume', pd.Series(0))
            sell_volume = order_flow_data.get('sell_volume', pd.Series(0))
            
            # Order flow imbalance
            total_volume = buy_volume + sell_volume
            imbalance = (buy_volume - sell_volume) / total_volume
            imbalance = imbalance.rolling(20).mean()
            
            # Large order detection
            avg_volume = total_volume.rolling(50).mean()
            large_order_ratio = (total_volume > 2 * avg_volume).rolling(20).mean()
            
            return {
                'order_flow_imbalance': imbalance.iloc[-1] if not imbalance.empty else 0.0,
                'large_order_ratio': large_order_ratio.iloc[-1] if not large_order_ratio.empty else 0.0,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {e}")
            return {}

    def _calculate_volume_profile(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile metrics."""
        try:
            # Volume-weighted average price (VWAP)
            vwap = (price_data['close'] * volume_data['volume']).rolling(20).sum() / volume_data['volume'].rolling(20).sum()
            
            # Volume price trend (VPT)
            vpt = (volume_data['volume'] * price_data['close'].pct_change()).cumsum()
            
            # Volume rate of change
            volume_roc = volume_data['volume'].pct_change(5)
            
            # Volume moving average ratio
            volume_ma_ratio = volume_data['volume'] / volume_data['volume'].rolling(20).mean()
            
            return {
                'vwap': vwap.iloc[-1] if not vwap.empty else price_data['close'].iloc[-1],
                'vpt': vpt.iloc[-1] if not vpt.empty else 0.0,
                'volume_roc': volume_roc.iloc[-1] if not volume_roc.empty else 0.0,
                'volume_ma_ratio': volume_ma_ratio.iloc[-1] if not volume_ma_ratio.empty else 1.0,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}

    def _engineer_adaptive_indicators(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Engineer adaptive technical indicators."""
        try:
            features = {}
            
            # Adaptive moving averages
            features.update(self._calculate_adaptive_moving_averages(price_data))
            
            # Adaptive RSI
            features.update(self._calculate_adaptive_rsi(price_data))
            
            # Adaptive Bollinger Bands
            features.update(self._calculate_adaptive_bollinger_bands(price_data))
            
            # Adaptive MACD
            features.update(self._calculate_adaptive_macd(price_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering adaptive indicators: {e}")
            return {}

    def _calculate_adaptive_moving_averages(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate adaptive moving averages based on volatility."""
        try:
            # Calculate volatility
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(20).std()
            
            # Adaptive periods based on volatility
            base_period = 20
            volatility_factor = volatility / volatility.rolling(100).mean()
            adaptive_period = (base_period * volatility_factor).clip(5, 50)
            
            # Adaptive SMA
            adaptive_sma = price_data['close'].rolling(window=adaptive_period.astype(int)).mean()
            
            # Adaptive EMA
            adaptive_alpha = 2 / (adaptive_period + 1)
            adaptive_ema = price_data['close'].ewm(alpha=adaptive_alpha).mean()
            
            return {
                'adaptive_sma': adaptive_sma.iloc[-1] if not adaptive_sma.empty else price_data['close'].iloc[-1],
                'adaptive_ema': adaptive_ema.iloc[-1] if not adaptive_ema.empty else price_data['close'].iloc[-1],
                'adaptive_period': adaptive_period.iloc[-1] if not adaptive_period.empty else base_period,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive moving averages: {e}")
            return {}

    def _select_optimal_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Select optimal features using feature importance and correlation analysis."""
        try:
            # Convert to DataFrame for analysis
            feature_df = pd.DataFrame([features])
            
            # Remove NaN values
            feature_df = feature_df.dropna(axis=1)
            
            # Remove constant features
            feature_df = feature_df.loc[:, feature_df.std() > 0]
            
            # Remove highly correlated features
            if len(feature_df.columns) > 1:
                correlation_matrix = feature_df.corr()
                upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                high_correlation = np.abs(correlation_matrix) > 0.95
                high_correlation = high_correlation & upper_triangle
                
                to_drop = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        if high_correlation.iloc[i, j]:
                            to_drop.append(correlation_matrix.columns[j])
                
                feature_df = feature_df.drop(columns=list(set(to_drop)))
            
            return feature_df.iloc[0].to_dict()
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal features: {e}")
            return features


class VolatilityRegimeModel:
    """Model volatility regimes using GARCH and other methods."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VolatilityRegimeModel")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize volatility model."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing volatility model: {e}")
            return False

    async def model_volatility(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Model volatility regimes."""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # Calculate various volatility measures
            realized_vol = returns.rolling(20).std()
            parkinson_vol = self._calculate_parkinson_volatility(price_data)
            garman_klass_vol = self._calculate_garman_klass_volatility(price_data)
            
            # Volatility regime classification
            vol_percentile = realized_vol.rank(pct=True).iloc[-1]
            
            if vol_percentile > 0.8:
                vol_regime = "high"
            elif vol_percentile < 0.2:
                vol_regime = "low"
            else:
                vol_regime = "medium"
            
            return {
                'realized_volatility': realized_vol.iloc[-1] if not realized_vol.empty else 0.0,
                'parkinson_volatility': parkinson_vol.iloc[-1] if not parkinson_vol.empty else 0.0,
                'garman_klass_volatility': garman_klass_vol.iloc[-1] if not garman_klass_vol.empty else 0.0,
                'volatility_regime': vol_regime,
                'volatility_percentile': vol_percentile,
            }
            
        except Exception as e:
            self.logger.error(f"Error modeling volatility: {e}")
            return {}

    def _calculate_parkinson_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        try:
            high_low_ratio = np.log(price_data['high'] / price_data['low']) ** 2
            parkinson_vol = np.sqrt(high_low_ratio / (4 * np.log(2)))
            return parkinson_vol.rolling(20).mean()
        except Exception:
            return pd.Series()

    def _calculate_garman_klass_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        try:
            c = np.log(price_data['close'] / price_data['close'].shift(1))
            h = np.log(price_data['high'] / price_data['close'].shift(1))
            l = np.log(price_data['low'] / price_data['close'].shift(1))
            
            gk_vol = np.sqrt(0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c ** 2)
            return gk_vol.rolling(20).mean()
        except Exception:
            return pd.Series()


class CorrelationAnalyzer:
    """Analyze correlations between different assets and timeframes."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("CorrelationAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize correlation analyzer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing correlation analyzer: {e}")
            return False

    async def analyze_correlations(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations."""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # Rolling correlations
            corr_5 = returns.rolling(5).corr(returns.shift(1))
            corr_20 = returns.rolling(20).corr(returns.shift(1))
            
            # Cross-timeframe correlations
            returns_5m = returns.resample('5T').last()
            returns_1h = returns.resample('1H').last()
            
            cross_corr = returns_5m.corr(returns_1h) if len(returns_5m) > 1 and len(returns_1h) > 1 else 0.0
            
            return {
                'autocorrelation_5': corr_5.iloc[-1] if not corr_5.empty else 0.0,
                'autocorrelation_20': corr_20.iloc[-1] if not corr_20.empty else 0.0,
                'cross_timeframe_correlation': cross_corr,
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {}


class MomentumAnalyzer:
    """Analyze momentum patterns and signals."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("MomentumAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize momentum analyzer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing momentum analyzer: {e}")
            return False

    async def analyze_momentum(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze momentum patterns."""
        try:
            returns = price_data['close'].pct_change().dropna()
            
            # Momentum indicators
            momentum_5 = returns.rolling(5).mean()
            momentum_20 = returns.rolling(20).mean()
            momentum_50 = returns.rolling(50).mean()
            
            # Momentum acceleration
            momentum_accel = momentum_5 - momentum_20
            
            # Momentum strength
            momentum_strength = momentum_5 / momentum_20.std()
            
            # Momentum divergence
            price_momentum = price_data['close'].pct_change(5)
            volume_momentum = price_data['volume'].pct_change(5) if 'volume' in price_data.columns else pd.Series(0)
            momentum_divergence = price_momentum - volume_momentum
            
            return {
                'momentum_5': momentum_5.iloc[-1] if not momentum_5.empty else 0.0,
                'momentum_20': momentum_20.iloc[-1] if not momentum_20.empty else 0.0,
                'momentum_50': momentum_50.iloc[-1] if not momentum_50.empty else 0.0,
                'momentum_acceleration': momentum_accel.iloc[-1] if not momentum_accel.empty else 0.0,
                'momentum_strength': momentum_strength.iloc[-1] if not momentum_strength.empty else 0.0,
                'momentum_divergence': momentum_divergence.iloc[-1] if not momentum_divergence.empty else 0.0,
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {}


class LiquidityAnalyzer:
    """Analyze liquidity conditions and market depth."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("LiquidityAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize liquidity analyzer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing liquidity analyzer: {e}")
            return False

    async def analyze_liquidity(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Analyze liquidity conditions."""
        try:
            # Volume-based liquidity measures
            avg_volume = volume_data['volume'].rolling(20).mean()
            volume_liquidity = volume_data['volume'] / avg_volume
            
            # Price-based liquidity measures
            price_changes = price_data['close'].pct_change()
            price_impact = np.abs(price_changes) / volume_data['volume']
            price_impact = price_impact.rolling(20).mean()
            
            # Spread-based liquidity (if order flow data available)
            spread_liquidity = 0.0
            if order_flow_data is not None and 'spread' in order_flow_data.columns:
                spread_liquidity = order_flow_data['spread'].rolling(20).mean().iloc[-1]
            
            # Liquidity regime classification
            liquidity_percentile = volume_liquidity.rank(pct=True).iloc[-1]
            
            if liquidity_percentile > 0.8:
                liquidity_regime = "high"
            elif liquidity_percentile < 0.2:
                liquidity_regime = "low"
            else:
                liquidity_regime = "medium"
            
            return {
                'volume_liquidity': volume_liquidity.iloc[-1] if not volume_liquidity.empty else 1.0,
                'price_impact': price_impact.iloc[-1] if not price_impact.empty else 0.0,
                'spread_liquidity': spread_liquidity,
                'liquidity_regime': liquidity_regime,
                'liquidity_percentile': liquidity_percentile,
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {e}")
            return {}
