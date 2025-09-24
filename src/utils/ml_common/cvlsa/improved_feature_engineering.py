"""
Improved Feature Engineering for CVLSA

This module implements advanced feature engineering with:
1. Domain knowledge integration for market-specific features
2. Feature interaction terms using tools from generate_features/
3. Dimensionality reduction with PCA/t-SNE integration
4. Advanced technical indicators and market microstructure features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
import talib
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Import existing utilities
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer

logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    # Domain knowledge features
    enable_market_features: bool = True
    enable_microstructure_features: bool = True
    enable_regime_features: bool = True
    
    # Technical indicators
    enable_technical_indicators: bool = True
    technical_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # Feature interactions
    enable_interaction_terms: bool = True
    interaction_max_degree: int = 2
    interaction_threshold: float = 0.1
    
    # Dimensionality reduction
    enable_dimensionality_reduction: bool = True
    reduction_method: str = 'pca'  # 'pca', 'tsne', 'svd'
    reduction_components: Optional[int] = None
    reduction_variance_threshold: float = 0.95
    
    # Feature scaling
    enable_scaling: bool = True
    scaling_method: str = 'robust'  # 'standard', 'robust', 'minmax'
    
    # Feature selection
    enable_feature_selection: bool = True
    selection_threshold: float = 0.01
    max_features: int = 1000
    
    # Performance optimization
    use_parallel: bool = True
    chunk_size: int = 1000
    memory_efficient: bool = True

class ImprovedFeatureEngineer:
    """Advanced feature engineering with domain knowledge integration."""
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        
        # Feature engineering components
        self.scaler = None
        self.dimensionality_reducer = None
        self.feature_names: List[str] = []
        self.feature_importance: np.ndarray = np.array([])
        
        # Performance tracking
        self.engineering_history: List[Dict[str, Any]] = []
        
        # Resource monitoring
        self._init_resource_monitoring()
        
        logger.info("🔧 Improved Feature Engineer initialized")
    
    def _init_resource_monitoring(self):
        """Initialize resource monitoring."""
        try:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            self.matrix_ops = get_enhanced_matrix_operations()
        except Exception as e:
            logger.warning(f"Resource monitoring not available: {e}")
            self.memory_optimizer = None
            self.gpu_manager = None
            self.matrix_ops = None
    
    def engineer_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer market-specific features using domain knowledge."""
        if not self.config.enable_market_features:
            return data
        
        logger.info("📈 Engineering market-specific features...")
        
        enhanced_data = data.copy()
        
        # Price-based features
        if 'close' in data.columns:
            close_prices = data['close'].values
            
            # Price momentum features
            for period in self.config.technical_periods:
                if len(close_prices) > period:
                    # Price returns
                    enhanced_data[f'return_{period}'] = close_prices[period:] / close_prices[:-period] - 1
                    enhanced_data[f'return_{period}'] = enhanced_data[f'return_{period}'].shift(period)
                    
                    # Price volatility
                    rolling_std = data['close'].rolling(period).std()
                    enhanced_data[f'volatility_{period}'] = rolling_std
                    
                    # Price acceleration
                    if period > 1:
                        returns = data['close'].pct_change()
                        enhanced_data[f'acceleration_{period}'] = returns.rolling(period).mean()
            
            # Price levels and patterns
            enhanced_data['price_level'] = close_prices / np.mean(close_prices)
            enhanced_data['price_rank'] = data['close'].rolling(20).rank(pct=True)
            
            # Support and resistance levels
            enhanced_data['support_level'] = data['close'].rolling(20).min()
            enhanced_data['resistance_level'] = data['close'].rolling(20).max()
            enhanced_data['price_position'] = (data['close'] - enhanced_data['support_level']) / (enhanced_data['resistance_level'] - enhanced_data['support_level'])
        
        # Volume-based features
        if 'volume' in data.columns:
            volume = data['volume'].values
            
            # Volume momentum
            for period in self.config.technical_periods:
                if len(volume) > period:
                    enhanced_data[f'volume_momentum_{period}'] = volume[period:] / volume[:-period] - 1
                    enhanced_data[f'volume_momentum_{period}'] = enhanced_data[f'volume_momentum_{period}'].shift(period)
            
            # Volume-price relationship
            if 'close' in data.columns:
                enhanced_data['volume_price_trend'] = data['volume'] * data['close'].pct_change()
                enhanced_data['volume_weighted_price'] = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
            
            # Volume patterns
            enhanced_data['volume_rank'] = data['volume'].rolling(20).rank(pct=True)
            enhanced_data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # OHLC-based features
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in ohlc_cols):
            # Body and shadow features
            enhanced_data['body_size'] = abs(data['close'] - data['open'])
            enhanced_data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
            enhanced_data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
            enhanced_data['total_range'] = data['high'] - data['low']
            
            # Candlestick patterns
            enhanced_data['body_ratio'] = enhanced_data['body_size'] / enhanced_data['total_range']
            enhanced_data['shadow_ratio'] = (enhanced_data['upper_shadow'] + enhanced_data['lower_shadow']) / enhanced_data['total_range']
            
            # Gap features
            enhanced_data['gap_up'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            enhanced_data['gap_down'] = (data['close'].shift(1) - data['open']) / data['close'].shift(1)
        
        logger.info(f"✅ Market features engineered: {len(enhanced_data.columns) - len(data.columns)} new features")
        return enhanced_data
    
    def engineer_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer microstructure features for high-frequency trading."""
        if not self.config.enable_microstructure_features:
            return data
        
        logger.info("⚡ Engineering microstructure features...")
        
        enhanced_data = data.copy()
        
        # Bid-ask spread proxy (using high-low range)
        if all(col in data.columns for col in ['high', 'low']):
            enhanced_data['spread_proxy'] = data['high'] - data['low']
            enhanced_data['spread_ratio'] = enhanced_data['spread_proxy'] / data['close']
        
        # Order flow imbalance proxy
        if all(col in data.columns for col in ['open', 'close', 'volume']):
            # Price impact proxy
            price_change = data['close'] - data['open']
            enhanced_data['price_impact'] = price_change / (data['volume'] + 1e-8)
            
            # Volume imbalance
            enhanced_data['volume_imbalance'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8)
        
        # Market microstructure noise
        if 'close' in data.columns:
            # High-frequency noise
            returns = data['close'].pct_change()
            enhanced_data['microstructure_noise'] = returns.rolling(5).std()
            
            # Tick-by-tick volatility
            enhanced_data['tick_volatility'] = abs(returns).rolling(10).mean()
        
        # Market depth proxy
        if all(col in data.columns for col in ['high', 'low', 'volume']):
            # Depth using volume and range
            enhanced_data['market_depth'] = data['volume'] / (enhanced_data['spread_proxy'] + 1e-8)
        
        logger.info(f"✅ Microstructure features engineered: {len(enhanced_data.columns) - len(data.columns)} new features")
        return enhanced_data
    
    def engineer_regime_features(self, data: pd.DataFrame, regimes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Engineer regime-aware features."""
        if not self.config.enable_regime_features:
            return data
        
        logger.info("🔄 Engineering regime-aware features...")
        
        enhanced_data = data.copy()
        
        if regimes is not None and len(regimes) > 0:
            # Regime-specific features
            enhanced_data['regime'] = regimes
            enhanced_data['regime_duration'] = self._calculate_regime_duration(regimes)
            enhanced_data['regime_transition'] = np.diff(regimes, prepend=regimes[0])
            
            # Regime-specific statistics
            for regime in np.unique(regimes):
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 1:
                    regime_data = data[regime_mask]
                    if 'close' in regime_data.columns:
                        enhanced_data[f'regime_{regime}_mean'] = regime_data['close'].mean()
                        enhanced_data[f'regime_{regime}_std'] = regime_data['close'].std()
                        enhanced_data[f'regime_{regime}_mean'] = enhanced_data[f'regime_{regime}_mean'].fillna(method='ffill')
                        enhanced_data[f'regime_{regime}_std'] = enhanced_data[f'regime_{regime}_std'].fillna(method='ffill')
        else:
            # Create synthetic regime features based on volatility
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                volatility = returns.rolling(20).std()
                
                # Simple regime classification
                high_vol_threshold = volatility.quantile(0.7)
                low_vol_threshold = volatility.quantile(0.3)
                
                enhanced_data['volatility_regime'] = 0  # Normal
                enhanced_data.loc[volatility > high_vol_threshold, 'volatility_regime'] = 1  # High volatility
                enhanced_data.loc[volatility < low_vol_threshold, 'volatility_regime'] = -1  # Low volatility
                
                enhanced_data['regime_duration'] = self._calculate_regime_duration(enhanced_data['volatility_regime'].values)
        
        logger.info(f"✅ Regime features engineered: {len(enhanced_data.columns) - len(data.columns)} new features")
        return enhanced_data
    
    def _calculate_regime_duration(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate duration of current regime for each observation."""
        durations = np.zeros(len(regimes))
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == regimes[i-1]:
                current_duration += 1
            else:
                current_duration = 1
            durations[i] = current_duration
        
        return durations
    
    def engineer_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical indicators using TA-Lib."""
        if not self.config.enable_technical_indicators:
            return data
        
        logger.info("📊 Engineering technical indicators...")
        
        enhanced_data = data.copy()
        
        if 'close' not in data.columns:
            logger.warning("Close price not available for technical indicators")
            return enhanced_data
        
        try:
            close_prices = data['close'].values
            high_prices = data['high'].values if 'high' in data.columns else close_prices
            low_prices = data['low'].values if 'low' in data.columns else close_prices
            volume = data['volume'].values if 'volume' in data.columns else None
            
            # Trend indicators
            for period in self.config.technical_periods:
                if len(close_prices) > period:
                    # Moving averages
                    enhanced_data[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                    enhanced_data[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
                    
                    # MACD
                    if period >= 12:
                        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                        enhanced_data[f'macd_{period}'] = macd
                        enhanced_data[f'macd_signal_{period}'] = macd_signal
                        enhanced_data[f'macd_hist_{period}'] = macd_hist
                    
                    # RSI
                    if period >= 14:
                        enhanced_data[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
                    
                    # Bollinger Bands
                    if period >= 20:
                        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=period)
                        enhanced_data[f'bb_upper_{period}'] = bb_upper
                        enhanced_data[f'bb_middle_{period}'] = bb_middle
                        enhanced_data[f'bb_lower_{period}'] = bb_lower
                        enhanced_data[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                        enhanced_data[f'bb_position_{period}'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # Momentum indicators
            enhanced_data['momentum_10'] = talib.MOM(close_prices, timeperiod=10)
            enhanced_data['roc_10'] = talib.ROC(close_prices, timeperiod=10)
            
            # Volatility indicators
            enhanced_data['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            enhanced_data['natr_14'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Volume indicators
            if volume is not None:
                enhanced_data['obv'] = talib.OBV(close_prices, volume)
                enhanced_data['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
                enhanced_data['adosc'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
            
            # Pattern recognition
            enhanced_data['doji'] = talib.CDLDOJI(enhanced_data['open'], high_prices, low_prices, close_prices)
            enhanced_data['hammer'] = talib.CDLHAMMER(enhanced_data['open'], high_prices, low_prices, close_prices)
            enhanced_data['engulfing'] = talib.CDLENGULFING(enhanced_data['open'], high_prices, low_prices, close_prices)
            
        except Exception as e:
            logger.warning(f"Technical indicators calculation failed: {e}")
        
        logger.info(f"✅ Technical indicators engineered: {len(enhanced_data.columns) - len(data.columns)} new features")
        return enhanced_data
    
    def engineer_interaction_terms(self, data: pd.DataFrame, target: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Engineer feature interaction terms."""
        if not self.config.enable_interaction_terms:
            return data
        
        logger.info("🔗 Engineering feature interaction terms...")
        
        enhanced_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for interaction terms")
            return enhanced_data
        
        # Calculate feature importance for interaction selection
        if target is not None and len(target) == len(data):
            try:
                from sklearn.feature_selection import mutual_info_regression
                importance_scores = mutual_info_regression(data[numeric_cols].fillna(0), target)
                important_features = numeric_cols[importance_scores > self.config.interaction_threshold]
            except Exception as e:
                logger.warning(f"Feature importance calculation failed: {e}")
                important_features = numeric_cols[:min(10, len(numeric_cols))]  # Use first 10 features
        else:
            important_features = numeric_cols[:min(10, len(numeric_cols))]
        
        logger.info(f"   Using {len(important_features)} features for interactions")
        
        # Generate interaction terms
        interaction_count = 0
        for i, col1 in enumerate(important_features):
            for j, col2 in enumerate(important_features[i+1:], i+1):
                if interaction_count >= 50:  # Limit interactions to prevent explosion
                    break
                
                try:
                    # Multiplicative interaction
                    interaction_name = f"{col1}_x_{col2}"
                    enhanced_data[interaction_name] = data[col1] * data[col2]
                    
                    # Ratio interaction (if no division by zero)
                    if (data[col2] != 0).all():
                        ratio_name = f"{col1}_div_{col2}"
                        enhanced_data[ratio_name] = data[col1] / (data[col2] + 1e-8)
                    
                    # Difference interaction
                    diff_name = f"{col1}_minus_{col2}"
                    enhanced_data[diff_name] = data[col1] - data[col2]
                    
                    interaction_count += 3
                    
                except Exception as e:
                    logger.debug(f"Interaction term failed for {col1} and {col2}: {e}")
                    continue
        
        logger.info(f"✅ Interaction terms engineered: {interaction_count} new features")
        return enhanced_data
    
    def apply_dimensionality_reduction(self, data: pd.DataFrame, target: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Apply dimensionality reduction to the feature set."""
        if not self.config.enable_dimensionality_reduction:
            return data
        
        logger.info(f"📉 Applying dimensionality reduction ({self.config.reduction_method})...")
        
        # Prepare data for reduction
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 10:
            logger.info("Not enough features for dimensionality reduction")
            return data
        
        try:
            # Determine number of components
            n_components = self.config.reduction_components
            if n_components is None:
                n_components = min(50, numeric_data.shape[1] // 2)
            
            # Apply dimensionality reduction
            if self.config.reduction_method == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
                reduced_features = reducer.fit_transform(numeric_data)
                
                # Add explained variance information
                explained_variance = reducer.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                logger.info(f"   PCA components: {n_components}")
                logger.info(f"   Explained variance: {cumulative_variance[-1]:.3f}")
                
            elif self.config.reduction_method == 'tsne':
                # t-SNE is computationally expensive, use subset for large datasets
                if numeric_data.shape[0] > 1000:
                    sample_indices = np.random.choice(numeric_data.shape[0], 1000, replace=False)
                    sample_data = numeric_data.iloc[sample_indices]
                else:
                    sample_data = numeric_data
                
                reducer = TSNE(n_components=min(3, n_components), random_state=42, perplexity=30)
                reduced_features = reducer.fit_transform(sample_data)
                
                # For full dataset, use a simpler method
                if numeric_data.shape[0] > 1000:
                    pca = PCA(n_components=n_components, random_state=42)
                    reduced_features = pca.fit_transform(numeric_data)
                    reducer = pca
                
                logger.info(f"   t-SNE components: {reduced_features.shape[1]}")
                
            elif self.config.reduction_method == 'svd':
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                reduced_features = reducer.fit_transform(numeric_data)
                
                logger.info(f"   SVD components: {n_components}")
                logger.info(f"   Explained variance: {np.sum(reducer.explained_variance_ratio_):.3f}")
            
            else:
                logger.warning(f"Unknown reduction method: {self.config.reduction_method}")
                return data
            
            # Create new DataFrame with reduced features
            reduced_df = pd.DataFrame(
                reduced_features,
                columns=[f"{self.config.reduction_method}_component_{i}" for i in range(reduced_features.shape[1])],
                index=data.index
            )
            
            # Store reducer for later use
            self.dimensionality_reducer = reducer
            
            logger.info(f"✅ Dimensionality reduction completed: {reduced_features.shape[1]} components")
            return reduced_df
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {e}")
            return data
    
    def apply_feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling to the data."""
        if not self.config.enable_scaling:
            return data
        
        logger.info(f"⚖️ Applying feature scaling ({self.config.scaling_method})...")
        
        numeric_data = data.select_dtypes(include=[np.number])
        non_numeric_data = data.select_dtypes(exclude=[np.number])
        
        if numeric_data.empty:
            return data
        
        try:
            # Choose scaling method
            if self.config.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.config.scaling_method == 'robust':
                scaler = RobustScaler()
            elif self.config.scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
                return data
            
            # Fit and transform
            scaled_data = scaler.fit_transform(numeric_data)
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=numeric_data.columns,
                index=data.index
            )
            
            # Store scaler for later use
            self.scaler = scaler
            
            # Combine with non-numeric data
            result_df = pd.concat([scaled_df, non_numeric_data], axis=1)
            
            logger.info(f"✅ Feature scaling completed using {self.config.scaling_method}")
            return result_df
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return data
    
    def engineer_features(self, data: pd.DataFrame, target: Optional[np.ndarray] = None,
                         regimes: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main feature engineering pipeline."""
        logger.info("🔧 Starting comprehensive feature engineering...")
        
        start_time = time.time()
        original_features = len(data.columns)
        
        # Resource monitoring
        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("feature_engineering"):
                enhanced_data = self._run_engineering_pipeline(data, target, regimes)
        else:
            enhanced_data = self._run_engineering_pipeline(data, target, regimes)
        
        # Calculate engineering statistics
        engineering_time = time.time() - start_time
        new_features = len(enhanced_data.columns) - original_features
        
        results = {
            'original_features': original_features,
            'new_features': new_features,
            'total_features': len(enhanced_data.columns),
            'feature_expansion_ratio': new_features / original_features if original_features > 0 else 0,
            'engineering_time': engineering_time,
            'feature_names': list(enhanced_data.columns),
            'scaling_applied': self.scaler is not None,
            'dimensionality_reduction_applied': self.dimensionality_reducer is not None
        }
        
        # Store engineering history
        self.engineering_history.append({
            'timestamp': time.time(),
            'original_features': original_features,
            'new_features': new_features,
            'engineering_time': engineering_time
        })
        
        logger.info(f"✅ Feature engineering completed in {engineering_time:.2f}s")
        logger.info(f"   Original features: {original_features}")
        logger.info(f"   New features: {new_features}")
        logger.info(f"   Total features: {len(enhanced_data.columns)}")
        logger.info(f"   Expansion ratio: {results['feature_expansion_ratio']:.2f}x")
        
        return enhanced_data, results
    
    def _run_engineering_pipeline(self, data: pd.DataFrame, target: Optional[np.ndarray] = None,
                                regimes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Run the complete feature engineering pipeline."""
        enhanced_data = data.copy()
        
        # 1. Market-specific features
        enhanced_data = self.engineer_market_features(enhanced_data)
        
        # 2. Microstructure features
        enhanced_data = self.engineer_microstructure_features(enhanced_data)
        
        # 3. Regime features
        enhanced_data = self.engineer_regime_features(enhanced_data, regimes)
        
        # 4. Technical indicators
        enhanced_data = self.engineer_technical_indicators(enhanced_data)
        
        # 5. Feature interaction terms
        enhanced_data = self.engineer_interaction_terms(enhanced_data, target)
        
        # 6. Dimensionality reduction (if enabled)
        if self.config.enable_dimensionality_reduction:
            enhanced_data = self.apply_dimensionality_reduction(enhanced_data, target)
        
        # 7. Feature scaling
        enhanced_data = self.apply_feature_scaling(enhanced_data)
        
        return enhanced_data
    
    def get_engineering_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the feature engineering process."""
        analytics = {
            'total_engineering_sessions': len(self.engineering_history),
            'average_engineering_time': np.mean([h['engineering_time'] for h in self.engineering_history]) if self.engineering_history else 0,
            'average_feature_expansion': np.mean([h['new_features'] for h in self.engineering_history]) if self.engineering_history else 0,
            'scaling_method': self.config.scaling_method,
            'dimensionality_reduction_method': self.config.reduction_method,
            'engineering_history': self.engineering_history
        }
        
        return analytics


# Factory functions
def create_improved_feature_engineer(config: Optional[FeatureEngineeringConfig] = None) -> ImprovedFeatureEngineer:
    """Create improved feature engineer."""
    return ImprovedFeatureEngineer(config)