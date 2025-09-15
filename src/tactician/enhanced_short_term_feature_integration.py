"""
Enhanced Short-Term Feature Integration

This module integrates the existing feature_engineering/ pipeline with the enhanced
short-term entry timing model, providing comprehensive feature engineering capabilities.

Key Features:
- Integration with existing feature_engineering/ utilities
- Enhanced features for short-term prediction
- Comprehensive feature selection and optimization
- Performance optimization and caching
- M1 hardware optimization support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates, traced

# Import existing feature engineering utilities
from src.feature_engineering.step06_enhanced_feature_engineering import EnhancedFeatureEngineering
from src.feature_engineering.optimized_cross_timeframe_analysis import OptimizedCrossTimeframeAnalysisPipeline
from src.feature_engineering.fractional_differentiation_pipeline import FractionalDifferentiationPipeline
from src.feature_engineering.step06_utility_container import Step06UtilityContainer, UtilityConfig
from src.feature_engineering.limited_microstructure_features import LimitedMicrostructureFeatures

logger = system_logger.getChild('EnhancedShortTermFeatureIntegration')


@dataclass
class EnhancedFeatureConfig:
    """Configuration for enhanced feature integration."""
    
    # Existing feature engineering
    enable_existing_features: bool = True
    existing_feature_types: List[str] = field(default_factory=lambda: [
        "technical_indicators", "cross_timeframe_features", "interaction_features",
        "advanced_features", "microstructure_features"
    ])
    
    # Enhanced features for short-term prediction
    enable_enhanced_features: bool = True
    enhanced_feature_types: List[str] = field(default_factory=lambda: [
        "ultra_short_term_features", "pre_movement_features", "entry_timing_features",
        "risk_assessment_features", "market_microstructure_features"
    ])
    
    # Feature selection and optimization
    enable_feature_selection: bool = True
    feature_selection_method: str = "lasso"
    max_features: int = 200
    correlation_threshold: float = 0.8
    
    # Performance optimization
    enable_caching: bool = True
    enable_m1_optimization: bool = True
    chunk_size: int = 10000
    max_workers: int = 4


class EnhancedShortTermFeatureIntegration:
    """
    Enhanced feature integration for short-term entry timing model.
    
    This class integrates existing feature_engineering/ utilities with enhanced
    features specifically designed for short-term prediction.
    """
    
    def __init__(self, config: Optional[EnhancedFeatureConfig] = None):
        """
        Initialize enhanced feature integration.
        
        Args:
            config: Feature integration configuration
        """
        self.config = config or EnhancedFeatureConfig()
        self.logger = logger.getChild('EnhancedShortTermFeatureIntegration')
        
        # Initialize existing feature engineering components
        self.existing_feature_engine = None
        self.cross_timeframe_analysis = None
        self.fractional_differentiation = None
        self.utility_container = None
        self.microstructure_features = None
        
        # Enhanced feature generators
        self.enhanced_feature_generators = {}
        
        # Feature cache
        self.feature_cache = {}
        
        self.logger.info("🚀 Initializing Enhanced Short-Term Feature Integration")
        self.logger.info(f"📊 Existing features: {self.config.enable_existing_features}")
        self.logger.info(f"🎯 Enhanced features: {self.config.enable_enhanced_features}")
        
    @handles_errors(
        error_handlers={
            ImportError: (False, 'Failed to import feature engineering utilities'),
            AttributeError: (False, 'Missing required feature engineering components'),
            ValueError: (False, 'Invalid feature engineering configuration')
        },
        default_return=False,
        context='feature integration initialization'
    )
    async def initialize(self) -> bool:
        """Initialize all feature engineering components."""
        
        try:
            self.logger.info("🔄 Initializing feature engineering components...")
            
            # Initialize utility container
            if self.config.enable_existing_features:
                utility_config = UtilityConfig(
                    enable_common_operations=True,
                    enable_data_processing=True,
                    enable_math_validation=True,
                    enable_m1_gpu=self.config.enable_m1_optimization,
                    enable_m1_memory=self.config.enable_m1_optimization,
                    enable_m1_cpu=self.config.enable_m1_optimization,
                    data_processing_chunk_size=self.config.chunk_size,
                    m1_max_workers=self.config.max_workers
                )
                
                self.utility_container = await Step06UtilityContainer.create(utility_config)
                
                # Initialize existing feature engineering
                self.existing_feature_engine = EnhancedFeatureEngineering(
                    config={}, utility_config=utility_config
                )
                
                # Initialize cross-timeframe analysis
                self.cross_timeframe_analysis = OptimizedCrossTimeframeAnalysisPipeline(
                    config={
                        'timeframes': ['1m', '5m', '15m', '30m'],
                        'base_timeframe': '1m',
                        'interaction_features': [
                            'correlation', 'momentum', 'volatility', 'volume', 'microstructure'
                        ]
                    }
                )
                
                # Initialize fractional differentiation
                self.fractional_differentiation = FractionalDifferentiationPipeline(
                    config={
                        'max_d': 0.5,
                        'min_d': 0.0,
                        'step': 0.1,
                        'adf_threshold': 0.05
                    }
                )
                
                # Initialize microstructure features
                self.microstructure_features = LimitedMicrostructureFeatures(
                    config={
                        'enable_price_impact': True,
                        'enable_volume_analysis': True,
                        'enable_order_flow': True
                    }
                )
            
            # Initialize enhanced feature generators
            if self.config.enable_enhanced_features:
                self._initialize_enhanced_feature_generators()
            
            self.logger.info("✅ Feature engineering components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize feature engineering components: {e}")
            return False
    
    def _initialize_enhanced_feature_generators(self) -> None:
        """Initialize enhanced feature generators for short-term prediction."""
        
        try:
            # Ultra-short-term feature generator
            self.enhanced_feature_generators['ultra_short_term'] = UltraShortTermFeatureGenerator()
            
            # Pre-movement feature generator
            self.enhanced_feature_generators['pre_movement'] = PreMovementFeatureGenerator()
            
            # Entry timing feature generator
            self.enhanced_feature_generators['entry_timing'] = EntryTimingFeatureGenerator()
            
            # Risk assessment feature generator
            self.enhanced_feature_generators['risk_assessment'] = RiskAssessmentFeatureGenerator()
            
            # Market microstructure feature generator
            self.enhanced_feature_generators['market_microstructure'] = MarketMicrostructureFeatureGenerator()
            
            self.logger.info(f"✅ Initialized {len(self.enhanced_feature_generators)} enhanced feature generators")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize enhanced feature generators: {e}")
    
    @handles_errors(
        error_handlers={
            ValueError: (None, 'Invalid input data for feature generation'),
            KeyError: (None, 'Missing required data columns'),
            IndexError: (None, 'Insufficient data for feature generation')
        },
        default_return=None,
        context='comprehensive feature generation'
    )
    async def generate_comprehensive_features(
        self,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate comprehensive features using both existing and enhanced feature engineering.
        
        Args:
            price_data: OHLCV price data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary containing all generated features
        """
        
        start_time = time.time()
        self.logger.info(f"🔄 Generating comprehensive features for {symbol} ({timeframe})")
        
        try:
            # Validate input data
            if not self._validate_price_data(price_data):
                return None
            
            # Initialize feature dictionary
            all_features = {}
            
            # Generate existing features
            if self.config.enable_existing_features and self.existing_feature_engine:
                existing_features = await self._generate_existing_features(price_data, symbol, timeframe)
                if existing_features:
                    all_features.update(existing_features)
                    self.logger.info(f"📊 Generated {len(existing_features)} existing feature types")
            
            # Generate enhanced features
            if self.config.enable_enhanced_features:
                enhanced_features = await self._generate_enhanced_features(price_data, symbol, timeframe)
                if enhanced_features:
                    all_features.update(enhanced_features)
                    self.logger.info(f"🎯 Generated {len(enhanced_features)} enhanced feature types")
            
            # Apply feature selection if enabled
            if self.config.enable_feature_selection and all_features:
                all_features = self._apply_feature_selection(all_features, price_data)
                self.logger.info("🔍 Applied feature selection")
            
            # Cache features if enabled
            if self.config.enable_caching:
                self._cache_features(symbol, timeframe, all_features)
            
            generation_time = time.time() - start_time
            total_features = sum(features.shape[1] if hasattr(features, 'shape') else len(features) 
                               for features in all_features.values())
            
            self.logger.info(f"✅ Generated {total_features} total features in {generation_time:.3f}s")
            self.logger.info(f"📊 Feature types: {list(all_features.keys())}")
            
            return all_features
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"❌ Feature generation failed after {generation_time:.3f}s: {e}")
            return None
    
    async def _generate_existing_features(
        self, 
        price_data: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Generate features using existing feature_engineering/ utilities."""
        
        try:
            existing_features = {}
            
            # Generate technical indicators
            if "technical_indicators" in self.config.existing_feature_types:
                tech_features = await self.existing_feature_engine.extract_technical_indicators(
                    price_data, symbol, timeframe
                )
                if tech_features is not None:
                    existing_features['technical_indicators'] = tech_features
            
            # Generate cross-timeframe features
            if "cross_timeframe_features" in self.config.existing_feature_types:
                cross_features = await self.cross_timeframe_analysis.analyze_cross_timeframe_interactions(
                    price_data, symbol, timeframe
                )
                if cross_features is not None:
                    existing_features['cross_timeframe_features'] = cross_features
            
            # Generate microstructure features
            if "microstructure_features" in self.config.existing_feature_types:
                micro_features = await self.microstructure_features.extract_microstructure_features(
                    price_data, symbol, timeframe
                )
                if micro_features is not None:
                    existing_features['microstructure_features'] = micro_features
            
            # Generate fractional differentiation features
            if "advanced_features" in self.config.existing_feature_types:
                frac_features = await self.fractional_differentiation.apply_fractional_differentiation(
                    price_data, symbol, timeframe
                )
                if frac_features is not None:
                    existing_features['fractional_differentiation'] = frac_features
            
            return existing_features if existing_features else None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate existing features: {e}")
            return None
    
    async def _generate_enhanced_features(
        self, 
        price_data: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Generate enhanced features for short-term prediction."""
        
        try:
            enhanced_features = {}
            
            # Generate ultra-short-term features
            if "ultra_short_term_features" in self.config.enhanced_feature_types:
                ultra_features = self.enhanced_feature_generators['ultra_short_term'].generate_features(
                    price_data, symbol, timeframe
                )
                if ultra_features is not None:
                    enhanced_features['ultra_short_term_features'] = ultra_features
            
            # Generate pre-movement features
            if "pre_movement_features" in self.config.enhanced_feature_types:
                pre_movement_features = self.enhanced_feature_generators['pre_movement'].generate_features(
                    price_data, symbol, timeframe
                )
                if pre_movement_features is not None:
                    enhanced_features['pre_movement_features'] = pre_movement_features
            
            # Generate entry timing features
            if "entry_timing_features" in self.config.enhanced_feature_types:
                entry_timing_features = self.enhanced_feature_generators['entry_timing'].generate_features(
                    price_data, symbol, timeframe
                )
                if entry_timing_features is not None:
                    enhanced_features['entry_timing_features'] = entry_timing_features
            
            # Generate risk assessment features
            if "risk_assessment_features" in self.config.enhanced_feature_types:
                risk_features = self.enhanced_feature_generators['risk_assessment'].generate_features(
                    price_data, symbol, timeframe
                )
                if risk_features is not None:
                    enhanced_features['risk_assessment_features'] = risk_features
            
            # Generate market microstructure features
            if "market_microstructure_features" in self.config.enhanced_feature_types:
                market_micro_features = self.enhanced_feature_generators['market_microstructure'].generate_features(
                    price_data, symbol, timeframe
                )
                if market_micro_features is not None:
                    enhanced_features['market_microstructure_features'] = market_micro_features
            
            return enhanced_features if enhanced_features else None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate enhanced features: {e}")
            return None
    
    def _apply_feature_selection(self, features: Dict[str, np.ndarray], price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Apply feature selection to reduce dimensionality."""
        
        try:
            # Combine all features
            all_feature_arrays = []
            feature_names = []
            
            for feature_type, feature_array in features.items():
                if hasattr(feature_array, 'shape') and len(feature_array.shape) == 2:
                    all_feature_arrays.append(feature_array)
                    feature_names.extend([f"{feature_type}_{i}" for i in range(feature_array.shape[1])])
            
            if not all_feature_arrays:
                return features
            
            # Combine features
            combined_features = np.column_stack(all_feature_arrays)
            
            # Apply correlation filtering
            if self.config.correlation_threshold < 1.0:
                combined_features = self._remove_correlated_features(
                    combined_features, self.config.correlation_threshold
                )
            
            # Apply feature selection method
            if self.config.feature_selection_method == "lasso":
                selected_features = self._apply_lasso_selection(combined_features, price_data)
            else:
                selected_features = combined_features
            
            # Limit number of features
            if selected_features.shape[1] > self.config.max_features:
                selected_features = selected_features[:, :self.config.max_features]
            
            # Return selected features as single array
            return {'selected_features': selected_features}
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return features
    
    def _remove_correlated_features(self, features: np.ndarray, threshold: float) -> np.ndarray:
        """Remove highly correlated features."""
        
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            
            # Find highly correlated features
            high_corr_pairs = np.where((np.abs(corr_matrix) > threshold) & (corr_matrix != 1.0))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i not in features_to_remove and j not in features_to_remove:
                    features_to_remove.add(j)  # Remove the second feature
            
            # Keep non-correlated features
            keep_indices = [i for i in range(features.shape[1]) if i not in features_to_remove]
            
            return features[:, keep_indices] if keep_indices else features
            
        except Exception as e:
            self.logger.error(f"❌ Correlation filtering failed: {e}")
            return features
    
    def _apply_lasso_selection(self, features: np.ndarray, price_data: pd.DataFrame) -> np.ndarray:
        """Apply LASSO feature selection."""
        
        try:
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler
            
            # Create dummy target (in practice, this would be real targets)
            target = price_data['close'].pct_change().fillna(0).values
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply LASSO
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(features_scaled, target)
            
            # Select features with non-zero coefficients
            selected_indices = np.where(lasso.coef_ != 0)[0]
            
            return features[:, selected_indices] if len(selected_indices) > 0 else features
            
        except Exception as e:
            self.logger.error(f"❌ LASSO selection failed: {e}")
            return features
    
    def _validate_price_data(self, price_data: pd.DataFrame) -> bool:
        """Validate price data format."""
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in price_data.columns:
                self.logger.error(f"❌ Missing required column: {col}")
                return False
        
        if len(price_data) < 100:
            self.logger.error(f"❌ Insufficient data: {len(price_data)} rows")
            return False
        
        return True
    
    def _cache_features(self, symbol: str, timeframe: str, features: Dict[str, np.ndarray]) -> None:
        """Cache generated features."""
        
        try:
            cache_key = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.feature_cache[cache_key] = {
                'features': features,
                'timestamp': datetime.now(),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # Limit cache size
            if len(self.feature_cache) > 100:
                oldest_key = min(self.feature_cache.keys(), 
                               key=lambda k: self.feature_cache[k]['timestamp'])
                del self.feature_cache[oldest_key]
            
        except Exception as e:
            self.logger.error(f"❌ Feature caching failed: {e}")
    
    def get_feature_summary(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get summary of generated features."""
        
        try:
            summary = {
                'total_feature_types': len(features),
                'feature_types': list(features.keys()),
                'total_features': 0,
                'feature_details': {}
            }
            
            for feature_type, feature_array in features.items():
                if hasattr(feature_array, 'shape'):
                    n_features = feature_array.shape[1] if len(feature_array.shape) > 1 else 1
                    summary['total_features'] += n_features
                    summary['feature_details'][feature_type] = {
                        'count': n_features,
                        'shape': feature_array.shape,
                        'dtype': str(feature_array.dtype)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Feature summary generation failed: {e}")
            return {'error': str(e)}


# Enhanced Feature Generators
class UltraShortTermFeatureGenerator:
    """Generator for ultra-short-term features (1-5 minutes)."""
    
    def generate_features(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Generate ultra-short-term features."""
        
        try:
            features = []
            
            # 1-5 minute momentum features
            for period in [1, 2, 3, 5]:
                momentum = price_data['close'].pct_change(period).fillna(0)
                features.append(momentum.values)
            
            # 1-5 minute volatility features
            for period in [1, 2, 3, 5]:
                volatility = price_data['close'].pct_change().rolling(period).std().fillna(0)
                features.append(volatility.values)
            
            # 1-5 minute volume features
            for period in [1, 2, 3, 5]:
                volume_ma = price_data['volume'].rolling(period).mean().fillna(0)
                volume_ratio = price_data['volume'] / volume_ma
                features.append(volume_ratio.fillna(1).values)
            
            return np.column_stack(features) if features else None
            
        except Exception as e:
            logger.error(f"❌ Ultra-short-term feature generation failed: {e}")
            return None


class PreMovementFeatureGenerator:
    """Generator for pre-movement prediction features."""
    
    def generate_features(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Generate pre-movement features."""
        
        try:
            features = []
            
            # Momentum divergence
            price_momentum = price_data['close'].pct_change(5)
            volume_momentum = price_data['volume'].pct_change(5)
            momentum_divergence = price_momentum - volume_momentum
            features.append(momentum_divergence.fillna(0).values)
            
            # Support/resistance levels
            rolling_high = price_data['high'].rolling(20).max()
            rolling_low = price_data['low'].rolling(20).min()
            support_distance = (price_data['close'] - rolling_low) / (rolling_high - rolling_low)
            features.append(support_distance.fillna(0.5).values)
            
            # Volatility clustering
            returns = price_data['close'].pct_change()
            volatility = returns.rolling(5).std()
            volatility_clustering = volatility / volatility.rolling(20).mean()
            features.append(volatility_clustering.fillna(1).values)
            
            # Price acceleration
            price_acceleration = price_data['close'].pct_change().diff()
            features.append(price_acceleration.fillna(0).values)
            
            return np.column_stack(features) if features else None
            
        except Exception as e:
            logger.error(f"❌ Pre-movement feature generation failed: {e}")
            return None


class EntryTimingFeatureGenerator:
    """Generator for entry timing features."""
    
    def generate_features(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Generate entry timing features."""
        
        try:
            features = []
            
            # Time-based features
            n_samples = len(price_data)
            time_of_day = np.sin(2 * np.pi * np.arange(n_samples) / 1440)  # Daily cycle
            features.append(time_of_day)
            
            day_of_week = np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly cycle
            features.append(day_of_week)
            
            # Time since last significant move
            returns = price_data['close'].pct_change()
            significant_moves = np.abs(returns) > returns.std() * 2
            time_since_move = np.zeros(n_samples)
            last_move_idx = 0
            for i in range(n_samples):
                if significant_moves.iloc[i]:
                    last_move_idx = i
                time_since_move[i] = i - last_move_idx
            features.append(time_since_move)
            
            return np.column_stack(features) if features else None
            
        except Exception as e:
            logger.error(f"❌ Entry timing feature generation failed: {e}")
            return None


class RiskAssessmentFeatureGenerator:
    """Generator for risk assessment features."""
    
    def generate_features(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Generate risk assessment features."""
        
        try:
            features = []
            
            # Volatility regime detection
            returns = price_data['close'].pct_change()
            short_vol = returns.rolling(5).std()
            long_vol = returns.rolling(20).std()
            vol_regime = short_vol / long_vol
            features.append(vol_regime.fillna(1).values)
            
            # Volatility momentum
            vol_momentum = vol_regime.pct_change(3)
            features.append(vol_momentum.fillna(0).values)
            
            # Mean reversion tendency
            price_zscore = (price_data['close'] - price_data['close'].rolling(20).mean()) / price_data['close'].rolling(20).std()
            features.append(price_zscore.fillna(0).values)
            
            return np.column_stack(features) if features else None
            
        except Exception as e:
            logger.error(f"❌ Risk assessment feature generation failed: {e}")
            return None


class MarketMicrostructureFeatureGenerator:
    """Generator for market microstructure features."""
    
    def generate_features(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """Generate market microstructure features."""
        
        try:
            features = []
            
            # Price impact features
            price_impact = (price_data['high'] - price_data['low']) / price_data['close']
            features.append(price_impact.fillna(0).values)
            
            # Volume-price relationship
            volume_price_corr = price_data['volume'].rolling(10).corr(price_data['close'].pct_change())
            features.append(volume_price_corr.fillna(0).values)
            
            # Bid-ask spread proxy
            spread_proxy = (price_data['high'] - price_data['low']) / price_data['close']
            features.append(spread_proxy.fillna(0).values)
            
            # Trade size distribution
            trade_size_volatility = price_data['volume'].rolling(5).std() / price_data['volume'].rolling(5).mean()
            features.append(trade_size_volatility.fillna(0).values)
            
            return np.column_stack(features) if features else None
            
        except Exception as e:
            logger.error(f"❌ Market microstructure feature generation failed: {e}")
            return None


# Convenience functions
def create_enhanced_feature_integration(
    enable_existing_features: bool = True,
    enable_enhanced_features: bool = True,
    max_features: int = 200
) -> EnhancedShortTermFeatureIntegration:
    """Create enhanced feature integration instance."""
    
    config = EnhancedFeatureConfig(
        enable_existing_features=enable_existing_features,
        enable_enhanced_features=enable_enhanced_features,
        max_features=max_features
    )
    
    return EnhancedShortTermFeatureIntegration(config)


# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced feature integration
    print("Enhanced Short-Term Feature Integration")
    print("=" * 45)
    
    # Create sample price data
    np.random.seed(42)
    n_samples = 1000
    base_price = 100.0
    
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    price_data = pd.DataFrame(data)
    
    # Create feature integration
    feature_integration = create_enhanced_feature_integration()
    
    print(f"✅ Created feature integration")
    print(f"📊 Existing features: {feature_integration.config.enable_existing_features}")
    print(f"🎯 Enhanced features: {feature_integration.config.enable_enhanced_features}")
    print(f"🔍 Feature selection: {feature_integration.config.enable_feature_selection}")
    
    # Initialize and generate features
    import asyncio
    
    async def main():
        success = await feature_integration.initialize()
        
        if success:
            print("✅ Feature integration initialized successfully")
            
            # Generate features
            features = await feature_integration.generate_comprehensive_features(
                price_data, "BTCUSDT", "1m"
            )
            
            if features:
                summary = feature_integration.get_feature_summary(features)
                print(f"🔮 Generated {summary['total_features']} total features")
                print(f"📊 Feature types: {summary['feature_types']}")
                
                for feature_type, details in summary['feature_details'].items():
                    print(f"   {feature_type}: {details['count']} features")
            else:
                print("❌ Feature generation failed")
        else:
            print("❌ Feature integration initialization failed")
    
    asyncio.run(main())