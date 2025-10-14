#!/usr/bin/env python3
"""
Comprehensive test for Enhanced Feature Generation with all new capabilities.

This script tests the enhanced feature generation functionality including:
- Extended lookback periods up to 600 minutes respecting base timeframe
- Enhanced interaction feature metadata with source and lookback information
- Feature comparisons between base, VWAP-based, volatility-adjusted, and z-score volume adjusted features
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class GeneratedFeature:
    """Generated feature with enhanced metadata."""
    name: str
    feature_type: str  # 'cross_timeframe', 'interaction', 'no_feature', 'comparison'
    formula: str
    parent_features: List[str]
    feature_series: pd.Series
    utility_score: float
    lookback_period: Optional[int] = None
    creation_method: Optional[str] = None
    base_timeframe_minutes: Optional[int] = None
    source_features: Optional[List[Dict[str, Any]]] = None  # For interaction features
    comparison_type: Optional[str] = None  # 'base', 'vwap', 'volatility_adjusted', 'zscore_volume'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.source_features is None:
            self.source_features = []

class ComprehensiveEnhancedFeatureGenerator:
    """
    Comprehensive enhanced feature generator with all new capabilities.
    
    Features:
    - Extended lookback periods up to 600 minutes respecting base timeframe
    - Enhanced interaction feature metadata with source and lookback information
    - Feature comparisons between base, VWAP-based, volatility-adjusted, and z-score volume adjusted features
    """
    
    def __init__(self, base_timeframe_minutes: int = 15):
        """Initialize the comprehensive enhanced feature generator."""
        self.base_timeframe_minutes = base_timeframe_minutes
        
        # Generate periods up to 600 minutes, respecting base timeframe
        base_periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 600]
        self.cross_timeframe_periods = [p * base_timeframe_minutes for p in base_periods]
        
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cross_timeframe_features_generated': 0,
            'interaction_features_generated': 0,
            'no_features_generated': 0,
            'comparison_features_generated': 0
        }
        
        print(f"✅ Comprehensive Enhanced Feature Generator initialized (base timeframe: {base_timeframe_minutes} minutes)")
        print(f"📊 Cross-timeframe periods: {len(self.cross_timeframe_periods)} periods up to {max(self.cross_timeframe_periods)} minutes")
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None,
        base_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive features with all new capabilities.
        
        Args:
            data: Input OHLCV data
            targets: Optional target series for utility scoring
            base_features: Optional base features for interaction generation
            
        Returns:
            Dictionary with all generated features
        """
        print("🚀 Starting comprehensive enhanced feature generation")
        print(f"📊 Data shape: {data.shape}")
        print(f"⏰ Base timeframe: {self.base_timeframe_minutes} minutes")
        
        start_time = time.time()
        
        try:
            # Initialize result containers
            cross_timeframe_features = []
            interaction_features = []
            no_features = []
            comparison_features = []
            
            # Generate cross-timeframe features
            print("\nStep 1: Generating cross-timeframe features")
            cross_timeframe_features = self._generate_cross_timeframe_features(data, targets)
            print(f"✅ Generated {len(cross_timeframe_features)} cross-timeframe features")
            
            # Generate interaction features
            if base_features is not None:
                print("\nStep 2: Generating interaction features")
                interaction_features = self._generate_interaction_features(base_features, targets)
                print(f"✅ Generated {len(interaction_features)} interaction features")
            
            # Generate no features
            print("\nStep 3: Generating no features")
            no_features = self._generate_no_features(data, targets)
            print(f"✅ Generated {len(no_features)} no features")
            
            # Generate comparison features
            print("\nStep 4: Generating comparison features")
            comparison_features = self._generate_comparison_features(data, targets)
            print(f"✅ Generated {len(comparison_features)} comparison features")
            
            # Combine all features
            all_features = cross_timeframe_features + interaction_features + no_features + comparison_features
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': execution_time,
                'cross_timeframe_features_generated': len(cross_timeframe_features),
                'interaction_features_generated': len(interaction_features),
                'no_features_generated': len(no_features),
                'comparison_features_generated': len(comparison_features)
            })
            
            print(f"✅ Comprehensive feature generation completed in {execution_time:.3f}s")
            print(f"🏆 Total features generated: {len(all_features)}")
            
            return {
                'cross_timeframe_features': cross_timeframe_features,
                'interaction_features': interaction_features,
                'no_features': no_features,
                'comparison_features': comparison_features,
                'all_features': all_features,
                'generation_time': execution_time,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Comprehensive feature generation failed: {e}")
            return {
                'cross_timeframe_features': [],
                'interaction_features': [],
                'no_features': [],
                'comparison_features': [],
                'all_features': [],
                'generation_time': time.time() - start_time,
                'success': False,
                'error_message': str(e)
            }
    
    def _generate_cross_timeframe_features(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate cross-timeframe features with extended periods."""
        features = []
        
        try:
            # Ensure we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in data.columns]
            if not available_cols:
                return features
            
            # Generate features for each timeframe period
            for period in self.cross_timeframe_periods:
                # Skip if period is too large for data
                if period >= len(data) // 4:
                    continue
                
                # Generate different types of cross-timeframe features
                period_features = self._generate_period_cross_timeframe_features(
                    data, period, available_cols, targets
                )
                features.extend(period_features)
            
            # Limit to max features
            if len(features) > 20:
                # Sort by utility score and take top features
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:20]
            
            return features
            
        except Exception as e:
            print(f"❌ Cross-timeframe feature generation failed: {e}")
            return []
    
    def _generate_period_cross_timeframe_features(
        self, 
        data: pd.DataFrame, 
        period: int, 
        available_cols: List[str],
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate cross-timeframe features for a specific period."""
        features = []
        
        try:
            # Price-based cross-timeframe features
            if 'close' in available_cols:
                close = data['close']
                
                # Multi-timeframe momentum
                short_momentum = close.pct_change(period)
                long_momentum = close.pct_change(period * 2)
                
                # Momentum divergence
                momentum_div = short_momentum - long_momentum
                features.append(GeneratedFeature(
                    name=f"momentum_divergence_{period}",
                    feature_type="cross_timeframe",
                    formula=f"pct_change({period}) - pct_change({period * 2})",
                    parent_features=["close"],
                    feature_series=momentum_div,
                    utility_score=self._calculate_utility_score(momentum_div, targets),
                    lookback_period=period,
                    base_timeframe_minutes=self.base_timeframe_minutes,
                    metadata={
                        'timeframe_period': period,
                        'feature_category': 'cross_timeframe',
                        'base_timeframe_minutes': self.base_timeframe_minutes,
                        'period_in_base_units': period // self.base_timeframe_minutes
                    }
                ))
                
                # Multi-timeframe volatility
                short_vol = close.rolling(period).std()
                long_vol = close.rolling(period * 2).std()
                
                # Volatility ratio
                vol_ratio = short_vol / (long_vol + 1e-8)
                features.append(GeneratedFeature(
                    name=f"volatility_ratio_{period}",
                    feature_type="cross_timeframe",
                    formula=f"std({period}) / std({period * 2})",
                    parent_features=["close"],
                    feature_series=vol_ratio,
                    utility_score=self._calculate_utility_score(vol_ratio, targets),
                    lookback_period=period,
                    base_timeframe_minutes=self.base_timeframe_minutes,
                    metadata={
                        'timeframe_period': period,
                        'feature_category': 'cross_timeframe',
                        'base_timeframe_minutes': self.base_timeframe_minutes,
                        'period_in_base_units': period // self.base_timeframe_minutes
                    }
                ))
            
            return features
            
        except Exception as e:
            print(f"Error generating period {period} cross-timeframe features: {e}")
            return []
    
    def _generate_interaction_features(
        self, 
        base_features: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate interaction features with enhanced metadata."""
        features = []
        
        try:
            feature_names = list(base_features.columns)
            creation_methods = ['add', 'subtract', 'multiply', 'divide', 'log', 'sqrt', 'power', 'ratio']
            
            # Generate 2-way interactions
            print("   Generating 2-way interactions")
            for i, feat1 in enumerate(feature_names):
                for j, feat2 in enumerate(feature_names[i+1:], i+1):
                    # Skip if same feature
                    if feat1 == feat2:
                        continue
                    
                    # Generate different types of interactions
                    interaction_features = self._create_feature_interactions(
                        base_features, feat1, feat2, targets, creation_methods
                    )
                    features.extend(interaction_features)
            
            # Limit to max features
            if len(features) > 30:
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:30]
            
            return features
            
        except Exception as e:
            print(f"❌ Interaction feature generation failed: {e}")
            return []
    
    def _create_feature_interactions(
        self, 
        base_features: pd.DataFrame, 
        feat1: str, 
        feat2: str, 
        targets: Optional[pd.Series] = None,
        creation_methods: List[str] = None
    ) -> List[GeneratedFeature]:
        """Create interaction features between two features with enhanced metadata."""
        features = []
        
        if creation_methods is None:
            creation_methods = ['add', 'subtract', 'multiply', 'divide', 'log', 'sqrt', 'power', 'ratio']
        
        try:
            series1 = base_features[feat1]
            series2 = base_features[feat2]
            
            # Generate interactions using different creation methods
            for method in creation_methods:
                try:
                    if method == 'add':
                        interaction_series = series1 + series2
                        formula = f"{feat1} + {feat2}"
                    elif method == 'subtract':
                        interaction_series = series1 - series2
                        formula = f"{feat1} - {feat2}"
                    elif method == 'multiply':
                        interaction_series = series1 * series2
                        formula = f"{feat1} * {feat2}"
                    elif method == 'divide':
                        interaction_series = series1 / (series2 + 1e-8)
                        formula = f"{feat1} / ({feat2} + 1e-8)"
                    elif method == 'log':
                        interaction_series = np.log(np.abs(series1) + 1e-8) * np.log(np.abs(series2) + 1e-8)
                        formula = f"log(|{feat1}|) * log(|{feat2}|)"
                    elif method == 'sqrt':
                        interaction_series = np.sqrt(np.abs(series1)) * np.sqrt(np.abs(series2))
                        formula = f"sqrt(|{feat1}|) * sqrt(|{feat2}|)"
                    elif method == 'power':
                        interaction_series = np.power(np.abs(series1), 0.5) * np.power(np.abs(series2), 0.5)
                        formula = f"pow(|{feat1}|, 0.5) * pow(|{feat2}|, 0.5)"
                    elif method == 'ratio':
                        interaction_series = series1 / (series2 + 1e-8) * series2 / (series1 + 1e-8)
                        formula = f"({feat1} / {feat2}) * ({feat2} / {feat1})"
                    else:
                        continue
                    
                    # Create feature with enhanced metadata
                    feature = GeneratedFeature(
                        name=f"{feat1}_{feat2}_{method}",
                        feature_type="interaction",
                        formula=formula,
                        parent_features=[feat1, feat2],
                        feature_series=interaction_series,
                        utility_score=self._calculate_utility_score(interaction_series, targets),
                        creation_method=method,
                        base_timeframe_minutes=self.base_timeframe_minutes,
                        source_features=[
                            {'name': feat1, 'lookback_period': None, 'feature_type': 'base'},
                            {'name': feat2, 'lookback_period': None, 'feature_type': 'base'}
                        ],
                        metadata={
                            'interaction_order': 2,
                            'feature_category': 'interaction',
                            'base_timeframe_minutes': self.base_timeframe_minutes
                        }
                    )
                    
                    features.append(feature)
                    
                except Exception as e:
                    print(f"Error creating {method} interaction between {feat1} and {feat2}: {e}")
                    continue
            
            return features
            
        except Exception as e:
            print(f"Error creating feature interactions: {e}")
            return []
    
    def _generate_no_features(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate features without lookback optimization."""
        features = []
        
        try:
            # Price-based no features
            if 'close' in data.columns:
                close = data['close']
                
                # Price change
                price_change = close.pct_change()
                features.append(GeneratedFeature(
                    name="price_change",
                    feature_type="no_feature",
                    formula="close.pct_change()",
                    parent_features=["close"],
                    feature_series=price_change,
                    utility_score=self._calculate_utility_score(price_change, targets),
                    creation_method="pct_change",
                    base_timeframe_minutes=self.base_timeframe_minutes,
                    metadata={
                        'feature_category': 'no_feature',
                        'base_timeframe_minutes': self.base_timeframe_minutes
                    }
                ))
                
                # Price log return
                log_return = np.log(close / close.shift(1))
                features.append(GeneratedFeature(
                    name="log_return",
                    feature_type="no_feature",
                    formula="log(close / close.shift(1))",
                    parent_features=["close"],
                    feature_series=log_return,
                    utility_score=self._calculate_utility_score(log_return, targets),
                    creation_method="log",
                    base_timeframe_minutes=self.base_timeframe_minutes,
                    metadata={
                        'feature_category': 'no_feature',
                        'base_timeframe_minutes': self.base_timeframe_minutes
                    }
                ))
            
            # Limit to max features
            if len(features) > 15:
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:15]
            
            return features
            
        except Exception as e:
            print(f"❌ No features generation failed: {e}")
            return []
    
    def _generate_comparison_features(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate comparison features between base, VWAP-based, volatility-adjusted, and z-score volume adjusted features."""
        features = []
        
        try:
            # Ensure we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in data.columns]
            if not available_cols:
                return features
            
            # Generate comparison features for different periods
            periods = [5, 10, 15, 30, 60, 120, 240]  # minutes
            
            for period in periods:
                # Skip if period is too large for data
                if period >= len(data) // 4:
                    continue
                
                # Generate different types of comparison features
                period_features = self._generate_period_comparison_features(
                    data, period, available_cols, targets
                )
                features.extend(period_features)
            
            # Limit to max features
            if len(features) > 20:
                # Sort by utility score and take top features
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:20]
            
            return features
            
        except Exception as e:
            print(f"❌ Comparison feature generation failed: {e}")
            return []
    
    def _generate_period_comparison_features(
        self, 
        data: pd.DataFrame, 
        period: int, 
        available_cols: List[str],
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate comparison features for a specific period."""
        features = []
        
        try:
            if 'close' in available_cols and 'volume' in available_cols:
                close = data['close']
                volume = data['volume']
                
                # Base features
                base_sma = close.rolling(period).mean()
                base_vol = close.rolling(period).std()
                
                # VWAP-based features
                vwap = (close * volume).rolling(period).sum() / volume.rolling(period).sum()
                vwap_sma = vwap.rolling(period).mean()
                vwap_vol = vwap.rolling(period).std()
                
                # Volatility-adjusted features
                vol_adjusted_close = close / (base_vol + 1e-8)
                vol_adjusted_sma = vol_adjusted_close.rolling(period).mean()
                vol_adjusted_vol = vol_adjusted_close.rolling(period).std()
                
                # Z-score volume adjusted features
                volume_zscore = (volume - volume.rolling(period).mean()) / (volume.rolling(period).std() + 1e-8)
                zscore_vol_adjusted_close = close * volume_zscore
                zscore_vol_adjusted_sma = zscore_vol_adjusted_close.rolling(period).mean()
                zscore_vol_adjusted_vol = zscore_vol_adjusted_close.rolling(period).std()
                
                # Generate comparison features
                comparison_types = [
                    ('base', base_sma, base_vol),
                    ('vwap', vwap_sma, vwap_vol),
                    ('volatility_adjusted', vol_adjusted_sma, vol_adjusted_vol),
                    ('zscore_volume', zscore_vol_adjusted_sma, zscore_vol_adjusted_vol)
                ]
                
                # Compare each type with others
                for i, (type1, sma1, vol1) in enumerate(comparison_types):
                    for j, (type2, sma2, vol2) in enumerate(comparison_types[i+1:], i+1):
                        # SMA comparison
                        sma_ratio = sma1 / (sma2 + 1e-8)
                        features.append(GeneratedFeature(
                            name=f"sma_ratio_{type1}_vs_{type2}_{period}",
                            feature_type="comparison",
                            formula=f"sma_{type1}({period}) / sma_{type2}({period})",
                            parent_features=["close", "volume"],
                            feature_series=sma_ratio,
                            utility_score=self._calculate_utility_score(sma_ratio, targets),
                            lookback_period=period,
                            base_timeframe_minutes=self.base_timeframe_minutes,
                            comparison_type=f"{type1}_vs_{type2}",
                            metadata={
                                'feature_category': 'comparison',
                                'comparison_types': [type1, type2],
                                'base_timeframe_minutes': self.base_timeframe_minutes,
                                'period_in_base_units': period // self.base_timeframe_minutes
                            }
                        ))
                        
                        # Volatility comparison
                        vol_ratio = vol1 / (vol2 + 1e-8)
                        features.append(GeneratedFeature(
                            name=f"vol_ratio_{type1}_vs_{type2}_{period}",
                            feature_type="comparison",
                            formula=f"vol_{type1}({period}) / vol_{type2}({period})",
                            parent_features=["close", "volume"],
                            feature_series=vol_ratio,
                            utility_score=self._calculate_utility_score(vol_ratio, targets),
                            lookback_period=period,
                            base_timeframe_minutes=self.base_timeframe_minutes,
                            comparison_type=f"{type1}_vs_{type2}",
                            metadata={
                                'feature_category': 'comparison',
                                'comparison_types': [type1, type2],
                                'base_timeframe_minutes': self.base_timeframe_minutes,
                                'period_in_base_units': period // self.base_timeframe_minutes
                            }
                        ))
            
            return features
            
        except Exception as e:
            print(f"Error generating period {period} comparison features: {e}")
            return []
    
    def _calculate_utility_score(
        self, 
        feature_series: pd.Series, 
        targets: Optional[pd.Series] = None
    ) -> float:
        """Calculate utility score for a feature."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(feature_series.var())
            
            # Align series
            aligned_feature = feature_series.dropna()
            aligned_targets = targets.reindex(aligned_feature.index).dropna()
            
            if len(aligned_feature) < 10 or len(aligned_targets) < 10:
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(aligned_feature, aligned_targets)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return abs(correlation)
            
        except Exception as e:
            print(f"Error calculating utility score: {e}")
            return 0.0

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    print(f"📊 Creating sample data with {n_samples} samples")
    
    # Generate realistic price data
    np.random.seed(42)
    
    # Create time index
    start_time = datetime.now() - timedelta(minutes=n_samples * 15)
    time_index = [start_time + timedelta(minutes=i * 15) for i in range(n_samples)]
    
    # Generate price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_samples)  # 0.01% mean return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = pd.DataFrame(index=time_index)
    data['open'] = prices * (1 + np.random.normal(0, 0.001, n_samples))
    data['high'] = np.maximum(data['open'], prices) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    data['low'] = np.minimum(data['open'], prices) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    data['close'] = prices
    data['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    
    print(f"✅ Sample data created: {data.shape}")
    print(f"   Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"   Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return data

def create_sample_targets(data: pd.DataFrame) -> pd.Series:
    """Create sample targets for supervised learning."""
    print("🎯 Creating sample targets")
    
    # Create forward returns as targets
    targets = data['close'].pct_change(5).shift(-5)  # 5-period forward returns
    targets = targets.dropna()
    
    print(f"✅ Targets created: {len(targets)} samples")
    print(f"   Target range: {targets.min():.4f} - {targets.max():.4f}")
    print(f"   Target mean: {targets.mean():.4f}, std: {targets.std():.4f}")
    
    return targets

def test_comprehensive_enhanced_feature_generation():
    """Test the comprehensive enhanced feature generation capabilities."""
    print("\n" + "="*80)
    print("🚀 TESTING COMPREHENSIVE ENHANCED FEATURE GENERATION")
    print("="*80)
    
    try:
        # Create sample data
        data = create_sample_data(500)  # Smaller dataset for faster testing
        targets = create_sample_targets(data)
        
        # Generate base features for interaction testing
        base_features = pd.DataFrame()
        base_features['price_change'] = data['close'].pct_change()
        base_features['volume_change'] = data['volume'].pct_change()
        base_features['high_low_ratio'] = data['high'] / data['low']
        base_features['close_open_ratio'] = data['close'] / data['open']
        
        # Create comprehensive enhanced feature generator
        print("\n🔧 Creating comprehensive enhanced feature generator")
        feature_generator = ComprehensiveEnhancedFeatureGenerator(base_timeframe_minutes=15)
        
        # Generate comprehensive enhanced features
        print("\n⚡ Generating comprehensive enhanced features")
        start_time = time.time()
        
        feature_result = feature_generator.generate_features(
            data, targets, base_features
        )
        
        generation_time = time.time() - start_time
        
        if feature_result['success']:
            print(f"✅ Comprehensive feature generation completed in {generation_time:.3f}s")
            print(f"   Cross-timeframe features: {len(feature_result['cross_timeframe_features'])}")
            print(f"   Interaction features: {len(feature_result['interaction_features'])}")
            print(f"   No features: {len(feature_result['no_features'])}")
            print(f"   Comparison features: {len(feature_result['comparison_features'])}")
            print(f"   Total features: {len(feature_result['all_features'])}")
            
            # Display sample features with enhanced metadata
            print("\n📋 Sample Cross-Timeframe Features:")
            for i, feature in enumerate(feature_result['cross_timeframe_features'][:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Lookback: {feature.lookback_period} minutes")
                print(f"      Base timeframe: {feature.base_timeframe_minutes} minutes")
                print(f"      Period in base units: {feature.metadata.get('period_in_base_units', 'N/A')}")
                print(f"      Method: {feature.creation_method}")
            
            print("\n📋 Sample Interaction Features:")
            for i, feature in enumerate(feature_result['interaction_features'][:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Parents: {feature.parent_features}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Method: {feature.creation_method}")
                print(f"      Source features: {feature.source_features}")
                print(f"      Interaction order: {feature.metadata.get('interaction_order', 'N/A')}")
            
            print("\n📋 Sample Comparison Features:")
            for i, feature in enumerate(feature_result['comparison_features'][:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Parents: {feature.parent_features}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Comparison type: {feature.comparison_type}")
                print(f"      Lookback: {feature.lookback_period} minutes")
                print(f"      Comparison types: {feature.metadata.get('comparison_types', 'N/A')}")
            
            # Test feature quality and metadata
            print("\n📊 Feature Quality and Metadata Analysis:")
            all_features = feature_result['all_features']
            if all_features:
                utilities = [f.utility_score for f in all_features]
                print(f"   Average utility: {np.mean(utilities):.4f}")
                print(f"   Max utility: {np.max(utilities):.4f}")
                print(f"   Min utility: {np.min(utilities):.4f}")
                print(f"   Features with utility > 0.1: {sum(1 for u in utilities if u > 0.1)}")
                
                # Check for different creation methods
                methods = [f.creation_method for f in all_features if f.creation_method]
                if methods:
                    method_counts = pd.Series(methods).value_counts()
                    print(f"   Creation methods used: {dict(method_counts)}")
                
                # Check for different feature types
                types = [f.feature_type for f in all_features]
                type_counts = pd.Series(types).value_counts()
                print(f"   Feature types: {dict(type_counts)}")
                
                # Check for different comparison types
                comparison_types = [f.comparison_type for f in all_features if f.comparison_type]
                if comparison_types:
                    comp_type_counts = pd.Series(comparison_types).value_counts()
                    print(f"   Comparison types: {dict(comp_type_counts)}")
                
                # Check for different lookback periods
                lookback_periods = [f.lookback_period for f in all_features if f.lookback_period]
                if lookback_periods:
                    lookback_counts = pd.Series(lookback_periods).value_counts()
                    print(f"   Lookback periods: {sorted(set(lookback_periods))}")
                    print(f"   Most common lookback: {lookback_counts.index[0]} minutes")
            
            # Test specific enhanced capabilities
            print("\n🧪 Testing Enhanced Capabilities:")
            
            # Test extended lookback periods
            cross_timeframe_features = feature_result['cross_timeframe_features']
            if cross_timeframe_features:
                print(f"   ✅ Cross-timeframe features: {len(cross_timeframe_features)}")
                lookback_periods = [f.lookback_period for f in cross_timeframe_features if f.lookback_period]
                if lookback_periods:
                    print(f"      Lookback periods: {sorted(set(lookback_periods))}")
                    print(f"      Max lookback: {max(lookback_periods)} minutes")
                    print(f"      Base timeframe respect: {all(f.base_timeframe_minutes == 15 for f in cross_timeframe_features)}")
            
            # Test enhanced interaction metadata
            interaction_features = feature_result['interaction_features']
            if interaction_features:
                print(f"   ✅ Interaction features: {len(interaction_features)}")
                source_features = [f.source_features for f in interaction_features if f.source_features]
                if source_features:
                    print(f"      Features with source metadata: {len(source_features)}")
                    print(f"      Sample source features: {source_features[0] if source_features else 'None'}")
            
            # Test comparison features
            comparison_features = feature_result['comparison_features']
            if comparison_features:
                print(f"   ✅ Comparison features: {len(comparison_features)}")
                comparison_types = [f.comparison_type for f in comparison_features if f.comparison_type]
                if comparison_types:
                    comp_type_counts = pd.Series(comparison_types).value_counts()
                    print(f"      Comparison types: {dict(comp_type_counts)}")
            
            print("\n🎉 Comprehensive enhanced feature generation test completed successfully!")
            
        else:
            print(f"❌ Comprehensive feature generation failed: {feature_result.get('error_message', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE ENHANCED FEATURE GENERATOR TEST")
    print("="*80)
    print("Testing comprehensive feature generation including:")
    print("✅ Extended lookback periods up to 600 minutes respecting base timeframe")
    print("✅ Enhanced interaction feature metadata with source and lookback information")
    print("✅ Feature comparisons between base, VWAP-based, volatility-adjusted, and z-score volume adjusted features")
    print("="*80)
    
    # Test comprehensive enhanced feature generator
    test_comprehensive_enhanced_feature_generation()
    
    print("\n" + "="*80)
    print("🎉 TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()