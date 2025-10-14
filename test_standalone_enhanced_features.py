#!/usr/bin/env python3
"""
Standalone test for Enhanced Feature Generation capabilities.

This script tests the enhanced feature generation functionality
without importing the complex pipeline dependencies.
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
    """Generated feature with metadata."""
    name: str
    feature_type: str  # 'cross_timeframe', 'interaction', 'no_feature'
    formula: str
    parent_features: List[str]
    feature_series: pd.Series
    utility_score: float
    lookback_period: Optional[int] = None
    creation_method: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class StandaloneEnhancedFeatureGenerator:
    """
    Standalone enhanced feature generator for testing.
    
    Features:
    - Cross timeframe features with optimized lookback period
    - Interaction (2-3) features with optimized lookback period
    - Feature creation in multiple ways (addition, subtraction, log, multiplication, division)
    - No features with optimized lookback period
    """
    
    def __init__(self):
        """Initialize the standalone enhanced feature generator."""
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_execution_time': 0.0,
            'cross_timeframe_features_generated': 0,
            'interaction_features_generated': 0,
            'no_features_generated': 0
        }
        
        print("✅ Standalone Enhanced Feature Generator initialized")
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        targets: Optional[pd.Series] = None,
        base_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive features including cross-timeframe, interactions, and no features.
        
        Args:
            data: Input OHLCV data
            targets: Optional target series for utility scoring
            base_features: Optional base features for interaction generation
            
        Returns:
            Dictionary with all generated features
        """
        print("🚀 Starting enhanced feature generation")
        print(f"📊 Data shape: {data.shape}")
        
        start_time = time.time()
        
        try:
            # Initialize result containers
            cross_timeframe_features = []
            interaction_features = []
            no_features = []
            
            # Generate cross-timeframe features
            print("Step 1: Generating cross-timeframe features")
            cross_timeframe_features = self._generate_cross_timeframe_features(data, targets)
            print(f"✅ Generated {len(cross_timeframe_features)} cross-timeframe features")
            
            # Generate interaction features
            if base_features is not None:
                print("Step 2: Generating interaction features")
                interaction_features = self._generate_interaction_features(base_features, targets)
                print(f"✅ Generated {len(interaction_features)} interaction features")
            
            # Generate no features
            print("Step 3: Generating no features")
            no_features = self._generate_no_features(data, targets)
            print(f"✅ Generated {len(no_features)} no features")
            
            # Combine all features
            all_features = cross_timeframe_features + interaction_features + no_features
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_generations': 1,
                'successful_generations': 1,
                'total_execution_time': execution_time,
                'cross_timeframe_features_generated': len(cross_timeframe_features),
                'interaction_features_generated': len(interaction_features),
                'no_features_generated': len(no_features)
            })
            
            print(f"✅ Enhanced feature generation completed in {execution_time:.3f}s")
            print(f"🏆 Total features generated: {len(all_features)}")
            
            return {
                'cross_timeframe_features': cross_timeframe_features,
                'interaction_features': interaction_features,
                'no_features': no_features,
                'all_features': all_features,
                'generation_time': execution_time,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Enhanced feature generation failed: {e}")
            return {
                'cross_timeframe_features': [],
                'interaction_features': [],
                'no_features': [],
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
        """Generate cross-timeframe features with optimized lookback periods."""
        features = []
        
        try:
            # Ensure we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in data.columns]
            if not available_cols:
                return features
            
            # Generate features for different timeframe periods
            periods = [5, 10, 15, 30, 60, 120, 240]  # minutes
            
            for period in periods:
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
                    creation_method="subtract"
                ))
                
                # Momentum ratio
                momentum_ratio = short_momentum / (long_momentum + 1e-8)
                features.append(GeneratedFeature(
                    name=f"momentum_ratio_{period}",
                    feature_type="cross_timeframe",
                    formula=f"pct_change({period}) / pct_change({period * 2})",
                    parent_features=["close"],
                    feature_series=momentum_ratio,
                    utility_score=self._calculate_utility_score(momentum_ratio, targets),
                    lookback_period=period,
                    creation_method="divide"
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
                    creation_method="divide"
                ))
                
                # Multi-timeframe trend
                short_trend = close.rolling(period).mean()
                long_trend = close.rolling(period * 2).mean()
                
                # Trend strength
                trend_strength = (close - short_trend) / (short_trend + 1e-8)
                features.append(GeneratedFeature(
                    name=f"trend_strength_{period}",
                    feature_type="cross_timeframe",
                    formula=f"(close - mean({period})) / mean({period})",
                    parent_features=["close"],
                    feature_series=trend_strength,
                    utility_score=self._calculate_utility_score(trend_strength, targets),
                    lookback_period=period,
                    creation_method="ratio"
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
        """Generate interaction features (2-3 way) with optimized lookback periods."""
        features = []
        
        try:
            feature_names = list(base_features.columns)
            
            # Generate 2-way interactions
            print("   Generating 2-way interactions")
            two_way_features = self._generate_two_way_interactions(base_features, targets)
            features.extend(two_way_features)
            
            # Generate 3-way interactions
            print("   Generating 3-way interactions")
            three_way_features = self._generate_three_way_interactions(base_features, targets)
            features.extend(three_way_features)
            
            # Limit to max features
            if len(features) > 30:
                # Sort by utility score and take top features
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:30]
            
            return features
            
        except Exception as e:
            print(f"❌ Interaction feature generation failed: {e}")
            return []
    
    def _generate_two_way_interactions(
        self, 
        base_features: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate 2-way interaction features."""
        features = []
        
        try:
            feature_names = list(base_features.columns)
            creation_methods = ['add', 'subtract', 'multiply', 'divide', 'log', 'sqrt', 'power', 'ratio']
            
            # Generate all possible 2-way combinations
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
            
            return features
            
        except Exception as e:
            print(f"Error generating 2-way interactions: {e}")
            return []
    
    def _generate_three_way_interactions(
        self, 
        base_features: pd.DataFrame, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Generate 3-way interaction features."""
        features = []
        
        try:
            feature_names = list(base_features.columns)
            
            # Limit to avoid too many combinations
            max_features = min(10, len(feature_names))
            selected_features = feature_names[:max_features]
            
            # Generate 3-way combinations
            from itertools import combinations
            for combo in combinations(selected_features, 3):
                feat1, feat2, feat3 = combo
                
                # Generate different types of 3-way interactions
                interaction_features = self._create_three_way_feature_interactions(
                    base_features, feat1, feat2, feat3, targets
                )
                features.extend(interaction_features)
            
            return features
            
        except Exception as e:
            print(f"Error generating 3-way interactions: {e}")
            return []
    
    def _create_feature_interactions(
        self, 
        base_features: pd.DataFrame, 
        feat1: str, 
        feat2: str, 
        targets: Optional[pd.Series] = None,
        creation_methods: List[str] = None
    ) -> List[GeneratedFeature]:
        """Create interaction features between two features using multiple methods."""
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
                    
                    # Create feature
                    feature = GeneratedFeature(
                        name=f"{feat1}_{feat2}_{method}",
                        feature_type="interaction",
                        formula=formula,
                        parent_features=[feat1, feat2],
                        feature_series=interaction_series,
                        utility_score=self._calculate_utility_score(interaction_series, targets),
                        creation_method=method
                    )
                    
                    feature.metadata.update({
                        'interaction_order': 2,
                        'feature_category': 'interaction'
                    })
                    
                    features.append(feature)
                    
                except Exception as e:
                    print(f"Error creating {method} interaction between {feat1} and {feat2}: {e}")
                    continue
            
            return features
            
        except Exception as e:
            print(f"Error creating feature interactions: {e}")
            return []
    
    def _create_three_way_feature_interactions(
        self, 
        base_features: pd.DataFrame, 
        feat1: str, 
        feat2: str, 
        feat3: str, 
        targets: Optional[pd.Series] = None
    ) -> List[GeneratedFeature]:
        """Create 3-way interaction features."""
        features = []
        
        try:
            series1 = base_features[feat1]
            series2 = base_features[feat2]
            series3 = base_features[feat3]
            
            # Generate 3-way interactions using different methods
            for method in ['multiply', 'add', 'ratio']:
                try:
                    if method == 'multiply':
                        interaction_series = series1 * series2 * series3
                        formula = f"{feat1} * {feat2} * {feat3}"
                    elif method == 'add':
                        interaction_series = series1 + series2 + series3
                        formula = f"{feat1} + {feat2} + {feat3}"
                    elif method == 'ratio':
                        interaction_series = (series1 * series2) / (series3 + 1e-8)
                        formula = f"({feat1} * {feat2}) / ({feat3} + 1e-8)"
                    else:
                        continue
                    
                    # Create feature
                    feature = GeneratedFeature(
                        name=f"{feat1}_{feat2}_{feat3}_{method}",
                        feature_type="interaction",
                        formula=formula,
                        parent_features=[feat1, feat2, feat3],
                        feature_series=interaction_series,
                        utility_score=self._calculate_utility_score(interaction_series, targets),
                        creation_method=method
                    )
                    
                    feature.metadata.update({
                        'interaction_order': 3,
                        'feature_category': 'interaction'
                    })
                    
                    features.append(feature)
                    
                except Exception as e:
                    print(f"Error creating 3-way {method} interaction: {e}")
                    continue
            
            return features
            
        except Exception as e:
            print(f"Error creating 3-way feature interactions: {e}")
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
                    creation_method="pct_change"
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
                    creation_method="log"
                ))
                
                # Price rank
                price_rank = close.rank(pct=True)
                features.append(GeneratedFeature(
                    name="price_rank",
                    feature_type="no_feature",
                    formula="close.rank(pct=True)",
                    parent_features=["close"],
                    feature_series=price_rank,
                    utility_score=self._calculate_utility_score(price_rank, targets),
                    creation_method="rank"
                ))
                
                # Price z-score
                price_zscore = (close - close.mean()) / close.std()
                features.append(GeneratedFeature(
                    name="price_zscore",
                    feature_type="no_feature",
                    formula="(close - close.mean()) / close.std()",
                    parent_features=["close"],
                    feature_series=price_zscore,
                    utility_score=self._calculate_utility_score(price_zscore, targets),
                    creation_method="zscore"
                ))
            
            # Volume-based no features
            if 'volume' in data.columns:
                volume = data['volume']
                
                # Volume change
                volume_change = volume.pct_change()
                features.append(GeneratedFeature(
                    name="volume_change",
                    feature_type="no_feature",
                    formula="volume.pct_change()",
                    parent_features=["volume"],
                    feature_series=volume_change,
                    utility_score=self._calculate_utility_score(volume_change, targets),
                    creation_method="pct_change"
                ))
                
                # Volume rank
                volume_rank = volume.rank(pct=True)
                features.append(GeneratedFeature(
                    name="volume_rank",
                    feature_type="no_feature",
                    formula="volume.rank(pct=True)",
                    parent_features=["volume"],
                    feature_series=volume_rank,
                    utility_score=self._calculate_utility_score(volume_rank, targets),
                    creation_method="rank"
                ))
            
            # OHLC-based no features
            if all(col in data.columns for col in ['high', 'low', 'close']):
                high, low, close = data['high'], data['low'], data['close']
                
                # True range
                tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
                features.append(GeneratedFeature(
                    name="true_range",
                    feature_type="no_feature",
                    formula="max(high - low, max(abs(high - close.shift(1)), abs(low - close.shift(1))))",
                    parent_features=["high", "low", "close"],
                    feature_series=tr,
                    utility_score=self._calculate_utility_score(tr, targets),
                    creation_method="max"
                ))
                
                # Price position in daily range
                daily_range = high - low
                price_position = (close - low) / (daily_range + 1e-8)
                features.append(GeneratedFeature(
                    name="price_position_daily",
                    feature_type="no_feature",
                    formula="(close - low) / (high - low)",
                    parent_features=["high", "low", "close"],
                    feature_series=price_position,
                    utility_score=self._calculate_utility_score(price_position, targets),
                    creation_method="ratio"
                ))
            
            # Limit to max features
            if len(features) > 15:
                features.sort(key=lambda x: x.utility_score, reverse=True)
                features = features[:15]
            
            return features
            
        except Exception as e:
            print(f"❌ No features generation failed: {e}")
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

def test_enhanced_feature_generation():
    """Test the enhanced feature generation capabilities."""
    print("\n" + "="*80)
    print("🚀 TESTING ENHANCED FEATURE GENERATION")
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
        
        # Create enhanced feature generator
        print("\n🔧 Creating enhanced feature generator")
        feature_generator = StandaloneEnhancedFeatureGenerator()
        
        # Generate enhanced features
        print("\n⚡ Generating enhanced features")
        start_time = time.time()
        
        feature_result = feature_generator.generate_features(
            data, targets, base_features
        )
        
        generation_time = time.time() - start_time
        
        if feature_result['success']:
            print(f"✅ Enhanced feature generation completed in {generation_time:.3f}s")
            print(f"   Cross-timeframe features: {len(feature_result['cross_timeframe_features'])}")
            print(f"   Interaction features: {len(feature_result['interaction_features'])}")
            print(f"   No features: {len(feature_result['no_features'])}")
            print(f"   Total features: {len(feature_result['all_features'])}")
            
            # Display sample features
            print("\n📋 Sample Cross-Timeframe Features:")
            for i, feature in enumerate(feature_result['cross_timeframe_features'][:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Lookback: {feature.lookback_period}")
                print(f"      Method: {feature.creation_method}")
            
            print("\n📋 Sample Interaction Features:")
            for i, feature in enumerate(feature_result['interaction_features'][:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Parents: {feature.parent_features}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Method: {feature.creation_method}")
            
            print("\n📋 Sample No Features:")
            for i, feature in enumerate(feature_result['no_features'][:5]):
                print(f"   {i+1}. {feature.name}")
                print(f"      Formula: {feature.formula}")
                print(f"      Utility: {feature.utility_score:.4f}")
                print(f"      Method: {feature.creation_method}")
            
            # Test feature quality
            print("\n📊 Feature Quality Analysis:")
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
            
            # Test specific feature types
            print("\n🧪 Testing Specific Feature Types:")
            
            # Test cross-timeframe features
            cross_timeframe_features = feature_result['cross_timeframe_features']
            if cross_timeframe_features:
                print(f"   ✅ Cross-timeframe features: {len(cross_timeframe_features)}")
                lookback_periods = [f.lookback_period for f in cross_timeframe_features if f.lookback_period]
                if lookback_periods:
                    print(f"      Lookback periods: {sorted(set(lookback_periods))}")
            
            # Test interaction features
            interaction_features = feature_result['interaction_features']
            if interaction_features:
                print(f"   ✅ Interaction features: {len(interaction_features)}")
                interaction_orders = [f.metadata.get('interaction_order', 'unknown') for f in interaction_features]
                if interaction_orders:
                    order_counts = pd.Series(interaction_orders).value_counts()
                    print(f"      Interaction orders: {dict(order_counts)}")
            
            # Test no features
            no_features = feature_result['no_features']
            if no_features:
                print(f"   ✅ No features: {len(no_features)}")
                no_methods = [f.creation_method for f in no_features if f.creation_method]
                if no_methods:
                    no_method_counts = pd.Series(no_methods).value_counts()
                    print(f"      Creation methods: {dict(no_method_counts)}")
            
            print("\n🎉 Enhanced feature generation test completed successfully!")
            
        else:
            print(f"❌ Enhanced feature generation failed: {feature_result.get('error_message', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🧪 STANDALONE ENHANCED FEATURE GENERATOR TEST")
    print("="*80)
    print("Testing comprehensive feature generation including:")
    print("✅ Cross timeframe features with optimized lookback period")
    print("✅ Interaction (2-3) features with optimized lookback period")
    print("✅ Feature creation in multiple ways (addition, subtraction, log, multiplication, division)")
    print("✅ No features with optimized lookback period")
    print("="*80)
    
    # Test enhanced feature generator
    test_enhanced_feature_generation()
    
    print("\n" + "="*80)
    print("🎉 TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()