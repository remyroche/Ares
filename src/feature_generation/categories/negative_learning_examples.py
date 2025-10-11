"""
Negative Learning Concrete Examples for ETHUSDT

This module provides concrete examples of how to implement negative learning
for specific ETHUSDT trading scenarios as described in the plugin plan.

Key Examples:
1. Momentum × High Volatility - Handles whipsaw in high vol
2. VWAP Distance × Wide Spread - Manages exhaustion signals
3. RSI × Chop - Adapts to different market regimes
4. Complete integration examples
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from src.utils.logger import system_logger
from src.feature_generation.categories.negative_learning import (
    NegativeLearningPlugin, 
    NegativeLearningConfig,
    FailureContextType
)
from src.feature_generation.categories.negative_learning_integration import (
    NegativeLearningPipelineManager,
    create_negative_learning_pipeline
)
from src.feature_generation.categories.negative_learning_selection import (
    NegativeLearningFeatureSelector,
    create_feature_selector
)
from src.feature_generation.categories.negative_learning_constraints import (
    ModelConstraintManager,
    create_constraint_manager
)
from src.feature_generation.categories.negative_learning_validation import (
    NegativeLearningValidator,
    create_negative_learning_validator
)


class ETHUSDTNegativeLearningExamples:
    """
    Concrete examples of negative learning implementation for ETHUSDT trading.
    Demonstrates the complete pipeline from feature generation to model training.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('ETHUSDTNegativeLearningExamples')
        
        # Initialize components
        self.pipeline_manager = create_negative_learning_pipeline()
        self.feature_selector = create_feature_selector()
        self.constraint_manager = create_constraint_manager()
        self.validator = create_negative_learning_validator()
    
    def example_1_momentum_high_volatility(self) -> Dict[str, Any]:
        """
        Example 1: Momentum × High Volatility
        
        Problem: Momentum signals flip in high volatility whipsaw conditions
        Solution: Gated twins that deactivate momentum in high vol, activate inverse in high vol
        
        Returns:
            Complete implementation example
        """
        self.logger.info("📊 Example 1: Momentum × High Volatility")
        
        # Create synthetic ETHUSDT data
        data = self._create_ethusdt_synthetic_data()
        
        # Define momentum features
        momentum_features = ['momentum_5m', 'momentum_15m', 'momentum_1h']
        
        # Create high volatility context using VectorBT
        if VECTORBT_AVAILABLE and len(data) > 1000:
            try:
                data['volatility'] = rolling_std(data['close'], window=20)
            except Exception as e:
                logger.warning(f"VectorBT volatility calculation failed: {e}, using pandas fallback")
                data['volatility'] = data['close'].rolling(20).std()
        else:
            data['volatility'] = data['close'].rolling(20).std()
        vol_threshold = data['volatility'].quantile(0.7)
        data['p_highvol'] = (data['volatility'] > vol_threshold).astype(float)
        
        # Create momentum features
        data['momentum_5m'] = data['close'].pct_change(5)
        data['momentum_15m'] = data['close'].pct_change(15)
        data['momentum_1h'] = data['close'].pct_change(60)
        
        # Create target (future returns)
        data['target'] = data['close'].pct_change(5).shift(-5)
        
        # Initialize negative learning plugin
        config = NegativeLearningConfig(
            max_negative_features=6,
            enable_gated_twins=True,
            enable_exception_interactions=True,
            enable_context_indicators=True
        )
        
        plugin = NegativeLearningPlugin(config)
        
        # Fit on training data
        train_data = data.iloc[:1000]
        plugin.fit(train_data, train_data['target'], momentum_features)
        
        # Generate negative learning features
        enhanced_data = plugin.transform(data)
        
        # Show the generated features
        negative_features = [col for col in enhanced_data.columns if col not in data.columns]
        
        example_result = {
            'description': 'Momentum × High Volatility - Handles whipsaw in high vol',
            'original_features': momentum_features,
            'negative_features': negative_features,
            'feature_examples': {
                'momentum_5m_pos': 'momentum_5m * (1 - p_highvol) - active where momentum should work',
                'momentum_5m_neg': '-momentum_5m * p_highvol - inverse where momentum fails',
                'momentum_5m_x_fail': 'momentum_5m * p_highvol - interaction for tree learning'
            },
            'monotone_constraints': {
                'momentum_5m_pos': '+1 (positive monotonicity)',
                'momentum_5m_neg': '-1 (negative monotonicity)',
                'momentum_5m_x_fail': '0 (no constraint)'
            },
            'expected_behavior': 'Trees learn to use momentum_5m_pos in normal conditions and momentum_5m_neg in high vol'
        }
        
        return example_result
    
    def example_2_vwap_widespread(self) -> Dict[str, Any]:
        """
        Example 2: VWAP Distance × Wide Spread
        
        Problem: VWAP pull signals fail when spread widens (exhaustion)
        Solution: Exception interactions that let trees down-weight VWAP when spread is wide
        
        Returns:
            Complete implementation example
        """
        self.logger.info("📊 Example 2: VWAP Distance × Wide Spread")
        
        # Create synthetic ETHUSDT data
        data = self._create_ethusdt_synthetic_data()
        
        # Create VWAP and spread features
        data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
        data['vwap_distance'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Create spread feature
        data['spread'] = data['high'] - data['low']
        spread_threshold = data['spread'].quantile(0.7)
        data['p_widespread'] = (data['spread'] > spread_threshold).astype(float)
        
        # Create target
        data['target'] = data['close'].pct_change(5).shift(-5)
        
        # Initialize negative learning plugin
        config = NegativeLearningConfig(
            max_negative_features=4,
            enable_gated_twins=False,  # Use lighter approach
            enable_exception_interactions=True,
            enable_context_indicators=True
        )
        
        plugin = NegativeLearningPlugin(config)
        
        # Fit and transform
        train_data = data.iloc[:1000]
        plugin.fit(train_data, train_data['target'], ['vwap_distance'])
        enhanced_data = plugin.transform(data)
        
        # Show generated features
        negative_features = [col for col in enhanced_data.columns if col not in data.columns]
        
        example_result = {
            'description': 'VWAP Distance × Wide Spread - Manages exhaustion signals',
            'original_features': ['vwap_distance'],
            'negative_features': negative_features,
            'feature_examples': {
                'vwap_distance_x_fail': 'vwap_distance * p_widespread - trees learn to down-weight when spread is wide',
                'vwap_distance_p_widespread': 'p_widespread - context indicator for splitting'
            },
            'monotone_constraints': {
                'vwap_distance_x_fail': '0 (no constraint - let trees learn)',
                'vwap_distance_p_widespread': '0 (context indicator)'
            },
            'expected_behavior': 'Trees learn to ignore VWAP signals when spread is wide (exhaustion)'
        }
        
        return example_result
    
    def example_3_rsi_chop(self) -> Dict[str, Any]:
        """
        Example 3: RSI × Chop
        
        Problem: RSI extremes work in chop but fail in trending markets
        Solution: Regime-aware RSI with different behavior in each context
        
        Returns:
            Complete implementation example
        """
        self.logger.info("📊 Example 3: RSI × Chop")
        
        # Create synthetic ETHUSDT data
        data = self._create_ethusdt_synthetic_data()
        
        # Create RSI features
        data['rsi_14'] = self._calculate_rsi(data['close'], 14)
        data['rsi_low'] = (data['rsi_14'] < 30).astype(float)  # Oversold
        data['rsi_high'] = (data['rsi_14'] > 70).astype(float)  # Overbought
        
        # Create chop detection (low R² of trend fit)
        data['p_chop'] = self._calculate_chop_flag(data['close'])
        
        # Create target
        data['target'] = data['close'].pct_change(5).shift(-5)
        
        # Initialize negative learning plugin
        config = NegativeLearningConfig(
            max_negative_features=6,
            enable_gated_twins=True,
            enable_exception_interactions=True,
            enable_context_indicators=True
        )
        
        plugin = NegativeLearningPlugin(config)
        
        # Fit and transform
        train_data = data.iloc[:1000]
        plugin.fit(train_data, train_data['target'], ['rsi_low', 'rsi_high'])
        enhanced_data = plugin.transform(data)
        
        # Show generated features
        negative_features = [col for col in enhanced_data.columns if col not in data.columns]
        
        example_result = {
            'description': 'RSI × Chop - Adapts to different market regimes',
            'original_features': ['rsi_low', 'rsi_high'],
            'negative_features': negative_features,
            'feature_examples': {
                'rsi_low_pos': 'rsi_low * p_chop - RSI oversold works in chop',
                'rsi_high_neg': '-rsi_high * (1 - p_chop) - RSI overbought fails in trending',
                'rsi_low_p_chop': 'p_chop - chop context indicator'
            },
            'monotone_constraints': {
                'rsi_low_pos': '+1 (positive in chop)',
                'rsi_high_neg': '-1 (negative in trending)',
                'rsi_low_p_chop': '0 (context indicator)'
            },
            'expected_behavior': 'RSI signals work in chop, fail in trending markets'
        }
        
        return example_result
    
    def example_4_complete_pipeline(self) -> Dict[str, Any]:
        """
        Example 4: Complete Pipeline Integration
        
        Shows how to integrate negative learning into the full Analyst/Tactician pipeline
        with proper time-series safety and latency budget management.
        
        Returns:
            Complete pipeline implementation
        """
        self.logger.info("📊 Example 4: Complete Pipeline Integration")
        
        # Create comprehensive ETHUSDT dataset
        data = self._create_comprehensive_ethusdt_data()
        
        # Split into Analyst (1h) and Tactician (15m) data
        analyst_data = data.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        tactician_data = data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Create features for each timeframe
        analyst_features = self._create_analyst_features(analyst_data)
        tactician_features = self._create_tactician_features(tactician_data)
        
        # Create targets
        analyst_target = analyst_data['close'].pct_change(4).shift(-4)  # 4h forward returns
        tactician_target = tactician_data['close'].pct_change(4).shift(-4)  # 1h forward returns
        
        # Retrain negative learning pipeline
        retrain_results = self.pipeline_manager.retrain_negative_learning(
            analyst_features=analyst_features,
            analyst_target=analyst_target,
            tactician_features=tactician_features,
            tactician_target=tactician_target,
            retrain_timestamp=datetime.now()
        )
        
        # Generate enhanced features for inference
        analyst_enhanced = self.pipeline_manager.get_analyst_features(analyst_features)
        tactician_enhanced = self.pipeline_manager.get_tactician_features(tactician_features)
        
        # Get model configurations
        model_configs = self.pipeline_manager.get_model_configs()
        
        # Get pipeline status
        pipeline_status = self.pipeline_manager.get_pipeline_status()
        
        example_result = {
            'description': 'Complete Pipeline Integration - Full Analyst/Tactician with negative learning',
            'retrain_results': retrain_results,
            'feature_counts': {
                'analyst_original': analyst_features.shape[1],
                'analyst_enhanced': analyst_enhanced.shape[1],
                'tactician_original': tactician_features.shape[1],
                'tactician_enhanced': tactician_enhanced.shape[1]
            },
            'model_configs': {
                'analyst_monotone_constraints': len(model_configs['analyst']['monotone_constraints']([])),
                'tactician_monotone_constraints': len(model_configs['tactician']['monotone_constraints']([]))
            },
            'pipeline_status': pipeline_status,
            'integration_steps': [
                '1. Retrain negative learning on both Analyst and Tactician data',
                '2. Generate enhanced features with time-series safety',
                '3. Apply monotone constraints and sample weights',
                '4. Monitor performance and drift',
                '5. Validate with bucketed performance analysis'
            ]
        }
        
        return example_result
    
    def example_5_hyperparameter_tuning(self) -> Dict[str, Any]:
        """
        Example 5: Hyperparameter Tuning for Tree Models
        
        Shows how to configure tree models with negative learning constraints
        and optimal hyperparameters for ETHUSDT trading.
        
        Returns:
            Hyperparameter configuration examples
        """
        self.logger.info("📊 Example 5: Hyperparameter Tuning")
        
        # LightGBM configuration with negative learning
        lightgbm_config = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 16,
            'max_depth': 4,
            'min_child_samples': 1000,
            'lambda_l2': 40,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # XGBoost configuration with negative learning
        xgboost_config = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,
            'min_child_weight': 1000,
            'lambda': 40,
            'alpha': 20,
            'colsample_bytree': 0.75,
            'subsample': 0.85,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # CatBoost configuration with negative learning
        catboost_config = {
            'loss_function': 'RMSE',
            'depth': 5,
            'l2_leaf_reg': 30,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 0.8,
            'od_type': 'Iter',
            'od_wait': 20,
            'learning_rate': 0.05,
            'iterations': 200,
            'random_seed': 42,
            'thread_count': -1,
            'verbose': False
        }
        
        # Negative learning specific configurations
        negative_learning_configs = {
            'analyst': {
                'max_negative_features': 8,
                'enable_gated_twins': True,
                'enable_exception_interactions': True,
                'enable_context_indicators': True,
                'monotone_constraints': 'auto',  # Will be generated
                'sample_weights': 'uncertainty_based'  # Will be generated
            },
            'tactician': {
                'max_negative_features': 6,
                'enable_gated_twins': True,
                'enable_exception_interactions': True,
                'enable_context_indicators': False,  # Lighter for 15m
                'monotone_constraints': 'auto',
                'sample_weights': 'uncertainty_based'
            }
        }
        
        example_result = {
            'description': 'Hyperparameter Tuning for Tree Models with Negative Learning',
            'lightgbm_config': lightgbm_config,
            'xgboost_config': xgboost_config,
            'catboost_config': catboost_config,
            'negative_learning_configs': negative_learning_configs,
            'tuning_notes': [
                'Use deeper trees (depth 4-6) to capture complex interactions',
                'Strong L2 regularization (20-60) to prevent overfitting',
                'Conservative learning rates (0.05) for stability',
                'Feature/bagging fractions (0.7-0.8) for robustness',
                'Monotone constraints on *_pos and *_neg features',
                'Sample weights down-weight uncertain failure zones'
            ],
            'ethusdt_specific': {
                'volatility_regime_detection': 'Use 20-period EWMA of volatility',
                'chop_detection': 'R² < 0.3 of 20-period linear trend fit',
                'spread_detection': 'Z-score > 0.52 (Q70) of rolling spread',
                'time_windows': 'First/last 30 minutes of trading day'
            }
        }
        
        return example_result
    
    def _create_ethusdt_synthetic_data(self, n_periods: int = 2000) -> pd.DataFrame:
        """Create synthetic ETHUSDT data for examples"""
        np.random.seed(42)
        
        # Generate price data with realistic ETHUSDT characteristics
        returns = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
        prices = 3000 * np.exp(np.cumsum(returns))  # Start at $3000
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1min'),
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_periods)
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def _create_comprehensive_ethusdt_data(self, n_days: int = 30) -> pd.DataFrame:
        """Create comprehensive ETHUSDT dataset for pipeline example"""
        np.random.seed(42)
        
        # Generate minute-level data
        n_periods = n_days * 24 * 60
        returns = np.random.normal(0, 0.001, n_periods)  # 0.1% per minute volatility
        prices = 3000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_periods, freq='1min'),
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(8, 1, n_periods)
        })
        
        data.set_index('timestamp', inplace=True)
        return data
    
    def _create_analyst_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create Analyst (1h) features"""
        features = data.copy()
        
        # HTF parent features using VectorBT
        if VECTORBT_AVAILABLE and len(data) > 1000:
            try:
                features['trend_strength'] = rolling_apply(data['close'], lambda x: np.polyfit(range(len(x)), x, 1)[0], window=20)
                features['volatility_regime'] = rolling_std(data['close'], window=20)
                features['volume_profile'] = rolling_mean(data['volume'], window=20)
            except Exception as e:
                logger.warning(f"VectorBT HTF features failed: {e}, using pandas fallback")
                features['trend_strength'] = data['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
                features['volatility_regime'] = data['close'].rolling(20).std()
                features['volume_profile'] = data['volume'].rolling(20).mean()
        else:
            features['trend_strength'] = data['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            features['volatility_regime'] = data['close'].rolling(20).std()
            features['volume_profile'] = data['volume'].rolling(20).mean()
        features['momentum_htf'] = data['close'].pct_change(20)
        
        return features
    
    def _create_tactician_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create Tactician (15m) features"""
        features = data.copy()
        
        # Fast features
        features['momentum_5m'] = data['close'].pct_change(5)
        features['momentum_15m'] = data['close'].pct_change(15)
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['vwap'] = (data['high'] + data['low'] + data['close']) / 3
        features['vwap_distance'] = (data['close'] - features['vwap']) / features['vwap']
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator using VectorBT"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        if VECTORBT_AVAILABLE and len(prices) > 1000:
            try:
                gain_mean = rolling_mean(gain, window=period)
                loss_mean = rolling_mean(loss, window=period)
            except Exception as e:
                logger.warning(f"VectorBT RSI calculation failed: {e}, using pandas fallback")
                gain_mean = gain.rolling(window=period).mean()
                loss_mean = loss.rolling(window=period).mean()
        else:
            gain_mean = gain.rolling(window=period).mean()
            loss_mean = loss.rolling(window=period).mean()
        
        rs = gain_mean / loss_mean
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_chop_flag(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate chop flag based on R² of trend fit"""
        r2_scores = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < 5:
                r2_scores.append(0)
                continue
            
            y = prices.iloc[start_idx:end_idx].values
            x = np.arange(len(y)).reshape(-1, 1)
            
            try:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(x, y)
                y_pred = reg.predict(x)
                r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
                r2_scores.append(max(0, r2))
            except:
                r2_scores.append(0)
        
        r2_series = pd.Series(r2_scores, index=prices.index)
        return (r2_series < 0.3).astype(float)


def run_all_examples() -> Dict[str, Any]:
    """Run all negative learning examples"""
    examples = ETHUSDTNegativeLearningExamples()
    
    results = {
        'example_1_momentum_high_volatility': examples.example_1_momentum_high_volatility(),
        'example_2_vwap_widespread': examples.example_2_vwap_widespread(),
        'example_3_rsi_chop': examples.example_3_rsi_chop(),
        'example_4_complete_pipeline': examples.example_4_complete_pipeline(),
        'example_5_hyperparameter_tuning': examples.example_5_hyperparameter_tuning()
    }
    
    return results


def get_quick_start_guide() -> str:
    """Get a quick start guide for implementing negative learning"""
    return """
# Negative Learning Quick Start Guide for ETHUSDT

## 1. Basic Setup
```python
from src.feature_generation.categories.negative_learning_integration import create_negative_learning_pipeline

# Create pipeline manager
pipeline = create_negative_learning_pipeline()

# Retrain on your data (once per retrain cycle)
retrain_results = pipeline.retrain_negative_learning(
    analyst_features=analyst_features,
    analyst_target=analyst_target,
    tactician_features=tactician_features,
    tactician_target=tactician_target
)
```

## 2. Generate Enhanced Features
```python
# For Analyst (1h)
analyst_enhanced = pipeline.get_analyst_features(analyst_features)

# For Tactician (15m)
tactician_enhanced = pipeline.get_tactician_features(tactician_features, analyst_outputs)
```

## 3. Get Model Configurations
```python
model_configs = pipeline.get_model_configs()

# Use in your model training
monotone_constraints = model_configs['analyst']['monotone_constraints'](feature_names)
sample_weights = model_configs['analyst']['sample_weights'](features_df)
```

## 4. Key Features Generated
- `feature_pos`: Active where rule should hold
- `feature_neg`: Inverse where rule tends to fail  
- `feature_x_fail`: Interaction for tree learning
- `feature_p_context`: Context indicators

## 5. Model Constraints
- Monotone constraints: +1 for *_pos, -1 for *_neg
- Sample weights: Down-weight uncertain failure zones
- Feature caps: Prevent extreme values

## 6. Validation
```python
from src.feature_generation.categories.negative_learning_validation import create_negative_learning_validator

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

validator = create_negative_learning_validator()
validation_results = validator.validate_negative_learning(
    features_df, target, negative_features, failure_contexts
)
```

## 7. ETHUSDT Specific Examples
- Momentum × High Vol: Handles whipsaw
- VWAP × Wide Spread: Manages exhaustion
- RSI × Chop: Adapts to regimes

## 8. Hyperparameters
- LightGBM: depth=4, lambda_l2=40, feature_fraction=0.75
- XGBoost: depth=4, lambda=40, colsample_bytree=0.75
- CatBoost: depth=5, l2_leaf_reg=30

## 9. Time-Series Safety
- All features built OOF on train
- As-of joined at inference
- No peeking past last HTF close

## 10. Latency Budget
- ≤10 negative learning features per head
- Estimated +30ms latency impact
- Budget compliance monitoring
"""


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Print quick start guide
    print(get_quick_start_guide())
    
    # Print example results
    for example_name, result in results.items():
        print(f"\n{'='*60}")
        print(f"EXAMPLE: {example_name}")
        print(f"{'='*60}")
        print(f"Description: {result['description']}")
        if 'feature_examples' in result:
            print("\nFeature Examples:")
            for feature, description in result['feature_examples'].items():
                print(f"  {feature}: {description}")
        if 'monotone_constraints' in result:
            print("\nMonotone Constraints:")
            for feature, constraint in result['monotone_constraints'].items():
                print(f"  {feature}: {constraint}")
        print(f"\nExpected Behavior: {result['expected_behavior']}")
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
