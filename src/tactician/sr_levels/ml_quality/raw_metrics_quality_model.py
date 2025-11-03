"""
Raw Metrics Quality Model

PURE DATA-DRIVEN APPROACH: Remove ALL heuristic thresholds and normalization.
Let the ML model learn what values are significant.

Current heuristics removed:
- bounce_threshold (e.g., 4% for 1h) → Model learns this
- hold_strength normalization (20 bars = 1.0) → Model learns this
- Fixed weights (0.25, 0.20, ...) → Model learns this
- Volume ratio thresholds (2.5x) → Model learns this
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from typing import Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class RawMetricsQualityModel:
    """
    Train on RAW performance metrics without heuristic normalization.
    
    Instead of:
        bounce_strength = min(weighted_bounce_pct / 0.04, 1.0)  # Heuristic threshold!
    
    We use:
        bounce_pct_raw = weighted_bounce_pct  # Let model learn threshold
    
    The ML model discovers:
    - What bounce % is actually significant (maybe 3%, not 4%)
    - What hold duration matters (maybe 15 bars, not 20)
    - How to combine metrics (learned weights, not 25/20/20/20/15)
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def prepare_raw_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract RAW performance metrics without normalization.
        
        Changes from current approach:
        1. bounce_pct_raw (not normalized to threshold)
        2. hold_bars_raw (not normalized to 20)
        3. trade_pnl_raw (not normalized to 1%)
        4. rejection_bars_raw (not converted to speed score)
        5. volume_ratio_raw (not normalized to 2.5x)
        
        Args:
            data: Training data with performance metrics
            
        Returns:
            DataFrame with raw features + original feature_* columns
        """
        
        raw_features = {}
        
        # 1. RAW BOUNCE METRICS (no thresholding)
        if 'bounce_strength' in data.columns:
            # Reverse the normalization if present, or use raw values
            # Current: bounce_strength = min(bounce_pct / threshold, 1.0)
            # We want: bounce_pct directly
            
            # If we have the raw bounce percentage, use it
            if 'weighted_bounce_pct_raw' in data.columns:
                raw_features['raw_bounce_pct'] = data['weighted_bounce_pct_raw']
            else:
                # Approximate: assume threshold was 4% (may need to store actual values)
                raw_features['raw_bounce_pct'] = data['bounce_strength'] * 0.04
            
            if 'max_bounce_strength' in data.columns:
                raw_features['raw_max_bounce_pct'] = data['max_bounce_strength'] * 0.04
        
        # 2. RAW HOLD METRICS (no normalization to 20 bars)
        if 'hold_strength' in data.columns:
            # Reverse: hold_strength = min(bars_until_break / 20, 1.0)
            if 'bars_until_break_raw' in data.columns:
                raw_features['raw_hold_bars'] = data['bars_until_break_raw']
            else:
                raw_features['raw_hold_bars'] = data['hold_strength'] * 20
            
            # Add binary: did it break at all?
            raw_features['raw_never_broke'] = (data['hold_strength'] == 1.0).astype(float)
        
        # 3. RAW TRADE METRICS (actual PnL %, not normalized)
        if 'trade_profit' in data.columns:
            # Current normalization: pnl_pct * 100 clipped to [-1, 1]
            # We want actual PnL %
            if 'trade_pnl_pct_raw' in data.columns:
                raw_features['raw_trade_pnl_pct'] = data['trade_pnl_pct_raw']
            else:
                raw_features['raw_trade_pnl_pct'] = data['trade_profit']  # Already somewhat raw
            
            # Add binary outcomes
            raw_features['raw_trade_won'] = (data['trade_profit'] > 0).astype(float)
            raw_features['raw_trade_lost'] = (data['trade_profit'] < 0).astype(float)
        
        # 4. RAW REJECTION METRICS (bars until rejection, not speed score)
        if 'rejection_speed' in data.columns:
            # Current: speed_score = 1.0 - (bar_index / 5.0)
            # Reverse to get bar index
            if 'rejection_bar_index_raw' in data.columns:
                raw_features['raw_rejection_bars'] = data['rejection_bar_index_raw']
            else:
                # Approximate (may be inaccurate)
                raw_features['raw_rejection_bars'] = (1.0 - data['rejection_speed']) * 5
            
            # Add binary: fast rejection (< 2 bars)
            raw_features['raw_fast_rejection'] = (data['rejection_speed'] > 0.6).astype(float)
        
        # 5. RAW VOLUME METRICS (actual ratios, not normalized)
        if 'volume_quality' in data.columns:
            # Current: volume_score = ratio / 2.5
            if 'test_volume_ratio_raw' in data.columns:
                raw_features['raw_test_volume_ratio'] = data['test_volume_ratio_raw']
            if 'bounce_volume_ratio_raw' in data.columns:
                raw_features['raw_bounce_volume_ratio'] = data['bounce_volume_ratio_raw']
            
            # Approximate if not available
            if 'test_volume_ratio_raw' not in data.columns:
                raw_features['raw_test_volume_ratio'] = data['volume_quality'] * 2.5
        
        # 6. TIMEFRAME ENCODING (let model learn timeframe-specific patterns)
        if 'timeframe' in data.columns:
            # One-hot encode timeframe (model learns different thresholds per TF)
            timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
            for tf in timeframes:
                raw_features[f'raw_is_tf_{tf}'] = (data['timeframe'] == tf).astype(float)
        
        # Combine with original feature_* columns
        raw_df = pd.DataFrame(raw_features, index=data.index)
        
        # Get all feature_* columns
        feature_cols = [c for c in data.columns if c.startswith('feature_')]
        features_df = data[feature_cols]
        
        # Merge
        result = pd.concat([features_df, raw_df], axis=1)
        
        return result
    
    def train(self, training_data: pd.DataFrame, 
              target_column: str = 'quality_score',
              n_folds: int = 5) -> Dict:
        """
        Train model on raw metrics.
        
        The model will learn:
        - What bounce % is significant (no hardcoded 4%)
        - What hold duration matters (no hardcoded 20 bars)
        - How to combine metrics (no hardcoded 0.25/0.20/...)
        - Different patterns per timeframe (no hardcoded adaptive thresholds)
        
        Args:
            training_data: Training data
            target_column: Target to predict
            n_folds: CV folds
            
        Returns:
            Training metrics
        """
        
        self.logger.info("🚀 Training Raw Metrics Quality Model (Pure Data-Driven)")
        self.logger.info("   Removed heuristics:")
        self.logger.info("     ❌ Fixed bounce thresholds (4%, 6%, 8%)")
        self.logger.info("     ❌ Fixed hold normalization (20 bars)")
        self.logger.info("     ❌ Fixed component weights (25/20/20/20/15)")
        self.logger.info("     ❌ Fixed volume thresholds (2.5x)")
        self.logger.info("   Model will learn all thresholds from data ✓")
        
        # Prepare raw features
        X = self.prepare_raw_features(training_data)
        y = training_data[target_column]
        
        self.feature_names = X.columns.tolist()
        
        self.logger.info(f"\n   Features: {len(self.feature_names)}")
        self.logger.info(f"   Samples: {len(X)}")
        
        # Train with CV
        tscv = TimeSeriesSplit(n_splits=n_folds)
        fold_models = []
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.05,
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            
            model = lgb.train(
                params,
                train_data_lgb,
                num_boost_round=1000,
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            cv_scores.append(r2)
            fold_models.append(model)
            
            self.logger.info(f"   Fold {fold_idx + 1}: R² = {r2:.3f}")
        
        # Use best model
        best_idx = np.argmax(cv_scores)
        self.model = fold_models[best_idx]
        
        self.logger.info(f"\n✅ Training complete")
        self.logger.info(f"   Avg R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Analyze what the model learned
        self._analyze_learned_patterns()
        
        return {'cv_scores': cv_scores}
    
    def _analyze_learned_patterns(self):
        """
        Analyze what thresholds and patterns the model learned.
        
        This reveals data-driven insights that replace our heuristics.
        """
        
        importance = self.model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"\n📊 LEARNED PATTERNS (replacing heuristics):")
        
        # Show top features
        self.logger.info(f"\n   Top 10 most important raw metrics:")
        for idx, row in importance_df.head(10).iterrows():
            self.logger.info(f"     {row['feature']:<40} {row['importance_pct']:>5.1f}%")
        
        # Compare component importance
        self.logger.info(f"\n   Component importance (compare to heuristic 25/20/20/20/15):")
        
        component_groups = {
            'Bounce': ['raw_bounce_pct', 'raw_max_bounce_pct'],
            'Hold': ['raw_hold_bars', 'raw_never_broke'],
            'Trade': ['raw_trade_pnl_pct', 'raw_trade_won', 'raw_trade_lost'],
            'Rejection': ['raw_rejection_bars', 'raw_fast_rejection'],
            'Volume': ['raw_test_volume_ratio', 'raw_bounce_volume_ratio']
        }
        
        for component_name, features in component_groups.items():
            total_importance = importance_df[
                importance_df['feature'].isin(features)
            ]['importance_pct'].sum()
            self.logger.info(f"     {component_name:<15} {total_importance:>5.1f}%")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict quality scores using learned thresholds."""
        
        X = self.prepare_raw_features(features)
        X = X[self.feature_names].fillna(0.0)
        
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 1)


class SurvivalAnalysisQualityModel:
    """
    ADVANCED: Model "time until break" using survival analysis.
    
    Current heuristic:
        hold_strength = min(bars_until_break / 20, 1.0)
    
    Data-driven alternative:
        Model the full survival curve: P(survives > t bars | features)
        
    This captures:
    - Hazard rate (risk of breaking at each time step)
    - Time-dependent covariates
    - Censored data (levels that never broke)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("⚠️  Survival analysis not implemented yet")
        self.logger.info("   Would require: lifelines library")
        self.logger.info("   Benefits: Proper modeling of time-to-break")
    
    def train(self, training_data: pd.DataFrame):
        """
        Train survival model.
        
        Would use Cox Proportional Hazards or Random Survival Forest:
        - Event: Level breaks
        - Time: Bars until break
        - Censoring: Levels that never broke (hold_strength = 1.0)
        - Covariates: All feature_* columns
        
        Output: Survival function S(t) = P(survives > t | features)
        Quality score = integral of S(t) or median survival time
        """
        pass


class ReinforcementLearningQualityModel:
    """
    ADVANCED: Learn quality scoring through trading simulation.
    
    Current approach: Hand-craft quality based on bounce/hold/trade metrics
    
    RL alternative:
    - Agent: SR quality scorer
    - Environment: Historical market data
    - Action: Assign quality score to level
    - Reward: Actual trading P&L when using that level
    - Policy: Learn Q(features) → quality_score that maximizes trading reward
    
    This is the MOST data-driven approach - directly optimizes for trading performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("⚠️  RL approach not implemented yet")
        self.logger.info("   Would require: stable-baselines3 or similar")
        self.logger.info("   Benefits: Optimizes directly for trading P&L")
    
    def train(self, training_data: pd.DataFrame):
        """
        Train RL agent.
        
        Setup:
        1. Create trading environment that:
           - Loads historical SR levels
           - Simulates trader using quality scores
           - Returns P&L as reward
        
        2. Train agent (e.g., PPO, SAC) to:
           - Observe level features
           - Assign quality score
           - Maximize cumulative trading reward
        
        3. Learned quality function directly reflects trading value
        """
        pass


def demonstrate_improvements():
    """
    Show how data-driven approaches improve on heuristics.
    """
    
    logger.info("="*80)
    logger.info("DATA-DRIVEN vs HEURISTIC COMPARISON")
    logger.info("="*80)
    
    comparisons = [
        {
            'component': 'Bounce Threshold',
            'heuristic': 'Fixed: 4% for 1h, 6% for 4h, 8% for 1d',
            'data_driven': 'Learned: Model discovers optimal threshold from data',
            'benefit': 'May find 3.5% is actually significant, not 4%'
        },
        {
            'component': 'Hold Normalization',
            'heuristic': 'Fixed: 20 bars = perfect (1.0)',
            'data_driven': 'Raw: Model learns what duration actually matters',
            'benefit': 'May find 15 bars is already strong for 1h'
        },
        {
            'component': 'Component Weights',
            'heuristic': 'Fixed: 25% bounce, 20% hold, 20% trade, 20% speed, 15% volume',
            'data_driven': 'Learned: Multi-task model discovers optimal combination',
            'benefit': 'Weights may vary by market regime (high vol → prioritize hold)'
        },
        {
            'component': 'Volume Threshold',
            'heuristic': 'Fixed: 2.5x average = good',
            'data_driven': 'Raw ratio: Model learns significance',
            'benefit': 'May find 1.8x is already strong signal'
        },
        {
            'component': 'Time-to-Break',
            'heuristic': 'Linear normalization: bars/20',
            'data_driven': 'Survival analysis: Model full survival curve',
            'benefit': 'Captures hazard rate, handles censoring properly'
        }
    ]
    
    for comp in comparisons:
        logger.info(f"\n{comp['component']}:")
        logger.info(f"  Heuristic:    {comp['heuristic']}")
        logger.info(f"  Data-Driven:  {comp['data_driven']}")
        logger.info(f"  Benefit:      {comp['benefit']}")

