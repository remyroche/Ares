"""
Multi-Target System for Feature Evaluation

This module implements a comprehensive multi-target system that evaluates features
against multiple targets including mean-reversion, trend-following, and other
target families with proper metrics and reporting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class MultiTargetSystem:
    """
    Multi-target system for comprehensive feature evaluation.
    """
    
    def __init__(self, 
                 horizons: List[int] = [1, 2, 3],
                 volatility_window: int = 20,
                 neutral_threshold: float = 0.001,
                 tail_quantile: float = 0.05,
                 breakout_std_multiplier: float = 2.0,
                 breakout_min_bars: int = 3,
                 profit_taking_upper: float = 0.006,  # 0.6%
                 profit_taking_lower: float = 0.003,  # 0.3%
                 stop_loss: float = 0.003,  # 0.3%
                 max_bars: int = 3,
                 timeframe_minutes: int = 15):
        """
        Initialize multi-target system.
        
        Args:
            horizons: Prediction horizons in bars (H = 1, 2, 3)
            volatility_window: Window for volatility calculation
            neutral_threshold: Threshold for neutral zones (ignore tiny |R|)
            tail_quantile: Quantile for tail risk events
            breakout_std_multiplier: Standard deviation multiplier for breakout detection
            breakout_min_bars: Minimum bars to stay outside band
            profit_taking_upper: Upper profit-taking barrier (0.6%)
            profit_taking_lower: Lower profit-taking barrier (0.3%)
            stop_loss: Stop-loss barrier (0.3%)
            max_bars: Maximum bars for triple barrier
            timeframe_minutes: Timeframe in minutes (15m)
        """
        self.horizons = horizons
        self.volatility_window = volatility_window
        self.neutral_threshold = neutral_threshold
        self.tail_quantile = tail_quantile
        self.breakout_std_multiplier = breakout_std_multiplier
        self.breakout_min_bars = breakout_min_bars
        self.profit_taking_upper = profit_taking_upper
        self.profit_taking_lower = profit_taking_lower
        self.stop_loss = stop_loss
        self.max_bars = max_bars
        self.timeframe_minutes = timeframe_minutes
        
        # Initialize results storage
        self.target_results = {}
        self.feature_performance = {}
        self.multi_target_metrics = {}
        
        # Initialize data manager
        self.data_manager = None
        self._initialize_data_manager()
    
    def _initialize_data_manager(self):
        """Initialize the KlinesParquetManager."""
        try:
            from src.utils.data.klines_parquet import KlinesParquetManager
            self.data_manager = KlinesParquetManager()
            logger.info("KlinesParquetManager initialized successfully")
        except ImportError as e:
            logger.warning(f"Could not import KlinesParquetManager: {e}")
            self.data_manager = None
    
    def load_market_data(self, 
                        symbol: str = "ETHUSDT",
                        interval: str = "15m",
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        data_type: str = "raw",
                        fallback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Load market data using KlinesParquetManager.
        
        Args:
            symbol: Trading symbol (default: ETHUSDT)
            interval: Data interval (default: 15m)
            start_date: Start date for filtering
            end_date: End date for filtering
            data_type: 'raw' or 'processed'
            fallback_days: Days to fallback to if no data in range
            
        Returns:
            DataFrame with market data or None if not found
        """
        if self.data_manager is None:
            logger.error("KlinesParquetManager not available")
            return None
        
        try:
            # Try to load data with specified date range
            data = self.data_manager.read_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type
            )
            
            # If no data found and we have a date range, try fallback
            if (data is None or data.empty) and (start_date is not None or end_date is not None):
                logger.info(f"No data found in specified range, trying last {fallback_days} days fallback")
                data = self.data_manager.read_last_x_days_data(
                    symbol=symbol,
                    interval=interval,
                    x_days=fallback_days,
                    data_type=data_type
                )
            
            if data is not None and not data.empty:
                # Ensure we have the required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns: {missing_columns}")
                    return None
                
                # Ensure data is sorted by timestamp
                if not data.index.is_monotonic_increasing:
                    data = data.sort_index()
                
                logger.info(f"Loaded {len(data)} records for {symbol} {interval}")
                logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
                
                return data
            else:
                logger.warning(f"No data available for {symbol} {interval}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return None
    
    def create_all_targets(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create all target families for evaluation.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with targets by family
        """
        logger.info("Creating multi-target system with all target families...")
        
        targets = {}
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate volatility
        volatility = returns.rolling(self.volatility_window).std()
        
        # 1. Mean-Reversion Targets
        targets['mean_reversion'] = self._create_mean_reversion_targets(returns, volatility)
        
        # 2. Trend-Following Targets
        targets['trend_following'] = self._create_trend_following_targets(returns, volatility)
        
        # 3. Directional & Probability Targets
        targets['directional'] = self._create_directional_targets(returns)
        
        # 4. Magnitude / Volatility Forecasting
        targets['volatility'] = self._create_volatility_targets(returns, data)
        
        # 5. Tail Risk / Jump Likelihood
        targets['tail_risk'] = self._create_tail_risk_targets(returns)
        
        # 6. Breakout / Reversal Speed
        targets['breakout'] = self._create_breakout_targets(returns, data)
        
        # 7. Risk-Adjusted Return
        targets['risk_adjusted'] = self._create_risk_adjusted_targets(returns, volatility)
        
        # 8. Meta-Labeling (Triple Barrier)
        targets['meta_labeling'] = self._create_meta_labeling_targets(returns, data)
        
        logger.info(f"Created targets for {len(targets)} families")
        return targets
    
    def _create_mean_reversion_targets(self, returns: pd.Series, volatility: pd.Series) -> pd.DataFrame:
        """Create mean-reversion targets."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Future returns
            future_returns = returns.shift(-H)
            
            # MR strength: y^MR_{t,H} = -r_t * R_{t->t+H}
            mr_strength = -returns * future_returns
            targets[f'mr_strength_{H}'] = mr_strength
            
            # Risk-adjusted MR: y~^MR_{t,H} = y^MR_{t,H} / σ_t(W)
            risk_adj_mr = mr_strength / (volatility + 1e-8)
            targets[f'risk_adj_mr_{H}'] = risk_adj_mr
            
            # MR hit (opposite sign): 1{sign(R_{t->t+H}) = -sign(r_t)}
            mr_hit = ((returns > 0) & (future_returns < 0)) | ((returns < 0) & (future_returns > 0))
            targets[f'mr_hit_{H}'] = mr_hit.astype(float)
            
            # Neutral zones (ignore tiny |R| below threshold)
            neutral_mask = returns.abs() < self.neutral_threshold
            targets[f'mr_hit_neutral_{H}'] = mr_hit.astype(float)
            targets[f'mr_hit_neutral_{H}'].loc[neutral_mask] = np.nan
        
        return targets
    
    def _create_trend_following_targets(self, returns: pd.Series, volatility: pd.Series) -> pd.DataFrame:
        """Create trend-following targets."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Future returns
            future_returns = returns.shift(-H)
            
            # Trend strength: y^TR_{t,H} = r_t * R_{t->t+H}
            trend_strength = returns * future_returns
            targets[f'trend_strength_{H}'] = trend_strength
            
            # Risk-adjusted trend: y~^TR_{t,H} = y^TR_{t,H} / σ_t(W)
            risk_adj_trend = trend_strength / (volatility + 1e-8)
            targets[f'risk_adj_trend_{H}'] = risk_adj_trend
            
            # Trend hit (same sign): 1{sign(R_{t->t+H}) = sign(r_t)}
            trend_hit = ((returns > 0) & (future_returns > 0)) | ((returns < 0) & (future_returns < 0))
            targets[f'trend_hit_{H}'] = trend_hit.astype(float)
            
            # Neutral zones
            neutral_mask = returns.abs() < self.neutral_threshold
            targets[f'trend_hit_neutral_{H}'] = trend_hit.astype(float)
            targets[f'trend_hit_neutral_{H}'].loc[neutral_mask] = np.nan
        
        return targets
    
    def _create_directional_targets(self, returns: pd.Series) -> pd.DataFrame:
        """Create directional and probability targets."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Future returns
            future_returns = returns.shift(-H)
            
            # Binary direction: 1{R_{t->t+H} > 0}
            binary_direction = (future_returns > 0).astype(float)
            targets[f'binary_direction_{H}'] = binary_direction
            
            # Calibrated probability: Pr(R_{t->t+H} > 0)
            # Use rolling probability estimation
            prob_window = 100
            rolling_prob = future_returns.rolling(prob_window).apply(
                lambda x: (x > 0).mean(), raw=True
            )
            targets[f'calibrated_prob_{H}'] = rolling_prob
        
        return targets
    
    def _create_volatility_targets(self, returns: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility forecasting targets."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Realized vol: sqrt(sum(r_{t+i}^2))
            realized_vol = returns.rolling(H).apply(
                lambda x: np.sqrt((x**2).sum()), raw=True
            ).shift(-H)
            targets[f'realized_vol_{H}'] = realized_vol
            
            # Range-based vol (Parkinson)
            if 'high' in data.columns and 'low' in data.columns:
                high = data['high']
                low = data['low']
                parkinson_vol = np.sqrt(0.25 * np.log(high / low)**2)
                range_vol = parkinson_vol.rolling(H).mean().shift(-H)
                targets[f'range_vol_{H}'] = range_vol
        
        return targets
    
    def _create_tail_risk_targets(self, returns: pd.Series) -> pd.DataFrame:
        """Create tail risk and jump likelihood targets."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Future returns
            future_returns = returns.shift(-H)
            
            # Calculate historical quantiles
            quantile_window = 500
            historical_quantiles = future_returns.rolling(quantile_window).quantile(self.tail_quantile)
            
            # Left-tail event: 1{R_{t->t+H} < q_p}
            left_tail = (future_returns < historical_quantiles).astype(float)
            targets[f'left_tail_{H}'] = left_tail
            
            # Right-tail event: 1{R_{t->t+H} > q_{1-p}}
            right_tail = (future_returns > historical_quantiles.rolling(quantile_window).quantile(1 - self.tail_quantile)).astype(float)
            targets[f'right_tail_{H}'] = right_tail
        
        return targets
    
    def _create_breakout_targets(self, returns: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
        """Create breakout and reversal speed targets."""
        targets = pd.DataFrame(index=returns.index)
        
        if 'volume' in data.columns:
            # Calculate VWAP
            vwap = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            basis = data['close'] - vwap
            
            for H in self.horizons:
                # VWAP mean-reversion speed: -basis_t * Δbasis_{t->t+H}
                basis_change = basis.shift(-H) - basis
                vwap_mr_speed = -basis * basis_change
                targets[f'vwap_mr_speed_{H}'] = vwap_mr_speed
        
        # Breakout detection
        for H in self.horizons:
            # Rolling band: ±kσ
            rolling_mean = data['close'].rolling(20).mean()
            rolling_std = data['close'].rolling(20).std()
            upper_band = rolling_mean + self.breakout_std_multiplier * rolling_std
            lower_band = rolling_mean - self.breakout_std_multiplier * rolling_std
            
            # Check if price exits band and stays outside
            breakout_upper = (data['close'] > upper_band).astype(int)
            breakout_lower = (data['close'] < lower_band).astype(int)
            
            # Count consecutive bars outside band
            upper_consecutive = breakout_upper.groupby((breakout_upper != breakout_upper.shift()).cumsum()).cumsum()
            lower_consecutive = breakout_lower.groupby((breakout_lower != breakout_lower.shift()).cumsum()).cumsum()
            
            # Breakout flag: 1 if stays outside ≥M bars
            breakout_flag = ((upper_consecutive >= self.breakout_min_bars) | 
                           (lower_consecutive >= self.breakout_min_bars)).astype(float)
            targets[f'breakout_flag_{H}'] = breakout_flag.shift(-H)
        
        return targets
    
    def _create_risk_adjusted_targets(self, returns: pd.Series, volatility: pd.Series) -> pd.DataFrame:
        """Create risk-adjusted return targets."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Future returns
            future_returns = returns.shift(-H)
            
            # Sharpe-like: R_{t->t+H} / σ_t(W)
            sharpe_like = future_returns / (volatility + 1e-8)
            targets[f'sharpe_like_{H}'] = sharpe_like
        
        return targets
    
    def _create_meta_labeling_targets(self, returns: pd.Series, data: pd.DataFrame) -> pd.DataFrame:
        """Create meta-labeling targets using triple barrier method."""
        targets = pd.DataFrame(index=returns.index)
        
        for H in self.horizons:
            # Triple barrier method
            # Upper barriers: 0.6% and 1%
            # Lower barrier: half of upper (0.3% and 0.5%)
            # Stop loss: 0.3%
            # Max bars: 3
            
            # Calculate barriers
            upper_barrier_1 = self.profit_taking_upper  # 0.6%
            upper_barrier_2 = self.profit_taking_upper * 1.67  # 1%
            lower_barrier_1 = self.profit_taking_lower  # 0.3%
            lower_barrier_2 = self.profit_taking_lower * 1.67  # 0.5%
            stop_loss = self.stop_loss  # 0.3%
            
            # Create meta-labels
            meta_labels = self._triple_barrier_method(
                data['close'], upper_barrier_1, upper_barrier_2,
                lower_barrier_1, lower_barrier_2, stop_loss, self.max_bars
            )
            
            targets[f'meta_label_{H}'] = meta_labels.shift(-H)
        
        return targets
    
    def _triple_barrier_method(self, prices: pd.Series, upper_1: float, upper_2: float,
                              lower_1: float, lower_2: float, stop_loss: float, max_bars: int) -> pd.Series:
        """Implement triple barrier method for meta-labeling."""
        meta_labels = pd.Series(0.0, index=prices.index)
        
        for i in range(len(prices) - max_bars):
            current_price = prices.iloc[i]
            
            # Set barriers based on current price
            upper_barrier_1_price = current_price * (1 + upper_1)
            upper_barrier_2_price = current_price * (1 + upper_2)
            lower_barrier_1_price = current_price * (1 - lower_1)
            lower_barrier_2_price = current_price * (1 - lower_2)
            stop_loss_price = current_price * (1 - stop_loss)
            
            # Check next max_bars for barrier hits
            for j in range(1, min(max_bars + 1, len(prices) - i)):
                future_price = prices.iloc[i + j]
                
                # Check upper barriers (profit taking)
                if future_price >= upper_barrier_2_price:
                    meta_labels.iloc[i] = 1.0  # Hit upper barrier 2
                    break
                elif future_price >= upper_barrier_1_price:
                    meta_labels.iloc[i] = 0.5  # Hit upper barrier 1
                    break
                
                # Check lower barriers (profit taking)
                elif future_price <= lower_barrier_2_price:
                    meta_labels.iloc[i] = 1.0  # Hit lower barrier 2
                    break
                elif future_price <= lower_barrier_1_price:
                    meta_labels.iloc[i] = 0.5  # Hit lower barrier 1
                    break
                
                # Check stop loss
                elif future_price <= stop_loss_price:
                    meta_labels.iloc[i] = -1.0  # Hit stop loss
                    break
        
        return meta_labels
    
    def evaluate_features_against_targets(self, X: pd.DataFrame, targets: Dict[str, pd.DataFrame],
                                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate features against all targets.
        
        Args:
            X: Feature matrix
            targets: Dictionary of target families
            feature_names: List of feature names to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Evaluating features against all targets...")
        
        if feature_names is None:
            feature_names = list(X.columns)
        
        results = {
            'target_families': {},
            'feature_performance': {},
            'multi_target_summary': {},
            'best_features_by_target': {},
            'correlation_analysis': {}
        }
        
        # Evaluate each target family
        for family_name, target_df in targets.items():
            logger.info(f"Evaluating features against {family_name} targets...")
            
            family_results = self._evaluate_target_family(X, target_df, feature_names, family_name)
            results['target_families'][family_name] = family_results
        
        # Generate multi-target summary
        results['multi_target_summary'] = self._generate_multi_target_summary(results)
        
        # Find best features by target
        results['best_features_by_target'] = self._find_best_features_by_target(results)
        
        # Correlation analysis
        results['correlation_analysis'] = self._analyze_target_correlations(targets)
        
        return results
    
    def _evaluate_target_family(self, X: pd.DataFrame, target_df: pd.DataFrame,
                               feature_names: List[str], family_name: str) -> Dict[str, Any]:
        """Evaluate features against a specific target family."""
        family_results = {
            'targets': list(target_df.columns),
            'feature_scores': {},
            'regression_metrics': {},
            'classification_metrics': {},
            'best_features': []
        }
        
        for target_name in target_df.columns:
            target_series = target_df[target_name].dropna()
            
            # Align features with target
            common_idx = X.index.intersection(target_series.index)
            X_aligned = X.loc[common_idx]
            y_aligned = target_series.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
            X_clean = X_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            if len(X_clean) < 100:  # Minimum sample requirement
                continue
            
            # Evaluate each feature
            target_scores = {}
            regression_metrics = {}
            classification_metrics = {}
            
            for feature_name in feature_names:
                if feature_name not in X_clean.columns:
                    continue
                
                feature_series = X_clean[feature_name].dropna()
                if len(feature_series) < 50:
                    continue
                
                # Align feature with target
                feature_aligned = feature_series.loc[feature_series.index.intersection(y_clean.index)]
                y_feature_aligned = y_clean.loc[feature_aligned.index]
                
                if len(feature_aligned) < 50:
                    continue
                
                # Calculate metrics
                feature_metrics = self._calculate_feature_target_metrics(
                    feature_aligned, y_feature_aligned, target_name
                )
                
                target_scores[feature_name] = feature_metrics['overall_score']
                regression_metrics[feature_name] = feature_metrics['regression_metrics']
                classification_metrics[feature_name] = feature_metrics['classification_metrics']
            
            # Store results for this target
            family_results['feature_scores'][target_name] = target_scores
            family_results['regression_metrics'][target_name] = regression_metrics
            family_results['classification_metrics'][target_name] = classification_metrics
            
            # Find best features for this target
            if target_scores:
                best_features = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                family_results['best_features'].extend([f[0] for f in best_features])
        
        return family_results
    
    def _calculate_feature_target_metrics(self, feature: pd.Series, target: pd.Series, 
                                        target_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for feature-target pair."""
        try:
            # Determine if target is regression or classification
            is_classification = target_name.endswith('_hit') or target_name.endswith('_cls') or target_name.endswith('_flag') or target_name.endswith('_label')
            
            metrics = {
                'regression_metrics': {},
                'classification_metrics': {},
                'overall_score': 0.0
            }
            
            if is_classification:
                # Classification metrics
                try:
                    # Remove NaN values
                    valid_mask = ~(feature.isna() | target.isna())
                    feature_clean = feature[valid_mask]
                    target_clean = target[valid_mask]
                    
                    if len(feature_clean) < 50:
                        return metrics
                    
                    # AUC
                    auc = roc_auc_score(target_clean, feature_clean)
                    metrics['classification_metrics']['auc'] = auc
                    
                    # F1 Score
                    threshold = np.median(feature_clean)
                    predictions = (feature_clean > threshold).astype(int)
                    f1 = f1_score(target_clean, predictions)
                    metrics['classification_metrics']['f1'] = f1
                    
                    # Brier Score
                    # Convert feature to probability using sigmoid
                    feature_prob = 1 / (1 + np.exp(-feature_clean))
                    brier = brier_score_loss(target_clean, feature_prob)
                    metrics['classification_metrics']['brier'] = brier
                    
                    # Log Loss
                    logloss = log_loss(target_clean, feature_prob)
                    metrics['classification_metrics']['logloss'] = logloss
                    
                    # PR-AUC
                    precision, recall, _ = precision_recall_curve(target_clean, feature_clean)
                    pr_auc = average_precision_score(target_clean, feature_clean)
                    metrics['classification_metrics']['pr_auc'] = pr_auc
                    
                    # Overall score for classification
                    metrics['overall_score'] = (auc + f1 + (1 - brier)) / 3
                    
                except Exception as e:
                    logger.warning(f"Error calculating classification metrics: {e}")
            
            else:
                # Regression metrics
                try:
                    # Remove NaN values
                    valid_mask = ~(feature.isna() | target.isna())
                    feature_clean = feature[valid_mask]
                    target_clean = target[valid_mask]
                    
                    if len(feature_clean) < 50:
                        return metrics
                    
                    # MAE
                    mae = mean_absolute_error(target_clean, feature_clean)
                    metrics['regression_metrics']['mae'] = mae
                    
                    # MSE
                    mse = mean_squared_error(target_clean, feature_clean)
                    metrics['regression_metrics']['mse'] = mse
                    
                    # RMSE
                    rmse = np.sqrt(mse)
                    metrics['regression_metrics']['rmse'] = rmse
                    
                    # R²
                    r2 = r2_score(target_clean, feature_clean)
                    metrics['regression_metrics']['r2'] = r2
                    
                    # Pearson correlation
                    correlation = np.corrcoef(feature_clean, target_clean)[0, 1]
                    metrics['regression_metrics']['pearson_corr'] = correlation
                    
                    # Spearman correlation
                    spearman_corr = stats.spearmanr(feature_clean, target_clean)[0]
                    metrics['regression_metrics']['spearman_corr'] = spearman_corr
                    
                    # MAPE (if target is positive)
                    if (target_clean > 0).all():
                        mape = np.mean(np.abs((target_clean - feature_clean) / target_clean)) * 100
                        metrics['regression_metrics']['mape'] = mape
                    
                    # Overall score for regression
                    metrics['overall_score'] = (abs(correlation) + abs(spearman_corr) + max(0, r2)) / 3
                    
                except Exception as e:
                    logger.warning(f"Error calculating regression metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating feature-target metrics: {e}")
            return {
                'regression_metrics': {},
                'classification_metrics': {},
                'overall_score': 0.0
            }
    
    def _generate_multi_target_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary across all targets."""
        summary = {
            'total_targets': 0,
            'total_features_evaluated': 0,
            'best_overall_features': [],
            'target_family_performance': {},
            'feature_consistency': {}
        }
        
        # Count targets and features
        all_features = set()
        target_count = 0
        
        for family_name, family_results in results['target_families'].items():
            summary['target_family_performance'][family_name] = {
                'targets': len(family_results['targets']),
                'features_evaluated': len(family_results['feature_scores']),
                'best_features': family_results['best_features'][:5]
            }
            
            target_count += len(family_results['targets'])
            
            # Collect all features
            for target_name, feature_scores in family_results['feature_scores'].items():
                all_features.update(feature_scores.keys())
        
        summary['total_targets'] = target_count
        summary['total_features_evaluated'] = len(all_features)
        
        # Find best overall features (average score across all targets)
        feature_scores = {}
        for family_name, family_results in results['target_families'].items():
            for target_name, scores in family_results['feature_scores'].items():
                for feature_name, score in scores.items():
                    if feature_name not in feature_scores:
                        feature_scores[feature_name] = []
                    feature_scores[feature_name].append(score)
        
        # Calculate average scores
        feature_avg_scores = {}
        for feature_name, scores in feature_scores.items():
            feature_avg_scores[feature_name] = np.mean(scores)
        
        # Sort by average score
        summary['best_overall_features'] = sorted(
            feature_avg_scores.items(), key=lambda x: x[1], reverse=True
        )[:20]
        
        # Calculate feature consistency (low variance across targets)
        feature_consistency = {}
        for feature_name, scores in feature_scores.items():
            if len(scores) > 1:
                consistency = 1 - (np.std(scores) / (np.mean(scores) + 1e-8))
                feature_consistency[feature_name] = consistency
        
        summary['feature_consistency'] = dict(sorted(
            feature_consistency.items(), key=lambda x: x[1], reverse=True
        )[:20])
        
        return summary
    
    def _find_best_features_by_target(self, results: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
        """Find best features for each target."""
        best_features = {}
        
        for family_name, family_results in results['target_families'].items():
            for target_name, feature_scores in family_results['feature_scores'].items():
                if feature_scores:
                    best_features[target_name] = sorted(
                        feature_scores.items(), key=lambda x: x[1], reverse=True
                    )[:10]
        
        return best_features
    
    def _analyze_target_correlations(self, targets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlations between targets."""
        correlation_analysis = {
            'target_correlations': {},
            'family_correlations': {},
            'highly_correlated_pairs': []
        }
        
        # Collect all targets
        all_targets = {}
        for family_name, target_df in targets.items():
            for target_name in target_df.columns:
                all_targets[f"{family_name}_{target_name}"] = target_df[target_name]
        
        # Calculate correlations
        target_names = list(all_targets.keys())
        correlation_matrix = np.zeros((len(target_names), len(target_names)))
        
        for i, target1 in enumerate(target_names):
            for j, target2 in enumerate(target_names):
                if i != j:
                    try:
                        # Align targets
                        common_idx = all_targets[target1].index.intersection(all_targets[target2].index)
                        if len(common_idx) > 100:
                            t1_aligned = all_targets[target1].loc[common_idx].dropna()
                            t2_aligned = all_targets[target2].loc[common_idx].dropna()
                            
                            if len(t1_aligned) > 50 and len(t2_aligned) > 50:
                                corr = np.corrcoef(t1_aligned, t2_aligned)[0, 1]
                                correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
                    except:
                        pass
        
        # Store correlation matrix
        correlation_analysis['target_correlations'] = {
            'targets': target_names,
            'correlation_matrix': correlation_matrix.tolist()
        }
        
        # Find highly correlated pairs
        for i in range(len(target_names)):
            for j in range(i + 1, len(target_names)):
                if abs(correlation_matrix[i, j]) > 0.8:
                    correlation_analysis['highly_correlated_pairs'].append({
                        'target1': target_names[i],
                        'target2': target_names[j],
                        'correlation': correlation_matrix[i, j]
                    })
        
        return correlation_analysis
    
    def generate_multi_target_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive multi-target report."""
        report = []
        report.append("# Multi-Target Feature Evaluation Report")
        report.append("=" * 60)
        report.append(f"Timeframe: {self.timeframe_minutes} minutes")
        report.append(f"Horizons: {self.horizons} bars")
        report.append("")
        
        # Summary statistics
        summary = results['multi_target_summary']
        report.append("## Summary Statistics")
        report.append(f"- Total targets: {summary['total_targets']}")
        report.append(f"- Total features evaluated: {summary['total_features_evaluated']}")
        report.append("")
        
        # Target family performance
        report.append("## Target Family Performance")
        for family_name, perf in summary['target_family_performance'].items():
            report.append(f"### {family_name.replace('_', ' ').title()}")
            report.append(f"- Targets: {perf['targets']}")
            report.append(f"- Features evaluated: {perf['features_evaluated']}")
            report.append(f"- Best features: {', '.join(perf['best_features'][:3])}")
            report.append("")
        
        # Best overall features
        report.append("## Best Overall Features")
        for i, (feature, score) in enumerate(summary['best_overall_features'][:10], 1):
            report.append(f"{i:2d}. {feature:30s} | Score: {score:.4f}")
        report.append("")
        
        # Feature consistency
        report.append("## Most Consistent Features")
        for i, (feature, consistency) in enumerate(summary['feature_consistency'][:10], 1):
            report.append(f"{i:2d}. {feature:30s} | Consistency: {consistency:.4f}")
        report.append("")
        
        # Best features by target
        report.append("## Best Features by Target")
        for target_name, best_features in results['best_features_by_target'].items():
            if best_features:
                report.append(f"### {target_name}")
                for i, (feature, score) in enumerate(best_features[:5], 1):
                    report.append(f"{i}. {feature:30s} | Score: {score:.4f}")
                report.append("")
        
        # Highly correlated targets
        if results['correlation_analysis']['highly_correlated_pairs']:
            report.append("## Highly Correlated Target Pairs")
            for pair in results['correlation_analysis']['highly_correlated_pairs'][:10]:
                report.append(f"- {pair['target1']} ↔ {pair['target2']}: {pair['correlation']:.3f}")
            report.append("")
        
        return "\n".join(report)
    
    def run_complete_evaluation_with_data_loading(self, 
                                                X: pd.DataFrame,
                                                symbol: str = "ETHUSDT",
                                                interval: str = "15m",
                                                start_date: Optional[datetime] = None,
                                                end_date: Optional[datetime] = None,
                                                data_type: str = "raw",
                                                fallback_days: int = 30,
                                                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete multi-target evaluation with automatic data loading.
        
        Args:
            X: Feature matrix
            symbol: Trading symbol for data loading
            interval: Data interval for data loading
            start_date: Start date for data loading
            end_date: End date for data loading
            data_type: 'raw' or 'processed' for data loading
            fallback_days: Days to fallback to if no data in range
            feature_names: List of feature names to evaluate
            
        Returns:
            Complete evaluation results
        """
        logger.info("Running complete multi-target evaluation with data loading...")
        
        # Load market data
        data = self.load_market_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type,
            fallback_days=fallback_days
        )
        
        if data is None:
            logger.error("Failed to load market data")
            return {
                'error': 'Failed to load market data',
                'target_families': {},
                'feature_performance': {},
                'multi_target_summary': {},
                'best_features_by_target': {},
                'correlation_analysis': {}
            }
        
        # Create all targets
        targets = self.create_all_targets(data)
        
        # Evaluate features against targets
        results = self.evaluate_features_against_targets(X, targets, feature_names)
        
        # Add data information to results
        results['data_info'] = {
            'symbol': symbol,
            'interval': interval,
            'data_type': data_type,
            'date_range': (data.index.min(), data.index.max()),
            'n_records': len(data),
            'n_targets': sum(len(target_df.columns) for target_df in targets.values())
        }
        
        logger.info(f"Complete evaluation finished with {results['data_info']['n_targets']} targets")
        
        return results