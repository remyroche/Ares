"""
Unified Data-Driven Feature Pipeline

Main orchestrator that coordinates all aspects of feature engineering:
- Period optimization
- Feature generation
- Interaction discovery
- Feature selection
- Performance monitoring

Uses Purged & Embargoed Walk-Forward CV to prevent leakage and overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import components
from .config import UnifiedPipelineConfig, create_default_config
from ..time_series_cv import PurgedEmbargoedWalkForwardCV, create_purged_embargoed_cv
from ..statistical_analysis import StatisticalAnalysisFramework
from ..feature_selection.multi_objective_selector import (
    MultiObjectiveFeatureSelector, 
    create_default_objectives,
    OutOfSampleSharpeObjective,
    DrawdownObjective,
    TurnoverObjective,
    StabilityObjective,
    DiversityObjective,
    MutualInformationObjective,
    ProfitCenteredObjective
)

# Import VectorBT utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    tprint_warning("VectorBT utilities not available, using fallback implementations")

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineResult:
    """Result of the unified feature pipeline."""
    
    # Selected features
    selected_features: List[str]
    feature_importance: Dict[str, float]
    
    # Objective values
    objective_values: Dict[str, float]
    
    # Pipeline metadata
    processing_time: float
    n_cv_splits: int
    n_candidates_evaluated: int
    
    # Performance metrics
    out_of_sample_sharpe: float
    max_drawdown: float
    stability_score: float
    diversity_score: float
    
    # Configuration used
    config: UnifiedPipelineConfig
    
    # Intermediate results
    period_optimization_result: Optional[Dict[str, Any]] = None
    lookback_optimization_result: Optional[Dict[str, Any]] = None
    interaction_generation_result: Optional[Dict[str, Any]] = None
    htf_interaction_result: Optional[Dict[str, Any]] = None
    feature_selection_result: Optional[Dict[str, Any]] = None


class UnifiedDataDrivenPipeline:
    """
    Main orchestrator for unified data-driven feature generation and selection.
    
    This class coordinates all aspects of feature engineering using a completely
    data-driven approach with strict time series validation to prevent leakage.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        """
        Initialize the unified data-driven pipeline.
        
        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or create_default_config()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        tprint_info("Unified Data-Driven Pipeline initialized")
        tprint_info(f"Configuration: {self.config}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        tprint_debug("Initializing pipeline components")
        
        # Statistical analysis framework
        self.stats_framework = StatisticalAnalysisFramework()
        
        # Time series CV
        self.cv_splitter = create_purged_embargoed_cv(
            n_splits=self.config.feature_selection.cv_config.n_splits,
            test_size=self.config.feature_selection.cv_config.test_size,
            train_size=self.config.feature_selection.cv_config.train_size,
            purge_fraction=self.config.feature_selection.cv_config.purge_fraction,
            embargo_fraction=self.config.feature_selection.cv_config.embargo_fraction
        )
        
        # VectorBT utilities
        if VECTORBT_AVAILABLE:
            self.rolling_optimizer = VectorBTRollingOptimizer()
            self.vectorization_manager = UnifiedVectorizationManager()
        else:
            self.rolling_optimizer = None
            self.vectorization_manager = None
            tprint_warning("VectorBT not available, using fallback implementations")
        
        # Multi-objective feature selector
        objectives = self._create_objectives()
        self.feature_selector = MultiObjectiveFeatureSelector(
            objectives=objectives,
            weights=self.config.feature_selection.multi_objective.objectives,
            max_features=self.config.feature_selection.multi_objective.max_features,
            min_features=self.config.feature_selection.multi_objective.min_features
        )
        
        tprint_success("Pipeline components initialized")
    
    def _create_objectives(self) -> List[Any]:
        """Create objective functions based on configuration."""
        objectives = []
        
        # Out-of-sample Sharpe ratio
        if 'out_of_sample_sharpe' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(OutOfSampleSharpeObjective())
        
        # Drawdown
        if 'drawdown' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(DrawdownObjective())
        
        # Turnover
        if 'turnover' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(TurnoverObjective())
        
        # Stability
        if 'stability' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(StabilityObjective())
        
        # Diversity
        if 'diversity' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(DiversityObjective(
                method=self.config.feature_selection.multi_objective.diversity_method
            ))
        
        # Mutual Information
        if 'mutual_information' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(MutualInformationObjective())
        
        # Profit-centered
        if 'profit_centered' in self.config.feature_selection.multi_objective.objectives:
            objectives.append(ProfitCenteredObjective())
        
        return objectives
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking."""
        self.performance_stats = {
            'total_processing_time': 0.0,
            'period_optimization_time': 0.0,
            'lookback_optimization_time': 0.0,
            'interaction_generation_time': 0.0,
            'htf_interaction_time': 0.0,
            'feature_selection_time': 0.0,
            'n_cv_splits': 0,
            'n_candidates_evaluated': 0,
            'memory_usage_mb': 0.0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0
        }
    
    def process(self, data: pd.DataFrame, 
                targets: Optional[pd.Series] = None,
                feature_columns: Optional[List[str]] = None) -> FeaturePipelineResult:
        """
        Main processing pipeline.
        
        Args:
            data: Input data with features
            targets: Target variable (returns, prices, etc.)
            feature_columns: Optional list of feature columns to use
            
        Returns:
            FeaturePipelineResult with selected features and performance metrics
        """
        tprint_info(f"Starting unified data-driven pipeline processing")
        tprint_info(f"Data shape: {data.shape}, Targets: {targets is not None}")
        
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(data, targets, feature_columns)
        
        # Prepare data
        processed_data, processed_targets = self._prepare_data(data, targets, feature_columns)
        
        # Analyze data characteristics
        tprint_info("Analyzing data characteristics")
        data_characteristics = self.stats_framework.analyze_data_characteristics(processed_data)
        
        # Detect patterns
        tprint_info("Detecting patterns in data")
        pattern_analysis = self.stats_framework.detect_patterns(processed_data)
        
        # Generate time series splits
        tprint_info("Generating time series splits")
        cv_splits = self.cv_splitter.split(processed_data, targets=processed_targets)
        self.performance_stats['n_cv_splits'] = len(cv_splits)
        
        # Validate no leakage
        if self.config.feature_selection.cv_config.check_leakage:
            tprint_info("Validating no leakage in splits")
            is_valid = self.cv_splitter.validate_no_leakage(processed_data)
            if not is_valid:
                tprint_error("Leakage detected in time series splits")
                raise ValueError("Leakage detected in time series splits")
        
        # Period optimization (if enabled)
        period_result = None
        if self.config.enable_period_optimization:
            tprint_info("Optimizing periods")
            period_start = time.time()
            period_result = self._optimize_periods(processed_data, data_characteristics)
            self.performance_stats['period_optimization_time'] = time.time() - period_start
        
        # Feature lookback optimization (if enabled)
        lookback_result = None
        if self.config.enable_feature_lookback_optimization:
            tprint_info("Optimizing feature lookback periods")
            lookback_start = time.time()
            lookback_result = self._optimize_feature_lookback(processed_data, processed_targets, data_characteristics)
            self.performance_stats['lookback_optimization_time'] = time.time() - lookback_start
        
        # Interaction generation (if enabled)
        interaction_result = None
        if self.config.enable_interaction_generation:
            tprint_info("Generating interactions")
            interaction_start = time.time()
            interaction_result = self._generate_interactions(processed_data, processed_targets, pattern_analysis)
            self.performance_stats['interaction_generation_time'] = time.time() - interaction_start
        
        # HTF-aware interaction generation (if enabled)
        htf_interaction_result = None
        if self.config.enable_htf_interactions:
            tprint_info("Generating HTF-aware interactions")
            htf_start = time.time()
            htf_interaction_result = self._generate_htf_interactions(processed_data, processed_targets, pattern_analysis)
            self.performance_stats['htf_interaction_time'] = time.time() - htf_start
        
        # Feature selection
        tprint_info("Selecting features using multi-objective optimization")
        selection_start = time.time()
        selection_result = self._select_features(processed_data, processed_targets, cv_splits)
        self.performance_stats['feature_selection_time'] = time.time() - selection_start
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(selection_result, processed_data, processed_targets)
        
        # Create result
        total_time = time.time() - start_time
        self.performance_stats['total_processing_time'] = total_time
        
        result = FeaturePipelineResult(
            selected_features=selection_result.selected_features,
            feature_importance=final_metrics['feature_importance'],
            objective_values=selection_result.objective_values,
            processing_time=total_time,
            n_cv_splits=len(cv_splits),
            n_candidates_evaluated=selection_result.optimization_metadata.get('n_candidates', 0),
            out_of_sample_sharpe=final_metrics['out_of_sample_sharpe'],
            max_drawdown=final_metrics['max_drawdown'],
            stability_score=final_metrics['stability_score'],
            diversity_score=final_metrics['diversity_score'],
            config=self.config,
            period_optimization_result=period_result,
            lookback_optimization_result=lookback_result,
            interaction_generation_result=interaction_result,
            htf_interaction_result=htf_interaction_result,
            feature_selection_result=selection_result.optimization_metadata
        )
        
        tprint_success(f"Pipeline processing completed in {total_time:.2f}s")
        tprint_success(f"Selected {len(selection_result.selected_features)} features")
        tprint_success(f"Out-of-sample Sharpe: {final_metrics['out_of_sample_sharpe']:.3f}")
        
        return result
    
    def _validate_inputs(self, data: pd.DataFrame, 
                        targets: Optional[pd.Series], 
                        feature_columns: Optional[List[str]]):
        """Validate input data."""
        tprint_debug("Validating inputs")
        
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if targets is not None:
            if not isinstance(targets, pd.Series):
                raise TypeError("Targets must be a pandas Series")
            
            if len(targets) != len(data):
                raise ValueError("Targets length must match data length")
        
        if feature_columns is not None:
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
        
        tprint_success("Input validation passed")
    
    def _prepare_data(self, data: pd.DataFrame, 
                     targets: Optional[pd.Series], 
                     feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for processing."""
        tprint_debug("Preparing data")
        
        # Select feature columns
        if feature_columns is not None:
            processed_data = data[feature_columns].copy()
        else:
            processed_data = data.copy()
        
        # Handle missing values
        if processed_data.isna().any().any():
            tprint_warning("Missing values detected, filling with forward fill")
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # Align targets with data
        processed_targets = None
        if targets is not None:
            processed_targets = targets.copy()
            
            # Align indices
            common_idx = processed_data.index.intersection(processed_targets.index)
            processed_data = processed_data.loc[common_idx]
            processed_targets = processed_targets.loc[common_idx]
        
        tprint_success(f"Data prepared: {processed_data.shape}")
        return processed_data, processed_targets
    
    def _optimize_periods(self, data: pd.DataFrame, 
                         characteristics: Any) -> Dict[str, Any]:
        """Optimize periods for different feature types using data-driven approach."""
        tprint_debug("Starting data-driven period optimization")
        
        # Configuration for period optimization
        timeframe_period_ranges = {
            "5m": (1, 100),   # 5m to 8.3 hours
            "15m": (1, 50),   # 15m to 12.5 hours
            "1h": (1, 24),    # 1h to 1 day
            "4h": (1, 12),    # 4h to 2 days
        }
        
        optimized_periods = {}
        confidence_scores = {}
        optimization_methods = {}
        
        # Analyze each timeframe
        for timeframe, (min_period, max_period) in timeframe_period_ranges.items():
            tprint_debug(f"Optimizing periods for {timeframe} (range: {min_period}-{max_period})")
            
            # Statistical analysis for period selection
            period_scores = self._analyze_periods_statistically(data, min_period, max_period)
            
            # Economic significance evaluation
            economic_scores = self._evaluate_economic_significance(data, period_scores, timeframe)
            
            # Combine statistical and economic analysis
            combined_scores = self._combine_period_scores(period_scores, economic_scores)
            
            # Select optimal periods
            optimal_periods = self._select_optimal_periods(combined_scores, max_periods=8)
            
            optimized_periods[timeframe] = optimal_periods
            confidence_scores[timeframe] = combined_scores
            optimization_methods[timeframe] = 'statistical_economic_combined'
            
            tprint_success(f"Selected {len(optimal_periods)} optimal periods for {timeframe}")
        
        result = {
            'optimized_periods': optimized_periods,
            'optimization_method': 'data_driven_statistical_economic',
            'confidence_scores': confidence_scores,
            'timeframe_ranges': timeframe_period_ranges,
            'optimization_methods': optimization_methods
        }
        
        tprint_success("Data-driven period optimization completed")
        return result
    
    def _analyze_periods_statistically(self, data: pd.DataFrame, min_period: int, max_period: int) -> Dict[int, float]:
        """Analyze periods using statistical methods."""
        period_scores = {}
        
        # Price-based features for analysis
        if 'close' not in data.columns:
            tprint_warning("No 'close' column found for period analysis")
            return period_scores
        
        close_prices = data['close'].dropna()
        
        for period in range(min_period, max_period + 1):
            try:
                # Calculate rolling statistics
                rolling_mean = close_prices.rolling(window=period).mean()
                rolling_std = close_prices.rolling(window=period).std()
                
                # Calculate period-specific metrics
                volatility = rolling_std.std()  # Stability of volatility
                trend_strength = abs(rolling_mean.diff().mean())  # Average trend strength
                data_quality = 1.0 - (rolling_mean.isna().sum() / len(rolling_mean))  # Data completeness
                
                # Combine metrics into score
                score = (volatility * 0.4 + trend_strength * 0.3 + data_quality * 0.3)
                period_scores[period] = score
                
            except Exception as e:
                tprint_debug(f"Error analyzing period {period}: {e}")
                period_scores[period] = 0.0
        
        return period_scores
    
    def _evaluate_economic_significance(self, data: pd.DataFrame, period_scores: Dict[int, float], timeframe: str) -> Dict[int, float]:
        """Evaluate economic significance of periods through backtesting."""
        economic_scores = {}
        
        if 'close' not in data.columns:
            return economic_scores
        
        close_prices = data['close'].dropna()
        
        # Calculate returns for backtesting
        returns = close_prices.pct_change().dropna()
        
        for period, stat_score in period_scores.items():
            try:
                # Simple backtesting: use period-based signals
                signals = self._generate_period_signals(close_prices, period)
                
                if len(signals) == 0:
                    economic_scores[period] = 0.0
                    continue
                
                # Calculate performance metrics
                strategy_returns = signals.shift(1) * returns  # Lagged signals
                strategy_returns = strategy_returns.dropna()
                
                if len(strategy_returns) < 20:  # Need minimum data
                    economic_scores[period] = 0.0
                    continue
                
                # Calculate Sharpe ratio
                sharpe_ratio = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
                
                # Calculate max drawdown
                cumulative_returns = (1 + strategy_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min())
                
                # Calculate win rate
                win_rate = (strategy_returns > 0).mean()
                
                # Combine economic metrics
                economic_score = (
                    max(0, sharpe_ratio) * 0.4 +
                    max(0, 1 - max_drawdown) * 0.3 +
                    win_rate * 0.3
                )
                
                economic_scores[period] = min(1.0, max(0.0, economic_score))
                
            except Exception as e:
                tprint_debug(f"Error evaluating economic significance for period {period}: {e}")
                economic_scores[period] = 0.0
        
        return economic_scores
    
    def _generate_period_signals(self, prices: pd.Series, period: int) -> pd.Series:
        """Generate trading signals based on period."""
        try:
            # Simple momentum signal
            sma = prices.rolling(window=period).mean()
            signals = np.where(prices > sma, 1, -1)
            return pd.Series(signals, index=prices.index)
        except:
            return pd.Series(dtype=float)
    
    def _combine_period_scores(self, stat_scores: Dict[int, float], econ_scores: Dict[int, float]) -> Dict[int, float]:
        """Combine statistical and economic scores."""
        combined_scores = {}
        
        # Normalize scores to 0-1 range
        stat_values = list(stat_scores.values())
        econ_values = list(econ_scores.values())
        
        if stat_values:
            stat_max = max(stat_values)
            stat_min = min(stat_values)
            stat_range = stat_max - stat_min if stat_max > stat_min else 1.0
        else:
            stat_range = 1.0
        
        if econ_values:
            econ_max = max(econ_values)
            econ_min = min(econ_values)
            econ_range = econ_max - econ_min if econ_max > econ_min else 1.0
        else:
            econ_range = 1.0
        
        # Combine with weights
        statistical_weight = 0.4
        economic_weight = 0.6
        
        for period in stat_scores:
            stat_score = stat_scores.get(period, 0.0)
            econ_score = econ_scores.get(period, 0.0)
            
            # Normalize
            norm_stat = (stat_score - stat_min) / stat_range if stat_range > 0 else 0.0
            norm_econ = (econ_score - econ_min) / econ_range if econ_range > 0 else 0.0
            
            # Combine
            combined_score = norm_stat * statistical_weight + norm_econ * economic_weight
            combined_scores[period] = combined_score
        
        return combined_scores
    
    def _select_optimal_periods(self, combined_scores: Dict[int, float], max_periods: int = 8) -> List[int]:
        """Select optimal periods based on combined scores."""
        if not combined_scores:
            return []
        
        # Sort by score (descending)
        sorted_periods = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top periods
        optimal_periods = [period for period, score in sorted_periods[:max_periods] if score > 0.1]
        
        return optimal_periods
    
    def _optimize_feature_lookback(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                  characteristics: Any) -> Dict[str, Any]:
        """Optimize feature lookback periods using data-driven approach."""
        tprint_debug("Starting feature lookback optimization")
        
        # Configuration for lookback optimization
        lookback_config = {
            'min_lookback': 5,
            'max_lookback': 100,
            'step_size': 5,
            'min_samples': 20,
            'walk_forward_splits': 3,
            'min_train_ratio': 0.4
        }
        
        optimized_lookbacks = {}
        optimization_metrics = {}
        
        # Get feature columns for optimization
        feature_columns = [col for col in data.columns if col not in ['close', 'open', 'high', 'low', 'volume']]
        
        if not feature_columns:
            tprint_warning("No feature columns found for lookback optimization")
            return {'optimized_lookbacks': {}, 'optimization_metrics': {}}
        
        tprint_debug(f"Optimizing lookback for {len(feature_columns)} features")
        
        # Optimize each feature individually
        for feature in feature_columns[:10]:  # Limit to first 10 features for performance
            try:
                tprint_debug(f"Optimizing lookback for feature: {feature}")
                
                # Generate walk-forward splits
                splits = self._generate_walk_forward_splits(data, lookback_config['walk_forward_splits'])
                
                # Test different lookback periods
                lookback_scores = {}
                for lookback in range(lookback_config['min_lookback'], 
                                   lookback_config['max_lookback'] + 1, 
                                   lookback_config['step_size']):
                    
                    score = self._evaluate_lookback_period(
                        data, feature, lookback, targets, splits
                    )
                    lookback_scores[lookback] = score
                
                # Select optimal lookback
                if lookback_scores:
                    optimal_lookback = max(lookback_scores.items(), key=lambda x: x[1])
                    optimized_lookbacks[feature] = optimal_lookback[0]
                    optimization_metrics[feature] = {
                        'best_score': optimal_lookback[1],
                        'all_scores': lookback_scores,
                        'optimization_method': 'walk_forward_validation'
                    }
                
            except Exception as e:
                tprint_debug(f"Error optimizing lookback for {feature}: {e}")
                continue
        
        result = {
            'optimized_lookbacks': optimized_lookbacks,
            'optimization_metrics': optimization_metrics,
            'optimization_method': 'data_driven_walk_forward',
            'config_used': lookback_config
        }
        
        tprint_success(f"Feature lookback optimization completed: {len(optimized_lookbacks)} features optimized")
        return result
    
    def _generate_walk_forward_splits(self, data: pd.DataFrame, n_splits: int) -> List[Dict[str, Any]]:
        """Generate walk-forward validation splits."""
        splits = []
        data_length = len(data)
        
        for i in range(n_splits):
            # Calculate split boundaries
            train_start = 0
            train_end = int(data_length * (0.6 + i * 0.1))  # Increasing train size
            test_start = train_end
            test_end = int(data_length * (0.8 + i * 0.1))  # Increasing test size
            
            if test_end > data_length:
                test_end = data_length
            
            if train_end - train_start < 20 or test_end - test_start < 10:
                continue  # Skip if too small
            
            splits.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_indices': data.index[train_start:train_end],
                'test_indices': data.index[test_start:test_end]
            })
        
        return splits
    
    def _evaluate_lookback_period(self, data: pd.DataFrame, feature: str, lookback: int, 
                                 targets: Optional[pd.Series], splits: List[Dict[str, Any]]) -> float:
        """Evaluate a specific lookback period using walk-forward validation."""
        if feature not in data.columns:
            return 0.0
        
        scores = []
        
        for split in splits:
            try:
                # Get train and test data
                train_data = data.loc[split['train_indices']]
                test_data = data.loc[split['test_indices']]
                
                if len(train_data) < lookback or len(test_data) < 5:
                    continue
                
                # Calculate feature with lookback
                feature_series = data[feature].dropna()
                if len(feature_series) < lookback:
                    continue
                
                # Create lookback feature
                lookback_feature = feature_series.rolling(window=lookback).mean()
                
                # Align with train/test data
                train_feature = lookback_feature.loc[split['train_indices']].dropna()
                test_feature = lookback_feature.loc[split['test_indices']].dropna()
                
                if len(train_feature) < 10 or len(test_feature) < 5:
                    continue
                
                # Calculate performance metrics
                if targets is not None:
                    train_targets = targets.loc[train_feature.index]
                    test_targets = targets.loc[test_feature.index]
                    
                    # Calculate correlation on training data
                    train_corr = train_feature.corr(train_targets)
                    if np.isnan(train_corr):
                        continue
                    
                    # Calculate out-of-sample correlation
                    test_corr = test_feature.corr(test_targets)
                    if np.isnan(test_corr):
                        continue
                    
                    # Combine in-sample and out-of-sample performance
                    score = abs(train_corr) * 0.3 + abs(test_corr) * 0.7
                else:
                    # Use variance and stability as metrics
                    train_var = train_feature.var()
                    test_var = test_feature.var()
                    
                    if train_var == 0 or test_var == 0:
                        continue
                    
                    # Prefer features with good variance and stability
                    variance_score = min(1.0, train_var)
                    stability_score = 1.0 - abs(train_var - test_var) / (train_var + test_var + 1e-8)
                    
                    score = variance_score * 0.6 + stability_score * 0.4
                
                scores.append(score)
                
            except Exception as e:
                tprint_debug(f"Error evaluating lookback {lookback} for {feature}: {e}")
                continue
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_interactions(self, data: pd.DataFrame, 
                             targets: Optional[pd.Series], 
                             patterns: Any) -> Dict[str, Any]:
        """Generate interaction features using data-driven approach."""
        tprint_debug("Starting data-driven interaction generation")
        
        # Step 1: Intelligent feature pre-selection
        tprint_debug("Step 1: Pre-selecting features from full feature bank")
        selected_features = self._pre_select_features(data, targets, target_count=40)
        
        # Step 2: Generate features for selected feature set
        tprint_debug("Step 2: Generating features for selected feature set")
        feature_df = self._generate_selected_features(data, selected_features)
        
        # Step 3: Generate interactions between selected features
        tprint_debug("Step 3: Generating interactions between selected features")
        interactions = self._generate_feature_interactions(feature_df, targets)
        
        # Step 4: Apply quality filtering and ranking
        tprint_debug("Step 4: Applying quality filtering and ranking")
        filtered_interactions = self._filter_and_rank_interactions(interactions, targets)
        
        result = {
            'generated_interactions': filtered_interactions,
            'interaction_types': list(set(i.get('interaction_type', 'unknown') for i in filtered_interactions)),
            'utility_scores': {i['name']: i['utility_score'] for i in filtered_interactions},
            'selected_features': selected_features,
            'feature_generation_metrics': self._calculate_feature_metrics(feature_df, filtered_interactions)
        }
        
        tprint_success(f"Interaction generation completed: {len(filtered_interactions)} interactions generated")
        return result
    
    def _pre_select_features(self, data: pd.DataFrame, targets: Optional[pd.Series], target_count: int = 40) -> List[Dict[str, Any]]:
        """Pre-select features from full feature bank using data-driven approach."""
        tprint_debug(f"Pre-selecting {target_count} features from {len(data.columns)} available features")
        
        # Categorize features by type
        feature_categories = self._categorize_features(data.columns)
        
        # Calculate feature scores for each category
        feature_scores = {}
        for category, features in feature_categories.items():
            tprint_debug(f"Evaluating {len(features)} features in category: {category}")
            category_scores = self._evaluate_features_in_category(data, features, targets, category)
            feature_scores.update(category_scores)
        
        # Select features ensuring diversity across categories
        selected_features = self._select_diverse_features(feature_scores, target_count)
        
        tprint_success(f"Selected {len(selected_features)} features across {len(set(f['category'] for f in selected_features))} categories")
        return selected_features
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type based on naming patterns."""
        categories = {
            'momentum': [],
            'volatility': [],
            'trend': [],
            'oscillator': [],
            'volume': [],
            'returns': [],
            'cross_timeframe': [],
            'microstructure': [],
            'entropy': [],
            'support_resistance': [],
            'candlestick_pattern': [],
            'time': [],
            'order_flow': [],
            'regime': [],
            'acceleration': [],
            'advanced_statistical': [],
            'spectral_wavelet': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            # Categorize based on naming patterns
            if any(x in feature_lower for x in ['mom', 'momentum', 'signal', 'alpha']):
                categories['momentum'].append(feature)
            elif any(x in feature_lower for x in ['vol', 'sigma', 'rv', 'gk', 'volatility']):
                categories['volatility'].append(feature)
            elif any(x in feature_lower for x in ['trend', 'ema', 'sma', 'macd']):
                categories['trend'].append(feature)
            elif any(x in feature_lower for x in ['rsi', 'stoch', 'osc', 'oscillator']):
                categories['oscillator'].append(feature)
            elif 'volume' in feature_lower:
                categories['volume'].append(feature)
            elif any(x in feature_lower for x in ['return', 'ret', 'pct_change']):
                categories['returns'].append(feature)
            elif any(x in feature_lower for x in ['cross', 'htf', 'timeframe']):
                categories['cross_timeframe'].append(feature)
            elif any(x in feature_lower for x in ['micro', 'tick', 'bid', 'ask']):
                categories['microstructure'].append(feature)
            elif any(x in feature_lower for x in ['entropy', 'ent']):
                categories['entropy'].append(feature)
            elif any(x in feature_lower for x in ['support', 'resistance', 's/r']):
                categories['support_resistance'].append(feature)
            elif any(x in feature_lower for x in ['candle', 'pattern', 'doji', 'hammer']):
                categories['candlestick_pattern'].append(feature)
            elif any(x in feature_lower for x in ['time', 'hour', 'day', 'session']):
                categories['time'].append(feature)
            elif any(x in feature_lower for x in ['order', 'flow', 'book']):
                categories['order_flow'].append(feature)
            elif any(x in feature_lower for x in ['regime', 'state']):
                categories['regime'].append(feature)
            elif any(x in feature_lower for x in ['accel', 'acceleration']):
                categories['acceleration'].append(feature)
            elif any(x in feature_lower for x in ['stat', 'statistical', 'advanced']):
                categories['advanced_statistical'].append(feature)
            elif any(x in feature_lower for x in ['spectral', 'wavelet', 'fourier']):
                categories['spectral_wavelet'].append(feature)
            else:
                # Default to momentum for uncategorized features
                categories['momentum'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _evaluate_features_in_category(self, data: pd.DataFrame, features: List[str], 
                                     targets: Optional[pd.Series], category: str) -> Dict[str, Dict[str, Any]]:
        """Evaluate features within a category."""
        feature_scores = {}
        
        for feature in features:
            if feature not in data.columns:
                continue
                
            try:
                feature_series = data[feature].dropna()
                
                if len(feature_series) < 10:  # Need minimum data
                    continue
                
                # Calculate feature metrics
                variance = feature_series.var()
                if variance == 0 or np.isnan(variance):
                    continue
                
                # Calculate correlation with target
                correlation_with_target = 0.0
                if targets is not None and len(targets) == len(feature_series):
                    try:
                        correlation = feature_series.corr(targets)
                        correlation_with_target = abs(correlation) if not np.isnan(correlation) else 0.0
                    except:
                        correlation_with_target = 0.0
                
                # Calculate information content (simplified)
                information_content = min(1.0, variance / (feature_series.std() + 1e-8))
                
                # Calculate uniqueness score (inverse correlation with other features)
                uniqueness_score = self._calculate_uniqueness_score(feature_series, data)
                
                # Calculate overall score
                overall_score = (
                    correlation_with_target * 0.4 +
                    information_content * 0.3 +
                    uniqueness_score * 0.3
                )
                
                feature_scores[feature] = {
                    'feature_name': feature,
                    'category': category,
                    'score': overall_score,
                    'variance': variance,
                    'correlation_with_target': correlation_with_target,
                    'information_content': information_content,
                    'uniqueness_score': uniqueness_score
                }
                
            except Exception as e:
                tprint_debug(f"Error evaluating feature {feature}: {e}")
                continue
        
        return feature_scores
    
    def _calculate_uniqueness_score(self, feature_series: pd.Series, data: pd.DataFrame) -> float:
        """Calculate uniqueness score based on correlation with other features."""
        try:
            correlations = []
            for col in data.columns:
                if col != feature_series.name:
                    try:
                        other_series = data[col].dropna()
                        if len(other_series) > 0:
                            corr = feature_series.corr(other_series)
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    except:
                        continue
            
            if not correlations:
                return 1.0
            
            # Uniqueness is inverse of average correlation
            avg_correlation = np.mean(correlations)
            return max(0.0, 1.0 - avg_correlation)
            
        except:
            return 0.5  # Default uniqueness score
    
    def _select_diverse_features(self, feature_scores: Dict[str, Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Select diverse features ensuring representation across categories."""
        # Group features by category
        category_features = {}
        for feature, scores in feature_scores.items():
            category = scores['category']
            if category not in category_features:
                category_features[category] = []
            category_features[category].append((feature, scores))
        
        # Sort features within each category by score
        for category in category_features:
            category_features[category].sort(key=lambda x: x[1]['score'], reverse=True)
        
        # Select features ensuring diversity
        selected_features = []
        min_per_category = 2
        max_per_category = 4
        
        # First pass: ensure minimum representation
        for category, features in category_features.items():
            if features:
                selected_count = min(min_per_category, len(features), target_count - len(selected_features))
                for i in range(selected_count):
                    selected_features.append(features[i][1])
        
        # Second pass: fill remaining slots
        remaining_slots = target_count - len(selected_features)
        if remaining_slots > 0:
            # Collect all remaining features
            all_remaining = []
            for category, features in category_features.items():
                already_selected = len([f for f in selected_features if f['category'] == category])
                remaining = features[already_selected:already_selected + max_per_category - already_selected]
                all_remaining.extend(remaining)
            
            # Sort by score and select
            all_remaining.sort(key=lambda x: x[1]['score'], reverse=True)
            for feature, scores in all_remaining[:remaining_slots]:
                selected_features.append(scores)
        
        return selected_features
    
    def _generate_selected_features(self, data: pd.DataFrame, selected_features: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features for the selected feature set."""
        tprint_debug(f"Generating features for {len(selected_features)} selected features")
        
        feature_df = pd.DataFrame(index=data.index)
        
        for feature_info in selected_features:
            feature_name = feature_info['feature_name']
            category = feature_info['category']
            
            if feature_name in data.columns:
                # Use existing feature
                feature_df[feature_name] = data[feature_name]
            else:
                # Generate feature based on category and name
                generated_feature = self._generate_feature_by_category(data, feature_name, category)
                if generated_feature is not None:
                    feature_df[feature_name] = generated_feature
        
        tprint_success(f"Generated feature DataFrame with shape: {feature_df.shape}")
        return feature_df
    
    def _generate_feature_by_category(self, data: pd.DataFrame, feature_name: str, category: str) -> Optional[pd.Series]:
        """Generate a feature based on its category and name."""
        try:
            if 'close' not in data.columns:
                return None
            
            close_prices = data['close']
            
            if category == 'momentum':
                if 'rsi' in feature_name.lower():
                    return self._calculate_rsi(close_prices, 14)
                elif 'stoch' in feature_name.lower():
                    return self._calculate_stochastic(close_prices, 14)
                else:
                    return close_prices.pct_change(14)  # 14-period momentum
            
            elif category == 'volatility':
                if 'vol' in feature_name.lower():
                    period = 20
                    if any(x in feature_name.lower() for x in ['10', 'ten']):
                        period = 10
                    elif any(x in feature_name.lower() for x in ['30', 'thirty']):
                        period = 30
                    return close_prices.rolling(period).std()
            
            elif category == 'trend':
                if 'sma' in feature_name.lower():
                    period = 20
                    if any(x in feature_name.lower() for x in ['10', 'ten']):
                        period = 10
                    elif any(x in feature_name.lower() for x in ['50', 'fifty']):
                        period = 50
                    return close_prices.rolling(period).mean()
                elif 'ema' in feature_name.lower():
                    period = 12
                    if any(x in feature_name.lower() for x in ['26', 'twenty']):
                        period = 26
                    return close_prices.ewm(span=period).mean()
            
            elif category == 'volume':
                if 'volume' in data.columns:
                    return data['volume'].rolling(20).mean()
            
            elif category == 'returns':
                return close_prices.pct_change()
            
            # Default: return price-based feature
            return close_prices.pct_change()
            
        except Exception as e:
            tprint_debug(f"Error generating feature {feature_name}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_stochastic(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic oscillator."""
        try:
            low_min = prices.rolling(period).min()
            high_max = prices.rolling(period).max()
            stoch = 100 * (prices - low_min) / (high_max - low_min)
            return stoch
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _generate_feature_interactions(self, feature_df: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate interactions between selected features."""
        tprint_debug(f"Generating interactions for {len(feature_df.columns)} features")
        
        interactions = []
        feature_names = list(feature_df.columns)
        
        # Generate different types of interactions
        interaction_types = [
            'product', 'ratio', 'difference', 'sum', 'log_product', 
            'log_ratio', 'polynomial', 'conditional', 'rolling_mean'
        ]
        
        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                for interaction_type in interaction_types:
                    try:
                        interaction = self._create_interaction(
                            feature_df, feat1, feat2, interaction_type, targets
                        )
                        if interaction:
                            interactions.append(interaction)
                    except Exception as e:
                        tprint_debug(f"Error creating {interaction_type} interaction {feat1}-{feat2}: {e}")
                        continue
        
        # Single-feature interactions
        for feat in feature_names:
            try:
                # Polynomial interaction
                poly_interaction = self._create_single_feature_interaction(
                    feature_df, feat, 'polynomial', targets
                )
                if poly_interaction:
                    interactions.append(poly_interaction)
                
                # Rolling mean interaction
                rolling_interaction = self._create_single_feature_interaction(
                    feature_df, feat, 'rolling_mean', targets
                )
                if rolling_interaction:
                    interactions.append(rolling_interaction)
                    
            except Exception as e:
                tprint_debug(f"Error creating single-feature interaction for {feat}: {e}")
                continue
        
        tprint_success(f"Generated {len(interactions)} interactions")
        return interactions
    
    def _create_interaction(self, feature_df: pd.DataFrame, feat1: str, feat2: str, 
                          interaction_type: str, targets: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        """Create a specific interaction between two features."""
        try:
            series1 = feature_df[feat1].dropna()
            series2 = feature_df[feat2].dropna()
            
            # Align series
            common_idx = series1.index.intersection(series2.index)
            if len(common_idx) < 10:
                return None
            
            series1 = series1.loc[common_idx]
            series2 = series2.loc[common_idx]
            
            # Generate interaction based on type
            if interaction_type == 'product':
                interaction_series = series1 * series2
            elif interaction_type == 'ratio':
                interaction_series = series1 / (series2 + 1e-8)
            elif interaction_type == 'difference':
                interaction_series = series1 - series2
            elif interaction_type == 'sum':
                interaction_series = series1 + series2
            elif interaction_type == 'log_product':
                # Ensure positive values for log
                s1_safe = np.where(series1 <= 0, np.abs(series1) + 1e-8, series1)
                s2_safe = np.where(series2 <= 0, np.abs(series2) + 1e-8, series2)
                interaction_series = np.log(s1_safe) * np.log(s2_safe)
            elif interaction_type == 'log_ratio':
                s1_safe = np.where(series1 <= 0, np.abs(series1) + 1e-8, series1)
                s2_safe = np.where(series2 <= 0, np.abs(series2) + 1e-8, series2)
                interaction_series = np.log(s1_safe) - np.log(s2_safe)
            elif interaction_type == 'conditional':
                # Conditional interaction: feat1 * (feat2 > feat2.median())
                condition = (series2 > series2.median()).astype(float)
                interaction_series = series1 * condition
            else:
                return None
            
            # Convert to pandas Series
            interaction_series = pd.Series(interaction_series, index=common_idx)
            
            # Calculate utility score
            utility_score = self._calculate_utility_score(interaction_series, targets)
            
            if utility_score < 0.1:  # Filter out low-utility interactions
                return None
            
            return {
                'name': f"{interaction_type}_{feat1}_{feat2}",
                'interaction_type': interaction_type,
                'parent_features': [feat1, feat2],
                'feature_series': interaction_series,
                'utility_score': utility_score,
                'metadata': {
                    'generation_method': 'data_driven',
                    'feature_count': 2
                }
            }
            
        except Exception as e:
            tprint_debug(f"Error creating interaction {interaction_type}({feat1}, {feat2}): {e}")
            return None
    
    def _create_single_feature_interaction(self, feature_df: pd.DataFrame, feat: str, 
                                         interaction_type: str, targets: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        """Create a single-feature interaction."""
        try:
            series = feature_df[feat].dropna()
            
            if len(series) < 10:
                return None
            
            if interaction_type == 'polynomial':
                interaction_series = series ** 2
            elif interaction_type == 'rolling_mean':
                interaction_series = series.rolling(20).mean()
            else:
                return None
            
            # Calculate utility score
            utility_score = self._calculate_utility_score(interaction_series, targets)
            
            if utility_score < 0.1:
                return None
            
            return {
                'name': f"{interaction_type}_{feat}",
                'interaction_type': interaction_type,
                'parent_features': [feat],
                'feature_series': interaction_series,
                'utility_score': utility_score,
                'metadata': {
                    'generation_method': 'data_driven',
                    'feature_count': 1
                }
            }
            
        except Exception as e:
            tprint_debug(f"Error creating single-feature interaction {interaction_type}({feat}): {e}")
            return None
    
    def _calculate_utility_score(self, interaction_series: pd.Series, targets: Optional[pd.Series]) -> float:
        """Calculate utility score for an interaction."""
        try:
            if targets is None:
                # Use variance as utility score
                return float(interaction_series.var())
            
            # Align with targets
            common_idx = interaction_series.index.intersection(targets.index)
            if len(common_idx) < 10:
                return 0.0
            
            aligned_series = interaction_series.loc[common_idx]
            aligned_targets = targets.loc[common_idx]
            
            # Calculate correlation
            correlation = aligned_series.corr(aligned_targets)
            if np.isnan(correlation):
                return 0.0
            
            # Use absolute correlation as utility score
            return abs(correlation)
            
        except Exception as e:
            tprint_debug(f"Error calculating utility score: {e}")
            return 0.0
    
    def _filter_and_rank_interactions(self, interactions: List[Dict[str, Any]], targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Filter and rank interactions by quality."""
        if not interactions:
            return []
        
        # Sort by utility score
        interactions.sort(key=lambda x: x['utility_score'], reverse=True)
        
        # Remove highly correlated interactions
        filtered_interactions = self._remove_correlated_interactions(interactions)
        
        # Limit to top interactions
        max_interactions = 100
        return filtered_interactions[:max_interactions]
    
    def _remove_correlated_interactions(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove highly correlated interactions."""
        if len(interactions) <= 1:
            return interactions
        
        # Create DataFrame of interaction features
        interaction_data = {}
        for interaction in interactions:
            interaction_data[interaction['name']] = interaction['feature_series']
        
        interaction_df = pd.DataFrame(interaction_data)
        
        # Calculate correlation matrix
        corr_matrix = interaction_df.corr()
        
        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:  # High correlation threshold
                    # Keep the one with higher utility score
                    if interactions[i]['utility_score'] >= interactions[j]['utility_score']:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
        
        # Filter out highly correlated interactions
        filtered = [interaction for i, interaction in enumerate(interactions) if i not in to_remove]
        
        return filtered
    
    def _calculate_feature_metrics(self, feature_df: pd.DataFrame, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for feature generation."""
        return {
            'total_features_generated': len(feature_df.columns),
            'total_interactions_generated': len(interactions),
            'average_utility_score': np.mean([i['utility_score'] for i in interactions]) if interactions else 0.0,
            'interaction_types': list(set(i['interaction_type'] for i in interactions)),
            'feature_categories_used': len(set(f.split('_')[0] for f in feature_df.columns if '_' in f))
        }
    
    def _generate_htf_interactions(self, data: pd.DataFrame, targets: Optional[pd.Series], 
                                  patterns: Any) -> Dict[str, Any]:
        """Generate HTF-aware interactions using template-based approach."""
        tprint_debug("Starting HTF-aware interaction generation")
        
        # Step 1: Create HTF features (simulated higher timeframe features)
        tprint_debug("Step 1: Creating HTF features")
        htf_features = self._create_htf_features(data)
        
        # Step 2: Generate core interactions using templates
        tprint_debug("Step 2: Generating core interactions")
        core_interactions = self._generate_core_template_interactions(data, targets)
        
        # Step 3: Generate HTF-aware interactions
        tprint_debug("Step 3: Generating HTF-aware interactions")
        htf_interactions = self._generate_htf_template_interactions(data, htf_features, targets)
        
        # Step 4: Combine and filter interactions
        tprint_debug("Step 4: Combining and filtering interactions")
        all_interactions = core_interactions + htf_interactions
        filtered_interactions = self._filter_htf_interactions(all_interactions, targets)
        
        result = {
            'generated_interactions': filtered_interactions,
            'htf_features': htf_features,
            'core_interactions': core_interactions,
            'htf_interactions': htf_interactions,
            'interaction_types': list(set(i.get('interaction_type', 'unknown') for i in filtered_interactions)),
            'utility_scores': {i['name']: i['utility_score'] for i in filtered_interactions}
        }
        
        tprint_success(f"HTF interaction generation completed: {len(filtered_interactions)} interactions generated")
        return result
    
    def _create_htf_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create simulated HTF (Higher Timeframe) features."""
        htf_features = {}
        
        if 'close' not in data.columns:
            return htf_features
        
        close_prices = data['close']
        
        # Create different HTF features
        htf_periods = [20, 50, 100]  # Simulate different HTF periods
        
        for period in htf_periods:
            # HTF trend features
            htf_features[f'htf_trend_{period}'] = close_prices.rolling(period).mean()
            htf_features[f'htf_ema_{period}'] = close_prices.ewm(span=period).mean()
            
            # HTF volatility features
            htf_features[f'htf_vol_{period}'] = close_prices.rolling(period).std()
            htf_features[f'htf_log_vol_{period}'] = np.log(close_prices.rolling(period).std() + 1e-8)
            
            # HTF momentum features
            htf_features[f'htf_momentum_{period}'] = close_prices.pct_change(period)
            htf_features[f'htf_log_momentum_{period}'] = np.log(close_prices / close_prices.shift(period) + 1e-8)
            
            # HTF regime features (simplified)
            sma = close_prices.rolling(period).mean()
            htf_features[f'htf_regime_{period}'] = (close_prices > sma).astype(float)
        
        # HTF anchor features
        if 'volume' in data.columns:
            htf_features['htf_vwap_50'] = (data['close'] * data['volume']).rolling(50).sum() / data['volume'].rolling(50).sum()
        
        tprint_debug(f"Created {len(htf_features)} HTF features")
        return htf_features
    
    def _generate_core_template_interactions(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate core interactions using predefined templates."""
        interactions = []
        
        # Core interaction templates
        core_templates = [
            {
                'name': 'price_vol_interaction',
                'formula': 'price_feature * volatility_feature',
                'required_features': ['price_feature', 'volatility_feature'],
                'max_instances': 5,
                'priority': 1
            },
            {
                'name': 'momentum_meanrev_interaction',
                'formula': 'momentum_feature * mean_reversion_feature',
                'required_features': ['momentum_feature', 'mean_reversion_feature'],
                'max_instances': 5,
                'priority': 1
            },
            {
                'name': 'vol_volume_interaction',
                'formula': 'volatility_feature * volume_feature',
                'required_features': ['volatility_feature', 'volume_feature'],
                'max_instances': 5,
                'priority': 1
            },
            {
                'name': 'ratio_interaction',
                'formula': 'feature1 / (feature2 + epsilon)',
                'required_features': ['feature1', 'feature2'],
                'max_instances': 5,
                'priority': 3
            },
            {
                'name': 'log_product_interaction',
                'formula': 'np.log(feature1 + epsilon) * np.log(feature2 + epsilon)',
                'required_features': ['feature1', 'feature2'],
                'max_instances': 4,
                'priority': 2
            }
        ]
        
        # Group features by type
        feature_groups = self._group_features_for_templates(data)
        
        # Generate interactions for each template
        for template in core_templates:
            template_interactions = self._generate_template_interactions(
                template, feature_groups, data, targets
            )
            interactions.extend(template_interactions)
        
        return interactions
    
    def _generate_htf_template_interactions(self, data: pd.DataFrame, htf_features: Dict[str, pd.Series], 
                                          targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate HTF-aware interactions using templates."""
        interactions = []
        
        # HTF-aware interaction templates
        htf_templates = [
            {
                'name': 'htf_trend_liquidity_interaction',
                'formula': 'htf_trend_feature * base_liquidity_feature',
                'required_features': ['htf_trend_feature', 'base_liquidity_feature'],
                'max_instances': 3,
                'priority': 1
            },
            {
                'name': 'htf_vol_signal_interaction',
                'formula': 'htf_volatility_feature * base_signal_feature',
                'required_features': ['htf_volatility_feature', 'base_signal_feature'],
                'max_instances': 3,
                'priority': 1
            },
            {
                'name': 'htf_momentum_conflict_interaction',
                'formula': 'htf_momentum_feature * (-base_momentum_feature)',
                'required_features': ['htf_momentum_feature', 'base_momentum_feature'],
                'max_instances': 3,
                'priority': 1
            },
            {
                'name': 'htf_log_trend_interaction',
                'formula': 'np.log(htf_trend_feature + epsilon) * base_feature',
                'required_features': ['htf_trend_feature', 'base_feature'],
                'max_instances': 3,
                'priority': 1
            }
        ]
        
        # Group HTF and base features
        htf_groups = self._group_htf_features(htf_features)
        base_groups = self._group_features_for_templates(data)
        
        # Generate HTF interactions
        for template in htf_templates:
            template_interactions = self._generate_htf_template_interactions_for_template(
                template, htf_groups, base_groups, data, targets
            )
            interactions.extend(template_interactions)
        
        return interactions
    
    def _group_features_for_templates(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by type for template matching."""
        groups = {
            'price_feature': [],
            'volatility_feature': [],
            'momentum_feature': [],
            'mean_reversion_feature': [],
            'liquidity_feature': [],
            'volume_feature': [],
            'base_feature': []
        }
        
        for col in data.columns:
            col_lower = col.lower()
            
            if any(x in col_lower for x in ['close', 'open', 'high', 'low', 'price']):
                groups['price_feature'].append(col)
            elif any(x in col_lower for x in ['vol', 'sigma', 'rv', 'volatility']):
                groups['volatility_feature'].append(col)
            elif any(x in col_lower for x in ['mom', 'momentum', 'signal', 'alpha']):
                groups['momentum_feature'].append(col)
            elif any(x in col_lower for x in ['rsi', 'stoch', 'mean_rev', 'osc']):
                groups['mean_reversion_feature'].append(col)
            elif any(x in col_lower for x in ['liquidity', 'depth', 'book']):
                groups['liquidity_feature'].append(col)
            elif 'volume' in col_lower:
                groups['volume_feature'].append(col)
            
            # Add to base features
            groups['base_feature'].append(col)
        
        return groups
    
    def _group_htf_features(self, htf_features: Dict[str, pd.Series]) -> Dict[str, List[str]]:
        """Group HTF features by type."""
        groups = {
            'htf_trend_feature': [],
            'htf_volatility_feature': [],
            'htf_momentum_feature': [],
            'htf_anchor_feature': [],
            'htf_regime_feature': []
        }
        
        for name, series in htf_features.items():
            name_lower = name.lower()
            
            if 'trend' in name_lower or 'ema' in name_lower:
                groups['htf_trend_feature'].append(name)
            elif 'vol' in name_lower:
                groups['htf_volatility_feature'].append(name)
            elif 'momentum' in name_lower:
                groups['htf_momentum_feature'].append(name)
            elif 'vwap' in name_lower or 'anchor' in name_lower:
                groups['htf_anchor_feature'].append(name)
            elif 'regime' in name_lower:
                groups['htf_regime_feature'].append(name)
        
        return groups
    
    def _generate_template_interactions(self, template: Dict[str, Any], feature_groups: Dict[str, List[str]], 
                                      data: pd.DataFrame, targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate interactions for a specific template."""
        interactions = []
        
        required_features = template['required_features']
        max_instances = template['max_instances']
        
        # Get feature combinations
        feature_combinations = self._get_feature_combinations(required_features, feature_groups)
        
        for combination in feature_combinations[:max_instances]:
            try:
                interaction = self._create_template_interaction(
                    template, combination, data, targets
                )
                if interaction:
                    interactions.append(interaction)
            except Exception as e:
                tprint_debug(f"Error creating template interaction {template['name']}: {e}")
                continue
        
        return interactions
    
    def _generate_htf_template_interactions_for_template(self, template: Dict[str, Any], 
                                                       htf_groups: Dict[str, List[str]], 
                                                       base_groups: Dict[str, List[str]], 
                                                       data: pd.DataFrame, 
                                                       targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Generate HTF interactions for a specific template."""
        interactions = []
        
        required_features = template['required_features']
        max_instances = template['max_instances']
        
        # Get HTF feature combinations
        feature_combinations = self._get_htf_feature_combinations(required_features, htf_groups, base_groups)
        
        for combination in feature_combinations[:max_instances]:
            try:
                interaction = self._create_htf_template_interaction(
                    template, combination, data, targets
                )
                if interaction:
                    interactions.append(interaction)
            except Exception as e:
                tprint_debug(f"Error creating HTF template interaction {template['name']}: {e}")
                continue
        
        return interactions
    
    def _get_feature_combinations(self, required_features: List[str], 
                                 feature_groups: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Get feature combinations for a template."""
        combinations = []
        
        # Get required feature lists
        required_lists = [feature_groups.get(req, []) for req in required_features]
        
        # Generate Cartesian product
        from itertools import product
        for combo in product(*required_lists):
            combination = dict(zip(required_features, combo))
            combinations.append(combination)
        
        return combinations
    
    def _get_htf_feature_combinations(self, required_features: List[str], 
                                    htf_groups: Dict[str, List[str]], 
                                    base_groups: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Get HTF feature combinations for a template."""
        combinations = []
        
        # Map required features to appropriate groups
        required_lists = []
        for req in required_features:
            if req.startswith('htf_'):
                required_lists.append(htf_groups.get(req, []))
            else:
                required_lists.append(base_groups.get(req, []))
        
        # Generate Cartesian product
        from itertools import product
        for combo in product(*required_lists):
            combination = dict(zip(required_features, combo))
            combinations.append(combination)
        
        return combinations
    
    def _create_template_interaction(self, template: Dict[str, Any], combination: Dict[str, str], 
                                   data: pd.DataFrame, targets: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        """Create a specific interaction from template and combination."""
        try:
            # Get feature data
            feature_data = {}
            for feature_type, feature_name in combination.items():
                if feature_name in data.columns:
                    feature_data[feature_type] = data[feature_name].dropna()
                else:
                    return None
            
            # Create interaction based on template formula
            interaction_series = self._evaluate_template_formula(template['formula'], feature_data)
            
            if interaction_series is None or len(interaction_series) < 10:
                return None
            
            # Calculate utility score
            utility_score = self._calculate_utility_score(interaction_series, targets)
            
            if utility_score < 0.1:
                return None
            
            return {
                'name': f"{template['name']}_{'_'.join(combination.values())}",
                'interaction_type': 'core_template',
                'parent_features': list(combination.values()),
                'feature_series': interaction_series,
                'utility_score': utility_score,
                'metadata': {
                    'template_name': template['name'],
                    'template_priority': template['priority'],
                    'generation_method': 'template_based'
                }
            }
            
        except Exception as e:
            tprint_debug(f"Error creating template interaction: {e}")
            return None
    
    def _create_htf_template_interaction(self, template: Dict[str, Any], combination: Dict[str, str], 
                                       data: pd.DataFrame, targets: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        """Create a specific HTF interaction from template and combination."""
        try:
            # Get feature data (mix of HTF and base features)
            feature_data = {}
            for feature_type, feature_name in combination.items():
                if feature_name in data.columns:
                    feature_data[feature_type] = data[feature_name].dropna()
                else:
                    # This might be an HTF feature, skip for now
                    return None
            
            # Create interaction based on template formula
            interaction_series = self._evaluate_template_formula(template['formula'], feature_data)
            
            if interaction_series is None or len(interaction_series) < 10:
                return None
            
            # Calculate utility score
            utility_score = self._calculate_utility_score(interaction_series, targets)
            
            if utility_score < 0.1:
                return None
            
            return {
                'name': f"{template['name']}_{'_'.join(combination.values())}",
                'interaction_type': 'htf_template',
                'parent_features': list(combination.values()),
                'feature_series': interaction_series,
                'utility_score': utility_score,
                'metadata': {
                    'template_name': template['name'],
                    'template_priority': template['priority'],
                    'generation_method': 'htf_template_based'
                }
            }
            
        except Exception as e:
            tprint_debug(f"Error creating HTF template interaction: {e}")
            return None
    
    def _evaluate_template_formula(self, formula: str, feature_data: Dict[str, pd.Series]) -> Optional[pd.Series]:
        """Evaluate a template formula with feature data."""
        try:
            # Simple formula evaluation (can be enhanced)
            if 'price_feature' in feature_data and 'volatility_feature' in feature_data:
                return feature_data['price_feature'] * feature_data['volatility_feature']
            elif 'feature1' in feature_data and 'feature2' in feature_data:
                return feature_data['feature1'] * feature_data['feature2']
            elif 'momentum_feature' in feature_data and 'mean_reversion_feature' in feature_data:
                return feature_data['momentum_feature'] * feature_data['mean_reversion_feature']
            elif 'volatility_feature' in feature_data and 'volume_feature' in feature_data:
                return feature_data['volatility_feature'] * feature_data['volume_feature']
            else:
                # Default: return first available feature
                if feature_data:
                    return list(feature_data.values())[0]
                return None
                
        except Exception as e:
            tprint_debug(f"Error evaluating formula {formula}: {e}")
            return None
    
    def _filter_htf_interactions(self, interactions: List[Dict[str, Any]], targets: Optional[pd.Series]) -> List[Dict[str, Any]]:
        """Filter HTF interactions by quality and remove duplicates."""
        if not interactions:
            return []
        
        # Sort by utility score
        interactions.sort(key=lambda x: x['utility_score'], reverse=True)
        
        # Remove highly correlated interactions
        filtered_interactions = self._remove_correlated_interactions(interactions)
        
        # Limit to top interactions
        max_interactions = 50
        return filtered_interactions[:max_interactions]
    
    def _select_features(self, data: pd.DataFrame, 
                        targets: pd.Series, 
                        cv_splits: List[Any]) -> Any:
        """Select features using multi-objective optimization."""
        tprint_debug("Selecting features")
        
        # Set CV splits for stability objective
        for obj in self.feature_selector.objectives:
            if hasattr(obj, 'cv_splits'):
                obj.cv_splits = cv_splits
        
        # Perform feature selection
        result = self.feature_selector.select_features(data, targets, cv_splits)
        
        tprint_success(f"Feature selection completed: {len(result.selected_features)} features selected")
        return result
    
    def _calculate_final_metrics(self, selection_result: Any, 
                                data: pd.DataFrame, 
                                targets: pd.Series) -> Dict[str, float]:
        """Calculate final performance metrics."""
        tprint_debug("Calculating final metrics")
        
        # Extract objective values
        objective_values = selection_result.objective_values
        
        # Calculate feature importance (simplified)
        feature_importance = {}
        for i, feature in enumerate(selection_result.selected_features):
            feature_importance[feature] = 1.0 / (i + 1)  # Simple ranking
        
        # Calculate additional metrics
        metrics = {
            'feature_importance': feature_importance,
            'out_of_sample_sharpe': objective_values.get('out_of_sample_sharpe', 0.0),
            'max_drawdown': objective_values.get('drawdown', 0.0),
            'stability_score': objective_values.get('stability', 0.0),
            'diversity_score': objective_values.get('diversity', 0.0)
        }
        
        tprint_success("Final metrics calculated")
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._initialize_performance_tracking()
        tprint_success("Performance statistics reset")
    
    def save_result(self, result: FeaturePipelineResult, 
                   output_path: Union[str, Path]) -> None:
        """Save pipeline result to file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save selected features
        features_df = pd.DataFrame({
            'feature': result.selected_features,
            'importance': [result.feature_importance.get(f, 0.0) for f in result.selected_features]
        })
        features_df.to_csv(output_path / 'selected_features.csv', index=False)
        
        # Save objective values
        objectives_df = pd.DataFrame(list(result.objective_values.items()), 
                                   columns=['objective', 'value'])
        objectives_df.to_csv(output_path / 'objective_values.csv', index=False)
        
        # Save metadata
        metadata = {
            'processing_time': result.processing_time,
            'n_cv_splits': result.n_cv_splits,
            'n_candidates_evaluated': result.n_candidates_evaluated,
            'out_of_sample_sharpe': result.out_of_sample_sharpe,
            'max_drawdown': result.max_drawdown,
            'stability_score': result.stability_score,
            'diversity_score': result.diversity_score
        }
        
        import json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        tprint_success(f"Result saved to {output_path}")


# Convenience functions
def create_unified_pipeline(config: Optional[UnifiedPipelineConfig] = None) -> UnifiedDataDrivenPipeline:
    """Create a unified data-driven pipeline."""
    return UnifiedDataDrivenPipeline(config)


def process_features(data: pd.DataFrame, 
                    targets: Optional[pd.Series] = None,
                    feature_columns: Optional[List[str]] = None,
                    config: Optional[UnifiedPipelineConfig] = None) -> FeaturePipelineResult:
    """Process features using the unified pipeline."""
    pipeline = create_unified_pipeline(config)
    return pipeline.process(data, targets, feature_columns)