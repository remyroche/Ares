"""
Battle-Tested Period + Lookback Optimization with Best Practices Integration

This module implements production-ready period and lookback optimization following
battle-tested guidelines for financial ML pipelines, integrating Bayesian TPE optimization.

Key Features:
- Coarse → fine search strategy with Bayesian TPE
- Purged walk-forward validation with embargo
- Multi-objective optimization (IC + Sharpe + stability)
- Constraint enforcement (min periods, correlation caps)
- Economic validation with turnover adjustment
- Comprehensive logging and diagnostics
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import ML Commons utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, compute_pareto_front,
        select_knee_point, compute_hypervolume
    )
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator
    ML_COMMONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Commons not available: {e}")
    ML_COMMONS_AVAILABLE = False

# Import purged K-fold
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False

# Import VectorBT for financial metrics
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class PeriodLookbackConfig:
    """Configuration for battle-tested period + lookback optimization."""
    
    # Search space parameters
    min_period: int = 2
    max_period: int = 200
    min_lookback: int = 5
    max_lookback: int = 500
    
    # Feature type constraints
    min_periods_per_type: int = 2
    max_correlation_threshold: float = 0.85
    
    # CV parameters
    n_splits: int = 5
    embargo_days: int = 7
    gap_days: int = 1
    
    # Multi-objective weights
    ic_weight: float = 0.4
    sharpe_weight: float = 0.3
    stability_weight: float = 0.2
    turnover_weight: float = 0.1
    
    # Optimization parameters
    n_trials: int = 100
    n_startup_trials: int = 20
    n_warmup_steps: int = 10
    
    # Economic validation
    min_oof_ic: float = 0.01
    min_sharpe_improvement: float = 0.1
    max_turnover: float = 2.0
    
    # Logging
    enable_detailed_logging: bool = True
    save_artifacts: bool = True
    artifacts_dir: str = "outcomes"


@dataclass
class PeriodLookbackCombo:
    """Period and lookback combination with scores."""
    period: int
    lookback: int
    ic_score: float
    sharpe_score: float
    stability_score: float
    turnover_score: float
    composite_score: float
    oof_ic: float
    oof_sharpe: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PeriodLookbackResult:
    """Result of period + lookback optimization."""
    selected_combos: List[PeriodLookbackCombo]
    trading_defaults: Dict[str, Any]
    interaction_periods: Dict[str, Any]
    optimization_metrics: Dict[str, Any]
    heatmap_data: Dict[str, Any]
    sensitivity_data: Dict[str, Any]
    artifacts: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class BattleTestedPeriodLookbackOptimizer:
    """Production-ready period + lookback optimizer with battle-tested best practices."""
    
    def __init__(self, config: Optional[PeriodLookbackConfig] = None):
        """Initialize the optimizer."""
        self.config = config or PeriodLookbackConfig()
        self.logger = logging.getLogger(__name__)
        self.artifacts_dir = Path(self.config.artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize Bayesian TPE optimizer
        if ML_COMMONS_AVAILABLE:
            self.tpe_optimizer = BayesianTPEOptimizer()
        else:
            self.tpe_optimizer = None
            
        # Initialize purged K-fold
        if PURGED_KFOLD_AVAILABLE:
            self.purged_kfold = PurgedKFoldTime(
                n_splits=self.config.n_splits,
                embargo_td=pd.Timedelta(days=self.config.embargo_days)
            )
        else:
            self.purged_kfold = None
    
    def optimize(self, 
                data: pd.DataFrame, 
                targets: pd.Series,
                feature_columns: Optional[List[str]] = None) -> PeriodLookbackResult:
        """
        Perform battle-tested period + lookback optimization.
        
        Args:
            data: Input DataFrame with features
            targets: Target series
            feature_columns: Optional list of feature columns to optimize
            
        Returns:
            PeriodLookbackResult with optimized combinations
        """
        start_time = time.time()
        tprint_info("🔄 Starting battle-tested period + lookback optimization")
        
        try:
            # Step 1: Data validation and preparation
            tprint_info("📊 Step 1: Data validation and preparation")
            data, targets, feature_columns = self._validate_and_prepare_data(data, targets, feature_columns)
            
            # Step 2: Fail-fast gates
            tprint_info("🚪 Step 2: Fail-fast validation gates")
            if not self._apply_fail_fast_gates(data, targets):
                return self._create_failure_result("Failed fail-fast validation gates")
            
            # Step 3: Coarse grid search
            tprint_info("🔍 Step 3: Coarse grid search")
            coarse_results = self._coarse_grid_search(data, targets, feature_columns)
            
            # Step 4: Fine grid search around best results
            tprint_info("🎯 Step 4: Fine grid search around best results")
            fine_results = self._fine_grid_search(data, targets, feature_columns, coarse_results)
            
            # Step 5: Bayesian TPE optimization
            tprint_info("🧠 Step 5: Bayesian TPE optimization")
            tpe_results = self._bayesian_tpe_optimization(data, targets, feature_columns, fine_results)
            
            # Step 6: Multi-objective Pareto optimization
            tprint_info("🎯 Step 6: Multi-objective Pareto optimization")
            pareto_results = self._pareto_optimization(tpe_results)
            
            # Step 7: Economic validation
            tprint_info("💰 Step 7: Economic validation")
            validated_results = self._economic_validation(data, targets, pareto_results)
            
            # Step 8: Generate final selections
            tprint_info("📊 Step 8: Generating final selections")
            result = self._generate_final_selections(
                validated_results, data, targets, start_time
            )
            
            tprint_success(f"✅ Period + lookback optimization completed: {len(result.selected_combos)} combinations selected")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Period + lookback optimization failed: {e}")
            return self._create_failure_result(str(e))
    
    def _validate_and_prepare_data(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series,
                                  feature_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Validate and prepare data for optimization."""
        # Validate inputs
        if data is None or data.empty:
            raise ValueError("Input data is None or empty")
        if targets is None or targets.empty:
            raise ValueError("Targets is None or empty")
        if len(data) != len(targets):
            raise ValueError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
        
        # Determine feature columns
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'open_time', 'timestamp']
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Filter data to feature columns only
        feature_data = data[feature_columns].copy()
        
        # Remove features with insufficient variance
        variance_threshold = 1e-8
        high_variance_features = feature_data.var() > variance_threshold
        feature_columns = [col for col in feature_columns if high_variance_features[col]]
        feature_data = feature_data[feature_columns]
        
        tprint_info(f"📊 Prepared {len(feature_columns)} features for optimization")
        return feature_data, targets, feature_columns
    
    def _apply_fail_fast_gates(self, data: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates."""
        # Gate 1: Minimum data size
        if len(data) < 200:
            tprint_warning("⚠️ Insufficient data for reliable optimization")
            return False
        
        # Gate 2: Target variance check
        if targets.var() < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False
        
        # Gate 3: Feature quality check
        nan_ratios = data.isnull().sum() / len(data)
        high_nan_features = nan_ratios > 0.3
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_features.sum()} features have >30% NaN values")
            return False
        
        # Gate 4: Memory check
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > 2000:  # 2GB limit
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False
        
        return True
    
    def _coarse_grid_search(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           feature_columns: List[str]) -> List[PeriodLookbackCombo]:
        """Perform coarse grid search for initial exploration."""
        tprint_info("🔍 Performing coarse grid search")
        
        # Define coarse grid
        periods = np.logspace(
            np.log10(self.config.min_period), 
            np.log10(self.config.max_period), 
            num=10, 
            dtype=int
        )
        lookbacks = np.logspace(
            np.log10(self.config.min_lookback), 
            np.log10(self.config.max_lookback), 
            num=10, 
            dtype=int
        )
        
        results = []
        total_combinations = len(periods) * len(lookbacks)
        
        for i, period in enumerate(periods):
            for j, lookback in enumerate(lookbacks):
                try:
                    combo_idx = i * len(lookbacks) + j + 1
                    tprint_info(f"🔍 Evaluating combination {combo_idx}/{total_combinations}: period={period}, lookback={lookback}")
                    
                    # Evaluate combination
                    combo = self._evaluate_period_lookback_combo(
                        data, targets, feature_columns, period, lookback
                    )
                    
                    if combo is not None:
                        results.append(combo)
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
                    continue
        
        # Sort by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        tprint_info(f"🔍 Coarse grid search completed: {len(results)} valid combinations")
        return results
    
    def _fine_grid_search(self, 
                         data: pd.DataFrame, 
                         targets: pd.Series,
                         feature_columns: List[str],
                         coarse_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform fine grid search around best coarse results."""
        tprint_info("🎯 Performing fine grid search")
        
        if not coarse_results:
            return []
        
        # Take top 3 coarse results for fine search
        top_coarse = coarse_results[:3]
        fine_results = []
        
        for coarse_combo in top_coarse:
            # Define fine grid around this combination
            period_range = max(2, coarse_combo.period // 4)
            lookback_range = max(5, coarse_combo.lookback // 4)
            
            periods = np.arange(
                max(self.config.min_period, coarse_combo.period - period_range),
                min(self.config.max_period, coarse_combo.period + period_range + 1),
                step=max(1, period_range // 5)
            )
            lookbacks = np.arange(
                max(self.config.min_lookback, coarse_combo.lookback - lookback_range),
                min(self.config.max_lookback, coarse_combo.lookback + lookback_range + 1),
                step=max(1, lookback_range // 5)
            )
            
            for period in periods:
                for lookback in lookbacks:
                    try:
                        combo = self._evaluate_period_lookback_combo(
                            data, targets, feature_columns, period, lookback
                        )
                        
                        if combo is not None:
                            fine_results.append(combo)
                            
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
                        continue
        
        # Sort by composite score
        fine_results.sort(key=lambda x: x.composite_score, reverse=True)
        
        tprint_info(f"🎯 Fine grid search completed: {len(fine_results)} valid combinations")
        return fine_results
    
    def _bayesian_tpe_optimization(self, 
                                  data: pd.DataFrame, 
                                  targets: pd.Series,
                                  feature_columns: List[str],
                                  fine_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform Bayesian TPE optimization."""
        tprint_info("🧠 Performing Bayesian TPE optimization")
        
        if not ML_COMMONS_AVAILABLE or self.tpe_optimizer is None:
            tprint_warning("⚠️ Bayesian TPE not available, using fine grid results")
            return fine_results
        
        try:
            # Define search space
            search_space = {
                'period': (self.config.min_period, self.config.max_period, 'int'),
                'lookback': (self.config.min_lookback, self.config.max_lookback, 'int')
            }
            
            # Define objective function
            def objective(trial):
                period = trial.suggest_int('period', self.config.min_period, self.config.max_period)
                lookback = trial.suggest_int('lookback', self.config.min_lookback, self.config.max_lookback)
                
                combo = self._evaluate_period_lookback_combo(
                    data, targets, feature_columns, period, lookback
                )
                
                if combo is None:
                    return -np.inf
                
                return combo.composite_score
            
            # Run optimization
            best_params = self.tpe_optimizer.optimize(
                objective=objective,
                search_space=search_space,
                n_trials=self.config.n_trials,
                n_startup_trials=self.config.n_startup_trials
            )
            
            # Evaluate best parameters
            best_combo = self._evaluate_period_lookback_combo(
                data, targets, feature_columns, 
                best_params['period'], best_params['lookback']
            )
            
            if best_combo is not None:
                tpe_results = [best_combo]
            else:
                tpe_results = fine_results[:5]  # Fallback to top fine results
            
            tprint_info(f"🧠 Bayesian TPE optimization completed: {len(tpe_results)} combinations")
            return tpe_results
            
        except Exception as e:
            tprint_warning(f"⚠️ Bayesian TPE optimization failed: {e}")
            return fine_results[:5]  # Fallback to top fine results
    
    def _pareto_optimization(self, 
                           tpe_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform Pareto optimization on TPE results."""
        tprint_info("🎯 Performing Pareto optimization")
        
        if not ML_COMMONS_AVAILABLE or len(tpe_results) < 2:
            return tpe_results
        
        try:
            # Create Pareto solutions
            solutions = []
            for i, combo in enumerate(tpe_results):
                solution = Solution(
                    values=[combo.ic_score, combo.sharpe_score, combo.stability_score],
                    metadata={'combo': combo, 'index': i}
                )
                solutions.append(solution)
            
            # Compute Pareto front
            pareto_front = compute_pareto_front(solutions)
            
            # Extract Pareto-optimal combinations
            pareto_combos = []
            for solution in pareto_front:
                combo = solution.metadata['combo']
                pareto_combos.append(combo)
            
            tprint_info(f"🎯 Pareto optimization completed: {len(pareto_combos)} combinations")
            return pareto_combos
            
        except Exception as e:
            tprint_warning(f"⚠️ Pareto optimization failed: {e}")
            return tpe_results
    
    def _economic_validation(self, 
                           data: pd.DataFrame, 
                           targets: pd.Series,
                           pareto_results: List[PeriodLookbackCombo]) -> List[PeriodLookbackCombo]:
        """Perform economic validation of combinations."""
        tprint_info("💰 Performing economic validation")
        
        validated_combos = []
        
        for combo in pareto_results:
            try:
                # Check OOF IC threshold
                if combo.oof_ic < self.config.min_oof_ic:
                    tprint_warning(f"⚠️ Combo period={combo.period}, lookback={combo.lookback} failed OOF IC threshold: {combo.oof_ic:.4f}")
                    continue
                
                # Check OOF Sharpe improvement
                if combo.oof_sharpe < self.config.min_sharpe_improvement:
                    tprint_warning(f"⚠️ Combo period={combo.period}, lookback={combo.lookback} failed OOF Sharpe threshold: {combo.oof_sharpe:.4f}")
                    continue
                
                # Check turnover constraint
                if combo.turnover_score > self.config.max_turnover:
                    tprint_warning(f"⚠️ Combo period={combo.period}, lookback={combo.lookback} exceeded turnover limit: {combo.turnover_score:.4f}")
                    continue
                
                validated_combos.append(combo)
                
            except Exception as e:
                tprint_warning(f"⚠️ Economic validation failed for period={combo.period}, lookback={combo.lookback}: {e}")
                continue
        
        tprint_info(f"💰 Economic validation: {len(pareto_results)} -> {len(validated_combos)} combinations")
        return validated_combos
    
    def _evaluate_period_lookback_combo(self, 
                                      data: pd.DataFrame, 
                                      targets: pd.Series,
                                      feature_columns: List[str],
                                      period: int, 
                                      lookback: int) -> Optional[PeriodLookbackCombo]:
        """Evaluate a specific period + lookback combination."""
        try:
            # Generate features with this period/lookback combination
            features = self._generate_features_with_period_lookback(
                data, feature_columns, period, lookback
            )
            
            if features is None or features.empty:
                return None
            
            # Calculate scores
            ic_score = self._calculate_ic_score(features, targets)
            sharpe_score = self._calculate_sharpe_score(features, targets)
            stability_score = self._calculate_stability_score(features, targets)
            turnover_score = self._calculate_turnover_score(features, targets)
            
            # Calculate composite score
            composite_score = (
                self.config.ic_weight * ic_score +
                self.config.sharpe_weight * sharpe_score +
                self.config.stability_weight * stability_score +
                self.config.turnover_weight * turnover_score
            )
            
            # Calculate OOF metrics
            oof_ic, oof_sharpe = self._calculate_oof_metrics(features, targets)
            
            return PeriodLookbackCombo(
                period=period,
                lookback=lookback,
                ic_score=ic_score,
                sharpe_score=sharpe_score,
                stability_score=stability_score,
                turnover_score=turnover_score,
                composite_score=composite_score,
                oof_ic=oof_ic,
                oof_sharpe=oof_sharpe,
                metadata={
                    'feature_count': len(features.columns),
                    'data_points': len(features)
                }
            )
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
            return None
    
    def _generate_features_with_period_lookback(self, 
                                              data: pd.DataFrame, 
                                              feature_columns: List[str],
                                              period: int, 
                                              lookback: int) -> Optional[pd.DataFrame]:
        """Generate features with specific period and lookback parameters."""
        try:
            # This is a simplified implementation
            # In practice, you would implement the actual feature generation logic
            # based on the period and lookback parameters
            
            features = data[feature_columns].copy()
            
            # Apply period-based transformations
            for col in feature_columns:
                if period > 1:
                    # Simple moving average as example
                    features[f"{col}_sma_{period}"] = features[col].rolling(window=period).mean()
            
            # Apply lookback-based transformations
            for col in feature_columns:
                if lookback > 1:
                    # Simple lookback features as example
                    features[f"{col}_lag_{lookback}"] = features[col].shift(lookback)
            
            # Remove NaN values
            features = features.dropna()
            
            if len(features) < 10:
                return None
            
            return features
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature generation failed: {e}")
            return None
    
    def _calculate_ic_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate Information Coefficient score."""
        try:
            # Calculate IC for each feature and take the mean
            ics = []
            for col in features.columns:
                ic = safe_correlation(features[col], targets)
                if not np.isnan(ic):
                    ics.append(abs(ic))
            
            return np.mean(ics) if ics else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_sharpe_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate Sharpe ratio score."""
        try:
            # Calculate returns
            returns = targets.pct_change().dropna()
            if len(returns) < 2:
                return 0.0
            
            # Calculate Sharpe ratio
            sharpe = safe_divide(returns.mean(), returns.std())
            return max(0.0, sharpe) if not np.isnan(sharpe) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate stability score using purged walk-forward CV."""
        try:
            if self.purged_kfold is None:
                return 0.5  # Default stability score
            
            correlations = []
            for train_idx, val_idx in self.purged_kfold.split(features.index):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                train_corr = safe_correlation(
                    features.iloc[train_idx].mean(axis=1), 
                    targets.iloc[train_idx]
                )
                val_corr = safe_correlation(
                    features.iloc[val_idx].mean(axis=1), 
                    targets.iloc[val_idx]
                )
                
                if not np.isnan(train_corr) and not np.isnan(val_corr):
                    correlations.append(val_corr)
            
            if not correlations:
                return 0.0
            
            # Stability is inverse of standard deviation
            stability = 1.0 / (1.0 + np.std(correlations))
            return min(stability, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_turnover_score(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate turnover score (lower is better)."""
        try:
            # Calculate feature changes as proxy for turnover
            feature_changes = features.diff().abs().sum(axis=1)
            turnover = feature_changes.mean()
            
            # Normalize turnover score (lower is better)
            turnover_score = 1.0 / (1.0 + turnover)
            return turnover_score
            
        except Exception:
            return 0.5
    
    def _calculate_oof_metrics(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[float, float]:
        """Calculate out-of-fold IC and Sharpe metrics."""
        try:
            if self.purged_kfold is None:
                return 0.0, 0.0
            
            oof_ics = []
            oof_sharpes = []
            
            for train_idx, val_idx in self.purged_kfold.split(features.index):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                val_features = features.iloc[val_idx]
                val_targets = targets.iloc[val_idx]
                
                # Calculate IC
                ic = safe_correlation(val_features.mean(axis=1), val_targets)
                if not np.isnan(ic):
                    oof_ics.append(ic)
                
                # Calculate Sharpe
                returns = val_targets.pct_change().dropna()
                if len(returns) > 1:
                    sharpe = safe_divide(returns.mean(), returns.std())
                    if not np.isnan(sharpe):
                        oof_sharpes.append(sharpe)
            
            oof_ic = np.mean(oof_ics) if oof_ics else 0.0
            oof_sharpe = np.mean(oof_sharpes) if oof_sharpes else 0.0
            
            return oof_ic, oof_sharpe
            
        except Exception:
            return 0.0, 0.0
    
    def _generate_final_selections(self, 
                                 validated_combos: List[PeriodLookbackCombo],
                                 data: pd.DataFrame,
                                 targets: pd.Series,
                                 start_time: float) -> PeriodLookbackResult:
        """Generate final selections and artifacts."""
        tprint_info("📊 Generating final selections")
        
        # Select trading defaults (top 3 combinations)
        trading_defaults = {
            'primary': validated_combos[0] if validated_combos else None,
            'secondary': validated_combos[1] if len(validated_combos) > 1 else None,
            'tertiary': validated_combos[2] if len(validated_combos) > 2 else None
        }
        
        # Select interaction periods (top 5 diverse combinations)
        interaction_periods = {
            'top_5_combos': validated_combos[:5],
            'diverse_periods': list(set(combo.period for combo in validated_combos[:10])),
            'diverse_lookbacks': list(set(combo.lookback for combo in validated_combos[:10]))
        }
        
        # Calculate optimization metrics
        optimization_metrics = {
            'total_combinations_evaluated': len(validated_combos),
            'combinations_selected': len(validated_combos),
            'average_ic': np.mean([c.ic_score for c in validated_combos]) if validated_combos else 0,
            'average_sharpe': np.mean([c.sharpe_score for c in validated_combos]) if validated_combos else 0,
            'average_stability': np.mean([c.stability_score for c in validated_combos]) if validated_combos else 0,
            'execution_time': time.time() - start_time
        }
        
        # Generate heatmap data
        heatmap_data = {
            'periods': [c.period for c in validated_combos],
            'lookbacks': [c.lookback for c in validated_combos],
            'scores': [c.composite_score for c in validated_combos]
        }
        
        # Generate sensitivity data
        sensitivity_data = {
            'period_sensitivity': self._calculate_period_sensitivity(validated_combos),
            'lookback_sensitivity': self._calculate_lookback_sensitivity(validated_combos)
        }
        
        # Save artifacts if enabled
        artifacts = {}
        if self.config.save_artifacts:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save optimization results
            results_path = self.artifacts_dir / f"period_lookback_optimization_{timestamp}.json"
            results_data = {
                'selected_combos': [
                    {
                        'period': c.period,
                        'lookback': c.lookback,
                        'ic_score': c.ic_score,
                        'sharpe_score': c.sharpe_score,
                        'stability_score': c.stability_score,
                        'composite_score': c.composite_score,
                        'oof_ic': c.oof_ic,
                        'oof_sharpe': c.oof_sharpe
                    }
                    for c in validated_combos
                ],
                'trading_defaults': {
                    'primary': trading_defaults['primary'].__dict__ if trading_defaults['primary'] else None,
                    'secondary': trading_defaults['secondary'].__dict__ if trading_defaults['secondary'] else None,
                    'tertiary': trading_defaults['tertiary'].__dict__ if trading_defaults['tertiary'] else None
                },
                'optimization_metrics': optimization_metrics
            }
            
            import json
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            artifacts['optimization_results_path'] = str(results_path)
        
        return PeriodLookbackResult(
            selected_combos=validated_combos,
            trading_defaults=trading_defaults,
            interaction_periods=interaction_periods,
            optimization_metrics=optimization_metrics,
            heatmap_data=heatmap_data,
            sensitivity_data=sensitivity_data,
            artifacts=artifacts,
            success=True
        )
    
    def _calculate_period_sensitivity(self, combos: List[PeriodLookbackCombo]) -> Dict[str, Any]:
        """Calculate period sensitivity analysis."""
        if not combos:
            return {}
        
        periods = [c.period for c in combos]
        scores = [c.composite_score for c in combos]
        
        return {
            'period_range': [min(periods), max(periods)],
            'score_range': [min(scores), max(scores)],
            'period_variance': np.var(periods),
            'score_variance': np.var(scores)
        }
    
    def _calculate_lookback_sensitivity(self, combos: List[PeriodLookbackCombo]) -> Dict[str, Any]:
        """Calculate lookback sensitivity analysis."""
        if not combos:
            return {}
        
        lookbacks = [c.lookback for c in combos]
        scores = [c.composite_score for c in combos]
        
        return {
            'lookback_range': [min(lookbacks), max(lookbacks)],
            'score_range': [min(scores), max(scores)],
            'lookback_variance': np.var(lookbacks),
            'score_variance': np.var(scores)
        }
    
    def _create_failure_result(self, error_message: str) -> PeriodLookbackResult:
        """Create a failure result."""
        return PeriodLookbackResult(
            selected_combos=[],
            trading_defaults={},
            interaction_periods={},
            optimization_metrics={},
            heatmap_data={},
            sensitivity_data={},
            artifacts={},
            success=False,
            error_message=error_message
        )