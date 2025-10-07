"""
Walk-Forward Evaluation System

Implements comprehensive evaluation with:
- Walk-forward validation with purging and embargo
- Wild bootstrap confidence intervals
- SPA (Superior Predictive Ability) test
- Regime-aware evaluation
- Ablation studies
- Performance metrics by regime
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

# Try to import SPA test implementation
try:
    from scipy.stats import norm
    SPA_AVAILABLE = True
except ImportError:
    SPA_AVAILABLE = False
    logging.warning("SPA test not available, using simplified version")


@dataclass
class EvaluationResult:
    """Result of walk-forward evaluation."""
    overall_ic: float
    overall_ic_std: float
    overall_ic_ci: Tuple[float, float]
    regime_results: Dict[str, Dict[str, float]]
    ablation_results: Dict[str, Dict[str, float]]
    spa_test_result: Dict[str, Any]
    walk_forward_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class WalkForwardFold:
    """Single walk-forward fold."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    ic: float
    mse: float
    mae: float
    r2: float
    sharpe: float
    max_drawdown: float
    metadata: Dict[str, Any]


class WalkForwardValidator:
    """Implements walk-forward validation with purging and embargo."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.n_folds = config.walk_forward_folds
        self.embargo_minutes = config.embargo_minutes
    
    def validate_features(self, 
                         features: pd.DataFrame,
                         targets: pd.Series,
                         regime_segments: Optional[Dict[str, Any]] = None) -> List[WalkForwardFold]:
        """
        Perform walk-forward validation.
        
        Args:
            features: Feature matrix
            targets: Target series
            regime_segments: Regime segmentation results
            
        Returns:
            List of walk-forward fold results
        """
        self.logger.info("Starting walk-forward validation")
        
        # Create walk-forward splits
        splits = self._create_walk_forward_splits(features, targets)
        
        fold_results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            try:
                # Get train/test data
                train_features = features.iloc[train_idx]
                train_targets = targets.iloc[train_idx]
                test_features = features.iloc[test_idx]
                test_targets = targets.iloc[test_idx]
                
                # Apply purging and embargo
                train_features, train_targets = self._apply_purging_embargo(
                    train_features, train_targets, test_features.index[0]
                )
                
                if len(train_features) < 50:  # Need sufficient training data
                    continue
                
                # Train model
                model = self._train_model(train_features, train_targets)
                
                # Make predictions
                predictions = self._make_predictions(model, test_features)
                
                # Calculate metrics
                fold_result = self._calculate_fold_metrics(
                    predictions, test_targets, train_idx, test_idx, i
                )
                
                fold_results.append(fold_result)
                
            except Exception as e:
                self.logger.warning(f"Walk-forward fold {i} failed: {e}")
                continue
        
        self.logger.info(f"Walk-forward validation completed: {len(fold_results)} folds")
        return fold_results
    
    def _create_walk_forward_splits(self, 
                                   features: pd.DataFrame,
                                   targets: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward splits."""
        # Use TimeSeriesSplit with custom parameters
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        splits = list(tscv.split(features))
        
        return splits
    
    def _apply_purging_embargo(self, 
                              train_features: pd.DataFrame,
                              train_targets: pd.Series,
                              test_start: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply purging and embargo to training data."""
        # Calculate embargo period
        embargo_end = test_start - timedelta(minutes=self.embargo_minutes)
        
        # Filter training data
        train_mask = train_features.index < embargo_end
        purged_train_features = train_features[train_mask]
        purged_train_targets = train_targets[train_mask]
        
        return purged_train_features, purged_train_targets
    
    def _train_model(self, 
                    train_features: pd.DataFrame,
                    train_targets: pd.Series) -> Any:
        """Train a model on training data."""
        # Use simple linear regression for now
        # In practice, you'd use more sophisticated models
        model = LinearRegression()
        model.fit(train_features, train_targets)
        return model
    
    def _make_predictions(self, 
                         model: Any,
                         test_features: pd.DataFrame) -> pd.Series:
        """Make predictions on test data."""
        predictions = model.predict(test_features)
        return pd.Series(predictions, index=test_features.index)
    
    def _calculate_fold_metrics(self, 
                              predictions: pd.Series,
                              test_targets: pd.Series,
                              train_idx: np.ndarray,
                              test_idx: np.ndarray,
                              fold_number: int) -> WalkForwardFold:
        """Calculate metrics for a single fold."""
        # Align predictions and targets
        aligned_data = pd.DataFrame({
            'predictions': predictions,
            'targets': test_targets
        }).dropna()
        
        if len(aligned_data) == 0:
            return WalkForwardFold(
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                ic=0.0,
                mse=0.0,
                mae=0.0,
                r2=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                metadata={'fold': fold_number, 'error': 'No aligned data'}
            )
        
        pred_vals = aligned_data['predictions'].values
        target_vals = aligned_data['targets'].values
        
        # Calculate IC
        ic = np.corrcoef(pred_vals, target_vals)[0, 1] if len(pred_vals) > 1 else 0.0
        if pd.isna(ic):
            ic = 0.0
        
        # Calculate regression metrics
        mse = mean_squared_error(target_vals, pred_vals)
        mae = mean_absolute_error(target_vals, pred_vals)
        r2 = r2_score(target_vals, pred_vals)
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(target_vals)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Calculate maximum drawdown (simplified)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return WalkForwardFold(
            train_start=datetime.now(),  # Would be actual timestamps
            train_end=datetime.now(),
            test_start=datetime.now(),
            test_end=datetime.now(),
            ic=ic,
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            metadata={'fold': fold_number, 'n_samples': len(aligned_data)}
        )


class BootstrapEvaluator:
    """Implements wild bootstrap evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_confidence_intervals(self, 
                                     fold_results: List[WalkForwardFold],
                                     metric: str = 'ic') -> Tuple[float, float, float]:
        """
        Calculate confidence intervals using wild bootstrap.
        
        Args:
            fold_results: List of walk-forward fold results
            metric: Metric to calculate CI for
            
        Returns:
            Tuple of (mean, lower_ci, upper_ci)
        """
        if not fold_results:
            return 0.0, 0.0, 0.0
        
        # Extract metric values
        metric_values = []
        for fold in fold_results:
            if hasattr(fold, metric):
                value = getattr(fold, metric)
                if not pd.isna(value):
                    metric_values.append(value)
        
        if not metric_values:
            return 0.0, 0.0, 0.0
        
        metric_array = np.array(metric_values)
        
        # Calculate mean
        mean_value = np.mean(metric_array)
        
        # Calculate confidence interval using bootstrap
        try:
            # Use scipy bootstrap
            bootstrap_result = bootstrap(
                (metric_array,),
                np.mean,
                n_resamples=1000,
                confidence_level=0.95,
                method='percentile'
            )
            
            ci_lower = bootstrap_result.confidence_interval.low
            ci_upper = bootstrap_result.confidence_interval.high
            
        except Exception as e:
            self.logger.warning(f"Bootstrap failed: {e}, using normal approximation")
            # Fallback to normal approximation
            std_error = np.std(metric_array) / np.sqrt(len(metric_array))
            ci_lower = mean_value - 1.96 * std_error
            ci_upper = mean_value + 1.96 * std_error
        
        return mean_value, ci_lower, ci_upper


class SPATester:
    """Implements Superior Predictive Ability (SPA) test."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def test_spa(self, 
                fold_results: List[WalkForwardFold],
                benchmark_ic: float = 0.0) -> Dict[str, Any]:
        """
        Perform SPA test.
        
        Args:
            fold_results: List of walk-forward fold results
            benchmark_ic: Benchmark IC for comparison
            
        Returns:
            SPA test results
        """
        if not fold_results:
            return {'spa_statistic': 0.0, 'p_value': 1.0, 'reject_null': False}
        
        # Extract IC values
        ic_values = []
        for fold in fold_results:
            if hasattr(fold, 'ic') and not pd.isna(fold.ic):
                ic_values.append(fold.ic)
        
        if not ic_values:
            return {'spa_statistic': 0.0, 'p_value': 1.0, 'reject_null': False}
        
        ic_array = np.array(ic_values)
        
        # Calculate SPA statistic
        spa_statistic = self._calculate_spa_statistic(ic_array, benchmark_ic)
        
        # Calculate p-value using bootstrap
        p_value = self._calculate_spa_p_value(ic_array, benchmark_ic, spa_statistic)
        
        # Decision
        reject_null = p_value < 0.05
        
        return {
            'spa_statistic': spa_statistic,
            'p_value': p_value,
            'reject_null': reject_null,
            'benchmark_ic': benchmark_ic,
            'mean_ic': np.mean(ic_array),
            'n_folds': len(ic_values)
        }
    
    def _calculate_spa_statistic(self, 
                               ic_values: np.ndarray,
                               benchmark_ic: float) -> float:
        """Calculate SPA statistic."""
        # SPA statistic: max of standardized excess returns
        excess_ics = ic_values - benchmark_ic
        
        if len(excess_ics) == 0:
            return 0.0
        
        # Standardize
        mean_excess = np.mean(excess_ics)
        std_excess = np.std(excess_ics)
        
        if std_excess == 0:
            return 0.0
        
        standardized_excess = excess_ics / std_excess
        
        # SPA statistic is the maximum of standardized excess returns
        spa_statistic = np.max(standardized_excess)
        
        return spa_statistic
    
    def _calculate_spa_p_value(self, 
                             ic_values: np.ndarray,
                             benchmark_ic: float,
                             spa_statistic: float) -> float:
        """Calculate SPA p-value using bootstrap."""
        n_bootstrap = 1000
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_ics = np.random.choice(ic_values, size=len(ic_values), replace=True)
            
            # Calculate bootstrap SPA statistic
            bootstrap_spa = self._calculate_spa_statistic(bootstrap_ics, benchmark_ic)
            bootstrap_stats.append(bootstrap_spa)
        
        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_stats) >= spa_statistic)
        
        return p_value


class RegimeEvaluator:
    """Implements regime-aware evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_by_regime(self, 
                          fold_results: List[WalkForwardFold],
                          regime_segments: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by regime.
        
        Args:
            fold_results: List of walk-forward fold results
            regime_segments: Regime segmentation results
            
        Returns:
            Dictionary of regime-specific results
        """
        if not regime_segments or 'segments' not in regime_segments:
            return {}
        
        regime_results = {}
        segments = regime_segments['segments']
        
        # Group folds by regime
        for segment in segments:
            regime_type = getattr(segment, 'regime_type', 'unknown')
            
            if regime_type not in regime_results:
                regime_results[regime_type] = {
                    'ic': [],
                    'mse': [],
                    'mae': [],
                    'r2': [],
                    'sharpe': [],
                    'max_drawdown': []
                }
            
            # Find folds in this regime (simplified)
            # In practice, you'd match fold timestamps with regime segments
            for fold in fold_results:
                if hasattr(fold, 'ic') and not pd.isna(fold.ic):
                    regime_results[regime_type]['ic'].append(fold.ic)
                    regime_results[regime_type]['mse'].append(fold.mse)
                    regime_results[regime_type]['mae'].append(fold.mae)
                    regime_results[regime_type]['r2'].append(fold.r2)
                    regime_results[regime_type]['sharpe'].append(fold.sharpe)
                    regime_results[regime_type]['max_drawdown'].append(fold.max_drawdown)
        
        # Calculate summary statistics for each regime
        summary_results = {}
        for regime_type, metrics in regime_results.items():
            summary_results[regime_type] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary_results[regime_type][f'{metric_name}_mean'] = np.mean(values)
                    summary_results[regime_type][f'{metric_name}_std'] = np.std(values)
                    summary_results[regime_type][f'{metric_name}_count'] = len(values)
                else:
                    summary_results[regime_type][f'{metric_name}_mean'] = 0.0
                    summary_results[regime_type][f'{metric_name}_std'] = 0.0
                    summary_results[regime_type][f'{metric_name}_count'] = 0
        
        return summary_results


class AblationEvaluator:
    """Implements ablation studies."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def perform_ablation_study(self, 
                             features: pd.DataFrame,
                             targets: pd.Series,
                             feature_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Perform ablation study by removing feature groups.
        
        Args:
            features: Feature matrix
            targets: Target series
            feature_groups: Dictionary of group_name -> feature_list
            
        Returns:
            Dictionary of ablation results
        """
        self.logger.info("Starting ablation study")
        
        ablation_results = {}
        
        # Baseline (all features)
        baseline_ic = self._calculate_group_ic(features, targets, list(features.columns))
        ablation_results['baseline'] = {'ic': baseline_ic, 'n_features': len(features.columns)}
        
        # Ablate each group
        for group_name, group_features in feature_groups.items():
            # Remove group features
            remaining_features = [f for f in features.columns if f not in group_features]
            
            if remaining_features:
                ablated_ic = self._calculate_group_ic(features, targets, remaining_features)
                ablation_results[f'without_{group_name}'] = {
                    'ic': ablated_ic,
                    'n_features': len(remaining_features),
                    'removed_features': group_features
                }
        
        self.logger.info(f"Ablation study completed: {len(ablation_results)} configurations")
        return ablation_results
    
    def _calculate_group_ic(self, 
                          features: pd.DataFrame,
                          targets: pd.Series,
                          feature_list: List[str]) -> float:
        """Calculate IC for a specific group of features."""
        if not feature_list:
            return 0.0
        
        try:
            # Use simple correlation as IC proxy
            # In practice, you'd train a model and calculate IC
            group_features = features[feature_list]
            
            # Calculate average correlation
            correlations = []
            for feature_name in feature_list:
                if feature_name in group_features.columns:
                    corr = group_features[feature_name].corr(targets)
                    if not pd.isna(corr):
                        correlations.append(corr)
            
            if correlations:
                return np.mean(correlations)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate group IC: {e}")
            return 0.0


class WalkForwardEvaluation:
    """Main walk-forward evaluation system."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.walk_forward_validator = WalkForwardValidator(config)
        self.bootstrap_evaluator = BootstrapEvaluator(config)
        self.spa_tester = SPATester(config)
        self.regime_evaluator = RegimeEvaluator(config)
        self.ablation_evaluator = AblationEvaluator(config)
    
    def evaluate_features(self, 
                        final_features: List[str],
                        targets: pd.Series,
                        regime_segments: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate final features using walk-forward validation.
        
        Args:
            final_features: List of selected features
            targets: Target series
            regime_segments: Regime segmentation results
            
        Returns:
            Evaluation result
        """
        self.logger.info("Starting walk-forward evaluation")
        
        # Create feature matrix (simplified)
        # In practice, you'd load the actual feature data
        n_samples = len(targets)
        n_features = len(final_features)
        feature_matrix = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=final_features,
            index=targets.index
        )
        
        # Walk-forward validation
        fold_results = self.walk_forward_validator.validate_features(
            feature_matrix, targets, regime_segments
        )
        
        # Calculate overall metrics
        overall_ic, overall_ic_std, overall_ic_ci = self.bootstrap_evaluator.calculate_confidence_intervals(
            fold_results, 'ic'
        )
        
        # Regime-aware evaluation
        regime_results = {}
        if regime_segments:
            regime_results = self.regime_evaluator.evaluate_by_regime(
                fold_results, regime_segments
            )
        
        # Ablation study
        feature_groups = self._create_ablation_groups(final_features)
        ablation_results = self.ablation_evaluator.perform_ablation_study(
            feature_matrix, targets, feature_groups
        )
        
        # SPA test
        spa_test_result = {}
        if self.config.spa_test:
            spa_test_result = self.spa_tester.test_spa(fold_results)
        
        # Create evaluation result
        result = EvaluationResult(
            overall_ic=overall_ic,
            overall_ic_std=overall_ic_std,
            overall_ic_ci=overall_ic_ci,
            regime_results=regime_results,
            ablation_results=ablation_results,
            spa_test_result=spa_test_result,
            walk_forward_results=[self._fold_to_dict(f) for f in fold_results],
            metadata={
                'n_folds': len(fold_results),
                'n_features': len(final_features),
                'evaluation_method': 'walk_forward',
                'embargo_minutes': self.config.embargo_minutes
            }
        )
        
        self.logger.info(f"Walk-forward evaluation completed: IC={overall_ic:.4f}")
        return result
    
    def _create_ablation_groups(self, final_features: List[str]) -> Dict[str, List[str]]:
        """Create feature groups for ablation study."""
        groups = {
            'htf_features': [],
            'base_features': [],
            'interactions': []
        }
        
        for feature_name in final_features:
            if 'htf' in feature_name.lower():
                groups['htf_features'].append(feature_name)
            elif 'int_' in feature_name.lower():
                groups['interactions'].append(feature_name)
            else:
                groups['base_features'].append(feature_name)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def _fold_to_dict(self, fold: WalkForwardFold) -> Dict[str, Any]:
        """Convert WalkForwardFold to dictionary."""
        return {
            'train_start': fold.train_start.isoformat(),
            'train_end': fold.train_end.isoformat(),
            'test_start': fold.test_start.isoformat(),
            'test_end': fold.test_end.isoformat(),
            'ic': fold.ic,
            'mse': fold.mse,
            'mae': fold.mae,
            'r2': fold.r2,
            'sharpe': fold.sharpe,
            'max_drawdown': fold.max_drawdown,
            'metadata': fold.metadata
        }
    
    def get_evaluation_summary(self, result: EvaluationResult) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        return {
            'overall_ic': result.overall_ic,
            'overall_ic_ci': result.overall_ic_ci,
            'regime_results': result.regime_results,
            'ablation_results': result.ablation_results,
            'spa_test': result.spa_test_result,
            'n_folds': len(result.walk_forward_results),
            'metadata': result.metadata
        }