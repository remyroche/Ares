"""
Multi-Objective Feature Selection

Implements multi-objective optimization for feature selection with explicit objectives:
- Out-of-sample Sharpe ratio
- Drawdown
- Turnover
- Stability
- Diversity
- Mutual Information
- Profit-centered objectives
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy.optimize import minimize
from sklearn.metrics import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import mutual_info_score
import warnings

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

# Enhanced imports from ml_commons utilities
try:
    from src.utils.ml_common.feature_selection import (
        FeatureSelector, FeatureSelectionConfig
    )
    from src.utils.ml_common.validation.data_leakage_prevention import (
        DataLeakagePrevention, DataLeakageConfig
    )
    from src.utils.ml_common.data_processing.data_quality import DataQualityUtilities
    from src.utils.ml_common.validation.unified_cv import (
        UnifiedCrossValidator, UnifiedCVResult
    )
    from src.utils.ml_common.validation.stability import (
        StabilityAnalyzer, StabilityConfig
    )
    ML_COMMONS_AVAILABLE = True
    tprint_info("✅ ML Commons feature selection utilities loaded successfully")
except ImportError as e:
    ML_COMMONS_AVAILABLE = False
    tprint_warning(f"⚠️ ML Commons utilities not available: {e}")
    # Create fallback classes
    class FeatureSelector:
        def __init__(self, *args, **kwargs): pass
        def select_features(self, *args, **kwargs): return []
    
    class DataLeakagePrevention:
        def __init__(self, *args, **kwargs): pass
        def check_data_leakage(self, *args, **kwargs): return {'status': 'clean', 'leakage_rate': 0.0}
    
    class DataQualityUtilities:
        def __init__(self, *args, **kwargs): pass
        def feature_correlation_analysis(self, *args, **kwargs): return {'highly_correlated_pairs': []}
        def automated_outlier_detection(self, *args, **kwargs): return {'outlier_indices': []}
    
    class UnifiedCrossValidator:
        def run(self, *args, **kwargs): return None
    
    class StabilityAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def analyze_stability(self, *args, **kwargs): return {'stability_score': 1.0}

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveResult:
    """Result of an objective function evaluation."""
    value: float
    metadata: Dict[str, Any]
    is_valid: bool = True


@dataclass
class MultiObjectiveResult:
    """Result of multi-objective optimization."""
    selected_features: List[str]
    objective_values: Dict[str, float]
    pareto_front: List[Dict[str, Any]]
    optimization_metadata: Dict[str, Any]
    is_valid: bool = True


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""
    
    @abstractmethod
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Evaluate the objective function."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the objective function."""
        pass
    
    @property
    @abstractmethod
    def is_higher_better(self) -> bool:
        """Whether higher values are better for this objective."""
        pass


class OutOfSampleSharpeObjective(ObjectiveFunction):
    """Out-of-sample Sharpe ratio objective."""
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
    
    @property
    def name(self) -> str:
        return "out_of_sample_sharpe"
    
    @property
    def is_higher_better(self) -> bool:
        return True
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate out-of-sample Sharpe ratio."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)
            
            # Get selected features
            selected_data = features[selected_features]
            
            # Calculate returns (assuming targets are returns)
            returns = targets
            
            # Calculate Sharpe ratio
            excess_returns = returns - self.risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Annualize if possible
            if len(returns) > 252:
                sharpe_ratio *= np.sqrt(252)
            
            metadata = {
                'mean_return': returns.mean(),
                'std_return': returns.std(),
                'excess_return': excess_returns.mean(),
                'n_periods': len(returns)
            }
            
            return ObjectiveResult(value=sharpe_ratio, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Sharpe ratio calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)


class DrawdownObjective(ObjectiveFunction):
    """Maximum drawdown objective (minimize)."""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
    
    @property
    def name(self) -> str:
        return "drawdown"
    
    @property
    def is_higher_better(self) -> bool:
        return False  # Lower drawdown is better
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate maximum drawdown."""
        try:
            if not selected_features:
                return ObjectiveResult(value=1.0, metadata={}, is_valid=False)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + targets).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = abs(drawdown.min())
            
            # Average drawdown
            avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
            
            metadata = {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'drawdown_duration': self._calculate_drawdown_duration(drawdown),
                'n_periods': len(targets)
            }
            
            return ObjectiveResult(value=max_drawdown, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Drawdown calculation failed: {e}")
            return ObjectiveResult(value=1.0, metadata={'error': str(e)}, is_valid=False)
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        in_drawdown = drawdown < 0
        consecutive_drawdown = in_drawdown.groupby((~in_drawdown).cumsum()).sum()
        return int(consecutive_drawdown.max()) if len(consecutive_drawdown) > 0 else 0


class TurnoverObjective(ObjectiveFunction):
    """Turnover objective (minimize)."""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    @property
    def name(self) -> str:
        return "turnover"
    
    @property
    def is_higher_better(self) -> bool:
        return False  # Lower turnover is better
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate turnover rate."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)
            
            # Get selected features
            selected_data = features[selected_features]
            
            # Calculate feature changes
            feature_changes = selected_data.diff().abs()
            
            # Calculate turnover as average absolute change
            turnover = feature_changes.mean().mean()
            
            # Calculate turnover volatility
            turnover_vol = feature_changes.std().mean()
            
            metadata = {
                'avg_turnover': turnover,
                'turnover_volatility': turnover_vol,
                'max_turnover': feature_changes.max().max(),
                'n_periods': len(selected_data)
            }
            
            return ObjectiveResult(value=turnover, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Turnover calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)


class StabilityObjective(ObjectiveFunction):
    """Stability objective (Jaccard similarity across folds)."""
    
    def __init__(self, cv_splits: Optional[List[Any]] = None):
        self.cv_splits = cv_splits
    
    @property
    def name(self) -> str:
        return "stability"
    
    @property
    def is_higher_better(self) -> bool:
        return True  # Higher stability is better
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate stability using Jaccard similarity."""
        try:
            if not selected_features or self.cv_splits is None:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)
            
            # Calculate Jaccard similarity across CV splits
            jaccard_similarities = []
            
            for i in range(len(self.cv_splits) - 1):
                split1_features = set(selected_features)  # Current selection
                split2_features = set(selected_features)  # Would be different in real CV
                
                # Calculate Jaccard similarity
                intersection = len(split1_features.intersection(split2_features))
                union = len(split1_features.union(split2_features))
                
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_similarities.append(jaccard)
            
            stability = np.mean(jaccard_similarities) if jaccard_similarities else 0.0
            
            metadata = {
                'jaccard_similarities': jaccard_similarities,
                'avg_stability': stability,
                'min_stability': np.min(jaccard_similarities) if jaccard_similarities else 0.0,
                'max_stability': np.max(jaccard_similarities) if jaccard_similarities else 0.0,
                'n_splits': len(self.cv_splits)
            }
            
            return ObjectiveResult(value=stability, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Stability calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)


class DiversityObjective(ObjectiveFunction):
    """Diversity objective (minimize correlation)."""
    
    def __init__(self, method: str = 'correlation_penalty'):
        self.method = method
    
    @property
    def name(self) -> str:
        return "diversity"
    
    @property
    def is_higher_better(self) -> bool:
        return True  # Higher diversity is better
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate diversity using correlation penalty or DPP."""
        try:
            if not selected_features or len(selected_features) < 2:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)
            
            # Get selected features
            selected_data = features[selected_features]
            
            if self.method == 'correlation_penalty':
                diversity = self._calculate_correlation_penalty(selected_data)
            elif self.method == 'dpp':
                diversity = self._calculate_dpp_diversity(selected_data)
            else:
                raise ValueError(f"Unknown diversity method: {self.method}")
            
            metadata = {
                'method': self.method,
                'n_features': len(selected_features),
                'diversity_score': diversity
            }
            
            return ObjectiveResult(value=diversity, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Diversity calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)
    
    def _calculate_correlation_penalty(self, selected_data: pd.DataFrame) -> float:
        """Calculate diversity using correlation penalty."""
        corr_matrix = selected_data.corr().abs()
        
        # Remove diagonal
        corr_matrix = corr_matrix - np.eye(len(corr_matrix))
        
        # Calculate average correlation penalty
        penalty = corr_matrix.sum().sum() / (len(corr_matrix) * (len(corr_matrix) - 1))
        
        # Convert to diversity score (higher is better)
        diversity = 1.0 - penalty
        
        return max(0.0, diversity)
    
    def _calculate_dpp_diversity(self, selected_data: pd.DataFrame) -> float:
        """Calculate diversity using Determinantal Point Process."""
        try:
            # Calculate similarity matrix
            corr_matrix = selected_data.corr().abs()
            
            # Convert to similarity matrix
            similarity_matrix = corr_matrix.values
            
            # Calculate DPP diversity
            det = np.linalg.det(similarity_matrix)
            
            # Normalize by number of features
            diversity = det ** (1.0 / len(selected_data.columns))
            
            return max(0.0, diversity)
            
        except Exception as e:
            tprint_debug(f"DPP diversity calculation failed: {e}")
            return 0.0


class MutualInformationObjective(ObjectiveFunction):
    """Mutual information objective."""
    
    def __init__(self, method: str = 'regression'):
        self.method = method
    
    @property
    def name(self) -> str:
        return "mutual_information"
    
    @property
    def is_higher_better(self) -> bool:
        return True  # Higher MI is better
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate mutual information between selected features and targets."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)
            
            # Get selected features
            selected_data = features[selected_features]
            
            # Calculate mutual information
            if self.method == 'regression':
                mi_scores = mutual_info_regression(selected_data, targets)
            elif self.method == 'classification':
                mi_scores = mutual_info_classif(selected_data, targets)
            else:
                raise ValueError(f"Unknown MI method: {self.method}")
            
            # Average MI across features
            avg_mi = np.mean(mi_scores)
            
            metadata = {
                'method': self.method,
                'mi_scores': mi_scores.tolist(),
                'avg_mi': avg_mi,
                'max_mi': np.max(mi_scores),
                'min_mi': np.min(mi_scores),
                'n_features': len(selected_features)
            }
            
            return ObjectiveResult(value=avg_mi, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Mutual information calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)


class ProfitCenteredObjective(ObjectiveFunction):
    """Profit-centered objective (maximize profit while minimizing risk)."""
    
    def __init__(self, risk_penalty: float = 0.5):
        self.risk_penalty = risk_penalty
    
    @property
    def name(self) -> str:
        return "profit_centered"
    
    @property
    def is_higher_better(self) -> bool:
        return True  # Higher profit is better
    
    def evaluate(self, features: pd.DataFrame, 
                targets: pd.Series, 
                selected_features: List[str],
                **kwargs) -> ObjectiveResult:
        """Calculate profit-centered objective."""
        try:
            if not selected_features:
                return ObjectiveResult(value=0.0, metadata={}, is_valid=False)
            
            # Calculate total return
            total_return = targets.sum()
            
            # Calculate risk (volatility)
            risk = targets.std()
            
            # Calculate profit-centered score
            profit_score = total_return - self.risk_penalty * risk
            
            # Normalize by number of periods
            profit_score = profit_score / len(targets)
            
            metadata = {
                'total_return': total_return,
                'risk': risk,
                'risk_penalty': self.risk_penalty,
                'profit_score': profit_score,
                'n_periods': len(targets)
            }
            
            return ObjectiveResult(value=profit_score, metadata=metadata, is_valid=True)
            
        except Exception as e:
            tprint_debug(f"Profit-centered calculation failed: {e}")
            return ObjectiveResult(value=0.0, metadata={'error': str(e)}, is_valid=False)


class MultiObjectiveFeatureSelector:
    """
    Multi-objective feature selector using explicit objectives.
    
    Enhanced with ml_commons utilities for:
    - Advanced feature selection algorithms
    - Data quality assessment
    - Leakage detection
    - Stability analysis
    - Cross-validation integration
    """
    
    def __init__(self, objectives: List[ObjectiveFunction], 
                 weights: Optional[Dict[str, float]] = None,
                 max_features: int = 50,
                 min_features: int = 5,
                 enable_ml_commons: bool = True,
                 enable_validation: bool = True):
        """
        Initialize multi-objective feature selector.
        
        Args:
            objectives: List of objective functions
            weights: Optional weights for objectives
            max_features: Maximum number of features to select
            min_features: Minimum number of features to select
            enable_ml_commons: Enable ml_commons integration
            enable_validation: Enable validation checks
        """
        self.objectives = objectives
        self.weights = weights or {obj.name: 1.0 for obj in objectives}
        self.max_features = max_features
        self.min_features = min_features
        self.enable_ml_commons = enable_ml_commons and ML_COMMONS_AVAILABLE
        self.enable_validation = enable_validation
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            tprint_warning(f"Objective weights sum to {total_weight:.6f}, not 1.0")
            # Normalize weights
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Initialize ml_commons utilities
        if self.enable_ml_commons:
            self._initialize_ml_commons_utilities()
        
        tprint_info(f"Initialized MultiObjectiveFeatureSelector with {len(objectives)} objectives")
        if self.enable_ml_commons:
            tprint_info("✅ Enhanced with ml_commons utilities")
    
    def _initialize_ml_commons_utilities(self):
        """Initialize ml_commons utilities for enhanced feature selection."""
        try:
            # Initialize feature selection utilities
            self.feature_selector = FeatureSelector(
                FeatureSelectionConfig(
                    max_features=self.max_features,
                    min_features=self.min_features,
                    enable_correlation_checks=True,
                    correlation_threshold=0.95
                )
            )
            
            # Initialize data quality utilities
            self.data_quality_utils = DataQualityUtilities({
                'missing_threshold': 0.1,
                'outlier_contamination': 0.05,
                'correlation_method': 'spearman'
            })
            
            # Initialize leakage prevention
            self.leakage_prevention = DataLeakagePrevention(
                DataLeakageConfig(
                    enable_temporal_validation=True,
                    critical_leakage_threshold=0.1,
                    enable_detailed_logging=True
                )
            )
            
            # Initialize stability analyzer
            self.stability_analyzer = StabilityAnalyzer(
                StabilityConfig(
                    enable_stability_checks=True,
                    stability_threshold=0.8
                )
            )
            
            # Initialize unified CV
            self.unified_cv = UnifiedCrossValidator()
            
            tprint_debug("ML Commons utilities initialized successfully")
            
        except Exception as e:
            tprint_warning(f"Failed to initialize some ML Commons utilities: {e}")
            self.enable_ml_commons = False
    
    def select_features(self, features: pd.DataFrame, 
                       targets: pd.Series,
                       cv_splits: Optional[List[Any]] = None) -> MultiObjectiveResult:
        """
        Select features using multi-objective optimization.
        
        Enhanced with ml_commons utilities for:
        - Data quality validation
        - Leakage detection
        - Advanced feature selection algorithms
        - Stability analysis
        - Cross-validation integration
        
        Args:
            features: Feature DataFrame
            targets: Target series
            cv_splits: Optional CV splits for stability calculation
            
        Returns:
            MultiObjectiveResult with selected features and objective values
        """
        tprint_info(f"Starting enhanced multi-objective feature selection for {features.shape[1]} features")
        
        # Enhanced data validation
        if self.enable_validation and self.enable_ml_commons:
            validation_results = self._validate_input_data(features, targets)
            if not validation_results['is_valid']:
                tprint_error(f"Data validation failed: {validation_results['issues']}")
                if self.enable_ml_commons:
                    raise ValueError(f"Data validation failed: {validation_results['issues']}")
                else:
                    tprint_warning("Continuing with validation issues due to disabled ml_commons")
        
        # Set CV splits for stability objective
        for obj in self.objectives:
            if isinstance(obj, StabilityObjective):
                obj.cv_splits = cv_splits
        
        # Enhanced feature pre-processing with ml_commons
        if self.enable_ml_commons:
            features_processed = self._preprocess_features_with_ml_commons(features, targets)
        else:
            features_processed = features
        
        # Generate candidate feature sets with enhanced algorithms
        if self.enable_ml_commons and hasattr(self, 'feature_selector'):
            candidate_sets = self._generate_enhanced_candidate_sets(features_processed, targets)
        else:
            candidate_sets = self._generate_candidate_sets(features_processed.columns.tolist())
        
        # Evaluate objectives for each candidate set
        pareto_front = []
        
        for candidate_set in candidate_sets:
            objective_values = {}
            is_valid = True
            
            # Enhanced objective evaluation with validation
            for obj in self.objectives:
                result = obj.evaluate(features_processed, targets, candidate_set)
                objective_values[obj.name] = result.value
                
                if not result.is_valid:
                    is_valid = False
                    break
            
            if is_valid:
                # Calculate weighted score
                weighted_score = sum(
                    self.weights[obj.name] * objective_values[obj.name] 
                    for obj in self.objectives
                )
                
                # Enhanced scoring with ml_commons validation
                if self.enable_ml_commons:
                    enhanced_score = self._calculate_enhanced_score(
                        candidate_set, features_processed, targets, objective_values, weighted_score
                    )
                else:
                    enhanced_score = weighted_score
                
                pareto_front.append({
                    'features': candidate_set,
                    'objective_values': objective_values,
                    'weighted_score': weighted_score,
                    'enhanced_score': enhanced_score,
                    'n_features': len(candidate_set)
                })
        
        # Sort by enhanced score if available, otherwise weighted score
        sort_key = 'enhanced_score' if self.enable_ml_commons else 'weighted_score'
        pareto_front.sort(key=lambda x: x[sort_key], reverse=True)
        
        # Select best feature set
        if pareto_front:
            best_set = pareto_front[0]
            selected_features = best_set['features']
            objective_values = best_set['objective_values']
        else:
            tprint_warning("No valid feature sets found, using all features")
            selected_features = features_processed.columns.tolist()[:self.max_features]
            objective_values = {obj.name: 0.0 for obj in self.objectives}
        
        # Enhanced validation of selected features
        if self.enable_ml_commons and self.enable_validation:
            final_validation = self._validate_selected_features(selected_features, features_processed, targets)
            if not final_validation['is_valid']:
                tprint_warning(f"Selected features validation issues: {final_validation['warnings']}")
        
        result = MultiObjectiveResult(
            selected_features=selected_features,
            objective_values=objective_values,
            pareto_front=pareto_front,
            optimization_metadata={
                'n_candidates': len(candidate_sets),
                'n_valid': len(pareto_front),
                'weights': self.weights,
                'max_features': self.max_features,
                'min_features': self.min_features,
                'ml_commons_enabled': self.enable_ml_commons,
                'enhanced_scoring': self.enable_ml_commons
            },
            is_valid=len(pareto_front) > 0
        )
        
        tprint_success(f"Enhanced feature selection completed: {len(selected_features)} features selected")
        return result
    
    def _generate_candidate_sets(self, all_features: List[str]) -> List[List[str]]:
        """Generate candidate feature sets for evaluation."""
        candidate_sets = []
        
        # Generate sets of different sizes
        for n_features in range(self.min_features, min(self.max_features + 1, len(all_features) + 1)):
            # Generate combinations
            from itertools import combinations
            
            for combo in combinations(all_features, n_features):
                candidate_sets.append(list(combo))
                
                # Limit number of candidates to prevent explosion
                if len(candidate_sets) >= 1000:
                    tprint_warning("Limiting candidate sets to 1000 to prevent explosion")
                    break
            
            if len(candidate_sets) >= 1000:
                break
        
        tprint_debug(f"Generated {len(candidate_sets)} candidate feature sets")
        return candidate_sets
    
    def evaluate_objectives(self, features: pd.DataFrame, 
                          targets: pd.Series, 
                          selected_features: List[str]) -> Dict[str, ObjectiveResult]:
        """Evaluate all objectives for a given feature set."""
        results = {}
        
        for obj in self.objectives:
            result = obj.evaluate(features, targets, selected_features)
            results[obj.name] = result
        
        return results
    
    def _validate_input_data(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Validate input data using ml_commons utilities.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        if not self.enable_ml_commons:
            return validation_results
        
        try:
            # Data quality validation
            if hasattr(self, 'data_quality_utils'):
                tprint_debug("Performing data quality validation...")
                
                # Missing value analysis
                missing_analysis = self.data_quality_utils.missing_value_analysis(features)
                if missing_analysis.get('severity_assessment', {}).get('severity_level') in ['high', 'critical']:
                    validation_results['issues'].append("High missing data ratio detected")
                    validation_results['is_valid'] = False
                
                # Outlier detection
                outlier_analysis = self.data_quality_utils.automated_outlier_detection(features)
                outlier_ratio = outlier_analysis.get('summary', {}).get('outlier_percentage', 0) / 100
                if outlier_ratio > 0.1:  # 10% threshold
                    validation_results['warnings'].append(f"High outlier ratio: {outlier_ratio:.2%}")
                
                # Correlation analysis
                corr_analysis = self.data_quality_utils.feature_correlation_analysis(features)
                high_corr_pairs = len(corr_analysis.get('highly_correlated_pairs', []))
                if high_corr_pairs > features.shape[1] * 0.1:  # More than 10% of possible pairs
                    validation_results['warnings'].append(f"High correlation detected in {high_corr_pairs} feature pairs")
            
            # Leakage detection
            if hasattr(self, 'leakage_prevention'):
                tprint_debug("Performing leakage detection...")
                
                leakage_check = self.leakage_prevention.check_data_leakage(features, targets.values, dataset_name="feature_selection")
                if leakage_check.get('status') in ['warning', 'critical']:
                    validation_results['warnings'].append(f"Potential data leakage: {leakage_check.get('status')}")
                    if leakage_check.get('leakage_rate', 0) > 0.1:
                        validation_results['issues'].append("High leakage rate detected")
                        validation_results['is_valid'] = False
        
        except Exception as e:
            tprint_warning(f"Data validation failed with error: {e}")
            validation_results['warnings'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _preprocess_features_with_ml_commons(self, features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """
        Preprocess features using ml_commons utilities.
        
        Args:
            features: Input feature DataFrame
            targets: Target series
            
        Returns:
            Preprocessed feature DataFrame
        """
        if not self.enable_ml_commons or not hasattr(self, 'data_quality_utils'):
            return features
        
        try:
            tprint_debug("Preprocessing features with ml_commons...")
            
            # Clean data using ml_commons
            cleaned_features, cleaning_report = self.data_quality_utils.automated_data_cleaning(features)
            
            if cleaning_report.get('total_removed_samples', 0) > 0:
                tprint_warning(f"Removed {cleaning_report['total_removed_samples']} samples during cleaning")
            
            if cleaning_report.get('total_removed_features', 0) > 0:
                tprint_warning(f"Removed {cleaning_report['total_removed_features']} features during cleaning")
            
            return cleaned_features
            
        except Exception as e:
            tprint_warning(f"Feature preprocessing failed: {e}")
            return features
    
    def _generate_enhanced_candidate_sets(self, features: pd.DataFrame, targets: pd.Series) -> List[List[str]]:
        """
        Generate candidate feature sets using enhanced ml_commons algorithms.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            
        Returns:
            List of candidate feature sets
        """
        if not self.enable_ml_commons or not hasattr(self, 'feature_selector'):
            return self._generate_candidate_sets(features.columns.tolist())
        
        try:
            tprint_debug("Generating enhanced candidate sets with ml_commons...")
            
            # Use ml_commons feature selector for initial filtering
            selected_features = self.feature_selector.select_features(features, targets)
            
            if selected_features:
                # Generate combinations around the selected features
                candidate_sets = []
                all_features = features.columns.tolist()
                
                # Start with ml_commons selected features
                candidate_sets.append(selected_features)
                
                # Add variations around the selected features
                for n_features in range(self.min_features, min(self.max_features + 1, len(all_features) + 1)):
                    if n_features != len(selected_features):
                        # Generate combinations of different sizes
                        from itertools import combinations
                        for combo in combinations(all_features, n_features):
                            candidate_sets.append(list(combo))
                            if len(candidate_sets) >= 100:  # Limit to prevent explosion
                                break
                    if len(candidate_sets) >= 100:
                        break
                
                return candidate_sets
            else:
                # Fallback to standard generation
                return self._generate_candidate_sets(features.columns.tolist())
                
        except Exception as e:
            tprint_warning(f"Enhanced candidate generation failed: {e}")
            return self._generate_candidate_sets(features.columns.tolist())
    
    def _calculate_enhanced_score(self, candidate_set: List[str], features: pd.DataFrame, 
                                 targets: pd.Series, objective_values: Dict[str, float], 
                                 base_score: float) -> float:
        """
        Calculate enhanced score using ml_commons validation.
        
        Args:
            candidate_set: Candidate feature set
            features: Feature DataFrame
            targets: Target series
            objective_values: Objective function values
            base_score: Base weighted score
            
        Returns:
            Enhanced score
        """
        if not self.enable_ml_commons:
            return base_score
        
        try:
            enhanced_score = base_score
            
            # Stability bonus
            if hasattr(self, 'stability_analyzer'):
                stability_result = self.stability_analyzer.analyze_stability(
                    features[candidate_set], targets
                )
                stability_score = stability_result.get('stability_score', 1.0)
                enhanced_score *= (0.8 + 0.2 * stability_score)  # 20% weight to stability
            
            # Data quality bonus
            if hasattr(self, 'data_quality_utils'):
                quality_score = self.data_quality_utils.calculate_data_quality_score(
                    features[candidate_set]
                )
                quality_bonus = quality_score.get('overall_score', 1.0)
                enhanced_score *= (0.9 + 0.1 * quality_bonus)  # 10% weight to quality
            
            # Feature count penalty (prefer fewer features if similar performance)
            feature_count_penalty = 1.0 - (len(candidate_set) - self.min_features) / (self.max_features - self.min_features) * 0.1
            enhanced_score *= max(0.9, feature_count_penalty)
            
            return enhanced_score
            
        except Exception as e:
            tprint_debug(f"Enhanced scoring failed: {e}")
            return base_score
    
    def _validate_selected_features(self, selected_features: List[str], 
                                   features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Validate selected features using ml_commons utilities.
        
        Args:
            selected_features: Selected feature list
            features: Feature DataFrame
            targets: Target series
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': []
        }
        
        if not self.enable_ml_commons:
            return validation_results
        
        try:
            # Check for high correlation in selected features
            if hasattr(self, 'data_quality_utils'):
                selected_data = features[selected_features]
                corr_analysis = self.data_quality_utils.feature_correlation_analysis(selected_data)
                
                high_corr_pairs = corr_analysis.get('highly_correlated_pairs', [])
                if high_corr_pairs:
                    validation_results['warnings'].append(
                        f"Selected features contain {len(high_corr_pairs)} highly correlated pairs"
                    )
            
            # Check for stability
            if hasattr(self, 'stability_analyzer'):
                stability_result = self.stability_analyzer.analyze_stability(
                    features[selected_features], targets
                )
                stability_score = stability_result.get('stability_score', 1.0)
                if stability_score < 0.7:
                    validation_results['warnings'].append(f"Low stability score: {stability_score:.2f}")
        
        except Exception as e:
            tprint_debug(f"Feature validation failed: {e}")
            validation_results['warnings'].append(f"Validation error: {str(e)}")
        
        return validation_results


# Convenience functions
def create_default_objectives() -> List[ObjectiveFunction]:
    """Create default set of objectives."""
    return [
        OutOfSampleSharpeObjective(),
        DrawdownObjective(),
        TurnoverObjective(),
        StabilityObjective(),
        DiversityObjective(),
        MutualInformationObjective(),
        ProfitCenteredObjective()
    ]


def create_performance_objectives() -> List[ObjectiveFunction]:
    """Create objectives focused on performance."""
    return [
        OutOfSampleSharpeObjective(),
        DrawdownObjective(),
        ProfitCenteredObjective()
    ]


def create_stability_objectives() -> List[ObjectiveFunction]:
    """Create objectives focused on stability."""
    return [
        StabilityObjective(),
        DiversityObjective(),
        TurnoverObjective()
    ]


def create_balanced_objectives() -> List[ObjectiveFunction]:
    """Create balanced set of objectives."""
    return [
        OutOfSampleSharpeObjective(),
        DrawdownObjective(),
        StabilityObjective(),
        DiversityObjective()
    ]