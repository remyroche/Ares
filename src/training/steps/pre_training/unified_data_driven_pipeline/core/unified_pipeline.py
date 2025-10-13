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
    interaction_generation_result: Optional[Dict[str, Any]] = None
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
            'interaction_generation_time': 0.0,
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
        
        # Interaction generation (if enabled)
        interaction_result = None
        if self.config.enable_interaction_generation:
            tprint_info("Generating interactions")
            interaction_start = time.time()
            interaction_result = self._generate_interactions(processed_data, processed_targets, pattern_analysis)
            self.performance_stats['interaction_generation_time'] = time.time() - interaction_start
        
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
            interaction_generation_result=interaction_result,
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
        """Optimize periods for different feature types."""
        tprint_debug("Optimizing periods")
        
        # This would implement period optimization
        # For now, return a placeholder
        result = {
            'optimized_periods': {},
            'optimization_method': 'statistical_analysis',
            'confidence_scores': {}
        }
        
        tprint_success("Period optimization completed")
        return result
    
    def _generate_interactions(self, data: pd.DataFrame, 
                             targets: Optional[pd.Series], 
                             patterns: Any) -> Dict[str, Any]:
        """Generate interaction features."""
        tprint_debug("Generating interactions")
        
        # This would implement interaction generation
        # For now, return a placeholder
        result = {
            'generated_interactions': [],
            'interaction_types': [],
            'utility_scores': {}
        }
        
        tprint_success("Interaction generation completed")
        return result
    
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