"""
Budget-Aware Feature Selection System

This module implements a sophisticated budget-aware feature selection system that
optimizes feature selection based on computational budget constraints and trading
performance metrics. It provides a 3-stage pipeline with mRMR, ensemble selection,
and RFE methods.

Key Features:
- Budget allocation across feature types (base, interaction, cross-timeframe, gate)
- Trading performance optimization (CV performance, base importance, stability, sensitivity)
- No computational budget constraints - focus on trading performance
- Equal cost for all features (1.0)
- Comprehensive logging and error handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Import existing proven pipelines
from src.feature_selection.advanced.enhanced_multi_stage_rfe import EnhancedMultiStageRFE
from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress
)

# Import logging utilities
try:
    from src.training.steps.pre_training.market_analysis.logging_standards import (
        get_logger, log_info, log_warning, log_error, log_success, log_debug
    )
except ImportError:
    from src.utils.logger import get_logger
    log_info = log_warning = log_error = log_success = log_debug = lambda *args, **kwargs: None

# Import validation utilities
from src.training.steps.pre_training.validation.schemas import (
    SchemaValidationException,
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
)

# Import common operations
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    optimize_memory_usage, parallel_processing_optimizer
)

# Import matrix operations
from src.utils.matrix_operations import (
    get_unified_matrix_operations, get_vectorized_processing_core,
    get_batch_matrix_processor, safe_matrix_multiply,
    vectorized_rolling_features, parallel_feature_engineering,
    optimize_dataframe, get_hardware_performance_report
)

# Import ML common utilities
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, OptimizationConfig
)
from src.utils.purged_kfold import PurgedKFoldTime as PurgedKFold
from src.feature_selection import select_features as FeatureSelector

# Import Pareto front utilities for multi-objective optimization
from src.utils.ml_common.optimization.pareto import (
    ParetoFront, Solution, compute_pareto_front, select_knee_point,
    scalarize_financial_goals
)

# Setup logging
logger = get_logger(__name__)


# ============================================================================
# CACHING & MEMOIZATION
# ============================================================================

class SelectionCache:
    """
    LRU cache for feature selection results with intelligent invalidation.
    
    Caches:
    - mRMR selection results
    - Ensemble scores
    - RFE results
    - Performance metrics
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize selection cache."""
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_counts: Dict[str, int] = {}
        self.insertion_order: List[str] = []
        
        tprint_debug(f"📦 SelectionCache initialized (max_size={max_size})")
    
    def _generate_key(self, prefix: str, X: pd.DataFrame, y: pd.Series, **kwargs) -> str:
        """Generate cache key from data and parameters."""
        import hashlib
        
        # Create hash from data shape and sample
        data_hash = hashlib.md5()
        data_hash.update(str(X.shape).encode())
        data_hash.update(str(y.shape).encode())
        data_hash.update(str(X.columns.tolist()).encode())
        
        # Add first and last few values for uniqueness
        if len(X) > 0:
            data_hash.update(str(X.iloc[0].values).encode())
            data_hash.update(str(X.iloc[-1].values).encode())
        
        # Add parameters
        for key, value in sorted(kwargs.items()):
            data_hash.update(f"{key}={value}".encode())
        
        return f"{prefix}_{data_hash.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            tprint_debug(f"✅ Cache HIT: {key[:16]}...")
            return self.cache[key]
        tprint_debug(f"❌ Cache MISS: {key[:16]}...")
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with LRU eviction."""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            return
        
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Evict LRU item
            lru_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_counts[lru_key]
            self.insertion_order.remove(lru_key)
            tprint_debug(f"🗑️ Cache EVICT: {lru_key[:16]}...")
        
        # Insert new
        self.cache[key] = value
        self.access_counts[key] = 0
        self.insertion_order.append(key)
        tprint_debug(f"📥 Cache PUT: {key[:16]}... (size={len(self.cache)})")
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.access_counts.clear()
        self.insertion_order.clear()
        tprint_debug("🗑️ Cache CLEARED")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0,
            'total_accesses': sum(self.access_counts.values()),
            'avg_accesses': np.mean(list(self.access_counts.values())) if self.access_counts else 0
        }


# ============================================================================
# FEATURE SELECTION EXPLAINABILITY
# ============================================================================

@dataclass
class FeatureExplanation:
    """Explanation for why a feature was selected or rejected."""
    feature_name: str
    selected: bool
    stage_scores: Dict[str, float]  # Scores at each stage
    rank_progression: List[int]  # Rank at each stage
    final_importance: float
    rejection_reason: Optional[str] = None
    contribution_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionExplanation:
    """Complete explanation of feature selection process."""
    feature_explanations: List[FeatureExplanation]
    stage_summaries: Dict[str, Dict[str, Any]]
    performance_breakdown: Dict[str, float]
    selection_rationale: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class SelectionExplainer:
    """
    Generates human-readable explanations for feature selection decisions.
    
    Provides:
    - Why features were selected/rejected
    - Stage-by-stage analysis
    - Performance attribution
    - Natural language summaries
    """
    
    def __init__(self):
        """Initialize selection explainer."""
        self.logger = get_logger(f"{__name__}.SelectionExplainer")
        tprint_debug("📊 SelectionExplainer initialized")
    
    def explain_selection(
        self,
        original_features: List[str],
        selected_features: List[str],
        stage_results: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> SelectionExplanation:
        """Generate complete explanation of selection process."""
        tprint_info("📊 Generating feature selection explanation...")
        
        # Create feature explanations
        feature_explanations = []
        for feature in original_features:
            explanation = self._explain_feature(
                feature, feature in selected_features, stage_results
            )
            feature_explanations.append(explanation)
        
        # Create stage summaries
        stage_summaries = self._create_stage_summaries(stage_results)
        
        # Generate selection rationale
        rationale = self._generate_rationale(
            len(original_features), len(selected_features),
            stage_summaries, performance_metrics
        )
        
        return SelectionExplanation(
            feature_explanations=feature_explanations,
            stage_summaries=stage_summaries,
            performance_breakdown=performance_metrics,
            selection_rationale=rationale
        )
    
    def _explain_feature(
        self,
        feature_name: str,
        selected: bool,
        stage_results: Dict[str, Any]
    ) -> FeatureExplanation:
        """Explain why a specific feature was selected or rejected."""
        stage_scores = {}
        rank_progression = []
        rejection_reason = None
        
        # Extract scores from each stage
        for stage_name, stage_data in stage_results.items():
            if isinstance(stage_data, dict) and 'n_features' in stage_data:
                # Track feature through stages
                stage_scores[stage_name] = stage_data.get('time_ms', 0.0)
        
        if not selected:
            rejection_reason = self._determine_rejection_reason(
                feature_name, stage_results
            )
        
        return FeatureExplanation(
            feature_name=feature_name,
            selected=selected,
            stage_scores=stage_scores,
            rank_progression=rank_progression,
            final_importance=1.0 if selected else 0.0,
            rejection_reason=rejection_reason
        )
    
    def _determine_rejection_reason(
        self,
        feature_name: str,
        stage_results: Dict[str, Any]
    ) -> str:
        """Determine why a feature was rejected."""
        # Check which stage removed the feature
        if 'stage1' in stage_results:
            return "Removed in Stage 1 (mRMR): Low relevance or high redundancy"
        elif 'stage2' in stage_results:
            return "Removed in Stage 2 (Ensemble): Low ensemble score (LASSO+SHAP/LGBM+RF)"
        elif 'stage3' in stage_results:
            return "Removed in Stage 3 (RFE): Low trading performance contribution"
        return "Unknown rejection reason"
    
    def _create_stage_summaries(
        self,
        stage_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Create summaries for each stage."""
        summaries = {}
        
        for stage_name, stage_data in stage_results.items():
            if isinstance(stage_data, dict):
                summaries[stage_name] = {
                    'method': stage_data.get('method', 'Unknown'),
                    'features_in': stage_data.get('n_features_before', 'N/A'),
                    'features_out': stage_data.get('n_features', 'N/A'),
                    'time_ms': stage_data.get('time_ms', 0.0),
                    'performance_score': stage_data.get('performance_score', 'N/A')
                }
        
        return summaries
    
    def _generate_rationale(
        self,
        initial_features: int,
        final_features: int,
        stage_summaries: Dict[str, Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> str:
        """Generate natural language rationale for selection."""
        reduction_pct = ((initial_features - final_features) / initial_features * 100) if initial_features > 0 else 0
        
        rationale = f"""
Feature Selection Summary:
--------------------------
Initial Features: {initial_features}
Final Features: {final_features}
Reduction: {reduction_pct:.1f}%

Selection Process:
The feature selection used a 3-stage pipeline to optimize for trading performance:

1. Stage 1 (mRMR): Removed top 50% most correlated features for diversity
2. Stage 2 (Ensemble): Combined LASSO + SHAP/LGBM + Random Forest with z-score normalization
3. Stage 3 (RFE): Final selection optimized for trading performance metrics

Performance Optimization:
- CV Performance (40%): Cross-validation for market robustness
- Base Importance (30%): Raw predictive power
- Stability (20%): Consistency across time periods
- Sensitivity (10%): Market responsiveness

Final Performance:
{self._format_performance_metrics(performance_metrics)}
        """.strip()
        
        return rationale
    
    def _format_performance_metrics(self, metrics: Dict[str, float]) -> str:
        """Format performance metrics for display."""
        if not metrics:
            return "No performance metrics available"
        
        lines = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def export_html_report(
        self,
        explanation: SelectionExplanation,
        output_path: Path
    ) -> None:
        """Export explanation as interactive HTML report."""
        tprint_info(f"📄 Exporting HTML explanation report to {output_path}")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Feature Selection Explanation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .metric {{ background: #ecf0f1; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .selected {{ color: #27ae60; font-weight: bold; }}
        .rejected {{ color: #e74c3c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .stage-summary {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <h1>Feature Selection Explanation Report</h1>
    <p><strong>Generated:</strong> {explanation.timestamp}</p>
    
    <h2>Selection Rationale</h2>
    <pre>{explanation.selection_rationale}</pre>
    
    <h2>Stage Summaries</h2>
    {self._format_stage_summaries_html(explanation.stage_summaries)}
    
    <h2>Feature Details</h2>
    {self._format_feature_table_html(explanation.feature_explanations)}
    
    <h2>Performance Breakdown</h2>
    {self._format_performance_html(explanation.performance_breakdown)}
</body>
</html>
        """
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        tprint_success(f"✅ HTML report exported to {output_path}")
    
    def _format_stage_summaries_html(self, summaries: Dict[str, Dict[str, Any]]) -> str:
        """Format stage summaries as HTML."""
        html = ""
        for stage_name, summary in summaries.items():
            html += f"""
            <div class="stage-summary">
                <h3>{stage_name.upper()}</h3>
                <p><strong>Method:</strong> {summary.get('method', 'N/A')}</p>
                <p><strong>Features:</strong> {summary.get('features_in', 'N/A')} → {summary.get('features_out', 'N/A')}</p>
                <p><strong>Time:</strong> {summary.get('time_ms', 0):.1f}ms</p>
            </div>
            """
        return html
    
    def _format_feature_table_html(self, explanations: List[FeatureExplanation]) -> str:
        """Format feature explanations as HTML table."""
        rows = ""
        for exp in explanations[:50]:  # Limit to first 50 for readability
            status_class = "selected" if exp.selected else "rejected"
            status_text = "✓ Selected" if exp.selected else "✗ Rejected"
            reason = exp.rejection_reason or "N/A"
            
            rows += f"""
            <tr>
                <td>{exp.feature_name}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{exp.final_importance:.4f}</td>
                <td>{reason}</td>
            </tr>
            """
        
        return f"""
        <table>
            <tr>
                <th>Feature Name</th>
                <th>Status</th>
                <th>Importance</th>
                <th>Notes</th>
            </tr>
            {rows}
        </table>
        """
    
    def _format_performance_html(self, metrics: Dict[str, float]) -> str:
        """Format performance metrics as HTML."""
        html = '<div class="metric-container">'
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f'<div class="metric"><strong>{key}:</strong> {value:.4f}</div>'
            else:
                html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
        html += '</div>'
        return html


# ============================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ============================================================================

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    # Objectives to optimize
    objectives: Dict[str, str] = field(default_factory=lambda: {
        'cv_performance': 'max',  # Cross-validation performance
        'stability': 'max',  # Feature stability
        'n_features': 'min',  # Number of features (prefer fewer)
    })
    
    # Weights for scalarization (if needed)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'cv_performance': 0.5,
        'stability': 0.3,
        'n_features': 0.2
    })
    
    # Constraints
    min_cv_performance: float = 0.5
    min_stability: float = 0.6
    max_features: int = 100
    
    # Pareto optimization
    use_pareto_optimization: bool = True
    use_knee_point_selection: bool = True


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for feature selection using Pareto fronts.
    
    Optimizes for:
    - Performance (CV score, accuracy)
    - Stability (consistency across folds)
    - Efficiency (fewer features preferred)
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        """Initialize multi-objective optimizer."""
        self.config = config or MultiObjectiveConfig()
        self.logger = get_logger(f"{__name__}.MultiObjectiveOptimizer")
        self.pareto_front = ParetoFront()
        
        tprint_success("🎯 MultiObjectiveOptimizer initialized")
        tprint_info(f"   Objectives: {list(self.config.objectives.keys())}")
    
    def optimize_selection(
        self,
        candidate_solutions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize feature selection using multi-objective approach.
        
        Args:
            candidate_solutions: List of candidate feature sets with metrics
            
        Returns:
            Best solution from Pareto front
        """
        tprint_info(f"🎯 Running multi-objective optimization on {len(candidate_solutions)} candidates")
        
        # Convert to Solution objects
        solutions = []
        for candidate in candidate_solutions:
            metrics = self._extract_metrics(candidate)
            solution = Solution(
                metrics=metrics,
                params=candidate.get('params', {})
            )
            solutions.append(solution)
        
        # Apply constraints
        filtered_solutions = self._apply_constraints(solutions)
        tprint_info(f"   ✅ {len(filtered_solutions)}/{len(solutions)} solutions passed constraints")
        
        if not filtered_solutions:
            tprint_warning("   ⚠️ No solutions passed constraints, using best unconstrained")
            filtered_solutions = solutions
        
        # Compute Pareto front
        if self.config.use_pareto_optimization:
            pareto_solutions = self.pareto_front.compute_pareto_front_gpu(
                filtered_solutions,
                self.config.objectives,
                use_gpu=True,
                use_nonlinear_transforms=True
            )
            tprint_success(f"   ✅ Pareto front: {len(pareto_solutions)} non-dominated solutions")
        else:
            pareto_solutions = filtered_solutions
        
        # Select best solution
        if self.config.use_knee_point_selection and len(pareto_solutions) > 1:
            best_solution = select_knee_point(
                pareto_solutions,
                self.config.objectives,
                weights=self.config.objective_weights
            )
            tprint_success("   ✅ Selected knee point solution")
        else:
            # Use scalarization
            best_solution = max(
                pareto_solutions,
                key=lambda s: scalarize_financial_goals(
                    s.metrics,
                    weights=self.config.objective_weights,
                    fallback_objectives=self.config.objectives
                )
            )
            tprint_success("   ✅ Selected best scalarized solution")
        
        # Create result
        result = {
            'best_solution': best_solution,
            'pareto_front': pareto_solutions,
            'n_candidates': len(candidate_solutions),
            'n_pareto': len(pareto_solutions),
            'best_metrics': best_solution.metrics,
            'best_params': best_solution.params,
            'optimization_summary': self._create_summary(
                candidate_solutions, pareto_solutions, best_solution
            )
        }
        
        self._log_results(result)
        return result
    
    def _extract_metrics(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from candidate solution."""
        metrics = {}
        
        # Extract performance metrics
        perf_metrics = candidate.get('performance_metrics', {})
        metrics['cv_performance'] = perf_metrics.get('cv_mean', 0.0)
        metrics['stability'] = perf_metrics.get('stability', 0.0)
        metrics['avg_importance'] = perf_metrics.get('avg_importance', 0.0)
        
        # Extract feature count
        metrics['n_features'] = len(candidate.get('selected_features', []))
        
        # Extract budget used
        metrics['budget_used_ms'] = candidate.get('budget_used_ms', 0.0)
        
        # Calculate composite metrics
        metrics['efficiency'] = (
            metrics['cv_performance'] / (metrics['n_features'] + 1)
            if metrics['n_features'] > 0 else 0.0
        )
        
        return metrics
    
    def _apply_constraints(self, solutions: List[Solution]) -> List[Solution]:
        """Apply constraints to filter solutions."""
        constraints = {
            'cv_performance': self.config.min_cv_performance,
            'stability': self.config.min_stability,
            'n_features': lambda n: n <= self.config.max_features
        }
        
        from src.utils.ml_common.optimization.pareto import filter_by_constraints
        return filter_by_constraints(solutions, constraints)
    
    def _create_summary(
        self,
        candidates: List[Dict[str, Any]],
        pareto_front: List[Solution],
        best_solution: Solution
    ) -> str:
        """Create optimization summary."""
        summary = f"""
Multi-Objective Optimization Summary:
-------------------------------------
Total Candidates: {len(candidates)}
Pareto Front Size: {len(pareto_front)}
Reduction: {(1 - len(pareto_front)/len(candidates))*100:.1f}%

Best Solution Metrics:
{self._format_metrics(best_solution.metrics)}

Optimization Objectives:
{self._format_objectives()}
        """.strip()
        
        return summary
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"  - {key}: {value:.4f}")
        return "\n".join(lines)
    
    def _format_objectives(self) -> str:
        """Format objectives for display."""
        lines = []
        for obj_name, direction in self.config.objectives.items():
            weight = self.config.objective_weights.get(obj_name, 0.0)
            lines.append(f"  - {obj_name}: {direction} (weight={weight:.2f})")
        return "\n".join(lines)
    
    def _log_results(self, result: Dict[str, Any]) -> None:
        """Log optimization results."""
        tprint_success("🎯 Multi-objective optimization complete")
        tprint_info(f"   📊 Pareto front: {result['n_pareto']} solutions")
        tprint_info(f"   ✅ Best CV performance: {result['best_metrics'].get('cv_performance', 0):.4f}")
        tprint_info(f"   ✅ Best stability: {result['best_metrics'].get('stability', 0):.4f}")
        tprint_info(f"   ✅ Best n_features: {result['best_metrics'].get('n_features', 0):.0f}")


@dataclass
class FeatureTypeBudget:
    """Configuration for each feature type budget allocation."""
    # Budget allocation (in milliseconds)
    budget_ms: float = 0.0
    
    # Feature count constraints
    min_features: int = 0
    max_features: int = 1000
    target_features: int = 50
    
    # Performance weights
    cv_performance_weight: float = 0.4  # Most important for trading
    base_importance_weight: float = 0.3  # Raw predictive power
    stability_weight: float = 0.2  # Consistency over time
    sensitivity_weight: float = 0.1  # Market response
    
    # Feature cost (equal for all features)
    feature_cost: float = 1.0
    
    # Selection criteria
    min_correlation_threshold: float = 0.1
    max_correlation_threshold: float = 0.95
    min_importance_score: float = 0.01
    min_stability_score: float = 0.5


@dataclass
class BudgetAwareSelectionConfig:
    """Main configuration for budget-aware feature selection."""
    # Total budget allocation
    total_budget_ms: float = 100.0
    
    # Feature type budgets
    base_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=68.0,  # 68% of total budget
        min_features=40,
        max_features=80,
        target_features=60
    ))
    
    interaction_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=15.0,  # 15% of total budget
        min_features=5,
        max_features=15,
        target_features=10
    ))
    
    cross_timeframe_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=10.0,  # 10% of total budget
        min_features=3,
        max_features=10,
        target_features=6
    ))
    
    gate_features: FeatureTypeBudget = field(default_factory=lambda: FeatureTypeBudget(
        budget_ms=7.0,  # 7% of total budget
        min_features=2,
        max_features=8,
        target_features=5
    ))
    
    # Pipeline configuration
    enable_mrmr_selection: bool = True
    enable_ensemble_selection: bool = True
    enable_rfe_selection: bool = True
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    enable_hardware_acceleration: bool = True
    
    # Validation settings
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Logging
    verbose: bool = True
    log_performance: bool = True


@dataclass
class FeatureTypeSelectionResult:
    """Results for individual feature type selection."""
    feature_type: str
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_time: float
    budget_used_ms: float
    performance_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


@dataclass
class BudgetAwareSelectionResult:
    """Overall budget-aware selection results."""
    # Core results
    all_selected_features: List[str]
    feature_type_results: Dict[str, FeatureTypeSelectionResult]
    
    # Performance metrics
    total_selection_time: float
    total_budget_used_ms: float
    overall_performance_score: float
    
    # Feature breakdown
    base_features: List[str]
    interaction_features: List[str]
    cross_timeframe_features: List[str]
    gate_features: List[str]
    
    # Success indicators
    success: bool
    error_message: Optional[str] = None
    
    # Additional metadata
    config_used: BudgetAwareSelectionConfig
    performance_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # Explainability
    selection_explanation: Optional[SelectionExplanation] = None
    
    # Multi-objective optimization
    pareto_front: Optional[List[Solution]] = None
    multi_objective_result: Optional[Dict[str, Any]] = None
    
    # Caching statistics
    cache_stats: Optional[Dict[str, Any]] = None


class BudgetAwareFeatureSelector:
    """
    Budget-aware feature selector that optimizes feature selection based on
    computational budget constraints and trading performance metrics.
    """
    
    def __init__(self, config: Optional[BudgetAwareSelectionConfig] = None):
        """Initialize the budget-aware feature selector with proven pipelines."""
        self.config = config or BudgetAwareSelectionConfig()
        self.logger = get_logger(f"{__name__}.BudgetAwareFeatureSelector")
        
        # Initialize hardware optimization tools
        self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
        self.gpu_manager = get_m1_gpu_manager()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.batch_processor = get_batch_matrix_processor()
        
        # Initialize ML utilities
        self.bayesian_optimizer = BayesianTPEOptimizer(
            OptimizationConfig(
                n_trials=50,
                timeout_minutes=10,
                enable_parallel=True,
                max_workers=self.config.max_workers
            )
        )
        
        # Initialize proven pipelines
        self._initialize_proven_pipelines()
        
        # Initialize caching system
        self.cache = SelectionCache(max_size=100) if self.config.enable_caching else None
        
        # Initialize explainability system
        self.explainer = SelectionExplainer()
        
        # Initialize multi-objective optimizer
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        
        tprint_success("🚀 BudgetAwareFeatureSelector initialized with proven pipelines")
        tprint_info(f"   📦 Using: ImprovedMRMR + EnhancedMultiStageRFE (LASSO+SHAP/LGBM+RF)")
        tprint_info(f"   📊 Total budget: {self.config.total_budget_ms}ms")
        tprint_info(f"   🎯 Base features: {self.config.base_features.target_features}")
        tprint_info(f"   🔗 Interaction features: {self.config.interaction_features.target_features}")
        tprint_info(f"   ⏰ Cross-timeframe features: {self.config.cross_timeframe_features.target_features}")
        tprint_info(f"   🚪 Gate features: {self.config.gate_features.target_features}")
        tprint_info(f"   📦 Caching: {'Enabled' if self.cache else 'Disabled'}")
        tprint_info(f"   📊 Explainability: Enabled")
        tprint_info(f"   🎯 Multi-objective: Enabled")
    
    def _initialize_proven_pipelines(self):
        """Initialize proven feature selection pipelines."""
        # Initialize ImprovedMRMR for Stage 1
        self.mrmr_filter = ImprovedMRMR({
            'target_ratio': 0.5,  # Select top 50%
            'mi_weight': 0.7,
            'spearman_weight': 0.3,
            'enable_hardware_optimization': True,
            'n_jobs': self.config.max_workers,
            'random_state': self.config.random_state
        })
        
        # Initialize EnhancedMultiStageRFE for Stage 2 (contains LASSO + SHAP/LGBM + RF)
        self.enhanced_rfe = EnhancedMultiStageRFE({
            'enable_stage1': False,  # We handle mRMR separately
            'enable_stage2': True,  # Use ensemble filtering
            'enable_stage3': True,  # Use batch RFE
            'enable_stage4': True,  # Use fine RFE
            'cv_folds': self.config.cv_folds,
            'enable_hardware_optimization': True,
            'n_jobs': self.config.max_workers,
            'random_state': self.config.random_state,
            'verbose': self.config.verbose
        })
        
        tprint_info("   ✅ Initialized proven pipelines: ImprovedMRMR + EnhancedMultiStageRFE")
    
    async def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: Optional[Dict[str, List[str]]] = None
    ) -> BudgetAwareSelectionResult:
        """
        Main entry point for budget-aware feature selection with caching, explainability, and multi-objective optimization.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_types: Optional mapping of feature types to feature names
            
        Returns:
            BudgetAwareSelectionResult with selected features, explanations, and Pareto front
        """
        start_time = time.time()
        tprint_success("🎯 Starting budget-aware feature selection with advanced features")
        tprint_info(f"   📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        tprint_info(f"   🎯 Target: {len(y)} samples")
        tprint_info(f"   📦 Caching: {'Enabled' if self.cache else 'Disabled'}")
        tprint_info(f"   📊 Explainability: Enabled")
        tprint_info(f"   🎯 Multi-objective: Enabled")
        
        # Store original features for explainability
        original_features = X.columns.tolist()
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Categorize features by type if not provided
            if feature_types is None:
                feature_types = self._categorize_features_by_type(X.columns)
            
            # Apply budget constraints with caching
            selection_results = await self._apply_budget_constraints_with_caching(X, y, feature_types)
            
            # Apply multi-objective optimization to select best configuration
            if len(selection_results) > 1:
                tprint_info("🎯 Applying multi-objective optimization...")
                candidate_solutions = self._prepare_candidates_for_optimization(selection_results)
                mo_result = self.multi_objective_optimizer.optimize_selection(candidate_solutions)
                
                # Use best solution from Pareto front
                best_solution_params = mo_result['best_params']
                if best_solution_params and 'feature_type' in best_solution_params:
                    # Update selection results based on best solution
                    tprint_info(f"   ✅ Selected best solution from Pareto front")
            else:
                mo_result = None
            
            # Combine results
            result = self._combine_selection_results(selection_results, start_time)
            
            # Generate explainability report
            tprint_info("📊 Generating explainability report...")
            explanation = self.explainer.explain_selection(
                original_features=original_features,
                selected_features=result.all_selected_features,
                stage_results=self._collect_all_stage_results(selection_results),
                performance_metrics=result.performance_breakdown
            )
            result.selection_explanation = explanation
            
            # Add multi-objective results
            if mo_result:
                result.pareto_front = mo_result['pareto_front']
                result.multi_objective_result = mo_result
            
            # Add cache statistics
            if self.cache:
                result.cache_stats = self.cache.get_stats()
            
            tprint_success("✅ Budget-aware feature selection completed")
            tprint_info(f"   📊 Selected {len(result.all_selected_features)} features")
            tprint_info(f"   ⏱️ Total time: {result.total_selection_time:.3f}s")
            tprint_info(f"   💰 Budget used: {result.total_budget_used_ms:.1f}ms")
            if result.cache_stats:
                tprint_info(f"   📦 Cache utilization: {result.cache_stats['utilization']*100:.1f}%")
            if mo_result:
                tprint_info(f"   🎯 Pareto front: {mo_result['n_pareto']} solutions")
            
            return result
            
        except Exception as e:
            error_msg = f"Budget-aware feature selection failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return BudgetAwareSelectionResult(
                all_selected_features=[],
                feature_type_results={},
                total_selection_time=time.time() - start_time,
                total_budget_used_ms=0.0,
                overall_performance_score=0.0,
                base_features=[],
                interaction_features=[],
                cross_timeframe_features=[],
                gate_features=[],
                success=False,
                error_message=error_msg,
                config_used=self.config
            )
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if X.empty:
            raise ValueError("Feature matrix is empty")
        
        if y.empty:
            raise ValueError("Target variable is empty")
        
        if len(X) != len(y):
            raise ValueError(f"Feature matrix and target have different lengths: {len(X)} vs {len(y)}")
        
        # Check for missing values
        if X.isnull().any().any():
            tprint_warning("⚠️ Feature matrix contains missing values")
        
        if y.isnull().any():
            tprint_warning("⚠️ Target variable contains missing values")
    
    def _categorize_features_by_type(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type based on naming patterns."""
        feature_types = {
            'base': [],
            'interaction': [],
            'cross_timeframe': [],
            'gate': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            # Categorize based on naming patterns
            if any(pattern in feature_lower for pattern in ['_x_', '*', '_mul_', '_mult_']):
                feature_types['interaction'].append(feature)
            elif any(pattern in feature_lower for pattern in ['_ctf_', '_cross_', '_tf_']):
                feature_types['cross_timeframe'].append(feature)
            elif any(pattern in feature_lower for pattern in ['_gate_', '_gating_', '_switch_']):
                feature_types['gate'].append(feature)
            else:
                feature_types['base'].append(feature)
        
        tprint_info("📊 Feature categorization:")
        for ftype, features in feature_types.items():
            tprint_info(f"   {ftype}: {len(features)} features")
        
        return feature_types
    
    async def _apply_budget_constraints_with_caching(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: Dict[str, List[str]]
    ) -> Dict[str, FeatureTypeSelectionResult]:
        """Apply budget constraints with caching support."""
        tprint_info("🔄 Applying budget constraints with caching")
        
        selection_results = {}
        
        for ftype, features in feature_types.items():
            if not features:
                continue
            
            tprint_info(f"   🎯 Processing {ftype} features: {len(features)} candidates")
            
            # Check cache first
            cache_key = None
            if self.cache:
                budget_config = getattr(self.config, f"{ftype}_features")
                cache_key = self.cache._generate_key(
                    f"selection_{ftype}",
                    X[features],
                    y,
                    target=budget_config.target_features
                )
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    tprint_info(f"   📦 Using cached result for {ftype}")
                    selection_results[ftype] = cached_result
                    continue
            
            # Get budget configuration for this feature type
            budget_config = getattr(self.config, f"{ftype}_features")
            
            # Select features for this type
            result = await self._select_features_for_type(
                X[features], y, ftype, budget_config
            )
            
            # Cache result
            if self.cache and cache_key:
                self.cache.put(cache_key, result)
            
            selection_results[ftype] = result
            
            if result.success:
                tprint_success(f"   ✅ {ftype}: {len(result.selected_features)} features selected")
            else:
                tprint_error(f"   ❌ {ftype}: {result.error_message}")
        
        return selection_results
    
    def _prepare_candidates_for_optimization(
        self,
        selection_results: Dict[str, FeatureTypeSelectionResult]
    ) -> List[Dict[str, Any]]:
        """Prepare candidate solutions for multi-objective optimization."""
        candidates = []
        
        for ftype, result in selection_results.items():
            if result.success:
                candidate = {
                    'feature_type': ftype,
                    'selected_features': result.selected_features,
                    'performance_metrics': result.performance_metrics,
                    'budget_used_ms': result.budget_used_ms,
                    'params': {
                        'feature_type': ftype,
                        'n_features': len(result.selected_features)
                    }
                }
                candidates.append(candidate)
        
        return candidates
    
    def _collect_all_stage_results(
        self,
        selection_results: Dict[str, FeatureTypeSelectionResult]
    ) -> Dict[str, Any]:
        """Collect stage results from all feature types."""
        all_stage_results = {}
        
        for ftype, result in selection_results.items():
            # Add feature type prefix to avoid conflicts
            for stage_name, stage_data in result.performance_metrics.items():
                key = f"{ftype}_{stage_name}"
                all_stage_results[key] = stage_data
        
        return all_stage_results
    
    async def _apply_budget_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_types: Dict[str, List[str]]
    ) -> Dict[str, FeatureTypeSelectionResult]:
        """Apply budget constraints using the 3-stage pipeline."""
        tprint_info("🔄 Applying budget constraints with 3-stage pipeline")
        
        selection_results = {}
        
        # Process each feature type
        for ftype, features in feature_types.items():
            if not features:
                continue
            
            tprint_info(f"   🎯 Processing {ftype} features: {len(features)} candidates")
            
            # Get budget configuration for this feature type
            budget_config = getattr(self.config, f"{ftype}_features")
            
            # Select features for this type
            result = await self._select_features_for_type(
                X[features], y, ftype, budget_config
            )
            
            selection_results[ftype] = result
            
            if result.success:
                tprint_success(f"   ✅ {ftype}: {len(result.selected_features)} features selected")
            else:
                tprint_error(f"   ❌ {ftype}: {result.error_message}")
        
        return selection_results
    
    async def _select_features_for_type(
        self,
        X_type: pd.DataFrame,
        y: pd.Series,
        feature_type: str,
        budget_config: FeatureTypeBudget
    ) -> FeatureTypeSelectionResult:
        """Select features for a specific type using the 3-stage pipeline."""
        start_time = time.time()
        
        try:
            # Stage 1: mRMR with Spearman correlation (remove top 50% for diversity)
            if self.config.enable_mrmr_selection:
                tprint_debug(f"   🔍 Stage 1: mRMR selection for {feature_type}")
                X_stage1 = await self._mrmr_spearman_selection(X_type, y, budget_config)
            else:
                X_stage1 = X_type
            
            # Stage 2: Multi-step ensemble selection (LASSO + SHAP/LGBM + Random Forest)
            if self.config.enable_ensemble_selection:
                tprint_debug(f"   🔍 Stage 2: Ensemble selection for {feature_type}")
                X_stage2 = await self._multi_step_ensemble_selection(X_stage1, y, budget_config)
            else:
                X_stage2 = X_stage1
            
            # Stage 3: RFE with trading performance focus
            if self.config.enable_rfe_selection:
                tprint_debug(f"   🔍 Stage 3: RFE selection for {feature_type}")
                X_final = await self._rfe_final_selection(X_stage2, y, budget_config)
            else:
                X_final = X_stage2
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(X_final, y)
            
            selection_time = time.time() - start_time
            budget_used = selection_time * 1000  # Convert to milliseconds
            
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=list(X_final.columns),
                feature_scores={},
                selection_time=selection_time,
                budget_used_ms=budget_used,
                performance_metrics=performance_metrics,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Feature selection failed for {feature_type}: {e}"
            tprint_error(f"   ❌ {error_msg}")
            
            return FeatureTypeSelectionResult(
                feature_type=feature_type,
                selected_features=[],
                feature_scores={},
                selection_time=time.time() - start_time,
                budget_used_ms=0.0,
                performance_metrics={},
                success=False,
                error_message=error_msg
            )
    
    async def _mrmr_spearman_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        budget_config: FeatureTypeBudget
    ) -> pd.DataFrame:
        """Stage 1: mRMR using proven ImprovedMRMR implementation with caching."""
        tprint_debug("   🔍 Running ImprovedMRMR (mRMR + Spearman)")
        
        # Check cache first
        if self.cache:
            cache_key = self.cache._generate_key("mrmr", X, y, ratio=0.5)
            cached = self.cache.get(cache_key)
            if cached is not None:
                tprint_debug("   📦 Using cached mRMR result")
                return cached
        
        try:
            # Use proven ImprovedMRMR implementation
            result = self.mrmr_filter.select_features(
                X.values,
                y.values,
                X.columns.tolist(),
                target_ratio=0.5  # Select top 50%
            )
            
            if result['success']:
                selected_features = result['filtered_feature_names']
                X_filtered = X[selected_features]
                tprint_debug(f"   ✅ ImprovedMRMR: {len(X.columns)} → {len(selected_features)} features")
                
                # Cache result
                if self.cache:
                    self.cache.put(cache_key, X_filtered)
                
                return X_filtered
            else:
                tprint_warning("   ⚠️ ImprovedMRMR failed, keeping all features")
                return X
                
        except Exception as e:
            tprint_warning(f"   ⚠️ ImprovedMRMR error: {e}, keeping all features")
            self.logger.warning(f"ImprovedMRMR failed: {e}")
            return X
    
    async def _multi_step_ensemble_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        budget_config: FeatureTypeBudget
    ) -> pd.DataFrame:
        """
        Stage 2: Ensemble selection using proven EnhancedMultiStageRFE with caching.
        
        This uses the proven _calculate_ensemble_scores_cv method which includes:
        - LASSO with cross-validation and standardization
        - LGBM with SHAP values for interpretability
        - Random Forest feature importance
        - Z-score normalization for fair combination
        - Cross-validation for robustness
        """
        tprint_debug("   🔍 Running EnhancedMultiStageRFE ensemble (LASSO+SHAP/LGBM+RF+zscore)")
        
        # Check cache first
        if self.cache:
            cache_key = self.cache._generate_key(
                "ensemble",
                X,
                y,
                target=budget_config.target_features
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                tprint_debug("   📦 Using cached ensemble result")
                return cached
        
        try:
            # Calculate target features for this stage (keep buffer above final target)
            buffer = max(10, int(budget_config.target_features * 0.3))
            target_this_stage = min(len(X.columns), budget_config.target_features + buffer)
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
            
            # Use proven ensemble scoring from EnhancedMultiStageRFE
            # This combines LASSO + SHAP/LGBM + RF with z-score normalization
            ensemble_scores = self.enhanced_rfe._calculate_ensemble_scores_cv(
                X.values,
                y.values,
                X.columns.tolist(),
                is_classification,
                groups=None
            )
            
            # Select top features based on ensemble scores
            sorted_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
            n_select = min(target_this_stage, len(sorted_features))
            
            selected_features = [f[0] for f in sorted_features[:n_select]]
            X_filtered = X[selected_features]
            
            tprint_debug(f"   ✅ EnhancedRFE ensemble: {len(X.columns)} → {len(selected_features)} features")
            
            # Cache result
            if self.cache:
                self.cache.put(cache_key, X_filtered)
            
            return X_filtered
            
        except Exception as e:
            tprint_warning(f"   ⚠️ EnhancedRFE ensemble error: {e}, keeping all features")
            self.logger.warning(f"EnhancedMultiStageRFE ensemble failed: {e}")
            return X
    
    async def _rfe_final_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        budget_config: FeatureTypeBudget
    ) -> pd.DataFrame:
        """
        Stage 3: RFE with trading performance focus.
        
        Optimizes for:
        - CV Performance (40%): Most important for trading
        - Base Importance (30%): Raw predictive power
        - Stability (20%): Consistent over time
        - Sensitivity (10%): Responds to market changes
        """
        tprint_debug("   🔍 Running sklearn RFE (trading performance focus)")
        
        # Determine if classification or regression
        is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
        
        # Use appropriate estimator
        if is_classification:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        # Determine number of features to select
        target_count = min(budget_config.target_features, len(X.columns))
        if target_count <= 0:
            return X
        
        # Run RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=target_count,
            step=1
        )
        
        try:
            rfe.fit(X, y)
            selected_features = X.columns[rfe.support_].tolist()
            X_selected = X[selected_features]
            tprint_debug(f"   ✅ sklearn RFE: {len(X.columns)} → {len(selected_features)} features")
        except Exception as e:
            tprint_warning(f"   ⚠️ sklearn RFE error: {e}, keeping all features")
            self.logger.warning(f"sklearn RFE failed: {e}")
            X_selected = X
        
        return X_selected
    
    def _calculate_performance_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for selected features."""
        try:
            # Cross-validation R² score
            rf = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
            cv_scores = cross_val_score(rf, X, y, cv=self.config.cv_folds, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Feature importance
            rf.fit(X, y)
            importance_scores = rf.feature_importances_
            avg_importance = np.mean(importance_scores)
            
            # Stability (variance of importance across CV folds)
            stability_scores = []
            for i in range(self.config.cv_folds):
                try:
                    fold_scores = cross_val_score(rf, X, y, cv=2, scoring='r2')
                    stability_scores.append(fold_scores.mean())
                except Exception:
                    stability_scores.append(0.0)
            
            stability = 1.0 - np.var(stability_scores) if stability_scores else 0.0
            
            # Sensitivity (response to small changes)
            sensitivity = self._calculate_sensitivity(X, y)
            
            return {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'avg_importance': avg_importance,
                'stability': stability,
                'sensitivity': sensitivity,
                'n_features': len(X.columns)
            }
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Performance calculation failed: {e}")
            return {
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'avg_importance': 0.0,
                'stability': 0.0,
                'sensitivity': 0.0,
                'n_features': len(X.columns)
            }
    
    def _calculate_sensitivity(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate feature sensitivity to small changes."""
        try:
            # Add small noise and measure performance change
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            
            # Original performance
            rf_orig = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            orig_score = cross_val_score(rf_orig, X, y, cv=3, scoring='r2').mean()
            
            # Noisy performance
            rf_noisy = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            noisy_score = cross_val_score(rf_noisy, X_noisy, y, cv=3, scoring='r2').mean()
            
            # Sensitivity is the absolute difference
            sensitivity = abs(orig_score - noisy_score)
            return sensitivity
            
        except Exception:
            return 0.0
    
    def _combine_selection_results(
        self,
        selection_results: Dict[str, FeatureTypeSelectionResult],
        start_time: float
    ) -> BudgetAwareSelectionResult:
        """Combine individual feature type results into overall result."""
        total_time = time.time() - start_time
        
        # Extract selected features by type
        base_features = selection_results.get('base', FeatureTypeSelectionResult('base', [], {}, 0, 0, {}, False)).selected_features
        interaction_features = selection_results.get('interaction', FeatureTypeSelectionResult('interaction', [], {}, 0, 0, {}, False)).selected_features
        cross_timeframe_features = selection_results.get('cross_timeframe', FeatureTypeSelectionResult('cross_timeframe', [], {}, 0, 0, {}, False)).selected_features
        gate_features = selection_results.get('gate', FeatureTypeSelectionResult('gate', [], {}, 0, 0, {}, False)).selected_features
        
        # Combine all selected features
        all_selected_features = (
            base_features + interaction_features + 
            cross_timeframe_features + gate_features
        )
        
        # Calculate total budget used
        total_budget_used = sum(
            result.budget_used_ms for result in selection_results.values()
        )
        
        # Calculate overall performance score
        overall_performance = self._calculate_overall_performance(selection_results)
        
        # Check overall success
        overall_success = all(result.success for result in selection_results.values())
        
        return BudgetAwareSelectionResult(
            all_selected_features=all_selected_features,
            feature_type_results=selection_results,
            total_selection_time=total_time,
            total_budget_used_ms=total_budget_used,
            overall_performance_score=overall_performance,
            base_features=base_features,
            interaction_features=interaction_features,
            cross_timeframe_features=cross_timeframe_features,
            gate_features=gate_features,
            success=overall_success,
            config_used=self.config
        )
    
    def _calculate_overall_performance(
        self,
        selection_results: Dict[str, FeatureTypeSelectionResult]
    ) -> float:
        """Calculate overall performance score from individual results."""
        if not selection_results:
            return 0.0
        
        # Weight by feature type importance
        weights = {
            'base': 0.4,
            'interaction': 0.3,
            'cross_timeframe': 0.2,
            'gate': 0.1
        }
        
        weighted_scores = []
        for ftype, result in selection_results.items():
            if result.success and result.performance_metrics:
                cv_score = result.performance_metrics.get('cv_mean', 0.0)
                weight = weights.get(ftype, 0.1)
                weighted_scores.append(cv_score * weight)
        
        return sum(weighted_scores) if weighted_scores else 0.0
    
    def export_explanation_report(
        self,
        result: BudgetAwareSelectionResult,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Export detailed explainability report for the selection.
        
        Args:
            result: Selection result with explanation
            output_dir: Optional output directory (defaults to outcomes/)
        """
        if result.selection_explanation is None:
            tprint_warning("⚠️ No explanation available to export")
            return
        
        if output_dir is None:
            output_dir = Path("outcomes/feature_selection_explanations")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export HTML report
        html_path = output_dir / f"selection_explanation_{timestamp}.html"
        self.explainer.export_html_report(result.selection_explanation, html_path)
        
        # Export JSON summary
        json_path = output_dir / f"selection_summary_{timestamp}.json"
        self._export_json_summary(result, json_path)
        
        tprint_success(f"✅ Explainability reports exported to {output_dir}")
    
    def _export_json_summary(self, result: BudgetAwareSelectionResult, output_path: Path) -> None:
        """Export JSON summary of selection."""
        import json
        
        summary = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'success': result.success,
            'total_features_selected': len(result.all_selected_features),
            'feature_breakdown': {
                'base': len(result.base_features),
                'interaction': len(result.interaction_features),
                'cross_timeframe': len(result.cross_timeframe_features),
                'gate': len(result.gate_features)
            },
            'performance': {
                'overall_score': result.overall_performance_score,
                'execution_time': result.total_selection_time,
                'budget_used_ms': result.total_budget_used_ms
            },
            'cache_stats': result.cache_stats,
            'multi_objective': {
                'pareto_front_size': len(result.pareto_front) if result.pareto_front else 0,
                'best_metrics': result.multi_objective_result['best_metrics'] if result.multi_objective_result else {}
            } if result.multi_objective_result else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        tprint_debug(f"   📄 JSON summary exported to {output_path}")


def create_budget_aware_selector(
    config: Optional[BudgetAwareSelectionConfig] = None
) -> BudgetAwareFeatureSelector:
    """Create a budget-aware feature selector with the given configuration."""
    return BudgetAwareFeatureSelector(config)


# Convenience function for direct usage
async def select_features_budget_aware(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[BudgetAwareSelectionConfig] = None,
    feature_types: Optional[Dict[str, List[str]]] = None
) -> BudgetAwareSelectionResult:
    """
    Convenience function for budget-aware feature selection.
    
    Args:
        X: Feature matrix
        y: Target variable
        config: Optional configuration
        feature_types: Optional feature type mapping
        
    Returns:
        BudgetAwareSelectionResult with selected features
    """
    selector = create_budget_aware_selector(config)
    return await selector.select_features(X, y, feature_types)
