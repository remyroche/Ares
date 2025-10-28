"""
Unified Clustering Optimization Goals Configuration

This module defines common optimization goals and metrics used across all clustering
components including:
- iterative_optimization.py
- iterative_optimization_tuner.py
- hdbscan_clustering optimization
- regime_clustering_step.py

By centralizing these goals, we ensure consistency and make tuning easier.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from enum import Enum


class OptimizationGoal(Enum):
    """Core clustering optimization goals."""
    CV_SCORE = "cv_score"  # Between/Within Variance Ratio (Calinski-Harabasz)
    SILHOUETTE = "silhouette_score"  # Cluster cohesion and separation
    DBI = "dbi_score"  # Davies-Bouldin Index
    BALANCE = "balance_score"  # Cluster size balance
    TEMPORAL_SMOOTHNESS = "temporal_smoothness"  # Temporal stability


class OptimizationObjective(Enum):
    """Optimization direction for each goal."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    MAINTAIN = "maintain"  # Soft constraint


@dataclass
class GoalConfig:
    """Configuration for a single optimization goal."""
    name: str
    objective: OptimizationObjective
    weight: float  # Importance weight in composite score
    target_range: Tuple[float, float]  # (min, max) acceptable values
    constraint_threshold: Optional[float] = None  # Hard constraint if specified
    description: str = ""


@dataclass
class ClusteringOptimizationGoals:
    """
    Unified clustering optimization goals and targets.
    
    These goals are shared across all clustering optimization components:
    - iterative_optimization.py: Uses these to tune optimization loop
    - iterative_optimization_tuner.py: Optimizes hyperparameters based on these
    - hdbscan_clustering: Uses these for quality assessment
    - regime_clustering_step.py: Validates clustering results against these
    """
    
    # ===== PRIMARY OPTIMIZATION GOALS =====
    
    # Goal 1: CV Score (Between/Within Variance Ratio)
    # Higher is better - measures cluster separation
    cv_score: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="CV Score",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.30,  # 30% of composite score
        target_range=(1.0, 10.0),  # Aim for >1.0, excellent >2.0
        constraint_threshold=None,  # No hard constraint
        description="Between/Within Variance Ratio - measures cluster separation quality"
    ))
    
    # Goal 2: Silhouette Score
    # Higher is better - measures cluster cohesion and separation
    # Range: -1 (worst) to 1 (best)
    silhouette_score: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Silhouette Score",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.25,  # 25% of composite score
        target_range=(0.2, 1.0),  # Aim for >0.2, excellent >0.5
        constraint_threshold=None,  # No hard constraint
        description="Cluster cohesion and separation score (range: -1 to 1)"
    ))
    
    # Goal 3: DBI Score (Davies-Bouldin Index)
    # Lower is better - measures cluster separation
    dbi_score: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="DBI Score",
        objective=OptimizationObjective.MINIMIZE,
        weight=0.20,  # 20% of composite score
        target_range=(0.5, 2.0),  # Aim for <2.0, excellent <1.0
        constraint_threshold=None,  # No hard constraint
        description="Davies-Bouldin Index - lower indicates better cluster separation"
    ))
    
    # ===== SECONDARY GOALS (SOFT CONSTRAINTS) =====
    
    # Goal 4: Balance Score
    # Maintain above threshold - ensures clusters are reasonably balanced
    balance_score: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Balance Score",
        objective=OptimizationObjective.MAINTAIN,
        weight=0.15,  # 15% of composite score
        target_range=(0.5, 1.0),  # Aim for >0.5
        constraint_threshold=0.5,  # Soft constraint: should be >0.5
        description="Cluster size balance - prevents overly imbalanced clusters"
    ))
    
    # Goal 5: Temporal Smoothness
    # Maintain above threshold - ensures temporal stability
    temporal_smoothness: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Temporal Smoothness",
        objective=OptimizationObjective.MAINTAIN,
        weight=0.10,  # 10% of composite score
        target_range=(0.85, 1.0),  # Aim for >0.85
        constraint_threshold=0.85,  # Soft constraint: should be >0.85
        description="Temporal stability - prevents excessive regime switching"
    ))
    
    # ===== STRUCTURAL CONSTRAINTS =====
    # These are hard constraints on cluster structure
    
    # Constraint 1: Number of Clusters
    # Target range for optimal number of clusters
    cluster_count_range: Tuple[int, int] = (6, 8)  # Preferred: 6-8 clusters
    cluster_count_min: int = 5  # Absolute minimum
    cluster_count_max: int = 10  # Absolute maximum
    
    # Constraint 2: Cluster Size Bounds
    # Minimum and maximum cluster size as percentage of total samples
    min_cluster_size_pct: float = 0.02  # 2% minimum (prevents tiny clusters)
    max_cluster_size_pct: float = 0.20  # 20% maximum (prevents dominant clusters)
    
    def get_all_goals(self) -> Dict[str, GoalConfig]:
        """Get all goal configurations as a dictionary."""
        return {
            OptimizationGoal.CV_SCORE.value: self.cv_score,
            OptimizationGoal.SILHOUETTE.value: self.silhouette_score,
            OptimizationGoal.DBI.value: self.dbi_score,
            OptimizationGoal.BALANCE.value: self.balance_score,
            OptimizationGoal.TEMPORAL_SMOOTHNESS.value: self.temporal_smoothness,
        }
    
    def get_primary_goals(self) -> Dict[str, GoalConfig]:
        """Get primary optimization goals (CV, Silhouette, DBI)."""
        return {
            OptimizationGoal.CV_SCORE.value: self.cv_score,
            OptimizationGoal.SILHOUETTE.value: self.silhouette_score,
            OptimizationGoal.DBI.value: self.dbi_score,
        }
    
    def get_constraint_goals(self) -> Dict[str, GoalConfig]:
        """Get soft constraint goals (Balance, Temporal)."""
        return {
            OptimizationGoal.BALANCE.value: self.balance_score,
            OptimizationGoal.TEMPORAL_SMOOTHNESS.value: self.temporal_smoothness,
        }
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights for composite score calculation."""
        return {
            'cv': self.cv_score.weight,
            'silhouette': self.silhouette_score.weight,
            'dbi': self.dbi_score.weight,
            'balance': self.balance_score.weight,
            'temporal': self.temporal_smoothness.weight,
        }
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0 (or close to it)."""
        total_weight = sum(self.get_weights_dict().values())
        return abs(total_weight - 1.0) < 1e-6
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        weights = self.get_weights_dict()
        total = sum(weights.values())
        if total > 0:
            self.cv_score.weight = weights['cv'] / total
            self.silhouette_score.weight = weights['silhouette'] / total
            self.dbi_score.weight = weights['dbi'] / total
            self.balance_score.weight = weights['balance'] / total
            self.temporal_smoothness.weight = weights['temporal'] / total


@dataclass
class OptimizationTargets:
    """
    Specific target values for optimization.
    
    These targets guide hyperparameter tuning and serve as quality thresholds
    for clustering validation.
    """
    
    # Primary targets (what we aim to achieve)
    min_cv_score: float = 1.0  # Minimum acceptable CV score
    min_silhouette_score: float = 0.2  # Minimum acceptable Silhouette
    max_dbi_score: float = 2.0  # Maximum acceptable DBI
    
    # Constraint targets (soft constraints)
    min_balance_score: float = 0.5  # Minimum cluster balance
    min_temporal_smoothness: float = 0.85  # Minimum temporal stability
    
    # Aspirational targets (excellent performance)
    target_cv_score: float = 1.5  # Target CV score
    target_silhouette_score: float = 0.3  # Target Silhouette
    target_dbi_score: float = 1.5  # Target DBI
    target_balance_score: float = 0.7  # Target balance
    target_temporal_smoothness: float = 0.95  # Target temporal smoothness
    
    # ===== STRUCTURAL CONSTRAINTS =====
    
    # Cluster count constraints
    min_clusters: int = 5  # Absolute minimum
    max_clusters: int = 10  # Absolute maximum
    target_clusters: Tuple[int, int] = (6, 8)  # Preferred range
    
    # Cluster size constraints (as percentage of total samples)
    min_cluster_size_pct: float = 0.02  # 2% minimum - prevents tiny clusters
    max_cluster_size_pct: float = 0.20  # 20% maximum - prevents dominant clusters
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'min_cv_score': self.min_cv_score,
            'min_silhouette_score': self.min_silhouette_score,
            'max_dbi_score': self.max_dbi_score,
            'min_balance_score': self.min_balance_score,
            'min_temporal_smoothness': self.min_temporal_smoothness,
            'target_cv_score': self.target_cv_score,
            'target_silhouette_score': self.target_silhouette_score,
            'target_dbi_score': self.target_dbi_score,
            'target_balance_score': self.target_balance_score,
            'target_temporal_smoothness': self.target_temporal_smoothness,
            'min_clusters': self.min_clusters,
            'max_clusters': self.max_clusters,
        }


# ===== GLOBAL INSTANCES =====

# Default goals configuration used across all clustering components
DEFAULT_CLUSTERING_GOALS = ClusteringOptimizationGoals()

# Default optimization targets
DEFAULT_OPTIMIZATION_TARGETS = OptimizationTargets()


# ===== UTILITY FUNCTIONS =====

def calculate_composite_score(
    cv_score: float,
    silhouette_score: float,
    dbi_score: float,
    balance_score: float,
    temporal_smoothness: float,
    goals: Optional[ClusteringOptimizationGoals] = None
) -> float:
    """
    Calculate weighted composite score from individual metrics.
    
    Args:
        cv_score: CV ratio (higher is better)
        silhouette_score: Silhouette score (higher is better)
        dbi_score: DBI score (lower is better)
        balance_score: Balance score (higher is better)
        temporal_smoothness: Temporal smoothness (higher is better)
        goals: Optional custom goals configuration
    
    Returns:
        Composite score (higher is better)
    """
    if goals is None:
        goals = DEFAULT_CLUSTERING_GOALS
    
    weights = goals.get_weights_dict()
    
    # Normalize DBI (invert since lower is better)
    dbi_normalized = 1.0 / (1.0 + dbi_score) if dbi_score > 0 else 0.0
    
    # Calculate weighted sum
    composite = (
        weights['cv'] * cv_score +
        weights['silhouette'] * max(0, silhouette_score) +  # Clip negative
        weights['dbi'] * dbi_normalized +
        weights['balance'] * balance_score +
        weights['temporal'] * temporal_smoothness
    )
    
    return composite


def validate_cluster_sizes(
    cluster_sizes: List[int],
    n_total_samples: int,
    targets: Optional[OptimizationTargets] = None
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate cluster sizes meet constraints.
    
    Args:
        cluster_sizes: List of cluster sizes
        n_total_samples: Total number of samples
        targets: Optional custom targets
    
    Returns:
        Tuple of (all_valid, details)
    """
    if targets is None:
        targets = DEFAULT_OPTIMIZATION_TARGETS
    
    min_size = int(n_total_samples * targets.min_cluster_size_pct)
    max_size = int(n_total_samples * targets.max_cluster_size_pct)
    
    violations = []
    for i, size in enumerate(cluster_sizes):
        size_pct = size / n_total_samples
        if size < min_size:
            violations.append({
                'cluster': i,
                'size': size,
                'size_pct': size_pct,
                'violation': 'too_small',
                'threshold': targets.min_cluster_size_pct
            })
        elif size > max_size:
            violations.append({
                'cluster': i,
                'size': size,
                'size_pct': size_pct,
                'violation': 'too_large',
                'threshold': targets.max_cluster_size_pct
            })
    
    details = {
        'all_valid': len(violations) == 0,
        'min_size': min_size,
        'max_size': max_size,
        'min_size_pct': targets.min_cluster_size_pct,
        'max_size_pct': targets.max_cluster_size_pct,
        'violations': violations,
        'n_violations': len(violations)
    }
    
    return len(violations) == 0, details


def meets_optimization_constraints(
    cv_score: float,
    silhouette_score: float,
    dbi_score: float,
    balance_score: float,
    temporal_smoothness: float,
    n_clusters: int,
    cluster_sizes: Optional[List[int]] = None,
    n_total_samples: Optional[int] = None,
    targets: Optional[OptimizationTargets] = None
) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if metrics meet minimum constraints.
    
    Args:
        cv_score: CV ratio
        silhouette_score: Silhouette score
        dbi_score: DBI score
        balance_score: Balance score
        temporal_smoothness: Temporal smoothness
        n_clusters: Number of clusters
        cluster_sizes: Optional list of cluster sizes for size validation
        n_total_samples: Optional total samples for size validation
        targets: Optional custom targets
    
    Returns:
        Tuple of (all_met, individual_checks)
    """
    if targets is None:
        targets = DEFAULT_OPTIMIZATION_TARGETS
    
    checks = {
        'cv_score': cv_score >= targets.min_cv_score,
        'silhouette_score': silhouette_score >= targets.min_silhouette_score,
        'dbi_score': dbi_score <= targets.max_dbi_score,
        'balance_score': balance_score >= targets.min_balance_score,
        'temporal_smoothness': temporal_smoothness >= targets.min_temporal_smoothness,
        'cluster_count': targets.min_clusters <= n_clusters <= targets.max_clusters,
        'cluster_count_preferred': targets.target_clusters[0] <= n_clusters <= targets.target_clusters[1],
    }
    
    # Validate cluster sizes if provided
    if cluster_sizes is not None and n_total_samples is not None:
        sizes_valid, size_details = validate_cluster_sizes(cluster_sizes, n_total_samples, targets)
        checks['cluster_sizes_valid'] = sizes_valid
        checks['cluster_sizes_details'] = size_details
    
    all_met = all(v if isinstance(v, bool) else True for v in checks.values())
    
    return all_met, checks


def format_metrics_report(
    cv_score: float,
    silhouette_score: float,
    dbi_score: float,
    balance_score: float,
    temporal_smoothness: float,
    n_clusters: int,
    targets: Optional[OptimizationTargets] = None
) -> str:
    """
    Format metrics into a human-readable report.
    
    Args:
        cv_score: CV ratio
        silhouette_score: Silhouette score
        dbi_score: DBI score
        balance_score: Balance score
        temporal_smoothness: Temporal smoothness
        n_clusters: Number of clusters
        targets: Optional custom targets
    
    Returns:
        Formatted report string
    """
    if targets is None:
        targets = DEFAULT_OPTIMIZATION_TARGETS
    
    all_met, checks = meets_optimization_constraints(
        cv_score, silhouette_score, dbi_score, balance_score,
        temporal_smoothness, n_clusters, targets
    )
    
    composite = calculate_composite_score(
        cv_score, silhouette_score, dbi_score, balance_score, temporal_smoothness
    )
    
    report = []
    report.append("=" * 60)
    report.append("CLUSTERING OPTIMIZATION METRICS REPORT")
    report.append("=" * 60)
    report.append(f"\nComposite Score: {composite:.4f}")
    report.append(f"Number of Clusters: {n_clusters}\n")
    report.append("Primary Metrics:")
    report.append(f"  CV Score:          {cv_score:.4f} (target: ≥{targets.min_cv_score:.2f}) {'✅' if checks['cv_score'] else '❌'}")
    report.append(f"  Silhouette:        {silhouette_score:.4f} (target: ≥{targets.min_silhouette_score:.2f}) {'✅' if checks['silhouette_score'] else '❌'}")
    report.append(f"  DBI Score:         {dbi_score:.4f} (target: ≤{targets.max_dbi_score:.2f}) {'✅' if checks['dbi_score'] else '❌'}")
    report.append("\nConstraint Metrics:")
    report.append(f"  Balance:           {balance_score:.4f} (target: ≥{targets.min_balance_score:.2f}) {'✅' if checks['balance_score'] else '❌'}")
    report.append(f"  Temporal:          {temporal_smoothness:.4f} (target: ≥{targets.min_temporal_smoothness:.2f}) {'✅' if checks['temporal_smoothness'] else '❌'}")
    report.append(f"\nOverall Status: {'✅ ALL CONSTRAINTS MET' if all_met else '❌ SOME CONSTRAINTS NOT MET'}")
    report.append("=" * 60)
    
    return "\n".join(report)


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example: Using default goals
    goals = DEFAULT_CLUSTERING_GOALS
    targets = DEFAULT_OPTIMIZATION_TARGETS
    
    print("Default Clustering Optimization Goals:")
    print("=" * 60)
    for goal_name, goal_config in goals.get_all_goals().items():
        print(f"\n{goal_config.name}:")
        print(f"  Objective: {goal_config.objective.value}")
        print(f"  Weight: {goal_config.weight:.2f}")
        print(f"  Target Range: {goal_config.target_range}")
        print(f"  Description: {goal_config.description}")
    
    print("\n\nDefault Optimization Targets:")
    print("=" * 60)
    for key, value in targets.to_dict().items():
        print(f"  {key}: {value}")
    
    # Example: Calculate composite score
    print("\n\nExample Metrics Evaluation:")
    print("=" * 60)
    cv = 1.45
    sil = 0.25
    dbi = 1.8
    bal = 0.68
    temp = 0.92
    n_clust = 7
    
    composite = calculate_composite_score(cv, sil, dbi, bal, temp)
    all_met, checks = meets_optimization_constraints(cv, sil, dbi, bal, temp, n_clust)
    
    report = format_metrics_report(cv, sil, dbi, bal, temp, n_clust)
    print(report)
