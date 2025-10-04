import asyncio
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve()
for _ in range(6):
    ROOT = ROOT.parent

module_path = ROOT / "src/training/steps/market_analysis/clusters/iterative_optimization.py"

package_defs = {
    "src": ROOT / "src",
    "src.training": ROOT / "src/training",
    "src.training.steps": ROOT / "src/training/steps",
    "src.training.steps.market_analysis": ROOT / "src/training/steps/market_analysis",
    "src.training.steps.market_analysis.clusters": ROOT / "src/training/steps/market_analysis/clusters",
}

for name, path in package_defs.items():
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

spec = importlib.util.spec_from_file_location(
    "src.training.steps.market_analysis.clusters.iterative_optimization",
    module_path,
)
iter_opt = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = iter_opt
spec.loader.exec_module(iter_opt)

ClusteringStats = iter_opt.ClusteringStats
IterativeOptimization = iter_opt.IterativeOptimization
NAgosticConstraints = iter_opt.NAgosticConstraints
SplitSkipGate = iter_opt.SplitSkipGate
StrictSplitPolicy = iter_opt.StrictSplitPolicy


def test_step3_break_large_clusters_triggers_split_and_resizes(monkeypatch):
    rng = np.random.default_rng(0)
    large_cluster = rng.normal(loc=0.0, scale=0.5, size=(40, 2))
    small_cluster = rng.normal(loc=5.0, scale=0.5, size=(10, 2))
    features = np.vstack([large_cluster, small_cluster])
    assignments = np.zeros(len(features), dtype=int)
    assignments[-len(small_cluster):] = 1

    stats = ClusteringStats(features, assignments)

    optimizer = IterativeOptimization(verbose=False)
    optimizer.features = features
    optimizer.assignments = stats.assignments.copy()
    optimizer.min_cluster_size = 1
    optimizer.micro_frac = 0.001
    optimizer.config.K_MIN = 1
    optimizer.SOFT_CAP = len(features)

    constraints = NAgosticConstraints(k_max=10, min_fraction=0.01)
    constraints.update_constraints(len(features))

    split_policy = StrictSplitPolicy()
    split_policy.min_parent_vs_target = 0.0
    split_policy.min_parent_vs_min = 0.0
    split_policy.min_child_vs_target = 0.0
    split_policy.min_child_vs_min = 0.0
    split_policy.balance_floor = 0.0
    split_policy.min_rel_gain = -1.0
    split_policy.per_round_split_limit = 10

    split_skip_gate = SplitSkipGate()

    original_K = stats.centroids.shape[0]
    original_transition_shape = stats.transition_counts.shape

    def wrapped_stats_update(self, stats_obj, labels, X=None):
        result = original_stats_update(stats_obj, labels, X)
        self.assignments = stats_obj.assignments.copy()
        return result

    original_stats_update = optimizer._stats_update
    monkeypatch.setattr(
        optimizer,
        "_stats_update",
        types.MethodType(wrapped_stats_update, optimizer),
    )

    delta = asyncio.run(
        optimizer._step3_break_large_clusters(
            features,
            stats,
            constraints,
            split_policy,
            split_skip_gate,
            current_round=0,
        )
    )

    assert isinstance(delta, (float, int))
    new_K = stats.centroids.shape[0]
    assert new_K >= original_K + 1
    assert stats.wcss_per_cluster.shape[0] == new_K
    assert stats.S.shape[0] == new_K
    assert stats.Q_trace.shape[0] == new_K
    assert stats.transition_counts.shape == (new_K, new_K)
    assert stats.transition_row_sums.shape[0] == new_K
    assert np.sum(stats.cluster_sizes) == len(features)
    assert stats.transition_counts.shape[0] >= original_transition_shape[0] + 1
