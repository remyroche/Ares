import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_iterative_module():
    module_name = "src.training.steps.market_analysis.clusters.iterative_optimization_test"
    module_path = ROOT_DIR / "src/training/steps/market_analysis/clusters/iterative_optimization.py"

    if module_name in sys.modules:
        return sys.modules[module_name]

    package_specs = [
        ("src", ROOT_DIR / "src"),
        ("src.training", ROOT_DIR / "src/training"),
        ("src.training.steps", ROOT_DIR / "src/training/steps"),
        ("src.training.steps.market_analysis", ROOT_DIR / "src/training/steps/market_analysis"),
        ("src.training.steps.market_analysis.clusters", ROOT_DIR / "src/training/steps/market_analysis/clusters"),
    ]

    for name, path in package_specs:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


iterative_module = load_iterative_module()
ClusteringStats = iterative_module.ClusteringStats
IterativeOptimization = iterative_module.IterativeOptimization


def test_step2_rollback_restores_statistics():
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.2, 0.8],
        ],
        dtype=np.float64,
    )
    assignments = np.array([0, 1, 1], dtype=np.int32)

    stats = ClusteringStats(features, assignments)
    optimizer = IterativeOptimization(verbose=False)

    initial_assignments = stats.assignments.copy()
    initial_centroids = stats.centroids.copy()
    initial_wcss = stats.wcss_per_cluster.copy()
    initial_total_wcss = stats.total_wcss
    initial_total_bcss = stats.total_bcss
    initial_variance_ratio = stats.get_cv_ratio()

    move = {
        "point_idx": 0,
        "from_cluster": int(stats.assignments[0]),
        "to_cluster": int(stats.assignments[1]),
    }

    initial_k = len(np.unique(stats.assignments))
    applied = optimizer._apply_step2_move_with_guard(stats, move, initial_k)

    assert applied is False
    np.testing.assert_array_equal(stats.assignments, initial_assignments)
    np.testing.assert_allclose(stats.centroids, initial_centroids)
    np.testing.assert_allclose(stats.wcss_per_cluster, initial_wcss)
    assert stats.total_wcss == pytest.approx(initial_total_wcss)
    assert stats.total_bcss == pytest.approx(initial_total_bcss)
    assert stats.get_cv_ratio() == pytest.approx(initial_variance_ratio)

    fresh_stats = ClusteringStats(features, initial_assignments.copy())
    np.testing.assert_allclose(stats.centroids, fresh_stats.centroids)
    np.testing.assert_allclose(stats.wcss_per_cluster, fresh_stats.wcss_per_cluster)
    assert stats.total_wcss == pytest.approx(fresh_stats.total_wcss)
    assert stats.total_bcss == pytest.approx(fresh_stats.total_bcss)
    assert stats.get_cv_ratio() == pytest.approx(fresh_stats.get_cv_ratio())
