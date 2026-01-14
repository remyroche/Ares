import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(module_name: str, relative_path: str):
    import sys
    import types

    if "h5py" not in sys.modules:
        sys.modules["h5py"] = types.ModuleType("h5py")

    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onc_clusters_transpose_guard():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(10, 3)), columns=["f1", "f2", "f3"])
    module = _load_module(
        "de_prado_feature_engine",
        "src/training/steps/labeling/de_prado_feature_engine.py",
    )
    engine = module.DePradoFeatureEngine(max_clusters=3)

    clusters = engine._get_onc_clusters(data.T)

    assert len(clusters) == 3

