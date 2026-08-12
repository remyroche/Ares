import numpy as np
import pandas as pd

from scripts.audit_current_r3_proxy_oof_mapping import _apply, _fit_map


def test_prior_side_map_is_monotone_and_finite() -> None:
    train = pd.DataFrame({"score": np.linspace(-1.0, 1.0, 200), "net_bps": np.linspace(-200.0, 200.0, 200)})
    edges, means, fallback = _fit_map(train, "score", "net_bps")
    mapped = _apply(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), edges, means)
    assert np.isfinite(mapped).all()
    assert np.all(np.diff(mapped) >= -1e-12)
    assert np.isfinite(fallback)
