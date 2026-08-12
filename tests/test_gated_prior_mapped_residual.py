from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_gated_prior_mapped_residual import _choose_gate, _quantile_edges


def test_economic_gate_rejects_residual_when_noop_has_better_tail() -> None:
    base = np.linspace(-100.0, 100.0, 200)
    net = -base  # the no-op ordering is intentionally the better one
    frame = pd.DataFrame({"causal_base_map_bps": base, "net_bps": net})
    pred = np.linspace(-50.0, 50.0, 200)
    lam, gates, audit = _choose_gate(frame, pred, side="long", edges=_quantile_edges(base))
    assert lam == 0.0
    assert not gates.any()
    assert any(row.get("lambda") == 0.0 for row in audit)


def test_economic_gate_can_admit_a_residual_that_improves_the_tail() -> None:
    # The production gate requires at least 200 rows per causal-map region;
    # use enough synthetic rows for every quantile region to be eligible.
    base = np.linspace(-100.0, 100.0, 1000)
    target = -base
    frame = pd.DataFrame({"causal_base_map_bps": base, "net_bps": target})
    pred = -2.0 * base
    lam, gates, audit = _choose_gate(frame, pred, side="long", edges=_quantile_edges(base))
    assert lam > 0.0
    assert gates.any()
    assert any(bool(row.get("beats_noop")) for row in audit)
