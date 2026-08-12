import numpy as np
import pandas as pd

from scripts.run_oof_gam_portability_ablation import (
    _fit_gam,
    _select_portable_fields,
    portability_score,
)


def _panel(n: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    signal = np.sin(np.arange(n) / 30.0)
    return pd.DataFrame(
        {
            "candidate_id": np.arange(n),
            "__ts__": ts,
            "execution_net_ev_12h": 0.0015 * signal + rng.normal(0.0, 0.001, n),
            "raw_trust_score": signal,
            "portable_context": signal + rng.normal(0.0, 0.05, n),
        }
    )


def test_portability_score_penalizes_dispersion_and_bad_worst_block() -> None:
    stable = portability_score([0.01, 0.011, 0.009])
    unstable = portability_score([0.03, 0.02, -0.04])
    assert stable > unstable


def test_portable_selection_keeps_trust_anchor_and_returns_causal_fields() -> None:
    frame = _panel()
    selected, audit = _select_portable_fields(
        frame,
        ["raw_trust_score", "portable_context"],
        "regime_gam",
    )
    assert selected[0] == "raw_trust_score"
    assert set(selected).issubset({"raw_trust_score", "portable_context"})
    assert {"feature", "portable_score", "coverage"}.issubset(audit.columns)


def test_simple_gam_is_finite_and_prior_fit_only() -> None:
    frame = _panel()
    train, evaluation = frame.iloc[:900], frame.iloc[900:]
    raw, mapped = _fit_gam(
        train,
        evaluation,
        ["raw_trust_score", "portable_context"],
        n_knots=2,
        degree=1,
        alpha=20.0,
    )
    assert len(raw) == len(evaluation)
    assert len(mapped) == len(evaluation)
    assert np.isfinite(raw).all()
    assert np.isfinite(mapped).all()
