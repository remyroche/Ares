import numpy as np
import pandas as pd

from extreme_price_movements.calibration import apply_logit_shift
from extreme_price_movements.model_race import ModelRace, Float64Wrapper
from extreme_price_movements.model_scoring import ece_at_mask, topk_mask
from sklearn.linear_model import LogisticRegression


def test_modelrace_bias_contract_and_calibration(monkeypatch):
    rng = np.random.default_rng(7)
    n = 1200
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    logits = -3.4 + 0.7 * X["f0"].values - 0.5 * X["f1"].values
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    # inflate weighted prevalence near balanced
    w = 1.0 + 20.0 * y
    returns = (0.02 * X["f0"].values + rng.normal(scale=0.01, size=n)).astype(float)

    # speed up race with a single simple model
    def _single_candidate(self, race_mode=True):
        return {"logreg": Float64Wrapper(LogisticRegression(max_iter=300, solver="lbfgs"))}

    monkeypatch.setattr(ModelRace, "_get_candidates", _single_candidate)

    race = ModelRace(kind="mr", n_splits=3)
    race.fit(X, y, sample_weight=w, returns=returns)

    assert race.calibration_state_ is not None
    state = race.calibration_state_
    for k in ["schema_version", "method", "target_unweighted_prevalence", "weighted_prevalence", "delta_logit", "eps", "calibration_input"]:
        assert k in state

    p_raw = race.predict_proba_raw(X)
    p_corr = apply_logit_shift(p_raw, state["delta_logit"], eps=state["eps"])
    p_cal = race.predict_proba(X)[:, 1]

    p_unweighted = float(np.mean(y))
    assert abs(float(np.mean(p_corr)) - p_unweighted) <= 0.02
    assert abs(float(np.mean(p_cal)) - float(np.mean(p_corr))) < 0.02

    m10 = topk_mask(p_cal, 0.10, groups=None)
    ece10 = ece_at_mask(y, p_cal, m10, n_bins=10)
    assert ece10 < 0.10

    # Optional regression guard: bypassing correction should be materially worse.
    if hasattr(race, "calibrator_"):
        p_bad = race.calibrator_.predict(p_raw)
        m10_bad = topk_mask(p_bad, 0.10, groups=None)
        ece10_bad = ece_at_mask(y, p_bad, m10_bad, n_bins=10)
        assert ece10_bad >= ece10
