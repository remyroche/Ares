import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bounded_bcf_demotion", ROOT / "scripts" / "run_strict_r3_p8u_meta_bounded_bcf_demotion_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_bounded_demotion_is_nonpositive_and_outside_range_is_zero():
    rng = np.random.default_rng(1729)
    size = 8_000
    frame = pd.DataFrame({
        "bcf_mc1_expected_bps": rng.uniform(20, 230, size),
        "current_mc1_expected_bps": rng.uniform(20, 180, size),
        "under_f120": rng.uniform(0, 1, size),
        "magnitude": rng.uniform(0, 1, size),
        "over": rng.uniform(0, 1, size),
        "state": rng.uniform(0, 1, size),
        "policy_path_valid": True,
        "policy_net_bps": rng.normal(80, 180, size),
    })
    correction, audit = MODULE._fit_demotion(frame, frame.copy(), limit_bps=100.0, target="severe100", authority=1.0)
    assert np.all(correction <= 1e-7)
    outside = ~frame["bcf_mc1_expected_bps"].between(30, 100).to_numpy()
    assert np.allclose(correction[outside], 0.0)
    assert audit["train_rows"] > 2_000


def test_month_parser_requires_chronological_folds():
    assert len(MODULE._months("2026-01,2026-02,2026-03,2026-04,2026-05")) == 5
    try:
        MODULE._months("2026-02,2026-01,2026-03,2026-04,2026-05")
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-order monthly folds must fail")
