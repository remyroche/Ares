import json

import pandas as pd

from extreme_price_movements.inference.parity import (
    apply_strategy_acceptance_filter,
    calibration_size_multiplier,
    calibrated_score_and_threshold,
    load_strategy_acceptance_filter,
    passes_rank_filter,
    validate_calibration_artifacts,
)


def test_load_strategy_acceptance_filter_reads_policy_artifact(tmp_path):
    run_id = "20260101_000000"
    p = tmp_path / "artifacts" / run_id
    p.mkdir(parents=True)
    payload = {
        "strategies": [
            {"strategy_id": "long_mr"},
            {"strategy_id": "short_tf"},
        ]
    }
    (p / "strategy_final_acceptation.json").write_text(json.dumps(payload))

    accepted = load_strategy_acceptance_filter(str(tmp_path), run_id)
    assert accepted == {"long_mr", "short_tf"}


def test_apply_strategy_acceptance_filter_blocks_non_accepted():
    df = pd.DataFrame(
        {
            "strategy": ["long_mr", "long_tf", "short_mr"],
            "x": [1, 2, 3],
        }
    )
    out = apply_strategy_acceptance_filter(df, {"long_mr", "short_mr"}, "strategy")
    assert out["strategy"].tolist() == ["long_mr", "short_mr"]


def test_rank_filter_uses_p75_threshold_when_available():
    calibration_data = {
        "long_mr": {
            "p75_threshold": 0.60,
            "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
        }
    }
    calibrated, threshold = calibrated_score_and_threshold(
        raw_score=0.62,
        strategy_id="long_mr",
        calibration_data=calibration_data,
        default_threshold=0.5,
    )
    assert calibrated >= 0.62 - 1e-12
    assert threshold == 0.60
    assert passes_rank_filter(0.62, "long_mr", calibration_data)
    assert not passes_rank_filter(0.58, "long_mr", calibration_data)
    mult = calibration_size_multiplier(0.62, "long_mr", calibration_data)
    assert mult >= 1.0


def test_validate_calibration_artifacts_requires_contract(tmp_path):
    run_id = "20260101_000000"
    (tmp_path / "artifacts" / run_id / "ridge_sizer").mkdir(parents=True)
    calibration_data = {"long_mr": {"p75_threshold": 0.6, "calibration_curve": []}}
    # strict=False tolerates missing contract
    assert not validate_calibration_artifacts(
        str(tmp_path), run_id, calibration_data, strict=False
    )
