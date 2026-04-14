from extreme_price_movements.inference.parity import validate_calibration_artifacts
from extreme_price_movements.simple_position_sizer import (
    load_calibration_contract,
    load_calibration_curves,
    save_calibration_curves,
)


def test_training_and_inference_share_calibration_artifact_schema(tmp_path):
    run_id = "20260101_010101"
    payload = {
        "long_mr": {
            "strategy_id": "long_mr",
            "n_samples": 32,
            "p75_threshold": 0.61,
            "calibration_curve": [(0.0, 0.0), (1.0, 1.0)],
        }
    }

    save_calibration_curves(payload, str(tmp_path), run_id)
    loaded = load_calibration_curves(str(tmp_path), run_id)

    assert "long_mr" in loaded
    assert "p75_threshold" in loaded["long_mr"]
    assert "calibration_curve" in loaded["long_mr"]
    assert isinstance(loaded["long_mr"]["calibration_curve"], list)
    contract = load_calibration_contract(str(tmp_path), run_id)
    assert contract.get("rank_semantics") == "calibrated_p75_threshold"
    assert validate_calibration_artifacts(str(tmp_path), run_id, loaded, strict=True)
