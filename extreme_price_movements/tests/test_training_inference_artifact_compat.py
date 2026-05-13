from extreme_price_movements.inference.parity import validate_calibration_artifacts
from extreme_price_movements.simple_position_sizer import (
    load_calibration_contract,
    load_calibration_curves,
    save_calibration_curves,
)
from extreme_price_movements.training import (
    _BASE_REG_INTERACTION_KEYS,
    _BASE_REG_UNCERTAINTY_KEYS,
    _append_base_oof_contract_uncertainty,
)

import numpy as np
from types import SimpleNamespace


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


def test_base_oof_uncertainty_export_schema_is_contract_complete():
    n = 8
    finite_mask = np.array([True, True, False, True, True, False, True, True])
    race = SimpleNamespace(
        oof_probs=np.linspace(0.1, 0.8, n, dtype=np.float32),
        oof_probs_raw_ebm=np.linspace(0.11, 0.81, n, dtype=np.float32),
        oof_uncertainty_features={
            "ebm_unc_logodds_var": np.arange(n, dtype=np.float32) / 10.0,
            "ebm_unc_pi_width": np.linspace(0.01, 0.08, n, dtype=np.float32),
        },
        best_model=None,
    )
    payload = {
        "oof_prob": race.oof_probs[finite_mask],
        "index": np.arange(n, dtype=np.int32)[finite_mask],
    }
    reg_head = {
        "oof_features": {
            "reg_q10": np.linspace(-1.0, 0.0, n, dtype=np.float32),
            "reg_q50": np.linspace(-0.5, 0.5, n, dtype=np.float32),
            "reg_q90": np.linspace(0.0, 1.0, n, dtype=np.float32),
        }
    }

    _append_base_oof_contract_uncertainty(
        payload,
        race=race,
        detailed_metrics={"oof_tree_vote_entropy": np.linspace(0.2, 0.3, n)},
        reg_head=reg_head,
        n=n,
        finite_mask=finite_mask,
    )

    expected = {
        "oof_prob_ebm_raw",
        "oof_prob_en",
        "oof_prob_uncertainty_weighted",
        "oof_sigma_trees",
        "oof_sigma_robust",
        "oof_ebm_unc_logodds_var",
        "oof_ebm_unc_entropy_mean",
        "oof_ebm_unc_conflict",
        "oof_ebm_unc_support_mean",
        "oof_ebm_unc_uncertainty_weight",
        "oof_ebm_unc_pi_width",
        *_BASE_REG_UNCERTAINTY_KEYS,
        *_BASE_REG_INTERACTION_KEYS,
    }
    missing = sorted(c for c in expected if c not in payload)
    bad_lengths = sorted(c for c in expected if c in payload and len(payload[c]) != int(finite_mask.sum()))

    assert missing == []
    assert bad_lengths == []
    assert np.isfinite(payload["oof_ebm_unc_logodds_var"]).any()
    assert np.isfinite(payload["oof_ebm_unc_pi_width"]).any()
    assert np.allclose(payload["reg_uncertainty"], 1.0)
