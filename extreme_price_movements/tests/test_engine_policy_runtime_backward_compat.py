from extreme_price_movements.entry_policy import compute_entry_policy_decision


def test_entry_policy_backward_compat_without_policy_params():
    out = compute_entry_policy_decision(
        entry_px=100.0,
        atr_frac=0.02,
        score=0.4,
        bucket_cfg={"tp_mult": 1.0, "sl_mult": 0.5},
    )
    assert out["place_order"] in (True, False)
    assert out["delta_atr_star"] >= 0.0
    assert out["limit_offset_bps_dynamic"] >= 0.0
