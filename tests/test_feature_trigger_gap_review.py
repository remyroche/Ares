from tools.review_feature_trigger_gaps import build_gap_review_payload


def test_gap_review_payload_marks_new_primitives_present():
    payload = build_gap_review_payload()

    feature_status = {
        row["target_name"]: row["status"] for row in payload["features"]["comparison"]
    }
    trigger_status = {
        row["target_name"]: row["status"] for row in payload["triggers"]["comparison"]
    }

    assert feature_status["atr_14"] == "exact_match"
    assert feature_status["ema_50"] == "exact_match"
    assert feature_status["volume_ma_20"] == "exact_match"
    assert feature_status["acceleration_close_atr"] == "exact_match"

    assert trigger_status["ema_reclaim_touch"] == "exact_match"
    assert trigger_status["simple_close_breakout"] == "exact_match"
    assert trigger_status["expansion_bar"] == "exact_match"
    assert trigger_status["impulse_bar"] == "exact_match"
    assert trigger_status["relaxed_sweep"] == "exact_match"
    assert trigger_status["compression_release"] == "exact_match"
    assert trigger_status["compressed_breakout_up_down"] == "exact_match"
