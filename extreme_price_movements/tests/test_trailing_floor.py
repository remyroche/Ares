from extreme_price_movements.risk import TrailingStop


def test_trailing_stop_long_never_below_activation_floor_after_activation():
    ts = TrailingStop(
        entry_px=100.0,
        side="long",
        atr_val=0.01,
        k_sl=1.0,
        k_trail_start=2.0,
        k_trail_dist=5.0,
    )

    # Activate trailing at +2% (high reaches 102)
    stopped, _, _ = ts.update(current_high=102.0, current_low=100.5, current_close=101.5)
    assert not stopped

    # Floor should be entry + activation_dist => 102.0
    assert ts.get_sl_px() >= 102.0


def test_trailing_stop_short_never_above_activation_floor_after_activation():
    ts = TrailingStop(
        entry_px=100.0,
        side="short",
        atr_val=0.01,
        k_sl=1.0,
        k_trail_start=2.0,
        k_trail_dist=5.0,
    )

    # Activate trailing at -2% (low reaches 98)
    stopped, _, _ = ts.update(current_high=99.5, current_low=98.0, current_close=98.5)
    assert not stopped

    # Floor for short is entry - activation_dist => 98.0 (stop cannot be above this after activation)
    assert ts.get_sl_px() <= 98.0
