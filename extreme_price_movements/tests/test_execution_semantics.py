import pytest
from extreme_price_movements.execution_semantics import (
    resolve_limit_fill,
    resolve_stop_fill,
    ExitReason,
    map_numba_exit_code_to_canonical,
    map_tbm_label_to_canonical_training_semantic
)

def test_limit_order_fill_long_no_gap():
    # Long Limit Entry or Short Limit Exit
    # Limit at 100. Open=105, High=110, Low=90, Close=95.
    did_fill, fill_price = resolve_limit_fill(open_price=105, high_price=110, low_price=90, limit_price=100, is_long=True)
    assert did_fill is True
    assert fill_price == 100.0  # NOT 90.0

def test_limit_order_fill_long_gap():
    # Limit at 100. Open=95, High=98, Low=90, Close=92.
    did_fill, fill_price = resolve_limit_fill(open_price=95, high_price=98, low_price=90, limit_price=100, is_long=True)
    assert did_fill is True
    assert fill_price == 95.0

def test_limit_order_fill_short_no_gap():
    # Short Limit Entry or Long Limit Exit
    # Limit at 100. Open=95, High=110, Low=90, Close=105.
    did_fill, fill_price = resolve_limit_fill(open_price=95, high_price=110, low_price=90, limit_price=100, is_long=False)
    assert did_fill is True
    assert fill_price == 100.0  # NOT 110.0

def test_limit_order_fill_short_gap():
    # Limit at 100. Open=105, High=110, Low=102, Close=105.
    did_fill, fill_price = resolve_limit_fill(open_price=105, high_price=110, low_price=102, limit_price=100, is_long=False)
    assert did_fill is True
    assert fill_price == 105.0

def test_stop_loss_long_no_gap():
    # Long Position Stop Loss at 95. Open=100, High=105, Low=90, Close=92
    did_fill, fill_price = resolve_stop_fill(open_price=100, high_price=105, low_price=90, stop_price=95, is_long=True)
    assert did_fill is True
    assert fill_price == 95.0

def test_stop_loss_long_gap():
    # Long Position Stop Loss at 95. Open=90, High=92, Low=85, Close=90
    did_fill, fill_price = resolve_stop_fill(open_price=90, high_price=92, low_price=85, stop_price=95, is_long=True)
    assert did_fill is True
    assert fill_price == 90.0

def test_stop_loss_short_no_gap():
    # Short Position Stop Loss at 105. Open=100, High=110, Low=90, Close=108
    did_fill, fill_price = resolve_stop_fill(open_price=100, high_price=110, low_price=90, stop_price=105, is_long=False)
    assert did_fill is True
    assert fill_price == 105.0

def test_stop_loss_short_gap():
    # Short Position Stop Loss at 105. Open=110, High=112, Low=108, Close=110
    did_fill, fill_price = resolve_stop_fill(open_price=110, high_price=112, low_price=108, stop_price=105, is_long=False)
    assert did_fill is True
    assert fill_price == 110.0

def test_mapping_functions():
    assert map_numba_exit_code_to_canonical(0) == "take_profit"
    assert map_numba_exit_code_to_canonical(1) == "stop_loss"
    assert map_tbm_label_to_canonical_training_semantic(0) == "ADVERSE_BARRIER_FIRST"
    assert map_tbm_label_to_canonical_training_semantic(2) == "FAVORABLE_BARRIER_FIRST_OR_TRAIL_ACTIVATION"

if __name__ == "__main__":
    pytest.main([__file__])
