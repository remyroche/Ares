#!/usr/bin/env python3
"""Test limit order pricer implementation."""

import numpy as np
from extreme_price_movements.limit_order_pricer import (
    estimate_entry_limit_offset,
    estimate_exit_limit_offset,
    estimate_fill_probability,
    get_limit_price_for_order,
    check_limit_order_fill,
    create_limit_order_config,
    get_fee_for_order_type,
    simulate_trade_with_limit_order,
)


def test_limit_price_for_order():
    """Test proper handling of high/low for long/short."""
    print("Testing get_limit_price_for_order...")
    
    signal_price = 100.0
    
    # Long: buy below signal
    long_limit = get_limit_price_for_order(signal_price, 20.0, is_long=True)
    expected_long = 100.0 * (1 - 0.0020)  # 99.8
    assert abs(long_limit - expected_long) < 0.01, f"Long limit: {long_limit} vs expected {expected_long}"
    
    # Short: sell above signal
    short_limit = get_limit_price_for_order(signal_price, 20.0, is_long=False)
    expected_short = 100.0 * (1 + 0.0020)  # 100.2
    assert abs(short_limit - expected_short) < 0.01, f"Short limit: {short_limit} vs expected {expected_short}"
    
    print(f"  Long limit @ {signal_price} with 20bps offset: {long_limit:.4f}")
    print(f"  Short limit @ {signal_price} with 20bps offset: {short_limit:.4f}")


def test_fill_check():
    """Test proper fill checking with high/low."""
    print("\nTesting check_limit_order_fill...")
    
    # Long fills when low goes below limit
    limit_price = 99.8
    high_price = 100.2
    low_price = 99.7  # Below limit = fill
    open_price = 100.0
    
    did_fill, fill_price = check_limit_order_fill(limit_price, True, high_price, low_price, open_price)
    assert did_fill, "Long should fill when low <= limit"
    print(f"  Long fill check: limit={limit_price}, low={low_price}, high={high_price} -> fill={did_fill}, price={fill_price:.4f}")
    
    # Long doesn't fill when low stays above limit
    low_price = 99.9  # Above limit = no fill
    did_fill, fill_price = check_limit_order_fill(limit_price, True, high_price, low_price, open_price)
    assert not did_fill, "Long should NOT fill when low > limit"
    print(f"  Long no-fill check: limit={limit_price}, low={low_price}, high={high_price} -> fill={did_fill}")
    
    # Short fills when high goes above limit
    limit_price = 100.2
    high_price = 100.3  # Above limit = fill
    low_price = 100.1
    open_price = 100.2
    
    did_fill, fill_price = check_limit_order_fill(limit_price, False, high_price, low_price, open_price)
    assert did_fill, "Short should fill when high >= limit"
    print(f"  Short fill check: limit={limit_price}, high={high_price}, low={low_price} -> fill={did_fill}, price={fill_price:.4f}")


def test_entry_offset_estimation():
    """Test MAE/MFE-based entry offset estimation."""
    print("\nTesting estimate_entry_limit_offset...")
    
    # High MAE, low MFE -> wider offset
    offset1 = estimate_entry_limit_offset(
        mae_hat=0.02,  # 2% predicted MAE
        mfe_hat=0.01,  # 1% predicted MFE
        u_hat=0.1,
        confidence=0.5,
    )
    print(f"  High MAE (2%), Low MFE (1%): offset = {offset1:.1f} bps")
    
    # Low MAE, high MFE -> tighter offset
    offset2 = estimate_entry_limit_offset(
        mae_hat=0.01,  # 1% predicted MAE
        mfe_hat=0.03,  # 3% predicted MFE
        u_hat=0.5,
        confidence=0.8,
    )
    print(f"  Low MAE (1%), High MFE (3%): offset = {offset2:.1f} bps")
    
    # Verify that higher MAE gives wider offset (in general)
    offset3 = estimate_entry_limit_offset(
        mae_hat=0.03,  # 3% predicted MAE
        mfe_hat=0.01,  # 1% predicted MFE
        u_hat=0.1,
        confidence=0.5,
    )
    print(f"  Higher MAE (3%), Low MFE (1%): offset = {offset3:.1f} bps")
    
    assert offset3 > offset1, "Higher MAE should give wider offset"


def test_fill_probability():
    """Test fill probability estimation."""
    print("\nTesting estimate_fill_probability...")
    
    # Small offset, high vol -> low fill prob
    prob1 = estimate_fill_probability(
        offset_bps=5.0,
        mae_hat=0.02,
        vol_regime=0.8,  # High vol
        liquidity=0.5,
    )
    print(f"  5bps offset, high vol: fill prob = {prob1:.2%}")
    
    # Large offset, low vol -> high fill prob
    prob2 = estimate_fill_probability(
        offset_bps=30.0,
        mae_hat=0.01,
        vol_regime=0.2,  # Low vol
        liquidity=0.8,  # High liquidity
    )
    print(f"  30bps offset, low vol, high liq: fill prob = {prob2:.2%}")
    
    assert prob2 > prob1, "Larger offset should have higher fill prob"


def test_exit_offset_estimation():
    """Test exit limit offset estimation."""
    print("\nTesting estimate_exit_limit_offset...")
    
    # No profit locked -> wider offset
    offset1 = estimate_exit_limit_offset(
        mfe_hat=0.02,
        duration_hat=2.0,
        profit_locked=0.0,
        mae_hat=0.01,
    )
    print(f"  No profit locked: exit offset = {offset1:.1f} bps")
    
    # High profit locked -> tighter offset
    offset2 = estimate_exit_limit_offset(
        mfe_hat=0.02,
        duration_hat=2.0,
        profit_locked=0.03,  # 3% profit
        mae_hat=0.01,
    )
    print(f"  3% profit locked: exit offset = {offset2:.1f} bps")
    
    assert offset2 < offset1, "More profit should give tighter exit offset"


def test_fee_helper():
    """Test fee helper function."""
    print("\nTesting get_fee_for_order_type...")
    
    cfg = {
        "fee_bps_market": 25.0,
        "fee_bps_limit_entry": 10.0,
        "fee_bps_limit_exit": 10.0,
        "fee_bps_market_exit": 25.0,
    }
    
    market_entry = get_fee_for_order_type("market", "entry", cfg)
    limit_entry = get_fee_for_order_type("limit", "entry", cfg)
    market_exit = get_fee_for_order_type("market", "exit", cfg)
    limit_exit = get_fee_for_order_type("limit", "exit", cfg)
    
    print(f"  Market entry: {market_entry*100:.2f}%")
    print(f"  Limit entry: {limit_entry*100:.2f}%")
    print(f"  Market exit: {market_exit*100:.2f}%")
    print(f"  Limit exit: {limit_exit*100:.2f}%")
    
    assert market_entry == 0.0025, "Market entry should be 0.25%"
    assert limit_entry == 0.0010, "Limit entry should be 0.10%"


def test_create_limit_order_config():
    """Test full limit order config creation."""
    print("\nTesting create_limit_order_config...")
    
    cfg = {
        "fee_bps_market": 25.0,
        "fee_bps_limit_entry": 10.0,
        "fee_bps_limit_exit": 10.0,
        "fee_bps_market_exit": 25.0,
    }
    
    config = create_limit_order_config(
        mae_hat=0.015,
        mfe_hat=0.025,
        u_hat=0.3,
        confidence=0.6,
        vol_regime=0.4,
        liquidity=0.7,
        profit_locked=0.0,
        duration_hat=1.5,
        cfg=cfg,
    )
    
    print(f"  Entry offset: {config['entry_offset_bps']:.1f} bps")
    print(f"  Exit offset: {config['exit_offset_bps']:.1f} bps")
    print(f"  Entry fill prob: {config['fill_prob_entry']:.2%}")
    print(f"  Exit fill prob: {config['fill_prob_exit']:.2%}")
    print(f"  Fee entry: {config['fee_entry']*100:.2f}%")
    print(f"  Fee exit: {config['fee_exit']*100:.2f}%")


def test_simulate_trade():
    """Test trade simulation with limit orders."""
    print("\nTesting simulate_trade_with_limit_order...")
    
    # Generate sample price data
    np.random.seed(42)
    n = 20
    signal_price = 100.0
    
    # Simulate price path
    returns = np.random.randn(n) * 0.005  # 0.5% std
    close_prices = signal_price * np.exp(np.cumsum(returns))
    high_prices = close_prices * (1 + np.abs(np.random.randn(n) * 0.002))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n) * 0.002))
    
    # Long trade with limit entry
    result = simulate_trade_with_limit_order(
        signal_price=signal_price,
        offset_bps_entry=15.0,
        offset_bps_exit=10.0,
        is_long=True,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        tp_mult=2.0,
        sl_mult=1.0,
        atr=0.02,
        fee_entry=0.001,
        fee_exit=0.001,
        max_bars=10,
    )
    
    print(f"  Trade filled: {result['filled']}")
    if result['filled']:
        print(f"  Entry price: {result['entry_fill_price']:.4f}")
        print(f"  Exit price: {result['exit_fill_price']:.4f}")
        print(f"  Exit reason: {result['exit_reason']}")
        print(f"  Return: {result['return']*100:.2f}%")
        print(f"  Fees: {result['fee_total']*100:.2f}%")


def main():
    print("=" * 60)
    print("LIMIT ORDER PRICER TESTS")
    print("=" * 60)
    
    test_limit_price_for_order()
    test_fill_check()
    test_entry_offset_estimation()
    test_fill_probability()
    test_exit_offset_estimation()
    test_fee_helper()
    test_create_limit_order_config()
    test_simulate_trade()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
