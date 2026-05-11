import pandas as pd

from extreme_price_movements.inference.liquidity_precheck import (
    compute_price_gap_rank_penalty,
    evaluate_orderbook_liquidity,
    fetch_ticker_snapshot,
    marketable_limit_price,
)
from extreme_price_movements.inference.portfolio_policy import PortfolioPolicyConfig


class _Exchange:
    def __init__(self, ticker=None, book=None):
        self.ticker = ticker or {"bid": 99.9, "ask": 100.1, "last": 100.0}
        self.book = book or {"asks": [[100.1, 10.0]], "bids": [[99.9, 10.0]]}

    def fetch_ticker(self, symbol):
        return dict(self.ticker)

    def fetch_order_book(self, symbol):
        return dict(self.book)


def test_ticker_snapshot_computes_spread_and_rejects_wide_spread():
    policy = PortfolioPolicyConfig()
    snap = fetch_ticker_snapshot(
        exchange=_Exchange({"bid": 99.0, "ask": 101.0, "last": 100.0}),
        symbol="BTC/USDC",
        side="long",
        policy=policy,
        mode="live",
        now=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    assert snap.hard_reject
    assert snap.reject_reason == "spread_above_hard_max"


def test_long_orderbook_walks_asks_within_50bps():
    policy = PortfolioPolicyConfig()
    ex = _Exchange(
        ticker={"bid": 99.9, "ask": 100.0, "last": 100.0},
        book={"asks": [[100.0, 10.0], [100.4, 10.0], [100.6, 100.0]], "bids": []},
    )
    ticker = fetch_ticker_snapshot(
        exchange=ex,
        symbol="BTC/USDC",
        side="long",
        policy=policy,
        mode="live",
        now=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    snap = evaluate_orderbook_liquidity(
        exchange=ex,
        symbol="BTC/USDC",
        side="long",
        intended_quote_size=1500.0,
        ticker_snapshot=ticker,
        policy=policy,
        mode="live",
    )
    assert not snap.hard_reject
    assert snap.orderbook_capacity_quote_within_slippage >= 1500.0
    assert snap.expected_fill_slippage_bps <= 50.0


def test_short_orderbook_walks_bids_and_shallow_book_rejects():
    policy = PortfolioPolicyConfig(min_liquidity_capacity_weight=0.5)
    ex = _Exchange(
        ticker={"bid": 100.0, "ask": 100.1, "last": 100.0},
        book={"bids": [[100.0, 1.0]], "asks": []},
    )
    ticker = fetch_ticker_snapshot(
        exchange=ex,
        symbol="BTC/USDC",
        side="short",
        policy=policy,
        mode="live",
        now=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    snap = evaluate_orderbook_liquidity(
        exchange=ex,
        symbol="BTC/USDC",
        side="short",
        intended_quote_size=1000.0,
        ticker_snapshot=ticker,
        policy=policy,
        mode="live",
    )
    assert snap.hard_reject
    assert snap.reject_reason == "liquidity_capacity_weight_below_min"


def test_orderbook_liquidity_caps_size_by_slippage_plus_half_spread():
    policy = PortfolioPolicyConfig()
    ex = _Exchange(
        ticker={"bid": 99.6008, "ask": 100.0, "last": 99.8},
        book={"asks": [[100.0, 5.0], [100.45, 20.0]], "bids": []},
    )
    ticker = fetch_ticker_snapshot(
        exchange=ex,
        symbol="BTC/USDC",
        side="long",
        policy=policy,
        mode="live",
        now=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    snap = evaluate_orderbook_liquidity(
        exchange=ex,
        symbol="BTC/USDC",
        side="long",
        intended_quote_size=1500.0,
        ticker_snapshot=ticker,
        policy=policy,
        mode="live",
    )

    assert not snap.hard_reject
    assert snap.orderbook_capacity_quote_within_slippage == 500.0
    assert (
        snap.liquidity_capacity_weight
        == snap.orderbook_capacity_quote_within_slippage / 1500.0
    )
    assert (
        snap.details["entry_friction_formula"]
        == "expected_fill_slippage_bps + spread_bps / 2"
    )
    assert (
        snap.details["effective_orderbook_slippage_cap_bps"]
        < policy.max_orderbook_slippage_bps
    )
    assert snap.expected_total_entry_friction_bps <= policy.max_entry_friction_bps


def test_price_gap_penalty_and_marketable_limit():
    policy = PortfolioPolicyConfig()
    penalty, info = compute_price_gap_rank_penalty(
        strategy_id="long_tf",
        side="long",
        signal_price=100.0,
        decision_mid=99.0,
        policy=policy,
    )
    assert penalty > 0
    assert info["signal_gap_bps"] < 0
    assert (
        abs(
            marketable_limit_price(side="long", decision_mid=100.0, policy=policy)
            - 100.3
        )
        < 1e-12
    )
    assert (
        abs(
            marketable_limit_price(side="short", decision_mid=100.0, policy=policy)
            - 99.7
        )
        < 1e-12
    )
