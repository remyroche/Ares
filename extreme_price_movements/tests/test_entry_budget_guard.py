from concurrent.futures import ThreadPoolExecutor

from extreme_price_movements.inference.entry_budget_guard import (
    PersistentEntryBudgetGuard,
)

BAR = "2026-07-17T13:00:00Z"


def _guard(tmp_path, cap=2):
    return PersistentEntryBudgetGuard(
        tmp_path / "entry_budget.sqlite",
        policy_id="policy_v1",
        max_entries_per_bar=cap,
    )


def test_budget_and_duplicate_survive_restart(tmp_path):
    guard = _guard(tmp_path)
    first = guard.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert first.allowed
    assert guard.commit(first.token, order_id="order-1")

    restarted = _guard(tmp_path)
    duplicate = restarted.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert not duplicate.allowed
    assert duplicate.reason == "already_reserved_or_committed_for_signal_bar"

    second = restarted.reserve(
        signal_bar_ts=BAR,
        symbol="BNT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert second.allowed
    assert restarted.commit(second.token, order_id="order-2")

    capped = restarted.reserve(
        signal_bar_ts=BAR,
        symbol="BERA/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert not capped.allowed
    assert capped.reason == "persistent_max_new_entries_per_bar_reached"
    assert restarted.snapshot(BAR)["active_count"] == 2


def test_released_definite_failure_restores_slot(tmp_path):
    guard = _guard(tmp_path, cap=1)
    reservation = guard.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert reservation.allowed
    assert guard.release(reservation.token, detail="order_rejected")
    replacement = guard.reserve(
        signal_bar_ts=BAR,
        symbol="BNT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert replacement.allowed


def test_each_signal_bar_has_an_independent_budget(tmp_path):
    guard = _guard(tmp_path, cap=1)
    first = guard.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert first.allowed
    assert guard.commit(first.token)
    next_bar = guard.reserve(
        signal_bar_ts="2026-07-17T14:00:00Z",
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert next_bar.allowed


def test_atomic_capacity_reservation_uses_pre_leverage_capital(tmp_path):
    guard = _guard(tmp_path, cap=4)
    first = guard.reserve(
        signal_bar_ts=BAR, symbol="A/USD:USD", side="long", strategy_id="global",
        requested_notional=300.0, effective_leverage=10.0,
        max_total_allocated_capital=80.0, open_allocated_capital=40.0,
    )
    assert first.allowed
    assert first.reserved_notional == 300.0
    assert first.reserved_allocated_capital == 30.0
    second = guard.reserve(
        signal_bar_ts=BAR, symbol="B/USD:USD", side="long", strategy_id="global",
        requested_notional=200.0, effective_leverage=5.0,
        max_total_allocated_capital=80.0, open_allocated_capital=40.0,
    )
    assert second.allowed
    assert second.reserved_notional == 50.0
    assert second.reserved_allocated_capital == 10.0
    assert guard.pending_allocated_capital() == 40.0


def test_trade_log_bootstrap_counts_existing_successes(tmp_path):
    guard = _guard(tmp_path, cap=2)
    inserted = guard.bootstrap_committed(
        signal_bar_ts=BAR,
        entries=[
            {
                "symbol": "LPT/USD:USD",
                "side": "short",
                "strategy_id": "global",
                "exchange_order_id": "order-1",
            },
            {
                "symbol": "BNT/USD:USD",
                "side": "short",
                "strategy_id": "global",
                "exchange_order_id": "order-2",
            },
        ],
    )
    assert inserted == 2
    assert guard.snapshot(BAR)["remaining"] == 0
    assert guard.bootstrap_committed(signal_bar_ts=BAR, entries=[]) == 0


def test_trade_log_bootstrap_promotes_existing_reservation(tmp_path):
    guard = _guard(tmp_path, cap=1)
    reservation = guard.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
        requested_notional=100.0,
        max_total_notional=1_000.0,
        open_marked_notional=0.0,
    )
    assert reservation.allowed
    assert guard.pending_reserved_notional() == 100.0

    changed = guard.bootstrap_committed(
        signal_bar_ts=BAR,
        entries=[
            {
                "symbol": "LPT/USD:USD",
                "side": "short",
                "strategy_id": "global",
                "exchange_order_id": "order-1",
            }
        ],
    )

    assert changed == 1
    assert guard.snapshot(BAR)["entries"][0]["status"] == "committed"
    assert guard.pending_reserved_notional() == 0.0


def test_reconcile_reserved_releases_stale_unconfirmed_slot(tmp_path):
    guard = _guard(tmp_path, cap=1)
    reservation = guard.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    assert reservation.allowed

    result = guard.reconcile_reserved(
        signal_bar_ts=BAR,
        now="2026-07-20T13:10:00Z",
        grace_seconds=120.0,
    )

    assert result["released_stale"] == 1
    assert guard.snapshot(BAR)["remaining"] == 1


def test_reconcile_reserved_keeps_or_commits_corroborated_slots(tmp_path):
    guard = _guard(tmp_path, cap=2)
    first = guard.reserve(
        signal_bar_ts=BAR,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
    )
    second = guard.reserve(
        signal_bar_ts=BAR,
        symbol="BNT/USD:USD",
        side="long",
        strategy_id="global",
    )
    assert first.allowed and second.allowed

    result = guard.reconcile_reserved(
        signal_bar_ts=BAR,
        active_positions=[{"symbol": "LPT/USD:USD", "side": "short"}],
        now="2026-07-20T13:10:00Z",
        grace_seconds=120.0,
    )

    assert result["committed_from_position"] == 1
    assert result["released_stale"] == 1
    statuses = {
        row["symbol"]: row["status"] for row in guard.snapshot(BAR)["entries"]
    }
    assert statuses == {"LPT/USD:USD": "committed", "BNT/USD:USD": "released"}


def test_reconcile_reserved_cleans_all_bars_and_matches_logs_by_bar(tmp_path):
    guard = _guard(tmp_path, cap=2)
    earlier_bar = "2026-07-17T12:00:00Z"
    logged = guard.reserve(
        signal_bar_ts=earlier_bar,
        symbol="LPT/USD:USD",
        side="short",
        strategy_id="global",
        requested_notional=100.0,
        max_total_notional=1_000.0,
        open_marked_notional=0.0,
    )
    stale = guard.reserve(
        signal_bar_ts=BAR,
        symbol="BNT/USD:USD",
        side="long",
        strategy_id="global",
        requested_notional=100.0,
        max_total_notional=1_000.0,
        open_marked_notional=0.0,
    )
    assert logged.allowed and stale.allowed

    result = guard.reconcile_reserved(
        successful_entries=[
            {
                "signal_bar_ts": earlier_bar,
                "symbol": "LPT/USD:USD",
                "side": "short",
                "strategy_id": "global",
            }
        ],
        now="2026-07-20T13:10:00Z",
        grace_seconds=120.0,
    )

    assert result["committed_from_log"] == 1
    assert result["released_stale"] == 1
    assert guard.snapshot(earlier_bar)["entries"][0]["status"] == "committed"
    assert guard.snapshot(BAR)["entries"][0]["status"] == "released"
    assert guard.pending_reserved_notional() == 0.0


def test_concurrent_reservations_cannot_exceed_cap(tmp_path):
    path = tmp_path / "entry_budget.sqlite"

    def reserve(index):
        guard = PersistentEntryBudgetGuard(
            path, policy_id="policy_v1", max_entries_per_bar=2
        )
        return guard.reserve(
            signal_bar_ts=BAR,
            symbol=f"ASSET{index}/USD:USD",
            side="long",
            strategy_id="global",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(8)))
    assert sum(result.allowed for result in results) == 2
    assert _guard(tmp_path).snapshot(BAR)["active_count"] == 2


def test_persistent_notional_reservation_clips_atomically_across_workers(tmp_path):
    path = tmp_path / "entry_budget.sqlite"

    def reserve(index):
        guard = PersistentEntryBudgetGuard(
            path, policy_id="policy_v1", max_entries_per_bar=8
        )
        return guard.reserve(
            signal_bar_ts=BAR,
            symbol=f"ASSET{index}/USD:USD",
            side="long",
            strategy_id="global",
            requested_notional=800.0,
            max_total_notional=8_000.0,
            open_marked_notional=7_000.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))

    assert sum(result.reserved_notional for result in results) == 1_000.0
    assert sorted(result.reserved_notional for result in results) == [200.0, 800.0]
    snapshot = _guard(tmp_path, cap=8).snapshot(BAR)
    assert snapshot["pending_reserved_notional"] == 1_000.0
    assert _guard(tmp_path, cap=8).pending_reserved_notional() == 1_000.0
