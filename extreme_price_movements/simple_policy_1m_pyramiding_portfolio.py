"""Minute-event portfolio allocator for executable favorable pyramiding.

The allocator consumes frozen initial-anchor exits and precomputed executable
tranche schedules.  New entries and DCA requests compete in one deterministic
auction while respecting the production wallet, position, slot, new-entry,
and same-symbol limits.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def allocate_pyramiding_portfolio(
    entry_minute: np.ndarray,
    symbol_code: np.ndarray,
    rank_score: np.ndarray,
    exit_bars: np.ndarray,
    fill_bars: np.ndarray,
    tranche_gross_return: np.ndarray,
    tranche_net_return: np.ndarray,
    base_target: np.ndarray,
    entry_eligible: np.ndarray,
    target_tranche_multiples: np.ndarray,
    *,
    wallet_cap: float = 0.75,
    position_cap: float = 0.15,
    max_open: int = 8,
    max_new_per_minute: int = 2,
    max_dca_per_minute: int = 2,
    dca_priority_bonus: float = 0.0,
    minimum_order: float = 0.001,
) -> tuple[np.ndarray, ...]:
    """Allocate initial and add orders against one fixed unit wallet.

    Rows must be sorted by entry minute and descending rank within each minute.
    Exits release capacity before the auction at the same minute.  If a DCA is
    blocked by a wallet/position/order-count constraint, that tranche and all
    later tranches for the position are cancelled; historical trigger levels
    are never backfilled.
    """

    n = len(entry_minute)
    n_tranches = len(target_tranche_multiples)
    selected = np.zeros(n, dtype=np.bool_)
    net_pnl = np.zeros(n, dtype=np.float64)
    gross_pnl = np.zeros(n, dtype=np.float64)
    allocated = np.zeros(n, dtype=np.float64)
    initial_allocated = np.zeros(n, dtype=np.float64)
    filled_tranches = np.zeros(n, dtype=np.int8)
    rejected_entry_liquidity = np.zeros(n, dtype=np.bool_)
    rejected_entry_slot = np.zeros(n, dtype=np.bool_)
    rejected_entry_book = np.zeros(n, dtype=np.bool_)
    rejected_dca_book = np.zeros(n, dtype=np.int16)
    rejected_dca_order_cap = np.zeros(n, dtype=np.int16)

    open_row = np.full(max_open, -1, dtype=np.int32)
    open_exit = np.full(max_open, -1, dtype=np.int64)
    open_symbol = np.full(max_open, -1, dtype=np.int32)
    open_alloc = np.zeros(max_open, dtype=np.float64)
    next_tranche = np.ones(max_open, dtype=np.int8)

    if n == 0:
        return (
            selected, net_pnl, gross_pnl, allocated, initial_allocated,
            filled_tranches, rejected_entry_liquidity, rejected_entry_slot,
            rejected_entry_book, rejected_dca_book, rejected_dca_order_cap,
            np.asarray([0.0, 0.0, 0.0, 0.0]),
        )

    first_minute = int(entry_minute[0])
    last_minute = first_minute
    for i in range(n):
        candidate_exit = int(entry_minute[i]) + max(int(exit_bars[i]) + 1, 1)
        if candidate_exit > last_minute:
            last_minute = candidate_exit

    book = 0.0
    peak_book = 0.0
    exposure_auc = 0.0
    book_cap_minutes = 0.0
    turnover = 0.0
    ptr = 0
    request_done = np.zeros(n, dtype=np.bool_)

    for minute in range(first_minute, last_minute + 1):
        # Frozen exits release book before any order at this minute.
        for slot in range(max_open):
            if open_row[slot] >= 0 and open_exit[slot] <= minute:
                book -= open_alloc[slot]
                if book < 0.0 and book > -1e-10:
                    book = 0.0
                turnover += open_alloc[slot]
                open_row[slot] = -1
                open_exit[slot] = -1
                open_symbol[slot] = -1
                open_alloc[slot] = 0.0
                next_tranche[slot] = 1

        while ptr < n and int(entry_minute[ptr]) < minute:
            ptr += 1
        entry_start = ptr
        entry_end = ptr
        while entry_end < n and int(entry_minute[entry_end]) == minute:
            request_done[entry_end] = False
            entry_end += 1

        add_done = np.zeros(max_open, dtype=np.bool_)
        new_count = 0
        dca_count = 0
        while True:
            best_priority = -1e100
            best_kind = -1  # 0 new, 1 DCA
            best_id = -1

            for i in range(entry_start, entry_end):
                if not request_done[i] and float(rank_score[i]) > best_priority:
                    best_priority = float(rank_score[i])
                    best_kind = 0
                    best_id = i

            for slot in range(max_open):
                row = open_row[slot]
                if row < 0 or add_done[slot]:
                    continue
                tranche = int(next_tranche[slot])
                if tranche >= n_tranches or tranche >= fill_bars.shape[1]:
                    continue
                bar = int(fill_bars[row, tranche])
                if bar < 0:
                    continue
                fill_minute = int(entry_minute[row]) + bar + 1
                if fill_minute != minute:
                    continue
                priority = float(rank_score[row]) + float(dca_priority_bonus)
                if priority > best_priority:
                    best_priority = priority
                    best_kind = 1
                    best_id = slot

            if best_kind < 0:
                break

            if best_kind == 0:
                i = best_id
                request_done[i] = True
                if not entry_eligible[i]:
                    rejected_entry_liquidity[i] = True
                    continue
                if new_count >= max_new_per_minute:
                    rejected_entry_slot[i] = True
                    continue

                free_slot = -1
                same_symbol = False
                for slot in range(max_open):
                    if open_row[slot] < 0 and free_slot < 0:
                        free_slot = slot
                    elif open_row[slot] >= 0 and open_symbol[slot] == symbol_code[i]:
                        same_symbol = True
                if free_slot < 0 or same_symbol:
                    rejected_entry_slot[i] = True
                    continue

                desired = max(float(base_target[i]), 0.0) * float(target_tranche_multiples[0])
                available = min(float(position_cap), float(wallet_cap) - book)
                amount = min(desired, max(available, 0.0))
                if amount < minimum_order:
                    rejected_entry_book[i] = True
                    continue

                selected[i] = True
                new_count += 1
                open_row[free_slot] = i
                open_exit[free_slot] = int(entry_minute[i]) + max(int(exit_bars[i]) + 1, 1)
                open_symbol[free_slot] = int(symbol_code[i])
                open_alloc[free_slot] = amount
                next_tranche[free_slot] = 1
                initial_allocated[i] = amount
                allocated[i] = amount
                filled_tranches[i] = 1
                gross_pnl[i] += amount * float(tranche_gross_return[i, 0])
                net_pnl[i] += amount * float(tranche_net_return[i, 0])
                book += amount
                turnover += amount
            else:
                slot = best_id
                add_done[slot] = True
                row = int(open_row[slot])
                tranche = int(next_tranche[slot])
                if dca_count >= max_dca_per_minute:
                    rejected_dca_order_cap[row] += 1
                    next_tranche[slot] = n_tranches
                    continue
                desired = max(float(base_target[row]), 0.0) * float(
                    target_tranche_multiples[tranche]
                )
                if desired < minimum_order:
                    # A deliberately zero/tiny optional tranche is absent,
                    # not a book-cap rejection.
                    next_tranche[slot] = n_tranches
                    continue
                available = min(
                    float(position_cap) - open_alloc[slot],
                    float(wallet_cap) - book,
                )
                amount = min(desired, max(available, 0.0))
                if amount < minimum_order:
                    rejected_dca_book[row] += 1
                    next_tranche[slot] = n_tranches
                    continue

                dca_count += 1
                open_alloc[slot] += amount
                allocated[row] += amount
                filled_tranches[row] += 1
                gross_pnl[row] += amount * float(tranche_gross_return[row, tranche])
                net_pnl[row] += amount * float(tranche_net_return[row, tranche])
                book += amount
                turnover += amount
                next_tranche[slot] += 1

            if book > peak_book:
                peak_book = book

        exposure_auc += book
        if book >= float(wallet_cap) - 1e-9:
            book_cap_minutes += 1.0
        ptr = entry_end

    diagnostics = np.asarray(
        [
            peak_book,
            exposure_auc / max(last_minute - first_minute + 1, 1),
            book_cap_minutes / max(last_minute - first_minute + 1, 1),
            turnover,
        ],
        dtype=np.float64,
    )
    return (
        selected,
        net_pnl,
        gross_pnl,
        allocated,
        initial_allocated,
        filled_tranches,
        rejected_entry_liquidity,
        rejected_entry_slot,
        rejected_entry_book,
        rejected_dca_book,
        rejected_dca_order_cap,
        diagnostics,
    )


__all__ = ["allocate_pyramiding_portfolio"]
