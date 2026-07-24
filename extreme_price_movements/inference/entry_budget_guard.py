"""Persistent per-signal-bar entry budget and idempotency guard."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

ENTRY_BUDGET_GUARD_SCHEMA = "entry_budget_guard_v3_pre_leverage_wallet"
_ACTIVE_STATUSES = ("reserved", "committed")


def _utc_iso(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid signal bar timestamp: {value!r}")
    return pd.Timestamp(ts).isoformat()


def _identity_part(value: Any) -> str:
    return str(value or "").strip().lower()


def entry_decision_key(*, symbol: Any, side: Any, strategy_id: Any) -> str:
    """Return the stable per-bar identity of one entry decision."""
    raw = "|".join(
        (_identity_part(symbol), _identity_part(side), _identity_part(strategy_id))
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EntryBudgetReservation:
    allowed: bool
    reason: str
    token: Optional[str]
    active_count: int
    remaining: int
    reserved_notional: float = 0.0
    pending_reserved_notional: float = 0.0
    remaining_notional: Optional[float] = None
    reserved_allocated_capital: float = 0.0
    pending_allocated_capital: float = 0.0
    remaining_allocated_capital: Optional[float] = None


class PersistentEntryBudgetGuard:
    """Atomically reserve entry slots shared by restarts and live processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        policy_id: str,
        max_entries_per_bar: int,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_id = str(policy_id or "latest")
        self.max_entries_per_bar = max(0, int(max_entries_per_bar))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                conn.close()
                raise
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entry_budget_slots (
                    policy_id TEXT NOT NULL,
                    signal_bar_ts TEXT NOT NULL,
                    decision_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reservation_token TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    order_id TEXT,
                    detail TEXT,
                    reserved_notional REAL NOT NULL DEFAULT 0.0,
                    effective_leverage REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (policy_id, signal_bar_ts, decision_key)
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(entry_budget_slots)")
            }
            if "reserved_notional" not in columns:
                conn.execute(
                    "ALTER TABLE entry_budget_slots "
                    "ADD COLUMN reserved_notional REAL NOT NULL DEFAULT 0.0"
                )
            if "effective_leverage" not in columns:
                conn.execute(
                    "ALTER TABLE entry_budget_slots "
                    "ADD COLUMN effective_leverage REAL NOT NULL DEFAULT 1.0"
                )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_entry_budget_bar_status
                ON entry_budget_slots(policy_id, signal_bar_ts, status)
                """
            )
            conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return pd.Timestamp.now(tz="UTC").isoformat()

    def _count_active(self, conn: sqlite3.Connection, signal_bar_ts: str) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM entry_budget_slots
            WHERE policy_id = ? AND signal_bar_ts = ?
              AND status IN ('reserved', 'committed')
            """,
            (self.policy_id, signal_bar_ts),
        ).fetchone()
        return int(row["n"] if row else 0)

    def _sum_pending_notional(self, conn: sqlite3.Connection) -> float:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(reserved_notional), 0.0) AS total
            FROM entry_budget_slots
            WHERE policy_id = ? AND status = 'reserved'
            """,
            (self.policy_id,),
        ).fetchone()
        return float(row["total"] if row else 0.0)

    def pending_reserved_notional(self) -> float:
        """Return unresolved quote-notional reservations for this policy."""
        with self._connect() as conn:
            return self._sum_pending_notional(conn)

    def _sum_pending_allocated_capital(self, conn: sqlite3.Connection) -> float:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(reserved_notional /
                CASE WHEN effective_leverage > 1.0 THEN effective_leverage ELSE 1.0 END
            ), 0.0) AS total
            FROM entry_budget_slots
            WHERE policy_id = ? AND status = 'reserved'
            """,
            (self.policy_id,),
        ).fetchone()
        return float(row["total"] if row else 0.0)

    def pending_allocated_capital(self) -> float:
        with self._connect() as conn:
            return self._sum_pending_allocated_capital(conn)

    def reserve(
        self,
        *,
        signal_bar_ts: Any,
        symbol: str,
        side: str,
        strategy_id: str,
        source: str = "live_auction",
        requested_notional: Optional[float] = None,
        max_total_notional: Optional[float] = None,
        open_marked_notional: float = 0.0,
        effective_leverage: float = 1.0,
        max_total_allocated_capital: Optional[float] = None,
        open_allocated_capital: float = 0.0,
    ) -> EntryBudgetReservation:
        bar_ts = _utc_iso(signal_bar_ts)
        key = entry_decision_key(symbol=symbol, side=side, strategy_id=strategy_id)
        now = self._now_iso()
        token = hashlib.sha256(
            f"{self.policy_id}|{bar_ts}|{key}|{now}".encode("utf-8")
        ).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT status, reservation_token
                FROM entry_budget_slots
                WHERE policy_id = ? AND signal_bar_ts = ? AND decision_key = ?
                """,
                (self.policy_id, bar_ts, key),
            ).fetchone()
            active_count = self._count_active(conn, bar_ts)
            pending_notional = self._sum_pending_notional(conn)
            pending_allocated = self._sum_pending_allocated_capital(conn)
            if existing is not None and str(existing["status"]) in _ACTIVE_STATUSES:
                conn.commit()
                return EntryBudgetReservation(
                    allowed=False,
                    reason="already_reserved_or_committed_for_signal_bar",
                    token=None,
                    active_count=active_count,
                    remaining=max(self.max_entries_per_bar - active_count, 0),
                    pending_reserved_notional=pending_notional,
                    pending_allocated_capital=pending_allocated,
                )
            if active_count >= self.max_entries_per_bar:
                conn.commit()
                return EntryBudgetReservation(
                    allowed=False,
                    reason="persistent_max_new_entries_per_bar_reached",
                    token=None,
                    active_count=active_count,
                    remaining=0,
                    pending_reserved_notional=pending_notional,
                    pending_allocated_capital=pending_allocated,
                )
            reserved_notional = 0.0
            reserved_allocated = 0.0
            remaining_notional: Optional[float] = None
            remaining_allocated: Optional[float] = None
            leverage = max(float(effective_leverage), 1.0)
            if max_total_allocated_capital is not None:
                requested = max(float(requested_notional or 0.0), 0.0)
                limit_allocated = max(float(max_total_allocated_capital), 0.0)
                opened_allocated = max(float(open_allocated_capital), 0.0)
                remaining_allocated = max(
                    limit_allocated - opened_allocated - pending_allocated, 0.0
                )
                remaining_notional = remaining_allocated * leverage
                reserved_notional = min(requested, remaining_notional)
                reserved_allocated = reserved_notional / leverage
                if reserved_notional <= 0.0:
                    conn.commit()
                    return EntryBudgetReservation(
                        allowed=False,
                        reason="max_pre_leverage_wallet_investment_reached",
                        token=None,
                        active_count=active_count,
                        remaining=max(self.max_entries_per_bar - active_count, 0),
                        pending_reserved_notional=pending_notional,
                        pending_allocated_capital=pending_allocated,
                        remaining_notional=remaining_notional,
                        remaining_allocated_capital=remaining_allocated,
                    )
            elif requested_notional is not None or max_total_notional is not None:
                requested = max(float(requested_notional or 0.0), 0.0)
                limit = max(float(max_total_notional or 0.0), 0.0)
                opened = max(float(open_marked_notional), 0.0)
                remaining_notional = max(limit - opened - pending_notional, 0.0)
                reserved_notional = min(requested, remaining_notional)
                reserved_allocated = reserved_notional / leverage
                if reserved_notional <= 0.0:
                    conn.commit()
                    return EntryBudgetReservation(
                        allowed=False,
                        reason="max_pre_leverage_wallet_investment_reached",
                        token=None,
                        active_count=active_count,
                        remaining=max(self.max_entries_per_bar - active_count, 0),
                        pending_reserved_notional=pending_notional,
                        remaining_notional=remaining_notional,
                    )
            conn.execute(
                """
                INSERT INTO entry_budget_slots (
                    policy_id, signal_bar_ts, decision_key, symbol, side,
                    strategy_id, status, reservation_token, source,
                    created_at, updated_at, order_id, detail, reserved_notional,
                    effective_leverage
                ) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, NULL, NULL, ?, ?)
                ON CONFLICT(policy_id, signal_bar_ts, decision_key) DO UPDATE SET
                    status = 'reserved',
                    reservation_token = excluded.reservation_token,
                    source = excluded.source,
                    updated_at = excluded.updated_at,
                    order_id = NULL,
                    detail = NULL,
                    reserved_notional = excluded.reserved_notional,
                    effective_leverage = excluded.effective_leverage
                """,
                (
                    self.policy_id,
                    bar_ts,
                    key,
                    str(symbol),
                    str(side),
                    str(strategy_id),
                    token,
                    str(source),
                    now,
                    now,
                    reserved_notional,
                    leverage,
                ),
            )
            active_count += 1
            conn.commit()
        return EntryBudgetReservation(
            allowed=True,
            reason="reserved",
            token=token,
            active_count=active_count,
            remaining=max(self.max_entries_per_bar - active_count, 0),
            reserved_notional=reserved_notional,
            pending_reserved_notional=pending_notional + reserved_notional,
            reserved_allocated_capital=reserved_allocated,
            pending_allocated_capital=pending_allocated + reserved_allocated,
            remaining_notional=(
                None
                if remaining_notional is None
                else max(remaining_notional - reserved_notional, 0.0)
            ),
            remaining_allocated_capital=(
                None
                if remaining_allocated is None
                else max(remaining_allocated - reserved_allocated, 0.0)
            ),
        )

    def commit(
        self,
        token: str,
        *,
        order_id: Any = None,
        detail: Any = None,
    ) -> bool:
        with self._connect() as conn:
            changed = conn.execute(
                """
                UPDATE entry_budget_slots
                SET status = 'committed', updated_at = ?, order_id = ?, detail = ?
                WHERE policy_id = ? AND reservation_token = ?
                  AND status = 'reserved'
                """,
                (
                    self._now_iso(),
                    None if order_id is None else str(order_id),
                    None if detail is None else str(detail),
                    self.policy_id,
                    str(token),
                ),
            ).rowcount
            conn.commit()
        return bool(changed)

    def release(self, token: str, *, detail: Any = None) -> bool:
        with self._connect() as conn:
            changed = conn.execute(
                """
                UPDATE entry_budget_slots
                SET status = 'released', updated_at = ?, detail = ?
                WHERE policy_id = ? AND reservation_token = ?
                  AND status = 'reserved'
                """,
                (
                    self._now_iso(),
                    None if detail is None else str(detail),
                    self.policy_id,
                    str(token),
                ),
            ).rowcount
            conn.commit()
        return bool(changed)

    def bootstrap_committed(
        self,
        *,
        signal_bar_ts: Any,
        entries: Iterable[Mapping[str, Any]],
        source: str = "trade_logger_bootstrap",
    ) -> int:
        """Import authoritative successful entries written before this guard ran."""
        bar_ts = _utc_iso(signal_bar_ts)
        now = self._now_iso()
        inserted = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for entry in entries:
                symbol = str(entry.get("symbol") or "")
                side = str(entry.get("side") or "")
                strategy_id = str(entry.get("strategy_id") or "")
                if not symbol or not side:
                    continue
                key = entry_decision_key(
                    symbol=symbol, side=side, strategy_id=strategy_id
                )
                token = hashlib.sha256(
                    f"bootstrap|{self.policy_id}|{bar_ts}|{key}".encode("utf-8")
                ).hexdigest()
                existing = conn.execute(
                    """
                    SELECT status, order_id
                    FROM entry_budget_slots
                    WHERE policy_id = ? AND signal_bar_ts = ? AND decision_key = ?
                    """,
                    (self.policy_id, bar_ts, key),
                ).fetchone()
                if existing is not None and str(existing["status"]) == "committed":
                    continue
                changed = conn.execute(
                    """
                    INSERT INTO entry_budget_slots (
                        policy_id, signal_bar_ts, decision_key, symbol, side,
                        strategy_id, status, reservation_token, source,
                        created_at, updated_at, order_id, detail
                    ) VALUES (?, ?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(policy_id, signal_bar_ts, decision_key) DO UPDATE SET
                        status = 'committed',
                        source = excluded.source,
                        updated_at = excluded.updated_at,
                        order_id = COALESCE(excluded.order_id, entry_budget_slots.order_id),
                        detail = excluded.detail,
                        reserved_notional = 0.0
                    """,
                    (
                        self.policy_id,
                        bar_ts,
                        key,
                        symbol,
                        side,
                        strategy_id,
                        token,
                        source,
                        now,
                        now,
                        str(entry.get("exchange_order_id") or "") or None,
                        "bootstrapped from canonical successful entry log",
                    ),
                ).rowcount
                inserted += int(changed or 0)
            conn.commit()
        return inserted

    def reconcile_reserved(
        self,
        *,
        signal_bar_ts: Any = None,
        successful_entries: Iterable[Mapping[str, Any]] = (),
        active_positions: Iterable[Mapping[str, Any]] = (),
        grace_seconds: float = 120.0,
        now: Any = None,
    ) -> dict[str, int]:
        """Resolve reservations left ambiguous by a restart or order timeout.

        Canonical successful entry logs are authoritative. An exchange position
        is sufficient corroboration when the logger write was interrupted. A
        reservation with neither form of evidence remains protected during a
        short grace period, then releases so lower-ranked auction candidates can
        use the slot on a retry of the same signal bar.
        """
        bar_ts = _utc_iso(signal_bar_ts) if signal_bar_ts is not None else None
        now_ts = pd.to_datetime(now, utc=True, errors="coerce")
        if pd.isna(now_ts):
            now_ts = pd.Timestamp.now(tz="UTC")
        grace = max(float(grace_seconds), 0.0)

        successful = {
            (
                _utc_iso(row.get("signal_bar_ts") or bar_ts),
                entry_decision_key(
                    symbol=row.get("symbol"),
                    side=row.get("side"),
                    strategy_id=row.get("strategy_id"),
                ),
            )
            for row in successful_entries
            if str(row.get("symbol") or "").strip()
            and str(row.get("side") or "").strip()
            and (row.get("signal_bar_ts") is not None or bar_ts is not None)
        }
        active_symbol_sides = {
            (_identity_part(row.get("symbol")), _identity_part(row.get("side")))
            for row in active_positions
            if str(row.get("symbol") or "").strip()
            and str(row.get("side") or "").strip()
        }
        counts = {
            "committed_from_log": 0,
            "committed_from_position": 0,
            "released_stale": 0,
            "kept_in_grace": 0,
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if bar_ts is None:
                rows = conn.execute(
                    """
                    SELECT signal_bar_ts, decision_key, symbol, side, created_at
                    FROM entry_budget_slots
                    WHERE policy_id = ? AND status = 'reserved'
                    """,
                    (self.policy_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT signal_bar_ts, decision_key, symbol, side, created_at
                    FROM entry_budget_slots
                    WHERE policy_id = ? AND signal_bar_ts = ? AND status = 'reserved'
                    """,
                    (self.policy_id, bar_ts),
                ).fetchall()
            for row in rows:
                row_bar_ts = str(row["signal_bar_ts"])
                key = str(row["decision_key"])
                symbol_side = (
                    _identity_part(row["symbol"]),
                    _identity_part(row["side"]),
                )
                source = None
                if (row_bar_ts, key) in successful:
                    source = "canonical_successful_entry_log"
                    counts["committed_from_log"] += 1
                elif symbol_side in active_symbol_sides:
                    source = "exchange_active_position"
                    counts["committed_from_position"] += 1
                if source is not None:
                    conn.execute(
                        """
                        UPDATE entry_budget_slots
                        SET status = 'committed', updated_at = ?, detail = ?,
                            reserved_notional = 0.0
                        WHERE policy_id = ? AND signal_bar_ts = ?
                          AND decision_key = ? AND status = 'reserved'
                        """,
                        (
                            self._now_iso(),
                            f"reconciled from {source}",
                            self.policy_id,
                            row_bar_ts,
                            key,
                        ),
                    )
                    continue
                created = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
                age_seconds = (
                    float((pd.Timestamp(now_ts) - pd.Timestamp(created)).total_seconds())
                    if not pd.isna(created)
                    else float("inf")
                )
                if age_seconds < grace:
                    counts["kept_in_grace"] += 1
                    continue
                conn.execute(
                    """
                    UPDATE entry_budget_slots
                    SET status = 'released', updated_at = ?,
                        detail = 'stale reservation without entry or active position',
                        reserved_notional = 0.0
                    WHERE policy_id = ? AND signal_bar_ts = ?
                      AND decision_key = ? AND status = 'reserved'
                    """,
                    (self._now_iso(), self.policy_id, row_bar_ts, key),
                )
                counts["released_stale"] += 1
            conn.commit()
        return counts

    def snapshot(self, signal_bar_ts: Any) -> dict[str, Any]:
        bar_ts = _utc_iso(signal_bar_ts)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, side, strategy_id, status, reservation_token,
                       source, created_at, updated_at, order_id, detail,
                       reserved_notional
                FROM entry_budget_slots
                WHERE policy_id = ? AND signal_bar_ts = ?
                ORDER BY created_at, decision_key
                """,
                (self.policy_id, bar_ts),
            ).fetchall()
            active_count = sum(str(row["status"]) in _ACTIVE_STATUSES for row in rows)
        return {
            "schema": ENTRY_BUDGET_GUARD_SCHEMA,
            "policy_id": self.policy_id,
            "signal_bar_ts": bar_ts,
            "max_entries_per_bar": self.max_entries_per_bar,
            "active_count": int(active_count),
            "pending_reserved_notional": float(
                sum(
                    float(row["reserved_notional"] or 0.0)
                    for row in rows
                    if str(row["status"]) == "reserved"
                )
            ),
            "remaining": max(self.max_entries_per_bar - int(active_count), 0),
            "entries": [dict(row) for row in rows],
            "path": str(self.path),
        }
