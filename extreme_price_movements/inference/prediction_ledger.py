"""Append-safe prediction ledger for traded and untraded live candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


class PredictionLedger:
    """Persist top live predictions and later outcome resolution state."""

    def __init__(self, path: str | Path = "data/live_state/prediction_ledger.parquet"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)

    def _write_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.path)

    def append_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows)
        if "timestamp" in new.columns:
            new["timestamp"] = pd.to_datetime(
                new["timestamp"], utc=True, errors="coerce"
            )
        old = self._read()
        out = new if old.empty else pd.concat([old, new], ignore_index=True, sort=False)
        subset = [
            c
            for c in ("timestamp", "symbol", "side", "strategy_id")
            if c in out.columns
        ]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="last")
        self._write_atomic(out)

    def load_unresolved(self, *, max_age_hours: int = 168) -> pd.DataFrame:
        df = self._read()
        if df.empty:
            return df
        ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(max_age_hours))
        status = df.get("outcome_status", pd.Series(index=df.index, dtype=object))
        unresolved = status.isna() | status.astype(str).isin({"", "pending", "open"})
        return df.loc[unresolved & (ts >= cutoff)].copy()

    def mark_resolved(self, updates: pd.DataFrame) -> None:
        if updates is None or updates.empty:
            return
        old = self._read()
        if old.empty:
            self._write_atomic(updates.copy())
            return
        key_cols = [
            c
            for c in ("timestamp", "symbol", "side", "strategy_id")
            if c in old.columns and c in updates.columns
        ]
        if not key_cols:
            self._write_atomic(pd.concat([old, updates], ignore_index=True, sort=False))
            return
        old_idx = old.set_index(key_cols, drop=False)
        upd_idx = updates.set_index(key_cols, drop=False)

        common_idx = upd_idx.index.intersection(old_idx.index)
        new_idx = upd_idx.index.difference(old_idx.index)

        new_cols = upd_idx.columns.difference(old_idx.columns)
        if not new_cols.empty:
            for col in new_cols:
                old_idx[col] = pd.NA

        if not common_idx.empty:
            # Drop empty/NA rows from update if any exist to prevent upcasting issues
            old_idx.loc[common_idx, upd_idx.columns] = upd_idx.loc[common_idx]

        if not new_idx.empty:
            old_idx = pd.concat([old_idx, upd_idx.loc[new_idx]], axis=0, sort=False)

        self._write_atomic(old_idx.reset_index(drop=True))


def top_fraction_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    rank_key: str = "normalized_rank_score",
    pct: float = 0.15,
) -> List[Dict[str, Any]]:
    """Return the top fraction per strategy for ledger persistence."""
    df = pd.DataFrame(list(rows))
    if df.empty or rank_key not in df.columns or "strategy_id" not in df.columns:
        return []
    keep_parts = []
    for _strategy, grp in df.groupby("strategy_id", sort=False):
        n_keep = max(1, int(len(grp) * float(pct)))
        keep_parts.append(grp.sort_values(rank_key, ascending=False).head(n_keep))
    return pd.concat(keep_parts, ignore_index=True).to_dict("records")
