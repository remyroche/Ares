"""Frozen historical score-to-rank mapping for an alternative meta model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HistoricalScoreRankReference:
    """Map raw scores to causal percentiles using only a frozen prior sample."""

    score_col: str = "score_alternative"
    side_col: str = "side_name"
    sorted_scores_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    sorted_scores_global: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    fit_start: str | None = None
    fit_end: str | None = None
    rank_method: str = "right"

    def fit(self, frame: pd.DataFrame) -> "HistoricalScoreRankReference":
        if self.score_col not in frame.columns:
            raise ValueError(
                f"Historical rank fit missing score column: {self.score_col}"
            )
        scores = pd.to_numeric(frame[self.score_col], errors="coerce")
        side = (
            frame.get(self.side_col, pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        self.sorted_scores_by_side = {}
        for key, positions in side.groupby(side, sort=True).groups.items():
            values = scores.loc[positions].to_numpy(dtype=np.float32)
            values = np.sort(values[np.isfinite(values)])
            if values.size:
                self.sorted_scores_by_side[str(key)] = values
        values = scores.to_numpy(dtype=np.float32)
        self.sorted_scores_global = np.sort(values[np.isfinite(values)])
        if "__ts__" in frame.columns:
            ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
            self.fit_start = str(ts.min())
            self.fit_end = str(ts.max())
        return self

    @staticmethod
    def _rank(
        values: np.ndarray,
        reference: np.ndarray,
        *,
        method: str = "right",
    ) -> np.ndarray:
        out = np.full(len(values), np.nan, dtype=np.float32)
        finite = np.isfinite(values)
        if reference.size and finite.any():
            if method == "right":
                rank = np.searchsorted(reference, values[finite], side="right")
            elif method == "midrank":
                left = np.searchsorted(reference, values[finite], side="left")
                right = np.searchsorted(reference, values[finite], side="right")
                rank = (left + right) / 2.0
            else:
                raise ValueError(f"Unsupported historical rank method: {method!r}")
            out[finite] = (rank / float(reference.size)).astype(np.float32)
        return out

    def transform(self, frame: pd.DataFrame, score_col: str | None = None) -> pd.Series:
        source = str(score_col or self.score_col)
        if source not in frame.columns:
            raise ValueError(
                f"Historical rank transform missing score column: {source}"
            )
        scores = pd.to_numeric(frame[source], errors="coerce").to_numpy(
            dtype=np.float32
        )
        side = (
            frame.get(self.side_col, pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        output = np.full(len(frame), np.nan, dtype=np.float32)
        for key, positions in side.groupby(side, sort=False).indices.items():
            idx = np.asarray(positions, dtype=np.int64)
            reference = self.sorted_scores_by_side.get(
                str(key), self.sorted_scores_global
            )
            output[idx] = self._rank(
                scores[idx],
                reference,
                method=str(getattr(self, "rank_method", "right")),
            )
        return pd.Series(output, index=frame.index, dtype=np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "alternative_meta_historical_rank_reference_v1",
            "score_col": self.score_col,
            "side_col": self.side_col,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "global_rows": int(self.sorted_scores_global.size),
            "side_rows": {
                key: int(value.size)
                for key, value in self.sorted_scores_by_side.items()
            },
            "rank_method": str(getattr(self, "rank_method", "right")),
            "leakage_contract": "Reference scores precede every transformed OOS row; no outcomes are used.",
        }
