"""Train-only complementary specialist-head selection.

The residual learner should not receive every correlated specialist score by
default.  This module estimates conditional mutual information inside bins of
the base score and greedily retains only heads with incremental information.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(len(values), -1, dtype=np.int16)
    if finite.sum() < 2:
        return out
    ranks = pd.Series(values[finite]).rank(method="average").to_numpy(float)
    out[finite] = np.minimum(bins - 1, np.floor((ranks - 1) * bins / len(ranks))).astype(np.int16)
    return out


def conditional_mi(score: Sequence[float], target: Sequence[int], condition: Sequence[float], *, bins: int = 10) -> float:
    """Empirical I(score; target | base-score-bin), in nats.

    All bin boundaries are fitted on the supplied (training-only) rows.  It is
    a stable selection statistic, not an inference feature or probability.
    """
    score_bins = _quantile_bins(np.asarray(score, dtype=float), bins)
    condition_bins = _quantile_bins(np.asarray(condition, dtype=float), bins)
    y = np.asarray(target)
    valid = (score_bins >= 0) & (condition_bins >= 0) & np.isfinite(y)
    if valid.sum() < max(20, bins):
        return 0.0
    total = int(valid.sum())
    result = 0.0
    for bucket in np.unique(condition_bins[valid]):
        index = valid & (condition_bins == bucket)
        if index.sum() >= 5 and np.unique(y[index]).size > 1:
            result += (index.sum() / total) * mutual_info_score(score_bins[index], y[index])
    return float(result)


def select_complementary_heads(frame: pd.DataFrame, head_columns: Sequence[str], *,
                               target_column: str, base_score_column: str,
                               max_heads: int | None = None, minimum_cmi: float = .001) -> tuple[list[str], pd.DataFrame]:
    """Greedily retain train-only heads with CMI beyond the base and peers."""
    missing = sorted(set([target_column, base_score_column, *head_columns]).difference(frame.columns))
    if missing:
        raise KeyError(f"head selection missing columns: {missing}")
    y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    base = pd.to_numeric(frame[base_score_column], errors="coerce").to_numpy(float)
    remaining = list(head_columns)
    selected: list[str] = []
    records: list[dict[str, object]] = []
    limit = len(remaining) if max_heads is None else max(0, int(max_heads))
    # Mean selected-head score is a compact conditioning proxy that prevents
    # duplicate heads dominating simply because each correlates with base.
    condition = base.copy()
    while remaining and len(selected) < limit:
        scored = []
        for name in remaining:
            value = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            cmi = conditional_mi(value, y, condition)
            corr = float(pd.Series(value).corr(pd.Series(condition), method="spearman"))
            scored.append((cmi, abs(corr) if np.isfinite(corr) else 1.0, name))
        cmi, correlation, winner = max(scored, key=lambda row: (row[0], -row[1], row[2]))
        records.append({"head": winner, "conditional_mi": cmi, "condition_spearman_abs": correlation,
                        "selection_rank": len(selected) + 1, "selected": cmi >= minimum_cmi})
        remaining.remove(winner)
        if cmi < minimum_cmi:
            break
        selected.append(winner)
        condition = np.nanmean(np.column_stack([base, *[pd.to_numeric(frame[name], errors="coerce").to_numpy(float) for name in selected]]), axis=1)
    for name in remaining:
        records.append({"head": name, "conditional_mi": np.nan, "condition_spearman_abs": np.nan,
                        "selection_rank": np.nan, "selected": False})
    return selected, pd.DataFrame(records).sort_values(["selected", "selection_rank", "head"], ascending=[False, True, True], kind="stable")
