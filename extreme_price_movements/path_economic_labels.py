"""Train-time path-economics taxonomy shared by state-discovery components.

The existing clean/dirty, bad-MAE, timeout, and net-EV labels remain the
canonical labels. This module only materializes mutually exclusive *research*
descriptors that distinguish economic mechanisms which otherwise collapse into
"negative EV". They are safe to use as training outcomes or cluster-selection
diagnostics, never as OOS feature inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PATH_ECONOMIC_LABEL_COLUMNS: tuple[str, ...] = (
    "path_label_acute_adverse",
    "path_label_slow_timeout_loss",
    "path_label_clean_negative_ev",
    "path_label_dirty_negative_ev",
    "path_label_other_negative_ev",
    "path_label_durable_clean_positive",
    "path_label_other_positive",
)

PATH_ECONOMIC_STATE_NAMES: tuple[str, ...] = (
    "acute_adverse",
    "slow_timeout_loss",
    "clean_negative_ev",
    "dirty_negative_ev",
    "other_negative_ev",
    "durable_clean_positive",
    "other_positive",
    "unavailable",
)


@dataclass(frozen=True)
class PathEconomicLabelConfig:
    """Column contract and thresholds for retrospective path classification."""

    ev_col: str = "ev_after_1pct"
    clean_col: str = "clean_exec"
    dirty_col: str = "dirty_positive"
    # Acute failure must reflect the adverse path encountered before a
    # meaningful favorable resolution.  Full-path MAE is intentionally not
    # used here: it is common after profitable excursions and describes path
    # roughness, not a stop-like false positive.
    acute_bad_mae_col: str = "first_touch_bad_mae_1r"
    full_path_bad_mae_col: str = "full_path_bad_mae_1r"
    timeout_col: str = "timeout"
    binary_threshold: float = 0.5


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), default, dtype=np.float32)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
    return np.where(np.isfinite(values), values, np.float32(default)).astype(
        np.float32, copy=False
    )


def materialize_path_economic_labels(
    frame: pd.DataFrame,
    config: PathEconomicLabelConfig | None = None,
) -> pd.DataFrame:
    """Return exclusive realized path/economics labels for ``frame``.

    Precedence puts acute bad-path loss before timeout and clean payoff
    mismatch. A row can have several raw flags, but each resolved row receives
    exactly one state. Missing net EV remains ``unavailable`` rather than being
    silently classed as a loss.
    """
    cfg = config or PathEconomicLabelConfig()
    ev = _numeric(frame, cfg.ev_col)
    clean = _numeric(frame, cfg.clean_col, 0.0) >= np.float32(cfg.binary_threshold)
    dirty = _numeric(frame, cfg.dirty_col, 0.0) >= np.float32(cfg.binary_threshold)
    acute_bad_mae_values = _numeric(frame, cfg.acute_bad_mae_col)
    if not np.isfinite(acute_bad_mae_values).any():
        # Older research ledgers may only contain the broader full-path flag.
        # Retain a documented fallback rather than silently producing no
        # acute labels.
        acute_bad_mae_values = _numeric(frame, cfg.full_path_bad_mae_col, 0.0)
    acute_bad_mae = acute_bad_mae_values >= np.float32(cfg.binary_threshold)
    timeout = _numeric(frame, cfg.timeout_col, 0.0) >= np.float32(cfg.binary_threshold)

    resolved = np.isfinite(ev)
    negative = resolved & (ev <= 0.0)
    positive = resolved & ~negative

    acute_adverse = negative & acute_bad_mae
    slow_timeout = negative & ~acute_adverse & timeout
    clean_negative = negative & ~acute_adverse & ~slow_timeout & clean
    dirty_negative = negative & ~acute_adverse & ~slow_timeout & ~clean & dirty
    other_negative = negative & ~(
        acute_adverse | slow_timeout | clean_negative | dirty_negative
    )
    durable_clean = positive & clean & ~acute_bad_mae & ~timeout
    other_positive = positive & ~durable_clean

    masks = (
        acute_adverse,
        slow_timeout,
        clean_negative,
        dirty_negative,
        other_negative,
        durable_clean,
        other_positive,
    )
    matrix = np.column_stack(masks).astype(np.float32, copy=False)
    codes = np.full(len(frame), len(PATH_ECONOMIC_STATE_NAMES) - 1, dtype=np.int8)
    for code, mask in enumerate(masks):
        codes[mask] = np.int8(code)

    output = pd.DataFrame(
        matrix, index=frame.index, columns=PATH_ECONOMIC_LABEL_COLUMNS
    )
    output["path_economic_state"] = pd.Categorical.from_codes(
        codes,
        categories=list(PATH_ECONOMIC_STATE_NAMES),
    )
    return output


def path_economic_label_manifest() -> dict[str, object]:
    return {
        "schema": "path_economic_labels_v1",
        "labels": list(PATH_ECONOMIC_LABEL_COLUMNS),
        "states": list(PATH_ECONOMIC_STATE_NAMES),
        "contract": (
            "Retrospective outcome descriptors only. They may supervise train-only "
            "state selection/semantics and OOS diagnostics, but are not inference inputs. "
            "Acute adversity uses first-touch adverse behavior; full-path MAE is a "
            "separate post-entry roughness diagnostic."
        ),
    }


__all__ = [
    "PATH_ECONOMIC_LABEL_COLUMNS",
    "PATH_ECONOMIC_STATE_NAMES",
    "PathEconomicLabelConfig",
    "materialize_path_economic_labels",
    "path_economic_label_manifest",
]
