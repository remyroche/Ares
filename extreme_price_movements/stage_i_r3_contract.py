"""Immutable content lineage for the Stage-I R3/economics label surface.

The selector matrix is deliberately stored separately from the labels.  This
module provides the small common contract used by materialisation, selection,
and feature-count ladders to prove that they are operating on the same R3
target, realised economics, and validity population rather than merely the
same candidate identities.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY_COLUMNS: tuple[str, ...] = ("candidate_id", "__ts__", "__symbol__")
R3_REQUIRED_COLUMNS: tuple[str, ...] = (
    "r3_class",
    "r3_metric_target",
    "exact_net_bps",
    "label_available_ts",
)
R3_SOURCE_COLUMNS: tuple[str, ...] = (
    "t2_tp6_sl4_event",
    "robust_clear_event_b25",
    "robust_clear_soft_b25_t50",
)
VALIDITY_COLUMNS: tuple[str, ...] = (
    "target_invalid",
    "label_valid",
    "path_complete",
)


class StageIR3ContractError(ValueError):
    """Raised when a selector label/economics surface is not canonical."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def frame_content_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash ordered columns and values without relying on a parquet writer.

    File digests still bind the exact persisted bytes.  This second digest makes
    label/economics and feature-value lineage explicit and remains meaningful
    when a caller supplies an in-memory frame.
    """

    names = tuple(map(str, columns))
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise StageIR3ContractError(f"content hash lacks required columns: {missing[:8]}")
    view = frame.loc[:, list(names)]
    if view.columns.duplicated().any():
        raise StageIR3ContractError("content hash columns must be unique")
    digest = sha256()
    digest.update(
        json.dumps(
            {
                "columns": list(names),
                "dtypes": [str(view[name].dtype) for name in names],
                "rows": int(len(view)),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    # pandas' stable uint64 row hashing retains the exact row ordering, which
    # is part of every Stage-I selector/OOF contract.
    hashes = pd.util.hash_pandas_object(view, index=False, categorize=True).to_numpy(
        dtype=np.uint64, copy=False
    )
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def selector_validity_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return the canonical supervised-validity mask without inventing labels."""

    valid = np.ones(len(frame), dtype=bool)
    for column, expected in (
        ("target_invalid", False),
        ("label_valid", True),
        ("path_complete", True),
    ):
        if column in frame.columns:
            values = frame[column]
            if values.isna().any():
                raise StageIR3ContractError(f"{column} has null validity provenance")
            observed = values.astype(bool).to_numpy()
            valid &= observed if expected else ~observed
    return valid


def r3_label_economics_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe and hash the R3 supervision/economics surface.

    Source path fields are included where present, allowing materialised
    selectors to prove the declared TP6/SL4 robust-clear semantics.  Compact
    synthetic tests without those source fields remain supported, but still
    bind the derived R3 values, exact net, timestamps, and validity flags.
    """

    missing = sorted(set((*IDENTITY_COLUMNS, *R3_REQUIRED_COLUMNS)).difference(frame.columns))
    if missing:
        raise StageIR3ContractError(f"R3 label/economics contract lacks {missing}")
    if frame.loc[:, list(IDENTITY_COLUMNS)].isna().any().any() or frame.loc[:, list(IDENTITY_COLUMNS)].duplicated().any():
        raise StageIR3ContractError("R3 label/economics identities must be unique and non-null")
    classes = pd.to_numeric(frame["r3_class"], errors="coerce").to_numpy()
    metric = pd.to_numeric(frame["r3_metric_target"], errors="coerce").to_numpy(float)
    exact_net = pd.to_numeric(frame["exact_net_bps"], errors="coerce").to_numpy(float)
    available = pd.to_datetime(frame["label_available_ts"], utc=True, errors="coerce")
    if not np.isin(classes, (0, 1, 2)).all() or not np.isfinite(metric).all() or not np.isfinite(exact_net).all() or available.isna().any():
        raise StageIR3ContractError("R3 class, metric target, exact net, and label availability must be finite")
    source = [column for column in R3_SOURCE_COLUMNS if column in frame.columns]
    if "t2_tp6_sl4_event" in source and "robust_clear_event_b25" in source:
        adverse = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce").eq(1.0).to_numpy()
        clear = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce").eq(1.0).to_numpy()
        expected_class = np.select([adverse, clear], [0, 2], default=1)
        if not np.array_equal(classes.astype(np.int8), expected_class.astype(np.int8)):
            raise StageIR3ContractError("r3_class no longer matches adverse-first TP6/SL4 robust-clear semantics")
        if "robust_clear_soft_b25_t50" in source:
            soft = pd.to_numeric(frame["robust_clear_soft_b25_t50"], errors="coerce").to_numpy(float)
            if not np.isfinite(soft).all() or not np.allclose(metric, soft - adverse.astype(float), atol=1e-6, rtol=0.0):
                raise StageIR3ContractError("r3_metric_target no longer matches robust-clear soft minus adverse contract")
    value_columns = (*IDENTITY_COLUMNS, *R3_REQUIRED_COLUMNS, *source, *(column for column in VALIDITY_COLUMNS if column in frame.columns))
    validity = selector_validity_mask(frame)
    payload: dict[str, Any] = {
        "schema": "stage_i_r3_label_economics_contract_v1",
        "hard_target": {
            "adverse_first": "t2_tp6_sl4_event == 1",
            "robust_clear": "robust_clear_event_b25 == 1",
            "class_order": {"0": "adverse", "1": "weak_or_unresolved", "2": "robust_clear"},
            "conflict_precedence": "adverse_first",
        },
        "soft_metric_target": "robust_clear_soft_b25_t50 - adverse_indicator",
        "economics": {"column": "exact_net_bps", "units": "bps"},
        "label_availability_column": "label_available_ts",
        "source_columns_present": source,
        "validity_columns_present": [column for column in VALIDITY_COLUMNS if column in frame.columns],
        "rows": int(len(frame)),
        "supervised_valid_rows": int(validity.sum()),
        "supervised_invalid_or_incomplete_rows": int((~validity).sum()),
        "value_columns": list(value_columns),
        "value_sha256": frame_content_sha256(frame, value_columns),
        "validity_sha256": frame_content_sha256(
            pd.DataFrame({"supervised_valid": validity.astype(np.int8)}),
            ("supervised_valid",),
        ),
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    return payload


def require_r3_label_economics_contract(
    frame: pd.DataFrame, expected_sha256: str,
) -> dict[str, Any]:
    """Recompute a contract and fail closed if its declared digest drifted."""

    contract = r3_label_economics_contract(frame)
    if str(expected_sha256) != contract["contract_sha256"]:
        raise StageIR3ContractError("selector R3 label/economics contract hash drift")
    return contract


__all__ = [
    "IDENTITY_COLUMNS", "R3_REQUIRED_COLUMNS", "R3_SOURCE_COLUMNS", "VALIDITY_COLUMNS",
    "StageIR3ContractError", "frame_content_sha256", "selector_validity_mask",
    "r3_label_economics_contract", "require_r3_label_economics_contract",
]
