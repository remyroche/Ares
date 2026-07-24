"""Live materialization for frozen residual-event AE/GMM state features.

The V9 residual/market-state calibrator is fitted offline, but its local
side-by-archetype and market AE/GMM transforms consume only pre-entry rows at
decision time.  This module packages that boundary explicitly so inference
cannot accidentally fall back to a policy that was validated with features it
does not actually have.
"""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.residual_event_archetypes import (
    OUTCOME_COLUMNS,
    ResidualEventArchetypeState,
    residual_event_feature_names,
    residual_event_market_feature_names,
)


STATE_FILENAME = "residual_event_state.joblib"
MANIFEST_FILENAME = "residual_event_state_manifest.json"
CONTRACT_FILENAME = "residual_event_state_contract.json"


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _canonical_policy_archetype(value: Any, side: str) -> str:
    """Match live classifier labels to the train-time residual-state keys."""

    text = str(value or "").strip().lower()
    side_text = str(side or "").strip().lower()
    for prefix in (f"{side_text}__", f"{side_text}||", f"{side_text}|"):
        if prefix and text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _state_path(data_root: str, run_id: str) -> Path:
    return Path(data_root) / "artifacts" / str(run_id) / "policy_params" / STATE_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_state_cached(path_text: str) -> ResidualEventArchetypeState:
    state = joblib.load(path_text)
    if not isinstance(state, ResidualEventArchetypeState):
        raise TypeError(
            f"Frozen residual-event state has unexpected type: {type(state).__name__}"
        )
    return state


def residual_event_state_input_feature_columns(
    payload: Mapping[str, Any] | None,
) -> set[str]:
    """Return raw pre-entry columns required by the packaged frozen state."""

    if not payload:
        return set()
    state = payload.get("state")
    if not isinstance(state, ResidualEventArchetypeState):
        return set()
    columns: set[str] = set()
    for model in state.local_models.values():
        columns.update(str(name) for name in model.feature_columns)
    for model in state.side_models.values():
        columns.update(str(name) for name in model.feature_columns)
    if state.market_model is not None:
        columns.update(str(name) for name in state.market_model.feature_columns)
    # ``score`` is injected from the frozen meta score; routing and timestamp
    # fields are similarly injected by ``materialize_live_residual_event_features``.
    columns.discard(str(state.config.score_col))
    columns.discard(str(state.config.side_col))
    columns.discard(str(state.config.archetype_col))
    columns.discard(str(state.config.timestamp_col))
    columns.discard(str(state.config.symbol_col))
    return columns


def load_live_residual_event_state_payload(
    data_root: str,
    run_id: str,
) -> dict[str, Any]:
    """Load the run-scoped V9 residual-event state if it was packaged."""

    path = _state_path(data_root, run_id)
    if not path.exists():
        return {}
    state = _load_state_cached(str(path.resolve()))
    manifest_path = path.with_name(MANIFEST_FILENAME)
    contract_path = path.with_name(CONTRACT_FILENAME)
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            manifest = {}
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Frozen residual-event state contract is unreadable: {contract_path}"
            ) from exc
        expected_hash = str(contract.get("state_sha256") or "")
        actual_hash = _sha256(path)
        if expected_hash and expected_hash != actual_hash:
            raise RuntimeError(
                "Frozen residual-event state checksum mismatch: "
                f"expected={expected_hash} actual={actual_hash}"
            )
    return {
        "state": state,
        "state_path": str(path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "contract_path": str(contract_path) if contract_path.exists() else "",
        "manifest": manifest,
        "input_feature_columns": sorted(residual_event_state_input_feature_columns({"state": state})),
        "generated_feature_columns": [
            *residual_event_feature_names(),
            *residual_event_market_feature_names(),
        ],
    }


def materialize_live_residual_event_features(
    features: pd.DataFrame,
    *,
    payload: Mapping[str, Any],
    side: str,
    policy_archetypes: Mapping[Any, Any] | Sequence[Any] | pd.Series,
    meta_scores: Mapping[Any, Any] | Sequence[Any] | pd.Series,
    signal_bar_ts: Any,
) -> pd.DataFrame:
    """Apply frozen residual-event transforms to a live batch.

    ``features`` contains one decision-time row per candidate.  The state does
    not receive realized outcomes: those only determined train-side clusters
    and priors.  Batch transformation also preserves the cross-sectional
    market-state calculation at the scored timestamp.
    """

    if not isinstance(features, pd.DataFrame) or features.empty:
        return pd.DataFrame(index=getattr(features, "index", None))
    state = payload.get("state")
    if not isinstance(state, ResidualEventArchetypeState):
        raise RuntimeError("Frozen residual-event state payload is unavailable")

    forbidden = [name for name in OUTCOME_COLUMNS if name in features.columns]
    if forbidden:
        raise ValueError(
            "Live residual-event input contains outcome columns: "
            + ", ".join(sorted(forbidden))
        )

    index = features.index
    work = features.copy()
    work[state.config.timestamp_col] = _as_utc_timestamp(signal_bar_ts)
    work[state.config.symbol_col] = index.astype(str).to_numpy(copy=False)
    work[state.config.side_col] = str(side).lower()
    archetypes = pd.Series(policy_archetypes, index=index, dtype="object").fillna("")
    work[state.config.archetype_col] = archetypes.map(
        lambda value: _canonical_policy_archetype(value, side)
    )
    work[state.config.score_col] = pd.to_numeric(
        pd.Series(meta_scores, index=index), errors="coerce"
    ).astype(np.float32, copy=False)

    transformed = state.transform_oos(work)
    expected = [
        *residual_event_feature_names(),
        *residual_event_market_feature_names(),
    ]
    missing = [name for name in expected if name not in transformed.columns]
    if missing:
        raise RuntimeError(
            "Frozen residual-event transform omitted generated features: "
            + ", ".join(missing[:20])
        )
    return transformed.reindex(index=index).astype(np.float32, copy=False)
