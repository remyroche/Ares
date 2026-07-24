"""Exact feature-source contracts for the base-to-meta handoff.

The meta model consumes a mixture of materialized numeric columns, encoded
categoricals, fold-derived priors and post-selection OOD summaries.  A selected
feature must resolve to one of those sources.  Silently manufacturing a missing
selected feature with ``DataFrame.reindex(..., fill_value=0)`` changes model
semantics when the feature later becomes available, so legacy constant-zero
features require an explicit, persisted declaration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd


META_INPUT_CONTRACT_SCHEMA = "s52_meta_input_contract_v1"


def meta_feature_contract_hash(feature_names: Iterable[str]) -> str:
    payload = json.dumps(
        list(map(str, feature_names)), separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_meta_input_contract(
    feature_names: Sequence[str],
    *,
    materialized_columns: Iterable[str],
    categorical_columns: Iterable[str] = (),
    generated_features: Iterable[str] = (),
    legacy_constant_zero_features: Iterable[str] = (),
) -> dict[str, Any]:
    """Resolve every encoded model feature to its causal source.

    Categorical features are represented by their encoded
    ``<source>_<category>`` names.  Longest-prefix matching is used because raw
    source names can themselves contain underscores.
    """

    names = list(dict.fromkeys(str(name) for name in feature_names))
    materialized = set(map(str, materialized_columns))
    generated = set(map(str, generated_features))
    legacy_zero = set(map(str, legacy_constant_zero_features))
    categoricals = sorted(
        set(map(str, categorical_columns)), key=lambda value: (-len(value), value)
    )
    entries: list[dict[str, str]] = []
    unresolved: list[str] = []
    for feature in names:
        if feature in materialized:
            source_type = "materialized_numeric_or_boolean"
            source_column = feature
        elif feature in generated:
            source_type = "fold_or_post_selection_generated"
            source_column = feature
        else:
            source_column = next(
                (
                    column
                    for column in categoricals
                    if feature.startswith(f"{column}_")
                ),
                "",
            )
            if source_column:
                source_type = "categorical_one_hot"
            elif feature in legacy_zero:
                source_type = "legacy_constant_zero"
                source_column = ""
            else:
                source_type = "unresolved"
                source_column = ""
                unresolved.append(feature)
        entries.append(
            {
                "feature": feature,
                "source_type": source_type,
                "source_column": source_column,
            }
        )
    return {
        "schema": META_INPUT_CONTRACT_SCHEMA,
        "feature_names": names,
        "feature_count": len(names),
        "feature_contract_hash": meta_feature_contract_hash(names),
        "entries": entries,
        "unresolved_features": unresolved,
        "legacy_constant_zero_features": [
            entry["feature"]
            for entry in entries
            if entry["source_type"] == "legacy_constant_zero"
        ],
    }


def require_resolved_meta_input_contract(
    contract: Mapping[str, Any], *, role: str
) -> None:
    unresolved = list(map(str, contract.get("unresolved_features", [])))
    if unresolved:
        preview = ", ".join(unresolved[:12])
        suffix = "" if len(unresolved) <= 12 else f" (+{len(unresolved) - 12} more)"
        raise ValueError(
            f"{role} has {len(unresolved)} unresolved selected meta features: "
            f"{preview}{suffix}. Missing selected features cannot be silently "
            "materialized as zero; declare an explicit legacy constant-zero "
            "contract only when reproducing a frozen historical model."
        )


def materialize_legacy_constant_zeros(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    """Materialize only explicitly declared historical constant-zero inputs."""

    features = list(map(str, contract.get("legacy_constant_zero_features", [])))
    if not features:
        return frame
    out = frame.copy()
    for feature in features:
        out[feature] = 0.0
    return out


def require_encoded_meta_matrix(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    role: str,
) -> pd.DataFrame:
    """Return an exact ordered matrix without silently adding columns."""

    names = list(map(str, feature_names))
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{role} is missing {len(missing)} encoded meta inputs: "
            + ", ".join(missing[:12])
        )
    return frame.loc[:, names]
