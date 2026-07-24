#!/usr/bin/env python3
"""Build a compact, causal feature-parity ledger for frozen overlay replay.

The output contains only keys and the serialized model features required by
the supplied overlay artifacts.  Existing frozen state columns are reused;
missing temporal mechanism fields are regenerated causally from the current
hourly feature store, with no model or outcome fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
    add_negative_residual_features,
)
from extreme_price_movements.unsupervised_regime_learning.economic_relevance import (
    materialize_composite_features,
)
from scripts import run_meta_residual_event_balanced_error_overlay as base
from scripts.run_meta_residual_interpretable_rule_overlay import (
    EPISODE_TRAJECTORY_MAX_ASOF_LAG,
    PERIOD_CONTEXT_FLOOR,
    PERIOD_STATE_FEATURES,
    EPISODE_TRAJECTORY_FEATURES,
    GLOBAL_MARKET_EPISODE_RISK,
    GLOBAL_MARKET_EPISODE_RISK_PCT,
    SIDE_MARKET_EPISODE_RISK,
    SIDE_MARKET_EPISODE_RISK_PCT,
    _add_episode_trajectory_features,
    _attach_trajectory_reference,
    _observable_market_state_panel,
    _trajectory_reference_panel,
    _trajectory_source_dependencies,
)
from scripts.backfill_negative_residual_temporal_mechanisms import _load_source_panels


def _bundle_path(overlay: Path, row: dict[str, Any]) -> Path:
    return overlay / "model__{arm}__{side}__{archetype}.joblib".format(
        arm=str(row["model_arm"]),
        side=str(row["side_name"]),
        archetype=str(row["archetype_policy_key"]),
    )


def _accepted_rows(overlay: Path) -> list[dict[str, Any]]:
    """Read active overlays while treating an empty research artifact as no-op."""

    path = overlay / "accepted_overlays.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _features(overlays: list[Path]) -> list[str]:
    names: list[str] = []
    for overlay in overlays:
        for row in _accepted_rows(overlay):
            bundle = joblib.load(_bundle_path(overlay, row))
            names.extend(str(name) for name in bundle["features"])
    return list(dict.fromkeys(names))


def _global_market_context_requirements(
    overlays: list[Path],
) -> tuple[list[str], list[tuple[Path, dict[str, Any]]]]:
    """Return frozen global-context primitives required by local bundles."""

    required: list[str] = []
    bundles: list[tuple[Path, dict[str, Any]]] = []
    for overlay in overlays:
        path = overlay / "global_market_episode_context.joblib"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        features = [str(name) for name in bundle.get("features", [])]
        required.extend(features)
        required.extend(_trajectory_source_dependencies(features))
        bundles.append((overlay, bundle))
    return list(dict.fromkeys(required)), bundles


def _side_market_context_requirements(
    overlays: list[Path],
) -> tuple[list[str], list[tuple[Path, str, dict[str, Any]]]]:
    """Return frozen same-side daily-context primitives and bundles."""

    required: list[str] = []
    bundles: list[tuple[Path, str, dict[str, Any]]] = []
    for overlay in overlays:
        for side in ("long", "short"):
            path = overlay / f"side_market_episode_context__{side}.joblib"
            if not path.exists():
                continue
            bundle = joblib.load(path)
            features = [str(name) for name in bundle.get("features", [])]
            required.extend(features)
            required.extend(_trajectory_source_dependencies(features))
            bundles.append((overlay, side, bundle))
    return list(dict.fromkeys(required)), bundles


def _composite_definitions(
    overlays: list[Path], required: list[str]
) -> list[dict[str, Any]]:
    """Load only frozen local composite definitions used by a model bundle."""

    required_set = set(required)
    selected: dict[str, dict[str, Any]] = {}
    for overlay in overlays:
        contract_path = overlay / "unsupervised_composite_contract.json"
        if not contract_path.exists():
            continue
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for definition in contract.get("composite_definitions", []):
            name = str(definition.get("name") or "")
            if name and (name in required_set or f"{name}__intensity" in required_set):
                selected[name] = dict(definition)
    return list(selected.values())


def _feature_schema_hash(features: list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_frozen_negative_panel(
    path: Path,
    required: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load the exact market-wide panel used by the model-training artifact."""

    schema = pq.read_schema(path)
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise ValueError(f"Frozen negative-residual panel lacks features: {missing}")
    panel = pd.read_parquet(path, columns=required)
    panel.index = pd.to_datetime(panel.index, utc=True, errors="coerce")
    panel = panel.loc[~panel.index.isna()]
    panel = panel.loc[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.loc[panel.index.to_series().between(start, end)]
    panel.index.name = "__ts__"
    return panel.reset_index()


def _materialize_global_market_episode_context(
    frame: pd.DataFrame,
    bundles: list[tuple[Path, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the frozen pooled market-day detector before local state scoring."""

    if not bundles:
        return frame, {"enabled": False, "bundles": []}
    if len(bundles) != 1:
        paths = [str(path) for path, _ in bundles]
        raise ValueError(
            "Forward parity supports one global market episode context bundle; "
            f"found {paths}"
        )
    overlay, bundle = bundles[0]
    features = [str(name) for name in bundle["features"]]
    dependencies = _trajectory_source_dependencies(features)
    raw = [
        name for name in [*features, *dependencies]
        if name not in set(EPISODE_TRAJECTORY_FEATURES)
    ]
    missing = [name for name in raw if name not in frame.columns]
    if missing:
        raise ValueError(f"Global market context parity lacks fields: {missing}")
    output = frame.copy()
    market = _observable_market_state_panel(output, raw)
    days = pd.to_datetime(output["__ts__"], utc=True).dt.floor("D")
    state = pd.DataFrame({"__ts__": pd.Index(days.unique()).sort_values()})
    if raw:
        state = pd.merge_asof(
            state.sort_values("__ts__", kind="stable"),
            market.rename(columns={"__ts__": "__market_ts__"}).sort_values(
                "__market_ts__", kind="stable"
            ),
            left_on="__ts__",
            right_on="__market_ts__",
            direction="backward",
            tolerance=EPISODE_TRAJECTORY_MAX_ASOF_LAG,
        ).drop(columns="__market_ts__")
    for name in features:
        if name not in state.columns:
            state[name] = np.float32(np.nan)
    trajectory_reference = _trajectory_reference_panel(output)
    state = _attach_trajectory_reference(state, trajectory_reference)
    state = _add_episode_trajectory_features(
        state, history=trajectory_reference
    ).set_index("__ts__")
    matrix = state.reindex(columns=features).to_numpy(dtype=np.float32, copy=True)
    scores = bundle["model"].predict_proba(bundle["robust"].transform(matrix))
    reference = np.asarray(bundle["reference"], dtype=np.float32)
    state[GLOBAL_MARKET_EPISODE_RISK] = scores.astype(np.float32)
    state[GLOBAL_MARKET_EPISODE_RISK_PCT] = base._midrank(scores, reference)
    output[GLOBAL_MARKET_EPISODE_RISK] = days.map(
        state[GLOBAL_MARKET_EPISODE_RISK]
    ).to_numpy(np.float32)
    output[GLOBAL_MARKET_EPISODE_RISK_PCT] = days.map(
        state[GLOBAL_MARKET_EPISODE_RISK_PCT]
    ).to_numpy(np.float32)
    return output, {
        "enabled": True,
        "overlay": str(overlay),
        "features": features,
        "feature_schema_hash": _feature_schema_hash(features),
        "state_days": int(len(state)),
        "trajectory_dependencies": dependencies,
        "contract": "One frozen pooled market score per UTC day, broadcast to rows by day.",
    }


def _materialize_side_market_episode_context(
    frame: pd.DataFrame,
    bundles: list[tuple[Path, str, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score frozen per-side daily context models without outcome inputs."""

    if not bundles:
        return frame, {"enabled": False, "bundles": []}
    by_side: dict[str, tuple[Path, dict[str, Any]]] = {}
    for overlay, side, bundle in bundles:
        if side in by_side:
            raise ValueError(f"Forward parity found multiple side context bundles for {side}")
        by_side[side] = (overlay, bundle)
    output = frame.copy()
    days = pd.to_datetime(output["__ts__"], utc=True).dt.floor("D")
    output[SIDE_MARKET_EPISODE_RISK] = np.float32(np.nan)
    output[SIDE_MARKET_EPISODE_RISK_PCT] = np.float32(np.nan)
    reports: list[dict[str, Any]] = []
    for side, (overlay, bundle) in by_side.items():
        features = [str(name) for name in bundle["features"]]
        dependencies = _trajectory_source_dependencies(features)
        raw = [
            name for name in [*features, *dependencies]
            if name not in set(EPISODE_TRAJECTORY_FEATURES)
        ]
        missing = [name for name in raw if name not in output.columns]
        if missing:
            raise ValueError(f"Side market context parity lacks fields for {side}: {missing}")
        market = _observable_market_state_panel(output, raw)
        state = pd.DataFrame({"__ts__": pd.Index(days.unique()).sort_values()})
        if raw:
            state = pd.merge_asof(
                state.sort_values("__ts__", kind="stable"),
                market.rename(columns={"__ts__": "__market_ts__"}).sort_values(
                    "__market_ts__", kind="stable"
                ),
                left_on="__ts__",
                right_on="__market_ts__",
                direction="backward",
                tolerance=EPISODE_TRAJECTORY_MAX_ASOF_LAG,
            ).drop(columns="__market_ts__")
        for name in features:
            if name not in state.columns:
                state[name] = np.float32(np.nan)
        reference = _trajectory_reference_panel(output)
        state = _add_episode_trajectory_features(
            _attach_trajectory_reference(state, reference), history=reference
        ).set_index("__ts__")
        matrix = state.reindex(columns=features).to_numpy(dtype=np.float32, copy=True)
        scores = bundle["model"].predict_proba(bundle["robust"].transform(matrix))
        percentile_reference = np.asarray(bundle["reference"], dtype=np.float32)
        state[SIDE_MARKET_EPISODE_RISK] = scores.astype(np.float32)
        state[SIDE_MARKET_EPISODE_RISK_PCT] = base._midrank(scores, percentile_reference)
        mask = output["side_name"].astype(str).eq(side)
        output.loc[mask, SIDE_MARKET_EPISODE_RISK] = days.loc[mask].map(
            state[SIDE_MARKET_EPISODE_RISK]
        ).to_numpy(np.float32)
        output.loc[mask, SIDE_MARKET_EPISODE_RISK_PCT] = days.loc[mask].map(
            state[SIDE_MARKET_EPISODE_RISK_PCT]
        ).to_numpy(np.float32)
        reports.append({
            "side": side,
            "overlay": str(overlay),
            "features": features,
            "feature_schema_hash": _feature_schema_hash(features),
            "state_days": int(len(state)),
            "trajectory_dependencies": dependencies,
        })
    return output, {
        "enabled": True,
        "bundles": reports,
        "contract": "One frozen pre-open market score per UTC day and side, broadcast to matching-side rows only.",
    }


def _materialize_period_state_features(
    frame: pd.DataFrame,
    overlays: list[Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild V15's causal side x archetype timestamp state in inference.

    The local period models fit on medians of the parent top-20 context at a
    timestamp, then broadcast one state score to parent top-10 rows.  Applying
    their bundles to raw asset rows would silently change the trained feature
    semantics.  This helper overwrites each selected bundle feature with its
    exact observable context-state value before the compact state ledger is
    written.
    """

    required_by_group: dict[tuple[str, str], tuple[list[str], str]] = {}
    for overlay in overlays:
        manifest_path = overlay / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("period_state_contract")
        if not contract:
            continue
        for row in _accepted_rows(overlay):
            bundle = joblib.load(_bundle_path(overlay, row))
            key = (str(row["side_name"]), str(row["archetype_policy_key"]))
            required_by_group[key] = (
                [str(name) for name in bundle["features"]],
                str(manifest.get("episode_state_granularity", "timestamp")),
            )
    if not required_by_group:
        return frame, {"enabled": False, "groups": []}
    if "parent_rank_v9" not in frame.columns:
        raise ValueError("Period-state parity requires parent_rank_v9 in the forward parent ledger")

    output = frame.copy()
    all_market_features = list(dict.fromkeys([
        name
        for features, _ in required_by_group.values()
        for name in [*features, *_trajectory_source_dependencies(features)]
        if name not in set(PERIOD_STATE_FEATURES)
        and name not in set(EPISODE_TRAJECTORY_FEATURES)
    ]))
    market_trajectory_reference = _trajectory_reference_panel(output)
    market_reference = _observable_market_state_panel(output, all_market_features)
    report_groups: list[dict[str, Any]] = []
    for (side, archetype), (features, granularity) in required_by_group.items():
        mask = (
            output["side_name"].astype(str).eq(side)
            & output["archetype_policy_key"].astype(str).eq(archetype)
        )
        if not mask.any():
            continue
        trajectory_dependencies = _trajectory_source_dependencies(features)
        derived = set(PERIOD_STATE_FEATURES) | set(EPISODE_TRAJECTORY_FEATURES)
        missing = [
            name
            for name in [*features, *trajectory_dependencies]
            if name not in output.columns and name not in derived
        ]
        if missing:
            raise ValueError(
                f"Period-state parity lacks bundle fields for {side}|{archetype}: {missing}"
            )
        if granularity == "daily_open":
            raw_features = list(dict.fromkeys([
                name for name in [*features, *trajectory_dependencies]
                if name not in derived
            ]))
            # These values have already been produced by a frozen same-side
            # context model.  Collapsing them again over the full universe
            # would average long and short risk and break the training-time
            # side contract.  Keep market primitives on the shared clock and
            # merge the side context from the matching local decision stream.
            side_context_features = [
                name for name in raw_features
                if name in {
                    SIDE_MARKET_EPISODE_RISK,
                    SIDE_MARKET_EPISODE_RISK_PCT,
                }
            ]
            market_raw_features = [
                name for name in raw_features if name not in set(side_context_features)
            ]
            days = pd.to_datetime(output.loc[mask, "__ts__"], utc=True).dt.floor("D")
            state = pd.DataFrame({"__ts__": pd.Index(days.unique()).sort_values()})
            if market_raw_features:
                reference = market_reference.loc[:, ["__ts__", *market_raw_features]].copy()
                state = pd.merge_asof(
                    state.sort_values("__ts__", kind="stable"),
                    reference.rename(columns={"__ts__": "__market_ts__"}).sort_values("__market_ts__", kind="stable"),
                    left_on="__ts__",
                    right_on="__market_ts__",
                    direction="backward",
                    tolerance=pd.Timedelta(minutes=90),
                ).drop(columns="__market_ts__")
            if side_context_features:
                local_context = output.loc[mask, ["__ts__", *side_context_features]].copy()
                local_context["day"] = pd.to_datetime(local_context["__ts__"], utc=True).dt.floor("D")
                local_context = (
                    local_context.groupby("day", observed=True, sort=True)[side_context_features]
                    .median()
                    .reset_index()
                    .rename(columns={"day": "__ts__"})
                )
                state = state.merge(
                    local_context, on="__ts__", how="left", validate="one_to_one"
                )
            for name in features:
                if name not in state:
                    state[name] = np.float32(np.nan)
            state = _attach_trajectory_reference(state, market_trajectory_reference)
            state = _add_episode_trajectory_features(
                state, history=market_trajectory_reference
            ).set_index("__ts__")
            for name in features:
                output.loc[mask, name] = days.map(state[name]).to_numpy(np.float32)
            report_groups.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "features": features,
                    "state_granularity": "daily_open",
                    "state_days": int(len(state)),
                    "trajectory_dependencies": trajectory_dependencies,
                    "trajectory_features": [
                        name for name in features if name in set(EPISODE_TRAJECTORY_FEATURES)
                    ],
                    "trajectory_clock": "full_forward_candidate_universe_timestamp_median",
                    "side_context_features": side_context_features,
                }
            )
            continue
        if granularity != "timestamp":
            raise ValueError(f"Unknown period-state granularity: {granularity}")
        context_mask = mask & pd.to_numeric(
            output["parent_rank_v9"], errors="coerce"
        ).ge(PERIOD_CONTEXT_FLOOR)
        raw_features = [
            name for name in [*features, *trajectory_dependencies]
            if name not in derived
        ]
        raw_features = list(dict.fromkeys(raw_features))
        context = output.loc[context_mask, ["__ts__", *raw_features]].copy()
        if context.empty:
            raise ValueError(f"No top-20 period context for {side}|{archetype}")
        if raw_features:
            for name in raw_features:
                context[name] = pd.to_numeric(context[name], errors="coerce").astype(np.float32)
            aggregate = context.groupby("__ts__", observed=True, sort=False)[raw_features].median()
            state = aggregate.reset_index()
        else:
            state = context.loc[:, ["__ts__"]].drop_duplicates()
        support = context.groupby("__ts__", observed=True, sort=False).size()
        state = state.merge(
            support.rename("period_context_rows").reset_index(),
            on="__ts__",
            how="left",
            validate="one_to_one",
        )
        for source, q90_name, iqr_name in (
            ("parent_rank_v9", "period_parent_rank_q90", "period_parent_rank_iqr"),
            ("score_meta_base_soft_label", "period_meta_score_q90", "period_meta_score_iqr"),
            ("hit_probability", "period_hit_probability_q90", "period_hit_probability_iqr"),
        ):
            if q90_name not in features and iqr_name not in features:
                continue
            if source not in output.columns:
                raise ValueError(
                    f"Period-state parity requires {source} for {side}|{archetype}"
                )
            source_values = pd.to_numeric(
                output.loc[context_mask, source], errors="coerce"
            )
            grouped = source_values.groupby(
                output.loc[context_mask, "__ts__"], observed=True, sort=False
            )
            if q90_name in features:
                state = state.merge(
                    grouped.quantile(0.90).rename(q90_name).reset_index(),
                    on="__ts__", how="left", validate="one_to_one",
                )
            if iqr_name in features:
                state = state.merge(
                    (grouped.quantile(0.75) - grouped.quantile(0.25))
                    .rename(iqr_name)
                    .reset_index(),
                    on="__ts__", how="left", validate="one_to_one",
                )
        state = _attach_trajectory_reference(state, market_trajectory_reference)
        state = _add_episode_trajectory_features(
            state,
            history=market_trajectory_reference,
        )
        state = state.set_index("__ts__")
        local_timestamps = output.loc[mask, "__ts__"]
        for name in features:
            output.loc[mask, name] = local_timestamps.map(state[name]).to_numpy(
                np.float32
            )
        report_groups.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "features": features,
                "state_granularity": "timestamp",
                "trajectory_dependencies": trajectory_dependencies,
                "trajectory_features": [
                    name for name in features if name in set(EPISODE_TRAJECTORY_FEATURES)
                ],
                "trajectory_clock": "full_forward_candidate_universe_timestamp_median",
                "context_rows": int(context_mask.sum()),
                "decision_rows": int(
                    (mask & pd.to_numeric(output["parent_rank_v9"], errors="coerce").ge(0.90)).sum()
                ),
                "state_timestamps": int(len(state)),
            }
        )
    return output, {
        "enabled": True,
        "context_floor": PERIOD_CONTEXT_FLOOR,
        "trajectory_clock": "full_forward_candidate_universe_timestamp_median",
        "groups": report_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frozen-negative-residual-panel",
        type=Path,
        help=(
            "Exact timestamp-indexed negative-residual panel used when the overlay "
            "was trained. When supplied, its values replace mutable store values "
            "for all required negative-residual fields."
        ),
    )
    parser.add_argument(
        "--enforce-existing-feature-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail closed when regenerated feature values materially disagree with "
            "stored observable values from the frozen source artifact."
        ),
    )
    parser.add_argument(
        "--max-existing-feature-relative-iqr-difference",
        type=float,
        default=0.10,
        help="Maximum mean absolute difference, normalized by stored-feature IQR.",
    )
    args = parser.parse_args()

    required = _features(args.overlay)
    global_context_requirements, global_context_bundles = (
        _global_market_context_requirements(args.overlay)
    )
    side_context_requirements, side_context_bundles = (
        _side_market_context_requirements(args.overlay)
    )
    period_state_required = False
    for overlay in args.overlay:
        manifest_path = overlay / "manifest.json"
        if manifest_path.exists():
            period_state_required |= bool(
                json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "period_state_contract"
                )
            )
    composite_definitions = _composite_definitions(args.overlay, required)
    composite_inputs = list(
        dict.fromkeys(
            feature
            for definition in composite_definitions
            for feature in (definition.get("feature"), definition.get("feature_b"))
            if feature
        )
    )
    period_source_features = (
        ["parent_rank_v9", "score_meta_base_soft_label", "hit_probability"]
        if period_state_required
        else []
    )
    trajectory_dependencies = _trajectory_source_dependencies(required)
    input_features = list(dict.fromkeys([
        *required,
        *global_context_requirements,
        *side_context_requirements,
        *composite_inputs,
        *period_source_features,
    ]))
    input_features = list(dict.fromkeys([*input_features, *trajectory_dependencies]))
    # The frozen parent ledger already carries a full-universe subset of the
    # observable residual/context features used at scoring time. Prefer those
    # exact forward values, then use the compact residual-state artifact only
    # to fill fields absent from that ledger.
    parent_schema = set(pq.read_schema(args.parent).names)
    parent_features = [name for name in input_features if name in parent_schema]
    parent_context = [
        name for name in ("side_name", "archetype_policy_key")
        if name in parent_schema and name not in base.KEYS
    ]
    parent = pd.read_parquet(
        args.parent, columns=[*base.KEYS, *parent_context, *parent_features]
    )
    parent["__ts__"] = pd.to_datetime(parent["__ts__"], utc=True)
    if parent.duplicated(base.KEYS).any():
        raise ValueError("Parent forward keys are not unique")
    state_schema = set(pq.read_schema(args.state).names)
    state_features = [name for name in input_features if name in state_schema]
    state = pd.read_parquet(args.state, columns=[*base.KEYS, *state_features])
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    state = state.loc[
        state["__ts__"].between(parent["__ts__"].min(), parent["__ts__"].max())
    ]
    if state.duplicated(base.KEYS).any():
        raise ValueError("Frozen state source has duplicate forward keys")
    out = parent.merge(
        state,
        on=base.KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "__state"),
    )
    for name in state_features:
        state_name = f"{name}__state"
        if state_name in out.columns:
            out[name] = out[name].combine_first(out[state_name]) if name in out else out[state_name]
            out.drop(columns=state_name, inplace=True)

    negative_required = [
        name for name in input_features if name in set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    ]
    regenerated: list[str] = []
    existing_parity: dict[str, dict[str, float]] = {}
    frozen_panel_manifest: dict[str, Any] | None = None
    if negative_required:
        if args.frozen_negative_residual_panel is not None:
            panel_path = args.frozen_negative_residual_panel
            if not panel_path.is_file():
                raise FileNotFoundError(f"Frozen negative-residual panel not found: {panel_path}")
            frozen_panel = _load_frozen_negative_panel(
                panel_path,
                negative_required,
                start=parent["__ts__"].min(),
                end=parent["__ts__"].max(),
            ).set_index("__ts__")
            required_timestamps = pd.DatetimeIndex(parent["__ts__"].unique())
            missing_timestamps = required_timestamps.difference(frozen_panel.index)
            if len(missing_timestamps):
                raise RuntimeError(
                    "Frozen negative-residual panel lacks parent timestamps: "
                    f"count={len(missing_timestamps)}, first={missing_timestamps.min()}"
                )
            for name in negative_required:
                out[name] = parent["__ts__"].map(frozen_panel[name]).to_numpy(np.float32)
            frozen_panel_manifest = {
                "path": str(panel_path),
                "sha256": _file_hash(panel_path),
                "rows": int(len(frozen_panel)),
                "timestamp_min": str(frozen_panel.index.min()),
                "timestamp_max": str(frozen_panel.index.max()),
                "features": negative_required,
                "parent_timestamp_match_rate": 1.0,
                "contract": "Exact frozen panel values override mutable source-store values.",
            }
        else:
            # The residual transformations use at most a 30-day robust reference
            # plus five-day persistence windows. Keep a small buffer while avoiding
            # a multi-year, full-universe panel for a 12-day forward replay.
            panels = _load_source_panels(
                args.source_root,
                start=parent["__ts__"].min() - pd.Timedelta(days=45),
            )
            end = parent["__ts__"].max()
            panels = {name: frame.loc[frame.index <= end] for name, frame in panels.items()}
            generated = add_negative_residual_features(
                panels,
                requested_feature_keys=negative_required,
                cfg={"feature_bars_per_hour": 1},
            )
            missing_generated = sorted(set(negative_required) - set(generated))
            if missing_generated:
                raise RuntimeError(f"Could not causally regenerate residual fields: {missing_generated}")
            ts = parent["__ts__"].to_numpy()
            symbols = parent["__symbol__"].astype(str).to_numpy()
            for name in negative_required:
                matrix = panels[name]
                indexer = matrix.index.get_indexer(ts)
                values = np.full(len(parent), np.nan, dtype=np.float32)
                for symbol in pd.unique(symbols):
                    if symbol not in matrix.columns:
                        continue
                    mask = symbols == symbol
                    valid = indexer[mask] >= 0
                    positions = np.flatnonzero(mask)
                    values[positions[valid]] = matrix[symbol].to_numpy(np.float32)[indexer[mask][valid]]
                if name in out:
                    old = pd.to_numeric(out[name], errors="coerce").to_numpy(np.float32)
                    both = np.isfinite(old) & np.isfinite(values)
                    if both.any():
                        diff = np.abs(old[both] - values[both])
                        old_iqr = float(
                            np.nanquantile(old[both], 0.75)
                            - np.nanquantile(old[both], 0.25)
                        )
                        existing_parity[name] = {
                            "overlap_rows": int(both.sum()),
                            "mean_abs_difference": float(diff.mean()),
                            "max_abs_difference": float(diff.max()),
                            "reference_iqr": old_iqr,
                            "mean_abs_difference_over_iqr": float(
                                diff.mean() / max(old_iqr, 1e-6)
                            ),
                        }
                out[name] = values
                regenerated.append(name)

    composite_columns: list[str] = []
    if composite_definitions:
        by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for definition in composite_definitions:
            key = (
                str(definition["side_name"]),
                str(definition["archetype_policy_key"]),
            )
            by_group.setdefault(key, []).append(definition)
        for (side, archetype), definitions in by_group.items():
            mask = (
                out["side_name"].astype(str).eq(side)
                & out["archetype_policy_key"].astype(str).eq(archetype)
            )
            if not mask.any():
                continue
            local = materialize_composite_features(
                out.loc[mask], definitions, include_intensity=True
            )
            for name in local.columns:
                if name not in out:
                    out[name] = np.float32(np.nan)
                out.loc[mask, name] = local[name].to_numpy(np.float32)
                composite_columns.append(name)

    global_market_context_report: dict[str, Any] = {"enabled": False, "bundles": []}
    if global_context_bundles:
        out, global_market_context_report = _materialize_global_market_episode_context(
            out, global_context_bundles
        )
    side_market_context_report: dict[str, Any] = {"enabled": False, "bundles": []}
    if side_context_bundles:
        out, side_market_context_report = _materialize_side_market_episode_context(
            out, side_context_bundles
        )

    period_state_report: dict[str, Any] = {"enabled": False, "groups": []}
    if period_state_required:
        out, period_state_report = _materialize_period_state_features(out, args.overlay)

    if args.enforce_existing_feature_parity:
        failures = {
            name: values
            for name, values in existing_parity.items()
            if values["mean_abs_difference_over_iqr"]
            > args.max_existing_feature_relative_iqr_difference
        }
        if failures:
            detail = ", ".join(
                f"{name}={values['mean_abs_difference_over_iqr']:.3f}"
                for name, values in sorted(failures.items())
            )
            raise RuntimeError(
                "Frozen feature parity failed; regenerated values differ from the "
                f"stored observable contract by more than "
                f"{args.max_existing_feature_relative_iqr_difference:.3f} IQR: {detail}"
            )

    missing = sorted(set(required) - set(out.columns))
    if missing:
        raise ValueError(f"Forward parity ledger still lacks features: {missing}")
    coverage = {name: float(out[name].notna().mean()) for name in required}
    zero_coverage = [name for name, value in coverage.items() if value == 0.0]
    if zero_coverage:
        raise ValueError(f"Forward features have zero coverage: {zero_coverage}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.loc[:, [*base.KEYS, *required]].to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": "interpretable_overlay_forward_state_v1",
        "parent": str(args.parent),
        "state": str(args.state),
        "source_root": str(args.source_root),
        "rows": int(len(out)),
        "timestamp_min": str(parent["__ts__"].min()),
        "timestamp_max": str(parent["__ts__"].max()),
        "required_features": required,
        "feature_schema_hash": _feature_schema_hash(required),
        "reused_parent_forward_features": parent_features,
        "reused_state_features": state_features,
        "frozen_negative_residual_panel": frozen_panel_manifest,
        "causally_regenerated_negative_residual_features": regenerated,
        "materialized_frozen_local_composites": sorted(set(composite_columns)),
        "period_state_aggregation": period_state_report,
        "global_market_episode_context": global_market_context_report,
        "side_market_episode_context": side_market_context_report,
        "composite_definition_count": int(len(composite_definitions)),
        "temporal_features_regenerated": [
            name for name in regenerated if name in set(NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS)
        ],
        "existing_feature_overlap_parity": existing_parity,
        "existing_feature_parity_contract": {
            "enforced": bool(args.enforce_existing_feature_parity),
            "max_relative_iqr_difference": float(
                args.max_existing_feature_relative_iqr_difference
            ),
        },
        "feature_coverage": coverage,
        "outcome_columns_read": [],
        "contract": "No model state, scaler, label, outcome, feature selection, or threshold is fitted by this materialization.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
