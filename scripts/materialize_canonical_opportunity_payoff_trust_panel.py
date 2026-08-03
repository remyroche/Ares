#!/usr/bin/env python3
"""Materialize the full canonical opportunity/payoff/trust input panel.

The output binds the accepted 509,868 February--April 2025 canonical base OOF
identities to:

* the exact frozen side-local base input matrices used to create each OOF row;
* outcome-free score/rank/cutoff context available at the signal timestamp;
* archived pre-entry market/regime state and exact-gap past-only transitions;
* exact 12-hour execution economics and mutually exclusive exit labels; and
* strictly causal online mapping components, segregated as diagnostic context.

No model is fitted here.  Timing, MAE, target-price and wait-action fields are
deliberately absent from the feature contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_febapr_strict_residual_gross_regime_context import (  # noqa: E402
    CORE_FEATURES,
    REGIME_SOURCE_FEATURES,
    add_causal_transition_deltas,
    feature_quality,
    forbidden_feature_names,
)


BASE_ROOT = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"
LEDGER = (
    ROOT
    / "data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1"
    / "ledgers/canonical_base_exact1m_current_spread_cf.parquet"
)
LABEL_ROOT = (
    ROOT
    / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2"
    / "labels"
)
CAUSAL_MAP = (
    ROOT
    / "data_perp/artifacts/historical_causal_score_economics_mapping_20260729_v1"
    / "canonical_base__score_base_alpha/causal_mapped_candidates.parquet"
)
OUT = (
    ROOT
    / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
)

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SIDES = ("long", "short")
MONTHS = ("2025_02", "2025_03", "2025_04")

LABEL_COLUMNS = (
    "execution_decision_utc",
    "execution_label_end_utc",
    "candidate_month",
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "execution_cost_return",
    "execution_exit_reason",
    "execution_exit_minute",
    "execution_exit_class",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_soft_positive_12h",
    "opportunity_gross_above_cost_0bps",
    "opportunity_gross_above_cost_25bps",
    "positive_net_12h",
    "exit_is_trailing",
    "exit_is_timeout",
    "exit_is_full_stop",
    "exit_is_adverse",
    "exit_is_adverse_exit",
    "__first_touch_target_soft__",
    "__first_touch_capture_net__",
    "fold_id",
    "fold_validation_start_utc",
    "fold_validation_end_utc",
    "base_label_resolution_utc",
    "effective_label_resolution_utc",
)

CAUSAL_MAP_FEATURES = (
    "causal_score_percentile",
    "causal_score_decile",
    "mapped_eligible",
    "map_reference_rows",
    "map_side_reference_rows",
    "map_cell_reference_rows",
    "mapped_direct_net",
    "mapped_expected_gross",
    "mapped_expected_cost",
    "mapped_cost_std",
    "mapped_opportunity_probability_0bps",
    "mapped_opportunity_probability_25bps",
    "mapped_opportunity_gross_q50",
    "mapped_opportunity_gross_q80",
    "mapped_opportunity_gross_q50_support",
    "mapped_opportunity_gross_q80_support",
    "mapped_opportunity_q50_net_diagnostic",
    "mapped_opportunity_q80_net_diagnostic",
    "mapped_adverse_probability",
    "mapped_timeout_probability",
    "mapped_exit_mixture_net_diagnostic",
    "mapped_exit_probability_trailing",
    "mapped_exit_conditional_net_trailing",
    "mapped_exit_probability_timeout",
    "mapped_exit_conditional_net_timeout",
    "mapped_exit_probability_full_stop",
    "mapped_exit_conditional_net_full_stop",
    "mapped_exit_probability_adverse_exit",
    "mapped_exit_conditional_net_adverse_exit",
)
CAUSAL_MAP_CATEGORICAL_PROVENANCE = (
    "mapped_opportunity_gross_q50_fallback",
    "mapped_opportunity_gross_q80_fallback",
)
CAUSAL_MAP_FALLBACK_INDICATORS = tuple(
    f"mapped_opportunity_fallback_is_{level}"
    for level in ("side_decile", "side", "global", "unavailable")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True).astype(str)
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(
        values.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str)
    for column in (
        "__ts__",
        "__signal_ts__",
        "__decision_ts__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "fold_validation_start_utc",
        "fold_validation_end_utc",
        "base_label_resolution_utc",
        "effective_label_resolution_utc",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    return result


def _feature_contracts() -> dict[str, tuple[str, ...]]:
    contracts: dict[str, tuple[str, ...]] = {}
    reference: dict[str, tuple[str, ...]] = {}
    for side in SIDES:
        for month in MONTHS:
            manifest = _json(BASE_ROOT / f"shards/{side}_{month}/manifest.json")
            features = tuple(map(str, manifest["contracts"][side]["features"]))
            if side in reference and reference[side] != features:
                raise ValueError(f"{side} base feature contract changes across months")
            reference[side] = features
        contracts[side] = reference[side]
    if len(contracts["long"]) != 31 or len(contracts["short"]) != 8:
        raise ValueError("expected exact frozen 31-long/8-short base contracts")
    return contracts


def load_frozen_base_inputs(
    contracts: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Bind retained validation matrices to their same-run OOF identities."""

    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for side in SIDES:
        for month in MONTHS:
            shard = BASE_ROOT / f"shards/{side}_{month}"
            prediction_path = shard / "oof_predictions.parquet"
            feature_path = shard / f"{side}/month_{month}/validation_features.parquet"
            prediction = _normalise(pd.read_parquet(prediction_path))
            features = pd.read_parquet(feature_path)
            expected = list(contracts[side])
            if list(features.columns) != expected:
                raise ValueError(f"{side}/{month} retained feature order changed")
            if len(prediction) != len(features):
                raise ValueError(f"{side}/{month} feature/OOF row count differs")
            if not prediction["side_name"].eq(side).all():
                raise ValueError(f"{side}/{month} prediction shard has wrong side")
            if not prediction["__ts__"].dt.strftime("%Y_%m").eq(month).all():
                raise ValueError(f"{side}/{month} prediction shard has wrong month")
            renamed = features.rename(
                columns={name: f"base_input__{name}" for name in expected}
            )
            piece = pd.concat(
                [
                    prediction.loc[
                        :,
                        [
                            *IDENTITY,
                            "__decision_ts__",
                            "base_oof_score",
                            "__first_touch_target_soft__",
                            "__w__",
                            "__first_touch_capture_net__",
                            "execution_net_ev_12h",
                            "fold_id",
                            "fold_validation_start_utc",
                            "fold_validation_end_utc",
                            "base_label_resolution_utc",
                            "effective_label_resolution_utc",
                        ],
                    ].reset_index(drop=True),
                    renamed.reset_index(drop=True),
                ],
                axis=1,
            )
            pieces.append(piece)
            hashes[str(prediction_path)] = sha256(prediction_path)
            hashes[str(feature_path)] = sha256(feature_path)
    output = _normalise(pd.concat(pieces, ignore_index=True))
    output = output.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if len(output) != 509_868 or output["candidate_id"].duplicated().any():
        raise ValueError("full canonical retained base matrix identity contract fails")
    return output, hashes


def add_score_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add deterministic contemporaneous rank, scale and top-40 cutoff fields."""

    output = frame.copy()
    score = "base_oof_score"
    if not np.isfinite(pd.to_numeric(output[score], errors="coerce")).all():
        raise ValueError("base OOF score is non-finite")
    output = output.sort_values(
        ["__ts__", "side_name", score, "__symbol__", "candidate_id"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    side_group = output.groupby(["__ts__", "side_name"], sort=False, observed=True)
    output["base_rank_timestamp_side"] = side_group.cumcount().astype(np.int32) + 1
    output["base_group_rows_timestamp_side"] = (
        side_group[score].transform("size").astype(np.int32)
    )
    output["base_rank_pct_timestamp_side"] = (
        output["base_rank_timestamp_side"]
        / output["base_group_rows_timestamp_side"]
    )
    side_mean = side_group[score].transform("mean")
    side_std = side_group[score].transform("std").replace(0.0, np.nan)
    output["base_score_z_timestamp_side"] = (
        (output[score] - side_mean) / side_std
    ).fillna(0.0)
    output["__base_score_std_timestamp_side__"] = side_std
    output["base_rank_decile_timestamp_side"] = (
        output["base_rank_pct_timestamp_side"].mul(10.0).clip(0.0, 9.999999)
    ).astype(np.int8)
    keep = np.ceil(output["base_group_rows_timestamp_side"] * 0.40).astype(np.int32)
    output["selected_top40_timestamp_side"] = (
        output["base_rank_timestamp_side"] <= keep
    )
    cutoff_rows = output.loc[
        output["base_rank_timestamp_side"].eq(keep),
        ["__ts__", "side_name", score],
    ].rename(columns={score: "base_top40_cutoff_timestamp_side"})
    if cutoff_rows.duplicated(["__ts__", "side_name"]).any():
        raise ValueError("top-40 cutoff is not unique by timestamp/side")
    output = output.merge(
        cutoff_rows, on=["__ts__", "side_name"], how="left", validate="many_to_one"
    )
    output["base_margin_to_top40_cutoff"] = (
        output[score] - output["base_top40_cutoff_timestamp_side"]
    )
    output["base_margin_to_top40_cutoff_z"] = (
        output["base_margin_to_top40_cutoff"]
        / output.pop("__base_score_std_timestamp_side__")
    ).fillna(0.0)

    output = output.sort_values(
        ["__ts__", score, "side_name", "__symbol__", "candidate_id"],
        ascending=[True, False, True, True, True],
        kind="stable",
    )
    pooled = output.groupby("__ts__", sort=False, observed=True)
    output["base_rank_timestamp_global"] = pooled.cumcount().astype(np.int32) + 1
    output["base_group_rows_timestamp_global"] = (
        pooled[score].transform("size").astype(np.int32)
    )
    output["base_rank_pct_timestamp_global"] = (
        output["base_rank_timestamp_global"]
        / output["base_group_rows_timestamp_global"]
    )
    pooled_std = pooled[score].transform("std").replace(0.0, np.nan)
    output["base_score_z_timestamp_global"] = (
        (output[score] - pooled[score].transform("mean")) / pooled_std
    ).fillna(0.0)
    generated = (
        "base_rank_timestamp_side",
        "base_group_rows_timestamp_side",
        "base_rank_pct_timestamp_side",
        "base_score_z_timestamp_side",
        "base_rank_decile_timestamp_side",
        "selected_top40_timestamp_side",
        "base_top40_cutoff_timestamp_side",
        "base_margin_to_top40_cutoff",
        "base_margin_to_top40_cutoff_z",
        "base_rank_timestamp_global",
        "base_group_rows_timestamp_global",
        "base_rank_pct_timestamp_global",
        "base_score_z_timestamp_global",
    )
    return output.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True), generated


def load_preentry_regime_context() -> tuple[
    pd.DataFrame, tuple[str, ...], tuple[str, ...], dict[str, str]
]:
    static_features = tuple(dict.fromkeys(CORE_FEATURES + REGIME_SOURCE_FEATURES))
    forbidden = forbidden_feature_names(static_features)
    if forbidden:
        raise ValueError(f"configured forbidden pre-entry fields: {forbidden}")
    source_columns = [*IDENTITY, "__signal_ts__", "__decision_ts__", *static_features]
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    # January is included only to supply genuine past context to early
    # February rows.  It never enters the scored canonical population.
    for month in (1, 2, 3, 4):
        for side in SIDES:
            path = LABEL_ROOT / f"train_global_{side}_5_2025_{month:02d}.parquet"
            missing = sorted(set(source_columns).difference(pq.read_schema(path).names))
            if missing:
                raise ValueError(f"{path.name} lacks pre-entry fields: {missing}")
            source = _normalise(pd.read_parquet(path, columns=source_columns))
            if not source["side_name"].eq(side).all():
                raise ValueError(f"{path.name} contains the wrong side")
            if source["candidate_id"].duplicated().any():
                raise ValueError(f"{path.name} contains duplicate candidates")
            frames.append(source)
            hashes[str(path)] = sha256(path)
    source = pd.concat(frames, ignore_index=True)
    source, transition_features = add_causal_transition_deltas(source)
    selected = source.loc[
        source["__ts__"].ge(pd.Timestamp("2025-02-01", tz="UTC")),
        [*IDENTITY, "__signal_ts__", "__decision_ts__", *static_features, *transition_features],
    ].copy()
    if selected["candidate_id"].duplicated().any():
        raise ValueError("pre-entry context candidate IDs are not globally unique")
    return selected, static_features, transition_features, hashes


def _validate_exit_classes(panel: pd.DataFrame) -> None:
    classes = {"trailing", "timeout", "full_stop", "adverse_exit"}
    if set(panel["execution_exit_class"].astype(str).unique()) != classes:
        raise ValueError("canonical four-exit class set changed")
    flags = panel[
        [
            "exit_is_trailing",
            "exit_is_timeout",
            "exit_is_full_stop",
            "exit_is_adverse_exit",
        ]
    ].astype(int)
    if not flags.sum(axis=1).eq(1).all():
        raise ValueError("canonical exit classes are not mutually exclusive")


def _coverage(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(month=frame["__ts__"].dt.strftime("%Y-%m"))
        .groupby(["month", "side_name"], observed=True, sort=True)
        .agg(
            rows=("candidate_id", "size"),
            symbols=("__symbol__", "nunique"),
            mapped_eligible=("mapped_eligible", "sum"),
            opportunity_0bps=("opportunity_gross_above_cost_0bps", "mean"),
            opportunity_25bps=("opportunity_gross_above_cost_25bps", "mean"),
            mean_gross=("execution_gross_ev_12h", "mean"),
            mean_cost=("execution_cost_return", "mean"),
            mean_net=("execution_net_ev_12h", "mean"),
        )
        .reset_index()
    )


def _assert_same_identity(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    if len(left) != len(right) or set(left["candidate_id"]) != set(right["candidate_id"]):
        raise ValueError(f"{name} does not cover the exact canonical candidate set")


def _source_hash_records(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    contracts = _feature_contracts()
    base, base_hashes = load_frozen_base_inputs(contracts)
    base, score_context = add_score_context(base)

    ledger = _normalise(pd.read_parquet(LEDGER))
    if len(ledger) != 509_868 or ledger["candidate_id"].duplicated().any():
        raise ValueError("canonical exact-policy ledger identity contract fails")
    _assert_same_identity(base, ledger, "exact-policy ledger")
    ledger_values = ledger.loc[:, ["candidate_id", *LABEL_COLUMNS]].copy()
    overlap = sorted(
        set(ledger_values.columns).intersection(base.columns).difference({"candidate_id"})
    )
    # These columns are deliberately compared before retaining the ledger copy.
    merged_check = base.merge(
        ledger_values.loc[:, ["candidate_id", *overlap]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "__ledger"),
    )
    for column in overlap:
        other = f"{column}__ledger"
        if pd.api.types.is_datetime64_any_dtype(merged_check[column]):
            equal = pd.to_datetime(merged_check[column], utc=True).eq(
                pd.to_datetime(merged_check[other], utc=True)
            )
        elif pd.api.types.is_numeric_dtype(merged_check[column]):
            equal = np.isclose(
                pd.to_numeric(merged_check[column], errors="coerce"),
                pd.to_numeric(merged_check[other], errors="coerce"),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            equal = merged_check[column].astype(str).eq(merged_check[other].astype(str))
        if not np.asarray(equal).all():
            raise ValueError(f"retained OOF and exact-policy ledger disagree on {column}")
    ledger_values = ledger_values.drop(columns=overlap)
    panel = base.merge(ledger_values, on="candidate_id", how="left", validate="one_to_one")

    regime, static_features, transition_features, regime_hashes = (
        load_preentry_regime_context()
    )
    regime = regime.loc[regime["candidate_id"].isin(panel["candidate_id"])].copy()
    _assert_same_identity(panel, regime, "pre-entry regime context")
    regime = regime.rename(
        columns={
            "side_name": "__regime_side_name__",
            "__symbol__": "__regime_symbol__",
            "__ts__": "__regime_ts__",
            "__decision_ts__": "__regime_decision_ts__",
        }
    )
    panel = panel.merge(regime, on="candidate_id", how="left", validate="one_to_one")
    identity_ok = (
        panel["side_name"].eq(panel.pop("__regime_side_name__"))
        & panel["__ts__"].eq(panel.pop("__regime_ts__"))
        & panel["__decision_ts__"].eq(panel.pop("__regime_decision_ts__"))
        & panel["__symbol__"]
        .str.replace("_", "/", regex=False)
        .eq(panel.pop("__regime_symbol__"))
        & panel["__signal_ts__"].eq(panel["__ts__"])
    )
    if not identity_ok.all():
        raise ValueError("pre-entry regime context disagrees with canonical identity")

    causal = _normalise(
        pd.read_parquet(
            CAUSAL_MAP,
            columns=[
                *IDENTITY,
                *CAUSAL_MAP_FEATURES,
                *CAUSAL_MAP_CATEGORICAL_PROVENANCE,
            ],
        )
    )
    _assert_same_identity(panel, causal, "causal online map")
    causal_values = causal.loc[
        :,
        [
            "candidate_id",
            *CAUSAL_MAP_FEATURES,
            *CAUSAL_MAP_CATEGORICAL_PROVENANCE,
        ],
    ]
    panel = panel.merge(causal_values, on="candidate_id", how="left", validate="one_to_one")
    if not panel[
        "mapped_opportunity_gross_q50_fallback"
    ].eq(panel["mapped_opportunity_gross_q80_fallback"]).all():
        raise ValueError("q50/q80 causal fallback provenance differs")
    allowed_fallback = {"side_decile", "side", "global", "unavailable"}
    observed_fallback = set(panel["mapped_opportunity_gross_q50_fallback"].astype(str))
    if observed_fallback != allowed_fallback:
        raise ValueError(f"unexpected causal fallback levels: {observed_fallback}")
    for level, column in zip(
        ("side_decile", "side", "global", "unavailable"),
        CAUSAL_MAP_FALLBACK_INDICATORS,
    ):
        panel[column] = panel["mapped_opportunity_gross_q50_fallback"].eq(level)

    panel["opportunity_margin_0bps"] = (
        panel["execution_gross_ev_12h"] - panel["execution_cost_return"]
    )
    panel["opportunity_margin_25bps"] = panel["opportunity_margin_0bps"] - 0.0025
    if not np.allclose(
        panel["execution_net_ev_12h"],
        panel["opportunity_margin_0bps"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("gross-cost does not reconcile exactly to canonical net")
    _validate_exit_classes(panel)
    if not panel["execution_label_end_utc"].eq(
        panel["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("execution label is not decision + 12 hours")
    if not panel["execution_decision_utc"].eq(
        panel["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError("execution decision is not signal + 1 hour")
    if panel["candidate_id"].duplicated().any() or len(panel) != 509_868:
        raise ValueError("final panel identity changed")

    base_inputs = {
        side: [f"base_input__{name}" for name in features]
        for side, features in contracts.items()
    }
    feature_columns = tuple(
        dict.fromkeys(
            ["base_oof_score"]
            + list(score_context)
            + list(static_features)
            + list(transition_features)
            + list(CAUSAL_MAP_FEATURES)
            + list(CAUSAL_MAP_FALLBACK_INDICATORS)
            + base_inputs["long"]
            + base_inputs["short"]
        )
    )
    forbidden = forbidden_feature_names(
        [
            name
            for name in feature_columns
            if not name.startswith("mapped_")
        ]
    )
    if forbidden:
        raise ValueError(f"post-entry fields entered the feature surface: {forbidden}")
    if any(
        token in name.lower()
        for name in feature_columns
        for token in ("timing", "target_price", "wait_action", "exit_minute")
    ):
        raise ValueError("action-layer field entered the EV feature surface")

    panel = panel.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    coverage = _coverage(panel)
    quality = feature_quality(panel, feature_columns)
    side_base_quality = pd.concat(
        [
            feature_quality(
                panel.loc[panel["side_name"].eq(side)],
                tuple(base_inputs[side]),
            ).assign(side_name=side)
            for side in SIDES
        ],
        ignore_index=True,
    )
    core = tuple(CORE_FEATURES)
    composites = tuple(name for name in static_features if name not in core)
    output_parent = OUT.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_parent, prefix=f".{OUT.name}."))
    panel.to_parquet(temporary / "panel.parquet", index=False, compression="zstd")
    coverage.to_parquet(
        temporary / "coverage_by_side_month.parquet", index=False, compression="zstd"
    )
    quality.to_parquet(
        temporary / "feature_quality.parquet", index=False, compression="zstd"
    )
    side_base_quality.to_parquet(
        temporary / "base_feature_quality_by_side.parquet",
        index=False,
        compression="zstd",
    )
    source_hashes = {
        **base_hashes,
        **regime_hashes,
        **_source_hash_records([LEDGER, CAUSAL_MAP, BASE_ROOT / "manifest.json"]),
    }
    manifest = {
        "schema": "canonical_opportunity_payoff_trust_panel_v2",
        "status": "IMMUTABLE_FULL_CANONICAL_INPUT_AND_LABEL_PANEL",
        "rows": int(len(panel)),
        "identity_columns": list(IDENTITY),
        "identity_sha256": identity_sha256(panel),
        "months": sorted(panel["__ts__"].dt.strftime("%Y-%m").unique()),
        "sides": sorted(panel["side_name"].unique()),
        "feature_groups": {
            "base_oof_score": ["base_oof_score"],
            "base_side_selected_inputs": base_inputs,
            "candidate_score_context": list(score_context),
            "preentry_core_market_state": list(core),
            "preentry_regime_composites": list(composites),
            "past_only_transition_deltas": list(transition_features),
            "causal_online_mapping_context_diagnostic_only": [
                *CAUSAL_MAP_FEATURES,
                *CAUSAL_MAP_FALLBACK_INDICATORS,
            ],
            "causal_mapping_categorical_provenance_not_model_inputs": list(
                CAUSAL_MAP_CATEGORICAL_PROVENANCE
            ),
        },
        "target_groups": {
            "primary_direct_economics": [
                "execution_gross_ev_12h",
                "execution_cost_return",
                "execution_net_ev_12h",
            ],
            "opportunity": [
                "opportunity_gross_above_cost_0bps",
                "opportunity_gross_above_cost_25bps",
                "opportunity_margin_0bps",
                "opportunity_margin_25bps",
            ],
            "four_exit_payoff": [
                "execution_exit_class",
                "exit_is_trailing",
                "exit_is_timeout",
                "exit_is_full_stop",
                "exit_is_adverse_exit",
            ],
            "diagnostic_outcomes_not_features": [
                "execution_mfe_return_12h",
                "execution_mae_return_12h",
                "execution_exit_minute",
            ],
        },
        "contracts": {
            "decision": "signal timestamp + 1 hour",
            "execution_label": "decision timestamp + 12 hours",
            "base_oof": "retained exact same-run validation matrices; frozen 31-long/8-short contracts",
            "transition": "same side/symbol signal-time value minus exact t-3h or t-12h value; January source rows supply early-February history",
            "cost": "current-spread counterfactual exact-policy cost; historical observed spread is unavailable; subtract exactly once",
            "selection": "evaluation must use one pooled global top-k with candidate_id tie-break; never per timestamp or side",
            "causal_online_mapping": "each daily snapshot uses only previously resolved labels; because April updates with earlier April outcomes, this group is diagnostic in a static untouched-April comparison unless frozen at April 1",
            "causal_fallback_provenance": "raw fallback strings are audit-only; model inputs are explicit Boolean level indicators",
            "timing_mae_target_price_wait": "excluded from EV input groups and retained downstream in the separate action layer",
            "trust_target": "must be constructed only from nested OOF expected-EV predictions; no in-sample trust target is materialized",
        },
        "validation": {
            "one_to_one_full_canonical_identity": True,
            "gross_minus_cost_equals_net": True,
            "four_exit_flags_sum_to_one": True,
            "no_action_layer_fields_in_ev_features": True,
            "coverage_by_side_month": coverage.to_dict(orient="records"),
        },
        "sources_sha256": source_hashes,
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n"
    )
    os.replace(temporary, OUT)
    print(
        json.dumps(
            {
                "output": str(OUT),
                "rows": len(panel),
                "features": len(feature_columns),
                "identity_sha256": manifest["identity_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
