#!/usr/bin/env python3
"""Offline :00-only strict-R3 two-stage residual/consensus research runner.

This producer intentionally begins from the immutable current-v5 B0 score
ledger.  It never imports a live bundle, writes an inference artifact, sends
orders, or changes the canonical contract.  The first layer is the frozen
current ten-head consensus (C1).  Layer two is trained only on a *cross-fitted*
Delta-1 correction of C1, so no L2 learner consumes an in-sample L1 prediction.

The runner is deliberately conservative about chronology:

* every supervised fit excludes the preceding 28 calendar days;
* labels must be resolved before that reserve starts;
* all score references are fitted on the training fold, never on held rows;
* policy labels are joined after target-free held predictions are finalised;
* only :00 control blocks are consumed.

It provides the mechanically reproducible Stage-0/Stage-2 substrate required
by the base-recall/residual/consensus research plan.  MC1, BCF mapping,
admission, and portfolio comparison are intentionally separate later stages.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.research.base_recall import (  # noqa: E402
    TimestampRouteSpec,
    timestamp_local_route,
)
from extreme_price_movements.research.consensus_ensemble import (  # noqa: E402
    agreement_statistics,
)
from extreme_price_movements.research.hierarchical_residual import (  # noqa: E402
    Delta1MapSpec,
    fit_delta1_map,
    ordinal_residual_grade,
)
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    ConsensusHeadSpec,
    _fit_consensus_head,
    load_conditional_consensus_contract,
    score_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    ScoreReference,
    _fit_medians,
    _numeric_matrix,
)


DEFAULT_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_20260809_v1/prequential_stack_ledger.parquet"
)
DEFAULT_B0_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1"
)
DEFAULT_CONTROL_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_"
    "reconstruction_2025_2026_20260816_v4"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_base_recall_residual2_consensus_research_"
    "20260822_v3"
)

RESERVE_DAYS = 28
BASE_ROUTE = 0.30
HISTORY_START = pd.Timestamp("2024-04-01T00:00:00Z")
LAYER2_EDGES = (-100.0, -25.0, 25.0, 100.0)
SEED = 1729
DELTA1_CORRECTION_SHRINKS = (.25, .50, .75, 1.00)
DELTA1_CONDITIONAL_MAP_SHRINKAGE = .50
LAYER2_SCALAR_FIELDS = (
    "base_score",
    "base_rank42",
    "base_anchor_bps",
    "conditional_consensus_rank",
    "upstream",
    "delta1_bps",
    "a1_bps",
)
PERIODS = {
    "development_2025q1q3": ("2025-01-01", "2025-10-01"),
    "frozen_holdout_2025q4": ("2025-10-01", "2026-01-01"),
    "frozen_oos_2026jan_jul": ("2026-01-01", "2026-08-01"),
}


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_hourly_phase_zero(frame: pd.DataFrame, *, source_name: str) -> None:
    """Fail closed unless every research decision is an exact UTC :00 bar."""

    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    nonzero = (timestamp.dt.minute != 0) | (timestamp.dt.second != 0) | (timestamp.dt.microsecond != 0)
    if nonzero.any():
        sample = timestamp.loc[nonzero].head(5).astype(str).tolist()
        raise AssertionError(f"{source_name} contains non-:00 decision timestamps: {sample}")


def _feature_contract(control_root: Path) -> tuple[str, ...]:
    import joblib

    paths = sorted(control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    if not paths:
        raise FileNotFoundError("no frozen current-v5 upstream bundles")
    contracts = {tuple(joblib.load(path).base_fields) for path in paths}
    if len(contracts) != 1:
        raise AssertionError("frozen B0 control does not have one base feature contract")
    fields = next(iter(contracts))
    if len(fields) != 120:
        raise AssertionError(f"expected frozen 120-field contract, found {len(fields)}")
    return fields


def _block_cutoff(name: str) -> pd.Timestamp:
    match = re.fullmatch(r"block=(\d{8}T\d{6}Z)(?:_finalcoverage)?", str(name))
    if match is None:
        raise ValueError(f"invalid control block name: {name!r}")
    return pd.Timestamp(match.group(1))


def _read_source(source: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "side_name", "policy_path_valid",
        "policy_net_bps", "policy_label_available_ts", "stack_is_prequential",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_upstream", *fields,
    ]
    frame = pd.read_parquet(source, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    _require_exact_hourly_phase_zero(frame, source_name="prequential source")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("prequential source has duplicate candidate identities")
    if not frame["stack_is_prequential"].fillna(False).astype(bool).all():
        raise AssertionError("source contains non-prequential base predictions")
    return frame


def _read_b0(b0_root: Path) -> pd.DataFrame:
    path = b0_root / "outcome_joined_recall_ledger.parquet"
    columns = [
        "candidate_id", "__decision_ts__", "side_name", "control_block", "base_score",
        "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
        "base_route_timestamp_top30", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    _require_exact_hourly_phase_zero(frame, source_name="frozen B0 ledger")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("B0 ledger has duplicate candidate identities")
    if (frame["__decision_ts__"] < pd.Timestamp("2025-01-01T00:00:00Z")).any():
        raise AssertionError("B0 current control unexpectedly includes pre-2025 candidates")
    return frame


def build_layer1_ledger(source: pd.DataFrame, b0: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    """Stitch pre-2025 historical OOF scores to the immutable B0 control.

    Source rows after 2025-01-01 are intentionally excluded unless their
    identity is present in B0.  This prevents a later source expansion from
    changing the frozen control universe.
    """

    start = pd.Timestamp("2025-01-01T00:00:00Z")
    old = source.loc[source["__decision_ts__"].lt(start)].copy()
    old = old.rename(columns={
        "prequential_base_score": "base_score",
        "prequential_base_rank42": "base_rank42",
        "prequential_base_anchor_bps": "base_anchor_bps",
        "prequential_consensus_rank": "conditional_consensus_rank",
        "prequential_upstream": "upstream",
    })
    old["control_block"] = "historical_prequential"
    old["base_route_timestamp_top30"] = timestamp_local_route(
        old.loc[:, ["candidate_id", "__decision_ts__", "base_score"]],
        "base_score", spec=TimestampRouteSpec(fraction=BASE_ROUTE),
    )
    old.loc[~old["base_route_timestamp_top30"], "conditional_consensus_rank"] = np.nan
    old["layer1_source"] = "historical_prequential"
    old["layer1_prediction_is_strict_oof"] = True

    b0_features = b0.loc[:, ["candidate_id"]].merge(
        source.loc[:, ["candidate_id", *fields]], on="candidate_id", how="left", validate="one_to_one",
    )
    if b0_features.loc[:, list(fields)].isna().all(axis=None):
        raise AssertionError("B0 candidate identities cannot be joined to frozen base features")
    current = b0.merge(b0_features, on="candidate_id", how="left", validate="one_to_one")
    if current.loc[:, list(fields)].isna().all(axis=None):
        raise AssertionError("B0 base feature join is empty")
    current["layer1_source"] = "frozen_B0_control"
    current["layer1_prediction_is_strict_oof"] = True
    result_columns = [
        "candidate_id", "__decision_ts__", "side_name", "control_block", "base_score",
        "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
        "base_route_timestamp_top30", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts", "layer1_source", "layer1_prediction_is_strict_oof", *fields,
    ]
    result = pd.concat([old.loc[:, result_columns], current.loc[:, result_columns]], ignore_index=True)
    result = result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("stitched layer-one ledger has duplicate candidate identities")
    if not result["layer1_prediction_is_strict_oof"].all():
        raise AssertionError("layer-one ledger contains non-OOF predictions")
    return result


def _score_schedule(b0: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return non-overlapping score windows; the B0 timeline owns 2025+."""

    first_b0 = pd.Timestamp("2025-01-01T00:00:00Z")
    starts = list(pd.date_range(HISTORY_START, first_b0, freq="28D", inclusive="left"))
    windows: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else first_b0
        windows.append((_utc(start), _utc(end), "historical_prequential"))
    meta = (
        b0.groupby("control_block", sort=True)["__decision_ts__"]
        .agg(["min", "max"]).reset_index().sort_values("min", kind="stable")
    )
    for row in meta.itertuples(index=False):
        start = _utc(row.min)
        end = _utc(row.max) + pd.Timedelta(hours=1)
        windows.append((start, end, str(row.control_block)))
    return windows


def _delta1_arm_name(map_family: str, correction_shrinkage: float) -> str:
    return f"{map_family}_s{int(round(100.0 * correction_shrinkage)):03d}"


def _select_delta1_arm(
    output: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Select a Delta-1 arm from development-only strict-OOF residuals."""

    joined = output.merge(
        ledger.loc[:, ["candidate_id", "policy_net_bps"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    development = joined.loc[
        joined["__decision_ts__"].ge(_utc(PERIODS["development_2025q1q3"][0]))
        & joined["__decision_ts__"].lt(_utc(PERIODS["development_2025q1q3"][1]))
    ].copy()
    rows: list[dict[str, object]] = []
    for field in sorted(column for column in output if column.startswith("a1__")):
        residual = (
            pd.to_numeric(development["policy_net_bps"], errors="coerce")
            - pd.to_numeric(development[field], errors="coerce")
        )
        valid = residual.notna()
        rows.append({
            "arm": field.removeprefix("a1__"),
            "development_rows": int(valid.sum()),
            "development_mean_abs_r2_bps": float(residual.loc[valid].abs().mean()) if valid.any() else float("inf"),
            "development_median_abs_r2_bps": float(residual.loc[valid].abs().median()) if valid.any() else float("inf"),
        })
    metrics = pd.DataFrame(rows)
    if metrics.empty or not np.isfinite(metrics["development_mean_abs_r2_bps"]).any():
        raise AssertionError("Delta-1 selection has no finite strict-OOF development residual")
    selected = metrics.sort_values(
        ["development_mean_abs_r2_bps", "development_median_abs_r2_bps", "arm"],
        kind="stable",
    ).iloc[0]
    metrics["selected_from_development_only"] = metrics["arm"].eq(selected["arm"])
    metrics["selection_period"] = "development_2025q1q3"
    return str(selected["arm"]), metrics


def build_delta1_oof(
    ledger: pd.DataFrame,
    schedule: Iterable[tuple[pd.Timestamp, pd.Timestamp, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score the declared Delta-1 map grid, selecting only on 2025 development."""

    output = ledger.loc[:, ["candidate_id", "__decision_ts__", "control_block"]].copy()
    output["delta1_prediction_is_strict_oof"] = False
    arm_columns = [
        _delta1_arm_name(family, shrink)
        for family in ("M1_global", "M1_base_conditioned")
        for shrink in DELTA1_CORRECTION_SHRINKS
    ]
    for arm in arm_columns:
        output[f"delta1__{arm}"] = np.nan
        output[f"a1__{arm}"] = np.nan
    audits: list[dict[str, object]] = []
    for cutoff, end, name in schedule:
        reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
        held_mask = ledger["__decision_ts__"].ge(cutoff) & ledger["__decision_ts__"].lt(end)
        if not held_mask.any():
            continue
        try:
            global_map = fit_delta1_map(
                ledger,
                cutoff=reserve_start,
                spec=Delta1MapSpec(
                    consensus_column="conditional_consensus_rank",
                    anchor_column="base_anchor_bps",
                    outcome_column="policy_net_bps",
                    minimum_bin_support=200,
                    clip_bps=300.0,
                    shrinkage=0.0,
                ),
                layer1_oof_column="layer1_prediction_is_strict_oof",
            )
            conditional_map = fit_delta1_map(
                ledger,
                cutoff=reserve_start,
                spec=Delta1MapSpec(
                    consensus_column="conditional_consensus_rank",
                    anchor_column="base_anchor_bps",
                    outcome_column="policy_net_bps",
                    minimum_bin_support=200,
                    clip_bps=300.0,
                    shrinkage=DELTA1_CONDITIONAL_MAP_SHRINKAGE,
                ),
                layer1_oof_column="layer1_prediction_is_strict_oof",
            )
        except ValueError as exc:
            audits.append({
                "window": name, "cutoff": cutoff.isoformat(), "reserve_start": reserve_start.isoformat(),
                "held_rows": int(held_mask.sum()), "status": "insufficient_support", "detail": str(exc),
            })
            continue
        held = ledger.loc[held_mask]
        anchor = pd.to_numeric(held["base_anchor_bps"], errors="coerce").to_numpy(float)
        predictions = {
            "M1_global": global_map.predict(held["conditional_consensus_rank"], held["base_anchor_bps"]),
            "M1_base_conditioned": conditional_map.predict(held["conditional_consensus_rank"], held["base_anchor_bps"]),
        }
        pos = np.flatnonzero(held_mask.to_numpy())
        for family, prediction in predictions.items():
            for shrinkage in DELTA1_CORRECTION_SHRINKS:
                arm = _delta1_arm_name(family, shrinkage)
                correction = float(shrinkage) * prediction
                output.loc[pos, f"delta1__{arm}"] = correction
                output.loc[pos, f"a1__{arm}"] = anchor + correction
        output.loc[pos, "delta1_prediction_is_strict_oof"] = np.isfinite(predictions["M1_global"])
        audits.append({
            "window": name, "cutoff": cutoff.isoformat(), "reserve_start": reserve_start.isoformat(),
            "held_rows": int(held_mask.sum()), "status": "scored", "detail": "",
            "global_support": global_map.global_support,
            "anchor_bin_support": json.dumps(conditional_map.conditional_support),
            "anchor_edges": json.dumps(conditional_map.anchor_edges.tolist()),
            "map_families": "M1_global,M1_base_conditioned",
            "conditional_map_shrinkage_to_global": DELTA1_CONDITIONAL_MAP_SHRINKAGE,
            "correction_shrinkages": json.dumps(DELTA1_CORRECTION_SHRINKS),
        })
    selected_arm, selection = _select_delta1_arm(output, ledger)
    output["delta1_bps"] = output[f"delta1__{selected_arm}"]
    output["a1_bps"] = output[f"a1__{selected_arm}"]
    return output, pd.DataFrame(audits), selection


def _layer2_specs(base_fields: tuple[str, ...]) -> tuple[ConsensusHeadSpec, ...]:
    base = load_conditional_consensus_contract(base_fields, side="long")
    # Six predeclared diversity heads: retain the first six frozen field/query/
    # weight layouts, but replace only the r2 ordinal boundaries.
    return tuple(
        ConsensusHeadSpec(
            name=f"r2_{spec.name}", cap=spec.cap, weight_mode=spec.weight_mode,
            query=spec.query, fields=spec.fields, target_edges_bps=LAYER2_EDGES,
            params=dict(spec.params),
        )
        for spec in base[:6]
    )


def _layer2_model_fields(base_fields: tuple[str, ...]) -> tuple[str, ...]:
    """Return the fixed R2/R3 feature contract for one residual coordinate.

    These are all decision-time or strict-OOF upstream values.  The original
    120-field Geometry/K9 input contract remains an ordered prefix; this
    separate residual learner neither changes nor refits Geometry/K9.
    """

    fields = tuple(dict.fromkeys((*base_fields, *LAYER2_SCALAR_FIELDS)))
    if len(fields) != len(base_fields) + len(LAYER2_SCALAR_FIELDS):
        raise AssertionError("layer-two scalar fields unexpectedly overlap frozen base fields")
    return fields


def _strict_train(frame: pd.DataFrame, reserve_start: pd.Timestamp) -> pd.DataFrame:
    valid = (
        frame["__decision_ts__"].lt(reserve_start)
        & frame["policy_label_available_ts"].lt(reserve_start)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
        & frame["delta1_prediction_is_strict_oof"].fillna(False).astype(bool)
        & pd.to_numeric(frame["a1_bps"], errors="coerce").notna()
    )
    return frame.loc[valid].copy()


def _classifier_params() -> dict[str, object]:
    return {
        "objective": "multiclass", "num_class": 3,
        "n_estimators": 180, "learning_rate": .05,
        "max_depth": 4, "num_leaves": 15, "min_child_samples": 1000,
        "subsample": .8, "subsample_freq": 1, "colsample_bytree": .8,
        "reg_lambda": 8.0, "random_state": SEED + 700, "n_jobs": 1,
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
    }


def _r2_direction_class(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return the declared three-class residual-direction target.

    The residual-D2 arm must answer whether the first-layer correction is
    materially adverse, unresolved, or materially favorable.  A binary
    ``r2 >= 0`` label would silently change that target and overweight tiny
    residual noise.
    """

    residual = np.asarray(values, dtype=float)
    result = np.full(len(residual), -1, dtype=np.int8)
    finite = np.isfinite(residual)
    result[finite] = 1
    result[finite & (residual <= -100.0)] = 0
    result[finite & (residual >= 100.0)] = 2
    return result


@dataclass(frozen=True)
class _RegressorBundle:
    medians: np.ndarray
    model: LGBMRegressor
    reference: ScoreReference

    def predict_rank(self, frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.predict(_numeric_matrix(frame, fields, self.medians))
        return raw.astype(np.float32), self.reference.cdf(raw).astype(np.float32)


def _day_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each resolved training day equal aggregate R3 loss authority.

    The weighting is constructed only from the fit population's decision
    timestamps.  It is therefore chronology-safe and avoids high-activity days
    silently dominating the absolute-residual calibration target.
    """

    if frame.empty:
        raise ValueError("cannot calculate day-balanced weights for an empty frame")
    day = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise").dt.normalize()
    daily_rows = pd.Series(1.0, index=frame.index).groupby(day, sort=False).transform("sum")
    weights = 1.0 / daily_rows.to_numpy(float)
    return weights / float(np.mean(weights))


def _fit_r3_huber(train: pd.DataFrame, fields: tuple[str, ...]) -> _RegressorBundle:
    values = np.clip(train["r2_bps"].to_numpy(float), -500.0, 500.0)
    medians = _fit_medians(train, fields)
    model = LGBMRegressor(
        objective="huber", alpha=.90,
        n_estimators=180, learning_rate=.05,
        max_depth=3, num_leaves=15, min_child_samples=1000,
        subsample=.8, subsample_freq=1, colsample_bytree=.8,
        reg_lambda=8.0, random_state=SEED + 701, n_jobs=1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    ).fit(
        _numeric_matrix(train, fields, medians), values,
        sample_weight=_day_balanced_weights(train),
    )
    raw = model.predict(_numeric_matrix(train, fields, medians))
    return _RegressorBundle(medians, model, ScoreReference.fit(raw, source="r2_huber_training_reference"))


def build_layer2_oof(
    ledger: pd.DataFrame,
    delta: pd.DataFrame,
    b0: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    checkpoint_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit R1/R2/R3 strictly before each B0 window and score its held rows."""

    work = ledger.merge(delta.loc[:, ["candidate_id", "delta1_bps", "a1_bps", "delta1_prediction_is_strict_oof"]], on="candidate_id", how="left", validate="one_to_one")
    work["r2_bps"] = pd.to_numeric(work["policy_net_bps"], errors="coerce") - pd.to_numeric(work["a1_bps"], errors="coerce")
    control = b0.loc[:, ["candidate_id", "control_block"]].copy()
    block_meta = control.groupby("control_block", sort=True).size().index.tolist()
    specs = _layer2_specs(fields)
    model_fields = _layer2_model_fields(fields)
    rows: list[pd.DataFrame | Path] = []
    audits: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    for block in block_meta:
        checkpoint = None
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_root / f"{block}.parquet"
            if checkpoint.is_file():
                rows.append(checkpoint)
                audits.append({"block": block, "status": "resumed_checkpoint"})
                continue
        held_ids = control.loc[control["control_block"].eq(block), "candidate_id"]
        held = work.loc[work["candidate_id"].isin(held_ids)].copy()
        if len(held) != len(held_ids):
            raise AssertionError(f"layer2 feature ledger lost B0 identities for {block}")
        cutoff = _block_cutoff(block)
        reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
        train = _strict_train(work, reserve_start)
        train["r2_bps"] = pd.to_numeric(train["policy_net_bps"], errors="coerce") - pd.to_numeric(train["a1_bps"], errors="coerce")
        grade = ordinal_residual_grade(train["r2_bps"], edges_bps=LAYER2_EDGES)
        train = train.loc[grade >= 0].copy()
        grade = ordinal_residual_grade(train["r2_bps"], edges_bps=LAYER2_EDGES)
        routed = held["base_route_timestamp_top30"].fillna(False).astype(bool).to_numpy()
        output = held.loc[:, [
            "candidate_id", "__decision_ts__", "control_block", "base_score", "base_rank42",
            "base_anchor_bps", "conditional_consensus_rank", "upstream", "delta1_bps", "a1_bps",
            "base_route_timestamp_top30",
        ]].copy()
        output["layer2_prediction_is_strict_oof"] = False
        if len(train) < 10_000 or grade.max(initial=-1) < 1:
            audits.append({"block": block, "cutoff": cutoff.isoformat(), "reserve_start": reserve_start.isoformat(), "status": "insufficient_training_support", "train_rows": int(len(train)), "held_rows": int(len(held))})
            if checkpoint is not None:
                output.to_parquet(checkpoint, index=False, compression="zstd")
                rows.append(checkpoint)
            else:
                rows.append(output)
            continue
        head_ranks: dict[str, np.ndarray] = {}
        for index, spec in enumerate(specs):
            head = _fit_consensus_head(train.loc[:, ["candidate_id", "__decision_ts__", "side_name", *spec.fields]], grade, spec, seed=SEED + 800 + index)
            raw = np.full(len(held), np.nan, dtype=np.float32)
            rank = np.full(len(held), np.nan, dtype=np.float32)
            if routed.any():
                raw[routed], rank[routed] = head.predict_rank(held.loc[routed, list(spec.fields)])
            output[f"layer2_head__{spec.name}__raw"] = raw
            output[f"layer2_head__{spec.name}__rank"] = rank
            head_ranks[spec.name] = rank
            head_rows.append({"block": block, "head": spec.name, "query": spec.query, "weight_mode": spec.weight_mode, "field_count": len(spec.fields), "train_rows": len(train), "rank_reference_rows": len(head.score_reference.sorted_values)})
            del head
            gc.collect()
        matrix = np.column_stack(list(head_ranks.values()))
        output["layer2_consensus_rank"] = np.nanmedian(matrix, axis=1)
        stats = agreement_statistics(head_ranks)
        for name, value in stats.items():
            output[f"layer2_agreement__{name}"] = value
        # R2: declared three-class residual-direction classifier, trained
        # under the same strict reserve.  It answers whether layer one
        # remains materially under- or over-confident, not simply whether
        # microscopic residual noise is positive.
        y = _r2_direction_class(train["r2_bps"])
        if (y < 0).any() or np.unique(y).size < 3:
            raise AssertionError("R2 training requires all three finite residual-direction classes")
        medians = _fit_medians(train, model_fields)
        classifier = LGBMClassifier(**_classifier_params()).fit(_numeric_matrix(train, model_fields, medians), y)
        lookup = {int(label): index for index, label in enumerate(classifier.classes_)}
        train_proba = classifier.predict_proba(_numeric_matrix(train, model_fields, medians))
        train_raw = train_proba[:, lookup[2]] - .5 * train_proba[:, lookup[0]]
        reference = ScoreReference.fit(train_raw, source="r2_D2_three_class_training_reference")
        r2_raw = np.full(len(held), np.nan, dtype=np.float32)
        r2_rank = np.full(len(held), np.nan, dtype=np.float32)
        r2_adverse = np.full(len(held), np.nan, dtype=np.float32)
        r2_weak = np.full(len(held), np.nan, dtype=np.float32)
        r2_clear = np.full(len(held), np.nan, dtype=np.float32)
        if routed.any():
            held_proba = classifier.predict_proba(_numeric_matrix(held.loc[routed], model_fields, medians))
            r2_adverse[routed] = held_proba[:, lookup[0]]
            r2_weak[routed] = held_proba[:, lookup[1]]
            r2_clear[routed] = held_proba[:, lookup[2]]
            r2_raw[routed] = r2_clear[routed] - .5 * r2_adverse[routed]
            r2_rank[routed] = reference.cdf(r2_raw[routed])
        output["r2_d2_p_adverse"] = r2_adverse
        output["r2_d2_p_weak"] = r2_weak
        output["r2_d2_p_clear"] = r2_clear
        output["r2_d2_raw"] = r2_raw
        output["r2_d2_rank"] = r2_rank
        # R3: robust clipped Huber residual.  It supplies a deliberately
        # different model class for the Stage-2 negative control.
        huber = _fit_r3_huber(train, model_fields)
        r3_raw = np.full(len(held), np.nan, dtype=np.float32)
        r3_rank = np.full(len(held), np.nan, dtype=np.float32)
        if routed.any():
            r3_raw[routed], r3_rank[routed] = huber.predict_rank(held.loc[routed], model_fields)
        output["r3_huber_raw"] = r3_raw
        output["r3_huber_rank"] = r3_rank
        output["layer2_prediction_is_strict_oof"] = routed
        # Initial Stage-2 evidence is deliberately late fusion only: these
        # outputs are passed to MC1 after its own strict OOF fit.  Bounded
        # authority over upstream/final score is a separate follow-up arm and
        # may run only if the late-fusion candidate advances.
        if checkpoint is not None:
            output.to_parquet(checkpoint, index=False, compression="zstd")
            rows.append(checkpoint)
        else:
            rows.append(output)
        audits.append({
            "block": block, "cutoff": cutoff.isoformat(), "reserve_start": reserve_start.isoformat(),
            "status": "scored", "train_rows": int(len(train)), "held_rows": int(len(held)),
            "routed_rows": int(routed.sum()), "head_count": len(specs),
        })
    materialised = [pd.read_parquet(row) if isinstance(row, Path) else row for row in rows]
    return pd.concat(materialised, ignore_index=True), pd.DataFrame(audits), pd.DataFrame(head_rows)


def _rank_ic(frame: pd.DataFrame, score: str) -> float:
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
        & pd.to_numeric(frame[score], errors="coerce").notna(),
        ["__decision_ts__", score, "policy_net_bps"],
    ].copy()
    if valid.empty:
        return float("nan")
    valid["x"] = valid.groupby("__decision_ts__", sort=False)[score].rank(method="average")
    valid["y"] = valid.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(method="average")
    corr = valid.groupby("__decision_ts__", sort=False)[["x", "y"]].corr().iloc[0::2, -1]
    return float(corr.mean()) if len(corr) else float("nan")


def residual_metrics(layer2: pd.DataFrame, label_ledger: pd.DataFrame) -> pd.DataFrame:
    """Outcome diagnostics; scores were already frozen before this join."""

    labels = label_ledger.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]]
    work = layer2.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    arms = ["upstream", "layer2_consensus_rank", "r2_d2_rank", "r3_huber_rank"]
    result: list[dict[str, object]] = []
    for arm in arms:
        for period, (start, end) in PERIODS.items():
            subset = work.loc[
                work["__decision_ts__"].ge(_utc(start)) & work["__decision_ts__"].lt(_utc(end))
                & work["base_route_timestamp_top30"].fillna(False).astype(bool)
            ].copy()
            valid = subset["policy_path_valid"].fillna(False).astype(bool) & subset["policy_net_bps"].notna()
            rank = subset.groupby("__decision_ts__", sort=False)[arm].rank(method="first", ascending=False)
            count = subset.groupby("__decision_ts__", sort=False)[arm].transform("count")
            for fraction in (.10, .20, .30):
                select = rank.le(np.ceil(fraction * count)) & valid
                result.append({
                    "arm": arm, "period": period, "tail_of_base_route": fraction,
                    "candidate_rows": int(len(subset)), "selected_rows": int(select.sum()),
                    "policy_net_mean_bps": float(subset.loc[select, "policy_net_bps"].mean()) if select.any() else float("nan"),
                    "policy_net_median_bps": float(subset.loc[select, "policy_net_bps"].median()) if select.any() else float("nan"),
                    "within_timestamp_policy_rank_ic": _rank_ic(subset, arm),
                })
    return pd.DataFrame(result)


def _control_parity(b0_root: Path, control_root: Path) -> dict[str, object]:
    audit_path = b0_root / "b0_block_parity.parquet"
    audit = pd.read_parquet(audit_path)
    fields = ("base_score", "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream")
    passed = bool(all(audit[f"{field}_parity"].fillna(False).all() for field in fields))
    return {
        "status": "passed" if passed else "failed",
        "control_root": str(control_root),
        "score_blocks": int(len(audit)),
        "fields": fields,
        "maximum_abs_delta": {field: float(audit[f"{field}_max_abs_delta"].max()) for field in fields},
        "individual_head_audit": (
            "Historical score blocks persisted only aggregate consensus ranks, not individual head outputs. "
            "The immutable head models/contract are therefore re-used deterministically and their median is "
            "validated through aggregate C1 parity; separate historical per-head-output parity is unavailable."
        ),
    }


def _target_free_block(source: Path, bundle: object) -> pd.DataFrame:
    cutoff = _utc(bundle.cutoff)
    end = _utc(bundle.end_exclusive)
    columns = ["candidate_id", "__decision_ts__", "side_name", *bundle.base_fields]
    frame = pd.read_parquet(
        source,
        columns=columns,
        filters=[("__decision_ts__", ">=", cutoff - pd.Timedelta(days=28)), ("__decision_ts__", "<", end)],
    )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    return frame


def reconstruct_individual_head_parity(
    *, source: Path, b0_root: Path, control_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct frozen individual C1 head values from immutable bundles.

    Legacy score blocks did not persist individual values.  This is the
    strongest available Stage-0 receipt: score the original immutable models
    over the original target-free population, preserve all ten reconstructed
    ranks, and require their median to match both the re-scored C1 value and
    the separately stored aggregate C1 exactly.
    """

    b0 = pd.read_parquet(
        b0_root / "b0_target_free_reconstruction.parquet",
        columns=["candidate_id", "control_block", "conditional_consensus_rank"],
    )
    rows: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for path in sorted(control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        block = path.parents[1].name
        bundle = joblib.load(path)
        frame = _target_free_block(source, bundle)
        scored = score_monthly_upstream_bundle(
            bundle, frame, allow_prior_reference=True,
            prior_reference_start=_utc(bundle.cutoff) - pd.Timedelta(days=28),
            route_top_fraction=BASE_ROUTE,
        )
        held_all = scored.loc[scored["__decision_ts__"].ge(_utc(bundle.cutoff))].copy()
        # The terminal B0 block is intentionally a partial-coverage window.
        # Its frozen candidate IDs—not the full source tail after the declared
        # evaluation end—define the control population.  Restrict only after
        # target-free scoring, so timestamp routing and score references keep
        # exactly the source semantics used by the stored control.
        expected = b0.loc[
            b0["control_block"].eq(block),
            ["candidate_id", "conditional_consensus_rank"],
        ]
        expected_ids = set(expected["candidate_id"])
        held = held_all.loc[held_all["candidate_id"].isin(expected_ids)].copy()
        if len(held) != len(expected) or held["candidate_id"].duplicated().any():
            raise AssertionError(
                f"{block} frozen B0 IDs are not a one-to-one subset of the target-free score "
                f"population: held={len(held)}, expected={len(expected)}"
            )
        head_columns = sorted(
            field for field in held.columns
            if field.startswith("conditional_head__") and field.endswith("__rank")
        )
        if len(head_columns) != 10:
            raise AssertionError(f"{block} expected ten frozen C1 heads, found {len(head_columns)}")
        reconstructed = np.full(len(held), np.nan, dtype=float)
        route = held["base_route_timestamp_top30"].fillna(False).to_numpy(bool)
        if route.any():
            # Preserve the frozen scorer's reduction dtype.  The ten head
            # ranks are float32 and the scorer takes their median before
            # assigning it into a float64 output vector.  Upcasting *before*
            # the reduction changes some tie-adjacent medians by one float32
            # ULP even though the model output is identical.
            reconstructed[route] = np.nanmedian(
                held.loc[route, head_columns].to_numpy(dtype=np.float32, copy=False),
                axis=1,
            )
        if not np.allclose(
            held["conditional_consensus_rank"].to_numpy(float), reconstructed,
            equal_nan=True, rtol=0.0, atol=0.0,
        ):
            raise AssertionError(f"{block} reconstructed individual C1 median differs from scorer aggregate")
        merged = held.loc[:, ["candidate_id", "conditional_consensus_rank", *head_columns]].merge(
            expected, on="candidate_id", how="outer", suffixes=("__reconstructed", "__stored"),
            indicator=True, validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            raise AssertionError(f"{block} head parity reconstruction changed identities: {merged['_merge'].value_counts().to_dict()}")
        left = merged["conditional_consensus_rank__reconstructed"].to_numpy(float)
        right = merged["conditional_consensus_rank__stored"].to_numpy(float)
        if not np.allclose(left, right, equal_nan=True, rtol=1e-4, atol=1e-8):
            raise AssertionError(f"{block} reconstructed C1 differs from stored C1")
        kept = held.loc[:, ["candidate_id", "__decision_ts__", "conditional_consensus_rank", *head_columns]].copy()
        kept["control_block"] = block
        rows.append(kept)
        finite = np.abs(left[np.isfinite(left)] - right[np.isfinite(right)])
        audit.append({
            "control_block": block, "held_rows": int(len(held)), "head_count": len(head_columns),
            "target_free_tail_rows_excluded_by_frozen_b0_population": int(len(held_all) - len(held)),
            "identity_parity": True, "aggregate_c1_parity": True,
            "max_abs_c1_delta": float(finite.max()) if len(finite) else 0.0,
            "reconstruction_source": "immutable_monthly_upstream_bundle_plus_target_free_source",
        })
    output = pd.concat(rows, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("individual C1 reconstruction overlaps candidate identities")
    return output, pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=("control", "residual"), default="residual")
    parser.add_argument(
        "--reconstruct-individual-heads", action="store_true",
        help="persist deterministic individual C1 head reconstruction in control stage",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if not args.source.is_file() or not args.b0_root.is_dir() or not args.control_root.is_dir():
        raise FileNotFoundError("source, B0 root, or control root is unavailable")
    args.out_dir.mkdir(parents=True)
    parity = _control_parity(args.b0_root, args.control_root)
    (args.out_dir / "control_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    if parity["status"] != "passed":
        raise AssertionError("B0 control parity must pass before residual research")
    if args.stage == "control":
        if args.reconstruct_individual_heads:
            head_values, head_audit = reconstruct_individual_head_parity(
                source=args.source, b0_root=args.b0_root, control_root=args.control_root,
            )
            head_values.to_parquet(args.out_dir / "control_individual_head_reconstruction.parquet", index=False, compression="zstd")
            head_audit.to_parquet(args.out_dir / "control_individual_head_parity.parquet", index=False)
        print(json.dumps({"event": "control_parity_complete", **parity}, sort_keys=True))
        return
    fields = _feature_contract(args.control_root)
    source = _read_source(args.source, fields)
    b0 = _read_b0(args.b0_root)
    ledger = build_layer1_ledger(source, b0, fields)
    # Persist no outcome columns in the target-free Layer-1 score receipt.
    ledger.loc[:, [
        "candidate_id", "__decision_ts__", "control_block", "layer1_source", "base_score",
        "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
        "base_route_timestamp_top30", "layer1_prediction_is_strict_oof",
    ]].to_parquet(args.out_dir / "layer1_oof_predictions.parquet", index=False, compression="zstd")
    delta, delta_audit, delta_selection = build_delta1_oof(ledger, _score_schedule(b0))
    delta.to_parquet(args.out_dir / "delta1_oof_predictions.parquet", index=False, compression="zstd")
    delta_audit.to_parquet(args.out_dir / "delta1_fit_audit.parquet", index=False)
    delta_selection.to_parquet(args.out_dir / "delta1_selection_metrics.parquet", index=False)
    layer2, residual_audit, head_audit = build_layer2_oof(
        ledger, delta, b0, fields, checkpoint_root=args.out_dir / "layer2_blocks",
    )
    # Keep labels out of the prediction artifact; labels are joined below only
    # after all target-free model/rank columns have been determined.
    layer2.to_parquet(args.out_dir / "layer2_oof_predictions.parquet", index=False, compression="zstd")
    residual_audit.to_parquet(args.out_dir / "residual_fit_audit.parquet", index=False)
    head_audit.to_parquet(args.out_dir / "head_audit.parquet", index=False)
    labels = b0.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]]
    metrics = residual_metrics(layer2, labels)
    metrics.to_parquet(args.out_dir / "residual_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_base_recall_residual2_consensus_research_v3",
        "scope": "offline :00-only research; no live, canonical, exchange, portfolio or exit artifact modified",
        "source": {"path": str(args.source), "sha256": _sha256(args.source)},
        "b0_root": str(args.b0_root), "control_root": str(args.control_root),
        "base_feature_count": len(fields), "base_route_fraction": BASE_ROUTE,
        "reserve_days": RESERVE_DAYS,
        "layer1": "frozen current-v5 ten-head C1 stitched to pre-2025 strict-prequential history",
        "delta1": {
            "maps": "training-only isotonic C1-to-policy-residual global plus anchor-conditioned map",
            "conditional_map_shrinkage_to_global": DELTA1_CONDITIONAL_MAP_SHRINKAGE,
            "correction_shrinkage_grid": DELTA1_CORRECTION_SHRINKS,
            "selection": "minimum development-2025q1q3 strict-OOF absolute r2 error; 2025-Q4 and 2026 excluded",
            "selected_arm": delta_selection.loc[
                delta_selection["selected_from_development_only"], "arm"
            ].item(),
        },
        "layer2": {
            "r1": "six frozen-layout ordinal LambdaRank heads on r2=policy_net-a1",
            "r2": "three-class residual-direction classifier: r2<=-100 / (-100,+100) / r2>=+100 bps",
            "r3": (
                "clipped LightGBM Huber absolute-residual challenger; depth=3, "
                "minimum leaf=1000, L2=8, day-balanced fit weights"
            ),
            "fusion": "late MC1 evidence only in the first Stage-2 pass; no upstream/final-score authority",
            "r2_r3_feature_contract": {
                "original_frozen_base_fields": len(fields),
                "strict_oof_scalar_fields": LAYER2_SCALAR_FIELDS,
                "total_fields": len(_layer2_model_fields(fields)),
            },
        },
        "labels": "policy labels joined only after target-free held predictions and rank references are final",
        "runner_sha256": _sha256(Path(__file__)),
        "control_parity": parity,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "ledger_rows": len(ledger), "layer2_rows": len(layer2), "metrics": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
