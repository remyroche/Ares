#!/usr/bin/env python3
"""Sequential, conditional-usefulness HPO for the canonical ten-head residual stack.

The experiment is intentionally narrow and chronological:

* the frozen 120-field long-only source panel is the only residual feature pool;
* the residual target is ``policy_net_bps - prequential base_anchor_bps``;
* every held month is scored by models trained only on labels mature before
  that month's start (the 12-hour policy horizon therefore supplies the purge);
* target/query, conditional permutation selection, and per-head LambdaRank
  HPO use development months only;
* the untouched months are opened once after the complete ten-head contract is
  frozen; and
* a head is judged by the full 75/25 base-plus-median-consensus downstream
  score, globally ranked across the fixed candidate population -- never by its
  standalone NDCG or IC.

This is a long-only research runner because the authoritative source panel
currently contains only long candidates.  It makes no claim about the short
side.  ``policy_net_bps`` comes from the frozen 15-minute TP6/SL4 policy
contract (H12, next-hour entry, SL=3 ATR, activation=0.5 ATR,
giveback=0.25 ATR, and 100 bps cost once).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.query_candidate_definitions import (
    assign_query_ids,
    materialize_query_membership,
    query_definitions_by_name,
)
from extreme_price_movements.query_construction_pipeline import (
    audit_query_validity,
    query_common_shock_metrics,
    query_geometry,
    query_oracle_metrics,
    query_pair_metrics,
)
from extreme_price_movements.query_funnel import (
    aggregate_portability,
    portability_metrics,
    select_pareto_shortlist,
)
from extreme_price_movements.residual_lambdarank_hpo import (
    conditional_downstream_summary,
    complexity_penalty,
    downstream_tail_summary,
    make_pruned_study,
    materialize_lambdarank_params,
    passes_conditional_promotion,
    ranker_early_stopping_callbacks,
    restore_broad_lambdarank_params,
    suggest_broad_lambdarank_params,
)

SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_source_panel_long_2022_2026_20260809_v1/"
    "canonical_source_panel.parquet"
)
UPSTREAM = ROOT / "data_perp/artifacts/strict_r3_full_inference_2025_2026_v2/predictions.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/ten_head_conditional_usefulness_20260810_v1"
DEFAULT_DEVELOPMENT_MONTHS = ("2025-05", "2025-06", "2025-07")
DEFAULT_FINAL_MONTHS = ("2025-08", "2025-09", "2025-10")
DEFAULT_TRAIN_START = "2025-02-01"
TAILS = (.005, .01, .02, .05, .10)
SEED = 2718
EARLY_STOPPING_ROUNDS = 30

DEFAULT_RANK_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 120,
    "learning_rate": .035,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 300,
    "feature_fraction": .82,
    "bagging_fraction": .82,
    "bagging_freq": 1,
    "lambda_l1": .02,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "label_gain": [0, .25, 1, 3, 7],
    "lambdarank_truncation_level": 10,
    "verbosity": -1,
}
TARGETS: dict[str, tuple[float, float, float, float]] = {
    "resid_default_150_50": (-150.0, -50.0, 50.0, 150.0),
    "resid_tight_100_50": (-100.0, -50.0, 50.0, 100.0),
    "resid_symmetric_50_25": (-50.0, -25.0, 25.0, 50.0),
    "resid_wide_200_75": (-200.0, -75.0, 75.0, 200.0),
}
QUERY_NAMES = (
    "q0_exact_timestamp_side",
    "q1_cycle_2h_side",
    "q1_cycle_4h_side",
    "q1_cycle_6h_side",
    "q1_cycle_8h_side",
    "q1_cycle_12h_side",
)
SOURCE_FIXED_COLUMNS = {
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "r3_class", "r3_label_available_ts", "policy_path_valid",
    "policy_label_available_ts", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_cost_bps", "h12_label_valid",
    "h12_label_available_ts", "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps",
    "geometry_definition_population_complete",
}


def _event(name: str, **values: Any) -> None:
    """Emit small resumable-run breadcrumbs without serialising model state."""
    print(json.dumps({"event": name, **values}, default=str), flush=True)


@dataclass(frozen=True)
class HeadSpec:
    """One canonical consensus member."""

    cap: int
    weight_mode: str

    @property
    def name(self) -> str:
        return f"cap{self.cap}_{self.weight_mode}"


HEAD_SPECS = tuple(
    HeadSpec(cap=cap, weight_mode=mode)
    for cap in (40, 60, 80, 100, 120)
    for mode in ("ordinary", "equal_month")
)
HEAD_POSITION = {spec.name: position for position, spec in enumerate(HEAD_SPECS)}


def _head_seed(head: str) -> int:
    """Keep train sampling and model randomness identical across candidate arms."""
    return SEED + 10_000 * HEAD_POSITION[str(head)]


@dataclass
class HeadConfig:
    """Frozen mutable state for an individual head during development."""

    spec: HeadSpec
    target_name: str = "resid_default_150_50"
    query_name: str = "q1_cycle_4h_side"
    fields: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RANK_PARAMS))
    hpo_selected: bool = False
    selection_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cap": self.spec.cap,
            "weight_mode": self.spec.weight_mode,
            "target_name": self.target_name,
            "target_edges_bps": list(TARGETS[self.target_name]),
            "query_name": self.query_name,
            "field_count": len(self.fields),
            "fields": list(self.fields),
            "params": self.params,
            "hpo_selected": self.hpo_selected,
            "selection_log": self.selection_log,
        }


@dataclass
class FittedHead:
    """A train-only transformed LambdaRank model for one outer fold."""

    model: LGBMRanker | None
    fields: list[str]
    medians: np.ndarray
    reference_scores: np.ndarray
    test: pd.DataFrame
    fit_meta: dict[str, Any]

    def predict_rank(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), .5, dtype=float)
        X = _model_matrix(frame, self.fields, self.medians)
        return _rank_against_reference(self.reference_scores, self.model.predict(X))


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _month_start(month: str) -> pd.Timestamp:
    return _utc(f"{month}-01")


def _month_end(month: str) -> pd.Timestamp:
    return _month_start(month) + pd.offsets.MonthBegin(1)


def _source_columns(path: Path = SOURCE) -> list[str]:
    return [field.name for field in ds.dataset(path, format="parquet").schema]


def source_feature_columns(columns: Sequence[str]) -> list[str]:
    """Return source fields in the frozen 120-field canonical ordering."""
    fields = [str(column) for column in columns if str(column) not in SOURCE_FIXED_COLUMNS]
    if len(fields) != 120:
        raise ValueError(f"source panel must expose exactly 120 raw feature fields, got {len(fields)}")
    if len(set(fields)) != len(fields):
        raise ValueError("source feature contract has duplicate names")
    return fields


def residual_grade(residual_bps: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Return 0--4 ordinal residual grades with the declared inclusive edges."""
    if len(edges) != 4 or any(float(a) >= float(b) for a, b in zip(edges, edges[1:])):
        raise ValueError("residual target needs four strictly increasing edges")
    values = np.asarray(residual_bps, dtype=float)
    e0, e1, e2, e3 = (float(edge) for edge in edges)
    return np.select(
        [values <= e0, values <= e1, values <= e2, values <= e3],
        [0, 1, 2, 3], default=4,
    ).astype(np.int32)


def _rank_against_reference(reference: Sequence[float], values: Sequence[float]) -> np.ndarray:
    """Map held raw scores through an empirical CDF fitted only on train scores."""
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    values_array = np.asarray(values, dtype=float)
    if ref.size == 0:
        return np.full(values_array.shape, .5, dtype=float)
    out = np.searchsorted(ref, values_array, side="right").astype(float) / float(ref.size)
    out[~np.isfinite(out)] = .5
    return np.clip(out, 1.0 / float(ref.size), 1.0)


def _filter_source_table(
    source: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Sequence[str],
) -> pd.DataFrame:
    dataset = ds.dataset(source, format="parquet")
    start_scalar = pa.scalar(start.to_pydatetime(), type=pa.timestamp("ns", tz="UTC"))
    end_scalar = pa.scalar(end.to_pydatetime(), type=pa.timestamp("ns", tz="UTC"))
    expression = (ds.field("__ts__") >= start_scalar) & (ds.field("__ts__") < end_scalar)
    return dataset.to_table(columns=list(columns), filter=expression).to_pandas()


def _read_upstream(
    upstream: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cols = [
        "candidate_id", "__ts__", "base_score", "base_anchor_bps", "base_rank",
        "consensus_rank", "final_score",
    ]
    dataset = ds.dataset(upstream, format="parquet")
    start_scalar = pa.scalar(start.to_pydatetime(), type=pa.timestamp("ns", tz="UTC"))
    end_scalar = pa.scalar(end.to_pydatetime(), type=pa.timestamp("ns", tz="UTC"))
    expression = (ds.field("__ts__") >= start_scalar) & (ds.field("__ts__") < end_scalar)
    return dataset.to_table(columns=cols, filter=expression).to_pandas()


def load_authoritative_panel(
    *,
    source: Path = SOURCE,
    upstream: Path = UPSTREAM,
    start: str | pd.Timestamp = DEFAULT_TRAIN_START,
    end: str | pd.Timestamp = "2025-11-01",
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load the long-only, label-valid residual substrate and audit its lineage."""
    start_ts, end_ts = _utc(start), _utc(end)
    columns = _source_columns(source)
    features = source_feature_columns(columns)
    required = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "policy_path_valid", "policy_label_available_ts", "policy_gross_bps",
        "policy_net_bps", "policy_cost_bps", *features,
    ]
    panel = _filter_source_table(source, start=start_ts, end=end_ts, columns=required)
    scores = _read_upstream(upstream, start=start_ts, end=end_ts)
    for value in (panel, scores):
        value["__ts__"] = pd.to_datetime(value["__ts__"], utc=True, errors="raise")
    panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
    panel["policy_label_available_ts"] = pd.to_datetime(
        panel["policy_label_available_ts"], utc=True, errors="coerce"
    )
    if panel.candidate_id.duplicated().any() or scores.candidate_id.duplicated().any():
        raise ValueError("authoritative source or upstream scores contain duplicate candidate IDs")
    merged = panel.merge(
        scores.drop(columns="__ts__"), on="candidate_id", how="inner", validate="one_to_one",
    )
    for column in ["policy_net_bps", "policy_gross_bps", "base_anchor_bps", "base_rank", "base_score"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    valid = (
        merged["policy_path_valid"].fillna(False).astype(bool)
        & merged["policy_label_available_ts"].notna()
        & np.isfinite(merged["policy_net_bps"])
        & np.isfinite(merged["policy_gross_bps"])
        & np.isfinite(merged["base_anchor_bps"])
        & np.isfinite(merged["base_rank"])
    )
    before = len(merged)
    merged = merged.loc[valid].copy()
    if merged.empty:
        raise ValueError("no label-valid prequential residual rows survived")
    merged["month"] = merged["__ts__"].dt.strftime("%Y-%m")
    merged["net_bps"] = merged["policy_net_bps"].astype(float)
    merged["gross_bps"] = merged["policy_gross_bps"].astype(float)
    # Float32 keeps the wide source panel below a gigabyte while preserving the
    # feature values needed by tree thresholds.  Labels and anchors remain f64.
    for column in features:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(np.float32)
    if set(merged.side_name.unique()) != {"long"}:
        raise ValueError("this source contract is expected to be long-only")
    audit = pd.DataFrame([{
        "source_rows_requested": int(len(panel)),
        "upstream_rows_requested": int(len(scores)),
        "joined_rows": int(before),
        "label_valid_rows": int(len(merged)),
        "dropped_invalid_or_unmapped_rows": int(before - len(merged)),
        "feature_fields": int(len(features)),
        "side": "long",
        "start": str(start_ts),
        "end": str(end_ts),
    }])
    return merged.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), features, audit


def freeze_feature_contract(
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    development_start: str,
    train_start: str = DEFAULT_TRAIN_START,
) -> tuple[list[str], pd.DataFrame]:
    """Freeze the 120-field input order using only the earliest train window.

    The source assembler already requires every field to be finite before a
    row enters the panel.  We still record coverage and variation on the first
    authorized training window to catch any future source-contract regression.
    Fields are not reordered or selected from later development/final months.
    """
    cutoff = _month_start(development_start)
    history = frame.loc[
        frame["__ts__"].ge(_utc(train_start))
        & frame["policy_label_available_ts"].lt(cutoff)
    ].copy()
    if history.empty:
        raise ValueError("no earliest authorized training window for feature contract")
    rows: list[dict[str, Any]] = []
    for field_name in fields:
        values = pd.to_numeric(history[field_name], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        rows.append({
            "field": field_name,
            "train_rows": int(len(history)),
            "finite_fraction": float(finite.mean()),
            "finite_std": float(np.nanstd(values)) if finite.any() else float("nan"),
            "nonconstant": bool(finite.any() and np.nanstd(values) > 0.0),
            "retained": True,
        })
    audit = pd.DataFrame(rows)
    # Requiring this source check makes the cap-120 member meaningful.  Some
    # pre-existing structural fields legitimately have no variation during an
    # early historical window; retain them in the immutable contract and audit
    # them explicitly rather than silently changing a cap member's semantics.
    # Conditional MDA can later prove that such a field is not useful.
    failed = audit.loc[audit.finite_fraction.lt(.90), "field"].tolist()
    if failed:
        raise ValueError(
            "canonical 120-field source violates the >=90%/variation contract: " + ", ".join(failed[:12])
        )
    return list(fields), audit


def _fold_rows(frame: pd.DataFrame, month: str, *, train_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = _month_start(month)
    end = _month_end(month)
    train = frame.loc[
        frame["__ts__"].ge(_utc(train_start))
        & frame["policy_label_available_ts"].lt(cutoff)
    ].copy()
    test = frame.loc[frame["__ts__"].ge(cutoff) & frame["__ts__"].lt(end)].copy()
    if train.empty or test.empty:
        raise ValueError(f"empty chronological fold for {month}")
    if not train["policy_label_available_ts"].lt(cutoff).all():
        raise AssertionError("unresolved policy labels entered residual training")
    if set(train.candidate_id).intersection(test.candidate_id):
        raise AssertionError("train/test candidate overlap")
    return train, test


def _group_sample(
    frame: pd.DataFrame,
    query_ids: pd.Series,
    *,
    max_rows: int,
    weight_mode: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Retain whole queries in a deterministic, optionally equal-month sample."""
    x = frame.copy()
    x["__query__"] = query_ids.astype("string").to_numpy()
    sizes = x.groupby("__query__", observed=True).size()
    x = x.loc[x["__query__"].isin(sizes.index[sizes.ge(2)])].copy()
    if x.empty:
        return x, np.empty(0, dtype=np.int64)
    if len(x) > max_rows:
        rng = np.random.default_rng(seed)
        group_meta = (
            x.groupby("__query__", observed=True)
            .agg(rows=("candidate_id", "size"), month=("month", "first"), first_ts=("__ts__", "min"))
            .reset_index()
        )
        keep: list[str] = []
        if weight_mode == "equal_month":
            groups_by_month = list(group_meta.groupby("month", sort=True, observed=True))
            allowance = max(2, max_rows // max(len(groups_by_month), 1))
            for _, group in groups_by_month:
                ordered = group.iloc[rng.permutation(len(group))]
                used = 0
                for query_value, row_count, _, _ in ordered[["__query__", "rows", "month", "first_ts"]].itertuples(index=False, name=None):
                    if used and used + int(row_count) > allowance:
                        continue
                    keep.append(str(query_value))
                    used += int(row_count)
        else:
            ordered = group_meta.iloc[rng.permutation(len(group_meta))]
            used = 0
            for query_value, row_count, _, _ in ordered[["__query__", "rows", "month", "first_ts"]].itertuples(index=False, name=None):
                if used and used + int(row_count) > max_rows:
                    continue
                keep.append(str(query_value))
                used += int(row_count)
        x = x.loc[x["__query__"].isin(keep)].copy()
    x = x.sort_values(["__query__", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    _, groups = np.unique(x["__query__"].to_numpy(), return_counts=True)
    return x, groups.astype(np.int64)


def _fit_medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(fields)].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    return medians


def _transform(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray) -> np.ndarray:
    values = frame.loc[:, list(fields)].to_numpy(dtype=np.float32, copy=True)
    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.take(medians, np.where(invalid)[1])
    return values


def _model_matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray) -> pd.DataFrame:
    """Return a named finite matrix so LightGBM checks train/predict lineage."""
    return pd.DataFrame(
        _transform(frame, fields, medians), columns=list(fields), index=frame.index,
    )


def _inner_split(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest 20% of complete queries for chronological stopping."""
    group_start = sample.groupby("__query__", observed=True)["__ts__"].min().sort_values(kind="stable")
    if len(group_start) < 12:
        return sample, sample.iloc[:0].copy()
    validation_groups = set(group_start.iloc[max(1, int(math.floor(.80 * len(group_start)))):].index)
    validation = sample.loc[sample["__query__"].isin(validation_groups)].copy()
    train = sample.loc[~sample["__query__"].isin(validation_groups)].copy()
    if len(train) < 1000 or len(validation) < 500:
        return sample, sample.iloc[:0].copy()
    return train, validation


def _ranker_params(params: dict[str, Any], *, training_rows: int, seed: int) -> dict[str, Any]:
    actual = dict(params)
    if "min_child_samples_fraction" in actual:
        actual = materialize_lambdarank_params(actual, training_rows=training_rows)
    actual.update({"random_state": int(seed), "n_jobs": 2, "verbosity": -1})
    return actual


def _weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "ordinary":
        return None
    if mode != "equal_month":
        raise ValueError(f"unknown residual head weight mode: {mode}")
    frequency = frame["month"].value_counts()
    values = frame["month"].map(lambda month: 1.0 / float(frequency.loc[month])).to_numpy(float)
    return values * len(values) / max(float(values.sum()), 1e-12)


def fit_head_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: HeadConfig,
    *,
    max_train_rows: int,
    seed: int,
    early_stopping: bool,
    return_fitted: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], FittedHead | None]:
    """Fit one head strictly before a held month and rank that held population."""
    query_definition = query_definitions_by_name([config.query_name])[0]
    train_target = residual_grade(
        train["net_bps"].to_numpy(float) - train["base_anchor_bps"].to_numpy(float),
        TARGETS[config.target_name],
    )
    test_target = residual_grade(
        test["net_bps"].to_numpy(float) - test["base_anchor_bps"].to_numpy(float),
        TARGETS[config.target_name],
    )
    train_with_target = train.copy()
    train_with_target["__target__"] = train_target
    sampled, train_groups = _group_sample(
        train_with_target,
        assign_query_ids(train_with_target, query_definition),
        max_rows=max_train_rows,
        weight_mode=config.spec.weight_mode,
        seed=seed,
    )
    if sampled.empty or sampled["__target__"].nunique() < 2:
        prediction = test[["candidate_id", "month"]].copy()
        prediction[config.name] = .5
        meta = {
            "head": config.name, "train_rows": int(len(sampled)), "test_rows": int(len(test)),
            "status": "neutral_insufficient_training_support", "query": config.query_name,
        }
        return prediction, meta, None

    if early_stopping:
        core, validation = _inner_split(sampled)
    else:
        core, validation = sampled, sampled.iloc[:0].copy()
    medians = _fit_medians(core, config.fields)
    core = core.sort_values(["__query__", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    _, core_groups = np.unique(core["__query__"].to_numpy(), return_counts=True)
    if core["__target__"].nunique() < 2 or len(core_groups) == 0:
        prediction = test[["candidate_id", "month"]].copy()
        prediction[config.name] = .5
        meta = {
            "head": config.name, "train_rows": int(len(core)), "test_rows": int(len(test)),
            "status": "neutral_inner_training_support", "query": config.query_name,
        }
        return prediction, meta, None
    params = _ranker_params(config.params, training_rows=len(core), seed=seed)
    model = LGBMRanker(**params)
    core_matrix = _model_matrix(core, config.fields, medians)
    fit_kwargs: dict[str, Any] = {
        "group": core_groups,
        "sample_weight": _weights(core, config.spec.weight_mode),
    }
    if not validation.empty:
        validation = validation.sort_values(["__query__", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        _, validation_groups = np.unique(validation["__query__"].to_numpy(), return_counts=True)
        if validation["__target__"].nunique() >= 2 and len(validation_groups):
            fit_kwargs.update({
                "eval_set": [(_model_matrix(validation, config.fields, medians), validation["__target__"].to_numpy(np.int32))],
                "eval_group": [validation_groups],
                "eval_sample_weight": [_weights(validation, config.spec.weight_mode)],
                "callbacks": ranker_early_stopping_callbacks(rounds=EARLY_STOPPING_ROUNDS),
            })
    model.fit(
        core_matrix, core["__target__"].to_numpy(np.int32), **fit_kwargs,
    )
    reference_scores = model.predict(core_matrix)
    ranks = _rank_against_reference(reference_scores, model.predict(_model_matrix(test, config.fields, medians)))
    prediction = test[["candidate_id", "month"]].copy()
    prediction[config.name] = ranks
    fit_meta = {
        "head": config.name,
        "target": config.target_name,
        "query": config.query_name,
        "field_count": len(config.fields),
        "train_rows": int(len(sampled)),
        "fit_rows": int(len(core)),
        "early_stop_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_query_count": int(len(core_groups)),
        "test_target_grade_count": int(np.unique(test_target).size),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or params.get("n_estimators", 0)),
        "status": "fit",
    }
    fitted = FittedHead(
        model=model, fields=list(config.fields), medians=medians,
        reference_scores=np.asarray(reference_scores, dtype=float), test=test.copy(), fit_meta=fit_meta,
    )
    return prediction, fit_meta, fitted if return_fitted else None


def head_predictions_for_months(
    frame: pd.DataFrame,
    config: HeadConfig,
    months: Sequence[str],
    *,
    train_start: str,
    max_train_rows: int,
    seed: int,
    early_stopping: bool,
    return_fitted: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FittedHead]]:
    predictions: list[pd.DataFrame] = []
    meta: list[dict[str, Any]] = []
    fitted: dict[str, FittedHead] = {}
    for offset, month in enumerate(months):
        train, test = _fold_rows(frame, month, train_start=train_start)
        prediction, fit_meta, model = fit_head_fold(
            train, test, config,
            max_train_rows=max_train_rows,
            seed=seed + offset,
            early_stopping=early_stopping,
            return_fitted=return_fitted,
        )
        prediction["__ts__"] = test["__ts__"].to_numpy()
        predictions.append(prediction)
        fit_meta["month"] = month
        meta.append(fit_meta)
        if model is not None:
            fitted[month] = model
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(meta), fitted


def evaluation_population(frame: pd.DataFrame, months: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[frame["month"].isin(list(months))].copy()
    if out.candidate_id.duplicated().any():
        raise AssertionError("evaluation population has duplicate candidate IDs")
    cols = [
        "candidate_id", "__ts__", "month", "side_name", "__symbol__",
        "net_bps", "gross_bps", "base_rank", "base_anchor_bps", "base_score",
    ]
    return out.loc[:, cols].sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def stack_scores(population: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    """Combine all ten train-CDF head ranks with the frozen prequential base rank."""
    expected = {spec.name for spec in HEAD_SPECS}
    missing = expected.difference(ranks.columns)
    if missing:
        raise KeyError(f"ten-head stack missing {sorted(missing)}")
    if ranks.candidate_id.duplicated().any():
        raise ValueError("head-rank table duplicated candidate IDs")
    x = population.merge(ranks[["candidate_id", *sorted(expected)]], on="candidate_id", how="inner", validate="one_to_one")
    if len(x) != len(population):
        raise AssertionError("a head changed the fixed evaluation candidate population")
    head_columns = [spec.name for spec in HEAD_SPECS]
    x["consensus_rank"] = np.nanmedian(x.loc[:, head_columns].to_numpy(float), axis=1)
    x["score"] = .75 * pd.to_numeric(x["base_rank"], errors="coerce") + .25 * x["consensus_rank"]
    return x


def _comparison(
    incumbent: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, float]:
    joined = incumbent[["candidate_id", "__ts__", "net_bps", "gross_bps", "score"]].merge(
        candidate[["candidate_id", "score"]], on="candidate_id", suffixes=("_incumbent", "_candidate"), validate="one_to_one",
    )
    return conditional_downstream_summary(
        joined,
        candidate_score_column="score_candidate",
        incumbent_score_column="score_incumbent",
    )


def _metrics_table(frame: pd.DataFrame, *, arm: str) -> pd.DataFrame:
    summary = downstream_tail_summary(frame, score_column="score", tails=TAILS)
    rows: list[dict[str, Any]] = []
    for tail, label in ((.005, "top0_5"), (.01, "top1"), (.02, "top2"), (.05, "top5"), (.10, "top10")):
        rows.append({
            "arm": arm,
            "scope": "pooled",
            "tail": tail,
            "rows": summary.get(f"{label}_rows"),
            "net_bps_per_trade": summary.get(f"{label}_net_bps"),
            "gross_bps_per_trade": summary.get(f"{label}_gross_bps"),
            "net_sum_bps": summary.get(f"{label}_net_sum_bps"),
            "month_worst_net_bps": summary.get(f"{label}_month_worst_net_bps"),
            "month_mad_net_bps": summary.get(f"{label}_month_mad_net_bps"),
        })
    x = frame.copy()
    x["__month__"] = pd.to_datetime(x["__ts__"], utc=True).dt.strftime("%Y-%m")
    for month, group in x.groupby("__month__", sort=True, observed=True):
        per_month = downstream_tail_summary(group, score_column="score", tails=TAILS)
        for tail, label in ((.005, "top0_5"), (.01, "top1"), (.02, "top2"), (.05, "top5"), (.10, "top10")):
            rows.append({
                "arm": arm,
                "scope": str(month),
                "tail": tail,
                "rows": per_month.get(f"{label}_rows"),
                "net_bps_per_trade": per_month.get(f"{label}_net_bps"),
                "gross_bps_per_trade": per_month.get(f"{label}_gross_bps"),
                "net_sum_bps": per_month.get(f"{label}_net_sum_bps"),
                "month_worst_net_bps": np.nan,
                "month_mad_net_bps": np.nan,
            })
    return pd.DataFrame(rows)


def _head_matrix_from_predictions(population: pd.DataFrame, values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = population[["candidate_id"]].copy()
    for name, prediction in values.items():
        if name not in prediction:
            raise KeyError(f"head prediction has no {name}")
        out = out.merge(prediction[["candidate_id", name]], on="candidate_id", how="left", validate="one_to_one")
        if out[name].isna().any():
            raise AssertionError(f"head {name} did not score every fixed evaluation row")
    return out


def _head_order(population: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    incumbent = stack_scores(population, ranks)
    rows: list[dict[str, Any]] = []
    for spec in HEAD_SPECS:
        without = ranks.drop(columns=spec.name)
        # Nine members remain, so use their median without introducing an
        # arbitrary neutral pseudo-head.  This measures head necessity in its
        # actual consensus context.
        x = population.merge(without, on="candidate_id", validate="one_to_one")
        columns = [name for name in without.columns if name != "candidate_id"]
        x["consensus_rank"] = np.nanmedian(x[columns].to_numpy(float), axis=1)
        x["score"] = .75 * x["base_rank"] + .25 * x["consensus_rank"]
        summary = _comparison(incumbent, x)
        rows.append({
            "head": spec.name,
            "removal_utility_change_bps": summary["conditional_utility_uplift_bps"],
            "necessity_bps": -summary["conditional_utility_uplift_bps"],
            **summary,
        })
    return pd.DataFrame(rows).sort_values(["necessity_bps", "head"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _screen_queries(
    development: pd.DataFrame,
    *,
    out: Path,
    target_name: str,
    limit: int,
) -> tuple[str, ...]:
    """Reuse the no-model screen before any target/query model fits."""
    definition = query_definitions_by_name(QUERY_NAMES)
    x = development[["candidate_id", "__ts__", "side_name", "month", "net_bps", "gross_bps", "base_anchor_bps"]].copy()
    x["grade"] = residual_grade(x.net_bps.to_numpy(float) - x.base_anchor_bps.to_numpy(float), TARGETS[target_name])
    x["fold"] = x["month"]
    membership = materialize_query_membership(x, definition)
    validity = audit_query_validity(x, membership, fold_column="fold")
    if (validity["candidate_duplicate_membership_rate"].ne(0) | validity["query_boundary_violation_count"].ne(0)).any():
        raise AssertionError("predeclared query grammar failed validity audit")
    geometry = query_geometry(x, membership, grade_column="grade")
    pairs = query_pair_metrics(x, membership, grade_column="grade")
    oracle = query_oracle_metrics(x, membership)
    shocks = query_common_shock_metrics(x, membership)
    portable = aggregate_portability(portability_metrics(x, membership, grade_column="grade", era_column="month"))
    summary = geometry.merge(pairs, on="query_candidate").merge(oracle, on="query_candidate").merge(shocks, on="query_candidate").merge(portable, on="query_candidate")
    chosen = select_pareto_shortlist(summary, limit=limit)
    names = tuple(chosen.loc[chosen["shortlisted"], "query_candidate"].astype(str).tolist())
    if "q1_cycle_4h_side" not in names:
        names = ("q1_cycle_4h_side", *names[: max(0, limit - 1)])
    out.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(out / "candidate_query_membership.parquet", index=False)
    validity.to_parquet(out / "query_validity_audit.parquet", index=False)
    summary.to_parquet(out / "query_screen_summary.parquet", index=False)
    chosen.to_parquet(out / "query_pareto_frontier.parquet", index=False)
    (out / "query_shortlist.json").write_text(json.dumps({"shortlist": names, "target": target_name}, indent=2) + "\n")
    return names


def _screen_targets(development: pd.DataFrame) -> pd.DataFrame:
    """Cheap label-semantic screen; downstream selection remains conditional."""
    rows: list[dict[str, Any]] = []
    residual = development.net_bps.to_numpy(float) - development.base_anchor_bps.to_numpy(float)
    for name, edges in TARGETS.items():
        grade = residual_grade(residual, edges)
        x = development.assign(__grade__=grade)
        means = x.groupby("__grade__", observed=True).net_bps.mean()
        rows.append({
            "target": name,
            "edges_bps": json.dumps(list(edges)),
            # ``development`` retains its source-panel index after filtering.
            # Preserve it here: an unindexed Series would align against a
            # different integer index and silently report an all-null
            # correlation.
            "grade_net_spearman": float(pd.Series(grade, index=x.index).corr(x["net_bps"], method="spearman")),
            "grade0_net_bps": float(means.get(0, np.nan)),
            "grade4_net_bps": float(means.get(4, np.nan)),
            "grade_spread_bps": float(means.get(4, np.nan) - means.get(0, np.nan)),
            "grade_entropy": float(-np.sum(np.bincount(grade, minlength=5) / len(grade) * np.log(np.maximum(np.bincount(grade, minlength=5) / len(grade), 1e-12)))),
        })
    return pd.DataFrame(rows).sort_values(["grade_spread_bps", "target"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _target_query_candidates(target_screen: pd.DataFrame, queries: Sequence[str], *, limit: int) -> list[tuple[str, str]]:
    """Bound target/query breadth while preserving both kinds of variation."""
    ranked_targets = target_screen["target"].astype(str).tolist()
    baseline = ("resid_default_150_50", "q1_cycle_4h_side")
    # The initial funnel must actually test target and query changes.  A naïve
    # Cartesian ordering would often consume its complete budget on one target
    # with several queries (or vice versa), then mislabel that as joint HPO.
    # Seed the short list with target changes at the canonical query and query
    # changes on the canonical target, then fill the remaining budget from the
    # predeclared Cartesian grid.
    canonical_query = baseline[1]
    target_changes = [(target, canonical_query) for target in ranked_targets if target != baseline[0]]
    query_changes = [(baseline[0], query) for query in queries if query != canonical_query]
    cartesian = [(target, query) for target in ranked_targets for query in queries]
    ordered = [baseline]
    for index in range(max(len(target_changes), len(query_changes))):
        if index < len(target_changes):
            ordered.append(target_changes[index])
        if index < len(query_changes):
            ordered.append(query_changes[index])
    ordered.extend(cartesian)
    ordered = list(dict.fromkeys(ordered))
    return ordered[: max(1, int(limit))]


def _fit_candidate_matrix(
    frame: pd.DataFrame,
    config: HeadConfig,
    months: Sequence[str],
    *,
    train_start: str,
    max_train_rows: int,
    seed: int,
    early_stopping: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction, meta, _ = head_predictions_for_months(
        frame, config, months, train_start=train_start, max_train_rows=max_train_rows,
        seed=seed, early_stopping=early_stopping,
    )
    return prediction, meta


def _conditional_target_query_stage(
    frame: pd.DataFrame,
    population: pd.DataFrame,
    ranks: pd.DataFrame,
    configs: dict[str, HeadConfig],
    order: Sequence[str],
    candidates: Sequence[tuple[str, str]],
    months: Sequence[str],
    *,
    train_start: str,
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, HeadConfig]]:
    records: list[dict[str, Any]] = []
    fit_records: list[pd.DataFrame] = []
    current = ranks.copy()
    for position, head in enumerate(order):
        _event("target_query_head_start", head=head, position=position, candidates=len(candidates))
        incumbent_score = stack_scores(population, current)
        incumbent_config = configs[head]
        arm_rows: list[dict[str, Any]] = []
        best_prediction: pd.DataFrame | None = None
        best_summary: dict[str, float] | None = None
        best_config: HeadConfig | None = None
        for index, (target_name, query_name) in enumerate(candidates):
            candidate_config = HeadConfig(
                spec=incumbent_config.spec,
                target_name=target_name,
                query_name=query_name,
                fields=list(incumbent_config.fields),
                params=dict(incumbent_config.params),
            )
            if target_name == incumbent_config.target_name and query_name == incumbent_config.query_name:
                # Re-fitting the named incumbent with a different subsample
                # seed would manufacture a target/query result from ordinary
                # model randomness.  Keep the actual incumbent rank exactly.
                prediction = current[["candidate_id", head]].copy()
                meta = pd.DataFrame([{
                    "head": head, "month": "all_development", "status": "cached_incumbent",
                    "train_rows": np.nan, "test_rows": len(prediction),
                }])
            else:
                prediction, meta = _fit_candidate_matrix(
                    frame, candidate_config, months,
                    train_start=train_start, max_train_rows=max_train_rows,
                    seed=_head_seed(head), early_stopping=False,
                )
            candidate_ranks = current.drop(columns=head).merge(prediction[["candidate_id", head]], on="candidate_id", validate="one_to_one")
            candidate_score = stack_scores(population, candidate_ranks)
            summary = _comparison(incumbent_score, candidate_score)
            record = {
                "stage": "target_query",
                "head": head,
                "position": position,
                "candidate_target": target_name,
                "candidate_query": query_name,
                "is_incumbent_contract": target_name == incumbent_config.target_name and query_name == incumbent_config.query_name,
                "promotable": passes_conditional_promotion(summary),
                **summary,
            }
            records.append(record)
            arm_rows.append(record)
            fit_records.append(meta.assign(stage="target_query", target_candidate=target_name, query_candidate=query_name))
            if best_summary is None or (
                summary["conditional_utility_uplift_bps"], summary["delta_top5_net_bps"], summary["delta_top1_net_bps"]
            ) > (
                best_summary["conditional_utility_uplift_bps"], best_summary["delta_top5_net_bps"], best_summary["delta_top1_net_bps"]
            ):
                best_prediction, best_summary, best_config = prediction, summary, candidate_config
        if best_summary is not None and best_config is not None and passes_conditional_promotion(best_summary):
            configs[head] = best_config
            configs[head].selection_log.append({"stage": "target_query", "promoted": True, **best_summary})
            current = current.drop(columns=head).merge(best_prediction[["candidate_id", head]], on="candidate_id", validate="one_to_one")
        else:
            configs[head].selection_log.append({
                "stage": "target_query", "promoted": False,
                "best_summary": best_summary,
            })
        _event("target_query_head_complete", head=head, promoted=bool(best_summary is not None and passes_conditional_promotion(best_summary)))
    return pd.DataFrame(records), pd.concat(fit_records, ignore_index=True), configs


def _stable_mda_population(population: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    """Return a candidate-only, equal-month deterministic MDA screen sample."""
    if max_rows <= 0 or len(population) <= max_rows:
        return population.copy()
    groups = list(population.groupby("month", sort=True, observed=True))
    quota = max(1, int(math.ceil(max_rows / max(len(groups), 1))))
    pieces: list[pd.DataFrame] = []
    for _, group in groups:
        hashed = pd.util.hash_pandas_object(group["candidate_id"], index=False).to_numpy(np.uint64)
        take = np.argsort(hashed, kind="stable")[:quota]
        pieces.append(group.iloc[take])
    return pd.concat(pieces, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _mda_stage(
    frame: pd.DataFrame,
    population: pd.DataFrame,
    ranks: pd.DataFrame,
    configs: dict[str, HeadConfig],
    order: Sequence[str],
    months: Sequence[str],
    *,
    train_start: str,
    max_train_rows: int,
    mda_max_eval_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, HeadConfig]]:
    """Conditional downstream MDA, followed by one all-selected refit per head."""
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    full_population = population
    full_current = ranks.copy()
    mda_population = _stable_mda_population(population, max_rows=mda_max_eval_rows)
    mda_current = mda_population[["candidate_id"]].merge(ranks, on="candidate_id", validate="one_to_one")
    for position, head in enumerate(order):
        _event("conditional_mda_head_start", head=head, position=position, fields=len(configs[head].fields))
        config = configs[head]
        incumbent_score = stack_scores(mda_population, mda_current)
        incumbent_full_score = stack_scores(full_population, full_current)
        # Store rank outputs as one compact float32 matrix.  Keeping a pandas
        # frame for every (field, month) would multiply the 120-field cap's
        # memory footprint during a three-month MDA by several times.
        candidate_position = pd.Series(
            np.arange(len(mda_population), dtype=np.int64), index=mda_population["candidate_id"].to_numpy(),
        )
        permutations = np.full((len(mda_population), len(config.fields)), .5, dtype=np.float32)
        for month_index, month in enumerate(months):
            train, test = _fold_rows(frame, month, train_start=train_start)
            test = test.loc[test["candidate_id"].isin(set(mda_population.loc[mda_population.month.eq(month), "candidate_id"]))].copy()
            if test.empty:
                raise AssertionError(f"MDA sample has no candidates for {month}")
            position_series = candidate_position.reindex(test["candidate_id"].to_numpy())
            if position_series.isna().any():
                raise AssertionError("MDA test candidates do not match the fixed evaluation population")
            positions = position_series.to_numpy(dtype=np.int64)
            if (positions < 0).any() or len(np.unique(positions)) != len(positions):
                raise AssertionError("MDA test candidates do not match the fixed evaluation population")
            _, _, fitted = fit_head_fold(
                train, test, config,
                max_train_rows=max_train_rows,
                seed=_head_seed(head) + month_index,
                early_stopping=False,
                return_fitted=True,
            )
            if fitted is None:
                continue
            base_matrix = _transform(test, fitted.fields, fitted.medians)
            for field_index, field_name in enumerate(config.fields):
                rng = np.random.default_rng(SEED + 1_000_000 * position + 10_000 * month_index + field_index)
                permuted = base_matrix.copy()
                permuted[:, field_index] = permuted[rng.permutation(len(permuted)), field_index]
                ranks_permuted = _rank_against_reference(
                    fitted.reference_scores,
                    fitted.model.predict(pd.DataFrame(permuted, columns=fitted.fields, index=test.index)),
                )
                permutations[positions, field_index] = ranks_permuted.astype(np.float32)
        impacts: list[tuple[str, dict[str, float]]] = []
        for field_index, field_name in enumerate(config.fields):
            prediction = mda_population[["candidate_id"]].copy()
            prediction[head] = permutations[:, field_index]
            candidate_ranks = mda_current.drop(columns=head).merge(prediction, on="candidate_id", validate="one_to_one")
            candidate_score = stack_scores(mda_population, candidate_ranks)
            summary = _comparison(incumbent_score, candidate_score)
            impact = -float(summary["conditional_utility_uplift_bps"])
            records.append({
                "stage": "conditional_mda", "head": head, "position": position,
                "field": field_name, "conditional_importance_bps": impact,
                "permutation_hurts_stack": impact > 0.0, **summary,
            })
            impacts.append((field_name, summary))
        positive = [
            field_name for field_name, summary in impacts
            if -float(summary["conditional_utility_uplift_bps"]) > 0.0
        ]
        selected = [field_name for field_name in config.fields if field_name in set(positive)]
        promoted = False
        selected_summary: dict[str, float] | None = None
        if len(selected) >= 12 and len(selected) < len(config.fields):
            selected_config = HeadConfig(
                spec=config.spec, target_name=config.target_name, query_name=config.query_name,
                fields=selected, params=dict(config.params), hpo_selected=config.hpo_selected,
            )
            prediction, _ = _fit_candidate_matrix(
                frame, selected_config, months, train_start=train_start,
                max_train_rows=max_train_rows, seed=_head_seed(head),
                early_stopping=False,
            )
            candidate_full_ranks = full_current.drop(columns=head).merge(prediction[["candidate_id", head]], on="candidate_id", validate="one_to_one")
            candidate_full_score = stack_scores(full_population, candidate_full_ranks)
            selected_summary = _comparison(incumbent_full_score, candidate_full_score)
            if passes_conditional_promotion(selected_summary):
                configs[head] = selected_config
                configs[head].selection_log.append({"stage": "conditional_mda", "promoted": True, **selected_summary})
                full_current = candidate_full_ranks
                mda_current = mda_population[["candidate_id"]].merge(full_current, on="candidate_id", validate="one_to_one")
                promoted = True
        if not promoted:
            configs[head].selection_log.append({
                "stage": "conditional_mda", "promoted": False,
                "positive_fields": selected,
                "selected_summary": selected_summary,
            })
        decisions.append({
            "head": head, "original_field_count": len(config.fields),
            "positive_individual_fields": len(selected),
            "selected_field_count": len(configs[head].fields), "promoted": promoted,
            "selected_fields_json": json.dumps(configs[head].fields),
            "mda_eval_rows": len(mda_population),
            "mda_is_candidate_only_screen": len(mda_population) < len(full_population),
            **({f"selection_{key}": value for key, value in (selected_summary or {}).items()}),
        })
        _event("conditional_mda_head_complete", head=head, promoted=promoted, selected_fields=len(configs[head].fields))
    return pd.DataFrame(records), pd.DataFrame(decisions), configs


def _trial_params(trial: Any, *, median_candidates: float) -> dict[str, Any]:
    return suggest_broad_lambdarank_params(
        trial, retained_fraction=.05, median_candidates_per_query=max(2.0, median_candidates),
    )


def _hpo_stage(
    frame: pd.DataFrame,
    population: pd.DataFrame,
    ranks: pd.DataFrame,
    configs: dict[str, HeadConfig],
    order: Sequence[str],
    months: Sequence[str],
    *,
    train_start: str,
    max_train_rows: int,
    trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, HeadConfig]]:
    """Run median-pruned HPO sequentially, conditional on the other nine heads."""
    records: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    current = ranks.copy()
    if trials <= 0:
        return pd.DataFrame(records), pd.DataFrame(fit_rows), configs
    for position, head in enumerate(order):
        _event("head_hpo_start", head=head, position=position, trials=trials)
        config = configs[head]
        incumbent_score = stack_scores(population, current)
        first_train, _ = _fold_rows(frame, months[0], train_start=train_start)
        definition = query_definitions_by_name([config.query_name])[0]
        group_sizes = assign_query_ids(first_train, definition).value_counts()
        median_candidates = float(group_sizes.median()) if len(group_sizes) else 2.0
        study = make_pruned_study(seed=SEED + 40_000 * position, n_startup_trials=min(4, max(1, trials // 2)), n_warmup_steps=1)

        def objective(trial: Any) -> float:
            suggested = _trial_params(trial, median_candidates=median_candidates)
            candidate_config = HeadConfig(
                spec=config.spec, target_name=config.target_name, query_name=config.query_name,
                fields=list(config.fields), params=suggested,
            )
            pieces: list[pd.DataFrame] = []
            best_iterations: list[int] = []
            for month_index, month in enumerate(months):
                prediction, meta = _fit_candidate_matrix(
                    frame, candidate_config, [month], train_start=train_start,
                    max_train_rows=max_train_rows, seed=_head_seed(head),
                    early_stopping=True,
                )
                pieces.append(prediction)
                candidate_part = current.loc[current.candidate_id.isin(set(prediction.candidate_id))].drop(columns=head).merge(
                    prediction[["candidate_id", head]], on="candidate_id", validate="one_to_one",
                )
                pop_part = population.loc[population.candidate_id.isin(set(prediction.candidate_id))]
                incremental = _comparison(
                    incumbent_score.loc[incumbent_score.candidate_id.isin(set(prediction.candidate_id))],
                    stack_scores(pop_part, candidate_part),
                )
                trial.report(float(incremental["conditional_utility_uplift_bps"]), step=month_index + 1)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
                fit_rows.append({"head": head, "trial": trial.number, "month": month, **meta.iloc[0].to_dict()})
                best_iterations.append(int(meta.iloc[0].get("best_iteration", suggested["n_estimators"])))
            prediction = pd.concat(pieces, ignore_index=True)
            candidate_ranks = current.drop(columns=head).merge(prediction[["candidate_id", head]], on="candidate_id", validate="one_to_one")
            candidate_score = stack_scores(population, candidate_ranks)
            summary = _comparison(incumbent_score, candidate_score)
            penalty = complexity_penalty(
                max_depth=int(suggested["max_depth"]), num_leaves=int(suggested["num_leaves"]),
            )
            for key, value in summary.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("complexity_penalty_bps", penalty)
            trial.set_user_attr("promotable", passes_conditional_promotion(summary))
            trial.set_user_attr("best_iterations", best_iterations)
            return float(summary["conditional_utility_uplift_bps"] - penalty)

        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        trial_records: list[dict[str, Any]] = []
        for trial in study.trials:
            trial_records.append({
                "stage": "head_hpo", "head": head, "position": position,
                "trial": int(trial.number), "state": trial.state.name, "objective": trial.value,
                **trial.params, **{f"metric_{key}": value for key, value in trial.user_attrs.items()},
            })
        records.extend(trial_records)
        complete = [
            trial for trial in study.trials
            if trial.state.name == "COMPLETE" and trial.value is not None
        ]
        if not complete:
            configs[head].selection_log.append({"stage": "head_hpo", "promoted": False, "reason": "all_trials_pruned"})
            continue
        best = max(complete, key=lambda trial: float(trial.value))
        params = restore_broad_lambdarank_params(best.params)
        # HPO itself uses a late chronological slice for early stopping.  The
        # conditional promotion recheck below must not give the challenger a
        # smaller effective training set than the incumbent, so refit on every
        # available mature row with the median fold-selected tree count.
        best_iterations = [int(value) for value in best.user_attrs.get("best_iterations", []) if int(value) > 0]
        params["n_estimators"] = int(np.median(best_iterations)) if best_iterations else int(params["n_estimators"])
        winner = HeadConfig(
            spec=config.spec, target_name=config.target_name, query_name=config.query_name,
            fields=list(config.fields), params=params, hpo_selected=True,
        )
        prediction, _ = _fit_candidate_matrix(
            frame, winner, months, train_start=train_start,
            max_train_rows=max_train_rows, seed=_head_seed(head),
            early_stopping=False,
        )
        candidate_ranks = current.drop(columns=head).merge(prediction[["candidate_id", head]], on="candidate_id", validate="one_to_one")
        candidate_score = stack_scores(population, candidate_ranks)
        summary = _comparison(incumbent_score, candidate_score)
        if passes_conditional_promotion(summary):
            winner.selection_log.append({"stage": "head_hpo", "promoted": True, "trial": best.number, **summary})
            configs[head] = winner
            current = candidate_ranks
        else:
            configs[head].selection_log.append({"stage": "head_hpo", "promoted": False, "trial": best.number, **summary})
        _event("head_hpo_complete", head=head, promoted=passes_conditional_promotion(summary), winner_trial=best.number)
    return pd.DataFrame(records), pd.DataFrame(fit_rows), configs


def _configs_default(fields: Sequence[str]) -> dict[str, HeadConfig]:
    return {
        spec.name: HeadConfig(spec=spec, fields=list(fields[: spec.cap]))
        for spec in HEAD_SPECS
    }


def _restore_after_target_query(
    *,
    fields: Sequence[str],
    target_query_trials: pd.DataFrame,
    head_order: Sequence[str],
) -> dict[str, HeadConfig]:
    """Reconstruct the frozen target/query contract from its persisted trials.

    Target/query changes never alter fields or ranker parameters, and each
    head is independently fitted.  The persisted conditional trial table plus
    the deterministic head order therefore fully identifies the configuration
    selected before MDA.  This lets a long MDA/HPO continuation resume without
    reopening development selection or touching final months.
    """
    required = {
        "head", "candidate_target", "candidate_query",
        "conditional_utility_uplift_bps", "delta_top5_net_bps", "delta_top1_net_bps", "promotable",
    }
    missing = required.difference(target_query_trials.columns)
    if missing:
        raise KeyError(f"target/query resume table missing {sorted(missing)}")
    configs = _configs_default(fields)
    for head in head_order:
        trials = target_query_trials.loc[target_query_trials["head"].eq(head)].copy()
        if trials.empty:
            raise ValueError(f"target/query resume has no trials for {head}")
        best = trials.sort_values(
            ["conditional_utility_uplift_bps", "delta_top5_net_bps", "delta_top1_net_bps"],
            ascending=[False, False, False], kind="stable",
        ).iloc[0]
        if bool(best["promotable"]):
            configs[head].target_name = str(best["candidate_target"])
            configs[head].query_name = str(best["candidate_query"])
            configs[head].selection_log.append({
                "stage": "target_query", "promoted": True,
                "restored_from": "target_query_conditional_trials.parquet",
                "conditional_utility_uplift_bps": float(best["conditional_utility_uplift_bps"]),
            })
        else:
            configs[head].selection_log.append({
                "stage": "target_query", "promoted": False,
                "restored_from": "target_query_conditional_trials.parquet",
            })
    return configs


def _all_head_predictions(
    frame: pd.DataFrame,
    configs: dict[str, HeadConfig],
    months: Sequence[str],
    *,
    train_start: str,
    max_train_rows: int,
    early_stopping: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = evaluation_population(frame, months)
    values: dict[str, pd.DataFrame] = {}
    meta: list[pd.DataFrame] = []
    for index, spec in enumerate(HEAD_SPECS):
        _event("head_refit_start", head=spec.name, month_count=len(months))
        config = configs[spec.name]
        prediction, fit = _fit_candidate_matrix(
            frame, config, months, train_start=train_start, max_train_rows=max_train_rows,
            seed=_head_seed(spec.name),
            early_stopping=early_stopping,
        )
        values[spec.name] = prediction
        meta.append(fit.assign(stage="all_head_refit"))
        _event("head_refit_complete", head=spec.name, month_count=len(months))
    return _head_matrix_from_predictions(population, values), pd.concat(meta, ignore_index=True)


def _write_report(
    path: Path,
    *,
    configs: dict[str, HeadConfig],
    development_control: pd.DataFrame,
    development_winner: pd.DataFrame,
    final_control: pd.DataFrame | None,
    final_winner: pd.DataFrame | None,
    head_order: pd.DataFrame,
    target_query: pd.DataFrame,
    mda_decisions: pd.DataFrame,
    hpo: pd.DataFrame,
) -> None:
    def _line(frame: pd.DataFrame, label: str) -> str:
        m = downstream_tail_summary(frame, score_column="score", tails=(.01, .02, .05))
        return (
            f"| {label} | {m.get('top1_net_bps', np.nan):+.2f} | {m.get('top2_net_bps', np.nan):+.2f} | "
            f"{m.get('top5_net_bps', np.nan):+.2f} | {m.get('top5_month_worst_net_bps', np.nan):+.2f} |"
        )
    text = [
        "# Ten-head conditional-usefulness residual funnel",
        "",
        "## Contract",
        "",
        "- Long-only strict source panel; 120 frozen causal feature fields.",
        "- Base inputs are the independently prequential `base_rank` and `base_anchor_bps` ledger.",
        "- Residual label: `policy_net_bps - base_anchor_bps`.",
        "- Policy outcome: 15-minute TP6/SL4 H12 contract with next-hour entry, SL 3 ATR, trailing activation 0.5 ATR, giveback 0.25 ATR, 100 bps cost once.",
        "- Development: May--July 2025. Final untouched confirmation: August--October 2025.",
        "- Every model train row satisfies `policy_label_available_ts < held_month_start`; held scores are transformed through a training-only score CDF.",
        "- A promotion requires positive conditional Top-1, Top-2, Top-5, Top-5 worst-month, and composite downstream utility versus the current other-nine-head ensemble.",
        "- Conditional MDA may use a deterministic candidate-only equal-month screen for speed; every proposed selected subset is then refit and gated on the full development population.",
        "",
        "## Downstream economics (net bps/trade)",
        "",
        "| Population / arm | Top-1% | Top-2% | Top-5% | Worst monthly Top-5% |",
        "|---|---:|---:|---:|---:|",
        _line(development_control, "Development control"),
        _line(development_winner, "Development frozen winner"),
    ]
    if final_control is not None and final_winner is not None:
        text.extend([_line(final_control, "Final control"), _line(final_winner, "Final frozen winner")])
    text.extend(["", "## Head contracts", "", "| Head | Target | Query | Fields | HPO selected |", "|---|---|---|---:|---|"])
    for spec in HEAD_SPECS:
        config = configs[spec.name]
        text.append(
            f"| {config.name} | {config.target_name} | {config.query_name} | {len(config.fields)} | {config.hpo_selected} |"
        )
    text.extend(["", "## Conditional selection audit", ""])
    if not head_order.empty:
        text.append("Heads were processed by leave-one-out consensus necessity.  A negative removal change means the remaining nine-head stack was better without the head.")
        text.append("")
        text.extend([
            "| Head | Necessity (bps) | Removal utility change (bps) |",
            "|---|---:|---:|",
        ])
        for row in head_order[["head", "necessity_bps", "removal_utility_change_bps"]].itertuples(index=False):
            text.append(f"| {row.head} | {float(row.necessity_bps):+.3f} | {float(row.removal_utility_change_bps):+.3f} |")
    text.extend(["", "## Search breadth", ""])
    text.append(f"- Target/query conditional arms: {len(target_query):,}.")
    text.append(f"- Conditional MDA field permutations: {int(mda_decisions.original_field_count.sum()) if not mda_decisions.empty else 0:,} field slots.")
    text.append(f"- LambdaRank HPO trials: {len(hpo):,}; completed: {int(hpo.state.eq('COMPLETE').sum()) if not hpo.empty and 'state' in hpo else 0:,}.")
    text.extend([
        "",
        "## Interpretation",
        "",
        "The final table is the only untouched confirmation.  A development promotion that fails it is retained as a negative result, not made canonical.  If no head passes the strict gate, the correct outcome is to keep the default ten-head control and investigate the residual feature/label contract rather than force an ensemble change.",
    ])
    path.write_text("\n".join(text) + "\n")


def run(
    *,
    out: Path = DEFAULT_OUT,
    source: Path = SOURCE,
    upstream: Path = UPSTREAM,
    development_months: Sequence[str] = DEFAULT_DEVELOPMENT_MONTHS,
    final_months: Sequence[str] = DEFAULT_FINAL_MONTHS,
    train_start: str = DEFAULT_TRAIN_START,
    max_train_rows: int = 60_000,
    mda_max_eval_rows: int = 90_000,
    target_query_candidates: int = 6,
    query_shortlist_limit: int = 3,
    hpo_trials: int = 6,
    stages: Sequence[str] = ("baseline", "target_query", "mda", "hpo", "final"),
    report: Path | None = None,
    resume_after_target_query: bool = False,
) -> Path:
    """Run the sequential funnel and write all artifacts under ``out``."""
    valid_stages = {"baseline", "target_query", "mda", "hpo", "final"}
    requested = tuple(dict.fromkeys(str(stage) for stage in stages))
    unknown = set(requested).difference(valid_stages)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    if not development_months:
        raise ValueError("at least one development month is required")
    out.mkdir(parents=True, exist_ok=True)
    _event("source_load_start", development_months=list(development_months), final_months=list(final_months))
    end = _month_end(max((*development_months, *final_months)))
    frame, source_fields, source_audit = load_authoritative_panel(
        source=source, upstream=upstream, start=train_start, end=end,
    )
    frozen_fields, feature_audit = freeze_feature_contract(
        frame, source_fields, development_start=min(development_months), train_start=train_start,
    )
    _event("source_load_complete", rows=len(frame), features=len(frozen_fields))
    source_audit.to_parquet(out / "source_contract_audit.parquet", index=False)
    feature_audit.to_parquet(out / "frozen_feature_contract_audit.parquet", index=False)
    development_population = evaluation_population(frame, development_months)
    target_query = pd.DataFrame()
    target_query_fit = pd.DataFrame()
    mda = pd.DataFrame()
    mda_decisions = pd.DataFrame()
    hpo = pd.DataFrame()
    hpo_fit = pd.DataFrame()
    candidates: list[tuple[str, str]] = []
    shortlist: list[str] = []
    if resume_after_target_query:
        required = [
            out / "development_control_predictions.parquet",
            out / "head_conditional_necessity.parquet",
            out / "target_query_conditional_trials.parquet",
            out / "target_query_candidates.json",
            out / "query_screen" / "query_shortlist.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("target-query resume requires " + ", ".join(missing))
        control = pd.read_parquet(out / "development_control_predictions.parquet")
        target_query = pd.read_parquet(out / "target_query_conditional_trials.parquet")
        head_order = pd.read_parquet(out / "head_conditional_necessity.parquet")
        candidates_payload = json.loads((out / "target_query_candidates.json").read_text())
        candidates = [tuple(item) for item in candidates_payload.get("candidates", [])]
        shortlist_payload = json.loads((out / "query_screen" / "query_shortlist.json").read_text())
        shortlist = [str(value) for value in shortlist_payload.get("shortlist", [])]
        order = head_order["head"].astype(str).tolist()
        configs = _restore_after_target_query(
            fields=frozen_fields, target_query_trials=target_query, head_order=order,
        )
        _event("target_query_resume_start", promoted_heads=sum(
            int(config.target_name != "resid_default_150_50" or config.query_name != "q1_cycle_4h_side")
            for config in configs.values()
        ))
        current_ranks, _ = _all_head_predictions(
            frame, configs, development_months, train_start=train_start,
            max_train_rows=max_train_rows, early_stopping=False,
        )
        _event("target_query_resume_complete", rows=len(current_ranks))
    else:
        configs = _configs_default(frozen_fields)
        baseline_ranks, baseline_fit = _all_head_predictions(
            frame, configs, development_months, train_start=train_start,
            max_train_rows=max_train_rows, early_stopping=False,
        )
        _event("development_control_complete", rows=len(baseline_ranks))
        baseline_ranks.to_parquet(out / "development_control_head_ranks.parquet", index=False)
        baseline_fit.to_parquet(out / "development_control_fit_audit.parquet", index=False)
        control = stack_scores(development_population, baseline_ranks)
        control.to_parquet(out / "development_control_predictions.parquet", index=False)
        current_ranks = baseline_ranks
        head_order = _head_order(development_population, current_ranks)
        head_order.to_parquet(out / "head_conditional_necessity.parquet", index=False)
        target_screen = _screen_targets(frame.loc[frame.month.isin(list(development_months))].copy())
        target_screen.to_parquet(out / "target_semantic_screen.parquet", index=False)
        shortlist = _screen_queries(
            frame.loc[frame.month.isin(list(development_months))].copy(),
            out=out / "query_screen", target_name="resid_default_150_50", limit=query_shortlist_limit,
        )
        candidates = _target_query_candidates(target_screen, shortlist, limit=target_query_candidates)
        _event("development_target_query_candidates", candidates=candidates)
        (out / "target_query_candidates.json").write_text(json.dumps({"candidates": candidates}, indent=2) + "\n")
        order = head_order["head"].astype(str).tolist()
        if "target_query" in requested:
            target_query, target_query_fit, configs = _conditional_target_query_stage(
                frame, development_population, current_ranks, configs, order, candidates, development_months,
                train_start=train_start, max_train_rows=max_train_rows,
            )
            target_query.to_parquet(out / "target_query_conditional_trials.parquet", index=False)
            target_query_fit.to_parquet(out / "target_query_fit_audit.parquet", index=False)
            current_ranks, _ = _all_head_predictions(
                frame, configs, development_months, train_start=train_start,
                max_train_rows=max_train_rows, early_stopping=False,
            )
            _event("target_query_stage_complete", rows=len(target_query))
    if "mda" in requested:
        mda, mda_decisions, configs = _mda_stage(
            frame, development_population, current_ranks, configs, order, development_months,
            train_start=train_start, max_train_rows=max_train_rows,
            mda_max_eval_rows=mda_max_eval_rows,
        )
        mda.to_parquet(out / "conditional_feature_mda.parquet", index=False)
        mda_decisions.to_parquet(out / "conditional_feature_selection_decisions.parquet", index=False)
        current_ranks, _ = _all_head_predictions(
            frame, configs, development_months, train_start=train_start,
            max_train_rows=max_train_rows, early_stopping=False,
        )
        _event("conditional_mda_stage_complete", rows=len(mda))
    if "hpo" in requested:
        hpo, hpo_fit, configs = _hpo_stage(
            frame, development_population, current_ranks, configs, order, development_months,
            train_start=train_start, max_train_rows=max_train_rows, trials=hpo_trials,
        )
        hpo.to_parquet(out / "per_head_conditional_hpo_trials.parquet", index=False)
        hpo_fit.to_parquet(out / "per_head_conditional_hpo_fit_audit.parquet", index=False)
        current_ranks, _ = _all_head_predictions(
            frame, configs, development_months, train_start=train_start,
            max_train_rows=max_train_rows, early_stopping=False,
        )
        _event("head_hpo_stage_complete", rows=len(hpo))
    development_winner = stack_scores(development_population, current_ranks)
    current_ranks.to_parquet(out / "development_frozen_winner_head_ranks.parquet", index=False)
    development_winner.to_parquet(out / "development_frozen_winner_predictions.parquet", index=False)
    config_manifest = {name: config.manifest() for name, config in configs.items()}
    (out / "frozen_head_configs.json").write_text(json.dumps(config_manifest, indent=2, default=str) + "\n")
    metrics = pd.concat([
        _metrics_table(control, arm="development_control"),
        _metrics_table(development_winner, arm="development_frozen_winner"),
    ], ignore_index=True)
    final_control: pd.DataFrame | None = None
    final_winner: pd.DataFrame | None = None
    if "final" in requested and final_months:
        _event("final_confirmation_start", months=list(final_months))
        final_population = evaluation_population(frame, final_months)
        final_default_configs = _configs_default(frozen_fields)
        final_control_ranks, final_control_fit = _all_head_predictions(
            frame, final_default_configs, final_months, train_start=train_start,
            max_train_rows=max_train_rows, early_stopping=False,
        )
        final_winner_ranks, final_winner_fit = _all_head_predictions(
            frame, configs, final_months, train_start=train_start,
            max_train_rows=max_train_rows, early_stopping=False,
        )
        final_control = stack_scores(final_population, final_control_ranks)
        final_winner = stack_scores(final_population, final_winner_ranks)
        final_control_ranks.to_parquet(out / "final_control_head_ranks.parquet", index=False)
        final_winner_ranks.to_parquet(out / "final_frozen_winner_head_ranks.parquet", index=False)
        final_control.to_parquet(out / "final_control_predictions.parquet", index=False)
        final_winner.to_parquet(out / "final_frozen_winner_predictions.parquet", index=False)
        pd.concat([final_control_fit.assign(arm="control"), final_winner_fit.assign(arm="frozen_winner")], ignore_index=True).to_parquet(
            out / "final_fit_audit.parquet", index=False,
        )
        metrics = pd.concat([
            metrics,
            _metrics_table(final_control, arm="final_control"),
            _metrics_table(final_winner, arm="final_frozen_winner"),
        ], ignore_index=True)
        comparison = _comparison(final_control, final_winner)
        pd.DataFrame([comparison]).to_parquet(out / "final_conditional_comparison.parquet", index=False)
        _event("final_confirmation_complete", rows=len(final_control))
    metrics.to_parquet(out / "downstream_metrics.parquet", index=False)
    correctness = {
        "status": "passed",
        "long_only_source": bool(set(frame.side_name.unique()) == {"long"}),
        "source_feature_count": len(frozen_fields),
        "development_months": list(development_months),
        "final_months": list(final_months),
        "training_label_maturity_checked": True,
        "train_test_candidate_overlap_checked": True,
        "base_prediction_provenance": "strict prequential upstream base_anchor_bps/base_rank ledger",
        "final_used_for_selection": False,
        "fixed_policy": {
            "entry_delay_hours": 1, "stop_loss_atr": 3.0,
            "trailing_activation_atr": .5, "trailing_giveback_atr": .25,
            "timeout_hours": 12, "cost_bps_once": 100.0, "source_bar_minutes": 15,
        },
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "ten_head_conditional_usefulness_funnel_v1",
        "source": str(source), "upstream_base_predictions": str(upstream),
        "side": "long", "source_feature_count": len(frozen_fields),
        "development_months": list(development_months), "final_months": list(final_months),
        "train_start": train_start, "max_train_rows": max_train_rows,
        "mda_max_eval_rows": mda_max_eval_rows,
        "target_query_candidates": candidates, "query_shortlist": list(shortlist),
        "hpo_trials_per_head": hpo_trials, "stages": list(requested),
        "heads": [asdict(spec) | {"name": spec.name} for spec in HEAD_SPECS],
        "selection_metric": "conditional downstream Top-1/2/5 net EV + Top-5 worst month; globally ranked after 0.75 base_rank + 0.25 median ten-head rank",
        "feature_selection_metric": "downstream conditional permutation usefulness while the other nine ranks stay fixed",
        "policy_outcome": "15m frozen policy: 1h entry; 3 ATR stop; 0.5 activation; 0.25 giveback; H12; 100 bps once",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = report or (out / "TEN_HEAD_CONDITIONAL_USEFULNESS_FUNNEL_20260810.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        report, configs=configs, development_control=control,
        development_winner=development_winner, final_control=final_control,
        final_winner=final_winner, head_order=head_order, target_query=target_query,
        mda_decisions=mda_decisions, hpo=hpo,
    )
    return out


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--development-months", nargs="*", default=list(DEFAULT_DEVELOPMENT_MONTHS))
    parser.add_argument("--final-months", nargs="*", default=list(DEFAULT_FINAL_MONTHS))
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--mda-max-eval-rows", type=int, default=90_000)
    parser.add_argument("--target-query-candidates", type=int, default=6)
    parser.add_argument("--query-shortlist-limit", type=int, default=3)
    parser.add_argument("--hpo-trials", type=int, default=6)
    parser.add_argument("--stages", nargs="*", default=["baseline", "target_query", "mda", "hpo", "final"])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--resume-after-target-query", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    print(run(
        out=arguments.out, source=arguments.source, upstream=arguments.upstream,
        development_months=tuple(arguments.development_months), final_months=tuple(arguments.final_months),
        train_start=arguments.train_start, max_train_rows=arguments.max_train_rows,
        mda_max_eval_rows=arguments.mda_max_eval_rows,
        target_query_candidates=arguments.target_query_candidates,
        query_shortlist_limit=arguments.query_shortlist_limit,
        hpo_trials=arguments.hpo_trials, stages=tuple(arguments.stages), report=arguments.report,
        resume_after_target_query=arguments.resume_after_target_query,
    ))
