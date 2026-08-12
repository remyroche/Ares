#!/usr/bin/env python3
"""Sequential long-only self-distillation funnel for strict-R3.

The first implemented phase screens D0--D4 on the base layer.  Teacher ranks
come from the previously frozen strict-prequential ledger and are never rebuilt
from a held month or timestamp-local population.  Outputs are intentionally
compact: the 120-field source contract remains in its immutable source panel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", message="X does not have valid feature names")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    BASE_PARAMS,
    RANK_PARAMS,
    RESIDUAL_CAPS,
    RESIDUAL_WEIGHT_MODES,
    SEED,
    ScoreReference,
    _fit_medians,
    _numeric_matrix,
    fit_policy_net_map,
    residual_grades,
)
from extreme_price_movements.strict_r3_self_distillation import (  # noqa: E402
    DistillationWeightSpec,
    build_distillation_weights,
    initial_screen_specs,
)
from scripts.run_tp6_sl4_exact170_canonical_consensus import _load_contract  # noqa: E402


SOURCE_PANEL = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_source_panel_targetfree_long_2023_aug7_2026_20260809_v2/"
    "canonical_source_panel.parquet"
)
TEACHER_LEDGER = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_20260809_v1/"
    "prequential_stack_ledger.parquet"
)
BASE_TRAIN_CAP = 240_000
OPTIMISED_POLICY = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_optimised_policy_replay_targetfree_long_2025_aug7_2026_20260809_v1/"
    "candidate_policy_outcomes.parquet"
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _base_fields() -> list[str]:
    fields = [str(value) for value in _load_contract()["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("strict-R3 base screen requires the frozen 120-field contract")
    missing = sorted(set(fields) - set(pq.ParquetFile(SOURCE_PANEL).schema.names))
    if missing:
        raise ValueError(f"source panel lacks frozen base fields: {missing[:10]}")
    return fields


def _read_source(start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    # One year is far wider than the recent 240k-row cap, while still keeping
    # this research runner bounded in memory.
    source_start = start - pd.DateOffset(years=1)
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "r3_class", "r3_label_available_ts", *fields,
    ]
    frame = pd.read_parquet(
        SOURCE_PANEL, columns=columns,
        filters=[("__decision_ts__", ">=", source_start), ("__decision_ts__", "<", end)],
    )
    for column in ("__ts__", "__decision_ts__", "r3_label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq("long")].copy()
    teacher = pd.read_parquet(
        TEACHER_LEDGER,
        columns=[
            "candidate_id", "prequential_base_rank42", "prequential_residual_rank",
            "prequential_upstream", "stack_is_prequential",
        ],
        filters=[("__decision_ts__", ">=", source_start), ("__decision_ts__", "<", end)],
    )
    teacher = teacher.loc[teacher["stack_is_prequential"].fillna(False).astype(bool)].drop(
        columns="stack_is_prequential"
    )
    if teacher["candidate_id"].duplicated().any():
        raise ValueError("teacher ledger has duplicate candidate identities")
    frame = frame.merge(teacher, on="candidate_id", how="left", validate="one_to_one")
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("base source is empty or duplicated")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _fold_index(start: pd.Timestamp) -> int:
    anchor = pd.Timestamp("2024-01-01", tz="UTC")
    return (start.year - anchor.year) * 12 + start.month - anchor.month


def _safe_predict_probabilities(
    model: LGBMClassifier, frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = model.predict_proba(_numeric_matrix(frame, fields, medians))
    lookup = {int(label): index for index, label in enumerate(model.classes_)}
    adverse = probability[:, lookup.get(0, 0)]
    weak = probability[:, lookup.get(1, min(1, probability.shape[1] - 1))]
    clear = probability[:, lookup.get(2, probability.shape[1] - 1)]
    return adverse, weak, clear, clear - 0.5 * adverse


def _fit_base_fold(
    frame: pd.DataFrame,
    *,
    held_start: pd.Timestamp,
    held_end: pd.Timestamp,
    fields: Sequence[str],
    spec: DistillationWeightSpec,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    train = frame.loc[
        frame["r3_label_available_ts"].lt(held_start) & frame["r3_class"].notna()
    ].sort_values("r3_label_available_ts", kind="stable").tail(BASE_TRAIN_CAP).copy()
    reference = frame.loc[
        frame["__decision_ts__"].ge(held_start - pd.Timedelta(days=42))
        & frame["__decision_ts__"].lt(held_start)
    ].copy()
    held = frame.loc[
        frame["__decision_ts__"].ge(held_start) & frame["__decision_ts__"].lt(held_end)
    ].copy()
    if len(train) < 100 or len(reference) < 2 or held.empty or train["r3_class"].nunique() < 2:
        raise ValueError(f"{spec.name} {held_start:%Y-%m}: insufficient base support")
    weight, weight_audit = build_distillation_weights(
        train,
        teacher_rank_column="prequential_base_rank42",
        layer="base",
        spec=spec,
    )
    medians = _fit_medians(train, fields)
    params = {**BASE_PARAMS, "random_state": SEED + _fold_index(held_start)}
    model = LGBMClassifier(**params).fit(
        _numeric_matrix(train, fields, medians),
        train["r3_class"].astype(int).to_numpy(),
        sample_weight=weight,
    )
    _, _, _, reference_score = _safe_predict_probabilities(model, reference, fields, medians)
    adverse, weak, clear, score = _safe_predict_probabilities(model, held, fields, medians)
    output = held[[
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "r3_class", "prequential_base_rank42",
    ]].copy()
    output["p_adverse"] = adverse.astype(np.float32)
    output["p_weak"] = weak.astype(np.float32)
    output["p_clear"] = clear.astype(np.float32)
    output["base_score"] = score.astype(np.float32)
    output["base_rank42"] = ScoreReference.fit(
        reference_score, source=f"{spec.name}_{held_start:%Y-%m}_same_model_prior42",
    ).cdf(score).astype(np.float32)
    output["arm"] = spec.name
    output["held_month"] = held_start.strftime("%Y-%m")
    importance = pd.DataFrame(
        {
            "arm": spec.name,
            "held_month": held_start.strftime("%Y-%m"),
            "feature": list(fields),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
        }
    )
    audit = {
        "arm": spec.name,
        "held_month": held_start.strftime("%Y-%m"),
        "fit_rows": int(len(train)),
        "reference_rows": int(len(reference)),
        "held_rows": int(len(held)),
        "fit_max_label_available_ts": train["r3_label_available_ts"].max(),
        "held_start": held_start,
        "held_outcomes_consumed": False,
        "teacher_rank_reference": "prior strict-R3 OOF/prequential global rank42",
        **{f"weight__{key}": value for key, value in weight_audit.items()},
    }
    return output, audit, importance


def _rank_ic(frame: pd.DataFrame) -> float:
    values: list[float] = []
    weights: list[int] = []
    for _, group in frame.groupby("__decision_ts__", sort=False):
        if len(group) < 3 or group["r3_class"].nunique() < 2 or group["base_score"].nunique() < 2:
            continue
        correlation = group["base_score"].rank(method="average").corr(
            group["r3_class"].rank(method="average"), method="pearson",
        )
        if np.isfinite(correlation):
            values.append(float(correlation))
            weights.append(len(group))
    return float(np.average(values, weights=weights)) if values else float("nan")


def _base_metrics(prediction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    for arm, arm_frame in prediction.groupby("arm", sort=True):
        for month, local in [("pooled", arm_frame), *list(arm_frame.groupby("held_month", sort=True))]:
            valid = local.loc[local["r3_class"].notna()].copy()
            y = valid["r3_class"].astype(int).to_numpy()
            probability = valid[["p_adverse", "p_weak", "p_clear"]].to_numpy(float)
            probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
            clear = y == 2
            row: dict[str, object] = {
                "arm": arm,
                "month": month,
                "rows": int(len(valid)),
                "clear_rows": int(clear.sum()),
                "within_query_rank_ic": _rank_ic(valid),
                "multiclass_log_loss": float(log_loss(y, probability, labels=[0, 1, 2])),
                "multiclass_brier": float(np.square(probability - np.eye(3)[y]).sum(axis=1).mean()),
                "top5_clear_uplift": float(
                    valid.nlargest(max(1, int(math.ceil(0.05 * len(valid)))), "base_score")["r3_class"].eq(2).mean()
                    - clear.mean()
                ),
            }
            for fraction in (0.30, 0.40):
                selected = valid.nlargest(max(1, int(math.ceil(fraction * len(valid)))), "base_score")
                row[f"top{int(fraction * 100)}_clear_recall"] = float(
                    selected["r3_class"].eq(2).sum() / max(int(clear.sum()), 1)
                )
            ordered = valid.sort_values(["base_score", "candidate_id"], kind="stable")
            ordered["decile"] = np.minimum(np.arange(len(ordered)) * 10 // max(len(ordered), 1), 9)
            decile_rate = ordered.groupby("decile", sort=True)["r3_class"].apply(lambda x: x.eq(2).mean())
            row["clear_decile_violations"] = int((np.diff(decile_rate.to_numpy(float)) < -1e-12).sum())
            (pooled if month == "pooled" else monthly).append(row)
    return pd.DataFrame(pooled), pd.DataFrame(monthly)


def _overlap(prediction: pd.DataFrame) -> pd.DataFrame:
    control = prediction.loc[prediction["arm"].eq("D0")]
    rows: list[dict[str, object]] = []
    for arm, frame in prediction.groupby("arm", sort=True):
        for tail in (0.01, 0.02, 0.05):
            count = max(1, int(math.ceil(tail * len(control))))
            a = set(control.nlargest(count, "base_score")["candidate_id"])
            b = set(frame.nlargest(count, "base_score")["candidate_id"])
            rows.append({
                "arm": arm, "tail": tail, "control_selected": len(a),
                "arm_selected": len(b), "intersection": len(a & b),
                "jaccard": len(a & b) / max(len(a | b), 1),
            })
    return pd.DataFrame(rows)


def _read_residual_ledger(start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "prequential_p_adverse", "prequential_p_weak", "prequential_p_clear",
        "prequential_base_score", "prequential_base_rank42",
        "prequential_base_anchor_bps", "prequential_consensus_rank",
        "prequential_residual_rank", "prequential_upstream", "stack_is_prequential",
        *fields,
    ]
    frame = pd.read_parquet(
        TEACHER_LEDGER, columns=columns,
        filters=[("__decision_ts__", ">=", pd.Timestamp("2024-01-01", tz="UTC")),
                 ("__decision_ts__", "<", end)],
    )
    for column in ("__ts__", "__decision_ts__", "policy_label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.loc[
        frame["side_name"].astype(str).str.lower().eq("long")
        & frame["stack_is_prequential"].fillna(False).astype(bool)
    ].copy()
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("strict-prequential residual ledger is empty or duplicated")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _apply_policy_outcome_overrides(
    frame: pd.DataFrame,
    *,
    path: Path,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Make the declared policy artifact authoritative wherever it overlaps."""

    columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ]
    override = pd.read_parquet(path, columns=columns)
    override["policy_label_available_ts"] = pd.to_datetime(
        override["policy_label_available_ts"], utc=True,
    )
    if override.empty or override["candidate_id"].duplicated().any():
        raise ValueError("policy outcome override is empty or duplicated")
    renamed = override.rename(
        columns={column: f"{column}__policy_override" for column in columns[1:]},
    )
    output = frame.merge(
        renamed, on="candidate_id", how="left", validate="one_to_one", indicator="__policy_join__",
    )
    matched = output["__policy_join__"].eq("both")
    for field in columns[1:]:
        source = f"{field}__policy_override"
        if field == "policy_path_valid":
            output[field] = pd.array(output[field], dtype="boolean")
            output.loc[matched, field] = pd.array(output.loc[matched, source], dtype="boolean")
        else:
            output.loc[matched, field] = output.loc[matched, source]
        output = output.drop(columns=source)
    output = output.drop(columns="__policy_join__")
    held = output["__decision_ts__"].ge(evaluation_start) & output["__decision_ts__"].lt(
        evaluation_end
    )
    if not matched.loc[held].all():
        raise ValueError(
            f"policy artifact covers {int(matched.loc[held].sum())}/{int(held.sum())} held rows"
        )
    return output, {
        "policy_override_rows": int(matched.sum()),
        "policy_override_held_coverage": float(matched.loc[held].mean()),
        "policy_override_authoritative_on_overlap": True,
    }


def _apply_base_prediction_overrides(
    frame: pd.DataFrame,
    *,
    path: Path,
    arm: str,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replace upstream base outputs with one strict-prequential base arm.

    Residual anchors and targets are intentionally recomputed later from the
    overridden rank.  This prevents a distilled base from inheriting the
    control base's score-to-policy map or residual labels.
    """

    override = pd.read_parquet(path)
    if "arm" in override:
        override = override.loc[override["arm"].astype(str).eq(arm)].copy()
    rename = {
        "p_adverse": "prequential_p_adverse",
        "p_weak": "prequential_p_weak",
        "p_clear": "prequential_p_clear",
        "base_score": "prequential_base_score",
        "base_rank42": "prequential_base_rank42",
    }
    missing = sorted(set(rename) - set(override))
    if missing:
        raise ValueError(f"base override lacks required fields: {missing}")
    override = override[["candidate_id", *rename]].rename(columns=rename)
    if override.empty or override["candidate_id"].duplicated().any():
        raise ValueError(f"base override {arm!r} is empty or duplicated")
    renamed = override.rename(
        columns={field: f"{field}__base_override" for field in rename.values()},
    )
    output = frame.merge(
        renamed, on="candidate_id", how="left", validate="one_to_one", indicator="__base_join__",
    )
    matched = output["__base_join__"].eq("both")
    for field in rename.values():
        source = f"{field}__base_override"
        output.loc[matched, field] = output.loc[matched, source]
        output = output.drop(columns=source)
    output = output.drop(columns="__base_join__")
    required = output["__decision_ts__"].ge(pd.Timestamp("2024-01-01", tz="UTC")) & output[
        "__decision_ts__"
    ].lt(evaluation_end)
    held = output["__decision_ts__"].ge(evaluation_start) & output["__decision_ts__"].lt(
        evaluation_end
    )
    if not matched.loc[required].all() or not matched.loc[held].all():
        raise ValueError(
            f"base override {arm!r} covers {int(matched.loc[required].sum())}/"
            f"{int(required.sum())} required prequential rows"
        )
    return output, {
        "base_override": str(path),
        "base_override_arm": arm,
        "base_override_rows": int(matched.sum()),
        "base_override_required_coverage": float(matched.loc[required].mean()),
        "base_override_held_coverage": float(matched.loc[held].mean()),
        "anchor_and_residual_recomputed": True,
    }


def _fit_distilled_ranker(
    frame: pd.DataFrame,
    fields: Sequence[str],
    grade: np.ndarray,
    *,
    cap: int,
    mode: str,
    medians: np.ndarray,
    spec: DistillationWeightSpec,
) -> tuple[LGBMRanker, ScoreReference, dict[str, object]]:
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True)
    query = timestamp.dt.floor("4h").astype(str) + "|" + frame["side_name"].astype(str)
    counts = query.value_counts()
    keep = query.map(counts).ge(2).to_numpy()
    positions = np.flatnonzero(keep)
    order = np.argsort(query.iloc[positions].to_numpy(), kind="stable")
    positions = positions[order]
    ordered_query = query.iloc[positions].to_numpy()
    _, group_sizes = np.unique(ordered_query, return_counts=True)
    existing = np.ones(len(positions), dtype=float)
    if mode == "equal_month":
        months = timestamp.dt.strftime("%Y-%m").iloc[positions]
        frequency = months.value_counts()
        existing = months.map(lambda month: 1.0 / float(frequency.loc[month])).to_numpy(float)
        existing *= len(existing) / max(existing.sum(), 1e-12)
    local = frame.iloc[positions].copy()
    local["policy_residual_bps"] = frame["policy_residual_bps"].to_numpy(float)[positions]
    weights, weight_audit = build_distillation_weights(
        local,
        teacher_rank_column="prequential_residual_rank",
        layer="residual",
        spec=spec,
        existing_weight=existing,
    )
    matrix = _numeric_matrix(frame, fields[:cap], medians[:cap])
    model = LGBMRanker(**RANK_PARAMS)
    model.fit(matrix[positions], grade[positions], group=group_sizes, sample_weight=weights)
    raw = model.predict(matrix)
    reference = ScoreReference.fit(
        raw, source=f"{spec.name}_residual_training_distribution_cap{cap}_{mode}",
    )
    return model, reference, weight_audit


def _cap_complete_queries(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Deterministically screen complete 4h query groups with equal-month reach."""

    if cap <= 0 or len(frame) <= cap:
        return frame.copy()
    timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True)
    query = timestamp.dt.floor("4h").astype(str) + "|" + frame["side_name"].astype(str)
    month = timestamp.dt.strftime("%Y-%m")
    meta = pd.DataFrame({"query": query, "month": month})
    groups = meta.groupby(["month", "query"], sort=True).size().rename("rows").reset_index()
    groups["hash"] = pd.util.hash_pandas_object(groups["query"], index=False).to_numpy(np.uint64)
    quota = int(math.ceil(cap / max(groups["month"].nunique(), 1)))
    chosen: list[str] = []
    for _, block in groups.groupby("month", sort=True):
        block = block.sort_values(["hash", "query"], kind="stable").copy()
        block["cumulative"] = block["rows"].cumsum()
        local = block.loc[block["cumulative"].le(quota), "query"].tolist()
        if not local:
            local = block["query"].head(1).tolist()
        chosen.extend(local)
    selected = frame.loc[query.isin(set(chosen))].copy()
    if len(selected) <= cap:
        return selected.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    selected_query = (
        pd.to_datetime(selected["__decision_ts__"], utc=True).dt.floor("4h").astype(str)
        + "|" + selected["side_name"].astype(str)
    )
    sizes = selected_query.value_counts().rename_axis("query").rename("rows").reset_index()
    sizes["hash"] = pd.util.hash_pandas_object(sizes["query"], index=False).to_numpy(np.uint64)
    sizes = sizes.sort_values(["hash", "query"], kind="stable")
    sizes["cumulative"] = sizes["rows"].cumsum()
    keep_queries = set(sizes.loc[sizes["cumulative"].le(cap), "query"])
    return selected.loc[selected_query.isin(keep_queries)].sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )


def _fit_residual_fold(
    frame: pd.DataFrame,
    *,
    held_start: pd.Timestamp,
    held_end: pd.Timestamp,
    fields: Sequence[str],
    spec: DistillationWeightSpec,
    model_cap: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    held = frame.loc[
        frame["__decision_ts__"].ge(held_start) & frame["__decision_ts__"].lt(held_end)
    ].copy()
    earlier = frame.loc[
        frame["__decision_ts__"].lt(held_start)
        & frame["policy_label_available_ts"].lt(held_start)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["prequential_base_rank42"], errors="coerce"))
    ].copy()
    if held.empty or len(earlier) < 100:
        raise ValueError(f"{spec.name} {held_start:%Y-%m}: insufficient residual support")
    policy_map = fit_policy_net_map(
        earlier["prequential_base_rank42"], earlier["policy_net_bps"],
    )
    earlier["refit_anchor_bps"] = policy_map.predict(earlier["prequential_base_rank42"])
    earlier["policy_residual_bps"] = (
        earlier["policy_net_bps"].to_numpy(float) - earlier["refit_anchor_bps"].to_numpy(float)
    )
    model_frame = _cap_complete_queries(earlier, model_cap)
    grade = residual_grades(model_frame["policy_residual_bps"])
    medians = _fit_medians(model_frame, fields)
    held_matrix = _numeric_matrix(held, fields, medians)
    ranks: list[np.ndarray] = []
    audits: list[dict[str, object]] = []
    for cap in RESIDUAL_CAPS:
        for mode in RESIDUAL_WEIGHT_MODES:
            model, reference, weight_audit = _fit_distilled_ranker(
                model_frame, fields, grade, cap=cap, mode=mode,
                medians=medians, spec=spec,
            )
            ranks.append(reference.cdf(model.predict(held_matrix[:, :cap])))
            audits.append({
                "arm": spec.name, "held_month": held_start.strftime("%Y-%m"),
                "cap": cap, "weight_mode": mode,
                "map_fit_rows": len(earlier), "fit_rows": len(model_frame),
                "model_row_cap": model_cap, "held_rows": len(held),
                "fit_max_policy_label_available_ts": earlier["policy_label_available_ts"].max(),
                "held_start": held_start, "held_outcomes_consumed": False,
                **{f"weight__{key}": value for key, value in weight_audit.items()},
            })
    output = held[[
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "prequential_p_adverse", "prequential_p_weak", "prequential_p_clear",
        "prequential_base_score", "prequential_base_rank42",
    ]].copy()
    output["prequential_base_anchor_bps"] = policy_map.predict(
        held["prequential_base_rank42"]
    ).astype(np.float32)
    output["prequential_consensus_rank"] = np.nanmedian(np.column_stack(ranks), axis=1).astype(np.float32)
    output["prequential_residual_rank"] = output["prequential_consensus_rank"]
    output["prequential_upstream"] = (
        0.75 * output["prequential_base_rank42"] + 0.25 * output["prequential_consensus_rank"]
    ).astype(np.float32)
    output["stack_is_prequential"] = True
    output["arm"] = spec.name
    output["held_month"] = held_start.strftime("%Y-%m")
    return output, audits


def _read_policy_outcomes(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    ]
    return pd.read_parquet(
        path, columns=columns,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )


def _economic_metrics(prediction: pd.DataFrame, policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = prediction.merge(policy, on="candidate_id", how="inner", validate="many_to_one")
    frame = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()
    global_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for arm, local in frame.groupby("arm", sort=True):
        for tail in (0.005, 0.01, 0.02, 0.05, 0.10):
            selected = local.nlargest(max(1, int(math.ceil(tail * len(local)))), "prequential_upstream")
            global_rows.append({
                "arm": arm, "tail": tail, "population_rows": len(local),
                "trades": len(selected),
                "gross_bps_per_trade": float(selected["policy_gross_bps"].mean()),
                "net_bps_per_trade": float(selected["policy_net_bps"].mean()),
                "positive_rate": float(selected["policy_net_bps"].gt(0).mean()),
            })
            for month, block in selected.groupby("held_month", sort=True):
                monthly_rows.append({
                    "arm": arm, "tail": tail, "month": month, "trades": len(block),
                    "gross_bps_per_trade": float(block["policy_gross_bps"].mean()),
                    "net_bps_per_trade": float(block["policy_net_bps"].mean()),
                    "positive_rate": float(block["policy_net_bps"].gt(0).mean()),
                })
    return pd.DataFrame(global_rows), pd.DataFrame(monthly_rows)


def _run_residual_screen(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
    power: float,
    policy_path: Path,
    out_dir: Path,
    model_cap: int,
    arms: Sequence[str] | None = None,
    base_predictions: Path | None = None,
    base_arm: str | None = None,
) -> dict[str, object]:
    frame = _read_residual_ledger(start, end, fields)
    frame, policy_override_audit = _apply_policy_outcome_overrides(
        frame, path=policy_path, evaluation_start=start, evaluation_end=end,
    )
    override_audit: dict[str, object] = {}
    if base_predictions is not None:
        if not base_arm:
            raise ValueError("--base-arm is required with --base-predictions")
        frame, override_audit = _apply_base_prediction_overrides(
            frame, path=base_predictions, arm=base_arm,
            evaluation_start=start, evaluation_end=end,
        )
    specs = initial_screen_specs(power=power)
    if arms:
        requested = set(arms)
        specs = tuple(spec for spec in specs if spec.name in requested)
        missing = sorted(requested - {spec.name for spec in specs})
        if missing:
            raise ValueError(f"unknown residual arms: {missing}")
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for spec in specs:
        print(json.dumps({"event": "arm_start", "arm": spec.name}), flush=True)
        for held_start in pd.date_range(start, end, freq="MS", inclusive="left"):
            output, local_audits = _fit_residual_fold(
                frame, held_start=held_start,
                held_end=min(held_start + pd.offsets.MonthBegin(1), end),
                fields=fields, spec=spec,
                model_cap=model_cap,
            )
            predictions.append(output)
            audits.extend(local_audits)
            print(json.dumps({
                "event": "fold_complete", "arm": spec.name,
                "month": held_start.strftime("%Y-%m"), "rows": len(output),
            }), flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    policy = _read_policy_outcomes(policy_path, start, end)
    global_metrics, monthly_metrics = _economic_metrics(prediction, policy)
    prediction.to_parquet(out_dir / "residual_oof_score_overrides.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out_dir / "weight_lineage_audit.parquet", index=False)
    global_metrics.to_parquet(out_dir / "upstream_economic_metrics.parquet", index=False)
    monthly_metrics.to_parquet(out_dir / "upstream_monthly_global_tail_contribution.parquet", index=False)
    return {
        "schema": "strict_r3_self_distillation_residual_screen_v1",
        "arms": [asdict(spec) for spec in specs],
        "policy_outcomes": str(policy_path),
        "target": "canonical policy_net_bps - causal base anchor; grades <=-150/-50/+50/+150",
        "heads": "10 LambdaRank heads: caps 40/60/80/100/120 x ordinary/equal-month",
        "screening_model_row_cap": model_cap,
        **policy_override_audit,
        **override_audit,
    }


def _base_specs(args: argparse.Namespace) -> tuple[DistillationWeightSpec, ...]:
    if args.phase == "base_screen":
        return initial_screen_specs(power=args.power)
    if args.phase == "base_refine_positive":
        return (
            DistillationWeightSpec("D0"),
            *tuple(
                DistillationWeightSpec(
                    f"D2_top{int(round(fraction * 100)):02d}_boost{args.positive_boost:g}",
                    positive_top_fraction=fraction,
                    positive_boost=args.positive_boost,
                )
                for fraction in args.positive_top_fractions
            ),
        )
    if args.phase == "base_refine_positive_boost":
        fraction = args.positive_top_fractions[0]
        return (
            DistillationWeightSpec("D0"),
            *tuple(
                DistillationWeightSpec(
                    f"D2_top{int(round(fraction * 100)):02d}_boost{boost:g}",
                    positive_top_fraction=fraction,
                    positive_boost=boost,
                )
                for boost in args.positive_boosts
            ),
        )
    if args.phase == "base_refine_power":
        return (
            DistillationWeightSpec("D0"),
            *tuple(
                DistillationWeightSpec(
                    f"D1_power{power:g}", use_score_weight=True, score_power=power,
                )
                for power in args.score_powers
            ),
        )
    raise ValueError(f"unsupported base phase: {args.phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "base_screen", "base_refine_positive", "base_refine_positive_boost",
            "base_refine_power", "residual_screen",
        ),
        default="base_screen",
    )
    parser.add_argument("--evaluation-start", default="2025-01-01")
    parser.add_argument("--evaluation-end", default="2025-08-01")
    parser.add_argument("--power", type=float, default=1.5)
    parser.add_argument(
        "--positive-top-fractions", type=float, nargs="+",
        default=[0.60, 0.50, 0.40, 0.30, 0.20],
    )
    parser.add_argument("--positive-boost", type=float, default=1.5)
    parser.add_argument("--positive-boosts", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    parser.add_argument("--score-powers", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--policy-outcomes", type=Path, default=OPTIMISED_POLICY)
    parser.add_argument("--model-cap", type=int, default=80_000)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--base-predictions", type=Path, default=None)
    parser.add_argument("--base-arm", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start, end = _utc(args.evaluation_start), _utc(args.evaluation_end)
    fields = _base_fields()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.phase == "residual_screen":
        manifest = _run_residual_screen(
            start=start, end=end, fields=fields, power=args.power,
            policy_path=args.policy_outcomes, out_dir=args.out_dir,
            model_cap=args.model_cap,
            arms=args.arms, base_predictions=args.base_predictions,
            base_arm=args.base_arm,
        )
        manifest.update({
            "side": "long", "phase": args.phase,
            "evaluation_start": start.isoformat(),
            "evaluation_end_exclusive": end.isoformat(),
            "teacher": str(TEACHER_LEDGER),
            "teacher_semantics": "own prior residual/consensus OOF/prequential global rank",
            "base_fields": list(fields), "rank_params": RANK_PARAMS,
            "status": "complete",
        })
        (args.out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
        print(json.dumps({"event": "complete", "out": str(args.out_dir)}), flush=True)
        return
    source = _read_source(start, end, fields)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    importance: list[pd.DataFrame] = []
    specs = _base_specs(args)
    months = list(pd.date_range(start, end, freq="MS", inclusive="left"))
    for spec in specs:
        print(json.dumps({"event": "arm_start", "arm": spec.name}), flush=True)
        for held_start in months:
            output, audit, fold_importance = _fit_base_fold(
                source, held_start=held_start,
                held_end=min(held_start + pd.offsets.MonthBegin(1), end),
                fields=fields, spec=spec,
            )
            predictions.append(output)
            audits.append(audit)
            importance.append(fold_importance)
            print(json.dumps({
                "event": "fold_complete", "arm": spec.name,
                "month": held_start.strftime("%Y-%m"), "rows": len(output),
            }), flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    pooled, monthly = _base_metrics(prediction)
    prediction.to_parquet(args.out_dir / "base_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "weight_lineage_audit.parquet", index=False)
    pd.concat(importance, ignore_index=True).to_parquet(
        args.out_dir / "feature_importance_by_fold.parquet", index=False,
    )
    pooled.to_parquet(args.out_dir / "base_metrics_pooled.parquet", index=False)
    monthly.to_parquet(args.out_dir / "base_metrics_monthly.parquet", index=False)
    _overlap(prediction).to_parquet(args.out_dir / "selected_overlap.parquet", index=False)
    manifest = {
        "schema": f"strict_r3_self_distillation_{args.phase}_v1",
        "side": "long",
        "phase": args.phase,
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "teacher": str(TEACHER_LEDGER),
        "teacher_semantics": "prior strict-R3 OOF/prequential rank; globally referenced, never held-month or per-timestamp ranked",
        "source": str(SOURCE_PANEL),
        "base_fields": list(fields),
        "base_params": BASE_PARAMS,
        "base_train_cap": BASE_TRAIN_CAP,
        "arms": [asdict(spec) for spec in specs],
        "weight_cap": [0.25, 4.0],
        "status": "complete",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
