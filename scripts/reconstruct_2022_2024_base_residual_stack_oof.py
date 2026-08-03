#!/usr/bin/env python3
"""Reconstruct research-only base -> residual-alpha -> EV OOF scores for 2022-24.

The early inverse population and the later frozen-PF population are deliberately
kept as separate lineages.  Within each lineage, every score is produced by a
side-local model that excludes the complete held-out calendar block, with a
12-hour overlap purge.  Walk-forward validation is not required for this
diagnostic, but held-out labels are never used for feature selection, fitting,
or EV calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "historical_base_residual_stack_calendar_block_oof_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
ALPHA_TARGET = "__reconstructed_soft_alpha_12h__"
NET_TARGET = "execution_net_ev_12h"
GROSS_TARGET = "execution_gross_ev_12h"
COST_TARGET = "execution_cost_return"
PURGE_HOURS = 12
SEED = 20260730

DEFAULT_INVERSE = (
    ROOT
    / "data_perp/artifacts/jan_jul_2022_inverse_pi_exact_id_research_panel_"
    "20260730_v1/inverse_exact_id_research_panel.parquet"
)
DEFAULT_PF_STAGE = (
    ROOT
    / "data_perp/artifacts/failure_2022_2023_pf_exact1m_request_stage_"
    "20260730_v1/staged_candidates.parquet"
)
DEFAULT_PF_LABELS = (
    ROOT
    / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_"
    "20260730_v1/joined_multitask_labels.parquet"
)
DEFAULT_FULL_2024_STAGE = (
    ROOT
    / "data_perp/artifacts/failure_2024_transition_exact1m_request_stage_"
    "20260730_v2/staged_candidates.parquet"
)
DEFAULT_FULL_2024_LABELS = (
    ROOT
    / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_"
    "20260730_v1/joined_multitask_labels.parquet"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_"
    "20260730_v3"
)

FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "future",
    "label",
    "execution_",
    "exec_",
    "first_touch",
    "full_path",
    "hit_probability",
    "ev_after",
    "clean_exec",
    "dirty_positive",
    "timeout",
    "selected_",
    "historical_rank",
    "score_meta",
    "base_score",
    "prediction_evidence",
    "evidence_scope",
    "archetype_policy",
    "policy_archetype",
    "barrier",
)

INVERSE_FOLDS = (
    ("2022-01-01T00:00:00Z", "2022-02-12T10:00:00Z"),
    ("2022-02-12T10:00:00Z", "2022-03-26T20:00:00Z"),
    ("2022-03-26T20:00:00Z", "2022-05-08T06:00:00Z"),
    ("2022-05-08T06:00:00Z", "2022-06-19T15:00:00Z"),
    ("2022-06-19T15:00:00Z", "2022-08-01T00:00:00Z"),
)
PF_FOLDS = (
    ("2022-08-30T00:00:00Z", "2022-11-01T00:00:00Z"),
    ("2022-11-01T00:00:00Z", "2023-02-01T00:00:00Z"),
    ("2023-02-01T00:00:00Z", "2023-05-01T00:00:00Z"),
    ("2023-05-01T00:00:00Z", "2023-08-01T00:00:00Z"),
    ("2023-08-01T00:00:00Z", "2023-11-01T00:00:00Z"),
    ("2023-11-01T00:00:00Z", "2024-02-01T00:00:00Z"),
    ("2024-02-01T00:00:00Z", "2024-04-01T00:00:00Z"),
    ("2024-04-01T00:00:00Z", "2024-06-01T00:00:00Z"),
    ("2024-06-01T00:00:00Z", "2024-08-01T00:00:00Z"),
    ("2024-08-01T00:00:00Z", "2024-10-01T00:00:00Z"),
    ("2024-10-01T00:00:00Z", "2024-12-01T00:00:00Z"),
    ("2024-12-01T00:00:00Z", "2025-01-01T00:00:00Z"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _soft_alpha(first_event: pd.Series) -> np.ndarray:
    mapping = {
        "favorable_first": 1.0,
        "timeout": 0.5,
        "adverse_first_or_conflict": 0.0,
    }
    result = first_event.astype(str).map(mapping)
    if result.isna().any():
        raise ValueError("unknown soft triple-barrier first-event value")
    return result.to_numpy(np.float32)


def _is_numeric(field: pa.Field) -> bool:
    return (
        pa.types.is_floating(field.type)
        or pa.types.is_integer(field.type)
        or pa.types.is_boolean(field.type)
    )


def eligible_raw_columns(path: Path) -> list[str]:
    schema = pq.read_schema(path)
    result = []
    for field in schema:
        name = field.name
        lower = name.lower()
        if not _is_numeric(field):
            continue
        if name.startswith("__") or any(token in lower for token in FORBIDDEN_FEATURE_TOKENS):
            continue
        result.append(name)
    if len(result) < 100:
        raise ValueError(f"too few eligible PIT columns in {path}: {len(result)}")
    return result


def _read_source_rows(stage: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(stage["source_shard_path"].astype(str).unique())
    common: set[str] | None = None
    for raw_path in paths:
        columns = set(eligible_raw_columns(Path(raw_path)))
        common = columns if common is None else common & columns
    feature_columns = sorted(common or ())
    if len(feature_columns) < 100:
        raise ValueError(f"historical feature intersection too small: {len(feature_columns)}")

    parts: list[pd.DataFrame] = []
    for raw_path, rows in stage.groupby("source_shard_path", sort=True):
        source = pd.read_parquet(raw_path, columns=feature_columns)
        positions = pd.to_numeric(rows["source_row_number"], errors="raise").astype(int)
        if positions.min() < 0 or positions.max() >= len(source):
            raise ValueError(f"source row out of bounds for {raw_path}")
        selected = source.iloc[positions.to_numpy()].reset_index(drop=True)
        selected["candidate_id"] = rows["candidate_id"].to_numpy()
        parts.append(selected)
    features = pd.concat(parts, ignore_index=True)
    if features["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate identity in reconstructed source features")
    return features, feature_columns


def load_pf_population(
    stages: Sequence[Path], labels: Sequence[Path]
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    stage_parts = []
    for path in stages:
        part = pd.read_parquet(
            path,
            columns=[
                "candidate_id",
                "signal_timestamp",
                "symbol",
                "side_name",
                "base_score",
                "source_row_number",
                "source_shard_path",
            ],
        )
        if part["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate candidate_id within stage {path}")
        stage_parts.append(part)
    stage = pd.concat(stage_parts, ignore_index=True)
    if stage["candidate_id"].duplicated().any():
        raise ValueError("candidate overlap across staged populations")
    feature_frame, feature_columns = _read_source_rows(stage)
    label_columns = [
        *IDENTITY,
        "__soft_tb_first_event__",
        NET_TARGET,
        GROSS_TARGET,
        COST_TARGET,
    ]
    outcome_parts = []
    for path in labels:
        part = pd.read_parquet(path, columns=label_columns)
        if part["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate candidate_id within labels {path}")
        outcome_parts.append(part)
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    if outcomes["candidate_id"].duplicated().any():
        raise ValueError("candidate overlap across supplied label artifacts")
    stage_ids = set(stage["candidate_id"])
    outcome_ids = set(outcomes["candidate_id"])
    if stage_ids != outcome_ids:
        raise ValueError(
            "stage and label candidate-ID sets differ: "
            f"stage_only={len(stage_ids - outcome_ids)}, "
            f"labels_only={len(outcome_ids - stage_ids)}"
        )
    identity = stage.rename(
        columns={"signal_timestamp": "__ts__", "symbol": "__symbol__"}
    ).loc[:, [*IDENTITY, "base_score"]]
    identity["__ts__"] = pd.to_datetime(identity["__ts__"], utc=True)
    outcome_identity = outcomes.loc[:, IDENTITY].copy()
    outcome_identity["__ts__"] = pd.to_datetime(outcome_identity["__ts__"], utc=True)
    identity_check = identity.loc[:, IDENTITY].merge(
        outcome_identity,
        on="candidate_id",
        how="outer",
        suffixes=("_stage", "_labels"),
        validate="one_to_one",
        indicator=True,
    )
    mismatch = identity_check["_merge"].ne("both")
    for column in ("__ts__", "__symbol__", "side_name"):
        mismatch |= identity_check[f"{column}_stage"].ne(
            identity_check[f"{column}_labels"]
        )
    if mismatch.any():
        raise ValueError(
            "stage and label candidate identities differ for "
            f"{int(mismatch.sum())} rows"
        )
    frame = identity.merge(feature_frame, on="candidate_id", how="inner", validate="one_to_one")
    frame = frame.merge(
        outcomes.drop(columns=["__ts__", "__symbol__", "side_name"]),
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame[ALPHA_TARGET] = _soft_alpha(frame["__soft_tb_first_event__"])
    lineage = [
        {
            "stage": str(path.resolve()),
            "stage_sha256": sha256(path),
        }
        for path in stages
    ] + [
        {
            "labels": str(path.resolve()),
            "labels_sha256": sha256(path),
        }
        for path in labels
    ]
    return frame, feature_columns, lineage


def load_inverse_population(path: Path) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    features = list(manifest["feature_columns"])
    columns = [
        "candidate_id",
        "signal_timestamp",
        "symbol",
        "side_name",
        "base_score",
        "__soft_tb_first_event__",
        NET_TARGET,
        GROSS_TARGET,
        COST_TARGET,
        *features,
    ]
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    frame = frame.rename(
        columns={"signal_timestamp": "__ts__", "symbol": "__symbol__"}
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame[ALPHA_TARGET] = _soft_alpha(frame["__soft_tb_first_event__"])
    return frame, features, [{"panel": str(path.resolve()), "panel_sha256": sha256(path)}]


def add_anchors(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    result["base_prediction"] = pd.to_numeric(result["base_score"], errors="coerce")
    grouped = result.groupby(["__ts__", "side_name"], sort=False)["base_score"]
    result["base_rank_timestamp_side"] = grouped.rank(method="first", ascending=False)
    result["base_rank_pct_timestamp_side"] = grouped.rank(pct=True, ascending=True)
    hour = result["__ts__"].dt.hour.to_numpy()
    dow = result["__ts__"].dt.dayofweek.to_numpy()
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    anchors = [
        "base_prediction",
        "base_rank_timestamp_side",
        "base_rank_pct_timestamp_side",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    return result, anchors


def finite_features(
    frame: pd.DataFrame, candidates: Iterable[str], *, minimum_fraction: float = 0.90
) -> list[str]:
    result = []
    for name in candidates:
        values = pd.to_numeric(frame[name], errors="coerce")
        if float(np.isfinite(values).mean()) >= minimum_fraction and values.nunique(dropna=True) > 1:
            frame[name] = values.astype(np.float32)
            result.append(name)
    return result


def _lgb_params(seed: int, estimators: int) -> dict[str, Any]:
    return {
        "objective": "regression_l1",
        "n_estimators": estimators,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 120,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.05,
        "reg_lambda": 0.50,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def select_fold_features(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    raw_features: Sequence[str],
    anchors: Sequence[str],
    count: int,
    seed: int,
) -> tuple[list[str], dict[str, float]]:
    train_positions = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    if len(train_positions) > 45_000:
        train_positions = np.sort(rng.choice(train_positions, 45_000, replace=False))
    model = lgb.LGBMRegressor(**_lgb_params(seed, 120))
    matrix = frame.iloc[train_positions].loc[:, raw_features]
    target = frame.iloc[train_positions][ALPHA_TARGET].to_numpy(float)
    model.fit(matrix, target)
    importance = dict(zip(raw_features, model.feature_importances_.astype(float)))
    selected_raw = sorted(raw_features, key=lambda name: (-importance[name], name))[
        : max(1, count - len(anchors))
    ]
    return [*anchors, *selected_raw], importance


def _isotonic(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)


def fit_lineage(
    frame: pd.DataFrame,
    raw_features: Sequence[str],
    folds: Sequence[tuple[str, str]],
    lineage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, anchors = add_anchors(frame)
    raw_features = finite_features(frame, raw_features)
    if len(raw_features) < 20:
        raise ValueError(f"{lineage}: too few finite historical raw features")
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for fold_number, (start_raw, end_raw) in enumerate(folds):
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        evaluation = frame["__ts__"].ge(start) & frame["__ts__"].lt(end)
        purge_start = start - pd.Timedelta(hours=PURGE_HOURS)
        purge_end = end + pd.Timedelta(hours=PURGE_HOURS)
        training = frame["__ts__"].lt(purge_start) | frame["__ts__"].ge(purge_end)
        for side in ("long", "short"):
            side_mask = frame["side_name"].eq(side).to_numpy()
            train_mask = training.to_numpy() & side_mask
            eval_mask = evaluation.to_numpy() & side_mask
            if train_mask.sum() < 5_000 or eval_mask.sum() < 500:
                raise ValueError(
                    f"{lineage} fold {fold_number} {side} lacks support: "
                    f"train={train_mask.sum()} eval={eval_mask.sum()}"
                )
            feature_count = 55 if side == "long" else 37
            selected, importance = select_fold_features(
                frame,
                train_mask,
                raw_features,
                anchors,
                feature_count,
                SEED + fold_number + (0 if side == "long" else 100),
            )
            for name, value in importance.items():
                importance_rows.append(
                    {
                        "lineage": lineage,
                        "fold": fold_number,
                        "side_name": side,
                        "feature": name,
                        "selection_gain": value,
                        "selected": name in selected,
                    }
                )

            train = frame.loc[train_mask]
            evaluate = frame.loc[eval_mask]
            base_alpha_map = _isotonic(
                train["base_score"].to_numpy(float),
                train[ALPHA_TARGET].to_numpy(float),
            )
            base_alpha_train = base_alpha_map.predict(train["base_score"].to_numpy(float))
            residual_target = train[ALPHA_TARGET].to_numpy(float) - base_alpha_train
            model = lgb.LGBMRegressor(
                **_lgb_params(SEED + 1_000 + fold_number, 320)
            ).fit(train.loc[:, selected], residual_target)
            train_delta = model.predict(train.loc[:, selected])
            eval_delta = model.predict(evaluate.loc[:, selected])
            train_alpha = np.clip(base_alpha_train + train_delta, 0.0, 1.0)
            eval_base_alpha = base_alpha_map.predict(evaluate["base_score"].to_numpy(float))
            eval_alpha = np.clip(eval_base_alpha + eval_delta, 0.0, 1.0)

            base_ev_map = _isotonic(
                base_alpha_train, train[NET_TARGET].to_numpy(float)
            )
            residual_ev_map = _isotonic(
                train_alpha, train[NET_TARGET].to_numpy(float)
            )
            result = evaluate.loc[
                :,
                [
                    *IDENTITY,
                    ALPHA_TARGET,
                    NET_TARGET,
                    GROSS_TARGET,
                    COST_TARGET,
                ],
            ].copy()
            result["score_base_alpha"] = eval_base_alpha
            result["score_residual_alpha"] = eval_alpha
            result["score_base_expected_ev"] = base_ev_map.predict(eval_base_alpha)
            result["score_residual_expected_ev"] = residual_ev_map.predict(eval_alpha)
            result["score_residual_delta_alpha"] = eval_delta
            result["stack_lineage"] = lineage
            result["residual_fold"] = f"{lineage}_block_{fold_number:02d}"
            result["residual_is_oof"] = True
            outputs.append(result)
            audits.append(
                {
                    "lineage": lineage,
                    "fold": fold_number,
                    "side_name": side,
                    "evaluation_start": start,
                    "evaluation_end_exclusive": end,
                    "train_rows": int(train_mask.sum()),
                    "evaluation_rows": int(eval_mask.sum()),
                    "purge_hours": PURGE_HOURS,
                    "selected_feature_count": len(selected),
                    "selected_features": selected,
                }
            )
    scored = pd.concat(outputs, ignore_index=True).sort_values(
        ["__ts__", "candidate_id"], kind="stable"
    )
    if scored["candidate_id"].duplicated().any():
        raise ValueError(f"{lineage}: duplicate OOF score")
    return scored, pd.DataFrame(audits), pd.DataFrame(importance_rows)


def _rank_ic(frame: pd.DataFrame, score: str) -> float:
    if len(frame) < 3 or frame[score].nunique() < 2:
        return math.nan
    return float(spearmanr(frame[score], frame[ALPHA_TARGET]).statistic)


def period_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frequency, grouper in (
        ("week", scored["__ts__"].dt.to_period("W-SUN").astype(str)),
        ("month", scored["__ts__"].dt.strftime("%Y-%m")),
    ):
        for period, frame in scored.groupby(grouper, sort=True):
            count = max(1, int(math.ceil(0.10 * len(frame))))
            selected = frame.nlargest(count, "score_residual_expected_ev")
            rows.append(
                {
                    "frequency": frequency,
                    "period": period,
                    "rows": len(frame),
                    "selected_rows": len(selected),
                    "base_alpha_ic": _rank_ic(frame, "score_base_alpha"),
                    "residual_alpha_ic": _rank_ic(frame, "score_residual_alpha"),
                    "top10_net_ev_bps": selected[NET_TARGET].mean() * 10_000,
                    "top10_gross_ev_bps": selected[GROSS_TARGET].mean() * 10_000,
                    "top10_cost_bps": selected[COST_TARGET].mean() * 10_000,
                    "long_share": selected["side_name"].eq("long").mean(),
                    "lineages": "|".join(sorted(frame["stack_lineage"].unique())),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inverse, inverse_features, inverse_sources = load_inverse_population(args.inverse)
    full_2024_labels = args.full_2024_labels or [DEFAULT_FULL_2024_LABELS]
    pf, pf_features, pf_sources = load_pf_population(
        [args.pf_stage, args.full_2024_stage],
        [args.pf_labels, *full_2024_labels],
    )
    inverse_scores, inverse_audit, inverse_importance = fit_lineage(
        inverse, inverse_features, INVERSE_FOLDS, "inverse_pi_2022_h1"
    )
    pf_scores, pf_audit, pf_importance = fit_lineage(
        pf, pf_features, PF_FOLDS, "frozen_pf_2022aug_2024"
    )
    scores = pd.concat([inverse_scores, pf_scores], ignore_index=True).sort_values(
        ["__ts__", "candidate_id"], kind="stable"
    )
    audits = pd.concat([inverse_audit, pf_audit], ignore_index=True)
    importance = pd.concat([inverse_importance, pf_importance], ignore_index=True)
    metrics = period_metrics(scores)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=args.output_dir.parent, prefix=f".{args.output_dir.name}."
        )
    )
    try:
        outputs = {
            "oof_scores.parquet": scores,
            "fold_audit.parquet": audits,
            "feature_selection_importance.parquet": importance,
            "period_metrics.csv": metrics,
        }
        hashes = {}
        for name, frame in outputs.items():
            path = temporary / name
            if path.suffix == ".parquet":
                frame.to_parquet(path, index=False, compression="zstd")
            else:
                frame.to_csv(path, index=False)
            hashes[name] = sha256(path)
        manifest = {
            "schema": SCHEMA,
            "status": "RESEARCH_OOF_BACKFILL_COMPLETE",
            "promotion_eligible": False,
            "architecture": "frozen base -> side-local residual alpha -> train-only monotonic execution-EV mapping",
            "validation": "non-walk-forward held-calendar-block OOF",
            "selection": "one pooled-global top10 across sides and timestamps per reporting period",
            "targets": {
                "alpha": (
                    "ATR-normalized soft triple-barrier first-event: favorable=1, "
                    "timeout=0.5, adverse/conflict=0"
                ),
                "economics": NET_TARGET,
                "horizon": "signal+1h decision, exact 1m [decision, decision+12h)",
            },
            "feature_selection": {
                "side_local": True,
                "long_count": 55,
                "short_count": 37,
                "method": "fold-train-only LightGBM gain screen",
                "approved_55_37_replay": (
                    "counts preserved; literal modern names unavailable in the "
                    "historical schema, so selection is performed from contemporaneous "
                    "PIT fields inside each training fold"
                ),
                "held_out_labels_used": False,
            },
            "coverage": {
                "start": scores["__ts__"].min(),
                "end": scores["__ts__"].max(),
                "rows": len(scores),
                "months": scores["__ts__"].dt.strftime("%Y-%m").nunique(),
                "inverse_rows": len(inverse_scores),
                "frozen_pf_rows": len(pf_scores),
            },
            "lineage_limitations": {
                "inverse_pi_2022_h1": (
                    "separate inverse-contract candidate population; current-spread "
                    "counterfactual, not deployed linear-perp parity"
                ),
                "frozen_pf_2022aug_2024": (
                    "frozen base backcast is diagnostic rather than historical base OOS; "
                    "residual and EV mappings are held-block OOF"
                ),
                "historical_l2_spread": "unavailable; frozen/current spread counterfactual",
            },
            "sources": [*inverse_sources, *pf_sources],
            "outputs_sha256": hashes,
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.sha256").write_text(
            f"{sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return args.output_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--inverse", type=Path, default=DEFAULT_INVERSE)
    result.add_argument("--pf-stage", type=Path, default=DEFAULT_PF_STAGE)
    result.add_argument("--pf-labels", type=Path, default=DEFAULT_PF_LABELS)
    result.add_argument(
        "--full-2024-stage", type=Path, default=DEFAULT_FULL_2024_STAGE
    )
    result.add_argument(
        "--full-2024-labels",
        type=Path,
        action="append",
        default=None,
        help=(
            "Repeat for disjoint label bundles whose exact union equals the full "
            "2024 stage. Defaults to the single full-year bundle."
        ),
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    print(run(parser().parse_args()))
