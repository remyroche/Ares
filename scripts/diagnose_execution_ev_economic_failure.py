#!/usr/bin/env python3
"""Decompose execution-EV failure on one immutable candidate population.

This diagnostic never fits a trading model or selects a threshold.  It joins
already-scored rows to exact-policy outcomes, ranks every score arm on the same
finite identity intersection, and reports:

* raw rank/discrimination and return-unit calibration;
* gross opportunity, MFE, MAE, timeout, stop, cost, and net economics;
* one pooled global top-k across timestamps and sides;
* month, week, side, and evaluation-origin slices; and
* selection overlap plus added/dropped-row economics for declared raw/mapped
  score pairs.

The output is diagnostic evidence only.  In particular, a calendar slice is
not a new untouched test when its rows were used for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA = "execution_ev_economic_failure_diagnosis_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
NET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
MFE = "execution_mfe_return_12h"
MAE = "execution_mae_return_12h"
COST = "execution_cost_return"
EXIT = "execution_exit_reason"
OUTCOME_COLUMNS = (DECISION, RESOLUTION, NET, GROSS, MFE, MAE, COST, EXIT)
RECONCILIATION_TOLERANCE = 1e-7


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def load_exact_population(
    ledger_path: Path,
    outcome_paths: Sequence[Path],
    score_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the exact identity intersection without silently multiplying rows."""

    score_columns = tuple(dict.fromkeys(map(str, score_columns)))
    if not score_columns:
        raise ValueError("at least one score column is required")
    ledger = pd.read_parquet(ledger_path)
    missing = [column for column in IDENTITY if column not in ledger]
    if missing:
        raise ValueError("score ledger is missing columns: " + ", ".join(missing))
    if ledger.duplicated(list(IDENTITY)).any():
        raise ValueError("score ledger identities must be unique")

    ledger_scores = [column for column in score_columns if column in ledger]
    ledger = ledger.loc[:, [*IDENTITY, *ledger_scores, *[
        column for column in ("evaluation_origin", "promotion_eligible")
        if column in ledger
    ]]]
    missing_score_sources = set(score_columns) - set(ledger_scores)
    outcome_frames: list[pd.DataFrame] = []
    for path in outcome_paths:
        available = pd.read_parquet(path)
        missing = [column for column in (*IDENTITY, *OUTCOME_COLUMNS) if column not in available]
        if missing:
            raise ValueError(f"{path} is missing outcome columns: " + ", ".join(missing))
        context_scores = [
            column for column in missing_score_sources if column in available
        ]
        outcome_frames.append(
            available.loc[:, [*IDENTITY, *OUTCOME_COLUMNS, *context_scores]]
        )
    if not outcome_frames:
        raise ValueError("at least one exact-policy outcome source is required")
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    duplicate = outcomes.duplicated(list(IDENTITY), keep=False)
    if duplicate.any():
        conflicts = (
            outcomes.loc[duplicate]
            .groupby(list(IDENTITY), dropna=False)[list(OUTCOME_COLUMNS)]
            .nunique(dropna=False)
            .max(axis=1)
        )
        if (conflicts > 1).any():
            raise ValueError("outcome sources contain conflicting duplicate identities")
        outcomes = outcomes.drop_duplicates(list(IDENTITY), keep="last")

    overlap = ledger.merge(outcomes, on=list(IDENTITY), how="inner", validate="one_to_one")
    missing = [column for column in score_columns if column not in overlap]
    if missing:
        raise ValueError(
            "score columns were not found in either ledger or outcome sources: "
            + ", ".join(missing)
        )
    numeric = _finite_numeric(overlap, [*score_columns, NET, GROSS, MFE, MAE, COST])
    finite = numeric.notna().all(axis=1)
    work = overlap.loc[finite].copy()
    work.loc[:, numeric.columns] = numeric.loc[finite].to_numpy(dtype=float)
    work[DECISION] = pd.to_datetime(work[DECISION], utc=True, errors="raise")
    work[RESOLUTION] = pd.to_datetime(work[RESOLUTION], utc=True, errors="raise")
    if (work[RESOLUTION] < work[DECISION]).any():
        raise ValueError("label resolution cannot precede the decision timestamp")
    reconciliation = np.abs(
        work[GROSS].to_numpy(dtype=float)
        - work[COST].to_numpy(dtype=float)
        - work[NET].to_numpy(dtype=float)
    )
    if reconciliation.max(initial=0.0) > RECONCILIATION_TOLERANCE:
        raise ValueError("gross - cost does not reconcile to net exactly once")
    work = work.sort_values([DECISION, *IDENTITY], kind="mergesort").reset_index(drop=True)
    audit = {
        "ledger_rows": int(len(ledger)),
        "unique_outcome_rows": int(len(outcomes)),
        "identity_intersection_rows": int(len(overlap)),
        "jointly_finite_rows": int(len(work)),
        "dropped_nonfinite_rows": int(len(overlap) - len(work)),
        "intersection_share_of_ledger": float(len(overlap) / len(ledger)) if len(ledger) else 0.0,
        "max_abs_gross_minus_cost_minus_net": float(reconciliation.max(initial=0.0)),
        "gross_cost_net_reconciliation_tolerance": RECONCILIATION_TOLERANCE,
    }
    return work, audit


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    if np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, score))


def _safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    if np.unique(target).size < 2:
        return float("nan")
    return float(average_precision_score(target, score))


def _selected_mask(score: np.ndarray, top_k_fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(score) * float(top_k_fraction))))
    order = np.argsort(-score, kind="mergesort")
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:count]] = True
    return selected


def _metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    selected: np.ndarray | None,
    higher_is_better: bool,
    return_unit_score: bool,
) -> dict[str, Any]:
    sample = frame if selected is None else frame.loc[selected]
    raw_score = frame[score_column].to_numpy(dtype=float)
    score = raw_score if higher_is_better else -raw_score
    net = frame[NET].to_numpy(dtype=float)
    positive = (net > 0.0).astype(np.int8)
    exit_reason = sample[EXIT].astype(str).str.lower()
    return {
        "rows": int(len(sample)),
        "selected_fraction": float(len(sample) / len(frame)) if len(frame) else 0.0,
        "score_spearman": float(frame[score_column].corr(frame[NET], method="spearman")),
        "positive_net_auc": _safe_auc(positive, score),
        "positive_net_average_precision": _safe_ap(positive, score),
        "score_orientation": "higher_is_better" if higher_is_better else "lower_is_better",
        "return_unit_score": bool(return_unit_score),
        "score_net_bias_bps": (
            float(np.mean(raw_score - net) * 10_000.0) if return_unit_score else np.nan
        ),
        "score_net_mae_bps": (
            float(np.mean(np.abs(raw_score - net)) * 10_000.0) if return_unit_score else np.nan
        ),
        "score_net_rmse_bps": (
            float(np.sqrt(np.mean(np.square(raw_score - net))) * 10_000.0)
            if return_unit_score else np.nan
        ),
        "mean_gross_ev_bps": float(sample[GROSS].mean() * 10_000.0),
        "mean_cost_bps": float(sample[COST].mean() * 10_000.0),
        "mean_net_ev_bps": float(sample[NET].mean() * 10_000.0),
        "sum_net_ev": float(sample[NET].sum()),
        "positive_gross_rate": float((sample[GROSS] > 0.0).mean()),
        "positive_net_rate": float((sample[NET] > 0.0).mean()),
        "mean_mfe_bps": float(sample[MFE].mean() * 10_000.0),
        "mean_mae_bps": float(sample[MAE].mean() * 10_000.0),
        "mfe_ge_cost_rate": float((sample[MFE] >= sample[COST]).mean()),
        "gross_edge_survives_cost_rate": float(
            ((sample[GROSS] > 0.0) & (sample[NET] > 0.0)).mean()
        ),
        "timeout_rate": float(exit_reason.eq("timeout").mean()),
        "full_stop_rate": float(exit_reason.isin(("full_sl", "stop", "full_stop")).mean()),
    }


def score_arm_metrics(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    top_k_fraction: float,
    lower_is_better: Iterable[str] = (),
    non_return_unit: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    selections: dict[str, np.ndarray] = {}
    lower = set(lower_is_better)
    non_return = set(non_return_unit)
    for score_column in score_columns:
        higher_is_better = score_column not in lower
        raw_score = frame[score_column].to_numpy(dtype=float)
        score = raw_score if higher_is_better else -raw_score
        selected = _selected_mask(score, top_k_fraction)
        selections[score_column] = selected
        rows.append({
            "score_arm": score_column,
            "scope": "all",
            **_metrics(
                frame,
                score_column,
                selected=None,
                higher_is_better=higher_is_better,
                return_unit_score=score_column not in non_return,
            ),
        })
        rows.append(
            {
                "score_arm": score_column,
                "scope": "pooled_global_topk",
                **_metrics(
                    frame,
                    score_column,
                    selected=selected,
                    higher_is_better=higher_is_better,
                    return_unit_score=score_column not in non_return,
                ),
            }
        )
    return pd.DataFrame(rows), selections


def sliced_topk_metrics(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    top_k_fraction: float,
    lower_is_better: Iterable[str] = (),
) -> pd.DataFrame:
    """Report local slices without changing the primary pooled admission set."""

    work = frame.copy()
    work["month"] = work[DECISION].dt.strftime("%Y-%m")
    work["week_start"] = (
        work[DECISION].dt.floor("D")
        - pd.to_timedelta(work[DECISION].dt.dayofweek, unit="D")
    ).dt.strftime("%Y-%m-%d")
    slice_columns = ["month", "week_start", "side_name"]
    if "evaluation_origin" in work:
        slice_columns.append("evaluation_origin")
    rows: list[dict[str, Any]] = []
    lower = set(lower_is_better)
    for score_column in score_columns:
        score = work[score_column].to_numpy(dtype=float)
        if score_column in lower:
            score = -score
        global_selected = _selected_mask(
            score, top_k_fraction
        )
        for slice_column in slice_columns:
            for value, positions in work.groupby(slice_column, dropna=False, sort=True).indices.items():
                index = np.asarray(positions, dtype=int)
                local = work.iloc[index]
                local_selected = global_selected[index]
                selected_frame = local.loc[local_selected]
                if selected_frame.empty:
                    continue
                exit_reason = selected_frame[EXIT].astype(str).str.lower()
                rows.append(
                    {
                        "score_arm": score_column,
                        "slice": slice_column,
                        "value": str(value),
                        "population_rows": int(len(local)),
                        "globally_selected_rows": int(local_selected.sum()),
                        "mean_net_ev_bps": float(selected_frame[NET].mean() * 10_000.0),
                        "mean_gross_ev_bps": float(selected_frame[GROSS].mean() * 10_000.0),
                        "mean_cost_bps": float(selected_frame[COST].mean() * 10_000.0),
                        "positive_net_rate": float((selected_frame[NET] > 0.0).mean()),
                        "mean_mfe_bps": float(selected_frame[MFE].mean() * 10_000.0),
                        "mean_mae_bps": float(selected_frame[MAE].mean() * 10_000.0),
                        "timeout_rate": float(exit_reason.eq("timeout").mean()),
                    }
                )
    return pd.DataFrame(rows)


def period_local_global_topk_metrics(
    frame: pd.DataFrame,
    score_columns: Sequence[str],
    *,
    top_k_fraction: float,
    lower_is_better: Iterable[str] = (),
) -> pd.DataFrame:
    """Rank once within each calendar period, pooled across time and sides."""

    work = frame.copy()
    work["month"] = work[DECISION].dt.strftime("%Y-%m")
    work["week_start"] = (
        work[DECISION].dt.floor("D")
        - pd.to_timedelta(work[DECISION].dt.dayofweek, unit="D")
    ).dt.strftime("%Y-%m-%d")
    lower = set(lower_is_better)
    rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        for period_column in ("month", "week_start"):
            for value, positions in work.groupby(period_column, sort=True).indices.items():
                index = np.asarray(positions, dtype=int)
                sample = work.iloc[index]
                score = sample[score_column].to_numpy(dtype=float)
                if score_column in lower:
                    score = -score
                selected = _selected_mask(score, top_k_fraction)
                chosen = sample.loc[selected]
                rows.append(
                    {
                        "score_arm": score_column,
                        "period": period_column,
                        "value": str(value),
                        "population_rows": int(len(sample)),
                        "selected_rows": int(len(chosen)),
                        "ranking_scope": "period_local_pooled_global_across_timestamps_and_sides",
                        "mean_net_ev_bps": float(chosen[NET].mean() * 10_000.0),
                        "mean_gross_ev_bps": float(chosen[GROSS].mean() * 10_000.0),
                        "mean_cost_bps": float(chosen[COST].mean() * 10_000.0),
                        "positive_net_rate": float((chosen[NET] > 0.0).mean()),
                        "mean_mfe_bps": float(chosen[MFE].mean() * 10_000.0),
                        "mean_mae_bps": float(chosen[MAE].mean() * 10_000.0),
                        "timeout_rate": float(
                            chosen[EXIT].astype(str).str.lower().eq("timeout").mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def selection_pair_metrics(
    frame: pd.DataFrame,
    selections: Mapping[str, np.ndarray],
    pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw, mapped in pairs:
        if raw not in selections or mapped not in selections:
            raise ValueError(f"selection pair {raw}={mapped} references an unknown score arm")
        raw_mask = selections[raw]
        mapped_mask = selections[mapped]
        intersection = raw_mask & mapped_mask
        union = raw_mask | mapped_mask
        added = mapped_mask & ~raw_mask
        dropped = raw_mask & ~mapped_mask
        rows.append(
            {
                "raw_score_arm": raw,
                "mapped_score_arm": mapped,
                "selected_rows_each": int(raw_mask.sum()),
                "intersection_rows": int(intersection.sum()),
                "jaccard": float(intersection.sum() / union.sum()) if union.any() else 1.0,
                "mapped_added_rows": int(added.sum()),
                "mapped_dropped_rows": int(dropped.sum()),
                "added_mean_net_ev_bps": (
                    float(frame.loc[added, NET].mean() * 10_000.0) if added.any() else np.nan
                ),
                "dropped_mean_net_ev_bps": (
                    float(frame.loc[dropped, NET].mean() * 10_000.0) if dropped.any() else np.nan
                ),
                "mapped_minus_raw_topk_net_ev_bps": float(
                    (frame.loc[mapped_mask, NET].mean() - frame.loc[raw_mask, NET].mean())
                    * 10_000.0
                ),
            }
        )
    return pd.DataFrame(rows)


def score_pair_drift(
    frame: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Measure pre/post score and rank movement without selecting local tails."""

    work = frame.copy()
    work["month"] = work[DECISION].dt.strftime("%Y-%m")
    groups: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.arange(len(work), dtype=int))
    ]
    for column in ("month", "side_name", "evaluation_origin"):
        if column not in work:
            continue
        groups.extend(
            (column, str(value), np.asarray(index, dtype=int))
            for value, index in work.groupby(column, dropna=False, sort=True).indices.items()
        )
    rows: list[dict[str, Any]] = []
    for raw, mapped in pairs:
        if raw not in work or mapped not in work:
            raise ValueError(f"score pair {raw}={mapped} references an unknown score arm")
        for slice_name, value, index in groups:
            sample = work.iloc[index]
            raw_score = sample[raw].to_numpy(dtype=float)
            mapped_score = sample[mapped].to_numpy(dtype=float)
            raw_rank = pd.Series(raw_score).rank(pct=True, method="average").to_numpy()
            mapped_rank = pd.Series(mapped_score).rank(pct=True, method="average").to_numpy()
            rows.append(
                {
                    "raw_score_arm": raw,
                    "mapped_score_arm": mapped,
                    "slice": slice_name,
                    "value": value,
                    "rows": int(len(sample)),
                    "score_spearman": float(pd.Series(raw_score).corr(pd.Series(mapped_score), method="spearman")),
                    "mean_score_delta_bps": float(np.mean(mapped_score - raw_score) * 10_000.0),
                    "median_score_delta_bps": float(np.median(mapped_score - raw_score) * 10_000.0),
                    "mean_abs_rank_percentile_delta": float(np.mean(np.abs(mapped_rank - raw_rank))),
                    "p90_abs_rank_percentile_delta": float(np.quantile(np.abs(mapped_rank - raw_rank), 0.90)),
                }
            )
    return pd.DataFrame(rows)


def _parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("score pairs must use RAW=MAPPED")
    raw, mapped = value.split("=", 1)
    if not raw or not mapped:
        raise argparse.ArgumentTypeError("score pairs must use non-empty RAW=MAPPED")
    return raw, mapped


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, population_audit = load_exact_population(
        args.ledger, args.outcomes, args.score_column
    )
    if not 0.0 < float(args.top_k_fraction) <= 1.0:
        raise ValueError("top-k fraction must be in (0, 1]")
    arm_metrics, selections = score_arm_metrics(
        frame,
        args.score_column,
        top_k_fraction=float(args.top_k_fraction),
        lower_is_better=args.lower_is_better,
        non_return_unit=args.non_return_unit,
    )
    slices = sliced_topk_metrics(
        frame,
        args.score_column,
        top_k_fraction=float(args.top_k_fraction),
        lower_is_better=args.lower_is_better,
    )
    period_local = period_local_global_topk_metrics(
        frame,
        args.score_column,
        top_k_fraction=float(args.top_k_fraction),
        lower_is_better=args.lower_is_better,
    )
    pairs = selection_pair_metrics(frame, selections, args.score_pair)
    drift = score_pair_drift(frame, args.score_pair)

    args.output_dir.mkdir(parents=True)
    arm_path = args.output_dir / "score_arm_metrics.csv"
    slice_path = args.output_dir / "global_topk_slice_metrics.csv"
    pair_path = args.output_dir / "mapping_selection_pairs.csv"
    drift_path = args.output_dir / "pre_post_score_drift.csv"
    period_path = args.output_dir / "period_local_global_topk_metrics.csv"
    row_path = args.output_dir / "diagnostic_rows.parquet"
    arm_metrics.to_csv(arm_path, index=False)
    slices.to_csv(slice_path, index=False)
    pairs.to_csv(pair_path, index=False)
    drift.to_csv(drift_path, index=False)
    period_local.to_csv(period_path, index=False)
    keep = [*IDENTITY, *OUTCOME_COLUMNS, *args.score_column]
    for optional in ("evaluation_origin", "promotion_eligible"):
        if optional in frame:
            keep.append(optional)
    frame.loc[:, list(dict.fromkeys(keep))].to_parquet(row_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "diagnostic_only_not_promotion_evidence",
        "contract": {
            "identity": list(IDENTITY),
            "candidate_population": "jointly finite exact identity intersection shared by every arm",
            "admission": f"one pooled global top {float(args.top_k_fraction):.6f} across timestamps and sides",
            "slice_rule": "month/week/side/origin rows are attribution of the pooled global selection, never local reranking",
            "period_diagnostic_rule": "month/week diagnostics rank once within that period across all timestamps and sides; never per timestamp or per side",
            "threshold_selection": "none",
            "model_fitting": "none",
            "cost_reconciliation": "gross - recorded exact-policy cost = net; costs are not subtracted again",
        },
        "inputs": {
            "ledger": str(args.ledger),
            "ledger_sha256": _sha256(args.ledger),
            "outcomes": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.outcomes
            ],
        },
        "score_columns": list(args.score_column),
        "lower_is_better": list(args.lower_is_better),
        "non_return_unit_scores": list(args.non_return_unit),
        "score_pairs": [list(pair) for pair in args.score_pair],
        "population_audit": population_audit,
        "outputs": {
            "score_arm_metrics": str(arm_path),
            "global_topk_slice_metrics": str(slice_path),
            "mapping_selection_pairs": str(pair_path),
            "pre_post_score_drift": str(drift_path),
            "period_local_global_topk_metrics": str(period_path),
            "diagnostic_rows": str(row_path),
        },
    }
    for path in (arm_path, slice_path, pair_path, drift_path, period_path, row_path):
        manifest["outputs"][f"{path.stem}_sha256"] = _sha256(path)
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, action="append", required=True)
    parser.add_argument("--score-column", action="append", required=True)
    parser.add_argument("--lower-is-better", action="append", default=[])
    parser.add_argument("--non-return-unit", action="append", default=[])
    parser.add_argument("--score-pair", type=_parse_pair, action="append", default=[])
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
