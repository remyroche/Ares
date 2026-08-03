#!/usr/bin/env python3
"""Run the preregistered 2x2 causal mapping-repair ablation.

The experiment changes only score-to-EV mapping.  Base, residual, direct-q25,
identities, exact H12 economics, costs, and the global top-k rule stay frozen.
March 3--19 is the only arm-selection window.  March 20--31 and April are
confirmation diagnostics on reused research months, never promotion evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from scripts.materialize_source_separated_ic_ev_waterfall import safe, sha256
from scripts.run_canonical_execution_reliability_mapping_ablation_v2 import (
    positive_huber,
)


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / (
    "data_perp/artifacts/"
    "marapr2025_identical_causal_score_bridge_20260730_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/"
    "marapr2025_causal_mapping_repair_ablation_20260730_v1"
)
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TIME = "execution_decision_utc"
END = "execution_label_end_utc"
NET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
WINDOW = pd.Timedelta(days=21)
MINIMUM_ROWS = 2_000
SHRINK_WEIGHT = 0.25
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260730
METHODS = ("raw", "I", "I_R", "I_S", "I_S_R")
MAPPED_METHODS = ("I", "I_R", "I_S", "I_S_R")
RAW_SOURCE_COLUMNS = {
    "base": "score_base_alpha",
    "direct_q25": "direct_q25_return",
}
PERIODS = {
    "selection_march03_19": (
        pd.Timestamp("2025-03-03T00:00:00Z"),
        pd.Timestamp("2025-03-20T00:00:00Z"),
    ),
    "confirmation_march20_31": (
        pd.Timestamp("2025-03-20T00:00:00Z"),
        pd.Timestamp("2025-04-01T00:00:00Z"),
    ),
    "confirmation_april": (
        pd.Timestamp("2025-04-01T00:00:00Z"),
        pd.Timestamp("2025-05-01T00:00:00Z"),
    ),
}


class MappingRepairError(RuntimeError):
    """Raised when a frozen-input, causal-map, or evaluation contract fails."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise MappingRepairError(f"{name} missing identity: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any():
        raise MappingRepairError(f"{name} duplicate identity")
    if not result["side_name"].isin(("long", "short")).all():
        raise MappingRepairError(f"{name} invalid side")
    return result


def identity_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True).astype(str)
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(values.to_csv(index=False).encode()).hexdigest()


def strict_rank_key(
    mapped: Sequence[float], raw: Sequence[float]
) -> np.ndarray:
    """Return descending lexicographic ranks for (mapped, raw)."""

    mapped_values = np.asarray(mapped, dtype=float)
    raw_values = np.asarray(raw, dtype=float)
    if mapped_values.shape != raw_values.shape:
        raise ValueError("mapped/raw shapes differ")
    order = np.lexsort((-raw_values, -mapped_values))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), 0, -1, dtype=np.int64)
    return ranks


def align_huber_to_isotonic(
    reference_iso: Sequence[float],
    reference_huber: Sequence[float],
    evaluate_huber: Sequence[float],
) -> np.ndarray:
    """Reference-only median/IQR alignment and 1st--99th percentile clipping."""

    iso = np.asarray(reference_iso, dtype=float)
    huber = np.asarray(reference_huber, dtype=float)
    evaluate = np.asarray(evaluate_huber, dtype=float)
    if not (
        np.isfinite(iso).all()
        and np.isfinite(huber).all()
        and np.isfinite(evaluate).all()
    ):
        raise ValueError("non-finite alignment input")
    iso_median = float(np.median(iso))
    huber_median = float(np.median(huber))
    iso_iqr = float(np.quantile(iso, 0.75) - np.quantile(iso, 0.25))
    huber_iqr = float(np.quantile(huber, 0.75) - np.quantile(huber, 0.25))
    scale = iso_iqr / huber_iqr if huber_iqr > 1e-15 else 0.0
    aligned = iso_median + (evaluate - huber_median) * scale
    clip_low, clip_high = np.quantile(iso, (0.01, 0.99))
    aligned = np.clip(aligned, clip_low, clip_high)
    return aligned


def build_day_maps(
    history: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    shrink_weight: float = SHRINK_WEIGHT,
    minimum_rows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit I and I-S on one legal daily reference snapshot."""

    history = history.copy()
    evaluate = evaluate.copy()
    required = {*IDENTITY, TIME, END, NET, "raw_score"}
    for name, frame in (("history", history), ("evaluate", evaluate)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise MappingRepairError(f"{name} missing: {missing}")
        frame[TIME] = pd.to_datetime(frame[TIME], utc=True, errors="raise")
        frame[END] = pd.to_datetime(frame[END], utc=True, errors="raise")
    snapshots = evaluate[TIME].dt.floor("D").unique()
    if len(snapshots) != 1:
        raise MappingRepairError("evaluate must contain one UTC-day snapshot")
    snapshot = pd.Timestamp(snapshots[0])
    history = history.loc[
        history[END].ge(snapshot - WINDOW)
        & history[END].lt(snapshot)
        & history["__ts__"].lt(snapshot)
    ].copy()
    if len(history) < int(minimum_rows):
        raise MappingRepairError("insufficient mapping history")
    if not history[END].lt(snapshot).all():
        raise MappingRepairError("mapping history includes unresolved labels")
    if not history[END].ge(snapshot - WINDOW).all():
        raise MappingRepairError("mapping history exceeds the 21d window")
    if not history["__ts__"].lt(snapshot).all():
        raise MappingRepairError("mapping history score is not available")
    overlap = len(
        set(map(tuple, history.loc[:, IDENTITY].astype(str).to_numpy())).intersection(
            set(map(tuple, evaluate.loc[:, IDENTITY].astype(str).to_numpy()))
        )
    )
    if overlap:
        raise MappingRepairError("mapping history overlaps evaluation identities")
    raw_history = pd.to_numeric(history["raw_score"], errors="raise").to_numpy(float)
    target_history = pd.to_numeric(history[NET], errors="raise").to_numpy(float)
    raw_evaluate = pd.to_numeric(evaluate["raw_score"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(raw_history).all()
        or not np.isfinite(target_history).all()
        or not np.isfinite(raw_evaluate).all()
    ):
        raise MappingRepairError("non-finite mapping values")
    if np.unique(raw_history).size < 2:
        constant = float(np.mean(target_history))
        reference_iso = np.full(len(history), constant, dtype=float)
        evaluate_iso = np.full(len(evaluate), constant, dtype=float)
        reference_huber = reference_iso.copy()
        evaluate_huber = evaluate_iso.copy()
    else:
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
        isotonic.fit(raw_history, target_history)
        reference_iso = isotonic.predict(raw_history)
        evaluate_iso = isotonic.predict(raw_evaluate)
        combined_huber = positive_huber(
            history, np.concatenate([raw_history, raw_evaluate])
        )
        reference_huber = combined_huber[: len(raw_history)]
        evaluate_huber = combined_huber[len(raw_history) :]
    aligned_huber = align_huber_to_isotonic(
        reference_iso, reference_huber, evaluate_huber
    )
    iso_median = float(np.median(reference_iso))
    huber_median = float(np.median(reference_huber))
    iso_iqr = float(
        np.quantile(reference_iso, 0.75) - np.quantile(reference_iso, 0.25)
    )
    huber_iqr = float(
        np.quantile(reference_huber, 0.75)
        - np.quantile(reference_huber, 0.25)
    )
    clip_low, clip_high = np.quantile(reference_iso, (0.01, 0.99))
    shrunk = (1.0 - float(shrink_weight)) * evaluate_iso + float(
        shrink_weight
    ) * aligned_huber
    output = evaluate.loc[:, list(IDENTITY)].copy()
    output["raw_score"] = raw_evaluate
    output["score__raw"] = raw_evaluate
    output["score__I"] = evaluate_iso
    output["score__I_R"] = evaluate_iso
    output["score__I_S"] = shrunk
    output["score__I_S_R"] = shrunk
    audit = pd.DataFrame([{
        "snapshot_utc": snapshot,
        "reference_rows": int(len(history)),
        "evaluation_rows": int(len(evaluate)),
        "reference_identity_sha256": identity_hash(history),
        "reference_label_end_max_utc": history[END].max(),
        "reference_max_label_end_utc": history[END].max(),
        "reference_score_available_max_utc": history["__ts__"].max(),
        "evaluation_reference_overlap": int(overlap),
        "all_reference_labels_before_snapshot": bool(history[END].lt(snapshot).all()),
        "zero_evaluation_reference_overlap": bool(overlap == 0),
        "shrink_weight": float(shrink_weight),
        "isotonic_reference_unique_scores": int(np.unique(raw_history).size),
        "isotonic_evaluation_unique_values": int(np.unique(evaluate_iso).size),
        "shrunk_evaluation_unique_values": int(np.unique(shrunk).size),
        "reference_iso_median": iso_median,
        "reference_iso_iqr": iso_iqr,
        "reference_huber_median": huber_median,
        "reference_huber_iqr": huber_iqr,
        "alignment_scale": float(iso_iqr / huber_iqr if huber_iqr > 1e-15 else 0.0),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
    }])
    return output, audit


def causal_history(frame: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    resolved = pd.to_datetime(frame[END], utc=True, errors="raise")
    score_time = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    return frame.loc[
        resolved.ge(snapshot - WINDOW)
        & resolved.lt(snapshot)
        & score_time.lt(snapshot)
    ].copy()


def global_top(
    frame: pd.DataFrame,
    score_col: str,
    raw_col: str | None,
    fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One pooled-global book; raw is used only as a mapped-score tie break."""

    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    columns = [score_col]
    ascending = [False]
    if raw_col is not None and raw_col != score_col:
        columns.append(raw_col)
        ascending.append(False)
    columns.extend(["candidate_id", "side_name", "__symbol__", "__ts__"])
    ascending.extend([True, True, True, True])
    selected = frame.sort_values(
        columns, ascending=ascending, kind="mergesort"
    ).head(count).copy()
    return selected, {
        "selection_scope": "pooled_global",
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "fraction": float(fraction),
        "raw_plateau_tie_break": bool(raw_col is not None and raw_col != score_col),
    }


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3 or pair.left.nunique() < 2 or pair.right.nunique() < 2:
        return float("nan")
    return float(pair.left.corr(pair.right, method="spearman"))


def _tie_metrics(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    score_col: str,
    raw_col: str | None,
) -> dict[str, Any]:
    cutoff = float(selected[score_col].iloc[-1])
    above = candidates.loc[candidates[score_col].gt(cutoff)]
    tied = candidates.loc[candidates[score_col].eq(cutoff)]
    slots = len(selected) - len(above)
    best = pd.concat([above, tied.nlargest(slots, NET)])
    worst = pd.concat([above, tied.nsmallest(slots, NET)])
    return {
        "cutoff_score": cutoff,
        "cutoff_tie_rows": int(len(tied)),
        "slots_from_cutoff_tie": int(slots),
        "candidate_id_driven_cutoff": bool(raw_col is None and len(tied) > slots),
        "tie_sensitivity_bps": float((best[NET].mean() - worst[NET].mean()) * 1e4),
    }


def _calendar_metrics(selected: pd.DataFrame) -> dict[str, float]:
    days = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    shares = days.value_counts(normalize=True)
    return {
        "selected_days": float(len(shares)),
        "effective_selected_days": float(1.0 / np.square(shares).sum()),
        "max_day_share": float(shares.max()),
        "top3_day_share": float(shares.nlargest(3).sum()),
        "top5_day_share": float(shares.nlargest(5).sum()),
    }


def evaluate_arm(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    method: str | None = None,
    period: str = "unspecified",
    arm: str | None = None,
    score_col: str = "mapped_score",
    raw_col: str | None = None,
    fraction: float = 0.10,
    use_raw_tie: bool | None = None,
    return_details: bool = False,
) -> Any:
    """Evaluate one global book and return summary, selection and side attribution."""

    method = str(method or arm or "unspecified")
    source = str(source or "unspecified")
    if use_raw_tie is None:
        use_raw_tie = method in {"I_R", "I_S_R", "I-R", "I-S-R"}
    if raw_col is None and use_raw_tie:
        raw_col = "raw_score"
    if not use_raw_tie:
        raw_col = None
    selected, selection_contract = global_top(
        frame, score_col, raw_col, fraction
    )
    net = selected[NET].astype(float)
    summary = {
        "source": source,
        "method": method,
        "period": period,
        "fraction": float(fraction),
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selection_scope": selection_contract["selection_scope"],
        "mean_gross_bps": float(selected[GROSS].mean() * 1e4),
        "mean_cost_bps": float(selected[COST].mean() * 1e4),
        "mean_net_bps": float(net.mean() * 1e4),
        "positive_net_rate": float(net.gt(0).mean()),
        "score_net_rank_ic": _rank_ic(frame[score_col], frame[NET]),
        "mapped_unique_values": int(frame[score_col].nunique()),
        "mapped_unique_fraction": float(frame[score_col].nunique() / len(frame)),
        **_tie_metrics(
            frame, selected, score_col=score_col, raw_col=raw_col
        ),
        **_calendar_metrics(selected),
    }
    summary["net_bps"] = summary["mean_net_bps"]
    summary["top_three_day_share"] = summary["top3_day_share"]
    side_rows: list[dict[str, Any]] = []
    for side, local in selected.groupby("side_name", sort=True, observed=True):
        side_rows.append(
            {
                "source": source,
                "method": method,
                "period": period,
                "fraction": float(fraction),
                "side_name": str(side),
                "selected_rows": int(len(local)),
                "selected_share": float(len(local) / len(selected)),
                "mean_net_bps": float(local[NET].mean() * 1e4),
                "net_contribution_bps": float(local[NET].sum() / len(selected) * 1e4),
                "positive_net_rate": float(local[NET].gt(0).mean()),
            }
        )
    side = pd.DataFrame.from_records(side_rows)
    if not np.isclose(
        side["net_contribution_bps"].sum(), summary["mean_net_bps"], atol=1e-10
    ):
        raise MappingRepairError("side contribution does not reconcile")
    summary["side_contribution_bps"] = dict(
        zip(side["side_name"], side["net_contribution_bps"])
    )
    chosen = selected.loc[:, [*IDENTITY, NET]].copy()
    chosen["source"] = source
    chosen["method"] = method
    chosen["period"] = period
    chosen["fraction"] = float(fraction)
    if return_details:
        return summary, chosen, side
    return summary


def _bootstrap_delta(
    raw_selected: pd.DataFrame,
    mapped_selected: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, float]:
    raw = raw_selected.copy()
    mapped = mapped_selected.copy()
    raw["day"] = pd.to_datetime(raw["__ts__"], utc=True).dt.floor("D")
    mapped["day"] = pd.to_datetime(mapped["__ts__"], utc=True).dt.floor("D")
    days = np.array(sorted(set(raw.day).union(mapped.day)))

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        grouped = (
            frame.groupby("day", observed=True)[NET]
            .agg(["sum", "count"])
            .reindex(days, fill_value=0)
        )
        return grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)

    raw_sum, raw_count = arrays(raw)
    mapped_sum, mapped_count = arrays(mapped)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = rng.integers(0, len(days), size=len(days))
        raw_mean = raw_sum[sample].sum() / raw_count[sample].sum()
        mapped_mean = mapped_sum[sample].sum() / mapped_count[sample].sum()
        deltas[draw] = (mapped_mean - raw_mean) * 1e4
    return {
        "delta_vs_raw_bootstrap_mean_bps": float(deltas.mean()),
        "delta_vs_raw_ci_low_bps": float(np.quantile(deltas, 0.025)),
        "delta_vs_raw_ci_high_bps": float(np.quantile(deltas, 0.975)),
    }


def build_predictions(
    raw: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    eval_ids = evaluation.loc[:, list(IDENTITY)]
    for source, column in RAW_SOURCE_COLUMNS.items():
        source_frame = raw.loc[
            :,
            [*IDENTITY, TIME, END, NET, GROSS, COST, column],
        ].rename(columns={column: "raw_score"})
        current = source_frame.merge(
            eval_ids, on=list(IDENTITY), how="inner", validate="one_to_one"
        )
        if len(current) != len(evaluation):
            raise MappingRepairError(f"{source} evaluation coverage changed")
        source_parts: list[pd.DataFrame] = []
        for day, local in current.groupby(
            current[TIME].dt.floor("D"), sort=True, observed=True
        ):
            history = causal_history(source_frame, pd.Timestamp(day))
            mapped, audit_frame = build_day_maps(
                history, local, minimum_rows=MINIMUM_ROWS
            )
            audit = audit_frame.iloc[0].to_dict()
            audit["source"] = source
            audits.append(audit)
            source_parts.append(mapped)
        mapped_source = pd.concat(source_parts, ignore_index=True)
        economics = current.loc[
            :, [*IDENTITY, TIME, END, NET, GROSS, COST]
        ]
        mapped_source = mapped_source.merge(
            economics, on=list(IDENTITY), validate="one_to_one"
        )
        for method in METHODS:
            part = mapped_source.loc[
                :, [*IDENTITY, TIME, END, NET, GROSS, COST, "raw_score"]
            ].copy()
            part["source"] = source
            part["method"] = method
            part["mapped_score"] = mapped_source[f"score__{method}"].to_numpy(float)
            part["uses_raw_plateau_tie_break"] = method in {"I_R", "I_S_R"}
            outputs.append(part)

    residual = evaluation.loc[
        :,
        [
            *IDENTITY,
            TIME,
            END,
            NET,
            GROSS,
            COST,
            "score_residual_expected_ev",
        ],
    ].rename(columns={"score_residual_expected_ev": "raw_score"})
    residual["source"] = "residual"
    residual["method"] = "raw_residual"
    residual["mapped_score"] = residual["raw_score"].astype(float)
    residual["uses_raw_plateau_tie_break"] = False
    outputs.append(residual)
    predictions = pd.concat(outputs, ignore_index=True)
    if predictions.duplicated(["source", "method", *IDENTITY]).any():
        raise MappingRepairError("prediction identities duplicate")
    audit_frame = pd.DataFrame.from_records(audits)
    hashes = audit_frame.groupby("snapshot_utc")["reference_identity_sha256"].nunique()
    if not hashes.eq(1).all():
        raise MappingRepairError("base/direct mapping reference identities differ")
    return predictions, audit_frame


def evaluate(
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    metric_rows: list[dict[str, Any]] = []
    selections: list[pd.DataFrame] = []
    sides: list[pd.DataFrame] = []
    lookup: dict[tuple[str, str, str, float], tuple[dict[str, Any], pd.DataFrame]] = {}
    for period, (start, end) in PERIODS.items():
        period_rows = predictions.loc[
            predictions["__ts__"].ge(start) & predictions["__ts__"].lt(end)
        ]
        for (source, method), local in period_rows.groupby(
            ["source", "method"], sort=True, observed=True
        ):
            for fraction in TOP_FRACTIONS:
                summary, selected, side = evaluate_arm(
                    local,
                    source=str(source),
                    method=str(method),
                    period=period,
                    fraction=fraction,
                    return_details=True,
                )
                metric_rows.append(summary)
                selections.append(selected)
                sides.append(side)
                lookup[(str(source), str(method), period, fraction)] = (
                    summary,
                    selected,
                )
    metrics = pd.DataFrame.from_records(metric_rows)
    selection_frame = pd.concat(selections, ignore_index=True)
    side_frame = pd.concat(sides, ignore_index=True)

    bootstrap: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    for source in ("base", "direct_q25"):
        for period in PERIODS:
            raw_summary, raw_selected = lookup[(source, "raw", period, 0.10)]
            residual_summary, residual_selected = lookup[
                ("residual", "raw_residual", period, 0.10)
            ]
            raw_ids = set(map(tuple, raw_selected.loc[:, IDENTITY].astype(str).to_numpy()))
            residual_ids = set(
                map(tuple, residual_selected.loc[:, IDENTITY].astype(str).to_numpy())
            )
            for method in MAPPED_METHODS:
                summary, selected = lookup[(source, method, period, 0.10)]
                selected_ids = set(
                    map(tuple, selected.loc[:, IDENTITY].astype(str).to_numpy())
                )
                bootstrap.append(
                    {
                        "source": source,
                        "method": method,
                        "period": period,
                        **_bootstrap_delta(
                            raw_selected,
                            selected,
                            seed=BOOTSTRAP_SEED
                            + len(bootstrap) * 17,
                        ),
                    }
                )
                overlap.append(
                    {
                        "source": source,
                        "method": method,
                        "period": period,
                        "raw_overlap": float(len(raw_ids & selected_ids) / len(selected_ids)),
                        "residual_overlap": float(
                            len(residual_ids & selected_ids) / len(selected_ids)
                        ),
                        "delta_vs_raw_net_bps": float(
                            summary["mean_net_bps"] - raw_summary["mean_net_bps"]
                        ),
                        "delta_vs_residual_net_bps": float(
                            summary["mean_net_bps"]
                            - residual_summary["mean_net_bps"]
                        ),
                    }
                )

    bootstrap_frame = pd.DataFrame.from_records(bootstrap)
    overlap_frame = pd.DataFrame.from_records(overlap)
    gates, decisions = selection_gates(
        metrics, side_frame, bootstrap_frame
    )
    gates["detail"] = gates["detail"].map(str)
    return (
        metrics,
        selection_frame,
        side_frame,
        bootstrap_frame,
        overlap_frame,
        gates.merge(decisions, on="source", how="left"),
    )


def selection_gates(
    metrics: pd.DataFrame,
    sides: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    top = metrics.loc[np.isclose(metrics["fraction"], 0.10)]
    side_top = sides.loc[np.isclose(sides["fraction"], 0.10)]
    for source in ("base", "direct_q25"):
        raw_selection = top.loc[
            top["source"].eq(source)
            & top["method"].eq("raw")
            & top["period"].eq("selection_march03_19")
        ].iloc[0]
        candidates: list[tuple[str, float, bool]] = []
        for method in MAPPED_METHODS:
            mapped = top.loc[
                top["source"].eq(source)
                & top["method"].eq(method)
                & top["period"].eq("selection_march03_19")
            ].iloc[0]
            plateau_pass = bool(
                (method in {"I_R", "I_S_R"} and not mapped.candidate_id_driven_cutoff)
                or mapped.tie_sensitivity_bps <= 5.0
            )
            calendar_pass = bool(
                mapped.effective_selected_days
                >= 0.90 * raw_selection.effective_selected_days
                and mapped.top3_day_share <= raw_selection.top3_day_share + 0.05
            )
            economics_pass = bool(
                mapped.mean_net_bps >= raw_selection.mean_net_bps - 5.0
            )
            mapped_side = side_top.loc[
                side_top["source"].eq(source)
                & side_top["method"].eq(method)
                & side_top["period"].eq("selection_march03_19")
            ].set_index("side_name")
            raw_side = side_top.loc[
                side_top["source"].eq(source)
                & side_top["method"].eq("raw")
                & side_top["period"].eq("selection_march03_19")
            ].set_index("side_name")
            side_deltas = (
                mapped_side["net_contribution_bps"]
                - raw_side["net_contribution_bps"]
            )
            side_pass = bool(side_deltas.ge(-5.0).all())
            all_pass = plateau_pass and calendar_pass and economics_pass and side_pass
            candidates.append((method, float(mapped.mean_net_bps), all_pass))
            for gate, passed, detail in (
                ("plateau", plateau_pass, float(mapped.tie_sensitivity_bps)),
                (
                    "calendar",
                    calendar_pass,
                    json.dumps(
                        {
                            "effective_days": mapped.effective_selected_days,
                            "raw_effective_days": raw_selection.effective_selected_days,
                            "top3_share": mapped.top3_day_share,
                            "raw_top3_share": raw_selection.top3_day_share,
                        },
                        sort_keys=True,
                    ),
                ),
                (
                    "economics_not_worse_5bps",
                    economics_pass,
                    float(mapped.mean_net_bps - raw_selection.mean_net_bps),
                ),
                (
                    "side_contribution_not_worse_5bps",
                    side_pass,
                    json.dumps(side_deltas.to_dict(), sort_keys=True),
                ),
            ):
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "gate_period": "selection_march03_19",
                        "gate": gate,
                        "pass": bool(passed),
                        "detail": detail,
                    }
                )
        passing = [candidate for candidate in candidates if candidate[2]]
        selected = max(passing, key=lambda item: (item[1], item[0]))[0] if passing else "ABSTAIN"
        decisions.append(
            {
                "source": source,
                "selected_mapping_arm": selected,
                "selection_rule": (
                    "highest March03-19 top10 net among arms passing plateau, "
                    "calendar, economics and side-safety gates; otherwise abstain"
                ),
                "promotion_eligible": False,
            }
        )
        for period in ("confirmation_march20_31", "confirmation_april"):
            if selected == "ABSTAIN":
                rows.append(
                    {
                        "source": source,
                        "method": selected,
                        "gate_period": period,
                        "gate": "confirmation_not_run_after_selection_abstention",
                        "pass": False,
                        "detail": "selection gates produced ABSTAIN",
                    }
                )
                continue
            mapped = top.loc[
                top["source"].eq(source)
                & top["method"].eq(selected)
                & top["period"].eq(period)
            ].iloc[0]
            raw = top.loc[
                top["source"].eq(source)
                & top["method"].eq("raw")
                & top["period"].eq(period)
            ].iloc[0]
            interval = bootstrap.loc[
                bootstrap["source"].eq(source)
                & bootstrap["method"].eq(selected)
                & bootstrap["period"].eq(period)
            ].iloc[0]
            passed = bool(mapped.mean_net_bps >= raw.mean_net_bps - 5.0)
            rows.append(
                {
                    "source": source,
                    "method": selected,
                    "gate_period": period,
                    "gate": "confirmation_economics_not_worse_5bps",
                    "pass": passed,
                    "detail": json.dumps(
                        {
                            "delta_bps": mapped.mean_net_bps - raw.mean_net_bps,
                            "bootstrap_low_bps": interval.delta_vs_raw_ci_low_bps,
                            "bootstrap_high_bps": interval.delta_vs_raw_ci_high_bps,
                        },
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(decisions)


def run(
    *,
    bridge_root: Path = BRIDGE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite sealed output: {output_dir}")
    manifest_path = bridge_root / "manifest.json"
    seal_path = bridge_root / "manifest.sha256"
    if (
        not manifest_path.is_file()
        or not seal_path.is_file()
        or seal_path.read_text().split()[0] != sha256(manifest_path)
    ):
        raise MappingRepairError("bridge manifest seal fails")
    bridge_manifest = _read_json(manifest_path)
    if bridge_manifest.get("schema") != "marapr2025_identical_causal_score_bridge_v1":
        raise MappingRepairError("unexpected bridge schema")
    bridge_path = bridge_root / "identical_score_bridge.parquet"
    if (
        bridge_manifest.get("outputs_sha256", {}).get(bridge_path.name)
        != sha256(bridge_path)
    ):
        raise MappingRepairError("bridge output hash fails")
    raw_record = bridge_manifest.get("sources", {}).get("raw_all_score", {})
    raw_path = Path(str(raw_record.get("path")))
    if str(raw_record.get("sha256")) != sha256(raw_path):
        raise MappingRepairError("raw source hash fails")
    raw = _normalise(pd.read_parquet(raw_path), "raw all-score source")
    evaluation = _normalise(pd.read_parquet(bridge_path), "sealed evaluation")
    for frame in (raw, evaluation):
        frame[TIME] = pd.to_datetime(frame[TIME], utc=True, errors="raise")
        frame[END] = pd.to_datetime(frame[END], utc=True, errors="raise")
    if len(raw) != 140_682 or len(evaluation) != 136_074:
        raise MappingRepairError("raw/evaluation row contract changed")

    predictions, audit = build_predictions(raw, evaluation)
    (
        metrics,
        selections,
        sides,
        bootstrap,
        overlap,
        gates,
    ) = evaluate(predictions)
    decision = gates.loc[
        :, ["source", "selected_mapping_arm", "selection_rule", "promotion_eligible"]
    ].drop_duplicates()

    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True, exist_ok=False)
        frames = {
            "mapped_predictions": predictions,
            "causal_mapping_audit": audit,
            "period_metrics": metrics,
            "selected_books": selections,
            "side_attribution": sides,
            "paired_day_bootstrap": bootstrap,
            "selection_overlap": overlap,
            "gates": gates.drop(
                columns=[
                    "selected_mapping_arm",
                    "selection_rule",
                    "promotion_eligible",
                ]
            ),
            "selection_decisions": decision,
        }
        outputs: dict[str, dict[str, Any]] = {}
        for name, frame in frames.items():
            path = stage / f"{name}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {
                "path": path.name,
                "rows": int(len(frame)),
                "sha256": sha256(path),
            }
        result = {
            "schema": "marapr2025_causal_mapping_repair_ablation_v1",
            "status": "SEALED_REUSED_MONTH_MAPPING_DIAGNOSTIC_NO_PROMOTION",
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "contract": {
                "evaluation_rows": int(len(evaluation)),
                "evaluation_identity_sha256": identity_hash(evaluation),
                "reference_rows": int(len(raw)),
                "sources": dict(RAW_SOURCE_COLUMNS),
                "residual_control": "score_residual_expected_ev unchanged",
                "methods": {
                    "raw": "unmapped source score",
                    "I": "pooled 21d causal isotonic",
                    "I_R": "I values; raw score resolves exact plateaus",
                    "I_S": (
                        "75% I + 25% positive-Huber/tanh aligned with "
                        "reference-only median/IQR and clipped to I p01-p99"
                    ),
                    "I_S_R": "I-S values; raw score resolves exact plateaus",
                },
                "mapping": (
                    "same prior-resolved reference identities; label_end in "
                    "[snapshot-21d,snapshot); 2000 minimum; pooled; no side map"
                ),
                "selection": (
                    "one pooled-global top 1/5/10/20 per declared period; "
                    "never per timestamp/side/asset"
                ),
                "arm_selection": (
                    "March03-19 only; safety-gated highest top10 net or ABSTAIN"
                ),
                "confirmation": (
                    "March20-31 and April reused diagnostics; not untouched OOS"
                ),
                "actions": "timing, MAE, wait and target-price layers excluded",
            },
            "source": {
                "bridge_manifest_path": str(manifest_path),
                "bridge_manifest_sha256": sha256(manifest_path),
                "bridge_path": str(bridge_path),
                "bridge_sha256": sha256(bridge_path),
                "raw_path": str(raw_path),
                "raw_sha256": sha256(raw_path),
            },
            "selection_decisions": decision.to_dict("records"),
            "outputs": outputs,
            "outputs_sha256": {
                record["path"]: record["sha256"] for record in outputs.values()
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
                "positive_huber_source_path": str(
                    ROOT
                    / "scripts/run_canonical_execution_reliability_mapping_ablation_v2.py"
                ),
                "positive_huber_source_sha256": sha256(
                    ROOT
                    / "scripts/run_canonical_execution_reliability_mapping_ablation_v2.py"
                ),
            },
        }
        result_path = stage / "manifest.json"
        result_path.write_text(
            json.dumps(safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage / "manifest.sha256").write_text(
            f"{sha256(result_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bridge-root", type=Path, default=BRIDGE)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    print(
        json.dumps(
            safe(run(bridge_root=args.bridge_root, output_dir=args.output_dir)),
            indent=2,
            sort_keys=True,
        )
    )
