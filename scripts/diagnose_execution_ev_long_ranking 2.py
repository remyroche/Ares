#!/usr/bin/env python3
"""Identical-row long-side ranking decomposition for execution EV.

This is a diagnostic, not a model-selection runner.  It keeps the canonical
candidate rows and exact-policy outcomes fixed, selects one pooled global book
for each score, and then reports the long contribution to that unchanged book.
A separate long-only top-decile is emitted only as a diagnostic slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
TARGET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
MFE = "execution_mfe_return_12h"
MAE = "execution_mae_return_12h"
EXIT = "execution_exit_reason"
SCORES: Mapping[str, tuple[str, bool]] = {
    "base_rank": ("base_candidate_rank_pct_timestamp_side", False),
    "raw_execution_ev": (
        "catboost__residual__without_hpo__all_features",
        True,
    ),
    "causal_global_21d_ev": ("causal_recent_isotonic_ev", True),
    "causal_side_21d_ev": ("causal_recent_side_isotonic_ev", True),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _select_top_fraction(
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool,
    fraction: float,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite ranking score: {column}")
    count = max(1, int(np.ceil(fraction * len(frame))))
    order = np.argsort(
        -values if higher_is_better else values,
        kind="mergesort",
    )
    selected = np.zeros(len(frame), dtype=bool)
    selected[order[:count]] = True
    return selected


def _metrics(
    frame: pd.DataFrame,
    selected: np.ndarray,
    *,
    score_name: str,
    selection_scope: str,
    report_slice: str,
) -> dict[str, Any]:
    selected_frame = frame.loc[selected]
    net = selected_frame[TARGET].to_numpy(float)
    gross = selected_frame[GROSS].to_numpy(float)
    cost = selected_frame[COST].to_numpy(float)
    return {
        "score": score_name,
        "selection_scope": selection_scope,
        "report_slice": report_slice,
        "population_rows": int(len(frame)),
        "selected_rows": int(len(selected_frame)),
        "mean_gross_bps": float(np.mean(gross) * 10_000.0),
        "mean_cost_bps": float(np.mean(cost) * 10_000.0),
        "mean_net_bps": float(np.mean(net) * 10_000.0),
        "positive_net_rate": float(np.mean(net > 0.0)),
        "mean_mfe_bps": float(selected_frame[MFE].mean() * 10_000.0),
        "mean_mae_bps": float(selected_frame[MAE].mean() * 10_000.0),
        "full_stop_rate": float(
            selected_frame[EXIT].astype(str).eq("full_stop").mean()
        ),
        "timeout_rate": float(
            selected_frame[EXIT].astype(str).eq("timeout").mean()
        ),
    }


def _deciles(
    frame: pd.DataFrame,
    *,
    score_name: str,
    column: str,
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    work = frame.loc[frame["side_name"].astype(str).eq("long")].copy()
    rank = work[column].rank(
        method="first", ascending=not higher_is_better, pct=True
    )
    work["score_decile"] = np.minimum(
        np.floor((rank.to_numpy(float) - np.finfo(float).eps) * 10.0),
        9,
    ).astype(int)
    rows: list[dict[str, Any]] = []
    for decile, part in work.groupby("score_decile", sort=True):
        rows.append(
            {
                "score": score_name,
                "score_decile": int(decile),
                "rows": int(len(part)),
                "mean_score": float(part[column].mean()),
                "mean_gross_bps": float(part[GROSS].mean() * 10_000.0),
                "mean_cost_bps": float(part[COST].mean() * 10_000.0),
                "mean_net_bps": float(part[TARGET].mean() * 10_000.0),
                "positive_net_rate": float((part[TARGET] > 0.0).mean()),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = pd.read_parquet(args.input)
    required = {
        *IDENTITY,
        TARGET,
        GROSS,
        COST,
        MFE,
        MAE,
        EXIT,
        *(column for column, _ in SCORES.values()),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("identical-row diagnostic requires unique identities")
    accounting_error = np.max(
        np.abs(
            frame[GROSS].to_numpy(float)
            - frame[COST].to_numpy(float)
            - frame[TARGET].to_numpy(float)
        )
    )
    if accounting_error > 1e-7:
        raise ValueError(f"gross-cost-net mismatch: {accounting_error}")
    frame = frame.reset_index(drop=True)
    long_mask = frame["side_name"].astype(str).eq("long").to_numpy()
    long_frame = frame.loc[long_mask].reset_index(drop=True)

    metric_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    selection_sets: dict[str, set[tuple[Any, ...]]] = {}
    for score_name, (column, higher_is_better) in SCORES.items():
        global_selected = _select_top_fraction(
            frame,
            column,
            higher_is_better=higher_is_better,
            fraction=args.top_fraction,
        )
        metric_rows.append(
            _metrics(
                frame,
                global_selected,
                score_name=score_name,
                selection_scope="one_pooled_global_top_fraction",
                report_slice="all_selected",
            )
        )
        metric_rows.append(
            _metrics(
                frame,
                global_selected & long_mask,
                score_name=score_name,
                selection_scope="one_pooled_global_top_fraction",
                report_slice="long_contribution_to_same_book",
            )
        )
        long_selected = _select_top_fraction(
            long_frame,
            column,
            higher_is_better=higher_is_better,
            fraction=args.top_fraction,
        )
        metric_rows.append(
            _metrics(
                long_frame,
                long_selected,
                score_name=score_name,
                selection_scope="long_only_top_fraction_diagnostic",
                report_slice="long_selected",
            )
        )
        selected_identity = frame.loc[
            global_selected, list(IDENTITY)
        ].itertuples(index=False, name=None)
        selection_sets[score_name] = set(selected_identity)
        decile_rows.extend(
            _deciles(
                frame,
                score_name=score_name,
                column=column,
                higher_is_better=higher_is_better,
            )
        )

    overlap_rows: list[dict[str, Any]] = []
    baseline = selection_sets["base_rank"]
    for score_name, identities in selection_sets.items():
        overlap = len(baseline & identities)
        union = len(baseline | identities)
        overlap_rows.append(
            {
                "baseline": "base_rank",
                "challenger": score_name,
                "baseline_rows": len(baseline),
                "challenger_rows": len(identities),
                "overlap_rows": overlap,
                "jaccard": overlap / union if union else 1.0,
            }
        )

    args.output_dir.mkdir(parents=True)
    outputs = {
        "ranking_metrics": args.output_dir / "ranking_metrics.csv",
        "long_calibration_deciles": args.output_dir
        / "long_calibration_deciles.csv",
        "selection_overlap": args.output_dir / "selection_overlap.csv",
    }
    pd.DataFrame(metric_rows).to_csv(outputs["ranking_metrics"], index=False)
    pd.DataFrame(decile_rows).to_csv(
        outputs["long_calibration_deciles"], index=False
    )
    pd.DataFrame(overlap_rows).to_csv(outputs["selection_overlap"], index=False)
    manifest = {
        "schema": "execution_ev_identical_row_long_ranking_v1",
        "status": "diagnostic_only_not_promotion_evidence",
        "input": {"path": str(args.input), "sha256": _sha256(args.input)},
        "contract": {
            "identities": list(IDENTITY),
            "rows": int(len(frame)),
            "long_rows": int(long_mask.sum()),
            "top_fraction": float(args.top_fraction),
            "primary_ranking": (
                "one pooled global book across timestamps and sides; side rows "
                "are only an after-selection contribution"
            ),
            "long_only_scope": "diagnostic_only",
            "cost_accounting": "gross - exact row cost = net",
            "score_orientation": {
                name: "higher_is_better" if high else "lower_is_better"
                for name, (_, high) in SCORES.items()
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "execution_ev_economic_failure_diagnosis_20260727_v2/"
            "diagnostic_rows.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(_parser()), indent=2))
