#!/usr/bin/env python3
"""Aggregate completed strict M6 cross-era cells without refitting any model.

The cell runner owns all fitting.  This finalizer only validates the complete
upper-triangular matrix, joins the per-cell metrics and shift diagnostics, and
writes a compact, auditable report.  It intentionally does not pool cell
predictions: those remain in their source cell directories to preserve each
train/test lineage and avoid a very large duplicate ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_tp6_m6_cross_era_transport import CONTEXT, ERAS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELLS = ROOT / "data_perp/artifacts/tp6_m6_cross_era_transport_20260809_v4_cells"
DEFAULT_STAGE = ROOT / "data_perp/artifacts/tp6_m6_cross_era_transport_20260809_v4_stage"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_m6_cross_era_transport_20260809_v4"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _required_cells() -> list[tuple[str, str, str]]:
    names = [x[0] for x in ERAS]
    return [(mode, names[i], test) for mode in ("single_era", "expanding_prefix")
            for i in range(len(names) - 1) for test in names[i + 1:]]


def _cell_dir(root: Path, mode: str, train: str, test: str) -> Path:
    return root / f"{mode}__{train}__{test}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    required = _required_cells()
    files = ("metrics.parquet", "covariate_shift.parquet", "concept_shift.parquet", "predictions.parquet")
    missing = [str(_cell_dir(args.cells, *cell)) for cell in required
               if not all((_cell_dir(args.cells, *cell) / f).is_file() for f in files)]
    if missing:
        raise RuntimeError(f"incomplete matrix: {len(missing)} missing cells; first={missing[:3]}")

    metrics = pd.concat([pd.read_parquet(_cell_dir(args.cells, *cell) / "metrics.parquet") for cell in required], ignore_index=True)
    covariate = pd.concat([pd.read_parquet(_cell_dir(args.cells, *cell) / "covariate_shift.parquet") for cell in required], ignore_index=True)
    concept = pd.concat([pd.read_parquet(_cell_dir(args.cells, *cell) / "concept_shift.parquet") for cell in required], ignore_index=True)
    if metrics.duplicated(["mode", "train_era", "test_era", "view", "metric", "top_fraction"]).any():
        raise RuntimeError("duplicate metric rows across cells")
    if set(covariate.feature.unique()) != set([*CONTEXT, "p_adverse", "p_weak", "p_clear", "base_raw"]):
        raise RuntimeError("covariate output does not match frozen feature contract")

    # Label shift belongs to a test era, not a fit; materialise it once from
    # the prepared immutable cohorts.
    label_rows = []
    for era, *_ in ERAS:
        x = pd.read_parquet(args.stage / f"{era}.parquet", columns=["side_name", "event", "net_bps", "p_clear"])
        for side, z in x.groupby("side_name", sort=True):
            label_rows.append({"era": era, "side_name": side, "n": len(z),
                               "p_net_gt_50": float(z.event.mean()), "mean_net_bps": float(z.net_bps.mean()),
                               "mean_net_given_positive": float(z.loc[z.event.eq(1), "net_bps"].mean()),
                               "mean_base_p_clear": float(z.p_clear.mean())})
    label_shift = pd.DataFrame(label_rows)

    # Compact matrix provides the decision evidence without discarding the
    # full per-metric table.  Ranking is global, by contract.
    top = metrics[(metrics.view.eq("global")) & (metrics.metric.eq("top")) & metrics.top_fraction.isin([.01, .05, .10])].copy()
    matrix = top.pivot_table(index=["mode", "train_era"], columns=["test_era", "top_fraction"], values="net_bps", aggfunc="first").sort_index()
    matrix.columns = [f"{era}__top{int(frac * 100)}_net_bps" for era, frac in matrix.columns]
    matrix = matrix.reset_index()
    shift_summary = covariate.groupby(["mode", "train_era", "test_era"], as_index=False).agg(
        median_psi=("psi", "median"), max_psi=("psi", "max"), median_wasserstein=("wasserstein", "median"),
        max_test_missing=("test_missing", "max"), max_train_missing=("train_missing", "max"))
    # Cell-level shift diagnostics are repeated in every per-side concept row.
    shifts = concept.groupby(["mode", "train_era", "test_era"], as_index=False).agg(
        adversarial_auc_in_sample=("adversarial_auc_in_sample", "first"),
        correlation_frobenius_shift=("correlation_frobenius_shift", "first"))
    shift_summary = shift_summary.merge(shifts, on=["mode", "train_era", "test_era"], validate="one_to_one")

    args.out.mkdir(parents=True)
    metrics.to_parquet(args.out / "transport_metrics.parquet", index=False)
    matrix.to_parquet(args.out / "transport_matrix.parquet", index=False)
    covariate.to_parquet(args.out / "covariate_shift.parquet", index=False)
    shift_summary.to_parquet(args.out / "covariate_shift_summary.parquet", index=False)
    concept.to_parquet(args.out / "concept_shift.parquet", index=False)
    label_shift.to_parquet(args.out / "label_shift.parquet", index=False)

    top1 = top[top.top_fraction.eq(.01)].sort_values(["mode", "train_era", "test_era"])
    lines = ["# Strict TP6/SL4/H12 M6 cross-era transport matrix", "", "## Frozen contract", "",
             "- Exact TP6/SL4/H12 outcome, fixed 100 bps round-trip cost; M6 event is exact net > +50 bps.",
             "- Each base score is pre-existing, strict chronological, same-side R3 OOF. Each M6 cell fits separately by side, then ranks the test population globally.",
             "- The matrix uses 14 high-coverage causal context fields plus same-side base outputs. No sparse-field imputation; no 2022 pooling into the incompatible 2023–24 contract.",
             "- All 56 strictly later train/test cells are present. This finalizer refits nothing.", "", "## Global top-1% net bps / trade", "",
             "| Mode | Train era | Test era | Net bps | ROC-AUC | PR-AUC | Net IC |", "|---|---|---|---:|---:|---:|---:|"]
    for _, r in top1.iterrows():
        lines.append(f"| {r.mode} | {r.train_era} | {r.test_era} | {r.net_bps:+.2f} | {r.roc_auc:.3f} | {r.pr_auc:.3f} | {r.score_net_ic:+.3f} |")
    lines += ["", "## Interpretation boundary", "", "A positive cell is not a promotable conversion head. Transport must be assessed across the full matrix, including worst-era net results, score-bin residuals, label shift, and covariate shift. The 2022 inverse-PI cohort remains a separately reported external transport diagnostic because its candidate/product/context schema is incompatible; it is neither pooled nor zero-imputed."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {"schema": "tp6_m6_cross_era_transport_v4", "status": "COMPLETED_DIAGNOSTIC", "geometry": "TP6/SL4/H12", "cost_bps": 100,
                "m6_target": "exact net > +50 bps", "matrix_cells": len(required), "modes": ["single_era", "expanding_prefix"],
                "eras": [x[0] for x in ERAS], "context": CONTEXT,
                "base_lineage": "pre-existing strict chronological same-side R3 OOF", "ranking": "globally pooled after side-local M6 scoring",
                "no_imputation": True, "external_2022": {"status": "separate_noncombinable", "reason": "inverse-PI candidate/product/context schema differs", "prior_top1_net_bps": -46.53},
                "files": {"cell_root": str(args.cells), "stage": str(args.stage), "finalizer_sha256": _sha(Path(__file__))}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cells": len(required), "metric_rows": len(metrics), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
