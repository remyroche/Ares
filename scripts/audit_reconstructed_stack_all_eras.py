#!/usr/bin/env python3
"""Audit all existing 2022--24 reconstructed OOF score cohorts without mixing contracts.

The report intentionally keeps the inverse-perpetual 2022H1 cohort separate
from the frozen linear-perpetual cohort.  Pooled-global rankings are computed
only within a compatible reporting cohort and are descriptive OOF diagnostics,
not a causal entry rule.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4/oof_scores.parquet"
OUT = ROOT / "data_perp/artifacts/reconstructed_stack_all_eras_audit_20260731_v1"
TARGET = "execution_net_ev_12h"
ALPHA = "__reconstructed_soft_alpha_12h__"
SCORES = ("score_base_alpha", "score_residual_alpha", "score_base_expected_ev", "score_residual_expected_ev")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def rank_ic(score: pd.Series, target: pd.Series) -> float:
    return float(score.rank().corr(target.rank()))


def top(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, float | int]:
    rows = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(int(np.ceil(len(frame) * fraction)))
    return {"rows": int(len(rows)), "net_bps": float(rows[TARGET].mean() * 1e4), "gross_bps": float(rows.execution_gross_ev_12h.mean() * 1e4), "cost_bps": float(rows.execution_cost_return.mean() * 1e4), "positive_net_fraction": float(rows[TARGET].gt(0).mean()), "long_share": float(rows.side_name.eq("long").mean())}


def summarize(frame: pd.DataFrame, cohort: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    months: list[dict[str, object]] = []
    for score in SCORES:
        record: dict[str, object] = {"cohort": cohort, "score": score, "rows": int(len(frame)), "alpha_rank_ic": rank_ic(frame[score], frame[ALPHA]), "net_rank_ic": rank_ic(frame[score], frame[TARGET])}
        if score.endswith("expected_ev"):
            selected = frame.loc[frame[score].gt(0.0)]
            record.update({"zero_threshold_rows": int(len(selected)), "zero_threshold_net_bps": float(selected[TARGET].mean() * 1e4) if len(selected) else None})
        for fraction in (.01, .05, .10):
            record[f"top_{int(fraction*100)}"] = top(frame, score, fraction)
        summary.append(record)
        for month, local in frame.groupby(frame["__ts__"].dt.strftime("%Y-%m"), sort=True):
            metric = top(local, score, .10)
            months.append({"cohort": cohort, "score": score, "month": month, **metric})
    return summary, months


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    data = pd.read_parquet(SOURCE)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="raise")
    if not data.residual_is_oof.astype(bool).all():
        raise ValueError("source has non-OOF residual rows")
    if not np.allclose(data.execution_gross_ev_12h - data.execution_cost_return, data[TARGET], atol=1e-12, rtol=0.0):
        raise ValueError("gross minus cost assertion failed")
    cohorts = {
        "inverse_pi_2022h1_separate_contract": data.stack_lineage.eq("inverse_pi_2022_h1"),
        "linear_pf_2022h2_2023": data.stack_lineage.eq("frozen_pf_2022aug_2024") & data["__ts__"].dt.year.le(2023),
        "linear_pf_2024": data.stack_lineage.eq("frozen_pf_2022aug_2024") & data["__ts__"].dt.year.eq(2024),
        "linear_pf_2022h2_2024_compatible": data.stack_lineage.eq("frozen_pf_2022aug_2024"),
    }
    aggregate: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    coverage: dict[str, object] = {}
    for name, mask in cohorts.items():
        local = data.loc[mask].copy()
        coverage[name] = {"rows": int(len(local)), "start": str(local["__ts__"].min()), "end": str(local["__ts__"].max()), "months": int(local["__ts__"].dt.strftime("%Y-%m").nunique())}
        a, b = summarize(local, name)
        aggregate.extend(a); monthly.extend(b)
    temporary = OUT.with_name(f".{OUT.name}.{os.getpid()}.partial")
    temporary.mkdir(parents=True)
    try:
        pd.json_normalize(aggregate, sep="__").to_csv(temporary / "aggregate_metrics.csv", index=False)
        pd.DataFrame(monthly).to_csv(temporary / "monthly_global_top10.csv", index=False)
        report = {"schema": "reconstructed_stack_all_eras_audit_v1", "status": "RESEARCH_ONLY_OOF_DIAGNOSTIC", "source": {"path": str(SOURCE), "sha256": sha(SOURCE)}, "coverage": coverage, "contract_separation": "2022H1 inverse PI remains separate; only frozen linear-PF 2022H2-2024 is a compatible aggregate", "selection": "pooled global top-k within each cohort only; no timestamp or per-side ranking", "limitation": "frozen base backcast and frozen/current-spread costs are diagnostic, not historical execution parity or promotion evidence"}
        write(temporary / "report.json", report)
        manifest = {**report, "outputs_sha256": {name: sha(temporary / name) for name in ("aggregate_metrics.csv", "monthly_global_top10.csv", "report.json")}}
        write(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(f"{sha(temporary / 'manifest.json')}  manifest.json\n")
        os.replace(temporary, OUT)
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
