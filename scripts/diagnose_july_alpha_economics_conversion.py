#!/usr/bin/env python3
"""Diagnose, without promotion claims, July alpha-to-economics conversion.

This is complementary to the main economics reporter.  It explains whether a
base/residual ordering reaches MFE, gross and net economic outcomes on the
same exact 12-hour policy rows, and quantifies loss from opportunity, payoff,
exit mix, cost and quantized mapped-EV ties.  July rows are retrospective only;
no result is OOS generalization evidence.
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

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "july_alpha_economics_conversion_diagnostic_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
DEFAULT_ROOT = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/july_alpha_economics_conversion_diagnostic_20260730_v1"


class JulyConversionDiagnosticError(RuntimeError):
    """Raised when the exact retrospective inputs cannot be bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_output_path(manifest: Mapping[str, Any], key: str) -> Path:
    record = manifest.get(key, {})
    value = record.get("path") if isinstance(record, Mapping) else None
    if not value:
        raise JulyConversionDiagnosticError(f"manifest has no {key}.path")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_bound(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored_root = root / "scored"
    labels_root = root / "labels_12h"
    scored_manifest_path = scored_root / "manifest.json"
    labels_manifest_path = labels_root / "manifest.json"
    if not scored_manifest_path.is_file() or not labels_manifest_path.is_file():
        raise JulyConversionDiagnosticError("scored and exact-policy label manifests are required")
    scored_manifest = json.loads(scored_manifest_path.read_text())
    labels_manifest = json.loads(labels_manifest_path.read_text())
    if (
        scored_manifest.get("schema") != "execution_ev_retrospective_scored_population_v1"
        or scored_manifest.get("status") != "research_only_retrospective_nonpromotable_not_forward_or_oos_evidence"
        or labels_manifest.get("schema") != "execution_ev_deployed_policy_1m_labels_v1"
        or int(labels_manifest.get("output", {}).get("rows", -1)) != 5760
    ):
        raise JulyConversionDiagnosticError("requires the exact v2 retrospective scored and 5,760-row policy-label artifacts")
    scored_path = scored_root / "scored_population.parquet"
    labels_path = _manifest_output_path(labels_manifest, "output")
    if not scored_path.is_file() or not labels_path.is_file():
        raise JulyConversionDiagnosticError("bound scored or labels parquet is missing")
    if labels_manifest["output"].get("sha256") != _sha256(labels_path):
        raise JulyConversionDiagnosticError("exact policy labels hash changed")
    scored = pd.read_parquet(scored_path)
    labels = pd.read_parquet(labels_path)
    required_scored = {*IDENTITY, "base_oof_score", "existing_alpha_ev", "mapped_execution_ev"}
    required_labels = {
        *IDENTITY, "execution_mfe_return_12h", "execution_gross_ev_12h",
        "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason",
    }
    if required_scored.difference(scored) or required_labels.difference(labels):
        raise JulyConversionDiagnosticError("scored or labels input lacks required conversion fields")
    for frame in (scored, labels):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if frame.duplicated(list(IDENTITY)).any() or frame["candidate_id"].duplicated().any():
            raise JulyConversionDiagnosticError("input candidate identities are not one-to-one")
    joined = scored.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one", suffixes=("", "_label"))
    if len(joined) != len(scored) or len(joined) != len(labels) or len(joined) != 5760:
        raise JulyConversionDiagnosticError("scored and exact-policy labels do not have the same 5,760 identities")
    for field in ("base_oof_score", "existing_alpha_ev", "mapped_execution_ev", "execution_mfe_return_12h", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        joined[field] = pd.to_numeric(joined[field], errors="coerce")
        if not np.isfinite(joined[field].to_numpy(dtype=float)).all():
            raise JulyConversionDiagnosticError(f"non-finite conversion field: {field}")
    joined["utc_day"] = joined["__ts__"].dt.strftime("%Y-%m-%d")
    return joined, {
        "scored_manifest": {"path": str(scored_manifest_path), "sha256": _sha256(scored_manifest_path)},
        "scored_population": {"path": str(scored_path), "sha256": _sha256(scored_path)},
        "labels_manifest": {"path": str(labels_manifest_path), "sha256": _sha256(labels_manifest_path)},
        "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
    }


def _rank_ic(frame: pd.DataFrame, score: str, target: str) -> float | None:
    if len(frame) < 3 or frame[score].nunique() < 2 or frame[target].nunique() < 2:
        return None
    value = frame[score].corr(frame[target], method="spearman")
    return float(value) if np.isfinite(value) else None


def _scope_metrics(frame: pd.DataFrame, *, score: str) -> dict[str, Any]:
    gross = frame["execution_gross_ev_12h"]
    net = frame["execution_net_ev_12h"]
    cost = frame["execution_cost_return"]
    favorable = net.gt(0.0)
    adverse = net.lt(0.0)
    exits = frame["execution_exit_reason"].astype(str).value_counts(normalize=True)
    return {
        "rows": int(len(frame)),
        "rank_ic": {
            "mfe": _rank_ic(frame, score, "execution_mfe_return_12h"),
            "gross": _rank_ic(frame, score, "execution_gross_ev_12h"),
            "net": _rank_ic(frame, score, "execution_net_ev_12h"),
            "cost": _rank_ic(frame, score, "execution_cost_return"),
        },
        "opportunity_incidence": float(gross.gt(cost).mean()),
        "favorable_payoff_bps": float(net.loc[favorable].mean() * 1e4) if favorable.any() else None,
        "adverse_payoff_bps": float(net.loc[adverse].mean() * 1e4) if adverse.any() else None,
        "favorable_fraction": float(favorable.mean()),
        "mean_mfe_bps": float(frame["execution_mfe_return_12h"].mean() * 1e4),
        "mean_gross_bps": float(gross.mean() * 1e4),
        "mean_cost_bps": float(cost.mean() * 1e4),
        "mean_net_bps": float(net.mean() * 1e4),
        "exit_mixture": {str(k): float(v) for k, v in exits.items()},
    }


def _daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for day, daily in frame.groupby("utc_day", sort=True):
        for scope, local in [("pooled_global", daily), *[(f"side_{side}", rows) for side, rows in daily.groupby("side_name", sort=True)]]:
            for score in ("base_oof_score", "existing_alpha_ev"):
                metric = _scope_metrics(local, score=score)
                records.append({"utc_day": day, "scope": scope, "score": score, **metric})
    return pd.DataFrame(records)


def _cells(frame: pd.DataFrame, *, minimum_rows: int = 12) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (day, side), local in frame.groupby(["utc_day", "side_name"], sort=True):
        work = local.copy()
        work["base_score_decile"] = pd.qcut(
            work["base_oof_score"].rank(method="first"), q=10, labels=False
        ).astype(int)
        book = work["policy_archetype"].astype(str) if "policy_archetype" in work else pd.Series("unknown", index=work.index)
        work["book_cell"] = book
        for (decile, cell), group in work.groupby(["base_score_decile", "book_cell"], observed=True):
            if len(group) < minimum_rows:
                continue
            metric = _scope_metrics(group, score="base_oof_score")
            records.append({"utc_day": day, "side_name": side, "base_score_decile": int(decile), "book_cell": str(cell), **metric})
    return pd.DataFrame(records)


def _top_with_tie_bounds(frame: pd.DataFrame, *, fraction: float = 0.10, bootstrap_draws: int = 1_000) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = [("all_july", frame), *[(str(day), local) for day, local in frame.groupby("utc_day", sort=True)]]
    for scope, local in scopes:
        k = max(1, int(math.ceil(len(local) * fraction)))
        ordered = local.sort_values(["mapped_execution_ev", "candidate_id"], ascending=[False, True], kind="mergesort")
        selected = ordered.head(k)
        cutoff = float(selected["mapped_execution_ev"].iloc[-1])
        above = local.loc[local["mapped_execution_ev"].gt(cutoff)]
        tied = local.loc[local["mapped_execution_ev"].eq(cutoff)].sort_values("candidate_id", kind="mergesort")
        slots = k - len(above)
        deterministic = pd.concat([above, tied.head(slots)], ignore_index=True)
        best = pd.concat([above, tied.nlargest(slots, "execution_net_ev_12h")], ignore_index=True)
        worst = pd.concat([above, tied.nsmallest(slots, "execution_net_ev_12h")], ignore_index=True)
        rng = np.random.default_rng(20260730 + len(local))
        bootstrap = np.empty(int(bootstrap_draws), dtype=float)
        tie_values = tied["execution_net_ev_12h"].to_numpy(dtype=float)
        above_sum = float(above["execution_net_ev_12h"].sum())
        for index in range(int(bootstrap_draws)):
            bootstrap[index] = (above_sum + rng.choice(tie_values, size=slots, replace=False).sum()) / k
        records.append({
            "scope": scope, "rows": int(len(local)), "top_k": k,
            "mapped_ev_unique_levels": int(local["mapped_execution_ev"].nunique()),
            "cutoff_mapped_ev": cutoff, "above_cutoff_rows": int(len(above)),
            "cutoff_tie_rows": int(len(tied)), "slots_from_cutoff_tie": int(slots),
            "arbitrary_candidate_id_tie_break": bool(len(tied) > slots),
            "deterministic_selected_net_bps": float(deterministic.execution_net_ev_12h.mean() * 1e4),
            "best_tie_selected_net_bps": float(best.execution_net_ev_12h.mean() * 1e4),
            "worst_tie_selected_net_bps": float(worst.execution_net_ev_12h.mean() * 1e4),
            "tie_selection_sensitivity_bps": float((best.execution_net_ev_12h.mean() - worst.execution_net_ev_12h.mean()) * 1e4),
            "tie_bootstrap_draws": int(bootstrap_draws),
            "tie_bootstrap_net_bps_p05": float(np.quantile(bootstrap, 0.05) * 1e4),
            "tie_bootstrap_net_bps_p50": float(np.quantile(bootstrap, 0.50) * 1e4),
            "tie_bootstrap_net_bps_p95": float(np.quantile(bootstrap, 0.95) * 1e4),
            "positive_floor_admissions": int(local["mapped_execution_ev"].ge(0.0).sum()),
            "positive_floor_tie_affected": bool(np.isclose(cutoff, 0.0) and len(tied) > slots),
        })
    return pd.DataFrame(records)


def run(*, root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frame, inputs = _load_bound(root)
    daily = _daily_metrics(frame)
    cells = _cells(frame)
    ties = _top_with_tie_bounds(frame)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        daily_path = stage / "daily_side_conversion_metrics.csv"; daily.to_csv(daily_path, index=False)
        cells_path = stage / "decile_book_cell_effects.csv"; cells.to_csv(cells_path, index=False)
        ties_path = stage / "mapped_ev_cutoff_ties.csv"; ties.to_csv(ties_path, index=False)
        report = {
            "schema": SCHEMA, "status": "research_diagnostic_non_promotion",
            "scope": "July 20-23 exact 12h policy outcomes; retrospective comparison only, not OOS generalization evidence.",
            "mechanism": "base/residual rank IC is decomposed against MFE, gross, cost and net; incidence/payoff/exit/cost are descriptive components, not causal estimates.",
            "next_ablations": [
                "Train conversion heads for opportunity incidence and conditional favorable/adverse payoff separately.",
                "Keep mapped-EV ranking continuous or apply deterministic secondary ranking inside isotonic tie levels.",
                "Model exit-mixture/geometry conversion separately from alpha ranking; test only after same-ID exact replay.",
            ],
            "rows": int(len(frame)), "days": sorted(frame.utc_day.unique().tolist()),
            "inputs": inputs,
            "outputs": {
                "daily_metrics": {"path": str(output_dir / daily_path.name), "sha256": _sha256(daily_path), "rows": int(len(daily))},
                "cells": {"path": str(output_dir / cells_path.name), "sha256": _sha256(cells_path), "rows": int(len(cells))},
                "mapped_ties": {"path": str(output_dir / ties_path.name), "sha256": _sha256(ties_path), "rows": int(len(ties))},
            },
        }
        report_path = stage / "report.json"; _write_json(report_path, report)
        manifest = {"schema": SCHEMA, "status": "research_diagnostic_non_promotion", "report_sha256": _sha256(report_path), "outputs": report["outputs"], "inputs": inputs}
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(_safe(run(root=args.root, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
