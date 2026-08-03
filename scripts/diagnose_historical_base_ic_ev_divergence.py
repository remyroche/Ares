#!/usr/bin/env python3
"""Immutable March/April diagnosis of base rank IC versus exact execution EV."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ID = ["candidate_id", "side_name", "__symbol__", "__ts__"]
TOP_FRACTION = 0.10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def _gate_loader():
    path = Path(__file__).with_name("run_historical_execution_ev_add_drop_gate.py")
    spec = importlib.util.spec_from_file_location("gate_for_ic_ev", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _spearman(rows: pd.DataFrame, column: str) -> float | None:
    value = rows.historical_base_soft_oof.corr(rows[column], method="spearman")
    return float(value) if np.isfinite(value) else None


def _economics(rows: pd.DataFrame) -> dict:
    count = int(np.ceil(len(rows) * TOP_FRACTION))
    picked = rows.nlargest(count, "historical_base_soft_oof")
    gross_oracle = set(rows.nlargest(count, "execution_gross_ev_12h").candidate_id)
    net_oracle = set(rows.nlargest(count, "execution_net_ev_12h").candidate_id)
    chosen = set(picked.candidate_id)
    return {
        "rows": len(picked), "gross_bps": float(picked.execution_gross_ev_12h.mean() * 1e4),
        "cost_bps": float(picked.execution_cost_return.mean() * 1e4), "net_bps": float(picked.execution_net_ev_12h.mean() * 1e4),
        "median_net_bps": float(picked.execution_net_ev_12h.median() * 1e4), "positive_net_precision": float(picked.execution_net_ev_12h.gt(0).mean()),
        "gross_exceeds_cost_rate": float(picked.execution_gross_ev_12h.gt(picked.execution_cost_return).mean()),
        "gross_oracle_recall": float(len(chosen & gross_oracle) / len(gross_oracle)), "net_oracle_recall": float(len(chosen & net_oracle) / len(net_oracle)),
        "side_capacity": [{"side": str(side), "rows": int(size)} for side, size in picked.groupby("side_name").size().items()],
        "exit_composition_selected": picked.execution_exit_reason.value_counts(normalize=True).rename_axis("exit").to_dict(),
        "exit_composition_all": rows.execution_exit_reason.value_counts(normalize=True).rename_axis("exit").to_dict(),
    }


def _strata(rows: pd.DataFrame, features: list[str]) -> list[dict]:
    records = []
    for feature in features:
        values = pd.to_numeric(rows[feature], errors="coerce")
        labels = pd.qcut(values.rank(method="first"), q=4, labels=["q1", "q2", "q3", "q4"])
        for label, group in rows.assign(_stratum=labels).groupby("_stratum", observed=True):
            records.append({"feature": feature, "stratum": str(label), "rows": len(group), "base_ic_gross": _spearman(group, "execution_gross_ev_12h"), "base_ic_cost": _spearman(group, "execution_cost_return"), "base_ic_net": _spearman(group, "execution_net_ev_12h"), "base_top10_net_bps": _economics(group)["net_bps"]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    partial = args.output_root.with_name(args.output_root.name + ".partial")
    if args.output_root.exists() or partial.exists():
        raise FileExistsError(args.output_root)
    gate_manifest = json.loads((args.gate_root / "manifest.json").read_text())
    context_manifest = json.loads((args.context_root / "manifest.json").read_text())
    if gate_manifest.get("schema") != "historical_execution_ev_add_drop_gate_v6" or context_manifest.get("status") != "IMMUTABLE_PREENTRY_ONLY_INPUT_PANEL":
        raise ValueError("requires frozen v6 gate and repaired immutable pre-entry context v3")
    if not all(_sha(args.context_root / name) == digest for name, digest in context_manifest["outputs_sha256"].items()):
        raise ValueError("context output hash mismatch")
    sources = gate_manifest["sources"]
    for item in sources.values():
        path = Path(item["path"])
        if _sha(path) != item["sha256"]:
            raise ValueError(f"gate source hash mismatch: {path}")
    loader = _gate_loader()
    x = loader.load(SimpleNamespace(residual=Path(sources["residual"]["path"]), context=Path(sources["context"]["path"]), aux=Path(sources["aux"]["path"]), population=Path(sources["population"]["path"]), six_root=Path(sources["six_long"]["path"]).parent.parent, risk_root=Path(sources["risk_long"]["path"]).parent.parent))
    population = pd.read_parquet(Path(sources["population"]["path"]), columns=[*ID, "execution_exit_reason"])
    context = pd.read_parquet(args.context_root / "panel.parquet", columns=[*ID, *context_manifest["feature_columns"]])
    x = x.merge(population, on=ID, validate="one_to_one").merge(context, on=ID, validate="one_to_one")
    if len(x) != 140682:
        raise ValueError("strict paired identity count changed")
    x["month"] = pd.to_datetime(x["__ts__"], utc=True).dt.strftime("%Y-%m")
    strata_features = ["range_24h_pct", "__meta_raw__volatility_zscore", "trend_r2_24", "jump_intensity", "preentry_transition__range_24h_pct__delta_3h", "preentry_transition__meta_raw__volatility_zscore__delta_3h", "preentry_transition__trend_r2_24__delta_3h"]
    result = {"schema": "historical_base_ic_ev_divergence_v1", "status": "diagnostic_non_promotion", "contract": "Base score is assessed against exact 12h gross, realized cost, and net; top10 is global within each month, never per timestamp. Regime/context fields are repaired v3 pre-entry-only data.", "months": {}}
    for month, rows in x.groupby("month", sort=True):
        result["months"][month] = {"rows": len(rows), "base_rank_ic": {"gross": _spearman(rows, "execution_gross_ev_12h"), "cost": _spearman(rows, "execution_cost_return"), "net": _spearman(rows, "execution_net_ev_12h")}, "base_global_top10": _economics(rows), "by_side": {side: {"rows": len(group), "gross_ic": _spearman(group, "execution_gross_ev_12h"), "cost_ic": _spearman(group, "execution_cost_return"), "net_ic": _spearman(group, "execution_net_ev_12h"), "top10_local_diagnostic": _economics(group)} for side, group in rows.groupby("side_name")}, "context_strata": _strata(rows, strata_features)}
    partial.mkdir(parents=True)
    report = partial / "report.json"; _json(report, result)
    manifest = {"schema": "historical_base_ic_ev_divergence_manifest_v1", "status": "diagnostic_non_promotion", "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha(Path(__file__).resolve())}, "gate_manifest": {"path": str(args.gate_root / "manifest.json"), "sha256": _sha(args.gate_root / "manifest.json")}, "context_manifest": {"path": str(args.context_root / "manifest.json"), "sha256": _sha(args.context_root / "manifest.json")}, "output_sha256": {"report.json": _sha(report)}}
    _json(partial / "manifest.json", manifest); partial.replace(args.output_root)


if __name__ == "__main__":
    main()
