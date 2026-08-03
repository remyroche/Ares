#!/usr/bin/env python3
"""Immutable, source-verified mapping diagnostics from the frozen v6 gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

ARMS = ("residual_only", "base_residual", "plus_risk", "plus_peak", "plus_six")
TOP_FRACTION = 0.10
RELIABILITY_MIN_ROWS = 5_000
RELIABILITY_CORRELATION_THRESHOLD = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def fit_iso(x: pd.Series, y: pd.Series) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(x, y)


def _jaccard(a: set[str], b: set[str]) -> float | None:
    union = a | b
    return None if not union else len(a & b) / len(union)


def adjacent_selected_asset_turnover(
    all_rows: pd.DataFrame, selected: pd.Series, frequency: str
) -> dict[str, float | int]:
    """Jaccard turnover of selected *assets*, including empty chronological buckets.

    Candidate IDs intentionally include timestamp and side, so they are unsuitable for
    this stability diagnostic.  Empty/empty comparisons are omitted: there is no
    selected portfolio for which a turnover claim can be made.
    """
    required = {"__ts__", "__symbol__"}
    if missing := required - set(all_rows.columns):
        raise ValueError(f"turnover inputs missing {sorted(missing)}")
    rows = all_rows.loc[:, ["__ts__", "__symbol__"]].copy()
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True)
    rows["bucket"] = rows["__ts__"].dt.floor(frequency)
    rows["selected"] = selected.to_numpy(dtype=bool)
    buckets = pd.date_range(rows.bucket.min(), rows.bucket.max(), freq=frequency)
    selected_assets = {
        bucket: set(group.loc[group.selected, "__symbol__"].astype(str))
        for bucket, group in rows.groupby("bucket", sort=True)
    }
    overlaps = [
        _jaccard(selected_assets.get(left, set()), selected_assets.get(right, set()))
        for left, right in zip(buckets, buckets[1:])
    ]
    usable = [value for value in overlaps if value is not None]
    mean = float(np.mean(usable)) if usable else 0.0
    return {
        "comparisons": len(usable),
        "selected_asset_jaccard_mean": mean,
        "selected_asset_turnover": 1.0 - mean if usable else 0.0,
    }


def metric(all_rows: pd.DataFrame, selected: pd.Series) -> dict[str, object]:
    selection = all_rows.loc[selected].copy()
    count = len(selection)
    gross_oracle = set(all_rows.nlargest(count, "execution_gross_ev_12h").candidate_id)
    net_oracle = set(all_rows.nlargest(count, "execution_net_ev_12h").candidate_id)
    picked = set(selection.candidate_id)
    value = selection.execution_net_ev_12h
    return {
        "rows": count,
        "gross_bps": float(selection.execution_gross_ev_12h.mean() * 1e4),
        "cost_bps": float(selection.execution_cost_return.mean() * 1e4),
        "net_bps": float(value.mean() * 1e4),
        "median_net_bps": float(value.median() * 1e4),
        "positive_net_precision": float(value.gt(0).mean()),
        "gross_exceeds_cost_rate": float(
            selection.execution_gross_ev_12h.gt(selection.execution_cost_return).mean()
        ),
        "gross_oracle_recall": float(len(picked & gross_oracle) / len(gross_oracle)),
        "net_oracle_recall": float(len(picked & net_oracle) / len(net_oracle)),
        "adjacent_hour_selected_asset": adjacent_selected_asset_turnover(
            all_rows, selected, "h"
        ),
        "adjacent_day_selected_asset": adjacent_selected_asset_turnover(
            all_rows, selected, "D"
        ),
        "side_capacity": [
            {"side": str(side), "rows": int(rows)}
            for side, rows in selection.groupby("side_name").size().items()
        ],
    }


def _expected_output(manifest: dict, arm: str, name: str) -> dict:
    try:
        return manifest["outputs"][arm][name]
    except KeyError as error:
        raise ValueError(f"v6 manifest lacks {arm}/{name}") from error


def validate_gate_manifest(gate_root: Path, manifest: dict, arms: tuple[str, ...] = ARMS) -> dict:
    if manifest.get("schema") != "historical_execution_ev_add_drop_gate_v6":
        raise ValueError("mapping repair requires the historical_execution_ev_add_drop_gate_v6 manifest")
    if manifest.get("status") not in {"COMPLETE", "research_only_diagnostic"}:
        raise ValueError("v6 gate must be complete or explicitly diagnostic")
    validated: dict[str, dict[str, dict[str, object]]] = {}
    for arm in arms:
        validated[arm] = {}
        for name in ("march_inner_oof_scores", "april_outer_predictions"):
            expected = _expected_output(manifest, arm, name)
            path = gate_root / expected["path"]
            if not path.is_file():
                raise ValueError(f"required v6 source is absent: {path}")
            actual = sha256(path)
            if actual != expected["sha256"]:
                raise ValueError(f"v6 source hash mismatch for {arm}/{name}")
            rows = len(pd.read_parquet(path, columns=["candidate_id"]))
            if rows != expected["rows"]:
                raise ValueError(f"v6 source row mismatch for {arm}/{name}: {rows} != {expected['rows']}")
            validated[arm][name] = {
                "path": expected["path"], "sha256": actual, "rows": rows,
            }
    return validated


def select_global_top_fraction(rows: pd.DataFrame, score: np.ndarray) -> pd.Series:
    count = int(np.ceil(len(rows) * TOP_FRACTION))
    # Preserve the frozen gate's global nlargest semantics, including its source-order
    # resolution of isotonic plateaus. The source is hash-verified and immutable.
    selected_index = pd.Series(score, index=rows.index).nlargest(count).index
    return rows.index.isin(selected_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    publication_root = args.output_root.with_name(args.output_root.name + ".partial")
    if args.output_root.exists():
        raise FileExistsError(f"immutable output already exists: {args.output_root}")
    if publication_root.exists():
        raise FileExistsError(f"incomplete atomic publication exists: {publication_root}")
    gate_manifest_path = args.gate_root / "manifest.json"
    manifest = json.loads(gate_manifest_path.read_text())
    validated_inputs = validate_gate_manifest(args.gate_root, manifest)

    result: dict[str, object] = {
        "schema": "historical_execution_ev_mapping_repair_v3",
        "research_status": "diagnostic_non_promotion_march_oof_mapping",
        "selection_contract": "Global top 10% of all April candidates per mapping variant; never per timestamp.",
        "mapping_reliability_contract": {
            "correlation": "Pearson correlation, side-local, between chronological March inner-OOF raw score and resolved exact execution_net_ev_12h.",
            "threshold": RELIABILITY_CORRELATION_THRESHOLD,
            "minimum_rows": RELIABILITY_MIN_ROWS,
            "interpretation": "Diagnostic admission heuristic only; not promotion-calibrated and not a statistical significance test.",
        },
        "turnover_contract": "Adjacent hour/day Jaccard turnover of selected assets (__symbol__), not timestamp-unique candidate IDs. Empty/empty buckets are omitted.",
        "arms": {},
    }
    publication_root.mkdir(parents=True)
    output_hashes: dict[str, str] = {}
    for arm in ARMS:
        inner = pd.read_parquet(args.gate_root / validated_inputs[arm]["march_inner_oof_scores"]["path"])
        outer = pd.read_parquet(args.gate_root / validated_inputs[arm]["april_outer_predictions"]["path"])
        inner["__ts__"] = pd.to_datetime(inner["__ts__"], utc=True)
        outer["__ts__"] = pd.to_datetime(outer["__ts__"], utc=True)
        raw = outer.raw_score.to_numpy(float)
        pooled_model = fit_iso(inner.score, inner.execution_net_ev_12h)
        pooled = pooled_model.predict(raw)
        ridge = Ridge(alpha=1.0).fit(
            inner.score.to_numpy().reshape(-1, 1), inner.execution_net_ev_12h
        ).predict(raw.reshape(-1, 1))
        local = np.empty(len(outer)); common = np.empty(len(outer)); hierarchical = np.empty(len(outer))
        side_models = {side: fit_iso(rows.score, rows.execution_net_ev_12h) for side, rows in inner.groupby("side_name")}
        side_predictions = {side: model.predict(inner.loc[inner.side_name.eq(side), "score"]) for side, model in side_models.items()}
        pooled_side = np.concatenate(list(side_predictions.values()))
        pooled_mean, pooled_std = float(pooled_side.mean()), float(pooled_side.std())
        reliability: dict[str, dict[str, object]] = {}
        for side, rows in outer.groupby("side_name", sort=True):
            mask = outer.side_name.eq(side).to_numpy()
            historic = inner.loc[inner.side_name.eq(side)]
            prediction = side_models[side].predict(rows.raw_score)
            historic_prediction = side_predictions[side]
            historic_scale = max(float(historic_prediction.std()), 1e-8)
            local[mask] = prediction
            common[mask] = (prediction - float(historic_prediction.mean())) / historic_scale * pooled_std + pooled_mean
            shrinkage = len(historic) / (len(historic) + 5_000.0)
            hierarchical[mask] = shrinkage * common[mask] + (1.0 - shrinkage) * pooled[mask]
            correlation = float(historic.score.corr(historic.execution_net_ev_12h, method="pearson"))
            reliability[str(side)] = {
                "inner_rows": len(historic), "support_pass": len(historic) >= RELIABILITY_MIN_ROWS,
                "inner_score_ev_pearson": correlation,
                "reliability_pass": bool(correlation >= RELIABILITY_CORRELATION_THRESHOLD),
            }
        variants = {
            "raw": raw, "pooled_isotonic": pooled, "pooled_ridge": ridge,
            "side_local": local, "side_common_unit": common, "hierarchical_shrinkage": hierarchical,
        }
        ledger = outer.copy()
        arm_result: dict[str, object] = {"reliability": reliability, "variants": {}}
        for name, score in variants.items():
            ledger[f"score_{name}"] = score
            ledger[f"selected_{name}"] = select_global_top_fraction(outer, score)
            arm_result["variants"][name] = metric(outer, ledger[f"selected_{name}"])
        admitted = outer.side_name.map(
            lambda side: reliability[str(side)]["support_pass"] and reliability[str(side)]["reliability_pass"]
        ).to_numpy(bool)
        ledger["admitted_hierarchical_reliability"] = admitted
        ledger["score_hierarchical_reliability_admission"] = hierarchical
        admission_rows = outer.loc[admitted].copy()
        admission_score = hierarchical[admitted]
        admission_selected = pd.Series(False, index=outer.index)
        if len(admission_rows):
            global_capacity = int(np.ceil(len(outer) * TOP_FRACTION))
            selected_index = pd.Series(admission_score, index=admission_rows.index).nlargest(
                min(global_capacity, len(admission_rows))
            ).index
            admission_selected.loc[selected_index] = True
        ledger["selected_hierarchical_reliability_admission"] = admission_selected.to_numpy(bool)
        arm_result["variants"]["hierarchical_reliability_admission"] = metric(
            outer.loc[admitted], admission_selected.loc[admitted]
        )
        arm_dir = publication_root / arm
        arm_dir.mkdir()
        ledger_path = arm_dir / "april_mapping_scores.parquet"
        atomic_parquet(ledger_path, ledger)
        output_hashes[str(ledger_path.relative_to(publication_root))] = sha256(ledger_path)
        result["arms"][arm] = arm_result
    report_path = publication_root / "report.json"
    atomic_json(report_path, result)
    output_hashes["report.json"] = sha256(report_path)
    output_manifest = {
        "schema": "historical_execution_ev_mapping_repair_manifest_v1",
        "status": "research_only_diagnostic",
        "source_gate_manifest": {"path": str(gate_manifest_path), "sha256": sha256(gate_manifest_path), "strict_identity_sha256": manifest.get("strict_identity_sha256")},
        "validated_inputs": validated_inputs,
        "mapping_reliability_contract": result["mapping_reliability_contract"],
        "output_sha256": output_hashes,
    }
    atomic_json(publication_root / "manifest.json", output_manifest)
    publication_root.replace(args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "arms": list(ARMS), "status": output_manifest["status"]}))


if __name__ == "__main__":
    main()
