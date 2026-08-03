#!/usr/bin/env python3
"""Materialise the raw-feature exact-H12 panel needed for a true base refit.

The existing long target ablation could only adapt a frozen base score.  This
runner instead reuses the staged, exact-1m historical candidate lineage to
emit the raw decision-time feature matrix with exact H12 current-policy
labels.  It writes no model and makes no performance claim.

The retained calendar is 2023-04 through 2024-11: twelve months can train a
base refit (2023-04..2024-03), leaving eight months for frozen base OOS
predictions and a subsequent 4+4 month residual split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reconstruct_2022_2024_base_residual_stack_oof import (
    DEFAULT_FULL_2024_LABELS,
    DEFAULT_FULL_2024_STAGE,
    DEFAULT_PF_LABELS,
    DEFAULT_PF_STAGE,
    IDENTITY,
    load_pf_population,
)


DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/long_exact_h12_raw_base_panel_20260730_v2"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2024-12-01T00:00:00Z")
LABEL_COLUMNS = (
    "candidate_id", "__decision_ts__", "__label_end_ts__", "__label_available_at__",
    "execution_label_end_utc", "execution_label_available_at",
    "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h",
    "__opportunity_occurred_12h__", "__favorable_payoff_return_12h__",
    "__adverse_competing_risk_12h__", "__timeout_outcome_12h__",
    "__exit_conversion_loss_return_12h__", "__peak_mfe_atr_12h__",
    "__time_to_first_meaningful_mfe_hours_12h__",
    "__mae_before_meaningful_mfe_atr_12h__",
    "__bars_before_price_stops_decreasing_12h__",
    "__future_slope_atr_per_hour_12h__",
)
DISALLOWED_RAW_PREFIXES = (
    "base_attr_", "threshold_basis_", "dae_", "gmm_", "__regime_source_",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_labels(paths: tuple[Path, ...]) -> pd.DataFrame:
    parts = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        missing = sorted(set(LABEL_COLUMNS).difference(available))
        if missing:
            raise ValueError(f"label source {path} misses {missing}")
        part = pd.read_parquet(path, columns=list(LABEL_COLUMNS))
        if part.candidate_id.duplicated().any():
            raise ValueError(f"duplicate candidate_id in {path}")
        parts.append(part)
    labels = pd.concat(parts, ignore_index=True)
    if labels.candidate_id.duplicated().any():
        raise ValueError("label sources overlap")
    for column in (
        "__decision_ts__", "__label_end_ts__", "__label_available_at__",
        "execution_label_end_utc", "execution_label_available_at",
    ):
        labels[column] = pd.to_datetime(labels[column], utc=True, errors="raise")
    return labels


def validate_panel(frame: pd.DataFrame, raw_features: list[str]) -> None:
    required = {*IDENTITY, "frozen_base_score", *LABEL_COLUMNS, *raw_features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"panel misses required fields: {missing}")
    if frame.duplicated(list(IDENTITY)).any() or frame.candidate_id.duplicated().any():
        raise ValueError("candidate identity is not unique")
    if not frame.side_name.isin(("long", "short")).all():
        raise ValueError("unexpected side")
    if not frame["__ts__"].ge(START).all() or not frame["__ts__"].lt(END).all():
        raise ValueError("output calendar exceeds requested 20-month panel")
    if not frame["execution_label_end_utc"].ge(frame["__decision_ts__"]).all():
        raise ValueError("H12 endpoint predates decision")
    if not frame["execution_label_available_at"].ge(frame["execution_label_end_utc"]).all():
        raise ValueError("outcome availability predates endpoint")
    numeric = frame.loc[:, raw_features].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("raw feature matrix contains infinity")
    prohibited = [name for name in raw_features if name.lower().startswith(DISALLOWED_RAW_PREFIXES)]
    if prohibited:
        raise ValueError(f"raw feature contract contains unsupported frozen geometry/selection fields: {prohibited}")
    months = frame["__ts__"].dt.strftime("%Y-%m").drop_duplicates().tolist()
    expected = pd.period_range("2023-04", "2024-11", freq="M").astype(str).tolist()
    if months != expected:
        raise ValueError(f"calendar month coverage mismatch: {months} != {expected}")


def run(*, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    stages = (Path(DEFAULT_PF_STAGE), Path(DEFAULT_FULL_2024_STAGE))
    label_paths = (Path(DEFAULT_PF_LABELS), Path(DEFAULT_FULL_2024_LABELS))
    source, raw_features, lineage = load_pf_population(stages, label_paths)
    original_raw_features = list(raw_features)
    raw_features = [
        name for name in raw_features
        if not name.lower().startswith(DISALLOWED_RAW_PREFIXES)
    ]
    if len(raw_features) < 100:
        raise ValueError("strict observable raw feature intersection is too small")
    source = source.drop(columns=sorted(set(original_raw_features).difference(raw_features)))
    labels = _read_labels(label_paths)
    source = source.rename(columns={"base_score": "frozen_base_score"})
    duplicated_label_values = sorted((set(source.columns) & set(LABEL_COLUMNS)) - {"candidate_id"})
    source = source.drop(columns=duplicated_label_values)
    frame = source.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(source):
        raise ValueError("label join lost staged candidates")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame = frame.loc[frame["__ts__"].ge(START) & frame["__ts__"].lt(END)].copy()
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    validate_panel(frame, raw_features)

    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        panel_path = stage / "raw_base_panel.parquet"
        feature_path = stage / "raw_feature_contract.json"
        coverage_path = stage / "coverage_by_month_side.csv"
        frame.to_parquet(panel_path, index=False, compression="zstd")
        _write_json(feature_path, {
            "schema": "long_exact_h12_raw_base_feature_contract_v1",
            "raw_feature_columns": raw_features,
            "raw_feature_count": len(raw_features),
            "frozen_base_score": "diagnostic/context only; not part of the raw-base feature contract",
            "forbidden": "all H12 outcomes, path labels, maps, OOF scores and action fields; frozen base-attribution, threshold-basis, DAE, GMM and regime-source fields are excluded from raw_feature_columns",
        })
        coverage = frame.assign(month=frame["__ts__"].dt.strftime("%Y-%m")).groupby(["month", "side_name"], as_index=False).agg(
            rows=("candidate_id", "size"),
            label_end_max=("execution_label_end_utc", "max"),
            label_available_max=("execution_label_available_at", "max"),
            finite_raw_fraction=(raw_features[0], lambda value: float(pd.to_numeric(value, errors="coerce").notna().mean())),
        )
        coverage.to_csv(coverage_path, index=False)
        outputs = (panel_path, feature_path, coverage_path)
        manifest = {
            "schema": "long_exact_h12_raw_base_panel_v1",
            "status": "MATERIALIZED_CANDIDATE_CONDITIONED_COUNTERFACTUAL_RESEARCH_INPUT_NO_MODEL_NO_PROMOTION",
            "calendar": {"start": START.isoformat(), "end_exclusive": END.isoformat(), "intended_split": "12-month base train; 8-month base OOS; first 4 OOS months meta fit; final 4 meta OOS"},
            "label_contract": "exact-1m current-policy H12 gross/cost/net and supporting path components; outcome availability is materialized",
            "raw_feature_contract": "strict intersection of numeric decision-time source-shard fields, with outcome/path/score/action tokens and frozen base-attribution/threshold/DAE/GMM/regime-source fields excluded",
            "limitations": [
                "Candidate-conditioned: labels exist only for old selected-top30/monitor candidates, not the full base universe.",
                "Historical current-spread counterfactual: source manifests set execution_parity_claim=false; this is not factual historical execution.",
                "Historical L2 and bit-exact pre-2025 path-geometry parity are unavailable.",
            ],
            "sources": {str(path): _sha256(path) for path in (*stages, *label_paths)},
            "lineage": lineage,
            "rows": int(len(frame)), "raw_feature_count": int(len(raw_features)),
            "outputs": {path.name: _sha256(path) for path in outputs},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
