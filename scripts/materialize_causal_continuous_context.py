#!/usr/bin/env python3
"""Materialize strict-prequential continuous market context without clusters.

This is intentionally separate from ``materialize_oof_market_regime_systems``:
the latter remains the diagnostic-only latent-state/reporting path.  This
script reads nine predeclared observable columns, computes relative context in
one vectorised pass, and optionally joins it backwards to an exact candidate
population.  It never fits a GMM, learns a state, or reads an outcome.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_market_regime_systems import (  # noqa: E402
    CONTINUOUS_CONTEXT_FEATURE_KEYS,
    CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
    CausalContinuousContextConfig,
    CausalRelationshipBreakConfig,
    RELATIONSHIP_BREAK_FEATURE_KEYS,
    build_causal_continuous_context_features,
    build_causal_relationship_break_features,
    continuous_context_feature_names,
)
from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity  # noqa: E402


SCHEMA = "causal_continuous_context_sidecar_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_continuous_context_2023q3_2024_20260803_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def materialize(
    *,
    panel_path: Path = DEFAULT_PANEL,
    output_dir: Path = DEFAULT_OUTPUT,
    evaluation_start: str,
    evaluation_end: str | None = None,
    candidate_path: Path | None = None,
    max_lag_hours: int = 2,
) -> Path:
    """Build an exact candidate sidecar using only decision-time observables."""

    panel_path, output_dir = Path(panel_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    start = pd.to_datetime(evaluation_start, utc=True, errors="raise")
    end = pd.to_datetime(evaluation_end, utc=True, errors="raise") if evaluation_end else None
    schema = set(pq.ParquetFile(panel_path).schema.names)
    missing = [name for name in CONTINUOUS_CONTEXT_SOURCE_CONTRACT.values() if name not in schema]
    if missing:
        raise KeyError(f"continuous source store lacks required observable fields: {missing}")
    hourly = pd.read_parquet(panel_path, columns=["source_utc", *CONTINUOUS_CONTEXT_SOURCE_CONTRACT.values()])
    hourly["source_utc"] = pd.to_datetime(hourly["source_utc"], utc=True, errors="raise")
    hourly = hourly.sort_values("source_utc", kind="stable").drop_duplicates("source_utc", keep="last").reset_index(drop=True)
    # Keep all available history before the requested population.  The rolling
    # implementation itself is left-closed; this is not a training window.
    feature_frame = build_causal_continuous_context_features(
        hourly,
        CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
        config=CausalContinuousContextConfig(timestamp_col="source_utc"),
    )
    relationship_breaks = build_causal_relationship_break_features(
        hourly,
        CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
        config=CausalRelationshipBreakConfig(timestamp_col="source_utc"),
    )
    timeline = pd.concat([hourly.loc[:, ["source_utc"]], feature_frame, relationship_breaks], axis=1)
    timeline = timeline.loc[timeline["source_utc"].ge(start)].copy()
    if end is not None:
        timeline = timeline.loc[timeline["source_utc"].lt(end)].copy()
    if timeline.empty:
        raise ValueError("no continuous context timeline rows in requested period")
    timeline["continuous_context_available_utc"] = timeline["source_utc"]
    timeline["continuous_context_history_start_utc"] = hourly["source_utc"].min()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        timeline.to_parquet(temporary / "hourly_causal_continuous_context.parquet", index=False, compression="zstd")
        candidate_rows = 0
        feature_keys = [*CONTINUOUS_CONTEXT_FEATURE_KEYS, *RELATIONSHIP_BREAK_FEATURE_KEYS]
        coverage: dict[str, float] = {name: float(timeline[name].notna().mean()) for name in feature_keys}
        if candidate_path is not None:
            # Candidate source panels can contain hundreds of non-identity
            # fields and, for a research replay, resolved outcome columns.
            # They are neither needed nor allowed here. Projection at read
            # time keeps the causal context join cheap and makes the boundary
            # explicit: the context sidecar sees identity/timestamp only.
            candidates = validate_candidate_identity(
                pd.read_parquet(candidate_path, columns=list(IDENTITY_COLUMNS))
            ).loc[:, list(IDENTITY_COLUMNS)].copy()
            candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
            candidates = candidates.loc[candidates["__ts__"].ge(start)].sort_values("__ts__", kind="stable")
            if end is not None:
                candidates = candidates.loc[candidates["__ts__"].lt(end)]
            candidate = pd.merge_asof(
                candidates,
                timeline.sort_values("source_utc", kind="stable"),
                left_on="__ts__", right_on="source_utc", direction="backward",
                tolerance=pd.Timedelta(hours=int(max_lag_hours)),
            )
            if (candidate["source_utc"] > candidate["__ts__"]).fillna(False).any():
                raise RuntimeError("continuous context candidate join looked ahead")
            candidate.to_parquet(temporary / "candidate_causal_continuous_context.parquet", index=False, compression="zstd")
            candidate_rows = int(len(candidate))
            coverage = {name: float(candidate[name].notna().mean()) for name in feature_keys}
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_STRICT_PREQUENTIAL_CONTINUOUS_CONTEXT",
            "inputs": {
                "hourly_multiview_panel": {"path": str(panel_path.resolve()), "sha256": _sha256(panel_path)},
                "candidate_path": str(Path(candidate_path).resolve()) if candidate_path else None,
            },
            "contract": {
                "model_inputs": "nine named raw observable dimensions only; no GMM/state/membership/cluster output",
                "features": feature_keys,
                "semantics": "left-closed rolling rank/z 90d/180d, exact 4h/24h change, left-closed 30d-median distance",
                "max_lag_hours": int(max_lag_hours),
                "candidate_join": "backward as-of only",
                "no_outcomes_or_candidate_scores": True,
                "relationship_breaks": "four prior-only rolling OLS residual pairs, signed and absolute at 30d/90d",
            },
            "coverage": {
                "hourly_rows": int(len(timeline)), "candidate_rows": candidate_rows,
                "evaluation_start_utc": start, "evaluation_end_exclusive_utc": end,
                "feature_coverage": coverage,
            },
            "outputs": {},
        }
        for path in temporary.iterdir():
            if path.is_file():
                manifest["outputs"][path.name] = _sha256(path)
        (temporary / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--max-lag-hours", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _args()
    print(materialize(
        panel_path=args.panel, output_dir=args.output_dir, evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end, candidate_path=args.candidates, max_lag_hours=args.max_lag_hours,
    ))
