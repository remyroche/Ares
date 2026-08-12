#!/usr/bin/env python3
"""Run the chronological continuous-context transport-selection audit.

Example:
  python scripts/audit_regime_transport_selection.py \
    --input candidates.parquet --features-file continuous_context.txt \
    --reference-feature score_residual_expected_ev --era-column evaluation_month \
    --output-dir data_perp/artifacts/regime_transport_selection_v1

The feature file contains one continuous candidate field per line.  Membership,
posterior, state-ID and cluster fields are rejected before any model is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transport_selection import (  # noqa: E402
    TransportAuditConfig,
    audit_continuous_context_transport,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_file(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names:
        raise ValueError("features file has no candidate fields")
    return names


def run(
    *,
    input_path: Path,
    features_file: Path,
    output_dir: Path,
    context_sidecar: Path | None = None,
    reference_features: Sequence[str] = (),
    timestamp_column: str = "__ts__",
    era_column: str = "era",
    target_column: str = "execution_net_ev_12h",
    candidate_id_column: str = "candidate_id",
    derive_halfyear_era: bool = False,
    threshold_bps: float = 0.0,
    embargo_hours: float = 12.0,
    top_fraction: float = 0.10,
) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")
    candidates = _feature_file(features_file)
    columns = list(dict.fromkeys([timestamp_column, era_column, target_column, candidate_id_column, *reference_features, *candidates]))
    base_columns = list(dict.fromkeys([
        *([candidate_id_column, timestamp_column, "__symbol__", "side_name"] if context_sidecar is not None else []),
        *(name for name in columns
        if name != era_column and (context_sidecar is None or name not in candidates)
        ),
    ]))
    frame = pd.read_parquet(input_path, columns=base_columns)
    if context_sidecar is not None:
        identity = [candidate_id_column, timestamp_column, "__symbol__", "side_name"]
        sidecar = pd.read_parquet(context_sidecar, columns=[*identity, *candidates])
        frame = frame.merge(sidecar, on=identity, how="inner", validate="one_to_one")
    elif set(candidates).difference(frame.columns):
        # Candidate fields normally reside in the input itself.  This explicit
        # guard makes the optional exact-keyed sidecar path unambiguous.
        missing = sorted(set(candidates).difference(frame.columns))
        raise KeyError(f"input lacks candidate feature fields: {missing[:8]}")
    if era_column not in frame:
        if not derive_halfyear_era:
            raise KeyError(f"input lacks requested era column: {era_column}")
        timestamp = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
        frame[era_column] = timestamp.dt.year.astype(str) + "H" + np.where(timestamp.dt.month.le(6), "1", "2")
    result = audit_continuous_context_transport(
        frame,
        candidate_features=candidates,
        reference_features=list(reference_features),
        config=TransportAuditConfig(
            timestamp_column=timestamp_column, era_column=era_column,
            target_column=target_column, candidate_id_column=candidate_id_column,
            threshold_bps=threshold_bps, embargo_hours=embargo_hours,
            top_fraction=top_fraction,
        ),
    )
    output.mkdir(parents=True)
    result.feature_audit.to_parquet(output / "continuous_context_transport_audit.parquet", index=False)
    result.split_mda.to_parquet(output / "continuous_context_split_mda.parquet", index=False)
    result.era_proxy.to_parquet(output / "continuous_context_era_proxy.parquet", index=False)
    manifest = {
        **result.manifest,
        "inputs": {
            "input": {"path": str(input_path.resolve()), "sha256": _sha(input_path)},
            "context_sidecar": ({"path": str(context_sidecar.resolve()), "sha256": _sha(context_sidecar)} if context_sidecar is not None else None),
            "features_file": {"path": str(features_file.resolve()), "sha256": _sha(features_file)},
            "reference_features": list(reference_features),
        },
        "outputs": ["continuous_context_transport_audit.parquet", "continuous_context_split_mda.parquet", "continuous_context_era_proxy.parquet"],
    }
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--context-sidecar", type=Path, help="exact candidate-keyed context fields to join to --input")
    parser.add_argument("--features-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-feature", action="append", default=[])
    parser.add_argument("--timestamp-column", default="__ts__")
    parser.add_argument("--era-column", default="era")
    parser.add_argument("--derive-halfyear-era", action="store_true", help="derive YYYYH1/YYYYH2 from the decision timestamp when the input has no era column")
    parser.add_argument("--target-column", default="execution_net_ev_12h")
    parser.add_argument("--candidate-id-column", default="candidate_id")
    parser.add_argument("--threshold-bps", type=float, default=0.0)
    parser.add_argument("--embargo-hours", type=float, default=12.0)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    values = _args()
    print(run(
        input_path=values.input, features_file=values.features_file, output_dir=values.output_dir, context_sidecar=values.context_sidecar,
        reference_features=values.reference_feature, timestamp_column=values.timestamp_column,
        era_column=values.era_column, target_column=values.target_column,
        candidate_id_column=values.candidate_id_column, threshold_bps=values.threshold_bps,
        embargo_hours=values.embargo_hours, top_fraction=values.top_fraction,
        derive_halfyear_era=values.derive_halfyear_era,
    ))
