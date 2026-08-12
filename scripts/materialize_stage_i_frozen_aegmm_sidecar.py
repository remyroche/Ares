#!/usr/bin/env python3
"""Attach a frozen AE/GMM projection to the immutable Stage-I selector population.

This is deliberately a representation bridge, rather than a refit.  The
frozen state may have been fitted on a later pre-existing artifact under the
approved research exception, but every resulting row is still computed only
from its selector-time feature row.  The output carries the exact selector
identity and a source-overlap audit so a downstream meta selection cannot
mistake a configured AE/GMM family for an evaluated one.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.stage_i_target_adapter import file_sha256


IDENTITY = ("candidate_id", "__ts__", "__symbol__")
SCHEMA = "stage_i_frozen_aegmm_sidecar_v1"


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    """Hash arbitrary frozen-state artifacts; parquet-only helpers are insufficient."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(
    *, selector_dir: Path, state_path: Path, output_path: Path,
    min_source_overlap: float = 0.50,
    output_prefix: str = "meta_lgbm_",
) -> dict[str, object]:
    """Write a fully identity-bound sidecar and its compact immutable manifest."""
    if not 0.0 < float(min_source_overlap) <= 1.0:
        raise ValueError("min source overlap must lie in (0,1]")
    features_path = selector_dir / "selector_features.parquet"
    manifest_path = selector_dir / "manifest.json"
    if not features_path.is_file() or not manifest_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("selector features/manifest and frozen AE/GMM state are required")
    selector_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = selector_manifest.get("artifact_integrity", {})
    if integrity.get("selector_features_sha256") != file_sha256(features_path):
        raise ValueError("selector feature artifact hash drift")
    raw = pd.read_parquet(features_path)
    selector_had_side = "side" in raw.columns
    if missing := set(IDENTITY).difference(raw.columns):
        raise ValueError(f"selector feature panel lacks identity: {sorted(missing)}")
    if raw.duplicated(list(IDENTITY)).any():
        raise ValueError("selector feature panel has duplicate immutable identities")
    state = pd.read_pickle(state_path)
    if not isinstance(state, dict) or not bool(state.get("enabled", False)):
        raise ValueError("frozen AE/GMM state is not enabled")
    # A frozen historical state can have been fitted with a sequential
    # smoothing contract.  Selector rows are an interleaved candidate panel,
    # not one causal time series, so force the projection itself to be row
    # independent.  This preserves the encoder/GMM geometry but prevents a
    # neighbouring row from changing an inference feature.
    projection_state = dict(state)
    source_temporal_contract = str(state.get("temporal_feature_contract") or "unspecified")
    projection_state["temporal_feature_contract"] = "row_independent_v1"
    inputs = tuple(map(str, projection_state.get("feature_columns", ())))
    if not inputs:
        raise ValueError("frozen AE/GMM state lacks ordered input fields")
    present = tuple(name for name in inputs if name in raw.columns)
    # Side was not a raw feature in this selector but is a causal candidate
    # attribute.  It is reconstructed only from the selector ledger when the
    # state explicitly requested it; no future state is introduced.
    ledger_path = selector_dir / "selector_ledger.parquet"
    if "side" in inputs and ledger_path.is_file():
        ledger = pd.read_parquet(ledger_path, columns=[*IDENTITY, "side_name"])
        keys = pd.MultiIndex.from_frame(ledger.loc[:, list(IDENTITY)])
        target = pd.MultiIndex.from_frame(raw.loc[:, list(IDENTITY)])
        pos = keys.get_indexer(target)
        if (pos < 0).any() or len(np.unique(pos)) != len(pos):
            raise ValueError("selector ledger identity drift while reconstructing causal side")
        raw = raw.copy()
        raw["side"] = ledger.iloc[pos].side_name.astype(str).str.lower().map({"long": 1.0, "short": -1.0})
        present = tuple(dict.fromkeys((*present, "side")))
    overlap = float(len(present) / len(inputs))
    if overlap < float(min_source_overlap):
        raise ValueError(
            f"frozen AE/GMM source overlap {overlap:.4f} is below minimum {float(min_source_overlap):.4f}"
        )
    source = raw.reindex(columns=list(inputs))
    generated = transform_ae_gmm_features(source, projection_state, index=raw.index, prefix=output_prefix)
    generated = generated.replace([np.inf, -np.inf], np.nan)
    varying = [
        name for name in generated.columns
        if np.isfinite(pd.to_numeric(generated[name], errors="coerce")).all()
        and float(np.nanstd(pd.to_numeric(generated[name], errors="coerce").to_numpy(float))) > 1e-8
    ]
    if not varying:
        raise ValueError("frozen AE/GMM projection has no materially varying output fields")
    sidecar = pd.concat([raw.loc[:, list(IDENTITY)].reset_index(drop=True), generated.loc[:, varying].reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_parquet(output_path, index=False, compression="zstd")
    coverage = {name: float(sidecar[name].notna().mean()) for name in varying}
    manifest = {
        "schema": SCHEMA,
        "selector_dir": str(selector_dir.resolve()),
        "selector_feature_sha256": file_sha256(features_path),
        "selector_manifest_sha256": file_sha256(manifest_path),
        "frozen_state_path": str(state_path.resolve()),
        "frozen_state_sha256": _file_hash(state_path),
        "representation_exception": "pre_existing_frozen_state_may_have_later_fit_rows_user_approved",
        "source_temporal_contract": source_temporal_contract,
        "projection_temporal_contract": "row_independent_v1",
        "projection_temporal_override": "disable sequential smoothing/deltas on interleaved selector rows",
        "identity_contract": list(IDENTITY),
        "state_input_count": len(inputs),
        "state_input_present_count": len(present),
        "state_input_overlap": overlap,
        "state_input_missing": [name for name in inputs if name not in present],
        "side_reconstructed_from_selector_ledger": bool("side" in present and not selector_had_side),
        "output_prefix": output_prefix,
        "output_fields": varying,
        "output_coverage": coverage,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
    }
    manifest["request_sha256"] = _sha({key: value for key, value in manifest.items() if key != "output_sha256"})
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--frozen-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-source-overlap", type=float, default=0.50)
    parser.add_argument("--output-prefix", default="meta_lgbm_")
    args = parser.parse_args()
    print(json.dumps(materialize(
        selector_dir=args.selector_dir, state_path=args.frozen_state, output_path=args.output,
        min_source_overlap=float(args.min_source_overlap), output_prefix=str(args.output_prefix),
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
