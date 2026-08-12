#!/usr/bin/env python3
"""Join causal OOF regime context into the two-year leaf-meta input.

Only decision-time numeric regime/transition/geometry outputs are copied.  The
sidecar's identity and provenance remain in its manifest; fold identifiers,
timestamps, state ids and outcome-like fields are deliberately not meta
features.  The downstream runner treats the added names as MDA candidates,
not unconditional raw meta inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity


DEFAULT_INPUT = ROOT / "data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v2/input.parquet"
DEFAULT_CONTEXT = ROOT / "data_perp/artifacts/correctness_leaf_regime_two_year_oof_market_context_20260803_v1/candidate_oof_market_regimes.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v3"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_safe_context_feature(name: str, dtype: object) -> bool:
    lower = str(name).lower()
    if name in IDENTITY_COLUMNS or not pd.api.types.is_numeric_dtype(dtype):
        return False
    if any(token in lower for token in (
        "target", "label", "outcome", "future", "realized", "realised",
        "pnl", "net_ev", "gross_ev", "mfe", "mae", "barrier", "timeout",
        "entry", "exit", "policy", "rank",
    )):
        return False
    # IDs and fitted-K counters are reporting provenance, not continuous
    # causal geometry.  Probabilities/entropy/margins/age/switch and distance
    # invariants remain eligible candidates.
    if lower.endswith("_id") or lower.endswith("__state_count"):
        return False
    return lower.startswith((
        "market_regime__", "geometry_regime__", "continuous_regime__",
        "regime_p_", "regime_state_", "transition_state_", "transition_",
        "regime_entropy", "regime_top2_", "state_age", "state_switch",
    ))


def materialize(*, input_path: Path = DEFAULT_INPUT, context_path: Path = DEFAULT_CONTEXT, out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(out)
    base = validate_candidate_identity(pd.read_parquet(input_path))
    context = validate_candidate_identity(pd.read_parquet(context_path))
    for frame in (base, context):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if context.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("context sidecar is not unique by candidate identity")
    feature_names = [name for name in context if _is_safe_context_feature(name, context[name].dtype)]
    if not feature_names:
        raise ValueError("no safe causal context features found")
    overlap = sorted(set(feature_names).intersection(base.columns))
    if overlap:
        raise ValueError(f"input already contains context fields: {overlap[:10]}")
    joined = base.merge(
        context.loc[:, [*IDENTITY_COLUMNS, *feature_names]], on=list(IDENTITY_COLUMNS),
        how="left", validate="one_to_one", sort=False,
    )
    if len(joined) != len(base) or joined["candidate_id"].tolist() != base["candidate_id"].tolist():
        raise RuntimeError("context join changed the candidate population or ordering")
    coverage = pd.DataFrame({
        "feature": joined.columns.astype(str),
        "coverage": [float(joined[name].notna().mean()) for name in joined],
        "nonconstant": [bool(joined[name].nunique(dropna=True) > 1) for name in joined],
    })
    coverage["usable_90pct_nonconstant"] = coverage.coverage.ge(.90) & coverage.nonconstant
    out.mkdir(parents=True)
    joined.to_parquet(out / "input.parquet", index=False, compression="zstd")
    coverage.to_parquet(out / "feature_availability.parquet", index=False, compression="zstd")
    contract = {
        "status": "COMPLETED_CAUSAL_CONTEXT_AUGMENT",
        "base_input": str(input_path.resolve()), "base_input_sha256": _sha(input_path),
        "context_sidecar": str(context_path.resolve()), "context_sidecar_sha256": _sha(context_path),
        "candidate_rows": int(len(joined)), "context_feature_count": int(len(feature_names)),
        "context_features": feature_names,
        "join": "exact candidate identity; sidecar generated from strictly-prior hourly fits and backward-only joins",
        "admission": "context fields are candidate-only and require nested side-local MDA before final meta use",
    }
    (out / "context_feature_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        **contract,
        "outputs": {name: _sha(out / name) for name in ("input.parquet", "feature_availability.parquet", "context_feature_contract.json")},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _args()
    print(materialize(input_path=args.input, context_path=args.context, out=args.out))
