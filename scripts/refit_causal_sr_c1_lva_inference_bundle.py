#!/usr/bin/env python3
"""Fit one immutable C1-LVA S/R source bundle from prior-resolved history.

This producer creates source-head artifacts only.  It has no score, MC1,
portfolio, exchange, or order authority.  Historical OOF testing must still
fit an equivalent source bundle at each held boundary; this final refit is
for the next forward period only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_causal_sr_heads as source


DEFAULT_SOURCE = ROOT / "data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1"
DEFAULT_PROFILE = ROOT / "data_perp/artifacts/causal_profile_geometry_2025_train_2026_score_20260831_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_features(path: Path, features: tuple[str, ...]) -> str:
    payload = {"features": list(features)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile-state", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--cutoff", required=True, help="exclusive UTC resolved-label cutoff")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"C1-LVA bundle output must be immutable: {output}")
    cutoff = pd.Timestamp(args.cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    source_root = args.source.resolve()
    profile_root = args.profile_state.resolve()
    source_manifest = source_root / "run_manifest.json"
    profile_path = profile_root / "profile_hourly_states.parquet"
    if not source_manifest.is_file() or not profile_path.is_file():
        raise FileNotFoundError("C1-LVA source or profile state is unavailable")
    interactions = pd.read_parquet(source_root / "interaction_events.parquet")
    interactions["event_ts"] = pd.to_datetime(interactions.event_ts, utc=True, errors="raise")
    interactions["label_available_ts"] = pd.to_datetime(
        interactions.label_available_ts, utc=True, errors="raise"
    )
    profile_fields = source.PROFILE_CONTEXT_GROUPS["levels"]
    states = pd.read_parquet(profile_path)
    interactions = source._merge_profile_context(
        interactions, states, timestamp="event_ts", fields=profile_fields,
    )
    conditional = (*source.CONDITIONAL_FEATURES, *profile_fields, source.PROFILE_CONTEXT_AVAILABLE)
    train = interactions.loc[
        interactions.event_ts.lt(cutoff) & interactions.label_available_ts.lt(cutoff)
    ].copy()
    if len(train) < 2_000:
        raise RuntimeError(f"insufficient strict prior-resolved C1-LVA rows: {len(train)}")
    missing = sorted(set(conditional).difference(train.columns))
    if missing:
        raise ValueError(f"C1-LVA training input contract missing {missing}")
    models = source._fit_models(train, conditional)
    output.mkdir(parents=True, exist_ok=False)
    names = ("prior_model.joblib", "conditional_model.joblib", "break_model.joblib", "magnitude_model.joblib")
    roles = ("prior_model", "conditional_model", "break_model", "magnitude_model")
    for model, name in zip(models, names, strict=True):
        joblib.dump(model, output / name)
    files = {
        role: {"name": name, "sha256": _sha256(output / name)}
        for role, name in zip(roles, names, strict=True)
    }
    features_sha = _write_features(output / "prior_feature_order.json", source.PRIOR_FEATURES)
    conditional_sha = _write_features(output / "conditional_feature_order.json", conditional)
    manifest = {
        "schema": "causal-sr-c1-lva-inference-bundle-v1",
        "status": "SEALED_NO_ORDER_C1_LVA_SOURCE_BUNDLE",
        "order_submission": False,
        "side": "long",
        "cutoff": cutoff.isoformat(),
        "training": {
            "eligible_rows_before_bounded_thinning": int(len(train)),
            "fit_cap": int(source.MAX_TRAIN_ROWS),
            "effective_fit_rows": int(min(len(train), source.MAX_TRAIN_ROWS)),
            "labels_resolved_before_cutoff": True,
            "source_model_contract": "C1-LVA levels/value-area, L1 d3/l7 source heads",
        },
        "features": {
            "prior": list(source.PRIOR_FEATURES),
            "conditional": list(conditional),
            "profile": list(profile_fields),
            "prior_feature_order_sha256": features_sha,
            "conditional_feature_order_sha256": conditional_sha,
        },
        "sources": {
            "sr_engine_manifest": str(source_manifest.relative_to(ROOT)),
            "sr_engine_manifest_sha256": _sha256(source_manifest),
            "profile_state": str(profile_path.relative_to(ROOT)),
            "profile_state_sha256": _sha256(profile_path),
            "source_head_runtime": "scripts/run_causal_sr_heads.py",
            "source_head_runtime_sha256": _sha256(ROOT / "scripts/run_causal_sr_heads.py"),
        },
        "files": files,
        "causality": {
            "inputs": "Causal S/R interactions and profile states are created from completed prior bars; profile merge is as-of only.",
            "labels": "Only interaction labels with event_ts and label_available_ts strictly before cutoff fit source heads.",
            "inference": "This source bundle has no candidate, outcome, MC1, portfolio, exchange, or order authority.",
        },
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
