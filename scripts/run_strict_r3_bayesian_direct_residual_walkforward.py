#!/usr/bin/env python3
"""Strict-OOF additive empirical-Bayes policy-residual challenger.

Unlike the historical B1 authority model, this estimates the quantity needed
by the canonical EV-map experiment directly: ``policy_net_bps -
raw_expected_bps``.  It uses only labels resolved before each held block,
the frozen non-posterior 45-field contract, and one frozen Geometry/K9 bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_forest_support_sizing import CANONICAL_N5_SPEC
from extreme_price_movements.trust_sizing_ablation import QuantileBins, residual_classes


SCHEMA = "strict_r3_bayesian_direct_residual_walkforward_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_unique(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "candidate_id" not in frame or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} must contain unique candidate_id")
    return frame


def _normalise_admission(frame: pd.DataFrame) -> pd.DataFrame:
    if "raw_expected_bps" not in frame:
        if "causal_21d_side_expected_net_bps" not in frame:
            raise ValueError("admission provenance lacks a causal expected-net column")
        frame = frame.rename(columns={"causal_21d_side_expected_net_bps": "raw_expected_bps"})
    if "mapped_ev_available" not in frame:
        frame["mapped_ev_available"] = np.isfinite(
            pd.to_numeric(frame["raw_expected_bps"], errors="coerce"),
        )
    return frame.loc[:, ["candidate_id", "raw_expected_bps", "mapped_ev_available"]].copy()


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = frame["__decision_ts__"].dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    quota = max(1, cap // len(month.unique()))
    positions: list[np.ndarray] = []
    for token in sorted(month.unique()):
        index = np.flatnonzero(month.eq(token).to_numpy())
        if len(index) > quota:
            index = np.sort(rng.choice(index, quota, replace=False))
        positions.append(index)
    selected = np.concatenate(positions)
    if len(selected) > cap:
        selected = np.sort(rng.choice(selected, cap, replace=False))
    return frame.iloc[selected].sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()


def _prepare(scored: pd.DataFrame, features: pd.DataFrame, admission: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "policy_label_available_ts", "policy_path_valid",
        "policy_net_bps", "final_score", "geometry_bundle_sha256",
    }
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"scored ledger lacks {missing}")
    missing = sorted(set(fields).difference(features.columns))
    if missing:
        raise ValueError(f"feature sidecar lacks {missing}")
    shared = [name for name in fields if name in scored.columns]
    if shared:
        comparison = scored.loc[:, ["candidate_id", *shared]].merge(
            features.loc[:, ["candidate_id", *shared]], on="candidate_id", how="left",
            validate="one_to_one", suffixes=("__ledger", "__sidecar"),
        )
        for name in shared:
            left = pd.to_numeric(comparison[f"{name}__ledger"], errors="coerce").to_numpy(float)
            right = pd.to_numeric(comparison[f"{name}__sidecar"], errors="coerce").to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError(f"ledger/feature sidecar mismatch for {name}")
    sidecar_only = [name for name in fields if name not in shared]
    frame = scored.merge(
        features.loc[:, ["candidate_id", *sidecar_only]], on="candidate_id", how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(admission, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(scored):
        raise ValueError("input sidecars do not cover the scored candidates exactly")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True)
    if frame["geometry_bundle_sha256"].dropna().astype(str).nunique() != 1:
        raise ValueError("direct-residual challenger requires exactly one frozen Geometry/K9 bundle")
    if any(name.startswith("k09__cluster_") for name in fields):
        raise ValueError("raw K9 membership coordinates are forbidden")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _select_fields(train: pd.DataFrame, fields: tuple[str, ...], top_fields: int) -> tuple[tuple[str, ...], QuantileBins]:
    bins = QuantileBins.fit(train, fields)
    codes = bins.transform(train)
    target = residual_classes(train["policy_net_bps"], train["raw_expected_bps"])
    # Conditional-on-EV binned MI.  It selects features that explain residual
    # conversion error, not raw outcome magnitude.
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    expected_bin = np.digitize(expected, np.unique(np.quantile(expected, [0.2, 0.4, 0.6, 0.8])), right=True)
    scores: list[tuple[float, str]] = []
    for idx, name in enumerate(fields):
        value = 0.0
        for state in range(5):
            mask = expected_bin == state
            if int(mask.sum()) < 100:
                continue
            joint = np.bincount(codes[mask, idx].astype(np.int64) * 6 + target[mask].astype(np.int64), minlength=96).reshape(16, 6)
            joint = joint + 0.5
            joint /= joint.sum()
            px = joint.sum(axis=1, keepdims=True)
            py = joint.sum(axis=0, keepdims=True)
            value += float(mask.mean()) * float(np.sum(joint * np.log(joint / (px * py))))
        scores.append((value, name))
    scores.sort(key=lambda item: (-item[0], item[1]))
    chosen = tuple(name for _, name in scores[:top_fields])
    return chosen, QuantileBins.fit(train, chosen)


def _predict(train: pd.DataFrame, held: pd.DataFrame, fields: tuple[str, ...], prior_strength: float) -> tuple[pd.DataFrame, dict[str, object]]:
    chosen, bins = _select_fields(train, fields, top_fields=min(12, len(fields)))
    train_codes, held_codes = bins.transform(train), bins.transform(held)
    residual = (
        pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
        - pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    )
    global_mean = float(np.mean(residual))
    effects: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for idx in range(len(chosen)):
        code_train, code_held = train_codes[:, idx], held_codes[:, idx]
        count = int(max(code_train.max(initial=0), code_held.max(initial=0))) + 1
        support = np.bincount(code_train, minlength=count).astype(float)
        total = np.bincount(code_train, weights=residual, minlength=count)
        mean = (total + prior_strength * global_mean) / (support + prior_strength)
        lookup = np.clip(code_held, 0, count - 1)
        effects.append(mean[lookup])
        supports.append(support[lookup])
    adjustment = np.mean(np.vstack(effects), axis=0)
    effective_support = np.median(np.vstack(supports), axis=0)
    expected = pd.to_numeric(held["raw_expected_bps"], errors="coerce").to_numpy(float) + adjustment
    predictive_sd = np.full(len(held), float(np.sqrt(np.mean(np.clip(residual - global_mean, -2000.0, 2000.0) ** 2))))
    output = pd.DataFrame({
        "posterior_expected_bps": expected.astype(np.float32),
        "posterior_residual_bps": adjustment.astype(np.float32),
        "posterior_predictive_sd": predictive_sd.astype(np.float32),
        "trust_effective_support": effective_support.astype(np.float32),
        "p_ev_positive": (1.0 / (1.0 + np.exp(-expected / np.maximum(predictive_sd, 25.0)))).astype(np.float32),
        "p_adverse_tail": (1.0 / (1.0 + np.exp((expected + 200.0) / np.maximum(predictive_sd, 25.0)))).astype(np.float32),
    })
    return output, {"selected_fields": list(chosen), "global_residual_mean_bps": global_mean, "predictive_sd_bps": float(predictive_sd[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--feature-sidecar", type=Path, required=True)
    parser.add_argument("--admission-provenance", type=Path, required=True)
    parser.add_argument("--conversion-block-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prior-strength", type=float, default=300.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if float(args.prior_strength) <= 0.0:
        raise ValueError("prior strength must be positive")
    contract = json.loads(args.contract.read_text())
    fields = tuple(map(str, contract["features"]))
    if len(fields) != 45 or len(set(fields)) != 45:
        raise ValueError("requires the frozen 45-field non-posterior contract")
    scored = _read_unique(args.scored_label_ledger, "scored label ledger")
    work = _prepare(
        scored, _read_unique(args.feature_sidecar, "feature sidecar"),
        _normalise_admission(_read_unique(args.admission_provenance, "admission provenance")), fields,
    )
    blocks = pd.read_parquet(args.conversion_block_audit).copy()
    required_blocks = {"cutoff", "held_end_exclusive", "geometry_bundle_sha256"}
    missing_blocks = sorted(required_blocks.difference(blocks.columns))
    if missing_blocks:
        raise ValueError(f"conversion block audit lacks {missing_blocks}")
    blocks["cutoff"] = pd.to_datetime(blocks["cutoff"], utc=True)
    blocks["held_end_exclusive"] = pd.to_datetime(blocks["held_end_exclusive"], utc=True)
    if blocks["geometry_bundle_sha256"].astype(str).nunique() != 1:
        raise ValueError("mixed geometry blocks are forbidden")
    output_parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for block_index, block in enumerate(blocks.sort_values("cutoff", kind="stable").itertuples(index=False)):
        cutoff, held_end = pd.Timestamp(block.cutoff), pd.Timestamp(block.held_end_exclusive)
        held = work.loc[work["__decision_ts__"].ge(cutoff) & work["__decision_ts__"].lt(held_end)].copy()
        start = cutoff - pd.DateOffset(months=CANONICAL_N5_SPEC.train_months)
        net = pd.to_numeric(work["policy_net_bps"], errors="coerce")
        expected = pd.to_numeric(work["raw_expected_bps"], errors="coerce")
        train_all = work.loc[
            work["__decision_ts__"].ge(start) & work["__decision_ts__"].lt(cutoff)
            & work["policy_label_available_ts"].lt(cutoff)
            & work["policy_path_valid"].fillna(False).astype(bool)
            & work["mapped_ev_available"].fillna(False).astype(bool)
            & np.isfinite(net) & np.isfinite(expected),
        ].copy()
        result = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        if len(train_all) < 1000:
            result["n5_available"] = False
            result["n5_unavailable_reason"] = "insufficient resolved prior support"
            result["n5_bundle_cutoff"] = cutoff
            result["trust_size_multiplier"] = np.float32(1.0)
            audits.append({"block_index": block_index, "cutoff": cutoff, "status": "warmup", "held_rows": len(held)})
        else:
            floor = float(np.quantile(train_all["final_score"], 1.0 - CANONICAL_N5_SPEC.top_fraction, method="higher"))
            train = _equal_month_sample(train_all.loc[train_all["final_score"].ge(floor)].copy(), CANONICAL_N5_SPEC.train_cap, seed=CANONICAL_N5_SPEC.seed)
            prediction, audit = _predict(train, held, fields, float(args.prior_strength))
            result = pd.concat([result.reset_index(drop=True), prediction], axis=1)
            result["n5_available"] = True
            result["n5_unavailable_reason"] = None
            result["n5_bundle_cutoff"] = cutoff
            result["trust_size_multiplier"] = np.float32(1.0)
            audits.append({"block_index": block_index, "cutoff": cutoff, "status": "complete", "held_rows": len(held), "train_rows": len(train), "training_score_floor": floor, **audit})
        output_parts.append(result)
        print(json.dumps({"event": "block_complete", **audits[-1]}, default=str), flush=True)
    output = pd.concat(output_parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(output) != len(work) or output["candidate_id"].duplicated().any():
        raise AssertionError("output changed candidate identity")
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "bayesian_direct_residual_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "bayesian_direct_residual_block_audit.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "target": "policy_net_bps - raw_expected_bps",
        "integration": "research only; no ranking, admission, or sizing change in this artifact",
        "prior_strength": float(args.prior_strength), "feature_count": len(fields), "rows": len(output),
        "geometry_refit_cadence": "never", "raw_k9_memberships_used": False,
        "source_hashes": {"scored_label_ledger": _sha(args.scored_label_ledger), "feature_sidecar": _sha(args.feature_sidecar), "admission_provenance": _sha(args.admission_provenance), "conversion_block_audit": _sha(args.conversion_block_audit), "contract": _sha(args.contract)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
