#!/usr/bin/env python3
"""Strict-OOF source-aligned empirical-Bayes finalist sizing challenger.

This deliberately mirrors the canonical v3 LDF runner's candidate identity,
three-month resolved training window, top-30% score gate, and frozen geometry
constraint.  It changes only the post-admission size multiplier.  In
particular it never reranks candidates or alters the precomputed causal EV
admission map.
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

from extreme_price_movements.n5_forest_support_sizing import CANONICAL_N5_SPEC  # noqa: E402
from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    TrustModelSpec,
    causal_size_multiplier,
    catalogue,
    fit_trust_model,
    sizing_quality,
)


SCHEMA = "strict_r3_bayesian_frozen_walkforward_v2"
FINALIST_SPECS = {
    spec.name: spec for spec in catalogue()["bayesian"]
    if spec.name in {
        "B1_raw_singleton_l100_mean",
        "B4_stable_rankfp_l125_predictive",
        "B5_stable_ranklossfp_l125_predictive",
    }
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_unique(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "candidate_id" not in frame or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} must contain unique candidate_id")
    return frame


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = pd.to_datetime(frame["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    quota = max(1, int(cap) // len(month.unique()))
    chosen: list[np.ndarray] = []
    for token in sorted(month.unique()):
        positions = np.flatnonzero(month.eq(token).to_numpy())
        if len(positions) > quota:
            positions = np.sort(rng.choice(positions, quota, replace=False))
        chosen.append(positions)
    selected = np.concatenate(chosen)
    if len(selected) > cap:
        selected = np.sort(rng.choice(selected, int(cap), replace=False))
    return frame.iloc[selected].sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()


def _prepare(
    scored_labels: pd.DataFrame,
    features: pd.DataFrame,
    admission: pd.DataFrame,
    *,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    labels = {
        "candidate_id", "__decision_ts__", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", "final_score", "geometry_bundle_sha256",
    }
    missing = sorted(labels.difference(scored_labels.columns))
    if missing:
        raise ValueError(f"scored label ledger lacks: {missing}")
    missing = sorted(set(fields).difference(features.columns))
    if missing:
        raise ValueError(f"Bayesian feature sidecar lacks: {missing}")
    required_admission = {"candidate_id", "raw_expected_bps", "mapped_ev_available"}
    missing = sorted(required_admission.difference(admission.columns))
    if missing:
        raise ValueError(f"causal admission provenance lacks: {missing}")
    shared = [field for field in fields if field in scored_labels.columns]
    if shared:
        comparison = scored_labels.loc[:, ["candidate_id", *shared]].merge(
            features.loc[:, ["candidate_id", *shared]], on="candidate_id",
            how="left", validate="one_to_one", suffixes=("__score", "__sidecar"),
        )
        for field in shared:
            left = pd.to_numeric(comparison[f"{field}__score"], errors="coerce").to_numpy(float)
            right = pd.to_numeric(comparison[f"{field}__sidecar"], errors="coerce").to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError(f"score/sidecar mismatch for {field}")
    sidecar_only = [field for field in fields if field not in shared]
    output = scored_labels.merge(
        features.loc[:, ["candidate_id", *sidecar_only]], on="candidate_id",
        how="inner", validate="one_to_one",
    ).merge(
        admission.loc[:, ["candidate_id", "raw_expected_bps", "mapped_ev_available"]],
        on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(output) != len(scored_labels):
        raise ValueError("Bayesian sidecars do not exactly cover score identities")
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True)
    output["policy_label_available_ts"] = pd.to_datetime(output["policy_label_available_ts"], utc=True)
    if output["geometry_bundle_sha256"].dropna().astype(str).nunique() != 1:
        raise ValueError("source-aligned Bayesian B1 requires one frozen Geometry/K9 identity")
    if any(field.startswith("k09__cluster_") for field in fields):
        raise ValueError("raw K9 memberships are prohibited")
    return output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _training_rows(work: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, float, ParentExpectation]:
    spec = CANONICAL_N5_SPEC
    start = cutoff - pd.DateOffset(months=spec.train_months)
    net = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    raw_expected = pd.to_numeric(work["raw_expected_bps"], errors="coerce")
    train_all = work.loc[
        work["__decision_ts__"].ge(start)
        & work["__decision_ts__"].lt(cutoff)
        & work["policy_label_available_ts"].lt(cutoff)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & work["mapped_ev_available"].fillna(False).astype(bool)
        & np.isfinite(net) & np.isfinite(raw_expected)
        & np.isfinite(pd.to_numeric(work["final_score"], errors="coerce")),
    ].copy()
    if len(train_all) < 1_000:
        raise ValueError("insufficient resolved prior support")
    parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
    floor = float(np.quantile(train_all["final_score"].to_numpy(float), 1.0 - spec.top_fraction, method="higher"))
    train = _equal_month_sample(
        train_all.loc[train_all["final_score"].ge(floor)].copy(), spec.train_cap, seed=spec.seed,
    )
    if len(train) < 1_000:
        raise ValueError("insufficient resolved top-30% support")
    return train, floor, parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--feature-sidecar", type=Path, required=True)
    parser.add_argument("--admission-provenance", type=Path, required=True)
    parser.add_argument("--conversion-block-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--spec", choices=sorted(FINALIST_SPECS), default="B1_raw_singleton_l100_mean")
    parser.add_argument(
        "--multiplier-cap", type=float, default=1.75,
        help="Post-admission authority cap; 1.0 tests a demoter-only trust overlay.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if not 0.25 <= float(args.multiplier_cap) <= 1.75:
        raise ValueError("--multiplier-cap must lie in [0.25, 1.75]")
    spec = FINALIST_SPECS[args.spec]
    contract = json.loads(args.contract.read_text())
    fields = tuple(map(str, contract["features"]))
    if len(fields) != 45 or len(set(fields)) != 45:
        raise ValueError("matched B1 requires the frozen 45-field non-posterior contract")
    work = _prepare(
        _read_unique(args.scored_label_ledger, "scored label ledger"),
        _read_unique(args.feature_sidecar, "Bayesian feature sidecar"),
        _read_unique(args.admission_provenance, "admission provenance"), fields=fields,
    )
    blocks = pd.read_parquet(args.conversion_block_audit).copy()
    required = {"cutoff", "held_end_exclusive", "geometry_bundle_sha256"}
    missing = sorted(required.difference(blocks.columns))
    if missing:
        raise ValueError(f"conversion block audit lacks: {missing}")
    blocks["cutoff"] = pd.to_datetime(blocks["cutoff"], utc=True)
    blocks["held_end_exclusive"] = pd.to_datetime(blocks["held_end_exclusive"], utc=True)
    if blocks["geometry_bundle_sha256"].astype(str).nunique() != 1:
        raise ValueError("matched B1 rejects mixed geometry")
    args.out_dir.mkdir(parents=True)
    output_parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for block_index, row in enumerate(blocks.sort_values("cutoff", kind="stable").itertuples(index=False)):
        cutoff, held_end = pd.Timestamp(row.cutoff), pd.Timestamp(row.held_end_exclusive)
        held = work.loc[work["__decision_ts__"].ge(cutoff) & work["__decision_ts__"].lt(held_end)].copy()
        result = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        try:
            train, floor, parent = _training_rows(work, cutoff)
        except ValueError as exc:
            result["n5_available"] = False
            result["n5_unavailable_reason"] = str(exc)
            result["n5_bundle_cutoff"] = cutoff
            result["trust_size_multiplier"] = np.float32(1.0)
            audits.append({"block_index": block_index, "cutoff": cutoff, "held_end_exclusive": held_end, "status": "unit_size_warmup", "reason": str(exc), "held_rows": len(held)})
        else:
            # The empirical-Bayes primitive estimates correction authority
            # around this fold-local parent.  Both values are therefore fit
            # exclusively from labels resolved before the held cutoff.
            train["parent_expected_bps"] = parent.predict(train["final_score"])
            held["parent_expected_bps"] = parent.predict(held["final_score"])
            train_prediction, held_prediction, fit_audit = fit_trust_model(train, held, fields, spec)
            reference_quality = sizing_quality(train_prediction, train, spec.sizing_mode)
            held_quality = sizing_quality(held_prediction, held, spec.sizing_mode)
            result = pd.concat([result.reset_index(drop=True), held_prediction.as_frame()], axis=1)
            result["n5_available"] = True
            result["n5_unavailable_reason"] = None
            result["n5_bundle_cutoff"] = cutoff
            result["trust_size_multiplier"] = causal_size_multiplier(
                reference_quality, held_quality, cap=float(args.multiplier_cap),
            ).astype(np.float32)
            audits.append({"block_index": block_index, "cutoff": cutoff, "held_end_exclusive": held_end, "status": "complete", "reason": None, "held_rows": len(held), "train_rows": len(train), "training_score_floor": floor, **fit_audit})
        output_parts.append(result)
        print(json.dumps({"event": "bayesian_b1_block_complete", **audits[-1]}, default=str), flush=True)
    output = pd.concat(output_parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(output) != len(work) or output["candidate_id"].duplicated().any():
        raise AssertionError("Bayesian B1 changed candidate identity")
    output.to_parquet(args.out_dir / "bayesian_b1_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "bayesian_b1_block_audit.parquet", index=False)
    manifest = {
        "schema": SCHEMA, "spec": spec.__dict__, "multiplier_cap": float(args.multiplier_cap),
        "rows": len(output), "feature_count": len(fields),
        "geometry_refit_cadence": "never", "raw_k9_memberships_used": False,
        "ranking_changes": False, "admission_changes": False,
        "source_hashes": {"scored_label_ledger": _sha(args.scored_label_ledger), "feature_sidecar": _sha(args.feature_sidecar), "admission_provenance": _sha(args.admission_provenance), "conversion_block_audit": _sha(args.conversion_block_audit), "contract": _sha(args.contract)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
