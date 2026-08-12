#!/usr/bin/env python3
"""Replay the original 45-field single-forest LDF on a frozen K9 contract.

This is deliberately separate from the compact two-forest v4 challenger.
It reproduces the selected ``N5_drf_support_l110_meanrisk`` architecture:

* score and causal-EV admission are already frozen upstream;
* a 10-bin parent policy-net map is fitted on resolved prior rows only;
* the LDF trains on the upstream top 30% of a three-month resolved window;
* its 45 stable fields and CMI interactions are fitted only on that window;
* it changes only a bounded post-admission relative size multiplier.

The producer rejects rolling geometry.  It never passes raw K9 memberships to
the forest, only frozen bundle-invariant K9/leaf summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_forest_support_sizing import (  # noqa: E402
    CANONICAL_N5_SPEC,
    CANONICAL_SCHEMA,
    fit_canonical_n5_bundle,
)
from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    discover_cmi_edges,
)


SCHEMA = "strict_r3_ldf_v3_frozen_walkforward_v1"


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


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = pd.to_datetime(frame["__decision_ts__"], utc=True).dt.to_period("M").astype(str)
    tokens = sorted(month.unique())
    quota = max(1, int(cap) // len(tokens))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for token in tokens:
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
    required_labels = {
        "candidate_id", "__decision_ts__", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", "final_score", "geometry_bundle_sha256",
    }
    missing = sorted(required_labels.difference(scored_labels.columns))
    if missing:
        raise ValueError(f"scored label ledger lacks: {missing}")
    missing = sorted(set(fields).difference(features.columns))
    if missing:
        raise ValueError(f"45-field LDF sidecar lacks fields: {missing}")
    required_admission = {"candidate_id", "raw_expected_bps", "mapped_ev_available"}
    missing = sorted(required_admission.difference(admission.columns))
    if missing:
        raise ValueError(f"causal admission provenance lacks: {missing}")
    # The score ledger is authoritative for fields it already owns.  Verify
    # the sidecar instead of accepting suffix-renamed duplicates silently.
    shared = [field for field in fields if field in scored_labels.columns]
    if shared:
        compare = scored_labels.loc[:, ["candidate_id", *shared]].merge(
            features.loc[:, ["candidate_id", *shared]], on="candidate_id",
            how="left", validate="one_to_one", suffixes=("__score", "__sidecar"),
        )
        for field in shared:
            left = pd.to_numeric(compare[f"{field}__score"], errors="coerce").to_numpy(float)
            right = pd.to_numeric(compare[f"{field}__sidecar"], errors="coerce").to_numpy(float)
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
        raise ValueError("LDF sidecars do not exactly cover score identities")
    output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True)
    output["policy_label_available_ts"] = pd.to_datetime(
        output["policy_label_available_ts"], utc=True,
    )
    identities = output["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(identities) != 1:
        raise ValueError("original pooled LDF requires one frozen geometry/K9 identity")
    if any(field.startswith("k09__cluster_") for field in fields):
        raise ValueError("original pooled LDF cannot use raw K9 memberships")
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
        & np.isfinite(net)
        & np.isfinite(raw_expected)
        & np.isfinite(pd.to_numeric(work["final_score"], errors="coerce")),
    ].copy()
    if len(train_all) < 1_000:
        raise ValueError("insufficient resolved prior support")
    parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
    floor = float(np.quantile(train_all["final_score"].to_numpy(float), 1.0 - spec.top_fraction, method="higher"))
    train = train_all.loc[train_all["final_score"].ge(floor)].copy()
    train = _equal_month_sample(train, spec.train_cap, seed=spec.seed)
    if len(train) < 1_000:
        raise ValueError("insufficient resolved top-30% support")
    return train, floor, parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument("--feature-sidecar", type=Path, required=True)
    parser.add_argument("--admission-provenance", type=Path, required=True)
    parser.add_argument("--conversion-block-audit", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=ROOT / "config/strict_r3_ldf_support_v3.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != CANONICAL_SCHEMA:
        raise ValueError("runner only supports the original v3 single-forest LDF contract")
    if contract.get("canonical_arm_legacy_id") != "N5_drf_support_l110_meanrisk":
        raise ValueError("runner requires N5_drf_support_l110_meanrisk")
    fields = tuple(map(str, contract["features"]))
    if len(fields) != 45 or len(set(fields)) != 45:
        raise ValueError("original LDF requires its frozen 45-field contract")

    work = _prepare(
        _read_unique(args.scored_label_ledger, "scored label ledger"),
        _read_unique(args.feature_sidecar, "LDF feature sidecar"),
        _read_unique(args.admission_provenance, "admission provenance"),
        fields=fields,
    )
    blocks = pd.read_parquet(args.conversion_block_audit).copy()
    # The repaired lockstep producer records a single immutable geometry hash
    # on every block but, unlike the older conversion audit, does not repeat a
    # redundant ``geometry_refit_cadence`` string.  Accept that newer audit
    # only after proving the same frozen identity is present for every block;
    # do not silently treat a mixed or episodic audit as frozen.
    if "geometry_refit_cadence" not in blocks.columns:
        required_lockstep = {"cutoff", "held_end_exclusive", "geometry_bundle_sha256"}
        missing_lockstep = sorted(required_lockstep.difference(blocks.columns))
        if missing_lockstep:
            raise ValueError(f"conversion block audit lacks: {missing_lockstep}")
        if blocks["geometry_bundle_sha256"].isna().any() or (
            blocks["geometry_bundle_sha256"].astype(str).nunique() != 1
        ):
            raise ValueError(
                "lockstep audit without geometry_refit_cadence must prove one frozen geometry hash",
            )
        blocks["geometry_refit_cadence"] = "never"
    required = {"cutoff", "held_end_exclusive", "geometry_bundle_sha256", "geometry_refit_cadence"}
    missing = sorted(required.difference(blocks.columns))
    if missing:
        raise ValueError(f"conversion block audit lacks: {missing}")
    blocks["cutoff"] = pd.to_datetime(blocks["cutoff"], utc=True)
    blocks["held_end_exclusive"] = pd.to_datetime(blocks["held_end_exclusive"], utc=True)
    if blocks["geometry_bundle_sha256"].astype(str).nunique() != 1 or not blocks["geometry_refit_cadence"].eq("never").all():
        raise ValueError("original LDF rejects periodic or mixed K9 geometry")
    args.out_dir.mkdir(parents=True)
    bundle_root = args.out_dir / "bundles"
    output_parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for block_index, row in enumerate(blocks.sort_values("cutoff", kind="stable").itertuples(index=False)):
        cutoff, held_end = pd.Timestamp(row.cutoff), pd.Timestamp(row.held_end_exclusive)
        held = work.loc[
            work["__decision_ts__"].ge(cutoff) & work["__decision_ts__"].lt(held_end)
        ].copy()
        if held.empty:
            raise ValueError(f"no score rows in LDF block {cutoff}")
        result = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        try:
            train, floor, parent = _training_rows(work, cutoff)
        except ValueError as exc:
            result["n5_available"] = False
            result["n5_unavailable_reason"] = str(exc)
            result["n5_bundle_cutoff"] = cutoff
            result["trust_size_multiplier"] = np.float32(1.0)
            audits.append({
                "block_index": block_index, "cutoff": cutoff, "held_end_exclusive": held_end,
                "status": "unit_size_warmup", "reason": str(exc), "held_rows": len(held),
                "geometry_bundle_sha256": str(row.geometry_bundle_sha256),
            })
        else:
            edges, _bins = discover_cmi_edges(
                train, fields, mode=CANONICAL_N5_SPEC.cmi_weighting, stable=True,
            )
            bundle = fit_canonical_n5_bundle(
                train, fields, edges, parent_expectation=parent, cutoff=cutoff,
                training_score_floor=floor,
            )
            bundle_dir = bundle_root / f"cutoff={cutoff:%Y%m%d}"
            bundle_dir.mkdir(parents=True)
            joblib.dump(bundle, bundle_dir / "ldf_v3_bundle.joblib", compress=3)
            manifest = bundle.manifest()
            manifest.update({
                "source_hashes": {
                    "scored_label_ledger": _sha(args.scored_label_ledger),
                    "feature_sidecar": _sha(args.feature_sidecar),
                    "admission_provenance": _sha(args.admission_provenance),
                    "conversion_block_audit": _sha(args.conversion_block_audit),
                },
                "geometry_bundle_sha256": str(row.geometry_bundle_sha256),
                "geometry_refit_cadence": "never",
                "training": "three prior resolved months; upstream top 30%; equal-month cap 60000",
            })
            (bundle_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            # ``final_score`` is deliberately one of the frozen 45 LDF
            # fields.  Do not append it again: duplicate pandas labels would
            # expand the robust-transform matrix at serving time.
            prediction, multiplier = bundle.size_multiplier(
                held.loc[:, ["raw_expected_bps", *fields]],
            )
            result = pd.concat([result.reset_index(drop=True), prediction.as_frame()], axis=1)
            result["n5_available"] = True
            result["n5_unavailable_reason"] = None
            result["n5_bundle_cutoff"] = cutoff
            result["trust_size_multiplier"] = multiplier.astype(np.float32)
            audits.append({
                "block_index": block_index, "cutoff": cutoff, "held_end_exclusive": held_end,
                "status": "complete", "reason": None, "held_rows": len(held),
                "train_rows": len(train), "training_score_floor": floor,
                "cmi_edges": len(edges), "geometry_bundle_sha256": str(row.geometry_bundle_sha256),
            })
        output_parts.append(result)
        print(json.dumps({"event": "ldf_v3_block_complete", **audits[-1]}, default=str), flush=True)
    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if len(output) != len(work) or output["candidate_id"].duplicated().any():
        raise AssertionError("LDF v3 changed candidate identity")
    output.to_parquet(args.out_dir / "ldf_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "ldf_block_audit.parquet", index=False)
    manifest = {
        "schema": SCHEMA,
        "contract": str(args.contract), "contract_sha256": _sha(args.contract),
        "canonical_arm_legacy_id": "N5_drf_support_l110_meanrisk",
        "rows": len(output), "feature_count": len(fields),
        "geometry_refit_cadence": "never", "raw_k9_memberships_used": False,
        "ranking_changes": False, "admission_changes": False,
        "source_hashes": {
            "scored_label_ledger": _sha(args.scored_label_ledger),
            "feature_sidecar": _sha(args.feature_sidecar),
            "admission_provenance": _sha(args.admission_provenance),
            "conversion_block_audit": _sha(args.conversion_block_audit),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
