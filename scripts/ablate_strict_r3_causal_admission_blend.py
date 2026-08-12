#!/usr/bin/env python3
"""Ablate a conservative/seeded blend of two strictly causal EV maps.

This does not touch scores, labels, model bundles, or portfolio mechanics.
It can only combine two precomputed candidate-keyed expected-policy-net maps
which have the same strict producer lineage, decision timestamp and realised
policy outcome contract.  Both maps must themselves have been produced using
only labels available before the decision.  The blend is therefore causal,
but remains a research ablation until evaluated on a later frozen period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LINEAGE = [
    "candidate_id", "__decision_ts__", "conversion_bundle_sha256",
    "upstream_bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id",
    "final_score", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summaries(
    frame: pd.DataFrame, *, expected: pd.Series, selected: pd.Series, arm: str,
) -> list[dict[str, object]]:
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    out: list[dict[str, object]] = []
    for frequency, values in (("all", pd.Series("all", index=frame.index)), ("M", decision.dt.tz_localize(None).dt.to_period("M").astype(str)), ("W-MON", decision.dt.tz_localize(None).dt.to_period("W-MON").astype(str))):
        for period, positions in values.groupby(values, observed=True, sort=True).groups.items():
            local = np.asarray(list(positions), dtype=np.int64)
            mask = selected.iloc[local] & valid.iloc[local]
            net = pd.to_numeric(frame["policy_net_bps"].iloc[local][mask], errors="coerce")
            gross = pd.to_numeric(frame["policy_gross_bps"].iloc[local][mask], errors="coerce")
            out.append({
                "arm": arm, "frequency": frequency, "period": str(period),
                "scored_rows": int(len(local)), "selected_rows": int(selected.iloc[local].sum()),
                "valid_selected_rows": int(mask.sum()),
                "expected_net_bps_per_trade": float(expected.iloc[local][mask].mean()) if mask.any() else np.nan,
                "gross_bps_per_trade": float(gross.mean()) if len(gross) else np.nan,
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_net_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conservative", type=Path, required=True)
    parser.add_argument("--seeded", type=Path, required=True)
    parser.add_argument("--seeded-weights", nargs="+", type=float, default=[0.50, 0.75])
    parser.add_argument("--net-floor-bps", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable admission-blend output exists: {args.out_dir}")
    weights = sorted(set(float(value) for value in args.seeded_weights))
    if not weights or any(value <= 0.0 or value >= 1.0 for value in weights):
        raise ValueError("seeded weights must lie strictly between zero and one")

    need = [*LINEAGE, "causal_21d_side_expected_net_bps", "ev_mapping_vintage_mode"]
    conservative = pd.read_parquet(args.conservative, columns=need).rename(
        columns={"causal_21d_side_expected_net_bps": "conservative_expected_bps", "ev_mapping_vintage_mode": "conservative_mode"}
    )
    seeded = pd.read_parquet(args.seeded, columns=need).rename(
        columns={"causal_21d_side_expected_net_bps": "seeded_expected_bps", "ev_mapping_vintage_mode": "seeded_mode"}
    )
    if conservative.candidate_id.duplicated().any() or seeded.candidate_id.duplicated().any():
        raise ValueError("both admission ledgers must have unique candidate identities")
    compare = [column for column in LINEAGE if column != "candidate_id"]
    work = conservative.merge(
        seeded.loc[:, ["candidate_id", *compare, "seeded_expected_bps", "seeded_mode"]],
        on="candidate_id", how="inner", suffixes=("", "__seeded"), validate="one_to_one",
    )
    if len(work) != len(conservative) or len(work) != len(seeded):
        raise AssertionError("causal-map blend requires identical candidate populations")
    for column in compare:
        left = work[column]
        right = work[f"{column}__seeded"]
        if pd.api.types.is_numeric_dtype(left):
            same = np.isclose(left.to_numpy(float), right.to_numpy(float), equal_nan=True)
        else:
            same = left.astype(str).eq(right.astype(str)).to_numpy()
        if not bool(np.all(same)):
            raise AssertionError(f"maps have incompatible {column} contract")
        work = work.drop(columns=f"{column}__seeded")
    if set(work.conservative_mode.dropna().astype(str)) != {"strict_oof_exact_producer_reserve_map_plus_causal_residual_v1"}:
        raise ValueError("conservative input is not the declared exact-producer reserve map")
    if set(work.seeded_mode.dropna().astype(str)) != {"same_model_42d_reserve_seeded_v1"}:
        raise ValueError("seeded input is not the declared same-model reserve-seeded map")
    for column in ("conservative_expected_bps", "seeded_expected_bps"):
        if not np.isfinite(pd.to_numeric(work[column], errors="coerce")).all():
            raise ValueError(f"causal-map blend has non-finite {column}")

    selection = work.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    metrics: list[dict[str, object]] = []
    for weight in weights:
        name = f"E{int(round(args.net_floor_bps)):03d}_exact_seed_w{int(round(weight * 100)):02d}"
        expected = (
            (1.0 - weight) * work["conservative_expected_bps"]
            + weight * work["seeded_expected_bps"]
        )
        selected = expected.ge(float(args.net_floor_bps)).fillna(False)
        selection[f"{name}__admitted"] = selected.to_numpy(bool)
        selection[f"{name}__expected_net_bps"] = expected.to_numpy(float)
        metrics.extend(_summaries(work, expected=expected, selected=selected, arm=name))
    args.out_dir.mkdir(parents=True)
    selection.to_parquet(args.out_dir / "threshold_selection.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(args.out_dir / "threshold_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_causal_admission_blend_v1",
        "conservative": str(args.conservative), "conservative_sha256": _sha(args.conservative),
        "seeded": str(args.seeded), "seeded_sha256": _sha(args.seeded),
        "seeded_weights": weights, "net_floor_bps": float(args.net_floor_bps),
        "contract": (
            "candidate-keyed convex blend of a strict exact-producer reserve map and a same-model reserve-seeded "
            "causal map; identical prequential score, policy outcome and producer lineage required; no raw score "
            "cross-vintage pooling, model refit, label change or held-period calibration"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(work), "arms": len(weights)}))


if __name__ == "__main__":
    main()
