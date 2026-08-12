#!/usr/bin/env python3
"""Train-bin-conditioned LDF sizing ablation.

The canonical LDF maps absolute predictive quality to a global multiplier.  At
high base-score tails that multiplier can saturate, leaving reliability
features no sizing authority.  This research-only runner adds a bounded,
strictly train-referenced *within-score-bin* correction.  It never changes
final-score ranking, candidate identity, or causal EV admission.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_forest_support_sizing import (  # noqa: E402
    BASELINE_N5_PARAMS,
    fit_n5_forest,
)
from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    causal_size_multiplier,
)
import scripts.run_strict_r3_k9weighted_mda as mda  # noqa: E402
import scripts.run_strict_r3_n5_canonical_selection as selection  # noqa: E402


SEED = 20260811

# The legacy compact LDF contract predates schema-v2's explicit name for the
# same prequential rank-domain value.  Preserve the frozen contract name when
# fitting, while sourcing it only from the causal ``base_rank42`` field.
FEATURE_ALIASES = {"base_rank": "base_rank42"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", default="2025-04-01")
    parser.add_argument("--fold-end", default="2025-07-01")
    parser.add_argument("--train-cap", type=int, default=40_000)
    return parser.parse_args()


def _conditional_multiplier(
    *,
    train_score: np.ndarray,
    train_quality: np.ndarray,
    held_score: np.ndarray,
    held_quality: np.ndarray,
    bins: int,
    floor: float,
    cap: float,
) -> np.ndarray:
    """Map quality within train-only score bins; no held score affects edges."""

    score = np.asarray(train_score, dtype=float)
    quality = np.asarray(train_quality, dtype=float)
    if len(score) != len(quality) or len(score) < bins * 20:
        raise ValueError("insufficient train support for conditional LDF sizing")
    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) < 3:
        return causal_size_multiplier(np.sort(quality), held_quality, floor=floor, cap=cap)
    train_bin = np.searchsorted(edges[1:-1], score, side="right")
    held_bin = np.searchsorted(edges[1:-1], np.asarray(held_score, dtype=float), side="right")
    output = np.empty(len(held_quality), dtype=np.float32)
    fallback = np.sort(quality[np.isfinite(quality)])
    for bucket in range(len(edges) - 1):
        reference = np.sort(quality[(train_bin == bucket) & np.isfinite(quality)])
        if len(reference) < 100:
            reference = fallback
        mask = held_bin == bucket
        if mask.any():
            output[mask] = causal_size_multiplier(
                reference, np.asarray(held_quality, dtype=float)[mask], floor=floor, cap=cap,
            )
    return output


def _relative_multiplier(
    global_multiplier: np.ndarray,
    conditional_multiplier: np.ndarray,
    global_reference: np.ndarray,
    *,
    alpha: float,
    floor: float,
    cap: float,
) -> np.ndarray:
    """Tilt an existing size by local-versus-global train-referenced quality."""

    relative = np.divide(
        conditional_multiplier, global_reference,
        out=np.ones_like(conditional_multiplier, dtype=float), where=global_reference > 0.0,
    )
    return np.clip(
        np.asarray(global_multiplier, dtype=float) * (1.0 + float(alpha) * (relative - 1.0)),
        float(floor), float(cap),
    ).astype(np.float32)


def _folds(surface: Path, fields: list[str], start: str, end: str, train_cap: int):
    source_fields = [FEATURE_ALIASES.get(field, field) for field in fields]
    required = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "geometry_bundle_sha256",
        "policy_path_valid", "policy_label_available_ts", "policy_net_bps", "policy_gross_bps",
        "policy_exit_reason", "final_score", *source_fields,
    ]))
    cutoffs = pd.date_range(start, end, freq="MS", tz="UTC")
    for ordinal, cutoff in enumerate(cutoffs):
        held_end = cutoff + pd.offsets.MonthBegin(1)
        window_start = cutoff - pd.DateOffset(months=3) - pd.Timedelta(days=21)
        frame = pd.read_parquet(
            surface, columns=required,
            filters=[("__decision_ts__", ">=", window_start), ("__decision_ts__", "<", held_end)],
        )
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
        for contract_field, source_field in FEATURE_ALIASES.items():
            if contract_field in fields and contract_field not in frame:
                frame[contract_field] = frame[source_field]
        admitted, _ = apply_causal_21d_side_admission(
            frame, score_column="final_score", net_column="policy_net_bps",
            decision_column="__decision_ts__", label_available_column="policy_label_available_ts",
            identity_column="candidate_id", spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
        )
        admitted["raw_expected_bps"] = pd.to_numeric(
            admitted["causal_21d_side_expected_net_bps"], errors="coerce"
        )
        admitted["mapped_ev_available"] = admitted["raw_expected_bps"].notna()
        train_start = cutoff - pd.DateOffset(months=3)
        train_all = admitted.loc[
            admitted["__decision_ts__"].ge(train_start)
            & admitted["__decision_ts__"].lt(cutoff)
            & admitted["policy_label_available_ts"].lt(cutoff)
            & admitted["policy_path_valid"].fillna(False).astype(bool)
            & admitted["mapped_ev_available"].astype(bool)
            & np.isfinite(pd.to_numeric(admitted["policy_net_bps"], errors="coerce"))
        ].copy()
        parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
        train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
        floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(floor)].copy()
        train = mda._equal_month_sample(train, int(train_cap), SEED + ordinal)
        held = admitted.loc[
            admitted["__decision_ts__"].ge(cutoff) & admitted["__decision_ts__"].lt(held_end)
        ].copy()
        held["parent_expected_bps"] = parent.predict(held["final_score"])
        held["trust_gate_active"] = held["mapped_ev_available"].astype(bool) & pd.to_numeric(
            held["final_score"], errors="coerce"
        ).ge(floor)
        yield ordinal, cutoff, train, held
        del frame, admitted, train_all
        gc.collect()


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    contract = json.loads(args.feature_contract.read_text())
    fields = list(contract["compact_fields"])
    # The conditional signal must be relative to the global LDF assessment.
    # Adding two absolute multipliers merely creates a universal leverage
    # increase.  These arms instead tilt the existing multiplier according to
    # local quality / global-quality within a train-only score bin.
    arms = (("global", None, 0.0), ("relative5_a025", 5, 0.25), ("relative5_a050", 5, 0.50))
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for fold, cutoff, train, held in _folds(
        args.surface, fields, str(args.fold_start), str(args.fold_end), int(args.train_cap),
    ):
        edges = selection._edges(train, fields)
        bundle, train_prediction = fit_n5_forest(train, fields, edges, params=BASELINE_N5_PARAMS)
        held_prediction, global_multiplier = bundle.size_multiplier(held)
        train_quality = train_prediction.quality(bundle.params.risk_aversion)
        held_quality = held_prediction.quality(bundle.params.risk_aversion)
        for arm, bins, alpha in arms:
            if bins is None:
                multiplier = global_multiplier
            else:
                conditional = _conditional_multiplier(
                    train_score=pd.to_numeric(train["final_score"], errors="coerce").to_numpy(float),
                    train_quality=train_quality,
                    held_score=pd.to_numeric(held["final_score"], errors="coerce").to_numpy(float),
                    held_quality=held_quality, bins=int(bins),
                    floor=bundle.params.size_floor, cap=bundle.params.size_cap,
                )
                global_reference = causal_size_multiplier(
                    np.sort(train_quality), held_quality,
                    floor=bundle.params.size_floor, cap=bundle.params.size_cap,
                )
                multiplier = _relative_multiplier(
                    global_multiplier, conditional, global_reference,
                    alpha=float(alpha), floor=bundle.params.size_floor, cap=bundle.params.size_cap,
                )
            output = selection._output(held, held_prediction, multiplier, arm=arm)
            output["fold"] = fold
            parts.append(output)
            audit.append({
                "fold": fold, "cutoff": str(cutoff), "arm": arm, "fields": len(fields),
                "train_rows": len(train), "held_rows": len(held),
                "mean_multiplier": float(np.mean(multiplier)),
                "p05_multiplier": float(np.quantile(multiplier, 0.05)),
                "p95_multiplier": float(np.quantile(multiplier, 0.95)),
            })
        del train, held
        gc.collect()
    output = pd.concat(parts, ignore_index=True)
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "oof_predictions.parquet", index=False, compression="zstd")
    metrics = []
    for arm, block in output.groupby("arm", sort=True):
        metrics.extend([
            selection._period_tail_metrics(block, arm=str(arm), period_kind="global").assign(metric_kind="global"),
            selection._period_tail_metrics(block, arm=str(arm), period_kind="month").assign(metric_kind="month"),
        ])
    pd.concat(metrics, ignore_index=True).to_parquet(args.out_dir / "metrics.parquet", index=False)
    pd.DataFrame(audit).to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_ldf_train_bin_conditional_sizing_ablation_v1",
        "surface": str(args.surface), "feature_contract": str(args.feature_contract),
        "target": "canonical policy net bps", "ranking": "unchanged final_score",
        "admission": "unchanged causal 21-day side-local EV admission",
        "conditional_reference": "LDF OOB train quality within five train-only final-score bins, divided by the train-only global quality reference",
        "integration": "bounded relative residual sizing correction; no candidate/admission/rank change",
        "arms": [arm for arm, _, _ in arms],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
