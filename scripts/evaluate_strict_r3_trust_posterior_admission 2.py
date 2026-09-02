#!/usr/bin/env python3
"""Evaluate causal trust posterior EV as admission and frozen-admission overlay.

The canonical control is an explicitly supplied causal EV map.  A trust
model fitted strictly before the held block already emits an expected policy
net value in common bps.  Re-binning its held-only score would create a new
producer-vintage cold start, so this evaluator tests the two executable uses:

1. posterior admission: admit when posterior expected net is at least +50 bps;
2. frozen admission overlay: preserve canonical admission and only reorder it
   by the posterior expected net.

Outcomes are joined in the supplied control ledger but are consulted only
after the two target-free selections have been constructed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
THRESHOLD_BPS = 50.0
BLEND_ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.50)
CORROBORATION_PROBABILITY = 0.65
CORROBORATION_RESIDUAL_Q25_BPS = -50.0
CORROBORATION_MIN_SUPPORT = 120.0


def _selected_metrics(
    frame: pd.DataFrame, *, arm: str, admitted: str, score: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], set[str]]]:
    work = frame.copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    work["week"] = work["__decision_ts__"].dt.strftime("%G-W%V")
    periods = [("all", "all", work)]
    periods.extend(("month", str(k), v) for k, v in work.groupby("month", sort=True))
    periods.extend(("week", str(k), v) for k, v in work.groupby("week", sort=True))
    rows: list[dict[str, object]] = []
    selections: dict[tuple[str, str, str], set[str]] = {}
    for scope, period, block in periods:
        sort_fields = [score]
        ascending = [False]
        # The causal EV map has only 20 bins.  Its mapped common-bps value is
        # authoritative, while the target-free canonical score supplies a
        # deterministic, economically relevant tie-break inside equal EV bins.
        if score != "final_score":
            sort_fields.append("final_score")
            ascending.append(False)
        sort_fields.append("candidate_id")
        ascending.append(True)
        eligible = block.loc[block[admitted].fillna(False).astype(bool)].sort_values(
            sort_fields, ascending=ascending, kind="stable",
        )
        for kind, selected in [("all_admitted", eligible), *[
            (
                f"admitted_top_{tail:g}",
                eligible.head(max(1, int(math.ceil(tail * len(eligible)))))
                if len(eligible) else eligible,
            )
            for tail in TAILS
        ]]:
            valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "period_scope": scope, "period": period, "kind": kind,
                "score_rows": int(len(block)), "admitted_rows": int(len(eligible)),
                "selected_rows": int(len(selected)), "valid_outcomes": int(len(valid)),
                "outcome_coverage": float(len(valid) / max(len(selected), 1)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
            selections[(scope, period, kind)] = set(selected["candidate_id"].astype(str))
    return pd.DataFrame(rows), selections


def _uplift(control: pd.DataFrame, challenger: pd.DataFrame) -> pd.DataFrame:
    keys = ["period_scope", "period", "kind"]
    merged = challenger.merge(
        control, on=keys, how="inner", suffixes=("_challenger", "_control"),
        validate="one_to_one",
    )
    output = merged.loc[:, keys].copy()
    for name in (
        "admitted_rows", "selected_rows", "valid_outcomes", "outcome_coverage",
        "net_bps_per_trade", "positive_rate",
    ):
        output[f"control_{name}"] = merged[f"{name}_control"]
        output[f"challenger_{name}"] = merged[f"{name}_challenger"]
        output[f"uplift_{name}"] = (
            pd.to_numeric(merged[f"{name}_challenger"], errors="coerce")
            - pd.to_numeric(merged[f"{name}_control"], errors="coerce")
        )
    return output


def _overlap(
    control: dict[tuple[str, str, str], set[str]],
    challenger: dict[tuple[str, str, str], set[str]],
    *, challenger_arm: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in sorted(set(control).intersection(challenger)):
        left, right = control[key], challenger[key]
        intersection, union = left & right, left | right
        rows.append({
            "challenger_arm": challenger_arm,
            "period_scope": key[0], "period": key[1], "kind": key[2],
            "control_rows": len(left), "challenger_rows": len(right),
            "intersection_rows": len(intersection),
            "jaccard": len(intersection) / len(union) if union else np.nan,
            "control_retained_fraction": len(intersection) / len(left) if left else np.nan,
        })
    return pd.DataFrame(rows)


def _ordered(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    fields, ascending = [score], [False]
    if score != "final_score":
        fields.append("final_score")
        ascending.append(False)
    fields.append("candidate_id")
    ascending.append(True)
    return frame.sort_values(fields, ascending=ascending, kind="stable")


def _net_summary(frame: pd.DataFrame) -> tuple[int, float, float]:
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ]
    value = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
    return (
        int(len(valid)),
        float(value.mean()) if len(value) else np.nan,
        float(value.std(ddof=1) / math.sqrt(len(value))) if len(value) > 1 else np.nan,
    )


def _matched_cardinality_audit(
    frame: pd.DataFrame,
    *,
    challenger_arm: str,
    challenger_admitted: str,
    challenger_score: str,
) -> pd.DataFrame:
    """Separate admission filtering from reranking at identical trade counts.

    The legacy tail table takes a percentage of each arm's own admitted set.
    When a demoter rejects rows, that compares different cardinalities and can
    exaggerate bps/trade uplift.  This audit adds two non-confounded views:

    * ``matched_challenger_count``: compare the challenger with the canonical
      top-N where N is the challenger's selected count;
    * ``fixed_canonical_population``: keep the canonical admitted population
      and its tail count fixed, changing only its ordering.
    """

    work = frame.copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    work["week"] = work["__decision_ts__"].dt.strftime("%G-W%V")
    periods = [("all", "all", work)]
    periods.extend(("month", str(k), v) for k, v in work.groupby("month", sort=True))
    periods.extend(("week", str(k), v) for k, v in work.groupby("week", sort=True))
    rows: list[dict[str, object]] = []
    for scope, period, block in periods:
        canonical_pool = _ordered(
            block.loc[block["frozen_admission"].fillna(False).astype(bool)],
            "causal_21d_side_expected_net_bps",
        )
        challenger_pool = _ordered(
            block.loc[block[challenger_admitted].fillna(False).astype(bool)],
            challenger_score,
        )
        for tail in (None, *TAILS):
            kind = "all_admitted" if tail is None else f"admitted_top_{tail:g}"
            challenger_n = (
                len(challenger_pool) if tail is None
                else (max(1, int(math.ceil(tail * len(challenger_pool)))) if len(challenger_pool) else 0)
            )
            canonical_n = (
                len(canonical_pool) if tail is None
                else (max(1, int(math.ceil(tail * len(canonical_pool)))) if len(canonical_pool) else 0)
            )
            matched_control = canonical_pool.head(challenger_n)
            matched_challenger = challenger_pool.head(challenger_n)
            fixed_control = canonical_pool.head(canonical_n)
            fixed_challenger = _ordered(canonical_pool, challenger_score).head(canonical_n)
            for mode, control_selected, challenger_selected in (
                ("matched_challenger_count", matched_control, matched_challenger),
                ("fixed_canonical_population", fixed_control, fixed_challenger),
            ):
                control_valid, control_net, control_se = _net_summary(control_selected)
                challenger_valid, challenger_net, challenger_se = _net_summary(challenger_selected)
                left = set(control_selected["candidate_id"].astype(str))
                right = set(challenger_selected["candidate_id"].astype(str))
                rows.append({
                    "challenger_arm": challenger_arm,
                    "period_scope": scope,
                    "period": period,
                    "kind": kind,
                    "comparison_mode": mode,
                    "canonical_admitted_rows": int(len(canonical_pool)),
                    "challenger_admitted_rows": int(len(challenger_pool)),
                    "control_selected_rows": int(len(control_selected)),
                    "challenger_selected_rows": int(len(challenger_selected)),
                    "control_valid_outcomes": control_valid,
                    "challenger_valid_outcomes": challenger_valid,
                    "control_net_bps_per_trade": control_net,
                    "challenger_net_bps_per_trade": challenger_net,
                    "uplift_net_bps_per_trade": challenger_net - control_net,
                    "control_net_standard_error": control_se,
                    "challenger_net_standard_error": challenger_se,
                    "selection_intersection_rows": int(len(left & right)),
                    "selection_jaccard": (
                        float(len(left & right) / len(left | right)) if left | right else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-ledger", type=Path, required=True)
    parser.add_argument("--trust-predictions", type=Path, required=True)
    parser.add_argument("--control-map-sidecar", type=Path, action="append", default=[])
    parser.add_argument(
        "--control-expected-field",
        default="cell_day_trim_15pct__expected_net_bps",
    )
    parser.add_argument(
        "--control-admitted-field",
        default="cell_day_trim_15pct__admitted",
    )
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start = pd.Timestamp(args.evaluation_start, tz="UTC")
    end = pd.Timestamp(args.evaluation_end, tz="UTC")
    control_columns = [
        "candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps",
        "final_score",
    ]
    legacy_map_columns = [
        "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps",
    ]
    control_names = set(pq.read_schema(args.control_ledger).names)
    if not args.control_map_sidecar:
        missing_legacy = sorted(set(legacy_map_columns).difference(control_names))
        if missing_legacy:
            raise ValueError(
                "control ledger lacks its embedded legacy EV map and no explicit "
                f"control-map sidecar was supplied: {missing_legacy}"
            )
        control_columns.extend(legacy_map_columns)
    control = pd.read_parquet(args.control_ledger, columns=control_columns)
    if args.control_map_sidecar:
        map_columns = [
            "candidate_id", "__decision_ts__", args.control_expected_field,
            args.control_admitted_field,
        ]
        mapped = pd.concat(
            [pd.read_parquet(path, columns=map_columns) for path in args.control_map_sidecar],
            ignore_index=True,
        )
        if mapped["candidate_id"].duplicated().any():
            raise ValueError("control-map sidecars contain duplicate candidate IDs")
        mapped = mapped.rename(columns={
            args.control_expected_field: "__control_expected_net_bps",
            args.control_admitted_field: "__control_admitted",
            "__decision_ts__": "__control_map_decision_ts",
        })
        control = control.merge(mapped, on="candidate_id", how="left", validate="one_to_one")
        overlap = control["__control_map_decision_ts"].notna()
        control["__decision_ts__"] = pd.to_datetime(control["__decision_ts__"], utc=True)
        control["__control_map_decision_ts"] = pd.to_datetime(
            control["__control_map_decision_ts"], utc=True,
        )
        if not control.loc[overlap, "__decision_ts__"].eq(
            control.loc[overlap, "__control_map_decision_ts"]
        ).all():
            raise ValueError("control-map sidecar identity/timestamp mismatch")
        control["causal_21d_side_expected_net_bps"] = control["__control_expected_net_bps"]
        control["causal_21d_side_admitted_ge_50bps"] = control["__control_admitted"]
    prediction_columns = [
        "candidate_id", "__decision_ts__", "arm", "posterior_expected_bps",
        "posterior_predictive_q10", "p_ev_positive", "p_adverse_tail",
    ]
    optional_prediction_columns = [
        "posterior_predictive_sd", "trust_effective_support",
        "posterior_residual_mean_bps", "posterior_residual_q10_bps",
        "posterior_residual_q25_bps", "p_map_overestimate_50bps",
        "p_map_overestimate_100bps", "p_map_overestimate_200bps",
    ]
    prediction_names = set(pq.read_schema(args.trust_predictions).names)
    prediction_columns.extend(
        field for field in optional_prediction_columns if field in prediction_names
    )
    trust = pd.read_parquet(args.trust_predictions, columns=prediction_columns)
    for frame, label in ((control, "control"), (trust, "trust")):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{label} ledger contains duplicate candidate IDs")
    control = control.loc[
        control["__decision_ts__"].ge(start) & control["__decision_ts__"].lt(end)
    ].copy()
    trust = trust.loc[
        trust["__decision_ts__"].ge(start) & trust["__decision_ts__"].lt(end)
    ].copy()
    joined = control.merge(
        trust, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one",
    )
    if joined["arm"].isna().any() or len(joined) != len(control):
        raise ValueError("trust predictions do not exactly cover the evaluation population")
    arm = str(joined["arm"].dropna().iloc[0])
    # The trust producer deliberately preserves every held candidate. A row can
    # still lack a posterior when its causal EV-map parent was unavailable. That
    # is not an identity failure: posterior-only admission fails closed, while a
    # frozen-admission overlay is a no-op and retains the canonical EV.
    joined["trust_prediction_available"] = np.isfinite(pd.to_numeric(
        joined["posterior_expected_bps"], errors="coerce",
    ))
    joined["posterior_expected_bps_raw"] = joined["posterior_expected_bps"]
    joined["posterior_expected_bps"] = pd.to_numeric(
        joined["posterior_expected_bps"], errors="coerce",
    ).where(
        joined["trust_prediction_available"],
        pd.to_numeric(joined["causal_21d_side_expected_net_bps"], errors="coerce"),
    )
    joined["posterior_admitted_ge_50bps"] = pd.to_numeric(
        joined["posterior_expected_bps"], errors="raise",
    ).ge(THRESHOLD_BPS) & joined["trust_prediction_available"]
    joined["frozen_admission"] = joined[
        "causal_21d_side_admitted_ge_50bps"
    ].eq(True)
    canonical_ev = pd.to_numeric(
        joined["causal_21d_side_expected_net_bps"], errors="raise",
    ).to_numpy(float)
    posterior_ev = pd.to_numeric(
        joined["posterior_expected_bps"], errors="raise",
    ).to_numpy(float)

    control_metrics, control_sets = _selected_metrics(
        joined, arm="canonical_reserve_seeded_control",
        admitted="frozen_admission", score="causal_21d_side_expected_net_bps",
    )
    posterior_metrics, posterior_sets = _selected_metrics(
        joined, arm=f"{arm}__posterior_admission",
        admitted="posterior_admitted_ge_50bps", score="posterior_expected_bps",
    )
    overlay_metrics, overlay_sets = _selected_metrics(
        joined, arm=f"{arm}__frozen_admission_overlay",
        admitted="frozen_admission", score="posterior_expected_bps",
    )
    metric_parts = [control_metrics, posterior_metrics, overlay_metrics]
    uplift_parts = [
        _uplift(control_metrics, posterior_metrics).assign(
            challenger_arm=f"{arm}__posterior_admission"
        ),
        _uplift(control_metrics, overlay_metrics).assign(
            challenger_arm=f"{arm}__frozen_admission_overlay"
        ),
    ]
    overlap_parts = [
        _overlap(control_sets, posterior_sets, challenger_arm=f"{arm}__posterior_admission"),
        _overlap(control_sets, overlay_sets, challenger_arm=f"{arm}__frozen_admission_overlay"),
    ]
    cardinality_parts = [
        _matched_cardinality_audit(
            joined,
            challenger_arm=f"{arm}__posterior_admission",
            challenger_admitted="posterior_admitted_ge_50bps",
            challenger_score="posterior_expected_bps",
        ),
        _matched_cardinality_audit(
            joined,
            challenger_arm=f"{arm}__frozen_admission_overlay",
            challenger_admitted="frozen_admission",
            challenger_score="posterior_expected_bps",
        ),
    ]
    decomposition_fields = [
        "candidate_id", "__decision_ts__", "posterior_expected_bps",
        "posterior_expected_bps_raw", "trust_prediction_available",
        "posterior_predictive_q10", "p_ev_positive", "p_adverse_tail",
        "posterior_admitted_ge_50bps", "frozen_admission",
        "causal_21d_side_expected_net_bps",
    ]
    decomposition_fields.extend(
        field for field in optional_prediction_columns if field in joined.columns
    )
    has_residual_distribution = {
        "posterior_residual_q25_bps", "p_map_overestimate_100bps",
        "trust_effective_support",
    }.issubset(joined.columns)
    if has_residual_distribution:
        residual_q25 = pd.to_numeric(
            joined["posterior_residual_q25_bps"], errors="raise",
        ).to_numpy(float)
        overestimate_probability = pd.to_numeric(
            joined["p_map_overestimate_100bps"], errors="raise",
        ).to_numpy(float)
        support = pd.to_numeric(
            joined["trust_effective_support"], errors="raise",
        ).to_numpy(float)
        probability_confidence = np.clip(
            (overestimate_probability - 0.50) / 0.50, 0.0, 1.0,
        )
        quantile_severity = np.clip(
            (-residual_q25 - 25.0) / 175.0, 0.0, 1.0,
        )
        support_factor = support / (support + 300.0)
        joined["trust_risk_corroborated"] = (
            (overestimate_probability >= CORROBORATION_PROBABILITY)
            & (residual_q25 <= CORROBORATION_RESIDUAL_Q25_BPS)
            & (support >= CORROBORATION_MIN_SUPPORT)
        )
        joined["trust_authority_unit"] = np.where(
            joined["trust_risk_corroborated"].to_numpy(bool),
            support_factor * np.sqrt(probability_confidence * quantile_severity),
            0.0,
        ).astype(np.float32)
        decomposition_fields.extend(["trust_risk_corroborated", "trust_authority_unit"])
    for alpha in BLEND_ALPHAS:
        token = f"a{int(round(alpha * 100)):02d}"
        symmetric_score = f"trust_symmetric_ev_{token}"
        symmetric_admitted = f"trust_symmetric_admitted_{token}"
        demotion_score = f"trust_demotion_ev_{token}"
        demotion_admitted = f"trust_demotion_admitted_{token}"
        joined[symmetric_score] = (
            (1.0 - alpha) * canonical_ev + alpha * posterior_ev
        ).astype(np.float32)
        joined[symmetric_admitted] = joined[symmetric_score].ge(THRESHOLD_BPS)
        joined[demotion_score] = (
            canonical_ev + alpha * np.minimum(posterior_ev - canonical_ev, 0.0)
        ).astype(np.float32)
        joined[demotion_admitted] = joined[demotion_score].ge(THRESHOLD_BPS)
        for mode, admitted_field, score_field in (
            ("symmetric", symmetric_admitted, symmetric_score),
            ("demotion_only", demotion_admitted, demotion_score),
        ):
            challenger_arm = f"{arm}__{mode}_{token}"
            challenger_metrics, challenger_sets = _selected_metrics(
                joined, arm=challenger_arm,
                admitted=admitted_field, score=score_field,
            )
            metric_parts.append(challenger_metrics)
            uplift_parts.append(
                _uplift(control_metrics, challenger_metrics).assign(
                    challenger_arm=challenger_arm,
                )
            )
            overlap_parts.append(
                _overlap(control_sets, challenger_sets, challenger_arm=challenger_arm)
            )
            cardinality_parts.append(
                _matched_cardinality_audit(
                    joined,
                    challenger_arm=challenger_arm,
                    challenger_admitted=admitted_field,
                    challenger_score=score_field,
                )
            )
        decomposition_fields.extend(
            [symmetric_score, symmetric_admitted, demotion_score, demotion_admitted]
        )
        if has_residual_distribution:
            corroborated_score = f"trust_corroborated_ev_{token}"
            corroborated_admitted = f"trust_corroborated_admitted_{token}"
            veto_admitted = f"trust_corroborated_veto_admitted_{token}"
            authority = alpha * joined["trust_authority_unit"].to_numpy(float)
            joined[corroborated_score] = (
                canonical_ev
                + authority * np.minimum(posterior_ev - canonical_ev, 0.0)
            ).astype(np.float32)
            joined[corroborated_admitted] = joined[corroborated_score].ge(THRESHOLD_BPS)
            joined[veto_admitted] = (
                joined["frozen_admission"]
                & ~joined["trust_risk_corroborated"]
            )
            for mode, admitted_field in (
                ("corroborated", corroborated_admitted),
                ("corroborated_fixed_admission", "frozen_admission"),
                ("corroborated_veto", veto_admitted),
            ):
                challenger_arm = f"{arm}__{mode}_{token}"
                challenger_metrics, challenger_sets = _selected_metrics(
                    joined, arm=challenger_arm,
                    admitted=admitted_field, score=corroborated_score,
                )
                metric_parts.append(challenger_metrics)
                uplift_parts.append(
                    _uplift(control_metrics, challenger_metrics).assign(
                        challenger_arm=challenger_arm,
                    )
                )
                overlap_parts.append(
                    _overlap(control_sets, challenger_sets, challenger_arm=challenger_arm)
                )
                cardinality_parts.append(
                    _matched_cardinality_audit(
                        joined,
                        challenger_arm=challenger_arm,
                        challenger_admitted=admitted_field,
                        challenger_score=corroborated_score,
                    )
                )
            decomposition_fields.extend(
                [corroborated_score, corroborated_admitted, veto_admitted]
            )
    metrics = pd.concat(metric_parts, ignore_index=True)
    uplift = pd.concat(uplift_parts, ignore_index=True)
    overlap = pd.concat(overlap_parts, ignore_index=True)
    cardinality = pd.concat(cardinality_parts, ignore_index=True)
    args.out_dir.mkdir(parents=True)
    joined.loc[:, decomposition_fields].to_parquet(
        args.out_dir / "admission_decomposition.parquet", index=False,
    )
    metrics.to_parquet(args.out_dir / "metrics.parquet", index=False)
    uplift.to_parquet(args.out_dir / "matched_uplift.parquet", index=False)
    overlap.to_parquet(args.out_dir / "matched_selection_overlap.parquet", index=False)
    cardinality.to_parquet(
        args.out_dir / "matched_cardinality_audit.parquet", index=False,
    )
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_trust_posterior_admission_v1",
        "control_ledger": str(args.control_ledger),
        "control_map_sidecar": [str(path) for path in args.control_map_sidecar],
        "control_expected_field": str(args.control_expected_field),
        "control_admitted_field": str(args.control_admitted_field),
        "trust_predictions": str(args.trust_predictions),
        "evaluation": [str(start), str(end)],
        "threshold_bps": THRESHOLD_BPS,
        "blend_alphas": list(BLEND_ALPHAS),
        "corroboration": {
            "probability": CORROBORATION_PROBABILITY,
            "residual_q25_bps": CORROBORATION_RESIDUAL_Q25_BPS,
            "minimum_effective_support": CORROBORATION_MIN_SUPPORT,
            "available": has_residual_distribution,
        },
        "trust_prediction_coverage": {
            "available_rows": int(joined["trust_prediction_available"].sum()),
            "total_rows": int(len(joined)),
            "posterior_only_admission": "fail_closed",
            "frozen_admission_overlay": "canonical_ev_noop",
        },
        "arms": metrics["arm"].unique().tolist(),
        "contract": (
            "train-only posterior expected policy net in common bps; no held-score EV "
            "rebinning; canonical same-model reserve-seeded admission retained as control"
        ),
        "outcomes_used_only_after_selection": True,
        "matched_cardinality_audit": (
            "separates admission filtering from within-canonical-population reranking"
        ),
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(joined), "arm": arm, "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
