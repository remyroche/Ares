#!/usr/bin/env python3
"""Fit a causal trust shrinker on the current exact-reserve score contract.

This is a deliberately narrow reconciliation experiment.  It consumes the
current exact-reserve provenance, not the legacy MDA surface, and emits one
strictly chronological held block.  Its output is intended for the existing
score-correction and causal-admission evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    TrustModelSpec,
    catalogue,
    fit_trust_model,
    residual_classes,
)


SEED = 20260812
RAW_K9_PREFIX = "k09__cluster_"
IDENTITY = {
    "candidate_id", "__decision_ts__", "__symbol__", "side_name", "policy_path_valid",
    "policy_label_available_ts", "policy_net_bps", "policy_gross_bps", "policy_exit_reason",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
}
EXCLUDE_PREFIXES = ("causal_", "ev_mapping_", "calibration_")
EXCLUDE_DIRECT = {
    "conversion_bundle_sha256", "geometry_bundle_sha256", "ev_score_family_id", "upstream_bundle_sha256",
    "stack_is_prequential", "policy_outcome_source", "severe_affects_final_score",
    "raw_expected_bps", "parent_expected_bps",
    "__n5_raw_expected_map_bps", "__n5_raw_expected_map_admitted",
    "__n5_map_decision_ts",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_equal_month(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    month = frame["__decision_ts__"].dt.to_period("M")
    groups = list(frame.groupby(month, sort=True))
    quota = max(1, cap // len(groups))
    selected: list[pd.DataFrame] = []
    for _, part in groups:
        part = part.sort_values(["candidate_id"], kind="stable")
        selected.append(part.head(quota))
    out = pd.concat(selected, ignore_index=False)
    if len(out) < cap:
        remaining = frame.drop(index=out.index).sort_values(["candidate_id"], kind="stable")
        out = pd.concat([out, remaining.head(cap - len(out))])
    return out.head(cap).copy()


def _eligible_fields(train: pd.DataFrame) -> list[str]:
    fields: list[str] = []
    for field in train.columns:
        if field in IDENTITY or field in EXCLUDE_DIRECT or field.startswith(EXCLUDE_PREFIXES) or field.startswith(RAW_K9_PREFIX):
            continue
        values = pd.to_numeric(train[field], errors="coerce")
        if values.notna().mean() >= 0.90 and values.var() > 1e-12:
            fields.append(field)
    if len(fields) < 12:
        raise ValueError(f"only {len(fields)} causal fields pass coverage/variance")
    return fields


def _residual_mi_field_selection(
    train: pd.DataFrame, fields: list[str], *, limit: int,
) -> tuple[list[str], list[dict[str, float | str]]]:
    """Choose a compact trust surface using train-only binned residual MI.

    The empirical-Bayes heads are additive singleton effects.  Giving them
    every coverage-valid field can accumulate marginal noise.  This selector
    ranks only their information about the binned policy residual conditional
    on the causal EV map; it never sees held rows or raw K9 memberships.
    """

    if limit < 12:
        raise ValueError("residual-MI feature limit must retain at least 12 fields")
    realised = pd.to_numeric(train["policy_net_bps"], errors="raise").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="raise").to_numpy(float)
    target = residual_classes(realised, expected)
    target_count = np.bincount(target.astype(int))
    target_probability = target_count[target.astype(int)] / max(1, len(target))
    target_entropy = float(-np.sum(
        (target_count[target_count > 0] / len(target))
        * np.log(target_count[target_count > 0] / len(target))
    ))
    scores: list[dict[str, float | str]] = []
    for field in fields:
        values = pd.to_numeric(train[field], errors="coerce")
        # Percentile bins avoid a high-variance scale dominating MI.  Rank is
        # deterministic after candidate-ID ordered equal-month sampling.
        ranks = values.rank(method="average", pct=True).fillna(0.5).to_numpy(float)
        codes = np.minimum(15, np.floor(ranks * 16.0).astype(np.int16))
        joint = np.zeros((16, int(target.max()) + 1), dtype=np.float64)
        np.add.at(joint, (codes, target.astype(int)), 1.0)
        probability = joint / max(1.0, joint.sum())
        px = probability.sum(axis=1, keepdims=True)
        py = probability.sum(axis=0, keepdims=True)
        valid = probability > 0.0
        mi = float(np.sum(probability[valid] * np.log(
            probability[valid] / (px * py)[valid]
        )))
        scores.append({
            "field": field,
            "residual_mi": mi,
            "normalized_residual_mi": mi / target_entropy if target_entropy > 0 else 0.0,
            "target_mean_probability": float(target_probability.mean()),
        })
    scores.sort(key=lambda item: (-float(item["residual_mi"]), str(item["field"])))
    return [str(item["field"]) for item in scores[:limit]], scores


def _train_cdf(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if len(reference) < 100:
        return np.full(len(value), 0.5, dtype=np.float32)
    return (np.searchsorted(reference, np.asarray(value, dtype=float), side="right") / len(reference)).astype(np.float32)


def _timestamp_top30(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(["__decision_ts__", "final_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    pos = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    active = pos < np.maximum(1, np.ceil(size * 0.30).astype(int))
    return pd.Series(active, index=ordered.index).reindex(frame.index).to_numpy(bool)


def _train_selection_mask(frame: pd.DataFrame, mode: str) -> tuple[np.ndarray, str]:
    """Select the high-conviction Bayesian training surface causally.

    The requested research contract is the top 30% *within each decision
    timestamp*.  Retain the old global score quantile only as an explicitly
    named historical control: different cross-sectional candidate counts must
    not determine whether a row enters the trust model's training set.
    """

    if mode == "timestamp_top30":
        return _timestamp_top30(frame), "within_decision_timestamp_top_30pct"
    if mode == "global_top30":
        floor = float(frame["final_score"].quantile(0.70))
        return frame["final_score"].ge(floor).to_numpy(bool), f"global_final_score_ge_{floor:.12g}"
    if mode == "mixed_top30_reference":
        return np.ones(len(frame), dtype=bool), "75pct_timestamp_top30_plus_25pct_lower_reference"
    raise ValueError(f"unsupported train selection {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-extra", type=Path, action="append", default=[],
        help=(
            "Optional additional identity-disjoint canonical ledger partitions "
            "with the identical ordered schema."
        ),
    )
    parser.add_argument(
        "--latest-fit-feature-sidecar", type=Path,
        help=(
            "Optional target-free ledger containing active_rule_* fields. It is "
            "joined by candidate_id onto the unchanged canonical score/admission "
            "source, isolating feature uplift from direct ranker changes."
        ),
    )
    parser.add_argument(
        "--latest-fit-feature-sidecar-extra", type=Path, action="append", default=[],
        help=(
            "Optional additional, identity-disjoint latest-fit ledgers. This permits "
            "a fold to span adjacent immutable replay partitions without materialising "
            "a duplicate concatenated parquet."
        ),
    )
    parser.add_argument(
        "--expected-map-sidecar", type=Path, action="append", default=[],
        help=(
            "Optional candidate-keyed causal EV-map partitions. When supplied, "
            "the declared expected-net field replaces the source's legacy 21-day "
            "map as N5's raw common-bps anchor."
        ),
    )
    parser.add_argument(
        "--expected-map-field",
        default="cell_day_trim_15pct__expected_net_bps",
    )
    parser.add_argument(
        "--expected-map-admitted-field",
        default="cell_day_trim_15pct__admitted",
    )
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--held-end", required=True)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument(
        "--train-selection",
        choices=("timestamp_top30", "global_top30", "mixed_top30_reference"),
        default="timestamp_top30",
        help="Use per-timestamp top-30%% by default; global_top30 is a legacy control.",
    )
    trust_specs = {
        spec.name: spec
        for pipeline in ("bayesian", "nonlinear")
        for spec in catalogue()[pipeline]
        if spec.model_family in {
            "empirical_bayes", "local_distribution_forest_proxy",
            "cell_day_residual_forest",
        }
    }
    parser.add_argument(
        "--trust-spec",
        choices=tuple(sorted(trust_specs)),
        default="B5_stable_ranklossfp_l125_predictive",
        help=(
            "Causal empirical-Bayes or Local Distribution Forest Proxy specification. "
            "All arms use the identical train/held population and field gate."
        ),
    )
    parser.add_argument(
        "--field-selection",
        choices=(
            "all_coverage", "legacy_coverage",
            "residual_mi_top12", "residual_mi_top24",
        ),
        default="all_coverage",
        help=(
            "all_coverage includes latest-fit active_rule fields; legacy_coverage "
            "removes them as the matched feature-substrate control. Compact MI "
            "selection is fitted on training rows only."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    train_start = pd.Timestamp(args.train_start, tz="UTC")
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    held_end = pd.Timestamp(args.held_end, tz="UTC")
    if not train_start < cutoff < held_end:
        raise ValueError("require train_start < cutoff < held_end")
    source_paths = [args.source, *list(args.source_extra)]
    source_schema = pq.read_schema(source_paths[0]).names
    source_parts: list[pd.DataFrame] = []
    for source_path in source_paths:
        if pq.read_schema(source_path).names != source_schema:
            raise ValueError("canonical source partitions do not share the same ordered schema")
        source_parts.append(pd.read_parquet(source_path))
    frame = pd.concat(source_parts, ignore_index=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    sidecar_audit: dict[str, object] = {"used": False}
    if args.latest_fit_feature_sidecar is not None:
        sidecar_paths = [
            args.latest_fit_feature_sidecar,
            *list(args.latest_fit_feature_sidecar_extra),
        ]
        sidecar_schema = pq.read_schema(sidecar_paths[0]).names
        active_fields = [field for field in sidecar_schema if field.startswith("active_rule_")]
        if len(active_fields) != 10:
            raise ValueError(
                f"latest-fit feature sidecar must expose exactly 10 active_rule fields, found {len(active_fields)}"
            )
        sidecar_parts: list[pd.DataFrame] = []
        for sidecar_path in sidecar_paths:
            fields_here = [
                field for field in pq.read_schema(sidecar_path).names
                if field.startswith("active_rule_")
            ]
            if fields_here != active_fields:
                raise ValueError(
                    "latest-fit sidecar partitions do not share the same ordered field contract"
                )
            sidecar_parts.append(pd.read_parquet(
                sidecar_path,
                columns=["candidate_id", "__decision_ts__", *active_fields],
            ))
        sidecar = pd.concat(sidecar_parts, ignore_index=True)
        sidecar["__decision_ts__"] = pd.to_datetime(
            sidecar["__decision_ts__"], utc=True, errors="raise",
        )
        if sidecar["candidate_id"].duplicated().any():
            raise ValueError("latest-fit feature sidecar contains duplicate candidate IDs")
        overlap = frame.loc[:, ["candidate_id", "__decision_ts__"]].merge(
            sidecar, on="candidate_id", how="inner", validate="one_to_one",
            suffixes=("", "__sidecar"),
        )
        timestamp_match = overlap["__decision_ts__"].eq(overlap["__decision_ts____sidecar"])
        if not timestamp_match.all():
            raise ValueError("latest-fit feature sidecar has an identity/timestamp mismatch")
        frame = frame.merge(
            sidecar.drop(columns="__decision_ts__"),
            on="candidate_id", how="left", validate="one_to_one",
        )
        sidecar_audit = {
            "used": True,
            "paths": [str(path) for path in sidecar_paths],
            "sha256": [_sha(path) for path in sidecar_paths],
            "fields": active_fields,
            "rows": int(len(sidecar)),
            "matched_source_rows": int(len(overlap)),
            "identity_timestamp_parity_on_overlap": True,
        }
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("source must be a unique long-only candidate ledger")
    if not frame["stack_is_prequential"].fillna(False).astype(bool).all():
        raise AssertionError("source contains non-prequential upstream scores")
    expected_map_audit: dict[str, object] = {
        "used": False,
        "field": "causal_21d_side_expected_net_bps",
    }
    expected_source_field = "causal_21d_side_expected_net_bps"
    if args.expected_map_sidecar:
        map_parts: list[pd.DataFrame] = []
        required_map_fields = [
            "candidate_id", "__decision_ts__", args.expected_map_field,
            args.expected_map_admitted_field,
        ]
        for path in args.expected_map_sidecar:
            names = pq.read_schema(path).names
            missing = sorted(set(required_map_fields).difference(names))
            if missing:
                raise ValueError(f"expected-map sidecar {path} lacks {missing}")
            map_parts.append(pd.read_parquet(path, columns=required_map_fields))
        mapped = pd.concat(map_parts, ignore_index=True)
        mapped["__decision_ts__"] = pd.to_datetime(
            mapped["__decision_ts__"], utc=True, errors="raise",
        )
        if mapped["candidate_id"].duplicated().any():
            raise ValueError("expected-map sidecars contain duplicate candidate IDs")
        expected_source_field = "__n5_raw_expected_map_bps"
        mapped = mapped.rename(columns={
            args.expected_map_field: expected_source_field,
            args.expected_map_admitted_field: "__n5_raw_expected_map_admitted",
            "__decision_ts__": "__n5_map_decision_ts",
        })
        frame = frame.merge(mapped, on="candidate_id", how="left", validate="one_to_one")
        overlap = frame["__n5_map_decision_ts"].notna()
        if not frame.loc[overlap, "__decision_ts__"].eq(
            frame.loc[overlap, "__n5_map_decision_ts"]
        ).all():
            raise ValueError("expected-map sidecar identity/timestamp mismatch")
        expected_map_audit = {
            "used": True,
            "paths": [str(path) for path in args.expected_map_sidecar],
            "sha256": [_sha(path) for path in args.expected_map_sidecar],
            "field": str(args.expected_map_field),
            "admitted_field": str(args.expected_map_admitted_field),
            "rows": int(len(mapped)),
            "matched_source_rows": int(overlap.sum()),
        }
    raw = pd.to_numeric(frame[expected_source_field], errors="coerce")
    usable = frame["policy_path_valid"].fillna(False).astype(bool) & raw.notna() & frame["policy_net_bps"].notna()
    train_all = frame.loc[
        frame["__decision_ts__"].ge(train_start) & frame["__decision_ts__"].lt(cutoff)
        & frame["policy_label_available_ts"].lt(cutoff) & usable
    ].copy()
    held = frame.loc[frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)].copy()
    if len(train_all) < 2_000 or held.empty:
        raise ValueError(f"insufficient support {len(train_all)=} {len(held)=}")
    if args.latest_fit_feature_sidecar is not None:
        missing_train = train_all[active_fields].isna().any(axis=1)
        missing_held = held[active_fields].isna().any(axis=1)
        if missing_train.any() or missing_held.any():
            raise ValueError(
                "latest-fit feature sidecar must exactly cover the requested train/held fold; "
                f"missing_train={int(missing_train.sum())} missing_held={int(missing_held.sum())}"
            )
        sidecar_audit["requested_train_rows_covered"] = int(len(train_all))
        sidecar_audit["requested_held_rows_covered"] = int(len(held))
    parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
    train_all["raw_expected_bps"] = raw.loc[train_all.index].to_numpy(float)
    held["raw_expected_bps"] = raw.loc[held.index].to_numpy(float)
    train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
    held["parent_expected_bps"] = parent.predict(held["final_score"])
    train_mask, selection_definition = _train_selection_mask(train_all, args.train_selection)
    if args.train_selection == "mixed_top30_reference":
        top_mask = _timestamp_top30(train_all)
        top_cap = int(round(int(args.train_cap) * 0.75))
        reference_cap = int(args.train_cap) - top_cap
        train = pd.concat([
            _sample_equal_month(train_all.loc[top_mask].copy(), top_cap),
            _sample_equal_month(train_all.loc[~top_mask].copy(), reference_cap),
        ], ignore_index=False)
    else:
        train = _sample_equal_month(train_all.loc[train_mask].copy(), int(args.train_cap))
    fields = _eligible_fields(train)
    feature_selection_audit: list[dict[str, float | str]] = []
    if args.field_selection == "legacy_coverage":
        fields = [field for field in fields if not field.startswith("active_rule_")]
        if len(fields) < 12:
            raise ValueError("legacy coverage control has fewer than 12 causal fields")
    if args.field_selection != "all_coverage":
        if args.field_selection.startswith("residual_mi_"):
            limit = 12 if args.field_selection.endswith("top12") else 24
            fields, feature_selection_audit = _residual_mi_field_selection(
                train, fields, limit=limit,
            )
    trust_spec = trust_specs[str(args.trust_spec)]
    train_pred, held_pred, audit = fit_trust_model(train, held, fields, trust_spec)
    output = held.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score", "policy_path_valid",
        "policy_label_available_ts", "policy_gross_bps", "policy_net_bps", "policy_exit_reason", "geometry_bundle_sha256",
        "raw_expected_bps", "parent_expected_bps",
    ]].copy().reset_index(drop=True)
    output = pd.concat([output, held_pred.as_frame().reset_index(drop=True)], axis=1)
    output["posterior_expected_rank_train"] = _train_cdf(train_pred.expected_bps, held_pred.expected_bps)
    output["posterior_adverse_rank_train"] = _train_cdf(train_pred.p_adverse_tail, held_pred.p_adverse_tail)
    output["timestamp_top30"] = _timestamp_top30(held)
    # Historical correction evaluators use ``bayes_*`` column names.  The
    # values are generic train-derived trust ranks and are equally valid for a
    # Bayesian or Local Distribution Forest Proxy arm.
    output["bayes_available"] = True
    map_token = "cell_day_trim15" if expected_map_audit["used"] else "exactreserve"
    output["arm"] = f"{trust_spec.name}_current_{map_token}"
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    correctness = {
        "unique_candidate_ids": bool(not output["candidate_id"].duplicated().any()),
        "strictly_prequential_source": True,
        "train_labels_resolved_before_cutoff": bool((train_all["policy_label_available_ts"] < cutoff).all()),
        "held_labels_after_decision": bool((output["policy_label_available_ts"] > output["__decision_ts__"]).all()),
        "raw_k9_memberships_used": False,
        "train_held_disjoint": bool(set(train_all["candidate_id"]).isdisjoint(set(output["candidate_id"]))),
    }
    (args.out_dir / "correctness_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "strict_r3_current_exactreserve_trust_fold_v2",
        "source": [str(path) for path in source_paths],
        "source_sha256": [_sha(path) for path in source_paths],
        "train_start": str(train_start), "cutoff": str(cutoff), "held_end": str(held_end),
        "train_rows_before_selection": int(len(train_all)), "train_rows_after_selection_before_cap": int(train_mask.sum()),
        "train_rows": int(len(train)), "held_rows": int(len(held)),
        "train_selection": str(args.train_selection), "train_selection_definition": selection_definition,
        "fields": fields, "field_count": len(fields), "field_selection": str(args.field_selection),
        "feature_selection_audit": feature_selection_audit, "spec": trust_spec.__dict__, "fit_audit": audit,
        "target": (
            "Cell-day mapping error: clipped policy_net_bps minus causal "
            "raw_expected_bps; local residual distribution and overestimation "
            "probabilities"
            if trust_spec.target_mode.startswith("cell_day_residual_") else
            "policy net bps, using the declared causal raw common-bps map versus "
            "a train-only parent final-score expectation"
        ),
        "expected_map": expected_map_audit,
        "causality": "current exact-reserve prequential source; fit labels resolve before cutoff; held outcomes used only after score construction",
        "geometry": "same current source bundle; raw K9 membership prohibited", "seed": SEED,
        "latest_fit_feature_sidecar": sidecar_audit,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(output)), "fields": len(fields), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
