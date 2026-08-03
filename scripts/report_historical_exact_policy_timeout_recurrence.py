#!/usr/bin/env python3
"""Diagnose recurrence of late 12h-timeout recoveries on historical exact rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_execution_ev_july_exact_economics import (  # noqa: E402
    COHORT_FLAGS,
    IDENTITY,
    LABEL_SCHEMA,
    _artifact_record,
    _bound_file,
    _read_json,
    _resolve,
    _sha256,
    _validate_identity,
)
from scripts.report_execution_ev_timeout_ablation import (  # noqa: E402
    _validate_label_values,
    _validate_shared_label_lineage,
    exit_mix_metrics,
    horizon_metrics,
    pair_horizon_labels,
    paired_delta_metrics,
)
from scripts.score_execution_ev_forward_population import apply_global_admission  # noqa: E402

SCHEMA = "historical_exact_policy_timeout_recurrence_v2"
DEFAULT_SCORE = Path(
    "data_perp/artifacts/current_exact_policy_global_book_mapping_source_20260729_v1/"
    "causal_mapped_candidates.parquet"
)
DEFAULT_SCORE_PROVENANCE = Path(
    "data_perp/artifacts/current_exact_policy_global_book_mapping_source_20260729_v1/"
    "manifest.json"
)
DEFAULT_12H_ROOT = Path(
    "data_perp/artifacts/execution_ev_policy_labels_12h_20260725_v1"
)
DEFAULT_24H_ROOT = Path(
    "data_perp/artifacts/execution_ev_policy_labels_20260725_v1"
)
SOURCE_SCORE_COLUMN = "mapped_direct_net"
SCORE_COLUMN = "mapped_execution_ev"
OOF_FLAG = "causal_recent_side_isotonic_ev__is_oof"


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy_component_hashes(policy_path: Path) -> dict[str, Any]:
    policy = _read_json(policy_path)
    core = {
        "policy_pathway_id": policy.get("policy_pathway_id"),
        "replay_timeframe": policy.get("replay_timeframe"),
        "exit_geometry_contract": policy.get("exit_geometry_contract"),
    }
    strategies = policy.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("signed policy has no strategy records")
    strategy_hashes: dict[str, str] = {}
    for strategy in strategies:
        strategy_id = str(
            strategy.get("canonical_strategy_id") or strategy.get("strategy_id") or ""
        )
        if not strategy_id or strategy_id in strategy_hashes:
            raise ValueError("signed policy strategy IDs are missing or duplicated")
        strategy_hashes[strategy_id] = _canonical_json_sha256(strategy)
    return {
        "core_sha256": _canonical_json_sha256(core),
        "strategies_sha256": _canonical_json_sha256(strategies),
        "strategy_record_sha256": strategy_hashes,
        "strategy_count": len(strategies),
    }


def _validate_historical_manifest(
    manifest: Mapping[str, Any],
    labels_path: Path,
    *,
    expected_horizon_minutes: int,
) -> None:
    if manifest.get("schema") != LABEL_SCHEMA:
        raise ValueError("unexpected historical exact-label schema")
    coverage = manifest.get("coverage", {}).get("overall", {})
    rows = int(manifest.get("output", {}).get("rows", -1))
    if (
        rows <= 0
        or int(coverage.get("complete", -1)) != rows
        or int(coverage.get("missing", -1)) != 0
        or float(coverage.get("coverage", -1.0)) != 1.0
    ):
        raise ValueError("historical exact labels do not have complete coverage")
    contract = manifest.get("exit_policy_contract", {})
    if (
        contract.get("replay_timeframe") != "1m"
        or int(contract.get("horizon_minutes", -1)) != expected_horizon_minutes
    ):
        raise ValueError("historical exact-label horizon is incorrect")
    accounting = manifest.get("accounting", {})
    if (
        accounting.get("candidate_local_exit_replay") is not True
        or accounting.get("portfolio_concurrency_applied") is not False
        or accounting.get("net_return") != "gross return minus fee return"
        or "spread drag is embedded in gross return"
        not in str(accounting.get("cost_return", ""))
    ):
        raise ValueError("historical exact-label accounting is incompatible")
    _bound_file(
        manifest,
        manifest["output"],
        labels_path,
        role=f"historical {expected_horizon_minutes}m labels",
    )


def build_frozen_historical_population(
    scores: pd.DataFrame,
    labels_12h: pd.DataFrame,
    labels_24h: pd.DataFrame,
    *,
    top_k_fraction: float = 0.10,
) -> pd.DataFrame:
    scores = _validate_identity(scores, role="historical causal mapped OOF scores")
    required = {
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        SOURCE_SCORE_COLUMN,
        OOF_FLAG,
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"historical score fields missing: {missing}")
    eligible = scores.copy()
    score = pd.to_numeric(eligible[SOURCE_SCORE_COLUMN], errors="coerce")
    strict = eligible[OOF_FLAG].fillna(False).astype(bool)
    eligible = eligible.loc[strict & score.notna() & np.isfinite(score)].copy()
    if eligible.empty:
        raise ValueError("historical causal mapping has no finite strict-OOF rows")
    eligible["execution_decision_utc"] = pd.to_datetime(
        eligible["execution_decision_utc"], utc=True, errors="raise"
    )
    eligible["mapped_execution_ev"] = pd.to_numeric(
        eligible[SOURCE_SCORE_COLUMN], errors="raise"
    )
    labels_12h = _validate_label_values(labels_12h, horizon_hours=12)
    labels_24h = _validate_label_values(labels_24h, horizon_hours=24)
    label_columns = [
        column
        for column in labels_12h.columns
        if column not in IDENTITY
    ]
    base = eligible.drop(
        columns=[
            column
            for column in label_columns
            if column in eligible.columns
        ],
        errors="ignore",
    ).merge(
        labels_12h,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not base["_merge"].eq("both").all():
        raise ValueError("12h labels do not cover every finite historical OOF score")
    base = base.drop(columns="_merge")
    for column in (
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
    ):
        mapped = pd.to_numeric(eligible.set_index("candidate_id")[column], errors="raise")
        exact = pd.to_numeric(base.set_index("candidate_id")[column], errors="raise")
        if not np.allclose(
            mapped.reindex(exact.index).to_numpy(),
            exact.to_numpy(),
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(
                f"causal mapped panel's exact-policy {column} disagrees with 12h labels"
            )
    frozen = apply_global_admission(
        base, score_column="mapped_execution_ev", top_k_fraction=top_k_fraction
    )
    labels_24h = labels_24h.set_index("candidate_id").reindex(
        frozen["candidate_id"]
    )
    if labels_24h.index.hasnans or labels_24h[list(IDENTITY[1:])].isna().any().any():
        raise ValueError("24h labels do not cover every finite historical OOF score")
    labels_24h = labels_24h.reset_index()
    paired = pair_horizon_labels(frozen, labels_24h)
    paired["utc_month"] = paired["execution_decision_utc"].dt.strftime("%Y-%m")
    paired["exit_transition"] = (
        paired["execution_exit_reason__12h"].astype(str)
        + " -> "
        + paired["execution_exit_reason__24h"].astype(str)
    )
    paired["timeout_to_trailing"] = (
        paired["execution_exit_reason__12h"].astype(str).eq("timeout")
        & paired["execution_exit_reason__24h"].astype(str).eq("trailing")
    )
    paired["snx_style_late_recovery"] = (
        paired["timeout_to_trailing"]
        & paired["loss_to_win_12h_to_24h"]
    )
    return paired


def _cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {"full_population": pd.Series(True, index=frame.index)}
    masks.update({name: frame[column].astype(bool) for name, column in COHORT_FLAGS.items()})
    return masks


def recurrence_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Report late-recovery support by period, side, regime, and exit state."""

    dimensions: list[tuple[str, Sequence[str]]] = [
        ("overall", ()),
        ("month", ("utc_month",)),
        ("side", ("side_name",)),
        ("month_side", ("utc_month", "side_name")),
        ("policy_regime", ("policy_archetype__12h",)),
        ("exit_transition", ("exit_transition",)),
        ("month_side_exit", ("utc_month", "side_name", "exit_transition")),
    ]
    rows: list[dict[str, Any]] = []
    for cohort, mask in _cohort_masks(frame).items():
        selected = frame.loc[mask]
        for scope, keys in dimensions:
            groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
            for values, group in groups:
                values = values if isinstance(values, tuple) else (values,)
                delta = group["paired_delta_net_24h_minus_12h"]
                net_12 = group["execution_net_ev_12h__12h"]
                net_24 = group["execution_net_ev_12h__24h"]
                late = group["snx_style_late_recovery"]
                late_rows = group.loc[late]
                late_symbol_days = (
                    late_rows["__symbol__"].astype(str)
                    + "|"
                    + late_rows["execution_decision_utc"].dt.strftime("%Y-%m-%d")
                )
                row = {
                    "cohort": cohort,
                    "scope": scope,
                    "utc_month": None,
                    "side_name": None,
                    "policy_archetype__12h": None,
                    "exit_transition": None,
                    "rows": int(len(group)),
                    "mean_net_12h_bps": float(net_12.mean() * 10_000.0),
                    "mean_net_24h_bps": float(net_24.mean() * 10_000.0),
                    "mean_paired_delta_bps": float(delta.mean() * 10_000.0),
                    "median_paired_delta_bps": float(delta.median() * 10_000.0),
                    "24h_better_rows": int((delta > 0.0).sum()),
                    "24h_worse_rows": int((delta < 0.0).sum()),
                    "loss_to_win_rows": int(group["loss_to_win_12h_to_24h"].sum()),
                    "win_to_loss_rows": int(group["win_to_loss_12h_to_24h"].sum()),
                    "timeout_to_trailing_rows": int(group["timeout_to_trailing"].sum()),
                    "snx_style_late_recovery_rows": int(late.sum()),
                    "snx_style_late_recovery_unique_assets": int(
                        late_rows["__symbol__"].nunique()
                    ),
                    "snx_style_late_recovery_symbol_days": int(
                        late_symbol_days.nunique()
                    ),
                    "snx_style_late_recovery_rate": float(late.mean()),
                    "late_recovery_mean_delta_bps": float(
                        delta.loc[late].mean() * 10_000.0
                    )
                    if late.any()
                    else np.nan,
                }
                for key, value in zip(keys, values):
                    row[key] = value
                rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "scores",
        "score_provenance",
        "labels_12h",
        "manifest_12h",
        "labels_24h",
        "manifest_24h",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir = _resolve(args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_12h = _read_json(args.manifest_12h)
    manifest_24h = _read_json(args.manifest_24h)
    score_provenance = _read_json(args.score_provenance)
    _validate_historical_manifest(
        manifest_12h, args.labels_12h, expected_horizon_minutes=720
    )
    _validate_historical_manifest(
        manifest_24h, args.labels_24h, expected_horizon_minutes=1440
    )
    _validate_shared_label_lineage(manifest_12h, manifest_24h)
    mapped_output = score_provenance.get("outputs", {}).get("mapped", {})
    if (
        score_provenance.get("schema")
        != "causal_score_economics_conversion_mapping_v1"
        or int(
            score_provenance.get("population_audit", {}).get(
                "strict_mapped_oof_rows", -1
            )
        )
        != 114_096
        or score_provenance.get("causal_contract", {}).get("mapping")
        != "causal_recent_side_isotonic_ev"
        or score_provenance.get("causal_contract", {}).get(
            "exact_policy_target_remap"
        )
        is not True
    ):
        raise ValueError("historical mapping manifest does not prove exact-policy causal OOF scoring")
    if (
        _sha256(args.scores) != mapped_output.get("sha256")
        or args.scores.name != mapped_output.get("path")
    ):
        raise ValueError("historical causal mapped score is not bound by its manifest")
    policy_12h = _resolve(manifest_12h["source"]["policy"])
    policy_24h = _resolve(manifest_24h["source"]["policy"])
    if (
        policy_12h != policy_24h
        or manifest_12h["source"]["policy_sha256"]
        != manifest_24h["source"]["policy_sha256"]
        or _sha256(policy_12h) != manifest_12h["source"]["policy_sha256"]
    ):
        raise ValueError("12h and 24h signed policy files are not identical")
    components_12h = _policy_component_hashes(policy_12h)
    components_24h = _policy_component_hashes(policy_24h)
    if components_12h != components_24h:
        raise ValueError("12h and 24h policy core/strategy component hashes differ")
    scores = pd.read_parquet(args.scores)
    labels_12h = pd.read_parquet(args.labels_12h)
    labels_24h = pd.read_parquet(args.labels_24h)
    paired = build_frozen_historical_population(
        scores,
        labels_12h,
        labels_24h,
        top_k_fraction=args.top_k_fraction,
    )
    horizons = horizon_metrics(paired)
    deltas = paired_delta_metrics(paired)
    exits = exit_mix_metrics(paired)
    recurrence = recurrence_metrics(paired)
    late_events = paired.loc[paired["snx_style_late_recovery"]].copy()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "paired_population": args.output_dir / "paired_population.parquet",
        "late_recovery_events": args.output_dir / "late_recovery_events.parquet",
        "horizon_metrics": args.output_dir / "horizon_metrics.csv",
        "paired_delta_metrics": args.output_dir / "paired_delta_metrics.csv",
        "exit_mix_metrics": args.output_dir / "exit_mix_metrics.csv",
        "recurrence_metrics": args.output_dir / "recurrence_metrics.csv",
    }
    paired.to_parquet(outputs["paired_population"], index=False, compression="zstd")
    late_events.to_parquet(
        outputs["late_recovery_events"], index=False, compression="zstd"
    )
    horizons.to_csv(outputs["horizon_metrics"], index=False)
    deltas.to_csv(outputs["paired_delta_metrics"], index=False)
    exits.to_csv(outputs["exit_mix_metrics"], index=False)
    recurrence.to_csv(outputs["recurrence_metrics"], index=False)
    headline = recurrence.loc[recurrence["scope"].eq("overall")].to_dict("records")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "research_only": True,
                "promotion_eligible": False,
                "headline": headline,
                "late_recovery_event_rows": int(len(late_events)),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "research_only_retrospective_nonpromotable",
        "promotion_eligible": False,
        "selection_contract": {
            "score": SOURCE_SCORE_COLUMN,
            "eligibility": (
                "finite mapped_direct_net and "
                "causal_recent_side_isotonic_ev__is_oof=true"
            ),
            "finite_oof_rows": int(len(paired)),
            "score_role": "strict-OOF causal 21-day side-isotonic exact-policy execution-EV map",
            "scope": "one pooled global book across the compatible historical period",
            "top_k_fraction": float(args.top_k_fraction),
            "cohorts_frozen_before_24h_outcome_join": list(_cohort_masks(paired)),
            "outcome_based_reselection": False,
            "per_timestamp_quota": False,
        },
        "paired_contract": {
            "identity": list(IDENTITY),
            "decision_min_utc": paired["execution_decision_utc"].min(),
            "decision_max_utc": paired["execution_decision_utc"].max(),
            "months": sorted(paired["utc_month"].unique()),
            "horizons_minutes": {"12h": 720, "24h": 1440},
            "shared_source_lineage_verified": True,
            "mapped_score_identity_joins_every_strict_oof_row_to_labels": True,
            "mapped_exact_policy_outcomes_match_12h_labels_atol_1e_10": True,
            "cost_reapplied": False,
            "policy_component_hashes_verified_equal": True,
        },
        "late_recovery_definition": (
            "12h exit=timeout AND 24h exit=trailing AND "
            "12h net<=0 AND 24h net>0"
        ),
        "inputs": {
            "causal_mapped_oof": _artifact_record(args.scores),
            "mapping_manifest": _artifact_record(args.score_provenance),
            "labels_12h": _artifact_record(args.labels_12h),
            "manifest_12h": _artifact_record(args.manifest_12h),
            "labels_24h": _artifact_record(args.labels_24h),
            "manifest_24h": _artifact_record(args.manifest_24h),
            "candidate_source": {
                "path": manifest_12h["source"]["candidates"],
                "sha256": manifest_12h["source"]["candidates_sha256"],
            },
            "context_source": {
                "path": manifest_12h["source"]["context"],
                "sha256": manifest_12h["source"]["context_sha256"],
            },
            "path_target_source": {
                "path": manifest_12h["source"]["path_targets"],
                "sha256": manifest_12h["source"]["path_targets_sha256"],
            },
            "signed_policy": {
                "path": manifest_12h["source"]["policy"],
                "sha256": manifest_12h["source"]["policy_sha256"],
                "component_hashes": components_12h,
            },
        },
        "outputs": {key: _artifact_record(path) for key, path in outputs.items()}
        | {"summary": _artifact_record(summary_path)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORE)
    parser.add_argument(
        "--score-provenance", type=Path, default=DEFAULT_SCORE_PROVENANCE
    )
    parser.add_argument(
        "--labels-12h",
        type=Path,
        default=DEFAULT_12H_ROOT / "execution_ev_policy_labels.parquet",
    )
    parser.add_argument(
        "--manifest-12h", type=Path, default=DEFAULT_12H_ROOT / "manifest.json"
    )
    parser.add_argument(
        "--labels-24h",
        type=Path,
        default=DEFAULT_24H_ROOT / "execution_ev_policy_labels.parquet",
    )
    parser.add_argument(
        "--manifest-24h", type=Path, default=DEFAULT_24H_ROOT / "manifest.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(manifest["paired_contract"], indent=2, default=str))


if __name__ == "__main__":
    main()
