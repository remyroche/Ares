#!/usr/bin/env python3
"""Paired 12h-vs-24h exact-policy timeout ablation on frozen July cohorts."""

from __future__ import annotations

import argparse
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
    DEFAULT_POLICY_CONFIG,
    DEFAULT_ROOT,
    IDENTITY,
    LABEL_SCHEMA,
    _artifact_record,
    _bound_file,
    _read_json,
    _resolve,
    _sha256,
    _validate_identity,
    load_joined_population,
    portfolio_replays,
)

SCHEMA = "execution_ev_exact_policy_timeout_ablation_v1"
LABEL_VALUE_COLUMNS = (
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
    "execution_exit_reason",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_entry_price",
    "execution_exit_price",
    "execution_expected_spread_bps",
    "execution_entry_half_spread_bps",
    "execution_exit_half_spread_bps",
    "execution_label_end_utc",
    "execution_label_available_at",
    "policy_archetype",
    "execution_geometry_key",
    "execution_geometry_source",
)
HORIZONS = ("12h", "24h")


def _validate_label_manifest(
    manifest: Mapping[str, Any],
    labels_path: Path,
    *,
    expected_horizon_minutes: int,
) -> None:
    if manifest.get("schema") != LABEL_SCHEMA:
        raise ValueError("unexpected exact-policy label manifest schema")
    coverage = manifest.get("coverage", {}).get("overall", {})
    if (
        float(coverage.get("coverage", -1.0)) != 1.0
        or int(coverage.get("missing", -1)) != 0
        or int(coverage.get("rows", -1)) != 5760
    ):
        raise ValueError("paired timeout labels require complete 5,760-row coverage")
    contract = manifest.get("exit_policy_contract", {})
    if (
        contract.get("replay_timeframe") != "1m"
        or int(contract.get("horizon_minutes", -1)) != expected_horizon_minutes
    ):
        raise ValueError("exact-policy label horizon does not match the requested arm")
    accounting = manifest.get("accounting", {})
    if (
        accounting.get("candidate_local_exit_replay") is not True
        or accounting.get("portfolio_concurrency_applied") is not False
        or accounting.get("net_return") != "gross return minus fee return"
        or "spread drag is embedded in gross return"
        not in str(accounting.get("cost_return", ""))
    ):
        raise ValueError("exact-policy label accounting is not single-charge compatible")
    _bound_file(
        manifest, manifest["output"], labels_path, role=f"{expected_horizon_minutes}m labels"
    )


def _validate_shared_label_lineage(
    manifest_12h: Mapping[str, Any], manifest_24h: Mapping[str, Any]
) -> None:
    for key in (
        "candidates_sha256",
        "context_sha256",
        "path_targets_sha256",
        "policy_sha256",
    ):
        left = manifest_12h.get("source", {}).get(key)
        right = manifest_24h.get("source", {}).get(key)
        if not left or left != right:
            raise ValueError(f"12h/24h source lineage differs for {key}")
    left_contract = manifest_12h["exit_policy_contract"]
    right_contract = manifest_24h["exit_policy_contract"]
    for key in (
        "geometry_scope",
        "policy_pathway_id",
        "replay_timeframe",
        "simulator",
        "source_policy_sha256",
        "trailing_activation_curve",
    ):
        if left_contract.get(key) != right_contract.get(key):
            raise ValueError(f"12h/24h exit-policy contract differs for {key}")
    if manifest_12h.get("accounting") != manifest_24h.get("accounting"):
        raise ValueError("12h/24h accounting contracts are not identical")


def _validate_label_values(frame: pd.DataFrame, *, horizon_hours: int) -> pd.DataFrame:
    work = _validate_identity(frame, role=f"{horizon_hours}h labels")
    missing = sorted(set(LABEL_VALUE_COLUMNS).difference(work.columns))
    if missing:
        raise ValueError(f"{horizon_hours}h labels missing fields: {missing}")
    for column in (
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_label_available_at",
    ):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    expected_end = work["execution_decision_utc"] + pd.Timedelta(hours=horizon_hours)
    if not work["execution_label_end_utc"].eq(expected_end).all():
        raise ValueError(f"{horizon_hours}h label end is not decision plus full horizon")
    if (work["execution_label_available_at"] < work["execution_label_end_utc"]).any():
        raise ValueError(f"{horizon_hours}h labels are available before resolution")
    values = work[
        ["execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{horizon_hours}h economics contain non-finite values")
    if not np.allclose(
        values["execution_net_ev_12h"],
        values["execution_gross_ev_12h"] - values["execution_cost_return"],
        atol=1e-10,
        rtol=1e-8,
    ):
        raise ValueError(f"{horizon_hours}h net is not gross minus stored fee")
    return work


def pair_horizon_labels(
    frame_12h: pd.DataFrame, labels_24h: pd.DataFrame
) -> pd.DataFrame:
    """Pair exact outcomes without changing score-derived cohort membership."""

    left = _validate_label_values(frame_12h, horizon_hours=12)
    right = _validate_label_values(labels_24h, horizon_hours=24)
    if set(left["candidate_id"]) != set(right["candidate_id"]) or len(left) != len(right):
        raise ValueError("12h and 24h candidate populations are not identical")
    score_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "mapped_execution_ev",
        "global_rank",
        *COHORT_FLAGS.values(),
    ]
    missing_scores = sorted(set(score_columns).difference(left.columns))
    if missing_scores:
        raise ValueError(f"frozen scored cohort fields missing: {missing_scores}")
    base = left.loc[:, score_columns].copy()
    output = base
    for horizon, source in (("12h", left), ("24h", right)):
        local = source.loc[:, [*IDENTITY, *LABEL_VALUE_COLUMNS]].copy()
        local = local.rename(
            columns={column: f"{column}__{horizon}" for column in LABEL_VALUE_COLUMNS}
        )
        output = output.merge(local, on=list(IDENTITY), how="left", validate="one_to_one")
    decision_24 = pd.to_datetime(
        right.set_index("candidate_id")["execution_decision_utc"].reindex(
            output["candidate_id"]
        ),
        utc=True,
        errors="raise",
    ).reset_index(drop=True)
    if not decision_24.eq(output["execution_decision_utc"].reset_index(drop=True)).all():
        raise ValueError("12h and 24h decisions differ")
    net_12 = output["execution_net_ev_12h__12h"]
    net_24 = output["execution_net_ev_12h__24h"]
    output["paired_delta_net_24h_minus_12h"] = net_24 - net_12
    output["paired_delta_gross_24h_minus_12h"] = (
        output["execution_gross_ev_12h__24h"]
        - output["execution_gross_ev_12h__12h"]
    )
    output["paired_delta_cost_24h_minus_12h"] = (
        output["execution_cost_return__24h"]
        - output["execution_cost_return__12h"]
    )
    output["paired_delta_holding_hours_24h_minus_12h"] = (
        output["execution_exit_hour__24h"] - output["execution_exit_hour__12h"]
    )
    output["loss_to_win_12h_to_24h"] = (net_12 <= 0.0) & (net_24 > 0.0)
    output["win_to_loss_12h_to_24h"] = (net_12 > 0.0) & (net_24 <= 0.0)
    output["positive_both_horizons"] = (net_12 > 0.0) & (net_24 > 0.0)
    output["negative_both_horizons"] = (net_12 <= 0.0) & (net_24 <= 0.0)
    output["exit_reason_changed"] = (
        output["execution_exit_reason__12h"].astype(str)
        != output["execution_exit_reason__24h"].astype(str)
    )
    output["utc_date"] = output["execution_decision_utc"].dt.strftime("%Y-%m-%d")
    return output


def _cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {"full_population": pd.Series(True, index=frame.index)}
    masks.update({name: frame[column].astype(bool) for name, column in COHORT_FLAGS.items()})
    return masks


def _groups(
    frame: pd.DataFrame,
) -> list[tuple[str, Sequence[str]]]:
    del frame
    return [
        ("overall", ()),
        ("side", ("side_name",)),
        ("day", ("utc_date",)),
        ("day_side", ("utc_date", "side_name")),
    ]


def horizon_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, mask in _cohort_masks(frame).items():
        selected = frame.loc[mask]
        for scope, keys in _groups(selected):
            groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
            for values, group in groups:
                values = values if isinstance(values, tuple) else (values,)
                for horizon in HORIZONS:
                    net = group[f"execution_net_ev_12h__{horizon}"]
                    gross = group[f"execution_gross_ev_12h__{horizon}"]
                    cost = group[f"execution_cost_return__{horizon}"]
                    row = {
                        "cohort": cohort,
                        "scope": scope,
                        "utc_date": None,
                        "side_name": None,
                        "horizon": horizon,
                        "rows": int(len(group)),
                        "mean_net_bps": float(net.mean() * 10_000.0),
                        "median_net_bps": float(net.median() * 10_000.0),
                        "positive_rate": float((net > 0.0).mean()),
                        "mean_gross_bps": float(gross.mean() * 10_000.0),
                        "mean_stored_fee_bps": float(cost.mean() * 10_000.0),
                        "mean_holding_hours": float(
                            group[f"execution_exit_hour__{horizon}"].mean()
                        ),
                        "median_holding_hours": float(
                            group[f"execution_exit_hour__{horizon}"].median()
                        ),
                    }
                    for key, value in zip(keys, values):
                        row[key] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def paired_delta_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, mask in _cohort_masks(frame).items():
        selected = frame.loc[mask]
        for scope, keys in _groups(selected):
            groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
            for values, group in groups:
                values = values if isinstance(values, tuple) else (values,)
                delta = group["paired_delta_net_24h_minus_12h"]
                row = {
                    "cohort": cohort,
                    "scope": scope,
                    "utc_date": None,
                    "side_name": None,
                    "rows": int(len(group)),
                    "mean_paired_delta_net_bps": float(delta.mean() * 10_000.0),
                    "median_paired_delta_net_bps": float(delta.median() * 10_000.0),
                    "mean_paired_delta_gross_bps": float(
                        group["paired_delta_gross_24h_minus_12h"].mean() * 10_000.0
                    ),
                    "mean_paired_delta_cost_bps": float(
                        group["paired_delta_cost_24h_minus_12h"].mean() * 10_000.0
                    ),
                    "mean_paired_delta_holding_hours": float(
                        group["paired_delta_holding_hours_24h_minus_12h"].mean()
                    ),
                    "24h_better_rows": int((delta > 0.0).sum()),
                    "24h_equal_rows": int(np.isclose(delta, 0.0, atol=1e-15).sum()),
                    "24h_worse_rows": int((delta < 0.0).sum()),
                    "loss_to_win_rows": int(group["loss_to_win_12h_to_24h"].sum()),
                    "win_to_loss_rows": int(group["win_to_loss_12h_to_24h"].sum()),
                    "exit_reason_changed_rows": int(group["exit_reason_changed"].sum()),
                }
                for key, value in zip(keys, values):
                    row[key] = value
                rows.append(row)
    return pd.DataFrame(rows)


def exit_mix_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, mask in _cohort_masks(frame).items():
        selected = frame.loc[mask]
        for scope, keys in _groups(selected):
            groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
            for values, group in groups:
                values = values if isinstance(values, tuple) else (values,)
                for horizon in HORIZONS:
                    reason_column = f"execution_exit_reason__{horizon}"
                    for reason, reason_group in group.groupby(reason_column, sort=True):
                        row = {
                            "cohort": cohort,
                            "scope": scope,
                            "utc_date": None,
                            "side_name": None,
                            "horizon": horizon,
                            "exit_reason": str(reason),
                            "rows": int(len(reason_group)),
                            "share": float(len(reason_group) / max(len(group), 1)),
                            "mean_net_bps": float(
                                reason_group[f"execution_net_ev_12h__{horizon}"].mean()
                                * 10_000.0
                            ),
                            "mean_holding_hours": float(
                                reason_group[f"execution_exit_hour__{horizon}"].mean()
                            ),
                        }
                        for key, value in zip(keys, values):
                            row[key] = value
                        rows.append(row)
    return pd.DataFrame(rows)


def horizon_frame(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon {horizon}")
    output = frame.copy()
    for column in LABEL_VALUE_COLUMNS:
        output[column] = output[f"{column}__{horizon}"]
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "scored",
        "scored_manifest",
        "labels_12h",
        "manifest_12h",
        "labels_24h",
        "manifest_24h",
        "preentry_manifest",
        "policy",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir = _resolve(args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_12h = _read_json(args.manifest_12h)
    manifest_24h = _read_json(args.manifest_24h)
    _validate_label_manifest(
        manifest_12h, args.labels_12h, expected_horizon_minutes=720
    )
    _validate_label_manifest(
        manifest_24h, args.labels_24h, expected_horizon_minutes=1440
    )
    _validate_shared_label_lineage(manifest_12h, manifest_24h)
    frame_12h, scored_manifest, _, packb_path = load_joined_population(
        scored_path=args.scored,
        scored_manifest_path=args.scored_manifest,
        labels_path=args.labels_12h,
        labels_manifest_path=args.manifest_12h,
        preentry_manifest_path=args.preentry_manifest,
        policy_path=args.policy,
        top_k_fraction=args.top_k_fraction,
    )
    labels_24h = pd.read_parquet(args.labels_24h)
    paired = pair_horizon_labels(frame_12h, labels_24h)
    horizons = horizon_metrics(paired)
    deltas = paired_delta_metrics(paired)
    exits = exit_mix_metrics(paired)
    portfolio_parts = []
    decision_parts = []
    equity_parts = []
    side_parts = []
    portfolio_contracts = {}
    for horizon in HORIZONS:
        summary, decisions, equity, side, contract = portfolio_replays(
            horizon_frame(paired, horizon),
            policy_path=args.policy,
            initial_wallet=args.initial_wallet,
        )
        for table in (summary, decisions, equity, side):
            table.insert(0, "horizon", horizon)
        portfolio_parts.append(summary)
        decision_parts.append(decisions)
        equity_parts.append(equity)
        side_parts.append(side)
        portfolio_contracts[horizon] = contract
    portfolio = pd.concat(portfolio_parts, ignore_index=True)
    decisions = pd.concat(decision_parts, ignore_index=True)
    equity = pd.concat(equity_parts, ignore_index=True)
    portfolio_side = pd.concat(side_parts, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "paired_population": args.output_dir / "paired_population.parquet",
        "horizon_metrics": args.output_dir / "horizon_metrics.csv",
        "paired_delta_metrics": args.output_dir / "paired_delta_metrics.csv",
        "exit_mix_metrics": args.output_dir / "exit_mix_metrics.csv",
        "portfolio_summary": args.output_dir / "portfolio_summary.csv",
        "portfolio_side_metrics": args.output_dir / "portfolio_side_metrics.csv",
        "portfolio_decisions": args.output_dir / "portfolio_decisions.parquet",
        "portfolio_equity": args.output_dir / "portfolio_equity.parquet",
    }
    paired.to_parquet(outputs["paired_population"], index=False, compression="zstd")
    horizons.to_csv(outputs["horizon_metrics"], index=False)
    deltas.to_csv(outputs["paired_delta_metrics"], index=False)
    exits.to_csv(outputs["exit_mix_metrics"], index=False)
    portfolio.to_csv(outputs["portfolio_summary"], index=False)
    portfolio_side.to_csv(outputs["portfolio_side_metrics"], index=False)
    decisions.to_parquet(outputs["portfolio_decisions"], index=False, compression="zstd")
    equity.to_parquet(outputs["portfolio_equity"], index=False, compression="zstd")
    headline = deltas.loc[deltas["scope"].eq("overall")].to_dict("records")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "research_only": True,
                "promotion_eligible": False,
                "paired_headline": headline,
                "portfolio": portfolio.to_dict("records"),
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
            "score": "mapped_execution_ev",
            "mapping": scored_manifest["contract"]["mapping"],
            "scope": "one pooled global book across all timestamps and sides",
            "top_k_fraction": float(args.top_k_fraction),
            "cohorts_frozen_before_outcome_join": list(_cohort_masks(paired)),
            "outcome_based_reselection": False,
            "per_timestamp_quota": False,
        },
        "paired_contract": {
            "rows": int(len(paired)),
            "identity": list(IDENTITY),
            "horizons_minutes": {"12h": 720, "24h": 1440},
            "only_intended_difference": "timeout horizon",
            "shared_source_lineage_verified": True,
            "cost_reapplied": False,
        },
        "portfolio_contracts": portfolio_contracts,
        "inputs": {
            "scored": _artifact_record(args.scored),
            "scored_manifest": _artifact_record(args.scored_manifest),
            "labels_12h": _artifact_record(args.labels_12h),
            "manifest_12h": _artifact_record(args.manifest_12h),
            "labels_24h": _artifact_record(args.labels_24h),
            "manifest_24h": _artifact_record(args.manifest_24h),
            "preentry_manifest": _artifact_record(args.preentry_manifest),
            "packb_context": _artifact_record(packb_path),
            "signed_simple_policy": _artifact_record(args.policy),
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
    parser.add_argument(
        "--scored", type=Path, default=DEFAULT_ROOT / "scored/scored_population.parquet"
    )
    parser.add_argument(
        "--scored-manifest", type=Path, default=DEFAULT_ROOT / "scored/manifest.json"
    )
    parser.add_argument(
        "--labels-12h",
        type=Path,
        default=DEFAULT_ROOT / "labels_12h/execution_ev_policy_labels.parquet",
    )
    parser.add_argument(
        "--manifest-12h", type=Path, default=DEFAULT_ROOT / "labels_12h/manifest.json"
    )
    parser.add_argument(
        "--labels-24h",
        type=Path,
        default=DEFAULT_ROOT / "labels_24h/execution_ev_policy_labels.parquet",
    )
    parser.add_argument(
        "--manifest-24h", type=Path, default=DEFAULT_ROOT / "labels_24h/manifest.json"
    )
    parser.add_argument(
        "--preentry-manifest",
        type=Path,
        default=DEFAULT_ROOT / "preentry/manifest.json",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(manifest["paired_contract"], indent=2))


if __name__ == "__main__":
    main()
