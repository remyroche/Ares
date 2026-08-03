#!/usr/bin/env python3
"""Report exact July execution-EV economics and constrained portfolio replay.

This is a research-only retrospective reporter.  It joins the frozen v2
retrospective scores to their exact 1-minute, 12-hour simple-policy labels,
reconstructs the one pooled global top-k book, reports diagnostic model-head
metrics, and replays each fixed admission book through the portfolio
constraints embedded in the *same* signed simple-policy artifact used for the
labels.  Stored net returns are consumed directly; no costs are re-applied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from scripts.score_execution_ev_forward_population import (  # noqa: E402
    apply_global_admission,
)

# Preserve the authoritative retrospective scorer's identity order because its
# manifest hash is order-sensitive.
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
SCORE_SCHEMA = "execution_ev_retrospective_scored_population_v1"
LABEL_SCHEMA = "execution_ev_deployed_policy_1m_labels_v1"
REPORT_SCHEMA = "execution_ev_july_exact_economics_report_v1"
DEFAULT_ROOT = Path(
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
)
DEFAULT_POLICY_CONFIG = Path(
    "data_perp/artifacts/"
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_"
    "20260717_v2/simple_policy_optimiser/deployment/best_policy_params_perps.json"
)
COHORT_FLAGS = {
    "global_top10": "global_top10_capacity_member",
    "admitted_gt_0bps": "globally_admitted_floor_0bps",
    "admitted_gt_25bps": "globally_admitted_floor_25bps",
    "admitted_gt_50bps": "globally_admitted_floor_50bps",
}
BASE_SCORE_SPECS = (
    ("base_score", "base_oof_score", "execution_net_ev_12h", "continuous"),
    ("base_alpha_ev", "base_alpha_ev", "execution_net_ev_12h", "continuous"),
    (
        "residual_enhanced_alpha",
        "existing_alpha_ev",
        "execution_net_ev_12h",
        "continuous",
    ),
    ("residual_delta", "residual_delta_ev", "execution_net_ev_12h", "continuous"),
    ("direct_execution_ev", "final_direct_net_raw", "execution_net_ev_12h", "continuous"),
    (
        "capture_probability",
        "final_capture_probability",
        "positive_net",
        "binary",
    ),
    (
        "mapped_execution_ev",
        "mapped_execution_ev",
        "execution_net_ev_12h",
        "continuous",
    ),
    (
        "clean_favorable_probability",
        "oof_clean_favorable_probability",
        "positive_net",
        "binary",
    ),
    (
        "peak_mfe_aux",
        "pred_peak_MFE_12h_ATR",
        "execution_mfe_return_12h",
        "continuous",
    ),
    (
        "catboost_usable_path_probability",
        "catboost_usable_path_probability",
        "positive_net",
        "binary",
    ),
    (
        "catboost_adverse_path_probability",
        "catboost_adverse_path_probability",
        "negative_net",
        "binary",
    ),
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _bound_file(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    supplied_path: Path,
    *,
    role: str,
) -> None:
    del manifest
    expected = str(record.get("sha256", ""))
    if not expected or _sha256(supplied_path) != expected:
        raise ValueError(f"{role} hash does not match its manifest")
    manifest_path = record.get("path")
    if manifest_path is not None and _resolve(str(manifest_path)).resolve() != supplied_path.resolve():
        raise ValueError(f"{role} path does not match its manifest")


def _identity_hash(frame: pd.DataFrame) -> str:
    ordered = (
        frame.loc[:, list(IDENTITY)]
        .astype(str)
        .sort_values(list(IDENTITY), kind="stable")
    )
    payload = "\n".join(
        "\x1f".join(row) for row in ordered.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_identity(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{role} identity columns missing: {missing}")
    if frame.duplicated(list(IDENTITY)).any() or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{role} identity is not globally one-to-one")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{role} contains a non-canonical side")
    return work


def _validate_manifests(
    *,
    scored_path: Path,
    scored_manifest_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    preentry_manifest_path: Path,
    policy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    scored_manifest = _read_json(scored_manifest_path)
    labels_manifest = _read_json(labels_manifest_path)
    preentry_manifest = _read_json(preentry_manifest_path)
    if scored_manifest.get("schema") != SCORE_SCHEMA:
        raise ValueError("unexpected scored-population manifest schema")
    if scored_manifest.get("retrospective") is not True:
        raise ValueError("scored population is not marked retrospective")
    if scored_manifest.get("promotion_eligible") is not False:
        raise ValueError("research reporter refuses a promotion-eligible score artifact")
    contract = scored_manifest.get("contract", {})
    if contract.get("ranking") != (
        "one pooled global top10 across timestamps and sides after causal mapping"
    ):
        raise ValueError("score manifest does not bind pooled global post-map ranking")
    _bound_file(
        scored_manifest,
        scored_manifest["outputs"]["scored_population"],
        scored_path,
        role="scored population",
    )
    if labels_manifest.get("schema") != LABEL_SCHEMA:
        raise ValueError("unexpected exact-label manifest schema")
    coverage = labels_manifest.get("coverage", {}).get("overall", {})
    if (
        float(coverage.get("coverage", -1.0)) != 1.0
        or int(coverage.get("missing", -1)) != 0
    ):
        raise ValueError("exact labels do not have complete population coverage")
    exit_contract = labels_manifest.get("exit_policy_contract", {})
    if (
        exit_contract.get("replay_timeframe") != "1m"
        or int(exit_contract.get("horizon_minutes", -1)) != 720
    ):
        raise ValueError("labels are not exact 1-minute 12-hour policy labels")
    accounting = labels_manifest.get("accounting", {})
    if (
        accounting.get("candidate_local_exit_replay") is not True
        or accounting.get("portfolio_concurrency_applied") is not False
        or accounting.get("net_return") != "gross return minus fee return"
        or "spread drag is embedded in gross return"
        not in str(accounting.get("cost_return", ""))
    ):
        raise ValueError("label accounting does not prove single-charge cost semantics")
    _bound_file(
        labels_manifest,
        labels_manifest["output"],
        labels_path,
        role="exact labels",
    )
    expected_policy = str(labels_manifest.get("source", {}).get("policy_sha256", ""))
    if not expected_policy or _sha256(policy_path) != expected_policy:
        raise ValueError("portfolio policy is not the exact policy bound to the labels")
    if preentry_manifest.get("schema") != "execution_ev_forward_preentry_v1":
        raise ValueError("unexpected preentry manifest schema")
    packb_record = preentry_manifest.get("inputs", {}).get("packb_context", {})
    packb_path = _resolve(str(packb_record.get("path", "")))
    if not packb_path.is_file() or _sha256(packb_path) != packb_record.get("sha256"):
        raise ValueError("PackB context does not match the preentry lineage")
    return scored_manifest, labels_manifest, preentry_manifest, packb_path


def load_joined_population(
    *,
    scored_path: Path,
    scored_manifest_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    preentry_manifest_path: Path,
    policy_path: Path,
    top_k_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], Path]:
    scored_manifest, labels_manifest, _, packb_path = _validate_manifests(
        scored_path=scored_path,
        scored_manifest_path=scored_manifest_path,
        labels_path=labels_path,
        labels_manifest_path=labels_manifest_path,
        preentry_manifest_path=preentry_manifest_path,
        policy_path=policy_path,
    )
    scored = _validate_identity(pd.read_parquet(scored_path), role="scored population")
    labels = _validate_identity(pd.read_parquet(labels_path), role="exact labels")
    if len(scored) != int(scored_manifest.get("rows", -1)):
        raise ValueError("scored row count does not match its manifest")
    if _identity_hash(scored) != scored_manifest.get("candidate_identity_sha256"):
        raise ValueError("scored identity hash does not match its manifest")
    if len(labels) != int(labels_manifest["output"].get("rows", -1)):
        raise ValueError("label row count does not match its manifest")
    label_columns = [column for column in labels.columns if column not in IDENTITY]
    joined = scored.merge(
        labels,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        suffixes=("", "__label"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or len(joined) != len(scored):
        raise ValueError("exact labels do not cover every scored candidate")
    joined = joined.drop(columns="_merge")
    for column in ("execution_decision_utc", "execution_label_end_utc", "execution_label_available_at"):
        joined[column] = pd.to_datetime(joined[column], utc=True, errors="raise")
    if "execution_decision_utc__label" in joined:
        label_decision = pd.to_datetime(
            joined.pop("execution_decision_utc__label"), utc=True, errors="raise"
        )
        if not label_decision.eq(joined["execution_decision_utc"]).all():
            raise ValueError("score and exact-label decisions do not match")
    if (
        (joined["execution_label_end_utc"] <= joined["execution_decision_utc"]).any()
        or (joined["execution_label_available_at"] < joined["execution_label_end_utc"]).any()
    ):
        raise ValueError("exact-label timing is not decision-safe")
    numeric_cost = joined[
        ["execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric_cost.to_numpy(dtype=float)).all():
        raise ValueError("exact policy economics contain non-finite values")
    if not np.allclose(
        numeric_cost["execution_net_ev_12h"],
        numeric_cost["execution_gross_ev_12h"] - numeric_cost["execution_cost_return"],
        atol=1e-10,
        rtol=1e-8,
    ):
        raise ValueError("net return is not exactly gross return minus stored fee return")
    if (numeric_cost["execution_cost_return"] < 0.0).any():
        raise ValueError("stored fee return cannot be negative")

    packb_columns = [
        *IDENTITY,
        "base_alpha_ev",
        "residual_delta_ev",
        "base_prediction",
        "prediction",
    ]
    packb = _validate_identity(
        pd.read_parquet(packb_path, columns=packb_columns), role="PackB context"
    )
    joined = joined.merge(packb, on=list(IDENTITY), how="left", validate="one_to_one")
    if joined[["base_alpha_ev", "residual_delta_ev"]].isna().any().any():
        raise ValueError("PackB base/residual diagnostics do not cover the score population")

    reconstructed = apply_global_admission(
        joined.drop(
            columns=[
                "global_top10_capacity_member",
                "globally_admitted_floor_0bps",
                "globally_admitted_floor_25bps",
                "globally_admitted_floor_50bps",
                "globally_admitted",
                "global_rank",
            ],
            errors="ignore",
        ),
        top_k_fraction=float(top_k_fraction),
    )
    for column in (
        "global_top10_capacity_member",
        "globally_admitted_floor_0bps",
        "globally_admitted_floor_25bps",
        "globally_admitted_floor_50bps",
        "globally_admitted",
        "global_rank",
    ):
        if column not in scored:
            raise ValueError(f"persisted scored population lacks {column}")
        expected = scored.set_index("candidate_id")[column].reindex(
            reconstructed["candidate_id"]
        )
        actual = reconstructed[column]
        if column == "global_rank":
            equal = np.array_equal(
                pd.to_numeric(expected, errors="raise").to_numpy(dtype=np.int64),
                pd.to_numeric(actual, errors="raise").to_numpy(dtype=np.int64),
            )
        else:
            equal = np.array_equal(
                expected.astype(bool).to_numpy(), actual.astype(bool).to_numpy()
            )
        if not equal:
            raise ValueError(f"persisted global admission disagrees with reconstruction: {column}")
    joined = reconstructed
    joined["utc_date"] = joined["execution_decision_utc"].dt.strftime("%Y-%m-%d")
    joined["positive_net"] = joined["execution_net_ev_12h"] > 0.0
    joined["negative_net"] = joined["execution_net_ev_12h"] < 0.0
    joined["catboost_usable_path_probability"] = joined[
        ["catboost_p_2", "catboost_p_3", "catboost_p_4", "catboost_p_5"]
    ].sum(axis=1)
    joined["catboost_adverse_path_probability"] = joined[
        ["catboost_p_0", "catboost_p_1", "catboost_p_6"]
    ].sum(axis=1)
    return joined, scored_manifest, labels_manifest, packb_path


def _economics(group: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(group["execution_net_ev_12h"], errors="raise")
    gross = pd.to_numeric(group["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(group["execution_cost_return"], errors="raise")
    return {
        "rows": int(len(group)),
        "mean_net_bps": float(net.mean() * 10_000.0) if len(group) else np.nan,
        "median_net_bps": float(net.median() * 10_000.0) if len(group) else np.nan,
        "net_sum": float(net.sum()),
        "positive_net_precision": float((net > 0.0).mean()) if len(group) else np.nan,
        "mean_gross_bps": float(gross.mean() * 10_000.0) if len(group) else np.nan,
        "mean_stored_fee_bps": float(cost.mean() * 10_000.0) if len(group) else np.nan,
        "mean_mfe_bps": float(group["execution_mfe_return_12h"].mean() * 10_000.0)
        if len(group)
        else np.nan,
        "mean_mae_bps": float(group["execution_mae_return_12h"].mean() * 10_000.0)
        if len(group)
        else np.nan,
    }


def cohort_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cohort_masks = {"full_population": np.ones(len(frame), dtype=bool)}
    cohort_masks.update(
        {name: frame[column].astype(bool).to_numpy() for name, column in COHORT_FLAGS.items()}
    )
    scopes: list[tuple[str, Sequence[str]]] = [
        ("overall", ()),
        ("side", ("side_name",)),
        ("day", ("utc_date",)),
        ("day_side", ("utc_date", "side_name")),
    ]
    for cohort, mask in cohort_masks.items():
        selected = frame.loc[mask]
        for scope, keys in scopes:
            groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
            for values, group in groups:
                values = values if isinstance(values, tuple) else (values,)
                row = {
                    "cohort": cohort,
                    "scope": scope,
                    "utc_date": None,
                    "side_name": None,
                    **_economics(group),
                }
                for key, value in zip(keys, values):
                    row[key] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _binary_metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _diagnostic_row(
    group: pd.DataFrame,
    *,
    head: str,
    score_column: str,
    target_column: str,
    target_kind: str,
) -> dict[str, Any]:
    score = pd.to_numeric(group[score_column], errors="coerce")
    target = pd.to_numeric(group[target_column], errors="coerce")
    valid = score.notna() & target.notna() & np.isfinite(score) & np.isfinite(target)
    score = score.loc[valid]
    target = target.loc[valid]
    rank_ic = float(score.corr(target, method="spearman")) if len(score) >= 3 else np.nan
    binary_target = target.astype(bool).to_numpy() if target_kind == "binary" else (
        pd.to_numeric(group.loc[valid, "execution_net_ev_12h"], errors="raise")
        .gt(0.0)
        .to_numpy()
    )
    auc, average_precision = _binary_metrics(binary_target, score.to_numpy(dtype=float))
    count = max(1, int(math.ceil(0.10 * len(score)))) if len(score) else 0
    order = np.lexsort(
        (
            group.loc[valid, "candidate_id"].astype(str).to_numpy(),
            -score.to_numpy(dtype=float),
        )
    )
    chosen_index = score.index[order[:count]]
    chosen_net = pd.to_numeric(
        group.loc[chosen_index, "execution_net_ev_12h"], errors="raise"
    )
    return {
        "head": head,
        "score_column": score_column,
        "target_column": target_column,
        "binary_target_definition": (
            target_column if target_kind == "binary" else "execution_net_ev_12h > 0"
        ),
        "rows": int(len(score)),
        "rank_ic": rank_ic,
        "binary_target_auc": auc,
        "binary_target_average_precision": average_precision,
        "score_global_top10_rows": int(count),
        "score_global_top10_positive_precision": float((chosen_net > 0.0).mean())
        if count
        else np.nan,
        "score_global_top10_mean_net_bps": float(chosen_net.mean() * 10_000.0)
        if count
        else np.nan,
    }


def score_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("overall", "all", frame)]
    scopes.extend(("side", str(side), group) for side, group in frame.groupby("side_name"))
    scopes.extend(("day", str(day), group) for day, group in frame.groupby("utc_date"))
    for scope, scope_value, group in scopes:
        for head, score_column, target_column, target_kind in BASE_SCORE_SPECS:
            if score_column not in group or target_column not in group:
                continue
            rows.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    **_diagnostic_row(
                        group,
                        head=head,
                        score_column=score_column,
                        target_column=target_column,
                        target_kind=target_kind,
                    ),
                }
            )
    return pd.DataFrame(rows)


def top_k_tie_sensitivity(
    frame: pd.DataFrame, *, top_k_fraction: float
) -> dict[str, Any]:
    """Describe deterministic cutoff ties and outcome-only sensitivity bounds."""

    if not 0.0 < float(top_k_fraction) <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    count = max(1, int(math.ceil(float(top_k_fraction) * len(frame))))
    score = pd.to_numeric(frame["mapped_execution_ev"], errors="raise")
    selected = frame["global_top10_capacity_member"].astype(bool)
    cutoff = float(score.loc[selected].min())
    above = score > cutoff
    tied = np.isclose(score.to_numpy(dtype=float), cutoff, atol=1e-15, rtol=0.0)
    tied = pd.Series(tied, index=frame.index)
    selected_tied = selected & tied
    required_from_tie = count - int(above.sum())
    if int(selected_tied.sum()) != required_from_tie:
        raise ValueError("persisted global top-k does not select the required cutoff ties")
    tied_net = pd.to_numeric(
        frame.loc[tied, "execution_net_ev_12h"], errors="raise"
    ).sort_values(kind="stable")
    above_net = pd.to_numeric(
        frame.loc[above, "execution_net_ev_12h"], errors="raise"
    )
    observed = pd.to_numeric(
        frame.loc[selected, "execution_net_ev_12h"], errors="raise"
    )
    pessimistic = pd.concat([above_net, tied_net.head(required_from_tie)])
    optimistic = pd.concat([above_net, tied_net.tail(required_from_tie)])
    side_rows = []
    for side in ("long", "short"):
        is_side = frame["side_name"].eq(side)
        side_rows.append(
            {
                "side": side,
                "tied_at_cutoff_rows": int((tied & is_side).sum()),
                "selected_from_tie_rows": int((selected_tied & is_side).sum()),
            }
        )
    positive = score > 0.0
    return {
        "cutoff_mapped_ev": cutoff,
        "cutoff_mapped_ev_bps": cutoff * 10_000.0,
        "selected_global_top_k_rows": count,
        "strictly_above_cutoff_rows": int(above.sum()),
        "tied_at_cutoff_rows": int(tied.sum()),
        "selected_from_cutoff_tie_rows": int(selected_tied.sum()),
        "deterministic_tie_break": "candidate_id ascending",
        "observed_top_k_mean_net_bps": float(observed.mean() * 10_000.0),
        "outcome_only_pessimistic_tie_mean_net_bps": float(
            pessimistic.mean() * 10_000.0
        ),
        "outcome_only_optimistic_tie_mean_net_bps": float(
            optimistic.mean() * 10_000.0
        ),
        "positive_floor_rows": int(positive.sum()),
        "positive_floor_intersects_cutoff_tie": bool((positive & tied).any()),
        "positive_floor_unaffected_by_cutoff_tie": bool(not (positive & tied).any()),
        "side_support": side_rows,
        "sensitivity_is_diagnostic_not_a_selection_rule": True,
    }


def _portfolio_params(policy_path: Path) -> tuple[PortfolioPolicyParams, dict[str, Any]]:
    signed = _read_json(policy_path)
    embedded = signed.get("portfolio_policy")
    if not isinstance(embedded, dict):
        raise ValueError("signed simple-policy artifact lacks embedded portfolio_policy")
    payload = json.loads(json.dumps(embedded))
    concurrency = payload.setdefault("concurrency", {})
    if not isinstance(concurrency, dict):
        raise ValueError("embedded concurrency contract is invalid")
    if "enforce_position_count_cap" not in concurrency:
        concurrency["enforce_position_count_cap"] = bool(
            payload.get("enforce_position_count_cap", False)
        )
    risk = payload.setdefault("risk", {})
    if not isinstance(risk, dict):
        raise ValueError("embedded risk contract is invalid")
    for key in (
        "max_consecutive_losing_trades_per_archetype",
        "archetype_loss_cooldown_hours",
    ):
        if key not in risk and key in payload:
            risk[key] = payload[key]
    params = portfolio_policy_params_from_live_config(payload)
    if (
        params.max_concurrent_per_symbol < 1
        or params.max_new_entries_per_bar < 1
        or not 0.0 < params.max_total_wallet_allocation_pct <= 1.0
    ):
        raise ValueError("embedded portfolio constraints are not executable")
    contract = {
        "source": "portfolio_policy embedded in exact-label signed simple policy",
        "constraint_only_replay": True,
        "model_admission_refit": False,
        "raw_bayesian_sizing_replayed": False,
        "effective_position_count_cap": bool(params.enforce_position_count_cap),
        "max_concurrent_positions": int(params.max_concurrent_positions),
        "max_concurrent_per_side": params.max_concurrent_per_side,
        "max_concurrent_per_symbol": int(params.max_concurrent_per_symbol),
        "max_new_entries_per_bar": int(params.max_new_entries_per_bar),
        "max_total_wallet_allocation_pct": float(
            params.max_total_wallet_allocation_pct
        ),
    }
    return params, contract


def build_portfolio_candidates(frame: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    selected = frame.loc[frame[flag_column].astype(bool)].copy()
    side = selected["side_name"].astype(str)
    exit_minutes = (
        pd.to_numeric(selected["execution_exit_hour"], errors="raise") * 60.0
    )
    if not np.isfinite(exit_minutes).all() or (exit_minutes <= 0.0).any():
        raise ValueError("exact policy exit hour must be finite and positive")
    actual_exit_timestamp = selected["execution_decision_utc"] + pd.to_timedelta(
        exit_minutes, unit="m"
    )
    if (actual_exit_timestamp > selected["execution_label_end_utc"]).any():
        raise ValueError("actual policy exit occurs after the exact label horizon")
    normalized_rank = 1.0 - (
        pd.to_numeric(selected["global_rank"], errors="raise") - 1.0
    ) / max(len(frame) - 1, 1)
    candidates = pd.DataFrame(
        {
            "timestamp": selected["execution_decision_utc"],
            "symbol": selected["__symbol__"].astype(str),
            "side": side,
            "strategy_id": side + "_s52_meta_threshold_handoff",
            "base_strategy_threshold": 0.0,
            "calibrated_score": selected["mapped_execution_ev"],
            "normalized_rank_score": normalized_rank,
            "entry_price": selected["execution_entry_price"],
            # Label end is availability at the full 12h horizon, not when the
            # simulated position actually closed.
            "exit_timestamp": actual_exit_timestamp,
            "exit_price": selected["execution_exit_price"],
            "net_return": selected["execution_net_ev_12h"],
            "gross_return": selected["execution_gross_ev_12h"],
            "holding_bars": selected["execution_exit_hour"] * 60.0,
            "simple_policy_exit_reason": selected["execution_exit_reason"].astype(str),
            # Diagnostic only. replay_candidates consumes net_return verbatim.
            "fees_bps": selected["execution_cost_return"] * 10_000.0,
            # Explicit zero prevents stored fees/spread from entering priority
            # as an additional cost.  Spread is already in gross_return.
            "expected_friction_bps": 0.0,
            "price_gap_bps": 0.0,
            "candidate_id": selected["candidate_id"].astype(str),
            "policy_archetype": selected["policy_archetype"].astype(str),
        }
    )
    return normalise_candidate_table(candidates)


def portfolio_replays(
    frame: pd.DataFrame,
    *,
    policy_path: Path,
    initial_wallet: float,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    params, contract = _portfolio_params(policy_path)
    identity_curve = {
        "schema": "monotone_ev_curve_v1",
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "ev_span": 1.0,
        "n_rows": 0,
    }
    summaries: list[dict[str, Any]] = []
    decisions: list[pd.DataFrame] = []
    equities: list[pd.DataFrame] = []
    side_summaries: list[dict[str, Any]] = []
    replay_arms = [("config_faithful", params)]
    if not params.enforce_position_count_cap:
        replay_arms.append(
            (
                f"explicit_count_cap_{params.max_concurrent_positions}",
                replace(params, enforce_position_count_cap=True),
            )
        )
    for replay_arm, replay_params in replay_arms:
        for cohort, flag in COHORT_FLAGS.items():
            candidates = build_portfolio_candidates(frame, flag)
            cohort_decisions, equity, metrics = replay_candidates(
                candidates,
                replay_params,
                mode="global_auction",
                ev_curve=identity_curve,
                initial_wallet=float(initial_wallet),
                market_mode="perps",
            )
            cohort_decisions.insert(0, "cohort", cohort)
            cohort_decisions.insert(0, "replay_arm", replay_arm)
            equity.insert(0, "cohort", cohort)
            equity.insert(0, "replay_arm", replay_arm)
            decisions.append(cohort_decisions)
            equities.append(equity)
            accepted = cohort_decisions.loc[
                cohort_decisions["accepted"].astype(bool)
            ]
            accepted_net = pd.to_numeric(
                accepted["position_net_return"], errors="coerce"
            )
            rejected = cohort_decisions.loc[
                ~cohort_decisions["accepted"].astype(bool), "rejection_reason"
            ]
            summaries.append(
                {
                    "replay_arm": replay_arm,
                    "cohort": cohort,
                    "fixed_global_book_rows": int(len(candidates)),
                    "accepted_rows": int(len(accepted)),
                    "acceptance_rate": float(
                        len(accepted) / max(len(candidates), 1)
                    ),
                    "accepted_mean_net_bps": float(
                        accepted_net.mean() * 10_000.0
                    )
                    if len(accepted)
                    else np.nan,
                    "accepted_positive_precision": float(
                        (accepted_net > 0.0).mean()
                    )
                    if len(accepted)
                    else np.nan,
                    "rejection_reasons_json": json.dumps(
                        rejected.value_counts().sort_index().to_dict(),
                        sort_keys=True,
                    ),
                    **{
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, (dict, list))
                    },
                }
            )
            for side, side_group in accepted.groupby("side", sort=True):
                side_net = pd.to_numeric(
                    side_group["position_net_return"], errors="raise"
                )
                size = pd.to_numeric(side_group["position_size"], errors="raise")
                side_summaries.append(
                    {
                        "replay_arm": replay_arm,
                        "cohort": cohort,
                        "side": side,
                        "accepted_rows": int(len(side_group)),
                        "accepted_mean_net_bps": float(side_net.mean() * 10_000.0),
                        "accepted_positive_precision": float((side_net > 0).mean()),
                        "net_pnl": float((side_net * size).sum()),
                        "max_open_side_count_before": int(
                            side_group["side_count_before"].max()
                        ),
                    }
                )
    return (
        pd.DataFrame(summaries),
        pd.concat(decisions, ignore_index=True),
        pd.concat(equities, ignore_index=True),
        pd.DataFrame(side_summaries),
        {
            "params": asdict(params),
            "contract": contract,
            "replay_arms": [name for name, _ in replay_arms],
        },
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "scored",
        "scored_manifest",
        "labels",
        "labels_manifest",
        "preentry_manifest",
        "policy",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    args.output_dir = _resolve(args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame, scored_manifest, labels_manifest, packb_path = load_joined_population(
        scored_path=args.scored,
        scored_manifest_path=args.scored_manifest,
        labels_path=args.labels,
        labels_manifest_path=args.labels_manifest,
        preentry_manifest_path=args.preentry_manifest,
        policy_path=args.policy,
        top_k_fraction=args.top_k_fraction,
    )
    cohorts = cohort_metrics(frame)
    diagnostics = score_diagnostics(frame)
    tie_sensitivity = top_k_tie_sensitivity(
        frame, top_k_fraction=args.top_k_fraction
    )
    (
        portfolio_summary,
        decisions,
        equity,
        portfolio_side,
        portfolio_contract,
    ) = portfolio_replays(frame, policy_path=args.policy, initial_wallet=args.initial_wallet)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "joined_population": args.output_dir / "joined_population.parquet",
        "cohort_metrics": args.output_dir / "cohort_metrics.csv",
        "score_diagnostics": args.output_dir / "score_diagnostics.csv",
        "portfolio_summary": args.output_dir / "portfolio_summary.csv",
        "portfolio_side_metrics": args.output_dir / "portfolio_side_metrics.csv",
        "portfolio_decisions": args.output_dir / "portfolio_decisions.parquet",
        "portfolio_equity": args.output_dir / "portfolio_equity.parquet",
        "top_k_tie_sensitivity": args.output_dir / "top_k_tie_sensitivity.json",
    }
    frame.to_parquet(outputs["joined_population"], index=False, compression="zstd")
    cohorts.to_csv(outputs["cohort_metrics"], index=False)
    diagnostics.to_csv(outputs["score_diagnostics"], index=False)
    portfolio_summary.to_csv(outputs["portfolio_summary"], index=False)
    portfolio_side.to_csv(outputs["portfolio_side_metrics"], index=False)
    decisions.to_parquet(outputs["portfolio_decisions"], index=False, compression="zstd")
    equity.to_parquet(outputs["portfolio_equity"], index=False, compression="zstd")
    outputs["top_k_tie_sensitivity"].write_text(
        json.dumps(tie_sensitivity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overall = cohorts.loc[cohorts["scope"].eq("overall")].set_index("cohort")
    summary = {
        cohort: {
            key: (None if pd.isna(value) else value)
            for key, value in overall.loc[cohort].to_dict().items()
            if key not in {"scope", "utc_date", "side_name"}
        }
        for cohort in overall.index
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "research_only": True,
                "promotion_eligible": False,
                "cohorts": summary,
                "top_k_tie_sensitivity": tie_sensitivity,
                "portfolio": portfolio_summary.to_dict("records"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": REPORT_SCHEMA,
        "status": "research_only_retrospective_nonpromotable",
        "promotion_eligible": False,
        "ranking_contract": {
            "score": "mapped_execution_ev",
            "mapping": scored_manifest["contract"]["mapping"],
            "scope": "one pooled global book across all timestamps and sides",
            "top_k_fraction": float(args.top_k_fraction),
            "per_timestamp_quota": False,
            "admission_floors_bps_strictly_greater_than": [0, 25, 50],
        },
        "label_contract": {
            "schema": labels_manifest["schema"],
            "horizon_minutes": labels_manifest["exit_policy_contract"][
                "horizon_minutes"
            ],
            "replay_timeframe": labels_manifest["exit_policy_contract"][
                "replay_timeframe"
            ],
            "coverage": labels_manifest["coverage"]["overall"],
            "cost_accounting": labels_manifest["accounting"],
            "cost_reapplied_in_portfolio": False,
        },
        "portfolio_replay": portfolio_contract,
        "top_k_tie_sensitivity": tie_sensitivity,
        "coverage": {
            "rows": int(len(frame)),
            "sides": frame["side_name"].value_counts().sort_index().to_dict(),
            "decision_min_utc": frame["execution_decision_utc"].min(),
            "decision_max_utc": frame["execution_decision_utc"].max(),
            "global_top10_rows": int(frame["global_top10_capacity_member"].sum()),
            "admitted_gt_0bps_rows": int(
                frame["globally_admitted_floor_0bps"].sum()
            ),
            "admitted_gt_25bps_rows": int(
                frame["globally_admitted_floor_25bps"].sum()
            ),
            "admitted_gt_50bps_rows": int(
                frame["globally_admitted_floor_50bps"].sum()
            ),
        },
        "inputs": {
            "scored": _artifact_record(args.scored),
            "scored_manifest": _artifact_record(args.scored_manifest),
            "labels": _artifact_record(args.labels),
            "labels_manifest": _artifact_record(args.labels_manifest),
            "preentry_manifest": _artifact_record(args.preentry_manifest),
            "packb_context": _artifact_record(packb_path),
            "signed_simple_policy": _artifact_record(args.policy),
        },
        "outputs": {
            key: _artifact_record(path) for key, path in outputs.items()
        }
        | {"summary": _artifact_record(summary_path)},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
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
        "--labels",
        type=Path,
        default=DEFAULT_ROOT / "labels_12h/execution_ev_policy_labels.parquet",
    )
    parser.add_argument(
        "--labels-manifest",
        type=Path,
        default=DEFAULT_ROOT / "labels_12h/manifest.json",
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
    print(json.dumps(manifest["coverage"], indent=2, default=str))


if __name__ == "__main__":
    main()
