#!/usr/bin/env python3
"""Bounded, source-separated recurrence diagnostic for older exact 1m labels.

This is deliberately a *diagnostic*, not a model fit or a promotion replay.  It
joins the frozen base-only candidate source to the immutable exact 12-hour
labels, selects one pooled global tail per calendar month, and keeps physical
path outcomes distinct from the policy counterfactual economics.  It refuses
to run until the label bundle proves full candidate-level minute coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA = "historical_exact1m_base_recurrence_diagnostic_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
BASE_SCORE = "base_score"
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
DEFAULT_TRANSITION_DATASET = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)

# These must never be accepted as diagnostic *inputs*.  They are execution
# outcomes, policy proxies, labels, or derived target descriptions.  A feature
# may still be reported as an outcome elsewhere, but it cannot enter the
# pre-entry association/learnability screen.
OUTCOME_TOKENS = (
    "target__",
    "label",
    "execution_",
    "ev_after",
    "clean_exec",
    "dirty_positive",
    "first_touch",
    "full_path",
    "timeout",
    "mfe",
    "mae",
    "return",
    "pnl",
    "profit",
    "outcome",
    "exit",
    "gross",
    "cost",
    "net",
    "hit_probability",
)
CONTEXT_HINTS = (
    "transition",
    "bocpd",
    "changepoint",
    "model_health",
    "health_",
    "drift",
    "surprise",
    "reliability",
    "residual",
)
TRANSITION_TARGET_COLUMNS = (
    "target__transition_active",
    "target__destination_state",
    "target__phase",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file_hash(record: Mapping[str, Any], path: Path, *, role: str) -> None:
    expected = record.get("sha256")
    if not expected or _sha256(path) != str(expected):
        raise ValueError(f"{role} does not match its manifest hash")


def _canonical(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY) - set(frame.columns))
    if missing:
        raise ValueError(f"{role} missing identity columns: {missing}")
    output = frame.copy()
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
    output["__symbol__"] = output["__symbol__"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    if not output["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{role} has non-canonical side values")
    if output["candidate_id"].astype(str).duplicated().any():
        raise ValueError(f"{role} has duplicate candidate IDs")
    return output


def _stable_top_k(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """One pooled global top-k; candidate ID resolves every score tie."""

    score = pd.to_numeric(frame[BASE_SCORE], errors="coerce")
    work = frame.loc[np.isfinite(score)].copy()
    if work.empty:
        return work
    count = max(1, int(np.ceil(float(fraction) * len(work))))
    return work.sort_values(
        [BASE_SCORE, "candidate_id"], ascending=[False, True], kind="mergesort"
    ).head(count)


def _rank_ic(frame: pd.DataFrame, left: str, right: str) -> float:
    pair = frame.loc[:, [left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return np.nan
    return float(pair[left].corr(pair[right], method="spearman"))


def _mean_bps(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").mean() * 10000.0)


def _validate_stage(stage_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = stage_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "historical_backcast_exact1m_request_stage_v2":
        raise ValueError("unexpected exact request-stage schema")
    if manifest.get("evidence_scope") != "frozen_backcast_diagnostic_not_oof":
        raise ValueError("older request stage is not explicitly non-OOF diagnostic")
    if bool(manifest.get("promotion_eligible")) or bool(manifest.get("execution_parity_claim")):
        raise ValueError("older request stage must remain non-promotable/non-parity")
    if int(manifest.get("path_horizon_minutes", -1)) != 720:
        raise ValueError("exact recurrence diagnostic requires a 720-minute stage")
    record = manifest.get("outputs", {}).get("staged_candidates", {})
    path = Path(record.get("path") or stage_dir / "staged_candidates.parquet")
    if not path.exists():
        path = stage_dir / "staged_candidates.parquet"
    _require_file_hash(record, path, role="staged candidates")
    staged = pd.read_parquet(path)
    required = {
        "candidate_id", "source_shard_path", "source_shard_sha256", "source_row_number",
        "signal_timestamp", "decision_timestamp", "symbol", "side_name", BASE_SCORE,
    }
    missing = sorted(required - set(staged.columns))
    if missing:
        raise ValueError(f"staged candidates missing fields: {missing}")
    staged = staged.rename(columns={"signal_timestamp": "__ts__", "symbol": "__symbol__"})
    staged = _canonical(staged, role="staged candidates")
    if len(staged) != int(manifest.get("selected_rows", -1)):
        raise ValueError("staged candidate row count does not match manifest")
    return manifest, staged


def _validate_labels(labels_root: Path, stage: Mapping[str, Any], stage_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = labels_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "historical_backcast_exact1m_execution_path_labels_v1":
        raise ValueError("unexpected exact multitask label schema")
    if manifest.get("status") != "materialized":
        raise ValueError("exact multitask labels are not materialized")
    if manifest.get("oof_status") != "not_oof" or bool(manifest.get("promotion_eligible")):
        raise ValueError("exact multitask labels must remain non-OOF and non-promotable")
    if bool(manifest.get("execution_parity_claim")):
        raise ValueError("older exact labels must not claim deployed execution parity")
    timing = manifest.get("label_timing", {})
    if timing.get("path") != "[decision, decision+12h)":
        raise ValueError("exact multitask labels do not have the 12h decision path")
    output = manifest.get("outputs", {}).get("joined_multitask_labels", {})
    labels_path = Path(output.get("path") or labels_root / "joined_multitask_labels.parquet")
    if not labels_path.exists():
        labels_path = labels_root / "joined_multitask_labels.parquet"
    _require_file_hash(output, labels_path, role="joined exact multitask labels")
    if int(output.get("rows", -1)) != int(manifest.get("rows", -2)):
        raise ValueError("label manifest output/total rows disagree")
    if int(manifest.get("rows", -1)) != len(stage_frame):
        raise ValueError("label count does not cover the frozen stage")

    coverage_record = manifest.get("sources", {}).get("candidate_coverage_manifest", {})
    coverage_path = Path(coverage_record.get("path", ""))
    if not coverage_path.exists():
        raise ValueError("exact labels do not provide a candidate-coverage manifest")
    _require_file_hash(coverage_record, coverage_path, role="candidate coverage manifest")
    coverage = _read_json(coverage_path)
    if (
        coverage.get("schema") != "historical_exact1m_candidate_coverage_v1"
        or coverage.get("status") != "complete"
        or float(coverage.get("candidate_coverage_fraction", 0.0)) != 1.0
        or int(coverage.get("complete_candidates", -1)) != len(stage_frame)
    ):
        raise ValueError("exact labels are incomplete: 720/720 candidate coverage is required")

    use = [
        *IDENTITY,
        "__decision_ts__", "__opportunity_occurred_12h__", "__peak_mfe_atr_12h__",
        "__favorable_payoff_return_12h__", "__adverse_competing_risk_12h__",
        "__timeout_outcome_12h__", "__exit_conversion_loss_return_12h__",
        "__opportunity_scarcity_proxy_12h__", "__exit_conversion_failure_proxy_12h__",
        "__timeout_degradation_proxy_12h__", "__adverse_payoff_expansion_proxy_12h__",
        "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h",
        "execution_exit_reason", "execution_exit_hour",
    ]
    schema_names = set(pq.read_schema(labels_path).names)
    missing = sorted(set(use) - schema_names)
    if missing:
        raise ValueError(f"exact multitask labels missing required fields: {missing}")
    labels = _canonical(pd.read_parquet(labels_path, columns=use), role="exact labels")
    labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True, errors="raise")
    if len(labels) != len(stage_frame):
        raise ValueError("exact label output does not contain every frozen candidate")
    return manifest, labels


def _requested_context_columns(columns: Iterable[str], explicit: str) -> list[str]:
    available = set(columns)
    if explicit.strip():
        requested = [part.strip() for part in explicit.split(",") if part.strip()]
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"requested context columns are unavailable: {missing}")
    else:
        requested = [
            column for column in columns
            if any(hint in column.lower() for hint in CONTEXT_HINTS)
        ]
    forbidden = [
        column for column in requested
        if any(token in column.lower() for token in OUTCOME_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"outcome-derived columns cannot be diagnostic inputs: {sorted(forbidden)}")
    return sorted(set(requested))


def _load_source_context(stage: Mapping[str, Any], staged: pd.DataFrame, *, explicit_context: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    source_records = stage.get("sources")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("request stage has no source-shard lineage")
    schemas: dict[str, list[str]] = {}
    all_columns: set[str] = set()
    for record in source_records:
        path = Path(record["path"])
        _require_file_hash(record, path, role="frozen candidate source shard")
        names = pq.read_schema(path).names
        schemas[str(path.resolve())] = names
        all_columns.update(names)
    requested = _requested_context_columns(all_columns, explicit_context)
    unavailable_by_shard = sorted(
        column for column in requested
        if any(column not in names for names in schemas.values())
    )
    # A field that is not present throughout the historical source cannot make a
    # coherent recurrence association.  Omit it rather than imputing by era.
    context_columns = [column for column in requested if column not in unavailable_by_shard]
    parts: list[pd.DataFrame] = []
    required = ["__ts__", "__symbol__", "side_name", BASE_SCORE, *context_columns]
    stage_groups = staged.groupby("source_shard_path", sort=True)
    for path_text, group in stage_groups:
        path = Path(str(path_text))
        resolved = str(path.resolve())
        if resolved not in schemas:
            raise ValueError("staged candidate references a source outside the frozen manifest")
        sha_values = group["source_shard_sha256"].astype(str).unique()
        if len(sha_values) != 1 or _sha256(path) != sha_values[0]:
            raise ValueError("staged source shard hash disagrees with candidate lineage")
        columns = [column for column in required if column in schemas[resolved]]
        source = pd.read_parquet(path, columns=columns)
        positions = group["source_row_number"].to_numpy(dtype=np.int64)
        if (positions < 0).any() or (positions >= len(source)).any():
            raise ValueError("staged source row number is outside its frozen shard")
        selected = source.iloc[positions].copy().reset_index(drop=True)
        reference = group.reset_index(drop=True)
        for source_col, stage_col in (("__ts__", "__ts__"), ("__symbol__", "__symbol__"), ("side_name", "side_name")):
            lhs = selected[source_col].astype(str)
            rhs = reference[stage_col].astype(str)
            if not lhs.eq(rhs).all():
                raise ValueError(f"frozen source identity disagrees with stage on {source_col}")
        if not np.allclose(
            pd.to_numeric(selected[BASE_SCORE], errors="raise"),
            pd.to_numeric(reference[BASE_SCORE], errors="raise"), rtol=0.0, atol=1e-12,
        ):
            raise ValueError("source base score disagrees with frozen stage")
        part = reference.loc[:, [*IDENTITY, BASE_SCORE]].copy()
        for column in context_columns:
            part[column] = selected[column].to_numpy()
        parts.append(part)
    source_context = _canonical(pd.concat(parts, ignore_index=True), role="frozen source context")
    if len(source_context) != len(staged):
        raise ValueError("frozen source context did not preserve all staged rows")
    return source_context, context_columns, unavailable_by_shard


def _join_transition_dataset(
    frame: pd.DataFrame,
    paths: Sequence[Path] | None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    if not paths:
        return frame, {"available": False, "reason": "no_transition_dataset"}, []
    resolved_paths = [Path(path) for path in paths]
    missing_paths = [str(path) for path in resolved_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"transition datasets do not exist: {sorted(missing_paths)}"
        )
    schemas = [pq.read_schema(path).names for path in resolved_paths]
    schema = schemas[0]
    if any(names != schema for names in schemas[1:]):
        raise ValueError("transition datasets do not share the exact column contract")
    if "execution_decision_utc" not in schema:
        raise ValueError("transition dataset lacks execution_decision_utc")
    safe_context = [
        column for column in schema
        if column not in {"source_utc", "execution_decision_utc", *TRANSITION_TARGET_COLUMNS}
        and not column.startswith("target__")
    ]
    target_columns = [column for column in TRANSITION_TARGET_COLUMNS if column in schema]
    transition = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["execution_decision_utc", *safe_context, *target_columns],
            )
            for path in resolved_paths
        ],
        ignore_index=True,
    )
    transition["execution_decision_utc"] = pd.to_datetime(
        transition["execution_decision_utc"], utc=True, errors="raise"
    )
    if transition["execution_decision_utc"].duplicated().any():
        raise ValueError("transition dataset has duplicate decision timestamps")
    transition = transition.rename(columns={column: f"transition_ctx__{column}" for column in safe_context})
    transition = transition.rename(columns={column: f"transition_target_diagnostic__{column.removeprefix('target__')}" for column in target_columns})
    work = frame.merge(transition, on="execution_decision_utc", how="left", validate="many_to_one", indicator="_transition_join")
    joined = work["_transition_join"].eq("both")
    years = pd.to_datetime(work["execution_decision_utc"], utc=True).dt.year
    coverage_by_year = {
        str(year): float(joined.loc[years.eq(year)].mean())
        for year in sorted(years.unique())
    }
    coverage_2022 = coverage_by_year.get("2022")
    if coverage_2022 is None:
        interpretation_2022 = "candidate_frame_has_no_2022_rows"
    elif coverage_2022 == 1.0:
        interpretation_2022 = "fully_covered_by_supplied_transition_panel"
    elif coverage_2022 == 0.0:
        interpretation_2022 = (
            "no_transition_dataset_coverage_explicitly_unobserved"
        )
    else:
        interpretation_2022 = (
            "partially_covered_by_supplied_transition_panel_do_not_impute"
        )
    report = {
        "available": True,
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in resolved_paths
        ],
        "context_columns": int(len(safe_context)),
        "target_columns_diagnostic_only": target_columns,
        "rows_with_context": int(joined.sum()),
        "coverage": float(joined.mean()),
        "coverage_by_year": coverage_by_year,
        "2022_interpretation": interpretation_2022,
        "target_handling": "transition target labels are diagnostic strata only, never inputs or promotion evidence",
    }
    return work.drop(columns="_transition_join"), report, [f"transition_ctx__{column}" for column in safe_context]


def _score_bridge(frame: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "physical_opportunity": "__opportunity_occurred_12h__",
        "physical_peak_mfe_atr": "__peak_mfe_atr_12h__",
        "policy_gross": "execution_gross_ev_12h",
        "policy_cost": "execution_cost_return",
        "policy_net": "execution_net_ev_12h",
    }
    rows: list[dict[str, Any]] = []
    for (month, side), group in frame.groupby(["month", "side_name"], sort=True):
        row: dict[str, Any] = {"month": month, "side_name": side, "rows": int(len(group))}
        for name, column in targets.items():
            row[f"base_score_rank_ic__{name}"] = _rank_ic(group, BASE_SCORE, column)
            row[f"mean__{name}"] = float(pd.to_numeric(group[column], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _monthly_topk(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        for fraction in TOP_FRACTIONS:
            selected = _stable_top_k(group, fraction)
            side_counts = selected["side_name"].value_counts()
            rows.append({
                "month": month, "top_fraction": fraction, "eligible_rows": int(len(group)),
                "selected_rows": int(len(selected)), "selection_scope": "pooled_global_month",
                "tie_break": "candidate_id_ascending", "mean_gross_bps": _mean_bps(selected["execution_gross_ev_12h"]),
                "mean_cost_bps": _mean_bps(selected["execution_cost_return"]),
                "mean_net_bps": _mean_bps(selected["execution_net_ev_12h"]),
                "sum_net_return": float(pd.to_numeric(selected["execution_net_ev_12h"], errors="coerce").sum()),
                "positive_net_rate": float(pd.to_numeric(selected["execution_net_ev_12h"], errors="coerce").gt(0).mean()),
                "long_selected_rows": int(side_counts.get("long", 0)), "short_selected_rows": int(side_counts.get("short", 0)),
            })
    return pd.DataFrame(rows)


def _top10_decomposition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        selected = _stable_top_k(group, 0.10)
        opportunity = selected["__opportunity_occurred_12h__"].astype(bool)
        positive = pd.to_numeric(selected["execution_net_ev_12h"], errors="coerce").gt(0)
        rows.append({
            "month": month, "selection_scope": "pooled_global_month", "selected_rows": int(len(selected)),
            "opportunity_prevalence": float(opportunity.mean()),
            "favorable_payoff_mean": float(pd.to_numeric(selected["__favorable_payoff_return_12h__"], errors="coerce").mean()),
            "favorable_payoff_mean_given_opportunity": float(pd.to_numeric(selected.loc[opportunity, "__favorable_payoff_return_12h__"], errors="coerce").mean()),
            "adverse_competing_risk_rate": float(selected["__adverse_competing_risk_12h__"].mean()),
            "timeout_rate": float(selected["__timeout_outcome_12h__"].mean()),
            "exit_conversion_loss_bps": _mean_bps(selected["__exit_conversion_loss_return_12h__"]),
            "conversion_failure_rate": float(selected["__exit_conversion_failure_proxy_12h__"].mean()),
            "timeout_degradation_rate": float(selected["__timeout_degradation_proxy_12h__"].mean()),
            "adverse_expansion_rate": float(selected["__adverse_payoff_expansion_proxy_12h__"].mean()),
            "mean_gross_bps": _mean_bps(selected["execution_gross_ev_12h"]),
            "mean_cost_bps": _mean_bps(selected["execution_cost_return"]),
            "mean_net_bps": _mean_bps(selected["execution_net_ev_12h"]),
            "positive_net_rate": float(positive.mean()),
            "mean_net_bps_given_positive": _mean_bps(selected.loc[positive, "execution_net_ev_12h"]),
            "mean_net_bps_given_nonpositive": _mean_bps(selected.loc[~positive, "execution_net_ev_12h"]),
        })
        for reason, exited in selected.groupby("execution_exit_reason", sort=True):
            exit_rows.append({
                "month": month, "exit_reason": str(reason), "rows": int(len(exited)),
                "share": float(len(exited) / len(selected)),
                "mean_net_bps": _mean_bps(exited["execution_net_ev_12h"]),
                "mean_gross_bps": _mean_bps(exited["execution_gross_ev_12h"]),
                "mean_hold_hours": float(pd.to_numeric(exited["execution_exit_hour"], errors="coerce").mean()),
                "positive_net_rate": float(pd.to_numeric(exited["execution_net_ev_12h"], errors="coerce").gt(0).mean()),
            })
    return pd.DataFrame(rows), pd.DataFrame(exit_rows)


def _rank_economics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        ordered = group.sort_values([BASE_SCORE, "candidate_id"], ascending=[False, True], kind="mergesort").copy()
        ordered["rank_decile"] = np.minimum(10, (np.arange(len(ordered)) * 10 // len(ordered)) + 1)
        for decile, bucket in ordered.groupby("rank_decile", sort=True):
            rows.append({
                "month": month, "rank_decile": int(decile), "rows": int(len(bucket)),
                "mean_score": float(pd.to_numeric(bucket[BASE_SCORE], errors="coerce").mean()),
                "opportunity_rate": float(bucket["__opportunity_occurred_12h__"].mean()),
                "mean_peak_mfe_atr": float(pd.to_numeric(bucket["__peak_mfe_atr_12h__"], errors="coerce").mean()),
                "mean_gross_bps": _mean_bps(bucket["execution_gross_ev_12h"]),
                "mean_cost_bps": _mean_bps(bucket["execution_cost_return"]),
                "mean_net_bps": _mean_bps(bucket["execution_net_ev_12h"]),
            })
    by_month = pd.DataFrame(rows)
    transitions: list[dict[str, Any]] = []
    months = sorted(by_month["month"].unique())
    for previous, current in zip(months, months[1:]):
        left = by_month.loc[by_month["month"].eq(previous)].set_index("rank_decile")
        right = by_month.loc[by_month["month"].eq(current)].set_index("rank_decile")
        for decile in sorted(set(left.index) & set(right.index)):
            transitions.append({
                "from_month": previous, "to_month": current, "rank_decile": int(decile),
                "delta_opportunity_rate": float(right.loc[decile, "opportunity_rate"] - left.loc[decile, "opportunity_rate"]),
                "delta_peak_mfe_atr": float(right.loc[decile, "mean_peak_mfe_atr"] - left.loc[decile, "mean_peak_mfe_atr"]),
                "delta_gross_bps": float(right.loc[decile, "mean_gross_bps"] - left.loc[decile, "mean_gross_bps"]),
                "delta_cost_bps": float(right.loc[decile, "mean_cost_bps"] - left.loc[decile, "mean_cost_bps"]),
                "delta_net_bps": float(right.loc[decile, "mean_net_bps"] - left.loc[decile, "mean_net_bps"]),
            })
    return by_month, pd.DataFrame(transitions)


def _associations(frame: pd.DataFrame, columns: Sequence[str], *, family: str) -> pd.DataFrame:
    output_columns = [
        "feature", "family", "rows", "available_fraction", "rank_ic_base_score",
        "rank_ic_physical_opportunity", "rank_ic_policy_net", "top_quartile_net_bps",
        "bottom_quartile_net_bps", "top_minus_bottom_net_bps",
        "top_quartile_opportunity_rate", "bottom_quartile_opportunity_rate", "screen_status",
    ]
    rows: list[dict[str, Any]] = []
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce")
        valid = value.notna()
        if valid.sum() < 20 or value.loc[valid].nunique() < 2:
            continue
        work = frame.loc[valid].copy()
        work["_feature"] = value.loc[valid].to_numpy()
        # This is a univariate association screen only.  It has no fitting,
        # threshold optimization, or treatment as an inference feature.
        q75 = float(work["_feature"].quantile(0.75))
        q25 = float(work["_feature"].quantile(0.25))
        high = work.loc[work["_feature"].ge(q75)]
        low = work.loc[work["_feature"].le(q25)]
        rows.append({
            "feature": column, "family": family, "rows": int(len(work)),
            "available_fraction": float(valid.mean()),
            "rank_ic_base_score": _rank_ic(work, "_feature", BASE_SCORE),
            "rank_ic_physical_opportunity": _rank_ic(work, "_feature", "__opportunity_occurred_12h__"),
            "rank_ic_policy_net": _rank_ic(work, "_feature", "execution_net_ev_12h"),
            "top_quartile_net_bps": _mean_bps(high["execution_net_ev_12h"]),
            "bottom_quartile_net_bps": _mean_bps(low["execution_net_ev_12h"]),
            "top_minus_bottom_net_bps": _mean_bps(high["execution_net_ev_12h"]) - _mean_bps(low["execution_net_ev_12h"]),
            "top_quartile_opportunity_rate": float(high["__opportunity_occurred_12h__"].mean()),
            "bottom_quartile_opportunity_rate": float(low["__opportunity_occurred_12h__"].mean()),
            "screen_status": "diagnostic_univariate_not_a_model",
        })
    return pd.DataFrame(rows, columns=output_columns)


def _transition_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    column = "transition_target_diagnostic__transition_active"
    if column not in frame.columns:
        return pd.DataFrame(columns=["month", "transition_active", "rows"])
    rows: list[dict[str, Any]] = []
    # Selection is performed *before* transition stratification.  Selecting a
    # new top 10% within active/inactive strata would change the global book
    # and is not the requested interaction diagnostic.
    for month, month_frame in frame.groupby("month", sort=True):
        selected = _stable_top_k(month_frame, 0.10).dropna(subset=[column])
        for active, group in selected.groupby(column, sort=True):
            net = pd.to_numeric(group["execution_net_ev_12h"], errors="coerce")
            n = int(net.notna().sum())
            mean = _mean_bps(net)
            stderr = float(net.std(ddof=1) * 10000.0 / np.sqrt(n)) if n > 1 else np.nan
            support_gate = n >= 50
            rows.append({
                "month": month, "transition_active": str(active), "eligible_rows_month": int(len(month_frame)),
                "selected_global_top10_rows_month": int(len(_stable_top_k(month_frame, 0.10))),
                "selected_rows_transition_stratum": int(len(group)),
                "selection_scope": "pooled_global_month_then_transition_stratum_diagnostic_only",
                "mean_net_bps": mean,
                "mean_net_bps_ci95_low_normal": mean - 1.96 * stderr if np.isfinite(stderr) else np.nan,
                "mean_net_bps_ci95_high_normal": mean + 1.96 * stderr if np.isfinite(stderr) else np.nan,
                "support_gate_min_rows": 50, "support_gate_pass": support_gate,
                "support_interpretation": "descriptive_only" if support_gate else "underpowered_do_not_conclude",
                "opportunity_rate": float(group["__opportunity_occurred_12h__"].mean()),
                "adverse_rate": float(group["__adverse_competing_risk_12h__"].mean()),
                "timeout_rate": float(group["__timeout_outcome_12h__"].mean()),
                "conversion_loss_bps": _mean_bps(group["__exit_conversion_loss_return_12h__"]),
            })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage, staged = _validate_stage(Path(args.stage_dir))
    label_manifest, labels = _validate_labels(Path(args.labels_root), stage, staged)
    source, source_context_columns, unavailable_context = _load_source_context(
        stage, staged, explicit_context=str(args.context_columns)
    )
    frame = source.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(frame) != len(staged):
        raise ValueError("source-to-exact-label join lost frozen candidates")
    expected_decision = pd.to_datetime(frame["__ts__"], utc=True) + pd.Timedelta(hours=1)
    if not pd.to_datetime(frame["__decision_ts__"], utc=True).eq(expected_decision).all():
        raise ValueError("exact label decision timing disagrees with the frozen signal")
    frame["execution_decision_utc"] = expected_decision
    transition_paths = None
    if not args.no_transition_dataset:
        configured_transition_paths = args.transition_dataset
        if isinstance(configured_transition_paths, (str, Path)):
            configured_transition_paths = [configured_transition_paths]
        transition_paths = [
            Path(path)
            for path in (
                configured_transition_paths
                or [DEFAULT_TRANSITION_DATASET]
            )
        ]
    frame, transition_report, transition_context = _join_transition_dataset(
        frame,
        transition_paths,
    )
    frame["month"] = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    bridge = _score_bridge(frame)
    topk = _monthly_topk(frame)
    decomposition, exit_mix = _top10_decomposition(frame)
    rank_by_month, rank_transition = _rank_economics(frame)
    source_association = _associations(frame, source_context_columns, family="source_preentry_context")
    transition_association = _associations(frame, transition_context, family="frozen_transition_context")
    association = pd.concat([source_association, transition_association], ignore_index=True)
    transition_interactions = _transition_interactions(frame)

    output.mkdir(parents=True, exist_ok=False)
    bridge.to_csv(output / "monthly_side_score_target_bridge.csv", index=False)
    topk.to_csv(output / "monthly_global_topk_economics.csv", index=False)
    decomposition.to_csv(output / "monthly_global_top10_decomposition.csv", index=False)
    exit_mix.to_csv(output / "monthly_global_top10_exit_mix.csv", index=False)
    rank_by_month.to_csv(output / "rank_to_economics_by_month.csv", index=False)
    rank_transition.to_csv(output / "rank_to_economics_month_transitions.csv", index=False)
    association.to_csv(output / "preentry_context_associations.csv", index=False)
    transition_interactions.to_csv(output / "transition_execution_component_interactions.csv", index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "diagnostic_complete_non_oof_non_promotable",
        "evidence_scope": "frozen_backcast_diagnostic_not_oof",
        "oof_status": "not_oof",
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "rows": int(len(frame)),
        "date_range": {"start": frame["__ts__"].min(), "end": frame["__ts__"].max()},
        "selection": {
            "scope": "one pooled global top-k per calendar month", "fractions": list(TOP_FRACTIONS),
            "tie_break": "candidate_id ascending", "side_quotas": False, "timestamp_quotas": False,
        },
        "source_separation": label_manifest.get("source_separation"),
        "physical_targets": ["__opportunity_occurred_12h__", "__peak_mfe_atr_12h__"],
        "policy_targets": ["execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"],
        "preentry_context": {
            "source_columns": source_context_columns,
            "unavailable_in_some_source_shards": unavailable_context,
            "transition_context_columns": transition_context,
            "outcome_derived_columns_rejected": True,
            "screen": "univariate descriptive association only; no fitted model, selection, threshold, or promotion claim",
        },
        "transition_join": transition_report,
        "sources": {
            "stage_manifest": {"path": str((Path(args.stage_dir) / "manifest.json").resolve()), "sha256": _sha256(Path(args.stage_dir) / "manifest.json")},
            "labels_manifest": {"path": str((Path(args.labels_root) / "manifest.json").resolve()), "sha256": _sha256(Path(args.labels_root) / "manifest.json")},
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": {},
    }
    for path in sorted(output.glob("*.csv")):
        manifest["outputs"][path.name] = {"rows": int(sum(1 for _ in path.open(encoding="utf-8")) - 1), "sha256": _sha256(path)}
    _write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--transition-dataset",
        type=Path,
        action="append",
        default=None,
        help=(
            "repeatable exact-schema transition panel; defaults to the frozen "
            "2023+ panel when omitted"
        ),
    )
    parser.add_argument("--no-transition-dataset", action="store_true")
    parser.add_argument("--context-columns", default="", help="comma-separated, pre-entry source columns; all are leakage-screened")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
