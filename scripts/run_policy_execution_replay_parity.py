#!/usr/bin/env python3
"""Replay-only row parity audit for the canonical policy and execution chain.

The harness deliberately starts from persisted post-meta rows. It does not
assert base/meta parity and never packages artifacts or controls inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.inference.canonical_meta_postprocessor import (
    CanonicalMetaPostprocessor,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)
from extreme_price_movements.portfolio_policy_replay import (
    load_portfolio_policy_params,
    replay_candidates,
)
from scripts.ablate_simple_policy_exit_geometry import _load_bundles
from scripts.materialize_canonical_exit_policy_replay import (
    _apply_policy_spread_to_returns,
    _load_ev_curve,
    _materialize_exit_rows,
    _policy_summary_path,
    _portfolio_candidates,
)


KEYS = ["timestamp", "symbol", "side_name"]
NUMERIC_TOLERANCE = 1e-6
CANONICAL_ADMISSION_POLICY_ID = "side_archetype_hier_ev_fixed70_trim10_21d_v1"
CANONICAL_POSTPROCESSOR_POLICY_ID = (
    "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _normalise_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts_col = _first_column(out, ("timestamp", "__ts__", "signal_ts", "signal_bar_ts"))
    symbol_col = _first_column(out, ("symbol", "__symbol__"))
    if ts_col is None or symbol_col is None:
        raise ValueError("rows require timestamp/__ts__ and symbol/__symbol__")
    out["timestamp"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out["symbol"] = out[symbol_col].astype(str)
    if "side_name" not in out.columns:
        side = pd.to_numeric(out.get("side"), errors="coerce")
        out["side_name"] = np.where(side.lt(0.0), "short", "long")
    out["side_name"] = out["side_name"].astype(str).str.lower()
    arch_col = _first_column(
        out,
        (
            "policy_archetype",
            "archetype_policy_key",
            "local_side_archetype",
            "archetype_label_family",
        ),
    )
    out["policy_archetype"] = (
        out[arch_col].fillna("missing").astype(str) if arch_col else "missing"
    )
    for side_name in ("long", "short"):
        prefix = f"{side_name}__"
        mask = out["side_name"].eq(side_name) & out["policy_archetype"].str.startswith(
            prefix, na=False
        )
        out.loc[mask, "policy_archetype"] = out.loc[
            mask, "policy_archetype"
        ].str.removeprefix(prefix)
    duplicated = out.duplicated(KEYS, keep=False)
    if bool(duplicated.any()):
        sample = out.loc[duplicated, KEYS].head(5).to_dict("records")
        raise ValueError(f"duplicate policy row keys: {sample}")
    return out.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _decision_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        parent_rank = row.get("historical_rank", row.get("policy_parent_rank"))
        expected_ev = row.get(
            "expected_net_ev_after_1pct",
            row.get("expected_net_ev_after_1pct_mlp_direct"),
        )
        expected_rank = row.get("expected_ev_rank_score", row.get("rank_mlp_direct"))
        chain = dict(row)
        chain.update(
            {
                "v9_tail95_predecessor_rank": parent_rank,
                "expected_net_ev_after_1pct_side_archetype": expected_ev,
                "expected_ev_rank_score": expected_rank,
            }
        )
        records.append(
            {
                "signal_bar_ts": row["timestamp"],
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "side_name": row["side_name"],
                "side": row["side_name"],
                "strategy_id": row.get(
                    "strategy_id", f"{row['side_name']}_canonical_meta_policy"
                ),
                "policy_archetype": row["policy_archetype"],
                "calibrated_score": expected_rank,
                "v9_tail95_predecessor_rank": parent_rank,
                "expected_net_ev_after_1pct_side_archetype": expected_ev,
                "expected_ev_rank_score": expected_rank,
                "chain_results": chain,
            }
        )
    return records


def replay_policy_chain(
    rows: pd.DataFrame,
    *,
    predecessor_bundle: Path,
    residual_event_state: Path,
    regime_ev_artifact: Path,
    admission_policy: Path,
) -> pd.DataFrame:
    """Apply V9, MLP/hierarchical EV, and causal 21-day admission."""
    source = _normalise_rows(rows)
    postprocessor = CanonicalMetaPostprocessor.load(
        predecessor_bundle_path=predecessor_bundle,
        residual_event_state_path=residual_event_state,
        regime_ev_artifact_path=regime_ev_artifact,
    )
    transformed = postprocessor.transform(source)
    decisions = _decision_records(_normalise_rows(transformed))
    apply_threshold_basis_policy_to_decisions(
        decisions,
        policy=load_threshold_basis_policy(admission_policy),
    )
    output = pd.DataFrame(decisions)
    chain = pd.json_normalize(output.pop("chain_results")).set_axis(output.index)
    for column in chain.columns:
        if column not in output.columns:
            output[column] = chain[column]
    return _normalise_rows(output)


def _numeric_detail(
    merged: pd.DataFrame,
    *,
    stage: str,
    metric: str,
    reference_col: str,
    replay_col: str,
    tolerance: float,
) -> pd.DataFrame:
    reference = pd.to_numeric(merged[reference_col], errors="coerce")
    replay = pd.to_numeric(merged[replay_col], errors="coerce")
    delta = replay - reference
    finite = np.isfinite(reference) & np.isfinite(replay)
    return pd.DataFrame(
        {
            **{key: merged[key] for key in KEYS},
            "stage": stage,
            "metric": metric,
            "reference_value": reference,
            "replay_value": replay,
            "delta": delta,
            "abs_delta": delta.abs(),
            "matched_finite": finite,
            "mismatch": (~finite) | delta.abs().gt(float(tolerance)),
        }
    )


def compare_policy_rows(
    reference_rows: pd.DataFrame,
    replay_rows: pd.DataFrame,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    reference = _normalise_rows(reference_rows)
    replay = _normalise_rows(replay_rows)
    reference = reference.rename(
        columns={name: f"{name}__reference" for name in reference.columns if name not in KEYS}
    )
    replay = replay.rename(
        columns={name: f"{name}__replay" for name in replay.columns if name not in KEYS}
    )
    merged = reference.merge(
        replay,
        on=KEYS,
        how="outer",
        indicator=True,
    )
    specs = {
        "v9_tail95": (
            ("historical_rank", "policy_parent_rank", "v9_tail95_predecessor_rank"),
            ("historical_rank", "v9_tail95_predecessor_rank", "policy_parent_rank"),
        ),
        "mlp_rank": (
            ("rank_mlp_direct", "score_regime_calibrated"),
            ("score_regime_calibrated", "rank_mlp_direct"),
        ),
        "hierarchical_expected_ev": (
            (
                "expected_net_ev_after_1pct_mlp_direct",
                "expected_net_ev_after_1pct",
            ),
            (
                "expected_net_ev_after_1pct",
                "expected_net_ev_after_1pct_mlp_direct",
            ),
        ),
        "hierarchical_expected_ev_rank": (
            ("expected_ev_rank_score",),
            ("expected_ev_rank_score",),
        ),
        "admission_corrected_ev": (
            ("threshold_basis_corrected_expected_ev",),
            ("threshold_basis_corrected_expected_ev",),
        ),
        "admission_rank": (
            ("threshold_basis_rank_score", "rank_pct"),
            ("threshold_basis_rank_score", "rank_pct"),
        ),
    }
    detail_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for stage, (reference_names, replay_names) in specs.items():
        reference_col = next(
            (f"{name}__reference" for name in reference_names if f"{name}__reference" in merged),
            None,
        )
        replay_col = next(
            (f"{name}__replay" for name in replay_names if f"{name}__replay" in merged),
            None,
        )
        if reference_col is None or replay_col is None:
            summaries.append(
                {
                    "layer": stage,
                    "status": "missing_columns",
                    "reference_candidates": list(reference_names),
                    "replay_candidates": list(replay_names),
                    "pass": False,
                }
            )
            continue
        detail = _numeric_detail(
            merged,
            stage=stage,
            metric=stage,
            reference_col=reference_col,
            replay_col=replay_col,
            tolerance=tolerance,
        )
        detail_parts.append(detail)
        mismatches = detail["mismatch"] | merged["_merge"].ne("both")
        summaries.append(
            {
                "layer": stage,
                "matched_rows": int(merged["_merge"].eq("both").sum()),
                "max_abs_delta": float(detail.loc[detail["matched_finite"], "abs_delta"].max()),
                "mean_abs_delta": float(detail.loc[detail["matched_finite"], "abs_delta"].mean()),
                "mismatch_count": int(mismatches.sum()),
                "first_divergence": (
                    merged.loc[mismatches, KEYS].head(1).to_dict("records") or [None]
                )[0],
                "pass": not bool(mismatches.any()),
            }
        )

    reference_selected = pd.Series(
        reference.get("threshold_basis_selected__reference", False), index=reference.index
    ).fillna(False).astype(bool)
    replay_selected = pd.Series(
        replay.get("threshold_basis_selected__replay", False), index=replay.index
    ).fillna(False).astype(bool)
    reference_keys = set(map(tuple, reference.loc[reference_selected, KEYS].itertuples(index=False, name=None)))
    replay_keys = set(map(tuple, replay.loc[replay_selected, KEYS].itertuples(index=False, name=None)))
    summaries.append(
        {
            "layer": "admission_decision",
            "reference_selected": len(reference_keys),
            "replay_selected": len(replay_keys),
            "reference_only": len(reference_keys - replay_keys),
            "replay_only": len(replay_keys - reference_keys),
            "mismatch_count": len(reference_keys ^ replay_keys),
            "pass": reference_keys == replay_keys,
        }
    )
    detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return detail, summaries


def summarize_fixed_ev_policy_rows(
    rows: pd.DataFrame,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Summarize independent matrix-versus-production fixed-EV decisions."""
    frame = _normalise_rows(rows)
    required = {
        "matrix_corrected_ev",
        "replay_corrected_ev",
        "matrix_admitted",
        "replay_admitted",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fixed-EV comparison rows are missing {missing}")
    matrix_ev = pd.to_numeric(frame["matrix_corrected_ev"], errors="coerce")
    replay_ev = pd.to_numeric(frame["replay_corrected_ev"], errors="coerce")
    finite = np.isfinite(matrix_ev) & np.isfinite(replay_ev)
    delta = replay_ev - matrix_ev
    frame["corrected_ev_delta"] = delta
    frame["corrected_ev_abs_delta"] = delta.abs()
    frame["corrected_ev_mismatch"] = (~finite) | delta.abs().gt(float(tolerance))
    matrix_admitted = frame["matrix_admitted"].fillna(False).astype(bool)
    replay_admitted = frame["replay_admitted"].fillna(False).astype(bool)
    frame["admission_mismatch"] = matrix_admitted.ne(replay_admitted)

    postprocessor_columns = {
        "v9_tail95_rank": ("v9_tail95_rank", "historical_rank", "policy_parent_rank"),
        "mlp_rank": ("mlp_rank", "rank_mlp_direct", "score_regime_calibrated"),
        "hierarchical_expected_ev": (
            "hierarchical_expected_ev",
            "expected_net_ev_after_1pct_mlp_direct",
            "expected_net_ev_after_1pct",
        ),
    }
    coverage: dict[str, int] = {}
    for metric, candidates in postprocessor_columns.items():
        column = next((name for name in candidates if name in frame.columns), None)
        coverage[metric] = (
            int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
            if column is not None
            else 0
        )
    summaries = [
        {
            "layer": "postprocessor_output_coverage",
            "matched_rows": int(len(frame)),
            "finite_counts": coverage,
            "mismatch_count": int(
                sum(int(count != len(frame)) for count in coverage.values())
            ),
            "pass": bool(coverage and all(count == len(frame) for count in coverage.values())),
        },
        {
            "layer": "fixed_ev_corrected_ev",
            "matched_rows": int(len(frame)),
            "matched_finite": int(finite.sum()),
            "max_abs_delta": float(delta.loc[finite].abs().max())
            if bool(finite.any())
            else None,
            "mean_abs_delta": float(delta.loc[finite].abs().mean())
            if bool(finite.any())
            else None,
            "mismatch_count": int(frame["corrected_ev_mismatch"].sum()),
            "first_divergence": (
                frame.loc[frame["corrected_ev_mismatch"], KEYS]
                .head(1)
                .to_dict("records")
                or [None]
            )[0],
            "pass": not bool(frame["corrected_ev_mismatch"].any()),
        },
        {
            "layer": "admission_decision",
            "matched_rows": int(len(frame)),
            "reference_selected": int(matrix_admitted.sum()),
            "replay_selected": int(replay_admitted.sum()),
            "mismatch_count": int(frame["admission_mismatch"].sum()),
            "first_divergence": (
                frame.loc[frame["admission_mismatch"], KEYS]
                .head(1)
                .to_dict("records")
                or [None]
            )[0],
            "pass": not bool(frame["admission_mismatch"].any()),
        },
    ]
    return frame, summaries


def _decision_context(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    context = candidates.reset_index(drop=True).reset_index(names="candidate_index")
    columns = [
        "candidate_index",
        "timestamp",
        "symbol",
        "side_name",
        "policy_archetype",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "net_return",
        "gross_return",
        "fee_return",
        "policy_size_multiplier",
        "simple_policy_exit_reason",
    ]
    columns = [
        column
        for column in columns
        if column in context.columns
        and (column == "candidate_index" or column not in decisions.columns)
    ]
    out = decisions.merge(context[columns], on="candidate_index", how="left")
    return _normalise_rows(out)


def replay_portfolio(
    exit_rows: pd.DataFrame,
    *,
    portfolio_config: Path,
    portfolio_ev_reference: Path,
) -> pd.DataFrame:
    candidates = _portfolio_candidates(exit_rows)
    decisions, _, _ = replay_candidates(
        candidates,
        load_portfolio_policy_params(portfolio_config),
        mode="global_auction",
        ev_curve=_load_ev_curve(portfolio_ev_reference),
        market_mode="perps",
    )
    return _decision_context(decisions, candidates)


def compare_portfolio_rows(
    reference: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = _normalise_rows(reference)
    right = _normalise_rows(replay)
    merged = left.merge(
        right,
        on=KEYS,
        how="outer",
        suffixes=("__reference", "__replay"),
        indicator=True,
    )
    both = merged["_merge"].eq("both")
    bool_mismatch = pd.Series(False, index=merged.index)
    reason_mismatch = pd.Series(False, index=merged.index)
    for name in ("accepted",):
        lcol, rcol = f"{name}__reference", f"{name}__replay"
        if lcol in merged and rcol in merged:
            bool_mismatch |= merged[lcol].fillna(False).astype(bool).ne(
                merged[rcol].fillna(False).astype(bool)
            )
    for name in ("rejection_reason", "position_exit_reason"):
        lcol, rcol = f"{name}__reference", f"{name}__replay"
        if lcol in merged and rcol in merged:
            reason_mismatch |= merged[lcol].fillna("").astype(str).ne(
                merged[rcol].fillna("").astype(str)
            )
    numeric_mismatch = pd.Series(False, index=merged.index)
    max_delta = 0.0
    for name in ("position_size", "position_net_return", "position_gross_return"):
        lcol, rcol = f"{name}__reference", f"{name}__replay"
        if lcol not in merged or rcol not in merged:
            continue
        delta = (
            pd.to_numeric(merged[rcol], errors="coerce")
            - pd.to_numeric(merged[lcol], errors="coerce")
        ).abs()
        left_value = pd.to_numeric(merged[lcol], errors="coerce")
        right_value = pd.to_numeric(merged[rcol], errors="coerce")
        both_missing = left_value.isna() & right_value.isna()
        numeric_mismatch |= (~both_missing) & delta.fillna(np.inf).gt(tolerance)
        if delta.notna().any():
            max_delta = max(max_delta, float(delta.max()))
    mismatch = ~both | bool_mismatch | reason_mismatch | numeric_mismatch
    merged["portfolio_mismatch"] = mismatch
    reference_accepted = pd.Series(
        merged.get("accepted__reference", False), index=merged.index
    ).fillna(False).astype(bool)
    replay_accepted = pd.Series(
        merged.get("accepted__replay", False), index=merged.index
    ).fillna(False).astype(bool)
    summary = {
        "layer": "portfolio_selection",
        "reference_rows": int(merged["_merge"].isin(["both", "left_only"]).sum()),
        "replay_rows": int(merged["_merge"].isin(["both", "right_only"]).sum()),
        "matched_rows": int(both.sum()),
        "reference_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "replay_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "reference_accepted": int(reference_accepted.sum()),
        "replay_accepted": int(replay_accepted.sum()),
        "accepted_row_mismatch_count": int(
            (both & reference_accepted.ne(replay_accepted)).sum()
        ),
        "rejection_reason_mismatch_count": int((both & reason_mismatch).sum()),
        "max_abs_delta": max_delta,
        "mismatch_count": int(mismatch.sum()),
        "first_divergence": (merged.loc[mismatch, KEYS].head(1).to_dict("records") or [None])[0],
        "pass": not bool(mismatch.any()),
    }
    return merged, summary


def compare_close_rows(
    reference: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = _normalise_rows(reference)
    right = _normalise_rows(replay)
    merged = left.merge(
        right,
        on=KEYS,
        how="outer",
        suffixes=("__reference", "__replay"),
        validate="one_to_one",
        indicator=True,
    )
    both = merged["_merge"].eq("both")
    reason_mismatch = pd.Series(False, index=merged.index)
    for name in ("simple_policy_exit_reason", "execution_policy_key"):
        lcol, rcol = f"{name}__reference", f"{name}__replay"
        if lcol in merged and rcol in merged:
            reason_mismatch |= merged[lcol].fillna("").astype(str).ne(
                merged[rcol].fillna("").astype(str)
            )
    timestamp_mismatch = pd.Series(False, index=merged.index)
    for name in ("exit_timestamp",):
        lcol, rcol = f"{name}__reference", f"{name}__replay"
        if lcol in merged and rcol in merged:
            timestamp_mismatch |= pd.to_datetime(
                merged[lcol], utc=True, errors="coerce"
            ).ne(
                pd.to_datetime(merged[rcol], utc=True, errors="coerce")
            )
    max_delta = 0.0
    numeric_mismatch = pd.Series(False, index=merged.index)
    numeric_metrics: dict[str, dict[str, Any]] = {}
    for name in (
        "entry_price",
        "exit_price",
        "holding_bars",
        "gross_return",
        "fee_return",
        "net_return",
        "policy_size_multiplier",
    ):
        lcol, rcol = f"{name}__reference", f"{name}__replay"
        if lcol not in merged or rcol not in merged:
            continue
        delta = (
            pd.to_numeric(merged[rcol], errors="coerce")
            - pd.to_numeric(merged[lcol], errors="coerce")
        ).abs()
        left_value = pd.to_numeric(merged[lcol], errors="coerce")
        right_value = pd.to_numeric(merged[rcol], errors="coerce")
        both_missing = left_value.isna() & right_value.isna()
        metric_mismatch = both & (~both_missing) & delta.fillna(np.inf).gt(tolerance)
        numeric_mismatch |= metric_mismatch
        if delta.notna().any():
            max_delta = max(max_delta, float(delta.max()))
        finite = both & left_value.notna() & right_value.notna()
        numeric_metrics[name] = {
            "matched_finite": int(finite.sum()),
            "max_abs_delta": (
                float(delta.loc[finite].max()) if bool(finite.any()) else None
            ),
            "mean_abs_delta": (
                float(delta.loc[finite].mean()) if bool(finite.any()) else None
            ),
            "mismatch_count": int(metric_mismatch.sum()),
        }
    mismatch = ~both | reason_mismatch | timestamp_mismatch | numeric_mismatch
    merged["close_mismatch"] = mismatch
    summary = {
        "layer": "exit_decision_and_net_pnl",
        "reference_positions": int(
            merged["_merge"].isin(["both", "left_only"]).sum()
        ),
        "replay_positions": int(
            merged["_merge"].isin(["both", "right_only"]).sum()
        ),
        "matched_positions": int(both.sum()),
        "reference_only_positions": int(merged["_merge"].eq("left_only").sum()),
        "replay_only_positions": int(merged["_merge"].eq("right_only").sum()),
        "close_reason_mismatch_count": int((both & reason_mismatch).sum()),
        "exit_timestamp_mismatch_count": int((both & timestamp_mismatch).sum()),
        "numeric_metrics": numeric_metrics,
        "max_abs_delta": max_delta,
        "mismatch_count": int(mismatch.sum()),
        "first_divergence": (merged.loc[mismatch, KEYS].head(1).to_dict("records") or [None])[0],
        "pass": not bool(mismatch.any()),
    }
    return merged, summary


def audit_policy_execution_contract(
    *,
    admission_policy_path: Path,
    portfolio_config_path: Path,
    exit_policy_dir: Path,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Audit the frozen policy contract without mutating or promoting it."""
    admission = _read_json(admission_policy_path)
    portfolio = _read_json(portfolio_config_path)
    selection = portfolio.get("selection")
    if not isinstance(selection, Mapping):
        selection = portfolio
    concurrency = portfolio.get("concurrency")
    if not isinstance(concurrency, Mapping):
        concurrency = {}

    parent_path = _policy_summary_path(exit_policy_dir, "side_parent_policy_summary")
    local_path = _policy_summary_path(exit_policy_dir, "side_archetype_policy_summary")
    parent = pd.read_csv(parent_path)
    local = pd.read_csv(local_path)
    parent_sides = set(parent.get("side", pd.Series(dtype=str)).astype(str).str.lower())
    local_sides = set(local.get("side", pd.Series(dtype=str)).astype(str).str.lower())
    local_archetypes = int(
        local.get("policy_archetype", pd.Series(dtype=str)).astype(str).nunique()
    )

    checks = {
        "admission_policy_id": str(admission.get("policy_id") or "")
        == CANONICAL_ADMISSION_POLICY_ID,
        "admission_family": str(admission.get("family") or "")
        == "side_archetype_expected_ev_recent_correction",
        "admission_selection_mode": str(admission.get("selection_mode") or "")
        == "fixed_corrected_ev_threshold",
        "admission_window_days": int(admission.get("window_days") or 0) == 21,
        "admission_symmetric_trim": abs(
            float(admission.get("robust_daily_residual_trim_fraction") or 0.0)
            - 0.10
        )
        <= tolerance,
        "admission_fixed_net_ev": abs(
            float(admission.get("fixed_target_net_ev") or 0.0) - 0.007
        )
        <= tolerance,
        "admission_horizon_hours": int(admission.get("outcome_horizon_hours") or 0)
        == 12,
        "portfolio_policy": str(portfolio.get("portfolio_policy_version") or "")
        == "global_auction_v1",
        "portfolio_postprocessor": str(
            portfolio.get("regime_ev_calibration_policy_id")
            or selection.get("regime_ev_calibration_policy_id")
            or ""
        )
        == CANONICAL_POSTPROCESSOR_POLICY_ID,
        "portfolio_max_new_entries": int(
            concurrency.get("max_new_entries_per_bar") or 0
        )
        == 2,
        "portfolio_max_concurrent": int(
            concurrency.get("max_concurrent_positions") or 0
        )
        == 8,
        "exit_parent_sides": parent_sides == {"long", "short"},
        "exit_local_sides": local_sides == {"long", "short"},
        "exit_local_archetypes_present": local_archetypes > 0,
    }
    return {
        "layer": "policy_execution_contract",
        "admission_policy_id": admission.get("policy_id"),
        "postprocessor_policy_id": portfolio.get(
            "regime_ev_calibration_policy_id"
        ),
        "parent_geometry_rows": int(len(parent)),
        "local_geometry_rows": int(len(local)),
        "local_archetypes": local_archetypes,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "artifacts": {
            str(admission_policy_path): _sha256(admission_policy_path),
            str(portfolio_config_path): _sha256(portfolio_config_path),
            str(parent_path): _sha256(parent_path),
            str(local_path): _sha256(local_path),
        },
        "pass": bool(all(checks.values())),
    }


def materialize_exit_replay(
    candidates_path: Path,
    *,
    exit_policy_dir: Path,
    data_root: str,
    path_len: int,
    round_trip_cost_pct: float,
) -> pd.DataFrame:
    """Materialize current side/archetype exits directly from replay candidates."""
    rows = pd.read_parquet(candidates_path)
    required = {"timestamp", "symbol", "strategy_id", "rank_pct", "barrier_pct"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"exit candidates are missing {missing}")
    rows = rows.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol", "strategy_id"]).copy()
    rows["rank_pct"] = pd.to_numeric(rows["rank_pct"], errors="coerce")
    rows = rows.loc[rows["rank_pct"].notna()].copy()
    if "side_name" not in rows.columns:
        side = pd.to_numeric(rows.get("side"), errors="coerce")
        rows["side_name"] = np.where(side.lt(0.0), "short", "long")
    if "side" not in rows.columns:
        rows["side"] = np.where(rows["side_name"].eq("short"), -1.0, 1.0)
    rows["symbol"] = rows["symbol"].astype(str)
    rows["strategy_id"] = rows["strategy_id"].astype(str)
    rows = rows.sort_values(["strategy_id", "timestamp", "symbol"], kind="stable")
    bundles = _load_bundles(
        rows,
        data_root=str(data_root),
        market_mode="perps",
        path_len=int(path_len),
        min_rows_per_strategy=1,
    )
    parent = pd.read_csv(_policy_summary_path(exit_policy_dir, "side_parent_policy_summary"))
    local = pd.read_csv(_policy_summary_path(exit_policy_dir, "side_archetype_policy_summary"))
    replay = _materialize_exit_rows(
        bundles,
        parent_summary=parent,
        archetype_summary=local,
        cost_pct=float(round_trip_cost_pct) / 2.0,
    )
    return _apply_policy_spread_to_returns(replay)


def cost_reconciliation(rows: pd.DataFrame, *, tolerance: float = 2e-6) -> dict[str, Any]:
    def numeric_series(name: str, default: float = np.nan) -> pd.Series:
        return pd.to_numeric(
            rows.get(name, pd.Series(default, index=rows.index)),
            errors="coerce",
        )

    gross = numeric_series("gross_return")
    net = numeric_series("net_return")
    fee = numeric_series("fee_return")
    if fee.notna().sum() == 0:
        fee = gross - net
        fee_source = "derived_gross_minus_net"
    else:
        fee_source = "explicit_fee_return"
    residual = (gross - fee - net).abs()
    has_spread_marker = "policy_spread_applied_to_returns" in rows.columns
    has_embedded_marker = "policy_spread_embedded_in_executable_prices" in rows.columns
    legacy_double_spread = pd.Series(
        rows.get("policy_spread_applied_to_returns", False), index=rows.index
    ).fillna(False).astype(bool)
    embedded_spread = pd.Series(
        rows.get("policy_spread_embedded_in_executable_prices", False),
        index=rows.index,
    ).fillna(False).astype(bool)
    spread_provenance_known = bool(
        (has_spread_marker and not legacy_double_spread.any())
        or (has_embedded_marker and embedded_spread.all())
    )
    spread_bps = numeric_series("spread_cost_bps", 0.0).fillna(0.0)
    spread_bps += numeric_series("exit_spread_cost_bps", 0.0).fillna(0.0)
    return {
        "rows": int(len(rows)),
        "fee_source": fee_source,
        "gross_minus_fee_minus_net_max_abs": float(residual.max()),
        "gross_minus_fee_minus_net_mean_abs": float(residual.mean()),
        "legacy_double_spread_rows": int(legacy_double_spread.sum()),
        "legacy_double_spread_mean_bps": float(spread_bps.loc[legacy_double_spread].mean())
        if bool(legacy_double_spread.any())
        else 0.0,
        "spread_provenance": (
            "embedded_in_executable_prices"
            if has_embedded_marker and embedded_spread.all()
            else "explicit_post_simulator_deduction"
            if has_spread_marker and legacy_double_spread.any()
            else "explicit_no_post_simulator_deduction"
            if has_spread_marker
            else "legacy_unverifiable"
        ),
        "spread_provenance_known": spread_provenance_known,
        "pass": bool(
            residual.max() <= tolerance
            and not legacy_double_spread.any()
            and spread_provenance_known
        ),
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = ["# Policy And Execution Replay Parity", ""]
    lines.append(f"Overall pass: **{bool(report.get('pass'))}**")
    lines.append("")
    lines.append("| Layer | Pass | Matched | Mismatches | Max delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for layer in report.get("layers", []):
        lines.append(
            "| {layer} | {passed} | {matched} | {mismatch} | {delta} |".format(
                layer=layer.get("layer"),
                passed=layer.get("pass"),
                matched=layer.get("matched_rows", layer.get("matched_positions", "")),
                mismatch=layer.get("mismatch_count", ""),
                delta=layer.get("max_abs_delta", ""),
            )
        )
    lines.extend(["", "## Costs", "", f"`{json.dumps(report.get('cost_reconciliation', {}), sort_keys=True)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-policy-rows", type=Path, default=None)
    parser.add_argument("--replay-policy-rows", type=Path, default=None)
    parser.add_argument("--fixed-ev-comparison-rows", type=Path, default=None)
    parser.add_argument("--policy-input-rows", type=Path, default=None)
    parser.add_argument("--predecessor-bundle", type=Path, default=None)
    parser.add_argument("--residual-event-state", type=Path, default=None)
    parser.add_argument("--regime-ev-artifact", type=Path, default=None)
    parser.add_argument("--admission-policy", type=Path, default=None)
    parser.add_argument("--replay-exit-rows", type=Path, default=None)
    parser.add_argument("--exit-candidates", type=Path, default=None)
    parser.add_argument("--exit-policy-dir", type=Path, default=None)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    parser.add_argument("--reference-exit-rows", type=Path, default=None)
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--portfolio-ev-reference", type=Path, required=True)
    parser.add_argument("--reference-portfolio-decisions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=NUMERIC_TOLERANCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers: list[dict[str, Any]] = []
    if args.fixed_ev_comparison_rows is not None:
        policy_detail, policy_layers = summarize_fixed_ev_policy_rows(
            pd.read_parquet(args.fixed_ev_comparison_rows),
            tolerance=float(args.tolerance),
        )
        replay_policy = policy_detail
        layers.extend(policy_layers)
    elif args.reference_policy_rows is None:
        raise ValueError(
            "provide --fixed-ev-comparison-rows or --reference-policy-rows"
        )
    elif args.policy_input_rows is not None:
        reference_policy = pd.read_parquet(args.reference_policy_rows)
        required = (
            args.predecessor_bundle,
            args.residual_event_state,
            args.regime_ev_artifact,
            args.admission_policy,
        )
        if any(path is None for path in required):
            raise ValueError("policy replay requires all V9/MLP/admission artifacts")
        replay_policy = replay_policy_chain(
            pd.read_parquet(args.policy_input_rows),
            predecessor_bundle=args.predecessor_bundle,
            residual_event_state=args.residual_event_state,
            regime_ev_artifact=args.regime_ev_artifact,
            admission_policy=args.admission_policy,
        )
        policy_detail, policy_layers = compare_policy_rows(
            reference_policy, replay_policy, tolerance=float(args.tolerance)
        )
        layers.extend(policy_layers)
    elif args.replay_policy_rows is not None:
        reference_policy = pd.read_parquet(args.reference_policy_rows)
        replay_policy = pd.read_parquet(args.replay_policy_rows)
        policy_detail, policy_layers = compare_policy_rows(
            reference_policy, replay_policy, tolerance=float(args.tolerance)
        )
        layers.extend(policy_layers)
    else:
        raise ValueError("provide --policy-input-rows or --replay-policy-rows")
    policy_detail.to_parquet(
        args.output_dir / "policy_row_comparison.parquet",
        index=False,
        compression="zstd",
    )
    replay_policy.to_parquet(
        args.output_dir / "replayed_policy_rows.parquet",
        index=False,
        compression="zstd",
    )

    if args.replay_exit_rows is not None:
        exit_rows = pd.read_parquet(args.replay_exit_rows)
    elif args.exit_candidates is not None and args.exit_policy_dir is not None:
        exit_rows = materialize_exit_replay(
            args.exit_candidates,
            exit_policy_dir=args.exit_policy_dir,
            data_root=str(args.data_root),
            path_len=int(args.path_len),
            round_trip_cost_pct=float(args.round_trip_cost_pct),
        )
        exit_rows.to_parquet(
            args.output_dir / "replayed_exit_rows.parquet",
            index=False,
            compression="zstd",
        )
    else:
        raise ValueError(
            "provide --replay-exit-rows or both --exit-candidates and --exit-policy-dir"
        )
    replayed_portfolio = replay_portfolio(
        exit_rows,
        portfolio_config=args.portfolio_config,
        portfolio_ev_reference=args.portfolio_ev_reference,
    )
    replayed_portfolio.to_parquet(
        args.output_dir / "replayed_portfolio_decisions.parquet",
        index=False,
        compression="zstd",
    )
    if args.reference_portfolio_decisions is not None:
        reference_decisions = pd.read_parquet(args.reference_portfolio_decisions)
        reference_candidates = _portfolio_candidates(exit_rows)
        reference_decisions = _decision_context(reference_decisions, reference_candidates)
        portfolio_detail, portfolio_summary = compare_portfolio_rows(
            reference_decisions,
            replayed_portfolio,
            tolerance=float(args.tolerance),
        )
        portfolio_detail.to_parquet(
            args.output_dir / "portfolio_row_comparison.parquet",
            index=False,
            compression="zstd",
        )
        layers.append(portfolio_summary)

    if args.reference_exit_rows is not None:
        close_detail, close_summary = compare_close_rows(
            pd.read_parquet(args.reference_exit_rows),
            exit_rows,
            tolerance=float(args.tolerance),
        )
        close_detail.to_parquet(
            args.output_dir / "close_position_comparison.parquet",
            index=False,
            compression="zstd",
        )
        layers.append(close_summary)

    contract_audit = None
    if args.admission_policy is not None and args.exit_policy_dir is not None:
        contract_audit = audit_policy_execution_contract(
            admission_policy_path=args.admission_policy,
            portfolio_config_path=args.portfolio_config,
            exit_policy_dir=args.exit_policy_dir,
        )
        (args.output_dir / "policy_execution_contract_audit.json").write_text(
            json.dumps(_json_safe(contract_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        layers.append(contract_audit)

    cost = cost_reconciliation(exit_rows)
    artifact_paths: Iterable[Path | None] = (
        args.reference_policy_rows,
        args.fixed_ev_comparison_rows,
        args.replay_policy_rows,
        args.policy_input_rows,
        args.predecessor_bundle,
        args.residual_event_state,
        args.regime_ev_artifact,
        args.admission_policy,
        args.replay_exit_rows,
        args.exit_candidates,
        args.reference_exit_rows,
        args.portfolio_config,
        args.portfolio_ev_reference,
        args.reference_portfolio_decisions,
    )
    artifacts = {
        str(path): _sha256(path)
        for path in artifact_paths
        if path is not None
    }
    report = {
        "schema": "policy_execution_replay_parity_v1",
        "scope": "policy_and_execution_only; upstream base/meta parity not assumed",
        "layers": layers,
        "cost_reconciliation": cost,
        "contract_audit": contract_audit,
        "artifacts": artifacts,
        "pass": bool(all(layer.get("pass", False) for layer in layers) and cost["pass"]),
    }
    (args.output_dir / "parity_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "parity_report.md", report)
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
