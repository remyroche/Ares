"""Evaluation and advancement gates for the sequential target-stack funnel.

This module is intentionally model-agnostic: runners materialise one strict
candidate-level row per ``trial_id`` and this module audits its provenance,
forms the one pooled-global book, and records whether a trial may enter the
*next development stage*.  It never selects a production model and never
opens or re-ranks a held-out final period.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.candidate_evaluation import FRACTIONS, EvaluationError, evaluate_global_book


class SequentialFunnelEvaluationError(EvaluationError):
    """Raised when a trial cannot be evaluated under the frozen contract."""


STAGE_ORDER = ("target_screen", "certainty", "distillation_gam", "ranking", "archetype")
FUTURE_ONLY_TOKENS = ("teacher", "future", "mfe", "mae", "path", "terminal", "giveback", "retention")


@dataclass(frozen=True)
class FunnelColumns:
    trial: str = "trial_id"
    score: str = "score"
    net: str = "execution_net_ev_12h"
    gross: str = "execution_gross_ev_12h"
    cost: str = "execution_cost_return"
    decision: str = "__decision_ts__"
    label_available: str = "__label_available_at__"
    strict_oof: str = "strict_prequential_oof"


def _as_utc(frame: pd.DataFrame, column: str) -> pd.Series:
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise SequentialFunnelEvaluationError(f"{column} must contain valid UTC timestamps")
    return value


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna(default).astype(bool)


def _trial_metadata(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    trials = manifest.get("trials", [])
    if not isinstance(trials, list):
        raise SequentialFunnelEvaluationError("trial manifest 'trials' must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for trial in trials:
        if not isinstance(trial, Mapping) or not trial.get("trial_id"):
            raise SequentialFunnelEvaluationError("each trial manifest entry requires trial_id")
        key = str(trial["trial_id"])
        if key in result:
            raise SequentialFunnelEvaluationError(f"duplicate trial manifest entry: {key}")
        stage = str(trial.get("stage", ""))
        if stage not in STAGE_ORDER:
            raise SequentialFunnelEvaluationError(f"trial {key} has unknown stage {stage!r}")
        result[key] = trial
    return result


def validate_nested_oof_provenance(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    columns: FunnelColumns = FunnelColumns(),
) -> pd.DataFrame:
    """Audit common rows plus every enabled upstream prediction lineage.

    A missing enabled-layer lineage is a failure, rather than being silently
    treated as ordinary OOF.  This is the key distinction between a stacked
    OOF claim and an independently OOF final score.
    """
    metadata = _trial_metadata(manifest)
    required = {"candidate_id", columns.trial, columns.score, columns.net, columns.gross, columns.cost,
                columns.decision, columns.label_available, columns.strict_oof}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SequentialFunnelEvaluationError(f"trial rows lack required fields: {missing}")
    work = frame.copy()
    work[columns.decision] = _as_utc(work, columns.decision)
    work[columns.label_available] = _as_utc(work, columns.label_available)
    rows: list[dict[str, Any]] = []
    trial_values = set(work[columns.trial].astype(str))
    rows.append({"check": "every_trial_declared", "passed": trial_values == set(metadata),
                 "value": json.dumps({"rows": sorted(trial_values), "manifest": sorted(metadata)}, sort_keys=True)})
    rows.append({"check": "one_score_per_candidate_trial", "passed": not work.duplicated(["candidate_id", columns.trial]).any(),
                 "value": str(int(work.duplicated(["candidate_id", columns.trial]).sum()))})
    identities = {trial: tuple(sorted(group.candidate_id.astype(str))) for trial, group in work.groupby(columns.trial, sort=True)}
    first = next(iter(identities.values()), ())
    rows.append({"check": "identical_candidate_ids_across_trials", "passed": all(ids == first for ids in identities.values()),
                 "value": json.dumps({trial: len(ids) for trial, ids in identities.items()}, sort_keys=True)})
    rows.append({"check": "h12_label_availability", "passed": bool(work[columns.label_available].ge(work[columns.decision] + pd.Timedelta(hours=12)).all()),
                 "value": "label_available_ts >= decision_ts + 12h"})
    rows.append({"check": "strict_final_score_oof", "passed": bool(_bool(work, columns.strict_oof).all()), "value": str(int(len(work)))})
    gross = pd.to_numeric(work[columns.gross], errors="coerce")
    cost = pd.to_numeric(work[columns.cost], errors="coerce")
    net = pd.to_numeric(work[columns.net], errors="coerce")
    rows.append({"check": "exact_net_cost_once", "passed": bool(np.isfinite(gross).all() and np.isfinite(cost).all() and np.isfinite(net).all() and np.allclose(gross - cost, net, atol=1e-7, rtol=0.0)),
                 "value": "gross - row_cost == net"})
    for trial, spec in metadata.items():
        local = work.loc[work[columns.trial].astype(str).eq(trial)]
        # The Round-1 report intentionally contains base-only, meta-only and
        # base-plus-meta views.  A missing layer is legitimate only when that
        # view explicitly declares it disabled; do not make a fake zero-model
        # lineage just to satisfy the audit.
        enabled = {"gam": bool(spec.get("uses_gam", False)), "teacher": bool(spec.get("uses_teacher", False)),
                   "base": bool(spec.get("uses_base", True)), "archetype": bool(spec.get("uses_archetypes", False)),
                   "meta": bool(spec.get("uses_meta", True))}
        for layer, required_layer in enabled.items():
            if not required_layer:
                continue
            fit = f"{layer}_prediction_fit_end_ts"
            generated = f"{layer}_prediction_generated_ts"
            model = f"{layer}_prediction_model_id"
            fold = f"{layer}_prediction_fold_id"
            fields = (fit, generated, model, fold)
            if any(name not in local for name in fields):
                rows.append({"trial_id": trial, "check": f"nested_oof_{layer}_lineage", "passed": False,
                             "value": f"missing fields: {sorted(name for name in fields if name not in local)}"})
                continue
            fit_ts, generated_ts = _as_utc(local, fit), _as_utc(local, generated)
            passed = bool(fit_ts.lt(local[columns.decision]).all() and generated_ts.le(local[columns.decision]).all()
                          and local[model].notna().all() and local[fold].notna().all())
            rows.append({"trial_id": trial, "check": f"nested_oof_{layer}_lineage", "passed": passed,
                         "value": "fit_end < decision; generated <= decision; model_id/fold_id present"})
        features = [str(x).lower() for x in spec.get("inference_features", [])]
        banned = sorted({feature for feature in features if any(token in feature for token in FUTURE_ONLY_TOKENS)})
        rows.append({"trial_id": trial, "check": "future_outputs_excluded_from_inference", "passed": not banned,
                     "value": json.dumps(banned)})
    return pd.DataFrame(rows)


def evaluate_funnel_trials(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    columns: FunnelColumns = FunnelColumns(),
    unit: str = "return",
    fractions: Sequence[float] = FRACTIONS,
) -> Mapping[str, pd.DataFrame]:
    """Evaluate frozen trial scores and calculate non-promotional gates."""
    checks = validate_nested_oof_provenance(frame, manifest, columns=columns)
    if not checks.passed.all():
        raise SequentialFunnelEvaluationError("sequential funnel provenance failed; inspect correctness_checks")
    metadata = _trial_metadata(manifest)
    work = frame.copy()
    work["__ts__"] = _as_utc(work, columns.decision)
    tails, attributions, diagnostics = [], [], []
    for trial, spec in metadata.items():
        local = work.loc[work[columns.trial].astype(str).eq(trial)].copy()
        trial_tails, trial_attr = evaluate_global_book(
            local, score_column=columns.score, net_column=columns.net, net_unit=unit,
            gross_column=columns.gross, gross_unit=unit, cost_column=columns.cost, cost_unit=unit,
            fractions=fractions, regime_column="regime" if "regime" in local else None,
            liquidity_column="liquidity" if "liquidity" in local else None,
            hurdle_column="opportunity_hurdle" if "opportunity_hurdle" in local else None,
        )
        trial_tails.insert(0, "trial_id", trial)
        trial_tails.insert(1, "stage", spec["stage"])
        trial_tails["description"] = str(spec.get("description", ""))
        tails.append(trial_tails)
        for dimension, table in trial_attr.items():
            if not table.empty:
                attributions.append(table.assign(trial_id=trial, stage=spec["stage"], attribution_scope=dimension))
        # Target metrics remain diagnostics and are runner-provided rather
        # than inferred from economic outcomes.
        target_metrics = {key: value for key, value in spec.get("target_metrics", {}).items() if isinstance(value, (int, float, str, bool))}
        diagnostics.append({"trial_id": trial, "stage": spec["stage"], "target_family": spec.get("target_family"),
                            "development_only": bool(spec.get("development_only", True)), **target_metrics})
    tail_frame = pd.concat(tails, ignore_index=True)
    attr_frame = pd.concat(attributions, ignore_index=True) if attributions else pd.DataFrame()
    decisions: list[dict[str, Any]] = []
    for trial, spec in metadata.items():
        local = tail_frame.loc[tail_frame.trial_id.eq(trial)].set_index("top_fraction")
        top10 = local.loc[.10] if .10 in local.index else pd.Series(dtype=float)
        side = attr_frame.loc[(attr_frame.trial_id.eq(trial)) & (attr_frame.attribution_scope.eq("side")) & (attr_frame.top_fraction.eq(.10))] if not attr_frame.empty else pd.DataFrame()
        month = attr_frame.loc[(attr_frame.trial_id.eq(trial)) & (attr_frame.attribution_scope.eq("month")) & (attr_frame.top_fraction.eq(.10))] if not attr_frame.empty else pd.DataFrame()
        side_ok = bool(not side.empty and (side.net_bps >= 0.0).all())
        latest_ok = bool(not month.empty and float(month.loc[month.dimension_value.eq(month.dimension_value.max()), "net_bps"].iloc[0]) >= 0.0)
        top10_net = float(top10.get("net_bps", np.nan))
        top10_gross = float(top10.get("gross_bps", np.nan))
        # The roadmap distinguishes research advancement (positive *gross*
        # top-10 plus robust attribution) from execution readiness (positive
        # *net* top-10).  Conflating those gates would suppress a useful
        # diagnosis of a learnable signal whose economics are still swallowed
        # by cost.  Final-OOS arms are never used to open another search.
        development_only = bool(spec.get("development_only", True))
        passed = bool(top10_gross > 0.0 and side_ok and latest_ok and development_only)
        decisions.append({"trial_id": trial, "stage": spec["stage"], "target_family": spec.get("target_family"),
                          "pooled_top10_gross_bps": top10_gross, "pooled_top10_net_bps": top10_net,
                          "both_sides_nonnegative_top10": side_ok, "latest_month_nonnegative_top10": latest_ok,
                          "development_only": development_only,
                          "may_advance_to_next_development_stage": passed,
                          "execution_readiness_net_top10": bool(top10_net > 0.0),
                          "terminal_decision": "CANDIDATE_FOR_NEXT_DEVELOPMENT_STAGE" if passed else "DOES_NOT_ADVANCE"})
    return {"correctness_checks": checks, "base_meta_stack_results": tail_frame,
            "base_meta_stack_attribution": attr_frame, "trial_target_diagnostics": pd.DataFrame(diagnostics),
            "sequential_advancement_gates": pd.DataFrame(decisions)}


def render_trial_report(tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> str:
    """Render a concise, complete Markdown report without selecting a winner."""
    metadata = _trial_metadata(manifest)
    checks = tables["correctness_checks"]
    gates = tables["sequential_advancement_gates"].set_index("trial_id")
    tails = tables["base_meta_stack_results"]
    lines = ["# Sequential funnel trial report", "", "Research-only: advancement is a development gate, never a production promotion.", "",
             f"Correctness checks: **{int(checks.passed.sum())}/{len(checks)} passed**.", ""]
    for trial, spec in metadata.items():
        gate = gates.loc[trial]
        lines.extend([f"## {trial}", "", f"Stage: `{spec['stage']}`  ", f"Target family: `{spec.get('target_family', 'unspecified')}`  ",
                      f"Description: {spec.get('description', 'No description supplied.')}", "",
                      "| Tail | Gross bps | Cost bps | Net bps | Rows |", "|---|---:|---:|---:|---:|"])
        for _, row in tails.loc[tails.trial_id.eq(trial)].sort_values("top_fraction").iterrows():
            lines.append(f"| top {row.top_fraction:.0%} | {row.gross_bps:.2f} | {row.cost_bps:.2f} | {row.net_bps:.2f} | {int(row.selected_rows)} |")
        lines.extend(["", f"Advancement: **{gate.terminal_decision}**. Top-10 net {gate.pooled_top10_net_bps:.2f} bps; both-side gate={bool(gate.both_sides_nonnegative_top10)}; latest-month gate={bool(gate.latest_month_nonnegative_top10)}.", ""])
    return "\n".join(lines)
