#!/usr/bin/env python3
"""Run the diagnostic-only target/population oracle ladder on the canonical ledger.

The runner deliberately ranks candidates *globally* at every top-k fraction.
Slice rows (side, month, symbol and regime) describe the globally selected
book; they never re-rank candidates inside a timestamp or slice.

This is a diagnostic, not a training or promotion runner.  Its explicit
unavailability records are intentional: frozen one-minute paths stop at the
original H12 horizon, so a causal H12 return after a delayed entry cannot be
reconstructed without an additional immutable tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = Path("data_perp/artifacts")
DEFAULT_LEDGER = ART / "root_cause_diagnostic_substrate_20260731_v4" / "diagnostic_row_ledger.parquet"
DEFAULT_SUBSTRATE_MANIFEST = ART / "root_cause_diagnostic_substrate_20260731_v4" / "diagnostic_population_manifest.json"
DEFAULT_OUTPUT = ART / "root_cause_oracle_ladder_20260731_v7"
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
BOOTSTRAP_REPS = 500
SEED = 20260731


class ContractError(RuntimeError):
    """The canonical diagnostic inputs have lost their exact-row contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def _atomic_dir(output: Path) -> Path:
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def _finalise(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable diagnostic artifact: {output}")
    staging.rename(output)


def _top_selected(frame: pd.DataFrame, score_col: str, fraction: float) -> pd.DataFrame:
    """Select a global, deterministic top-k book, never a timestamp-local one."""
    count = max(1, int(math.ceil(len(frame) * fraction)))
    ordered = frame.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="mergesort")
    return ordered.iloc[:count].copy()


def _state_score(series: pd.Series, favourable: set[str]) -> np.ndarray:
    return series.astype(str).isin(favourable).astype(float).to_numpy()


def _daily_shuffle(frame: pd.DataFrame, target_col: str, seed: int) -> np.ndarray:
    """Break row-level target association while preserving each UTC-day distribution."""
    rng = np.random.default_rng(seed)
    answer = np.empty(len(frame), dtype=float)
    # ``sort=False`` preserves the ledger order, so this is byte-stable.
    for _, index in frame.groupby("utc_day", sort=False).groups.items():
        positions = np.asarray(index, dtype=int)
        answer[positions] = rng.permutation(frame.loc[positions, target_col].to_numpy(float))
    return answer


def _add_scores(ledger: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frame = ledger.copy()
    # Timestamp is already canonical UTC.  String slicing is substantially
    # cheaper than ``dt.strftime`` at this scale while retaining its exact
    # calendar semantics for a UTC timestamp.
    decision_strings = frame.decision_ts.astype(str)
    frame["utc_day"] = decision_strings.str.slice(0, 10)
    frame["month"] = decision_strings.str.slice(0, 7)

    # These priors are deliberately hindsight null controls: they are a
    # ceiling for a coarse group lookup, not a deployable estimator.
    frame["score_o0_constant"] = 0.0
    frame["score_o0_side_month_hindsight_prior"] = frame.groupby(["side", "month"], observed=True).net_h12_bps.transform("mean")
    frame["score_o0_day_grouped_shuffled_gross"] = _daily_shuffle(frame, "gross_h12_bps", SEED)
    frame["score_o1_realised_gross_h12"] = frame.gross_h12_bps
    frame["score_o2_realised_net_h12"] = frame.net_h12_bps
    frame["score_o3_clean_before_adverse"] = _state_score(frame.postcost_h0_event, {"clear_cost_first"})
    frame["score_o3_retained_given_clear"] = _state_score(frame.postcost_h0_four_state, {"clear_then_retained"})
    frame["score_o3_state_ladder"] = np.select(
        [
            frame.postcost_h0_four_state.eq("clear_then_retained"),
            frame.postcost_h0_four_state.eq("clear_then_giveback"),
            frame.postcost_h0_event.eq("timeout"),
        ],
        [3.0, 2.0, 1.0],
        default=0.0,
    )
    frame["o4_permitted_action_gross_bps"] = frame.gross_h12_bps
    action_rows = frame.action_continue_execution_adjusted_gross_bps.notna()
    frame.loc[action_rows, "o4_permitted_action_gross_bps"] = np.maximum(
        frame.loc[action_rows, "action_continue_execution_adjusted_gross_bps"],
        frame.loc[action_rows, "action_exit_execution_adjusted_gross_bps"],
    )
    frame["score_o4_hindsight_permitted_action"] = frame.o4_permitted_action_gross_bps
    # Exact-ID frozen OOF score controls.  They are not current live scores;
    # their only role here is an apples-to-apples regret comparator.
    frame["score_current_base_alpha"] = frame.score_base_alpha
    frame["score_current_residual_alpha"] = frame.score_residual_alpha
    frame["score_current_base_plus_residual_delta"] = frame.score_base_alpha + frame.score_residual_delta_alpha

    definitions = [
        {"name": "O0_constant", "column": "score_o0_constant", "kind": "NULL_CONTROL", "target": "none"},
        {"name": "O0_side_month_hindsight_prior", "column": "score_o0_side_month_hindsight_prior", "kind": "HINDSIGHT_NULL_CONTROL", "target": "net_h12"},
        {"name": "O0_utc_day_grouped_shuffled_gross", "column": "score_o0_day_grouped_shuffled_gross", "kind": "SHUFFLED_LABEL_CONTROL", "target": "gross_h12"},
        {"name": "O1_realised_gross_h12", "column": "score_o1_realised_gross_h12", "kind": "OUTCOME_ORACLE", "target": "gross_h12"},
        {"name": "O2_realised_net_h12", "column": "score_o2_realised_net_h12", "kind": "OUTCOME_ORACLE", "target": "net_h12"},
        {"name": "O3_clean_before_adverse", "column": "score_o3_clean_before_adverse", "kind": "STATE_ORACLE", "target": "postcost_h0_event"},
        {"name": "O3_retained_given_clear", "column": "score_o3_retained_given_clear", "kind": "STATE_ORACLE", "target": "postcost_h0_four_state"},
        {"name": "O3_clean_adverse_timeout_state_ladder", "column": "score_o3_state_ladder", "kind": "STATE_ORACLE", "target": "postcost states"},
        {"name": "O4_hindsight_permitted_action", "column": "score_o4_hindsight_permitted_action", "kind": "POLICY_ORACLE", "target": "permitted action gross"},
        {"name": "CURRENT_base_alpha_OOF", "column": "score_current_base_alpha", "kind": "CURRENT_OOF_REFERENCE", "target": "alpha"},
        {"name": "CURRENT_residual_alpha_OOF", "column": "score_current_residual_alpha", "kind": "CURRENT_OOF_REFERENCE", "target": "alpha"},
        {"name": "CURRENT_base_plus_residual_delta_OOF", "column": "score_current_base_plus_residual_delta", "kind": "CURRENT_OOF_REFERENCE", "target": "alpha"},
    ]
    return frame, definitions


def _metrics_for_slice(selected: pd.DataFrame, evaluation_col: str, net_available: bool) -> dict[str, float | int | str]:
    gross = selected[evaluation_col].to_numpy(float)
    normal_gross = selected.gross_h12_bps.to_numpy(float)
    answer: dict[str, float | int | str] = {
        "candidate_support": int(len(selected)),
        "mean_evaluation_gross_bps": float(np.mean(gross)),
        "median_evaluation_gross_bps": float(np.median(gross)),
        "mean_entry_execution_adjusted_gross_bps": float(np.mean(normal_gross)),
        "gross_positive_rate": float(np.mean(gross > 0.0)),
        "gross_sum_bps": float(np.sum(gross)),
        "net_status": "AVAILABLE" if net_available else "NOT_AVAILABLE_ACTION_PATH_COST_NOT_CAUSALLY_RECONSTRUCTED",
        "mean_net_bps": float(np.mean(selected.net_h12_bps)) if net_available else np.nan,
        "median_net_bps": float(np.median(selected.net_h12_bps)) if net_available else np.nan,
        "net_positive_rate": float(np.mean(selected.net_h12_bps > 0.0)) if net_available else np.nan,
    }
    return answer


def _slice_rows(selected: pd.DataFrame, evaluation_col: str, net_available: bool) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "pooled", "ALL", selected
    for field, label in (("side", "side"), ("month", "month"), ("symbol", "symbol"), ("policy_archetype", "regime")):
        for value, part in selected.groupby(field, observed=True, sort=True):
            yield label, str(value), part


def _make_results(frame: pd.DataFrame, definitions: list[dict[str, str]]) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, float], pd.DataFrame] = {}
    for definition in definitions:
        evaluation_col = "o4_permitted_action_gross_bps" if definition["name"] == "O4_hindsight_permitted_action" else "gross_h12_bps"
        net_available = definition["name"] != "O4_hindsight_permitted_action"
        for fraction in TOP_FRACTIONS:
            selected = _top_selected(frame, definition["column"], fraction)
            selections[(definition["name"], fraction)] = selected
            # This can be a 55k-row book at top-20.  Compute its identity
            # once, rather than re-hashing the same candidate list for every
            # descriptive slice (which is needlessly quadratic in IO/CPU).
            selected_digest = _id_digest(selected.candidate_id)
            for slice_kind, slice_value, part in _slice_rows(selected, evaluation_col, net_available):
                row: dict[str, Any] = {
                    "oracle": definition["name"],
                    "oracle_kind": definition["kind"],
                    "score_target": definition["target"],
                    "selection_scope": "GLOBAL_TOP_K",  # never per timestamp
                    "top_fraction": fraction,
                    "slice_kind": slice_kind,
                    "slice_value": slice_value,
                    "evaluation_gross_definition": evaluation_col,
                    "selected_candidate_id_sha256": selected_digest,
                }
                row.update(_metrics_for_slice(part, evaluation_col, net_available))
                rows.append(row)
    return pd.DataFrame(rows), selections


def _paired_day_bootstrap(
    selections: dict[tuple[str, float], pd.DataFrame],
    reference_name: str,
    reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    """Paired UTC-day bootstrap of entry gross vs the frozen OOF reference."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        reference = selections[(reference_name, fraction)].groupby("utc_day", observed=True).gross_h12_bps.mean()
        for (oracle, oracle_fraction), selected in selections.items():
            if oracle_fraction != fraction or oracle == reference_name:
                continue
            # Action oracle gets a policy-gross comparison explicitly labelled
            # as such; all others use entry execution-adjusted gross.
            value = selected.o4_permitted_action_gross_bps if oracle == "O4_hindsight_permitted_action" else selected.gross_h12_bps
            trial = pd.DataFrame({"utc_day": selected.utc_day, "value": value}).groupby("utc_day", observed=True).value.mean()
            days = reference.index.intersection(trial.index).sort_values()
            if len(days) < 2:
                continue
            diff = (trial.loc[days] - reference.loc[days]).to_numpy(float)
            indices = rng.integers(0, len(diff), size=(reps, len(diff)))
            samples = diff[indices].mean(axis=1)
            rows.append({
                "oracle": oracle,
                "reference": reference_name,
                "top_fraction": fraction,
                "paired_utc_days": int(len(days)),
                "bootstrap_reps": reps,
                "mean_difference_bps": float(diff.mean()),
                "ci_low_bps": float(np.quantile(samples, 0.025)),
                "ci_high_bps": float(np.quantile(samples, 0.975)),
                "probability_positive": float(np.mean(samples > 0.0)),
                "comparison_definition": "O4_permitted_action_gross_vs_reference_entry_gross" if oracle == "O4_hindsight_permitted_action" else "entry_execution_adjusted_gross_bps",
            })
    return pd.DataFrame(rows)


def _policy_regret(selections: dict[tuple[str, float], pd.DataFrame], reference_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_by_fraction = {fraction: selections[(reference_name, fraction)] for fraction in TOP_FRACTIONS}
    oracle_names = ("O1_realised_gross_h12", "O2_realised_net_h12", "O4_hindsight_permitted_action")
    for fraction in TOP_FRACTIONS:
        ref = reference_by_fraction[fraction]
        ref_entry_gross = float(ref.gross_h12_bps.mean())
        ref_net = float(ref.net_h12_bps.mean())
        for oracle in oracle_names:
            selected = selections[(oracle, fraction)]
            evaluation = selected.o4_permitted_action_gross_bps if oracle == "O4_hindsight_permitted_action" else selected.gross_h12_bps
            rows.append({
                "oracle": oracle,
                "reference": reference_name,
                "top_fraction": fraction,
                "oracle_mean_gross_bps": float(evaluation.mean()),
                "reference_selected_entry_gross_bps": ref_entry_gross,
                "entry_regret_bps": float(evaluation.mean() - ref_entry_gross),
                "reference_selected_net_bps": ref_net,
                "oracle_net_status": "NOT_AVAILABLE_ACTION_PATH_COST_NOT_CAUSALLY_RECONSTRUCTED" if oracle == "O4_hindsight_permitted_action" else "AVAILABLE",
                "oracle_mean_net_bps": np.nan if oracle == "O4_hindsight_permitted_action" else float(selected.net_h12_bps.mean()),
            })
    return pd.DataFrame(rows)


def _target_sensitivities(frame: pd.DataFrame, selections: dict[tuple[str, float], pd.DataFrame]) -> pd.DataFrame:
    """Report materialised sensitivities and explicit non-reconstructable cells."""
    clear0 = frame.postcost_h0_event.eq("clear_cost_first")
    clear25 = frame.get("postcost_h25_event", pd.Series(index=frame.index, dtype=object)).eq("clear_cost_first")
    rows: list[dict[str, Any]] = []
    if "postcost_h25_event" in frame:
        for left_name, right_name, left, right in (("fixed_cost_hurdle_0bps", "fixed_cost_hurdle_25bps", clear0, clear25),):
            intersection = int((left & right).sum())
            union = int((left | right).sum())
            rows.append({
                "sensitivity": f"{left_name}_vs_{right_name}",
                "status": "AVAILABLE_EVENT_LABEL_ONLY",
                "rank_spearman": np.nan,
                "event_label_agreement": float(np.mean(left.eq(right))),
                "top_tail_membership_jaccard": float(intersection / union) if union else np.nan,
                "gross_ev_sensitivity_bps": np.nan,
                "reason": "Only fixed-cost state labels are materialised; this is not a barrier perturbation.",
            })
    unavailable = {
        "entry_delay_0m": "AVAILABLE_BASELINE_EXACT_H12_EXECUTION_ADJUSTED_GROSS",
        "entry_delay_1m": "NOT_AVAILABLE: delayed H12 needs immutable post-H12 path tail",
        "entry_delay_5m": "NOT_AVAILABLE: delayed H12 needs immutable post-H12 path tail",
        "entry_delay_10m": "NOT_AVAILABLE: delayed H12 needs immutable post-H12 path tail",
        "path_resolution_1m": "AVAILABLE_BASELINE_PATH_RESOLUTION",
        "path_resolution_5m": "NOT_AVAILABLE: no matched frozen 5m replay materialised",
        "path_resolution_15m": "NOT_AVAILABLE: no matched frozen 15m replay materialised",
        "small_barrier_perturbation": "NOT_AVAILABLE: immutable materialised labels have no alternate barrier replay",
        "timeout_perturbation": "NOT_AVAILABLE: immutable materialised labels have no alternate timeout replay",
        "entry_price_perturbation": "NOT_AVAILABLE: raw ideal fill / post-entry tail is not materialised",
    }
    for sensitivity, status in unavailable.items():
        rows.append({
            "sensitivity": sensitivity,
            "status": status,
            "rank_spearman": 1.0 if sensitivity in {"entry_delay_0m", "path_resolution_1m"} else np.nan,
            "event_label_agreement": 1.0 if sensitivity in {"entry_delay_0m", "path_resolution_1m"} else np.nan,
            "top_tail_membership_jaccard": 1.0 if sensitivity in {"entry_delay_0m", "path_resolution_1m"} else np.nan,
            "gross_ev_sensitivity_bps": 0.0 if sensitivity in {"entry_delay_0m", "path_resolution_1m"} else np.nan,
            "reason": "Baseline is exact frozen execution-adjusted pre-fee H12 gross." if status.startswith("AVAILABLE") else status,
        })
    return pd.DataFrame(rows)


def _decision(results: pd.DataFrame) -> dict[str, Any]:
    pooled = results.loc[(results.slice_kind == "pooled") & (results.top_fraction == 0.20)]
    gross = pooled.loc[pooled.oracle.eq("O1_realised_gross_h12"), "mean_evaluation_gross_bps"].iloc[0]
    # O2's ranking selection can happen to equal O1's when realised cost is
    # monotonic in gross, but its diagnostic quantity is still the selected
    # *net* return.  Never substitute the companion gross statistic here.
    net = pooled.loc[pooled.oracle.eq("O2_realised_net_h12"), "mean_net_bps"].iloc[0]
    policy = pooled.loc[pooled.oracle.eq("O4_hindsight_permitted_action"), "mean_evaluation_gross_bps"].iloc[0]
    if gross < 0.0:
        classification = "CANDIDATE_POPULATION_OR_TARGET_FAILURE_AT_BROAD_GLOBAL_TOP20"
    elif net < 0.0:
        classification = "COST_CONVERSION_PROBLEM_GROSS_TAIL_POSITIVE_NET_TAIL_NEGATIVE"
    elif policy > gross:
        classification = "ENTRY_OUTCOME_POSITIVE_POLICY_OPPORTUNITY_DIAGNOSTIC_ONLY"
    else:
        classification = "NO_PRIMARY_PHASE1_FAILURE_FROM_BROAD_TAIL"
    return {"broad_global_top_fraction": 0.20, "o1_gross_bps": gross, "o2_net_bps": net, "o4_policy_gross_bps": policy, "classification": classification}


def run(ledger_path: Path = DEFAULT_LEDGER, substrate_manifest_path: Path = DEFAULT_SUBSTRATE_MANIFEST, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"artifact already exists: {output}")
    ledger = pd.read_parquet(ledger_path)
    required = {
        "candidate_id", "side", "decision_ts", "gross_h12_bps", "net_h12_bps", "postcost_h0_event", "postcost_h0_four_state",
        "action_continue_execution_adjusted_gross_bps", "action_exit_execution_adjusted_gross_bps", "score_base_alpha", "score_residual_alpha", "score_residual_delta_alpha",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise ContractError(f"diagnostic ledger misses required fields: {sorted(missing)}")
    if ledger.candidate_id.duplicated().any() or not ledger.residual_is_oof.astype(bool).all():
        raise ContractError("ledger must contain one row per candidate and exact OOF residual scores")
    if not ledger.feature_cutoff_ts.le(ledger.decision_ts).all():
        raise ContractError("ledger contains a score/feature availability violation")
    frame, definitions = _add_scores(ledger)
    results, selections = _make_results(frame, definitions)
    reference = "CURRENT_base_plus_residual_delta_OOF"
    bootstrap = _paired_day_bootstrap(selections, reference)
    regret = _policy_regret(selections, reference)
    sensitivity = _target_sensitivities(frame, selections)
    decision = _decision(results)

    staging = _atomic_dir(output)
    results.to_parquet(staging / "oracle_ladder_results.parquet", index=False)
    bootstrap.to_parquet(staging / "oracle_paired_utc_day_bootstrap.parquet", index=False)
    regret.to_parquet(staging / "oracle_regret_vs_current_oof.parquet", index=False)
    sensitivity.to_parquet(staging / "target_sensitivity_results.parquet", index=False)
    manifest = {
        "schema": "root_cause_oracle_ladder_v2",
        "status": "DIAGNOSTIC_ONLY_NOT_PROMOTION_ELIGIBLE",
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "substrate_manifest": str(substrate_manifest_path),
        "substrate_manifest_sha256": _sha256(substrate_manifest_path),
        "rows": int(len(frame)),
        "candidate_id_sha256": _id_digest(frame.candidate_id),
        "selection_contract": "top-k is GLOBAL over the entire evaluation population; slice results never re-rank",
        "top_fractions": list(TOP_FRACTIONS),
        "bootstrap": {"method": "paired UTC-day bootstrap", "reps": BOOTSTRAP_REPS, "seed": SEED, "reference": reference},
        "score_definitions": definitions,
        "target_sensitivity_contract": "Unavailable cells are recorded explicitly rather than fabricated from truncated frozen paths.",
        "phase1_decision": decision,
        "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
    }
    for name in ("oracle_ladder_results.parquet", "oracle_paired_utc_day_bootstrap.parquet", "oracle_regret_vs_current_oof.parquet", "target_sensitivity_results.parquet"):
        manifest.setdefault("outputs_sha256", {})[name] = _sha256(staging / name)
    (staging / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / "manifest.sha256").write_text(_sha256(staging / "run_manifest.json") + "\n")
    _finalise(staging, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--substrate-manifest", type=Path, default=DEFAULT_SUBSTRATE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.ledger, args.substrate_manifest, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
