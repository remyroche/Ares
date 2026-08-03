#!/usr/bin/env python3
"""Historical-OOF selection of a continuous secondary rank for mapped-EV ties.

The deployed causal isotonic mapped-EV level remains the *only* primary
ranking key.  This script changes only the ordering of rows that have exactly
the same frozen mapped level.  It selects a fixed secondary-key recipe from a
pre-July-20 strict OOF ledger, then evaluates that frozen recipe once on the
exact July 20--23 policy outcomes.  It is deliberately retrospective and
non-promotable: July outcomes never participate in choosing the recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

SCHEMA = "mapped_ev_historical_oof_tie_repair_ablation_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
DEFAULT_ROOT = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
DEFAULT_HISTORY = ROOT / "data_perp/artifacts/strict_oof_ic_ev_conversion_ablation_20260730_v2/support_head_oof_ledger.parquet"
DEFAULT_HISTORY_MANIFEST = DEFAULT_HISTORY.parent / "manifest.json"
DEFAULT_HISTORY_CONTEXT = ROOT / "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"
DEFAULT_POLICY_CONFIG = ROOT / (
    "data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_"
    "20260717_v2/simple_policy_optimiser/deployment/best_policy_params_perps.json"
)
HISTORY_WINDOW = "later_july_forward"

# There are no fitted weights in this grid.  It is a finite, predeclared set of
# lexicographic keys; the historical selector can only choose one of these.
RANK_SPECS: Mapping[str, tuple[str, ...]] = {
    "candidate_id_baseline": (),
    "raw_direct_ev": ("direct",),
    "capture_probability": ("capture",),
    "base_alpha": ("base",),
    "residual_alpha": ("residual",),
    "raw_direct_then_capture": ("direct", "capture"),
    "raw_direct_then_residual": ("direct", "residual"),
    "raw_direct_then_base": ("direct", "base"),
    "residual_then_capture": ("residual", "capture"),
    "base_then_capture": ("base", "capture"),
}


class TieRepairError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], *, role: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise TieRepairError(f"{role} is missing columns: {sorted(missing)}")
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise TieRepairError(f"{role} has non-finite ranking fields")


def _rank_within_primary_bins(
    frame: pd.DataFrame,
    *,
    primary: str,
    secondary: Sequence[str],
) -> pd.DataFrame:
    """Return primary-preserving rank: secondary keys operate only on ties."""

    _require_finite(frame, [primary, *secondary], role="rank input")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise TieRepairError("rank input candidate_id is not unique")
    work = frame.copy()
    columns = [primary, *secondary, "candidate_id"]
    # Stable sorting makes candidate id the declared final deterministic key.
    work = work.sort_values(columns, ascending=[False] * (len(columns) - 1) + [True], kind="mergesort")
    work["tie_repair_rank"] = np.arange(1, len(work) + 1, dtype=np.int64)
    return work


def _select_top_k(
    frame: pd.DataFrame,
    *,
    primary: str,
    secondary: Sequence[str],
    top_k_fraction: float,
) -> pd.DataFrame:
    if not 0.0 < float(top_k_fraction) <= 1.0:
        raise TieRepairError("top_k_fraction must be in (0, 1]")
    ranked = _rank_within_primary_bins(frame, primary=primary, secondary=secondary)
    k = max(1, int(math.ceil(float(top_k_fraction) * len(ranked))))
    ranked["tie_repair_top_k"] = ranked["tie_repair_rank"].le(k)
    # This assertion is the core repair contract: secondary ranking must never
    # change either the frozen mapped value or any row strictly above cutoff.
    cutoff = float(ranked.loc[ranked["tie_repair_top_k"], primary].min())
    strictly_above = ranked[primary].gt(cutoff)
    if not ranked.loc[strictly_above, "tie_repair_top_k"].all():
        raise AssertionError("secondary rank displaced a row above the frozen mapped cutoff")
    if ranked.loc[ranked["tie_repair_top_k"], primary].nunique() != ranked.loc[
        ranked["tie_repair_rank"].le(k), primary
    ].nunique():
        raise AssertionError("primary mapped-EV levels changed under tie repair")
    return ranked.sort_index()


def _history_panel(history_path: Path, context_path: Path) -> pd.DataFrame:
    if not history_path.is_file() or not context_path.is_file():
        raise TieRepairError("strict-OOF history ledger and canonical context are required")
    historical = pd.read_parquet(history_path)
    required = {
        *IDENTITY,
        "window", "execution_decision_utc", "support_label_available_utc",
        "execution_net_ev_12h", "side_causal_oof_ev_direct_net_residual",
        "raw_direct_net", "p_exit_favorable_positive", "direct_net_residual",
    }
    if required.difference(historical.columns):
        raise TieRepairError("strict-OOF history ledger does not expose required score families")
    historical = historical.loc[historical["window"].astype(str).eq(HISTORY_WINDOW)].copy()
    if historical.empty or historical.duplicated(list(IDENTITY)).any():
        raise TieRepairError("strict-OOF later-July history is empty or identity is non-unique")
    for column in ("execution_decision_utc", "support_label_available_utc"):
        historical[column] = pd.to_datetime(historical[column], utc=True, errors="raise")
    # This cutoff is deliberately hard-coded and independently checked.  It
    # prevents accidentally selecting a tie rule from the July 20--23 labels.
    cutoff = pd.Timestamp("2026-07-20T00:00:00Z")
    if not historical["support_label_available_utc"].lt(cutoff).all():
        raise TieRepairError("historical selector contains a July-20-or-later resolved outcome")
    context = pd.read_parquet(context_path, columns=[*IDENTITY, "existing_alpha_ev", "base_oof_score"])
    if context.duplicated(list(IDENTITY)).any():
        raise TieRepairError("canonical context identities are non-unique")
    panel = historical.merge(context, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(panel) != len(historical):
        raise TieRepairError("strict-OOF history is not fully covered by base/residual alpha context")
    panel = panel.rename(columns={
        "side_causal_oof_ev_direct_net_residual": "primary_mapped_ev",
        "raw_direct_net": "direct",
        "p_exit_favorable_positive": "capture",
        "base_oof_score": "base",
        "existing_alpha_ev": "residual",
    })
    # The first expanding folds deliberately use an uncalibrated fallback.
    # They are valid model rows but not comparable mapped-OOF evidence, so the
    # selector has a predeclared complete-score requirement rather than silently
    # treating missing OOF calibration as a numeric rank.
    complete = ["primary_mapped_ev", "direct", "capture", "base", "residual", "execution_net_ev_12h"]
    numeric = panel.loc[:, complete].apply(pd.to_numeric, errors="coerce")
    panel = panel.loc[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)].copy()
    if panel.empty:
        raise TieRepairError("strict-OOF selector has no complete calibrated rows")
    _require_finite(panel, ["primary_mapped_ev", "direct", "capture", "base", "residual", "execution_net_ev_12h"], role="strict-OOF selector")
    panel["calendar_month"] = panel["execution_decision_utc"].dt.strftime("%Y-%m")
    return panel


def select_recipe_from_history(panel: pd.DataFrame, *, top_k_fraction: float) -> tuple[str, pd.DataFrame]:
    """Predeclare worst-month then aggregate net as the historical rule."""

    records: list[dict[str, Any]] = []
    for recipe, secondary in RANK_SPECS.items():
        chosen = _select_top_k(panel, primary="primary_mapped_ev", secondary=secondary, top_k_fraction=top_k_fraction)
        selected = chosen.loc[chosen["tie_repair_top_k"]]
        monthly = selected.groupby("calendar_month", sort=True)["execution_net_ev_12h"].mean()
        records.append({
            "recipe": recipe,
            "secondary_keys": "+".join(secondary) if secondary else "candidate_id",
            "historical_oof_rows": int(len(panel)),
            "historical_oof_top_k_rows": int(len(selected)),
            "months": int(len(monthly)),
            "worst_month_top10_net_bps": float(monthly.min() * 1e4),
            "mean_month_top10_net_bps": float(monthly.mean() * 1e4),
            "pooled_top10_net_bps": float(selected["execution_net_ev_12h"].mean() * 1e4),
        })
    comparison = pd.DataFrame(records).sort_values(
        ["worst_month_top10_net_bps", "mean_month_top10_net_bps", "pooled_top10_net_bps", "recipe"],
        ascending=[False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    comparison["historical_selection_rank"] = np.arange(1, len(comparison) + 1)
    return str(comparison.iloc[0]["recipe"]), comparison


def _current_recipes(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work = work.rename(columns={
        "mapped_execution_ev": "primary_mapped_ev",
        "final_direct_net_raw": "direct",
        "final_capture_probability": "capture",
        "base_oof_score": "base",
        "existing_alpha_ev": "residual",
    })
    _require_finite(work, ["primary_mapped_ev", "direct", "capture", "base", "residual", "execution_net_ev_12h"], role="July evaluation")
    # The retrospective cohort is defined by the signal/candidate timestamp:
    # Jul-20 00:00 through Jul-23 23:00.  The associated decision is one hour
    # later, so grouping by decision time would create a misleading one-row
    # Jul-24 bucket.
    work["utc_date"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise").dt.strftime("%Y-%m-%d")
    return work


def _apply_recipe(frame: pd.DataFrame, *, recipe: str, top_k_fraction: float) -> pd.DataFrame:
    if recipe not in RANK_SPECS:
        raise TieRepairError(f"unknown recipe: {recipe}")
    selected = _select_top_k(frame, primary="primary_mapped_ev", secondary=RANK_SPECS[recipe], top_k_fraction=top_k_fraction)
    result = selected.copy()
    result["global_top10_capacity_member"] = result["tie_repair_top_k"].astype(bool)
    # The calibrated positive-floor rule stays exactly as deployed: strict
    # positive mapped EV, *and* membership in the one pooled global top-k.
    for floor in (0, 25, 50):
        result[f"globally_admitted_floor_{floor}bps"] = result["tie_repair_top_k"] & result["primary_mapped_ev"].gt(floor / 1e4)
    result["globally_admitted"] = result["globally_admitted_floor_0bps"]
    result["global_rank"] = result["tie_repair_rank"].astype(np.int64)
    result["mapped_execution_ev"] = result["primary_mapped_ev"]
    return result


def _metric_rows(frame: pd.DataFrame, *, recipe: str) -> list[dict[str, Any]]:
    cohorts = {
        "global_top10": frame["global_top10_capacity_member"].astype(bool),
        "admitted_gt_0bps": frame["globally_admitted_floor_0bps"].astype(bool),
        "admitted_gt_25bps": frame["globally_admitted_floor_25bps"].astype(bool),
        "admitted_gt_50bps": frame["globally_admitted_floor_50bps"].astype(bool),
    }
    scopes: list[tuple[str, Sequence[str]]] = [("overall", ()) , ("day", ("utc_date",)), ("side", ("side_name",)), ("day_side", ("utc_date", "side_name"))]
    records: list[dict[str, Any]] = []
    for cohort, mask in cohorts.items():
        selected = frame.loc[mask]
        for scope, keys in scopes:
            groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
            for values, group in groups:
                values = values if isinstance(values, tuple) else (values,)
                net = pd.to_numeric(group["execution_net_ev_12h"], errors="raise")
                record: dict[str, Any] = {
                    "recipe": recipe, "cohort": cohort, "scope": scope,
                    "utc_date": None, "side_name": None, "rows": int(len(group)),
                    "mean_net_bps": float(net.mean() * 1e4) if len(group) else np.nan,
                    "positive_net_precision": float(net.gt(0).mean()) if len(group) else np.nan,
                    "mean_mapped_ev_bps": float(group["primary_mapped_ev"].mean() * 1e4) if len(group) else np.nan,
                }
                for key, value in zip(keys, values):
                    record[key] = value
                records.append(record)
    return records


def _tie_support(frame: pd.DataFrame, *, recipe: str) -> dict[str, Any]:
    selected = frame.loc[frame["global_top10_capacity_member"].astype(bool)]
    cutoff = float(selected["primary_mapped_ev"].min())
    tied = frame["primary_mapped_ev"].eq(cutoff)
    above = frame["primary_mapped_ev"].gt(cutoff)
    return {
        "recipe": recipe,
        "cutoff_mapped_ev": cutoff,
        "cutoff_mapped_ev_bps": cutoff * 1e4,
        "top_k_rows": int(len(selected)),
        "strictly_above_cutoff_rows": int(above.sum()),
        "tied_cutoff_rows": int(tied.sum()),
        "selected_from_tie_rows": int((tied & frame["global_top10_capacity_member"].astype(bool)).sum()),
        "selected_from_tie_by_side": frame.loc[tied & frame["global_top10_capacity_member"].astype(bool), "side_name"].value_counts().sort_index().to_dict(),
        "positive_floor_rows": int(frame["globally_admitted_floor_0bps"].sum()),
        "positive_floor_unchanged_by_secondary_rank": True,
    }


def _portfolio_for_recipe(frame: pd.DataFrame, *, policy_path: Path, initial_wallet: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # The policy reporter is contract-compatible: exact 1m/12h label returns,
    # no re-charged costs, and the signed simple-policy portfolio limits.
    # Importing the full replay stack is intentionally deferred so the pure
    # historical-selection unit tests do not initialize its Numba cache.
    from scripts.report_execution_ev_july_exact_economics import portfolio_replays

    summary, decisions, _, side, contract = portfolio_replays(frame, policy_path=policy_path, initial_wallet=initial_wallet)
    summary.insert(0, "recipe", "")
    decisions.insert(0, "recipe", "")
    side.insert(0, "recipe", "")
    return summary, side, {"contract": contract, "decision_rows": int(len(decisions))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    history_path, history_manifest, context_path = map(Path, (args.history, args.history_manifest, args.history_context))
    if not history_manifest.is_file():
        raise TieRepairError("strict-OOF history manifest is required")
    historical_manifest = json.loads(history_manifest.read_text())
    contract_text = json.dumps(historical_manifest.get("contract", {})).lower()
    if "strict oof" not in contract_text and "temporal oof" not in contract_text:
        raise TieRepairError("history manifest does not attest temporal/strict OOF semantics")
    history = _history_panel(history_path, context_path)
    winner, historical_comparison = select_recipe_from_history(history, top_k_fraction=float(args.top_k_fraction))

    root = Path(args.root)
    from scripts.report_execution_ev_july_exact_economics import load_joined_population

    current, score_manifest, label_manifest, _ = load_joined_population(
        scored_path=root / "scored/scored_population.parquet",
        scored_manifest_path=root / "scored/manifest.json",
        labels_path=root / "labels_12h/execution_ev_policy_labels.parquet",
        labels_manifest_path=root / "labels_12h/manifest.json",
        preentry_manifest_path=root / "preentry/manifest.json",
        policy_path=Path(args.policy), top_k_fraction=float(args.top_k_fraction),
    )
    current = _current_recipes(current)
    # Never make the July outcomes reachable from the historical selection code.
    all_metrics: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    portfolio_frames: list[pd.DataFrame] = []
    portfolio_side_frames: list[pd.DataFrame] = []
    portfolio_contract: dict[str, Any] | None = None
    candidate_rows: list[pd.DataFrame] = []
    for recipe in RANK_SPECS:
        evaluated = _apply_recipe(current, recipe=recipe, top_k_fraction=float(args.top_k_fraction))
        all_metrics.extend(_metric_rows(evaluated, recipe=recipe))
        tie_rows.append(_tie_support(evaluated, recipe=recipe))
        candidate_rows.append(evaluated.loc[:, ["candidate_id", "tie_repair_rank", "global_top10_capacity_member", "globally_admitted_floor_0bps"]].assign(recipe=recipe))
        # All arms are identical-label, signed-policy constraint replays; no
        # portfolio setting is re-optimized for a secondary rank.
        summary, side, contract = _portfolio_for_recipe(evaluated, policy_path=Path(args.policy), initial_wallet=float(args.initial_wallet))
        summary["recipe"] = recipe
        side["recipe"] = recipe
        portfolio_frames.append(summary)
        portfolio_side_frames.append(side)
        portfolio_contract = contract

    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        outputs = {
            "historical_selection": stage / "historical_oof_recipe_selection.csv",
            "july_metrics": stage / "july_recipe_metrics.csv",
            "tie_support": stage / "july_tie_support.json",
            "candidate_rankings": stage / "july_candidate_rankings.parquet",
            "portfolio_summary": stage / "portfolio_summary.csv",
            "portfolio_side": stage / "portfolio_side_metrics.csv",
        }
        historical_comparison.to_csv(outputs["historical_selection"], index=False)
        pd.DataFrame(all_metrics).to_csv(outputs["july_metrics"], index=False)
        _write_json(outputs["tie_support"], {"recipes": tie_rows})
        pd.concat(candidate_rows, ignore_index=True).to_parquet(outputs["candidate_rankings"], index=False, compression="zstd")
        pd.concat(portfolio_frames, ignore_index=True).to_csv(outputs["portfolio_summary"], index=False)
        pd.concat(portfolio_side_frames, ignore_index=True).to_csv(outputs["portfolio_side"], index=False)
        report = {
            "schema": SCHEMA,
            "status": "research_only_historical_oof_selected_july_single_evaluation_nonpromotable",
            "promotion_eligible": False,
            "selected_recipe": winner,
            "selection_rule": "predeclared finite lexicographic grid; maximize worst calendar-month pooled-global-top10 exact-net bps on strict pre-Jul20 temporal OOF, then mean-month, pooled mean, recipe name",
            "evaluation_rule": "exact July20-23 outcomes are evaluated after, and never passed to, the historical selector",
            "primary_mapping": "frozen causal side-local isotonic mapped_execution_ev; unchanged as first key",
            "secondary_mapping": "continuous secondary keys apply only where the frozen mapped value is equal; candidate_id ascending remains final deterministic key",
            "positive_floors": "unchanged deployed rule: pooled top10 membership AND mapped EV strictly greater than 0/25/50 bps",
            "history_score_family_caveat": "direct/capture scores are temporal-OOF family counterparts; base/residual fields are joined pinned alpha context. This selects an ordering family, not a claim that historical and final-refit score scales are identical.",
            "portfolio_replay": portfolio_contract,
        }
        _write_json(stage / "report.json", report)
        manifest = {
            **report,
            "inputs": {
                "strict_oof_history": _record(history_path),
                "strict_oof_history_manifest": _record(history_manifest),
                "historical_alpha_context": _record(context_path),
                "scored_manifest": _record(root / "scored/manifest.json"),
                "scored_population": _record(root / "scored/scored_population.parquet"),
                "exact_labels_manifest": _record(root / "labels_12h/manifest.json"),
                "exact_labels": _record(root / "labels_12h/execution_ev_policy_labels.parquet"),
                "signed_policy": _record(Path(args.policy)),
            },
            "coverage": {"history_oof_rows": int(len(history)), "july_rows": int(len(current)), "july_days": sorted(current["utc_date"].unique().tolist())},
            # Hash the staged bytes, but publish their final locations.  A
            # manifest must stay resolvable after the atomic directory rename.
            "outputs": {
                key: {"path": str(output_dir / path.name), "sha256": _sha256(path)}
                for key, path in outputs.items()
            } | {
                "report": {"path": str(output_dir / "report.json"), "sha256": _sha256(stage / "report.json")}
            },
            "scored_contract": score_manifest.get("contract"),
            "label_schema": label_manifest.get("schema"),
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--history-manifest", type=Path, default=DEFAULT_HISTORY_MANIFEST)
    parser.add_argument("--history-context", type=Path, default=DEFAULT_HISTORY_CONTEXT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser.parse_args(argv)


def main() -> None:
    manifest = run(_parser())
    print(json.dumps({"selected_recipe": manifest["selected_recipe"], "coverage": manifest["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
