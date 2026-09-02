#!/usr/bin/env python3
"""Strict-OOS causal S/R augmentation of the paired BCF/current MC1 mapper.

This is a narrow, offline challenger.  It keeps the frozen target-free
candidate universe, rich-policy label source, MC1-residual target, model
geometry, dual-admission rule and BCF-priority portfolio auction fixed.  The
only change is appending independently OOF causal support/resistance outputs
to the candidate-level MC1 residual mapper.

The S/R outputs are snapshots known before the decision timestamp.  They are
*not* oracle interaction outcomes; missing snapshots remain a model feature
handled by LightGBM rather than a candidate-eligibility gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_entry_e2_demotion_residual_ablation as control
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as feature_study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base


SR_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v3_entrypivotfix"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_mc1_residual_input_ablation_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")

# These are causal, OOF snapshot *outputs* emitted for candidate timestamps.
# Keep this explicit so neither raw future interaction fields nor any oracle
# diagnostic file can enter the mapper by accident.
SR_FEATURES = (
    "sr_long_support_hold_strength",
    "sr_long_resistance_break_probability",
    "sr_long_downside_break_probability",
    "sr_long_resistance_rejection_strength",
    "sr_long_structure_balance",
    "sr_long_support_distance_atr",
    "sr_long_resistance_distance_atr",
    "sr_support_prior_strength",
    "sr_resistance_prior_strength",
    "sr_support_reaction_magnitude_q50",
    "sr_resistance_reaction_magnitude_q50",
)
CAUSAL_SNAPSHOT_FILE = "entry_sr_oof_features.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_causal_sr_root(root: Path) -> Path:
    """Validate the immutable causal-only source before any merge."""
    resolved = root.resolve()
    if "NONCAUSAL" in str(resolved).upper() or "ORACLE" in str(resolved).upper():
        raise ValueError("S/R mapper may consume only causal OOF snapshots, never oracle diagnostics")
    manifest_path = resolved / "run_manifest.json"
    snapshot_path = resolved / CAUSAL_SNAPSHOT_FILE
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("causal S/R root lacks its manifest or entry snapshot output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = " ".join((str(manifest.get("schema", "")), str(manifest.get("causality", "")))).upper()
    if "CAUSAL-SR-HEADS-OOF" not in provenance:
        raise AssertionError("S/R source does not declare an OOF causal contract")
    return snapshot_path


def _merge_causal_sr(panel: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-to-one candidate-time merge without changing target-free identity."""
    snapshot_path = _assert_causal_sr_root(root)
    snapshots = pd.read_parquet(snapshot_path, columns=["candidate_id", "snapshot_ts", *SR_FEATURES])
    snapshots["candidate_id"] = snapshots["candidate_id"].astype(str)
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots["snapshot_ts"], utc=True, errors="raise")
    keys = ["candidate_id", "snapshot_ts"]
    if snapshots.duplicated(keys).any():
        raise AssertionError("causal S/R output duplicates a candidate-time identity")
    work = panel.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["snapshot_ts"] = work["__decision_ts__"]
    if work.duplicated("candidate_id").any():
        raise AssertionError("target-free MC1 panel has duplicate candidate identities")
    merged = work.merge(snapshots, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(work) or not np.array_equal(
        merged["candidate_id"].to_numpy(str), work["candidate_id"].to_numpy(str)
    ):
        raise AssertionError("S/R merge changed target-free candidate identity or order")
    merged["sr_snapshot_available"] = merged.loc[:, list(SR_FEATURES)].notna().any(axis=1).astype(np.int8)
    coverage = merged.groupby(merged["__decision_ts__"].dt.to_period("M"), observed=True).agg(
        rows=("candidate_id", "size"),
        causal_sr_available=("sr_snapshot_available", "sum"),
    ).reset_index(names="decision_month")
    return merged, coverage


def _source_aligned_labels(root: Path, *, workers: int = 8) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load only readable immutable label parts and disclose any exclusion.

    An unreadable source-aligned parquet part may not be replaced from a
    different outcome source.  Its whole symbol is instead removed from every
    matched arm before fitting or target-free selection, exactly as in the
    existing causal S/R ceiling receipt.
    """
    parts = sorted(root.resolve().glob("policy_parts/symbol=*/policy_labels.parquet"))
    if not parts:
        raise FileNotFoundError(f"no rich-policy label parts beneath {root}")
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    def read_part(path: Path) -> tuple[str, pd.DataFrame | None, str | None]:
        symbol = path.parent.name.removeprefix("symbol=")
        try:
            return symbol, pd.read_parquet(path, columns=columns), None
        except (ArrowInvalid, OSError, ValueError) as exc:
            return symbol, None, f"unreadable_parquet:{type(exc).__name__}"

    # Label parts are immutable and independent.  Bounded concurrency reduces
    # filesystem round-trips while ``map`` preserves the sorted source order;
    # it cannot alter rows, labels, exclusions, or downstream model inputs.
    workers = max(1, min(int(workers), 8, len(parts)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(read_part, parts))
    frames = [frame for _, frame, error in results if frame is not None and error is None]
    unavailable = [
        {"symbol": symbol, "reason": str(error)}
        for symbol, _, error in results if error is not None
    ]
    if not frames:
        raise RuntimeError("no readable source-aligned rich-policy label parts")
    labels = pd.concat(frames, ignore_index=True)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("source-aligned policy labels are not one-to-one")
    valid = labels["policy_path_valid"].fillna(False).astype(bool)
    if not np.isclose(
        pd.to_numeric(labels.loc[valid, "policy_gross_bps"], errors="coerce")
        - pd.to_numeric(labels.loc[valid, "policy_net_bps"], errors="coerce"),
        100.0, rtol=0.0, atol=1e-8,
    ).all():
        raise AssertionError("rich-policy labels must embed the 100-bps cost exactly once")
    labels["policy_label_available_ts"] = pd.to_datetime(
        labels["policy_label_available_ts"], utc=True, errors="raise"
    )
    return labels, unavailable


def _scope_replay(
    selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    timestamps = pd.to_datetime(selection["__decision_ts__"], utc=True, errors="raise")
    for scope, subset in (
        ("selection_jun_jul", selection.loc[timestamps.lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[timestamps.ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if subset.empty:
            continue
        metric = base._replay(subset, labels, f"{arm}__{scope}", output)
        metric["model_arm"], metric["evaluation_scope"] = arm, scope
        summaries.append(metric)
    return summaries


def _candidate_features(held_months: tuple[pd.Timestamp, ...]) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Use the frozen E3 projection already used by the MC1 residual control."""
    feature_file = control.h0.FEATURE_STUDY / "stable_selected_features.parquet"
    return control._candidate_features(feature_file, held_months)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-root", type=Path, default=SR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--label-read-workers", type=int, default=8)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; defaults Jun--Aug 2026")
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict MC1 residual training needs at least two preceding calendar months")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    held_months = (
        tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.held_month)
        if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    )
    base_features = _candidate_features(held_months)
    raw_target_free = feature_study._candidate_frame(
        feature_study._load_panel(feature_study.OLD_PANEL, feature_study.VWAP_PANEL)
    )
    target_free, coverage = _merge_causal_sr(raw_target_free, args.sr_root)
    labels, unavailable_label_parts = _source_aligned_labels(
        base.LABEL_ROOT, workers=args.label_read_workers
    )
    unavailable_symbols = frozenset(item["symbol"] for item in unavailable_label_parts)
    if unavailable_symbols:
        target_free = target_free.loc[~target_free["__symbol__"].isin(unavailable_symbols)].copy()
    labelled = target_free.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled["policy_path_valid"].fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(
        labelled["policy_label_available_ts"], utc=True, errors="raise"
    )
    output.mkdir(parents=True, exist_ok=False)

    arms = {
        "M_mc1_pair_residual_control": False,
        "M_mc1_pair_residual_plus_causal_sr": True,
    }
    selections: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arms}
    traces: list[pd.DataFrame] = []
    scored: list[pd.DataFrame] = []
    fold_trace: list[dict[str, object]] = []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train = labelled.loc[
            labelled["__decision_ts__"].ge(start)
            & labelled["__decision_ts__"].lt(held)
            & labelled["policy_label_available_ts"].lt(held)
        ].copy()
        test = target_free.loc[
            target_free["__decision_ts__"].ge(held) & target_free["__decision_ts__"].lt(end)
        ].copy()
        observed = set(train["__decision_ts__"].dt.strftime("%Y-%m"))
        required = {(held - pd.DateOffset(months=n)).strftime("%Y-%m") for n in range(1, args.train_months + 1)}
        if len(train) < 500 or test.empty or not required.issubset(observed):
            raise RuntimeError(f"{held:%Y-%m}: strict MC1/SR fold lacks prior resolved support")
        base_fields = base_features[held]
        missing_base = set(base_fields).difference(train.columns) | set(base_fields).difference(test.columns)
        if missing_base:
            raise AssertionError(f"{held:%Y-%m}: baseline MC1 feature contract missing {sorted(missing_base)}")
        target = pd.to_numeric(train["policy_net_bps"], errors="raise") - (
            pd.to_numeric(train["bcf_mc1_expected_bps"], errors="raise")
            + pd.to_numeric(train["current_mc1_expected_bps"], errors="raise")
        ) / 2.0
        clip_low, clip_high = np.quantile(target.to_numpy(float), [.02, .98])
        for arm, uses_sr in arms.items():
            fields = (*base_fields, *SR_FEATURES, "sr_snapshot_available") if uses_sr else base_fields
            # LightGBM treats unavailable S/R snapshots as missing values; the
            # explicit availability flag prevents missingness from silently
            # becoming a candidate/admission filter.
            model = control._fit_residual(train, fields, target)
            residual = model.predict(test.loc[:, fields])
            residual = np.clip(residual, clip_low, clip_high)
            if not np.isfinite(residual).all():
                raise AssertionError(f"{held:%Y-%m}/{arm}: non-finite mapped residual")
            for weight in control.RESIDUAL_WEIGHTS:
                arm_weighted = f"{arm}_w{int(weight * 100):03d}"
                selection, trace = control._select_adjusted(
                    test, arm=arm_weighted, prediction=residual, weight=weight
                )
                selection["held_month"] = held.strftime("%Y-%m")
                selections[arm].append(selection.assign(arm=arm_weighted))
                trace["held_month"], trace["mapper_arm"], trace["weight"] = (
                    held.strftime("%Y-%m"), arm, weight
                )
                traces.append(trace)
                scored.append(pd.DataFrame({
                    "candidate_id": test["candidate_id"].astype(str),
                    "__decision_ts__": test["__decision_ts__"],
                    "mapper_arm": arm,
                    "weight": weight,
                    "mc1_residual_bps": residual,
                    "residual_clip_low_bps": clip_low,
                    "residual_clip_high_bps": clip_high,
                    "held_month": held.strftime("%Y-%m"),
                    "sr_snapshot_available": test["sr_snapshot_available"].to_numpy(np.int8),
                }))
        fold_trace.append({
            "held_month": held.strftime("%Y-%m"),
            "train_start": str(start), "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_sr_available": int(train["sr_snapshot_available"].sum()),
            "test_sr_available": int(test["sr_snapshot_available"].sum()),
            "residual_clip_low_bps": float(clip_low), "residual_clip_high_bps": float(clip_high),
        })

    summary_rows: list[dict[str, object]] = []
    for arm, frames in selections.items():
        selection = pd.concat(frames, ignore_index=True)
        for weight in control.RESIDUAL_WEIGHTS:
            arm_weighted = f"{arm}_w{int(weight * 100):03d}"
            subset = selection.loc[selection["arm"].eq(arm_weighted)].copy()
            if subset["candidate_id"].duplicated().any():
                raise AssertionError(f"{arm_weighted}: selected candidate identity is duplicated")
            if subset.groupby("__decision_ts__").size().gt(base.MAX_NEW_ENTRIES).any():
                raise AssertionError(f"{arm_weighted}: mapper expanded the two-entry timestamp cap")
            target_free_cols = [column for column in subset.columns if not column.startswith("policy_")]
            subset.loc[:, target_free_cols].to_parquet(
                output / f"{arm_weighted}_selection_target_free.parquet", index=False, compression="zstd"
            )
            summary_rows.extend(_scope_replay(subset.loc[:, target_free_cols], labels, arm_weighted, output))
    summary = pd.DataFrame(summary_rows)
    summary["total_ev_per_abs_drawdown"] = summary["total_policy_net_bps"] / summary["max_drawdown"].abs().replace(0.0, np.nan)
    for scope, rows in summary.groupby("evaluation_scope", sort=False):
        reference = rows.loc[rows["model_arm"].eq("M_mc1_pair_residual_control_w050")]
        if len(reference) != 1:
            raise AssertionError(f"{scope}: matched mapper control missing")
        for metric in (
            "portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown",
            "worst_week", "sortino", "total_ev_per_abs_drawdown",
        ):
            summary.loc[rows.index, f"delta_vs_control_{metric}"] = rows[metric] - reference.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    coverage.to_parquet(output / "sr_merge_coverage.parquet", index=False)
    pd.DataFrame(fold_trace).to_parquet(output / "fold_trace.parquet", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(output / "admission_trace_target_free.parquet", index=False, compression="zstd")
    pd.concat(scored, ignore_index=True).to_parquet(output / "mc1_residual_scores_target_free.parquet", index=False, compression="zstd")
    snapshot_path = _assert_causal_sr_root(args.sr_root)
    manifest = {
        "schema": "causal-sr-mc1-residual-input-ablation-v1",
        "scope": "offline strict-OOS challenger; no live/canonical/execution mutation",
        "change_isolated": "only independently OOF causal S/R snapshot outputs are appended to the pre-existing MC1 residual mapper input contract",
        "baseline_mapper": "same paired BCF/current MC1 residual target, L1 model geometry, residual clipping, dual >=30 admission and BCF-priority top-two route",
        "target": "rich-policy net bps minus the mean of contemporaneous BCF/current MC1 expected bps",
        "s_r_source": str(args.sr_root.resolve()),
        "s_r_snapshot": str(snapshot_path),
        "s_r_manifest_sha256": _sha256(args.sr_root.resolve() / "run_manifest.json"),
        "s_r_fields": list(SR_FEATURES),
        "s_r_missingness": "allowed as a mapper feature via LightGBM plus sr_snapshot_available; never filters candidate identity or eligibility",
        "unavailable_source_aligned_label_parts_excluded_from_all_arms": unavailable_label_parts,
        "folds": fold_trace,
        "held_months": [f"{value:%Y-%m}" for value in held_months],
        "selection_period": "June--July 2026; August is holdout and not used to select an arm",
        "training": f"up to {args.train_months} complete prior calendar months; labels resolve before held boundary",
        "label_part_read_workers": int(max(1, min(args.label_read_workers, 8))),
        "outcome_contract": "all mapper scores and selections target-free; rich-policy outcomes join only for training and post-selection replay; 100-bps policy cost embedded once",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
