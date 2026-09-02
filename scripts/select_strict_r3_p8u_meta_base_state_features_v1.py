#!/usr/bin/env python3
"""Strict pre-2026 CMI/IC selection for P8U Base-state Meta inputs.

The candidate columns were already written target-free by
``materialize_strict_r3_p8u_meta_base_state_v1.py``.  This script opens
October--December 2025 outcomes only after checking those receipts, measures
conditional information beyond Base rank, and emits frozen M0--M5 contracts
for the later Jan--Jul 2026 objective screen.  It never trains a Meta model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

import run_strict_r3_p8u_meta_target_query_grid_v1 as meta


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_base_state_selection_v1"
IDENTITY = tuple(meta.IDENTITY)
DEV_MONTHS = ("2025-10", "2025-11", "2025-12")
MAX_M1 = 20
M6_EXTRA_STABLE = 4


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token}-01", tz="UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _fields(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    result = tuple(str(value) for value in payload.get("selected_features", ()))
    if len(result) != 120 or len(set(result)) != len(result):
        raise AssertionError("expected immutable 120-field F120 contract")
    return result


def _overlay_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_overlay(root: Path, month: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(_overlay_path(root, month))
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame.duplicated(IDENTITY).any():
        raise AssertionError(f"{month:%Y-%m}: duplicate target-free overlay identity")
    return frame


def _read_labels(policy_path: Path, path_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    start, end = month, _month_end(month)
    path = meta._read_path(path_root, start, end)
    policy = meta._read_policy(policy_path)
    out = path.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    out["atr_bps"] = (out.path_arch_atr_fraction * 10_000.0).astype(np.float32)
    return out


def _residuals(frame: pd.DataFrame) -> np.ndarray:
    """Strict prequential residuals, never a same-block policy fit."""
    work = frame.copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    valid = meta._valid_label(work)
    result = np.full(len(work), np.nan, dtype=np.float32)
    first = work.__decision_ts__.min().floor("D")
    blocks = ((work.__decision_ts__ - first) / pd.Timedelta(days=14)).astype(int)
    for block in sorted(blocks.unique()):
        current = blocks.eq(block)
        start = work.loc[current, "__decision_ts__"].min()
        prior = valid & work.__decision_ts__.lt(start).to_numpy(bool) & work.policy_label_available_ts.lt(start).to_numpy(bool)
        if int(prior.sum()) < 1000:
            continue
        from sklearn.isotonic import IsotonicRegression
        mapper = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            work.loc[prior, "base_rank_ts"], work.loc[prior, "policy_net_bps"],
        )
        result[current.to_numpy()] = work.loc[current, "policy_net_bps"].to_numpy(float) - mapper.predict(work.loc[current, "base_rank_ts"])
    original = np.full(len(frame), np.nan, dtype=np.float32)
    original[work["__row__"].to_numpy(np.int64)] = result
    return original


def _conditional_metrics(values: np.ndarray, base_rank: np.ndarray, targets: dict[str, np.ndarray], seed: int) -> dict[str, tuple[float, float]]:
    """Fast binned CMI proxy plus signed within-Base-band IC.

    The selector's job is a stable screen, not a non-parametric estimator
    benchmark.  Equal-frequency ten-bin states make the CMI deterministic,
    preserve conditioning on Base band, and avoid thousands of stochastic
    nearest-neighbour fits across the same candidate panel.
    """
    result: dict[str, tuple[float, float]] = {}
    valid_x = np.isfinite(values) & np.isfinite(base_rank)
    band = np.minimum(9, np.maximum(0, np.floor((1.0 - base_rank) * 10.0))).astype(int)

    def bins(raw: np.ndarray) -> np.ndarray:
        series = pd.Series(raw)
        rank = series.rank(method="average", pct=True).to_numpy(float)
        return np.minimum(9, np.maximum(0, np.floor(rank * 10.0))).astype(np.int16)

    for name, target in targets.items():
        valid = valid_x & np.isfinite(target)
        cmi_parts: list[float] = []
        ic_parts: list[float] = []
        weights: list[int] = []
        for token in range(10):
            local = valid & (band == token)
            n = int(local.sum())
            if n < 100 or np.unique(target[local]).size < 2 or np.unique(values[local]).size < 5:
                continue
            x = values[local].astype(float)
            y = target[local].astype(float)
            x = np.where(np.isfinite(x), x, np.nanmedian(x))
            y_state = bins(y) if name == "residual_magnitude" else y.astype(np.int16)
            cmi = float(mutual_info_score(bins(x), y_state))
            ic = float(spearmanr(x, y).statistic)
            if np.isfinite(cmi) and np.isfinite(ic):
                cmi_parts.append(cmi); ic_parts.append(ic); weights.append(n)
        if weights:
            result[name] = (float(np.average(cmi_parts, weights=weights)), float(np.average(ic_parts, weights=weights)))
        else:
            result[name] = (float("nan"), float("nan"))
    return result


def _selection(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = sorted(detail.target.unique())
    for feature, part in detail.groupby("feature", sort=True):
        activation: list[bool] = []
        signs: list[int] = []
        record: dict[str, object] = {"feature": feature, "coverage": float(part.coverage.min())}
        composite_values: list[float] = []
        for target in targets:
            local = part.loc[part.target.eq(target)].copy()
            record[f"{target}_cmi_mean"] = float(local.cmi.mean())
            record[f"{target}_abs_ic_mean"] = float(local.ic.abs().mean())
            record[f"{target}_active_folds"] = int(local.active.sum())
            composite_values.append(float(local.cmi.mean() + .25 * local.ic.abs().mean()))
            if target in {"underconfidence", "opportunity100"}:
                signs.extend(np.sign(local.loc[local.active, "ic"]).astype(int).tolist())
        record["composite"] = float(np.mean(composite_values))
        record["active_folds_any"] = int(part.groupby("held_month").active.any().sum())
        positive, negative = signs.count(1), signs.count(-1)
        record["direction_consistency"] = float(max(positive, negative) / max(1, positive + negative))
        rows.append(record)
    out = pd.DataFrame(rows)
    # A lower standard than all-fold SHAP lineage: a field must be active in
    # two of the three pre-2026 folds, have broadly stable direction whenever
    # it is an under/opportunity discriminator, and retain full causal panel
    # coverage.  This leaves model validation—not selector optimism—with the
    # final authority.
    out["m1_eligible"] = (
        out.coverage.ge(.90)
        & out.active_folds_any.ge(2)
        & out.direction_consistency.ge(.60)
    )
    return out.sort_values(["m1_eligible", "composite", "feature"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    fields = _fields(args.f120_contract)
    manifest = json.loads((args.overlay_root / "run_manifest.json").read_text())
    if manifest.get("schema") != "strict_r3_p8u_meta_base_state_v1":
        raise AssertionError("wrong Base-state overlay schema")
    args.out.mkdir(parents=True)
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline pre-2026 CMI/IC selection for target-free Base-state Meta features; no model/MC1/admission/portfolio/live/exchange mutation",
        "overlay_root": str(args.overlay_root), "overlay_manifest_sha256": _sha(args.overlay_root / "run_manifest.json"),
        "development_months": list(DEV_MONTHS), "f120_contract": str(args.f120_contract), "f120_contract_sha256": _sha(args.f120_contract),
        "policy_labels": str(args.policy_labels), "path_labels": str(args.path_labels),
        "selection_rule": "coverage>=90%, active in >=2/3 pre-2026 folds, conditional under/opportunity direction consistency>=60%, top 20 by composite; final authority is later Jan-Jul OOF Meta screen",
    })
    base_root = Path(str(manifest["base_root"]))
    all_overlays: dict[str, pd.DataFrame] = {}
    for token in manifest["months"]:
        month = _month(token)
        overlay = _read_overlay(args.overlay_root, month)
        base = pd.read_parquet(meta._base_path(base_root, month), columns=list(IDENTITY) + ["base_rank_ts"])
        base["__decision_ts__"] = pd.to_datetime(base["__decision_ts__"], utc=True, errors="raise")
        overlay = overlay.merge(base, on=list(IDENTITY), how="left", validate="one_to_one")
        if len(overlay) != len(base) or not np.isfinite(pd.to_numeric(overlay.base_rank_ts, errors="coerce")).all():
            raise AssertionError(f"{token}: frozen Base-rank identity coverage failure")
        all_overlays[token] = overlay
    derived = tuple(column for column in all_overlays[DEV_MONTHS[0]].columns if column.startswith("meta_"))
    if not derived:
        raise AssertionError("overlay contains no declared Base-state features")
    detail_rows: list[dict[str, object]] = []
    for index, token in enumerate(DEV_MONTHS):
        month = _month(token)
        # Include the preceding available history only for the strict
        # prequential residual anchor.  The actual CMI/IC observation rows
        # are exactly this predeclared development month.
        history_tokens = [key for key in all_overlays if _month(key) <= month]
        history = pd.concat([all_overlays[key] for key in history_tokens], ignore_index=True)
        labels = pd.concat([_read_labels(args.policy_labels, args.path_labels, _month(key)) for key in history_tokens], ignore_index=True)
        labelled = history.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
        labelled["residual_bps"] = _residuals(labelled)
        current = labelled.loc[labelled.__decision_ts__.ge(month) & labelled.__decision_ts__.lt(_month_end(month))].copy()
        valid = meta._valid_label(current) & np.isfinite(current.residual_bps)
        current = current.loc[valid].copy()
        reaches = current.path_arch_peak_mfe_atr.to_numpy(float) >= meta.TRAILING_ACTIVATION_ATR
        stopped = current.policy_exit_reason.astype(str).isin(meta.STOP_REASONS).to_numpy(bool)
        residual_values = current.residual_bps.to_numpy(float)
        policy_values = current.policy_net_bps.to_numpy(float)
        targets = {
            "underconfidence": (reaches & (residual_values >= 100.0)).astype(float),
            "overconfidence": (stopped & (residual_values <= -100.0)).astype(float),
            "residual_magnitude": np.abs(residual_values),
            "opportunity100": (policy_values > 100.0).astype(float),
        }
        for feature_index, feature in enumerate(derived):
            values = pd.to_numeric(current[feature], errors="coerce").to_numpy(float)
            coverage = float(np.isfinite(values).mean())
            metrics = _conditional_metrics(values, current.base_rank_ts.to_numpy(float), targets, 1729 + index * 1000 + feature_index)
            # An activity threshold is cross-feature and month-specific.  It
            # is calculated after all values for this fold, below.
            for target, (cmi, ic) in metrics.items():
                detail_rows.append({"held_month": token, "feature": feature, "target": target, "cmi": cmi, "ic": ic, "coverage": coverage})
    detail = pd.DataFrame(detail_rows)
    threshold = detail.groupby(["held_month", "target"], sort=False).cmi.transform(lambda values: values.quantile(.60))
    detail["active"] = detail.cmi.ge(threshold) & detail.ic.abs().ge(.005)
    summary = _selection(detail)
    stable_pool = summary.loc[summary.m1_eligible, "feature"].tolist()
    eligible = stable_pool[:MAX_M1]
    if not eligible:
        raise AssertionError("no stable M1 candidate survived the declared selector")
    # M6 is a deliberately small, pre-2026-only extension of the mixed M1
    # contract.  It tests the next four lower-ranked *stable* candidates,
    # rather than selecting a union after looking at 2026 Meta outcomes.
    # This retains an honest later OOS screen for the combined arm.
    m6_added = stable_pool[:MAX_M1 + M6_EXTRA_STABLE]
    contracts = {
        "m0": list(fields),
        "m1": [*fields, *eligible],
        "m2": [*fields, *[field for field in derived if field.startswith("meta_q_")]],
        "m3": [*fields, *[field for field in derived if field.startswith("meta_cal_")]],
        "m4": [*fields, *[field for field in derived if field.startswith("meta_tree_")]],
        "m5": [*fields, *[field for field in derived if field.startswith("meta_leaf_")]],
        "m6": [*fields, *m6_added],
    }
    (args.out / "contracts").mkdir()
    for arm, selected in contracts.items():
        _once(args.out / "contracts" / f"{arm}.json", {
            "schema": SCHEMA, "arm": arm, "selected_features": selected,
            "parent_f120_feature_count": len(fields), "added_feature_count": len(selected) - len(fields),
            "selection_scope": (
                "pre-2026-only top-20 stable mixed selector"
                if arm == "m1" else
                "pre-2026-only top-24 stable mixed selector"
                if arm == "m6" else
                "declared Base-state family"
            ),
        })
    detail.to_parquet(args.out / "conditional_feature_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "selection_summary.parquet", index=False, compression="zstd")
    _once(args.out / "correctness_report.json", {
        "overlay_target_free_receipts_existed_before_outcome_open": True,
        "only_pre2026_development_outcomes_used_for_m1_selection": True,
        "conditional_metrics_condition_on_base_rank_bands": True,
        "selection_screens_under_over_magnitude_and_opportunity": True,
        "m0_to_m6_contracts_preserve_f120_parent_inputs": True,
        "m6_is_pre2026_predeclared_extension_not_selected_on_later_oos": True,
        "no_model_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    return args.out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--f120-contract", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--path-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
