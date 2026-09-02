#!/usr/bin/env python3
"""Outcome-only evaluation of two already-persisted target-free Under heads.

Both scoring roots must already exist before this program opens policy/path
outcomes.  It evaluates Base, a declared control confirmation blend, and one
declared challenger using the same valid rich-policy rows and timestamp-local
selection.  It does not fit a model, map EV, admit a trade, or mutate live
state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_under_targetfree_pair_evaluation_v1"
IDENTITY = screen.IDENTITY
FRACTIONS = (0.01, 0.02, 0.05, 0.10)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    output = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if not output or len(output) != len(set(output)) or tuple(sorted(output)) != output:
        raise ValueError("--months must be unique chronological YYYY-MM values")
    return output


def _score_path(root: Path, trial: str, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / trial / f"month={month:%Y-%m}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_target_free(root: Path, trial: str, month: pd.Timestamp) -> pd.DataFrame:
    path = _score_path(root, trial, month)
    score = pd.read_parquet(path)
    required = {*IDENTITY, "base_rank_ts", "meta_raw_score", "meta_rank_ts", "target_free"}
    missing = sorted(required.difference(score.columns))
    if missing or not score.target_free.fillna(False).astype(bool).all():
        raise AssertionError(f"{path}: not a complete target-free score receipt ({missing})")
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
    if score.duplicated(IDENTITY).any() or not score.side_name.eq("long").all():
        raise AssertionError(f"{path}: invalid target-free identity")
    return score.loc[:, [*IDENTITY, "base_score", "base_rank_ts", "meta_raw_score", "meta_rank_ts"]].copy()


def _rank_desc(frame: pd.DataFrame, column: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", column]].copy()
    work["row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(work), dtype=np.float32)
    result[work.row.to_numpy(np.int64)] = (1.0 - (ordinal - .5) / size).astype(np.float32)
    return result


def _timestamp_metric(frame: pd.DataFrame, score: str, fraction: float) -> tuple[float, int]:
    selected: list[float] = []
    count = 0
    for _, part in frame.groupby("__decision_ts__", sort=False):
        k = max(1, int(np.ceil(len(part) * fraction)))
        chosen = part.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(k)
        selected.extend(chosen.policy_net_bps.to_numpy(float)); count += len(chosen)
    return float(np.mean(selected)) if selected else float("nan"), count


def _timestamp_ic(frame: pd.DataFrame, score: str) -> float:
    values: list[float] = []
    for _, part in frame.groupby("__decision_ts__", sort=False):
        if len(part) < 8 or part[score].nunique() < 3 or part.policy_net_bps.nunique() < 3:
            continue
        value = float(spearmanr(part[score], part.policy_net_bps).statistic)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _rows(frame: pd.DataFrame, month: str, name: str, score: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "month": month, "score_family": name, "valid_rows": int(len(frame)),
        "timestamps": int(frame.__decision_ts__.nunique()), "timestamp_rank_ic": _timestamp_ic(frame, score),
    }
    for fraction in FRACTIONS:
        ev, count = _timestamp_metric(frame, score, fraction)
        token = f"top{int(fraction * 100)}pct" if fraction >= .01 else str(fraction)
        result[f"{token}_net_bps"] = ev
        result[f"{token}_rows"] = count
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--challenger-root", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--path-label-root", type=Path, required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-tag", default="f120", help="stable label for the frozen control, e.g. f120")
    parser.add_argument("--challenger-tag", default="f123", help="stable label for the challenger, e.g. f125")
    args = parser.parse_args()
    control, challenger, policy_path, path_root, out = (args.control_root.resolve(), args.challenger_root.resolve(), args.policy_labels.resolve(), args.path_label_root.resolve(), args.out.resolve())
    months = _months(args.months)
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    # Assert every receipt before opening either label source.
    for root in (control, challenger):
        for month in months:
            _read_target_free(root, args.trial, month)

    out.mkdir(parents=True)
    policy = screen._read_policy(policy_path)
    per_month: list[dict[str, Any]] = []
    output_rows: list[pd.DataFrame] = []
    for month in months:
        c = _read_target_free(control, args.trial, month).rename(columns={"meta_raw_score": "f120_raw", "meta_rank_ts": "f120_rank"})
        x = _read_target_free(challenger, args.trial, month).rename(columns={
            "base_score": "challenger_base_score",
            "base_rank_ts": "challenger_base_rank_ts",
            "meta_raw_score": "f123_raw",
            "meta_rank_ts": "f123_rank",
        })
        keys = list(IDENTITY)
        pair = c.merge(x.loc[:, [*keys, "challenger_base_score", "challenger_base_rank_ts", "f123_raw", "f123_rank"]], on=keys, how="inner", validate="one_to_one")
        if len(pair) != len(c) or len(pair) != len(x):
            raise AssertionError(f"{month:%Y-%m}: target-free F120/F123 identity mismatch")
        if not np.array_equal(pair["base_score"].to_numpy(), pair["challenger_base_score"].to_numpy()) or not np.array_equal(pair["base_rank_ts"].to_numpy(), pair["challenger_base_rank_ts"].to_numpy()):
            raise AssertionError(f"{month:%Y-%m}: control/challenger Base coordinate mismatch")
        path = screen._read_path(path_root, month, month + pd.offsets.MonthBegin(1))
        labelled = pair.merge(path, on=keys, how="left", validate="one_to_one").merge(policy, on="candidate_id", how="left", validate="one_to_one")
        labelled["atr_bps"] = pd.to_numeric(labelled.path_arch_atr_fraction, errors="coerce") * 10_000.0
        valid = screen._valid_label(labelled)
        labelled = labelled.loc[valid].copy()
        if len(labelled) < 2_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient matched valid rich-policy outcomes")
        labelled["base_rank"] = labelled.base_rank_ts
        labelled["f120_current"] = .75 * labelled.base_rank_ts + .25 * labelled.f120_rank
        labelled["f123_current"] = .75 * labelled.base_rank_ts + .25 * labelled.f123_rank
        labelled["f120_current_rank"] = _rank_desc(labelled, "f120_current")
        labelled["f123_current_rank"] = _rank_desc(labelled, "f123_current")
        held = f"{month:%Y-%m}"
        per_month.extend([
            _rows(labelled, held, "base", "base_rank"),
            _rows(labelled, held, f"under_{args.control_tag}_raw", "f120_rank"),
            _rows(labelled, held, f"under_{args.challenger_tag}_raw", "f123_rank"),
            _rows(labelled, held, f"current_{args.control_tag}_75_25", "f120_current_rank"),
            _rows(labelled, held, f"current_{args.challenger_tag}_75_25", "f123_current_rank"),
        ])
        output_rows.append(labelled.loc[:, [*keys, "policy_net_bps", "base_rank", "f120_rank", "f123_rank", "f120_current_rank", "f123_current_rank"]].assign(month=held))
    metric = pd.DataFrame(per_month)
    numeric = [column for column in metric.columns if column not in {"month", "score_family"}]
    aggregate = metric.groupby("score_family", sort=False)[numeric].mean().reset_index()
    pivot = aggregate.set_index("score_family")
    ref = pivot.loc[f"current_{args.control_tag}_75_25"]
    delta_rows = []
    for name in (f"current_{args.challenger_tag}_75_25", f"under_{args.challenger_tag}_raw"):
        candidate = pivot.loc[name]
        delta_rows.append({"score_family": name, **{f"delta_vs_{args.control_tag}_{field}": float(candidate[field] - ref[field]) for field in ["timestamp_rank_ic", "top1pct_net_bps", "top2pct_net_bps", "top5pct_net_bps", "top10pct_net_bps"]}})
    metric.to_parquet(out / "per_month_matched_metrics.parquet", index=False, compression="zstd")
    aggregate.to_parquet(out / "aggregate_matched_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(delta_rows).to_parquet(out / "challenger_delta_vs_control.parquet", index=False, compression="zstd")
    pd.concat(output_rows, ignore_index=True).to_parquet(out / "matched_outcome_score_panel.parquet", index=False, compression="zstd")
    correctness = {
        "all_target_free_receipts_existed_before_outcome_open": True,
        "control_challenger_target_free_identities_match_exactly": True,
        "control_challenger_base_coordinates_match_exactly": True,
        "outcomes_used_only_for_post_score_metrics": True,
        "validity_matches_canonical_policy_and_path_contract": True,
        "no_model_mc1_admission_portfolio_live_or_exchange_mutation": True,
    }
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline matched outcome evaluation of persisted target-free Under scores only",
        "months": [f"{item:%Y-%m}" for item in months], "control_root": str(control), "challenger_root": str(challenger),
        "trial": args.trial, "control_tag": args.control_tag, "challenger_tag": args.challenger_tag,
        "policy_labels": str(policy_path), "path_label_root": str(path_root),
        "source_hashes": {"control": _sha(control), "challenger": _sha(challenger), "policy": _sha(policy_path), "path": _sha(path_root)},
        "correctness": correctness,
    })


if __name__ == "__main__":
    main()
