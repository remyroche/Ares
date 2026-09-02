#!/usr/bin/env python3
"""Re-evaluate immutable strict-OOF Meta scores at bounded rank authorities.

This is a research-only diagnostic for the P8u Meta objective.  It never
refits an upstream Base or Meta model: every candidate score must already be
persisted by ``run_strict_r3_p8u_meta_lgbm_objective_screen_v1.py`` before
this program opens any held policy/path labels.  The utility is to distinguish
an unhelpful target from a potentially useful, but over-authorised, correction
layer.

It intentionally stops before MC1, admission, portfolio construction, or any
live/exchange operation.  Direct strict-OOF and then frozen downstream replay
remain the advancement path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = screen.IDENTITY


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _months(raw: dict[str, Any]) -> tuple[pd.Timestamp, ...]:
    return tuple(screen._utc_month(value) for value in raw["folds"]["held_months"])


def _base_train_frame(*, base_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [*IDENTITY, "base_score", "base_rank_ts"]
    parts: list[pd.DataFrame] = []
    for month in screen._month_range(start, end):
        panel = pd.read_parquet(screen._base_path(base_root, month), columns=columns)
        panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
        parts.append(panel.loc[panel.__decision_ts__.ge(start) & panel.__decision_ts__.lt(end)].copy())
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(IDENTITY).any():
        raise AssertionError("duplicate target-free base identity while rebuilding fold anchor")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _held_anchor(
    *, raw: dict[str, Any], base_root: Path, policy: pd.DataFrame, path_root: Path, held_month: pd.Timestamp,
) -> Any:
    reserve = held_month - pd.Timedelta(days=int(raw["folds"]["resolved_label_reserve_days"]))
    start = reserve - pd.DateOffset(months=int(raw["folds"]["train_months"]))
    train = _base_train_frame(base_root=base_root, start=start, end=reserve)
    labelled = screen._labelled(train, policy, path_root, start, reserve)
    valid = screen._valid_label(labelled, reserve)
    return screen._fit_anchor(labelled, valid)


def _spec_with_authority(raw: dict[str, Any], authority: float, config_path: Path) -> screen.Spec:
    if not 0.0 <= authority <= 1.0:
        raise ValueError("authority must lie in [0, 1]")
    payload = copy.deepcopy(raw)
    payload["provisional_meta_blend"]["base_rank_weight"] = 1.0 - authority
    payload["provisional_meta_blend"]["meta_rank_weight"] = authority
    return screen.Spec(raw=payload, config_path=config_path)


def _read_score(path: Path) -> pd.DataFrame:
    required = [*IDENTITY, "base_score", "base_rank_ts", "meta_raw_score", "meta_rank_ts"]
    score = pd.read_parquet(path, columns=required)
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
    if score.duplicated(IDENTITY).any() or not score.side_name.eq("long").all():
        raise AssertionError(f"{path}: invalid target-free score identity")
    return score


def _aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metrics = [
        "valid_policy_rows", "weeks", "smeta_week_robust_average", "smeta_week_lower_tail", "sstable_meta",
        "residual_spearman_ic", "conditional_mi_meta_policy_given_base", "mean_top2_substitution_ev_bps",
        "mean_top2_substitution_utility_bps", "mean_admission_substitution_utility_bps", "worst_week_smeta",
        "worst_week_delta_ev_top2_bps", "mean_iccond", "mean_utility_spreadcond_bps",
        "mean_potential_utility_recall", "mean_net_rescue_separation_bps",
    ]
    result = frame.groupby(["score_root", "trial", "meta_rank_weight"], sort=True)[metrics].mean().reset_index()
    result = result.sort_values(
        ["sstable_meta", "mean_top2_substitution_ev_bps", "mean_admission_substitution_utility_bps", "trial"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, action="append", required=True)
    parser.add_argument("--weights", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    weights = tuple(float(value) for value in args.weights.split(",") if value.strip())
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("--weights must be a unique, non-empty comma-separated list")
    if any(not 0.0 <= value <= 1.0 for value in weights):
        raise ValueError("all --weights must lie in [0, 1]")
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")

    config = args.config.resolve()
    raw = json.loads(config.read_text())
    base_root = (ROOT / str(raw["source"]["base_target_free_root"])).resolve()
    policy_path = (ROOT / str(raw["source"]["policy_labels"])).resolve()
    path_root = (ROOT / str(raw["source"]["path_labels"])).resolve()
    policy = screen._read_policy(policy_path)
    months = _months(raw)
    args.out.mkdir(parents=True)

    selected: dict[Path, tuple[str, ...]] = {}
    for root in (path.resolve() for path in args.score_root):
        summary_path = root / "objective_summary.parquet"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = pd.read_parquet(summary_path).sort_values("rank", kind="stable")
        names = tuple(summary.head(args.top_n).trial.astype(str))
        if not names:
            raise AssertionError(f"{root}: no source trials")
        selected[root] = names

    (args.out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_p8u_meta_authority_v1",
        "scope": "research-only post-score OOF authority diagnostic; no model refit, MC1, admission, portfolio, live, or exchange mutation",
        "config": str(config), "config_sha256": _sha(config), "weights": list(weights), "top_n": int(args.top_n),
        "score_roots": {str(root): list(names) for root, names in selected.items()},
        "source": {"base": str(base_root), "policy": str(policy_path), "path": str(path_root)},
        "source_hashes": {"policy": _sha(policy_path)},
        "causality": "all source scores were persisted target-free before held labels are joined; fold anchor uses only labels resolved before the held reserve",
    }, indent=2, sort_keys=True) + "\n")

    anchors = {
        month: _held_anchor(raw=raw, base_root=base_root, policy=policy, path_root=path_root, held_month=month)
        for month in months
    }
    results: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    bands_parts: list[pd.DataFrame] = []
    for root, names in selected.items():
        for trial in names:
            # Keep only one trial's seven held panels resident.  This is a
            # post-score diagnostic over large panels, so retaining every
            # root/trial/month would inflate memory without changing a result.
            held_scores: dict[pd.Timestamp, pd.DataFrame] = {}
            held_labelled: dict[pd.Timestamp, pd.DataFrame] = {}
            for month in months:
                score_path = root / "target_free_scores" / trial / f"month={month:%Y-%m}.parquet"
                score = _read_score(score_path)
                held_scores[month] = score
                held_labelled[month] = screen._labelled(score, policy, path_root, month, screen._month_end(month))
            for weight in weights:
                spec = _spec_with_authority(raw, weight, config)
                for month in months:
                    weekly, bands, metrics = screen._metrics(
                        score=held_scores[month], held_labelled=held_labelled[month],
                        held_anchor=anchors[month], spec=spec,
                    )
                    results.append({
                        "score_root": root.name, "trial": trial, "held_month": f"{month:%Y-%m}",
                        "meta_rank_weight": weight, **metrics,
                    })
                    weekly["score_root"] = root.name; weekly["trial"] = trial
                    weekly["held_month"] = f"{month:%Y-%m}"; weekly["meta_rank_weight"] = weight
                    weekly_parts.append(weekly)
                    if not bands.empty:
                        bands["score_root"] = root.name; bands["trial"] = trial
                        bands["held_month"] = f"{month:%Y-%m}"; bands["meta_rank_weight"] = weight
                        bands_parts.append(bands)
    fold = pd.DataFrame(results)
    fold.to_parquet(args.out / "authority_fold_metrics.parquet", index=False, compression="zstd")
    _aggregate(results).to_parquet(args.out / "authority_summary.parquet", index=False, compression="zstd")
    pd.concat(weekly_parts, ignore_index=True).to_parquet(args.out / "authority_weekly_metrics.parquet", index=False, compression="zstd")
    (pd.concat(bands_parts, ignore_index=True) if bands_parts else pd.DataFrame()).to_parquet(
        args.out / "authority_base_band_metrics.parquet", index=False, compression="zstd"
    )
    (args.out / "correctness_report.json").write_text(json.dumps({
        "source_scores_preexisted_as_target_free": True,
        "held_labels_joined_only_after_score_read": True,
        "held_anchor_uses_prior_resolved_labels_only": True,
        "no_model_refit": True,
        "no_feature_contract_mutation": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
