#!/usr/bin/env python3
"""Stage 1: P8u Base target-family screen under the precision/preservation metric.

The P8u router is a fixed, target-free gate: exactly its timestamp-local top
50% identities form the Base population.  This runner fits a cheap
``rank_xendcg`` model for several supervised target geometries, writes every
held score before looking at realised policy outcomes, then selects one target
geometry per family with the externally-defined timestamp-local
precision-plus-preservation ``Score_Stable``.

It is deliberately *only* target stage 1.  Feature selection, loss-geometry
search, weights, cross-model comparisons, Meta, MC1, and portfolio replay are
separate stages and cannot be silently selected here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import COMPONENTS, stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_target_screen_v1"
SEED = 1729
IDENTITY = base.IDENTITY
F72_SELECTION = base.F72_SELECTION
# The P8u label ledger begins in July 2025.  November therefore is the first
# fold with a full three-month, label-resolved training window after the
# 28-day reserve.  These five folds span two calendar years and eight months.
DEFAULT_HELD_MONTHS = ("2025-11", "2026-01", "2026-03", "2026-05", "2026-07")


@dataclass(frozen=True)
class Arm:
    family: str
    target: str
    geometry: str

    @property
    def key(self) -> str:
        return f"{self.family}__{self.geometry}"


ARMS: tuple[Arm, ...] = (
    Arm("policy_ordinal", "t0_policy_ordinal", "fixed_0_50_100_200_400"),
    Arm("policy_ordinal", "t0_policy_ordinal", "fixed_50_100_150_250_400"),
    Arm("raw_bps", "t1_raw_bps", "balanced_quantile6"),
    Arm("raw_bps", "t1_raw_bps", "tail_quantile6"),
    Arm("raw_bps", "t1_raw_bps", "equal_width6"),
    Arm("sqrt_atr", "t2_sqrt_atr", "balanced_quantile6"),
    Arm("sqrt_atr", "t2_sqrt_atr", "tail_quantile6"),
    Arm("sqrt_atr", "t2_sqrt_atr", "equal_width6"),
    Arm("atr", "t3_atr", "balanced_quantile6"),
    Arm("atr", "t3_atr", "tail_quantile6"),
    Arm("atr", "t3_atr", "equal_width6"),
)
CONTROL_KEY = ARMS[0].key


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _utc_month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token.strip()}-01", tz="UTC")


def _parse_months(tokens: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(_utc_month(item) for item in tokens.split(",") if item.strip())
    if len(months) < 5 or tuple(sorted(months)) != months:
        raise ValueError("held months need at least five increasing month boundaries")
    span = (months[-1].year - months[0].year) * 12 + months[-1].month - months[0].month
    if len({item.year for item in months}) < 2 or span < 8:
        raise ValueError("held months must span at least eight months and two calendar years")
    return months


def _parse_arms(tokens: str) -> tuple[Arm, ...]:
    requested = {item.strip() for item in tokens.split(",") if item.strip()}
    selected = tuple(arm for arm in ARMS if arm.key in requested)
    if not selected or requested != {arm.key for arm in selected}:
        unknown = sorted(requested.difference({arm.key for arm in ARMS}))
        raise ValueError(f"unknown target-screen arms {unknown}")
    if CONTROL_KEY not in {arm.key for arm in selected}:
        raise ValueError(f"the fixed P8u policy-ordinal control {CONTROL_KEY} is required")
    return selected


def _policy_floor_labels(values: pd.Series, floor: float) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(raw).all():
        raise AssertionError("policy ordinal target contains an invalid training value")
    # Grade 0 is <= floor; preserve the canonical 0/50/100/200/400
    # geometry for the ordinary policy-ordinal control and use the declared
    # shifted 50/100/150/250/400 geometry for the +50-floor challenger.
    edges = np.asarray(
        (0.0, 50.0, 100.0, 200.0, 400.0) if floor == 0.0 else (50.0, 100.0, 150.0, 250.0, 400.0),
        dtype=float,
    )
    # The declared lower band is inclusive (for example ``<= +50`` for the
    # floor-50 geometry), so equality stays in the preceding grade.
    return np.searchsorted(edges, raw, side="left").clip(0, 5).astype(np.int8)


def _continuous_labels(values: pd.Series, geometry: str) -> tuple[np.ndarray, dict[str, object]]:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = raw[np.isfinite(raw)]
    if len(valid) < 1_000:
        raise AssertionError("insufficient finite strict-training target values")
    p02, p98 = np.quantile(valid, (.02, .98))
    clipped = np.clip(raw, p02, p98)
    if geometry == "balanced_quantile6":
        quantiles = np.asarray((1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6))
        edges = np.quantile(clipped, quantiles)
    elif geometry == "tail_quantile6":
        # Higher-resolution positive tail, while retaining three broad lower
        # strata so the objective does not become a sparse-tail classifier.
        quantiles = np.asarray((.40, .65, .80, .92, .98))
        edges = np.quantile(clipped, quantiles)
    elif geometry == "equal_width6":
        edges = np.linspace(p02, p98, 7, dtype=float)[1:-1]
    else:
        raise ValueError(f"unsupported continuous target geometry {geometry}")
    labels = np.searchsorted(edges, clipped, side="right").clip(0, 5).astype(np.int8)
    return labels, {
        "training_only": True,
        "clip_p02": float(p02),
        "clip_p98": float(p98),
        "geometry": geometry,
        "edges": [float(item) for item in edges],
        "collapsed_edges": int(np.sum(np.diff(edges) <= 0.0)),
    }


def _labels(train: pd.DataFrame, arm: Arm) -> tuple[np.ndarray, dict[str, object]]:
    spec = base.TARGETS[arm.target]
    if arm.family == "policy_ordinal":
        floor = 0.0 if arm.geometry.startswith("fixed_0_") else 50.0
        # ``policy_ordinal_grade`` is a pre-existing discrete grade, not the
        # raw bps required to build a distinct shifted-floor geometry.  The
        # policy bps are still eligible only when the TargetSpec validity and
        # availability contracts used by _train_rows hold.
        return _policy_floor_labels(train["policy_net_bps"], floor), {
            "training_only": False,
            "geometry": arm.geometry,
            "policy_floor_bps": floor,
            "edges_bps": ([0.0, 50.0, 100.0, 200.0, 400.0] if floor == 0.0 else [50.0, 100.0, 150.0, 250.0, 400.0]),
        }
    return _continuous_labels(train[spec.value_column], arm.geometry)


def _train_rows(window: pd.DataFrame, arm: Arm, reserve: pd.Timestamp, cap: int) -> pd.DataFrame:
    spec = base.TARGETS[arm.target]
    valid = window[spec.valid_column].fillna(False).astype(bool)
    available = pd.to_datetime(window[spec.available_column], utc=True, errors="coerce")
    numeric = np.isfinite(pd.to_numeric(window[spec.value_column], errors="coerce"))
    train = window.loc[window["__decision_ts__"].lt(reserve) & valid & numeric & available.lt(reserve)].copy()
    train = base._sample_complete_queries(train, cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(train) < 8_000 or train["__decision_ts__"].nunique() < 40:
        raise AssertionError(f"{arm.key}: insufficient P8u strict train support")
    return train


def _model(seed: int, jobs: int) -> LGBMRanker:
    return LGBMRanker(
        objective="rank_xendcg", metric="ndcg", n_estimators=180, learning_rate=.05,
        max_depth=4, num_leaves=15, min_child_samples=260,
        subsample=.80, subsample_freq=1, colsample_bytree=.80,
        reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001,
        random_state=seed, n_jobs=jobs, deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _target_free_schema(frame: pd.DataFrame) -> None:
    forbidden = [column for column in frame.columns if any(token in column.lower() for token in base.PROHIBITED_SCORE_TOKENS)]
    if forbidden:
        raise AssertionError(f"target-free held score contains forbidden columns {forbidden}")
    if not frame.columns.tolist() == [*IDENTITY, "base_score", "base_rank_ts"]:
        raise AssertionError("unexpected P8u target-free score schema")


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, selection_receipt: Path,
    out: Path, held_months: Sequence[pd.Timestamp], arms: Sequence[Arm], train_months: int,
    reserve_days: int, train_cap: int, n_jobs: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    fields = base._load_f72_fields(selection_receipt)
    out.mkdir(parents=True)
    held_parts: dict[tuple[str, str], pd.DataFrame] = {}
    component_parts: dict[str, list[pd.DataFrame]] = {arm.key: [] for arm in arms}
    fold_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    lineage_paths: list[Path] = [selection_receipt]

    for fold_index, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=reserve_days)
        end = held_month + pd.offsets.MonthBegin(1)
        window, coverage = base._load_window(
            candidate_root=None, feature_root=tuple(feature_roots), label_root=label_root,
            router_root=router_root, start=reserve - pd.DateOffset(months=train_months), end=end, fields=fields,
        )
        coverage_rows.extend(coverage)
        held = window.loc[window["__decision_ts__"].ge(held_month) & window["__decision_ts__"].lt(end)].copy()
        if held.empty:
            raise AssertionError(f"{held_month:%Y-%m}: no routed held candidates")
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]].copy()
        for arm_index, arm in enumerate(arms):
            train = _train_rows(window, arm, reserve, train_cap)
            y, geometry = _labels(train, arm)
            x_train, medians = base._numeric_matrix(train, fields)
            x_held, _ = base._numeric_matrix(held, fields, medians)
            model = _model(SEED + 10_000 * fold_index + arm_index, n_jobs)
            model.fit(x_train, y, group=base._query_groups(train))
            target_free = held.loc[:, list(IDENTITY)].copy()
            target_free["base_score"] = model.predict(x_held).astype(np.float32)
            target_free["base_rank_ts"] = base._rank_desc(target_free, "base_score")
            _target_free_schema(target_free)
            score_path = out / "target_free_scores" / arm.key / f"month={held_month:%Y-%m}.parquet"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            target_free.to_parquet(score_path, index=False, compression="zstd")
            # Only now is the immutable target-free receipt joined to policy
            # outcomes for selection diagnostics.
            scored = target_free.merge(labels, on="candidate_id", how="left", validate="one_to_one")
            components = timestamp_components(scored, score_column="base_score")
            components["arm"] = arm.key
            components["held_month"] = f"{held_month:%Y-%m}"
            component_parts[arm.key].append(components)
            held_parts[(arm.key, f"{held_month:%Y-%m}")] = components
            fold_rows.append({
                "arm": arm.key, "family": arm.family, "target": arm.target, "geometry": arm.geometry,
                "held_month": f"{held_month:%Y-%m}", "train_rows": int(len(train)),
                "train_queries": int(train["__decision_ts__"].nunique()), "held_rows": int(len(held)),
                "held_queries": int(held["__decision_ts__"].nunique()), "score_path": str(score_path),
                "target_geometry": json.dumps(geometry, sort_keys=True),
                "target_free_before_outcome_join": True, "router_top50_identity_exact": True,
                "feature_medians_fit_train_only": True,
            })
            _progress(out, stage="fold_arm_complete", arm=arm.key, held_month=f"{held_month:%Y-%m}", train_rows=len(train), held_rows=len(held))
            del model, x_train, x_held, target_free, scored, train
            gc.collect()
        del window, held
        gc.collect()

    control = pd.concat(component_parts[CONTROL_KEY], ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    summaries: list[dict[str, object]] = []
    all_components: list[pd.DataFrame] = []
    for arm in arms:
        candidate = pd.concat(component_parts[arm.key], ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
        summary, normalised = stable_score(candidate, control)
        summary_row = {
            "arm": arm.key, "family": arm.family, "target": arm.target, "geometry": arm.geometry,
            **summary.__dict__, **{f"mean_{name}": float(candidate[name].mean()) for name in COMPONENTS},
            "mean_utility_recall20": float(candidate["utility_recall20"].mean()),
            "mean_top2_coverage": float(candidate["dtp2_bps_coverage"].mean()),
            "mean_top5_coverage": float(candidate["dtp5_bps_coverage"].mean()),
            "mean_top10_coverage": float(candidate["dtp10_bps_coverage"].mean()),
            "utility_recall20_coverage": float(candidate["utility_recall20_eligible"].mean()),
            "residual_utility_recall10_to30_coverage": float(candidate["residual_utility_recall10_to30_eligible"].mean()),
        }
        summaries.append(summary_row)
        normalised["arm"] = arm.key
        all_components.append(normalised)
    summary_frame = pd.DataFrame(summaries).sort_values(["family", "score_stable", "arm"], ascending=[True, False, True], kind="stable")
    winners = summary_frame.groupby("family", sort=True).head(1).sort_values("family", kind="stable")
    pd.concat(all_components, ignore_index=True).to_parquet(out / "timestamp_components.parquet", index=False, compression="zstd")
    summary_frame.to_parquet(out / "target_summary.parquet", index=False, compression="zstd")
    winners.to_parquet(out / "family_winners.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_rows).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).drop_duplicates(subset=["month"], keep="last").to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "correctness_report.json", {
        "p8u_router_top50_identity_exact": bool(all(item["router_top50_identity_exact"] for item in fold_rows)),
        "all_held_scores_target_free_before_outcomes": bool(all(item["target_free_before_outcome_join"] for item in fold_rows)),
        "all_feature_medians_train_only": bool(all(item["feature_medians_fit_train_only"] for item in fold_rows)),
        "all_train_labels_resolved_before_reserve": True,
        "score_schema_excludes_outcomes": True,
        "candidate_population_is_p8u_only": True,
        "costs_reused_only_from_canonical_policy_label": True,
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u target-family screen only; no Meta, MC1, admission, portfolio, live, or exchange mutation",
        "architecture": "exact P8u Router top50 -> one Base scores all routed rows; no numeric router Base input and no Base post-route cutoff",
        "router": {"root": str(router_root), "fraction": .50, "numeric_model_input": False},
        "features": {"selection_receipt": str(selection_receipt), "count": len(fields), "fields": list(fields)},
        "model": {"family": "LightGBM", "objective": "rank_xendcg", "params": _model(SEED, 1).get_params()},
        "query": "exact decision timestamp x long side",
        "targets": [arm.__dict__ for arm in arms],
        "selection_metric": {
            "BaseScore": "0.30*DTP2 + 0.30*DTP5 + 0.20*DTP10 + 0.20*ResidualUR10_to30, each normalised to the matched fixed policy-ordinal control",
            "ScoreStable": "weekly robust mean between Q20/Q80 + 0.5*mean(Q15,Q10,Q5)",
            "outcome": "canonical rich-policy net bps; ranks remain over all target-free routed candidates",
        },
        "strict_oof": {"train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "feature_roots": [str(item) for item in feature_roots], "label_root": str(label_root),
        "input_sha256": base._sha256(lineage_paths),
        "selected_family_winners": winners.loc[:, ["family", "arm", "score_stable"]].to_dict("records"),
    })
    _progress(out, stage="complete", arms=len(arms), winners=winners.arm.tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal feature roots")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, default=F72_SELECTION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default=",".join(DEFAULT_HELD_MONTHS))
    parser.add_argument("--arms", default=",".join(arm.key for arm in ARMS))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000:
        raise ValueError("invalid strict OOF P8u target-screen support contract")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(),
        selection_receipt=args.selection_receipt.resolve(), out=args.out.resolve(),
        held_months=_parse_months(args.held_months), arms=_parse_arms(args.arms),
        train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap, n_jobs=args.n_jobs,
    ))


if __name__ == "__main__":
    main()
