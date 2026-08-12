#!/usr/bin/env python3
"""Focused, causal reliability re-ranking for the strict-R3 long stack.

The prior LDF/Bayesian tests trained on the *global* top 30% of an upstream
score distribution and then only changed position size.  That has two
limitations: a position already in the global score tail is almost always at
the sizing cap, and the training domain still contains many candidates that
would never compete at their decision timestamp.

This diagnostic instead:

* selects the top 30% *within each decision timestamp* before fitting;
* trains a small LambdaRank reliability head only on that actionable domain;
* uses deliberately coarse policy-net relevance grades, rather than noisy raw
  residual regression; and
* applies its output as a bounded score correction, not a size change.

All ranking is based on contemporaneously available scores.  This first
producer is deliberately a *global-tail ranking diagnostic*: it does not
claim an executable causal-admission replay after changing the score.  An
executable arm must rebuild the 21-day EV map prequentially against the
corrected score.  No held outcome is used to fit, normalise, or select a held
score.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
import pyarrow.parquet as pq
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEED = 20260811
TOP_FRACTION = 0.30
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
ALPHAS = (0.025, 0.05, 0.10)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str

    def grade(self, net_bps: np.ndarray) -> np.ndarray:
        # The bps bands deliberately exceed the ordinary per-trade path noise.
        # Every valid top-30 row remains in the query; ambiguous observations
        # are a neutral relevance grade, not silently dropped supervision.
        if self.name == "coarse_net5":
            return np.select(
                [net_bps <= -200.0, net_bps <= -50.0, net_bps < 50.0, net_bps < 150.0],
                [0, 1, 2, 3], default=4,
            ).astype(np.int8)
        if self.name == "robust_clear3":
            return np.select(
                [net_bps <= -100.0, net_bps < 50.0], [0, 1], default=2,
            ).astype(np.int8)
        if self.name == "low_risk_binary":
            return np.where(net_bps <= -100.0, 0, 1).astype(np.int8)
        raise ValueError(f"unknown target {self.name!r}")


TARGETS = (
    TargetSpec(
        "coarse_net5",
        "policy-net ordinal bins: <=-200, (-200,-50], (-50,50), [50,150), >=150 bps",
    ),
    TargetSpec(
        "robust_clear3",
        "robust adverse / unresolved / robust clear: <=-100, (-100,50), >=50 bps",
    ),
    TargetSpec(
        "low_risk_binary",
        "coarse downside target: policy net > -100 bps versus policy net <= -100 bps",
    ),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, type=Path, help="Primary OOF surface.")
    parser.add_argument(
        "--history-surface", type=Path,
        help="Earlier compatible surface used only as strict prequential fitting support.",
    )
    parser.add_argument("--feature-contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--first-held-month", required=True)
    parser.add_argument("--last-held-month", required=True)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument(
        "--feature-mode", choices=("compact84", "trust_regime"), default="compact84",
        help="`trust_regime` restricts the learner to causal support/OOD/K9/recent-state/committee/regime inputs plus upstream conditioning scores.",
    )
    parser.add_argument(
        "--targets", default=",".join(target.name for target in TARGETS),
        help="Comma-separated predeclared target names.  Use a frozen development winner for validation.",
    )
    parser.add_argument(
        "--alphas", default=",".join(str(alpha) for alpha in ALPHAS),
        help="Comma-separated bounded score-correction authorities.",
    )
    return parser.parse_args()


def _timestamp_top_fraction(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark the score-leading candidates using only their current timestamp.

    Candidate ID is the deterministic tie break.  We select ``ceil(0.30*n)``
    rows per timestamp, never a percentile computed over the held month.
    """

    ordered = frame.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).copy()
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    keep_count = np.maximum(1, np.ceil(count * TOP_FRACTION).astype(int))
    ordered["timestamp_top30"] = position < keep_count
    ordered["timestamp_candidate_count"] = count.astype(np.int16)
    ordered["timestamp_top30_count"] = keep_count.astype(np.int16)
    return ordered.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = frame["__decision_ts__"].dt.strftime("%Y-%m")
    tokens = sorted(month.unique())
    quota = max(1, int(math.ceil(cap / len(tokens))))
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for token in tokens:
        block = frame.loc[month.eq(token)]
        if len(block) > quota:
            block = block.iloc[np.sort(rng.choice(len(block), quota, replace=False))]
        parts.append(block)
    output = pd.concat(parts, ignore_index=True)
    if len(output) > cap:
        output = output.iloc[np.sort(rng.choice(len(output), int(cap), replace=False))]
    return output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()


def _numeric_matrix(frame: pd.DataFrame, fields: tuple[str, ...], medians: np.ndarray) -> np.ndarray:
    raw = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return np.where(np.isfinite(raw), raw, medians).astype(np.float32)


def _trust_regime_candidates(columns: set[str]) -> tuple[str, ...]:
    """Return causal trust/risk candidates under one frozen K9 identity.

    The nine posterior membership coordinates are an explicit exception to the
    normal no-raw-K9 rule: this runner rejects mixed geometry hashes per fold,
    so ``k09__cluster_XX__membership`` has one persistent frozen Oct--Dec
    meaning throughout its training and held rows.  Distances/confidences are
    intentionally still excluded; the request is for membership posterior.
    """

    prefixes = (
        "k9_", "leaf_", "cluster_", "reliability_", "residual_heads_", "continuous_regime__",
    )
    conditioning = {
        "final_score", "base_score", "base_anchor_bps", "base_rank", "consensus_rank",
        "upstream", "residual_rank", "severe200_probability",
    }
    output = [
        field for field in sorted(columns)
        if field in conditioning or field.startswith(prefixes)
    ]
    posterior = sorted(
        field for field in columns
        if field.startswith("k09__cluster_") and field.endswith("__membership")
    )
    if len(posterior) not in {0, 9}:
        raise ValueError(f"expected exactly nine frozen K9 posterior coordinates, found {len(posterior)}")
    return tuple([*output, *posterior])


def _eligible_fields(train: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Per-fold 90% coverage/variance gate for a genuine trust-only learner."""

    kept: list[str] = []
    for field in candidates:
        value = pd.to_numeric(train[field], errors="coerce").to_numpy(float)
        finite = np.isfinite(value)
        if finite.mean() < 0.90 or finite.sum() < 500:
            continue
        if np.nanstd(value[finite]) <= 1e-9:
            continue
        kept.append(field)
    if len(kept) < 12:
        raise ValueError(f"trust-regime coverage/variance gate left only {len(kept)} fields")
    return tuple(kept)


def _fit_ranker(
    train: pd.DataFrame, fields: tuple[str, ...], target: TargetSpec,
) -> tuple[LGBMRanker, np.ndarray, dict[str, object]]:
    net = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    grade = target.grade(net)
    query = train["__decision_ts__"].astype("int64").to_numpy()
    _, group_sizes = np.unique(query, return_counts=True)
    if len(train) < 1_000 or len(group_sizes) < 100 or np.unique(grade).size < 2:
        raise ValueError("insufficient top-30 timestamp-query support for focused ranker")
    matrix_raw = train.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    medians = np.nanmedian(matrix_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    matrix = np.where(np.isfinite(matrix_raw), matrix_raw, medians).astype(np.float32)
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3, 7, 12],
        n_estimators=220, learning_rate=0.03, max_depth=4, num_leaves=15,
        min_child_samples=350, colsample_bytree=0.75, subsample=0.80,
        reg_lambda=12.0, reg_alpha=0.15, lambdarank_norm=True,
        lambdarank_truncation_level=3, random_state=SEED, n_jobs=1, verbosity=-1,
    )
    model.fit(matrix, grade, group=group_sizes)
    audit = {
        "target": target.name, "target_description": target.description,
        "train_rows": len(train), "train_queries": len(group_sizes),
        "mean_query_size": float(np.mean(group_sizes)),
        "target_counts": {str(index): int(value) for index, value in zip(*np.unique(grade, return_counts=True))},
    }
    return model, medians, audit


def _within_timestamp_rank(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Rank a held prediction only against current-timestamp predictions."""

    work = frame.loc[:, ["__decision_ts__", "candidate_id"]].copy()
    work["__raw__"] = np.asarray(raw, dtype=float)
    work = work.sort_values(["__decision_ts__", "__raw__", "candidate_id"], ascending=[True, True, True], kind="stable")
    position = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    # Bottom=0 and top=1.  Singleton timestamp queries are neutral.
    work["focused_rank"] = np.divide(position, count - 1.0, out=np.full(len(work), 0.5), where=count > 1.0)
    return work.sort_index()["focused_rank"].to_numpy(np.float32)


def _tail_metrics(frame: pd.DataFrame, *, arm: str, kind: str) -> pd.DataFrame:
    if kind == "global":
        groups = [("all", frame)]
    elif kind == "month":
        groups = frame.groupby(frame["__decision_ts__"].dt.strftime("%Y-%m"), sort=True)
    elif kind == "week":
        groups = frame.groupby(frame["__decision_ts__"].dt.strftime("%G-W%V"), sort=True)
    else:
        raise ValueError(kind)
    rows: list[dict[str, object]] = []
    for period, block in groups:
        population = block.loc[np.isfinite(pd.to_numeric(block["corrected_score"], errors="coerce"))]
        for tail in TAILS:
            selected = population.nlargest(max(1, int(math.ceil(len(population) * tail))), "corrected_score", keep="first")
            valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "period_kind": kind, "period": str(period), "tail": tail,
                "selected_score_rows": len(selected), "valid_outcomes": len(valid),
                "outcome_coverage": float(len(valid) / max(1, len(selected))),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
    return pd.DataFrame(rows)


def _read_window(paths: tuple[Path, ...], columns: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in paths:
        piece = pd.read_parquet(
            path, columns=columns,
            filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
        )
        if len(piece):
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=columns)
    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise ValueError("surfaces overlap on candidate identities")
    return output


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    contract = json.loads(args.feature_contract.read_text())
    compact_fields = tuple(map(str, contract["compact_fields"]))
    if not compact_fields:
        raise ValueError("feature contract has no compact fields")
    start = pd.Timestamp(args.first_held_month, tz="UTC")
    end = pd.Timestamp(args.last_held_month, tz="UTC") + pd.offsets.MonthBegin(1)
    if start.day != 1 or end.day != 1 or start >= end:
        raise ValueError("held months must be ascending calendar starts")
    paths = tuple(path for path in (args.history_surface, args.surface) if path is not None)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    source_columns = set(pq.ParquetFile(paths[0]).schema_arrow.names)
    for path in paths[1:]:
        source_columns.intersection_update(pq.ParquetFile(path).schema_arrow.names)
    if args.feature_mode == "compact84":
        candidate_fields = compact_fields
    else:
        candidate_fields = _trust_regime_candidates(source_columns)
    absent = sorted(set(candidate_fields).difference(source_columns))
    if absent:
        raise ValueError(f"all source surfaces must expose feature candidates: {absent}")
    columns = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "geometry_bundle_sha256", "final_score",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps", *candidate_fields,
    ]))
    requested_names = tuple(token.strip() for token in str(args.targets).split(",") if token.strip())
    lookup = {target.name: target for target in TARGETS}
    unknown = sorted(set(requested_names).difference(lookup))
    if unknown:
        raise ValueError(f"unknown requested targets: {unknown}")
    targets = tuple(lookup[name] for name in requested_names)
    alphas = tuple(float(token.strip()) for token in str(args.alphas).split(",") if token.strip())
    if not targets or not alphas or any(not 0.0 < alpha <= 0.20 for alpha in alphas):
        raise ValueError("targets must be nonempty and every alpha must lie in (0, 0.20]")
    args.out_dir.mkdir(parents=True)
    all_predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold, cutoff in enumerate(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS", tz="UTC")):
        held_end = cutoff + pd.offsets.MonthBegin(1)
        train_start = cutoff - pd.DateOffset(months=int(args.train_months))
        frame = _read_window(paths, columns, train_start, held_end)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
        identities = frame["geometry_bundle_sha256"].dropna().astype(str).unique()
        if len(identities) != 1:
            raise ValueError(f"fold {cutoff:%Y-%m} mixes Geometry/K9 bundle semantics")
        frame = _timestamp_top_fraction(frame)
        train_all = frame.loc[
            frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["timestamp_top30"].astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        train = _equal_month_sample(train_all, int(args.train_cap), seed=SEED + fold)
        held = frame.loc[frame["__decision_ts__"].ge(cutoff)].copy()
        if held.empty:
            raise ValueError(f"no held candidates for {cutoff:%Y-%m}")
        baseline = held.loc[:, [
            "candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps", "final_score",
            "timestamp_top30", "timestamp_candidate_count", "timestamp_top30_count",
        ]].copy()
        baseline["focused_raw_score"] = np.nan
        baseline["focused_rank"] = 0.5
        for target in targets:
            fields = _eligible_fields(train, candidate_fields)
            model, medians, fit_audit = _fit_ranker(train, fields, target)
            active = held.loc[held["timestamp_top30"].astype(bool)].copy()
            raw = model.predict(_numeric_matrix(active, fields, medians))
            rank = _within_timestamp_rank(active, raw)
            prediction = baseline.copy()
            prediction.loc[active.index, "focused_raw_score"] = raw
            prediction.loc[active.index, "focused_rank"] = rank
            # ``final_score`` remains the same causal upstream score.  A small
            # local correction gives the focused model ranking authority but
            # leaves every position-size cap untouched.
            for alpha in alphas:
                scored = prediction.copy()
                scored["corrected_score"] = pd.to_numeric(scored["final_score"], errors="coerce")
                active_mask = scored["timestamp_top30"].astype(bool).to_numpy()
                scored.loc[active_mask, "corrected_score"] += float(alpha) * (
                    scored.loc[active_mask, "focused_rank"].to_numpy(float) - 0.5
                )
                scored["arm"] = f"{target.name}_scorecorr_a{alpha:g}"
                scored["fold"] = fold
                all_predictions.append(scored)
            fit_audit.update({
                "fold": fold, "cutoff": str(cutoff), "held_end_exclusive": str(held_end),
                "train_rows_before_cap": len(train_all), "held_rows": len(held),
                "held_timestamp_top30_fraction": float(held["timestamp_top30"].mean()),
                "geometry_bundle_sha256": str(identities[0]),
                "feature_mode": str(args.feature_mode), "field_count": len(fields), "fields": list(fields),
            })
            valid_active = active.loc[
                active["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(active["policy_net_bps"], errors="coerce"))
            ]
            if len(valid_active) > 10 and np.unique(rank).size > 1:
                rank_by_identity = pd.Series(rank, index=active.index)
                fit_audit["held_top30_rank_ic"] = float(spearmanr(
                    rank_by_identity.loc[valid_active.index],
                    pd.to_numeric(valid_active["policy_net_bps"], errors="coerce"),
                ).statistic)
            else:
                fit_audit["held_top30_rank_ic"] = np.nan
            audits.append(fit_audit)
        control = baseline.copy()
        control["corrected_score"] = pd.to_numeric(control["final_score"], errors="coerce")
        control["arm"] = "control_final_score"
        control["fold"] = fold
        all_predictions.append(control)
        print(json.dumps({"event": "focused_top30_fold_complete", "fold": fold, "cutoff": str(cutoff), "train": len(train), "held": len(held)}), flush=True)
        del frame, train_all, train, held
        gc.collect()
    output = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat([
        _tail_metrics(block, arm=str(arm), kind=kind)
        for arm, block in output.groupby("arm", sort=True)
        for kind in ("global", "month", "week")
    ], ignore_index=True)
    output.to_parquet(args.out_dir / "oof_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(args.out_dir / "metrics.parquet", index=False)
    pd.DataFrame(audits).to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_timestamp_top30_reliability_score_correction_v1",
        "surface": str(args.surface), "history_surface": str(args.history_surface) if args.history_surface else None,
        "source_sha256": {str(path): _sha(path) for path in paths},
        "feature_contract": str(args.feature_contract), "feature_contract_sha256": _sha(args.feature_contract),
        "feature_mode": str(args.feature_mode), "candidate_fields": list(candidate_fields), "candidate_field_count": len(candidate_fields),
        "compact_feature_contract": str(args.feature_contract),
        "training": "three prior months; valid policy labels resolved before held cutoff; top 30 percent selected within each decision timestamp; equal-month cap",
        "targets": {target.name: target.description for target in targets},
        "alphas": list(alphas),
        "score_integration": "final_score + alpha * (within-current-timestamp focused LambdaRank percentile - 0.5), only for timestamp top-30 candidates",
        "admission": "not replayed here: this is a global-tail ranking diagnostic.  A later executable test must fit the causal 21-day EV map on the corrected score using only prior resolved labels.",
        "sizing": "unchanged; no multiplier or leverage cap modification",
        "causality": "timestamp selection and percentile use only contemporaneous candidate scores; no held outcomes are used for prediction, normalisation, or selection",
        "geometry": "one frozen Geometry/K9 bundle required per fold; the nine k09__cluster_XX__membership posteriors are allowed only because this runner rejects mixed bundle identities; raw distances/confidences remain excluded",
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(output)}))


if __name__ == "__main__":
    main()
