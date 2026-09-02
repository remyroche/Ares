#!/usr/bin/env python3
"""Select short R3 base fields with the reusable Stage-I selection process.

The deployed long base has a frozen 120-field causal contract. The retained
Stage-I selection code documents its process: side-local causal coverage and
variance gates, an univariate/gain pre-screen, target-free Spearman redundancy
control, chronological economic permutation MDA, then the smallest prefix
within one standard error of the best inner-fold result. The original long
selector artifact was retired with old research artifacts, so this script
reproduces that process on the repaired short source rather than inventing a
different selector.

The script reads only January--March 2024. April--June is reserved for the
subsequent target comparison and is never read here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_ordinal_base_target_ablation import (  # noqa: E402
    FROZEN_BASE_PARAMS,
    OOS_START,
    TRAIN_START,
    _coverage_fields,
    _feature_fields,
    _load_candidates,
    _load_features,
    _matrix,
    _r3_target,
    _side_paths,
    _spearman,
    _utc,
    _valid_label,
)


SEED = 17
INNER_FOLDS: tuple[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp], ...] = (
    ("2024-02", pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-02-01T00:00:00Z"), pd.Timestamp("2024-03-01T00:00:00Z")),
    ("2024-03", pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-03-01T00:00:00Z"), pd.Timestamp("2024-04-01T00:00:00Z")),
)
PRE_SCREEN_MAX = 72
MDA_MAX = 48
MDA_REPEATS = 3
MDA_TRAIN_CAP = 20_000
# Stage-I permits bounded vectorised MDA.  Eight thousand time-spread rows
# retain 80/160/400 rows in the top 1/2/5% objectives while avoiding native
# allocation growth under 3x48 repeated prediction calls.
MDA_EVAL_CAP = 8_000
PREFIXES = (15, 20, 30, 40, 60)
# Match the established bounded long-selector ReliefF rescue geometry.  It is
# deliberately a recall-oriented pre-screen: final retention still requires
# chronological economic MDA evidence.
RELIEF_ANCHOR_MAX_ROWS = 768
RELIEF_NEIGHBOR_CANDIDATES = 2_048
RELIEF_NEIGHBORS = 8
RELIEF_RESCUE_MAX = 48
RELIEF_RESCUE_MIN = 20
RELIEF_RESCUE_FRAC = 0.25


@dataclass(frozen=True)
class FoldAudit:
    name: str
    train_start: str
    validation_start: str
    validation_end: str
    train_rows: int
    validation_rows: int
    target_free_train_candidate_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    return frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_q1_labels(root: Path) -> pd.DataFrame:
    """Read exactly the selection window; never materialise held OOS labels."""
    parts = []
    for month in pd.date_range(TRAIN_START, OOS_START, freq="MS", inclusive="left"):
        path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = _utc(frame[column])
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("Q1 short label identities are invalid")
    return frame


def _tail_utility(score: np.ndarray, net_bps: np.ndarray) -> float:
    """Global top-1/2/5%-weighted net utility, robustly clipped per tail."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    values = []
    for fraction in (0.01, 0.02, 0.05):
        take = max(1, int(math.ceil(len(order) * fraction)))
        values.append(float(np.clip(np.nanmean(net_bps[order[:take]]), -300.0, 300.0)))
    return float(np.dot(values, (0.45, 0.35, 0.20)))


def _time_sample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.reset_index(drop=True)
    ordered = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    index = np.unique(np.linspace(0, len(ordered) - 1, cap, dtype=np.int64))
    return ordered.iloc[index].reset_index(drop=True)


def _relief_scores(matrix: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Bounded deterministic ReliefF score, matching the long pre-screen.

    The calculation has bounded 768 x 2,048 neighbourhood work regardless of
    raw population size.  It operates only on a time-spread training sample
    and is a rescue mechanism; neither its scores nor its target enter the
    inference feature contract.
    """
    values = matrix.to_numpy(dtype=np.float32, copy=True)
    median = np.nanmedian(values, axis=0).astype(np.float32, copy=False)
    median[~np.isfinite(median)] = 0.0
    values = np.where(np.isfinite(values), values, median[None, :]).astype(np.float32, copy=False)
    q25 = np.nanpercentile(values, 25.0, axis=0).astype(np.float32, copy=False)
    q75 = np.nanpercentile(values, 75.0, axis=0).astype(np.float32, copy=False)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0).astype(np.float32, copy=False)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32, copy=False)
    values = np.clip((values - median[None, :]) / scale[None, :], -8.0, 8.0).astype(np.float32, copy=False)
    n_rows, n_fields = values.shape
    binary = (np.asarray(labels, dtype=np.int8) >= 1).astype(np.int8)
    if n_rows < 4 or n_fields == 0 or len(np.unique(binary)) < 2:
        return np.zeros(n_fields, dtype=np.float32)
    rng = np.random.default_rng(SEED + 2_017)
    anchors = rng.choice(n_rows, size=min(n_rows, RELIEF_ANCHOR_MAX_ROWS), replace=False)
    candidate_count = min(n_rows, max(RELIEF_NEIGHBOR_CANDIDATES, len(anchors)))
    candidates = rng.choice(n_rows, size=candidate_count, replace=False) if candidate_count < n_rows else np.arange(n_rows, dtype=np.int32)
    candidates = np.unique(np.concatenate([candidates.astype(np.int32), anchors.astype(np.int32)])).astype(np.int32)
    candidate_values = values[candidates]
    candidate_labels = binary[candidates]
    candidate_norm = np.einsum("ij,ij->i", candidate_values, candidate_values).astype(np.float32, copy=False)
    scores = np.zeros(n_fields, dtype=np.float64)
    used = 0
    for anchor in anchors:
        point = values[int(anchor)]
        distance = candidate_norm + float(np.dot(point, point)) - 2.0 * np.dot(candidate_values, point)
        distance = np.maximum(distance, 0.0)
        distance[candidates == int(anchor)] = np.inf
        same = candidate_labels == binary[int(anchor)]
        hit = np.flatnonzero(same & np.isfinite(distance))
        miss = np.flatnonzero(~same & np.isfinite(distance))
        if not len(hit) or not len(miss):
            continue
        hit = hit[np.argsort(distance[hit])[:min(RELIEF_NEIGHBORS, len(hit))]]
        miss = miss[np.argsort(distance[miss])[:min(RELIEF_NEIGHBORS, len(miss))]]
        scores += (
            np.mean(np.abs(candidate_values[miss] - point[None, :]), axis=0)
            - np.mean(np.abs(candidate_values[hit] - point[None, :]), axis=0)
        )
        used += 1
    return (scores / float(used)).astype(np.float32, copy=False) if used else np.zeros(n_fields, dtype=np.float32)


def _representatives(train: pd.DataFrame, fields: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Stage-I pre-screen and target-free correlation representative policy."""
    y = _r3_target(train).astype(int)
    soft = pd.to_numeric(train["robust_clear_soft_b25_t50"], errors="coerce")
    numeric = _numeric(train, fields)
    records = []
    for feature in fields:
        value = numeric[feature]
        records.append({
            "feature": feature,
            "class_spearman_abs": abs(_spearman(value, y)),
            "soft_spearman_abs": abs(_spearman(value, soft)),
            "coverage": float(value.notna().mean()),
            "variance": float(value.var(skipna=True)),
        })
    audit = pd.DataFrame(records)
    for column in ("class_spearman_abs", "soft_spearman_abs"):
        rank = f"rank_{column}"
        audit[rank] = audit[column].rank(method="dense", ascending=False)
    audit["univariate_rank"] = audit[["rank_class_spearman_abs", "rank_soft_spearman_abs"]].mean(axis=1)
    audit["univariate_selected"] = audit.univariate_rank.rank(method="first").le(PRE_SCREEN_MAX)
    # The original Stage-I MDA contract caps each selector model at 20k rows.
    # It retains time-spread rather than random sampling and avoids native
    # histogram allocation from competing with the live producer.
    fit_rows = _time_sample(train, MDA_TRAIN_CAP)
    fit_y = _r3_target(fit_rows).astype(int)
    relief = _relief_scores(_numeric(fit_rows, fields), fit_y.to_numpy())
    audit["relief_score"] = relief.astype(float)
    rescue_count = min(RELIEF_RESCUE_MAX, max(RELIEF_RESCUE_MIN, int(RELIEF_RESCUE_FRAC * int(audit.univariate_selected.sum()))))
    audit["relief_rank"] = audit.relief_score.rank(method="dense", ascending=False)
    audit["relief_selected"] = audit.relief_score.gt(1e-6) & audit.relief_rank.le(rescue_count)
    audit["relief_rescued"] = audit.relief_selected & ~audit.univariate_selected
    model = lgb.LGBMClassifier(**FROZEN_BASE_PARAMS).fit(
        _matrix(fit_rows, fields, numeric.median()), fit_y.to_numpy()
    )
    audit["gain"] = audit.feature.map(dict(zip(fields, model.booster_.feature_importance(importance_type="gain"), strict=True))).astype(float)
    for column in ("gain",):
        rank = f"rank_{column}"
        audit[rank] = audit[column].rank(method="dense", ascending=False)
    audit["screen_selected"] = audit.univariate_selected | audit.relief_selected
    # ReliefF is a rescue: a strong non-univariate feature receives the same
    # candidate priority as an equivalently ranked univariate one.  Gain only
    # breaks ties, never substitutes for either mandatory pre-screen.
    audit["prescreen_rank"] = np.minimum(audit.univariate_rank, audit.relief_rank)
    screen = audit.loc[audit.screen_selected].sort_values(["prescreen_rank", "rank_gain", "feature"], kind="stable")
    # Exact Stage-I representative policy: coverage, then variance, then field
    # order. It deliberately does not use labels/economics at this stage.
    representative_order = screen.sort_values(["coverage", "variance", "feature"], ascending=[False, False, True], kind="stable")
    corr = numeric.loc[:, representative_order.feature.tolist()].corr(method="spearman").abs()
    kept: list[str] = []
    representatives: dict[str, str] = {}
    for feature in representative_order.feature.astype(str):
        match = next((current for current in kept if float(corr.loc[feature, current]) >= 0.95), None)
        representatives[feature] = match or feature
        if match is None:
            kept.append(feature)
    audit["correlation_representative"] = audit.feature.map(representatives).fillna(audit.feature)
    audit["post_correlation_kept"] = audit.feature.isin(kept)
    selected = [
        str(feature) for feature in screen.sort_values(["prescreen_rank", "rank_gain", "feature"], kind="stable").feature
        if str(feature) in set(kept)
    ][:MDA_MAX]
    audit["mda_candidate"] = audit.feature.isin(selected)
    return selected, audit


def _fit(train: pd.DataFrame, validation: pd.DataFrame, fields: list[str]) -> tuple[lgb.LGBMClassifier, pd.Series, np.ndarray, np.ndarray]:
    fit_rows = _time_sample(train, MDA_TRAIN_CAP)
    medians = _numeric(fit_rows, fields).median()
    model = lgb.LGBMClassifier(**FROZEN_BASE_PARAMS).fit(
        _matrix(fit_rows, fields, medians), _r3_target(fit_rows).astype(int).to_numpy()
    )
    probabilities = np.asarray(model.predict_proba(_matrix(validation, fields, medians)), dtype=np.float32)
    score = probabilities[:, 2] - 0.5 * probabilities[:, 0]
    net = pd.to_numeric(validation["t4_tp6_sl4_net_bps"], errors="coerce").to_numpy(np.float64)
    return model, medians, score, net


def _mda(model: lgb.LGBMClassifier, medians: pd.Series, validation: pd.DataFrame, fields: list[str], fold: str, full_utility: float) -> list[dict[str, Any]]:
    sampled = _time_sample(validation, MDA_EVAL_CAP)
    # A single reusable float32 matrix is materially less memory-intensive
    # than allocating one pandas block manager for every feature x repeat.
    # Restore the column after each prediction so the MDA permutations remain
    # independent and preserve every other feature exactly.
    matrix = _matrix(sampled, fields, medians).to_numpy(dtype=np.float32, copy=True)
    net = pd.to_numeric(sampled["t4_tp6_sl4_net_bps"], errors="coerce").to_numpy(np.float64)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float32)
    baseline = _tail_utility(probabilities[:, 2] - 0.5 * probabilities[:, 0], net)
    rows: list[dict[str, Any]] = []
    for offset, feature in enumerate(fields):
        original = matrix[:, offset].copy()
        drops = []
        for repeat in range(MDA_REPEATS):
            rng = np.random.default_rng(SEED + 1_000 * offset + 17 * repeat + (0 if fold == "2024-02" else 1))
            matrix[:, offset] = original[rng.permutation(len(original))]
            p = np.asarray(model.predict_proba(matrix), dtype=np.float32)
            drops.append(baseline - _tail_utility(p[:, 2] - 0.5 * p[:, 0], net))
        matrix[:, offset] = original
        rows.append({
            "fold": fold, "feature": feature, "full_validation_utility_bps": full_utility,
            "mda_validation_rows": len(sampled), "mda_baseline_utility_bps": baseline,
            "mda_drop_mean_bps": float(np.mean(drops)),
            "mda_drop_std_bps": float(np.std(drops, ddof=1)), "mda_repeats": MDA_REPEATS,
        })
        if (offset + 1) % 8 == 0:
            gc.collect()
    del matrix, probabilities
    gc.collect()
    return rows


def _prefix_metrics(train: pd.DataFrame, validation: pd.DataFrame, ranked: list[str], fold: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_sizes: set[int] = set()
    for requested in PREFIXES:
        fields = ranked[:min(requested, len(ranked))]
        if len(fields) < 4 or len(fields) in seen_sizes:
            continue
        seen_sizes.add(len(fields))
        _model, _medians, score, net = _fit(train, validation, fields)
        records.append({"fold": fold, "prefix_size": len(fields), "utility_bps": _tail_utility(score, net)})
    return records


def _smallest_within_one_se(metrics: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    summary = metrics.pivot(index="prefix_size", columns="fold", values="utility_bps").sort_index()
    summary = summary.assign(mean_utility_bps=summary.mean(axis=1), std_utility_bps=summary.std(axis=1, ddof=1), fold_count=summary.notna().sum(axis=1)).reset_index()
    summary["se_utility_bps"] = summary.std_utility_bps / np.sqrt(summary.fold_count.clip(lower=1))
    best = summary.loc[summary.mean_utility_bps.idxmax()]
    viable = summary.loc[summary.mean_utility_bps.ge(float(best.mean_utility_bps - best.se_utility_bps))]
    return int(viable.prefix_size.min()), summary


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True)
    defaults = _side_paths("short")
    fields = _feature_fields("short")
    candidates = _load_candidates(defaults["candidates"], "short")
    candidates = candidates.loc[candidates.__ts__.ge(TRAIN_START) & candidates.__ts__.lt(OOS_START)].copy()
    features = _load_features(defaults["features"], fields, candidates, "short")
    labels = _load_q1_labels(defaults["labels"])
    ledger = features.merge(labels, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    del features, labels, candidates
    gc.collect()
    ledger["entry_executable"] = ledger.entry_executable.astype(bool)
    target_free_population = ledger.loc[ledger.entry_executable]
    covered, coverage = _coverage_fields(target_free_population, fields)
    variable = [field for field in covered if int(_numeric(target_free_population, [field])[field].nunique(dropna=True)) >= 2]
    if len(variable) < 20:
        raise ValueError(f"only {len(variable)} target-free finite/varying short features")
    selection_rows = ledger.loc[ledger.entry_executable & _valid_label(ledger) & ledger.__label_available_at__.lt(OOS_START)]
    mda_fields, prescreen = _representatives(selection_rows, variable)
    if len(mda_fields) < 15:
        raise ValueError(f"pre-MDA field pool too small: {len(mda_fields)}")
    fold_audits: list[dict[str, Any]] = []
    mda_rows: list[dict[str, Any]] = []
    for name, train_start, validation_start, validation_end in INNER_FOLDS:
        train = ledger.loc[ledger.entry_executable & ledger.__ts__.ge(train_start) & ledger.__ts__.lt(validation_start) & _valid_label(ledger) & ledger.__label_available_at__.lt(validation_start)]
        validation = ledger.loc[ledger.entry_executable & ledger.__ts__.ge(validation_start) & ledger.__ts__.lt(validation_end) & _valid_label(ledger)]
        if train.empty or validation.empty:
            raise ValueError(f"{name}: empty purged chronological selection fold")
        print(f"MDA fold {name}: fitting and permuting {len(mda_fields)} fields", flush=True)
        model, medians, score, net = _fit(train, validation, mda_fields)
        utility = _tail_utility(score, net)
        mda_rows.extend(_mda(model, medians, validation, mda_fields, name, utility))
        fold_audits.append(asdict(FoldAudit(name, str(train_start), str(validation_start), str(validation_end), len(train), len(validation), int(ledger.loc[ledger.entry_executable & ledger.__ts__.ge(train_start) & ledger.__ts__.lt(validation_start)].shape[0]))))
        del model, medians, score, net
        gc.collect()
    mda = pd.DataFrame(mda_rows)
    summary = mda.groupby("feature", as_index=False).agg(mean_mda_drop_bps=("mda_drop_mean_bps", "mean"), min_mda_drop_bps=("mda_drop_mean_bps", "min"), positive_fold_count=("mda_drop_mean_bps", lambda x: int((x > 0).sum())), mean_mda_std_bps=("mda_drop_std_bps", "mean"))
    summary = summary.merge(prescreen, on="feature", how="left", validate="one_to_one").sort_values(["positive_fold_count", "mean_mda_drop_bps", "min_mda_drop_bps", "prescreen_rank", "feature"], ascending=[False, False, False, True, True], kind="stable").reset_index(drop=True)
    ranked = summary.loc[summary.positive_fold_count.eq(len(INNER_FOLDS)), "feature"].astype(str).tolist()
    if len(ranked) < 15:
        ranked = summary.feature.astype(str).tolist()[:max(15, min(30, len(summary)))]
    prefix_rows = []
    for name, train_start, validation_start, validation_end in INNER_FOLDS:
        train = ledger.loc[ledger.entry_executable & ledger.__ts__.ge(train_start) & ledger.__ts__.lt(validation_start) & _valid_label(ledger) & ledger.__label_available_at__.lt(validation_start)]
        validation = ledger.loc[ledger.entry_executable & ledger.__ts__.ge(validation_start) & ledger.__ts__.lt(validation_end) & _valid_label(ledger)]
        prefix_rows.extend(_prefix_metrics(train, validation, ranked, name))
        del train, validation
        gc.collect()
    prefix = pd.DataFrame(prefix_rows)
    selected_count, prefix_summary = _smallest_within_one_se(prefix)
    selected = ranked[:selected_count]
    contract = {
        "schema": "strict_r3_short_stagei_style_feature_selection_v1", "side": "short",
        "selected_features": selected, "selected_feature_count": len(selected),
        "selection": "causal coverage/variance -> univariate plus bounded ReliefF rescue (gain tie-break) -> target-free Spearman(0.95) representatives -> two chronological R3 economic MDA folds -> smallest prefix within one SE",
        "training_window": "2024-01-01 through 2024-03-31", "inner_folds": fold_audits,
    }
    (out / "selected_features.json").write_text(json.dumps(contract, indent=2) + "\n")
    pd.DataFrame({"feature": fields, "target_free_executable_coverage": [float(coverage[x]) for x in fields], "kept_after_variance": [x in variable for x in fields]}).to_parquet(out / "coverage_variance_audit.parquet", index=False)
    prescreen.to_parquet(out / "prescreen_audit.parquet", index=False)
    mda.to_parquet(out / "chronological_economic_mda.parquet", index=False)
    summary.assign(selected=summary.feature.isin(selected)).to_parquet(out / "mda_summary.parquet", index=False)
    prefix.to_parquet(out / "prefix_fold_metrics.parquet", index=False)
    prefix_summary.assign(selected=prefix_summary.prefix_size.eq(selected_count)).to_parquet(out / "prefix_summary.parquet", index=False)
    manifest = {
        **contract, "status": "complete",
        "candidate_pool": {"frozen_short_contract": len(fields), "coverage_and_variance_pass": len(variable), "pre_mda": len(mda_fields), "mda_ranked": len(ranked)},
        "economic_mda_metric": "global top1/top2/top5 H12 net bps, weights 0.45/0.35/0.20, each tail mean clipped to +/-300 bps",
        "mda": {"repeats": MDA_REPEATS, "train_cap": MDA_TRAIN_CAP, "evaluation_cap": MDA_EVAL_CAP, "correlation_threshold": 0.95, "prefixes": list(PREFIXES)},
        "label_contract": "R3 adverse/weak/robust-clear, exact H12 TP6/SL4 net, cost 100 bps once",
        "selection_boundary": "No April-June candidate, feature, label, or outcome is loaded.",
        "input_hashes": {"features": _sha256(defaults["features"]), "candidates": _sha256(defaults["candidates"]), "labels_manifest": _sha256(defaults["labels"] / "run_manifest.json"), "feature_contract": _sha256(ROOT / "config/strict_r3_canonical_v2_feature_contract.json")},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/strict_r3_short_stagei_style_feature_selection_2024q1_20260820_v1")
    args = parser.parse_args()
    print(run(args.out.resolve()))


if __name__ == "__main__":
    main()
