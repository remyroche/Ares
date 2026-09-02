#!/usr/bin/env python3
"""Strict-OOF screen of the expanded H12 path-label families.

This is a deliberately narrow, offline stage of the O3-v2 meta research.  It
uses the canonical causal 120-field base contract plus prequential base/stack
outputs, routes candidates by the *timestamp-local top 30% base score*, and
tests three families of future labels one at a time:

* path order / pre-opportunity adversity;
* favourable and adverse magnitude;
* opportunity and policy-event timing.

Every held score receipt is written without outcomes or raw future labels.
Only after that immutable receipt exists are policy outcomes joined to produce
research metrics.  It does not change any live bundle, MC1 map, or policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for candidate in (ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_strict_r3_long_supportive_label_funnel as supportive  # noqa: E402


SCHEMA = "strict_r3_o3v2_path_auxiliary_funnel_v1"
SEED = 1729
BASE_ROUTE = 0.30
EMBARGO = pd.Timedelta(hours=12)
MAX_TRAIN_ROWS = 180_000
MIN_TRAIN_ROWS = 20_000
MIN_TARGET_ROWS = 5_000
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
DEFAULT_AUXILIARY = ROOT / "data_perp/artifacts/strict_r3_o3v2_path_auxiliary_labels_20240801_20260731_20260825_v1"
TAILS = (0.01, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class OuterFold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    cohort: str


@dataclass(frozen=True)
class TargetSpec:
    name: str
    family: str
    column: str
    direction: float
    lower_quantile: float | None = 0.005
    upper_quantile: float | None = 0.995


FOLDS: tuple[OuterFold, ...] = (
    OuterFold("dev_2025_q2", pd.Timestamp("2025-04-01T00:00:00Z"), pd.Timestamp("2025-07-01T00:00:00Z"), "development"),
    OuterFold("dev_2025_q3", pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2025-10-01T00:00:00Z"), "development"),
    OuterFold("holdout_2025_q4", pd.Timestamp("2025-10-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z"), "holdout"),
    OuterFold("oos_2026_q1", pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z"), "portability"),
    OuterFold("oos_2026_q2", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z"), "portability"),
    OuterFold("oos_2026_jul", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z"), "portability"),
)

# The targets use frozen path labels only.  Timings are H12-censored and all
# continuous quantities are clipped from train-only quantiles inside the fit.
CONTROL_TARGETS: tuple[TargetSpec, ...] = (
    # The prior direct path-efficiency winner is always fitted under the same
    # top-30 route and feature contract.  New path-label families are retained
    # only when they improve on this matched control, never in isolation.
    TargetSpec("control_path_efficiency", "control", "aux_path_efficiency", 1.0),
)

TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("path_mfe_before_mae", "path_order", "aux_path_mfe_before_mae", 1.0, None, None),
    TargetSpec("path_mae_before_100bps", "path_order", "aux_mae_before_100bps_atr", -1.0),
    TargetSpec("path_mae_before_200bps", "path_order", "aux_mae_before_200bps_atr", -1.0),
    TargetSpec("path_mae_before_250bps", "path_order", "aux_mae_before_250bps_atr", -1.0),
    TargetSpec("magnitude_reach_100bps", "magnitude", "aux_reached_100bps", 1.0, None, None),
    TargetSpec("magnitude_reach_200bps", "magnitude", "aux_reached_200bps", 1.0, None, None),
    TargetSpec("magnitude_reach_300bps", "magnitude", "aux_reached_300bps", 1.0, None, None),
    TargetSpec("magnitude_reach_500bps", "magnitude", "aux_reached_500bps", 1.0, None, None),
    *(TargetSpec(f"magnitude_mfe_{hour}h", "magnitude", f"aux_mfe_atr_{hour}h", 1.0) for hour in (1, 3, 6, 9, 12)),
    *(TargetSpec(f"magnitude_mae_{hour}h", "magnitude", f"aux_mae_atr_{hour}h", -1.0) for hour in (1, 3, 6, 9, 12)),
    *(TargetSpec(f"timing_to_{bps}bps", "timing", f"aux_time_to_{bps}bps_h", -1.0, None, None) for bps in (100, 200, 300)),
    *(TargetSpec(f"timing_to_{multiple}atr", "timing", f"aux_time_to_{multiple}atr_h", -1.0, None, None) for multiple in (1, 2, 3)),
    TargetSpec("timing_to_trailing_activation", "timing", "aux_time_to_trailing_activation_h", -1.0, None, None),
    TargetSpec("timing_to_stop_loss", "timing", "aux_time_to_stop_loss_h", 1.0, None, None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files: Iterable[Path] = sorted(path.rglob("*.parquet")) if path.is_dir() else (path,)
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _exact_timestamp_top_fraction(frame: pd.DataFrame, field: str, fraction: float) -> pd.Series:
    """Deterministic timestamp-local route with candidate-ID cutoff ties."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    working = frame.loc[:, ["__decision_ts__", "candidate_id", field]].copy()
    working["__value__"] = _finite(working[field]).fillna(-np.inf)
    working["__order__"] = np.arange(len(working), dtype=np.int64)
    working = working.sort_values(["__decision_ts__", "__value__", "candidate_id", "__order__"], ascending=[True, False, True, True], kind="stable")
    size = working.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    rank = working.groupby("__decision_ts__", sort=False).cumcount()
    selected = rank < np.ceil(size * fraction).astype(int)
    result = pd.Series(False, index=frame.index)
    result.iloc[working["__order__"].to_numpy()] = selected.to_numpy(bool)
    return result


def _fields(ledger: Path) -> tuple[str, ...]:
    names = pq.ParquetFile(ledger).schema.names
    fields = tuple(names[23:143])
    if len(fields) != 120 or len(set(fields)) != 120:
        raise AssertionError(f"expected frozen 120 causal base fields, got {len(fields)}")
    return fields


def _months(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    first = (start - pd.Timedelta(hours=1)).normalize().replace(day=1)
    last = end.normalize().replace(day=1)
    yield from pd.date_range(first, last, freq="MS", inclusive="left", tz="UTC")


def _score_leak_columns(columns: Sequence[str]) -> list[str]:
    """Raw labels are forbidden; a *predicted* policy-bps score is allowed."""
    forbidden_exact = {"policy_net_bps", "policy_path_valid", "aux_path_valid", "aux_label_available_ts"}
    return [
        column for column in columns
        if column.startswith("aux_") or column in forbidden_exact
    ]


def _read_auxiliary(root: Path, *, start: pd.Timestamp, end: pd.Timestamp, columns: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    required = list(dict.fromkeys([*IDENTITY, "aux_label_available_ts", "aux_path_valid", *columns]))
    for month in _months(start, end):
        path = root / "parts" / f"month={month:%Y-%m}" / "auxiliary_path_labels.parquet"
        if not path.exists():
            if month < start.normalize().replace(day=1):
                continue
            raise FileNotFoundError(path)
        pieces.append(pd.read_parquet(path, columns=required))
    if not pieces:
        return pd.DataFrame(columns=required)
    result = pd.concat(pieces, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "aux_label_available_ts"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    result = result.loc[result["__decision_ts__"].ge(start) & result["__decision_ts__"].lt(end)].copy()
    if result["candidate_id"].duplicated().any():
        raise AssertionError("auxiliary labels duplicate candidate IDs after decision-time filtering")
    return result


def _read_population(ledger: Path, *, start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    stack = supportive.STACK_FIELDS
    columns = [
        *IDENTITY,
        "r3_label_available_ts", "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "base_contract_complete", "base_feature_available_fraction", "prequential_base_score",
        *fields, *stack,
    ]
    result = pd.read_parquet(ledger, columns=list(dict.fromkeys(columns)), filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)]).copy()
    for column in ("__ts__", "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("ledger duplicate candidate IDs in path auxiliary window")
    if not result["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("path auxiliary funnel is long-only")
    return result


def _joined_population(ledger: Path, auxiliary: Path, *, start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str], target_columns: Sequence[str]) -> pd.DataFrame:
    population = _read_population(ledger, start=start, end=end, fields=fields)
    labels = _read_auxiliary(auxiliary, start=start, end=end, columns=target_columns)
    check = labels.loc[:, list(IDENTITY)].rename(columns={column: f"auxiliary_{column}" for column in IDENTITY if column != "candidate_id"})
    population = population.merge(check, on="candidate_id", how="left", validate="one_to_one")
    for column in IDENTITY[1:]:
        paired = f"auxiliary_{column}"
        if population[paired].isna().any() or not population[column].eq(population[paired]).all():
            raise AssertionError(f"candidate identity mismatch in auxiliary labels: {column}")
    population = population.drop(columns=[f"auxiliary_{column}" for column in IDENTITY[1:]])
    payload = labels.drop(columns=[column for column in IDENTITY if column != "candidate_id"])
    result = population.merge(payload, on="candidate_id", how="left", validate="one_to_one")
    if result["aux_path_valid"].isna().any():
        raise AssertionError("candidate-to-auxiliary label coverage failed")
    return result


def _eligible_train(frame: pd.DataFrame, *, cutoff: pd.Timestamp, target: str) -> pd.DataFrame:
    valid = (
        frame["aux_path_valid"].fillna(False).astype(bool)
        & frame["base_contract_complete"].fillna(False).astype(bool)
        & _finite(frame["base_feature_available_fraction"]).ge(0.90)
        & frame["aux_label_available_ts"].lt(cutoff - EMBARGO)
        & frame["r3_label_available_ts"].lt(cutoff - EMBARGO)
        & _finite(frame[target]).notna()
    )
    return frame.loc[valid].copy()


def _eligible_held(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["base_contract_complete"].fillna(False).astype(bool) & _finite(frame["base_feature_available_fraction"]).ge(0.90)
    output = frame.loc[valid].copy()
    output["meta_base_top30"] = _exact_timestamp_top_fraction(output, "prequential_base_score", BASE_ROUTE).to_numpy(bool)
    return output.loc[output["meta_base_top30"]].copy()


def _clip_train_target(train: pd.DataFrame, spec: TargetSpec) -> tuple[pd.DataFrame, dict[str, float | None]]:
    output = train.copy()
    values = _finite(output[spec.column])
    low = float(values.quantile(spec.lower_quantile)) if spec.lower_quantile is not None else None
    high = float(values.quantile(spec.upper_quantile)) if spec.upper_quantile is not None else None
    if low is not None or high is not None:
        values = values.clip(lower=low, upper=high)
    output["__target__"] = values.astype(np.float32)
    return output, {"clip_low": low, "clip_high": high}


def _direct_score(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], direction: float, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    ordered = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    oof = np.full(len(ordered), np.nan, dtype=float)
    policy = _finite(ordered["policy_net_bps"]).to_numpy(float)
    boundaries = np.linspace(0, len(ordered), 5, dtype=int)
    for index in range(3):
        fit_end, valid_end = int(boundaries[index + 1]), int(boundaries[index + 2])
        if fit_end < MIN_TRAIN_ROWS // 3 or valid_end <= fit_end:
            continue
        fit, valid = ordered.iloc[:fit_end], ordered.iloc[fit_end:valid_end]
        target = _finite(fit["__target__"]).to_numpy(float)
        usable = np.isfinite(target)
        if usable.sum() < 1_000:
            continue
        x_fit, medians = supportive._matrix(fit.loc[usable], fields)
        x_valid, _ = supportive._matrix(valid, fields, medians=medians)
        model = supportive._model_regressor(seed=seed + index)
        model.fit(x_fit, target[usable])
        oof[fit_end:valid_end] = direction * model.predict(x_valid)
    usable_oof = np.isfinite(oof) & np.isfinite(policy)
    if usable_oof.sum() < 2_000 or np.unique(oof[usable_oof]).size < 10:
        return np.full(len(held), np.nan), {"status": "insufficient_oof", "map_rows": int(usable_oof.sum())}
    mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
    mapper.fit(oof[usable_oof], policy[usable_oof])
    target = _finite(train["__target__"]).to_numpy(float)
    usable = np.isfinite(target)
    x_train, medians = supportive._matrix(train.loc[usable], fields)
    x_held, _ = supportive._matrix(held, fields, medians=medians)
    model = supportive._model_regressor(seed=seed + 100)
    model.fit(x_train, target[usable])
    return mapper.predict(direction * model.predict(x_held)).astype(np.float32), {"status": "ok", "map_rows": int(usable_oof.sum())}


def _metric_rows(scores: pd.DataFrame, policy: pd.DataFrame, *, fold: OuterFold, spec: TargetSpec) -> list[dict[str, object]]:
    joined = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    outcome = _finite(joined["policy_net_bps"]).to_numpy(float)
    score = _finite(joined["predicted_policy_net_bps"]).to_numpy(float)
    valid = np.isfinite(score) & np.isfinite(outcome)
    rows: list[dict[str, object]] = []
    ic = float(spearmanr(score[valid], outcome[valid]).statistic) if valid.sum() >= 12 else np.nan
    for tail in TAILS:
        count = int(np.ceil(tail * valid.sum()))
        order = np.argsort(score[valid], kind="stable")[-max(1, count):]
        values = outcome[valid][order]
        rows.append({
            "fold": fold.name, "cohort": fold.cohort, "target": spec.name, "family": spec.family,
            "tail": tail, "trades": int(len(values)), "net_ev_bps_per_trade": float(values.mean()),
            "net_sum_bps": float(values.sum()), "policy_rank_ic": ic,
        })
    return rows


def _contract(*, ledger: Path, auxiliary: Path, out: Path, families: Sequence[str], folds: Sequence[OuterFold]) -> dict[str, object]:
    return {
        "schema": SCHEMA, "ledger": str(ledger.resolve()), "auxiliary": str(auxiliary.resolve()),
        "out": str(out.resolve()), "families": list(families), "base_route": BASE_ROUTE,
        "folds": [{"name": fold.name, "start": str(fold.start), "end": str(fold.end), "cohort": fold.cohort} for fold in folds],
        "targets": [spec.__dict__ for spec in (*CONTROL_TARGETS, *(spec for spec in TARGETS if spec.family in families))],
    }


def _finalise(out: Path, policy: pd.DataFrame, *, selected: Sequence[TargetSpec], folds: Sequence[OuterFold]) -> None:
    metrics: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for spec in selected:
        for fold in folds:
            score_path = out / "target_free_scores" / spec.family / spec.name / f"fold={fold.name}.parquet"
            audit_path = out / "audit_parts" / f"{spec.name}__{fold.name}.json"
            if not score_path.exists() or not audit_path.exists():
                raise AssertionError(f"missing target/fold receipt: {spec.name} {fold.name}")
            scores = pd.read_parquet(score_path)
            prohibited = _score_leak_columns(scores.columns)
            if prohibited:
                raise AssertionError(f"target-free path score receipt leaked labels: {prohibited}")
            metrics.extend(_metric_rows(scores, policy, fold=fold, spec=spec))
            audits.append(json.loads(audit_path.read_text()))
    pd.DataFrame(metrics).to_parquet(out / "path_auxiliary_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "path_auxiliary_audit.parquet", index=False, compression="zstd")
    summary = pd.DataFrame(metrics).groupby(["target", "family", "cohort", "tail"], as_index=False).agg(
        mean_net_ev_bps_per_trade=("net_ev_bps_per_trade", "mean"),
        median_net_ev_bps_per_trade=("net_ev_bps_per_trade", "median"),
        worst_net_ev_bps_per_trade=("net_ev_bps_per_trade", "min"),
        mean_rank_ic=("policy_rank_ic", "mean"),
        folds=("fold", "nunique"),
    )
    summary.to_parquet(out / "path_auxiliary_summary.parquet", index=False, compression="zstd")


def run(*, ledger: Path, auxiliary: Path, out: Path, families: Sequence[str], folds: Sequence[OuterFold], resume: bool, max_jobs: int | None) -> Path:
    selected = (*CONTROL_TARGETS, *(spec for spec in TARGETS if spec.family in families))
    if not selected:
        raise ValueError("no selected target families")
    contract = _contract(ledger=ledger, auxiliary=auxiliary, out=out, families=families, folds=folds)
    contract_path = out / "run_contract.json"
    if out.exists():
        if not resume or not contract_path.exists() or json.loads(contract_path.read_text()) != contract:
            raise FileExistsError("output exists without matching resumable path-auxiliary contract")
    else:
        out.mkdir(parents=True, exist_ok=False)
        _write_json_exclusive(contract_path, contract)
    (out / "target_free_scores").mkdir(exist_ok=True)
    (out / "audit_parts").mkdir(exist_ok=True)
    source_fields = _fields(ledger)
    predictors = (*source_fields, *supportive.STACK_FIELDS)
    policy = pd.read_parquet(ledger, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger duplicate identity")
    history_start = pd.Timestamp("2024-01-01T00:00:00Z")
    completed = 0
    all_target_columns = tuple(sorted({spec.column for spec in selected}))
    for fold_index, fold in enumerate(folds):
        train_raw = _joined_population(ledger, auxiliary, start=history_start, end=fold.start, fields=source_fields, target_columns=all_target_columns)
        held_raw = _joined_population(ledger, auxiliary, start=fold.start, end=fold.end, fields=source_fields, target_columns=all_target_columns)
        held = _eligible_held(held_raw)
        if len(held) < 5_000:
            raise AssertionError(f"{fold.name}: insufficient timestamp-local base-routed held support: {len(held)}")
        for target_index, spec in enumerate(selected):
            score_root = out / "target_free_scores" / spec.family / spec.name
            score_root.mkdir(parents=True, exist_ok=True)
            score_path = score_root / f"fold={fold.name}.parquet"
            audit_path = out / "audit_parts" / f"{spec.name}__{fold.name}.json"
            if score_path.exists() and audit_path.exists():
                continue
            if score_path.exists() != audit_path.exists():
                raise AssertionError(f"partial immutable receipt: {spec.name} {fold.name}")
            if max_jobs is not None and completed >= max_jobs:
                break
            train = _eligible_train(train_raw, cutoff=fold.start, target=spec.column)
            if len(train) < MIN_TRAIN_ROWS:
                raise AssertionError(f"{spec.name} {fold.name}: insufficient strict train support {len(train)}")
            train = supportive._sample_month_balanced(train, MAX_TRAIN_ROWS, seed=SEED + fold_index * 100 + target_index)
            train, clipping = _clip_train_target(train, spec)
            if int(_finite(train["__target__"]).notna().sum()) < MIN_TARGET_ROWS:
                raise AssertionError(f"{spec.name} {fold.name}: insufficient target support after clipping")
            score, extra = _direct_score(train, held, predictors, spec.direction, seed=SEED + fold_index * 1000 + target_index * 31)
            receipt = pd.DataFrame({
                "candidate_id": held["candidate_id"].to_numpy(),
                "__decision_ts__": held["__decision_ts__"].to_numpy(),
                "side_name": held["side_name"].to_numpy(),
                "meta_base_top30": held["meta_base_top30"].to_numpy(bool),
                "predicted_policy_net_bps": score,
            })
            prohibited = _score_leak_columns(receipt.columns)
            if prohibited:
                raise AssertionError(f"attempted target leak into score receipt: {prohibited}")
            receipt.to_parquet(score_path, index=False, compression="zstd")
            _write_json_exclusive(audit_path, {
                "target": spec.name, "family": spec.family, "fold": fold.name, "cohort": fold.cohort,
                "train_rows": int(len(train)), "held_routed_rows": int(len(held)),
                "target_finite_rows": int(_finite(train["__target__"]).notna().sum()),
                "map_rows": int(extra["map_rows"]), "status": str(extra["status"]),
                "clip_low": clipping["clip_low"], "clip_high": clipping["clip_high"],
                "fit_cutoff": str(fold.start), "embargo_hours": 12,
                "route": "exact timestamp-local top 30% of prequential_base_score",
            })
            completed += 1
            print(json.dumps({"event": "scored", "target": spec.name, "fold": fold.name, **extra}, sort_keys=True), flush=True)
        if max_jobs is not None and completed >= max_jobs:
            break
    expected = [(spec, fold) for spec in selected for fold in folds]
    complete = all(
        (out / "target_free_scores" / spec.family / spec.name / f"fold={fold.name}.parquet").exists()
        and (out / "audit_parts" / f"{spec.name}__{fold.name}.json").exists()
        for spec, fold in expected
    )
    if complete:
        _finalise(out, policy, selected=selected, folds=folds)
        _write_json_exclusive(out / "run_manifest.json", {
            "schema": SCHEMA,
            "scope": "offline target-only path auxiliary-label research; no live, policy, MC1, or inference mutation",
            "ledger": str(ledger.resolve()), "ledger_sha256": _sha256(ledger),
            "auxiliary": str(auxiliary.resolve()), "auxiliary_manifest_sha256": _sha256(auxiliary / "run_manifest.json"),
            "targets": [spec.__dict__ for spec in selected], "base_route": BASE_ROUTE,
            "predictors": list(predictors),
            "causality": {
                "training": "auxiliary H12 labels and R3 labels strictly resolved before fold start minus 12h embargo",
                "held_scores": "persisted target-free before policy outcomes are joined for metrics",
                "route": "timestamp-local base top-30%; no future label participates in routing",
                "targets": "all auxiliary path fields are forbidden from the predictor contract and score receipts",
            },
        })
        print(json.dumps({"event": "finalised", "jobs": len(expected)}), flush=True)
    else:
        progress = out / f"progress_{sum((out / 'target_free_scores' / s.family / s.name / f'fold={f.name}.parquet').exists() for s, f in expected):03d}_of_{len(expected):03d}.json"
        if not progress.exists():
            _write_json_exclusive(progress, {"schema": SCHEMA, "completed_jobs": completed, "expected_jobs": len(expected)})
        print(json.dumps({"event": "checkpoint", "completed_jobs": completed, "expected_jobs": len(expected)}), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--auxiliary", type=Path, default=DEFAULT_AUXILIARY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default="path_order,magnitude,timing", help="comma-separated subset of path_order,magnitude,timing")
    parser.add_argument("--folds", help="comma-separated outer fold names; default all six")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    families = tuple(args.families.split(","))
    unknown = sorted(set(families) - {"path_order", "magnitude", "timing"})
    if unknown:
        parser.error(f"unsupported families: {unknown}")
    folds = tuple(fold for fold in FOLDS if args.folds is None or fold.name in set(args.folds.split(",")))
    if not folds:
        parser.error("no selected folds")
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    print(run(ledger=args.ledger.resolve(), auxiliary=args.auxiliary.resolve(), out=args.out.resolve(), families=families, folds=folds, resume=args.resume, max_jobs=args.max_jobs))


if __name__ == "__main__":
    main()
