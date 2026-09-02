#!/usr/bin/env python3
"""Strict-OOF threshold-specific Router heads and fixed-capacity blend grid.

The experiment keeps the current long Router's 30-field feature contract,
three-month chronological training window, 28-day resolved-label reserve,
query geometry, and ranker parameters.  It changes only the main *training
label*:

* R50:  policy net outcome > +50 bps;
* R100: policy net outcome > +100 bps; and
* R200: policy net outcome > +200 bps.

Each head scores the complete held candidate universe before any held outcome
is joined.  Scores are CDF ranks against their own strict-training reference.
The blender then averages only those target-free ranks and routes the exact
top 50% at each timestamp.  Realised policy outcomes are joined afterwards
solely to report Recall@50/100/200 and economics.

This is development research.  It must not alter the live/canonical Router.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "strict_r3_router_threshold_recall_heads_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
TARGET_NAMES = {
    "R50_policy_net_gt50": "r50",
    "R100_policy_net_gt100": "r100",
    "R200_policy_net_gt200": "r200",
}


def _load_router_module():
    path = ROOT / "scripts" / "run_strict_r3_economic_recall_router.py"
    spec = importlib.util.spec_from_file_location("_strict_r3_router_threshold_parent", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


router = _load_router_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise AssertionError(f"{path}: expected {SCHEMA}")
    if payload.get("side") != "long":
        raise AssertionError("this ablation is long-only")
    if float(payload.get("route_fraction", 0.0)) != .5:
        raise AssertionError("threshold Router heads must retain the exact top-50% route")
    targets = tuple(payload.get("targets", ()))
    if targets != tuple(TARGET_NAMES):
        raise AssertionError(f"unexpected threshold targets {targets}")
    return payload


def _absolute(config_path: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else config_path.parents[1] / candidate


def _months(config: dict[str, object]) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in config["months"])
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise AssertionError("months must be unique and chronological")
    return result


def _parent_args(
    *, config_path: Path, config: dict[str, object], target: str, out: Path, resume: bool,
) -> Namespace:
    ranker = dict(config["ranker"])
    source = _absolute(config_path, str(config["source_current_router"]))
    return Namespace(
        feature_root=[_absolute(config_path, str(value)) for value in config["feature_roots"]],
        aux_root=ROOT / "data_perp/artifacts/strict_r3_o3v2_recall_router_aux_labels_20240201_20260731_20260825_v3",
        policy_path=_absolute(config_path, str(config["policy_path"])),
        bundle=_absolute(config_path, str(config["bundle"])),
        sealed_feature_contract=None,
        feature_list=None,
        # The current Router's immutable run contract holds its ordered 30
        # fields.  The parent refuses runtime/name-based feature discovery.
        full_feature_contract=source / "run_contract.json",
        out=out,
        months=_months(config),
        primary_target=target,
        aux_groups=(),
        train_months=int(config["train_months"]),
        reserve_days=int(config["reserve_days"]),
        train_cap=int(config["train_cap"]),
        route_fractions=(float(config["route_fraction"]),),
        n_jobs=int(ranker["n_jobs"]),
        reuse_aux_source=None,
        primary_only=True,
        n_estimators=int(ranker["n_estimators"]),
        learning_rate=float(ranker["learning_rate"]),
        max_depth=int(ranker["max_depth"]),
        num_leaves=int(ranker["num_leaves"]),
        min_child_fraction=float(ranker["min_child_fraction"]),
        min_child_floor=int(ranker["min_child_floor"]),
        min_split_gain=float(ranker["min_split_gain"]),
        subsample=float(ranker["subsample"]),
        feature_fraction=float(ranker["feature_fraction"]),
        l1=float(ranker["l1"]),
        l2=float(ranker["l2"]),
        max_bin=int(ranker["max_bin"]),
        truncation=int(ranker["truncation"]),
        label_gains=",".join(str(value) for value in ranker["label_gains"]),
        objective=str(ranker["objective"]),
        row_weight_scheme=str(ranker["row_weight_scheme"]),
        row_weight_floor_bps=float(ranker["row_weight_floor_bps"]),
        row_weight_cap_bps=float(ranker["row_weight_cap_bps"]),
        early_stopping_rounds=int(ranker["early_stopping_rounds"]),
        inner_validation_fraction=float(ranker["inner_validation_fraction"]),
        max_jobs=None,
        resume=resume,
        defer_aggregate=True,
    )


def _score_path(root: Path, month: pd.Timestamp) -> Path:
    return root / "target_free_scores" / f"month={month:%Y-%m}.parquet"


def _assert_target_free(frame: pd.DataFrame, *, source: Path) -> None:
    missing = set(IDENTITY) - set(frame.columns)
    if missing:
        raise AssertionError(f"{source}: missing identity fields {sorted(missing)}")
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("policy_", "label", "outcome", "gross_bps", "net_bps", "path_valid"))
    ]
    if forbidden:
        raise AssertionError(f"{source}: held score frame contains outcome fields {forbidden}")
    if frame["candidate_id"].duplicated().any() or frame[list(IDENTITY)].isna().any().any():
        raise AssertionError(f"{source}: invalid target-free candidate identities")


def _blend_specs(step: float) -> list[tuple[str, tuple[float, float, float]]]:
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0) or units < 2:
        raise ValueError("blend grid step must divide one and yield at least two units")
    specs: list[tuple[str, tuple[float, float, float]]] = []
    for r50 in range(units + 1):
        for r100 in range(units - r50 + 1):
            r200 = units - r50 - r100
            weights = (r50 / units, r100 / units, r200 / units)
            if max(weights) == 1.0:
                continue  # individual-head arms are named separately below.
            label = "blend_" + "_".join(
                f"{name}{int(round(weight * 100)):02d}"
                for name, weight in zip(("r50", "r100", "r200"), weights)
            )
            specs.append((label, weights))
    return specs


def _timestamp_select(frame: pd.DataFrame, field: str, fraction: float) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(
        ["__decision_ts__", "__score__", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(np.int64)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(np.int64)
    selected = (ordinal < np.ceil(fraction * size)) & np.isfinite(work["__score__"].to_numpy(float))
    out = pd.Series(selected, index=work["__row__"].to_numpy()).reindex(np.arange(len(frame))).to_numpy(bool)
    expected = frame.groupby("__decision_ts__", sort=False).size().map(lambda count: int(np.ceil(fraction * count)))
    actual = pd.Series(out, index=frame.index).groupby(frame["__decision_ts__"], sort=False).sum()
    if not actual.reindex(expected.index).eq(expected).all():
        raise AssertionError(f"{field}: timestamp-local top-{fraction:.0%} capacity mismatch")
    return out


def _fold_for_month(config: dict[str, object], month: str) -> str:
    for name, members in dict(config["folds"]).items():
        if month in members:
            return str(name)
    raise AssertionError(f"{month}: no configured fold")


def _timestamp_metrics(joined: pd.DataFrame, field: str, fraction: float) -> pd.DataFrame:
    selected = _timestamp_select(joined, field, fraction)
    work = joined.loc[:, ["candidate_id", "__decision_ts__", "month", "fold", "__valid__", "__net__"]].copy()
    work["selected"] = selected
    valid = work["__valid__"].to_numpy(bool)
    net = work["__net__"].to_numpy(float)
    work["selected_valid"] = selected & valid
    work["selected_net"] = np.where(work["selected_valid"], net, 0.0)
    for hurdle, suffix in ((0.0, "positive"), (50.0, "50"), (100.0, "100"), (200.0, "200")):
        winner = valid & (net > hurdle)
        work[f"winner_{suffix}"] = winner
        work[f"selected_winner_{suffix}"] = selected & winner
    grouped = work.groupby("__decision_ts__", sort=False).agg(
        month=("month", "first"), fold=("fold", "first"),
        candidate_rows=("candidate_id", "size"), selected_rows=("selected", "sum"),
        selected_valid_rows=("selected_valid", "sum"), selected_net_bps=("selected_net", "sum"),
        winners_positive=("winner_positive", "sum"), selected_winners_positive=("selected_winner_positive", "sum"),
        winners_50=("winner_50", "sum"), selected_winners_50=("selected_winner_50", "sum"),
        winners_100=("winner_100", "sum"), selected_winners_100=("selected_winner_100", "sum"),
        winners_200=("winner_200", "sum"), selected_winners_200=("selected_winner_200", "sum"),
    ).reset_index()
    for suffix in ("positive", "50", "100", "200"):
        denominator = grouped[f"winners_{suffix}"].to_numpy(float)
        grouped[f"recall_{suffix}"] = np.divide(
            grouped[f"selected_winners_{suffix}"].to_numpy(float), denominator,
            out=np.full(len(grouped), np.nan), where=denominator > 0,
        )
    denom = grouped["selected_valid_rows"].to_numpy(float)
    grouped["net_ev_bps_per_trade"] = np.divide(
        grouped["selected_net_bps"].to_numpy(float), denom,
        out=np.full(len(grouped), np.nan), where=denom > 0,
    )
    grouped["score"] = field
    return grouped


def _summaries(timestamp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: Iterable[tuple[str, pd.Series]] = (
        ("global", pd.Series("global", index=timestamp.index)),
        ("fold", timestamp["fold"].astype(str)),
        ("month", timestamp["month"].astype(str)),
    )
    for score, score_work in timestamp.groupby("score", sort=False):
        for scope, key in scopes:
            for period, work in score_work.groupby(key, sort=True):
                row: dict[str, object] = {
                    "score": score, "scope": scope, "period": str(period),
                    "timestamps": int(len(work)), "candidate_rows": int(work["candidate_rows"].sum()),
                    "selected_rows": int(work["selected_rows"].sum()),
                    "selected_valid_rows": int(work["selected_valid_rows"].sum()),
                    "net_sum_bps": float(work["selected_net_bps"].sum()),
                }
                for suffix in ("positive", "50", "100", "200"):
                    row[f"recall_at_{suffix}"] = float(work[f"recall_{suffix}"].mean())
                    row[f"recall_at_{suffix}_timestamps"] = int(work[f"recall_{suffix}"].notna().sum())
                row["net_ev_bps_per_trade_timestamp_macro"] = float(work["net_ev_bps_per_trade"].mean())
                count = row["selected_valid_rows"]
                row["net_ev_bps_per_trade_trade_weighted"] = row["net_sum_bps"] / count if count else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _combine_target_free(
    *, config_path: Path, config: dict[str, object], out: Path, arm_roots: dict[str, Path],
) -> list[Path]:
    source = _absolute(config_path, str(config["source_current_router"]))
    source_col = str(config["source_current_router_score_column"])
    paths: list[Path] = []
    for month in _months(config):
        current_path = _score_path(source, month)
        current = pd.read_parquet(current_path, columns=[*IDENTITY, source_col])
        _assert_target_free(current, source=current_path)
        current = current.rename(columns={source_col: "current_router_rank"})
        current["__decision_ts__"] = pd.to_datetime(current["__decision_ts__"], utc=True, errors="raise")
        merged = current
        for target, label in TARGET_NAMES.items():
            path = _score_path(arm_roots[label], month)
            frame = pd.read_parquet(path, columns=[*IDENTITY, "router_primary_rank"])
            _assert_target_free(frame, source=path)
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
            frame = frame.rename(columns={"router_primary_rank": f"{label}_rank"})
            merged = merged.merge(frame, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(merged) != len(current):
            raise AssertionError(f"{month:%Y-%m}: threshold-head identities differ from current Router universe")
        for label, weights in _blend_specs(float(config["blend_weight_grid_step"])):
            merged[label] = sum(
                weight * merged[f"{name}_rank"].to_numpy(float)
                for name, weight in zip(("r50", "r100", "r200"), weights)
            ).astype(np.float32)
        merged["router_r50"] = merged["r50_rank"].astype(np.float32)
        merged["router_r100"] = merged["r100_rank"].astype(np.float32)
        merged["router_r200"] = merged["r200_rank"].astype(np.float32)
        merged["current_router_control"] = merged["current_router_rank"].astype(np.float32)
        forbidden = [column for column in merged if any(token in column.lower() for token in ("policy_", "label", "outcome", "net_bps", "gross_bps"))]
        if forbidden:
            raise AssertionError(f"{month:%Y-%m}: combined held scores contain outcomes {forbidden}")
        target = out / "target_free_scores" / f"month={month:%Y-%m}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old = pd.read_parquet(target, columns=list(IDENTITY))
            if not old.equals(merged.loc[:, list(IDENTITY)]):
                raise AssertionError(f"{target}: immutable target-free identities differ")
        else:
            merged.to_parquet(target, index=False, compression="zstd")
        paths.append(target)
    return paths


def _evaluate(*, config_path: Path, config: dict[str, object], out: Path) -> None:
    policy_path = _absolute(config_path, str(config["policy_path"]))
    policy = pd.read_parquet(policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identities")
    outputs: list[pd.DataFrame] = []
    for month in _months(config):
        score = pd.read_parquet(out / "target_free_scores" / f"month={month:%Y-%m}.parquet")
        score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
        score["month"] = month.strftime("%Y-%m")
        score["fold"] = _fold_for_month(config, month.strftime("%Y-%m"))
        joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        joined["__valid__"] = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
        joined["__net__"] = pd.to_numeric(joined["policy_net_bps"], errors="coerce").fillna(0.0)
        score_fields = [
            "current_router_control", "router_r50", "router_r100", "router_r200",
            *[label for label, _ in _blend_specs(float(config["blend_weight_grid_step"]))],
        ]
        outputs.extend(_timestamp_metrics(joined, field, float(config["route_fraction"])) for field in score_fields)
    timestamp = pd.concat(outputs, ignore_index=True)
    timestamp.to_parquet(out / "outcome_joined_timestamp_metrics.parquet", index=False, compression="zstd")
    summary = _summaries(timestamp)
    summary.to_parquet(out / "outcome_joined_summary.parquet", index=False, compression="zstd")


def _receipt(config_path: Path, config: dict[str, object], out: Path, arm_roots: dict[str, Path]) -> dict[str, object]:
    source = _absolute(config_path, str(config["source_current_router"]))
    fields = json.loads((source / "run_contract.json").read_text())["feature_contract"]
    return {
        "schema": SCHEMA,
        "status": "complete",
        "long_only": True,
        "route_fraction": config["route_fraction"],
        "strict_prequential_training": True,
        "held_outcomes_used_for_training_or_target_free_scoring": False,
        "target_free_score_before_outcome_join": True,
        "timestamp_local_capacity_exact": True,
        "current_router_feature_contract": fields,
        "current_router_feature_contract_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "source_current_router": str(source),
        "source_current_router_contract_sha256": _sha256(source / "run_contract.json"),
        "threshold_arm_contracts": {
            name: {"path": str(path), "contract_sha256": _sha256(path / "run_contract.json")}
            for name, path in arm_roots.items()
        },
        "config_sha256": _sha256(config_path),
        "blends": [{"name": name, "weights": weights} for name, weights in _blend_specs(float(config["blend_weight_grid_step"]))],
        "outputs": ["target_free_scores", "outcome_joined_timestamp_metrics.parquet", "outcome_joined_summary.parquet"],
        "promotion": "development-only; no current Router, canonical stack, or live scoring mutation",
    }


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _read_config(config_path)
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    contract_path = args.out / "run_contract.json"
    initial_contract = {
        "schema": SCHEMA, "config": str(config_path), "config_sha256": _sha256(config_path),
        "scope": config["scope"], "status": "running",
    }
    if not contract_path.exists():
        _write_json_exclusive(contract_path, initial_contract)
    elif json.loads(contract_path.read_text()) != initial_contract:
        raise AssertionError("refusing resume with a different immutable threshold-Router contract")
    arm_roots: dict[str, Path] = {}
    for target, label in TARGET_NAMES.items():
        root = args.out / "threshold_arms" / label
        arm_roots[label] = root
        parent_args = _parent_args(
            config_path=config_path, config=config, target=target, out=root, resume=bool(args.resume),
        )
        router.run(parent_args)
        missing = [month for month in _months(config) if not _score_path(root, month).exists()]
        if missing:
            raise AssertionError(f"{label}: incomplete target-free scoring {missing}")
    _combine_target_free(config_path=config_path, config=config, out=args.out, arm_roots=arm_roots)
    _evaluate(config_path=config_path, config=config, out=args.out)
    receipt = _receipt(config_path, config, args.out, arm_roots)
    receipt_path = args.out / "correctness_report.json"
    if not receipt_path.exists():
        _write_json_exclusive(receipt_path, receipt)
    manifest_path = args.out / "run_manifest.json"
    if not manifest_path.exists():
        _write_json_exclusive(manifest_path, {**receipt, "status": "complete"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
