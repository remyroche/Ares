#!/usr/bin/env python3
"""Causal per-score-band mixing of sealed strict-OOF Router scores.

This does not fit a new model.  It uses the existing target-free current
Router, R50/R100/R200, and fixed-blend score receipts.  Before each test
month, it chooses a three-band recipe policy only from earlier fully resolved
OOF months; it then routes the test month's full candidate universe at the
same timestamp-local 50% capacity as the base Router.

Score recipes differ only by incumbent-score band, not by outcome-derived
state at inference.  All realised policy outcomes are joined after the
target-free candidate scores have been read and validated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_router_scoreband_mixing_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")


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


def _absolute(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parents[1] / path


def _read_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema") != SCHEMA or config.get("side") != "long":
        raise AssertionError("wrong score-band mixing configuration")
    if float(config.get("route_fraction", 0.0)) != .5:
        raise AssertionError("score-band mixing must preserve the Router's 50% route")
    bands = np.asarray(config["score_bands"], dtype=float)
    if len(bands) != 4 or not np.all(np.diff(bands) > 0) or bands[0] != 0.0 or bands[-1] < 1.0:
        raise AssertionError("expected three fixed score bands spanning [0, 1]")
    recipes = tuple(config["recipe_fields"])
    if len(recipes) != 5 or len(set(recipes)) != len(recipes):
        raise AssertionError("exactly five distinct frozen score recipes are required")
    return config


def _months(config: dict[str, object]) -> tuple[str, ...]:
    result = tuple(str(value) for value in config["months"])
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise AssertionError("months must be chronological and unique")
    return result


def _assert_target_free(frame: pd.DataFrame, source: Path, recipes: tuple[str, ...], band_source: str) -> None:
    needed = {*IDENTITY, *recipes, band_source}
    missing = needed - set(frame.columns)
    if missing:
        raise AssertionError(f"{source}: missing {sorted(missing)}")
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("policy_", "label", "outcome", "gross_bps", "net_bps", "path_valid"))
    ]
    if forbidden or frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{source}: not a valid target-free score receipt")


def _load_target_free(config_path: Path, config: dict[str, object]) -> pd.DataFrame:
    source = _absolute(config_path, str(config["input_scores"]))
    recipes = tuple(str(value) for value in config["recipe_fields"])
    band_source = str(config["band_source"])
    pieces: list[pd.DataFrame] = []
    for month in _months(config):
        path = source / "target_free_scores" / f"month={month}.parquet"
        frame = pd.read_parquet(path, columns=[*IDENTITY, *recipes])
        _assert_target_free(frame, path, recipes, band_source)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["month"] = month
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def _attach_outcomes(config_path: Path, config: dict[str, object], scores: pd.DataFrame) -> pd.DataFrame:
    policy = pd.read_parquet(
        _absolute(config_path, str(config["policy_path"])),
        columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"],
    )
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate IDs")
    result = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    result["__valid__"] = (
        result["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(result["policy_net_bps"], errors="coerce"))
    )
    result["__net__"] = pd.to_numeric(result["policy_net_bps"], errors="coerce").fillna(0.0)
    return result


def _score_band(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    work = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    # right=False gives [0,.5), [.5,.75), [.75,1.000001); clamp protects
    # numerical endpoints while keeping the three meanings fixed.
    return np.clip(np.searchsorted(boundaries[1:], work, side="right"), 0, len(boundaries) - 2).astype(np.int8)


def _timestamp_codes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps, ts_code = np.unique(frame["__decision_ts__"].to_numpy(dtype="datetime64[ns]"), return_inverse=True)
    candidates = frame["candidate_id"].astype(str).to_numpy()
    # ``np.unique`` returns lexically sorted candidates, so codes are a stable
    # deterministic candidate-ID tie breaker inside timestamp score ties.
    _, candidate_code = np.unique(candidates, return_inverse=True)
    size = np.bincount(ts_code, minlength=len(timestamps))
    return ts_code.astype(np.int32), candidate_code.astype(np.int32), size.astype(np.int32)


def _metrics_from_score(
    *, score: np.ndarray, ts_code: np.ndarray, candidate_code: np.ndarray, size: np.ndarray,
    valid: np.ndarray, net: np.ndarray,
) -> dict[str, float]:
    finite = np.isfinite(score)
    order = np.lexsort((candidate_code, -np.nan_to_num(score, nan=-np.inf), ts_code))
    ordered_ts = ts_code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_ts)) + 1]
    ordinal = np.arange(len(order), dtype=np.int32) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    quota = np.ceil(.5 * size[ordered_ts]).astype(np.int32)
    chosen_order = (ordinal < quota) & finite[order]
    selected = np.zeros(len(score), dtype=bool)
    selected[order] = chosen_order
    n_ts = len(size)
    selected_valid = selected & valid
    selected_count = np.bincount(ts_code, weights=selected_valid.astype(np.int8), minlength=n_ts)
    selected_net = np.bincount(ts_code, weights=np.where(selected_valid, net, 0.0), minlength=n_ts)
    out: dict[str, float] = {
        "selected_rows": float(selected.sum()),
        "selected_valid_rows": float(selected_valid.sum()),
        "net_sum_bps": float(selected_net.sum()),
        "net_ev_bps_per_trade": float(selected_net.sum() / selected_valid.sum()) if selected_valid.any() else np.nan,
        "net_ev_bps_timestamp_macro": float(np.nanmean(np.divide(selected_net, selected_count, out=np.full(n_ts, np.nan), where=selected_count > 0))),
    }
    for hurdle, label in ((0.0, "positive"), (50.0, "50"), (100.0, "100"), (200.0, "200")):
        winner = valid & (net > hurdle)
        numerator = np.bincount(ts_code, weights=(selected & winner).astype(np.int8), minlength=n_ts)
        denominator = np.bincount(ts_code, weights=winner.astype(np.int8), minlength=n_ts)
        recall = np.divide(numerator, denominator, out=np.full(n_ts, np.nan), where=denominator > 0)
        out[f"recall_at_{label}"] = float(np.nanmean(recall))
        out[f"recall_at_{label}_timestamps"] = float(np.isfinite(recall).sum())
    return out


def _policy_score(frame: pd.DataFrame, recipes: tuple[str, ...], band: np.ndarray, policy: tuple[int, int, int]) -> np.ndarray:
    values = np.column_stack([pd.to_numeric(frame[field], errors="coerce").to_numpy(float) for field in recipes])
    selection = np.asarray(policy, dtype=np.int32)[band]
    return values[np.arange(len(frame)), selection]


def _select_policy(calibration: pd.DataFrame, recipes: tuple[str, ...], boundaries: np.ndarray) -> tuple[tuple[int, int, int], pd.DataFrame]:
    ts_code, candidate_code, size = _timestamp_codes(calibration)
    valid = calibration["__valid__"].to_numpy(bool)
    net = calibration["__net__"].to_numpy(float)
    band = _score_band(calibration["current_router_control"].to_numpy(float), boundaries)
    rows: list[dict[str, object]] = []
    for policy in itertools.product(range(len(recipes)), repeat=3):
        metrics = _metrics_from_score(
            score=_policy_score(calibration, recipes, band, policy), ts_code=ts_code,
            candidate_code=candidate_code, size=size, valid=valid, net=net,
        )
        rows.append({
            "policy": policy,
            "low_recipe": recipes[policy[0]], "mid_recipe": recipes[policy[1]], "high_recipe": recipes[policy[2]],
            **metrics,
        })
    table = pd.DataFrame(rows)
    # Exact predeclared lexicographic selection: the user asked to maximise
    # all-positive opportunity recall, while the bps recall tiers and realised
    # economics protect against an arbitrary broad-recall-only solution.
    table = table.sort_values(
        ["recall_at_positive", "recall_at_50", "recall_at_100", "net_ev_bps_per_trade", "low_recipe", "mid_recipe", "high_recipe"],
        ascending=[False, False, False, False, True, True, True], kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "calibration_rank", np.arange(1, len(table) + 1, dtype=np.int16))
    return tuple(table.iloc[0]["policy"]), table


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _read_config(config_path)
    if args.out.exists():
        raise FileExistsError(args.out)
    recipes = tuple(str(value) for value in config["recipe_fields"])
    boundaries = np.asarray(config["score_bands"], dtype=float)
    months = _months(config)
    first_test = str(config["first_test_month"])
    if first_test not in months or months.index(first_test) < 3:
        raise AssertionError("need at least three prior OOF months before the first test")
    target_free = _load_target_free(config_path, config)
    # Persist the input identity receipt before reading any outcome quantity.
    args.out.mkdir(parents=True)
    _write_json_exclusive(args.out / "run_contract.json", {
        "schema": SCHEMA, "config": str(config_path), "config_sha256": _sha256(config_path),
        "input_score_manifest_sha256": _sha256(_absolute(config_path, str(config["input_scores"])) / "run_manifest.json"),
        "target_free_rows": int(len(target_free)), "months": list(months), "recipes": list(recipes),
        "score_bands": boundaries.tolist(), "route_fraction": config["route_fraction"],
        "status": "running",
    })
    identity = target_free.loc[:, [*IDENTITY, "month"]].copy()
    identity.to_parquet(args.out / "target_free_input_identity.parquet", index=False, compression="zstd")

    joined = _attach_outcomes(config_path, config, target_free)
    test_months = months[months.index(first_test):]
    all_monthly: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    calibration_tables: list[pd.DataFrame] = []
    for test_month in test_months:
        test_start = pd.Timestamp(f"{test_month}-01", tz="UTC")
        calibration = joined.loc[
            joined["month"].lt(test_month)
            & pd.to_datetime(joined["policy_label_available_ts"], utc=True, errors="coerce").lt(test_start)
        ].copy()
        held = joined.loc[joined["month"].eq(test_month)].copy()
        policy, table = _select_policy(calibration, recipes, boundaries)
        table["test_month"] = test_month
        calibration_tables.append(table)
        band = _score_band(held[str(config["band_source"])].to_numpy(float), boundaries)
        ts_code, candidate_code, size = _timestamp_codes(held)
        valid = held["__valid__"].to_numpy(bool)
        net = held["__net__"].to_numpy(float)
        scores = {
            "current_router_control": held["current_router_control"].to_numpy(float),
            "router_r50": held["router_r50"].to_numpy(float),
            "causal_scoreband_mix": _policy_score(held, recipes, band, policy),
        }
        for name, score in scores.items():
            metrics = _metrics_from_score(score=score, ts_code=ts_code, candidate_code=candidate_code, size=size, valid=valid, net=net)
            all_monthly.append(pd.DataFrame([{"month": test_month, "score": name, **metrics}]))
        selections.append({
            "test_month": test_month, "calibration_months": sorted(calibration["month"].unique().tolist()),
            "low_band": f"[{boundaries[0]:.2f}, {boundaries[1]:.2f})", "mid_band": f"[{boundaries[1]:.2f}, {boundaries[2]:.2f})", "high_band": f"[{boundaries[2]:.2f}, {boundaries[3]:.2f})",
            "low_recipe": recipes[policy[0]], "mid_recipe": recipes[policy[1]], "high_recipe": recipes[policy[2]],
        })
    # One exact outcome-joined summary per test month and score.  Recall is
    # computed as a timestamp macro inside _metrics_from_score, never by
    # pooling candidate rows across timestamps.
    monthly = pd.concat(all_monthly, ignore_index=True)
    monthly.to_parquet(args.out / "strict_oof_monthly_metrics.parquet", index=False, compression="zstd")
    summary_rows: list[dict[str, object]] = []
    for score, work in monthly.groupby("score", sort=False):
        row: dict[str, object] = {"score": score, "test_months": int(len(work)), "selected_rows": int(work["selected_rows"].sum()), "selected_valid_rows": int(work["selected_valid_rows"].sum()), "net_sum_bps": float(work["net_sum_bps"].sum())}
        for field in ("recall_at_positive", "recall_at_50", "recall_at_100", "recall_at_200", "net_ev_bps_per_trade", "net_ev_bps_timestamp_macro"):
            row[field] = float(work[field].mean())
            row[f"worst_{field}"] = float(work[field].min())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(args.out / "strict_oof_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(selections).to_parquet(args.out / "prequential_band_policies.parquet", index=False, compression="zstd")
    pd.concat(calibration_tables, ignore_index=True).to_parquet(args.out / "prior_only_grid_metrics.parquet", index=False, compression="zstd")
    _write_json_exclusive(args.out / "correctness_report.json", {
        "schema": SCHEMA, "long_only": True, "target_free_scores_read_before_outcome_join": True,
        "new_model_fit": False, "recipes_are_existing_strict_oof_score_receipts": True,
        "band_boundaries_frozen": boundaries.tolist(), "band_source": config["band_source"],
        "per_test_selection_uses_prior_months_only": True, "timestamp_local_top50_exact": True,
        "test_months": list(test_months), "promotion": "research-only; no live/canonical mutation",
    })
    _write_json_exclusive(args.out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "outputs": ["target_free_input_identity.parquet", "prequential_band_policies.parquet", "prior_only_grid_metrics.parquet", "strict_oof_monthly_metrics.parquet", "strict_oof_summary.parquet"],
        "config_sha256": _sha256(config_path), "scope": config["scope"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
