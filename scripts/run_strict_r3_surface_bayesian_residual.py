#!/usr/bin/env python3
"""Strict Bayesian policy-residual challenger on one frozen-geometry surface.

This is the rich-context counterpart to the base-contract ledger challenger.
It accepts only a predeclared non-posterior feature contract, verifies that all
input surfaces share one frozen Geometry/K9 identity, and fits each monthly
prediction from prior resolved labels only.  The output is diagnostic until a
bounded blend improves both 2025 development and untouched 2026 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.trust_sizing_ablation import QuantileBins, residual_classes


SEED = 20260812
SCHEMA = "strict_r3_surface_bayesian_residual_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(path: Path, key: str) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    fields = tuple(map(str, payload[key]))
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("feature contract must provide a non-empty unique field list")
    if any(name.startswith("k09__cluster_") for name in fields):
        raise ValueError("raw K9 membership coordinates are forbidden")
    return fields


def _sample(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = frame["__decision_ts__"].dt.strftime("%Y-%m")
    rng, quota = np.random.default_rng(seed), int(math.ceil(cap / month.nunique()))
    pieces: list[np.ndarray] = []
    for token in sorted(month.unique()):
        pos = np.flatnonzero(month.eq(token).to_numpy())
        if len(pos) > quota:
            pos = np.sort(rng.choice(pos, quota, replace=False))
        pieces.append(pos)
    pos = np.concatenate(pieces)
    if len(pos) > cap:
        pos = np.sort(rng.choice(pos, cap, replace=False))
    return frame.iloc[pos].sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()


def _select(train: pd.DataFrame, fields: tuple[str, ...], maximum: int) -> tuple[tuple[str, ...], QuantileBins]:
    all_bins, code = QuantileBins.fit(train, fields), None
    code = all_bins.transform(train)
    target = residual_classes(train["policy_net_bps"], train["base_anchor_bps"])
    rank = pd.to_numeric(train["final_score"], errors="coerce").to_numpy(float)
    state = np.digitize(rank, np.unique(np.quantile(rank, [.2, .4, .6, .8])), right=True)
    scored: list[tuple[float, str]] = []
    for index, name in enumerate(fields):
        value = 0.0
        for group in range(5):
            mask = state == group
            if mask.sum() < 100:
                continue
            # 5 ordinary bins + missing, times five residual grades.
            joint = np.bincount(code[mask, index].astype(np.int64) * 5 + target[mask].astype(np.int64), minlength=30).reshape(6, 5).astype(float) + .5
            joint /= joint.sum()
            px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
            value += float(mask.mean()) * float(np.sum(joint * np.log(joint / (px * py))))
        scored.append((value, name))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    chosen = tuple(name for _, name in scored[:min(maximum, len(scored))])
    return chosen, QuantileBins.fit(train, chosen)


def _predict(train: pd.DataFrame, held: pd.DataFrame, fields: tuple[str, ...], strength: float) -> tuple[pd.DataFrame, dict[str, object]]:
    chosen, bins = _select(train, fields, 12)
    left, right = bins.transform(train), bins.transform(held)
    residual = (pd.to_numeric(train["policy_net_bps"], errors="coerce") - pd.to_numeric(train["base_anchor_bps"], errors="coerce")).to_numpy(float)
    global_mean = float(residual.mean())
    effects, support_rows = [], []
    for index in range(len(chosen)):
        train_code, held_code = left[:, index], right[:, index]
        count = int(max(train_code.max(initial=0), held_code.max(initial=0))) + 1
        support = np.bincount(train_code, minlength=count).astype(float)
        total = np.bincount(train_code, weights=residual, minlength=count)
        posterior = (total + strength * global_mean) / (support + strength)
        lookup = np.clip(held_code, 0, count - 1)
        effects.append(posterior[lookup])
        support_rows.append(support[lookup])
    adjustment = np.mean(np.vstack(effects), axis=0)
    expected = pd.to_numeric(held["base_anchor_bps"], errors="coerce").to_numpy(float) + adjustment
    prior_expected = pd.to_numeric(train["base_anchor_bps"], errors="coerce").to_numpy(float) + global_mean
    domain = np.sort(prior_expected[np.isfinite(prior_expected)], kind="stable")
    rank = np.searchsorted(domain, expected, side="right") / max(1, len(domain))
    return pd.DataFrame({"bayes_residual_bps": adjustment.astype(np.float32), "bayes_expected_bps": expected.astype(np.float32), "bayes_rank_traincdf": np.clip(rank, 0.0, 1.0).astype(np.float32), "bayes_effective_support": np.median(np.vstack(support_rows), axis=0).astype(np.float32)}), {"selected_fields": list(chosen), "global_residual_mean_bps": global_mean}


def _metrics(frame: pd.DataFrame, score: str, arm: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    valid = frame.loc[np.isfinite(pd.to_numeric(frame[score], errors="coerce")) & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))].copy()
    for period, block in [("all", valid), *[(str(year), value) for year, value in valid.groupby(valid["__decision_ts__"].dt.year, sort=True)]]:
        ic = float(spearmanr(block[score], block["policy_net_bps"]).statistic)
        for tail in (.005, .01, .02, .05):
            rows = block.nlargest(max(1, int(math.ceil(tail * len(block)))), score)
            output.append({"arm": arm, "period": period, "tail": tail, "rows": len(block), "selected": len(rows), "net_bps_per_trade": float(rows["policy_net_bps"].mean()), "rank_ic": ic})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=Path, nargs="+", required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--feature-key", default="mda_proposed_fields")
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--top-fraction", type=float, default=.30)
    parser.add_argument("--prior-strength", type=float, default=300.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    fields = _fields(args.feature_contract, args.feature_key)
    base = ["candidate_id", "__decision_ts__", "side_name", "geometry_bundle_sha256", "policy_path_valid", "policy_label_available_ts", "policy_net_bps", "base_anchor_bps", "final_score"]
    pieces = [pd.read_parquet(path, columns=[*base, *fields]) for path in args.surfaces]
    frame = pd.concat(pieces, ignore_index=True)
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq(args.side).all():
        raise ValueError("candidate identity or side contract failed")
    bundle_ids = frame["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(bundle_ids) != 1:
        raise ValueError("all rich-context surfaces must share exactly one frozen Geometry/K9 identity")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    parts, audits = [], []
    for index, start in enumerate(sorted(frame["__decision_ts__"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC").unique())):
        end = start + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
        train = frame.loc[
            frame["__decision_ts__"].ge(start - pd.DateOffset(months=args.train_months)) & frame["__decision_ts__"].lt(start)
            & frame["policy_label_available_ts"].lt(start) & frame["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce")) & np.isfinite(pd.to_numeric(frame["base_anchor_bps"], errors="coerce")) & np.isfinite(pd.to_numeric(frame["final_score"], errors="coerce")),
        ].copy()
        result = held.loc[:, ["candidate_id", "__decision_ts__", "side_name", "policy_label_available_ts", "policy_path_valid", "policy_net_bps", "base_anchor_bps", "final_score"]].copy()
        if len(train) < 1000:
            result["bayes_available"] = False
            result["bayes_rank_traincdf"] = result["final_score"]
            result[["bayes_residual_bps", "bayes_expected_bps", "bayes_effective_support"]] = np.nan
            audits.append({"held_month": str(start)[:7], "status": "warmup", "held_rows": len(held), "train_rows_before_top": len(train)})
        else:
            floor = float(np.quantile(train["final_score"], 1.0 - args.top_fraction, method="higher"))
            sampled = _sample(train.loc[train["final_score"].ge(floor)].copy(), args.train_cap, SEED + index)
            prediction, audit = _predict(sampled, held, fields, args.prior_strength)
            result = pd.concat([result.reset_index(drop=True), prediction], axis=1)
            result["bayes_available"] = True
            audits.append({"held_month": str(start)[:7], "status": "complete", "held_rows": len(held), "train_rows_before_top": len(train), "train_rows": len(sampled), "train_score_floor": floor, **audit})
        for alpha in (.025, .05, .10):
            result[f"bayes_blend_{alpha:.3f}"] = (1.0 - alpha) * result["final_score"] + alpha * result["bayes_rank_traincdf"]
        parts.append(result)
        print(json.dumps({"event": "month_complete", **audits[-1]}, default=str), flush=True)
    output = pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(output) != len(frame) or output["candidate_id"].duplicated().any():
        raise AssertionError("candidate population changed")
    metrics: list[dict[str, object]] = []
    eligible = output.loc[output["bayes_available"].astype(bool)].copy()
    for score in ["final_score", "bayes_blend_0.025", "bayes_blend_0.050", "bayes_blend_0.100"]:
        metrics.extend(_metrics(eligible, score, score))
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "surface_bayesian_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "surface_bayesian_fold_audit.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(args.out_dir / "surface_bayesian_metrics.parquet", index=False)
    manifest = {"schema": SCHEMA, "surfaces": [str(path) for path in args.surfaces], "surface_sha256": {str(path): _sha(path) for path in args.surfaces}, "feature_contract": str(args.feature_contract), "feature_contract_sha256": _sha(args.feature_contract), "feature_key": args.feature_key, "field_count": len(fields), "fields": list(fields), "side": args.side, "geometry_bundle_sha256": str(bundle_ids[0]), "target": "policy_net_bps - base_anchor_bps", "training": {"months": args.train_months, "cap": args.train_cap, "top_fraction": args.top_fraction, "prior_strength": args.prior_strength, "max_fields": 12}, "strict_prequential": True, "raw_k9_memberships_used": False, "integration": "diagnostic score blend only; no admission or sizing change", "rows": len(output)}
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": len(output)}))


if __name__ == "__main__":
    main()
