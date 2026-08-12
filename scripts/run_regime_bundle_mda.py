#!/usr/bin/env python3
"""Causal, conditional group-permutation MDA for the regime direct model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_execution_plan import (
    LABEL_DELAY_HOURS,
    LEVERAGE,
    LOOKBACK_DAYS,
    PHASE,
    PRIMARY_PERSISTENT,
    _fit_ridge,
    _metrics,
    _rank_ic,
    _read_generator_bindings,
    _top_mask,
    direct_arms,
    load_generator_panel,
)


SCHEMA = "regime_bundle_mda_v1"
TOPS = (0.01, 0.05, 0.10)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _permuted(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Jointly permute one bundle without destroying within-bundle geometry."""

    out = frame.copy()
    order = np.argsort(frame["candidate_id"].astype(str).to_numpy(), kind="stable")
    shift = max(1, len(order) // 2)
    donor = np.roll(order, shift)
    out.loc[:, list(columns)] = frame.iloc[donor].loc[:, list(columns)].to_numpy()
    return out


def _scores(panel: pd.DataFrame, features: Sequence[str], groups: dict[str, list[str]]) -> pd.DataFrame:
    out = panel.loc[:, ["candidate_id", "__ts__", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"]].copy()
    names = ["full", *[f"permute_{name}" for name in groups]]
    for name in names:
        out[f"score__{name}"] = np.nan
    months = pd.date_range(panel["__ts__"].min().floor("D").replace(day=1), panel["__ts__"].max().floor("D").replace(day=1) + pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
    for start, end in zip(months[:-1], months[1:]):
        evaluate = panel["__ts__"].ge(start) & panel["__ts__"].lt(end)
        train = panel["__ts__"].lt(start - pd.Timedelta(hours=LABEL_DELAY_HOURS)) & panel["__ts__"].ge(start - pd.Timedelta(days=LOOKBACK_DAYS))
        if int(train.sum()) < 500:
            prediction = panel.loc[evaluate, "score_residual_expected_ev"].to_numpy(float)
            out.loc[evaluate, "score__full"] = prediction
            for name in groups:
                out.loc[evaluate, f"score__permute_{name}"] = prediction
            continue
        # ``_fit_ridge`` is used with a copied evaluation frame; fitting is
        # train-only and permutations happen after the fit, only in held-out
        # candidate context.
        _, prediction = _fit_ridge(panel.loc[train], panel.loc[evaluate], features)
        out.loc[evaluate, "score__full"] = prediction
        for name, columns in groups.items():
            _, permuted = _fit_ridge(panel.loc[train], _permuted(panel.loc[evaluate], columns), features)
            out.loc[evaluate, f"score__permute_{name}"] = permuted
    if out.filter(regex=r"^score__").isna().any().any():
        raise ValueError("MDA left an unscored row")
    return out


def _result_rows(generator: str, scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full = scores["score__full"]
    for name in [column.removeprefix("score__") for column in scores if column.startswith("score__")]:
        score = scores[f"score__{name}"]
        for fraction in TOPS:
            selected = _top_mask(scores, score, fraction)
            rows.append({"generator": generator, "arm": name, "top_fraction": fraction, "net_rank_ic": _rank_ic(score, scores["execution_net_ev_12h"]), **_metrics(scores, selected)})
    results = pd.DataFrame(rows)
    baseline = results.loc[results.arm.eq("full"), ["top_fraction", "net_bps", "net_rank_ic"]].rename(columns={"net_bps": "full_net_bps", "net_rank_ic": "full_net_rank_ic"})
    return results.merge(baseline, on="top_fraction", how="left", validate="many_to_one").assign(
        mda_net_loss_bps=lambda frame: frame["full_net_bps"] - frame["net_bps"],
        mda_ic_loss=lambda frame: frame["full_net_rank_ic"] - frame["net_rank_ic"],
    )


def run(*, scores: Path, geometry: Path, source_panel: Path, generators: dict[str, Path], output_dir: Path) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    parts: list[pd.DataFrame] = []
    contracts: dict[str, Any] = {}
    for name, path in generators.items():
        panel, source, _coverage = load_generator_panel(scores_path=scores, generator_path=path, geometry_path=geometry, panel_path=source_panel)
        arms = direct_arms(panel, source)
        features = arms["U0_primary_leverage_transition"]
        groups = {
            "primary": [column for column in PRIMARY_PERSISTENT if column in features],
            "leverage": [column for column in LEVERAGE if column in features],
            "transition": [column for column in PHASE if column in features],
        }
        parts.append(_result_rows(name, _scores(panel, features, groups)))
        contracts[name] = {"features": features, "groups": groups}
    result = pd.concat(parts, ignore_index=True)
    output.mkdir(parents=True)
    result.to_csv(output / "conditional_bundle_mda.csv", index=False)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETED_CAUSAL_CONDITIONAL_GROUP_MDA",
        "contract": {"model": "monthly trailing-180d Ridge fit only on resolved prior labels", "permutation": "joint deterministic candidate-id permutation of one held-out bundle at a time", "selection": "pooled global top-k", "feature_contracts": contracts},
        "inputs": {"scores": {"path": str(scores.resolve()), "sha256": _sha(scores)}, "geometry": {"path": str(geometry.resolve()), "sha256": _sha(geometry)}, "generators": {name: {"path": str(path.resolve()), "sha256": _sha(path)} for name, path in generators.items()}},
        "outputs": {"conditional_bundle_mda.csv": _sha(output / "conditional_bundle_mda.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--geometry-states", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--generator", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    values = _args()
    print(run(scores=values.scores, geometry=values.geometry_states, source_panel=values.source_panel, generators=_read_generator_bindings(values.generator), output_dir=values.output_dir))
