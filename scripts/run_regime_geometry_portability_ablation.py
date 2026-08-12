#!/usr/bin/env python3
"""Matched causal-regime feature lift and transport ablation.

This runner uses frozen OOF base/residual scores plus the new hourly regime
sidecar.  It does not change upstream models, tune on realised economics, or
select per timestamp.  Each arm is mapped with a train-only trailing Ridge
model and then ranked globally in common net-bps units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity  # noqa: E402


SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
STATES = ROOT / "data_perp/artifacts/oof_causal_market_regime_systems_2023q3_2024_20260803_v1/candidate_oof_market_regimes.parquet"
OUTPUT = ROOT / "data_perp/artifacts/regime_geometry_portability_ablation_20260803_v1"
SCHEMA = "regime_geometry_portability_ablation_v1"
TOPS = (0.01, 0.05, 0.10)
LOOKBACK_DAYS = 180
LABEL_DELAY_HOURS = 12
RIDGE_ALPHA = 30.0
ACTION_TOKENS = ("timing", "wait", "target_price", "action", "entry_price", "mae", "mfe")
GEOMETRIES = ("trend_volatility", "breadth_dependence", "leverage_flow", "liquidity")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("ablation context has non-finite values")


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(left.loc[mask].rank(method="average").corr(right.loc[mask].rank(method="average")))


def _top_mask(frame: pd.DataFrame, score: pd.Series, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(frame) * fraction)))
    order = pd.DataFrame({"score": score.to_numpy(float), "candidate_id": frame["candidate_id"].astype(str)}, index=frame.index)
    return frame.index.isin(order.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(count).index)


def _primary_columns(columns: Iterable[str]) -> list[str]:
    names = list(columns)
    chosen = [
        "market_regime__entropy", "market_regime__top2_margin",
        "market_regime__state_age_hours", "market_regime__state_switch_probability",
        "market_regime__ood_distance_percentile", "market_regime__phase_entropy",
        "market_regime__phase_top2_margin",
    ]
    chosen += [name for name in names if name.startswith("market_regime__phase_p_")]
    return [name for name in chosen if name in names]


def _semantic_primary_membership(columns: Iterable[str]) -> list[str]:
    """The train-only named five-state simplex, never raw fold coordinates."""
    names = list(columns)
    return [
        name for name in names
        if name.startswith("regime_p_s")
        or name in {
            "market_regime__direction_score",
            "market_regime__direction_positive_probability",
            "market_direction_sign",
        }
    ]


def _geometry_columns(
    columns: Iterable[str], geometry: str, *, include_memberships: bool = False,
) -> list[str]:
    prefix = f"geometry_regime__{geometry}__"
    allowed = {
        "entropy", "top2_margin", "state_age_hours",
        "state_switch_probability", "ood_distance_percentile",
    }
    if include_memberships:
        allowed |= {f"state_p_{state}" for state in range(6)}
    return [
        name for name in columns
        if name.startswith(prefix)
        and name.rsplit("__", 1)[-1] in allowed
    ]


def arm_features(columns: Iterable[str]) -> dict[str, list[str]]:
    base = ["score_residual_expected_ev", "side_is_long"]
    primary = _primary_columns(columns)
    arms = {"A0_baseline": base, "A1_primary": [*base, *primary]}
    for geometry in GEOMETRIES:
        arms[f"A2_{geometry}"] = [*base, *primary, *_geometry_columns(columns, geometry)]
    arms["A3_all_geometry"] = [*base, *primary, *[name for geometry in GEOMETRIES for name in _geometry_columns(columns, geometry)]]
    # The semantic primary simplex has a fixed train-only ontology.  Geometry
    # memberships have only a fold-local coordinate system: retain them as a
    # deliberately labelled diagnostic arm until the separate alignment gate
    # proves that they transport.  This makes the evidence visible without
    # silently promoting arbitrary GMM component IDs.
    semantic = _semantic_primary_membership(columns)
    if semantic:
        arms["B1_primary_semantic_membership"] = [*base, *primary, *semantic]
    for geometry in GEOMETRIES:
        membership = _geometry_columns(columns, geometry, include_memberships=True)
        arms[f"B2_{geometry}_fold_local_membership_diagnostic"] = [
            *base, *primary, *membership,
        ]
    arms["B3_all_geometry_fold_local_membership_diagnostic"] = [
        *base, *primary,
        *[
            name
            for geometry in GEOMETRIES
            for name in _geometry_columns(columns, geometry, include_memberships=True)
        ],
    ]
    for arm, features in arms.items():
        if not features or any(any(token in name.lower() for token in ACTION_TOKENS) for name in features):
            raise ValueError(f"{arm} feature contract is empty or contains action fields")
    return {arm: list(dict.fromkeys(features)) for arm, features in arms.items()}


def load_panel(scores_path: Path, states_path: Path) -> pd.DataFrame:
    wanted = [*IDENTITY_COLUMNS, "__reconstructed_soft_alpha_12h__", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "score_residual_expected_ev", "residual_is_oof"]
    scores = pd.read_parquet(scores_path, columns=wanted)
    states = pd.read_parquet(states_path)
    for frame in (scores, states):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    scores = validate_candidate_identity(scores)
    states = validate_candidate_identity(states)
    panel = scores.merge(states, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    if len(panel) != len(states):
        raise ValueError("regime sidecar is not an exact OOF score population")
    if not panel["residual_is_oof"].astype(bool).all():
        raise ValueError("upstream residual score must be OOF")
    for column in ("regime_available_utc", "transition_available_utc"):
        if (pd.to_datetime(panel[column], utc=True) > panel["__ts__"]).any():
            raise ValueError(f"{column} is after candidate decision")
    for column in ("regime_train_end_utc", "transition_train_end_utc"):
        if (pd.to_datetime(panel[column], utc=True) >= panel["__ts__"]).any():
            raise ValueError(f"{column} is not strictly prior to candidate decision")
    panel["side_is_long"] = panel["side_name"].astype(str).eq("long").astype(float)
    return panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def causal_monthly_map(panel: pd.DataFrame, arms: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    records: list[dict[str, Any]] = []
    months = pd.date_range(out["__ts__"].min().floor("D").replace(day=1), out["__ts__"].max().floor("D").replace(day=1) + pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
    for start, end in zip(months[:-1], months[1:]):
        evaluate = out["__ts__"].ge(start) & out["__ts__"].lt(end)
        train = out["__ts__"].lt(start - pd.Timedelta(hours=LABEL_DELAY_HOURS)) & out["__ts__"].ge(start - pd.Timedelta(days=LOOKBACK_DAYS))
        for arm, features in arms.items():
            column = f"mapped__{arm}"
            if int(train.sum()) < 500:
                out.loc[evaluate, column] = out.loc[evaluate, "score_residual_expected_ev"]
                mode = "cold_start_raw_residual"
            else:
                model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
                model.fit(out.loc[train, features], out.loc[train, "execution_net_ev_12h"])
                out.loc[evaluate, column] = model.predict(out.loc[evaluate, features])
                mode = "trailing_180d_causal_ridge"
            records.append({"arm": arm, "month": str(start.date())[:7], "mode": mode, "train_rows": int(train.sum()), "evaluation_rows": int(evaluate.sum()), "train_end_utc": out.loc[train, "__ts__"].max() if train.any() else None})
    mapped = [f"mapped__{name}" for name in arms]
    _finite(out, mapped)
    return out, pd.DataFrame(records)


def _selected_metrics(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    local = frame.loc[mask]
    return {
        "trades": int(len(local)),
        "gross_bps": float(local["execution_gross_ev_12h"].mean() * 10_000),
        "net_bps": float(local["execution_net_ev_12h"].mean() * 10_000),
        "cost_bps": float(local["execution_cost_return"].mean() * 10_000),
        "positive_net_rate": float(local["execution_net_ev_12h"].gt(0).mean()),
    }


def evaluate(mapped: pd.DataFrame, arms: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    side: list[dict[str, Any]] = []
    phase: list[dict[str, Any]] = []
    state: list[dict[str, Any]] = []
    for arm in arms:
        score = mapped[f"mapped__{arm}"]
        for fraction in TOPS:
            selected = _top_mask(mapped, score, fraction)
            aggregate.append({"arm": arm, "top_fraction": fraction, "rows": int(len(mapped)), "selection": "pooled_global_post_causal_mapping", "alpha_rank_ic": _rank_ic(score, mapped["__reconstructed_soft_alpha_12h__"]), "net_rank_ic": _rank_ic(score, mapped["execution_net_ev_12h"]), **_selected_metrics(mapped, selected)})
            chosen = mapped.loc[selected].copy()
            chosen["month"] = chosen["__ts__"].dt.strftime("%Y-%m")
            for month, local in chosen.groupby("month", observed=True, sort=True):
                monthly.append({"arm": arm, "top_fraction": fraction, "month": month, **_selected_metrics(local, np.ones(len(local), dtype=bool))})
            for name, local in chosen.groupby("side_name", observed=True, sort=True):
                side.append({"arm": arm, "top_fraction": fraction, "side_name": name, **_selected_metrics(local, np.ones(len(local), dtype=bool))})
            phase_name = chosen.filter(regex=r"^market_regime__phase_p_").idxmax(axis=1).str.removeprefix("market_regime__phase_p_")
            for name, local in chosen.assign(phase=phase_name).groupby("phase", observed=True, sort=True):
                phase.append({"arm": arm, "top_fraction": fraction, "phase": name, **_selected_metrics(local, np.ones(len(local), dtype=bool))})
            # Primary component ids are meaningful only inside their frozen
            # chronological fold.  Keep the fold id attached so this is a
            # within-fold diagnostic, never a cross-era semantic comparison.
            state_key = (
                chosen["regime_fold_id"].astype(str)
                + ":state_"
                + pd.to_numeric(chosen["regime_state_id"], errors="raise").astype(int).astype(str)
            )
            for name, local in chosen.assign(fold_local_state=state_key).groupby("fold_local_state", observed=True, sort=True):
                state.append({"arm": arm, "top_fraction": fraction, "fold_local_state": name, **_selected_metrics(local, np.ones(len(local), dtype=bool))})
    return pd.DataFrame(aggregate), pd.DataFrame(monthly), pd.DataFrame(side), pd.DataFrame(phase), pd.DataFrame(state)


def transport(panel: pd.DataFrame, arms: dict[str, list[str]]) -> pd.DataFrame:
    """Frozen 2023Q4 -> 2024 and 2024H1 -> 2024H2 transport diagnostics."""
    windows = (
        ("2023q4_to_2024", "2023-09-01", "2024-01-01", "2025-01-01"),
        ("2024h1_to_2024h2", "2024-01-01", "2024-07-01", "2025-01-01"),
    )
    rows: list[dict[str, Any]] = []
    for name, train_start, split, test_end in windows:
        train = panel["__ts__"].ge(pd.Timestamp(train_start, tz="UTC")) & panel["__ts__"].lt(pd.Timestamp(split, tz="UTC") - pd.Timedelta(hours=LABEL_DELAY_HOURS))
        test = panel["__ts__"].ge(pd.Timestamp(split, tz="UTC")) & panel["__ts__"].lt(pd.Timestamp(test_end, tz="UTC"))
        for arm, features in arms.items():
            model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
            model.fit(panel.loc[train, features], panel.loc[train, "execution_net_ev_12h"])
            score = pd.Series(model.predict(panel.loc[test, features]), index=panel.index[test])
            local = panel.loc[test].copy()
            local["score"] = score.to_numpy()
            selected = _top_mask(local, local["score"], 0.10)
            rows.append({"transport": name, "arm": arm, "train_rows": int(train.sum()), "test_rows": int(test.sum()), "top_fraction": 0.10, "net_rank_ic": _rank_ic(local["score"], local["execution_net_ev_12h"]), **_selected_metrics(local, selected)})
    return pd.DataFrame(rows)


def run(*, scores_path: Path = SCORES, states_path: Path = STATES, output_dir: Path = OUTPUT) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    panel = load_panel(Path(scores_path), Path(states_path))
    arms = arm_features(panel.columns)
    for fields in arms.values():
        _finite(panel, fields)
    mapped, folds = causal_monthly_map(panel, arms)
    aggregate, monthly, side, phase, state = evaluate(mapped, arms)
    transports = transport(panel, arms)
    coverage = pd.DataFrame([{"feature": name, "coverage": float(panel[name].notna().mean()), "nonconstant": bool(panel[name].nunique(dropna=True) > 1)} for name in sorted(set(field for values in arms.values() for field in values if field not in {"score_residual_expected_ev", "side_is_long"}))])
    output.mkdir(parents=True)
    for name, frame in {"aggregate_metrics.csv": aggregate, "monthly_global_topk.csv": monthly, "side_global_topk.csv": side, "phase_global_topk.csv": phase, "fold_local_state_global_topk.csv": state, "transport_metrics.csv": transports, "causal_mapping_folds.csv": folds, "feature_coverage.csv": coverage}.items():
        frame.to_csv(output / name, index=False)
    mapped.loc[:, [*IDENTITY_COLUMNS, "__reconstructed_soft_alpha_12h__", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", *[f"mapped__{arm}" for arm in arms]]].to_parquet(output / "mapped_scores.parquet", index=False, compression="zstd")
    baseline = aggregate.loc[(aggregate.arm == "A0_baseline") & (aggregate.top_fraction == 0.10)].iloc[0]
    gate = aggregate.loc[aggregate.top_fraction.eq(0.10), ["arm", "net_bps", "net_rank_ic"]].copy()
    gate["net_uplift_vs_baseline"] = gate["net_bps"] - baseline.net_bps
    gate["ic_uplift_vs_baseline"] = gate["net_rank_ic"] - baseline.net_rank_ic
    gate["advances"] = (gate["arm"] != "A0_baseline") & gate["net_uplift_vs_baseline"].gt(0) & gate["ic_uplift_vs_baseline"].gt(0)
    gate.to_csv(output / "advancement_gate.csv", index=False)
    manifest = {"schema": SCHEMA, "status": "COMPLETED_MATCHED_CAUSAL_ABLATION", "inputs": {"scores": {"path": str(Path(scores_path).resolve()), "sha256": _sha(Path(scores_path))}, "states": {"path": str(Path(states_path).resolve()), "sha256": _sha(Path(states_path))}}, "contract": {"arms": {name: fields for name, fields in arms.items()}, "selection": "one globally pooled top-k after each causal common-bps map", "no_actions": True, "no_state_ids": True, "mapping": "trailing 180d resolved-label Ridge; labels end before evaluation month", "transport": "frozen train-era to later-era diagnostic"}, "coverage": {"rows": int(len(panel)), "start": panel["__ts__"].min().isoformat(), "end": panel["__ts__"].max().isoformat(), "min_context_coverage": float(coverage.coverage.min())}, "outputs": {path.name: _sha(path) for path in output.iterdir() if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--states", type=Path, default=STATES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    values = _args()
    print(run(scores_path=values.scores, states_path=values.states, output_dir=values.output_dir))
