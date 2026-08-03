#!/usr/bin/env python3
"""No-refit direct-versus-gross-cost-hurdle blend ablation over sealed v3.

All scores are linear blends of *already frozen* v3 predictions.  The selected
weight is chosen once from the earlier May development OOF ledger only.  The
May--June and later-July forward ledgers are evaluation-only and never enter
weight selection, calibration, mapping, fitting, or policy optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data_perp/artifacts/exact_strict_oof_hurdle_distributional_ablation_20260730_v3"
SCHEMA = "frozen_hurdle_blend_ablation_v1"
WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
TARGET = "execution_net_ev_12h"
DECISION = "execution_decision_utc"
SIDE = "side_name"
DIRECT = "direct_net_residual"
HURDLE = "gross_cost_hurdle_ev"
MAPPED_DIRECT = "canonical_recent_ev_score_direct_net_residual"
MAPPED_HURDLE = "canonical_recent_ev_score_gross_cost_hurdle_ev"
OOF_DIRECT = "side_causal_oof_ev_direct_net_residual"
OOF_HURDLE = "side_causal_oof_ev_gross_cost_hurdle_ev"
DEVELOPMENT_WINDOW = "may_to_june_forward_control"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def blend(direct: np.ndarray, hurdle: np.ndarray, weight: float) -> np.ndarray:
    """Fixed convex blend; endpoints remain bit-identical source controls."""

    if float(weight) not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}")
    return (1.0 - float(weight)) * np.asarray(direct, dtype=float) + float(weight) * np.asarray(hurdle, dtype=float)


def _select(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> np.ndarray:
    """One pooled-global order with explicit immutable-ID cutoff ties."""

    score = np.asarray(score, dtype=float)
    valid = np.flatnonzero(np.isfinite(score))
    if not len(valid):
        raise ValueError("no finite score rows")
    count = max(1, int(np.ceil(float(fraction) * len(valid))))
    ranks = frame.iloc[valid].loc[:, list(IDENTITY)].copy()
    ranks["_score"] = score[valid]
    ranks["_position"] = valid
    for col in IDENTITY:
        ranks[col] = ranks[col].astype(str)
    ranks = ranks.sort_values(
        ["_score", "candidate_id", "__ts__", "__symbol__", SIDE],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return ranks["_position"].to_numpy(int)[:count]


def _tie_stats(score: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    score = np.asarray(score, dtype=float)
    finite = score[np.isfinite(score)]
    cutoff = float(score[selected[-1]])
    tie = np.flatnonzero(np.isfinite(score) & np.isclose(score, cutoff, rtol=0.0, atol=1e-14))
    return {
        "score_std": float(np.std(finite)),
        "score_distinct_values": int(pd.Series(finite).nunique(dropna=True)),
        "cutoff_score": cutoff,
        "cutoff_tie_rows": int(len(tie)),
        "cutoff_tie_selected_rows": int(np.isin(selected, tie).sum()),
        "cutoff_tie_fraction_selected": float(np.isin(selected, tie).mean()),
    }


def _metric_rows(frame: pd.DataFrame, score: np.ndarray, *, stage: str, weight: float, window: str, frozen_weight: float | None, development: bool) -> list[dict[str, Any]]:
    decision = pd.to_datetime(frame[DECISION], utc=True, errors="raise")
    latest_week = decision.max() - pd.Timedelta(days=7)
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        selected = _select(frame, score, fraction)
        chosen = frame.iloc[selected]
        week = chosen.loc[pd.to_datetime(chosen[DECISION], utc=True).ge(latest_week)]
        target = chosen[TARGET].to_numpy(float)
        rows.append({
            "window": window,
            "stage": stage,
            "weight_hurdle": float(weight),
            "is_frozen_weight": bool(frozen_weight is not None and np.isclose(weight, frozen_weight)),
            "development_only": bool(development),
            "top_fraction": float(fraction),
            "eligible_rows": int(np.isfinite(score).sum()),
            "top_k_rows": int(len(chosen)),
            "top_k_mean_net_ev": float(target.mean()),
            "top_k_sum_net_ev": float(target.sum()),
            "top_k_mean_net_bps": float(target.mean() * 1e4),
            "top_k_positive_rate": float((target > 0.0).mean()),
            "top_k_predicted_net_ev": float(np.asarray(score, dtype=float)[selected].mean()),
            "top_k_long_share": float(chosen[SIDE].astype(str).eq("long").mean()),
            "top_k_short_share": float(chosen[SIDE].astype(str).eq("short").mean()),
            "latest_week_selected_rows": int(len(week)),
            "latest_week_mean_net_bps": float(week[TARGET].mean() * 1e4) if len(week) else np.nan,
            "latest_week_positive_rate": float(week[TARGET].gt(0.0).mean()) if len(week) else np.nan,
            "latest_week_long_share": float(week[SIDE].astype(str).eq("long").mean()) if len(week) else np.nan,
            **_tie_stats(score, selected),
        })
    return rows


def _assert_source_seal(source: Path) -> dict[str, Any]:
    manifest = source / "manifest.json"
    seal = source / "manifest.sha256"
    if not manifest.exists() or not seal.exists():
        raise FileNotFoundError("v3 source must be sealed")
    expected = seal.read_text(encoding="utf-8").split()[0]
    actual = _sha256(manifest)
    if expected != actual:
        raise RuntimeError("v3 manifest seal mismatch")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for detail in payload["outputs"].values():
        artifact = Path(detail["path"])
        if not artifact.exists() or _sha256(artifact) != detail["sha256"]:
            raise RuntimeError(f"v3 source output binding invalid: {artifact}")
    return {"path": str(manifest), "sha256": actual}


def _choose_weight(development: pd.DataFrame, *, development_cutoff: pd.Timestamp) -> tuple[float, pd.DataFrame]:
    """Choose exactly once using the earlier resolved OOF ledger.

    Primary criterion is pooled-global top-10 exact net EV.  Ties use higher
    positive rate then lower hurdle weight.  The full fixed grid is retained;
    no forward score/outcome is read here.
    """

    required = {TARGET, OOF_DIRECT, OOF_HURDLE, "support_label_available_utc", "oof_fold"}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError("development OOF lacks columns: " + ", ".join(missing))
    available = pd.to_datetime(development["support_label_available_utc"], utc=True, errors="raise")
    # The development ledger ends before its own forward June window.  Require
    # resolved labels and real OOF folds explicitly rather than trusting names.
    usable = development.loc[
        development["oof_fold"].gt(0)
        & available.lt(pd.Timestamp(development_cutoff))
        & development[[OOF_DIRECT, OOF_HURDLE, TARGET]].notna().all(axis=1)
    ].copy()
    if usable.empty:
        raise ValueError("no resolved development OOF records")
    rows: list[dict[str, Any]] = []
    for weight in WEIGHTS:
        score = blend(usable[OOF_DIRECT].to_numpy(float), usable[OOF_HURDLE].to_numpy(float), weight)
        top = usable.iloc[_select(usable, score, 0.10)]
        rows.append({
            "weight_hurdle": weight,
            "development_rows": int(len(usable)),
            "development_end_utc": pd.to_datetime(usable[DECISION], utc=True).max(),
            "development_label_cutoff_utc": pd.Timestamp(development_cutoff),
            "top10_net_bps": float(top[TARGET].mean() * 1e4),
            "top10_positive_rate": float(top[TARGET].gt(0.0).mean()),
            "top10_long_share": float(top[SIDE].astype(str).eq("long").mean()),
        })
    table = pd.DataFrame(rows).sort_values(
        ["top10_net_bps", "top10_positive_rate", "weight_hurdle"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["selection_rank_development_only"] = np.arange(1, len(table) + 1)
    selected = float(table.loc[0, "weight_hurdle"])
    table["selected_frozen_weight"] = np.isclose(table["weight_hurdle"], selected)
    return selected, table


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_binding = _assert_source_seal(args.source_dir)
    # Only evaluation timestamps are opened before selection, solely to prove
    # the development label cutoff.  Forward scores/outcomes are read later.
    forward_metadata = pd.read_parquet(args.source_dir / "forward_predictions.parquet", columns=["window", DECISION])
    forward_start = pd.to_datetime(forward_metadata[DECISION], utc=True, errors="raise").min()
    oof = pd.read_parquet(args.source_dir / "support_head_oof_ledger.parquet")
    development = oof.loc[oof["window"].eq(DEVELOPMENT_WINDOW)].copy()
    frozen_weight, selection = _choose_weight(development, development_cutoff=forward_start)
    forward = pd.read_parquet(args.source_dir / "forward_predictions.parquet")
    output_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    exactness: list[dict[str, Any]] = []
    for window, frame in forward.groupby("window", sort=True):
        frame = frame.copy().reset_index(drop=True)
        for stage, direct_col, hurdle_col in (
            ("pre_existing_map", DIRECT, HURDLE),
            ("existing_causal_common_unit_map", MAPPED_DIRECT, MAPPED_HURDLE),
        ):
            if frame[[direct_col, hurdle_col, TARGET]].isna().any().any():
                raise ValueError(f"{window}/{stage} has missing frozen source score or outcome")
            for weight in WEIGHTS:
                name = f"blend_hurdle_{weight:0.2f}"
                score = blend(frame[direct_col].to_numpy(float), frame[hurdle_col].to_numpy(float), weight)
                output_rows.extend(_metric_rows(frame, score, stage=stage, weight=weight, window=str(window), frozen_weight=frozen_weight, development=False))
                prediction_parts.append(pd.DataFrame({
                    **{key: frame[key].to_numpy() for key in (*IDENTITY, DECISION, TARGET, "window")},
                    "stage": stage,
                    "weight_hurdle": weight,
                    "blend_score": score,
                    "is_frozen_weight": np.isclose(weight, frozen_weight),
                }))
                if weight in (0.0, 1.0):
                    control = frame[direct_col if weight == 0.0 else hurdle_col].to_numpy(float)
                    exactness.append({
                        "window": str(window), "stage": stage, "weight_hurdle": weight,
                        "control_column": direct_col if weight == 0.0 else hurdle_col,
                        "max_abs_difference": float(np.max(np.abs(score - control))),
                        "exact_control_reproduction": bool(np.array_equal(score, control)),
                    })
    metrics = pd.DataFrame(output_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    exact_controls = pd.DataFrame(exactness)
    if not exact_controls["exact_control_reproduction"].all():
        raise RuntimeError("pure direct/hurdle endpoints did not exactly reproduce v3 controls")
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    paths = {
        "development_weight_selection": staging / "development_weight_selection.csv",
        "forward_metrics": staging / "forward_metrics.csv",
        "forward_blend_predictions": staging / "forward_blend_predictions.parquet",
        "control_exactness": staging / "control_exactness.csv",
    }
    selection.to_csv(paths["development_weight_selection"], index=False)
    metrics.to_csv(paths["forward_metrics"], index=False)
    predictions.to_parquet(paths["forward_blend_predictions"], index=False)
    exact_controls.to_csv(paths["control_exactness"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_nonpromotion_evidence",
        "promotion_eligible": False,
        "source": source_binding,
        "contract": {
            "no_refit": "only sealed v3 OOF ledgers and forward predictions are read; no model, calibration, mapping, feature, or policy fitting is performed",
            "development_selection": "one weight selected exactly once from resolved May development OOF only by pooled-global top10 exact net EV; forward outcomes excluded",
            "weights": list(WEIGHTS),
            "frozen_weight": frozen_weight,
            "scores": "pre-map linear blend of frozen direct and hurdle scores; mapped linear blend of their existing frozen causal common-unit map outputs (no blend map fit)",
            "ranking": "one pooled-global candidate-ID-stable order at each fixed top 1/5/10/20%; never timestamp/side/asset local",
            "policy_portfolio": "not run; no forward arm meets economics gates",
        },
        "outputs": {name: {"path": str(args.output_dir / path.name), "sha256": _sha256(path)} for name, path in paths.items()},
    }
    manifest_path = staging / "manifest.json"
    _write_json(manifest_path, manifest)
    (staging / "manifest.sha256").write_text(f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8")
    os.replace(staging, args.output_dir)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=V3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
