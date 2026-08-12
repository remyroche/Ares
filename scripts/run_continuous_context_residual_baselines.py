#!/usr/bin/env python3
"""Strict-prequential continuous context residual baselines.

This is a deliberately small diagnostic ladder for the question: can a
*continuous*, causal market-context surface repair broad residual-location
error?  It is not a regime expert, cluster test, candidate generator, or HPO
search.  The four fixed arms are:

``P0``
    A global residual-location estimate.  The estimate is reported, while its
    selection score remains exactly the frozen residual expected-EV score; it
    cannot rerank candidates, including across folds.
``P2``
    Regularised linear correction from compact continuous context and explicit
    base-output x context interactions.
``P3``
    The same contract through low-knot additive splines plus Ridge.  This is a
    GAM-like non-linear control, not a high-dimensional feature search.
``P4``
    A shallow, strongly regularised histogram tree correction.

All fitting rows satisfy ``label_available_ts < evaluation_start``.  Inputs
explicitly reject state IDs, posterior/membership vectors and cluster/GMM
outputs.  Selection is one pooled global top-k after the common expected-net
score; month tables are diagnostics, never per-timestamp rankings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity
from extreme_price_movements.causal_market_regime_systems import CONTINUOUS_CONTEXT_FEATURE_KEYS


SCHEMA = "continuous_context_residual_baselines_v1"
DEFAULT_SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
DEFAULT_CONTEXT = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/continuous_context_residual_baselines_20260803_v1"
LABEL_DELAY = pd.Timedelta(hours=12)
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_MIN_TRAIN_ROWS = 1_000
TOP_FRACTIONS = (0.01, 0.05, 0.10)

# One field per broad observable geometry.  These are raw, causal multiview
# features at a 6h-or-shorter horizon appropriate to the 6--12h trade.  They
# intentionally are not learned latent-state coordinates.
DEFAULT_CONTEXT_FIELDS = (
    "mv__market_state_transition_entropy_5d__robust_z_6h",
    "mv__breadth_dispersion__robust_z_6h",
    "mv__negative_breadth_pct__robust_z_6h",
    "mv__correlation_breakdown_dispersion__robust_z_6h",
    "mv__peer_volatility_decoupling__robust_z_6h",
    "mv__liquidity__liquidity_xs__amihud_illiq__mean__stress_1h",
    "mv__funding_deleveraging_divergence__robust_z_6h",
    "mv__btc_resilience_alt_weakness__robust_z_6h",
    "mv__deleveraging_without_followthrough__robust_z_6h",
)
BASE_OUTPUTS = ("score_base_expected_ev", "score_residual_expected_ev")
FORBIDDEN_CONTEXT_TOKENS = (
    "cluster", "gmm", "membership", "posterior", "state_p_", "state_id",
    "market_regime__", "geometry_regime__", "phase_p_", "latent", "archetype",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_ic(score: pd.Series, outcome: pd.Series) -> float:
    mask = score.notna() & outcome.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(score.loc[mask].rank(method="average").corr(outcome.loc[mask].rank(method="average")))


def _top_mask(frame: pd.DataFrame, score: pd.Series, fraction: float) -> np.ndarray:
    n = max(1, int(np.ceil(len(frame) * float(fraction))))
    order = pd.DataFrame({"score": score.to_numpy(float), "candidate_id": frame["candidate_id"].astype(str)}, index=frame.index)
    return frame.index.isin(order.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(n).index)


def _top_metrics(frame: pd.DataFrame, score: pd.Series, fraction: float) -> dict[str, float]:
    local = frame.loc[_top_mask(frame, score, fraction)]
    return {
        "trades": int(len(local)),
        "gross_bps": float(local["execution_gross_ev_12h"].mean() * 10_000),
        "net_bps": float(local["execution_net_ev_12h"].mean() * 10_000),
        "cost_bps": float(local["execution_cost_return"].mean() * 10_000),
        "positive_net_rate": float(local["execution_net_ev_12h"].gt(0.0).mean()),
    }


def _validate_context_fields(fields: Sequence[str]) -> list[str]:
    names = list(dict.fromkeys(str(name) for name in fields))
    if not names:
        raise ValueError("at least one continuous context field is required")
    forbidden = [name for name in names if any(token in name.lower() for token in FORBIDDEN_CONTEXT_TOKENS)]
    if forbidden:
        raise ValueError(f"state/membership/cluster-derived context is forbidden: {forbidden}")
    return names


def _default_context_fields(path: Path) -> tuple[str, ...]:
    """Prefer the fully materialized relative contract when it is present.

    The raw hourly store remains a supported diagnostic input, whose compact
    source fields are intentionally different.  Detecting the sidecar schema
    here prevents a caller from accidentally requesting raw-store names from
    an exact candidate sidecar.
    """

    names = set(pq.ParquetFile(path).schema.names)
    if set(CONTINUOUS_CONTEXT_FEATURE_KEYS).issubset(names):
        return tuple(CONTINUOUS_CONTEXT_FEATURE_KEYS)
    return tuple(DEFAULT_CONTEXT_FIELDS)


def _candidate_sidecar(path: Path) -> bool:
    """Whether ``path`` can be joined by the exact candidate identity.

    The pre-existing source panel is hourly, while the new materializer emits
    candidate-keyed continuous context.  Prefer the latter when all identity
    fields are present: this prevents an accidental many-to-one hourly join.
    """

    try:
        sample = pd.read_parquet(path, columns=list(IDENTITY_COLUMNS))
    except Exception:  # the hourly source intentionally has no candidate key
        return False
    return set(IDENTITY_COLUMNS).issubset(sample.columns)


def _validate_sidecar_availability(sidecar: pd.DataFrame) -> str:
    availability = [name for name in sidecar if name.endswith("_available_utc")]
    if "source_utc" in sidecar:
        availability.append("source_utc")
    for name in dict.fromkeys(availability):
        value = pd.to_datetime(sidecar[name], utc=True, errors="raise")
        if (value > sidecar["__ts__"]).fillna(False).any():
            raise ValueError(f"candidate context sidecar looks ahead via {name}")
    # A keyed sidecar's identity timestamp is itself the minimum lineage
    # contract.  An explicit availability field is audited when supplied.
    return ",".join(dict.fromkeys(availability)) or "candidate_identity_timestamp"


def _load_panel(*, scores_path: Path, context_path: Path, context_fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = _validate_context_fields(context_fields)
    required = [
        *IDENTITY_COLUMNS, "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return",
        "score_base_expected_ev", "score_residual_expected_ev", "residual_is_oof",
    ]
    scores = validate_candidate_identity(pd.read_parquet(scores_path, columns=required)).copy()
    if not scores["residual_is_oof"].astype(bool).all():
        raise ValueError("all score rows must be residual OOF rows")
    scores["__ts__"] = pd.to_datetime(scores["__ts__"], utc=True, errors="raise")
    for column in ("execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", *BASE_OUTPUTS):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if not np.isfinite(scores.loc[:, ["execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", *BASE_OUTPUTS]].to_numpy(float)).all():
        raise ValueError("score ledger has non-finite outcome or base-output fields")

    if _candidate_sidecar(context_path):
        sidecar = validate_candidate_identity(pd.read_parquet(context_path, columns=[*IDENTITY_COLUMNS, *fields]))
        # Read optional availability lineage without imposing it on an older
        # exact-keyed sidecar.  Its absence is visible in the coverage audit.
        for optional in ("context_available_utc", "feature_available_utc", "regime_available_utc", "source_utc"):
            try:
                lineage = pd.read_parquet(context_path, columns=[*IDENTITY_COLUMNS, optional])
            except Exception:
                continue
            sidecar = sidecar.merge(lineage, on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")
        sidecar["__ts__"] = pd.to_datetime(sidecar["__ts__"], utc=True, errors="raise")
        availability_contract = _validate_sidecar_availability(sidecar)
        score_ids = scores.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        sidecar_ids = sidecar.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        # A materialized sidecar may intentionally cover a later bounded OOF
        # era.  Permit that exact *time-contiguous* subset, but never allow a
        # missing candidate inside the claimed sidecar interval or an extra
        # sidecar identity.  All P arms are subsequently evaluated only on
        # this same declared intersection.
        side_start, side_end = sidecar["__ts__"].min(), sidecar["__ts__"].max()
        score_in_span = scores.loc[scores["__ts__"].between(side_start, side_end)].copy()
        expected_ids = score_in_span.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        if not expected_ids.equals(sidecar_ids):
            raise ValueError("candidate context sidecar does not exactly match the score universe within its time span")
        scores = score_in_span
        joined = scores.merge(sidecar.loc[:, [*IDENTITY_COLUMNS, *fields]], on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        join_mode = "exact_candidate_identity_sidecar"
    else:
        source = pd.read_parquet(context_path, columns=["source_utc", *fields]).copy()
        source["source_utc"] = pd.to_datetime(source["source_utc"], utc=True, errors="raise")
        source = source.sort_values("source_utc", kind="stable").drop_duplicates("source_utc", keep="last")
        joined = pd.merge_asof(
            scores.sort_values("__ts__", kind="stable"), source,
            left_on="__ts__", right_on="source_utc", direction="backward", tolerance=pd.Timedelta(hours=2),
        )
        if (joined["source_utc"] > joined["__ts__"]).fillna(False).any():
            raise ValueError("continuous context join looked ahead")
        availability_contract = "source_utc"
        join_mode = "hourly_source_asof_backward_2h"
    coverage = pd.DataFrame({
        "feature": fields,
        "pre_intersection_coverage": [float(joined[name].notna().mean()) for name in fields],
        "nonconstant": [bool(pd.to_numeric(joined[name], errors="coerce").nunique(dropna=True) > 1) for name in fields],
        "forbidden_cluster_or_state_feature": False,
    })
    # An older score ledger can predate a newly materialised causal context
    # panel.  Availability is not an economic outcome, so restrict every arm
    # to the same causal-support intersection rather than median-imputing an
    # entire absent era or silently using P0 on a different universe.
    coverage["admitted"] = coverage["pre_intersection_coverage"].gt(0.05) & coverage["nonconstant"]
    admitted = coverage.loc[coverage["admitted"], "feature"].tolist()
    if not admitted:
        raise ValueError("no context feature has causal support and non-constant values")
    rows_before_intersection = len(joined)
    joined = joined.loc[joined.loc[:, admitted].notna().all(axis=1)].copy()
    if joined.empty:
        raise ValueError("causal context support intersection is empty")
    coverage["post_intersection_coverage"] = [float(joined[name].notna().mean()) for name in fields]
    coverage["post_intersection_rows"] = int(len(joined))
    coverage["rows_before_intersection"] = int(rows_before_intersection)
    coverage["context_join_mode"] = join_mode
    coverage["availability_lineage"] = availability_contract
    if not coverage.loc[coverage["admitted"], "post_intersection_coverage"].ge(0.90).all():
        raise ValueError("admitted continuous features do not reach 90% coverage after support intersection")
    # All P arms use the same valid-row population.  Median imputation occurs
    # inside each training fold, not from future rows.
    for column in admitted:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    joined["label_available_ts"] = joined["__ts__"] + LABEL_DELAY
    joined["side_is_long"] = joined["side_name"].astype(str).eq("long").astype(float)
    return joined.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), coverage


def _add_interactions(frame: pd.DataFrame, context: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    names = [*BASE_OUTPUTS, "side_is_long", *context]
    derived: dict[str, np.ndarray] = {}
    for output in BASE_OUTPUTS:
        for field in context:
            name = f"interaction__{output}__x__{field}"
            # Assemble in one frame below: inserting 126 columns one-by-one
            # fragments the candidate ledger and is needlessly expensive.
            derived[name] = out[output].to_numpy(float) * out[field].to_numpy(float)
            names.append(name)
    return pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1), names


@dataclass(frozen=True)
class Arm:
    name: str
    description: str
    model: str


ARMS = (
    Arm("P0_location_only", "prior resolved global residual-location estimate; frozen score is untouched", "location"),
    Arm("P2_linear_continuous", "regularised linear continuous context residual correction", "linear"),
    Arm("P3_spline_additive", "low-knot additive spline/Ridge continuous correction", "spline"),
    Arm("P4_shallow_tree", "shallow strongly regularised continuous-context tree correction", "tree"),
)


def _fit_predict(train: pd.DataFrame, evaluate: pd.DataFrame, features: Sequence[str], arm: Arm) -> np.ndarray:
    raw = train["score_residual_expected_ev"].to_numpy(float)
    target = train["execution_net_ev_12h"].to_numpy(float) - raw
    if arm.model == "location":
        # P0 estimates and reports the broad residual location separately,
        # but deliberately leaves the ranking score untouched.  Even a
        # fold-specific additive intercept would reshuffle candidates across
        # folds once pooled globally, contrary to this control's contract.
        return evaluate["score_residual_expected_ev"].to_numpy(float)
    x = train.loc[:, features]
    z = evaluate.loc[:, features]
    if arm.model == "linear":
        model: Any = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=80.0)),
        ])
    elif arm.model == "spline":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(n_knots=4, degree=2, extrapolation="linear")),
            ("ridge", Ridge(alpha=120.0)),
        ])
    elif arm.model == "tree":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("tree", HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=120, max_leaf_nodes=8, max_depth=3,
                min_samples_leaf=500, l2_regularization=25.0, random_state=20260803,
            )),
        ])
    else:  # pragma: no cover - arm table is fixed above
        raise ValueError(arm.model)
    model.fit(x, target)
    correction = np.asarray(model.predict(z), dtype=float)
    return evaluate["score_residual_expected_ev"].to_numpy(float) + correction


def causal_monthly_predictions(
    panel: pd.DataFrame,
    *,
    features: Sequence[str],
    lookback_days: int,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = panel.copy()
    audits: list[dict[str, Any]] = []
    first = result["__ts__"].min()
    last = result["__ts__"].max()
    month_start = pd.Timestamp(year=first.year, month=first.month, day=1, tz="UTC")
    month_after_last = pd.Timestamp(year=last.year, month=last.month, day=1, tz="UTC") + pd.offsets.MonthBegin(1)
    starts = pd.date_range(
        month_start,
        month_after_last,
        freq="MS", tz="UTC",
    )
    for start, end in zip(starts[:-1], starts[1:]):
        test = result["__ts__"].ge(start) & result["__ts__"].lt(end)
        train = result["label_available_ts"].lt(start) & result["__ts__"].ge(start - pd.Timedelta(days=lookback_days))
        if train.any() and result.loc[train, "label_available_ts"].max() >= start:
            raise ValueError("unresolved label entered a chronological training fold")
        mode = "strict_prequential"
        for arm in ARMS:
            column = f"score__{arm.name}"
            if int(train.sum()) < int(min_train_rows):
                result.loc[test, column] = result.loc[test, "score_residual_expected_ev"]
                if arm.model == "location":
                    result.loc[test, "location_shift__P0"] = 0.0
                mode = "cold_start_raw_residual"
            else:
                result.loc[test, column] = _fit_predict(result.loc[train], result.loc[test], features, arm)
                if arm.model == "location":
                    result.loc[test, "location_shift__P0"] = float((result.loc[train, "execution_net_ev_12h"] - result.loc[train, "score_residual_expected_ev"]).mean())
        # Exact invariant test, per fold.  P0 only relocates the entire fold.
        local = result.loc[test]
        if not np.array_equal(local["score_residual_expected_ev"].to_numpy(float), local["score__P0_location_only"].to_numpy(float)):
            raise AssertionError("P0 changed a candidate score")
        audits.append({
            "evaluation_month": start.strftime("%Y-%m"), "mode": mode,
            "train_rows": int(train.sum()), "evaluation_rows": int(test.sum()),
            "train_start_utc": result.loc[train, "__ts__"].min() if train.any() else None,
            "train_label_available_max_utc": result.loc[train, "label_available_ts"].max() if train.any() else None,
            "evaluation_start_utc": start,
            "ranking_invariant_p0_within_fold": True,
        })
    scores = [f"score__{arm.name}" for arm in ARMS]
    if result.loc[:, scores].isna().any().any() or not np.isfinite(result.loc[:, scores].to_numpy(float)).all():
        raise ValueError("prediction panel has missing/non-finite causal scores")
    return result, pd.DataFrame(audits)


def _aggregate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    side: list[dict[str, Any]] = []
    for arm in ARMS:
        score = predictions[f"score__{arm.name}"]
        for top in TOP_FRACTIONS:
            selected = _top_mask(predictions, score, top)
            aggregate.append({
                "arm": arm.name, "top_fraction": top, "selection_basis": "pooled_global_post_common_expected_net_score",
                "rows": int(len(predictions)), "net_rank_ic": _rank_ic(score, predictions["execution_net_ev_12h"]),
                "gross_rank_ic": _rank_ic(score, predictions["execution_gross_ev_12h"]), **_top_metrics(predictions, score, top),
            })
            chosen = predictions.loc[selected].copy()
            chosen["month"] = chosen["__ts__"].dt.strftime("%Y-%m")
            for month, local in chosen.groupby("month", observed=True, sort=True):
                monthly.append({"arm": arm.name, "top_fraction": top, "selection_basis": "members_of_pooled_global_top_k", "month": month, **_top_metrics(local, local[f"score__{arm.name}"], 1.0)})
            for name, local in chosen.groupby("side_name", observed=True, sort=True):
                side.append({"arm": arm.name, "top_fraction": top, "side_name": name, **_top_metrics(local, local[f"score__{arm.name}"], 1.0)})
            # Independent month top-k is a diagnostic only; it is useful for a
            # worst-period check without pretending the live policy is local.
            for month, local in predictions.assign(month=predictions["__ts__"].dt.strftime("%Y-%m")).groupby("month", observed=True, sort=True):
                monthly.append({"arm": arm.name, "top_fraction": top, "selection_basis": "diagnostic_month_local_top_k_not_live_policy", "month": month, **_top_metrics(local, local[f"score__{arm.name}"], top)})
    return pd.DataFrame(aggregate), pd.DataFrame(monthly), pd.DataFrame(side)


TRANSPORT_WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    # Match the predeclared historical regime-transport controls.  These are
    # fixed chronological train -> later-evaluation windows, never selected
    # from results and never local timestamp rankings.
    ("2023q4_to_2024", "2023-09-01", "2024-01-01", "2025-01-01"),
    ("2024h1_to_2024h2", "2024-01-01", "2024-07-01", "2025-01-01"),
)


def _transport_metrics(panel: pd.DataFrame, features: Sequence[str], *, min_train_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, train_start, split, test_end in TRANSPORT_WINDOWS:
        start = pd.Timestamp(train_start, tz="UTC")
        split_at = pd.Timestamp(split, tz="UTC")
        end = pd.Timestamp(test_end, tz="UTC")
        train = panel["__ts__"].ge(start) & panel["label_available_ts"].lt(split_at)
        test = panel["__ts__"].ge(split_at) & panel["__ts__"].lt(end)
        if int(train.sum()) < int(min_train_rows) or not test.any():
            continue
        for arm in ARMS:
            score = _fit_predict(panel.loc[train], panel.loc[test], features, arm)
            local = panel.loc[test].copy()
            local["score"] = score
            for top in TOP_FRACTIONS:
                rows.append({
                    "transport": name, "arm": arm.name, "top_fraction": top,
                    "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                    "train_label_available_max_utc": panel.loc[train, "label_available_ts"].max(),
                    "net_rank_ic": _rank_ic(local["score"], local["execution_net_ev_12h"]),
                    "gross_rank_ic": _rank_ic(local["score"], local["execution_gross_ev_12h"]),
                    **_top_metrics(local, local["score"], top),
                })
    return pd.DataFrame(rows)


def _worst_periods(monthly: pd.DataFrame, transport: pd.DataFrame) -> pd.DataFrame:
    live = monthly.loc[monthly["selection_basis"].eq("members_of_pooled_global_top_k")]
    month_summary = live.groupby(["arm", "top_fraction"], observed=True).agg(
        worst_month_net_bps=("net_bps", "min"), median_month_net_bps=("net_bps", "median"),
        months_with_selected_trades=("month", "nunique"),
    ).reset_index()
    if transport.empty:
        month_summary["worst_transport_net_bps"] = np.nan
        return month_summary
    transport_summary = transport.groupby(["arm", "top_fraction"], observed=True)["net_bps"].min().rename("worst_transport_net_bps").reset_index()
    return month_summary.merge(transport_summary, on=["arm", "top_fraction"], how="left", validate="one_to_one")


def run(
    *,
    scores_path: Path = DEFAULT_SCORES,
    context_path: Path = DEFAULT_CONTEXT,
    output_dir: Path = DEFAULT_OUTPUT,
    context_fields: Sequence[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    selected_context_fields = tuple(context_fields) if context_fields is not None else _default_context_fields(Path(context_path))
    panel, coverage = _load_panel(scores_path=Path(scores_path), context_path=Path(context_path), context_fields=selected_context_fields)
    admitted = coverage.loc[coverage["admitted"], "feature"].tolist()
    panel, features = _add_interactions(panel, admitted)
    predictions, folds = causal_monthly_predictions(panel, features=features, lookback_days=lookback_days, min_train_rows=min_train_rows)
    aggregate, monthly, side = _aggregate_metrics(predictions)
    transport = _transport_metrics(panel, features, min_train_rows=min_train_rows)
    worst = _worst_periods(monthly, transport)
    output_tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        keep = [*IDENTITY_COLUMNS, "label_available_ts", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", *BASE_OUTPUTS, *admitted, "location_shift__P0", *(f"score__{arm.name}" for arm in ARMS)]
        outputs: dict[str, pd.DataFrame] = {
            "causal_oof_predictions.parquet": predictions.loc[:, keep],
            "feature_coverage.csv": coverage,
            "causal_folds.csv": folds,
            "aggregate_metrics.csv": aggregate,
            "monthly_global_topk.csv": monthly,
            "side_global_topk.csv": side,
            "transport_metrics.csv": transport,
            "worst_period_metrics.csv": worst,
        }
        for name, frame in outputs.items():
            path = output_tmp / name
            if path.suffix == ".parquet":
                frame.to_parquet(path, index=False)
            else:
                frame.to_csv(path, index=False)
        manifest = {
            "schema": SCHEMA, "status": "COMPLETED_STRICT_PREQUENTIAL_CONTINUOUS_CONTEXT_DIAGNOSTIC",
            "inputs": {"scores": {"path": str(Path(scores_path).resolve()), "sha256": _sha(Path(scores_path))}, "context": {"path": str(Path(context_path).resolve()), "sha256": _sha(Path(context_path))}},
            "contract": {
                "target": "execution_net_ev_12h - frozen score_residual_expected_ev", "label_availability": "candidate decision time + 12h; train labels strictly resolve before evaluation month/year", "rolling_training": f"{lookback_days} days", "ranking": "one pooled global top-k after common expected-net score", "p0": "global residual location is reported in location_shift__P0 but its selection score is exactly the frozen residual score; P0 never reranks a candidate", "no_discrete_state_membership_or_cluster_inputs": True,
                "base_output_x_context_interactions": list(features[len(BASE_OUTPUTS) + 1 + len(admitted):]),
                "arms": [{"name": arm.name, "description": arm.description, "model": arm.model} for arm in ARMS],
                "admitted_context_features": admitted,
            },
            "outputs": {name: _sha(output_tmp / name) for name in outputs},
        }
        (output_tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(output_tmp, output)
    except Exception:
        shutil.rmtree(output_tmp, ignore_errors=True)
        raise
    return output


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-field", action="append", default=None, help="repeat to override the compact default causal continuous fields")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    values = _args()
    print(run(scores_path=values.scores, context_path=values.context, output_dir=values.output_dir, context_fields=values.context_field, lookback_days=values.lookback_days, min_train_rows=values.min_train_rows))
