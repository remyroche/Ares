#!/usr/bin/env python3
"""Materialize causal frozen-AE/GMM transition features for meta candidates.

The frozen AE/GMM cycle state is deliberately row-independent.  Its posterior,
entropy, OOD and reconstruction outputs are therefore stable under replay/live
batch layout, but its built-in temporal outputs are zero by contract.  This
utility derives *separate*, causal state transitions on complete symbol time
series and joins them to a sparse candidate ledger.

No outcome, target, selection, or recent-performance field is read.  Market
transition breadth is calculated on the full static universe before candidate
rows are selected, so the result cannot depend on the candidate population.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.inference.live_meta_feature_overlays import (  # noqa: E402
    materialize_live_source_regime_features,
)
from scripts.materialize_full_cross_section_source_regimes import (  # noqa: E402
    _store_timestamp,
    configured_causal_source_columns,
    full_static_source_panel,
    required_source_columns,
)


TRANSITION_FEATURES = (
    "meta_aegmm_transition_posterior_tv_1h",
    "meta_aegmm_transition_posterior_tv_4h",
    "meta_aegmm_transition_cluster_switch_1h",
    "meta_aegmm_transition_cluster_switch_4h",
    "meta_aegmm_transition_entropy_delta_1h",
    "meta_aegmm_transition_entropy_delta_4h",
    "meta_aegmm_transition_posterior_max_delta_1h",
    "meta_aegmm_transition_posterior_max_delta_4h",
    "meta_aegmm_transition_reconstruction_delta_1h",
    "meta_aegmm_transition_reconstruction_delta_4h",
    "meta_aegmm_transition_mahal_delta_1h",
    "meta_aegmm_transition_mahal_delta_4h",
    "meta_aegmm_transition_ood_delta_1h",
    "meta_aegmm_transition_ood_delta_4h",
    "meta_aegmm_transition_latent_speed_1h",
    "meta_aegmm_transition_latent_speed_4h",
    "meta_aegmm_transition_market_breadth_1h",
    "meta_aegmm_transition_market_breadth_4h",
    "meta_aegmm_transition_market_entropy_delta_1h",
    "meta_aegmm_transition_market_ood_delta_1h",
)
AEGMM_COMPONENT_COUNT = 6
COMPONENT_TRANSITION_FEATURES = tuple(
    feature
    for component in range(AEGMM_COMPONENT_COUNT)
    for feature in (
        f"meta_aegmm_transition_prob_{component}_delta_1h",
        f"meta_aegmm_transition_prob_{component}_delta_4h",
        f"meta_aegmm_transition_prob_{component}_enter_breadth_1h",
        f"meta_aegmm_transition_prob_{component}_exit_breadth_1h",
    )
)
# A posterior change is not necessarily a durable state transition.  These
# fields describe whether the dominant frozen component has just changed, how
# long the new state has persisted, and whether the transition is systemic.
# They are deliberately causal and reset at every internal time-series gap.
DOMINANT_STATE_TRANSITION_FEATURES = (
    "meta_aegmm_transition_dominant_state_age_24h_norm",
    "meta_aegmm_transition_dominant_switch_count_4h",
    "meta_aegmm_transition_dominant_switch_count_8h",
    "meta_aegmm_transition_market_dominant_switch_breadth_1h",
    "meta_aegmm_transition_market_dominant_switch_breadth_4h",
    "meta_aegmm_transition_market_dominant_concentration",
    "meta_aegmm_transition_market_dominant_entropy",
)
ALL_TRANSITION_FEATURES = (
    *TRANSITION_FEATURES,
    *COMPONENT_TRANSITION_FEATURES,
    *DOMINANT_STATE_TRANSITION_FEATURES,
)

_POINT_COLUMNS = (
    "gmm_cluster_id",
    "gmm_entropy",
    "gmm_posterior_max",
    "AE_reconstruction_error",
    "mahalanobis_distance",
    "gmm_ood_score",
    *tuple(f"gmm_prob_{idx}" for idx in range(12)),
    *tuple(f"dae_b16_{idx:02d}" for idx in range(16)),
)


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _normalise_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    timestamp_col = "__ts__" if "__ts__" in out.columns else "timestamp"
    symbol_col = "__symbol__" if "__symbol__" in out.columns else "symbol"
    if timestamp_col not in out or symbol_col not in out:
        raise ValueError("candidate ledger requires __ts__/__symbol__ or timestamp/symbol")
    out["__ts__"] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    out["__symbol__"] = out[symbol_col].astype(str)
    if "side_name" not in out:
        raw_side = pd.to_numeric(out.get("side", 1.0), errors="coerce").fillna(1.0)
        out["side_name"] = np.where(raw_side.to_numpy() < 0.0, "short", "long")
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out = out.loc[out["__ts__"].notna() & out["__symbol__"].ne("")].copy()
    return out


def _lagged(values: pd.DataFrame, *, lag_hours: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return exact-hour lags; gaps remain missing rather than being bridged."""

    grouped = values.groupby(["side_name", "__symbol__"], observed=True, sort=False)
    lagged = grouped[list(values.columns.difference(["side_name", "__symbol__", "__ts__"]))].shift(
        int(lag_hours)
    )
    previous_ts = grouped["__ts__"].shift(int(lag_hours))
    contiguous = (
        (values["__ts__"] - previous_ts).eq(pd.Timedelta(hours=int(lag_hours))).to_numpy()
    )
    return lagged, contiguous


def _add_dominant_state_transition_features(
    panel: pd.DataFrame,
    out: pd.DataFrame,
    probs: list[str],
) -> pd.DataFrame:
    """Add gap-safe state-entry/dwell and full-universe transition context."""

    probability = panel.loc[:, probs[:AEGMM_COMPONENT_COUNT]].to_numpy(
        dtype=np.float32, copy=False
    )
    dominant = np.argmax(probability, axis=1).astype(np.int16, copy=False)
    same_stream = (
        panel["side_name"].eq(panel["side_name"].shift())
        & panel["__symbol__"].eq(panel["__symbol__"].shift())
    )
    contiguous = (
        same_stream
        & panel["__ts__"].sub(panel["__ts__"].shift()).eq(pd.Timedelta(hours=1))
    )
    prior_dominant = np.roll(dominant, 1)
    state_changed = dominant != prior_dominant
    switch = np.where(contiguous.to_numpy(), state_changed.astype(np.float32), np.nan)

    # Each gap begins a new sequence.  State runs also restart on a dominant
    # component change, yielding a bounded causal age for the current state.
    segment_id = (~contiguous).cumsum()
    state_run_id = ((~contiguous).to_numpy() | state_changed).cumsum()
    age = pd.Series(np.arange(len(panel)), dtype=np.int32).groupby(
        state_run_id, sort=False
    ).cumcount().to_numpy(dtype=np.float32)
    out["meta_aegmm_transition_dominant_state_age_24h_norm"] = np.minimum(
        age / np.float32(24.0), np.float32(1.0)
    ).astype(np.float32)
    switch_series = pd.Series(switch, dtype=np.float32)
    for window in (4, 8):
        count = (
            switch_series.groupby(segment_id, sort=False)
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .reindex(switch_series.index)
            .to_numpy(dtype=np.float32)
        )
        count[~contiguous.to_numpy()] = np.nan
        out[f"meta_aegmm_transition_dominant_switch_count_{window}h"] = count

    work = pd.DataFrame(
        {
            "__ts__": out["__ts__"],
            "side_name": out["side_name"],
            "dominant": dominant,
            "switch": switch,
        }
    )
    for window in (1, 4):
        source = "switch" if window == 1 else "meta_aegmm_transition_dominant_switch_count_4h"
        values = work["switch"] if window == 1 else out[source]
        breadth = (
            pd.DataFrame(
                {"__ts__": out["__ts__"], "side_name": out["side_name"], "value": values}
            )
            .groupby(["__ts__", "side_name"], observed=True)["value"]
            .mean()
            .rename(f"meta_aegmm_transition_market_dominant_switch_breadth_{window}h")
            .reset_index()
        )
        out = out.merge(breadth, on=["__ts__", "side_name"], how="left", validate="many_to_one")

    # Cross-sectional concentration/entropy tells the meta layer whether a
    # local state change is isolated or part of a synchronized market move.
    counts = (
        work.groupby(["__ts__", "side_name", "dominant"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["share"] = counts["count"].div(
        counts.groupby(["__ts__", "side_name"], observed=True)["count"].transform("sum")
    )
    counts["entropy_term"] = -counts["share"] * np.log(
        np.clip(counts["share"].to_numpy(dtype=np.float64), 1e-8, 1.0)
    )
    summary = (
        counts.groupby(["__ts__", "side_name"], observed=True)
        .agg(
            meta_aegmm_transition_market_dominant_concentration=("share", "max"),
            meta_aegmm_transition_market_dominant_entropy=("entropy_term", "sum"),
        )
        .reset_index()
    )
    out = out.merge(summary, on=["__ts__", "side_name"], how="left", validate="many_to_one")
    return out


def add_causal_aegmm_transition_features(points: pd.DataFrame) -> pd.DataFrame:
    """Derive batch-invariant state transitions from complete ordered panels.

    ``points`` must contain one row per timestamp/symbol/side with frozen,
    row-independent AE/GMM outputs.  It may be sparse at internal gaps; exact
    timestamp checks ensure a missing bar never becomes a fabricated transition.
    """

    required = {"__ts__", "__symbol__", "side_name", *_POINT_COLUMNS}
    missing = sorted(required.difference(points.columns))
    if missing:
        raise ValueError(f"AE/GMM point panel missing transition inputs: {missing[:12]}")
    panel = points.loc[:, ["__ts__", "__symbol__", "side_name", *_POINT_COLUMNS]].copy()
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["__ts__", "__symbol__"])
    panel = panel.sort_values(["side_name", "__symbol__", "__ts__"], kind="stable").reset_index(drop=True)
    numeric = [column for column in _POINT_COLUMNS if column in panel]
    panel.loc[:, numeric] = panel.loc[:, numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    out = panel.loc[:, ["__ts__", "__symbol__", "side_name"]].copy()
    probs = [column for column in panel.columns if column.startswith("gmm_prob_")]
    latents = [column for column in panel.columns if column.startswith("dae_b16_")]

    for hours, suffix in ((1, "1h"), (4, "4h")):
        previous, contiguous = _lagged(panel, lag_hours=hours)
        current_prob = panel.loc[:, probs].to_numpy(dtype=np.float32, copy=False)
        prior_prob = previous.loc[:, probs].to_numpy(dtype=np.float32, copy=False)
        posterior_tv = 0.5 * np.abs(current_prob - prior_prob).sum(axis=1)
        posterior_tv[~contiguous] = np.nan
        out[f"meta_aegmm_transition_posterior_tv_{suffix}"] = posterior_tv.astype(np.float32)
        current_cluster = panel["gmm_cluster_id"].to_numpy(dtype=np.float32, copy=False)
        prior_cluster = previous["gmm_cluster_id"].to_numpy(dtype=np.float32, copy=False)
        switched = (current_cluster != prior_cluster).astype(np.float32)
        switched[~contiguous] = np.nan
        out[f"meta_aegmm_transition_cluster_switch_{suffix}"] = switched
        for source, destination in (
            ("gmm_entropy", f"meta_aegmm_transition_entropy_delta_{suffix}"),
            ("gmm_posterior_max", f"meta_aegmm_transition_posterior_max_delta_{suffix}"),
            ("AE_reconstruction_error", f"meta_aegmm_transition_reconstruction_delta_{suffix}"),
            ("mahalanobis_distance", f"meta_aegmm_transition_mahal_delta_{suffix}"),
            ("gmm_ood_score", f"meta_aegmm_transition_ood_delta_{suffix}"),
        ):
            delta = (
                panel[source].to_numpy(dtype=np.float32, copy=False)
                - previous[source].to_numpy(dtype=np.float32, copy=False)
            )
            delta[~contiguous] = np.nan
            out[destination] = delta.astype(np.float32)
        current_latent = panel.loc[:, latents].to_numpy(dtype=np.float32, copy=False)
        prior_latent = previous.loc[:, latents].to_numpy(dtype=np.float32, copy=False)
        speed = np.sqrt(np.square(current_latent - prior_latent).sum(axis=1))
        speed[~contiguous] = np.nan
        out[f"meta_aegmm_transition_latent_speed_{suffix}"] = speed.astype(np.float32)
        # Total posterior variation loses the economically important direction
        # of a state change.  Persist per-component signed deltas so a local
        # side x archetype recognizer can distinguish entry into an exhaustion
        # component from an exit from it.  The frozen state has six active
        # components; inactive compatibility columns are never used here.
        for component, column in enumerate(probs[:AEGMM_COMPONENT_COUNT]):
            component_delta = current_prob[:, component] - prior_prob[:, component]
            component_delta[~contiguous] = np.nan
            out[f"meta_aegmm_transition_prob_{component}_delta_{suffix}"] = component_delta.astype(np.float32)

    for suffix in ("1h", "4h"):
        state_change = pd.to_numeric(out[f"meta_aegmm_transition_posterior_tv_{suffix}"], errors="coerce")
        breadth = (
            pd.DataFrame({"__ts__": out["__ts__"], "side_name": out["side_name"], "value": state_change.gt(0.10).astype(np.float32)})
            .groupby(["__ts__", "side_name"], observed=True)["value"]
            .mean()
            .rename(f"meta_aegmm_transition_market_breadth_{suffix}")
            .reset_index()
        )
        out = out.merge(breadth, on=["__ts__", "side_name"], how="left", validate="many_to_one")
    for source, destination in (
        ("meta_aegmm_transition_entropy_delta_1h", "meta_aegmm_transition_market_entropy_delta_1h"),
        ("meta_aegmm_transition_ood_delta_1h", "meta_aegmm_transition_market_ood_delta_1h"),
    ):
        market = (
            out.groupby(["__ts__", "side_name"], observed=True)[source]
            .median()
            .rename(destination)
            .reset_index()
        )
        out = out.merge(market, on=["__ts__", "side_name"], how="left", validate="many_to_one")
    # Cross-sectional entry/exit breadth is calculated from complete symbol
    # panels, never from candidates.  A 0.10 posterior move is intentionally
    # a descriptive threshold only; the downstream residual probability sees
    # continuous component deltas as well and is not a hard policy gate.
    for component in range(AEGMM_COMPONENT_COUNT):
        delta_col = f"meta_aegmm_transition_prob_{component}_delta_1h"
        for direction, reducer in (("enter", "gt"), ("exit", "lt")):
            values = pd.to_numeric(out[delta_col], errors="coerce")
            event = values.gt(0.10) if reducer == "gt" else values.lt(-0.10)
            destination = f"meta_aegmm_transition_prob_{component}_{direction}_breadth_1h"
            breadth = (
                pd.DataFrame(
                    {"__ts__": out["__ts__"], "side_name": out["side_name"], "value": event.astype(np.float32)}
                )
                .groupby(["__ts__", "side_name"], observed=True)["value"]
                .mean()
                .rename(destination)
                .reset_index()
            )
            out = out.merge(breadth, on=["__ts__", "side_name"], how="left", validate="many_to_one")
    out = _add_dominant_state_transition_features(panel, out, probs)
    return out.reindex(columns=["__ts__", "__symbol__", "side_name", *ALL_TRANSITION_FEATURES])


def _point_panel_for_month(
    *,
    data_root: Path,
    features_dir: Path,
    state: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    history_hours: int,
    symbol_batch_size: int,
) -> pd.DataFrame:
    feature_columns = [str(column) for column in state.get("feature_columns", [])]
    source_columns = required_source_columns(state)
    static_columns = list(dict.fromkeys([*feature_columns, *configured_causal_source_columns()]))
    static_columns = [column for column in static_columns if column != "side" and column not in source_columns]
    panel = full_static_source_panel(
        data_root=data_root,
        features_dir=features_dir,
        source_columns=static_columns,
        start=start - pd.Timedelta(hours=int(history_hours)),
        end=end,
        symbol_batch_size=int(symbol_batch_size),
    )
    enriched = materialize_live_source_regime_features(
        panel.sort_values(["__symbol__", "__ts__"], kind="stable").reset_index(drop=True),
        side="long",
        signal_bar_ts=None,
        required_columns=source_columns,
        overwrite_existing=True,
    )
    base = enriched.reindex(columns=["__ts__", "__symbol__", *feature_columns], fill_value=0.0)
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    parts: list[pd.DataFrame] = []
    for side_name, side_value in (("long", 1.0), ("short", -1.0)):
        x = base.reindex(columns=feature_columns, fill_value=0.0).copy()
        if "side" in x:
            x["side"] = np.float32(side_value)
        transformed = transform_ae_gmm_features(x, state, index=base.index)
        part = pd.concat([base.loc[:, ["__ts__", "__symbol__"]].reset_index(drop=True), transformed.reset_index(drop=True)], axis=1)
        part["side_name"] = side_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True, copy=False)


def materialize_candidate_aegmm_transitions(
    *,
    candidates_path: Path,
    features_dir: Path,
    ae_gmm_state_path: Path,
    out_path: Path,
    data_root: Path,
    history_hours: int = 6,
    symbol_batch_size: int = 24,
) -> dict[str, Any]:
    candidates = _normalise_candidates(pd.read_parquet(candidates_path))
    state = load_ae_gmm_state_artifact(ae_gmm_state_path)
    if str(state.get("temporal_feature_contract") or "") != "row_independent_v1":
        raise ValueError("transition materialization requires row_independent_v1 frozen AE/GMM state")
    candidates["__month__"] = candidates["__ts__"].dt.to_period("M").astype(str)
    outputs: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    for month, candidate_month in candidates.groupby("__month__", observed=True, sort=True):
        start = _utc(candidate_month["__ts__"].min())
        end = _utc(candidate_month["__ts__"].max())
        print(
            json.dumps(
                {
                    "event": "aegmm_transition_month_start",
                    "month": str(month),
                    "candidate_rows": int(len(candidate_month)),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                }
            ),
            flush=True,
        )
        point_panel = _point_panel_for_month(
            data_root=data_root,
            features_dir=features_dir,
            state=state,
            start=start,
            end=end,
            history_hours=int(history_hours),
            symbol_batch_size=int(symbol_batch_size),
        )
        transition = add_causal_aegmm_transition_features(point_panel)
        transition = transition.loc[(transition["__ts__"] >= start) & (transition["__ts__"] <= end)]
        merged = candidate_month.merge(
            transition,
            on=["__ts__", "__symbol__", "side_name"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        complete = merged.loc[:, list(ALL_TRANSITION_FEATURES)].notna().all(axis=1)
        reports[str(month)] = {
            "candidate_rows": int(len(merged)),
            "point_rows": int(len(point_panel)),
            "symbols": int(point_panel["__symbol__"].nunique()),
            "transition_complete_rate": float(complete.mean()),
            "transition_feature_nonzero": int((merged.loc[:, list(ALL_TRANSITION_FEATURES)].abs().sum(axis=0) > 1e-8).sum()),
        }
        print(
            json.dumps(
                {"event": "aegmm_transition_month_complete", "month": str(month), **reports[str(month)]},
                sort_keys=True,
            ),
            flush=True,
        )
        outputs.append(merged)
    out = pd.concat(outputs, ignore_index=True, copy=False).drop(columns=["__month__"], errors="ignore")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    manifest = {
        "generated_by": "materialize_frozen_aegmm_transition_context",
        "candidates": str(candidates_path),
        "features_dir": str(features_dir),
        "ae_gmm_state_path": str(ae_gmm_state_path),
        "state_temporal_feature_contract": str(state.get("temporal_feature_contract")),
        "history_hours": int(history_hours),
        "symbol_batch_size": int(symbol_batch_size),
        "transition_features": list(ALL_TRANSITION_FEATURES),
        "month_reports": reports,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--ae-gmm-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--symbol-batch-size", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(materialize_candidate_aegmm_transitions(
        candidates_path=args.candidates,
        features_dir=args.features_dir,
        ae_gmm_state_path=args.ae_gmm_state,
        out_path=args.out,
        data_root=args.data_root,
        history_hours=args.history_hours,
        symbol_batch_size=args.symbol_batch_size,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
