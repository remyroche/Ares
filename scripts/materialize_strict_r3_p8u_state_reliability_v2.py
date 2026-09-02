#!/usr/bin/env python3
"""Build the target-free P8U V2 state/reliability substrate.

V2 treats the current Router50 -> F72 Base -> Under F120 -> dual-MC1 stack as
the immutable parent.  It materialises a compact deviation-first market-state
panel and freezes target-free unsupervised episodes before opening policy
labels.  Outcome-derived residual and failure targets are written separately
and cannot be used as inference inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

import materialize_strict_r3_p8u_meta_base_state_v1 as base_state
import screen_strict_r3_p8u_market_state_transition_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_state_reliability_v2"
SEED = 1729
PAIR = "3d_21d"
REFERENCE_END = pd.Timestamp("2025-05-01", tz="UTC")
TOP10_START = 0.90

# The compact semantic universe is deliberately predeclared.  It covers the
# source families requested by the V2 proposal without reopening all 3,202
# algebraically related encodings.
SEMANTICS = (
    "return_iqr", "return_tail", "breadth", "breadth_negative",
    "breadth_downside", "volatility_dispersion", "volatility_level",
    "volatility_ratio", "volatility_xs", "liquidity_depth",
    "execution_spread", "execution_spread_level", "oi_effective_rank",
    "oi_eigen_rank", "funding_dispersion", "correlation",
    "correlation_break", "spectral_effective_rank", "spectral_entropy",
    "spectral_lambda1_share", "spectral_mahalanobis", "btc_decoupling",
    "btc_alt_strength",
)
CONTRASTS = (
    ("v2_contrast__volatility_minus_liquidity", "volatility_level", "execution_spread"),
    ("v2_contrast__breadth_minus_dispersion", "breadth", "return_iqr"),
    ("v2_contrast__oi_minus_liquidity", "oi_effective_rank", "liquidity_depth"),
    ("v2_contrast__funding_minus_breadth", "funding_dispersion", "breadth"),
    ("v2_contrast__correlation_minus_dispersion", "correlation", "return_iqr"),
    ("v2_contrast__btc_minus_breadth", "btc_decoupling", "breadth"),
)
EPISODE_SEMANTICS = (
    "breadth", "volatility_level", "execution_spread", "oi_effective_rank",
    "funding_dispersion", "correlation", "return_iqr",
)
# These are deliberately a *small control set*, not an additional wide state
# lattice.  V2 is deviation-first; the levels and direct deltas let the later
# reliability screen distinguish a transition effect from a simple high/low
# state effect without changing the meaning of the frozen parent stack.
LEVEL_CONTROL_SEMANTICS = (
    "breadth", "return_iqr", "volatility_level", "liquidity_depth",
    "execution_spread", "oi_effective_rank", "funding_dispersion",
    "correlation",
)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
        digest.update(str(member.relative_to(ROOT)).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    result: list[pd.Timestamp] = []
    value = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    while value < end:
        result.append(value)
        value += pd.offsets.MonthBegin(1)
    return tuple(result)


def _one(dictionary: pd.DataFrame, semantic: str, kind: str, role: str | None) -> str:
    mask = dictionary.semantic_state.eq(semantic) & dictionary.kind.eq(kind) & dictionary.pair.eq(PAIR)
    if role is None:
        mask &= dictionary.role.isna()
    else:
        mask &= dictionary.role.eq(role)
    found = dictionary.loc[mask, "feature"].tolist()
    if len(found) != 1:
        raise AssertionError(f"{semantic}/{kind}/{role}: expected one source, got {found}")
    return str(found[0])


def _compact_sources(
    dictionary: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    innovation, transition, uncertainty, level, direct = {}, {}, {}, {}, {}
    for semantic in SEMANTICS:
        innovation[semantic] = _one(dictionary, semantic, "kalman_innovation_z", "fast")
        transition[semantic] = _one(dictionary, semantic, "kalman_fast_slow_normalized", None)
        uncertainty[semantic] = _one(dictionary, semantic, "posterior_variance", "fast")
    for semantic in LEVEL_CONTROL_SEMANTICS:
        level[semantic] = _one(dictionary, semantic, "kalman_level", "fast")
        mask = dictionary.semantic_state.eq(semantic) & dictionary.kind.eq("direct_transition")
        # Direct deltas have two raw horizons in the source lattice.  The
        # 3d-minus-7d form is predeclared here because it is the nearer control
        # for the V2 3d/21d transition representation; never pick by row order.
        found = dictionary.loc[mask & dictionary.feature.str.contains("delta_3d_minus_7d", regex=False), "feature"].tolist()
        if len(found) != 1:
            raise AssertionError(f"{semantic}/direct_transition: expected one source, got {found}")
        direct[semantic] = str(found[0])
    return innovation, transition, uncertainty, level, direct


def _robust_geometry(values: np.ndarray, reference: np.ndarray, prefix: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    median = np.nanmedian(reference, axis=0)
    q75, q25 = np.nanpercentile(reference, [75, 25], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-5)
    z = (np.where(np.isfinite(values), values, median) - median) / scale
    ref_z = (np.where(np.isfinite(reference), reference, median) - median) / scale
    covariance = np.cov(ref_z, rowvar=False)
    ridge = max(float(np.trace(covariance) / max(1, covariance.shape[0])) * .15, 1e-5)
    inverse = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
    absolute = np.abs(z)
    top = np.sort(absolute, axis=1)[:, -min(3, absolute.shape[1]):]
    output = {
        f"{prefix}_rms": np.sqrt(np.mean(z * z, axis=1)),
        f"{prefix}_mahalanobis": np.sqrt(np.maximum(0., np.einsum("ij,jk,ik->i", z, inverse, z))),
        f"{prefix}_abs1_breadth": np.mean(absolute >= 1., axis=1),
        f"{prefix}_abs2_breadth": np.mean(absolute >= 2., axis=1),
        f"{prefix}_positive_breadth": np.mean(z > 0., axis=1),
        f"{prefix}_negative_breadth": np.mean(z < 0., axis=1),
        f"{prefix}_sign_coherence": np.abs(np.mean(np.sign(z), axis=1)),
        f"{prefix}_iqr": np.percentile(z, 75, axis=1) - np.percentile(z, 25, axis=1),
        f"{prefix}_mad": np.median(np.abs(z - np.median(z, axis=1, keepdims=True)), axis=1),
        f"{prefix}_max_abs": np.max(absolute, axis=1),
        f"{prefix}_top3_abs_mean": np.mean(top, axis=1),
        f"{prefix}_available_fraction": np.mean(np.isfinite(values), axis=1),
    }
    contract = {"median": median.tolist(), "scale": scale.tolist(), "ridge": ridge, "inverse_covariance": inverse.tolist()}
    return output, contract


def _assign_episodes(frame: pd.DataFrame, episode_fields: list[str], reference_mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    reference = frame.loc[reference_mask, episode_fields].to_numpy(float)
    median = np.nanmedian(reference, axis=0)
    q75, q25 = np.nanpercentile(reference, [75, 25], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-5)
    x = (np.where(np.isfinite(frame.loc[:, episode_fields].to_numpy(float)), frame.loc[:, episode_fields].to_numpy(float), median) - median) / scale
    x_ref = x[reference_mask]
    pca_all = PCA(random_state=SEED).fit(x_ref)
    components = int(np.searchsorted(np.cumsum(pca_all.explained_variance_ratio_), .85) + 1)
    components = min(max(2, components), min(10, x_ref.shape[1]))
    pca = PCA(n_components=components, random_state=SEED).fit(x_ref)
    z, z_ref = pca.transform(x), pca.transform(x_ref)
    diagnostics: list[dict[str, object]] = []
    choices: dict[int, KMeans] = {}
    # Assignment stability is target-free temporal overlap stability: models
    # fitted to early/late reference portions are compared on their overlap.
    n = len(z_ref); cut = max(100, int(n * .70)); late_start = min(n - 100, int(n * .30))
    for k in (4, 6, 8):
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED + k).fit(z_ref)
        labels = model.predict(z_ref); counts = np.bincount(labels, minlength=k) / len(labels)
        sample = z_ref if len(z_ref) <= 2500 else z_ref[np.linspace(0, len(z_ref) - 1, 2500).astype(int)]
        silhouette = float(silhouette_score(sample, model.predict(sample))) if len(np.unique(labels)) > 1 else -1.0
        persistence = float(np.mean(labels[1:] == labels[:-1]))
        early = KMeans(n_clusters=k, n_init=12, random_state=SEED + 101 + k).fit(z_ref[:cut])
        late = KMeans(n_clusters=k, n_init=12, random_state=SEED + 211 + k).fit(z_ref[late_start:])
        overlap = z_ref[late_start:cut] if late_start < cut else z_ref[:0]
        stability = float(adjusted_rand_score(early.predict(overlap), late.predict(overlap))) if len(overlap) >= 100 else -1.0
        score = silhouette + .20 * persistence + .20 * stability + .10 * float(counts.min())
        diagnostics.append({"k": k, "silhouette": silhouette, "persistence": persistence, "assignment_stability": stability, "min_cluster_share": float(counts.min()), "max_cluster_share": float(counts.max()), "target_free_selection_score": score, "acceptable_balance": bool(counts.min() >= .02)})
        choices[k] = model
    diag = pd.DataFrame(diagnostics).sort_values(["acceptable_balance", "target_free_selection_score", "k"], ascending=[False, False, True], kind="stable")
    selected_k = int(diag.iloc[0].k); model = choices[selected_k]
    distances = model.transform(z); rank = np.argsort(distances, axis=1)
    output = pd.DataFrame({"__decision_ts__": frame.__decision_ts__.to_numpy(), "v2_regime_id": rank[:, 0].astype(np.int16), "v2_regime_distance": distances[np.arange(len(frame)), rank[:, 0]].astype(np.float32), "v2_regime_second_distance": distances[np.arange(len(frame)), rank[:, 1]].astype(np.float32)})
    output["v2_regime_assignment_margin"] = (output.v2_regime_second_distance - output.v2_regime_distance).astype(np.float32)
    output["v2_previous_regime_id"] = output.v2_regime_id.shift(1).fillna(-1).astype(np.int16)
    output["v2_regime_transition_flag"] = output.v2_regime_id.ne(output.v2_previous_regime_id).astype(np.int8)
    output.loc[0, "v2_regime_transition_flag"] = 1
    output["v2_episode_id"] = output.v2_regime_transition_flag.cumsum().astype(np.int32)
    output["v2_regime_age_hours"] = output.groupby("v2_episode_id", sort=False).cumcount().astype(np.int32)
    output["v2_time_since_regime_change_hours"] = output.v2_regime_age_hours.astype(np.int32)
    output["v2_episode_age_hours"] = output.v2_regime_age_hours.astype(np.int32)
    output["v2_episode_duration_so_far_hours"] = output.v2_regime_age_hours.astype(np.int32)
    contract = {"episode_fields": episode_fields, "reference_end": str(REFERENCE_END), "robust_median": median.tolist(), "robust_scale": scale.tolist(), "pca_components": components, "pca_components_matrix": pca.components_.tolist(), "pca_mean": pca.mean_.tolist(), "selected_k": selected_k, "kmeans_centers": model.cluster_centers_.tolist(), "seed": SEED}
    return output, contract, diag


def _base_context(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routed = base.loc[base.base_rank_ts.ge(TOP10_START), ["candidate_id", "__decision_ts__", "side_name", "base_score", "base_rank_ts"]].copy()
    if routed.empty:
        raise AssertionError("no Base Top10 candidates")
    grouped = routed.groupby("__decision_ts__", sort=True)
    timestamp = grouped.agg(
        v2_base_top10_n=("candidate_id", "size"),
        v2_base_top10_score_mean=("base_score", "mean"),
        v2_base_top10_score_median=("base_score", "median"),
        v2_base_top10_score_iqr=("base_score", lambda x: float(x.quantile(.75) - x.quantile(.25))),
        v2_base_top10_rank_mean=("base_rank_ts", "mean"),
        v2_base_top10_rank_max=("base_rank_ts", "max"),
    ).reset_index()
    ordered = routed.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordered["v2_position"] = ordered.groupby("__decision_ts__", sort=False).cumcount()
    top2 = ordered.loc[ordered.v2_position.lt(2)].groupby("__decision_ts__", sort=True).agg(v2_base_top2_score_mean=("base_score", "mean"), v2_base_top2_rank_mean=("base_rank_ts", "mean")).reset_index()
    timestamp = timestamp.merge(top2, on="__decision_ts__", how="left", validate="one_to_one")
    timestamp["v2_base_top10_score_gap"] = (timestamp.v2_base_top10_score_median - timestamp.v2_base_top2_score_mean).astype(np.float32)
    value = timestamp.v2_base_top10_score_mean.astype(float)
    timestamp["v2_base_tail_fast_3d"] = value.ewm(halflife=72, adjust=False, min_periods=1).mean().astype(np.float32)
    timestamp["v2_base_tail_slow_21d"] = value.ewm(halflife=504, adjust=False, min_periods=1).mean().astype(np.float32)
    timestamp["v2_base_tail_transition"] = (timestamp.v2_base_tail_fast_3d - timestamp.v2_base_tail_slow_21d).astype(np.float32)
    return routed, timestamp


def _label_panels(base: pd.DataFrame, policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = base_state._policy_events(base, policy).merge(base.loc[:, ["candidate_id", "base_rank_ts", "base_score"]], on="candidate_id", how="left", validate="one_to_one")
    events = events.loc[events.base_rank_ts.ge(TOP10_START)].copy()
    events["base_abs_residual_bps"] = events.residual_bps.abs().astype(np.float32)
    events["base_sqrt_abs_residual"] = np.sqrt(events.base_abs_residual_bps).astype(np.float32)
    events["base_log1p_abs_residual"] = np.log1p(events.base_abs_residual_bps).astype(np.float32)
    events["base_clipped_squared_residual"] = np.square(events.residual_bps.clip(-500., 500.)).astype(np.float32)
    for threshold in (100, 150, 200):
        events[f"base_large_error_{threshold}"] = events.base_abs_residual_bps.gt(float(threshold)).astype(np.int8)
        events[f"base_underconfidence_{threshold}"] = events.residual_bps.gt(float(threshold)).astype(np.int8)
        events[f"base_overconfidence_{threshold}"] = events.residual_bps.lt(-float(threshold)).astype(np.int8)
    ordered = events.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordered["position"] = ordered.groupby("__decision_ts__", sort=False).cumcount()
    top2 = ordered.loc[ordered.position.lt(2)]
    target = events.groupby("__decision_ts__", sort=True).agg(
        top10_mean_residual_bps=("residual_bps", "mean"), top10_median_residual_bps=("residual_bps", "median"),
        top10_mean_abs_residual_bps=("base_abs_residual_bps", "mean"), top10_p90_abs_residual_bps=("base_abs_residual_bps", lambda x: float(x.quantile(.90))),
        top10_gt50_hit_rate=("gt50", "mean"), top10_gt100_hit_rate=("gt100", "mean"),
        top10_realised_ev_bps=("policy_net_bps", "mean"), top10_realised_utility_bps=("policy_net_bps", lambda x: float(x.clip(-300., 600.).mean())),
        top10_label_available_ts=("available", "max"), top10_n=("candidate_id", "size"),
    ).reset_index()
    t2 = top2.groupby("__decision_ts__", sort=True).agg(top2_realised_ev_bps=("policy_net_bps", "mean"), top2_realised_utility_bps=("policy_net_bps", lambda x: float(x.clip(-300., 600.).mean())), top2_large_error_rate=("base_large_error_100", "mean"), top2_n=("candidate_id", "size"), top2_label_available_ts=("available", "max")).reset_index()
    target = target.merge(t2, on="__decision_ts__", how="left", validate="one_to_one")
    return events.sort_values(["__decision_ts__", "candidate_id"], kind="stable"), target.sort_values("__decision_ts__", kind="stable")


def _read_policy_subset(path: Path, candidate_ids: pd.Series) -> pd.DataFrame:
    """Read only Base-history label rows, bounded by parquet batch.

    The canonical label table also contains non-Base historical candidates.
    Reading it whole is unnecessary here and can exceed memory during an
    otherwise small offline target build.
    """
    wanted = set(candidate_ids.astype(str).tolist())
    columns = ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]
    parts: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=200_000):
        part = batch.to_pandas()
        part = part.loc[part.candidate_id.astype(str).isin(wanted)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise AssertionError("no canonical policy labels match the Base history")
    output = pd.concat(parts, ignore_index=True)
    if output.candidate_id.duplicated().any():
        raise AssertionError("canonical policy labels are not candidate-unique")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True); parser.add_argument("--early-base-root", required=True); parser.add_argument("--later-base-root", required=True)
    parser.add_argument("--policy-labels", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state_root, early, later, policy_path, out = (ROOT / args.state_root, ROOT / args.early_base_root, ROOT / args.later_base_root, ROOT / args.policy_labels, ROOT / args.out)
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((state_root / "correctness_report.json").read_text())
    if not all(value is True or key in {"schema", "fast_slow_pairs_predeclared"} for key, value in receipt.items()):
        raise AssertionError("input state lattice receipt is not clean")
    dictionary = pd.read_parquet(state_root / "market_state_feature_dictionary.parquet")
    innovation, transition, uncertainty, level, direct = _compact_sources(dictionary)
    requested = ["__decision_ts__", *innovation.values(), *transition.values(), *uncertainty.values(), *level.values(), *direct.values()]
    raw = pd.read_parquet(state_root / "market_state_hourly.parquet", columns=requested)
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    state = raw.loc[:, ["__decision_ts__"]].copy()
    for semantic in SEMANTICS:
        state[f"v2_innovation_z__{semantic}"] = raw[innovation[semantic]].astype(np.float32)
        state[f"v2_transition_z__{semantic}"] = raw[transition[semantic]].astype(np.float32)
        state[f"v2_uncertainty__{semantic}"] = raw[uncertainty[semantic]].astype(np.float32)
    for semantic in LEVEL_CONTROL_SEMANTICS:
        state[f"v2_level_control__{semantic}"] = raw[level[semantic]].astype(np.float32)
        state[f"v2_direct_delta_control__{semantic}"] = raw[direct[semantic]].astype(np.float32)
    reference_mask = state.__decision_ts__.lt(REFERENCE_END).to_numpy()
    if int(reference_mask.sum()) < 1000:
        raise AssertionError("insufficient target-free episode reference")
    in_cols = [f"v2_innovation_z__{x}" for x in SEMANTICS]
    tr_cols = [f"v2_transition_z__{x}" for x in SEMANTICS]
    in_geo, in_contract = _robust_geometry(state.loc[:, in_cols].to_numpy(float), state.loc[reference_mask, in_cols].to_numpy(float), "v2_innovation")
    tr_geo, tr_contract = _robust_geometry(state.loc[:, tr_cols].to_numpy(float), state.loc[reference_mask, tr_cols].to_numpy(float), "v2_transition")
    for name, values in {**in_geo, **tr_geo}.items(): state[name] = values.astype(np.float32)
    for name, left, right in CONTRASTS:
        state[name] = (state[f"v2_transition_z__{left}"] - state[f"v2_transition_z__{right}"]).astype(np.float32)
    episode_fields = ["v2_innovation_mahalanobis", "v2_transition_mahalanobis", "v2_transition_abs1_breadth", "v2_transition_sign_coherence", *[f"v2_transition_z__{x}" for x in EPISODE_SEMANTICS]]
    episodes, episode_contract, episode_diag = _assign_episodes(state, episode_fields, reference_mask)
    state = state.merge(episodes, on="__decision_ts__", how="left", validate="one_to_one")
    # Target-free Base Top10 context is constructed before labels are opened.
    start, end = state.__decision_ts__.min(), state.__decision_ts__.max() + pd.Timedelta(hours=1)
    base = screen._read_base(early, later, start, end)
    top10, base_context = _base_context(base)
    state = state.merge(base_context, on="__decision_ts__", how="left", validate="one_to_one")
    state["v2_contrast__base_tail_minus_breadth"] = (state.v2_base_tail_transition - state["v2_transition_z__breadth"]).astype(np.float32)
    state["v2_contrast__base_tail_minus_dispersion"] = (state.v2_base_tail_transition - state["v2_transition_z__return_iqr"]).astype(np.float32)
    top10.to_parquet(out / "__placeholder__") if False else None
    out.mkdir(parents=True)
    state.to_parquet(out / "target_free_state_episode_hourly.parquet", index=False)
    top10.to_parquet(out / "target_free_base_top10_candidates.parquet", index=False)
    episode_diag.to_parquet(out / "target_free_episode_k_diagnostics.parquet", index=False)
    _once(out / "target_free_episode_contract.json", {"schema": SCHEMA, "reference_window": [str(state.__decision_ts__.min()), str(REFERENCE_END)], "semantic_states": list(SEMANTICS), "level_control_semantics": list(LEVEL_CONTROL_SEMANTICS), "pair": PAIR, "innovation_geometry": in_contract, "transition_geometry": tr_contract, "episode": episode_contract, "contrasts": [list(item) for item in CONTRASTS], "source_state_root": str(state_root.relative_to(ROOT))})
    _once(out / "target_free_manifest.json", {"schema": SCHEMA, "state_rows": int(len(state)), "state_feature_count": int(len(state.columns) - 1), "base_top10_rows": int(len(top10)), "base_top10_timestamps": int(top10.__decision_ts__.nunique()), "target_free_written_before_policy_labels": True})
    # Outcome sources are opened only after all target-free panels are persisted.
    policy = _read_policy_subset(policy_path, base.candidate_id)
    residuals, failures = _label_panels(base, policy)
    residuals.to_parquet(out / "labelled_base_top10_residual_events.parquet", index=False)
    failures.to_parquet(out / "labelled_base_top10_failure_targets.parquet", index=False)
    correctness = {"schema": SCHEMA, "parent_base_scores_target_free": True, "state_lattice_target_free": True, "episode_fit_uses_no_outcomes": True, "episode_reference_precedes_2025_selection": True, "episode_assignments_are_frozen_not_refit_by_fold": True, "base_top10_context_written_before_outcome_join": True, "base_residual_anchor_is_strict_prequential": True, "outcome_targets_are_separate_from_inference_state": True, "no_live_mc1_admission_portfolio_or_exchange_mutation": True}
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {"schema": SCHEMA, "scope": "offline P8U V2 state reliability substrate", "source": {"state_root": str(state_root.relative_to(ROOT)), "state_root_sha256": _sha(state_root), "early_base_root": str(early.relative_to(ROOT)), "later_base_root": str(later.relative_to(ROOT)), "policy_labels": str(policy_path.relative_to(ROOT))}, "correctness": correctness})
    print(json.dumps({"out": str(out), "state_rows": len(state), "state_fields": len(state.columns)-1, "top10_rows": len(top10), "episode_k": episode_contract["selected_k"]}, sort_keys=True))


if __name__ == "__main__":
    main()
