#!/usr/bin/env python3
"""Materialize a non-pooled Jan--Jul 2022 inverse-PI regime supplement.

The inverse-PI candidate lineage is intentionally distinct from the later
frozen perpetual-futures population.  This runner uses leave-month-out OOF
research only (not walk-forward), causal features available at each signal
hour, and exact 12-hour labels.  Its GMM identifiers are local to this
supplement and must never be treated as later PF taxonomy identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "data_perp/artifacts/jan_jul_2022_inverse_pi_causal_features_20260730_v3"
LABEL_ROOT = ROOT / "data_perp/artifacts/jan_jul_2022_inverse_pi_causal_multitask_labels_20260730_v2"
SCORE_ROOT = ROOT / "data_perp/artifacts/jan_jul_2022_inverse_pi_direct_utility_multitask_ablation_20260730_v2"
LATER_LEDGER = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
EXT_FEATURE_ROOT = ROOT / "data_perp/artifacts/aug2022_inverse_pi_causal_features_20260730_v1"
EXT_LABEL_ROOT = ROOT / "data_perp/artifacts/aug2022_inverse_pi_causal_multitask_labels_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/early_2022_inverse_pi_regime_supplement_20260730_v3"

STATE_FEATURES = (
    "market_median_ret_1h", "market_median_ret_4h", "market_median_ret_24h",
    "market_dispersion_1h", "market_dispersion_4h", "market_median_rv_24h",
    "market_median_atr_fraction", "market_negative_breadth_4h",
    "market_negative_breadth_24h", "market_average_pair_corr_24h",
    "btc_minus_alt_median_ret_24h", "rv_6h", "rv_24h", "rv_72h",
    "atr_fraction_14h", "range_24h_fraction", "path_efficiency_24h",
)
TRANSITION_FEATURE_PREFIX = "transition_raw__"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return [_safe(item) for item in value.tolist()]
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1) / np.log(probabilities.shape[1])


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    return float(left.loc[valid].corr(right.loc[valid], method="spearman")) if valid.sum() >= 3 else float("nan")


def _component_alignment(model: GaussianMixture, scaler: StandardScaler) -> tuple[dict[int, int], dict[int, str], pd.DataFrame]:
    """Map arbitrary GMM components to local deterministic semantic IDs."""
    means = model.means_ * scaler.scale_ + scaler.mean_
    index = {name: position for position, name in enumerate(STATE_FEATURES)}
    risk = (
        means[:, index["market_median_rv_24h"]]
        + means[:, index["market_negative_breadth_24h"]]
        + means[:, index["market_average_pair_corr_24h"]]
    )
    trend = means[:, index["market_median_ret_24h"]]
    # Sorting on observable component signatures is deterministic and is the
    # only alignment used across leave-month-out fits.
    ordered = sorted(range(model.n_components), key=lambda item: (float(risk[item]), float(trend[item]), int(item)))
    mapping = {component: position for position, component in enumerate(ordered)}
    median_risk = float(np.median(risk)); median_abs_trend = float(np.median(np.abs(trend)))
    semantic: dict[int, str] = {}
    profile: list[dict[str, Any]] = []
    for component in range(model.n_components):
        local_id = mapping[component]
        if risk[component] >= median_risk:
            descriptor = "high_risk_down" if trend[component] < 0 else "high_risk_up"
        else:
            descriptor = "calm_trend" if abs(trend[component]) >= median_abs_trend else "calm_range"
        # The ordinal ID remains identical across every fold.  The descriptor is
        # evidence about that fold's component profile, not a mutable taxonomy
        # suffix that would make the aligned ID appear to change meaning.
        semantic[component] = f"early22_s{local_id}"
        profile.append({"raw_component": component, "local_state_id": local_id, "local_state": semantic[component], "semantic_descriptor": descriptor, "risk_signature": float(risk[component]), "trend_signature": float(trend[component])})
    return mapping, semantic, pd.DataFrame(profile).sort_values("local_state_id", kind="stable")


def materialize_state_oof(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []; profiles: list[pd.DataFrame] = []
    months = sorted(hourly.month.unique())
    for held_month in months:
        train = hourly.loc[hourly.month.ne(held_month)]; test = hourly.loc[hourly.month.eq(held_month)].copy()
        imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(train.loc[:, STATE_FEATURES]))
        x_test = scaler.transform(imputer.transform(test.loc[:, STATE_FEATURES]))
        model = GaussianMixture(n_components=4, covariance_type="full", reg_covar=1e-5, n_init=4, random_state=1729).fit(x_train)
        mapping, semantic, profile = _component_alignment(model, scaler)
        probability = model.predict_proba(x_test); raw_component = probability.argmax(axis=1)
        aligned = np.empty_like(probability)
        for component, local_id in mapping.items(): aligned[:, local_id] = probability[:, component]
        test["held_out_month"] = held_month; test["gmm_raw_component"] = raw_component
        test["local_state_id"] = [mapping[int(item)] for item in raw_component]
        test["local_state"] = [semantic[int(item)] for item in raw_component]
        test["state_max_probability"] = aligned.max(axis=1); test["state_entropy"] = _entropy(aligned)
        ordered = np.sort(aligned, axis=1); test["state_top2_margin"] = ordered[:, -1] - ordered[:, -2]
        for component in range(4): test[f"p_local_state_{component}"] = aligned[:, component]
        records.append(test); profiles.append(profile.assign(held_out_month=held_month, train_hours=len(train), test_hours=len(test)))
    result = pd.concat(records, ignore_index=True).sort_values("__ts__", kind="stable").reset_index(drop=True)
    return result, pd.concat(profiles, ignore_index=True)


def materialize_transition_oof(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = states.sort_values("__ts__", kind="stable").copy()
    lookup = work.set_index("__ts__")["local_state"]
    future_state = work["__ts__"].add(pd.Timedelta(hours=3)).map(lookup)
    work["target_transition_within_3h"] = np.where(future_state.notna(), future_state.ne(work.local_state).astype(float), np.nan)
    transition_features = [column for column in work.columns if column.startswith("transition_raw__")]
    months = sorted(work.month.unique()); records: list[pd.DataFrame] = []
    for held_month in months:
        train = work.loc[work.month.ne(held_month) & work.target_transition_within_3h.notna()]
        test = work.loc[work.month.eq(held_month)].copy()
        pipeline = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=1729))
        pipeline.fit(train.loc[:, transition_features], train.target_transition_within_3h.astype(int))
        test["p_transition_within_3h"] = pipeline.predict_proba(test.loc[:, transition_features])[:, 1]
        test["transition_train_hours"] = len(train); test["transition_held_out_month"] = held_month
        records.append(test)
    result = pd.concat(records, ignore_index=True).sort_values("__ts__", kind="stable").reset_index(drop=True)
    prior = result.p_transition_within_3h.shift(1).fillna(0.0); recent = result.p_transition_within_3h.rolling(3, min_periods=1).max().shift(1).fillna(0.0)
    result["transition_phase"] = np.select(
        [result.p_transition_within_3h.ge(0.60) & prior.lt(0.40), result.p_transition_within_3h.ge(0.60), result.p_transition_within_3h.lt(0.40) & recent.ge(0.60)],
        ["onset", "active", "decay"], default="stable",
    )
    result["transition_probability_entropy"] = -(np.clip(result.p_transition_within_3h, 1e-12, 1 - 1e-12) * np.log(np.clip(result.p_transition_within_3h, 1e-12, 1 - 1e-12)) + (1 - np.clip(result.p_transition_within_3h, 1e-12, 1 - 1e-12)) * np.log(1 - np.clip(result.p_transition_within_3h, 1e-12, 1 - 1e-12))) / np.log(2)
    quality: list[dict[str, Any]] = []
    for month, local in result.loc[result.target_transition_within_3h.notna()].groupby("month", observed=True):
        target = local.target_transition_within_3h.astype(int); pred = local.p_transition_within_3h
        quality.append({"month": month, "rows": len(local), "prevalence": float(target.mean()), "roc_auc": float(roc_auc_score(target, pred)) if target.nunique() == 2 else np.nan, "average_precision": float(average_precision_score(target, pred)) if target.nunique() == 2 else np.nan, "brier": float(brier_score_loss(target, pred))})
    return result, pd.DataFrame(quality)


def _load_candidates(root: Path, extension_root: Path | None = None) -> pd.DataFrame:
    roots = [root] + ([extension_root] if extension_root is not None else [])
    chunks = [pd.read_parquet(path) for item in roots for path in sorted((item / "candidate_shards").glob("*.parquet"))]
    frame = pd.concat(chunks, ignore_index=True); frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True); frame["month"] = frame.__ts__.dt.strftime("%Y-%m")
    return frame


def _performance(joined: pd.DataFrame) -> pd.DataFrame:
    # Exactly one global (cross-side) top 10% selection per calendar month.
    joined = joined.copy(); joined["selected_global_top10_raw_oof"] = False
    for _, indices in joined.groupby("month", observed=True).groups.items():
        ranked = joined.loc[indices].sort_values(["raw_direct_score", "candidate_id"], ascending=[False, True], kind="stable")
        joined.loc[ranked.index[:max(1, int(np.ceil(len(ranked) * 0.10)))], "selected_global_top10_raw_oof"] = True
    rows: list[dict[str, Any]] = []
    groups = ["month", "side_name", "local_state", "transition_phase"]
    for keys, local in joined.groupby(groups, observed=True):
        selected = local.loc[local.selected_global_top10_raw_oof]
        values = dict(zip(groups, keys)); values.update({
            "candidate_rows": len(local), "selected_rows": len(selected), "state_support_hours": int(local.__ts__.nunique()),
            "alpha_rank_ic_opportunity": _rank_ic(local.raw_direct_score, local["__opportunity_occurred_12h__"]),
            "execution_net_rank_ic": _rank_ic(local.raw_direct_score, local.execution_net_ev_12h),
            "mean_gross_bps_selected": float(selected.execution_gross_ev_12h.mean() * 1e4) if len(selected) else np.nan,
            "mean_net_bps_selected": float(selected.execution_net_ev_12h.mean() * 1e4) if len(selected) else np.nan,
            "mean_gross_bps_all": float(local.execution_gross_ev_12h.mean() * 1e4), "mean_net_bps_all": float(local.execution_net_ev_12h.mean() * 1e4),
            "positive_net_rate_selected": float((selected.execution_net_ev_12h > 0).mean()) if len(selected) else np.nan,
            "mean_peak_mfe_atr_selected": float(selected["__peak_mfe_atr_12h__"].mean()) if len(selected) else np.nan,
            "mean_mae_return_selected": float(selected.execution_mae_return_12h.mean()) if len(selected) else np.nan,
            "mean_future_slope_atr_per_hour_selected": float(selected["__future_slope_atr_per_hour_12h__"].mean()) if len(selected) else np.nan,
            "mean_state_probability": float(local.state_max_probability.mean()), "mean_state_entropy": float(local.state_entropy.mean()),
            "mean_transition_probability": float(local.p_transition_within_3h.mean()), "mean_transition_entropy": float(local.transition_probability_entropy.mean()),
        }); rows.append(values)
    return pd.DataFrame(rows).sort_values(groups, kind="stable").reset_index(drop=True)


def _cross_lineage_bridge() -> pd.DataFrame:
    # These are feature-family correspondences only.  Different product,
    # sampling and transform contracts prohibit taxonomy identity matching.
    return pd.DataFrame([
        {"early22_observable": "market_median_rv_24h", "later_observable_family": "volatility/regime change", "comparability": "family_only", "taxonomy_alignment_allowed": False, "reason": "No identically defined later ledger observable."},
        {"early22_observable": "market_negative_breadth_4h/24h", "later_observable_family": "negative_breadth_pct/downside_breadth_intensity", "comparability": "window_and_universe_mismatch", "taxonomy_alignment_allowed": False, "reason": "Both measure breadth stress but use distinct aggregation windows/universes."},
        {"early22_observable": "market_average_pair_corr_24h", "later_observable_family": "correlation heterogeneity/concentration", "comparability": "family_only", "taxonomy_alignment_allowed": False, "reason": "Later states use transformed correlation summaries, not the same raw statistic."},
        {"early22_observable": "market_dispersion_1h/4h", "later_observable_family": "breadth_dispersion/correlation_breakdown_dispersion", "comparability": "family_only", "taxonomy_alignment_allowed": False, "reason": "The observable family overlaps but definitions and candidate population do not."},
        {"early22_observable": "btc_minus_alt_median_ret_24h", "later_observable_family": "btc_resilience_alt_weakness", "comparability": "family_only", "taxonomy_alignment_allowed": False, "reason": "Directional BTC-versus-alt relation is conceptually comparable only."},
    ])


def run(*, feature_root: Path, label_root: Path, score_root: Path, output_dir: Path, extension_feature_root: Path | None = None, extension_label_root: Path | None = None) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    candidates = _load_candidates(feature_root, extension_feature_root)
    primitive_columns = list(STATE_FEATURES) + [column for column in candidates.columns if column.startswith(TRANSITION_FEATURE_PREFIX)]
    hourly = candidates.groupby("__ts__", observed=True)[primitive_columns].median().reset_index(); hourly["month"] = hourly.__ts__.dt.strftime("%Y-%m")
    states, profiles = materialize_state_oof(hourly)
    transitions, transition_quality = materialize_transition_oof(states)
    labels = pd.read_parquet(label_root / "joined_multitask_labels.parquet")
    if extension_label_root is not None:
        labels = pd.concat([labels, pd.read_parquet(extension_label_root / "joined_multitask_labels.parquet")], ignore_index=True)
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    scores = pd.read_parquet(score_root / "oof_scores.parquet")
    scores = scores.loc[scores.arm.eq("market_transition_interactions__economic_multitask"), ["candidate_id", "raw_direct_score", "arm", "mapping_status"]].copy()
    candidate_labels = candidates.merge(labels, on=["__ts__", "__symbol__", "side_name"], validate="one_to_one")
    missing_score = candidate_labels.loc[~candidate_labels.candidate_id.isin(scores.candidate_id)].copy()
    if len(missing_score):
        numeric = [column for column in candidates.columns if column in set(STATE_FEATURES) or column.startswith(TRANSITION_FEATURE_PREFIX) or column in {"ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_72h", "ret_168h", "volume_z_24h", "volume_z_72h", "jump_intensity_24h"}]
        train = candidate_labels.loc[~candidate_labels.candidate_id.isin(missing_score.candidate_id)]
        model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, l2_regularization=1.0, random_state=1729))
        model.fit(train.loc[:, numeric], train.execution_net_ev_12h)
        extra = missing_score.loc[:, ["candidate_id"]].copy(); extra["raw_direct_score"] = model.predict(missing_score.loc[:, numeric]); extra["arm"] = "august_leave_month_out_direct_hgb"; extra["mapping_status"] = "raw_oof_no_mapping"; scores = pd.concat([scores, extra], ignore_index=True)
    joined = candidate_labels.merge(scores, on="candidate_id", validate="one_to_one").merge(transitions.loc[:, ["__ts__", "local_state", "local_state_id", "state_max_probability", "state_entropy", "state_top2_margin", "p_transition_within_3h", "transition_phase", "transition_probability_entropy", "target_transition_within_3h", "transition_train_hours"]], on="__ts__", validate="many_to_one")
    performance = _performance(joined)
    state_support = transitions.groupby(["month", "local_state", "transition_phase"], observed=True).agg(hours=("__ts__", "size"), mean_state_probability=("state_max_probability", "mean"), mean_state_entropy=("state_entropy", "mean"), mean_transition_probability=("p_transition_within_3h", "mean"), transition_events=("target_transition_within_3h", "sum")).reset_index()
    bridge = _cross_lineage_bridge()
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name, table in (("hourly_state_transition_oof.parquet", transitions), ("gmm_component_alignment.csv", profiles), ("transition_quality_by_month.csv", transition_quality), ("state_transition_support.csv", state_support), ("performance_by_month_side_state_phase.csv", performance), ("cross_lineage_bridge.csv", bridge)):
            path = stage / name
            if name.endswith(".parquet"): table.to_parquet(path, index=False, compression="zstd")
            else: table.to_csv(path, index=False)
            outputs[name] = {"path": str(output_dir / name), "rows": int(len(table)), "sha256": sha256(path)}
        input_paths = [feature_root / "manifest.json", label_root / "manifest.json", score_root / "manifest.json"] + ([extension_feature_root / "manifest.json", extension_label_root / "manifest.json"] if extension_feature_root is not None and extension_label_root is not None else [])
        manifest = {"schema": "early_2022_inverse_pi_separate_regime_transition_supplement_v1", "status": "MATERIALIZED_SEPARATE_NON_POOLED_RESEARCH_OOF", "promotion_eligible": False, "population_lineage": "jan_jul_2022_inverse_pi_market_grid_causal_features_v1", "coverage": "2022-01-01T00:00:00Z through 2022-08-30T00:00:00Z signal time; exact labels resolve through 2022-08-30T12:00:00Z", "separation": "No early22 GMM state ID, state probability, transition phase or performance result is a later PF taxonomy identity or pooled metric.", "validation": "non-walk-forward leave-calendar-month-out OOF; GMM and transition classifier refit excluding each scored month", "causal_contract": "state/transition inputs are feature_root primitives available at signal timestamp; exact labels resolve decision+12h", "performance_score": "existing raw block-OOF score for Jan-Jul; August uses a separately labelled August leave-month-out direct HGB score; mapped scores are excluded", "global_selection": "one pooled cross-side top 10% per calendar month by raw OOF score", "inputs": {str(path): sha256(path) for path in input_paths}, "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}
        (stage / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT); parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--score-root", type=Path, default=SCORE_ROOT); parser.add_argument("--extension-feature-root", type=Path, default=EXT_FEATURE_ROOT); parser.add_argument("--extension-label-root", type=Path, default=EXT_LABEL_ROOT); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv); print(json.dumps(run(feature_root=args.feature_root, label_root=args.label_root, score_root=args.score_root, output_dir=args.output_dir, extension_feature_root=args.extension_feature_root, extension_label_root=args.extension_label_root), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
