#!/usr/bin/env python3
"""Strict 2022--25 sticky full-covariance regime challenger; assess once in 2026.

This intentionally does not touch the diagonal-GMM baseline.  It is a
separate, frozen identity with the same hourly panel and calendar boundary.
Transition fields are forbidden: this is a current-regime representation only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data_perp/artifacts/strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1"
PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DIAGONAL = ROOT / "data_perp/artifacts/strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3"
START = pd.Timestamp("2022-08-30", tz="UTC")
CUT = pd.Timestamp("2026-01-01", tz="UTC")
FAMILIES = ("volatility", "liquidity_proxy", "dependence_covariance", "distribution_dynamics")
COMPONENTS = (3, 4, 5, 6)
STICKY_PRIORS = (10.0, 50.0, 200.0, 1000.0, 5000.0)
MAX_PER_FAMILY = 8


def status(message: str) -> None:
    if os.environ.get("REGIME_CHALLENGER_PROGRESS") == "1":
        print(message, flush=True)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family(name: str) -> str:
    name = name.lower()
    if any(x in name for x in ("liquidity", "spread", "depth", "amihud")):
        return "liquidity_proxy"
    if any(x in name for x in ("corr", "covar", "depend", "dispersion")):
        return "dependence_covariance"
    if any(x in name for x in ("vol", "atr", "range")):
        return "volatility"
    return "distribution_dynamics"


def transition_counts(states: np.ndarray, segments: np.ndarray, n_states: int) -> np.ndarray:
    result = np.zeros((n_states, n_states), dtype=float)
    contiguous = segments[1:] == segments[:-1]
    np.add.at(result, (states[:-1][contiguous], states[1:][contiguous]), 1.0)
    return result


def causal_filter(emissions: np.ndarray, segments: np.ndarray, transition: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Causal HMM-style filtering, with no propagation over calendar gaps."""
    filtered = np.empty_like(emissions)
    filtered[0] = emissions[0] / max(float(emissions[0].sum()), 1e-12)
    evidence: list[float] = []
    switches: list[bool] = []
    for row in range(1, len(emissions)):
        if segments[row] != segments[row - 1]:
            filtered[row] = emissions[row] / max(float(emissions[row].sum()), 1e-12)
            continue
        previous = int(filtered[row - 1].argmax())
        unnormalised = (filtered[row - 1] @ transition) * emissions[row]
        probability = max(float(unnormalised.sum()), 1e-12)
        filtered[row] = unnormalised / probability
        evidence.append(float(np.log(probability)))
        switches.append(int(filtered[row].argmax()) != previous)
    return filtered, float(np.mean(evidence)), float(np.mean(switches))


def run_lengths(states: np.ndarray, segments: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    active = 1
    for row in range(1, len(states)):
        if segments[row] != segments[row - 1] or states[row] != states[row - 1]:
            lengths.append(active)
            active = 1
        else:
            active += 1
    lengths.append(active)
    return np.asarray(lengths, dtype=int)


def max_abs_correlation(values: np.ndarray, candidate: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    correlations = np.corrcoef(np.column_stack([values[:, candidate], values[:, selected]]), rowvar=False)[0, 1:]
    correlations = np.abs(correlations[np.isfinite(correlations)])
    return float(correlations.max()) if len(correlations) else 0.0


def select_features(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Feature selection used only inside the pre-selection training block."""
    # Calling ``frame[candidates]`` here would make a second 14k-column copy
    # of the panel.  Pandas can compute the numeric variances blockwise on the
    # existing frame, after which only the shortlisted columns are materialised.
    variance = (
        frame.var(numeric_only=True)
        .reindex(candidates)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    shortlist: list[str] = []
    for group in FAMILIES:
        shortlist.extend([name for name in variance.sort_values(ascending=False).index if family(name) == group][:40])
    shortlist = list(dict.fromkeys(shortlist))
    raw = SimpleImputer(strategy="median").fit_transform(frame[shortlist])
    lo, hi = np.quantile(raw, .005, axis=0), np.quantile(raw, .995, axis=0)
    scaled = RobustScaler().fit_transform(np.clip(raw, lo, hi))
    chosen: list[int] = []
    for group in FAMILIES:
        for index, name in enumerate(shortlist):
            if family(name) == group and max_abs_correlation(scaled, index, chosen) < .95:
                chosen.append(index)
                if sum(family(shortlist[i]) == group for i in chosen) == MAX_PER_FAMILY:
                    break
    return [shortlist[index] for index in chosen]


def fit_transform(train: pd.DataFrame, other: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    raw_train = imputer.fit_transform(train[features])
    raw_other = imputer.transform(other[features])
    lower, upper = np.quantile(raw_train, .005, axis=0), np.quantile(raw_train, .995, axis=0)
    scaler = RobustScaler().fit(np.clip(raw_train, lower, upper))
    state = {"features": features, "imputer": imputer, "lower": lower, "upper": upper, "scaler": scaler}
    return scaler.transform(np.clip(raw_train, lower, upper)), scaler.transform(np.clip(raw_other, lower, upper)), state


def model(components: int) -> GaussianMixture:
    return GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-3, n_init=1, random_state=1729, max_iter=150)


def stability_row(label: str, states: np.ndarray, segments: np.ndarray) -> dict[str, object]:
    runs = run_lengths(states, segments)
    same = segments[1:] == segments[:-1]
    switches = states[1:][same] != states[:-1][same]
    return {"representation": label, "runs": len(runs), "mean_hours": float(runs.mean()), "median_hours": float(np.median(runs)), "p90_hours": float(np.quantile(runs, .9)), "max_hours": int(runs.max()), "temporal_switch_rate": float(switches.mean())}


def diagonal_metrics() -> dict[str, object]:
    manifest = json.loads((DIAGONAL / "manifest.json").read_text())
    dwell = pd.read_csv(DIAGONAL / "2026_dwell_stability.csv")
    filtered = dwell.loc[dwell["representation"].eq("filtered")].iloc[0].to_dict()
    return {
        "predictive_log_score": manifest["test_predictive_log_score"],
        "temporal_switch_rate": manifest["test_temporal_switch_rate"],
        **filtered,
    }


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    status("loading hourly panel")
    panel = pd.read_parquet(PANEL)
    panel["source_utc"] = pd.to_datetime(panel["source_utc"], utc=True)
    panel = panel.loc[panel["source_utc"].ge(START)].sort_values("source_utc").reset_index(drop=True)
    cut_index = int(panel.source_utc.searchsorted(CUT, side="left"))
    train, test = panel.iloc[:cut_index], panel.iloc[cut_index:]
    if train.source_utc.max() >= CUT or test.source_utc.min() < CUT:
        raise AssertionError("The frozen 2022-2025/2026 boundary was violated")
    candidates = [name for name in panel.columns if name not in ("source_utc", "calendar_segment_id") and "transition" not in name.lower() and pd.api.types.is_numeric_dtype(panel[name])]

    # Both geometry and persistence use a final blocked 20%, and feature
    # selection sees only the preceding block.  The 2026 rows are untouched.
    split = len(train) - max(1000, len(train) // 5)
    selection, blocked = train.iloc[:split], train.iloc[split:]
    status("selecting pre-block features")
    features = select_features(selection, candidates)
    x_selection, x_blocked, selection_state = fit_transform(selection, blocked, features)
    selection_segments = selection.calendar_segment_id.astype(str).to_numpy()
    blocked_segments = blocked.calendar_segment_id.astype(str).to_numpy()
    sweep: list[dict[str, object]] = []
    best: tuple[tuple[int, float, float], GaussianMixture, float] | None = None
    for components in COMPONENTS:
        status(f"blocked geometry/persistence sweep k={components}")
        fitted = model(components).fit(x_selection)
        emissions = fitted.predict_proba(x_selection)
        states = emissions.argmax(axis=1)
        counts = transition_counts(states, selection_segments, components)
        for sticky in STICKY_PRIORS:
            transition = counts + sticky * np.eye(components)
            transition /= transition.sum(axis=1, keepdims=True)
            filtered, score, switching = causal_filter(fitted.predict_proba(x_blocked), blocked_segments, transition)
            check = stability_row("blocked_filtered", filtered.argmax(axis=1), blocked_segments)
            viable = bool(
                check["median_hours"] >= 6
                and check["temporal_switch_rate"] <= .10
            )
            objective = score - .05 * switching
            row = {"components": components, "sticky_prior": sticky, "blocked_predictive_log_score": score, "blocked_temporal_switch_rate": switching, "blocked_median_dwell_hours": check["median_hours"], "blocked_objective": objective, "persistent_state_gate_passed": viable}
            sweep.append(row)
            # Gate first; then predictive evidence with a smaller, deterministic
            # component-count tiebreak makes this a true train-only HPO choice.
            key = (int(viable), objective, -components)
            if best is None or key > best[0]:
                best = (key, fitted, sticky)
    assert best is not None
    selected_components, selected_sticky = int(-best[0][2]), best[2]

    # Refit the frozen identity after train-only choice.  Transformations now
    # use all pre-2026 rows, while no selection decision is revisited.
    status("refitting frozen full-covariance identity")
    x_train, x_test, frozen_state = fit_transform(train, test, features)
    gmm = model(selected_components).fit(x_train)
    train_emissions = gmm.predict_proba(x_train)
    train_states = train_emissions.argmax(axis=1)
    train_segments = train.calendar_segment_id.astype(str).to_numpy()
    frozen_transition = transition_counts(train_states, train_segments, selected_components) + selected_sticky * np.eye(selected_components)
    frozen_transition /= frozen_transition.sum(axis=1, keepdims=True)
    test_emissions = gmm.predict_proba(x_test)
    raw_states = test_emissions.argmax(axis=1)
    test_segments = test.calendar_segment_id.astype(str).to_numpy()
    filtered, predictive_score, switch_rate = causal_filter(test_emissions, test_segments, frozen_transition)
    states = filtered.argmax(axis=1)
    entropy = -(np.clip(filtered, 1e-12, 1) * np.log(np.clip(filtered, 1e-12, 1))).sum(axis=1) / np.log(selected_components)
    ordered = np.sort(filtered, axis=1)
    ood_train = -gmm.score_samples(x_train)
    ood = -gmm.score_samples(x_test)
    ood_threshold = float(np.quantile(ood_train, .99))
    identity_payload = {"model_family": "sticky_full_covariance_gmm", "components": selected_components, "sticky_prior": selected_sticky, "features": features, "panel_sha256": sha(PANEL), "training_end_exclusive_utc": CUT.isoformat()}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:16]

    sidecar = test[["source_utc", "calendar_segment_id"]].copy()
    sidecar["regime_model_identity"] = f"strict_fullcov_{identity}"
    sidecar["regime_state_id_raw"], sidecar["regime_state_id"] = raw_states, states
    sidecar["regime_entropy"], sidecar["regime_margin"] = entropy, ordered[:, -1] - ordered[:, -2]
    sidecar["regime_ood_score"], sidecar["regime_is_ood"], sidecar["regime_available_utc"] = ood, ood > ood_threshold, sidecar["source_utc"]
    for state in range(selected_components):
        sidecar[f"regime_state_p_raw__{state}"] = test_emissions[:, state]
        sidecar[f"regime_state_p__{state}"] = filtered[:, state]
    sidecar["month"] = sidecar.source_utc.dt.strftime("%Y-%m")
    same_segment = sidecar.calendar_segment_id.eq(sidecar.calendar_segment_id.shift())
    sidecar["state_changed"] = sidecar.regime_state_id.ne(sidecar.regime_state_id.shift()) & same_segment
    sidecar["raw_state_changed"] = sidecar.regime_state_id_raw.ne(sidecar.regime_state_id_raw.shift()) & same_segment
    monthly = sidecar.groupby("month", as_index=False).agg(rows=("source_utc", "size"), states=("regime_state_id", "nunique"), mean_entropy=("regime_entropy", "mean"), mean_margin=("regime_margin", "mean"), ood_fraction=("regime_is_ood", "mean"), state_change_fraction=("state_changed", "mean"), raw_state_change_fraction=("raw_state_changed", "mean"))
    dwell = pd.DataFrame([stability_row("raw", raw_states, test_segments), stability_row("filtered", states, test_segments)])
    profiles = []
    for state in range(selected_components):
        means = pd.Series(x_train[train_states == state].mean(axis=0), index=features)
        top = means.abs().sort_values(ascending=False).head(8)
        profiles.append({"regime_state_id": state, "train_rows": int((train_states == state).sum()), "semantic_train_only": " | ".join(top.index), "top_feature_signed_robust_means": json.dumps({name: float(means[name]) for name in top.index})})
    challenger = {"predictive_log_score": predictive_score, **dwell.loc[dwell.representation.eq("filtered")].iloc[0].to_dict()}
    comparison = pd.DataFrame([{"arm": "rejected_diagonal_gmm_v3", **diagonal_metrics()}, {"arm": "sticky_full_covariance_challenger_v1", **challenger}])
    comparison["persistent_state_gate_passed"] = (comparison.median_hours >= 6) & (comparison.temporal_switch_rate <= .10)
    status("materialising sealed forward sidecar and comparison")
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        sidecar.to_parquet(temporary / "regime_only_forward_2026_sidecar.parquet", index=False)
        pd.DataFrame(profiles).to_csv(temporary / "semantic_train_only_profiles.csv", index=False)
        pd.DataFrame(sweep).to_csv(temporary / "train_only_geometry_and_persistence_sweep.csv", index=False)
        monthly.to_csv(temporary / "2026_monthly_coverage_stability.csv", index=False)
        dwell.to_csv(temporary / "2026_dwell_stability.csv", index=False)
        comparison.to_csv(temporary / "direct_rejected_diagonal_comparison.csv", index=False)
        joblib.dump({**frozen_state, "gmm": gmm, "transition": frozen_transition, "ood_threshold": ood_threshold, "identity": identity_payload}, temporary / "frozen_regime_model.joblib", compress=3)
        (temporary / "feature_contract.json").write_text(json.dumps({"selected_train_only_features": features, "selected_family_counts": pd.Series([family(name) for name in features]).value_counts().to_dict(), "selection": "pre-block-only variance shortlist; four-family balance; 0.5/99.5% winsorisation; robust scaling; abs-correlation<0.95", "hpo": "blocked-2022-2025 full-covariance GMM k=3..6 and sticky prior sweep", "persistent_state_gate": "median dwell >=6h and hourly temporal switching <=10%", "transition_outputs_excluded": True, "state_identity": "frozen-model-local; never equated to transition state or another regime model's state IDs"}, indent=2) + "\n")
        files = [path for path in temporary.iterdir() if path.is_file()]
        passed = bool(comparison.loc[comparison.arm.eq("sticky_full_covariance_challenger_v1"), "persistent_state_gate_passed"].iloc[0])
        manifest = {"schema": "strict_forward_sticky_fullcov_regime_challenger_v1", "status": "SEALED_STRICT_FORWARD_CHALLENGER" if passed else "SEALED_STRICT_FORWARD_CHALLENGER_REJECTED", "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "training_start_utc": str(train.source_utc.min()), "training_end_exclusive_utc": CUT.isoformat(), "train_rows": len(train), "eval_start_utc": CUT.isoformat(), "eval_end_utc": str(test.source_utc.max()), "eval_rows": len(test), "strict_split_contract": "all feature selection, preprocessing, geometry, persistence, uncertainty thresholds and semantics use 2022-2025 only; 2026 is untouched assessment", "same_hourly_panel_as_rejected_diagonal": True, "transition_outputs_excluded": True, "test_predictive_log_score": predictive_score, "test_temporal_switch_rate": switch_rate, "persistent_state_gate_passed": passed, "promotion_eligible": False, "inputs": {str(PANEL): sha(PANEL), str(DIAGONAL / "manifest.json"): sha(DIAGONAL / "manifest.json")}, "outputs_sha256": {path.name: sha(path) for path in files}}
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.sha256").write_text(f"{sha(manifest_path)}  manifest.json\n")
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
