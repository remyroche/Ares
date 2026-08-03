#!/usr/bin/env python3
"""Strict 1h duration-aware semi-Markov regime challenger.

The representation is a frozen pre-2026 Gaussian emission model plus a
state-specific discrete duration model.  Filtering is causal: at an hourly
row it only uses the current emission, the previous posterior and duration
hazards learned before 2026.  Calendar gaps reset the filter.  This is a
research-only current-regime representation; transition/outcome fields are
excluded from fitting and no economics enter selection or the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_forward_semimarkov_regime_challenger_2022aug_2025_to_2026_20260730_v1"
START = pd.Timestamp("2022-08-30T00:00:00Z")
CUT = pd.Timestamp("2026-01-01T00:00:00Z")
FAMILIES = ("volatility", "liquidity_proxy", "dependence_covariance", "distribution_dynamics")
COMPONENTS = (3, 4, 5, 6)
MIN_DWELLS = (3, 6, 12)
MAX_DURATION = 72
MAX_PER_FAMILY = 8
SEED = 20260730

# Frozen before the 2026 assessment is read.  The gate is deliberately
# structural: it cannot optimise any trade outcome or target on the holdout.
PERSISTENT_STATE_GATE = {
    "median_dwell_hours_min": 6.0,
    "temporal_switch_rate_max": 0.10,
    "minimum_state_occupancy_min": 0.02,
    "mean_max_posterior_min": 0.55,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("liquidity", "spread", "depth", "amihud", "volume")):
        return "liquidity_proxy"
    if any(token in lower for token in ("corr", "covar", "depend", "dispersion")):
        return "dependence_covariance"
    if any(token in lower for token in ("vol", "atr", "range")):
        return "volatility"
    return "distribution_dynamics"


def contiguous_segments(timestamps: pd.Series, supplied: pd.Series) -> np.ndarray:
    """Refine supplied segment IDs at any non-hourly gap, fail-closed."""
    ts = pd.to_datetime(timestamps, utc=True)
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("panel must have unique, ordered UTC hourly decision rows")
    delta = ts.diff().dt.total_seconds().div(3600.0)
    boundaries = delta.ne(1.0) | supplied.astype(str).ne(supplied.astype(str).shift())
    boundaries.iloc[0] = True
    return boundaries.cumsum().astype(str).to_numpy()


def max_abs_correlation(values: np.ndarray, candidate: int, selected: list[int]) -> float:
    if not selected:
        return 0.0
    corr = np.corrcoef(np.column_stack([values[:, candidate], values[:, selected]]), rowvar=False)[0, 1:]
    corr = np.abs(corr[np.isfinite(corr)])
    return float(corr.max()) if len(corr) else 0.0


def select_features(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    variance = frame.var(numeric_only=True).reindex(candidates).replace([np.inf, -np.inf], np.nan).dropna()
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
                if sum(family(shortlist[position]) == group for position in chosen) == MAX_PER_FAMILY:
                    break
    if len(chosen) < len(FAMILIES):
        raise ValueError("feature selection did not retain every required family")
    return [shortlist[index] for index in chosen]


def transform(train: pd.DataFrame, other: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    raw_train = imputer.fit_transform(train[features])
    raw_other = imputer.transform(other[features])
    lower, upper = np.quantile(raw_train, .005, axis=0), np.quantile(raw_train, .995, axis=0)
    scaler = RobustScaler().fit(np.clip(raw_train, lower, upper))
    state = {"features": features, "imputer": imputer, "lower": lower, "upper": upper, "scaler": scaler}
    return scaler.transform(np.clip(raw_train, lower, upper)), scaler.transform(np.clip(raw_other, lower, upper)), state


def gmm_model(components: int) -> GaussianMixture:
    return GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-3, n_init=1, random_state=SEED, max_iter=150)


def runs(states: np.ndarray, segments: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    state, length = int(states[0]), 1
    for index in range(1, len(states)):
        if segments[index] != segments[index - 1] or states[index] != state:
            result.append((state, length))
            state, length = int(states[index]), 1
        else:
            length += 1
    result.append((state, length))
    return result


def duration_model(states: np.ndarray, segments: np.ndarray, n_states: int, minimum_dwell: int, max_duration: int = MAX_DURATION) -> dict[str, np.ndarray]:
    """Fit state duration hazards and exit destinations from pre-2026 states."""
    observed_runs = runs(states, segments)
    lengths = [np.array([length for state, length in observed_runs if state == item], dtype=int) for item in range(n_states)]
    hazards = np.empty((n_states, max_duration), dtype=float)
    for state, local in enumerate(lengths):
        if not len(local):
            hazards[state] = 1.0 / max(minimum_dwell, 1)
            continue
        for age in range(1, max_duration + 1):
            at_risk = int((local >= age).sum())
            ending = int((local == age).sum()) if age < max_duration else int((local >= age).sum())
            # A small prior avoids a brittle zero-hazard estimate.  Hard
            # minimum dwell is the explicit-duration component of this model.
            hazard = (ending + 1.0) / (at_risk + 8.0)
            hazards[state, age - 1] = 0.0 if age < minimum_dwell else float(np.clip(hazard, .002, .80))
    exits = np.ones((n_states, n_states), dtype=float) - np.eye(n_states)
    for previous, current, same in zip(states[:-1], states[1:], segments[1:] == segments[:-1]):
        if same and previous != current:
            exits[int(previous), int(current)] += 1.0
    exits[np.arange(n_states), np.arange(n_states)] = 0.0
    exits /= exits.sum(axis=1, keepdims=True)
    initial = np.bincount(states, minlength=n_states).astype(float) + 1.0
    initial /= initial.sum()
    return {"hazards": hazards, "exits": exits, "initial": initial, "minimum_dwell": np.array([minimum_dwell], dtype=int)}


def semimarkov_filter(emissions: np.ndarray, segments: np.ndarray, duration: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Causal explicit-duration filter over (state, capped age)."""
    n_rows, n_states = emissions.shape
    max_duration = duration["hazards"].shape[1]
    joint = np.zeros((n_states, max_duration), dtype=float)
    posterior, age_mean, exit_probability, log_evidence = [], [], [], []
    previous_segment: str | None = None
    for index in range(n_rows):
        if previous_segment != segments[index]:
            prior = np.zeros_like(joint)
            prior[:, 0] = duration["initial"]
        else:
            prior = np.zeros_like(joint)
            for state in range(n_states):
                for age in range(max_duration):
                    mass = joint[state, age]
                    if mass <= 0.0:
                        continue
                    hazard = duration["hazards"][state, age]
                    next_age = min(age + 1, max_duration - 1)
                    prior[state, next_age] += mass * (1.0 - hazard)
                    prior[:, 0] += mass * hazard * duration["exits"][state]
        weighted = prior * emissions[index, :, None]
        normalizer = max(float(weighted.sum()), 1e-300)
        joint = weighted / normalizer
        state_p = joint.sum(axis=1)
        posterior.append(state_p)
        age_mean.append(float((joint * np.arange(1, max_duration + 1)[None, :]).sum()))
        exit_probability.append(float((joint * duration["hazards"]).sum()))
        if previous_segment == segments[index]:
            log_evidence.append(float(np.log(normalizer)))
        previous_segment = segments[index]
    return np.asarray(posterior), np.asarray(age_mean), np.asarray(exit_probability), np.asarray(log_evidence), float(np.mean(log_evidence))


def stability(states: np.ndarray, segments: np.ndarray, posterior: np.ndarray) -> dict[str, float | int]:
    lengths = np.asarray([length for _, length in runs(states, segments)], dtype=float)
    same = segments[1:] == segments[:-1]
    switch = states[1:][same] != states[:-1][same]
    occupancy = np.bincount(states, minlength=posterior.shape[1]) / len(states)
    return {
        "runs": int(len(lengths)), "mean_dwell_hours": float(lengths.mean()), "median_dwell_hours": float(np.median(lengths)),
        "p90_dwell_hours": float(np.quantile(lengths, .90)), "temporal_switch_rate": float(switch.mean()) if len(switch) else 0.0,
        "minimum_state_occupancy": float(occupancy.min()), "mean_max_posterior": float(posterior.max(axis=1).mean()),
    }


def gate(metrics: dict[str, float | int]) -> bool:
    return bool(metrics["median_dwell_hours"] >= PERSISTENT_STATE_GATE["median_dwell_hours_min"] and metrics["temporal_switch_rate"] <= PERSISTENT_STATE_GATE["temporal_switch_rate_max"] and metrics["minimum_state_occupancy"] >= PERSISTENT_STATE_GATE["minimum_state_occupancy_min"] and metrics["mean_max_posterior"] >= PERSISTENT_STATE_GATE["mean_max_posterior_min"])


def profile_rows(values: np.ndarray, states: np.ndarray, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in range(values.shape[1] if False else int(states.max()) + 1):
        subset = values[states == state]
        means = pd.Series(subset.mean(axis=0), index=features)
        family_means = {group: float(means[[name for name in features if family(name) == group]].mean()) for group in FAMILIES}
        long_features = [name for name in features if "long" in name.lower()]
        short_features = [name for name in features if "short" in name.lower()]
        top = means.abs().sort_values(ascending=False).head(8)
        rows.append({"regime_state_id": state, "train_rows": int(len(subset)), "volatility_qualifier": family_means["volatility"], "liquidity_qualifier": family_means["liquidity_proxy"], "dependence_qualifier": family_means["dependence_covariance"], "distribution_qualifier": family_means["distribution_dynamics"], "long_signal_context_qualifier": float(means[long_features].mean()) if long_features else np.nan, "short_signal_context_qualifier": float(means[short_features].mean()) if short_features else np.nan, "semantic_train_only": " | ".join(top.index), "top_feature_signed_robust_means": json.dumps({name: float(means[name]) for name in top.index})})
    return pd.DataFrame(rows)


def run(output: Path = DEFAULT_OUTPUT, panel_path: Path = PANEL) -> Path:
    output, panel_path = Path(output), Path(panel_path)
    if output.exists():
        raise FileExistsError(output)
    panel = pd.read_parquet(panel_path)
    panel["source_utc"] = pd.to_datetime(panel["source_utc"], utc=True)
    panel = panel.loc[panel.source_utc.ge(START)].sort_values("source_utc").reset_index(drop=True)
    if "calendar_segment_id" not in panel:
        raise ValueError("authoritative panel lacks calendar_segment_id")
    segments = contiguous_segments(panel.source_utc, panel.calendar_segment_id)
    cut_index = int(panel.source_utc.searchsorted(CUT, side="left"))
    train, test = panel.iloc[:cut_index].copy(), panel.iloc[cut_index:].copy()
    train_segments, test_segments = segments[:cut_index], segments[cut_index:]
    if len(train) == 0 or len(test) == 0 or train.source_utc.max() >= CUT or test.source_utc.min() < CUT:
        raise AssertionError("frozen 2022-2025/2026 boundary violated")
    candidates = [name for name in panel.columns if name not in ("source_utc", "calendar_segment_id") and "transition" not in name.lower() and pd.api.types.is_numeric_dtype(panel[name])]
    selection_end = len(train) - max(1000, len(train) // 5)
    selection, blocked = train.iloc[:selection_end], train.iloc[selection_end:]
    features = select_features(selection, candidates)
    x_selection, x_blocked, _ = transform(selection, blocked, features)
    selection_segments = train_segments[:selection_end]
    blocked_segments = train_segments[selection_end:]
    sweep: list[dict[str, Any]] = []
    best: tuple[tuple[int, float, int, int], int, int] | None = None
    for components in COMPONENTS:
        fitted = gmm_model(components).fit(x_selection)
        selection_states = fitted.predict(x_selection)
        for minimum_dwell in MIN_DWELLS:
            duration = duration_model(selection_states, selection_segments, components, minimum_dwell)
            posterior, _, _, _, score = semimarkov_filter(fitted.predict_proba(x_blocked), blocked_segments, duration)
            metrics = stability(posterior.argmax(axis=1), blocked_segments, posterior)
            passed = gate(metrics)
            row = {"components": components, "minimum_dwell_hours": minimum_dwell, "blocked_predictive_log_score": score, **metrics, "persistent_state_gate_passed": passed}
            sweep.append(row)
            key = (int(passed), score, -components, -minimum_dwell)
            if best is None or key > best[0]:
                best = (key, components, minimum_dwell)
    assert best is not None
    components, minimum_dwell = best[1], best[2]
    x_train, x_test, preprocess = transform(train, test, features)
    gmm = gmm_model(components).fit(x_train)
    train_states = gmm.predict(x_train)
    duration = duration_model(train_states, train_segments, components, minimum_dwell)
    posterior, age_mean, exit_probability, _, predictive_log_score = semimarkov_filter(gmm.predict_proba(x_test), test_segments, duration)
    states = posterior.argmax(axis=1)
    metrics = stability(states, test_segments, posterior)
    raw = gmm.predict(x_test)
    entropy = -(np.clip(posterior, 1e-12, 1.0) * np.log(np.clip(posterior, 1e-12, 1.0))).sum(axis=1) / np.log(components)
    ordered = np.sort(posterior, axis=1)
    ood_train, ood_test = -gmm.score_samples(x_train), -gmm.score_samples(x_test)
    ood_threshold = float(np.quantile(ood_train, .99))
    sidecar = test[["source_utc", "calendar_segment_id"]].copy()
    identity_payload = {"family": "explicit_duration_semimarkov_gmm", "components": components, "minimum_dwell_hours": minimum_dwell, "max_duration_hours": MAX_DURATION, "features": features, "panel_sha256": sha256(panel_path), "training_end_exclusive_utc": CUT.isoformat()}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:16]
    sidecar["regime_model_identity"] = f"strict_semimarkov_{identity}"
    sidecar["regime_state_id_raw"] = raw
    sidecar["regime_state_id"] = states
    sidecar["regime_entropy"] = entropy
    sidecar["regime_margin"] = ordered[:, -1] - ordered[:, -2]
    sidecar["regime_expected_dwell_age_hours"] = age_mean
    sidecar["regime_exit_hazard_next_hour"] = exit_probability
    sidecar["regime_ood_score"] = ood_test
    sidecar["regime_is_ood"] = ood_test > ood_threshold
    sidecar["regime_available_utc"] = sidecar.source_utc
    for state in range(components):
        sidecar[f"regime_state_p__{state}"] = posterior[:, state]
    sidecar["month"] = sidecar.source_utc.dt.strftime("%Y-%m")
    same = sidecar.calendar_segment_id.eq(sidecar.calendar_segment_id.shift())
    sidecar["state_changed"] = sidecar.regime_state_id.ne(sidecar.regime_state_id.shift()) & same
    monthly = sidecar.groupby("month", as_index=False).agg(rows=("source_utc", "size"), states=("regime_state_id", "nunique"), mean_entropy=("regime_entropy", "mean"), mean_margin=("regime_margin", "mean"), mean_expected_dwell_age_hours=("regime_expected_dwell_age_hours", "mean"), mean_exit_hazard_next_hour=("regime_exit_hazard_next_hour", "mean"), ood_fraction=("regime_is_ood", "mean"), state_change_fraction=("state_changed", "mean"))
    profiles = profile_rows(x_train, train_states, features)
    gate_result = gate(metrics)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        preregistration = {"schema": "strict_semimarkov_persistent_state_gate_v1", "frozen_before_2026_assessment": True, "gate": PERSISTENT_STATE_GATE, "selection": "pre-2026 blocked selection: gate first, then causal blocked predictive log evidence, then lower component/minimum-dwell tie breaks", "forbidden": ["2026 model selection", "economics targets", "transition fields"]}
        (temporary / "pre_registered_persistent_state_gate.json").write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
        sidecar.to_parquet(temporary / "regime_only_forward_2026_sidecar.parquet", index=False)
        pd.DataFrame(sweep).to_csv(temporary / "pre2026_geometry_duration_sweep.csv", index=False)
        pd.DataFrame([{**metrics, "persistent_state_gate_passed": gate_result, "predictive_log_score": predictive_log_score}]).to_csv(temporary / "2026_persistence_metrics.csv", index=False)
        monthly.to_csv(temporary / "2026_monthly_coverage_stability.csv", index=False)
        profiles.to_csv(temporary / "train_only_state_qualification.csv", index=False)
        joblib.dump({**preprocess, "gmm": gmm, "duration": duration, "ood_threshold": ood_threshold, "identity": identity_payload}, temporary / "frozen_semimarkov_regime_model.joblib", compress=3)
        contract = {"selected_train_only_features": features, "selected_family_counts": pd.Series([family(name) for name in features]).value_counts().to_dict(), "selection": "pre-block-only variance shortlist; four-family balance; 0.5/99.5% winsorisation; robust scaling; abs-correlation<0.95", "hpo": "pre-2026 blocked sweep: full-covariance GMM k=3..6 and explicit minimum dwell 3/6/12h", "semimarkov": "state-specific empirical discrete duration hazards with a hard minimum dwell and causal age-augmented filtering", "state_outputs": "causal posterior, entropy, margin, expected duration age, next-hour exit hazard and OOD", "qualifiers": "train-only standardized distribution, volatility, liquidity-proxy, dependence and long/short signal-context summaries; not trade outcomes", "cadence": "1h model and assessment; 1m labels only", "transition_outputs_excluded": True, "state_identity": "frozen-model-local; never equated to transition state or another regime model's state IDs"}
        (temporary / "feature_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        files = [path for path in temporary.iterdir() if path.is_file()]
        manifest = {"schema": "strict_forward_semimarkov_regime_challenger_v1", "status": "SEALED_STRICT_FORWARD_CHALLENGER" if gate_result else "SEALED_STRICT_FORWARD_CHALLENGER_REJECTED", "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "training_start_utc": str(train.source_utc.min()), "training_end_exclusive_utc": CUT.isoformat(), "train_rows": len(train), "eval_start_utc": CUT.isoformat(), "eval_end_utc": str(test.source_utc.max()), "eval_rows": len(test), "strict_split_contract": "all feature selection, preprocessing, GMM geometry, duration selection/hazards, uncertainty/OOD thresholds and semantic qualification use 2022-2025 only; 2026 is untouched assessment", "transition_outputs_excluded": True, "test_predictive_log_score": predictive_log_score, "test_persistence_metrics": metrics, "persistent_state_gate": PERSISTENT_STATE_GATE, "persistent_state_gate_passed": gate_result, "promotion_eligible": False, "inputs": {str(panel_path): sha256(panel_path)}, "outputs_sha256": {path.name: sha256(path) for path in files}}
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel", type=Path, default=PANEL)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(run(**vars(parse_args())))
