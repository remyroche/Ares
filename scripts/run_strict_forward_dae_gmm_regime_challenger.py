#!/usr/bin/env python3
"""Strict DAE→GMM regime challenger on the common 1h 2022--2026 panel.

The DAE is a denoising representation, not a transition model.  Every choice
that can affect it (raw fields, bottleneck, training noise, GMM geometry,
persistence, density and reconstruction thresholds) is made before 2026.
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
import torch
from sklearn.mixture import GaussianMixture
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_forward_sticky_fullcov_regime_challenger import (
    CUT, DIAGONAL, FAMILIES, PANEL, START, STICKY_PRIORS, causal_filter,
    diagonal_metrics, family, fit_transform, run_lengths, select_features,
    sha, stability_row, transition_counts,
)

OUT = ROOT / "data_perp/artifacts/strict_forward_dae_gmm_regime_challenger_2022aug_2025_to_2026_20260730_v1"
STICKY_FULLCOV = ROOT / "data_perp/artifacts/strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1"
ECON = ROOT / "data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet"
DAE_HPO = ((4, 0.05), (8, 0.05), (12, 0.05))
GMM_COMPONENTS = (3, 4, 5, 6)
DAE_MAX_ROWS = 16_000
SEED = 1729


def status(message: str) -> None:
    if os.environ.get("REGIME_CHALLENGER_PROGRESS") == "1":
        print(message, flush=True)


class DenoisingAE(nn.Module):
    def __init__(self, inputs: int, bottleneck: int):
        super().__init__()
        width = max(24, min(64, inputs * 2))
        self.encoder = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, bottleneck))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(bottleneck, width), nn.ReLU(), nn.Linear(width, inputs))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def deterministic_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]


def fit_dae(train: np.ndarray, *, bottleneck: int, noise: float) -> DenoisingAE:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    net = DenoisingAE(train.shape[1], bottleneck)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = nn.MSELoss()
    values = torch.from_numpy(deterministic_sample(train.astype(np.float32), DAE_MAX_ROWS))
    net.train()
    for _ in range(8):
        for rows in torch.randperm(len(values)).split(512):
            clean = values[rows]
            noisy = clean + noise * torch.randn_like(clean)
            optimizer.zero_grad()
            error = loss(net(noisy), clean)
            error.backward()
            optimizer.step()
    net.eval()
    return net


def representation(net: DenoisingAE, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        tensor = torch.from_numpy(values.astype(np.float32))
        latent = net.encoder(tensor).cpu().numpy()
        reconstruction = net(tensor).cpu().numpy()
    error = np.mean(np.square(values - reconstruction), axis=1)
    return latent.astype(np.float64), error.astype(np.float64)


def density_input(latent: np.ndarray, error: np.ndarray, train_error: np.ndarray) -> np.ndarray:
    scale = max(float(np.std(train_error)), 1e-9)
    return np.column_stack([latent, (error - float(np.mean(train_error))) / scale])


def gmm(components: int) -> GaussianMixture:
    return GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-3, n_init=1, max_iter=150, random_state=SEED)


def arm_metrics(artifact: Path, arm: str) -> dict[str, object]:
    manifest = json.loads((artifact / "manifest.json").read_text())
    dwell = pd.read_csv(artifact / "2026_dwell_stability.csv")
    row = dwell.loc[dwell.representation.eq("filtered")].iloc[0].to_dict()
    return {"arm": arm, "predictive_log_score": manifest["test_predictive_log_score"], "temporal_switch_rate": manifest["test_temporal_switch_rate"], **row}


def economics_by_arm(sidecar: pd.DataFrame, arm: str) -> pd.DataFrame:
    economics = pd.read_parquet(ECON, columns=["__ts__", "side_name", "catboost_archetype", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"])
    economics["__ts__"] = pd.to_datetime(economics["__ts__"], utc=True)
    materialized = economics.merge(sidecar.rename(columns={"source_utc": "__ts__"}), on="__ts__", how="inner", validate="many_to_one")
    materialized["month"] = materialized["__ts__"].dt.strftime("%Y-%m")
    result = materialized.groupby(["month", "regime_state_id", "side_name", "catboost_archetype"], as_index=False).agg(rows=("execution_net_ev_12h", "size"), net_ev=("execution_net_ev_12h", "mean"), gross_ev=("execution_gross_ev_12h", "mean"), cost=("execution_cost_return", "mean"))
    result.insert(0, "arm", arm)
    return result


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    status("loading common hourly panel")
    panel = pd.read_parquet(PANEL)
    panel["source_utc"] = pd.to_datetime(panel["source_utc"], utc=True)
    panel = panel.loc[panel.source_utc.ge(START)].sort_values("source_utc").reset_index(drop=True)
    cut_index = int(panel.source_utc.searchsorted(CUT, side="left"))
    train, test = panel.iloc[:cut_index], panel.iloc[cut_index:]
    if train.source_utc.max() >= CUT or test.source_utc.min() < CUT:
        raise AssertionError("The frozen 2022-2025/2026 boundary was violated")
    candidates = [name for name in panel.columns if name not in ("source_utc", "calendar_segment_id") and "transition" not in name.lower() and pd.api.types.is_numeric_dtype(panel[name])]
    split = len(train) - max(1000, len(train) // 5)
    selection, blocked = train.iloc[:split], train.iloc[split:]
    status("train-only family-balanced feature selection")
    features = select_features(selection, candidates)
    x_selection, x_blocked, _ = fit_transform(selection, blocked, features)
    selection_segments = selection.calendar_segment_id.astype(str).to_numpy()
    blocked_segments = blocked.calendar_segment_id.astype(str).to_numpy()
    sweep: list[dict[str, object]] = []
    best: tuple[tuple[int, float, int, int], int, float, int, GaussianMixture] | None = None
    for bottleneck, noise in DAE_HPO:
        status(f"blocked DAE/GMM sweep bottleneck={bottleneck}")
        dae = fit_dae(x_selection, bottleneck=bottleneck, noise=noise)
        latent_selection, error_selection = representation(dae, x_selection)
        latent_blocked, error_blocked = representation(dae, x_blocked)
        density_selection = density_input(latent_selection, error_selection, error_selection)
        density_blocked = density_input(latent_blocked, error_blocked, error_selection)
        for components in GMM_COMPONENTS:
            fitted = gmm(components).fit(density_selection)
            emissions = fitted.predict_proba(density_selection)
            states = emissions.argmax(axis=1)
            counts = transition_counts(states, selection_segments, components)
            for sticky in STICKY_PRIORS:
                transition = counts + sticky * np.eye(components)
                transition /= transition.sum(axis=1, keepdims=True)
                filtered, score, switching = causal_filter(fitted.predict_proba(density_blocked), blocked_segments, transition)
                check = stability_row("blocked_filtered", filtered.argmax(axis=1), blocked_segments)
                viable = bool(check["median_hours"] >= 6 and check["temporal_switch_rate"] <= .10)
                objective = score - .05 * switching
                row = {"bottleneck": bottleneck, "noise": noise, "components": components, "sticky_prior": sticky, "blocked_predictive_log_score": score, "blocked_temporal_switch_rate": switching, "blocked_median_dwell_hours": check["median_hours"], "blocked_objective": objective, "persistent_state_gate_passed": viable}
                sweep.append(row)
                key = (int(viable), objective, -components, -bottleneck)
                if best is None or key > best[0]:
                    best = (key, bottleneck, noise, components, fitted)
    assert best is not None
    _, selected_bottleneck, selected_noise, selected_components, _ = best
    selected = max(sweep, key=lambda row: (int(row["persistent_state_gate_passed"]), float(row["blocked_objective"]), -int(row["components"]), -int(row["bottleneck"])))
    selected_sticky = float(selected["sticky_prior"])

    status("refitting selected frozen DAE/GMM representation")
    x_train, x_test, frozen_transform = fit_transform(train, test, features)
    dae = fit_dae(x_train, bottleneck=selected_bottleneck, noise=selected_noise)
    latent_train, error_train = representation(dae, x_train)
    latent_test, error_test = representation(dae, x_test)
    density_train = density_input(latent_train, error_train, error_train)
    density_test = density_input(latent_test, error_test, error_train)
    density_model = gmm(selected_components).fit(density_train)
    train_emissions = density_model.predict_proba(density_train)
    train_states = train_emissions.argmax(axis=1)
    train_segments = train.calendar_segment_id.astype(str).to_numpy()
    transition = transition_counts(train_states, train_segments, selected_components) + selected_sticky * np.eye(selected_components)
    transition /= transition.sum(axis=1, keepdims=True)
    test_emissions = density_model.predict_proba(density_test)
    raw_states = test_emissions.argmax(axis=1)
    test_segments = test.calendar_segment_id.astype(str).to_numpy()
    filtered, predictive_score, switching = causal_filter(test_emissions, test_segments, transition)
    states = filtered.argmax(axis=1)
    entropy = -(np.clip(filtered, 1e-12, 1) * np.log(np.clip(filtered, 1e-12, 1))).sum(axis=1) / np.log(selected_components)
    ordered = np.sort(filtered, axis=1)
    density_ood_train = -density_model.score_samples(density_train)
    density_ood = -density_model.score_samples(density_test)
    density_threshold, reconstruction_threshold = float(np.quantile(density_ood_train, .99)), float(np.quantile(error_train, .99))
    identity_payload = {"model_family": "denoising_autoencoder_to_full_covariance_gmm", "features": features, "bottleneck": selected_bottleneck, "noise": selected_noise, "epochs": 8, "components": selected_components, "sticky_prior": selected_sticky, "panel_sha256": sha(PANEL), "training_end_exclusive_utc": CUT.isoformat()}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:16]

    sidecar = test[["source_utc", "calendar_segment_id"]].copy()
    sidecar["regime_model_identity"] = f"strict_dae_gmm_{identity}"
    sidecar["regime_state_id_raw"], sidecar["regime_state_id"] = raw_states, states
    sidecar["regime_entropy"], sidecar["regime_margin"] = entropy, ordered[:, -1] - ordered[:, -2]
    sidecar["regime_density_ood_score"], sidecar["regime_reconstruction_error"] = density_ood, error_test
    sidecar["regime_is_density_ood"], sidecar["regime_is_reconstruction_ood"] = density_ood > density_threshold, error_test > reconstruction_threshold
    sidecar["regime_is_ood"], sidecar["regime_available_utc"] = sidecar.regime_is_density_ood | sidecar.regime_is_reconstruction_ood, sidecar.source_utc
    for state in range(selected_components):
        sidecar[f"regime_state_p_raw__{state}"] = test_emissions[:, state]
        sidecar[f"regime_state_p__{state}"] = filtered[:, state]
    sidecar["month"] = sidecar.source_utc.dt.strftime("%Y-%m")
    same_segment = sidecar.calendar_segment_id.eq(sidecar.calendar_segment_id.shift())
    sidecar["state_changed"] = sidecar.regime_state_id.ne(sidecar.regime_state_id.shift()) & same_segment
    sidecar["raw_state_changed"] = sidecar.regime_state_id_raw.ne(sidecar.regime_state_id_raw.shift()) & same_segment
    monthly = sidecar.groupby("month", as_index=False).agg(rows=("source_utc", "size"), states=("regime_state_id", "nunique"), mean_entropy=("regime_entropy", "mean"), mean_margin=("regime_margin", "mean"), ood_fraction=("regime_is_ood", "mean"), density_ood_fraction=("regime_is_density_ood", "mean"), reconstruction_ood_fraction=("regime_is_reconstruction_ood", "mean"), state_change_fraction=("state_changed", "mean"), raw_state_change_fraction=("raw_state_changed", "mean"))
    dwell = pd.DataFrame([stability_row("raw", raw_states, test_segments), stability_row("filtered", states, test_segments)])
    profiles = []
    for state in range(selected_components):
        means = pd.Series(x_train[train_states == state].mean(axis=0), index=features)
        top = means.abs().sort_values(ascending=False).head(8)
        profiles.append({"regime_state_id": state, "train_rows": int((train_states == state).sum()), "semantic_train_only": " | ".join(top.index), "top_feature_signed_robust_means": json.dumps({name: float(means[name]) for name in top.index})})
    dae_metrics = {"arm": "dae_to_fullcov_gmm_challenger_v1", "predictive_log_score": predictive_score, "temporal_switch_rate": switching, **dwell.loc[dwell.representation.eq("filtered")].iloc[0].to_dict()}
    structural = pd.DataFrame([{"arm": "rejected_diagonal_gmm_v3", **diagonal_metrics()}, arm_metrics(STICKY_FULLCOV, "rejected_sticky_fullcov_gmm_v1"), dae_metrics])
    structural["persistent_state_gate_passed"] = (structural.median_hours >= 6) & (structural.temporal_switch_rate <= .10)
    baseline_monthly = []
    for artifact, arm in ((DIAGONAL, "rejected_diagonal_gmm_v3"), (STICKY_FULLCOV, "rejected_sticky_fullcov_gmm_v1")):
        frame = pd.read_csv(artifact / "2026_monthly_coverage_stability.csv")
        frame.insert(0, "arm", arm)
        baseline_monthly.append(frame)
    monthly_comparison = pd.concat([*baseline_monthly, monthly.assign(arm="dae_to_fullcov_gmm_challenger_v1")], ignore_index=True, sort=False)
    all_economics = pd.concat([economics_by_arm(pd.read_parquet(DIAGONAL / "regime_only_forward_2026_sidecar.parquet"), "rejected_diagonal_gmm_v3"), economics_by_arm(pd.read_parquet(STICKY_FULLCOV / "regime_only_forward_2026_sidecar.parquet"), "rejected_sticky_fullcov_gmm_v1"), economics_by_arm(sidecar, "dae_to_fullcov_gmm_challenger_v1")], ignore_index=True)

    status("sealing sidecar, structural comparison and attribution")
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        sidecar.to_parquet(temporary / "regime_only_forward_2026_sidecar.parquet", index=False)
        pd.DataFrame(sweep).to_csv(temporary / "train_only_dae_geometry_and_persistence_sweep.csv", index=False)
        pd.DataFrame(profiles).to_csv(temporary / "semantic_train_only_profiles.csv", index=False)
        dwell.to_csv(temporary / "2026_dwell_stability.csv", index=False)
        monthly.to_csv(temporary / "2026_monthly_coverage_stability.csv", index=False)
        structural.to_csv(temporary / "three_arm_structural_comparison.csv", index=False)
        monthly_comparison.to_csv(temporary / "three_arm_monthly_stability_comparison.csv", index=False)
        all_economics.to_parquet(temporary / "three_arm_exact_side_economic_attribution.parquet", index=False)
        joblib.dump({**frozen_transform, "gmm": density_model, "transition": transition, "density_ood_threshold": density_threshold, "reconstruction_ood_threshold": reconstruction_threshold, "identity": identity_payload}, temporary / "frozen_dae_gmm_transform_and_density.joblib", compress=3)
        torch.save({"state_dict": dae.state_dict(), "inputs": len(features), "bottleneck": selected_bottleneck, "noise": selected_noise, "epochs": 8}, temporary / "frozen_dae_state_dict.pt")
        (temporary / "feature_contract.json").write_text(json.dumps({"selected_train_only_features": features, "selected_family_counts": pd.Series([family(name) for name in features]).value_counts().to_dict(), "representation_hpo": "pre-2026 blocked sweep over denoising-AE bottlenecks 4/8/12, fixed noise 0.05 and 8 epochs", "gmm_hpo": "same blocked training segment; full-covariance GMM k=3..6 with sticky prior sweep", "gmm_input": "frozen DAE latent vector plus train-standardized reconstruction error", "persistent_state_gate": "median dwell >=6h and hourly temporal switching <=10%", "ood": "train 99th percentile of GMM negative log density and DAE reconstruction error", "transition_outputs_excluded": True, "state_identity": "frozen-model-local; never equated to transition state or other regime state IDs"}, indent=2) + "\n")
        files = [path for path in temporary.iterdir() if path.is_file()]
        passed = bool(structural.loc[structural.arm.eq("dae_to_fullcov_gmm_challenger_v1"), "persistent_state_gate_passed"].iloc[0])
        manifest = {"schema": "strict_forward_dae_gmm_regime_challenger_v1", "status": "SEALED_STRICT_FORWARD_CHALLENGER" if passed else "SEALED_STRICT_FORWARD_CHALLENGER_REJECTED", "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "training_start_utc": str(train.source_utc.min()), "training_end_exclusive_utc": CUT.isoformat(), "train_rows": len(train), "eval_start_utc": CUT.isoformat(), "eval_end_utc": str(test.source_utc.max()), "eval_rows": len(test), "strict_split_contract": "all feature selection, DAE representation/HPO, GMM geometry, persistence, OOD thresholds and semantics use 2022-2025 only; 2026 is untouched assessment", "same_hourly_panel_as_both_gmm_baselines": True, "transition_outputs_excluded": True, "test_predictive_log_score": predictive_score, "test_temporal_switch_rate": switching, "persistent_state_gate_passed": passed, "promotion_eligible": False, "economic_attribution_scope": "exact candidate economics only; never used in representation, HPO, state selection, persistence or OOD", "inputs": {str(PANEL): sha(PANEL), str(DIAGONAL / "manifest.json"): sha(DIAGONAL / "manifest.json"), str(STICKY_FULLCOV / "manifest.json"): sha(STICKY_FULLCOV / "manifest.json"), str(ECON): sha(ECON)}, "outputs_sha256": {path.name: sha(path) for path in files}}
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
