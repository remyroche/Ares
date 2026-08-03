#!/usr/bin/env python3
"""Frozen regime model: fit/select on 2022-2025 and assess once on 2026."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data_perp/artifacts/strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3"
PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
ECON = ROOT / "data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet"
START = pd.Timestamp("2022-08-30", tz="UTC")
CUT = pd.Timestamp("2026-01-01", tz="UTC")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def family(name: str) -> str:
    q = name.lower()
    if any(k in q for k in ("liquidity", "spread", "depth", "amihud")):
        return "liquidity_proxy"
    if any(k in q for k in ("corr", "covar", "depend", "dispersion")):
        return "dependence_covariance"
    if any(k in q for k in ("vol", "atr", "range")):
        return "volatility"
    return "distribution_dynamics"


def transition_counts(states: np.ndarray, segments: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    contiguous = segments[1:] == segments[:-1]
    np.add.at(counts, (states[:-1][contiguous], states[1:][contiguous]), 1.0)
    return counts


def causal_filter(
    emissions: np.ndarray, segments: np.ndarray, transition: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Filter without crossing calendar gaps and return predictive score/switch rate."""
    filtered = np.empty_like(emissions)
    filtered[0] = emissions[0] / max(float(emissions[0].sum()), 1e-12)
    log_scores: list[float] = []
    switches: list[bool] = []
    for row in range(1, len(emissions)):
        if segments[row] != segments[row - 1]:
            filtered[row] = emissions[row] / max(float(emissions[row].sum()), 1e-12)
            continue
        previous_state = int(filtered[row - 1].argmax())
        unnormalised = (filtered[row - 1] @ transition) * emissions[row]
        evidence = max(float(unnormalised.sum()), 1e-12)
        filtered[row] = unnormalised / evidence
        log_scores.append(float(np.log(evidence)))
        switches.append(int(filtered[row].argmax()) != previous_state)
    return (
        filtered,
        float(np.mean(log_scores)) if log_scores else np.nan,
        float(np.mean(switches)) if switches else np.nan,
    )


def run_lengths(states: np.ndarray, segments: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    current = 1
    for row in range(1, len(states)):
        if segments[row] != segments[row - 1] or states[row] != states[row - 1]:
            lengths.append(current)
            current = 1
        else:
            current += 1
    lengths.append(current)
    return np.asarray(lengths, dtype=int)


def maximum_absolute_correlation(
    values: np.ndarray, candidate_index: int, selected_indices: list[int]
) -> float:
    if not selected_indices:
        return 0.0
    matrix = np.column_stack(
        [values[:, candidate_index], values[:, selected_indices]]
    )
    correlations = np.corrcoef(matrix, rowvar=False)[0, 1:]
    finite = np.abs(correlations[np.isfinite(correlations)])
    return float(finite.max()) if len(finite) else 0.0


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    panel = pd.read_parquet(PANEL)
    panel["source_utc"] = pd.to_datetime(panel["source_utc"], utc=True)
    panel = panel.loc[panel["source_utc"].ge(START)].sort_values("source_utc").reset_index(drop=True)
    train = panel.loc[panel["source_utc"].lt(CUT)].copy()
    test = panel.loc[panel["source_utc"].ge(CUT)].copy()
    if train["source_utc"].max() >= CUT or test["source_utc"].min() < CUT:
        raise AssertionError("The frozen 2022-2025/2026 boundary was violated")

    # Transition-labelled fields, identifiers and nonnumeric fields are forbidden.
    candidates = [
        c
        for c in panel.columns
        if c not in ["source_utc", "calendar_segment_id"]
        and "transition" not in c.lower()
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    variance = (
        train[candidates]
        .apply(pd.to_numeric, errors="coerce")
        .var()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(ascending=False)
    )

    # Train-only, family-balanced shortlist. Winsorisation/scaling and the
    # redundancy screen are learned on training data only.
    shortlist: list[str] = []
    for feature_family in (
        "volatility",
        "liquidity_proxy",
        "dependence_covariance",
        "distribution_dynamics",
    ):
        shortlist.extend([c for c in variance.index if family(c) == feature_family][:48])
    shortlist = list(dict.fromkeys(shortlist))
    imputer0 = SimpleImputer(strategy="median")
    raw_short = imputer0.fit_transform(train[shortlist])
    low0 = np.quantile(raw_short, 0.005, axis=0)
    high0 = np.quantile(raw_short, 0.995, axis=0)
    scaler0 = RobustScaler().fit(np.clip(raw_short, low0, high0))
    scaled_short = scaler0.transform(np.clip(raw_short, low0, high0))

    chosen_indices: list[int] = []
    for feature_family in (
        "volatility",
        "liquidity_proxy",
        "dependence_covariance",
        "distribution_dynamics",
    ):
        for index, name in enumerate(shortlist):
            if family(name) != feature_family:
                continue
            if maximum_absolute_correlation(
                scaled_short, index, chosen_indices
            ) >= 0.95:
                continue
            chosen_indices.append(index)
            if sum(family(shortlist[i]) == feature_family for i in chosen_indices) >= 16:
                break
    for index in range(len(shortlist)):
        if len(chosen_indices) >= 64:
            break
        if index in chosen_indices:
            continue
        if maximum_absolute_correlation(
            scaled_short, index, chosen_indices
        ) < 0.95:
            chosen_indices.append(index)
    features = [shortlist[i] for i in chosen_indices]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    raw_train = imputer.fit_transform(train[features])
    raw_test = imputer.transform(test[features])
    lower = np.quantile(raw_train, 0.005, axis=0)
    upper = np.quantile(raw_train, 0.995, axis=0)
    x_train = scaler.fit_transform(np.clip(raw_train, lower, upper))
    x_test = scaler.transform(np.clip(raw_test, lower, upper))

    geometry_sweep: list[dict[str, float | int]] = []
    best: tuple[float, GaussianMixture] | None = None
    for components in [3, 4, 5, 6]:
        model = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=3,
            random_state=1729,
            max_iter=250,
        ).fit(x_train)
        bic = float(model.bic(x_train))
        geometry_sweep.append({"components": components, "train_bic": bic})
        if best is None or bic < best[0]:
            best = (bic, model)
    assert best is not None
    gmm = best[1]
    train_emissions = gmm.predict_proba(x_train)
    train_states = train_emissions.argmax(axis=1)
    train_segments = train["calendar_segment_id"].astype(str).to_numpy()

    # Select persistence only on the final blocked 20% of 2022-2025. Transition
    # counts come only from the preceding training block and never cross gaps.
    holdout_rows = max(1000, len(x_train) // 5)
    split = len(x_train) - holdout_rows
    pre_counts = transition_counts(
        train_states[:split], train_segments[:split], gmm.n_components
    )
    persistence_sweep: list[dict[str, float | int]] = []
    best_persistence: tuple[float, float] | None = None
    for sticky in [1.0, 10.0, 50.0, 200.0, 1000.0]:
        transition = pre_counts + sticky * np.eye(gmm.n_components)
        transition /= transition.sum(axis=1, keepdims=True)
        _, predictive_log_score, temporal_switch_rate = causal_filter(
            train_emissions[split:], train_segments[split:], transition
        )
        objective = predictive_log_score - 0.05 * temporal_switch_rate
        persistence_sweep.append(
            {
                "sticky_prior": sticky,
                "blocked_predictive_log_score": predictive_log_score,
                "blocked_temporal_switch_rate": temporal_switch_rate,
                "blocked_objective": objective,
            }
        )
        if best_persistence is None or objective > best_persistence[0]:
            best_persistence = (objective, sticky)
    assert best_persistence is not None
    selected_sticky = best_persistence[1]

    # Refit the transition matrix on all 2022-2025 after selection, then apply
    # it causally to untouched 2026 with a reset at every calendar gap.
    all_counts = transition_counts(train_states, train_segments, gmm.n_components)
    frozen_transition = all_counts + selected_sticky * np.eye(gmm.n_components)
    frozen_transition /= frozen_transition.sum(axis=1, keepdims=True)
    test_emissions = gmm.predict_proba(x_test)
    raw_states = test_emissions.argmax(axis=1)
    test_segments = test["calendar_segment_id"].astype(str).to_numpy()
    filtered, test_predictive_log_score, test_switch_rate = causal_filter(
        test_emissions, test_segments, frozen_transition
    )
    states = filtered.argmax(axis=1)
    entropy = -(
        np.clip(filtered, 1e-12, 1) * np.log(np.clip(filtered, 1e-12, 1))
    ).sum(axis=1) / np.log(gmm.n_components)
    ordered = np.sort(filtered, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    ood = -gmm.score_samples(x_test)
    ood_threshold = float(np.quantile(-gmm.score_samples(x_train), 0.99))

    sidecar = test[["source_utc", "calendar_segment_id"]].copy()
    sidecar["regime_state_id_raw"] = raw_states
    sidecar["regime_state_id"] = states
    sidecar["regime_entropy"] = entropy
    sidecar["regime_margin"] = margin
    sidecar["regime_ood_score"] = ood
    sidecar["regime_is_ood"] = ood > ood_threshold
    sidecar["regime_available_utc"] = sidecar["source_utc"]
    for state in range(gmm.n_components):
        sidecar[f"regime_state_p_raw__{state}"] = test_emissions[:, state]
        sidecar[f"regime_state_p__{state}"] = filtered[:, state]

    profiles: list[dict[str, object]] = []
    for state in range(gmm.n_components):
        means = pd.Series(
            x_train[train_states == state].mean(axis=0), index=features
        )
        top = means.abs().sort_values(ascending=False).head(8)
        profiles.append(
            {
                "regime_state_id": state,
                "train_rows": int((train_states == state).sum()),
                "semantic_train_only": " | ".join(top.index),
                "top_feature_signed_robust_means": json.dumps(
                    {name: float(means[name]) for name in top.index}
                ),
            }
        )
    profile_frame = pd.DataFrame(profiles)

    sidecar["month"] = sidecar["source_utc"].dt.strftime("%Y-%m")
    same_segment = sidecar["calendar_segment_id"].eq(
        sidecar["calendar_segment_id"].shift()
    )
    sidecar["state_changed"] = (
        sidecar["regime_state_id"].ne(sidecar["regime_state_id"].shift())
        & same_segment
    )
    sidecar["raw_state_changed"] = (
        sidecar["regime_state_id_raw"].ne(sidecar["regime_state_id_raw"].shift())
        & same_segment
    )
    stability = sidecar.groupby("month", as_index=False).agg(
        rows=("source_utc", "size"),
        states=("regime_state_id", "nunique"),
        mean_entropy=("regime_entropy", "mean"),
        mean_margin=("regime_margin", "mean"),
        ood_fraction=("regime_is_ood", "mean"),
        state_change_fraction=("state_changed", "mean"),
        raw_state_change_fraction=("raw_state_changed", "mean"),
    )
    dwell = pd.DataFrame(
        [
            {
                "representation": label,
                "runs": len(lengths),
                "mean_hours": float(lengths.mean()),
                "median_hours": float(np.median(lengths)),
                "p90_hours": float(np.quantile(lengths, 0.9)),
                "max_hours": int(lengths.max()),
            }
            for label, lengths in (
                ("raw", run_lengths(raw_states, test_segments)),
                ("filtered", run_lengths(states, test_segments)),
            )
        ]
    )
    shifts: list[dict[str, object]] = []
    for name in features:
        before = pd.to_numeric(train[name], errors="coerce")
        after = pd.to_numeric(test[name], errors="coerce")
        scale = max(float(before.std()), 1e-9)
        shifts.append(
            {
                "feature": name,
                "train_mean": before.mean(),
                "eval_mean": after.mean(),
                "standardized_mean_shift": (after.mean() - before.mean()) / scale,
            }
        )

    # Exact candidate economics is attribution only and never participates in fit.
    economics = pd.read_parquet(
        ECON,
        columns=[
            "__ts__",
            "side_name",
            "catboost_archetype",
            "execution_net_ev_12h",
            "execution_gross_ev_12h",
            "execution_cost_return",
        ],
    )
    economics["__ts__"] = pd.to_datetime(economics["__ts__"], utc=True)
    attributed = economics.merge(
        sidecar.rename(columns={"source_utc": "__ts__"}),
        on="__ts__",
        how="inner",
        validate="many_to_one",
    )
    attributed["month"] = attributed["__ts__"].dt.strftime("%Y-%m")
    attribution = attributed.groupby(
        ["month", "regime_state_id", "side_name", "catboost_archetype"],
        as_index=False,
    ).agg(
        rows=("execution_net_ev_12h", "size"),
        net_ev=("execution_net_ev_12h", "mean"),
        gross_ev=("execution_gross_ev_12h", "mean"),
        cost=("execution_cost_return", "mean"),
    )

    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    try:
        sidecar.to_parquet(temporary / "regime_only_forward_2026_sidecar.parquet", index=False)
        profile_frame.to_csv(temporary / "semantic_train_only_profiles.csv", index=False)
        pd.DataFrame(geometry_sweep).to_csv(
            temporary / "train_only_geometry_sweep.csv", index=False
        )
        pd.DataFrame(persistence_sweep).to_csv(
            temporary / "train_only_persistence_sweep.csv", index=False
        )
        stability.to_csv(temporary / "2026_monthly_coverage_stability.csv", index=False)
        dwell.to_csv(temporary / "2026_dwell_stability.csv", index=False)
        pd.DataFrame(shifts).sort_values(
            "standardized_mean_shift",
            key=lambda values: values.abs(),
            ascending=False,
        ).to_csv(temporary / "2026_feature_shifts.csv", index=False)
        attribution.to_parquet(
            temporary / "2026_exact_economic_attribution_may_july.parquet",
            index=False,
        )
        (temporary / "feature_contract.json").write_text(
            json.dumps(
                {
                    "selected_train_only_features": features,
                    "selection": (
                        "train-only variance shortlist by four feature families, "
                        "0.5/99.5% winsorisation, robust scaling and abs-correlation<0.95"
                    ),
                    "selected_family_counts": pd.Series(
                        [family(name) for name in features]
                    ).value_counts().to_dict(),
                    "hpo": "train-only diagonal-GMM BIC sweep k=3..6",
                    "persistence": (
                        "final blocked 20% of 2022-2025; predictive evidence before "
                        "normalisation minus temporal switch penalty; gap resets"
                    ),
                    "selected_sticky_prior": selected_sticky,
                    "excluded_transition_fields": True,
                    "inverse_pi_jan_aug_2022": (
                        "excluded; no harmonized identical feature contract proved"
                    ),
                    "state_identity": (
                        "frozen-model-local; never equated to transition state or "
                        "another regime model's state IDs"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        files = [path for path in temporary.iterdir() if path.is_file()]
        manifest = {
            "schema": "strict_forward_regime_only_v3",
            "status": "SEALED_POST_FREEZE_2026_AUTHORITATIVE",
            "training_start_utc": str(train["source_utc"].min()),
            "training_end_exclusive_utc": CUT.isoformat(),
            "train_rows": len(train),
            "eval_start_utc": CUT.isoformat(),
            "eval_end_utc": str(test["source_utc"].max()),
            "eval_rows": len(test),
            "strict_split_contract": (
                "all feature selection, preprocessing, geometry, persistence and "
                "semantics use 2022-2025 only; 2026 is untouched assessment"
            ),
            "transition_outputs_excluded": True,
            "test_predictive_log_score": test_predictive_log_score,
            "test_temporal_switch_rate": test_switch_rate,
            "promotion_eligible": False,
            "economic_attribution_scope": (
                "exact candidate economics only May-July 2026; labels never used "
                "in state selection, HPO or persistence"
            ),
            "inputs": {str(PANEL): sha(PANEL), str(ECON): sha(ECON)},
            "outputs_sha256": {path.name: sha(path) for path in files},
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.sha256").write_text(
            f"{sha(manifest_path)}  manifest.json\n"
        )
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
