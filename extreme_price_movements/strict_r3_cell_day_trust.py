"""Frozen R5 residual-trust model for canonical strict-R3 admission.

The model learns the error of the causal 28-day Cell-day expected-net map from
prior-resolved rows.  Its historical integration was demotion-only.  The
canonical 9-month integration uses the model's posterior expected policy net
as the fail-closed admission and auction value.  Both integrations remain
explicitly versioned so old artifacts remain reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor

from .trust_sizing_ablation import (
    CMIEdge,
    RobustTransform,
    cmi_weights,
    discover_cmi_edges,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONTRACT_PATH = ROOT / "config" / "strict_r3_cell_day_residual_trust_overlay_v1.json"
POSTERIOR_CONTRACT_PATH = ROOT / "config" / "strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1.json"
POSTERIOR_MODEL_CONTRACT_PATH = ROOT / "config" / "strict_r3_cell_day_residual_trust_model_r5_9m_v1.json"
# Backward-compatible name used by historical callers.
CONTRACT_PATH = MODEL_CONTRACT_PATH
SCHEMA = "strict_r3_cell_day_residual_trust_bundle_v1"
MAP_FIELD = "raw_expected_bps"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract(path: Path = MODEL_CONTRACT_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "strict_r3_cell_day_residual_trust_overlay_v1":
        raise ValueError("not the canonical R5 residual-trust contract")
    fields = tuple(map(str, value.get("features", ())))
    if len(fields) != 66 or len(set(fields)) != 66:
        raise ValueError("canonical R5 requires exactly 66 unique fields")
    if any(field.startswith("k09__cluster_") for field in fields):
        raise ValueError("canonical R5 prohibits raw K9 memberships")
    if value.get("admission_changes") is not False:
        raise ValueError("canonical R5 may not change admission")
    return value


def _posterior_contract(path: Path = POSTERIOR_CONTRACT_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1":
        raise ValueError("not the canonical R5 posterior-admission contract")
    if int(value.get("ev_map", {}).get("window_calendar_days", -1)) != 28:
        raise ValueError("canonical R5 posterior admission requires the 28-day map")
    if int(value.get("training", {}).get("window_months", -1)) != 9:
        raise ValueError("canonical R5 posterior admission requires nine training months")
    integration = value.get("integration", {})
    if integration.get("missing_posterior") != "fail_closed":
        raise ValueError("canonical R5 posterior admission must fail closed")
    if integration.get("changes_canonical_admission") is not True:
        raise ValueError("posterior contract must explicitly own canonical admission")
    return value


def _model_contract(path: Path) -> dict[str, Any]:
    selected = json.loads(Path(path).read_text())
    if selected.get("schema") == "strict_r3_cell_day_residual_trust_overlay_v1":
        return _contract(Path(path))
    if selected.get("schema") != "strict_r3_cell_day_residual_trust_model_r5_9m_v1":
        raise ValueError("unknown R5 model contract")
    base = _contract(MODEL_CONTRACT_PATH)
    if selected.get("base_model_contract") != base["schema"]:
        raise ValueError("R5 9-month model contract has the wrong base contract")
    if selected.get("feature_order") != "validated_fold_order_active_rule_before_k9_summary_v1":
        raise ValueError("R5 9-month model contract has the wrong feature order")
    tail = [
        "k9_entropy", "k9_top2_margin", "k9_ood_distance",
        "k9_path_support_effective_28d", "k9_model_ood_marginal",
        "k9_model_drift_psi",
    ]
    fields = [field for field in base["features"] if field not in tail] + tail
    if len(fields) != 66 or len(set(fields)) != 66:
        raise ValueError("R5 9-month model contract must resolve to 66 unique fields")
    return {**base, "schema": selected["schema"], "features": fields}


def _interactions(x: np.ndarray, fields: Sequence[str], edges: Sequence[CMIEdge]) -> np.ndarray:
    if not edges:
        return np.empty((len(x), 0), dtype=np.float32)
    index = {field: position for position, field in enumerate(fields)}
    return np.column_stack([
        x[:, index[edge.left]] * x[:, index[edge.right]] for edge in edges
    ]).astype(np.float32)


def _timestamp_top30(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    selected = position < np.maximum(1, np.ceil(size * 0.30).astype(int))
    return pd.Series(selected, index=ordered.index).reindex(frame.index).to_numpy(bool)


def _equal_month(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    month = frame["__decision_ts__"].dt.to_period("M")
    groups = list(frame.groupby(month, sort=True))
    quota = max(1, cap // len(groups))
    selected = [
        part.sort_values("candidate_id", kind="stable").head(quota)
        for _token, part in groups
    ]
    result = pd.concat(selected)
    if len(result) < cap:
        remainder = frame.drop(index=result.index).sort_values("candidate_id", kind="stable")
        result = pd.concat([result, remainder.head(cap - len(result))])
    return result.head(cap).copy()


@dataclass
class CellDayResidualTrustBundle:
    cutoff: pd.Timestamp
    fields: tuple[str, ...]
    transform: RobustTransform
    edges: tuple[CMIEdge, ...]
    model: RandomForestRegressor
    leaf_statistics: tuple[dict[int, tuple[float, ...]], ...]
    global_mean: float
    global_q10: float
    global_q25: float
    global_probabilities: tuple[float, float, float]
    residual_noise: float
    manifest: dict[str, Any]

    @property
    def schema(self) -> str:
        return SCHEMA

    def _local_distribution(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = self.transform.transform(frame)
        design = np.hstack([raw, _interactions(raw, self.fields, self.edges)])
        leaves = self.model.apply(design)
        rows, trees = leaves.shape
        values = {
            key: np.empty((rows, trees), dtype=np.float32)
            for key in ("support", "mean", "q10", "q25", "p50", "p100", "p200")
        }
        for tree_index, table in enumerate(self.leaf_statistics):
            local = np.asarray([table[int(leaf)] for leaf in leaves[:, tree_index]], dtype=np.float32)
            for offset, key in enumerate(values):
                values[key][:, tree_index] = local[:, offset]
        support = np.median(values["support"], axis=1)
        shrink = support / (support + 300.0)
        priors = {
            "mean": self.global_mean,
            "q10": self.global_q10,
            "q25": self.global_q25,
            "p50": self.global_probabilities[0],
            "p100": self.global_probabilities[1],
            "p200": self.global_probabilities[2],
        }
        result = {"support": support, "shrink": shrink}
        for key, prior in priors.items():
            result[key] = shrink * values[key].mean(axis=1) + (1.0 - shrink) * prior
        result["mean_sd"] = values["mean"].std(axis=1)
        return result

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"candidate_id", MAP_FIELD, *self.fields}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"canonical R5 scoring frame lacks fields: {missing}")
        local = self._local_distribution(frame)
        expected = pd.to_numeric(frame[MAP_FIELD], errors="raise").to_numpy(float)
        posterior = expected + local["mean"]
        predictive_sd = np.sqrt(local["mean_sd"] ** 2 + self.residual_noise**2)
        q10 = posterior + student_t.ppf(0.10, df=5.0) * predictive_sd
        p_positive = 1.0 - student_t.cdf((0.0 - posterior) / predictive_sd, df=5.0)
        p_adverse = student_t.cdf((-200.0 - posterior) / predictive_sd, df=5.0)
        contract = _contract(MODEL_CONTRACT_PATH)
        gate = contract["corroboration"]
        corroborated = (
            (local["p100"] >= float(gate["p_overestimate_100_min"]))
            & (local["q25"] <= float(gate["residual_q25_max_bps"]))
            & (local["support"] >= float(gate["effective_support_min"]))
        )
        authority_contract = contract["authority"]
        probability_confidence = np.clip((local["p100"] - 0.50) / 0.50, 0.0, 1.0)
        quantile_severity = np.clip((-local["q25"] - 25.0) / 175.0, 0.0, 1.0)
        authority = float(authority_contract["cap"]) * local["shrink"] * np.sqrt(
            probability_confidence * quantile_severity
        )
        authority = np.where(corroborated, authority, 0.0)
        corrected = expected + authority * np.minimum(posterior - expected, 0.0)
        return pd.DataFrame({
            "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
            "trust_posterior_expected_bps": posterior.astype(np.float32),
            "trust_posterior_predictive_q10_bps": q10.astype(np.float32),
            "trust_p_ev_positive": p_positive.astype(np.float32),
            "trust_p_adverse_200bps": p_adverse.astype(np.float32),
            "trust_effective_support": local["support"].astype(np.float32),
            "trust_residual_q25_bps": local["q25"].astype(np.float32),
            "trust_p_map_overestimate_100bps": local["p100"].astype(np.float32),
            "trust_risk_corroborated": corroborated,
            "trust_authority": authority.astype(np.float32),
            # Preserve the exact demotion-only inequality at the auction
            # boundary.  Float32 rounding can otherwise move an unchanged
            # negative EV a few micro-bps above its source map value.
            "trust_corrected_expected_net_bps": corrected.astype(np.float64),
            "auction_rank_adjustment_bps": (corrected - expected).astype(np.float64),
        })


def train_cell_day_residual_trust_bundle(
    ledger: pd.DataFrame,
    *,
    cutoff: object,
    source_hashes: Mapping[str, Any] | None = None,
    integration_contract_path: Path | None = POSTERIOR_CONTRACT_PATH,
) -> CellDayResidualTrustBundle:
    integration_contract = (
        None if integration_contract_path is None
        else _posterior_contract(Path(integration_contract_path))
    )
    model_contract_path = (
        MODEL_CONTRACT_PATH
        if integration_contract is None else POSTERIOR_MODEL_CONTRACT_PATH
    )
    contract = _model_contract(model_contract_path)
    if integration_contract is not None and integration_contract.get("model_contract") != contract["schema"]:
        raise ValueError("posterior integration and R5 model contracts disagree")
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts.tz_convert("UTC")
    fields = tuple(map(str, contract["features"]))
    required = {
        "candidate_id", "__decision_ts__", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", MAP_FIELD, *fields,
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"canonical R5 training ledger lacks fields: {missing}")
    work = ledger.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True, errors="raise",
    )
    training = (
        contract["training"] if integration_contract is None
        else integration_contract["training"]
    )
    start = cutoff_ts - pd.DateOffset(months=int(training["window_months"]))
    mapped_history_start = work.loc[
        np.isfinite(pd.to_numeric(work[MAP_FIELD], errors="coerce")),
        "__decision_ts__",
    ].min()
    if pd.isna(mapped_history_start) or pd.Timestamp(mapped_history_start) > start:
        raise ValueError(
            "canonical R5 has insufficient resolved prior support: complete "
            f"{int(training['window_months'])}-calendar-month mapped history required"
        )
    train_all = work.loc[
        work["__decision_ts__"].ge(start)
        & work["__decision_ts__"].lt(cutoff_ts)
        & work["policy_label_available_ts"].lt(cutoff_ts)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(work[MAP_FIELD], errors="coerce"))
    ].copy()
    if len(train_all) < 2_000:
        raise ValueError("canonical R5 has insufficient resolved prior support")
    train = _equal_month(
        train_all.loc[_timestamp_top30(train_all)].copy(),
        int(training["row_cap"]),
    )
    if train.loc[:, fields].notna().mean().min() < 0.90:
        raise ValueError("canonical R5 feature coverage falls below 90%")
    cmi_source = train.loc[
        pd.to_numeric(train["final_score"], errors="coerce").ge(
            pd.to_numeric(train["final_score"], errors="coerce").quantile(0.80)
        )
    ].copy()
    edges, _ = discover_cmi_edges(
        cmi_source, fields, mode="rank_loss_false_positive", stable=True,
        max_edges=8, sample_cap=40_000,
    )
    transform = RobustTransform.fit(train, fields)
    raw = transform.transform(train)
    design = np.hstack([raw, _interactions(raw, fields, edges)])
    realised = pd.to_numeric(train["policy_net_bps"], errors="raise").to_numpy(float)
    expected = pd.to_numeric(train[MAP_FIELD], errors="raise").to_numpy(float)
    residual_raw = realised - expected
    residual = np.clip(residual_raw, -500.0, 500.0)
    weight = cmi_weights(train, "rank_loss_false_positive")
    model = RandomForestRegressor(
        n_estimators=64, max_depth=8, min_samples_leaf=120,
        max_features=0.70, bootstrap=True, max_samples=0.75,
        n_jobs=4, random_state=20260810,
    ).fit(design, residual, sample_weight=weight)
    leaves = model.apply(design)
    statistics: list[dict[int, tuple[float, ...]]] = []
    for tree_index in range(leaves.shape[1]):
        table: dict[int, tuple[float, ...]] = {}
        for leaf in np.unique(leaves[:, tree_index]):
            mask = leaves[:, tree_index] == leaf
            values, raw_values = residual[mask], residual_raw[mask]
            table[int(leaf)] = (
                float(len(values)), float(values.mean()),
                float(np.quantile(values, 0.10)), float(np.quantile(values, 0.25)),
                float(np.mean(raw_values <= -50.0)),
                float(np.mean(raw_values <= -100.0)),
                float(np.mean(raw_values <= -200.0)),
            )
        statistics.append(table)
    global_mean = float(np.average(residual, weights=weight))
    global_q10, global_q25 = np.quantile(residual, [0.10, 0.25])
    probabilities = tuple(float(np.average(residual_raw <= -value, weights=weight)) for value in (50, 100, 200))
    shell = CellDayResidualTrustBundle(
        cutoff_ts, fields, transform, tuple(edges), model, tuple(statistics),
        global_mean, float(global_q10), float(global_q25), probabilities, 0.0, {},
    )
    local = shell._local_distribution(train)
    noise = float(np.sqrt(np.average(
        np.clip(residual_raw - local["mean"], -2_000.0, 2_000.0) ** 2,
        weights=weight,
    )))
    manifest = {
        "schema": SCHEMA,
        "cutoff": cutoff_ts.isoformat(),
        "training_start": start.isoformat(),
        "mapped_history_start": pd.Timestamp(mapped_history_start).isoformat(),
        "full_declared_history_window": True,
        "train_rows_before_selection": int(len(train_all)),
        "train_rows": int(len(train)),
        "field_count": len(fields),
        "fields": list(fields),
        "edge_count": len(edges),
        "edges": [edge.__dict__ for edge in edges],
        "target": "clip(policy_net_bps - 28d_cell_day_expected_net_bps, -500, 500)",
        "training_window_months": int(training["window_months"]),
        "model_contract": contract["schema"],
        "model_contract_path": str(model_contract_path),
        "model_contract_sha256": _sha(model_contract_path),
        "integration_contract": (
            None if integration_contract is None else integration_contract["schema"]
        ),
        "integration_contract_path": (
            None if integration_contract_path is None else str(Path(integration_contract_path))
        ),
        "integration_contract_sha256": (
            None if integration_contract_path is None else _sha(Path(integration_contract_path))
        ),
        "admission_changes": integration_contract is not None,
        "admission_value": (
            "28d_cell_day_expected_net_bps"
            if integration_contract is None else "trust_posterior_expected_bps"
        ),
        "admission_threshold_bps": (
            None if integration_contract is None
            else float(integration_contract["ev_map"]["minimum_expected_net_bps"])
        ),
        "missing_posterior": (
            None if integration_contract is None
            else integration_contract["integration"]["missing_posterior"]
        ),
        "source_hashes": dict(source_hashes or {}),
    }
    shell.residual_noise = noise
    shell.manifest = manifest
    return shell


def persist_cell_day_residual_trust_bundle(
    bundle: CellDayResidualTrustBundle, directory: Path,
) -> dict[str, Any]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable R5 bundle exists: {directory}")
    directory.mkdir(parents=True)
    payload = directory / "cell_day_residual_trust.joblib"
    joblib.dump(bundle, payload, compress=3)
    manifest = {
        **bundle.manifest,
        "bundle_file": payload.name,
        "bundle_sha256": _sha(payload),
        "contract_path": bundle.manifest.get("model_contract_path", str(MODEL_CONTRACT_PATH)),
        "contract_sha256": bundle.manifest.get("model_contract_sha256", _sha(MODEL_CONTRACT_PATH)),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_cell_day_residual_trust_bundle(directory: Path) -> CellDayResidualTrustBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("not a canonical R5 residual-trust bundle")
    payload = directory / manifest["bundle_file"]
    if _sha(payload) != manifest["bundle_sha256"]:
        raise ValueError("canonical R5 bundle hash mismatch")
    bundle = joblib.load(payload)
    contract_path = Path(manifest.get("model_contract_path", MODEL_CONTRACT_PATH))
    contract = _model_contract(contract_path)
    if not isinstance(bundle, CellDayResidualTrustBundle):
        raise ValueError("canonical R5 payload type mismatch")
    if tuple(bundle.fields) != tuple(contract["features"]):
        raise ValueError("canonical R5 payload feature contract mismatch")
    return bundle


__all__ = [
    "CellDayResidualTrustBundle", "CONTRACT_PATH", "MODEL_CONTRACT_PATH",
    "POSTERIOR_CONTRACT_PATH", "POSTERIOR_MODEL_CONTRACT_PATH", "SCHEMA",
    "load_cell_day_residual_trust_bundle",
    "persist_cell_day_residual_trust_bundle",
    "train_cell_day_residual_trust_bundle",
]
