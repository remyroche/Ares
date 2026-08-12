"""Executable bounded-A5 trust layer for the canonical strict-R3 stack.

A5 is not a replacement for the causal A0/R5 estimate.  It is a bounded,
causally calibrated correction around A0 and is active only in the
timestamp-local top-15 percent of the upstream score:

    A5_10 = A0 + 0.10 * (calibrated_A4 - A0)

Admission remains A0 >= 50 bps AND timestamp-local top-15.  The A4 model is
the independently-supported, neutral-mean residual forest validated by the
longer A5 study.  Its calibration is fitted only from earlier OOS A4
predictions whose policy outcomes have resolved before the bundle cutoff.
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
from scipy.special import ndtr
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor

from .strict_r3_cell_day_trust import _model_contract
from .trust_sizing_ablation import (
    CMIEdge,
    RobustTransform,
    discover_cmi_edges,
    independent_experience_support,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONTRACT_PATH = ROOT / "config" / "strict_r3_cell_day_residual_trust_model_r5_9m_v1.json"
INTEGRATION_CONTRACT_PATH = ROOT / "config" / "strict_r3_a5_bounded_10pct_canonical_v1.json"
SCHEMA = "strict_r3_a5_bounded_trust_bundle_v1"
A4_SPEC = "R5_cell_day_residual_clip500_neutralmean_independent"
MAP_FIELD = "raw_expected_bps"
MIN_CALIBRATION_ROWS = 2_000
CALIBRATION_CAP = 120_000
SEED = 20260812


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _interactions(
    values: np.ndarray,
    fields: Sequence[str],
    edges: Sequence[CMIEdge],
) -> np.ndarray:
    if not edges:
        return np.empty((len(values), 0), dtype=np.float32)
    locations = {field: index for index, field in enumerate(fields)}
    return np.column_stack([
        values[:, locations[edge.left]] * values[:, locations[edge.right]]
        for edge in edges
    ]).astype(np.float32)


def _timestamp_top_fraction(
    frame: pd.DataFrame,
    *,
    fraction: float,
    score_field: str = "final_score",
) -> np.ndarray:
    """Deterministic contemporaneous cross-sectional domain mask."""
    required = {"candidate_id", "__decision_ts__", score_field}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"timestamp-domain frame lacks {missing}")
    ordered = frame.sort_values(
        ["__decision_ts__", score_field, "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    selected = position < np.maximum(1, np.ceil(size * float(fraction)).astype(int))
    return pd.Series(selected, index=ordered.index).reindex(frame.index).to_numpy(bool)


def _equal_month_sample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    month = frame["__decision_ts__"].dt.to_period("M")
    groups = list(frame.groupby(month, sort=True))
    quota = max(1, cap // len(groups))
    selected = [
        part.sort_values("candidate_id", kind="stable").head(quota)
        for _token, part in groups
    ]
    output = pd.concat(selected)
    if len(output) < cap:
        remaining = frame.drop(index=output.index).sort_values("candidate_id", kind="stable")
        output = pd.concat([output, remaining.head(cap - len(output))])
    return output.head(cap).copy()


def _equal_month_calibration_cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Match the validated A5 calibrator's evenly spaced monthly cap."""
    if len(frame) <= cap:
        return frame.copy()
    month = frame["__decision_ts__"].dt.strftime("%Y-%m")
    groups = list(frame.groupby(month, sort=True))
    quota = max(1, cap // len(groups))
    selected: list[pd.DataFrame] = []
    for _token, part in groups:
        if len(part) <= quota:
            selected.append(part)
        else:
            index = np.linspace(0, len(part) - 1, quota).round().astype(int)
            selected.append(part.iloc[index])
    return pd.concat(selected, ignore_index=True).iloc[:cap].copy()


def _mixed_top30_reference(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    top = _timestamp_top_fraction(frame, fraction=0.30)
    top_cap = int(round(cap * 0.75))
    reference_cap = cap - top_cap
    return pd.concat([
        _equal_month_sample(frame.loc[top].copy(), top_cap),
        _equal_month_sample(frame.loc[~top].copy(), reference_cap),
    ], ignore_index=False)


@dataclass
class A4IndependentResidualBundle:
    cutoff: pd.Timestamp
    fields: tuple[str, ...]
    transform: RobustTransform
    edges: tuple[CMIEdge, ...]
    model: RandomForestRegressor
    leaf_statistics: tuple[dict[int, tuple[float, ...]], ...]
    global_statistics: tuple[float, ...]
    manifest: dict[str, Any]

    def _distribution(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = self.transform.transform(frame)
        design = np.hstack([raw, _interactions(raw, self.fields, self.edges)])
        leaves = self.model.apply(design)
        names = ("support", "mean", "q10", "q25", "p50", "p100", "p200", "variance")
        values = {
            name: np.empty(leaves.shape, dtype=np.float32) for name in names
        }
        for tree_index, table in enumerate(self.leaf_statistics):
            local = np.asarray(
                [table[int(leaf)] for leaf in leaves[:, tree_index]], dtype=np.float32,
            )
            for offset, name in enumerate(names):
                values[name][:, tree_index] = local[:, offset]
        support = np.median(values["support"], axis=1)
        authority = support / (support + 300.0)
        result: dict[str, np.ndarray] = {"support": support, "authority": authority}
        for offset, name in enumerate(names[1:]):
            result[name] = (
                authority * values[name].mean(axis=1)
                + (1.0 - authority) * self.global_statistics[offset]
            )
        result["mean_sd"] = values["mean"].std(axis=1)
        return result

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"candidate_id", MAP_FIELD, *self.fields}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"A4 scoring frame lacks {missing}")
        local = self._distribution(frame)
        anchor = pd.to_numeric(frame[MAP_FIELD], errors="raise").to_numpy(float)
        expected = anchor + local["mean"]
        predictive_sd = np.sqrt(np.maximum(local["mean_sd"] ** 2 + local["variance"], 1.0))
        return pd.DataFrame({
            "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
            "a4_raw_expected_bps": expected.astype(np.float32),
            "a4_raw_predictive_sd_bps": predictive_sd.astype(np.float32),
            "a4_effective_support": local["support"].astype(np.float32),
            "a4_p_ev_positive_raw": (
                1.0 - student_t.cdf((0.0 - expected) / predictive_sd, df=5.0)
            ).astype(np.float32),
        })


@dataclass(frozen=True)
class A5CausalCalibration:
    cutoff: pd.Timestamp
    slope: float
    intercept: float
    predictive_sd_scale: float
    prior_oos_rows: int
    status: str
    source_hashes: tuple[tuple[str, str], ...]

    def apply(self, raw_expected: Sequence[float], raw_sd: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        mean = self.intercept + self.slope * np.asarray(raw_expected, dtype=float)
        sd = np.maximum(np.asarray(raw_sd, dtype=float) * self.predictive_sd_scale, 1.0)
        return mean, ndtr(mean / sd)


def train_a4_bundle(
    ledger: pd.DataFrame,
    *,
    cutoff: object,
    source_hashes: Mapping[str, Any] | None = None,
) -> A4IndependentResidualBundle:
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts.tz_convert("UTC")
    fields = tuple(map(str, _model_contract(MODEL_CONTRACT_PATH)["features"]))
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "final_score",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        MAP_FIELD, *fields,
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"A4 training ledger lacks {missing}")
    work = ledger.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True, errors="raise",
    )
    start = cutoff_ts - pd.DateOffset(months=9)
    mapped_start = work.loc[
        np.isfinite(pd.to_numeric(work[MAP_FIELD], errors="coerce")), "__decision_ts__",
    ].min()
    if pd.isna(mapped_start) or pd.Timestamp(mapped_start) > start:
        raise ValueError("A4 requires a complete nine-calendar-month mapped-history window")
    eligible = work.loc[
        work["__decision_ts__"].ge(start)
        & work["__decision_ts__"].lt(cutoff_ts)
        & work["policy_label_available_ts"].lt(cutoff_ts)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(work[MAP_FIELD], errors="coerce"))
    ].copy()
    if len(eligible) < 2_000:
        raise ValueError("A4 has insufficient prior resolved support")
    train = _mixed_top30_reference(eligible, 60_000)
    if train.loc[:, fields].notna().mean().min() < 0.90:
        raise ValueError("A4 feature coverage falls below 90%")
    cmi_source = train.loc[
        pd.to_numeric(train["final_score"], errors="coerce").ge(
            pd.to_numeric(train["final_score"], errors="coerce").quantile(0.80)
        )
    ]
    edges, _ = discover_cmi_edges(
        cmi_source, fields, mode="rank_loss_false_positive", stable=True,
        max_edges=8, sample_cap=40_000,
    )
    transform = RobustTransform.fit(train, fields)
    raw = transform.transform(train)
    design = np.hstack([raw, _interactions(raw, fields, edges)])
    realised = pd.to_numeric(train["policy_net_bps"], errors="raise").to_numpy(float)
    anchor = pd.to_numeric(train[MAP_FIELD], errors="raise").to_numpy(float)
    residual_raw = realised - anchor
    residual = np.clip(residual_raw, -500.0, 500.0)
    weight = np.ones(len(train), dtype=float)
    model = RandomForestRegressor(
        n_estimators=64, max_depth=8, min_samples_leaf=120,
        max_features=0.70, bootstrap=True, max_samples=0.75,
        n_jobs=4, random_state=SEED,
    ).fit(design, residual, sample_weight=weight)
    leaves = model.apply(design)
    statistics: list[dict[int, tuple[float, ...]]] = []
    for tree_index in range(leaves.shape[1]):
        table: dict[int, tuple[float, ...]] = {}
        for leaf in np.unique(leaves[:, tree_index]):
            mask = leaves[:, tree_index] == leaf
            values = residual[mask]
            raw_values = residual_raw[mask]
            table[int(leaf)] = (
                independent_experience_support(train, mask, weight),
                float(values.mean()),
                float(np.quantile(values, 0.10)),
                float(np.quantile(values, 0.25)),
                float(np.mean(raw_values <= -50.0)),
                float(np.mean(raw_values <= -100.0)),
                float(np.mean(raw_values <= -200.0)),
                float(np.var(values)),
            )
        statistics.append(table)
    global_statistics = (
        float(residual.mean()), float(np.quantile(residual, 0.10)),
        float(np.quantile(residual, 0.25)),
        float(np.mean(residual_raw <= -50.0)),
        float(np.mean(residual_raw <= -100.0)),
        float(np.mean(residual_raw <= -200.0)), float(np.var(residual)),
    )
    manifest = {
        "schema": SCHEMA,
        "component": "A4_independent_local",
        "spec": A4_SPEC,
        "cutoff": cutoff_ts.isoformat(),
        "training_start": start.isoformat(),
        "mapped_history_start": pd.Timestamp(mapped_start).isoformat(),
        "training_window_months": 9,
        "train_rows_before_selection": int(len(eligible)),
        "train_rows": int(len(train)),
        "training_selection": "75pct_timestamp_top30_plus_25pct_lower_reference",
        "mean_weighting": "uniform",
        "support_mode": "independent_experience",
        "uncertainty_mode": "local_leaf",
        "target": "clip(policy_net_bps - 28d_cell_day_expected_net_bps, -500, 500)",
        "fields": list(fields), "field_count": len(fields),
        "edges": [edge.__dict__ for edge in edges], "edge_count": len(edges),
        "source_hashes": dict(source_hashes or {}),
        "raw_k9_memberships_used": False,
    }
    return A4IndependentResidualBundle(
        cutoff_ts, fields, transform, tuple(edges), model,
        tuple(statistics), global_statistics, manifest,
    )


def fit_a5_calibration(
    oos_predictions: pd.DataFrame,
    *,
    cutoff: object,
    source_hashes: Mapping[str, str] | None = None,
) -> A5CausalCalibration:
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts.tz_convert("UTC")
    required = {
        "__decision_ts__", "policy_label_available_ts", "policy_path_valid",
        "policy_net_bps", "posterior_expected_bps", "posterior_predictive_sd",
    }
    missing = sorted(required.difference(oos_predictions.columns))
    if missing:
        raise ValueError(f"A5 calibration OOS ledger lacks {missing}")
    prior = oos_predictions.copy()
    prior["__decision_ts__"] = pd.to_datetime(prior["__decision_ts__"], utc=True, errors="raise")
    prior["policy_label_available_ts"] = pd.to_datetime(
        prior["policy_label_available_ts"], utc=True, errors="raise",
    )
    prior = prior.loc[
        prior["__decision_ts__"].lt(cutoff_ts)
        & prior["policy_label_available_ts"].lt(cutoff_ts)
        & prior["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(prior["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(prior["posterior_expected_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(prior["posterior_predictive_sd"], errors="coerce"))
    ].copy()
    prior = _equal_month_calibration_cap(prior, CALIBRATION_CAP)
    if len(prior) < MIN_CALIBRATION_ROWS:
        raise ValueError("A5 requires at least 2,000 prior resolved OOS A4 predictions")
    x = pd.to_numeric(prior["posterior_expected_bps"], errors="raise").to_numpy(float)
    y = pd.to_numeric(prior["policy_net_bps"], errors="raise").to_numpy(float)
    model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=250).fit(x[:, None], y)
    slope = float(model.coef_[0])
    if not np.isfinite(slope) or slope < 0.0:
        raise ValueError("A5 calibration rejected a non-monotonic fit")
    intercept = float(model.intercept_)
    fitted = intercept + slope * x
    raw_sd = np.maximum(
        pd.to_numeric(prior["posterior_predictive_sd"], errors="raise").to_numpy(float), 1.0,
    )
    z80 = float(np.quantile(np.abs(y - fitted) / raw_sd, 0.80, method="linear"))
    scale = float(np.clip(z80 / 1.2815515655446004, 0.25, 4.0))
    return A5CausalCalibration(
        cutoff=cutoff_ts, slope=slope, intercept=intercept,
        predictive_sd_scale=scale, prior_oos_rows=int(len(prior)),
        status="prior_oos_huber_and_80pct_scale",
        source_hashes=tuple(sorted((source_hashes or {}).items())),
    )


def apply_a5_bounded_10pct(
    frame: pd.DataFrame,
    *,
    a0_expected_field: str = "trust_posterior_expected_bps",
    a4_expected_field: str = "a4_raw_expected_bps",
    a4_sd_field: str = "a4_raw_predictive_sd_bps",
    calibration: A5CausalCalibration,
) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "final_score",
        a0_expected_field, a4_expected_field, a4_sd_field,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"A5 integration frame lacks {missing}")
    a0 = pd.to_numeric(frame[a0_expected_field], errors="coerce").to_numpy(float)
    a4_raw = pd.to_numeric(frame[a4_expected_field], errors="coerce").to_numpy(float)
    a4_sd = pd.to_numeric(frame[a4_sd_field], errors="coerce").to_numpy(float)
    calibrated, probability = calibration.apply(a4_raw, a4_sd)
    domain = _timestamp_top_fraction(frame, fraction=0.15, score_field="final_score")
    available = np.isfinite(a0) & np.isfinite(calibrated) & np.isfinite(probability)
    expected = a0 + 0.10 * (calibrated - a0)
    admitted = available & (a0 >= 50.0) & domain
    return pd.DataFrame({
        "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
        "a5_calibrated_expected_bps": calibrated.astype(np.float32),
        "a5_calibrated_p_positive": probability.astype(np.float32),
        "a5_bounded10_expected_bps": expected.astype(np.float32),
        "a5_timestamp_top15": domain,
        "a5_bounded10_available": available,
        "a5_bounded10_admitted": admitted,
    })


def persist_a5_bundle(
    a4: A4IndependentResidualBundle,
    calibration: A5CausalCalibration,
    directory: Path,
) -> dict[str, Any]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable A5 bundle exists: {directory}")
    directory.mkdir(parents=True)
    model_path = directory / "a4_independent_residual.joblib"
    calibration_path = directory / "a5_causal_calibration.joblib"
    joblib.dump(a4, model_path, compress=3)
    joblib.dump(calibration, calibration_path, compress=3)
    manifest = {
        **a4.manifest,
        "schema": SCHEMA,
        "integration_contract": json.loads(INTEGRATION_CONTRACT_PATH.read_text())["schema"],
        "integration_contract_path": str(INTEGRATION_CONTRACT_PATH),
        "integration_contract_sha256": _sha(INTEGRATION_CONTRACT_PATH),
        "a4_bundle_file": model_path.name,
        "a4_bundle_sha256": _sha(model_path),
        "calibration_file": calibration_path.name,
        "calibration_sha256": _sha(calibration_path),
        "calibration": {
            "cutoff": calibration.cutoff.isoformat(),
            "slope": calibration.slope,
            "intercept": calibration.intercept,
            "predictive_sd_scale": calibration.predictive_sd_scale,
            "prior_oos_rows": calibration.prior_oos_rows,
            "status": calibration.status,
        },
        "bounded_alpha": 0.10,
        "domain": "timestamp_local_top15_by_final_score",
        "admission": "A0_expected_ge_50_bps_AND_timestamp_local_top15",
        "A5_may_change_admission_relative_to_A0_top15": False,
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_a5_bundle(directory: Path) -> tuple[A4IndependentResidualBundle, A5CausalCalibration]:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("not a canonical bounded-A5 bundle")
    model_path = directory / manifest["a4_bundle_file"]
    calibration_path = directory / manifest["calibration_file"]
    if _sha(model_path) != manifest["a4_bundle_sha256"]:
        raise ValueError("A4 bundle hash mismatch")
    if _sha(calibration_path) != manifest["calibration_sha256"]:
        raise ValueError("A5 calibration hash mismatch")
    a4 = joblib.load(model_path)
    calibration = joblib.load(calibration_path)
    if not isinstance(a4, A4IndependentResidualBundle):
        raise ValueError("A4 payload type mismatch")
    if not isinstance(calibration, A5CausalCalibration):
        raise ValueError("A5 calibration payload type mismatch")
    if a4.cutoff != calibration.cutoff:
        raise ValueError("A4 and A5 calibration cutoffs differ")
    return a4, calibration


__all__ = [
    "A4IndependentResidualBundle", "A5CausalCalibration", "A4_SPEC", "SCHEMA",
    "apply_a5_bounded_10pct", "fit_a5_calibration", "load_a5_bundle",
    "persist_a5_bundle", "train_a4_bundle",
]
