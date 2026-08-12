#!/usr/bin/env python3
"""Causal bounded-MLP trust × market-context ablation for the C3 long stack.

This is a deliberately small neural overlay, not a second alpha stack.  It
starts from the *already scored* schema-v3 canonical C3 predictions and learns
only a bounded reliability multiplier for the current
``raw_correctness_demote`` score:

    lambda = clip(0.25 + P(policy residual > +100 bps), 0.25, 1.25)
    raw_mlp = raw_correctness_demote * lambda

The MLP is trained on only prior, resolved policy outcomes.  Its inputs are
the canonical score geometry and aggregate C3/leaf trust state.  The market
feature family is deliberately small and may either enter as main effects or
only through CMI-selected trust × market interactions:

* M0: trust only;
* M1: trust + market main effects;
* M2: trust + CMI-selected interactions (no market main-effect route);
* M3: trust + market main effects + interactions.

The market/interaction contract is discovered once using 2025 rows whose
labels resolved before 2025-02-26.  It is fixed for all 2025 development
blocks and the 2026 confirmation.  This prevents the MLP from quietly turning
every conversion block into a fresh feature-selection exercise.

The MLP itself is intentionally light: 24 -> 8 hidden units, CPU training,
robust training-only scaling, chronological early stopping, L2 weight decay,
and independent family dropout for support, OOD/drift, market, and interaction
families.  The canonical C3 final score is the matched control.  Every score
arm is evaluated by global rank before future outcome coverage, then through
the same causal 21/42/84-day admission and portfolio auction.

This is a development/confirmation research ablation.  The 2025 source period
is used to select the one MLP feature contract; 2026 is a reused historical
confirmation, not an untouched final test.  No canonical promotion occurs
inside this runner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, mutual_info_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec  # noqa: E402
from extreme_price_movements.strict_r3_canonical_v2 import ScoreReference  # noqa: E402
from scripts.replay_strict_r3_forward_portfolio import _auction_candidates  # noqa: E402
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402


SCHEMA = "c3_bounded_mlp_trust_ablation_v1"
SEED = 20260810
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
DISCOVERY_CUTOFF = pd.Timestamp("2025-02-26", tz="UTC")
DEV_EVALUATION_START = pd.Timestamp("2025-04-23", tz="UTC")
DEV_EVALUATION_END = pd.Timestamp("2025-08-01", tz="UTC")
CONFIRM_EVALUATION_START = pd.Timestamp("2026-04-23", tz="UTC")
CONFIRM_EVALUATION_END = pd.Timestamp("2026-08-01", tz="UTC")
TRAIN_MONTHS = 6
REFERENCE_DAYS = 42
LABEL_HORIZON = pd.Timedelta(hours=12)
MIN_TRAIN_ROWS = 20_000
MAX_TRAIN_ROWS = 72_000
MAX_MARKET_FIELDS = 12
MAX_INTERACTIONS = 6

SOURCE_PANEL = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_source_panel_targetfree_long_2023_aug7_2026_20260809_v2/"
    "canonical_source_panel.parquet"
)
PREDICTION_PATHS = {
    2025: ROOT / (
        "data_perp/artifacts/"
        "strict_r3_self_distillation_combined_base_d2_residual_d0_"
        "fullstack_long_2025_janjul_20260810_v1/predictions.parquet"
    ),
    2026: ROOT / (
        "data_perp/artifacts/"
        "strict_r3_self_distillation_combined_base_d2_residual_d0_"
        "fullstack_long_2026_janjul_exact_policy_20260810_v1/predictions.parquet"
    ),
}
DEFAULT_OUT = ROOT / "data_perp/artifacts/c3_bounded_mlp_trust_ablation_20260810_v1"
DEFAULT_REPORT = ROOT / "docs/C3_BOUNDED_MLP_TRUST_ABLATION_20260810.md"

# The context pool is intentionally a compact market-state surface rather than
# another 120-field alpha bank.  Every item is present in the frozen base
# contract and is observable at the candidate decision timestamp.
MARKET_CANDIDATES = (
    "mkt_return_accel_1h",
    "mkt_ret_4h",
    "mkt_rv_4h",
    "prior_volatility",
    "negative_breadth_pct",
    "market_breadth_drawdown_from_6h_max",
    "breadth_recovery_from_6h_min",
    "breadth_chg_15m",
    "cross_asset_corr_1h",
    "state_spectral_eig_condition",
    "state_spectral_eig_gap_1_2",
    "mkt_oi_chg_accel_1h",
    "mkt_oi_flush_z_30d",
    "mkt_oi_dispersion_24h",
    "post_liquidation_rebound_score",
)

SCORE_FIELDS = (
    "base_score",
    "base_rank",
    "base_anchor_bps",
    "consensus_rank",
    "upstream",
    "correctness_raw",
    "correctness_rank",
    "raw_correctness_demote",
)
SUPPORT_FIELDS = (
    "leaf_support_effective",
    "leaf_support_p05",
    "leaf_support_p50",
    "leaf_support_adequate_fraction",
    "k9_path_support_effective_28d",
    "k9_path_support_adequate_fraction",
)
OOD_FIELDS = (
    "leaf_ood_marginal",
    "leaf_ood_joint",
    "k9_ood_distance",
    "k9_model_ood_marginal",
    "k9_model_drift_psi",
    "k9_entropy",
    "k9_top2_margin",
)
TRUST_FIELDS = (*SCORE_FIELDS, *SUPPORT_FIELDS, *OOD_FIELDS)
INTERACTION_TRUST_FIELDS = (
    "base_rank",
    "correctness_rank",
    "leaf_support_effective",
    "leaf_ood_marginal",
    "k9_path_support_effective_28d",
    "k9_model_ood_marginal",
    "k9_model_drift_psi",
    "k9_entropy",
)
ARMS = ("control", "M0_trust", "M1_market_main", "M2_trust_x_market", "M3_main_plus_interactions")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _target(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame["policy_net_bps"], errors="coerce")
        - pd.to_numeric(frame["base_anchor_bps"], errors="coerce")
        > 100.0
    ).astype(np.int8).to_numpy()


def _resolved_training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cutoff = _utc(cutoff)
    start = cutoff - pd.DateOffset(months=TRAIN_MONTHS)
    valid = (
        frame["__decision_ts__"].ge(start)
        & frame["__decision_ts__"].lt(cutoff)
        & frame["policy_label_available_ts"].lt(cutoff)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["base_anchor_bps"], errors="coerce"))
    )
    return frame.loc[valid].sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()


def _quantile_codes(values: Sequence[float], *, bins: int = 8) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    result = np.full(len(numeric), -1, dtype=np.int16)
    valid = np.isfinite(numeric.to_numpy(float))
    if int(valid.sum()) < max(50, bins * 5):
        return result
    rank = numeric.loc[valid].rank(method="average", pct=True).to_numpy(float)
    result[valid] = np.minimum(bins - 1, np.floor(rank * bins).astype(np.int16))
    return result


def _conditional_mi(feature_codes: np.ndarray, target: np.ndarray, condition_codes: np.ndarray) -> float:
    valid = (feature_codes >= 0) & (condition_codes >= 0) & np.isfinite(target)
    support = int(valid.sum())
    if support < 200:
        return 0.0
    score = 0.0
    for bucket in np.unique(condition_codes[valid]):
        local = valid & (condition_codes == bucket)
        count = int(local.sum())
        if count < 50 or np.unique(target[local]).size < 2:
            continue
        score += (count / support) * float(mutual_info_score(feature_codes[local], target[local]))
    return float(score)


@dataclass(frozen=True)
class FeatureDiscovery:
    market_fields: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    market_audit: pd.DataFrame
    interaction_audit: pd.DataFrame


def discover_feature_contract(frame: pd.DataFrame) -> FeatureDiscovery:
    """Select a small fixed market/interactions contract on resolved train rows."""
    required = {"upstream", "policy_net_bps", "base_anchor_bps", *TRUST_FIELDS, *MARKET_CANDIDATES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"feature discovery missing {missing}")
    target = _target(frame)
    condition = _quantile_codes(frame["upstream"], bins=10)
    market_codes = {field: _quantile_codes(frame[field]) for field in MARKET_CANDIDATES}
    market_rows: list[dict[str, Any]] = []
    for field in MARKET_CANDIDATES:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        market_rows.append(
            {
                "field": field,
                "conditional_mi": _conditional_mi(market_codes[field], target, condition),
                "coverage": float(np.isfinite(values).mean()),
                "nunique": int(pd.Series(values).nunique(dropna=True)),
            }
        )
    market_audit = pd.DataFrame(market_rows).sort_values(
        ["conditional_mi", "field"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    market_audit["selected"] = False
    eligible_market = market_audit.loc[
        market_audit["coverage"].ge(0.90) & market_audit["nunique"].ge(8)
    ].head(MAX_MARKET_FIELDS)
    market_audit.loc[eligible_market.index, "selected"] = True
    selected_market = tuple(eligible_market["field"].astype(str))

    trust_codes = {field: _quantile_codes(frame[field]) for field in INTERACTION_TRUST_FIELDS}
    trust_mi = {
        field: _conditional_mi(codes, target, condition)
        for field, codes in trust_codes.items()
    }
    market_mi = {
        field: _conditional_mi(codes, target, condition)
        for field, codes in market_codes.items()
    }
    interaction_rows: list[dict[str, Any]] = []
    for trust_field, trust_code in trust_codes.items():
        for market_field, market_code in market_codes.items():
            joint = np.where(
                (trust_code >= 0) & (market_code >= 0),
                trust_code.astype(np.int32) * 8 + market_code.astype(np.int32),
                -1,
            ).astype(np.int16)
            joint_mi = _conditional_mi(joint, target, condition)
            interaction_rows.append(
                {
                    "trust_field": trust_field,
                    "market_field": market_field,
                    "trust_conditional_mi": trust_mi[trust_field],
                    "market_conditional_mi": market_mi[market_field],
                    "joint_conditional_mi": joint_mi,
                    "incremental_cmi": joint_mi - max(trust_mi[trust_field], market_mi[market_field]),
                }
            )
    interaction_audit = pd.DataFrame(interaction_rows).sort_values(
        ["incremental_cmi", "joint_conditional_mi", "trust_field", "market_field"],
        ascending=[False, False, True, True], kind="stable",
    ).reset_index(drop=True)
    interaction_audit["selected"] = False
    selected: list[tuple[str, str]] = []
    seen_market: set[str] = set()
    trust_counts: dict[str, int] = {}
    for row in interaction_audit.itertuples():
        if len(selected) >= MAX_INTERACTIONS or float(row.incremental_cmi) <= 0.0:
            continue
        if str(row.market_field) in seen_market or trust_counts.get(str(row.trust_field), 0) >= 2:
            continue
        selected.append((str(row.trust_field), str(row.market_field)))
        seen_market.add(str(row.market_field))
        trust_counts[str(row.trust_field)] = trust_counts.get(str(row.trust_field), 0) + 1
    for trust_field, market_field in selected:
        interaction_audit.loc[
            interaction_audit["trust_field"].eq(trust_field)
            & interaction_audit["market_field"].eq(market_field),
            "selected",
        ] = True
    return FeatureDiscovery(
        market_fields=selected_market,
        interactions=tuple(selected),
        market_audit=market_audit,
        interaction_audit=interaction_audit,
    )


@dataclass(frozen=True)
class ArmContract:
    arm: str
    direct_fields: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]

    @property
    def source_fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.direct_fields, *(value for pair in self.interactions for value in pair)]))


def arm_contract(arm: str, discovery: FeatureDiscovery) -> ArmContract:
    if arm == "M0_trust":
        return ArmContract(arm, tuple(TRUST_FIELDS), ())
    if arm == "M1_market_main":
        return ArmContract(arm, tuple((*TRUST_FIELDS, *discovery.market_fields)), ())
    if arm == "M2_trust_x_market":
        # Market fields may reach this MLP only through the selected products.
        return ArmContract(arm, tuple(TRUST_FIELDS), discovery.interactions)
    if arm == "M3_main_plus_interactions":
        return ArmContract(arm, tuple((*TRUST_FIELDS, *discovery.market_fields)), discovery.interactions)
    raise ValueError(f"no MLP contract for {arm}")


def _family(field: str) -> str:
    if field in SCORE_FIELDS:
        return "score_geometry"
    if field in SUPPORT_FIELDS:
        return "support"
    if field in OOD_FIELDS:
        return "ood_drift"
    if field in MARKET_CANDIDATES:
        return "market"
    if field.startswith("interaction__"):
        return "interaction"
    raise ValueError(f"unclassified MLP field {field}")


@dataclass
class MatrixTransform:
    source_fields: tuple[str, ...]
    direct_fields: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    source_medians: np.ndarray
    source_iqrs: np.ndarray
    output_medians: np.ndarray
    output_iqrs: np.ndarray
    feature_names: tuple[str, ...]
    family_indices: dict[str, np.ndarray]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(self.source_fields)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        values = np.where(np.isfinite(values), values, self.source_medians[None, :])
        z = np.clip((values - self.source_medians[None, :]) / self.source_iqrs[None, :], -8.0, 8.0)
        lookup = {field: index for index, field in enumerate(self.source_fields)}
        columns = [z[:, lookup[field]] for field in self.direct_fields]
        for trust_field, market_field in self.interactions:
            columns.append(
                np.clip(z[:, lookup[trust_field]] * z[:, lookup[market_field]], -16.0, 16.0)
            )
        matrix = np.column_stack(columns).astype(np.float32)
        matrix = (matrix - self.output_medians[None, :]) / self.output_iqrs[None, :]
        return np.clip(matrix, -8.0, 8.0).astype(np.float32)


def fit_matrix_transform(train: pd.DataFrame, contract: ArmContract) -> MatrixTransform:
    source_fields = contract.source_fields
    values = train.loc[:, list(source_fields)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    iqrs = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    filled = np.where(np.isfinite(values), values, medians[None, :])
    z = np.clip((filled - medians[None, :]) / iqrs[None, :], -8.0, 8.0)
    lookup = {field: index for index, field in enumerate(source_fields)}
    columns = [z[:, lookup[field]] for field in contract.direct_fields]
    names = list(contract.direct_fields)
    for trust_field, market_field in contract.interactions:
        columns.append(np.clip(z[:, lookup[trust_field]] * z[:, lookup[market_field]], -16.0, 16.0))
        names.append(f"interaction__{trust_field}__x__{market_field}")
    matrix = np.column_stack(columns).astype(np.float32)
    output_medians = np.nanmedian(matrix, axis=0)
    output_medians = np.where(np.isfinite(output_medians), output_medians, 0.0).astype(np.float32)
    output_iqrs = np.maximum(
        np.nanquantile(matrix, 0.75, axis=0) - np.nanquantile(matrix, 0.25, axis=0),
        1e-4,
    ).astype(np.float32)
    family_indices: dict[str, np.ndarray] = {}
    for family in ("score_geometry", "support", "ood_drift", "market", "interaction"):
        positions = [index for index, name in enumerate(names) if _family(name) == family]
        if positions:
            family_indices[family] = np.asarray(positions, dtype=np.int64)
    return MatrixTransform(
        source_fields=source_fields,
        direct_fields=contract.direct_fields,
        interactions=contract.interactions,
        source_medians=medians,
        source_iqrs=iqrs,
        output_medians=output_medians,
        output_iqrs=output_iqrs,
        feature_names=tuple(names),
        family_indices=family_indices,
    )


def _equal_month_subsample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    period = frame["__decision_ts__"].dt.to_period("M")
    blocks = list(frame.assign(__month__=period).groupby("__month__", sort=True, observed=True))
    quota = max(1, int(math.ceil(maximum / len(blocks))))
    pieces: list[pd.DataFrame] = []
    for _, block in blocks:
        if len(block) <= quota:
            pieces.append(block)
            continue
        positions = np.linspace(0, len(block) - 1, quota, dtype=np.int64)
        pieces.append(block.iloc[np.unique(positions)])
    output = pd.concat(pieces, ignore_index=True)
    if len(output) > maximum:
        positions = np.linspace(0, len(output) - 1, maximum, dtype=np.int64)
        output = output.iloc[np.unique(positions)].copy()
    return output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


class LightFamilyMLP(nn.Module):
    """A small deterministic CPU MLP with feature-family dropout."""

    def __init__(self, features: int, family_indices: Mapping[str, np.ndarray]) -> None:
        super().__init__()
        self.first = nn.Linear(features, 24)
        self.second = nn.Linear(24, 8)
        self.output = nn.Linear(8, 1)
        self.family_indices = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in family_indices.items()
            if name != "score_geometry"
        }
        self.family_dropout = {
            "support": 0.10,
            "ood_drift": 0.10,
            "market": 0.15,
            "interaction": 0.15,
        }

    def _drop_families(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return values
        output = values.clone()
        for name, indices in self.family_indices.items():
            probability = self.family_dropout.get(name, 0.0)
            if probability <= 0.0 or indices.numel() == 0:
                continue
            keep = (torch.rand((len(values), 1), device=values.device) >= probability).to(values.dtype)
            keep = keep / (1.0 - probability)
            output[:, indices] = output[:, indices] * keep
        return output

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self._drop_families(values)
        hidden = F.gelu(self.first(values))
        hidden = F.dropout(hidden, p=0.05, training=self.training)
        hidden = F.gelu(self.second(hidden))
        return self.output(hidden).squeeze(1)


@dataclass
class FittedMLP:
    model: LightFamilyMLP
    transform: MatrixTransform
    audit: dict[str, Any]


def fit_light_mlp(train: pd.DataFrame, contract: ArmContract, *, cutoff: pd.Timestamp, seed: int) -> FittedMLP:
    """Fit with a chronological validation tail and an explicit H12 purge."""
    train = _equal_month_subsample(train, MAX_TRAIN_ROWS)
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"MLP has only {len(train):,} resolved train rows")
    split = max(1, int(math.floor(0.85 * len(train))))
    validation = train.iloc[split:].copy()
    fit = train.iloc[:split].copy()
    validation_start = validation["__decision_ts__"].min()
    fit = fit.loc[fit["__decision_ts__"].lt(validation_start - LABEL_HORIZON)].copy()
    if len(fit) < MIN_TRAIN_ROWS // 2 or len(validation) < 1_000:
        raise ValueError("MLP chronological train/validation split lacks support")
    if not fit["policy_label_available_ts"].lt(cutoff).all():
        raise AssertionError("MLP fit consumed an unresolved label")
    y_fit = _target(fit)
    y_validation = _target(validation)
    if np.unique(y_fit).size < 2 or np.unique(y_validation).size < 2:
        raise ValueError("MLP correctness target is degenerate")
    transform = fit_matrix_transform(fit, contract)
    x_fit = transform.transform(fit)
    x_validation = transform.transform(validation)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.set_num_threads(4)
    model = LightFamilyMLP(x_fit.shape[1], transform.family_indices)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(x_fit)
    y_tensor = torch.from_numpy(y_fit.astype(np.float32))
    x_val_tensor = torch.from_numpy(x_validation)
    y_val_tensor = torch.from_numpy(y_validation.astype(np.float32))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, 26):
        model.train()
        order = torch.randperm(len(x_tensor), generator=generator)
        for start in range(0, len(order), 2048):
            index = order[start : start + 2048]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_tensor[index]), y_tensor[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(x_val_tensor), y_val_tensor).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("MLP never produced a validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_validation = torch.sigmoid(model(x_val_tensor)).numpy()
    audit = {
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "fit_max_label_available_ts": fit["policy_label_available_ts"].max(),
        "validation_start": validation_start,
        "target_positive_rate_fit": float(y_fit.mean()),
        "target_positive_rate_validation": float(y_validation.mean()),
        "best_epoch": int(best_epoch),
        "best_validation_log_loss": float(best_loss),
        "validation_auc": float(roc_auc_score(y_validation, p_validation)),
        "validation_brier": float(brier_score_loss(y_validation, p_validation)),
        "feature_count": int(x_fit.shape[1]),
        "feature_names": list(transform.feature_names),
        "family_sizes": {name: int(len(indices)) for name, indices in transform.family_indices.items()},
        "family_dropout": dict(model.family_dropout),
        "hidden_layers": [24, 8],
        "weight_decay": 0.05,
        "learning_rate": 8e-4,
        "epochs_ceiling": 25,
    }
    return FittedMLP(model=model, transform=transform, audit=audit)


def predict_light_mlp(fitted: FittedMLP, frame: pd.DataFrame) -> np.ndarray:
    values = torch.from_numpy(fitted.transform.transform(frame))
    fitted.model.eval()
    with torch.no_grad():
        result = torch.sigmoid(fitted.model(values)).numpy()
    return np.clip(result, 1e-4, 1.0 - 1e-4).astype(np.float32)


def bounded_lambda(probability: Sequence[float]) -> np.ndarray:
    """The predeclared trust authority: 0.25 <= lambda <= 1.25."""
    return np.clip(0.25 + np.asarray(probability, dtype=float), 0.25, 1.25).astype(np.float32)


def _load_year(year: int) -> pd.DataFrame:
    path = PREDICTION_PATHS[int(year)]
    if not path.exists() or not SOURCE_PANEL.exists():
        raise FileNotFoundError(f"missing canonical input for {year}: {path}")
    prediction_columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "base_score", "base_rank", "base_anchor_bps", "consensus_rank", "upstream",
        "correctness_raw", "correctness_rank", "raw_correctness_demote", "final_score",
        *SUPPORT_FIELDS, *OOD_FIELDS,
        "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price", "policy_label_available_ts",
        "model_cutoff", "model_held_end_exclusive", "geometry_bundle_sha256", "geometry_bundle_id",
        "geometry_fit_start", "geometry_fit_end_exclusive", "meta_training_start",
        "training_months", "burnin_months", "cadence_weeks", "arm",
    ]
    available = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(prediction_columns) - available)
    if missing:
        raise ValueError(f"canonical prediction artifact lacks {missing}")
    frame = pd.read_parquet(path, columns=prediction_columns)
    for column in ("__ts__", "__decision_ts__", "policy_label_available_ts", "model_cutoff", "model_held_end_exclusive"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{year} canonical prediction identities are invalid")
    start, end = frame["__decision_ts__"].min().floor("D"), (frame["__decision_ts__"].max() + pd.Timedelta(days=1)).ceil("D")
    raw = pd.read_parquet(
        SOURCE_PANEL,
        columns=["candidate_id", "__decision_ts__", *MARKET_CANDIDATES],
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    if raw["candidate_id"].duplicated().any():
        raise ValueError(f"{year} raw source duplicate candidate IDs")
    merged = frame.merge(
        raw.drop(columns="__decision_ts__"), on="candidate_id", how="left",
        validate="one_to_one", indicator="__raw_join__",
    )
    if not merged["__raw_join__"].eq("both").all():
        raise AssertionError(f"{year} raw context does not cover canonical predictions")
    merged = merged.drop(columns="__raw_join__")
    coverage = merged.loc[:, list(MARKET_CANDIDATES)].apply(pd.to_numeric, errors="coerce").notna().mean()
    low = coverage.loc[coverage.lt(0.90)]
    if not low.empty:
        raise ValueError(f"{year} market context coverage below 90%: {low.to_dict()}")
    merged["year"] = int(year)
    return merged.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _score_mlp_arm(
    frame: pd.DataFrame,
    *,
    contract: ArmContract,
    first_active_cutoff: pd.Timestamp,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one predeclared MLP arm block-by-block, with causal warm-up fallback."""
    parts: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    held_metrics: list[dict[str, Any]] = []
    for block_index, cutoff_raw in enumerate(sorted(frame["model_cutoff"].unique())):
        cutoff = _utc(cutoff_raw)
        held = frame.loc[frame["model_cutoff"].eq(cutoff)].copy()
        reference = frame.loc[
            frame["__decision_ts__"].ge(cutoff - pd.Timedelta(days=REFERENCE_DAYS))
            & frame["__decision_ts__"].lt(cutoff)
        ].copy()
        output = held.copy()
        output["mlp_arm"] = contract.arm
        output["mlp_active"] = False
        output["mlp_probability_correct100"] = np.nan
        output["mlp_lambda"] = 1.0
        output["mlp_raw_score"] = output["raw_correctness_demote"].to_numpy(float)
        output["mlp_final_score"] = output["final_score"].to_numpy(float)
        audit: dict[str, Any] = {
            "arm": contract.arm,
            "cutoff": cutoff,
            "held_end_exclusive": held["model_held_end_exclusive"].iloc[0],
            "held_rows": int(len(held)),
            "reference_rows": int(len(reference)),
            "status": "canonical_fallback_before_discovery",
            "feature_contract": {
                "direct_fields": list(contract.direct_fields),
                "interactions": [list(pair) for pair in contract.interactions],
            },
            "held_outcomes_consumed": False,
        }
        if cutoff >= first_active_cutoff:
            train = _resolved_training_rows(frame, cutoff)
            if len(reference) >= 1_000 and len(train) >= MIN_TRAIN_ROWS:
                try:
                    fitted = fit_light_mlp(
                        train, contract, cutoff=cutoff, seed=seed + block_index,
                    )
                    combined = pd.concat(
                        [reference.assign(__score_role__="reference"), held.assign(__score_role__="held")],
                        ignore_index=True,
                    ).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
                    probability = predict_light_mlp(fitted, combined)
                    multiplier = bounded_lambda(probability)
                    raw = pd.to_numeric(combined["raw_correctness_demote"], errors="coerce").to_numpy(float) * multiplier
                    reference_mask = combined["__score_role__"].eq("reference").to_numpy()
                    score_reference = ScoreReference.fit(
                        raw[reference_mask], source="same_mlp_prior42_raw_score",
                    )
                    combined["mlp_probability_correct100"] = probability
                    combined["mlp_lambda"] = multiplier
                    combined["mlp_raw_score"] = raw.astype(np.float32)
                    combined["mlp_final_score"] = score_reference.cdf(raw).astype(np.float32)
                    scored = combined.loc[~reference_mask].copy()
                    output = held.drop(columns=[
                        "mlp_probability_correct100", "mlp_lambda", "mlp_raw_score", "mlp_final_score",
                    ], errors="ignore").merge(
                        scored.loc[:, [
                            "candidate_id", "mlp_probability_correct100", "mlp_lambda", "mlp_raw_score", "mlp_final_score",
                        ]],
                        on="candidate_id", how="left", validate="one_to_one",
                    )
                    output["mlp_arm"] = contract.arm
                    output["mlp_active"] = True
                    if output["mlp_final_score"].isna().any():
                        raise AssertionError("MLP block failed to score every held candidate")
                    held_valid = (
                        output["policy_path_valid"].fillna(False).astype(bool)
                        & np.isfinite(pd.to_numeric(output["policy_net_bps"], errors="coerce"))
                        & np.isfinite(pd.to_numeric(output["base_anchor_bps"], errors="coerce"))
                    )
                    y_held = _target(output.loc[held_valid])
                    p_held = output.loc[held_valid, "mlp_probability_correct100"].to_numpy(float)
                    held_metrics.append({
                        "arm": contract.arm, "cutoff": cutoff, "rows": int(held_valid.sum()),
                        "target_positive_rate": float(y_held.mean()),
                        "auc": float(roc_auc_score(y_held, p_held)) if np.unique(y_held).size > 1 else np.nan,
                        "brier": float(brier_score_loss(y_held, p_held)),
                        "log_loss": float(log_loss(y_held, p_held, labels=[0, 1])),
                    })
                    audit.update({
                        "status": "fit_and_score",
                        "train_window_start": cutoff - pd.DateOffset(months=TRAIN_MONTHS),
                        "train_rows_resolved": int(len(train)),
                        "reference_uses_same_mlp": True,
                        "lambda_min": float(np.min(multiplier)),
                        "lambda_max": float(np.max(multiplier)),
                        **fitted.audit,
                    })
                except ValueError as exc:
                    audit.update({"status": "canonical_fallback_insufficient_support", "reason": str(exc), "train_rows_resolved": int(len(train))})
            else:
                audit.update({
                    "status": "canonical_fallback_insufficient_support",
                    "reason": "reference or resolved train support below threshold",
                    "train_rows_resolved": int(len(train)),
                })
        parts.append(output)
        fold_audit.append(audit)
    prediction = pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if prediction["candidate_id"].duplicated().any():
        raise AssertionError("MLP scoring duplicated candidate identities")
    return prediction, pd.DataFrame(fold_audit), pd.DataFrame(held_metrics)


def _control_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["mlp_arm"] = "control"
    output["mlp_active"] = False
    output["mlp_probability_correct100"] = np.nan
    output["mlp_lambda"] = 1.0
    output["mlp_raw_score"] = output["raw_correctness_demote"].to_numpy(float)
    output["mlp_final_score"] = output["final_score"].to_numpy(float)
    return output


def global_tail_metrics(frame: pd.DataFrame, *, arm: str, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = frame.loc[np.isfinite(pd.to_numeric(frame[score_column], errors="coerce"))].copy()
    rows: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    for tail in TAILS:
        selected = population.nlargest(max(1, int(math.ceil(tail * len(population)))), score_column, keep="first")
        valid = selected.loc[
            selected["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
        ].copy()
        rows.append({
            "arm": arm, "tail": tail, "population_rows": int(len(population)),
            "selected_score_rows": int(len(selected)), "valid_outcomes": int(len(valid)),
            "outcome_coverage": float(len(valid) / max(len(selected), 1)), "trades": int(len(valid)),
            "gross_bps_per_trade": float(valid["policy_gross_bps"].mean()),
            "net_bps_per_trade": float(valid["policy_net_bps"].mean()),
            "net_sum_bps": float(valid["policy_net_bps"].sum()),
            "positive_rate": float(valid["policy_net_bps"].gt(0.0).mean()),
        })
        for month, selected_month in selected.groupby(selected["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
            block = selected_month.loc[
                selected_month["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected_month["policy_net_bps"], errors="coerce"))
            ]
            monthly.append({
                "arm": arm, "tail": tail, "month": str(month),
                "selected_score_rows": int(len(selected_month)), "valid_outcomes": int(len(block)),
                "outcome_coverage": float(len(block) / max(len(selected_month), 1)), "trades": int(len(block)),
                "gross_bps_per_trade": float(block["policy_gross_bps"].mean()),
                "net_bps_per_trade": float(block["policy_net_bps"].mean()),
                "net_sum_bps": float(block["policy_net_bps"].sum()),
                "positive_rate": float(block["policy_net_bps"].gt(0.0).mean()),
            })
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def stability_metrics(global_metrics: pd.DataFrame, monthly_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, global_block in global_metrics.groupby("arm", sort=True):
        global_top2 = global_block.loc[global_block["tail"].eq(0.02), "net_bps_per_trade"]
        monthly_top2 = monthly_metrics.loc[
            monthly_metrics["arm"].eq(arm) & monthly_metrics["tail"].eq(0.02), "net_bps_per_trade"
        ].dropna().to_numpy(float)
        if not len(global_top2) or not len(monthly_top2):
            continue
        median = float(np.median(monthly_top2))
        mad = float(np.median(np.abs(monthly_top2 - median)))
        worst = float(np.min(monthly_top2))
        rows.append({
            "arm": str(arm), "top2_pooled_net_bps": float(global_top2.iloc[0]),
            "top2_month_median_net_bps": median, "top2_month_mad_bps": mad,
            "top2_worst_month_net_bps": worst, "top2_positive_months": int((monthly_top2 > 0.0).sum()),
            "top2_months": int(len(monthly_top2)),
            "top2_portability_score": float(median - 0.5 * mad - max(0.0, -worst)),
        })
    return pd.DataFrame(rows).sort_values(
        ["top2_portability_score", "top2_pooled_net_bps", "arm"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)


def _portfolio(frame: pd.DataFrame, *, arm: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the canonical causal map and constrained auction to one score arm."""
    from extreme_price_movements.stage_i_causal_admission import apply_causal_21d_side_admission

    admitted, audit = apply_causal_21d_side_admission(
        frame.rename(columns={"mlp_final_score": "final_score"}) if "mlp_final_score" in frame else frame,
        score_column="final_score", net_column="policy_net_bps", decision_column="__decision_ts__",
        label_available_column="policy_label_available_ts", identity_column="candidate_id",
        spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    evaluation = admitted.loc[
        admitted["__decision_ts__"].ge(start) & admitted["__decision_ts__"].lt(end)
    ].copy()
    try:
        candidates = _auction_candidates(evaluation, strategy_prefix="c3_bounded_mlp")
    except ValueError:
        empty = pd.DataFrame([{
            "arm": arm, "accepted_trades": 0, "trades_per_calendar_day": 0.0,
            "gross_bps_per_trade": np.nan, "net_bps_per_trade": np.nan,
            "positive_rate": np.nan, "max_drawdown": np.nan,
        }])
        return empty, audit.assign(arm=arm), pd.DataFrame()
    _, _, monthly, summary = _run(
        candidates, 0.0, arm, initial_wallet=1_000.0, perp_leverage=7.0,
        margin_slot_wallet_fraction=0.10,
    )
    replay = summary.get("replay_metric_summary", {})
    if isinstance(replay, str):
        replay = json.loads(replay)
    days = max((end - start).total_seconds() / 86_400.0, 1.0)
    result = pd.DataFrame([{
        "arm": arm, "accepted_trades": int(summary["accepted_trades"]),
        "trades_per_calendar_day": float(summary["accepted_trades"] / days),
        "gross_bps_per_trade": float(summary["gross_bps_per_trade"]),
        "net_bps_per_trade": float(summary["net_bps_per_trade"]),
        "positive_rate": float(summary["positive_rate"]),
        "max_drawdown": float(replay.get("max_drawdown", np.nan)),
    }])
    return result, audit.assign(arm=arm), monthly.assign(arm=arm)


def _summary_table(metrics: pd.DataFrame, stability: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    net = metrics.pivot(index="arm", columns="tail", values="net_bps_per_trade").rename(
        columns={tail: f"top{tail * 100:g}_net_bps" for tail in TAILS}
    )
    coverage = metrics.pivot(index="arm", columns="tail", values="outcome_coverage").rename(
        columns={tail: f"top{tail * 100:g}_coverage" for tail in TAILS}
    )
    return net.join(coverage).reset_index().merge(stability, on="arm", how="left").merge(portfolio, on="arm", how="left")


def _render_report(
    path: Path,
    *,
    discovery: FeatureDiscovery,
    development_summary: pd.DataFrame,
    confirmation_summary: pd.DataFrame,
    winner: str,
    development_target_metrics: pd.DataFrame,
    confirmation_target_metrics: pd.DataFrame,
    first_active_cutoff: pd.Timestamp,
) -> None:
    def table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
        existing = [column for column in columns if column in frame]
        if frame.empty or not existing:
            return "_No rows._"
        return frame.loc[:, existing].to_markdown(index=False, floatfmt=".2f")

    lines = [
        "# C3 bounded MLP trust × market ablation",
        "",
        "## Scope",
        "",
        "A light, post-C3 reliability MLP is compared with the exact schema-v3 C3 canonical score. It does not replace the strict-R3 base, ten-head consensus, current C3 correctness ranker, causal admission map, or constrained portfolio auction.",
        "",
        "The MLP target is `1[policy_net_bps - base_anchor_bps > +100]`. It maps its probability to `lambda = clip(0.25 + p, 0.25, 1.25)` and applies that multiplier only to the existing `raw_correctness_demote`, followed by the same-model prior-42-day CDF.",
        "",
        f"The fixed market/interactions contract was discovered from rows resolved before {DISCOVERY_CUTOFF.isoformat()} and MLP scoring began at {first_active_cutoff.isoformat()}. 2025 is development; 2026 uses the unchanged selected contract as confirmation. Neither period is an untouched final test.",
        "",
        "## Frozen feature contract",
        "",
        "Trust inputs are canonical score geometry, leaf support, and aggregate K9 support/OOD/drift. Raw K9 membership vectors remain excluded. The model has 24 and 8 hidden units, training-only robust scaling, chronological early stopping, AdamW weight decay 0.05, and independent family dropout (support/OOD 10%; market/interactions 15%).",
        "",
        "### Market fields selected by train-only conditional MI",
        "",
        table(discovery.market_audit, ["field", "conditional_mi", "coverage", "nunique", "selected"]),
        "",
        "### Trust × market interactions selected by incremental conditional MI",
        "",
        table(discovery.interaction_audit.loc[discovery.interaction_audit["selected"]], ["trust_field", "market_field", "trust_conditional_mi", "market_conditional_mi", "joint_conditional_mi", "incremental_cmi"]),
        "",
        "## Development: 2025-04-23 to 2025-08-01",
        "",
        table(development_summary, ["arm", "top0.5_net_bps", "top1_net_bps", "top2_net_bps", "top5_net_bps", "top10_net_bps", "top2_portability_score", "top2_worst_month_net_bps", "top2_positive_months", "accepted_trades", "net_bps_per_trade"]),
        "",
        f"The selected MLP feature arm is **{winner}**, chosen before confirmation by top-2 portability, then pooled top-2 net, then arm name. The control is never silently displaced.",
        "",
        "### MLP target diagnostics, development",
        "",
        table(development_target_metrics, ["arm", "cutoff", "rows", "target_positive_rate", "auc", "brier", "log_loss"]),
        "",
        "## Confirmation: 2026-04-23 to 2026-08-01",
        "",
        table(confirmation_summary, ["arm", "top0.5_net_bps", "top1_net_bps", "top2_net_bps", "top5_net_bps", "top10_net_bps", "top2_portability_score", "top2_worst_month_net_bps", "top2_positive_months", "accepted_trades", "net_bps_per_trade"]),
        "",
        "### MLP target diagnostics, confirmation",
        "",
        table(confirmation_target_metrics, ["arm", "cutoff", "rows", "target_positive_rate", "auc", "brier", "log_loss"]),
        "",
        "## Interpretation gate",
        "",
        "An MLP arm is only an encouraging challenger if it improves the matched control in both score-tail and causal-admission economics without worsening the worst month. This report records a frozen-contract confirmation, but does not promote the MLP because the canonical source periods have already been used in upstream research and outcome coverage remains an explicit diagnostic dimension.",
        "",
        "All tail tables select from the complete finite-score candidate population before checking future policy-path coverage. Costs are embedded once in the selected-policy net labels used by the existing canonical replay.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_and_evaluate(
    frame: pd.DataFrame,
    *,
    contracts: Mapping[str, ArmContract],
    first_active_cutoff: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    arms: Sequence[str],
    run_portfolio: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}
    global_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    portfolio_parts: list[pd.DataFrame] = []
    admission_parts: list[pd.DataFrame] = []
    for arm_index, arm in enumerate(arms):
        if arm == "control":
            scored = _control_prediction(frame)
            folds = pd.DataFrame([{
                "arm": "control", "status": "exact_canonical_control", "held_rows": int(len(scored)),
                "held_outcomes_consumed": False,
            }])
            target_metrics = pd.DataFrame()
        else:
            scored, folds, target_metrics = _score_mlp_arm(
                frame, contract=contracts[arm], first_active_cutoff=first_active_cutoff,
                seed=SEED + 1000 * arm_index,
            )
        predictions[arm] = scored
        evaluation = scored.loc[
            scored["__decision_ts__"].ge(evaluation_start) & scored["__decision_ts__"].lt(evaluation_end)
        ].copy()
        global_metrics, monthly_metrics = global_tail_metrics(evaluation, arm=arm, score_column="mlp_final_score")
        global_parts.append(global_metrics)
        monthly_parts.append(monthly_metrics)
        fold_parts.append(folds.assign(year=int(evaluation_start.year)))
        if not target_metrics.empty:
            target_parts.append(target_metrics.assign(year=int(evaluation_start.year)))
        if run_portfolio:
            portfolio, admission, _ = _portfolio(
                scored, arm=arm, start=evaluation_start, end=evaluation_end,
            )
            portfolio_parts.append(portfolio)
            admission_parts.append(admission.assign(year=int(evaluation_start.year)))
        else:
            portfolio_parts.append(pd.DataFrame([{
                "arm": arm, "accepted_trades": np.nan, "trades_per_calendar_day": np.nan,
                "gross_bps_per_trade": np.nan, "net_bps_per_trade": np.nan,
                "positive_rate": np.nan, "max_drawdown": np.nan,
            }]))
    global_frame = pd.concat(global_parts, ignore_index=True)
    monthly_frame = pd.concat(monthly_parts, ignore_index=True)
    stability = stability_metrics(global_frame, monthly_frame)
    return (
        predictions, global_frame, monthly_frame, stability, pd.concat(fold_parts, ignore_index=True),
        pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame(),
        pd.concat(portfolio_parts, ignore_index=True),
        pd.concat(admission_parts, ignore_index=True) if admission_parts else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-portfolio", action="store_true")
    parser.add_argument("--development-start", default=DEV_EVALUATION_START.isoformat())
    parser.add_argument("--development-end", default=DEV_EVALUATION_END.isoformat())
    parser.add_argument("--confirmation-start", default=CONFIRM_EVALUATION_START.isoformat())
    parser.add_argument("--confirmation-end", default=CONFIRM_EVALUATION_END.isoformat())
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    development_start, development_end = _utc(args.development_start), _utc(args.development_end)
    confirmation_start, confirmation_end = _utc(args.confirmation_start), _utc(args.confirmation_end)
    if development_start < DISCOVERY_CUTOFF + pd.Timedelta(days=42):
        raise ValueError("development start must leave 42 days of post-discovery MLP score history")

    print(json.dumps({"event": "load_2025"}), flush=True)
    panel_2025 = _load_year(2025)
    discovery_rows = panel_2025.loc[
        panel_2025["policy_label_available_ts"].lt(DISCOVERY_CUTOFF)
        & panel_2025["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel_2025["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel_2025["base_anchor_bps"], errors="coerce"))
    ].copy()
    if len(discovery_rows) < MIN_TRAIN_ROWS:
        raise ValueError(f"feature discovery only has {len(discovery_rows):,} resolved rows")
    discovery = discover_feature_contract(discovery_rows)
    contracts = {arm: arm_contract(arm, discovery) for arm in ARMS if arm != "control"}
    print(json.dumps({
        "event": "feature_contract_frozen", "discovery_rows": len(discovery_rows),
        "market_fields": list(discovery.market_fields), "interactions": [list(pair) for pair in discovery.interactions],
    }), flush=True)
    dev_arms = list(ARMS)
    dev = _score_and_evaluate(
        panel_2025, contracts=contracts, first_active_cutoff=DISCOVERY_CUTOFF,
        evaluation_start=development_start, evaluation_end=development_end,
        arms=dev_arms, run_portfolio=not args.skip_portfolio,
    )
    (
        dev_predictions, dev_global, dev_monthly, dev_stability, dev_folds,
        dev_target, dev_portfolio, dev_admission,
    ) = dev
    mlp_stability = dev_stability.loc[dev_stability["arm"].ne("control")].copy()
    if mlp_stability.empty:
        raise RuntimeError("no MLP arm completed development")
    winner = str(mlp_stability.iloc[0]["arm"])
    print(json.dumps({"event": "development_winner_frozen", "winner": winner}), flush=True)

    print(json.dumps({"event": "load_2026", "winner": winner}), flush=True)
    panel_2026 = _load_year(2026)
    confirm = _score_and_evaluate(
        panel_2026, contracts=contracts, first_active_cutoff=pd.Timestamp("2026-01-29", tz="UTC"),
        evaluation_start=confirmation_start, evaluation_end=confirmation_end,
        arms=("control", winner), run_portfolio=not args.skip_portfolio,
    )
    (
        confirm_predictions, confirm_global, confirm_monthly, confirm_stability, confirm_folds,
        confirm_target, confirm_portfolio, confirm_admission,
    ) = confirm

    args.out.mkdir(parents=True)
    discovery.market_audit.to_parquet(args.out / "market_cmi_audit.parquet", index=False)
    discovery.interaction_audit.to_parquet(args.out / "trust_market_interaction_cmi_audit.parquet", index=False)
    dev_global.to_parquet(args.out / "development_global_tail_metrics.parquet", index=False)
    dev_monthly.to_parquet(args.out / "development_monthly_tail_metrics.parquet", index=False)
    dev_stability.to_parquet(args.out / "development_stability.parquet", index=False)
    dev_folds.to_parquet(args.out / "development_mlp_fold_audit.parquet", index=False)
    dev_target.to_parquet(args.out / "development_mlp_target_metrics.parquet", index=False)
    dev_portfolio.to_parquet(args.out / "development_portfolio_metrics.parquet", index=False)
    if not dev_admission.empty:
        dev_admission.to_parquet(args.out / "development_admission_audit.parquet", index=False)
    confirm_global.to_parquet(args.out / "confirmation_global_tail_metrics.parquet", index=False)
    confirm_monthly.to_parquet(args.out / "confirmation_monthly_tail_metrics.parquet", index=False)
    confirm_stability.to_parquet(args.out / "confirmation_stability.parquet", index=False)
    confirm_folds.to_parquet(args.out / "confirmation_mlp_fold_audit.parquet", index=False)
    confirm_target.to_parquet(args.out / "confirmation_mlp_target_metrics.parquet", index=False)
    confirm_portfolio.to_parquet(args.out / "confirmation_portfolio_metrics.parquet", index=False)
    if not confirm_admission.empty:
        confirm_admission.to_parquet(args.out / "confirmation_admission_audit.parquet", index=False)
    # Retain only the selected MLP's scored rows plus the matched control; all
    # losing-arm score populations are represented by their metrics/audits.
    pd.concat(
        [
            dev_predictions["control"].assign(split="development"),
            dev_predictions[winner].assign(split="development"),
            confirm_predictions["control"].assign(split="confirmation"),
            confirm_predictions[winner].assign(split="confirmation"),
        ], ignore_index=True,
    ).to_parquet(args.out / "selected_arm_predictions.parquet", index=False, compression="zstd")

    development_summary = _summary_table(dev_global, dev_stability, dev_portfolio)
    confirmation_summary = _summary_table(confirm_global, confirm_stability, confirm_portfolio)
    development_summary.to_parquet(args.out / "development_summary.parquet", index=False)
    confirmation_summary.to_parquet(args.out / "confirmation_summary.parquet", index=False)
    _render_report(
        args.report, discovery=discovery, development_summary=development_summary,
        confirmation_summary=confirmation_summary, winner=winner,
        development_target_metrics=dev_target, confirmation_target_metrics=confirm_target,
        first_active_cutoff=DISCOVERY_CUTOFF,
    )
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETED_NONPROMOTABLE_RESEARCH_ABLATION",
        "side": "long",
        "source_panel": str(SOURCE_PANEL),
        "prediction_paths": {str(year): str(path) for year, path in PREDICTION_PATHS.items()},
        "source_contract": "schema-v3 canonical C3 predictions + immutable target-free market fields",
        "base_control": "exact stored canonical final_score; no base/consensus/C3 refit",
        "target": "1[policy_net_bps - causal base_anchor_bps > +100]",
        "modulation": "raw_correctness_demote * clip(0.25 + P(correct_100), 0.25, 1.25), then same-model prior-42d CDF",
        "feature_discovery_cutoff": DISCOVERY_CUTOFF,
        "feature_discovery_rows": int(len(discovery_rows)),
        "market_fields": list(discovery.market_fields),
        "interactions": [list(pair) for pair in discovery.interactions],
        "contracts": {
            arm: {"direct_fields": list(contract.direct_fields), "interactions": [list(pair) for pair in contract.interactions]}
            for arm, contract in contracts.items()
        },
        "mlp": {"hidden_layers": [24, 8], "weight_decay": 0.05, "learning_rate": 8e-4, "epochs_ceiling": 25, "family_dropout": {"support": 0.10, "ood_drift": 0.10, "market": 0.15, "interaction": 0.15}},
        "selection": {"period": [development_start, development_end], "objective": "top2 portability, then pooled top2 net", "winner": winner},
        "confirmation": {"period": [confirmation_start, confirmation_end], "arms": ["control", winner], "feature_contract_refit": False},
        "tail_selection": "all finite scored candidates -> global top-k -> valid policy outcome coverage",
        "admission": "same causal 21/42/84-day side-local hierarchical +50bps map and constrained auction",
        "cost": "existing selected-policy policy_net_bps; cost embedded exactly once",
        "purge": "training labels require policy_label_available_ts < cutoff; chronological MLP validation has 12h purge",
        "code_sha256": _sha256(Path(__file__)),
        "inputs_sha256": {str(year): _sha256(path) for year, path in PREDICTION_PATHS.items()},
        "report": str(args.report),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(args.out), "winner": winner, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
