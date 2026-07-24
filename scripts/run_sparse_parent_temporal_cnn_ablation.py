#!/usr/bin/env python3
"""Causal TCN/CNN replacement ablation above the frozen sparse-parent score.

The experiment deliberately does *not* use local subtype or sub-archetype
features.  It asks a narrower question: can a causal temporal representation
of the inference-available market/meta state replace marginal decisions from
the current sparse-parent ranker?

For every expanding OOS month, the model:
  * uses only feature bars ending at the decision timestamp;
  * trains separate long and short residual-EV experts;
  * retains a train-only, side-aware temporal-channel contract selected from
    the April train window; and
  * selects the correction blend solely from completed earlier OOS months.

All arms share rows, base/meta parent, costs, monthly fixed activity and
post-processing. ``cnn`` has ordinary convolutions, ``tcn`` uses dilations
1/2/4, and ``*_interaction`` adds a regularized nonlinear context branch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the exact parent, causal residual target, fixed-activity budget and
# policy gate used in the sparse local MLP research path.  The new temporal
# layer never imports or materialises local_subtype fields.
from scripts.run_v9_tail_full_contract_mlp_ablation import (  # noqa: E402
    KEYS,
    OUTCOME_TOKENS,
    PROTECTED_ANCHORS,
    _direct_causal_residual,
    _fixed_activity_mask,
    _load_rows,
    _metric_rows,
    _month_ids,
    _num,
    _policy_params,
    _rank_weights,
    _stability,
    _tail_fit_mask,
)
from extreme_price_movements.data_store import (  # noqa: E402
    _feature_schema_names,
    read_symbol_features,
)
from extreme_price_movements.causal_change_state import (  # noqa: E402
    build_causal_change_state,
    build_streaming_long_change_state,
)
from extreme_price_movements.supervised_market_state_calibration import (  # noqa: E402
    fit_hierarchical_ev_calibrator,
    predict_hierarchical_ev,
)


torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


STATIC_ANCHORS = (
    "existing_sparse_parent_score",
    "policy_parent_rank",
    "score",
    "base_score_rank_pct_train_prior",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "hit_probability",
)
STATIC_RELIABILITY_HINTS = (
    "leaf_", "support_", "ood", "drift", "gmm_", "aegmm_",
    "reconstruction", "mahal", "posterior", "entropy", "cluster_",
)
FEATURE_AGE_CHANNEL = "__feature_staleness_hours__"
TEMPORAL_TOKENS = (
    "mkt_", "market_", "xasset", "xs_", "cs_", "iqr_", "q_", "oi", "funding", "breadth",
    "ret", "return", "price", "atr", "rv", "vol", "volume", "range",
    "wick", "liquid", "order", "ob_", "spread", "depth", "imbalance",
    "entropy", "shock", "recovery", "delever", "gmm", "aegmm", "ae_",
    "reconstruction", "mahal", "cluster", "posterior", "leaf", "support",
    "ood", "drift", "dislocation", "seasonality", "spike",
)
MARKET_PRETRAIN_PREFIXES = (
    "cs_", "xs_", "iqr_", "q_", "mkt_", "market_", "xasset_",
)
EXCLUDED_TEMPORAL_TOKENS = (
    "local_subtype", "target", "label", "future", "oracle", "realized_",
    "ev_after", "clean_exec", "dirty_positive", "bad_mae", "timeout",
    "full_stop", "first_touch", "outcome", "strategy_id",
)


@dataclass
class TemporalContract:
    side: str
    channels: list[str]
    channel_report: list[dict[str, Any]]
    static_features: list[str]
    archetypes: list[str]
    lookback_bars: int
    coverage: dict[str, Any]


@dataclass
class TemporalModel:
    side: str
    architecture: str
    contract: TemporalContract
    medians: np.ndarray
    scales: np.ndarray
    static_medians: np.ndarray
    static_scales: np.ndarray
    target_center: float
    target_scale: float
    model: nn.Module
    ood_q50: float
    ood_q95: float
    archetype_support: dict[str, int]
    validation_loss: float
    fit_rows: int


class _CausalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.trim = padding

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv(value)
        if self.trim:
            output = output[:, :, :-self.trim]
        output = torch.relu(self.norm(output))
        output = self.dropout(output)
        return torch.relu(output + self.skip(value))


class CausalTemporalResidualNet(nn.Module):
    """Underfit-biased temporal encoder with optional context interactions."""

    def __init__(
        self,
        channels: int,
        static_dim: int,
        architecture: str,
        dropout: float,
        widths: tuple[int, int, int],
        lookback_bars: int,
        archetype_dim: int = 0,
        mechanism_positions: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.architecture = architecture
        base_architecture = architecture.removesuffix("_interaction").removesuffix("_lowrank")
        self.has_interaction_branch = architecture.endswith("_interaction")
        self.has_lowrank_branch = architecture.endswith("_lowrank")
        self.archetype_dim = int(archetype_dim)
        self.mechanism_positions = tuple(int(value) for value in mechanism_positions)
        if base_architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * lookback_bars, widths[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(widths[0], widths[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(widths[1], widths[2]),
                nn.ReLU(),
            )
            previous = widths[-1]
        else:
            dilations = (1, 2, 4) if base_architecture == "tcn" else (1, 1, 1)
            blocks: list[nn.Module] = []
            previous = channels
            for width, dilation in zip(widths, dilations):
                blocks.append(_CausalBlock(previous, width, dilation, dropout))
                previous = width
            self.encoder = nn.Sequential(*blocks)
        interaction_dim = max(8, min(16, widths[-1] // 2))
        if self.has_interaction_branch:
            hidden = max(16, min(32, static_dim // 2))
            self.static_interaction = nn.Sequential(
                nn.Linear(static_dim, hidden),
                nn.SiLU(),
                nn.Dropout(min(0.35, dropout + 0.08)),
                nn.Linear(hidden, interaction_dim),
                nn.SiLU(),
            )
        else:
            self.static_interaction = None
            interaction_dim = 0
        lowrank_dim = 0
        if self.has_lowrank_branch and self.mechanism_positions and self.archetype_dim > 0:
            lowrank_dim = 8
            self.mechanism_projection = nn.Sequential(
                nn.Linear(len(self.mechanism_positions), lowrank_dim, bias=False),
                nn.Tanh(),
            )
            self.archetype_projection = nn.Linear(self.archetype_dim, lowrank_dim, bias=False)
            self.lowrank_dropout = nn.Dropout(min(0.40, dropout + 0.12))
        else:
            self.mechanism_projection = None
            self.archetype_projection = None
            self.lowrank_dropout = None
        # Raw anchors remain directly observable while the compact branch can
        # express side/archetype x transition-state conditions.
        self.head = nn.Linear(previous + static_dim + interaction_dim + lowrank_dim, 1)

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(sequence)
        parts = [encoded, static]
        if self.static_interaction is not None:
            parts.append(self.static_interaction(static))
        if self.mechanism_projection is not None and self.archetype_projection is not None:
            mechanism = static[:, self.mechanism_positions]
            archetype = static[:, -self.archetype_dim:]
            local_interaction = self.mechanism_projection(mechanism) * self.archetype_projection(archetype)
            parts.append(self.lowrank_dropout(local_interaction))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence)
        return encoded if encoded.ndim == 2 else encoded[:, :, -1]


def _family(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("gmm", "aegmm", "ae_", "reconstruction", "mahal", "cluster", "posterior", "entropy")):
        return "latent_state"
    if "oi" in lower:
        return "open_interest"
    if "funding" in lower:
        return "funding"
    if any(token in lower for token in ("breadth", "mkt_", "market_", "xasset", "xs_", "corr", "dispersion")):
        return "market_cross_asset"
    if any(token in lower for token in ("ob_", "spread", "depth", "imbalance", "liquid")):
        return "liquidity_orderbook"
    if any(token in lower for token in ("rv", "vol", "atr", "range", "volume", "shock", "wick")):
        return "volatility_shock"
    if any(token in lower for token in ("ret", "price", "recovery", "momentum", "trend")):
        return "price_path"
    if any(token in lower for token in ("leaf", "support", "ood", "drift", "dislocation", "seasonality", "spike")):
        return "model_reliability"
    return "other"


def _safe_symbol_file(root: Path, symbol: str) -> Path:
    return root / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _feature_candidates(frame: pd.DataFrame, feature_root: Path) -> list[str]:
    stores = list(feature_root.glob("symbol=*.parquet"))
    if not stores:
        raise FileNotFoundError(f"no feature parquet files in {feature_root}")
    # Incremental feature columns may exist only in a symbol's DuckDB delta.
    # Build the full schema union instead of treating the first base parquet as
    # the complete feature contract.
    store_columns: set[str] = set()
    for path in stores:
        store_columns.update(_feature_schema_names(str(path)))
    result: list[str] = []
    for name in frame.columns:
        lower = str(name).lower()
        if name not in store_columns or name in {"ts", "__symbol__"}:
            continue
        if name in STATIC_ANCHORS or name in KEYS:
            continue
        if any(token in lower for token in EXCLUDED_TEMPORAL_TOKENS):
            continue
        if any(token in lower for token in OUTCOME_TOKENS):
            continue
        if not any(token in lower for token in TEMPORAL_TOKENS):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().mean() < 0.90 or values.nunique(dropna=True) < 16:
            continue
        result.append(name)
    return sorted(set(result))


def _market_store_candidates(feature_root: Path, minimum_schema_share: float = 0.0) -> list[str]:
    stores = list(feature_root.glob("symbol=*.parquet"))
    counts: dict[str, int] = {}
    for path in stores:
        for name in _feature_schema_names(str(path)):
            lower = str(name).lower()
            broad_market = (
                str(name).startswith(MARKET_PRETRAIN_PREFIXES)
                or "iqr" in lower
                or any(token in lower for token in ("cross_asset", "breadth", "synchron", "entropy", "shock"))
            )
            if not broad_market or any(token in lower for token in EXCLUDED_TEMPORAL_TOKENS):
                continue
            counts[name] = counts.get(name, 0) + 1
    required = max(1, int(math.ceil(len(stores) * minimum_schema_share)))
    return sorted(name for name, count in counts.items() if count >= required and name not in {"ts", "__symbol__"})


def _static_features(frame: pd.DataFrame) -> list[str]:
    """Keep current-time parent-reliability context out of temporal pruning."""
    result = [name for name in STATIC_ANCHORS if name in frame]
    for name in frame.columns:
        lower = str(name).lower()
        if (
            name not in result
            and not any(token in lower for token in EXCLUDED_TEMPORAL_TOKENS)
            and any(hint in lower for hint in STATIC_RELIABILITY_HINTS)
            and pd.to_numeric(frame[name], errors="coerce").notna().mean() >= 0.90
        ):
            result.append(name)
    return result


def _temporal_causal_residual(frame: pd.DataFrame) -> np.ndarray:
    """EV residual against the exact frozen parent score being corrected."""
    aligned = frame.copy()
    aligned["policy_parent_rank"] = _num(aligned, "existing_sparse_parent_score", np.nan)
    return _direct_causal_residual(aligned)


def _channel_scores(
    train: pd.DataFrame,
    candidates: list[str],
    side: str,
    rank_floor: float,
) -> pd.DataFrame:
    group = train.loc[train["side_name"].eq(side)].sort_values("__ts__", kind="stable").reset_index(drop=True)
    residual = _temporal_causal_residual(group)
    rank = _num(group, "policy_parent_rank", 0.5)
    eligible = (rank >= rank_floor) & (rank <= 0.995) & np.isfinite(residual)
    group = group.loc[eligible].reset_index(drop=True)
    target = residual[eligible]
    rows: list[dict[str, Any]] = []
    if len(group) < 1_000:
        return pd.DataFrame(rows)
    blocks = np.array_split(np.arange(len(group)), 3)
    for name in candidates:
        value = pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(value)
        if finite.mean() < 0.90 or int(finite.sum()) < 800:
            continue
        correlations: list[float] = []
        for positions in blocks:
            mask = finite[positions]
            if int(mask.sum()) < 200:
                continue
            x = value[positions][mask]
            y = target[positions][mask]
            correlation = np.corrcoef(x, y)[0, 1]
            if np.isfinite(correlation):
                correlations.append(float(correlation))
        if len(correlations) < 2:
            continue
        # Binned residual spread captures nonlinear but monotonic or U-shaped
        # predictive effects.  Block agreement prevents a single event period
        # from selecting a temporal channel.
        edges = np.unique(np.nanquantile(value[finite], np.linspace(0.0, 1.0, 9)))
        if len(edges) < 4:
            continue
        bins = np.clip(np.searchsorted(edges, value[finite], side="right") - 1, 0, len(edges) - 2)
        sums = np.bincount(bins, weights=target[finite], minlength=len(edges) - 1)
        counts = np.bincount(bins, minlength=len(edges) - 1)
        means = sums / np.maximum(counts, 1)
        nonlinear = float(np.std(means) / (np.std(target[finite]) + 1e-6))
        mean_corr = float(np.mean(correlations))
        agreement = float(abs(np.mean(np.sign(correlations))))
        score = (abs(mean_corr) + 0.55 * nonlinear) * agreement * math.sqrt(float(finite.mean()))
        rows.append({
            "side_name": side, "feature": name, "family": _family(name),
            "coverage": float(finite.mean()), "mean_block_corr": mean_corr,
            "block_sign_agreement": agreement, "binned_residual_spread": nonlinear,
            "temporal_channel_score": score,
        })
    return pd.DataFrame(rows).sort_values("temporal_channel_score", ascending=False, kind="stable")


def _select_channels(report: pd.DataFrame, floor: int = 8, ceiling: int = 28) -> tuple[list[str], pd.DataFrame]:
    if report.empty:
        return [], report
    work = report.copy()
    best = float(work["temporal_channel_score"].max())
    threshold = max(0.010, best * 0.18)
    per_family: dict[str, int] = {}
    selected: list[str] = []
    # The count is data-determined.  ``ceiling`` is only an architecture/memory
    # safety limit; it is not a fixed selected-feature target.
    for row in work.itertuples(index=False):
        family = str(row.family)
        if per_family.get(family, 0) >= 4:
            continue
        if float(row.temporal_channel_score) < threshold and len(selected) >= floor:
            continue
        selected.append(str(row.feature))
        per_family[family] = per_family.get(family, 0) + 1
        if len(selected) >= ceiling:
            break
    if len(selected) < floor:
        for feature in work["feature"]:
            if feature not in selected:
                selected.append(str(feature))
            if len(selected) >= min(floor, len(work)):
                break
    work["selected"] = work["feature"].isin(selected)
    return selected, work


def _select_pretraining_channels(
    report: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    ceiling: int,
) -> tuple[list[str], pd.DataFrame]:
    """Let every broad market feature compete, while retaining family depth."""
    if report.empty:
        return [], report
    work = report.copy()
    work["market_pretrain_candidate"] = work["feature"].map(
        lambda value: str(value).startswith(MARKET_PRETRAIN_PREFIXES)
        or any(token in str(value).lower() for token in ("breadth", "cross_asset", "dispersion", "synchron", "entropy", "shock"))
    )
    pool = work.loc[work["market_pretrain_candidate"]].copy()
    if pool.empty:
        pool = work.copy()
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    # Start with the strongest feature from every represented family, then
    # fill by residual relevance. All candidates were scored before this step.
    for _family_name, group in pool.groupby("family", observed=True, sort=False):
        feature = str(group.sort_values("temporal_channel_score", ascending=False).iloc[0]["feature"])
        selected.append(feature)
        family_counts[str(_family_name)] = 1
    for row in pool.sort_values("temporal_channel_score", ascending=False).itertuples(index=False):
        feature, family = str(row.feature), str(row.family)
        if feature in selected or family_counts.get(family, 0) >= 12:
            continue
        if pd.to_numeric(frame[feature], errors="coerce").notna().mean() < 0.90:
            continue
        selected.append(feature)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= ceiling:
            break
    selected_mask = work["feature"].isin(selected)
    # Keep the common selector contract used by TemporalContract while also
    # retaining an explicit audit field for the pretraining-specific path.
    work["selected"] = selected_mask
    work["selected_for_self_supervised_pretrain"] = selected_mask
    return selected[:ceiling], work


def _historical_pretrain_rows(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    stride_hours: int,
    max_rows: int,
) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, end=end, freq=f"{stride_hours}h", inclusive="left", tz="UTC")
    symbol_values = np.repeat(np.asarray(symbols, dtype=object), len(timestamps))
    ts_values = np.tile(timestamps.to_numpy(), len(symbols))
    rows = pd.DataFrame({"__symbol__": symbol_values, "__ts__": ts_values})
    rows = rows.sort_values(["__ts__", "__symbol__"], kind="stable").reset_index(drop=True)
    if len(rows) > max_rows:
        rows = rows.iloc[np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64)].reset_index(drop=True)
    return rows


def _pretrain_encoder(
    sequence: np.ndarray,
    architecture: str,
    widths: tuple[int, int, int],
    dropout: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    """Predict the masked final market-state bar from its causal history."""
    complete = np.isfinite(sequence).all(axis=(1, 2))
    data = sequence[complete]
    if len(data) < 5_000:
        return None, {"usable_rows": int(len(data)), "reason": "insufficient_complete_history"}
    normalised, _, _, _ = _normalise_sequence(data, data)
    split = max(int(len(normalised) * 0.90), 1)
    train = normalised[:split]
    valid = normalised[split:]
    network = CausalTemporalResidualNet(
        normalised.shape[2], 1, architecture, dropout, widths, normalised.shape[1]
    )
    decoder = nn.Linear(widths[-1], normalised.shape[2])
    parameters = list(network.encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=max(weight_decay, 0.20))
    loss_fn = nn.HuberLoss(delta=1.0)
    rng = np.random.default_rng(seed)
    best_loss, best_state, patience = np.inf, None, 0
    for epoch in range(epochs):
        network.encoder.train(); decoder.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), 768):
            positions = order[start:start + 768]
            raw = train[positions]
            corrupted = raw.copy()
            corrupted[:, -1, :] = 0.0
            noise = rng.normal(0.0, 0.035, size=corrupted.shape).astype(np.float32)
            corrupted += noise
            x = torch.from_numpy(np.swapaxes(corrupted, 1, 2))
            target = torch.from_numpy(raw[:, -1, :])
            optimizer.zero_grad(set_to_none=True)
            encoded = network.encode(x)
            loss = loss_fn(decoder(encoded), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        network.encoder.eval(); decoder.eval()
        values: list[float] = []
        with torch.no_grad():
            for start in range(0, len(valid), 2048):
                raw = valid[start:start + 2048]
                corrupted = raw.copy(); corrupted[:, -1, :] = 0.0
                encoded = network.encode(torch.from_numpy(np.swapaxes(corrupted, 1, 2)))
                values.append(float(loss_fn(decoder(encoded), torch.from_numpy(raw[:, -1, :])).item()))
        validation_loss = float(np.mean(values)) if values else np.inf
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in network.encoder.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break
        print(json.dumps({"event": "temporal_pretrain_epoch", "epoch": epoch + 1, "validation_loss": validation_loss, "best_loss": best_loss}), flush=True)
    return best_state, {
        "usable_rows": int(len(data)), "train_rows": int(len(train)),
        "validation_rows": int(len(valid)), "best_validation_loss": float(best_loss),
        "epochs_completed": int(epoch + 1),
    }


def _load_sequences(
    rows: pd.DataFrame,
    feature_root: Path,
    channels: list[str],
    lookback_bars: int,
    bar_minutes: int,
    max_stale_bars: int,
    allow_missing_channels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load causal windows with an explicit, bounded feature-age channel."""
    sequences = np.full((len(rows), lookback_bars, len(channels)), np.nan, dtype=np.float32)
    usable = np.zeros(len(rows), dtype=bool)
    if not len(rows):
        return sequences, usable
    real_channels = [name for name in channels if name != FEATURE_AGE_CHANNEL]
    age_channel = channels.index(FEATURE_AGE_CHANNEL) if FEATURE_AGE_CHANNEL in channels else None
    target_ts = pd.to_datetime(rows["__ts__"], utc=True).to_numpy(dtype="datetime64[ns]")
    for symbol, pos_index in rows.groupby("__symbol__", observed=True, sort=False).groups.items():
        positions = np.asarray(list(pos_index), dtype=np.int64)
        path = _safe_symbol_file(feature_root, str(symbol))
        if not path.exists():
            continue
        available = _feature_schema_names(str(path))
        symbol_channels = [name for name in real_channels if name in available]
        if not allow_missing_channels and len(symbol_channels) != len(real_channels):
            continue
        if not symbol_channels:
            continue
        symbol_targets = target_ts[positions]
        start = pd.Timestamp(symbol_targets.min()).tz_localize("UTC") - pd.Timedelta(
            minutes=bar_minutes * (lookback_bars - 1 + max_stale_bars)
        )
        end = pd.Timestamp(symbol_targets.max()).tz_localize("UTC")
        store = read_symbol_features(
            str(path), columns=symbol_channels, start_ts=start, end_ts=end
        )
        if store.empty or (not allow_missing_channels and any(name not in store.columns for name in real_channels)):
            continue
        store = store.loc[~store.index.isna()].sort_index(kind="stable")
        store = store.loc[~store.index.duplicated(keep="last")]
        observed_index = pd.DatetimeIndex(pd.to_datetime(store.index, utc=True))
        regular_index = pd.date_range(start=start, end=end, freq=f"{bar_minutes}min", tz="UTC")
        regular = store.loc[:, symbol_channels].apply(pd.to_numeric, errors="coerce").reindex(regular_index)
        regular = regular.reindex(columns=real_channels)
        observed = regular_index.isin(observed_index)
        positions_all = np.arange(len(regular_index), dtype=np.int32)
        last_observed = np.maximum.accumulate(np.where(observed, positions_all, -10_000_000))
        age_bars = positions_all - last_observed
        regular = regular.ffill(limit=max_stale_bars)
        timestamps = regular_index.to_numpy(dtype="datetime64[ns]")
        values = np.full((len(regular), len(channels)), np.nan, dtype=np.float32)
        real_indices = [channels.index(name) for name in real_channels]
        values[:, real_indices] = regular.to_numpy(dtype=np.float32, copy=False)
        if age_channel is not None:
            values[:, age_channel] = (age_bars.astype(np.float32) * float(bar_minutes) / 60.0)
            values[age_bars > max_stale_bars, age_channel] = np.nan
        found = pd.Index(timestamps).get_indexer(target_ts[positions])
        valid = found >= lookback_bars - 1
        if not valid.any():
            continue
        target_positions = positions[valid]
        ends = found[valid]
        offsets = np.arange(lookback_bars - 1, -1, -1, dtype=np.int64)
        indices = ends[:, None] - offsets[None, :]
        window_ts = timestamps[indices]
        expected = target_ts[target_positions, None] - (offsets[None, :] * np.timedelta64(bar_minutes, "m"))
        exact = np.all(window_ts == expected, axis=1)
        if exact.any():
            dest = target_positions[exact]
            sequences[dest] = values[indices[exact]]
            usable[dest] = np.isfinite(sequences[dest]).all(axis=(1, 2))
    return sequences, usable


def _load_streaming_long_change_summaries(
    rows: pd.DataFrame,
    feature_root: Path,
    channels: list[str],
    bar_minutes: int,
    max_stale_bars: int,
    mechanism_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize compact 2/4/7-day state at candidate timestamps.

    Processing is symbol-streamed. Only one regularized symbol frame and its
    family-level rolling summaries are resident at a time.
    """
    result: dict[str, np.ndarray] = {}
    covered = np.zeros(len(rows), dtype=bool)
    if not len(rows) or not channels:
        return pd.DataFrame(index=rows.index), {"covered_rows": 0, "output_features": 0}
    target_ts = pd.to_datetime(rows["__ts__"], utc=True).to_numpy(dtype="datetime64[ns]")
    warmup_hours = 720 + 2 * 168 + 24
    symbols_read = 0
    skipped = {"missing_path": 0, "low_channel_coverage": 0, "empty_store": 0, "target_not_found": 0}
    for symbol, pos_index in rows.groupby("__symbol__", observed=True, sort=False).groups.items():
        positions = np.asarray(list(pos_index), dtype=np.int64)
        path = _safe_symbol_file(feature_root, str(symbol))
        if not path.exists():
            skipped["missing_path"] += 1
            continue
        available = _feature_schema_names(str(path))
        present = [name for name in channels if name in available]
        if len(present) < 4:
            skipped["low_channel_coverage"] += 1
            continue
        symbol_targets = target_ts[positions]
        start = pd.Timestamp(symbol_targets.min()).tz_localize("UTC") - pd.Timedelta(hours=warmup_hours)
        end = pd.Timestamp(symbol_targets.max()).tz_localize("UTC")
        store = read_symbol_features(str(path), columns=present, start_ts=start, end_ts=end)
        if store.empty:
            skipped["empty_store"] += 1
            continue
        store = store.loc[~store.index.isna()].sort_index(kind="stable")
        store = store.loc[~store.index.duplicated(keep="last")]
        observed_index = pd.DatetimeIndex(pd.to_datetime(store.index, utc=True))
        regular_index = pd.date_range(start=start, end=end, freq=f"{bar_minutes}min", tz="UTC")
        regular = store.loc[:, present].apply(pd.to_numeric, errors="coerce").reindex(regular_index)
        observed = regular_index.isin(observed_index)
        sequence_position = np.arange(len(regular_index), dtype=np.int32)
        last_observed = np.maximum.accumulate(np.where(observed, sequence_position, -10_000_000))
        age = sequence_position - last_observed
        regular = regular.ffill(limit=max_stale_bars).reindex(columns=channels)
        regular.loc[age > max_stale_bars, :] = np.nan
        summaries = build_streaming_long_change_state(regular, channels)
        if mechanism_only:
            summaries = summaries.loc[:, [
                name for name in summaries.columns
                if name.startswith("cp_long_mechanism_")
            ]]
        found = summaries.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(symbol_targets, utc=True)))
        valid = found >= 0
        if not valid.any():
            skipped["target_not_found"] += 1
            continue
        destination = positions[valid]
        selected = summaries.iloc[found[valid]]
        for name in selected.columns:
            output = result.setdefault(name, np.full(len(rows), np.nan, dtype=np.float32))
            output[destination] = selected[name].to_numpy(dtype=np.float32, copy=False)
        covered[destination] = selected.notna().any(axis=1).to_numpy()
        symbols_read += 1
    frame = pd.DataFrame(result, index=rows.index, dtype=np.float32)
    return frame, {
        "symbols_read": symbols_read,
        "covered_rows": int(covered.sum()),
        "coverage": float(covered.mean()) if len(covered) else 0.0,
        "output_features": int(len(frame.columns)),
        "source_channels": channels,
        "durations_hours": [48, 96, 168],
        "normalization": "one_bar_lagged_720h_ewm_location_scale",
        "mechanism_only": bool(mechanism_only),
        "skipped_symbols": skipped,
    }


def _prune_joint_coverage(sequence: np.ndarray, channels: list[str], minimum: float = 0.90) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    active = list(range(len(channels)))
    per_feature = np.isfinite(sequence).all(axis=1)
    while len(active) > 4:
        joint = per_feature[:, active].all(axis=1)
        if float(joint.mean()) >= minimum:
            break
        coverage = per_feature[:, active].mean(axis=0)
        active.pop(int(np.argmin(coverage)))
    joint = per_feature[:, active].all(axis=1)
    report = {
        "initial_channels": list(channels),
        "retained_channels": [channels[i] for i in active],
        "joint_complete_sequence_coverage": float(joint.mean()),
        "individual_sequence_coverage": {channels[i]: float(per_feature[:, i].mean()) for i in active},
    }
    return [channels[i] for i in active], joint, report


def _normalise_sequence(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    if len(flat) > 240_000:
        flat = flat[np.linspace(0, len(flat) - 1, 240_000, dtype=np.int64)]
    medians = np.nanmedian(flat, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    q25, q75 = np.nanquantile(flat, (0.25, 0.75), axis=0)
    scales = (q75 - q25).astype(np.float32)
    scales[~np.isfinite(scales) | (scales < 1e-5)] = 1.0
    def transform(value: np.ndarray) -> np.ndarray:
        result = value.copy()
        missing = ~np.isfinite(result)
        if missing.any():
            result[missing] = np.take(medians, np.nonzero(missing)[2])
        return np.clip((result - medians) / scales, -8.0, 8.0).astype(np.float32)
    return transform(train), transform(score), medians, scales


def _static_matrix(
    frame: pd.DataFrame,
    features: list[str],
    archetypes: list[str],
    medians: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    if medians is None:
        medians = np.nanmedian(raw, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(raw)
    if missing.any():
        raw[missing] = np.take(medians, np.nonzero(missing)[1])
    if scales is None:
        q25, q75 = np.quantile(raw, (0.25, 0.75), axis=0)
        scales = (q75 - q25).astype(np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-5)] = 1.0
    numeric = np.clip((raw - medians) / scales, -8.0, 8.0)
    one_hot = np.zeros((len(frame), len(archetypes)), dtype=np.float32)
    mapping = {name: index for index, name in enumerate(archetypes)}
    for row, name in enumerate(frame["archetype_policy_key"].astype(str)):
        if name in mapping:
            one_hot[row, mapping[name]] = 1.0
    return np.column_stack([numeric, one_hot]).astype(np.float32), medians, scales


def _fit_temporal_model(
    train: pd.DataFrame,
    train_sequence: np.ndarray,
    contract: TemporalContract,
    architecture: str,
    seed: int,
    epochs: int,
    dropout: float,
    embargo_hours: int,
    rank_floor: float,
    widths: tuple[int, int, int],
    learning_rate: float,
    weight_decay: float,
    pretrained_encoder_state: dict[str, torch.Tensor] | None = None,
) -> TemporalModel | None:
    group = train.loc[train["side_name"].eq(contract.side)].copy().reset_index(drop=True)
    sequence = train_sequence[train["side_name"].eq(contract.side).to_numpy()]
    residual = _temporal_causal_residual(group)
    rank = _num(group, "policy_parent_rank", 0.5)
    # A temporal model must not learn from partially fabricated windows.
    # Joint sequence coverage is separately audited at 90%+, while this row
    # level guard keeps the actual fit strictly complete-case.
    eligible = (rank >= rank_floor) & (rank <= 0.995) & np.isfinite(residual) & np.isfinite(sequence).all(axis=(1, 2))
    group = group.loc[eligible].reset_index(drop=True)
    sequence = sequence[eligible]
    residual = residual[eligible]
    if len(group) < 1_500:
        return None
    validation_start = group["__ts__"].quantile(0.84)
    validation = group["__ts__"] >= validation_start
    # Strict time purge for overlapping 3–12h label paths.
    train_mask = group["__ts__"] < (validation_start - pd.Timedelta(hours=embargo_hours))
    if int(train_mask.sum()) < 900 or int(validation.sum()) < 300:
        return None
    static_features = [feature for feature in contract.static_features if feature in group]
    static_all, static_medians, static_scales = _static_matrix(group, static_features, contract.archetypes)
    seq_train, seq_all, medians, scales = _normalise_sequence(sequence[train_mask.to_numpy()], sequence)
    static_train = static_all[train_mask.to_numpy()]
    target_center = float(np.mean(residual[train_mask.to_numpy()]))
    target_scale = float(np.std(residual[train_mask.to_numpy()]))
    target_scale = max(target_scale, 1e-4)
    y = ((residual - target_center) / target_scale).astype(np.float32)
    weights = _rank_weights(group)
    model = CausalTemporalResidualNet(
        seq_all.shape[2], static_all.shape[1], architecture, dropout, widths,
        seq_all.shape[1],
        archetype_dim=len(contract.archetypes),
        mechanism_positions=tuple(
            index for index, name in enumerate(static_features)
            if name.startswith(("cp_mechanism__", "cp_long_mechanism_"))
        ),
    )
    if pretrained_encoder_state is not None:
        model.encoder.load_state_dict(pretrained_encoder_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    rng = np.random.default_rng(seed)
    train_positions = np.flatnonzero(train_mask.to_numpy())
    valid_positions = np.flatnonzero(validation.to_numpy())
    best_loss, best_state, patience = np.inf, None, 0
    batch_size = 768
    print(json.dumps({
        "event": "temporal_fit_start", "side": contract.side,
        "architecture": architecture, "fit_rows": int(train_mask.sum()),
        "validation_rows": int(validation.sum()),
    }), flush=True)
    for _epoch in range(epochs):
        model.train()
        order = train_positions[rng.permutation(len(train_positions))]
        for start in range(0, len(order), batch_size):
            position = order[start:start + batch_size]
            x_seq = torch.from_numpy(np.swapaxes(seq_all[position], 1, 2))
            x_static = torch.from_numpy(static_all[position])
            y_batch = torch.from_numpy(y[position])
            weight_batch = torch.from_numpy(weights[position])
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_seq, x_static)
            loss = (loss_fn(pred, y_batch) * weight_batch).sum() / weight_batch.sum().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            values: list[float] = []
            for start in range(0, len(valid_positions), 2048):
                position = valid_positions[start:start + 2048]
                pred = model(torch.from_numpy(np.swapaxes(seq_all[position], 1, 2)), torch.from_numpy(static_all[position]))
                values.append(float(torch.mean(loss_fn(pred, torch.from_numpy(y[position]))).item()))
            validation_loss = float(np.mean(values)) if values else np.inf
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
        if _epoch == 0 or (_epoch + 1) % 5 == 0:
            print(json.dumps({
                "event": "temporal_epoch", "side": contract.side,
                "architecture": architecture, "epoch": _epoch + 1,
                "validation_loss": validation_loss, "best_loss": best_loss,
                "patience": patience,
            }), flush=True)
    if best_state is None:
        return None
    model.load_state_dict(best_state)
    model.eval()
    ood = np.sqrt(np.mean(np.square(seq_train), axis=(1, 2)))
    support = group.loc[train_mask, "archetype_policy_key"].astype(str).value_counts().to_dict()
    return TemporalModel(
        side=contract.side, architecture=architecture, contract=contract,
        medians=medians, scales=scales, static_medians=static_medians, static_scales=static_scales,
        target_center=target_center, target_scale=target_scale, model=model,
        ood_q50=float(np.quantile(ood, 0.50)), ood_q95=float(np.quantile(ood, 0.95)),
        archetype_support={str(key): int(value) for key, value in support.items()},
        validation_loss=float(best_loss), fit_rows=int(train_mask.sum()),
    )


def _predict_temporal(model: TemporalModel | None, score: pd.DataFrame, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correction = np.zeros(len(score), dtype=np.float32)
    quality = np.zeros(len(score), dtype=np.float32)
    ood = np.full(len(score), np.nan, dtype=np.float32)
    if model is None or not len(score):
        return correction, quality, ood
    position = score["side_name"].eq(model.side).to_numpy()
    position &= np.isfinite(sequence).all(axis=(1, 2))
    if not position.any():
        return correction, quality, ood
    chosen = np.flatnonzero(position)
    static, _, _ = _static_matrix(
        score.loc[position], model.contract.static_features, model.contract.archetypes,
        model.static_medians, model.static_scales,
    )
    raw = sequence[position]
    filled = raw.copy()
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = np.take(model.medians, np.nonzero(missing)[2])
    normalised = np.clip((filled - model.medians) / model.scales, -8.0, 8.0).astype(np.float32)
    values = np.zeros(len(chosen), dtype=np.float32)
    for start in range(0, len(chosen), 2048):
        slice_ = slice(start, start + 2048)
        with torch.no_grad():
            pred = model.model(
                torch.from_numpy(np.swapaxes(normalised[slice_], 1, 2)),
                torch.from_numpy(static[slice_]),
            ).numpy()
        values[slice_] = pred.astype(np.float32)
    correction[chosen] = values * model.target_scale + model.target_center
    local_ood = np.sqrt(np.mean(np.square(normalised), axis=(1, 2)))
    ood[chosen] = local_ood
    denom = max(model.ood_q95 - model.ood_q50, 1e-5)
    ood_conf = np.clip((model.ood_q95 - local_ood) / denom, 0.0, 1.0)
    archetypes = score.loc[position, "archetype_policy_key"].astype(str).to_numpy()
    support = np.array([model.archetype_support.get(value, 0) for value in archetypes], dtype=np.float32)
    quality[chosen] = ood_conf * np.minimum(1.0, support / 1_800.0)
    return correction, quality, ood


def _write_metrics(output: Path, frame: pd.DataFrame, score: np.ndarray, arm: str) -> np.ndarray:
    selected = _fixed_activity_mask(frame, score)
    rows = _metric_rows(frame, selected, arm)
    pd.DataFrame(rows).to_csv(output / f"metrics_{arm}.csv", index=False)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-oos", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260713/ev_mapped_side_base_residual_expert_fullcurrent_top30_replay_20260714/oos_predictions.parquet"))
    parser.add_argument("--canonical-manifest", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260713/ev_mapped_side_base_residual_expert_fullcurrent_top30_replay_20260714/manifest.json"))
    parser.add_argument("--sparse-mlp-oos", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260713/canonical_meta_postprocessor_20260714/mlp_hier_ev_hpo20_expanding_sparse_v3_retry1/oos_predictions.parquet"))
    parser.add_argument("--scored-ledger", type=Path, default=Path("data_perp/reports/s59_h5_fullthroughjul10_base_configfull_freshmda_fixedparams_wf30_20260713/meta_handoff_top30_allsafe_aegmmfull_fullcoverage_20260714/s52_trailing_regime_scored_ledger.parquet"))
    parser.add_argument("--feature-root", type=Path, default=Path("data_perp/features/20260710_170000"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260714/sparse_parent_temporal_tcn_cnn_ablation_v1"))
    parser.add_argument("--architectures", default="tcn,cnn")
    parser.add_argument("--lookback-bars", type=int, default=16, help="Causal feature bars ending at the decision bar.")
    parser.add_argument("--bar-minutes", type=int, default=60, help="Feature-store cadence; current base/meta store is hourly.")
    parser.add_argument("--max-stale-bars", type=int, default=4, help="Maximum causal carry-forward gap; feature age is supplied to the model.")
    parser.add_argument(
        "--change-point-representation", action="store_true",
        help="Append continuous causal multiscale transition features to each residual model.",
    )
    parser.add_argument(
        "--select-change-point-features", action="store_true",
        help="Apply the diagnostic univariate CP selector; default retains conditional CP inputs.",
    )
    parser.add_argument(
        "--long-change-stream", action="store_true",
        help="Append compact streaming 48h/96h/168h change-state summaries.",
    )
    parser.add_argument(
        "--long-change-mechanism-bottleneck", action="store_true",
        help="Retain only bounded economic mechanisms and cross-scale summaries from the long stream.",
    )
    parser.add_argument(
        "--broad-market-channel-selection", action="store_true",
        help="Use the broad market-family channel selector without requiring encoder pretraining.",
    )
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--fit-rank-floor", type=float, default=0.80, help="Broader train-only residual region; deployment remains fixed top-10 activity.")
    parser.add_argument("--widths", default="24,32,24", help="Three causal-convolution widths.")
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.30)
    parser.add_argument("--embargo-hours", type=int, default=12)
    parser.add_argument("--self-supervised-pretrain", action="store_true")
    parser.add_argument("--pretrain-start", default="2025-01-01")
    parser.add_argument("--pretrain-end", default="2026-04-01")
    parser.add_argument("--pretrain-stride-hours", type=int, default=24)
    parser.add_argument("--pretrain-max-rows", type=int, default=60_000)
    parser.add_argument("--pretrain-max-channels", type=int, default=48)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    widths = tuple(int(value) for value in args.widths.split(","))
    if len(widths) != 3 or min(widths) < 8:
        raise ValueError("--widths must contain three integers of at least 8")
    if not 0.0 < args.fit_rank_floor < 0.95:
        raise ValueError("--fit-rank-floor must be in (0, 0.95)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_rows(args)
    frame = frame.loc[frame["__ts__"].between(pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC"), inclusive="left")].copy()
    frame["existing_sparse_parent_score"] = _num(frame, "expected_ev_rank_score", np.nan)
    if not np.isfinite(_num(frame, "existing_sparse_parent_score", np.nan)).all():
        raise ValueError("temporal ablation requires finite sparse-parent expected_ev_rank_score")
    forbidden = [name for name in frame.columns if "local_subtype" in str(name).lower()]
    if forbidden:
        frame = frame.drop(columns=forbidden)
    pretrain_store_candidates: list[str] = []
    if args.self_supervised_pretrain:
        pretrain_store_candidates = _market_store_candidates(args.feature_root)
        missing_market = [name for name in pretrain_store_candidates if name not in frame.columns]
        if missing_market:
            current_values, _ = _load_sequences(
                frame, args.feature_root, missing_market, 1,
                args.bar_minutes, args.max_stale_bars,
                allow_missing_channels=True,
            )
            projected = pd.DataFrame(
                current_values[:, 0, :], columns=missing_market, index=frame.index,
                dtype=np.float32,
            )
            frame = pd.concat([frame, projected], axis=1, copy=False)
            del current_values
    candidates = _feature_candidates(frame, args.feature_root)
    source_audit.update({
        "evaluation_rows": int(len(frame)),
        "candidate_temporal_channels": len(candidates),
        "pretrain_store_market_candidates": len(pretrain_store_candidates),
        "pretrain_store_market_candidates_present_after_projection": int(sum(name in frame for name in pretrain_store_candidates)),
        "pretrain_store_market_candidate_names": pretrain_store_candidates,
        "excluded_local_subtype_columns": forbidden,
        "temporal_input_contract": "feature-store causal bars only",
        "monthly_rows": _month_ids(frame).value_counts().sort_index().to_dict(),
    })
    if args.dry_run:
        print(json.dumps(source_audit, indent=2, default=str))
        return
    april = frame.loc[_month_ids(frame).eq("2026-04")].copy()
    contracts: dict[str, TemporalContract] = {}
    reports: list[pd.DataFrame] = []
    for side in ("long", "short"):
        report = _channel_scores(april, candidates, side, args.fit_rank_floor)
        if args.self_supervised_pretrain or args.broad_market_channel_selection:
            channels, report = _select_pretraining_channels(
                report, april, ceiling=args.pretrain_max_channels
            )
        else:
            channels, report = _select_channels(report)
        if len(channels) < 4:
            raise RuntimeError(f"insufficient temporal channels for {side}")
        channels.append(FEATURE_AGE_CHANNEL)
        reports.append(report)
        arches = sorted(april.loc[april["side_name"].eq(side), "archetype_policy_key"].astype(str).unique())
        static = _static_features(april)
        contracts[side] = TemporalContract(side, channels, report.loc[report["selected"]].to_dict("records"), static, arches, args.lookback_bars, {})
    feature_report = pd.concat(reports, ignore_index=True)
    feature_report.to_csv(args.output_dir / "temporal_channel_selection_april.csv", index=False)
    allowed_architectures = {
        "tcn", "cnn", "mlp", "tcn_interaction", "cnn_interaction", "mlp_interaction",
        "tcn_lowrank", "cnn_lowrank", "mlp_lowrank",
    }
    architectures = [
        value.strip() for value in args.architectures.split(",")
        if value.strip() in allowed_architectures
    ]
    if not architectures:
        raise ValueError(f"--architectures must contain values from {sorted(allowed_architectures)}")
    pretrained_states: dict[tuple[str, str], dict[str, torch.Tensor] | None] = {}
    pretraining_manifest: dict[str, Any] = {}
    if args.self_supervised_pretrain:
        historical_rows = _historical_pretrain_rows(
            sorted(frame["__symbol__"].astype(str).unique()),
            pd.Timestamp(args.pretrain_start, tz="UTC"),
            pd.Timestamp(args.pretrain_end, tz="UTC"),
            args.pretrain_stride_hours,
            args.pretrain_max_rows,
        )
        historical_union = list(dict.fromkeys([
            channel for contract in contracts.values() for channel in contract.channels
        ]))
        historical_sequence, _ = _load_sequences(
            historical_rows, args.feature_root, historical_union,
            args.lookback_bars, args.bar_minutes, args.max_stale_bars,
        )
        historical_by_channel = {name: index for index, name in enumerate(historical_union)}
        for architecture in architectures:
            for side, contract in contracts.items():
                indices = [historical_by_channel[name] for name in contract.channels]
                if architecture.startswith("mlp"):
                    state, audit = None, {
                        "reason": "self_supervised_pretraining_reserved_for_causal_convolution"
                    }
                else:
                    state, audit = _pretrain_encoder(
                        historical_sequence[:, :, indices], architecture, widths,
                        args.dropout, args.weight_decay, args.pretrain_epochs,
                        args.seed + (0 if side == "long" else 100) + (0 if architecture == "tcn" else 10),
                    )
                pretrained_states[(architecture, side)] = state
                pretraining_manifest[f"{architecture}::{side}"] = audit
        del historical_sequence, historical_rows
    # Sequence construction depends only on the row timestamp and frozen
    # feature contract. Materialise it once, then slice by expanding fold. This
    # avoids rereading every symbol for each train/validation pair.
    frame = frame.reset_index(drop=True)
    channel_union = list(dict.fromkeys([channel for contract in contracts.values() for channel in contract.channels]))
    sequence_all, sequence_available_all = _load_sequences(
        frame, args.feature_root, channel_union, args.lookback_bars,
        args.bar_minutes, args.max_stale_bars,
    )
    by_channel = {name: index for index, name in enumerate(channel_union)}
    change_state_manifest: dict[str, Any] = {}
    if args.long_change_stream:
        long_columns: dict[str, np.ndarray] = {}
        long_manifest: dict[str, Any] = {}
        for side, contract in contracts.items():
            source_names = [name for name in contract.channels if name != FEATURE_AGE_CHANNEL]
            summaries, audit = _load_streaming_long_change_summaries(
                frame, args.feature_root, source_names,
                args.bar_minutes, args.max_stale_bars,
                mechanism_only=args.long_change_mechanism_bottleneck,
            )
            side_positions = frame["side_name"].eq(side).to_numpy()
            for name in summaries.columns:
                destination = long_columns.setdefault(
                    name, np.full(len(frame), np.nan, dtype=np.float32)
                )
                destination[side_positions] = summaries.loc[side_positions, name].to_numpy(
                    dtype=np.float32, copy=False
                )
            contract.static_features = list(dict.fromkeys([
                *contract.static_features, *summaries.columns.tolist()
            ]))
            long_manifest[side] = audit
        frame = pd.concat(
            [frame, pd.DataFrame(long_columns, index=frame.index, dtype=np.float32)],
            axis=1, copy=False,
        )
        change_state_manifest["streaming_long_state"] = long_manifest
    if args.change_point_representation:
        change_columns: dict[str, np.ndarray] = {}
        side_change_names: dict[str, list[str]] = {}
        for side, contract in contracts.items():
            side_positions = frame["side_name"].eq(side).to_numpy()
            source_names = [name for name in contract.channels if name != FEATURE_AGE_CHANNEL]
            source_indices = [by_channel[name] for name in source_names]
            change_matrix, change_names = build_causal_change_state(
                sequence_all[:, :, source_indices], source_names,
            )
            side_change_names[side] = change_names
            for index, name in enumerate(change_names):
                destination = change_columns.setdefault(
                    name, np.full(len(frame), np.nan, dtype=np.float32)
                )
                destination[side_positions] = change_matrix[side_positions, index]
            change_state_manifest[side] = {
                "source_channels": source_names,
                "candidate_output_features": change_names,
                "finite_coverage": float(np.isfinite(change_matrix[side_positions]).mean()),
            }
        frame = pd.concat(
            [frame, pd.DataFrame(change_columns, index=frame.index, dtype=np.float32)],
            axis=1,
            copy=False,
        )
        cp_reports: list[pd.DataFrame] = []
        april_with_change = frame.loc[_month_ids(frame).eq("2026-04")].copy()
        for side, contract in contracts.items():
            cp_report = _channel_scores(
                april_with_change, side_change_names[side], side, args.fit_rank_floor
            )
            if args.select_change_point_features:
                selected_cp, cp_report = _select_channels(cp_report, floor=12, ceiling=36)
            else:
                selected_cp = list(side_change_names[side])
                cp_report["selected"] = cp_report["feature"].isin(selected_cp)
            contract.static_features = list(dict.fromkeys([*contract.static_features, *selected_cp]))
            cp_report["selection_layer"] = "causal_change_state"
            cp_reports.append(cp_report)
            change_state_manifest[side]["selection_mode"] = (
                "univariate_diagnostic" if args.select_change_point_features else "retain_for_conditional_interactions"
            )
            change_state_manifest[side]["selected_output_features"] = selected_cp
            change_state_manifest[side]["selected_output_count"] = len(selected_cp)
        pd.concat(cp_reports, ignore_index=True).to_csv(
            args.output_dir / "change_point_selection_april.csv", index=False
        )
    # Reuse the same arrays for both encoders and every expanding fold.
    all_scored: dict[str, list[pd.DataFrame]] = {architecture: [] for architecture in architectures}
    fold_manifest: list[dict[str, Any]] = []
    for fold_no, month in enumerate(("2026-05", "2026-06", "2026-07")):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        train_mask = (frame["__ts__"] < start).to_numpy()
        valid_mask = frame["__ts__"].between(start, end, inclusive="left").to_numpy()
        train = frame.loc[train_mask].copy().reset_index(drop=True)
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        valid["sparse_parent_rank_score"] = _num(valid, "existing_sparse_parent_score")
        valid["parent_expected_ev"] = np.nan
        for side in ("long", "short"):
            train_side = train.loc[train["side_name"].eq(side)]
            valid_side = valid["side_name"].eq(side).to_numpy()
            ev_map = fit_hierarchical_ev_calibrator(
                train_side,
                _num(train_side, "existing_sparse_parent_score"),
                _num(train_side, "ev_after_1pct"),
                shrink_rows=1_500.0,
                min_local_rows=500,
                local_weight_cap=0.65,
                tail_weight_top10=5.0,
                tail_weight_top20=2.5,
            )
            valid.loc[valid_side, "parent_expected_ev"] = predict_hierarchical_ev(
                ev_map,
                valid.loc[valid_side],
                _num(valid.loc[valid_side], "existing_sparse_parent_score"),
            )
        if not np.isfinite(_num(valid, "parent_expected_ev", np.nan)).all():
            raise ValueError(f"non-finite train-only parent EV map in {month}")
        train_seq_union = sequence_all[train_mask]
        valid_seq_union = sequence_all[valid_mask]
        train_contiguous = sequence_available_all[train_mask]
        valid_contiguous = sequence_available_all[valid_mask]
        side_sequences: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        coverage_details: dict[str, Any] = {}
        for side, contract in contracts.items():
            channel_idx = [by_channel[name] for name in contract.channels]
            train_seq = train_seq_union[:, :, channel_idx]
            valid_seq = valid_seq_union[:, :, channel_idx]
            keep_channels, joint, coverage = _prune_joint_coverage(train_seq, contract.channels)
            if len(keep_channels) < 4:
                raise RuntimeError(f"joint sequence coverage dropped {side} below four channels")
            if keep_channels != contract.channels:
                # The frozen April selection must maintain coverage in later
                # train windows.  Removing a later feature would alter the OOS
                # contract, so fail loudly instead of silently drifting.
                raise RuntimeError(f"frozen {side} temporal contract fails 90% joint coverage in {month}: {coverage}")
            coverage["train_contiguous_rate"] = float(train_contiguous.mean())
            coverage["valid_contiguous_rate"] = float(valid_contiguous.mean())
            coverage_details[side] = coverage
            side_sequences[side] = (train_seq, valid_seq)
        fold_models: dict[str, dict[str, TemporalModel | None]] = {architecture: {} for architecture in architectures}
        for architecture in architectures:
            for side in ("long", "short"):
                fold_models[architecture][side] = _fit_temporal_model(
                    train, side_sequences[side][0], contracts[side], architecture,
                    args.seed + fold_no * 1_000 + (0 if side == "long" else 100) + (0 if architecture == "tcn" else 10),
                    args.epochs, args.dropout, args.embargo_hours,
                    args.fit_rank_floor, widths, args.learning_rate, args.weight_decay,
                    pretrained_states.get((architecture, side)),
                )
        for architecture in architectures:
            scored = valid.copy()
            scored["temporal_correction"] = 0.0
            scored["temporal_quality"] = 0.0
            scored["temporal_ood_distance"] = np.nan
            for side in ("long", "short"):
                correction, quality, ood = _predict_temporal(fold_models[architecture][side], scored, side_sequences[side][1])
                # A missing/gapped feature-store lookback is not neutral
                # evidence.  Retain the frozen parent score and withhold the
                # temporal correction until a complete causal window exists.
                incomplete = ~np.isfinite(side_sequences[side][1]).all(axis=(1, 2))
                correction[incomplete] = 0.0
                quality[incomplete] = 0.0
                ood[incomplete] = np.nan
                scored["temporal_correction"] += correction
                scored["temporal_quality"] += quality
                assigned = scored["side_name"].eq(side).to_numpy()
                scored.loc[assigned, "temporal_ood_distance"] = ood[assigned]
            scored["temporal_sequence_complete"] = valid_contiguous
            prior = pd.concat(all_scored[architecture], ignore_index=True) if all_scored[architecture] else pd.DataFrame(columns=scored.columns)
            # Same conservative prior-OOS policy selection as the MLP ablation;
            # May is necessarily a no-op because no completed prior OOS exists.
            renamed = prior.rename(columns={"temporal_correction": "mlp_correction", "temporal_quality": "mlp_quality"})
            alpha, gates, policy = _policy_params(renamed)
            gate_array = np.array([gates.get((str(side), str(arch)), False) for side, arch in zip(scored["side_name"], scored["archetype_policy_key"])], dtype=np.float32)
            scored["temporal_alpha"] = np.float32(alpha)
            scored["temporal_gate"] = gate_array
            scored["temporal_adjusted_score"] = _num(scored, "parent_expected_ev") + alpha * _num(scored, "temporal_correction") * _num(scored, "temporal_quality") * gate_array
            all_scored[architecture].append(scored)
            fold_manifest.append({
                "fold_month": month, "architecture": architecture, "train_rows": int(len(train)), "valid_rows": int(len(valid)),
                "prior_oos_rows": int(len(prior)), "policy_alpha": float(alpha), "policy": policy,
                "enabled_local_gates": int(sum(gates.values())), "models": {
                    side: None if fold_models[architecture][side] is None else {
                        "fit_rows": fold_models[architecture][side].fit_rows,
                        "validation_loss": fold_models[architecture][side].validation_loss,
                    } for side in ("long", "short")
                }, "sequence_coverage": coverage_details,
            })
            print(json.dumps({"event": "temporal_fold_complete", "month": month, "architecture": architecture, "alpha": alpha, "prior_oos_rows": len(prior)}), flush=True)
    summaries: list[dict[str, Any]] = []
    baseline_frame = pd.concat(next(iter(all_scored.values())), ignore_index=True)
    sparse_parent_score = _num(baseline_frame, "sparse_parent_rank_score")
    parent_selected = _write_metrics(args.output_dir, baseline_frame, sparse_parent_score, "sparse_parent")
    parent_stats = _stability(baseline_frame, parent_selected)
    mapped_selected = _write_metrics(
        args.output_dir, baseline_frame, _num(baseline_frame, "parent_expected_ev"), "ev_mapped_parent"
    )
    mapped_stats = _stability(baseline_frame, mapped_selected)
    summaries.append({
        "arm": "ev_mapped_parent", "selected_rows": int(mapped_selected.sum()), **mapped_stats,
        "delta_mean_ev_vs_parent": float(mapped_stats["mean_ev"] - parent_stats["mean_ev"]),
        "delta_worst_week_vs_parent": float(mapped_stats["worst_week"] - parent_stats["worst_week"]),
        "delta_worst_month_vs_parent": float(mapped_stats["worst_month"] - parent_stats["worst_month"]),
    })
    for architecture, parts in all_scored.items():
        scored = pd.concat(parts, ignore_index=True)
        score = _num(scored, "temporal_adjusted_score")
        selected = _write_metrics(args.output_dir, scored, score, architecture)
        stats = _stability(scored, selected)
        scored.to_parquet(args.output_dir / f"oos_predictions_{architecture}.parquet", index=False)
        summaries.append({
            "arm": architecture, "selected_rows": int(selected.sum()), **stats,
            "delta_mean_ev_vs_parent": float(stats["mean_ev"] - parent_stats["mean_ev"]),
            "delta_worst_week_vs_parent": float(stats["worst_week"] - parent_stats["worst_week"]),
            "delta_worst_month_vs_parent": float(stats["worst_month"] - parent_stats["worst_month"]),
            "delta_mean_ev_vs_ev_mapped_parent": float(stats["mean_ev"] - mapped_stats["mean_ev"]),
            "promotable_vs_sparse_parent": bool(
                stats["mean_ev"] > parent_stats["mean_ev"]
                and stats["worst_week"] >= parent_stats["worst_week"]
                and stats["worst_month"] >= parent_stats["worst_month"]
            ),
        })
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "fold_manifest.json").write_text(json.dumps(fold_manifest, indent=2, default=str) + "\n")
    source_audit["contracts"] = {side: {"channels": contract.channels, "static_anchors": contract.static_features, "archetypes": contract.archetypes} for side, contract in contracts.items()}
    source_audit["leakage_contract"] = {
        "sub_archetypes_used": False,
        "sequence_end": "decision timestamp inclusive",
        "feature_bar_minutes": args.bar_minutes,
        "max_causal_staleness_bars": args.max_stale_bars,
        "fit_rank_floor": args.fit_rank_floor,
        "widths": widths,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_only_channel_selection": "April 2026 before May OOS",
        "expanding_fit_windows": ["before 2026-05", "before 2026-06", "before 2026-07"],
        "label_overlap_embargo_hours": args.embargo_hours,
        "policy_blend_evidence": "completed earlier OOS months only",
        "parent_score": "frozen existing_sparse_parent_score mapped train-only into side/archetype EV units",
        "residual_target": "ev_after_1pct minus causal expectation of existing_sparse_parent_score",
        "change_point_representation": bool(args.change_point_representation),
        "select_change_point_features": bool(args.select_change_point_features),
        "long_change_stream": bool(args.long_change_stream),
        "long_change_mechanism_bottleneck": bool(args.long_change_mechanism_bottleneck),
        "broad_market_channel_selection": bool(args.broad_market_channel_selection),
        "change_point_manifest": change_state_manifest,
        "self_supervised_pretrain": bool(args.self_supervised_pretrain),
        "pretraining_period": [args.pretrain_start, args.pretrain_end],
        "pretraining_candidate_prefixes": list(MARKET_PRETRAIN_PREFIXES),
        "pretraining_manifest": pretraining_manifest,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(source_audit, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
