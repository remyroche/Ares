"""Shared, immutable data contract for the feature/leaf portability study.

The study deliberately evaluates one frozen candidate population.  This module
is the only place that joins the stored candidate panel to the TP6/SL4 H12
economics and the R3 robust-clear labels.  Keeping the join here makes it
harder for a later ablation to accidentally change entry, cost, labels, or
the side-local base feature contract.

All returned timestamps are UTC.  ``decision_ts`` refers to the completed
candidate bar; the underlying source contract enters at the next hourly open.
``label_available_ts`` is the next-open-plus-H12 resolution timestamp and must be used by
callers for every training/prequential boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
DEFAULT_WINNER = ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1"
DEFAULT_ROBUST = ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1"
DEFAULT_FEATURE_MANIFEST = ROOT / "data_perp/artifacts/full_universe_base_hpo_mda_20260802_v1"

SIDES: tuple[str, ...] = ("long", "short")
TP6_SL4_COST_BPS = 100.0
# The barrier path is H12 *from the next-hour entry*.  The decision occurs at
# the completed source bar, therefore its label resolves after the one-hour
# entry delay plus the twelve-hour path.
HORIZON_HOURS = 12
LABEL_RESOLUTION_HOURS = 13

# This is a *frozen existing control* context contract.  The feature/leaf
# experiment is allowed to add only artifacts it materialises causally from
# it or from strict OOF base reasoning; it is not allowed to silently import a
# later latent-state representation.
FROZEN_META_CONTEXT: tuple[str, ...] = (
    "mkt_ret_eq_24h",
    "regime_liquidity_score",
    "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h",
    "mkt_funding_dispersion",
    "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score",
    "negative_breadth_pct",
    "btc_resilience_alt_weakness",
    "short_covering_score_market",
    "deleveraging_without_followthrough",
    "short_signal_recovery_conflict",
    "market_state_transition_entropy_5d",
    "breakout_retention_4h",
)


class TP6PortabilityDataError(ValueError):
    """Raised when the frozen source data violates its study contract."""


@dataclass(frozen=True)
class TP6PortabilityContract:
    """Paths and immutable semantic details required by every ablation."""

    panel: Path = DEFAULT_PANEL
    winner: Path = DEFAULT_WINNER
    robust: Path = DEFAULT_ROBUST
    feature_manifest: Path = DEFAULT_FEATURE_MANIFEST
    geometry: str = "TP6/SL4/H12"
    entry: str = "next_hourly_open_after_candidate_bar_close"
    cost_bps: float = TP6_SL4_COST_BPS
    target: str = "R3 robust-clear/adverse/weak; clear requires cost +25bps before lower barrier"

    def __post_init__(self) -> None:
        for field in ("panel", "winner", "robust", "feature_manifest"):
            path = Path(getattr(self, field))
            if not path.exists():
                raise TP6PortabilityDataError(f"required frozen contract path is absent: {path}")
            object.__setattr__(self, field, path)
        if not np.isfinite(float(self.cost_bps)) or float(self.cost_bps) < 0:
            raise TP6PortabilityDataError("cost_bps must be finite and non-negative")


def frozen_base_features(contract: TP6PortabilityContract, side: str) -> list[str]:
    """Return the retained side-local 32-feature T2 selection exactly."""
    if side not in SIDES:
        raise TP6PortabilityDataError(f"unknown canonical side: {side!r}")
    path = contract.feature_manifest / side / "target_family_manifest.json"
    if not path.exists():
        raise TP6PortabilityDataError(f"missing selected-feature manifest: {path}")
    payload = json.loads(path.read_text())
    key = f"T2_soft_barrier|tp3_sl2|{side}"
    features = list(payload.get("feature_contract", {}).get(key, []))
    if not 30 <= len(features) <= 40 or len(features) != len(set(features)):
        raise TP6PortabilityDataError(
            f"unexpected side-local frozen feature contract for {side}: {len(features)} fields"
        )
    return features


def all_frozen_base_features(contract: TP6PortabilityContract) -> dict[str, list[str]]:
    return {side: frozen_base_features(contract, side) for side in SIDES}


def _utc(value: object | None, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    out = pd.Timestamp(value)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    if pd.isna(out):
        raise TP6PortabilityDataError(f"{name} must be a finite UTC timestamp")
    return out


def _part_names(contract: TP6PortabilityContract) -> list[str]:
    names = sorted(path.name for path in (contract.panel / "parts").glob("*.parquet"))
    if not names:
        raise TP6PortabilityDataError("candidate panel contains no parquet parts")
    missing = [
        name
        for name in names
        if not (contract.winner / "parts" / name).exists()
        or not (contract.robust / "parts" / name).exists()
    ]
    if missing:
        raise TP6PortabilityDataError(f"source parts are incomplete: {missing[:8]}")
    return names


def _asset_from_candidate_id(values: pd.Series) -> pd.Series:
    # Candidate IDs are ``<symbol>|<timestamp>|1h|<side>``.  Parsing this
    # explicit ID preserves the source's per-asset definition without asking
    # a later feature generator to reconstruct it.
    return values.astype(str).str.split("|", n=1, regex=False).str[0].astype("string")


def _read_one_part(
    contract: TP6PortabilityContract,
    name: str,
    *,
    columns: Sequence[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    sides: Sequence[str],
) -> pd.DataFrame:
    panel_columns = ["candidate_id", "__ts__", "side_name", *columns]
    panel_columns = list(dict.fromkeys(panel_columns))
    panel_path = contract.panel / "parts" / name
    # Parquet metadata avoids a second, accidental 780-column data read for
    # every asset part merely to validate the contract.
    available = set(pq.ParquetFile(panel_path).schema_arrow.names)
    missing = sorted(set(panel_columns).difference(available))
    if missing:
        raise TP6PortabilityDataError(f"{name} lacks frozen inference columns: {missing[:10]}")
    panel = pd.read_parquet(panel_path, columns=panel_columns)
    panel["decision_ts"] = pd.to_datetime(panel.pop("__ts__"), utc=True, errors="coerce")
    if panel["decision_ts"].isna().any():
        raise TP6PortabilityDataError(f"{name} has invalid decision timestamps")
    panel = panel.loc[panel["side_name"].isin(tuple(sides))]
    if start is not None:
        panel = panel.loc[panel["decision_ts"].ge(start)]
    if end is not None:
        panel = panel.loc[panel["decision_ts"].lt(end)]
    if panel.empty:
        return panel

    winner = pd.read_parquet(
        contract.winner / "parts" / name,
        columns=["candidate_id", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "__label_available_at__"],
    ).rename(
        columns={
            "t4_tp6_sl4_gross_bps": "gross_bps",
            "t4_tp6_sl4_net_bps": "net_bps",
            "__label_available_at__": "label_available_ts",
        }
    )
    robust_columns = [
        "candidate_id", "label_valid", "lower_touch_minute",
        "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50",
    ]
    robust = pd.read_parquet(contract.robust / "parts" / name, columns=robust_columns)
    out = (
        panel.merge(winner, on="candidate_id", how="inner", validate="one_to_one")
        .merge(robust, on="candidate_id", how="left", validate="one_to_one")
    )
    out["label_available_ts"] = pd.to_datetime(out["label_available_ts"], utc=True, errors="coerce")
    if out["label_available_ts"].isna().any():
        raise TP6PortabilityDataError(f"{name} has invalid H12 label timestamps")
    # Invalid paths remain observable in the population only when requested;
    # never silently coerce them into weak or adverse economic labels.
    out["label_valid"] = out["label_valid"].fillna(False).astype(bool)
    out["asset"] = _asset_from_candidate_id(out["candidate_id"])
    # The source panel is intentionally broad enough to serve F0--F3.  Keep
    # its numeric contract compact while it is still part-sized; waiting until
    # after ``concat`` would briefly retain a second full float64 population.
    # All supplied model/context fields are continuous and LightGBM consumes
    # float32 natively.  The path flags keep their compact nullable-free
    # integer representation below when the R3 target is formed.
    compact_numeric = [
        *columns,
        "gross_bps",
        "net_bps",
        "lower_touch_minute",
        "robust_clear_event_b0",
        "robust_clear_event_b25",
        "robust_clear_event_b50",
    ]
    for field in dict.fromkeys(compact_numeric):
        if field in out:
            out[field] = pd.to_numeric(out[field], errors="coerce").astype(np.float32)
    return out


def load_tp6_population(
    *,
    contract: TP6PortabilityContract = TP6PortabilityContract(),
    columns: Sequence[str] = (),
    start: object | None = None,
    end: object | None = None,
    sides: Sequence[str] = SIDES,
    valid_labels_only: bool = True,
) -> pd.DataFrame:
    """Load one exact candidate population with side-preserving H12 labels.

    The call is intentionally column-pruned.  Large ablations should pass
    only frozen base fields, declared meta context, and explicitly predeclared
    diagnostic fields; no full 780-column read is necessary.
    """
    start_ts, end_ts = _utc(start, name="start"), _utc(end, name="end")
    if start_ts is not None and end_ts is not None and end_ts <= start_ts:
        raise TP6PortabilityDataError("end must be after start")
    selected_sides = tuple(dict.fromkeys(map(str, sides)))
    if not selected_sides or not set(selected_sides).issubset(SIDES):
        raise TP6PortabilityDataError("sides must be a non-empty subset of canonical long/short")
    selected_columns = tuple(dict.fromkeys(map(str, columns)))
    forbidden = {
        "gross_bps", "net_bps", "label_available_ts", "decision_ts", "candidate_id",
        "side_name", "asset", "label_valid", "r3_class",
    }
    if overlap := sorted(set(selected_columns).intersection(forbidden)):
        raise TP6PortabilityDataError(f"source columns duplicate contract fields: {overlap}")
    parts = [
        _read_one_part(
            contract, name, columns=selected_columns, start=start_ts, end=end_ts,
            sides=selected_sides,
        )
        for name in _part_names(contract)
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        raise TP6PortabilityDataError("requested range contains no candidates")
    out = pd.concat(parts, ignore_index=True)
    # Asset/side are inference identifiers, not numerical learning features.
    # Categorical storage avoids retaining millions of duplicate Python
    # strings throughout the sequential ablations.
    out["asset"] = out["asset"].astype("category")
    out["side_name"] = out["side_name"].astype("category")
    if out["candidate_id"].duplicated().any():
        raise TP6PortabilityDataError("candidate IDs must be unique after source join")
    if valid_labels_only:
        out = out.loc[out["label_valid"]].copy()
    if out.empty:
        raise TP6PortabilityDataError("requested range contains no valid H12 labels")
    if not np.allclose(
        out["gross_bps"].to_numpy(float) - float(contract.cost_bps),
        out["net_bps"].to_numpy(float), atol=2e-3, rtol=0.0,
    ):
        raise TP6PortabilityDataError("gross/net contract does not charge the declared cost exactly once")
    horizon = (out["label_available_ts"] - out["decision_ts"]).dt.total_seconds().to_numpy(float) / 3600.0
    if not np.allclose(horizon, LABEL_RESOLUTION_HOURS, atol=1e-6, rtol=0.0):
        raise TP6PortabilityDataError(
            "the study requires H12 paths resolving exactly 13h after the decision bar"
        )
    out["r3_class"] = np.select(
        [out["robust_clear_event_b25"].eq(1), out["lower_touch_minute"].ge(0)],
        [2, 0],
        default=1,
    ).astype(np.int8)
    # Deterministic ordering is a causal guard as well as a reproducibility
    # requirement for all prequential maps and rolling transforms.
    return out.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)


def frozen_input_columns(contract: TP6PortabilityContract) -> list[str]:
    """The complete raw input set needed for base and control-meta arms."""
    base = all_frozen_base_features(contract)
    return list(dict.fromkeys([*base["long"], *base["short"], *FROZEN_META_CONTEXT, "atr_1h"]))


__all__ = [
    "DEFAULT_FEATURE_MANIFEST", "DEFAULT_PANEL", "DEFAULT_ROBUST", "DEFAULT_WINNER",
    "FROZEN_META_CONTEXT", "HORIZON_HOURS", "LABEL_RESOLUTION_HOURS", "ROOT", "SIDES", "TP6_SL4_COST_BPS",
    "TP6PortabilityContract", "TP6PortabilityDataError", "all_frozen_base_features",
    "frozen_base_features", "frozen_input_columns", "load_tp6_population",
]
