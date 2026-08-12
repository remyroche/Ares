"""Strict, reusable Transport A/B evaluation for the frozen TP6 stack.

This module deliberately does not fit a model, choose a feature, or build a
label.  It is the one place a portability ablation can prove that its input
scores have a prior-resolved lineage and then evaluate them consistently:

* train labels resolve before the first held-out decision;
* every score/map/calibrator reference ends no later than its row decision;
* economics are already in common net/gross bps with the 100-bps cost charged
  exactly once; and
* top-k selection is global across long and short candidates, followed only
  by descriptive side/month/quarter splits.

The helper intentionally has no regime-state dependency.  A caller supplies
the frozen base/meta score and any already selected causal feature columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TP6_COST_BPS = 100.0
TP6_TARGET_HORIZON_HOURS = 12.0
# The path itself is H12 from next-hour entry.  The source contract records
# label availability one additional hour later than the decision identity.
TP6_LABEL_RESOLUTION_HOURS = 13.0
DEFAULT_TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10)
IDENTITY_COLUMNS: tuple[str, ...] = ("candidate_id", "side_name", "decision_ts")
ECONOMIC_COLUMNS: tuple[str, ...] = ("net_bps", "gross_bps")

# These are outcomes, identities, timestamps, or target-bearing control
# columns.  They cannot enter a score feature contract.  The explicit
# prequential value map is intentionally not blacklisted: it is allowed only
# when its own prior-resolved provenance is declared to ``evaluate_transport``.
FORBIDDEN_MODEL_FEATURES: frozenset[str] = frozenset(
    {
        "candidate_id",
        "side_name",
        "__ts__",
        "decision_ts",
        "label_available_ts",
        "net_bps",
        "gross_bps",
        "exact_net_bps",
        "exact_gross_bps",
        "r3_class",
        "r3_metric_target",
        "event",
        "target",
        "target_invalid",
        "label_valid",
        "path_complete",
    }
)


class TP6TransportValidationError(ValueError):
    """Raised when a portability input violates the frozen evaluation contract."""


def _as_utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise TP6TransportValidationError(f"{name} must be a finite UTC timestamp")
    return timestamp


@dataclass(frozen=True)
class TimeWindow:
    """Half-open UTC decision-time window used by a transport split."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start = _as_utc(self.start, name="window start")
        end = _as_utc(self.end, name="window end")
        if end <= start:
            raise TP6TransportValidationError("a transport window must have end > start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class TransportDefinition:
    """Chronological train/test windows with no implicit gap filling."""

    name: str
    train_windows: tuple[TimeWindow, ...]
    test_windows: tuple[TimeWindow, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise TP6TransportValidationError("a transport name is required")
        if not self.train_windows or not self.test_windows:
            raise TP6TransportValidationError("transport requires both train and test windows")
        for kind, windows in (("train", self.train_windows), ("test", self.test_windows)):
            ordered = sorted(windows, key=lambda x: x.start)
            if tuple(ordered) != tuple(windows):
                raise TP6TransportValidationError(f"{kind} windows must be chronological")
            if any(left.end > right.start for left, right in zip(windows, windows[1:])):
                raise TP6TransportValidationError(f"{kind} windows may not overlap")
        if max(window.end for window in self.train_windows) > min(window.start for window in self.test_windows):
            raise TP6TransportValidationError("all train windows must end before the first test window")

    @property
    def first_test_decision(self) -> pd.Timestamp:
        return self.test_windows[0].start


# The names and contiguous windows are the authoritative portability contract.
# Transport B is H2-to-date: November is held aside as the final untouched OOS.
TRANSPORT_A = TransportDefinition(
    name="transport_2023q4_to_2024",
    train_windows=(TimeWindow("2023-04-01", "2024-01-01"),),
    test_windows=(TimeWindow("2024-01-01", "2024-07-01"),),
)
TRANSPORT_B = TransportDefinition(
    name="transport_2024h1_to_h2_to_date",
    train_windows=(TimeWindow("2023-04-01", "2024-07-01"),),
    test_windows=(TimeWindow("2024-07-01", "2024-11-01"),),
)
FINAL_OOS = TransportDefinition(
    name="final_oos_2024-11",
    train_windows=(TimeWindow("2023-04-01", "2024-11-01"),),
    test_windows=(TimeWindow("2024-11-01", "2024-12-01"),),
)
TRANSPORTS: tuple[TransportDefinition, ...] = (TRANSPORT_A, TRANSPORT_B, FINAL_OOS)

# ``TransportSpec`` is the portable-runner spelling.  Keep the descriptive
# class name too for direct use in this module and existing callers.
TransportSpec = TransportDefinition


def make_final_oos_spec(
    *,
    name: str,
    train_start: object,
    test_start: object,
    test_end: object,
) -> TransportDefinition:
    """Create a caller-declared contiguous final-OOS transport specification."""
    return TransportDefinition(
        name=str(name),
        train_windows=(TimeWindow(train_start, test_start),),
        test_windows=(TimeWindow(test_start, test_end),),
    )


@dataclass(frozen=True)
class TransportEvaluation:
    """Tables produced by one immutable Transport A/B evaluation."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_coverage: pd.DataFrame
    metrics: pd.DataFrame
    transport_gates: pd.DataFrame
    audit: dict[str, object]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, kind: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TP6TransportValidationError(f"missing {kind} columns: {missing[:12]}")


def _normalise_frame(
    frame: pd.DataFrame,
    *,
    score_column: str,
    expected_cost_bps: float,
    enforce_h12: bool,
) -> pd.DataFrame:
    _require_columns(
        frame,
        (*IDENTITY_COLUMNS, "label_available_ts", *ECONOMIC_COLUMNS, score_column),
        kind="TP6 transport",
    )
    if frame.empty:
        raise TP6TransportValidationError("TP6 transport input is empty")
    if not np.isfinite(float(expected_cost_bps)) or float(expected_cost_bps) < 0.0:
        raise TP6TransportValidationError("expected_cost_bps must be finite and non-negative")
    output = frame.copy()
    if output["candidate_id"].isna().any() or output["candidate_id"].duplicated().any():
        raise TP6TransportValidationError("candidate_id must be non-null and unique")
    output["decision_ts"] = pd.to_datetime(output["decision_ts"], utc=True, errors="coerce")
    output["label_available_ts"] = pd.to_datetime(output["label_available_ts"], utc=True, errors="coerce")
    if output[["decision_ts", "label_available_ts"]].isna().any().any():
        raise TP6TransportValidationError("decision_ts and label_available_ts must be valid UTC timestamps")
    if not set(output["side_name"].dropna().astype(str)).issubset({"long", "short"}):
        raise TP6TransportValidationError("side_name must contain only canonical long/short values")
    if output["side_name"].isna().any():
        raise TP6TransportValidationError("side_name must be non-null")
    numeric = output.loc[:, [*ECONOMIC_COLUMNS, score_column]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise TP6TransportValidationError("net, gross, and common-bps score must be finite")
    output.loc[:, numeric.columns] = numeric
    observed_cost = output["gross_bps"].to_numpy(float) - output["net_bps"].to_numpy(float)
    if not np.allclose(observed_cost, float(expected_cost_bps), atol=0.02, rtol=0.0):
        raise TP6TransportValidationError("gross_bps - net_bps must equal the declared cost exactly once")
    if (output["label_available_ts"] < output["decision_ts"]).any():
        raise TP6TransportValidationError("label availability cannot precede the decision")
    if enforce_h12:
        horizon_hours = (output["label_available_ts"] - output["decision_ts"]).dt.total_seconds().to_numpy(float) / 3600.0
        if not np.allclose(horizon_hours, TP6_LABEL_RESOLUTION_HOURS, atol=1e-6, rtol=0.0):
            raise TP6TransportValidationError(
                "TP6 H12 path labels must resolve exactly 13h after decision"
            )
    return output


def validate_feature_contract(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    min_coverage: float = 0.99,
) -> pd.DataFrame:
    """Fail closed on missing, forbidden, or under-covered model features.

    This validates the declared inference contract only.  It does not impute;
    callers use the resulting all-finite mask to make the scored population
    explicit in every metric table.
    """
    features = tuple(dict.fromkeys(map(str, feature_columns)))
    if not features:
        raise TP6TransportValidationError("a non-empty feature contract is required")
    if len(features) != len(feature_columns):
        raise TP6TransportValidationError("feature contract contains duplicates")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise TP6TransportValidationError("min_coverage must be in (0, 1]")
    _require_columns(frame, features, kind="feature contract")
    forbidden = sorted(set(features).intersection(FORBIDDEN_MODEL_FEATURES))
    if forbidden:
        raise TP6TransportValidationError(f"outcome/control fields cannot be model features: {forbidden}")
    numeric = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(float))
    rows = []
    for index, feature in enumerate(features):
        values = numeric.iloc[:, index].to_numpy(float)
        finite_values = values[finite[:, index]]
        coverage = float(finite[:, index].mean())
        rows.append(
            {
                "feature": feature,
                "rows": int(len(frame)),
                "finite_rows": int(finite[:, index].sum()),
                "finite_coverage": coverage,
                "unique_finite_values": int(np.unique(finite_values).size),
                "minimum_coverage": float(min_coverage),
                "passes_coverage": bool(coverage >= float(min_coverage)),
            }
        )
    result = pd.DataFrame(rows)
    failures = result.loc[~result["passes_coverage"], "feature"].tolist()
    if failures:
        raise TP6TransportValidationError(f"feature coverage below contract: {failures[:12]}")
    return result


def _window_mask(values: pd.Series, windows: Sequence[TimeWindow]) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for window in windows:
        mask |= ((values >= window.start) & (values < window.end)).to_numpy()
    return mask


def split_transport(
    frame: pd.DataFrame,
    transport: TransportDefinition,
    *,
    prior_resolved_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return the chronological transport split and prove its causal gates.

    ``prior_resolved_columns`` are per-row cutoff timestamps for every fitted
    score component that reached the supplied score (base fit, value map,
    meta fit, calibration, or admission map).  Requiring them prevents a
    caller from presenting an otherwise OOF-looking score that used future
    resolved outcomes in a post-processing statistic.
    """
    columns = tuple(dict.fromkeys(map(str, prior_resolved_columns)))
    if not columns:
        raise TP6TransportValidationError("at least one prior-resolved provenance column is required")
    _require_columns(frame, columns, kind="prior-resolved provenance")
    work = frame.copy()
    for column in columns:
        work[column] = pd.to_datetime(work[column], utc=True, errors="coerce")
        if work[column].isna().any():
            raise TP6TransportValidationError(f"{column} has missing/invalid prior-resolved timestamps")
        if (work[column] > work["decision_ts"]).any():
            raise TP6TransportValidationError(f"{column} is later than its row decision timestamp")
    train_mask = _window_mask(work["decision_ts"], transport.train_windows)
    test_mask = _window_mask(work["decision_ts"], transport.test_windows)
    if (train_mask & test_mask).any():
        raise TP6TransportValidationError("transport train/test decision windows overlap")
    train, test = work.loc[train_mask].copy(), work.loc[test_mask].copy()
    if train.empty or test.empty:
        raise TP6TransportValidationError(f"{transport.name} has an empty train or test population")
    first_test = transport.first_test_decision
    if not (train["label_available_ts"] < first_test).all():
        raise TP6TransportValidationError("training labels are not all resolved before the first test decision")
    # Score/state cutoffs must be prior to the actual score decision.  For a
    # pure train-to-test arm, the test cutoff must additionally precede the
    # entire held-out period; this catches a same-period expanding map unless
    # the caller deliberately materialises it as an independent prequential arm.
    for column in columns:
        if not (test[column] < test["decision_ts"]).all():
            raise TP6TransportValidationError(f"test {column} must be strictly earlier than the scored decision")
    audit: dict[str, object] = {
        "transport": transport.name,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_max_label_available_ts": train["label_available_ts"].max().isoformat(),
        "first_test_decision_ts": first_test.isoformat(),
        "train_labels_strictly_prior_resolved": True,
        "prior_resolved_columns": list(columns),
        "test_min_decision_ts": test["decision_ts"].min().isoformat(),
        "test_max_decision_ts": test["decision_ts"].max().isoformat(),
    }
    return train, test, audit


def _metric_row(
    selected: pd.DataFrame,
    population: pd.DataFrame,
    *,
    transport: str,
    top_fraction: float,
    scope: str,
    period: str | None = None,
    side_name: str | None = None,
) -> dict[str, object]:
    net = selected["net_bps"].to_numpy(float)
    gross = selected["gross_bps"].to_numpy(float)
    has_trades = len(selected) > 0
    return {
        "transport": transport,
        "ranking_basis": "global_common_bps_score",
        "top_fraction": float(top_fraction),
        "scope": scope,
        "period": period,
        "side_name": side_name,
        "population_rows": int(len(population)),
        "trades": int(len(selected)),
        "net_bps_per_trade": float(net.mean()) if has_trades else np.nan,
        "gross_bps_per_trade": float(gross.mean()) if has_trades else np.nan,
        "net_pnl_bps": float(net.sum()),
        "gross_pnl_bps": float(gross.sum()),
        "cost_bps_total": float((gross - net).sum()),
        "long_fraction": float(selected["side_name"].eq("long").mean()) if has_trades else np.nan,
        "min_decision_ts": selected["decision_ts"].min().isoformat() if has_trades else None,
        "max_decision_ts": selected["decision_ts"].max().isoformat() if has_trades else None,
    }


def global_common_bps_topk_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    transport: str,
    top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS,
) -> pd.DataFrame:
    """Rank once globally, then describe the selected tail by time and side.

    No per-timestamp or per-side ranking is performed.  ``side``/``month``/
    ``quarter`` rows are decompositions of the exact same globally selected
    candidates and cannot change admission membership.
    """
    if frame.empty:
        raise TP6TransportValidationError("cannot score an empty transport test frame")
    fractions = tuple(float(value) for value in top_fractions)
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise TP6TransportValidationError("top fractions must be non-empty values in (0, 1]")
    if len(set(fractions)) != len(fractions):
        raise TP6TransportValidationError("top fractions must be unique")
    ordered = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable")
    rows: list[dict[str, object]] = []
    by_side_population = {side: ordered.loc[ordered["side_name"].eq(side)] for side in ("long", "short")}
    # Calendar buckets are display-only diagnostics.  Deliberately strip the
    # already-normalised UTC timezone before period conversion to avoid a
    # pandas warning that could be mistaken for a local-time split.
    decision_calendar = ordered["decision_ts"].dt.tz_localize(None)
    month = decision_calendar.dt.to_period("M").astype(str)
    quarter = decision_calendar.dt.to_period("Q").astype(str)
    for fraction in fractions:
        selected = ordered.head(max(1, int(np.ceil(len(ordered) * fraction)))).copy()
        selected_calendar = selected["decision_ts"].dt.tz_localize(None)
        selected["__month__"] = selected_calendar.dt.to_period("M").astype(str)
        selected["__quarter__"] = selected_calendar.dt.to_period("Q").astype(str)
        rows.append(_metric_row(selected, ordered, transport=transport, top_fraction=fraction, scope="global"))
        for side, population in by_side_population.items():
            if not population.empty:
                part = selected.loc[selected["side_name"].eq(side)]
                rows.append(_metric_row(part, population, transport=transport, top_fraction=fraction, scope="side", side_name=side))
        for period in sorted(month.unique()):
            population = ordered.loc[month.eq(period)]
            part = selected.loc[selected["__month__"].eq(period)]
            rows.append(_metric_row(part, population, transport=transport, top_fraction=fraction, scope="month", period=str(period)))
        for period in sorted(quarter.unique()):
            population = ordered.loc[quarter.eq(period)]
            part = selected.loc[selected["__quarter__"].eq(period)]
            rows.append(_metric_row(part, population, transport=transport, top_fraction=fraction, scope="quarter", period=str(period)))
        for side, period in sorted(set(zip(ordered["side_name"].astype(str), month.astype(str)))):
            population = ordered.loc[ordered["side_name"].eq(side) & month.eq(period)]
            part = selected.loc[selected["side_name"].eq(side) & selected["__month__"].eq(period)]
            rows.append(_metric_row(part, population, transport=transport, top_fraction=fraction, scope="side_month", period=str(period), side_name=side))
    return pd.DataFrame(rows).sort_values(["top_fraction", "scope", "side_name", "period"], kind="stable", na_position="first").reset_index(drop=True)


def build_transport_gates(
    metrics: pd.DataFrame,
    *,
    min_net_bps_per_trade: float = 0.0,
    min_trades: int = 1,
) -> pd.DataFrame:
    """Create a row-level, auditable gate table consumable by ablation results.

    The result deliberately reports every global, side, month, quarter and
    side-month cell.  Consumers may impose stricter advancement logic, but
    cannot silently omit a losing period or side.
    """
    required = {"transport", "top_fraction", "scope", "period", "side_name", "trades", "net_bps_per_trade"}
    _require_columns(metrics, required, kind="metric")
    if int(min_trades) < 1:
        raise TP6TransportValidationError("min_trades must be positive")
    rows: list[dict[str, object]] = []
    for record in metrics.to_dict("records"):
        support_passed = int(record["trades"]) >= int(min_trades)
        rows.append(
            {
                "transport": record["transport"],
                "top_fraction": float(record["top_fraction"]),
                "scope": record["scope"],
                "period": record["period"],
                "side_name": record["side_name"],
                "gate": "minimum_trade_support",
                "value": float(record["trades"]),
                "threshold": float(min_trades),
                "passed": bool(support_passed),
            }
        )
        rows.append(
            {
                "transport": record["transport"],
                "top_fraction": float(record["top_fraction"]),
                "scope": record["scope"],
                "period": record["period"],
                "side_name": record["side_name"],
                "gate": "net_bps_per_trade",
                "value": float(record["net_bps_per_trade"]),
                "threshold": float(min_net_bps_per_trade),
                "passed": bool(support_passed and np.isfinite(float(record["net_bps_per_trade"])) and float(record["net_bps_per_trade"]) > float(min_net_bps_per_trade)),
            }
        )
    return pd.DataFrame(rows).sort_values(["transport", "top_fraction", "scope", "side_name", "period", "gate"], kind="stable", na_position="first").reset_index(drop=True)


def evaluate_transport(
    frame: pd.DataFrame,
    *,
    transport: TransportDefinition,
    score_column: str,
    feature_columns: Sequence[str],
    prior_resolved_columns: Sequence[str],
    expected_cost_bps: float = TP6_COST_BPS,
    min_feature_coverage: float = 0.99,
    top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS,
    enforce_h12: bool = True,
) -> TransportEvaluation:
    """Validate and evaluate a single frozen Transport A/B score surface."""
    work = _normalise_frame(
        frame,
        score_column=score_column,
        expected_cost_bps=expected_cost_bps,
        enforce_h12=enforce_h12,
    )
    coverage = validate_feature_contract(work, feature_columns, min_coverage=min_feature_coverage)
    train, test, audit = split_transport(work, transport, prior_resolved_columns=prior_resolved_columns)
    numeric = test.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    scored = test.loc[complete].copy()
    if scored.empty:
        raise TP6TransportValidationError("no test rows are complete on the frozen feature contract")
    audit.update(
        {
            "feature_columns": list(map(str, feature_columns)),
            "feature_contract_rows": int(len(test)),
            "feature_contract_complete_rows": int(complete.sum()),
            "feature_contract_incomplete_rows": int((~complete).sum()),
            "feature_contract_complete_fraction": float(complete.mean()),
            "expected_cost_bps": float(expected_cost_bps),
            "top_fractions": [float(value) for value in top_fractions],
            "ranking_basis": "global_common_bps_score",
        }
    )
    metrics = global_common_bps_topk_metrics(scored, score_column=score_column, transport=transport.name, top_fractions=top_fractions)
    gates = build_transport_gates(metrics)
    return TransportEvaluation(train=train, test=scored, feature_coverage=coverage, metrics=metrics, transport_gates=gates, audit=audit)


def write_transport_evaluation(result: TransportEvaluation, output_dir: Path) -> dict[str, Path]:
    """Persist the two canonical artifacts without fitting or changing a model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.parquet"
    gates_path = output_dir / "transport_gates.parquet"
    coverage_path = output_dir / "feature_contract_coverage.parquet"
    result.metrics.to_parquet(metrics_path, index=False)
    result.transport_gates.to_parquet(gates_path, index=False)
    result.feature_coverage.to_parquet(coverage_path, index=False)
    return {"metrics": metrics_path, "transport_gates": gates_path, "feature_contract_coverage": coverage_path}


__all__ = [
    "DEFAULT_TOP_FRACTIONS",
    "FINAL_OOS",
    "FORBIDDEN_MODEL_FEATURES",
    "TP6_COST_BPS",
    "TP6_LABEL_RESOLUTION_HOURS",
    "TP6_TARGET_HORIZON_HOURS",
    "TP6TransportValidationError",
    "TimeWindow",
    "TransportDefinition",
    "TransportSpec",
    "TransportEvaluation",
    "TRANSPORT_A",
    "TRANSPORT_B",
    "TRANSPORTS",
    "build_transport_gates",
    "evaluate_transport",
    "global_common_bps_topk_metrics",
    "make_final_oos_spec",
    "split_transport",
    "validate_feature_contract",
    "write_transport_evaluation",
]
