"""Strict-OOF, sequential R3 base-recall ablation utilities.

This is intentionally a narrow experiment surface.  It does not change the
candidate universe, entry convention, TP6/SL4 adverse definition, features,
or LGBM parameters.  It evaluates, in order:

1. score-only contrasts from an already strict-OOF canonical R3 B25 simplex;
2. a small set of exact pre-adverse robust-clear definitions and weights;
3. a query-aware ranker only when its preceding classifier candidate clears
   the declared R3 ranking gate.

Every supervised row must have a fully resolved H12 next-entry path.  The
path-derived MFE fields are labels/weights only and are rejected as features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


R3_COST_BPS = 100.0
R3_CLASSES = (0, 1, 2)  # adverse, weak/unresolved, robust clear
QUERY_COLUMNS = ("decision_ts", "side_name")
LABEL_COLUMNS = (
    "pre_adverse_mfe_bps",
    "atr_bps",
    "lower_touch_minute",
    "robust_clear_event_b0",
    "robust_clear_event_b25",
    "robust_clear_event_b50",
)


class BaseR3RecallAblationError(ValueError):
    """Raised when an ablation cannot prove its causal R3 contract."""


@dataclass(frozen=True)
class R3ClearDefinition:
    """A predeclared clear hurdle, all measured before the SL4 touch.

    ``kind='bps'`` uses ``100 + buffer_bps``.  The two ATR variants are
    deliberately specified rather than searched:

    * ``max_150bps_or_1atr``: ``max(100 + 50, 1.0 * ATR_bps)``;
    * ``150bps_plus_half_atr``: ``100 + 50 + 0.5 * ATR_bps``.
    """

    name: str
    kind: str
    buffer_bps: float = 0.0
    atr_multiplier: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"bps", "max_bps_or_atr", "bps_plus_atr"}:
            raise BaseR3RecallAblationError("unknown R3 clear definition kind")
        if not self.name or not np.isfinite(self.buffer_bps) or self.buffer_bps < 0.0:
            raise BaseR3RecallAblationError("R3 clear buffer must be finite and non-negative")
        if not np.isfinite(self.atr_multiplier) or self.atr_multiplier < 0.0:
            raise BaseR3RecallAblationError("R3 ATR multiplier must be finite and non-negative")
        if self.kind == "bps" and self.atr_multiplier != 0.0:
            raise BaseR3RecallAblationError("bps target cannot carry an ATR multiplier")

    def hurdle_bps(self, atr_bps: Sequence[float]) -> np.ndarray:
        atr = np.asarray(atr_bps, dtype=float)
        if not np.isfinite(atr).all() or (atr <= 0.0).any():
            raise BaseR3RecallAblationError("ATR bps must be finite and strictly positive")
        bps = R3_COST_BPS + float(self.buffer_bps)
        if self.kind == "bps":
            return np.full(len(atr), bps, dtype=float)
        if self.kind == "max_bps_or_atr":
            return np.maximum(bps, float(self.atr_multiplier) * atr)
        return bps + float(self.atr_multiplier) * atr


R3_CLEAR_DEFINITIONS: tuple[R3ClearDefinition, ...] = (
    R3ClearDefinition("b25_current", "bps", buffer_bps=25.0),
    R3ClearDefinition("b50", "bps", buffer_bps=50.0),
    R3ClearDefinition("b75", "bps", buffer_bps=75.0),
    R3ClearDefinition("b100", "bps", buffer_bps=100.0),
    R3ClearDefinition("b125", "bps", buffer_bps=125.0),
    R3ClearDefinition("b150", "bps", buffer_bps=150.0),
    R3ClearDefinition("max_150bps_or_1atr", "max_bps_or_atr", buffer_bps=50.0, atr_multiplier=1.0),
    R3ClearDefinition("150bps_plus_half_atr", "bps_plus_atr", buffer_bps=50.0, atr_multiplier=0.5),
)


@dataclass(frozen=True)
class WeightDefinition:
    name: str
    certainty_exponent: float = 0.0
    sqrt_class_balance: bool = False

    def __post_init__(self) -> None:
        if self.name not in {"uniform", "certainty_half", "certainty_quarter", "sqrt_class", "current_certainty_sqrt_class"}:
            raise BaseR3RecallAblationError("weight definition is not predeclared")
        if self.certainty_exponent not in {0.0, 0.5, 0.75, 1.0}:
            raise BaseR3RecallAblationError("unsupported certainty exponent")


R3_WEIGHT_DEFINITIONS: tuple[WeightDefinition, ...] = (
    WeightDefinition("uniform"),
    WeightDefinition("certainty_half", certainty_exponent=0.5),
    WeightDefinition("certainty_quarter", certainty_exponent=0.75),
    WeightDefinition("sqrt_class", sqrt_class_balance=True),
    # Current frozen-control reconstruction: consensus certainty times
    # square-root inverse class support, clipped / mean-one below.
    WeightDefinition("current_certainty_sqrt_class", certainty_exponent=1.0, sqrt_class_balance=True),
)


@dataclass(frozen=True)
class ScoreDefinition:
    name: str
    adverse_lambda: float
    kind: str = "linear"

    def __post_init__(self) -> None:
        if self.name not in {
            "p_clear", "contrast_l0p25", "contrast_l0p35", "contrast_l0p40",
            "contrast_l0p45", "contrast_l0p5", "contrast_l0p55",
            "contrast_l0p60", "contrast_l0p65", "contrast_l0p75",
            "contrast_l1p0", "contrast_l1p25", "clear_x_no_adverse",
            "clear_vs_adverse_ratio",
        }:
            raise BaseR3RecallAblationError("score definition is not predeclared")
        if self.adverse_lambda not in {0.0, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.75, 1.0, 1.25}:
            raise BaseR3RecallAblationError("score lambda is not predeclared")
        if self.kind not in {"linear", "clear_x_no_adverse", "clear_vs_adverse_ratio"}:
            raise BaseR3RecallAblationError("score kind is not predeclared")


R3_SCORE_DEFINITIONS: tuple[ScoreDefinition, ...] = (
    ScoreDefinition("p_clear", 0.0),
    ScoreDefinition("contrast_l0p25", 0.25),
    ScoreDefinition("contrast_l0p35", 0.35),
    ScoreDefinition("contrast_l0p40", 0.40),
    ScoreDefinition("contrast_l0p45", 0.45),
    ScoreDefinition("contrast_l0p5", 0.5),
    ScoreDefinition("contrast_l0p55", 0.55),
    ScoreDefinition("contrast_l0p60", 0.60),
    ScoreDefinition("contrast_l0p65", 0.65),
    ScoreDefinition("contrast_l0p75", 0.75),
    ScoreDefinition("contrast_l1p0", 1.0),
    ScoreDefinition("contrast_l1p25", 1.25),
    # These two nonlinear contrasts are deliberately the only non-linear
    # additions: they test whether adverse risk should be a probability of
    # failure (product) or a direct clear-vs-adverse competition (ratio).
    ScoreDefinition("clear_x_no_adverse", 0.0, "clear_x_no_adverse"),
    ScoreDefinition("clear_vs_adverse_ratio", 0.0, "clear_vs_adverse_ratio"),
)


@dataclass(frozen=True)
class SequentialAblationPlan:
    """The narrow funnel; no target/weight/ranker factorial is permitted."""

    phase1_scores: tuple[ScoreDefinition, ...] = R3_SCORE_DEFINITIONS
    phase2_targets: tuple[R3ClearDefinition, ...] = R3_CLEAR_DEFINITIONS
    phase3_weights: tuple[WeightDefinition, ...] = R3_WEIGHT_DEFINITIONS
    ranker_gate_metric: str = "within_query_rank_ic_and_recall30_40_no_worse_than_control"
    ranker_objective: str = "lambdarank_binary_robust_clear_grouped_by_decision_ts_side"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase1_score_only": [asdict(x) for x in self.phase1_scores],
            "phase2_target_uniform_weight": [asdict(x) for x in self.phase2_targets],
            "phase3_winner_target_weight": [asdict(x) for x in self.phase3_weights],
            "phase4_query_ranker_gate": self.ranker_gate_metric,
            "phase4_query_ranker": self.ranker_objective,
            "prohibited": "no full target_x_weight_x_score_x_ranker factorial; no net-PnL fit target",
        }


def _require_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise BaseR3RecallAblationError(f"R3 recall ablation lacks columns: {missing}")


def materialize_r3_classes(frame: pd.DataFrame, definition: R3ClearDefinition) -> pd.DataFrame:
    """Return exact R3 classes using pre-adverse path primitives only.

    A row is clear only when its stored pre-adverse MFE clears the declared
    hurdle.  As that MFE is calculated strictly before a lower touch, no
    after-adverse recovery can become clear.  Invalid/incomplete rows are not
    accepted by this function: callers must filter them at the source boundary.
    """
    _require_columns(frame, ("pre_adverse_mfe_bps", "atr_bps", "lower_touch_minute"))
    pre = pd.to_numeric(frame["pre_adverse_mfe_bps"], errors="coerce").to_numpy(float)
    atr = pd.to_numeric(frame["atr_bps"], errors="coerce").to_numpy(float)
    lower = pd.to_numeric(frame["lower_touch_minute"], errors="coerce").to_numpy(float)
    if not np.isfinite(pre).all() or not np.isfinite(atr).all() or not np.isfinite(lower).all():
        raise BaseR3RecallAblationError("target primitives must be finite valid-path fields")
    hurdle = definition.hurdle_bps(atr)
    clear = pre > hurdle
    classes = np.select([clear, lower >= 0.0], [2, 0], default=1).astype(np.int8)
    return pd.DataFrame({
        "r3_class": classes,
        "r3_clear_event": clear.astype(np.int8),
        "r3_hurdle_bps": hurdle.astype(np.float32),
        "r3_margin_bps": (pre - hurdle).astype(np.float32),
    }, index=frame.index)


def r3_consensus_certainty(frame: pd.DataFrame) -> np.ndarray:
    """Training-only agreement certainty from the frozen B0/B25/B50 contract."""
    _require_columns(frame, ("robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"))
    events = frame.loc[:, ["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(events).all() or not np.isin(events, (0.0, 1.0)).all():
        raise BaseR3RecallAblationError("frozen consensus events must be binary valid-path fields")
    # Agreement is deliberately bounded rather than a hard reject: unsettled
    # near-boundary paths retain a nonzero training contribution.
    agreement = (events.max(axis=1) == events.min(axis=1)).astype(float)
    return 0.5 + 0.5 * agreement


def build_r3_sample_weight(frame: pd.DataFrame, classes: Sequence[int], definition: WeightDefinition) -> np.ndarray:
    """Create mean-one bounded training weights; never inference inputs."""
    y = np.asarray(classes, dtype=np.int8)
    if y.ndim != 1 or not np.isin(y, R3_CLASSES).all():
        raise BaseR3RecallAblationError("weights require R3 classes 0/1/2")
    weight = np.ones(len(y), dtype=float)
    if definition.certainty_exponent:
        weight *= np.power(r3_consensus_certainty(frame), definition.certainty_exponent)
    if definition.sqrt_class_balance:
        counts = np.bincount(y, minlength=3).astype(float)
        if (counts <= 0.0).any():
            raise BaseR3RecallAblationError("class-balanced arm requires all R3 classes in its training split")
        weight *= np.sqrt(len(y) / (3.0 * counts[y]))
    weight = np.clip(weight, 0.25, 4.0)
    weight /= weight.mean()
    if not np.isfinite(weight).all() or (weight <= 0.0).any():
        raise BaseR3RecallAblationError("constructed R3 weights must be finite and positive")
    return weight


def score_r3_simplex(probability: np.ndarray, definition: ScoreDefinition) -> np.ndarray:
    """Score a direct same-side R3 probability simplex without remapping."""
    p = np.asarray(probability, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise BaseR3RecallAblationError("R3 scores require a finite adverse/weak/clear simplex")
    adverse, clear = p[:, 0], p[:, 2]
    if definition.kind == "linear":
        return clear - float(definition.adverse_lambda) * adverse
    if definition.kind == "clear_x_no_adverse":
        return clear * (1.0 - adverse)
    # The epsilon only prevents an undefined score on a degenerate simplex;
    # it is far below the model's numerical resolution and has no label input.
    return clear / np.maximum(clear + adverse, 1e-12)


def _ranked_queries(frame: pd.DataFrame, *, score_column: str, target_column: str) -> pd.DataFrame:
    _require_columns(frame, ("candidate_id", *QUERY_COLUMNS, score_column, target_column))
    local = frame.copy()
    local["decision_ts"] = pd.to_datetime(local["decision_ts"], utc=True, errors="coerce")
    if local["decision_ts"].isna().any() or local["candidate_id"].isna().any():
        raise BaseR3RecallAblationError("query identities must be valid")
    local[score_column] = pd.to_numeric(local[score_column], errors="coerce")
    local[target_column] = pd.to_numeric(local[target_column], errors="coerce")
    if not np.isfinite(local[[score_column, target_column]].to_numpy(float)).all():
        raise BaseR3RecallAblationError("scores and targets must be finite")
    if local.duplicated([*QUERY_COLUMNS, "candidate_id"]).any():
        raise BaseR3RecallAblationError("candidate must be unique within query")
    local = local.sort_values([*QUERY_COLUMNS, score_column, "candidate_id"], ascending=[True, True, False, True], kind="stable")
    local["_rank"] = local.groupby(list(QUERY_COLUMNS), observed=True).cumcount() + 1
    local["_n"] = local.groupby(list(QUERY_COLUMNS), observed=True)["candidate_id"].transform("size")
    return local


def query_r3_ranking_metrics(frame: pd.DataFrame, *, score_column: str, target_column: str = "r3_class", net_column: str | None = "net_bps") -> dict[str, float]:
    """Candidate-weighted query-local metrics used for every phase gate."""
    ranked = _ranked_queries(frame, score_column=score_column, target_column=target_column)
    target = ranked[target_column].to_numpy(float)
    score_rank = ranked.groupby(list(QUERY_COLUMNS), observed=True)[score_column].rank(method="average", ascending=True)
    target_rank = ranked.groupby(list(QUERY_COLUMNS), observed=True)[target_column].rank(method="average", ascending=True)
    # Pearson correlation of within-query ranks is a deterministic broad rank IC.
    valid = ranked.groupby(list(QUERY_COLUMNS), observed=True).filter(lambda x: len(x) > 1).index
    if len(valid) == 0:
        raise BaseR3RecallAblationError("at least one query with two candidates is required")
    ic = np.corrcoef(score_rank.loc[valid].to_numpy(float), target_rank.loc[valid].to_numpy(float))[0, 1]
    result: dict[str, float] = {"within_query_rank_ic": float(ic), "rows": float(len(ranked))}
    base_clear_rate = float((target == 2).mean())
    for fraction in (0.05, 0.30, 0.40):
        count = np.ceil(float(fraction) * ranked["_n"].to_numpy(float))
        selected = ranked["_rank"].to_numpy(float) <= count
        selected_clear = target[selected].eq(2) if isinstance(target[selected], pd.Series) else (target[selected] == 2)
        # Per-query recall: a clear in a small candidate set must not be
        # swamped by high-candidate-count timestamps.
        work = ranked.assign(_selected=selected, _clear=(target == 2).astype(np.int8))
        per_query = work.groupby(list(QUERY_COLUMNS), observed=True).agg(
            clear=("_clear", "sum"), selected_clear=("_clear", lambda x: int(x[work.loc[x.index, "_selected"]].sum())),
            selected=("_selected", "sum"),
        )
        supported = per_query.loc[per_query.clear.gt(0)]
        result[f"top{int(fraction * 100)}_winner_recall"] = float((supported.selected_clear / supported.clear).mean()) if len(supported) else np.nan
        selected_rate = float((target[selected] == 2).mean()) if selected.any() else np.nan
        result[f"top{int(fraction * 100)}_clear_uplift"] = selected_rate - base_clear_rate
        if net_column is not None and net_column in ranked.columns:
            net = pd.to_numeric(ranked[net_column], errors="coerce").to_numpy(float)
            if np.isfinite(net).all():
                result[f"top{int(fraction * 100)}_net_uplift_bps"] = float(net[selected].mean() - net.mean())
    # Equal-count global deciles preserve rank direction but never alter the
    # timestamp-local ordering used above.
    ranked["_decile"] = pd.qcut(ranked[score_column].rank(method="first"), 10, labels=False, duplicates="drop")
    curve = ranked.groupby("_decile", observed=True)[target_column].mean().to_numpy(float)
    result["target_decile_adjacent_violations"] = float((np.diff(curve) < 0.0).sum())
    return result


def ranker_may_advance(control: Mapping[str, float], candidate: Mapping[str, float], *, tolerance: float = 1e-9) -> bool:
    """Gate ranker training: no worse R3 IC and recall30/40 than control."""
    required = ("within_query_rank_ic", "top30_winner_recall", "top40_winner_recall")
    if any(key not in control or key not in candidate for key in required):
        raise BaseR3RecallAblationError("ranker gate lacks required candidate/control metrics")
    return all(float(candidate[key]) + tolerance >= float(control[key]) for key in required)


def query_group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Return LightGBM group sizes after causal deterministic query sorting."""
    _require_columns(frame, ("candidate_id", *QUERY_COLUMNS))
    local = frame.loc[:, ["candidate_id", *QUERY_COLUMNS]].copy()
    local["decision_ts"] = pd.to_datetime(local["decision_ts"], utc=True, errors="coerce")
    if local["decision_ts"].isna().any() or local.duplicated([*QUERY_COLUMNS, "candidate_id"]).any():
        raise BaseR3RecallAblationError("ranker groups require unique valid query identities")
    return local.groupby(list(QUERY_COLUMNS), observed=True, sort=True).size().to_numpy(dtype=np.int32)


__all__ = [
    "BaseR3RecallAblationError", "LABEL_COLUMNS", "QUERY_COLUMNS", "R3_CLEAR_DEFINITIONS",
    "R3_SCORE_DEFINITIONS", "R3_WEIGHT_DEFINITIONS", "R3ClearDefinition", "ScoreDefinition",
    "SequentialAblationPlan", "WeightDefinition", "build_r3_sample_weight", "materialize_r3_classes",
    "query_group_sizes", "query_r3_ranking_metrics", "r3_consensus_certainty", "ranker_may_advance",
    "score_r3_simplex",
]
