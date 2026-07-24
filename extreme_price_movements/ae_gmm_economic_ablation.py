"""Economic screening utilities for frozen AE/GMM representation ablations.

The experiment is intentionally representation-transductive: candidate states
may be fitted on outcome-free beginning/middle/end covariates from the complete
available period. Supervised base/meta fits remain chronological. Results from
this module must therefore be described as representation comparisons, not as
untouched OOS evidence for state discovery.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features_gmm_ae import AE_GMM_FEATURE_COLUMNS, load_ae_gmm_state_artifact


CURRENT_CONTEXT_ARM = "baseline_current_full_context"
BASELINE_ARM = "baseline_current_full_outputs"
NO_STATE_ARM = "baseline_no_aegmm"
CURRENT_FULL_ARM = BASELINE_ARM
TOP_FRACTIONS = (0.10, 0.20, 0.30)


@dataclass(frozen=True)
class AEGMMArm:
    arm_id: str
    mode: str
    input_features: tuple[str, ...] = ()
    state_path: str | None = None
    cluster_candidates: tuple[int, ...] = ()
    reg_covar_candidates: tuple[float, ...] = ()
    covariance_type_candidates: tuple[str, ...] = ("diag",)
    ae_max_train_rows: int = 15_000
    gmm_max_train_rows: int = 100_000
    ae_max_iter: int = 80
    seed: int = 42
    admit_all_outputs: bool = True
    notes: str = ""

    def validate(self) -> None:
        if not self.arm_id or any(ch.isspace() for ch in self.arm_id):
            raise ValueError(f"Invalid arm_id: {self.arm_id!r}")
        if self.mode not in {"none", "frozen", "fit"}:
            raise ValueError(f"Unsupported AE/GMM arm mode: {self.mode!r}")
        if self.mode == "frozen" and not self.state_path:
            raise ValueError(f"Frozen arm {self.arm_id} requires state_path")
        if self.mode == "fit" and len(self.input_features) < 2:
            raise ValueError(f"Fit arm {self.arm_id} requires at least two inputs")
        if any(value not in {"diag", "tied", "full"} for value in self.covariance_type_candidates):
            raise ValueError(
                f"Invalid covariance types for {self.arm_id}: "
                f"{self.covariance_type_candidates}"
            )

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_feature_count"] = len(self.input_features)
        payload["input_feature_hash"] = feature_list_hash(self.input_features)
        return payload


def feature_list_hash(features: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_ae_gmm_feature(name: str) -> bool:
    raw = str(name)
    candidates = {raw}
    for prefix in ("base_lgbm_", "meta_lgbm_"):
        if raw.startswith(prefix):
            candidates.add(raw.removeprefix(prefix))
    generated = set(map(str, AE_GMM_FEATURE_COLUMNS))
    return bool(candidates.intersection(generated))


def strip_ae_gmm_features(features: Sequence[str]) -> list[str]:
    """Return the shared model feature contract without current state outputs."""

    return list(
        dict.fromkeys(
            str(feature)
            for feature in features
            if str(feature).strip() and not _is_ae_gmm_feature(str(feature))
        )
    )


def model_ae_gmm_features(*, include_hard_ids: bool = False) -> list[str]:
    excluded = set()
    if not include_hard_ids:
        excluded = {"gmm_cluster_id", "cluster_t"}
    return [
        str(feature)
        for feature in AE_GMM_FEATURE_COLUMNS
        if str(feature) not in excluded
    ]


def load_feature_contract(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        else:
            values = (
                payload.get("selected_feature_union")
                or payload.get("selected_features")
                or payload.get("feature_names")
                or payload.get("columns")
                or []
            )
    else:
        frame = pd.read_csv(path)
        if "selected" in frame.columns:
            selected = frame["selected"].fillna(False).astype(bool)
            frame = frame.loc[selected]
        column = next(
            (name for name in ("feature", "feature_name", "column", "name") if name in frame.columns),
            frame.columns[0],
        )
        values = frame[column].tolist()
    output = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
    if not output:
        raise ValueError(f"No feature contract found in {path}")
    return output


def write_feature_contract(path: Path, features: Sequence[str], *, source: str) -> Path:
    values = list(dict.fromkeys(map(str, features)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "ae_gmm_economic_ablation_feature_contract_v1",
                "source": str(source),
                "selected_features": values,
                "selected_feature_count": len(values),
                "feature_contract_hash": feature_list_hash(values),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def default_arms(
    *,
    current_state_path: Path,
    components: Sequence[int] = (3, 4, 5, 6, 7, 8),
    reg_covars: Sequence[float] = (5e-4, 1e-3, 3e-3),
    covariance_types: Sequence[str] = ("diag",),
    seed: int = 42,
) -> list[AEGMMArm]:
    current = load_ae_gmm_state_artifact(current_state_path)
    inputs = tuple(map(str, current.get("feature_columns", [])))
    current_outcome_free = bool(
        current.get("representation_selection_outcome_free", False)
    )
    current_selection_note = (
        "outcome-free representation selection"
        if current_outcome_free
        else (
            "target-informed GMM configuration selection; inference transforms "
            "remain pre-entry only"
        )
    )
    arms = [
        AEGMMArm(arm_id=NO_STATE_ARM, mode="none", admit_all_outputs=False),
        AEGMMArm(
            arm_id=CURRENT_CONTEXT_ARM,
            mode="frozen",
            state_path=str(current_state_path),
            input_features=inputs,
            admit_all_outputs=False,
            notes=(
                "Exact current production selected-feature contract; "
                + current_selection_note
                + "."
            ),
        ),
        AEGMMArm(
            arm_id=CURRENT_FULL_ARM,
            mode="frozen",
            state_path=str(current_state_path),
            input_features=inputs,
            admit_all_outputs=True,
            notes=(
                "Current state with the same full-output admission as challengers; "
                + current_selection_note
                + "."
            ),
        ),
    ]
    for covariance in covariance_types:
        for component in components:
            arms.append(
                AEGMMArm(
                    arm_id=f"candidate_k{int(component)}_{covariance}",
                    mode="fit",
                    input_features=inputs,
                    cluster_candidates=(int(component),),
                    reg_covar_candidates=tuple(map(float, reg_covars)),
                    covariance_type_candidates=(str(covariance),),
                    seed=int(seed),
                )
            )
    for arm in arms:
        arm.validate()
    return arms


def load_arms(path: Path) -> list[AEGMMArm]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("arms", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("AE/GMM arm configuration must contain a list of arms")
    arms: list[AEGMMArm] = []
    for record in records:
        values = dict(record)
        for name in (
            "input_features",
            "cluster_candidates",
            "reg_covar_candidates",
            "covariance_type_candidates",
        ):
            if name in values:
                values[name] = tuple(values[name] or ())
        arm = AEGMMArm(**values)
        arm.validate()
        arms.append(arm)
    ids = [arm.arm_id for arm in arms]
    if len(ids) != len(set(ids)):
        raise ValueError("AE/GMM arm IDs must be unique")
    return arms


def split_months(months: Sequence[str], *, base_selection_months: int = 3) -> dict[str, list[str]]:
    periods = sorted(pd.Period(str(month)) for month in months)
    if len(periods) != 5:
        raise ValueError(f"Expected exactly five base OOS months, got {months}")
    expected = [periods[0] + i for i in range(5)]
    if periods != expected:
        raise ValueError(f"Base OOS months must be contiguous, got {months}")
    if int(base_selection_months) != 3:
        raise ValueError("The current meta contract requires three train and two OOS months")
    values = [str(period) for period in periods]
    return {
        "base_oos": values,
        "base_selection": values[:3],
        "meta_train": values[:3],
        "meta_oos": values[3:],
    }


def arm_model_features(
    arm: AEGMMArm,
    *,
    production_features: Sequence[str],
    core_features: Sequence[str],
) -> list[str]:
    if arm.arm_id == CURRENT_CONTEXT_ARM:
        return list(dict.fromkeys(map(str, production_features)))
    if arm.mode == "none":
        return list(dict.fromkeys(map(str, core_features)))
    if arm.admit_all_outputs:
        return list(dict.fromkeys([*map(str, core_features), *model_ae_gmm_features()]))
    return list(dict.fromkeys(map(str, production_features)))


def state_path_for_arm(arm: AEGMMArm, state_root: Path) -> Path | None:
    if arm.mode == "none":
        return None
    if arm.mode == "frozen":
        return Path(str(arm.state_path))
    return state_root / arm.arm_id / "cycle__global_state.pkl"


def _first_existing(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _numeric(frame: pd.DataFrame, name: str | None, default: float = np.nan) -> pd.Series:
    if name is None or name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce")


def _safe_autocorr(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return float("nan")
    left = clean.iloc[:-1].to_numpy(dtype=np.float64)
    right = clean.iloc[1:].to_numpy(dtype=np.float64)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _top_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    valid = np.isfinite(score)
    count = int(valid.sum())
    take = max(1, int(math.ceil(count * float(fraction)))) if count else 0
    mask = np.zeros(len(score), dtype=bool)
    if take <= 0:
        return mask
    positions = np.flatnonzero(valid)
    local = score[positions]
    chosen = np.argpartition(local, len(local) - take)[-take:]
    mask[positions[chosen]] = True
    return mask


def _metric_values(rows: pd.DataFrame, *, score_col: str) -> dict[str, Any]:
    ts_col = _first_existing(rows, ("__ts__", "timestamp"))
    ev_col = _first_existing(
        rows,
        ("ev_after_1pct", "__u_policy_net__", "__first_touch_capture_net__", "ret_net"),
    )
    soft_col = _first_existing(
        rows,
        ("__first_touch_target_soft__", "target_soft", "__first_touch_policy_soft__", "__y_lbl__"),
    )
    clean_col = _first_existing(rows, ("clean_exec", "clean_exec_label", "__first_touch_net_positive__"))
    bad_col = _first_existing(rows, ("full_path_bad_mae_1r", "__path_full_bad_mae_1r__"))
    timeout_col = _first_existing(rows, ("timeout", "__first_touch_timeout__", "__is_timeout__"))
    stop_col = _first_existing(rows, ("stop_or_adverse", "__first_touch_stop__", "full_stop_loss"))
    score = _numeric(rows, score_col)
    ev = _numeric(rows, ev_col)
    soft = _numeric(rows, soft_col)
    clean = _numeric(rows, clean_col)
    bad = _numeric(rows, bad_col)
    timeout = _numeric(rows, timeout_col)
    stop = _numeric(rows, stop_col)
    result: dict[str, Any] = {
        "selected_rows": int(len(rows)),
        "mean_ev_after_1pct": float(ev.mean()),
        "sum_ev_after_1pct": float(ev.sum(min_count=1)),
        "positive_ev_rate": float(ev.gt(0).mean()),
        "clean_exec_precision": float(clean.mean()),
        "bad_mae_rate": float(bad.mean()),
        "timeout_rate": float(timeout.mean()),
        "stop_or_adverse_rate": float(stop.mean()),
    }
    if ts_col is not None:
        ts = pd.to_datetime(rows[ts_col], utc=True, errors="coerce")
        days = max(int(ts.dt.floor("D").nunique()), 1)
        result["trades_per_day"] = float(len(rows) / days)
        daily = pd.DataFrame(
            {
                "day": ts.dt.floor("D"),
                "residual": soft.to_numpy(dtype=np.float64) - score.clip(0, 1).to_numpy(dtype=np.float64),
                "hit_surprise": clean.to_numpy(dtype=np.float64) - score.clip(0, 1).to_numpy(dtype=np.float64),
                "ev": ev.to_numpy(dtype=np.float64),
            }
        ).dropna(subset=["day"])
        daily_rows = daily
        daily = daily_rows.groupby("day", sort=True, observed=True).mean(numeric_only=True)
        week_start = daily_rows["day"] - pd.to_timedelta(
            daily_rows["day"].dt.weekday, unit="D"
        )
        weekly_ev = daily_rows.assign(week_start=week_start).groupby(
            "week_start", sort=True, observed=True
        )["ev"].mean()
        monthly_ev = daily_rows.assign(
            month=daily_rows["day"].dt.tz_localize(None).dt.to_period("M").astype(str)
        ).groupby("month", sort=True, observed=True)["ev"].mean()
        result.update(
            {
                "signed_residual_mean": float(daily["residual"].mean()),
                "signed_residual_autocorr": _safe_autocorr(daily["residual"]),
                "signed_hit_surprise_mean": float(daily["hit_surprise"].mean()),
                "signed_hit_surprise_autocorr": _safe_autocorr(daily["hit_surprise"]),
                "positive_hit_surprise_autocorr": _safe_autocorr(daily["hit_surprise"].clip(lower=0)),
                "negative_hit_surprise_autocorr": _safe_autocorr((-daily["hit_surprise"]).clip(lower=0)),
                "worst_day_ev": float(daily["ev"].min()),
                "worst_week_ev": float(weekly_ev.min()),
                "worst_month_ev": float(monthly_ev.min()),
            }
        )
    return result


def economic_metrics(
    frame: pd.DataFrame,
    *,
    arm: str,
    score_col: str = "score",
    months: Sequence[str] | None = None,
    selection_basis: str = "global",
) -> pd.DataFrame:
    if selection_basis not in {"global", "per_side"}:
        raise ValueError("selection_basis must be 'global' or 'per_side'")
    work = frame.copy(deep=False)
    ts_col = _first_existing(work, ("__ts__", "timestamp"))
    if ts_col is None:
        raise ValueError("Economic ledger has no timestamp column")
    ts = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    ts_naive = ts.dt.tz_localize(None)
    work = work.assign(
        __metric_month__=ts_naive.dt.to_period("M").astype(str),
        __metric_week__=(ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")).astype(str),
    )
    if months:
        work = work.loc[work["__metric_month__"].isin(map(str, months))]
    side_col = _first_existing(work, ("side_name", "side"))
    archetype_col = _first_existing(
        work,
        (
            "archetype_label_family",
            "__archetype_label_family__",
            "policy_archetype",
            "__archetype_policy_key__",
            # The fixed-base meta ablation preserves the canonical handoff
            # archetype under this public name. Keep its report contract
            # identical to the other archetype sources.
            "archetype_policy_key",
        ),
    )
    score = _numeric(work, score_col).to_numpy(dtype=np.float64)
    if selection_basis == "per_side" and side_col is None:
        raise ValueError("per_side selection requires side_name or side")

    def select_top_fraction(fraction: float) -> np.ndarray:
        if selection_basis == "global":
            return _top_mask(score, fraction)
        # Side scores are calibrated against different label base rates. Rank
        # them locally so a global cutoff cannot suppress the lower-base-rate
        # side merely because its probability scale is lower.
        side_codes, _ = pd.factorize(work[side_col], sort=False)  # type: ignore[index]
        selected = np.zeros(len(work), dtype=bool)
        for code in np.unique(side_codes[side_codes >= 0]):
            positions = np.flatnonzero(side_codes == code)
            selected[positions] = _top_mask(score[positions], fraction)
        return selected

    output: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("month", ["__metric_month__"]),
        ("week", ["__metric_week__"]),
    ]
    if side_col:
        group_specs.append(("side", [side_col]))
    if archetype_col:
        group_specs.append(("archetype", [archetype_col]))
    if side_col and archetype_col:
        group_specs.append(("side_archetype", [side_col, archetype_col]))
    for fraction in TOP_FRACTIONS:
        selected = work.loc[select_top_fraction(fraction)].copy(deep=False)
        for scope, columns in group_specs:
            groups = [((), selected)] if not columns else selected.groupby(columns, observed=True, sort=True, dropna=False)
            for key, rows in groups:
                keys = key if isinstance(key, tuple) else (key,)
                record: dict[str, Any] = {
                    "arm": str(arm),
                    "score_col": str(score_col),
                    "top_frac": float(fraction),
                    "scope": scope,
                    "selection_basis": f"{selection_basis}_topk",
                }
                for column, value in zip(columns, keys):
                    public = {
                        "__metric_month__": "month",
                        "__metric_week__": "week_start",
                        side_col: "side_name",
                        archetype_col: "archetype_label_family",
                    }.get(column, column)
                    record[public] = value
                record.update(_metric_values(rows, score_col=score_col))
                output.append(record)
    return pd.DataFrame(output)


def add_baseline_deltas(metrics: pd.DataFrame, *, baseline_arm: str = BASELINE_ARM) -> pd.DataFrame:
    dimensions = [
        name
        for name in (
            "score_col", "top_frac", "scope", "selection_basis", "month", "week_start",
            "side_name", "archetype_label_family",
        )
        if name in metrics.columns
    ]
    metric_columns = [
        name
        for name in metrics.columns
        if name not in {"arm", *dimensions} and pd.api.types.is_numeric_dtype(metrics[name])
    ]
    baseline = metrics.loc[metrics["arm"].eq(baseline_arm), dimensions + metric_columns].copy()
    baseline = baseline.rename(columns={name: f"baseline_{name}" for name in metric_columns})
    joined = metrics.merge(baseline, on=dimensions, how="left")
    for name in metric_columns:
        joined[f"delta_vs_{baseline_arm}__{name}"] = pd.to_numeric(joined[name], errors="coerce") - pd.to_numeric(
            joined[f"baseline_{name}"], errors="coerce"
        )
    return joined


def base_selection_ranking(metrics: pd.DataFrame, *, months: Sequence[str]) -> pd.DataFrame:
    rows = metrics.loc[
        metrics["scope"].eq("overall")
        & metrics["top_frac"].isin(TOP_FRACTIONS)
    ].copy()
    # Overall rows are already restricted to the requested selection months by
    # the caller. The weighting prioritizes the tails that are actually traded.
    pivot = rows.pivot_table(index="arm", columns="top_frac", values="mean_ev_after_1pct", aggfunc="first")
    for fraction in TOP_FRACTIONS:
        if fraction not in pivot:
            pivot[fraction] = np.nan
    pivot["base_representation_score"] = (
        0.50 * pivot[0.10] + 0.30 * pivot[0.20] + 0.20 * pivot[0.30]
    )
    pivot["selection_months"] = ",".join(map(str, months))
    return pivot.reset_index().sort_values("base_representation_score", ascending=False)
