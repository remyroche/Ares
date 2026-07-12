"""Leakage-safe regime feature builders for train_meta ablation studies.

The builders in this module are deliberately fold-local.  They are fit on a
train slice, emit features for that train slice, and assign OOS rows with
frozen scalers/clusterers/classifiers.  Outcome/error columns may be used to
define train-side clusters or priors, but they are stripped before OOS
assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

OUTCOME_COLUMNS = {
    "__first_touch_target_soft__",
    "__first_touch_policy_soft__",
    "target_soft",
    "__target_soft__",
    "target_hard",
    "__target_hard__",
    "exec_margin",
    "ev_after_1pct",
    "ret_net",
    "u_policy_net",
    "first_touch_gross",
    "first_touch_net",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "clean_exec",
    "clean_exec_label",
    "dirty_positive",
    "bad_path_label",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "mae_norm",
    "mfe_norm",
    "first_touch_full_path_mae_norm",
    "underwater_bars_before_mfe_1r",
    "long_path_full_bad_mae_1r",
    "long_path_time_to_profit_bars",
    "long_path_slow_profit",
    "long_path_post_mfe_drawdown_norm",
    "long_path_post_mfe_bad_drawdown",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_quality_soft",
    "long_bad_path_label",
}

NEVER_FEATURE_COLUMNS = {"__ts__", "__symbol__", "month"}
PHASE_STATE_PREFIX = "state_phase__"
PHASE_CONTEXT_PREFIX = "ctx_phase__"

CROSS_ASSET_CONTEXT_FEATURES = {
    "cs_rank_oi_value_z_30d",
    "eth_ret_24h",
    "q_tail_width__ob_spread_z_x_rv_24h",
    "state_spectral_abs_pc3_z",
    "state_spectral_eig_lambda1_share",
    "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
}

CURRENT_ARCHETYPE_REGIME_COLUMNS = (
    "source_semantic_family",
    "source_semantic_family_base",
    "long_source_regime_split",
    "aegmm_cluster",
    "side_aegmm_cluster",
    "aegmm_entropy_bin",
    "aegmm_distance_bin",
    "aegmm_expected_distance_bin",
    "reconstruction_bin",
    "dae_reconstruction_bin",
    "latent_speed_bin",
    "regime_lgbm_leaf_bad_mae_k4",
    "regime_lgbm_leaf_exec_margin_k4",
    "regime_first_touch_bad_mae_score_bin",
    "regime_timeout_score_bin",
    "regime_dirty_positive_score_bin",
    "regime_clean_exec_score_bin",
)


def drop_oos_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an OOS-safe view with realized outcome/target columns removed."""

    return frame.drop(
        columns=[col for col in OUTCOME_COLUMNS if col in frame.columns],
        errors="ignore",
    )


@dataclass
class FrozenPhaseStateContext:
    """Join deterministic, point-in-time market phase coordinates to meta rows.

    The source state table can retain train-only targets for discovery work, but
    this adapter reads *only* ``state_phase__*`` fields.  It is deliberately not
    a learned fold transform: the phase fields are fixed current/past OHLCV/OI
    combinations, and every candidate gets the most recent state at or before
    its decision timestamp.  This mirrors the live feature contract without
    exposing realized outcomes to the meta model.
    """

    source_path: Path
    timestamp_col: str = "__ts__"
    side_col: str = "side_name"
    max_lag_minutes: int = 60
    _states: pd.DataFrame = field(init=False, repr=False)
    _source_features: list[str] = field(init=False, default_factory=list)
    _feature_names: list[str] = field(init=False, default_factory=list)
    _source_start: pd.Timestamp | None = field(init=False, default=None)
    _source_end: pd.Timestamp | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        path = Path(self.source_path)
        if not path.exists():
            raise FileNotFoundError(f"Frozen phase-state source not found: {path}")
        try:
            import pyarrow.parquet as pq

            schema_names = list(pq.read_schema(path).names)
        except Exception:
            schema_names = list(pd.read_parquet(path).columns)
        required = [self.timestamp_col, self.side_col]
        missing = [name for name in required if name not in schema_names]
        if missing:
            raise ValueError(
                f"Frozen phase-state source missing required columns: {missing}"
            )
        self._source_features = sorted(
            name for name in schema_names if str(name).startswith(PHASE_STATE_PREFIX)
        )
        if not self._source_features:
            raise ValueError(
                f"Frozen phase-state source {path} has no {PHASE_STATE_PREFIX!r} features."
            )
        # The output namespace intentionally makes the context role explicit
        # and prevents colliding with state columns already present in a handoff.
        self._feature_names = [
            f"{PHASE_CONTEXT_PREFIX}{name.removeprefix(PHASE_STATE_PREFIX)}"
            for name in self._source_features
        ] + ["ctx_phase_available", "ctx_phase_age_minutes"]
        states = pd.read_parquet(path, columns=[*required, *self._source_features])
        states = states.copy(deep=False)
        states[self.timestamp_col] = pd.to_datetime(
            states[self.timestamp_col], utc=True, errors="coerce"
        )
        states["__phase_side__"] = states[self.side_col].astype(str).str.lower()
        states = states.loc[
            states[self.timestamp_col].notna() & states["__phase_side__"].ne("")
        ].copy()
        if states.duplicated([self.timestamp_col, "__phase_side__"]).any():
            duplicate_count = int(
                states.duplicated([self.timestamp_col, "__phase_side__"]).sum()
            )
            raise ValueError(
                "Frozen phase-state source must be unique per timestamp and side; "
                f"found {duplicate_count} duplicate rows."
            )
        states["__phase_state_ts__"] = states[self.timestamp_col]
        rename = {
            source: destination
            for source, destination in zip(
                self._source_features, self._feature_names, strict=False
            )
        }
        states = states.rename(columns=rename)
        for name in self._feature_names[:-2]:
            states[name] = pd.to_numeric(states[name], errors="coerce").astype(
                np.float32
            )
        self._states = (
            states[
                [
                    self.timestamp_col,
                    "__phase_side__",
                    "__phase_state_ts__",
                    *self._feature_names[:-2],
                ]
            ]
            .sort_values([self.timestamp_col, "__phase_side__"], kind="stable")
            .reset_index(drop=True)
        )
        self._source_start = self._states[self.timestamp_col].min()
        self._source_end = self._states[self.timestamp_col].max()

    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _join(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [
            name
            for name in (self.timestamp_col, self.side_col)
            if name not in frame.columns
        ]
        if missing:
            raise ValueError(f"Meta frame missing frozen phase join keys: {missing}")
        if frame.empty:
            return pd.DataFrame(
                index=frame.index, columns=self._feature_names, dtype=np.float32
            )
        left = pd.DataFrame(
            {
                self.timestamp_col: pd.to_datetime(
                    frame[self.timestamp_col], utc=True, errors="coerce"
                ),
                "__phase_side__": frame[self.side_col].astype(str).str.lower(),
                "__phase_row_order__": np.arange(len(frame), dtype=np.int64),
            },
            index=frame.index,
        )
        valid = left[self.timestamp_col].notna() & left["__phase_side__"].ne("")
        result = pd.DataFrame(index=frame.index)
        for name in self._feature_names[:-2]:
            result[name] = np.float32(0.0)
        result["ctx_phase_available"] = np.float32(0.0)
        result["ctx_phase_age_minutes"] = np.float32(float(self.max_lag_minutes))
        if not bool(valid.any()):
            return result.astype(np.float32)
        left_valid = left.loc[valid].sort_values(
            [self.timestamp_col, "__phase_side__"], kind="stable"
        )
        merged = pd.merge_asof(
            left_valid,
            self._states,
            on=self.timestamp_col,
            by="__phase_side__",
            direction="backward",
            allow_exact_matches=True,
            tolerance=pd.Timedelta(minutes=int(self.max_lag_minutes)),
        ).sort_values("__phase_row_order__", kind="stable")
        target_index = frame.index.take(
            merged["__phase_row_order__"].to_numpy(dtype=np.int64)
        )
        for name in self._feature_names[:-2]:
            result.loc[target_index, name] = (
                pd.to_numeric(merged[name], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        state_ts = pd.to_datetime(
            merged["__phase_state_ts__"], utc=True, errors="coerce"
        )
        age = (
            (
                pd.to_datetime(merged[self.timestamp_col], utc=True, errors="coerce")
                - state_ts
            )
            .dt.total_seconds()
            .div(60.0)
        )
        available = state_ts.notna()
        result.loc[target_index, "ctx_phase_available"] = available.astype(
            np.float32
        ).to_numpy()
        result.loc[target_index, "ctx_phase_age_minutes"] = (
            age.fillna(float(self.max_lag_minutes))
            .clip(lower=0.0, upper=float(self.max_lag_minutes))
            .astype(np.float32)
            .to_numpy()
        )
        return result.astype(np.float32)

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._join(train)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._join(oos_without_outcomes)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": "causal_phase_state_context",
            "source_path": str(Path(self.source_path)),
            "source_feature_columns": list(self._source_features),
            "output_feature_columns": self.feature_names(),
            "source_start": self._source_start,
            "source_end": self._source_end,
            "join": {
                "timestamp_col": self.timestamp_col,
                "side_col": self.side_col,
                "direction": "backward",
                "allow_exact_matches": True,
                "max_lag_minutes": int(self.max_lag_minutes),
            },
            "leakage_contract": (
                "Only deterministic state_phase__ columns are read from the state table. "
                "target_ / outcome columns are never loaded or emitted; OOS rows use only "
                "the latest point-in-time phase state at or before the decision timestamp."
            ),
        }


def archetype_series(frame: pd.DataFrame) -> pd.Series:
    for col in (
        "archetype_label_family",
        "__archetype_label_family__",
        "policy_archetype",
        "archetype_policy_key",
        "__archetype_policy_key__",
        "local_side_archetype",
        "source_archetype",
        "source_semantic_family",
    ):
        if col in frame.columns:
            ser = frame[col].astype(str).replace({"nan": "", "None": ""})
            if bool(ser.str.len().gt(0).any()):
                return ser.where(ser.str.len().gt(0), "missing")
    return pd.Series("missing", index=frame.index, dtype="object")


def _safe_feature_token(value: object) -> str:
    text = str(value).strip().lower()
    token = "".join(char if char.isalnum() else "_" for char in text)
    return token.strip("_") or "missing"


@dataclass
class SideArchetypeIdentityContext:
    """Frozen side x base-archetype identity for a global meta model.

    This context is a pre-entry categorical descriptor, not an outcome prior.
    It gives a global LGBM enough capacity to learn a different response to the
    same observable phase state in, for example, long mixed and short mixed.
    """

    keys: tuple[str, ...]
    side_col: str = "side_name"
    _feature_names: list[str] = field(init=False, default_factory=list)
    _key_to_column: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(dict.fromkeys(str(key) for key in self.keys if str(key).strip()))
        )
        object.__setattr__(self, "keys", normalized)
        mapping = {
            key: f"ctx_identity__side_archetype__{_safe_feature_token(key)}"
            for key in normalized
        }
        object.__setattr__(self, "_key_to_column", mapping)
        object.__setattr__(self, "_feature_names", list(mapping.values()))

    @classmethod
    def from_parquet(
        cls, path: Path, *, side_col: str = "side_name"
    ) -> "SideArchetypeIdentityContext":
        path = Path(path)
        try:
            import pyarrow.parquet as pq

            schema_names = list(pq.read_schema(path).names)
        except Exception:
            schema_names = list(pd.read_parquet(path).columns)
        if side_col not in schema_names:
            raise ValueError(f"Identity source missing {side_col!r}: {path}")
        archetype_candidates = (
            "archetype_label_family",
            "__archetype_label_family__",
            "policy_archetype",
            "archetype_policy_key",
            "__archetype_policy_key__",
            "local_side_archetype",
            "source_archetype",
            "source_semantic_family",
        )
        source_arch_cols = [
            name for name in archetype_candidates if name in schema_names
        ]
        if not source_arch_cols:
            raise ValueError(
                f"Identity source has no archetype descriptor column: {path}"
            )
        source = pd.read_parquet(path, columns=[side_col, *source_arch_cols])
        side = (
            source[side_col]
            .astype(str)
            .str.lower()
            .replace({"nan": "missing", "None": "missing"})
        )
        arch = archetype_series(source).astype(str).str.lower()
        keys = tuple((side + "__" + arch).dropna().unique().tolist())
        return cls(keys=keys, side_col=side_col)

    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.side_col not in frame.columns:
            raise ValueError(
                f"Meta frame missing side column for identity context: {self.side_col}"
            )
        side = (
            frame[self.side_col]
            .astype(str)
            .str.lower()
            .replace({"nan": "missing", "None": "missing"})
        )
        arch = archetype_series(frame).astype(str).str.lower()
        row_keys = (side + "__" + arch).to_numpy(dtype=object)
        output = np.zeros((len(frame), len(self._feature_names)), dtype=np.float32)
        positions = {key: idx for idx, key in enumerate(self.keys)}
        codes = (
            pd.Series(row_keys, index=frame.index)
            .map(positions)
            .fillna(-1)
            .to_numpy(dtype=np.int16)
        )
        valid = codes >= 0
        if bool(valid.any()):
            output[np.flatnonzero(valid), codes[valid]] = np.float32(1.0)
        return pd.DataFrame(output, index=frame.index, columns=self._feature_names)

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._transform(train)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._transform(oos_without_outcomes)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": "side_archetype_identity_context",
            "keys": list(self.keys),
            "feature_columns": self.feature_names(),
            "leakage_contract": (
                "Side and base-archetype identity are observable pre-entry descriptors. "
                "No realized outcome, historical performance, or target-derived prior is used."
            ),
        }


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = _num(frame, "clean_exec_label", np.nan)
    if clean.isna().all():
        clean = _num(frame, "clean_exec", 0.0)
    bad = _num(frame, "full_path_bad_mae_1r", 0.0)
    dirty = _num(frame, "dirty_positive", 0.0)
    timeout = _num(frame, "timeout", 0.0)
    exec_margin = _num(frame, "exec_margin", 0.0)
    return pd.DataFrame(
        {
            "clean": clean.fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "bad_mae": bad.fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "dirty": dirty.fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "timeout": timeout.fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "exec": exec_margin.fillna(0.0).astype(np.float32),
        },
        index=frame.index,
    )


def _candidate_pre_entry_columns(
    frame: pd.DataFrame, *, temporal_only: bool = False
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    temporal_markers = (
        "hit_surprise",
        "hit_recent",
        "hit_expected",
        "support_",
        "drift",
        "leaf",
        "aegmm",
        "reconstruction",
        "regime_",
        "confidence",
        "precision",
    )
    for col in frame.columns:
        name = str(col)
        if (
            name in OUTCOME_COLUMNS
            or name in NEVER_FEATURE_COLUMNS
            or name.startswith("selected_top")
        ):
            continue
        if name.endswith("__ledger") or name.startswith("ab_"):
            continue
        if temporal_only and not any(marker in name for marker in temporal_markers):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or frame[col].dtype == bool:
            numeric.append(name)
        elif frame[col].nunique(dropna=True) <= 128:
            categorical.append(name)
    return sorted(set(numeric)), sorted(set(categorical))


class MatrixEncoder:
    def __init__(
        self,
        *,
        max_categories_per_col: int = 32,
        max_numeric_cols: int = 240,
        max_categorical_cols: int = 48,
        max_profile_rows: int = 120_000,
    ) -> None:
        self.max_categories_per_col = int(max_categories_per_col)
        self.max_numeric_cols = int(max_numeric_cols)
        self.max_categorical_cols = int(max_categorical_cols)
        self.max_profile_rows = int(max_profile_rows)
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.categories_: dict[str, list[str]] = {}
        self.medians_: pd.Series = pd.Series(dtype=np.float32)
        self.columns_: list[str] = []

    def _profile_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= self.max_profile_rows:
            return frame
        idx = np.unique(
            np.concatenate(
                [
                    np.linspace(
                        0,
                        len(frame) // 3 - 1,
                        num=max(1, self.max_profile_rows // 3),
                        dtype=np.int64,
                    ),
                    np.linspace(
                        len(frame) // 3,
                        (2 * len(frame)) // 3 - 1,
                        num=max(1, self.max_profile_rows // 3),
                        dtype=np.int64,
                    ),
                    np.linspace(
                        (2 * len(frame)) // 3,
                        len(frame) - 1,
                        num=max(1, self.max_profile_rows // 3),
                        dtype=np.int64,
                    ),
                ]
            )
        )
        return frame.iloc[idx]

    def _select_numeric_cols(
        self, frame: pd.DataFrame, numeric_cols: list[str]
    ) -> list[str]:
        cols = [col for col in numeric_cols if col in frame.columns]
        if len(cols) <= self.max_numeric_cols:
            return cols
        sample = self._profile_frame(frame)
        nums = (
            sample.loc[:, cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        coverage = nums.notna().mean(axis=0).astype(float)
        std = (
            nums.std(axis=0, numeric_only=True)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        score = coverage * np.log1p(std.clip(lower=0.0))
        ranked = score.sort_values(ascending=False)
        return [str(col) for col in ranked.head(self.max_numeric_cols).index]

    def _select_categorical_cols(
        self, frame: pd.DataFrame, categorical_cols: list[str]
    ) -> list[str]:
        cols = [col for col in categorical_cols if col in frame.columns]
        if len(cols) <= self.max_categorical_cols:
            return cols
        sample = self._profile_frame(frame)
        scores: list[tuple[float, str]] = []
        for col in cols:
            vals = sample[col].astype(str).replace({"nan": ""})
            coverage = float(vals.str.len().gt(0).mean())
            nunique = int(vals.nunique(dropna=True))
            scores.append((coverage * math.log1p(max(nunique, 1)), col))
        scores.sort(reverse=True)
        return [col for _, col in scores[: self.max_categorical_cols]]

    def fit(
        self, frame: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
    ) -> "MatrixEncoder":
        self.numeric_cols = self._select_numeric_cols(frame, numeric_cols)
        self.categorical_cols = self._select_categorical_cols(frame, categorical_cols)
        if self.numeric_cols:
            nums = (
                frame.loc[:, self.numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            self.medians_ = (
                nums.median(numeric_only=True)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            self.medians_ = pd.Series(dtype=np.float32)
        self.categories_ = {}
        for col in self.categorical_cols:
            counts = frame[col].astype(str).fillna("missing").value_counts(dropna=False)
            self.categories_[col] = list(
                counts.head(self.max_categories_per_col).index.astype(str)
            )
        self.columns_ = list(self.transform(frame).columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if self.numeric_cols:
            nums = (
                frame.reindex(columns=self.numeric_cols)
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            parts.append(nums.fillna(self.medians_).fillna(0.0).astype(np.float32))
        cat_parts: list[pd.Series] = []
        for col in self.categorical_cols:
            vals = (
                frame[col].astype(str).fillna("missing")
                if col in frame.columns
                else pd.Series("missing", index=frame.index)
            )
            allowed = set(self.categories_.get(col, []))
            vals = vals.where(vals.isin(allowed), "__other__")
            for cat in self.categories_.get(col, []):
                cat_parts.append(vals.eq(cat).astype(np.float32).rename(f"{col}={cat}"))
            cat_parts.append(
                vals.eq("__other__").astype(np.float32).rename(f"{col}=__other__")
            )
        if cat_parts:
            parts.append(pd.concat(cat_parts, axis=1))
        if not parts:
            return pd.DataFrame(index=frame.index)
        out = pd.concat(parts, axis=1)
        if self.columns_:
            out = out.reindex(columns=self.columns_, fill_value=0.0)
        return out.astype(np.float32)


def _time_spread_indices(n_rows: int, max_rows: int) -> np.ndarray:
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    per_segment = max(1, max_rows // 3)
    bounds = (
        (0, n_rows // 3),
        (n_rows // 3, (2 * n_rows) // 3),
        ((2 * n_rows) // 3, n_rows),
    )
    parts: list[np.ndarray] = []
    for start, stop in bounds:
        width = max(0, stop - start)
        if width == 0:
            continue
        take = min(per_segment, width)
        parts.append(np.linspace(start, stop - 1, num=take, dtype=np.int64))
    if not parts:
        return np.arange(min(n_rows, max_rows), dtype=np.int64)
    idx = np.unique(np.concatenate(parts))
    if len(idx) > max_rows:
        idx = idx[:max_rows]
    return idx.astype(np.int64, copy=False)


def _fit_gmm(
    x: np.ndarray, *, n_clusters: int, seed: int, max_fit_rows: int = 120_000
) -> GaussianMixture | None:
    if x.shape[0] < max(30, n_clusters * 10) or x.shape[1] == 0:
        return None
    x64 = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x64).all():
        x64 = np.nan_to_num(x64, nan=0.0, posinf=0.0, neginf=0.0)
    fit_idx = _time_spread_indices(len(x64), int(max_fit_rows))
    x_fit = x64[fit_idx]
    if np.unique(x_fit[: min(len(x_fit), 10_000)], axis=0).shape[0] < 2:
        return None
    k = int(min(max(2, n_clusters), max(2, x_fit.shape[0] // 20)))
    last_error: Exception | None = None
    for k_try in range(k, 1, -1):
        for reg_covar in (1e-4, 1e-3, 1e-2, 5e-2):
            for init_params in ("kmeans", "random"):
                try:
                    return GaussianMixture(
                        n_components=k_try,
                        covariance_type="diag",
                        reg_covar=reg_covar,
                        random_state=int(seed),
                        max_iter=100,
                        n_init=2,
                        init_params=init_params,
                    ).fit(x_fit)
                except ValueError as exc:
                    last_error = exc
                    continue
    if last_error is not None:
        return None
    return None


def _entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-8, 1.0)
    return (
        -(p * np.log(p)).sum(axis=1) / max(math.log(max(p.shape[1], 2)), 1e-8)
    ).astype(np.float32)


def _prior_features(
    *,
    prefix: str,
    train_clusters: np.ndarray,
    target: pd.DataFrame,
    out_clusters: np.ndarray,
    train_mode: bool,
    shrinkage_k: float = 50.0,
) -> pd.DataFrame:
    global_vals = target.mean(numeric_only=True).to_dict()
    work = target.reset_index(drop=True).copy()
    work["_cluster"] = np.asarray(train_clusters, dtype=np.int32)
    grouped = work.groupby("_cluster", dropna=False)
    counts = grouped.size().astype(float)
    sums = grouped[["clean", "bad_mae", "dirty", "timeout", "exec"]].sum()
    out = pd.DataFrame(index=np.arange(len(out_clusters)))
    for name in ["clean", "bad_mae", "dirty", "timeout", "exec"]:
        raw_sum = (
            pd.Series(np.asarray(out_clusters, dtype=np.int32))
            .map(sums[name])
            .astype(float)
            .fillna(0.0)
        )
        raw_count = (
            pd.Series(np.asarray(out_clusters, dtype=np.int32))
            .map(counts)
            .astype(float)
            .fillna(0.0)
        )
        if train_mode and len(out_clusters) == len(train_clusters):
            same_cluster = (
                np.asarray(out_clusters, dtype=np.int32)
                == np.asarray(train_clusters, dtype=np.int32)
            ).astype(float)
            raw_sum = (
                raw_sum
                - target[name].reset_index(drop=True).astype(float) * same_cluster
            )
            raw_count = (raw_count - same_cluster).clip(lower=0.0)
        prior = (raw_sum + float(shrinkage_k) * float(global_vals.get(name, 0.0))) / (
            raw_count + float(shrinkage_k)
        )
        out[f"{prefix}_prior_{name}"] = prior.astype(np.float32)
    support = (
        pd.Series(np.asarray(out_clusters, dtype=np.int32))
        .map(counts)
        .astype(float)
        .fillna(0.0)
    )
    out[f"{prefix}_support_log1p"] = np.log1p(support).astype(np.float32)
    return out


def _gmm_feature_frame(
    *,
    prefix: str,
    gmm: GaussianMixture | None,
    x: np.ndarray,
    max_clusters: int,
    index: pd.Index,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = pd.DataFrame(index=index)
    if gmm is None:
        proba = np.zeros((len(index), max_clusters), dtype=np.float32)
        clusters = np.zeros(len(index), dtype=np.int32)
        out[f"{prefix}_cluster_id"] = clusters.astype(np.float32)
        for i in range(max_clusters):
            out[f"{prefix}_posterior_{i}"] = proba[:, i]
        out[f"{prefix}_entropy"] = np.float32(1.0)
        out[f"{prefix}_distance"] = np.float32(0.0)
        return out, clusters
    raw = gmm.predict_proba(x).astype(np.float32)
    clusters = np.asarray(np.argmax(raw, axis=1), dtype=np.int32)
    out[f"{prefix}_cluster_id"] = clusters.astype(np.float32)
    for i in range(max_clusters):
        out[f"{prefix}_posterior_{i}"] = raw[:, i] if i < raw.shape[1] else 0.0
    out[f"{prefix}_entropy"] = _entropy(raw)
    out[f"{prefix}_distance"] = (-gmm.score_samples(x)).astype(np.float32)
    return out, clusters


@dataclass
class RegimeFeatureBuilder:
    name: str
    prefix: str
    n_clusters: int = 5
    seed: int = 52
    min_rows: int = 250
    metadata_: dict[str, Any] = field(default_factory=dict)

    def feature_names(self) -> list[str]:
        names = [f"{self.prefix}_cluster_id"]
        names += [f"{self.prefix}_posterior_{i}" for i in range(int(self.n_clusters))]
        names += [f"{self.prefix}_entropy", f"{self.prefix}_distance"]
        names += [
            f"{self.prefix}_prior_{name}"
            for name in ("clean", "bad_mae", "dirty", "timeout", "exec")
        ]
        names += [f"{self.prefix}_support_log1p"]
        return names

    def fit(self, train: pd.DataFrame) -> "RegimeFeatureBuilder":
        return self

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return train.copy()

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        return oos_without_outcomes.copy()

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "prefix": self.prefix, **self.metadata_}


class GMMRegimeBuilder(RegimeFeatureBuilder):
    def __init__(
        self,
        *,
        name: str,
        prefix: str,
        n_clusters: int = 5,
        seed: int = 52,
        temporal_only: bool = False,
    ) -> None:
        super().__init__(name=name, prefix=prefix, n_clusters=n_clusters, seed=seed)
        self.temporal_only = bool(temporal_only)
        self.encoder = MatrixEncoder()
        self.scaler = StandardScaler()
        self.gmm: GaussianMixture | None = None
        self.train_clusters: np.ndarray = np.zeros(0, dtype=np.int32)
        self.target: pd.DataFrame = pd.DataFrame()

    def fit(self, train: pd.DataFrame) -> "GMMRegimeBuilder":
        num, cat = _candidate_pre_entry_columns(train, temporal_only=self.temporal_only)
        self.encoder.fit(train, num, cat)
        x = self.encoder.transform(train)
        if x.shape[1] == 0:
            self.metadata_ = {"status": "no_features", "feature_count": 0}
            return self
        xs = self.scaler.fit_transform(x)
        self.gmm = _fit_gmm(xs, n_clusters=self.n_clusters, seed=self.seed)
        _, self.train_clusters = _gmm_feature_frame(
            prefix=self.prefix,
            gmm=self.gmm,
            x=xs,
            max_clusters=self.n_clusters,
            index=train.index,
        )
        self.target = _target_frame(train).reset_index(drop=True)
        self.metadata_ = {
            "status": "ok" if self.gmm is not None else "fallback",
            "feature_count": int(x.shape[1]),
        }
        return self

    def _transform(self, frame: pd.DataFrame, *, train_mode: bool) -> pd.DataFrame:
        x = self.encoder.transform(frame)
        xs = (
            self.scaler.transform(x)
            if x.shape[1]
            else np.zeros((len(frame), 0), dtype=np.float32)
        )
        features, clusters = _gmm_feature_frame(
            prefix=self.prefix,
            gmm=self.gmm,
            x=xs,
            max_clusters=self.n_clusters,
            index=frame.index,
        )
        priors = _prior_features(
            prefix=self.prefix,
            train_clusters=self.train_clusters,
            target=self.target,
            out_clusters=clusters,
            train_mode=train_mode,
        )
        priors.index = frame.index
        return pd.concat([features, priors], axis=1)

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._transform(train, train_mode=True)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._transform(oos_without_outcomes, train_mode=False)


class CurrentArchetypeRegimeBuilder(RegimeFeatureBuilder):
    def __init__(self, *, seed: int = 52) -> None:
        super().__init__(
            name="current_archetype_meta_regimes",
            prefix="ab_current_arch",
            n_clusters=1,
            seed=seed,
        )
        self.priors: pd.DataFrame = pd.DataFrame()
        self.global_target: dict[str, float] = {}

    def feature_names(self) -> list[str]:
        return [
            f"{self.prefix}_prior_{name}"
            for name in ("clean", "bad_mae", "dirty", "timeout", "exec")
        ] + [f"{self.prefix}_support_log1p"]

    def fit(self, train: pd.DataFrame) -> "CurrentArchetypeRegimeBuilder":
        keys = self._keys(train)
        target = _target_frame(train).reset_index(drop=True)
        self.global_target = target.mean(numeric_only=True).to_dict()
        work = pd.concat([keys.reset_index(drop=True), target], axis=1)
        self.priors = work.groupby(["side", "arch", "regime"], dropna=False).agg(
            rows=("clean", "size"),
            clean=("clean", "sum"),
            bad_mae=("bad_mae", "sum"),
            dirty=("dirty", "sum"),
            timeout=("timeout", "sum"),
            exec=("exec", "sum"),
        )
        self.metadata_ = {"status": "ok", "group_count": int(len(self.priors))}
        return self

    def _keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        regime_parts = []
        for col in CURRENT_ARCHETYPE_REGIME_COLUMNS:
            if col in frame.columns:
                regime_parts.append(frame[col].astype(str).fillna("missing"))
        if not regime_parts:
            regime = pd.Series("missing", index=frame.index)
        else:
            regime = regime_parts[0]
            for part in regime_parts[1:4]:
                regime = regime + "|" + part
        return pd.DataFrame(
            {
                "side": frame.get("side_name", pd.Series("missing", index=frame.index))
                .astype(str)
                .str.lower(),
                "arch": archetype_series(frame).astype(str),
                "regime": regime.astype(str),
            },
            index=frame.index,
        )

    def _transform(
        self, frame: pd.DataFrame, *, train_mode: bool = False
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        keys = self._keys(frame).reset_index(drop=True)
        joined = keys.merge(
            self.priors.reset_index(), on=["side", "arch", "regime"], how="left"
        )
        rows = pd.to_numeric(joined.get("rows"), errors="coerce").fillna(0.0)
        own_target = (
            _target_frame(frame).reset_index(drop=True)
            if train_mode
            else pd.DataFrame(index=frame.index)
        )
        for name in ("clean", "bad_mae", "dirty", "timeout", "exec"):
            vals = pd.to_numeric(joined.get(name), errors="coerce").fillna(0.0)
            if train_mode and name in own_target.columns:
                vals = vals - own_target[name].astype(float)
                rows_eff = (rows - 1.0).clip(lower=0.0)
            else:
                rows_eff = rows
            prior = (vals + 50.0 * float(self.global_target.get(name, 0.0))) / (
                rows_eff + 50.0
            )
            out[f"{self.prefix}_prior_{name}"] = prior.astype(np.float32).to_numpy()
        out[f"{self.prefix}_support_log1p"] = (
            np.log1p(rows).astype(np.float32).to_numpy()
        )
        return out

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._transform(train, train_mode=True)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._transform(oos_without_outcomes, train_mode=False)


class ErrorSignatureRegimeBuilder(RegimeFeatureBuilder):
    def __init__(
        self,
        *,
        name: str,
        prefix: str,
        n_clusters: int = 6,
        seed: int = 52,
        joint_features: bool = False,
    ) -> None:
        super().__init__(name=name, prefix=prefix, n_clusters=n_clusters, seed=seed)
        self.joint_features = bool(joint_features)
        self.encoder = MatrixEncoder(max_categories_per_col=24)
        self.scaler = StandardScaler()
        self.cluster_scaler = StandardScaler()
        self.gmm: GaussianMixture | None = None
        self.classifier: ExtraTreesClassifier | None = None
        self.train_clusters: np.ndarray = np.zeros(0, dtype=np.int32)
        self.train_oof_proba: np.ndarray | None = None
        self.target = pd.DataFrame()

    def _descriptor_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        score = _num(frame, "score", 0.0).fillna(0.0)
        exec_margin = _num(frame, "exec_margin", 0.0).fillna(0.0)
        clean = _target_frame(frame)["clean"]
        dirty = _target_frame(frame)["dirty"]
        first_bad = _num(frame, "first_touch_bad_mae_1r", 0.0).fillna(0.0)
        full_bad = _num(frame, "full_path_bad_mae_1r", 0.0).fillna(0.0)
        timeout = _num(frame, "timeout", 0.0).fillna(0.0)
        expected = _num(frame, "base_score_rank_pct_train_prior", 0.5).fillna(0.5)
        false_positive = score.rank(pct=True).ge(0.70).astype(float) * (1.0 - clean)
        missed = score.rank(pct=True).le(0.30).astype(float) * clean
        residual = exec_margin - expected
        side_sign = (
            frame.get("side_name", pd.Series("long", index=frame.index))
            .astype(str)
            .str.lower()
            .map({"short": -1.0, "long": 1.0})
            .fillna(0.0)
        )
        return (
            pd.DataFrame(
                {
                    "base_score_pct": score.rank(pct=True).fillna(0.5),
                    "realized_exec_margin": exec_margin,
                    "clean_positive": clean,
                    "dirty_positive": dirty,
                    "first_touch_bad_mae": first_bad,
                    "full_path_bad_mae": full_bad,
                    "timeout": timeout,
                    "false_positive": false_positive,
                    "missed_opportunity": missed,
                    "calibration_residual": residual,
                    "side_specific_residual": residual * side_sign,
                },
                index=frame.index,
            )
            .fillna(0.0)
            .astype(np.float32)
        )

    def fit(self, train: pd.DataFrame) -> "ErrorSignatureRegimeBuilder":
        num, cat = _candidate_pre_entry_columns(train)
        self.encoder.fit(train, num, cat)
        x_pre = self.encoder.transform(train)
        desc = self._descriptor_matrix(train)
        cluster_input = desc
        if self.joint_features and not x_pre.empty:
            x_scaled = pd.DataFrame(StandardScaler().fit_transform(x_pre)).reset_index(
                drop=True
            )
            cluster_input = pd.concat(
                [
                    x_scaled.iloc[:, : min(80, x_scaled.shape[1])],
                    desc.reset_index(drop=True),
                ],
                axis=1,
            )
            cluster_input.columns = [str(col) for col in cluster_input.columns]
        z = self.cluster_scaler.fit_transform(cluster_input)
        self.gmm = _fit_gmm(z, n_clusters=self.n_clusters, seed=self.seed)
        _, self.train_clusters = _gmm_feature_frame(
            prefix=self.prefix,
            gmm=self.gmm,
            x=z,
            max_clusters=self.n_clusters,
            index=train.index,
        )
        self.target = _target_frame(train).reset_index(drop=True)
        x = (
            self.scaler.fit_transform(x_pre)
            if x_pre.shape[1]
            else np.zeros((len(train), 0), dtype=np.float32)
        )
        self.train_oof_proba = _time_oof_classifier_proba(
            x, self.train_clusters, self.n_clusters, seed=self.seed
        )
        self.classifier = _fit_classifier_final(x, self.train_clusters, seed=self.seed)
        self.metadata_ = {
            "status": "ok"
            if self.gmm is not None and self.classifier is not None
            else "fallback",
            "pre_entry_feature_count": int(x_pre.shape[1]),
            "cluster_input_features": int(cluster_input.shape[1]),
            "joint_features": bool(self.joint_features),
        }
        return self

    def _proba(self, frame: pd.DataFrame, *, train_mode: bool) -> np.ndarray:
        if (
            train_mode
            and self.train_oof_proba is not None
            and len(self.train_oof_proba) == len(frame)
        ):
            return self.train_oof_proba
        x_pre = self.encoder.transform(frame)
        x = (
            self.scaler.transform(x_pre)
            if x_pre.shape[1]
            else np.zeros((len(frame), 0), dtype=np.float32)
        )
        if self.classifier is None or x.shape[1] == 0:
            out = np.zeros((len(frame), self.n_clusters), dtype=np.float32)
            out[:, 0] = 1.0
            return out
        raw = self.classifier.predict_proba(x)
        out = np.zeros((len(frame), self.n_clusters), dtype=np.float32)
        for idx, klass in enumerate(self.classifier.classes_):
            if int(klass) < self.n_clusters:
                out[:, int(klass)] = raw[:, idx]
        rowsum = out.sum(axis=1, keepdims=True)
        out = np.divide(out, np.where(rowsum <= 0.0, 1.0, rowsum))
        return out.astype(np.float32)

    def _transform(self, frame: pd.DataFrame, *, train_mode: bool) -> pd.DataFrame:
        proba = self._proba(frame, train_mode=train_mode)
        clusters = np.asarray(np.argmax(proba, axis=1), dtype=np.int32)
        out = pd.DataFrame(index=frame.index)
        out[f"{self.prefix}_cluster_id"] = clusters.astype(np.float32)
        for i in range(self.n_clusters):
            out[f"{self.prefix}_posterior_{i}"] = proba[:, i]
        out[f"{self.prefix}_entropy"] = _entropy(proba)
        out[f"{self.prefix}_distance"] = (1.0 - np.max(proba, axis=1)).astype(
            np.float32
        )
        priors = _prior_features(
            prefix=self.prefix,
            train_clusters=self.train_clusters,
            target=self.target,
            out_clusters=clusters,
            train_mode=train_mode,
        )
        priors.index = frame.index
        return pd.concat([out, priors], axis=1)

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._transform(train, train_mode=True)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._transform(oos_without_outcomes, train_mode=False)


class SideArchetypeLocalRegimeBuilder(RegimeFeatureBuilder):
    def __init__(
        self, *, seed: int = 52, n_clusters: int = 4, min_rows: int = 400
    ) -> None:
        super().__init__(
            name="side_archetype_local_regimes",
            prefix="ab_local_arch",
            n_clusters=n_clusters,
            seed=seed,
            min_rows=min_rows,
        )
        self.models: dict[tuple[str, str], GMMRegimeBuilder] = {}
        self.side_models: dict[str, GMMRegimeBuilder] = {}
        self.global_model: GMMRegimeBuilder | None = None

    def fit(self, train: pd.DataFrame) -> "SideArchetypeLocalRegimeBuilder":
        side = (
            train.get("side_name", pd.Series("missing", index=train.index))
            .astype(str)
            .str.lower()
        )
        arch = archetype_series(train).astype(str)
        self.global_model = GMMRegimeBuilder(
            name=self.name,
            prefix=self.prefix,
            n_clusters=self.n_clusters,
            seed=self.seed,
        ).fit(train)
        for side_key, group in train.groupby(side, dropna=False):
            if len(group) >= self.min_rows:
                self.side_models[(str(side_key))] = GMMRegimeBuilder(
                    name=f"{self.name}_{side_key}",
                    prefix=self.prefix,
                    n_clusters=self.n_clusters,
                    seed=self.seed,
                ).fit(group)
        keys = pd.DataFrame({"side": side, "arch": arch}, index=train.index)
        for (side_key, arch_key), idx in keys.groupby(
            ["side", "arch"], dropna=False
        ).groups.items():
            group = train.loc[idx]
            if len(group) >= self.min_rows:
                self.models[(str(side_key), str(arch_key))] = GMMRegimeBuilder(
                    name=f"{self.name}_{side_key}_{arch_key}",
                    prefix=self.prefix,
                    n_clusters=self.n_clusters,
                    seed=self.seed,
                ).fit(group)
        self.metadata_ = {
            "local_models": int(len(self.models)),
            "side_fallback_models": int(len(self.side_models)),
        }
        return self

    def _transform_partitioned(
        self, frame: pd.DataFrame, *, train_mode: bool
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_names():
            out[col] = 0.0
        side = (
            frame.get("side_name", pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        arch = archetype_series(frame).astype(str)
        keys = pd.DataFrame({"side": side, "arch": arch}, index=frame.index)
        for (side_key, arch_key), idx in keys.groupby(
            ["side", "arch"], dropna=False
        ).groups.items():
            model = (
                self.models.get((str(side_key), str(arch_key)))
                or self.side_models.get(str(side_key))
                or self.global_model
            )
            if model is None:
                continue
            part = frame.loc[idx]
            transformed = (
                model.transform_train(part)
                if train_mode
                else model.transform_oos(drop_oos_outcome_columns(part))
            )
            for col in self.feature_names():
                if col in transformed.columns:
                    out.loc[idx, col] = (
                        pd.to_numeric(transformed[col], errors="coerce")
                        .fillna(0.0)
                        .to_numpy()
                    )
        return out

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._transform_partitioned(train, train_mode=True)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._transform_partitioned(oos_without_outcomes, train_mode=False)


class SupervisedEmbeddingRegimeBuilder(RegimeFeatureBuilder):
    def __init__(self, *, seed: int = 52, n_clusters: int = 6) -> None:
        super().__init__(
            name="supervised_embedding_regimes",
            prefix="ab_sup_embed",
            n_clusters=n_clusters,
            seed=seed,
        )
        self.encoder = MatrixEncoder(max_categories_per_col=24)
        self.scaler = StandardScaler()
        self.models: dict[str, Any] = {}
        self.embed_scaler = StandardScaler()
        self.gmm: GaussianMixture | None = None
        self.train_clusters: np.ndarray = np.zeros(0, dtype=np.int32)
        self.train_embedding: np.ndarray | None = None
        self.target = pd.DataFrame()

    def fit(self, train: pd.DataFrame) -> "SupervisedEmbeddingRegimeBuilder":
        num, cat = _candidate_pre_entry_columns(train)
        self.encoder.fit(train, num, cat)
        x_pre = self.encoder.transform(train)
        x = (
            self.scaler.fit_transform(x_pre)
            if x_pre.shape[1]
            else np.zeros((len(train), 0), dtype=np.float32)
        )
        targets = _target_frame(train)
        embeddings = []
        final_models: dict[str, Any] = {}
        for name, y in {
            "clean": targets["clean"],
            "dirty": targets["dirty"],
            "bad_mae": targets["bad_mae"],
            "timeout": targets["timeout"],
        }.items():
            y_arr = y.to_numpy(dtype=np.float32)
            embeddings.append(
                _time_oof_classifier_score(x, y_arr, seed=self.seed + len(embeddings))
            )
            final_models[name] = _fit_classifier_final(
                x, (y_arr >= 0.5).astype(np.int32), seed=self.seed + len(embeddings)
            )
        exec_y = targets["exec"].to_numpy(dtype=np.float32)
        embeddings.append(_time_oof_regression_score(x, exec_y, seed=self.seed + 99))
        final_models["exec"] = _fit_regressor_final(x, exec_y, seed=self.seed + 99)
        emb = np.column_stack(embeddings).astype(np.float32)
        self.train_embedding = emb
        z = self.embed_scaler.fit_transform(emb)
        self.gmm = _fit_gmm(z, n_clusters=self.n_clusters, seed=self.seed)
        _, self.train_clusters = _gmm_feature_frame(
            prefix=self.prefix,
            gmm=self.gmm,
            x=z,
            max_clusters=self.n_clusters,
            index=train.index,
        )
        self.target = targets.reset_index(drop=True)
        self.models = final_models
        self.metadata_ = {
            "status": "ok" if self.gmm is not None else "fallback",
            "pre_entry_feature_count": int(x_pre.shape[1]),
        }
        return self

    def _embedding(self, frame: pd.DataFrame, *, train_mode: bool) -> np.ndarray:
        if (
            train_mode
            and self.train_embedding is not None
            and len(self.train_embedding) == len(frame)
        ):
            return self.train_embedding
        x_pre = self.encoder.transform(frame)
        x = (
            self.scaler.transform(x_pre)
            if x_pre.shape[1]
            else np.zeros((len(frame), 0), dtype=np.float32)
        )
        cols = []
        for name in ("clean", "dirty", "bad_mae", "timeout"):
            model = self.models.get(name)
            cols.append(_predict_classifier_score(model, x))
        model = self.models.get("exec")
        cols.append(_predict_regressor_score(model, x))
        return np.column_stack(cols).astype(np.float32)

    def _transform(self, frame: pd.DataFrame, *, train_mode: bool) -> pd.DataFrame:
        emb = self._embedding(frame, train_mode=train_mode)
        z = self.embed_scaler.transform(emb)
        features, clusters = _gmm_feature_frame(
            prefix=self.prefix,
            gmm=self.gmm,
            x=z,
            max_clusters=self.n_clusters,
            index=frame.index,
        )
        for i, name in enumerate(
            ("clean_pred", "dirty_pred", "bad_mae_pred", "timeout_pred", "exec_pred")
        ):
            features[f"{self.prefix}_{name}"] = emb[:, i].astype(np.float32)
        priors = _prior_features(
            prefix=self.prefix,
            train_clusters=self.train_clusters,
            target=self.target,
            out_clusters=clusters,
            train_mode=train_mode,
        )
        priors.index = frame.index
        return pd.concat([features, priors], axis=1)

    def feature_names(self) -> list[str]:
        return super().feature_names() + [
            f"{self.prefix}_{name}"
            for name in (
                "clean_pred",
                "dirty_pred",
                "bad_mae_pred",
                "timeout_pred",
                "exec_pred",
            )
        ]

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self._transform(train, train_mode=True)

    def transform_oos(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        leaked = sorted(set(oos_without_outcomes.columns).intersection(OUTCOME_COLUMNS))
        if leaked:
            raise ValueError(
                f"OOS frame still contains target/outcome columns: {leaked[:10]}"
            )
        return self._transform(oos_without_outcomes, train_mode=False)


def _fit_classifier_final(
    x: np.ndarray, y: np.ndarray, *, seed: int, max_fit_rows: int = 120_000
) -> ExtraTreesClassifier | None:
    if x.shape[1] == 0 or len(np.unique(y)) < 2:
        return None
    if len(x) > int(max_fit_rows):
        idx = _time_spread_indices(len(x), int(max_fit_rows))
        x = x[idx]
        y = np.asarray(y)[idx]
        if len(np.unique(y)) < 2:
            return None
    return ExtraTreesClassifier(
        n_estimators=80,
        min_samples_leaf=30,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=1,
    ).fit(x, y)


def _fit_regressor_final(
    x: np.ndarray, y: np.ndarray, *, seed: int, max_fit_rows: int = 120_000
) -> ExtraTreesRegressor | None:
    if x.shape[1] == 0 or len(y) < 30:
        return None
    if len(x) > int(max_fit_rows):
        idx = _time_spread_indices(len(x), int(max_fit_rows))
        x = x[idx]
        y = np.asarray(y)[idx]
    return ExtraTreesRegressor(
        n_estimators=80,
        min_samples_leaf=30,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=1,
    ).fit(x, y)


def _time_splits(n: int, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    if n < 90:
        return []
    edges = np.linspace(0, n, n_splits + 2, dtype=int)
    splits = []
    for i in range(1, len(edges) - 1):
        tr = np.arange(0, edges[i], dtype=np.int64)
        va = np.arange(edges[i], edges[i + 1], dtype=np.int64)
        if len(tr) >= 30 and len(va) > 0:
            splits.append((tr, va))
    return splits


def _time_oof_classifier_proba(
    x: np.ndarray, y: np.ndarray, n_classes: int, *, seed: int
) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    counts = np.bincount(np.asarray(y, dtype=np.int32), minlength=n_classes).astype(
        np.float32
    )
    prior = counts / max(float(counts.sum()), 1.0)
    out[:] = prior
    for tr, va in _time_splits(len(y)):
        model = _fit_classifier_final(x[tr], y[tr], seed=seed)
        if model is None:
            continue
        raw = model.predict_proba(x[va])
        for idx, klass in enumerate(model.classes_):
            if int(klass) < n_classes:
                out[va, int(klass)] = raw[:, idx]
    rowsum = out.sum(axis=1, keepdims=True)
    return np.divide(out, np.where(rowsum <= 0.0, 1.0, rowsum)).astype(np.float32)


def _time_oof_classifier_score(
    x: np.ndarray, y: np.ndarray, *, seed: int
) -> np.ndarray:
    y_bin = (np.asarray(y) >= 0.5).astype(np.int32)
    out = np.full(
        len(y_bin), float(y_bin.mean()) if len(y_bin) else 0.0, dtype=np.float32
    )
    for tr, va in _time_splits(len(y_bin)):
        model = _fit_classifier_final(x[tr], y_bin[tr], seed=seed)
        if model is not None:
            out[va] = _predict_classifier_score(model, x[va])
    return out


def _time_oof_regression_score(
    x: np.ndarray, y: np.ndarray, *, seed: int
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    out = np.full(len(y), float(np.nanmean(y)) if len(y) else 0.0, dtype=np.float32)
    for tr, va in _time_splits(len(y)):
        model = _fit_regressor_final(x[tr], y[tr], seed=seed)
        if model is not None:
            out[va] = _predict_regressor_score(model, x[va])
    return out


def _predict_classifier_score(
    model: ExtraTreesClassifier | None, x: np.ndarray
) -> np.ndarray:
    if model is None or x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    raw = model.predict_proba(x)
    if 1 in set(int(c) for c in model.classes_):
        idx = list(model.classes_).index(1)
        return raw[:, idx].astype(np.float32)
    return np.zeros(x.shape[0], dtype=np.float32)


def _predict_regressor_score(
    model: ExtraTreesRegressor | None, x: np.ndarray
) -> np.ndarray:
    if model is None or x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def make_regime_builder(arm: str, *, seed: int = 52) -> RegimeFeatureBuilder | None:
    if arm in {
        "baseline_current_full_context",
        "baseline_no_cross_context",
        "causal_phase_state_context",
        "side_archetype_identity_context",
        "causal_phase_side_archetype_context",
    }:
        return None
    if arm == "current_archetype_meta_regimes":
        return CurrentArchetypeRegimeBuilder(seed=seed)
    if arm == "meta_feature_only_regimes":
        return GMMRegimeBuilder(
            name=arm, prefix="ab_meta_feat", n_clusters=5, seed=seed
        )
    if arm == "base_error_signature_regimes":
        return ErrorSignatureRegimeBuilder(
            name=arm,
            prefix="ab_base_error",
            n_clusters=6,
            seed=seed,
            joint_features=False,
        )
    if arm == "joint_feature_error_regimes":
        return ErrorSignatureRegimeBuilder(
            name=arm,
            prefix="ab_joint_error",
            n_clusters=6,
            seed=seed,
            joint_features=True,
        )
    if arm == "side_archetype_local_regimes":
        return SideArchetypeLocalRegimeBuilder(seed=seed, n_clusters=4)
    if arm == "temporal_reliability_regimes":
        return GMMRegimeBuilder(
            name=arm, prefix="ab_temp_rel", n_clusters=5, seed=seed, temporal_only=True
        )
    if arm == "supervised_embedding_regimes":
        return SupervisedEmbeddingRegimeBuilder(seed=seed, n_clusters=6)
    raise ValueError(f"Unknown meta regime ablation arm: {arm}")


def regime_feature_names(arm: str, *, seed: int = 52) -> list[str]:
    builder = make_regime_builder(arm, seed=seed)
    return [] if builder is None else builder.feature_names()


def apply_regime_builder_fold(
    arm: str,
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    builder = make_regime_builder(arm, seed=seed)
    if builder is None:
        return train, valid, [], {"arm": arm, "status": "baseline_no_builder"}
    builder.fit(train)
    train_aug_features = builder.transform_train(train)
    valid_safe = drop_oos_outcome_columns(valid)
    valid_aug_features = builder.transform_oos(valid_safe)
    train_aug = train.copy(deep=False)
    valid_aug = valid.copy(deep=False)
    for col in builder.feature_names():
        if col in train_aug_features.columns:
            train_aug[col] = train_aug_features[col].to_numpy()
        else:
            train_aug[col] = 0.0
        if col in valid_aug_features.columns:
            valid_aug[col] = valid_aug_features[col].to_numpy()
        else:
            valid_aug[col] = 0.0
    return train_aug, valid_aug, builder.feature_names(), builder.manifest()
