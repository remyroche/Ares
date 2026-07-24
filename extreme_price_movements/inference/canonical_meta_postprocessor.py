"""Frozen V9 tail-95, market-state MLP, and hierarchical-EV inference chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.regime_ev_calibration import (
    apply_regime_ev_calibration,
    load_regime_ev_calibration,
    required_feature_columns,
)
from extreme_price_movements.residual_event_archetypes import OUTCOME_COLUMNS
from extreme_price_movements.residual_event_archetypes import (
    residual_event_feature_names,
    residual_event_market_feature_names,
)


V9_TAIL95_POLICY_ID = (
    "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_"
    "20260712_v9::forced_local_tail_0.950"
)
MLP_HIER_EV_POLICY_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
V9_TAIL_ONLY_POLICY_ID = "meta_residual_v9_tail95_v1"
V9_TAIL_HIER_EV_POLICY_ID = "meta_residual_v9_tail95_hier_ev_no_mlp_v1"

POST_V9_DERIVED_FEATURES = {
    "hit_probability",
    "policy_parent_rank",
    "meta_hit_probability_local_top10_margin",
    "meta_hit_probability_uncertainty_p1mp",
    "meta_parent_rank_margin_top10",
    "meta_parent_rank_uncertainty_p1mp",
}


def _add_post_v9_reliability_features(
    frame: pd.DataFrame,
    parent_rank: pd.Series,
) -> pd.DataFrame:
    """Reproduce the observable reliability inputs used during MLP fitting."""

    rank = pd.to_numeric(parent_rank, errors="coerce").clip(0.0, 1.0)
    out = frame.copy(deep=False)
    out["policy_parent_rank"] = rank.astype(np.float32)
    # The promoted MLP training contract defines hit probability from the
    # frozen predecessor rank. Local q90 margins are added inside each frozen
    # local overlay from its train-only reliability reference.
    out["hit_probability"] = rank.astype(np.float32)
    uncertainty = (rank * (1.0 - rank)).astype(np.float32)
    out["meta_hit_probability_uncertainty_p1mp"] = uncertainty
    out["meta_parent_rank_uncertainty_p1mp"] = uncertainty
    out["meta_parent_rank_margin_top10"] = (rank - 0.90).astype(np.float32)
    return out


def _normalize_model_archetype_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the unprefixed archetype keys present in the frozen train contract."""
    name = "archetype_policy_key"
    if name not in frame.columns:
        return frame
    out = frame.copy()
    values = out[name].astype("string")
    sides = out.get("side_name", pd.Series("", index=out.index)).astype("string")
    for side in ("long", "short"):
        mask = sides.str.lower().eq(side) & values.str.startswith(f"{side}__", na=False)
        values.loc[mask] = values.loc[mask].str[len(side) + 2 :]
    out[name] = values.astype(object)
    return out


def _constant_recognizer_inputs(recognizer: Any) -> set[str]:
    """Return structurally constant recognizer inputs with no scoring effect."""
    if recognizer is None:
        return set()
    models = list(getattr(recognizer, "local_models", {}).values()) + list(
        getattr(recognizer, "side_models", {}).values()
    )
    occurrences: dict[str, list[bool]] = {}
    for model in models:
        columns = list(getattr(model, "feature_columns", []) or [])
        medians = np.asarray(getattr(model, "medians", []), dtype=np.float64)
        lows = np.asarray(getattr(model, "clip_low", []), dtype=np.float64)
        highs = np.asarray(getattr(model, "clip_high", []), dtype=np.float64)
        if not (len(columns) == medians.size == lows.size == highs.size):
            continue
        for idx, name in enumerate(columns):
            values = np.asarray(
                [medians[idx], lows[idx], highs[idx]], dtype=np.float64
            )
            is_constant = bool(
                np.isfinite(values).all()
                and np.max(np.abs(values - values[0])) <= 1e-12
            )
            occurrences.setdefault(str(name), []).append(is_constant)
    return {name for name, flags in occurrences.items() if flags and all(flags)}


@dataclass(frozen=True)
class CanonicalMetaPostprocessor:
    """Apply the exact frozen post-meta chain used by historical policy replay."""

    predecessor_bundle: Any
    residual_event_state: Any
    regime_ev_artifact: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        *,
        predecessor_bundle_path: str | Path,
        residual_event_state_path: str | Path,
        regime_ev_artifact_path: str | Path,
    ) -> "CanonicalMetaPostprocessor":
        predecessor_path = Path(predecessor_bundle_path)
        residual_path = Path(residual_event_state_path)
        regime_path = Path(regime_ev_artifact_path)
        for label, path in (
            ("V9 predecessor", predecessor_path),
            ("residual-event state", residual_path),
            ("regime-EV artifact", regime_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Canonical {label} must be a file, got: {path}"
                )
        predecessor = joblib.load(predecessor_path)
        residual_state = joblib.load(residual_path)
        artifact = load_regime_ev_calibration(regime_path)
        instance = cls(predecessor, residual_state, artifact)
        instance.validate_contract()
        return instance

    def validate_contract(self) -> None:
        policy_id = str(self.regime_ev_artifact.get("policy_id") or "")
        predecessor_id = str(
            self.regime_ev_artifact.get("predecessor_policy_id") or ""
        )
        if policy_id != MLP_HIER_EV_POLICY_ID:
            raise ValueError(
                f"Unexpected canonical postprocessor policy: {policy_id!r}"
            )
        if predecessor_id != V9_TAIL95_POLICY_ID:
            raise ValueError(
                f"Unexpected canonical predecessor policy: {predecessor_id!r}"
            )
        rank_reference = getattr(
            self.predecessor_bundle, "historical_rank_reference", None
        )
        if rank_reference is None:
            raise ValueError("V9 predecessor bundle has no frozen historical rank")
        if not isinstance(self.regime_ev_artifact.get("expected_ev_mapping"), Mapping):
            raise ValueError("Canonical postprocessor has no hierarchical EV mapping")

    def required_input_features(self) -> list[str]:
        required = list(self.predecessor_bundle.required_input_features())
        state = self.residual_event_state
        for model in getattr(state, "local_models", {}).values():
            required.extend(getattr(model, "feature_columns", []) or [])
        market_model = getattr(state, "market_model", None)
        if market_model is not None:
            required.extend(getattr(market_model, "feature_columns", []) or [])
        generated = set(residual_event_feature_names())
        generated.update(residual_event_market_feature_names())
        injected = {
            "archetype_policy_key",
            "calibrated_score",
            "hit_probability",
            "score",
            "score_base",
            "score_meta_base_soft_label",
            "score_meta_base_soft_label_raw_refit",
            "side_name",
            "__symbol__",
            "__ts__",
        }
        required.extend(
            name
            for name in required_feature_columns(self.regime_ev_artifact)
            if str(name) not in generated
            and str(name) not in injected
            and str(name) not in POST_V9_DERIVED_FEATURES
        )
        # Some legacy categorical regime labels were passed through a numeric
        # recognizer adapter and therefore collapsed to a constant training
        # value. The frozen recognizer restores that same median when absent;
        # requesting such columns from the raw market store only causes an
        # expensive recomputation with no possible scoring effect.
        recognizer = getattr(self.predecessor_bundle, "residual_recognizer", None)
        constant_recognizer_inputs = _constant_recognizer_inputs(recognizer)
        return list(
            dict.fromkeys(
                str(name)
                for name in required
                if str(name)
                and str(name) not in generated
                and str(name) not in injected
                and str(name) not in constant_recognizer_inputs
            )
        )

    def complete_case_report(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Report finite observable inputs for each row's actual frozen models.

        Internal PCA/AE/GMM outputs and structurally constant recognizer inputs
        are not raw requirements. Every observable input used by the applicable
        V9 recognizer, residual-state model, and local market-state calibrator is.
        """
        safe = _normalize_model_archetype_keys(frame)
        requirements = {idx: set() for idx in safe.index}
        all_rows = set(safe.index)
        predecessor = self.predecessor_bundle

        global_required = set(
            str(name) for name in getattr(predecessor, "raw_selected_features", [])
        )
        representation = getattr(predecessor, "residual_representation_state", None)
        representation_outputs: set[str] = set()
        if isinstance(representation, Mapping):
            global_required.update(
                str(name) for name in representation.get("columns", [])
            )
            representation_outputs.update(
                str(name) for name in representation.get("output_columns", [])
            )
        shock_state = getattr(predecessor, "shock_overlay_state", None)
        if shock_state is not None:
            global_required.update(str(name) for name in shock_state.required_features())
        for idx in all_rows:
            requirements[idx].update(global_required)

        sides = safe.get("side_name", pd.Series("", index=safe.index)).astype(str)
        arches = safe.get(
            "archetype_policy_key", pd.Series("", index=safe.index)
        ).astype(str)

        recognizer = getattr(predecessor, "residual_recognizer", None)
        constant_inputs = _constant_recognizer_inputs(recognizer)
        if recognizer is not None:
            for idx, side, arch in zip(safe.index, sides, arches, strict=True):
                model = getattr(recognizer, "local_models", {}).get((side, arch))
                if model is None and bool(
                    getattr(getattr(recognizer, "config", None), "allow_side_fallback", False)
                ):
                    model = getattr(recognizer, "side_models", {}).get(side)
                if model is not None:
                    requirements[idx].update(
                        str(name)
                        for name in getattr(model, "feature_columns", [])
                        if str(name) not in representation_outputs
                        and str(name) not in constant_inputs
                    )

        residual_state = self.residual_event_state
        for idx, side, arch in zip(safe.index, sides, arches, strict=True):
            local_key = f"{side}|{arch or 'missing'}"
            model = getattr(residual_state, "local_models", {}).get(local_key)
            if model is None and bool(
                getattr(getattr(residual_state, "config", None), "allow_side_fallback", False)
            ):
                model = getattr(residual_state, "side_models", {}).get(side)
            if model is not None:
                requirements[idx].update(
                    str(name) for name in getattr(model, "feature_columns", [])
                )
        market_model = getattr(residual_state, "market_model", None)
        if market_model is not None:
            market_required = {
                str(name) for name in getattr(market_model, "feature_columns", [])
            }
            for idx in all_rows:
                requirements[idx].update(market_required)

        generated = set(residual_event_feature_names())
        generated.update(residual_event_market_feature_names())
        generated.update({"calibrated_score", "hit_probability"})
        generated.update(POST_V9_DERIVED_FEATURES)
        for effect in self.regime_ev_artifact.get("effects", []) or []:
            if not isinstance(effect, Mapping):
                continue
            effect_side = str(effect.get("side_name") or effect.get("side") or "")
            effect_arch = str(
                effect.get("archetype_policy_key") or effect.get("archetype") or ""
            )
            raw = {
                str(name)
                for name in effect.get("feature_cols", []) or []
                if str(name)
                and str(name) not in generated
                and not str(name).startswith("__")
            }
            feature_col = str(effect.get("feature_col") or "")
            if feature_col and feature_col not in generated and not feature_col.startswith("__"):
                raw.add(feature_col)
            for idx, side, arch in zip(safe.index, sides, arches, strict=True):
                if effect_side not in {"", "*", side}:
                    continue
                if effect_arch not in {"", "*", arch}:
                    continue
                requirements[idx].update(raw)

        injected_numeric = (
            "score",
            "score_base",
            "score_meta_base_soft_label",
        )
        # Rows share a small number of side/archetype-specific contracts. Check
        # each unique contract as one numeric matrix instead of performing a
        # scalar pandas conversion for every row-feature pair.
        contract_rows: dict[tuple[str, ...], list[Any]] = {}
        for idx in safe.index:
            requirements[idx].difference_update(POST_V9_DERIVED_FEATURES)
            names = tuple(sorted(requirements[idx].union(injected_numeric)))
            contract_rows.setdefault(names, []).append(idx)

        complete = pd.Series(False, index=safe.index, dtype=bool)
        required_count = pd.Series(0, index=safe.index, dtype=np.int32)
        missing_count = pd.Series(0, index=safe.index, dtype=np.int32)
        missing_features = pd.Series("", index=safe.index, dtype=object)
        available_columns = set(safe.columns)
        for names, indices in contract_rows.items():
            required_count.loc[indices] = len(names)
            absent = [name for name in names if name not in available_columns]
            present = [name for name in names if name in available_columns]
            if present:
                matrix = (
                    safe.loc[indices, present]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float64, copy=False)
                )
                finite = np.isfinite(matrix)
            else:
                finite = np.empty((len(indices), 0), dtype=bool)
            row_missing = (~finite).sum(axis=1).astype(np.int32) + len(absent)
            complete.loc[indices] = row_missing == 0
            missing_count.loc[indices] = row_missing
            if bool((row_missing > 0).any()):
                absent_text = ",".join(absent)
                for row_position in np.flatnonzero(row_missing > 0):
                    nonfinite_names = [
                        present[column_position]
                        for column_position in np.flatnonzero(~finite[row_position])
                    ]
                    names_text = ",".join([*absent, *nonfinite_names])
                    missing_features.at[indices[int(row_position)]] = names_text
        return pd.DataFrame(
            {
                "complete_case": complete,
                "required_feature_count": required_count,
                "missing_feature_count": missing_count,
                "missing_features": missing_features,
            },
            index=safe.index,
        )

    def _prepare_predecessor_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Historical backfill once forced newer lifecycle columns to NaN so the
        # frozen wrapper would median-fill them. Canonical replay/live scoring
        # now materializes those observable inputs and rejects incomplete rows.
        return _normalize_model_archetype_keys(frame)

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        """Return V9/MLP/hierarchical-EV outputs without using outcome columns."""
        out = frame.copy() if copy else frame
        safe = out.drop(
            columns=[name for name in OUTCOME_COLUMNS if name in out.columns],
            errors="ignore",
        )
        safe = self._prepare_predecessor_input(safe)
        complete_case = self.complete_case_report(safe)
        rejected = ~complete_case["complete_case"].astype(bool)
        if bool(rejected.any()):
            sample = complete_case.loc[
                rejected, ["missing_feature_count", "missing_features"]
            ].head(5)
            raise RuntimeError(
                "Canonical postprocessor received non-complete rows: "
                f"rejected={int(rejected.sum())}/{len(complete_case)} "
                f"sample={sample.to_dict(orient='index')}"
            )

        predecessor = self.predict_predecessor(safe)
        predecessor_input = self.attach_predecessor(safe, predecessor)
        state_features = self.residual_event_state.transform_oos(predecessor_input)
        return self.apply_from_components(
            out,
            predecessor=predecessor,
            residual_state_features=state_features,
            copy=False,
        )

    def predict_predecessor(self, frame: pd.DataFrame) -> pd.DataFrame:
        safe = frame.drop(
            columns=[name for name in OUTCOME_COLUMNS if name in frame.columns],
            errors="ignore",
        )
        safe = self._prepare_predecessor_input(safe)
        predecessor = self.predecessor_bundle.predict(safe)
        if "historical_rank" not in predecessor.columns:
            raise RuntimeError("V9 predecessor did not emit historical_rank")
        return predecessor.set_axis(frame.index)

    def attach_predecessor(
        self, frame: pd.DataFrame, predecessor: pd.DataFrame
    ) -> pd.DataFrame:
        frame = self._prepare_predecessor_input(frame)
        predecessor = predecessor.set_axis(frame.index)
        out = pd.concat(
            [
                frame.drop(
                    columns=[
                        name for name in predecessor.columns if name in frame.columns
                    ],
                    errors="ignore",
                ),
                predecessor,
            ],
            axis=1,
            copy=False,
        )
        if "historical_rank" in out:
            out = _add_post_v9_reliability_features(
                out,
                pd.to_numeric(out["historical_rank"], errors="coerce"),
            )
        return out

    def apply_from_components(
        self,
        frame: pd.DataFrame,
        *,
        predecessor: pd.DataFrame,
        residual_state_features: pd.DataFrame,
        copy: bool = True,
    ) -> pd.DataFrame:
        out = frame.copy() if copy else frame
        predecessor = predecessor.set_axis(out.index)
        state_features = residual_state_features.set_axis(out.index)
        post_state_overlay = getattr(
            self.predecessor_bundle, "apply_residual_overlay", None
        )
        if callable(post_state_overlay):
            overlay_input = pd.concat(
                [
                    frame.drop(
                        columns=[
                            name for name in state_features.columns if name in frame.columns
                        ],
                        errors="ignore",
                    ),
                    state_features,
                ],
                axis=1,
                copy=False,
            )
            predecessor = post_state_overlay(overlay_input, predecessor).set_axis(
                out.index
            )
        generated = pd.concat([predecessor, state_features], axis=1, copy=False)
        out = pd.concat(
            [
                out.drop(
                    columns=[name for name in generated.columns if name in out.columns],
                    errors="ignore",
                ),
                generated,
            ],
            axis=1,
            copy=False,
        )

        parent_rank = pd.to_numeric(out["historical_rank"], errors="coerce")
        if not bool(np.isfinite(parent_rank.to_numpy(dtype=np.float64)).all()):
            raise RuntimeError("V9 predecessor historical rank contains non-finite rows")
        out["calibrated_score"] = parent_rank.astype(np.float32)
        # Preserve the actual final-meta prediction separately. Historical replay
        # used calibrated_score as the canonical parent input for this chain.
        out["meta_postprocessor_parent_rank"] = parent_rank.astype(np.float32)
        out = _add_post_v9_reliability_features(out, parent_rank)
        return apply_regime_ev_calibration(
            out,
            self.regime_ev_artifact,
            source_score_col="calibrated_score",
            adjusted_score_col="score_regime_calibrated",
            copy=False,
        )


@dataclass(frozen=True)
class V9TailPostprocessor:
    """Frozen V9 residual-tail overlay with an optional hierarchical EV map.

    The map is retained when the market-state MLP is retired: V9's frozen rank
    is mapped directly to the same side x archetype expected-EV unit.  No MLP
    effects or score correction are evaluated.
    """

    predecessor_bundle: Any
    residual_event_state: Any
    hierarchical_ev_artifact: Mapping[str, Any] | None = None

    @classmethod
    def load(
        cls,
        *,
        predecessor_bundle_path: str | Path,
        residual_event_state_path: str | Path,
        hierarchical_ev_artifact_path: str | Path | None = None,
    ) -> "V9TailPostprocessor":
        predecessor_path = Path(predecessor_bundle_path)
        residual_path = Path(residual_event_state_path)
        for label, path in (("V9 predecessor", predecessor_path), ("residual-event state", residual_path)):
            if not path.is_file():
                raise FileNotFoundError(f"V9-only {label} must be a file, got: {path}")
        hierarchical_ev_artifact = None
        if hierarchical_ev_artifact_path:
            map_path = Path(hierarchical_ev_artifact_path)
            if not map_path.is_file():
                raise FileNotFoundError(
                    f"V9 hierarchical-EV artifact must be a file, got: {map_path}"
                )
            source_artifact = load_regime_ev_calibration(map_path)
            expected_ev_mapping = source_artifact.get("expected_ev_mapping")
            if not isinstance(expected_ev_mapping, Mapping) or not expected_ev_mapping:
                raise ValueError("V9 hierarchical-EV artifact has no expected-EV mapping")
            # Keep mapping and aliases, but remove every learned MLP/regime
            # effect.  With zero risk adjustment, V9 historical_rank is the
            # direct input to the monotonic side x archetype EV curves.
            hierarchical_ev_artifact = dict(source_artifact)
            hierarchical_ev_artifact["effects"] = []
            hierarchical_ev_artifact["policy_id"] = V9_TAIL_HIER_EV_POLICY_ID
            hierarchical_ev_artifact["predecessor_policy_id"] = V9_TAIL95_POLICY_ID
        return cls(
            joblib.load(predecessor_path),
            joblib.load(residual_path),
            hierarchical_ev_artifact,
        )

    def required_input_features(self) -> list[str]:
        required = list(self.predecessor_bundle.required_input_features())
        for model in getattr(self.residual_event_state, "local_models", {}).values():
            required.extend(getattr(model, "feature_columns", []) or [])
        market_model = getattr(self.residual_event_state, "market_model", None)
        if market_model is not None:
            required.extend(getattr(market_model, "feature_columns", []) or [])
        generated = set(residual_event_feature_names()) | set(residual_event_market_feature_names())
        return list(dict.fromkeys(str(name) for name in required if str(name) not in generated))

    def _contract_shell(self) -> CanonicalMetaPostprocessor:
        mapping_only = dict(self.hierarchical_ev_artifact or {})
        mapping_only["effects"] = []
        return CanonicalMetaPostprocessor(
            self.predecessor_bundle,
            self.residual_event_state,
            mapping_only,
        )

    def complete_case_report(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._contract_shell().complete_case_report(frame)

    def predict_predecessor(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._contract_shell().predict_predecessor(frame)

    def attach_predecessor(
        self, frame: pd.DataFrame, predecessor: pd.DataFrame
    ) -> pd.DataFrame:
        return self._contract_shell().attach_predecessor(frame, predecessor)

    def transform(self, frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
        out = frame.copy() if copy else frame
        safe = _normalize_model_archetype_keys(
            out.drop(columns=[name for name in OUTCOME_COLUMNS if name in out.columns], errors="ignore")
        )
        predecessor = self.predict_predecessor(safe)
        predecessor_input = self.attach_predecessor(safe, predecessor)
        state_features = self.residual_event_state.transform_oos(predecessor_input).set_axis(out.index)
        return self.apply_from_components(
            out,
            predecessor=predecessor,
            residual_state_features=state_features,
            copy=False,
        )

    def apply_from_components(
        self,
        frame: pd.DataFrame,
        *,
        predecessor: pd.DataFrame,
        residual_state_features: pd.DataFrame,
        copy: bool = True,
    ) -> pd.DataFrame:
        out = frame.copy() if copy else frame
        safe = _normalize_model_archetype_keys(
            out.drop(
                columns=[name for name in OUTCOME_COLUMNS if name in out.columns],
                errors="ignore",
            )
        )
        predecessor = predecessor.set_axis(out.index)
        state_features = residual_state_features.set_axis(out.index)
        apply_overlay = getattr(self.predecessor_bundle, "apply_residual_overlay", None)
        if callable(apply_overlay):
            overlay_input = pd.concat(
                [
                    safe.drop(
                        columns=[
                            name
                            for name in state_features.columns
                            if name in safe.columns
                        ],
                        errors="ignore",
                    ),
                    state_features,
                ],
                axis=1,
                copy=False,
            )
            predecessor = apply_overlay(overlay_input, predecessor).set_axis(out.index)
        parent_rank = pd.to_numeric(predecessor["historical_rank"], errors="coerce")
        if not np.isfinite(parent_rank.to_numpy(dtype=np.float64)).all():
            raise RuntimeError("V9-only postprocessor emitted non-finite historical rank")
        generated = pd.concat([predecessor, state_features], axis=1, copy=False)
        out = pd.concat(
            [out.drop(columns=[name for name in generated.columns if name in out.columns], errors="ignore"), generated],
            axis=1,
            copy=False,
        )
        out["calibrated_score"] = parent_rank.astype(np.float32)
        out["meta_postprocessor_parent_rank"] = parent_rank.astype(np.float32)
        if self.hierarchical_ev_artifact:
            mapping_only_artifact = dict(self.hierarchical_ev_artifact)
            mapping_only_artifact["effects"] = []
            mapping_only_artifact["policy_id"] = V9_TAIL_HIER_EV_POLICY_ID
            mapping_only_artifact["predecessor_policy_id"] = V9_TAIL95_POLICY_ID
            out = apply_regime_ev_calibration(
                out,
                mapping_only_artifact,
                source_score_col="calibrated_score",
                adjusted_score_col="score_regime_calibrated",
                copy=False,
            )
            out["market_state_mlp_score_correction"] = np.float32(0.0)
            out["meta_postprocessor_policy_id"] = V9_TAIL_HIER_EV_POLICY_ID
        else:
            out["meta_postprocessor_policy_id"] = V9_TAIL_ONLY_POLICY_ID
        return out
