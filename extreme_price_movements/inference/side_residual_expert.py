"""Frozen side-local base-residual expert used by meta inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.supervised_market_state_calibration import (
    expected_ev_rank,
    predict_hierarchical_ev,
)


SUPPORTED_SCHEMAS = {
    "side_base_residual_expert_staged_selection_ablation_v1",
    "side_base_residual_expert_inference_v2",
}
SIDE_ROUTED_SCHEMA = "side_routed_side_residual_expert_v1"


def _normalize_archetype_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if "archetype_policy_key" not in frame:
        return frame
    out = frame.copy(deep=False)
    values = out["archetype_policy_key"].astype("string").copy()
    sides = out["side_name"].astype("string").str.lower()
    for side in ("long", "short"):
        mask = sides.eq(side) & values.str.startswith(f"{side}__", na=False)
        values.loc[mask] = values.loc[mask].str[len(side) + 2 :]
    out = out.copy()
    out["archetype_policy_key"] = values.astype(object)
    return out


@dataclass(frozen=True)
class SideResidualExpertBundle:
    """Apply the final side-local residual experts without refitting state."""

    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "SideResidualExpertBundle":
        payload = joblib.load(Path(path))
        if not isinstance(payload, Mapping):
            raise TypeError("Side residual expert bundle must contain a mapping")
        bundle = cls(payload)
        bundle.validate_contract()
        return bundle

    def validate_contract(self) -> None:
        schema = str(self.payload.get("schema") or "")
        if schema == SIDE_ROUTED_SCHEMA:
            routes = self.payload.get("routes")
            if not isinstance(routes, Mapping):
                raise ValueError("Side-routed residual expert has no routes")
            for side in ("long", "short"):
                route = routes.get(side)
                if not isinstance(route, Mapping):
                    raise ValueError(f"Side-routed residual expert has no {side} route")
                SideResidualExpertBundle(route).validate_contract()
            return
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(f"Unsupported side residual expert schema: {schema!r}")
        sides = {"long", "short"}
        for key in (
            "feature_contract",
            "residual_models",
            "model_params_by_side",
            "alpha_by_side",
        ):
            value = self.payload.get(key)
            if not isinstance(value, Mapping) or not sides.issubset(value):
                raise ValueError(f"Bundle has no complete side-local {key}")
        if self.payload.get("baseline_ev_map") is None:
            raise ValueError("Bundle has no baseline hierarchical EV map")
        if self.payload.get("corrected_ev_map") is None:
            raise ValueError("Bundle has no corrected hierarchical EV map")
        if float(self.payload.get("round_trip_cost", np.nan)) != 0.01:
            raise ValueError("Bundle round-trip cost contract must be exactly 1%")

    def _bundle_for_side(self, side: str) -> "SideResidualExpertBundle":
        side_name = str(side or "").strip().lower()
        if str(self.payload.get("schema") or "") != SIDE_ROUTED_SCHEMA:
            return self
        routes = self.payload.get("routes") or {}
        route = routes.get(side_name)
        if not isinstance(route, Mapping):
            raise ValueError(f"Unknown routed residual-expert side: {side_name!r}")
        return SideResidualExpertBundle(route)

    def feature_contract(self, side: str) -> list[str]:
        bundle = self._bundle_for_side(side)
        return [str(feature) for feature in bundle.payload["feature_contract"][side]]

    def required_input_features(self, side: str | None = None) -> list[str]:
        if str(self.payload.get("schema") or "") == SIDE_ROUTED_SCHEMA:
            requested_sides = [side] if side is not None else ["long", "short"]
            required: list[str] = []
            for name in requested_sides:
                required.extend(self._bundle_for_side(str(name)).required_input_features(str(name)))
            return list(dict.fromkeys(required))
        contracts = self.payload["feature_contract"]
        requested_sides = [side] if side is not None else ["long", "short"]
        required = ["side_name", "archetype_policy_key"]
        required.extend(
            str(self.payload.get("backbone_score_col") or "score") for _ in [0]
        )
        for name in requested_sides:
            if name not in contracts:
                raise ValueError(f"Unknown side: {name!r}")
            required.extend(str(feature) for feature in contracts[name])
        return list(dict.fromkeys(required))

    def complete_case_mask(self, frame: pd.DataFrame) -> np.ndarray:
        if str(self.payload.get("schema") or "") == SIDE_ROUTED_SCHEMA:
            if "side_name" not in frame:
                return np.zeros(len(frame), dtype=bool)
            sides = frame["side_name"].astype(str).str.lower().to_numpy()
            complete = np.zeros(len(frame), dtype=bool)
            for side in ("long", "short"):
                positions = np.flatnonzero(sides == side)
                if not len(positions):
                    continue
                routed = self._bundle_for_side(side).complete_case_mask(
                    frame.iloc[positions]
                )
                complete[positions] = routed
            return complete
        if "side_name" not in frame or "archetype_policy_key" not in frame:
            return np.zeros(len(frame), dtype=bool)
        sides = frame["side_name"].astype(str).str.lower().to_numpy()
        complete = np.isin(sides, ["long", "short"])
        contracts = self.payload["feature_contract"]
        for side in ("long", "short"):
            pos = sides == side
            if not pos.any():
                continue
            features = list(dict.fromkeys(contracts[side]))
            if any(feature not in frame for feature in features):
                complete[pos] = False
                continue
            values = frame.loc[pos, features].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float32, copy=False)
            complete[pos] = np.isfinite(values).all(axis=1)
        return complete

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score complete rows; incomplete rows remain explicitly unscored."""
        if str(self.payload.get("schema") or "") == SIDE_ROUTED_SCHEMA:
            output = pd.DataFrame(index=frame.index)
            output["meta_residual_expert_complete_case"] = False
            score_columns = (
                "score_base_ev_mapped",
                "score_base_ev_residual_expert",
                "score_base_ev_residual_expert_hier_mapped",
                "meta_residual_expert_delta_ev",
                "score_base_residual_ev_rank_train_reference",
            )
            for column in score_columns:
                output[column] = np.nan
            if "side_name" not in frame:
                return output
            sides = frame["side_name"].astype(str).str.lower().to_numpy()
            for side in ("long", "short"):
                positions = np.flatnonzero(sides == side)
                if not len(positions):
                    continue
                routed = self._bundle_for_side(side).transform(frame.iloc[positions])
                output.iloc[positions, :] = routed.reindex(columns=output.columns).to_numpy()
            output["meta_residual_expert_complete_case"] = output[
                "meta_residual_expert_complete_case"
            ].astype(bool)
            return output
        normalized = _normalize_archetype_keys(frame)
        required_sides: list[str] = []
        if "side_name" in normalized:
            observed_sides = normalized["side_name"].astype(str).str.lower()
            required_sides = [
                side for side in ("long", "short") if observed_sides.eq(side).any()
            ]
        required_features = ["side_name", "archetype_policy_key"]
        required_features.append(str(self.payload.get("backbone_score_col") or "score"))
        for side in required_sides:
            required_features.extend(self.payload["feature_contract"][side])
        missing_columns = sorted(
            set(required_features) - set(normalized.columns)
        )
        if missing_columns:
            raise ValueError(
                "Side residual expert input is missing required columns: "
                + ", ".join(missing_columns[:20])
            )
        complete = self.complete_case_mask(normalized)
        output = pd.DataFrame(index=frame.index)
        output["meta_residual_expert_complete_case"] = complete
        score_columns = (
            "score_base_ev_mapped",
            "score_base_ev_residual_expert",
            "score_base_ev_residual_expert_hier_mapped",
            "meta_residual_expert_delta_ev",
            "score_base_residual_ev_rank_train_reference",
        )
        for column in score_columns:
            output[column] = np.nan
        if not complete.any():
            return output

        work = normalized.loc[complete]
        raw = pd.to_numeric(
            work[str(self.payload.get("backbone_score_col") or "score")],
            errors="coerce",
        ).to_numpy(dtype=np.float32)
        baseline = predict_hierarchical_ev(
            self.payload["baseline_ev_map"], work, raw
        )
        residual = np.zeros(len(work), dtype=np.float32)
        sides = work["side_name"].astype(str).str.lower().to_numpy()
        for side, model in self.payload["residual_models"].items():
            pos = sides == side
            if not pos.any():
                continue
            features = self.payload["feature_contract"][side]
            matrix = work.loc[pos, features].to_numpy(dtype=np.float32, copy=False)
            residual[pos] = np.asarray(model.predict(matrix), dtype=np.float32)
        alpha = (
            work["side_name"]
            .astype(str)
            .str.lower()
            .map(self.payload["alpha_by_side"])
            .to_numpy(dtype=np.float32)
        )
        delta = alpha * residual
        corrected = baseline + delta
        mapped = predict_hierarchical_ev(
            self.payload["corrected_ev_map"], work, corrected
        )
        rank = expected_ev_rank(
            self.payload["corrected_ev_map"], mapped, corrected
        )
        # Assign positionally: batched replay frames can legitimately repeat their
        # source index across hourly batches, while label-based ``.loc`` expands
        # every duplicate label and changes the assignment length.
        target_pos = np.flatnonzero(complete)
        for column, values in (
            ("score_base_ev_mapped", baseline),
            ("score_base_ev_residual_expert", corrected),
            ("score_base_ev_residual_expert_hier_mapped", mapped),
            ("meta_residual_expert_delta_ev", delta),
            ("score_base_residual_ev_rank_train_reference", rank),
        ):
            output.iloc[target_pos, output.columns.get_loc(column)] = values
        return output
