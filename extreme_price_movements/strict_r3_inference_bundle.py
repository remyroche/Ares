"""Validation for a sealed strict-R3 shadow inference bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA_V1 = "strict_r3_inference_bundle_v1"
SCHEMA_V2 = "strict_r3_inference_bundle_v2"
SCHEMA_V3 = "strict_r3_inference_bundle_v3_28d_r5"
SCHEMA_V4 = "strict_r3_inference_bundle_v4_28d_r5_9m_posterior"
SCHEMA_V5 = "strict_r3_inference_bundle_v5_28d_a5_bounded10"
SCHEMA = SCHEMA_V5


def validate_live_feature_frame(
    frame: pd.DataFrame,
    *,
    fields: list[str],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a live feature panel violates the sealed parity gate."""
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise ValueError(f"live feature frame lacks frozen fields: {missing}")
    numeric = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)),
        index=numeric.index, columns=numeric.columns,
    )
    per_field = finite.mean(axis=0)
    per_row = finite.mean(axis=1)
    all_fields = finite.all(axis=1)
    minimum_row = float(requirements["minimum_row_feature_fraction"])
    minimum_cycle = float(requirements["minimum_cycle_complete_fraction"])
    minimum_field = float(requirements["minimum_per_field_finite_fraction"])
    checks = {
        "all_frozen_fields_present": not missing,
        "row_coverage_fraction_meets_cycle_gate": bool(
            per_row.ge(minimum_row).mean() >= minimum_cycle
        ),
        "complete_row_fraction_meets_cycle_gate": bool(
            all_fields.mean() >= minimum_cycle
        ),
        "every_field_meets_finite_gate": bool(per_field.ge(minimum_field).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"live feature parity gate failed: {checks}")
    return {
        "fields": int(len(fields)),
        "rows": int(len(frame)),
        "minimum_row_finite_fraction": float(per_row.min()),
        "rows_meeting_minimum_fraction": float(per_row.ge(minimum_row).mean()),
        "all_fields_complete_fraction": float(all_fields.mean()),
        "minimum_per_field_finite_fraction": float(per_field.min()),
        "checks": checks,
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StrictR3InferenceBundle:
    root: Path
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path, *, root: str | Path) -> "StrictR3InferenceBundle":
        source = Path(path)
        payload = json.loads(source.read_text())
        schema = str(payload.get("schema") or "")
        if schema not in {SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5}:
            raise ValueError(
                "inference bundle has an unsupported strict-R3 schema",
            )
        if str(payload.get("scope") or "") != "long_only_shadow":
            raise ValueError("canonical bundle must be long-only and shadow-only")
        if str(payload.get("outside_window") or "") != "fail_closed":
            raise ValueError("canonical inference bundle must fail closed outside its fit window")
        runtime = payload.get("runtime") or {}
        if runtime.get("mode") != "shadow-only" or runtime.get("exchange_io") is not False or runtime.get("order_submission") is not False:
            raise ValueError("sealed inference bundle may not expose exchange or order authority")
        if schema in {SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5}:
            expected_runtime = {
                "candidate_feature_population": (
                    "complete_frozen_universe_before_current_spread_filter"
                ),
                "current_spread_gate": (
                    "official_kraken_signal_hour_bid_ask_bps_le_100"
                ),
            }
            for key, expected in expected_runtime.items():
                if runtime.get(key) != expected:
                    raise ValueError(
                        f"schema-v2 inference bundle requires {key}={expected}",
                    )
            if not runtime.get("feature_history_start"):
                raise ValueError(
                    "schema-v2 inference bundle requires feature_history_start",
                )
        if schema == SCHEMA_V3:
            if int(payload.get("reference_window_days", -1)) != 28:
                raise ValueError("schema-v3 inference requires a 28-day reference reserve")
            if payload.get("admission_contract") != (
                "strict_oof_exact_producer_cell_day_trim15_28d_v1"
            ):
                raise ValueError("schema-v3 inference has the wrong Cell-day admission contract")
            if payload.get("trust_overlay_contract") != (
                "strict_r3_cell_day_residual_trust_overlay_v1"
            ):
                raise ValueError("schema-v3 inference requires the canonical R5 overlay")
        if schema == SCHEMA_V4:
            if int(payload.get("reference_window_days", -1)) != 28:
                raise ValueError("schema-v4 inference requires a 28-day reference reserve")
            if payload.get("admission_contract") != (
                "strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1"
            ):
                raise ValueError("schema-v4 inference has the wrong posterior admission contract")
            if payload.get("trust_overlay_contract") != (
                "strict_r3_cell_day_residual_trust_model_r5_9m_v1"
            ):
                raise ValueError("schema-v4 inference requires the canonical R5 model")
        if schema == SCHEMA_V5:
            if int(payload.get("reference_window_days", -1)) != 28:
                raise ValueError("schema-v5 inference requires a 28-day reference reserve")
            if payload.get("admission_contract") != "strict_r3_a5_bounded_10pct_canonical_v1":
                raise ValueError("schema-v5 inference has the wrong bounded-A5 contract")
            if payload.get("trust_overlay_contract") != "A5_bounded10_over_A0_top15":
                raise ValueError("schema-v5 inference requires bounded A5 over A0 top-15")
        return cls(root=Path(root).resolve(), payload=payload)

    def path(self, name: str) -> Path:
        raw = (self.payload.get("paths") or {}).get(name)
        if not raw:
            raise ValueError(f"inference bundle has no path named {name}")
        candidate = (self.root / str(raw)).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError(f"bundle path escapes repository root: {name}")
        return candidate

    def validate(self, *, decision_ts: str | pd.Timestamp) -> dict[str, Any]:
        decision = pd.Timestamp(decision_ts)
        decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        activation = pd.Timestamp(self.payload["activation_ts"])
        end = pd.Timestamp(self.payload["end_exclusive_ts"])
        if not activation <= decision < end:
            raise ValueError(
                f"decision {decision.isoformat()} is outside sealed producer window "
                f"[{activation.isoformat()}, {end.isoformat()})",
            )
        expected = self.payload.get("sha256") or {}
        hash_paths = {
            "conversion_bundle": self.path("conversion_bundle_dir") / "four_week_conversion_bundle.joblib",
            "conversion_manifest": self.path("conversion_bundle_dir") / "run_manifest.json",
            "upstream_bundle": self.path("upstream_bundle_dir") / "monthly_upstream_bundle.joblib",
            "upstream_manifest": self.path("upstream_bundle_dir") / "run_manifest.json",
            "frozen_geometry_bundle": self.path("frozen_geometry_bundle"),
            "frozen_geometry_manifest": self.path("frozen_geometry_bundle").with_name("run_manifest.json"),
            "feature_contract": self.path("feature_contract"),
            "frozen_universe_manifest": self.path("frozen_universe_manifest"),
            "same_model_reference_candidates": self.path("same_model_reference_candidates"),
            "same_model_reference_features": self.path("same_model_reference_features"),
            "resolved_score_label_ledger": self.path("resolved_score_label_ledger"),
            "immediate_calibration_index": self.path("immediate_calibration_index"),
            "ev_bridge_bundle": self.path("ev_bridge_bundle"),
            "exit_policy": self.path("exit_policy"),
            "portfolio_policy": self.path("portfolio_policy"),
        }
        if str(self.payload["schema"]) in {SCHEMA_V3, SCHEMA_V4, SCHEMA_V5}:
            hash_paths.update({
                "cell_day_trust_bundle": (
                    self.path("cell_day_trust_bundle_dir")
                    / "cell_day_residual_trust.joblib"
                ),
                "cell_day_trust_manifest": (
                    self.path("cell_day_trust_bundle_dir") / "run_manifest.json"
                ),
                "cell_day_trust_contract": self.path("cell_day_trust_contract"),
            })
            if str(self.payload["schema"]) in {SCHEMA_V4, SCHEMA_V5}:
                hash_paths["cell_day_trust_integration_contract"] = self.path(
                    "cell_day_trust_integration_contract"
                )
            if str(self.payload["schema"]) == SCHEMA_V5:
                hash_paths.update({
                    "a5_model": self.path("a5_bundle_dir") / "a4_independent_residual.joblib",
                    "a5_calibration": self.path("a5_bundle_dir") / "a5_causal_calibration.joblib",
                    "a5_manifest": self.path("a5_bundle_dir") / "run_manifest.json",
                    "a5_contract": self.path("a5_contract"),
                })
        observed: dict[str, str] = {}
        for name, source in hash_paths.items():
            if not source.is_file():
                raise FileNotFoundError(f"sealed inference input is missing: {source}")
            observed[name] = _sha(source)
            if observed[name] != str(expected.get(name) or ""):
                raise ValueError(f"sealed inference input hash mismatch: {name}")
        conversion_manifest = json.loads(hash_paths["conversion_manifest"].read_text())
        upstream_manifest = json.loads(hash_paths["upstream_manifest"].read_text())
        if pd.Timestamp(conversion_manifest["cutoff"]) != activation or pd.Timestamp(upstream_manifest["cutoff"]) != activation:
            raise ValueError("bundle activation does not match model cutoffs")
        if pd.Timestamp(conversion_manifest["end_exclusive"]) != end or pd.Timestamp(upstream_manifest["end_exclusive"]) != end:
            raise ValueError("bundle expiry does not match model fit windows")
        if conversion_manifest.get("geometry_refit_cadence") != "never":
            raise ValueError("geometry/K9 must remain frozen across inference bundles")
        producer = self.payload.get("producer") or {}
        if conversion_manifest.get("bundle_sha256") != producer.get("conversion_bundle_sha256"):
            raise ValueError("conversion producer hash mismatch")
        if upstream_manifest.get("bundle_sha256") != producer.get("upstream_bundle_sha256"):
            raise ValueError("upstream producer hash mismatch")
        if conversion_manifest.get("geometry_bundle_sha256") != producer.get("geometry_bundle_sha256"):
            raise ValueError("geometry semantic identity mismatch")
        if str(self.payload["schema"]) in {SCHEMA_V3, SCHEMA_V4, SCHEMA_V5}:
            trust_manifest = json.loads(hash_paths["cell_day_trust_manifest"].read_text())
            if pd.Timestamp(trust_manifest["cutoff"]) != activation:
                raise ValueError("R5 trust bundle cutoff does not match producer activation")
            if str(self.payload["schema"]) == SCHEMA_V3:
                if trust_manifest.get("admission_changes") is not False:
                    raise ValueError("schema-v3 R5 trust bundle may not change admission")
            else:
                if trust_manifest.get("admission_changes") is not True:
                    raise ValueError("schema-v4/v5 A0 bundle must expose posterior admission")
                if int(trust_manifest.get("training_window_months", -1)) != 9:
                    raise ValueError("schema-v4 R5 trust bundle must use nine months")
                if trust_manifest.get("missing_posterior") != "fail_closed":
                    raise ValueError("schema-v4/v5 R5 posterior must fail closed")
            if str(self.payload["schema"]) == SCHEMA_V5:
                a5_manifest = json.loads(hash_paths["a5_manifest"].read_text())
                if pd.Timestamp(a5_manifest["cutoff"]) != activation:
                    raise ValueError("A5 bundle cutoff does not match producer activation")
                calibration = a5_manifest.get("calibration") or {}
                if pd.Timestamp(calibration.get("cutoff")) != activation:
                    raise ValueError("A5 calibration cutoff does not match producer activation")
                if int(calibration.get("prior_oos_rows", -1)) < 2_000:
                    raise ValueError("A5 calibration lacks prior OOS support")
                if not np.isclose(float(a5_manifest.get("bounded_alpha", np.nan)), 0.10):
                    raise ValueError("A5 inference requires alpha 0.10")
                if a5_manifest.get("domain") != "timestamp_local_top15_by_final_score":
                    raise ValueError("A5 inference requires the frozen top-15 domain")
        return {
            "schema": str(self.payload["schema"]),
            "decision_ts": decision.isoformat(),
            "activation_ts": activation.isoformat(),
            "end_exclusive_ts": end.isoformat(),
            "hashes_verified": len(observed),
            "conversion_bundle_sha256": producer.get("conversion_bundle_sha256"),
            "upstream_bundle_sha256": producer.get("upstream_bundle_sha256"),
            "geometry_bundle_sha256": producer.get("geometry_bundle_sha256"),
            "mode": "shadow-only",
        }


__all__ = [
    "SCHEMA", "SCHEMA_V1", "SCHEMA_V2", "SCHEMA_V3", "SCHEMA_V4", "SCHEMA_V5",
    "StrictR3InferenceBundle",
    "validate_live_feature_frame",
]
