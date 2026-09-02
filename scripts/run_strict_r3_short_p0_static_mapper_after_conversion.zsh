#!/bin/zsh
# Fail-closed handoff: conversion -> frozen short current/BCF MC1 OOS scoring.
#
# This is deliberately not a live or policy producer.  It advances only the
# mapper-validation stage after the immutable short conversion ledger exists.

set -euo pipefail

cd "$(dirname "$0")/.."

CONVERSION_DIR="data_perp/artifacts/strict_r3_short_p0_same_model_conversion_oof_2025apr_2026jul_20260821_v1"
CONVERSION_MANIFEST="$CONVERSION_DIR/run_manifest.json"
CURRENT_BUNDLE="data_perp/artifacts/strict_r3_short_p0_current_mc1_bundle_20250701_20260821_v1"
BCF_LEDGER_DIR="data_perp/artifacts/strict_r3_short_p0_bcf_native_ledger_20250701_20260821_v1"
BCF_LEDGER="$BCF_LEDGER_DIR/short_bcf_native_policy_ledger.parquet"
BCF_BUNDLE="data_perp/artifacts/strict_r3_short_p0_bcf_mc1_bundle_20250701_20260821_v1"
MAPPER_OUT="data_perp/artifacts/strict_r3_short_p0_static_mc1_oof_2025jul_2026jul_20260821_v1"

# The existing conversion chain is the sole producer of this input.  If it
# exits without an immutable manifest, stop rather than inventing a fallback.
while [[ ! -f "$CONVERSION_MANIFEST" ]]; do
  if ! kill -0 38448 2>/dev/null; then
    print -u2 "CONVERSION_EXITED_WITHOUT_MANIFEST"
    exit 1
  fi
  sleep 30
done

python3 - "$CONVERSION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload.get("status") == "complete", payload.get("status")
assert payload.get("side") == "short", payload.get("side")
assert payload.get("selection_eligible_only_for_promotion") is True
PY

for path in "$CURRENT_BUNDLE" "$BCF_LEDGER_DIR" "$BCF_BUNDLE" "$MAPPER_OUT"; do
  if [[ -e "$path" ]]; then
    print -u2 "IMMUTABLE_OUTPUT_ALREADY_EXISTS=$path"
    exit 1
  fi
done

python3 scripts/build_strict_r3_mc1_d2_canonical_bundle.py \
  --ledger "$CONVERSION_DIR/short_same_model_conversion_oof_predictions.parquet" \
  --champion-config config/strict_r3_short_mc1_d2_research_champion_20260821_v1.json \
  --fit-cutoff 2025-07-01T00:00:00Z \
  --side short \
  --out-dir "$CURRENT_BUNDLE"

# BCF requires the frozen, promoted all-head agreement contract.  A single
# promoted head has no agreement geometry, so it remains explicitly
# unavailable; current-MC1 OOS scoring is still a valid control in that case.
if python3 - "$CONVERSION_DIR/short_bcf_promoted_head_contract.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload.get("side") == "short"
raise SystemExit(0 if payload.get("enabled") is True else 1)
PY
then
  python3 scripts/materialize_strict_r3_bcf_mc1_ledger.py \
    --bcf-scores "$CONVERSION_DIR/short_bcf_score_family_oof_predictions.parquet" \
    --policy-labels "$CONVERSION_DIR/short_policy_outcomes_source.parquet" \
    --side short \
    --rank-fields-json "$CONVERSION_DIR/short_bcf_promoted_head_contract.json" \
    --out-path "$BCF_LEDGER"
  python3 scripts/build_strict_r3_bcf_mc1_d2_bundle.py \
    --ledger "$BCF_LEDGER" \
    --champion-config config/strict_r3_short_bcf_mc1_d2_research_champion_20260821_v1.json \
    --fit-cutoff 2025-07-01T00:00:00Z \
    --side short \
    --out-dir "$BCF_BUNDLE"
  python3 scripts/run_strict_r3_short_p0_static_mc1_oof.py \
    --current-scores "$CONVERSION_DIR/short_same_model_conversion_oof_predictions.parquet" \
    --current-bundle "$CURRENT_BUNDLE" \
    --bcf-scores "$CONVERSION_DIR/short_bcf_score_family_oof_predictions.parquet" \
    --bcf-bundle "$BCF_BUNDLE" \
    --start 2025-07-01T00:00:00Z \
    --end-exclusive 2026-08-01T00:00:00Z \
    --out "$MAPPER_OUT"
else
  python3 scripts/run_strict_r3_short_p0_static_mc1_oof.py \
    --current-scores "$CONVERSION_DIR/short_same_model_conversion_oof_predictions.parquet" \
    --current-bundle "$CURRENT_BUNDLE" \
    --start 2025-07-01T00:00:00Z \
    --end-exclusive 2026-08-01T00:00:00Z \
    --out "$MAPPER_OUT"
fi

print "$MAPPER_OUT"
